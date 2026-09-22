//! HF shard index (`model.safetensors.index.json`): tensor name to shard file.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::de::{MapAccess, Visitor};

use crate::safetensors::{SafetensorsError, SafetensorsFile};

/// The filesystem's identity for a file: its device and inode.
///
/// Two names for one file — a hard link, a symlink, a different spelling —
/// share an identity, and two files that merely share a name do not. No path
/// string can answer that question, which is why identity is taken from the
/// filesystem itself.
///
/// This module is unix-only, matching the rest of the workspace: the engine
/// builds for macOS and Linux, and identity by canonical path (the only
/// alternative Rust offers portably) cannot represent a hard link on any
/// platform.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId {
    dev: u64,
    ino: u64,
}

/// Identity of an existing path. Errors if it does not exist.
pub fn file_identity(path: &Path) -> std::io::Result<FileId> {
    use std::os::unix::fs::MetadataExt;
    let md = std::fs::metadata(path)?;
    Ok(FileId {
        dev: md.dev(),
        ino: md.ino(),
    })
}

/// Which file holds each tensor of a sharded checkpoint.
#[derive(Debug)]
pub struct ShardIndex {
    dir: PathBuf,
    /// Tensor name to shard file name.
    map: HashMap<String, String>,
}

impl ShardIndex {
    /// Load the one `*.index.json` in a checkpoint directory.
    ///
    /// Components name it differently — `model.safetensors.index.json` for the
    /// text encoder, `diffusion_pytorch_model.safetensors.index.json` for the
    /// transformer — so the file is discovered rather than assumed.
    pub fn open_dir(dir: &Path) -> Result<Self, SafetensorsError> {
        let mut found = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let p = entry?.path();
            if let Some(n) = p.file_name().and_then(|n| n.to_str()) {
                if n.ends_with(".index.json") {
                    found.push(n.to_string());
                }
            }
        }
        found.sort();
        let index_name = match found.len() {
            1 => found.remove(0),
            n => {
                return Err(SafetensorsError::MissingField {
                    tensor: dir.display().to_string(),
                    field: if n == 0 {
                        "shard index"
                    } else {
                        "a single shard index"
                    },
                })
            }
        };
        Self::open(dir, &index_name)
    }

    /// Load a named index from a checkpoint directory.
    fn open(dir: &Path, index_name: &str) -> Result<Self, SafetensorsError> {
        let raw = std::fs::read(dir.join(index_name))?;
        let doc: IndexDoc = serde_json::from_slice(&raw)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            map: doc.weight_map.0,
        })
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn shard_of(&self, tensor: &str) -> Option<&str> {
        self.map.get(tensor).map(|s| s.as_str())
    }

    pub fn path_of(&self, shard: &str) -> PathBuf {
        self.dir.join(shard)
    }

    /// Every tensor name, sorted, so conversion order is deterministic.
    pub fn names_sorted(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.map.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }

    /// Distinct shard file names, sorted.
    /// Every `.safetensors` at or below `dir`, identified by canonical path.
    ///
    /// The walk follows directory symlinks and descends into subdirectories,
    /// because a shard reachable only through a symlinked or nested directory is
    /// just as reachable to the loader as one sitting at the top. Canonical
    /// paths are what make that sound: a visited set keyed on them terminates on
    /// a symlink cycle, and two spellings of one file collapse to one entry.
    fn discover_shards(dir: &Path) -> Result<Vec<(FileId, PathBuf)>, CrossCheckError> {
        let mut found: Vec<(FileId, PathBuf)> = Vec::new();
        let mut seen: HashSet<FileId> = HashSet::new();
        let mut visited: HashSet<PathBuf> = HashSet::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            let canonical_dir = current.canonicalize().map_err(|e| CrossCheckError::Scan {
                dir: current.display().to_string(),
                detail: e.to_string(),
            })?;
            if !visited.insert(canonical_dir) {
                continue;
            }
            let rd = std::fs::read_dir(&current).map_err(|e| CrossCheckError::Scan {
                dir: current.display().to_string(),
                detail: e.to_string(),
            })?;
            for entry in rd {
                let entry = entry.map_err(|e| CrossCheckError::Scan {
                    dir: current.display().to_string(),
                    detail: e.to_string(),
                })?;
                let path = entry.path();
                // `metadata` follows symlinks, so a link to a directory is
                // walked and a link to a shard is counted.
                match std::fs::metadata(&path) {
                    Ok(md) if md.is_dir() => stack.push(path),
                    Ok(_) => {
                        if path
                            .extension()
                            .map(|e| e == "safetensors")
                            .unwrap_or(false)
                        {
                            let id = file_identity(&path).map_err(|e| CrossCheckError::Scan {
                                dir: current.display().to_string(),
                                detail: e.to_string(),
                            })?;
                            // One file reachable twice — a hard link, or a link
                            // through a second directory — is one shard.
                            if seen.insert(id.clone()) {
                                found.push((id, path));
                            }
                        }
                    }
                    // A broken entry that looks like a shard is a broken shard; a
                    // broken entry that does not is none of this check's business.
                    Err(e) => {
                        if path
                            .extension()
                            .map(|x| x == "safetensors")
                            .unwrap_or(false)
                        {
                            return Err(CrossCheckError::Scan {
                                dir: current.display().to_string(),
                                detail: format!("{}: {e}", path.display()),
                            });
                        }
                    }
                }
            }
        }
        found.sort_by(|a, b| a.1.cmp(&b.1));
        Ok(found)
    }

    /// The single shard of a component stored without a shard index.
    ///
    /// Rejects a directory holding more than one shard, including one reachable
    /// only through a subdirectory or symlink: picking one and ignoring the rest
    /// would drop their tensors silently.
    pub fn single_shard(dir: &Path) -> Result<PathBuf, CrossCheckError> {
        let mut found = Self::discover_shards(dir)?;
        match found.len() {
            1 => Ok(found.remove(0).1),
            n => Err(CrossCheckError::Scan {
                dir: dir.display().to_string(),
                detail: format!("expected one shard, found {n}"),
            }),
        }
    }

    /// Check the index against what the shards actually contain.
    ///
    /// The index alone cannot prove completeness: a tensor present in a shard
    /// but absent from `weight_map` would never be converted, and counting
    /// iterations over the index would agree with itself. So the shards on disk
    /// are enumerated and compared with the index in both directions.
    ///
    /// Every comparison is on file identity, never on name strings, so a
    /// spelling that merely looks different (`./a.safetensors`) is the same file
    /// and a name that merely looks the same under a different directory is not.
    ///
    /// The checkpoint is required to be unchanged for the duration of the call.
    /// Identity is read from the filesystem, so a shard replaced between two
    /// reads could hand a different file the identity just recorded. Converting
    /// a checkpoint another process is rewriting is not a supported operation.
    pub fn cross_check(&self) -> Result<CrossCheck, CrossCheckError> {
        // What the index declares, grouped by the file that physically holds it.
        // Identity is the file itself (device and inode on unix), not a path
        // string: two names for one file — a spelling, a symlink or a hard link
        // — must group together, and one name reached under two directories must
        // not.
        let mut declared: std::collections::BTreeMap<FileId, (PathBuf, Vec<String>)> =
            std::collections::BTreeMap::new();
        for tensor in self.map.keys() {
            let shard = self.map.get(tensor).expect("key came from this map");
            let path = self.path_of(shard);
            let id = file_identity(&path).map_err(|e| CrossCheckError::Read {
                shard: shard.clone(),
                detail: e.to_string(),
            })?;
            declared
                .entry(id)
                .or_insert_with(|| (path, Vec::new()))
                .1
                .push(tensor.clone());
        }

        // Each declared file must contain exactly the tensors declared for it.
        // Comparing the two sets per file is what detects a tensor placed in the
        // wrong file, not merely recorded twice.
        let mut actual: HashSet<String> = HashSet::new();
        let mut per_shard: HashMap<String, Vec<String>> = HashMap::new();
        let mut misplaced: Vec<String> = Vec::new();
        let mut duplicated: Vec<String> = Vec::new();
        for (path, declared_here) in declared.values() {
            let st = SafetensorsFile::open(path).map_err(|e| CrossCheckError::Read {
                shard: path.display().to_string(),
                detail: e.to_string(),
            })?;
            let mut found: Vec<String> = st.tensors().iter().map(|(n, _)| n.clone()).collect();
            found.sort();
            for name in &found {
                if !actual.insert(name.clone()) {
                    duplicated.push(name.clone());
                }
            }
            let declared_set: HashSet<&String> = declared_here.iter().collect();
            let found_set: HashSet<&String> = found.iter().collect();
            for name in declared_set.difference(&found_set) {
                misplaced.push(format!("{name}: declared in {} but absent", path.display()));
            }
            for name in found_set.difference(&declared_set) {
                misplaced.push(format!(
                    "{name}: present in {} but not declared there",
                    path.display()
                ));
            }
            per_shard.insert(path.display().to_string(), found);
        }

        let indexed: HashSet<String> = self.map.keys().cloned().collect();
        let mut missing: Vec<String> = indexed.difference(&actual).cloned().collect();
        let mut unexpected: Vec<String> = actual.difference(&indexed).cloned().collect();
        missing.sort();
        unexpected.sort();
        misplaced.sort();
        duplicated.sort();
        duplicated.dedup();

        // Every shard on disk the index does not reference, compared by identity
        // so a hard link to a referenced shard is not reported as a second file.
        let referenced: HashSet<FileId> = declared.keys().cloned().collect();
        let mut unreferenced: Vec<String> = Self::discover_shards(&self.dir)?
            .into_iter()
            .filter(|(id, _)| !referenced.contains(id))
            .map(|(_, p)| p.display().to_string())
            .collect();
        unreferenced.sort();

        Ok(CrossCheck {
            indexed: indexed.len(),
            actual: actual.len(),
            missing,
            unexpected,
            duplicated,
            misplaced,
            unreferenced,
            per_shard,
        })
    }
}

/// Result of comparing an index with the shards on disk.
#[derive(Debug)]
pub struct CrossCheck {
    pub indexed: usize,
    pub actual: usize,
    /// In the index but not in any shard.
    pub missing: Vec<String>,
    /// In a shard but not in the index — these would be silently dropped.
    pub unexpected: Vec<String>,
    /// Declared in more than one shard, or listed twice.
    pub duplicated: Vec<String>,
    /// Found in a different file from the one the index assigns it to.
    pub misplaced: Vec<String>,
    /// `.safetensors` files present but not named by the index.
    pub unreferenced: Vec<String>,
    /// Tensor names per shard, sorted.
    pub per_shard: HashMap<String, Vec<String>>,
}

impl CrossCheck {
    pub fn is_clean(&self) -> bool {
        self.missing.is_empty()
            && self.unexpected.is_empty()
            && self.duplicated.is_empty()
            && self.misplaced.is_empty()
            && self.unreferenced.is_empty()
            && self.indexed == self.actual
    }
}

#[derive(Debug)]
pub enum CrossCheckError {
    Read { shard: String, detail: String },
    Scan { dir: String, detail: String },
}

impl std::fmt::Display for CrossCheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read { shard, detail } => write!(f, "reading shard {shard}: {detail}"),
            Self::Scan { dir, detail } => write!(f, "listing {dir}: {detail}"),
        }
    }
}

impl std::error::Error for CrossCheckError {}

/// A `{tensor: shard}` map that refuses a repeated key.
///
/// `serde_json::Value` keeps only the last value for a repeated key, so a
/// `weight_map` naming a tensor twice would parse and hide the duplication.
/// Deserializing into this type visits every entry, so the repeat is seen.
struct StrictMap(HashMap<String, String>);

impl<'de> serde::Deserialize<'de> for StrictMap {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = StrictMap;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "an object mapping tensor names to shard names")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut acc: A) -> Result<StrictMap, A::Error> {
                let mut out = HashMap::with_capacity(acc.size_hint().unwrap_or(0));
                while let Some((k, v)) = acc.next_entry::<String, String>()? {
                    if out.insert(k.clone(), v).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "tensor {k} appears more than once in weight_map"
                        )));
                    }
                }
                Ok(StrictMap(out))
            }
        }
        d.deserialize_map(V)
    }
}

/// The index document, with `weight_map` parsed strictly.
#[derive(serde::Deserialize)]
struct IndexDoc {
    weight_map: StrictMap,
}
