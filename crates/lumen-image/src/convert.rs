//! Checkpoint conversion: safetensors to `.lbi`.
//!
//! One component (transformer, vae, text_encoder) becomes one `.lbi`. Tensors
//! keep the dtype the checkpoint stores them in — BF16 stays BF16, F32 stays
//! F32 — so a later comparison against a reference cannot be blamed on the
//! container having changed the weights.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use lumen_format::QuantScheme;

use crate::lbi::{staging_paths, LbiError, LbiFile, LbiWriter};
use crate::safetensors::{SafetensorsError, SafetensorsFile, DTYPE_BF16, DTYPE_F16, DTYPE_F32};
use crate::shard_index::ShardIndex;

#[derive(Debug)]
pub enum ConvertError {
    Safetensors(SafetensorsError),
    Lbi(LbiError),
    Io(std::io::Error),
    /// The checkpoint file declares a dtype this converter does not store.
    UnsupportedDtype {
        tensor: String,
        dtype: String,
    },
    /// A tensor listed in the index is absent from the shard that should hold it.
    MissingTensor {
        tensor: String,
        shard: String,
    },
    /// The shard index and the shards disagree about a tensor's presence.
    IndexMismatch {
        name: String,
        indexed: usize,
        found: usize,
    },
    /// The shards hold tensors the index never mentions, or vice versa. These
    /// would otherwise be dropped silently. Boxed because it carries the full
    /// discrepancy detail and would otherwise dominate the enum's size.
    ShardIndexMismatch(Box<ShardIndexMismatch>),
    /// A written `.lbi`, opened again, does not hold the tensors written.
    ReadBack {
        written: usize,
        read: usize,
    },
    /// Converting one component of a checkpoint failed.
    Component {
        component: String,
        error: Box<ConvertError>,
    },
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Safetensors(e) => write!(f, "{e}"),
            Self::Lbi(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "io: {e}"),
            Self::UnsupportedDtype { tensor, dtype } => {
                write!(f, "tensor {tensor} has dtype {dtype}, which is not stored")
            }
            Self::MissingTensor { tensor, shard } => {
                write!(f, "tensor {tensor} is not in shard {shard}")
            }
            Self::IndexMismatch {
                name,
                indexed,
                found,
            } => write!(
                f,
                "{name}: index lists {indexed} tensors but the shards hold {found}"
            ),
            Self::ShardIndexMismatch(m) => write!(
                f,
                "{}: the index and the shards disagree — \
                 indexed-but-absent={:?} present-but-unindexed={:?} \
                 duplicated={:?} misplaced={:?} unindexed shard files={:?}",
                m.component, m.missing, m.unexpected, m.duplicated, m.misplaced, m.unreferenced
            ),
            Self::ReadBack { written, read } => {
                write!(f, "wrote {written} tensors but read back {read}")
            }
            Self::Component { component, error } => write!(f, "convert {component}: {error}"),
        }
    }
}

impl std::error::Error for ConvertError {}

impl From<SafetensorsError> for ConvertError {
    fn from(e: SafetensorsError) -> Self {
        Self::Safetensors(e)
    }
}

impl From<LbiError> for ConvertError {
    fn from(e: LbiError) -> Self {
        Self::Lbi(e)
    }
}

impl From<std::io::Error> for ConvertError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Detail carried by [`ConvertError::ShardIndexMismatch`].
#[derive(Debug)]
pub struct ShardIndexMismatch {
    pub component: String,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    pub duplicated: Vec<String>,
    pub misplaced: Vec<String>,
    pub unreferenced: Vec<String>,
}

/// Storage scheme for a safetensors dtype.
fn scheme_for(dtype: &str, tensor: &str) -> Result<QuantScheme, ConvertError> {
    match dtype {
        DTYPE_F32 => Ok(QuantScheme::F32),
        DTYPE_F16 => Ok(QuantScheme::F16),
        DTYPE_BF16 => Ok(QuantScheme::Bf16),
        other => Err(ConvertError::UnsupportedDtype {
            tensor: tensor.to_string(),
            dtype: other.to_string(),
        }),
    }
}

/// What one conversion produced.
#[derive(Debug, Clone)]
pub struct ConvertReport {
    pub component: String,
    pub tensor_count: usize,
    pub total_bytes: u64,
    /// Count of tensors per dtype, for a printed summary.
    pub dtypes: HashMap<String, usize>,
}

/// The three components a Qwen-Image-2.1 checkpoint directory holds, each
/// converted into `<out_dir>/<component>.lbi`.
pub const COMPONENTS: [&str; 3] = ["transformer", "vae", "text_encoder"];

/// Convert `<checkpoint_dir>/{transformer,vae,text_encoder}` into
/// `<out_dir>/{transformer,vae,text_encoder}.lbi`, each from its shard index
/// when it has one and from its single file otherwise, then open every file
/// written and check it holds the tensors written; a file that does not is
/// removed. `converted` is called with each component's report, in
/// [`COMPONENTS`] order, as soon as that component is done.
pub fn convert_checkpoint(
    checkpoint_dir: &Path,
    out_dir: &Path,
    mut converted: impl FnMut(&ConvertReport),
) -> Result<(), ConvertError> {
    std::fs::create_dir_all(out_dir).map_err(|e| {
        ConvertError::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", out_dir.display()),
        ))
    })?;
    remove_staging(out_dir)?;
    for component in COMPONENTS {
        let target = out_dir.join(format!("{component}.lbi"));
        let report = convert_one(checkpoint_dir, component, &target).map_err(|e| {
            ConvertError::Component {
                component: component.to_string(),
                error: Box::new(e),
            }
        })?;
        converted(&report);
    }
    Ok(())
}

/// Convert one component into `target` and check it reads back whole.
fn convert_one(
    checkpoint_dir: &Path,
    component: &str,
    target: &Path,
) -> Result<ConvertReport, ConvertError> {
    let comp_dir = checkpoint_dir.join(component);
    let config = component_config(&comp_dir)?;
    let report = if has_shard_index(&comp_dir) {
        convert_component(checkpoint_dir, component, target, config)?
    } else {
        convert_single_file(&single_shard(&comp_dir)?, component, target, config)?
    };
    let read = LbiFile::open(target).map(|f| f.len());
    if read.as_ref().ok() != Some(&report.tensor_count) {
        let _ = std::fs::remove_file(target);
        return Err(match read {
            Ok(read) => ConvertError::ReadBack {
                written: report.tensor_count,
                read,
            },
            Err(e) => e.into(),
        });
    }
    Ok(report)
}

/// Remove the staging files a conversion into `out_dir` that died left
/// behind, for every component.
pub fn remove_staging(out_dir: &Path) -> Result<(), ConvertError> {
    for component in COMPONENTS {
        let (part, tmp) = staging_paths(&out_dir.join(format!("{component}.lbi")));
        for stale in [part, tmp] {
            match std::fs::remove_file(&stale) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(ConvertError::Io(e)),
            }
        }
    }
    Ok(())
}

/// The component's `config.json`, which the `.lbi` carries.
fn component_config(dir: &Path) -> Result<serde_json::Value, ConvertError> {
    let path = dir.join("config.json");
    let at = |e: &dyn std::fmt::Display| {
        ConvertError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: {e}", path.display()),
        ))
    };
    let bytes = std::fs::read(&path).map_err(|e| at(&e))?;
    serde_json::from_slice(&bytes).map_err(|e| at(&e))
}

/// Build `<component>.lbi` from the shards named in `<component>/model.safetensors.index.json`.
///
/// Shards are opened one at a time and tensors streamed through, so peak
/// memory is one shard header plus one tensor buffer.
pub fn convert_component(
    checkpoint_dir: &Path,
    component: &str,
    out_path: &Path,
    config: serde_json::Value,
) -> Result<ConvertReport, ConvertError> {
    let comp_dir = checkpoint_dir.join(component);
    let index = ShardIndex::open_dir(&comp_dir)?;

    // Enumerating the shards is what makes completeness checkable: without it
    // a tensor held in a shard but missing from `weight_map` is never
    // converted, and a count over the index agrees with itself.
    let cross = index
        .cross_check()
        .map_err(|e| ConvertError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    if !cross.is_clean() {
        return Err(ConvertError::ShardIndexMismatch(Box::new(
            ShardIndexMismatch {
                component: component.to_string(),
                missing: cross.missing,
                unexpected: cross.unexpected,
                duplicated: cross.duplicated,
                misplaced: cross.misplaced,
                unreferenced: cross.unreferenced,
            },
        )));
    }

    let mut writer = LbiWriter::create(out_path, config)?;
    let mut total_bytes = 0u64;
    let mut dtypes: HashMap<String, usize> = HashMap::new();

    // Group by shard so each file is opened once, preserving name order so the
    // index is deterministic.
    let mut by_shard: HashMap<&str, Vec<&str>> = HashMap::new();
    for name in index.names_sorted() {
        let shard = index.shard_of(name).expect("name came from the index");
        by_shard.entry(shard).or_default().push(name);
    }
    let mut shards: Vec<&str> = by_shard.keys().copied().collect();
    shards.sort_unstable();

    let mut found = 0usize;
    for shard in shards {
        let mut st = SafetensorsFile::open(&index.path_of(shard))?;
        for name in by_shard[shard]
            .iter()
            .copied()
            .map(str::to_owned)
            .collect::<Vec<_>>()
        {
            let t = st.get(&name).ok_or_else(|| ConvertError::MissingTensor {
                tensor: name.clone(),
                shard: shard.to_string(),
            })?;
            let quant = scheme_for(&t.dtype, &name)?;
            let shape = t.shape.clone();
            let dtype = t.dtype.clone();
            let data = st.read(&name)?;
            writer.append(&name, &shape, quant, &data)?;
            total_bytes += data.len() as u64;
            *dtypes.entry(dtype).or_insert(0) += 1;
            found += 1;
        }
    }

    if found != index.len() {
        return Err(ConvertError::IndexMismatch {
            name: component.to_string(),
            indexed: index.len(),
            found,
        });
    }

    writer.finish()?;
    Ok(ConvertReport {
        component: component.to_string(),
        tensor_count: found,
        total_bytes,
        dtypes,
    })
}

/// Build `<component>.lbi` from a single-file checkpoint (no shard index).
pub fn convert_single_file(
    file: &Path,
    component: &str,
    out_path: &Path,
    config: serde_json::Value,
) -> Result<ConvertReport, ConvertError> {
    let mut st = SafetensorsFile::open(file)?;
    let names: Vec<String> = st.tensors().iter().map(|(n, _)| n.clone()).collect();
    let mut writer = LbiWriter::create(out_path, config)?;
    let mut total_bytes = 0u64;
    let mut dtypes: HashMap<String, usize> = HashMap::new();
    for name in &names {
        let t = st.get(name).expect("name taken from this file");
        let quant = scheme_for(&t.dtype, name)?;
        let shape = t.shape.clone();
        let dtype = t.dtype.clone();
        let data = st.read(name)?;
        writer.append(name, &shape, quant, &data)?;
        total_bytes += data.len() as u64;
        *dtypes.entry(dtype).or_insert(0) += 1;
    }
    writer.finish()?;
    Ok(ConvertReport {
        component: component.to_string(),
        tensor_count: names.len(),
        total_bytes,
        dtypes,
    })
}

/// Whether a component directory carries a shard index.
pub fn has_shard_index(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().any(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.ends_with(".index.json"))
            })
        })
        .unwrap_or(false)
}

/// The one shard of a component stored without a shard index.
///
/// Discovery is the same canonical, recursive walk the index path uses, so a
/// second shard in a subdirectory or behind a symlink is an error rather than
/// silently ignored.
pub fn single_shard(dir: &Path) -> Result<PathBuf, ConvertError> {
    ShardIndex::single_shard(dir)
        .map_err(|e| ConvertError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
}
