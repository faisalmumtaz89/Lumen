//! Check a converted `.lbi` against its source checkpoint.
//!
//! Every source tensor is present with the same shape and
//! scheme, and its stored bytes are identical to the source bytes. The
//! comparison is exact — the container stores dtypes unchanged, so there is
//! nothing to tolerate.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use lumen_format::QuantScheme;
use lumen_image::convert::{has_shard_index, single_shard};
use lumen_image::lbi::LbiFile;
use lumen_image::safetensors::{SafetensorsFile, DTYPE_BF16, DTYPE_F16, DTYPE_F32};
use lumen_image::shard_index::ShardIndex;

fn expected_scheme(dtype: &str) -> Option<QuantScheme> {
    match dtype {
        DTYPE_F32 => Some(QuantScheme::F32),
        DTYPE_F16 => Some(QuantScheme::F16),
        DTYPE_BF16 => Some(QuantScheme::Bf16),
        _ => None,
    }
}

/// Where each source tensor lives, so the source is opened once per shard.
enum Source {
    Single(PathBuf),
    Sharded {
        index: ShardIndex,
        by_shard: HashMap<String, Vec<String>>,
    },
}

impl Source {
    fn open(comp_dir: &Path) -> Result<Self, String> {
        if has_shard_index(comp_dir) {
            let index = ShardIndex::open_dir(comp_dir).map_err(|e| e.to_string())?;
            // Completeness must be established here too, not inherited from the
            // index: verifying only the names the index happens to list cannot
            // detect a tensor the index omits.
            let cross = index.cross_check().map_err(|e| e.to_string())?;
            if !cross.is_clean() {
                return Err(format!(
                    "{}: the index and the shards disagree — missing={:?} extra={:?} \
                     dup={:?} misplaced={:?} unindexed_shards={:?}",
                    comp_dir.display(),
                    cross.missing,
                    cross.unexpected,
                    cross.duplicated,
                    cross.misplaced,
                    cross.unreferenced
                ));
            }
            let mut by_shard: HashMap<String, Vec<String>> = HashMap::new();
            for n in index.names_sorted() {
                by_shard
                    .entry(index.shard_of(n).unwrap().to_string())
                    .or_default()
                    .push(n.to_string());
            }
            Ok(Source::Sharded { index, by_shard })
        } else {
            Ok(Source::Single(
                single_shard(comp_dir).map_err(|e| e.to_string())?,
            ))
        }
    }

    /// Shard names in a deterministic order, empty for a single file.
    fn shards(&self) -> Vec<String> {
        match self {
            Source::Single(_) => vec![String::new()],
            Source::Sharded { by_shard, .. } => {
                let mut v: Vec<String> = by_shard.keys().cloned().collect();
                v.sort();
                v
            }
        }
    }

    fn path_for(&self, shard: &str) -> PathBuf {
        match self {
            Source::Single(p) => p.clone(),
            Source::Sharded { index, .. } => index.path_of(shard),
        }
    }

    fn names_in(&self, shard: &str) -> Option<&Vec<String>> {
        match self {
            Source::Single(_) => None,
            Source::Sharded { by_shard, .. } => by_shard.get(shard),
        }
    }
}

fn verify_component(ckpt: &Path, component: &str, lbi_path: &Path) -> Result<(usize, u64), String> {
    let comp_dir = ckpt.join(component);
    let f = LbiFile::open(lbi_path).map_err(|e| format!("{}: {e}", lbi_path.display()))?;
    let source = Source::open(&comp_dir)?;

    // Gather the source inventory first: name, shape, dtype.
    let mut inv: Vec<(String, Vec<u64>, String)> = Vec::new();
    let mut opened: HashMap<String, SafetensorsFile> = HashMap::new();
    for shard in source.shards() {
        let st = SafetensorsFile::open(&source.path_for(&shard)).map_err(|e| e.to_string())?;
        match source.names_in(&shard) {
            Some(names) => {
                for n in names {
                    let t = st
                        .get(n)
                        .ok_or_else(|| format!("{component}/{n}: missing from shard {shard}"))?;
                    inv.push((n.clone(), t.shape.clone(), t.dtype.clone()));
                }
            }
            None => {
                for (n, t) in st.tensors() {
                    inv.push((n.clone(), t.shape.clone(), t.dtype.clone()));
                }
            }
        }
        opened.insert(shard, st);
    }

    // The tensor sets must be equal, in both directions.
    let mut src_names: Vec<&str> = inv.iter().map(|(n, _, _)| n.as_str()).collect();
    let mut lbi_names: Vec<&str> = f.entries().iter().map(|e| e.name.as_str()).collect();
    src_names.sort_unstable();
    lbi_names.sort_unstable();
    if src_names != lbi_names {
        let missing: Vec<_> = src_names
            .iter()
            .filter(|n| lbi_names.binary_search(n).is_err())
            .take(5)
            .collect();
        let extra: Vec<_> = lbi_names
            .iter()
            .filter(|n| src_names.binary_search(n).is_err())
            .take(5)
            .collect();
        return Err(format!(
            "{component}: tensor sets differ ({} source vs {} stored) missing={missing:?} extra={extra:?}",
            src_names.len(),
            lbi_names.len()
        ));
    }

    // Shape, scheme, and bytes, tensor by tensor.
    let shard_of: HashMap<&str, String> = match &source {
        Source::Single(_) => inv
            .iter()
            .map(|(n, _, _)| (n.as_str(), String::new()))
            .collect(),
        Source::Sharded { index, .. } => inv
            .iter()
            .map(|(n, _, _)| (n.as_str(), index.shard_of(n).unwrap().to_string()))
            .collect(),
    };
    let mut bytes = 0u64;
    for (name, shape, dtype) in &inv {
        let e = f.get(name).expect("name sets equal");
        if &e.shape != shape {
            return Err(format!(
                "{component}/{name}: shape {:?} != {shape:?}",
                e.shape
            ));
        }
        let want = expected_scheme(dtype)
            .ok_or_else(|| format!("{component}/{name}: source dtype {dtype} unsupported"))?;
        if e.quant != want {
            return Err(format!(
                "{component}/{name}: scheme {:?} != {want:?}",
                e.quant
            ));
        }
        let shard = &shard_of[name.as_str()];
        let src = opened
            .get_mut(shard)
            .expect("shard opened above")
            .read(name)
            .map_err(|e| e.to_string())?;
        let got = f.tensor_bytes(name).expect("entry present");
        if got != &src[..] {
            return Err(format!(
                "{component}/{name}: stored {} bytes differ from source {} bytes",
                got.len(),
                src.len()
            ));
        }
        bytes += got.len() as u64;
    }
    Ok((inv.len(), bytes))
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = || {
        eprintln!("usage: lbi-verify <checkpoint-dir> <lbi-dir>");
        std::process::exit(2)
    };
    let ckpt = PathBuf::from(args.next().unwrap_or_else(usage));
    let dir = PathBuf::from(args.next().unwrap_or_else(usage));

    let mut total = 0usize;
    for component in ["transformer", "vae", "text_encoder"] {
        let (n, bytes) = verify_component(&ckpt, component, &dir.join(format!("{component}.lbi")))?;
        total += n;
        println!(
            "{component:14} {n:4} tensors  {:.2} GiB byte-exact",
            bytes as f64 / (1u64 << 30) as f64
        );
    }
    println!("all {total} tensors verified byte-exact against the checkpoint");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
