//! Checkpoint conversion: safetensors to `.lbi`.
//!
//! One component (transformer, vae, text_encoder) becomes one `.lbi`. Tensors
//! keep the dtype the checkpoint stores them in — BF16 stays BF16, F32 stays
//! F32 — so a later comparison against a reference cannot be blamed on the
//! container having changed the weights.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use lumen_format::QuantScheme;

use crate::lbi::{LbiError, LbiWriter};
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
