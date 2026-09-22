//! The `.lbi` container: a named, shaped tensor store for image models.
//!
//! LBC (see `lumen-format`) carries a fixed set of LLM slot names and derives
//! every tensor's shape from the model hyperparameters. An image model has conv
//! stacks, attention blocks and a VAE decoder whose shapes cannot be derived
//! that way, so `.lbi` stores each tensor's shape beside its bytes.
//!
//! ```text
//! [magic "LBI1"][u32 version][u32 tensor_count][u32 name_bytes][u32 config_len]
//! [names    : concatenated UTF-8, addressed by name_off/name_len]
//! [u32 config_len][config json]
//! [index    : tensor_count fixed-width entries]
//! [padding to 64-byte alignment]
//! [blob 0][blob 1] ... [blob N-1]   each blob 64-byte aligned
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use lumen_format::QuantScheme;

pub const LBI_MAGIC: [u8; 4] = *b"LBI1";
pub const LBI_VERSION: u32 = 1;
/// Blob offsets and the start of the blob region are aligned to this.
pub const LBI_ALIGN: u64 = 64;
/// Fixed size of the header that precedes the name section.
const HEADER_BYTES: u64 = 4 + 4 + 4 + 4 + 4;
/// Sanity bounds, so a corrupt header cannot drive an unbounded allocation.
const MAX_TENSOR_COUNT: u32 = 1_000_000;
const MAX_NAME_BYTES: u32 = 64 << 20;
const MAX_CONFIG_BYTES: u32 = 16 << 20;
/// Fixed index-entry size: 8 (name) + 4 (rank plus padding) + 24 (shape)
/// + 1 (quant) + 7 (pad) + 8 (offset) + 8 (length) = 60 bytes.
const INDEX_ENTRY_BYTES: u64 = 8 + 4 + 24 + 1 + 7 + 8 + 8;
const SHAPE_BUDGET: usize = 24;
const MAX_RANK: usize = SHAPE_BUDGET / 4;
/// Dimensions are stored as u32, so this is the largest representable extent.
const MAX_DIM: u64 = u32::MAX as u64;

#[derive(Debug)]
pub enum LbiError {
    Io(std::io::Error),
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
    TooManyTensors(u32),
    NameSectionTooLarge(u32),
    ConfigTooLarge(u32),
    /// A section or blob range runs past the end of the file.
    Truncated {
        what: &'static str,
        needed: u64,
        available: u64,
    },
    UnknownQuantTag(u8),
    InvalidUtf8(String),
    /// Two entries carry the same name.
    DuplicateTensor(String),
    /// The stored byte length does not match the shape and scheme.
    LengthMismatch {
        name: String,
        expected: u64,
        actual: u64,
    },
    /// A rank above what the fixed index entry can hold.
    RankTooLarge {
        name: String,
        rank: usize,
    },
    /// A dimension, or the product of a shape, exceeds what the index stores.
    ShapeTooLarge {
        name: String,
    },
    /// The scheme has no storage rule yet.
    UnsupportedScheme(QuantScheme),
    /// A tensor's blob offset does not sit on the container's alignment.
    MisalignedOffset {
        name: String,
        offset: u64,
    },
}

impl std::fmt::Display for LbiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::BadMagic(m) => write!(f, "not an .lbi file (magic {m:?})"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported .lbi version {v}"),
            Self::TooManyTensors(n) => write!(f, "tensor count {n} exceeds the limit"),
            Self::NameSectionTooLarge(n) => write!(f, "name section {n} bytes exceeds the limit"),
            Self::ConfigTooLarge(n) => write!(f, "config section {n} bytes exceeds the limit"),
            Self::Truncated {
                what,
                needed,
                available,
            } => write!(f, "{what} needs {needed} bytes but only {available} remain"),
            Self::UnknownQuantTag(t) => write!(f, "unknown quantization tag {t}"),
            Self::InvalidUtf8(n) => write!(f, "tensor name is not valid UTF-8: {n}"),
            Self::DuplicateTensor(n) => write!(f, "tensor {n} appears more than once"),
            Self::LengthMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {name} stores {actual} bytes but its shape needs {expected}"
            ),
            Self::RankTooLarge { name, rank } => {
                write!(
                    f,
                    "tensor {name} has rank {rank}, above the {MAX_RANK} the index holds"
                )
            }
            Self::UnsupportedScheme(s) => write!(f, "no storage rule for scheme {s:?}"),
            Self::ShapeTooLarge { name } => write!(
                f,
                "tensor {name} has a dimension or element count above what the index stores"
            ),
            Self::MisalignedOffset { name, offset } => write!(
                f,
                "tensor {name} starts at offset {offset}, which is not {LBI_ALIGN}-byte aligned"
            ),
        }
    }
}

impl std::error::Error for LbiError {}

impl From<std::io::Error> for LbiError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for LbiError {
    fn from(e: serde_json::Error) -> Self {
        // The config section is JSON; a parse failure there is an io-level
        // corruption report, not a separate category.
        Self::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// One tensor's place in the file.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<u64>,
    pub quant: QuantScheme,
    /// Offset from the start of the blob region.
    pub offset: u64,
    pub length: u64,
}

impl TensorEntry {
    /// Element count, rejecting a product that overflows `u64`.
    pub fn num_elements(&self) -> Result<u64, LbiError> {
        self.shape.iter().try_fold(1u64, |acc, &d| {
            acc.checked_mul(d).ok_or_else(|| LbiError::ShapeTooLarge {
                name: self.name.clone(),
            })
        })
    }

    /// Bytes this tensor occupies for its shape and scheme.
    ///
    /// Only the unquantized schemes have a storage rule so far; a quantized
    /// scheme reports an error rather than a guessed size.
    pub fn expected_length(&self) -> Result<u64, LbiError> {
        let n = self.num_elements()?;
        let bytes = match self.quant {
            QuantScheme::F32 => n.checked_mul(4),
            QuantScheme::F16 | QuantScheme::Bf16 => n.checked_mul(2),
            other => return Err(LbiError::UnsupportedScheme(other)),
        };
        bytes.ok_or_else(|| LbiError::ShapeTooLarge {
            name: self.name.clone(),
        })
    }
}

/// Round `offset` up to the container's alignment.
fn align_up(offset: u64) -> u64 {
    offset.div_ceil(LBI_ALIGN) * LBI_ALIGN
}

/// Writes an `.lbi` file.
///
/// Tensor bytes stream through a temporary file, so converting a checkpoint
/// never holds more than the caller's own buffer in memory, and the index can
/// be written in one pass at the end when every name and size is known.
pub struct LbiWriter {
    out: BufWriter<File>,
    blob: BufWriter<File>,
    path: PathBuf,
    tmp: PathBuf,
    config: Vec<u8>,
    entries: Vec<TensorEntry>,
    names: HashMap<String, ()>,
    cursor: u64,
}

impl LbiWriter {
    /// Start a writer. The final path is not created until
    /// [`finish`](Self::finish); tensor bytes accumulate in a sibling
    /// temporary file.
    pub fn create(path: &Path, config: serde_json::Value) -> Result<Self, LbiError> {
        let tmp = path.with_extension("lbi.blobs.tmp");
        Ok(Self {
            out: BufWriter::new(File::create(path)?),
            // Truncating create: any leftover temporary from an interrupted run
            // is discarded rather than appended to.
            blob: BufWriter::new(File::create(&tmp)?),
            path: path.to_path_buf(),
            tmp,
            config: serde_json::to_vec(&config)?,
            entries: Vec::new(),
            names: HashMap::new(),
            cursor: 0,
        })
    }

    /// Append a tensor. `data` must be exactly the length its shape and scheme
    /// require, so a mis-sized read cannot be stored silently.
    pub fn append(
        &mut self,
        name: &str,
        shape: &[u64],
        quant: QuantScheme,
        data: &[u8],
    ) -> Result<(), LbiError> {
        if shape.len() > MAX_RANK {
            return Err(LbiError::RankTooLarge {
                name: name.to_string(),
                rank: shape.len(),
            });
        }
        // Dimensions narrow to u32 on the wire; reject rather than truncate.
        if shape.iter().any(|&d| d > MAX_DIM) {
            return Err(LbiError::ShapeTooLarge {
                name: name.to_string(),
            });
        }
        if self.names.insert(name.to_string(), ()).is_some() {
            return Err(LbiError::DuplicateTensor(name.to_string()));
        }
        // The offset must be where the data lands, which is after the padding
        // that aligns this tensor. Recording the pre-padding cursor points the
        // entry at the pad bytes instead of the tensor.
        let pad = align_up(self.cursor) - self.cursor;
        let offset = self.cursor + pad;
        let entry = TensorEntry {
            name: name.to_string(),
            shape: shape.to_vec(),
            quant,
            offset,
            length: data.len() as u64,
        };
        let expected = entry.expected_length()?;
        if expected != entry.length {
            return Err(LbiError::LengthMismatch {
                name: name.to_string(),
                expected,
                actual: entry.length,
            });
        }

        for _ in 0..pad {
            self.blob.write_all(&[0u8])?;
        }
        self.blob.write_all(data)?;
        self.cursor += pad + entry.length;
        self.entries.push(entry);
        Ok(())
    }

    pub fn tensor_count(&self) -> usize {
        self.entries.len()
    }

    /// Write the header, names, config and index, append the accumulated blob
    /// region, and remove the temporary file.
    pub fn finish(mut self) -> Result<(), LbiError> {
        let mut names = Vec::new();
        let mut offsets = Vec::with_capacity(self.entries.len());
        for e in &self.entries {
            offsets.push((names.len() as u32, e.name.len() as u32));
            names.extend_from_slice(e.name.as_bytes());
        }
        // The same limits the reader enforces, applied before writing: a file
        // the reader would reject must not be produced.
        if self.entries.len() as u64 > MAX_TENSOR_COUNT as u64 {
            return Err(LbiError::TooManyTensors(self.entries.len() as u32));
        }
        if names.len() as u64 > MAX_NAME_BYTES as u64 {
            return Err(LbiError::NameSectionTooLarge(names.len() as u32));
        }
        if self.config.len() as u64 > MAX_CONFIG_BYTES as u64 {
            return Err(LbiError::ConfigTooLarge(self.config.len() as u32));
        }
        let index_bytes = self.entries.len() as u64 * INDEX_ENTRY_BYTES;
        let blob_start = align_up(
            HEADER_BYTES + names.len() as u64 + 4 + self.config.len() as u64 + index_bytes,
        );
        // Flush the tensor bytes before the blob region is appended.
        self.blob.flush()?;

        self.out.write_all(&LBI_MAGIC)?;
        self.out.write_all(&LBI_VERSION.to_le_bytes())?;
        self.out
            .write_all(&(self.entries.len() as u32).to_le_bytes())?;
        self.out.write_all(&(names.len() as u32).to_le_bytes())?;
        self.out
            .write_all(&(self.config.len() as u32).to_le_bytes())?;
        self.out.write_all(&names)?;
        self.out
            .write_all(&(self.config.len() as u32).to_le_bytes())?;
        self.out.write_all(&self.config)?;

        for (e, (noff, nlen)) in self.entries.iter().zip(&offsets) {
            let mut idx = [0u8; INDEX_ENTRY_BYTES as usize];
            idx[0..4].copy_from_slice(&noff.to_le_bytes());
            idx[4..8].copy_from_slice(&nlen.to_le_bytes());
            idx[8] = e.shape.len() as u8;
            for (r, d) in e.shape.iter().enumerate() {
                let p = 12 + r * 4;
                idx[p..p + 4].copy_from_slice(&(*d as u32).to_le_bytes());
            }
            idx[12 + SHAPE_BUDGET] = e.quant.to_u8();
            let off_at = 12 + SHAPE_BUDGET + 8;
            idx[off_at..off_at + 8].copy_from_slice(&e.offset.to_le_bytes());
            idx[off_at + 8..off_at + 16].copy_from_slice(&e.length.to_le_bytes());
            self.out.write_all(&idx)?;
        }

        let written =
            HEADER_BYTES + names.len() as u64 + 4 + self.config.len() as u64 + index_bytes;
        for _ in written..blob_start {
            self.out.write_all(&[0u8])?;
        }

        // Stream the blob region across; never the whole file at once.
        let mut src = File::open(&self.tmp)?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = src.read(&mut buf)?;
            if n == 0 {
                break;
            }
            self.out.write_all(&buf[..n])?;
        }
        self.out.flush()?;
        self.out
            .into_inner()
            .map_err(|e| LbiError::Io(e.into_error()))?;
        std::fs::remove_file(&self.tmp)?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Reads an `.lbi` file.
#[derive(Debug)]
pub struct LbiFile {
    data: memmap2::Mmap,
    config: serde_json::Value,
    entries: Vec<TensorEntry>,
    by_name: HashMap<String, usize>,
    blob_start: u64,
}

impl LbiFile {
    pub fn open(path: &Path) -> Result<Self, LbiError> {
        let file = File::open(path)?;
        // Safety: the mapping is read-only and outlives every borrow of it.
        let data = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_bytes(data)
    }

    fn from_bytes(data: memmap2::Mmap) -> Result<Self, LbiError> {
        let len = data.len() as u64;
        if len < HEADER_BYTES {
            return Err(LbiError::Truncated {
                what: "header",
                needed: HEADER_BYTES,
                available: len,
            });
        }
        let magic: [u8; 4] = data[0..4].try_into().unwrap();
        if magic != LBI_MAGIC {
            return Err(LbiError::BadMagic(magic));
        }
        let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
        let version = u32_at(4);
        if version != LBI_VERSION {
            return Err(LbiError::UnsupportedVersion(version));
        }
        let tensor_count = u32_at(8);
        let name_bytes = u32_at(12);
        let config_len = u32_at(16);
        if tensor_count > MAX_TENSOR_COUNT {
            return Err(LbiError::TooManyTensors(tensor_count));
        }
        if name_bytes > MAX_NAME_BYTES {
            return Err(LbiError::NameSectionTooLarge(name_bytes));
        }
        if config_len > MAX_CONFIG_BYTES {
            return Err(LbiError::ConfigTooLarge(config_len));
        }

        let names_start = HEADER_BYTES;
        let names_end = names_start + name_bytes as u64;
        let cfg_prefix_at = names_end;
        let cfg_start = cfg_prefix_at + 4;
        let cfg_end = cfg_start + config_len as u64;
        let index_start = cfg_end;
        let index_end = index_start + tensor_count as u64 * INDEX_ENTRY_BYTES;
        for (what, need) in [
            ("name section", names_end),
            ("config prefix", cfg_start),
            ("config", cfg_end),
            ("index", index_end),
        ] {
            if need > len {
                return Err(LbiError::Truncated {
                    what,
                    needed: need,
                    available: len,
                });
            }
        }
        let cfg_prefix = u32_at(cfg_prefix_at as usize);
        if cfg_prefix != config_len {
            return Err(LbiError::Truncated {
                what: "config length prefix",
                needed: cfg_prefix as u64,
                available: config_len as u64,
            });
        }
        let config: serde_json::Value =
            serde_json::from_slice(&data[cfg_start as usize..cfg_end as usize]).map_err(|e| {
                LbiError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            })?;

        if align_up(index_end) > len {
            return Err(LbiError::Truncated {
                what: "blob region",
                needed: align_up(index_end),
                available: len,
            });
        }
        let names = &data[names_start as usize..names_end as usize];
        let blob_start = align_up(index_end);
        let mut entries = Vec::with_capacity(tensor_count as usize);
        let mut by_name = HashMap::with_capacity(tensor_count as usize);
        for i in 0..tensor_count as u64 {
            let o = (index_start + i * INDEX_ENTRY_BYTES) as usize;
            let name_off = u32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as usize;
            let name_len = u32::from_le_bytes(data[o + 4..o + 8].try_into().unwrap()) as usize;
            if name_off + name_len > names.len() {
                return Err(LbiError::Truncated {
                    what: "tensor name",
                    needed: (name_off + name_len) as u64,
                    available: names.len() as u64,
                });
            }
            let name = std::str::from_utf8(&names[name_off..name_off + name_len])
                .map_err(|e| LbiError::InvalidUtf8(e.to_string()))?
                .to_string();
            let rank = data[o + 8] as usize;
            if rank > MAX_RANK {
                return Err(LbiError::RankTooLarge { name, rank });
            }
            let mut shape = Vec::with_capacity(rank);
            for r in 0..rank {
                let p = o + 12 + r * 4;
                shape.push(u32::from_le_bytes(data[p..p + 4].try_into().unwrap()) as u64);
            }
            let quant_at = o + 12 + SHAPE_BUDGET;
            let quant = QuantScheme::from_u8(data[quant_at])
                .map_err(|_| LbiError::UnknownQuantTag(data[quant_at]))?;
            let off_at = quant_at + 8;
            let offset = u64::from_le_bytes(data[off_at..off_at + 8].try_into().unwrap());
            let length = u64::from_le_bytes(data[off_at + 8..off_at + 16].try_into().unwrap());
            // Every tensor must start on an aligned offset, and its absolute
            // range must lie inside the file. Both additions are checked so a
            // hostile offset cannot wrap into a valid-looking range.
            if offset % LBI_ALIGN != 0 {
                return Err(LbiError::MisalignedOffset {
                    name: name.clone(),
                    offset,
                });
            }
            let abs_start = blob_start.checked_add(offset);
            let abs_end = abs_start.and_then(|s| s.checked_add(length));
            match abs_end {
                Some(e) if e <= len => {}
                _ => {
                    return Err(LbiError::Truncated {
                        what: "tensor blob",
                        needed: abs_end.unwrap_or(u64::MAX),
                        available: len,
                    })
                }
            }
            let entry = TensorEntry {
                name: name.clone(),
                shape,
                quant,
                offset,
                length,
            };
            // A stored length that disagrees with the shape is a corrupt index.
            let expected = entry.expected_length()?;
            if expected != entry.length {
                return Err(LbiError::LengthMismatch {
                    name: name.clone(),
                    expected,
                    actual: entry.length,
                });
            }
            if by_name.insert(name.clone(), entries.len()).is_some() {
                return Err(LbiError::DuplicateTensor(name));
            }
            entries.push(entry);
        }

        Ok(Self {
            data,
            config,
            entries,
            by_name,
            blob_start,
        })
    }

    pub fn config(&self) -> &serde_json::Value {
        &self.config
    }

    pub fn entries(&self) -> &[TensorEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.by_name.get(name).map(|&i| &self.entries[i])
    }

    /// Raw stored bytes for a tensor.
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let e = self.get(name)?;
        let start = (self.blob_start + e.offset) as usize;
        Some(&self.data[start..start + e.length as usize])
    }

    /// Decode a tensor to `f32`.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, LbiError> {
        let e = self
            .get(name)
            .ok_or_else(|| LbiError::DuplicateTensor(name.to_string()))?;
        let bytes = self.tensor_bytes(name).expect("entry resolved above");
        let n = e.num_elements()? as usize;
        let mut out = vec![0f32; n];
        match e.quant {
            QuantScheme::F32 => {
                for (i, c) in bytes.chunks_exact(4).enumerate() {
                    out[i] = f32::from_le_bytes(c.try_into().unwrap());
                }
            }
            QuantScheme::Bf16 => {
                for (i, c) in bytes.chunks_exact(2).enumerate() {
                    let bits = u16::from_le_bytes(c.try_into().unwrap());
                    out[i] = f32::from_bits((bits as u32) << 16);
                }
            }
            QuantScheme::F16 => {
                for (i, c) in bytes.chunks_exact(2).enumerate() {
                    out[i] = half_to_f32(u16::from_le_bytes(c.try_into().unwrap()));
                }
            }
            other => return Err(LbiError::UnsupportedScheme(other)),
        }
        Ok(out)
    }
}

/// IEEE 754 half to single precision.
pub fn half_to_f32(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = (h as u32 >> 10) & 0x1f;
    let man = h as u32 & 0x3ff;
    let bits = match exp {
        0 => {
            if man == 0 {
                sign
            } else {
                // subnormal: normalize into the f32 exponent range
                let mut e = 113u32;
                let mut m = man;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3ff) << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}
