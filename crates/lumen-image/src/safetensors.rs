//! Minimal read-only safetensors reader.
//!
//! The format is an 8-byte little-endian header length, that many bytes of
//! JSON, then the tensor data. Each tensor entry carries `dtype`, `shape` and
//! its byte range inside the data section. Only the dtypes this pipeline stores
//! are recognized; anything else is reported rather than guessed.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

pub const DTYPE_F32: &str = "F32";
pub const DTYPE_F16: &str = "F16";
pub const DTYPE_BF16: &str = "BF16";
pub const HEADER_LEN_BYTES: u64 = 8;
/// Upper bound on the JSON header, so a corrupt length cannot reserve memory.
pub const MAX_HEADER_BYTES: u64 = 256 << 20;

#[derive(Debug)]
pub enum SafetensorsError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// The declared header length exceeds the file.
    HeaderTooLarge {
        declared: u64,
        file: u64,
    },
    /// A tensor entry lacks a field the format requires.
    MissingField {
        tensor: String,
        field: &'static str,
    },
    MalformedShape(String),
    /// A tensor's byte range runs past the end of the file.
    OutOfRange {
        tensor: String,
        end: u64,
        file: u64,
    },
    /// The dtype is valid safetensors but not one this pipeline stores.
    UnsupportedDtype(String),
}

impl std::fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Json(e) => write!(f, "header json: {e}"),
            Self::HeaderTooLarge { declared, file } => {
                write!(f, "header declares {declared} bytes but the file is {file}")
            }
            Self::MissingField { tensor, field } => {
                write!(f, "tensor {tensor} has no {field}")
            }
            Self::MalformedShape(t) => write!(f, "tensor {t} has a malformed shape"),
            Self::OutOfRange { tensor, end, file } => {
                write!(f, "tensor {tensor} ends at {end} but the file is {file}")
            }
            Self::UnsupportedDtype(d) => write!(f, "unsupported dtype {d}"),
        }
    }
}

impl std::error::Error for SafetensorsError {}

impl From<std::io::Error> for SafetensorsError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for SafetensorsError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

/// A safetensors header that refuses a repeated tensor key.
struct StrictHeader(std::collections::BTreeMap<String, serde_json::Value>);

impl<'de> serde::Deserialize<'de> for StrictHeader {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{MapAccess, Visitor};

        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = StrictHeader;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a safetensors header object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut acc: A) -> Result<StrictHeader, A::Error> {
                let mut out = std::collections::BTreeMap::new();
                while let Some((k, v)) = acc.next_entry::<String, serde_json::Value>()? {
                    if out.insert(k.clone(), v).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "tensor {k} appears more than once in the header"
                        )));
                    }
                }
                Ok(StrictHeader(out))
            }
        }
        d.deserialize_map(V)
    }
}

/// One tensor's dtype, shape and byte span within the data section.
#[derive(Debug, Clone)]
pub struct StTensor {
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Byte offset from the start of the data section.
    pub begin: u64,
    pub end: u64,
}

impl StTensor {
    pub fn byte_len(&self) -> u64 {
        self.end - self.begin
    }
}

/// A parsed safetensors file. Keeps the header in memory and reads data lazily.
#[derive(Debug)]
pub struct SafetensorsFile {
    file: BufReader<File>,
    data_start: u64,
    tensors: Vec<(String, StTensor)>,
    by_name: HashMap<String, usize>,
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self, SafetensorsError> {
        let file_len = std::fs::metadata(path)?.len();
        let mut file = BufReader::new(File::open(path)?);
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        let header_len = u64::from_le_bytes(len_bytes);
        // Bound the allocation before it happens: a hostile length must not
        // reserve address space, and the data section must fit the file.
        if header_len > MAX_HEADER_BYTES {
            return Err(SafetensorsError::HeaderTooLarge {
                declared: header_len,
                file: file_len,
            });
        }
        let data_start = HEADER_LEN_BYTES
            .checked_add(header_len)
            .filter(|s| *s <= file_len)
            .ok_or(SafetensorsError::HeaderTooLarge {
                declared: header_len,
                file: file_len,
            })?;
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)?;
        // The header is parsed through a strict map: `serde_json::Value` keeps
        // only the last value for a repeated key, which would hide a tensor
        // declared twice and silently drop the first payload.
        let raw: StrictHeader = serde_json::from_slice(&header)?;
        let obj = &raw.0;

        let mut tensors = Vec::with_capacity(obj.len());
        let mut by_name = HashMap::with_capacity(obj.len());
        for (name, v) in obj {
            // `__metadata__` is an optional free-form map, not a tensor.
            if name == "__metadata__" {
                continue;
            }
            let dtype = v
                .get("dtype")
                .and_then(|d| d.as_str())
                .ok_or_else(|| SafetensorsError::MissingField {
                    tensor: name.clone(),
                    field: "dtype",
                })?
                .to_string();
            let shape = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| SafetensorsError::MissingField {
                    tensor: name.clone(),
                    field: "shape",
                })?
                .iter()
                .map(|d| d.as_u64())
                .collect::<Option<Vec<u64>>>()
                .ok_or_else(|| SafetensorsError::MalformedShape(name.clone()))?;
            let offsets = v
                .get("data_offsets")
                .and_then(|s| s.as_array())
                .ok_or_else(|| SafetensorsError::MissingField {
                    tensor: name.clone(),
                    field: "data_offsets",
                })?;
            if offsets.len() != 2 {
                return Err(SafetensorsError::MalformedShape(name.clone()));
            }
            // Both bounds must be present, numeric and ordered; a missing or
            // non-numeric bound is a corrupt entry, not a zero.
            let begin = offsets[0]
                .as_u64()
                .ok_or_else(|| SafetensorsError::MissingField {
                    tensor: name.clone(),
                    field: "data_offsets[0]",
                })?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| SafetensorsError::MissingField {
                    tensor: name.clone(),
                    field: "data_offsets[1]",
                })?;
            if end < begin {
                return Err(SafetensorsError::MalformedShape(name.clone()));
            }
            let abs_end = data_start.checked_add(end);
            match abs_end {
                Some(e) if e <= file_len => {}
                _ => {
                    return Err(SafetensorsError::OutOfRange {
                        tensor: name.clone(),
                        end: abs_end.unwrap_or(u64::MAX),
                        file: file_len,
                    })
                }
            }
            by_name.insert(name.clone(), tensors.len());
            tensors.push((
                name.clone(),
                StTensor {
                    dtype,
                    shape,
                    begin,
                    end,
                },
            ));
        }
        Ok(Self {
            file,
            data_start,
            tensors,
            by_name,
        })
    }

    pub fn tensors(&self) -> &[(String, StTensor)] {
        &self.tensors
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&StTensor> {
        self.by_name.get(name).map(|&i| &self.tensors[i].1)
    }

    /// Read a tensor's raw bytes exactly as stored.
    pub fn read(&mut self, name: &str) -> Result<Vec<u8>, SafetensorsError> {
        let t = self
            .get(name)
            .ok_or_else(|| SafetensorsError::MissingField {
                tensor: name.to_string(),
                field: "tensor",
            })?
            .clone();
        let mut buf = vec![0u8; t.byte_len() as usize];
        let abs = self.data_start + t.begin;
        self.file.seek(SeekFrom::Start(abs))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }
}
