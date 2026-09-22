//! A minimal `.npy` reader, so the check binaries can compare against the tensors
//! the oracle dumps without a Python step in the loop.
//!
//! Only what the oracle writes is supported: C-order float32 and float64, in
//! version 1, 2 or 3 headers.

use std::path::Path;

#[derive(Debug)]
pub enum NpyError {
    Io(std::io::Error),
    NotNpy,
    UnsupportedVersion(u8),
    UnsupportedDtype(String),
    FortranOrder,
    BadHeader(String),
    LengthMismatch { expected: usize, found: usize },
}

impl std::fmt::Display for NpyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::NotNpy => write!(f, "not a .npy file"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported .npy version {v}"),
            Self::UnsupportedDtype(d) => write!(f, "unsupported dtype {d}"),
            Self::FortranOrder => write!(f, "fortran order is not supported"),
            Self::BadHeader(h) => write!(f, "malformed header: {h}"),
            Self::LengthMismatch { expected, found } => {
                write!(f, "expected {expected} bytes, found {found}")
            }
        }
    }
}

impl std::error::Error for NpyError {}

impl From<std::io::Error> for NpyError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// A decoded array. Values are always widened to f32.
#[derive(Debug, Clone)]
pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Npy {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

fn parse_shape(header: &str) -> Result<Vec<usize>, NpyError> {
    let at = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| NpyError::BadHeader("no shape".into()))?;
    let open = header[at..]
        .find('(')
        .ok_or_else(|| NpyError::BadHeader("no shape tuple".into()))?
        + at;
    let close = header[open..]
        .find(')')
        .ok_or_else(|| NpyError::BadHeader("unterminated shape".into()))?
        + open;
    let inner = &header[open + 1..close];
    let mut out = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        out.push(
            p.parse::<usize>()
                .map_err(|_| NpyError::BadHeader(format!("bad dimension {p}")))?,
        );
    }
    Ok(out)
}

fn parse_descr(header: &str) -> Result<String, NpyError> {
    let at = header
        .find("'descr'")
        .or_else(|| header.find("\"descr\""))
        .ok_or_else(|| NpyError::BadHeader("no descr".into()))?;
    let rest = &header[at + 7..];
    let first = rest
        .find('\'')
        .map(|i| (i, '\''))
        .or_else(|| rest.find('"').map(|i| (i, '"')))
        .ok_or_else(|| NpyError::BadHeader("no descr value".into()))?;
    let value = &rest[first.0 + 1..];
    let end = value
        .find(first.1)
        .ok_or_else(|| NpyError::BadHeader("unterminated descr".into()))?;
    Ok(value[..end].to_string())
}

/// Load a `.npy` file, widening every element to f32.
pub fn load(path: &Path) -> Result<Npy, NpyError> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(NpyError::NotNpy);
    }
    let major = bytes[6];
    let (hlen, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(NpyError::NotNpy);
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12usize,
            )
        }
        v => return Err(NpyError::UnsupportedVersion(v)),
    };
    let header_end = header_start + hlen;
    if header_end > bytes.len() {
        return Err(NpyError::BadHeader("header runs past the file".into()));
    }
    let header = String::from_utf8_lossy(&bytes[header_start..header_end]).to_string();
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true") {
        return Err(NpyError::FortranOrder);
    }
    let shape = parse_shape(&header)?;
    let descr = parse_descr(&header)?;
    let body = &bytes[header_end..];

    let n: usize = shape.iter().product();
    let data = match descr.as_str() {
        "<f4" | "|f4" => {
            if body.len() != n * 4 {
                return Err(NpyError::LengthMismatch {
                    expected: n * 4,
                    found: body.len(),
                });
            }
            body.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        "<f8" | "|f8" => {
            if body.len() != n * 8 {
                return Err(NpyError::LengthMismatch {
                    expected: n * 8,
                    found: body.len(),
                });
            }
            body.chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect()
        }
        "<i8" | "|i8" => {
            if body.len() != n * 8 {
                return Err(NpyError::LengthMismatch {
                    expected: n * 8,
                    found: body.len(),
                });
            }
            body.chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect()
        }
        other => return Err(NpyError::UnsupportedDtype(other.to_string())),
    };
    Ok(Npy { shape, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_float32_array() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
        let hlen = header.len() as u16;
        bytes.extend_from_slice(&hlen.to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let dir = std::env::temp_dir().join(format!("npy-t-{}", std::process::id()));
        std::fs::write(&dir, &bytes).unwrap();
        let a = load(&dir).unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let _ = std::fs::remove_file(dir);
    }

    #[test]
    fn a_non_npy_file_is_rejected() {
        let dir = std::env::temp_dir().join(format!("npy-bad-{}", std::process::id()));
        std::fs::write(&dir, b"not an npy at all").unwrap();
        assert!(matches!(load(&dir), Err(NpyError::NotNpy)));
        let _ = std::fs::remove_file(dir);
    }

    #[test]
    fn a_truncated_body_is_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }";
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 8]); // half the needed body
        let dir = std::env::temp_dir().join(format!("npy-short-{}", std::process::id()));
        std::fs::write(&dir, &bytes).unwrap();
        assert!(matches!(load(&dir), Err(NpyError::LengthMismatch { .. })));
        let _ = std::fs::remove_file(dir);
    }
}
