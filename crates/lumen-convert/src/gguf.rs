//! Zero-dependency GGUF file format parser.
//!
//! GGUF model format parser. Parses the header, metadata
//! key-value pairs, and tensor info entries. It does NOT load tensor data into
//! memory (which can be many GB), making it suitable for streaming conversion.
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

use std::io::Read;

// ---------------------------------------------------------------------------
// Safety limits -- prevent unbounded allocations from untrusted wire data
// ---------------------------------------------------------------------------

/// Maximum number of tensor info entries allowed in a single GGUF file.
const MAX_TENSOR_ENTRIES: u64 = 100_000;

/// Maximum number of metadata key-value pairs allowed.
const MAX_METADATA_ENTRIES: u64 = 1_000_000;

/// Maximum byte length of a single GGUF string (64 MiB).
const MAX_STRING_LENGTH: u64 = 64 * 1024 * 1024;

/// Maximum number of elements in a metadata array.
const MAX_ARRAY_ELEMENTS: u64 = 10_000_000;

/// Maximum number of dimensions for a single tensor.
const MAX_TENSOR_DIMS: u32 = 16;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when parsing a GGUF file.
#[derive(Debug)]
pub enum GgufError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// File does not start with the "GGUF" magic bytes.
    InvalidMagic([u8; 4]),
    /// GGUF version is not supported (v2 and v3 are handled).
    UnsupportedVersion(u32),
    /// Encountered an unknown metadata value type tag.
    InvalidValueType(u32),
    /// Encountered an unknown GGML tensor type tag (kept for strict parsing).
    InvalidGgmlType(u32),
    /// A length-prefixed string contained invalid UTF-8.
    InvalidUtf8,
    /// The file ended before all expected data was read.
    TruncatedFile,
    /// A count from the wire format exceeds a safety limit.
    ResourceLimit(String),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidMagic(m) => write!(f, "invalid GGUF magic: {m:02x?}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported GGUF version: {v}"),
            Self::InvalidValueType(t) => write!(f, "invalid metadata value type: {t}"),
            Self::InvalidGgmlType(t) => write!(f, "invalid GGML tensor type: {t}"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 in string"),
            Self::TruncatedFile => write!(f, "unexpected end of file"),
            Self::ResourceLimit(msg) => write!(f, "resource limit exceeded: {msg}"),
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Self::TruncatedFile
        } else {
            Self::Io(e)
        }
    }
}

// ---------------------------------------------------------------------------
// GGML tensor types
// ---------------------------------------------------------------------------

/// GGML tensor element type (quantization format).
// GGML-convention names like Q4_K are standard in the LLM quantization
// ecosystem. We preserve them for clarity and interoperability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    /// MXFP4 (Microscaling FP4): 32 elements in 17 bytes (1-byte E8M0 scale + 16 nibbles).
    /// Used by unsloth/Qwen3.5 GGUF files for shared expert gate/up weights.
    MXFP4,
    /// Forward-compatibility: any type tag we do not recognize.
    Unknown(u32),
}

impl GgmlType {
    /// Decode a GGML type tag from the wire format.
    pub fn from_u32(tag: u32) -> Self {
        match tag {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            // 4, 5 are deprecated/removed
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            39 => Self::MXFP4,
            other => Self::Unknown(other),
        }
    }

    /// Encode back to the wire tag.
    pub fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
            Self::IQ2_XXS => 16,
            Self::IQ2_XS => 17,
            Self::IQ3_XXS => 18,
            Self::IQ1_S => 19,
            Self::IQ4_NL => 20,
            Self::IQ3_S => 21,
            Self::IQ2_S => 22,
            Self::IQ4_XS => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::IQ1_M => 29,
            Self::BF16 => 30,
            Self::MXFP4 => 39,
            Self::Unknown(t) => t,
        }
    }

    /// Number of elements per quantization block.
    ///
    /// For unquantized types (F32, F16, BF16, F64) and integer types this is 1.
    /// Returns `None` for types whose block geometry we do not know.
    pub fn block_size(&self) -> Option<u64> {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => Some(1),
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => Some(1),
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => Some(32),
            Self::Q8_0 | Self::Q8_1 => Some(32),
            Self::IQ4_NL | Self::IQ4_XS => Some(32),
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => {
                Some(256)
            }
            Self::IQ2_XXS | Self::IQ2_XS | Self::IQ2_S => Some(256),
            Self::IQ3_XXS | Self::IQ3_S => Some(256),
            Self::IQ1_S | Self::IQ1_M => Some(256),
            Self::MXFP4 => Some(32),
            Self::Unknown(_) => None,
        }
    }

    /// Bytes per quantization block.
    ///
    /// Returns `None` for types whose layout we do not know.
    pub fn type_size(&self) -> Option<u64> {
        match self {
            Self::F32 => Some(4),
            Self::F16 => Some(2),
            Self::BF16 => Some(2),
            Self::F64 => Some(8),
            Self::I8 => Some(1),
            Self::I16 => Some(2),
            Self::I32 => Some(4),
            Self::I64 => Some(8),
            Self::Q4_0 => Some(18),
            Self::Q4_1 => Some(20),
            Self::Q5_0 => Some(22),
            Self::Q5_1 => Some(24),
            Self::Q8_0 => Some(34),
            Self::Q8_1 => Some(36),
            Self::Q2_K => Some(84),
            Self::Q3_K => Some(110),
            Self::Q4_K => Some(144),
            Self::Q5_K => Some(176),
            Self::Q6_K => Some(210),
            Self::Q8_K => Some(292),
            Self::IQ2_XXS => Some(66),
            Self::IQ2_XS => Some(74),
            Self::IQ2_S => Some(82),
            Self::IQ3_XXS => Some(98),
            Self::IQ3_S => Some(110),
            Self::IQ1_S => Some(50),
            Self::IQ1_M => Some(56),
            Self::IQ4_NL => Some(18),
            Self::IQ4_XS => Some(36),
            Self::MXFP4 => Some(17),
            Self::Unknown(_) => None,
        }
    }

    /// Total byte size for `n_elements` values of this type.
    ///
    /// Returns `None` if the type's block geometry is unknown.
    pub fn byte_size_for(&self, n_elements: u64) -> Option<u64> {
        let bs = self.block_size()?;
        let ts = self.type_size()?;
        let blocks = n_elements.div_ceil(bs);
        Some(blocks * ts)
    }

    /// Map this GGML type to the corresponding LBC [`QuantScheme`], if one
    /// exists.
    ///
    /// Types without a direct LBC equivalent return `None`.
    pub fn to_lbc_quant(&self) -> Option<lumen_format::QuantScheme> {
        use lumen_format::QuantScheme;
        match self {
            Self::F32 => Some(QuantScheme::F32),
            Self::F16 => Some(QuantScheme::F16),
            Self::BF16 => Some(QuantScheme::Bf16),
            Self::Q4_0 => Some(QuantScheme::Q4_0),
            Self::Q4_1 => Some(QuantScheme::Q4_1),
            Self::Q5_0 => Some(QuantScheme::Q5_0),
            Self::Q8_0 => Some(QuantScheme::Q8_0),
            Self::Q4_K => Some(QuantScheme::Q4_K),
            Self::Q5_K => Some(QuantScheme::Q5_K),
            Self::Q6_K => Some(QuantScheme::Q6_K),
            Self::Q2_K => Some(QuantScheme::Q2_K),
            Self::Q3_K => Some(QuantScheme::Q3_K),
            _ => None,
        }
    }

    /// Whether `dequantize_to_f32_bytes` can convert this type to F32.
    ///
    /// This is the authoritative check used by `try_compute_opt_slice` to
    /// decide whether a tensor with no direct LBC mapping can be force-
    /// dequantized to F32 during conversion (vs. silently skipped).
    pub fn has_dequant_path(&self) -> bool {
        matches!(self,
            Self::F32 | Self::F16 | Self::BF16
            | Self::Q8_0 | Self::Q8_1
            | Self::Q4_0 | Self::Q4_1
            | Self::Q5_0 | Self::Q5_1
            | Self::Q4_K | Self::Q5_K | Self::Q6_K
            | Self::Q2_K | Self::Q3_K
            | Self::MXFP4
        )
    }
}

// ---------------------------------------------------------------------------
// Metadata value types
// ---------------------------------------------------------------------------

/// A GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(GgufArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// A typed array of GGUF metadata values.
#[derive(Debug, Clone)]
pub struct GgufArray {
    /// The element type tag (same encoding as the value type enum).
    pub element_type: u32,
    /// The elements. All elements share the same type.
    pub values: Vec<GgufValue>,
}

// ---------------------------------------------------------------------------
// Tensor info
// ---------------------------------------------------------------------------

/// Information about a single tensor in the GGUF file.
///
/// This is the metadata only -- the actual weight bytes are not loaded.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g. `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Shape dimensions (outermost first, matching GGML convention).
    pub dims: Vec<u64>,
    /// GGML element type / quantization format.
    pub ggml_type: GgmlType,
    /// Byte offset relative to the start of the tensor data section.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements (product of all dimensions).
    pub fn n_elements(&self) -> u64 {
        if self.dims.is_empty() {
            0
        } else {
            self.dims.iter().copied().product()
        }
    }

    /// Total byte size of this tensor's data, computed from type and shape.
    ///
    /// Returns `None` if the GGML type's block geometry is unknown.
    pub fn byte_size(&self) -> Option<u64> {
        self.ggml_type.byte_size_for(self.n_elements())
    }
}

// ---------------------------------------------------------------------------
// Parsed GGUF file
// ---------------------------------------------------------------------------

/// A parsed GGUF file.
///
/// Contains the header, metadata key-value pairs, and tensor descriptors.
/// Tensor data is NOT loaded -- callers can use [`Self::tensor_data_offset`]
/// to seek into the file and read individual tensors on demand.
pub struct GgufFile {
    /// GGUF format version (2 or 3).
    pub version: u32,
    /// Ordered metadata key-value pairs.
    pub metadata: Vec<(String, GgufValue)>,
    /// Tensor descriptors.
    pub tensors: Vec<GgufTensorInfo>,
    /// Absolute byte offset in the file where tensor data begins.
    pub data_offset: u64,
    /// Alignment boundary in bytes (from `general.alignment` or default 32).
    pub alignment: u32,
}

impl GgufFile {
    /// Parse a GGUF file from a reader.
    ///
    /// Only reads the header, metadata, and tensor info. Does NOT read any
    /// tensor data. The reader must be positioned at the start of the file.
    /// Supports GGUF v2 and v3 formats.
    pub fn parse<R: Read>(reader: &mut R) -> Result<Self, GgufError> {
        let mut cr = CountingReader::new(reader);

        // -- Header ----------------------------------------------------------
        let magic = read_bytes::<4>(&mut cr)?;
        if magic != *b"GGUF" {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = read_u32_le(&mut cr)?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // In v2, n_tensors and n_kv are u32; in v3 they are u64.
        let n_tensors: u64 = if version == 2 {
            read_u32_le(&mut cr)? as u64
        } else {
            read_u64_le(&mut cr)?
        };
        if n_tensors > MAX_TENSOR_ENTRIES {
            return Err(GgufError::ResourceLimit(format!(
                "tensor count {n_tensors} exceeds maximum {MAX_TENSOR_ENTRIES}"
            )));
        }

        let n_kv: u64 = if version == 2 {
            read_u32_le(&mut cr)? as u64
        } else {
            read_u64_le(&mut cr)?
        };
        if n_kv > MAX_METADATA_ENTRIES {
            return Err(GgufError::ResourceLimit(format!(
                "metadata entry count {n_kv} exceeds maximum {MAX_METADATA_ENTRIES}"
            )));
        }

        // -- Metadata KV pairs -----------------------------------------------
        let mut metadata = Vec::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = read_gguf_string_v(&mut cr, version)?;
            let value_type = read_u32_le(&mut cr)?;
            let value = read_value_v(&mut cr, value_type, version)?;
            metadata.push((key, value));
        }

        // Check for alignment override.
        let alignment = metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| match v {
                GgufValue::U32(a) => Some(*a),
                GgufValue::U64(a) => Some(*a as u32),
                GgufValue::I32(a) if *a > 0 => Some(*a as u32),
                _ => None,
            })
            .unwrap_or(32);

        // -- Tensor info entries ---------------------------------------------
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_gguf_string_v(&mut cr, version)?;
            let n_dims = read_u32_le(&mut cr)?;
            if n_dims > MAX_TENSOR_DIMS {
                return Err(GgufError::ResourceLimit(format!(
                    "tensor dimension count {n_dims} exceeds maximum {MAX_TENSOR_DIMS}"
                )));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64_le(&mut cr)?);
            }
            let type_tag = read_u32_le(&mut cr)?;
            let ggml_type = GgmlType::from_u32(type_tag);
            let offset = read_u64_le(&mut cr)?;
            tensors.push(GgufTensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        // -- Compute data_offset (aligned past all header bytes) -------------
        let total_header_bytes = cr.bytes_read;
        let align = alignment as u64;
        let data_offset = align_up(total_header_bytes, align);

        // Skip alignment padding so reader is positioned at tensor data.
        let padding = data_offset - total_header_bytes;
        if padding > 0 {
            skip_bytes(&mut cr, padding)?;
        }

        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
        })
    }

    /// Parse a GGUF file at the given path.
    pub fn open(path: &std::path::Path) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::parse(&mut reader)
    }

    // -- Metadata accessors --------------------------------------------------

    /// Look up a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Get a string metadata value.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.get_metadata(key)? {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get a `u32` metadata value, with safe coercion from compatible types.
    ///
    /// Returns the value if it is a U32, or if it is a U64 that fits in u32,
    /// or if it is an I32 that is non-negative (and thus fits in u32).
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.get_metadata(key)? {
            GgufValue::U32(v) => Some(*v),
            GgufValue::U64(v) => u32::try_from(*v).ok(),
            GgufValue::I32(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Get a `u64` metadata value, with safe coercion from compatible types.
    ///
    /// Returns the value if it is a U64, or if it is a U32 (lossless widening),
    /// or if it is an I32 that is non-negative.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.get_metadata(key)? {
            GgufValue::U64(v) => Some(*v),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    /// Get an `f32` metadata value.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.get_metadata(key)? {
            GgufValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Get an array of strings metadata value.
    pub fn get_string_array(&self, key: &str) -> Option<Vec<&str>> {
        match self.get_metadata(key)? {
            GgufValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.values.len());
                for v in &arr.values {
                    match v {
                        GgufValue::String(s) => result.push(s.as_str()),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Get an array of f32 metadata values (for tokenizer.ggml.scores).
    pub fn get_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        match self.get_metadata(key)? {
            GgufValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.values.len());
                for v in &arr.values {
                    match v {
                        GgufValue::F32(f) => result.push(*f),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Get an array of u32 metadata values, with i32 coercion (for tokenizer.ggml.token_type).
    pub fn get_u32_array(&self, key: &str) -> Option<Vec<u32>> {
        match self.get_metadata(key)? {
            GgufValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.values.len());
                for v in &arr.values {
                    match v {
                        GgufValue::U32(n) => result.push(*n),
                        GgufValue::I32(n) => result.push(u32::try_from(*n).ok()?),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    // -- Tensor accessors ----------------------------------------------------

    /// Find a tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get the absolute file offset for a tensor's data.
    pub fn tensor_data_offset(&self, tensor: &GgufTensorInfo) -> u64 {
        self.data_offset + tensor.offset
    }
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("version", &self.version)
            .field("n_metadata", &self.metadata.len())
            .field("n_tensors", &self.tensors.len())
            .field("data_offset", &self.data_offset)
            .field("alignment", &self.alignment)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Wire helpers (all little-endian)
// ---------------------------------------------------------------------------

fn read_bytes<const N: usize>(r: &mut impl Read) -> Result<[u8; N], GgufError> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_u8_le(r: &mut impl Read) -> Result<u8, GgufError> {
    Ok(read_bytes::<1>(r)?[0])
}

fn read_i8_le(r: &mut impl Read) -> Result<i8, GgufError> {
    Ok(read_bytes::<1>(r)?[0] as i8)
}

fn read_u16_le(r: &mut impl Read) -> Result<u16, GgufError> {
    Ok(u16::from_le_bytes(read_bytes(r)?))
}

fn read_i16_le(r: &mut impl Read) -> Result<i16, GgufError> {
    Ok(i16::from_le_bytes(read_bytes(r)?))
}

fn read_u32_le(r: &mut impl Read) -> Result<u32, GgufError> {
    Ok(u32::from_le_bytes(read_bytes(r)?))
}

fn read_i32_le(r: &mut impl Read) -> Result<i32, GgufError> {
    Ok(i32::from_le_bytes(read_bytes(r)?))
}

fn read_u64_le(r: &mut impl Read) -> Result<u64, GgufError> {
    Ok(u64::from_le_bytes(read_bytes(r)?))
}

fn read_i64_le(r: &mut impl Read) -> Result<i64, GgufError> {
    Ok(i64::from_le_bytes(read_bytes(r)?))
}

fn read_f32_le(r: &mut impl Read) -> Result<f32, GgufError> {
    Ok(f32::from_le_bytes(read_bytes(r)?))
}

fn read_f64_le(r: &mut impl Read) -> Result<f64, GgufError> {
    Ok(f64::from_le_bytes(read_bytes(r)?))
}

/// Version-aware GGUF string reader.
///
/// In v2 the string length prefix is a u32; in v3 it is a u64.
fn read_gguf_string_v(r: &mut impl Read, version: u32) -> Result<String, GgufError> {
    let len: u64 = if version == 2 {
        read_u32_le(r)? as u64
    } else {
        read_u64_le(r)?
    };
    if len > MAX_STRING_LENGTH {
        return Err(GgufError::ResourceLimit(format!(
            "string length {len} exceeds maximum {MAX_STRING_LENGTH}"
        )));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| GgufError::InvalidUtf8)
}

/// Version-aware metadata value reader.
///
/// In v2 the array count and string length prefix are u32; in v3 they are u64.
fn read_value_v(r: &mut impl Read, value_type: u32, version: u32) -> Result<GgufValue, GgufError> {
    match value_type {
        0 => Ok(GgufValue::U8(read_u8_le(r)?)),
        1 => Ok(GgufValue::I8(read_i8_le(r)?)),
        2 => Ok(GgufValue::U16(read_u16_le(r)?)),
        3 => Ok(GgufValue::I16(read_i16_le(r)?)),
        4 => Ok(GgufValue::U32(read_u32_le(r)?)),
        5 => Ok(GgufValue::I32(read_i32_le(r)?)),
        6 => Ok(GgufValue::F32(read_f32_le(r)?)),
        7 => {
            let b = read_u8_le(r)?;
            Ok(GgufValue::Bool(b != 0))
        }
        8 => Ok(GgufValue::String(read_gguf_string_v(r, version)?)),
        9 => {
            let element_type = read_u32_le(r)?;
            let count: u64 = if version == 2 {
                read_u32_le(r)? as u64
            } else {
                read_u64_le(r)?
            };
            if count > MAX_ARRAY_ELEMENTS {
                return Err(GgufError::ResourceLimit(format!(
                    "array element count {count} exceeds maximum {MAX_ARRAY_ELEMENTS}"
                )));
            }
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                values.push(read_value_v(r, element_type, version)?);
            }
            Ok(GgufValue::Array(GgufArray {
                element_type,
                values,
            }))
        }
        10 => Ok(GgufValue::U64(read_u64_le(r)?)),
        11 => Ok(GgufValue::I64(read_i64_le(r)?)),
        12 => Ok(GgufValue::F64(read_f64_le(r)?)),
        other => Err(GgufError::InvalidValueType(other)),
    }
}

/// Skip `n` bytes from the reader.
fn skip_bytes(r: &mut impl Read, n: u64) -> Result<(), GgufError> {
    let mut remaining = n;
    let mut buf = [0u8; 4096];
    while remaining > 0 {
        let to_read = remaining.min(buf.len() as u64) as usize;
        r.read_exact(&mut buf[..to_read])?;
        remaining -= to_read as u64;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CountingReader -- tracks total bytes read through a Read adapter
// ---------------------------------------------------------------------------

struct CountingReader<'a, R> {
    inner: &'a mut R,
    bytes_read: u64,
}

impl<'a, R: Read> CountingReader<'a, R> {
    fn new(inner: &'a mut R) -> Self {
        Self {
            inner,
            bytes_read: 0,
        }
    }
}

impl<R: Read> Read for CountingReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes_read += n as u64;
        Ok(n)
    }
}

// ---------------------------------------------------------------------------
// GGUF Builder (for tests and future use)
// ---------------------------------------------------------------------------

/// Builder for creating GGUF files in memory.
///
/// Primarily used for testing the parser, but also useful for generating
/// synthetic GGUF files for benchmarking.
pub struct GgufBuilder {
    version: u32,
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<(String, GgmlType, Vec<u64>, Vec<u8>)>,
    alignment: u32,
}

impl GgufBuilder {
    /// Create a new builder with GGUF v3 defaults.
    pub fn new() -> Self {
        Self {
            version: 3,
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: 32,
        }
    }

    /// Set the GGUF version to emit (2 or 3).
    pub fn version(&mut self, version: u32) -> &mut Self {
        self.version = version;
        self
    }

    /// Add a string metadata key-value pair.
    pub fn add_string(&mut self, key: &str, value: &str) -> &mut Self {
        self.metadata.push((
            key.to_string(),
            GgufValue::String(value.to_string()),
        ));
        self
    }

    /// Add a u32 metadata key-value pair.
    pub fn add_u32(&mut self, key: &str, value: u32) -> &mut Self {
        self.metadata
            .push((key.to_string(), GgufValue::U32(value)));
        self
    }

    /// Add a u64 metadata key-value pair.
    pub fn add_u64(&mut self, key: &str, value: u64) -> &mut Self {
        self.metadata
            .push((key.to_string(), GgufValue::U64(value)));
        self
    }

    /// Add an i32 metadata key-value pair.
    pub fn add_i32(&mut self, key: &str, value: i32) -> &mut Self {
        self.metadata
            .push((key.to_string(), GgufValue::I32(value)));
        self
    }

    /// Add an f32 metadata key-value pair.
    pub fn add_f32(&mut self, key: &str, value: f32) -> &mut Self {
        self.metadata
            .push((key.to_string(), GgufValue::F32(value)));
        self
    }

    /// Add a bool metadata key-value pair.
    pub fn add_bool(&mut self, key: &str, value: bool) -> &mut Self {
        self.metadata
            .push((key.to_string(), GgufValue::Bool(value)));
        self
    }

    /// Add an f32 array metadata key-value pair.
    pub fn add_f32_array(&mut self, key: &str, values: &[f32]) -> &mut Self {
        let arr = GgufArray {
            element_type: 6, // F32
            values: values.iter().map(|v| GgufValue::F32(*v)).collect(),
        };
        self.metadata
            .push((key.to_string(), GgufValue::Array(arr)));
        self
    }

    /// Add a u32 array metadata key-value pair.
    pub fn add_u32_array(&mut self, key: &str, values: &[u32]) -> &mut Self {
        let arr = GgufArray {
            element_type: 4, // U32
            values: values.iter().map(|v| GgufValue::U32(*v)).collect(),
        };
        self.metadata
            .push((key.to_string(), GgufValue::Array(arr)));
        self
    }

    /// Add a string array metadata key-value pair.
    pub fn add_string_array(&mut self, key: &str, values: &[&str]) -> &mut Self {
        let arr = GgufArray {
            element_type: 8, // STRING
            values: values
                .iter()
                .map(|s| GgufValue::String(s.to_string()))
                .collect(),
        };
        self.metadata
            .push((key.to_string(), GgufValue::Array(arr)));
        self
    }

    /// Add a tensor with raw data bytes.
    pub fn add_tensor(
        &mut self,
        name: &str,
        ggml_type: GgmlType,
        dims: &[u64],
        data: Vec<u8>,
    ) -> &mut Self {
        self.tensors
            .push((name.to_string(), ggml_type, dims.to_vec(), data));
        self
    }

    /// Add a tensor with F32 data from a slice.
    pub fn add_f32_tensor(&mut self, name: &str, dims: &[u64], values: &[f32]) -> &mut Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_tensor(name, GgmlType::F32, dims, data)
    }

    /// Build the GGUF file as a byte vector.
    ///
    /// When `version` is 2, header counts and string length prefixes are
    /// emitted as u32 instead of u64 (the only wire-format difference
    /// between GGUF v2 and v3).
    pub fn build(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let v = self.version;

        // Header
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&v.to_le_bytes());
        if v == 2 {
            buf.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
            buf.extend_from_slice(&(self.metadata.len() as u32).to_le_bytes());
        } else {
            buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            buf.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
        }

        // Metadata KV pairs
        for (key, value) in &self.metadata {
            write_string_to_v(&mut buf, key, v);
            write_value_to_v(&mut buf, value, v);
        }

        // Tensor infos -- compute data-section-relative offsets
        let align = self.alignment as u64;
        let mut tensor_offsets = Vec::new();
        let mut data_cursor = 0u64;
        for (name, ggml_type, dims, data) in &self.tensors {
            data_cursor = align_up(data_cursor, align);

            write_string_to_v(&mut buf, name, v);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&ggml_type.to_u32().to_le_bytes());
            buf.extend_from_slice(&data_cursor.to_le_bytes());

            tensor_offsets.push(data_cursor);
            data_cursor += data.len() as u64;
        }

        // Alignment padding to start of tensor data
        let current_len = buf.len() as u64;
        let tensor_data_start = align_up(current_len, align);
        let padding = (tensor_data_start - current_len) as usize;
        buf.resize(buf.len() + padding, 0);

        // Write tensor data
        for (i, (_name, _ggml_type, _dims, data)) in self.tensors.iter().enumerate() {
            let target_offset = tensor_data_start + tensor_offsets[i];
            let current = buf.len() as u64;
            if target_offset > current {
                let pad = (target_offset - current) as usize;
                buf.resize(buf.len() + pad, 0);
            }
            buf.extend_from_slice(data);
        }

        buf
    }
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Builder serialization helpers
// ---------------------------------------------------------------------------

/// Version-aware string writer. In v2 the length prefix is u32; in v3 it is u64.
fn write_string_to_v(buf: &mut Vec<u8>, s: &str, version: u32) {
    if version == 2 {
        buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    } else {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    }
    buf.extend_from_slice(s.as_bytes());
}

/// Version-aware metadata value writer (type tag + body).
fn write_value_to_v(buf: &mut Vec<u8>, value: &GgufValue, version: u32) {
    match value {
        GgufValue::U8(v) => {
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.push(*v);
        }
        GgufValue::I8(v) => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.push(*v as u8);
        }
        GgufValue::U16(v) => {
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I16(v) => {
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::U32(v) => {
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I32(v) => {
            buf.extend_from_slice(&5u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::F32(v) => {
            buf.extend_from_slice(&6u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::Bool(v) => {
            buf.extend_from_slice(&7u32.to_le_bytes());
            buf.push(u8::from(*v));
        }
        GgufValue::String(v) => {
            buf.extend_from_slice(&8u32.to_le_bytes());
            write_string_to_v(buf, v, version);
        }
        GgufValue::Array(arr) => {
            buf.extend_from_slice(&9u32.to_le_bytes());
            buf.extend_from_slice(&arr.element_type.to_le_bytes());
            if version == 2 {
                buf.extend_from_slice(&(arr.values.len() as u32).to_le_bytes());
            } else {
                buf.extend_from_slice(&(arr.values.len() as u64).to_le_bytes());
            }
            for elem in &arr.values {
                write_value_body_v(buf, elem, version);
            }
        }
        GgufValue::U64(v) => {
            buf.extend_from_slice(&10u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I64(v) => {
            buf.extend_from_slice(&11u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::F64(v) => {
            buf.extend_from_slice(&12u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// Version-aware body writer (no type tag prefix) -- used for array elements.
fn write_value_body_v(buf: &mut Vec<u8>, value: &GgufValue, version: u32) {
    match value {
        GgufValue::U8(v) => buf.push(*v),
        GgufValue::I8(v) => buf.push(*v as u8),
        GgufValue::U16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::U32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Bool(v) => buf.push(u8::from(*v)),
        GgufValue::String(v) => write_string_to_v(buf, v, version),
        GgufValue::U64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Array(_) => {
            // Nested arrays: write full tagged form.
            write_value_to_v(buf, value, version);
        }
    }
}

/// Round `offset` up to the next multiple of `alignment`.
fn align_up(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return offset;
    }
    (offset + alignment - 1) & !(alignment - 1)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Test data helpers ---------------------------------------------------

    fn write_test_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn pad_to_alignment(buf: &mut Vec<u8>, alignment: usize) {
        let current = buf.len();
        let aligned = (current + alignment - 1) & !(alignment - 1);
        buf.resize(aligned, 0);
    }

    fn make_minimal_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        write_test_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_test_string(&mut buf, "llama");

        write_test_string(&mut buf, "token_embd.weight");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&8u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        pad_to_alignment(&mut buf, 32);
        buf.extend_from_slice(&vec![0u8; 1024]);
        buf
    }

    // -- Parsing tests -------------------------------------------------------

    #[test]
    fn parse_minimal_gguf() {
        let data = make_minimal_gguf();
        let file = GgufFile::parse(&mut data.as_slice()).unwrap();

        assert_eq!(file.version, 3);
        assert_eq!(file.alignment, 32);
        assert_eq!(file.metadata.len(), 1);
        assert_eq!(file.tensors.len(), 1);
        assert_eq!(file.get_string("general.architecture"), Some("llama"));

        let t = file.find_tensor("token_embd.weight").unwrap();
        assert_eq!(t.dims, vec![32, 8]);
        assert_eq!(t.ggml_type, GgmlType::F32);
        assert_eq!(t.offset, 0);
        assert_eq!(t.n_elements(), 256);
        assert_eq!(t.byte_size(), Some(1024));
        assert_eq!(file.data_offset % 32, 0);
        assert_eq!(file.tensor_data_offset(t), file.data_offset);
    }

    #[test]
    fn parse_multiple_kv_types() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&7u64.to_le_bytes());

        write_test_string(&mut buf, "test.u8");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.push(42);

        write_test_string(&mut buf, "test.i32");
        buf.extend_from_slice(&5u32.to_le_bytes());
        buf.extend_from_slice(&(-100i32).to_le_bytes());

        write_test_string(&mut buf, "test.f32");
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&3.14f32.to_le_bytes());

        write_test_string(&mut buf, "test.bool");
        buf.extend_from_slice(&7u32.to_le_bytes());
        buf.push(1);

        write_test_string(&mut buf, "test.string");
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_test_string(&mut buf, "hello");

        write_test_string(&mut buf, "test.u64");
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&999u64.to_le_bytes());

        write_test_string(&mut buf, "test.f64");
        buf.extend_from_slice(&12u32.to_le_bytes());
        buf.extend_from_slice(&2.718f64.to_le_bytes());

        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        assert_eq!(file.metadata.len(), 7);

        match file.get_metadata("test.u8").unwrap() {
            GgufValue::U8(v) => assert_eq!(*v, 42),
            other => panic!("expected U8, got {other:?}"),
        }
        match file.get_metadata("test.i32").unwrap() {
            GgufValue::I32(v) => assert_eq!(*v, -100),
            other => panic!("expected I32, got {other:?}"),
        }
        assert_eq!(file.get_f32("test.f32"), Some(3.14));
        match file.get_metadata("test.bool").unwrap() {
            GgufValue::Bool(v) => assert!(*v),
            other => panic!("expected Bool, got {other:?}"),
        }
        assert_eq!(file.get_string("test.string"), Some("hello"));
        assert_eq!(file.get_u64("test.u64"), Some(999));
        match file.get_metadata("test.f64").unwrap() {
            GgufValue::F64(v) => assert!((v - 2.718).abs() < 1e-10),
            other => panic!("expected F64, got {other:?}"),
        }
    }

    #[test]
    fn parse_i16_u16_i8_i64_values() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());

        write_test_string(&mut buf, "test.i8");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.push(0xFE);

        write_test_string(&mut buf, "test.u16");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&1000u16.to_le_bytes());

        write_test_string(&mut buf, "test.i16");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(-500i16).to_le_bytes());

        write_test_string(&mut buf, "test.i64");
        buf.extend_from_slice(&11u32.to_le_bytes());
        buf.extend_from_slice(&(-1_000_000i64).to_le_bytes());

        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();

        match file.get_metadata("test.i8").unwrap() {
            GgufValue::I8(v) => assert_eq!(*v, -2),
            other => panic!("expected I8, got {other:?}"),
        }
        match file.get_metadata("test.u16").unwrap() {
            GgufValue::U16(v) => assert_eq!(*v, 1000),
            other => panic!("expected U16, got {other:?}"),
        }
        match file.get_metadata("test.i16").unwrap() {
            GgufValue::I16(v) => assert_eq!(*v, -500),
            other => panic!("expected I16, got {other:?}"),
        }
        match file.get_metadata("test.i64").unwrap() {
            GgufValue::I64(v) => assert_eq!(*v, -1_000_000),
            other => panic!("expected I64, got {other:?}"),
        }
    }

    #[test]
    fn parse_array_of_u32s() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        write_test_string(&mut buf, "test.array");
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&20u32.to_le_bytes());
        buf.extend_from_slice(&30u32.to_le_bytes());

        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        match file.get_metadata("test.array").unwrap() {
            GgufValue::Array(arr) => {
                assert_eq!(arr.element_type, 4);
                assert_eq!(arr.values.len(), 3);
                match &arr.values[0] {
                    GgufValue::U32(v) => assert_eq!(*v, 10),
                    other => panic!("expected U32, got {other:?}"),
                }
                match &arr.values[2] {
                    GgufValue::U32(v) => assert_eq!(*v, 30),
                    other => panic!("expected U32, got {other:?}"),
                }
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[test]
    fn parse_array_of_strings() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        write_test_string(&mut buf, "test.strings");
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        write_test_string(&mut buf, "alpha");
        write_test_string(&mut buf, "beta");

        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        match file.get_metadata("test.strings").unwrap() {
            GgufValue::Array(arr) => {
                assert_eq!(arr.element_type, 8);
                assert_eq!(arr.values.len(), 2);
                match &arr.values[0] {
                    GgufValue::String(s) => assert_eq!(s, "alpha"),
                    other => panic!("expected String, got {other:?}"),
                }
                match &arr.values[1] {
                    GgufValue::String(s) => assert_eq!(s, "beta"),
                    other => panic!("expected String, got {other:?}"),
                }
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[test]
    fn parse_multiple_tensors() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        write_test_string(&mut buf, "weight");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        write_test_string(&mut buf, "bias");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&64u64.to_le_bytes());

        pad_to_alignment(&mut buf, 32);
        buf.extend_from_slice(&vec![0u8; 128]);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        assert_eq!(file.tensors.len(), 2);

        let w = file.find_tensor("weight").unwrap();
        assert_eq!(w.dims, vec![4, 4]);
        assert_eq!(w.ggml_type, GgmlType::F32);
        assert_eq!(w.n_elements(), 16);
        assert_eq!(w.byte_size(), Some(64));

        let b = file.find_tensor("bias").unwrap();
        assert_eq!(b.dims, vec![4]);
        assert_eq!(b.ggml_type, GgmlType::F16);
        assert_eq!(b.n_elements(), 4);
        assert_eq!(b.byte_size(), Some(8));
        assert_eq!(file.tensor_data_offset(b), file.data_offset + 64);
    }

    #[test]
    fn parse_zero_dim_tensor() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        write_test_string(&mut buf, "scalar");
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        let t = file.find_tensor("scalar").unwrap();
        assert!(t.dims.is_empty());
        assert_eq!(t.n_elements(), 0);
    }

    #[test]
    fn parse_empty_file() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        assert_eq!(file.version, 3);
        assert!(file.metadata.is_empty());
        assert!(file.tensors.is_empty());
        assert_eq!(file.data_offset % 32, 0);
    }

    #[test]
    fn parse_bool_false() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        write_test_string(&mut buf, "test.boolfalse");
        buf.extend_from_slice(&7u32.to_le_bytes());
        buf.push(0);

        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        match file.get_metadata("test.boolfalse").unwrap() {
            GgufValue::Bool(v) => assert!(!v),
            other => panic!("expected Bool(false), got {other:?}"),
        }
    }

    // -- GgmlType tests ------------------------------------------------------

    #[test]
    fn ggml_type_roundtrip() {
        let types = [
            GgmlType::F32,  GgmlType::F16,  GgmlType::Q4_0, GgmlType::Q4_1,
            GgmlType::Q5_0, GgmlType::Q5_1, GgmlType::Q8_0, GgmlType::Q8_1,
            GgmlType::Q2_K, GgmlType::Q3_K, GgmlType::Q4_K, GgmlType::Q5_K,
            GgmlType::Q6_K, GgmlType::Q8_K,
            GgmlType::IQ2_XXS, GgmlType::IQ2_XS, GgmlType::IQ3_XXS, GgmlType::IQ1_S,
            GgmlType::IQ4_NL, GgmlType::IQ3_S, GgmlType::IQ2_S, GgmlType::IQ4_XS,
            GgmlType::I8, GgmlType::I16, GgmlType::I32, GgmlType::I64,
            GgmlType::F64, GgmlType::IQ1_M, GgmlType::BF16, GgmlType::MXFP4,
        ];
        for t in types {
            let tag = t.to_u32();
            let recovered = GgmlType::from_u32(tag);
            assert_eq!(t, recovered, "roundtrip failed for tag {tag}");
        }
    }

    #[test]
    fn ggml_type_unknown_roundtrip() {
        let t = GgmlType::from_u32(9999);
        assert_eq!(t, GgmlType::Unknown(9999));
        assert_eq!(t.to_u32(), 9999);
    }

    #[test]
    fn ggml_type_deprecated_tags_become_unknown() {
        assert_eq!(GgmlType::from_u32(4), GgmlType::Unknown(4));
        assert_eq!(GgmlType::from_u32(5), GgmlType::Unknown(5));
    }

    #[test]
    fn ggml_block_size_and_type_size() {
        assert_eq!(GgmlType::F32.block_size(), Some(1));
        assert_eq!(GgmlType::F32.type_size(), Some(4));
        assert_eq!(GgmlType::F16.block_size(), Some(1));
        assert_eq!(GgmlType::F16.type_size(), Some(2));
        assert_eq!(GgmlType::BF16.block_size(), Some(1));
        assert_eq!(GgmlType::BF16.type_size(), Some(2));
        assert_eq!(GgmlType::F64.block_size(), Some(1));
        assert_eq!(GgmlType::F64.type_size(), Some(8));
        assert_eq!(GgmlType::Q4_0.block_size(), Some(32));
        assert_eq!(GgmlType::Q4_0.type_size(), Some(18));
        assert_eq!(GgmlType::Q4_1.block_size(), Some(32));
        assert_eq!(GgmlType::Q4_1.type_size(), Some(20));
        assert_eq!(GgmlType::Q5_0.block_size(), Some(32));
        assert_eq!(GgmlType::Q5_0.type_size(), Some(22));
        assert_eq!(GgmlType::Q5_1.block_size(), Some(32));
        assert_eq!(GgmlType::Q5_1.type_size(), Some(24));
        assert_eq!(GgmlType::Q8_0.block_size(), Some(32));
        assert_eq!(GgmlType::Q8_0.type_size(), Some(34));
        assert_eq!(GgmlType::Q8_1.block_size(), Some(32));
        assert_eq!(GgmlType::Q8_1.type_size(), Some(36));
        assert_eq!(GgmlType::Q2_K.block_size(), Some(256));
        assert_eq!(GgmlType::Q2_K.type_size(), Some(84));
        assert_eq!(GgmlType::Q4_K.block_size(), Some(256));
        assert_eq!(GgmlType::Q4_K.type_size(), Some(144));
        assert_eq!(GgmlType::Q6_K.block_size(), Some(256));
        assert_eq!(GgmlType::Q6_K.type_size(), Some(210));
        assert_eq!(GgmlType::Q8_K.block_size(), Some(256));
        assert_eq!(GgmlType::Q8_K.type_size(), Some(292));
        assert_eq!(GgmlType::MXFP4.block_size(), Some(32));
        assert_eq!(GgmlType::MXFP4.type_size(), Some(17));
        assert_eq!(GgmlType::Unknown(9999).block_size(), None);
        assert_eq!(GgmlType::Unknown(9999).type_size(), None);
    }

    #[test]
    fn ggml_byte_size_for() {
        assert_eq!(GgmlType::F32.byte_size_for(256), Some(1024));
        assert_eq!(GgmlType::Q4_0.byte_size_for(256), Some(144));
        assert_eq!(GgmlType::Q4_K.byte_size_for(256), Some(144));
        assert_eq!(GgmlType::Q4_0.byte_size_for(33), Some(36));
        assert_eq!(GgmlType::MXFP4.byte_size_for(1048576), Some(557056)); // 32768 blocks * 17 bytes
        assert_eq!(GgmlType::F32.byte_size_for(0), Some(0));
        assert_eq!(GgmlType::Unknown(9999).byte_size_for(100), None);
    }

    // -- to_lbc_quant --------------------------------------------------------

    #[test]
    fn to_lbc_quant_mapping() {
        use lumen_format::QuantScheme;

        assert_eq!(GgmlType::F32.to_lbc_quant(), Some(QuantScheme::F32));
        assert_eq!(GgmlType::F16.to_lbc_quant(), Some(QuantScheme::F16));
        assert_eq!(GgmlType::BF16.to_lbc_quant(), Some(QuantScheme::Bf16));
        assert_eq!(GgmlType::Q4_0.to_lbc_quant(), Some(QuantScheme::Q4_0));
        assert_eq!(GgmlType::Q4_1.to_lbc_quant(), Some(QuantScheme::Q4_1));
        assert_eq!(GgmlType::Q5_0.to_lbc_quant(), Some(QuantScheme::Q5_0));
        assert_eq!(GgmlType::Q8_0.to_lbc_quant(), Some(QuantScheme::Q8_0));
        assert_eq!(GgmlType::Q4_K.to_lbc_quant(), Some(QuantScheme::Q4_K));
        assert_eq!(GgmlType::Q5_K.to_lbc_quant(), Some(QuantScheme::Q5_K));
        assert_eq!(GgmlType::Q6_K.to_lbc_quant(), Some(QuantScheme::Q6_K));
        assert_eq!(GgmlType::Q2_K.to_lbc_quant(), Some(QuantScheme::Q2_K));
        assert_eq!(GgmlType::Q3_K.to_lbc_quant(), Some(QuantScheme::Q3_K));
        assert_eq!(GgmlType::Q8_1.to_lbc_quant(), None);
        assert_eq!(GgmlType::Q5_1.to_lbc_quant(), None);
        assert_eq!(GgmlType::Q8_K.to_lbc_quant(), None);
        assert_eq!(GgmlType::F64.to_lbc_quant(), None);
        assert_eq!(GgmlType::I8.to_lbc_quant(), None);
        assert_eq!(GgmlType::IQ2_XXS.to_lbc_quant(), None);
        assert_eq!(GgmlType::MXFP4.to_lbc_quant(), None);
        assert_eq!(GgmlType::Unknown(9999).to_lbc_quant(), None);
    }

    // -- Metadata accessors --------------------------------------------------

    #[test]
    fn metadata_accessors_return_none_for_wrong_type() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        write_test_string(&mut buf, "test.num");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&42u32.to_le_bytes());

        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();

        assert_eq!(file.get_u32("test.num"), Some(42));
        assert_eq!(file.get_string("test.num"), None);
        assert_eq!(file.get_f32("test.num"), None);
        // get_u64 now coerces U32 -> u64, so this returns Some(42)
        assert_eq!(file.get_u64("test.num"), Some(42));
        assert_eq!(file.get_u32("nonexistent"), None);
    }

    #[test]
    fn get_metadata_returns_ref() {
        let data = make_minimal_gguf();
        let file = GgufFile::parse(&mut data.as_slice()).unwrap();
        let val = file.get_metadata("general.architecture").unwrap();
        match val {
            GgufValue::String(s) => assert_eq!(s, "llama"),
            other => panic!("expected String, got {other:?}"),
        }
        assert!(file.get_metadata("nonexistent").is_none());
    }

    #[test]
    fn find_tensor_missing() {
        let data = make_minimal_gguf();
        let file = GgufFile::parse(&mut data.as_slice()).unwrap();
        assert!(file.find_tensor("nonexistent").is_none());
    }

    // -- Error tests ---------------------------------------------------------

    #[test]
    fn error_invalid_magic() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGML");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let err = GgufFile::parse(&mut buf.as_slice()).unwrap_err();
        match err {
            GgufError::InvalidMagic(m) => assert_eq!(&m, b"GGML"),
            other => panic!("expected InvalidMagic, got {other}"),
        }
    }

    #[test]
    fn error_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let err = GgufFile::parse(&mut buf.as_slice()).unwrap_err();
        match err {
            GgufError::UnsupportedVersion(v) => assert_eq!(v, 1),
            other => panic!("expected UnsupportedVersion, got {other}"),
        }
    }

    #[test]
    fn error_truncated_file() {
        let buf = b"GGUF";
        let err = GgufFile::parse(&mut buf.as_slice()).unwrap_err();
        match err {
            GgufError::TruncatedFile => {}
            other => panic!("expected TruncatedFile, got {other}"),
        }
    }

    #[test]
    fn error_invalid_value_type() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        write_test_string(&mut buf, "bad.type");
        buf.extend_from_slice(&99u32.to_le_bytes());

        let err = GgufFile::parse(&mut buf.as_slice()).unwrap_err();
        match err {
            GgufError::InvalidValueType(t) => assert_eq!(t, 99),
            other => panic!("expected InvalidValueType, got {other}"),
        }
    }

    #[test]
    fn error_display_and_source() {
        let err = GgufError::InvalidMagic(*b"GGML");
        let msg = format!("{err}");
        assert!(msg.contains("invalid GGUF magic"));

        let io_err = GgufError::Io(std::io::Error::new(std::io::ErrorKind::Other, "test"));
        assert!(std::error::Error::source(&io_err).is_some());
        assert!(std::error::Error::source(&GgufError::TruncatedFile).is_none());
    }

    #[test]
    fn error_is_std_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<GgufError>();
    }

    // -- Alignment override --------------------------------------------------

    #[test]
    fn custom_alignment() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());

        write_test_string(&mut buf, "general.alignment");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&64u32.to_le_bytes());

        pad_to_alignment(&mut buf, 64);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        assert_eq!(file.alignment, 64);
        assert_eq!(file.data_offset % 64, 0);
    }

    // -- Quantized tensor byte size ------------------------------------------

    #[test]
    fn quantized_tensor_byte_sizes() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        write_test_string(&mut buf, "q4_weight");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&4096u64.to_le_bytes());
        buf.extend_from_slice(&4096u64.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        let t = file.find_tensor("q4_weight").unwrap();
        assert_eq!(t.ggml_type, GgmlType::Q4_0);
        assert_eq!(t.n_elements(), 4096 * 4096);
        assert_eq!(t.byte_size(), Some(9_437_184));
    }

    // -- Builder tests -------------------------------------------------------

    #[test]
    fn builder_roundtrip() {
        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "llama");
        builder.add_u32("llama.block_count", 2);
        builder.add_u32("llama.attention.head_count", 4);
        builder.add_u32("llama.embedding_length", 32);

        let data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        builder.add_f32_tensor("token_embd.weight", &[8, 1], &data);

        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        assert_eq!(file.version, 3);
        assert_eq!(file.get_string("general.architecture"), Some("llama"));
        assert_eq!(file.get_u32("llama.block_count"), Some(2));
        assert_eq!(file.tensors.len(), 1);
        assert_eq!(file.tensors[0].name, "token_embd.weight");
        assert_eq!(file.tensors[0].n_elements(), 8);
        assert_eq!(file.tensors[0].byte_size(), Some(32));
    }

    #[test]
    fn builder_empty_file() {
        let builder = GgufBuilder::new();
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        assert_eq!(file.version, 3);
        assert!(file.metadata.is_empty());
        assert!(file.tensors.is_empty());
    }

    #[test]
    fn builder_multiple_tensors_with_data_readback() {
        let mut builder = GgufBuilder::new();
        builder.add_string("general.architecture", "llama");

        let data1: Vec<f32> = vec![1.0; 16];
        let data2: Vec<f32> = vec![2.0; 32];
        builder.add_f32_tensor("tensor_a", &[16], &data1);
        builder.add_f32_tensor("tensor_b", &[32], &data2);

        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        assert_eq!(file.tensors.len(), 2);
        assert_eq!(file.find_tensor("tensor_a").unwrap().byte_size(), Some(64));
        assert_eq!(file.find_tensor("tensor_b").unwrap().byte_size(), Some(128));

        let ta = file.find_tensor("tensor_a").unwrap();
        let offset = file.tensor_data_offset(ta) as usize;
        let end = offset + ta.byte_size().unwrap() as usize;
        let read_data: Vec<f32> = bytes[offset..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(read_data, data1);
    }

    #[test]
    fn builder_string_array() {
        let mut builder = GgufBuilder::new();
        builder.add_string_array("tokenizer.tokens", &["hello", "world", "foo"]);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        match file.get_metadata("tokenizer.tokens").unwrap() {
            GgufValue::Array(arr) => {
                assert_eq!(arr.values.len(), 3);
                match &arr.values[0] {
                    GgufValue::String(s) => assert_eq!(s, "hello"),
                    other => panic!("expected String, got {other:?}"),
                }
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[test]
    fn builder_f32_metadata() {
        let mut builder = GgufBuilder::new();
        builder.add_f32("test.temperature", 0.7);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        assert_eq!(file.get_f32("test.temperature"), Some(0.7));
    }

    // -- Metadata type coercion tests ----------------------------------------

    #[test]
    fn get_u32_coerces_from_u64() {
        let mut builder = GgufBuilder::new();
        builder.add_u64("fits", 42);
        builder.add_u64("too_big", u64::from(u32::MAX) + 1);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        assert_eq!(file.get_u32("fits"), Some(42));
        assert_eq!(file.get_u32("too_big"), None); // does not fit in u32
    }

    #[test]
    fn get_u32_coerces_from_i32() {
        let mut builder = GgufBuilder::new();
        builder.add_i32("positive", 100);
        builder.add_i32("negative", -1);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        assert_eq!(file.get_u32("positive"), Some(100));
        assert_eq!(file.get_u32("negative"), None); // negative can't be u32
    }

    #[test]
    fn get_u64_coerces_from_u32() {
        let mut builder = GgufBuilder::new();
        builder.add_u32("small", 99);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        assert_eq!(file.get_u64("small"), Some(99));
    }

    #[test]
    fn get_u64_coerces_from_i32() {
        let mut builder = GgufBuilder::new();
        builder.add_i32("pos", 500);
        builder.add_i32("neg", -10);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        assert_eq!(file.get_u64("pos"), Some(500));
        assert_eq!(file.get_u64("neg"), None); // negative can't be u64
    }

    #[test]
    fn get_u32_no_coercion_from_string() {
        let mut builder = GgufBuilder::new();
        builder.add_string("key", "42");
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        assert_eq!(file.get_u32("key"), None); // string should not coerce
    }

    // -- Debug impl ----------------------------------------------------------

    #[test]
    fn gguf_file_debug() {
        let data = make_minimal_gguf();
        let file = GgufFile::parse(&mut data.as_slice()).unwrap();
        let dbg = format!("{file:?}");
        assert!(dbg.contains("GgufFile"));
        assert!(dbg.contains("version: 3"));
    }

    // -- Metadata order preservation -----------------------------------------

    #[test]
    fn metadata_preserves_insertion_order() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());

        for key in &["zebra", "alpha", "middle"] {
            write_test_string(&mut buf, key);
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
        }

        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        let keys: Vec<&str> = file.metadata.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["zebra", "alpha", "middle"]);
    }

    // -- align_up utility ----------------------------------------------------

    #[test]
    fn align_up_cases() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(100, 64), 128);
        assert_eq!(align_up(0, 0), 0);
    }

    // -- GGUF v2 parsing tests -----------------------------------------------

    /// Helper: write a GGUF v2 string (u32 length prefix).
    fn write_test_string_v2(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn make_minimal_gguf_v2() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&2u32.to_le_bytes());
        // v2: n_tensors and n_kv as u32
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_kv

        // metadata: general.architecture = "llama" (v2 string = u32 len)
        write_test_string_v2(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // type STRING
        write_test_string_v2(&mut buf, "llama");

        // tensor info
        write_test_string_v2(&mut buf, "token_embd.weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&32u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&8u64.to_le_bytes());  // dim 1
        buf.extend_from_slice(&0u32.to_le_bytes());  // ggml_type = F32
        buf.extend_from_slice(&0u64.to_le_bytes());  // offset

        pad_to_alignment(&mut buf, 32);
        buf.extend_from_slice(&vec![0u8; 1024]);
        buf
    }

    #[test]
    fn parse_v2_minimal() {
        let data = make_minimal_gguf_v2();
        let file = GgufFile::parse(&mut data.as_slice()).unwrap();

        assert_eq!(file.version, 2);
        assert_eq!(file.alignment, 32);
        assert_eq!(file.metadata.len(), 1);
        assert_eq!(file.tensors.len(), 1);
        assert_eq!(file.get_string("general.architecture"), Some("llama"));

        let t = file.find_tensor("token_embd.weight").unwrap();
        assert_eq!(t.dims, vec![32, 8]);
        assert_eq!(t.ggml_type, GgmlType::F32);
        assert_eq!(t.offset, 0);
        assert_eq!(t.n_elements(), 256);
        assert_eq!(t.byte_size(), Some(1024));
        assert_eq!(file.data_offset % 32, 0);
    }

    #[test]
    fn parse_v2_empty_file() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&0u32.to_le_bytes()); // n_kv
        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        assert_eq!(file.version, 2);
        assert!(file.metadata.is_empty());
        assert!(file.tensors.is_empty());
    }

    #[test]
    fn parse_v2_array_of_strings() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_kv

        write_test_string_v2(&mut buf, "test.strings");
        buf.extend_from_slice(&9u32.to_le_bytes()); // type ARRAY
        buf.extend_from_slice(&8u32.to_le_bytes()); // element_type STRING
        buf.extend_from_slice(&2u32.to_le_bytes()); // count (v2 = u32)
        write_test_string_v2(&mut buf, "alpha");
        write_test_string_v2(&mut buf, "beta");

        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        match file.get_metadata("test.strings").unwrap() {
            GgufValue::Array(arr) => {
                assert_eq!(arr.element_type, 8);
                assert_eq!(arr.values.len(), 2);
                match &arr.values[0] {
                    GgufValue::String(s) => assert_eq!(s, "alpha"),
                    other => panic!("expected String, got {other:?}"),
                }
                match &arr.values[1] {
                    GgufValue::String(s) => assert_eq!(s, "beta"),
                    other => panic!("expected String, got {other:?}"),
                }
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    // -- v2 builder roundtrip ------------------------------------------------

    #[test]
    fn builder_v2_roundtrip() {
        let mut builder = GgufBuilder::new();
        builder.version(2);
        builder.add_string("general.architecture", "llama");
        builder.add_u32("llama.block_count", 2);
        builder.add_string_array("tokenizer.tokens", &["hello", "world"]);

        let data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        builder.add_f32_tensor("token_embd.weight", &[8, 1], &data);

        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        assert_eq!(file.version, 2);
        assert_eq!(file.get_string("general.architecture"), Some("llama"));
        assert_eq!(file.get_u32("llama.block_count"), Some(2));
        assert_eq!(file.tensors.len(), 1);
        assert_eq!(file.tensors[0].name, "token_embd.weight");
        assert_eq!(file.tensors[0].n_elements(), 8);
        assert_eq!(file.tensors[0].byte_size(), Some(32));

        // Verify tensor data can be read back correctly.
        let t = file.find_tensor("token_embd.weight").unwrap();
        let offset = file.tensor_data_offset(t) as usize;
        let end = offset + t.byte_size().unwrap() as usize;
        let read_data: Vec<f32> = bytes[offset..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(read_data, data);

        // Verify string array roundtrips through v2.
        let tokens = file.get_string_array("tokenizer.tokens").unwrap();
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn builder_v2_empty_file() {
        let mut builder = GgufBuilder::new();
        builder.version(2);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        assert_eq!(file.version, 2);
        assert!(file.metadata.is_empty());
        assert!(file.tensors.is_empty());
    }

    // -- ResourceLimit tests -------------------------------------------------

    #[test]
    fn error_resource_limit_n_tensors() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(MAX_TENSOR_ENTRIES + 1).to_le_bytes()); // n_tensors
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_kv

        let err = GgufFile::parse(&mut buf.as_slice()).unwrap_err();
        match err {
            GgufError::ResourceLimit(msg) => assert!(msg.contains("tensor count")),
            other => panic!("expected ResourceLimit, got {other}"),
        }
    }

    #[test]
    fn error_resource_limit_n_kv() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&(MAX_METADATA_ENTRIES + 1).to_le_bytes()); // n_kv

        let err = GgufFile::parse(&mut buf.as_slice()).unwrap_err();
        match err {
            GgufError::ResourceLimit(msg) => assert!(msg.contains("metadata entry count")),
            other => panic!("expected ResourceLimit, got {other}"),
        }
    }

    #[test]
    fn error_resource_limit_string_length() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv = 1

        // Write a key string with an absurd length prefix.
        buf.extend_from_slice(&(MAX_STRING_LENGTH + 1).to_le_bytes());

        let err = GgufFile::parse(&mut buf.as_slice()).unwrap_err();
        match err {
            GgufError::ResourceLimit(msg) => assert!(msg.contains("string length")),
            other => panic!("expected ResourceLimit, got {other}"),
        }
    }

    #[test]
    fn error_resource_limit_array_count() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv = 1

        // Write a valid key string.
        write_test_string(&mut buf, "test.array");
        // Value type = ARRAY (9)
        buf.extend_from_slice(&9u32.to_le_bytes());
        // Element type = U32 (4)
        buf.extend_from_slice(&4u32.to_le_bytes());
        // Absurd count
        buf.extend_from_slice(&(MAX_ARRAY_ELEMENTS + 1).to_le_bytes());

        let err = GgufFile::parse(&mut buf.as_slice()).unwrap_err();
        match err {
            GgufError::ResourceLimit(msg) => assert!(msg.contains("array element count")),
            other => panic!("expected ResourceLimit, got {other}"),
        }
    }

    #[test]
    fn error_resource_limit_n_dims() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors = 1
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_kv = 0

        // Tensor info with absurd n_dims.
        write_test_string(&mut buf, "bad_tensor");
        buf.extend_from_slice(&(MAX_TENSOR_DIMS + 1).to_le_bytes()); // n_dims

        let err = GgufFile::parse(&mut buf.as_slice()).unwrap_err();
        match err {
            GgufError::ResourceLimit(msg) => assert!(msg.contains("tensor dimension count")),
            other => panic!("expected ResourceLimit, got {other}"),
        }
    }

    #[test]
    fn error_resource_limit_display() {
        let err = GgufError::ResourceLimit("test message".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("resource limit exceeded"));
        assert!(msg.contains("test message"));
        assert!(std::error::Error::source(&err).is_none());
    }

    // -- f32 array accessor tests --------------------------------------------

    #[test]
    fn get_f32_array_basic() {
        let mut builder = GgufBuilder::new();
        builder.add_f32_array("test.scores", &[1.0, 2.5, -0.3]);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        let arr = file.get_f32_array("test.scores").unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], 1.0);
        assert_eq!(arr[1], 2.5);
        assert_eq!(arr[2], -0.3);
    }

    #[test]
    fn get_f32_array_empty() {
        let mut builder = GgufBuilder::new();
        builder.add_f32_array("test.empty", &[]);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        let arr = file.get_f32_array("test.empty").unwrap();
        assert!(arr.is_empty());
    }

    #[test]
    fn get_f32_array_missing_key() {
        let builder = GgufBuilder::new();
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        assert!(file.get_f32_array("nonexistent").is_none());
    }

    #[test]
    fn get_f32_array_wrong_type() {
        let mut builder = GgufBuilder::new();
        builder.add_u32("test.not_array", 42);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        assert!(file.get_f32_array("test.not_array").is_none());
    }

    #[test]
    fn get_f32_array_wrong_element_type() {
        let mut builder = GgufBuilder::new();
        builder.add_string_array("test.strings", &["a", "b"]);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        assert!(file.get_f32_array("test.strings").is_none());
    }

    // -- u32 array accessor tests --------------------------------------------

    #[test]
    fn get_u32_array_basic() {
        let mut builder = GgufBuilder::new();
        builder.add_u32_array("test.types", &[1, 2, 3, 1]);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        let arr = file.get_u32_array("test.types").unwrap();
        assert_eq!(arr, vec![1, 2, 3, 1]);
    }

    #[test]
    fn get_u32_array_empty() {
        let mut builder = GgufBuilder::new();
        builder.add_u32_array("test.empty", &[]);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        let arr = file.get_u32_array("test.empty").unwrap();
        assert!(arr.is_empty());
    }

    #[test]
    fn get_u32_array_missing_key() {
        let builder = GgufBuilder::new();
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        assert!(file.get_u32_array("nonexistent").is_none());
    }

    #[test]
    fn get_u32_array_wrong_type() {
        let mut builder = GgufBuilder::new();
        builder.add_string("test.str", "hello");
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        assert!(file.get_u32_array("test.str").is_none());
    }

    #[test]
    fn get_u32_array_i32_coercion() {
        // Build a GGUF with an I32 array and verify get_u32_array coerces it.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv

        write_test_string(&mut buf, "test.i32_array");
        buf.extend_from_slice(&9u32.to_le_bytes()); // Array type tag
        buf.extend_from_slice(&5u32.to_le_bytes()); // I32 element type
        buf.extend_from_slice(&3u64.to_le_bytes()); // 3 elements
        buf.extend_from_slice(&0i32.to_le_bytes());
        buf.extend_from_slice(&1i32.to_le_bytes());
        buf.extend_from_slice(&3i32.to_le_bytes());

        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        let arr = file.get_u32_array("test.i32_array").unwrap();
        assert_eq!(arr, vec![0, 1, 3]);
    }

    #[test]
    fn get_u32_array_negative_i32_returns_none() {
        // Negative I32 values cannot convert to u32.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv

        write_test_string(&mut buf, "test.neg_i32");
        buf.extend_from_slice(&9u32.to_le_bytes()); // Array type tag
        buf.extend_from_slice(&5u32.to_le_bytes()); // I32 element type
        buf.extend_from_slice(&2u64.to_le_bytes()); // 2 elements
        buf.extend_from_slice(&1i32.to_le_bytes());
        buf.extend_from_slice(&(-1i32).to_le_bytes());

        pad_to_alignment(&mut buf, 32);

        let file = GgufFile::parse(&mut buf.as_slice()).unwrap();
        assert!(file.get_u32_array("test.neg_i32").is_none());
    }

    // -- Builder bool/f32_array/u32_array roundtrip tests --------------------

    #[test]
    fn builder_bool_roundtrip() {
        let mut builder = GgufBuilder::new();
        builder.add_bool("test.true", true);
        builder.add_bool("test.false", false);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        match file.get_metadata("test.true").unwrap() {
            GgufValue::Bool(v) => assert!(*v),
            other => panic!("expected Bool(true), got {other:?}"),
        }
        match file.get_metadata("test.false").unwrap() {
            GgufValue::Bool(v) => assert!(!v),
            other => panic!("expected Bool(false), got {other:?}"),
        }
    }

    #[test]
    fn builder_f32_array_roundtrip() {
        let mut builder = GgufBuilder::new();
        builder.add_f32_array("test.scores", &[0.5, 1.0, -2.0]);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        let arr = file.get_f32_array("test.scores").unwrap();
        assert_eq!(arr, vec![0.5, 1.0, -2.0]);
    }

    #[test]
    fn builder_u32_array_roundtrip() {
        let mut builder = GgufBuilder::new();
        builder.add_u32_array("test.types", &[1, 3, 6]);
        let bytes = builder.build();
        let file = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        let arr = file.get_u32_array("test.types").unwrap();
        assert_eq!(arr, vec![1, 3, 6]);
    }
}
