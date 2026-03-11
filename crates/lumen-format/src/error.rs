//! Error types for LBC format parsing and writing.

#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("invalid magic bytes: expected {expected:#010x}, found {found:#010x}")]
    InvalidMagic { expected: u32, found: u32 },

    #[error("unsupported LBC version: {version} (max supported: {max_supported})")]
    UnsupportedVersion { version: u32, max_supported: u32 },

    #[error("header checksum mismatch: expected {expected:#010x}, computed {computed:#010x}")]
    ChecksumMismatch { expected: u32, computed: u32 },

    #[error("layer {layer} tensor {tensor_name}: offset {offset} + length {length} exceeds blob size {file_size}")]
    LayerOutOfBounds {
        layer: usize,
        tensor_name: &'static str,
        offset: u64,
        length: u64,
        file_size: u64,
    },

    #[error("unsupported quantization scheme: {0}")]
    UnsupportedQuantization(String),

    #[error("invalid endianness byte: {0}")]
    InvalidEndianness(u8),

    #[error("invalid RoPE scaling type: {0}")]
    InvalidRopeScalingType(u8),

    #[error("layer count mismatch: header says {header_count}, hyperparams say {hyperparams_count}")]
    LayerCountMismatch {
        header_count: u32,
        hyperparams_count: u32,
    },

    #[error("alignment violation at offset {offset}: required {required}, actual {actual}")]
    AlignmentViolation {
        offset: u64,
        required: u64,
        actual: u64,
    },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Header truncated or file too small.
    #[error("unexpected end of file: needed {needed} bytes, available {available}")]
    UnexpectedEof { needed: u64, available: u64 },
}
