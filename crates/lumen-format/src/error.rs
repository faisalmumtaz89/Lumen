//! Error types for LBC format parsing and writing.

#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error(
        "not a valid LBC model file (bad magic: expected {expected:#010x}, found {found:#010x}). \
         If this is a GGUF file, convert it first with `lumen convert <file.gguf>`."
    )]
    InvalidMagic { expected: u32, found: u32 },

    #[error(
        "model file is LBC format v{version}, but this build of Lumen supports up to v{max_supported}. \
         The file was created by a newer Lumen build — update Lumen, or regenerate the model with this \
         build via `lumen pull <model>` (re-download) or `lumen convert <file.gguf>` (re-convert)."
    )]
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

    #[error(
        "layer count mismatch: header says {header_count}, hyperparams say {hyperparams_count}"
    )]
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

#[cfg(test)]
mod tests {
    use super::*;

    // The version/magic errors are the ones users hit with the wrong model file,
    // so their messages must stay actionable (tell the user what to run), not just
    // report numbers. Lock that in so a future edit can't quietly make them cryptic.
    #[test]
    fn unsupported_version_message_is_actionable() {
        let msg = FormatError::UnsupportedVersion {
            version: 5,
            max_supported: 4,
        }
        .to_string();
        assert!(
            msg.contains("v5") && msg.contains("v4"),
            "states the versions: {msg}"
        );
        assert!(
            msg.contains("lumen pull")
                || msg.contains("lumen convert")
                || msg.contains("update Lumen"),
            "tells the user what to do: {msg}"
        );
    }

    #[test]
    fn invalid_magic_message_hints_gguf() {
        let msg = FormatError::InvalidMagic {
            expected: 0x01_43_42_4C,
            found: 0xDEAD_BEEF,
        }
        .to_string();
        assert!(
            msg.contains("lumen convert"),
            "hints GGUF conversion: {msg}"
        );
    }
}
