//! Runtime error types.

use lumen_format::FormatError;

/// Errors that can occur during inference runtime operations.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("format error: {0}")]
    Format(#[from] FormatError),

    #[error("layer {layer} not available: {reason}")]
    LayerUnavailable { layer: usize, reason: String },

    #[error("compute error: {0}")]
    Compute(String),

    #[error("storage I/O error: {0}")]
    StorageIo(#[from] std::io::Error),

    #[error("KV cache error: {0}")]
    KvCache(String),

    #[error("pipeline error: {0}")]
    Pipeline(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("unsupported: {0}")]
    Unsupported(String),
}
