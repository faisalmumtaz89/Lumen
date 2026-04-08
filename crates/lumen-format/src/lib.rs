//! # lumen-format
//!
//! Types and utilities for the Layer-Blob Container (LBC) file format.
//!
//! LBC is a binary format for LLM inference. It stores weights in large
//! contiguous "layer blobs" with page-aligned offsets for efficient I/O,
//! supporting both GPU-resident loading and streaming layer-by-layer access.
//!
//! ## Layout
//!
//! ```text
//! [Header] [LayerIndex * L] [ExpertIndex?] [LayerBlob_0] [LayerBlob_1] ... [LayerBlob_{L-1}]
//! ```

pub mod crc;
pub mod error;
pub mod header;
pub mod hyperparams;
pub mod index;
pub mod large_model;
pub mod quantization;
pub mod reader;
pub(crate) mod rng;
pub mod streaming_writer;
pub mod test_model;
pub mod tokenizer;
pub mod writer;

pub use error::FormatError;
pub use header::{Endianness, GlobalTensorRange, LbcHeader, LBC_MAGIC, LBC_VERSION};
pub use hyperparams::ModelHyperparams;
pub use index::{ExpertSlice, LayerIndex, SubtensorOffsets, TensorSlice};
pub use large_model::{LargeModelConfig, generate_large_model, generate_large_model_f16};
pub use quantization::{QuantGroupSize, QuantScheme, QuantizationDescriptor};
pub use reader::LbcFile;
pub use streaming_writer::{LayerShape, StreamingLbcWriter};
pub use tokenizer::TokenizerSection;
pub use writer::GlobalTensors;
