//! # lumen-image
//!
//! Types and utilities for `.lbi`, the container that holds an image model's
//! weights (transformer, autoencoder, text encoder).
//!
//! Unlike LBC, which addresses a fixed set of LLM slot names and derives shapes
//! from the model hyperparameters, `.lbi` stores each tensor's name and shape
//! with its bytes, so convolution stacks and decoder blocks can be represented.
//!
//! Shard identification uses device and inode, so the crate builds on unix
//! (macOS and Linux), which is what the workspace targets.

pub mod convert;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dit;
pub mod lbi;
pub mod npy;
pub mod pipeline;
pub mod png;
pub mod safetensors;
pub mod scheduler;
pub mod shard_index;
pub mod tensor;
pub mod text_encoder;
pub mod tokenizer;
pub mod vae;

pub use convert::{
    convert_component, convert_single_file, single_shard, ConvertError, ConvertReport,
};
pub use lbi::{half_to_f32, LbiError, LbiFile, LbiWriter, TensorEntry};
pub use safetensors::{SafetensorsError, SafetensorsFile};
pub use scheduler::{calculate_shift, SchedulerConfig, SigmaSchedule};
pub use shard_index::ShardIndex;
pub use text_encoder::{image_pad_mask, TextEncoder, TextEncoderConfig, TextEncoderError};
