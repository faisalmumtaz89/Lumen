//! CUDA GPU compute backend for NVIDIA GPUs.
//!
//! Implements `ComputeBackend` using CUDA compute kernels compiled at runtime
//! via NVRTC. This avoids requiring `nvcc` at build time -- only a CUDA-capable
//! GPU and driver are needed at runtime.
//!
//! # Build requirements
//!
//! - **macOS (dev)**: `cargo check --features cuda` passes (cudarc with
//!   `fallback-dynamic-loading` compiles without CUDA SDK).
//! - **Linux (GPU)**: `cargo build --features cuda` requires CUDA 12.x driver.
//!
//! # Current status
//!
//! - `embed_token` (F32 + Q8_0): GPU kernel execution via NVRTC-compiled PTX.
//! - `compute_layer` (F32 + Q8_0): Full transformer layer decode (RMSNorm, QKV, RoPE,
//!   GQA attention, SwiGLU MLP, residual connections).
//! - `compute_final` (F32 + Q8_0): Final RMSNorm + output projection to logits.

/// CUDA device wrapper (context, stream, buffer management).
pub mod ffi;
/// Embedded CUDA kernel source strings, compiled to PTX at runtime via NVRTC.
pub mod shaders;
pub(crate) mod types;
pub(crate) mod gpu_buffers;
pub(crate) mod kv_cache;
pub(crate) mod decode;
pub(crate) mod gdn;
pub(crate) mod graph;
pub(crate) mod prefill;
pub(crate) mod prefill_attention;
mod backend_impl;

pub use backend_impl::CudaBackend;
