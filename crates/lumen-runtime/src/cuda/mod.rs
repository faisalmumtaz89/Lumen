//! CUDA GPU compute backend for NVIDIA GPUs.
//!
//! Implements `ComputeBackend` using CUDA compute kernels compiled at runtime
//! via NVRTC. This avoids requiring `nvcc` at build time -- only a CUDA-capable
//! GPU and driver are needed at runtime.
//!
//! # Build requirements
//!
//! -**macOS (dev)**: `cargo check --features cuda` passes (cudarc with
//!   `fallback-dynamic-loading` compiles without CUDA SDK).
//! -**Linux (GPU)**: `cargo build --features cuda` requires CUDA 12.x driver.
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
/// CUDA MoE forward-path types.
pub(crate) mod moe;
mod backend_impl;

pub use backend_impl::CudaBackend;

// ---------------------------------------------------------------------------
// BF16 GemmEx fault-injection hooks.
//
// Re-exported only when the test-fault-injection feature is enabled (or
// in lib `cargo test` builds where `cfg(test)` applies). Production
// release builds without the feature have neither the helpers nor the
// underlying state -- everything compiles away.
//
// Consumed by the `cuda_bf16_gemmex_fault_injection_test` integration
// suite to drive the wrapper's per-call cuBLAS-failure -> legacy-kernel
// fall-through arm under a real BF16 matvec dispatch on Modal A100.
#[cfg(any(test, feature = "test-fault-injection"))]
pub use backend_impl::{
    bf16_gemmex_fallback_armed_for_tests,
    bf16_gemmex_runtime_warning_emitted_for_tests,
    inject_next_bf16_cublas_failure,
    reset_bf16_gemmex_state_for_tests,
};
