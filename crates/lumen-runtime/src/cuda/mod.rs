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

mod backend_impl;
pub(crate) mod decode;
/// CUDA device wrapper (context, stream, buffer management).
pub mod ffi;
pub(crate) mod gdn;
pub(crate) mod gpu_buffers;
pub(crate) mod kv_cache;
/// CUDA MoE forward-path types.
pub(crate) mod moe;
pub(crate) mod prefill;
pub(crate) mod prefill_attention;
pub(crate) mod profiler;
/// Persistent on-disk cache for NVRTC-compiled kernel PTX (cold-start fix).
pub(crate) mod ptx_cache;
/// Embedded CUDA kernel source strings, compiled to PTX at runtime via NVRTC.
pub mod shaders;
pub(crate) mod types;

pub use backend_impl::CudaBackend;

/// The decode-attention launch geometry, for out-of-crate callers that drive
/// the kernels in [`shaders`] directly — the correctness suite
/// (`tests/cuda_attention_splitk_gqa6_test.rs`) and the standalone A/B
/// harness (`examples/attn_decode_ab.rs`).
///
/// Those callers cannot reach `prefill`, so without this they would hand-copy
/// the split counts, chunk length, block dimensions and shared-memory sizes
/// the launcher uses, and a retune here would leave their copy silently
/// describing a geometry production no longer runs. Each name below has one
/// of those two consumers; nothing outside the crate is expected to dispatch
/// attention.
pub use decode::{ATTN_DECODE_TILED_BLOCK_DIM, ATTN_DECODE_TILED_T_C};
pub use prefill::{
    attn_splitk_chunks, attn_splitk_gqa6_chunks, attn_splitk_gqa6_max_seq_len,
    attn_splitk_gqa6_merge_shared_bytes, attn_splitk_gqa6_partial_shared_bytes,
    attn_splitk_gqa6_partial_shared_bytes_f16, ATTN_SPLITK_GQA6_CHUNK, ATTN_SPLITK_GQA6_DIM_TILES,
    ATTN_SPLITK_GQA6_HEAD_DIM,
};

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
    bf16_gemmex_fallback_armed_for_tests, bf16_gemmex_runtime_warning_emitted_for_tests,
    inject_next_bf16_cublas_failure, reset_bf16_gemmex_state_for_tests,
};
