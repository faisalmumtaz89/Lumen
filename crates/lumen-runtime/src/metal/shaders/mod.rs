//! Metal Shading Language (MSL) kernel source code.
//!
//! All GPU compute kernels are compiled at runtime from this source string
//! via `MTLDevice.newLibraryWithSource()`. Keeping shaders as a Rust string
//! constant avoids file I/O at init time and keeps the single-binary deployment
//! model intact.
//!
//! Kernels are hyper-optimized for Apple Silicon M-series:
//! - SIMD group reductions (simd_sum / simd_max) for fast parallel sums/max
//! - Threadgroup memory tiling for input vector reuse across output rows
//! - 32-wide SIMD groups (Apple GPU architecture)
//! - Fused operations where profitable
//!
//! Compatibility note: `thread_index_in_simdgroup` and `simdgroup_index_in_threadgroup`
//! are passed as kernel function arguments with [[attribute]] syntax for broad
//! Metal version compatibility.

#[cfg(target_os = "macos")]
pub const METAL_SHADER_SOURCE: &str = concat!(
    include_str!("common.msl"),
    include_str!("matmul_f32.msl"),
    include_str!("matmul_f16.msl"),
    include_str!("norm.msl"),
    include_str!("matmul_q8_0.msl"),
    include_str!("ffn_fused_norm.msl"),
    include_str!("matmul_q8_0_alt.msl"),
    include_str!("rope_activation.msl"),
    include_str!("attention.msl"),
    include_str!("embed.msl"),
    include_str!("gemm_f32.msl"),
    include_str!("gemm_q8_0.msl"),
    include_str!("batched_ops.msl"),
    include_str!("gemm_residual_f16.msl"),
    include_str!("matmul_q4.msl"),
    include_str!("gemm_q4.msl"),
    include_str!("ffn_elementwise.msl"),
    include_str!("moe.msl"),
    include_str!("gdn_core.msl"),
    include_str!("gdn_advanced.msl"),
);
