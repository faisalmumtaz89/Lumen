//! CUDA kernel compilation and dispatch helpers for the decode path.
//!
//! Provides `KernelSet` (all compiled CUDA function handles) and helper
//! functions for launching individual kernels. The backend_impl module
//! orchestrates these into the full compute_layer/compute_final pipeline.

use crate::error::RuntimeError;
use super::ffi::CudaDevice;
use super::shaders;
use cudarc::driver::CudaFunction;

/// All compiled CUDA kernel functions needed for single-token decode.
///
/// Compiled once during `init()` via NVRTC and reused across all layers.
pub(crate) struct KernelSet {
    // Normalization
    pub(crate) rmsnorm: CudaFunction,
    pub(crate) rmsnorm_per_head: CudaFunction,

    // F32 matrix-vector multiply
    pub(crate) matvec_f32: CudaFunction,
    pub(crate) matvec_f32_residual: CudaFunction,

    // F16 matrix-vector multiply (custom NVRTC kernel, fallback for non-cuBLAS path)
    pub(crate) matvec_f16: CudaFunction,
    pub(crate) matvec_f16_residual: CudaFunction,

    // F32 <-> F16 conversion kernels (for cuBLAS HGEMM activation conversion)
    pub(crate) f32_to_f16_vec: CudaFunction,
    // Vectorized F32->F16: 4 elements/thread with float4 loads + uint2 stores.
    // Used in prefill GEMM where conversion sizes are large and aligned.
    pub(crate) f32_to_f16_vec4: Option<CudaFunction>,
    pub(crate) f16_to_f32_vec: CudaFunction,

    // Q8_0 matrix-vector multiply (on-the-fly dequantization)
    pub(crate) matvec_q8_0: CudaFunction,
    pub(crate) matvec_q8_0_residual: CudaFunction,

    // Q4_0 matrix-vector multiply (on-the-fly 4-bit dequantization)
    pub(crate) matvec_q4_0: CudaFunction,
    pub(crate) matvec_q4_0_residual: CudaFunction,

    // Rotary position embeddings
    pub(crate) rope_apply: CudaFunction,

    // Elementwise activations
    pub(crate) swiglu_inplace: CudaFunction,
    pub(crate) residual_add: CudaFunction,
    /// Fused residual add + copy: dst[i] = a[i] + b[i].
    /// Replaces residual_add (in-place) + memcpy_dtod in graph decode pipeline.
    /// Saves 1 graph node per layer (36 fewer nodes for 3B model).
    pub(crate) residual_add_copy: CudaFunction,

    // Multi-head attention
    pub(crate) attention_decode: CudaFunction,

    // Tiled GEMM for batched prefill
    pub(crate) gemm_f32: CudaFunction,
    pub(crate) gemm_f32_residual: CudaFunction,

    // Fused RMSNorm + MatVec (two-pass: rms_scale scalar + inline norm)
    pub(crate) compute_rms_scale: CudaFunction,
    pub(crate) fused_norm_matvec_f32: CudaFunction,
    pub(crate) fused_norm_dual_matvec_f32: CudaFunction,

    // Batched prefill kernels
    pub(crate) embed_batch_f32: CudaFunction,
    pub(crate) embed_batch_q8_0: CudaFunction,
    pub(crate) embed_batch_f16: CudaFunction,
    pub(crate) embed_batch_q4_0: CudaFunction,
    pub(crate) rmsnorm_batched: CudaFunction,
    pub(crate) rope_apply_batched: CudaFunction,
    pub(crate) kv_cache_write_batch: CudaFunction,
    pub(crate) swiglu_batched: CudaFunction,
    pub(crate) residual_add_batched: CudaFunction,
    pub(crate) extract_row: CudaFunction,
    pub(crate) scatter_row: CudaFunction,

    // Flash Attention v2 for causal prefill (online softmax)
    pub(crate) flash_attention_v2: CudaFunction,
    pub(crate) flash_attention_br4: CudaFunction,

    // Q8_0 dequantization for batched prefill GEMM (replaces per-row matvec fallback)
    pub(crate) dequant_q8_0_to_f16: CudaFunction,
    pub(crate) dequant_q8_0_to_f32: CudaFunction,
    pub(crate) dequant_q4_0_to_f16: CudaFunction,
    pub(crate) dequant_q4_0_to_f32: CudaFunction,

    // Q8_0 dp4a INT8 dot product kernels (native __dp4a(), SM 6.1+).
    // On-the-fly x quantization to int8 for full integer dot product pipeline.
    // Optional: compilation requires SM 6.1+; falls back to scalar matvec_q8_0 if unavailable.
    pub(crate) matvec_q8_0_dp4a: Option<CudaFunction>,
    pub(crate) matvec_q8_0_dp4a_residual: Option<CudaFunction>,

    // Q8_0 native warp-cooperative: scalar dequant+FMA, no x-quantization.
    // 2 warps per row, deferred reduction. Reads 1.0625 bytes/elem.
    pub(crate) matvec_q8_0_native: Option<CudaFunction>,
    pub(crate) matvec_q8_0_native_residual: Option<CudaFunction>,

    // Q8_0 aligned dp4a kernels: 36-byte blocks with 4-byte-aligned quant data.
    // Uses native int* loads (8 instructions) instead of byte-level packing (56 ops).
    // Optional: compilation requires SM 6.1+; dispatch requires Q8Aligned weight buffers.
    pub(crate) matvec_q8_0_aligned: Option<CudaFunction>,
    pub(crate) matvec_q8_0_aligned_residual: Option<CudaFunction>,

    // Repack Q8_0 from 34-byte to 36-byte aligned blocks (one-time during preload).
    pub(crate) repack_q8_0_to_aligned36: Option<CudaFunction>,

    // Q8_0 v4: dp4a + cooperative x quantization + K-tiling. NR=4 rows/block.
    // DISABLED: regressive (75% lane waste, 30% slower than v1). Compiled but not dispatched.
    // Kept for future redesign experiments. See tracker.md Wave C8.
    #[allow(dead_code)]
    pub(crate) matvec_q8_0_v4: Option<CudaFunction>,
    #[allow(dead_code)]
    pub(crate) matvec_q8_0_v4_residual: Option<CudaFunction>,

    // Q8_0 shared-memory matvec: x-vector cached in shmem, scalar dequant+FMA.
    // NR=2 rows/block, 256 threads. PRIMARY Q8_0 decode path — reads native Q8_0
    // (1.0625 B/elem) instead of HGEMV's pre-dequanted F16 (2 B/elem).
    // Requires dynamic shared memory = in_dim * 4 bytes.
    pub(crate) matvec_q8_0_smem: Option<CudaFunction>,
    pub(crate) matvec_q8_0_smem_residual: Option<CudaFunction>,

    // Q4_0 shared-memory matvec: x-vector cached in shmem, nibble unpack+FMA.
    // NR=2 rows/block, 256 threads. PRIMARY Q4_0 decode path — reads native Q4_0
    // (0.5625 B/elem) instead of HGEMV's pre-dequanted F16 (2 B/elem).
    pub(crate) matvec_q4_0_smem: Option<CudaFunction>,
    pub(crate) matvec_q4_0_smem_residual: Option<CudaFunction>,

    // GDN (GatedDeltaNet) kernels for Qwen3.5 hybrid layers.
    pub(crate) ssm_conv1d_decode: Option<CudaFunction>,
    pub(crate) gdn_compute_gates: Option<CudaFunction>,
    pub(crate) l2_normalize_heads: Option<CudaFunction>,
    pub(crate) gdn_state_update: Option<CudaFunction>,
    pub(crate) silu_inplace: Option<CudaFunction>,
    pub(crate) silu_elementwise_mul: Option<CudaFunction>,

    // GDN fused decode megakernels (8 launches -> 2 per GDN layer).
    // gdn_decode_megakernel: fuses conv1d+silu, gates, L2 norm, state update.
    // gdn_rmsnorm_silu_gate: fuses RMSNorm + SiLU(gate) * normed.
    pub(crate) gdn_decode_megakernel: Option<CudaFunction>,
    pub(crate) gdn_rmsnorm_silu_gate: Option<CudaFunction>,

    // GDN fused prefill kernels (eliminate per-token loop over decode kernels).
    // ssm_conv1d_silu_prefill: batched conv1d+SiLU across T tokens.
    // gdn_compute_gates_batched: batched gate computation for T * num_heads.
    // l2_normalize_qk_strided: batched L2 norm for Q and K across T tokens.
    // gdn_prefill_fused_v3: warp-parallel fused state update (4x unrolled).
    // gdn_prefill_norm_gate: batched RMSNorm + SiLU gate on raw output.
    pub(crate) ssm_conv1d_silu_prefill: Option<CudaFunction>,
    pub(crate) gdn_compute_gates_batched: Option<CudaFunction>,
    pub(crate) l2_normalize_qk_strided: Option<CudaFunction>,
    pub(crate) gdn_prefill_fused_v3: Option<CudaFunction>,
    pub(crate) gdn_prefill_norm_gate: Option<CudaFunction>,

    // Tensor-core Flash Attention (WMMA via inline PTX, SM 80+).
    // 16x16 query tiles via mma.sync.aligned.m16n8k16 for QK^T and PV.
    // Optional: compilation requires SM 8.0+; falls back to scalar flash_attention_br4 if unavailable.
    pub(crate) flash_attention_wmma: Option<CudaFunction>,

    // GPU-side argmax: finds index of max value in logits buffer.
    // Single block of 1024 threads, reads back 4 bytes instead of vocab_size*4.
    pub(crate) argmax_f32: CudaFunction,

    // Fused F16 decode kernels (dispatch count reduction for F16 HGEMV path).
    // fused_rmsnorm_f16: RMSNorm + F32->F16 output in one kernel.
    // Replaces rmsnorm + f32_to_f16_vec at 2 sites/layer (attn_norm, ffn_norm).
    pub(crate) fused_rmsnorm_f16: Option<CudaFunction>,
    // swiglu_f32_to_f16: SwiGLU activation + F32->F16 output in one kernel.
    // Replaces swiglu_inplace + f32_to_f16_vec at 1 site/layer (FFN).
    pub(crate) swiglu_f32_to_f16: Option<CudaFunction>,
    // fused_residual_rmsnorm_f16: Residual add + RMSNorm + F16 output in one kernel.
    // Fuses end of layer L (residual add) with start of layer L+1 (attn RMSNorm).
    // Saves 1 dispatch per inter-layer boundary (35 fewer for 36-layer models).
    pub(crate) fused_residual_rmsnorm_f16: Option<CudaFunction>,
    // fused_residual_rmsnorm_f32: Residual add + RMSNorm (F32 output) in one kernel.
    // For Q8_0/Q4_0 inter-layer: fuses residual_add_copy + rmsnorm.
    // Saves 1 dispatch per inter-layer boundary (47 fewer for 48-layer models).
    pub(crate) fused_residual_rmsnorm_f32: Option<CudaFunction>,
    // fused_residual_rms_scale: Residual add + compute_rms_scale (scalar output).
    // For fused_glu_gemv inter-layer: fuses residual_add_copy + compute_rms_scale.
    // Saves 1 dispatch per inter-layer boundary.
    pub(crate) fused_residual_rms_scale: Option<CudaFunction>,

    // Fused F16 prefill kernels (dispatch count reduction for F16 HGEMM path).
    // fused_rmsnorm_f16_batched: Batched RMSNorm + F32->F16 output.
    // Replaces rmsnorm_batched + f32_to_f16_vec at 2 sites/layer (attn_norm, ffn_norm).
    // Saves 64 dispatches per prefill on 32-layer models.
    pub(crate) fused_rmsnorm_f16_batched: Option<CudaFunction>,
    // swiglu_f32_to_f16_batched: Batched SwiGLU + F32->F16 output.
    // Replaces swiglu_batched + f32_to_f16_vec at 1 site/layer (FFN).
    // Saves 32 dispatches per prefill on 32-layer models.
    pub(crate) swiglu_f32_to_f16_batched: Option<CudaFunction>,

    // Q8_0 dequant-in-register HGEMV: F16 x-vector in shmem, NR=4.
    // Covers in_dim up to 24576 (shmem = in_dim * 2 bytes vs smem kernel's * 4).
    // Primary Q8_0 path for 12288 < in_dim <= 24576 (e.g. FFN down on 8B+).
    pub(crate) hgemv_q8_0: Option<CudaFunction>,
    pub(crate) hgemv_q8_0_residual: Option<CudaFunction>,

    // Q4_0 dequant-in-register HGEMV: F16 x-vector in shmem, NR=4.
    // Covers in_dim up to 24576 (shmem = in_dim * 2 bytes).
    // Primary Q4_0 path for 12288 < in_dim <= 24576.
    pub(crate) hgemv_q4_0: Option<CudaFunction>,
    pub(crate) hgemv_q4_0_residual: Option<CudaFunction>,

    // dp4a with pre-quantized Q8_1 input.
    // quantize_f32_to_q8_1: pre-quantize x once per token.
    // matvec_q8_0_q8_1: dp4a dot product with native int* input loads.
    // NR=2 rows/block, 128 threads, NO shmem for input. SM 6.1+.
    pub(crate) quantize_f32_to_q8_1: Option<CudaFunction>,
    pub(crate) matvec_q8_0_q8_1: Option<CudaFunction>,
    pub(crate) matvec_q8_0_q8_1_residual: Option<CudaFunction>,

    // Q4_0 dp4a kernels: native Q4_0 weights + pre-quantized Q8_1 input.
    // Unpacks nibbles to sequential int8 for dp4a, includes zero-point correction.
    // NR=4 rows/block, 256 threads. SM 6.1+.
    pub(crate) matvec_q4_0_dp4a: Option<CudaFunction>,
    pub(crate) matvec_q4_0_dp4a_residual: Option<CudaFunction>,

    // Q4Aligned + Q8_1 input dp4a kernels (NR=4).
    // Combines aligned int* nibble loads (20-byte blocks) with pre-quantized Q8_1 input.
    // 4 int* loads vs 16 byte loads per block. Dispatch priority: this > matvec_q4_0_dp4a.
    pub(crate) matvec_q4_aligned_q8_1: Option<CudaFunction>,
    pub(crate) matvec_q4_aligned_q8_1_residual: Option<CudaFunction>,

    // Repack Q4_0 from 18-byte to 20-byte aligned blocks (one-time during preload).
    pub(crate) repack_q4_0_to_aligned20: Option<CudaFunction>,

    // Optimal Q8Aligned + Q8_1 input dp4a kernels (NR=2).
    // Combines aligned int* weight loads with pre-quantized Q8_1 input.
    // Both sides use native int* loads (zero byte-packing), dp4a, NR=2.
    // Dispatch priority: this > matvec_q8_0_aligned (on-the-fly quant).
    pub(crate) matvec_q8_aligned_q8_1: Option<CudaFunction>,
    pub(crate) matvec_q8_aligned_q8_1_residual: Option<CudaFunction>,

    // Fused gate+up+SwiGLU GEMV with inline RMSNorm for single-token decode.
    // Reads the input vector ONCE, computes gate and up projections simultaneously,
    // and applies SwiGLU inline. Eliminates 2-4 kernel launches per layer.
    //
    // Q8_0 variant (F32 x in shmem): hidden_dim * 4 <= 48KB -> hidden_dim <= 12288.
    pub(crate) fused_glu_gemv_q8_0: Option<CudaFunction>,
    // Q4_0 variant (F32 x in shmem): hidden_dim * 4 <= 48KB -> hidden_dim <= 12288.
    pub(crate) fused_glu_gemv_q4_0: Option<CudaFunction>,
    // F16 variant (F32 x in shmem): hidden_dim * 4 <= 48KB -> hidden_dim <= 12288.
    pub(crate) fused_glu_gemv_f16: Option<CudaFunction>,
    // Q8_0 variant with F16 x in shmem (large dims): hidden_dim * 2 <= 48KB -> hidden_dim <= 24576.
    pub(crate) fused_glu_gemv_q8_0_hg: Option<CudaFunction>,
    // Q4_0 variant with F16 x in shmem (large dims): hidden_dim * 2 <= 48KB -> hidden_dim <= 24576.
    pub(crate) fused_glu_gemv_q4_0_hg: Option<CudaFunction>,
    // Q8Aligned variant (F32 x in shmem): 36-byte blocks, hidden_dim * 4 <= 48KB -> hidden_dim <= 12288.
    pub(crate) fused_glu_gemv_q8_aligned: Option<CudaFunction>,
    // Q8Aligned variant with F16 x in shmem (large dims): hidden_dim * 2 <= 48KB -> hidden_dim <= 24576.
    pub(crate) fused_glu_gemv_q8_aligned_hg: Option<CudaFunction>,

    // Fused down projection: inline F32->Q8_1 quantize + dp4a matvec.
    // Eliminates separate quantize_f32_to_q8_1 dispatch for down projection.
    // matvec_q8_aligned_f32: reads F32 (from fused_glu output), quantizes inline, dp4a.
    pub(crate) matvec_q8_aligned_f32: Option<CudaFunction>,
    pub(crate) matvec_q8_aligned_f32_residual: Option<CudaFunction>,
    // matvec_q8_aligned_f32_swiglu: fuses SwiGLU + quantize + dp4a (3 dispatches -> 1).
    pub(crate) matvec_q8_aligned_f32_swiglu: Option<CudaFunction>,
    pub(crate) matvec_q8_aligned_f32_swiglu_residual: Option<CudaFunction>,

    // Fused down projection for Q4Aligned: inline F32->Q8_1 quantize + dp4a.
    // Same approach as Q8Aligned fused down but for Q4Aligned weights (20-byte blocks,
    // __byte_perm nibble unpack, zero-point correction).
    // matvec_q4_aligned_f32: reads F32, quantizes inline, dp4a against Q4Aligned.
    pub(crate) matvec_q4_aligned_f32: Option<CudaFunction>,
    pub(crate) matvec_q4_aligned_f32_residual: Option<CudaFunction>,
    // matvec_q4_aligned_f32_swiglu: fuses SwiGLU + quantize + dp4a (3 dispatches -> 1).
    pub(crate) matvec_q4_aligned_f32_swiglu: Option<CudaFunction>,
    pub(crate) matvec_q4_aligned_f32_swiglu_residual: Option<CudaFunction>,

    // Fused RMSNorm + Q8_1 quantization (dispatch count reduction for Q8_0 dp4a path).
    // rmsnorm_to_q8_1: RMSNorm + Q8_1 quantize in one kernel.
    // Replaces rmsnorm + quantize_f32_to_q8_1 at 2 sites/layer (attn_norm, ffn_norm).
    pub(crate) rmsnorm_to_q8_1: Option<CudaFunction>,
    // fused_residual_rmsnorm_q8_1: Residual add + RMSNorm + Q8_1 quantize.
    // For Q8_0 inter-layer boundaries: fuses residual_add_copy + rmsnorm + quantize_f32_to_q8_1.
    pub(crate) fused_residual_rmsnorm_q8_1: Option<CudaFunction>,
}

/// Compile all CUDA kernels via NVRTC and return the kernel function handles.
///
/// Each .cu source is compiled into a separate PTX module. This avoids
/// symbol conflicts between kernels that define identically-named device
/// helper functions (e.g., `warp_reduce_sum` appears in multiple .cu files).
pub(crate) fn compile_all_kernels(device: &CudaDevice) -> Result<KernelSet, RuntimeError> {
    let load_fn = |source: &str, name: &str| -> Result<CudaFunction, RuntimeError> {
        let module = device.compile_and_load(source)?;
        module.load_function(name).map_err(|e| {
            RuntimeError::Compute(format!("Failed to load CUDA kernel '{name}': {e}"))
        })
    };

    // For kernels needing SM 80+ features (dp4a, WMMA tensor cores).
    let load_fn_sm80 = |source: &str, name: &str| -> Result<CudaFunction, RuntimeError> {
        let module = device.compile_and_load_with_arch(source, "compute_80")?;
        module.load_function(name).map_err(|e| {
            RuntimeError::Compute(format!("Failed to load SM80 CUDA kernel '{name}': {e}"))
        })
    };

    // For dp4a kernels: SM 80+ with --use_fast_math (--fmad=true --ftz=true
    // --prec-div=false --prec-sqrt=false). Accelerates scale multiplication
    // in bandwidth-bound GEMV kernels where full FP precision is unnecessary.
    let load_fn_sm80_fast_math = |source: &str, name: &str| -> Result<CudaFunction, RuntimeError> {
        let module = device.compile_and_load_with_arch_fast_math(source, "compute_80")?;
        module.load_function(name).map_err(|e| {
            RuntimeError::Compute(format!("Failed to load SM80 fast_math CUDA kernel '{name}': {e}"))
        })
    };

    Ok(KernelSet {
        rmsnorm: load_fn(shaders::NORM_KERNEL_SOURCE, "rmsnorm")?,
        rmsnorm_per_head: load_fn(shaders::NORM_KERNEL_SOURCE, "rmsnorm_per_head")?,
        matvec_f32: load_fn(shaders::MATVEC_F32_KERNEL_SOURCE, "matvec_f32")?,
        matvec_f32_residual: load_fn(
            shaders::MATVEC_F32_KERNEL_SOURCE,
            "matvec_f32_residual",
        )?,
        matvec_f16: load_fn(shaders::MATVEC_F16_KERNEL_SOURCE, "matvec_f16")?,
        matvec_f16_residual: load_fn(
            shaders::MATVEC_F16_KERNEL_SOURCE,
            "matvec_f16_residual",
        )?,
        f32_to_f16_vec: load_fn(shaders::CONVERT_F16_KERNEL_SOURCE, "f32_to_f16_vec")?,
        f32_to_f16_vec4: load_fn(shaders::CONVERT_F16_KERNEL_SOURCE, "f32_to_f16_vec4").ok(),
        f16_to_f32_vec: load_fn(shaders::CONVERT_F16_KERNEL_SOURCE, "f16_to_f32_vec")?,
        matvec_q8_0: load_fn(shaders::MATVEC_Q8_0_KERNEL_SOURCE, "matvec_q8_0")?,
        matvec_q8_0_residual: load_fn(
            shaders::MATVEC_Q8_0_KERNEL_SOURCE,
            "matvec_q8_0_residual",
        )?,
        // v2, v3 removed (dead code — never dispatched, wastes compile time + GPU memory)
        matvec_q4_0: load_fn(shaders::MATVEC_Q4_0_KERNEL_SOURCE, "matvec_q4_0")?,
        matvec_q4_0_residual: load_fn(
            shaders::MATVEC_Q4_0_KERNEL_SOURCE,
            "matvec_q4_0_residual",
        )?,
        rope_apply: load_fn(shaders::ROPE_KERNEL_SOURCE, "rope_apply")?,
        swiglu_inplace: load_fn(shaders::ACTIVATIONS_KERNEL_SOURCE, "swiglu_inplace")?,
        residual_add: load_fn(shaders::ACTIVATIONS_KERNEL_SOURCE, "residual_add")?,
        residual_add_copy: load_fn(shaders::ACTIVATIONS_KERNEL_SOURCE, "residual_add_copy")?,
        attention_decode: load_fn(shaders::ATTENTION_KERNEL_SOURCE, "attention_decode")?,
        gemm_f32: load_fn(shaders::GEMM_F32_KERNEL_SOURCE, "gemm_f32")?,
        gemm_f32_residual: load_fn(shaders::GEMM_F32_KERNEL_SOURCE, "gemm_f32_residual")?,
        compute_rms_scale: load_fn(
            shaders::FUSED_RMSNORM_MATVEC_KERNEL_SOURCE,
            "compute_rms_scale",
        )?,
        fused_norm_matvec_f32: load_fn(
            shaders::FUSED_RMSNORM_MATVEC_KERNEL_SOURCE,
            "fused_norm_matvec_f32",
        )?,
        fused_norm_dual_matvec_f32: load_fn(
            shaders::FUSED_RMSNORM_MATVEC_KERNEL_SOURCE,
            "fused_norm_dual_matvec_f32",
        )?,
        embed_batch_f32: load_fn(shaders::PREFILL_KERNEL_SOURCE, "embed_batch_f32")?,
        embed_batch_q8_0: load_fn(shaders::PREFILL_KERNEL_SOURCE, "embed_batch_q8_0")?,
        embed_batch_f16: load_fn(shaders::PREFILL_EMBED_KERNEL_SOURCE, "embed_batch_f16")?,
        embed_batch_q4_0: load_fn(shaders::PREFILL_EMBED_KERNEL_SOURCE, "embed_batch_q4_0")?,
        rmsnorm_batched: load_fn(shaders::PREFILL_KERNEL_SOURCE, "rmsnorm_batched")?,
        rope_apply_batched: load_fn(shaders::PREFILL_KERNEL_SOURCE, "rope_apply_batched")?,
        kv_cache_write_batch: load_fn(
            shaders::PREFILL_KERNEL_SOURCE,
            "kv_cache_write_batch",
        )?,
        swiglu_batched: load_fn(shaders::PREFILL_KERNEL_SOURCE, "swiglu_batched")?,
        residual_add_batched: load_fn(
            shaders::PREFILL_KERNEL_SOURCE,
            "residual_add_batched",
        )?,
        extract_row: load_fn(shaders::PREFILL_KERNEL_SOURCE, "extract_row")?,
        scatter_row: load_fn(shaders::PREFILL_KERNEL_SOURCE, "scatter_row")?,
        flash_attention_v2: load_fn(
            shaders::FLASH_ATTENTION_KERNEL_SOURCE,
            "flash_attention_causal_v2",
        )?,
        flash_attention_br4: load_fn(
            shaders::FLASH_ATTENTION_KERNEL_SOURCE,
            "flash_attention_causal_br4",
        )?,
        dequant_q8_0_to_f16: load_fn(
            shaders::DEQUANT_Q8_0_KERNEL_SOURCE,
            "dequant_q8_0_to_f16",
        )?,
        dequant_q8_0_to_f32: load_fn(
            shaders::DEQUANT_Q8_0_KERNEL_SOURCE,
            "dequant_q8_0_to_f32",
        )?,
        dequant_q4_0_to_f16: load_fn(
            shaders::DEQUANT_Q4_0_F16_KERNEL_SOURCE,
            "dequant_q4_0_to_f16",
        )?,
        dequant_q4_0_to_f32: load_fn(
            shaders::DEQUANT_Q4_0_F16_KERNEL_SOURCE,
            "dequant_q4_0_to_f32",
        )?,
        matvec_q8_0_native: match load_fn(
            shaders::MATVEC_Q8_0_NATIVE_KERNEL_SOURCE,
            "matvec_q8_0_native",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] Q8_0 native warp kernel: FAILED: {e}"); None }
        },
        matvec_q8_0_native_residual: match load_fn(
            shaders::MATVEC_Q8_0_NATIVE_KERNEL_SOURCE,
            "matvec_q8_0_native_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] Q8_0 native warp residual: FAILED: {e}"); None }
        },
        matvec_q8_0_dp4a: match load_fn_sm80(
            shaders::MATVEC_Q8_0_DP4A_KERNEL_SOURCE,
            "matvec_q8_0_dp4a",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] dp4a kernel: FAILED (fallback to v1): {e}"); None }
        },
        matvec_q8_0_dp4a_residual: match load_fn_sm80(
            shaders::MATVEC_Q8_0_DP4A_KERNEL_SOURCE,
            "matvec_q8_0_dp4a_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] dp4a_residual: FAILED: {e}"); None }
        },
        matvec_q8_0_aligned: match load_fn_sm80(
            shaders::MATVEC_Q8_0_ALIGNED_KERNEL_SOURCE,
            "matvec_q8_0_aligned",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] Q8_0 aligned dp4a kernel: FAILED: {e}"); None }
        },
        matvec_q8_0_aligned_residual: match load_fn_sm80(
            shaders::MATVEC_Q8_0_ALIGNED_KERNEL_SOURCE,
            "matvec_q8_0_aligned_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] Q8_0 aligned dp4a residual: FAILED: {e}"); None }
        },
        repack_q8_0_to_aligned36: match load_fn(
            shaders::REPACK_Q8_ALIGNED_KERNEL_SOURCE,
            "repack_q8_0_to_aligned36",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] repack_q8_0_to_aligned36: FAILED: {e}"); None }
        },
        matvec_q8_0_v4: load_fn_sm80(
            shaders::MATVEC_Q8_0_V4_KERNEL_SOURCE,
            "matvec_q8_0_v4",
        ).ok(),
        matvec_q8_0_v4_residual: load_fn_sm80(
            shaders::MATVEC_Q8_0_V4_KERNEL_SOURCE,
            "matvec_q8_0_v4_residual",
        ).ok(),
        flash_attention_wmma: match load_fn_sm80(
            shaders::FLASH_ATTENTION_WMMA_KERNEL_SOURCE,
            "flash_attention_wmma",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] WMMA flash attention: FAILED: {e}"); None }
        },
        argmax_f32: load_fn(shaders::ARGMAX_KERNEL_SOURCE, "argmax_f32")?,
        // Q8_0 shared-memory matvec (PRIMARY Q8_0 decode path)
        matvec_q8_0_smem: match load_fn(
            shaders::MATVEC_Q8_0_SMEM_KERNEL_SOURCE,
            "matvec_q8_0_smem",
        ) {
            Ok(f) => { eprintln!("[CUDA] Q8_0 smem matvec: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] Q8_0 smem matvec: FAILED: {e}"); None }
        },
        matvec_q8_0_smem_residual: match load_fn(
            shaders::MATVEC_Q8_0_SMEM_KERNEL_SOURCE,
            "matvec_q8_0_smem_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] Q8_0 smem matvec residual: FAILED: {e}"); None }
        },
        // Q4_0 shared-memory matvec (PRIMARY Q4_0 decode path)
        matvec_q4_0_smem: match load_fn(
            shaders::MATVEC_Q4_0_SMEM_KERNEL_SOURCE,
            "matvec_q4_0_smem",
        ) {
            Ok(f) => { eprintln!("[CUDA] Q4_0 smem matvec: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] Q4_0 smem matvec: FAILED: {e}"); None }
        },
        matvec_q4_0_smem_residual: match load_fn(
            shaders::MATVEC_Q4_0_SMEM_KERNEL_SOURCE,
            "matvec_q4_0_smem_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] Q4_0 smem matvec residual: FAILED: {e}"); None }
        },
        // GDN kernels (for Qwen3.5 hybrid layers)
        ssm_conv1d_decode: load_fn(shaders::GDN_KERNEL_SOURCE, "ssm_conv1d_decode").ok(),
        gdn_compute_gates: load_fn(shaders::GDN_KERNEL_SOURCE, "gdn_compute_gates").ok(),
        l2_normalize_heads: load_fn(shaders::GDN_KERNEL_SOURCE, "l2_normalize_heads").ok(),
        gdn_state_update: load_fn(shaders::GDN_KERNEL_SOURCE, "gdn_state_update").ok(),
        silu_inplace: load_fn(shaders::ACTIVATIONS_KERNEL_SOURCE, "silu_inplace").ok(),
        silu_elementwise_mul: load_fn(shaders::ACTIVATIONS_KERNEL_SOURCE, "silu_elementwise_mul").ok(),
        // GDN fused decode megakernels (8 -> 2 launches per GDN layer)
        gdn_decode_megakernel: match load_fn(
            shaders::GDN_MEGAKERNEL_SOURCE,
            "gdn_decode_megakernel",
        ) {
            Ok(f) => { eprintln!("[CUDA] gdn_decode_megakernel: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] gdn_decode_megakernel: FAILED: {e}"); None }
        },
        gdn_rmsnorm_silu_gate: match load_fn(
            shaders::GDN_MEGAKERNEL_SOURCE,
            "gdn_rmsnorm_silu_gate",
        ) {
            Ok(f) => { eprintln!("[CUDA] gdn_rmsnorm_silu_gate: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] gdn_rmsnorm_silu_gate: FAILED: {e}"); None }
        },
        // GDN fused prefill kernels
        ssm_conv1d_silu_prefill: load_fn(shaders::GDN_KERNEL_SOURCE, "ssm_conv1d_silu_prefill").ok(),
        gdn_compute_gates_batched: load_fn(shaders::GDN_KERNEL_SOURCE, "gdn_compute_gates_batched").ok(),
        l2_normalize_qk_strided: load_fn(shaders::GDN_KERNEL_SOURCE, "l2_normalize_qk_strided").ok(),
        gdn_prefill_fused_v3: load_fn(shaders::GDN_KERNEL_SOURCE, "gdn_prefill_fused_v3").ok(),
        gdn_prefill_norm_gate: load_fn(shaders::GDN_KERNEL_SOURCE, "gdn_prefill_norm_gate").ok(),
        // Fused F16 decode kernels (dispatch count reduction)
        fused_rmsnorm_f16: match load_fn(
            shaders::FUSED_F16_KERNEL_SOURCE,
            "fused_rmsnorm_f16",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_rmsnorm_f16: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_rmsnorm_f16: FAILED: {e}"); None }
        },
        swiglu_f32_to_f16: match load_fn(
            shaders::FUSED_F16_KERNEL_SOURCE,
            "swiglu_f32_to_f16",
        ) {
            Ok(f) => { eprintln!("[CUDA] swiglu_f32_to_f16: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] swiglu_f32_to_f16: FAILED: {e}"); None }
        },
        fused_residual_rmsnorm_f16: match load_fn(
            shaders::FUSED_F16_KERNEL_SOURCE,
            "fused_residual_rmsnorm_f16",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_residual_rmsnorm_f16: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_residual_rmsnorm_f16: FAILED: {e}"); None }
        },
        fused_residual_rmsnorm_f32: match load_fn(
            shaders::NORM_KERNEL_SOURCE,
            "fused_residual_rmsnorm_f32",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_residual_rmsnorm_f32: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_residual_rmsnorm_f32: FAILED: {e}"); None }
        },
        fused_residual_rms_scale: match load_fn(
            shaders::NORM_KERNEL_SOURCE,
            "fused_residual_rms_scale",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_residual_rms_scale: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_residual_rms_scale: FAILED: {e}"); None }
        },
        // Fused F16 prefill kernels (dispatch count reduction for batched HGEMM)
        fused_rmsnorm_f16_batched: match load_fn(
            shaders::FUSED_F16_KERNEL_SOURCE,
            "fused_rmsnorm_f16_batched",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_rmsnorm_f16_batched: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_rmsnorm_f16_batched: FAILED: {e}"); None }
        },
        swiglu_f32_to_f16_batched: match load_fn(
            shaders::FUSED_F16_KERNEL_SOURCE,
            "swiglu_f32_to_f16_batched",
        ) {
            Ok(f) => { eprintln!("[CUDA] swiglu_f32_to_f16_batched: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] swiglu_f32_to_f16_batched: FAILED: {e}"); None }
        },
        // Q8_0 dequant-in-register HGEMV (F16 x-vector shmem, NR=4)
        hgemv_q8_0: match load_fn(
            shaders::HGEMV_Q8_0_KERNEL_SOURCE,
            "hgemv_q8_0",
        ) {
            Ok(f) => { eprintln!("[CUDA] hgemv_q8_0: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] hgemv_q8_0: FAILED: {e}"); None }
        },
        hgemv_q8_0_residual: match load_fn(
            shaders::HGEMV_Q8_0_KERNEL_SOURCE,
            "hgemv_q8_0_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] hgemv_q8_0_residual: FAILED: {e}"); None }
        },
        // Q4_0 dequant-in-register HGEMV (F16 x-vector shmem, NR=4)
        hgemv_q4_0: match load_fn(
            shaders::HGEMV_Q4_0_KERNEL_SOURCE,
            "hgemv_q4_0",
        ) {
            Ok(f) => { eprintln!("[CUDA] hgemv_q4_0: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] hgemv_q4_0: FAILED: {e}"); None }
        },
        hgemv_q4_0_residual: match load_fn(
            shaders::HGEMV_Q4_0_KERNEL_SOURCE,
            "hgemv_q4_0_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] hgemv_q4_0_residual: FAILED: {e}"); None }
        },
        // dp4a with pre-quantized Q8_1 input
        // Compiled with --use_fast_math for accelerated scale multiplication.
        quantize_f32_to_q8_1: match load_fn_sm80_fast_math(
            shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE,
            "quantize_f32_to_q8_1",
        ) {
            Ok(f) => { eprintln!("[CUDA] quantize_f32_to_q8_1: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] quantize_f32_to_q8_1: FAILED: {e}"); None }
        },
        matvec_q8_0_q8_1: match load_fn_sm80_fast_math(
            shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE,
            "matvec_q8_0_q8_1",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q8_0_q8_1: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q8_0_q8_1: FAILED: {e}"); None }
        },
        matvec_q8_0_q8_1_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_DP4A_Q8_1_KERNEL_SOURCE,
            "matvec_q8_0_q8_1_residual",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] matvec_q8_0_q8_1_residual: FAILED: {e}"); None }
        },
        // Q4_0 dp4a with pre-quantized Q8_1 input (NR=4, nibble unpack + dp4a)
        // Compiled with --use_fast_math for accelerated scale multiplication.
        matvec_q4_0_dp4a: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_0_DP4A_KERNEL_SOURCE,
            "matvec_q4_0_dp4a",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q4_0_dp4a: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q4_0_dp4a: FAILED: {e}"); None }
        },
        matvec_q4_0_dp4a_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_0_DP4A_KERNEL_SOURCE,
            "matvec_q4_0_dp4a_residual",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q4_0_dp4a_residual: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q4_0_dp4a_residual: FAILED: {e}"); None }
        },
        // Q4Aligned + Q8_1 input dp4a (NR=4, aligned int* nibble loads)
        // Compiled with --use_fast_math for accelerated scale multiplication.
        matvec_q4_aligned_q8_1: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_ALIGNED_Q8_1_KERNEL_SOURCE,
            "matvec_q4_aligned_q8_1",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q4_aligned_q8_1: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q4_aligned_q8_1: FAILED: {e}"); None }
        },
        matvec_q4_aligned_q8_1_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_ALIGNED_Q8_1_KERNEL_SOURCE,
            "matvec_q4_aligned_q8_1_residual",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q4_aligned_q8_1_residual: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q4_aligned_q8_1_residual: FAILED: {e}"); None }
        },
        // Repack Q4_0 to 20-byte aligned blocks
        repack_q4_0_to_aligned20: match load_fn(
            shaders::REPACK_Q4_ALIGNED_KERNEL_SOURCE,
            "repack_q4_0_to_aligned20",
        ) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("[CUDA] repack_q4_0_to_aligned20: FAILED: {e}"); None }
        },
        // Optimal Q8Aligned + Q8_1 input dp4a (NR=2, both sides aligned)
        // Compiled with --use_fast_math for accelerated scale multiplication.
        matvec_q8_aligned_q8_1: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_Q8_1_KERNEL_SOURCE,
            "matvec_q8_aligned_q8_1",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q8_aligned_q8_1: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q8_aligned_q8_1: FAILED: {e}"); None }
        },
        matvec_q8_aligned_q8_1_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_Q8_1_KERNEL_SOURCE,
            "matvec_q8_aligned_q8_1_residual",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q8_aligned_q8_1_residual: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q8_aligned_q8_1_residual: FAILED: {e}"); None }
        },
        // Fused gate+up+SwiGLU GEMV with inline RMSNorm
        fused_glu_gemv_q8_0: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_q8_0",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_glu_gemv_q8_0: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_glu_gemv_q8_0: FAILED: {e}"); None }
        },
        fused_glu_gemv_q4_0: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_q4_0",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_glu_gemv_q4_0: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_glu_gemv_q4_0: FAILED: {e}"); None }
        },
        fused_glu_gemv_f16: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_f16",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_glu_gemv_f16: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_glu_gemv_f16: FAILED: {e}"); None }
        },
        fused_glu_gemv_q8_0_hg: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_q8_0_hg",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_glu_gemv_q8_0_hg: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_glu_gemv_q8_0_hg: FAILED: {e}"); None }
        },
        fused_glu_gemv_q4_0_hg: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_q4_0_hg",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_glu_gemv_q4_0_hg: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_glu_gemv_q4_0_hg: FAILED: {e}"); None }
        },
        // Fused gate+up+SwiGLU GEMV for Q8Aligned (36-byte blocks)
        fused_glu_gemv_q8_aligned: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_q8_aligned",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_glu_gemv_q8_aligned: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_glu_gemv_q8_aligned: FAILED: {e}"); None }
        },
        fused_glu_gemv_q8_aligned_hg: match load_fn(
            shaders::FUSED_GLU_GEMV_KERNEL_SOURCE,
            "fused_glu_gemv_q8_aligned_hg",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_glu_gemv_q8_aligned_hg: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_glu_gemv_q8_aligned_hg: FAILED: {e}"); None }
        },
        // Fused down projection: inline F32->Q8_1 quantize + dp4a matvec
        // Compiled with --use_fast_math for accelerated scale multiplication.
        matvec_q8_aligned_f32: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q8_aligned_f32",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q8_aligned_f32: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q8_aligned_f32: FAILED: {e}"); None }
        },
        matvec_q8_aligned_f32_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q8_aligned_f32_residual",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q8_aligned_f32_residual: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q8_aligned_f32_residual: FAILED: {e}"); None }
        },
        matvec_q8_aligned_f32_swiglu: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q8_aligned_f32_swiglu",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q8_aligned_f32_swiglu: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q8_aligned_f32_swiglu: FAILED: {e}"); None }
        },
        matvec_q8_aligned_f32_swiglu_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q8_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q8_aligned_f32_swiglu_residual",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q8_aligned_f32_swiglu_residual: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q8_aligned_f32_swiglu_residual: FAILED: {e}"); None }
        },
        // Fused down projection for Q4Aligned: inline F32->Q8_1 quantize + dp4a.
        // Compiled with --use_fast_math for accelerated scale multiplication.
        matvec_q4_aligned_f32: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q4_aligned_f32",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q4_aligned_f32: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q4_aligned_f32: FAILED: {e}"); None }
        },
        matvec_q4_aligned_f32_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q4_aligned_f32_residual",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q4_aligned_f32_residual: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q4_aligned_f32_residual: FAILED: {e}"); None }
        },
        matvec_q4_aligned_f32_swiglu: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q4_aligned_f32_swiglu",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q4_aligned_f32_swiglu: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q4_aligned_f32_swiglu: FAILED: {e}"); None }
        },
        matvec_q4_aligned_f32_swiglu_residual: match load_fn_sm80_fast_math(
            shaders::MATVEC_Q4_ALIGNED_FUSED_DOWN_KERNEL_SOURCE,
            "matvec_q4_aligned_f32_swiglu_residual",
        ) {
            Ok(f) => { eprintln!("[CUDA] matvec_q4_aligned_f32_swiglu_residual: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] matvec_q4_aligned_f32_swiglu_residual: FAILED: {e}"); None }
        },
        // Fused RMSNorm + Q8_1 quantization (dispatch count reduction for Q8_0 dp4a path)
        rmsnorm_to_q8_1: match load_fn(
            shaders::RMSNORM_Q8_1_KERNEL_SOURCE,
            "rmsnorm_to_q8_1",
        ) {
            Ok(f) => { eprintln!("[CUDA] rmsnorm_to_q8_1: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] rmsnorm_to_q8_1: FAILED: {e}"); None }
        },
        fused_residual_rmsnorm_q8_1: match load_fn(
            shaders::RMSNORM_Q8_1_KERNEL_SOURCE,
            "fused_residual_rmsnorm_q8_1",
        ) {
            Ok(f) => { eprintln!("[CUDA] fused_residual_rmsnorm_q8_1: OK"); Some(f) }
            Err(e) => { eprintln!("[CUDA] fused_residual_rmsnorm_q8_1: FAILED: {e}"); None }
        },
    })
}

/// Shared memory bytes needed for the RMSNorm kernel.
///
/// The kernel uses (block_size / 32) floats for per-warp partial sums.
pub(crate) fn rmsnorm_shared_bytes(block_size: u32) -> u32 {
    (block_size / 32) * 4
}

/// Shared memory bytes needed for the attention_decode kernel.
///
/// Layout: 8 floats for warp partial reduction + seq_len floats for scores.
pub(crate) fn attention_shared_bytes(seq_len: u32) -> u32 {
    (8 + seq_len) * 4
}

/// Block size for RMSNorm: use min(dim, 1024), rounded down to a multiple of 32.
///
/// The kernel runs as a single block, so all threads must fit in one block.
pub(crate) fn rmsnorm_block_size(dim: usize) -> u32 {
    let bs = dim.min(1024);
    // Round down to nearest multiple of 32 (warp size).
    let bs = (bs / 32) * 32;
    bs.max(32) as u32
}

/// Block size for F32/Q4_0 matvec: 256 threads (one block per output row).
pub(crate) fn matvec_block_size() -> u32 {
    256
}

/// Q8_0 matvec constants matching matvec_q8_0.cu kernel defines.
///
/// The Q8_0 kernel uses NR=2 rows per thread block with 128 threads (4 warps).
/// Grid must be ceil(out_dim / NR), not out_dim.
pub(crate) const Q8_0_NR: u32 = 2;
pub(crate) const Q8_0_BLOCK_DIM: u32 = 128;

/// Grid size for Q8_0 matvec: ceil(out_dim / NR) blocks.
pub(crate) fn matvec_q8_0_grid(out_dim: u32) -> u32 {
    (out_dim + Q8_0_NR - 1) / Q8_0_NR
}

/// Smem Q8_0/Q4_0 matvec constants (must match matvec_q8_0_smem.cu / matvec_q4_0_smem.cu).
///
/// NR=2 rows per block, 256 threads (8 warps). Shared memory = in_dim * 4 bytes.
pub(crate) const SMEM_NR: u32 = 2;
pub(crate) const SMEM_BLOCK_DIM: u32 = 256;

/// Grid size for smem matvec: ceil(out_dim / NR) blocks.
pub(crate) fn matvec_smem_grid(out_dim: u32) -> u32 {
    (out_dim + SMEM_NR - 1) / SMEM_NR
}

/// Shared memory bytes for smem matvec: in_dim floats for x-vector cache.
pub(crate) fn matvec_smem_shared_bytes(in_dim: u32) -> u32 {
    in_dim * 4
}

/// Block size for attention: min(seq_len, 256), rounded up to multiple of 32.
///
/// Must be at least 32 (one warp) for the reduction helpers to work correctly.
pub(crate) fn attention_block_size(seq_len: usize) -> u32 {
    let bs = seq_len.min(256);
    // Round up to nearest multiple of 32.
    let bs = ((bs + 31) / 32) * 32;
    bs.max(32) as u32
}

/// Block size for fused norm+matvec F32: 256 threads (matches FUSED_BLOCK_SIZE define).
pub(crate) fn fused_norm_matvec_block_size() -> u32 {
    256
}

/// GEMM tile dimensions (must match gemm_f32.cu defines).
pub(crate) const GEMM_TILE_M: u32 = 32;
pub(crate) const GEMM_TILE_N: u32 = 32;

/// GEMM block dimensions: 32x32 threads per block (one thread per output element in a tile).
pub(crate) fn gemm_block_dim() -> (u32, u32) {
    (GEMM_TILE_N, GEMM_TILE_M) // (x=col, y=row)
}

/// GEMM grid dimensions for an M x N output.
pub(crate) fn gemm_grid_dim(m: usize, n: usize) -> (u32, u32) {
    let grid_x = (n as u32).div_ceil(GEMM_TILE_N);
    let grid_y = (m as u32).div_ceil(GEMM_TILE_M);
    (grid_x, grid_y)
}

// ------------------------------------------------------------------
// Flash Attention v2 constants and helpers (must match flash_attention.cu)
// ------------------------------------------------------------------

/// KV tile size for flash attention (must match FA_BC in flash_attention.cu).
pub(crate) const FA_BC: u32 = 32;

/// Query tile size for Br=4 variant (must match FA_BR in flash_attention.cu).
pub(crate) const FA_BR: u32 = 4;

/// Block size for flash_attention_causal_v2 (Br=1).
///
/// One thread block per (head, query_token). 128 threads provides 4 warps
/// for cooperative dot products and V accumulation.
pub(crate) fn flash_attention_v2_block_size() -> u32 {
    128
}

/// Shared memory bytes for flash_attention_causal_v2.
///
/// Layout: partial[8] + q_row[head_dim] + s_tile[FA_BC].
pub(crate) fn flash_attention_v2_shared_bytes(head_dim: u32) -> u32 {
    (8 + head_dim + FA_BC) * 4
}

/// Block size for flash_attention_causal_br4 (Br=4).
///
/// 128 threads = 4 warps, one warp per query row.
pub(crate) fn flash_attention_br4_block_size() -> u32 {
    128
}

/// Shared memory bytes for flash_attention_causal_br4.
///
/// Layout: q_rows[4][head_dim] + s_tiles[4][FA_BC].
pub(crate) fn flash_attention_br4_shared_bytes(head_dim: u32) -> u32 {
    FA_BR * (head_dim + FA_BC) * 4
}

// ------------------------------------------------------------------
// WMMA Flash Attention constants and helpers (must match flash_attention_wmma.cu)
// ------------------------------------------------------------------

/// Query tile rows for WMMA variant (must match FA_TC_BR in flash_attention_wmma.cu).
pub(crate) const FA_TC_BR: u32 = 16;

/// KV tile columns for WMMA variant (must match FA_TC_BC in flash_attention_wmma.cu).
pub(crate) const FA_TC_BC: u32 = 16;

/// Block size for flash_attention_wmma (128 threads = 4 warps).
pub(crate) fn flash_attention_wmma_block_size() -> u32 {
    128
}

/// Shared memory bytes for flash_attention_wmma.
///
/// Layout:
///   half Q_sh[BR * head_dim]        = BR * hd * 2 bytes
///   half KV_sh[BC * head_dim]       = BC * hd * 2 bytes (reused for K then V)
///   float S_sh[BR * BC]             = BR * BC * 4 bytes
///   half P_sh[BR * BC]              = BR * BC * 2 bytes
///   float O_acc[BR * head_dim]      = BR * hd * 4 bytes
///   float rowmax[BR]                = BR * 4 bytes
///   float rowsum[BR]                = BR * 4 bytes
pub(crate) fn flash_attention_wmma_shared_bytes(head_dim: u32) -> u32 {
    let br = FA_TC_BR;
    let bc = FA_TC_BC;
    let hd = head_dim;

    let q_sh = br * hd * 2;         // half Q_sh[BR][hd]
    let kv_sh = bc * hd * 2;        // half KV_sh[BC][hd]
    let s_sh = br * bc * 4;         // float S_sh[BR][BC]
    let p_sh = br * bc * 2;         // half P_sh[BR][BC]
    let o_acc = br * hd * 4;        // float O_acc[BR][hd]
    let rowmax = br * 4;            // float rowmax[BR]
    let rowsum = br * 4;            // float rowsum[BR]

    q_sh + kv_sh + s_sh + p_sh + o_acc + rowmax + rowsum
}

// ------------------------------------------------------------------
// HGEMV (dequant-in-register) constants and helpers
// (must match hgemv_q8_0.cu / hgemv_q4_0.cu)
// ------------------------------------------------------------------

/// Rows per thread block for HGEMV kernels.
pub(crate) const HGEMV_NR: u32 = 4;

/// Threads per block for HGEMV kernels.
pub(crate) const HGEMV_BLOCK_DIM: u32 = 256;

/// Grid size for HGEMV: ceil(out_dim / NR) blocks.
pub(crate) fn hgemv_grid(out_dim: u32) -> u32 {
    (out_dim + HGEMV_NR - 1) / HGEMV_NR
}

/// Shared memory bytes for HGEMV: in_dim * 2 (F16 x-vector cache).
pub(crate) fn hgemv_shared_bytes(in_dim: u32) -> u32 {
    in_dim * 2
}

/// Maximum shared memory (bytes) per block on A100. Default limit before
/// requesting dynamic shared memory extension via cudaFuncSetAttribute.
/// 48 KB = 49152 bytes. HGEMV uses in_dim * 2 bytes for F16 x-vector,
/// so covers up to in_dim = 24576.
pub(crate) const HGEMV_SHMEM_LIMIT: u32 = 49152;

// ------------------------------------------------------------------
// dp4a Q8_1 constants and helpers
// (must match matvec_dp4a_q8_1.cu defines)
// ------------------------------------------------------------------

/// Q8_1 block size in bytes: [f16 scale, f16 sum, 32 × int8] = 36 bytes.
pub(crate) const Q8_1_BLOCK_BYTES: u32 = 36;

/// Elements per Q8_1 block (same as Q8_0).
pub(crate) const Q8_1_BLOCK_SIZE: u32 = 32;

/// Threads per block for the Q8_1 quantization kernel (1 warp per block).
pub(crate) const Q8_1_QUANT_BLOCK_DIM: u32 = 32;

/// Threads per block for the dp4a Q8_1 matvec kernel (4 warps).
pub(crate) const DP4A_Q8_1_BLOCK_DIM: u32 = 128;

/// Size of the Q8_1 buffer in bytes for a given input dimension.
///
/// Each group of 32 F32 elements becomes one 36-byte Q8_1 block.
pub(crate) fn q8_1_buffer_bytes(in_dim: u32) -> u32 {
    let num_blocks = in_dim / Q8_1_BLOCK_SIZE;
    num_blocks * Q8_1_BLOCK_BYTES
}

/// Grid size for the Q8_1 quantization kernel: one warp per block.
pub(crate) fn q8_1_quant_grid(in_dim: u32) -> u32 {
    in_dim / Q8_1_BLOCK_SIZE
}

/// Grid size for the dp4a Q8_1 matvec: ceil(out_dim / MV_NR) blocks.
/// MV_NR=2: each block processes 2 output rows.
/// Used by Q8_0, Q8Aligned, and Q8_1 kernels (NR=2).
pub(crate) fn dp4a_q8_1_grid(out_dim: u32) -> u32 {
    (out_dim + 1) / 2
}

/// Threads per block for Q4_0 dp4a kernels (NR=4, 8 warps).
/// Q4 kernels use NR=4 with 256 threads for better x-vector amortization:
/// 4x weight bandwidth amortization per x-vector load vs 2x for Q8.
/// Q4 blocks are smaller (18-20 bytes vs 34-36 for Q8), so NR=4 doesn't
/// cause register spill despite processing 4 rows per block.
pub(crate) const DP4A_Q4_BLOCK_DIM: u32 = 256;

/// Grid size for Q4_0 dp4a matvec: ceil(out_dim / 4) blocks.
/// NR=4: each block processes 4 output rows.
pub(crate) fn dp4a_q4_grid(out_dim: u32) -> u32 {
    (out_dim + 3) / 4
}

// ------------------------------------------------------------------
// Fused GLU GEMV constants and helpers
// (must match fused_glu_gemv.cu defines)
// ------------------------------------------------------------------

/// Rows per thread block for fused GLU GEMV (must match NR in fused_glu_gemv.cu).
pub(crate) const FUSED_GLU_NR: u32 = 2;

/// Threads per block for fused GLU GEMV (must match BLOCK_DIM in fused_glu_gemv.cu).
pub(crate) const FUSED_GLU_BLOCK_DIM: u32 = 256;

/// Grid size for fused GLU GEMV: ceil(inter_dim / NR) blocks.
pub(crate) fn fused_glu_grid(inter_dim: u32) -> u32 {
    (inter_dim + FUSED_GLU_NR - 1) / FUSED_GLU_NR
}

/// Shared memory bytes for fused GLU GEMV (F32 x-vector variant):
/// hidden_dim * 4 bytes for the normed x-vector.
pub(crate) fn fused_glu_shared_bytes_f32(hidden_dim: u32) -> u32 {
    hidden_dim * 4
}

/// Shared memory bytes for fused GLU GEMV (F16 x-vector variant):
/// hidden_dim * 2 bytes for the normed F16 x-vector.
pub(crate) fn fused_glu_shared_bytes_f16(hidden_dim: u32) -> u32 {
    hidden_dim * 2
}

/// Maximum shared memory per block (48KB on A100).
/// F32 variant covers hidden_dim <= 12288 (12288 * 4 = 49152).
/// F16 variant covers hidden_dim <= 24576 (24576 * 2 = 49152).
pub(crate) const FUSED_GLU_SHMEM_LIMIT: u32 = 49152;
