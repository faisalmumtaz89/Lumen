//! CUDA kernel source code.
//!
//! GPU compute kernels are compiled at runtime via NVRTC (runtime compilation)
//! from these embedded source strings. This mirrors the Metal backend's pattern
//! of embedding MSL source as `include_str!()` constants.
//!
//! NVRTC compiles .cu source to PTX at runtime, which is then loaded as a
//! module by the CUDA driver. This avoids requiring `nvcc` at build time and
//! keeps the single-binary deployment model intact.

/// Token embedding lookup kernels (F32 row copy, Q8_0 dequantize).
pub const EMBED_KERNEL_SOURCE: &str = include_str!("embed.cu");

/// RMSNorm kernel (single-block shared memory reduction + warp shuffle).
pub const NORM_KERNEL_SOURCE: &str = include_str!("norm.cu");

/// Rotary Position Embedding kernel (interleaved pairs, pre-computed cos/sin).
pub const ROPE_KERNEL_SOURCE: &str = include_str!("rope.cu");

/// Elementwise activation kernels (SwiGLU, residual_add, softmax).
pub const ACTIVATIONS_KERNEL_SOURCE: &str = include_str!("activations.cu");

/// F32 matrix-vector multiply kernels (matvec, matvec + residual).
pub const MATVEC_F32_KERNEL_SOURCE: &str = include_str!("matvec_f32.cu");

/// Q8_0 matrix-vector multiply kernels (matvec with on-the-fly dequantization).
pub const MATVEC_Q8_0_KERNEL_SOURCE: &str = include_str!("matvec_q8_0.cu");

/// Q8_0 matrix-vector multiply v2 (shared-memory x vector caching).
pub const MATVEC_Q8_0_V2_KERNEL_SOURCE: &str = include_str!("matvec_q8_0_v2.cu");

/// Q8_0 matrix-vector multiply v3 (vectorized int loads for quants + float4 x loads).
pub const MATVEC_Q8_0_V3_KERNEL_SOURCE: &str = include_str!("matvec_q8_0_v3.cu");

/// Q8_0 matrix-vector multiply dp4a (INT8 dot product via __dp4a(), on-the-fly x quantization).
/// Requires compute capability >= 6.1 (Pascal+).
pub const MATVEC_Q8_0_DP4A_KERNEL_SOURCE: &str = include_str!("matvec_q8_0_dp4a.cu");

/// Q8_0 vectorized matvec variants (float4 x loads, 64-bit quant loads, deferred scale, fused FFN).
pub const MATVEC_Q8_0_VEC_KERNEL_SOURCE: &str = include_str!("matvec_q8_0_vec.cu");

/// Q8_0 dequantization kernels (Q8_0→F32 and Q8_0→F16 for cuBLAS GEMM prefill).
pub const DEQUANT_Q8_0_KERNEL_SOURCE: &str = include_str!("dequant_q8_0_f16.cu");

/// Q4_0 matrix-vector multiply kernels (matvec with on-the-fly 4-bit dequantization).
pub const MATVEC_Q4_0_KERNEL_SOURCE: &str = include_str!("matvec_q4_0.cu");

/// KV cache scatter-write kernel (head-first layout).
pub const KV_CACHE_KERNEL_SOURCE: &str = include_str!("kv_cache.cu");

/// Multi-head attention decode kernel with GQA support.
pub const ATTENTION_KERNEL_SOURCE: &str = include_str!("attention.cu");

/// Tiled GEMM F32 kernels for batched prefill (32x32 tiles, shared memory).
pub const GEMM_F32_KERNEL_SOURCE: &str = include_str!("gemm_f32.cu");

/// F16 matrix-vector multiply kernels (on-the-fly f16->f32 dequantization).
pub const MATVEC_F16_KERNEL_SOURCE: &str = include_str!("matvec_f16.cu");

/// F16 KV cache kernels (f32->f16 write, f16->f32 read).
pub const KV_CACHE_F16_KERNEL_SOURCE: &str = include_str!("kv_cache_f16.cu");

/// F32 <-> F16 bulk conversion kernels (f32_to_f16_vec, f16_to_f32_vec).
pub const CONVERT_F16_KERNEL_SOURCE: &str = include_str!("convert_f16.cu");

/// GatedDeltaNet (GDN) kernels (conv1d, gates, L2 norm, state update).
pub const GDN_KERNEL_SOURCE: &str = include_str!("gdn.cu");

/// GDN decode megakernel: fuses 8 per-token launches into 2
/// (gdn_decode_megakernel + gdn_rmsnorm_silu_gate).
pub const GDN_MEGAKERNEL_SOURCE: &str = include_str!("gdn_megakernel.cu");

/// Fused RMSNorm + MatVec kernels (two-pass: compute_rms_scale + fused_norm_matvec_f32).
///
/// Eliminates the intermediate `normed[hidden_dim]` buffer by precomputing the
/// RMS scale as a single scalar, then normalizing `x` inline during the dot
/// product. Saves 1 kernel launch + 2 * hidden_dim * 4 bytes of global memory
/// traffic per fusion site.
pub const FUSED_RMSNORM_MATVEC_KERNEL_SOURCE: &str =
    include_str!("fused_rmsnorm_matvec.cu");

/// Batched prefill kernels: embed_batch, rmsnorm_batched, rope_apply_batched,
/// kv_cache_write_batch, swiglu_batched, residual_add_batched, extract_row.
///
/// These operate on [batch, dim] activation matrices. Used with the tiled GEMM
/// kernel (`gemm_f32`) to replace token-at-a-time matvec with batched GEMM,
/// reducing kernel launches from O(batch * layers * ops) to O(layers * ops).
pub const PREFILL_KERNEL_SOURCE: &str = include_str!("prefill_kernels.cu");

/// Batched embedding lookup kernels (F32, Q8_0, F16, Q4_0) for prefill.
pub const PREFILL_EMBED_KERNEL_SOURCE: &str = include_str!("prefill_embed.cu");

/// Batched RMSNorm kernel (one block per row, grid=batch).
pub const PREFILL_NORM_KERNEL_SOURCE: &str = include_str!("prefill_norm.cu");

/// Batched RoPE kernel (per-token position offsets).
pub const PREFILL_ROPE_KERNEL_SOURCE: &str = include_str!("prefill_rope.cu");

/// Batched KV cache scatter-write kernel (write N tokens at once).
pub const PREFILL_KV_KERNEL_SOURCE: &str = include_str!("prefill_kv.cu");

/// Batched elementwise kernels (SwiGLU, residual_add, extract/scatter row).
pub const PREFILL_ELEMENTWISE_KERNEL_SOURCE: &str = include_str!("prefill_elementwise.cu");

/// Flash Attention v2 causal kernels for batched prefill (online softmax, tiled).
///
/// Contains `flash_attention_causal_v2` (base) and `flash_attention_causal_br4`
/// (4-row batch variant). Both use online softmax with tile-level rescaling for
/// numerically stable attention without materializing the full score matrix.
pub const FLASH_ATTENTION_KERNEL_SOURCE: &str = include_str!("flash_attention.cu");

/// Tensor-core Flash Attention v2 (WMMA via inline PTX, SM 80+).
///
/// Uses `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` for the QK^T and
/// PV matrix multiplies, providing up to 16x throughput over scalar F32 on A100.
/// Falls back to `flash_attention_f16_scalar` on SM < 80 (F16 storage, F32 compute).
///
/// Kernels: `flash_attention_wmma` (tensor core, Br=16), `flash_attention_f16_scalar` (F16 fallback).
pub const FLASH_ATTENTION_WMMA_KERNEL_SOURCE: &str = include_str!("flash_attention_wmma.cu");

/// Q8_0 v4 matvec: dp4a INT8 dot product + cooperative x quantization + K-tiling.
/// NR=4 rows/block, 128 threads, 256-element K-tiles. Requires SM 6.1+ (dp4a).
pub const MATVEC_Q8_0_V4_KERNEL_SOURCE: &str = include_str!("matvec_q8_0_v4.cu");

/// Q4_0 dequantization to F16 for cuBLAS HGEMM prefill.
pub const DEQUANT_Q4_0_F16_KERNEL_SOURCE: &str = include_str!("dequant_q4_0_f16.cu");

/// GPU-side argmax: single-block parallel reduction to find max index.
pub const ARGMAX_KERNEL_SOURCE: &str = include_str!("argmax.cu");

/// Q8_0 native warp-cooperative matvec: scalar dequant+FMA, NO x-quantization.
/// 2 warps per row, deferred reduction. Reads 1.0625 bytes/elem (native Q8_0).
pub const MATVEC_Q8_0_NATIVE_KERNEL_SOURCE: &str = include_str!("matvec_q8_0_native.cu");

/// Q8_0 aligned matvec dp4a: 36-byte blocks with 4-byte-aligned quant data.
/// Enables native int* loads (8 instructions) instead of byte-level packing (56 ops).
/// Requires compute capability >= 6.1 (Pascal+).
pub const MATVEC_Q8_0_ALIGNED_KERNEL_SOURCE: &str = include_str!("matvec_q8_0_aligned.cu");

/// Repack Q8_0 blocks from 34 bytes to 36 bytes (insert 2-byte padding for alignment).
/// One-time kernel run during preload_weights, not on the decode hot path.
pub const REPACK_Q8_ALIGNED_KERNEL_SOURCE: &str = include_str!("repack_q8_aligned.cu");

/// Q8_0 shared-memory matvec: x-vector cached in shmem, scalar dequant+FMA.
/// NR=2 rows/block, 256 threads. Reads native Q8_0 (1.0625 B/elem) — bypasses
/// the 2x bandwidth penalty of HGEMV (2 B/elem for pre-dequanted F16).
/// PRIMARY Q8_0 decode path. Falls back to dp4a/v1 if shmem is insufficient.
pub const MATVEC_Q8_0_SMEM_KERNEL_SOURCE: &str = include_str!("matvec_q8_0_smem.cu");

/// Q4_0 shared-memory matvec: x-vector cached in shmem, nibble unpack+FMA.
/// NR=2 rows/block, 256 threads. Reads native Q4_0 (0.5625 B/elem) — bypasses
/// the 3.5x bandwidth penalty of HGEMV (2 B/elem for pre-dequanted F16).
/// PRIMARY Q4_0 decode path.
pub const MATVEC_Q4_0_SMEM_KERNEL_SOURCE: &str = include_str!("matvec_q4_0_smem.cu");

/// Fused F16 decode kernels: fused_rmsnorm_f16, swiglu_f32_to_f16.
///
/// Eliminates intermediate dispatches in the F16 HGEMV decode path:
/// - fused_rmsnorm_f16: RMSNorm + F32->F16 in one kernel (saves 1 dispatch per norm site)
/// - swiglu_f32_to_f16: SwiGLU + F32->F16 in one kernel (saves 1 dispatch per FFN)
/// Total: 3 fewer dispatches per layer, 96 fewer per 32-layer model.
pub const FUSED_F16_KERNEL_SOURCE: &str = include_str!("fused_f16.cu");

/// Q8_0 dequant-in-register HGEMV: NR=4, F16 x-vector in shmem (in_dim * 2 bytes).
///
/// Reads native Q8_0 (1.0625 B/elem) with F16 x-vector caching in shared memory.
/// Covers up to in_dim = 24576 (vs 12288 for F32-shmem smem kernel).
/// Primary path for Q8_0 decode when in_dim > 12288 (e.g. FFN down on 8B+ models).
/// Falls back to existing smem kernel for in_dim <= 12288 (smem kernel uses F32 x
/// and NR=2, so identical register pressure — no reason to replace it).
pub const HGEMV_Q8_0_KERNEL_SOURCE: &str = include_str!("hgemv_q8_0.cu");

/// Q4_0 dequant-in-register HGEMV: NR=4, F16 x-vector in shmem (in_dim * 2 bytes).
///
/// Reads native Q4_0 (0.5625 B/elem) with F16 x-vector caching in shared memory.
/// Covers up to in_dim = 24576. Primary path for Q4_0 when in_dim > 12288.
pub const HGEMV_Q4_0_KERNEL_SOURCE: &str = include_str!("hgemv_q4_0.cu");

/// dp4a GEMV with pre-quantized Q8_1 input vector.
///
/// Two-phase approach for optimized Q8_0 × Q8_1 decode:
///   1. `quantize_f32_to_q8_1`: Pre-quantize F32 x to Q8_1 (once per token)
///   2. `matvec_q8_0_q8_1`: dp4a dot product with native int* input loads
///
/// Q8_1 format: [f16 scale, f16 sum, 32 × int8] = 36 bytes per 32 elements.
/// Quant data at byte 4 is 4-byte aligned → native int* loads on the input side.
/// Weight reads: 1.0625 B/elem (Q8_0). Input reads: 1.125 B/elem (Q8_1).
/// NR=2 rows/block, 128 threads, NO shmem for input (L2 cache reuse).
/// Requires SM 6.1+ (dp4a).
pub const MATVEC_DP4A_Q8_1_KERNEL_SOURCE: &str = include_str!("matvec_dp4a_q8_1.cu");

/// Fused gate+up+SwiGLU GEMV with inline RMSNorm for single-token decode.
///
/// Reads the input vector ONCE, computes BOTH gate and up projections
/// simultaneously, and applies SwiGLU inline:
///   output[row] = silu(dot(w_gate[row], normed_x)) * dot(w_up[row], normed_x)
///
/// Eliminates 2-4 kernel launches per layer vs separate dispatch:
///   (rmsnorm + convert + gate GEMV + up GEMV + swiglu) -> (rms_scale + fused_glu)
///
/// Variants: Q8_0, Q4_0, F16 weights with F32 or F16 x-vector in shmem.
/// Falls back to existing separate dispatch if compilation fails.
pub const FUSED_GLU_GEMV_KERNEL_SOURCE: &str = include_str!("fused_glu_gemv.cu");

/// Q4_0 dp4a decode: native Q4_0 weights + pre-quantized Q8_1 input + dp4a + NR=2.
///
/// Unpacks Q4_0 nibbles into sequential int8 words for dp4a dot product against
/// Q8_1 pre-quantized input. Includes zero-point correction (-8 * w_scale * x_sum).
/// Reuses `quantize_f32_to_q8_1` from MATVEC_DP4A_Q8_1_KERNEL_SOURCE.
///
/// Kernels: `matvec_q4_0_dp4a`, `matvec_q4_0_dp4a_residual`.
/// Requires SM 6.1+ (dp4a).
pub const MATVEC_Q4_0_DP4A_KERNEL_SOURCE: &str = include_str!("matvec_q4_0_dp4a.cu");

/// Optimal Q8_0 decode: Q8Aligned weights + pre-quantized Q8_1 input + dp4a + NR=2.
///
/// Combines aligned int* weight loads (36-byte blocks) with pre-quantized Q8_1
/// input (also 36-byte blocks, quant data at offset 4). Both sides use native
/// int* loads (8 instructions each), dp4a for 4 MAC/instruction, and NR=2 to
/// halve x-vector bandwidth. Zero byte-packing overhead on either side.
///
/// Kernels: `matvec_q8_aligned_q8_1`, `matvec_q8_aligned_q8_1_residual`.
/// Reuses `quantize_f32_to_q8_1` from MATVEC_DP4A_Q8_1_KERNEL_SOURCE.
/// Requires SM 6.1+ (dp4a).
pub const MATVEC_Q8_ALIGNED_Q8_1_KERNEL_SOURCE: &str =
    include_str!("matvec_q8_aligned_q8_1.cu");

/// Repack Q4_0 blocks from 18 bytes to 20 bytes (insert 2-byte padding for alignment).
/// One-time kernel run during preload_weights, not on the decode hot path.
/// Nibble data at offset +4 is 4-byte aligned, enabling int* loads (4 instructions
/// loading 16 nibble bytes) instead of 16 individual byte loads per block.
pub const REPACK_Q4_ALIGNED_KERNEL_SOURCE: &str = include_str!("repack_q4_aligned.cu");

/// Optimal Q4_0 decode: Q4Aligned weights + pre-quantized Q8_1 input + dp4a + NR=2.
///
/// Combines aligned int* nibble loads (20-byte blocks, nibble data at offset +4)
/// with pre-quantized Q8_1 input. Weight nibble data loaded as 4 int* instructions
/// (vs 16 byte loads in the unaligned matvec_q4_0_dp4a kernel).
///
/// Kernels: `matvec_q4_aligned_q8_1`, `matvec_q4_aligned_q8_1_residual`.
/// Reuses `quantize_f32_to_q8_1` from MATVEC_DP4A_Q8_1_KERNEL_SOURCE.
/// Requires SM 6.1+ (dp4a).
pub const MATVEC_Q4_ALIGNED_Q8_1_KERNEL_SOURCE: &str =
    include_str!("matvec_q4_aligned_q8_1.cu");

/// Fused down projection: inline F32->Q8_1 quantization + dp4a matvec.
///
/// Eliminates the separate `quantize_f32_to_q8_1` dispatch by quantizing the
/// input vector on-the-fly within each thread's block iteration. Also includes
/// SwiGLU variants that fuse `swiglu_inplace` + quantize + matvec into one kernel.
///
/// Kernels: `matvec_q8_aligned_f32`, `matvec_q8_aligned_f32_residual`,
///          `matvec_q8_aligned_f32_swiglu`, `matvec_q8_aligned_f32_swiglu_residual`.
/// Requires SM 6.1+ (dp4a).
pub const MATVEC_Q8_ALIGNED_FUSED_DOWN_KERNEL_SOURCE: &str =
    include_str!("matvec_q8_aligned_fused_down.cu");

/// Fused down projection for Q4Aligned: inline F32->Q8_1 quantization
/// + __byte_perm nibble unpack + dp4a matvec against Q4Aligned weights.
///
/// Eliminates separate `quantize_f32_to_q8_1` dispatch (and optionally
/// `swiglu_inplace`) by fusing them into the dp4a matvec kernel.
///
/// Kernels: `matvec_q4_aligned_f32`, `matvec_q4_aligned_f32_residual`,
///          `matvec_q4_aligned_f32_swiglu`, `matvec_q4_aligned_f32_swiglu_residual`.
/// Requires SM 6.1+ (dp4a, __byte_perm).
pub const MATVEC_Q4_ALIGNED_FUSED_DOWN_KERNEL_SOURCE: &str =
    include_str!("matvec_q4_aligned_fused_down.cu");

/// Fused RMSNorm + Q8_1 quantization kernels (dispatch count reduction for Q8_0 dp4a path).
///
/// `rmsnorm_to_q8_1`: RMSNorm + Q8_1 quantize in one kernel (saves 1 dispatch per norm site).
/// `fused_residual_rmsnorm_q8_1`: Residual add + RMSNorm + Q8_1 quantize (saves 2 dispatches
/// per inter-layer boundary).
///
/// Replaces the separate `rmsnorm` + `quantize_f32_to_q8_1` dispatch pair for Q8_0 decode
/// paths that use dp4a with pre-quantized Q8_1 input. For 36-layer models: 72 fewer dispatches.
pub const RMSNORM_Q8_1_KERNEL_SOURCE: &str = include_str!("rmsnorm_q8_1.cu");

/// CUDA graph-compatible kernel variants (fixed geometry, parameter indirection).
pub const GRAPH_KERNEL_SOURCE: &str = include_str!("graph_kernels.cu");
