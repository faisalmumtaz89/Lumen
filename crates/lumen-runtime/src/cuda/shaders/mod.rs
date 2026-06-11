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

/// MMQ-style Q8_0 batched matmul.
///
/// MMQ-style INT8 x INT8 -> INT32 -> F32-scale math. Used to close the
/// qkv_pre_conv 5.85e-2 max-abs drift identified that HGEMM-F16
/// and SGEMM-F32 paths could not close.
///
/// Env-gated: `LUMEN_CUDA_Q8_PROJ_MMQ=1` routes Q8 projection matmul (QKV,
/// gate, alpha, beta in GDN layers, and MoE FFN projections) through this
/// kernel. Default OFF preserves byte-identical behaviour vs main.
pub const MMQ_Q8_0_KERNEL_SOURCE: &str = include_str!("mmq_q8_0.cu");

/// q4-specific MMQ twin of `MMQ_Q8_0_KERNEL_SOURCE`: Q4_0 weights x
/// per-token-INT8-quantized activation via dp4a (de-interleaved nibbles + -8
/// zero-point), matching llama.cpp `mul_mat_q` INT4 numerics for MoE q4
/// prefill projections. Default OFF; MoE-gated. Two extern "C" kernels:
/// `mmq_q4_0_batched` and `mmq_q4_0_batched_residual`.
pub const MMQ_Q4_0_KERNEL_SOURCE: &str = include_str!("mmq_q4_0.cu");

/// Q8_1-activation x {Q8_0,Q4_0}-weight matvec with dp4a INT8
/// dot-product, plus the `quantize_q8_1` activation pre-pass.
///
/// Three extern "C" kernels:
///   - `quantize_q8_1_rawsum`: F32 [in_dim] -> block_q8_1 [ceil(in_dim/32)*36 bytes]
///   - `mul_mat_vec_q_q8_0`: Q8_0 weights × Q8_1 activation -> F32 [out_dim]
///   - `mul_mat_vec_q_q4_0`: Q4_0 weights × Q8_1 activation -> F32 [out_dim]
///
/// Env-gated: `LUMEN_CUDA_MMV_Q_DP4A=1` (sub-gates per call site) replaces
/// `matvec_q8_0_smem`, `matvec_q4_0`, and the MoE FFN batched paths with
/// the dp4a-mmvq dispatch. Default ON for production (byte-identical
/// correctness verified; +7.1% Q8 / +6.3% Q4 isolated; carries into the
/// integrated stack that delivers BF16 0.902× llama.cpp).
// mmv_q.cu kernel (sentinel: v3 with QI4_0=4, s=raw_sum).
pub const MMV_Q_DP4A_KERNEL_SOURCE: &str = include_str!("mmv_q.cu");

// mmv_q_moe.cu — Q8_0/Q4_0 batched MoE FFN matvec (gate+up+SwiGLU + down).
pub const MMV_Q_MOE_DP4A_KERNEL_SOURCE: &str = include_str!("mmv_q_moe.cu");

// mul_mat_vec_f_bf16.cu — BF16 output_proj matvec kernel
// (16.7% TPOT ncu trace). The single decisive empirical lever that clears
// the 0.9× llama.cpp gate for BF16 (5/5 trials at or above gate;
// median 91.4 = 0.902× llama.cpp). Env-gated `LUMEN_CUDA_MMV_BF16_OUTPUT_PROJ=1`,
// **default ON**; users can opt out with `=0` for A/B testing.
pub const MMV_F_BF16_KERNEL_SOURCE: &str = include_str!("mul_mat_vec_f_bf16.cu");

/// Q4_0 matrix-vector multiply kernels (matvec with on-the-fly 4-bit dequantization).
pub const MATVEC_Q4_0_KERNEL_SOURCE: &str = include_str!("matvec_q4_0.cu");

/// KV cache scatter-write kernel (head-first layout).
pub const KV_CACHE_KERNEL_SOURCE: &str = include_str!("kv_cache.cu");

/// Multi-head attention decode kernel with GQA support.
pub const ATTENTION_KERNEL_SOURCE: &str = include_str!("attention.cu");

/// Tiled streaming-softmax decode attention kernel.
///
/// Removes the single-block `attention_decode` kernel's `seq_len <= 40_950`
/// ceiling by streaming the softmax over fixed-size KV tiles using Dao 2022
/// online-softmax mechanics. Per-CTA shared memory is constant in `seq_len`
/// (~1.6 KB at T_C=128, head_dim=256); no dynamic-shmem opt-in required.
///
/// Capability-gated via `decode::attention_decode_variant`: routes to this
/// kernel when `seq_len > LUMEN_CUDA_DECODE_TILED_THRESHOLD` ( default
/// 0 = "tiled-always") OR `LUMEN_CUDA_DECODE_TILED=1` is set. Operators can
/// set `LUMEN_CUDA_DECODE_TILED_THRESHOLD=4294967295` to opt out (force
/// single-block below the 40_950 structural ceiling).
pub const ATTENTION_DECODE_TILED_KERNEL_SOURCE: &str =
    include_str!("attention_decode_tiled.cu");

/// Tiled GEMM F32 kernels for batched prefill (32x32 tiles, shared memory).
pub const GEMM_F32_KERNEL_SOURCE: &str = include_str!("gemm_f32.cu");

/// F16 matrix-vector multiply kernels (on-the-fly f16->f32 dequantization).
pub const MATVEC_F16_KERNEL_SOURCE: &str = include_str!("matvec_f16.cu");

/// BF16 matrix-vector multiply kernels (on-the-fly bf16->f32 dequantization).
///
/// BF16 has the same dynamic range as F32 (8-bit exponent) with 7 mantissa
/// bits of precision. Industry-standard precision for modern LLM inference
/// and the default for safetensors-source models. BF16 -> F32 is a 16-bit
/// left-shift; on SM_80+ the compiler emits the dedicated CVT instruction.
pub const MATVEC_BF16_KERNEL_SOURCE: &str = include_str!("matvec_bf16.cu");

/// F16 KV cache kernels (f32->f16 write, f16->f32 read).
pub const KV_CACHE_F16_KERNEL_SOURCE: &str = include_str!("kv_cache_f16.cu");

/// F32 <-> F16 bulk conversion kernels (f32_to_f16_vec, f16_to_f32_vec).
pub const CONVERT_F16_KERNEL_SOURCE: &str = include_str!("convert_f16.cu");

/// F32 -> BF16 bulk conversion kernels (f32_to_bf16_vec, f32_to_bf16_vec4).
///
/// Required by the BF16 prefill path: F32 activations are converted to BF16
/// before `cublasGemmEx` (CUDA_R_16BF) tensor-core HGEMM. Uses hardware
/// `cvt.rn.bf16.f32` on SM_80+ and a software RNE round on older targets.
pub const CONVERT_BF16_KERNEL_SOURCE: &str = include_str!("convert_bf16.cu");

/// GatedDeltaNet (GDN) kernels (conv1d, gates, L2 norm, state update).
pub const GDN_KERNEL_SOURCE: &str = include_str!("gdn.cu");

/// GDN decode megakernel: fuses 8 per-token launches into 2
/// (gdn_decode_megakernel + gdn_rmsnorm_silu_gate).
pub const GDN_MEGAKERNEL_SOURCE: &str = include_str!("gdn_megakernel.cu");

/// GDN two-launch decode kernel pair: register-resident delta-rule state
/// update (Phase 1-3 + Phase 4 split, replacing the per-token monolithic
/// kernel).
///
/// Two kernels:
///   - `gdn_phase123_register_resident`: conv1d + SiLU + gates + L2 norm (same as
///     existing megakernel Phases 1-3, but writes Q_norm/K_norm/V/alpha/beta
///     to device buffers instead of carrying them in shared memory).
///   - `gdn_phase4_register_resident`: register-resident delta-rule state update.
///     Grid (num_heads, 1, ceil(head_dim/4)). Each warp owns one column;
///     each lane keeps 4 state rows in registers (s_shard[4]).
///     Reads h_state ONCE per token, writes ONCE per token (vs Lumen's
///     existing 2R+2W per element).
///
/// Hardcoded for S_v = head_dim = 128 (Qwen3.5-9B).
/// Env gate: `LUMEN_CUDA_GDN_REGISTER_RESIDENT=1` selects this pair in place
/// of the existing `gdn_decode_megakernel` at decode dispatch.
pub const GDN_REGISTER_RESIDENT_KERNEL_SOURCE: &str = include_str!("gdn_register_resident.cu");

/// GDN F64-internal-accumulator kernels.
///
/// Per-element F32 inputs are promoted to F64 on load; reductions / state
/// arithmetic / per-token retrieval-and-update happen in F64; outputs cast
/// back to F32. This eliminates the cumulative F32-ULP non-associativity
/// drift that bisected as the source of the L0 `linear_attn_out`
/// 3.93% reference-comparison gap.
///
/// Env-gated default OFF via `LUMEN_CUDA_GDN_F64_ACCUM=1`. When the gate
/// is off the original `gdn_prefill_fused_v3` / `gdn_prefill_norm_gate`
/// / `l2_normalize_qk_strided` / `gdn_phase4_register_resident[_coal]`
/// / `gdn_rmsnorm_silu_gate` run unchanged.
pub const GDN_F64ACCUM_KERNEL_SOURCE: &str = include_str!("gdn_f64accum.cu");

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

/// Flash Attention 2 with mask block-skip and Split-K reduce (long-context prefill).
///
/// Three kernels:
///   - `flash_attention_fa2_causal`: single-kernel FA2 with per-Q-tile
///     block-skip past the causal boundary. Br=4 rows per block (one warp
///     per row), Bc=64 KV positions per tile.
///   - `flash_attention_fa2_splitk_partial`: per-split partial that emits
///     (O, m, l) for a KV slice. Single warp per (q_idx, head, split) block.
///   - `flash_attention_fa2_splitk_reduce`: merges per-split (O, m, l)
///     tuples into the final output using the FA2 online-softmax combine
///     rule (Dao et al., 2022).
///
/// The launcher selects single-kernel vs Split-K based on `seq_len`:
/// typically Split-K above ~4096 KV positions, where the per-CTA workload
/// is large enough to leave the GPU under-occupied.
pub const FLASH_ATTENTION_FA2_KERNEL_SOURCE: &str = include_str!("flash_attention_fa2.cu");

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

/// Qwen3.5 Q+gate fusion kernels (deinterleave, sigmoid_mul, per-head RMSNorm).
pub const QGATE_FUSION_KERNEL_SOURCE: &str = include_str!("qgate_fusion.cu");

// ---------------------------------------------------------------------------
// Per-row split (SoA) layout kernels and matvec_q8_aligned_q8_1_hw variant.
//
// Constants below are registered for compilation but their dispatch routing
// into the production decode path is gated behind opt-in env vars. Registering
// the sources here keeps the `.cu` files compilable, lets downstream tests
// reference them via `KERNEL_SOURCE`, and avoids stale files drifting from
// the runtime build pipeline.
// ---------------------------------------------------------------------------

/// Q8_0 split (SoA) matvec against pre-quantized Q8_1 input (dp4a, NR=2).
///
/// Consumes the per-row split layout `[scales * nb][quants[32] * nb]` produced
/// by `repack_q8_raw_to_split`. Same density as Q8Raw (34*nb bytes/row), but
/// the quants stream is contiguous and 4-byte aligned -- enabling native
/// `int*` loads instead of byte packing. Sibling of `matvec_q8_aligned_q8_1`.
///
/// Kernels: `matvec_q8_split_q8_1`, `matvec_q8_split_q8_1_residual`.
/// Requires SM 6.1+ (dp4a).
pub const MATVEC_Q8_SPLIT_Q8_1_KERNEL_SOURCE: &str =
    include_str!("matvec_q8_split_q8_1.cu");

/// Q8_0 SPLIT matvec, 4-threads-per-block thread mapping (NR=2, 128 threads, K-trip=4+).
///
/// Same byte layout as `MATVEC_Q8_SPLIT_Q8_1_KERNEL_SOURCE` but with a
/// thread->K mapping: 4 threads cooperate per Q8_0 block (vdr=2 int32s each,
/// blocks_per_iter=32). At in_dim=4096, K-loop trip count = 4 vs Lumen's
/// K-trip=1 in the production split kernel -- addresses the PTX/SASS root
/// cause of warp scheduler starvation observed at Q8 decode shapes.
///
/// Kernels: `matvec_q8_split_q8_1_4thread`, `matvec_q8_split_q8_1_4thread_residual`.
/// Env gate: `LUMEN_CUDA_Q8_SPLIT_4THREAD=1` selects these in place of the
/// production split kernels at decode dispatch.
/// Requires SM 6.1+ (dp4a).
pub const MATVEC_Q8_SPLIT_Q8_1_4THREAD_KERNEL_SOURCE: &str =
    include_str!("matvec_q8_split_q8_1_4thread.cu");

/// Q8_0 SPLIT matvec, NR=8 dp4a-mmvq variant.
///
/// Same byte layout as the SPLIT kernels but combines:
///   - 4-threads-per-block thread mapping: 4 threads/Q8_0 block, vdr=2,
///     blocks_per_iter=32, K-trip ≥ 4 at all FFN shapes (UNCHANGED from the
///     4THREAD variant)
///   -**NR = 8 rows per CTA**(UNTRIED — 4THREAD used NR=2)
///   - `__launch_bounds__(128, 1)` (5 CTAs/SM × 4 warps = 20 warps/SM)
///
/// Previous attempts used either NR=8 with Lumen's K-trip=1
/// (matvec_q8_split_output_proj_nr8 -- used only on output_proj, not FFN)
/// OR 4-threads-per-block + K-trip=4 mapping with NR=2 (matvec_q8_split_q8_1_
/// 4thread -- regressed at FFN shapes). Only NR=8 stacked with
/// 4-threads-per-block + K-trip≥4 is the untried combination on FFN shapes.
///
/// Kernels: `matvec_q8_split_q8_1_nr8`, `matvec_q8_split_q8_1_nr8_residual`.
/// Env gate: `LUMEN_CUDA_Q8_SPLIT_NR8=1` selects these in place of the
/// production split kernels at decode dispatch. Default OFF (default-off contract).
/// Requires SM 6.1+ (dp4a).
pub const MATVEC_Q8_SPLIT_Q8_1_NR8_KERNEL_SOURCE: &str =
    include_str!("matvec_q8_split_q8_1_nr8.cu");

/// Q8_0 AoS (36-byte) matvec, 4-threads-per-block + NR=8 dp4a-mmvq variant.
///
/// AoS variant of the SPLIT NR=8 dp4a-mmvq kernel. Earlier SPLIT+NR=8 benches
/// showed a -4.85% e2e regression at FFN shapes. This kernel applies the
/// dp4a-mmvq structure (4 threads/block cooperation, vdr=2, blocks_per_iter=32,
/// NR=8) onto Lumen's AoS 36-byte block layout where scales and quants are
/// colocated within the same block (vs SPLIT's `2*nb` byte gap between
/// scale and matching quants).
///
/// Hypothesis: if the FFN regression was driven by scale-stream L2 cache
/// pressure separate from the quant stream, AoS layout colocates them within
/// an L1 sector and should show a different verdict.
///
/// Kernels: `matvec_q8_aligned_nr8`, `matvec_q8_aligned_nr8_residual`.
/// Env gate: `LUMEN_CUDA_Q8_AOS_NR8=1` selects these in place of the
/// production AoS kernels at decode dispatch. Default OFF (default-off contract).
/// Mutually exclusive with `LUMEN_CUDA_Q8_SPLIT_NR8` (the SPLIT variant);
/// if both are set the AoS path wins on the AoS dispatch.
/// Requires SM 6.1+ (dp4a).
pub const MATVEC_Q8_ALIGNED_NR8_KERNEL_SOURCE: &str =
    include_str!("matvec_q8_aligned_nr8.cu");

/// Q4_0 split (SoA) matvec against pre-quantized Q8_1 input (dp4a, NR=4).
///
/// Consumes the per-row split layout `[scales * nb][nibbles[16] * nb]` produced
/// by `repack_q4_raw_to_split`. Same density as Q4Raw (18*nb bytes/row, 10%
/// denser than Q4Aligned's 20*nb), with the nibble stream contiguous and
/// 4-byte aligned for native `int*` loads.
///
/// Kernels: `matvec_q4_split_q8_1`, `matvec_q4_split_q8_1_residual`.
/// Requires SM 6.1+ (dp4a).
pub const MATVEC_Q4_SPLIT_Q8_1_KERNEL_SOURCE: &str =
    include_str!("matvec_q4_split_q8_1.cu");

/// Q8_0 split matvec dedicated to the final `output_proj` shape.
///
/// Same split layout as `MATVEC_Q8_SPLIT_Q8_1_KERNEL_SOURCE` but tuned for the
/// large `[vocab_size, hidden]` output projection (Qwen3.5-9B: 248320 x 4096).
/// Configurable NR via `MATVEC_Q8_SPLIT_OUTPUT_PROJ_NR` macro at compile time.
pub const MATVEC_Q8_SPLIT_OUTPUT_PROJ_KERNEL_SOURCE: &str =
    include_str!("matvec_q8_split_output_proj.cu");

/// One-time repack from Q8Raw (34-byte AoS) to per-row split (SoA) layout.
/// Runs once during `preload_weights`, NOT on the decode hot path.
pub const REPACK_Q8_RAW_TO_SPLIT_KERNEL_SOURCE: &str =
    include_str!("repack_q8_raw_to_split.cu");

/// One-time repack from Q4Raw (18-byte AoS) to per-row split (SoA) layout.
/// Runs once during `preload_weights`, NOT on the decode hot path.
pub const REPACK_Q4_RAW_TO_SPLIT_KERNEL_SOURCE: &str =
    include_str!("repack_q4_split.cu");

/// Q8_0 aligned matvec with halfword (16-bit) scale loads.
///
/// Structural variant of `matvec_q8_aligned_q8_1` that reads each f16 scale
/// via a single `unsigned short*` cast instead of two byte loads OR-ed
/// together. Equivalent numerically; reduces ALU per block iter by ~5 instr.
/// Selected at runtime when `LUMEN_CUDA_Q8_SCALE_HW=1` is set.
pub const MATVEC_Q8_ALIGNED_Q8_1_HW_KERNEL_SOURCE: &str =
    include_str!("matvec_q8_aligned_q8_1_hw.cu");

/// Q8_0 tile-grouped matvec against pre-quantized Q8_1 input (dp4a, NR=2).
///
/// Per tile = 8 blocks = 272 bytes (16 scales colocated with 256 quants);
/// row stride = num_tiles * 272 bytes. Targets L1-sector locality by
/// shrinking the scales-to-quants distance from `2*nb` bytes (per-row split)
/// to 16 bytes (per-tile colocation). Predicted +0.5-1.5% Q8 decode.
///
/// Kernels: `matvec_q8_tile_q8_1`, `matvec_q8_tile_q8_1_residual`.
/// Requires SM 6.1+ (dp4a) and `nb % 8 == 0` (every Qwen3.5-9B dim satisfies).
pub const MATVEC_Q8_TILE_Q8_1_KERNEL_SOURCE: &str =
    include_str!("matvec_q8_tile_q8_1.cu");

/// Q4_0 tile-grouped matvec against pre-quantized Q8_1 input (dp4a, NR=4).
///
/// Per tile = 8 blocks = 144 bytes (16 scales colocated with 128 nibbles);
/// row stride = num_tiles * 144 bytes. Adapts the Q8 tile pattern to
/// Q4_0 -- same density as Q4Split (0.5625 B/element) but with the scale
/// stream colocated within 144 contiguous bytes (vs the per-row split
/// layout where scales live `2*nb` bytes from the nibble stream).
///
/// Kernels: `matvec_q4_tile_q8_1`, `matvec_q4_tile_q8_1_residual`.
/// Requires SM 6.1+ (dp4a) and `nb % 8 == 0`.
pub const MATVEC_Q4_TILE_Q8_1_KERNEL_SOURCE: &str =
    include_str!("matvec_q4_tile_q8_1.cu");

/// One-time repack from Q4Raw (18-byte AoS) to per-tile layout (144 B / 8 blocks).
/// Runs once during `preload_weights`, NOT on the decode hot path.
pub const REPACK_Q4_TILE_KERNEL_SOURCE: &str =
    include_str!("repack_q4_tile.cu");

/// One-time repack from Q8Raw (34-byte AoS) to per-tile layout (272 B / 8 blocks).
/// Runs once during `preload_weights`, NOT on the decode hot path.
pub const REPACK_Q8_TILE_KERNEL_SOURCE: &str =
    include_str!("repack_q8_tile.cu");

/// MoE top-K router kernel.
///
/// One CTA per token; reads `router_weight` and `normed_x`, writes
/// `expert_ids[top_k]` and renormalized `expert_weights[top_k]` via fused
/// max-subtraction softmax + iterated argmax-with-mask. Numerical-stability
/// behavior is bit-equivalent to `metal/shaders/moe.msl:1-100`.
pub const MOE_ROUTER_KERNEL_SOURCE: &str = include_str!("moe_router.cu");

/// MoE weighted accumulation kernels.
///
/// Two variants:
/// - `moe_expert_accum_option_a`: dense top-K layout (per-expert path default).
/// - `moe_expert_accum_batched_b`: sparse num_experts layout (batched option).
///
/// Both compute `x[i] = residual[i] + Σ_k expert_weights[k] * expert_out[k][i]`.
pub const MOE_ACCUM_KERNEL_SOURCE: &str = include_str!("moe_accum.cu");

/// MoE batched-expert FFN kernels.
///
/// Batched-expert path — two kernels:
/// - `moe_batched_gate_up_swiglu_q8_0`: one launch processes all K active experts
///   (gridDim.y = top_k), eliminating K-fold launch overhead vs the per-expert path.
/// - `moe_batched_down_accum_q8_0`: batched down-proj + weighted accumulation
///   across all K experts in one launch.
///
/// Gated behind `LUMEN_CUDA_MOE_BATCHED=1` env var (default OFF).
/// validates correctness equivalence and measures decode delta vs per-expert.
pub const MOE_BATCHED_KERNEL_SOURCE: &str = include_str!("moe_batched.cu");

/// MoE BF16 expert FFN kernels.
///
/// BF16 port of the Q8_0 per-expert (V1) and batched (V1) MoE kernels.
/// Four kernels:
/// - `moe_expert_gate_up_swiglu_bf16` / `moe_expert_down_bf16` (reference path).
/// - `moe_batched_gate_up_swiglu_bf16` (one launch processes all K experts).
/// - `moe_batched_down_accum_bf16` (batched down + weighted accum in one launch).
///
/// BF16 weights are plain row-major (no block structure, no scales); each
/// element is 2 bytes. Conversion to F32 uses the proven `(bits << 16)`
/// bit-cast (identical to `matvec_bf16.cu`), so the dispatch is NVRTC-clean
/// on SM_70+ with no `cuda_bf16.h` requirement. Algebraically identical to
/// the Q8_0 path (same dot-product order, same SwiGLU formulation).
pub const MOE_BATCHED_BF16_KERNEL_SOURCE: &str = include_str!("moe_batched_bf16.cu");

/// MoE per-expert FFN kernels.
///
/// Two kernels:
/// - `moe_expert_gate_up_swiglu_q8_0`: gate · x + up · x + SwiGLU,
///   one launch per (expert, token) pair.
/// - `moe_expert_down_q8_0`: down · swiglu_out, one launch per (expert, token).
///
/// Reads weights from a per-layer raw byte blob (`moe_layer_blob` on
/// `LayerWeightsGpu`) at runtime-computed byte offsets supplied by the
/// caller from `CudaMoeMeta`'s per-expert tables. Bandwidth-bound; correct
/// to within Q8_0 accumulator precision vs the existing dense FFN kernels.
pub const MOE_EXPERT_KERNEL_SOURCE: &str = include_str!("moe_expert.cu");

/// MoE shared-expert auxiliary kernels.
///
/// Three small kernels that complete the shared-expert FFN dispatch (the
/// heavy gate/up/down projections reuse existing `matvec_q4_0` and
/// `swiglu_inplace` kernels):
///
/// - `moe_shared_dot_f32`: scalar F32 dot product
///   (logit = dot(ffn_gate_inp_shexp, normed_x)).
/// - `moe_shared_sigmoid_gated_accum`: x_out[i] += sigmoid(logit[0]) * shared_out[i].
/// - `moe_shared_residual_accum`: x_out[i] += shared_out[i] (fallback for
///   shared-expert variants without a gate weight).
///
/// Mirrors `metal/shaders/moe.msl::sigmoid_scale_add` + the dot kernel used
/// in `metal/moe.rs::encode_shared_expert_ffn_decode_raw`.
pub const MOE_SHARED_ACCUM_KERNEL_SOURCE: &str = include_str!("moe_shared_accum.cu");

/// MoE per-expert FFN kernels -- Q4_0 variant.
///
/// Sibling of `MOE_EXPERT_KERNEL_SOURCE` (Q8_0). Same dispatch contract
/// (one launch per (expert, token) pair), Q4_0 18-byte blocks with GGML
/// de-interleaved nibble layout. Two kernels:
/// - `moe_expert_gate_up_swiglu_q4_0`
/// - `moe_expert_down_q4_0`
pub const MOE_EXPERT_Q4_0_KERNEL_SOURCE: &str = include_str!("moe_expert_q4_0.cu");

/// MoE batched-expert FFN kernels -- Q4_0 variant.
///
/// Sibling of `MOE_BATCHED_KERNEL_SOURCE` (Q8_0). Same dispatch contract
/// (gridDim.y = top_k batches all K active experts in one launch), Q4_0
/// 18-byte blocks with GGML de-interleaved nibble layout. Four kernels:
/// - `moe_batched_gate_up_swiglu_q4_0`           (V1 simple)
/// - `moe_batched_down_accum_q4_0`               (V1 simple, fuses accum)
/// - `moe_batched_gate_up_swiglu_q4_0_v2`        (NR=2 cooperative)
/// - `moe_batched_down_v2_q4_0`                  (NR=2 cooperative)
pub const MOE_BATCHED_Q4_0_KERNEL_SOURCE: &str = include_str!("moe_batched_q4_0.cu");

/// Fused topK MoE router (sigmoid/softmax + top-K + renorm
/// + scale in ONE kernel).
///
/// Replaces the 8.7%-TPOT `moe_router_softmax_finalize_v2` second-launch with
/// a single warp-parallel kernel (target TPOT ~2.7%, was 8.7%).
///
/// Three template instantiations are exposed (`topk_moe_fused_64_no_bias`,
/// `_128_no_bias`, `_256_no_bias`). Production dispatch picks the matching
/// num_experts variant at runtime; non-power-of-two num_experts falls back to
/// the V2 path.: gated by `LUMEN_CUDA_TOPK_MOE_FUSED=1`, default ON for
/// production (broad +6-8%, no regression, 4/4 multi-prompt MATCH).
pub const TOPK_MOE_FUSED_KERNEL_SOURCE: &str = include_str!("topk_moe_fused.cu");
