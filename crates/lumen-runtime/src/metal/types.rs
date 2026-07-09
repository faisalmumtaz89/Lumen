//! Shared type definitions for the Metal backend.
//!
//! Contains GPU pipeline states, per-layer cached metadata, scratch buffers,
//! and MoE/GDN runtime types used across `mod.rs`, `gdn.rs`, and `moe.rs`.

use super::ffi::{MetalBuffer, MetalCommandBuffer, MetalPipelineState, MetalSharedEvent};
use lumen_format::quantization::QuantScheme;

// ============================================================================
// MoE (Mixture of Experts) runtime cache types
// ============================================================================
//
// ExpertSlice from lumen_format::index is used via SubtensorOffsets.experts
// field access. CachedMoeMeta is runtime-specific:
// pre-computed absolute byte offsets into the unified GPU weight buffer.

/// Cached MoE metadata for GPU-resident decode.
/// Pre-computed absolute byte offsets into the unified weight buffer.
#[derive(Debug, Clone)]
pub(crate) struct CachedMoeMeta {
    /// Router weight absolute byte offset in the unified buffer.
    pub(crate) router_weight_off: u64,
    /// Per-expert absolute byte offsets for gate/up/down projections.
    pub(crate) expert_gate_offs: Vec<u64>,
    pub(crate) expert_up_offs: Vec<u64>,
    pub(crate) expert_down_offs: Vec<u64>,
    pub(crate) expert_gate_quant: QuantScheme,
    pub(crate) expert_down_quant: QuantScheme,
}

// ============================================================================
// Pipeline states for all kernels
// ============================================================================

#[allow(dead_code)]
pub(crate) struct MetalPipelines {
    pub(crate) matmul_f32: MetalPipelineState,
    pub(crate) matmul_f32_deferred: MetalPipelineState,
    pub(crate) matmul_bytes_f32: MetalPipelineState,
    // F16 (half-precision) decode kernels — NR2 deferred reduction
    pub(crate) matmul_f16_deferred_nr2: MetalPipelineState,
    pub(crate) matmul_f16_deferred_residual_nr2: MetalPipelineState,
    pub(crate) matmul_f16_deferred_bias_nr2: MetalPipelineState,
    // BF16 (brain-float) decode kernels — NR2 deferred reduction.
    // Mirrors the F16 NR2 family for the BF16 prefill+decode foundation.
    // Same dispatch signature; only the on-device weight type
    // differs (bfloat vs half). Built unconditionally so BF16 LBC models
    // run out of the box on Apple Silicon with MSL 3.0+ bfloat support.
    pub(crate) matmul_bf16_deferred_nr2: MetalPipelineState,
    pub(crate) matmul_bf16_deferred_residual_nr2: MetalPipelineState,
    pub(crate) matmul_bf16_deferred_bias_nr2: MetalPipelineState,
    // BF16 QMV (vectorized ushort4-load) decode matvec kernels. Same dispatch
    // geometry as the deferred family, but read weights as ushort4 (8-byte
    // coalesced loads); these are the decode-path matvec kernels for BF16.
    pub(crate) matmul_bf16_qmv_nr2: MetalPipelineState,
    pub(crate) matmul_bf16_qmv_residual_nr2: MetalPipelineState,
    pub(crate) matmul_bf16_qmv_bias_nr2: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0: MetalPipelineState,
    pub(crate) rmsnorm: MetalPipelineState,
    pub(crate) rmsnorm_bytes: MetalPipelineState,
    pub(crate) rope: MetalPipelineState,
    pub(crate) rope_neox: Option<MetalPipelineState>,
    pub(crate) swiglu: MetalPipelineState,
    pub(crate) softmax: MetalPipelineState,
    pub(crate) attention_scores: MetalPipelineState,
    pub(crate) attention_output: MetalPipelineState,
    pub(crate) write_kv_cache: MetalPipelineState,
    // Fused RoPE Q + RoPE K + KV cache write (saves 2 dispatches/layer)
    pub(crate) fused_rope_kv_write: MetalPipelineState,
    pub(crate) fused_rope_kv_mha: MetalPipelineState,
    // Variant of fused_rope_kv_mha that holds the per-head attention scores in
    // threadgroup memory instead of a device scratch buffer (eliminates the
    // transient score vector's DRAM round-trips). Byte-identical math.
    pub(crate) fused_rope_kv_mha_tgscores: MetalPipelineState,
    // MMA (simdgroup_matrix) variant of the decode attention: Q.K^T + softmax +
    pub(crate) fused_rope_neox_kv_write: Option<MetalPipelineState>,
    pub(crate) multi_head_attention: MetalPipelineState,
    pub(crate) flash_decode_attention: MetalPipelineState,
    pub(crate) flash_decode_reduce: MetalPipelineState,
    pub(crate) add_residual: MetalPipelineState,
    pub(crate) embed_token: MetalPipelineState,
    pub(crate) embed_token_q8_0: MetalPipelineState,
    pub(crate) embed_token_q4_0: MetalPipelineState,
    pub(crate) embed_token_f16: MetalPipelineState,
    pub(crate) embed_token_bf16: MetalPipelineState,
    // Token-id-from-GPU-buffer embed variants for the lean GPU-pipelined greedy
    // decode (the default path). Same math as the matching embed_token_*
    // above, but read the token id from a GPU buffer so the GPU can chain
    // tokens (prior argmax -> this embed) with no CPU involvement.
    pub(crate) embed_token_bufid: MetalPipelineState,
    pub(crate) embed_token_q8_0_bufid: MetalPipelineState,
    pub(crate) embed_token_q4_0_bufid: MetalPipelineState,
    pub(crate) embed_token_f16_bufid: MetalPipelineState,
    pub(crate) embed_token_bf16_bufid: MetalPipelineState,

    // Fused kernels
    pub(crate) dequant_matmul_q8_0_residual: MetalPipelineState,

    // Multi-row decode kernels (2 rows per threadgroup, halves x-bandwidth)
    pub(crate) dequant_matmul_q8_0_multirow: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_residual_multirow: MetalPipelineState,
    // 4-row and 8-row decode kernels (128/256 threads per threadgroup)
    pub(crate) dequant_matmul_q8_0_4row: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_residual_4row: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_8row: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_residual_8row: MetalPipelineState,
    // Deferred-reduction decode kernels (NR=2 pattern, 2 sync points vs 64)
    pub(crate) dequant_matmul_q8_0_deferred: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_deferred_residual: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_deferred_bias: MetalPipelineState,
    // NR0=2 deferred variants (2 rows/TG for better occupancy on small output dims)
    pub(crate) dequant_matmul_q8_0_deferred_nr2: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_deferred_residual_nr2: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_deferred_bias_nr2: MetalPipelineState,
    // 2-SG independent row ownership (zero barriers, zero shmem)
    pub(crate) dequant_matmul_q8_0_2sg: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_2sg_residual: MetalPipelineState,
    pub(crate) ffn_fused_gate_up_swiglu_q8_0_2sg: MetalPipelineState,
    // Q4_0 decode kernels
    pub(crate) dequant_matmul_q4_0: MetalPipelineState,
    pub(crate) dequant_matmul_q4_0_residual: MetalPipelineState,
    pub(crate) dequant_matmul_q4_0_4row: MetalPipelineState,
    pub(crate) dequant_matmul_q4_0_residual_4row: MetalPipelineState,
    // Deferred-reduction Q4_0 decode kernels (same pattern as Q8_0 deferred)
    pub(crate) dequant_matmul_q4_0_deferred: MetalPipelineState,
    pub(crate) dequant_matmul_q4_0_deferred_residual: MetalPipelineState,
    pub(crate) dequant_matmul_q4_0_deferred_bias: MetalPipelineState,
    // NR0=2 deferred Q4_0 variants (2 rows/TG for better occupancy on M3 Ultra)
    pub(crate) dequant_matmul_q4_0_deferred_nr2: MetalPipelineState,
    pub(crate) dequant_matmul_q4_0_deferred_residual_nr2: MetalPipelineState,
    pub(crate) dequant_matmul_q4_0_deferred_bias_nr2: MetalPipelineState,
    // MLX-style decode GEMV (separated sequential-nibble + f32 scales layout).
    pub(crate) qmv_q4_0_residual: MetalPipelineState,
    // OPTIONAL (non-fatal): glue-side elision Wo kernel. Folds sigmoid_mul_fused +
    // qmv_q4_0_residual + residual_add_copy into one dispatch, byte-identically, on
    // the Qwen3.5 full-attn Wo path. None => the fold does not engage and the three
    // separate dispatches run.
    pub(crate) qmv_q4_0_wo_glue: Option<MetalPipelineState>,
    // OPTIONAL (non-fatal): f16-scales variant of qmv_q4_0_residual (FFN-down). A
    // missing/uncompilable kernel yields None and the FFN-down dispatch falls back
    // to the f32-scale qmv_q4_0_residual. env LUMEN_METAL_Q4_QMV_DOWN_F16SC.
    pub(crate) qmv_q4_0_residual_f16sc: Option<MetalPipelineState>,
    // OPTIONAL (non-fatal): F16-MATH variant of qmv_q4_0_residual_f16sc (FFN-down).
    // Stages x as half + accumulates the per-32-block dequant MAC in half (~2x Apple
    // OPTIONAL (non-fatal): HALF2-VECTORIZED variant of qmv_q4_0_residual_f16sc
    // (FFN-down). Stages x as half2 + accumulates the per-32-block dequant MAC in
    // half2 (TWO half FMAs / ALU slot) -> attacks the COMPUTE-bound Q4 unpack on the
    // LONGEST-K matvec (in=12288, 384 blocks/row). Twin of the gate/up and lm_head
    // h2math kernels. f32 cross-block reduction + f32 scale -> near-tie, not
    // byte-identical. A missing/uncompilable kernel yields None and the FFN-down
    // f16sc dispatch falls back to f16sc/f32. env
    // LUMEN_METAL_Q4_DOWN_H2MATH.
    pub(crate) qmv_q4_0_residual_f16sc_h2math: Option<MetalPipelineState>,
    // Two-pass deterministic SPLIT-K of qmv_q4_0_residual: pass-1 partial (K-slice
    // per grid.y) + pass-2 reduce+residual. env LUMEN_METAL_Q4_QMV_DOWN_SPLITK,
    // default OFF. Raises threadgroup concurrency for the row-starved FFN-down.
    pub(crate) qmv_q4_0_splitk_partial: MetalPipelineState,
    pub(crate) qmv_q4_0_splitk_reduce: MetalPipelineState,
    // Pass-2 reduce+SwiGLU for the dense-FFN gate/up SPLIT-K (reuses
    // qmv_q4_0_splitk_partial twice on pre-normed x). env LUMEN_METAL_Q4_GATEUP_SPLITK.
    pub(crate) gateup_splitk_reduce_swiglu: MetalPipelineState,
    // Fused K+V full-attn projection: one dispatch over [2*kv_dim] rows (doubles
    // threadgroup occupancy for the row-starved K/V matvecs). Byte-identical to the
    // two separate qmv_q4_0_rmsnorm dispatches. env LUMEN_METAL_Q4_KV_FUSE.
    pub(crate) qmv_q4_0_rmsnorm_kv: MetalPipelineState,
    // Fused Q+gate, K AND V full-attn projection: ONE dispatch over the concatenated
    // [qgate_dim + 2*kv_dim] rows (~42.7 SG/core for Qwen3.5-9B, past the occupancy
    // knee). Byte-identical to the three separate qmv_q4_0_rmsnorm dispatches.
    // env LUMEN_METAL_Q4_QGATEKV_FUSE.
    pub(crate) qmv_q4_0_rmsnorm_qgatekv: MetalPipelineState,
    // Bare single-matrix decode GEMV on PRE-normed x (64-thread; no residual, no
    // fused RMSNorm) — for the rmsnorm-once -> bare qmv gate/up -> swiglu path.
    pub(crate) qmv_q4_0: MetalPipelineState,
    // 256-thread (8 SG/TG, lane-parallel K) variant of qmv_q4_0.
    pub(crate) qmv_q4_0_8sg: MetalPipelineState,
    // MLX-style decode GEMV with fused RMSNorm (GDN qkv projection; no residual).
    pub(crate) qmv_q4_0_rmsnorm: MetalPipelineState,
    // OPTIONAL (non-fatal): f16-scales variant of qmv_q4_0_rmsnorm (lm_head + GDN
    // QKV/attn_gate). None falls back to the f32-scale qmv_q4_0_rmsnorm. env
    // LUMEN_METAL_Q4_LMHEAD_F16SC / LUMEN_METAL_Q4_PROJ_F16SC.
    pub(crate) qmv_q4_0_rmsnorm_f16sc: Option<MetalPipelineState>,
    // HALF2-VECTORIZED single-matrix rmsnorm variant (half2 dequant MAC); used by
    // the lm_head dispatch. None falls back to the f16math/f16sc/f32 path.
    // env LUMEN_METAL_Q4_LMHEAD_H2MATH (default OFF).
    pub(crate) qmv_q4_0_rmsnorm_f16sc_h2math: Option<MetalPipelineState>,
    // A/B EXPERIMENT: llama.cpp lane->block mapping variant of qmv_q4_0_rmsnorm
    // (same buffers/geometry/RMSNorm, only the coalescing pattern differs).
    // env LUMEN_METAL_Q4_QMV_PROJ_LCMAP (default OFF).
    pub(crate) qmv_q4_0_rmsnorm_llamacpp: MetalPipelineState,
    // MLX-style DUAL-matrix decode GEMV: fused RMSNorm + gate/up + SwiGLU
    // (dense FFN gate/up; env LUMEN_METAL_Q4_QMV_GATEUP).
    pub(crate) qmv_q4_0_gate_up_swiglu: MetalPipelineState,
    // OPTIONAL (non-fatal): f16-scales variant of qmv_q4_0_gate_up_swiglu. None
    // falls back to the f32-scale dual-matrix kernel. env LUMEN_METAL_Q4_GATEUP_F16SC.
    pub(crate) qmv_q4_0_gate_up_swiglu_f16sc: Option<MetalPipelineState>,
    // OPTIONAL (non-fatal): 1-simdgroup-per-TG variant of the f16-scales gate/up
    // kernel. Byte-identical math; ONLY the geometry differs (1 SG/TG, 4 rows/TG,
    // inter_dim/4 TGs = 2x more threadgroups for deeper wavefront latency hiding).
    // None falls back to the 2-SG f16sc/f32 path. env LUMEN_METAL_Q4_GATEUP_1SG.
    pub(crate) qmv_q4_0_gate_up_swiglu_f16sc_1sg: Option<MetalPipelineState>,
    // OPTIONAL (non-fatal): 8-rows-per-SG variant of the f16-scales gate/up kernel.
    // Byte-identical math (per output row the FP add order + simd_sum are unchanged);
    // ONLY the geometry differs (2 SG/TG, 8 rows/SG, inter_dim/16 TGs = HALF the
    // threadgroups, 2x x-register reuse to lift arithmetic intensity per fetched
    // activation byte). None falls back to the 4-row f16sc path. env
    // LUMEN_METAL_Q4_GATEUP_8ROW.
    pub(crate) qmv_q4_0_gate_up_swiglu_f16sc_8row: Option<MetalPipelineState>,
    // OPTIONAL (non-fatal): F16-MATH variant of the f16-scales gate/up kernel.
    // Same 2 SG/TG, 4 rows/SG, inter_dim/8 TGs geometry + same bindings; the per-
    // 32-block dequant MAC runs in `half` (~2x Apple GPU half ALU) while the cross-
    // block reduction / sum-of-x / scale / RMSNorm / SwiGLU stay f32. Near-tie (not
    // guaranteed byte-identical); attacks the COMPUTE half of the dominant FFN
    // matvec. None falls back to the f16sc/8row/1sg path. env
    // LUMEN_METAL_Q4_GATEUP_F16MATH.
    pub(crate) qmv_q4_0_gate_up_swiglu_f16sc_f16math: Option<MetalPipelineState>,
    // OPTIONAL (non-fatal): HALF2-VECTORIZED variant of the f16-scales gate/up
    // kernel. Same 2 SG/TG, 4 rows/SG, inter_dim/8 TGs geometry + same bindings as
    // the f16math kernel; the per-32-block dequant MAC accumulates in `half2` (two
    // half FMAs per Apple GPU vector ALU slot) instead of scalar `half`, halving
    // the dequant-MAC instruction count AGAIN on the dominant FFN matvec. Near-tie
    // (the half-lane partial-sum grouping differs from the scalar half kernel; the
    // cross-block reduction / sumx / scale / RMSNorm / SwiGLU stay f32). None falls
    // back to the f16math/f16sc/8row/1sg path. env LUMEN_METAL_Q4_GATEUP_H2MATH.
    pub(crate) qmv_q4_0_gate_up_swiglu_f16sc_h2math: Option<MetalPipelineState>,
    // OPTIONAL (non-fatal): INTERLEAVED gate+up variant. Byte-identical math; reads
    // ONE co-resident packed nibble buffer + ONE packed f16-scale buffer instead of
    // four separate streams (gate/up nibbles + gate/up scales). None falls back to
    // the f16sc/8row/default path. env LUMEN_METAL_Q4_GATEUP_IL.
    pub(crate) qmv_q4_0_gate_up_swiglu_il: Option<MetalPipelineState>,
    // OPTIONAL (non-fatal): LM-head-structure (LS) single-stream gate+up variant.
    // Byte-identical math to the h2math dual kernel; reads a ROW-INTERLEAVED
    // gate|up buffer single-stream at 2*inter_dim/8 TGs (lm_head structure)
    // instead of the DUAL gate+up stream at inter_dim/8 TGs. None falls back to
    // the h2math/default path.
    pub(crate) qmv_q4_0_gate_up_swiglu_ls_h2math: Option<MetalPipelineState>,
    // WIDE-load (uint4/256-thread) variant of the dense FFN gate/up GEMV; reads
    // the same separated sequential-nibble layout (env LUMEN_METAL_Q4_GATEUP_WIDE).
    pub(crate) rmsnorm_ffn_gate_up_swiglu_q4_0_wide: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0_residual_batched: MetalPipelineState,
    // Q4_0 batched prefill kernels
    pub(crate) dequant_tiled_matmul_q4_0: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_residual_batched: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_splitk: MetalPipelineState,
    // Q4_1 kernels
    pub(crate) dequant_tiled_matmul_q4_1: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_1_residual_batched: MetalPipelineState,
    pub(crate) dequant_matmul_q4_1_deferred: MetalPipelineState, // decode matvec
    pub(crate) tiled_matmul_bytes_f32_residual: MetalPipelineState,
    pub(crate) tiled_matmul_f16: MetalPipelineState,
    pub(crate) tiled_matmul_f16_residual: MetalPipelineState,
    pub(crate) tiled_matmul_f16_k64: MetalPipelineState,
    pub(crate) tiled_matmul_f16_k64_residual: MetalPipelineState,
    // BF16 prefill GEMM kernels.
    // Same simdgroup MMA tile geometry as F16 (32x32 output tile, 128
    // threads, 4 simdgroups), but uses `simdgroup_bfloat8x8` on M3+
    // (Apple GPU family 9). F32 accumulators preserve precision through
    // the inner loop; output written F32 for downstream consumers.
    pub(crate) tiled_matmul_bf16: MetalPipelineState,
    pub(crate) tiled_matmul_bf16_residual: MetalPipelineState,
    pub(crate) tiled_matmul_bf16_k64: MetalPipelineState,
    pub(crate) tiled_matmul_bf16_k64_residual: MetalPipelineState,
    /// BF16 GDN qkv-proj + attn-gate-proj paired GEMM. Consumes a
    /// runtime-repacked concat-then-stripe weight buffer (`repack_bf16.rs`).
    /// Single dispatch, dual-output (Y_qkv, Y_gate). BC-pipeline variant
    /// (boundary-checked) since M may not be a clean multiple of TILE_M.
    pub(crate) tiled_matmul_bf16_k64_qkv_gate_paired: MetalPipelineState,
    /// Aligned variant (FC_BC_{M,N,K}=false) for the common case when
    /// M is a multiple of TILE_M, both projection N dims are multiples of
    /// TILE_N=32, and hidden_dim is a multiple of TILE_K_64=64 (Qwen3.5-9B
    /// GDN: M=131 may misalign but N=8192/4096 and K=4096 are aligned —
    /// the BC variant is used in practice when M is misaligned).
    pub(crate) tiled_matmul_bf16_k64_qkv_gate_paired_aligned: MetalPipelineState,
    /// Minimal warmup kernel for the BF16 GDN paired repack
    /// buffer. Touches one BF16 element per layer at load time so that the
    /// Apple Metal driver commits the GPU page-table mapping for the 96 MB
    /// packed buffer upfront, not at first-prefill time. Cost: ~1µs per
    /// layer; the alternative is ~280 ms on the cold first prefill per the
    /// diagnostic.
    pub(crate) bf16_paired_warmup: MetalPipelineState,
    pub(crate) matmul_bytes_f32_residual: MetalPipelineState,

    // Buffer ops (GPU-side activation transfer)
    pub(crate) copy_buffer: MetalPipelineState,
    pub(crate) add_write: MetalPipelineState,

    // Split-K GEMM kernels (for GPU core saturation during small prefill)
    pub(crate) dequant_tiled_matmul_q8_0_splitk: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0_k64_splitk: MetalPipelineState,
    pub(crate) reduce_splitk: MetalPipelineState,
    pub(crate) reduce_splitk_add_residual: MetalPipelineState,

    // K64 (TILE_K=64) GEMM variants for fewer barriers
    pub(crate) dequant_tiled_matmul_q8_0_k64: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0_k64_residual_batched: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_k64: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_k64_residual_batched: MetalPipelineState,

    // Joint dual-output gate+up GEMM with register-resident SwiGLU.
    // Replaces 2 separate dispatches (gate, up) + 1 swiglu_batched dispatch.
    pub(crate) dequant_tiled_matmul_q8_0_gate_up_swiglu_fused: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0_gate_up_swiglu_fused_aligned: MetalPipelineState,

    // packed-layout Q8_0 kernels. Consume runtime-repacked stripe SoA
    // weight buffers (see `metal/repack_q8.rs`). Both BC and aligned variants
    // are registered so the dispatch site can pick the fast path when M/N/K
    // are aligned to the tile dimensions.
    pub(crate) dequant_tiled_matmul_q8_0_k64_residual_batched_packed: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0_k64_residual_batched_packed_aligned: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0_gate_up_swiglu_fused_packed: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0_gate_up_swiglu_fused_packed_aligned: MetalPipelineState,

    // Q4_0 port of the fused gate+up+SwiGLU kernel.
    // Same kernel structure; Q4_0 de-interleaved nibble dequant path.
    pub(crate) dequant_tiled_matmul_q4_0_gate_up_swiglu_fused: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_gate_up_swiglu_fused_aligned: MetalPipelineState,

    // packed-layout Q4_0 kernels. Consume runtime-repacked stripe SoA
    // weight buffers (see `metal/repack_q4.rs`). Both BC and aligned variants
    // are registered so the dispatch site can pick the fast path when M/N/K
    // are aligned to the tile dimensions. Q4 port of.
    pub(crate) dequant_tiled_matmul_q4_0_k64_residual_batched_packed: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_k64_residual_batched_packed_aligned: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_gate_up_swiglu_fused_packed: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_gate_up_swiglu_fused_packed_aligned: MetalPipelineState,

    // ggml-metal ported Q8_0 GEMM (gated by LUMEN_METAL_GEMM_GGML_PORT=1).
    // Adapted from ggml-org/ggml `kernel_mul_mm_q8_0_f32` — MIT.
    // See `shaders/gemm_q8_0_ported.msl`.
    pub(crate) kernel_mul_mm_q8_0_f32_ported: MetalPipelineState,

    // Function-constant-specialized GEMM variants (BC_M=false, BC_N=false, BC_K=false).
    // Used when M, N, K are all aligned to tile dimensions, eliminating all
    // boundary checks in the inner loop via dead-code elimination.
    pub(crate) dequant_tiled_matmul_q8_0_aligned: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0_k64_aligned: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0_k64_residual_batched_aligned: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0_residual_batched_aligned: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_k64_aligned: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_k64_residual_batched_aligned: MetalPipelineState,
    pub(crate) tiled_matmul_f16_k64_aligned: MetalPipelineState,
    pub(crate) tiled_matmul_f16_k64_residual_aligned: MetalPipelineState,
    // BF16 aligned (FC_BC_*=false) variants for prefill GEMM on
    // aligned dimensions. Dead-code-eliminates all per-element boundary
    // checks for the typical Qwen3.5-9B dims (hidden=4096, ffn=12288).
    pub(crate) tiled_matmul_bf16_k64_aligned: MetalPipelineState,
    pub(crate) tiled_matmul_bf16_k64_residual_aligned: MetalPipelineState,

    // BF16 port of the fused gate+up+SwiGLU kernel.
    // Same kernel structure as the Q8 variant but operates on BF16 weights
    // via simdgroup_bfloat8x8 MMA. Eliminates the 3-dispatch
    // (gate / up / swiglu_batched) chain on the BF16 FFN prefill path.
    pub(crate) bf16_matmul_gate_up_swiglu_fused: MetalPipelineState,
    pub(crate) bf16_matmul_gate_up_swiglu_fused_aligned: MetalPipelineState,

    // NR microtile sweep variants of the BF16 fused gate+up+SwiGLU
    // kernel. NR refers to the number of 8x8 simdgroup MMA accumulator rows
    // owned per simdgroup along the M axis. Baseline above is NR=2 (mc[2][2]).
    //   - NR=1 (TILE_M=16): shmem 10 KB, 3 TG/CU at M3 knee. Probes lower
    //     M-tile / higher occupancy.
    //   - NR=4 (TILE_M=64): shmem 16 KB, 1 TG/CU at M3 knee. Probes higher
    //     M-tile / better weight-load amortisation.
    // Selected at dispatch time via `LUMEN_METAL_BF16_GATE_UP_NR=<1|2|4>` env
    // var (default 2 = baseline). Each variant has BC + aligned function-const
    // pipelines mirroring the baseline.
    pub(crate) bf16_matmul_gate_up_swiglu_fused_nr1: MetalPipelineState,
    pub(crate) bf16_matmul_gate_up_swiglu_fused_nr1_aligned: MetalPipelineState,
    pub(crate) bf16_matmul_gate_up_swiglu_fused_nr4: MetalPipelineState,
    pub(crate) bf16_matmul_gate_up_swiglu_fused_nr4_aligned: MetalPipelineState,

    // BF16 K64 Split-K GEMM. Pairs with the existing
    // reduce_splitk_add_residual (which is quant-agnostic and operates on F32
    // partials) to deliver a Split-K FFN-down path for BF16. Same dispatch
    // pattern as the Q8 variant; only the B-tile load differs (u16->bfloat).
    pub(crate) bf16_matmul_k64_splitk: MetalPipelineState,
    pub(crate) bf16_matmul_k64_splitk_aligned: MetalPipelineState,

    // Batched prefill kernels
    pub(crate) tiled_matmul_f32: MetalPipelineState,
    pub(crate) tiled_matmul_bytes_f32: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0: MetalPipelineState,
    pub(crate) rmsnorm_batched: MetalPipelineState,
    pub(crate) rmsnorm_batched_bytes: MetalPipelineState,
    pub(crate) rope_batched: MetalPipelineState,
    pub(crate) rope_batched_neox: Option<MetalPipelineState>,
    pub(crate) add_residual_batched: MetalPipelineState,
    /// Determinism diagnostic: zero a half buffer (scores-buffer clear).
    pub(crate) memset_half_zero: MetalPipelineState,
    pub(crate) swiglu_batched: MetalPipelineState,
    pub(crate) embed_tokens_batched: MetalPipelineState,
    pub(crate) embed_tokens_batched_q8_0: MetalPipelineState,
    pub(crate) embed_tokens_batched_q4_0: MetalPipelineState,
    pub(crate) embed_tokens_batched_f16: MetalPipelineState,
    pub(crate) embed_tokens_batched_bf16: MetalPipelineState,
    pub(crate) kv_cache_write_batched: MetalPipelineState,
    pub(crate) v_cache_write_batched: MetalPipelineState,
    pub(crate) attention_scores_batched: MetalPipelineState,
    pub(crate) softmax_batched: MetalPipelineState,
    pub(crate) attention_output_batched: MetalPipelineState,
    pub(crate) attention_scores_tiled: MetalPipelineState,
    pub(crate) attention_output_tiled: MetalPipelineState,

    // Fused RMSNorm + Q8_0 matvec NR2 (eliminates separate RMSNorm dispatch)
    pub(crate) rmsnorm_dequant_matmul_q8_0_deferred_nr2: MetalPipelineState,
    pub(crate) rmsnorm_dequant_matmul_q8_0_deferred_residual_nr2: MetalPipelineState,
    // Fused RMSNorm + Q4_0 matvec NR2 (eliminates separate RMSNorm dispatch)
    pub(crate) rmsnorm_dequant_matmul_q4_0_deferred_nr2: MetalPipelineState,
    pub(crate) rmsnorm_dequant_matmul_q4_0_deferred_residual_nr2: MetalPipelineState,
    // Fused RMSNorm + F16 matvec NR2 (eliminates separate RMSNorm dispatch)
    pub(crate) rmsnorm_matmul_f16_deferred_nr2: MetalPipelineState,
    pub(crate) rmsnorm_matmul_f16_deferred_residual_nr2: MetalPipelineState,
    // Fused RMSNorm + BF16 matvec NR2. Same dispatch shape as the
    // F16 fused variant; weights read as bfloat instead of half.
    pub(crate) rmsnorm_matmul_bf16_deferred_nr2: MetalPipelineState,
    pub(crate) rmsnorm_matmul_bf16_deferred_residual_nr2: MetalPipelineState,
    // Vectorized-load (ushort4) fused RMSNorm + BF16 matvec (decode path).
    pub(crate) rmsnorm_matmul_bf16_qmv_nr2: MetalPipelineState,
    // Fused RMSNorm + FFN Gate+Up+SwiGLU Q8_0 deferred
    pub(crate) rmsnorm_ffn_fused_gate_up_swiglu_q8_0_deferred: MetalPipelineState,
    // Fused RMSNorm + FFN Gate+Up+SwiGLU Q8_0 8-row (8 SGs, zero barriers)
    pub(crate) rmsnorm_ffn_fused_gate_up_swiglu_q8_0_8row: MetalPipelineState,
    // Fused RMSNorm + FFN Gate+Up+SwiGLU Q4_0 deferred
    pub(crate) rmsnorm_ffn_fused_gate_up_swiglu_q4_0_deferred: MetalPipelineState,
    // Fused RMSNorm + FFN Gate+Up+SwiGLU Q4_0 8-row (8 SGs, zero barriers)
    pub(crate) rmsnorm_ffn_fused_gate_up_swiglu_q4_0_8row: MetalPipelineState,
    // Fused RMSNorm + FFN Gate+Up+SwiGLU F16 deferred
    pub(crate) rmsnorm_ffn_fused_gate_up_swiglu_f16_deferred: MetalPipelineState,

    // FFN fused gate+up+swiglu kernel (decode only)
    pub(crate) ffn_fused_gate_up_swiglu_q8_0: MetalPipelineState,
    pub(crate) ffn_fused_gate_up_swiglu_q8_0_deferred: MetalPipelineState,
    pub(crate) ffn_fused_gate_up_swiglu_q4_0: MetalPipelineState,
    pub(crate) ffn_fused_gate_up_swiglu_q4_0_deferred: MetalPipelineState,
    pub(crate) ffn_fused_gate_up_swiglu_q4_1_deferred: MetalPipelineState,
    pub(crate) ffn_fused_gate_up_swiglu_f16_deferred: MetalPipelineState,

    // GPU-side argmax for greedy decode (eliminates 128KB logits readback)
    pub(crate) argmax: MetalPipelineState,

    // Two-pass tiled GPU argmax (the greedy-decode token-selection path). Fills
    // the machine with N threadgroups instead of the single-TG `argmax` (which is
    // bandwidth-starved at ~2.9 GB/s over the vocab logits). Bit-identical token
    // selection to `argmax` for every input (see ffn_elementwise.msl). Pass-1
    // writes per-tile (max_val, arg_idx) partials; pass-2 reduces them to the token.
    pub(crate) argmax_tiled_partial: MetalPipelineState,
    pub(crate) argmax_tiled_reduce: MetalPipelineState,

    // GPU-side temperature sampler (Option A: lean-sampled decode path,
    // LUMEN_METAL_GPU_SAMPLER=1, default OFF). Parity-matched to the CPU
    // `sample_logits`; finalizes a sampled token on-GPU so temp>0 decode can
    // pipeline exactly like the greedy argmax chain. None when the kernel
    // failed to compile (then the path is unavailable and we keep the CPU
    // sampler).
    pub(crate) gpu_sampler: Option<MetalPipelineState>,
    // Latency-hiding variant of `gpu_sampler` (the DEFAULT for the GPU-sampler
    // path): hides device-memory latency on the two O(vocab) reductions with all
    // 256 threads (hierarchical block sum + bounded within-block walk) instead of
    // a single serial thread. Parity-faithful (same RNG/penalty/exp; block-sum
    // re-association is the same O(1e-6) class as the exp difference, verified by
    // the parity test). The exact single-thread `gpu_sampler` is used only when
    // LUMEN_METAL_GPU_SAMPLER_EXACT=1 (validation).
    pub(crate) gpu_sampler_fast: Option<MetalPipelineState>,

    // QKV bias addition (Qwen2-family models)
    pub(crate) bias_add: MetalPipelineState,
    pub(crate) bias_add_batched: MetalPipelineState,

    // Fused QKV deinterleave (splits [M][qkv_dim] -> Q, K, V buffers)
    pub(crate) deinterleave_qkv: MetalPipelineState,

    // MoE (Mixture of Experts) pipeline states.
    // Option to allow graceful error messages if shader compilation fails.
    // The MoE kernels are included in METAL_SHADER_SOURCE.
    pub(crate) moe_router_softmax: Option<MetalPipelineState>,
    // Parallel router — per-expert logits + small top-k softmax.
    pub(crate) moe_router_logits_f32: Option<MetalPipelineState>,
    pub(crate) moe_router_topk_softmax: Option<MetalPipelineState>,
    // Fused single-dispatch router (logits + top-k, grid=experts,
    // last-TG reduction) — eliminates the 1-TG top-k drain bubble on decode.
    pub(crate) moe_router_fused_topk: Option<MetalPipelineState>,
    pub(crate) moe_router_softmax_batched: Option<MetalPipelineState>,
    pub(crate) moe_router_softmax_biased: Option<MetalPipelineState>,
    pub(crate) moe_expert_accum: Option<MetalPipelineState>,
    pub(crate) moe_expert_accum_batched: Option<MetalPipelineState>,
    pub(crate) moe_expert_accum_option_a: Option<MetalPipelineState>,
    // Expert-grouped prefill MoE (mul_mat_id style) index/copy kernels.
    pub(crate) moe_prefill_route_sort: Option<MetalPipelineState>,
    pub(crate) moe_prefill_route_sort_par: Option<MetalPipelineState>,
    pub(crate) moe_prefill_route_sort_atomic: Option<MetalPipelineState>,
    pub(crate) moe_prefill_gather: Option<MetalPipelineState>,
    pub(crate) moe_prefill_gather_vec4: Option<MetalPipelineState>,
    pub(crate) moe_prefill_scatter_vec4: Option<MetalPipelineState>,
    pub(crate) moe_prefill_scatter: Option<MetalPipelineState>,
    pub(crate) moe_prefill_assign_expert: Option<MetalPipelineState>,
    pub(crate) moe_grouped_gemm_q8_0: Option<MetalPipelineState>,
    pub(crate) moe_grouped_gemm_q8_0_tilemap: Option<MetalPipelineState>,
    pub(crate) moe_grouped_gemm_q4_0_tilemap: Option<MetalPipelineState>,
    pub(crate) moe_prefill_build_tile_map: Option<MetalPipelineState>,
    // Batched MoE expert FFN — GPU-side routing, no CPU readback.
    pub(crate) moe_batched_gate_up_swiglu_q4_0: Option<MetalPipelineState>,
    pub(crate) moe_batched_gate_up_swiglu_q4_1: Option<MetalPipelineState>,
    pub(crate) moe_batched_gate_up_swiglu_q8_0: Option<MetalPipelineState>,
    pub(crate) moe_batched_down_accum_q4_0: Option<MetalPipelineState>,
    pub(crate) moe_batched_down_accum_q4_1: Option<MetalPipelineState>,
    pub(crate) moe_batched_down_accum_q8_0: Option<MetalPipelineState>,
    // One-simdgroup-per-row redesign of routed gate+up+swiglu (q8).
    pub(crate) moe_batched_gate_up_swiglu_q8_0_v2: Option<MetalPipelineState>,
    // Fused down+accum+shared_expert kernels (eliminates 3 dispatches per MoE layer)
    pub(crate) moe_batched_down_accum_shared_q8_0: Option<MetalPipelineState>,
    pub(crate) moe_batched_down_accum_shared_q8_0_se_q4_0: Option<MetalPipelineState>,
    // One-simdgroup-per-row redesign of the mixed q8/q4 down kernel.
    pub(crate) moe_batched_down_accum_shared_q8_0_se_q4_0_v2: Option<MetalPipelineState>,
    pub(crate) moe_batched_down_accum_shared_q4_0: Option<MetalPipelineState>,
    pub(crate) sigmoid_scale_add: Option<MetalPipelineState>,

    // GatedDeltaNet (linear attention) pipeline states for Qwen3.5-35B-A3B.
    // Option to allow graceful startup when model does not use delta net layers.
    pub(crate) ssm_conv1d_decode: Option<MetalPipelineState>,
    pub(crate) l2_normalize_heads: Option<MetalPipelineState>,
    pub(crate) sigmoid_gate: Option<MetalPipelineState>,
    pub(crate) silu_elementwise_mul: Option<MetalPipelineState>,
    pub(crate) gated_delta_net_state_update: Option<MetalPipelineState>,
    pub(crate) gated_delta_net_output: Option<MetalPipelineState>,

    // Additional GDN pipeline states for full forward pass.
    pub(crate) gated_delta_net_state_update_v2: Option<MetalPipelineState>,
    pub(crate) gdn_compute_gates: Option<MetalPipelineState>,
    pub(crate) elementwise_mul_f32: Option<MetalPipelineState>,
    pub(crate) ssm_l2_norm_scale: Option<MetalPipelineState>,

    // Fused element-wise kernels for GDN dispatch reduction.
    pub(crate) sigmoid_mul_fused: Option<MetalPipelineState>,
    pub(crate) residual_add_copy: Option<MetalPipelineState>,
    pub(crate) l2_normalize_qk: Option<MetalPipelineState>,

    // SiLU activation (in-place) for post-conv1d GDN activation.
    pub(crate) silu_inplace: Option<MetalPipelineState>,
    // Fused Conv1D + SiLU for GDN decode (eliminates 1 dispatch + 1 barrier per layer).
    pub(crate) ssm_conv1d_silu_decode: Option<MetalPipelineState>,

    // Q+gate de-interleave for Qwen3.5 full-attention layers.
    pub(crate) deinterleave_qgate: Option<MetalPipelineState>,
    // Per-head RMSNorm for Q and K (Qwen3.5 full-attention layers).
    pub(crate) rmsnorm_per_head: Option<MetalPipelineState>,
    // Sigmoid-scale for shared expert gating.
    pub(crate) sigmoid_scale_buffer: Option<MetalPipelineState>,
    // Batched sigmoid-scale-add for shared expert gating during prefill.
    pub(crate) sigmoid_scale_add_batched: Option<MetalPipelineState>,

    // Fused GDN mega-kernels for further dispatch reduction.
    pub(crate) gdn_state_output_norm: Option<MetalPipelineState>,
    pub(crate) dequant_matmul_q8_0_deferred_residual_copy: Option<MetalPipelineState>,
    pub(crate) dequant_matmul_q8_0_deferred_residual_copy_nr2: Option<MetalPipelineState>,
    pub(crate) dequant_matmul_q4_0_deferred_residual_copy: Option<MetalPipelineState>,
    pub(crate) dequant_matmul_q4_0_deferred_residual_copy_nr2: Option<MetalPipelineState>,

    // Fused deinterleave+norm+assemble for full-attention Q+gate layers
    pub(crate) deinterleave_norm_assemble: Option<MetalPipelineState>,
    // Full-attention bookend elision: deinterleave+norm+rope+kv-write folded into
    // one dispatch. None => the six-dispatch incumbent bookend runs.
    pub(crate) deinterleave_norm_rope_kvwrite: Option<MetalPipelineState>,
    // Fused L2-normalize + state-update + output + RMSNorm (eliminates l2_normalize_qk dispatch)
    pub(crate) gdn_state_output_norm_l2: Option<MetalPipelineState>,
    // Simdgroup-parallel state update (4096 TGs of 32 threads, writes raw output)
    pub(crate) gdn_state_output_l2_sg: Option<MetalPipelineState>,
    // Same as gdn_state_output_l2_sg but read-once/write-once (dead store removed)
    pub(crate) gdn_state_output_l2_sg_h1: Option<MetalPipelineState>,
    // Diagnostic (timing only): same as gdn_state_output_l2_sg but the per-TG
    // Q/K L2-norm is skipped (output garbage) to isolate its compute cost
    pub(crate) gdn_state_output_l2_sg_normskip: Option<MetalPipelineState>,
    // Diagnostic (timing only): same grid/loads as gdn_state_output_l2_sg but the
    // ENTIRE recurrence (decay/retrieval/delta/update/output/state-writes/simd_sums)
    // is removed (output garbage) to isolate the full recurrence GPU cost
    pub(crate) gdn_state_output_l2_sg_recurskip: Option<MetalPipelineState>,
    // Same as gdn_state_output_l2_sg but persistent h_state stored in bfloat (half traffic)
    pub(crate) gdn_state_output_l2_sg_bf16: Option<MetalPipelineState>,
    // Same as gdn_state_output_l2_sg but persistent h_state stored in half (half traffic)
    pub(crate) gdn_state_output_l2_sg_f16: Option<MetalPipelineState>,
    // F16 state recurrence WITHOUT the dead decayed write-back (LUMEN_METAL_GDN_F16_STATE_H1):
    // union of the f16-state (half R+W) + h1 dead-store-elision (drop the redundant decayed store).
    pub(crate) gdn_state_output_l2_sg_f16_h1: Option<MetalPipelineState>,
    // VI-amortized f16+h1 recurrence (LUMEN_METAL_GDN_F16_STATE_H1_V2): each TG handles
    // 2 adjacent val_dim columns, computing the (vi-invariant) Q/K L2-norm + load ONCE and
    // reusing across both -> halves the redundant per-vi norm ALU + Q/K device reads on the
    // recurrence critical path. Byte-identical to gdn_state_output_l2_sg_f16_h1 (same reduction).
    pub(crate) gdn_state_output_l2_sg_f16_h1_v2: Option<MetalPipelineState>,
    // 4-way VI-amortized f16+h1 recurrence (LUMEN_METAL_GDN_F16_STATE_H1_V4): each TG handles
    // 4 adjacent val_dim columns, computing the (vi-invariant) Q/K L2-norm + load ONCE and
    // reusing across all four -> cuts the redundant per-vi norm ALU + Q/K device reads 4x vs
    // reference (2x vs v2). Byte-identical to gdn_state_output_l2_sg_f16_h1 (same reduction).
    pub(crate) gdn_state_output_l2_sg_f16_h1_v4: Option<MetalPipelineState>,
    // One-time F32->F16 converter for the GDN h_state buffer (LUMEN_METAL_GDN_F16_STATE_DECODE)
    pub(crate) gdn_state_f32_to_f16: Option<MetalPipelineState>,
    // RMSNorm + scale on raw GDN decode output (pairs with gdn_state_output_l2_sg)
    pub(crate) gdn_decode_norm_scale: Option<MetalPipelineState>,
    // Fused Conv1D+SiLU + L2-normalize + state-update + output + RMSNorm (eliminates conv1d dispatch + barrier)
    pub(crate) gdn_state_output_norm_l2_conv: Option<MetalPipelineState>,
    // Full GDN decode megakernel: Conv1D+SiLU + inline gates + L2 + state + output + RMSNorm
    pub(crate) gdn_decode_megakernel: Option<MetalPipelineState>,
    // Fused SiLU-gated Q8_0 matvec + residual + copy (eliminates silu_elementwise_mul dispatch)
    pub(crate) dequant_matmul_q8_0_silu_deferred_residual_copy_nr2: Option<MetalPipelineState>,
    // Fused SiLU-gated Q4_0 matvec + residual + copy (eliminates silu_elementwise_mul dispatch)
    pub(crate) dequant_matmul_q4_0_silu_deferred_residual_copy_nr2: Option<MetalPipelineState>,
    // Fused dual alpha+beta RMSNorm+matvec+gates for GDN decode (eliminates 2 dispatches + 1 barrier)
    pub(crate) dequant_matmul_q8_0_dual_gates_nr2: Option<MetalPipelineState>,

    // Batched GDN prefill kernels
    pub(crate) gdn_prefill_state_output_norm: Option<MetalPipelineState>,
    pub(crate) gdn_prefill_fused: Option<MetalPipelineState>,
    pub(crate) gdn_prefill_fused_v2: Option<MetalPipelineState>,
    pub(crate) gdn_prefill_fused_v3_chunked: Option<MetalPipelineState>,
    /// (32, NSG=4, 1) threadgroup geometry for Phase 2a — 1024 TGs
    /// of 128 threads (4 simdgroups per TG) instead of 4096 TGs of 32
    /// threads. Algorithmically bit-identical to `gdn_prefill_fused_v3_
    /// chunked`. Each TG owns 4 consecutive rows of state and the 4
    /// simdgroups share Q/K HBM fetches via L1. Opt-in via
    /// `LUMEN_METAL_GDN_PHASE2A_NSG4=1`.
    pub(crate) gdn_prefill_fused_v3_chunked_nsg4: Option<MetalPipelineState>,
    /// Chunk-parallel gated-delta-rule Phase 2a. One TG per
    /// (head, 32-value-tile); builds the per-chunk T/H matrices once and
    /// forward-solves over C tokens, cutting the O(T)-serial recurrence to
    /// O(T/C) serial chunks. Opt-in via `LUMEN_METAL_GDN_PREFILL_CHUNKED=1`.
    pub(crate) gdn_prefill_chunkscan: Option<MetalPipelineState>,
    pub(crate) gdn_prefill_norm_gate: Option<MetalPipelineState>,
    pub(crate) ssm_conv1d_prefill: Option<MetalPipelineState>,
    pub(crate) ssm_conv1d_silu_prefill: Option<MetalPipelineState>,
    pub(crate) ssm_conv1d_silu_prefill_parallel: Option<MetalPipelineState>,
    /// Determinism fix: race-free conv_state update, dispatched after the
    /// token-parallel conv1d compute (separated by a barrier).
    pub(crate) ssm_conv1d_state_update: Option<MetalPipelineState>,
    pub(crate) l2_normalize_heads_batched: Option<MetalPipelineState>,
    pub(crate) l2_normalize_qk_strided: Option<MetalPipelineState>,
    pub(crate) l2_normalize_qk_strided_sg: Option<MetalPipelineState>,
    /// Fused conv1d+SiLU+L2 for Q/K (collapses the conv->L2 spine).
    pub(crate) conv1d_silu_l2_qk_fused: Option<MetalPipelineState>,
    /// Conv1d+SiLU for the V channel sub-range (no L2).
    pub(crate) conv1d_silu_vrange: Option<MetalPipelineState>,
    pub(crate) gdn_compute_gates_batched: Option<MetalPipelineState>,
    pub(crate) dequant_batched_matvec_q8_0: Option<MetalPipelineState>,
    pub(crate) dequant_batched_matvec_q8_0_dual: Option<MetalPipelineState>,
}

impl MetalPipelines {
    /// BF16 decode matvec (plain / bias-less). Returns the vectorized ushort4-load
    /// (QMV) kernel: reading weights 4-at-a-time coalesces the loads and better
    /// saturates Apple GPU bandwidth than the scalar 2-byte-per-weight path, with
    /// identical NR0=2 / 128-thread dispatch geometry.
    #[inline]
    pub(crate) fn bf16_matvec_nr2(&self) -> &MetalPipelineState {
        &self.matmul_bf16_qmv_nr2
    }

    /// BF16 decode matvec + residual add.
    #[inline]
    pub(crate) fn bf16_matvec_residual_nr2(&self) -> &MetalPipelineState {
        &self.matmul_bf16_qmv_residual_nr2
    }

    /// BF16 decode matvec + fused QKV bias.
    #[inline]
    pub(crate) fn bf16_matvec_bias_nr2(&self) -> &MetalPipelineState {
        &self.matmul_bf16_qmv_bias_nr2
    }

    /// Fused RMSNorm + BF16 decode matvec.
    #[inline]
    pub(crate) fn bf16_rmsnorm_matvec_nr2(&self) -> &MetalPipelineState {
        &self.rmsnorm_matmul_bf16_qmv_nr2
    }
}

// ============================================================================
// Cached per-layer metadata (avoids WeightProvider calls in GPU-resident decode)
// ============================================================================

/// Pre-computed subtensor offsets and quantization schemes for one layer.
/// Populated once during `preload_weights_gpu_resident` so that
/// `decode_token_single_cb` can skip the `begin_pass`/`try_get_layer`
/// loop entirely -- eliminating 22 x LayerView allocations per token.
pub(crate) struct CachedLayerMeta {
    pub(crate) attn_norm_off: u64,
    pub(crate) wq_off: u64,
    pub(crate) wo_off: u64,
    pub(crate) ffn_norm_off: u64,
    pub(crate) w_gate_off: u64,
    pub(crate) w_up_off: u64,
    pub(crate) w_down_off: u64,
    pub(crate) wq_quant: QuantScheme,
    pub(crate) wo_quant: QuantScheme,
    pub(crate) w_gate_quant: QuantScheme,
    pub(crate) w_up_quant: QuantScheme,
    pub(crate) w_down_quant: QuantScheme,
    // Optional QKV bias offsets (Qwen2-family models).
    // When Some, the bias_add kernel is dispatched after the QKV projection.
    pub(crate) bq_off: Option<u64>,
    pub(crate) bk_off: Option<u64>,
    pub(crate) bv_off: Option<u64>,

    // MoE (Mixture of Experts) cached metadata.
    // When Some, this layer uses MoE FFN instead of the dense FFN path.
    // Populated from SubtensorOffsets.router_weight and SubtensorOffsets.experts.
    pub(crate) moe_meta: Option<CachedMoeMeta>,

    // -- Shared expert offsets (Qwen3.5-MoE) --
    // When Some, this MoE layer has an always-active shared expert whose output
    // is added to the routed expert output before the residual connection.
    pub(crate) shared_expert_gate_off: Option<u64>,
    pub(crate) shared_expert_up_off: Option<u64>,
    pub(crate) shared_expert_down_off: Option<u64>,
    pub(crate) shared_expert_gate_quant: Option<QuantScheme>,
    pub(crate) shared_expert_down_quant: Option<QuantScheme>,

    // -- Extended attention fields (Qwen3.5-MoE hybrid layers) --
    // attn_gate: element-wise gating with SiLU on attention output (full attention layers).
    // attn_post_norm: RMSNorm after Wo projection, before attn_gate and residual.
    pub(crate) attn_gate_off: Option<u64>,
    pub(crate) attn_gate_quant: Option<QuantScheme>,
    pub(crate) attn_post_norm_off: Option<u64>,

    // -- Separate K/V weight offsets (Qwen3.5 full-attention layers) --
    // For full-attention layers where Q+gate are fused in wq (attn_q.weight produces
    // q_dim+q_dim outputs = Q +), K and V must be projected separately.
    // When has_qgate_fusion is true:
    //   - wq_off points to attn_q.weight (output dim = 2*q_dim = Q +)
    //   - wk_off/wv_off point to separate attn_k.weight / attn_v.weight
    //   - The decode path projects Q+gate, K, V separately and applies sigmoid gate.
    pub(crate) has_qgate_fusion: bool,
    pub(crate) wk_off: Option<u64>,
    pub(crate) wv_off: Option<u64>,
    pub(crate) wk_quant: Option<QuantScheme>,
    pub(crate) wv_quant: Option<QuantScheme>,
    // Per-head Q and K RMSNorm weights (Qwen3.5 full-attention layers).
    // Shape: [head_dim] F32, shared across all heads.
    pub(crate) attn_q_norm_off: Option<u64>,
    pub(crate) attn_k_norm_off: Option<u64>,
    // Shared expert gate input weight: sigmoid(dot(ffn_gate_inp_shexp, input)) gates shared expert output.
    // Shape: [hidden_dim] F32.
    pub(crate) ffn_gate_inp_shexp_off: Option<u64>,

    // -- Layer type discriminator --
    // 0 = full attention (standard transformer), 1 = linear attention (GatedDeltaNet).
    // None for models that don't have hybrid layer types.
    pub(crate) layer_type: Option<u8>,

    // -- GatedDeltaNet (linear attention) cached offsets --
    // Populated for GDN layers (layer_type=1). None for full attention layers.
    pub(crate) ssm_a_off: Option<u64>,
    pub(crate) ssm_conv1d_off: Option<u64>,
    pub(crate) ssm_dt_off: Option<u64>,
    pub(crate) ssm_beta_off: Option<u64>,
    pub(crate) ssm_alpha_off: Option<u64>,
    pub(crate) ssm_norm_off: Option<u64>,
    pub(crate) ssm_out_off: Option<u64>,
    pub(crate) ssm_out_quant: Option<QuantScheme>,
    /// Index into gdn_h_states/gdn_conv_states vectors for this GDN layer.
    /// None for full attention layers. Sequential 0, 1, 2, ... for GDN layers.
    pub(crate) gdn_layer_idx: Option<usize>,
}

// ============================================================================
// Router diagnostics
// ============================================================================

/// Per-layer routing statistics from a single decode token.
/// Captures expert_ids and expert_weights for diagnostic analysis.
///
/// The `weight_spread` field (top1_weight - top2_weight) diagnoses
/// degenerate routing caused by near-uniform softmax output. When spread < 0.01,
/// routing is effectively random and the strict `>` argmax tiebreaker always
/// picks expert 0.
pub struct RouterLayerStats {
    pub layer: usize,
    pub expert_ids: Vec<u32>,
    pub expert_weights: Vec<f32>,
    /// Difference between top-1 and top-2 softmax weights.
    /// Near-zero spread indicates the router cannot distinguish experts.
    pub weight_spread: f32,
}

// ============================================================================
// Scratch buffers (GPU-resident, reused across calls)
// ============================================================================

#[allow(dead_code)]
pub(crate) struct MetalScratch {
    // Persistent activation buffer: reused across layers via write_f32.
    // Allocated once in init() for [hidden_dim] floats.
    pub(crate) x_buf: MetalBuffer,

    // Activation buffers
    pub(crate) normed_buf: MetalBuffer,
    // Fused QKV output buffer: [q_dim + kv_dim + kv_dim] floats for decode.
    // Q at byte offset 0, K at q_dim*4, V at (q_dim+kv_dim)*4.
    pub(crate) qkv_buf: MetalBuffer,
    pub(crate) q_buf: MetalBuffer,
    pub(crate) k_buf: MetalBuffer,
    pub(crate) v_buf: MetalBuffer,
    pub(crate) attn_out_buf: MetalBuffer,
    pub(crate) scores_buf: MetalBuffer,
    pub(crate) attn_proj_buf: MetalBuffer,
    pub(crate) gate_buf: MetalBuffer,
    pub(crate) up_buf: MetalBuffer,
    pub(crate) down_buf: MetalBuffer,
    /// SPLIT-K pass-1 partials scratch [hidden_dim * MAX_K_SPLITS(8)] f32, reused
    /// by the FFN-down two-pass SPLIT-K kernel (env LUMEN_METAL_Q4_QMV_DOWN_SPLITK).
    pub(crate) splitk_partials_buf: MetalBuffer,
    /// gate/up SPLIT-K partials scratch: gate [inter*MAX_K_SPLITS(8)] then up
    /// [inter*8] f32 (one buffer, up offset = inter*8*4 bytes). env
    /// LUMEN_METAL_Q4_GATEUP_SPLITK. + normed-x scratch [hidden] (the pre-pass RMSNorm output).
    pub(crate) splitk_gateup_partials_buf: MetalBuffer,
    pub(crate) splitk_normed_buf: MetalBuffer,
    pub(crate) logits_buf: MetalBuffer,
    /// Second logits buffer for the deferred-async-commit ("Option B") decode
    /// prototype (env `LUMEN_METAL_DECODE_ASYNC_COMMIT=1`, default OFF). When the
    /// async-commit path defers token N's wait+readback to the start of token
    /// N+1's call, N+1's lm_head must write to a DIFFERENT logits buffer than the
    /// one still holding N's un-read logits — otherwise N+1 clobbers N before the
    /// CPU sampler consumes it. The two buffers are ping-ponged by token parity.
    /// `None` until the async-commit path first allocates it (lazy; the default
    /// synchronous path never touches it). Lazily sized to `logits_buf`.
    pub(crate) logits_buf_b: Option<MetalBuffer>,

    // GPU-side argmax result: 1 x u32 (4 bytes). Eliminates 128KB logits readback
    // for greedy sampling (temperature <= 0).
    pub(crate) argmax_result_buf: MetalBuffer,

    // Two-pass tiled-argmax pass-1 partials scratch.
    // Layout: [ARGMAX_MAX_TILES] f32 max-vals at byte 0, then [ARGMAX_MAX_TILES]
    // u32 arg-idxs at byte ARGMAX_MAX_TILES*4. Sized once for the max tile count;
    // pass-1 writes the first `num_tiles` entries, pass-2 reduces them.
    pub(crate) argmax_partials_buf: MetalBuffer,

    // ---- Lean GPU-pipelined greedy decode (the default greedy path) ----
    // A small ring of CPU-visible u32 token buffers used to chain tokens on the
    // GPU while overlapping the CPU-encode of token N+1 with the GPU-execute of
    // token N. For command buffer index k (0-based decode step within a pipeline
    // run): the embed reads `pipe_token_ring[k % R]` and the argmax writes
    // `pipe_token_ring[(k+1) % R]`. The CPU reads `pipe_token_ring[(k+1) % R]`
    // for the token CB(k) produced. R is chosen large enough that a slot is
    // never overwritten before BOTH its GPU consumer (the next embed) and its
    // CPU reader have used it (see decode_greedy_pipelined). Allocated lazily on
    // first pipelined call. Empty in the default sequential path.
    pub(crate) pipe_token_ring: Vec<MetalBuffer>,
    // Per-in-flight-CB pipeline state, FIFO. Each entry is (command_buffer,
    // step_index, seq_pos) for a CB that has been committed (async) but whose
    // output token has not yet been read back on the CPU. Drained in order.
    pub(crate) pipe_inflight: std::collections::VecDeque<(MetalCommandBuffer, usize, usize)>,
    // [metal-R9 pos79 probe] Split-CB staging (LUMEN_METAL_SPLIT_CB_AT_ORD). When
    // the probe flag is set, decode_token_greedy_core commits CB1 (first half,
    // encoders ord 0..N) mid-token and stashes it in `pipe_split_stage`; the lean
    // driver moves it into `pipe_split_inflight` in lockstep with `pipe_inflight`
    // (one Option per in-flight token) so BOTH halves' GPUStart/End timestamps can
    // be read after the token's terminal CB2 completes. Empty/None unless the
    // probe flag is set (byte-identical single-CB behavior otherwise).
    pub(crate) pipe_split_stage: Option<MetalCommandBuffer>,
    pub(crate) pipe_split_inflight: std::collections::VecDeque<Option<MetalCommandBuffer>>,
    // Monotonic decode-step counter for the current pipeline run; reset when the
    // pipeline is (re)started (i.e. when pipe_inflight is empty at entry).
    pub(crate) pipe_step: usize,
    // Absolute sequence position the NEXT pipelined CB will write KV / apply
    // RoPE for. Seeded from `kv.seq_len()` when a run starts and incremented per
    // ENCODED CB (so it leads the CPU `kv.seq_len()` by the in-flight depth).
    pub(crate) pipe_seq_pos: usize,
    // GPU->GPU ordering event for the pipelined decode. CB(k) signals
    // `pipe_event_base + k + 1` at its end; CB(k+1) waits for that value at its
    // start, forcing CB(k) to fully complete (KV/h_state/token-slot writes
    // visible) before CB(k+1) begins -- i.e. completion ordering even if the
    // single queue would otherwise allow same-queue command buffers to overlap.
    // Allocated lazily with the ring. `pipe_event_base` is bumped past the last
    // used value whenever a fresh pipeline run starts so values stay monotonic
    // across runs (MTLSharedEvent values must only increase).
    pub(crate) pipe_event: Option<MetalSharedEvent>,
    pub(crate) pipe_event_base: u64,

    // ---- GPU temperature sampler (Option A, LUMEN_METAL_GPU_SAMPLER=1) ----
    // RNG-state ring, parallel to `pipe_token_ring`. Each entry is one u64
    // (8 bytes) xorshift64 state. CB(step) reads `gpu_sampler_rng_ring[step%R]`
    // and writes the once-advanced state into `[(step+1)%R]` (the sampler kernel
    // performs exactly one next_u64 per token, so the state advances by exactly
    // one draw per emitted token -> draw-count parity with the CPU sampler).
    // Seeded at run start with `Xorshift64::new(seed)`'s post-finalizer state.
    // Same ring-size as the token ring so a slot is never overwritten before its
    // GPU consumer + CPU reader are done. Empty until the GPU-sampler path runs.
    pub(crate) gpu_sampler_rng_ring: Vec<MetalBuffer>,
    // Persistent GPU history frequency array: [vocab] u32, one occurrence count
    // per token id over the FULL history (prompt + generated). The sampler
    // kernel reads it to apply per-unique-token penalties (bit-identical to the
    // CPU full-history freq map) and, after selecting a token, atomically
    // increments freq_arr[sel] so the next pipelined CB sees it -- the GPU-side
    // analogue of SamplerState. Seeded at run start with the prompt's token
    // counts (zeroed first). Allocated lazily on the first GPU-sampler call;
    // unused (None) in every other decode path.
    pub(crate) gpu_sampler_freq_arr: Option<MetalBuffer>,

    // RoPE cos/sin tables
    pub(crate) rope_cos_buf: MetalBuffer,
    pub(crate) rope_sin_buf: MetalBuffer,

    // GPU-resident KV cache: persistent buffers sized for max_seq_len.
    // Indexed by layer. Each buffer holds [max_seq_len * kv_dim] floats.
    // K,V projections are written directly here via `write_kv_cache` kernel,
    // eliminating the CPU<->GPU round-trip per token.
    pub(crate) gpu_k_cache: Vec<MetalBuffer>,
    pub(crate) gpu_v_cache: Vec<MetalBuffer>,

    // Multi-head attention scratch: [num_heads * max_seq_len] floats.
    // Used by multi_head_attention kernel when seq_len > 4096 (threadgroup
    // memory limit). For seq_len <= 4096, the kernel uses threadgroup memory
    // and this buffer is unused.
    pub(crate) mha_scores_buf: MetalBuffer,

    // Flash decode partial results buffer:
    // [num_heads * max_kv_tiles * (head_dim + 2)] floats.
    // Each tile writes head_dim floats of weighted V, plus max and sum.
    pub(crate) flash_decode_partial_buf: MetalBuffer,

    // Model dimensions (computed once in init)
    pub(crate) hidden_dim: usize,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) num_layers: usize,
    pub(crate) head_dim: usize,
    pub(crate) inter_dim: usize,
    pub(crate) eps: f32,
    pub(crate) q_dim: usize,
    pub(crate) kv_dim: usize,
    pub(crate) qkv_dim: usize, // q_dim + 2 * kv_dim (for fused QKV projection)
    pub(crate) gqa_ratio: usize,
    pub(crate) vocab_size: usize,
    pub(crate) half_dim: usize,
    pub(crate) max_seq_len: usize,
    pub(crate) attn_scale: f32,

    // Threadgroup configuration
    pub(crate) matmul_tg_size: u64,
    pub(crate) norm_tg_size: u64,
    pub(crate) mha_tg_size: u64,

    // GPU activation state: when true, x_buf already contains valid data
    // from the previous layer, so we can skip the CPU→GPU upload.
    pub(crate) gpu_x_valid: bool,
    /// Last async command buffer — waited on at start of new forward pass
    /// to ensure previous pass's GPU work completes before CPU writes to buffers.
    pub(crate) last_async_cmd: Option<MetalCommandBuffer>,

    /// Option-B async-commit decode bookkeeping. `async_inflight_logits_b` is
    /// true when the in-flight (not-yet-waited) decode CB wrote its logits to
    /// `logits_buf_b` (false => `logits_buf`); the next call reads from that
    /// buffer after waiting `last_async_cmd`, then writes THIS token's logits to
    /// the OTHER buffer. Default sync path never reads these.
    pub(crate) async_inflight_logits_b: bool,

    /// Cached per-layer zero-copy Metal buffers (avoid re-creating on every call).
    /// Indexed by layer_idx. Populated lazily on first access per layer.
    pub(crate) layer_buf_cache: Vec<Option<(usize, MetalBuffer)>>, // (ptr, buffer)

    /// Cached partial (non-expert) Metal buffers for MoE streaming.
    /// For MoE layers with expert caching active, this stores a smaller buffer
    /// covering only attention+norm+router data, avoiding page-faults on the
    /// expert byte range in the mmap'd layer blob.
    /// Indexed by layer_idx. Stores (blob_ptr, non_expert_end_bytes, buffer).
    pub(crate) moe_partial_buf_cache: Vec<Option<(usize, usize, MetalBuffer)>>,

    /// GPU-resident weight buffers: persistent Metal buffers pre-loaded at init.
    /// When populated, these bypass the mmap zero-copy path entirely.
    /// Each buffer contains a full copy of one layer's weight data in
    /// Metal-managed memory, eliminating TLB misses and page table walks.
    pub(crate) gpu_resident_layers: Option<Vec<MetalBuffer>>,

    /// Single contiguous buffer holding ALL layer weights + global tensors (GPU-resident mode).
    /// Uses StorageModePrivate for GPU memory controller optimizations.
    /// Reduces TLB pressure from 22+ separate virtual address ranges to 1.
    pub(crate) gpu_unified_weight_buf: Option<MetalBuffer>,
    /// Per-layer base offset into the unified buffer (page-aligned).
    pub(crate) gpu_layer_offsets: Vec<usize>,
    /// Global tensor offsets into the unified buffer: (embed_offset, norm_offset, output_proj_offset).
    pub(crate) gpu_global_offsets: Option<(usize, usize, usize)>,

    // Batched prefill scratch buffers (allocated for max_batch_size)
    // These are Option so they can be lazily initialized when prefill is first called.
    pub(crate) batch_x_buf: Option<MetalBuffer>, // [batch, hidden_dim]
    pub(crate) batch_normed_buf: Option<MetalBuffer>, // [batch, hidden_dim]
    pub(crate) batch_qkv_buf: Option<MetalBuffer>, // [batch, qkv_dim] fused QKV output
    pub(crate) batch_q_buf: Option<MetalBuffer>, // [batch, q_dim]
    pub(crate) batch_k_buf: Option<MetalBuffer>, // [batch, kv_dim]
    pub(crate) batch_v_buf: Option<MetalBuffer>, // [batch, kv_dim]
    pub(crate) batch_attn_out_buf: Option<MetalBuffer>, // [batch, q_dim]
    pub(crate) batch_attn_proj_buf: Option<MetalBuffer>, // [batch, hidden_dim]
    pub(crate) batch_gate_buf: Option<MetalBuffer>, // [batch, inter_dim]
    pub(crate) batch_up_buf: Option<MetalBuffer>, // [batch, inter_dim]
    pub(crate) batch_down_buf: Option<MetalBuffer>, // [batch, hidden_dim]
    pub(crate) batch_scores_buf: Option<MetalBuffer>, // [batch, num_heads, max_seq_len]
    pub(crate) splitk_partial_buf: Option<MetalBuffer>, // [K_SPLITS * max_M * max_N] floats for Split-K

    // ====================================================================
    // de-aliased GDN scratch buffers
    // ====================================================================
    // The legacy GDN prefill (`encode_batched_gdn_prefill`) packs four
    // semantic roles into `batch_qkv_buf` (Phase 1 QKV output, Phase 2a
    // raw_out write, Phase 2b ssm_in write, Phase 3 ssm_in read) and three
    // roles into the alpha/beta/conv_out scratch slice. Apple's hazard
    // tracker on `MTLDispatchTypeConcurrent` is whole-MTLBuffer granularity
    // any consumer of one role inherits the
    // producer-retirement stall of all roles on the same buffer.
    //
    // the path splits the multi-role buffers into separate MTLBuffers
    // so resource-scoped barriers (`memoryBarrierWithResources:`) can scope
    // each barrier to just the buffer the next phase actually reads. The
    // four scratch buffers below cover the GDN dispatch chain; legacy
    // `batch_qkv_buf` continues to hold the Phase 1 QKV GEMM output (the
    // sole role on `qkv_buf` once de-aliased).
    //
    // Allocated only when the GDN concurrent-encoder path is enabled
    // (`LUMEN_METAL_GDN_CONCURRENT_ENCODER=1`). Default OFF preserves legacy behaviour.
    pub(crate) batch_gdn_raw_out_buf: Option<MetalBuffer>, // [batch * q_dim] Phase 2a state-update output
    pub(crate) batch_gdn_ssm_in_buf: Option<MetalBuffer>, // [batch * q_dim] Phase 2b ssm_in / Phase 3 input
    pub(crate) batch_gdn_alpha_buf: Option<MetalBuffer>, // [batch * num_heads] alpha gate (Phase 1 -> Phase 2a)
    pub(crate) batch_gdn_beta_buf: Option<MetalBuffer>, // [batch * num_heads] beta gate (Phase 1 -> Phase 2a)
    pub(crate) batch_gdn_conv_out_buf: Option<MetalBuffer>, // [batch * qkv_dim] post-conv1d SiLU+L2-normalized QKV

    pub(crate) splitk_alloc_elems: usize, // tracks allocated Split-K buffer capacity (in floats)
    pub(crate) current_max_batch: usize,  // tracks allocated batch size

    /// Pre-allocated logits readback buffer: [vocab_size] floats.
    /// Reused every decode token to avoid a 128 KB heap allocation per token.
    pub(crate) logits_readback: Vec<f32>,

    /// Cached per-layer subtensor metadata for GPU-resident decode.
    /// Populated in `preload_weights_gpu_resident`; when non-empty,
    /// `decode_token_single_cb` skips all WeightProvider interaction.
    pub(crate) cached_layer_meta: Vec<CachedLayerMeta>,

    // ====================================================================
    // MoE (Mixture of Experts) scratch buffers and parameters
    // ====================================================================
    // Only allocated when the model has num_experts > 0.
    /// Number of experts in the MoE layer (e.g., 8 for Mixtral).
    /// 0 for dense models.
    pub(crate) moe_num_experts: usize,
    /// Number of active (top-K selected) experts per token (e.g., 2 for Mixtral).
    /// 0 for dense models.
    pub(crate) moe_num_active_experts: usize,
    /// Per-expert intermediate dimension. Same as inter_dim for uniform MoE.
    pub(crate) moe_expert_inter_dim: usize,

    // Decode scratch buffers (single token)
    /// Router logits: [num_experts] f32 -- output of router matmul
    pub(crate) moe_router_logits: Option<MetalBuffer>,
    /// Grid-wide finish counter (atomic_uint, 1 elem) for the
    /// fused single-dispatch router's last-threadgroup reduction. Zeroed each
    /// layer before the dispatch.
    pub(crate) moe_router_counter: Option<MetalBuffer>,
    /// Selected expert IDs after top-K: [top_k] u32
    pub(crate) moe_expert_ids: Option<MetalBuffer>,
    /// Routing weights for selected experts: [top_k] f32
    pub(crate) moe_expert_weights: Option<MetalBuffer>,
    /// Per-expert FFN output: [num_experts * hidden_dim] f32
    /// Option B: we dispatch ALL experts and rely on zero-weight for non-selected.
    pub(crate) moe_expert_output: Option<MetalBuffer>,

    // Prefill scratch buffers (batched)
    /// Batched router logits: [max_batch * num_experts] f32
    pub(crate) moe_batch_router_logits: Option<MetalBuffer>,
    /// Batched expert IDs: [max_batch * top_k] u32
    pub(crate) moe_batch_expert_ids: Option<MetalBuffer>,
    /// Batched routing weights: [max_batch * top_k] f32
    pub(crate) moe_batch_expert_weights: Option<MetalBuffer>,
    /// Batched per-expert FFN output: [max_batch * num_experts * hidden_dim] f32
    pub(crate) moe_batch_expert_output: Option<MetalBuffer>,

    // Expert-grouped prefill MoE scratch (mul_mat_id style).
    // Sized for max_batch * top_k assignments. Far smaller than the dense
    // [batch * num_experts * hidden] expert_output (which we still write into for
    // the byte-identical accum).
    /// Per-expert segment offsets (prefix sums): [num_experts+1] u32.
    pub(crate) moe_grp_seg_off: Option<MetalBuffer>,
    /// Token index per assignment, grouped by expert: [max_batch * top_k] u32.
    pub(crate) moe_grp_tok: Option<MetalBuffer>,
    /// Slot (k) per assignment: [max_batch * top_k] u32.
    pub(crate) moe_grp_slot: Option<MetalBuffer>,
    /// Expert id per assignment (expanded from seg_off): [max_batch * top_k] u32.
    pub(crate) moe_grp_assign_expert: Option<MetalBuffer>,
    /// Gathered per-assignment normed input: [max_batch * top_k * hidden] f32.
    pub(crate) moe_grp_in: Option<MetalBuffer>,
    /// Per-assignment SwiGLU activation: [max_batch * top_k * inter] f32.
    pub(crate) moe_grp_swiglu: Option<MetalBuffer>,
    /// Per-assignment down output (pre-scatter): [max_batch * top_k * hidden] f32.
    pub(crate) moe_grp_down: Option<MetalBuffer>,
    /// Flattened grouped-GEMM work-tile map. Entry 0 =
    /// n_work_tiles; entries 1.. = packed (expert<<16)|m_tile_local. Lets the
    /// grouped GEMM dispatch exactly the non-empty M-tiles instead of
    /// max_m_tiles*num_experts (19.5x over-subscription at batch=1239/256e).
    pub(crate) moe_grp_tile_map: Option<MetalBuffer>,

    /// Per-layer expert IDs buffers for GPU-resident decode profiling.
    /// When allocated, each MoE layer writes its top-K expert selections to
    /// a dedicated buffer so all layers can be read back after a single
    /// commit_and_wait(). One buffer per model layer (None entry for dense layers).
    /// Avoids the problem where a shared expert_ids buffer gets overwritten by
    /// each successive MoE layer in a single command buffer.
    pub(crate) moe_per_layer_expert_ids: Vec<Option<MetalBuffer>>,

    /// Per-layer expert weights buffers for router diagnostics.
    /// When allocated, each MoE layer writes its top-K expert routing weights
    /// to a dedicated buffer (same pattern as per_layer_expert_ids).
    /// Only populated when router_debug is enabled on the backend.
    pub(crate) moe_per_layer_expert_weights: Vec<Option<MetalBuffer>>,

    // Batched MoE expert FFN offset tables (GPU buffers).
    // Per-MoE-layer: gate+up offset table [num_experts * 2] u64
    pub(crate) moe_gate_up_offsets: Vec<Option<MetalBuffer>>,
    // Per-MoE-layer: down offset table [num_experts] u64
    pub(crate) moe_down_offsets: Vec<Option<MetalBuffer>>,
    // Scratch for batched swiglu output: [top_k * inter_dim] f32
    pub(crate) moe_batched_swiglu_buf: Option<MetalBuffer>,
    // Per-MoE-layer: shared expert down weight byte offset [1] u64
    pub(crate) moe_shared_down_offsets: Vec<Option<MetalBuffer>>,
    // Shared expert gating scalar buffer [1] f32 (reused across layers)
    pub(crate) moe_shared_gate_scalar_buf: Option<MetalBuffer>,

    // ====================================================================
    // Qwen3.5-MoE scratch
    // ====================================================================
    /// RoPE theta value from hyperparams. Used for RoPE table recomputation in gpu_resident.
    pub(crate) rope_theta: f64,
    /// True when the model uses NeoX-style half-split RoPE (e.g. Qwen2, Qwen3.5).
    /// Set from hp.rope_neox in init(). Used for all RoPE dispatch site selection.
    pub(crate) rope_neox: bool,
    /// Effective rotary dimension (partial RoPE). Equals head_dim for standard models,
    /// head_dim/4 for Qwen3.5-MoE (partial_rotary_factor=0.25).
    pub(crate) rotary_dim: usize,
    /// Shared expert intermediate dimension (512 for Qwen3.5-35B-A3B).
    pub(crate) shared_expert_inter_dim: usize,
    /// Scratch buffer for shared expert fused gate+up output: [shared_expert_inter_dim] f32.
    pub(crate) shared_expert_gate_buf: Option<MetalBuffer>,
    /// Scratch buffer for shared expert down projection output: [hidden_dim] f32.
    pub(crate) shared_expert_down_buf: Option<MetalBuffer>,
    /// Scratch buffer for attention output gate logits: [hidden_dim] f32.
    pub(crate) attn_gate_buf: Option<MetalBuffer>,

    // ====================================================================
    // GatedDeltaNet (linear attention) persistent state
    // ====================================================================
    /// Persistent GDN recurrent state: one buffer per GDN layer.
    /// Shape: [num_heads, head_dim, head_dim] f32 per layer (head_dim x head_dim matrix per head).
    /// These MUST persist across tokens and be reset between sequences.
    pub(crate) gdn_h_states: Vec<MetalBuffer>,
    /// Lazily-allocated half-size F16 mirror of each `gdn_h_states` entry, used
    /// ONLY when `LUMEN_METAL_GDN_F16_STATE_DECODE=1`. `None` until the first
    /// decode touch of that GDN layer converts its prefill F32 state into F16
    /// (via `gdn_state_f32_to_f16`); thereafter the F16 decode recurrence
    /// reads/writes this buffer and the F32 `gdn_h_states` entry is dormant.
    /// Reset (cleared) between sequences alongside `gdn_h_states`.
    /// `RefCell` so the lazy first-touch convert can populate it through the
    /// shared `&MetalScratch` borrow held across the decode layer loop (the loop
    /// already holds immutable borrows of `gpu_resident_layers`/`cached_layer_meta`).
    pub(crate) gdn_h_states_f16: Vec<std::cell::RefCell<Option<MetalBuffer>>>,
    /// Circular buffer for GDN conv1d: one per GDN layer.
    /// Shape: [(kernel_size - 1) * conv_dim] f32 per layer.
    /// conv_dim = num_kv_heads * head_dim * 2 (k and v concatenated).
    pub(crate) gdn_conv_states: Vec<MetalBuffer>,
    /// Current write position in each GDN conv circular buffer [0..kernel_size-2].
    pub(crate) gdn_conv_positions: Vec<u32>,
    /// GDN scratch: alpha (decay) per head [num_heads] f32.
    pub(crate) gdn_alpha_buf: Option<MetalBuffer>,
    /// GDN scratch: beta (mixing rate) per head [num_heads] f32.
    pub(crate) gdn_beta_buf: Option<MetalBuffer>,
    /// GDN scratch: output of state query [num_heads * head_dim] f32 (= q_dim).
    pub(crate) gdn_output_buf: Option<MetalBuffer>,
    /// GDN scratch: ssm output projection result [hidden_dim] f32.
    pub(crate) gdn_ssm_proj_buf: Option<MetalBuffer>,
    /// GDN scratch: attention gate sigmoid output [hidden_dim] f32.
    pub(crate) gdn_gate_sigmoid_buf: Option<MetalBuffer>,
    /// GDN scratch: ssm_norm scaled output [GDN_NUM_HEADS * GDN_HEAD_DIM] f32 (= 4096).
    pub(crate) gdn_normed_out_buf: Option<MetalBuffer>,
    /// GDN scratch: Q8_0 matvec output for alpha gates [GDN_NUM_HEADS] f32.
    pub(crate) gdn_alpha_raw_buf: Option<MetalBuffer>,
    /// GDN scratch: Q8_0 matvec output for beta gates [GDN_NUM_HEADS] f32.
    pub(crate) gdn_beta_raw_buf: Option<MetalBuffer>,
    /// GDN scratch: conv1d output for all QKV channels [GDN_QKV_DIM=8192] f32.
    pub(crate) gdn_qkv_conv_buf: Option<MetalBuffer>,
    /// GDN conv kernel size (`ssm.conv_kernel`, typically 4).
    pub(crate) gdn_conv_kernel_size: usize,
    /// GDN number of state / V heads (`ssm.time_step_rank`): 32 for Qwen3.5-9B,
    /// 48 for Qwen3.6-27B. Resolved from `hyperparams.gdn_dims()` at init.
    pub(crate) gdn_num_v_heads: usize,
    /// GDN number of Q/K heads before GQA repeat (`ssm.group_count`, 16).
    pub(crate) gdn_num_k_heads: usize,
    /// GDN per-head dimension (`ssm.state_size`, 128).
    pub(crate) gdn_head_dim: usize,
    /// Number of GDN layers in the model (layer_type=1 count).
    pub(crate) gdn_num_layers: usize,
    /// Maps layer_idx -> gdn_idx for streaming path lazy allocation.
    /// Empty until first GDN layer encountered in compute_layer.
    pub(crate) gdn_layer_idx_map: Vec<Option<usize>>,

    // ========================================================================
    // Runtime Q8_0 hot-weight repack storage.
    // ========================================================================
    //
    // When `LUMEN_METAL_Q8_REPACKED=1` is set at load time, the following
    // buffers hold repacked copies of hot FFN tensors in a stripe SoA layout
    // (see `repack_q8.rs` for the exact byte layout). The repacked kernels
    // (`*_packed`) consume these buffers; the original buffers + AoS kernels
    // remain available as a fallback.
    //
    // Indexed by layer_idx. `None` entries indicate the layer is not eligible
    // (e.g. quant != Q8_0, or repack disabled, or dimensions misaligned).
    //
    // Allocated once at `preload_weights_gpu_resident` time. Stays alive for
    // the lifetime of the backend instance.
    /// Per-layer FFN-down (`w_down`) repacked Q8_0 buffer.
    /// Shape per layer: same bytes as raw Q8_0, restructured into stripes of
    /// 32 rows × 32 K-elements. Set when `LUMEN_METAL_Q8_REPACKED_FFN_DOWN=1`.
    pub(crate) repacked_ffn_down: Vec<Option<MetalBuffer>>,
    /// Per-layer FFN gate+up pair-packed Q8_0 buffer.
    /// Shape per layer: 2× the bytes of a single FFN-gate or FFN-up tensor,
    /// holding both gate and up in an interleaved SoA stripe layout.
    /// Set when `LUMEN_METAL_Q8_REPACKED_GATE_UP=1`.
    pub(crate) repacked_ffn_gate_up: Vec<Option<MetalBuffer>>,

    // Q4_0 port of — runtime hot-weight repack for FFN tensors.
    // Identical semantics to `repacked_ffn_down` / `repacked_ffn_gate_up` above,
    // but for Q4_0 quant. The Q4 stripe layout uses 18-byte source blocks (vs
    // Q8's 34 bytes), so the packed stride differs (576 vs 1088 bytes per
    // single-tensor (row_group, k_block); 1152 vs 2176 for pair-packed).
    //
    // Allocated when `LUMEN_METAL_Q4_REPACKED=1` is set at load time.
    /// Per-layer FFN-down (`w_down`) repacked Q4_0 buffer.
    /// Set when `LUMEN_METAL_Q4_REPACKED_FFN_DOWN=1`.
    pub(crate) repacked_ffn_down_q4: Vec<Option<MetalBuffer>>,
    /// Per-layer FFN gate+up pair-packed Q4_0 buffer.
    /// Set when `LUMEN_METAL_Q4_REPACKED_GATE_UP=1`.
    pub(crate) repacked_ffn_gate_up_q4: Vec<Option<MetalBuffer>>,
    /// Per-layer FFN-down MLX-style decode-qmv buffers (sequential-nibble qweights
    /// + f32 scales). Set when `LUMEN_METAL_Q4_QMV_DOWN=1`. Consumed by the
    /// `qmv_q4_0_residual` kernel. Empty => fall back to the NR2 decode kernel.
    pub(crate) qmv_down_qw: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_down_scales: Vec<Option<MetalBuffer>>,
    /// Per-GDN-layer MLX-style decode-qmv buffers for the GDN qkv projection
    /// (sequential-nibble qweights + f32 scales). Indexed by `gdn_idx` (sequential
    /// GDN layer counter, matching `gdn_h_states`). Set when `LUMEN_METAL_Q4_QMV_PROJ=1`.
    /// Consumed by the `qmv_q4_0_rmsnorm` kernel. Empty / None => NR2 fused fallback.
    pub(crate) qmv_gdn_qkv_qw: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_gdn_qkv_scales: Vec<Option<MetalBuffer>>,
    /// Per-GDN-layer MLX-style decode-qmv buffers for the GDN attn_gate projection
    /// (`st.attn_gate`, [q_dim, hidden_dim]). Indexed by `gdn_idx` (same convention as
    /// `qmv_gdn_qkv_*`). Set when `LUMEN_METAL_Q4_QMV_PROJ=1`. Consumed by the
    /// `qmv_q4_0_rmsnorm` kernel (RMSNorm with attn_norm). Empty / None => NR2 fallback.
    pub(crate) qmv_gdn_attn_gate_qw: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_gdn_attn_gate_scales: Vec<Option<MetalBuffer>>,
    /// Per-layer MLX-style decode-qmv buffers for the FULL-ATTENTION Q+gate projection
    /// (`st.wq`, [qgate_dim, hidden_dim], qgate_dim = 2*q_dim under Q+gate fusion).
    /// Indexed by `layer_idx` (0..num_layers; None for GDN layers, same convention as
    /// `qmv_down_*`). Set when `LUMEN_METAL_Q4_QMV_PROJ=1`. Consumed by the
    /// `qmv_q4_0_rmsnorm` kernel (RMSNorm with attn_norm). Empty / None => NR2 fallback.
    pub(crate) qmv_attn_wq_qw: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_attn_wq_scales: Vec<Option<MetalBuffer>>,
    /// Per-layer MLX-style decode-qmv buffers for the FULL-ATTENTION output projection
    /// (`st.wo`, [hidden_dim, q_dim]). Indexed by `layer_idx` (0..num_layers; None for
    /// GDN layers). Set when `LUMEN_METAL_Q4_QMV_PROJ=1`. Consumed by the
    /// `qmv_q4_0_residual` kernel (matvec + residual). Empty / None => NR2 fallback.
    pub(crate) qmv_attn_wo_qw: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_attn_wo_scales: Vec<Option<MetalBuffer>>,
    /// Per-layer MLX-style decode-qmv buffers for the FULL-ATTENTION K and V
    /// projections (`st.wk` / `st.wv`, each [kv_dim, hidden_dim]; in_dim = hidden,
    /// out = kv_dim). Indexed by `layer_idx` (0..num_layers; None for GDN layers).
    /// Set when `LUMEN_METAL_Q4_QMV_KV=1`. Both read the SAME pre-norm hidden as
    /// Q (rmsnorm-fused) -> consumed by the `qmv_q4_0_rmsnorm` kernel writing
    /// `k_buf` / `v_buf` at offset 0, exactly as the NR2 path does. Independent of
    /// `LUMEN_METAL_Q4_QMV_PROJ` (Q+gate/Wo) so it can be A/B'd in isolation.
    /// Empty / None => the existing NR2 (`rmsnorm_dequant_matmul_q4_0_deferred_nr2`)
    /// fused-norm fallback.
    pub(crate) qmv_attn_wk_qw: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_attn_wk_scales: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_attn_wv_qw: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_attn_wv_scales: Vec<Option<MetalBuffer>>,
    /// Per-GDN-layer MLX-style decode-qmv buffers for the GDN ssm_out / output
    /// projection (`st.ssm_out`, [hidden_dim, q_dim]; in_dim = q_dim(value_dim),
    /// out = hidden_dim). Indexed by `gdn_idx` (same convention as `qmv_gdn_qkv_*`).
    /// Set when `LUMEN_METAL_Q4_QMV_SSMOUT=1`. The SiLU gate is applied to the
    /// matvec input by a tiny predecessor `silu_elementwise_mul` dispatch, then
    /// `qmv_q4_0_residual` runs the matvec (with the zero-residual buffer so the
    /// downstream `residual_add_copy` keeps the existing accum+copy semantics).
    /// Empty / None => the existing fused silu+matvec+residual+copy NR2 path.
    pub(crate) qmv_gdn_ssm_out_qw: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_gdn_ssm_out_scales: Vec<Option<MetalBuffer>>,
    /// Per-GDN-layer standalone Q4_0 (native GGUF block layout) weight buffer for
    /// the GDN ssm_out projection, RE-QUANTIZED from the Q8_0 ssm_out at load when
    /// `LUMEN_METAL_Q4_SSMOUT_NR2=1`. Indexed by `gdn_idx` (same convention as
    /// `qmv_gdn_ssm_out_*` / `gdn_h_states`). When present, the fused decode
    /// ssm_out dispatch binds this buffer to buffer(0) and runs the Q4_0 fused
    /// `dequant_matmul_q4_0_silu_deferred_residual_copy_nr2` kernel (one dispatch)
    /// instead of the Q8_0 fused kernel reading the layer blob — halving the
    /// ssm_out weight stream with no added dispatch. Empty / None => the existing
    /// Q8_0 (or native-Q4) fused ssm_out path is unchanged.
    pub(crate) q4nr2_ssm_out: Vec<Option<MetalBuffer>>,
    /// Per-layer MLX-style decode-qmv buffers for the DENSE FFN gate/up pair
    /// (`st.w_gate` and `st.w_up`, each [inter_dim, hidden_dim]; in_dim = hidden,
    /// out = inter_dim). Indexed by `layer_idx` (0..num_layers; every dense-FFN
    /// layer participates). Set when `LUMEN_METAL_Q4_QMV_GATEUP=1`. Both matrices
    /// repacked into SEPARATE decode-qmv buffers; consumed together by the
    /// `qmv_q4_0_gate_up_swiglu` dual-matrix kernel (fused RMSNorm + SwiGLU).
    /// Any None => the existing `rmsnorm_ffn_fused_gate_up_swiglu_q4_0_8row` path.
    pub(crate) qmv_ffn_gate_qw: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_ffn_gate_scales: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_ffn_up_qw: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_ffn_up_scales: Vec<Option<MetalBuffer>>,
    /// INTERLEAVED gate+up decode-qmv buffers (env LUMEN_METAL_Q4_GATEUP_IL): per
    /// layer, ONE co-resident packed nibble buffer + ONE packed f16-scale buffer
    /// consumed by `qmv_q4_0_gate_up_swiglu_il`. Built only when the flag is on +
    /// the IL pipeline compiled; any None => fall back to the f16sc/8row/default
    /// path for that layer. Byte-identical to the separated f16sc kernel.
    pub(crate) qmv_ffn_gate_up_il_qw: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_ffn_gate_up_il_scales: Vec<Option<MetalBuffer>>,
    /// LM-head-structure (LS) gate+up decode-qmv buffers: per layer, ONE
    /// ROW-INTERLEAVED packed nibble buffer + ONE row-interleaved packed f16-scale
    /// buffer (physical row 2d = whole gate row d, 2d+1 = whole up row d) consumed by
    /// `qmv_q4_0_gate_up_swiglu_ls_h2math`. Built when the LS pipeline compiled; any
    /// None => fall back to the h2math/default path for that layer. Byte-identical to
    /// the h2math dual kernel.
    pub(crate) qmv_ffn_gate_up_ls_qw: Vec<Option<MetalBuffer>>,
    pub(crate) qmv_ffn_gate_up_ls_scales: Vec<Option<MetalBuffer>>,
    /// Persistent all-zero f32 buffer (length >= hidden_dim) used as the `residual`
    /// argument when `qmv_q4_0_residual` services the Q+gate-fusion Wo path, which is
    /// mathematically NON-residual (the residual is added downstream by `residual_add_copy`).
    /// Feeding a zero residual makes `Wo*x + 0 == Wo*x` exactly (IEEE-754 +0.0). Allocated
    /// once (zero-initialized) alongside the Wo qmv buffers. None => no qmv Wo wiring.
    pub(crate) qmv_zero_residual_buf: Option<MetalBuffer>,
    /// GLOBAL (single, non-per-layer) MLX-style decode-qmv buffers for the lm_head /
    /// output projection, built by RE-QUANTIZING the Q8_0 output_proj weights to Q4_0
    /// at load time (`LUMEN_METAL_Q4_QMV_LMHEAD=1`, default OFF). `qmv_lmhead_qw`
    /// holds sequential-nibble Q4_0 qweights [vocab, hidden/2]; `qmv_lmhead_scales`
    /// holds the per-block f32 scales [vocab, hidden/32]. Both Some => the final
    /// RMSNorm+logits dispatch uses `qmv_q4_0_rmsnorm` (in_dim=hidden, out=vocab,
    /// norm_w=final_norm); None => the existing Q8_0 fused-rmsnorm lm_head path.
    /// This is a MLX-precision-match (4-bit lm_head) speed lever, NOT byte-identical.
    pub(crate) qmv_lmhead_qw: Option<MetalBuffer>,
    pub(crate) qmv_lmhead_scales: Option<MetalBuffer>,

    // BF16 GDN qkv-proj + attn-gate-proj concat-then-stripe repacked buffer.
    // Per-layer entry contains both projections in a single Metal buffer of size
    // `(qkv_n + gate_n) * hidden_dim * 2` bytes, byte-permuted into the
    // stripe layout. Only populated for the 24 GDN layers in Qwen3.5-9B (layers
    // where `attn_gate_off` is Some). VRAM ~2.3 GB for the full set, well under
    // the 4.8 GB Apple AGX TLB threshold.
    //
    // Set when `LUMEN_METAL_BF16_GDN_QKV_GATE_PAIRED=1`.
    pub(crate) repacked_gdn_qkv_gate_bf16: Vec<Option<MetalBuffer>>,
}
