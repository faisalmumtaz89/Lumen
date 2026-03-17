//! Shared type definitions for the Metal backend.
//!
//! Contains GPU pipeline states, per-layer cached metadata, scratch buffers,
//! and MoE/GDN runtime types used across `mod.rs`, `gdn.rs`, and `moe.rs`.

use super::ffi::{MetalBuffer, MetalCommandBuffer, MetalPipelineState};
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
    pub(crate) fused_rope_neox_kv_write: Option<MetalPipelineState>,
    pub(crate) multi_head_attention: MetalPipelineState,
    pub(crate) flash_decode_attention: MetalPipelineState,
    pub(crate) flash_decode_reduce: MetalPipelineState,
    pub(crate) add_residual: MetalPipelineState,
    pub(crate) embed_token: MetalPipelineState,
    pub(crate) embed_token_q8_0: MetalPipelineState,
    pub(crate) embed_token_q4_0: MetalPipelineState,
    pub(crate) embed_token_f16: MetalPipelineState,

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
    // Deferred-reduction decode kernels (llama.cpp NQ=8 pattern, 2 sync points vs 64)
    pub(crate) dequant_matmul_q8_0_deferred: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_deferred_residual: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_deferred_bias: MetalPipelineState,
    // NR0=2 deferred variants (2 rows/TG for better occupancy on small output dims)
    pub(crate) dequant_matmul_q8_0_deferred_nr2: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_deferred_residual_nr2: MetalPipelineState,
    pub(crate) dequant_matmul_q8_0_deferred_bias_nr2: MetalPipelineState,
    // MLX-style 2-SG independent row ownership (zero barriers, zero shmem)
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
    pub(crate) dequant_tiled_matmul_q8_0_residual_batched: MetalPipelineState,
    // Q4_0 batched prefill kernels
    pub(crate) dequant_tiled_matmul_q4_0: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_residual_batched: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_splitk: MetalPipelineState,
    // Q4_1 kernels
    pub(crate) dequant_tiled_matmul_q4_1: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_1_residual_batched: MetalPipelineState,
    pub(crate) dequant_matmul_q4_1_deferred: MetalPipelineState,  // decode matvec
    pub(crate) tiled_matmul_bytes_f32_residual: MetalPipelineState,
    pub(crate) tiled_matmul_f16: MetalPipelineState,
    pub(crate) tiled_matmul_f16_residual: MetalPipelineState,
    pub(crate) tiled_matmul_f16_k64: MetalPipelineState,
    pub(crate) tiled_matmul_f16_k64_residual: MetalPipelineState,
    pub(crate) matmul_bytes_f32_residual: MetalPipelineState,

    // Buffer ops (GPU-side activation transfer)
    pub(crate) copy_buffer: MetalPipelineState,
    pub(crate) add_write: MetalPipelineState,

    // Split-K GEMM kernels (for GPU core saturation during small prefill)
    pub(crate) dequant_tiled_matmul_q8_0_splitk: MetalPipelineState,
    pub(crate) reduce_splitk: MetalPipelineState,

    // K64 (TILE_K=64) GEMM variants for fewer barriers
    pub(crate) dequant_tiled_matmul_q8_0_k64: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0_k64_residual_batched: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_k64: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q4_0_k64_residual_batched: MetalPipelineState,

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

    // Batched prefill kernels
    pub(crate) tiled_matmul_f32: MetalPipelineState,
    pub(crate) tiled_matmul_bytes_f32: MetalPipelineState,
    pub(crate) dequant_tiled_matmul_q8_0: MetalPipelineState,
    pub(crate) rmsnorm_batched: MetalPipelineState,
    pub(crate) rmsnorm_batched_bytes: MetalPipelineState,
    pub(crate) rope_batched: MetalPipelineState,
    pub(crate) rope_batched_neox: Option<MetalPipelineState>,
    pub(crate) add_residual_batched: MetalPipelineState,
    pub(crate) swiglu_batched: MetalPipelineState,
    pub(crate) embed_tokens_batched: MetalPipelineState,
    pub(crate) embed_tokens_batched_q8_0: MetalPipelineState,
    pub(crate) embed_tokens_batched_q4_0: MetalPipelineState,
    pub(crate) embed_tokens_batched_f16: MetalPipelineState,
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

    // QKV bias addition (Qwen2-family models)
    pub(crate) bias_add: MetalPipelineState,
    pub(crate) bias_add_batched: MetalPipelineState,

    // Fused QKV deinterleave (splits [M][qkv_dim] -> Q, K, V buffers)
    pub(crate) deinterleave_qkv: MetalPipelineState,

    // MoE (Mixture of Experts) pipeline states.
    // Option to allow graceful error messages if shader compilation fails.
    // The MoE kernels are included in METAL_SHADER_SOURCE.
    pub(crate) moe_router_softmax: Option<MetalPipelineState>,
    pub(crate) moe_router_softmax_batched: Option<MetalPipelineState>,
    pub(crate) moe_router_softmax_biased: Option<MetalPipelineState>,
    pub(crate) moe_expert_accum: Option<MetalPipelineState>,
    pub(crate) moe_expert_accum_batched: Option<MetalPipelineState>,
    pub(crate) moe_expert_accum_option_a: Option<MetalPipelineState>,
    // Batched MoE expert FFN — GPU-side routing, no CPU readback.
    pub(crate) moe_batched_gate_up_swiglu_q4_0: Option<MetalPipelineState>,
    pub(crate) moe_batched_gate_up_swiglu_q4_1: Option<MetalPipelineState>,
    pub(crate) moe_batched_gate_up_swiglu_q8_0: Option<MetalPipelineState>,
    pub(crate) moe_batched_down_accum_q4_0: Option<MetalPipelineState>,
    pub(crate) moe_batched_down_accum_q4_1: Option<MetalPipelineState>,
    pub(crate) moe_batched_down_accum_q8_0: Option<MetalPipelineState>,
    // Fused down+accum+shared_expert kernels (eliminates 3 dispatches per MoE layer)
    pub(crate) moe_batched_down_accum_shared_q8_0: Option<MetalPipelineState>,
    pub(crate) moe_batched_down_accum_shared_q8_0_se_q4_0: Option<MetalPipelineState>,
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
    // Fused L2-normalize + state-update + output + RMSNorm (eliminates l2_normalize_qk dispatch)
    pub(crate) gdn_state_output_norm_l2: Option<MetalPipelineState>,
    // Simdgroup-parallel state update (4096 TGs of 32 threads, writes raw output)
    pub(crate) gdn_state_output_l2_sg: Option<MetalPipelineState>,
    // RMSNorm + scale on raw GDN decode output (pairs with gdn_state_output_l2_sg)
    pub(crate) gdn_decode_norm_scale: Option<MetalPipelineState>,
    // Fused Conv1D+SiLU + L2-normalize + state-update + output + RMSNorm (eliminates conv1d dispatch + barrier)
    pub(crate) gdn_state_output_norm_l2_conv: Option<MetalPipelineState>,
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
    pub(crate) gdn_prefill_norm_gate: Option<MetalPipelineState>,
    pub(crate) ssm_conv1d_prefill: Option<MetalPipelineState>,
    pub(crate) ssm_conv1d_silu_prefill: Option<MetalPipelineState>,
    pub(crate) ssm_conv1d_silu_prefill_parallel: Option<MetalPipelineState>,
    pub(crate) l2_normalize_heads_batched: Option<MetalPipelineState>,
    pub(crate) l2_normalize_qk_strided: Option<MetalPipelineState>,
    pub(crate) gdn_compute_gates_batched: Option<MetalPipelineState>,
    pub(crate) dequant_batched_matvec_q8_0: Option<MetalPipelineState>,
    pub(crate) dequant_batched_matvec_q8_0_dual: Option<MetalPipelineState>,
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
    // q_dim+q_dim outputs = Q + gate), K and V must be projected separately.
    // When has_qgate_fusion is true:
    //   - wq_off points to attn_q.weight (output dim = 2*q_dim = Q + gate)
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
    pub(crate) logits_buf: MetalBuffer,

    // GPU-side argmax result: 1 x u32 (4 bytes). Eliminates 128KB logits readback
    // for greedy sampling (temperature <= 0).
    pub(crate) argmax_result_buf: MetalBuffer,

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
    pub(crate) qkv_dim: usize,  // q_dim + 2 * kv_dim (for fused QKV projection)
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

    /// Cached per-layer zero-copy Metal buffers (avoid re-creating on every call).
    /// Indexed by layer_idx. Populated lazily on first access per layer.
    pub(crate) layer_buf_cache: Vec<Option<(usize, MetalBuffer)>>,  // (ptr, buffer)

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
    pub(crate) batch_x_buf: Option<MetalBuffer>,         // [batch, hidden_dim]
    pub(crate) batch_normed_buf: Option<MetalBuffer>,     // [batch, hidden_dim]
    pub(crate) batch_qkv_buf: Option<MetalBuffer>,        // [batch, qkv_dim] fused QKV output
    pub(crate) batch_q_buf: Option<MetalBuffer>,          // [batch, q_dim]
    pub(crate) batch_k_buf: Option<MetalBuffer>,          // [batch, kv_dim]
    pub(crate) batch_v_buf: Option<MetalBuffer>,          // [batch, kv_dim]
    pub(crate) batch_attn_out_buf: Option<MetalBuffer>,   // [batch, q_dim]
    pub(crate) batch_attn_proj_buf: Option<MetalBuffer>,  // [batch, hidden_dim]
    pub(crate) batch_gate_buf: Option<MetalBuffer>,       // [batch, inter_dim]
    pub(crate) batch_up_buf: Option<MetalBuffer>,         // [batch, inter_dim]
    pub(crate) batch_down_buf: Option<MetalBuffer>,       // [batch, hidden_dim]
    pub(crate) batch_scores_buf: Option<MetalBuffer>,     // [batch, num_heads, max_seq_len]
    pub(crate) splitk_partial_buf: Option<MetalBuffer>,   // [K_SPLITS * max_M * max_N] floats for Split-K
    pub(crate) splitk_alloc_elems: usize,                 // tracks allocated Split-K buffer capacity (in floats)
    pub(crate) current_max_batch: usize,                  // tracks allocated batch size

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

    /// True when the model is detected as Qwen3.5-MoE (hybrid attention + shared expert).
    pub(crate) is_qwen35moe: bool,
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
    /// GDN conv kernel size (typically 4).
    pub(crate) gdn_conv_kernel_size: usize,
    /// Number of GDN layers in the model (layer_type=1 count).
    pub(crate) gdn_num_layers: usize,
    /// Maps layer_idx -> gdn_idx for streaming path lazy allocation.
    /// Empty until first GDN layer encountered in compute_layer.
    pub(crate) gdn_layer_idx_map: Vec<Option<usize>>,
}
