//! Metal GPU F32 compute backend for Apple Silicon.
//!
//! Implements `ComputeBackend` using Metal GPU compute shaders. On Apple Silicon
//! unified memory, weight data from mmap is already in GPU-accessible memory,
//! enabling zero-copy weight access via `MTLBuffer(bytesNoCopy:)`.
//!
//! Decode path: each `compute_layer` call encodes and executes GPU commands per
//! layer (async commit). Prefill path: ALL layers are encoded into a SINGLE
//! Metal command buffer with one commit_and_wait() at the end, eliminating
//! N-1 GPU-CPU sync barriers.
//!
//! # Performance characteristics
//!
//! - Matrix-vector multiply: GPU-parallelized across output rows
//! - RMSNorm: SIMD group reductions for fast sum-of-squares
//! - Attention: Scores computed in parallel, softmax on GPU, value accumulation parallel
//! - Activation buffers: Metal shared-mode buffers (CPU/GPU zero-copy)

pub(crate) mod ffi;
pub(crate) mod shaders;
pub(crate) mod io;
mod gdn;
mod moe;

use crate::weight::cache::LayerView;
use crate::compute::{ActivationBuffer, ComputeBackend, ComputeDtype, Logits};
use crate::error::RuntimeError;
use crate::expert::cache::ExpertLfuCache;
use crate::expert::profiler::ExpertActivationProfiler;
use crate::expert::reader::ExpertReader;
use crate::kv::KvCacheView;
use self::ffi::{
    MetalBuffer, MetalCommandBuffer, MetalCommandQueue,
    MetalDevice, MetalFunctionConstantValues, MetalPipelineState, MTLSize,
};
use self::io::MetalIOQueue;
use self::shaders::METAL_SHADER_SOURCE;
use lumen_format::hyperparams::ModelHyperparams;
use lumen_format::quantization::QuantScheme;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Page size for alignment checks (4 KiB on all Apple Silicon).
const PAGE_SIZE: usize = 4096;

/// Async expert prefetch state for one-layer lookahead.
///
/// After layer N's router produces expert_ids, a background thread pre-reads
/// those same experts for layer N+1. If layer N+1's router selects the same
/// experts (common due to routing locality), the prefetch result is used
/// directly, avoiding synchronous disk I/O.
struct PrefetchState {
    /// The target layer index for which experts were prefetched.
    target_layer: usize,
    /// The expert IDs that were prefetched (retained for diagnostics).
    #[allow(dead_code)]
    expert_ids: Vec<u32>,
    /// Join handle resolving to prefetched expert data.
    /// Each result corresponds to the expert_ids in order.
    handle: std::thread::JoinHandle<Vec<(u32, Result<(Vec<u8>, lumen_format::index::ExpertSlice), crate::expert::reader::ExpertReaderError>)>>,
}

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

// ============================================================================
// MetalF32Backend
// ============================================================================

/// Metal GPU F32 compute backend.
///
/// Identical API to NaiveF32Backend and SimdF32Backend. The engine interacts
/// with it through `Box<dyn ComputeBackend>` with no awareness of the GPU.
pub struct MetalF32Backend {
    device: MetalDevice,
    queue: MetalCommandQueue,
    pipelines: Option<MetalPipelines>,

    // Global tensors (GPU buffers)
    embedding_buf: Option<MetalBuffer>,
    final_norm_buf: Option<MetalBuffer>,
    output_proj_buf: Option<MetalBuffer>,

    // Keep CPU copies for set_global_tensors / embed_token fallback
    embedding: Vec<f32>,
    final_norm: Vec<f32>,
    output_proj: Vec<f32>,
    /// Raw output_proj bytes for Q8_0 GPU dispatch (avoids CPU dequant).
    output_proj_raw: Option<Vec<u8>>,
    /// Quantization scheme of the output_proj tensor.
    output_proj_quant: QuantScheme,
    /// Raw embedding bytes for Q8_0/Q4_0 GPU dequant kernels.
    embedding_raw: Option<Vec<u8>>,
    /// Quantization scheme of the embedding tensor.
    embedding_quant: QuantScheme,
    /// Whether output_proj shares embedding storage (weight tying).
    weight_tying: bool,

    scratch: Mutex<Option<MetalScratch>>,
    cached_hidden_dim: usize,
    cached_vocab_size: usize,

    // ====================================================================
    // MoE expert caching infrastructure
    // ====================================================================
    // Only active for MoE models in streaming mode (non-GPU-resident).
    // Records expert activation patterns and caches hot experts to avoid
    // redundant SSD reads on subsequent tokens.

    /// Expert activation profiler: tracks per-(layer, expert) activation counts.
    /// Initialized when the model has num_experts > 0.
    expert_profiler: Option<Mutex<ExpertActivationProfiler>>,

    /// LFU cache for expert weights: keeps hot experts in RAM.
    /// Checked before loading from disk in the streaming MoE decode path.
    expert_cache: Option<Mutex<ExpertLfuCache>>,

    /// Direct byte-range reader for individual expert weights from LBC file.
    /// Used on cache misses to load only the needed expert (not the full layer blob).
    expert_reader: Option<Mutex<ExpertReader>>,

    /// Path to the LBC model file (stored for ExpertReader initialization).
    lbc_path: Option<PathBuf>,

    /// Number of profiling tokens remaining before triggering cache warm-up.
    /// When this reaches 0, `warm_from_profile()` is called to pre-populate the
    /// expert cache with the hottest experts observed during the profiling phase.
    /// Uses AtomicUsize for interior mutability (called from &self methods).
    profiling_tokens_remaining: AtomicUsize,
    /// Number of top-K experts per layer to cache during warmup.
    profiling_top_k: usize,
    /// Whether cache warmup has been completed.
    /// Uses AtomicBool for interior mutability (called from &self methods).
    warmup_complete: AtomicBool,

    // Cache-conditional routing bias
    // ====================================================================

    /// Bias magnitude for cache-conditional routing. When > 0.0, cached experts
    /// receive a logit boost of `cache_bias_lambda` before softmax in the MoE
    /// router, nudging borderline selections toward already-cached experts.
    /// Default 0.0 (disabled). Set via `configure_routing_bias()`.
    cache_bias_lambda: f32,

    // ====================================================================
    // MoE I/O instrumentation
    // ====================================================================

    /// Bytes of expert data loaded from disk via ExpertReader (Tier 2 misses).
    expert_bytes_from_disk: AtomicU64,
    /// Bytes of expert data served from ExpertLfuCache (Tier 1 + Tier 2 hits).
    expert_bytes_from_cache: AtomicU64,
    /// Bytes of expert data accessed via full layer blob fallback (Tier 3).
    expert_bytes_from_blob: AtomicU64,

    // ====================================================================
    // Option A dispatch
    // ====================================================================

    /// When true, MoE decode dispatches only the top-K selected expert FFNs
    /// instead of all num_experts (Option B). In streaming mode, expert_ids are
    /// available CPU-side after synchronous router readback. In
    /// GPU-resident mode, a two-CB split per MoE layer achieves the same
    /// selective dispatch. Default false (opt-in via
    /// `configure_option_a(true)`).
    use_option_a: bool,

    // ====================================================================
    // Async expert prefetching
    // ====================================================================

    /// One-layer lookahead prefetch handle. After layer N's router produces
    /// expert_ids, a background thread pre-reads the same experts for layer N+1
    /// from disk. At layer N+1, the prefetch result is checked before falling
    /// back to synchronous load. Only active when use_option_a is true.
    ///
    /// The handle contains: (target_layer, expert_ids, join_handle).
    /// The join_handle resolves to Vec<(expert_id, Result<(Vec<u8>, ExpertSlice)>)>.
    prefetch_handle: Mutex<Option<PrefetchState>>,

    // ====================================================================
    // Router diagnostics
    // ====================================================================

    /// When true, router debug readback is active: after each decode token,
    /// expert_ids and expert_weights are read back for all MoE layers and
    /// stored in `router_debug_log`.
    router_debug_enabled: bool,

    /// Accumulated per-layer routing stats from decode tokens.
    /// Only populated when `router_debug_enabled` is true.
    router_debug_log: Mutex<Vec<RouterLayerStats>>,

    // ====================================================================
    // Metal IO command queue for direct NVMe-to-GPU DMA
    // ====================================================================

    /// Metal IO command queue for direct file-to-GPU DMA transfers.
    /// Available on Metal 3 (M2+) with macOS 13+. When present, streaming
    /// expert loading bypasses CPU memory and loads directly from NVMe SSD
    /// into the Metal buffer. Falls back to pread + blit when None.
    metal_io_queue: Option<MetalIOQueue>,
}

impl MetalF32Backend {
    /// Create a new Metal compute backend.
    ///
    /// Returns an error if Metal is not available.
    pub fn new() -> Result<Self, RuntimeError> {
        let device = MetalDevice::system_default().ok_or_else(|| {
            RuntimeError::Compute("Metal GPU not available on this system".into())
        })?;

        let queue = device.new_command_queue().ok_or_else(|| {
            RuntimeError::Compute("Failed to create Metal command queue".into())
        })?;

        // Attempt to create a Metal IO command queue (Metal 3 / macOS 13+).
        // This enables direct NVMe-to-GPU DMA for streaming expert loading.
        let metal_io_queue = MetalIOQueue::new(&device);
        if metal_io_queue.is_some() {
            eprintln!("[metal] MTLIOCommandQueue available (Metal 3 direct DMA enabled)");
        }

        Ok(Self {
            device,
            queue,
            pipelines: None,
            embedding_buf: None,
            final_norm_buf: None,
            output_proj_buf: None,
            embedding: Vec::new(),
            final_norm: Vec::new(),
            output_proj: Vec::new(),
            output_proj_raw: None,
            output_proj_quant: QuantScheme::F32,
            embedding_raw: None,
            embedding_quant: QuantScheme::F32,
            weight_tying: false,
            scratch: Mutex::new(None),
            cached_hidden_dim: 0,
            cached_vocab_size: 0,
            expert_profiler: None,
            expert_cache: None,
            expert_reader: None,
            lbc_path: None,
            profiling_tokens_remaining: AtomicUsize::new(0),
            profiling_top_k: 0,
            warmup_complete: AtomicBool::new(false),
            cache_bias_lambda: 0.0,
            expert_bytes_from_disk: AtomicU64::new(0),
            expert_bytes_from_cache: AtomicU64::new(0),
            expert_bytes_from_blob: AtomicU64::new(0),
            use_option_a: false,
            prefetch_handle: Mutex::new(None),
            router_debug_enabled: false,
            router_debug_log: Mutex::new(Vec::new()),
            metal_io_queue,
        })
    }




    /// Returns whether expert cache warmup has been completed.
    pub fn is_warmup_complete(&self) -> bool {
        self.warmup_complete.load(Ordering::Relaxed)
    }

    /// Returns a snapshot of expert activation profiler statistics.
    /// Returns None if the model is not MoE or profiler is not initialized.
    pub fn expert_profiler_summary(&self) -> Option<crate::expert::profiler::ProfilerSummary> {
        self.expert_profiler.as_ref().map(|p| p.lock().unwrap().summary())
    }

    /// Returns a snapshot of expert cache statistics.
    /// Returns None if expert caching is not configured.
    pub fn expert_cache_stats(&self) -> Option<crate::expert::cache::CacheStats> {
        self.expert_cache.as_ref().map(|c| c.lock().unwrap().stats())
    }

    /// Returns cumulative MoE expert I/O byte counters.
    ///
    /// Returns `(bytes_from_disk, bytes_from_cache, bytes_from_blob)`:
    /// - `bytes_from_disk`: loaded via ExpertReader on cache miss (Tier 2)
    /// - `bytes_from_cache`: served from ExpertLfuCache (Tier 1 + Tier 2 hits)
    /// - `bytes_from_blob`: accessed via full layer blob fallback (Tier 3)
    pub fn expert_io_stats(&self) -> (u64, u64, u64) {
        (
            self.expert_bytes_from_disk.load(Ordering::Relaxed),
            self.expert_bytes_from_cache.load(Ordering::Relaxed),
            self.expert_bytes_from_blob.load(Ordering::Relaxed),
        )
    }

    /// Returns whether Metal IO DMA (MTLIOCommandQueue) is available.
    ///
    /// When true, streaming expert cache misses use direct NVMe-to-GPU DMA
    /// instead of pread + CPU copy.
    pub fn has_metal_io_queue(&self) -> bool {
        self.metal_io_queue.is_some()
    }

    /// Enable or disable router debug readback.
    ///
    /// When enabled, after each decode token the backend reads back per-layer

    /// Returns the accumulated router debug log and clears it.
    ///
    /// Each entry is a `RouterLayerStats` captured from one MoE layer during
    /// one decode token. The log contains entries for ALL MoE layers across
    /// ALL tokens decoded since the last call to this method (or since init).
    pub fn get_router_debug_log(&self) -> Vec<RouterLayerStats> {
        let mut log = self.router_debug_log.lock().unwrap();
        std::mem::take(&mut *log)
    }


    pub fn set_global_tensors(
        &mut self,
        embedding: Vec<f32>,
        final_norm: Vec<f32>,
        output_proj: Vec<f32>,
    ) {
        self.embedding = embedding;
        self.final_norm = final_norm;
        self.output_proj = output_proj;
    }

    /// Set raw Q8_0 output projection bytes for GPU-native dequant-matmul.
    ///
    /// When called, compute_final() will use the fused dequant_matmul_q8_0
    /// kernel instead of matmul_f32, reducing bandwidth 3.76x.
    /// The F32 `output_proj` from set_global_tensors is still needed for
    /// CPU-side embed_token (if output_proj is tied to embedding).
    pub fn set_output_proj_q8(&mut self, raw_bytes: Vec<u8>, quant: QuantScheme) {
        self.output_proj_quant = quant;
        self.output_proj_raw = Some(raw_bytes);
    }

    /// Set raw quantized embedding bytes for GPU-native dequant embed kernels.
    ///
    /// When called, embed_token() and batched prefill embed will use the
    /// appropriate dequant kernel (Q8_0 or Q4_0) instead of the F32 kernel.
    /// The F32 `embedding` from set_global_tensors is still needed for CPU fallback.
    pub fn set_embedding_raw(&mut self, raw_bytes: Vec<u8>, quant: QuantScheme) {
        self.embedding_quant = quant;
        self.embedding_raw = Some(raw_bytes);
    }

    /// Enable weight tying: output_proj shares embedding storage.
    pub fn set_weight_tying(&mut self, enabled: bool) {
        self.weight_tying = enabled;
    }

    /// Get the device name (for diagnostics).
    pub fn device_name(&self) -> String {
        self.device.name()
    }

    /// Compile all Metal shader pipelines.
    fn compile_pipelines(&self) -> Result<MetalPipelines, RuntimeError> {
        let lib = self
            .device
            .new_library_with_source(METAL_SHADER_SOURCE)
            .map_err(RuntimeError::Compute)?;

        macro_rules! make_pipeline {
            ($name:expr) => {{
                let func = lib.get_function($name).ok_or_else(|| {
                    RuntimeError::Compute(format!("Metal kernel '{}' not found", $name))
                })?;
                self.device
                    .new_compute_pipeline_state(&func)
                    .map_err(|e| RuntimeError::Compute(e))?
            }};
        }

        // Create a pipeline specialized with function constants for aligned GEMM.
        // BC_M=false, BC_N=false, BC_K=false: the Metal compiler dead-code-eliminates
        // all boundary checks, producing a faster kernel for aligned dimensions.
        macro_rules! make_aligned_pipeline {
            ($name:expr) => {{
                let fcv = MetalFunctionConstantValues::new();
                fcv.set_bool(false, 10); // FC_BC_M = false (M aligned to TILE_M)
                fcv.set_bool(false, 11); // FC_BC_N = false (N aligned to TILE_N)
                fcv.set_bool(false, 12); // FC_BC_K = false (K aligned to TILE_K)
                let func = lib.get_function_with_constants($name, &fcv)
                    .map_err(RuntimeError::Compute)?;
                self.device
                    .new_compute_pipeline_state(&func)
                    .map_err(|e| RuntimeError::Compute(e))?
            }};
        }

        // Create a pipeline for kernels that use function constants, with BC_M/N/K=true
        // (boundary-checked fallback). Required because once a kernel declares
        // [[function_constant]] attributes, plain newFunctionWithName: no longer works.
        macro_rules! make_bc_pipeline {
            ($name:expr) => {{
                let fcv = MetalFunctionConstantValues::new();
                fcv.set_bool(true, 10);  // FC_BC_M = true (boundary checks enabled)
                fcv.set_bool(true, 11);  // FC_BC_N = true
                fcv.set_bool(true, 12);  // FC_BC_K = true
                let func = lib.get_function_with_constants($name, &fcv)
                    .map_err(RuntimeError::Compute)?;
                self.device
                    .new_compute_pipeline_state(&func)
                    .map_err(|e| RuntimeError::Compute(e))?
            }};
        }

        Ok(MetalPipelines {
            matmul_f32: make_pipeline!("matmul_f32"),
            matmul_f32_deferred: make_pipeline!("matmul_f32_deferred"),
            matmul_bytes_f32: make_pipeline!("matmul_bytes_f32"),
            // F16 decode kernels
            matmul_f16_deferred_nr2: make_pipeline!("matmul_f16_deferred_nr2"),
            matmul_f16_deferred_residual_nr2: make_pipeline!("matmul_f16_deferred_residual_nr2"),
            matmul_f16_deferred_bias_nr2: make_pipeline!("matmul_f16_deferred_bias_nr2"),
            dequant_matmul_q8_0: make_pipeline!("dequant_matmul_q8_0"),
            rmsnorm: make_pipeline!("rmsnorm"),
            rmsnorm_bytes: make_pipeline!("rmsnorm_bytes"),
            rope: make_pipeline!("rope"),
            rope_neox: lib.get_function("rope_neox")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            swiglu: make_pipeline!("swiglu"),
            softmax: make_pipeline!("softmax"),
            attention_scores: make_pipeline!("attention_scores"),
            attention_output: make_pipeline!("attention_output"),
            write_kv_cache: make_pipeline!("write_kv_cache"),
            fused_rope_kv_write: make_pipeline!("fused_rope_kv_write"),
            fused_rope_kv_mha: make_pipeline!("fused_rope_kv_mha"),
            fused_rope_neox_kv_write: lib.get_function("fused_rope_neox_kv_write")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            multi_head_attention: make_pipeline!("multi_head_attention"),
            flash_decode_attention: make_pipeline!("flash_decode_attention"),
            flash_decode_reduce: make_pipeline!("flash_decode_reduce"),
            add_residual: make_pipeline!("add_residual"),
            embed_token: make_pipeline!("embed_token"),
            embed_token_q8_0: make_pipeline!("embed_token_q8_0"),
            embed_token_q4_0: make_pipeline!("embed_token_q4_0"),
            embed_token_f16: make_pipeline!("embed_token_f16"),
            dequant_matmul_q8_0_residual: make_pipeline!("dequant_matmul_q8_0_residual"),
            dequant_matmul_q8_0_multirow: make_pipeline!("dequant_matmul_q8_0_multirow"),
            dequant_matmul_q8_0_residual_multirow: make_pipeline!("dequant_matmul_q8_0_residual_multirow"),
            dequant_matmul_q8_0_4row: make_pipeline!("dequant_matmul_q8_0_4row"),
            dequant_matmul_q8_0_residual_4row: make_pipeline!("dequant_matmul_q8_0_residual_4row"),
            dequant_matmul_q8_0_8row: make_pipeline!("dequant_matmul_q8_0_8row"),
            dequant_matmul_q8_0_residual_8row: make_pipeline!("dequant_matmul_q8_0_residual_8row"),
            dequant_matmul_q8_0_deferred: make_pipeline!("dequant_matmul_q8_0_deferred"),
            dequant_matmul_q8_0_deferred_residual: make_pipeline!("dequant_matmul_q8_0_deferred_residual"),
            dequant_matmul_q8_0_deferred_bias: make_pipeline!("dequant_matmul_q8_0_deferred_bias"),
            dequant_matmul_q8_0_deferred_nr2: make_pipeline!("dequant_matmul_q8_0_deferred_nr2"),
            dequant_matmul_q8_0_deferred_residual_nr2: make_pipeline!("dequant_matmul_q8_0_deferred_residual_nr2"),
            dequant_matmul_q8_0_deferred_bias_nr2: make_pipeline!("dequant_matmul_q8_0_deferred_bias_nr2"),
            // MLX-style 2-SG kernels
            dequant_matmul_q8_0_2sg: make_pipeline!("dequant_matmul_q8_0_2sg"),
            dequant_matmul_q8_0_2sg_residual: make_pipeline!("dequant_matmul_q8_0_2sg_residual"),
            ffn_fused_gate_up_swiglu_q8_0_2sg: make_pipeline!("ffn_fused_gate_up_swiglu_q8_0_2sg"),
            // Q4_0 decode kernels
            dequant_matmul_q4_0: make_pipeline!("dequant_matmul_q4_0"),
            dequant_matmul_q4_0_residual: make_pipeline!("dequant_matmul_q4_0_residual"),
            dequant_matmul_q4_0_4row: make_pipeline!("dequant_matmul_q4_0_4row"),
            dequant_matmul_q4_0_residual_4row: make_pipeline!("dequant_matmul_q4_0_residual_4row"),
            dequant_matmul_q4_0_deferred: make_pipeline!("dequant_matmul_q4_0_deferred"),
            dequant_matmul_q4_0_deferred_residual: make_pipeline!("dequant_matmul_q4_0_deferred_residual"),
            dequant_matmul_q4_0_deferred_bias: make_pipeline!("dequant_matmul_q4_0_deferred_bias"),
            dequant_matmul_q4_0_deferred_nr2: make_pipeline!("dequant_matmul_q4_0_deferred_nr2"),
            dequant_matmul_q4_0_deferred_residual_nr2: make_pipeline!("dequant_matmul_q4_0_deferred_residual_nr2"),
            dequant_matmul_q4_0_deferred_bias_nr2: make_pipeline!("dequant_matmul_q4_0_deferred_bias_nr2"),
            dequant_tiled_matmul_q8_0_residual_batched: make_bc_pipeline!("dequant_tiled_matmul_q8_0_residual_batched"),
            // Q4_0 batched prefill kernels
            dequant_tiled_matmul_q4_0: make_pipeline!("dequant_tiled_matmul_q4_0"),
            dequant_tiled_matmul_q4_0_residual_batched: make_pipeline!("dequant_tiled_matmul_q4_0_residual_batched"),
            dequant_tiled_matmul_q4_0_splitk: make_pipeline!("dequant_tiled_matmul_q4_0_splitk"),
            // Q4_1 kernels
            dequant_tiled_matmul_q4_1: make_pipeline!("dequant_tiled_matmul_q4_1"),
            dequant_tiled_matmul_q4_1_residual_batched: make_pipeline!("dequant_tiled_matmul_q4_1_residual_batched"),
            dequant_matmul_q4_1_deferred: make_pipeline!("dequant_matmul_q4_1_deferred"),
            tiled_matmul_bytes_f32_residual: make_pipeline!("tiled_matmul_bytes_f32_residual"),
            tiled_matmul_f16: make_pipeline!("tiled_matmul_f16"),
            tiled_matmul_f16_residual: make_pipeline!("tiled_matmul_f16_residual"),
            tiled_matmul_f16_k64: make_bc_pipeline!("tiled_matmul_f16_k64"),
            tiled_matmul_f16_k64_residual: make_bc_pipeline!("tiled_matmul_f16_k64_residual"),
            matmul_bytes_f32_residual: make_pipeline!("matmul_bytes_f32_residual"),
            copy_buffer: make_pipeline!("copy_buffer"),
            add_write: make_pipeline!("add_write"),

            // Split-K GEMM kernels
            dequant_tiled_matmul_q8_0_splitk: make_pipeline!("dequant_tiled_matmul_q8_0_splitk"),
            reduce_splitk: make_pipeline!("reduce_splitk"),

            // K64 GEMM variants
            dequant_tiled_matmul_q8_0_k64: make_bc_pipeline!("dequant_tiled_matmul_q8_0_k64"),
            dequant_tiled_matmul_q8_0_k64_residual_batched: make_bc_pipeline!("dequant_tiled_matmul_q8_0_k64_residual_batched"),
            dequant_tiled_matmul_q4_0_k64: make_bc_pipeline!("dequant_tiled_matmul_q4_0_k64"),
            dequant_tiled_matmul_q4_0_k64_residual_batched: make_bc_pipeline!("dequant_tiled_matmul_q4_0_k64_residual_batched"),

            // Function-constant-specialized aligned GEMM variants (no boundary checks)
            dequant_tiled_matmul_q8_0_aligned: make_aligned_pipeline!("dequant_tiled_matmul_q8_0"),
            dequant_tiled_matmul_q8_0_k64_aligned: make_aligned_pipeline!("dequant_tiled_matmul_q8_0_k64"),
            dequant_tiled_matmul_q8_0_k64_residual_batched_aligned: make_aligned_pipeline!("dequant_tiled_matmul_q8_0_k64_residual_batched"),
            dequant_tiled_matmul_q8_0_residual_batched_aligned: make_aligned_pipeline!("dequant_tiled_matmul_q8_0_residual_batched"),
            dequant_tiled_matmul_q4_0_k64_aligned: make_aligned_pipeline!("dequant_tiled_matmul_q4_0_k64"),
            dequant_tiled_matmul_q4_0_k64_residual_batched_aligned: make_aligned_pipeline!("dequant_tiled_matmul_q4_0_k64_residual_batched"),
            tiled_matmul_f16_k64_aligned: make_aligned_pipeline!("tiled_matmul_f16_k64"),
            tiled_matmul_f16_k64_residual_aligned: make_aligned_pipeline!("tiled_matmul_f16_k64_residual"),

            // Batched prefill kernels
            tiled_matmul_f32: make_pipeline!("tiled_matmul_f32"),
            tiled_matmul_bytes_f32: make_pipeline!("tiled_matmul_bytes_f32"),
            dequant_tiled_matmul_q8_0: make_bc_pipeline!("dequant_tiled_matmul_q8_0"),
            rmsnorm_batched: make_pipeline!("rmsnorm_batched"),
            rmsnorm_batched_bytes: make_pipeline!("rmsnorm_batched_bytes"),
            rope_batched: make_pipeline!("rope_batched"),
            rope_batched_neox: lib.get_function("rope_batched_neox")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            add_residual_batched: make_pipeline!("add_residual_batched"),
            swiglu_batched: make_pipeline!("swiglu_batched"),
            embed_tokens_batched: make_pipeline!("embed_tokens_batched"),
            embed_tokens_batched_q8_0: make_pipeline!("embed_tokens_batched_q8_0"),
            embed_tokens_batched_q4_0: make_pipeline!("embed_tokens_batched_q4_0"),
            embed_tokens_batched_f16: make_pipeline!("embed_tokens_batched_f16"),
            kv_cache_write_batched: make_pipeline!("kv_cache_write_batched"),
            v_cache_write_batched: make_pipeline!("v_cache_write_batched"),
            attention_scores_batched: make_pipeline!("attention_scores_batched"),
            softmax_batched: make_pipeline!("softmax_batched"),
            attention_output_batched: make_pipeline!("attention_output_batched"),
            attention_scores_tiled: make_pipeline!("attention_scores_tiled"),
            attention_output_tiled: make_pipeline!("attention_output_tiled"),
            rmsnorm_dequant_matmul_q8_0_deferred_nr2: make_pipeline!("rmsnorm_dequant_matmul_q8_0_deferred_nr2"),
            rmsnorm_dequant_matmul_q8_0_deferred_residual_nr2: make_pipeline!("rmsnorm_dequant_matmul_q8_0_deferred_residual_nr2"),
            rmsnorm_dequant_matmul_q4_0_deferred_nr2: make_pipeline!("rmsnorm_dequant_matmul_q4_0_deferred_nr2"),
            rmsnorm_dequant_matmul_q4_0_deferred_residual_nr2: make_pipeline!("rmsnorm_dequant_matmul_q4_0_deferred_residual_nr2"),
            // Fused RMSNorm + F16 matvec NR2
            rmsnorm_matmul_f16_deferred_nr2: make_pipeline!("rmsnorm_matmul_f16_deferred_nr2"),
            rmsnorm_matmul_f16_deferred_residual_nr2: make_pipeline!("rmsnorm_matmul_f16_deferred_residual_nr2"),
            rmsnorm_ffn_fused_gate_up_swiglu_q8_0_deferred: make_pipeline!("rmsnorm_ffn_fused_gate_up_swiglu_q8_0_deferred"),
            rmsnorm_ffn_fused_gate_up_swiglu_q8_0_8row: make_pipeline!("rmsnorm_ffn_fused_gate_up_swiglu_q8_0_8row"),
            rmsnorm_ffn_fused_gate_up_swiglu_q4_0_deferred: make_pipeline!("rmsnorm_ffn_fused_gate_up_swiglu_q4_0_deferred"),
            rmsnorm_ffn_fused_gate_up_swiglu_q4_0_8row: make_pipeline!("rmsnorm_ffn_fused_gate_up_swiglu_q4_0_8row"),
            rmsnorm_ffn_fused_gate_up_swiglu_f16_deferred: make_pipeline!("rmsnorm_ffn_fused_gate_up_swiglu_f16_deferred"),
            ffn_fused_gate_up_swiglu_q8_0: make_pipeline!("ffn_fused_gate_up_swiglu_q8_0"),
            ffn_fused_gate_up_swiglu_q8_0_deferred: make_pipeline!("ffn_fused_gate_up_swiglu_q8_0_deferred"),
            ffn_fused_gate_up_swiglu_q4_0: make_pipeline!("ffn_fused_gate_up_swiglu_q4_0"),
            ffn_fused_gate_up_swiglu_q4_0_deferred: make_pipeline!("ffn_fused_gate_up_swiglu_q4_0_deferred"),
            ffn_fused_gate_up_swiglu_q4_1_deferred: make_pipeline!("ffn_fused_gate_up_swiglu_q4_1_deferred"),
            ffn_fused_gate_up_swiglu_f16_deferred: make_pipeline!("ffn_fused_gate_up_swiglu_f16_deferred"),
            argmax: make_pipeline!("argmax"),
            bias_add: make_pipeline!("bias_add"),
            bias_add_batched: make_pipeline!("bias_add_batched"),
            deinterleave_qkv: make_pipeline!("deinterleave_qkv"),

            // MoE pipeline states.
            // Option to provide clear runtime error if Metal compilation fails.
            moe_router_softmax: lib.get_function("moe_router_softmax")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_router_softmax_batched: lib.get_function("moe_router_softmax_batched")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_router_softmax_biased: lib.get_function("moe_router_softmax_biased")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_expert_accum: lib.get_function("moe_expert_accum")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_expert_accum_batched: lib.get_function("moe_expert_accum_batched")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_expert_accum_option_a: lib.get_function("moe_expert_accum_option_a")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Batched MoE expert FFN kernels.
            moe_batched_gate_up_swiglu_q4_0: lib.get_function("moe_batched_gate_up_swiglu_q4_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_gate_up_swiglu_q4_1: lib.get_function("moe_batched_gate_up_swiglu_q4_1")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_gate_up_swiglu_q8_0: lib.get_function("moe_batched_gate_up_swiglu_q8_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_down_accum_q4_0: lib.get_function("moe_batched_down_accum_q4_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_down_accum_q4_1: lib.get_function("moe_batched_down_accum_q4_1")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_down_accum_q8_0: lib.get_function("moe_batched_down_accum_q8_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_down_accum_shared_q8_0: lib.get_function("moe_batched_down_accum_shared_q8_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_down_accum_shared_q8_0_se_q4_0: lib.get_function("moe_batched_down_accum_shared_q8_0_se_q4_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            moe_batched_down_accum_shared_q4_0: lib.get_function("moe_batched_down_accum_shared_q4_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            sigmoid_scale_add: lib.get_function("sigmoid_scale_add")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // GatedDeltaNet (linear attention) pipeline states.
            ssm_conv1d_decode: lib.get_function("ssm_conv1d_decode")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            l2_normalize_heads: lib.get_function("l2_normalize_heads")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            sigmoid_gate: lib.get_function("sigmoid_gate")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            silu_elementwise_mul: lib.get_function("silu_elementwise_mul")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gated_delta_net_state_update: lib.get_function("gated_delta_net_state_update")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gated_delta_net_output: lib.get_function("gated_delta_net_output")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // Additional GDN pipeline states for full forward pass.
            gated_delta_net_state_update_v2: lib.get_function("gated_delta_net_state_update_v2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gdn_compute_gates: lib.get_function("gdn_compute_gates")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            elementwise_mul_f32: lib.get_function("elementwise_mul_f32")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            ssm_l2_norm_scale: lib.get_function("ssm_l2_norm_scale")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // Fused element-wise kernels for GDN dispatch reduction.
            sigmoid_mul_fused: lib.get_function("sigmoid_mul_fused")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            residual_add_copy: lib.get_function("residual_add_copy")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            l2_normalize_qk: lib.get_function("l2_normalize_qk")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // SiLU activation (in-place) for post-conv1d GDN activation.
            silu_inplace: lib.get_function("silu_inplace")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Fused Conv1D + SiLU for GDN decode.
            ssm_conv1d_silu_decode: lib.get_function("ssm_conv1d_silu_decode")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // Q+gate de-interleave for Qwen3.5 full-attention layers.
            deinterleave_qgate: lib.get_function("deinterleave_qgate")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Per-head RMSNorm for Q and K (Qwen3.5 full-attention layers).
            rmsnorm_per_head: lib.get_function("rmsnorm_per_head")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Sigmoid-scale for shared expert gating.
            sigmoid_scale_buffer: lib.get_function("sigmoid_scale_buffer")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Batched sigmoid-scale-add for shared expert gating during prefill.
            sigmoid_scale_add_batched: lib.get_function("sigmoid_scale_add_batched")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // Fused GDN mega-kernels for further dispatch reduction.
            gdn_state_output_norm: lib.get_function("gdn_state_output_norm")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            dequant_matmul_q8_0_deferred_residual_copy: lib.get_function("dequant_matmul_q8_0_deferred_residual_copy")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            dequant_matmul_q8_0_deferred_residual_copy_nr2: lib.get_function("dequant_matmul_q8_0_deferred_residual_copy_nr2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            dequant_matmul_q4_0_deferred_residual_copy: lib.get_function("dequant_matmul_q4_0_deferred_residual_copy")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            dequant_matmul_q4_0_deferred_residual_copy_nr2: lib.get_function("dequant_matmul_q4_0_deferred_residual_copy_nr2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // Fused deinterleave+norm+assemble for full-attention Q+gate layers
            deinterleave_norm_assemble: lib.get_function("deinterleave_norm_assemble")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Fused L2-normalize + state-update + output + RMSNorm
            gdn_state_output_norm_l2: lib.get_function("gdn_state_output_norm_l2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Simdgroup-parallel state update (4096 TGs)
            gdn_state_output_l2_sg: lib.get_function("gdn_state_output_l2_sg")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // RMSNorm + scale for decode (pairs with gdn_state_output_l2_sg)
            gdn_decode_norm_scale: lib.get_function("gdn_decode_norm_scale")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Fused Conv1D+SiLU + L2-normalize + state-update + output + RMSNorm
            gdn_state_output_norm_l2_conv: lib.get_function("gdn_state_output_norm_l2_conv")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Fused SiLU-gated Q8_0 matvec + residual + copy
            dequant_matmul_q8_0_silu_deferred_residual_copy_nr2: lib.get_function("dequant_matmul_q8_0_silu_deferred_residual_copy_nr2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Fused SiLU-gated Q4_0 matvec + residual + copy
            dequant_matmul_q4_0_silu_deferred_residual_copy_nr2: lib.get_function("dequant_matmul_q4_0_silu_deferred_residual_copy_nr2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            // Fused dual alpha+beta RMSNorm+matvec+gates for GDN decode
            dequant_matmul_q8_0_dual_gates_nr2: lib.get_function("dequant_matmul_q8_0_dual_gates_nr2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),

            // Batched GDN prefill kernels
            gdn_prefill_state_output_norm: lib.get_function("gdn_prefill_state_output_norm")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gdn_prefill_fused: lib.get_function("gdn_prefill_fused")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gdn_prefill_fused_v2: lib.get_function("gdn_prefill_fused_v2")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gdn_prefill_fused_v3_chunked: lib.get_function("gdn_prefill_fused_v3_chunked")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gdn_prefill_norm_gate: lib.get_function("gdn_prefill_norm_gate")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            ssm_conv1d_prefill: lib.get_function("ssm_conv1d_prefill")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            ssm_conv1d_silu_prefill: lib.get_function("ssm_conv1d_silu_prefill")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            ssm_conv1d_silu_prefill_parallel: lib.get_function("ssm_conv1d_silu_prefill_parallel")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            l2_normalize_heads_batched: lib.get_function("l2_normalize_heads_batched")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            l2_normalize_qk_strided: lib.get_function("l2_normalize_qk_strided")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            gdn_compute_gates_batched: lib.get_function("gdn_compute_gates_batched")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            dequant_batched_matvec_q8_0: lib.get_function("dequant_batched_matvec_q8_0")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
            dequant_batched_matvec_q8_0_dual: lib.get_function("dequant_batched_matvec_q8_0_dual")
                .and_then(|f| self.device.new_compute_pipeline_state(&f).ok()),
        })
    }

    /// Upload f32 data to a GPU buffer.
    fn upload_f32(&self, data: &[f32]) -> Result<MetalBuffer, RuntimeError> {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.device.new_buffer_with_bytes(bytes).ok_or_else(|| {
            RuntimeError::Compute("Failed to create Metal buffer".into())
        })
    }

    /// Create a zero-copy MetalBuffer wrapping the entire layer blob.
    ///
    /// On Apple Silicon, mmap'd data is in unified memory shared between CPU and GPU.
    /// `MTLBuffer(bytesNoCopy:)` wraps it without copying -- the GPU accesses the same
    /// physical pages. Subtensors within the blob are accessed via buffer offsets in
    /// `set_buffer(&buf, offset, index)`.
    ///
    /// # Page alignment
    ///
    /// `bytesNoCopy` requires page-aligned pointers (4096 bytes on Apple Silicon).
    /// mmap'd data is always page-aligned. If the pointer is NOT page-aligned
    /// (heap-allocated LayerView from async provider), we fall back to
    /// `new_buffer_with_bytes` which copies.
    fn create_layer_buffer(&self, weights: &LayerView) -> Result<MetalBuffer, RuntimeError> {
        let blob = weights.as_bytes();
        let ptr = blob.as_ptr();
        let len = blob.len();

        if len == 0 {
            return self.device.new_buffer(4).ok_or_else(|| {
                RuntimeError::Compute("Failed to create empty layer buffer".into())
            });
        }

        // Check page alignment for zero-copy path
        if (ptr as usize) % PAGE_SIZE == 0 {
            // Page-aligned: use bytesNoCopy (zero-copy).
            // Round length up to page boundary as required by Metal.
            let aligned_len = (len + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

            // SAFETY: The LayerView's backing memory (mmap) outlives this buffer.
            // The engine holds a borrow on &dyn WeightProvider during generate(),
            // which keeps the mmap alive. The buffer is used only within this
            // compute_layer call and dropped before returning.
            let buf = unsafe {
                self.device.new_buffer_no_copy(ptr as *mut c_void, aligned_len)
            };
            if let Some(buf) = buf {
                return Ok(buf);
            }
            // Fall through to copy path if bytesNoCopy fails (shouldn't happen
            // with page-aligned mmap, but defensive).
        }

        // Not page-aligned (heap data from async provider): copy.
        self.device.new_buffer_with_bytes(blob).ok_or_else(|| {
            RuntimeError::Compute("Failed to create layer buffer (copy fallback)".into())
        })
    }

    /// Create a Metal buffer covering only the non-expert portion of a
    /// MoE layer blob. This avoids page-faulting the expert byte range from mmap,
    /// since expert data will be served from the LFU cache instead.
    ///
    /// `non_expert_end` is the byte offset in the blob where expert data begins.
    /// The returned buffer covers `blob[0..non_expert_end]` (rounded up to page size).
    fn create_partial_layer_buffer(
        &self,
        weights: &LayerView,
        non_expert_end: usize,
    ) -> Result<MetalBuffer, RuntimeError> {
        let blob = weights.as_bytes();
        let ptr = blob.as_ptr();
        let len = non_expert_end.min(blob.len());

        if len == 0 {
            return self.device.new_buffer(4).ok_or_else(|| {
                RuntimeError::Compute("Failed to create empty partial layer buffer".into())
            });
        }

        // Check page alignment for zero-copy path
        if (ptr as usize) % PAGE_SIZE == 0 {
            // Round length up to page boundary as required by Metal.
            let aligned_len = (len + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
            // Ensure we don't exceed the blob's total length (page-rounded).
            let max_aligned = (blob.len() + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
            let aligned_len = aligned_len.min(max_aligned);

            let buf = unsafe {
                self.device.new_buffer_no_copy(ptr as *mut c_void, aligned_len)
            };
            if let Some(buf) = buf {
                return Ok(buf);
            }
        }

        // Not page-aligned: copy only the non-expert portion.
        self.device.new_buffer_with_bytes(&blob[..len]).ok_or_else(|| {
            RuntimeError::Compute("Failed to create partial layer buffer (copy fallback)".into())
        })
    }

    /// Compute the byte offset where expert data begins in a MoE layer blob.
    ///
    /// Returns the end offset of the last non-expert tensor (attention weights,
    /// norms, router, biases). Everything before this offset is non-expert data;
    /// everything at or after it is expert data. If the layer has no experts,
    /// returns the full blob length.
    fn non_expert_byte_end(st: &lumen_format::index::SubtensorOffsets) -> usize {
        let mut end: u64 = 0;

        // Attention weights
        let slices = [&st.wq, &st.wk, &st.wv, &st.wo, &st.attn_norm, &st.ffn_norm];
        for s in &slices {
            let s_end = s.offset + s.length;
            if s_end > end {
                end = s_end;
            }
        }

        // Dense FFN weights (zero-length sentinels for MoE, but check anyway)
        for s in &[&st.w_gate, &st.w_up, &st.w_down] {
            let s_end = s.offset + s.length;
            if s_end > end {
                end = s_end;
            }
        }

        // Optional biases
        for opt in &[&st.bq, &st.bk, &st.bv] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        // Router weight (non-expert, always loaded)
        if let Some(ref s) = st.router_weight {
            let s_end = s.offset + s.length;
            if s_end > end {
                end = s_end;
            }
        }

        // Shared expert weights (always loaded, non-expert).
        // Qwen3.5-MoE has an always-active shared expert whose gate/up/down
        // weights live in the layer blob alongside attention/norm/router data.
        for opt in &[&st.shared_expert_gate, &st.shared_expert_up, &st.shared_expert_down] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        // Extended attention fields (always loaded, non-expert).
        // attn_gate, attn_post_norm are per-layer tensors used by hybrid models.
        for opt in &[&st.attn_gate, &st.attn_post_norm] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        // SSM / linear attention fields (always loaded, non-expert).
        // These are per-layer tensors for GatedDeltaNet hybrid layers.
        for opt in &[&st.ssm_a, &st.ssm_conv1d, &st.ssm_dt, &st.ssm_beta, &st.ssm_alpha, &st.ssm_norm, &st.ssm_out] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        // Per-head Q/K RMSNorm weights and shared expert gate input weight.
        for opt in &[&st.attn_q_norm, &st.attn_k_norm, &st.ffn_gate_inp_shexp] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        end as usize
    }

    /// Pre-load ALL layer weights into a single private (GPU-only) Metal buffer.
    ///
    /// Packs all layer weight data and global tensors into one contiguous private
    /// buffer using page-aligned offsets. Data is staged in a shared buffer then
    /// blit-copied to the private buffer. This:
    /// - Eliminates TLB misses from first-touch page faults on mmap'd memory
    /// - Reduces virtual address ranges from 22+ to 1 (lower TLB pressure)
    /// - Enables GPU memory controller optimizations via StorageModePrivate
    /// - Eliminates buffer object creation overhead per layer per token
    ///
    /// Memory cost: ~model_size bytes of GPU memory (e.g. 1.4 GB for TinyLlama Q8_0).
    pub fn preload_weights_gpu_resident(
        &self,
        weights: &dyn crate::weight::cache::WeightProvider,
    ) -> Result<(), RuntimeError> {
        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("Metal scratch not initialized: call init() first".into())
        })?;

        let num_layers = s.num_layers;

        println!("Pre-loading {} layers into single private GPU buffer...", num_layers);

        // === Pass 1: Collect layer blobs and compute page-aligned offsets ===
        let align = |size: usize| -> usize { (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1) };

        let mut layer_blobs: Vec<Vec<u8>> = Vec::with_capacity(num_layers);
        let mut layer_offsets: Vec<usize> = Vec::with_capacity(num_layers);
        let mut layer_metas: Vec<CachedLayerMeta> = Vec::with_capacity(num_layers);
        let mut cursor: usize = 0;
        let mut gdn_layer_counter: usize = 0;

        for layer in 0..num_layers {
            let layer_view = weights.get_layer_blocking(layer).map_err(|e| {
                RuntimeError::Compute(format!(
                    "Failed to get layer {} for GPU-resident loading: {}", layer, e
                ))
            })?;
            let blob = layer_view.as_bytes();
            let base = cursor as u64;
            let st = &layer_view.subtensors;
            layer_metas.push(CachedLayerMeta {
                attn_norm_off: base + st.attn_norm.offset,
                wq_off: base + st.wq.offset,
                wo_off: base + st.wo.offset,
                // Prefer attn_post_norm when ffn_norm sentinel is absent (length=0).
                // Qwen3.5-35B-A3B uses post_attention_norm.weight as the FFN pre-norm;
                // ffn_norm is left as a zero-sentinel (offset=0, length=0) in the LBC.
                // Using offset=0 would read attn_qkv/attn_q Q4_0 data as F32 → NaN.
                ffn_norm_off: if st.ffn_norm.length == 0 {
                    st.attn_post_norm.map_or(0, |s| base + s.offset)
                } else {
                    base + st.ffn_norm.offset
                },
                w_gate_off: base + st.w_gate.offset,
                w_up_off: base + st.w_up.offset,
                w_down_off: base + st.w_down.offset,
                wq_quant: st.wq.quant,
                wo_quant: st.wo.quant,
                w_gate_quant: st.w_gate.quant,
                w_up_quant: st.w_up.quant,
                w_down_quant: st.w_down.quant,
                bq_off: st.bq.map(|b| base + b.offset),
                bk_off: st.bk.map(|b| base + b.offset),
                bv_off: st.bv.map(|b| base + b.offset),
                // MoE metadata: populated from SubtensorOffsets when this layer
                // has router_weight and experts (MoE model). None for dense layers.
                moe_meta: match (&st.router_weight, &st.experts) {
                    (Some(router), Some(experts)) if !experts.is_empty() => {
                        // Use the first expert's quant schemes as representative
                        // (all experts in a layer share the same quantization).
                        let first = &experts[0];
                        Some(CachedMoeMeta {
                            router_weight_off: base + router.offset,
                            expert_gate_offs: experts.iter().map(|e| base + e.gate.offset).collect(),
                            expert_up_offs: experts.iter().map(|e| base + e.up.offset).collect(),
                            expert_down_offs: experts.iter().map(|e| base + e.down.offset).collect(),
                            expert_gate_quant: first.gate.quant,
                            expert_down_quant: first.down.quant,
                        })
                    }
                    _ => None,
                },
                // Shared expert offsets (Qwen3.5-MoE).
                shared_expert_gate_off: st.shared_expert_gate.map(|s| base + s.offset),
                shared_expert_up_off: st.shared_expert_up.map(|s| base + s.offset),
                shared_expert_down_off: st.shared_expert_down.map(|s| base + s.offset),
                shared_expert_gate_quant: st.shared_expert_gate.map(|s| s.quant),
                shared_expert_down_quant: st.shared_expert_down.map(|s| s.quant),
                // Extended attention fields.
                attn_gate_off: st.attn_gate.map(|s| base + s.offset),
                attn_gate_quant: st.attn_gate.map(|s| s.quant),
                attn_post_norm_off: st.attn_post_norm.map(|s| base + s.offset),

                // Q+gate fusion: active for Qwen3.5 full-attention layers where
                // attn_q.weight contains interleaved Q+gate (8192 output rows).
                // Detected by presence of attn_q_norm (per-head Q RMSNorm), which
                // only exists on full-attention layers with separate Q/K/V projections.
                // When true, the decode path deinterleaves Q+gate, projects K/V
                // separately from wk/wv, and applies SiLU-gated output.
                has_qgate_fusion: st.attn_q_norm.is_some(),
                wk_off: if st.wk.length > 0 { Some(base + st.wk.offset) } else { None },
                wv_off: if st.wv.length > 0 { Some(base + st.wv.offset) } else { None },
                wk_quant: if st.wk.length > 0 { Some(st.wk.quant) } else { None },
                wv_quant: if st.wv.length > 0 { Some(st.wv.quant) } else { None },
                // Per-head Q/K RMSNorm weights.
                attn_q_norm_off: st.attn_q_norm.map(|s| base + s.offset),
                attn_k_norm_off: st.attn_k_norm.map(|s| base + s.offset),
                // Shared expert gate input weight.
                ffn_gate_inp_shexp_off: st.ffn_gate_inp_shexp.map(|s| base + s.offset),

                // Layer type discriminator.
                layer_type: st.layer_type,

                // GatedDeltaNet offsets.
                ssm_a_off: st.ssm_a.map(|s| base + s.offset),
                ssm_conv1d_off: st.ssm_conv1d.map(|s| base + s.offset),
                ssm_dt_off: st.ssm_dt.map(|s| base + s.offset),
                ssm_beta_off: st.ssm_beta.map(|s| base + s.offset),
                ssm_alpha_off: st.ssm_alpha.map(|s| base + s.offset),
                ssm_norm_off: st.ssm_norm.map(|s| base + s.offset),
                ssm_out_off: st.ssm_out.map(|s| base + s.offset),
                ssm_out_quant: st.ssm_out.map(|s| s.quant),
                gdn_layer_idx: if st.layer_type == Some(1) {
                    let idx = gdn_layer_counter;
                    gdn_layer_counter += 1;
                    Some(idx)
                } else {
                    None
                },
            });
            layer_offsets.push(cursor);
            layer_blobs.push(blob.to_vec());
            cursor = align(cursor + blob.len());
        }

        // Append global tensors at page-aligned offsets.
        // For large-vocab models (>64K vocab), the embedding + output_proj tables
        // can exceed 2 GB, causing a 3 GB private buffer that degrades GPU cache
        // performance. Only pack globals into the unified buffer when they're small.
        let embed_buf_ref = self.embedding_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Embedding buffer not initialized for unified preload".into())
        })?;
        let embed_len = embed_buf_ref.length() as usize;

        let norm_buf_ref = self.final_norm_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Final norm buffer not initialized for unified preload".into())
        })?;
        let norm_len = norm_buf_ref.length() as usize;

        let proj_buf_ref = self.output_proj_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Output proj buffer not initialized for unified preload".into())
        })?;
        let proj_len = proj_buf_ref.length() as usize;

        // Weight tying: output_proj shares embedding storage (no separate allocation)
        let effective_proj_len = if self.weight_tying { 0 } else { proj_len };
        let global_bytes = embed_len + norm_len + effective_proj_len;
        // Include globals in the unified private buffer.
        let include_globals = true;

        let (embed_offset, norm_offset, proj_offset) = if include_globals {
            let eo = cursor;
            cursor = align(cursor + embed_len);
            let no = cursor;
            cursor = align(cursor + norm_len);
            if self.weight_tying {
                // output_proj reuses embedding offset
                (eo, no, eo)
            } else {
                let po = cursor;
                cursor = align(cursor + proj_len);
                (eo, no, po)
            }
        } else {
            (0, 0, 0)
        };

        let total_size = cursor;

        // === Pass 2: Allocate shared staging buffer and copy all data via CPU ===
        let staging_buf = self.device.new_buffer(total_size).ok_or_else(|| {
            RuntimeError::Compute(format!(
                "Failed to allocate staging buffer ({} bytes, {:.1} MB)",
                total_size, total_size as f64 / (1024.0 * 1024.0)
            ))
        })?;

        let dst_base = staging_buf.contents() as *mut u8;
        let mut layer_bytes_total: usize = 0;

        for (layer, blob) in layer_blobs.iter().enumerate() {
            let off = layer_offsets[layer];
            unsafe {
                std::ptr::copy_nonoverlapping(blob.as_ptr(), dst_base.add(off), blob.len());
            }
            layer_bytes_total += blob.len();
        }

        if include_globals {
            // Copy global tensors from their existing Metal buffers
            unsafe {
                std::ptr::copy_nonoverlapping(
                    embed_buf_ref.contents() as *const u8, dst_base.add(embed_offset), embed_len,
                );
                std::ptr::copy_nonoverlapping(
                    norm_buf_ref.contents() as *const u8, dst_base.add(norm_offset), norm_len,
                );
                if !self.weight_tying {
                    std::ptr::copy_nonoverlapping(
                        proj_buf_ref.contents() as *const u8, dst_base.add(proj_offset), proj_len,
                    );
                }
            }
        }

        // Free temporary layer blobs before allocating private buffer
        drop(layer_blobs);

        // === Pass 3: Blit copy from shared staging to private GPU-only buffer ===
        let private_buf = self.device.new_buffer_private(total_size).ok_or_else(|| {
            RuntimeError::Compute(format!(
                "Failed to allocate private GPU buffer ({} bytes, {:.1} MB)",
                total_size, total_size as f64 / (1024.0 * 1024.0)
            ))
        })?;

        let blit_cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for weight blit".into())
        })?;
        let blit_enc = blit_cmd.new_blit_encoder().ok_or_else(|| {
            RuntimeError::Compute("Failed to create blit encoder for weight copy".into())
        })?;
        blit_enc.copy_from_buffer(&staging_buf, 0, &private_buf, 0, total_size as u64);
        blit_enc.end_encoding();
        blit_cmd.commit_and_wait();

        // Staging buffer dropped here, freeing shared memory
        drop(staging_buf);

        let layer_mb = layer_bytes_total as f64 / (1024.0 * 1024.0);
        let global_mb = if include_globals { global_bytes as f64 / (1024.0 * 1024.0) } else { 0.0 };
        let total_mb = total_size as f64 / (1024.0 * 1024.0);
        if include_globals {
            println!(
                "GPU-resident private buffer: {} layers ({:.1} MB) + globals ({:.1} MB) = {:.1} MB (StorageModePrivate)",
                num_layers, layer_mb, global_mb, total_mb,
            );
        } else {
            println!(
                "GPU-resident private buffer: {} layers ({:.1} MB) = {:.1} MB (StorageModePrivate); globals ({:.1} MB) in shared buffers",
                num_layers, layer_mb, total_mb,
                global_bytes as f64 / (1024.0 * 1024.0),
            );
        }

        s.gpu_unified_weight_buf = Some(private_buf);
        s.gpu_layer_offsets = layer_offsets;
        if include_globals {
            s.gpu_global_offsets = Some((embed_offset, norm_offset, proj_offset));
        } else {
            s.gpu_global_offsets = None;  // Forces fallback to separate shared buffers
        }
        s.cached_layer_meta = layer_metas;

        // ====================================================================
        // Qwen3.5-MoE detection
        // ====================================================================
        // Detect hybrid architecture from format-level metadata:
        //   1. Has shared expert weights (shared_expert_gate on at least one layer)
        //   2. Has layer_type discriminators (some layers have layer_type = Some(0) or Some(1))
        //   3. Has MoE routing (at least one layer with moe_meta)
        {
            let has_layer_types = s.cached_layer_meta.iter().any(|m| m.layer_type.is_some());
            let has_moe = s.cached_layer_meta.iter().any(|m| m.moe_meta.is_some());

            // Detection: Qwen3.5 family has hybrid layer types (GDN + full attention).
            // MoE variant (Qwen3.5-35B-A3B) also has MoE routing.
            // Dense variant (Qwen3.5-9B) has layer_types but no MoE.
            if has_layer_types {
                // All Qwen3.5 variants (MoE and dense) use NeoX-style RoPE
                s.is_qwen35moe = true;
                if has_moe {
                    // Shared expert intermediate dimension
                    s.shared_expert_inter_dim = s.inter_dim;
                    let se_inter = s.shared_expert_inter_dim;
                    let hidden = s.hidden_dim;
                    s.shared_expert_gate_buf = Some(self.device.new_buffer(se_inter * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate shared_expert_gate_buf".into())
                    })?);
                    s.shared_expert_down_buf = Some(self.device.new_buffer(hidden * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate shared_expert_down_buf".into())
                    })?);
                }

                // Partial RoPE: Qwen3.5 uses partial_rotary_factor=0.25,
                // meaning only the first head_dim/4 dimensions of each head are rotated.
                let head_dim = s.head_dim;
                if head_dim >= 128 {
                    s.rotary_dim = head_dim / 4;  // 128/4 = 32 for Qwen3.5-9B, 256/4 = 64 for -35B
                }

                // Allocate attention gate scratch buffer (for full attention layers with attn_gate)
                let has_attn_gate = s.cached_layer_meta.iter().any(|m| m.attn_gate_off.is_some());
                if has_attn_gate {
                    let hidden = s.hidden_dim;
                    s.attn_gate_buf = Some(self.device.new_buffer(hidden * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate attn_gate_buf".into())
                    })?);
                }

                // Recompute RoPE cos/sin tables for partial rotation with Qwen3.5 theta.
                // Qwen3.5 uses rope_theta = 10_000_000 (10M) and partial rotation.
                let rotary_half_dim = s.rotary_dim / 2;
                let theta: f64 = 10_000_000.0;
                let max_seq = s.max_seq_len;
                let mut cos_table = vec![0.0f32; max_seq * rotary_half_dim];
                let mut sin_table = vec![0.0f32; max_seq * rotary_half_dim];
                for pos in 0..max_seq {
                    for i in 0..rotary_half_dim {
                        let freq = 1.0 / theta.powf((2 * i) as f64 / s.rotary_dim as f64);
                        let angle = pos as f64 * freq;
                        cos_table[pos * rotary_half_dim + i] = angle.cos() as f32;
                        sin_table[pos * rotary_half_dim + i] = angle.sin() as f32;
                    }
                }
                // Upload new tables to existing RoPE buffers (resize if needed).
                let new_rope_bytes = cos_table.len() * 4;
                s.rope_cos_buf = self.device.new_buffer(new_rope_bytes).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate partial RoPE cos buffer".into())
                })?;
                s.rope_sin_buf = self.device.new_buffer(new_rope_bytes).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate partial RoPE sin buffer".into())
                })?;
                s.rope_cos_buf.write_f32(&cos_table);
                s.rope_sin_buf.write_f32(&sin_table);

                // Count layer types for diagnostics
                let n_linear = s.cached_layer_meta.iter().filter(|m| m.layer_type == Some(1)).count();
                let n_full = s.cached_layer_meta.iter().filter(|m| m.layer_type == Some(0)).count();
                let n_moe = s.cached_layer_meta.iter().filter(|m| m.moe_meta.is_some()).count();
                let n_shared = s.cached_layer_meta.iter().filter(|m| m.shared_expert_gate_off.is_some()).count();
                let se_inter_display = s.shared_expert_inter_dim;
                println!(
                    "Qwen3.5 hybrid detected: {} layers ({} linear_attn, {} full_attn), {} MoE, {} shared_expert, rotary_dim={}, se_inter={}",
                    num_layers, n_linear, n_full, n_moe, n_shared, s.rotary_dim, se_inter_display,
                );

            }
        }

        // ====================================================================
        // GatedDeltaNet state allocation
        // ====================================================================
        // Allocate persistent h_state and conv_state buffers for all GDN layers.
        // This runs for ANY model with layer_type=1 layers (both MoE and dense).
        {
            let n_linear = s.cached_layer_meta.iter().filter(|m| m.layer_type == Some(1)).count();
            if n_linear > 0 {
                const GDN_NUM_HEADS: usize = 32;    // ssm.time_step_rank
                const GDN_HEAD_DIM: usize = 128;    // ssm.state_size
                const GDN_QKV_DIM: usize = 8192;    // Q(2048) + K(2048) + V(4096)
                let conv_kernel_size: usize = 4;    // Qwen3.5 uses kernel_size=4
                let hidden = s.hidden_dim;

                // h_state: [GDN_NUM_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM] per GDN layer
                let h_state_size = GDN_NUM_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM;
                // conv_state: [(kernel_size - 1) * GDN_QKV_DIM] per GDN layer
                let conv_state_size = (conv_kernel_size - 1) * GDN_QKV_DIM;

                let mut h_states = Vec::with_capacity(n_linear);
                let mut conv_states = Vec::with_capacity(n_linear);
                for _ in 0..n_linear {
                    let h_buf = self.device.new_buffer(h_state_size * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN h_state buffer".into())
                    })?;
                    // Zero-initialize h_state (new sequence starts with zero state)
                    h_buf.write_f32(&vec![0.0f32; h_state_size]);
                    h_states.push(h_buf);

                    let c_buf = self.device.new_buffer(conv_state_size * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN conv_state buffer".into())
                    })?;
                    c_buf.write_f32(&vec![0.0f32; conv_state_size]);
                    conv_states.push(c_buf);
                }

                s.gdn_h_states = h_states;
                s.gdn_conv_states = conv_states;
                s.gdn_conv_positions = vec![0u32; n_linear];
                s.gdn_conv_kernel_size = conv_kernel_size;
                s.gdn_num_layers = n_linear;

                // Allocate GDN scratch buffers using GDN-specific dimensions
                let gdn_q_dim = GDN_NUM_HEADS * GDN_HEAD_DIM; // 4096
                s.gdn_alpha_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN alpha buffer".into())
                })?);
                s.gdn_beta_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN beta buffer".into())
                })?);
                // Output of state query: [GDN_NUM_HEADS * GDN_HEAD_DIM] = 4096
                s.gdn_output_buf = Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN output buffer".into())
                })?);
                // SSM output projection result: [hidden_dim]
                s.gdn_ssm_proj_buf = Some(self.device.new_buffer(hidden * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN ssm_proj buffer".into())
                })?);
                // Attention gate sigmoid output: [q_dim=4096] (gate applied BEFORE ssm_out_proj)
                s.gdn_gate_sigmoid_buf = Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN gate_sigmoid buffer".into())
                })?);
                // L2-norm scaled output: [GDN_NUM_HEADS * GDN_HEAD_DIM] = 4096
                s.gdn_normed_out_buf = Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN normed_out buffer".into())
                })?);
                // Q8_0 matvec outputs for alpha/beta gate projections [GDN_NUM_HEADS] f32
                s.gdn_alpha_raw_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN alpha_raw buffer".into())
                })?);
                s.gdn_beta_raw_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN beta_raw buffer".into())
                })?);
                // Conv1d output for all QKV channels [GDN_QKV_DIM=8192] f32
                s.gdn_qkv_conv_buf = Some(self.device.new_buffer(GDN_QKV_DIM * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN qkv_conv buffer".into())
                })?);

                let h_state_mb = (n_linear * h_state_size * 4) as f64 / (1024.0 * 1024.0);
                let conv_mb = (n_linear * conv_state_size * 4) as f64 / (1024.0 * 1024.0);
                println!(
                    "GDN state: {} layers, h_state={:.1} MB ({} heads x {}x{} per layer), conv={:.1} MB (kernel_size={})",
                    n_linear, h_state_mb, GDN_NUM_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM, conv_mb, conv_kernel_size,
                );
            }
        }

        // Clear legacy per-layer buffers (unified replaces them)
        s.gpu_resident_layers = None;

        // ====================================================================
        // Build MoE expert offset tables for batched GPU-side dispatch.
        // Upload per-layer offset arrays to GPU buffers so the batched kernels
        // can look up expert weight positions without CPU readback.
        // ====================================================================
        {
            let n_experts = s.moe_num_experts;
            if n_experts > 0 {
                let mut gate_up_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
                let mut down_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
                for meta in &s.cached_layer_meta {
                    if let Some(ref moe_meta) = meta.moe_meta {
                        // Build gate+up offset table: [n_experts * 2] u64
                        let mut gu_offsets = vec![0u64; n_experts * 2];
                        for e in 0..n_experts.min(moe_meta.expert_gate_offs.len()) {
                            gu_offsets[e * 2]     = moe_meta.expert_gate_offs[e];
                            gu_offsets[e * 2 + 1] = moe_meta.expert_up_offs[e];
                        }
                        let gu_bytes: Vec<u8> = gu_offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let gu_buf = self.device.new_buffer_with_bytes(&gu_bytes).ok_or_else(|| {
                            RuntimeError::Compute("Failed to allocate MoE gate_up offset table".into())
                        })?;
                        gate_up_vecs.push(Some(gu_buf));

                        // Build down offset table: [n_experts] u64
                        let mut d_offsets = vec![0u64; n_experts];
                        for e in 0..n_experts.min(moe_meta.expert_down_offs.len()) {
                            d_offsets[e] = moe_meta.expert_down_offs[e];
                        }
                        let d_bytes: Vec<u8> = d_offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let d_buf = self.device.new_buffer_with_bytes(&d_bytes).ok_or_else(|| {
                            RuntimeError::Compute("Failed to allocate MoE down offset table".into())
                        })?;
                        down_vecs.push(Some(d_buf));
                    } else {
                        gate_up_vecs.push(None);
                        down_vecs.push(None);
                    }
                }
                s.moe_gate_up_offsets = gate_up_vecs;
                s.moe_down_offsets = down_vecs;

                // Build shared expert down offset tables for fused kernels.
                // Each MoE layer with a shared expert gets a single u64 GPU buffer
                // containing the byte offset of the shared expert down weight matrix.
                let mut se_down_vecs: Vec<Option<MetalBuffer>> = Vec::with_capacity(num_layers);
                for meta in &s.cached_layer_meta {
                    if let Some(se_down_off) = meta.shared_expert_down_off {
                        let off_bytes: Vec<u8> = se_down_off.to_le_bytes().to_vec();
                        let buf = self.device.new_buffer_with_bytes(&off_bytes).ok_or_else(|| {
                            RuntimeError::Compute("Failed to allocate MoE shared expert down offset".into())
                        })?;
                        se_down_vecs.push(Some(buf));
                    } else {
                        se_down_vecs.push(None);
                    }
                }
                s.moe_shared_down_offsets = se_down_vecs;
            }
        }

        Ok(())
    }

    /// Dispatch a matmul_bytes_f32 kernel: out = W_bytes * x
    ///
    /// Note: Not used by the optimized compute_layer (which inlines encoding into
    /// batched command buffers with zero-copy offsets), but retained for testing
    /// and for potential use by future code paths.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_matmul_bytes(
        &self,
        pipelines: &MetalPipelines,
        w_bytes: &[u8],
        x_buf: &MetalBuffer,
        out_buf: &MetalBuffer,
        out_dim: usize,
        in_dim: usize,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        // Create a buffer wrapping the weight bytes (copy for safety)
        let w_buf = self.device.new_buffer_with_bytes(w_bytes).ok_or_else(|| {
            RuntimeError::Compute("Failed to create weight buffer for matmul".into())
        })?;

        let in_dim_u32 = in_dim as u32;

        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for matmul".into())
        })?;
        let enc = cmd.new_compute_encoder().ok_or_else(|| {
            RuntimeError::Compute("Failed to create compute encoder for matmul".into())
        })?;

        enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
        enc.set_buffer(&w_buf, 0, 0);
        enc.set_buffer(x_buf, 0, 1);
        enc.set_buffer(out_buf, 0, 2);
        enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
        enc.dispatch_threadgroups(
            MTLSize::new(out_dim as u64, 1, 1),
            MTLSize::new(scratch.matmul_tg_size, 1, 1),
        );
        enc.end_encoding();
        cmd.commit_and_wait();

        Ok(())
    }

    /// Dispatch a dequant_matmul_q8_0 kernel: out = dequant(W_q8) * x
    ///
    /// The kernel performs fused Q8_0 dequantization and matrix-vector multiply.
    /// `in_dim` is the element count (not byte stride). The kernel computes the
    /// Q8_0 row byte stride internally from `in_dim`.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_matmul_q8_0(
        &self,
        pipelines: &MetalPipelines,
        w_bytes: &[u8],
        x_buf: &MetalBuffer,
        out_buf: &MetalBuffer,
        out_dim: usize,
        in_dim: usize,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        let w_buf = self.device.new_buffer_with_bytes(w_bytes).ok_or_else(|| {
            RuntimeError::Compute("Failed to create weight buffer for Q8_0 matmul".into())
        })?;

        let in_dim_u32 = in_dim as u32;

        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for Q8_0 matmul".into())
        })?;
        let enc = cmd.new_compute_encoder().ok_or_else(|| {
            RuntimeError::Compute("Failed to create compute encoder for Q8_0 matmul".into())
        })?;

        enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0);
        enc.set_buffer(&w_buf, 0, 0);
        enc.set_buffer(x_buf, 0, 1);
        enc.set_buffer(out_buf, 0, 2);
        enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
        enc.dispatch_threadgroups(
            MTLSize::new(out_dim as u64, 1, 1),
            MTLSize::new(scratch.matmul_tg_size, 1, 1),
        );
        enc.end_encoding();
        cmd.commit_and_wait();

        Ok(())
    }

    /// Dispatch the appropriate matmul kernel based on quantization scheme.
    ///
    /// For Q8_0 weights, uses the fused `dequant_matmul_q8_0` kernel.
    /// For F32/unquantized weights, uses `matmul_bytes_f32` (cast uchar* to float*).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_matmul_for_quant(
        &self,
        pipelines: &MetalPipelines,
        w_bytes: &[u8],
        x_buf: &MetalBuffer,
        out_buf: &MetalBuffer,
        out_dim: usize,
        in_dim: usize,
        quant: QuantScheme,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        match quant {
            QuantScheme::Q8_0 => {
                self.dispatch_matmul_q8_0(
                    pipelines, w_bytes, x_buf, out_buf, out_dim, in_dim, scratch,
                )
            }
            _ => {
                self.dispatch_matmul_bytes(
                    pipelines, w_bytes, x_buf, out_buf, out_dim, in_dim, scratch,
                )
            }
        }
    }

    /// Dispatch rmsnorm_bytes kernel.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_rmsnorm_bytes(
        &self,
        pipelines: &MetalPipelines,
        x_buf: &MetalBuffer,
        w_bytes: &[u8],
        out_buf: &MetalBuffer,
        dim: usize,
        eps: f32,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        let w_buf = self.device.new_buffer_with_bytes(w_bytes).ok_or_else(|| {
            RuntimeError::Compute("Failed to create weight buffer for rmsnorm".into())
        })?;
        let dim_u32 = dim as u32;

        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for rmsnorm".into())
        })?;
        let enc = cmd.new_compute_encoder().ok_or_else(|| {
            RuntimeError::Compute("Failed to create compute encoder for rmsnorm".into())
        })?;

        enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
        enc.set_buffer(x_buf, 0, 0);
        enc.set_buffer(&w_buf, 0, 1);
        enc.set_buffer(out_buf, 0, 2);
        enc.set_bytes(&dim_u32.to_le_bytes(), 3);
        enc.set_bytes(&eps.to_le_bytes(), 4);
        enc.dispatch_threadgroups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(scratch.norm_tg_size, 1, 1),
        );
        enc.end_encoding();
        cmd.commit_and_wait();

        Ok(())
    }

    // ========================================================================
    // Batched prefill methods
    // ========================================================================

    /// Ensure batch scratch buffers are allocated for at least `batch_size` tokens.
    /// Lazily (re-)allocates if `batch_size > current_max_batch`.
    ///
    /// The Split-K partial buffer is sized based on the actual GEMM dimensions that
    /// trigger Split-K at this batch_size. This avoids over-allocating for models
    /// with large inter_dim (e.g. Llama 3.2: inter_dim=8192 never triggers Split-K
    /// at pp128, saving 20 MB of allocation).
    fn ensure_batch_buffers(
        &self,
        scratch: &mut MetalScratch,
        batch_size: usize,
    ) -> Result<(), RuntimeError> {
        let hidden_dim = scratch.hidden_dim;
        let q_dim = scratch.q_dim;
        let kv_dim = scratch.kv_dim;
        let qkv_dim = scratch.qkv_dim;
        let inter_dim = scratch.inter_dim;
        let num_heads = scratch.num_heads;
        let max_seq_len = scratch.max_seq_len;

        // Check if activation buffers need (re-)allocation
        if batch_size > scratch.current_max_batch {
            let make = |n: usize| -> Result<MetalBuffer, RuntimeError> {
                let len = n.max(1) * 4;
                self.device.new_buffer(len).ok_or_else(|| {
                    RuntimeError::Compute(format!("Failed to allocate batch buffer of {len} bytes"))
                })
            };

            scratch.batch_x_buf = Some(make(batch_size * hidden_dim)?);
            scratch.batch_normed_buf = Some(make(batch_size * hidden_dim)?);
            // Q+gate fusion (Qwen3.5 full-attention layers): attn_q.weight outputs
            // 2*q_dim interleaved Q+gate, larger than qkv_dim. Allocate the max.
            let qgate_dim = q_dim * 2;
            scratch.batch_qkv_buf = Some(make(batch_size * qkv_dim.max(qgate_dim))?);
            scratch.batch_q_buf = Some(make(batch_size * q_dim)?);
            scratch.batch_k_buf = Some(make(batch_size * kv_dim)?);
            scratch.batch_v_buf = Some(make(batch_size * kv_dim)?);
            scratch.batch_attn_out_buf = Some(make(batch_size * q_dim)?);
            scratch.batch_attn_proj_buf = Some(make(batch_size * hidden_dim)?);
            // Gate buf must hold max(inter_dim, q_dim) for FFN gate and Q+gate fusion.
            scratch.batch_gate_buf = Some(make(batch_size * inter_dim.max(q_dim))?);
            scratch.batch_up_buf = Some(make(batch_size * inter_dim)?);
            scratch.batch_down_buf = Some(make(batch_size * hidden_dim)?);
            // Scores buffer for batched attention (f16). With dense stride (scores_stride=max_attend_len),
            // the actual usage is batch_size * num_heads * max_attend_len. Since prefill starts at
            // position 0, max_attend_len = batch_size. Allocate exactly for batch_size (the maximum
            // attend length for a single prefill starting at position 0). If chunked prefill is
            // added later, this buffer will be re-allocated via the batch_size check above.
            let scores_max_attend = max_seq_len.min(batch_size);
            let scores_bytes = (batch_size * num_heads * scores_max_attend).max(1) * 2; // f16 = 2 bytes
            scratch.batch_scores_buf = Some(self.device.new_buffer(scores_bytes).ok_or_else(|| {
                RuntimeError::Compute(format!("Failed to allocate batch_scores_buf of {scores_bytes} bytes"))
            })?);

            // MoE batched scratch buffers (only when model has experts)
            if scratch.moe_num_experts > 0 {
                let ne = scratch.moe_num_experts;
                let top_k = scratch.moe_num_active_experts;
                scratch.moe_batch_router_logits = Some(make(batch_size * ne)?);
                scratch.moe_batch_expert_ids = Some(
                    self.device.new_buffer((batch_size * top_k).max(1) * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate MoE batch expert_ids".into())
                    })?
                );
                scratch.moe_batch_expert_weights = Some(make(batch_size * top_k)?);
                scratch.moe_batch_expert_output = Some(make(batch_size * ne * hidden_dim)?);
            }

            scratch.current_max_batch = batch_size;
        }

        // Split-K partial buffer: sized for the ACTUAL maximum N that triggers
        // Split-K at this batch_size. Unlike activation buffers which grow
        // monotonically with batch_size, the Split-K requirement can be LARGER
        // for smaller batch sizes (more GEMMs trigger Split-K when there are
        // fewer threadgroups). We track this independently.
        let k_splits: usize = 8;
        let max_splitk_n = Self::max_splitk_output_dim(
            batch_size, hidden_dim, q_dim, kv_dim, qkv_dim, inter_dim,
        );
        let required_splitk_size = if max_splitk_n > 0 {
            k_splits * batch_size * max_splitk_n
        } else {
            1 // minimal allocation (4 bytes) -- no GEMM triggers Split-K
        };
        if required_splitk_size > scratch.splitk_alloc_elems {
            let len = required_splitk_size.max(1) * 4;
            scratch.splitk_partial_buf = Some(
                self.device.new_buffer(len).ok_or_else(|| {
                    RuntimeError::Compute(format!("Failed to allocate splitk buffer of {len} bytes"))
                })?
            );
            scratch.splitk_alloc_elems = required_splitk_size;
        }

        Ok(())
    }

    /// Determine whether Split-K GEMM should be used for a given GEMM shape.
    ///
    /// Split-K adds parallelism along the K dimension when the 2D tile grid
    /// (ceil(N/TILE_N) * ceil(M/TILE_M)) is too small to saturate the GPU.
    ///
    /// Returns the number of K splits to use (0 = don't use Split-K).
    fn splitk_splits(_m: usize, _n: usize, _k: usize, _batch_size: usize) -> u32 {
        // Split-K disabled. Benchmarking across 3 models showed that Split-K
        // overhead (partial-buffer writes + reduce kernel + extra encoder) always
        // exceeds the occupancy benefit on M3 Ultra (60 cores). Even at pp32 with
        // 64 TGs (~1 TG/core), the non-Split-K path matches or beats Split-K.
        //
        // The Split-K kernel and reduce_splitk pipeline are retained for future
        // use if needed for very small batch sizes on GPUs with more cores.
        0
    }

    /// Compute the maximum output dimension (N) that actually triggers Split-K
    /// for a given batch_size. Used to right-size the Split-K partial buffer.
    ///
    /// With Split-K disabled, this always returns 0.
    fn max_splitk_output_dim(
        batch_size: usize,
        hidden_dim: usize,
        q_dim: usize,
        _kv_dim: usize,
        qkv_dim: usize,
        inter_dim: usize,
    ) -> usize {
        // All GEMM dispatch sites in encode_layer_batched:
        // 1. Fused QKV:  N=qkv_dim,   K=hidden_dim
        // 2. Attn proj:  N=hidden_dim, K=q_dim
        // 3. Gate (FFN): N=inter_dim,  K=hidden_dim
        // 4. Up (FFN):   N=inter_dim,  K=hidden_dim
        // 5. Down (FFN): N=hidden_dim, K=inter_dim
        let sites: [(usize, usize); 5] = [
            (qkv_dim, hidden_dim),
            (hidden_dim, q_dim),
            (inter_dim, hidden_dim),
            (inter_dim, hidden_dim),
            (hidden_dim, inter_dim),
        ];

        let mut max_n = 0usize;
        for (n, k) in sites {
            if Self::splitk_splits(batch_size, n, k, batch_size) > 0 {
                max_n = max_n.max(n);
            }
        }
        max_n
    }

    /// Dispatch a Split-K Q8_0 GEMM: two-phase (split GEMM + reduce).
    ///
    /// Phase 1: dispatch dequant_tiled_matmul_q8_0_splitk with 3D grid
    /// Phase 2: dispatch reduce_splitk to sum partial results
    ///
    /// Both phases share a single compute encoder with a memory barrier between them.
    #[allow(clippy::too_many_arguments)]
    fn encode_splitk_q8_gemm(
        enc: &crate::metal::ffi::MetalComputeEncoder,
        pipelines: &MetalPipelines,
        w_buf: &MetalBuffer,
        w_off: u64,
        x_buf: &MetalBuffer,
        y_buf: &MetalBuffer,
        partial_buf: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        k_splits: u32,
    ) {
        const TILE_M: u64 = 32;
        const TILE_N: u64 = 32;

        // Phase 1: Split-K GEMM into partial buffer
        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_splitk);
        enc.set_threadgroup_memory_length(4096, 0);
        enc.set_buffer(w_buf, w_off, 0);
        enc.set_buffer(x_buf, 0, 1);
        enc.set_buffer(partial_buf, 0, 2);
        enc.set_bytes(&m.to_le_bytes(), 3);
        enc.set_bytes(&n.to_le_bytes(), 4);
        enc.set_bytes(&k.to_le_bytes(), 5);
        enc.set_bytes(&k_splits.to_le_bytes(), 6);
        enc.dispatch_threadgroups(
            MTLSize::new(
                (n as u64).div_ceil(TILE_N),
                (m as u64).div_ceil(TILE_M),
                k_splits as u64,
            ),
            MTLSize::new(128, 1, 1),
        );

        // Memory barrier: ensure all Split-K writes complete before reduce reads
        enc.memory_barrier_with_scope(1); // MTLBarrierScope.buffers = 1

        // Phase 2: Reduce partial results into final output
        enc.set_pipeline_state(&pipelines.reduce_splitk);
        enc.set_buffer(partial_buf, 0, 0);
        enc.set_buffer(y_buf, 0, 1);
        enc.set_bytes(&m.to_le_bytes(), 2);
        enc.set_bytes(&n.to_le_bytes(), 3);
        enc.set_bytes(&k_splits.to_le_bytes(), 4);
        let total = (m as u64) * (n as u64);
        enc.dispatch_threadgroups(
            MTLSize::new(total.div_ceil(256), 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Dispatch a Split-K Q4_0 GEMM: same as Q8_0 variant but uses Q4_0 kernel.
    #[allow(clippy::too_many_arguments)]
    fn encode_splitk_q4_gemm(
        enc: &crate::metal::ffi::MetalComputeEncoder,
        pipelines: &MetalPipelines,
        w_buf: &MetalBuffer,
        w_off: u64,
        x_buf: &MetalBuffer,
        y_buf: &MetalBuffer,
        partial_buf: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        k_splits: u32,
    ) {
        const TILE_M: u64 = 32;
        const TILE_N: u64 = 32;

        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_splitk);
        enc.set_threadgroup_memory_length(4096, 0);
        enc.set_buffer(w_buf, w_off, 0);
        enc.set_buffer(x_buf, 0, 1);
        enc.set_buffer(partial_buf, 0, 2);
        enc.set_bytes(&m.to_le_bytes(), 3);
        enc.set_bytes(&n.to_le_bytes(), 4);
        enc.set_bytes(&k.to_le_bytes(), 5);
        enc.set_bytes(&k_splits.to_le_bytes(), 6);
        enc.dispatch_threadgroups(
            MTLSize::new(
                (n as u64).div_ceil(TILE_N),
                (m as u64).div_ceil(TILE_M),
                k_splits as u64,
            ),
            MTLSize::new(128, 1, 1),
        );

        enc.memory_barrier_with_scope(1);

        enc.set_pipeline_state(&pipelines.reduce_splitk);
        enc.set_buffer(partial_buf, 0, 0);
        enc.set_buffer(y_buf, 0, 1);
        enc.set_bytes(&m.to_le_bytes(), 2);
        enc.set_bytes(&n.to_le_bytes(), 3);
        enc.set_bytes(&k_splits.to_le_bytes(), 4);
        let total = (m as u64) * (n as u64);
        enc.dispatch_threadgroups(
            MTLSize::new(total.div_ceil(256), 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Dispatch Split-K GEMM for either Q8_0 or Q4_0 based on quant scheme.
    #[allow(clippy::too_many_arguments)]
    fn encode_splitk_gemm_for_quant(
        enc: &crate::metal::ffi::MetalComputeEncoder,
        pipelines: &MetalPipelines,
        quant: QuantScheme,
        w_buf: &MetalBuffer,
        w_off: u64,
        x_buf: &MetalBuffer,
        y_buf: &MetalBuffer,
        partial_buf: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        k_splits: u32,
    ) {
        match quant {
            QuantScheme::Q4_0 => Self::encode_splitk_q4_gemm(
                enc, pipelines, w_buf, w_off, x_buf, y_buf, partial_buf, m, n, k, k_splits,
            ),
            _ => Self::encode_splitk_q8_gemm(
                enc, pipelines, w_buf, w_off, x_buf, y_buf, partial_buf, m, n, k, k_splits,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_layer_batched(
        &self,
        cmd: &MetalCommandBuffer,
        layer_idx: usize,
        batch_size: usize,
        weights: &LayerView,
        kv: &mut KvCacheView,
        pipelines: &MetalPipelines,
        scratch: &mut MetalScratch,
    ) -> Result<(), RuntimeError> {
        let hidden_dim = scratch.hidden_dim;
        let q_dim = scratch.q_dim;
        let kv_dim = scratch.kv_dim;
        let qkv_dim = scratch.qkv_dim;
        let inter_dim = scratch.inter_dim;
        let head_dim = scratch.head_dim;
        let num_heads = scratch.num_heads;
        let num_kv_heads = scratch.num_kv_heads;
        let eps = scratch.eps;
        let attn_scale = scratch.attn_scale;
        let max_seq_len = scratch.max_seq_len;
        let norm_tg_size = scratch.norm_tg_size;
        let seq_pos_start = kv.seq_len;

        let st = &weights.subtensors;

        // GDN (linear attention, layer_type=1) layers skip the standard attention block
        // during batched prefill. Standard attention uses wrong dimensions for GDN
        // (qkv_dim=5120 vs GDN's 8192, Wo reads offset 0 which maps to attn_norm bytes).
        // This corrupts x_buf with garbage. Bypass: identity attention (attn_proj_buf = x_buf),
        // so only the MoE FFN contributes. GDN h_state accumulates at decode time.
        let is_linear_attn = st.layer_type == Some(1);

        // GPU-resident path: prefer unified private buffer, then per-layer buffers,
        // then fall back to cached zero-copy layer buffer.
        let layer_buf: &MetalBuffer;
        let base_off: u64;
        if let Some(ref ubuf) = scratch.gpu_unified_weight_buf {
            layer_buf = ubuf;
            base_off = scratch.gpu_layer_offsets[layer_idx] as u64;
        } else if let Some(ref layers) = scratch.gpu_resident_layers {
            if let Some(buf) = layers.get(layer_idx) {
                layer_buf = buf;
                base_off = 0;
            } else {
                return Err(RuntimeError::Compute(format!(
                    "gpu_resident_layers missing layer {}", layer_idx
                )));
            }
        } else {
            // Zero-copy layer buffer: cached per-layer (same as decode path).
            let blob = weights.as_bytes();
            let blob_ptr = blob.as_ptr() as usize;
            let cached = scratch.layer_buf_cache.get(layer_idx).and_then(|c| c.as_ref());
            let need_create = match cached {
                Some((ptr, _)) => *ptr != blob_ptr,
                None => true,
            };
            if need_create {
                let buf = self.create_layer_buffer(weights)?;
                scratch.layer_buf_cache[layer_idx] = Some((blob_ptr, buf));
            }
            layer_buf = &scratch.layer_buf_cache[layer_idx].as_ref().unwrap().1;
            base_off = 0;
        }

        // Subtensor offsets into the layer buffer (base_off shifts for unified buffer)
        let attn_norm_off = base_off + st.attn_norm.offset;
        let wq_off = base_off + st.wq.offset;
        let _wk_off = base_off + st.wk.offset;
        let _wv_off = base_off + st.wv.offset;
        let wo_off = base_off + st.wo.offset;
        // Prefer attn_post_norm when ffn_norm is the zero-sentinel (Qwen3.5-35B-A3B).
        let ffn_norm_off = if st.ffn_norm.length == 0 {
            st.attn_post_norm.map_or(0, |s| base_off + s.offset)
        } else {
            base_off + st.ffn_norm.offset
        };
        let w_gate_off = base_off + st.w_gate.offset;
        let w_up_off = base_off + st.w_up.offset;
        let w_down_off = base_off + st.w_down.offset;

        // Get batch buffer references (must be allocated via ensure_batch_buffers first)
        let x_buf = scratch.batch_x_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_x_buf not allocated".into())
        })?;
        let normed_buf = scratch.batch_normed_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_normed_buf not allocated".into())
        })?;
        let qkv_buf = scratch.batch_qkv_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_qkv_buf not allocated".into())
        })?;
        let q_buf = scratch.batch_q_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_q_buf not allocated".into())
        })?;
        let k_buf = scratch.batch_k_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_k_buf not allocated".into())
        })?;
        let v_buf = scratch.batch_v_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_v_buf not allocated".into())
        })?;
        let attn_out_buf = scratch.batch_attn_out_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_attn_out_buf not allocated".into())
        })?;
        let attn_proj_buf = scratch.batch_attn_proj_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_attn_proj_buf not allocated".into())
        })?;
        let gate_buf = scratch.batch_gate_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_gate_buf not allocated".into())
        })?;
        let up_buf = scratch.batch_up_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_up_buf not allocated".into())
        })?;
        let down_buf = scratch.batch_down_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_down_buf not allocated".into())
        })?;
        let scores_buf = scratch.batch_scores_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_scores_buf not allocated".into())
        })?;

        let splitk_partial_buf = scratch.splitk_partial_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("splitk_partial_buf not allocated".into())
        })?;

        // Tile constants for tiled matmul (must match MSL kernel)
        const TILE_M: u64 = 32;
        const TILE_N: u64 = 32;

        // Q+gate fusion flag for this layer (Qwen3.5 full-attention layers).
        let has_qgate_fusion = st.attn_q_norm.is_some();
        // Use partial rotary dimension (e.g. 64 for Qwen3.5) instead of full head_dim/2.
        let rope_half_dim = scratch.rotary_dim / 2;

        if !is_linear_attn {
        // ---- 1+2. Attention RMSNorm + QKV projection ----
        // Two paths: Q+gate fusion (Qwen3.5 full-attention) vs fused QKV (standard).
        {
            let m_u32 = batch_size as u32;
            let k_u32 = hidden_dim as u32;

            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;

            // RMSNorm dispatch: [batch, hidden_dim] -> normed_buf
            let dim_u32 = hidden_dim as u32;
            enc.set_pipeline_state(&pipelines.rmsnorm_batched_bytes);
            enc.set_buffer(x_buf, 0, 0);
            enc.set_buffer(layer_buf, attn_norm_off, 1);
            enc.set_buffer(normed_buf, 0, 2);
            enc.set_bytes(&dim_u32.to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(batch_size as u64, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );

            if has_qgate_fusion {
                // Q+gate fusion (Qwen3.5 full-attention layers, batched prefill).
                // attn_q.weight outputs interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...]
                // K and V come from separate attn_k.weight / attn_v.weight.
                let qgate_dim = q_dim * 2; // 8192 = 16 heads * 2 * 256 head_dim

                // 2a. GEMM: normed_buf @ attn_q.weight -> qkv_buf [batch, 2*q_dim]
                {
                    let n_qgate_u32 = qgate_dim as u32;
                    let gemm_aligned = batch_size % 32 == 0 && qgate_dim % 32 == 0 && hidden_dim % 32 == 0;
                    match st.wq.quant {
                        QuantScheme::Q8_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if gemm_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::Q8_0 => {
                            if gemm_aligned {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0);
                            }
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Q4_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if gemm_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::Q4_0 => {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::F16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if gemm_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::F16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        _ => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                    }
                    enc.set_buffer(layer_buf, wq_off, 0);
                    enc.set_buffer(normed_buf, 0, 1);
                    enc.set_buffer(qkv_buf, 0, 2);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&n_qgate_u32.to_le_bytes(), 4);
                    enc.set_bytes(&k_u32.to_le_bytes(), 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new((qgate_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }

                // 2b. GEMM: normed_buf @ attn_k.weight -> k_buf [batch, kv_dim]
                {
                    let wk_off_val = base_off + st.wk.offset;
                    let n_kv_u32 = kv_dim as u32;
                    let gemm_aligned = batch_size % 32 == 0 && kv_dim % 32 == 0 && hidden_dim % 32 == 0;
                    match st.wk.quant {
                        QuantScheme::Q8_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if gemm_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::Q8_0 => {
                            if gemm_aligned {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0);
                            }
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Q4_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if gemm_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::Q4_0 => {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::F16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if gemm_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::F16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        _ => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                    }
                    enc.set_buffer(layer_buf, wk_off_val, 0);
                    enc.set_buffer(normed_buf, 0, 1);
                    enc.set_buffer(k_buf, 0, 2);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&n_kv_u32.to_le_bytes(), 4);
                    enc.set_bytes(&k_u32.to_le_bytes(), 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new((kv_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }

                // 2c. GEMM: normed_buf @ attn_v.weight -> v_buf [batch, kv_dim]
                {
                    let wv_off_val = base_off + st.wv.offset;
                    let n_kv_u32 = kv_dim as u32;
                    let gemm_aligned = batch_size % 32 == 0 && kv_dim % 32 == 0 && hidden_dim % 32 == 0;
                    match st.wv.quant {
                        QuantScheme::Q8_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if gemm_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::Q8_0 => {
                            if gemm_aligned {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0);
                            }
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Q4_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if gemm_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::Q4_0 => {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::F16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if gemm_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::F16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        _ => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                    }
                    enc.set_buffer(layer_buf, wv_off_val, 0);
                    enc.set_buffer(normed_buf, 0, 1);
                    enc.set_buffer(v_buf, 0, 2);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&n_kv_u32.to_le_bytes(), 4);
                    enc.set_bytes(&k_u32.to_le_bytes(), 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new((kv_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
                enc.end_encoding();

                // ---- 3+4. Deinterleave Q+gate, per-head Q/K RMSNorm, RoPE ----
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for Q+gate deinterleave".into())
                })?;

                // Deinterleave Q+gate: qkv_buf[batch, 2*q_dim] -> q_buf[batch, q_dim] + gate_buf[batch, q_dim]
                // Treat batch*num_heads as effective num_heads for the kernel.
                {
                    let pso = pipelines.deinterleave_qgate.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("deinterleave_qgate pipeline not compiled".into())
                    })?;
                    enc.set_pipeline_state(pso);
                    enc.set_buffer(qkv_buf, 0, 0);
                    enc.set_buffer(q_buf, 0, 1);
                    enc.set_buffer(gate_buf, 0, 2);
                    enc.set_bytes(&(head_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&((batch_size * num_heads) as u32).to_le_bytes(), 4);
                    let total_q = (batch_size * q_dim) as u64;
                    let tg_di = 256u64.min(total_q).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new(total_q.div_ceil(tg_di), 1, 1),
                        MTLSize::new(tg_di, 1, 1),
                    );
                }

                // Per-head Q RMSNorm: q_buf[batch * num_heads * head_dim]
                // Each threadgroup handles one head across all batch items.
                if let Some(ref q_norm) = st.attn_q_norm {
                    let q_norm_off = base_off + q_norm.offset;
                    let pso = pipelines.rmsnorm_per_head.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("rmsnorm_per_head pipeline not compiled".into())
                    })?;
                    let head_dim_u32 = head_dim as u32;
                    let tg_rms = 256u64.min(head_dim as u64).max(32);
                    enc.set_pipeline_state(pso);
                    enc.set_buffer(q_buf, 0, 0);
                    enc.set_buffer(layer_buf, q_norm_off, 1);
                    enc.set_buffer(q_buf, 0, 2);
                    enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                    enc.set_bytes(&eps.to_le_bytes(), 4);
                    enc.dispatch_threadgroups(
                        MTLSize::new((batch_size * num_heads) as u64, 1, 1),
                        MTLSize::new(tg_rms, 1, 1),
                    );
                }

                // Per-head K RMSNorm: k_buf[batch * num_kv_heads * head_dim]
                if let Some(ref k_norm) = st.attn_k_norm {
                    let k_norm_off = base_off + k_norm.offset;
                    let pso = pipelines.rmsnorm_per_head.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("rmsnorm_per_head pipeline not compiled".into())
                    })?;
                    let head_dim_u32 = head_dim as u32;
                    let tg_rms = 256u64.min(head_dim as u64).max(32);
                    enc.set_pipeline_state(pso);
                    enc.set_buffer(k_buf, 0, 0);
                    enc.set_buffer(layer_buf, k_norm_off, 1);
                    enc.set_buffer(k_buf, 0, 2);
                    enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                    enc.set_bytes(&eps.to_le_bytes(), 4);
                    enc.dispatch_threadgroups(
                        MTLSize::new((batch_size * num_kv_heads) as u64, 1, 1),
                        MTLSize::new(tg_rms, 1, 1),
                    );
                }

                // Q RoPE dispatch (partial rotation: rope_half_dim, not head_dim/2)
                let total_q_elems = (batch_size * num_heads * rope_half_dim) as u32;
                let total_k_elems = (batch_size * num_kv_heads * rope_half_dim) as u32;
                let head_dim_u32 = head_dim as u32;
                let rope_half_dim_u32 = rope_half_dim as u32;
                let pos_start_u32 = seq_pos_start as u32;
                let total_dim_u32 = q_dim as u32;
                let total_kv_dim_u32 = kv_dim as u32;

                // Qwen3.5 uses NeoX-style (half-offset) dimension pairing
                let rope_b_pipe = if scratch.is_qwen35moe {
                    pipelines.rope_batched_neox.as_ref().unwrap_or(&pipelines.rope_batched)
                } else {
                    &pipelines.rope_batched
                };
                enc.set_pipeline_state(rope_b_pipe);
                enc.set_buffer(q_buf, 0, 0);
                enc.set_buffer(&scratch.rope_cos_buf, 0, 1);
                enc.set_buffer(&scratch.rope_sin_buf, 0, 2);
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 3);
                enc.set_bytes(&head_dim_u32.to_le_bytes(), 4);
                enc.set_bytes(&rope_half_dim_u32.to_le_bytes(), 5);
                enc.set_bytes(&pos_start_u32.to_le_bytes(), 6);
                enc.set_bytes(&total_dim_u32.to_le_bytes(), 7);
                let tg_q = 256u64.min(total_q_elems as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((total_q_elems as u64).div_ceil(tg_q), 1, 1),
                    MTLSize::new(tg_q, 1, 1),
                );

                // K RoPE dispatch
                enc.set_pipeline_state(rope_b_pipe);
                enc.set_buffer(k_buf, 0, 0);
                enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 3);
                enc.set_bytes(&total_kv_dim_u32.to_le_bytes(), 7);
                let tg_k = 256u64.min(total_k_elems as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((total_k_elems as u64).div_ceil(tg_k), 1, 1),
                    MTLSize::new(tg_k, 1, 1),
                );
                enc.end_encoding();
            } else {
            // Standard fused QKV path (non-Qwen3.5 models).
            let n_qkv_u32 = qkv_dim as u32;

            let splitk = match st.wq.quant {
                QuantScheme::Q8_0 => Self::splitk_splits(batch_size, qkv_dim, hidden_dim, batch_size),
                QuantScheme::Q4_0 => Self::splitk_splits(batch_size, qkv_dim, hidden_dim, batch_size),
                _ => 0,
            };

            // QKV GEMM dispatch (serial ordering ensures normed_buf is ready)
            if splitk > 0 {
                Self::encode_splitk_gemm_for_quant(
                    &enc, pipelines, st.wq.quant, layer_buf, wq_off, normed_buf,
                    qkv_buf, splitk_partial_buf, m_u32, n_qkv_u32, k_u32, splitk,
                );
            } else {
                // Select aligned pipeline when M, N, K are all tile-aligned
                let gemm_aligned = batch_size % 32 == 0 && qkv_dim % 32 == 0 && hidden_dim % 32 == 0;
                match st.wq.quant {
                    QuantScheme::Q8_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                        if gemm_aligned && hidden_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::Q8_0 => {
                        if gemm_aligned {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0);
                        }
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::Q4_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                        if gemm_aligned && hidden_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::Q4_0 => {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::F16 => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    _ => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                }
                enc.set_buffer(layer_buf, wq_off, 0);
                enc.set_buffer(normed_buf, 0, 1);
                enc.set_buffer(qkv_buf, 0, 2);
                enc.set_bytes(&m_u32.to_le_bytes(), 3);
                enc.set_bytes(&n_qkv_u32.to_le_bytes(), 4);
                enc.set_bytes(&k_u32.to_le_bytes(), 5);
                enc.dispatch_threadgroups(
                    MTLSize::new((qkv_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                    MTLSize::new(128, 1, 1),
                );
            }
            enc.end_encoding();

            // ---- 3+4. Deinterleave + optional bias + RoPE (merged concurrent encoder) ----
            let enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create concurrent encoder".into())
            })?;

            // Deinterleave dispatch
            enc.set_pipeline_state(&pipelines.deinterleave_qkv);
            enc.set_buffer(qkv_buf, 0, 0);
            enc.set_buffer(q_buf, 0, 1);
            enc.set_buffer(k_buf, 0, 2);
            enc.set_buffer(v_buf, 0, 3);
            enc.set_bytes(&m_u32.to_le_bytes(), 4);
            enc.set_bytes(&(q_dim as u32).to_le_bytes(), 5);
            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 6);
            enc.set_bytes(&n_qkv_u32.to_le_bytes(), 7);
            let total_elems = (batch_size * qkv_dim) as u64;
            let tg = 256u64.min(total_elems).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new(total_elems.div_ceil(tg), 1, 1),
                MTLSize::new(tg, 1, 1),
            );

            // Barrier: deinterleave must complete before bias/RoPE reads q_buf/k_buf/v_buf
            enc.memory_barrier_with_scope(1); // MTLBarrierScope.buffers

            // QKV bias addition (Qwen2-family models, batched) -- zero-cost skip if no bias
            if st.bq.is_some() || st.bk.is_some() || st.bv.is_some() {
                enc.set_pipeline_state(&pipelines.bias_add_batched);
                if let Some(ref bq) = st.bq {
                    let bq_off = base_off + bq.offset;
                    let q_total = (batch_size * q_dim) as u32;
                    enc.set_buffer(q_buf, 0, 0);
                    enc.set_buffer(layer_buf, bq_off, 1);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 2);
                    enc.set_bytes(&q_total.to_le_bytes(), 3);
                    let n_tg = (q_total as u64 + 255) / 256;
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
                }
                if let Some(ref bk) = st.bk {
                    let bk_off = base_off + bk.offset;
                    let k_total = (batch_size * kv_dim) as u32;
                    enc.set_buffer(k_buf, 0, 0);
                    enc.set_buffer(layer_buf, bk_off, 1);
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 2);
                    enc.set_bytes(&k_total.to_le_bytes(), 3);
                    let n_tg = (k_total as u64 + 255) / 256;
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
                }
                if let Some(ref bv) = st.bv {
                    let bv_off = base_off + bv.offset;
                    let v_total = (batch_size * kv_dim) as u32;
                    enc.set_buffer(v_buf, 0, 0);
                    enc.set_buffer(layer_buf, bv_off, 1);
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 2);
                    enc.set_bytes(&v_total.to_le_bytes(), 3);
                    let n_tg = (v_total as u64 + 255) / 256;
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(256, 1, 1));
                }
                // Barrier: bias must complete before RoPE reads q_buf/k_buf
                enc.memory_barrier_with_scope(1); // MTLBarrierScope.buffers
            }

            // Q RoPE dispatch (use rope_half_dim for partial rotation support)
            let total_q_elems = (batch_size * num_heads * rope_half_dim) as u32;
            let total_k_elems = (batch_size * num_kv_heads * rope_half_dim) as u32;
            let head_dim_u32 = head_dim as u32;
            let rope_half_dim_u32 = rope_half_dim as u32;
            let pos_start_u32 = seq_pos_start as u32;
            let total_dim_u32 = q_dim as u32;
            let total_kv_dim_u32 = kv_dim as u32;
            let num_heads_u32 = num_heads as u32;
            let num_kv_heads_u32_rope = num_kv_heads as u32;

            // Qwen3.5 uses NeoX-style (half-offset) dimension pairing
            let rope_b_pipe = if scratch.is_qwen35moe {
                pipelines.rope_batched_neox.as_ref().unwrap_or(&pipelines.rope_batched)
            } else {
                &pipelines.rope_batched
            };
            enc.set_pipeline_state(rope_b_pipe);
            enc.set_buffer(q_buf, 0, 0);
            enc.set_buffer(&scratch.rope_cos_buf, 0, 1);
            enc.set_buffer(&scratch.rope_sin_buf, 0, 2);
            enc.set_bytes(&num_heads_u32.to_le_bytes(), 3);
            enc.set_bytes(&head_dim_u32.to_le_bytes(), 4);
            enc.set_bytes(&rope_half_dim_u32.to_le_bytes(), 5);
            enc.set_bytes(&pos_start_u32.to_le_bytes(), 6);
            enc.set_bytes(&total_dim_u32.to_le_bytes(), 7);
            let tg_q = 256u64.min(total_q_elems as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((total_q_elems as u64).div_ceil(tg_q), 1, 1),
                MTLSize::new(tg_q, 1, 1),
            );

            // K RoPE dispatch (concurrent with Q RoPE -- different output buffer)
            enc.set_pipeline_state(rope_b_pipe);
            enc.set_buffer(k_buf, 0, 0);
            enc.set_bytes(&num_kv_heads_u32_rope.to_le_bytes(), 3);
            enc.set_bytes(&total_kv_dim_u32.to_le_bytes(), 7);
            let tg_k = 256u64.min(total_k_elems as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((total_k_elems as u64).div_ceil(tg_k), 1, 1),
                MTLSize::new(tg_k, 1, 1),
            );
            enc.end_encoding();
            } // end standard fused QKV path
        }

        // ---- 5+6. KV cache write + Batched causal attention (merged concurrent encoder) ----
        // KV cache writes and attention are merged into a single concurrent encoder.
        // K/V cache writes happen first, then a barrier ensures the cache is populated
        // before attention scores read it. Saves 1 encoder transition per layer.
        {
            let num_heads_u32 = num_heads as u32;
            let num_kv_heads_u32 = num_kv_heads as u32;
            let head_dim_u32 = head_dim as u32;
            let kv_dim_u32 = kv_dim as u32;
            let start_pos_u32 = seq_pos_start as u32;
            let batch_size_u32 = batch_size as u32;
            let max_attend_len = seq_pos_start + batch_size;
            let scores_stride_u32 = max_attend_len as u32;

            let enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create concurrent encoder".into())
            })?;

            // 5a. K cache write dispatch
            let kv_total_elems = (batch_size * kv_dim) as u64;
            let kv_tg = 256u64.min(kv_total_elems).max(1);
            enc.set_pipeline_state(&pipelines.kv_cache_write_batched);
            enc.set_buffer(k_buf, 0, 0);
            enc.set_buffer(&scratch.gpu_k_cache[layer_idx], 0, 1);
            enc.set_bytes(&kv_dim_u32.to_le_bytes(), 2);
            enc.set_bytes(&start_pos_u32.to_le_bytes(), 3);
            enc.set_bytes(&batch_size_u32.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(kv_total_elems.div_ceil(kv_tg), 1, 1),
                MTLSize::new(kv_tg, 1, 1),
            );

            // 5b. V cache write dispatch (transposed layout, concurrent with K)
            enc.set_pipeline_state(&pipelines.v_cache_write_batched);
            enc.set_buffer(v_buf, 0, 0);
            enc.set_buffer(&scratch.gpu_v_cache[layer_idx], 0, 1);
            enc.set_bytes(&kv_dim_u32.to_le_bytes(), 2);
            enc.set_bytes(&start_pos_u32.to_le_bytes(), 3);
            enc.set_bytes(&batch_size_u32.to_le_bytes(), 4);
            enc.set_bytes(&(max_seq_len as u32).to_le_bytes(), 5);
            enc.dispatch_threadgroups(
                MTLSize::new(kv_total_elems.div_ceil(kv_tg), 1, 1),
                MTLSize::new(kv_tg, 1, 1),
            );

            // Barrier: KV cache writes must complete before attention reads cache
            enc.memory_barrier_with_scope(1); // MTLBarrierScope.buffers

            // 6a. Attention scores (tiled GEMM): Q * K^T
            enc.set_pipeline_state(&pipelines.attention_scores_tiled);
            enc.set_buffer(q_buf, 0, 0);
            enc.set_buffer(&scratch.gpu_k_cache[layer_idx], 0, 1);
            enc.set_buffer(scores_buf, 0, 2);
            enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
            enc.set_bytes(&kv_dim_u32.to_le_bytes(), 4);
            enc.set_bytes(&num_heads_u32.to_le_bytes(), 5);
            enc.set_bytes(&num_kv_heads_u32.to_le_bytes(), 6);
            enc.set_bytes(&attn_scale.to_le_bytes(), 7);
            enc.set_bytes(&start_pos_u32.to_le_bytes(), 8);
            enc.set_bytes(&scores_stride_u32.to_le_bytes(), 9);
            enc.set_bytes(&batch_size_u32.to_le_bytes(), 10);
            enc.dispatch_threadgroups(
                MTLSize::new(
                    (max_attend_len as u64).div_ceil(32),
                    (batch_size as u64).div_ceil(32),
                    num_heads as u64,
                ),
                MTLSize::new(128, 1, 1),
            );

            // Barrier: attention scores must complete before softmax reads scores_buf
            enc.memory_barrier_with_scope(1); // MTLBarrierScope.buffers

            // 6b. Softmax over scores
            enc.set_pipeline_state(&pipelines.softmax_batched);
            enc.set_buffer(scores_buf, 0, 0);
            enc.set_bytes(&scores_stride_u32.to_le_bytes(), 1);
            enc.set_bytes(&num_heads_u32.to_le_bytes(), 2);
            enc.set_bytes(&start_pos_u32.to_le_bytes(), 3);
            enc.set_bytes(&batch_size_u32.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(num_heads as u64, batch_size as u64, 1),
                MTLSize::new(256u64.min(max_attend_len as u64).max(1), 1, 1),
            );

            // Barrier: softmax must complete before attention output reads scores_buf
            enc.memory_barrier_with_scope(1); // MTLBarrierScope.buffers

            // 6c. Attention output (tiled GEMM): Scores * V
            // Dispatch: (ceil(head_dim/32), ceil(batch_size/32), num_heads)
            enc.set_pipeline_state(&pipelines.attention_output_tiled);
            enc.set_buffer(scores_buf, 0, 0);
            enc.set_buffer(&scratch.gpu_v_cache[layer_idx], 0, 1);
            enc.set_buffer(attn_out_buf, 0, 2);
            enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
            enc.set_bytes(&kv_dim_u32.to_le_bytes(), 4);
            enc.set_bytes(&num_heads_u32.to_le_bytes(), 5);
            enc.set_bytes(&num_kv_heads_u32.to_le_bytes(), 6);
            enc.set_bytes(&start_pos_u32.to_le_bytes(), 7);
            enc.set_bytes(&scores_stride_u32.to_le_bytes(), 8);
            enc.set_bytes(&batch_size_u32.to_le_bytes(), 9);
            enc.set_bytes(&(max_seq_len as u32).to_le_bytes(), 10);
            enc.dispatch_threadgroups(
                MTLSize::new(
                    (head_dim as u64).div_ceil(32),
                    (batch_size as u64).div_ceil(32),
                    num_heads as u64,
                ),
                MTLSize::new(128, 1, 1),
            );

            enc.end_encoding();
        }

        // ---- 6d. Sigmoid gate (Q+gate fusion only) ----
        // Apply sigmoid(gate) * attn_out before Wo projection (Qwen3.5 full-attention).
        if has_qgate_fusion {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for sigmoid gate".into())
            })?;
            let pso = pipelines.sigmoid_mul_fused.as_ref().ok_or_else(|| {
                RuntimeError::Compute("sigmoid_mul_fused pipeline not compiled".into())
            })?;
            enc.set_pipeline_state(pso);
            enc.set_buffer(gate_buf, 0, 0);       // gate [batch * q_dim]
            enc.set_buffer(attn_out_buf, 0, 1);    // attn output [batch * q_dim]
            enc.set_buffer(attn_out_buf, 0, 2);    // output (in-place)
            let total_gate_elems = (batch_size * q_dim) as u32;
            enc.set_bytes(&total_gate_elems.to_le_bytes(), 3);
            let tg = 256u64.min(total_gate_elems as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((total_gate_elems as u64).div_ceil(tg), 1, 1),
                MTLSize::new(tg, 1, 1),
            );
            enc.end_encoding();
        }

        // ---- 7+9. Wo projection + Residual 1 + FFN RMSNorm (merged encoder) ----
        // Wo+residual writes attn_proj_buf, FFN RMSNorm reads attn_proj_buf.
        // Serial encoder guarantees Wo+residual completes before RMSNorm starts.
        // Saves 1 encoder transition per layer vs. separate encoders.
        {
            let m_u32 = batch_size as u32;
            let n_u32 = hidden_dim as u32;
            let k_u32 = q_dim as u32;
            let splitk = match st.wo.quant {
                QuantScheme::Q8_0 => Self::splitk_splits(batch_size, hidden_dim, q_dim, batch_size),
                QuantScheme::Q4_0 => Self::splitk_splits(batch_size, hidden_dim, q_dim, batch_size),
                _ => 0,
            };
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            if splitk > 0 {
                // Split-K path: GEMM to attn_proj_buf, then separate residual add
                Self::encode_splitk_gemm_for_quant(
                    &enc, pipelines, st.wo.quant, layer_buf, wo_off, attn_out_buf,
                    attn_proj_buf, splitk_partial_buf, m_u32, n_u32, k_u32, splitk,
                );
                enc.end_encoding();
                // add x_buf into attn_proj_buf, then FFN RMSNorm (same encoder)
                let enc2 = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
                let total_elems = (batch_size * hidden_dim) as u32;
                enc2.set_pipeline_state(&pipelines.add_residual_batched);
                enc2.set_buffer(attn_proj_buf, 0, 0);
                enc2.set_buffer(x_buf, 0, 1);
                enc2.set_bytes(&total_elems.to_le_bytes(), 2);
                let tg = 256u64.min(total_elems as u64).max(1);
                enc2.dispatch_threadgroups(
                    MTLSize::new((total_elems as u64).div_ceil(tg), 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
                // FFN RMSNorm (serial ordering: residual add completes first)
                let dim_u32 = hidden_dim as u32;
                enc2.set_pipeline_state(&pipelines.rmsnorm_batched_bytes);
                enc2.set_buffer(attn_proj_buf, 0, 0);
                enc2.set_buffer(layer_buf, ffn_norm_off, 1);
                enc2.set_buffer(normed_buf, 0, 2);
                enc2.set_bytes(&dim_u32.to_le_bytes(), 3);
                enc2.set_bytes(&eps.to_le_bytes(), 4);
                enc2.dispatch_threadgroups(
                    MTLSize::new(batch_size as u64, 1, 1),
                    MTLSize::new(norm_tg_size, 1, 1),
                );
                enc2.end_encoding();
            } else {
                // Non-split-K: fused GEMM+residual then FFN RMSNorm (same serial encoder)
                let wo_aligned = batch_size % 32 == 0 && hidden_dim % 32 == 0 && q_dim % 32 == 0;
                match st.wo.quant {
                    QuantScheme::Q8_0 if q_dim % 64 == 0 && batch_size <= 4096 => {
                        if wo_aligned && q_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::Q8_0 => {
                        if wo_aligned {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_residual_batched_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_residual_batched);
                        }
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::Q4_0 if q_dim % 64 == 0 && batch_size <= 4096 => {
                        if wo_aligned && q_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::Q4_0 => {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_residual_batched);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::F16 if q_dim % 64 == 0 && batch_size <= 4096 => {
                        if wo_aligned && q_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64_residual_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64_residual);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::F16 => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_f16_residual);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    _ => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32_residual);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                }
                enc.set_buffer(layer_buf, wo_off, 0);
                enc.set_buffer(attn_out_buf, 0, 1);
                enc.set_buffer(attn_proj_buf, 0, 2);
                enc.set_bytes(&m_u32.to_le_bytes(), 3);
                enc.set_bytes(&n_u32.to_le_bytes(), 4);
                enc.set_bytes(&k_u32.to_le_bytes(), 5);
                enc.set_buffer(x_buf, 0, 6);
                enc.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                    MTLSize::new(128, 1, 1),
                );
                // FFN RMSNorm (serial ordering: GEMM+residual completes first)
                let dim_u32 = hidden_dim as u32;
                enc.set_pipeline_state(&pipelines.rmsnorm_batched_bytes);
                enc.set_buffer(attn_proj_buf, 0, 0);
                enc.set_buffer(layer_buf, ffn_norm_off, 1);
                enc.set_buffer(normed_buf, 0, 2);
                enc.set_bytes(&dim_u32.to_le_bytes(), 3);
                enc.set_bytes(&eps.to_le_bytes(), 4);
                enc.dispatch_threadgroups(
                    MTLSize::new(batch_size as u64, 1, 1),
                    MTLSize::new(norm_tg_size, 1, 1),
                );
                enc.end_encoding();
            }
        }

        } else {
            // ---- GDN prefill: sequential per-token processing ----
            // Process each prompt token through the GDN layer one at a time,
            // using the single-token decode path. This builds up the recurrent
            // GDN h_state and conv_state correctly across all prompt tokens.
            //
            // For each token t:
            //   1. Copy batch_x_buf[t] -> s.x_buf (single-token hidden state)
            //   2. Run full GDN decode path (conv1d + gates + state update + output)
            //   3. Copy s.x_buf -> batch_x_buf[t] (write back updated hidden state)
            {
                // Build a CachedLayerMeta for this layer from SubtensorOffsets.
                let gdn_meta = CachedLayerMeta {
                    attn_norm_off: base_off + st.attn_norm.offset,
                    wq_off: base_off + st.wq.offset,
                    wo_off: base_off + st.wo.offset,
                    ffn_norm_off,
                    w_gate_off: base_off + st.w_gate.offset,
                    w_up_off: base_off + st.w_up.offset,
                    w_down_off: base_off + st.w_down.offset,
                    wq_quant: st.wq.quant,
                    wo_quant: st.wo.quant,
                    w_gate_quant: st.w_gate.quant,
                    w_up_quant: st.w_up.quant,
                    w_down_quant: st.w_down.quant,
                    bq_off: st.bq.map(|b| base_off + b.offset),
                    bk_off: st.bk.map(|b| base_off + b.offset),
                    bv_off: st.bv.map(|b| base_off + b.offset),
                    moe_meta: None,
                    shared_expert_gate_off: st.shared_expert_gate.map(|s| base_off + s.offset),
                    shared_expert_up_off: st.shared_expert_up.map(|s| base_off + s.offset),
                    shared_expert_down_off: st.shared_expert_down.map(|s| base_off + s.offset),
                    shared_expert_gate_quant: st.shared_expert_gate.map(|s| s.quant),
                    shared_expert_down_quant: st.shared_expert_down.map(|s| s.quant),
                    attn_gate_off: st.attn_gate.map(|s| base_off + s.offset),
                    attn_gate_quant: st.attn_gate.map(|s| s.quant),
                    attn_post_norm_off: st.attn_post_norm.map(|s| base_off + s.offset),
                    has_qgate_fusion: false, // GDN prefill path: no Q+gate fusion
                    wk_off: None,
                    wv_off: None,
                    wk_quant: None,
                    wv_quant: None,
                    attn_q_norm_off: None,
                    attn_k_norm_off: None,
                    ffn_gate_inp_shexp_off: st.ffn_gate_inp_shexp.map(|s| base_off + s.offset),
                    layer_type: st.layer_type,
                    ssm_a_off: st.ssm_a.map(|s| base_off + s.offset),
                    ssm_conv1d_off: st.ssm_conv1d.map(|s| base_off + s.offset),
                    ssm_dt_off: st.ssm_dt.map(|s| base_off + s.offset),
                    ssm_beta_off: st.ssm_beta.map(|s| base_off + s.offset),
                    ssm_alpha_off: st.ssm_alpha.map(|s| base_off + s.offset),
                    ssm_norm_off: st.ssm_norm.map(|s| base_off + s.offset),
                    ssm_out_off: st.ssm_out.map(|s| base_off + s.offset),
                    ssm_out_quant: st.ssm_out.map(|s| s.quant),
                    gdn_layer_idx: if st.layer_type == Some(1) {
                        // Count GDN layers (layer_type=1) with index < layer_idx.
                        // GDN layers are every layer EXCEPT every 4th (full attention).
                        // layer_type=1 for GDN, layer_type=0 for full attention.
                        // gdn_idx is the sequential count of GDN layers before this one.
                        let mut gdn_count = 0usize;
                        // Use gdn_layer_idx_map if populated (GPU-resident path),
                        // otherwise compute from layer_idx assuming every 4th layer
                        // is full attention (Qwen3.5 pattern: full_attention_interval=4).
                        if let Some(Some(idx)) = scratch.gdn_layer_idx_map.get(layer_idx) {
                            Some(*idx)
                        } else {
                            // Count non-full-attention layers before this one.
                            // Full attention layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39
                            // (every 4th starting from 3, i.e. (layer + 1) % 4 == 0)
                            for l in 0..layer_idx {
                                if (l + 1) % 4 != 0 {
                                    gdn_count += 1;
                                }
                            }
                            Some(gdn_count)
                        }
                    } else {
                        None
                    },
                };

                let gdn_idx = gdn_meta.gdn_layer_idx.ok_or_else(|| {
                    RuntimeError::Compute(format!(
                        "GDN prefill: layer {} has layer_type=1 but no gdn_layer_idx", layer_idx
                    ))
                })?;

                // Batched GDN prefill: batches GEMM operations while processing
                // state updates sequentially per token via fused kernel.
                let use_batched = true;

                if use_batched {
                    // Batched path: ~15 dispatches + per-token conv1d/state-update
                    // instead of 128*(15+2) per-token dispatches.
                    // FFN RMSNorm is fused into the same encoder (saves 1 encoder boundary).
                    let new_conv_pos = Self::encode_batched_gdn_prefill(
                        cmd, pipelines, scratch, layer_buf, &gdn_meta, gdn_idx,
                        x_buf, normed_buf, qkv_buf, attn_out_buf, gate_buf, attn_proj_buf,
                        batch_size,
                    )?;
                    scratch.gdn_conv_positions[gdn_idx] = new_conv_pos;
                } else {
                    // Fallback: token-by-token processing using single-token decode path.
                    let tok_bytes = (hidden_dim * 4) as u64;

                    for t in 0..batch_size {
                        let src_off = (t as u64) * tok_bytes;
                        // 1. Copy batch_x_buf[t] -> s.x_buf
                        {
                            let blit = cmd.new_blit_encoder().ok_or_else(|| {
                                RuntimeError::Compute("GDN prefill: blit copy in".into())
                            })?;
                            blit.copy_from_buffer(x_buf, src_off, &scratch.x_buf, 0, tok_bytes);
                            blit.end_encoding();
                        }
                        // 2. Run GDN decode path for this single token
                        let new_conv_pos = Self::encode_gdn_layer_decode(
                            cmd, pipelines, scratch, layer_buf, &gdn_meta, gdn_idx,
                        )?;
                        scratch.gdn_conv_positions[gdn_idx] = new_conv_pos;
                        // 3. Copy s.x_buf -> batch_x_buf[t]
                        {
                            let blit = cmd.new_blit_encoder().ok_or_else(|| {
                                RuntimeError::Compute("GDN prefill: blit copy out".into())
                            })?;
                            blit.copy_from_buffer(&scratch.x_buf, 0, x_buf, src_off, tok_bytes);
                            blit.end_encoding();
                        }
                    }
                    // After token-by-token GDN: copy batch_x_buf -> attn_proj_buf for FFN norm
                    {
                        let blit = cmd.new_blit_encoder().ok_or_else(|| {
                            RuntimeError::Compute("GDN prefill: blit copy attn_proj".into())
                        })?;
                        blit.copy_from_buffer(x_buf, 0, attn_proj_buf, 0, (batch_size * hidden_dim * 4) as u64);
                        blit.end_encoding();
                    }
                    // FFN RMSNorm (fallback path only -- batched path fuses this into the encoder)
                    {
                        let enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("GDN prefill: FFN RMSNorm encoder".into())
                        })?;
                        let dim_u32 = hidden_dim as u32;
                        enc.set_pipeline_state(&pipelines.rmsnorm_batched_bytes);
                        enc.set_buffer(attn_proj_buf, 0, 0);
                        enc.set_buffer(layer_buf, ffn_norm_off, 1);
                        enc.set_buffer(normed_buf, 0, 2);
                        enc.set_bytes(&dim_u32.to_le_bytes(), 3);
                        enc.set_bytes(&eps.to_le_bytes(), 4);
                        enc.dispatch_threadgroups(
                            MTLSize::new(batch_size as u64, 1, 1),
                            MTLSize::new(norm_tg_size, 1, 1),
                        );
                        enc.end_encoding();
                    }
                }
            }
        }

        // ---- FFN block: MoE vs dense path for batched prefill ----
        // If this layer has MoE experts, dispatch the MoE FFN (router + experts + accum)
        // and skip the dense FFN path below.
        let moe_handled = if scratch.moe_num_experts > 0 {
            if let (Some(ref router), Some(ref experts)) = (&st.router_weight, &st.experts) {
                let gate_offs: Vec<u64> = experts.iter().map(|e| base_off + e.gate.offset).collect();
                let up_offs: Vec<u64> = experts.iter().map(|e| base_off + e.up.offset).collect();
                let down_offs: Vec<u64> = experts.iter().map(|e| base_off + e.down.offset).collect();
                let first = &experts[0];
                Self::encode_moe_ffn_batched(
                    &cmd, pipelines, scratch, layer_buf, batch_size,
                    base_off + router.offset,
                    router.quant,
                    &gate_offs,
                    &up_offs,
                    &down_offs,
                    first.gate.quant,
                    first.down.quant,
                )?;

                // ---- Shared expert FFN (batched prefill) ----
                // In llama.cpp's build_layer_ffn, every MoE layer adds a shared expert
                // contribution: cur = moe_out + sigmoid(gate_inp_shexp @ cur) * FFN(cur)
                // where FFN = down_shexp(SwiGLU(gate_shexp(cur), up_shexp(cur)))
                if let (Some(se_gate), Some(se_up), Some(se_down)) = (
                    st.shared_expert_gate,
                    st.shared_expert_up,
                    st.shared_expert_down,
                ) {
                    let se_inter = scratch.shared_expert_inter_dim;
                    let se_gate_off = base_off + se_gate.offset;
                    let se_up_off = base_off + se_up.offset;
                    let se_down_off = base_off + se_down.offset;
                    let se_gate_quant = se_gate.quant;
                    let se_down_quant = se_down.quant;

                    const TILE_M_SE: u64 = 32;
                    const TILE_N_SE: u64 = 32;

                    // Step SE-1: Gate+Up+SwiGLU (batched tiled GEMM)
                    // gate_buf = gate_shexp @ normed_buf  [batch, se_inter]
                    // up_buf   = up_shexp   @ normed_buf  [batch, se_inter]
                    // gate_buf = silu(gate_buf) * up_buf
                    {
                        let enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for shared expert batched gate+up".into())
                        })?;

                        let m_u32 = batch_size as u32;
                        let n_u32 = se_inter as u32;
                        let k_u32 = hidden_dim as u32;

                        // Gate dispatch
                        match se_gate_quant {
                            QuantScheme::Q8_0 => {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                            QuantScheme::Q4_0 => {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                            QuantScheme::Q4_1 => {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_1);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                            QuantScheme::F16 => {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                            _ => {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                        }
                        enc.set_buffer(layer_buf, se_gate_off, 0);
                        enc.set_buffer(normed_buf, 0, 1);
                        enc.set_buffer(gate_buf, 0, 2);
                        enc.set_bytes(&m_u32.to_le_bytes(), 3);
                        enc.set_bytes(&n_u32.to_le_bytes(), 4);
                        enc.set_bytes(&k_u32.to_le_bytes(), 5);
                        enc.dispatch_threadgroups(
                            MTLSize::new((se_inter as u64).div_ceil(TILE_N_SE), (batch_size as u64).div_ceil(TILE_M_SE), 1),
                            MTLSize::new(128, 1, 1),
                        );

                        // Up dispatch (same encoder, sequential)
                        match se_gate_quant {
                            QuantScheme::Q8_0 => {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                            QuantScheme::Q4_0 => {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                            QuantScheme::Q4_1 => {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_1);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                            QuantScheme::F16 => {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                            _ => {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                        }
                        enc.set_buffer(layer_buf, se_up_off, 0);
                        enc.set_buffer(normed_buf, 0, 1);
                        enc.set_buffer(up_buf, 0, 2);
                        enc.set_bytes(&m_u32.to_le_bytes(), 3);
                        enc.set_bytes(&n_u32.to_le_bytes(), 4);
                        enc.set_bytes(&k_u32.to_le_bytes(), 5);
                        enc.dispatch_threadgroups(
                            MTLSize::new((se_inter as u64).div_ceil(TILE_N_SE), (batch_size as u64).div_ceil(TILE_M_SE), 1),
                            MTLSize::new(128, 1, 1),
                        );
                        enc.end_encoding();
                    }

                    // SwiGLU: gate_buf = silu(gate_buf) * up_buf
                    {
                        let enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for shared expert batched swiglu".into())
                        })?;
                        let total_elems = (batch_size * se_inter) as u32;
                        enc.set_pipeline_state(&pipelines.swiglu_batched);
                        enc.set_buffer(gate_buf, 0, 0);
                        enc.set_buffer(up_buf, 0, 1);
                        enc.set_bytes(&total_elems.to_le_bytes(), 2);
                        let tg_swiglu = 256u64.min(total_elems as u64).max(1);
                        enc.dispatch_threadgroups(
                            MTLSize::new((total_elems as u64).div_ceil(tg_swiglu), 1, 1),
                            MTLSize::new(tg_swiglu, 1, 1),
                        );
                        enc.end_encoding();
                    }

                    // Step SE-2: Down projection -> attn_proj_buf [batch, hidden_dim]
                    // (attn_proj_buf is free after MoE accum consumed it as residual)
                    {
                        let enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for shared expert batched down".into())
                        })?;
                        let m_u32 = batch_size as u32;
                        let n_u32 = hidden_dim as u32;
                        let k_u32 = se_inter as u32;
                        match se_down_quant {
                            QuantScheme::Q8_0 => {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                            QuantScheme::Q4_0 => {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                            QuantScheme::Q4_1 => {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_1);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                            QuantScheme::F16 => {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                            _ => {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                                enc.set_threadgroup_memory_length(4096, 0);
                            }
                        }
                        enc.set_buffer(layer_buf, se_down_off, 0);
                        enc.set_buffer(gate_buf, 0, 1);
                        enc.set_buffer(attn_proj_buf, 0, 2);
                        enc.set_bytes(&m_u32.to_le_bytes(), 3);
                        enc.set_bytes(&n_u32.to_le_bytes(), 4);
                        enc.set_bytes(&k_u32.to_le_bytes(), 5);
                        enc.dispatch_threadgroups(
                            MTLSize::new((hidden_dim as u64).div_ceil(TILE_N_SE), (batch_size as u64).div_ceil(TILE_M_SE), 1),
                            MTLSize::new(128, 1, 1),
                        );
                        enc.end_encoding();
                    }

                    // Step SE-3: Gate scalar + sigmoid-scale-add
                    // gate_scalar[b] = dot(ffn_gate_inp_shexp, normed_buf[b]) for each token b
                    // x_buf[b] += sigmoid(gate_scalar[b]) * attn_proj_buf[b]
                    if let Some(gis) = st.ffn_gate_inp_shexp {
                        let gis_off = base_off + gis.offset;
                        // Matmul: [batch, hidden_dim] @ [hidden_dim, 1] -> [batch, 1]
                        // Using tiled matmul with N=1: each token produces 1 scalar
                        {
                            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                                RuntimeError::Compute("Failed to create encoder for shared expert gate_inp batched".into())
                            })?;
                            // Use F32 matmul for gate_inp_shexp (it's always F32)
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                            enc.set_threadgroup_memory_length(4096, 0);
                            enc.set_buffer(layer_buf, gis_off, 0);   // weight [1, hidden_dim]
                            enc.set_buffer(normed_buf, 0, 1);        // input [batch, hidden_dim]
                            enc.set_buffer(up_buf, 0, 2);            // output [batch, 1] (reuse up_buf)
                            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 3); // M = batch
                            enc.set_bytes(&(1u32).to_le_bytes(), 4);              // N = 1
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5); // K = hidden_dim
                            enc.dispatch_threadgroups(
                                MTLSize::new(1, (batch_size as u64).div_ceil(TILE_M_SE), 1),
                                MTLSize::new(128, 1, 1),
                            );
                            enc.end_encoding();
                        }

                        // Sigmoid-scale-add: x_buf += sigmoid(up_buf[b]) * attn_proj_buf
                        {
                            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                                RuntimeError::Compute("Failed to create encoder for shared expert sigmoid_scale_add".into())
                            })?;
                            let pso = pipelines.sigmoid_scale_add_batched.as_ref().ok_or_else(|| {
                                RuntimeError::Compute("sigmoid_scale_add_batched pipeline not compiled".into())
                            })?;
                            enc.set_pipeline_state(pso);
                            enc.set_buffer(up_buf, 0, 0);          // gate_scalars [batch]
                            enc.set_buffer(attn_proj_buf, 0, 1);   // src [batch * hidden_dim]
                            enc.set_buffer(x_buf, 0, 2);           // dst [batch * hidden_dim]
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 4);
                            let total = (batch_size * hidden_dim) as u64;
                            let tg = 256u64.min(total).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new(total.div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                            enc.end_encoding();
                        }
                    } else {
                        // No gate_inp_shexp weight -- just add shared expert output directly
                        let enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for shared expert add".into())
                        })?;
                        enc.set_pipeline_state(&pipelines.add_residual_batched);
                        enc.set_buffer(x_buf, 0, 0);
                        enc.set_buffer(attn_proj_buf, 0, 1);
                        enc.set_bytes(&((batch_size * hidden_dim) as u32).to_le_bytes(), 2);
                        let total = (batch_size * hidden_dim) as u64;
                        let tg = 256u64.min(total).max(1);
                        enc.dispatch_threadgroups(
                            MTLSize::new(total.div_ceil(tg), 1, 1),
                            MTLSize::new(tg, 1, 1),
                        );
                        enc.end_encoding();
                    }
                }

                true
            } else {
                return Err(RuntimeError::Compute(
                    "Model has num_experts > 0 but layer is missing router_weight/experts \
                     in SubtensorOffsets. The LBC file may need re-conversion.".into()
                ));
            }
        } else {
            false
        };

        // ---- Dense FFN path (steps 10-12): only executed for non-MoE layers ----
        if !moe_handled {

        // ---- 10. Gate + Up projections (merged into single encoder) ----
        {
            let m_u32 = batch_size as u32;
            let n_u32 = inter_dim as u32;
            let k_u32 = hidden_dim as u32;
            let splitk_gate = match st.w_gate.quant {
                QuantScheme::Q8_0 => Self::splitk_splits(batch_size, inter_dim, hidden_dim, batch_size),
                QuantScheme::Q4_0 => Self::splitk_splits(batch_size, inter_dim, hidden_dim, batch_size),
                _ => 0,
            };
            let splitk_up = match st.w_up.quant {
                QuantScheme::Q8_0 => Self::splitk_splits(batch_size, inter_dim, hidden_dim, batch_size),
                QuantScheme::Q4_0 => Self::splitk_splits(batch_size, inter_dim, hidden_dim, batch_size),
                _ => 0,
            };

            // If either needs split-K, use separate encoders for that one
            if splitk_gate > 0 || splitk_up > 0 {
                // Gate
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
                if splitk_gate > 0 {
                    Self::encode_splitk_gemm_for_quant(
                        &enc, pipelines, st.w_gate.quant, layer_buf, w_gate_off, normed_buf,
                        gate_buf, splitk_partial_buf, m_u32, n_u32, k_u32, splitk_gate,
                    );
                } else {
                    let ffn_aligned = batch_size % 32 == 0 && inter_dim % 32 == 0 && hidden_dim % 32 == 0;
                    match st.w_gate.quant {
                        QuantScheme::Q8_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if ffn_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::Q8_0 => {
                            if ffn_aligned {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0);
                            }
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Q4_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if ffn_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::Q4_0 => {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Q4_1 => {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_1);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::F16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if ffn_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::F16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        _ => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                    }
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(normed_buf, 0, 1);
                    enc.set_buffer(gate_buf, 0, 2);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&k_u32.to_le_bytes(), 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
                enc.end_encoding();
                // Up
                let enc2 = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
                if splitk_up > 0 {
                    Self::encode_splitk_gemm_for_quant(
                        &enc2, pipelines, st.w_up.quant, layer_buf, w_up_off, normed_buf,
                        up_buf, splitk_partial_buf, m_u32, n_u32, k_u32, splitk_up,
                    );
                } else {
                    let ffn_aligned = batch_size % 32 == 0 && inter_dim % 32 == 0 && hidden_dim % 32 == 0;
                    match st.w_up.quant {
                        QuantScheme::Q8_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if ffn_aligned && hidden_dim % 64 == 0 {
                                enc2.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                            } else {
                                enc2.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                            }
                            enc2.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::Q8_0 => {
                            if ffn_aligned {
                                enc2.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_aligned);
                            } else {
                                enc2.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0);
                            }
                            enc2.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Q4_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if ffn_aligned && hidden_dim % 64 == 0 {
                                enc2.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_aligned);
                            } else {
                                enc2.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64);
                            }
                            enc2.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::Q4_0 => {
                            enc2.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                            enc2.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Q4_1 => {
                            enc2.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_1);
                            enc2.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::F16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if ffn_aligned && hidden_dim % 64 == 0 {
                                enc2.set_pipeline_state(&pipelines.tiled_matmul_f16_k64_aligned);
                            } else {
                                enc2.set_pipeline_state(&pipelines.tiled_matmul_f16_k64);
                            }
                            enc2.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::F16 => {
                            enc2.set_pipeline_state(&pipelines.tiled_matmul_f16);
                            enc2.set_threadgroup_memory_length(4096, 0);
                        }
                        _ => {
                            enc2.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                            enc2.set_threadgroup_memory_length(4096, 0);
                        }
                    }
                    enc2.set_buffer(layer_buf, w_up_off, 0);
                    enc2.set_buffer(normed_buf, 0, 1);
                    enc2.set_buffer(up_buf, 0, 2);
                    enc2.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc2.set_bytes(&n_u32.to_le_bytes(), 4);
                    enc2.set_bytes(&k_u32.to_le_bytes(), 5);
                    enc2.dispatch_threadgroups(
                        MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
                enc2.end_encoding();
            } else {
                // No split-K: merge Gate + Up + SwiGLU into single concurrent encoder.
                // Gate and Up read from normed_buf (same input) and write to separate
                // output buffers (gate_buf, up_buf). They can run concurrently.
                // SwiGLU reads both gate_buf and up_buf, so it must wait for both via barrier.
                let enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create concurrent encoder".into())
                })?;

                // Gate dispatch
                let ffn_aligned = batch_size % 32 == 0 && inter_dim % 32 == 0 && hidden_dim % 32 == 0;
                match st.w_gate.quant {
                    QuantScheme::Q8_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                        if ffn_aligned && hidden_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::Q8_0 => {
                        if ffn_aligned {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0);
                        }
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::Q4_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                        if ffn_aligned && hidden_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::Q4_0 => {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::Q4_1 => {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_1);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::F16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                        if ffn_aligned && hidden_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::F16 => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    _ => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                }
                enc.set_buffer(layer_buf, w_gate_off, 0);
                enc.set_buffer(normed_buf, 0, 1);
                enc.set_buffer(gate_buf, 0, 2);
                enc.set_bytes(&m_u32.to_le_bytes(), 3);
                enc.set_bytes(&n_u32.to_le_bytes(), 4);
                enc.set_bytes(&k_u32.to_le_bytes(), 5);
                enc.dispatch_threadgroups(
                    MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                    MTLSize::new(128, 1, 1),
                );

                // Up dispatch (concurrent with Gate -- different output buffers)
                match st.w_up.quant {
                    QuantScheme::Q8_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                        if ffn_aligned && hidden_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::Q8_0 => {
                        if ffn_aligned {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0);
                        }
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::Q4_0 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                        if ffn_aligned && hidden_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::Q4_0 => {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::Q4_1 => {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_1);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::F16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                        if ffn_aligned && hidden_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::F16 => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    _ => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                }
                enc.set_buffer(layer_buf, w_up_off, 0);
                // normed_buf already set at index 1
                enc.set_buffer(up_buf, 0, 2);
                // M, N, K already set at indices 3, 4, 5
                enc.dispatch_threadgroups(
                    MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                    MTLSize::new(128, 1, 1),
                );

                // Barrier: Gate and Up must complete before SwiGLU reads gate_buf + up_buf
                enc.memory_barrier_with_scope(1); // MTLBarrierScope.buffers

                // SwiGLU dispatch: gate = silu(gate) * up
                let total_elems = (batch_size * inter_dim) as u32;
                enc.set_pipeline_state(&pipelines.swiglu_batched);
                enc.set_buffer(gate_buf, 0, 0);
                enc.set_buffer(up_buf, 0, 1);
                enc.set_bytes(&total_elems.to_le_bytes(), 2);
                let tg_swiglu = 256u64.min(total_elems as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((total_elems as u64).div_ceil(tg_swiglu), 1, 1),
                    MTLSize::new(tg_swiglu, 1, 1),
                );

                enc.end_encoding();
            }
        }

        // ---- 11. SwiGLU (only for split-K path; non-split-K path merges SwiGLU above) ----
        if {
            let splitk_gate = match st.w_gate.quant {
                QuantScheme::Q8_0 => Self::splitk_splits(batch_size, inter_dim, hidden_dim, batch_size),
                QuantScheme::Q4_0 => Self::splitk_splits(batch_size, inter_dim, hidden_dim, batch_size),
                _ => 0,
            };
            let splitk_up = match st.w_up.quant {
                QuantScheme::Q8_0 => Self::splitk_splits(batch_size, inter_dim, hidden_dim, batch_size),
                QuantScheme::Q4_0 => Self::splitk_splits(batch_size, inter_dim, hidden_dim, batch_size),
                _ => 0,
            };
            splitk_gate > 0 || splitk_up > 0
        } {
            let total_elems = (batch_size * inter_dim) as u32;
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            enc.set_pipeline_state(&pipelines.swiglu_batched);
            enc.set_buffer(gate_buf, 0, 0);
            enc.set_buffer(up_buf, 0, 1);
            enc.set_bytes(&total_elems.to_le_bytes(), 2);
            let tg = 256u64.min(total_elems as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((total_elems as u64).div_ceil(tg), 1, 1),
                MTLSize::new(tg, 1, 1),
            );
            enc.end_encoding();
        }

        // ---- 12. Down projection + Residual 2 (fused): x_buf = Down * gate + attn_proj_buf ----
        {
            let m_u32 = batch_size as u32;
            let n_u32 = hidden_dim as u32;
            let k_u32 = inter_dim as u32;
            let splitk = match st.w_down.quant {
                QuantScheme::Q8_0 => Self::splitk_splits(batch_size, hidden_dim, inter_dim, batch_size),
                QuantScheme::Q4_0 => Self::splitk_splits(batch_size, hidden_dim, inter_dim, batch_size),
                _ => 0,
            };
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            if splitk > 0 {
                // Split-K path: GEMM to down_buf, then separate residual add
                Self::encode_splitk_gemm_for_quant(
                    &enc, pipelines, st.w_down.quant, layer_buf, w_down_off, gate_buf,
                    down_buf, splitk_partial_buf, m_u32, n_u32, k_u32, splitk,
                );
                enc.end_encoding();
                // x_buf = attn_proj_buf + down_buf
                let enc2 = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
                let total_elems = (batch_size * hidden_dim) as u32;
                enc2.set_pipeline_state(&pipelines.add_write);
                enc2.set_buffer(attn_proj_buf, 0, 0);
                enc2.set_buffer(down_buf, 0, 1);
                enc2.set_buffer(x_buf, 0, 2);
                enc2.dispatch_threadgroups(
                    MTLSize::new((total_elems as u64).div_ceil(256), 1, 1),
                    MTLSize::new(256u64.min(total_elems as u64).max(1), 1, 1),
                );
                enc2.end_encoding();
            } else {
                // Non-split-K: use fused GEMM+residual kernel
                let down_aligned = batch_size % 32 == 0 && hidden_dim % 32 == 0 && inter_dim % 32 == 0;
                match st.w_down.quant {
                    QuantScheme::Q8_0 if inter_dim % 64 == 0 && batch_size <= 4096 => {
                        if down_aligned && inter_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::Q8_0 => {
                        if down_aligned {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_residual_batched_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_residual_batched);
                        }
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::Q4_0 if inter_dim % 64 == 0 && batch_size <= 4096 => {
                        if down_aligned && inter_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::Q4_0 => {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_residual_batched);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::Q4_1 => {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_1_residual_batched);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::F16 if inter_dim % 64 == 0 && batch_size <= 4096 => {
                        if down_aligned && inter_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64_residual_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16_k64_residual);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::F16 => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_f16_residual);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    _ => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32_residual);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                }
                enc.set_buffer(layer_buf, w_down_off, 0);
                enc.set_buffer(gate_buf, 0, 1);
                enc.set_buffer(x_buf, 0, 2);  // output to x_buf
                enc.set_bytes(&m_u32.to_le_bytes(), 3);
                enc.set_bytes(&n_u32.to_le_bytes(), 4);
                enc.set_bytes(&k_u32.to_le_bytes(), 5);
                enc.set_buffer(attn_proj_buf, 0, 6);  // residual from attn_proj_buf
                enc.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                    MTLSize::new(128, 1, 1),
                );
                enc.end_encoding();
            }
        }

        } // end if !moe_handled (dense FFN path)

        // No commit here -- caller owns the command buffer and commits
        // after all layers are encoded. This eliminates N-1 sync barriers.

        // ---- 14. Update KV cache position ----
        kv.seq_len = seq_pos_start + batch_size;

        Ok(())
    }

    /// Embed a batch of token ids into the batch x_buf on the GPU.
    /// Encode token embedding lookup into an existing command buffer.
    ///
    /// Previously this created its own command buffer + commit_and_wait(),
    /// adding ~11ms of GPU-CPU roundtrip overhead per prefill. Now the embed
    /// dispatch is the FIRST encoder in the prefill command buffer, and Metal's
    /// implicit encoder barriers guarantee the embed output lands in batch_x_buf
    /// before layer 0 reads it.
    fn encode_embed_batched(
        &self,
        cmd: &MetalCommandBuffer,
        token_ids: &[u32],
        pipelines: &MetalPipelines,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        let batch_size = token_ids.len();
        let hidden_dim = scratch.hidden_dim;

        let x_buf = scratch.batch_x_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_x_buf not allocated for embed".into())
        })?;

        // Upload token ids
        let ids_bytes: Vec<u8> = token_ids.iter()
            .flat_map(|id| id.to_le_bytes())
            .collect();
        let ids_buf = self.device.new_buffer_with_bytes(&ids_bytes).ok_or_else(|| {
            RuntimeError::Compute("Failed to create token ids buffer".into())
        })?;

        // Resolve embedding buffer: prefer unified private buffer, fall back to separate
        let (embed_buf_ref, embed_off): (&MetalBuffer, u64) =
            if let Some((emb_o, _, _)) = scratch.gpu_global_offsets {
                (scratch.gpu_unified_weight_buf.as_ref().unwrap(), emb_o as u64)
            } else {
                let eb = self.embedding_buf.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Embedding buffer not initialized".into())
                })?;
                (eb, 0u64)
            };

        let hidden_dim_u32 = hidden_dim as u32;

        let enc = cmd.new_compute_encoder().ok_or_else(|| {
            RuntimeError::Compute("Failed to create encoder for batched embed".into())
        })?;

        match self.embedding_quant {
            QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.embed_tokens_batched_q8_0),
            QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.embed_tokens_batched_q4_0),
            QuantScheme::F16 => enc.set_pipeline_state(&pipelines.embed_tokens_batched_f16),
            _ => enc.set_pipeline_state(&pipelines.embed_tokens_batched),
        }
        enc.set_buffer(embed_buf_ref, embed_off, 0);
        enc.set_buffer(x_buf, 0, 1);
        enc.set_buffer(&ids_buf, 0, 2);
        enc.set_bytes(&hidden_dim_u32.to_le_bytes(), 3);
        let batch_size_u32 = batch_size as u32;
        enc.set_bytes(&batch_size_u32.to_le_bytes(), 4);

        let total_elems = (batch_size * hidden_dim) as u64;
        let tg = 256u64.min(total_elems).max(1);
        let tg_count = total_elems.div_ceil(tg);
        enc.dispatch_threadgroups(
            MTLSize::new(tg_count, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
        enc.end_encoding();

        Ok(())
    }

    /// Read the last token's hidden state from the batch x_buf.
    ///
    /// Returns a Vec<f32> of length hidden_dim that can be used for
    /// final norm + output projection.
    fn read_last_hidden(
        &self,
        batch_size: usize,
        scratch: &MetalScratch,
    ) -> Result<Vec<f32>, RuntimeError> {
        let hidden_dim = scratch.hidden_dim;
        let x_buf = scratch.batch_x_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_x_buf not allocated for read".into())
        })?;

        // Read entire batch, extract last token
        let total = batch_size * hidden_dim;
        let mut all_data = vec![0.0f32; total];
        x_buf.read_f32(&mut all_data);

        let last_start = (batch_size - 1) * hidden_dim;
        Ok(all_data[last_start..last_start + hidden_dim].to_vec())
    }

    /// Run batched prefill: process all prompt tokens through all layers on the GPU.
    ///
    /// Optimized: ALL layers are encoded into a SINGLE Metal command buffer.
    /// Previous: 1 command buffer per layer = N sync barriers per prefill.
    /// Now: 1 command buffer for ALL N layers = 1 sync barrier for entire prefill.
    ///
    /// This eliminates 21 GPU-CPU round-trips for a 22-layer model. The GPU
    /// processes all layers back-to-back without waiting for CPU acknowledgment
    /// between layers. Metal encoder barriers (implicit between compute encoders
    /// in the same command buffer) ensure correct ordering.
    ///
    /// Memory safety: All LayerViews are collected into a Vec and kept alive
    /// until after commit_and_wait(). For mmap-backed weights, the underlying
    /// memory is the mmap region (outlives the function call). For Arc-backed
    /// weights, holding the LayerView keeps the Arc alive. The zero-copy Metal
    /// buffers (bytesNoCopy) in layer_buf_cache reference this same memory,
    /// so they remain valid for the duration of GPU execution.
    ///
    /// Returns the final hidden state of the LAST token.
    pub fn prefill(
        &self,
        prompt_tokens: &[u32],
        weights: &dyn crate::weight::cache::WeightProvider,
        kv: &mut crate::kv::KvCache,
    ) -> Result<Vec<f32>, RuntimeError> {
        let batch_size = prompt_tokens.len();
        if batch_size == 0 {
            return Err(RuntimeError::Compute("empty prompt".into()));
        }

        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Metal pipelines not initialized: call init() first".into())
        })?;

        // ================================================================
        // MUTEX-FREE ENCODING: Acquire the scratch lock ONCE for the
        // entire prefill operation. Previous code locked/unlocked ~26
        // times (1 ensure + 1 num_layers + 1 embed + 22 layers + 1 read
        // = 26 lock ops). Each Mutex::lock() has overhead on macOS, but
        // more importantly each lock boundary forces the compiler to
        // reload all scratch-derived references from memory (no cross-
        // lock aliasing). Holding a single guard lets the compiler keep
        // buffer pointers in registers across the entire encoding loop.
        // ================================================================
        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("Metal scratch not initialized".into())
        })?;

        // Ensure batch buffers are large enough
        self.ensure_batch_buffers(s, batch_size)?;

        let num_layers = s.num_layers;

        // ================================================================
        // SINGLE command buffer for embed + ALL layers.
        // Previous: separate CB for embed (11ms overhead) + CB for layers.
        // Now: embed is the FIRST encoder in the unified CB. Metal implicit
        // barriers guarantee embed output lands in batch_x_buf before layer 0.
        // ================================================================
        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for prefill".into())
        })?;

        // Embed all tokens as first encoder in the prefill command buffer
        self.encode_embed_batched(&cmd, prompt_tokens, pipelines, s)?;

        // Pre-load all layer views and keep them alive until after GPU commit.
        // This ensures the backing memory (mmap pointers or Arc<[u8]>) remains
        // valid while the GPU processes the command buffer.
        let mut layer_views = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            weights.begin_pass();
            let layer_view = match weights.try_get_layer(layer) {
                Some(view) => view,
                None => weights.get_layer_blocking(layer)?,
            };
            layer_views.push(layer_view);
        }

        // Encode all layers into the single command buffer.
        // Each layer adds ~13 compute encoders. Metal guarantees ordering
        // between encoders within the same command buffer (implicit barriers).
        // The scratch guard is held throughout, eliminating 22 lock/unlock
        // cycles in the inner loop.
        for layer in 0..num_layers {
            let mut kv_view = kv.view_mut(layer)?;

            self.encode_layer_batched(
                &cmd, layer, batch_size, &layer_views[layer],
                &mut kv_view, pipelines, s,
            )?;

            kv.commit_view(kv_view)?;
        }

        // Single sync point for the ENTIRE prefill (all layers).
        // We hold the mutex through GPU sync -- this is fine because
        // prefill is single-threaded and no other thread needs scratch
        // during this operation.
        cmd.commit_and_wait();

        // LayerViews are dropped here, after GPU has finished.
        // This is the key safety guarantee: backing memory was alive
        // throughout GPU execution.
        drop(layer_views);

        // Read the last token's hidden state (must be after commit_and_wait
        // since it reads GPU memory via batch_x_buf.read_f32).
        let last_hidden = self.read_last_hidden(batch_size, s)?;

        // Advance KV cache
        for _ in 0..batch_size {
            kv.advance_seq_len()?;
        }

        Ok(last_hidden)
    }

    /// Returns true if GPU-resident weights are loaded.
    pub fn is_gpu_resident(&self) -> bool {
        let scratch_guard = self.scratch.lock().unwrap();
        scratch_guard
            .as_ref()
            .map(|s| s.gpu_unified_weight_buf.is_some() || s.gpu_resident_layers.is_some())
            .unwrap_or(false)
    }

    /// Returns the number of GatedDeltaNet layers in the model.
    ///
    /// Zero for non-GDN models (TinyLlama, Qwen2, Mixtral, etc.).
    /// Non-zero for Qwen3.5-35B-A3B and similar hybrid SSM/attention models.
    /// Used to decide whether to use sequential vs. batched prefill.
    pub fn gdn_num_layers(&self) -> usize {
        let scratch_guard = self.scratch.lock().unwrap();
        scratch_guard
            .as_ref()
            .map(|s| s.gdn_num_layers)
            .unwrap_or(0)
    }


    /// Full-token decode in a SINGLE command buffer (GPU-resident only).
    ///
    /// Encodes embed + all layers + final norm + output projection into one
    /// Metal command buffer with a single commit_and_wait(). Eliminates N-1
    /// CB create/commit cycles and N-1 mutex lock/unlock pairs.
    ///
    /// Why this works: a previous attempt at single-CB used the STREAMING path
    /// where CPU loads weights via mmap between layers. GPU starved waiting for
    /// CPU to encode all layers (-20%). In GPU-RESIDENT mode, all weights are
    /// in Metal buffers -- CPU encodes in microseconds.
    pub fn decode_token_single_cb(
        &self,
        token_id: u32,
        _weights: &dyn crate::weight::cache::WeightProvider,
        kv: &mut crate::kv::KvCache,
    ) -> Result<Logits, RuntimeError> {
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Metal pipelines not initialized: call init() first".into())
        })?;
        let embedding_buf = self.embedding_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Embedding buffer not initialized".into())
        })?;
        let final_norm_buf = self.final_norm_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Final norm buffer not initialized".into())
        })?;
        let output_proj_buf = self.output_proj_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Output proj buffer not initialized".into())
        })?;
        let output_proj_quant = self.output_proj_quant;

        let seq_pos = kv.seq_len();

        // Single mutex acquisition for the entire token.
        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("Metal scratch not initialized".into())
        })?;
        if let Some(prev_cmd) = s.last_async_cmd.take() {
            prev_cmd.wait_until_completed();
        }
        // GPU-resident check: unified private buffer OR per-layer buffers
        let has_unified = s.gpu_unified_weight_buf.is_some();
        let has_per_layer = s.gpu_resident_layers.is_some();
        if !has_unified && !has_per_layer {
            return Err(RuntimeError::Compute(
                "decode_token_single_cb requires GPU-resident weights".into(),
            ));
        }

        let hidden_dim = s.hidden_dim;
        let num_heads = s.num_heads;
        let num_kv_heads = s.num_kv_heads;
        let num_layers = s.num_layers;
        let head_dim = s.head_dim;
        let inter_dim = s.inter_dim;
        let eps = s.eps;
        let q_dim = s.q_dim;
        let kv_dim = s.kv_dim;
        let qkv_dim = s.qkv_dim;
        let attn_scale = s.attn_scale;
        let matmul_tg_size = s.matmul_tg_size;
        let norm_tg_size = s.norm_tg_size;
        let vocab_size = s.vocab_size;

        // ONE command buffer for embed + ALL layers + final projection.
        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for single-CB decode".into())
        })?;
        // Resolve embedding buffer
        let (sc_embed_buf, sc_embed_off): (&MetalBuffer, u64) =
            if let Some((emb_o, _, _)) = s.gpu_global_offsets {
                (s.gpu_unified_weight_buf.as_ref().unwrap(), emb_o as u64)
            } else {
                (embedding_buf, 0u64)
            };

        // --- Embed token into x_buf ---
        // For pure dense models (no GDN, no MoE), use a serial encoder.
        // Dense decode is a strict dependency chain -- every dispatch reads the
        // previous dispatch's output. The concurrent encoder's overlap-tracking
        // metadata is pure overhead when no overlap is possible. Serial encoders
        // guarantee completion ordering: each dispatch finishes before the next
        // begins, making memory_barrier_with_scope calls unnecessary (skipped
        // for serial via the all_dense flag to reduce CPU-side encoding cost).
        // GDN/MoE models keep the concurrent encoder for overlap of independent
        // small dispatches.
        let all_dense = s.cached_layer_meta.iter().all(|m| {
            m.gdn_layer_idx.is_none() && m.moe_meta.is_none()
        });
        let needs_barriers = !all_dense;
        let mut enc = if all_dense {
            cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create serial encoder".into())
            })?
        } else {
            cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create concurrent encoder".into())
            })?
        };

        {
            match self.embedding_quant {
                QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.embed_token_q8_0),
                QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.embed_token_q4_0),
                QuantScheme::F16 => enc.set_pipeline_state(&pipelines.embed_token_f16),
                _ => enc.set_pipeline_state(&pipelines.embed_token),
            }
            enc.set_buffer(sc_embed_buf, sc_embed_off, 0);
            enc.set_buffer(&s.x_buf, 0, 1);
            enc.set_bytes(&token_id.to_le_bytes(), 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            let tg = 256u64.min(hidden_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                MTLSize::new(tg, 1, 1),
            );
            // Barrier: embed writes x_buf, layer 0 RMSNorm reads x_buf
            if needs_barriers { enc.memory_barrier_with_scope(1); }
        }

        // --- ALL layers ---

        for layer_idx in 0..num_layers {
            // Resolve layer buffer: prefer unified private buffer, then per-layer
            let layer_buf: &MetalBuffer;
            if let Some(ref ubuf) = s.gpu_unified_weight_buf {
                layer_buf = ubuf;
            } else {
                let gpu_layers = s.gpu_resident_layers.as_ref().unwrap();
                layer_buf = &gpu_layers[layer_idx];
            }
            // Use cached metadata (pre-computed absolute offsets + quant schemes).
            let meta = &s.cached_layer_meta[layer_idx];
            let attn_norm_off = meta.attn_norm_off;
            let wq_off = meta.wq_off;
            let wo_off = meta.wo_off;
            let ffn_norm_off = meta.ffn_norm_off;
            let w_gate_off = meta.w_gate_off;
            let w_up_off = meta.w_up_off;
            let w_down_off = meta.w_down_off;
            let new_seq_len = seq_pos + 1;
            let q_byte_off: u64 = 0;
            let k_byte_off: u64 = (q_dim * 4) as u64;
            let v_byte_off: u64 = ((q_dim + kv_dim) * 4) as u64;

            // Reuse the single concurrent encoder (no per-layer encoder creation).

            // ================================================================
            // ATTENTION BLOCK
            // ================================================================
            if meta.gdn_layer_idx.is_none() {
                // Standard softmax attention path

                // Fused RMSNorm + QKV Q8_0 matvec.
                // Eliminates 1 dispatch + 1 barrier + normed_buf write/read per layer.
                // Also works for Q+gate fusion: all 3 matmuls (Q+gate, K, V) fuse
                // RMSNorm inline, reading x_buf directly. Eliminates separate RMSNorm
                // dispatch + barrier, and allows K/V to dispatch in parallel with Q+gate.
                let use_fused_attn_norm = matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
                    && !(meta.bq_off.is_some() && meta.bk_off.is_some() && meta.bv_off.is_some())
                    && (!meta.has_qgate_fusion
                        || (matches!(meta.wk_quant, Some(QuantScheme::Q8_0) | Some(QuantScheme::Q4_0) | Some(QuantScheme::F16))
                            && matches!(meta.wv_quant, Some(QuantScheme::Q8_0) | Some(QuantScheme::Q4_0) | Some(QuantScheme::F16))));

                if use_fused_attn_norm && !meta.has_qgate_fusion {
                    // Fused RMSNorm + QKV matvec NR2: reads x_buf, applies inline
                    // normalization (x[i]*scale*norm_w[i]), writes qkv_buf directly.
                    // (Q+gate fusion handles its own fused matmuls below.)
                    match meta.wq_quant {
                        QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2),
                        QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                        QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                        _ => unreachable!(),
                    }
                    enc.set_buffer(layer_buf, wq_off, 0);
                    enc.set_buffer(&s.x_buf, 0, 1);
                    enc.set_buffer(&s.qkv_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
                    enc.set_buffer(layer_buf, attn_norm_off, 5);
                    enc.set_bytes(&eps.to_le_bytes(), 6);
                    let n_tg = ((qkv_dim as u64) + 1) / 2;
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                } else if !use_fused_attn_norm {
                // Non-fused: separate RMSNorm + QKV matvec
                // Attention RMSNorm
                enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
                enc.set_buffer(&s.x_buf, 0, 0);
                enc.set_buffer(layer_buf, attn_norm_off, 1);
                enc.set_buffer(&s.normed_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&eps.to_le_bytes(), 4);
                enc.dispatch_threadgroups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(norm_tg_size, 1, 1),
                );

                // Barrier: RMSNorm writes normed_buf, QKV matmul reads normed_buf
                if needs_barriers { enc.memory_barrier_with_scope(1); }
                }

                // QKV projection: two paths depending on Q+gate fusion.
                if meta.has_qgate_fusion {
                    // Q+gate fusion (Qwen3.5 full-attention layers).
                    // attn_q.weight output is interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...].
                    // K and V come from separate attn_k.weight / attn_v.weight.
                    // sigmoid(gate) applied to attention output BEFORE Wo projection.
                    let qgate_dim = q_dim * 2;
                    // When fused, Q+gate/K/V all fuse RMSNorm inline (read x_buf),
                    // run in parallel, then a single barrier before deinterleave.
                    // Saves 1 dispatch (RMSNorm) + 2 barriers per layer vs non-fused path.

                    // Project Q+gate into qkv_buf
                    {
                        if use_fused_attn_norm {
                            match meta.wq_quant {
                                QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                                QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                                _ => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2),
                            }
                            enc.set_buffer(layer_buf, wq_off, 0);
                            enc.set_buffer(&s.x_buf, 0, 1);
                            enc.set_buffer(&s.qkv_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(qgate_dim as u32).to_le_bytes(), 4);
                            enc.set_buffer(layer_buf, attn_norm_off, 5);
                            enc.set_bytes(&eps.to_le_bytes(), 6);
                        } else {
                            let _tg = match meta.wq_quant {
                                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                            };
                            enc.set_buffer(layer_buf, wq_off, 0);
                            enc.set_buffer(&s.normed_buf, 0, 1);
                            enc.set_buffer(&s.qkv_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            if matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                                enc.set_bytes(&(qgate_dim as u32).to_le_bytes(), 4);
                            }
                        }
                        let n_tg = match meta.wq_quant { QuantScheme::Q8_0 => ((qgate_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((qgate_dim as u64) + 1) / 2, QuantScheme::F16 => ((qgate_dim as u64) + 1) / 2, _ => qgate_dim as u64 };
                        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                    }
                    // Project K from wk (parallel with Q+gate when fused)
                    {
                        let wk_off_val = meta.wk_off.unwrap();
                        let wk_quant = meta.wk_quant.unwrap();
                        if use_fused_attn_norm {
                            match wk_quant {
                                QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                                QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                                _ => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2),
                            }
                            enc.set_buffer(layer_buf, wk_off_val, 0);
                            enc.set_buffer(&s.x_buf, 0, 1);
                            enc.set_buffer(&s.k_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                            enc.set_buffer(layer_buf, attn_norm_off, 5);
                            enc.set_bytes(&eps.to_le_bytes(), 6);
                        } else {
                            let _tg = match wk_quant {
                                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                            };
                            enc.set_buffer(layer_buf, wk_off_val, 0);
                            enc.set_buffer(&s.normed_buf, 0, 1);
                            enc.set_buffer(&s.k_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            if matches!(wk_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                                enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                            }
                        }
                        let n_tg = match wk_quant { QuantScheme::Q8_0 => ((kv_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((kv_dim as u64) + 1) / 2, QuantScheme::F16 => ((kv_dim as u64) + 1) / 2, _ => kv_dim as u64 };
                        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                    }
                    // Project V from wv (parallel with Q+gate and K when fused)
                    {
                        let wv_off_val = meta.wv_off.unwrap();
                        let wv_quant = meta.wv_quant.unwrap();
                        if use_fused_attn_norm {
                            match wv_quant {
                                QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                                QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                                _ => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2),
                            }
                            enc.set_buffer(layer_buf, wv_off_val, 0);
                            enc.set_buffer(&s.x_buf, 0, 1);
                            enc.set_buffer(&s.v_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                            enc.set_buffer(layer_buf, attn_norm_off, 5);
                            enc.set_bytes(&eps.to_le_bytes(), 6);
                        } else {
                            let _tg = match wv_quant {
                                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                            };
                            enc.set_buffer(layer_buf, wv_off_val, 0);
                            enc.set_buffer(&s.normed_buf, 0, 1);
                            enc.set_buffer(&s.v_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            if matches!(wv_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                                enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                            }
                        }
                        let n_tg = match wv_quant { QuantScheme::Q8_0 => ((kv_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((kv_dim as u64) + 1) / 2, QuantScheme::F16 => ((kv_dim as u64) + 1) / 2, _ => kv_dim as u64 };
                        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                    }
                    // Barrier: Q+gate/K/V projections all complete
                    if needs_barriers { enc.memory_barrier_with_scope(1); }

                    // Fused deinterleave + norm + assemble (saves 5 dispatches + 2 barriers per layer).
                    // Falls back to separate dispatches if fused kernel or norm weights unavailable.
                    let use_fused_dna = pipelines.deinterleave_norm_assemble.is_some()
                        && meta.attn_q_norm_off.is_some()
                        && meta.attn_k_norm_off.is_some();

                    if use_fused_dna {
                        let pso = pipelines.deinterleave_norm_assemble.as_ref().unwrap();
                        let q_norm_off = meta.attn_q_norm_off.unwrap();
                        let k_norm_off = meta.attn_k_norm_off.unwrap();
                        enc.set_pipeline_state(pso);
                        enc.set_buffer(&s.qkv_buf, 0, 0);            // qgate_interleaved (input)
                        enc.set_buffer(&s.k_buf, 0, 1);              // k_data
                        enc.set_buffer(&s.v_buf, 0, 2);              // v_data
                        enc.set_buffer(layer_buf, q_norm_off, 3);     // q_norm_weight
                        enc.set_buffer(layer_buf, k_norm_off, 4);     // k_norm_weight
                        enc.set_buffer(&s.qkv_buf, 0, 5);            // qkv_out (K/V assembled here)
                        enc.set_buffer(&s.gate_buf, 0, 6);            // gate_out
                        enc.set_bytes(&(num_heads as u32).to_le_bytes(), 7);
                        enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 8);
                        enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 10);
                        enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 11);
                        enc.set_bytes(&eps.to_le_bytes(), 12);
                        enc.set_buffer(&s.q_buf, 0, 13);             // q_out (separate to avoid aliasing)
                        let total_tgs = (num_heads + num_kv_heads) as u64;
                        let tg_threads = 256u64.min(head_dim as u64).max(32);
                        enc.dispatch_threadgroups(
                            MTLSize::new(total_tgs, 1, 1),
                            MTLSize::new(tg_threads, 1, 1),
                        );
                        // Copy normalized Q from q_buf to qkv_buf[0..q_dim]
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                        enc.set_pipeline_state(&pipelines.copy_buffer);
                        enc.set_buffer(&s.q_buf, 0, 0);
                        enc.set_buffer(&s.qkv_buf, 0, 1);
                        {
                            let tg = 256u64.min(q_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((q_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                    } else {
                        // Fallback: separate deinterleave + norm + copy
                        {
                            let pso = pipelines.deinterleave_qgate.as_ref().ok_or_else(|| {
                                RuntimeError::Compute("deinterleave_qgate pipeline not compiled".into())
                            })?;
                            enc.set_pipeline_state(pso);
                            enc.set_buffer(&s.qkv_buf, 0, 0);
                            enc.set_buffer(&s.q_buf, 0, 1);
                            enc.set_buffer(&s.gate_buf, 0, 2);
                            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);
                            let tg_di = 256u64.min(q_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((q_dim as u64).div_ceil(tg_di), 1, 1),
                                MTLSize::new(tg_di, 1, 1),
                            );
                        }
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                        if let (Some(q_norm_off), Some(k_norm_off)) = (meta.attn_q_norm_off, meta.attn_k_norm_off) {
                            let pso = pipelines.rmsnorm_per_head.as_ref().ok_or_else(|| {
                                RuntimeError::Compute("rmsnorm_per_head pipeline not compiled".into())
                            })?;
                            let head_dim_u32 = head_dim as u32;
                            let tg_rms = 256u64.min(head_dim as u64).max(32);
                            enc.set_pipeline_state(pso);
                            enc.set_buffer(&s.q_buf, 0, 0);
                            enc.set_buffer(layer_buf, q_norm_off, 1);
                            enc.set_buffer(&s.q_buf, 0, 2);
                            enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                            enc.set_bytes(&eps.to_le_bytes(), 4);
                            enc.dispatch_threadgroups(
                                MTLSize::new(num_heads as u64, 1, 1),
                                MTLSize::new(tg_rms, 1, 1),
                            );
                            enc.set_pipeline_state(pso);
                            enc.set_buffer(&s.k_buf, 0, 0);
                            enc.set_buffer(layer_buf, k_norm_off, 1);
                            enc.set_buffer(&s.k_buf, 0, 2);
                            enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                            enc.set_bytes(&eps.to_le_bytes(), 4);
                            enc.dispatch_threadgroups(
                                MTLSize::new(num_kv_heads as u64, 1, 1),
                                MTLSize::new(tg_rms, 1, 1),
                            );
                        }
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                        enc.set_pipeline_state(&pipelines.copy_buffer);
                        enc.set_buffer(&s.q_buf, 0, 0);
                        enc.set_buffer(&s.qkv_buf, 0, 1);
                        {
                            let tg = 256u64.min(q_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((q_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                        enc.set_buffer(&s.k_buf, 0, 0);
                        enc.set_buffer(&s.qkv_buf, k_byte_off, 1);
                        {
                            let tg = 256u64.min(kv_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((kv_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                        enc.set_buffer(&s.v_buf, 0, 0);
                        enc.set_buffer(&s.qkv_buf, v_byte_off, 1);
                        {
                            let tg = 256u64.min(kv_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((kv_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                    }
                    // gate_buf holds pre-sigmoid gate [q_dim], applied after attention.
                } else if !use_fused_attn_norm {
                // Fused QKV projection (+ fused bias for Qwen2-family models)
                // Skipped when fused RMSNorm+QKV already wrote qkv_buf.
                {
                    let has_bias = meta.bq_off.is_some() && meta.bk_off.is_some() && meta.bv_off.is_some();
                    let tg = if has_bias && matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                        match meta.wq_quant {
                            QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_bias_nr2),
                            QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_bias_nr2),
                            QuantScheme::F16 => enc.set_pipeline_state(&pipelines.matmul_f16_deferred_bias_nr2),
                            _ => unreachable!(),
                        };
                        128u64
                    } else {
                        match meta.wq_quant {
                            QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                            QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                            QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                            _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                        }
                    };
                    enc.set_buffer(layer_buf, wq_off, 0);
                    enc.set_buffer(&s.normed_buf, 0, 1);
                    enc.set_buffer(&s.qkv_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    if matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                        enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
                    }
                    if has_bias && matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                        enc.set_buffer(layer_buf, meta.bq_off.unwrap(), 5);
                        enc.set_buffer(layer_buf, meta.bk_off.unwrap(), 6);
                        enc.set_buffer(layer_buf, meta.bv_off.unwrap(), 7);
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 8);
                        let qk_dim = (q_dim + kv_dim) as u32;
                        enc.set_bytes(&qk_dim.to_le_bytes(), 9);
                    }
                    let n_tg = if tg == 64 {
                        ((qkv_dim as u64) + 7) / 8  // (dead path: Q8_0 now uses deferred with tg=128)
                    } else {
                        match meta.wq_quant { QuantScheme::Q8_0 => ((qkv_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((qkv_dim as u64) + 1) / 2, QuantScheme::F16 => ((qkv_dim as u64) + 1) / 2, _ => qkv_dim as u64 }
                    };
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
                }

                // QKV bias addition fallback (only for F32 weights with bias, rare)
                if !matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
                    && (meta.bq_off.is_some() || meta.bk_off.is_some() || meta.bv_off.is_some())
                {
                    enc.set_pipeline_state(&pipelines.bias_add);
                    if let Some(bq_off) = meta.bq_off {
                        enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                        enc.set_buffer(layer_buf, bq_off, 1);
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 2);
                        let n_tg_bq = (q_dim as u64 + 255) / 256;
                        enc.dispatch_threadgroups(MTLSize::new(n_tg_bq, 1, 1), MTLSize::new(256, 1, 1));
                    }
                    if let Some(bk_off) = meta.bk_off {
                        enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
                        enc.set_buffer(layer_buf, bk_off, 1);
                        enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 2);
                        let n_tg_bk = (kv_dim as u64 + 255) / 256;
                        enc.dispatch_threadgroups(MTLSize::new(n_tg_bk, 1, 1), MTLSize::new(256, 1, 1));
                    }
                    if let Some(bv_off) = meta.bv_off {
                        enc.set_buffer(&s.qkv_buf, v_byte_off, 0);
                        enc.set_buffer(layer_buf, bv_off, 1);
                        enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 2);
                        let n_tg_bv = (kv_dim as u64 + 255) / 256;
                        enc.dispatch_threadgroups(MTLSize::new(n_tg_bv, 1, 1), MTLSize::new(256, 1, 1));
                    }
                }
                }

                // Barrier: QKV projection writes qkv_buf, RoPE reads qkv_buf
                if needs_barriers { enc.memory_barrier_with_scope(1); }

                // Fused RoPE Q + RoPE K + KV cache write (1 dispatch instead of 3)
                // Only used for full RoPE (rotary_dim == head_dim) on non-linear attention layers.
                // Partial RoPE (Qwen3.5-MoE) and linear attention layers fall back to separate dispatches.
                let is_linear_attn = meta.layer_type == Some(1);
                let rope_half_dim = s.rotary_dim / 2;
                let use_fused_rope_kv = !is_linear_attn && s.rotary_dim == head_dim;
                const FLASH_DECODE_THRESHOLD: usize = 257; // FLASH_DECODE_TILE_SIZE + 1: single-tile flash_decode is a no-op reduce

                // Fused RoPE + KV cache write + MHA (eliminates 2 barriers per layer)
                // Only for: standard RoPE (not NeoX), short sequences, full rotary_dim
                let use_fused_rope_kv_mha = use_fused_rope_kv && !s.is_qwen35moe && new_seq_len < FLASH_DECODE_THRESHOLD;

                if use_fused_rope_kv_mha {
                    // Single dispatch: RoPE Q/K + KV cache write + MHA
                    let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
                    enc.set_pipeline_state(&pipelines.fused_rope_kv_mha);
                    enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                    enc.set_buffer(&s.qkv_buf, k_byte_off, 1);
                    enc.set_buffer(&s.qkv_buf, v_byte_off, 2);
                    enc.set_buffer(&s.rope_cos_buf, 0, 3);
                    enc.set_buffer(&s.rope_sin_buf, 0, 4);
                    enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 5);
                    enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 6);
                    enc.set_buffer(&s.attn_out_buf, 0, 7);
                    enc.set_buffer(&s.mha_scores_buf, 0, 8);
                    enc.set_bytes(&(num_heads as u32).to_le_bytes(), 9);
                    enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 10);
                    enc.set_bytes(&(head_dim as u32).to_le_bytes(), 11);
                    enc.set_bytes(&(rope_half_dim as u32).to_le_bytes(), 12);
                    enc.set_bytes(&pos_offset_u32.to_le_bytes(), 13);
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 14);
                    enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 15);
                    enc.set_bytes(&attn_scale.to_le_bytes(), 16);
                    enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 17);
                    let tg_threads = 256u64.min((head_dim.max(new_seq_len) as u64).max(32));
                    enc.dispatch_threadgroups(
                        MTLSize::new(num_heads as u64, 1, 1),
                        MTLSize::new(tg_threads, 1, 1),
                    );
                } else {

                // Fused RoPE Q + RoPE K + KV cache write (1 dispatch instead of 3)
                // Only used for full RoPE (rotary_dim == head_dim) on non-linear attention layers.
                // Partial RoPE (Qwen3.5-MoE) and linear attention layers fall back to separate dispatches.
                let is_linear_attn = meta.layer_type == Some(1);
                let rope_half_dim = s.rotary_dim / 2;
                let use_fused_rope_kv = !is_linear_attn && s.rotary_dim == head_dim;
                if use_fused_rope_kv {
                    let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
                    let fused_pipe = if s.is_qwen35moe {
                        pipelines.fused_rope_neox_kv_write.as_ref().unwrap_or(&pipelines.fused_rope_kv_write)
                    } else {
                        &pipelines.fused_rope_kv_write
                    };
                    enc.set_pipeline_state(fused_pipe);
                    enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                    enc.set_buffer(&s.qkv_buf, k_byte_off, 1);
                    enc.set_buffer(&s.qkv_buf, v_byte_off, 2);
                    enc.set_buffer(&s.rope_cos_buf, 0, 3);
                    enc.set_buffer(&s.rope_sin_buf, 0, 4);
                    enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 5);
                    enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 6);
                    enc.set_bytes(&(num_heads as u32).to_le_bytes(), 7);
                    enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 8);
                    enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);
                    enc.set_bytes(&(rope_half_dim as u32).to_le_bytes(), 10);
                    enc.set_bytes(&pos_offset_u32.to_le_bytes(), 11);
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 12);
                    enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 13);
                    enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 14);
                    let total_threads = (num_heads * rope_half_dim + num_kv_heads * rope_half_dim + kv_dim) as u64;
                    let tg = 64u64.min(total_threads.max(1));
                    enc.dispatch_threadgroups(
                        MTLSize::new(total_threads.div_ceil(tg), 1, 1),
                        MTLSize::new(tg, 1, 1),
                    );
                } else {
                    if !is_linear_attn {
                        let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
                        let rope_pipe = if s.is_qwen35moe {
                            pipelines.rope_neox.as_ref().unwrap_or(&pipelines.rope)
                        } else {
                            &pipelines.rope
                        };
                        enc.set_pipeline_state(rope_pipe);
                        enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                        enc.set_buffer(&s.rope_cos_buf, 0, 1);
                        enc.set_buffer(&s.rope_sin_buf, 0, 2);
                        enc.set_bytes(&(num_heads as u32).to_le_bytes(), 3);
                        enc.set_bytes(&(head_dim as u32).to_le_bytes(), 4);
                        enc.set_bytes(&(rope_half_dim as u32).to_le_bytes(), 5);
                        enc.set_bytes(&pos_offset_u32.to_le_bytes(), 6);
                        let q_total_half = (num_heads * rope_half_dim) as u64;
                        let tg_q = 64u64.min(q_total_half.max(1));
                        enc.dispatch_threadgroups(
                            MTLSize::new(q_total_half.div_ceil(tg_q), 1, 1),
                            MTLSize::new(tg_q, 1, 1),
                        );
                        enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
                        enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 3);
                        let k_total_half = (num_kv_heads * rope_half_dim) as u64;
                        let tg_k = 64u64.min(k_total_half.max(1));
                        enc.dispatch_threadgroups(
                            MTLSize::new(k_total_half.div_ceil(tg_k), 1, 1),
                            MTLSize::new(tg_k, 1, 1),
                        );
                    }
                    enc.set_pipeline_state(&pipelines.write_kv_cache);
                    enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
                    enc.set_buffer(&s.qkv_buf, v_byte_off, 1);
                    enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 2);
                    enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 3);
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                    enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 5);
                    enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 6);
                    {
                        let tg = 64u64.min(kv_dim as u64).max(1);
                        enc.dispatch_threadgroups(
                            MTLSize::new((kv_dim as u64).div_ceil(tg), 1, 1),
                            MTLSize::new(tg, 1, 1),
                        );
                    }
                }

                // Barrier: RoPE+KV cache write complete, attention reads KV cache + qkv_buf Q
                if needs_barriers { enc.memory_barrier_with_scope(1); }
                // Attention (flash decode or MHA)
                {
                    let num_heads_u32 = num_heads as u32;
                    let num_kv_heads_u32 = num_kv_heads as u32;
                    let head_dim_u32 = head_dim as u32;
                    let kv_dim_u32 = kv_dim as u32;
                    let seq_len_u32 = new_seq_len as u32;
                    let max_seq_len_u32 = s.max_seq_len as u32;
                    const FLASH_DECODE_TILE_SIZE: u32 = 256;
                    const FLASH_DECODE_THRESHOLD: usize = FLASH_DECODE_TILE_SIZE as usize + 1; // 257: single-tile is a no-op reduce

                    if new_seq_len >= FLASH_DECODE_THRESHOLD {
                        let num_tiles = ((new_seq_len as u32) + FLASH_DECODE_TILE_SIZE - 1) / FLASH_DECODE_TILE_SIZE;
                        enc.set_pipeline_state(&pipelines.flash_decode_attention);
                        enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                        enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 1);
                        enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 2);
                        enc.set_buffer(&s.flash_decode_partial_buf, 0, 3);
                        enc.set_bytes(&num_heads_u32.to_le_bytes(), 4);
                        enc.set_bytes(&num_kv_heads_u32.to_le_bytes(), 5);
                        enc.set_bytes(&head_dim_u32.to_le_bytes(), 6);
                        enc.set_bytes(&kv_dim_u32.to_le_bytes(), 7);
                        enc.set_bytes(&seq_len_u32.to_le_bytes(), 8);
                        enc.set_bytes(&attn_scale.to_le_bytes(), 9);
                        enc.set_bytes(&FLASH_DECODE_TILE_SIZE.to_le_bytes(), 10);
                        enc.set_bytes(&num_tiles.to_le_bytes(), 11);
                        enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 12);
                        enc.dispatch_threadgroups(
                            MTLSize::new((num_heads as u64) * (num_tiles as u64), 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                        // Barrier: flash_decode writes partial_buf, reduce reads partial_buf
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                        enc.set_pipeline_state(&pipelines.flash_decode_reduce);
                        enc.set_buffer(&s.flash_decode_partial_buf, 0, 0);
                        enc.set_buffer(&s.attn_out_buf, 0, 1);
                        enc.set_bytes(&num_heads_u32.to_le_bytes(), 2);
                        enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                        enc.set_bytes(&num_tiles.to_le_bytes(), 4);
                        let tg_threads = (head_dim as u64).max(1).min(256);
                        enc.dispatch_threadgroups(
                            MTLSize::new(num_heads as u64, 1, 1),
                            MTLSize::new(tg_threads, 1, 1),
                        );
                    } else {
                        let mha_tg_size = s.mha_tg_size;
                        enc.set_pipeline_state(&pipelines.multi_head_attention);
                        enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                        enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 1);
                        enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 2);
                        enc.set_buffer(&s.attn_out_buf, 0, 3);
                        enc.set_buffer(&s.mha_scores_buf, 0, 4);
                        enc.set_bytes(&num_heads_u32.to_le_bytes(), 5);
                        enc.set_bytes(&num_kv_heads_u32.to_le_bytes(), 6);
                        enc.set_bytes(&head_dim_u32.to_le_bytes(), 7);
                        enc.set_bytes(&kv_dim_u32.to_le_bytes(), 8);
                        enc.set_bytes(&seq_len_u32.to_le_bytes(), 9);
                        enc.set_bytes(&attn_scale.to_le_bytes(), 10);
                        enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 11);
                        let tg_threads = mha_tg_size.min((head_dim.max(new_seq_len) as u64).max(1));
                        enc.dispatch_threadgroups(
                            MTLSize::new(num_heads as u64, 1, 1),
                            MTLSize::new(tg_threads, 1, 1),
                        );
                    }
                }

                } // end fallback (non-fused RoPE+KV+MHA)

                // Barrier: attention writes attn_out_buf, Wo reads attn_out_buf
                if needs_barriers { enc.memory_barrier_with_scope(1); }
                // Wo projection + Residual
                let has_attn_extras = meta.attn_post_norm_off.is_some() || meta.attn_gate_off.is_some() || meta.has_qgate_fusion;
                if has_attn_extras {
                    // Apply sigmoid(gate) * attn_out BEFORE Wo (Q+gate fusion).
                    if meta.has_qgate_fusion {
                        let pso = pipelines.sigmoid_mul_fused.as_ref().ok_or_else(|| {
                            RuntimeError::Compute("sigmoid_mul_fused pipeline not compiled".into())
                        })?;
                        enc.set_pipeline_state(pso);
                        enc.set_buffer(&s.gate_buf, 0, 0);     // gate [q_dim]
                        enc.set_buffer(&s.attn_out_buf, 0, 1);  // attn output [q_dim]
                        enc.set_buffer(&s.attn_out_buf, 0, 2);  // output (in-place)
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                        let tg = 256u64.min(q_dim as u64).max(1);
                        enc.dispatch_threadgroups(
                            MTLSize::new((q_dim as u64).div_ceil(tg), 1, 1),
                            MTLSize::new(tg, 1, 1),
                        );
                    }

                    // Barrier: sigmoid_mul writes attn_out_buf, Wo reads it
                    if needs_barriers { enc.memory_barrier_with_scope(1); }
                    // Non-fused Wo: attn_proj_buf = Wo * attn_out (NO residual)
                    {
                        let tg_wo = match meta.wo_quant {
                            QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                            QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                            QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                            _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                        };
                        enc.set_buffer(layer_buf, wo_off, 0);
                        enc.set_buffer(&s.attn_out_buf, 0, 1);
                        enc.set_buffer(&s.attn_proj_buf, 0, 2);
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                        if matches!(meta.wo_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                        }
                        let n_tg_wo = match meta.wo_quant { QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2, _ => hidden_dim as u64 };
                        enc.dispatch_threadgroups(MTLSize::new(n_tg_wo, 1, 1), MTLSize::new(tg_wo, 1, 1));
                    }
                    // Barrier: Wo writes attn_proj_buf
                    if needs_barriers { enc.memory_barrier_with_scope(1); }
                    // Post-attention RMSNorm: only for architectures that have
                    // BOTH attn_post_norm AND attn_gate (not Q+gate fusion).
                    // For Qwen3.5 Q+gate fusion, post_attention_norm is the
                    // pre-FFN norm (via ffn_norm_off) — must not be applied here.
                    let did_post_norm = meta.attn_gate_off.is_some() && meta.attn_post_norm_off.is_some();
                    if let (true, Some(post_norm_off)) = (did_post_norm, meta.attn_post_norm_off) {
                        enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
                        enc.set_buffer(&s.attn_proj_buf, 0, 0);
                        enc.set_buffer(layer_buf, post_norm_off, 1);
                        enc.set_buffer(&s.down_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        enc.set_bytes(&eps.to_le_bytes(), 4);
                        enc.dispatch_threadgroups(
                            MTLSize::new(1, 1, 1),
                            MTLSize::new(norm_tg_size, 1, 1),
                        );
                    }
                    // Barrier: only needed when post_norm dispatched (writes down_buf
                    // for subsequent gate matmul). When !did_post_norm, the Wo barrier
                    // above already covers attn_proj_buf visibility.
                    if did_post_norm {
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                    }
                    // Attention output gate
                    if let Some(gate_off) = meta.attn_gate_off {
                        let gate_quant = meta.attn_gate_quant.unwrap_or(QuantScheme::F32);
                        let src_buf = if did_post_norm { &s.down_buf } else { &s.attn_proj_buf };
                        let attn_gate_buf = s.attn_gate_buf.as_ref().ok_or_else(|| {
                            RuntimeError::Compute("attn_gate_buf not allocated".into())
                        })?;
                        // Gate matmul
                        {
                            let tg_gate = match gate_quant {
                                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                            };
                            enc.set_buffer(layer_buf, gate_off, 0);
                            enc.set_buffer(src_buf, 0, 1);
                            enc.set_buffer(attn_gate_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            if matches!(gate_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                            }
                            let n_tg_gate = match gate_quant { QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2, _ => hidden_dim as u64 };
                            enc.dispatch_threadgroups(MTLSize::new(n_tg_gate, 1, 1), MTLSize::new(tg_gate, 1, 1));
                        }
                        // Barrier: gate matmul writes attn_gate_buf, SwiGLU reads it
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                        // SwiGLU gate
                        enc.set_pipeline_state(&pipelines.swiglu);
                        enc.set_buffer(attn_gate_buf, 0, 0);
                        enc.set_buffer(src_buf, 0, 1);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
                        {
                            let tg = 256u64.min(hidden_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                        // Barrier: SwiGLU writes attn_gate_buf, residual reads it
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                        // Fused residual + copy (saves 1 dispatch + 1 barrier)
                        // x_buf += attn_gate_buf; attn_proj_buf = x_buf
                        {
                            let pso = pipelines.residual_add_copy.as_ref().unwrap_or(&pipelines.add_residual);
                            if pipelines.residual_add_copy.is_some() {
                                enc.set_pipeline_state(pso);
                                enc.set_buffer(&s.x_buf, 0, 0);
                                enc.set_buffer(attn_gate_buf, 0, 1);
                                enc.set_buffer(&s.attn_proj_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                let tg = 256u64.min(hidden_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                                    MTLSize::new(tg, 1, 1),
                                );
                            } else {
                                // Fallback: separate add + barrier + copy
                                enc.set_pipeline_state(&pipelines.add_residual);
                                enc.set_buffer(&s.x_buf, 0, 0);
                                enc.set_buffer(attn_gate_buf, 0, 1);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
                                let tg = 256u64.min(hidden_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                                    MTLSize::new(tg, 1, 1),
                                );
                                if needs_barriers { enc.memory_barrier_with_scope(1); }
                                enc.set_pipeline_state(&pipelines.copy_buffer);
                                enc.set_buffer(&s.x_buf, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                let tg2 = 256u64.min(hidden_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((hidden_dim as u64).div_ceil(tg2), 1, 1),
                                    MTLSize::new(tg2, 1, 1),
                                );
                            }
                        }
                    } else {
                        // No attn_gate: fused residual + copy
                        // x_buf += attn_proj_buf; attn_proj_buf = x_buf
                        {
                            let pso = pipelines.residual_add_copy.as_ref().unwrap_or(&pipelines.add_residual);
                            if pipelines.residual_add_copy.is_some() {
                                enc.set_pipeline_state(pso);
                                enc.set_buffer(&s.x_buf, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                enc.set_buffer(&s.attn_proj_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                let tg = 256u64.min(hidden_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                                    MTLSize::new(tg, 1, 1),
                                );
                            } else {
                                // Fallback: separate add + barrier + copy
                                enc.set_pipeline_state(&pipelines.add_residual);
                                enc.set_buffer(&s.x_buf, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
                                let tg = 256u64.min(hidden_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                                    MTLSize::new(tg, 1, 1),
                                );
                                if needs_barriers { enc.memory_barrier_with_scope(1); }
                                enc.set_pipeline_state(&pipelines.copy_buffer);
                                enc.set_buffer(&s.x_buf, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                let tg2 = 256u64.min(hidden_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((hidden_dim as u64).div_ceil(tg2), 1, 1),
                                    MTLSize::new(tg2, 1, 1),
                                );
                            }
                        }
                    }
                } else {
                    // Standard fused Wo + Residual path
                    let tg_wo = match meta.wo_quant {
                        QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_residual_nr2); 128u64 },
                        QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_residual_nr2); 128u64 },
                        QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_residual_nr2); 128u64 },
                        _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32_residual); matmul_tg_size },
                    };
                    enc.set_buffer(layer_buf, wo_off, 0);
                    enc.set_buffer(&s.attn_out_buf, 0, 1);
                    enc.set_buffer(&s.attn_proj_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    enc.set_buffer(&s.x_buf, 0, 4);
                    if matches!(meta.wo_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
                    }
                    let n_tg_wo = match meta.wo_quant { QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2, _ => hidden_dim as u64 };
                    enc.dispatch_threadgroups(MTLSize::new(n_tg_wo, 1, 1), MTLSize::new(tg_wo, 1, 1));
                }
            } else {
                // GatedDeltaNet layer: linear attention forward pass
                let gdn_idx = meta.gdn_layer_idx.unwrap();
                // Fused variant: all GDN dispatches go through the layer encoder.
                let new_conv_pos = Self::encode_gdn_layer_decode_fused(
                    &enc, pipelines, s, layer_buf, meta, gdn_idx,
                )?;
                s.gdn_conv_positions[gdn_idx] = new_conv_pos;
            }

            // ================================================================
            // FFN BLOCK
            // ================================================================

            // Fused RMSNorm + FFN for dense Q8_0 gate+up.
            // Eliminates FFN RMSNorm dispatch + 1 barrier + normed_buf write/read.
            let use_fused_ffn_norm = meta.moe_meta.is_none()
                && matches!(meta.w_gate_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
                && meta.w_gate_quant == meta.w_up_quant;

            if !use_fused_ffn_norm {
            // Non-fused path: separate FFN RMSNorm + barrier
            // Barrier: Wo+residual writes attn_proj_buf, FFN RMSNorm reads it
            if needs_barriers { enc.memory_barrier_with_scope(1); }
            // FFN RMSNorm
            enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
            enc.set_buffer(&s.attn_proj_buf, 0, 0);
            enc.set_buffer(layer_buf, ffn_norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );

            // Barrier: FFN RMSNorm writes normed_buf, gate+up reads normed_buf
            if needs_barriers { enc.memory_barrier_with_scope(1); }
            }

            // FFN block: branch on MoE vs dense
            if let Some(ref moe_meta) = meta.moe_meta {
                // use_batched no longer requires use_option_a. The batched
                // MoE path uses 2 kernel dispatches within the current encoder instead
                // of ~258 separate encoders (legacy path). The legacy path created
                // ~10,360 encoders on a single command buffer for 40-layer MoE models,
                // which caused Metal to produce corrupted router output buffers.
                let has_down_kernel = match moe_meta.expert_down_quant {
                    QuantScheme::Q4_1 => pipelines.moe_batched_down_accum_q4_1.is_some(),
                    QuantScheme::Q4_0 => pipelines.moe_batched_down_accum_q4_0.is_some(),
                    QuantScheme::Q8_0 => pipelines.moe_batched_down_accum_q8_0.is_some(),
                    _ => false,
                };
                let has_gate_kernel = match moe_meta.expert_gate_quant {
                    QuantScheme::Q4_0 => pipelines.moe_batched_gate_up_swiglu_q4_0.is_some(),
                    QuantScheme::Q4_1 => pipelines.moe_batched_gate_up_swiglu_q4_1.is_some(),
                    QuantScheme::Q8_0 => pipelines.moe_batched_gate_up_swiglu_q8_0.is_some(),
                    _ => false,
                };
                let use_batched = has_gate_kernel
                    && has_down_kernel
                    && s.moe_gate_up_offsets.get(layer_idx).and_then(|o| o.as_ref()).is_some()
                    && s.moe_down_offsets.get(layer_idx).and_then(|o| o.as_ref()).is_some()
                    && s.moe_batched_swiglu_buf.is_some();

                if use_batched {
                    let per_layer_ids_buf = s.moe_per_layer_expert_ids
                        .get(layer_idx)
                        .and_then(|opt| opt.as_ref());
                    let expert_ids_buf = per_layer_ids_buf.unwrap_or_else(|| {
                        s.moe_expert_ids.as_ref().unwrap()
                    });
                    let expert_weights_buf = s.moe_expert_weights.as_ref().unwrap();
                    let gate_up_off_buf = s.moe_gate_up_offsets[layer_idx].as_ref().unwrap();
                    let down_off_buf = s.moe_down_offsets[layer_idx].as_ref().unwrap();

                    // Router dispatch (all within same encoder)
                    {
                        let router_softmax = pipelines.moe_router_softmax.as_ref().ok_or_else(|| {
                            RuntimeError::Compute("MoE router_softmax pipeline not compiled.".into())
                        })?;
                        enc.set_pipeline_state(router_softmax);
                        enc.set_buffer(&s.normed_buf, 0, 0);
                        enc.set_buffer(layer_buf, moe_meta.router_weight_off, 1);
                        enc.set_buffer(expert_ids_buf, 0, 2);
                        enc.set_buffer(expert_weights_buf, 0, 3);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                        enc.set_bytes(&(s.moe_num_experts as u32).to_le_bytes(), 5);
                        enc.set_bytes(&(s.moe_num_active_experts as u32).to_le_bytes(), 6);
                        let tg = 256u64.min(hidden_dim as u64).max(1);
                        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
                    }

                    // Barrier: router writes expert_ids/weights, batched FFN reads them
                    if needs_barriers { enc.memory_barrier_with_scope(1); }
                    // Batched expert FFN + shared expert (fused when available)
                    Self::encode_moe_ffn_with_shared_fused(
                        &enc, pipelines, s, layer_buf, layer_idx,
                        moe_meta, meta,
                        expert_ids_buf, expert_weights_buf,
                        gate_up_off_buf, down_off_buf,
                    )?;
                } else {
                    // Legacy path: per-expert dispatch (needs its own encoders).
                    // End current encoder, run legacy, then this is the last thing
                    // in the layer so we skip re-opening.
                    enc.end_encoding();
                    let per_layer_ids_buf = s.moe_per_layer_expert_ids
                        .get(layer_idx)
                        .and_then(|opt| opt.as_ref());
                    let per_layer_wts_buf = s.moe_per_layer_expert_weights
                        .get(layer_idx)
                        .and_then(|opt| opt.as_ref());
                    Self::encode_moe_ffn_decode(
                        &cmd, pipelines, s, layer_buf, moe_meta,
                        per_layer_ids_buf,
                        None,
                        None, 0.0,
                        None,
                        false,
                        per_layer_wts_buf,
                    )?;
                    // Shared expert dispatch (legacy path)
                    if meta.shared_expert_gate_off.is_some() {
                        Self::encode_shared_expert_ffn_decode(
                            &cmd, pipelines, s, layer_buf, meta,
                        )?;
                    }
                    // Re-create concurrent encoder for remaining layers
                    enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to re-create concurrent encoder".into())
                    })?;
                    continue; // encoder already ended, skip end_encoding below
                }
            } else {
                // Dense FFN path
                if use_fused_ffn_norm {
                    // Fused RMSNorm + FFN gate+up+SwiGLU.
                    // Reads attn_proj_buf directly, applies inline normalization.
                    // Barrier: Wo+residual writes attn_proj_buf, fused FFN reads it
                    if needs_barriers { enc.memory_barrier_with_scope(1); }
                    if matches!(meta.w_gate_quant, QuantScheme::F16) {
                        // F16: always use deferred 1-row-per-TG pattern (no block structure)
                        enc.set_pipeline_state(&pipelines.rmsnorm_ffn_fused_gate_up_swiglu_f16_deferred);
                        enc.set_buffer(layer_buf, w_gate_off, 0);
                        enc.set_buffer(&s.attn_proj_buf, 0, 1);
                        enc.set_buffer(&s.gate_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                        enc.set_buffer(layer_buf, w_up_off, 5);
                        enc.set_buffer(layer_buf, ffn_norm_off, 6);
                        enc.set_bytes(&eps.to_le_bytes(), 7);
                        enc.dispatch_threadgroups(
                            MTLSize::new(inter_dim as u64, 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                    // For small hidden_dim (x fits in L1 cache), use 8-row
                    // pattern: 8 SGs independently own 1 row each, zero TG barriers.
                    } else if hidden_dim <= 4096 {
                        match meta.w_gate_quant {
                            QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_ffn_fused_gate_up_swiglu_q4_0_8row),
                            _ => enc.set_pipeline_state(&pipelines.rmsnorm_ffn_fused_gate_up_swiglu_q8_0_8row),
                        }
                        enc.set_buffer(layer_buf, w_gate_off, 0);
                        enc.set_buffer(&s.attn_proj_buf, 0, 1);
                        enc.set_buffer(&s.gate_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                        enc.set_buffer(layer_buf, w_up_off, 5);
                        enc.set_buffer(layer_buf, ffn_norm_off, 6);
                        enc.set_bytes(&eps.to_le_bytes(), 7);
                        enc.dispatch_threadgroups(
                            MTLSize::new(((inter_dim as u64) + 7) / 8, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                    } else {
                        match meta.w_gate_quant {
                            QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_ffn_fused_gate_up_swiglu_q4_0_deferred),
                            _ => enc.set_pipeline_state(&pipelines.rmsnorm_ffn_fused_gate_up_swiglu_q8_0_deferred),
                        }
                        enc.set_buffer(layer_buf, w_gate_off, 0);
                        enc.set_buffer(&s.attn_proj_buf, 0, 1);
                        enc.set_buffer(&s.gate_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                        enc.set_buffer(layer_buf, w_up_off, 5);
                        enc.set_buffer(layer_buf, ffn_norm_off, 6);
                        enc.set_bytes(&eps.to_le_bytes(), 7);
                        enc.dispatch_threadgroups(
                            MTLSize::new(inter_dim as u64, 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                    }
                } else if meta.w_gate_quant == QuantScheme::Q8_0 && meta.w_up_quant == QuantScheme::Q8_0 {
                    enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q8_0_deferred);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(&s.normed_buf, 0, 1);
                    enc.set_buffer(&s.gate_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                    enc.set_buffer(layer_buf, w_up_off, 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new(inter_dim as u64, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else if matches!(meta.w_gate_quant, QuantScheme::Q4_0) {
                    enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q4_0_deferred);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(&s.normed_buf, 0, 1);
                    enc.set_buffer(&s.gate_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                    enc.set_buffer(layer_buf, w_up_off, 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new(inter_dim as u64, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else if matches!(meta.w_gate_quant, QuantScheme::F16) {
                    // F16 fused gate+up+SwiGLU (non-norm path -- norm already applied)
                    enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_f16_deferred);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(&s.normed_buf, 0, 1);
                    enc.set_buffer(&s.gate_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                    enc.set_buffer(layer_buf, w_up_off, 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new(inter_dim as u64, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else {
                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(&s.normed_buf, 0, 1);
                    enc.set_buffer(&s.gate_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.dispatch_threadgroups(
                        MTLSize::new(inter_dim as u64, 1, 1),
                        MTLSize::new(matmul_tg_size, 1, 1),
                    );
                    enc.set_buffer(layer_buf, w_up_off, 0);
                    enc.set_buffer(&s.up_buf, 0, 2);
                    enc.dispatch_threadgroups(
                        MTLSize::new(inter_dim as u64, 1, 1),
                        MTLSize::new(matmul_tg_size, 1, 1),
                    );
                    // Barrier: gate+up matmul write gate_buf+up_buf, SwiGLU reads both
                    if needs_barriers { enc.memory_barrier_with_scope(1); }
                    // SwiGLU
                    enc.set_pipeline_state(&pipelines.swiglu);
                    enc.set_buffer(&s.gate_buf, 0, 0);
                    enc.set_buffer(&s.up_buf, 0, 1);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 2);
                    let tg = 256u64.min(inter_dim as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((inter_dim as u64).div_ceil(tg), 1, 1),
                        MTLSize::new(tg, 1, 1),
                    );
                }
                // Barrier: gate+up+SwiGLU writes gate_buf, down proj reads gate_buf
                if needs_barriers { enc.memory_barrier_with_scope(1); }
                // Down projection + Residual 2 (fused)
                {
                    let tg_down = match meta.w_down_quant {
                        QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_residual_nr2); 128u64 },
                        QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_residual_nr2); 128u64 },
                        QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_residual_nr2); 128u64 },
                        _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32_residual); matmul_tg_size },
                    };
                    enc.set_buffer(layer_buf, w_down_off, 0);
                    enc.set_buffer(&s.gate_buf, 0, 1);
                    enc.set_buffer(&s.x_buf, 0, 2);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
                    enc.set_buffer(&s.attn_proj_buf, 0, 4);
                    if matches!(meta.w_down_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
                    }
                    let n_tg_down = match meta.w_down_quant { QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2, _ => hidden_dim as u64 };
                    enc.dispatch_threadgroups(MTLSize::new(n_tg_down, 1, 1), MTLSize::new(tg_down, 1, 1));
                }
            } // end MoE vs dense FFN branch

            // Barrier: down+residual writes x_buf, next layer's RMSNorm reads x_buf
            if needs_barriers { enc.memory_barrier_with_scope(1); }

        } // end layer loop

        // Resolve global tensor buffers for final norm + output projection
        let (sc_norm_buf, sc_norm_off): (&MetalBuffer, u64) =
            if let Some((_, norm_o, _)) = s.gpu_global_offsets {
                (s.gpu_unified_weight_buf.as_ref().unwrap(), norm_o as u64)
            } else {
                (final_norm_buf, 0u64)
            };
        let (sc_proj_buf, sc_proj_off): (&MetalBuffer, u64) =
            if let Some((_, _, proj_o)) = s.gpu_global_offsets {
                (s.gpu_unified_weight_buf.as_ref().unwrap(), proj_o as u64)
            } else {
                (output_proj_buf, 0u64)
            };

        // --- Final RMSNorm + Logits ---
            // Fuse final RMSNorm into output projection for Q8_0/Q4_0.
            // Eliminates 1 dispatch + 1 barrier + normed_buf write/read.
            if matches!(output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                match output_proj_quant {
                    QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                    QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                    _ => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2),
                }
                enc.set_buffer(sc_proj_buf, sc_proj_off, 0);
                enc.set_buffer(&s.x_buf, 0, 1);
                enc.set_buffer(&s.logits_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 4);
                enc.set_buffer(sc_norm_buf, sc_norm_off, 5);
                enc.set_bytes(&eps.to_le_bytes(), 6);
                let n_tg = ((vocab_size as u64) + 1) / 2;
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
            } else {
            // Non-fused path for non-Q8_0 output projections
            // Final RMSNorm
            enc.set_pipeline_state(&pipelines.rmsnorm);
            enc.set_buffer(&s.x_buf, 0, 0);
            enc.set_buffer(sc_norm_buf, sc_norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );
            // Barrier: Final RMSNorm writes normed_buf, logits projection reads normed_buf
            if needs_barriers { enc.memory_barrier_with_scope(1); }
            // Logits projection
            let (proj_tg, proj_rows_per_tg) = match output_proj_quant {
                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); (128u64, 2u64) },
                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); (128u64, 2u64) },
                _ => { enc.set_pipeline_state(&pipelines.matmul_f32_deferred); (128u64, 4u64) },
            };
            enc.set_buffer(sc_proj_buf, sc_proj_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(&s.logits_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 4);
            {
                let n_tg = ((vocab_size as u64) + proj_rows_per_tg - 1) / proj_rows_per_tg;
                enc.dispatch_threadgroups(
                    MTLSize::new(n_tg, 1, 1),
                    MTLSize::new(proj_tg, 1, 1),
                );
            }
            }

        enc.end_encoding();

        // Single sync point for the entire token.
        cmd.commit_and_wait();

        // Record MoE expert activations for ALL layers (per-layer profiling).
        if s.moe_num_experts > 0 {
            if let Some(ref profiler) = self.expert_profiler {
                let top_k = s.moe_num_active_experts;
                let mut ids = vec![0u32; top_k];
                for layer in 0..num_layers {
                    if let Some(Some(ref per_layer_buf)) = s.moe_per_layer_expert_ids.get(layer) {
                        per_layer_buf.read_u32(&mut ids);
                        profiler.lock().unwrap().record(layer, &ids);
                    }
                }
            }

            // Router debug readback -- capture per-layer expert_ids + expert_weights.
            if self.router_debug_enabled {
                let top_k = s.moe_num_active_experts;
                let mut ids = vec![0u32; top_k];
                let mut wts = vec![0.0f32; top_k];
                let mut log = self.router_debug_log.lock().unwrap();
                for layer in 0..num_layers {
                    let has_ids = s.moe_per_layer_expert_ids.get(layer)
                        .and_then(|opt| opt.as_ref());
                    let has_wts = s.moe_per_layer_expert_weights.get(layer)
                        .and_then(|opt| opt.as_ref());
                    if let (Some(ids_buf), Some(wts_buf)) = (has_ids, has_wts) {
                        ids_buf.read_u32(&mut ids);
                        wts_buf.read_f32(&mut wts);
                        let spread = if wts.len() >= 2 { wts[0] - wts[1] } else { 0.0 };
                        log.push(RouterLayerStats {
                            layer,
                            expert_ids: ids.clone(),
                            expert_weights: wts.clone(),
                            weight_spread: spread,
                        });
                    }
                }
            }
        }

        // Check if profiling phase is complete and trigger cache warmup.
        self.maybe_trigger_warmup();

        s.gpu_x_valid = false;
        s.last_async_cmd = None;

        // Advance KV cache (CPU tracking -- GPU KV cache already written).
        kv.advance_seq_len()?;

        // Read logits from GPU.
        let mut logits_data = vec![0.0f32; vocab_size];
        s.logits_buf.read_f32(&mut logits_data);

        drop(scratch_guard);

        Ok(Logits { data: logits_data })
    }

    /// Full-token decode with GPU-side argmax for greedy sampling.
    ///
    /// Identical to `decode_token_single_cb` but appends an argmax kernel
    /// after the logits projection in the SAME command buffer, then reads
    /// back only 4 bytes (u32 token ID) instead of 128 KB of logits.
    ///
    /// This eliminates:
    /// - 128 KB GPU->CPU logits readback
    /// - 128 KB CPU-side Vec<f32> allocation
    /// - CPU-side linear scan of 32000 floats for argmax
    pub fn decode_token_greedy(
        &self,
        token_id: u32,
        _weights: &dyn crate::weight::cache::WeightProvider,
        kv: &mut crate::kv::KvCache,
    ) -> Result<u32, RuntimeError> {
        // Batched MoE path is handled inline in the MoE section below.
        // Only fall back to the old two-CB Option A path if batched kernels are unavailable.
        {
            let has_batched = self.pipelines.as_ref()
                .map(|p| p.moe_batched_gate_up_swiglu_q4_0.is_some())
                .unwrap_or(false);
            if self.use_option_a && !has_batched {
                return self.decode_token_option_a_gpu_resident(token_id, _weights, kv);
            }
        }

        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Metal pipelines not initialized: call init() first".into())
        })?;
        let embedding_buf = self.embedding_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Embedding buffer not initialized".into())
        })?;
        let final_norm_buf = self.final_norm_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Final norm buffer not initialized".into())
        })?;
        let output_proj_buf = self.output_proj_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Output proj buffer not initialized".into())
        })?;
        let output_proj_quant = self.output_proj_quant;

        let seq_pos = kv.seq_len();

        // Single mutex acquisition for the entire token.
        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("Metal scratch not initialized".into())
        })?;
        if let Some(prev_cmd) = s.last_async_cmd.take() {
            prev_cmd.wait_until_completed();
        }
        // GPU-resident check: unified private buffer OR per-layer buffers
        let has_unified = s.gpu_unified_weight_buf.is_some();
        let has_per_layer = s.gpu_resident_layers.is_some();
        if !has_unified && !has_per_layer {
            return Err(RuntimeError::Compute(
                "decode_token_greedy requires GPU-resident weights".into(),
            ));
        }

        let hidden_dim = s.hidden_dim;
        let num_layers = s.num_layers;
        let num_heads = s.num_heads;
        let num_kv_heads = s.num_kv_heads;
        let head_dim = s.head_dim;
        let inter_dim = s.inter_dim;
        let eps = s.eps;
        let q_dim = s.q_dim;
        let kv_dim = s.kv_dim;
        let qkv_dim = s.qkv_dim;
        let attn_scale = s.attn_scale;
        let matmul_tg_size = s.matmul_tg_size;
        let norm_tg_size = s.norm_tg_size;
        let vocab_size = s.vocab_size;

        // ONE command buffer for embed + ALL layers + final projection + argmax.
        // Single CONCURRENT encoder for entire token. Uses
        // MTLDispatchTypeConcurrent to allow GPU overlap of non-dependent dispatches.
        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for greedy decode".into())
        })?;

        // --- Embed token into x_buf ---
        let (sc_embed_buf, sc_embed_off): (&MetalBuffer, u64) =
            if let Some((emb_o, _, _)) = s.gpu_global_offsets {
                (s.gpu_unified_weight_buf.as_ref().unwrap(), emb_o as u64)
            } else {
                (embedding_buf, 0u64)
            };
        // For pure dense models (no GDN, no MoE), use a serial encoder.
        // Dense decode is a strict dependency chain -- every dispatch reads the
        // previous dispatch's output. The concurrent encoder's overlap-tracking
        // metadata is pure overhead when no overlap is possible. Serial encoders
        // guarantee completion ordering: each dispatch finishes before the next
        // begins, making memory_barrier_with_scope calls unnecessary (skipped
        // for serial via the all_dense flag to reduce CPU-side encoding cost).
        // GDN/MoE models keep the concurrent encoder for overlap of independent
        // small dispatches.
        let all_dense = s.cached_layer_meta.iter().all(|m| {
            m.gdn_layer_idx.is_none() && m.moe_meta.is_none()
        });
        let needs_barriers = !all_dense;
        let mut enc = if all_dense {
            cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create serial encoder".into())
            })?
        } else {
            cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create concurrent encoder".into())
            })?
        };

        {
            match self.embedding_quant {
                QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.embed_token_q8_0),
                QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.embed_token_q4_0),
                QuantScheme::F16 => enc.set_pipeline_state(&pipelines.embed_token_f16),
                _ => enc.set_pipeline_state(&pipelines.embed_token),
            }
            enc.set_buffer(sc_embed_buf, sc_embed_off, 0);
            enc.set_buffer(&s.x_buf, 0, 1);
            enc.set_bytes(&token_id.to_le_bytes(), 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            let tg = 256u64.min(hidden_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                MTLSize::new(tg, 1, 1),
            );
            // Barrier: embed writes x_buf, layer 0 RMSNorm reads x_buf
            if needs_barriers { enc.memory_barrier_with_scope(1); }
        }

        // --- ALL layers ---

        for layer_idx in 0..num_layers {
            // Resolve layer buffer: prefer unified private buffer, then per-layer
            let layer_buf: &MetalBuffer;
            if let Some(ref ubuf) = s.gpu_unified_weight_buf {
                layer_buf = ubuf;
            } else {
                let gpu_layers = s.gpu_resident_layers.as_ref().unwrap();
                layer_buf = &gpu_layers[layer_idx];
            }
            // Use cached metadata (pre-computed absolute offsets + quant schemes).
            let meta = &s.cached_layer_meta[layer_idx];
            let attn_norm_off = meta.attn_norm_off;
            let wq_off = meta.wq_off;
            let wo_off = meta.wo_off;
            let ffn_norm_off = meta.ffn_norm_off;
            let w_gate_off = meta.w_gate_off;
            let w_up_off = meta.w_up_off;
            let w_down_off = meta.w_down_off;
            let new_seq_len = seq_pos + 1;
            let q_byte_off: u64 = 0;
            let k_byte_off: u64 = (q_dim * 4) as u64;
            let v_byte_off: u64 = ((q_dim + kv_dim) * 4) as u64;

            // Reuse the single concurrent encoder (no per-layer encoder creation).

            // ================================================================
            // ATTENTION BLOCK
            // ================================================================
            if meta.gdn_layer_idx.is_none() {
                // Standard softmax attention path

                // Fused RMSNorm + QKV Q8_0 matvec.
                // Eliminates 1 dispatch + 1 barrier + normed_buf write/read per layer.
                // Also works for Q+gate fusion: all 3 matmuls (Q+gate, K, V) fuse
                // RMSNorm inline, reading x_buf directly. Eliminates separate RMSNorm
                // dispatch + barrier, and allows K/V to dispatch in parallel with Q+gate.
                let use_fused_attn_norm = matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
                    && !(meta.bq_off.is_some() && meta.bk_off.is_some() && meta.bv_off.is_some())
                    && (!meta.has_qgate_fusion
                        || (matches!(meta.wk_quant, Some(QuantScheme::Q8_0) | Some(QuantScheme::Q4_0) | Some(QuantScheme::F16))
                            && matches!(meta.wv_quant, Some(QuantScheme::Q8_0) | Some(QuantScheme::Q4_0) | Some(QuantScheme::F16))));

                if use_fused_attn_norm && !meta.has_qgate_fusion {
                    // Fused RMSNorm + QKV matvec NR2: reads x_buf, applies inline
                    // normalization (x[i]*scale*norm_w[i]), writes qkv_buf directly.
                    // (Q+gate fusion handles its own fused matmuls below.)
                    match meta.wq_quant {
                        QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2),
                        QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                        QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                        _ => unreachable!(),
                    }
                    enc.set_buffer(layer_buf, wq_off, 0);
                    enc.set_buffer(&s.x_buf, 0, 1);
                    enc.set_buffer(&s.qkv_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
                    enc.set_buffer(layer_buf, attn_norm_off, 5);
                    enc.set_bytes(&eps.to_le_bytes(), 6);
                    let n_tg = ((qkv_dim as u64) + 1) / 2;
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                } else if !use_fused_attn_norm {
                // Non-fused: separate RMSNorm + QKV matvec
                // Attention RMSNorm
                enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
                enc.set_buffer(&s.x_buf, 0, 0);
                enc.set_buffer(layer_buf, attn_norm_off, 1);
                enc.set_buffer(&s.normed_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&eps.to_le_bytes(), 4);
                enc.dispatch_threadgroups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(norm_tg_size, 1, 1),
                );

                // Barrier: RMSNorm writes normed_buf, QKV matmul reads normed_buf
                if needs_barriers { enc.memory_barrier_with_scope(1); }
                }

                // QKV projection: two paths depending on Q+gate fusion.
                if meta.has_qgate_fusion {
                    // Q+gate fusion (Qwen3.5 full-attention layers).
                    // attn_q.weight output is interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...].
                    // K and V come from separate attn_k.weight / attn_v.weight.
                    // sigmoid(gate) applied to attention output BEFORE Wo projection.
                    let qgate_dim = q_dim * 2;
                    // When fused, Q+gate/K/V all fuse RMSNorm inline (read x_buf),
                    // run in parallel, then a single barrier before deinterleave.
                    // Saves 1 dispatch (RMSNorm) + 2 barriers per layer vs non-fused path.

                    // Project Q+gate into qkv_buf
                    {
                        if use_fused_attn_norm {
                            match meta.wq_quant {
                                QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                                QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                                _ => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2),
                            }
                            enc.set_buffer(layer_buf, wq_off, 0);
                            enc.set_buffer(&s.x_buf, 0, 1);
                            enc.set_buffer(&s.qkv_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(qgate_dim as u32).to_le_bytes(), 4);
                            enc.set_buffer(layer_buf, attn_norm_off, 5);
                            enc.set_bytes(&eps.to_le_bytes(), 6);
                        } else {
                            let _tg = match meta.wq_quant {
                                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                            };
                            enc.set_buffer(layer_buf, wq_off, 0);
                            enc.set_buffer(&s.normed_buf, 0, 1);
                            enc.set_buffer(&s.qkv_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            if matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                                enc.set_bytes(&(qgate_dim as u32).to_le_bytes(), 4);
                            }
                        }
                        let n_tg = match meta.wq_quant { QuantScheme::Q8_0 => ((qgate_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((qgate_dim as u64) + 1) / 2, QuantScheme::F16 => ((qgate_dim as u64) + 1) / 2, _ => qgate_dim as u64 };
                        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                    }
                    // Project K from wk (parallel with Q+gate when fused)
                    {
                        let wk_off_val = meta.wk_off.unwrap();
                        let wk_quant = meta.wk_quant.unwrap();
                        if use_fused_attn_norm {
                            match wk_quant {
                                QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                                QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                                _ => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2),
                            }
                            enc.set_buffer(layer_buf, wk_off_val, 0);
                            enc.set_buffer(&s.x_buf, 0, 1);
                            enc.set_buffer(&s.k_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                            enc.set_buffer(layer_buf, attn_norm_off, 5);
                            enc.set_bytes(&eps.to_le_bytes(), 6);
                        } else {
                            let _tg = match wk_quant {
                                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                            };
                            enc.set_buffer(layer_buf, wk_off_val, 0);
                            enc.set_buffer(&s.normed_buf, 0, 1);
                            enc.set_buffer(&s.k_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            if matches!(wk_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                                enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                            }
                        }
                        let n_tg = match wk_quant { QuantScheme::Q8_0 => ((kv_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((kv_dim as u64) + 1) / 2, QuantScheme::F16 => ((kv_dim as u64) + 1) / 2, _ => kv_dim as u64 };
                        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                    }
                    // Project V from wv (parallel with Q+gate and K when fused)
                    {
                        let wv_off_val = meta.wv_off.unwrap();
                        let wv_quant = meta.wv_quant.unwrap();
                        if use_fused_attn_norm {
                            match wv_quant {
                                QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                                QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                                _ => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2),
                            }
                            enc.set_buffer(layer_buf, wv_off_val, 0);
                            enc.set_buffer(&s.x_buf, 0, 1);
                            enc.set_buffer(&s.v_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                            enc.set_buffer(layer_buf, attn_norm_off, 5);
                            enc.set_bytes(&eps.to_le_bytes(), 6);
                        } else {
                            let _tg = match wv_quant {
                                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                            };
                            enc.set_buffer(layer_buf, wv_off_val, 0);
                            enc.set_buffer(&s.normed_buf, 0, 1);
                            enc.set_buffer(&s.v_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            if matches!(wv_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                                enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                            }
                        }
                        let n_tg = match wv_quant { QuantScheme::Q8_0 => ((kv_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((kv_dim as u64) + 1) / 2, QuantScheme::F16 => ((kv_dim as u64) + 1) / 2, _ => kv_dim as u64 };
                        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                    }
                    // Barrier: Q+gate/K/V projections all complete
                    if needs_barriers { enc.memory_barrier_with_scope(1); }

                    // Fused deinterleave + norm + assemble (saves 5 dispatches + 2 barriers per layer).
                    // Falls back to separate dispatches if fused kernel or norm weights unavailable.
                    let use_fused_dna = pipelines.deinterleave_norm_assemble.is_some()
                        && meta.attn_q_norm_off.is_some()
                        && meta.attn_k_norm_off.is_some();

                    if use_fused_dna {
                        let pso = pipelines.deinterleave_norm_assemble.as_ref().unwrap();
                        let q_norm_off = meta.attn_q_norm_off.unwrap();
                        let k_norm_off = meta.attn_k_norm_off.unwrap();
                        enc.set_pipeline_state(pso);
                        enc.set_buffer(&s.qkv_buf, 0, 0);            // qgate_interleaved (input)
                        enc.set_buffer(&s.k_buf, 0, 1);              // k_data
                        enc.set_buffer(&s.v_buf, 0, 2);              // v_data
                        enc.set_buffer(layer_buf, q_norm_off, 3);     // q_norm_weight
                        enc.set_buffer(layer_buf, k_norm_off, 4);     // k_norm_weight
                        enc.set_buffer(&s.qkv_buf, 0, 5);            // qkv_out (K/V assembled here)
                        enc.set_buffer(&s.gate_buf, 0, 6);            // gate_out
                        enc.set_bytes(&(num_heads as u32).to_le_bytes(), 7);
                        enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 8);
                        enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 10);
                        enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 11);
                        enc.set_bytes(&eps.to_le_bytes(), 12);
                        enc.set_buffer(&s.q_buf, 0, 13);             // q_out (separate to avoid aliasing)
                        let total_tgs = (num_heads + num_kv_heads) as u64;
                        let tg_threads = 256u64.min(head_dim as u64).max(32);
                        enc.dispatch_threadgroups(
                            MTLSize::new(total_tgs, 1, 1),
                            MTLSize::new(tg_threads, 1, 1),
                        );
                        // Copy normalized Q from q_buf to qkv_buf[0..q_dim]
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                        enc.set_pipeline_state(&pipelines.copy_buffer);
                        enc.set_buffer(&s.q_buf, 0, 0);
                        enc.set_buffer(&s.qkv_buf, 0, 1);
                        {
                            let tg = 256u64.min(q_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((q_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                    } else {
                        // Fallback: separate deinterleave + norm + copy
                        {
                            let pso = pipelines.deinterleave_qgate.as_ref().ok_or_else(|| {
                                RuntimeError::Compute("deinterleave_qgate pipeline not compiled".into())
                            })?;
                            enc.set_pipeline_state(pso);
                            enc.set_buffer(&s.qkv_buf, 0, 0);
                            enc.set_buffer(&s.q_buf, 0, 1);
                            enc.set_buffer(&s.gate_buf, 0, 2);
                            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);
                            let tg_di = 256u64.min(q_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((q_dim as u64).div_ceil(tg_di), 1, 1),
                                MTLSize::new(tg_di, 1, 1),
                            );
                        }
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                        if let (Some(q_norm_off), Some(k_norm_off)) = (meta.attn_q_norm_off, meta.attn_k_norm_off) {
                            let pso = pipelines.rmsnorm_per_head.as_ref().ok_or_else(|| {
                                RuntimeError::Compute("rmsnorm_per_head pipeline not compiled".into())
                            })?;
                            let head_dim_u32 = head_dim as u32;
                            let tg_rms = 256u64.min(head_dim as u64).max(32);
                            enc.set_pipeline_state(pso);
                            enc.set_buffer(&s.q_buf, 0, 0);
                            enc.set_buffer(layer_buf, q_norm_off, 1);
                            enc.set_buffer(&s.q_buf, 0, 2);
                            enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                            enc.set_bytes(&eps.to_le_bytes(), 4);
                            enc.dispatch_threadgroups(
                                MTLSize::new(num_heads as u64, 1, 1),
                                MTLSize::new(tg_rms, 1, 1),
                            );
                            enc.set_pipeline_state(pso);
                            enc.set_buffer(&s.k_buf, 0, 0);
                            enc.set_buffer(layer_buf, k_norm_off, 1);
                            enc.set_buffer(&s.k_buf, 0, 2);
                            enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                            enc.set_bytes(&eps.to_le_bytes(), 4);
                            enc.dispatch_threadgroups(
                                MTLSize::new(num_kv_heads as u64, 1, 1),
                                MTLSize::new(tg_rms, 1, 1),
                            );
                        }
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                        enc.set_pipeline_state(&pipelines.copy_buffer);
                        enc.set_buffer(&s.q_buf, 0, 0);
                        enc.set_buffer(&s.qkv_buf, 0, 1);
                        {
                            let tg = 256u64.min(q_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((q_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                        enc.set_buffer(&s.k_buf, 0, 0);
                        enc.set_buffer(&s.qkv_buf, k_byte_off, 1);
                        {
                            let tg = 256u64.min(kv_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((kv_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                        enc.set_buffer(&s.v_buf, 0, 0);
                        enc.set_buffer(&s.qkv_buf, v_byte_off, 1);
                        {
                            let tg = 256u64.min(kv_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((kv_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                    }
                    // gate_buf holds pre-sigmoid gate [q_dim], applied after attention.
                } else if !use_fused_attn_norm {
                // Fused QKV projection (+ fused bias for Qwen2-family models)
                // Skipped when fused RMSNorm+QKV already wrote qkv_buf.
                {
                    let has_bias = meta.bq_off.is_some() && meta.bk_off.is_some() && meta.bv_off.is_some();
                    let tg = if has_bias && matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                        match meta.wq_quant {
                            QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_bias_nr2),
                            QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_bias_nr2),
                            QuantScheme::F16 => enc.set_pipeline_state(&pipelines.matmul_f16_deferred_bias_nr2),
                            _ => unreachable!(),
                        };
                        128u64
                    } else {
                        match meta.wq_quant {
                            QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                            QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                            QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                            _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                        }
                    };
                    enc.set_buffer(layer_buf, wq_off, 0);
                    enc.set_buffer(&s.normed_buf, 0, 1);
                    enc.set_buffer(&s.qkv_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    if matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                        enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
                    }
                    if has_bias && matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                        enc.set_buffer(layer_buf, meta.bq_off.unwrap(), 5);
                        enc.set_buffer(layer_buf, meta.bk_off.unwrap(), 6);
                        enc.set_buffer(layer_buf, meta.bv_off.unwrap(), 7);
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 8);
                        let qk_dim = (q_dim + kv_dim) as u32;
                        enc.set_bytes(&qk_dim.to_le_bytes(), 9);
                    }
                    let n_tg = if tg == 64 {
                        ((qkv_dim as u64) + 7) / 8  // (dead path: Q8_0 now uses deferred with tg=128)
                    } else {
                        match meta.wq_quant { QuantScheme::Q8_0 => ((qkv_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((qkv_dim as u64) + 1) / 2, QuantScheme::F16 => ((qkv_dim as u64) + 1) / 2, _ => qkv_dim as u64 }
                    };
                    enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
                }

                // QKV bias addition fallback (only for F32 weights with bias, rare)
                if !matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
                    && (meta.bq_off.is_some() || meta.bk_off.is_some() || meta.bv_off.is_some())
                {
                    enc.set_pipeline_state(&pipelines.bias_add);
                    if let Some(bq_off) = meta.bq_off {
                        enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                        enc.set_buffer(layer_buf, bq_off, 1);
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 2);
                        let n_tg_bq = (q_dim as u64 + 255) / 256;
                        enc.dispatch_threadgroups(MTLSize::new(n_tg_bq, 1, 1), MTLSize::new(256, 1, 1));
                    }
                    if let Some(bk_off) = meta.bk_off {
                        enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
                        enc.set_buffer(layer_buf, bk_off, 1);
                        enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 2);
                        let n_tg_bk = (kv_dim as u64 + 255) / 256;
                        enc.dispatch_threadgroups(MTLSize::new(n_tg_bk, 1, 1), MTLSize::new(256, 1, 1));
                    }
                    if let Some(bv_off) = meta.bv_off {
                        enc.set_buffer(&s.qkv_buf, v_byte_off, 0);
                        enc.set_buffer(layer_buf, bv_off, 1);
                        enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 2);
                        let n_tg_bv = (kv_dim as u64 + 255) / 256;
                        enc.dispatch_threadgroups(MTLSize::new(n_tg_bv, 1, 1), MTLSize::new(256, 1, 1));
                    }
                }
                }

                // Barrier: QKV projection writes qkv_buf, RoPE reads qkv_buf
                if needs_barriers { enc.memory_barrier_with_scope(1); }

                // Fused RoPE Q + RoPE K + KV cache write (1 dispatch instead of 3)
                // Only used for full RoPE (rotary_dim == head_dim) on non-linear attention layers.
                // Partial RoPE (Qwen3.5-MoE) and linear attention layers fall back to separate dispatches.
                let is_linear_attn = meta.layer_type == Some(1);
                let rope_half_dim = s.rotary_dim / 2;
                let use_fused_rope_kv = !is_linear_attn && s.rotary_dim == head_dim;
                const FLASH_DECODE_THRESHOLD: usize = 257; // FLASH_DECODE_TILE_SIZE + 1: single-tile flash_decode is a no-op reduce

                // Fused RoPE + KV cache write + MHA (eliminates 2 barriers per layer)
                // Only for: standard RoPE (not NeoX), short sequences, full rotary_dim
                let use_fused_rope_kv_mha = use_fused_rope_kv && !s.is_qwen35moe && new_seq_len < FLASH_DECODE_THRESHOLD;

                if use_fused_rope_kv_mha {
                    // Single dispatch: RoPE Q/K + KV cache write + MHA
                    let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
                    enc.set_pipeline_state(&pipelines.fused_rope_kv_mha);
                    enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                    enc.set_buffer(&s.qkv_buf, k_byte_off, 1);
                    enc.set_buffer(&s.qkv_buf, v_byte_off, 2);
                    enc.set_buffer(&s.rope_cos_buf, 0, 3);
                    enc.set_buffer(&s.rope_sin_buf, 0, 4);
                    enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 5);
                    enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 6);
                    enc.set_buffer(&s.attn_out_buf, 0, 7);
                    enc.set_buffer(&s.mha_scores_buf, 0, 8);
                    enc.set_bytes(&(num_heads as u32).to_le_bytes(), 9);
                    enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 10);
                    enc.set_bytes(&(head_dim as u32).to_le_bytes(), 11);
                    enc.set_bytes(&(rope_half_dim as u32).to_le_bytes(), 12);
                    enc.set_bytes(&pos_offset_u32.to_le_bytes(), 13);
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 14);
                    enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 15);
                    enc.set_bytes(&attn_scale.to_le_bytes(), 16);
                    enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 17);
                    let tg_threads = 256u64.min((head_dim.max(new_seq_len) as u64).max(32));
                    enc.dispatch_threadgroups(
                        MTLSize::new(num_heads as u64, 1, 1),
                        MTLSize::new(tg_threads, 1, 1),
                    );
                } else {

                // Fused RoPE Q + RoPE K + KV cache write (1 dispatch instead of 3)
                // Only used for full RoPE (rotary_dim == head_dim) on non-linear attention layers.
                // Partial RoPE (Qwen3.5-MoE) and linear attention layers fall back to separate dispatches.
                let is_linear_attn = meta.layer_type == Some(1);
                let rope_half_dim = s.rotary_dim / 2;
                let use_fused_rope_kv = !is_linear_attn && s.rotary_dim == head_dim;
                if use_fused_rope_kv {
                    let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
                    let fused_pipe = if s.is_qwen35moe {
                        pipelines.fused_rope_neox_kv_write.as_ref().unwrap_or(&pipelines.fused_rope_kv_write)
                    } else {
                        &pipelines.fused_rope_kv_write
                    };
                    enc.set_pipeline_state(fused_pipe);
                    enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                    enc.set_buffer(&s.qkv_buf, k_byte_off, 1);
                    enc.set_buffer(&s.qkv_buf, v_byte_off, 2);
                    enc.set_buffer(&s.rope_cos_buf, 0, 3);
                    enc.set_buffer(&s.rope_sin_buf, 0, 4);
                    enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 5);
                    enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 6);
                    enc.set_bytes(&(num_heads as u32).to_le_bytes(), 7);
                    enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 8);
                    enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);
                    enc.set_bytes(&(rope_half_dim as u32).to_le_bytes(), 10);
                    enc.set_bytes(&pos_offset_u32.to_le_bytes(), 11);
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 12);
                    enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 13);
                    enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 14);
                    let total_threads = (num_heads * rope_half_dim + num_kv_heads * rope_half_dim + kv_dim) as u64;
                    let tg = 64u64.min(total_threads.max(1));
                    enc.dispatch_threadgroups(
                        MTLSize::new(total_threads.div_ceil(tg), 1, 1),
                        MTLSize::new(tg, 1, 1),
                    );
                } else {
                    if !is_linear_attn {
                        let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
                        let rope_pipe = if s.is_qwen35moe {
                            pipelines.rope_neox.as_ref().unwrap_or(&pipelines.rope)
                        } else {
                            &pipelines.rope
                        };
                        enc.set_pipeline_state(rope_pipe);
                        enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                        enc.set_buffer(&s.rope_cos_buf, 0, 1);
                        enc.set_buffer(&s.rope_sin_buf, 0, 2);
                        enc.set_bytes(&(num_heads as u32).to_le_bytes(), 3);
                        enc.set_bytes(&(head_dim as u32).to_le_bytes(), 4);
                        enc.set_bytes(&(rope_half_dim as u32).to_le_bytes(), 5);
                        enc.set_bytes(&pos_offset_u32.to_le_bytes(), 6);
                        let q_total_half = (num_heads * rope_half_dim) as u64;
                        let tg_q = 64u64.min(q_total_half.max(1));
                        enc.dispatch_threadgroups(
                            MTLSize::new(q_total_half.div_ceil(tg_q), 1, 1),
                            MTLSize::new(tg_q, 1, 1),
                        );
                        enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
                        enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 3);
                        let k_total_half = (num_kv_heads * rope_half_dim) as u64;
                        let tg_k = 64u64.min(k_total_half.max(1));
                        enc.dispatch_threadgroups(
                            MTLSize::new(k_total_half.div_ceil(tg_k), 1, 1),
                            MTLSize::new(tg_k, 1, 1),
                        );
                    }
                    enc.set_pipeline_state(&pipelines.write_kv_cache);
                    enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
                    enc.set_buffer(&s.qkv_buf, v_byte_off, 1);
                    enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 2);
                    enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 3);
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                    enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 5);
                    enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 6);
                    {
                        let tg = 64u64.min(kv_dim as u64).max(1);
                        enc.dispatch_threadgroups(
                            MTLSize::new((kv_dim as u64).div_ceil(tg), 1, 1),
                            MTLSize::new(tg, 1, 1),
                        );
                    }
                }

                // Barrier: RoPE+KV cache write complete, attention reads KV cache + qkv_buf Q
                if needs_barriers { enc.memory_barrier_with_scope(1); }
                // Attention (flash decode or MHA)
                {
                    let num_heads_u32 = num_heads as u32;
                    let num_kv_heads_u32 = num_kv_heads as u32;
                    let head_dim_u32 = head_dim as u32;
                    let kv_dim_u32 = kv_dim as u32;
                    let seq_len_u32 = new_seq_len as u32;
                    let max_seq_len_u32 = s.max_seq_len as u32;
                    const FLASH_DECODE_TILE_SIZE: u32 = 256;
                    const FLASH_DECODE_THRESHOLD: usize = FLASH_DECODE_TILE_SIZE as usize + 1; // 257: single-tile is a no-op reduce

                    if new_seq_len >= FLASH_DECODE_THRESHOLD {
                        let num_tiles = ((new_seq_len as u32) + FLASH_DECODE_TILE_SIZE - 1) / FLASH_DECODE_TILE_SIZE;
                        enc.set_pipeline_state(&pipelines.flash_decode_attention);
                        enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                        enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 1);
                        enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 2);
                        enc.set_buffer(&s.flash_decode_partial_buf, 0, 3);
                        enc.set_bytes(&num_heads_u32.to_le_bytes(), 4);
                        enc.set_bytes(&num_kv_heads_u32.to_le_bytes(), 5);
                        enc.set_bytes(&head_dim_u32.to_le_bytes(), 6);
                        enc.set_bytes(&kv_dim_u32.to_le_bytes(), 7);
                        enc.set_bytes(&seq_len_u32.to_le_bytes(), 8);
                        enc.set_bytes(&attn_scale.to_le_bytes(), 9);
                        enc.set_bytes(&FLASH_DECODE_TILE_SIZE.to_le_bytes(), 10);
                        enc.set_bytes(&num_tiles.to_le_bytes(), 11);
                        enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 12);
                        enc.dispatch_threadgroups(
                            MTLSize::new((num_heads as u64) * (num_tiles as u64), 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                        // Barrier: flash_decode writes partial_buf, reduce reads partial_buf
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                        enc.set_pipeline_state(&pipelines.flash_decode_reduce);
                        enc.set_buffer(&s.flash_decode_partial_buf, 0, 0);
                        enc.set_buffer(&s.attn_out_buf, 0, 1);
                        enc.set_bytes(&num_heads_u32.to_le_bytes(), 2);
                        enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                        enc.set_bytes(&num_tiles.to_le_bytes(), 4);
                        let tg_threads = (head_dim as u64).max(1).min(256);
                        enc.dispatch_threadgroups(
                            MTLSize::new(num_heads as u64, 1, 1),
                            MTLSize::new(tg_threads, 1, 1),
                        );
                    } else {
                        let mha_tg_size = s.mha_tg_size;
                        enc.set_pipeline_state(&pipelines.multi_head_attention);
                        enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                        enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 1);
                        enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 2);
                        enc.set_buffer(&s.attn_out_buf, 0, 3);
                        enc.set_buffer(&s.mha_scores_buf, 0, 4);
                        enc.set_bytes(&num_heads_u32.to_le_bytes(), 5);
                        enc.set_bytes(&num_kv_heads_u32.to_le_bytes(), 6);
                        enc.set_bytes(&head_dim_u32.to_le_bytes(), 7);
                        enc.set_bytes(&kv_dim_u32.to_le_bytes(), 8);
                        enc.set_bytes(&seq_len_u32.to_le_bytes(), 9);
                        enc.set_bytes(&attn_scale.to_le_bytes(), 10);
                        enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 11);
                        let tg_threads = mha_tg_size.min((head_dim.max(new_seq_len) as u64).max(1));
                        enc.dispatch_threadgroups(
                            MTLSize::new(num_heads as u64, 1, 1),
                            MTLSize::new(tg_threads, 1, 1),
                        );
                    }
                }

                } // end fallback (non-fused RoPE+KV+MHA)

                // Barrier: attention writes attn_out_buf, Wo reads attn_out_buf
                if needs_barriers { enc.memory_barrier_with_scope(1); }
                // Wo projection + Residual
                let has_attn_extras = meta.attn_post_norm_off.is_some() || meta.attn_gate_off.is_some() || meta.has_qgate_fusion;
                if has_attn_extras {
                    // Apply sigmoid(gate) * attn_out BEFORE Wo (Q+gate fusion).
                    if meta.has_qgate_fusion {
                        let pso = pipelines.sigmoid_mul_fused.as_ref().ok_or_else(|| {
                            RuntimeError::Compute("sigmoid_mul_fused pipeline not compiled".into())
                        })?;
                        enc.set_pipeline_state(pso);
                        enc.set_buffer(&s.gate_buf, 0, 0);     // gate [q_dim]
                        enc.set_buffer(&s.attn_out_buf, 0, 1);  // attn output [q_dim]
                        enc.set_buffer(&s.attn_out_buf, 0, 2);  // output (in-place)
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                        let tg = 256u64.min(q_dim as u64).max(1);
                        enc.dispatch_threadgroups(
                            MTLSize::new((q_dim as u64).div_ceil(tg), 1, 1),
                            MTLSize::new(tg, 1, 1),
                        );
                    }

                    // Barrier: sigmoid_mul writes attn_out_buf, Wo reads it
                    if needs_barriers { enc.memory_barrier_with_scope(1); }
                    // Non-fused Wo: attn_proj_buf = Wo * attn_out (NO residual)
                    {
                        let tg_wo = match meta.wo_quant {
                            QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                            QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                            QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                            _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                        };
                        enc.set_buffer(layer_buf, wo_off, 0);
                        enc.set_buffer(&s.attn_out_buf, 0, 1);
                        enc.set_buffer(&s.attn_proj_buf, 0, 2);
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                        if matches!(meta.wo_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                        }
                        let n_tg_wo = match meta.wo_quant { QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2, _ => hidden_dim as u64 };
                        enc.dispatch_threadgroups(MTLSize::new(n_tg_wo, 1, 1), MTLSize::new(tg_wo, 1, 1));
                    }
                    // Barrier: Wo writes attn_proj_buf
                    if needs_barriers { enc.memory_barrier_with_scope(1); }
                    // Post-attention RMSNorm: only for architectures that have
                    // BOTH attn_post_norm AND attn_gate (not Q+gate fusion).
                    // For Qwen3.5 Q+gate fusion, post_attention_norm is the
                    // pre-FFN norm (via ffn_norm_off) — must not be applied here.
                    let did_post_norm = meta.attn_gate_off.is_some() && meta.attn_post_norm_off.is_some();
                    if let (true, Some(post_norm_off)) = (did_post_norm, meta.attn_post_norm_off) {
                        enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
                        enc.set_buffer(&s.attn_proj_buf, 0, 0);
                        enc.set_buffer(layer_buf, post_norm_off, 1);
                        enc.set_buffer(&s.down_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        enc.set_bytes(&eps.to_le_bytes(), 4);
                        enc.dispatch_threadgroups(
                            MTLSize::new(1, 1, 1),
                            MTLSize::new(norm_tg_size, 1, 1),
                        );
                    }
                    // Barrier: only needed when post_norm dispatched (writes down_buf
                    // for subsequent gate matmul). When !did_post_norm, the Wo barrier
                    // above already covers attn_proj_buf visibility.
                    if did_post_norm {
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                    }
                    // Attention output gate
                    if let Some(gate_off) = meta.attn_gate_off {
                        let gate_quant = meta.attn_gate_quant.unwrap_or(QuantScheme::F32);
                        let src_buf = if did_post_norm { &s.down_buf } else { &s.attn_proj_buf };
                        let attn_gate_buf = s.attn_gate_buf.as_ref().ok_or_else(|| {
                            RuntimeError::Compute("attn_gate_buf not allocated".into())
                        })?;
                        // Gate matmul
                        {
                            let tg_gate = match gate_quant {
                                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                            };
                            enc.set_buffer(layer_buf, gate_off, 0);
                            enc.set_buffer(src_buf, 0, 1);
                            enc.set_buffer(attn_gate_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            if matches!(gate_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                            }
                            let n_tg_gate = match gate_quant { QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2, _ => hidden_dim as u64 };
                            enc.dispatch_threadgroups(MTLSize::new(n_tg_gate, 1, 1), MTLSize::new(tg_gate, 1, 1));
                        }
                        // Barrier: gate matmul writes attn_gate_buf, SwiGLU reads it
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                        // SwiGLU gate
                        enc.set_pipeline_state(&pipelines.swiglu);
                        enc.set_buffer(attn_gate_buf, 0, 0);
                        enc.set_buffer(src_buf, 0, 1);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
                        {
                            let tg = 256u64.min(hidden_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                        // Barrier: SwiGLU writes attn_gate_buf, residual reads it
                        if needs_barriers { enc.memory_barrier_with_scope(1); }
                        // Fused residual + copy (saves 1 dispatch + 1 barrier)
                        // x_buf += attn_gate_buf; attn_proj_buf = x_buf
                        {
                            let pso = pipelines.residual_add_copy.as_ref().unwrap_or(&pipelines.add_residual);
                            if pipelines.residual_add_copy.is_some() {
                                enc.set_pipeline_state(pso);
                                enc.set_buffer(&s.x_buf, 0, 0);
                                enc.set_buffer(attn_gate_buf, 0, 1);
                                enc.set_buffer(&s.attn_proj_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                let tg = 256u64.min(hidden_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                                    MTLSize::new(tg, 1, 1),
                                );
                            } else {
                                // Fallback: separate add + barrier + copy
                                enc.set_pipeline_state(&pipelines.add_residual);
                                enc.set_buffer(&s.x_buf, 0, 0);
                                enc.set_buffer(attn_gate_buf, 0, 1);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
                                let tg = 256u64.min(hidden_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                                    MTLSize::new(tg, 1, 1),
                                );
                                if needs_barriers { enc.memory_barrier_with_scope(1); }
                                enc.set_pipeline_state(&pipelines.copy_buffer);
                                enc.set_buffer(&s.x_buf, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                let tg2 = 256u64.min(hidden_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((hidden_dim as u64).div_ceil(tg2), 1, 1),
                                    MTLSize::new(tg2, 1, 1),
                                );
                            }
                        }
                    } else {
                        // No attn_gate: fused residual + copy
                        // x_buf += attn_proj_buf; attn_proj_buf = x_buf
                        {
                            let pso = pipelines.residual_add_copy.as_ref().unwrap_or(&pipelines.add_residual);
                            if pipelines.residual_add_copy.is_some() {
                                enc.set_pipeline_state(pso);
                                enc.set_buffer(&s.x_buf, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                enc.set_buffer(&s.attn_proj_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                let tg = 256u64.min(hidden_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                                    MTLSize::new(tg, 1, 1),
                                );
                            } else {
                                // Fallback: separate add + barrier + copy
                                enc.set_pipeline_state(&pipelines.add_residual);
                                enc.set_buffer(&s.x_buf, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
                                let tg = 256u64.min(hidden_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                                    MTLSize::new(tg, 1, 1),
                                );
                                if needs_barriers { enc.memory_barrier_with_scope(1); }
                                enc.set_pipeline_state(&pipelines.copy_buffer);
                                enc.set_buffer(&s.x_buf, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                let tg2 = 256u64.min(hidden_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((hidden_dim as u64).div_ceil(tg2), 1, 1),
                                    MTLSize::new(tg2, 1, 1),
                                );
                            }
                        }
                    }
                } else {
                    // Standard fused Wo + Residual path
                    let tg_wo = match meta.wo_quant {
                        QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_residual_nr2); 128u64 },
                        QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_residual_nr2); 128u64 },
                        QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_residual_nr2); 128u64 },
                        _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32_residual); matmul_tg_size },
                    };
                    enc.set_buffer(layer_buf, wo_off, 0);
                    enc.set_buffer(&s.attn_out_buf, 0, 1);
                    enc.set_buffer(&s.attn_proj_buf, 0, 2);
                    enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                    enc.set_buffer(&s.x_buf, 0, 4);
                    if matches!(meta.wo_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
                    }
                    let n_tg_wo = match meta.wo_quant { QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2, _ => hidden_dim as u64 };
                    enc.dispatch_threadgroups(MTLSize::new(n_tg_wo, 1, 1), MTLSize::new(tg_wo, 1, 1));
                }
            } else {
                // GatedDeltaNet layer: linear attention forward pass
                // Fused variant: all GDN dispatches go through the layer encoder.
                let gdn_idx = meta.gdn_layer_idx.unwrap();
                let new_conv_pos = Self::encode_gdn_layer_decode_fused(
                    &enc, pipelines, s, layer_buf, meta, gdn_idx,
                )?;
                s.gdn_conv_positions[gdn_idx] = new_conv_pos;
            }

            // ================================================================
            // FFN BLOCK
            // ================================================================

            // Fused RMSNorm + FFN for dense Q8_0 gate+up.
            // Eliminates FFN RMSNorm dispatch + 1 barrier + normed_buf write/read.
            let use_fused_ffn_norm = meta.moe_meta.is_none()
                && matches!(meta.w_gate_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
                && meta.w_gate_quant == meta.w_up_quant;

            if !use_fused_ffn_norm {
            // Non-fused path: separate FFN RMSNorm + barrier
            // Barrier: Wo+residual writes attn_proj_buf, FFN RMSNorm reads it
            if needs_barriers { enc.memory_barrier_with_scope(1); }
            // FFN RMSNorm
            enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
            enc.set_buffer(&s.attn_proj_buf, 0, 0);
            enc.set_buffer(layer_buf, ffn_norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );

            // Barrier: FFN RMSNorm writes normed_buf, gate+up reads normed_buf
            if needs_barriers { enc.memory_barrier_with_scope(1); }
            }

            // FFN block: branch on MoE vs dense
            if let Some(ref moe_meta) = meta.moe_meta {
                // use_batched no longer requires use_option_a. The batched
                // MoE path uses 2 kernel dispatches within the current encoder instead
                // of ~258 separate encoders (legacy path). The legacy path created
                // ~10,360 encoders on a single command buffer for 40-layer MoE models,
                // which caused Metal to produce corrupted router output buffers.
                let has_down_kernel = match moe_meta.expert_down_quant {
                    QuantScheme::Q4_1 => pipelines.moe_batched_down_accum_q4_1.is_some(),
                    QuantScheme::Q4_0 => pipelines.moe_batched_down_accum_q4_0.is_some(),
                    QuantScheme::Q8_0 => pipelines.moe_batched_down_accum_q8_0.is_some(),
                    _ => false,
                };
                let has_gate_kernel = match moe_meta.expert_gate_quant {
                    QuantScheme::Q4_0 => pipelines.moe_batched_gate_up_swiglu_q4_0.is_some(),
                    QuantScheme::Q4_1 => pipelines.moe_batched_gate_up_swiglu_q4_1.is_some(),
                    QuantScheme::Q8_0 => pipelines.moe_batched_gate_up_swiglu_q8_0.is_some(),
                    _ => false,
                };
                let use_batched = has_gate_kernel
                    && has_down_kernel
                    && s.moe_gate_up_offsets.get(layer_idx).and_then(|o| o.as_ref()).is_some()
                    && s.moe_down_offsets.get(layer_idx).and_then(|o| o.as_ref()).is_some()
                    && s.moe_batched_swiglu_buf.is_some();

                if use_batched {
                    let per_layer_ids_buf = s.moe_per_layer_expert_ids
                        .get(layer_idx)
                        .and_then(|opt| opt.as_ref());
                    let expert_ids_buf = per_layer_ids_buf.unwrap_or_else(|| {
                        s.moe_expert_ids.as_ref().unwrap()
                    });
                    let expert_weights_buf = s.moe_expert_weights.as_ref().unwrap();
                    let gate_up_off_buf = s.moe_gate_up_offsets[layer_idx].as_ref().unwrap();
                    let down_off_buf = s.moe_down_offsets[layer_idx].as_ref().unwrap();

                    // Router dispatch (all within same encoder)
                    {
                        let router_softmax = pipelines.moe_router_softmax.as_ref().ok_or_else(|| {
                            RuntimeError::Compute("MoE router_softmax pipeline not compiled.".into())
                        })?;
                        enc.set_pipeline_state(router_softmax);
                        enc.set_buffer(&s.normed_buf, 0, 0);
                        enc.set_buffer(layer_buf, moe_meta.router_weight_off, 1);
                        enc.set_buffer(expert_ids_buf, 0, 2);
                        enc.set_buffer(expert_weights_buf, 0, 3);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                        enc.set_bytes(&(s.moe_num_experts as u32).to_le_bytes(), 5);
                        enc.set_bytes(&(s.moe_num_active_experts as u32).to_le_bytes(), 6);
                        let tg = 256u64.min(hidden_dim as u64).max(1);
                        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
                    }

                    // Barrier: router writes expert_ids/weights, batched FFN reads them
                    if needs_barriers { enc.memory_barrier_with_scope(1); }
                    // Batched expert FFN + shared expert (fused when available)
                    Self::encode_moe_ffn_with_shared_fused(
                        &enc, pipelines, s, layer_buf, layer_idx,
                        moe_meta, meta,
                        expert_ids_buf, expert_weights_buf,
                        gate_up_off_buf, down_off_buf,
                    )?;
                } else {
                    // Legacy path: per-expert dispatch (needs its own encoders).
                    // End current encoder, run legacy, then this is the last thing
                    // in the layer so we skip re-opening.
                    enc.end_encoding();
                    let per_layer_ids_buf = s.moe_per_layer_expert_ids
                        .get(layer_idx)
                        .and_then(|opt| opt.as_ref());
                    let per_layer_wts_buf = s.moe_per_layer_expert_weights
                        .get(layer_idx)
                        .and_then(|opt| opt.as_ref());
                    Self::encode_moe_ffn_decode(
                        &cmd, pipelines, s, layer_buf, moe_meta,
                        per_layer_ids_buf,
                        None,
                        None, 0.0,
                        None,
                        false,
                        per_layer_wts_buf,
                    )?;
                    // Shared expert dispatch (legacy path)
                    if meta.shared_expert_gate_off.is_some() {
                        Self::encode_shared_expert_ffn_decode(
                            &cmd, pipelines, s, layer_buf, meta,
                        )?;
                    }
                    // Re-create concurrent encoder for remaining layers
                    enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to re-create concurrent encoder".into())
                    })?;
                    continue; // encoder already ended, skip end_encoding below
                }
            } else {
                // Dense FFN path
                if use_fused_ffn_norm {
                    // Fused RMSNorm + FFN gate+up+SwiGLU.
                    // Reads attn_proj_buf directly, applies inline normalization.
                    // Barrier: Wo+residual writes attn_proj_buf, fused FFN reads it
                    if needs_barriers { enc.memory_barrier_with_scope(1); }
                    if matches!(meta.w_gate_quant, QuantScheme::F16) {
                        // F16: always use deferred 1-row-per-TG pattern (no block structure)
                        enc.set_pipeline_state(&pipelines.rmsnorm_ffn_fused_gate_up_swiglu_f16_deferred);
                        enc.set_buffer(layer_buf, w_gate_off, 0);
                        enc.set_buffer(&s.attn_proj_buf, 0, 1);
                        enc.set_buffer(&s.gate_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                        enc.set_buffer(layer_buf, w_up_off, 5);
                        enc.set_buffer(layer_buf, ffn_norm_off, 6);
                        enc.set_bytes(&eps.to_le_bytes(), 7);
                        enc.dispatch_threadgroups(
                            MTLSize::new(inter_dim as u64, 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                    // For small hidden_dim (x fits in L1 cache), use 8-row
                    // pattern: 8 SGs independently own 1 row each, zero TG barriers.
                    } else if hidden_dim <= 4096 {
                        match meta.w_gate_quant {
                            QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_ffn_fused_gate_up_swiglu_q4_0_8row),
                            _ => enc.set_pipeline_state(&pipelines.rmsnorm_ffn_fused_gate_up_swiglu_q8_0_8row),
                        }
                        enc.set_buffer(layer_buf, w_gate_off, 0);
                        enc.set_buffer(&s.attn_proj_buf, 0, 1);
                        enc.set_buffer(&s.gate_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                        enc.set_buffer(layer_buf, w_up_off, 5);
                        enc.set_buffer(layer_buf, ffn_norm_off, 6);
                        enc.set_bytes(&eps.to_le_bytes(), 7);
                        enc.dispatch_threadgroups(
                            MTLSize::new(((inter_dim as u64) + 7) / 8, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                    } else {
                        match meta.w_gate_quant {
                            QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_ffn_fused_gate_up_swiglu_q4_0_deferred),
                            _ => enc.set_pipeline_state(&pipelines.rmsnorm_ffn_fused_gate_up_swiglu_q8_0_deferred),
                        }
                        enc.set_buffer(layer_buf, w_gate_off, 0);
                        enc.set_buffer(&s.attn_proj_buf, 0, 1);
                        enc.set_buffer(&s.gate_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                        enc.set_buffer(layer_buf, w_up_off, 5);
                        enc.set_buffer(layer_buf, ffn_norm_off, 6);
                        enc.set_bytes(&eps.to_le_bytes(), 7);
                        enc.dispatch_threadgroups(
                            MTLSize::new(inter_dim as u64, 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                    }
                } else if meta.w_gate_quant == QuantScheme::Q8_0 && meta.w_up_quant == QuantScheme::Q8_0 {
                    enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q8_0_deferred);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(&s.normed_buf, 0, 1);
                    enc.set_buffer(&s.gate_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                    enc.set_buffer(layer_buf, w_up_off, 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new(inter_dim as u64, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else if matches!(meta.w_gate_quant, QuantScheme::Q4_0) {
                    enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q4_0_deferred);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(&s.normed_buf, 0, 1);
                    enc.set_buffer(&s.gate_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                    enc.set_buffer(layer_buf, w_up_off, 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new(inter_dim as u64, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else if matches!(meta.w_gate_quant, QuantScheme::F16) {
                    // F16 fused gate+up+SwiGLU (non-norm path -- norm already applied)
                    enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_f16_deferred);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(&s.normed_buf, 0, 1);
                    enc.set_buffer(&s.gate_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                    enc.set_buffer(layer_buf, w_up_off, 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new(inter_dim as u64, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else {
                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(&s.normed_buf, 0, 1);
                    enc.set_buffer(&s.gate_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.dispatch_threadgroups(
                        MTLSize::new(inter_dim as u64, 1, 1),
                        MTLSize::new(matmul_tg_size, 1, 1),
                    );
                    enc.set_buffer(layer_buf, w_up_off, 0);
                    enc.set_buffer(&s.up_buf, 0, 2);
                    enc.dispatch_threadgroups(
                        MTLSize::new(inter_dim as u64, 1, 1),
                        MTLSize::new(matmul_tg_size, 1, 1),
                    );
                    // Barrier: gate+up matmul write gate_buf+up_buf, SwiGLU reads both
                    if needs_barriers { enc.memory_barrier_with_scope(1); }
                    // SwiGLU
                    enc.set_pipeline_state(&pipelines.swiglu);
                    enc.set_buffer(&s.gate_buf, 0, 0);
                    enc.set_buffer(&s.up_buf, 0, 1);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 2);
                    let tg = 256u64.min(inter_dim as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((inter_dim as u64).div_ceil(tg), 1, 1),
                        MTLSize::new(tg, 1, 1),
                    );
                }
                // Barrier: gate+up+SwiGLU writes gate_buf, down proj reads gate_buf
                if needs_barriers { enc.memory_barrier_with_scope(1); }
                // Down projection + Residual 2 (fused)
                {
                    let tg_down = match meta.w_down_quant {
                        QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_residual_nr2); 128u64 },
                        QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_residual_nr2); 128u64 },
                        QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_residual_nr2); 128u64 },
                        _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32_residual); matmul_tg_size },
                    };
                    enc.set_buffer(layer_buf, w_down_off, 0);
                    enc.set_buffer(&s.gate_buf, 0, 1);
                    enc.set_buffer(&s.x_buf, 0, 2);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
                    enc.set_buffer(&s.attn_proj_buf, 0, 4);
                    if matches!(meta.w_down_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
                    }
                    let n_tg_down = match meta.w_down_quant { QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2, _ => hidden_dim as u64 };
                    enc.dispatch_threadgroups(MTLSize::new(n_tg_down, 1, 1), MTLSize::new(tg_down, 1, 1));
                }
            } // end MoE vs dense FFN branch

            // Barrier: down+residual writes x_buf, next layer's RMSNorm reads x_buf
            if needs_barriers { enc.memory_barrier_with_scope(1); }
        } // end layer loop

        // Resolve global tensor buffers for final norm + output projection
        let (sc_norm_buf, sc_norm_off): (&MetalBuffer, u64) =
            if let Some((_, norm_o, _)) = s.gpu_global_offsets {
                (s.gpu_unified_weight_buf.as_ref().unwrap(), norm_o as u64)
            } else {
                (final_norm_buf, 0u64)
            };
        let (sc_proj_buf, sc_proj_off): (&MetalBuffer, u64) =
            if let Some((_, _, proj_o)) = s.gpu_global_offsets {
                (s.gpu_unified_weight_buf.as_ref().unwrap(), proj_o as u64)
            } else {
                (output_proj_buf, 0u64)
            };

        // --- Final RMSNorm + Logits + Argmax ---
            // Fuse final RMSNorm into output projection for Q8_0/Q4_0.
            // Eliminates 1 dispatch + 1 barrier + normed_buf write/read.
            if matches!(output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                match output_proj_quant {
                    QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2),
                    QuantScheme::F16 => enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                    _ => enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2),
                }
                enc.set_buffer(sc_proj_buf, sc_proj_off, 0);
                enc.set_buffer(&s.x_buf, 0, 1);
                enc.set_buffer(&s.logits_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 4);
                enc.set_buffer(sc_norm_buf, sc_norm_off, 5);
                enc.set_bytes(&eps.to_le_bytes(), 6);
                let n_tg = ((vocab_size as u64) + 1) / 2;
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
            } else {
            // Non-fused path for non-Q8_0 output projections
            // Final RMSNorm
            enc.set_pipeline_state(&pipelines.rmsnorm);
            enc.set_buffer(&s.x_buf, 0, 0);
            enc.set_buffer(sc_norm_buf, sc_norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );
            // Barrier: Final RMSNorm writes normed_buf, logits projection reads normed_buf
            if needs_barriers { enc.memory_barrier_with_scope(1); }
            // Logits projection
            let (proj_tg, proj_rows_per_tg) = match output_proj_quant {
                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); (128u64, 2u64) },
                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); (128u64, 2u64) },
                _ => { enc.set_pipeline_state(&pipelines.matmul_f32_deferred); (128u64, 4u64) },
            };
            enc.set_buffer(sc_proj_buf, sc_proj_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(&s.logits_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 4);
            {
                let n_tg = ((vocab_size as u64) + proj_rows_per_tg - 1) / proj_rows_per_tg;
                enc.dispatch_threadgroups(
                    MTLSize::new(n_tg, 1, 1),
                    MTLSize::new(proj_tg, 1, 1),
                );
            }
            }
            // Barrier: logits projection writes logits_buf, argmax reads logits_buf
            if needs_barriers { enc.memory_barrier_with_scope(1); }
            // GPU-side argmax
            enc.set_pipeline_state(&pipelines.argmax);
            enc.set_buffer(&s.logits_buf, 0, 0);
            enc.set_buffer(&s.argmax_result_buf, 0, 1);
            enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 2);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(256, 1, 1),
            );

        enc.end_encoding();

        // Single sync point for the entire token.
        cmd.commit_and_wait();

        // Record MoE expert activations for ALL layers (per-layer profiling).
        if s.moe_num_experts > 0 {
            if let Some(ref profiler) = self.expert_profiler {
                let top_k = s.moe_num_active_experts;
                let mut ids = vec![0u32; top_k];
                for layer in 0..num_layers {
                    if let Some(Some(ref per_layer_buf)) = s.moe_per_layer_expert_ids.get(layer) {
                        per_layer_buf.read_u32(&mut ids);
                        profiler.lock().unwrap().record(layer, &ids);
                    }
                }
            }

            // Router debug readback -- capture per-layer expert_ids + expert_weights.
            if self.router_debug_enabled {
                let top_k = s.moe_num_active_experts;
                let mut ids = vec![0u32; top_k];
                let mut wts = vec![0.0f32; top_k];
                let mut log = self.router_debug_log.lock().unwrap();
                for layer in 0..num_layers {
                    let has_ids = s.moe_per_layer_expert_ids.get(layer)
                        .and_then(|opt| opt.as_ref());
                    let has_wts = s.moe_per_layer_expert_weights.get(layer)
                        .and_then(|opt| opt.as_ref());
                    if let (Some(ids_buf), Some(wts_buf)) = (has_ids, has_wts) {
                        ids_buf.read_u32(&mut ids);
                        wts_buf.read_f32(&mut wts);
                        let spread = if wts.len() >= 2 { wts[0] - wts[1] } else { 0.0 };
                        log.push(RouterLayerStats {
                            layer,
                            expert_ids: ids.clone(),
                            expert_weights: wts.clone(),
                            weight_spread: spread,
                        });
                    }
                }
            }
        }

        // Check if profiling phase is complete and trigger cache warmup.
        self.maybe_trigger_warmup();

        s.gpu_x_valid = false;
        s.last_async_cmd = None;

        // Advance KV cache (CPU tracking -- GPU KV cache already written).
        kv.advance_seq_len()?;

        // Read only 4 bytes (u32 token ID) instead of 128 KB logits.
        let mut result = [0u32; 1];
        s.argmax_result_buf.read_u32(&mut result);

        drop(scratch_guard);

        Ok(result[0])
    }

}

impl ComputeBackend for MetalF32Backend {
    fn init(&mut self, hyperparams: &ModelHyperparams) -> Result<(), RuntimeError> {
        self.cached_hidden_dim = hyperparams.hidden_dim as usize;
        self.cached_vocab_size = hyperparams.vocab_size as usize;

        // Compile shader pipelines
        let pipelines = self.compile_pipelines()?;

        // Determine threadgroup sizes based on pipeline capabilities
        let matmul_tg_size = pipelines.matmul_bytes_f32
            .max_total_threads_per_threadgroup()
            .min(256); // cap at 256 for matmul (good balance)
        let norm_tg_size = pipelines.rmsnorm_bytes
            .max_total_threads_per_threadgroup()
            .min(256);
        let mha_tg_size = pipelines.multi_head_attention
            .max_total_threads_per_threadgroup()
            .min(256); // threads per head in multi_head_attention

        self.pipelines = Some(pipelines);

        // Upload global tensors to GPU
        // Upload embedding: use raw quantized bytes if available, else F32
        if let Some(ref raw) = self.embedding_raw {
            if matches!(self.embedding_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                self.embedding_buf = Some(
                    self.device.new_buffer_with_bytes(raw).ok_or_else(|| {
                        RuntimeError::Compute("Failed to create quantized embedding buffer".into())
                    })?
                );
            } else {
                self.embedding_buf = Some(self.upload_f32(&self.embedding)?);
            }
        } else {
            self.embedding_buf = Some(self.upload_f32(&self.embedding)?);
        }
        self.final_norm_buf = Some(self.upload_f32(&self.final_norm)?);
        // Upload output_proj: use raw Q8_0 bytes if available, else F32
        if let Some(ref raw) = self.output_proj_raw {
            if matches!(self.output_proj_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                self.output_proj_buf = Some(
                    self.device.new_buffer_with_bytes(raw).ok_or_else(|| {
                        RuntimeError::Compute("Failed to create Q8_0 output_proj buffer".into())
                    })?
                );
            } else {
                self.output_proj_buf = Some(self.upload_f32(&self.output_proj)?);
            }
        } else {
            self.output_proj_buf = Some(self.upload_f32(&self.output_proj)?);
        }

        // Compute dimensions
        let hidden_dim = hyperparams.hidden_dim as usize;
        let num_heads = hyperparams.num_heads as usize;
        let num_kv_heads = hyperparams.num_kv_heads as usize;
        let num_layers = hyperparams.num_layers as usize;
        let head_dim = hyperparams.head_dim as usize;
        let inter_dim = hyperparams.intermediate_dim as usize;
        let vocab_size = hyperparams.vocab_size as usize;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        // Use the max_seq_len from hyperparams. The caller (engine / CLI) is
        // responsible for capping this to a reasonable value via --context-len.
        // Models with very large context windows (e.g. 128K) allocate massive
        // KV caches that can devastate GPU performance (8+ GB KV = 5-7x slower
        // due to GPU memory pressure / TLB thrashing).
        let max_seq_len = hyperparams.max_seq_len as usize;
        let qkv_dim = q_dim + 2 * kv_dim;
        let half_dim = head_dim / 2;
        let gqa_ratio = num_heads / num_kv_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();

        // MoE dimensions (0 for dense models)
        let moe_num_experts = hyperparams.num_experts.unwrap_or(0) as usize;
        let moe_num_active_experts = hyperparams.num_active_experts.unwrap_or(0) as usize;
        // For MoE models, inter_dim is the per-expert intermediate dimension
        // (same as dense inter_dim for uniform-expert architectures like Mixtral).
        let moe_expert_inter_dim = if moe_num_experts > 0 { inter_dim } else { 0 };

        // Pre-compute RoPE tables and upload to GPU
        let theta = hyperparams.rope_params.as_ref().map(|r| r.theta).unwrap_or(10000.0);
        let mut rope_cos = vec![0.0f32; max_seq_len * half_dim];
        let mut rope_sin = vec![0.0f32; max_seq_len * half_dim];
        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                rope_cos[pos * half_dim + i] = angle.cos();
                rope_sin[pos * half_dim + i] = angle.sin();
            }
        }

        let rope_cos_buf = self.upload_f32(&rope_cos)?;
        let rope_sin_buf = self.upload_f32(&rope_sin)?;

        // Create scratch GPU buffers
        let make_buf = |n: usize| -> Result<MetalBuffer, RuntimeError> {
            let len = n.max(1) * 4; // at least 4 bytes
            self.device.new_buffer(len).ok_or_else(|| {
                RuntimeError::Compute(format!("Failed to allocate Metal buffer of {len} bytes"))
            })
        };

        // Allocate GPU-resident KV cache buffers: one pair per layer,
        // each sized for [max_seq_len * kv_dim] halfs (f16 = 2 bytes per element).
        let kv_cache_size = max_seq_len * kv_dim;
        let make_buf_f16 = |n: usize| -> Result<MetalBuffer, RuntimeError> {
            let len = n.max(1) * 2; // f16: 2 bytes per element
            self.device.new_buffer(len).ok_or_else(|| {
                RuntimeError::Compute(format!("Failed to allocate Metal buffer of {len} bytes"))
            })
        };
        let mut gpu_k_cache = Vec::with_capacity(num_layers);
        let mut gpu_v_cache = Vec::with_capacity(num_layers);
        for _layer in 0..num_layers {
            gpu_k_cache.push(make_buf_f16(kv_cache_size)?);
            gpu_v_cache.push(make_buf_f16(kv_cache_size)?);
        }

        // Allocate MoE decode scratch buffers when the model has experts.
        // Option B dispatches ALL num_experts expert FFNs; non-selected experts
        // produce zero output via zero routing weights.
        let (moe_router_logits, moe_expert_ids, moe_expert_weights, moe_expert_output) =
            if moe_num_experts > 0 {
                println!(
                    "MoE model detected: {} experts, top-{} active. Allocating MoE scratch buffers.",
                    moe_num_experts, moe_num_active_experts,
                );
                (
                    Some(make_buf(moe_num_experts)?),                           // [num_experts] f32
                    Some(self.device.new_buffer((moe_num_active_experts.max(1)) * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate MoE expert_ids buffer".into())
                    })?),                                                       // [top_k] u32
                    Some(make_buf(moe_num_active_experts)?),                    // [top_k] f32
                    Some(make_buf(moe_num_experts * hidden_dim)?),              // [num_experts * hidden_dim] f32
                )
            } else {
                (None, None, None, None)
            };

        let scratch = MetalScratch {
            // Persistent activation buffer: allocated once, reused every layer.
            x_buf: make_buf(hidden_dim)?,
            normed_buf: make_buf(hidden_dim)?,
            qkv_buf: make_buf(qkv_dim.max(8192))?,   // GDN needs 8192 (Q4096+K2048+V2048)
            q_buf: make_buf(q_dim)?,
            k_buf: make_buf(kv_dim.max(2048))?,    // GDN needs 2048 (16 KV heads * 128 dim)
            v_buf: make_buf(kv_dim.max(2048))?,    // GDN needs 2048 (16 KV heads * 128 dim)
            attn_out_buf: make_buf(q_dim)?,
            scores_buf: make_buf(max_seq_len)?,
            attn_proj_buf: make_buf(hidden_dim)?,
            gate_buf: make_buf(inter_dim.max(q_dim))?,  // max of FFN inter_dim and attn q_dim (Q+gate deinterleave)
            up_buf: make_buf(inter_dim)?,
            down_buf: make_buf(hidden_dim)?,
            logits_buf: make_buf(vocab_size)?,
            argmax_result_buf: self.device.new_buffer(4).ok_or_else(|| {
                RuntimeError::Compute("Failed to allocate argmax result buffer (4 bytes)".into())
            })?,
            rope_cos_buf,
            rope_sin_buf,
            gpu_k_cache,
            gpu_v_cache,
            mha_scores_buf: make_buf(num_heads * max_seq_len)?,
            // Flash decode: tile_size=256, max_tiles = ceil(max_seq/256)
            // Each tile: head_dim + 2 floats (weighted_v + max + sum)
            flash_decode_partial_buf: make_buf(
                num_heads * ((max_seq_len + 255) / 256) * (head_dim + 2)
            )?,
            hidden_dim,
            num_heads,
            num_kv_heads,
            num_layers,
            head_dim,
            inter_dim,
            eps: hyperparams.norm_eps,
            q_dim,
            kv_dim,
            qkv_dim,
            gqa_ratio,
            vocab_size,
            half_dim,
            max_seq_len,
            attn_scale,
            matmul_tg_size,
            norm_tg_size,
            mha_tg_size,

            gpu_x_valid: false,
            last_async_cmd: None,
            layer_buf_cache: (0..num_layers).map(|_| None).collect(),
            moe_partial_buf_cache: (0..num_layers).map(|_| None).collect(),
            gpu_resident_layers: None,
            gpu_unified_weight_buf: None,
            gpu_layer_offsets: Vec::new(),
            gpu_global_offsets: None,

            // Batched buffers: lazily allocated on first prefill call
            batch_x_buf: None,
            batch_normed_buf: None,
            batch_qkv_buf: None,
            batch_q_buf: None,
            batch_k_buf: None,
            batch_v_buf: None,
            batch_attn_out_buf: None,
            batch_attn_proj_buf: None,
            batch_gate_buf: None,
            batch_up_buf: None,
            batch_down_buf: None,
            batch_scores_buf: None,
            splitk_partial_buf: None,
            splitk_alloc_elems: 0,
            current_max_batch: 0,

            logits_readback: vec![0.0f32; vocab_size],
            cached_layer_meta: Vec::new(),

            // MoE parameters and scratch buffers
            moe_num_experts,
            moe_num_active_experts,
            moe_expert_inter_dim,
            moe_router_logits,
            moe_expert_ids,
            moe_expert_weights,
            moe_expert_output,
            // Batched MoE buffers: lazily allocated in ensure_batch_buffers
            moe_batch_router_logits: None,
            moe_batch_expert_ids: None,
            moe_batch_expert_weights: None,
            moe_batch_expert_output: None,

            // Per-layer expert IDs buffers for GPU-resident profiling.
            // Allocated for MoE models so each layer's expert selections are preserved.
            moe_per_layer_expert_ids: if moe_num_experts > 0 {
                (0..num_layers).map(|_| {
                    // Each layer gets its own [top_k] u32 buffer.
                    Some(self.device.new_buffer((moe_num_active_experts.max(1)) * 4)
                        .expect("Failed to allocate per-layer MoE expert_ids buffer"))
                }).collect()
            } else {
                Vec::new()
            },

            // Per-layer expert weights buffers for router diagnostics.
            // Allocated for MoE models when router_debug is enabled.
            moe_per_layer_expert_weights: if moe_num_experts > 0 && self.router_debug_enabled {
                (0..num_layers).map(|_| {
                    Some(self.device.new_buffer((moe_num_active_experts.max(1)) * 4)
                        .expect("Failed to allocate per-layer MoE expert_weights buffer"))
                }).collect()
            } else {
                Vec::new()
            },

            // Batched MoE offset tables — populated in preload_weights_gpu_resident.
            moe_gate_up_offsets: Vec::new(),
            moe_down_offsets: Vec::new(),
            moe_batched_swiglu_buf: if moe_num_experts > 0 && moe_num_active_experts > 0 {
                Some(self.device.new_buffer((moe_num_active_experts * moe_expert_inter_dim * 4).max(4))
                    .expect("Failed to allocate batched swiglu buffer"))
            } else {
                None
            },
            // Populated in preload_weights_gpu_resident alongside moe_gate_up_offsets
            moe_shared_down_offsets: Vec::new(),
            moe_shared_gate_scalar_buf: if moe_num_experts > 0 {
                // Single f32 for shared expert gating scalar
                Some(self.device.new_buffer(4)
                    .expect("Failed to allocate shared expert gate scalar buffer"))
            } else {
                None
            },

            // Qwen3.5-MoE scratch: defaults for non-hybrid models.
            // Overridden in preload_weights_gpu_resident when hybrid model is detected.
            is_qwen35moe: false,
            rotary_dim: head_dim,
            shared_expert_inter_dim: 0,
            shared_expert_gate_buf: None,
            shared_expert_down_buf: None,
            attn_gate_buf: None,

            // GDN state: defaults for non-hybrid models. Overridden in
            // preload_weights_gpu_resident when Qwen3.5-MoE is detected.
            gdn_h_states: Vec::new(),
            gdn_conv_states: Vec::new(),
            gdn_conv_positions: Vec::new(),
            gdn_alpha_buf: None,
            gdn_beta_buf: None,
            gdn_output_buf: None,
            gdn_ssm_proj_buf: None,
            gdn_gate_sigmoid_buf: None,
            gdn_normed_out_buf: None,
            gdn_alpha_raw_buf: None,
            gdn_beta_raw_buf: None,
            gdn_qkv_conv_buf: None,
            gdn_conv_kernel_size: 4,
            gdn_num_layers: 0,
            gdn_layer_idx_map: Vec::new(),
        };

        *self.scratch.lock().unwrap() = Some(scratch);

        // Pre-allocate batch buffers for common prefill sizes (up to 512 tokens).
        // This moves Metal buffer allocation from the first prefill call into
        // init(), so the first prefill is as fast as subsequent ones.
        {
            let default_batch = max_seq_len.min(512);
            let mut scratch_guard = self.scratch.lock().unwrap();
            let s = scratch_guard.as_mut().unwrap();
            self.ensure_batch_buffers(s, default_batch)?;
        }

        // Initialize MoE expert caching infrastructure.
        // Only activated for MoE models (num_experts > 0).
        if moe_num_experts > 0 {
            // Always initialize the profiler for MoE models.
            self.expert_profiler = Some(Mutex::new(
                ExpertActivationProfiler::new(num_layers, moe_num_experts),
            ));

            // Initialize ExpertReader if LBC path was provided via configure_expert_cache.
            if let Some(ref path) = self.lbc_path {
                match ExpertReader::open(path) {
                    Ok(reader) => {
                        self.expert_reader = Some(Mutex::new(reader));
                        println!(
                            "MoE expert cache: reader initialized for {}",
                            path.display(),
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: failed to open ExpertReader at {}: {e}. \
                             Expert caching disabled, falling back to full-layer loading.",
                            path.display(),
                        );
                    }
                }
            }

            if let Some(ref cache) = self.expert_cache {
                let stats = cache.lock().unwrap().stats();
                println!(
                    "MoE expert cache: capacity={} experts, reader={}",
                    stats.capacity,
                    if self.expert_reader.is_some() { "active" } else { "inactive" },
                );
            }
        }

        Ok(())
    }

    fn compute_layer(
        &self,
        layer_idx: usize,
        x: &mut ActivationBuffer,
        weights: &LayerView,
        kv: Option<&mut KvCacheView>,
        seq_pos: usize,
    ) -> Result<(), RuntimeError> {
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Metal pipelines not initialized: call init() first".into())
        })?;

        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("Metal scratch not initialized: call init() first".into())
        })?;

        let hidden_dim = s.hidden_dim;
        let num_heads = s.num_heads;
        let num_kv_heads = s.num_kv_heads;
        let head_dim = s.head_dim;
        let inter_dim = s.inter_dim;
        let eps = s.eps;
        let q_dim = s.q_dim;
        let kv_dim = s.kv_dim;
        let qkv_dim = s.qkv_dim;
        let half_dim = s.half_dim;
        let attn_scale = s.attn_scale;
        let matmul_tg_size = s.matmul_tg_size;
        let norm_tg_size = s.norm_tg_size;

        // GPU activation persistence: skip CPU->GPU upload if GPU already
        // has valid activation data. When embed_token wrote directly to x_buf
        // on GPU (gpu_x_valid=true), we skip the upload even at layer_idx==0.
        // compute_final always resets gpu_x_valid=false, ensuring fresh data
        // is loaded at the start of each forward pass.
        if !s.gpu_x_valid {
            // Drain any pending async GPU work before CPU writes to shared buffers.
            // This prevents races where a previous forward pass's async copy_buffer
            // hasn't completed yet.
            if let Some(prev_cmd) = s.last_async_cmd.take() {
                prev_cmd.wait_until_completed();
            }
            let x_f32 = x.as_f32_slice();
            s.x_buf.write_f32(x_f32);
        }

        // GPU-resident path: prefer unified private buffer, then per-layer buffers,
        // then fall back to cached zero-copy layer buffer.
        let layer_buf: &MetalBuffer;
        let base_off: u64;
        if let Some(ref ubuf) = s.gpu_unified_weight_buf {
            layer_buf = ubuf;
            base_off = s.gpu_layer_offsets[layer_idx] as u64;
        } else if let Some(ref layers) = s.gpu_resident_layers {
            if let Some(buf) = layers.get(layer_idx) {
                layer_buf = buf;
                base_off = 0;
            } else {
                return Err(RuntimeError::Compute(format!(
                    "gpu_resident_layers missing layer {}", layer_idx
                )));
            }
        } else {
            // Zero-copy layer buffer: cached per-layer to avoid re-creating Metal buffers.
            let blob = weights.as_bytes();
            let blob_ptr = blob.as_ptr() as usize;

            // For MoE layers with expert cache, use a partial buffer
            // covering only non-expert data (attention, norms, router). This avoids
            // page-faulting the expert byte range from mmap. Expert weights are served
            // from the LFU cache or loaded individually via ExpertReader.
            let is_moe_layer = s.moe_num_experts > 0
                && weights.subtensors.experts.is_some()
                && self.expert_cache.is_some();
            let use_partial = is_moe_layer && {
                let cache = self.expert_cache.as_ref().unwrap().lock().unwrap();
                let num_exp = s.moe_num_experts;
                (0..num_exp).all(|e| cache.contains(&(layer_idx, e as u32)))
            };

            if use_partial {
                // Partial buffer: only non-expert bytes.
                let non_exp_end = Self::non_expert_byte_end(&weights.subtensors);
                let cached = s.moe_partial_buf_cache.get(layer_idx).and_then(|c| c.as_ref());
                let need_create = match cached {
                    Some((ptr, end, _)) => *ptr != blob_ptr || *end != non_exp_end,
                    None => true,
                };
                if need_create {
                    let buf = self.create_partial_layer_buffer(weights, non_exp_end)?;
                    s.moe_partial_buf_cache[layer_idx] = Some((blob_ptr, non_exp_end, buf));
                }
                layer_buf = &s.moe_partial_buf_cache[layer_idx].as_ref().unwrap().2;

                // Hint to OS that expert byte pages can be reclaimed.
                // madvise(MADV_DONTNEED) on expert range [non_exp_end..blob_len].
                // Best-effort: no error handling if the call fails.
                if non_exp_end < blob.len() {
                    let expert_page_start = (non_exp_end + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
                    if expert_page_start < blob.len() {
                        unsafe {
                            libc::madvise(
                                (blob_ptr + expert_page_start) as *mut libc::c_void,
                                blob.len() - expert_page_start,
                                libc::MADV_DONTNEED,
                            );
                        }
                    }
                }
            } else {
                // Full buffer: all bytes (dense layers or MoE without full cache).
                let cached = s.layer_buf_cache.get(layer_idx).and_then(|c| c.as_ref());
                let need_create = match cached {
                    Some((ptr, _)) => *ptr != blob_ptr,
                    None => true,
                };
                if need_create {
                    let buf = self.create_layer_buffer(weights)?;
                    s.layer_buf_cache[layer_idx] = Some((blob_ptr, buf));
                }
                layer_buf = &s.layer_buf_cache[layer_idx].as_ref().unwrap().1;
            }
            base_off = 0;
        }
        let st = &weights.subtensors;

        let attn_norm_off = base_off + st.attn_norm.offset;
        let wq_off = base_off + st.wq.offset;  // Wq/Wk/Wv are contiguous; wq_off is the fused QKV start
        let wo_off = base_off + st.wo.offset;
        // Prefer attn_post_norm when ffn_norm is the zero-sentinel (Qwen3.5-35B-A3B).
        let ffn_norm_off = if st.ffn_norm.length == 0 {
            st.attn_post_norm.map_or(0, |s| base_off + s.offset)
        } else {
            base_off + st.ffn_norm.offset
        };
        let w_gate_off = base_off + st.w_gate.offset;
        let w_up_off = base_off + st.w_up.offset;
        let w_down_off = base_off + st.w_down.offset;

        let is_gdn_layer = st.layer_type == Some(1);

        let kv = kv.ok_or_else(|| {
            RuntimeError::Compute("KV cache view required for attention".into())
        })?;
        let new_seq_len = kv.seq_len + 1;

        // ================================================================
        // Lazy GDN state allocation for streaming path.
        // GPU-resident path allocates in preload_weights_gpu_resident;
        // streaming path needs lazy init on first encounter of each GDN layer.
        // ================================================================
        if is_gdn_layer {
            // Extend gdn_layer_idx_map to cover this layer_idx if needed.
            if s.gdn_layer_idx_map.len() <= layer_idx {
                s.gdn_layer_idx_map.resize(layer_idx + 1, None);
            }
            if s.gdn_layer_idx_map[layer_idx].is_none() {
                let gdn_idx = s.gdn_h_states.len();
                s.gdn_layer_idx_map[layer_idx] = Some(gdn_idx);

                // GDN dimensions differ from full-attention hyperparams.
                const GDN_NUM_HEADS: usize = 32;    // ssm.time_step_rank
                const GDN_HEAD_DIM: usize = 128;    // ssm.state_size
                const GDN_QKV_DIM: usize = 8192;    // Q(2048) + K(2048) + V(4096)
                let gdn_q_dim = GDN_NUM_HEADS * GDN_HEAD_DIM; // 4096
                let conv_kernel_size = 4usize;
                let h_state_size = GDN_NUM_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM; // 32*128*128
                let conv_state_size = (conv_kernel_size - 1) * GDN_QKV_DIM;     // 3*8192

                let h_buf = self.device.new_buffer(h_state_size * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN h_state".into())
                })?;
                h_buf.write_f32(&vec![0.0f32; h_state_size]);

                let c_buf = self.device.new_buffer(conv_state_size * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate GDN conv_state".into())
                })?;
                c_buf.write_f32(&vec![0.0f32; conv_state_size]);

                s.gdn_h_states.push(h_buf);
                s.gdn_conv_states.push(c_buf);
                s.gdn_conv_positions.push(0);
                s.gdn_conv_kernel_size = conv_kernel_size;
                s.gdn_num_layers = s.gdn_h_states.len();

                // Allocate GDN scratch buffers if not yet done (first GDN layer).
                if s.gdn_alpha_buf.is_none() {
                    s.gdn_alpha_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN alpha buf".into())
                    })?);
                    s.gdn_beta_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN beta buf".into())
                    })?);
                    s.gdn_output_buf = Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN output buf".into())
                    })?);
                    s.gdn_ssm_proj_buf = Some(self.device.new_buffer(hidden_dim * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN ssm_proj buf".into())
                    })?);
                    s.gdn_gate_sigmoid_buf = Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN gate sigmoid buf".into())
                    })?);
                    s.gdn_normed_out_buf = Some(self.device.new_buffer(gdn_q_dim * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN normed_out buf".into())
                    })?);
                    s.gdn_alpha_raw_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN alpha_raw buf".into())
                    })?);
                    s.gdn_beta_raw_buf = Some(self.device.new_buffer(GDN_NUM_HEADS * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN beta_raw buf".into())
                    })?);
                    s.gdn_qkv_conv_buf = Some(self.device.new_buffer(GDN_QKV_DIM * 4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate GDN qkv_conv buf".into())
                    })?);
                }
            }
        }

        // ================================================================
        // SINGLE command buffer for the ENTIRE layer (16 encoders, 1 sync).
        // Previous: 3 command buffers × 22 layers = 66 sync barriers/token.
        // Now: 1 command buffer × 22 layers = 22 sync barriers/token.
        // ================================================================
        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer".into())
        })?;

        // ================================================================
        // GDN vs softmax attention routing.
        // GDN layers use encode_gdn_layer_decode (linear attention with
        // recurrent state). Softmax layers use the standard RoPE + KV cache
        // + multi-head attention path. Both produce valid attn_proj_buf for
        // the downstream FFN norm.
        // ================================================================
        if is_gdn_layer {
            // Build CachedLayerMeta from SubtensorOffsets for GDN dispatch.
            let gdn_idx = s.gdn_layer_idx_map[layer_idx].unwrap();
            let gdn_meta = CachedLayerMeta {
                attn_norm_off,
                wq_off,
                wo_off,
                ffn_norm_off,
                w_gate_off,
                w_up_off,
                w_down_off,
                wq_quant: st.wq.quant,
                wo_quant: st.wo.quant,
                w_gate_quant: st.w_gate.quant,
                w_up_quant: st.w_up.quant,
                w_down_quant: st.w_down.quant,
                bq_off: st.bq.map(|b| base_off + b.offset),
                bk_off: st.bk.map(|b| base_off + b.offset),
                bv_off: st.bv.map(|b| base_off + b.offset),
                moe_meta: None,
                shared_expert_gate_off: None,
                shared_expert_up_off: None,
                shared_expert_down_off: None,
                shared_expert_gate_quant: None,
                shared_expert_down_quant: None,
                attn_gate_off: st.attn_gate.map(|g| base_off + g.offset),
                attn_gate_quant: st.attn_gate.map(|g| g.quant),
                attn_post_norm_off: st.attn_post_norm.map(|n| base_off + n.offset),
                layer_type: st.layer_type,
                ssm_a_off: st.ssm_a.map(|t| base_off + t.offset),
                ssm_conv1d_off: st.ssm_conv1d.map(|t| base_off + t.offset),
                ssm_dt_off: st.ssm_dt.map(|t| base_off + t.offset),
                ssm_beta_off: st.ssm_beta.map(|t| base_off + t.offset),
                ssm_alpha_off: st.ssm_alpha.map(|t| base_off + t.offset),
                ssm_norm_off: st.ssm_norm.map(|t| base_off + t.offset),
                ssm_out_off: st.ssm_out.map(|t| base_off + t.offset),
                ssm_out_quant: st.ssm_out.map(|t| t.quant),
                gdn_layer_idx: Some(gdn_idx),
                // GDN layers never have Q+gate fusion or separate K/V weights.
                has_qgate_fusion: false,
                wk_off: None,
                wv_off: None,
                wk_quant: None,
                wv_quant: None,
                attn_q_norm_off: None,
                attn_k_norm_off: None,
                ffn_gate_inp_shexp_off: st.ffn_gate_inp_shexp.map(|t| base_off + t.offset),
            };

            let new_conv_pos = Self::encode_gdn_layer_decode(
                &cmd, pipelines, s, layer_buf, &gdn_meta, gdn_idx,
            )?;
            s.gdn_conv_positions[gdn_idx] = new_conv_pos;
        } else {
        // --- Encoder 1: Attention RMSNorm ---
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let dim_u32 = hidden_dim as u32;
            enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
            enc.set_buffer(&s.x_buf, 0, 0);
            enc.set_buffer(layer_buf, attn_norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&dim_u32.to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );
            enc.end_encoding();
        }

        // --- Encoder 2: QKV projection ---
        // Two paths: Q+gate fusion (Qwen3.5 full-attention) vs fused QKV (standard).
        let has_qgate_fusion = st.attn_q_norm.is_some();
        let q_byte_off: u64 = 0;
        let k_byte_off: u64 = (q_dim * 4) as u64;
        let v_byte_off: u64 = ((q_dim + kv_dim) * 4) as u64;
        // Use partial rotary dimension when set (e.g. Qwen3.5: rotary_dim=64, not head_dim=256).
        let rope_half_dim = if s.rotary_dim > 0 && s.rotary_dim < head_dim { s.rotary_dim / 2 } else { half_dim };
        let use_partial_rope = rope_half_dim != half_dim;

        if has_qgate_fusion {
            // Q+gate fusion path (Qwen3.5 full-attention layers).
            // Q weight [hidden, q_dim*2] produces interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...].
            // K and V come from separate wk, wv weights.
            let qgate_dim = q_dim * 2;
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for Q+gate matmul".into())
                })?;
                let tg = match st.wq.quant {
                    QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                    QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                    QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                    _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                };
                enc.set_buffer(layer_buf, wq_off, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(&s.qkv_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                if matches!(st.wq.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                    enc.set_bytes(&(qgate_dim as u32).to_le_bytes(), 4);
                }
                let n_tg = match st.wq.quant {
                    QuantScheme::Q8_0 => ((qgate_dim as u64) + 1) / 2,
                    QuantScheme::Q4_0 => ((qgate_dim as u64) + 1) / 2,
                    QuantScheme::F16 => ((qgate_dim as u64) + 1) / 2,
                    _ => qgate_dim as u64,
                };
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
                enc.end_encoding();
            }
            // De-interleave Q+gate -> q_buf + gate_buf
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for deinterleave".into())
                })?;
                let pso = pipelines.deinterleave_qgate.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("deinterleave_qgate pipeline not compiled".into())
                })?;
                enc.set_pipeline_state(pso);
                enc.set_buffer(&s.qkv_buf, 0, 0);
                enc.set_buffer(&s.q_buf, 0, 1);
                enc.set_buffer(&s.gate_buf, 0, 2);
                enc.set_bytes(&(head_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&(num_heads as u32).to_le_bytes(), 4);
                let di_tg = 256u64.min(q_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((q_dim as u64).div_ceil(di_tg), 1, 1),
                    MTLSize::new(di_tg, 1, 1),
                );
                enc.end_encoding();
            }
            // Project K from wk
            {
                let wk_off_val = base_off + st.wk.offset;
                let wk_quant = st.wk.quant;
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for K matmul".into())
                })?;
                let tg = match wk_quant {
                    QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                    QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                    QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                    _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                };
                enc.set_buffer(layer_buf, wk_off_val, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(&s.k_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                if matches!(wk_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                }
                let n_tg = match wk_quant {
                    QuantScheme::Q8_0 => ((kv_dim as u64) + 1) / 2,
                    QuantScheme::Q4_0 => ((kv_dim as u64) + 1) / 2,
                    QuantScheme::F16 => ((kv_dim as u64) + 1) / 2,
                    _ => kv_dim as u64,
                };
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
                enc.end_encoding();
            }
            // Project V from wv
            {
                let wv_off_val = base_off + st.wv.offset;
                let wv_quant = st.wv.quant;
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for V matmul".into())
                })?;
                let tg = match wv_quant {
                    QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                    QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                    QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                    _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                };
                enc.set_buffer(layer_buf, wv_off_val, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(&s.v_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                if matches!(wv_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                }
                let n_tg = match wv_quant {
                    QuantScheme::Q8_0 => ((kv_dim as u64) + 1) / 2,
                    QuantScheme::Q4_0 => ((kv_dim as u64) + 1) / 2,
                    QuantScheme::F16 => ((kv_dim as u64) + 1) / 2,
                    _ => kv_dim as u64,
                };
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
                enc.end_encoding();
            }
            // Per-head Q/K RMSNorm
            if let (Some(ref q_norm), Some(ref k_norm)) = (&st.attn_q_norm, &st.attn_k_norm) {
                let q_norm_off = base_off + q_norm.offset;
                let k_norm_off = base_off + k_norm.offset;
                let pso = pipelines.rmsnorm_per_head.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("rmsnorm_per_head pipeline not compiled".into())
                })?;
                let head_dim_u32 = head_dim as u32;
                let tg_rms = 256u64.min(head_dim as u64).max(32);
                // Q RMSNorm
                {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder for Q RMSNorm".into())
                    })?;
                    enc.set_pipeline_state(pso);
                    enc.set_buffer(&s.q_buf, 0, 0);
                    enc.set_buffer(layer_buf, q_norm_off, 1);
                    enc.set_buffer(&s.q_buf, 0, 2);
                    enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                    enc.set_bytes(&eps.to_le_bytes(), 4);
                    enc.dispatch_threadgroups(
                        MTLSize::new(num_heads as u64, 1, 1),
                        MTLSize::new(tg_rms, 1, 1),
                    );
                    enc.end_encoding();
                }
                // K RMSNorm
                {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder for K RMSNorm".into())
                    })?;
                    enc.set_pipeline_state(pso);
                    enc.set_buffer(&s.k_buf, 0, 0);
                    enc.set_buffer(layer_buf, k_norm_off, 1);
                    enc.set_buffer(&s.k_buf, 0, 2);
                    enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                    enc.set_bytes(&eps.to_le_bytes(), 4);
                    enc.dispatch_threadgroups(
                        MTLSize::new(num_kv_heads as u64, 1, 1),
                        MTLSize::new(tg_rms, 1, 1),
                    );
                    enc.end_encoding();
                }
            }
            // Assemble Q|K|V into qkv_buf for RoPE/attention
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for QKV assembly".into())
                })?;
                enc.set_pipeline_state(&pipelines.copy_buffer);
                // Copy Q
                enc.set_buffer(&s.q_buf, 0, 0);
                enc.set_buffer(&s.qkv_buf, 0, 1);
                let tg_q = 256u64.min(q_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((q_dim as u64).div_ceil(tg_q), 1, 1),
                    MTLSize::new(tg_q, 1, 1),
                );
                // Copy K
                enc.set_buffer(&s.k_buf, 0, 0);
                enc.set_buffer(&s.qkv_buf, k_byte_off, 1);
                let tg_k = 256u64.min(kv_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((kv_dim as u64).div_ceil(tg_k), 1, 1),
                    MTLSize::new(tg_k, 1, 1),
                );
                // Copy V
                enc.set_buffer(&s.v_buf, 0, 0);
                enc.set_buffer(&s.qkv_buf, v_byte_off, 1);
                let tg_v = 256u64.min(kv_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((kv_dim as u64).div_ceil(tg_v), 1, 1),
                    MTLSize::new(tg_v, 1, 1),
                );
                enc.end_encoding();
            }
        } else {
        // Fused QKV projection (standard path: Wq|Wk|Wv contiguous).
        {
            let has_bias = st.bq.is_some() && st.bk.is_some() && st.bv.is_some();
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let in_dim_u32 = hidden_dim as u32;

            let tg = if has_bias && matches!(st.wq.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                match st.wq.quant {
                    QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_bias_nr2),
                    QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_bias_nr2),
                    QuantScheme::F16 => enc.set_pipeline_state(&pipelines.matmul_f16_deferred_bias_nr2),
                    _ => unreachable!(),
                };
                128u64
            } else {
                match st.wq.quant {
                    QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); 128u64 },
                    QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                    QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                    _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); matmul_tg_size },
                }
            };
            // Weight pointer starts at Wq; contiguous through Wk, Wv
            enc.set_buffer(layer_buf, wq_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(&s.qkv_buf, 0, 2);
            enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
            let qkv_out_dim_u32 = qkv_dim as u32;
            if matches!(st.wq.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                enc.set_bytes(&qkv_out_dim_u32.to_le_bytes(), 4);
            }
            if has_bias && matches!(st.wq.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                let bq_off_abs = base_off + st.bq.as_ref().unwrap().offset;
                let bk_off_abs = base_off + st.bk.as_ref().unwrap().offset;
                let bv_off_abs = base_off + st.bv.as_ref().unwrap().offset;
                enc.set_buffer(layer_buf, bq_off_abs, 5);
                enc.set_buffer(layer_buf, bk_off_abs, 6);
                enc.set_buffer(layer_buf, bv_off_abs, 7);
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 8);
                let qk_dim = (q_dim + kv_dim) as u32;
                enc.set_bytes(&qk_dim.to_le_bytes(), 9);
            }
            let n_tg_qkv = if tg == 64 {
                ((qkv_dim as u64) + 7) / 8  // (dead path: Q8_0 now uses deferred with tg=128)
            } else {
                match st.wq.quant {
                    QuantScheme::Q8_0 => ((qkv_dim as u64) + 1) / 2,
                    QuantScheme::Q4_0 => ((qkv_dim as u64) + 1) / 2,
                    QuantScheme::F16 => ((qkv_dim as u64) + 1) / 2,
                    _ => qkv_dim as u64,
                }
            };
            enc.dispatch_threadgroups(
                MTLSize::new(n_tg_qkv, 1, 1),
                MTLSize::new(tg, 1, 1),
            );

            enc.end_encoding();
        }

        // --- QKV bias addition fallback (F32 weights with bias only) ---
        if !matches!(st.wq.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
            && (st.bq.is_some() || st.bk.is_some() || st.bv.is_some())
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for QKV bias".into())
            })?;
            enc.set_pipeline_state(&pipelines.bias_add);
            if let Some(ref bq) = st.bq {
                let bq_off = base_off + bq.offset;
                enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                enc.set_buffer(layer_buf, bq_off, 1);
                enc.set_bytes(&(q_dim as u32).to_le_bytes(), 2);
                let n_tg_bq = (q_dim as u64 + 255) / 256;
                enc.dispatch_threadgroups(MTLSize::new(n_tg_bq, 1, 1), MTLSize::new(256, 1, 1));
            }
            if let Some(ref bk) = st.bk {
                let bk_off = base_off + bk.offset;
                enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
                enc.set_buffer(layer_buf, bk_off, 1);
                enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 2);
                let n_tg_bk = (kv_dim as u64 + 255) / 256;
                enc.dispatch_threadgroups(MTLSize::new(n_tg_bk, 1, 1), MTLSize::new(256, 1, 1));
            }
            if let Some(ref bv) = st.bv {
                let bv_off = base_off + bv.offset;
                enc.set_buffer(&s.qkv_buf, v_byte_off, 0);
                enc.set_buffer(layer_buf, bv_off, 1);
                enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 2);
                let n_tg_bv = (kv_dim as u64 + 255) / 256;
                enc.dispatch_threadgroups(MTLSize::new(n_tg_bv, 1, 1), MTLSize::new(256, 1, 1));
            }
            enc.end_encoding();
        }
        } // end if has_qgate_fusion else

        // --- RoPE + KV cache write ---
        // For partial RoPE (Qwen3.5: rotary_dim=64 < head_dim=256), use separate dispatches.
        // For full RoPE, use fused dispatch.
        if use_partial_rope {
            // Separate RoPE Q + RoPE K + KV cache write (partial rotary dim).
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for partial RoPE".into())
            })?;
            let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
            let rope_pipe = if s.is_qwen35moe {
                pipelines.rope_neox.as_ref().unwrap_or(&pipelines.rope)
            } else {
                &pipelines.rope
            };
            // RoPE Q
            enc.set_pipeline_state(rope_pipe);
            enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
            enc.set_buffer(&s.rope_cos_buf, 0, 1);
            enc.set_buffer(&s.rope_sin_buf, 0, 2);
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 3);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(rope_half_dim as u32).to_le_bytes(), 5);
            enc.set_bytes(&pos_offset_u32.to_le_bytes(), 6);
            let q_total_half = (num_heads * rope_half_dim) as u64;
            let tg_q = 64u64.min(q_total_half.max(1));
            enc.dispatch_threadgroups(
                MTLSize::new(q_total_half.div_ceil(tg_q), 1, 1),
                MTLSize::new(tg_q, 1, 1),
            );
            // RoPE K
            enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 3);
            let k_total_half = (num_kv_heads * rope_half_dim) as u64;
            let tg_k = 64u64.min(k_total_half.max(1));
            enc.dispatch_threadgroups(
                MTLSize::new(k_total_half.div_ceil(tg_k), 1, 1),
                MTLSize::new(tg_k, 1, 1),
            );
            // KV cache write
            enc.set_pipeline_state(&pipelines.write_kv_cache);
            enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
            enc.set_buffer(&s.qkv_buf, v_byte_off, 1);
            enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 2);
            enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 3);
            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 5);
            enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 6);
            let tg_kv = 64u64.min(kv_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((kv_dim as u64).div_ceil(tg_kv), 1, 1),
                MTLSize::new(tg_kv, 1, 1),
            );
            enc.end_encoding();
        } else {
            // Fused RoPE Q + RoPE K + KV cache write (full rotary dim).
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let pos_offset_u32 = (seq_pos * half_dim) as u32;
            let fused_pipe = if s.is_qwen35moe {
                pipelines.fused_rope_neox_kv_write.as_ref().unwrap_or(&pipelines.fused_rope_kv_write)
            } else {
                &pipelines.fused_rope_kv_write
            };
            enc.set_pipeline_state(fused_pipe);
            enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
            enc.set_buffer(&s.qkv_buf, k_byte_off, 1);
            enc.set_buffer(&s.qkv_buf, v_byte_off, 2);
            enc.set_buffer(&s.rope_cos_buf, 0, 3);
            enc.set_buffer(&s.rope_sin_buf, 0, 4);
            enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 5);
            enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 6);
            enc.set_bytes(&(num_heads as u32).to_le_bytes(), 7);
            enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 8);
            enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);
            enc.set_bytes(&(half_dim as u32).to_le_bytes(), 10);
            enc.set_bytes(&pos_offset_u32.to_le_bytes(), 11);
            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 12);
            enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 13);
            enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 14);
            let total_threads = (num_heads * half_dim + num_kv_heads * half_dim + kv_dim) as u64;
            let tg = 64u64.min(total_threads.max(1));
            enc.dispatch_threadgroups(
                MTLSize::new(total_threads.div_ceil(tg), 1, 1),
                MTLSize::new(tg, 1, 1),
            );
            enc.end_encoding();
        }

        // --- Encoder 8: Attention (flash decode for long KV, original for short) ---
        //
        // Flash Decoding splits the KV sequence into tiles processed by separate
        // threadgroups, then reduces. This provides parallelism across the KV
        // dimension which is critical when seq_len is large (e.g., 512+).
        // For short sequences (<=128), the original single-threadgroup kernel
        // is used since the overhead of the two-phase approach is not justified.
        {
            let num_heads_u32 = num_heads as u32;
            let num_kv_heads_u32 = num_kv_heads as u32;
            let head_dim_u32 = head_dim as u32;
            let kv_dim_u32 = kv_dim as u32;
            let seq_len_u32 = new_seq_len as u32;
            let max_seq_len_u32 = s.max_seq_len as u32;

            const FLASH_DECODE_TILE_SIZE: u32 = 256;
            const FLASH_DECODE_THRESHOLD: usize = FLASH_DECODE_TILE_SIZE as usize + 1; // 257: single-tile is a no-op reduce

            if new_seq_len >= FLASH_DECODE_THRESHOLD {
                // --- Flash Decode Phase 1: tiled attention ---
                let tile_size_u32 = FLASH_DECODE_TILE_SIZE;
                let num_tiles = ((new_seq_len as u32) + tile_size_u32 - 1) / tile_size_u32;
                let num_tiles_u32 = num_tiles;

                {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder".into())
                    })?;
                    enc.set_pipeline_state(&pipelines.flash_decode_attention);
                    enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                    enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 1);
                    enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 2);
                    enc.set_buffer(&s.flash_decode_partial_buf, 0, 3);
                    enc.set_bytes(&num_heads_u32.to_le_bytes(), 4);
                    enc.set_bytes(&num_kv_heads_u32.to_le_bytes(), 5);
                    enc.set_bytes(&head_dim_u32.to_le_bytes(), 6);
                    enc.set_bytes(&kv_dim_u32.to_le_bytes(), 7);
                    enc.set_bytes(&seq_len_u32.to_le_bytes(), 8);
                    enc.set_bytes(&attn_scale.to_le_bytes(), 9);
                    enc.set_bytes(&tile_size_u32.to_le_bytes(), 10);
                    enc.set_bytes(&num_tiles_u32.to_le_bytes(), 11);
                    enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 12);
                    // Each threadgroup gets 128 threads (tile_size=256, each thread handles multiple scores)
                    // Use flattened 1D grid: threadgroup i = head * num_tiles + tile_idx
                    let tg_threads = 128u64;
                    let total_tgs = (num_heads as u64) * (num_tiles as u64);
                    enc.dispatch_threadgroups(
                        MTLSize::new(total_tgs, 1, 1),
                        MTLSize::new(tg_threads, 1, 1),
                    );
                    enc.end_encoding();
                }

                // --- Flash Decode Phase 2: reduce across tiles ---
                {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder".into())
                    })?;
                    enc.set_pipeline_state(&pipelines.flash_decode_reduce);
                    enc.set_buffer(&s.flash_decode_partial_buf, 0, 0);
                    enc.set_buffer(&s.attn_out_buf, 0, 1);
                    enc.set_bytes(&num_heads_u32.to_le_bytes(), 2);
                    enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                    enc.set_bytes(&num_tiles_u32.to_le_bytes(), 4);
                    // One threadgroup per head, enough threads for head_dim
                    let tg_threads = (head_dim as u64).max(1).min(256);
                    enc.dispatch_threadgroups(
                        MTLSize::new(num_heads as u64, 1, 1),
                        MTLSize::new(tg_threads, 1, 1),
                    );
                    enc.end_encoding();
                }
            } else {
                // --- Original single-threadgroup MHA for short sequences ---
                let mha_tg_size = s.mha_tg_size;
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
                enc.set_pipeline_state(&pipelines.multi_head_attention);
                enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 1);
                enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 2);
                enc.set_buffer(&s.attn_out_buf, 0, 3);
                enc.set_buffer(&s.mha_scores_buf, 0, 4);
                enc.set_bytes(&num_heads_u32.to_le_bytes(), 5);
                enc.set_bytes(&num_kv_heads_u32.to_le_bytes(), 6);
                enc.set_bytes(&head_dim_u32.to_le_bytes(), 7);
                enc.set_bytes(&kv_dim_u32.to_le_bytes(), 8);
                enc.set_bytes(&seq_len_u32.to_le_bytes(), 9);
                enc.set_bytes(&attn_scale.to_le_bytes(), 10);
                enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 11);
                let tg_threads = mha_tg_size.min(
                    (head_dim.max(new_seq_len) as u64).max(1)
                );
                enc.dispatch_threadgroups(
                    MTLSize::new(num_heads as u64, 1, 1),
                    MTLSize::new(tg_threads, 1, 1),
                );
                enc.end_encoding();
            }
        }

        // --- Sigmoid gate for Q+gate fusion (Qwen3.5 full-attention) ---
        // sigmoid(gate_buf) * attn_out_buf -> attn_out_buf (in-place)
        if has_qgate_fusion {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for sigmoid gate".into())
            })?;
            let pso = pipelines.sigmoid_mul_fused.as_ref().ok_or_else(|| {
                RuntimeError::Compute("sigmoid_mul_fused pipeline not compiled".into())
            })?;
            enc.set_pipeline_state(pso);
            enc.set_buffer(&s.gate_buf, 0, 0);       // gate [q_dim]
            enc.set_buffer(&s.attn_out_buf, 0, 1);   // attn output [q_dim]
            enc.set_buffer(&s.attn_out_buf, 0, 2);   // output (in-place)
            let total_gate_elems = q_dim as u32;
            enc.set_bytes(&total_gate_elems.to_le_bytes(), 3);
            let tg = 256u64.min(total_gate_elems as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((total_gate_elems as u64).div_ceil(tg), 1, 1),
                MTLSize::new(tg, 1, 1),
            );
            enc.end_encoding();
        }

        // --- Encoder 6: Wo projection + Residual 1 (fused) ---
        // attn_proj_buf = Wo * attn_out + x_buf  (eliminates separate add_residual)
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let in_dim_u32 = q_dim as u32;
            let tg_wo = match st.wo.quant {
                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_residual_nr2); 128u64 },
                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_residual_nr2); 128u64 },
                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_residual_nr2); 128u64 },
                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32_residual); matmul_tg_size },
            };
            enc.set_buffer(layer_buf, wo_off, 0);
            enc.set_buffer(&s.attn_out_buf, 0, 1);
            enc.set_buffer(&s.attn_proj_buf, 0, 2);
            enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
            enc.set_buffer(&s.x_buf, 0, 4);  // residual input
            if matches!(st.wo.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                let wo_out_dim_u32 = hidden_dim as u32;
                enc.set_bytes(&wo_out_dim_u32.to_le_bytes(), 5);
            }
            let n_tg_wo = match st.wo.quant {
                QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2,
                QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2,
                QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2,
                _ => hidden_dim as u64,
            };
            enc.dispatch_threadgroups(
                MTLSize::new(n_tg_wo, 1, 1),
                MTLSize::new(tg_wo, 1, 1),
            );
            enc.end_encoding();
        }
        } // end if !is_gdn_layer (softmax attention block)

        // --- Encoder 11: FFN RMSNorm ---
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let dim_u32 = hidden_dim as u32;
            enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
            enc.set_buffer(&s.attn_proj_buf, 0, 0);
            enc.set_buffer(layer_buf, ffn_norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&dim_u32.to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );
            enc.end_encoding();
        }

        // --- FFN block: MoE vs dense path for streaming decode ---
        // If this layer has MoE experts, dispatch via encode_moe_ffn_decode and skip
        // the dense FFN path below.
        //
        // Option A: When use_option_a is true and we're in streaming mode,
        // split into two CBs: CB1 (attention+norm+router) -> readback -> CB2 (top-K FFNs).
        // This eliminates (num_experts - top_k) / num_experts of expert FFN compute.
        //
        // Partial layer loading: Assemble expert weights from
        // LFU cache (hits) + ExpertReader (misses), avoiding the need to
        // create a Metal buffer for the full layer blob's expert region.
        // When all experts are cached, the layer_buf is a partial buffer
        // covering only non-expert bytes -- zero expert I/O from disk.
        // When some experts are missing, they are loaded individually via
        // ExpertReader, which reads only the needed gate+up+down byte
        // ranges from the LBC file.

        // Check if Option A is active for this layer.
        // Option A requires: (1) use_option_a flag, (2) streaming mode (not GPU-resident),
        // (3) MoE model with expert cache or reader available.
        let is_streaming_mode = s.gpu_unified_weight_buf.is_none()
            && s.gpu_resident_layers.is_none();
        let option_a_active = self.use_option_a
            && is_streaming_mode
            && s.moe_num_experts > 0
            && self.expert_profiler.is_some();

        let moe_handled = if s.moe_num_experts > 0 {
            if let (Some(ref router), Some(ref experts)) = (&st.router_weight, &st.experts) {
                let first = &experts[0];
                let num_experts = s.moe_num_experts;
                let top_k = s.moe_num_active_experts;

                // ==============================================================
                // Option A: Two-CB split for selective expert dispatch
                // ==============================================================
                if option_a_active {
                    // Phase 1: Encode router into the current CB, then commit+wait.
                    // The router writes expert_ids and expert_weights to GPU buffers.
                    {
                        // Build is_cached buffer for biased routing.
                        let (is_cached_buf, bias_lambda) = if self.cache_bias_lambda > 0.0
                            && self.warmup_complete.load(Ordering::Relaxed)
                            && self.expert_cache.is_some()
                        {
                            let cache = self.expert_cache.as_ref().unwrap().lock().unwrap();
                            let is_cached_data: Vec<u8> = (0..num_experts)
                                .map(|e| if cache.contains(&(layer_idx, e as u32)) { 1u8 } else { 0u8 })
                                .collect();
                            drop(cache);
                            (self.device.new_buffer_with_bytes(&is_cached_data), self.cache_bias_lambda)
                        } else {
                            (None, 0.0)
                        };

                        let use_biased = bias_lambda > 0.0 && is_cached_buf.is_some();
                        let router_softmax = if use_biased {
                            pipelines.moe_router_softmax_biased.as_ref().ok_or_else(|| {
                                RuntimeError::Compute("MoE router_softmax_biased pipeline not compiled.".into())
                            })?
                        } else {
                            pipelines.moe_router_softmax.as_ref().ok_or_else(|| {
                                RuntimeError::Compute("MoE router_softmax pipeline not compiled.".into())
                            })?
                        };

                        let expert_ids_buf = s.moe_expert_ids.as_ref().ok_or_else(|| {
                            RuntimeError::Compute("MoE expert_ids buffer not allocated".into())
                        })?;
                        let expert_weights_buf = s.moe_expert_weights.as_ref().ok_or_else(|| {
                            RuntimeError::Compute("MoE expert_weights buffer not allocated".into())
                        })?;

                        let enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for MoE router".into())
                        })?;
                        enc.set_pipeline_state(router_softmax);
                        enc.set_buffer(&s.normed_buf, 0, 0);
                        enc.set_buffer(layer_buf, base_off + router.offset, 1);
                        enc.set_buffer(expert_ids_buf, 0, 2);
                        enc.set_buffer(expert_weights_buf, 0, 3);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                        enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
                        enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
                        if use_biased {
                            enc.set_buffer(is_cached_buf.as_ref().unwrap(), 0, 7);
                            enc.set_bytes(&bias_lambda.to_le_bytes(), 8);
                        }
                        let tg = 256u64.min(hidden_dim as u64).max(1);
                        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
                        enc.end_encoding();
                    }

                    // Commit CB1 (attention + FFN norm + router) and wait.
                    cmd.commit();
                    cmd.wait_until_completed();

                    // Read back expert_ids from GPU.
                    let expert_ids_buf = s.moe_expert_ids.as_ref().unwrap();
                    let mut cpu_ids = vec![0u32; top_k];
                    expert_ids_buf.read_u32(&mut cpu_ids);

                    // Record expert activation in profiler.
                    if let Some(ref profiler) = self.expert_profiler {
                        profiler.lock().unwrap().record(layer_idx, &cpu_ids);
                    }

                    // Populate expert cache with selected experts from blob.
                    if let Some(ref cache_mutex) = self.expert_cache {
                        let blob = weights.as_bytes();
                        let mut cache = cache_mutex.lock().unwrap();
                        for &eid in &cpu_ids {
                            let key = (layer_idx, eid);
                            if !cache.contains(&key) {
                                let eid_usize = eid as usize;
                                if eid_usize < experts.len() {
                                    let expert_slice = &experts[eid_usize];
                                    let gate_start = expert_slice.gate.offset as usize;
                                    let gate_end = gate_start + expert_slice.gate.length as usize;
                                    let up_start = expert_slice.up.offset as usize;
                                    let up_end = up_start + expert_slice.up.length as usize;
                                    let down_start = expert_slice.down.offset as usize;
                                    let down_end = down_start + expert_slice.down.length as usize;

                                    if gate_end <= blob.len() && up_end <= blob.len() && down_end <= blob.len() {
                                        let mut data = Vec::with_capacity(
                                            expert_slice.gate.length as usize
                                            + expert_slice.up.length as usize
                                            + expert_slice.down.length as usize
                                        );
                                        data.extend_from_slice(&blob[gate_start..gate_end]);
                                        data.extend_from_slice(&blob[up_start..up_end]);
                                        data.extend_from_slice(&blob[down_start..down_end]);

                                        let local_slice = lumen_format::index::ExpertSlice {
                                            gate: lumen_format::index::TensorSlice {
                                                offset: 0,
                                                length: expert_slice.gate.length,
                                                quant: expert_slice.gate.quant,
                                            },
                                            up: lumen_format::index::TensorSlice {
                                                offset: expert_slice.gate.length,
                                                length: expert_slice.up.length,
                                                quant: expert_slice.up.quant,
                                            },
                                            down: lumen_format::index::TensorSlice {
                                                offset: expert_slice.gate.length + expert_slice.up.length,
                                                length: expert_slice.down.length,
                                                quant: expert_slice.down.quant,
                                            },
                                        };
                                        cache.insert(key, data, local_slice);
                                    }
                                }
                            }
                        }
                    }

                    // Phase 2: Assemble only top-K expert weights and dispatch.
                    // Try prefetch first, then cache, then ExpertReader, then full blob.

                    // Check if async prefetch from the previous layer produced
                    // results for THIS layer. Prefetch hit means zero-wait I/O.
                    let mut prefetched: Vec<Option<(Vec<u8>, lumen_format::index::ExpertSlice)>> =
                        (0..top_k).map(|_| None).collect();
                    {
                        let mut ph_guard = self.prefetch_handle.lock().unwrap();
                        if let Some(ps) = ph_guard.take() {
                            if ps.target_layer == layer_idx {
                                // Join the prefetch thread (should be fast -- I/O already done).
                                match ps.handle.join() {
                                    Ok(results) => {
                                        for (eid, result) in results {
                                            if let Ok((data, slices)) = result {
                                                // Find which slot k this expert corresponds to.
                                                for (k, &cpu_eid) in cpu_ids.iter().enumerate() {
                                                    if cpu_eid == eid && prefetched[k].is_none() {
                                                        self.expert_bytes_from_disk.fetch_add(data.len() as u64, Ordering::Relaxed);
                                                        // Also insert into cache.
                                                        if let Some(ref cm) = self.expert_cache {
                                                            let mut c = cm.lock().unwrap();
                                                            c.insert((layer_idx, eid), data.clone(), slices.clone());
                                                        }
                                                        prefetched[k] = Some((data, slices));
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        // Prefetch thread panicked -- ignore and fall back.
                                    }
                                }
                            }
                            // If target_layer didn't match, discard the stale prefetch.
                        }
                    }

                    let mut assembled = Vec::new();
                    let mut expert_gate_offs = Vec::with_capacity(top_k);
                    let mut expert_up_offs = Vec::with_capacity(top_k);
                    let mut expert_down_offs = Vec::with_capacity(top_k);
                    let mut assembly_ok = true;

                    // Two-pass assembly: prefetch hits + cache hits first, then disk for misses.
                    if let Some(ref cache_mutex) = self.expert_cache {
                        // Pass 1: collect from prefetch and cache, identify misses.
                        let mut per_expert: Vec<Option<(Vec<u8>, lumen_format::index::ExpertSlice)>> =
                            (0..top_k).map(|_| None).collect();
                        let mut miss_indices: Vec<usize> = Vec::new();

                        // Use prefetch hits first.
                        for k in 0..top_k {
                            if let Some(data_slices) = prefetched[k].take() {
                                per_expert[k] = Some(data_slices);
                            }
                        }

                        {
                            let mut cache = cache_mutex.lock().unwrap();
                            for (k, &eid) in cpu_ids.iter().enumerate() {
                                if per_expert[k].is_some() {
                                    continue; // Already from prefetch
                                }
                                let key = (layer_idx, eid);
                                if let Some((data, slices)) = cache.get_with_slices(&key) {
                                    self.expert_bytes_from_cache.fetch_add(data.len() as u64, Ordering::Relaxed);
                                    per_expert[k] = Some((data.to_vec(), slices));
                                } else {
                                    miss_indices.push(k);
                                }
                            }
                        }

                        // Pass 2: load misses from disk.
                        //
                        // When MetalIOQueue is available, use Metal IO DMA to
                        // load expert weight byte ranges directly from NVMe SSD into a
                        // Metal shared buffer, bypassing CPU memory allocation and page
                        // faults. Falls back to load_experts_parallel (pread).
                        if !miss_indices.is_empty() {
                            let loaded_via_metal_io = if let (Some(ref io_queue), Some(ref lbc_path)) =
                                (&self.metal_io_queue, &self.lbc_path)
                            {
                                // Metal IO DMA path for cache misses.
                                // Build read plans using ExpertReader's validation,
                                // then load directly via MTLIOCommandQueue.
                                if let Some(ref reader_mutex) = self.expert_reader {
                                    let reader = reader_mutex.lock().unwrap();
                                    let layer_indices = reader.layer_indices();

                                    // Validate layer and build file offsets for each miss.
                                    let mut dma_plans: Vec<(usize, u32, u64, u64, u64, u64, u64, u64,
                                        lumen_format::quantization::QuantScheme,
                                        lumen_format::quantization::QuantScheme,
                                        lumen_format::quantization::QuantScheme)> = Vec::new();
                                    let mut plans_ok = true;

                                    if let Some(layer_idx_entry) = layer_indices.get(layer_idx) {
                                        let blob_off = layer_idx_entry.layer_offset_bytes;
                                        if let Some(ref expert_entries) = layer_idx_entry.subtensors.experts {
                                            for &k in &miss_indices {
                                                let eid = cpu_ids[k] as usize;
                                                if eid < expert_entries.len() {
                                                    let es = &expert_entries[eid];
                                                    dma_plans.push((
                                                        k, cpu_ids[k],
                                                        blob_off + es.gate.offset, es.gate.length,
                                                        blob_off + es.up.offset, es.up.length,
                                                        blob_off + es.down.offset, es.down.length,
                                                        es.gate.quant, es.up.quant, es.down.quant,
                                                    ));
                                                } else {
                                                    plans_ok = false;
                                                    break;
                                                }
                                            }
                                        } else {
                                            plans_ok = false;
                                        }
                                    } else {
                                        plans_ok = false;
                                    }
                                    drop(reader);

                                    if plans_ok && !dma_plans.is_empty() {
                                        // Compute total size and build DMA ranges.
                                        let mut total_dma_bytes: u64 = 0;
                                        let mut range_info: Vec<(usize, u32, u64, u64, u64, u64,
                                            lumen_format::quantization::QuantScheme,
                                            lumen_format::quantization::QuantScheme,
                                            lumen_format::quantization::QuantScheme)> = Vec::new();
                                        let mut io_ranges: Vec<(u64, u64, u64)> = Vec::new();

                                        for &(k, eid, gate_off, gate_len, up_off, up_len, down_off, down_len,
                                              gq, uq, dq) in &dma_plans
                                        {
                                            let base = total_dma_bytes;
                                            // gate
                                            io_ranges.push((base, gate_off, gate_len));
                                            // up
                                            io_ranges.push((base + gate_len, up_off, up_len));
                                            // down
                                            io_ranges.push((base + gate_len + up_len, down_off, down_len));
                                            let expert_total = gate_len + up_len + down_len;
                                            range_info.push((k, eid, base, gate_len, up_len, down_len, gq, uq, dq));
                                            total_dma_bytes += expert_total;
                                        }

                                        // Allocate a shared Metal buffer and DMA load all ranges.
                                        if total_dma_bytes > 0 {
                                            if let Some(dma_buf) = self.device.new_buffer(total_dma_bytes as usize) {
                                                match io_queue.load_ranges_sync(
                                                    &self.device, &dma_buf, lbc_path, &io_ranges,
                                                ) {
                                                    Ok(()) => {
                                                        // DMA succeeded. Read back into per_expert
                                                        // and populate cache.
                                                        let ptr = dma_buf.contents() as *const u8;
                                                        for &(k, eid, base, gate_len, up_len, down_len, gq, uq, dq) in &range_info {
                                                            let expert_total = (gate_len + up_len + down_len) as usize;
                                                            let mut data = vec![0u8; expert_total];
                                                            unsafe {
                                                                std::ptr::copy_nonoverlapping(
                                                                    ptr.add(base as usize),
                                                                    data.as_mut_ptr(),
                                                                    expert_total,
                                                                );
                                                            }
                                                            let slices = lumen_format::index::ExpertSlice {
                                                                gate: lumen_format::index::TensorSlice {
                                                                    offset: 0,
                                                                    length: gate_len,
                                                                    quant: gq,
                                                                },
                                                                up: lumen_format::index::TensorSlice {
                                                                    offset: gate_len,
                                                                    length: up_len,
                                                                    quant: uq,
                                                                },
                                                                down: lumen_format::index::TensorSlice {
                                                                    offset: gate_len + up_len,
                                                                    length: down_len,
                                                                    quant: dq,
                                                                },
                                                            };
                                                            self.expert_bytes_from_disk.fetch_add(expert_total as u64, Ordering::Relaxed);
                                                            if let Some(ref cm) = self.expert_cache {
                                                                let mut c = cm.lock().unwrap();
                                                                c.insert((layer_idx, eid), data.clone(), slices.clone());
                                                            }
                                                            per_expert[k] = Some((data, slices));
                                                        }
                                                        true // Successfully loaded via Metal IO
                                                    }
                                                    Err(_) => false, // Fall back to pread
                                                }
                                            } else {
                                                false // Buffer allocation failed
                                            }
                                        } else {
                                            true // No bytes to load
                                        }
                                    } else {
                                        false // Plan validation failed
                                    }
                                } else {
                                    false // No expert reader
                                }
                            } else {
                                false // No Metal IO queue or no LBC path
                            };

                            // Fallback: load misses via ExpertReader pread.
                            if !loaded_via_metal_io {
                                if let Some(ref reader_mutex) = self.expert_reader {
                                    let reader = reader_mutex.lock().unwrap();
                                    let requests: Vec<(usize, u32)> = miss_indices
                                        .iter()
                                        .map(|&k| (layer_idx, cpu_ids[k]))
                                        .collect();
                                    let results = reader.load_experts_parallel(&requests);
                                    drop(reader);
                                    for (ri, &k) in miss_indices.iter().enumerate() {
                                        let eid = cpu_ids[k];
                                        match &results[ri] {
                                            Ok((data, slices)) => {
                                                self.expert_bytes_from_disk.fetch_add(data.len() as u64, Ordering::Relaxed);
                                                if let Some(ref cm) = self.expert_cache {
                                                    let mut c = cm.lock().unwrap();
                                                    c.insert((layer_idx, eid), data.clone(), slices.clone());
                                                }
                                                per_expert[k] = Some((data.clone(), slices.clone()));
                                            }
                                            Err(_) => {
                                                assembly_ok = false;
                                                break;
                                            }
                                        }
                                    }
                                } else {
                                    assembly_ok = false;
                                }
                            }
                        }

                        // Assemble buffer from collected per-expert data.
                        if assembly_ok {
                            for k in 0..top_k {
                                if let Some((ref data, ref slices)) = per_expert[k] {
                                    let base = assembled.len() as u64;
                                    expert_gate_offs.push(base + slices.gate.offset);
                                    expert_up_offs.push(base + slices.up.offset);
                                    expert_down_offs.push(base + slices.down.offset);
                                    assembled.extend_from_slice(data);
                                } else {
                                    assembly_ok = false;
                                    break;
                                }
                            }
                        }
                    } else {
                        assembly_ok = false;
                    }

                    // If assembly from cache/reader succeeded for ALL top-K experts:
                    if assembly_ok && expert_gate_offs.len() == top_k {
                        let moe_meta = CachedMoeMeta {
                            router_weight_off: base_off + router.offset,
                            expert_gate_offs,
                            expert_up_offs,
                            expert_down_offs,
                            expert_gate_quant: first.gate.quant,
                            expert_down_quant: first.down.quant,
                        };

                        let cmd2 = self.queue.new_command_buffer().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create CB2 for Option A expert FFNs".into())
                        })?;
                        match self.device.new_buffer_with_bytes(&assembled) {
                            Some(ewb) => {
                                Self::encode_moe_ffn_decode(
                                    &cmd2, pipelines, s, layer_buf, &moe_meta,
                                    None,
                                    Some(&ewb),
                                    None, 0.0,
                                    Some(&cpu_ids),
                                    true,  // Skip router (already ran in CB1)
                                    None,  // No per-layer routing weights in streaming
                                )?;
                                cmd2.commit();
                                cmd2.wait_until_completed();
                            }
                            None => {
                                // Buffer creation failed -- fall back to full blob in CB2.
                                let moe_meta_blob = CachedMoeMeta {
                                    router_weight_off: base_off + router.offset,
                                            expert_gate_offs: experts.iter().map(|e| base_off + e.gate.offset).collect(),
                                    expert_up_offs: experts.iter().map(|e| base_off + e.up.offset).collect(),
                                    expert_down_offs: experts.iter().map(|e| base_off + e.down.offset).collect(),
                                    expert_gate_quant: first.gate.quant,
                                            expert_down_quant: first.down.quant,
                                };
                                Self::encode_moe_ffn_decode(
                                    &cmd2, pipelines, s, layer_buf, &moe_meta_blob,
                                    None, None,
                                    None, 0.0,
                                    Some(&cpu_ids),
                                    true,
                                    None,  // No per-layer routing weights in streaming
                                )?;
                                cmd2.commit();
                                cmd2.wait_until_completed();
                            }
                        }
                    } else {
                        // Fallback: assemble from full blob for top-K only.
                        let moe_meta_blob = CachedMoeMeta {
                            router_weight_off: base_off + router.offset,
                            expert_gate_offs: experts.iter().map(|e| base_off + e.gate.offset).collect(),
                            expert_up_offs: experts.iter().map(|e| base_off + e.up.offset).collect(),
                            expert_down_offs: experts.iter().map(|e| base_off + e.down.offset).collect(),
                            expert_gate_quant: first.gate.quant,
                            expert_down_quant: first.down.quant,
                        };
                        let expert_blob_bytes: u64 = cpu_ids.iter()
                            .filter_map(|&eid| experts.get(eid as usize))
                            .map(|e| (e.gate.length + e.up.length + e.down.length) as u64)
                            .sum();
                        self.expert_bytes_from_blob.fetch_add(expert_blob_bytes, Ordering::Relaxed);

                        let cmd2 = self.queue.new_command_buffer().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create CB2 for Option A fallback".into())
                        })?;
                        Self::encode_moe_ffn_decode(
                            &cmd2, pipelines, s, layer_buf, &moe_meta_blob,
                            None, None,
                            None, 0.0,
                            Some(&cpu_ids),
                            true,  // Skip router
                            None,  // No per-layer routing weights in streaming
                        )?;
                        cmd2.commit();
                        cmd2.wait_until_completed();
                    }

                    // Dispatch shared expert (always-active) for Qwen3.5-MoE.
                    // The shared expert runs on every token in addition to the top-K
                    // routed experts. Its output is added to x_buf after the routed
                    // expert accumulation. Uses CB3 since CB2 is already committed.
                    if let (Some(se_gate), Some(se_up), Some(se_down)) = (
                        &st.shared_expert_gate, &st.shared_expert_up, &st.shared_expert_down,
                    ) {
                        let cmd3 = self.queue.new_command_buffer().ok_or_else(|| {
                            RuntimeError::Compute(
                                "Failed to create CB3 for Option A shared expert".into(),
                            )
                        })?;
                        Self::encode_shared_expert_ffn_decode_raw(
                            &cmd3, pipelines, s, layer_buf,
                            base_off + se_gate.offset,
                            base_off + se_up.offset,
                            base_off + se_down.offset,
                            se_gate.quant,
                            se_down.quant,
                            st.ffn_gate_inp_shexp.map(|fgis| base_off + fgis.offset),
                        )?;
                        cmd3.commit();
                        cmd3.wait_until_completed();
                    }

                    // Check if profiling phase is complete and trigger warmup.
                    self.maybe_trigger_warmup();

                    // Spawn async prefetch for layer N+1's experts.
                    // Speculative: assume layer N+1 will select the same experts.
                    // If not, the prefetch result is discarded and fallback to sync load.
                    // Uses load_experts_parallel for concurrent pread with
                    // F_NOCACHE file descriptors, increasing NVMe queue depth.
                    let next_layer = layer_idx + 1;
                    if next_layer < s.num_layers && self.lbc_path.is_some() {
                        let lbc_path = self.lbc_path.as_ref().unwrap().clone();
                        let prefetch_ids = cpu_ids.clone();
                        let prefetch_layer = next_layer;

                        let handle = std::thread::spawn(move || {
                            let mut results = Vec::new();
                            match ExpertReader::open(&lbc_path) {
                                Ok(reader) => {
                                    let requests: Vec<(usize, u32)> = prefetch_ids
                                        .iter()
                                        .map(|&eid| (prefetch_layer, eid))
                                        .collect();
                                    let par_results = reader.load_experts_parallel(&requests);
                                    for (i, result) in par_results.into_iter().enumerate() {
                                        results.push((prefetch_ids[i], result));
                                    }
                                }
                                Err(_) => {
                                    // Reader open failed -- return empty results.
                                }
                            }
                            results
                        });

                        let ps = PrefetchState {
                            target_layer: next_layer,
                            expert_ids: cpu_ids,
                            handle,
                        };
                        *self.prefetch_handle.lock().unwrap() = Some(ps);
                    }

                    // Mark that we already committed synchronously -- clear last_async_cmd.
                    s.last_async_cmd = None;
                    s.gpu_x_valid = true;
                    kv.seq_len = new_seq_len;
                    return Ok(());
                }

                // ==============================================================
                // Option B: Original full-dispatch path (all experts)
                // ==============================================================
                let cache_assembled_buf: Option<MetalBuffer> = if let Some(ref cache_mutex) = self.expert_cache {
                    let mut cache = cache_mutex.lock().unwrap();

                    // Classify each expert as cached or not.
                    let mut cached_mask = vec![false; num_experts];
                    let mut num_cached = 0usize;
                    for e in 0..num_experts {
                        if cache.contains(&(layer_idx, e as u32)) {
                            cached_mask[e] = true;
                            num_cached += 1;
                        }
                    }

                    // Build is_cached GPU buffer for biased routing.
                    // Only created when cache_bias_lambda > 0 and warmup is complete.
                    let is_cached_buf: Option<MetalBuffer> = if self.cache_bias_lambda > 0.0
                        && self.warmup_complete.load(Ordering::Relaxed)
                    {
                        let is_cached_data: Vec<u8> = cached_mask.iter()
                            .map(|&c| if c { 1u8 } else { 0u8 })
                            .collect();
                        self.device.new_buffer_with_bytes(&is_cached_data)
                    } else {
                        None
                    };
                    let bias_lambda = self.cache_bias_lambda;

                    if num_cached == num_experts {
                        // Tier 1: All experts cached -- assemble from cache only.
                        let mut assembled = Vec::new();
                        let mut expert_gate_offs = Vec::with_capacity(num_experts);
                        let mut expert_up_offs = Vec::with_capacity(num_experts);
                        let mut expert_down_offs = Vec::with_capacity(num_experts);
                        let mut ok = true;

                        for eid in 0..num_experts {
                            let key = (layer_idx, eid as u32);
                            if let Some((data, slices)) = cache.get_with_slices(&key) {
                                let base = assembled.len() as u64;
                                expert_gate_offs.push(base + slices.gate.offset);
                                expert_up_offs.push(base + slices.up.offset);
                                expert_down_offs.push(base + slices.down.offset);
                                assembled.extend_from_slice(&data);
                            } else {
                                ok = false;
                                break;
                            }
                        }

                        if ok {
                            // Tier 1 -- all bytes from cache.
                            self.expert_bytes_from_cache.fetch_add(
                                assembled.len() as u64, Ordering::Relaxed,
                            );
                            let moe_meta = CachedMoeMeta {
                                router_weight_off: base_off + router.offset,
                                    expert_gate_offs,
                                expert_up_offs,
                                expert_down_offs,
                                expert_gate_quant: first.gate.quant,
                                    expert_down_quant: first.down.quant,
                            };
                            drop(cache);
                            match self.device.new_buffer_with_bytes(&assembled) {
                                Some(buf) => {
                                    Self::encode_moe_ffn_decode(
                                        &cmd, pipelines, s, layer_buf, &moe_meta,
                                        None,
                                        Some(&buf),
                                        is_cached_buf.as_ref(), bias_lambda,
                                        None,  // Option A handled below after readback
                                        false, // Do not skip router
                                        None,  // No per-layer routing weights in streaming
                                    )?;
                                    Some(buf)
                                }
                                None => None,
                            }
                        } else {
                            None
                        }
                    } else if num_cached > 0 && self.expert_reader.is_some() {
                        // Tier 2: Some experts cached, load misses via ExpertReader.
                        // This avoids reading the full layer blob for a partial hit.
                        let mut assembled = Vec::new();
                        let mut expert_gate_offs = Vec::with_capacity(num_experts);
                        let mut expert_up_offs = Vec::with_capacity(num_experts);
                        let mut expert_down_offs = Vec::with_capacity(num_experts);
                        let mut ok = true;
                        let mut tier2_cache_bytes = 0u64;
                        let mut tier2_disk_bytes = 0u64;

                        // Collect cached experts first (holding cache lock).
                        let mut per_expert: Vec<Option<(Vec<u8>, lumen_format::index::ExpertSlice)>> =
                            (0..num_experts).map(|_| None).collect();
                        for eid in 0..num_experts {
                            if cached_mask[eid] {
                                let key = (layer_idx, eid as u32);
                                if let Some((data, slices)) = cache.get_with_slices(&key) {
                                    tier2_cache_bytes += data.len() as u64;
                                    per_expert[eid] = Some((data.to_vec(), slices));
                                }
                            }
                        }
                        drop(cache); // Release cache lock before disk I/O.

                        // Load missing experts via ExpertReader.
                        {
                            let mut reader = self.expert_reader.as_ref().unwrap().lock().unwrap();
                            for eid in 0..num_experts {
                                if per_expert[eid].is_none() {
                                    match reader.load_expert(layer_idx, eid as u32) {
                                        Ok((data, slices)) => {
                                            tier2_disk_bytes += data.len() as u64;
                                            // Insert into cache for future tokens.
                                            if let Some(ref cm) = self.expert_cache {
                                                let mut c = cm.lock().unwrap();
                                                c.insert((layer_idx, eid as u32), data.clone(), slices.clone());
                                            }
                                            per_expert[eid] = Some((data, slices));
                                        }
                                        Err(_e) => {
                                            ok = false;
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        // Tier 2 I/O counters.
                        self.expert_bytes_from_cache.fetch_add(tier2_cache_bytes, Ordering::Relaxed);
                        self.expert_bytes_from_disk.fetch_add(tier2_disk_bytes, Ordering::Relaxed);

                        if ok {
                            for eid in 0..num_experts {
                                let (ref data, ref slices) = per_expert[eid].as_ref().unwrap();
                                let base = assembled.len() as u64;
                                expert_gate_offs.push(base + slices.gate.offset);
                                expert_up_offs.push(base + slices.up.offset);
                                expert_down_offs.push(base + slices.down.offset);
                                assembled.extend_from_slice(data);
                            }

                            let moe_meta = CachedMoeMeta {
                                router_weight_off: base_off + router.offset,
                                    expert_gate_offs,
                                expert_up_offs,
                                expert_down_offs,
                                expert_gate_quant: first.gate.quant,
                                    expert_down_quant: first.down.quant,
                            };
                            match self.device.new_buffer_with_bytes(&assembled) {
                                Some(buf) => {
                                    Self::encode_moe_ffn_decode(
                                        &cmd, pipelines, s, layer_buf, &moe_meta,
                                        None,
                                        Some(&buf),
                                        is_cached_buf.as_ref(), bias_lambda,
                                        None,  // Option A handled below after readback
                                        false, // Do not skip router
                                        None,  // No per-layer routing weights in streaming
                                    )?;
                                    Some(buf)
                                }
                                None => None,
                            }
                        } else {
                            None // ExpertReader failed, fall back to full blob
                        }
                    } else {
                        None // No cached experts or no reader
                    }
                } else {
                    None // No expert cache configured
                };

                // Tier 3 fallback: expert weights from full layer blob.
                if cache_assembled_buf.is_none() {
                    // Tier 3 -- all expert bytes from full blob.
                    let expert_blob_bytes: u64 = experts.iter()
                        .map(|e| (e.gate.length + e.up.length + e.down.length) as u64)
                        .sum();
                    self.expert_bytes_from_blob.fetch_add(expert_blob_bytes, Ordering::Relaxed);

                    let moe_meta = CachedMoeMeta {
                        router_weight_off: base_off + router.offset,
                        expert_gate_offs: experts.iter().map(|e| base_off + e.gate.offset).collect(),
                        expert_up_offs: experts.iter().map(|e| base_off + e.up.offset).collect(),
                        expert_down_offs: experts.iter().map(|e| base_off + e.down.offset).collect(),
                        expert_gate_quant: first.gate.quant,
                        expert_down_quant: first.down.quant,
                    };
                    Self::encode_moe_ffn_decode(
                        &cmd, pipelines, s, layer_buf, &moe_meta,
                        None,
                        None,
                        None, 0.0,  // No cache bias in Tier 3 fallback
                        None,  // No Option A in Tier 3 fallback
                        false, // Do not skip router
                        None,  // No per-layer routing weights in streaming
                    )?;
                }

                true
            } else {
                return Err(RuntimeError::Compute(
                    "Model has num_experts > 0 but layer is missing router_weight/experts \
                     in SubtensorOffsets. The LBC file may need re-conversion.".into()
                ));
            }
        } else {
            false
        };

        // --- Dense FFN path: only executed for non-MoE layers ---
        if !moe_handled {

        // --- Fused Gate + Up + SwiGLU (single dispatch for Q8_0, deferred reduction) ---
        // Prefill uses original fused kernel (batched GEMM style, 4 rows/TG)
        if st.w_gate.quant == QuantScheme::Q8_0 && st.w_up.quant == QuantScheme::Q8_0 {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q8_0);
            enc.set_buffer(layer_buf, w_gate_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(&s.gate_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
            enc.set_buffer(layer_buf, w_up_off, 5);
            enc.dispatch_threadgroups(
                MTLSize::new(((inter_dim as u64) + 3) / 4, 1, 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
        } else if st.w_gate.quant == QuantScheme::Q4_0 && st.w_up.quant == QuantScheme::Q4_0 {
            // Q4_0: fused Gate + Up + SwiGLU with deferred reduction (1 row/TG)
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q4_0_deferred);
            enc.set_buffer(layer_buf, w_gate_off, 0);    // gate weights
            enc.set_buffer(&s.normed_buf, 0, 1);         // normed input x
            enc.set_buffer(&s.gate_buf, 0, 2);           // output (SwiGLU result)
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
            enc.set_buffer(layer_buf, w_up_off, 5);      // up weights
            enc.dispatch_threadgroups(
                MTLSize::new(inter_dim as u64, 1, 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
        } else {
            // Fallback: separate Gate + Up + SwiGLU for F32
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
                let in_dim_u32 = hidden_dim as u32;
                enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                enc.set_buffer(layer_buf, w_gate_off, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(&s.gate_buf, 0, 2);
                enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
                enc.dispatch_threadgroups(
                    MTLSize::new(inter_dim as u64, 1, 1),
                    MTLSize::new(matmul_tg_size, 1, 1),
                );
                enc.set_buffer(layer_buf, w_up_off, 0);
                enc.set_buffer(&s.up_buf, 0, 2);
                enc.dispatch_threadgroups(
                    MTLSize::new(inter_dim as u64, 1, 1),
                    MTLSize::new(matmul_tg_size, 1, 1),
                );
                enc.end_encoding();
            }
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
                enc.set_pipeline_state(&pipelines.swiglu);
                enc.set_buffer(&s.gate_buf, 0, 0);
                enc.set_buffer(&s.up_buf, 0, 1);
                enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 2);
                let tg = 256u64.min(inter_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((inter_dim as u64).div_ceil(tg), 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
                enc.end_encoding();
            }
        }

                // --- Encoder 11: Down projection + Residual 2 (fused) ---
        // x_buf = W_down * gate + attn_proj_buf  (eliminates separate add_write)
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let tg_down = match st.w_down.quant {
                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_residual_nr2); 128u64 },
                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_residual_nr2); 128u64 },
                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_residual_nr2); 128u64 },
                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32_residual); matmul_tg_size },
            };
            enc.set_buffer(layer_buf, w_down_off, 0);
            enc.set_buffer(&s.gate_buf, 0, 1);
            enc.set_buffer(&s.x_buf, 0, 2);  // write directly to x_buf
            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
            enc.set_buffer(&s.attn_proj_buf, 0, 4);  // residual input
            if matches!(st.w_down.quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16) {
                let down_out_dim_u32 = hidden_dim as u32;
                enc.set_bytes(&down_out_dim_u32.to_le_bytes(), 5);
            }
            let n_tg_down = match st.w_down.quant {
                QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2,
                QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2,
                QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2,
                _ => hidden_dim as u64,
            };
            enc.dispatch_threadgroups(
                MTLSize::new(n_tg_down, 1, 1),
                MTLSize::new(tg_down, 1, 1),
            );
            enc.end_encoding();
        }

        } // end if !moe_handled (dense FFN path)

        // For MoE layers in streaming mode: commit synchronously so we can
        // read back expert_ids from the GPU and record activation patterns.
        // In streaming mode, I/O dominates latency, so the sync wait cost
        // is negligible. For non-MoE layers, continue with async commit.
        let moe_sync_needed = moe_handled
            && !s.gpu_unified_weight_buf.is_some()
            && !s.gpu_resident_layers.is_some()
            && self.expert_profiler.is_some();

        if moe_sync_needed {
            cmd.commit();
            cmd.wait_until_completed();

            // Read back expert_ids from GPU buffer and record in profiler.
            let top_k = s.moe_num_active_experts;
            if let Some(ref expert_ids_buf) = s.moe_expert_ids {
                let mut ids = vec![0u32; top_k];
                expert_ids_buf.read_u32(&mut ids);

                // Drop scratch guard before locking profiler to avoid
                // holding two locks simultaneously (even though they can't
                // deadlock, it's cleaner).
                // Actually, we still need `s` for the rest of this function,
                // so just lock the profiler while holding scratch. The profiler
                // mutex is always locked after scratch, so ordering is consistent.
                if let Some(ref profiler) = self.expert_profiler {
                    profiler.lock().unwrap().record(layer_idx, &ids);
                }

                // Populate the expert LFU cache with expert weight data from
                // the current layer blob. On future tokens, these cached weights
                // will enable selective expert loading.
                if let Some(ref cache_mutex) = self.expert_cache {
                    let blob = weights.as_bytes();
                    if let Some(ref experts) = weights.subtensors.experts {
                        let mut cache = cache_mutex.lock().unwrap();
                        for &eid in &ids {
                            let key = (layer_idx, eid);
                            if !cache.contains(&key) {
                                let eid_usize = eid as usize;
                                if eid_usize < experts.len() {
                                    let expert_slice = &experts[eid_usize];
                                    // Extract gate+up+down bytes from the layer blob.
                                    let gate_start = expert_slice.gate.offset as usize;
                                    let gate_end = gate_start + expert_slice.gate.length as usize;
                                    let up_start = expert_slice.up.offset as usize;
                                    let up_end = up_start + expert_slice.up.length as usize;
                                    let down_start = expert_slice.down.offset as usize;
                                    let down_end = down_start + expert_slice.down.length as usize;

                                    if gate_end <= blob.len() && up_end <= blob.len() && down_end <= blob.len() {
                                        let mut data = Vec::with_capacity(
                                            expert_slice.gate.length as usize
                                            + expert_slice.up.length as usize
                                            + expert_slice.down.length as usize
                                        );
                                        data.extend_from_slice(&blob[gate_start..gate_end]);
                                        data.extend_from_slice(&blob[up_start..up_end]);
                                        data.extend_from_slice(&blob[down_start..down_end]);

                                        // Build local ExpertSlice with offsets relative to
                                        // the concatenated data buffer.
                                        let local_slice = lumen_format::index::ExpertSlice {
                                            gate: lumen_format::index::TensorSlice {
                                                offset: 0,
                                                length: expert_slice.gate.length,
                                                quant: expert_slice.gate.quant,
                                            },
                                            up: lumen_format::index::TensorSlice {
                                                offset: expert_slice.gate.length,
                                                length: expert_slice.up.length,
                                                quant: expert_slice.up.quant,
                                            },
                                            down: lumen_format::index::TensorSlice {
                                                offset: expert_slice.gate.length + expert_slice.up.length,
                                                length: expert_slice.down.length,
                                                quant: expert_slice.down.quant,
                                            },
                                        };
                                        cache.insert(key, data, local_slice);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Check if profiling phase is complete and trigger cache warmup.
            self.maybe_trigger_warmup();

            // Clear last_async_cmd since we already waited.
            s.last_async_cmd = None;
        } else {
            // Async commit: GPU processes this layer while CPU prepares the next.
            // Metal queue is FIFO — layers execute in order, each completing before
            // the next starts. No CPU wait needed between layers.
            cmd.commit();

            // Store command buffer reference so we can wait on it at the start of the
            // next forward pass (when layer_idx == 0 needs to write fresh embeddings).
            // Drop the previous one first — if it hasn't completed, Metal still tracks it.
            s.last_async_cmd = Some(cmd);
        }

        // Mark GPU activation as valid. compute_final will skip CPU→GPU upload.
        s.gpu_x_valid = true;

        // Update KV cache seq_len (CPU tracking only — GPU KV cache is authoritative).
        // Skip reading K,V back to CPU since the GPU KV cache has the data and
        // the CPU KV cache data is never read during Metal inference.
        kv.seq_len = new_seq_len;

        Ok(())
    }

    fn compute_final(&self, x: &ActivationBuffer) -> Result<Logits, RuntimeError> {
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Metal pipelines not initialized".into())
        })?;

        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard.as_mut().ok_or_else(|| {
            RuntimeError::Compute("Metal scratch not initialized".into())
        })?;

        let hidden_dim = s.hidden_dim;
        let vocab_size = s.vocab_size;
        let eps = s.eps;
        let norm_tg_size = s.norm_tg_size;

        // Drain any pending async GPU work before CPU reads/writes.
        if let Some(prev_cmd) = s.last_async_cmd.take() {
            prev_cmd.wait_until_completed();
        }

        // GPU activation persistence: skip CPU→GPU upload if GPU already has valid data
        if !s.gpu_x_valid {
            let x_f32 = x.as_f32_slice();
            s.x_buf.write_f32(x_f32);
        }
        // Reset for next token (embed_token creates fresh CPU buffer)
        s.gpu_x_valid = false;

        // Resolve global tensor buffers: prefer unified buffer, fall back to separate
        let (norm_buf_ref, norm_off): (&MetalBuffer, u64) =
            if let Some((_, norm_o, _)) = s.gpu_global_offsets {
                let ubuf = s.gpu_unified_weight_buf.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Unified buffer missing but global offsets set".into())
                })?;
                (ubuf, norm_o as u64)
            } else {
                let fnb = self.final_norm_buf.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Final norm buffer not initialized".into())
                })?;
                (fnb, 0u64)
            };
        let (proj_buf_ref, proj_off): (&MetalBuffer, u64) =
            if let Some((_, _, proj_o)) = s.gpu_global_offsets {
                let ubuf = s.gpu_unified_weight_buf.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Unified buffer missing but global offsets set".into())
                })?;
                (ubuf, proj_o as u64)
            } else {
                let opb = self.output_proj_buf.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("Output proj buffer not initialized".into())
                })?;
                (opb, 0u64)
            };

        // Single command buffer for final RMSNorm + logits projection (2 encoders, 1 sync)

        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for compute_final".into())
        })?;

        // --- Encoder 1: Final RMSNorm ---
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for final rmsnorm".into())
            })?;
            let dim_u32 = hidden_dim as u32;
            enc.set_pipeline_state(&pipelines.rmsnorm);
            enc.set_buffer(&s.x_buf, 0, 0);
            enc.set_buffer(norm_buf_ref, norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&dim_u32.to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(norm_tg_size, 1, 1),
            );
            enc.end_encoding();
        }

        // --- Encoder 2: Logits projection (deferred NR0=2 for Q8_0, NR0=4 for others, 128 threads) ---
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for logits projection".into())
            })?;
            let in_dim_u32 = hidden_dim as u32;
            let (proj_tg, proj_rows_per_tg) = match self.output_proj_quant {
                QuantScheme::Q8_0 => {
                    enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2);
                    (128u64, 2u64)
                }
                QuantScheme::Q4_0 => {
                    enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2);
                    (128u64, 2u64)
                }
                QuantScheme::F16 => {
                    enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                    (128u64, 2u64)
                }
                _ => {
                    enc.set_pipeline_state(&pipelines.matmul_f32_deferred);
                    (128u64, 4u64)
                }
            };
            enc.set_buffer(proj_buf_ref, proj_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(&s.logits_buf, 0, 2);
            enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
            enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 4);
            let n_tg = ((vocab_size as u64) + proj_rows_per_tg - 1) / proj_rows_per_tg;
            enc.dispatch_threadgroups(
                MTLSize::new(n_tg, 1, 1),
                MTLSize::new(proj_tg, 1, 1),
            );
            enc.end_encoding();
        }

        cmd.commit_and_wait();

        // Read logits from GPU
        let mut logits_data = vec![0.0f32; vocab_size];
        s.logits_buf.read_f32(&mut logits_data);

        Ok(Logits { data: logits_data })
    }

    fn embed_token(&self, token_id: u32) -> Result<ActivationBuffer, RuntimeError> {
        let hidden_dim = self.cached_hidden_dim;
        let vocab_size = self.cached_vocab_size;
        let tid = token_id as usize;

        if tid >= vocab_size {
            return Err(RuntimeError::Compute(format!(
                "token_id {tid} out of range (vocab_size={vocab_size})",
            )));
        }

        // GPU path: dispatch embed_token kernel to write directly to x_buf.
        // This eliminates the CPU lookup + CPU->GPU upload roundtrip per decode token.
        // The returned ActivationBuffer is a placeholder -- compute_layer will use
        // the GPU-resident x_buf directly since gpu_x_valid is set to true.
        if let (Some(_embedding_buf), Some(pipelines)) = (&self.embedding_buf, &self.pipelines) {
            let mut scratch_guard = self.scratch.lock().unwrap();
            if let Some(s) = scratch_guard.as_mut() {
                // Drain any pending async GPU work before writing to x_buf.
                if let Some(prev_cmd) = s.last_async_cmd.take() {
                    prev_cmd.wait_until_completed();
                }

                // Resolve embedding buffer: prefer unified, fall back to separate
                let (embed_buf_ref, embed_off): (&MetalBuffer, u64) =
                    if let Some((emb_o, _, _)) = s.gpu_global_offsets {
                        (s.gpu_unified_weight_buf.as_ref().unwrap(), emb_o as u64)
                    } else {
                        (self.embedding_buf.as_ref().unwrap(), 0u64)
                    };

                let token_id_u32 = token_id;
                let hidden_dim_u32 = hidden_dim as u32;

                let cmd = self.queue.new_command_buffer().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create command buffer for embed_token".into())
                })?;
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for embed_token".into())
                })?;

                // Select the appropriate kernel based on embedding quantization
                match self.embedding_quant {
                    QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.embed_token_q8_0),
                    QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.embed_token_q4_0),
                    _ => enc.set_pipeline_state(&pipelines.embed_token),
                }
                enc.set_buffer(embed_buf_ref, embed_off, 0);
                enc.set_buffer(&s.x_buf, 0, 1);
                enc.set_bytes(&token_id_u32.to_le_bytes(), 2);
                enc.set_bytes(&hidden_dim_u32.to_le_bytes(), 3);
                let tg = 256u64.min(hidden_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
                enc.end_encoding();
                // Async commit: Metal queue FIFO guarantees this embed kernel
                // completes before compute_layer's command buffer executes.
                cmd.commit();

                // Store as last_async_cmd so it gets drained when needed.
                s.last_async_cmd = Some(cmd);

                // Mark GPU x_buf as valid -- compute_layer will skip CPU->GPU upload.
                s.gpu_x_valid = true;

                // Return a placeholder ActivationBuffer. The engine passes this to
                // compute_layer, but with gpu_x_valid=true the data is ignored.
                return Ok(ActivationBuffer {
                    data: vec![0u8; hidden_dim * 4],
                    num_elements: hidden_dim,
                    dtype: ComputeDtype::F32,
                });
            }
        }

        // CPU fallback: used when Metal buffers are not yet initialized.
        let start = tid * hidden_dim;
        let end = start + hidden_dim;
        let embed = &self.embedding[start..end];

        let mut data = Vec::with_capacity(embed.len() * 4);
        data.reserve(embed.len() * 4);
        #[cfg(target_endian = "little")]
        {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    embed.as_ptr() as *const u8,
                    data.as_mut_ptr(),
                    embed.len() * 4,
                );
                data.set_len(embed.len() * 4);
            }
        }
        #[cfg(target_endian = "big")]
        {
            for &v in embed {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }

        Ok(ActivationBuffer {
            data,
            num_elements: embed.len(),
            dtype: ComputeDtype::F32,
        })
    }
}


#[cfg(test)]
mod tests;
