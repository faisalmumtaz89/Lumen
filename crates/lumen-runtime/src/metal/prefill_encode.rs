//! Batched prefill encoding methods for Metal backend.
//!
//! Extracted from mod.rs for modularity.
//! These are methods on MetalF32Backend that encode batched prefill GPU dispatch sequences.

use crate::error::RuntimeError;
use crate::kv::KvCacheView;
use crate::weight::cache::LayerView;
use crate::metal::ffi::{
    MetalBuffer, MetalCommandBuffer, MetalComputeEncoder, MTLSize,
};
use lumen_format::quantization::QuantScheme;
use super::{MetalPipelines, MetalScratch, CachedLayerMeta, MetalF32Backend, use_ggml_ported_q8_0_gemm};
use super::graph_reorder::{
    self, Access, AccessList, BufferId, LayerOp, OrderClass, gdn_concurrent_encoder_enabled,
};

/// Prefill full-attention P@V precision mode (diagnostic, no-op when unset).
///
/// Mirrors CUDA's `LUMEN_CUDA_ATTN_PRECISE` (exact-F32 P@V attention variant).
/// The default prefill P@V kernel `attention_output_tiled` truncates the softmax
/// probabilities P and V to F16 simdgroup-matrix operands (F32 accumulate). Since
/// P ∈ [0,1], the F16 mantissa cannot represent the near-1.0 dominant weight plus
/// the small tail, distorting the V-weighted sum — the same mantissa defect the
/// CUDA RCA localized to P@V.
///
/// `LUMEN_METAL_ATTN_PRECISE=2` routes the prefill P@V to the exact-F32 scalar
/// kernel `attention_output_batched`, which accumulates `(float)P * (float)V` with
/// no F16 operand truncation (QK^T stays on the default F16 tensor-core path).
/// Unset (or any other value) = byte-identical legacy `attention_output_tiled`.
#[inline]
fn metal_attn_precise_pvf32() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let on = std::env::var("LUMEN_METAL_ATTN_PRECISE").ok().as_deref() == Some("2");
        if on {
            eprintln!("[ATTN_PRECISE] LUMEN_METAL_ATTN_PRECISE=2 ACTIVE: prefill P@V -> exact-F32 attention_output_batched");
        }
        on
    })
}

/// Diagnostic-only engagement counter (no-op unless `LUMEN_METAL_ATTN_PRECISE_DBG=1`).
/// Proves at runtime that the exact-F32 P@V kernel actually dispatches on the
/// full-attention prefill layers of the loaded model (and how many times), as
/// opposed to the env-read print which only confirms the lever was enabled.
/// Emits a `[ATTN_PRECISE_DBG]` line on each engagement with the running count.
/// PRODUCES evidence only; changes no computation.
fn metal_attn_precise_dbg() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("LUMEN_METAL_ATTN_PRECISE_DBG").ok().as_deref() == Some("1")
    })
}

fn metal_attn_precise_engage(path: &str, layer_idx: usize, batch_size: usize, num_heads: usize, head_dim: usize) {
    if !metal_attn_precise_dbg() {
        return;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    eprintln!(
        "[ATTN_PRECISE_DBG] F32 P@V engaged #{n} path={path} layer={layer_idx} batch={batch_size} num_heads={num_heads} head_dim={head_dim}"
    );
}

/// Dispatch a Q8_0 × F32 → F32 batched-prefill GEMM.
///
/// When `LUMEN_METAL_GEMM_GGML_PORT=1` AND the underlying weight is Q8_0,
/// reroutes the dispatch through the ggml-ported `kernel_mul_mm_q8_0_f32_ported`
/// kernel (tile NR0=64 × NR1=32, threadgroup memory 8192 B). Otherwise
/// performs the original dispatch with the supplied grid.
///
/// `m` is the batch size, `n` is the output dim. Both are needed because the
/// ported kernel uses a different tile shape than the in-tree tiled GEMM.
#[inline]
fn dispatch_q8_0_or_orig(
    enc: &MetalComputeEncoder,
    pipelines: &MetalPipelines,
    is_q8_0: bool,
    m: u32,
    n: u32,
    orig_grid: MTLSize,
) {
    if is_q8_0 && use_ggml_ported_q8_0_gemm() {
        enc.set_pipeline_state(&pipelines.kernel_mul_mm_q8_0_f32_ported);
        enc.set_threadgroup_memory_length(8192, 0);
        // Ported kernel tile = NR0=64 (N_out) × NR1=32 (M_batch)
        // tg_pos.y indexes ceil(N/64); tg_pos.x indexes ceil(M/32)
        enc.dispatch_threadgroups(
            MTLSize::new((m as u64).div_ceil(32), (n as u64).div_ceil(64), 1),
            MTLSize::new(128, 1, 1),
        );
    } else {
        enc.dispatch_threadgroups(orig_grid, MTLSize::new(128, 1, 1));
    }
}

impl MetalF32Backend {
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
    pub(crate) fn ensure_batch_buffers(
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

            // GDN (Gated-DeltaNet) projection dims, resolved from the SSM head
            // counts on `scratch` (9B {32,16,128} → v=4096/qk=2048/qkv=8192;
            // 27B {48,16,128} → v=6144/qk=2048/qkv=10240). These are INDEPENDENT
            // of the full-attention dims above: 27B full-attn q_dim=6144 with
            // hidden=5120, while GDN qkv_dim=10240. Several batched-GDN-prefill
            // scratch buffers are SHARED with the full-attention roles (sized
            // below via max(...)), and the dedicated GDN buffers must be sized
            // from these, not from the full-attn q_dim/qkv_dim/num_heads (which
            // only coincide with the GDN dims for the 9B).
            let gdn_v_dim = scratch.gdn_num_v_heads * scratch.gdn_head_dim;
            let gdn_qk_dim = scratch.gdn_num_k_heads * scratch.gdn_head_dim;
            let gdn_qkv_dim = 2 * gdn_qk_dim + gdn_v_dim;
            let gdn_nheads = scratch.gdn_num_v_heads;

            scratch.batch_x_buf = Some(make(batch_size * hidden_dim)?);
            scratch.batch_normed_buf = Some(make(batch_size * hidden_dim)?);
            // Q+gate fusion (Qwen3.5 full-attention layers): attn_q.weight outputs
            // 2*q_dim interleaved Q+gate, larger than qkv_dim. Allocate the max.
            // GDN also writes this buffer: Phase-1 QKV needs gdn_qkv_dim, and the
            // LEGACY (non-concurrent-encoder) path additionally stores Phase-2a
            // raw_out at byte offset batch*gdn_v_dim with width gdn_v_dim, so the
            // GDN role requires batch*max(gdn_qkv_dim, 2*gdn_v_dim). For the 9B
            // these GDN terms are <= qgate_dim, so the value is unchanged.
            let qgate_dim = q_dim * 2;
            let qkv_buf_elems = qkv_dim
                .max(qgate_dim)
                .max(gdn_qkv_dim)
                .max(2 * gdn_v_dim);
            scratch.batch_qkv_buf = Some(make(batch_size * qkv_buf_elems)?);
            scratch.batch_q_buf = Some(make(batch_size * q_dim)?);
            scratch.batch_k_buf = Some(make(batch_size * kv_dim)?);
            scratch.batch_v_buf = Some(make(batch_size * kv_dim)?);
            // attn_out_buf doubles as the GDN attn-gate holder (`gate_all_buf`),
            // written at width gdn_v_dim (= GDN value/q dim). For the 9B
            // gdn_v_dim == q_dim, so the allocation is unchanged.
            scratch.batch_attn_out_buf = Some(make(batch_size * q_dim.max(gdn_v_dim))?);
            scratch.batch_attn_proj_buf = Some(make(batch_size * hidden_dim)?);
            // Gate buf must hold max(inter_dim, q_dim) for FFN gate and Q+gate
            // fusion. It is ALSO passed to the batched GDN prefill as the
            // `scratch_buf` role: in the LEGACY path it packs alpha [batch*gdn_nheads]
            // at offset 0, beta [batch*gdn_nheads] at offset batch*gdn_nheads, and
            // conv_out [batch*gdn_qkv_dim] at offset 2*batch*gdn_nheads, so it must
            // hold batch*(2*gdn_nheads + gdn_qkv_dim). For the 9B this is <=
            // inter_dim, so the allocation is unchanged.
            let gate_buf_elems = inter_dim
                .max(q_dim)
                .max(2 * gdn_nheads + gdn_qkv_dim);
            scratch.batch_gate_buf = Some(make(batch_size * gate_buf_elems)?);
            scratch.batch_up_buf = Some(make(batch_size * inter_dim)?);
            scratch.batch_down_buf = Some(make(batch_size * hidden_dim)?);
            // Scores buffer for batched attention (f16). The scores stride is padded
            // to a multiple of 8 for half4-aligned vectorized reads in the tiled
            // attention output kernel. Allocation uses the padded stride.
            let scores_max_attend = max_seq_len.min(batch_size);
            let scores_stride_padded = (scores_max_attend + 7) & !7; // match runtime padding
            let scores_bytes = (batch_size * num_heads * scores_stride_padded).max(1) * 2; // f16 = 2 bytes
            scratch.batch_scores_buf = Some(self.device.new_buffer(scores_bytes).ok_or_else(|| {
                RuntimeError::Compute(format!("Failed to allocate batch_scores_buf of {scores_bytes} bytes"))
            })?);

            // De-aliased GDN scratch buffers. Allocated when the GDN
            // concurrent-encoder path is enabled so the GDN prefill chain can issue
            // resource-scoped barriers per role (vs whole-MTLBuffer
            // serialisation). These hold GDN-SSM roles ONLY, so they are sized
            // from the resolved GDN dims (gdn_v_dim/gdn_qkv_dim/gdn_nheads), NOT
            // the full-attention q_dim/qkv_dim/num_heads. For the 9B the GDN dims
            // equal the full-attn dims (q_dim=4096, qkv_dim=8192, num_heads=32),
            // so the allocations are byte-identical; for the 27B they are larger
            // (gdn_v_dim=6144, gdn_qkv_dim=10240, gdn_nheads=48).
            if gdn_concurrent_encoder_enabled() {
                // Phase 2a raw_out [batch * gdn_v_dim] and Phase 2b ssm_in
                // [batch * gdn_v_dim] each hold a single role across the chain.
                scratch.batch_gdn_raw_out_buf = Some(make(batch_size * gdn_v_dim)?);
                scratch.batch_gdn_ssm_in_buf = Some(make(batch_size * gdn_v_dim)?);
                // Alpha/beta tile each carry [batch * gdn_nheads] floats, where
                // gdn_nheads is the SSM V-head count (time_step_rank): 32 for 9B,
                // 48 for 27B.
                scratch.batch_gdn_alpha_buf = Some(make(batch_size * gdn_nheads)?);
                scratch.batch_gdn_beta_buf = Some(make(batch_size * gdn_nheads)?);
                // Conv1d+SiLU output is [batch * gdn_qkv_dim] (8192 for 9B,
                // 10240 for 27B).
                scratch.batch_gdn_conv_out_buf = Some(make(batch_size * gdn_qkv_dim)?);
            }

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
        let mut required_splitk_size = if max_splitk_n > 0 {
            k_splits * batch_size * max_splitk_n
        } else {
            1 // minimal allocation (4 bytes) -- no GEMM triggers Split-K
        };

        // FFN-down Split-K K64 path. When opt-in env var is set,
        // size the partial buffer for FFN-down (N=hidden_dim, M=batch_size).
        // We always allocate the worst case (k_splits=8) so the env var can
        // be re-toggled without resizing.
        let ffn_splitk_k = super::graph_reorder::ffn_down_splitk_value() as usize;
        if ffn_splitk_k > 0 {
            let want = 8usize * batch_size * hidden_dim;
            if want > required_splitk_size {
                required_splitk_size = want;
            }
        }

        // BF16 FFN-down Split-K path. Same sizing as Q8 (the partial
        // buffer holds F32 partials regardless of weight quantisation).
        let ffn_splitk_k_bf16 = super::graph_reorder::ffn_down_splitk_bf16_value() as usize;
        if ffn_splitk_k_bf16 > 0 {
            let want = 8usize * batch_size * hidden_dim;
            if want > required_splitk_size {
                required_splitk_size = want;
            }
        }

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
    pub(crate) fn splitk_splits(_m: usize, _n: usize, _k: usize, _batch_size: usize) -> u32 {
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
    pub(crate) fn max_splitk_output_dim(
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
    pub(crate) fn encode_splitk_q8_gemm(
        enc: &super::ffi::MetalComputeEncoder,
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

    /// Dispatch a Split-K K64 Q8_0 GEMM with fused residual writeback.
    ///
    /// Layout matches `encode_splitk_q8_gemm`:
    ///   Phase 1: 3D dispatch of `dequant_tiled_matmul_q8_0_k64_splitk`
    ///   Phase 2: 1D dispatch of `reduce_splitk_add_residual`
    /// The reduce kernel folds the residual into the same writeback, replacing
    /// the separate `add_write` dispatch used by the legacy splitk path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_splitk_q8_gemm_k64_residual(
        enc: &super::ffi::MetalComputeEncoder,
        pipelines: &MetalPipelines,
        w_buf: &MetalBuffer,
        w_off: u64,
        x_buf: &MetalBuffer,
        r_buf: &MetalBuffer,
        y_buf: &MetalBuffer,
        partial_buf: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        k_splits: u32,
    ) {
        const TILE_M: u64 = 32;
        const TILE_N: u64 = 32;

        // Phase 1: Split-K K64 GEMM into partial buffer (no residual).
        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_splitk);
        enc.set_threadgroup_memory_length(8192, 0);
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

        // Memory barrier: ensure all Split-K writes complete before reduce reads.
        enc.memory_barrier_with_scope(1);

        // Phase 2: Reduce partials + residual into final output in a single pass.
        enc.set_pipeline_state(&pipelines.reduce_splitk_add_residual);
        enc.set_buffer(partial_buf, 0, 0);
        enc.set_buffer(r_buf, 0, 1);
        enc.set_buffer(y_buf, 0, 2);
        enc.set_bytes(&m.to_le_bytes(), 3);
        enc.set_bytes(&n.to_le_bytes(), 4);
        enc.set_bytes(&k_splits.to_le_bytes(), 5);
        let total = (m as u64) * (n as u64);
        enc.dispatch_threadgroups(
            MTLSize::new(total.div_ceil(256), 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Dispatch a Split-K K64 BF16 GEMM with fused residual writeback.
    ///
    /// BF16 analogue of `encode_splitk_q8_gemm_k64_residual`. The Phase 2
    /// reduce kernel (`reduce_splitk_add_residual`) is identical between
    /// quantisations because it operates on F32 partials; only Phase 1 differs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_splitk_bf16_gemm_k64_residual(
        enc: &super::ffi::MetalComputeEncoder,
        pipelines: &MetalPipelines,
        w_buf: &MetalBuffer,
        w_off: u64,
        x_buf: &MetalBuffer,
        r_buf: &MetalBuffer,
        y_buf: &MetalBuffer,
        partial_buf: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        k_splits: u32,
    ) {
        const TILE_M: u64 = 32;
        const TILE_N: u64 = 32;

        // Phase 1: Split-K K64 BF16 GEMM into partial buffer.
        let aligned = m % 32 == 0 && n % 32 == 0 && k % 64 == 0;
        if aligned {
            enc.set_pipeline_state(&pipelines.bf16_matmul_k64_splitk_aligned);
        } else {
            enc.set_pipeline_state(&pipelines.bf16_matmul_k64_splitk);
        }
        enc.set_threadgroup_memory_length(8192, 0);
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

        // Memory barrier: ensure all Split-K writes complete before reduce reads.
        enc.memory_barrier_with_scope(1);

        // Phase 2: Reduce partials + residual (shared with Q8 path).
        enc.set_pipeline_state(&pipelines.reduce_splitk_add_residual);
        enc.set_buffer(partial_buf, 0, 0);
        enc.set_buffer(r_buf, 0, 1);
        enc.set_buffer(y_buf, 0, 2);
        enc.set_bytes(&m.to_le_bytes(), 3);
        enc.set_bytes(&n.to_le_bytes(), 4);
        enc.set_bytes(&k_splits.to_le_bytes(), 5);
        let total = (m as u64) * (n as u64);
        enc.dispatch_threadgroups(
            MTLSize::new(total.div_ceil(256), 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Dispatch a Split-K Q4_0 GEMM: same as Q8_0 variant but uses Q4_0 kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_splitk_q4_gemm(
        enc: &super::ffi::MetalComputeEncoder,
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
    pub(crate) fn encode_splitk_gemm_for_quant(
        enc: &super::ffi::MetalComputeEncoder,
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
    pub(crate) fn encode_layer_batched(
        &self,
        cmd: &MetalCommandBuffer,
        layer_idx: usize,
        batch_size: usize,
        weights: &LayerView,
        kv: &mut KvCacheView,
        pipelines: &MetalPipelines,
        scratch: &mut MetalScratch,
    ) -> Result<(), RuntimeError> {
        // Per-layer concurrent encoder + greedy wavefront scheduler.
        // Eligible only for Qwen3.5 full-attention layers with dense FFN, no
        // Split-K, no deep-profile (which intentionally splits encoders for
        // per-section attribution). GDN linear-attn and MoE layers continue
        // to use the legacy multi-encoder path.
        //
        // pass `outer_enc=None` so the concurrent-encoder path opens its own per-layer
        // encoder (legacy behaviour). The whole-prefill outer encoder caller
        // in `prefill.rs::prefill()` invokes `encode_layer_batched_concurrent` directly
        // with `Some(&outer_enc)`.
        if graph_reorder::concurrent_encoder_enabled() && self.concurrent_encoder_layer_eligible(scratch, weights, batch_size) {
            return self.encode_layer_batched_concurrent(
                cmd, None, layer_idx, batch_size, weights, kv, pipelines, scratch,
            );
        }

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

            // Profile section: attention RMSNorm + QKV (or Q+gate) projection.
            // Auto-split CB boundary when LUMEN_METAL_PROFILE=1.
            super::profile::set_section("attn/rmsnorm+qkv_gemm");
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            // Determinism fix: coalescing-safe barrier on encoder open.

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
                        QuantScheme::Bf16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if gemm_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::F16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Bf16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16);
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
                    dispatch_q8_0_or_orig(
                        &enc, pipelines, st.wq.quant == QuantScheme::Q8_0,
                        batch_size as u32, qgate_dim as u32,
                        MTLSize::new((qgate_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
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
                        QuantScheme::Bf16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if gemm_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::F16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Bf16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16);
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
                    dispatch_q8_0_or_orig(
                        &enc, pipelines, st.wk.quant == QuantScheme::Q8_0,
                        batch_size as u32, kv_dim as u32,
                        MTLSize::new((kv_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
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
                        QuantScheme::Bf16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if gemm_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::F16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Bf16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16);
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
                    dispatch_q8_0_or_orig(
                        &enc, pipelines, st.wv.quant == QuantScheme::Q8_0,
                        batch_size as u32, kv_dim as u32,
                        MTLSize::new((kv_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                    );
                }
                enc.end_encoding();

                // ---- 3+4. Deinterleave Q+gate, per-head Q/K RMSNorm, RoPE ----
                super::profile::set_section("attn/deinterleave+qknorm+rope");
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

                // NeoX-style (half-offset) RoPE for Qwen2/Qwen3.5 models
                let rope_b_pipe = if scratch.rope_neox {
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
                    QuantScheme::Bf16 => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bf16);
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
                dispatch_q8_0_or_orig(
                    &enc, pipelines, st.wq.quant == QuantScheme::Q8_0,
                    batch_size as u32, qkv_dim as u32,
                    MTLSize::new((qkv_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                );
            }
            enc.end_encoding();

            // ---- 3+4. Deinterleave + optional bias + RoPE (merged concurrent encoder) ----
            super::profile::set_section("attn/deinterleave+rope");
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

            // NeoX-style (half-offset) RoPE for Qwen2/Qwen3.5 models
            let rope_b_pipe = if scratch.rope_neox {
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
            // Pad scores stride to multiple of 8 for half4-aligned reads in
            // attention_output_tiled. The tiled kernel casts score rows to half4*
            // for vectorized loads; when q_head * stride is not half4-aligned
            // (stride not divisible by 4), the cast reads garbage bytes. Padding
            // to 8 ensures alignment for both half4 (4 halfs) and half8 reads.
            let scores_stride = (max_attend_len + 7) & !7; // round up to multiple of 8
            let scores_stride_u32 = scores_stride as u32;

            super::profile::set_section("attn/kvwrite+scores+softmax+output");
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

            // -------- Full-attention scores -> softmax -> output (3-pass) --------
            {
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

                // 6c. Attention output: Scores * V
                // Default: tiled GEMM (F16 simdgroup operands, F32 accumulate).
                // LUMEN_METAL_ATTN_PRECISE=2: exact-F32 scalar P@V (no F16
                // operand truncation) — diagnostic precision lever, see
                // metal_attn_precise_pvf32 (exact-F32 P@V; diagnostic, default-off).
                if metal_attn_precise_pvf32() {
                    // attention_output_batched: one thread per output element,
                    // accumulates (float)P * (float)V in F32. Same 11 buffers.
                    metal_attn_precise_engage("legacy", layer_idx, batch_size, num_heads as usize, head_dim);
                    let total_elems = (batch_size * num_heads as usize * head_dim) as u64;
                    let pv_tg = 256u64.min(total_elems).max(1);
                    enc.set_pipeline_state(&pipelines.attention_output_batched);
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
                        MTLSize::new(total_elems.div_ceil(pv_tg), 1, 1),
                        MTLSize::new(pv_tg, 1, 1),
                    );
                } else {
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
                }
            }

            enc.end_encoding();
        }

        // ---- 6d. Sigmoid gate (Q+gate fusion only) ----
        // Apply sigmoid(gate) * attn_out before Wo projection (Qwen3.5 full-attention).
        if has_qgate_fusion {
            super::profile::set_section("attn/sigmoid_gate");
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
            super::profile::set_section("attn/wo_proj+residual+ffn_norm");
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
                // add x_buf into attn_proj_buf, then FFN RMSNorm (same encoder).
                // Use a sub-label so the residual+norm work is distinguished
                // from the Wo GEMM at the report level.
                super::profile::set_section("attn/wo_residual+ffn_norm");
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
                    QuantScheme::Bf16 if q_dim % 64 == 0 && batch_size <= 4096 => {
                        if wo_aligned && q_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_residual_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_residual);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::F16 => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_f16_residual);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::Bf16 => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_residual);
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
                // Re-enabled after conv_pos fix (was disabled due to stale conv position
                // when batch_size >= buf_slots).
                let use_batched = true;

                if use_batched {
                    // Batched path: ~15 dispatches + per-token conv1d/state-update
                    // instead of 128*(15+2) per-token dispatches.
                    // FFN RMSNorm is fused into the same encoder (saves 1 encoder boundary).
                    super::profile::set_section("gdn/batched_prefill");
                    let new_conv_pos = Self::encode_batched_gdn_prefill(
                        cmd, None, pipelines, scratch, layer_buf, &gdn_meta, gdn_idx,
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
                // Qwen MoE architecture: every MoE layer adds a shared expert
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
                            QuantScheme::Bf16 => {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16);
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
                            QuantScheme::Bf16 => {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16);
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
                            QuantScheme::Bf16 => {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16);
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
                super::profile::set_section("ffn/gate_gemm");
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
                        QuantScheme::Bf16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if ffn_aligned && hidden_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::F16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Bf16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16);
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
                    dispatch_q8_0_or_orig(
                        &enc, pipelines, st.w_gate.quant == QuantScheme::Q8_0,
                        batch_size as u32, inter_dim as u32,
                        MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                    );
                }
                enc.end_encoding();
                // Up
                super::profile::set_section("ffn/up_gemm");
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
                        QuantScheme::Bf16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                            if ffn_aligned && hidden_dim % 64 == 0 {
                                enc2.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_aligned);
                            } else {
                                enc2.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64);
                            }
                            enc2.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::F16 => {
                            enc2.set_pipeline_state(&pipelines.tiled_matmul_f16);
                            enc2.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Bf16 => {
                            enc2.set_pipeline_state(&pipelines.tiled_matmul_bf16);
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
                    dispatch_q8_0_or_orig(
                        &enc2, pipelines, st.w_up.quant == QuantScheme::Q8_0,
                        batch_size as u32, inter_dim as u32,
                        MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                    );
                }
                enc2.end_encoding();
            } else if {
                // fast path: joint gate+up+SwiGLU fused kernel.
                // Requirements: Q8_0 for both gate and up, K (= hidden_dim) divisible
                // by 64, batch_size <= 4096. Default ON via
                // `super::graph_reorder::ffn_gate_up_swiglu_fused_q8_enabled()`;
                // explicit `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED=0` or the
                // master kill-switch `LUMEN_METAL_DEFAULTS_OFF=1` restore
                // the legacy default-OFF behaviour.
                // Output is written directly to gate_buf in the layout expected by the
                // FFN down GEMM, eliminating the up_buf hidden_inter write and the
                // standalone swiglu_batched dispatch.
                let env_on = super::graph_reorder::ffn_gate_up_swiglu_fused_q8_enabled();
                let both_q8 = st.w_gate.quant == QuantScheme::Q8_0 && st.w_up.quant == QuantScheme::Q8_0;
                env_on && both_q8 && hidden_dim % 64 == 0 && batch_size <= 4096
            } {
                // Joint dual-output gate+up+SwiGLU. Single dispatch into the
                // same compute encoder, no barrier with the rest of the FFN
                // (down GEMM consumes gate_buf only after this encoder ends).
                super::profile::set_section("ffn/gate_up_swiglu_fused");
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder (fused gate+up+swiglu)".into())
                })?;
                let ffn_aligned = batch_size % 32 == 0 && inter_dim % 32 == 0 && hidden_dim % 32 == 0;

                // prefer the packed-layout kernel when a repacked buffer
                // is available for this layer. The packed kernel consumes a single
                // paired gate+up buffer instead of two separate weight regions.
                let packed_buf: Option<&MetalBuffer> = scratch.repacked_ffn_gate_up
                    .get(layer_idx)
                    .and_then(|opt| opt.as_ref());
                if let Some(buf_gu) = packed_buf {
                    if ffn_aligned && hidden_dim % 64 == 0 {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_gate_up_swiglu_fused_packed_aligned);
                    } else {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_gate_up_swiglu_fused_packed);
                    }
                    enc.set_threadgroup_memory_length(12288, 0);
                    enc.set_buffer(buf_gu, 0, 0);   // paired packed buffer
                    enc.set_buffer(normed_buf, 0, 1);
                    enc.set_buffer(gate_buf, 0, 2);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&k_u32.to_le_bytes(), 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else {
                    if ffn_aligned && hidden_dim % 64 == 0 {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_gate_up_swiglu_fused_aligned);
                    } else {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_gate_up_swiglu_fused);
                    }
                    // 12 KB shmem: sa[32*64] + sb_gate[32*64] + sb_up[32*64] = 6144 halfs.
                    enc.set_threadgroup_memory_length(12288, 0);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(normed_buf, 0, 1);
                    enc.set_buffer(gate_buf, 0, 2);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&k_u32.to_le_bytes(), 5);
                    enc.set_buffer(layer_buf, w_up_off, 6);
                    enc.dispatch_threadgroups(
                        MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
                enc.end_encoding();
            } else if {
                // fast path: Q4_0 port of the fused kernel.
                // Requirements: Q4_0 for both gate and up, hidden_dim % 64 == 0,
                // batch_size <= 4096. Default ON via
                // `super::graph_reorder::ffn_gate_up_swiglu_fused_q4_enabled()`;
                // explicit `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_Q4=0` or the
                // master kill-switch `LUMEN_METAL_DEFAULTS_OFF=1` restore
                // the legacy default-OFF behaviour.
                let env_on = super::graph_reorder::ffn_gate_up_swiglu_fused_q4_enabled();
                let both_q4 = st.w_gate.quant == QuantScheme::Q4_0 && st.w_up.quant == QuantScheme::Q4_0;
                env_on && both_q4 && hidden_dim % 64 == 0 && batch_size <= 4096
            } {
                super::profile::set_section("ffn/gate_up_swiglu_fused_q4");
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder (fused gate+up+swiglu q4)".into())
                })?;
                let ffn_aligned = batch_size % 32 == 0 && inter_dim % 32 == 0 && hidden_dim % 32 == 0;

                // prefer the packed-layout Q4 kernel when a repacked buffer
                // is available for this layer. The packed kernel consumes a single
                // paired gate+up buffer instead of two separate weight regions.
                let packed_buf_q4: Option<&MetalBuffer> = scratch.repacked_ffn_gate_up_q4
                    .get(layer_idx)
                    .and_then(|opt| opt.as_ref());
                if let Some(buf_gu) = packed_buf_q4 {
                    if ffn_aligned && hidden_dim % 64 == 0 {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_gate_up_swiglu_fused_packed_aligned);
                    } else {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_gate_up_swiglu_fused_packed);
                    }
                    enc.set_threadgroup_memory_length(12288, 0);
                    enc.set_buffer(buf_gu, 0, 0);   // paired packed buffer
                    enc.set_buffer(normed_buf, 0, 1);
                    enc.set_buffer(gate_buf, 0, 2);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&k_u32.to_le_bytes(), 5);
                    enc.dispatch_threadgroups(
                        MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else {
                    if ffn_aligned && hidden_dim % 64 == 0 {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_gate_up_swiglu_fused_aligned);
                    } else {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_gate_up_swiglu_fused);
                    }
                    enc.set_threadgroup_memory_length(12288, 0);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(normed_buf, 0, 1);
                    enc.set_buffer(gate_buf, 0, 2);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&k_u32.to_le_bytes(), 5);
                    enc.set_buffer(layer_buf, w_up_off, 6);
                    enc.dispatch_threadgroups(
                        MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
                enc.end_encoding();
            } else if {
                // fast path: BF16 port of the fused kernel.
                // Requirements: BF16 for both gate and up, hidden_dim % 64 == 0,
                // batch_size <= 4096. Default OFF; opt-in via
                // `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_BF16=1`.
                let env_on = super::graph_reorder::ffn_gate_up_swiglu_fused_bf16_enabled();
                let both_bf16 = st.w_gate.quant == QuantScheme::Bf16
                    && st.w_up.quant == QuantScheme::Bf16;
                env_on && both_bf16 && hidden_dim % 64 == 0 && batch_size <= 4096
            } {
                super::profile::set_section("ffn/gate_up_swiglu_fused_bf16");
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder (fused gate+up+swiglu bf16)".into())
                })?;
                let ffn_aligned = batch_size % 32 == 0 && inter_dim % 32 == 0 && hidden_dim % 32 == 0;
                // NR microtile selection. Baseline NR=2 unchanged when
                // `LUMEN_METAL_BF16_GATE_UP_NR` is unset or set to a non-{1,4} value.
                let nr = super::graph_reorder::bf16_gate_up_nr();
                // Per-variant TILE_M (the dispatch grid Y-dim divisor) and shmem size.
                // shmem layout invariants are encoded in the MSL kernel; we mirror the
                // byte count here so set_threadgroup_memory_length matches the kernel's
                // shmem declaration exactly.
                let (tile_m_dispatch, shmem_bytes): (u64, u64) = match nr {
                    1 => (16, 10240), // 5120 bfloats * 2 bytes
                    4 => (64, 16384), // 8192 bfloats * 2 bytes
                    _ => (32, 12288), // 6144 bfloats * 2 bytes (NR=2 baseline)
                };
                let pso = match (nr, ffn_aligned && hidden_dim % 64 == 0) {
                    (1, true)  => &pipelines.bf16_matmul_gate_up_swiglu_fused_nr1_aligned,
                    (1, false) => &pipelines.bf16_matmul_gate_up_swiglu_fused_nr1,
                    (4, true)  => &pipelines.bf16_matmul_gate_up_swiglu_fused_nr4_aligned,
                    (4, false) => &pipelines.bf16_matmul_gate_up_swiglu_fused_nr4,
                    (_, true)  => &pipelines.bf16_matmul_gate_up_swiglu_fused_aligned,
                    (_, false) => &pipelines.bf16_matmul_gate_up_swiglu_fused,
                };
                enc.set_pipeline_state(pso);
                enc.set_threadgroup_memory_length(shmem_bytes, 0);
                enc.set_buffer(layer_buf, w_gate_off, 0);
                enc.set_buffer(normed_buf, 0, 1);
                enc.set_buffer(gate_buf, 0, 2);
                enc.set_bytes(&m_u32.to_le_bytes(), 3);
                enc.set_bytes(&n_u32.to_le_bytes(), 4);
                enc.set_bytes(&k_u32.to_le_bytes(), 5);
                enc.set_buffer(layer_buf, w_up_off, 6);
                enc.dispatch_threadgroups(
                    MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(tile_m_dispatch), 1),
                    MTLSize::new(128, 1, 1),
                );
                enc.end_encoding();
            } else {
                // No split-K: merge Gate + Up + SwiGLU into single concurrent encoder.
                // Gate and Up read from normed_buf (same input) and write to separate
                // output buffers (gate_buf, up_buf). They can run concurrently.
                // SwiGLU reads both gate_buf and up_buf, so it must wait for both via barrier.
                //
                // DEEP-PROFILE override: when `LUMEN_METAL_PROFILE_DEEP=1` we split
                // the concurrent encoder into THREE separate compute encoders so
                // the per-section profiler can attribute time to gate / up / swiglu
                // individually. Adds barrier-style serialisation (no longer truly
                // concurrent) so absolute prefill is slower under deep mode, but
                // the relative ms/section is now observable.
                let deep_profile = super::profile::is_enabled()
                    && std::env::var("LUMEN_METAL_PROFILE_DEEP").ok().as_deref() == Some("1");
                if deep_profile {
                    super::profile::set_section("ffn/gate_gemm");
                } else {
                    super::profile::set_section("ffn/gate+up+swiglu_concurrent");
                }
                let enc = if deep_profile {
                    cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder (deep-profile)".into())
                    })?
                } else {
                    cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create concurrent encoder".into())
                    })?
                };

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
                    QuantScheme::Bf16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                        if ffn_aligned && hidden_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::F16 => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::Bf16 => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bf16);
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
                dispatch_q8_0_or_orig(
                    &enc, pipelines, st.w_gate.quant == QuantScheme::Q8_0,
                    batch_size as u32, inter_dim as u32,
                    MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                );

                // Deep-profile: split gate from up so each is its own profiler
                // section. Concurrent encoder semantics are lost (serialised),
                // making absolute timing slower but per-kernel observability
                // possible. Production path is unchanged when deep_profile=false.
                let enc = if deep_profile {
                    enc.end_encoding();
                    super::profile::set_section("ffn/up_gemm");
                    cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder (deep-profile up)".into())
                    })?
                } else {
                    enc
                };

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
                    QuantScheme::Bf16 if hidden_dim % 64 == 0 && batch_size <= 4096 => {
                        if ffn_aligned && hidden_dim % 64 == 0 {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_aligned);
                        } else {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64);
                        }
                        enc.set_threadgroup_memory_length(8192, 0);
                    }
                    QuantScheme::F16 => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_f16);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    QuantScheme::Bf16 => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bf16);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                    _ => {
                        enc.set_pipeline_state(&pipelines.tiled_matmul_bytes_f32);
                        enc.set_threadgroup_memory_length(4096, 0);
                    }
                }
                enc.set_buffer(layer_buf, w_up_off, 0);
                // Deep-profile: when split, we must re-bind buffer args at idx 1
                // because each fresh encoder loses its arg table.
                if deep_profile {
                    enc.set_buffer(normed_buf, 0, 1);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&k_u32.to_le_bytes(), 5);
                }
                // normed_buf already set at index 1 (when not deep_profile)
                enc.set_buffer(up_buf, 0, 2);
                // M, N, K already set at indices 3, 4, 5
                dispatch_q8_0_or_orig(
                    &enc, pipelines, st.w_up.quant == QuantScheme::Q8_0,
                    batch_size as u32, inter_dim as u32,
                    MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                );

                // Deep-profile: split up from swiglu.
                let enc = if deep_profile {
                    enc.end_encoding();
                    super::profile::set_section("ffn/swiglu_concurrent");
                    cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder (deep-profile swiglu)".into())
                    })?
                } else {
                    // Barrier: Gate and Up must complete before SwiGLU reads gate_buf + up_buf
                    enc.memory_barrier_with_scope(1); // MTLBarrierScope.buffers
                    enc
                };

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
        super::profile::set_section("ffn/swiglu_splitk");
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

            // opt-in Split-K K64 path for FFN-down on Q8 weights.
            // Eligibility: env var > 0, Q8 quant, M<=192, K>=8192, K%64==0.
            // K%64==0 is required because the K64 kernel processes 64 elements
            // per tile and k_per_split is rounded to that multiple; FFN-down
            // K=12288 satisfies this for Qwen3.5-9B.
            let ffn_splitk_k_env = super::graph_reorder::ffn_down_splitk_value();
            let use_ffn_down_splitk = ffn_splitk_k_env > 0
                && matches!(st.w_down.quant, QuantScheme::Q8_0)
                && batch_size <= 192
                && inter_dim >= 8192
                && inter_dim % 64 == 0;

            // opt-in Split-K K64 path for FFN-down on BF16 weights.
            // Same eligibility shape as the Q8 path. Default OFF.
            let ffn_splitk_bf16_env = super::graph_reorder::ffn_down_splitk_bf16_value();
            let use_ffn_down_splitk_bf16 = ffn_splitk_bf16_env > 0
                && matches!(st.w_down.quant, QuantScheme::Bf16)
                && batch_size <= 192
                && inter_dim >= 8192
                && inter_dim % 64 == 0;

            let splitk = match st.w_down.quant {
                QuantScheme::Q8_0 => Self::splitk_splits(batch_size, hidden_dim, inter_dim, batch_size),
                QuantScheme::Q4_0 => Self::splitk_splits(batch_size, hidden_dim, inter_dim, batch_size),
                _ => 0,
            };
            super::profile::set_section("ffn/down_gemm+residual");
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            if use_ffn_down_splitk {
                // path: K64-Split-K with fused residual writeback.
                Self::encode_splitk_q8_gemm_k64_residual(
                    &enc, pipelines, layer_buf, w_down_off, gate_buf, attn_proj_buf,
                    x_buf, splitk_partial_buf, m_u32, n_u32, k_u32, ffn_splitk_k_env,
                );
                enc.end_encoding();
            } else if use_ffn_down_splitk_bf16 {
                // path: BF16 K64-Split-K with shared reduce kernel.
                Self::encode_splitk_bf16_gemm_k64_residual(
                    &enc, pipelines, layer_buf, w_down_off, gate_buf, attn_proj_buf,
                    x_buf, splitk_partial_buf, m_u32, n_u32, k_u32, ffn_splitk_bf16_env,
                );
                enc.end_encoding();
            } else if splitk > 0 {
                // Split-K path: GEMM to down_buf, then separate residual add
                Self::encode_splitk_gemm_for_quant(
                    &enc, pipelines, st.w_down.quant, layer_buf, w_down_off, gate_buf,
                    down_buf, splitk_partial_buf, m_u32, n_u32, k_u32, splitk,
                );
                enc.end_encoding();
                // x_buf = attn_proj_buf + down_buf
                super::profile::set_section("ffn/down_residual_add");
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

                // prefer packed-layout FFN-down kernel when available.
                let packed_down_buf: Option<&MetalBuffer> = if matches!(st.w_down.quant, QuantScheme::Q8_0)
                    && inter_dim % 64 == 0
                    && batch_size <= 4096
                {
                    scratch.repacked_ffn_down
                        .get(layer_idx)
                        .and_then(|opt| opt.as_ref())
                } else {
                    None
                };

                // prefer packed-layout Q4_0 FFN-down kernel when available.
                let packed_down_buf_q4: Option<&MetalBuffer> = if packed_down_buf.is_none()
                    && matches!(st.w_down.quant, QuantScheme::Q4_0)
                    && inter_dim % 64 == 0
                    && batch_size <= 4096
                {
                    scratch.repacked_ffn_down_q4
                        .get(layer_idx)
                        .and_then(|opt| opt.as_ref())
                } else {
                    None
                };

                if let Some(buf_d) = packed_down_buf {
                    if down_aligned && inter_dim % 64 == 0 {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_packed_aligned);
                    } else {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_packed);
                    }
                    enc.set_threadgroup_memory_length(8192, 0);
                    enc.set_buffer(buf_d, 0, 0);
                    enc.set_buffer(gate_buf, 0, 1);
                    enc.set_buffer(x_buf, 0, 2);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&k_u32.to_le_bytes(), 5);
                    enc.set_buffer(attn_proj_buf, 0, 6);
                    enc.dispatch_threadgroups(
                        MTLSize::new((hidden_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else if let Some(buf_d) = packed_down_buf_q4 {
                    if down_aligned && inter_dim % 64 == 0 {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched_packed_aligned);
                    } else {
                        enc.set_pipeline_state(&pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched_packed);
                    }
                    enc.set_threadgroup_memory_length(8192, 0);
                    enc.set_buffer(buf_d, 0, 0);
                    enc.set_buffer(gate_buf, 0, 1);
                    enc.set_buffer(x_buf, 0, 2);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&k_u32.to_le_bytes(), 5);
                    enc.set_buffer(attn_proj_buf, 0, 6);
                    enc.dispatch_threadgroups(
                        MTLSize::new((hidden_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else {
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
                        QuantScheme::Bf16 if inter_dim % 64 == 0 && batch_size <= 4096 => {
                            if down_aligned && inter_dim % 64 == 0 {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_residual_aligned);
                            } else {
                                enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_k64_residual);
                            }
                            enc.set_threadgroup_memory_length(8192, 0);
                        }
                        QuantScheme::F16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_f16_residual);
                            enc.set_threadgroup_memory_length(4096, 0);
                        }
                        QuantScheme::Bf16 => {
                            enc.set_pipeline_state(&pipelines.tiled_matmul_bf16_residual);
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
                }
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

    // ========================================================================
    // path: per-layer concurrent encoder with greedy wavefront
    // scheduler. Current concurrent-encoder scope: Qwen3.5 full-attention layers + dense FFN.
    // ========================================================================

    /// Returns true when this layer is in the concurrent-encoder fast-path subset.
    ///
    /// Excludes:
    /// - GDN linear-attention layers (`layer_type == Some(1)`).
    /// - MoE layers (`moe_num_experts > 0`).
    /// - Non-Qwen3.5 layers (no Q+gate fusion).
    /// - Any layer where Split-K would fire (none today; future-proof).
    /// - `LUMEN_METAL_PROFILE_DEEP=1` (intentional encoder splitting).
    fn concurrent_encoder_layer_eligible(
        &self,
        scratch: &MetalScratch,
        weights: &LayerView,
        batch_size: usize,
    ) -> bool {
        let st = &weights.subtensors;
        let is_linear_attn = st.layer_type == Some(1);
        if is_linear_attn { return false; }
        if scratch.moe_num_experts > 0 { return false; }
        let has_qgate_fusion = st.attn_q_norm.is_some();
        if !has_qgate_fusion { return false; }
        // Defensive: Split-K is currently disabled (returns 0). If a future
        // tuning re-enables it for any GEMM site, fall back to legacy.
        let kv_dim = scratch.kv_dim;
        let q_dim = scratch.q_dim;
        let inter_dim = scratch.inter_dim;
        let hidden_dim = scratch.hidden_dim;
        let qkv_dim = scratch.qkv_dim;
        if Self::splitk_splits(batch_size, qkv_dim,    hidden_dim, batch_size) > 0 { return false; }
        if Self::splitk_splits(batch_size, kv_dim,     hidden_dim, batch_size) > 0 { return false; }
        if Self::splitk_splits(batch_size, hidden_dim, q_dim,      batch_size) > 0 { return false; }
        if Self::splitk_splits(batch_size, inter_dim,  hidden_dim, batch_size) > 0 { return false; }
        if Self::splitk_splits(batch_size, hidden_dim, inter_dim,  batch_size) > 0 { return false; }
        // Deep-profile splits encoders intentionally; preserve that.
        if super::profile::is_enabled() &&
            std::env::var("LUMEN_METAL_PROFILE_DEEP").ok().as_deref() == Some("1") {
            return false;
        }
        true
    }

    /// returns true when this layer can be emitted into the
    /// whole-prefill outer concurrent encoder.
    ///
    /// Two layer kinds qualify:
    ///   * **Concurrent-encoder eligible full-attn / dense FFN** (`concurrent_encoder_layer_eligible`):
    ///     dispatches go through `encode_layer_batched_concurrent`, which already
    ///     accepts an outer-encoder argument (Edit 1).
    ///   * **GDN linear-attn with concurrent-encoder active** (`gdn_concurrent_encoder_enabled()`
    ///     plus all five de-aliased buffers present, and not deep-profile):
    ///     dispatches go through `encode_batched_gdn_prefill`, which
    ///     accepts an outer-encoder argument (Edit 2). The legacy GDN
    ///     legacy per-layer path opens its own blit + compute encoders interleaved,
    ///     incompatible with an outer encoder.
    ///
    /// Returns false for MoE, sequential GDN (legacy per-layer path), non-Qwen3.5
    /// layers, deep-profile, and any other path that interleaves encoder
    /// open/close mid-layer.
    pub(crate) fn layer_outer_eligible(
        &self,
        scratch: &MetalScratch,
        weights: &LayerView,
        batch_size: usize,
    ) -> bool {
        let st = &weights.subtensors;
        let is_linear_attn = st.layer_type == Some(1);
        if is_linear_attn {
            // GDN: require gdn_concurrent_encoder_enabled + all 5 buffers + no deep-profile.
            if !graph_reorder::gdn_concurrent_encoder_enabled() { return false; }
            if super::profile::is_gdn_deep_enabled() { return false; }
            if scratch.batch_gdn_raw_out_buf.is_none()
                || scratch.batch_gdn_ssm_in_buf.is_none()
                || scratch.batch_gdn_alpha_buf.is_none()
                || scratch.batch_gdn_beta_buf.is_none()
                || scratch.batch_gdn_conv_out_buf.is_none()
            {
                return false;
            }
            true
        } else {
            // Full-attn: defer to `concurrent_encoder_layer_eligible`. That gates on
            // deep-profile already.
            graph_reorder::concurrent_encoder_enabled() && self.concurrent_encoder_layer_eligible(scratch, weights, batch_size)
        }
    }

    /// Concurrent-encoder fast path: emit the layer's full op plan into a single per-layer
    /// compute encoder using the greedy wavefront scheduler.
    ///
    /// In production this is a `new_concurrent_compute_encoder()`; in
    /// `LUMEN_METAL_CONCURRENT_ENCODER_VALIDATE=1` mode it is a `new_compute_encoder()` and
    /// barriers are suppressed (serial encoders already serialise on the
    /// GPU). Used to isolate emit-helper bugs from scheduling bugs.
    ///
    /// `outer_enc`:
    ///   `None` → per-layer encoder lifecycle (legacy path: open,
    ///             emit plan, end_encoding).
    ///   `Some(enc)` → emit the plan into `enc` provided by the whole-prefill
    ///                 caller in `prefill.rs::prefill()`. Skip both the
    ///                 per-layer `new_*_compute_encoder()` and the matching
    ///                 `end_encoding()`. Cross-layer hazards are handled by
    ///                 the caller via `graph_reorder::emit_cross_layer_barrier`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_layer_batched_concurrent(
        &self,
        cmd: &MetalCommandBuffer,
        outer_enc: Option<&MetalComputeEncoder>,
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
        let inter_dim = scratch.inter_dim;
        let head_dim = scratch.head_dim;
        let num_heads = scratch.num_heads;
        let num_kv_heads = scratch.num_kv_heads;
        let eps = scratch.eps;
        let attn_scale = scratch.attn_scale;
        let max_seq_len = scratch.max_seq_len;
        let norm_tg_size = scratch.norm_tg_size;
        let seq_pos_start = kv.seq_len;
        let rope_neox = scratch.rope_neox;
        let rope_half_dim = scratch.rotary_dim / 2;

        let st = &weights.subtensors;

        // ---- Resolve base offset / layer buffer (same logic as legacy) ----
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
                    "concurrent-encoder path:gpu_resident_layers missing layer {}", layer_idx)));
            }
        } else {
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

        let attn_norm_off = base_off + st.attn_norm.offset;
        let wq_off = base_off + st.wq.offset;
        let wk_off = base_off + st.wk.offset;
        let wv_off = base_off + st.wv.offset;
        let wo_off = base_off + st.wo.offset;
        let ffn_norm_off = if st.ffn_norm.length == 0 {
            st.attn_post_norm.map_or(0, |s| base_off + s.offset)
        } else {
            base_off + st.ffn_norm.offset
        };
        let w_gate_off = base_off + st.w_gate.offset;
        let w_up_off = base_off + st.w_up.offset;
        let w_down_off = base_off + st.w_down.offset;
        let q_norm_off = st.attn_q_norm.map(|s| base_off + s.offset);
        let k_norm_off = st.attn_k_norm.map(|s| base_off + s.offset);

        let x_buf = scratch.batch_x_buf.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:batch_x_buf not allocated".into()))?;
        let normed_buf = scratch.batch_normed_buf.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:batch_normed_buf not allocated".into()))?;
        let qkv_buf = scratch.batch_qkv_buf.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:batch_qkv_buf not allocated".into()))?;
        let q_buf = scratch.batch_q_buf.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:batch_q_buf not allocated".into()))?;
        let k_buf = scratch.batch_k_buf.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:batch_k_buf not allocated".into()))?;
        let v_buf = scratch.batch_v_buf.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:batch_v_buf not allocated".into()))?;
        let attn_out_buf = scratch.batch_attn_out_buf.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:batch_attn_out_buf not allocated".into()))?;
        let attn_proj_buf = scratch.batch_attn_proj_buf.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:batch_attn_proj_buf not allocated".into()))?;
        let gate_buf = scratch.batch_gate_buf.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:batch_gate_buf not allocated".into()))?;
        let up_buf = scratch.batch_up_buf.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:batch_up_buf not allocated".into()))?;
        let scores_buf = scratch.batch_scores_buf.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:batch_scores_buf not allocated".into()))?;
        // BF16 FFN-down Split-K K64 needs the Split-K partial buffer in the concurrent-encoder
        // path. Allocation is unconditional (see ensure_batch_scratch around
        // line 188); buffer is reused across FFN-down sites that opt in.
        let splitk_partial_buf = scratch.splitk_partial_buf.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:splitk_partial_buf not allocated".into()))?;
        let k_cache = &scratch.gpu_k_cache[layer_idx];
        let v_cache = &scratch.gpu_v_cache[layer_idx];
        let rope_cos = &scratch.rope_cos_buf;
        let rope_sin = &scratch.rope_sin_buf;

        // optional packed-layout buffers for this layer (None if
        // repack disabled or not eligible). When Some, the FFN gate+up fused
        // and FFN-down dispatches below use the packed kernels instead.
        let repacked_gate_up: Option<&MetalBuffer> = scratch.repacked_ffn_gate_up
            .get(layer_idx).and_then(|opt| opt.as_ref());
        let repacked_down: Option<&MetalBuffer> = scratch.repacked_ffn_down
            .get(layer_idx).and_then(|opt| opt.as_ref());
        // optional Q4 packed buffers for this layer (None when
        // repack is disabled or the layer is not Q4_0).
        let repacked_gate_up_q4: Option<&MetalBuffer> = scratch.repacked_ffn_gate_up_q4
            .get(layer_idx).and_then(|opt| opt.as_ref());
        let repacked_down_q4: Option<&MetalBuffer> = scratch.repacked_ffn_down_q4
            .get(layer_idx).and_then(|opt| opt.as_ref());

        let qgate_dim = q_dim * 2; // Qwen3.5 attn_q.weight outputs [2*q_dim] interleaved Q+gate
        let max_attend_len = seq_pos_start + batch_size;
        let scores_stride = (max_attend_len + 7) & !7;

        const TILE_M: u64 = 32;
        const TILE_N: u64 = 32;

        // ---- Build the per-layer op plan ----
        // Buffer-ID byte ranges below use the FULL logical span of each
        // mutable buffer. This is conservative but cheap and matches the
        // design guidance ("for most mutable buffers: use the full
        // logical tensor byte span"). scores_buf uses the padded stride.
        let _ = layer_idx;
        let mut plan: Vec<LayerOp<'_>> = Vec::with_capacity(24);

        // Op 1: Attn RMSNorm: x_buf -> normed_buf
        let p_rmsnorm = &pipelines.rmsnorm_batched_bytes;
        let dim_u32 = hidden_dim as u32;
        plan.push(LayerOp {
            label: "attn_norm",
            accesses: AccessList::from_iter_inline([Access::read(BufferId::X), Access::write(BufferId::Normed)]),
            order_class: OrderClass::Free,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(p_rmsnorm);
                enc.set_buffer(x_buf, 0, 0);
                enc.set_buffer(layer_buf, attn_norm_off, 1);
                enc.set_buffer(normed_buf, 0, 2);
                enc.set_bytes(&dim_u32.to_le_bytes(), 3);
                enc.set_bytes(&eps.to_le_bytes(), 4);
                enc.dispatch_threadgroups(
                    MTLSize::new(batch_size as u64, 1, 1),
                    MTLSize::new(norm_tg_size, 1, 1),
                );
                Ok(())
            }),
        });

        // Op 2: Q+gate GEMM: normed @ wq -> qkv_buf [batch, 2*q_dim]
        let m_u32 = batch_size as u32;
        let k_u32 = hidden_dim as u32;
        plan.push(Self::concurrent_encoder_emit_gemm(
            "qgate_gemm",
            BufferId::Normed, BufferId::Qkv,
            layer_buf, wq_off, normed_buf, qkv_buf,
            pipelines, st.wq.quant, m_u32, qgate_dim as u32, k_u32,
            batch_size, qgate_dim, hidden_dim,
            TILE_M, TILE_N,
        ));

        // Op 3: K GEMM: normed @ wk -> k_buf [batch, kv_dim]
        plan.push(Self::concurrent_encoder_emit_gemm(
            "k_gemm",
            BufferId::Normed, BufferId::K,
            layer_buf, wk_off, normed_buf, k_buf,
            pipelines, st.wk.quant, m_u32, kv_dim as u32, k_u32,
            batch_size, kv_dim, hidden_dim,
            TILE_M, TILE_N,
        ));

        // Op 4: V GEMM: normed @ wv -> v_buf [batch, kv_dim]
        plan.push(Self::concurrent_encoder_emit_gemm(
            "v_gemm",
            BufferId::Normed, BufferId::V,
            layer_buf, wv_off, normed_buf, v_buf,
            pipelines, st.wv.quant, m_u32, kv_dim as u32, k_u32,
            batch_size, kv_dim, hidden_dim,
            TILE_M, TILE_N,
        ));

        // Op 5: Deinterleave Q+gate: qkv_buf -> q_buf + gate_buf
        let p_deinter = pipelines.deinterleave_qgate.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:deinterleave_qgate pipeline missing".into()))?;
        let total_q = (batch_size * q_dim) as u64;
        let tg_di = 256u64.min(total_q).max(1);
        let head_dim_u32 = head_dim as u32;
        let num_heads_bx = (batch_size * num_heads) as u32;
        plan.push(LayerOp {
            label: "deinterleave_qgate",
            accesses: AccessList::from_iter_inline([
                Access::read(BufferId::Qkv),
                Access::write(BufferId::Q),
                Access::write(BufferId::Gate),
            ]),
            order_class: OrderClass::Free,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(p_deinter);
                enc.set_buffer(qkv_buf, 0, 0);
                enc.set_buffer(q_buf, 0, 1);
                enc.set_buffer(gate_buf, 0, 2);
                enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                enc.set_bytes(&num_heads_bx.to_le_bytes(), 4);
                enc.dispatch_threadgroups(
                    MTLSize::new(total_q.div_ceil(tg_di), 1, 1),
                    MTLSize::new(tg_di, 1, 1),
                );
                Ok(())
            }),
        });

        // Op 6: Per-head Q RMSNorm (in-place on q_buf, ReadWrite)
        if let Some(q_norm_off) = q_norm_off {
            let p_rms_head = pipelines.rmsnorm_per_head.as_ref()
                .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:rmsnorm_per_head pipeline missing".into()))?;
            let tg_rms = 256u64.min(head_dim as u64).max(32);
            plan.push(LayerOp {
                label: "q_rmsnorm_per_head",
                accesses: AccessList::from_iter_inline([Access::read_write(BufferId::Q)]),
                order_class: OrderClass::Free,
                emit: Box::new(move |enc| {
                    enc.set_pipeline_state(p_rms_head);
                    enc.set_buffer(q_buf, 0, 0);
                    enc.set_buffer(layer_buf, q_norm_off, 1);
                    enc.set_buffer(q_buf, 0, 2);
                    enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                    enc.set_bytes(&eps.to_le_bytes(), 4);
                    enc.dispatch_threadgroups(
                        MTLSize::new((batch_size * num_heads) as u64, 1, 1),
                        MTLSize::new(tg_rms, 1, 1),
                    );
                    Ok(())
                }),
            });
        }

        // Op 7: Per-head K RMSNorm (in-place on k_buf, ReadWrite)
        if let Some(k_norm_off) = k_norm_off {
            let p_rms_head = pipelines.rmsnorm_per_head.as_ref()
                .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:rmsnorm_per_head pipeline missing".into()))?;
            let tg_rms = 256u64.min(head_dim as u64).max(32);
            plan.push(LayerOp {
                label: "k_rmsnorm_per_head",
                accesses: AccessList::from_iter_inline([Access::read_write(BufferId::K)]),
                order_class: OrderClass::Free,
                emit: Box::new(move |enc| {
                    enc.set_pipeline_state(p_rms_head);
                    enc.set_buffer(k_buf, 0, 0);
                    enc.set_buffer(layer_buf, k_norm_off, 1);
                    enc.set_buffer(k_buf, 0, 2);
                    enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                    enc.set_bytes(&eps.to_le_bytes(), 4);
                    enc.dispatch_threadgroups(
                        MTLSize::new((batch_size * num_kv_heads) as u64, 1, 1),
                        MTLSize::new(tg_rms, 1, 1),
                    );
                    Ok(())
                }),
            });
        }

        // Op 8: RoPE Q (in-place on q_buf, ReadWrite)
        // Op 9: RoPE K (in-place on k_buf, ReadWrite)
        let rope_pipe = if rope_neox {
            pipelines.rope_batched_neox.as_ref().unwrap_or(&pipelines.rope_batched)
        } else {
            &pipelines.rope_batched
        };
        let rope_half_dim_u32 = rope_half_dim as u32;
        let pos_start_u32 = seq_pos_start as u32;
        let q_dim_u32 = q_dim as u32;
        let kv_dim_u32 = kv_dim as u32;
        let num_heads_u32 = num_heads as u32;
        let num_kv_heads_u32 = num_kv_heads as u32;
        let total_q_elems = (batch_size * num_heads * rope_half_dim) as u64;
        let total_k_elems = (batch_size * num_kv_heads * rope_half_dim) as u64;
        let tg_q_rope = 256u64.min(total_q_elems).max(1);
        let tg_k_rope = 256u64.min(total_k_elems).max(1);
        plan.push(LayerOp {
            label: "rope_q",
            accesses: AccessList::from_iter_inline([Access::read_write(BufferId::Q)]),
            order_class: OrderClass::Free,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(rope_pipe);
                enc.set_buffer(q_buf, 0, 0);
                enc.set_buffer(rope_cos, 0, 1);
                enc.set_buffer(rope_sin, 0, 2);
                enc.set_bytes(&num_heads_u32.to_le_bytes(), 3);
                enc.set_bytes(&head_dim_u32.to_le_bytes(), 4);
                enc.set_bytes(&rope_half_dim_u32.to_le_bytes(), 5);
                enc.set_bytes(&pos_start_u32.to_le_bytes(), 6);
                enc.set_bytes(&q_dim_u32.to_le_bytes(), 7);
                enc.dispatch_threadgroups(
                    MTLSize::new(total_q_elems.div_ceil(tg_q_rope), 1, 1),
                    MTLSize::new(tg_q_rope, 1, 1),
                );
                Ok(())
            }),
        });
        plan.push(LayerOp {
            label: "rope_k",
            accesses: AccessList::from_iter_inline([Access::read_write(BufferId::K)]),
            order_class: OrderClass::Free,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(rope_pipe);
                enc.set_buffer(k_buf, 0, 0);
                enc.set_buffer(rope_cos, 0, 1);
                enc.set_buffer(rope_sin, 0, 2);
                enc.set_bytes(&num_kv_heads_u32.to_le_bytes(), 3);
                enc.set_bytes(&head_dim_u32.to_le_bytes(), 4);
                enc.set_bytes(&rope_half_dim_u32.to_le_bytes(), 5);
                enc.set_bytes(&pos_start_u32.to_le_bytes(), 6);
                enc.set_bytes(&kv_dim_u32.to_le_bytes(), 7);
                enc.dispatch_threadgroups(
                    MTLSize::new(total_k_elems.div_ceil(tg_k_rope), 1, 1),
                    MTLSize::new(tg_k_rope, 1, 1),
                );
                Ok(())
            }),
        });

        // Op 10: K cache write
        // Op 11: V cache write
        let start_pos_u32 = seq_pos_start as u32;
        let batch_size_u32 = batch_size as u32;
        let kv_total_elems = (batch_size * kv_dim) as u64;
        let kv_tg = 256u64.min(kv_total_elems).max(1);
        let p_kw = &pipelines.kv_cache_write_batched;
        let p_vw = &pipelines.v_cache_write_batched;
        let max_seq_len_u32 = max_seq_len as u32;
        plan.push(LayerOp {
            label: "k_cache_write",
            accesses: AccessList::from_iter_inline([Access::read(BufferId::K), Access::write(BufferId::KCache)]),
            order_class: OrderClass::Free,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(p_kw);
                enc.set_buffer(k_buf, 0, 0);
                enc.set_buffer(k_cache, 0, 1);
                enc.set_bytes(&kv_dim_u32.to_le_bytes(), 2);
                enc.set_bytes(&start_pos_u32.to_le_bytes(), 3);
                enc.set_bytes(&batch_size_u32.to_le_bytes(), 4);
                enc.dispatch_threadgroups(
                    MTLSize::new(kv_total_elems.div_ceil(kv_tg), 1, 1),
                    MTLSize::new(kv_tg, 1, 1),
                );
                Ok(())
            }),
        });
        plan.push(LayerOp {
            label: "v_cache_write",
            accesses: AccessList::from_iter_inline([Access::read(BufferId::V), Access::write(BufferId::VCache)]),
            order_class: OrderClass::Free,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(p_vw);
                enc.set_buffer(v_buf, 0, 0);
                enc.set_buffer(v_cache, 0, 1);
                enc.set_bytes(&kv_dim_u32.to_le_bytes(), 2);
                enc.set_bytes(&start_pos_u32.to_le_bytes(), 3);
                enc.set_bytes(&batch_size_u32.to_le_bytes(), 4);
                enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 5);
                enc.dispatch_threadgroups(
                    MTLSize::new(kv_total_elems.div_ceil(kv_tg), 1, 1),
                    MTLSize::new(kv_tg, 1, 1),
                );
                Ok(())
            }),
        });

        // Op 12: Attn scores (Strict — depends on KCache + Q; produces scores)
        let scores_stride_u32 = scores_stride as u32;
        let p_attn_scores = &pipelines.attention_scores_tiled;
        plan.push(LayerOp {
            label: "attn_scores",
            accesses: AccessList::from_iter_inline([
                Access::read(BufferId::Q),
                Access::read(BufferId::KCache),
                Access::write(BufferId::Scores),
            ]),
            order_class: OrderClass::Strict,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(p_attn_scores);
                enc.set_buffer(q_buf, 0, 0);
                enc.set_buffer(k_cache, 0, 1);
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
                Ok(())
            }),
        });

        // Op 13: Softmax (Strict — in-place on scores)
        let p_softmax = &pipelines.softmax_batched;
        plan.push(LayerOp {
            label: "softmax",
            accesses: AccessList::from_iter_inline([Access::read_write(BufferId::Scores)]),
            order_class: OrderClass::Strict,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(p_softmax);
                enc.set_buffer(scores_buf, 0, 0);
                enc.set_bytes(&scores_stride_u32.to_le_bytes(), 1);
                enc.set_bytes(&num_heads_u32.to_le_bytes(), 2);
                enc.set_bytes(&start_pos_u32.to_le_bytes(), 3);
                enc.set_bytes(&batch_size_u32.to_le_bytes(), 4);
                enc.dispatch_threadgroups(
                    MTLSize::new(num_heads as u64, batch_size as u64, 1),
                    MTLSize::new(256u64.min(max_attend_len as u64).max(1), 1, 1),
                );
                Ok(())
            }),
        });

        // Op 14: Attn output (Strict — reads softmax probs, V cache; writes attn_out)
        // Default: tiled GEMM (F16 simdgroup P@V). LUMEN_METAL_ATTN_PRECISE=2 routes
        // to the exact-F32 scalar attention_output_batched (no F16 P@V truncation).
        let pv_precise = metal_attn_precise_pvf32();
        let p_attn_out = if pv_precise {
            metal_attn_precise_engage("concurrent", layer_idx, batch_size, num_heads, head_dim);
            &pipelines.attention_output_batched
        } else {
            &pipelines.attention_output_tiled
        };
        plan.push(LayerOp {
            label: "attn_output",
            accesses: AccessList::from_iter_inline([
                Access::read(BufferId::Scores),
                Access::read(BufferId::VCache),
                Access::write(BufferId::AttnOut),
            ]),
            order_class: OrderClass::Strict,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(p_attn_out);
                enc.set_buffer(scores_buf, 0, 0);
                enc.set_buffer(v_cache, 0, 1);
                enc.set_buffer(attn_out_buf, 0, 2);
                enc.set_bytes(&head_dim_u32.to_le_bytes(), 3);
                enc.set_bytes(&kv_dim_u32.to_le_bytes(), 4);
                enc.set_bytes(&num_heads_u32.to_le_bytes(), 5);
                enc.set_bytes(&num_kv_heads_u32.to_le_bytes(), 6);
                enc.set_bytes(&start_pos_u32.to_le_bytes(), 7);
                enc.set_bytes(&scores_stride_u32.to_le_bytes(), 8);
                enc.set_bytes(&batch_size_u32.to_le_bytes(), 9);
                enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 10);
                if pv_precise {
                    let total_elems = (batch_size * num_heads as usize * head_dim) as u64;
                    let pv_tg = 256u64.min(total_elems).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new(total_elems.div_ceil(pv_tg), 1, 1),
                        MTLSize::new(pv_tg, 1, 1),
                    );
                } else {
                    enc.dispatch_threadgroups(
                        MTLSize::new(
                            (head_dim as u64).div_ceil(32),
                            (batch_size as u64).div_ceil(32),
                            num_heads as u64,
                        ),
                        MTLSize::new(128, 1, 1),
                    );
                }
                Ok(())
            }),
        });

        // Op 15: Sigmoid gate (Strict, in-place on attn_out, reads)
        let p_sigmoid_gate = pipelines.sigmoid_mul_fused.as_ref()
            .ok_or_else(|| RuntimeError::Compute("concurrent-encoder path:sigmoid_mul_fused pipeline missing".into()))?;
        let total_gate_elems = (batch_size * q_dim) as u32;
        let tg_sg = 256u64.min(total_gate_elems as u64).max(1);
        plan.push(LayerOp {
            label: "sigmoid_gate",
            accesses: AccessList::from_iter_inline([
                Access::read(BufferId::Gate),
                Access::read_write(BufferId::AttnOut),
            ]),
            order_class: OrderClass::Strict,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(p_sigmoid_gate);
                enc.set_buffer(gate_buf, 0, 0);
                enc.set_buffer(attn_out_buf, 0, 1);
                enc.set_buffer(attn_out_buf, 0, 2);
                enc.set_bytes(&total_gate_elems.to_le_bytes(), 3);
                enc.dispatch_threadgroups(
                    MTLSize::new((total_gate_elems as u64).div_ceil(tg_sg), 1, 1),
                    MTLSize::new(tg_sg, 1, 1),
                );
                Ok(())
            }),
        });

        // Op 16: Wo proj + residual (fused kernel: attn_out @ wo + x_buf -> attn_proj)
        //         Strict: reads x_buf as residual, writes attn_proj.
        let wo_m_u32 = batch_size as u32;
        let wo_n_u32 = hidden_dim as u32;
        let wo_k_u32 = q_dim as u32;
        plan.push(Self::concurrent_encoder_emit_gemm_residual(
            "wo_proj_residual",
            BufferId::AttnOut, BufferId::AttnProj, BufferId::X,
            layer_buf, wo_off, attn_out_buf, attn_proj_buf, x_buf,
            pipelines, st.wo.quant, wo_m_u32, wo_n_u32, wo_k_u32,
            batch_size, hidden_dim, q_dim,
            TILE_M, TILE_N,
        ));

        // Op 17: FFN RMSNorm: attn_proj_buf -> normed_buf (Strict; consumes attn_proj)
        let p_rmsnorm_ffn = &pipelines.rmsnorm_batched_bytes;
        plan.push(LayerOp {
            label: "ffn_norm",
            accesses: AccessList::from_iter_inline([
                Access::read(BufferId::AttnProj),
                Access::write(BufferId::Normed),
            ]),
            order_class: OrderClass::Strict,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(p_rmsnorm_ffn);
                enc.set_buffer(attn_proj_buf, 0, 0);
                enc.set_buffer(layer_buf, ffn_norm_off, 1);
                enc.set_buffer(normed_buf, 0, 2);
                enc.set_bytes(&dim_u32.to_le_bytes(), 3);
                enc.set_bytes(&eps.to_le_bytes(), 4);
                enc.dispatch_threadgroups(
                    MTLSize::new(batch_size as u64, 1, 1),
                    MTLSize::new(norm_tg_size, 1, 1),
                );
                Ok(())
            }),
        });

        // Op 18+: FFN gate+up [+swiglu]
        let ffn_n_u32 = inter_dim as u32;
        let ffn_k_u32 = hidden_dim as u32;
        // default-on the fused gate+up+swiglu kernel via
        // super::graph_reorder::ffn_gate_up_swiglu_fused_q{8,4}_enabled().
        // Explicit `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED=0` still disables;
        // the master kill-switch `LUMEN_METAL_DEFAULTS_OFF=1` restores
        // the legacy default-OFF behaviour for both Q8 and Q4 variants.
        // BF16 fused-kernel arm — wires the kernel
        // (`bf16_matmul_gate_up_swiglu_fused{,_aligned}`) into the concurrent-encoder plan
        // builder so that BF16 prefill no longer falls to the unfused 3-op
        // path under the concurrent-encoder default. Opt-in via
        // `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_BF16=1` (default OFF until
        // a multi-prompt bench validates the win).
        let fused_q8 = st.w_gate.quant == QuantScheme::Q8_0
            && st.w_up.quant == QuantScheme::Q8_0
            && hidden_dim % 64 == 0 && batch_size <= 4096
            && super::graph_reorder::ffn_gate_up_swiglu_fused_q8_enabled();
        let fused_q4 = st.w_gate.quant == QuantScheme::Q4_0
            && st.w_up.quant == QuantScheme::Q4_0
            && hidden_dim % 64 == 0 && batch_size <= 4096
            && super::graph_reorder::ffn_gate_up_swiglu_fused_q4_enabled();
        let fused_bf16 = st.w_gate.quant == QuantScheme::Bf16
            && st.w_up.quant == QuantScheme::Bf16
            && hidden_dim % 64 == 0 && batch_size <= 4096
            && super::graph_reorder::ffn_gate_up_swiglu_fused_bf16_enabled();
        let ffn_aligned = batch_size % 32 == 0 && inter_dim % 32 == 0 && hidden_dim % 32 == 0;
        if fused_q8 {
            // packed-layout variant when repacked buffer is present.
            if let Some(buf_gu) = repacked_gate_up {
                let pso = if ffn_aligned && hidden_dim % 64 == 0 {
                    &pipelines.dequant_tiled_matmul_q8_0_gate_up_swiglu_fused_packed_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q8_0_gate_up_swiglu_fused_packed
                };
                plan.push(LayerOp {
                    label: "ffn_gate_up_swiglu_fused_q8_packed",
                    accesses: AccessList::from_iter_inline([
                        Access::read(BufferId::Normed),
                        Access::write(BufferId::Gate),
                    ]),
                    order_class: OrderClass::Free,
                    emit: Box::new(move |enc| {
                        enc.set_pipeline_state(pso);
                        enc.set_threadgroup_memory_length(12288, 0);
                        enc.set_buffer(buf_gu, 0, 0);   // paired packed buffer
                        enc.set_buffer(normed_buf, 0, 1);
                        enc.set_buffer(gate_buf, 0, 2);
                        enc.set_bytes(&m_u32.to_le_bytes(), 3);
                        enc.set_bytes(&ffn_n_u32.to_le_bytes(), 4);
                        enc.set_bytes(&ffn_k_u32.to_le_bytes(), 5);
                        enc.dispatch_threadgroups(
                            MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                            MTLSize::new(128, 1, 1),
                        );
                        Ok(())
                    }),
                });
            } else {
                let pso = if ffn_aligned && hidden_dim % 64 == 0 {
                    &pipelines.dequant_tiled_matmul_q8_0_gate_up_swiglu_fused_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q8_0_gate_up_swiglu_fused
                };
                plan.push(LayerOp {
                    label: "ffn_gate_up_swiglu_fused_q8",
                    accesses: AccessList::from_iter_inline([
                        Access::read(BufferId::Normed),
                        Access::write(BufferId::Gate),
                    ]),
                    order_class: OrderClass::Free,
                    emit: Box::new(move |enc| {
                        enc.set_pipeline_state(pso);
                        enc.set_threadgroup_memory_length(12288, 0);
                        enc.set_buffer(layer_buf, w_gate_off, 0);
                        enc.set_buffer(normed_buf, 0, 1);
                        enc.set_buffer(gate_buf, 0, 2);
                        enc.set_bytes(&m_u32.to_le_bytes(), 3);
                        enc.set_bytes(&ffn_n_u32.to_le_bytes(), 4);
                        enc.set_bytes(&ffn_k_u32.to_le_bytes(), 5);
                        enc.set_buffer(layer_buf, w_up_off, 6);
                        enc.dispatch_threadgroups(
                            MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                            MTLSize::new(128, 1, 1),
                        );
                        Ok(())
                    }),
                });
            }
        } else if fused_q4 {
            // packed-layout Q4 variant when repacked buffer is present.
            if let Some(buf_gu) = repacked_gate_up_q4 {
                let pso = if ffn_aligned && hidden_dim % 64 == 0 {
                    &pipelines.dequant_tiled_matmul_q4_0_gate_up_swiglu_fused_packed_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q4_0_gate_up_swiglu_fused_packed
                };
                plan.push(LayerOp {
                    label: "ffn_gate_up_swiglu_fused_q4_packed",
                    accesses: AccessList::from_iter_inline([
                        Access::read(BufferId::Normed),
                        Access::write(BufferId::Gate),
                    ]),
                    order_class: OrderClass::Free,
                    emit: Box::new(move |enc| {
                        enc.set_pipeline_state(pso);
                        enc.set_threadgroup_memory_length(12288, 0);
                        enc.set_buffer(buf_gu, 0, 0);   // paired packed buffer
                        enc.set_buffer(normed_buf, 0, 1);
                        enc.set_buffer(gate_buf, 0, 2);
                        enc.set_bytes(&m_u32.to_le_bytes(), 3);
                        enc.set_bytes(&ffn_n_u32.to_le_bytes(), 4);
                        enc.set_bytes(&ffn_k_u32.to_le_bytes(), 5);
                        enc.dispatch_threadgroups(
                            MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                            MTLSize::new(128, 1, 1),
                        );
                        Ok(())
                    }),
                });
            } else {
                let pso = if ffn_aligned && hidden_dim % 64 == 0 {
                    &pipelines.dequant_tiled_matmul_q4_0_gate_up_swiglu_fused_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q4_0_gate_up_swiglu_fused
                };
                plan.push(LayerOp {
                    label: "ffn_gate_up_swiglu_fused_q4",
                    accesses: AccessList::from_iter_inline([
                        Access::read(BufferId::Normed),
                        Access::write(BufferId::Gate),
                    ]),
                    order_class: OrderClass::Free,
                    emit: Box::new(move |enc| {
                        enc.set_pipeline_state(pso);
                        enc.set_threadgroup_memory_length(12288, 0);
                        enc.set_buffer(layer_buf, w_gate_off, 0);
                        enc.set_buffer(normed_buf, 0, 1);
                        enc.set_buffer(gate_buf, 0, 2);
                        enc.set_bytes(&m_u32.to_le_bytes(), 3);
                        enc.set_bytes(&ffn_n_u32.to_le_bytes(), 4);
                        enc.set_bytes(&ffn_k_u32.to_le_bytes(), 5);
                        enc.set_buffer(layer_buf, w_up_off, 6);
                        enc.dispatch_threadgroups(
                            MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                            MTLSize::new(128, 1, 1),
                        );
                        Ok(())
                    }),
                });
            }
        } else if fused_bf16 {
            // BF16 fused gate+up+SwiGLU dispatch on the concurrent-encoder plan
            // builder. Mirrors the legacy per-layer path at line 2348-2371.
            // BF16 has no packed SoA repack ( F2), so this arm
            // dispatches the layer-buffer-resident weights directly.
            //
            // NR microtile selection (env `LUMEN_METAL_BF16_GATE_UP_NR`).
            // NR=2 (baseline, 12 KB shmem). NR=1 (10 KB). NR=4 (16 KB). The
            // dispatch-grid Y divisor and threadgroup memory length must match
            // the kernel's TILE_M and shmem declaration respectively.
            let nr = super::graph_reorder::bf16_gate_up_nr();
            let (tile_m_dispatch, shmem_bytes): (u64, u64) = match nr {
                1 => (16, 10240),
                4 => (64, 16384),
                _ => (32, 12288),
            };
            let pso = match (nr, ffn_aligned && hidden_dim % 64 == 0) {
                (1, true)  => &pipelines.bf16_matmul_gate_up_swiglu_fused_nr1_aligned,
                (1, false) => &pipelines.bf16_matmul_gate_up_swiglu_fused_nr1,
                (4, true)  => &pipelines.bf16_matmul_gate_up_swiglu_fused_nr4_aligned,
                (4, false) => &pipelines.bf16_matmul_gate_up_swiglu_fused_nr4,
                (_, true)  => &pipelines.bf16_matmul_gate_up_swiglu_fused_aligned,
                (_, false) => &pipelines.bf16_matmul_gate_up_swiglu_fused,
            };
            plan.push(LayerOp {
                label: "ffn_gate_up_swiglu_fused_bf16",
                accesses: AccessList::from_iter_inline([
                    Access::read(BufferId::Normed),
                    Access::write(BufferId::Gate),
                ]),
                order_class: OrderClass::Free,
                emit: Box::new(move |enc| {
                    enc.set_pipeline_state(pso);
                    enc.set_threadgroup_memory_length(shmem_bytes, 0);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(normed_buf, 0, 1);
                    enc.set_buffer(gate_buf, 0, 2);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&ffn_n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&ffn_k_u32.to_le_bytes(), 5);
                    enc.set_buffer(layer_buf, w_up_off, 6);
                    enc.dispatch_threadgroups(
                        MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(tile_m_dispatch), 1),
                        MTLSize::new(128, 1, 1),
                    );
                    Ok(())
                }),
            });
        } else {
            // Unfused path: gate GEMM + up GEMM + SwiGLU. Gate and Up can be
            // scheduled in parallel (different output buffers); SwiGLU is
            // Strict because it reads both gate and up.
            plan.push(Self::concurrent_encoder_emit_gemm(
                "ffn_gate_gemm",
                BufferId::Normed, BufferId::Gate,
                layer_buf, w_gate_off, normed_buf, gate_buf,
                pipelines, st.w_gate.quant, m_u32, ffn_n_u32, ffn_k_u32,
                batch_size, inter_dim, hidden_dim,
                TILE_M, TILE_N,
            ));
            plan.push(Self::concurrent_encoder_emit_gemm(
                "ffn_up_gemm",
                BufferId::Normed, BufferId::Up,
                layer_buf, w_up_off, normed_buf, up_buf,
                pipelines, st.w_up.quant, m_u32, ffn_n_u32, ffn_k_u32,
                batch_size, inter_dim, hidden_dim,
                TILE_M, TILE_N,
            ));
            let p_swiglu = &pipelines.swiglu_batched;
            let total_elems = (batch_size * inter_dim) as u32;
            let tg_swiglu = 256u64.min(total_elems as u64).max(1);
            plan.push(LayerOp {
                label: "ffn_swiglu",
                accesses: AccessList::from_iter_inline([
                    Access::read(BufferId::Up),
                    Access::read_write(BufferId::Gate),
                ]),
                order_class: OrderClass::Strict,
                emit: Box::new(move |enc| {
                    enc.set_pipeline_state(p_swiglu);
                    enc.set_buffer(gate_buf, 0, 0);
                    enc.set_buffer(up_buf, 0, 1);
                    enc.set_bytes(&total_elems.to_le_bytes(), 2);
                    enc.dispatch_threadgroups(
                        MTLSize::new((total_elems as u64).div_ceil(tg_swiglu), 1, 1),
                        MTLSize::new(tg_swiglu, 1, 1),
                    );
                    Ok(())
                }),
            });
        }

        // Op final: FFN down + residual (fused): gate @ wdown + attn_proj -> x_buf
        let dn_m_u32 = batch_size as u32;
        let dn_n_u32 = hidden_dim as u32;
        let dn_k_u32 = inter_dim as u32;
        // when packed FFN-down is available AND Q8 K64 conditions hold,
        // emit the packed-layout dispatch inline; otherwise use the standard helper.
        let dn_k64_eligible_q8 = matches!(st.w_down.quant, QuantScheme::Q8_0)
            && inter_dim % 64 == 0 && batch_size <= 4096;
        // same for Q4_0 packed FFN-down.
        let dn_k64_eligible_q4 = matches!(st.w_down.quant, QuantScheme::Q4_0)
            && inter_dim % 64 == 0 && batch_size <= 4096;
        // BF16 FFN-down Split-K K64 in the concurrent-encoder path. Mirrors the legacy
        // per-layer dispatch at site `prefill_encode.rs:~2675`. Eligibility matches
        // `ffn_down_splitk_bf16_value()` doc: BF16 weights, M <= 192, K >= 8192,
        // K % 64 == 0. Honors `LUMEN_METAL_FFN_DOWN_SPLITK_BF16=<2|4|8>`;
        // default OFF until empirical validation (per the env resolver, which
        // is intentionally NOT gated under metal_defaults_active).
        let ffn_splitk_bf16_env = super::graph_reorder::ffn_down_splitk_bf16_value();
        let dn_k64_eligible_bf16_splitk = ffn_splitk_bf16_env > 0
            && matches!(st.w_down.quant, QuantScheme::Bf16)
            && batch_size <= 192
            && inter_dim >= 8192
            && inter_dim % 64 == 0;
        if dn_k64_eligible_bf16_splitk {
            let k_splits = ffn_splitk_bf16_env;
            plan.push(LayerOp {
                label: "ffn_down_residual_splitk_bf16",
                accesses: AccessList::from_iter_inline([
                    Access::read(BufferId::Gate),
                    Access::write(BufferId::X),
                    Access::read(BufferId::AttnProj),
                ]),
                order_class: OrderClass::Strict,
                emit: Box::new(move |enc| {
                    Self::encode_splitk_bf16_gemm_k64_residual(
                        enc, pipelines,
                        layer_buf, w_down_off,
                        gate_buf, attn_proj_buf, x_buf, splitk_partial_buf,
                        dn_m_u32, dn_n_u32, dn_k_u32, k_splits,
                    );
                    Ok(())
                }),
            });
        } else if dn_k64_eligible_q8 && repacked_down.is_some() {
            let buf_d = repacked_down.unwrap();
            let dn_aligned = batch_size % 32 == 0 && hidden_dim % 32 == 0 && inter_dim % 32 == 0;
            let pso = if dn_aligned {
                &pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_packed_aligned
            } else {
                &pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_packed
            };
            plan.push(LayerOp {
                label: "ffn_down_residual_packed",
                accesses: AccessList::from_iter_inline([
                    Access::read(BufferId::Gate),
                    Access::write(BufferId::X),
                    Access::read(BufferId::AttnProj),
                ]),
                order_class: OrderClass::Strict,
                emit: Box::new(move |enc| {
                    enc.set_pipeline_state(pso);
                    enc.set_threadgroup_memory_length(8192, 0);
                    enc.set_buffer(buf_d, 0, 0);
                    enc.set_buffer(gate_buf, 0, 1);
                    enc.set_buffer(x_buf, 0, 2);
                    enc.set_bytes(&dn_m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&dn_n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&dn_k_u32.to_le_bytes(), 5);
                    enc.set_buffer(attn_proj_buf, 0, 6);
                    enc.dispatch_threadgroups(
                        MTLSize::new((hidden_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                    Ok(())
                }),
            });
        } else if dn_k64_eligible_q4 && repacked_down_q4.is_some() {
            let buf_d = repacked_down_q4.unwrap();
            let dn_aligned = batch_size % 32 == 0 && hidden_dim % 32 == 0 && inter_dim % 32 == 0;
            let pso = if dn_aligned {
                &pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched_packed_aligned
            } else {
                &pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched_packed
            };
            plan.push(LayerOp {
                label: "ffn_down_residual_packed_q4",
                accesses: AccessList::from_iter_inline([
                    Access::read(BufferId::Gate),
                    Access::write(BufferId::X),
                    Access::read(BufferId::AttnProj),
                ]),
                order_class: OrderClass::Strict,
                emit: Box::new(move |enc| {
                    enc.set_pipeline_state(pso);
                    enc.set_threadgroup_memory_length(8192, 0);
                    enc.set_buffer(buf_d, 0, 0);
                    enc.set_buffer(gate_buf, 0, 1);
                    enc.set_buffer(x_buf, 0, 2);
                    enc.set_bytes(&dn_m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&dn_n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&dn_k_u32.to_le_bytes(), 5);
                    enc.set_buffer(attn_proj_buf, 0, 6);
                    enc.dispatch_threadgroups(
                        MTLSize::new((hidden_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                    Ok(())
                }),
            });
        } else {
            plan.push(Self::concurrent_encoder_emit_gemm_residual(
                "ffn_down_residual",
                BufferId::Gate, BufferId::X, BufferId::AttnProj,
                layer_buf, w_down_off, gate_buf, x_buf, attn_proj_buf,
                pipelines, st.w_down.quant, dn_m_u32, dn_n_u32, dn_k_u32,
                batch_size, hidden_dim, inter_dim,
                TILE_M, TILE_N,
            ));
        }

        // ---- Emit the plan into a single per-layer encoder ----
        super::profile::set_section("concurrent_encoder/layer");
        let serial_validate = graph_reorder::concurrent_encoder_validate_serial();
        // Buffer-id -> MetalBuffer lookup for resource-scoped barriers.
        // Returns `None` for `Down` (not used in the initial concurrent-encoder plan); the
        // scheduler falls back to whole-buffer scope only when no mapping
        // is known. KCache / VCache map to the per-layer slot.
        let lookup = |id: BufferId| -> Option<&MetalBuffer> {
            Some(match id {
                BufferId::X        => x_buf,
                BufferId::Normed   => normed_buf,
                BufferId::Qkv      => qkv_buf,
                BufferId::Q        => q_buf,
                BufferId::K        => k_buf,
                BufferId::V        => v_buf,
                BufferId::AttnOut  => attn_out_buf,
                BufferId::AttnProj => attn_proj_buf,
                BufferId::Gate     => gate_buf,
                BufferId::Up       => up_buf,
                BufferId::Scores   => scores_buf,
                BufferId::KCache   => k_cache,
                BufferId::VCache   => v_cache,
                BufferId::Down     => return None,
            })
        };

        // when `outer_enc` is supplied (whole-prefill encoder caller),
        // emit the plan into the caller's encoder and skip both the per-layer
        // open/close. When `None`, open a per-layer encoder as.
        //
        // Both modes use the same scheduler call: barriers are always
        // emitted between wavefronts. Validate mode (`LUMEN_METAL_CONCURRENT_ENCODER_VALIDATE=1`)
        // forces a serial encoder (`new_compute_encoder`) when this function
        // owns the encoder lifecycle, so per-dispatch reordering cannot
        // occur; production mode uses `new_concurrent_compute_encoder` and
        // lets the GPU overlap non-conflicting dispatches. Apple Metal does
        // NOT implicitly serialise dispatches inside a serial encoder when
        // they write to the same buffer — `memoryBarrierWithScope:` is
        // mandatory in both modes.
        if let Some(enc) = outer_enc {
            graph_reorder::emit_plan_into_encoder(enc, &mut plan, false, Some(&lookup))?;
        } else {
            let enc = if serial_validate {
                cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("concurrent-encoder path:failed to create validate encoder".into())
                })?
            } else {
                cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("concurrent-encoder path:failed to create concurrent encoder".into())
                })?
            };
            graph_reorder::emit_plan_into_encoder(&enc, &mut plan, false, Some(&lookup))?;
            enc.end_encoding();
        }

        // Update KV cache position.
        kv.seq_len = seq_pos_start + batch_size;

        Ok(())
    }

    /// Construct a `LayerOp` for a plain quant GEMM `output = weight @ input`.
    ///
    /// Pipeline selection mirrors `encode_layer_batched`'s match arms. The
    /// `ggml-port` fast path (`use_ggml_ported_q8_0_gemm()`) is preserved
    /// through `dispatch_q8_0_or_orig`.
    #[allow(clippy::too_many_arguments)]
    fn concurrent_encoder_emit_gemm<'a>(
        label: &'static str,
        input_id: BufferId,
        output_id: BufferId,
        layer_buf: &'a MetalBuffer,
        w_off: u64,
        in_buf: &'a MetalBuffer,
        out_buf: &'a MetalBuffer,
        pipelines: &'a MetalPipelines,
        quant: QuantScheme,
        m_u32: u32,
        n_u32: u32,
        k_u32: u32,
        batch_size: usize,
        n: usize,
        k: usize,
        tile_m: u64,
        tile_n: u64,
    ) -> LayerOp<'a> {
        let pso_select = Self::concurrent_encoder_select_gemm_pipeline(pipelines, quant, batch_size, n, k);
        let (pso, tg_mem) = pso_select;
        LayerOp {
            label,
            accesses: AccessList::from_iter_inline([Access::read(input_id), Access::write(output_id)]),
            order_class: OrderClass::Free,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(pso);
                enc.set_threadgroup_memory_length(tg_mem, 0);
                enc.set_buffer(layer_buf, w_off, 0);
                enc.set_buffer(in_buf, 0, 1);
                enc.set_buffer(out_buf, 0, 2);
                enc.set_bytes(&m_u32.to_le_bytes(), 3);
                enc.set_bytes(&n_u32.to_le_bytes(), 4);
                enc.set_bytes(&k_u32.to_le_bytes(), 5);
                dispatch_q8_0_or_orig(
                    enc, pipelines, quant == QuantScheme::Q8_0,
                    batch_size as u32, n_u32,
                    MTLSize::new((n as u64).div_ceil(tile_n),
                                 (batch_size as u64).div_ceil(tile_m), 1),
                );
                Ok(())
            }),
        }
    }

    /// Construct a `LayerOp` for a fused `output = weight @ input + residual` GEMM.
    ///
    /// Strict-classified because the residual read may overlap byte ranges
    /// with neighbouring ops.
    #[allow(clippy::too_many_arguments)]
    fn concurrent_encoder_emit_gemm_residual<'a>(
        label: &'static str,
        input_id: BufferId,
        output_id: BufferId,
        residual_id: BufferId,
        layer_buf: &'a MetalBuffer,
        w_off: u64,
        in_buf: &'a MetalBuffer,
        out_buf: &'a MetalBuffer,
        res_buf: &'a MetalBuffer,
        pipelines: &'a MetalPipelines,
        quant: QuantScheme,
        m_u32: u32,
        n_u32: u32,
        k_u32: u32,
        batch_size: usize,
        n: usize,
        k: usize,
        tile_m: u64,
        tile_n: u64,
    ) -> LayerOp<'a> {
        let (pso, tg_mem) = Self::concurrent_encoder_select_gemm_residual_pipeline(pipelines, quant, batch_size, n, k);
        LayerOp {
            label,
            accesses: AccessList::from_iter_inline([
                Access::read(input_id),
                Access::read(residual_id),
                Access::write(output_id),
            ]),
            order_class: OrderClass::Strict,
            emit: Box::new(move |enc| {
                enc.set_pipeline_state(pso);
                enc.set_threadgroup_memory_length(tg_mem, 0);
                enc.set_buffer(layer_buf, w_off, 0);
                enc.set_buffer(in_buf, 0, 1);
                enc.set_buffer(out_buf, 0, 2);
                enc.set_bytes(&m_u32.to_le_bytes(), 3);
                enc.set_bytes(&n_u32.to_le_bytes(), 4);
                enc.set_bytes(&k_u32.to_le_bytes(), 5);
                enc.set_buffer(res_buf, 0, 6);
                enc.dispatch_threadgroups(
                    MTLSize::new((n as u64).div_ceil(tile_n),
                                 (batch_size as u64).div_ceil(tile_m), 1),
                    MTLSize::new(128, 1, 1),
                );
                Ok(())
            }),
        }
    }

    /// Pick the pipeline + threadgroup-memory pair for a non-residual GEMM.
    fn concurrent_encoder_select_gemm_pipeline<'a>(
        pipelines: &'a MetalPipelines,
        quant: QuantScheme,
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> (&'a super::ffi::MetalPipelineState, u64) {
        let aligned = batch_size % 32 == 0 && n % 32 == 0 && k % 32 == 0;
        match quant {
            QuantScheme::Q8_0 if k % 64 == 0 && batch_size <= 4096 => {
                let p = if aligned && k % 64 == 0 {
                    &pipelines.dequant_tiled_matmul_q8_0_k64_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q8_0_k64
                };
                (p, 8192)
            }
            QuantScheme::Q8_0 => {
                let p = if aligned {
                    &pipelines.dequant_tiled_matmul_q8_0_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q8_0
                };
                (p, 4096)
            }
            QuantScheme::Q4_0 if k % 64 == 0 && batch_size <= 4096 => {
                let p = if aligned && k % 64 == 0 {
                    &pipelines.dequant_tiled_matmul_q4_0_k64_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q4_0_k64
                };
                (p, 8192)
            }
            QuantScheme::Q4_0 => (&pipelines.dequant_tiled_matmul_q4_0, 4096),
            QuantScheme::Q4_1 => (&pipelines.dequant_tiled_matmul_q4_1, 4096),
            QuantScheme::F16 if k % 64 == 0 && batch_size <= 4096 => {
                let p = if aligned && k % 64 == 0 {
                    &pipelines.tiled_matmul_f16_k64_aligned
                } else {
                    &pipelines.tiled_matmul_f16_k64
                };
                (p, 8192)
            }
            QuantScheme::F16 => (&pipelines.tiled_matmul_f16, 4096),
            QuantScheme::Bf16 if k % 64 == 0 && batch_size <= 4096 => {
                let p = if aligned && k % 64 == 0 {
                    &pipelines.tiled_matmul_bf16_k64_aligned
                } else {
                    &pipelines.tiled_matmul_bf16_k64
                };
                (p, 8192)
            }
            QuantScheme::Bf16 => (&pipelines.tiled_matmul_bf16, 4096),
            _ => (&pipelines.tiled_matmul_bytes_f32, 4096),
        }
    }

    /// Pick the pipeline + threadgroup-memory pair for a GEMM-with-residual.
    fn concurrent_encoder_select_gemm_residual_pipeline<'a>(
        pipelines: &'a MetalPipelines,
        quant: QuantScheme,
        batch_size: usize,
        n: usize,
        k: usize,
    ) -> (&'a super::ffi::MetalPipelineState, u64) {
        let aligned = batch_size % 32 == 0 && n % 32 == 0 && k % 32 == 0;
        match quant {
            QuantScheme::Q8_0 if k % 64 == 0 && batch_size <= 4096 => {
                let p = if aligned && k % 64 == 0 {
                    &pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched
                };
                (p, 8192)
            }
            QuantScheme::Q8_0 => {
                let p = if aligned {
                    &pipelines.dequant_tiled_matmul_q8_0_residual_batched_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q8_0_residual_batched
                };
                (p, 4096)
            }
            QuantScheme::Q4_0 if k % 64 == 0 && batch_size <= 4096 => {
                let p = if aligned && k % 64 == 0 {
                    &pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched
                };
                (p, 8192)
            }
            QuantScheme::Q4_0 => (&pipelines.dequant_tiled_matmul_q4_0_residual_batched, 4096),
            QuantScheme::Q4_1 => (&pipelines.dequant_tiled_matmul_q4_1_residual_batched, 4096),
            QuantScheme::F16 if k % 64 == 0 && batch_size <= 4096 => {
                let p = if aligned && k % 64 == 0 {
                    &pipelines.tiled_matmul_f16_k64_residual_aligned
                } else {
                    &pipelines.tiled_matmul_f16_k64_residual
                };
                (p, 8192)
            }
            QuantScheme::F16 => (&pipelines.tiled_matmul_f16_residual, 4096),
            QuantScheme::Bf16 if k % 64 == 0 && batch_size <= 4096 => {
                let p = if aligned && k % 64 == 0 {
                    &pipelines.tiled_matmul_bf16_k64_residual_aligned
                } else {
                    &pipelines.tiled_matmul_bf16_k64_residual
                };
                (p, 8192)
            }
            QuantScheme::Bf16 => (&pipelines.tiled_matmul_bf16_residual, 4096),
            _ => (&pipelines.tiled_matmul_bytes_f32_residual, 4096),
        }
    }

    // ========================================================================
    // Shared FFN block plan builder.
    //
    // Both `encode_layer_batched_concurrent` (full-attn layers) and
    // `encode_layer_batched_into` (GDN layers outer-encoder mode)
    // need to emit the same FFN block dispatch chain:
    //   * FFN RMSNorm  (attn_proj_buf -> normed_buf)
    //   * FFN gate+up [+ SwiGLU] (normed_buf -> gate_buf)
    //   * FFN down + residual (gate_buf @ wdown + attn_proj_buf -> x_buf)
    //
    // The concurrent-encoder path includes FFN-norm in the plan (it follows the Wo+residual
    // op that wrote attn_proj_buf). The GDN path's `encode_batched_gdn_prefill`
    // ALREADY emits the FFN-norm dispatch at the end of its phase chain, so
    // the GDN caller passes `skip_ffn_norm=true` to avoid emitting it
    // a second time.
    //
    // Returning a `Vec<LayerOp>` rather than a closure-emitting iterator
    // lets the caller append the FFN ops to its existing plan (full-attn
    // case) or schedule them on their own (GDN case).
    // ========================================================================

    /// Build the FFN-block ops (RMSNorm, gate+up [+SwiGLU], down+residual)
    /// for a single layer and append to `plan`.
    ///
    /// `skip_ffn_norm` — when true (GDN case), the FFN RMSNorm op is
    /// omitted (the caller's earlier dispatch chain already produced
    /// `normed_buf` from `attn_proj_buf`).
    #[allow(clippy::too_many_arguments)]
    fn concurrent_encoder_extend_plan_with_ffn_block<'a>(
        plan: &mut Vec<LayerOp<'a>>,
        skip_ffn_norm: bool,
        pipelines: &'a MetalPipelines,
        st: &lumen_format::SubtensorOffsets,
        layer_buf: &'a MetalBuffer,
        normed_buf: &'a MetalBuffer,
        attn_proj_buf: &'a MetalBuffer,
        gate_buf: &'a MetalBuffer,
        up_buf: &'a MetalBuffer,
        x_buf: &'a MetalBuffer,
        // BF16 FFN-down Split-K K64 in the concurrent-encoder path needs the Split-K partial
        // buffer. Allocated unconditionally by ensure_batch_scratch (size set
        // to the worst case when LUMEN_METAL_FFN_DOWN_SPLITK_BF16 > 0).
        splitk_partial_buf: &'a MetalBuffer,
        ffn_norm_off: u64,
        w_gate_off: u64,
        w_up_off: u64,
        w_down_off: u64,
        repacked_gate_up: Option<&'a MetalBuffer>,
        repacked_down: Option<&'a MetalBuffer>,
        repacked_gate_up_q4: Option<&'a MetalBuffer>,
        repacked_down_q4: Option<&'a MetalBuffer>,
        dim_u32: u32,
        eps: f32,
        norm_tg_size: u64,
        m_u32: u32,
        batch_size: usize,
        hidden_dim: usize,
        inter_dim: usize,
    ) {
        const TILE_M: u64 = 32;
        const TILE_N: u64 = 32;

        // Op 17: FFN RMSNorm: attn_proj_buf -> normed_buf
        if !skip_ffn_norm {
            let p_rmsnorm_ffn = &pipelines.rmsnorm_batched_bytes;
            plan.push(LayerOp {
                label: "ffn_norm",
                accesses: AccessList::from_iter_inline([
                    Access::read(BufferId::AttnProj),
                    Access::write(BufferId::Normed),
                ]),
                order_class: OrderClass::Strict,
                emit: Box::new(move |enc| {
                    enc.set_pipeline_state(p_rmsnorm_ffn);
                    enc.set_buffer(attn_proj_buf, 0, 0);
                    enc.set_buffer(layer_buf, ffn_norm_off, 1);
                    enc.set_buffer(normed_buf, 0, 2);
                    enc.set_bytes(&dim_u32.to_le_bytes(), 3);
                    enc.set_bytes(&eps.to_le_bytes(), 4);
                    enc.dispatch_threadgroups(
                        MTLSize::new(batch_size as u64, 1, 1),
                        MTLSize::new(norm_tg_size, 1, 1),
                    );
                    Ok(())
                }),
            });
        }

        // Op 18+: FFN gate+up [+swiglu]
        let ffn_n_u32 = inter_dim as u32;
        let ffn_k_u32 = hidden_dim as u32;
        // see the matching block above for rationale; same helpers.
        // BF16 fused-kernel arm mirrors site A in
        // `encode_layer_batched_concurrent` so the GDN-layer-shared FFN block also
        // honours `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_BF16=1` under the
        // default. Required because Qwen3.5-9B has 24 GDN layers
        // that walk this helper plus 8 full-attn layers that walk site A;
        // both must dispatch the fused kernel for end-to-end BF16 coverage.
        let fused_q8 = st.w_gate.quant == QuantScheme::Q8_0
            && st.w_up.quant == QuantScheme::Q8_0
            && hidden_dim % 64 == 0 && batch_size <= 4096
            && super::graph_reorder::ffn_gate_up_swiglu_fused_q8_enabled();
        let fused_q4 = st.w_gate.quant == QuantScheme::Q4_0
            && st.w_up.quant == QuantScheme::Q4_0
            && hidden_dim % 64 == 0 && batch_size <= 4096
            && super::graph_reorder::ffn_gate_up_swiglu_fused_q4_enabled();
        let fused_bf16 = st.w_gate.quant == QuantScheme::Bf16
            && st.w_up.quant == QuantScheme::Bf16
            && hidden_dim % 64 == 0 && batch_size <= 4096
            && super::graph_reorder::ffn_gate_up_swiglu_fused_bf16_enabled();
        let ffn_aligned = batch_size % 32 == 0 && inter_dim % 32 == 0 && hidden_dim % 32 == 0;
        if fused_q8 {
            if let Some(buf_gu) = repacked_gate_up {
                let pso = if ffn_aligned && hidden_dim % 64 == 0 {
                    &pipelines.dequant_tiled_matmul_q8_0_gate_up_swiglu_fused_packed_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q8_0_gate_up_swiglu_fused_packed
                };
                plan.push(LayerOp {
                    label: "ffn_gate_up_swiglu_fused_q8_packed",
                    accesses: AccessList::from_iter_inline([
                        Access::read(BufferId::Normed),
                        Access::write(BufferId::Gate),
                    ]),
                    order_class: OrderClass::Free,
                    emit: Box::new(move |enc| {
                        enc.set_pipeline_state(pso);
                        enc.set_threadgroup_memory_length(12288, 0);
                        enc.set_buffer(buf_gu, 0, 0);
                        enc.set_buffer(normed_buf, 0, 1);
                        enc.set_buffer(gate_buf, 0, 2);
                        enc.set_bytes(&m_u32.to_le_bytes(), 3);
                        enc.set_bytes(&ffn_n_u32.to_le_bytes(), 4);
                        enc.set_bytes(&ffn_k_u32.to_le_bytes(), 5);
                        enc.dispatch_threadgroups(
                            MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                            MTLSize::new(128, 1, 1),
                        );
                        Ok(())
                    }),
                });
            } else {
                let pso = if ffn_aligned && hidden_dim % 64 == 0 {
                    &pipelines.dequant_tiled_matmul_q8_0_gate_up_swiglu_fused_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q8_0_gate_up_swiglu_fused
                };
                plan.push(LayerOp {
                    label: "ffn_gate_up_swiglu_fused_q8",
                    accesses: AccessList::from_iter_inline([
                        Access::read(BufferId::Normed),
                        Access::write(BufferId::Gate),
                    ]),
                    order_class: OrderClass::Free,
                    emit: Box::new(move |enc| {
                        enc.set_pipeline_state(pso);
                        enc.set_threadgroup_memory_length(12288, 0);
                        enc.set_buffer(layer_buf, w_gate_off, 0);
                        enc.set_buffer(normed_buf, 0, 1);
                        enc.set_buffer(gate_buf, 0, 2);
                        enc.set_bytes(&m_u32.to_le_bytes(), 3);
                        enc.set_bytes(&ffn_n_u32.to_le_bytes(), 4);
                        enc.set_bytes(&ffn_k_u32.to_le_bytes(), 5);
                        enc.set_buffer(layer_buf, w_up_off, 6);
                        enc.dispatch_threadgroups(
                            MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                            MTLSize::new(128, 1, 1),
                        );
                        Ok(())
                    }),
                });
            }
        } else if fused_q4 {
            if let Some(buf_gu) = repacked_gate_up_q4 {
                let pso = if ffn_aligned && hidden_dim % 64 == 0 {
                    &pipelines.dequant_tiled_matmul_q4_0_gate_up_swiglu_fused_packed_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q4_0_gate_up_swiglu_fused_packed
                };
                plan.push(LayerOp {
                    label: "ffn_gate_up_swiglu_fused_q4_packed",
                    accesses: AccessList::from_iter_inline([
                        Access::read(BufferId::Normed),
                        Access::write(BufferId::Gate),
                    ]),
                    order_class: OrderClass::Free,
                    emit: Box::new(move |enc| {
                        enc.set_pipeline_state(pso);
                        enc.set_threadgroup_memory_length(12288, 0);
                        enc.set_buffer(buf_gu, 0, 0);
                        enc.set_buffer(normed_buf, 0, 1);
                        enc.set_buffer(gate_buf, 0, 2);
                        enc.set_bytes(&m_u32.to_le_bytes(), 3);
                        enc.set_bytes(&ffn_n_u32.to_le_bytes(), 4);
                        enc.set_bytes(&ffn_k_u32.to_le_bytes(), 5);
                        enc.dispatch_threadgroups(
                            MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                            MTLSize::new(128, 1, 1),
                        );
                        Ok(())
                    }),
                });
            } else {
                let pso = if ffn_aligned && hidden_dim % 64 == 0 {
                    &pipelines.dequant_tiled_matmul_q4_0_gate_up_swiglu_fused_aligned
                } else {
                    &pipelines.dequant_tiled_matmul_q4_0_gate_up_swiglu_fused
                };
                plan.push(LayerOp {
                    label: "ffn_gate_up_swiglu_fused_q4",
                    accesses: AccessList::from_iter_inline([
                        Access::read(BufferId::Normed),
                        Access::write(BufferId::Gate),
                    ]),
                    order_class: OrderClass::Free,
                    emit: Box::new(move |enc| {
                        enc.set_pipeline_state(pso);
                        enc.set_threadgroup_memory_length(12288, 0);
                        enc.set_buffer(layer_buf, w_gate_off, 0);
                        enc.set_buffer(normed_buf, 0, 1);
                        enc.set_buffer(gate_buf, 0, 2);
                        enc.set_bytes(&m_u32.to_le_bytes(), 3);
                        enc.set_bytes(&ffn_n_u32.to_le_bytes(), 4);
                        enc.set_bytes(&ffn_k_u32.to_le_bytes(), 5);
                        enc.set_buffer(layer_buf, w_up_off, 6);
                        enc.dispatch_threadgroups(
                            MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                            MTLSize::new(128, 1, 1),
                        );
                        Ok(())
                    }),
                });
            }
        } else if fused_bf16 {
            // BF16 fused gate+up+SwiGLU dispatch on the concurrent-encoder GDN-shared
            // FFN block. Same kernel selection / threadgroup geometry as site
            // A above; layer-buffer-resident weights with 12 KB shmem.
            let pso = if ffn_aligned && hidden_dim % 64 == 0 {
                &pipelines.bf16_matmul_gate_up_swiglu_fused_aligned
            } else {
                &pipelines.bf16_matmul_gate_up_swiglu_fused
            };
            plan.push(LayerOp {
                label: "ffn_gate_up_swiglu_fused_bf16",
                accesses: AccessList::from_iter_inline([
                    Access::read(BufferId::Normed),
                    Access::write(BufferId::Gate),
                ]),
                order_class: OrderClass::Free,
                emit: Box::new(move |enc| {
                    enc.set_pipeline_state(pso);
                    enc.set_threadgroup_memory_length(12288, 0);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(normed_buf, 0, 1);
                    enc.set_buffer(gate_buf, 0, 2);
                    enc.set_bytes(&m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&ffn_n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&ffn_k_u32.to_le_bytes(), 5);
                    enc.set_buffer(layer_buf, w_up_off, 6);
                    enc.dispatch_threadgroups(
                        MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                    Ok(())
                }),
            });
        } else {
            // Unfused path: gate + up + SwiGLU separately.
            plan.push(Self::concurrent_encoder_emit_gemm(
                "ffn_gate_gemm",
                BufferId::Normed, BufferId::Gate,
                layer_buf, w_gate_off, normed_buf, gate_buf,
                pipelines, st.w_gate.quant, m_u32, ffn_n_u32, ffn_k_u32,
                batch_size, inter_dim, hidden_dim,
                TILE_M, TILE_N,
            ));
            plan.push(Self::concurrent_encoder_emit_gemm(
                "ffn_up_gemm",
                BufferId::Normed, BufferId::Up,
                layer_buf, w_up_off, normed_buf, up_buf,
                pipelines, st.w_up.quant, m_u32, ffn_n_u32, ffn_k_u32,
                batch_size, inter_dim, hidden_dim,
                TILE_M, TILE_N,
            ));
            let p_swiglu = &pipelines.swiglu_batched;
            let total_elems = (batch_size * inter_dim) as u32;
            let tg_swiglu = 256u64.min(total_elems as u64).max(1);
            plan.push(LayerOp {
                label: "ffn_swiglu",
                accesses: AccessList::from_iter_inline([
                    Access::read(BufferId::Up),
                    Access::read_write(BufferId::Gate),
                ]),
                order_class: OrderClass::Strict,
                emit: Box::new(move |enc| {
                    enc.set_pipeline_state(p_swiglu);
                    enc.set_buffer(gate_buf, 0, 0);
                    enc.set_buffer(up_buf, 0, 1);
                    enc.set_bytes(&total_elems.to_le_bytes(), 2);
                    enc.dispatch_threadgroups(
                        MTLSize::new((total_elems as u64).div_ceil(tg_swiglu), 1, 1),
                        MTLSize::new(tg_swiglu, 1, 1),
                    );
                    Ok(())
                }),
            });
        }

        // Op final: FFN down + residual: gate @ wdown + attn_proj -> x_buf
        let dn_m_u32 = batch_size as u32;
        let dn_n_u32 = hidden_dim as u32;
        let dn_k_u32 = inter_dim as u32;
        let dn_k64_eligible_q8 = matches!(st.w_down.quant, QuantScheme::Q8_0)
            && inter_dim % 64 == 0 && batch_size <= 4096;
        let dn_k64_eligible_q4 = matches!(st.w_down.quant, QuantScheme::Q4_0)
            && inter_dim % 64 == 0 && batch_size <= 4096;
        // BF16 FFN-down Split-K K64 in the concurrent-encoder path. Mirrors the legacy
        // per-layer dispatch at site `prefill_encode.rs:~2675`. Eligibility matches
        // `ffn_down_splitk_bf16_value()` doc: BF16 weights, M <= 192, K >= 8192,
        // K % 64 == 0. Honors `LUMEN_METAL_FFN_DOWN_SPLITK_BF16=<2|4|8>`;
        // default OFF until empirical validation (per the env resolver, which
        // is intentionally NOT gated under metal_defaults_active).
        let ffn_splitk_bf16_env = super::graph_reorder::ffn_down_splitk_bf16_value();
        let dn_k64_eligible_bf16_splitk = ffn_splitk_bf16_env > 0
            && matches!(st.w_down.quant, QuantScheme::Bf16)
            && batch_size <= 192
            && inter_dim >= 8192
            && inter_dim % 64 == 0;
        if dn_k64_eligible_bf16_splitk {
            let k_splits = ffn_splitk_bf16_env;
            plan.push(LayerOp {
                label: "ffn_down_residual_splitk_bf16",
                accesses: AccessList::from_iter_inline([
                    Access::read(BufferId::Gate),
                    Access::write(BufferId::X),
                    Access::read(BufferId::AttnProj),
                ]),
                order_class: OrderClass::Strict,
                emit: Box::new(move |enc| {
                    Self::encode_splitk_bf16_gemm_k64_residual(
                        enc, pipelines,
                        layer_buf, w_down_off,
                        gate_buf, attn_proj_buf, x_buf, splitk_partial_buf,
                        dn_m_u32, dn_n_u32, dn_k_u32, k_splits,
                    );
                    Ok(())
                }),
            });
        } else if dn_k64_eligible_q8 && repacked_down.is_some() {
            let buf_d = repacked_down.unwrap();
            let dn_aligned = batch_size % 32 == 0 && hidden_dim % 32 == 0 && inter_dim % 32 == 0;
            let pso = if dn_aligned {
                &pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_packed_aligned
            } else {
                &pipelines.dequant_tiled_matmul_q8_0_k64_residual_batched_packed
            };
            plan.push(LayerOp {
                label: "ffn_down_residual_packed",
                accesses: AccessList::from_iter_inline([
                    Access::read(BufferId::Gate),
                    Access::write(BufferId::X),
                    Access::read(BufferId::AttnProj),
                ]),
                order_class: OrderClass::Strict,
                emit: Box::new(move |enc| {
                    enc.set_pipeline_state(pso);
                    enc.set_threadgroup_memory_length(8192, 0);
                    enc.set_buffer(buf_d, 0, 0);
                    enc.set_buffer(gate_buf, 0, 1);
                    enc.set_buffer(x_buf, 0, 2);
                    enc.set_bytes(&dn_m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&dn_n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&dn_k_u32.to_le_bytes(), 5);
                    enc.set_buffer(attn_proj_buf, 0, 6);
                    enc.dispatch_threadgroups(
                        MTLSize::new((hidden_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                    Ok(())
                }),
            });
        } else if dn_k64_eligible_q4 && repacked_down_q4.is_some() {
            let buf_d = repacked_down_q4.unwrap();
            let dn_aligned = batch_size % 32 == 0 && hidden_dim % 32 == 0 && inter_dim % 32 == 0;
            let pso = if dn_aligned {
                &pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched_packed_aligned
            } else {
                &pipelines.dequant_tiled_matmul_q4_0_k64_residual_batched_packed
            };
            plan.push(LayerOp {
                label: "ffn_down_residual_packed_q4",
                accesses: AccessList::from_iter_inline([
                    Access::read(BufferId::Gate),
                    Access::write(BufferId::X),
                    Access::read(BufferId::AttnProj),
                ]),
                order_class: OrderClass::Strict,
                emit: Box::new(move |enc| {
                    enc.set_pipeline_state(pso);
                    enc.set_threadgroup_memory_length(8192, 0);
                    enc.set_buffer(buf_d, 0, 0);
                    enc.set_buffer(gate_buf, 0, 1);
                    enc.set_buffer(x_buf, 0, 2);
                    enc.set_bytes(&dn_m_u32.to_le_bytes(), 3);
                    enc.set_bytes(&dn_n_u32.to_le_bytes(), 4);
                    enc.set_bytes(&dn_k_u32.to_le_bytes(), 5);
                    enc.set_buffer(attn_proj_buf, 0, 6);
                    enc.dispatch_threadgroups(
                        MTLSize::new((hidden_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                        MTLSize::new(128, 1, 1),
                    );
                    Ok(())
                }),
            });
        } else {
            plan.push(Self::concurrent_encoder_emit_gemm_residual(
                "ffn_down_residual",
                BufferId::Gate, BufferId::X, BufferId::AttnProj,
                layer_buf, w_down_off, gate_buf, x_buf, attn_proj_buf,
                pipelines, st.w_down.quant, dn_m_u32, dn_n_u32, dn_k_u32,
                batch_size, hidden_dim, inter_dim,
                TILE_M, TILE_N,
            ));
        }
    }

    // ========================================================================
    // whole-prefill outer encoder dispatcher.
    //
    // The whole-prefill caller in `prefill.rs::prefill()` opens ONE outer
    // concurrent compute encoder right after embed and threads it through
    // every layer. `encode_layer_batched_into` is the per-layer
    // dispatcher: it routes concurrent-encoder-eligible layers to `encode_layer_batched_concurrent`
    // and GDN linear-attn layers to `encode_batched_gdn_prefill`, in both
    // cases threading the outer encoder through. Cross-layer hazards are
    // emitted by the caller via `graph_reorder::emit_cross_layer_barrier`
    // on `BufferId::X` before each layer (except layer 0).
    //
    // The caller MUST pre-check eligibility via `layer_outer_eligible`
    // for every layer. If ANY layer is ineligible, the caller MUST fall back
    // to the legacy per-layer encoder path (because the outer encoder
    // cannot coexist with the legacy multi-encoder dispatch sites).
    // ========================================================================

    /// dispatch one layer's full op chain into the caller's outer
    /// concurrent encoder.
    ///
    /// Precondition: `layer_outer_eligible(scratch, weights, batch_size) == true`.
    /// (The caller has already validated this.)
    ///
    /// Two dispatch branches:
    /// 1. Full-attn / dense FFN (`!is_linear_attn`): defers to
    ///    `encode_layer_batched_concurrent(cmd, Some(outer_enc), ...)`. The concurrent-encoder
    ///    plan is constructed identically to the per-layer path but
    ///    emitted into `outer_enc` instead of a fresh per-layer encoder.
    /// 2. GDN linear-attn (`is_linear_attn`): resolves layer-buffer +
    ///    base offset, builds a `CachedLayerMeta` from the layer's
    ///    `SubtensorOffsets`, then calls
    ///    `encode_batched_gdn_prefill(cmd, Some(outer_enc), ...)`. The
    ///    construction code mirrors the legacy `encode_layer_batched`'s
    ///    `else` (linear-attn) branch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_layer_batched_into(
        &self,
        cmd: &MetalCommandBuffer,
        outer_enc: &MetalComputeEncoder,
        layer_idx: usize,
        batch_size: usize,
        weights: &LayerView,
        kv: &mut KvCacheView,
        pipelines: &MetalPipelines,
        scratch: &mut MetalScratch,
    ) -> Result<(), RuntimeError> {
        let st = &weights.subtensors;
        let is_linear_attn = st.layer_type == Some(1);

        if !is_linear_attn {
            // Full-attn / dense FFN: route through concurrent-encoder fast path with outer
            // encoder. `concurrent_encoder_layer_eligible` is guaranteed by the caller's
            // `layer_outer_eligible` precheck.
            return self.encode_layer_batched_concurrent(
                cmd, Some(outer_enc), layer_idx, batch_size, weights, kv, pipelines, scratch,
            );
        }

        // ---- GDN linear-attn path ----
        // Mirror the legacy `encode_layer_batched`'s linear-attn branch
        // construction of layer_buf / base_off / gdn_meta, then call
        // `encode_batched_gdn_prefill(cmd, Some(outer_enc), ...)`.
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
                    ": gpu_resident_layers missing layer {}", layer_idx)));
            }
        } else {
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

        let ffn_norm_off = if st.ffn_norm.length == 0 {
            st.attn_post_norm.map_or(0, |s| base_off + s.offset)
        } else {
            base_off + st.ffn_norm.offset
        };

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
            has_qgate_fusion: false,
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
                if let Some(Some(idx)) = scratch.gdn_layer_idx_map.get(layer_idx) {
                    Some(*idx)
                } else {
                    let mut gdn_count = 0usize;
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
                " GDN prefill: layer {} has layer_type=1 but no gdn_layer_idx",
                layer_idx
            ))
        })?;

        // Resolve batch buffers for the GDN call.
        let x_buf = scratch.batch_x_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute(" GDN: batch_x_buf not allocated".into())
        })?;
        let normed_buf = scratch.batch_normed_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute(" GDN: batch_normed_buf not allocated".into())
        })?;
        let qkv_buf = scratch.batch_qkv_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute(" GDN: batch_qkv_buf not allocated".into())
        })?;
        let attn_out_buf = scratch.batch_attn_out_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute(" GDN: batch_attn_out_buf not allocated".into())
        })?;
        let gate_buf = scratch.batch_gate_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute(" GDN: batch_gate_buf not allocated".into())
        })?;
        let attn_proj_buf = scratch.batch_attn_proj_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute(" GDN: batch_attn_proj_buf not allocated".into())
        })?;

        let seq_pos_start = kv.seq_len;

        super::profile::set_section("gdn/batched_prefill");
        let new_conv_pos = Self::encode_batched_gdn_prefill(
            cmd, Some(outer_enc), pipelines, scratch, layer_buf, &gdn_meta, gdn_idx,
            x_buf, normed_buf, qkv_buf, attn_out_buf, gate_buf, attn_proj_buf,
            batch_size,
        )?;
        scratch.gdn_conv_positions[gdn_idx] = new_conv_pos;

        // ---- FFN block ----
        //
        // GDN's `encode_batched_gdn_prefill` writes the residual into `x_buf`
        // (Phase 3 SSM_OUT + residual) and ALREADY runs the FFN-input RMSNorm
        // at its tail (writes `normed_buf` from `attn_proj_buf`). The legacy
        // `encode_layer_batched` for GDN layers continues into the dense FFN
        // block (gate+up + SwiGLU + down+residual) after the GDN return, and
        // must do the same — otherwise `x_buf` carries only the GDN/SSM
        // residual without the FFN contribution and downstream layers read
        // the wrong hidden state.
        //
        // We schedule the FFN ops into the outer encoder via the concurrent-encoder wavefront
        // scheduler so they interact correctly with the GDN function's
        // earlier dispatches (which the scheduler does NOT see; the
        // GDN function emitted them inline). Because the GDN function
        // already serialised everything in its tail (FFN-norm wrote
        // normed_buf last), and our outer-encoder cross-layer barrier in
        // `prefill.rs` will serialise before the next layer reads x_buf,
        // the only within-layer hazards are between the FFN ops themselves —
        // exactly what the scheduler handles.
        //
        // `skip_ffn_norm=true` because GDN already emitted it.
        super::profile::set_section("gdn/ffn_block");
        let ffn_norm_off = gdn_meta.ffn_norm_off;
        let w_gate_off = gdn_meta.w_gate_off;
        let w_up_off = gdn_meta.w_up_off;
        let w_down_off = gdn_meta.w_down_off;
        let up_buf = scratch.batch_up_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute(" GDN FFN: batch_up_buf not allocated".into())
        })?;
        let repacked_gate_up: Option<&MetalBuffer> = scratch.repacked_ffn_gate_up
            .get(layer_idx).and_then(|opt| opt.as_ref());
        let repacked_down: Option<&MetalBuffer> = scratch.repacked_ffn_down
            .get(layer_idx).and_then(|opt| opt.as_ref());
        let repacked_gate_up_q4: Option<&MetalBuffer> = scratch.repacked_ffn_gate_up_q4
            .get(layer_idx).and_then(|opt| opt.as_ref());
        let repacked_down_q4: Option<&MetalBuffer> = scratch.repacked_ffn_down_q4
            .get(layer_idx).and_then(|opt| opt.as_ref());
        // Split-K partial buffer for the BF16 FFN-down Split-K
        // dispatch inside the shared FFN helper.
        let splitk_partial_buf = scratch.splitk_partial_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute(" GDN FFN: splitk_partial_buf not allocated".into())
        })?;
        let mut ffn_plan: Vec<LayerOp<'_>> = Vec::with_capacity(4);
        Self::concurrent_encoder_extend_plan_with_ffn_block(
            &mut ffn_plan,
            true, // skip_ffn_norm — GDN already did it
            pipelines,
            st,
            layer_buf,
            normed_buf,
            attn_proj_buf,
            gate_buf,
            up_buf,
            x_buf,
            splitk_partial_buf,
            ffn_norm_off,
            w_gate_off,
            w_up_off,
            w_down_off,
            repacked_gate_up,
            repacked_down,
            repacked_gate_up_q4,
            repacked_down_q4,
            scratch.hidden_dim as u32,
            scratch.eps,
            scratch.norm_tg_size,
            batch_size as u32,
            batch_size,
            scratch.hidden_dim,
            scratch.inter_dim,
        );

        // Cross-section barrier: the GDN function ended with a dispatch
        // that wrote `normed_buf` (FFN-input RMSNorm). The concurrent-encoder scheduler
        // is invoked fresh for the FFN plan and doesn't know about that
        // prior dispatch, so it cannot emit an implicit `normed_buf`
        // barrier before the first FFN op (ffn_gate_up). On a concurrent
        // outer encoder, ffn_gate_up could read stale `normed_buf` before
        // GDN's FFN-norm write retires. Emit an explicit barrier here on
        // both buffers carried from the GDN function into the FFN plan:
        //   * `normed_buf`: written by GDN's FFN-norm, read by ffn_gate_up
        //   * `attn_proj_buf`: written by GDN's Phase 3 SSM_OUT+residual,
        //      read by ffn_down (residual addend)
        outer_enc.memory_barrier_with_resources(&[normed_buf, attn_proj_buf]);

        // Emit the FFN plan into the outer encoder. The concurrent-encoder wavefront
        // scheduler handles within-plan hazards (gate+up may run in
        // parallel; SwiGLU is Strict because it reads both; down+residual
        // is Strict). `serial_validate=false` is consistent with how
        // encode_layer_batched_concurrent calls the scheduler.
        let lookup = |id: BufferId| -> Option<&MetalBuffer> {
            Some(match id {
                BufferId::X        => x_buf,
                BufferId::Normed   => normed_buf,
                BufferId::AttnProj => attn_proj_buf,
                BufferId::Gate     => gate_buf,
                BufferId::Up       => up_buf,
                // The GDN FFN block doesn't touch Qkv / Q / K / V / AttnOut /
                // Scores / KCache / VCache. Returning None falls back to
                // whole-buffer scope on those (no-op since they're not in the
                // plan's access list).
                _ => return None,
            })
        };
        graph_reorder::emit_plan_into_encoder(outer_enc, &mut ffn_plan, false, Some(&lookup))?;

        // Update kv view position so the next layer (full-attn) sees the
        // correct seq_pos_start. The legacy `encode_layer_batched` does
        // this unconditionally for both branches; mirror that here. The
        // `kv` here is a per-layer `KvCacheView`, so this update is local
        // to the view (does not propagate to the parent KvCache — that
        // is done in `prefill.rs` via `kv.advance_seq_len()` after commit).
        kv.seq_len = seq_pos_start + batch_size;

        Ok(())
    }
}
