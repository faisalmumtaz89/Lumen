//! Batched prefill encoding methods for Metal backend.
//!
//! Extracted from mod.rs for modularity.
//! These are methods on MetalF32Backend that encode batched prefill GPU dispatch sequences.

use crate::error::RuntimeError;
use crate::kv::KvCacheView;
use crate::weight::cache::LayerView;
use crate::metal::ffi::{
    MetalBuffer, MetalCommandBuffer, MTLSize,
};
use lumen_format::quantization::QuantScheme;
use super::{MetalPipelines, MetalScratch, CachedLayerMeta, MetalF32Backend};

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

}
