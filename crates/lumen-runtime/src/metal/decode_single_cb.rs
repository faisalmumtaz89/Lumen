//! Single command-buffer decode path for Metal backend.
//!
//! Extracted from mod.rs for modularity.
//! Contains `decode_token_single_cb` which encodes embed + ALL layers + final
//! projection into a single Metal command buffer with one commit_and_wait().

use crate::compute::Logits;
use crate::error::RuntimeError;
use crate::metal::ffi::{
    MetalBuffer, MTLSize,
};
use lumen_format::quantization::QuantScheme;
use super::{MetalF32Backend, RouterLayerStats};

impl MetalF32Backend {
    /// Single command-buffer decode path.
    ///
    /// Encodes embed + ALL layers + final projection into ONE
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
}
