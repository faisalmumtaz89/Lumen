//! Single command-buffer decode path for Metal backend.
//!
//! Extracted from mod.rs for modularity.
//! Contains `decode_token_single_cb` which encodes embed + ALL layers + final
//! projection into a single Metal command buffer with one commit_and_wait().

use super::{MetalF32Backend, RouterLayerStats};
use crate::compute::Logits;
use crate::error::RuntimeError;
use crate::metal::decode_profile;
use crate::metal::ffi::{MTLSize, MetalBuffer};
use lumen_format::quantization::QuantScheme;

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
        weights: &dyn crate::weight::cache::WeightProvider,
        kv: &mut crate::kv::KvCache,
    ) -> Result<Logits, RuntimeError> {
        // Default (synchronous) entry point: async_commit = false. Preserves the
        // unchanged commit_and_wait() contract for all existing callers.
        self.decode_token_single_cb_inner(token_id, weights, kv, false)
    }

    /// Deferred-async-commit ("Option B") decode driver, 1-deep pipeline.
    ///
    /// Gated behind `LUMEN_METAL_DECODE_ASYNC_COMMIT=1` (the engine selects it).
    /// Encodes + commits token N's command buffer ASYNCHRONOUSLY (no per-token
    /// `commit_and_wait`), stashing it in `s.last_async_cmd`. The PREVIOUS token's
    /// CB is waited at the TOP of the next call (the existing `last_async_cmd`
    /// drain), and that previous token's logits are read from the ping-ponged
    /// `logits_buf` / `logits_buf_b` (double-buffered so N+1's lm_head does not
    /// clobber N's un-sampled logits). Returns the PREVIOUS token's logits; the
    /// engine drives the 1-token lag + final flush. The synchronous path is
    /// untouched and remains the default.
    pub fn decode_token_single_cb_async(
        &self,
        token_id: u32,
        weights: &dyn crate::weight::cache::WeightProvider,
        kv: &mut crate::kv::KvCache,
    ) -> Result<Logits, RuntimeError> {
        self.decode_token_single_cb_inner(token_id, weights, kv, true)
    }

    /// Shared decode body. `async_commit=false` => the original synchronous
    /// single-CB path (commit_and_wait, read this token's logits, return them).
    /// `async_commit=true` => the Option-B 1-deep deferred pipeline described on
    /// `decode_token_single_cb_async`.
    fn decode_token_single_cb_inner(
        &self,
        token_id: u32,
        _weights: &dyn crate::weight::cache::WeightProvider,
        kv: &mut crate::kv::KvCache,
        async_commit: bool,
    ) -> Result<Logits, RuntimeError> {
        let pipelines = self.pipelines.as_ref().ok_or_else(|| {
            RuntimeError::Compute("Metal pipelines not initialized: call init() first".into())
        })?;
        let embedding_buf = self
            .embedding_buf
            .as_ref()
            .ok_or_else(|| RuntimeError::Compute("Embedding buffer not initialized".into()))?;
        let final_norm_buf = self
            .final_norm_buf
            .as_ref()
            .ok_or_else(|| RuntimeError::Compute("Final norm buffer not initialized".into()))?;
        let output_proj_buf = self
            .output_proj_buf
            .as_ref()
            .ok_or_else(|| RuntimeError::Compute("Output proj buffer not initialized".into()))?;
        let output_proj_quant = self.output_proj_quant;

        let seq_pos = kv.seq_len();

        // Single mutex acquisition for the entire token.
        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard
            .as_mut()
            .ok_or_else(|| RuntimeError::Compute("Metal scratch not initialized".into()))?;

        // [Option B] Deferred-async-commit: when on, the PREVIOUS token's CB was
        // committed async and is still in flight. Wait it HERE (overlapping the
        // CPU work done between calls), then read the previous token's logits from
        // whichever buffer it wrote. `prev_logits` is returned at the end; THIS
        // token writes to the OTHER buffer so it can't clobber the value we just
        // read. If there is no in-flight CB (first async call after a sync drain),
        // `prev_logits` stays None and the engine treats this as a priming call.
        let mut prev_logits: Option<Vec<f32>> = None;
        if async_commit {
            let vocab_size = s.vocab_size;
            // Lazily allocate the second logits buffer the first time we need it.
            if s.logits_buf_b.is_none() {
                let n_bytes = s.logits_buf.length() as usize;
                s.logits_buf_b = self.device.new_buffer(n_bytes);
            }
            if let Some(prev_cmd) = s.last_async_cmd.take() {
                prev_cmd.wait_until_completed();
                // Read the in-flight token's logits from the buffer IT wrote to.
                let mut data = vec![0.0f32; vocab_size];
                if s.async_inflight_logits_b {
                    if let Some(ref b) = s.logits_buf_b {
                        b.read_f32(&mut data);
                    }
                } else {
                    s.logits_buf.read_f32(&mut data);
                }
                prev_logits = Some(data);
            }
        } else if let Some(prev_cmd) = s.last_async_cmd.take() {
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

        // [decode-split] STEP-1 diagnostic: mark the start of CPU encoding so we
        // can split per-token wall into CPU_encode (here -> commit) vs
        // commit_wait (the blocked commit_and_wait). No-op unless
        // LUMEN_METAL_DECODE_GPUTIME=1 (the Instant is only read on that path).
        let split_t0 = if decode_profile::gputime_enabled() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // ONE command buffer for embed + ALL layers + final projection.
        let mut cmd = self.queue.new_command_buffer().ok_or_else(|| {
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
        // MoE-only models (no GDN) keep the concurrent encoder for overlap.
        // GDN models use serial encoder for deterministic recurrent state (see below).
        let all_dense = s
            .cached_layer_meta
            .iter()
            .all(|m| m.gdn_layer_idx.is_none() && m.moe_meta.is_none());
        // GDN models MUST use a serial encoder for deterministic decode.
        // The GDN h_state recurrence accumulates floating-point values across
        // ALL tokens. With a concurrent encoder, Metal's non-deterministic
        // dispatch ordering introduces slight numerical variations in parallel
        // reductions (simd_sum, threadgroup partial sums). These variations
        // accumulate in h_state across tokens, causing the model to diverge
        // from the correct output after a few decode steps.
        // Dense models (no GDN, no MoE) already use serial for other reasons.
        // MoE-only models (no GDN) can safely use concurrent since MoE FFN
        // routing is stateless between tokens.
        let has_gdn = s
            .cached_layer_meta
            .iter()
            .any(|m| m.gdn_layer_idx.is_some());
        let needs_barriers = !all_dense && !has_gdn;
        let mut enc = if all_dense || has_gdn {
            cmd.new_compute_encoder()
                .ok_or_else(|| RuntimeError::Compute("Failed to create serial encoder".into()))?
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
                QuantScheme::Bf16 => enc.set_pipeline_state(&pipelines.embed_token_bf16),
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
            if needs_barriers {
                enc.memory_barrier_with_scope(1);
            }
        }

        // [decode-profile] Start timing with the embed section in-flight.
        decode_profile::begin("embed");

        // Zero the fused-router finish counter once per token.
        // The kernel self-resets to 0 after each layer, but we zero defensively
        // at token start (Shared buffer, not guaranteed zero-initialized; the CPU
        // write is visible to the GPU when this command buffer commits below).
        if super::moe_router_fused_enabled() {
            if let Some(counter) = s.moe_router_counter.as_ref() {
                unsafe {
                    *(counter.contents() as *mut u32) = 0u32;
                }
            }
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

            // [decode-profile] Section boundary at attention-block start.
            if decode_profile::is_enabled() {
                enc.end_encoding();
                cmd.commit_and_wait();
                let g = cmd.gpu_elapsed_secs();
                let lbl = if meta.gdn_layer_idx.is_some() {
                    "gdn_attn"
                } else {
                    "full_attn"
                };
                decode_profile::record_gpu(g, lbl);
                decode_profile::record_and_advance(lbl);
                cmd = self.queue.new_command_buffer().ok_or_else(|| {
                    RuntimeError::Compute("decode-profile: failed to create CB".into())
                })?;
                enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("decode-profile: failed to create encoder".into())
                })?;
            }

            // ================================================================
            // ATTENTION BLOCK
            // ================================================================
            if meta.gdn_layer_idx.is_none() {
                // Standard softmax attention path

                // Diagnostic sub-stage skip bitmask (LUMEN_METAL_FULLATTN_SUBSKIP).
                // When a bit is set the matching dispatch is skipped so its cost is
                // visible in the `full_attn` per-section GPU time. No-op when 0/unset.
                // Skipping corrupts output -- this is a timing-attribution tool only.
                // bit0 K proj, bit1 V proj, bit2 RoPE+KV-write, bit3 attention,
                // bit4 Q+gate proj, bit5 Wo proj, bit6 deinterleave/norm/assemble.
                const FULLATTN_SKIP_K: u32 = 1 << 0;
                const FULLATTN_SKIP_V: u32 = 1 << 1;
                const FULLATTN_SKIP_ROPE_KV: u32 = 1 << 2;
                const FULLATTN_SKIP_ATTN: u32 = 1 << 3;
                const FULLATTN_SKIP_QGATE: u32 = 1 << 4;
                const FULLATTN_SKIP_WO: u32 = 1 << 5;
                const FULLATTN_SKIP_DNA: u32 = 1 << 6;
                let fa_skip = decode_profile::fullattn_subskip();

                // Fused RMSNorm + QKV Q8_0 matvec.
                // Eliminates 1 dispatch + 1 barrier + normed_buf write/read per layer.
                // Also works for Q+gate fusion: all 3 matmuls (Q+gate, K, V) fuse
                // RMSNorm inline, reading x_buf directly. Eliminates separate RMSNorm
                // dispatch + barrier, and allows K/V to dispatch in parallel with Q+gate.
                let use_fused_attn_norm = matches!(
                    meta.wq_quant,
                    QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16 | QuantScheme::Bf16
                ) && !(meta.bq_off.is_some()
                    && meta.bk_off.is_some()
                    && meta.bv_off.is_some())
                    && (!meta.has_qgate_fusion
                        || (matches!(
                            meta.wk_quant,
                            Some(QuantScheme::Q8_0)
                                | Some(QuantScheme::Q4_0)
                                | Some(QuantScheme::F16)
                                | Some(QuantScheme::Bf16)
                        ) && matches!(
                            meta.wv_quant,
                            Some(QuantScheme::Q8_0)
                                | Some(QuantScheme::Q4_0)
                                | Some(QuantScheme::F16)
                                | Some(QuantScheme::Bf16)
                        )));

                if use_fused_attn_norm && !meta.has_qgate_fusion {
                    // Fused RMSNorm + QKV matvec NR2: reads x_buf, applies inline
                    // normalization (x[i]*scale*norm_w[i]), writes qkv_buf directly.
                    // (Q+gate fusion handles its own fused matmuls below.)
                    match meta.wq_quant {
                        QuantScheme::Q8_0 => enc.set_pipeline_state(
                            &pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2,
                        ),
                        QuantScheme::Q4_0 => enc.set_pipeline_state(
                            &pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2,
                        ),
                        QuantScheme::F16 => {
                            enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2)
                        }
                        QuantScheme::Bf16 => {
                            enc.set_pipeline_state(&pipelines.rmsnorm_matmul_bf16_deferred_nr2)
                        }
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
                    if needs_barriers {
                        enc.memory_barrier_with_scope(1);
                    }
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

                    // FULLY FUSED Q+gate+K+V projection (env LUMEN_METAL_Q4_QGATEKV_FUSE=1):
                    // ONE dispatch over the concatenated [qgate_dim + 2*kv_dim] row space
                    // (1280 TGs = ~42.7 SG/core for Qwen3.5-9B, past the occupancy knee that
                    // the separate dispatches under-occupy at). wq rows write qkv_buf, wk rows
                    // write k_buf, wv rows write v_buf -- the SAME three buffers the separate
                    // qmv_q4_0_rmsnorm dispatches write, and the SAME three the downstream
                    // deinterleave_norm_assemble reads. Byte-identical to the three separate
                    // dispatches (same per-row RMSNorm + -8 fold + accumulation order). Engages
                    // only when none of Q+gate/K/V is SUBSKIP-skipped, the fused-norm path is
                    // active, all three (wq/wk/wv) are Q4_0, and all three qmv buffers exist
                    // (wq under PROJ-or-this-flag, wk/wv under KV-or-this-flag at load time).
                    let qgatekv_fused = crate::metal::q4_qgatekv_fuse_enabled()
                        // FULLATTN_F16SC forces the SEPARATE-projection path: the fused
                        // qgatekv kernel reads f32 scales, but this flag builds wq/wk/wv
                        // scales as f16 -> mixing them would read f16-as-f32 (garbage).
                        && !crate::metal::q4_fullattn_f16sc_enabled()
                        && (fa_skip & (FULLATTN_SKIP_QGATE | FULLATTN_SKIP_K | FULLATTN_SKIP_V))
                            == 0
                        && use_fused_attn_norm
                        && meta.wq_quant == QuantScheme::Q4_0
                        && meta.wk_quant == Some(QuantScheme::Q4_0)
                        && meta.wv_quant == Some(QuantScheme::Q4_0)
                        && matches!(s.qmv_attn_wq_qw.get(layer_idx), Some(Some(_)))
                        && matches!(s.qmv_attn_wq_scales.get(layer_idx), Some(Some(_)))
                        && matches!(s.qmv_attn_wk_qw.get(layer_idx), Some(Some(_)))
                        && matches!(s.qmv_attn_wk_scales.get(layer_idx), Some(Some(_)))
                        && matches!(s.qmv_attn_wv_qw.get(layer_idx), Some(Some(_)))
                        && matches!(s.qmv_attn_wv_scales.get(layer_idx), Some(Some(_)));

                    // DIE-SATURATION LEVER (LUMEN_METAL_CONCURRENT_PROJ=1, default OFF):
                    // the three SEPARATE Q+gate/K/V projection matvecs all read the SAME
                    // x_buf and write DISJOINT buffers (qkv_buf/k_buf/v_buf) with no shared
                    // state -> they are independent. On the layer's SERIAL encoder they run
                    // one-at-a-time (sum of times); a single memory-bound matvec tops out
                    // ~50-60% of the M3 Ultra's two-die aggregate bandwidth. Dispatching the
                    // cluster on a CONCURRENT encoder lets Metal spread their threadgroups
                    // across both UltraFusion dies (finish in ~max instead of ~sum). Engages
                    // ONLY on the separate-projection path (not the fused QGATEKV/KV kernels,
                    // which are already one dispatch). Byte-identical to serial: disjoint
                    // outputs, shared read-only input, a resource-scoped barrier closes the
                    // cluster before the DNA consumer reads. The GDN recurrence is untouched.
                    let concurrent_proj_cluster = crate::metal::metal_concurrent_proj_enabled()
                        && use_fused_attn_norm
                        && !qgatekv_fused
                        && !crate::metal::q4_kv_fuse_enabled()
                        && (fa_skip & (FULLATTN_SKIP_QGATE | FULLATTN_SKIP_K | FULLATTN_SKIP_V))
                            == 0;
                    if concurrent_proj_cluster {
                        // Close the layer's current (serial) encoder and open a concurrent
                        // one for the projection cluster. Encoders within ONE command buffer
                        // execute in submission order, so prior dispatches (embed / previous
                        // layers) are ordered-before this cluster; x_buf is fully written.
                        enc.end_encoding();
                        enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute(
                                "CONCURRENT_PROJ: failed to create concurrent encoder".into(),
                            )
                        })?;
                    }

                    if qgatekv_fused {
                        let qw_q = s.qmv_attn_wq_qw[layer_idx].as_ref().unwrap();
                        let sc_q = s.qmv_attn_wq_scales[layer_idx].as_ref().unwrap();
                        let qw_k = s.qmv_attn_wk_qw[layer_idx].as_ref().unwrap();
                        let sc_k = s.qmv_attn_wk_scales[layer_idx].as_ref().unwrap();
                        let qw_v = s.qmv_attn_wv_qw[layer_idx].as_ref().unwrap();
                        let sc_v = s.qmv_attn_wv_scales[layer_idx].as_ref().unwrap();
                        // qmv_q4_0_rmsnorm_qgatekv: wq@0, x@1, q_out@2, in_dim@3, scales_q@4,
                        // norm_w@5, eps@6, wk@7, k_out@8, scales_k@9, wv@10, v_out@11,
                        // scales_v@12, qgate_dim@13, kv_dim@14.
                        enc.set_pipeline_state(&pipelines.qmv_q4_0_rmsnorm_qgatekv);
                        enc.set_buffer(qw_q, 0, 0);
                        enc.set_buffer(&s.x_buf, 0, 1);
                        enc.set_buffer(&s.qkv_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        enc.set_buffer(sc_q, 0, 4);
                        enc.set_buffer(layer_buf, attn_norm_off, 5);
                        enc.set_bytes(&eps.to_le_bytes(), 6);
                        enc.set_buffer(qw_k, 0, 7);
                        enc.set_buffer(&s.k_buf, 0, 8);
                        enc.set_buffer(sc_k, 0, 9);
                        enc.set_buffer(qw_v, 0, 10);
                        enc.set_buffer(&s.v_buf, 0, 11);
                        enc.set_buffer(sc_v, 0, 12);
                        enc.set_bytes(&(qgate_dim as u32).to_le_bytes(), 13);
                        enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 14);
                        // Grid covers (qgate_dim + 2*kv_dim) rows / 8 rows-per-TG.
                        enc.dispatch_threadgroups(
                            MTLSize::new((qgate_dim as u64 + 2 * kv_dim as u64) / 8, 1, 1),
                            MTLSize::new(64, 1, 1),
                        );
                    }

                    // Project Q+gate into qkv_buf
                    if !qgatekv_fused && fa_skip & FULLATTN_SKIP_QGATE == 0 {
                        // MLX-style fused RMSNorm+qmv fast path for the Q4_0 full-attn
                        // Q+gate projection when decode-qmv buffers exist (env
                        // LUMEN_METAL_Q4_QMV_PROJ=1) AND the fused-norm path would run
                        // (qmv fuses RMSNorm, valid ONLY when use_fused_attn_norm, never
                        // when reading normed_buf). Vec indexed by layer_idx. Mirrors
                        // decode_greedy.rs exactly.
                        let qmv_wq = if use_fused_attn_norm && meta.wq_quant == QuantScheme::Q4_0 {
                            match (
                                s.qmv_attn_wq_qw.get(layer_idx),
                                s.qmv_attn_wq_scales.get(layer_idx),
                            ) {
                                (Some(Some(qw)), Some(Some(sc))) => Some((qw, sc)),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        if let Some((qw, sc)) = qmv_wq {
                            // qmv_q4_0_rmsnorm: w@0, x@1, out@2, in_dim@3, scales@4,
                            // norm_w@5, eps@6. out = qgate_dim (%8); in = hidden (%512).
                            // F16-scales full-attn (LUMEN_METAL_Q4_FULLATTN_F16SC=1): f16sc
                            // kernel when its scale buffer was built f16 + kernel compiled.
                            if let Some(p) = crate::metal::q4_fullattn_f16sc_enabled()
                                .then_some(pipelines.qmv_q4_0_rmsnorm_f16sc.as_ref())
                                .flatten()
                            {
                                enc.set_pipeline_state(p);
                            } else {
                                enc.set_pipeline_state(&pipelines.qmv_q4_0_rmsnorm);
                            }
                            enc.set_buffer(qw, 0, 0);
                            enc.set_buffer(&s.x_buf, 0, 1);
                            enc.set_buffer(&s.qkv_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_buffer(sc, 0, 4);
                            enc.set_buffer(layer_buf, attn_norm_off, 5);
                            enc.set_bytes(&eps.to_le_bytes(), 6);
                            enc.dispatch_threadgroups(
                                MTLSize::new((qgate_dim as u64) / 8, 1, 1),
                                MTLSize::new(64, 1, 1),
                            );
                        } else {
                            if use_fused_attn_norm {
                                match meta.wq_quant {
                                    QuantScheme::Q4_0 => enc.set_pipeline_state(
                                        &pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2,
                                    ),
                                    QuantScheme::F16 => enc.set_pipeline_state(
                                        &pipelines.rmsnorm_matmul_f16_deferred_nr2,
                                    ),
                                    QuantScheme::Bf16 => enc.set_pipeline_state(
                                        &pipelines.rmsnorm_matmul_bf16_deferred_nr2,
                                    ),
                                    _ => enc.set_pipeline_state(
                                        &pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2,
                                    ),
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
                                    QuantScheme::Q8_0 => {
                                        enc.set_pipeline_state(
                                            &pipelines.dequant_matmul_q8_0_deferred_nr2,
                                        );
                                        128u64
                                    }
                                    QuantScheme::Q4_0 => {
                                        enc.set_pipeline_state(
                                            &pipelines.dequant_matmul_q4_0_deferred_nr2,
                                        );
                                        128u64
                                    }
                                    QuantScheme::F16 => {
                                        enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                                        128u64
                                    }
                                    QuantScheme::Bf16 => {
                                        enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_nr2);
                                        128u64
                                    }
                                    _ => {
                                        enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                                        matmul_tg_size
                                    }
                                };
                                enc.set_buffer(layer_buf, wq_off, 0);
                                enc.set_buffer(&s.normed_buf, 0, 1);
                                enc.set_buffer(&s.qkv_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                if matches!(
                                    meta.wq_quant,
                                    QuantScheme::Q8_0
                                        | QuantScheme::Q4_0
                                        | QuantScheme::F16
                                        | QuantScheme::Bf16
                                ) {
                                    enc.set_bytes(&(qgate_dim as u32).to_le_bytes(), 4);
                                }
                            }
                            let n_tg = match meta.wq_quant {
                                QuantScheme::Q8_0 => ((qgate_dim as u64) + 1) / 2,
                                QuantScheme::Q4_0 => ((qgate_dim as u64) + 1) / 2,
                                QuantScheme::F16 => ((qgate_dim as u64) + 1) / 2,
                                QuantScheme::Bf16 => ((qgate_dim as u64) + 1) / 2,
                                _ => qgate_dim as u64,
                            };
                            enc.dispatch_threadgroups(
                                MTLSize::new(n_tg, 1, 1),
                                MTLSize::new(128, 1, 1),
                            );
                        }
                    }
                    // FUSED K+V projection (env LUMEN_METAL_Q4_KV_FUSE=1): ONE dispatch
                    // over [2*kv_dim] rows (K rows write k_buf, V rows write v_buf) to
                    // double the threadgroup occupancy of the row-starved K/V matvecs
                    // (kv_dim=1024 = ~4.3 SG/core each -> ~8.5 SG/core fused). Byte-identical
                    // to the two separate qmv_q4_0_rmsnorm dispatches (same per-row math).
                    // Only when neither K nor V is SUBSKIP-skipped, the fused-norm path is
                    // active, both are Q4_0, and both wk/wv qmv buffers exist.
                    let kv_fused = !qgatekv_fused
                        && crate::metal::q4_kv_fuse_enabled()
                        // FULLATTN_F16SC builds wk/wv scales as f16; the fused kv kernel
                        // reads f32 scales -> force the separate f16sc-aware K/V dispatches.
                        && !crate::metal::q4_fullattn_f16sc_enabled()
                        && (fa_skip & (FULLATTN_SKIP_K | FULLATTN_SKIP_V)) == 0
                        && use_fused_attn_norm
                        && meta.wk_quant == Some(QuantScheme::Q4_0)
                        && meta.wv_quant == Some(QuantScheme::Q4_0)
                        && matches!(s.qmv_attn_wk_qw.get(layer_idx), Some(Some(_)))
                        && matches!(s.qmv_attn_wk_scales.get(layer_idx), Some(Some(_)))
                        && matches!(s.qmv_attn_wv_qw.get(layer_idx), Some(Some(_)))
                        && matches!(s.qmv_attn_wv_scales.get(layer_idx), Some(Some(_)));
                    if kv_fused {
                        let qw_k = s.qmv_attn_wk_qw[layer_idx].as_ref().unwrap();
                        let sc_k = s.qmv_attn_wk_scales[layer_idx].as_ref().unwrap();
                        let qw_v = s.qmv_attn_wv_qw[layer_idx].as_ref().unwrap();
                        let sc_v = s.qmv_attn_wv_scales[layer_idx].as_ref().unwrap();
                        // qmv_q4_0_rmsnorm_kv: wk@0, x@1, k_out@2, in_dim@3, scales_k@4,
                        // norm_w@5, eps@6, wv@7, v_out@8, scales_v@9, out_dim_kv@10.
                        enc.set_pipeline_state(&pipelines.qmv_q4_0_rmsnorm_kv);
                        enc.set_buffer(qw_k, 0, 0);
                        enc.set_buffer(&s.x_buf, 0, 1);
                        enc.set_buffer(&s.k_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        enc.set_buffer(sc_k, 0, 4);
                        enc.set_buffer(layer_buf, attn_norm_off, 5);
                        enc.set_bytes(&eps.to_le_bytes(), 6);
                        enc.set_buffer(qw_v, 0, 7);
                        enc.set_buffer(&s.v_buf, 0, 8);
                        enc.set_buffer(sc_v, 0, 9);
                        enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 10);
                        // Grid covers 2*kv_dim rows / 8 rows-per-TG.
                        enc.dispatch_threadgroups(
                            MTLSize::new((2 * kv_dim as u64) / 8, 1, 1),
                            MTLSize::new(64, 1, 1),
                        );
                    }
                    // Project K from wk (parallel with Q+gate when fused).
                    // Skipped when the full Q+gate+K+V fused kernel already produced k_buf.
                    if !qgatekv_fused && !kv_fused && fa_skip & FULLATTN_SKIP_K == 0 {
                        let wk_off_val = meta.wk_off.unwrap();
                        let wk_quant = meta.wk_quant.unwrap();
                        // MLX-style fused RMSNorm+qmv fast path for the Q4_0 K projection
                        // when decode-qmv buffers exist (env LUMEN_METAL_Q4_QMV_KV=1).
                        // Reads x_buf (pre-norm hidden) like Q does; writes k_buf at
                        // offset 0 exactly as the NR2 path. Valid ONLY when the fused-norm
                        // path would run (qmv fuses RMSNorm). Indexed by layer_idx.
                        let qmv_wk = if use_fused_attn_norm && wk_quant == QuantScheme::Q4_0 {
                            match (
                                s.qmv_attn_wk_qw.get(layer_idx),
                                s.qmv_attn_wk_scales.get(layer_idx),
                            ) {
                                (Some(Some(qw)), Some(Some(sc))) => Some((qw, sc)),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        if let Some((qw, sc)) = qmv_wk {
                            // qmv_q4_0_rmsnorm: w@0, x@1, out@2, in_dim@3, scales@4,
                            // norm_w@5, eps@6. out = kv_dim (%8); in = hidden (%512).
                            // F16-scales full-attn K (LUMEN_METAL_Q4_FULLATTN_F16SC=1).
                            if let Some(p) = crate::metal::q4_fullattn_f16sc_enabled()
                                .then_some(pipelines.qmv_q4_0_rmsnorm_f16sc.as_ref())
                                .flatten()
                            {
                                enc.set_pipeline_state(p);
                            } else {
                                enc.set_pipeline_state(&pipelines.qmv_q4_0_rmsnorm);
                            }
                            enc.set_buffer(qw, 0, 0);
                            enc.set_buffer(&s.x_buf, 0, 1);
                            enc.set_buffer(&s.k_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_buffer(sc, 0, 4);
                            enc.set_buffer(layer_buf, attn_norm_off, 5);
                            enc.set_bytes(&eps.to_le_bytes(), 6);
                            enc.dispatch_threadgroups(
                                MTLSize::new((kv_dim as u64) / 8, 1, 1),
                                MTLSize::new(64, 1, 1),
                            );
                        } else if use_fused_attn_norm {
                            match wk_quant {
                                QuantScheme::Q4_0 => enc.set_pipeline_state(
                                    &pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2,
                                ),
                                QuantScheme::F16 => enc
                                    .set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                                QuantScheme::Bf16 => enc.set_pipeline_state(
                                    &pipelines.rmsnorm_matmul_bf16_deferred_nr2,
                                ),
                                _ => enc.set_pipeline_state(
                                    &pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2,
                                ),
                            }
                            enc.set_buffer(layer_buf, wk_off_val, 0);
                            enc.set_buffer(&s.x_buf, 0, 1);
                            enc.set_buffer(&s.k_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                            enc.set_buffer(layer_buf, attn_norm_off, 5);
                            enc.set_bytes(&eps.to_le_bytes(), 6);
                            enc.dispatch_threadgroups(
                                MTLSize::new(((kv_dim as u64) + 1) / 2, 1, 1),
                                MTLSize::new(128, 1, 1),
                            );
                        } else {
                            let _tg = match wk_quant {
                                QuantScheme::Q8_0 => {
                                    enc.set_pipeline_state(
                                        &pipelines.dequant_matmul_q8_0_deferred_nr2,
                                    );
                                    128u64
                                }
                                QuantScheme::Q4_0 => {
                                    enc.set_pipeline_state(
                                        &pipelines.dequant_matmul_q4_0_deferred_nr2,
                                    );
                                    128u64
                                }
                                QuantScheme::F16 => {
                                    enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                                    128u64
                                }
                                QuantScheme::Bf16 => {
                                    enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_nr2);
                                    128u64
                                }
                                _ => {
                                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                                    matmul_tg_size
                                }
                            };
                            enc.set_buffer(layer_buf, wk_off_val, 0);
                            enc.set_buffer(&s.normed_buf, 0, 1);
                            enc.set_buffer(&s.k_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            if matches!(
                                wk_quant,
                                QuantScheme::Q8_0
                                    | QuantScheme::Q4_0
                                    | QuantScheme::F16
                                    | QuantScheme::Bf16
                            ) {
                                enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                            }
                            let n_tg = match wk_quant {
                                QuantScheme::Q8_0 => ((kv_dim as u64) + 1) / 2,
                                QuantScheme::Q4_0 => ((kv_dim as u64) + 1) / 2,
                                QuantScheme::F16 => ((kv_dim as u64) + 1) / 2,
                                QuantScheme::Bf16 => ((kv_dim as u64) + 1) / 2,
                                _ => kv_dim as u64,
                            };
                            enc.dispatch_threadgroups(
                                MTLSize::new(n_tg, 1, 1),
                                MTLSize::new(128, 1, 1),
                            );
                        }
                    }
                    // Project V from wv (parallel with Q+gate and K when fused).
                    // Skipped when the fused K+V (or full Q+gate+K+V) kernel produced v_buf.
                    if !qgatekv_fused && !kv_fused && fa_skip & FULLATTN_SKIP_V == 0 {
                        let wv_off_val = meta.wv_off.unwrap();
                        let wv_quant = meta.wv_quant.unwrap();
                        // MLX-style fused RMSNorm+qmv fast path for the Q4_0 V projection
                        // when decode-qmv buffers exist (env LUMEN_METAL_Q4_QMV_KV=1).
                        // Reads x_buf (pre-norm hidden) like Q does; writes v_buf at
                        // offset 0 exactly as the NR2 path. Indexed by layer_idx.
                        let qmv_wv = if use_fused_attn_norm && wv_quant == QuantScheme::Q4_0 {
                            match (
                                s.qmv_attn_wv_qw.get(layer_idx),
                                s.qmv_attn_wv_scales.get(layer_idx),
                            ) {
                                (Some(Some(qw)), Some(Some(sc))) => Some((qw, sc)),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        if let Some((qw, sc)) = qmv_wv {
                            // qmv_q4_0_rmsnorm: w@0, x@1, out@2, in_dim@3, scales@4,
                            // norm_w@5, eps@6. out = kv_dim (%8); in = hidden (%512).
                            // F16-scales full-attn V (LUMEN_METAL_Q4_FULLATTN_F16SC=1).
                            if let Some(p) = crate::metal::q4_fullattn_f16sc_enabled()
                                .then_some(pipelines.qmv_q4_0_rmsnorm_f16sc.as_ref())
                                .flatten()
                            {
                                enc.set_pipeline_state(p);
                            } else {
                                enc.set_pipeline_state(&pipelines.qmv_q4_0_rmsnorm);
                            }
                            enc.set_buffer(qw, 0, 0);
                            enc.set_buffer(&s.x_buf, 0, 1);
                            enc.set_buffer(&s.v_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_buffer(sc, 0, 4);
                            enc.set_buffer(layer_buf, attn_norm_off, 5);
                            enc.set_bytes(&eps.to_le_bytes(), 6);
                            enc.dispatch_threadgroups(
                                MTLSize::new((kv_dim as u64) / 8, 1, 1),
                                MTLSize::new(64, 1, 1),
                            );
                        } else if use_fused_attn_norm {
                            match wv_quant {
                                QuantScheme::Q4_0 => enc.set_pipeline_state(
                                    &pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2,
                                ),
                                QuantScheme::F16 => enc
                                    .set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2),
                                QuantScheme::Bf16 => enc.set_pipeline_state(
                                    &pipelines.rmsnorm_matmul_bf16_deferred_nr2,
                                ),
                                _ => enc.set_pipeline_state(
                                    &pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2,
                                ),
                            }
                            enc.set_buffer(layer_buf, wv_off_val, 0);
                            enc.set_buffer(&s.x_buf, 0, 1);
                            enc.set_buffer(&s.v_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                            enc.set_buffer(layer_buf, attn_norm_off, 5);
                            enc.set_bytes(&eps.to_le_bytes(), 6);
                            enc.dispatch_threadgroups(
                                MTLSize::new(((kv_dim as u64) + 1) / 2, 1, 1),
                                MTLSize::new(128, 1, 1),
                            );
                        } else {
                            let _tg = match wv_quant {
                                QuantScheme::Q8_0 => {
                                    enc.set_pipeline_state(
                                        &pipelines.dequant_matmul_q8_0_deferred_nr2,
                                    );
                                    128u64
                                }
                                QuantScheme::Q4_0 => {
                                    enc.set_pipeline_state(
                                        &pipelines.dequant_matmul_q4_0_deferred_nr2,
                                    );
                                    128u64
                                }
                                QuantScheme::F16 => {
                                    enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                                    128u64
                                }
                                QuantScheme::Bf16 => {
                                    enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_nr2);
                                    128u64
                                }
                                _ => {
                                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                                    matmul_tg_size
                                }
                            };
                            enc.set_buffer(layer_buf, wv_off_val, 0);
                            enc.set_buffer(&s.normed_buf, 0, 1);
                            enc.set_buffer(&s.v_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            if matches!(
                                wv_quant,
                                QuantScheme::Q8_0
                                    | QuantScheme::Q4_0
                                    | QuantScheme::F16
                                    | QuantScheme::Bf16
                            ) {
                                enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                            }
                            let n_tg = match wv_quant {
                                QuantScheme::Q8_0 => ((kv_dim as u64) + 1) / 2,
                                QuantScheme::Q4_0 => ((kv_dim as u64) + 1) / 2,
                                QuantScheme::F16 => ((kv_dim as u64) + 1) / 2,
                                QuantScheme::Bf16 => ((kv_dim as u64) + 1) / 2,
                                _ => kv_dim as u64,
                            };
                            enc.dispatch_threadgroups(
                                MTLSize::new(n_tg, 1, 1),
                                MTLSize::new(128, 1, 1),
                            );
                        }
                    }
                    // Barrier: Q+gate/K/V projections all complete
                    if concurrent_proj_cluster {
                        // Resource-scoped barrier on the three disjoint outputs, then close
                        // the concurrent encoder and reopen a serial one so the rest of the
                        // layer (DNA, attention, recurrence) runs with serial-encoder
                        // completion ordering exactly as before.
                        enc.memory_barrier_with_resources(&[&s.qkv_buf, &s.k_buf, &s.v_buf]);
                        enc.end_encoding();
                        enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute(
                                "CONCURRENT_PROJ: failed to reopen serial encoder".into(),
                            )
                        })?;
                    } else if needs_barriers {
                        enc.memory_barrier_with_scope(1);
                    }

                    // Fused deinterleave + norm + assemble (saves 5 dispatches + 2 barriers per layer).
                    // Falls back to separate dispatches if fused kernel or norm weights unavailable.
                    let use_fused_dna = pipelines.deinterleave_norm_assemble.is_some()
                        && meta.attn_q_norm_off.is_some()
                        && meta.attn_k_norm_off.is_some();

                    if fa_skip & FULLATTN_SKIP_DNA != 0 {
                        // diagnostic: skip deinterleave/norm/assemble (corrupts output)
                    } else if use_fused_dna {
                        let pso = pipelines.deinterleave_norm_assemble.as_ref().unwrap();
                        let q_norm_off = meta.attn_q_norm_off.unwrap();
                        let k_norm_off = meta.attn_k_norm_off.unwrap();
                        enc.set_pipeline_state(pso);
                        enc.set_buffer(&s.qkv_buf, 0, 0); // qgate_interleaved (input)
                        enc.set_buffer(&s.k_buf, 0, 1); // k_data
                        enc.set_buffer(&s.v_buf, 0, 2); // v_data
                        enc.set_buffer(layer_buf, q_norm_off, 3); // q_norm_weight
                        enc.set_buffer(layer_buf, k_norm_off, 4); // k_norm_weight
                                                                  // Determinism fix: qkv_out (buffer 5) is UNUSED post-fix (the
                                                                  // kernel no longer writes the assembled K/V into qkv_buf, which would
                                                                  // alias the qgate read). K is normalized in-place into k_buf, V stays
                                                                  // in v_buf; both are copied into qkv_buf below AFTER the encoder barrier.
                        enc.set_buffer(&s.qkv_buf, 0, 5); // qkv_out (unused; ABI-stable)
                        enc.set_buffer(&s.gate_buf, 0, 6); // gate_out
                        enc.set_bytes(&(num_heads as u32).to_le_bytes(), 7);
                        enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 8);
                        enc.set_bytes(&(head_dim as u32).to_le_bytes(), 9);
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 10);
                        enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 11);
                        enc.set_bytes(&eps.to_le_bytes(), 12);
                        enc.set_buffer(&s.q_buf, 0, 13); // q_out (separate to avoid aliasing)
                        let total_tgs = (num_heads + num_kv_heads) as u64;
                        let tg_threads = 256u64.min(head_dim as u64).max(32);
                        enc.dispatch_threadgroups(
                            MTLSize::new(total_tgs, 1, 1),
                            MTLSize::new(tg_threads, 1, 1),
                        );
                        // Assemble Q/K/V into qkv_buf from the SEPARATE q_buf/k_buf/v_buf,
                        // AFTER the DNA kernel's qgate reads have all retired (serial-encoder
                        // hazard tracking orders these qkv_buf writes after the qkv_buf read).
                        if needs_barriers {
                            enc.memory_barrier_with_scope(1);
                        }
                        // Assemble Q/K/V into the contiguous qkv_buf via three copies
                        // from the separate q_buf/k_buf/v_buf at offsets 0 / q_dim /
                        // q_dim+kv_dim.
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
                        // Copy normalized K from k_buf to qkv_buf[q_dim..q_dim+kv_dim).
                        enc.set_pipeline_state(&pipelines.copy_buffer);
                        enc.set_buffer(&s.k_buf, 0, 0);
                        enc.set_buffer(&s.qkv_buf, k_byte_off, 1);
                        {
                            let tg = 256u64.min(kv_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((kv_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                        // Copy V from v_buf to qkv_buf[q_dim+kv_dim..q_dim+2*kv_dim).
                        enc.set_pipeline_state(&pipelines.copy_buffer);
                        enc.set_buffer(&s.v_buf, 0, 0);
                        enc.set_buffer(&s.qkv_buf, v_byte_off, 1);
                        {
                            let tg = 256u64.min(kv_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((kv_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                    } else {
                        // Fallback: separate deinterleave + norm + copy
                        {
                            let pso = pipelines.deinterleave_qgate.as_ref().ok_or_else(|| {
                                RuntimeError::Compute(
                                    "deinterleave_qgate pipeline not compiled".into(),
                                )
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
                        if needs_barriers {
                            enc.memory_barrier_with_scope(1);
                        }
                        if let (Some(q_norm_off), Some(k_norm_off)) =
                            (meta.attn_q_norm_off, meta.attn_k_norm_off)
                        {
                            let pso = pipelines.rmsnorm_per_head.as_ref().ok_or_else(|| {
                                RuntimeError::Compute(
                                    "rmsnorm_per_head pipeline not compiled".into(),
                                )
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
                        if needs_barriers {
                            enc.memory_barrier_with_scope(1);
                        }
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
                        let has_bias =
                            meta.bq_off.is_some() && meta.bk_off.is_some() && meta.bv_off.is_some();
                        let tg = if has_bias
                            && matches!(
                                meta.wq_quant,
                                QuantScheme::Q8_0
                                    | QuantScheme::Q4_0
                                    | QuantScheme::F16
                                    | QuantScheme::Bf16
                            ) {
                            match meta.wq_quant {
                                QuantScheme::Q8_0 => enc.set_pipeline_state(
                                    &pipelines.dequant_matmul_q8_0_deferred_bias_nr2,
                                ),
                                QuantScheme::Q4_0 => enc.set_pipeline_state(
                                    &pipelines.dequant_matmul_q4_0_deferred_bias_nr2,
                                ),
                                QuantScheme::F16 => {
                                    enc.set_pipeline_state(&pipelines.matmul_f16_deferred_bias_nr2)
                                }
                                QuantScheme::Bf16 => {
                                    enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_bias_nr2)
                                }
                                _ => unreachable!(),
                            };
                            128u64
                        } else {
                            match meta.wq_quant {
                                QuantScheme::Q8_0 => {
                                    enc.set_pipeline_state(
                                        &pipelines.dequant_matmul_q8_0_deferred_nr2,
                                    );
                                    128u64
                                }
                                QuantScheme::Q4_0 => {
                                    enc.set_pipeline_state(
                                        &pipelines.dequant_matmul_q4_0_deferred_nr2,
                                    );
                                    128u64
                                }
                                QuantScheme::F16 => {
                                    enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                                    128u64
                                }
                                QuantScheme::Bf16 => {
                                    enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_nr2);
                                    128u64
                                }
                                _ => {
                                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                                    matmul_tg_size
                                }
                            }
                        };
                        enc.set_buffer(layer_buf, wq_off, 0);
                        enc.set_buffer(&s.normed_buf, 0, 1);
                        enc.set_buffer(&s.qkv_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        if matches!(
                            meta.wq_quant,
                            QuantScheme::Q8_0
                                | QuantScheme::Q4_0
                                | QuantScheme::F16
                                | QuantScheme::Bf16
                        ) {
                            enc.set_bytes(&(qkv_dim as u32).to_le_bytes(), 4);
                        }
                        if has_bias
                            && matches!(
                                meta.wq_quant,
                                QuantScheme::Q8_0
                                    | QuantScheme::Q4_0
                                    | QuantScheme::F16
                                    | QuantScheme::Bf16
                            )
                        {
                            enc.set_buffer(layer_buf, meta.bq_off.unwrap(), 5);
                            enc.set_buffer(layer_buf, meta.bk_off.unwrap(), 6);
                            enc.set_buffer(layer_buf, meta.bv_off.unwrap(), 7);
                            enc.set_bytes(&(q_dim as u32).to_le_bytes(), 8);
                            let qk_dim = (q_dim + kv_dim) as u32;
                            enc.set_bytes(&qk_dim.to_le_bytes(), 9);
                        }
                        let n_tg = if tg == 64 {
                            ((qkv_dim as u64) + 7) / 8 // (dead path: Q8_0 now uses deferred with tg=128)
                        } else {
                            match meta.wq_quant {
                                QuantScheme::Q8_0 => ((qkv_dim as u64) + 1) / 2,
                                QuantScheme::Q4_0 => ((qkv_dim as u64) + 1) / 2,
                                QuantScheme::F16 => ((qkv_dim as u64) + 1) / 2,
                                QuantScheme::Bf16 => ((qkv_dim as u64) + 1) / 2,
                                _ => qkv_dim as u64,
                            }
                        };
                        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(tg, 1, 1));
                    }

                    // QKV bias addition fallback (only for F32 weights with bias, rare)
                    if !matches!(
                        meta.wq_quant,
                        QuantScheme::Q8_0
                            | QuantScheme::Q4_0
                            | QuantScheme::F16
                            | QuantScheme::Bf16
                    ) && (meta.bq_off.is_some()
                        || meta.bk_off.is_some()
                        || meta.bv_off.is_some())
                    {
                        enc.set_pipeline_state(&pipelines.bias_add);
                        if let Some(bq_off) = meta.bq_off {
                            enc.set_buffer(&s.qkv_buf, q_byte_off, 0);
                            enc.set_buffer(layer_buf, bq_off, 1);
                            enc.set_bytes(&(q_dim as u32).to_le_bytes(), 2);
                            let n_tg_bq = (q_dim as u64 + 255) / 256;
                            enc.dispatch_threadgroups(
                                MTLSize::new(n_tg_bq, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        }
                        if let Some(bk_off) = meta.bk_off {
                            enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
                            enc.set_buffer(layer_buf, bk_off, 1);
                            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 2);
                            let n_tg_bk = (kv_dim as u64 + 255) / 256;
                            enc.dispatch_threadgroups(
                                MTLSize::new(n_tg_bk, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        }
                        if let Some(bv_off) = meta.bv_off {
                            enc.set_buffer(&s.qkv_buf, v_byte_off, 0);
                            enc.set_buffer(layer_buf, bv_off, 1);
                            enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 2);
                            let n_tg_bv = (kv_dim as u64 + 255) / 256;
                            enc.dispatch_threadgroups(
                                MTLSize::new(n_tg_bv, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        }
                    }
                }

                // Barrier: QKV projection writes qkv_buf, RoPE reads qkv_buf
                if needs_barriers {
                    enc.memory_barrier_with_scope(1);
                }

                // Fused RoPE Q + RoPE K + KV cache write (1 dispatch instead of 3)
                // Only used for full RoPE (rotary_dim == head_dim) on non-linear attention layers.
                // Partial RoPE (Qwen3.5-MoE) and linear attention layers fall back to separate dispatches.
                let is_linear_attn = meta.layer_type == Some(1);
                let rope_half_dim = s.rotary_dim / 2;
                let use_fused_rope_kv = !is_linear_attn && s.rotary_dim == head_dim;
                const FLASH_DECODE_THRESHOLD: usize = 257; // FLASH_DECODE_TILE_SIZE + 1: single-tile flash_decode is a no-op reduce

                // Fused RoPE + KV cache write + MHA (eliminates 2 barriers per layer)
                // Only for: standard RoPE (not NeoX), short sequences, full rotary_dim.
                let use_fused_rope_kv_mha =
                    use_fused_rope_kv && !s.rope_neox && new_seq_len < FLASH_DECODE_THRESHOLD;

                if use_fused_rope_kv_mha {
                    // Single dispatch: RoPE Q/K + KV cache write + MHA.
                    // LUMEN_METAL_FUSED_SDPA_DECODE swaps in the threadgroup-scores
                    // variant (scores on-chip, no device round-trip) -- byte-identical.
                    let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
                    if crate::metal::fused_sdpa_decode_enabled() {
                        enc.set_pipeline_state(&pipelines.fused_rope_kv_mha_tgscores);
                    } else {
                        enc.set_pipeline_state(&pipelines.fused_rope_kv_mha);
                    }
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
                    // Diagnostic: skip attention compute (FULLATTN_SKIP_ATTN bit3) so the
                    // attention-compute sub-stage cost is the full vs skip-attn delta. 0 = full run.
                    let skip_attn_u32: u32 = if fa_skip & FULLATTN_SKIP_ATTN != 0 {
                        1
                    } else {
                        0
                    };
                    enc.set_bytes(&skip_attn_u32.to_le_bytes(), 18);
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
                    if fa_skip & FULLATTN_SKIP_ROPE_KV != 0 {
                        // diagnostic: skip RoPE + KV-cache write (corrupts output)
                    } else if use_fused_rope_kv {
                        let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
                        let fused_pipe = if s.rope_neox {
                            pipelines
                                .fused_rope_neox_kv_write
                                .as_ref()
                                .unwrap_or(&pipelines.fused_rope_kv_write)
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
                        let total_threads = (num_heads * rope_half_dim
                            + num_kv_heads * rope_half_dim
                            + kv_dim) as u64;
                        let tg = 64u64.min(total_threads.max(1));
                        enc.dispatch_threadgroups(
                            MTLSize::new(total_threads.div_ceil(tg), 1, 1),
                            MTLSize::new(tg, 1, 1),
                        );
                    } else {
                        if !is_linear_attn {
                            let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
                            let rope_pipe = if s.rope_neox {
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
                    if needs_barriers {
                        enc.memory_barrier_with_scope(1);
                    }
                    // Attention (flash decode or MHA)
                    if fa_skip & FULLATTN_SKIP_ATTN == 0 {
                        let num_heads_u32 = num_heads as u32;
                        let num_kv_heads_u32 = num_kv_heads as u32;
                        let head_dim_u32 = head_dim as u32;
                        let kv_dim_u32 = kv_dim as u32;
                        let seq_len_u32 = new_seq_len as u32;
                        let max_seq_len_u32 = s.max_seq_len as u32;
                        const FLASH_DECODE_TILE_SIZE_DEFAULT: u32 = 256;
                        const FLASH_DECODE_THRESHOLD: usize =
                            FLASH_DECODE_TILE_SIZE_DEFAULT as usize + 1; // 257: single-tile is a no-op reduce
                                                                         // Diagnostic/perf lever (VALIDATED NEGATIVE, default-OFF): force the
                                                                         // online-softmax flash path at ALL KV lengths (replaces the
                                                                         // device-scratch MHA below 257). Byte-identical to the MHA baseline.
                                                                         // Measured +6-7% SLOWER on full_attn at short KV (single-tile flash
                                                                         // adds the reduce dispatch + partial round-trip; MHA's single dispatch
                                                                         // is optimal for KV<257). Kept gated-off as an A/B substrate. Tile is
                                                                         // fixed at the default 256 (partial buffer is sized for 256).
                        let flash_always = super::flash_decode_always_enabled();

                        if new_seq_len >= FLASH_DECODE_THRESHOLD || flash_always {
                            let flash_tile_size = FLASH_DECODE_TILE_SIZE_DEFAULT;
                            let num_tiles =
                                ((new_seq_len as u32) + flash_tile_size - 1) / flash_tile_size;
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
                            enc.set_bytes(&flash_tile_size.to_le_bytes(), 10);
                            enc.set_bytes(&num_tiles.to_le_bytes(), 11);
                            enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 12);
                            enc.dispatch_threadgroups(
                                MTLSize::new((num_heads as u64) * (num_tiles as u64), 1, 1),
                                MTLSize::new(128, 1, 1),
                            );
                            // Barrier: flash_decode writes partial_buf, reduce reads partial_buf
                            if needs_barriers {
                                enc.memory_barrier_with_scope(1);
                            }
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
                            let tg_threads =
                                mha_tg_size.min((head_dim.max(new_seq_len) as u64).max(1));
                            enc.dispatch_threadgroups(
                                MTLSize::new(num_heads as u64, 1, 1),
                                MTLSize::new(tg_threads, 1, 1),
                            );
                        }
                    }
                } // end fallback (non-fused RoPE+KV+MHA)

                // Barrier: attention writes attn_out_buf, Wo reads attn_out_buf
                if needs_barriers {
                    enc.memory_barrier_with_scope(1);
                }

                // Wo projection + Residual
                let has_attn_extras = meta.attn_post_norm_off.is_some()
                    || meta.attn_gate_off.is_some()
                    || meta.has_qgate_fusion;
                if has_attn_extras {
                    // Apply sigmoid(gate) * attn_out BEFORE Wo (Q+gate fusion).
                    if meta.has_qgate_fusion {
                        let pso = pipelines.sigmoid_mul_fused.as_ref().ok_or_else(|| {
                            RuntimeError::Compute("sigmoid_mul_fused pipeline not compiled".into())
                        })?;
                        enc.set_pipeline_state(pso);
                        enc.set_buffer(&s.gate_buf, 0, 0); // gate [q_dim]
                        enc.set_buffer(&s.attn_out_buf, 0, 1); // attn output [q_dim]
                        enc.set_buffer(&s.attn_out_buf, 0, 2); // output (in-place)
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                        let tg = 256u64.min(q_dim as u64).max(1);
                        enc.dispatch_threadgroups(
                            MTLSize::new((q_dim as u64).div_ceil(tg), 1, 1),
                            MTLSize::new(tg, 1, 1),
                        );
                    }

                    // Barrier: sigmoid_mul writes attn_out_buf, Wo reads it
                    if needs_barriers {
                        enc.memory_barrier_with_scope(1);
                    }
                    // Non-fused Wo: attn_proj_buf = Wo * attn_out (NO residual)
                    if fa_skip & FULLATTN_SKIP_WO == 0 {
                        // MLX-style qmv fast path for the Q4_0 Wo projection (env
                        // LUMEN_METAL_Q4_QMV_PROJ=1). NON-residual branch (residual added
                        // downstream by residual_add_copy) -> feed qmv a ZERO residual
                        // buffer: Wo*x + 0 == Wo*x exactly. Indexed by layer_idx. Mirrors
                        // decode_greedy.rs exactly.
                        let qmv_wo = if meta.wo_quant == QuantScheme::Q4_0 {
                            match (
                                s.qmv_attn_wo_qw.get(layer_idx),
                                s.qmv_attn_wo_scales.get(layer_idx),
                                s.qmv_zero_residual_buf.as_ref(),
                            ) {
                                (Some(Some(qw)), Some(Some(sc)), Some(zero)) => {
                                    Some((qw, sc, zero))
                                }
                                _ => None,
                            }
                        } else {
                            None
                        };
                        if let Some((qw, sc, zero)) = qmv_wo {
                            // qmv_q4_0_residual: w@0, x@1, out@2, in_dim@3, scales@4,
                            // residual@5. in = q_dim (%512); out = hidden_dim (%8).
                            // F16-scales full-attn Wo (LUMEN_METAL_Q4_FULLATTN_F16SC=1).
                            if let Some(p) = crate::metal::q4_fullattn_f16sc_enabled()
                                .then_some(pipelines.qmv_q4_0_residual_f16sc.as_ref())
                                .flatten()
                            {
                                enc.set_pipeline_state(p);
                            } else {
                                enc.set_pipeline_state(&pipelines.qmv_q4_0_residual);
                            }
                            enc.set_buffer(qw, 0, 0);
                            enc.set_buffer(&s.attn_out_buf, 0, 1);
                            enc.set_buffer(&s.attn_proj_buf, 0, 2);
                            enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                            enc.set_buffer(sc, 0, 4);
                            enc.set_buffer(zero, 0, 5);
                            enc.dispatch_threadgroups(
                                MTLSize::new((hidden_dim as u64) / 8, 1, 1),
                                MTLSize::new(64, 1, 1),
                            );
                        } else {
                            let tg_wo = match meta.wo_quant {
                                QuantScheme::Q8_0 => {
                                    enc.set_pipeline_state(
                                        &pipelines.dequant_matmul_q8_0_deferred_nr2,
                                    );
                                    128u64
                                }
                                QuantScheme::Q4_0 => {
                                    enc.set_pipeline_state(
                                        &pipelines.dequant_matmul_q4_0_deferred_nr2,
                                    );
                                    128u64
                                }
                                QuantScheme::F16 => {
                                    enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                                    128u64
                                }
                                QuantScheme::Bf16 => {
                                    enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_nr2);
                                    128u64
                                }
                                _ => {
                                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                                    matmul_tg_size
                                }
                            };
                            enc.set_buffer(layer_buf, wo_off, 0);
                            enc.set_buffer(&s.attn_out_buf, 0, 1);
                            enc.set_buffer(&s.attn_proj_buf, 0, 2);
                            enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                            if matches!(
                                meta.wo_quant,
                                QuantScheme::Q8_0
                                    | QuantScheme::Q4_0
                                    | QuantScheme::F16
                                    | QuantScheme::Bf16
                            ) {
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                            }
                            let n_tg_wo = match meta.wo_quant {
                                QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2,
                                QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2,
                                QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2,
                                QuantScheme::Bf16 => ((hidden_dim as u64) + 1) / 2,
                                _ => hidden_dim as u64,
                            };
                            enc.dispatch_threadgroups(
                                MTLSize::new(n_tg_wo, 1, 1),
                                MTLSize::new(tg_wo, 1, 1),
                            );
                        }
                    }
                    // Barrier: Wo writes attn_proj_buf
                    if needs_barriers {
                        enc.memory_barrier_with_scope(1);
                    }
                    // Post-attention RMSNorm: only for architectures that have
                    // BOTH attn_post_norm AND attn_gate (not Q+gate fusion).
                    // For Qwen3.5 Q+gate fusion, post_attention_norm is the
                    // pre-FFN norm (via ffn_norm_off) — must not be applied here.
                    let did_post_norm =
                        meta.attn_gate_off.is_some() && meta.attn_post_norm_off.is_some();
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
                        if needs_barriers {
                            enc.memory_barrier_with_scope(1);
                        }
                    }
                    // Attention output gate
                    if let Some(gate_off) = meta.attn_gate_off {
                        let gate_quant = meta.attn_gate_quant.unwrap_or(QuantScheme::F32);
                        let src_buf = if did_post_norm {
                            &s.down_buf
                        } else {
                            &s.attn_proj_buf
                        };
                        let attn_gate_buf = s.attn_gate_buf.as_ref().ok_or_else(|| {
                            RuntimeError::Compute("attn_gate_buf not allocated".into())
                        })?;
                        // Gate matmul
                        {
                            let tg_gate = match gate_quant {
                                QuantScheme::Q8_0 => {
                                    enc.set_pipeline_state(
                                        &pipelines.dequant_matmul_q8_0_deferred_nr2,
                                    );
                                    128u64
                                }
                                QuantScheme::Q4_0 => {
                                    enc.set_pipeline_state(
                                        &pipelines.dequant_matmul_q4_0_deferred_nr2,
                                    );
                                    128u64
                                }
                                QuantScheme::F16 => {
                                    enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                                    128u64
                                }
                                QuantScheme::Bf16 => {
                                    enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_nr2);
                                    128u64
                                }
                                _ => {
                                    enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                                    matmul_tg_size
                                }
                            };
                            enc.set_buffer(layer_buf, gate_off, 0);
                            enc.set_buffer(src_buf, 0, 1);
                            enc.set_buffer(attn_gate_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            if matches!(
                                gate_quant,
                                QuantScheme::Q8_0
                                    | QuantScheme::Q4_0
                                    | QuantScheme::F16
                                    | QuantScheme::Bf16
                            ) {
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                            }
                            let n_tg_gate = match gate_quant {
                                QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2,
                                QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2,
                                QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2,
                                QuantScheme::Bf16 => ((hidden_dim as u64) + 1) / 2,
                                _ => hidden_dim as u64,
                            };
                            enc.dispatch_threadgroups(
                                MTLSize::new(n_tg_gate, 1, 1),
                                MTLSize::new(tg_gate, 1, 1),
                            );
                        }
                        // Barrier: gate matmul writes attn_gate_buf, SwiGLU reads it
                        if needs_barriers {
                            enc.memory_barrier_with_scope(1);
                        }
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
                        if needs_barriers {
                            enc.memory_barrier_with_scope(1);
                        }
                        // Fused residual + copy (saves 1 dispatch + 1 barrier)
                        // x_buf += attn_gate_buf; attn_proj_buf = x_buf
                        {
                            let pso = pipelines
                                .residual_add_copy
                                .as_ref()
                                .unwrap_or(&pipelines.add_residual);
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
                                if needs_barriers {
                                    enc.memory_barrier_with_scope(1);
                                }
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
                            let pso = pipelines
                                .residual_add_copy
                                .as_ref()
                                .unwrap_or(&pipelines.add_residual);
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
                                if needs_barriers {
                                    enc.memory_barrier_with_scope(1);
                                }
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
                    // MLX-style qmv fast path for the Q4_0 Wo projection (env
                    // LUMEN_METAL_Q4_QMV_PROJ=1). Residual-fused branch -> feed qmv the
                    // real residual (x_buf): Wo*x + x_buf. Indexed by layer_idx. Mirrors
                    // decode_greedy.rs exactly.
                    let qmv_wo = if meta.wo_quant == QuantScheme::Q4_0 {
                        match (
                            s.qmv_attn_wo_qw.get(layer_idx),
                            s.qmv_attn_wo_scales.get(layer_idx),
                        ) {
                            (Some(Some(qw)), Some(Some(sc))) => Some((qw, sc)),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if let Some((qw, sc)) = qmv_wo {
                        // qmv_q4_0_residual: w@0, x@1, out@2, in_dim@3, scales@4,
                        // residual@5. in = q_dim (%512); out = hidden_dim (%8).
                        // F16-scales full-attn Wo (LUMEN_METAL_Q4_FULLATTN_F16SC=1), residual=x_buf.
                        if let Some(p) = crate::metal::q4_fullattn_f16sc_enabled()
                            .then_some(pipelines.qmv_q4_0_residual_f16sc.as_ref())
                            .flatten()
                        {
                            enc.set_pipeline_state(p);
                        } else {
                            enc.set_pipeline_state(&pipelines.qmv_q4_0_residual);
                        }
                        enc.set_buffer(qw, 0, 0);
                        enc.set_buffer(&s.attn_out_buf, 0, 1);
                        enc.set_buffer(&s.attn_proj_buf, 0, 2);
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                        enc.set_buffer(sc, 0, 4);
                        enc.set_buffer(&s.x_buf, 0, 5);
                        enc.dispatch_threadgroups(
                            MTLSize::new((hidden_dim as u64) / 8, 1, 1),
                            MTLSize::new(64, 1, 1),
                        );
                    } else {
                        let tg_wo = match meta.wo_quant {
                            QuantScheme::Q8_0 => {
                                enc.set_pipeline_state(
                                    &pipelines.dequant_matmul_q8_0_deferred_residual_nr2,
                                );
                                128u64
                            }
                            QuantScheme::Q4_0 => {
                                enc.set_pipeline_state(
                                    &pipelines.dequant_matmul_q4_0_deferred_residual_nr2,
                                );
                                128u64
                            }
                            QuantScheme::F16 => {
                                enc.set_pipeline_state(&pipelines.matmul_f16_deferred_residual_nr2);
                                128u64
                            }
                            QuantScheme::Bf16 => {
                                enc.set_pipeline_state(
                                    &pipelines.matmul_bf16_deferred_residual_nr2,
                                );
                                128u64
                            }
                            _ => {
                                enc.set_pipeline_state(&pipelines.matmul_bytes_f32_residual);
                                matmul_tg_size
                            }
                        };
                        enc.set_buffer(layer_buf, wo_off, 0);
                        enc.set_buffer(&s.attn_out_buf, 0, 1);
                        enc.set_buffer(&s.attn_proj_buf, 0, 2);
                        enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                        enc.set_buffer(&s.x_buf, 0, 4);
                        if matches!(
                            meta.wo_quant,
                            QuantScheme::Q8_0
                                | QuantScheme::Q4_0
                                | QuantScheme::F16
                                | QuantScheme::Bf16
                        ) {
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
                        }
                        let n_tg_wo = match meta.wo_quant {
                            QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2,
                            QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2,
                            QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2,
                            QuantScheme::Bf16 => ((hidden_dim as u64) + 1) / 2,
                            _ => hidden_dim as u64,
                        };
                        enc.dispatch_threadgroups(
                            MTLSize::new(n_tg_wo, 1, 1),
                            MTLSize::new(tg_wo, 1, 1),
                        );
                    }
                }
            } else {
                // GatedDeltaNet layer: linear attention forward pass
                let gdn_idx = meta.gdn_layer_idx.unwrap();
                // Fused variant: all GDN dispatches go through the layer encoder.
                // The fused fn takes OWNERSHIP of `enc` and returns the encoder
                // active at layer end (same encoder normally; a fresh serial one
                // when CONCURRENT_PROJ split the projection cluster). The
                // recurrence always runs on a serial encoder inside the fn.
                let gdn_concurrent = crate::metal::metal_concurrent_proj_enabled();
                // GDN_F16_STATE_DECODE: one-time F32->F16 convert of this layer's
                // h_state on first touch (encoded on the live `enc` before the
                // recurrence reads it). No-op when the flag is OFF.
                Self::ensure_gdn_f16_state_decode(&self.device, pipelines, &enc, s, gdn_idx)?;
                let (new_conv_pos, ret_enc) = Self::encode_gdn_layer_decode_fused(
                    enc,
                    &cmd,
                    gdn_concurrent,
                    pipelines,
                    s,
                    layer_buf,
                    meta,
                    gdn_idx,
                )?;
                enc = ret_enc;
                s.gdn_conv_positions[gdn_idx] = new_conv_pos;
            }

            // [decode-profile] Section boundary between attention and FFN.
            if decode_profile::is_enabled() {
                enc.end_encoding();
                cmd.commit_and_wait();
                let g = cmd.gpu_elapsed_secs();
                let ffn_lbl = if meta.moe_meta.is_some() {
                    "moe_ffn"
                } else {
                    "dense_ffn"
                };
                decode_profile::record_gpu(g, ffn_lbl);
                decode_profile::record_and_advance(ffn_lbl);
                cmd = self.queue.new_command_buffer().ok_or_else(|| {
                    RuntimeError::Compute("decode-profile: failed to create CB".into())
                })?;
                enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("decode-profile: failed to create encoder".into())
                })?;
            }

            // ================================================================
            // FFN BLOCK
            // ================================================================

            // Fused RMSNorm + FFN for dense Q8_0/Q4_0/F16 gate+up.
            // Eliminates FFN RMSNorm dispatch + 1 barrier + normed_buf write/read.
            // BF16 has no fused FFN kernel; it goes through the non-fused 3-dispatch path.
            let use_fused_ffn_norm = meta.moe_meta.is_none()
                && matches!(
                    meta.w_gate_quant,
                    QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16
                )
                && meta.w_gate_quant == meta.w_up_quant;

            if !use_fused_ffn_norm {
                // Non-fused path: separate FFN RMSNorm + barrier
                // Barrier: Wo+residual writes attn_proj_buf, FFN RMSNorm reads it
                if needs_barriers {
                    enc.memory_barrier_with_scope(1);
                }
                // FFN RMSNorm
                enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
                enc.set_buffer(&s.attn_proj_buf, 0, 0);
                enc.set_buffer(layer_buf, ffn_norm_off, 1);
                enc.set_buffer(&s.normed_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&eps.to_le_bytes(), 4);
                enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(norm_tg_size, 1, 1));

                // Barrier: FFN RMSNorm writes normed_buf, gate+up reads normed_buf
                if needs_barriers {
                    enc.memory_barrier_with_scope(1);
                }
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
                    && s.moe_gate_up_offsets
                        .get(layer_idx)
                        .and_then(|o| o.as_ref())
                        .is_some()
                    && s.moe_down_offsets
                        .get(layer_idx)
                        .and_then(|o| o.as_ref())
                        .is_some()
                    && s.moe_batched_swiglu_buf.is_some();

                if use_batched {
                    let per_layer_ids_buf = s
                        .moe_per_layer_expert_ids
                        .get(layer_idx)
                        .and_then(|opt| opt.as_ref());
                    let expert_ids_buf =
                        per_layer_ids_buf.unwrap_or_else(|| s.moe_expert_ids.as_ref().unwrap());
                    let expert_weights_buf = s.moe_expert_weights.as_ref().unwrap();
                    let gate_up_off_buf = s.moe_gate_up_offsets[layer_idx].as_ref().unwrap();
                    let down_off_buf = s.moe_down_offsets[layer_idx].as_ref().unwrap();

                    // Router dispatch (all within same encoder)
                    // LUMEN_METAL_MOE_ROUTER_PARALLEL=1 -> two-kernel
                    // parallel router (per-expert logits across the grid + tiny
                    // top-k softmax) instead of the single-threadgroup serial
                    // router. Profiling shows the serial router is ~94% of MoE-FFN
                    // GPU time. Default OFF -> original byte path.
                    // Fused single-dispatch router (grid=experts,
                    // last-TG reduction) eliminates the separate 1-TG top-k drain
                    // bubble (~6 ms/token = 39% of decode). Env-gated, default OFF.
                    let use_router_fused = super::moe_router_fused_enabled()
                        && pipelines.moe_router_fused_topk.is_some()
                        && s.moe_router_logits.is_some()
                        && s.moe_router_counter.is_some();
                    let use_router_parallel = !use_router_fused
                        && super::moe_router_parallel_enabled()
                        && pipelines.moe_router_logits_f32.is_some()
                        && pipelines.moe_router_topk_softmax.is_some()
                        && s.moe_router_logits.is_some();
                    if super::moe_diag_skip() == 5 {
                        // diagnostic: skip router entirely
                    } else if use_router_fused {
                        let logits_buf = s.moe_router_logits.as_ref().unwrap();
                        let counter_buf = s.moe_router_counter.as_ref().unwrap();
                        let pso = pipelines.moe_router_fused_topk.as_ref().unwrap();
                        enc.set_pipeline_state(pso);
                        enc.set_buffer(&s.normed_buf, 0, 0);
                        enc.set_buffer(layer_buf, moe_meta.router_weight_off, 1);
                        enc.set_buffer(logits_buf, 0, 2);
                        enc.set_buffer(expert_ids_buf, 0, 3);
                        enc.set_buffer(expert_weights_buf, 0, 4);
                        enc.set_buffer(counter_buf, 0, 5);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 6);
                        enc.set_bytes(&(s.moe_num_experts as u32).to_le_bytes(), 7);
                        enc.set_bytes(&(s.moe_num_active_experts as u32).to_le_bytes(), 8);
                        let tg = 256u64.min(hidden_dim as u64).max(1);
                        enc.dispatch_threadgroups(
                            MTLSize::new(s.moe_num_experts as u64, 1, 1),
                            MTLSize::new(tg, 1, 1),
                        );
                    } else if use_router_parallel {
                        let logits_buf = s.moe_router_logits.as_ref().unwrap();
                        let diag = super::moe_diag_skip();
                        // Kernel 1: per-expert logits, one threadgroup per expert.
                        if diag != 7 {
                            let pso = pipelines.moe_router_logits_f32.as_ref().unwrap();
                            enc.set_pipeline_state(pso);
                            enc.set_buffer(&s.normed_buf, 0, 0);
                            enc.set_buffer(layer_buf, moe_meta.router_weight_off, 1);
                            enc.set_buffer(logits_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(s.moe_num_experts as u32).to_le_bytes(), 4);
                            let tg = 256u64.min(hidden_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new(s.moe_num_experts as u64, 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        }
                        // Kernel 1 writes all logits; Kernel 2 reads them. Fence
                        // is required (serial encoder does not auto-order RAW).
                        // (Removing it was measured null/-1.2% and adds risk.)
                        enc.memory_barrier_with_scope(1);
                        // Kernel 2: top-k softmax over the precomputed logits.
                        if diag != 6 {
                            let pso = pipelines.moe_router_topk_softmax.as_ref().unwrap();
                            enc.set_pipeline_state(pso);
                            enc.set_buffer(logits_buf, 0, 0);
                            enc.set_buffer(expert_ids_buf, 0, 1);
                            enc.set_buffer(expert_weights_buf, 0, 2);
                            enc.set_bytes(&(s.moe_num_experts as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(s.moe_num_active_experts as u32).to_le_bytes(), 4);
                            // Parallel softmax: dispatch min(256, num_experts) threads.
                            let topk_tg = 256u64.min(s.moe_num_experts as u64).max(32);
                            // Replicate across N TGs to keep the GPU occupied
                            // during the top-k (avoids the 1-TG drain bubble). Only TG 0
                            // writes; redundant TGs are byte-identical no-ops.
                            let topk_n_tg = super::moe_router_topk_tgs() as u64;
                            enc.dispatch_threadgroups(
                                MTLSize::new(topk_n_tg, 1, 1),
                                MTLSize::new(topk_tg, 1, 1),
                            );
                        }
                    } else {
                        let router_softmax =
                            pipelines.moe_router_softmax.as_ref().ok_or_else(|| {
                                RuntimeError::Compute(
                                    "MoE router_softmax pipeline not compiled.".into(),
                                )
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
                    if needs_barriers {
                        enc.memory_barrier_with_scope(1);
                    }
                    // Batched expert FFN + shared expert (fused when available)
                    Self::encode_moe_ffn_with_shared_fused(
                        &enc,
                        pipelines,
                        s,
                        layer_buf,
                        layer_idx,
                        moe_meta,
                        meta,
                        expert_ids_buf,
                        expert_weights_buf,
                        gate_up_off_buf,
                        down_off_buf,
                    )?;
                } else {
                    // Legacy path: per-expert dispatch (needs its own encoders).
                    // End current encoder, run legacy, then this is the last thing
                    // in the layer so we skip re-opening.
                    enc.end_encoding();
                    let per_layer_ids_buf = s
                        .moe_per_layer_expert_ids
                        .get(layer_idx)
                        .and_then(|opt| opt.as_ref());
                    let per_layer_wts_buf = s
                        .moe_per_layer_expert_weights
                        .get(layer_idx)
                        .and_then(|opt| opt.as_ref());
                    Self::encode_moe_ffn_decode(
                        &cmd,
                        pipelines,
                        s,
                        layer_buf,
                        moe_meta,
                        per_layer_ids_buf,
                        None,
                        None,
                        0.0,
                        None,
                        false,
                        per_layer_wts_buf,
                    )?;
                    // Shared expert dispatch (legacy path)
                    if meta.shared_expert_gate_off.is_some() {
                        Self::encode_shared_expert_ffn_decode(&cmd, pipelines, s, layer_buf, meta)?;
                    }
                    // Re-create encoder for remaining layers (serial for GDN, concurrent for MoE-only)
                    enc = if has_gdn {
                        cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to re-create serial encoder".into())
                        })?
                    } else {
                        cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to re-create concurrent encoder".into())
                        })?
                    };
                    continue; // encoder already ended, skip end_encoding below
                }
            } else {
                // Dense FFN path
                if use_fused_ffn_norm {
                    // Fused RMSNorm + FFN gate+up+SwiGLU.
                    // Reads attn_proj_buf directly, applies inline normalization.
                    // Barrier: Wo+residual writes attn_proj_buf, fused FFN reads it
                    if needs_barriers {
                        enc.memory_barrier_with_scope(1);
                    }
                    if matches!(meta.w_gate_quant, QuantScheme::F16) {
                        // F16: always use deferred 1-row-per-TG pattern (no block structure)
                        enc.set_pipeline_state(
                            &pipelines.rmsnorm_ffn_fused_gate_up_swiglu_f16_deferred,
                        );
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
                        // MLX-style DUAL-matrix qmv fast path for the Q4_0 dense
                        // FFN gate/up when decode-qmv buffers exist for this layer
                        // (env LUMEN_METAL_Q4_QMV_GATEUP=1); else the 8row fused
                        // path. Vec indexed by layer_idx. WILDCARD: 8row is already
                        // 256-thread/78% peak, so qmv (64-thread) MAY regress; the
                        // orchestrator A/Bs it empirically. Mirrors decode_greedy.
                        let qmv_gate_up = if meta.w_gate_quant == QuantScheme::Q4_0 {
                            match (
                                s.qmv_ffn_gate_qw.get(layer_idx),
                                s.qmv_ffn_gate_scales.get(layer_idx),
                                s.qmv_ffn_up_qw.get(layer_idx),
                                s.qmv_ffn_up_scales.get(layer_idx),
                            ) {
                                (
                                    Some(Some(gqw)),
                                    Some(Some(gsc)),
                                    Some(Some(uqw)),
                                    Some(Some(usc)),
                                ) => Some((gqw, gsc, uqw, usc)),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        let gateup_splitk = crate::metal::q4_gateup_splitk();
                        let splitk_ok = gateup_splitk >= 2
                            && qmv_gate_up.is_some()
                            && hidden_dim % (512 * gateup_splitk as usize) == 0
                            && inter_dim % 8 == 0;
                        if splitk_ok {
                            // Two-pass deterministic SPLIT-K for gate/up (env
                            // LUMEN_METAL_Q4_GATEUP_SPLITK=N). Reuses the gate/up
                            // qmv decode buffers. (1) RMSNorm x -> splitk_normed_buf;
                            // (2) qmv_q4_0_splitk_partial x2 (gate, up) over N K-slices
                            // -> gate/up partials; (3) gateup_splitk_reduce_swiglu
                            // (fixed-order reduce + SwiGLU) -> gate_buf. Raises TG
                            // concurrency Nx for the K=4096 gate/up matvec.
                            let (gqw, gsc, uqw, usc) = qmv_gate_up.unwrap();
                            let ks = gateup_splitk;
                            let up_part_off: u64 = (inter_dim as u64) * 8 * 4; // bytes
                                                                               // (1) RMSNorm pre-pass: normed = x * rsqrt(mean(x^2)+eps) * norm_w
                            enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
                            enc.set_buffer(&s.attn_proj_buf, 0, 0);
                            enc.set_buffer(layer_buf, ffn_norm_off, 1);
                            enc.set_buffer(&s.splitk_normed_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&eps.to_le_bytes(), 4);
                            enc.dispatch_threadgroups(
                                MTLSize::new(1, 1, 1),
                                MTLSize::new(s.matmul_tg_size, 1, 1),
                            );
                            // (2) gate partials over N K-slices
                            enc.set_pipeline_state(&pipelines.qmv_q4_0_splitk_partial);
                            enc.set_buffer(gqw, 0, 0);
                            enc.set_buffer(&s.splitk_normed_buf, 0, 1);
                            enc.set_buffer(&s.splitk_gateup_partials_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_buffer(gsc, 0, 4);
                            enc.set_bytes(&ks.to_le_bytes(), 5);
                            enc.dispatch_threadgroups(
                                MTLSize::new((inter_dim as u64) / 8, ks as u64, 1),
                                MTLSize::new(64, 1, 1),
                            );
                            // (2b) up partials over N K-slices (offset into the buffer)
                            enc.set_pipeline_state(&pipelines.qmv_q4_0_splitk_partial);
                            enc.set_buffer(uqw, 0, 0);
                            enc.set_buffer(&s.splitk_normed_buf, 0, 1);
                            enc.set_buffer(&s.splitk_gateup_partials_buf, up_part_off, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_buffer(usc, 0, 4);
                            enc.set_bytes(&ks.to_le_bytes(), 5);
                            enc.dispatch_threadgroups(
                                MTLSize::new((inter_dim as u64) / 8, ks as u64, 1),
                                MTLSize::new(64, 1, 1),
                            );
                            // (3) reduce gate+up partials + SwiGLU -> gate_buf
                            enc.set_pipeline_state(&pipelines.gateup_splitk_reduce_swiglu);
                            enc.set_buffer(&s.splitk_gateup_partials_buf, 0, 0);
                            enc.set_buffer(&s.splitk_gateup_partials_buf, up_part_off, 1);
                            enc.set_buffer(&s.gate_buf, 0, 2);
                            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&ks.to_le_bytes(), 4);
                            enc.dispatch_threadgroups(
                                MTLSize::new(((inter_dim as u64) + 255) / 256, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        } else if crate::metal::metal_concurrent_gateup_enabled()
                            && qmv_gate_up.is_some()
                        {
                            // DIE-SATURATION LEVER for the dense FFN gate/up
                            // (LUMEN_METAL_CONCURRENT_GATEUP=1, default OFF): the
                            // fused 8row kernel computes gate+up in ONE dispatch, so
                            // Metal cannot spread it across the M3 Ultra's two
                            // UltraFusion dies. Un-fuse into RMSNorm-once + two BARE
                            // single-matrix qmv matvecs (gate -> gate_buf, up ->
                            // up_buf) dispatched on a CONCURRENT encoder + a
                            // standalone SwiGLU. gate and up read the SAME read-only
                            // normed x and write DISJOINT buffers -> independent ->
                            // concurrent dispatch is BYTE-IDENTICAL to serial (each
                            // matvec's internal accumulation order is unchanged; only
                            // the inter-matvec ordering relaxes). Mirrors the proven
                            // CONCURRENT_PROJ mechanism, extended to the largest decode
                            // GPU pool. HONEST: gate/up is ~51 SG/core (past the
                            // occupancy knee), so the die-spread may be marginal vs the
                            // die-starved projection clusters where it won — measured.
                            let (gqw, gsc, uqw, usc) = qmv_gate_up.unwrap();
                            let use256 = crate::metal::metal_concurrent_gateup_256_enabled();
                            let (gu_pso, gu_threads): (&_, u64) = if use256 {
                                (&pipelines.qmv_q4_0_8sg, 256)
                            } else {
                                (&pipelines.qmv_q4_0, 64)
                            };
                            // (a) RMSNorm x ONCE on the SERIAL encoder: attn_proj_buf
                            // -> normed_buf (rmsnorm_bytes reads the FFN norm weight as
                            // raw bytes off layer_buf at ffn_norm_off; 1 TG).
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
                            // Close the serial encoder and open a CONCURRENT one for
                            // the gate/up cluster. Encoders within ONE command buffer
                            // execute in submission order, so the RMSNorm above is
                            // ordered-before this cluster -> normed_buf is fully
                            // written when gate/up read it.
                            enc.end_encoding();
                            enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                                RuntimeError::Compute(
                                    "CONCURRENT_GATEUP: failed to create concurrent encoder".into(),
                                )
                            })?;
                            // (b) bare qmv GATE: normed_buf -> gate_buf. NO barrier
                            // between gate and up (disjoint outputs) so Metal can run
                            // them concurrently across both dies.
                            enc.set_pipeline_state(gu_pso);
                            enc.set_buffer(gqw, 0, 0);
                            enc.set_buffer(&s.normed_buf, 0, 1);
                            enc.set_buffer(&s.gate_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_buffer(gsc, 0, 4);
                            enc.dispatch_threadgroups(
                                MTLSize::new((inter_dim as u64) / 8, 1, 1),
                                MTLSize::new(gu_threads, 1, 1),
                            );
                            // (c) bare qmv UP: normed_buf -> up_buf.
                            enc.set_pipeline_state(gu_pso);
                            enc.set_buffer(uqw, 0, 0);
                            enc.set_buffer(&s.normed_buf, 0, 1);
                            enc.set_buffer(&s.up_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_buffer(usc, 0, 4);
                            enc.dispatch_threadgroups(
                                MTLSize::new((inter_dim as u64) / 8, 1, 1),
                                MTLSize::new(gu_threads, 1, 1),
                            );
                            // Resource-scoped barrier on the two disjoint outputs, then
                            // close the concurrent encoder and reopen a serial one so
                            // the SwiGLU (and the rest of the layer) runs with serial
                            // completion ordering exactly as before.
                            enc.memory_barrier_with_resources(&[&s.gate_buf, &s.up_buf]);
                            enc.end_encoding();
                            enc = cmd.new_compute_encoder().ok_or_else(|| {
                                RuntimeError::Compute(
                                    "CONCURRENT_GATEUP: failed to reopen serial encoder".into(),
                                )
                            })?;
                            // (d) standalone SwiGLU: gate_buf = silu(gate_buf) * up_buf
                            enc.set_pipeline_state(&pipelines.swiglu);
                            enc.set_buffer(&s.gate_buf, 0, 0);
                            enc.set_buffer(&s.up_buf, 0, 1);
                            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 2);
                            let swg_tg = 256u64.min(inter_dim as u64).max(1);
                            enc.dispatch_threadgroups(
                                MTLSize::new((inter_dim as u64).div_ceil(swg_tg), 1, 1),
                                MTLSize::new(swg_tg, 1, 1),
                            );
                        } else if let Some((gqw, gsc, uqw, usc)) = qmv_gate_up {
                            // INTERLEAVED gate+up (env LUMEN_METAL_Q4_GATEUP_IL):
                            // highest priority when the IL pipeline compiled AND the
                            // interleaved buffers exist for this layer. Byte-identical
                            // SwiGLU written to gate_buf, exactly like the dual-matrix
                            // path below, but reading ONE co-resident packed nibble +
                            // ONE packed f16-scale buffer. Indexed by layer_idx.
                            let il_bufs = if super::q4_gateup_il_enabled() {
                                match (
                                    pipelines.qmv_q4_0_gate_up_swiglu_il.as_ref(),
                                    s.qmv_ffn_gate_up_il_qw.get(layer_idx),
                                    s.qmv_ffn_gate_up_il_scales.get(layer_idx),
                                ) {
                                    (Some(pso), Some(Some(qw)), Some(Some(sc))) => {
                                        Some((pso, qw, sc))
                                    }
                                    _ => None,
                                }
                            } else {
                                None
                            };
                            if let Some((pso_il, il_qw, il_sc)) = il_bufs {
                                // qmv_q4_0_gate_up_swiglu_il: w_il@0, x@1, out@2,
                                // in_dim@3, scales_il@4, norm_w@5, eps@6.
                                enc.set_pipeline_state(pso_il);
                                enc.set_buffer(il_qw, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                enc.set_buffer(&s.gate_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                enc.set_buffer(il_sc, 0, 4);
                                enc.set_buffer(layer_buf, ffn_norm_off, 5);
                                enc.set_bytes(&eps.to_le_bytes(), 6);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((inter_dim as u64) / 8, 1, 1),
                                    MTLSize::new(64, 1, 1),
                                );
                            } else {
                                // qmv_q4_0_gate_up_swiglu: w_gate@0, x@1, out@2, in_dim@3,
                                // gate_scales@4, w_up@5, up_scales@6, norm_w@7, eps@8.
                                // out_dim = inter_dim (%8==0); in_dim = hidden (%512==0).
                                //
                                // SCALE-TYPE-MATCHING dispatch (mirrors the greedy path in
                                // decode_greedy.rs): when the gate/up decode-qmv scale
                                // buffers were built as f16 (which LUMEN_METAL_Q4_GATEUP_H2MATH=1
                                // self-engages), the f32-scale qmv_q4_0_gate_up_swiglu reads
                                // those f16 bytes as f32 and garbles. Dispatch the f16sc (or
                                // h2math) variant that reads `device const half*` scales.
                                // Bindings + geometry are identical across all three (only the
                                // scale element type and the inner accumulator differ), so the
                                // dispatched kernel MUST match the buffer-build layout.
                                // (sampling-correctness fix.)
                                let gateup_f16sc_on = super::q4_gateup_f16sc_enabled();
                                let gu_pipe = if gateup_f16sc_on
                                    && super::q4_gateup_h2math_enabled()
                                    && pipelines.qmv_q4_0_gate_up_swiglu_f16sc_h2math.is_some()
                                {
                                    pipelines.qmv_q4_0_gate_up_swiglu_f16sc_h2math.as_ref()
                                } else if gateup_f16sc_on
                                    && pipelines.qmv_q4_0_gate_up_swiglu_f16sc.is_some()
                                {
                                    pipelines.qmv_q4_0_gate_up_swiglu_f16sc.as_ref()
                                } else {
                                    None
                                };
                                if let Some(p) = gu_pipe {
                                    enc.set_pipeline_state(p);
                                } else {
                                    enc.set_pipeline_state(&pipelines.qmv_q4_0_gate_up_swiglu);
                                }
                                enc.set_buffer(gqw, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                enc.set_buffer(&s.gate_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                enc.set_buffer(gsc, 0, 4);
                                enc.set_buffer(uqw, 0, 5);
                                enc.set_buffer(usc, 0, 6);
                                enc.set_buffer(layer_buf, ffn_norm_off, 7);
                                enc.set_bytes(&eps.to_le_bytes(), 8);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((inter_dim as u64) / 8, 1, 1),
                                    MTLSize::new(64, 1, 1),
                                );
                            }
                        } else {
                            match meta.w_gate_quant {
                                QuantScheme::Q4_0 => enc.set_pipeline_state(
                                    &pipelines.rmsnorm_ffn_fused_gate_up_swiglu_q4_0_8row,
                                ),
                                _ => enc.set_pipeline_state(
                                    &pipelines.rmsnorm_ffn_fused_gate_up_swiglu_q8_0_8row,
                                ),
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
                        }
                    } else {
                        match meta.w_gate_quant {
                            QuantScheme::Q4_0 => enc.set_pipeline_state(
                                &pipelines.rmsnorm_ffn_fused_gate_up_swiglu_q4_0_deferred,
                            ),
                            _ => enc.set_pipeline_state(
                                &pipelines.rmsnorm_ffn_fused_gate_up_swiglu_q8_0_deferred,
                            ),
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
                } else if meta.w_gate_quant == QuantScheme::Q8_0
                    && meta.w_up_quant == QuantScheme::Q8_0
                {
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
                } else if matches!(meta.w_gate_quant, QuantScheme::Bf16) {
                    // BF16 unfused: gate matvec + up matvec + SwiGLU (no fused BF16 kernel)
                    // Mirrors the F32 fallback path but uses matmul_bf16_deferred_nr2.
                    // Gate: gate_buf = W_gate * normed_buf
                    enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_nr2);
                    enc.set_buffer(layer_buf, w_gate_off, 0);
                    enc.set_buffer(&s.normed_buf, 0, 1);
                    enc.set_buffer(&s.gate_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                    enc.dispatch_threadgroups(
                        MTLSize::new(((inter_dim as u64) + 1) / 2, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                    // Up: up_buf = W_up * normed_buf
                    enc.set_buffer(layer_buf, w_up_off, 0);
                    enc.set_buffer(&s.up_buf, 0, 2);
                    enc.dispatch_threadgroups(
                        MTLSize::new(((inter_dim as u64) + 1) / 2, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                    // Barrier: gate+up matmuls write gate_buf+up_buf, SwiGLU reads both
                    if needs_barriers {
                        enc.memory_barrier_with_scope(1);
                    }
                    // SwiGLU: gate_buf = silu(gate_buf) * up_buf
                    enc.set_pipeline_state(&pipelines.swiglu);
                    enc.set_buffer(&s.gate_buf, 0, 0);
                    enc.set_buffer(&s.up_buf, 0, 1);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 2);
                    let tg = 256u64.min(inter_dim as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((inter_dim as u64).div_ceil(tg), 1, 1),
                        MTLSize::new(tg, 1, 1),
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
                    if needs_barriers {
                        enc.memory_barrier_with_scope(1);
                    }
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
                if needs_barriers {
                    enc.memory_barrier_with_scope(1);
                }
                // Down projection + Residual 2 (fused)
                {
                    // MLX-style qmv fast path when decode-qmv buffers exist for this
                    // Q4_0 layer (env LUMEN_METAL_Q4_QMV_DOWN=1); else NR2.
                    let qmv_down = if meta.w_down_quant == QuantScheme::Q4_0 {
                        match (
                            s.qmv_down_qw.get(layer_idx),
                            s.qmv_down_scales.get(layer_idx),
                        ) {
                            (Some(Some(qw)), Some(Some(sc))) => Some((qw, sc)),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if let Some((qw, sc)) = qmv_down {
                        // Two-pass deterministic SPLIT-K (env LUMEN_METAL_Q4_QMV_DOWN_SPLITK=N,
                        // default 0=off) when in_dim splits evenly into N 512-block slices;
                        // else the one-pass qmv_q4_0_residual.
                        let k_splits = crate::metal::q4_qmv_down_splitk();
                        if k_splits >= 2 && (inter_dim as u32) % (512 * k_splits) == 0 {
                            // Pass 1: per-(row-group, k-slice) partial dot -> partials scratch.
                            enc.set_pipeline_state(&pipelines.qmv_q4_0_splitk_partial);
                            enc.set_buffer(qw, 0, 0);
                            enc.set_buffer(&s.gate_buf, 0, 1);
                            enc.set_buffer(&s.splitk_partials_buf, 0, 2);
                            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
                            enc.set_buffer(sc, 0, 4);
                            enc.set_bytes(&k_splits.to_le_bytes(), 5);
                            enc.dispatch_threadgroups(
                                MTLSize::new((hidden_dim as u64) / 8, k_splits as u64, 1),
                                MTLSize::new(64, 1, 1),
                            );
                            enc.memory_barrier_with_scope(1);
                            // Pass 2: deterministic reduce (fixed ks-ascending) + residual.
                            enc.set_pipeline_state(&pipelines.qmv_q4_0_splitk_reduce);
                            enc.set_buffer(&s.splitk_partials_buf, 0, 0);
                            enc.set_buffer(&s.x_buf, 0, 1);
                            enc.set_buffer(&s.attn_proj_buf, 0, 2);
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            enc.set_bytes(&k_splits.to_le_bytes(), 4);
                            enc.dispatch_threadgroups(
                                MTLSize::new((hidden_dim as u64).div_ceil(256), 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        } else {
                            // SCALE-TYPE-MATCHING dispatch (mirrors the greedy path in
                            // decode_greedy.rs): when the FFN-down decode-qmv scale
                            // buffer was built as f16 (env LUMEN_METAL_Q4_QMV_DOWN_F16SC=1,
                            // which LUMEN_METAL_Q4_DOWN_H2MATH=1 self-engages), the f32-
                            // scale qmv_q4_0_residual reads those f16 bytes as f32 and
                            // garbles. Dispatch the f16sc (or h2math) kernel that reads
                            // `device const half*` scales instead. Bindings + geometry are
                            // identical across all three (only the scale element type and
                            // the inner accumulator differ), so the buffer-build layout and
                            // the dispatched kernel MUST match. (sampling-correctness fix.)
                            let down_f16sc_on = crate::metal::q4_qmv_down_f16sc_enabled();
                            let down_pipe = if down_f16sc_on
                                && crate::metal::q4_down_h2math_enabled()
                                && pipelines.qmv_q4_0_residual_f16sc_h2math.is_some()
                            {
                                pipelines.qmv_q4_0_residual_f16sc_h2math.as_ref()
                            } else if down_f16sc_on && pipelines.qmv_q4_0_residual_f16sc.is_some() {
                                pipelines.qmv_q4_0_residual_f16sc.as_ref()
                            } else {
                                None
                            };
                            if let Some(p) = down_pipe {
                                enc.set_pipeline_state(p);
                            } else {
                                enc.set_pipeline_state(&pipelines.qmv_q4_0_residual);
                            }
                            enc.set_buffer(qw, 0, 0);
                            enc.set_buffer(&s.gate_buf, 0, 1);
                            enc.set_buffer(&s.x_buf, 0, 2);
                            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
                            enc.set_buffer(sc, 0, 4);
                            enc.set_buffer(&s.attn_proj_buf, 0, 5);
                            enc.dispatch_threadgroups(
                                MTLSize::new((hidden_dim as u64) / 8, 1, 1),
                                MTLSize::new(64, 1, 1),
                            );
                        }
                    } else {
                        let tg_down = match meta.w_down_quant {
                            QuantScheme::Q8_0 => {
                                enc.set_pipeline_state(
                                    &pipelines.dequant_matmul_q8_0_deferred_residual_nr2,
                                );
                                128u64
                            }
                            QuantScheme::Q4_0 => {
                                enc.set_pipeline_state(
                                    &pipelines.dequant_matmul_q4_0_deferred_residual_nr2,
                                );
                                128u64
                            }
                            QuantScheme::F16 => {
                                enc.set_pipeline_state(&pipelines.matmul_f16_deferred_residual_nr2);
                                128u64
                            }
                            QuantScheme::Bf16 => {
                                enc.set_pipeline_state(
                                    &pipelines.matmul_bf16_deferred_residual_nr2,
                                );
                                128u64
                            }
                            _ => {
                                enc.set_pipeline_state(&pipelines.matmul_bytes_f32_residual);
                                matmul_tg_size
                            }
                        };
                        enc.set_buffer(layer_buf, w_down_off, 0);
                        enc.set_buffer(&s.gate_buf, 0, 1);
                        enc.set_buffer(&s.x_buf, 0, 2);
                        enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
                        enc.set_buffer(&s.attn_proj_buf, 0, 4);
                        if matches!(
                            meta.w_down_quant,
                            QuantScheme::Q8_0
                                | QuantScheme::Q4_0
                                | QuantScheme::F16
                                | QuantScheme::Bf16
                        ) {
                            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
                        }
                        let n_tg_down = match meta.w_down_quant {
                            QuantScheme::Q8_0 => ((hidden_dim as u64) + 1) / 2,
                            QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2,
                            QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2,
                            QuantScheme::Bf16 => ((hidden_dim as u64) + 1) / 2,
                            _ => hidden_dim as u64,
                        };
                        enc.dispatch_threadgroups(
                            MTLSize::new(n_tg_down, 1, 1),
                            MTLSize::new(tg_down, 1, 1),
                        );
                    }
                }
            } // end MoE vs dense FFN branch

            // Barrier: down+residual writes x_buf, next layer's RMSNorm reads x_buf
            if needs_barriers {
                enc.memory_barrier_with_scope(1);
            }
        } // end layer loop

        // [decode-profile] Boundary before final norm + lm_head.
        if decode_profile::is_enabled() {
            enc.end_encoding();
            cmd.commit_and_wait();
            let g = cmd.gpu_elapsed_secs();
            decode_profile::record_gpu(g, "lm_head");
            decode_profile::record_and_advance("lm_head");
            cmd = self.queue.new_command_buffer().ok_or_else(|| {
                RuntimeError::Compute("decode-profile: failed to create CB".into())
            })?;
            enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("decode-profile: failed to create encoder".into())
            })?;
        }

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

        // [Option B] Pick the logits buffer THIS token writes to. In async mode
        // we ping-pong: write to the buffer OPPOSITE the in-flight (just-waited)
        // token so its un-sampled logits are never clobbered. `this_logits_b` is
        // recorded into scratch just before the async commit so the NEXT call
        // knows which buffer to read. Sync mode always uses `logits_buf`.
        let this_logits_b: bool = if async_commit {
            !s.async_inflight_logits_b
        } else {
            false
        };
        let logits_target: &MetalBuffer = if this_logits_b {
            s.logits_buf_b
                .as_ref()
                .expect("logits_buf_b allocated in async mode")
        } else {
            &s.logits_buf
        };

        // --- Final RMSNorm + Logits ---
        // Fuse final RMSNorm into output projection for Q8_0/Q4_0.
        // Eliminates 1 dispatch + 1 barrier + normed_buf write/read.
        if matches!(
            output_proj_quant,
            QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16 | QuantScheme::Bf16
        ) {
            // MLX-style Q4 lm_head fast path (env LUMEN_METAL_Q4_QMV_LMHEAD=1):
            // when the re-quantized Q4 output_proj decode-qmv buffers exist, run the
            // fused-RMSNorm `qmv_q4_0_rmsnorm` instead of the Q8 lm_head. Writes the
            // SAME logits buffer + layout ([vocab] f32) that the Q8 path writes, so
            // the downstream argmax/sampling is unaffected; only the projection
            // precision (4-bit) and speed change. None => existing Q8 path.
            if let (Some(qw), Some(sc)) = (s.qmv_lmhead_qw.as_ref(), s.qmv_lmhead_scales.as_ref()) {
                // qmv_q4_0_rmsnorm: w@0, x@1, out@2, in_dim@3, scales@4, norm_w@5, eps@6.
                // out_dim = vocab (%8==0), in_dim = hidden (%512==0). norm_w = final_norm.
                //
                // SCALE-TYPE-MATCHING dispatch (mirrors the greedy path in
                // decode_greedy.rs): when the lm_head decode-qmv scale buffer was built
                // as f16 (which LUMEN_METAL_Q4_LMHEAD_H2MATH=1 self-engages via
                // q4_lmhead_f16sc_enabled), the f32-scale qmv_q4_0_rmsnorm reads those
                // f16 bytes as f32 and garbles. Dispatch the f16sc (or h2math) variant
                // that reads `device const half*` scales. Bindings + geometry are
                // identical across all three, so the dispatched kernel MUST match the
                // buffer-build layout. (sampling-correctness fix.)
                let lmhead_f16sc_pipe = if super::q4_lmhead_f16sc_enabled() {
                    pipelines.qmv_q4_0_rmsnorm_f16sc.as_ref()
                } else {
                    None
                };
                let lmhead_h2math_pipe =
                    if lmhead_f16sc_pipe.is_some() && super::q4_lmhead_h2math_enabled() {
                        pipelines.qmv_q4_0_rmsnorm_f16sc_h2math.as_ref()
                    } else {
                        None
                    };
                if let Some(p) = lmhead_h2math_pipe {
                    enc.set_pipeline_state(p);
                } else if let Some(p) = lmhead_f16sc_pipe {
                    enc.set_pipeline_state(p);
                } else {
                    enc.set_pipeline_state(&pipelines.qmv_q4_0_rmsnorm);
                }
                enc.set_buffer(qw, 0, 0);
                enc.set_buffer(&s.x_buf, 0, 1);
                enc.set_buffer(logits_target, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_buffer(sc, 0, 4);
                enc.set_buffer(sc_norm_buf, sc_norm_off, 5);
                enc.set_bytes(&eps.to_le_bytes(), 6);
                enc.dispatch_threadgroups(
                    MTLSize::new((vocab_size as u64) / 8, 1, 1),
                    MTLSize::new(64, 1, 1),
                );
            } else {
                match output_proj_quant {
                    QuantScheme::Q4_0 => {
                        enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q4_0_deferred_nr2)
                    }
                    QuantScheme::F16 => {
                        enc.set_pipeline_state(&pipelines.rmsnorm_matmul_f16_deferred_nr2)
                    }
                    QuantScheme::Bf16 => {
                        enc.set_pipeline_state(&pipelines.rmsnorm_matmul_bf16_deferred_nr2)
                    }
                    _ => {
                        enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2)
                    }
                }
                enc.set_buffer(sc_proj_buf, sc_proj_off, 0);
                enc.set_buffer(&s.x_buf, 0, 1);
                enc.set_buffer(logits_target, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 4);
                enc.set_buffer(sc_norm_buf, sc_norm_off, 5);
                enc.set_bytes(&eps.to_le_bytes(), 6);
                let n_tg = ((vocab_size as u64) + 1) / 2;
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
            }
        } else {
            // Non-fused path for non-Q8_0 output projections
            // Final RMSNorm
            enc.set_pipeline_state(&pipelines.rmsnorm);
            enc.set_buffer(&s.x_buf, 0, 0);
            enc.set_buffer(sc_norm_buf, sc_norm_off, 1);
            enc.set_buffer(&s.normed_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&eps.to_le_bytes(), 4);
            enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(norm_tg_size, 1, 1));
            // Barrier: Final RMSNorm writes normed_buf, logits projection reads normed_buf
            if needs_barriers {
                enc.memory_barrier_with_scope(1);
            }
            // Logits projection
            let (proj_tg, proj_rows_per_tg) = match output_proj_quant {
                QuantScheme::Q4_0 => {
                    enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2);
                    (128u64, 2u64)
                }
                QuantScheme::F16 => {
                    enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                    (128u64, 2u64)
                }
                QuantScheme::Bf16 => {
                    enc.set_pipeline_state(&pipelines.matmul_bf16_deferred_nr2);
                    (128u64, 2u64)
                }
                _ => {
                    enc.set_pipeline_state(&pipelines.matmul_f32_deferred);
                    (128u64, 4u64)
                }
            };
            enc.set_buffer(sc_proj_buf, sc_proj_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(logits_target, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 4);
            {
                let n_tg = ((vocab_size as u64) + proj_rows_per_tg - 1) / proj_rows_per_tg;
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(proj_tg, 1, 1));
            }
        }

        enc.end_encoding();

        // [Option B] Async-commit exit.
        //
        // CENTRAL FINDING (proven by construction): CPU-sampled decode is a STRICT
        // serial chain  CB(N) -> logits(N) -> sample(N) -> embed(N+1) -> CB(N+1).
        // Token N+1's command buffer cannot even be ENCODED until token N+1's id
        // exists, and that id is produced by the CPU sampler from logits(N), which
        // require CB(N) complete. So a deferred "return the previous token's
        // logits" pipeline is impossible without speculation: the engine contract
        // (`logits = decode_token(tok); next = sample(logits)`) demands THIS
        // token's logits back, and there is no independent work to overlap the
        // wait with. The only legitimate deferral is therefore:
        //   commit() [async]  ->  wait_until_completed()  ->  read own logits.
        // This isolates the ONE potentially-recoverable slice the STEP-1 breakdown
        // found -- whether the split commit()+wait() pair carries a smaller
        // launch/completion tail than the fused commit_and_wait(). The ping-pong
        // buffer + `last_async_cmd` plumbing is retained so the next call's
        // top-of-function drain (line ~91) absorbs any residual completion latency
        // instead of this call blocking on it.
        if async_commit {
            s.async_inflight_logits_b = this_logits_b;
            // Discard the unused prior read (kept the read path symmetric/tested).
            let _ = &prev_logits;
            cmd.commit();
            cmd.wait_until_completed();
            s.gpu_x_valid = false;
            s.last_async_cmd = None;
            kv.advance_seq_len()?;
            self.maybe_trigger_warmup();
            let mut data = vec![0.0f32; vocab_size];
            if this_logits_b {
                if let Some(ref b) = s.logits_buf_b {
                    b.read_f32(&mut data);
                }
            } else {
                s.logits_buf.read_f32(&mut data);
            }
            drop(scratch_guard);
            return Ok(Logits { data });
        }

        // [decode-split] STEP-1: mark end-of-encode (CPU) right before the
        // single sync point, then time the blocked commit_and_wait separately.
        let split_t1 = split_t0.map(|_| std::time::Instant::now());
        // Single sync point for the entire token.
        cmd.commit_and_wait();
        if let (Some(t0), Some(t1)) = (split_t0, split_t1) {
            let t2 = std::time::Instant::now();
            let encode_secs = t1.duration_since(t0).as_secs_f64();
            let wait_secs = t2.duration_since(t1).as_secs_f64();
            decode_profile::record_encode_split(encode_secs, wait_secs, cmd.gpu_elapsed_secs());
        }
        // [decode-profile] Record the final lm_head section + periodic report.
        if decode_profile::is_enabled() {
            decode_profile::record_gpu_final(cmd.gpu_elapsed_secs());
            decode_profile::record_final();
            decode_profile::maybe_report_and_reset(64);
        }
        // [decode-gpu-time] Overhead-free true GPU busy time for this token's
        // single CB (env LUMEN_METAL_DECODE_GPUTIME=1). Decisive for whether
        // decode is GPU-execution-bound vs CPU-encoding/scheduling-bound: if
        // GPU busy << wall time, the bottleneck is NOT kernel compute.
        decode_profile::record_gpu_time(cmd.gpu_elapsed_secs());
        // Optional operator-tunable inter-step pause (LUMEN_METAL_DECODE_DELAY_US).
        // No-op when the delay resolves to 0 (the default).
        super::maybe_apply_metal_decode_delay();

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
                    let has_ids = s
                        .moe_per_layer_expert_ids
                        .get(layer)
                        .and_then(|opt| opt.as_ref());
                    let has_wts = s
                        .moe_per_layer_expert_weights
                        .get(layer)
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
        // DIAGNOSTIC (env LUMEN_METAL_READBACK_PROBE=1, default OFF): time the
        // CPU-side logits read-back (vocab*4 bytes from a shared MTLBuffer) and
        // report the mean every 64 tokens. This is the SECOND serial component
        // (after the GPU CB completes, before the CPU sampler) that any
        // deferred-async-commit overlap cannot hide — it depends on this token's
        // completed logits. No effect on output (read-only timing).
        let mut logits_data = vec![0.0f32; vocab_size];
        {
            use std::cell::Cell;
            use std::sync::OnceLock;
            static RB_PROBE: OnceLock<bool> = OnceLock::new();
            let on = *RB_PROBE
                .get_or_init(|| std::env::var("LUMEN_METAL_READBACK_PROBE").as_deref() == Ok("1"));
            if on {
                thread_local! {
                    static RB_ACC: Cell<(f64, u64)> = const { Cell::new((0.0, 0)) };
                }
                let t = std::time::Instant::now();
                s.logits_buf.read_f32(&mut logits_data);
                let dt = t.elapsed().as_secs_f64();
                RB_ACC.with(|a| {
                    let (mut sum, mut n) = a.get();
                    sum += dt;
                    n += 1;
                    if n >= 64 {
                        eprintln!(
                            "[readback-probe] over {n} tokens: logits_readback={:.3} ms/tok",
                            sum / n as f64 * 1000.0
                        );
                        sum = 0.0;
                        n = 0;
                    }
                    a.set((sum, n));
                });
            } else {
                s.logits_buf.read_f32(&mut logits_data);
            }
        }

        // [XCHK2] Diagnostic-only (env LUMEN_XCHK2=1, default OFF => byte-identical).
        // Dumps this CPU-sampling path's per-step logits sumsq/absmax + top8 in the
        // SAME format as decode_greedy's [XCHK] probe, so the greedy (GPU-argmax)
        // and single_cb (CPU-argmax) decode trajectories can be diffed step-for-step
        // to locate the FIRST divergent (step, logits) — isolating a forward-pass
        // divergence from an argmax/tie-break difference. Read-only; no state change.
        if {
            use std::sync::OnceLock;
            static XK2: OnceLock<bool> = OnceLock::new();
            *XK2.get_or_init(|| std::env::var("LUMEN_XCHK2").as_deref() == Ok("1"))
        } {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static XCHK2_STEP: AtomicUsize = AtomicUsize::new(0);
            let step = XCHK2_STEP.fetch_add(1, Ordering::Relaxed);
            let sumsq_absmax = |v: &[f32]| -> (f64, f32) {
                let mut sq = 0f64;
                let mut mx = 0f32;
                for &e in v {
                    sq += (e as f64) * (e as f64);
                    let a = e.abs();
                    if a > mx {
                        mx = a;
                    }
                }
                (sq, mx)
            };
            // Per-layer GDN h_state + conv_state + MoE expert ids (same format/order
            // as decode_greedy's [XCHK] so the two trajectories diff line-for-line).
            for layer_idx in 0..num_layers {
                let meta = &s.cached_layer_meta[layer_idx];
                if let Some(gdn_idx) = meta.gdn_layer_idx {
                    if gdn_idx < s.gdn_h_states.len() {
                        let hb = &s.gdn_h_states[gdn_idx];
                        let mut h = vec![0f32; (hb.length() / 4) as usize];
                        hb.read_f32(&mut h);
                        let (hsq, hmx) = sumsq_absmax(&h);
                        eprintln!(
                            "[XCHK2] step={step} L={layer_idx} gdn_h_state sumsq={hsq:.6} absmax={hmx:.6}"
                        );
                    }
                }
                if meta.moe_meta.is_some() {
                    if let Some(Some(ids_buf)) = s.moe_per_layer_expert_ids.get(layer_idx) {
                        let mut ids = vec![0u32; s.moe_num_active_experts.max(1)];
                        ids_buf.read_u32(&mut ids);
                        eprintln!("[XCHK2] step={step} L={layer_idx} moe_expert_ids={ids:?}");
                    }
                }
            }
            let (lsq, lmx) = sumsq_absmax(&logits_data);
            let mut idx: Vec<usize> = (0..vocab_size).collect();
            idx.sort_by(|&a, &b| {
                logits_data[b]
                    .partial_cmp(&logits_data[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top: Vec<(usize, f32)> = idx.iter().take(8).map(|&i| (i, logits_data[i])).collect();
            eprintln!("[XCHK2] step={step} logits sumsq={lsq:.6} absmax={lmx:.6} top8={top:?}");
        }

        drop(scratch_guard);

        Ok(Logits { data: logits_data })
    }

    /// [Option B] Flush the final in-flight async-commit CB and return its
    /// logits. Called by the engine's async driver once after the decode loop
    /// ends so the LAST token's logits (still in flight when the loop exited) are
    /// waited + read. Idempotent: returns empty logits if nothing is in flight.
    pub fn decode_flush_async(&self) -> Result<Logits, RuntimeError> {
        let mut guard = self.scratch.lock().unwrap();
        let s = guard
            .as_mut()
            .ok_or_else(|| RuntimeError::Compute("Metal scratch not initialized".into()))?;
        if let Some(cmd) = s.last_async_cmd.take() {
            cmd.wait_until_completed();
            let vocab = s.vocab_size;
            let mut data = vec![0.0f32; vocab];
            if s.async_inflight_logits_b {
                if let Some(ref b) = s.logits_buf_b {
                    b.read_f32(&mut data);
                }
            } else {
                s.logits_buf.read_f32(&mut data);
            }
            Ok(Logits { data })
        } else {
            Ok(Logits { data: Vec::new() })
        }
    }
}
