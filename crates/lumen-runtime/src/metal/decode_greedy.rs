//! Greedy decode path for Metal backend.
//!
//! Extracted from mod.rs for modularity.
//! Contains `decode_token_greedy` which encodes embed + ALL layers + final
//! projection + argmax into a single Metal command buffer, then reads back
//! only 4 bytes (u32 token ID) instead of 128 KB of logits.

use super::{MetalF32Backend, RouterLayerStats};
use crate::error::RuntimeError;
use crate::metal::decode_profile;
use crate::metal::ffi::{MTLSize, MetalBuffer, MetalCommandBuffer};
use lumen_format::quantization::QuantScheme;

/// Per-call pipeline wiring for the greedy decode encode. `None` selects the
/// default SEQUENTIAL behaviour (embed token id via `set_bytes`, argmax writes
/// `argmax_result_buf`, the method commits-and-waits and returns the token).
/// `Some(..)` selects the PIPELINED behaviour (embed reads a token-ring slot,
/// argmax writes a token-ring slot, the method signals an ordering event then
/// commits ASYNC and returns the still-in-flight command buffer for the caller
/// to drain with a one-token lag).
/// Per-CB sampler wiring for the Option-A GPU-sampled lean path. `Some(..)` on
/// `PipeWiring::sampler` swaps the final selection kernel from `argmax` to the
/// parity-matched `gpu_sampler`: it reads this CB's RNG state from
/// `rng_read_slot`, applies penalties (from the per-call penalty buffers staged
/// by the driver) + temperature + softmax + inverse-CDF draw, writes the chosen
/// token into the token ring (`argmax_write_slot`, shared with the greedy
/// chain), and writes the once-advanced RNG state into `rng_write_slot`. All
/// parity-sensitive math is f32 and the final reductions are single-thread
/// sequential (see the `gpu_sampler` kernel) so the emitted token + post-token
/// RNG state match the CPU `sample_logits` for the same seed + history.
struct SamplerWiring {
    /// RNG-state ring slot this CB reads its xorshift64 state from.
    rng_read_slot: usize,
    /// RNG-state ring slot this CB writes the once-advanced state into.
    rng_write_slot: usize,
    /// 1.0 / temperature (temperature > 0 enforced by the caller's route gate).
    inv_temp: f32,
    /// 1 if any penalty (rep != 1, presence != 0, freq != 0) is active -> the
    /// kernel runs its penalty phase over the GPU freq array and appends the
    /// chosen token to it. 0 -> pure temperature sampling (no freq-array touch),
    /// byte-identical to the CPU sampler with all penalties off.
    pen_active: u32,
    /// repetition / presence / frequency penalty values (match SamplingParams;
    /// `apply_penalty_one` semantics).
    rep: f32,
    presence: f32,
    freq: f32,
}

struct PipeWiring<'a> {
    /// Absolute sequence position this CB writes KV / applies RoPE for. The
    /// driver owns this (a pipeline-internal counter that may run AHEAD of the
    /// CPU `kv.seq_len()` because of the in-flight speculative CB), so the core
    /// must NOT read `kv.seq_len()` nor call `kv.advance_seq_len()` in the
    /// pipelined path -- the driver advances the CPU KV counter by exactly one
    /// per RETURNED token, keeping the emitted-token / stop boundary identical
    /// to sequential.
    seq_pos: usize,
    /// Token-ring slot the embed kernel reads its token id from.
    embed_read_slot: usize,
    /// Token-ring slot the final selection kernel (argmax OR gpu_sampler) writes
    /// the chosen next token into.
    argmax_write_slot: usize,
    /// `None` -> greedy argmax finalizer (default, byte-identical). `Some(..)` ->
    /// the Option-A GPU temperature sampler finalizer (LUMEN_METAL_GPU_SAMPLER).
    sampler: Option<SamplerWiring>,
    /// Ordering event: CB signals `signal_value` at its end; the next CB waits
    /// for `wait_value` at its start (0 = no wait, used for the first CB).
    ///
    /// `None` (LEAN driver): skip event encoding entirely. The backend submits
    /// every decode CB on ONE `MetalCommandQueue`, which Metal documents as
    /// FIFO -- CB(k) fully retires (KV-cache, GDN h_state, token-ring writes all
    /// visible) before CB(k+1) begins executing -- so the event is redundant for
    /// correctness on a single queue. Dropping it removes a per-token Obj-C
    /// event retain/release plus two GPU event ops (`encode_wait_for_event` +
    /// `encode_signal_event`) that an event-ordered driver would pay.
    event: Option<&'a crate::metal::ffi::MetalSharedEvent>,
    signal_value: u64,
    wait_value: u64,
}

impl MetalF32Backend {
    /// Public SEQUENTIAL greedy decode: encode one token, commit-and-wait, read
    /// back the 4-byte argmax, advance KV, return the token. Byte-identical to
    /// the long-standing single-CB path. Thin wrapper over the shared encode
    /// core with no pipeline wiring.
    pub fn decode_token_greedy(
        &self,
        token_id: u32,
        weights: &dyn crate::weight::cache::WeightProvider,
        kv: &mut crate::kv::KvCache,
    ) -> Result<u32, RuntimeError> {
        match self.decode_token_greedy_core(token_id, weights, kv, None)? {
            CoreResult::Token(t) => Ok(t),
            // Sequential mode never returns an in-flight CB.
            CoreResult::InFlight(_) => unreachable!("sequential decode returned an in-flight CB"),
        }
    }
}

/// Result of the shared decode encode core: either the decoded token (sequential
/// commit-and-wait) or a committed-but-unwaited command buffer (pipelined async).
enum CoreResult {
    Token(u32),
    InFlight(MetalCommandBuffer),
}

impl MetalF32Backend {
    /// Shared greedy-decode encode core. Encodes embed + all layers + final
    /// projection + argmax into ONE command buffer.
    ///
    /// `pipe = None`: SEQUENTIAL -- create a fresh CB, drain any prior async CB,
    /// commit-and-wait, read back the argmax token, advance KV, return
    /// `CoreResult::Token`.
    ///
    /// `pipe = Some(w)`: PIPELINED -- the CB is freshly created here; the embed
    /// reads `pipe_token_ring[w.embed_read_slot]`, the argmax writes
    /// `pipe_token_ring[w.argmax_write_slot]`; at the end the CB waits on the
    /// ordering event for `w.wait_value` (encoded BEFORE the first encoder via
    /// the queue), signals `w.signal_value`, then commits ASYNC. KV is advanced
    /// (so the next encode sees seq_pos+1). Returns `CoreResult::InFlight` with
    /// the committed CB; the caller reads the token from the ring with a lag.
    ///
    /// The two paths run the SAME kernels in the SAME order with the SAME
    /// per-CB `seq_pos`; only the token-passing buffer and the commit/wait
    /// timing differ, so the decoded token stream is byte-identical.
    fn decode_token_greedy_core(
        &self,
        token_id: u32,
        _weights: &dyn crate::weight::cache::WeightProvider,
        kv: &mut crate::kv::KvCache,
        pipe: Option<PipeWiring<'_>>,
    ) -> Result<CoreResult, RuntimeError> {
        // Batched MoE path is handled inline in the MoE section below.
        // Only fall back to the old two-CB Option A path if batched kernels are unavailable.
        {
            let has_batched = self
                .pipelines
                .as_ref()
                .map(|p| p.moe_batched_gate_up_swiglu_q4_0.is_some())
                .unwrap_or(false);
            if self.use_option_a && !has_batched {
                // Option A path predates pipelining; it is never selected for the
                // pipelined Qwen3.5 batched-MoE model. Fall back sequentially.
                debug_assert!(pipe.is_none(), "pipelined decode requires batched kernels");
                return Ok(CoreResult::Token(
                    self.decode_token_option_a_gpu_resident(token_id, _weights, kv)?,
                ));
            }
        }

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

        // SEQUENTIAL: position is the live CPU KV length. PIPELINED: the driver
        // supplies the absolute position (its internal counter, which may lead
        // the CPU KV counter by the in-flight speculative CB).
        let seq_pos = match pipe {
            Some(ref w) => w.seq_pos,
            None => kv.seq_len(),
        };

        // Single mutex acquisition for the entire token.
        let mut scratch_guard = self.scratch.lock().unwrap();
        let s = scratch_guard
            .as_mut()
            .ok_or_else(|| RuntimeError::Compute("Metal scratch not initialized".into()))?;
        // SEQUENTIAL path: drain any prior async CB AND any leftover pipelined
        // CBs before we reuse scratch buffers (a sequential decode may follow a
        // pipelined run within the same process). PIPELINED path: the driver
        // owns the in-flight CBs (it never leaves work in `last_async_cmd`), and
        // intra-CB ordering is enforced by the shared event below, so do NOT
        // drain here.
        if pipe.is_none() {
            if let Some(prev_cmd) = s.last_async_cmd.take() {
                prev_cmd.wait_until_completed();
            }
            if !s.pipe_inflight.is_empty() {
                Self::pipe_drain_locked(s);
            }
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
        let mut cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for greedy decode".into())
        })?;

        // PIPELINED: force completion-ordering between consecutive token CBs.
        // Encode the GPU-side wait for the previous CB's end-of-execution signal
        // BEFORE opening any compute encoder (signal/wait operate at CB
        // granularity). This guarantees CB(k) has fully retired -- its KV-cache,
        // GDN h_state, and token-ring writes are visible -- before CB(k+1)
        // begins, so the chained token read and the recurrent state are exactly
        // as in the sequential commit-and-wait path. `wait_value == 0` for the
        // first CB of a run (nothing to wait on).
        if let Some(ref w) = pipe {
            if let Some(event) = w.event {
                if w.wait_value > 0 {
                    cmd.encode_wait_for_event(event, w.wait_value);
                }
            }
        }

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
        //
        // GDN models also use serial encoder for deterministic decode:
        // the GDN h_state recurrence accumulates floating-point values across
        // all tokens. Concurrent dispatch nondeterminism in parallel reductions
        // causes accumulated divergence in h_state, leading to degenerate output.
        // MoE-only models (no GDN) can safely use concurrent since MoE routing
        // is stateless between tokens.
        let all_dense = s
            .cached_layer_meta
            .iter()
            .all(|m| m.gdn_layer_idx.is_none() && m.moe_meta.is_none());
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

        // [decode-profile] Start timing with the embed section in-flight.
        decode_profile::begin("embed");

        {
            // PIPELINED: read the token id from a GPU ring slot (the prior CB's
            // argmax wrote it) via the `_bufid` kernel variant -- IDENTICAL math
            // to the set_bytes path. SEQUENTIAL: bake the token id as a constant.
            if let Some(ref w) = pipe {
                match self.embedding_quant {
                    QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.embed_token_q8_0_bufid),
                    QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.embed_token_q4_0_bufid),
                    QuantScheme::F16 => enc.set_pipeline_state(&pipelines.embed_token_f16_bufid),
                    QuantScheme::Bf16 => enc.set_pipeline_state(&pipelines.embed_token_bf16_bufid),
                    _ => enc.set_pipeline_state(&pipelines.embed_token_bufid),
                }
                enc.set_buffer(sc_embed_buf, sc_embed_off, 0);
                enc.set_buffer(&s.x_buf, 0, 1);
                enc.set_buffer(&s.pipe_token_ring[w.embed_read_slot], 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            } else {
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
            }
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
            // Splits the in-flight CB so the prior section's GPU time is timed.
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
                            enc.set_pipeline_state(pipelines.bf16_rmsnorm_matvec_nr2())
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

                // Full-attention bookend elision. When engaged, the
                // deinterleave/norm/assemble + rope + kv-write bookend collapses to
                // a single `deinterleave_norm_rope_kvwrite` dispatch (six dispatches
                // eliminated per full-attn layer). Only the Qwen3.5 Q+gate-fusion
                // NeoX partial-RoPE shape is byte-identical under the fold; anything
                // else (full RoPE fused path, linear-attn, non-NeoX, missing norms,
                // or any diagnostic fa_skip bit set) keeps the incumbent path.
                let is_linear_attn_be = meta.layer_type == Some(1);
                let use_fused_rope_kv_be = !is_linear_attn_be && s.rotary_dim == head_dim;
                let bookend_elided = fa_skip == 0
                    && meta.has_qgate_fusion
                    && !is_linear_attn_be
                    && s.rope_neox
                    && !use_fused_rope_kv_be
                    && meta.attn_q_norm_off.is_some()
                    && meta.attn_k_norm_off.is_some()
                    && pipelines.deinterleave_norm_rope_kvwrite.is_some();

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

                    // DIE-SATURATION LEVER (LUMEN_METAL_CONCURRENT_PROJ=1, default OFF):
                    // the three SEPARATE Q+gate/K/V projection matvecs all read the SAME
                    // x_buf and write DISJOINT buffers (qkv_buf/k_buf/v_buf) with no shared
                    // state -> they are independent. On the layer's SERIAL encoder they run
                    // one-at-a-time (sum of times); a single memory-bound matvec tops out
                    // ~50-60% of the M3 Ultra's two-die aggregate bandwidth. Dispatching the
                    // cluster on a CONCURRENT encoder lets Metal spread their threadgroups
                    // across both UltraFusion dies (finish in ~max instead of ~sum). Byte-
                    // identical to serial: disjoint outputs, shared read-only input, a
                    // resource-scoped barrier closes the cluster before the DNA consumer
                    // reads. The GDN recurrence is untouched. (Greedy has no qgatekv/kv-fuse
                    // single-dispatch paths, so the only guard is the separate-projection +
                    // fused-norm condition.) Mirrors decode_single_cb's validated wiring.
                    let concurrent_proj_cluster = crate::metal::metal_concurrent_proj_enabled()
                        && use_fused_attn_norm
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

                    // Project Q+gate into qkv_buf
                    if fa_skip & FULLATTN_SKIP_QGATE == 0 {
                        // MLX-style fused RMSNorm+qmv fast path for the Q4_0 full-attn
                        // Q+gate projection when decode-qmv buffers exist (env
                        // LUMEN_METAL_Q4_QMV_PROJ=1) AND the fused-norm path would run
                        // (the qmv kernel fuses RMSNorm, so it is ONLY valid when
                        // use_fused_attn_norm is true -- never when reading normed_buf).
                        // Vec indexed by layer_idx (None for GDN layers).
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
                            // norm_w@5, eps@6. out_dim = qgate_dim (%8==0); in = hidden (%512==0).
                            // F16-scales full-attn (env LUMEN_METAL_Q4_FULLATTN_F16SC=1): when the
                            // scale buffer was built as f16 and the f16sc kernel compiled, dispatch
                            // qmv_q4_0_rmsnorm_f16sc (reads `half*` scales); byte-identical.
                            if let Some(p) = super::q4_fullattn_f16sc_enabled()
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
                                    QuantScheme::Bf16 => {
                                        enc.set_pipeline_state(pipelines.bf16_rmsnorm_matvec_nr2())
                                    }
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
                                        enc.set_pipeline_state(pipelines.bf16_matvec_nr2());
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
                    // Project K from wk (parallel with Q+gate when fused)
                    if fa_skip & FULLATTN_SKIP_K == 0 {
                        let wk_off_val = meta.wk_off.unwrap();
                        let wk_quant = meta.wk_quant.unwrap();
                        // MLX-style fused RMSNorm+qmv fast path for the Q4_0 K projection
                        // when decode-qmv buffers exist (env LUMEN_METAL_Q4_QMV_KV=1).
                        // Reads x_buf (pre-norm hidden) like Q does; writes k_buf at
                        // offset 0 exactly as the NR2 path. Indexed by layer_idx.
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
                            // F16-scales full-attn K (LUMEN_METAL_Q4_FULLATTN_F16SC=1): f16sc
                            // kernel when its scale buffer was built f16 + kernel compiled.
                            if let Some(p) = super::q4_fullattn_f16sc_enabled()
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
                                QuantScheme::Bf16 => {
                                    enc.set_pipeline_state(pipelines.bf16_rmsnorm_matvec_nr2())
                                }
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
                                    enc.set_pipeline_state(pipelines.bf16_matvec_nr2());
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
                    // Project V from wv (parallel with Q+gate and K when fused)
                    if fa_skip & FULLATTN_SKIP_V == 0 {
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
                            // F16-scales full-attn V (LUMEN_METAL_Q4_FULLATTN_F16SC=1): f16sc
                            // kernel when its scale buffer was built f16 + kernel compiled.
                            if let Some(p) = super::q4_fullattn_f16sc_enabled()
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
                                QuantScheme::Bf16 => {
                                    enc.set_pipeline_state(pipelines.bf16_rmsnorm_matvec_nr2())
                                }
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
                                    enc.set_pipeline_state(pipelines.bf16_matvec_nr2());
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
                    } else if bookend_elided {
                        // Bookend elision: one dispatch does deinterleave +
                        // per-head Q/K RMSNorm + NeoX partial rope + direct K/V cache
                        // write. Eliminates copy_buffer x3 + rope_neox x2 +
                        // write_kv_cache (folded below). Q lands in q_buf; SDPA reads
                        // it directly. Byte-identical: all-f32 intermediates, same
                        // arithmetic order, same half() cache cast.
                        let pso = pipelines.deinterleave_norm_rope_kvwrite.as_ref().unwrap();
                        let q_norm_off = meta.attn_q_norm_off.unwrap();
                        let k_norm_off = meta.attn_k_norm_off.unwrap();
                        let rope_half_dim = s.rotary_dim / 2;
                        let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
                        enc.set_pipeline_state(pso);
                        enc.set_buffer(&s.qkv_buf, 0, 0); // qgate_interleaved (Q|gate)
                        enc.set_buffer(&s.k_buf, 0, 1); // k_data (normed in-place)
                        enc.set_buffer(&s.v_buf, 0, 2); // v_data
                        enc.set_buffer(layer_buf, q_norm_off, 3); // q_norm_w
                        enc.set_buffer(layer_buf, k_norm_off, 4); // k_norm_w
                        enc.set_buffer(&s.rope_cos_buf, 0, 5);
                        enc.set_buffer(&s.rope_sin_buf, 0, 6);
                        enc.set_buffer(&s.gate_buf, 0, 7); // gate_out
                        enc.set_buffer(&s.q_buf, 0, 8); // q_out (normed+roped Q)
                        enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 9);
                        enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 10);
                        enc.set_bytes(&(num_heads as u32).to_le_bytes(), 11);
                        enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 12);
                        enc.set_bytes(&(head_dim as u32).to_le_bytes(), 13);
                        enc.set_bytes(&(rope_half_dim as u32).to_le_bytes(), 14);
                        enc.set_bytes(&pos_offset_u32.to_le_bytes(), 15);
                        enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 16);
                        enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 17);
                        enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 18);
                        enc.set_bytes(&eps.to_le_bytes(), 19);
                        let total_tgs = (num_heads + num_kv_heads) as u64;
                        let tg_threads = 256u64.min(head_dim as u64).max(32);
                        enc.dispatch_threadgroups(
                            MTLSize::new(total_tgs, 1, 1),
                            MTLSize::new(tg_threads, 1, 1),
                        );
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
                        enc.set_buffer(&s.qkv_buf, 0, 5); // qkv_out (UNUSED post DET-001 RACE#2 fix; K stays in k_buf, V in v_buf)
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
                        // DET-001 RACE#2 fix: the DNA kernel no longer writes K/V into qkv_buf
                        // (that aliased the qgate read region); K is normalized in-place in
                        // k_buf, V stays in v_buf, so BOTH must be copied into qkv_buf here.
                        // (This mirrors decode_single_cb's determinism-fixed assemble. The
                        // prior greedy code copied ONLY Q and relied on the now-removed kernel
                        // K/V write, leaving qkv_buf[K]/[V] stale -> corrupt attention.)
                        if needs_barriers {
                            enc.memory_barrier_with_scope(1);
                        }
                        // Copy normalized Q from q_buf to qkv_buf[0..q_dim).
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
                                    enc.set_pipeline_state(pipelines.bf16_matvec_bias_nr2())
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
                                    enc.set_pipeline_state(pipelines.bf16_matvec_nr2());
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
                // Only for: standard RoPE (not NeoX), short sequences, full rotary_dim
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
                    // Diagnostic: skip attention compute (FULLATTN_SKIP_ATTN bit3). 0 = full run.
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
                    if bookend_elided {
                        // Bookend elision: RoPE Q/K + KV-cache write already
                        // performed inside the fused deinterleave_norm_rope_kvwrite
                        // dispatch above; nothing to do here.
                    } else if fa_skip & FULLATTN_SKIP_ROPE_KV != 0 {
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
                        // Bookend elision: the fused kernel wrote normed+roped
                        // Q to q_buf (byte-identical to the incumbent roped qkv_buf[Q]),
                        // so attention reads Q from q_buf. Otherwise read qkv_buf[Q].
                        let (q_attn_buf, q_attn_off) = if bookend_elided {
                            (&s.q_buf, 0u64)
                        } else {
                            (&s.qkv_buf, q_byte_off)
                        };
                        let num_heads_u32 = num_heads as u32;
                        let num_kv_heads_u32 = num_kv_heads as u32;
                        let head_dim_u32 = head_dim as u32;
                        let kv_dim_u32 = kv_dim as u32;
                        let seq_len_u32 = new_seq_len as u32;
                        let max_seq_len_u32 = s.max_seq_len as u32;
                        const FLASH_DECODE_TILE_SIZE: u32 = 256;
                        // Flash decode engages only at K >= 513 (two full tiles). For K in
                        // [257, 512] the 2-tile flash path spills the 1..256 keys of the
                        // second tile into a near-empty tile plus a separate reduce dispatch,
                        // which is slower than -- and byte-identical to -- the exact
                        // single-dispatch multi_head_attention over the same keys. So
                        // single-tile MHA serves K <= 512 and flash takes over at K >= 513.
                        let flash_decode_threshold: usize = FLASH_DECODE_TILE_SIZE as usize * 2 + 1; // 513

                        if new_seq_len >= flash_decode_threshold {
                            let num_tiles = ((new_seq_len as u32) + FLASH_DECODE_TILE_SIZE - 1)
                                / FLASH_DECODE_TILE_SIZE;
                            enc.set_pipeline_state(&pipelines.flash_decode_attention);
                            enc.set_buffer(q_attn_buf, q_attn_off, 0);
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
                            enc.set_buffer(q_attn_buf, q_attn_off, 0);
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
                    // Glue-side elision: fold sigmoid_mul + Wo qmv + residual_add_copy
                    // into `qmv_q4_0_wo_glue`. Engages ONLY on the Qwen3.5 full-attn
                    // f32-scale Q4_0 qmv-Wo path with Q+gate fusion and NO separate
                    // attn_gate (so the residual sits immediately after Wo); the
                    // f32-scale path is required (the f16sc Wo path keeps the three
                    // separate dispatches). All other paths keep them too.
                    let glue_fold = meta.has_qgate_fusion
                        && meta.attn_gate_off.is_none()
                        && meta.wo_quant == QuantScheme::Q4_0
                        && !super::q4_fullattn_f16sc_enabled()
                        && pipelines.qmv_q4_0_wo_glue.is_some()
                        && matches!(s.qmv_attn_wo_qw.get(layer_idx), Some(Some(_)))
                        && matches!(s.qmv_attn_wo_scales.get(layer_idx), Some(Some(_)))
                        && (fa_skip & FULLATTN_SKIP_WO == 0);
                    // Apply sigmoid(gate) * attn_out BEFORE Wo (Q+gate fusion).
                    // SKIPPED when glue_fold: the folded Wo kernel applies the gate.
                    if meta.has_qgate_fusion && !glue_fold {
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
                        // MLX-style qmv fast path for the Q4_0 Wo projection when
                        // decode-qmv buffers exist (env LUMEN_METAL_Q4_QMV_PROJ=1).
                        // This branch is mathematically NON-residual (the residual is
                        // added downstream by residual_add_copy), so feed qmv a ZERO
                        // residual buffer: Wo*x + 0 == Wo*x exactly. Indexed by layer_idx.
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
                        if glue_fold {
                            // Glue-side elision: one dispatch folds the
                            // sigmoid gate (on attn_out), the Wo matvec, the residual
                            // add (x_buf), and the dual-write (x_buf + attn_proj).
                            // Bindings: w@0, x=attn_out@1, accum=x_buf@2, in_dim@3,
                            // scales@4, copy_dst=attn_proj@5, gate=gate_buf@6.
                            // gate_fold guaranteed qmv_wo Some + glue pipeline Some.
                            let (qw, sc, _zero) = qmv_wo.expect("glue_fold implies qmv_wo");
                            let glue_pso = pipelines
                                .qmv_q4_0_wo_glue
                                .as_ref()
                                .expect("glue_fold implies qmv_q4_0_wo_glue compiled");
                            enc.set_pipeline_state(glue_pso);
                            enc.set_buffer(qw, 0, 0);
                            enc.set_buffer(&s.attn_out_buf, 0, 1);
                            enc.set_buffer(&s.x_buf, 0, 2); // accum = residual src + write
                            enc.set_bytes(&(q_dim as u32).to_le_bytes(), 3);
                            enc.set_buffer(sc, 0, 4);
                            enc.set_buffer(&s.attn_proj_buf, 0, 5); // copy_dst
                            enc.set_buffer(&s.gate_buf, 0, 6);
                            enc.dispatch_threadgroups(
                                MTLSize::new((hidden_dim as u64) / 8, 1, 1),
                                MTLSize::new(64, 1, 1),
                            );
                        } else if let Some((qw, sc, zero)) = qmv_wo {
                            // qmv_q4_0_residual: w@0, x@1, out@2, in_dim@3, scales@4,
                            // residual@5. in = q_dim (%512); out = hidden_dim (%8).
                            // F16-scales full-attn Wo (LUMEN_METAL_Q4_FULLATTN_F16SC=1): when its
                            // scale buffer was built f16 + the f16sc kernel compiled, dispatch
                            // qmv_q4_0_residual_f16sc (reads `half*` scales); byte-identical.
                            if let Some(p) = super::q4_fullattn_f16sc_enabled()
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
                                    enc.set_pipeline_state(pipelines.bf16_matvec_nr2());
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
                                    enc.set_pipeline_state(pipelines.bf16_matvec_nr2());
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
                        // Glue-side elision: SKIPPED when glue_fold — the
                        // folded qmv_q4_0_wo_glue already did x_buf += Wo_out and
                        // attn_proj = x_buf (dual-write) in the Wo dispatch.
                        if !glue_fold {
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
                    // MLX-style qmv fast path for the Q4_0 Wo projection when decode-qmv
                    // buffers exist (env LUMEN_METAL_Q4_QMV_PROJ=1). This branch IS
                    // residual-fused, so feed qmv the real residual buffer (x_buf):
                    // Wo*x + x_buf, matching the NR2 residual kernel. Indexed by layer_idx.
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
                        // F16-scales full-attn Wo (LUMEN_METAL_Q4_FULLATTN_F16SC=1): same as
                        // the non-fused Wo path but with the real residual (x_buf) at @5.
                        if let Some(p) = super::q4_fullattn_f16sc_enabled()
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
                                enc.set_pipeline_state(pipelines.bf16_matvec_residual_nr2());
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
                // Fused variant: all GDN dispatches go through the layer encoder.
                // The fused fn takes OWNERSHIP of `enc` and returns the encoder
                // active at layer end (same encoder normally; a fresh serial one
                // when CONCURRENT_PROJ split the GDN qkv/alpha-beta/attn_gate
                // projection cluster). The recurrence always runs on a serial
                // encoder inside the fn. CONCURRENT_PROJ (default OFF) splits ONLY
                // the independent projection cluster -> byte-identical.
                let gdn_idx = meta.gdn_layer_idx.unwrap();
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
                )?; // s passed immutably; F16 mirror lives behind RefCell
                enc = ret_enc;
                s.gdn_conv_positions[gdn_idx] = new_conv_pos;
            }

            // [decode-profile] Section boundary between attention and FFN.
            // The section that just finished IS the attention block; advance the
            // in-flight label to the FFN type so the FFN time accrues next.
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

                    // Router dispatch (all within same encoder).
                    // Parallel two-kernel router when enabled.
                    let use_router_parallel = super::moe_router_parallel_enabled()
                        && pipelines.moe_router_logits_f32.is_some()
                        && pipelines.moe_router_topk_softmax.is_some()
                        && s.moe_router_logits.is_some();
                    if use_router_parallel {
                        let logits_buf = s.moe_router_logits.as_ref().unwrap();
                        {
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
                        enc.memory_barrier_with_scope(1);
                        {
                            let pso = pipelines.moe_router_topk_softmax.as_ref().unwrap();
                            enc.set_pipeline_state(pso);
                            enc.set_buffer(logits_buf, 0, 0);
                            enc.set_buffer(expert_ids_buf, 0, 1);
                            enc.set_buffer(expert_weights_buf, 0, 2);
                            enc.set_bytes(&(s.moe_num_experts as u32).to_le_bytes(), 3);
                            enc.set_bytes(&(s.moe_num_active_experts as u32).to_le_bytes(), 4);
                            let topk_tg = 256u64.min(s.moe_num_experts as u64).max(32);
                            enc.dispatch_threadgroups(
                                MTLSize::new(1, 1, 1),
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
                        // orchestrator A/Bs it empirically.
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
                        if let Some((gqw, gsc, uqw, usc)) = qmv_gate_up {
                            // INTERLEAVED gate+up (env LUMEN_METAL_Q4_GATEUP_IL):
                            // highest priority when the IL pipeline compiled AND the
                            // interleaved buffers were built for this layer. Reads ONE
                            // co-resident packed nibble buffer + ONE packed f16-scale
                            // buffer (gate|up woven per 512-value super-iter) and writes
                            // gate_buf = SwiGLU(gate,up) directly, exactly like the
                            // f16sc path. Byte-identical math. Indexed by layer_idx.
                            // LM-head-structure (LS) single-stream gate+up: takes
                            // priority when the LS pipeline compiled AND the LS
                            // row-interleaved buffers were built for this layer.
                            // Reads ONE weight stream per simdgroup row from a
                            // row-interleaved gate|up buffer (row 2d=gate[d],
                            // 2d+1=up[d]) at 2*inter_dim/8 TGs and writes
                            // gate_buf = SwiGLU(gate,up). Byte-identical to the
                            // h2math dual kernel. Indexed by layer_idx.
                            let ls_bufs = match (
                                pipelines.qmv_q4_0_gate_up_swiglu_ls_h2math.as_ref(),
                                s.qmv_ffn_gate_up_ls_qw.get(layer_idx),
                                s.qmv_ffn_gate_up_ls_scales.get(layer_idx),
                            ) {
                                (Some(pso), Some(Some(qw)), Some(Some(sc))) => Some((pso, qw, sc)),
                                _ => None,
                            };
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
                            if let Some((pso_ls, ls_qw, ls_sc)) = ls_bufs {
                                // qmv_q4_0_gate_up_swiglu_ls_h2math: w_ls@0, x@1,
                                // out@2, in_dim@3, ls_scales@4, norm_w@5, eps@6.
                                // Grid = 2*inter_dim/8 TGs, 64 threads.
                                enc.set_pipeline_state(pso_ls);
                                enc.set_buffer(ls_qw, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                enc.set_buffer(&s.gate_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                enc.set_buffer(ls_sc, 0, 4);
                                enc.set_buffer(layer_buf, ffn_norm_off, 5);
                                enc.set_bytes(&eps.to_le_bytes(), 6);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((inter_dim as u64) * 2 / 8, 1, 1),
                                    MTLSize::new(64, 1, 1),
                                );
                            } else if let Some((pso_il, il_qw, il_sc)) = il_bufs {
                                // qmv_q4_0_gate_up_swiglu_il: w_il@0, x@1, out@2,
                                // in_dim@3, scales_il@4, norm_w@5, eps@6.
                                // out_dim = inter_dim (%8==0); in_dim = hidden (%512==0).
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
                            } else if crate::metal::metal_concurrent_gateup_enabled() {
                                // CONCURRENT GATE/UP die-saturation lever (mirrors the
                                // decode_single_cb wiring exactly; see that comment).
                                // RMSNorm once on the serial encoder -> normed_buf;
                                // gate + up bare qmv on a CONCURRENT encoder (disjoint
                                // gate_buf/up_buf, shared read-only normed x ->
                                // byte-identical to serial); resource barrier; serial
                                // SwiGLU. Default OFF.
                                let use256 = crate::metal::metal_concurrent_gateup_256_enabled();
                                let (gu_pso, gu_threads): (&_, u64) = if use256 {
                                    (&pipelines.qmv_q4_0_8sg, 256)
                                } else {
                                    (&pipelines.qmv_q4_0, 64)
                                };
                                // (a) RMSNorm x ONCE (serial): attn_proj_buf -> normed_buf
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
                                // Close serial, open concurrent for the gate/up cluster.
                                enc.end_encoding();
                                enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                                    RuntimeError::Compute(
                                        "CONCURRENT_GATEUP: failed to create concurrent encoder"
                                            .into(),
                                    )
                                })?;
                                // (b) gate: normed_buf -> gate_buf (no inter-barrier)
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
                                // (c) up: normed_buf -> up_buf
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
                                // Resource barrier on disjoint outputs, close concurrent,
                                // reopen serial for the SwiGLU + rest of layer.
                                enc.memory_barrier_with_resources(&[&s.gate_buf, &s.up_buf]);
                                enc.end_encoding();
                                enc = cmd.new_compute_encoder().ok_or_else(|| {
                                    RuntimeError::Compute(
                                        "CONCURRENT_GATEUP: failed to reopen serial encoder".into(),
                                    )
                                })?;
                                // (d) standalone swiglu: gate_buf = silu(gate_buf)*up_buf
                                enc.set_pipeline_state(&pipelines.swiglu);
                                enc.set_buffer(&s.gate_buf, 0, 0);
                                enc.set_buffer(&s.up_buf, 0, 1);
                                enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 2);
                                let swg_tg = 256u64.min(inter_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((inter_dim as u64).div_ceil(swg_tg), 1, 1),
                                    MTLSize::new(swg_tg, 1, 1),
                                );
                            } else if super::q4_gateup_bareqmv_enabled() {
                                // BARE-QMV: RMSNorm the FFN input ONCE
                                // (attn_proj_buf -> normed_buf), then run a BARE
                                // single-matrix qmv on the pre-normed x for gate
                                // and up (no per-matrix RMSNorm recompute), then
                                // a standalone swiglu. The 64-thread qmv_q4_0 or
                                // (with _256) the 256-thread qmv_q4_0_8sg.
                                let use256 = super::q4_gateup_bareqmv_256_enabled();
                                use std::sync::Once;
                                static BAREQMV_ONCE: Once = Once::new();
                                BAREQMV_ONCE.call_once(|| {
                                    eprintln!(
                                        "[lumen] bare-qmv gate/up branch active (256={})",
                                        use256
                                    )
                                });
                                // (a) RMSNorm x ONCE: attn_proj_buf -> normed_buf.
                                // Mirror the proven FFN-norm dispatch (rmsnorm_bytes
                                // reads the FFN norm weight as raw bytes off the
                                // layer buffer at ffn_norm_off; 1 TG of norm_tg_size).
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
                                if needs_barriers {
                                    enc.memory_barrier_with_scope(1);
                                }
                                // (b) bare qmv GATE: normed_buf -> gate_buf
                                enc.set_pipeline_state(if use256 {
                                    &pipelines.qmv_q4_0_8sg
                                } else {
                                    &pipelines.qmv_q4_0
                                });
                                enc.set_buffer(gqw, 0, 0);
                                enc.set_buffer(&s.normed_buf, 0, 1);
                                enc.set_buffer(&s.gate_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                enc.set_buffer(gsc, 0, 4);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((inter_dim as u64) / 8, 1, 1),
                                    MTLSize::new(if use256 { 256 } else { 64 }, 1, 1),
                                );
                                if needs_barriers {
                                    enc.memory_barrier_with_scope(1);
                                }
                                // (c) bare qmv UP: normed_buf -> up_buf
                                enc.set_pipeline_state(if use256 {
                                    &pipelines.qmv_q4_0_8sg
                                } else {
                                    &pipelines.qmv_q4_0
                                });
                                enc.set_buffer(uqw, 0, 0);
                                enc.set_buffer(&s.normed_buf, 0, 1);
                                enc.set_buffer(&s.up_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                enc.set_buffer(usc, 0, 4);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((inter_dim as u64) / 8, 1, 1),
                                    MTLSize::new(if use256 { 256 } else { 64 }, 1, 1),
                                );
                                if needs_barriers {
                                    enc.memory_barrier_with_scope(1);
                                }
                                // (d) standalone swiglu: gate_buf = silu(gate_buf)*up_buf
                                enc.set_pipeline_state(&pipelines.swiglu);
                                enc.set_buffer(&s.gate_buf, 0, 0);
                                enc.set_buffer(&s.up_buf, 0, 1);
                                enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 2);
                                let swg_tg = 256u64.min(inter_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((inter_dim as u64).div_ceil(swg_tg), 1, 1),
                                    MTLSize::new(swg_tg, 1, 1),
                                );
                            } else if super::q4_gateup_unfused_enabled() {
                                // UNFUSED: gate + up as TWO single-matrix
                                // qmv_q4_0_rmsnorm GEMVs (half the register
                                // pressure of the dual-matrix kernel, proven 84%
                                // peak on lm_head) + standalone swiglu.
                                // gate -> gate_buf
                                enc.set_pipeline_state(&pipelines.qmv_q4_0_rmsnorm);
                                enc.set_buffer(gqw, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                enc.set_buffer(&s.gate_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                enc.set_buffer(gsc, 0, 4);
                                enc.set_buffer(layer_buf, ffn_norm_off, 5);
                                enc.set_bytes(&eps.to_le_bytes(), 6);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((inter_dim as u64) / 8, 1, 1),
                                    MTLSize::new(64, 1, 1),
                                );
                                if needs_barriers {
                                    enc.memory_barrier_with_scope(1);
                                }
                                // up -> up_buf
                                enc.set_pipeline_state(&pipelines.qmv_q4_0_rmsnorm);
                                enc.set_buffer(uqw, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                enc.set_buffer(&s.up_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                enc.set_buffer(usc, 0, 4);
                                enc.set_buffer(layer_buf, ffn_norm_off, 5);
                                enc.set_bytes(&eps.to_le_bytes(), 6);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((inter_dim as u64) / 8, 1, 1),
                                    MTLSize::new(64, 1, 1),
                                );
                                if needs_barriers {
                                    enc.memory_barrier_with_scope(1);
                                }
                                // swiglu: gate_buf = silu(gate_buf) * up_buf
                                enc.set_pipeline_state(&pipelines.swiglu);
                                enc.set_buffer(&s.gate_buf, 0, 0);
                                enc.set_buffer(&s.up_buf, 0, 1);
                                enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 2);
                                let swg_tg = 256u64.min(inter_dim as u64).max(1);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((inter_dim as u64).div_ceil(swg_tg), 1, 1),
                                    MTLSize::new(swg_tg, 1, 1),
                                );
                            } else if super::q4_gateup_wide_enabled() {
                                // WIDE-load 256-thread variant: w_gate@0, x@1,
                                // out@2, in_dim@3, out_dim@4, w_up@5,
                                // gate_scales@6, up_scales@7, norm_w@8, eps@9.
                                enc.set_pipeline_state(
                                    &pipelines.rmsnorm_ffn_gate_up_swiglu_q4_0_wide,
                                );
                                enc.set_buffer(gqw, 0, 0);
                                enc.set_buffer(&s.attn_proj_buf, 0, 1);
                                enc.set_buffer(&s.gate_buf, 0, 2);
                                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                                enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                                enc.set_buffer(uqw, 0, 5);
                                enc.set_buffer(gsc, 0, 6);
                                enc.set_buffer(usc, 0, 7);
                                enc.set_buffer(layer_buf, ffn_norm_off, 8);
                                enc.set_bytes(&eps.to_le_bytes(), 9);
                                enc.dispatch_threadgroups(
                                    MTLSize::new((inter_dim as u64).div_ceil(8), 1, 1),
                                    MTLSize::new(256, 1, 1),
                                );
                            } else if let Some(pso_f16sc) = super::q4_gateup_f16sc_enabled()
                                .then_some(pipelines.qmv_q4_0_gate_up_swiglu_f16sc.as_ref())
                                .flatten()
                            {
                                // F16-SCALES gate/up: identical bindings + geometry to
                                // qmv_q4_0_gate_up_swiglu, but gsc/usc are f16 scale
                                // buffers (built f16 in preload_weights_gpu_resident) and
                                // the kernel reads them as `half`. ~10% fewer weight bytes.
                                //
                                // 1-SG OCCUPANCY VARIANT (LUMEN_METAL_Q4_GATEUP_1SG): when
                                // set and the 1sg kernel compiled, dispatch the
                                // 1-simdgroup-per-TG kernel (32 threads/TG, 4 rows/TG) over
                                // inter_dim/4 threadgroups = 2x more TGs than the 2-SG
                                // kernel's inter_dim/8 @ 64 threads. Byte-identical math
                                // (same per-row dequant + per-simdgroup RMSNorm reduction);
                                // only the TG/SG partition + thread count differ.
                                // 8-ROWS-PER-SG variant (LUMEN_METAL_Q4_GATEUP_8ROW):
                                // takes priority over 1sg when set + compiled +
                                // inter_dim%16==0. 2 SG/TG, 8 rows/SG, inter_dim/16 TGs
                                // (HALF the 4-row kernel's TGs, 2x x-register reuse).
                                // Byte-identical per output element.
                                let eight_row =
                                    super::q4_gateup_8row_enabled() && (inter_dim as u64) % 16 == 0;
                                let pso_8row = if eight_row {
                                    pipelines.qmv_q4_0_gate_up_swiglu_f16sc_8row.as_ref()
                                } else {
                                    None
                                };
                                let one_sg =
                                    super::q4_gateup_1sg_enabled() && (inter_dim as u64) % 4 == 0;
                                let pso_1sg = if one_sg {
                                    pipelines.qmv_q4_0_gate_up_swiglu_f16sc_1sg.as_ref()
                                } else {
                                    None
                                };
                                // F16-MATH variant (env LUMEN_METAL_Q4_GATEUP_F16MATH):
                                // SAME bindings + geometry as f16sc (2 SG/TG, 4 rows/SG,
                                // inter_dim/8 TGs, 64 threads) — only the per-32-block
                                // dequant MAC runs in half (~2x ALU). Near-tie; attacks
                                // the compute half of the dominant FFN matvec.
                                let pso_f16math = if super::q4_gateup_f16math_enabled() {
                                    pipelines.qmv_q4_0_gate_up_swiglu_f16sc_f16math.as_ref()
                                } else {
                                    None
                                };
                                // HALF2-MATH variant (env LUMEN_METAL_Q4_GATEUP_H2MATH):
                                // HIGHEST priority when set + compiled. SAME bindings +
                                // geometry as f16sc; the per-32-block dequant MAC runs in
                                // half2 (two half FMAs / vector ALU slot) — halves the
                                // dequant-MAC instruction count again on the dominant
                                // FFN matvec. Near-tie (even/odd half-lane partial-sum
                                // grouping; cross-block reduce/scale/SwiGLU stay f32).
                                let pso_h2math = if super::q4_gateup_h2math_enabled() {
                                    pipelines.qmv_q4_0_gate_up_swiglu_f16sc_h2math.as_ref()
                                } else {
                                    None
                                };
                                let (pso_gu, tg_count, tg_threads) = match pso_h2math {
                                    Some(p) => (p, (inter_dim as u64) / 8, 64u64),
                                    None => match pso_f16math {
                                        Some(p) => (p, (inter_dim as u64) / 8, 64u64),
                                        None => match pso_8row {
                                            Some(p) => (p, (inter_dim as u64) / 16, 64u64),
                                            None => match pso_1sg {
                                                Some(p) => (p, (inter_dim as u64) / 4, 32u64),
                                                None => (pso_f16sc, (inter_dim as u64) / 8, 64u64),
                                            },
                                        },
                                    },
                                };
                                enc.set_pipeline_state(pso_gu);
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
                                    MTLSize::new(tg_count, 1, 1),
                                    MTLSize::new(tg_threads, 1, 1),
                                );
                            } else {
                                // qmv_q4_0_gate_up_swiglu: w_gate@0, x@1, out@2, in_dim@3,
                                // gate_scales@4, w_up@5, up_scales@6, norm_w@7, eps@8.
                                // out_dim = inter_dim (%8==0); in_dim = hidden (%512==0).
                                enc.set_pipeline_state(&pipelines.qmv_q4_0_gate_up_swiglu);
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
                    enc.set_pipeline_state(pipelines.bf16_matvec_nr2());
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
                        // default 0=off); mirrors decode_single_cb.rs.
                        let k_splits = crate::metal::q4_qmv_down_splitk();
                        if k_splits >= 2 && (inter_dim as u32) % (512 * k_splits) == 0 {
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
                            // F16-scales FFN-down: when the scale buffer was built
                            // as f16 (env LUMEN_METAL_Q4_QMV_DOWN_F16SC=1 + kernel
                            // compiled) dispatch the f16sc kernel that reads
                            // `device const half*` scales; else the f32 kernel.
                            // BYTE-IDENTICAL math (only the scale element type and
                            // its widening cast differ). The buffer-build picks the
                            // matching layout, so the pipeline must match it here.
                            // HALF2-MATH variant (env LUMEN_METAL_Q4_DOWN_H2MATH=1):
                            // when the down f16-scale path is engaged (this flag
                            // self-engages it) AND the h2math kernel compiled, prefer
                            // the half2-vectorized dequant-MAC down kernel (2 half FMAs/
                            // ALU slot on the LONGEST-K matvec). Near-tie, not
                            // byte-identical. Falls back to f16sc, then the f32-scale
                            // kernel.
                            // F16-MATH variant (env LUMEN_METAL_Q4_DOWN_F16MATH=1):
                            // when the down f16-scale path is engaged AND the f16math
                            // kernel compiled, use the half-precision dequant-MAC down
                            // kernel (same f16 scale buffer, ~2x ALU on the unpack).
                            // Near-tie, not byte-identical. Falls back to f16sc then
                            // the f32-scale kernel.
                            let down_f16sc_on = crate::metal::q4_qmv_down_f16sc_enabled();
                            let down_pipe = if down_f16sc_on
                                && crate::metal::q4_down_h2math_enabled()
                                && pipelines.qmv_q4_0_residual_f16sc_h2math.is_some()
                            {
                                pipelines.qmv_q4_0_residual_f16sc_h2math.as_ref()
                            } else if down_f16sc_on {
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
                                enc.set_pipeline_state(pipelines.bf16_matvec_residual_nr2());
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

        // [decode-profile] Boundary before final norm + lm_head + argmax.
        // The just-finished section is the last layer's FFN; advance to "lm_head".
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

        // --- Final RMSNorm + Logits + Argmax ---
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
                // F16-scales fast path (env LUMEN_METAL_Q4_LMHEAD_F16SC=1): if the
                // scales buffer was built as f16 (2 B/block) and the f16sc kernel
                // compiled, dispatch qmv_q4_0_rmsnorm_f16sc (reads `half*` scales).
                // Byte-identical to the f32-scale kernel (f16 is the on-disk Q4_0
                // native scale precision). None => f32-scale qmv_q4_0_rmsnorm.
                let lmhead_f16sc_pipe = if super::q4_lmhead_f16sc_enabled() {
                    pipelines.qmv_q4_0_rmsnorm_f16sc.as_ref()
                } else {
                    None
                };
                // HALF2-MATH lm_head (env LUMEN_METAL_Q4_LMHEAD_H2MATH): when the
                // f16-scale path is engaged (lmhead_f16sc_pipe is Some — this flag
                // also turns f16sc on) AND the h2math kernel compiled, prefer the
                // half2-vectorized dequant-MAC variant qmv_q4_0_rmsnorm_f16sc_h2math.
                // SAME bindings + geometry (2 SG/TG, 4 rows/SG, vocab/8 TGs, 64
                // threads) — only the per-32-block dequant MAC runs as half2 (2 half
                // FMAs/ALU slot) on the single LARGEST matvec pool. Takes precedence
                // over plain f16sc when set.
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
                enc.set_buffer(&s.logits_buf, 0, 2);
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
                        enc.set_pipeline_state(pipelines.bf16_rmsnorm_matvec_nr2())
                    }
                    _ => {
                        enc.set_pipeline_state(&pipelines.rmsnorm_dequant_matmul_q8_0_deferred_nr2)
                    }
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
                    enc.set_pipeline_state(pipelines.bf16_matvec_nr2());
                    (128u64, 2u64)
                }
                _ => {
                    enc.set_pipeline_state(&pipelines.matmul_f32_deferred);
                    (128u64, 4u64)
                }
            };
            enc.set_buffer(sc_proj_buf, sc_proj_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(&s.logits_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 4);
            {
                let n_tg = ((vocab_size as u64) + proj_rows_per_tg - 1) / proj_rows_per_tg;
                enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(proj_tg, 1, 1));
            }
        }
        // Barrier: logits projection writes logits_buf, the final selection
        // kernel (argmax or gpu_sampler) reads logits_buf.
        if needs_barriers {
            enc.memory_barrier_with_scope(1);
        }
        // Final token selection. Default: GPU-side argmax (greedy). Option A:
        // when `pipe.sampler` is Some, the parity-matched `gpu_sampler` kernel
        // applies penalties + temperature + softmax + inverse-CDF draw and
        // writes the sampled token instead. SEQUENTIAL: write the shared
        // argmax_result_buf (read back below). PIPELINED: write the next
        // token-ring slot so the FOLLOWING CB's embed chains off it on the GPU
        // and the CPU reads it with a lag.
        let sampler_wiring = pipe.as_ref().and_then(|w| w.sampler.as_ref());
        if let Some(sw) = sampler_wiring {
            // Select the sampler kernel: the latency-hiding `gpu_sampler_fast`
            // by default, or the bit-exact single-thread `gpu_sampler` when
            // LUMEN_METAL_GPU_SAMPLER_EXACT=1 (validation). The route gate
            // (supports_gpu_sampler) ensured the chosen kernel compiled.
            let use_exact = super::metal_gpu_sampler_exact_enabled();
            let sampler_pso = if use_exact {
                pipelines.gpu_sampler.as_ref()
            } else {
                pipelines.gpu_sampler_fast.as_ref()
            }
            .ok_or_else(|| RuntimeError::Compute("gpu_sampler pipeline not compiled".into()))?;
            // GPU history frequency array (staged + seeded by the driver). Always
            // bound (Metal requires bound args); the kernel only reads/updates it
            // when pen_active != 0.
            let freq_arr = s
                .gpu_sampler_freq_arr
                .as_ref()
                .ok_or_else(|| RuntimeError::Compute("gpu_sampler freq array missing".into()))?;
            let w = pipe.as_ref().unwrap();
            enc.set_pipeline_state(sampler_pso);
            enc.set_buffer(&s.logits_buf, 0, 0);
            enc.set_buffer(&s.pipe_token_ring[w.argmax_write_slot], 0, 1);
            enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 2);
            enc.set_bytes(&sw.inv_temp.to_le_bytes(), 3);
            enc.set_buffer(freq_arr, 0, 4);
            enc.set_bytes(&sw.pen_active.to_le_bytes(), 5);
            enc.set_bytes(&sw.rep.to_le_bytes(), 6);
            enc.set_bytes(&sw.presence.to_le_bytes(), 7);
            enc.set_bytes(&sw.freq.to_le_bytes(), 8);
            enc.set_buffer(&s.gpu_sampler_rng_ring[sw.rng_read_slot], 0, 9);
            enc.set_buffer(&s.gpu_sampler_rng_ring[sw.rng_write_slot], 0, 10);
            enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        } else {
            // Two-pass tiled argmax for greedy decode. Pass-1: N threadgroups each
            // reduce a contiguous logits slice to a (max_val, arg_idx) partial (fills
            // the machine — a single-threadgroup argmax is bandwidth-starved at
            // ~2.9 GB/s over the vocab logits). Pass-2: one 256-thread TG reduces the
            // <=256 partials to the token id. Bit-identical selection to the
            // single-threadgroup argmax for every input (composite (value, i%256, i)
            // tie-break; see ffn_elementwise.msl). Writes the SAME output buffer + u32
            // format, so the device-side next-token embed is unchanged.
            let num_tiles = super::TILED_ARGMAX_TILES as u64;
            let tile_size = ((vocab_size as u64) + num_tiles - 1) / num_tiles;
            let actual_tiles = ((vocab_size as u64) + tile_size - 1) / tile_size;
            let idx_off = (super::ARGMAX_MAX_TILES * 4) as u64;
            // Pass-1: per-tile partials into argmax_partials_buf (vals@0, idxs@idx_off).
            enc.set_pipeline_state(&pipelines.argmax_tiled_partial);
            enc.set_buffer(&s.logits_buf, 0, 0);
            enc.set_buffer(&s.argmax_partials_buf, 0, 1);
            enc.set_buffer(&s.argmax_partials_buf, idx_off, 2);
            enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 3);
            enc.set_bytes(&(tile_size as u32).to_le_bytes(), 4);
            enc.dispatch_threadgroups(MTLSize::new(actual_tiles, 1, 1), MTLSize::new(256, 1, 1));
            // Barrier: pass-1 writes the partials, pass-2 reads them. The serial
            // encoder (has_gdn/all_dense) already orders dispatches; the concurrent
            // encoder needs the explicit barrier (mirrors the projection->argmax edge).
            if needs_barriers {
                enc.memory_barrier_with_scope(1);
            }
            // Pass-2: reduce partials to the token id in the argmax output buffer.
            enc.set_pipeline_state(&pipelines.argmax_tiled_reduce);
            enc.set_buffer(&s.argmax_partials_buf, 0, 0);
            enc.set_buffer(&s.argmax_partials_buf, idx_off, 1);
            if let Some(ref w) = pipe {
                enc.set_buffer(&s.pipe_token_ring[w.argmax_write_slot], 0, 2);
            } else {
                enc.set_buffer(&s.argmax_result_buf, 0, 2);
            }
            enc.set_bytes(&(actual_tiles as u32).to_le_bytes(), 3);
            enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        }

        enc.end_encoding();

        // ============================ PIPELINED ============================
        // Signal the ordering event at end-of-CB, commit ASYNC (do NOT wait),
        // and hand the in-flight CB back to the driver. The driver reads the
        // produced token from the ring with a one-token lag and advances the
        // CPU KV counter by exactly one per RETURNED token (so this method does
        // NOT touch kv -- the GPU KV write already used `w.seq_pos`). No
        // readback / XCHK / profiling here: those require a completed CB and are
        // only valid on the sequential path.
        if let Some(ref w) = pipe {
            if let Some(event) = w.event {
                cmd.encode_signal_event(event, w.signal_value);
            }
            cmd.commit();
            s.gpu_x_valid = false;
            return Ok(CoreResult::InFlight(cmd));
        }
        // =========================== SEQUENTIAL ===========================

        // Single sync point for the entire token.
        cmd.commit_and_wait();
        // [decode-gputime] Clean single-CB measurement: read the REAL command
        // buffer's GPUEndTime-GPUStartTime (true GPU busy) vs wall. No CB
        // splitting -> no distortion. Tells us CPU-encode vs GPU-execute split.
        // No-op unless LUMEN_METAL_DECODE_GPUTIME=1 (and NOT DECODE_PROFILE,
        // which splits the CB). Diagnostic only.
        if decode_profile::gputime_enabled() && !decode_profile::is_enabled() {
            decode_profile::record_gpu_time(cmd.gpu_elapsed_secs());
        }
        // [decode-profile] Record the final lm_head section and print a report
        // every 64 tokens (cheap; only when LUMEN_METAL_DECODE_PROFILE=1).
        if decode_profile::is_enabled() {
            decode_profile::record_gpu_final(cmd.gpu_elapsed_secs());
            decode_profile::record_final();
            decode_profile::maybe_report_and_reset(64);
        }
        // DET-001: stabilise the GPU-scheduler near-tie window on repeated
        // in-process decode calls (no-op when the delay resolves to 0).
        super::maybe_apply_metal_decode_delay();

        // [XCHK] Cross-backend per-op forensic probe (env LUMEN_XCHK=1, default
        // OFF -> byte-identical). Dumps layout-INDEPENDENT whole-buffer sumsq +
        // absmax of the SAME logical tensors the CUDA backend dumps, so the two
        // backends' decode trajectories can be aligned offline op-for-op and the
        // FIRST structurally-divergent (step, layer, tensor) located. The sharpest
        // signal is the per-MoE-layer top-K EXPERT IDS (router-flip origin); the
        // GDN h_state sumsq is the prime suspect to walk back to. Persistent GPU
        // buffers (gdn_h_states, gdn_conv_states, moe_per_layer_expert_ids,
        // logits_buf) are valid here after commit_and_wait().
        if {
            use std::sync::OnceLock;
            static XK: OnceLock<bool> = OnceLock::new();
            *XK.get_or_init(|| std::env::var("LUMEN_XCHK").as_deref() == Ok("1"))
        } {
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
            // step = 0-based DECODE ordinal (first generated token = step 0), so
            // it aligns byte-for-byte with the CUDA backend's `decode_token_count`
            // (which also starts at 0 for the first generated token) regardless of
            // prompt length. abs_pos is the KV position for cross-reference.
            use std::sync::atomic::{AtomicUsize, Ordering};
            static XCHK_STEP: AtomicUsize = AtomicUsize::new(0);
            let step = XCHK_STEP.fetch_add(1, Ordering::Relaxed);
            let abs_pos = seq_pos;
            eprintln!("[XCHK] step={step} abs_pos={abs_pos} === BEGIN decode-step ===");
            // Per-layer: GDN h_state / conv_state sumsq + MoE expert ids/weights.
            for layer_idx in 0..num_layers {
                let meta = &s.cached_layer_meta[layer_idx];
                if let Some(gdn_idx) = meta.gdn_layer_idx {
                    if gdn_idx < s.gdn_h_states.len() {
                        let hb = &s.gdn_h_states[gdn_idx];
                        let mut h = vec![0f32; (hb.length() / 4) as usize];
                        hb.read_f32(&mut h);
                        let (hsq, hmx) = sumsq_absmax(&h);
                        let cb = &s.gdn_conv_states[gdn_idx];
                        let mut c = vec![0f32; (cb.length() / 4) as usize];
                        cb.read_f32(&mut c);
                        let (csq, cmx) = sumsq_absmax(&c);
                        eprintln!(
                            "[XCHK] step={step} L={layer_idx} gdn_h_state sumsq={hsq:.6} absmax={hmx:.6}"
                        );
                        eprintln!(
                            "[XCHK] step={step} L={layer_idx} gdn_conv_state sumsq={csq:.6} absmax={cmx:.6}"
                        );
                    }
                }
                if meta.moe_meta.is_some() {
                    if let Some(Some(ids_buf)) = s.moe_per_layer_expert_ids.get(layer_idx) {
                        let mut ids = vec![0u32; s.moe_num_active_experts.max(1)];
                        ids_buf.read_u32(&mut ids);
                        // Router weights only if the per-layer weights buffer exists
                        // (router_debug). Otherwise dump ids alone (the decisive signal).
                        if let Some(Some(wts_buf)) = s.moe_per_layer_expert_weights.get(layer_idx) {
                            let mut wts = vec![0f32; s.moe_num_active_experts.max(1)];
                            wts_buf.read_f32(&mut wts);
                            eprintln!(
                                "[XCHK] step={step} L={layer_idx} moe_expert_ids={ids:?} gate_w={wts:?}"
                            );
                        } else {
                            eprintln!("[XCHK] step={step} L={layer_idx} moe_expert_ids={ids:?}");
                        }
                    }
                }
            }
            // Final pre-lm_head hidden (final-norm output = normed_buf after the
            // last layer wrote it for the logits projection).
            {
                let mut nb = vec![0f32; hidden_dim];
                s.normed_buf.read_f32(&mut nb);
                let (nsq, nmx) = sumsq_absmax(&nb);
                eprintln!("[XCHK] step={step} final_hidden sumsq={nsq:.6} absmax={nmx:.6}");
            }
            // Final LOGITS: whole-buffer sumsq + the top-8 (id, value).
            {
                let mut lg = vec![0f32; vocab_size];
                s.logits_buf.read_f32(&mut lg);
                let (lsq, lmx) = sumsq_absmax(&lg);
                let mut idx: Vec<usize> = (0..vocab_size).collect();
                idx.sort_by(|&a, &b| {
                    lg[b]
                        .partial_cmp(&lg[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let top: Vec<(usize, f32)> = idx.iter().take(8).map(|&i| (i, lg[i])).collect();
                eprintln!("[XCHK] step={step} logits sumsq={lsq:.6} absmax={lmx:.6} top8={top:?}");
            }
        }

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

        // Read only 4 bytes (u32 token ID) instead of 128 KB logits.
        let mut result = [0u32; 1];
        s.argmax_result_buf.read_u32(&mut result);

        drop(scratch_guard);

        Ok(CoreResult::Token(result[0]))
    }
}

/// Token ring size for the pipelined decode. Must be > pipeline depth so a slot
/// is never re-written before both its GPU consumer (the next CB's embed) and
/// its CPU reader have used it. With depth 2 the next writer of any slot is 4
/// CBs later (`RING_SIZE` apart), well outside the 2-deep in-flight window.
const PIPE_RING_SIZE: usize = 4;
/// Target number of command buffers kept in flight. Depth 2 overlaps the
/// CPU-encode of CB(k+1) with the GPU-execute of CB(k) — the latency it hides.
const PIPE_DEPTH: usize = 2;

impl MetalF32Backend {
    /// Drain (wait on) every in-flight pipelined-decode command buffer and clear
    /// the pipeline run state. Safe to call any time; a no-op when no pipeline is
    /// in flight. Called at the start of each generation (via
    /// `reset_recurrent_state`) so a fresh run never inherits a prior run's
    /// in-flight CBs, and defensively before the sequential decode path reuses
    /// scratch buffers.
    pub(crate) fn pipe_drain_locked(s: &mut super::types::MetalScratch) {
        while let Some((cmd, _, _)) = s.pipe_inflight.pop_front() {
            cmd.wait_until_completed();
        }
        s.pipe_step = 0;
    }

    /// Greedy decode driver: the single lean GPU-pipelined path (the one greedy decode path).
    ///
    /// Byte-identical GPU token-chaining + one-token-lag overlap, with the
    /// per-token driver overhead an event-ordered pipeline would pay stripped
    /// out (the mutex/VecDeque/event tax that cancels the ~0.6 ms overlap).
    /// Omitted vs an event-ordered driver:
    ///   1. The `MTLSharedEvent` ordering primitive ENTIRELY (`event: None`):
    ///      no per-token Obj-C event retain/release, no
    ///      `encode_wait_for_event` / `encode_signal_event` GPU ops. Correctness
    ///      relies only on the backend's single `MetalCommandQueue` being FIFO
    ///      (CB(k) retires before CB(k+1) executes -- exactly the invariant the
    ///      existing sampled-path `last_async_cmd` code already trusts).
    ///   2. The extra scratch-lock acquisitions an event-ordered driver takes: it would lock to poll
    ///      `need_more`, locks AGAIN to clone the event, locks AGAIN to push the
    ///      in-flight CB (3+ locks/token on top of the core's own lock). This
    ///      driver locks once for fresh-run setup, lets the core take its single
    ///      internal lock to encode, and locks once to record the in-flight CB
    ///      and once to harvest the front token.
    ///
    /// Determinism: byte-identical token stream -- same kernels, same order,
    /// same absolute `seq_pos` per CB, same embed-reads-prior-argmax token ring.
    /// Reuses the `pipe_token_ring` / `pipe_inflight` / `pipe_step` /
    /// `pipe_seq_pos` scratch state, so `pipe_drain_locked` (called from
    /// `reset_recurrent_state` at every generation start and before the
    /// sequential path reuses scratch) cleans it up unchanged.
    pub fn decode_token_greedy_lean(
        &self,
        prev_token: u32,
        weights: &dyn crate::weight::cache::WeightProvider,
        kv: &mut crate::kv::KvCache,
    ) -> Result<u32, RuntimeError> {
        // The lean pipeline splits the CB lifecycle (commit async / wait later);
        // incompatible with the decode-profile CB-splitting probe. Fall back to
        // the sequential path when profiling is enabled.
        if decode_profile::is_enabled() {
            return self.decode_token_greedy(prev_token, weights, kv);
        }

        // Effective in-flight depth: `PIPE_DEPTH` (2). The ring must hold > depth
        // slots so a slot is never re-written before its GPU consumer (the next CB's
        // embed) and CPU reader have used it; the next writer of any slot is
        // `ring_size` CBs later, so `depth + 2` is always safe.
        let depth = PIPE_DEPTH.max(1);
        let ring_size = PIPE_RING_SIZE.max(depth + 2);

        // -- Lazy one-time ring setup + fresh-run init. Single lock. No event
        //    allocation (the lean path never uses one). --
        {
            let mut guard = self.scratch.lock().unwrap();
            let s = guard
                .as_mut()
                .ok_or_else(|| RuntimeError::Compute("Metal scratch not initialized".into()))?;
            if s.pipe_token_ring.len() < ring_size {
                let mut ring = Vec::with_capacity(ring_size);
                for _ in 0..ring_size {
                    ring.push(self.device.new_buffer(4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate pipeline token buffer".into())
                    })?);
                }
                s.pipe_token_ring = ring;
            }
            if s.pipe_inflight.is_empty() {
                // Fresh run: step from 0, GPU position from the live CPU KV
                // length, seed ring[0] with the bootstrap token the first CB's
                // embed consumes.
                s.pipe_step = 0;
                s.pipe_seq_pos = kv.seq_len();
                s.pipe_token_ring[0].write_u32(&[prev_token]);
                #[cfg(debug_assertions)]
                {
                    use std::sync::OnceLock;
                    static ANNOUNCED: OnceLock<()> = OnceLock::new();
                    ANNOUNCED.get_or_init(|| {
                        eprintln!("[pipeline-lean] active (event-free, single-queue FIFO)");
                    });
                }
            }
        }

        // -- Top up the in-flight queue to `depth`. The core's CPU-encode of
        //    CB(step) overlaps the GPU-execute of the already-committed CB(s). --
        loop {
            // Read the encode counters (cheap, brief lock).
            let (need_more, step, seq_pos) = {
                let guard = self.scratch.lock().unwrap();
                let s = guard.as_ref().unwrap();
                (s.pipe_inflight.len() < depth, s.pipe_step, s.pipe_seq_pos)
            };
            if !need_more {
                break;
            }
            // Ring slots: CB(step) embed reads ring[step % R]; its argmax writes
            // ring[(step+1) % R].
            let embed_read_slot = step % ring_size;
            let argmax_write_slot = (step + 1) % ring_size;
            // event: None -> no ordering primitive; the single FIFO queue orders
            // CB(step) after CB(step-1). signal/wait values are unused.
            let wiring = PipeWiring {
                seq_pos,
                embed_read_slot,
                argmax_write_slot,
                sampler: None,
                event: None,
                signal_value: 0,
                wait_value: 0,
            };
            let cmd = match self.decode_token_greedy_core(prev_token, weights, kv, Some(wiring))? {
                CoreResult::InFlight(cmd) => cmd,
                CoreResult::Token(_) => {
                    return Err(RuntimeError::Compute(
                        "lean pipelined core unexpectedly returned a token".into(),
                    ));
                }
            };
            // Record the in-flight CB and advance the encode counters.
            let mut guard = self.scratch.lock().unwrap();
            let s = guard.as_mut().unwrap();
            s.pipe_inflight.push_back((cmd, step, seq_pos));
            s.pipe_step = step + 1;
            s.pipe_seq_pos = seq_pos + 1;
        }

        // -- Wait the FRONT in-flight CB, read its produced token, advance the
        //    CPU KV counter by one, return. --
        let (front_cmd, front_step) = {
            let mut guard = self.scratch.lock().unwrap();
            let s = guard.as_mut().unwrap();
            let (cmd, step, _seq) = s.pipe_inflight.pop_front().ok_or_else(|| {
                RuntimeError::Compute("lean pipelined decode: no in-flight CB to drain".into())
            })?;
            (cmd, step)
        };
        front_cmd.wait_until_completed();
        // [decode-gputime] STEP-2 lean-path measurement (no-op unless
        // LUMEN_METAL_DECODE_GPUTIME=1): record token-completion-to-completion
        // wall AND the front CB's true GPU-busy time. This is the async
        // analogue of the sequential record_gpu_time hook; it lets us read the
        // lean pipeline's effective wall/tok and GPU_util in-process (robust to
        // the shell/python harness jitter that contaminates external timing).
        if decode_profile::gputime_enabled() && !decode_profile::is_enabled() {
            decode_profile::record_lean_wall(front_cmd.gpu_elapsed_secs());
        }
        let token = {
            let guard = self.scratch.lock().unwrap();
            let s = guard.as_ref().unwrap();
            let slot = (front_step + 1) % ring_size;
            let mut out = [0u32; 1];
            s.pipe_token_ring[slot].read_u32(&mut out);
            out[0]
        };
        kv.advance_seq_len()?;
        Ok(token)
    }

    /// LEAN GPU-SAMPLED decode driver (Option A, `LUMEN_METAL_GPU_SAMPLER=1`).
    ///
    /// Identical pipelining to `decode_token_greedy_lean` -- same event-free,
    /// single-queue-FIFO, depth-`PIPE_DEPTH` token-ring chaining that overlaps
    /// CB(k+1)'s CPU-encode with CB(k)'s GPU-execute -- but the FINAL selection
    /// kernel is `gpu_sampler` (parity-matched to the CPU `sample_logits`)
    /// instead of `argmax`. This makes the SAMPLED (temperature>0) decode path
    /// pipeline at the SAME depth as the greedy win, removing the serial CPU
    /// sampler + logit readback that capped its rate.
    ///
    /// PARITY / STATE RINGS (checklist 1,2,5,7): besides the token ring it keeps
    /// (a) an RNG-state ring (`gpu_sampler_rng_ring`): CB(step) reads `[step%R]`,
    /// the kernel does exactly ONE xorshift64 next_u64, and writes the advanced
    /// state into `[(step+1)%R]` -- one draw per token, bit-identical to the CPU
    /// `Xorshift64`; and (b) a GPU history frequency array
    /// (`gpu_sampler_freq_arr`): the penalty phase reads it (one penalty per
    /// unique token == the CPU full-history freq map) and the kernel appends the
    /// chosen token to it, so a speculative CB(k+1) sees CB(k)'s token WITHOUT a
    /// CPU round-trip (the FIFO queue orders append-before-read). At fresh-run we
    /// seed ring[0] from `cfg.rng_seed_state` (== CPU `Xorshift64::new(seed)`)
    /// and the freq array from the prompt history -- exactly the CPU sampler's
    /// `Xorshift64::new` + `SamplerState` prompt seeding.
    ///
    /// The engine route gate (`supports_gpu_sampler`) guarantees: temperature>0,
    /// no top_k/top_p/min_p, full-history window (`repeat_last_n == None`), the
    /// kernel compiled, and the byte-guard inactive (always true for temp>0). So
    /// this driver is only ever invoked for the exact subset the kernel matches.
    pub fn decode_token_sampled_lean(
        &self,
        prev_token: u32,
        weights: &dyn crate::weight::cache::WeightProvider,
        kv: &mut crate::kv::KvCache,
        cfg: &crate::compute::GpuSamplerRunCfg,
    ) -> Result<u32, RuntimeError> {
        // Profiling splits the CB lifecycle; fall back to the sequential CPU
        // sampler path is handled by the engine (which routes to decode_token
        // when profiling). Here we mirror the greedy-lean guard defensively.
        if decode_profile::is_enabled() {
            // No sequential GPU-sampler path exists; signal the engine should not
            // have routed here. Return an error so the misroute is loud.
            return Err(RuntimeError::Compute(
                "gpu-sampler lean path is incompatible with decode profiling; \
                 use the CPU sampler"
                    .into(),
            ));
        }

        let depth = PIPE_DEPTH.max(1);
        let ring_size = PIPE_RING_SIZE.max(depth + 2);
        let inv_temp = cfg.inv_temp;
        let pen_active = cfg.pen_active;

        // -- Lazy ring/array setup + fresh-run seeding. Single lock. --
        {
            let mut guard = self.scratch.lock().unwrap();
            let s = guard
                .as_mut()
                .ok_or_else(|| RuntimeError::Compute("Metal scratch not initialized".into()))?;
            let vocab = s.vocab_size;
            // Token ring (shared with greedy lean).
            if s.pipe_token_ring.len() < ring_size {
                let mut ring = Vec::with_capacity(ring_size);
                for _ in 0..ring_size {
                    ring.push(self.device.new_buffer(4).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate pipeline token buffer".into())
                    })?);
                }
                s.pipe_token_ring = ring;
            }
            // RNG-state ring (8 bytes / u64 per slot).
            if s.gpu_sampler_rng_ring.len() < ring_size {
                let mut ring = Vec::with_capacity(ring_size);
                for _ in 0..ring_size {
                    ring.push(self.device.new_buffer(8).ok_or_else(|| {
                        RuntimeError::Compute("Failed to allocate gpu-sampler rng buffer".into())
                    })?);
                }
                s.gpu_sampler_rng_ring = ring;
            }
            // History frequency array (vocab u32). Always allocated so the encode
            // can bind it even when pen_active is false.
            if s.gpu_sampler_freq_arr.is_none() {
                let buf = self.device.new_buffer(vocab * 4).ok_or_else(|| {
                    RuntimeError::Compute("Failed to allocate gpu-sampler freq array".into())
                })?;
                s.gpu_sampler_freq_arr = Some(buf);
            }

            if s.pipe_inflight.is_empty() {
                // Fresh run: step from 0, GPU position from live CPU KV length,
                // seed the token ring[0] with the bootstrap token, seed the RNG
                // ring[0] with the CPU sampler's exact post-`new(seed)` state, and
                // seed the freq array from the prompt history.
                s.pipe_step = 0;
                s.pipe_seq_pos = kv.seq_len();
                s.pipe_token_ring[0].write_u32(&[prev_token]);
                s.gpu_sampler_rng_ring[0].write_u64_one(cfg.rng_seed_state);
                if let Some(ref fa) = s.gpu_sampler_freq_arr {
                    // Zero, then accumulate prompt-history counts (only meaningful
                    // when penalties are active; still zeroed otherwise so a prior
                    // run's residue never leaks).
                    let mut counts = vec![0u32; vocab];
                    if pen_active {
                        for &t in cfg.prompt_history.iter() {
                            let idx = t as usize;
                            if idx < vocab {
                                counts[idx] = counts[idx].saturating_add(1);
                            }
                        }
                    }
                    fa.write_u32(&counts);
                }
                {
                    // One-time release-visible confirmation (gated behind the
                    // GPU-sampler flag being on, so it never prints on any default
                    // path). LUMEN_METAL_GPU_SAMPLER_QUIET=1 silences it.
                    use std::sync::OnceLock;
                    static ANNOUNCED: OnceLock<()> = OnceLock::new();
                    ANNOUNCED.get_or_init(|| {
                        if std::env::var("LUMEN_METAL_GPU_SAMPLER_QUIET").as_deref() != Ok("1") {
                            let exact = super::metal_gpu_sampler_exact_enabled();
                            eprintln!(
                                "[gpu-sampler-lean] ACTIVE (kernel={}, event-free single-queue FIFO)",
                                if exact { "exact" } else { "fast" }
                            );
                        }
                    });
                }
            }
        }

        // -- Top up the in-flight queue to `depth`. --
        loop {
            let (need_more, step, seq_pos) = {
                let guard = self.scratch.lock().unwrap();
                let s = guard.as_ref().unwrap();
                (s.pipe_inflight.len() < depth, s.pipe_step, s.pipe_seq_pos)
            };
            if !need_more {
                break;
            }
            let embed_read_slot = step % ring_size;
            let argmax_write_slot = (step + 1) % ring_size;
            let rng_read_slot = step % ring_size;
            let rng_write_slot = (step + 1) % ring_size;
            let wiring = PipeWiring {
                seq_pos,
                embed_read_slot,
                argmax_write_slot,
                sampler: Some(SamplerWiring {
                    rng_read_slot,
                    rng_write_slot,
                    inv_temp,
                    pen_active: if pen_active { 1 } else { 0 },
                    rep: cfg.rep,
                    presence: cfg.presence,
                    freq: cfg.freq,
                }),
                event: None,
                signal_value: 0,
                wait_value: 0,
            };
            let cmd = match self.decode_token_greedy_core(prev_token, weights, kv, Some(wiring))? {
                CoreResult::InFlight(cmd) => cmd,
                CoreResult::Token(_) => {
                    return Err(RuntimeError::Compute(
                        "gpu-sampler lean core unexpectedly returned a token".into(),
                    ));
                }
            };
            let mut guard = self.scratch.lock().unwrap();
            let s = guard.as_mut().unwrap();
            s.pipe_inflight.push_back((cmd, step, seq_pos));
            s.pipe_step = step + 1;
            s.pipe_seq_pos = seq_pos + 1;
        }

        // -- Wait the FRONT in-flight CB, read its produced token, advance KV. --
        let (front_cmd, front_step) = {
            let mut guard = self.scratch.lock().unwrap();
            let s = guard.as_mut().unwrap();
            let (cmd, step, _seq) = s.pipe_inflight.pop_front().ok_or_else(|| {
                RuntimeError::Compute("gpu-sampler lean decode: no in-flight CB to drain".into())
            })?;
            (cmd, step)
        };
        front_cmd.wait_until_completed();
        if decode_profile::gputime_enabled() && !decode_profile::is_enabled() {
            decode_profile::record_lean_wall(front_cmd.gpu_elapsed_secs());
        }
        let token = {
            let guard = self.scratch.lock().unwrap();
            let s = guard.as_ref().unwrap();
            let slot = (front_step + 1) % ring_size;
            let mut out = [0u32; 1];
            s.pipe_token_ring[slot].read_u32(&mut out);
            out[0]
        };
        kv.advance_seq_len()?;
        Ok(token)
    }
}
