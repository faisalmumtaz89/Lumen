//! Prefill orchestration methods for Metal backend.
//!
//! Extracted from mod.rs for modularity.
//! These are methods on MetalF32Backend that orchestrate the batched prefill pipeline.

use crate::error::RuntimeError;
use crate::metal::ffi::MetalCommandBuffer;
use super::{MetalPipelines, MetalScratch, MetalF32Backend, MetalBuffer, MTLSize};
use super::graph_reorder;
use lumen_format::quantization::QuantScheme;

impl MetalF32Backend {
    /// Embed a batch of token ids into the batch x_buf on the GPU.
    /// Encode token embedding lookup into an existing command buffer.
    ///
    /// Previously this created its own command buffer + commit_and_wait(),
    /// adding ~11ms of GPU-CPU roundtrip overhead per prefill. Now the embed
    /// dispatch is the FIRST encoder in the prefill command buffer, and Metal's
    /// implicit encoder barriers guarantee the embed output lands in batch_x_buf
    /// before layer 0 reads it.
    pub(crate) fn encode_embed_batched(
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
            QuantScheme::Bf16 => enc.set_pipeline_state(&pipelines.embed_tokens_batched_bf16),
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
    pub(crate) fn read_last_hidden(
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
        //
        // PROFILE MODE (LUMEN_METAL_PROFILE=1): use the profilable CB
        // constructor so that each call to `new_compute_encoder()` at a
        // labelled section boundary auto-splits the CB and records GPU
        // time CPU-side. Single-CB structure preserved when profile is OFF.
        // ================================================================
        // opt into commandBufferWithUnretainedReferences when
        // LUMEN_METAL_UNRETAINED_CMDBUFS=1. Safe because every MTLBuffer /
        // MTLComputePipelineState bound by encoders below is owned by
        // `MetalF32Backend.s` (the `&self` of this method) which outlives the
        // commit_and_wait(). Profile-mode honours the opt-in too so the
        // sectioned splits also avoid the retain/release pair.
        let unretained = super::graph_reorder::unretained_cmdbufs_enabled();

        // number of command buffers to split the prefill into.
        // K=1: legacy single-CB path. K>=2: multi-CB in-flight
        // path where CB[0..K-1] are commit()'d (no wait) once their layers
        // are encoded, and CB[K-1] is the only one that calls
        // commit_and_wait(). Same MTLCommandQueue guarantees FIFO submission
        // order, so the implicit cross-CB ordering on `batch_x_buf` is
        // preserved without an explicit fence.
        //
        // Multi-CB is INCOMPATIBLE with the concurrent_encoder_full outer-encoder
        // path (single encoder spans all layers and cannot cross a CB
        // boundary), with profile mode (the profiler splits CBs at its
        // own granularity), and with batch_size < num_layers (no point
        // splitting). Each of these falls back to K=1.
        let cb_count_request = super::graph_reorder::multi_cb_count();
        let helper_new_cb = |unretained: bool, profilable: bool| -> Result<MetalCommandBuffer, RuntimeError> {
            let res = match (profilable, unretained) {
                (true, true)  => self.queue.new_command_buffer_profilable_unretained(),
                (true, false) => self.queue.new_command_buffer_profilable(),
                (false, true) => self.queue.new_command_buffer_unretained(),
                (false, false) => self.queue.new_command_buffer(),
            };
            res.ok_or_else(|| {
                RuntimeError::Compute("Failed to create command buffer for prefill".into())
            })
        };
        let cmd = helper_new_cb(unretained, super::profile::is_enabled())?;

        // Initialise per-section profile state for this prefill (cheap no-op
        // when LUMEN_METAL_PROFILE=0). The first section is "prefill/embed";
        // we record it (with the empty-CB elapsed) immediately so subsequent
        // splits attribute correctly under the section that was actually
        // encoded since the last split.
        super::profile::reset_accum();
        super::profile::set_section("prefill/embed");
        // Promote NEXT_SECTION -> IN_FLIGHT_SECTION via a zero-time record.
        // The zero-cost entry under "(unlabelled)" is harmless and shows up
        // last in the sorted report.
        super::profile::mark_section_start();
        super::profile::record_section_end();
        // Now IN_FLIGHT_SECTION = "prefill/embed"; restart the timer for it.
        super::profile::mark_section_start();

        // Embed all tokens as first encoder in the prefill command buffer
        self.encode_embed_batched(&cmd, prompt_tokens, pipelines, s)?;

        // Pre-load all layer views and keep them alive until after GPU commit.
        // This ensures the backing memory (mmap pointers or Arc<[u8]>) remains
        // valid while the GPU processes the command buffer.
        //
        // E.5: `begin_pass` is "called once per forward pass" per the
        // trait contract (`cache.rs:309-313`). the prior loop body called
        // it per-layer, but after layer 0 the compute cursor is set to 0 by
        // `begin_pass` and immediately overwritten by `try_get_layer`'s
        // `fetch_max(layer, ...)` — the per-iteration resets in iterations
        // 2..N were no-ops at best and semantic noise at worst. Moving the
        // call outside the loop matches the documented intent at
        // `provider_mmap.rs:243-249` and eliminates 21-31 redundant atomic
        // stores per prefill.
        weights.begin_pass();
        let mut layer_views = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            // get_layer_raw (NOT get_layer_blocking) on the fallback path: the
            // batched-prefill kernels read weight bytes from the GPU-resident
            // *raw* unified buffer (uploaded via get_layer_raw) but take the
            // per-subtensor OFFSETS from this LayerView's `subtensors`. Those
            // offsets MUST describe the raw native-quant blob layout.
            // SyncWeightProvider::get_layer_blocking returns a re-quantized F32
            // blob with DIFFERENT offsets (and stale ssm_* offsets), so using it
            // here made the kernels read the raw buffer at the wrong offsets ->
            // pad-token garbage. MmapWeightProvider hits the `Some` arm above
            // (try_get_layer returns its pre-built raw zero-copy view), so this
            // fallback only fires for the sync/async providers, which both
            // return raw bytes + raw offsets from get_layer_raw.
            let layer_view = match weights.try_get_layer(layer) {
                Some(view) => view,
                None => weights.get_layer_raw(layer)?,
            };
            layer_views.push(layer_view);
        }

        // ================================================================
        // whole-prefill outer concurrent encoder
        // ================================================================
        //
        // When `LUMEN_METAL_CONCURRENT_ENCODER_FULL=1` is set AND every layer in this prefill
        // is outer-encoder eligible, open ONE concurrent compute encoder
        // right here and thread it through all `num_layers` layers. This
        // consolidates 32 per-layer concurrent encoders into
        // 1 outer encoder, eliminating 31 encoder-boundary transitions per
        // prefill. Cross-layer hazards are expressed via an explicit
        // resource-scoped `memoryBarrierWithResources:` listing all 11
        // shared layer-local activation buffers (x_buf, normed_buf, qkv_buf,
        // q_buf, k_buf, v_buf, attn_out_buf, attn_proj_buf, gate_buf,
        // up_buf, scores_buf) between consecutive layers. Per-layer KV
        // cache slots are not in the barrier because they are layer-local
        // and serialised within the wavefront scheduler's emit-plan.
        //
        // Validate mode (`LUMEN_METAL_CONCURRENT_ENCODER_FULL_VALIDATE=1`) forces the outer
        // encoder to be a SERIAL `new_compute_encoder()` instead of a
        // concurrent one, so dispatch ordering matches the legacy per-layer
        // path modulo encoder boundaries. Used to isolate cross-layer
        // barrier bugs from concurrent-scheduling bugs.
        //
        // Eligibility (per-layer): `layer_outer_eligible`. If any
        // layer is ineligible (MoE, legacy per-layer GDN, deep-profile, etc.),
        // the outer-encoder path is silently skipped and the legacy
        // per-layer encoder loop runs.
        let concurrent_encoder_full = graph_reorder::concurrent_encoder_full_enabled() && graph_reorder::concurrent_encoder_enabled();
        let serial_validate_full = graph_reorder::concurrent_encoder_full_validate_serial();

        let all_layers_outer_eligible = concurrent_encoder_full
            && (0..num_layers).all(|l| {
                self.layer_outer_eligible(s, &layer_views[l], batch_size)
            });

        if all_layers_outer_eligible {
            // fast path: one outer encoder spans all layers.
            //
            // Edit 4 — Validate gate: when LUMEN_METAL_CONCURRENT_ENCODER_FULL_VALIDATE=1,
            // open the outer encoder as a serial `new_compute_encoder()`
            // so dispatches run in program order (matches legacy serial
            // semantics). Cross-layer barriers are no-ops in serial mode
            // (the `emit_cross_layer_barrier` helper short-circuits when
            // `serial_validate` is true).
            let outer_enc = if serial_validate_full {
                cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute(
                        ": failed to create outer serial validate encoder".into(),
                    )
                })?
            } else {
                cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute(
                        ": failed to create outer concurrent encoder".into(),
                    )
                })?
            };

            // Resolve the layer-shared activation buffers ONCE so the
            // per-iteration cross-layer barrier can reference them by
            // resource scope (cheaper than whole-buffer scope on Apple's
            // hazard tracker). All these buffers are allocated by
            // `ensure_batch_buffers` and live for the entire prefill.
            //
            // Storing raw pointers to bypass the `s: &mut MetalScratch`
            // borrow that the per-layer dispatcher needs. The buffers are
            // alive for the entire prefill (scratch holds them in fields
            // that are not dropped until the function returns), and we
            // only use the references to emit cross-layer barriers (Metal
            // API calls that do not retain). Constructing `&MetalBuffer`
            // from known-live raw pointers is safe.
            let x_buf_ptr: *const MetalBuffer = s.batch_x_buf.as_ref()
                .ok_or_else(|| RuntimeError::Compute(": batch_x_buf missing".into()))?;
            let normed_buf_ptr: *const MetalBuffer = s.batch_normed_buf.as_ref()
                .ok_or_else(|| RuntimeError::Compute(": batch_normed_buf missing".into()))?;
            let qkv_buf_ptr: *const MetalBuffer = s.batch_qkv_buf.as_ref()
                .ok_or_else(|| RuntimeError::Compute(": batch_qkv_buf missing".into()))?;
            let q_buf_ptr: *const MetalBuffer = s.batch_q_buf.as_ref()
                .ok_or_else(|| RuntimeError::Compute(": batch_q_buf missing".into()))?;
            let k_buf_ptr: *const MetalBuffer = s.batch_k_buf.as_ref()
                .ok_or_else(|| RuntimeError::Compute(": batch_k_buf missing".into()))?;
            let v_buf_ptr: *const MetalBuffer = s.batch_v_buf.as_ref()
                .ok_or_else(|| RuntimeError::Compute(": batch_v_buf missing".into()))?;
            let attn_out_buf_ptr: *const MetalBuffer = s.batch_attn_out_buf.as_ref()
                .ok_or_else(|| RuntimeError::Compute(": batch_attn_out_buf missing".into()))?;
            let attn_proj_buf_ptr: *const MetalBuffer = s.batch_attn_proj_buf.as_ref()
                .ok_or_else(|| RuntimeError::Compute(": batch_attn_proj_buf missing".into()))?;
            let gate_buf_ptr: *const MetalBuffer = s.batch_gate_buf.as_ref()
                .ok_or_else(|| RuntimeError::Compute(": batch_gate_buf missing".into()))?;
            let up_buf_ptr: *const MetalBuffer = s.batch_up_buf.as_ref()
                .ok_or_else(|| RuntimeError::Compute(": batch_up_buf missing".into()))?;
            let scores_buf_ptr: *const MetalBuffer = s.batch_scores_buf.as_ref()
                .ok_or_else(|| RuntimeError::Compute(": batch_scores_buf missing".into()))?;

            for layer in 0..num_layers {
                let mut kv_view = kv.view_mut(layer)?;

                // Cross-layer barrier on ALL shared layer-local buffers.
                // The per-layer wavefront scheduler emits within-layer
                // barriers, but the cross-layer state is the union of all
                // hazards on buffers carried across layer boundaries:
                //   x_buf (residual carrier, WAR)
                //   normed_buf, qkv_buf, q_buf, k_buf, v_buf, attn_out_buf,
                //     attn_proj_buf, gate_buf, up_buf, scores_buf (WAW + WAR)
                //
                // Use resource-scoped barriers on each carried buffer
                // (cheaper than whole-buffer scope: Apple's hazard tracker
                // only serialises against the listed resources, leaving
                // unrelated in-flight work free to retire). In legacy
                // each layer opened its own encoder and
                // the encoder-end / encoder-begin transition served as
                // this implicit serialisation point; expresses it
                // explicitly here.
                //
                // Skip layer 0: embed wrote x_buf via its own encoder
                // (`encode_embed_batched` opens + closes its own encoder),
                // and Metal's between-encoder boundary already serialised
                // that write.
                //
                // In serial validate mode (`LUMEN_METAL_CONCURRENT_ENCODER_FULL_VALIDATE=1`),
                // dispatches execute in program order even within a single
                // encoder, so the barrier is a no-op for correctness but
                // is emitted anyway for cache-coherence robustness.
                if layer > 0 {
                    // SAFETY: all the *_ptr values point to MetalBuffers
                    // owned by `s` (MetalScratch), which is borrowed
                    // mutably for the entire `for` loop. The buffers are
                    // not moved or dropped during the loop.
                    let bufs: [&MetalBuffer; 11] = unsafe {[
                        &*x_buf_ptr, &*normed_buf_ptr, &*qkv_buf_ptr,
                        &*q_buf_ptr, &*k_buf_ptr, &*v_buf_ptr,
                        &*attn_out_buf_ptr, &*attn_proj_buf_ptr,
                        &*gate_buf_ptr, &*up_buf_ptr, &*scores_buf_ptr,
                    ]};
                    outer_enc.memory_barrier_with_resources(&bufs);
                }

                self.encode_layer_batched_into(
                    &cmd, &outer_enc, layer, batch_size, &layer_views[layer],
                    &mut kv_view, pipelines, s,
                )?;

                kv.commit_view(kv_view)?;
            }

            outer_enc.end_encoding();
            // concurrent_encoder_full path uses the single `cmd` from the outset (no multi-CB
            // because the outer encoder spans all layers). Commit and wait
            // here so the fall-through below skips the path.
            cmd.commit_and_wait();
        } else {
            // Existing default path: 32 iterations, per-layer encoders.
            // Each layer adds ~13 compute encoders. Metal guarantees ordering
            // between encoders within the same command buffer (implicit
            // barriers). The scratch guard is held throughout, eliminating
            // 22 lock/unlock cycles in the inner loop.
            //
            // when LUMEN_METAL_MULTI_CB=1, split the prefill
            // across `cb_count_request` CBs (default 2). Each CB carries
            // a contiguous range of layers. CB[k] is commit()'d (no wait)
            // once its last layer is encoded, then encoding of CB[k+1]
            // proceeds while CB[k] runs on the GPU. The final CB is the
            // one that calls commit_and_wait() at the end of prefill().
            //
            // Multi-CB is disabled when profile mode is active (profiler
            // already splits CBs at section boundaries, double-splitting
            // would scramble attribution). When K=1, the multi-CB branch is
            // bypassed and the code path is bit-exact to the K=1 loop above.
            let multi_cb = cb_count_request > 1 && !super::profile::is_enabled();
            if multi_cb {
                let k = cb_count_request.min(num_layers);
                // Per-CB layer count, with the remainder loaded onto the
                // FINAL CB so the early CBs are smaller and finish first.
                // Example for num_layers=32, K=2: CB[0] has 16, CB[1] has
                // 16. For num_layers=32, K=3: CB[0]=10, CB[1]=10, CB[2]=12.
                let base = num_layers / k;
                // The first CB is `cmd` (already constructed above and
                // already contains the embed encoder). Subsequent CBs are
                // constructed lazily within the loop.
                let mut current_cb = cmd;
                let mut cur_idx: usize = 0;
                // Layer ranges by CB: cb 0 -> [0, base); cb 1 -> [base, 2*base); ...; cb K-1 -> [(K-1)*base, num_layers).
                // The remainder goes to the LAST CB.
                let cb_end = |k_idx: usize| -> usize {
                    if k_idx + 1 == k { num_layers } else { (k_idx + 1) * base + 0 }
                };
                // Track the queue of in-flight CBs we still hold a handle to.
                // We only need to wait on the LAST one (FIFO ordering on
                // same queue means later CBs imply earlier are done), but
                // we MUST keep early CBs alive until they have been
                // committed — `commit()` is non-blocking; the CB handle's
                // Drop releases the underlying MTLCommandBuffer pointer.
                // After commit() the GPU still references the CB so it is
                // safe for the handle to be dropped, but to keep semantics
                // crystal-clear we hold them in a Vec
                // until the final wait completes.
                let mut held_cbs: Vec<MetalCommandBuffer> = Vec::with_capacity(k);
                for layer in 0..num_layers {
                    let mut kv_view = kv.view_mut(layer)?;
                    self.encode_layer_batched(
                        &current_cb, layer, batch_size, &layer_views[layer],
                        &mut kv_view, pipelines, s,
                    )?;
                    kv.commit_view(kv_view)?;

                    // If this layer is the LAST in its CB, commit and (unless
                    // it's the final CB) start the next CB. The final CB is
                    // committed via the unified `commit_and_wait()` below.
                    let is_last_in_cb = (layer + 1) == cb_end(cur_idx);
                    let is_final_cb = cur_idx + 1 == k;
                    if is_last_in_cb && !is_final_cb {
                        // Replace current_cb with a freshly constructed CB
                        // for the next slice. We keep the old `current_cb`
                        // alive in `held_cbs` so its underlying MTLCommandBuffer
                        // is not released until the final wait completes.
                        let next_cb = helper_new_cb(unretained, false)?;
                        let prev_cb = std::mem::replace(&mut current_cb, next_cb);
                        // commit() is the asynchronous variant; CPU returns
                        // immediately and GPU enqueues this CB behind any
                        // prior CBs on the same queue.
                        prev_cb.commit();
                        held_cbs.push(prev_cb);
                        cur_idx += 1;
                    }
                }
                // The final CB is `current_cb`. Single sync point closes
                // the entire prefill — earlier CBs are guaranteed done
                // before this returns (FIFO ordering on same queue).
                current_cb.commit_and_wait();
                // Now safe to drop held CBs; their GPU work is complete.
                drop(held_cbs);
            } else {
                // NaN dump (opt-in via `LUMEN_METAL_NAN_DUMP=1`):
                // commit each layer's CB separately and scan x_buf for NaN
                // after each layer. Used to pinpoint the first layer that
                // produces NaN. Default off; production path is the loop
                // immediately below.
                let nan_dump = std::env::var("LUMEN_METAL_NAN_DUMP")
                    .ok()
                    .as_deref()
                    == Some("1");
                if nan_dump {
                    let hidden_dim = s.hidden_dim;
                    let total = batch_size * hidden_dim;
                    cmd.commit_and_wait();
                    {
                        let xb = s.batch_x_buf.as_ref().unwrap();
                        let mut buf = vec![0.0f32; total];
                        xb.read_f32(&mut buf);
                        let nans = buf.iter().filter(|v| v.is_nan()).count();
                        let infs = buf.iter().filter(|v| v.is_infinite()).count();
                        eprintln!("post-embed batch_size={batch_size} NaN={nans}/{total} Inf={infs}");
                    }
                    for layer in 0..num_layers {
                        let layer_cmd = helper_new_cb(unretained, false)?;
                        let mut kv_view = kv.view_mut(layer)?;
                        self.encode_layer_batched(
                            &layer_cmd, layer, batch_size, &layer_views[layer],
                            &mut kv_view, pipelines, s,
                        )?;
                        kv.commit_view(kv_view)?;
                        layer_cmd.commit_and_wait();
                        let xb = s.batch_x_buf.as_ref().unwrap();
                        let mut buf = vec![0.0f32; total];
                        xb.read_f32(&mut buf);
                        let nans = buf.iter().filter(|v| v.is_nan()).count();
                        let infs = buf.iter().filter(|v| v.is_infinite()).count();
                        let any_nan = nans > 0;
                        let first_f = buf.iter().copied().take(4).collect::<Vec<_>>();
                        eprintln!(
                            "post-layer {layer:02} NaN={nans}/{total} Inf={infs} first4={first_f:?}{}",
                            if any_nan { "  <-- NaN ENTERED" } else { "" }
                        );
                    }
                } else {
                    for layer in 0..num_layers {
                        let mut kv_view = kv.view_mut(layer)?;
                        self.encode_layer_batched(
                            &cmd, layer, batch_size, &layer_views[layer],
                            &mut kv_view, pipelines, s,
                        )?;
                        kv.commit_view(kv_view)?;
                    }
                    cmd.commit_and_wait();
                }
            }
        }

        // both prefill paths (the `concurrent_encoder_full` branch above, and
        // the default-with-optional-multi-CB below) now commit + wait
        // before returning. `cmd` and any auxiliary CBs in `held_cbs` are
        // dropped when their scope exits — either inside the concurrent_encoder_full
        // branch (where `cmd` is moved into `commit_and_wait`) or inside
        // the multi-CB / single-CB sub-branch of the else (where
        // `current_cb` is consumed by `commit_and_wait`).
        //
        // Profile mode: record GPU time for the final in-flight section
        // (the one that did not get a follow-up `new_compute_encoder` call
        // to split it). No-op when profiling is OFF.
        super::profile::record_section_end();

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

}
