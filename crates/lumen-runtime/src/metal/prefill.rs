//! Prefill orchestration methods for Metal backend.
//!
//! Extracted from mod.rs for modularity.
//! These are methods on MetalF32Backend that orchestrate the batched prefill pipeline.

use crate::error::RuntimeError;
use crate::metal::ffi::MetalCommandBuffer;
use super::{MetalPipelines, MetalScratch, MetalF32Backend, MetalBuffer, MTLSize};
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

}
