//! MoE (Mixture of Experts) encode functions and configuration for Metal backend.
//!
//! Extracted from mod.rs for modularity.

use crate::error::RuntimeError;
use crate::expert::cache::ExpertLfuCache;
use crate::metal::ffi::{
    MetalBuffer, MetalCommandBuffer, MetalComputeEncoder, MTLSize,
};
use lumen_format::quantization::QuantScheme;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use super::{
    MetalPipelines, MetalScratch, CachedLayerMeta, CachedMoeMeta,
    MetalF32Backend, RouterLayerStats,
};

impl MetalF32Backend {

    /// Configure MoE expert caching for streaming decode.
    ///
    /// Must be called BEFORE `init()`. Only effective for MoE models
    /// (num_experts > 0) in streaming mode (non-GPU-resident).
    ///
    /// - `lbc_path`: Path to the LBC model file for per-expert byte-range reads.
    /// - `cache_capacity`: Maximum number of experts to cache (0 = profiling only).
    pub fn configure_expert_cache(
        &mut self,
        lbc_path: &std::path::Path,
        cache_capacity: usize,
    ) {
        self.lbc_path = Some(lbc_path.to_path_buf());
        if cache_capacity > 0 {
            self.expert_cache = Some(Mutex::new(ExpertLfuCache::new(cache_capacity)));
        }
    }

    /// Configure profiling-based cache warm-up.
    ///
    /// After `profiling_tokens` tokens have been decoded, the profiler's
    /// activation counts are used to pre-populate the expert cache with
    /// the `top_k_per_layer` hottest experts from each MoE layer.
    ///
    /// Must be called BEFORE inference starts. Has no effect if expert_cache
    /// is not configured or the model is not MoE.
    ///
    /// - `profiling_tokens`: Number of tokens to observe before triggering warmup.
    /// - `top_k_per_layer`: Number of experts per layer to cache (e.g., 4 for top-4).
    pub fn configure_expert_warmup(
        &mut self,
        profiling_tokens: usize,
        top_k_per_layer: usize,
    ) {
        self.profiling_tokens_remaining.store(profiling_tokens, Ordering::Relaxed);
        self.profiling_top_k = top_k_per_layer;
        self.warmup_complete.store(false, Ordering::Relaxed);
    }

    /// Configure cache-conditional routing bias.
    ///
    /// When `lambda > 0.0`, the MoE router adds `lambda` to logits of cached
    /// experts before softmax, nudging borderline expert selections toward
    /// already-cached experts to reduce disk I/O. A typical value is 0.05-0.2.
    ///
    /// Set to 0.0 (default) to disable biased routing entirely.
    pub fn configure_routing_bias(&mut self, lambda: f32) {
        self.cache_bias_lambda = lambda;
    }

    /// Configure Option A dispatch for MoE decode.
    ///
    /// When enabled, the MoE FFN decode path dispatches only the top-K selected
    /// expert FFNs instead of all num_experts (Option B). This eliminates 75%+
    /// of expert FFN compute for models like Mixtral (8 experts, top-2).
    ///
    /// In streaming mode: two-CB split with expert cache/reader I/O.
    /// In GPU-resident mode: two-CB split per MoE layer with
    /// synchronous router readback from per-layer expert_ids buffers.
    ///
    /// Default: false (Option B -- all experts dispatched).
    pub fn configure_option_a(&mut self, enabled: bool) {
        self.use_option_a = enabled;
    }

    /// expert_ids and expert_weights from GPU. The results are accumulated in
    /// an internal log retrievable via `get_router_debug_log()`.
    ///
    /// Must be called BEFORE `init()` so that per-layer expert_weights buffers
    /// are allocated. Calling after init has no effect on buffer allocation
    /// (only the readback flag is toggled).
    pub fn configure_router_debug(&mut self, enabled: bool) {
        self.router_debug_enabled = enabled;
    }

    /// Returns a human-readable router debug summary string.
    ///
    /// Analyzes the accumulated router debug log to report per-layer entropy,
    /// most frequently selected expert, and whether routing diversity exists.
    /// Returns None if no router debug data has been collected.
    pub fn router_debug_summary(&self) -> Option<String> {
        let log = self.router_debug_log.lock().unwrap();
        if log.is_empty() {
            return None;
        }

        // Group entries by layer.
        let max_layer = log.iter().map(|s| s.layer).max().unwrap_or(0);
        let num_layers = max_layer + 1;

        let mut out = String::new();
        out.push_str("--- Router Diagnostics ---\n");

        for layer in 0..num_layers {
            let entries: Vec<&RouterLayerStats> = log.iter()
                .filter(|s| s.layer == layer)
                .collect();
            if entries.is_empty() {
                continue;
            }

            // Collect all expert selections and weights.
            let mut expert_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
            let mut weight_sums: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
            let num_tokens = entries.len();

            for entry in &entries {
                for (i, &eid) in entry.expert_ids.iter().enumerate() {
                    *expert_counts.entry(eid).or_insert(0) += 1;
                    *weight_sums.entry(eid).or_insert(0.0) += entry.expert_weights.get(i).copied().unwrap_or(0.0);
                }
            }

            // Compute selection entropy (over expert_ids[0] -- the top-1 expert).
            let top1_counts: std::collections::HashMap<u32, usize> = entries.iter()
                .filter_map(|e| e.expert_ids.first().copied())
                .fold(std::collections::HashMap::new(), |mut acc, eid| {
                    *acc.entry(eid).or_insert(0) += 1;
                    acc
                });
            let total_top1 = num_tokens as f64;
            let entropy: f64 = top1_counts.values()
                .map(|&c| {
                    let p = c as f64 / total_top1;
                    if p > 0.0 { -p * p.log2() } else { 0.0 }
                })
                .sum();

            // Find most frequent top-1 expert.
            let (most_freq_expert, most_freq_count) = top1_counts.iter()
                .max_by_key(|(_, &c)| c)
                .map(|(&eid, &c)| (eid, c))
                .unwrap_or((0, 0));

            // Average top-1 weight.
            let avg_top1_weight: f32 = entries.iter()
                .filter_map(|e| e.expert_weights.first().copied())
                .sum::<f32>() / num_tokens.max(1) as f32;

            // Average weight spread (top1 - top2).
            let avg_spread: f32 = entries.iter()
                .map(|e| e.weight_spread)
                .sum::<f32>() / num_tokens.max(1) as f32;

            // Use abs() to avoid displaying "-0.000" when entropy is negative zero.
            out.push_str(&format!(
                "  Layer {:2}: entropy={:.3}, top1_expert={} ({}/{} = {:.1}%), avg_top1_weight={:.4}, avg_spread={:.4}\n",
                layer, entropy.abs(), most_freq_expert, most_freq_count, num_tokens,
                most_freq_count as f64 / total_top1 * 100.0, avg_top1_weight, avg_spread,
            ));
        }

        // Overall determination.
        let all_entropies: Vec<f64> = (0..num_layers).map(|layer| {
            let entries: Vec<&RouterLayerStats> = log.iter()
                .filter(|s| s.layer == layer)
                .collect();
            if entries.is_empty() { return 0.0; }
            let top1_counts: std::collections::HashMap<u32, usize> = entries.iter()
                .filter_map(|e| e.expert_ids.first().copied())
                .fold(std::collections::HashMap::new(), |mut acc, eid| {
                    *acc.entry(eid).or_insert(0) += 1;
                    acc
                });
            let total = entries.len() as f64;
            top1_counts.values()
                .map(|&c| { let p = c as f64 / total; if p > 0.0 { -p * p.log2() } else { 0.0 } })
                .sum()
        }).collect();
        let mean_entropy: f64 = all_entropies.iter().sum::<f64>() / all_entropies.len().max(1) as f64;
        let any_nonzero = all_entropies.iter().any(|&e| e > 0.001);

        // Compute average weight spread across all entries.
        let all_spreads: Vec<f32> = log.iter().map(|e| e.weight_spread).collect();
        let mean_spread: f32 = all_spreads.iter().sum::<f32>() / all_spreads.len().max(1) as f32;

        if !any_nonzero {
            out.push_str("\n  FINDING: All layers have entropy=0 (always same expert).\n");
            out.push_str("  This indicates degenerate routing. Possible causes:\n");
            out.push_str("    - Router weights are near-zero or collapsed after quantization\n");
            out.push_str("    - Bug in router dispatch (hidden state, weight offsets)\n");
            out.push_str("    - Model genuinely converged to single-expert routing\n");

            // Weight spread diagnosis.
            out.push_str(&format!(
                "\n  Weight spread analysis: mean_spread={:.6}\n", mean_spread
            ));
            if mean_spread < 0.01 {
                out.push_str("  DIAGNOSIS: Near-zero weight spread confirms softmax output is near-uniform.\n");
                out.push_str("  The router cannot distinguish between experts. When all softmax\n");
                out.push_str("  probabilities are ~1/num_experts, the strict `>` argmax tiebreaker\n");
                out.push_str("  always selects expert 0 (lowest index wins ties).\n");
                out.push_str("  ROOT CAUSE: Router weight matrix has insufficient variance.\n");
                out.push_str("  This is a model quality issue (Q4_0 quantization or undertrained router),\n");
                out.push_str("  not a Lumen code bug. The kernel is analytically correct.\n");
            }
        } else {
            out.push_str(&format!(
                "\n  Mean entropy across layers: {:.3} (mean_spread={:.4})\n", mean_entropy, mean_spread
            ));
            out.push_str("  Routing appears diverse (non-degenerate).\n");
        }

        Some(out)
    }

    /// Check if profiling phase has ended and trigger cache warmup.
    ///
    /// Called after each token's expert activation recording. Decrements the
    /// profiling counter and, when it reaches zero, uses the profiler's
    /// accumulated activation data to pre-populate the expert cache with
    /// hot experts loaded via ExpertReader.
    ///
    /// This method requires `&mut self` because it mutates the profiling
    /// counter and warmup flag. Callers must NOT hold the scratch lock
    /// when calling this (since it acquires profiler and cache locks).
    pub(crate) fn maybe_trigger_warmup(&self) {
        if self.warmup_complete.load(Ordering::Relaxed) {
            return;
        }

        let remaining = self.profiling_tokens_remaining.load(Ordering::Relaxed);
        if remaining == 0 {
            return; // Not configured for warmup
        }

        let new_remaining = self.profiling_tokens_remaining.fetch_sub(1, Ordering::Relaxed) - 1;
        if new_remaining > 0 {
            return;
        }

        // Profiling phase complete. Trigger warmup.
        let top_k = self.profiling_top_k;
        if top_k == 0 {
            self.warmup_complete.store(true, Ordering::Relaxed);
            return;
        }

        let hot_experts = match self.expert_profiler {
            Some(ref profiler) => profiler.lock().unwrap().top_k_per_layer(top_k),
            None => {
                self.warmup_complete.store(true, Ordering::Relaxed);
                return;
            }
        };

        // Check if both cache and reader are available.
        let has_cache = self.expert_cache.is_some();
        let has_reader = self.expert_reader.is_some();

        if !has_cache || !has_reader {
            println!(
                "MoE warmup: profiling complete but cache={} reader={}. Skipping warmup.",
                if has_cache { "yes" } else { "no" },
                if has_reader { "yes" } else { "no" },
            );
            self.warmup_complete.store(true, Ordering::Relaxed);
            return;
        }

        // Collect the (layer, expert_id) pairs for warm-up.
        let mut warm_requests: Vec<(usize, u32)> = Vec::new();
        for (layer, experts) in hot_experts.iter().enumerate() {
            for &eid in experts {
                warm_requests.push((layer, eid));
            }
        }

        if warm_requests.is_empty() {
            println!("MoE warmup: no hot experts found during profiling. Skipping.");
            self.warmup_complete.store(true, Ordering::Relaxed);
            return;
        }

        // Load hot experts via ExpertReader and insert into cache.
        let mut loaded = 0usize;
        let mut failed = 0usize;
        {
            let mut reader = self.expert_reader.as_ref().unwrap().lock().unwrap();
            let mut cache = self.expert_cache.as_ref().unwrap().lock().unwrap();
            for &(layer, eid) in &warm_requests {
                let key = (layer, eid);
                if cache.contains(&key) {
                    continue; // Already cached from streaming path
                }
                match reader.load_expert(layer, eid) {
                    Ok((data, slices)) => {
                        cache.insert(key, data, slices);
                        loaded += 1;
                    }
                    Err(e) => {
                        eprintln!(
                            "MoE warmup: failed to load layer={} expert={}: {}",
                            layer, eid, e
                        );
                        failed += 1;
                    }
                }
            }
        }

        let cache_stats = self.expert_cache.as_ref().unwrap().lock().unwrap().stats();
        println!(
            "MoE warmup complete: loaded {} experts ({} failed), \
             cache has {}/{} experts ({:.1} MB)",
            loaded,
            failed,
            cache_stats.cached_experts,
            cache_stats.capacity,
            cache_stats.cached_bytes as f64 / (1024.0 * 1024.0),
        );

        self.warmup_complete.store(true, Ordering::Relaxed);
    }

    /// Set global tensors (embedding, final_norm, output_proj).
    /// These are uploaded to GPU buffers on the next `init()` call.

    // ========================================================================
    // MoE (Mixture of Experts) FFN dispatch
    // ========================================================================

    /// Encode MoE FFN block for decode (single token) into an existing command buffer.
    ///
    /// Flow:
    /// 1. Router: matmul(normed_x, router_weight) -> logits -> softmax + top-K selection
    /// 2. For each expert: gate+up+SwiGLU+down FFN
    ///    - Option A (cpu_expert_ids provided): only dispatch top-K selected experts
    ///    - Option B (cpu_expert_ids is None): dispatch ALL num_experts
    /// 3. Accumulate: weighted sum of expert outputs + residual -> x_buf
    ///
    /// Option A: When `cpu_expert_ids` is provided (from a prior synchronous
    /// router readback in streaming mode), only the top-K selected experts are dispatched.
    /// Expert outputs are packed densely into slots 0..top_k in expert_output_buf, and the
    /// `moe_expert_accum_option_a` kernel accumulates without expert_ids indexing. This
    /// eliminates (num_experts - top_k) / num_experts of expert FFN compute (75% for
    /// Mixtral 8x7B with top-2).
    ///
    /// Option B (default): Dispatches ALL experts and relies on zero routing weights
    /// for non-selected experts. Correctness-first path.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_moe_ffn_decode(
        cmd: &MetalCommandBuffer,
        pipelines: &MetalPipelines,
        s: &MetalScratch,
        layer_buf: &MetalBuffer,
        moe_meta: &CachedMoeMeta,
        expert_ids_override: Option<&MetalBuffer>,
        expert_weight_buf: Option<&MetalBuffer>,
        is_cached_buf: Option<&MetalBuffer>,
        cache_bias_lambda: f32,
        cpu_expert_ids: Option<&[u32]>,
        skip_router: bool,
        routing_weights_override: Option<&MetalBuffer>,
    ) -> Result<(), RuntimeError> {
        let hidden_dim = s.hidden_dim;
        let inter_dim = s.moe_expert_inter_dim;
        let num_experts = s.moe_num_experts;
        let top_k = s.moe_num_active_experts;

        let expert_ids_buf = match expert_ids_override {
            Some(buf) => buf,
            None => s.moe_expert_ids.as_ref().ok_or_else(|| {
                RuntimeError::Compute("MoE expert_ids buffer not allocated".into())
            })?,
        };
        // When routing_weights_override is provided, use it instead of
        // the shared expert_weights buffer. The per-layer buffer persists across
        // layers in the CB, enabling post-commit readback for diagnostics.
        let expert_weights_buf = match routing_weights_override {
            Some(buf) => buf,
            None => s.moe_expert_weights.as_ref().ok_or_else(|| {
                RuntimeError::Compute("MoE expert_weights buffer not allocated".into())
            })?,
        };
        let expert_output_buf = s.moe_expert_output.as_ref().ok_or_else(|| {
            RuntimeError::Compute("MoE expert_output buffer not allocated".into())
        })?;

        // ---- Step 1: Router dispatch ----
        // Skip when the router was already encoded in a separate CB (Option A two-phase).
        if !skip_router {
            // Select biased or standard router pipeline.
            let use_biased = cache_bias_lambda > 0.0 && is_cached_buf.is_some();
            let router_softmax = if use_biased {
                pipelines.moe_router_softmax_biased.as_ref().ok_or_else(|| {
                    RuntimeError::Compute(
                        "MoE router_softmax_biased pipeline not compiled. \
                         Ensure biased kernel is in METAL_SHADER_SOURCE.".into()
                    )
                })?
            } else {
                pipelines.moe_router_softmax.as_ref().ok_or_else(|| {
                    RuntimeError::Compute(
                        "MoE router_softmax pipeline not compiled. \
                         Ensure MoE shader kernels are in METAL_SHADER_SOURCE.".into()
                    )
                })?
            };

            // router_logits_buf: validated for kernel needs.
            let _router_logits_buf = s.moe_router_logits.as_ref().ok_or_else(|| {
                RuntimeError::Compute("MoE router_logits buffer not allocated".into())
            })?;

            // normed_buf contains the FFN-normed hidden state (set by caller).
            // Compute router_logits = normed_buf * router_weight^T, then softmax + top-K.
            // When cache bias is active, the biased kernel adds lambda * is_cached[e]
            // to each expert's logit before softmax, nudging selection toward cached experts.
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for MoE router".into())
            })?;
            enc.set_pipeline_state(router_softmax);
            enc.set_buffer(&s.normed_buf, 0, 0);                           // hidden state [hidden_dim] float
            enc.set_buffer(layer_buf, moe_meta.router_weight_off, 1);       // router weight [num_experts * hidden_dim]
            enc.set_buffer(expert_ids_buf, 0, 2);                           // output: expert_ids [top_k] u32
            enc.set_buffer(expert_weights_buf, 0, 3);                       // output: expert_weights [top_k] f32
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
            enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
            // Biased kernel: pass is_cached buffer and lambda at buffer(7) and buffer(8).
            if use_biased {
                enc.set_buffer(is_cached_buf.unwrap(), 0, 7);               // is_cached [num_experts] u8
                enc.set_bytes(&cache_bias_lambda.to_le_bytes(), 8);         // cache_bias_lambda f32
            }
            // Single threadgroup processes all experts for one token
            let tg = 256u64.min(hidden_dim as u64).max(1);
            enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
        }

        // ---- Step 2: Expert FFN dispatch ----
        // Option A: When cpu_expert_ids is provided, iterate only top-K selected
        // experts. Output slot k (byte offset k * hidden_dim * 4) contains expert_ids[k]'s
        // output, packed densely. This skips (num_experts - top_k) expert FFNs entirely.
        //
        // Option B (default): Iterate all num_experts. Output slot e (byte offset
        // e * hidden_dim * 4) contains expert e's output, sparse layout.
        let ewb = expert_weight_buf.unwrap_or(layer_buf);

        // Determine whether the offset arrays are dense (slot-indexed, len == top_k)
        // or sparse (expert-indexed, len == num_experts).
        //
        // - Streaming Option A with assembled buffer: offsets are dense (8 entries for top-K).
        //   Use `output_slot` (0..top_k-1) to index offsets.
        // - GPU-resident Option A: offsets are sparse (256 entries for all experts).
        //   Use `expert_idx` (raw expert ID) to index offsets.
        // - Option B (no cpu_expert_ids): offsets are sparse. Use `expert_idx`.
        let dense_offsets = moe_meta.expert_gate_offs.len() == top_k && top_k < num_experts;

        let expert_loop: Vec<(usize, usize)> = if let Some(ids) = cpu_expert_ids {
            // Option A: only dispatch top-K selected experts.
            // ids[k] = expert index, output slot = k (dense).
            ids.iter().enumerate().map(|(slot, &eid)| (eid as usize, slot)).collect()
        } else {
            // Option B: dispatch all experts, output slot = expert_idx (sparse).
            (0..num_experts).map(|e| (e, e)).collect()
        };

        for &(expert_idx, output_slot) in &expert_loop {
            // When offsets are dense (streaming Option A assembled buffer), index by
            // output_slot (0..top_k-1). When sparse (GPU-resident or Option B), index
            // by expert_idx (raw expert ID, 0..num_experts-1).
            let off_idx = if dense_offsets { output_slot } else { expert_idx };
            let gate_off = moe_meta.expert_gate_offs[off_idx];
            let up_off = moe_meta.expert_up_offs[off_idx];
            let down_off = moe_meta.expert_down_offs[off_idx];
            let expert_out_byte_off = (output_slot * hidden_dim * 4) as u64;

            // Fused Gate + Up + SwiGLU for this expert (Q8_0/Q4_0/Q4_1)
            // or separate Gate + Up + SwiGLU (F32/other quant fallback).
            match moe_meta.expert_gate_quant {
                QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::Q4_1 => {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder for MoE expert FFN".into())
                    })?;
                    let (ffn_tg, ffn_n_tg) = match moe_meta.expert_gate_quant {
                        QuantScheme::Q8_0 => {
                            enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q8_0_2sg);
                            (64u64, ((inter_dim as u64) + 7) / 8)
                        }
                        QuantScheme::Q4_0 => {
                            enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q4_0_deferred);
                            (128u64, inter_dim as u64)
                        }
                        QuantScheme::Q4_1 => {
                            enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q4_1_deferred);
                            (128u64, inter_dim as u64)
                        }
                        _ => unreachable!(),
                    };
                    enc.set_buffer(ewb, gate_off, 0);                // gate weights
                    enc.set_buffer(&s.normed_buf, 0, 1);            // normed input x
                    enc.set_buffer(&s.gate_buf, 0, 2);              // output (SwiGLU result -> gate_buf as temp)
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                    enc.set_buffer(ewb, up_off, 5);                  // up weights
                    enc.dispatch_threadgroups(
                        MTLSize::new(ffn_n_tg, 1, 1),
                        MTLSize::new(ffn_tg, 1, 1),
                    );
                    enc.end_encoding();
                }
                _ => {
                    // F16/F32/other quant fallback: separate gate + up matvecs, then SwiGLU.
                    // Gate: gate_buf = W_gate * normed_buf
                    // Up:   up_buf   = W_up   * normed_buf
                    let (fb_pso, fb_tg, fb_n_tg) = if moe_meta.expert_gate_quant == QuantScheme::F16 {
                        (&pipelines.matmul_f16_deferred_nr2, 128u64, ((inter_dim as u64) + 1) / 2)
                    } else {
                        (&pipelines.matmul_bytes_f32, s.matmul_tg_size, inter_dim as u64)
                    };
                    {
                        let enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for MoE expert gate".into())
                        })?;
                        enc.set_pipeline_state(fb_pso);
                        enc.set_buffer(ewb, gate_off, 0);
                        enc.set_buffer(&s.normed_buf, 0, 1);
                        enc.set_buffer(&s.gate_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        if moe_meta.expert_gate_quant == QuantScheme::F16 {
                            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                        }
                        enc.dispatch_threadgroups(
                            MTLSize::new(fb_n_tg, 1, 1),
                            MTLSize::new(fb_tg, 1, 1),
                        );
                        enc.end_encoding();
                    }
                    {
                        let enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for MoE expert up".into())
                        })?;
                        enc.set_pipeline_state(fb_pso);
                        enc.set_buffer(ewb, up_off, 0);
                        enc.set_buffer(&s.normed_buf, 0, 1);
                        enc.set_buffer(&s.up_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        if moe_meta.expert_gate_quant == QuantScheme::F16 {
                            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                        }
                        enc.dispatch_threadgroups(
                            MTLSize::new(fb_n_tg, 1, 1),
                            MTLSize::new(fb_tg, 1, 1),
                        );
                        enc.end_encoding();
                    }
                    // SwiGLU: gate_buf = swiglu(gate_buf, up_buf)
                    {
                        let enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for MoE expert swiglu".into())
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
            }

            // Down projection: expert_output[slot] = W_down * gate_buf
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for MoE expert down proj".into())
                })?;
                match moe_meta.expert_down_quant {
                    QuantScheme::Q8_0 => {
                        enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_2sg);
                        enc.set_buffer(ewb, down_off, 0);
                        enc.set_buffer(&s.gate_buf, 0, 1);
                        enc.set_buffer(expert_output_buf, expert_out_byte_off, 2);
                        enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                        let n_tg = ((hidden_dim as u64) + 7) / 8;
                        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(64, 1, 1));
                    }
                    QuantScheme::Q4_0 => {
                        enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2);
                        enc.set_buffer(ewb, down_off, 0);
                        enc.set_buffer(&s.gate_buf, 0, 1);
                        enc.set_buffer(expert_output_buf, expert_out_byte_off, 2);
                        enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                        let n_tg = ((hidden_dim as u64) + 1) / 2;
                        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                    }
                    QuantScheme::Q4_1 => {
                        // Q4_1 expert down projection (decode matvec)
                        enc.set_pipeline_state(&pipelines.dequant_matmul_q4_1_deferred);
                        enc.set_buffer(ewb, down_off, 0);
                        enc.set_buffer(&s.gate_buf, 0, 1);
                        enc.set_buffer(expert_output_buf, expert_out_byte_off, 2);
                        enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                        let n_tg = ((hidden_dim as u64) + 3) / 4;
                        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                    }
                    QuantScheme::F16 => {
                        enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                        enc.set_buffer(ewb, down_off, 0);
                        enc.set_buffer(&s.gate_buf, 0, 1);
                        enc.set_buffer(expert_output_buf, expert_out_byte_off, 2);
                        enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                        let n_tg = ((hidden_dim as u64) + 1) / 2;
                        enc.dispatch_threadgroups(MTLSize::new(n_tg, 1, 1), MTLSize::new(128, 1, 1));
                    }
                    _ => {
                        // F32 fallback: matmul_bytes_f32 (cast uchar* to float*)
                        enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                        enc.set_buffer(ewb, down_off, 0);
                        enc.set_buffer(&s.gate_buf, 0, 1);
                        enc.set_buffer(expert_output_buf, expert_out_byte_off, 2);
                        enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 3);
                        enc.dispatch_threadgroups(
                            MTLSize::new(hidden_dim as u64, 1, 1),
                            MTLSize::new(s.matmul_tg_size, 1, 1),
                        );
                    }
                }
                enc.end_encoding();
            }
        }

        // ---- Step 3: Weighted accumulation + residual ----
        // Option A uses moe_expert_accum_option_a with dense [top_k, hidden_dim]
        // layout (no expert_ids indexing needed). Option B uses original moe_expert_accum
        // with sparse [num_experts, hidden_dim] layout + expert_ids indirection.
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for MoE accum".into())
            })?;

            if cpu_expert_ids.is_some() {
                // Option A: dense layout, no expert_ids needed in kernel.
                let expert_accum_a = pipelines.moe_expert_accum_option_a.as_ref().ok_or_else(|| {
                    RuntimeError::Compute(
                        "MoE expert_accum_option_a pipeline not compiled. \
                         Ensure MoE shader kernels are in METAL_SHADER_SOURCE.".into()
                    )
                })?;
                enc.set_pipeline_state(expert_accum_a);
                enc.set_buffer(expert_output_buf, 0, 0);         // expert_outputs [top_k * hidden_dim]
                enc.set_buffer(expert_weights_buf, 0, 1);         // expert_weights [top_k]
                enc.set_buffer(&s.x_buf, 0, 2);                   // output (hidden state)
                enc.set_buffer(&s.attn_proj_buf, 0, 3);            // residual (pre-FFN)
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                enc.set_bytes(&(top_k as u32).to_le_bytes(), 5);
            } else {
                // Option B: sparse layout with expert_ids indirection.
                let expert_accum = pipelines.moe_expert_accum.as_ref().ok_or_else(|| {
                    RuntimeError::Compute(
                        "MoE expert_accum pipeline not compiled. \
                         Ensure MoE shader kernels are in METAL_SHADER_SOURCE.".into()
                    )
                })?;
                enc.set_pipeline_state(expert_accum);
                enc.set_buffer(expert_output_buf, 0, 0);         // expert_outputs [num_experts * hidden_dim]
                enc.set_buffer(expert_weights_buf, 0, 1);         // expert_weights [top_k]
                enc.set_buffer(expert_ids_buf, 0, 2);             // expert_ids [top_k] u32
                enc.set_buffer(&s.x_buf, 0, 3);                   // output (hidden state)
                enc.set_buffer(&s.attn_proj_buf, 0, 4);            // residual (pre-FFN)
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
                enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
            }

            let tg_count = ((hidden_dim as u64) + 255) / 256;
            enc.dispatch_threadgroups(
                MTLSize::new(tg_count, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }

        Ok(())
    }

    /// Encode shared (always-active) expert FFN for decode.
    ///
    /// Qwen3.5-MoE has a shared expert that runs on every token in addition to
    /// the top-K routed experts. Its output is added to x_buf via residual.
    ///
    /// Structure: Gate+Up+SwiGLU -> Down -> add to x_buf
    pub(crate) fn encode_shared_expert_ffn_decode(
        cmd: &MetalCommandBuffer,
        pipelines: &MetalPipelines,
        s: &MetalScratch,
        layer_buf: &MetalBuffer,
        meta: &CachedLayerMeta,
    ) -> Result<(), RuntimeError> {
        let gate_off = meta.shared_expert_gate_off.ok_or_else(|| {
            RuntimeError::Compute("shared_expert_gate_off missing".into())
        })?;
        let up_off = meta.shared_expert_up_off.ok_or_else(|| {
            RuntimeError::Compute("shared_expert_up_off missing".into())
        })?;
        let down_off = meta.shared_expert_down_off.ok_or_else(|| {
            RuntimeError::Compute("shared_expert_down_off missing".into())
        })?;
        let gate_quant = meta.shared_expert_gate_quant.unwrap_or(QuantScheme::F32);
        let down_quant = meta.shared_expert_down_quant.unwrap_or(QuantScheme::F32);
        Self::encode_shared_expert_ffn_decode_raw(
            cmd, pipelines, s, layer_buf,
            gate_off, up_off, down_off, gate_quant, down_quant,
            meta.ffn_gate_inp_shexp_off,
        )
    }

    /// Encode shared expert FFN using raw byte offsets.
    ///
    /// Factored out of `encode_shared_expert_ffn_decode` so both the
    /// GPU-resident path (which has `CachedLayerMeta`) and the streaming
    /// Option A path (which has `SubtensorOffsets`) can dispatch the
    /// always-active shared expert.
    pub(crate) fn encode_shared_expert_ffn_decode_raw(
        cmd: &MetalCommandBuffer,
        pipelines: &MetalPipelines,
        s: &MetalScratch,
        layer_buf: &MetalBuffer,
        gate_off: u64,
        up_off: u64,
        down_off: u64,
        gate_quant: QuantScheme,
        down_quant: QuantScheme,
        gate_inp_shexp_off: Option<u64>,
    ) -> Result<(), RuntimeError> {
        let hidden_dim = s.hidden_dim;
        let se_inter = s.shared_expert_inter_dim;

        let se_gate_buf = s.shared_expert_gate_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("shared_expert_gate_buf not allocated".into())
        })?;
        let se_down_buf = s.shared_expert_down_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("shared_expert_down_buf not allocated".into())
        })?;

        // Step 1: Fused Gate+Up+SwiGLU (input = normed_buf, output = se_gate_buf)
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for shared expert gate+up".into())
            })?;
            let (se_ffn_tg, se_ffn_n_tg) = match gate_quant {
                QuantScheme::Q8_0 => {
                    enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q8_0_2sg);
                    (64u64, ((se_inter as u64) + 7) / 8)
                }
                QuantScheme::Q4_0 => {
                    enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q4_0_deferred);
                    (128u64, se_inter as u64)
                }
                QuantScheme::Q4_1 => {
                    enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q4_1_deferred);
                    (128u64, se_inter as u64)
                }
                _ => {
                    // F16/F32 fallback: separate gate, up, then swiglu
                    let (fb_pso, fb_tg, fb_n_tg) = if gate_quant == QuantScheme::F16 {
                        (&pipelines.matmul_f16_deferred_nr2, 128u64, ((se_inter as u64) + 1) / 2)
                    } else {
                        (&pipelines.matmul_bytes_f32, s.matmul_tg_size, se_inter as u64)
                    };
                    enc.set_pipeline_state(fb_pso);
                    enc.set_buffer(layer_buf, gate_off, 0);
                    enc.set_buffer(&s.normed_buf, 0, 1);
                    enc.set_buffer(se_gate_buf, 0, 2);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                    if gate_quant == QuantScheme::F16 {
                        enc.set_bytes(&(se_inter as u32).to_le_bytes(), 4);
                    }
                    enc.dispatch_threadgroups(
                        MTLSize::new(fb_n_tg, 1, 1),
                        MTLSize::new(fb_tg, 1, 1),
                    );
                    // Up into down_buf as temporary
                    enc.set_buffer(layer_buf, up_off, 0);
                    enc.set_buffer(se_down_buf, 0, 2);
                    enc.dispatch_threadgroups(
                        MTLSize::new(fb_n_tg, 1, 1),
                        MTLSize::new(fb_tg, 1, 1),
                    );
                    enc.end_encoding();
                    // SwiGLU: se_gate_buf = silu(se_gate_buf) * se_down_buf
                    let enc2 = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder for shared expert swiglu".into())
                    })?;
                    enc2.set_pipeline_state(&pipelines.swiglu);
                    enc2.set_buffer(se_gate_buf, 0, 0);
                    enc2.set_buffer(se_down_buf, 0, 1);
                    enc2.set_bytes(&(se_inter as u32).to_le_bytes(), 2);
                    let tg = 256u64.min(se_inter as u64).max(1);
                    enc2.dispatch_threadgroups(
                        MTLSize::new((se_inter as u64).div_ceil(tg), 1, 1),
                        MTLSize::new(tg, 1, 1),
                    );
                    enc2.end_encoding();
                    // Skip the fused path below since we handled F32 inline.
                    // Fall through to Step 2 (down projection).
                    // Step 2: Down projection: se_down_buf = W_down * se_gate_buf
                    {
                        let enc3 = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for shared expert down".into())
                        })?;
                        let (d_pso, d_tg, d_n_tg) = if down_quant == QuantScheme::F16 {
                            (&pipelines.matmul_f16_deferred_nr2, 128u64, ((hidden_dim as u64) + 1) / 2)
                        } else {
                            (&pipelines.matmul_bytes_f32, s.matmul_tg_size, hidden_dim as u64)
                        };
                        enc3.set_pipeline_state(d_pso);
                        enc3.set_buffer(layer_buf, down_off, 0);
                        enc3.set_buffer(se_gate_buf, 0, 1);
                        enc3.set_buffer(se_down_buf, 0, 2);
                        enc3.set_bytes(&(se_inter as u32).to_le_bytes(), 3);
                        if down_quant == QuantScheme::F16 {
                            enc3.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                        }
                        enc3.dispatch_threadgroups(
                            MTLSize::new(d_n_tg, 1, 1),
                            MTLSize::new(d_tg, 1, 1),
                        );
                        enc3.end_encoding();
                    }
                    // Step 2.5: Shared expert gating (if weight present).
                    // se_down_buf *= sigmoid(dot(ffn_gate_inp_shexp, normed_buf))
                    if let Some(gis_off) = gate_inp_shexp_off {
                        // Dot product: [1 x hidden_dim] @ normed_buf -> se_gate_buf[0]
                        let enc_dot = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for shared expert gate dot".into())
                        })?;
                        enc_dot.set_pipeline_state(&pipelines.matmul_bytes_f32);
                        enc_dot.set_buffer(layer_buf, gis_off, 0);
                        enc_dot.set_buffer(&s.normed_buf, 0, 1);
                        enc_dot.set_buffer(se_gate_buf, 0, 2);
                        enc_dot.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        enc_dot.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(s.matmul_tg_size, 1, 1));
                        enc_dot.end_encoding();

                        // Fused sigmoid-scale + residual add in one encoder:
                        //   x_buf[i] += sigmoid(se_gate_buf[0]) * se_down_buf[i]
                        let enc_fused = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for shared expert sigmoid+add".into())
                        })?;
                        if let Some(ref pso_fused) = pipelines.sigmoid_scale_add {
                            enc_fused.set_pipeline_state(pso_fused);
                            enc_fused.set_buffer(se_gate_buf, 0, 0);
                            enc_fused.set_buffer(se_down_buf, 0, 1);
                            enc_fused.set_buffer(&s.x_buf, 0, 2);
                            enc_fused.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                            let tg = 256u64.min(hidden_dim as u64).max(1);
                            enc_fused.dispatch_threadgroups(
                                MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                                MTLSize::new(tg, 1, 1),
                            );
                        } else {
                            // Fallback: sigmoid_scale_buffer (modifies se_down_buf in place)
                            let pso_sig = pipelines.sigmoid_scale_buffer.as_ref().ok_or_else(|| {
                                RuntimeError::Compute("sigmoid_scale_buffer pipeline not compiled".into())
                            })?;
                            enc_fused.set_pipeline_state(pso_sig);
                            enc_fused.set_buffer(se_gate_buf, 0, 0);
                            enc_fused.set_buffer(se_down_buf, 0, 1);
                            enc_fused.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
                            let tg_sig = 256u64.min(hidden_dim as u64).max(1);
                            enc_fused.dispatch_threadgroups(
                                MTLSize::new((hidden_dim as u64).div_ceil(tg_sig), 1, 1),
                                MTLSize::new(tg_sig, 1, 1),
                            );
                        }
                        enc_fused.end_encoding();
                    }
                    if pipelines.sigmoid_scale_add.is_none() || gate_inp_shexp_off.is_none() {
                        // Residual add needed: either no gating, or fallback path
                        // that modified se_down_buf in-place.
                        let enc4 = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder for shared expert residual".into())
                        })?;
                        enc4.set_pipeline_state(&pipelines.add_residual);
                        enc4.set_buffer(&s.x_buf, 0, 0);
                        enc4.set_buffer(se_down_buf, 0, 1);
                        enc4.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
                        let tg = 256u64.min(hidden_dim as u64).max(1);
                        enc4.dispatch_threadgroups(
                            MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                            MTLSize::new(tg, 1, 1),
                        );
                        enc4.end_encoding();
                    }
                    return Ok(());
                }
            };
            // Fused Q8_0/Q4_0/Q4_1 path: gate+up+swiglu in one kernel
            enc.set_buffer(layer_buf, gate_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(se_gate_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(se_inter as u32).to_le_bytes(), 4);
            enc.set_buffer(layer_buf, up_off, 5);
            enc.dispatch_threadgroups(
                MTLSize::new(se_ffn_n_tg, 1, 1),
                MTLSize::new(se_ffn_tg, 1, 1),
            );
            enc.end_encoding();
        }

        // Step 2: Down projection: se_down_buf = W_down * se_gate_buf
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for shared expert down".into())
            })?;
            let tg_down = match down_quant {
                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_2sg); 64u64 },
                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                QuantScheme::Q4_1 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_1_deferred); 128u64 },
                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); s.matmul_tg_size },
            };
            enc.set_buffer(layer_buf, down_off, 0);
            enc.set_buffer(se_gate_buf, 0, 1);
            enc.set_buffer(se_down_buf, 0, 2);
            enc.set_bytes(&(se_inter as u32).to_le_bytes(), 3);
            if matches!(down_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::Q4_1 | QuantScheme::F16) {
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
            }
            let n_tg_down = match down_quant { QuantScheme::Q8_0 => ((hidden_dim as u64) + 7) / 8, QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::Q4_1 => ((hidden_dim as u64) + 3) / 4, QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2, _ => hidden_dim as u64 };
            enc.dispatch_threadgroups(MTLSize::new(n_tg_down, 1, 1), MTLSize::new(tg_down, 1, 1));
            enc.end_encoding();
        }

        // Step 2.5: Shared expert gating (if weight present) + residual add.
        if let Some(gis_off) = gate_inp_shexp_off {
            // Dot product: [1 x hidden_dim] @ normed_buf -> se_gate_buf[0]
            let enc_dot = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for shared expert gate dot".into())
            })?;
            enc_dot.set_pipeline_state(&pipelines.matmul_bytes_f32);
            enc_dot.set_buffer(layer_buf, gis_off, 0);
            enc_dot.set_buffer(&s.normed_buf, 0, 1);
            enc_dot.set_buffer(se_gate_buf, 0, 2);
            enc_dot.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc_dot.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(s.matmul_tg_size, 1, 1));
            enc_dot.end_encoding();

            // Fused sigmoid-scale + residual add:
            //   x_buf[i] += sigmoid(se_gate_buf[0]) * se_down_buf[i]
            let enc_fused = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for shared expert sigmoid+add".into())
            })?;
            if let Some(ref pso_fused) = pipelines.sigmoid_scale_add {
                enc_fused.set_pipeline_state(pso_fused);
                enc_fused.set_buffer(se_gate_buf, 0, 0);
                enc_fused.set_buffer(se_down_buf, 0, 1);
                enc_fused.set_buffer(&s.x_buf, 0, 2);
                enc_fused.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                let tg = 256u64.min(hidden_dim as u64).max(1);
                enc_fused.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
            } else {
                // Fallback: sigmoid_scale_buffer (modifies se_down_buf in place)
                let pso_sig = pipelines.sigmoid_scale_buffer.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("sigmoid_scale_buffer pipeline not compiled".into())
                })?;
                enc_fused.set_pipeline_state(pso_sig);
                enc_fused.set_buffer(se_gate_buf, 0, 0);
                enc_fused.set_buffer(se_down_buf, 0, 1);
                enc_fused.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
                let tg_sig = 256u64.min(hidden_dim as u64).max(1);
                enc_fused.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(tg_sig), 1, 1),
                    MTLSize::new(tg_sig, 1, 1),
                );
            }
            enc_fused.end_encoding();
        }

        // Step 3: Add shared expert output to x_buf (only when no gating or fallback path)
        if pipelines.sigmoid_scale_add.is_none() || gate_inp_shexp_off.is_none() {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for shared expert residual".into())
            })?;
            enc.set_pipeline_state(&pipelines.add_residual);
            enc.set_buffer(&s.x_buf, 0, 0);
            enc.set_buffer(se_down_buf, 0, 1);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
            let tg = 256u64.min(hidden_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                MTLSize::new(tg, 1, 1),
            );
            enc.end_encoding();
        }

        Ok(())
    }




    /// Fused single-encoder variant of encode_moe_ffn_decode_batched.
    pub(crate) fn encode_moe_ffn_decode_batched_fused(
        enc: &MetalComputeEncoder,
        pipelines: &MetalPipelines,
        s: &MetalScratch,
        layer_buf: &MetalBuffer,
        moe_meta: &CachedMoeMeta,
        expert_ids_buf: &MetalBuffer,
        expert_weights_buf: &MetalBuffer,
        gate_up_offsets_buf: &MetalBuffer,
        down_offsets_buf: &MetalBuffer,
    ) -> Result<(), RuntimeError> {
        let hidden_dim = s.hidden_dim;
        let inter_dim = s.moe_expert_inter_dim;
        let top_k = s.moe_num_active_experts;

        let swiglu_buf = s.moe_batched_swiglu_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("moe_batched_swiglu_buf not allocated".into())
        })?;

        // Phase 1: Batched gate+up+swiglu (select kernel based on expert_gate_quant)
        {
            let pipeline = match moe_meta.expert_gate_quant {
                QuantScheme::Q8_0 => pipelines.moe_batched_gate_up_swiglu_q8_0.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("moe_batched_gate_up_swiglu_q8_0 pipeline not compiled".into())
                })?,
                _ => pipelines.moe_batched_gate_up_swiglu_q4_0.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("moe_batched_gate_up_swiglu_q4_0 pipeline not compiled".into())
                })?,
            };
            enc.set_pipeline_state(pipeline);
            enc.set_buffer(layer_buf, 0, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(swiglu_buf, 0, 2);
            enc.set_buffer(expert_ids_buf, 0, 3);
            enc.set_buffer(gate_up_offsets_buf, 0, 4);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 6);
            enc.set_bytes(&(top_k as u32).to_le_bytes(), 7);
            let n_tg = (top_k * inter_dim) as u64;
            enc.dispatch_threadgroups(
                MTLSize::new(n_tg, 1, 1),
                MTLSize::new(128, 1, 1),
            );
        }

        // Barrier: Phase 1 writes swiglu_buf, Phase 2 reads it
        enc.memory_barrier_with_scope(1);

        // Phase 2: Batched down+accum (select kernel based on expert_down_quant)
        {
            let pipeline = match moe_meta.expert_down_quant {
                QuantScheme::Q4_1 => pipelines.moe_batched_down_accum_q4_1.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("moe_batched_down_accum_q4_1 pipeline not compiled".into())
                })?,
                QuantScheme::Q8_0 => pipelines.moe_batched_down_accum_q8_0.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("moe_batched_down_accum_q8_0 pipeline not compiled".into())
                })?,
                _ => pipelines.moe_batched_down_accum_q4_0.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("moe_batched_down_accum_q4_0 pipeline not compiled".into())
                })?,
            };
            enc.set_pipeline_state(pipeline);
            enc.set_buffer(layer_buf, 0, 0);
            enc.set_buffer(swiglu_buf, 0, 1);
            enc.set_buffer(&s.x_buf, 0, 2);
            enc.set_buffer(&s.attn_proj_buf, 0, 3);
            enc.set_buffer(expert_ids_buf, 0, 4);
            enc.set_buffer(expert_weights_buf, 0, 5);
            enc.set_buffer(down_offsets_buf, 0, 6);
            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 7);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 8);
            enc.set_bytes(&(top_k as u32).to_le_bytes(), 9);
            // Q8_0 down uses deferred pattern with 4 rows/TG just like Q4_0
            let n_tg = ((hidden_dim as u64) + 3) / 4;
            enc.dispatch_threadgroups(
                MTLSize::new(n_tg, 1, 1),
                MTLSize::new(128, 1, 1),
            );
        }

        Ok(())
    }

    /// Encode the complete MoE FFN block (routed experts + shared expert) using
    /// the fused down+accum+shared kernel when available. This is the preferred
    /// dispatch path for MoE decode -- it runs routed phase1, shared expert
    /// gate+up+SwiGLU, and gating dot product in parallel, then combines
    /// everything in a single fused phase2 kernel (4 dispatches + 2 barriers
    /// instead of 8 dispatches).
    ///
    /// Falls back to separate batched + shared expert dispatches when the fused
    /// kernel is not available (e.g., mismatched quant between routed/shared).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_moe_ffn_with_shared_fused(
        enc: &MetalComputeEncoder,
        pipelines: &MetalPipelines,
        s: &MetalScratch,
        layer_buf: &MetalBuffer,
        layer_idx: usize,
        moe_meta: &CachedMoeMeta,
        meta: &CachedLayerMeta,
        expert_ids_buf: &MetalBuffer,
        expert_weights_buf: &MetalBuffer,
        gate_up_off_buf: &MetalBuffer,
        down_off_buf: &MetalBuffer,
    ) -> Result<(), RuntimeError> {
        let hidden_dim = s.hidden_dim;
        let has_shared = meta.shared_expert_gate_off.is_some();

        // Check if we can use the fused down+accum+shared kernel
        let has_fused_shared_kernel = has_shared && {
            let down_quant = moe_meta.expert_down_quant;
            let se_down_quant = meta.shared_expert_down_quant.unwrap_or(QuantScheme::F32);
            let fused_avail = match (down_quant, se_down_quant) {
                (QuantScheme::Q8_0, QuantScheme::Q8_0) => pipelines.moe_batched_down_accum_shared_q8_0.is_some(),
                (QuantScheme::Q8_0, QuantScheme::Q4_0) => pipelines.moe_batched_down_accum_shared_q8_0_se_q4_0.is_some(),
                (QuantScheme::Q4_0, QuantScheme::Q4_0) => pipelines.moe_batched_down_accum_shared_q4_0.is_some(),
                _ => false,
            };
            fused_avail
                && s.moe_shared_down_offsets.get(layer_idx).and_then(|o| o.as_ref()).is_some()
                && s.moe_shared_gate_scalar_buf.is_some()
                && s.shared_expert_gate_buf.is_some()
        };

        if has_fused_shared_kernel {
            // ==== FUSED PATH: Parallel phase1 + shared expert, then fused phase2 ====
            let se_inter = s.shared_expert_inter_dim;
            let gate_off = meta.shared_expert_gate_off.unwrap();
            let up_off = meta.shared_expert_up_off.unwrap();
            let gate_quant = meta.shared_expert_gate_quant.unwrap_or(QuantScheme::F32);
            let se_gate_buf = s.shared_expert_gate_buf.as_ref().unwrap();
            let se_gate_scalar_buf = s.moe_shared_gate_scalar_buf.as_ref().unwrap();
            let se_down_off_buf = s.moe_shared_down_offsets[layer_idx].as_ref().unwrap();
            let swiglu_buf = s.moe_batched_swiglu_buf.as_ref().unwrap();
            let inter_dim = s.moe_expert_inter_dim;
            let top_k = s.moe_num_active_experts;

            // Dispatch 1: Batched gate+up+SwiGLU (routed experts) -- reads normed_buf
            {
                let pipeline = match moe_meta.expert_gate_quant {
                    QuantScheme::Q8_0 => pipelines.moe_batched_gate_up_swiglu_q8_0.as_ref().unwrap(),
                    _ => pipelines.moe_batched_gate_up_swiglu_q4_0.as_ref().unwrap(),
                };
                enc.set_pipeline_state(pipeline);
                enc.set_buffer(layer_buf, 0, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(swiglu_buf, 0, 2);
                enc.set_buffer(expert_ids_buf, 0, 3);
                enc.set_buffer(gate_up_off_buf, 0, 4);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
                enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 6);
                enc.set_bytes(&(top_k as u32).to_le_bytes(), 7);
                let n_tg = (top_k * inter_dim) as u64;
                enc.dispatch_threadgroups(
                    MTLSize::new(n_tg, 1, 1),
                    MTLSize::new(128, 1, 1),
                );
            }

            // Dispatch 2: Shared expert gate+up+SwiGLU -- reads normed_buf (parallel with dispatch 1)
            {
                let (se_ffn_tg, se_ffn_n_tg) = match gate_quant {
                    QuantScheme::Q8_0 => {
                        enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q8_0_2sg);
                        (64u64, ((se_inter as u64) + 7) / 8)
                    },
                    QuantScheme::Q4_0 => {
                        enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q4_0_deferred);
                        (128u64, se_inter as u64)
                    },
                    QuantScheme::Q4_1 => {
                        enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q4_1_deferred);
                        (128u64, se_inter as u64)
                    },
                    QuantScheme::F16 => {
                        enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2);
                        (128u64, ((se_inter as u64) + 1) / 2)
                    },
                    _ => {
                        enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                        (s.matmul_tg_size, se_inter as u64)
                    },
                };
                enc.set_buffer(layer_buf, gate_off, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(se_gate_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&(se_inter as u32).to_le_bytes(), 4);
                enc.set_buffer(layer_buf, up_off, 5);
                enc.dispatch_threadgroups(
                    MTLSize::new(se_ffn_n_tg, 1, 1),
                    MTLSize::new(se_ffn_tg, 1, 1),
                );
            }

            // Dispatch 3: Gating dot product (if gate_inp_shexp exists) -- reads normed_buf (parallel)
            let has_gating = meta.ffn_gate_inp_shexp_off.is_some();
            if let Some(gis_off) = meta.ffn_gate_inp_shexp_off {
                enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
                enc.set_buffer(layer_buf, gis_off, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(se_gate_scalar_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(s.matmul_tg_size, 1, 1));
            }

            // Barrier: wait for all 3 parallel dispatches before fused phase 2
            enc.memory_barrier_with_scope(1);

            // Dispatch 4: Fused down+accum+shared
            {
                let down_quant = moe_meta.expert_down_quant;
                let se_down_quant = meta.shared_expert_down_quant.unwrap_or(QuantScheme::F32);
                let pipeline = match (down_quant, se_down_quant) {
                    (QuantScheme::Q8_0, QuantScheme::Q8_0) => pipelines.moe_batched_down_accum_shared_q8_0.as_ref().unwrap(),
                    (QuantScheme::Q8_0, QuantScheme::Q4_0) => pipelines.moe_batched_down_accum_shared_q8_0_se_q4_0.as_ref().unwrap(),
                    _ => pipelines.moe_batched_down_accum_shared_q4_0.as_ref().unwrap(),
                };
                enc.set_pipeline_state(pipeline);
                enc.set_buffer(layer_buf, 0, 0);          // layer_buf
                enc.set_buffer(swiglu_buf, 0, 1);         // swiglu_in
                enc.set_buffer(&s.x_buf, 0, 2);           // output
                enc.set_buffer(&s.attn_proj_buf, 0, 3);   // residual
                enc.set_buffer(expert_ids_buf, 0, 4);     // expert_ids
                enc.set_buffer(expert_weights_buf, 0, 5); // expert_weights
                enc.set_buffer(down_off_buf, 0, 6);       // down_offsets
                enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 7);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 8);
                enc.set_bytes(&(top_k as u32).to_le_bytes(), 9);
                enc.set_buffer(se_gate_buf, 0, 10);       // se_swiglu
                enc.set_buffer(se_down_off_buf, 0, 11);   // se_down_off
                enc.set_buffer(se_gate_scalar_buf, 0, 12); // se_gate_scalar
                enc.set_bytes(&(se_inter as u32).to_le_bytes(), 13);
                let use_sigmoid: u32 = if has_gating { 1 } else { 0 };
                enc.set_bytes(&use_sigmoid.to_le_bytes(), 14);
                let n_tg = ((hidden_dim as u64) + 3) / 4;
                enc.dispatch_threadgroups(
                    MTLSize::new(n_tg, 1, 1),
                    MTLSize::new(128, 1, 1),
                );
            }
        } else {
            // ==== NON-FUSED PATH: Original batched + separate shared expert ====
            Self::encode_moe_ffn_decode_batched_fused(
                enc, pipelines, s, layer_buf, moe_meta,
                expert_ids_buf, expert_weights_buf,
                gate_up_off_buf, down_off_buf,
            )?;

            // Barrier between routed and shared expert dispatches
            enc.memory_barrier_with_scope(1);

            // Shared expert dispatch (separate kernels)
            if has_shared {
                let gate_off = meta.shared_expert_gate_off.unwrap();
                let up_off = meta.shared_expert_up_off.unwrap();
                let down_off = meta.shared_expert_down_off.unwrap();
                let gate_quant = meta.shared_expert_gate_quant.unwrap_or(QuantScheme::F32);
                let down_quant = meta.shared_expert_down_quant.unwrap_or(QuantScheme::F32);
                Self::encode_shared_expert_ffn_decode_fused(
                    enc, pipelines, s, layer_buf,
                    gate_off, up_off, down_off, gate_quant, down_quant,
                    meta.ffn_gate_inp_shexp_off,
                )?;
            }
        }

        Ok(())
    }

    /// Fused single-encoder variant of encode_shared_expert_ffn_decode_raw.
    pub(crate) fn encode_shared_expert_ffn_decode_fused(
        enc: &MetalComputeEncoder,
        pipelines: &MetalPipelines,
        s: &MetalScratch,
        layer_buf: &MetalBuffer,
        gate_off: u64,
        up_off: u64,
        down_off: u64,
        gate_quant: QuantScheme,
        down_quant: QuantScheme,
        gate_inp_shexp_off: Option<u64>,
    ) -> Result<(), RuntimeError> {
        let hidden_dim = s.hidden_dim;
        let se_inter = s.shared_expert_inter_dim;

        let se_gate_buf = s.shared_expert_gate_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("shared_expert_gate_buf not allocated".into())
        })?;
        let se_down_buf = s.shared_expert_down_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("shared_expert_down_buf not allocated".into())
        })?;

        // Gate+Up+SwiGLU
        match gate_quant {
            QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::Q4_1 => {
                let (se_ffn_tg, se_ffn_n_tg) = match gate_quant {
                    QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q8_0_2sg); (64u64, ((se_inter as u64) + 7) / 8) },
                    QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q4_0_deferred); (128u64, se_inter as u64) },
                    QuantScheme::Q4_1 => { enc.set_pipeline_state(&pipelines.ffn_fused_gate_up_swiglu_q4_1_deferred); (128u64, se_inter as u64) },
                    _ => unreachable!(),
                };
                enc.set_buffer(layer_buf, gate_off, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(se_gate_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                enc.set_bytes(&(se_inter as u32).to_le_bytes(), 4);
                enc.set_buffer(layer_buf, up_off, 5);
                enc.dispatch_threadgroups(
                    MTLSize::new(se_ffn_n_tg, 1, 1),
                    MTLSize::new(se_ffn_tg, 1, 1),
                );
            }
            _ => {
                // F16/F32 fallback: gate matmul
                let (fb_pso, fb_tg, fb_n_tg) = if gate_quant == QuantScheme::F16 {
                    (&pipelines.matmul_f16_deferred_nr2, 128u64, ((se_inter as u64) + 1) / 2)
                } else {
                    (&pipelines.matmul_bytes_f32, s.matmul_tg_size, se_inter as u64)
                };
                enc.set_pipeline_state(fb_pso);
                enc.set_buffer(layer_buf, gate_off, 0);
                enc.set_buffer(&s.normed_buf, 0, 1);
                enc.set_buffer(se_gate_buf, 0, 2);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                if gate_quant == QuantScheme::F16 {
                    enc.set_bytes(&(se_inter as u32).to_le_bytes(), 4);
                }
                enc.dispatch_threadgroups(
                    MTLSize::new(fb_n_tg, 1, 1),
                    MTLSize::new(fb_tg, 1, 1),
                );
                // Up into down_buf as temp
                enc.set_buffer(layer_buf, up_off, 0);
                enc.set_buffer(se_down_buf, 0, 2);
                enc.dispatch_threadgroups(
                    MTLSize::new(fb_n_tg, 1, 1),
                    MTLSize::new(fb_tg, 1, 1),
                );
                // SwiGLU
                enc.set_pipeline_state(&pipelines.swiglu);
                enc.set_buffer(se_gate_buf, 0, 0);
                enc.set_buffer(se_down_buf, 0, 1);
                enc.set_bytes(&(se_inter as u32).to_le_bytes(), 2);
                let tg = 256u64.min(se_inter as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((se_inter as u64).div_ceil(tg), 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
            }
        }

        // Barrier: Gate+Up+SwiGLU writes se_gate_buf, Down projection reads it
        enc.memory_barrier_with_scope(1);

        // Down projection
        {
            let tg_down = match down_quant {
                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_2sg); 64u64 },
                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); 128u64 },
                QuantScheme::Q4_1 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_1_deferred); 128u64 },
                QuantScheme::F16 => { enc.set_pipeline_state(&pipelines.matmul_f16_deferred_nr2); 128u64 },
                _ => { enc.set_pipeline_state(&pipelines.matmul_bytes_f32); s.matmul_tg_size },
            };
            enc.set_buffer(layer_buf, down_off, 0);
            enc.set_buffer(se_gate_buf, 0, 1);
            enc.set_buffer(se_down_buf, 0, 2);
            enc.set_bytes(&(se_inter as u32).to_le_bytes(), 3);
            if matches!(down_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::Q4_1 | QuantScheme::F16) {
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
            }
            let n_tg_down = match down_quant { QuantScheme::Q8_0 => ((hidden_dim as u64) + 7) / 8, QuantScheme::Q4_0 => ((hidden_dim as u64) + 1) / 2, QuantScheme::Q4_1 => ((hidden_dim as u64) + 3) / 4, QuantScheme::F16 => ((hidden_dim as u64) + 1) / 2, _ => hidden_dim as u64 };
            enc.dispatch_threadgroups(MTLSize::new(n_tg_down, 1, 1), MTLSize::new(tg_down, 1, 1));
        }

        // Barrier: Down writes se_down_buf; gating+sigmoid read se_down_buf/se_gate_buf
        enc.memory_barrier_with_scope(1);

        // Shared expert gating + residual add.
        if let Some(gis_off) = gate_inp_shexp_off {
            // Dot product: [1 x hidden_dim] @ normed_buf -> se_gate_buf[0]
            enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
            enc.set_buffer(layer_buf, gis_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(se_gate_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(s.matmul_tg_size, 1, 1));

            // Barrier: dot product writes se_gate_buf[0], sigmoid+scale+add reads it
            enc.memory_barrier_with_scope(1);

            // Fused sigmoid-scale + residual add:
            //   x_buf[i] += sigmoid(se_gate_buf[0]) * se_down_buf[i]
            // Replaces 2 dispatches (sigmoid_scale_buffer + add_residual) with 1.
            if let Some(ref pso_fused) = pipelines.sigmoid_scale_add {
                enc.set_pipeline_state(pso_fused);
                enc.set_buffer(se_gate_buf, 0, 0);     // scalar [1] float
                enc.set_buffer(se_down_buf, 0, 1);      // src [dim] float (read-only)
                enc.set_buffer(&s.x_buf, 0, 2);          // dst [dim] float (R/W)
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                let tg = 256u64.min(hidden_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
            } else {
                // Fallback: separate sigmoid_scale + add_residual
                let pso_sig = pipelines.sigmoid_scale_buffer.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("sigmoid_scale_buffer pipeline not compiled".into())
                })?;
                enc.set_pipeline_state(pso_sig);
                enc.set_buffer(se_gate_buf, 0, 0);
                enc.set_buffer(se_down_buf, 0, 1);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
                let tg_sig = 256u64.min(hidden_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(tg_sig), 1, 1),
                    MTLSize::new(tg_sig, 1, 1),
                );
                // Barrier: sigmoid_scale writes se_down_buf, add_residual reads it
                enc.memory_barrier_with_scope(1);
                // Residual: x_buf += se_down_buf
                enc.set_pipeline_state(&pipelines.add_residual);
                enc.set_buffer(&s.x_buf, 0, 0);
                enc.set_buffer(se_down_buf, 0, 1);
                enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
                let tg = 256u64.min(hidden_dim as u64).max(1);
                enc.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
            }
        } else {
            // No gating: simple residual add
            enc.set_pipeline_state(&pipelines.add_residual);
            enc.set_buffer(&s.x_buf, 0, 0);
            enc.set_buffer(se_down_buf, 0, 1);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 2);
            let tg = 256u64.min(hidden_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new((hidden_dim as u64).div_ceil(tg), 1, 1),
                MTLSize::new(tg, 1, 1),
            );
        }

        Ok(())
    }

    /// Encode MoE FFN block for prefill (batched) into an existing command buffer.
    ///
    /// For each token in the batch, the router selects top-K experts independently.
    /// Option B: dispatch ALL experts for ALL tokens.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_moe_ffn_batched(
        cmd: &MetalCommandBuffer,
        pipelines: &MetalPipelines,
        scratch: &MetalScratch,
        layer_buf: &MetalBuffer,
        batch_size: usize,
        router_weight_off: u64,
        router_weight_quant: QuantScheme,
        expert_gate_offs: &[u64],
        expert_up_offs: &[u64],
        expert_down_offs: &[u64],
        expert_gate_quant: QuantScheme,
        expert_down_quant: QuantScheme,
    ) -> Result<(), RuntimeError> {
        let hidden_dim = scratch.hidden_dim;
        let inter_dim = scratch.moe_expert_inter_dim;
        let num_experts = scratch.moe_num_experts;
        let _top_k = scratch.moe_num_active_experts;

        let router_softmax_batched = pipelines.moe_router_softmax_batched.as_ref().ok_or_else(|| {
            RuntimeError::Compute(
                "MoE router_softmax_batched pipeline not compiled. \
                 Ensure MoE shader kernels are in METAL_SHADER_SOURCE.".into()
            )
        })?;
        let expert_accum_batched = pipelines.moe_expert_accum_batched.as_ref().ok_or_else(|| {
            RuntimeError::Compute(
                "MoE expert_accum_batched pipeline not compiled. \
                 Ensure MoE shader kernels are in METAL_SHADER_SOURCE.".into()
            )
        })?;

        let batch_expert_ids_buf = scratch.moe_batch_expert_ids.as_ref().ok_or_else(|| {
            RuntimeError::Compute("MoE batch expert_ids buffer not allocated".into())
        })?;
        let batch_expert_weights_buf = scratch.moe_batch_expert_weights.as_ref().ok_or_else(|| {
            RuntimeError::Compute("MoE batch expert_weights buffer not allocated".into())
        })?;
        let batch_expert_output_buf = scratch.moe_batch_expert_output.as_ref().ok_or_else(|| {
            RuntimeError::Compute("MoE batch expert_output buffer not allocated".into())
        })?;

        let normed_buf = scratch.batch_normed_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_normed_buf not allocated".into())
        })?;
        let gate_buf = scratch.batch_gate_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_gate_buf not allocated".into())
        })?;
        let x_buf = scratch.batch_x_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_x_buf not allocated".into())
        })?;
        let attn_proj_buf = scratch.batch_attn_proj_buf.as_ref().ok_or_else(|| {
            RuntimeError::Compute("batch_attn_proj_buf not allocated".into())
        })?;

        let _router_weight_quant = router_weight_quant;

        // ---- Step 1: Batched router dispatch ----
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for batched MoE router".into())
            })?;
            enc.set_pipeline_state(router_softmax_batched);
            enc.set_buffer(normed_buf, 0, 0);
            enc.set_buffer(layer_buf, router_weight_off, 1);
            enc.set_buffer(batch_expert_ids_buf, 0, 2);
            enc.set_buffer(batch_expert_weights_buf, 0, 3);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
            enc.set_bytes(&(num_experts as u32).to_le_bytes(), 5);
            enc.set_bytes(&(_top_k as u32).to_le_bytes(), 6);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 7);
            let tg = 256u64.min(hidden_dim as u64).max(1);
            enc.dispatch_threadgroups(
                MTLSize::new(batch_size as u64, 1, 1),
                MTLSize::new(tg, 1, 1),
            );
            enc.end_encoding();
        }

        // ---- Step 2: Expert FFN dispatch (Option B: ALL experts for ALL tokens) ----
        const TILE_M: u64 = 32;
        const TILE_N: u64 = 32;

        for expert_idx in 0..num_experts {
            let gate_off = expert_gate_offs[expert_idx];
            let up_off = expert_up_offs[expert_idx];
            let down_off = expert_down_offs[expert_idx];
            let expert_out_byte_off = (expert_idx * batch_size * hidden_dim * 4) as u64;

            // Gate + Up + SwiGLU (batched tiled GEMM)
            {
                let enc = cmd.new_concurrent_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for MoE batched expert FFN".into())
                })?;

                let m_u32 = batch_size as u32;
                let n_u32 = inter_dim as u32;
                let k_u32 = hidden_dim as u32;

                match expert_gate_quant {
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
                enc.set_buffer(layer_buf, gate_off, 0);
                enc.set_buffer(normed_buf, 0, 1);
                enc.set_buffer(gate_buf, 0, 2);
                enc.set_bytes(&m_u32.to_le_bytes(), 3);
                enc.set_bytes(&n_u32.to_le_bytes(), 4);
                enc.set_bytes(&k_u32.to_le_bytes(), 5);
                enc.dispatch_threadgroups(
                    MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                    MTLSize::new(128, 1, 1),
                );

                let up_buf = scratch.batch_up_buf.as_ref().ok_or_else(|| {
                    RuntimeError::Compute("batch_up_buf not allocated".into())
                })?;
                match expert_gate_quant {
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
                enc.set_buffer(layer_buf, up_off, 0);
                enc.set_buffer(normed_buf, 0, 1);
                enc.set_buffer(up_buf, 0, 2);
                enc.set_bytes(&m_u32.to_le_bytes(), 3);
                enc.set_bytes(&n_u32.to_le_bytes(), 4);
                enc.set_bytes(&k_u32.to_le_bytes(), 5);
                enc.dispatch_threadgroups(
                    MTLSize::new((inter_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                    MTLSize::new(128, 1, 1),
                );

                enc.memory_barrier_with_scope(1);

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

            // Down projection -> expert_output_buf at expert_idx offset
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for MoE batched down proj".into())
                })?;
                let m_u32 = batch_size as u32;
                let n_u32 = hidden_dim as u32;
                let k_u32 = inter_dim as u32;
                match expert_down_quant {
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
                enc.set_buffer(layer_buf, down_off, 0);
                enc.set_buffer(gate_buf, 0, 1);
                enc.set_buffer(batch_expert_output_buf, expert_out_byte_off, 2);
                enc.set_bytes(&m_u32.to_le_bytes(), 3);
                enc.set_bytes(&n_u32.to_le_bytes(), 4);
                enc.set_bytes(&k_u32.to_le_bytes(), 5);
                enc.dispatch_threadgroups(
                    MTLSize::new((hidden_dim as u64).div_ceil(TILE_N), (batch_size as u64).div_ceil(TILE_M), 1),
                    MTLSize::new(128, 1, 1),
                );
                enc.end_encoding();
            }
        }

        // ---- Step 3: Batched weighted accumulation + residual ----
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder for batched MoE accum".into())
            })?;
            enc.set_pipeline_state(expert_accum_batched);
            enc.set_buffer(batch_expert_output_buf, 0, 0);
            enc.set_buffer(batch_expert_weights_buf, 0, 1);
            enc.set_buffer(batch_expert_ids_buf, 0, 2);
            enc.set_buffer(x_buf, 0, 3);
            enc.set_buffer(attn_proj_buf, 0, 4);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 5);
            enc.set_bytes(&(_top_k as u32).to_le_bytes(), 6);
            enc.set_bytes(&(batch_size as u32).to_le_bytes(), 7);
            let total_elems = (batch_size * hidden_dim) as u64;
            let tg_count = total_elems.div_ceil(256);
            enc.dispatch_threadgroups(
                MTLSize::new(tg_count, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }

        Ok(())
    }

    /// Encode one transformer layer into an existing command buffer (batched prefill).
    ///
    /// Optimized: Encodes compute work into the CALLER's command buffer.
    /// The caller is responsible for creating the command buffer and calling
    /// commit_and_wait() after all layers are encoded. This allows encoding
    /// ALL layers into a SINGLE command buffer, eliminating per-layer
    /// GPU-CPU sync barriers.
    ///
    /// Previous: 1 command buffer × N layers = N sync barriers per prefill.
    /// Now: 1 command buffer for ALL N layers = 1 sync barrier per prefill.
    ///
    /// Uses cached zero-copy layer buffer with subtensor offsets (same as decode path)
    /// instead of creating new Metal buffers from bytes each time.
    ///
    /// Flow: RMSNorm -> Q/K/V GEMM -> RoPE -> KV cache write -> Batched attention ->
    ///       Wo GEMM -> Residual -> FFN RMSNorm -> Gate/Up GEMM -> SwiGLU -> Down GEMM -> Residual

    /// GPU-resident Option A decode: two-CB split per MoE layer.
    ///
    /// For each MoE layer, CB1 encodes attention + FFN norm + router, then
    /// synchronously commits and reads back the top-K expert IDs (8 bytes).
    /// CB2 dispatches only the selected expert FFNs via `encode_moe_ffn_decode`
    /// with `cpu_expert_ids` and `skip_router=true`.
    ///
    /// Dense layers (no MoE) are handled identically to `decode_token_greedy`
    /// within the current command buffer with no extra CB split.
    ///
    /// Extends Option A from streaming-only to GPU-resident mode.
    /// Expected 2.5-3x decode speedup for Mixtral 8x7B (2/8 experts = 75% skip).
    pub fn decode_token_option_a_gpu_resident(
        &self,
        token_id: u32,
        _weights: &dyn crate::weight::cache::WeightProvider,
        kv: &mut crate::kv::KvCache,
    ) -> Result<u32, RuntimeError> {
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
                "decode_token_option_a_gpu_resident requires GPU-resident weights".into(),
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
        let top_k = s.moe_num_active_experts;

        // Start the first command buffer: embed + first layer's attention.
        let mut cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for Option A GPU-resident decode".into())
        })?;

        // --- Embed token into x_buf ---
        let (sc_embed_buf, sc_embed_off): (&MetalBuffer, u64) =
            if let Some((emb_o, _, _)) = s.gpu_global_offsets {
                (s.gpu_unified_weight_buf.as_ref().unwrap(), emb_o as u64)
            } else {
                (embedding_buf, 0u64)
            };
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            match self.embedding_quant {
                QuantScheme::Q8_0 => enc.set_pipeline_state(&pipelines.embed_token_q8_0),
                QuantScheme::Q4_0 => enc.set_pipeline_state(&pipelines.embed_token_q4_0),
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
            enc.end_encoding();
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

            // ================================================================
            // GatedDeltaNet branch: GDN layers use linear attention.
            // The GDN function handles norm, QKV, conv, gates, state update,
            // output, gating, residual, and attn_proj_buf copy. Then FFN norm
            // and FFN block proceed as normal (shared path below).
            // ================================================================
            if meta.gdn_layer_idx.is_none() {

            // Attention RMSNorm
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
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
                enc.end_encoding();
            }
            // Fused QKV projection (+ fused bias for Qwen2-family models)
            {
                let has_bias = meta.bq_off.is_some() && meta.bk_off.is_some() && meta.bv_off.is_some();
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
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
                enc.end_encoding();
            }
            // QKV bias addition fallback (only for F32 weights with bias, rare)
            if !matches!(meta.wq_quant, QuantScheme::Q8_0 | QuantScheme::Q4_0 | QuantScheme::F16)
                && (meta.bq_off.is_some() || meta.bk_off.is_some() || meta.bv_off.is_some())
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder for QKV bias".into())
                })?;
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
                enc.end_encoding();
            }
            // Fused RoPE Q + RoPE K + KV cache write (1 dispatch instead of 3)
            let is_linear_attn = meta.layer_type == Some(1);
            let rope_half_dim = s.rotary_dim / 2;
            let use_fused_rope_kv = !is_linear_attn && s.rotary_dim == head_dim;
            if use_fused_rope_kv {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
                let pos_offset_u32 = (seq_pos * rope_half_dim) as u32;
                let fused_pipe = if s.rope_neox {
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
                enc.end_encoding();
            } else {
                if !is_linear_attn {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder".into())
                    })?;
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
                    enc.end_encoding();
                }
                {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder".into())
                    })?;
                    enc.set_pipeline_state(&pipelines.write_kv_cache);
                    enc.set_buffer(&s.qkv_buf, k_byte_off, 0);
                    enc.set_buffer(&s.qkv_buf, v_byte_off, 1);
                    enc.set_buffer(&s.gpu_k_cache[layer_idx], 0, 2);
                    enc.set_buffer(&s.gpu_v_cache[layer_idx], 0, 3);
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 4);
                    enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 5);
                    enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 6);
                    let tg = 64u64.min(kv_dim as u64).max(1);
                    enc.dispatch_threadgroups(
                        MTLSize::new((kv_dim as u64).div_ceil(tg), 1, 1),
                        MTLSize::new(tg, 1, 1),
                    );
                    enc.end_encoding();
                }
            }
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
                        enc.set_bytes(&FLASH_DECODE_TILE_SIZE.to_le_bytes(), 10);
                        enc.set_bytes(&num_tiles.to_le_bytes(), 11);
                        enc.set_bytes(&max_seq_len_u32.to_le_bytes(), 12);
                        enc.dispatch_threadgroups(
                            MTLSize::new((num_heads as u64) * (num_tiles as u64), 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                        enc.end_encoding();
                    }
                    {
                        let enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder".into())
                        })?;
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
                        enc.end_encoding();
                    }
                } else {
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
                    enc.set_bytes(&(num_heads as u32).to_le_bytes(), 5);
                    enc.set_bytes(&(num_kv_heads as u32).to_le_bytes(), 6);
                    enc.set_bytes(&(head_dim as u32).to_le_bytes(), 7);
                    enc.set_bytes(&(kv_dim as u32).to_le_bytes(), 8);
                    enc.set_bytes(&(new_seq_len as u32).to_le_bytes(), 9);
                    enc.set_bytes(&attn_scale.to_le_bytes(), 10);
                    enc.set_bytes(&(s.max_seq_len as u32).to_le_bytes(), 11);
                    let tg_threads = mha_tg_size.min((head_dim.max(new_seq_len) as u64).max(1));
                    enc.dispatch_threadgroups(
                        MTLSize::new(num_heads as u64, 1, 1),
                        MTLSize::new(tg_threads, 1, 1),
                    );
                    enc.end_encoding();
                }
            }
            // Wo projection + Residual 1 (fused)
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
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
                enc.end_encoding();
            }

            } else {
                // GatedDeltaNet layer: linear attention forward pass
                let gdn_idx = meta.gdn_layer_idx.unwrap();
                let new_conv_pos = Self::encode_gdn_layer_decode(
                    &cmd, pipelines, s, layer_buf, meta, gdn_idx,
                )?;
                s.gdn_conv_positions[gdn_idx] = new_conv_pos;
            }

            // FFN RMSNorm
            {
                let enc = cmd.new_compute_encoder().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create encoder".into())
                })?;
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
                enc.end_encoding();
            }
            // FFN block: branch on MoE vs dense
            if let Some(ref moe_meta) = meta.moe_meta {
                // ============================================================
                // Option A GPU-resident: two-CB split per MoE layer
                // ============================================================
                // CB1 contains attention + FFN norm + router (encode router into
                // per_layer_ids_buf). Commit synchronously, read back expert_ids.
                // CB2 dispatches only top-K expert FFNs + shared expert.
                // CB2 uses async commit — Metal FIFO guarantees ordering.

                // Encode router into the current CB (CB1) using per-layer buffer.
                let per_layer_ids_buf = s.moe_per_layer_expert_ids
                    .get(layer_idx)
                    .and_then(|opt| opt.as_ref());
                {
                    let router_softmax = pipelines.moe_router_softmax.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("MoE router_softmax pipeline not compiled.".into())
                    })?;
                    let expert_ids_buf = per_layer_ids_buf.unwrap_or_else(|| {
                        s.moe_expert_ids.as_ref().unwrap()
                    });
                    let expert_weights_buf = s.moe_expert_weights.as_ref().ok_or_else(|| {
                        RuntimeError::Compute("MoE expert_weights buffer not allocated".into())
                    })?;
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder for MoE router".into())
                    })?;
                    enc.set_pipeline_state(router_softmax);
                    enc.set_buffer(&s.normed_buf, 0, 0);
                    enc.set_buffer(layer_buf, moe_meta.router_weight_off, 1);
                    enc.set_buffer(expert_ids_buf, 0, 2);
                    enc.set_buffer(expert_weights_buf, 0, 3);
                    enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 4);
                    enc.set_bytes(&(s.moe_num_experts as u32).to_le_bytes(), 5);
                    enc.set_bytes(&(top_k as u32).to_le_bytes(), 6);
                    let tg = 256u64.min(hidden_dim as u64).max(1);
                    enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(tg, 1, 1));
                    enc.end_encoding();
                }

                // Synchronous commit: flush CB1 (attn + norm + router).
                cmd.commit_and_wait();

                // Read back expert_ids from per-layer buffer (top_k * 4 bytes).
                let ids_buf = per_layer_ids_buf.unwrap_or_else(|| {
                    s.moe_expert_ids.as_ref().unwrap()
                });
                let mut cpu_ids = vec![0u32; top_k];
                ids_buf.read_u32(&mut cpu_ids);

                // Record expert activation in profiler.
                if let Some(ref profiler) = self.expert_profiler {
                    profiler.lock().unwrap().record(layer_idx, &cpu_ids);
                }

                // CB2: dispatch only top-K expert FFNs.
                cmd = self.queue.new_command_buffer().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create CB2 for Option A GPU-resident FFN".into())
                })?;
                Self::encode_moe_ffn_decode(
                    &cmd, pipelines, s, layer_buf, moe_meta,
                    per_layer_ids_buf,
                    None,  // GPU-resident: all weights in layer_buf
                    None, 0.0,  // No cache bias in GPU-resident mode
                    Some(&cpu_ids),  // Option A: only top-K experts
                    true,  // Skip router (already ran in CB1)
                    None,  // No per-layer routing weights in Option A CB2
                )?;

                // Shared expert dispatch for GPU-resident Option A.
                // After routed experts, add the always-active shared expert output.
                if meta.shared_expert_gate_off.is_some() {
                    Self::encode_shared_expert_ffn_decode(
                        &cmd, pipelines, s, layer_buf, meta,
                    )?;
                }

                // Async commit CB2: Metal FIFO ordering guarantees the next CB
                // won't execute until this one finishes, so the next layer's
                // attention naturally waits for x_buf. No CPU-side block needed.
                // Changed from commit_and_wait to async commit.
                cmd.commit();

                // New CB for next layer (or final norm).
                cmd = self.queue.new_command_buffer().ok_or_else(|| {
                    RuntimeError::Compute("Failed to create command buffer for next layer".into())
                })?;

            } else {
                // Dense FFN path (unchanged from decode_token_greedy)
                if meta.w_gate_quant == QuantScheme::Q8_0 && meta.w_up_quant == QuantScheme::Q8_0 {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder".into())
                    })?;
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
                    enc.end_encoding();
                } else if matches!(meta.w_gate_quant, QuantScheme::Q4_0) {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder".into())
                    })?;
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
                    enc.end_encoding();
                } else {
                    let (fb_pso, fb_tg, fb_n_tg) = if meta.w_gate_quant == QuantScheme::F16 {
                        (&pipelines.matmul_f16_deferred_nr2, 128u64, ((inter_dim as u64) + 1) / 2)
                    } else {
                        (&pipelines.matmul_bytes_f32, matmul_tg_size, inter_dim as u64)
                    };
                    {
                        let enc = cmd.new_compute_encoder().ok_or_else(|| {
                            RuntimeError::Compute("Failed to create encoder".into())
                        })?;
                        enc.set_pipeline_state(fb_pso);
                        enc.set_buffer(layer_buf, w_gate_off, 0);
                        enc.set_buffer(&s.normed_buf, 0, 1);
                        enc.set_buffer(&s.gate_buf, 0, 2);
                        enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
                        if meta.w_gate_quant == QuantScheme::F16 {
                            enc.set_bytes(&(inter_dim as u32).to_le_bytes(), 4);
                        }
                        enc.dispatch_threadgroups(
                            MTLSize::new(fb_n_tg, 1, 1),
                            MTLSize::new(fb_tg, 1, 1),
                        );
                        enc.set_buffer(layer_buf, w_up_off, 0);
                        enc.set_buffer(&s.up_buf, 0, 2);
                        enc.dispatch_threadgroups(
                            MTLSize::new(fb_n_tg, 1, 1),
                            MTLSize::new(fb_tg, 1, 1),
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
                // Down projection + Residual 2 (fused)
                {
                    let enc = cmd.new_compute_encoder().ok_or_else(|| {
                        RuntimeError::Compute("Failed to create encoder".into())
                    })?;
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
                    enc.end_encoding();
                }
            } // end MoE vs dense FFN branch
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

        // --- Final RMSNorm ---
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
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
            enc.end_encoding();
        }

        // --- Logits projection (deferred NR0=2 for Q8_0, NR0=4 for others, 128 threads) ---
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            let (proj_tg, proj_rows_per_tg) = match output_proj_quant {
                QuantScheme::Q8_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0_deferred_nr2); (128u64, 2u64) },
                QuantScheme::Q4_0 => { enc.set_pipeline_state(&pipelines.dequant_matmul_q4_0_deferred_nr2); (128u64, 2u64) },
                _ => { enc.set_pipeline_state(&pipelines.matmul_f32_deferred); (128u64, 4u64) },
            };
            enc.set_buffer(sc_proj_buf, sc_proj_off, 0);
            enc.set_buffer(&s.normed_buf, 0, 1);
            enc.set_buffer(&s.logits_buf, 0, 2);
            enc.set_bytes(&(hidden_dim as u32).to_le_bytes(), 3);
            enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 4);
            let n_tg = ((vocab_size as u64) + proj_rows_per_tg - 1) / proj_rows_per_tg;
            enc.dispatch_threadgroups(
                MTLSize::new(n_tg, 1, 1),
                MTLSize::new(proj_tg, 1, 1),
            );
            enc.end_encoding();
        }

        // --- GPU-side argmax (replaces 128 KB logits readback) ---
        {
            let enc = cmd.new_compute_encoder().ok_or_else(|| {
                RuntimeError::Compute("Failed to create encoder".into())
            })?;
            enc.set_pipeline_state(&pipelines.argmax);
            enc.set_buffer(&s.logits_buf, 0, 0);
            enc.set_buffer(&s.argmax_result_buf, 0, 1);
            enc.set_bytes(&(vocab_size as u32).to_le_bytes(), 2);
            enc.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }

        // Final sync: last CB contains final norm + logits + argmax.
        cmd.commit_and_wait();

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
