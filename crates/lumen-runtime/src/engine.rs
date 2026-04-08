//! The core inference engine implementing the minimal "atomic" loop.
//!
//! This is Section 4 of the design spec: the token generation loop.
//! Everything else in the runtime exists to make `ensure(weights available)`
//! non-blocking and cheap.
//!
//! ```text
//! for each output token:
//!   for layer in 0..L-1:
//!     ensure(weights[layer] available in compute memory)
//!     if layer uses attention:
//!       ensure(KV needed for this layer available)
//!     y = backend.compute_layer(layer, x, KV)
//!     if layer uses attention:
//!       backend.update_kv(layer, KV_new)
//!     x = y
//!   logits = backend.final(x)
//!   token = sample(logits)
//! ```

use crate::weight::cache::{PrefetchPriority, WeightProvider};
use crate::compute::{ActivationBuffer, ComputeBackend, ComputeDtype, Logits};
use crate::config::RuntimeConfig;
use crate::error::RuntimeError;
use crate::kv::{KvCache, KvCacheConfig};
use crate::pipeline::PipelineMode;
use crate::telemetry::{InferenceMetrics, IoMetrics, PerLayerTiming};
use lumen_format::ModelHyperparams;
use std::time::Instant;

#[cfg(target_os = "macos")]
use crate::accelerate::AccelerateBatchBackend;

/// Sampling parameters for token generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for softmax (1.0 = no scaling, <1.0 = sharper, >1.0 = flatter).
    pub temperature: f32,

    /// Random seed for reproducible sampling.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            seed: None,
        }
    }
}

/// Minimal xorshift64 PRNG — deterministic, zero deps.
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        // Ensure non-zero state
        Self { state: if seed == 0 { 0xDEAD_BEEF_CAFE_BABEu64 } else { seed } }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        const SCALE: f32 = 1.0 / (1u64 << 24) as f32;
        (self.next_u64() >> 40) as f32 * SCALE
    }
}

/// Softmax with max-subtraction for numerical stability (best practice 4.1).
/// Shared between sampling and compute backends.
///
/// On aarch64, phases 1 (find max) and 3 (normalize) use NEON SIMD with
/// 4-accumulator unrolling (16 floats/iter). Phase 2 (subtract-max + exp + sum)
/// uses scalar f32::exp() for bit-identical results across backends.
#[cfg(target_arch = "aarch64")]
pub fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }

    unsafe {
        use std::arch::aarch64::*;
        let len = logits.len();
        let ptr = logits.as_mut_ptr();

        // Phase 1: Find max using NEON with 4 accumulators for ILP (16 floats/iter).
        let mut max0 = vdupq_n_f32(f32::NEG_INFINITY);
        let mut max1 = max0;
        let mut max2 = max0;
        let mut max3 = max0;

        let chunks16 = len / 16;
        let mut i = 0;
        for _ in 0..chunks16 {
            max0 = vmaxq_f32(max0, vld1q_f32(ptr.add(i)));
            max1 = vmaxq_f32(max1, vld1q_f32(ptr.add(i + 4)));
            max2 = vmaxq_f32(max2, vld1q_f32(ptr.add(i + 8)));
            max3 = vmaxq_f32(max3, vld1q_f32(ptr.add(i + 12)));
            i += 16;
        }
        // 4-wide remainder
        while i + 4 <= len {
            max0 = vmaxq_f32(max0, vld1q_f32(ptr.add(i)));
            i += 4;
        }
        // Reduce 4 vectors to scalar
        max0 = vmaxq_f32(vmaxq_f32(max0, max1), vmaxq_f32(max2, max3));
        let mut max_val = vmaxvq_f32(max0);
        // Scalar tail
        while i < len {
            max_val = max_val.max(*ptr.add(i));
            i += 1;
        }

        if !max_val.is_finite() {
            let uniform = 1.0 / len as f32;
            logits.fill(uniform);
            return;
        }

        // Phase 2: Subtract max + exp + sum (scalar for bit-identical results).
        let mut sum = 0.0f32;
        for v in logits.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }

        if sum <= f32::EPSILON {
            let uniform = 1.0 / len as f32;
            logits.fill(uniform);
            return;
        }

        // Phase 3: Normalize using NEON multiply (16 floats/iter).
        let inv_sum = 1.0 / sum;
        let inv_sum_v = vdupq_n_f32(inv_sum);
        i = 0;
        for _ in 0..chunks16 {
            vst1q_f32(ptr.add(i), vmulq_f32(vld1q_f32(ptr.add(i)), inv_sum_v));
            vst1q_f32(ptr.add(i + 4), vmulq_f32(vld1q_f32(ptr.add(i + 4)), inv_sum_v));
            vst1q_f32(ptr.add(i + 8), vmulq_f32(vld1q_f32(ptr.add(i + 8)), inv_sum_v));
            vst1q_f32(ptr.add(i + 12), vmulq_f32(vld1q_f32(ptr.add(i + 12)), inv_sum_v));
            i += 16;
        }
        // 4-wide remainder
        while i + 4 <= len {
            vst1q_f32(ptr.add(i), vmulq_f32(vld1q_f32(ptr.add(i)), inv_sum_v));
            i += 4;
        }
        // Scalar tail
        while i < len {
            *ptr.add(i) *= inv_sum;
            i += 1;
        }
    }
}

/// Scalar fallback for non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
pub fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let uniform = 1.0 / logits.len() as f32;
        logits.fill(uniform);
        return;
    }
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum <= f32::EPSILON {
        let uniform = 1.0 / logits.len() as f32;
        logits.fill(uniform);
        return;
    }
    for v in logits.iter_mut() {
        *v /= sum;
    }
}

/// Sample a token from logits using temperature scaling and a mutable RNG.
///
/// - temperature <= 0.0: greedy (argmax)
/// - temperature > 0.0: scale logits, softmax, then sample from distribution
///
/// The RNG is passed by `&mut` so its state advances across calls,
/// producing different random draws for each token.
pub fn sample_token(logits: &mut Logits, params: &SamplingParams, rng: &mut Xorshift64) -> u32 {
    if logits.data.is_empty() {
        return 0;
    }
    if params.temperature <= 0.0 {
        return logits.argmax() as u32;
    }

    // Temperature scaling (in-place, SIMD on aarch64)
    let inv_temp = 1.0 / params.temperature;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        unsafe {
            let scale_v = vdupq_n_f32(inv_temp);
            let ptr = logits.data.as_mut_ptr();
            let len = logits.data.len();
            let chunks16 = len / 16;
            let mut i = 0;
            for _ in 0..chunks16 {
                vst1q_f32(ptr.add(i), vmulq_f32(vld1q_f32(ptr.add(i)), scale_v));
                vst1q_f32(ptr.add(i + 4), vmulq_f32(vld1q_f32(ptr.add(i + 4)), scale_v));
                vst1q_f32(ptr.add(i + 8), vmulq_f32(vld1q_f32(ptr.add(i + 8)), scale_v));
                vst1q_f32(ptr.add(i + 12), vmulq_f32(vld1q_f32(ptr.add(i + 12)), scale_v));
                i += 16;
            }
            while i + 4 <= len {
                vst1q_f32(ptr.add(i), vmulq_f32(vld1q_f32(ptr.add(i)), scale_v));
                i += 4;
            }
            while i < len {
                *ptr.add(i) *= inv_temp;
                i += 1;
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for v in logits.data.iter_mut() {
            *v *= inv_temp;
        }
    }

    // Softmax with max-subtraction (best practice 4.1)
    softmax_inplace(&mut logits.data);

    // Sample from the probability distribution
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in logits.data.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }

    // Fallback to last token (rounding error)
    (logits.data.len() - 1) as u32
}

/// When to stop generating tokens.
#[derive(Debug, Clone)]
pub enum StopCondition {
    /// Stop after generating exactly this many tokens.
    MaxTokens(usize),

    /// Stop when any of these token IDs are generated.
    EosTokens(Vec<u32>),

    /// Stop on max tokens OR EOS, whichever comes first.
    MaxTokensOrEos {
        max_tokens: usize,
        eos_tokens: Vec<u32>,
    },
}

impl StopCondition {
    /// Check if generation should stop given the latest token and count.
    pub fn should_stop(&self, token: u32, generated_count: usize) -> bool {
        match self {
            Self::MaxTokens(max) => generated_count >= *max,
            Self::EosTokens(eos) => eos.contains(&token),
            Self::MaxTokensOrEos { max_tokens, eos_tokens } => {
                generated_count >= *max_tokens || eos_tokens.contains(&token)
            }
        }
    }
}

/// The core inference engine.
///
/// Orchestrates the token generation loop by coordinating the weight
/// provider (cache + storage), compute backend, KV cache, and pipeline
/// scheduler.
pub struct InferenceEngine {
    config: RuntimeConfig,
    hyperparams: ModelHyperparams,
}

impl InferenceEngine {
    /// Create a new inference engine with the given configuration.
    pub fn new(config: RuntimeConfig, hyperparams: ModelHyperparams) -> Self {
        Self {
            config,
            hyperparams,
        }
    }

    /// Run the unified token generation loop.
    ///
    /// Dispatches prefill and decode via `backend.caps()`:
    ///
    /// - **Prefill**: If `caps.batched_prefill`, uses `backend.prefill()` (GPU
    ///   batched GEMM). Otherwise, processes tokens one at a time through
    ///   `forward_pass()` + `kv.advance_seq_len()`.
    ///
    /// - **Decode**: Three paths, selected by capability flags:
    ///   1. `gpu_resident && gpu_argmax && temperature <= 0.0`: greedy GPU decode
    ///      via `backend.decode_token_greedy()` (4-byte readback per token).
    ///   2. `gpu_resident`: GPU decode via `backend.decode_token()` (logits
    ///      readback + CPU sampling).
    ///   3. Fallback: per-layer `forward_pass()` + `compute_final()` + CPU
    ///      sampling.
    ///
    /// All three trait methods (`prefill`, `decode_token`, `decode_token_greedy`)
    /// advance `kv.seq_len()` internally. The fallback `forward_pass()` path
    /// requires the caller to call `kv.advance_seq_len()`.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        weights: &dyn WeightProvider,
        backend: &dyn ComputeBackend,
        stop: &StopCondition,
        sampling: &SamplingParams,
    ) -> Result<GenerationResult, RuntimeError> {
        let num_layers = self.hyperparams.num_layers as usize;
        if num_layers == 0 {
            return Err(RuntimeError::Config("model has 0 layers".into()));
        }
        let total_start = Instant::now();
        let caps = backend.caps();

        // Initialize RNG once for the entire generation session (C-1 fix).
        let mut rng = Xorshift64::new(sampling.seed.unwrap_or(42));

        // Initialize KV cache.
        let mut kv = KvCache::new(KvCacheConfig {
            max_seq_len: self.config.max_seq_len,
            num_layers,
            num_kv_heads: self.hyperparams.num_kv_heads as usize,
            head_dim: self.hyperparams.head_dim as usize,
            precision: self.config.kv_precision,
        })?;

        // Derive the maximum expected generated tokens from the stop condition
        // to pre-allocate output and timing vectors accurately.
        // Clamp to context window to prevent multi-GB allocations from huge --max-tokens.
        let context_budget = self.config.max_seq_len.saturating_sub(prompt_tokens.len());
        let max_expected = match stop {
            StopCondition::MaxTokens(n) => (*n).min(context_budget),
            StopCondition::EosTokens(_) => context_budget,
            StopCondition::MaxTokensOrEos { max_tokens, .. } => (*max_tokens).min(context_budget),
        };
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_expected);
        let expected_total_tokens = prompt_tokens.len() + max_expected;
        let mut per_layer_timings: Vec<PerLayerTiming> = if self.config.collect_per_layer_timings {
            Vec::with_capacity(num_layers * expected_total_tokens)
        } else {
            Vec::new()
        };

        // Reset GDN recurrent state for this new sequence (h_state, conv_state).
        // No-op for non-GDN backends.
        backend.reset_recurrent_state();

        // ---- Prefill ----
        let prefill_start = Instant::now();

        let mut logits = if caps.batched_prefill {
            // Batched prefill: single GPU dispatch for all prompt tokens.
            // prefill() advances kv.seq_len() internally -- do NOT advance here.
            let last_hidden = backend.prefill(prompt_tokens, weights, &mut kv)?;
            let mut current_x = ActivationBuffer::zeros(last_hidden.len(), ComputeDtype::F32);
            current_x.write_f32_from(&last_hidden);
            backend.compute_final(&current_x)?
        } else {
            // Token-at-a-time prefill: forward_pass() does NOT advance kv.seq_len(),
            // so we advance after each token.
            let mut x: Option<ActivationBuffer> = None;
            for &token_id in prompt_tokens {
                x = Some(self.forward_pass(
                    token_id,
                    num_layers,
                    weights,
                    backend,
                    &mut kv,
                    &mut per_layer_timings,
                )?);
                kv.advance_seq_len()?;
            }
            let current_x = x.ok_or_else(|| {
                RuntimeError::Compute("empty prompt: no tokens to process".to_string())
            })?;
            backend.compute_final(&current_x)?
        };

        let prefill_time = prefill_start.elapsed();

        // ---- Decode: generate tokens one at a time ----
        let decode_start = Instant::now();

        let mut next_token = sample_token(&mut logits, sampling, &mut rng);
        generated_tokens.push(next_token);

        let use_gpu_greedy = caps.gpu_resident && caps.gpu_argmax && sampling.temperature <= 0.0;

        if use_gpu_greedy {
            // GPU-RESIDENT GREEDY fast path: argmax on GPU, 4-byte readback.
            // decode_token_greedy() advances kv.seq_len() internally.
            while !stop.should_stop(next_token, generated_tokens.len()) {
                next_token = backend.decode_token_greedy(next_token, weights, &mut kv)?;
                generated_tokens.push(next_token);
            }
        } else if caps.gpu_resident {
            // GPU-RESIDENT fast path: single command buffer per token.
            // decode_token() advances kv.seq_len() internally.
            while !stop.should_stop(next_token, generated_tokens.len()) {
                logits = backend.decode_token(next_token, weights, &mut kv)?;
                next_token = sample_token(&mut logits, sampling, &mut rng);
                generated_tokens.push(next_token);
            }
        } else {
            // Streaming/CPU path: per-layer forward_pass + compute_final.
            // forward_pass() does NOT advance kv.seq_len() -- we advance here.
            while !stop.should_stop(next_token, generated_tokens.len()) {
                let x = self.forward_pass(
                    next_token,
                    num_layers,
                    weights,
                    backend,
                    &mut kv,
                    &mut per_layer_timings,
                )?;
                kv.advance_seq_len()?;

                logits = backend.compute_final(&x)?;
                next_token = sample_token(&mut logits, sampling, &mut rng);
                generated_tokens.push(next_token);
            }
        }

        let decode_time = decode_start.elapsed();
        let total_time = total_start.elapsed();

        let io = match weights.io_snapshot() {
            Some(snap) => IoMetrics {
                bytes_read: snap.bytes_read,
                read_ops: snap.read_ops,
                duration: total_time,
                ..Default::default()
            },
            None => IoMetrics {
                duration: total_time,
                ..Default::default()
            },
        };

        let metrics = InferenceMetrics {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated_tokens.len(),
            prefill_tokens_per_sec: if prefill_time.is_zero() {
                0.0
            } else {
                prompt_tokens.len() as f64 / prefill_time.as_secs_f64()
            },
            decode_tokens_per_sec: if decode_time.is_zero() {
                0.0
            } else {
                generated_tokens.len() as f64 / decode_time.as_secs_f64()
            },
            tpot_ms: if generated_tokens.is_empty() {
                0.0
            } else {
                decode_time.as_secs_f64() * 1000.0 / generated_tokens.len() as f64
            },
            total_time,
            prefill_time,
            decode_time,
            io,
            weight_cache_hit_rate: weights.stats().hit_rate(),
            per_layer_timings,
            ..Default::default()
        };

        Ok(GenerationResult {
            tokens: generated_tokens,
            metrics,
        })
    }

    /// Run token generation with batched Accelerate prefill.
    ///
    /// Uses `AccelerateBatchBackend` for AMX-accelerated batched prefill
    /// (matrix-matrix GEMM), then falls back to the standard `ComputeBackend`
    /// for token-at-a-time decode (matrix-vector, SIMD Q8_0).
    #[cfg(target_os = "macos")]
    pub fn generate_with_prefill(
        &self,
        prompt_tokens: &[u32],
        weights: &dyn WeightProvider,
        backend: &dyn ComputeBackend,
        accel: &mut AccelerateBatchBackend,
        stop: &StopCondition,
        sampling: &SamplingParams,
    ) -> Result<GenerationResult, RuntimeError> {
        let num_layers = self.hyperparams.num_layers as usize;
        if num_layers == 0 {
            return Err(RuntimeError::Config("model has 0 layers".into()));
        }
        let total_start = Instant::now();

        let mut rng = Xorshift64::new(sampling.seed.unwrap_or(42));

        // Initialize KV cache.
        let mut kv = KvCache::new(KvCacheConfig {
            max_seq_len: self.config.max_seq_len,
            num_layers,
            num_kv_heads: self.hyperparams.num_kv_heads as usize,
            head_dim: self.hyperparams.head_dim as usize,
            precision: self.config.kv_precision,
        })?;

        let context_budget = self.config.max_seq_len.saturating_sub(prompt_tokens.len());
        let max_expected = match stop {
            StopCondition::MaxTokens(n) => (*n).min(context_budget),
            StopCondition::EosTokens(_) => context_budget,
            StopCondition::MaxTokensOrEos { max_tokens, .. } => (*max_tokens).min(context_budget),
        };
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_expected);
        let expected_total_tokens = prompt_tokens.len() + max_expected;
        let mut per_layer_timings: Vec<PerLayerTiming> = if self.config.collect_per_layer_timings {
            Vec::with_capacity(num_layers * expected_total_tokens)
        } else {
            Vec::new()
        };

        // ---- Batched prefill via Accelerate/AMX ----
        let prefill_start = Instant::now();

        let last_hidden = accel.prefill(prompt_tokens, weights, &mut kv)?;

        // Convert last hidden state to ActivationBuffer for the decode backend
        let mut current_x = ActivationBuffer::zeros(last_hidden.len(), ComputeDtype::F32);
        current_x.write_f32_from(&last_hidden);

        let prefill_time = prefill_start.elapsed();

        // ---- Decode: generate tokens one at a time (standard SIMD path) ----
        let decode_start = Instant::now();

        let mut logits = backend.compute_final(&current_x)?;
        let mut next_token = sample_token(&mut logits, sampling, &mut rng);
        generated_tokens.push(next_token);

        while !stop.should_stop(next_token, generated_tokens.len()) {
            current_x = self.forward_pass(
                next_token,
                num_layers,
                weights,
                backend,
                &mut kv,
                &mut per_layer_timings,
            )?;
            kv.advance_seq_len()?;

            logits = backend.compute_final(&current_x)?;
            next_token = sample_token(&mut logits, sampling, &mut rng);
            generated_tokens.push(next_token);
        }

        let decode_time = decode_start.elapsed();
        let total_time = total_start.elapsed();

        let io = match weights.io_snapshot() {
            Some(snap) => IoMetrics {
                bytes_read: snap.bytes_read,
                read_ops: snap.read_ops,
                duration: total_time,
                ..Default::default()
            },
            None => IoMetrics {
                duration: total_time,
                ..Default::default()
            },
        };

        let metrics = InferenceMetrics {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated_tokens.len(),
            prefill_tokens_per_sec: if prefill_time.is_zero() {
                0.0
            } else {
                prompt_tokens.len() as f64 / prefill_time.as_secs_f64()
            },
            decode_tokens_per_sec: if decode_time.is_zero() {
                0.0
            } else {
                generated_tokens.len() as f64 / decode_time.as_secs_f64()
            },
            tpot_ms: if generated_tokens.is_empty() {
                0.0
            } else {
                decode_time.as_secs_f64() * 1000.0 / generated_tokens.len() as f64
            },
            total_time,
            prefill_time,
            decode_time,
            io,
            weight_cache_hit_rate: weights.stats().hit_rate(),
            per_layer_timings,
            ..Default::default()
        };

        Ok(GenerationResult {
            tokens: generated_tokens,
            metrics,
        })
    }

    /// Execute a single forward pass through all layers for one token.
    ///
    /// This is the inner loop from Section 4. For each layer:
    /// 1. Prefetch upcoming layers.
    /// 2. Ensure current layer weights are available.
    /// 3. Compute the layer with KV cache.
    /// 4. Commit KV updates.
    /// 5. Release weight hint (MinMem mode).
    fn forward_pass(
        &self,
        token_id: u32,
        num_layers: usize,
        weights: &dyn WeightProvider,
        backend: &dyn ComputeBackend,
        kv: &mut KvCache,
        timings: &mut Vec<PerLayerTiming>,
    ) -> Result<ActivationBuffer, RuntimeError> {
        let prefetch_dist = self.config.prefetch_distance;
        let seq_pos = kv.seq_len();
        let collect_timings = self.config.collect_per_layer_timings;

        // Reset per-token state in the weight provider (e.g. compute cursor
        // for windowed prefetch). Must happen before any prefetch or get calls.
        weights.begin_pass();

        let mut x = backend.embed_token(token_id)?;

        for layer in 0..num_layers {
            // Prefetch upcoming layers.
            let max_ahead = prefetch_dist.min(num_layers - 1 - layer);
            for ahead in 1..=max_ahead {
                let priority = if ahead == 1 {
                    PrefetchPriority::High
                } else {
                    PrefetchPriority::Normal
                };
                let _ = weights.prefetch_layer(layer + ahead, priority);
            }

            // Ensure current layer weights are available.
            // Gate Instant::now() behind collect_timings -- it is a syscall.
            let load_start = if collect_timings { Some(Instant::now()) } else { None };
            let (layer_view, weight_cache_hit) = match weights.try_get_layer(layer) {
                Some(view) => (view, true),
                None => (weights.get_layer_blocking(layer)?, false),
            };
            // Snapshot weight_load_time immediately to get accurate duration.
            let weight_load_time = load_start.map(|t| t.elapsed());

            // Compute with KV cache.
            let mut kv_view = kv.view_mut(layer)?;
            let compute_start = if collect_timings { Some(Instant::now()) } else { None };
            backend.compute_layer(layer, &mut x, &layer_view, Some(&mut kv_view), seq_pos)?;
            let compute_time = compute_start.map(|t| t.elapsed());

            // Commit KV updates.
            let kv_save_start = if collect_timings { Some(Instant::now()) } else { None };
            kv.commit_view(kv_view)?;
            let kv_save_time = kv_save_start.map(|t| t.elapsed());

            // Release weight hint in MinMem mode.
            if self.config.pipeline_mode == PipelineMode::MinMem {
                weights.release_layer_hint(layer);
            }

            // Only construct and push PerLayerTiming when collecting.
            if collect_timings {
                let wlt = weight_load_time.unwrap();
                timings.push(PerLayerTiming {
                    layer_idx: layer,
                    weight_load_time: wlt,
                    compute_time: compute_time.unwrap(),
                    kv_save_time: kv_save_time.unwrap(),
                    stall_time: if weight_cache_hit {
                        std::time::Duration::ZERO
                    } else {
                        wlt
                    },
                    weight_cache_hit,
                    ..Default::default()
                });
            }
        }

        Ok(x)
    }

    /// Returns the runtime configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Returns the model hyperparameters.
    pub fn hyperparams(&self) -> &ModelHyperparams {
        &self.hyperparams
    }
}

impl std::fmt::Debug for InferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEngine")
            .field("pipeline_mode", &self.config.pipeline_mode)
            .field("num_layers", &self.hyperparams.num_layers)
            .field("prefetch_distance", &self.config.prefetch_distance)
            .finish()
    }
}

/// The result of a token generation session.
#[derive(Debug)]
pub struct GenerationResult {
    /// The generated token IDs (not including the prompt).
    pub tokens: Vec<u32>,

    /// Inference metrics for the session.
    pub metrics: InferenceMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Xorshift64 tests ---

    #[test]
    fn xorshift64_zero_seed_uses_fallback() {
        let mut rng = Xorshift64::new(0);
        // Should not be stuck at 0
        let v = rng.next_u64();
        assert_ne!(v, 0);
    }

    #[test]
    fn xorshift64_determinism() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn xorshift64_next_f32_in_range() {
        let mut rng = Xorshift64::new(12345);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!(v >= 0.0, "got {v} < 0.0");
            assert!(v < 1.0, "got {v} >= 1.0");
        }
    }

    // --- softmax_inplace tests ---

    #[test]
    fn softmax_empty() {
        let mut v: Vec<f32> = vec![];
        softmax_inplace(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn softmax_single_element() {
        let mut v = vec![5.0];
        softmax_inplace(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_all_same_gives_uniform() {
        let mut v = vec![3.0; 4];
        softmax_inplace(&mut v);
        for &p in &v {
            assert!((p - 0.25).abs() < 1e-6, "expected uniform 0.25, got {p}");
        }
    }

    #[test]
    fn softmax_all_nan_gives_uniform() {
        let mut v = vec![f32::NAN; 3];
        softmax_inplace(&mut v);
        for &p in &v {
            assert!((p - 1.0 / 3.0).abs() < 1e-6, "expected uniform, got {p}");
        }
    }

    #[test]
    fn softmax_all_neg_inf_gives_uniform() {
        let mut v = vec![f32::NEG_INFINITY; 3];
        softmax_inplace(&mut v);
        for &p in &v {
            assert!((p - 1.0 / 3.0).abs() < 1e-6, "expected uniform, got {p}");
        }
    }

    // --- sample_token tests ---

    #[test]
    fn sample_token_empty_logits() {
        let mut logits = Logits { data: vec![] };
        let params = SamplingParams { temperature: 1.0, seed: None };
        let mut rng = Xorshift64::new(42);
        assert_eq!(sample_token(&mut logits, &params, &mut rng), 0);
    }

    #[test]
    fn sample_token_greedy_returns_argmax() {
        let mut logits = Logits { data: vec![1.0, 5.0, 3.0, 2.0] };
        let params = SamplingParams { temperature: 0.0, seed: None };
        let mut rng = Xorshift64::new(42);
        assert_eq!(sample_token(&mut logits, &params, &mut rng), 1);
    }

    #[test]
    fn sample_token_deterministic_with_seed() {
        let params = SamplingParams { temperature: 0.8, seed: Some(42) };

        let mut rng1 = Xorshift64::new(42);
        let mut logits1 = Logits { data: vec![1.0, 2.0, 3.0, 4.0] };
        let t1 = sample_token(&mut logits1, &params, &mut rng1);

        let mut rng2 = Xorshift64::new(42);
        let mut logits2 = Logits { data: vec![1.0, 2.0, 3.0, 4.0] };
        let t2 = sample_token(&mut logits2, &params, &mut rng2);

        assert_eq!(t1, t2);
    }

    // --- StopCondition tests ---

    #[test]
    fn stop_condition_max_tokens_boundary() {
        let stop = StopCondition::MaxTokens(5);
        assert!(!stop.should_stop(0, 4));
        assert!(stop.should_stop(0, 5));
        assert!(stop.should_stop(0, 6));
    }

    #[test]
    fn stop_condition_eos_tokens() {
        let stop = StopCondition::EosTokens(vec![10, 20]);
        assert!(stop.should_stop(10, 1));
        assert!(stop.should_stop(20, 1));
        assert!(!stop.should_stop(15, 1));
    }

    #[test]
    fn stop_condition_eos_empty_list() {
        let stop = StopCondition::EosTokens(vec![]);
        assert!(!stop.should_stop(0, 1));
        assert!(!stop.should_stop(999, 100));
    }

    #[test]
    fn stop_condition_max_tokens_or_eos() {
        let stop = StopCondition::MaxTokensOrEos {
            max_tokens: 5,
            eos_tokens: vec![99],
        };
        // EOS triggers
        assert!(stop.should_stop(99, 1));
        // Max tokens triggers
        assert!(stop.should_stop(0, 5));
        // Neither triggers
        assert!(!stop.should_stop(0, 3));
    }

    // --- SamplingParams tests ---

    #[test]
    fn sampling_params_default() {
        let p = SamplingParams::default();
        assert_eq!(p.temperature, 1.0);
        assert!(p.seed.is_none());
    }
}
