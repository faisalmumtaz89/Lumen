//! The core inference engine implementing the minimal "atomic" loop.
//!
//! This is the token generation loop: the token generation loop.
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

// SamplingParams is now defined in `crate::sampling`; it carries
// the full top-k / top-p / min-p / repetition-penalty / repeat-last-n
// surface used by both the real sampler and the anti-degeneration
// penalized-greedy). The struct still has `temperature: f32` and
// `seed: Option<u64>` as its first two fields, so existing 2-field
// constructors `SamplingParams { temperature, seed }` continue to compile.
// All other fields are `Option<_>` defaulting to `None`, so call sites
// using `..Default::default()` get the legacy "no shaping, no penalties"
// behavior unchanged.
// unify Xorshift64 + SamplerState + SamplingParams with the
// canonical implementations in `crate::sampling`. Engine-level re-exports
// keep historical import paths working (compute backends, session,
// lumen-cli, lumen-server, tests all import via `engine::`).
pub use crate::sampling::{SamplerState, SamplingParams, Xorshift64};

// softmax_inplace is the canonical impl in `crate::sampling`
// (same SIMD/scalar split, same numerical contract). The re-export below
// preserves historical `engine::softmax_inplace` imports.
pub use crate::sampling::softmax_inplace;

/// Sample a token from logits using temperature scaling and a mutable RNG.
///
/// - temperature <= 0.0: greedy (argmax)
/// - temperature > 0.0: scale logits, softmax, then sample from distribution
///
/// The RNG is passed by `&mut` so its state advances across calls,
/// producing different random draws for each token.
pub fn sample_token(logits: &mut Logits, params: &SamplingParams, rng: &mut Xorshift64) -> u32 {
    // stateless shim now delegates to the unified CPU sampler
    // in `crate::sampling`. Behavior with the legacy 2-field params
    // (temperature, seed; all others None) is byte-identical to the
    // earlier inline softmax/CDF path. Callers that need history-aware
    // anti-degeneration penalties must use `sample_token_with_state` and
    // thread a `SamplerState` across decode steps.
    let mut transient_state = SamplerState::new();
    crate::sampling::sample_logits(&mut logits.data, params, &mut transient_state, rng)
}

/// History-aware token sampler. `state.history` is appended with the
/// selected token before return so the next call's penalty pass sees it.
///
/// this is the function the decode loops call. With penalties
/// active (default CLI: `repetition_penalty = 1.1`, `repeat_last_n = 64`)
/// it breaks MoE loop attractors even at `temperature = 0` (penalized
/// greedy). With penalties disabled it is byte-identical to the legacy
/// `sample_token`.
pub fn sample_token_with_state(
    logits: &mut Logits,
    params: &SamplingParams,
    state: &mut SamplerState,
    rng: &mut Xorshift64,
) -> u32 {
    let token = crate::sampling::sample_logits(&mut logits.data, params, state, rng);
    state.record(token);
    token
}

/// True iff the decode loop must take the logits-readback CPU sampling
/// path. The GPU-resident greedy fast path (`decode_token_greedy`) does
/// GPU argmax with a 4-byte readback and never returns logits to the CPU,
/// so a history-aware penalty cannot be applied to its output.
/// disables this fast path when ANY penalty is active so penalized-greedy
/// works correctly even at `temperature <= 0`.
///
/// It ALSO disables the fast path when `anti_restate` is on: the greedy
/// anti-degeneration veto (`guarded_argmax`, applied on the CPU sampling path)
/// inspects the emitted-token history and can override the raw argmax, which the
/// GPU-argmax readback cannot do. Without this term the veto was silently
/// BYPASSED on GPU-argmax backends (Metal, `gpu_argmax=true`) while always
/// applied on backends that read logits back to the CPU (CUDA, `gpu_argmax=
/// false`) — an asymmetry that left the BF16-MoE greedy path on Metal without
/// the veto. `anti_restate` is NOT part of `penalties_active()` (it is a
/// deterministic post-argmax veto, not a logit penalty), so it must be tested
/// explicitly here. Dense models keep `anti_restate=false`, so this term leaves
/// every non-MoE greedy path byte-identical, and CUDA was already on the CPU
/// path (`gpu_argmax=false`) so its behaviour is unchanged.
pub fn use_gpu_greedy_predicate(params: &SamplingParams, gpu_resident: bool, gpu_argmax: bool) -> bool {
    gpu_resident
        && gpu_argmax
        && params.temperature <= 0.0
        && !params.penalties_active()
        && !params.anti_restate
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
    /// Optional per-token-id byte decoder for the greedy anti-restate guard's
    /// sub-word-doubling rule. Installed by the CLI (which owns a
    /// `BpeTokenizer`) via [`InferenceEngine::set_token_decoder`]; the engine
    /// copies it onto each generation's `SamplerState`. `None` leaves the
    /// byte-level rule disabled (the id-only n-gram restate rule still runs).
    token_decoder: Option<crate::sampling::TokenByteDecoder>,
}

impl InferenceEngine {
    /// Create a new inference engine with the given configuration.
    pub fn new(config: RuntimeConfig, hyperparams: ModelHyperparams) -> Self {
        Self {
            config,
            hyperparams,
            token_decoder: None,
        }
    }

    /// Install the per-token-id byte decoder used by the greedy anti-restate
    /// guard's sub-word-doubling rule. The CLI calls this once after building
    /// the engine and its tokenizer. A no-op for the id-only n-gram rule,
    /// which never consults the decoder.
    pub fn set_token_decoder(&mut self, decoder: crate::sampling::TokenByteDecoder) {
        self.token_decoder = Some(decoder);
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

        // Validate KV precision against backend capabilities before allocating
        // the KV cache. Metal pins KV to F16; CUDA pins KV to F32. Without
        // this check a mismatched config would silently corrupt KV writes
        backend.validate_kv_precision(self.config.kv_precision)?;

        // library-side prompt-length guard. The engine path
        // is used by both the CLI direct call and tests; a too-long prompt
        // here would otherwise reach the per-layer prefill kernel and write
        // off the end of the GPU KV buffer (UB on Metal, OOB on CUDA). Reject
        // up front with the same actionable message the server N18 emits.
        if prompt_tokens.len() > self.config.max_seq_len {
            return Err(RuntimeError::Compute(format!(
                "prompt is {} tokens but max_seq_len is {}; reduce the prompt or \
                 restart with a larger --context-len",
                prompt_tokens.len(),
                self.config.max_seq_len,
            )));
        }

        // Initialize RNG once for the entire generation session.
        let mut rng = Xorshift64::new(sampling.seed.unwrap_or(42));

        // SamplerState seeded with the prompt so the rolling
        // repeat-last-n window sees recent context tokens. Penalized-greedy
        // then suppresses tokens that already appear in the most recent
        // window.
        let mut sampler_state = SamplerState::new();
        if let Some(ref d) = self.token_decoder {
            sampler_state.set_decoder(d.clone());
        }
        for &t in prompt_tokens.iter() {
            sampler_state.record(t);
        }

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
            if std::env::var("LUMEN_PREFILL_DEBUG").is_ok() {
                let n = prompt_tokens.len();
                let head: Vec<u32> = prompt_tokens.iter().take(12).copied().collect();
                let tail: Vec<u32> = prompt_tokens.iter().rev().take(6).rev().copied().collect();
                let hsum: f64 = last_hidden.iter().map(|&v| v as f64).sum();
                let hmin = last_hidden.iter().cloned().fold(f32::INFINITY, f32::min);
                let hmax = last_hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[prefill-debug] generate prompt_len={n} head={head:?} tail={tail:?} hidden_len={} hidden_sum={hsum:.4} hidden_min={hmin:.4} hidden_max={hmax:.4}",
                    last_hidden.len()
                );
            }
            let mut current_x = ActivationBuffer::zeros(last_hidden.len(), ComputeDtype::F32);
            current_x.write_f32_from(&last_hidden);
            let lg = backend.compute_final(&current_x)?;
            if std::env::var("LUMEN_PREFILL_DEBUG").is_ok() {
                let argmax = lg.data.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i).unwrap_or(0);
                eprintln!("[prefill-debug] generate first_logits_argmax={argmax} logit_val={:.4}", lg.data.get(argmax).copied().unwrap_or(0.0));
            }
            lg
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

        // first token comes from the prefill logits; history-aware
        // penalties + history append happen via sample_token_with_state.
        let mut next_token = sample_token_with_state(&mut logits, sampling, &mut sampler_state, &mut rng);
        generated_tokens.push(next_token);

        // GPU-argmax fast path requires NO active penalty (it
        // returns 4-byte argmax, the CPU never sees logits). With a penalty
        // we must route through decode_token() (now also F1-fixed: real
        // logits readback) so the penalty can be applied on CPU.
        let use_gpu_greedy = use_gpu_greedy_predicate(sampling, caps.gpu_resident, caps.gpu_argmax);

        // DIAGNOSTIC (env LUMEN_FORCE_PREFILL_DECODE=1, default OFF): generate
        // every subsequent token by re-prefilling the full sequence from scratch
        // (fresh KV + reset recurrent state) via the batched prefill path only --
        // `decode_token` is never called. Isolates whether the prefill path is a
        // coherent generator vs. the single-token decode path (CUDA MoE-35B
        // router-flip root cause). Remove before commit.
        let force_prefill_decode =
            std::env::var("LUMEN_FORCE_PREFILL_DECODE").as_deref() == Ok("1");

        if force_prefill_decode {
            while !stop.should_stop(next_token, generated_tokens.len()) {
                let mut full: Vec<u32> = prompt_tokens.to_vec();
                full.extend_from_slice(&generated_tokens);
                kv.truncate_to(0);
                backend.reset_recurrent_state();
                let last_hidden = backend.prefill(&full, weights, &mut kv)?;
                let mut cx = ActivationBuffer::zeros(last_hidden.len(), ComputeDtype::F32);
                cx.write_f32_from(&last_hidden);
                logits = backend.compute_final(&cx)?;
                next_token =
                    sample_token_with_state(&mut logits, sampling, &mut sampler_state, &mut rng);
                generated_tokens.push(next_token);
            }
        } else if use_gpu_greedy {
            // GPU-RESIDENT GREEDY fast path: argmax on GPU, 4-byte readback.
            // decode_token_greedy() advances kv.seq_len() internally.
            while !stop.should_stop(next_token, generated_tokens.len()) {
                next_token = backend.decode_token_greedy(next_token, weights, &mut kv)?;
                sampler_state.record(next_token);
                generated_tokens.push(next_token);
            }
        } else if caps.gpu_resident {
            // GPU-RESIDENT fast path: single command buffer per token.
            // decode_token() advances kv.seq_len() internally.
            while !stop.should_stop(next_token, generated_tokens.len()) {
                logits = backend.decode_token(next_token, weights, &mut kv)?;
                next_token = sample_token_with_state(&mut logits, sampling, &mut sampler_state, &mut rng);
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
                next_token = sample_token_with_state(&mut logits, sampling, &mut sampler_state, &mut rng);
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
            // surface live KV cache stats so the CLI banner
            // and operators can observe context utilisation at end-of-gen.
            kv: crate::telemetry::KvCacheStats::from_kv(&kv),
            // peak memory snapshot at end-of-gen. Backend
            // returns 0 if not instrumented (default impl).
            peak_memory_bytes: backend.peak_memory_bytes(),
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

        // Validate KV precision against the decode backend (the prefill side
        // shares the same `KvCache` byte buffers; the decode backend's
        // storage contract is what matters).
        backend.validate_kv_precision(self.config.kv_precision)?;

        // library-side prompt-length guard. See generate()
        // for rationale.
        if prompt_tokens.len() > self.config.max_seq_len {
            return Err(RuntimeError::Compute(format!(
                "prompt is {} tokens but max_seq_len is {}; reduce the prompt or \
                 restart with a larger --context-len",
                prompt_tokens.len(),
                self.config.max_seq_len,
            )));
        }

        let mut rng = Xorshift64::new(sampling.seed.unwrap_or(42));

        // SamplerState seeded with the prompt for the
        // repeat-last-n window. See generate() for rationale.
        let mut sampler_state = SamplerState::new();
        if let Some(ref d) = self.token_decoder {
            sampler_state.set_decoder(d.clone());
        }
        for &t in prompt_tokens.iter() {
            sampler_state.record(t);
        }

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
        let mut next_token = sample_token_with_state(&mut logits, sampling, &mut sampler_state, &mut rng);
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
            next_token = sample_token_with_state(&mut logits, sampling, &mut sampler_state, &mut rng);
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
            // surface live KV cache stats so the CLI banner
            // and operators can observe context utilisation at end-of-gen.
            kv: crate::telemetry::KvCacheStats::from_kv(&kv),
            // peak memory snapshot at end-of-gen. Backend
            // returns 0 if not instrumented (default impl).
            peak_memory_bytes: backend.peak_memory_bytes(),
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

    /// Run token generation with a caller-supplied [`crate::session::Session`].
    ///
    /// This is the entry point used by the CLI's `--session-resume` /
    /// `--session-save` flow. The caller passes in a
    /// `Session` that is either (a) freshly constructed with the same
    /// `RuntimeConfig` + hyperparams + sampling that the engine carries, or
    /// (b) loaded from disk via [`crate::session::Session::load_from_disk`].
    ///
    /// Generation flow:
    /// 1. If the session is empty AND the prompt is non-empty, call
    ///    [`Session::extend`] (full cold prefill). If the session already
    ///    has tokens (resumed-from-disk case), call
    ///    [`Session::extend_with_cache`] so the common prefix is reused —
    ///    this is what makes `--session-resume foo.kv` actually skip the
    ///    cold prefill on a continuing conversation.
    /// 2. Stream tokens via [`Session::stream`] respecting the stop condition.
    ///
    /// Returns the same [`GenerationResult`] shape as [`Self::generate`], so
    /// CLI code can treat the two paths interchangeably for reporting.
    ///
    /// `prompt_tokens` is the full prompt the operator typed; for a resumed
    /// session whose first-turn prompt was identical to this prompt's
    /// prefix, the cache-reuse path skips that prefix entirely. For an
    /// empty `prompt_tokens` + resumed session (caller wants to continue
    /// generation from the cached state), the prefill phase is a no-op and
    /// decode runs directly.
    pub fn generate_with_session(
        &self,
        session: &mut crate::session::Session,
        prompt_tokens: &[u32],
        weights: &dyn WeightProvider,
        backend: &dyn ComputeBackend,
        stop: &StopCondition,
    ) -> Result<GenerationResult, RuntimeError> {
        let total_start = Instant::now();

        // Validate KV precision once at top of generation, mirroring the
        // contract `generate()` enforces. Failing here surfaces the same
        // actionable "Metal requires F16" / "CUDA requires F32" message.
        backend.validate_kv_precision(self.config.kv_precision)?;

        // Propagate the anti-restate byte decoder (if the CLI installed one)
        // onto the session so its sub-word-doubling rule can decode candidate
        // token bytes. Idempotent; no-op when no decoder was installed.
        if let Some(ref d) = self.token_decoder {
            session.set_token_decoder(d.clone());
        }

        // library-side prompt-length guard. The session's
        // own `extend` / `extend_with_cache` re-check, but failing here
        // produces the same operator-facing error format as `generate()`.
        if prompt_tokens.len() > self.config.max_seq_len {
            return Err(RuntimeError::Compute(format!(
                "prompt is {} tokens but max_seq_len is {}; reduce the prompt \
                 or restart with a larger --context-len",
                prompt_tokens.len(),
                self.config.max_seq_len,
            )));
        }

        // ---- Prefill ----
        let prefill_start = Instant::now();
        let suffix_threshold = crate::session::Session::resolve_suffix_threshold();
        if !prompt_tokens.is_empty() {
            if session.token_count() == 0 {
                // Cold path: a fresh session sees the whole prompt; same
                // behaviour as `engine.generate(prompt, ...)`.
                session.extend(prompt_tokens, backend, weights)?;
            } else {
                // Resumed-from-disk path: try to reuse the cached prefix
                // for any common-prefix tokens, fall back to cold restart
                // on divergence (the standard `extend_with_cache` matrix).
                session.extend_with_cache(
                    prompt_tokens,
                    backend,
                    weights,
                    suffix_threshold,
                )?;
            }
        } else if session.token_count() == 0 {
            // Edge case: empty prompt on a fresh session. Mirror the
            // `engine.generate` behaviour and reject up front; there's
            // nothing to generate from.
            return Err(RuntimeError::Compute(
                "empty prompt and empty session: nothing to generate from".into(),
            ));
        }
        let prefill_time = prefill_start.elapsed();

        // ---- Decode ----
        let decode_start = Instant::now();
        // The stream wrapper handles MaxTokens / MaxTokensOrEos / EosTokens
        // by feeding `eos_tokens` to its iterator and capping at
        // `max_tokens`. Mirror the stop-condition unpack here.
        let (max_tokens, eos_tokens): (usize, Vec<u32>) = match stop {
            StopCondition::MaxTokens(n) => (*n, Vec::new()),
            StopCondition::EosTokens(eos) => (usize::MAX, eos.clone()),
            StopCondition::MaxTokensOrEos { max_tokens, eos_tokens } => {
                (*max_tokens, eos_tokens.clone())
            }
        };
        let mut generated_tokens: Vec<u32> = Vec::new();
        // Track the pre-stream count so the generated-only slice excludes
        // the prompt.
        let prompt_len = session.token_count();
        for r in session.stream(backend, weights, max_tokens, &eos_tokens) {
            generated_tokens.push(r?);
        }
        // The stream wrapper appends generated tokens to the session's
        // `tokens` vec via `next_token`; the local `generated_tokens` vec
        // is what we return so the caller can decode it without the
        // prompt.
        let _ = prompt_len;
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
            per_layer_timings: session.take_timings(),
            // live KV stats at end of generation.
            kv: crate::telemetry::KvCacheStats::from_kv(session.kv()),
            // peak memory snapshot at end-of-gen.
            peak_memory_bytes: backend.peak_memory_bytes(),
            ..Default::default()
        };

        Ok(GenerationResult {
            tokens: generated_tokens,
            metrics,
        })
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
        let params = SamplingParams { temperature: 1.0, seed: None, ..Default::default() };
        let mut rng = Xorshift64::new(42);
        assert_eq!(sample_token(&mut logits, &params, &mut rng), 0);
    }

    #[test]
    fn sample_token_greedy_returns_argmax() {
        let mut logits = Logits { data: vec![1.0, 5.0, 3.0, 2.0] };
        let params = SamplingParams { temperature: 0.0, seed: None, ..Default::default() };
        let mut rng = Xorshift64::new(42);
        assert_eq!(sample_token(&mut logits, &params, &mut rng), 1);
    }

    #[test]
    fn sample_token_deterministic_with_seed() {
        let params = SamplingParams { temperature: 0.8, seed: Some(42), ..Default::default() };

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

    // --- F10: GPU-greedy predicate must yield to the anti_restate veto ---

    #[test]
    fn gpu_greedy_predicate_disabled_by_anti_restate_on_every_backend() {
        // With anti_restate=true, temperature=0, rep=1.0 (no penalty active),
        // the GPU-argmax fast path MUST be disabled regardless of the backend
        // caps, so the CPU `guarded_argmax` veto is forced on. Previously the
        // predicate was TRUE for Metal (gpu_argmax=true) — silently bypassing
        // the veto — while CUDA (gpu_argmax=false) always took the CPU path.
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: Some(1.0), // no penalty active
            anti_restate: true,
            ..Default::default()
        };
        // Sanity: rep=1.0 means no logit penalty is active, so ONLY the new
        // anti_restate term can be what disables the fast path here.
        assert!(!params.penalties_active());
        for gpu_resident in [false, true] {
            for gpu_argmax in [false, true] {
                assert!(
                    !use_gpu_greedy_predicate(&params, gpu_resident, gpu_argmax),
                    "anti_restate=true must disable GPU greedy for \
                     (gpu_resident={gpu_resident}, gpu_argmax={gpu_argmax})"
                );
            }
        }
    }

    #[test]
    fn gpu_greedy_predicate_unaffected_when_anti_restate_off() {
        // Dense models (anti_restate=false) keep the fast path on the gpu_resident
        // + gpu_argmax + temp<=0 + no-penalty cell — byte-identical to before the
        // F10 term was added. This also confirms CUDA's CPU-path behaviour
        // (gpu_argmax=false) is unchanged.
        let greedy = SamplingParams {
            temperature: 0.0,
            repetition_penalty: Some(1.0),
            anti_restate: false,
            ..Default::default()
        };
        // The fast path is ON only when gpu_resident && gpu_argmax (the
        // pre-existing terms), and OFF otherwise.
        assert!(use_gpu_greedy_predicate(&greedy, true, true));
        assert!(!use_gpu_greedy_predicate(&greedy, true, false)); // CUDA decode caps
        assert!(!use_gpu_greedy_predicate(&greedy, false, true));
        assert!(!use_gpu_greedy_predicate(&greedy, false, false));
    }
}
