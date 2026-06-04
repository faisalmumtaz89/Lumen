//! Token sampling: parameters, history-aware penalties, distribution shaping.
//!
//! The pipeline order matches the standard formulation used across
//! mainstream reference implementations:
//!
//! ```text
//!   raw logits -> penalties -> top-k -> temperature -> softmax -> top-p -> min-p -> sample
//! ```
//!
//! Penalties run on raw logits (before softmax) because they shift relative
//! scores. Top-k truncates the candidate set before softmax to bound the cost
//! of nucleus / min-p shaping. Top-p and min-p run on probabilities.
//!
//! # GPU porting contract
//!
//! `Sampler::sample(&mut self, logits: &mut [f32]) -> u32` is the kernel
//! boundary. A future CUDA implementation reads `self.params` and
//! `self.state` (already on the CPU side) and writes the selected token
//! id. The CPU code below is the reference spec for that kernel.
//!
//! # Primitives
//!
//! `Xorshift64` and `softmax_inplace` live in this module -- they
//! are exported back from `engine` for callers that import them via the
//! historical path (`use crate::engine::{Xorshift64, softmax_inplace};`).

use std::collections::HashMap;

/// Minimal xorshift64 PRNG -- deterministic, zero deps.
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Construct from a u64 seed.
    ///
    /// Small seeds (notably the CLI's default `42`)
    /// produce a first `next_u64()` whose top 24 bits are zero, making
    /// `next_f32()` return exactly `0.0` on the first draw. Under the
    /// strict `r < cumsum` inverse-CDF in `sample_categorical`, `r=0.0`
    /// would deterministically pick index 0 (typically a control / byte
    /// token) for the first sampled token of every default-seed
    /// invocation. We avalanche the state through a single MurmurHash-3
    /// finalizer (splitmix-style; one mul + one xorshift) which provides
    /// a high-quality bit mix in O(1) — the standard fix for `xorshift64`
    /// cold-start. `xorshift64_determinism` is preserved (the
    /// transformation is deterministic) and any two `Xorshift64`s built
    /// from the same seed remain bit-identical from `next_u64()` onward.
    pub fn new(seed: u64) -> Self {
        // Ensure non-zero state, then mix.
        let base = if seed == 0 { 0xDEAD_BEEF_CAFE_BABEu64 } else { seed };
        // SplitMix64 finalizer (Stafford Mix13) — well-tested public-domain
        // avalanche, used by Java SplitableRandom and Rust's smallrng init.
        let mut x = base.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        // Guard: SplitMix can theoretically reach 0; map it to the same
        // non-zero fallback as the `seed == 0` case so Xorshift64 doesn't
        // get stuck at 0.
        let state = if x == 0 { 0xDEAD_BEEF_CAFE_BABEu64 } else { x };
        Self { state }
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

/// Sampling parameters for token generation.
///
/// All new fields are `Option<_>` so the default behaviour matches the
/// legacy two-field struct: temperature scaling + multinomial sampling
/// with no shaping or penalties.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Softmax temperature. `<= 0.0` selects greedy (argmax) and skips
    /// every other shaping step.
    pub temperature: f32,

    /// Optional RNG seed. `None` -> use a default constant (42).
    pub seed: Option<u64>,

    /// Nucleus sampling: keep the smallest prefix of sorted tokens whose
    /// cumulative probability is `>= top_p`, then renormalize. Range `(0, 1]`.
    pub top_p: Option<f32>,

    /// Keep only the `top_k` largest logits before softmax. `0` disables
    /// the cut.
    pub top_k: Option<usize>,

    /// Drop tokens whose probability is below `min_p * max_prob`. Range
    /// `[0, 1]`. Applied after `top_p`.
    pub min_p: Option<f32>,

    /// Repetition penalty applied to logits of tokens that already appear
    /// in `SamplerState::history`. `1.0` is a no-op; values `>1` suppress
    /// repeats. Sign-aware: positive logits are divided, negative logits
    /// are multiplied (the standard convention).
    pub repetition_penalty: Option<f32>,

    /// Flat additive penalty per unique token in history. The logit of any
    /// previously-seen token is reduced by `presence_penalty`. `0.0` is a
    /// no-op.
    pub presence_penalty: Option<f32>,

    /// Additive penalty proportional to per-token frequency in history:
    /// `logit[t] -= frequency_penalty * count[t]`. `0.0` is a no-op.
    pub frequency_penalty: Option<f32>,

    /// rolling window for repetition/presence/frequency penalties.
    /// Penalties only consider the last `repeat_last_n` tokens of
    /// `SamplerState::history` (a rolling penalty window over the last N
    /// tokens). `None` or `Some(0)` -> consider the full history. `Some(64)`
    /// is a sensible anti-degeneration window for Lumen MoE.
    pub repeat_last_n: Option<usize>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            seed: None,
            top_p: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
            repeat_last_n: None,
        }
    }
}

impl SamplingParams {
    /// Returns true iff any history-aware penalty is active. Used by the
    /// engine's decode-path predicate to disable the GPU-argmax fast path
    /// (which never returns logits to the CPU) when a penalty needs to be
    /// applied to the logits before token selection.
    pub fn penalties_active(&self) -> bool {
        let rep = self.repetition_penalty.unwrap_or(1.0);
        let presence = self.presence_penalty.unwrap_or(0.0);
        let freq = self.frequency_penalty.unwrap_or(0.0);
        (rep - 1.0).abs() >= f32::EPSILON || presence != 0.0 || freq != 0.0
    }
}

/// History accumulated across sample calls in the same sequence.
///
/// The `freq` map lets `frequency_penalty` scale by occurrence count
/// without rescanning `history` each step.
#[derive(Debug, Default, Clone)]
pub struct SamplerState {
    pub history: Vec<u32>,
    pub freq: HashMap<u32, u32>,
}

impl SamplerState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `token` to history and bump its frequency counter.
    pub fn record(&mut self, token: u32) {
        self.history.push(token);
        *self.freq.entry(token).or_insert(0) += 1;
    }

    /// Reset between sequences.
    pub fn clear(&mut self) {
        self.history.clear();
        self.freq.clear();
    }
}

/// Stateful sampler. Holds params, history, and the RNG together so a
/// future CUDA port has a single object to shadow.
pub struct Sampler {
    pub params: SamplingParams,
    pub state: SamplerState,
    pub rng: Xorshift64,
}

impl Sampler {
    /// Construct from params, seeding the RNG from `params.seed` (default 42).
    pub fn new(params: SamplingParams) -> Self {
        let seed = params.seed.unwrap_or(42);
        Self {
            params,
            state: SamplerState::new(),
            rng: Xorshift64::new(seed),
        }
    }

    /// Construct with a caller-provided RNG (used by the legacy shim).
    pub fn with_rng(params: SamplingParams, rng: Xorshift64) -> Self {
        Self { params, state: SamplerState::new(), rng }
    }

    /// Sample one token, updating `state` so subsequent calls see this
    /// token in history.
    ///
    /// `logits` is mutated in place (penalties, top-k mask, temperature,
    /// softmax). After this returns, the slice holds either junk or a
    /// probability distribution -- callers must not depend on it.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        let token = sample_logits(logits, &self.params, &mut self.state, &mut self.rng);
        self.state.record(token);
        token
    }
}

/// Core CPU sampling routine.
///
/// Public so the legacy `sample_token` shim in `engine.rs` (which threads
/// an externally-owned RNG) can call it without instantiating a full
/// `Sampler`. Does NOT update `state.history` -- callers that want
/// history-aware penalties must do so explicitly. `Sampler::sample` is
/// the wrapper that does both.
///
/// semantics for `temperature <= 0.0`:
/// - If no penalties are active -> returns `argmax(logits)` directly
///   (legacy fast path, byte-identical to original callers).
/// - If ANY penalty is active -> applies repetition/presence/frequency
///   penalties (windowed by `repeat_last_n`) THEN returns the argmax of
///   the penalized logits ("penalized greedy"). This is the
///   anti-degeneration mode used to break MoE near-tie loop attractors
///   without changing kernel numerics.
pub fn sample_logits(
    logits: &mut [f32],
    params: &SamplingParams,
    state: &mut SamplerState,
    rng: &mut Xorshift64,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // Greedy path: apply penalties (if any) THEN argmax. With penalties
    // disabled this returns the raw argmax in O(n) -- the legacy fast path.
    // the CLI now defaults penalties OFF so this branch is
    // pure greedy by default. With penalties explicitly enabled it becomes
    // "penalized greedy" — kept available as an explicit opt-in but no
    // longer the default.
    if params.temperature <= 0.0 {
        if params.penalties_active() {
            apply_penalties(logits, params, state);
        }
        return argmax(logits) as u32;
    }

    // Standard sampler-chain ordering.
    //
    // Lumen implements TOP_K, TOP_P, MIN_P on raw logits / temp=1
    // probabilities (PRE-temperature), then temperature applies LAST
    // before the final softmax + multinomial draw. This is the published
    // convention for nucleus + min-p + temp sampling. Lumen used to apply
    // temperature BEFORE softmax which made top_p / min_p operate on
    // temperature-
    // skewed probabilities and skewed sampling-quality comparisons.
    //
    // 1. Penalties on raw logits.
    apply_penalties(logits, params, state);

    // 2. Top-k mask on raw logits (rank-based; temperature-invariant).
    if let Some(k) = params.top_k {
        if k > 0 && k < logits.len() {
            apply_top_k(logits, k);
        }
    }

    // 3. Top-p (nucleus) on raw logits. Filter at temp=1: compute
    //    softmax-at-temp-1 in scratch, find the kept indices, mask the
    //    *logits* outside the kept set to -inf. The user's temperature
    //    applies in step 5.
    if let Some(p) = params.top_p {
        if p > 0.0 && p < 1.0 {
            apply_top_p_on_logits(logits, p);
        }
    }

    // 4. Min-p relative cutoff on raw logits (`log(min_p)` shift in logit
    //    space — equivalent to filtering probs at temp=1).
    if let Some(m) = params.min_p {
        if m > 0.0 {
            apply_min_p_on_logits(logits, m);
        }
    }

    // 5. Temperature scale (applied LAST, AFTER all filters).
    let inv_temp = 1.0 / params.temperature;
    for v in logits.iter_mut() {
        *v *= inv_temp;
    }

    // 6. Final softmax -> probabilities.
    softmax_inplace(logits);

    // 7. Multinomial draw via cumulative-sum walk.
    sample_categorical(logits, rng)
}

/// Argmax with deterministic NaN handling. Returns 0 on empty input.
fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Apply repetition / presence / frequency penalties to raw logits.
///
/// Sign-aware repetition penalty (standard convention):
/// `logit > 0 -> divide`, `logit <= 0 -> multiply`. This keeps the
/// penalty direction consistent regardless of logit sign.
///
/// honors `params.repeat_last_n`. When set to `Some(n)` with `n > 0`,
/// only the last `n` tokens of `state.history` contribute to the penalty
/// (rolling window). `None` or `Some(0)` falls back to the full history
/// (the freq map). The windowed branch rebuilds a small frequency map on
/// the fly over at most `n` tokens, which is O(n) per decode step -- cheap
/// for n=64 (the recommended default).
fn apply_penalties(logits: &mut [f32], params: &SamplingParams, state: &SamplerState) {
    let rep = params.repetition_penalty.unwrap_or(1.0);
    let presence = params.presence_penalty.unwrap_or(0.0);
    let freq = params.frequency_penalty.unwrap_or(0.0);

    if (rep - 1.0).abs() < f32::EPSILON && presence == 0.0 && freq == 0.0 {
        return;
    }
    if state.history.is_empty() {
        return;
    }

    // Build the active-window frequency map. With a window we re-derive it
    // each step (the persistent `state.freq` map covers full history).
    let window: Option<usize> = params.repeat_last_n.filter(|&n| n > 0);
    if let Some(n) = window {
        let start = state.history.len().saturating_sub(n);
        let mut local: HashMap<u32, u32> = HashMap::with_capacity(n.min(state.history.len()));
        for &tok in &state.history[start..] {
            *local.entry(tok).or_insert(0) += 1;
        }
        for (&tok, &count) in local.iter() {
            apply_penalty_one(logits, tok, count, rep, presence, freq);
        }
    } else {
        for (&tok, &count) in state.freq.iter() {
            apply_penalty_one(logits, tok, count, rep, presence, freq);
        }
    }
}

#[inline]
fn apply_penalty_one(
    logits: &mut [f32],
    tok: u32,
    count: u32,
    rep: f32,
    presence: f32,
    freq: f32,
) {
    let idx = tok as usize;
    if idx >= logits.len() {
        return;
    }
    let v = logits[idx];
    let mut new_v = if (rep - 1.0).abs() >= f32::EPSILON {
        if v > 0.0 { v / rep } else { v * rep }
    } else {
        v
    };
    new_v -= presence;
    new_v -= freq * count as f32;
    logits[idx] = new_v;
}

/// Mask all logits outside the top-K to `-inf`. Caller clamps to
/// `0 < k < len`.
fn apply_top_k(logits: &mut [f32], k: usize) {
    // Find the k-th largest value via a sort on a copy. O(n log n) -- the
    // spec is correctness-first; a quickselect refactor is a later
    // optimization.
    let mut sorted: Vec<f32> = logits.iter().copied().collect();
    sorted.sort_by(|a, b| b.total_cmp(a)); // descending
    let cutoff = sorted[k - 1];
    for v in logits.iter_mut() {
        if v.total_cmp(&cutoff).is_lt() {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Top-p filter applied to LOGITS at temp=1.
///
/// Computes a softmax-at-temp-1 in a scratch buffer to identify the
/// nucleus, then masks every logit outside the nucleus to `-inf`.
/// Logits inside the nucleus are unchanged so the user's temperature
/// (applied after this in `sample_logits`) sees them at their original
/// magnitude. `top_p` runs before `temperature` in the standard chain.
fn apply_top_p_on_logits(logits: &mut [f32], p: f32) {
    let n = logits.len();
    if n == 0 {
        return;
    }
    // Softmax in a scratch buffer (don't mutate caller's logits).
    let mut probs: Vec<f32> = logits.to_vec();
    softmax_inplace(&mut probs);

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a])); // descending by prob

    let mut cum = 0.0f32;
    let mut kept = 0usize;
    for (rank, &i) in idx.iter().enumerate() {
        cum += probs[i];
        kept = rank + 1;
        if cum >= p {
            break;
        }
    }
    // Mask the discarded set in the LOGITS to -inf.
    for &i in idx.iter().skip(kept) {
        logits[i] = f32::NEG_INFINITY;
    }
}

/// Min-p filter applied to LOGITS in log space.
///
/// Min-p relative cutoff in log space: for each token, keep iff
/// `prob(token) >= min_p * prob(max)`. In log space this is
/// `logit(token) >= max_logit + log(min_p)` — no softmax needed,
/// purely a comparison on raw logits. Tokens that fail the bound are
/// masked to `-inf`.
fn apply_min_p_on_logits(logits: &mut [f32], min_p: f32) {
    if logits.is_empty() {
        return;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return;
    }
    // log(min_p) is well-defined for `0 < min_p <= 1`. Callers already
    // gate `min_p > 0.0`, and `min_p >= 1.0` would zero everything but
    // the argmax (degenerate); we follow the same semantic.
    let log_min_p = min_p.ln();
    let bound = max_logit + log_min_p;
    for v in logits.iter_mut() {
        if *v < bound {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Draw one index from a probability distribution via inverse-CDF walk.
///
/// Strict `r < cumsum` is the correct inverse-CDF semantic: bucket `i`
/// owns `[cumsum_{i-1}, cumsum_i)`. The RNG-edge fix lives in
/// `Xorshift64::new` (state warm-up) so that `r==0.0` on the first
/// draw under the default seed no longer occurs — see the Xorshift
/// constructor for the rationale.
fn sample_categorical(probs: &[f32], rng: &mut Xorshift64) -> u32 {
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }
    // Floating-point slack -- fall through to the last index.
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ---- greedy preservation ----

    #[test]
    fn greedy_returns_argmax_with_all_default_params() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        let mut s = Sampler::new(SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        });
        assert_eq!(s.sample(&mut logits), 1);
    }

    #[test]
    fn greedy_ignores_distribution_shaping_but_applies_penalties() {
        // contract change for anti-degeneration fix. The greedy
        // fast path now applies repetition/presence/frequency penalties (so
        // a temp=0 caller can break MoE loop attractors at zero kernel cost)
        // but still skips top_p / top_k / min_p, which only shape a
        // multinomial draw. With history={1} and presence=10, logits[1] is
        // pushed from 5.0 to -7.0 (5.0 -> 5/2 + (-10) + (-10) = -17.5), so
        // token 2 (logit 3.0) becomes argmax.
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let mut s = Sampler::new(SamplingParams {
            temperature: 0.0,
            top_p: Some(0.5),
            top_k: Some(1),
            min_p: Some(0.9),
            repetition_penalty: Some(2.0),
            presence_penalty: Some(10.0),
            frequency_penalty: Some(10.0),
            ..SamplingParams::default()
        });
        s.state.record(1);
        // Penalized greedy: top_p/top_k/min_p are irrelevant; penalty drives
        // the choice off token 1.
        assert_eq!(s.sample(&mut logits.clone()), 2);
    }

    #[test]
    fn greedy_unaffected_when_penalties_disabled() {
        // With NO penalties set, greedy is bit-exact when disabled:
        // returns the raw argmax regardless of top_p/top_k/min_p.
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let mut s = Sampler::new(SamplingParams {
            temperature: 0.0,
            top_p: Some(0.5),
            top_k: Some(1),
            min_p: Some(0.9),
            ..SamplingParams::default()
        });
        s.state.record(1);
        assert_eq!(s.sample(&mut logits.clone()), 1);
    }

    #[test]
    fn greedy_with_repeat_last_n_windows_penalty() {
        // rolling-window contract. With repeat_last_n=2 and history
        // [0, 0, 1, 1], only the last 2 tokens (both 1s) contribute, so
        // token 0 is NOT penalized. logits = [3.0, 3.5, ...], freq=10 on
        // token 1 drops it to 3.5 - 20 = -16.5 -> token 0 wins.
        let logits = vec![3.0, 3.5];
        let mut s = Sampler::new(SamplingParams {
            temperature: 0.0,
            frequency_penalty: Some(10.0),
            repeat_last_n: Some(2),
            ..SamplingParams::default()
        });
        s.state.record(0);
        s.state.record(0);
        s.state.record(1);
        s.state.record(1);
        assert_eq!(s.sample(&mut logits.clone()), 0);
    }

    #[test]
    fn windowed_penalty_ignores_tokens_outside_window() {
        // Same history but repeat_last_n=1 -> only the most recent token (1)
        // is penalized. With freq=10 on count=1, logit[1] = 3.5 - 10 = -6.5.
        // Token 0 wins via logit 3.0.
        let logits = vec![3.0, 3.5];
        let mut s = Sampler::new(SamplingParams {
            temperature: 0.0,
            frequency_penalty: Some(10.0),
            repeat_last_n: Some(1),
            ..SamplingParams::default()
        });
        s.state.record(0);
        s.state.record(0);
        s.state.record(0);
        s.state.record(1);
        assert_eq!(s.sample(&mut logits.clone()), 0);
    }

    // ---- determinism ----

    #[test]
    fn determinism_same_seed_same_output() {
        let logits_base: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let params = || SamplingParams {
            temperature: 0.7,
            seed: Some(0xC0FFEE),
            top_p: Some(0.9),
            top_k: Some(40),
            ..SamplingParams::default()
        };
        let mut s1 = Sampler::new(params());
        let mut s2 = Sampler::new(params());
        for _ in 0..100 {
            let mut l1 = logits_base.clone();
            let mut l2 = logits_base.clone();
            assert_eq!(s1.sample(&mut l1), s2.sample(&mut l2));
        }
    }

    // ---- top-k in isolation ----

    #[test]
    fn top_k_one_acts_as_greedy() {
        let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: Some(1),
            seed: Some(1),
            ..SamplingParams::default()
        });
        for _ in 0..50 {
            assert_eq!(s.sample(&mut logits.clone()), 1);
        }
    }

    #[test]
    fn top_k_zero_disables() {
        // k=0 should match the no-top-k path.
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut s_off = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: Some(0),
            seed: Some(99),
            ..SamplingParams::default()
        });
        let mut s_none = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: None,
            seed: Some(99),
            ..SamplingParams::default()
        });
        for _ in 0..20 {
            assert_eq!(
                s_off.sample(&mut logits.clone()),
                s_none.sample(&mut logits.clone())
            );
        }
    }

    #[test]
    fn top_k_larger_than_vocab_is_noop() {
        let logits = vec![1.0, 2.0, 3.0];
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: Some(10),
            seed: Some(6),
            ..SamplingParams::default()
        });
        let t = s.sample(&mut logits.clone()) as usize;
        assert!(t < logits.len());
    }

    // ---- top-p in isolation ----

    #[test]
    fn top_p_keeps_only_nucleus() {
        // Token 3 has ~99% probability after softmax.
        let logits = vec![0.0, 0.0, 0.0, 10.0];
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_p: Some(0.5),
            seed: Some(2),
            ..SamplingParams::default()
        });
        for _ in 0..50 {
            assert_eq!(s.sample(&mut logits.clone()), 3);
        }
    }

    #[test]
    fn top_p_one_is_noop() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut s_one = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_p: Some(1.0),
            seed: Some(8),
            ..SamplingParams::default()
        });
        let mut s_none = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_p: None,
            seed: Some(8),
            ..SamplingParams::default()
        });
        for _ in 0..30 {
            assert_eq!(
                s_one.sample(&mut logits.clone()),
                s_none.sample(&mut logits.clone())
            );
        }
    }

    // ---- min-p in isolation ----

    #[test]
    fn min_p_drops_low_probability_tokens() {
        let logits = vec![0.0, 0.0, 0.0, 10.0];
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            min_p: Some(0.5),
            seed: Some(3),
            ..SamplingParams::default()
        });
        for _ in 0..50 {
            assert_eq!(s.sample(&mut logits.clone()), 3);
        }
    }

    #[test]
    fn min_p_zero_is_noop() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut s_zero = Sampler::new(SamplingParams {
            temperature: 1.0,
            min_p: Some(0.0),
            seed: Some(9),
            ..SamplingParams::default()
        });
        let mut s_none = Sampler::new(SamplingParams {
            temperature: 1.0,
            min_p: None,
            seed: Some(9),
            ..SamplingParams::default()
        });
        for _ in 0..30 {
            assert_eq!(
                s_zero.sample(&mut logits.clone()),
                s_none.sample(&mut logits.clone())
            );
        }
    }

    // ---- repetition penalty ----

    #[test]
    fn repetition_penalty_suppresses_history_tokens() {
        // Two equally-likely tokens. With history={0} and rep=10, the
        // first sampled token should be 1 the vast majority of the time:
        //   logits[0] = 1.0 / 10 = 0.1, logits[1] = 1.0
        //   P(1) = exp(1.0) / (exp(0.1) + exp(1.0)) ~= 0.71
        // We use sample_logits directly with a fresh state per draw so the
        // history stays {0} -- otherwise `Sampler::sample` records each
        // outcome and both tokens are eventually penalized.
        let logits = vec![1.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            repetition_penalty: Some(10.0),
            seed: Some(4),
            ..SamplingParams::default()
        };
        let mut rng = Xorshift64::new(4);
        let mut ones = 0usize;
        for _ in 0..1000 {
            let mut state = SamplerState::new();
            state.record(0);
            if sample_logits(&mut logits.clone(), &params, &mut state, &mut rng) == 1 {
                ones += 1;
            }
        }
        // Expected ~710. Demand >= 650 to allow noise.
        assert!(ones > 650, "expected token 1 dominant, got {ones}/1000");
    }

    #[test]
    fn repetition_penalty_sign_aware_for_negative_logits() {
        // v <= 0 multiplies. With rep=2.0:
        //   logit -1.0 -> -2.0 (more suppressed)
        //   logit  1.0 ->  0.5
        let mut logits = vec![-1.0f32, 1.0f32];
        let params = SamplingParams {
            temperature: 1.0,
            repetition_penalty: Some(2.0),
            ..SamplingParams::default()
        };
        let mut state = SamplerState::new();
        state.record(0);
        state.record(1);
        apply_penalties(&mut logits, &params, &state);
        assert!(approx_eq(logits[0], -2.0, 1e-6), "got {}", logits[0]);
        assert!(approx_eq(logits[1], 0.5, 1e-6), "got {}", logits[1]);
    }

    // ---- presence penalty ----

    #[test]
    fn presence_penalty_uniform_subtraction() {
        let mut logits = vec![2.0, 2.0];
        let params = SamplingParams {
            temperature: 1.0,
            presence_penalty: Some(0.5),
            ..SamplingParams::default()
        };
        let mut state = SamplerState::new();
        state.record(0);
        state.record(0); // same token twice -> presence still subtracts once
        apply_penalties(&mut logits, &params, &state);
        assert!(approx_eq(logits[0], 1.5, 1e-6));
        assert!(approx_eq(logits[1], 2.0, 1e-6));
    }

    // ---- frequency penalty ----

    #[test]
    fn frequency_penalty_scales_with_count() {
        let mut logits = vec![2.0, 2.0, 2.0];
        let params = SamplingParams {
            temperature: 1.0,
            frequency_penalty: Some(0.5),
            ..SamplingParams::default()
        };
        let mut state = SamplerState::new();
        state.record(0); // count=1 -> -0.5
        state.record(1);
        state.record(1);
        state.record(1); // count=3 -> -1.5
        apply_penalties(&mut logits, &params, &state);
        assert!(approx_eq(logits[0], 1.5, 1e-6));
        assert!(approx_eq(logits[1], 0.5, 1e-6));
        assert!(approx_eq(logits[2], 2.0, 1e-6));
    }

    // ---- composition ----

    #[test]
    fn composition_top_p_top_k_and_repetition_penalty() {
        // 4 tokens: [3.0, 2.0, 1.0, 0.5]
        // history={3}, rep=10 -> logits[3] = 0.5 * 10 = 5.0 (>0 sign so /10
        //                       wait: 0.5 > 0 so divide -> 0.05).
        // After penalty: [3.0, 2.0, 1.0, 0.05]
        // top_k=2 -> [3.0, 2.0, -inf, -inf]
        // top_p=0.6 + softmax(temp=1) -> P(0)~0.73, P(1)~0.27. Token 0 wins
        // outright; top_p=0.6 keeps only token 0 (cumulative 0.73 >= 0.6).
        // Expected: token 0 always sampled, tokens 1-3 never.
        // Each sample uses a fresh state to keep history stable across draws.
        let logits_base = vec![3.0, 2.0, 1.0, 0.5];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: Some(0.6),
            top_k: Some(2),
            repetition_penalty: Some(10.0),
            seed: Some(5),
            ..SamplingParams::default()
        };
        let mut rng = Xorshift64::new(5);
        let mut counts = [0usize; 4];
        for _ in 0..500 {
            let mut state = SamplerState::new();
            state.record(3);
            let t = sample_logits(&mut logits_base.clone(), &params, &mut state, &mut rng) as usize;
            counts[t] += 1;
        }
        assert_eq!(counts[0], 500, "token 0 must dominate top-p=0.6 nucleus");
        assert_eq!(counts[1], 0);
        assert_eq!(counts[2], 0, "top_k=2 should mask token 2");
        assert_eq!(counts[3], 0, "top_k=2 should mask token 3 (also penalty)");
    }

    // ---- sampler state ----

    #[test]
    fn sampler_records_token_after_sample() {
        let mut logits = vec![1.0, 0.0, 0.0];
        let mut s = Sampler::new(SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        });
        let t = s.sample(&mut logits);
        assert_eq!(t, 0);
        assert_eq!(s.state.history, vec![0]);
        assert_eq!(s.state.freq.get(&0).copied(), Some(1));
    }

    #[test]
    fn sampler_state_clear_resets_history_and_freq() {
        let mut s = SamplerState::new();
        s.record(7);
        s.record(7);
        s.clear();
        assert!(s.history.is_empty());
        assert!(s.freq.is_empty());
    }

    // ---- edge cases ----

    #[test]
    fn empty_logits_returns_zero() {
        let mut s = Sampler::new(SamplingParams::default());
        assert_eq!(s.sample(&mut []), 0);
    }

    #[test]
    fn default_params_match_pre_z1_behavior() {
        // temperature=1.0 + no shaping -> plain temperature softmax +
        // multinomial. The reference uses `r <= cum` to match
        // the production sampler's first-draw-0.0 fix (see
        // `sample_categorical` for rationale).
        let logits_base = vec![1.0, 2.0, 3.0, 4.0];
        let mut sampler = Sampler::new(SamplingParams {
            temperature: 1.0,
            seed: Some(42),
            ..SamplingParams::default()
        });

        // Replay the CDF walk locally as the reference.
        let mut rng_ref = Xorshift64::new(42);
        for _ in 0..30 {
            let mut logits_ref = logits_base.clone();
            softmax_inplace(&mut logits_ref);
            let r = rng_ref.next_f32();
            let mut cum = 0.0f32;
            let mut want = (logits_ref.len() - 1) as u32;
            for (i, &p) in logits_ref.iter().enumerate() {
                cum += p;
                if r <= cum { want = i as u32; break; }
            }
            assert_eq!(sampler.sample(&mut logits_base.clone()), want);
        }
    }

    // ---- RNG first-draw=0.0 corruption fix ----

    #[test]
    fn rng_seed_42_first_draw_is_no_longer_zero() {
        // The original bug: `Xorshift64::new(42).next_f32() == 0.0` — the top 24
        // bits of state-after-one-shift for state=42 are all zero, so the
        // strict-`<` inverse-CDF in `sample_categorical` deterministically
        // picked index 0 (a control / byte token) for the first sampled
        // token of every default-seed CLI invocation.
        //
        // Post-: `Xorshift64::new` runs a SplitMix64 finalizer over
        // the seed before storing it as the initial state, so seed=42
        // (and the other small CLI seeds operators tend to type) now
        // produce a well-mixed first state — `next_f32()` returns a
        // non-zero value in `(0, 1)`.
        let mut rng = Xorshift64::new(42);
        let r0 = rng.next_f32();
        assert!(r0 > 0.0 && r0 < 1.0,
            " Xorshift64::new(42).next_f32() must be in (0, 1); got {r0}");
    }

    #[test]
    fn rng_first_draw_does_not_select_index_zero_under_skewed_dist() {
        // Construct a distribution that previously triggered the RNG-edge bug:
        // a tiny `p_0` so `[0, p_0)` is a near-empty bucket. With the
        // pre-fix RNG returning r=0.0 on seed=42 first draw, the strict
        // `r < cum` always selected index 0 (since `0 < small positive
        // p_0` is true). Post-fix: with SplitMix avalanche, r > 0, and
        // r < p_0 only when the dist genuinely places mass there.
        //
        // logits=[-5.0, 5.0, 0.0, 0.0] -> probs ~= [4.5e-5, 0.99, 0.0067, 0.0067]
        // With seed=42 post-fix, r is large enough to bypass the
        // ~4.5e-5 bucket and land in {1} (the argmax) most of the time.
        let logits = vec![-5.0, 5.0, 0.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            seed: Some(42),
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        let t = s.sample(&mut logits.clone());
        assert_ne!(t, 0,
            " default seed must not deterministically corrupt the first sampled token to index 0");
    }

    // ---- standard sampler ordering ----

    #[test]
    fn temperature_applies_after_top_p_min_p_filters() {
        // F3 contract: top_p / min_p run on logits-at-temp-1, then
        // temperature scales the SURVIVING set. Construct a case where
        // applying temp BEFORE top_p (the prior order) keeps a
        // different nucleus than applying temp AFTER (standard order).
        //
        // logits = [4.0, 3.0, 2.0, 1.0]
        // - At temp=1: softmax = [~0.643, ~0.236, ~0.087, ~0.032]
        //   top_p=0.85 keeps {0, 1} (cum 0.879 >= 0.85), drops {2, 3}.
        // - At temp=0.5 (the prior path applied temp FIRST):
        //   scaled = [8.0, 6.0, 4.0, 2.0], softmax = [~0.867, ~0.117,
        //   ~0.016, ~0.0002] — top_p=0.85 keeps only {0} (0.867 >= 0.85).
        //
        // The standard order keeps {0, 1}; the pre-fix order keeps {0}.
        // With a non-trivial RNG, the standard chain will sample 1 some
        // of the time; the pre-fix chain would always sample 0. We
        // verify the standard order by checking that token 1 is sampled
        // at least once across many draws.
        let logits = vec![4.0, 3.0, 2.0, 1.0];
        let params = SamplingParams {
            temperature: 0.5,
            top_p: Some(0.85),
            seed: Some(0xC0FFEE),
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        let mut saw_one = false;
        let mut saw_two_or_three = false;
        for _ in 0..500 {
            let t = s.sample(&mut logits.clone());
            if t == 1 {
                saw_one = true;
            }
            if t == 2 || t == 3 {
                saw_two_or_three = true;
            }
        }
        assert!(saw_one, "standard ordering must let token 1 enter nucleus (kept at temp=1)");
        assert!(!saw_two_or_three, "tokens 2-3 must stay outside top_p=0.85 nucleus");
    }

    #[test]
    fn min_p_on_logits_equivalent_to_log_shift() {
        // F3 contract: min-p in logit space uses `max_logit + log(min_p)`
        // as the cutoff. With min_p=0.1, log(0.1) ~= -2.3026.
        // logits = [5.0, 2.0, 1.0, 0.0] -> max=5.0, bound=5.0-2.3026=2.6974.
        // Surviving set: tokens with logit >= 2.6974 -> only {0} (5.0 >= 2.6974).
        // Token 1 at 2.0 < 2.6974 -> dropped.
        let logits = vec![5.0, 2.0, 1.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            min_p: Some(0.1),
            seed: Some(11),
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        for _ in 0..100 {
            assert_eq!(s.sample(&mut logits.clone()), 0,
                "min_p=0.1 should leave only token 0 (logit 5.0 above the log-shift bound)");
        }
    }
}
