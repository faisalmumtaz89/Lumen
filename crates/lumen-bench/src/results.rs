//! Benchmark result types and aggregation.

use std::time::Duration;

/// Result of a single benchmark iteration.
#[derive(Debug, Clone)]
pub struct BenchResult {
    // Timing
    pub total_time: Duration,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub tpot_ms: f64,

    // I/O
    pub bytes_read: u64,
    pub read_ops: u64,
    pub bandwidth_gibs: f64,

    // Cache
    pub weight_cache_hit_rate: f64,
    pub initial_residency: f64,
    pub final_residency: f64,

    // Pipeline
    pub total_stall_time: Duration,
    pub stall_fraction: f64,

    // Counts
    pub prompt_tokens: usize,
    pub generated_tokens: usize,

    /// Full generated token stream for this trial.
    /// Default is an empty `Vec` so older callers that did not populate
    /// this field compile unchanged; the determinism check treats an
    /// empty stream as "not captured" (gate becomes a no-op).
    pub generated_token_ids: Vec<u32>,

    /// Peak memory residency in bytes for this
    /// trial, as reported by `ComputeBackend::peak_memory_bytes()`.
    /// 0 means "not instrumented for this backend".
    pub peak_memory_bytes: u64,
}

/// Summary across multiple iterations.
#[derive(Debug, Clone)]
pub struct BenchSummary {
    pub label: String,
    pub iterations: usize,
    pub results: Vec<BenchResult>,
}

impl BenchSummary {
    pub fn new(label: String, results: Vec<BenchResult>) -> Self {
        let iterations = results.len();
        Self { label, iterations, results }
    }

    /// Mean TPOT across iterations.
    pub fn mean_tpot_ms(&self) -> f64 {
        if self.results.is_empty() { return 0.0; }
        let sum: f64 = self.results.iter().map(|r| r.tpot_ms).sum();
        sum / self.results.len() as f64
    }

    /// Median TPOT across iterations.
    pub fn median_tpot_ms(&self) -> f64 {
        percentile(&self.results.iter().map(|r| r.tpot_ms).collect::<Vec<_>>(), 0.5)
    }

    /// P95 TPOT across iterations.
    pub fn p95_tpot_ms(&self) -> f64 {
        percentile(&self.results.iter().map(|r| r.tpot_ms).collect::<Vec<_>>(), 0.95)
    }

    /// Standard deviation of TPOT across iterations.
    pub fn std_dev_tpot_ms(&self) -> f64 {
        std_dev(&self.results.iter().map(|r| r.tpot_ms).collect::<Vec<_>>())
    }

    /// Coefficient of variation of TPOT (std_dev / mean * 100) as a percentage.
    /// Returns 0.0 if mean is zero or there are fewer than 2 samples.
    pub fn cv_tpot_percent(&self) -> f64 {
        let mean = self.mean_tpot_ms();
        if mean <= f64::EPSILON || self.results.len() < 2 {
            return 0.0;
        }
        self.std_dev_tpot_ms() / mean * 100.0
    }

    /// Mean I/O bandwidth in GiB/s.
    pub fn mean_bandwidth_gibs(&self) -> f64 {
        if self.results.is_empty() { return 0.0; }
        let sum: f64 = self.results.iter().map(|r| r.bandwidth_gibs).sum();
        sum / self.results.len() as f64
    }

    /// Mean stall fraction.
    pub fn mean_stall_fraction(&self) -> f64 {
        if self.results.is_empty() { return 0.0; }
        let sum: f64 = self.results.iter().map(|r| r.stall_fraction).sum();
        sum / self.results.len() as f64
    }

    /// Mean total time.
    pub fn mean_total_time(&self) -> Duration {
        if self.results.is_empty() { return Duration::ZERO; }
        let sum: Duration = self.results.iter().map(|r| r.total_time).sum();
        sum / self.results.len() as u32
    }

    /// Cross-trial determinism check.
    ///
    /// Returns a [`DeterminismVerdict`] reporting whether trial-1's
    /// token stream byte-matches every subsequent trial.  Designed for
    /// the 5-trial / greedy + fixed-seed bench loop already wired in
    /// `runner.rs`.
    ///
    /// Semantics:
    /// - If fewer than 2 trials were collected, the verdict is
    ///   `n_total=K, n_matching=K, pass=true` — there is nothing to
    ///   compare against.
    /// - If any trial has `generated_token_ids.is_empty()` AND the
    ///   reference (trial 1) also has an empty stream, that pair is
    ///   considered a match (token capture is opt-in).
    /// - If the reference has tokens but a subsequent trial has an
    ///   empty stream, that subsequent trial is treated as a
    ///   MISMATCH — the caller dropped capture for one trial which
    ///   makes the gate unauditable.
    /// - Otherwise: `trial[i].generated_token_ids == trial[0].generated_token_ids`
    ///   (byte-identical) is the pass criterion.
    ///
    /// This is the determinism check; without it, a per-trial mismatch
    /// is unobservable from the caller.
    pub fn determinism_verdict(&self) -> DeterminismVerdict {
        let n = self.results.len();
        if n < 2 {
            return DeterminismVerdict {
                n_total_trials: n,
                n_matching_trials: n,
                first_divergence_trial: None,
                first_divergence_token_index: None,
                pass: true,
            };
        }

        // Trial 0 is the reference.
        let reference = &self.results[0].generated_token_ids;

        // Edge case: if neither the reference nor any trial captured
        // tokens, treat as "not captured" — gate vacuously passes (the
        // operator who wants determinism enforcement must wire token
        // capture in the runner; the harness does NOT silently lie).
        let any_captured = self
            .results
            .iter()
            .any(|r| !r.generated_token_ids.is_empty());
        if !any_captured {
            return DeterminismVerdict {
                n_total_trials: n,
                n_matching_trials: n,
                first_divergence_trial: None,
                first_divergence_token_index: None,
                pass: true,
            };
        }

        let mut n_matching = 1usize; // trial 0 trivially matches itself
        let mut first_divergence_trial: Option<usize> = None;
        let mut first_divergence_token_index: Option<usize> = None;

        for (i, r) in self.results.iter().enumerate().skip(1) {
            let other = &r.generated_token_ids;
            if other == reference {
                n_matching += 1;
                continue;
            }
            // Mismatch — record first divergence location for the
            // PASS row.  If lengths differ, divergence index is the
            // length of the shorter stream.
            if first_divergence_trial.is_none() {
                first_divergence_trial = Some(i);
                let min_len = reference.len().min(other.len());
                let mut idx = min_len; // default: length mismatch
                for j in 0..min_len {
                    if reference[j] != other[j] {
                        idx = j;
                        break;
                    }
                }
                first_divergence_token_index = Some(idx);
            }
        }

        let pass = n_matching == n;
        DeterminismVerdict {
            n_total_trials: n,
            n_matching_trials: n_matching,
            first_divergence_trial,
            first_divergence_token_index,
            pass,
        }
    }
}

/// Outcome of cross-trial determinism check.
///
/// All fields are present (no Option for the counters) so the row can
/// be JSON-serialised directly into the results matrix cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterminismVerdict {
    /// Total number of trials that were compared (== `BenchSummary::iterations`).
    pub n_total_trials: usize,

    /// How many trials byte-match trial 0's token stream.  When
    /// `pass = true` this equals `n_total_trials`.
    pub n_matching_trials: usize,

    /// 1-based-by-position trial index of the first non-matching trial,
    /// or `None` if every trial matches (gate passes).  Trial 0 cannot
    /// be a divergence (it is the reference).
    pub first_divergence_trial: Option<usize>,

    /// Position of the first token at which the diverging trial
    /// differs from trial 0.  `None` when no divergence was found.
    /// If the two streams have different lengths but identical
    /// prefixes, this points to `min(reference.len(), other.len())`.
    pub first_divergence_token_index: Option<usize>,

    /// Pass iff `n_matching_trials == n_total_trials` (every captured
    /// stream matches trial 0).  See [`BenchSummary::determinism_verdict`]
    /// for the empty-capture vacuous-pass rule.
    pub pass: bool,
}

/// Compute a percentile using linear interpolation between adjacent ranks.
///
/// For small N (e.g., 7 iterations), rounding-based percentile always returns
/// the maximum value for p95. Linear interpolation gives a more meaningful
/// estimate by interpolating between the two nearest data points.
fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.len() == 1 {
        return sorted[0];
    }

    // Use linear interpolation (same method as NumPy's default "linear").
    // virtual_idx is a continuous index into the sorted array.
    let virtual_idx = (sorted.len() as f64 - 1.0) * p;
    let lo = virtual_idx.floor() as usize;
    let hi = virtual_idx.ceil().min((sorted.len() - 1) as f64) as usize;
    let fraction = virtual_idx - lo as f64;

    sorted[lo] * (1.0 - fraction) + sorted[hi] * fraction
}

/// Compute sample standard deviation (Bessel's correction: N-1 denominator).
/// Returns 0.0 for fewer than 2 samples.
fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_single_element() {
        assert_eq!(percentile(&[42.0], 0.5), 42.0);
        assert_eq!(percentile(&[42.0], 0.95), 42.0);
    }

    #[test]
    fn percentile_empty() {
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn percentile_sorted() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let p50 = percentile(&data, 0.5);
        assert!((p50 - 50.5).abs() < 1.0, "p50={p50}");
    }

    #[test]
    fn percentile_linear_interpolation_small_n() {
        // With 3 data points [10, 20, 30], the old rounding p95 would give 30 (max).
        // With linear interpolation: virtual_idx = 2 * 0.95 = 1.9
        // result = 20 * 0.1 + 30 * 0.9 = 29.0
        let data = vec![10.0, 20.0, 30.0];
        let p95 = percentile(&data, 0.95);
        assert!((p95 - 29.0).abs() < 1e-10, "expected 29.0, got {p95}");
    }

    #[test]
    fn percentile_p50_of_two_is_midpoint() {
        let data = vec![10.0, 20.0];
        let p50 = percentile(&data, 0.5);
        assert!((p50 - 15.0).abs() < 1e-10, "expected 15.0, got {p50}");
    }

    #[test]
    fn std_dev_empty_or_single() {
        assert_eq!(std_dev(&[]), 0.0);
        assert_eq!(std_dev(&[42.0]), 0.0);
    }

    #[test]
    fn std_dev_known_values() {
        // [2, 4, 4, 4, 5, 5, 7, 9] -> mean=5, sample variance = 4.571..., std_dev ~ 2.138
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&data);
        assert!((sd - 2.1380899352993952).abs() < 1e-10, "expected ~2.138, got {sd}");
    }

    /// Local fixture builder so the new fields are
    /// covered exactly once per test and existing tests stay readable.
    fn mk_result(
        tpot_ms: f64,
        bandwidth_gibs: f64,
        tokens: Vec<u32>,
        peak_memory_bytes: u64,
    ) -> BenchResult {
        BenchResult {
            total_time: Duration::from_millis(100),
            prefill_time: Duration::from_millis(10),
            decode_time: Duration::from_millis(90),
            tpot_ms,
            bytes_read: 1_000_000,
            read_ops: 100,
            bandwidth_gibs,
            weight_cache_hit_rate: 0.5,
            initial_residency: 0.0,
            final_residency: 0.8,
            total_stall_time: Duration::from_millis(5),
            stall_fraction: 0.05,
            prompt_tokens: 128,
            generated_tokens: tokens.len(),
            generated_token_ids: tokens,
            peak_memory_bytes,
        }
    }

    #[test]
    fn bench_summary_aggregates() {
        let results = vec![
            mk_result(10.0, 1.0, vec![1, 2, 3], 0),
            mk_result(12.0, 1.1, vec![1, 2, 3], 0),
        ];

        let summary = BenchSummary::new("test".to_string(), results);
        assert_eq!(summary.iterations, 2);
        assert!(summary.median_tpot_ms() > 0.0);
        assert!(summary.mean_bandwidth_gibs() > 0.0);
        // mean TPOT = (10 + 12) / 2 = 11
        assert!((summary.mean_tpot_ms() - 11.0).abs() < 1e-10);
        // std_dev of [10, 12] with Bessel's: sqrt((1+1)/1) = sqrt(2) ~ 1.414
        assert!((summary.std_dev_tpot_ms() - std::f64::consts::SQRT_2).abs() < 1e-10);
        // CV = 1.414 / 11 * 100 ~ 12.856
        assert!(summary.cv_tpot_percent() > 12.0);
        assert!(summary.cv_tpot_percent() < 13.0);
    }

    // ---- cross-trial determinism tests ----

    #[test]
    fn determinism_single_trial_vacuously_passes() {
        let summary = BenchSummary::new(
            "single".to_string(),
            vec![mk_result(10.0, 1.0, vec![1, 2, 3, 4], 0)],
        );
        let v = summary.determinism_verdict();
        assert!(v.pass);
        assert_eq!(v.n_total_trials, 1);
        assert_eq!(v.n_matching_trials, 1);
        assert_eq!(v.first_divergence_trial, None);
        assert_eq!(v.first_divergence_token_index, None);
    }

    #[test]
    fn determinism_zero_trials_vacuously_passes() {
        let summary = BenchSummary::new("empty".to_string(), vec![]);
        let v = summary.determinism_verdict();
        assert!(v.pass);
        assert_eq!(v.n_total_trials, 0);
    }

    #[test]
    fn determinism_five_identical_trials_pass() {
        let token_stream = vec![5, 17, 42, 100, 200, 300, 400, 500];
        let results: Vec<BenchResult> = (0..5)
            .map(|_| mk_result(10.0, 1.0, token_stream.clone(), 0))
            .collect();
        let summary = BenchSummary::new("five-id".to_string(), results);
        let v = summary.determinism_verdict();
        assert!(v.pass, "{v:?}");
        assert_eq!(v.n_total_trials, 5);
        assert_eq!(v.n_matching_trials, 5);
        assert_eq!(v.first_divergence_trial, None);
    }

    #[test]
    fn determinism_one_diverging_trial_fails_and_pinpoints() {
        // trial 3 of 5 diverges at token 4.
        let good = vec![1u32, 2, 3, 4, 5];
        let bad = vec![1u32, 2, 3, 4, 99]; // diverges at index 4
        let results = vec![
            mk_result(10.0, 1.0, good.clone(), 0),
            mk_result(10.0, 1.0, good.clone(), 0),
            mk_result(10.0, 1.0, bad, 0),
            mk_result(10.0, 1.0, good.clone(), 0),
            mk_result(10.0, 1.0, good.clone(), 0),
        ];
        let summary = BenchSummary::new("race".to_string(), results);
        let v = summary.determinism_verdict();
        assert!(!v.pass, "{v:?}");
        assert_eq!(v.n_total_trials, 5);
        assert_eq!(v.n_matching_trials, 4);
        assert_eq!(v.first_divergence_trial, Some(2));
        assert_eq!(v.first_divergence_token_index, Some(4));
    }

    #[test]
    fn determinism_length_mismatch_pinpoints_at_short_length() {
        // Reference is 5 tokens; trial 1 is 3 tokens (truncated).  The
        // first divergence is at index 3 (the length of the shorter
        // stream).
        let reference = vec![1u32, 2, 3, 4, 5];
        let short = vec![1u32, 2, 3];
        let results = vec![
            mk_result(10.0, 1.0, reference, 0),
            mk_result(10.0, 1.0, short, 0),
        ];
        let summary = BenchSummary::new("len".to_string(), results);
        let v = summary.determinism_verdict();
        assert!(!v.pass);
        assert_eq!(v.first_divergence_trial, Some(1));
        assert_eq!(v.first_divergence_token_index, Some(3));
    }

    #[test]
    fn determinism_no_capture_anywhere_vacuously_passes() {
        // No trial captured tokens (empty vecs all around).  The
        // operator opted out of the gate — verdict must NOT silently
        // claim "verified determinism" when there is nothing to verify.
        // We collapse this to vacuous-pass with `n_matching_trials = n`
        // and the caller is responsible for interpreting "every stream
        // was empty" as "capture was disabled, not determinism was
        // proved".
        let results: Vec<BenchResult> =
            (0..5).map(|_| mk_result(10.0, 1.0, vec![], 0)).collect();
        let summary = BenchSummary::new("nocap".to_string(), results);
        let v = summary.determinism_verdict();
        assert!(v.pass);
        assert_eq!(v.n_matching_trials, 5);
    }

    #[test]
    fn determinism_one_empty_trial_amid_captured_trials_fails() {
        // Reference has captured tokens; trial 1 has an empty stream.
        // This is "capture dropped for one trial" — the gate is
        // unauditable so we MUST surface it as a FAIL.
        let results = vec![
            mk_result(10.0, 1.0, vec![1, 2, 3], 0),
            mk_result(10.0, 1.0, vec![], 0),
            mk_result(10.0, 1.0, vec![1, 2, 3], 0),
        ];
        let summary = BenchSummary::new("partial-cap".to_string(), results);
        let v = summary.determinism_verdict();
        assert!(!v.pass, "{v:?}");
        assert_eq!(v.first_divergence_trial, Some(1));
        // empty stream of length 0 vs reference of length 3 →
        // divergence at index 0 (the shorter length).
        assert_eq!(v.first_divergence_token_index, Some(0));
    }
}
