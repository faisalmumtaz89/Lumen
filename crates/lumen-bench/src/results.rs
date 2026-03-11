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

    #[test]
    fn bench_summary_aggregates() {
        let results = vec![
            BenchResult {
                total_time: Duration::from_millis(100),
                prefill_time: Duration::from_millis(10),
                decode_time: Duration::from_millis(90),
                tpot_ms: 10.0,
                bytes_read: 1_000_000,
                read_ops: 100,
                bandwidth_gibs: 1.0,
                weight_cache_hit_rate: 0.5,
                initial_residency: 0.0,
                final_residency: 0.8,
                total_stall_time: Duration::from_millis(5),
                stall_fraction: 0.05,
                prompt_tokens: 128,
                generated_tokens: 32,
            },
            BenchResult {
                total_time: Duration::from_millis(120),
                prefill_time: Duration::from_millis(12),
                decode_time: Duration::from_millis(108),
                tpot_ms: 12.0,
                bytes_read: 1_100_000,
                read_ops: 110,
                bandwidth_gibs: 1.1,
                weight_cache_hit_rate: 0.6,
                initial_residency: 0.0,
                final_residency: 0.9,
                total_stall_time: Duration::from_millis(6),
                stall_fraction: 0.06,
                prompt_tokens: 128,
                generated_tokens: 32,
            },
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
}
