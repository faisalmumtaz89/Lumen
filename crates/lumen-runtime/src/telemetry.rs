//! Telemetry and metrics types.
//!
//! The runtime exposes instrumentation for: per-layer timings, I/O bandwidth,
//! cache hit rates, memory pressure signals, and overall throughput metrics.
//! These are used both for user-visible reporting and for the auto-tuner.

use std::time::Duration;

/// Timing breakdown for a single layer's processing.
#[derive(Debug, Clone, Default)]
pub struct PerLayerTiming {
    /// Layer index.
    pub layer_idx: usize,

    /// Time spent loading weights from storage (0 if cache hit).
    pub weight_load_time: Duration,

    /// Time spent loading KV cache entries (0 if RAM-resident or no attention).
    pub kv_load_time: Duration,

    /// Time spent in compute (the actual kernel execution).
    pub compute_time: Duration,

    /// Time spent saving KV cache entries.
    pub kv_save_time: Duration,

    /// Time the compute was stalled waiting for I/O (pipeline bubble).
    pub stall_time: Duration,

    /// Whether the weights were a cache hit.
    pub weight_cache_hit: bool,
}

impl PerLayerTiming {
    /// Total wall-clock time for this layer (excluding overlap with other tasks).
    pub fn total_time(&self) -> Duration {
        self.weight_load_time + self.kv_load_time + self.compute_time + self.kv_save_time
    }

    /// Returns the fraction of time spent stalled (pipeline bubble).
    pub fn stall_fraction(&self) -> f64 {
        let total = self.total_time();
        if total.is_zero() {
            0.0
        } else {
            self.stall_time.as_secs_f64() / total.as_secs_f64()
        }
    }
}

/// I/O bandwidth and request statistics.
#[derive(Debug, Clone, Default)]
pub struct IoMetrics {
    /// Total bytes read from storage during this measurement period.
    pub bytes_read: u64,

    /// Total bytes written to storage (KV saves, etc.).
    pub bytes_written: u64,

    /// Wall-clock duration of the measurement period.
    pub duration: Duration,

    /// Number of read operations issued.
    pub read_ops: u64,

    /// Number of write operations issued.
    pub write_ops: u64,
}

impl IoMetrics {
    /// Read bandwidth in bytes per second.
    pub fn read_bandwidth_bps(&self) -> f64 {
        if self.duration.is_zero() {
            0.0
        } else {
            self.bytes_read as f64 / self.duration.as_secs_f64()
        }
    }

    /// Read bandwidth in GiB/s (convenient for display).
    pub fn read_bandwidth_gibs(&self) -> f64 {
        self.read_bandwidth_bps() / (1024.0 * 1024.0 * 1024.0)
    }

    /// Average read request size in bytes.
    pub fn avg_read_size(&self) -> f64 {
        if self.read_ops == 0 {
            0.0
        } else {
            self.bytes_read as f64 / self.read_ops as f64
        }
    }
}

/// Top-level inference metrics for a generation session.
#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    /// Number of tokens in the input prompt.
    pub prompt_tokens: usize,

    /// Number of tokens generated (output).
    pub generated_tokens: usize,

    /// Prefill throughput (prompt processing) in tokens per second.
    pub prefill_tokens_per_sec: f64,

    /// Decode throughput (generation) in tokens per second.
    pub decode_tokens_per_sec: f64,

    /// Time-per-output-token (TPOT) for decode, in milliseconds.
    pub tpot_ms: f64,

    /// Total wall-clock time for the entire generation.
    pub total_time: Duration,

    /// Time spent in prefill phase.
    pub prefill_time: Duration,

    /// Time spent in decode phase.
    pub decode_time: Duration,

    /// Aggregated I/O metrics for the session.
    pub io: IoMetrics,

    /// Weight cache hit rate over the session.
    pub weight_cache_hit_rate: f64,

    /// Per-layer timings. Length = num_layers * generated_tokens.
    /// Stored as `timings[token_idx * num_layers + layer_idx]`.
    pub per_layer_timings: Vec<PerLayerTiming>,

    /// Peak memory usage in bytes (weights + KV + activations + scratch).
    pub peak_memory_bytes: u64,
}

impl InferenceMetrics {
    /// Overall throughput in tokens per second (prompt + generated).
    pub fn overall_tokens_per_sec(&self) -> f64 {
        let total_tokens = (self.prompt_tokens + self.generated_tokens) as f64;
        if self.total_time.is_zero() {
            0.0
        } else {
            total_tokens / self.total_time.as_secs_f64()
        }
    }

    /// Returns a summary string suitable for display.
    pub fn summary(&self) -> String {
        format!(
            "Prompt: {} tok, Generated: {} tok\n\
             Prefill: {:.1} tok/s ({:.1}ms)\n\
             Decode:  {:.1} tok/s, TPOT: {:.1}ms\n\
             I/O:     read {:.2} GiB/s, avg request {:.0} KiB\n\
             Cache:   weight hit rate {:.1}%\n\
             Memory:  peak {:.1} MiB",
            self.prompt_tokens,
            self.generated_tokens,
            self.prefill_tokens_per_sec,
            self.prefill_time.as_secs_f64() * 1000.0,
            self.decode_tokens_per_sec,
            self.tpot_ms,
            self.io.read_bandwidth_gibs(),
            self.io.avg_read_size() / 1024.0,
            self.weight_cache_hit_rate * 100.0,
            self.peak_memory_bytes as f64 / (1024.0 * 1024.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_layer_timing_total_time() {
        let t = PerLayerTiming {
            weight_load_time: Duration::from_millis(10),
            kv_load_time: Duration::from_millis(5),
            compute_time: Duration::from_millis(20),
            kv_save_time: Duration::from_millis(3),
            stall_time: Duration::from_millis(100), // NOT included in total
            ..Default::default()
        };
        assert_eq!(t.total_time(), Duration::from_millis(38));
    }

    #[test]
    fn per_layer_timing_stall_fraction_zero_total() {
        let t = PerLayerTiming::default();
        assert_eq!(t.stall_fraction(), 0.0);
    }

    #[test]
    fn io_metrics_zero_duration() {
        let m = IoMetrics::default();
        assert_eq!(m.read_bandwidth_bps(), 0.0);
        assert_eq!(m.read_bandwidth_gibs(), 0.0);
        assert_eq!(m.avg_read_size(), 0.0);
    }

    #[test]
    fn inference_metrics_overall_tokens_per_sec_zero_time() {
        let m = InferenceMetrics {
            prompt_tokens: 10,
            generated_tokens: 5,
            ..Default::default()
        };
        assert_eq!(m.overall_tokens_per_sec(), 0.0);
    }

    #[test]
    fn inference_metrics_summary_no_panic() {
        let m = InferenceMetrics::default();
        let s = m.summary();
        assert!(!s.is_empty());
        assert!(s.contains("Prompt:"));
    }
}
