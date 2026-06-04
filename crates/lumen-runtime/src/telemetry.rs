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

    /// KV cache telemetry captured at the end of generation.
    ///
    /// Populated by the engine path when it observes the live `KvCache`.
    /// Library callers that bypass the engine (direct `Session` use) can fill
    /// this manually via [`InferenceMetrics::with_kv`].
    pub kv: KvCacheStats,
}

/// Snapshot of the KV cache size + position at the end of an inference
/// session.
///
/// Fields are unsigned so the type can be embedded directly in
/// [`InferenceMetrics`] without `Option` overhead. Default = zeros, which
/// matches "no KV reported" semantics for non-engine callers.
///
/// `allocated_bytes` is the constant pre-allocated capacity (matches
/// `KvCache::allocated_bytes`); `used_bytes` is the live byte count that
/// scales with `seq_len` (matches `KvCache::total_bytes`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvCacheStats {
    /// Pre-allocated CPU-side capacity in bytes.
    /// Disk-KV `save_atomic` writes exactly this many bytes.
    pub allocated_bytes: u64,

    /// Live byte count (scales with `seq_len`).
    pub used_bytes: u64,

    /// Current sequence length in tokens.
    pub seq_len: usize,

    /// Maximum sequence length the cache was sized for.
    pub max_seq_len: usize,
}

impl KvCacheStats {
    /// Capture a stats snapshot from a live [`crate::kv::KvCache`].
    pub fn from_kv(kv: &crate::kv::KvCache) -> Self {
        Self {
            allocated_bytes: kv.allocated_bytes(),
            used_bytes: kv.total_bytes(),
            seq_len: kv.seq_len(),
            max_seq_len: kv.max_seq_len(),
        }
    }
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
             KV:      {}/{} tok, {:.1}/{:.1} MiB (used/allocated)\n\
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
            self.kv.seq_len,
            self.kv.max_seq_len,
            self.kv.used_bytes as f64 / (1024.0 * 1024.0),
            self.kv.allocated_bytes as f64 / (1024.0 * 1024.0),
            self.peak_memory_bytes as f64 / (1024.0 * 1024.0),
        )
    }

    /// Attach a KV cache snapshot, consuming and returning `self` for use in
    /// the `..InferenceMetrics::default()` builder pattern. Most call-sites
    /// will instead set the `kv` field directly via the struct literal.
    pub fn with_kv(mut self, kv_stats: KvCacheStats) -> Self {
        self.kv = kv_stats;
        self
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

    // ---- KV telemetry tests ----

    #[test]
    fn kv_cache_stats_default_is_zero() {
        let s = KvCacheStats::default();
        assert_eq!(s.allocated_bytes, 0);
        assert_eq!(s.used_bytes, 0);
        assert_eq!(s.seq_len, 0);
        assert_eq!(s.max_seq_len, 0);
    }

    #[test]
    fn kv_cache_stats_from_kv_matches_kv_methods() {
        use crate::kv::{KvCache, KvCacheConfig, KvPrecision};
        let cfg = KvCacheConfig {
            max_seq_len: 64,
            num_layers: 4,
            num_kv_heads: 2,
            head_dim: 8,
            precision: KvPrecision::F32,
        };
        let mut kv = KvCache::new(cfg).unwrap();
        // Empty cache: used = 0, allocated = full capacity.
        let s = KvCacheStats::from_kv(&kv);
        assert_eq!(s.seq_len, 0);
        assert_eq!(s.max_seq_len, 64);
        assert_eq!(s.used_bytes, 0);
        assert_eq!(s.allocated_bytes, kv.allocated_bytes());
        // Allocated is constant; used grows with seq_len.
        kv.advance_seq_len().unwrap();
        kv.advance_seq_len().unwrap();
        let s2 = KvCacheStats::from_kv(&kv);
        assert_eq!(s2.seq_len, 2);
        assert_eq!(s2.allocated_bytes, s.allocated_bytes);
        assert!(s2.used_bytes > 0);
        assert!(s2.used_bytes < s2.allocated_bytes);
    }

    #[test]
    fn inference_metrics_with_kv_round_trips() {
        let stats = KvCacheStats {
            allocated_bytes: 4096,
            used_bytes: 256,
            seq_len: 4,
            max_seq_len: 64,
        };
        let m = InferenceMetrics::default().with_kv(stats.clone());
        assert_eq!(m.kv, stats);
    }

    #[test]
    fn inference_metrics_summary_contains_kv_section() {
        let stats = KvCacheStats {
            allocated_bytes: 4096,
            used_bytes: 256,
            seq_len: 4,
            max_seq_len: 64,
        };
        let m = InferenceMetrics::default().with_kv(stats);
        let s = m.summary();
        assert!(s.contains("KV:"), "summary must report KV: {s}");
        assert!(s.contains("4/64"), "summary must report seq_len/max_seq_len: {s}");
    }

    // ---- per-component server memory breakdown tests ----

    #[test]
    fn server_memory_breakdown_default_is_zero() {
        let b = ServerMemoryBreakdown::default();
        assert_eq!(b.kv_used_bytes, 0);
        assert_eq!(b.kv_allocated_bytes, 0);
        assert_eq!(b.kv_seq_len, 0);
        assert_eq!(b.kv_max_seq_len, 0);
        assert_eq!(b.session_tokens_len, 0);
        assert_eq!(b.session_pending_logits_bytes, 0);
        assert_eq!(b.session_timings_bytes, 0);
        assert_eq!(b.metal_current_allocated_bytes, 0);
        assert_eq!(b.tokio_active_tasks, 0);
        assert_eq!(b.engine_inbox_capacity, 0);
        assert_eq!(b.engine_inbox_len, 0);
        assert_eq!(b.disk_kv_used_bytes, 0);
        assert_eq!(b.update_count, 0);
        assert_eq!(b.last_update_unix, 0);
    }

    #[test]
    fn server_memory_breakdown_to_json_keys_are_stable() {
        // The soak harness sampler parses this JSON shape into JSONL, so the
        // key set is load-bearing.  Pin every key here so a future field
        // rename forces a test update + soak-harness update in lock-step.
        let mut b = ServerMemoryBreakdown::default();
        b.kv_used_bytes = 1_000;
        b.kv_allocated_bytes = 2_000;
        b.kv_seq_len = 3;
        b.kv_max_seq_len = 4;
        b.session_tokens_len = 5;
        b.session_pending_logits_bytes = 6;
        b.session_timings_bytes = 7;
        b.metal_current_allocated_bytes = 8;
        b.tokio_active_tasks = 9;
        b.engine_inbox_capacity = 10;
        b.engine_inbox_len = 11;
        b.disk_kv_used_bytes = 12;
        b.update_count = 13;
        b.last_update_unix = 14;
        let s = b.to_jsonl();
        for k in [
            "kv_used_bytes", "kv_allocated_bytes", "kv_seq_len", "kv_max_seq_len",
            "session_tokens_len", "session_pending_logits_bytes", "session_timings_bytes",
            "metal_current_allocated_bytes", "tokio_active_tasks",
            "engine_inbox_capacity", "engine_inbox_len",
            "disk_kv_used_bytes", "update_count", "last_update_unix",
        ] {
            assert!(
                s.contains(&format!("\"{}\":", k)),
                "ServerMemoryBreakdown.to_jsonl missing key {k:?}; got {s}",
            );
        }
        // Spot-check a value to catch field/value cross-wiring.
        assert!(
            s.contains("\"kv_used_bytes\":1000"),
            "ServerMemoryBreakdown.to_jsonl wired kv_used_bytes wrong: {s}",
        );
        // Output must be a single line with no embedded newlines so the
        // soak harness can append it directly to a JSONL file.
        assert!(!s.contains('\n'), "to_jsonl must be single-line: {s}");
        // Output must start with `{` and end with `}` so the JSONL is
        // well-formed without needing string-trimming.
        assert!(s.starts_with('{') && s.ends_with('}'), "malformed JSON: {s}");
    }
}

// =====================================================================
// ServerMemoryBreakdown — per-component RSS attribution
// =====================================================================
//
// lumen-server soak background: found RSS growing at +6.5%/h.
// RSS is a single number — it cannot tell us which component is
// leaking. This struct adds a per-component breakdown that the soak harness
// samples alongside RSS on the same 30 s cadence, so the evaluation
// phase can attribute growth to one of the 7 hypothesis classes:
// (KV bytes, scratch pool, pipeline cache, Tokio tasks,
// sampler map, mmap'd file bytes, MTLBuffer pool).
//
// Default-path constraints:
//
// - The struct lives in lumen-runtime (already a public surface for
//   telemetry).
// - All field reads are O(1) snapshots (atomic loads / single Mutex peek)
//   so the breakdown sampler does not contend with the engine hot path.
// - The struct + its `to_jsonl()` are public so the soak harness can
//   capture them.  The HTTP `/debug/memory_breakdown` endpoint that
//   actually serves the JSON is env-gated in `lumen-server` (returns
//   404 when disabled), so the default path is byte-identical.

/// Snapshot of the lumen-server's per-component memory residency.
///
/// Fields cover the 7 hypothesis classes from plus secondary
/// counters (engine inbox depth, update count) that are useful for ruling
/// out "did the sampler actually fire?" sanity questions.
///
/// Field semantics:
///
/// - `kv_*` mirror [`KvCacheStats`] for the single live `Session` the
///   `EngineWorker` holds.
/// - `session_tokens_len` is `session.tokens.len()`; multiplied by 4
///   gives the token-vector heap footprint.  Stored as token count (not
///   bytes) because the operator usually wants to spot the runaway pattern
///   ("tokens never decay") more than the exact byte total.
/// - `session_pending_logits_bytes` is the byte cost of the `pending_logits`
///   buffer (vocab_size × 4 bytes when present, 0 when None).
/// - `session_timings_bytes` is `timings.len() * size_of::<PerLayerTiming>()`
///   — should be zero in production (per-layer timings off by default) but
///   we report it so a misconfiguration is visible.
/// - `metal_current_allocated_bytes` is what `MTLDevice.currentAllocatedSize`
///   returns on Apple Silicon — total bytes outstanding for all MTLBuffer
///   MTLTexture / MTLHeap objects this process holds. 0 on non-Metal
///   backends.
/// - `tokio_active_tasks` is the Tokio runtime's `metrics().num_alive_tasks()`
///   when available; 0 if the metrics surface is unbuildable (Tokio < 1.34
///   or `tokio_unstable` not enabled).
/// - `engine_inbox_capacity` / `engine_inbox_len` are the mpsc job inbox
///   stats: capacity is constant (16 by default), len is "jobs waiting" —
///   should oscillate at low single digits in a healthy system.
/// - `disk_kv_used_bytes` re-reads the disk-KV directory size.  Already
///   tracked by the soak harness's `dir_size_bytes`; we mirror it here so
///   the breakdown JSON is self-contained.
/// - `update_count` is incremented once per breakdown refresh; the soak
///   harness uses this to detect "snapshot got stale" (the breakdown only
///   updates after a job completes; a quiet inbox produces a stale
///   snapshot).
/// - `last_update_unix` is the wall-clock time of the most recent refresh.
///
/// The struct is `Default` so a fresh `EngineWorker` starts with all
/// zeros, and serializes to a single-line JSON via `to_jsonl()` for
/// direct append to the soak harness's `soak-breakdown.jsonl` artifact.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServerMemoryBreakdown {
    // ---- Hypothesis class 1: KvCache bytes ----
    pub kv_used_bytes: u64,
    pub kv_allocated_bytes: u64,
    pub kv_seq_len: usize,
    pub kv_max_seq_len: usize,

    // ---- Hypothesis class 2 + 5: Session-owned Rust state (tokens,
    //      logits, timings — these are the per-Session structures most
    //      likely to grow without bound under repeated extend_with_cache).
    pub session_tokens_len: usize,
    pub session_pending_logits_bytes: u64,
    pub session_timings_bytes: u64,

    // ---- Hypothesis class 3 + 7: Metal driver residency. ----
    // The driver counts MTLBuffer / MTLTexture / MTLHeap objects for the
    // calling process group.  If `MetalScratch` is leaking buffers this
    // number rises monotonically.  If it stays flat the leak is in the
    // Rust heap, not the Metal driver.
    pub metal_current_allocated_bytes: u64,

    // ---- Hypothesis class 4: Tokio task count ----
    pub tokio_active_tasks: u64,

    // ---- Engine inbox depth (rule out backpressure starvation) ----
    pub engine_inbox_capacity: usize,
    pub engine_inbox_len: usize,

    // ---- Disk-KV byte count (mirrors the soak harness's directory walk) ----
    pub disk_kv_used_bytes: u64,

    // ---- Sampler heartbeat ----
    pub update_count: u64,
    pub last_update_unix: u64,
}

impl ServerMemoryBreakdown {
    /// Serialise this breakdown as one JSONL line.
    ///
    /// Output: `{"kv_used_bytes":..,...,"last_update_unix":..}` with no
    /// embedded newline (the soak harness appends one itself).  Numeric
    /// formatting matches `serde_json` defaults (no thousands separators,
    /// integers without decimals).
    ///
    /// The JSON keys are pinned by
    /// `server_memory_breakdown_to_json_keys_are_stable` so renames force
    /// a soak-harness update in lock-step.
    pub fn to_jsonl(&self) -> String {
        format!(
            r#"{{"kv_used_bytes":{},"kv_allocated_bytes":{},"kv_seq_len":{},"kv_max_seq_len":{},"session_tokens_len":{},"session_pending_logits_bytes":{},"session_timings_bytes":{},"metal_current_allocated_bytes":{},"tokio_active_tasks":{},"engine_inbox_capacity":{},"engine_inbox_len":{},"disk_kv_used_bytes":{},"update_count":{},"last_update_unix":{}}}"#,
            self.kv_used_bytes,
            self.kv_allocated_bytes,
            self.kv_seq_len,
            self.kv_max_seq_len,
            self.session_tokens_len,
            self.session_pending_logits_bytes,
            self.session_timings_bytes,
            self.metal_current_allocated_bytes,
            self.tokio_active_tasks,
            self.engine_inbox_capacity,
            self.engine_inbox_len,
            self.disk_kv_used_bytes,
            self.update_count,
            self.last_update_unix,
        )
    }
}
