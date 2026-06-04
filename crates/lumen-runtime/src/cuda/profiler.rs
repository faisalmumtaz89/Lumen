//! CUDA decode/prefill stage profiler.
//!
//! Per-stage timing harness for the CUDA forward pass. Uses `cudaEventRecord`
//! around groups of kernel launches that form a logical stage (e.g., the four
//! shared-quant GDN projections, the GDN state update block, the FFN gate/up
//! GLU). Events are recorded on the same single stream as the kernels, so they
//! are queued in submission order with no extra synchronization until the
//! summary is collected.
//!
//! # Enabling
//!
//! Set `LUMEN_CUDA_PROFILE=1` before invoking the CUDA backend. When unset
//! (the default), `begin` / `end` short-circuit to a single `Option::is_none()`
//! check -- no event allocation, no Vec push, no FFI overhead.
//!
//! # Layer taxonomy
//!
//! Each stage is tagged with a `LayerType`:
//! * `Gdn`     - dispatched per-GDN-layer (24 instances on Qwen3.5-9B)
//! * `Full`    - dispatched per-full-attention layer (8 instances)
//! * `Whole`   - dispatched once per token (embed, final norm, output proj, ...)
//!
//! The summary aggregates samples across all calls; the printed table reports
//! median microseconds per token (24x or 8x the per-call median for layer-typed
//! stages) so the rows line up with the forward pass.

use std::collections::BTreeMap;

#[cfg(feature = "cuda")]
use cudarc::driver::result::event;
#[cfg(feature = "cuda")]
use cudarc::driver::sys as cuda_sys;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaStream;

/// Coarse classification used by the summary table to attribute work to the
/// per-layer cost on Qwen3.5-9B (24 GDN + 8 full + once-per-token whole).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LayerType {
    /// Dispatched inside a GDN layer (Qwen3.5-9B: 24 instances per token).
    Gdn,
    /// Dispatched inside a full softmax-attention layer (Qwen3.5-9B: 8 per token).
    Full,
    /// Dispatched once per token (embed, final norm, output projection, ...).
    Whole,
}

impl LayerType {
    fn as_str(self) -> &'static str {
        match self {
            LayerType::Gdn => "GDN",
            LayerType::Full => "FULL",
            LayerType::Whole => "WHOLE",
        }
    }
}

#[cfg(feature = "cuda")]
struct StageBuckets {
    layer_type: LayerType,
    starts: Vec<cuda_sys::CUevent>,
    ends: Vec<cuda_sys::CUevent>,
    /// Index of the next open `begin` without a matching `end`. When this
    /// equals `ends.len()`, the stage is balanced.
    open: usize,
}

#[cfg(feature = "cuda")]
impl StageBuckets {
    fn new(layer_type: LayerType) -> Self {
        Self {
            layer_type,
            starts: Vec::new(),
            ends: Vec::new(),
            open: 0,
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for StageBuckets {
    fn drop(&mut self) {
        // Destroy all events. Safe: the device context owning these events
        // outlives the profiler (state.lock().Some(MutableState) is dropped
        // before the device context).
        unsafe {
            for e in self.starts.drain(..) {
                let _ = event::destroy(e);
            }
            for e in self.ends.drain(..) {
                let _ = event::destroy(e);
            }
        }
    }
}

// CUevent is `*mut CUevent_st` (raw pointer) which is not Send/Sync by default.
// All access to the profiler is gated by `Mutex<Option<MutableState>>` in
// CudaBackend, so the buckets are never touched from more than one thread at a
// time. CUevent handles themselves are safe to migrate between threads in CUDA.
#[cfg(feature = "cuda")]
unsafe impl Send for StageBuckets {}
#[cfg(feature = "cuda")]
unsafe impl Sync for StageBuckets {}

/// Stage timing collector. Enabled when `LUMEN_CUDA_PROFILE=1`.
#[cfg(feature = "cuda")]
pub struct StageProfiler {
    /// `BTreeMap` keeps the stages in a stable, sorted iteration order for
    /// reproducible output.
    stages: BTreeMap<&'static str, StageBuckets>,
    /// Number of decode tokens recorded (used to normalize µs/token in the
    /// summary). Caller increments via `record_token`.
    decode_tokens: u64,
}

#[cfg(feature = "cuda")]
impl StageProfiler {
    /// Construct a profiler if the env switch is set; otherwise `None`.
    pub fn from_env() -> Option<Self> {
        match std::env::var("LUMEN_CUDA_PROFILE") {
            Ok(v) if v == "1" => Some(Self {
                stages: BTreeMap::new(),
                decode_tokens: 0,
            }),
            _ => None,
        }
    }

    /// Mark the start of a stage on `stream`. Allocates a fresh CUevent.
    pub fn begin(&mut self, stage: &'static str, layer_type: LayerType, stream: &CudaStream) {
        let bucket = self
            .stages
            .entry(stage)
            .or_insert_with(|| StageBuckets::new(layer_type));
        let ev = match event::create(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT) {
            Ok(e) => e,
            Err(_) => return,
        };
        // SAFETY: ev was just created; stream outlives this call.
        if unsafe { event::record(ev, stream.cu_stream()) }.is_err() {
            unsafe { let _ = event::destroy(ev); };
            return;
        }
        bucket.starts.push(ev);
        bucket.open += 1;
    }

    /// Mark the end of the most-recent `begin` for `stage` on `stream`.
    pub fn end(&mut self, stage: &'static str, stream: &CudaStream) {
        let bucket = match self.stages.get_mut(stage) {
            Some(b) => b,
            None => return,
        };
        if bucket.open == 0 {
            return;
        }
        let ev = match event::create(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT) {
            Ok(e) => e,
            Err(_) => return,
        };
        if unsafe { event::record(ev, stream.cu_stream()) }.is_err() {
            unsafe { let _ = event::destroy(ev); };
            return;
        }
        bucket.ends.push(ev);
        bucket.open -= 1;
    }

    /// Record that a decode token has completed (for µs/token normalization).
    pub fn record_token(&mut self) {
        self.decode_tokens = self.decode_tokens.saturating_add(1);
    }

    /// Collect all recorded events into a summary. Synchronizes the last event
    /// in every stage so elapsed times are valid. Drops the event handles.
    pub fn collect(&mut self) -> ProfilerSummary {
        let mut entries: Vec<StageEntry> = Vec::with_capacity(self.stages.len());
        for (name, bucket) in std::mem::take(&mut self.stages) {
            let n = bucket.starts.len().min(bucket.ends.len());
            if n == 0 {
                continue;
            }
            // Synchronize the last event to ensure all preceding records are
            // visible to the host (single barrier per stage).
            unsafe { let _ = event::synchronize(bucket.ends[n - 1]); };
            let mut samples: Vec<f64> = Vec::with_capacity(n);
            for i in 0..n {
                let ms = match unsafe { event::elapsed(bucket.starts[i], bucket.ends[i]) } {
                    Ok(v) => v as f64,
                    Err(_) => continue,
                };
                samples.push(ms * 1000.0); // ms -> µs
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            entries.push(StageEntry {
                stage: name.to_string(),
                layer_type: bucket.layer_type,
                mean_us: mean(&samples),
                median_us: median(&samples),
                std_us: std_dev(&samples),
                samples,
            });
        }
        entries.sort_by(|a, b| {
            a.layer_type
                .cmp(&b.layer_type)
                .then_with(|| b.median_us.partial_cmp(&a.median_us).unwrap_or(std::cmp::Ordering::Equal))
        });
        ProfilerSummary {
            entries,
            decode_tokens: self.decode_tokens,
        }
    }
}

/// One row in the profiler output.
#[derive(Clone, Debug)]
pub struct StageEntry {
    pub stage: String,
    pub layer_type: LayerType,
    /// Raw per-call samples in microseconds.
    pub samples: Vec<f64>,
    pub mean_us: f64,
    pub median_us: f64,
    pub std_us: f64,
}

/// Aggregated profile result.
#[derive(Clone, Debug)]
pub struct ProfilerSummary {
    pub entries: Vec<StageEntry>,
    pub decode_tokens: u64,
}

impl ProfilerSummary {
    /// Total median µs/token across all stages (sum of per-stage medians,
    /// multiplied by the per-token call multiplicity).
    pub fn total_us_per_token(&self) -> f64 {
        self.entries
            .iter()
            .map(|e| e.median_us * (e.samples.len() as f64).max(1.0))
            .sum::<f64>()
            / (self.decode_tokens.max(1) as f64)
    }

    /// Emit JSON to a writer.
    pub fn write_json<W: std::io::Write>(&self, mut w: W) -> std::io::Result<()> {
        writeln!(w, "{{")?;
        writeln!(w, "  \"decode_tokens\": {},", self.decode_tokens)?;
        writeln!(w, "  \"total_us_per_token\": {:.3},", self.total_us_per_token())?;
        writeln!(w, "  \"stages\": [")?;
        for (i, e) in self.entries.iter().enumerate() {
            let comma = if i + 1 == self.entries.len() { "" } else { "," };
            writeln!(
                w,
                "    {{\"stage\":\"{}\",\"layer_type\":\"{}\",\"calls\":{},\"mean_us\":{:.3},\"median_us\":{:.3},\"std_us\":{:.3}}}{}",
                e.stage,
                e.layer_type.as_str(),
                e.samples.len(),
                e.mean_us,
                e.median_us,
                e.std_us,
                comma
            )?;
        }
        writeln!(w, "  ]")?;
        writeln!(w, "}}")?;
        Ok(())
    }

    /// Emit a human-readable per-token table to a writer.
    pub fn write_table<W: std::io::Write>(&self, mut w: W) -> std::io::Result<()> {
        let total_per_tok = self.total_us_per_token().max(1e-9);
        writeln!(w, "")?;
        writeln!(w, "=== Lumen CUDA Stage Profile ===")?;
        writeln!(w, "Tokens recorded: {}", self.decode_tokens)?;
        writeln!(w, "Total per-token (sum of stage medians): {:.1} µs", total_per_tok)?;
        writeln!(w, "")?;
        writeln!(
            w,
            "| {:<10} | {:<28} | {:>9} | {:>9} | {:>7} | {:>9} |",
            "Layer", "Stage", "µs/token", "% total", "calls", "std_us"
        )?;
        writeln!(w, "|{:-<12}|{:-<30}|{:->11}|{:->11}|{:->9}|{:->11}|", "", "", "", "", "", "")?;
        for e in &self.entries {
            // µs/token = median_us * calls / decode_tokens
            let per_tok = e.median_us * (e.samples.len() as f64) / (self.decode_tokens.max(1) as f64);
            let pct = 100.0 * per_tok / total_per_tok;
            writeln!(
                w,
                "| {:<10} | {:<28} | {:>9.1} | {:>8.2}% | {:>7} | {:>9.2} |",
                e.layer_type.as_str(),
                e.stage,
                per_tok,
                pct,
                e.samples.len(),
                e.std_us,
            )?;
        }
        writeln!(w, "")?;
        Ok(())
    }

    /// Identify the top-N slowest stages by µs/token contribution.
    pub fn top_stages(&self, n: usize) -> Vec<(String, f64, f64)> {
        let total_per_tok = self.total_us_per_token().max(1e-9);
        let mut ranked: Vec<(String, f64, f64)> = self
            .entries
            .iter()
            .map(|e| {
                let per_tok = e.median_us * (e.samples.len() as f64) / (self.decode_tokens.max(1) as f64);
                let pct = 100.0 * per_tok / total_per_tok;
                (format!("{} {}", e.layer_type.as_str(), e.stage), per_tok, pct)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(n);
        ranked
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    xs.iter().sum::<f64>() / (xs.len() as f64)
}

fn median(xs: &[f64]) -> f64 {
    // xs is pre-sorted by caller.
    if xs.is_empty() { return 0.0; }
    let mid = xs.len() / 2;
    if xs.len() % 2 == 0 {
        0.5 * (xs[mid - 1] + xs[mid])
    } else {
        xs[mid]
    }
}

fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 { return 0.0; }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / ((xs.len() - 1) as f64);
    var.sqrt()
}

// ---------------------------------------------------------------------------
// Convenience macros for instrumentation sites.
// ---------------------------------------------------------------------------

/// Invoke `$blk` with `begin(stage)` / `end(stage)` bracketing when a profiler
/// is active. Compiles down to a single Option check when disabled. Caller
/// must hold both `&CudaStream` and `&mut Option<StageProfiler>`.
#[macro_export]
macro_rules! cuda_profile_stage {
    ($prof:expr, $stream:expr, $stage:expr, $layer:expr, $blk:block) => {{
        if let Some(p) = $prof.as_mut() {
            p.begin($stage, $layer, $stream);
        }
        let __res = $blk;
        if let Some(p) = $prof.as_mut() {
            p.end($stage, $stream);
        }
        __res
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_basic() {
        let xs = [10.0, 20.0, 30.0];
        assert!((mean(&xs) - 20.0).abs() < 1e-9);
        assert!((median(&xs) - 20.0).abs() < 1e-9);
        assert!(std_dev(&xs) > 0.0);
    }

    #[test]
    fn median_even() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        assert!((median(&xs) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn empty_stats_are_zero() {
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(median(&[]), 0.0);
        assert_eq!(std_dev(&[]), 0.0);
        assert_eq!(std_dev(&[42.0]), 0.0);
    }

    #[test]
    fn layer_type_string() {
        assert_eq!(LayerType::Gdn.as_str(), "GDN");
        assert_eq!(LayerType::Full.as_str(), "FULL");
        assert_eq!(LayerType::Whole.as_str(), "WHOLE");
    }

    #[test]
    fn summary_total_zero_decode_tokens() {
        let s = ProfilerSummary { entries: vec![], decode_tokens: 0 };
        assert!(s.total_us_per_token() >= 0.0);
    }
}
