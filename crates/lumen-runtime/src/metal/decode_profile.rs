//! Per-section CPU-side GPU timing for Metal DECODE (single-token).
//!
//! GATED: only active when `LUMEN_METAL_DECODE_PROFILE=1` is set.
//!
//! ## Approach (mirrors `metal/profile.rs` for prefill)
//!
//! Production decode submits embed + ALL layers + lm_head + argmax into ONE
//! command buffer with ONE `commit_and_wait()` (see `decode_greedy.rs`). That
//! is optimal for latency but hides per-section GPU time.
//!
//! When this profiler is enabled, the decode loop calls `boundary(label)` at
//! known section boundaries. `boundary` ends the current encoder, commits and
//! waits the in-flight CB, attributes the elapsed wall time to the PREVIOUS
//! label, then starts a fresh CB + serial encoder and returns it so encoding
//! continues.
//!
//! Adding `commit_and_wait()` between sections forces GPU-CPU sync and
//! serialises sections that could overlap, so ABSOLUTE timing is an upper
//! bound (slower than production). The RELATIVE ranking among sections is the
//! informative output for hot-section triage.
//!
//! The accumulator aggregates by label across ALL layers of ALL tokens in a
//! run, so e.g. "gdn_attn" sums every GDN attention block, "moe_ffn" sums
//! every MoE FFN block, letting us see which layer-type dominates decode.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

static ENABLED: AtomicBool = AtomicBool::new(false);

#[inline]
pub(crate) fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

static GPUTIME_ENABLED: AtomicBool = AtomicBool::new(false);

pub(crate) fn init_from_env() {
    if std::env::var("LUMEN_METAL_DECODE_PROFILE").ok().as_deref() == Some("1") {
        ENABLED.store(true, Ordering::Relaxed);
    }
    if std::env::var("LUMEN_METAL_DECODE_GPUTIME").ok().as_deref() == Some("1") {
        GPUTIME_ENABLED.store(true, Ordering::Relaxed);
    }
}

thread_local! {
    /// (sum of per-token true GPU busy seconds, count, last wall Instant).
    static GPU_ACC: RefCell<(f64, u64)> = const { RefCell::new((0.0, 0)) };
    static GPU_WALL_LAST: RefCell<Option<Instant>> = const { RefCell::new(None) };
    static GPU_WALL_ACC: RefCell<f64> = const { RefCell::new(0.0) };
}

/// Accumulate the true GPU busy time (seconds) for one decode token's CB and,
/// every 64 tokens, print GPU-busy vs wall-clock so we can see whether decode
/// is GPU-execution-bound or CPU/scheduling-bound. No-op unless
/// LUMEN_METAL_DECODE_GPUTIME=1.
pub(crate) fn record_gpu_time(gpu_secs: f64) {
    if !GPUTIME_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let now = Instant::now();
    let wall = GPU_WALL_LAST.with(|l| {
        let prev = l.borrow_mut().replace(now);
        prev.map(|p| now.duration_since(p).as_secs_f64())
    });
    if let Some(w) = wall {
        GPU_WALL_ACC.with(|a| *a.borrow_mut() += w);
    }
    let fire = GPU_ACC.with(|a| {
        let mut a = a.borrow_mut();
        a.0 += gpu_secs;
        a.1 += 1;
        a.1 >= 64
    });
    if fire {
        let (gpu_sum, n) = GPU_ACC.with(|a| { let v = *a.borrow(); v });
        let wall_sum = GPU_WALL_ACC.with(|a| *a.borrow());
        let gpu_ms = gpu_sum / n as f64 * 1000.0;
        let wall_ms = if n > 1 { wall_sum / (n as f64 - 1.0) * 1000.0 } else { 0.0 };
        let util = if wall_ms > 0.0 { gpu_ms / wall_ms * 100.0 } else { 0.0 };
        eprintln!(
            "[decode-gputime] over {} tokens: GPU_busy={:.3} ms/tok  wall={:.3} ms/tok  \
             GPU_util={:.1}%  (idle/CPU gap={:.3} ms/tok)",
            n, gpu_ms, wall_ms, util, (wall_ms - gpu_ms).max(0.0)
        );
        GPU_ACC.with(|a| *a.borrow_mut() = (0.0, 0));
        GPU_WALL_ACC.with(|a| *a.borrow_mut() = 0.0);
    }
}

thread_local! {
    /// Label of the in-flight (not-yet-committed) section.
    static IN_FLIGHT: RefCell<&'static str> = const { RefCell::new("(start)") };
    /// Accumulator: label -> (total_duration, call_count).
    static ACCUM: RefCell<HashMap<&'static str, (Duration, u64)>> =
        RefCell::new(HashMap::new());
    /// Instant the in-flight section started encoding.
    static MARK: RefCell<Option<Instant>> = const { RefCell::new(None) };
    /// Count of completed tokens since the last report.
    static TOK_COUNT: RefCell<u64> = const { RefCell::new(0) };
}

/// Increment the token counter; when it reaches `every`, print a report and
/// reset both the accumulator and the counter. Lets a long generation emit a
/// periodic, statistically meaningful breakdown.
pub(crate) fn maybe_report_and_reset(every: u64) {
    if !is_enabled() {
        return;
    }
    let fire = TOK_COUNT.with(|c| {
        let mut c = c.borrow_mut();
        *c += 1;
        if *c >= every { *c = 0; true } else { false }
    });
    if fire {
        print_report();
        reset();
    }
}

/// Reset accumulator + clock; sets the first in-flight label.
pub(crate) fn begin(first_label: &'static str) {
    if !is_enabled() {
        return;
    }
    IN_FLIGHT.with(|s| *s.borrow_mut() = first_label);
    MARK.with(|m| *m.borrow_mut() = Some(Instant::now()));
}

/// Record elapsed time since the last mark under the in-flight label, then
/// adopt `next_label` as the new in-flight section and restart the clock.
/// Called by the decode loop AFTER it has committed-and-waited the prior CB.
pub(crate) fn record_and_advance(next_label: &'static str) {
    if !is_enabled() {
        return;
    }
    let elapsed = MARK.with(|m| m.borrow_mut().take().map(|t| t.elapsed()).unwrap_or_default());
    let label = IN_FLIGHT.with(|s| *s.borrow());
    ACCUM.with(|a| {
        let mut a = a.borrow_mut();
        let e = a.entry(label).or_insert((Duration::ZERO, 0));
        e.0 += elapsed;
        e.1 += 1;
    });
    IN_FLIGHT.with(|s| *s.borrow_mut() = next_label);
    MARK.with(|m| *m.borrow_mut() = Some(Instant::now()));
}

thread_local! {
    /// True GPU-time accumulator: label -> (total_gpu_secs, count).
    static GPU_SECTION: RefCell<HashMap<&'static str, (f64, u64)>> =
        RefCell::new(HashMap::new());
}

/// Record the TRUE GPU busy time (seconds, from GPUStartTime/GPUEndTime) of the
/// just-committed sub-CB under the CURRENT in-flight label. Does NOT advance the
/// label (the companion `record_and_advance` owns label advancement). Call this
/// at the SAME boundary, immediately BEFORE `record_and_advance`, so it attributes
/// to the section that just finished. The `_label` argument is the section name
/// for readability at the call site and is ignored. Overhead-free.
pub(crate) fn record_gpu(gpu_secs: f64, _label: &'static str) {
    if !is_enabled() {
        return;
    }
    let label = IN_FLIGHT.with(|s| *s.borrow());
    GPU_SECTION.with(|a| {
        let mut a = a.borrow_mut();
        let e = a.entry(label).or_insert((0.0, 0));
        e.0 += gpu_secs;
        e.1 += 1;
    });
}

/// Record the final section's GPU time (no advance).
pub(crate) fn record_gpu_final(gpu_secs: f64) {
    if !is_enabled() {
        return;
    }
    let label = IN_FLIGHT.with(|s| *s.borrow());
    GPU_SECTION.with(|a| {
        let mut a = a.borrow_mut();
        let e = a.entry(label).or_insert((0.0, 0));
        e.0 += gpu_secs;
        e.1 += 1;
    });
}

/// Record the final in-flight section (call after the last commit_and_wait of
/// a token) without starting a new one.
pub(crate) fn record_final() {
    if !is_enabled() {
        return;
    }
    let elapsed = MARK.with(|m| m.borrow_mut().take().map(|t| t.elapsed()).unwrap_or_default());
    let label = IN_FLIGHT.with(|s| *s.borrow());
    ACCUM.with(|a| {
        let mut a = a.borrow_mut();
        let e = a.entry(label).or_insert((Duration::ZERO, 0));
        e.0 += elapsed;
        e.1 += 1;
    });
}

pub(crate) fn reset() {
    ACCUM.with(|a| a.borrow_mut().clear());
    GPU_SECTION.with(|a| a.borrow_mut().clear());
    MARK.with(|m| *m.borrow_mut() = None);
    IN_FLIGHT.with(|s| *s.borrow_mut() = "(start)");
}

/// Print a formatted report to stderr, sorted by total time descending.
pub(crate) fn print_report() {
    if !is_enabled() {
        return;
    }
    let mut v: Vec<(&'static str, Duration, u64)> = ACCUM.with(|a| {
        a.borrow().iter().map(|(k, (d, n))| (*k, *d, *n)).collect()
    });
    if v.is_empty() {
        eprintln!("[decode-profile] no samples");
        return;
    }
    v.sort_by(|x, y| y.1.cmp(&x.1));
    let total: Duration = v.iter().map(|(_, d, _)| *d).sum();
    eprintln!();
    eprintln!("===== Metal DECODE per-section profile (split-CB, Option A) =====");
    eprintln!("{:<24} {:>12} {:>10} {:>9} {:>12}", "section", "total_ms", "calls", "% tok", "us/call");
    eprintln!("{}", "-".repeat(72));
    for (label, dur, n) in &v {
        let ms = dur.as_secs_f64() * 1000.0;
        let pct = if total.as_nanos() > 0 {
            (dur.as_nanos() as f64 / total.as_nanos() as f64) * 100.0
        } else { 0.0 };
        let us_call = if *n > 0 { (ms * 1000.0) / *n as f64 } else { 0.0 };
        eprintln!("{:<24} {:>12.3} {:>10} {:>8.2}% {:>12.2}", label, ms, n, pct, us_call);
    }
    eprintln!("{}", "-".repeat(72));
    eprintln!("{:<24} {:>12.3}", "TOTAL", total.as_secs_f64() * 1000.0);
    eprintln!("NOTE: split-CB commit_and_wait per section. Absolute > production;");
    eprintln!("      relative ranking is the informative output.");
    eprintln!();

    // TRUE GPU-time table (overhead-free; the accurate per-section breakdown).
    let mut g: Vec<(&'static str, f64, u64)> = GPU_SECTION.with(|a| {
        a.borrow().iter().map(|(k, (s, n))| (*k, *s, *n)).collect()
    });
    if !g.is_empty() {
        g.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        let gtotal: f64 = g.iter().map(|(_, s, _)| *s).sum();
        eprintln!("===== Metal DECODE per-section TRUE GPU time (GPUStartTime/EndTime) =====");
        eprintln!("{:<24} {:>12} {:>10} {:>9} {:>12}", "section", "gpu_ms", "calls", "% gpu", "us/call");
        eprintln!("{}", "-".repeat(72));
        for (label, secs, n) in &g {
            let ms = secs * 1000.0;
            let pct = if gtotal > 0.0 { secs / gtotal * 100.0 } else { 0.0 };
            let us_call = if *n > 0 { ms * 1000.0 / *n as f64 } else { 0.0 };
            eprintln!("{:<24} {:>12.3} {:>10} {:>8.2}% {:>12.2}", label, ms, n, pct, us_call);
        }
        eprintln!("{}", "-".repeat(72));
        eprintln!("{:<24} {:>12.3}", "GPU TOTAL", gtotal * 1000.0);
        eprintln!();
    }
}
