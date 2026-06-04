//! Per-section CPU-side GPU timing for Metal prefill.
//!
//! GATED: only active when `LUMEN_METAL_PROFILE=1` is set in the environment.
//!
//! ## Approach
//!
//! Lumen's prefill submits all kernels into ONE command buffer with ONE
//! `commit_and_wait()` at the end (see `metal/prefill.rs::prefill`). That
//! design is optimal for production latency (1 GPU-CPU round-trip) but
//! hides per-kernel timing.
//!
//! When profiling is enabled, we split the single CB into many short CBs
//! at known section boundaries: each call to
//! `MetalCommandBuffer::new_compute_encoder()` (and `_concurrent`) first
//! commits-and-waits the prior CB, records the elapsed CPU-side wall time
//! under the `current_section` label, then creates a fresh CB on the
//! original queue and continues encoding.
//!
//! Each call site near a section boundary calls `set_section(label)` BEFORE
//! the next `new_compute_encoder()` to attribute the just-finished section.
//! Inner dispatches inside one section (the same encoder) keep the same
//! label, so multiple dispatches inside one encoder accumulate into one
//! bucket.
//!
//! ## Limitations of Option A (per-encoder commit/wait)
//!
//! Adding `commit_and_wait()` between sections forces GPU-CPU synchronisation
//! and serialises sections that could have overlapped. The ABSOLUTE timing
//! is therefore an upper bound (slower than production). The RELATIVE
//! ranking among sections is still informative for finding hot kernels —
//! kernels that are MUCH bigger than others will still dominate even with
//! the per-section sync overhead added.
//!
//! For production-grade per-kernel timing without sync overhead, use the
//! Apple `MTLCounterSampleBuffer` with `MTLCommonCounterSetTimestamp`. That
//! is a follow-up once the hot kernels are identified.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Global enable flag for Metal profiling. Read once at backend
/// construction from `LUMEN_METAL_PROFILE=1` and from `set_profile()`.
static PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Global enable flag for GDN deep-profile mode. Read once at backend
/// construction from `LUMEN_METAL_PROFILE_GDN=1`. When set, the GDN
/// batched-prefill megakernel encoder is split into one CB per
/// sub-dispatch (RMSNorm, QKV GEMM, attn-gate GEMM, alpha GEMM, beta
/// GEMM, compute-gates, conv1d+silu, l2-norm, phase2a state, phase2b
/// norm+gate, ssm-out GEMM, ffn-norm). Adds CPU-GPU sync overhead per
/// split so absolute timing is inflated; the relative distribution
/// across sub-dispatches is the informative output.
static PROFILE_GDN_ENABLED: AtomicBool = AtomicBool::new(false);

/// Returns whether Metal profiling is currently enabled.
#[inline]
pub(crate) fn is_enabled() -> bool {
    PROFILE_ENABLED.load(Ordering::Relaxed)
}

/// Returns whether GDN deep-profile mode is enabled. Implies `is_enabled()`.
#[inline]
pub(crate) fn is_gdn_deep_enabled() -> bool {
    PROFILE_GDN_ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable Metal profiling globally.
pub(crate) fn set_enabled(on: bool) {
    PROFILE_ENABLED.store(on, Ordering::Relaxed);
}

/// Initialize profiling state from the `LUMEN_METAL_PROFILE` env var.
/// Called once during backend construction.
pub(crate) fn init_from_env() {
    if std::env::var("LUMEN_METAL_PROFILE").ok().as_deref() == Some("1") {
        set_enabled(true);
    }
    // GDN deep-profile implies overall profiling.
    if std::env::var("LUMEN_METAL_PROFILE_GDN").ok().as_deref() == Some("1") {
        set_enabled(true);
        PROFILE_GDN_ENABLED.store(true, Ordering::Relaxed);
    }
}

thread_local! {
    /// Label of the section that is currently being encoded but has NOT yet
    /// been committed. The FFI layer reads this when it commits the
    /// in-flight CB (record_section_end attributes the elapsed time here).
    ///
    /// Distinct from `NEXT_SECTION` so that a call site can announce the
    /// next section label (via `set_section`) BEFORE the in-flight section
    /// is committed without misattributing the in-flight section's time.
    static IN_FLIGHT_SECTION: RefCell<&'static str> = const { RefCell::new("(unlabelled)") };

    /// Label to adopt as `IN_FLIGHT_SECTION` once the in-flight CB has been
    /// committed. Set by `set_section`, adopted by `record_section_end`.
    static NEXT_SECTION: RefCell<Option<&'static str>> = const { RefCell::new(None) };

    /// Accumulator: section_label -> (total_duration, call_count)
    static ACCUM: RefCell<HashMap<&'static str, (Duration, u64)>> =
        RefCell::new(HashMap::new());

    /// Instant of the last `mark_section_start()` call. Used to time the
    /// commit-and-wait that closes the prior CB.
    static LAST_MARK: RefCell<Option<Instant>> = const { RefCell::new(None) };
}

/// Announce the label for the section that will be encoded NEXT (the one
/// whose `new_compute_encoder()` call follows). Labels are static so the
/// accumulator keys do not allocate.
///
/// Must be called BEFORE the corresponding `new_compute_encoder()`. Stores
/// the label in `NEXT_SECTION` rather than mutating `IN_FLIGHT_SECTION`
/// directly — the in-flight section's elapsed time is recorded under its
/// own label by `record_section_end`, NOT the next one.
pub(crate) fn set_section(label: &'static str) {
    if !is_enabled() {
        return;
    }
    NEXT_SECTION.with(|n| {
        *n.borrow_mut() = Some(label);
    });
}

/// Begin GPU-side timing for the current section. Called by the FFI layer
/// right BEFORE encoder dispatch begins (after the prior CB has been
/// committed and waited). Captures the start instant.
pub(crate) fn mark_section_start() {
    if !is_enabled() {
        return;
    }
    LAST_MARK.with(|m| {
        *m.borrow_mut() = Some(Instant::now());
    });
}

/// Accumulate the elapsed time since `mark_section_start()` under the
/// CURRENT in-flight section label (NOT the next one announced by
/// `set_section`). Then promote `NEXT_SECTION` to `IN_FLIGHT_SECTION` so
/// subsequent records attribute to the new section.
///
/// Called by the FFI layer right AFTER commit-and-wait completes for the
/// in-flight CB. Idempotent for missing LAST_MARK (first call records 0).
pub(crate) fn record_section_end() {
    if !is_enabled() {
        return;
    }
    let elapsed = LAST_MARK.with(|m| {
        m.borrow_mut().take().map(|t| t.elapsed()).unwrap_or_default()
    });
    let label = IN_FLIGHT_SECTION.with(|s| *s.borrow());
    ACCUM.with(|a| {
        let mut a = a.borrow_mut();
        let entry = a.entry(label).or_insert((Duration::ZERO, 0));
        entry.0 += elapsed;
        entry.1 += 1;
    });
    // Promote the next label to in-flight (if any was announced).
    if let Some(next) = NEXT_SECTION.with(|n| n.borrow_mut().take()) {
        IN_FLIGHT_SECTION.with(|s| {
            *s.borrow_mut() = next;
        });
    }
}

/// Reset the accumulator. Called before a new profiling run.
pub(crate) fn reset_accum() {
    ACCUM.with(|a| a.borrow_mut().clear());
    LAST_MARK.with(|m| *m.borrow_mut() = None);
    NEXT_SECTION.with(|n| *n.borrow_mut() = None);
    IN_FLIGHT_SECTION.with(|s| *s.borrow_mut() = "(unlabelled)");
}

/// Returns a snapshot of the accumulator, sorted by total time descending.
/// `(label, total_duration, call_count)`.
pub(crate) fn snapshot() -> Vec<(&'static str, Duration, u64)> {
    ACCUM.with(|a| {
        let mut v: Vec<(&'static str, Duration, u64)> = a
            .borrow()
            .iter()
            .map(|(k, (d, n))| (*k, *d, *n))
            .collect();
        v.sort_by(|x, y| y.1.cmp(&x.1));
        v
    })
}

/// Print a formatted profile report to stderr.
pub(crate) fn print_report() {
    if !is_enabled() {
        eprintln!("[metal-profile] not enabled (set LUMEN_METAL_PROFILE=1 or pass --profile)");
        return;
    }
    let snap = snapshot();
    if snap.is_empty() {
        eprintln!("[metal-profile] no samples recorded");
        return;
    }
    let total: Duration = snap.iter().map(|(_, d, _)| *d).sum();
    eprintln!();
    eprintln!("===== Metal prefill per-section profile =====");
    eprintln!(
        "{:<40} {:>10} {:>10} {:>8} {:>10}",
        "section", "total_ms", "calls", "% prefill", "ms/call"
    );
    eprintln!("{}", "-".repeat(82));
    for (label, dur, n) in &snap {
        let ms = dur.as_secs_f64() * 1000.0;
        let pct = if total.as_nanos() > 0 {
            (dur.as_nanos() as f64 / total.as_nanos() as f64) * 100.0
        } else {
            0.0
        };
        let per_call = if *n > 0 { ms / *n as f64 } else { 0.0 };
        eprintln!(
            "{:<40} {:>10.3} {:>10} {:>7.2}% {:>10.4}",
            label, ms, n, pct, per_call
        );
    }
    eprintln!("{}", "-".repeat(82));
    eprintln!(
        "{:<40} {:>10.3} {:>10} {:>8}",
        "TOTAL",
        total.as_secs_f64() * 1000.0,
        snap.iter().map(|(_, _, n)| n).sum::<u64>(),
        ""
    );
    eprintln!();
    eprintln!("NOTE: Option A profiler (split CB + commit_and_wait per section).");
    eprintln!("      Absolute total > production prefill (sync overhead added).");
    eprintln!("      Relative ranking remains informative for hot-kernel triage.");
    eprintln!();
}
