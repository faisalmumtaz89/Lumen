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

/// True when LUMEN_METAL_DECODE_GPUTIME=1. Used to gate the clean single-CB
/// GPU-busy-vs-wall measurement in the greedy decode path (no CB splitting).
#[inline]
pub(crate) fn gputime_enabled() -> bool {
    GPUTIME_ENABLED.load(Ordering::Relaxed)
}

pub(crate) fn init_from_env() {
    if std::env::var("LUMEN_METAL_DECODE_PROFILE").ok().as_deref() == Some("1") {
        ENABLED.store(true, Ordering::Relaxed);
    }
    if std::env::var("LUMEN_METAL_DECODE_GPUTIME").ok().as_deref() == Some("1") {
        GPUTIME_ENABLED.store(true, Ordering::Relaxed);
    }
}

/// Diagnostic sub-stage skip bitmask for the full-attention decode block, parsed
/// once from `LUMEN_METAL_FULLATTN_SUBSKIP` (decimal or 0x-hex u32). When a bit is
/// set, the matching sub-stage's GPU dispatch is skipped so its cost can be read
/// off the `full_attn` per-section GPU time. Skipping corrupts the output (this is
/// a timing-attribution tool only). 0 / unset => no-op (every sub-stage runs).
///
/// Bit layout (see `FULLATTN_SKIP_*` constants in `decode_greedy.rs`):
///   bit0 K proj, bit1 V proj, bit2 RoPE+KV-write, bit3 attention (MHA/flash),
///   bit4 Q+gate proj, bit5 Wo proj, bit6 deinterleave/norm/assemble + misc.
#[inline]
pub(crate) fn fullattn_subskip() -> u32 {
    use std::sync::atomic::AtomicU32;
    // Sentinel bit31 marks "already resolved" so a genuine mask of 0 is cached.
    const RESOLVED: u32 = 0x8000_0000;
    static CACHE: AtomicU32 = AtomicU32::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur & RESOLVED != 0 {
        return cur & !RESOLVED;
    }
    let v = std::env::var("LUMEN_METAL_FULLATTN_SUBSKIP")
        .ok()
        .and_then(|s| {
            let t = s.trim();
            if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                u32::from_str_radix(hex, 16).ok()
            } else {
                t.parse::<u32>().ok()
            }
        })
        .unwrap_or(0)
        & !RESOLVED; // never let the sentinel be a user-supplied bit
    CACHE.store(v | RESOLVED, Ordering::Relaxed);
    v
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
        let (gpu_sum, n) = GPU_ACC.with(|a| {
            let v = *a.borrow();
            v
        });
        let wall_sum = GPU_WALL_ACC.with(|a| *a.borrow());
        let gpu_ms = gpu_sum / n as f64 * 1000.0;
        let wall_ms = if n > 1 {
            wall_sum / (n as f64 - 1.0) * 1000.0
        } else {
            0.0
        };
        let util = if wall_ms > 0.0 {
            gpu_ms / wall_ms * 100.0
        } else {
            0.0
        };
        eprintln!(
            "[decode-gputime] over {} tokens: GPU_busy={:.3} ms/tok  wall={:.3} ms/tok  \
             GPU_util={:.1}%  (idle/CPU gap={:.3} ms/tok)",
            n,
            gpu_ms,
            wall_ms,
            util,
            (wall_ms - gpu_ms).max(0.0)
        );
        GPU_ACC.with(|a| *a.borrow_mut() = (0.0, 0));
        GPU_WALL_ACC.with(|a| *a.borrow_mut() = 0.0);
    }
}

thread_local! {
    /// Lean-pipeline wall accumulator: (sum_gpu_secs, sum_wall_secs, count,
    /// last completion Instant). Wall = time between successive front-CB
    /// completions (the steady-state per-token wall of the async path).
    static LEAN_ACC: RefCell<(f64, f64, u64)> = const { RefCell::new((0.0, 0.0, 0)) };
    static LEAN_LAST: RefCell<Option<Instant>> = const { RefCell::new(None) };
}

/// Record one lean-pipeline token: accumulate the front CB's true GPU-busy time
/// and the wall between successive completions, and every 64 tokens print
/// lean's effective wall/tok + GPU_util. Comparable head-to-head with the
/// sequential `record_gpu_time` output. No-op unless DECODE_GPUTIME=1.
pub(crate) fn record_lean_wall(gpu_secs: f64) {
    if !GPUTIME_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let now = Instant::now();
    let wall = LEAN_LAST.with(|l| {
        let prev = l.borrow_mut().replace(now);
        prev.map(|p| now.duration_since(p).as_secs_f64())
    });
    let fire = LEAN_ACC.with(|a| {
        let mut a = a.borrow_mut();
        a.0 += gpu_secs;
        if let Some(w) = wall {
            a.1 += w;
        }
        a.2 += 1;
        a.2 >= 64
    });
    if fire {
        let (gpu_sum, wall_sum, n) = LEAN_ACC.with(|a| *a.borrow());
        let gpu_ms = gpu_sum / n as f64 * 1000.0;
        // wall accumulates (n-1) intervals (first token has no predecessor).
        let wall_ms = if n > 1 {
            wall_sum / (n as f64 - 1.0) * 1000.0
        } else {
            0.0
        };
        let util = if wall_ms > 0.0 {
            gpu_ms / wall_ms * 100.0
        } else {
            0.0
        };
        eprintln!(
            "[decode-lean] over {n} tokens: wall={wall_ms:.3} ms/tok  \
             GPU_busy={gpu_ms:.3} ms/tok  GPU_util={util:.1}%  \
             (recovered_vs_serial_gap = wall-GPU = {:.3} ms/tok)",
            (wall_ms - gpu_ms).max(0.0)
        );
        LEAN_ACC.with(|a| *a.borrow_mut() = (0.0, 0.0, 0));
    }
}

thread_local! {
    /// Encode/exec split accumulator (sum_encode_secs, sum_wait_secs,
    /// sum_gpu_secs, count). Encode = CPU time from new_command_buffer to
    /// commit; wait = wall time the commit_and_wait blocks; gpu = true GPU
    /// busy (GPUEndTime-GPUStartTime). Reported with the same 64-tok window
    /// as record_gpu_time. No-op unless LUMEN_METAL_DECODE_GPUTIME=1.
    static SPLIT_ACC: RefCell<(f64, f64, f64, u64)> = const { RefCell::new((0.0, 0.0, 0.0, 0)) };
}

/// Accumulate the CPU-encode-vs-GPU-exec split for one decode token and, every
/// 64 tokens, print the decomposition: how much of the per-token wall is spent
/// (a) CPU-encoding the command buffer, (b) blocked in commit_and_wait, and how
/// that wait compares to the true GPU-busy time. This is the decisive STEP-1
/// measurement for whether an encode-once-replay / CPU-GPU-overlap paradigm can
/// recover wall time: if CPU-encode is a large fraction of wall AND it is fully
/// serial before the GPU runs, overlap can hide it. No-op unless
/// LUMEN_METAL_DECODE_GPUTIME=1.
pub(crate) fn record_encode_split(encode_secs: f64, wait_secs: f64, gpu_secs: f64) {
    if !GPUTIME_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let fire = SPLIT_ACC.with(|a| {
        let mut a = a.borrow_mut();
        a.0 += encode_secs;
        a.1 += wait_secs;
        a.2 += gpu_secs;
        a.3 += 1;
        a.3 >= 64
    });
    if fire {
        let (enc_sum, wait_sum, gpu_sum, n) = SPLIT_ACC.with(|a| *a.borrow());
        let nf = n as f64;
        let enc_ms = enc_sum / nf * 1000.0;
        let wait_ms = wait_sum / nf * 1000.0;
        let gpu_ms = gpu_sum / nf * 1000.0;
        // Within-token wall = encode + wait (the loop body is serial:
        // encode the CB, then block on commit_and_wait). The wait should be
        // ~= gpu_busy + launch/scheduling tail; encode is the pure CPU cost
        // an overlap paradigm targets.
        let body_ms = enc_ms + wait_ms;
        let tail_ms = (wait_ms - gpu_ms).max(0.0);
        eprintln!(
            "[decode-split] over {n} tokens: CPU_encode={enc_ms:.3} ms/tok  \
             commit_wait={wait_ms:.3} ms/tok  GPU_busy={gpu_ms:.3} ms/tok  \
             (within-tok body={body_ms:.3}  wait_tail_over_gpu={tail_ms:.3})  \
             encode_frac_of_body={:.1}%",
            if body_ms > 0.0 {
                enc_ms / body_ms * 100.0
            } else {
                0.0
            }
        );
        SPLIT_ACC.with(|a| *a.borrow_mut() = (0.0, 0.0, 0.0, 0));
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
    // Diagnostic override: report after this many decode tokens instead of the
    // built-in cadence. Lets a short run (e.g. one whose output hits EOS before
    // the default 64-token mark) still emit a valid per-call average; the us/call
    // is a per-call mean and is independent of the token count. Unset => default.
    let every = std::env::var("LUMEN_METAL_DECODE_PROFILE_EVERY")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(every);
    let fire = TOK_COUNT.with(|c| {
        let mut c = c.borrow_mut();
        *c += 1;
        if *c >= every {
            *c = 0;
            true
        } else {
            false
        }
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
    let elapsed = MARK.with(|m| {
        m.borrow_mut()
            .take()
            .map(|t| t.elapsed())
            .unwrap_or_default()
    });
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
    let elapsed = MARK.with(|m| {
        m.borrow_mut()
            .take()
            .map(|t| t.elapsed())
            .unwrap_or_default()
    });
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
    let mut v: Vec<(&'static str, Duration, u64)> =
        ACCUM.with(|a| a.borrow().iter().map(|(k, (d, n))| (*k, *d, *n)).collect());
    if v.is_empty() {
        eprintln!("[decode-profile] no samples");
        return;
    }
    v.sort_by(|x, y| y.1.cmp(&x.1));
    let total: Duration = v.iter().map(|(_, d, _)| *d).sum();
    eprintln!();
    eprintln!("===== Metal DECODE per-section profile (split-CB, Option A) =====");
    eprintln!(
        "{:<24} {:>12} {:>10} {:>9} {:>12}",
        "section", "total_ms", "calls", "% tok", "us/call"
    );
    eprintln!("{}", "-".repeat(72));
    for (label, dur, n) in &v {
        let ms = dur.as_secs_f64() * 1000.0;
        let pct = if total.as_nanos() > 0 {
            (dur.as_nanos() as f64 / total.as_nanos() as f64) * 100.0
        } else {
            0.0
        };
        let us_call = if *n > 0 {
            (ms * 1000.0) / *n as f64
        } else {
            0.0
        };
        eprintln!(
            "{:<24} {:>12.3} {:>10} {:>8.2}% {:>12.2}",
            label, ms, n, pct, us_call
        );
    }
    eprintln!("{}", "-".repeat(72));
    eprintln!("{:<24} {:>12.3}", "TOTAL", total.as_secs_f64() * 1000.0);
    eprintln!("NOTE: split-CB commit_and_wait per section. Absolute > production;");
    eprintln!("      relative ranking is the informative output.");
    eprintln!();

    // TRUE GPU-time table (overhead-free; the accurate per-section breakdown).
    let mut g: Vec<(&'static str, f64, u64)> =
        GPU_SECTION.with(|a| a.borrow().iter().map(|(k, (s, n))| (*k, *s, *n)).collect());
    if !g.is_empty() {
        g.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        let gtotal: f64 = g.iter().map(|(_, s, _)| *s).sum();
        eprintln!("===== Metal DECODE per-section TRUE GPU time (GPUStartTime/EndTime) =====");
        eprintln!(
            "{:<24} {:>12} {:>10} {:>9} {:>12}",
            "section", "gpu_ms", "calls", "% gpu", "us/call"
        );
        eprintln!("{}", "-".repeat(72));
        for (label, secs, n) in &g {
            let ms = secs * 1000.0;
            let pct = if gtotal > 0.0 {
                secs / gtotal * 100.0
            } else {
                0.0
            };
            let us_call = if *n > 0 { ms * 1000.0 / *n as f64 } else { 0.0 };
            eprintln!(
                "{:<24} {:>12.3} {:>10} {:>8.2}% {:>12.2}",
                label, ms, n, pct, us_call
            );
        }
        eprintln!("{}", "-".repeat(72));
        eprintln!("{:<24} {:>12.3}", "GPU TOTAL", gtotal * 1000.0);
        eprintln!();
    }
}
