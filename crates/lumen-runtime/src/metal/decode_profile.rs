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
use std::sync::{Mutex, OnceLock};
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

/// [metal-R9 pos79 probe / Phase-2] Encoder ordinal(s) at which to split the
/// single per-token CB. `LUMEN_METAL_SPLIT_CB_AT_ORD=<N>` OR a comma list
/// `=a,b,c` (default unset => empty slice => byte-identical single-CB behavior).
///
/// A single value reproduces the Phase-1 pos79 probe EXACTLY (one split -> two
/// CBs). A comma list splits at each listed ordinal, yielding a multi-boundary
/// GPU-idle map (per-boundary `GPUStart(next)-GPUEnd(prev)` gaps) in ONE clean,
/// UNTRACED run -- P1 proved splits are throughput-free.
///
/// Only full-attn concurrent-proj CLOSE ordinals (odd N = 2*layer+1 for a
/// full-attn layer) ever match the split site (`decode_token_greedy_core`);
/// GDN-only ordinals are silently ignored (they never fire). Parsed, sorted, and
/// de-duplicated once; diagnostic only.
pub(crate) fn split_cb_ords() -> &'static [u32] {
    static CACHE: OnceLock<Vec<u32>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            std::env::var("LUMEN_METAL_SPLIT_CB_AT_ORD")
                .ok()
                .map(|s| {
                    let mut v: Vec<u32> = s
                        .split(',')
                        .filter_map(|x| x.trim().parse::<u32>().ok())
                        .collect();
                    v.sort_unstable();
                    v.dedup();
                    v
                })
                .unwrap_or_default()
        })
        .as_slice()
}

/// [metal-R9 Phase-2 Q2(a)] Microseconds to `usleep` at the driver's front-wait
/// point, gated by `LUMEN_METAL_DELAY_US=<us>` (default 0 => no-op). Perturbs the
/// CPU loop phase WITHOUT touching GPU work: if the ~62%-phase stall FOLLOWS the
/// delayed CPU event it moves; if it stays put it is GPU/OS-side. Cached once.
pub(crate) fn delay_us() -> u32 {
    use std::sync::atomic::AtomicI64;
    static CACHE: AtomicI64 = AtomicI64::new(-1);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur >= 0 {
        return cur as u32;
    }
    let v = std::env::var("LUMEN_METAL_DELAY_US")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0);
    CACHE.store(v as i64, Ordering::Relaxed);
    v
}

/// `usleep(3)` — libSystem. Q2(a) CPU-phase perturbation only.
pub(crate) fn usleep_us(us: u32) {
    extern "C" {
        fn usleep(useconds: u32) -> i32;
    }
    if us > 0 {
        unsafe {
            usleep(us);
        }
    }
}

// ============================================================================
// [metal-R9 Phase-2] CPU per-token event timeline (LUMEN_METAL_CPU_TIMELINE=1)
// ============================================================================
//
// Records `mach_absolute_time()` at every per-token CPU driver event in the lean
// decode loop into a global buffer, flushed to stderr ONCE at process exit
// (`atexit`). Zero cost when the flag is off (one relaxed atomic load + early
// return). Correlation: each event's mach time -> host seconds via the mach
// timebase, the SAME domain as `MTLCommandBuffer.GPUStartTime/GPUEndTime` on
// Apple Silicon (both == `CACurrentMediaTime`). Post-analysis compares each CPU
// event's time against the per-token GPU stall window
// `[GPUEnd(CB1), GPUStart(CB2)]` (emitted by the split-CB `[IDLEMAP]` log) to
// answer: which CPU event, if any, coincides with the ~62%-phase GPU stall?
// Domain agreement is validated empirically by the ordering invariants
// (commit <= GPUStart, GPUEnd <= wait-return) reported by the analyzer; a
// `sampleTimestamps` pair at run start/end reports the CPU<->GPU drift.

static CPU_TL_ENABLED: AtomicBool = AtomicBool::new(false);

#[inline]
pub(crate) fn cpu_timeline_enabled() -> bool {
    CPU_TL_ENABLED.load(Ordering::Relaxed)
}

// Event ids (labels mirrored in the analyzer).
pub(crate) const EV_ENTRY: u8 = 1; // lean driver call entry
pub(crate) const EV_ENCODE_START: u8 = 2; // before decode_token_greedy_core (encode CB(step))
pub(crate) const EV_ENCODE_END: u8 = 3; // after core returns the in-flight CB
pub(crate) const EV_SUBCB_COMMIT_PRE: u8 = 4; // before a split sub-CB commit (aux = enc_ord)
pub(crate) const EV_SUBCB_COMMIT_POST: u8 = 5; // after
pub(crate) const EV_TERM_COMMIT_PRE: u8 = 6; // before the terminal CB commit (core tail)
pub(crate) const EV_TERM_COMMIT_POST: u8 = 7; // after
pub(crate) const EV_WAIT_PRE: u8 = 8; // before front_cmd.wait_until_completed()
pub(crate) const EV_WAIT_POST: u8 = 9; // after the wait returns (CPU observes GPU done)
pub(crate) const EV_READBACK: u8 = 10; // after the token is read back from the ring
pub(crate) const EV_EXIT: u8 = 11; // lean driver call exit (return token)
pub(crate) const EV_DELAY_PRE: u8 = 12; // before a Q2(a) injected usleep (aux = us)
pub(crate) const EV_DELAY_POST: u8 = 13; // after

#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}
extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
    fn atexit(cb: extern "C" fn()) -> i32;
    fn pthread_threadid_np(thread: *mut core::ffi::c_void, thread_id: *mut u64) -> i32;
}

fn timebase() -> (u32, u32) {
    static TB: OnceLock<(u32, u32)> = OnceLock::new();
    *TB.get_or_init(|| {
        let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
        unsafe {
            mach_timebase_info(&mut info as *mut _);
        }
        (info.numer.max(1), info.denom.max(1))
    })
}

/// mach ticks -> host seconds (the `CACurrentMediaTime` / `GPUStartTime` domain).
pub(crate) fn mach_to_secs(t: u64) -> f64 {
    let (n, d) = timebase();
    (t as f64) * (n as f64) / (d as f64) / 1e9
}

#[inline]
pub(crate) fn mach_now() -> u64 {
    unsafe { mach_absolute_time() }
}

fn cur_tid() -> u64 {
    thread_local! {
        static TID: u64 = {
            let mut id = 0u64;
            unsafe { pthread_threadid_np(core::ptr::null_mut(), &mut id as *mut u64); }
            id
        };
    }
    TID.with(|t| *t)
}

// (mach_time, event, tok_tag, aux, tid)
type TlEvent = (u64, u8, u32, u32, u64);
static CPU_TL: Mutex<Vec<TlEvent>> = Mutex::new(Vec::new());

/// Record one CPU event. `tok` is the best-effort token tag (seq_pos or step);
/// `aux` carries an event-specific value (enc_ord for sub-CB commits, us for a
/// delay). No-op unless `LUMEN_METAL_CPU_TIMELINE=1`.
#[inline]
pub(crate) fn tl_mark(event: u8, tok: u32, aux: u32) {
    if !cpu_timeline_enabled() {
        return;
    }
    let t = mach_now();
    let tid = cur_tid();
    if let Ok(mut buf) = CPU_TL.lock() {
        buf.push((t, event, tok, aux, tid));
    }
}

extern "C" fn cpu_tl_flush_atexit() {
    flush_cpu_timeline();
}

/// Drain the CPU timeline buffer to stderr (one line per event). Called once at
/// process exit via `atexit`; clears the buffer so a manual call cannot double.
pub(crate) fn flush_cpu_timeline() {
    let mut buf = match CPU_TL.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if buf.is_empty() {
        return;
    }
    use std::io::Write;
    let stderr = std::io::stderr();
    let mut h = stderr.lock();
    let _ = writeln!(h, "[CPUTL-BEGIN] count={}", buf.len());
    for (t, ev, tok, aux, tid) in buf.iter() {
        let _ = writeln!(
            h,
            "[CPUTL] t_s={:.9} ev={} tok={} aux={} tid={}",
            mach_to_secs(*t),
            ev,
            tok,
            aux,
            tid
        );
    }
    let _ = writeln!(h, "[CPUTL-END]");
    let _ = h.flush();
    buf.clear();
}

pub(crate) fn init_from_env() {
    if std::env::var("LUMEN_METAL_DECODE_PROFILE").ok().as_deref() == Some("1") {
        ENABLED.store(true, Ordering::Relaxed);
    }
    if std::env::var("LUMEN_METAL_DECODE_GPUTIME").ok().as_deref() == Some("1") {
        GPUTIME_ENABLED.store(true, Ordering::Relaxed);
    }
    if std::env::var("LUMEN_METAL_CPU_TIMELINE").ok().as_deref() == Some("1") {
        CPU_TL_ENABLED.store(true, Ordering::Relaxed);
        if let Ok(mut b) = CPU_TL.lock() {
            b.reserve(1 << 16);
        }
        // Flush the in-memory timeline to stderr once at normal process exit.
        unsafe {
            atexit(cpu_tl_flush_atexit);
        }
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
