//! CUDA per-phase decode profiler (CUDA-event brackets).
//!
//! # What this measures
//!
//! A decode token is bracketed by CUDA events recorded on the single decode
//! stream. Each *phase* is a source region containing one or more kernel
//! launches; `begin`/`end` record an event before and after the region, and
//! `elapsed(begin, end)` yields the **GPU-timeline span** of that region.
//!
//! A span is NOT "busy time". It is the wall interval on the GPU timeline
//! between two markers, so it includes any idle the GPU spent inside the
//! region waiting for the host to submit the next launch. This is deliberate
//! and is the reason three separate quantities are reported per token:
//!
//! * `wall_us`         -- host `Instant` across the whole token
//! * `gpu_span_us`     -- events around the whole token (GPU timeline)
//! * `attributed_us`   -- sum of the depth-0 phase spans
//!
//! from which the two honesty residuals fall out:
//!
//! * `gpu_unattributed_us = gpu_span_us - attributed_us`
//!   GPU-timeline time inside the token that no phase claimed: inter-phase
//!   submission gaps plus any launch site that is not bracketed. Work that is
//!   silently missing from the phase table therefore SHOWS UP HERE rather than
//!   vanishing. It is never negative by construction (phases nest inside the
//!   token bracket).
//! * `host_outside_span_us = wall_us - gpu_span_us`
//!   Host time before the first event executes and after the last one:
//!   submission latency, `cuStreamSynchronize`, the 4-byte D2H readback.
//!
//! # Phase depth
//!
//! Depth-0 phases partition the token: every kernel launch on the decode path
//! lies inside exactly one of them. Only depth-0 spans are summed into
//! `attributed_us`.
//!
//! Depth-1 phases are strictly nested inside a depth-0 phase and refine it.
//! They are reported but NEVER summed into `attributed_us`, so no work is
//! double counted.
//!
//! # Enabling
//!
//! `LUMEN_CUDA_PROFILE` (default unset = OFF):
//!
//! * unset / `0` / `off` -- disabled. Every entry point returns after reading
//!   one cached `u8`. No event is created, none is recorded, no lock is taken,
//!   and no synchronization is added: behaviour is unchanged.
//! * `1` -- coarse. Depth-0 phases only. On Qwen3.5-9B (24 GDN + 8 full-attn)
//!   that is 1 embed + 24 gdn_attn + 8 full_attn + 32 ffn + 32 layer_commit +
//!   1 head + 1 argmax = 99 brackets = 198 event records per token.
//! * `2` -- fine. Depth-0 + depth-1: 99 + (8*7 + 24*4 + 32*2 + 1) = 316
//!   brackets = 634 event records per token, against roughly 400 existing
//!   kernel launches per token.
//!
//! The overhead of either level is UNMEASURED. Establish it with an A/B (flag
//! off vs on, same weights, same prompt) before quoting any absolute number,
//! and compare level 1 against level 2 -- their disagreement is the
//! measurement of what the depth-1 brackets cost.
//! * `cupti` -- the Rust event profiler stays OFF; see
//!   `tools/cupti-inject/README.md` for the out-of-process CUPTI mode.
//!
//! # Why no extra synchronization
//!
//! Events are recorded into a pool and their elapsed times are read only in
//! [`token_settle`], which the decode path calls *after* the
//! `self.device.synchronize()` it already performs to make the argmax index
//! host-visible. The pool index resets each token, so the event set is
//! allocated once and reused for the whole run -- bounded memory, no per-token
//! `cuEventCreate` traffic after the first token.
//!
//! Contrast the pre-existing `[GDN-SUBSTAGE]` instrument
//! (`backend_impl.rs`, `prefill_gdn_layer`), which forces a
//! `synchronize()` per substage and therefore destroys the pipelining it is
//! trying to measure. This profiler adds zero syncs.

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use cudarc::driver::result::event as cu_event;
use cudarc::driver::sys as cuda_sys;
use cudarc::driver::CudaStream;

// ---------------------------------------------------------------------------
// Phase taxonomy
// ---------------------------------------------------------------------------

/// A bracketed source region on the decode path.
///
/// Depth-0 variants partition the token; depth-1 variants refine a depth-0
/// parent and are excluded from the attributed sum.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    // ---- depth 0: partition of the token -------------------------------
    /// Token embedding lookup.
    Embed = 0,
    /// Whole GDN attention block for one GDN layer.
    GdnAttn = 1,
    /// Whole softmax-attention block for one full-attention layer.
    FullAttn = 2,
    /// MoE FFN block for one MoE layer (never fires on a dense model).
    MoeFfn = 3,
    /// Dense FFN block for one layer (runs for GDN and full-attention layers).
    Ffn = 4,
    /// Per-layer `attn_proj -> x_gpu` device-to-device commit copy.
    LayerCommit = 5,
    /// Final RMSNorm + lm_head projection.
    Head = 6,
    /// GPU argmax over the logits.
    Argmax = 7,

    // ---- depth 1: inside FullAttn ---------------------------------------
    /// Input RMSNorm fused with the Q/K/V projections (inseparable: the norm
    /// kernel is chosen inside each weight-format arm, and one arm fuses it
    /// into the projection kernel itself).
    AttnQkv = 8,
    /// Per-head Q and K RMSNorm.
    AttnQkNorm = 9,
    /// Q/K/V bias add (Qwen2 family; dead on Qwen3.5).
    AttnBias = 10,
    /// RoPE application.
    AttnRope = 11,
    /// KV-cache append.
    AttnKvWrite = 12,
    /// The attention kernel itself.
    AttnCore = 13,
    /// Q+gate sigmoid gating, the gating copy, and the output projection
    /// (which folds the post-attention residual).
    AttnWo = 14,

    // ---- depth 1: inside GdnAttn ----------------------------------------
    /// GDN input projections: RMSNorm, fused qkv, alpha, beta, gate.
    GdnQkv = 15,
    /// GDN causal conv1d and the recurrent delta-rule state update. Merged
    /// because three of the four dispatch arms fuse conv and recurrence into
    /// a single kernel, so no honest split exists across all arms.
    GdnConvRecur = 16,
    /// GDN output RMSNorm, silu(z) gating, and the ssm_out projection.
    GdnOut = 17,
    /// GDN residual add + copy into `attn_proj`.
    GdnGlue = 18,

    // ---- depth 1: inside Ffn --------------------------------------------
    /// Pre-FFN RMSNorm + gate/up projections + SwiGLU. Merged because the
    /// norm is fused into each weight-format arm and SwiGLU is fused into
    /// either the gate/up or the down kernel depending on the arm.
    FfnGateUp = 19,
    /// FFN down projection and the final residual add. Merged because two
    /// arms fold the residual into the down store and return early.
    FfnDownResid = 20,

    // ---- depth 1: inside Head -------------------------------------------
    /// Final RMSNorm. `lm_head` is reported as the derived residual
    /// `Head - FinalNorm`, because the lm_head dispatch chain has six early
    /// returns and cannot be bracketed without a close before each.
    FinalNorm = 21,
}

/// Number of [`Phase`] variants.
pub const PHASE_COUNT: usize = 22;

const ALL_PHASES: [Phase; PHASE_COUNT] = [
    Phase::Embed,
    Phase::GdnAttn,
    Phase::FullAttn,
    Phase::MoeFfn,
    Phase::Ffn,
    Phase::LayerCommit,
    Phase::Head,
    Phase::Argmax,
    Phase::AttnQkv,
    Phase::AttnQkNorm,
    Phase::AttnBias,
    Phase::AttnRope,
    Phase::AttnKvWrite,
    Phase::AttnCore,
    Phase::AttnWo,
    Phase::GdnQkv,
    Phase::GdnConvRecur,
    Phase::GdnOut,
    Phase::GdnGlue,
    Phase::FfnGateUp,
    Phase::FfnDownResid,
    Phase::FinalNorm,
];

impl Phase {
    /// Stable snake_case identifier used in `[PROFILE]` output.
    pub fn name(self) -> &'static str {
        match self {
            Phase::Embed => "embed",
            Phase::GdnAttn => "gdn_attn",
            Phase::FullAttn => "full_attn",
            Phase::MoeFfn => "moe_ffn",
            Phase::Ffn => "ffn",
            Phase::LayerCommit => "layer_commit",
            Phase::Head => "head",
            Phase::Argmax => "argmax",
            Phase::AttnQkv => "attn_qkv",
            Phase::AttnQkNorm => "attn_qk_norm",
            Phase::AttnBias => "attn_bias",
            Phase::AttnRope => "attn_rope",
            Phase::AttnKvWrite => "attn_kv_write",
            Phase::AttnCore => "attn_core",
            Phase::AttnWo => "attn_wo",
            Phase::GdnQkv => "gdn_qkv",
            Phase::GdnConvRecur => "gdn_conv_recur",
            Phase::GdnOut => "gdn_out",
            Phase::GdnGlue => "gdn_glue",
            Phase::FfnGateUp => "ffn_gate_up",
            Phase::FfnDownResid => "ffn_down_resid",
            Phase::FinalNorm => "final_norm",
        }
    }

    /// 0 for the token-partitioning phases, 1 for nested refinements.
    pub fn depth(self) -> u8 {
        match self {
            Phase::Embed
            | Phase::GdnAttn
            | Phase::FullAttn
            | Phase::MoeFfn
            | Phase::Ffn
            | Phase::LayerCommit
            | Phase::Head
            | Phase::Argmax => 0,
            _ => 1,
        }
    }

    /// The depth-0 phase this phase refines, if any.
    pub fn parent(self) -> Option<Phase> {
        match self {
            Phase::AttnQkv
            | Phase::AttnQkNorm
            | Phase::AttnBias
            | Phase::AttnRope
            | Phase::AttnKvWrite
            | Phase::AttnCore
            | Phase::AttnWo => Some(Phase::FullAttn),
            Phase::GdnQkv | Phase::GdnConvRecur | Phase::GdnOut | Phase::GdnGlue => {
                Some(Phase::GdnAttn)
            }
            Phase::FfnGateUp | Phase::FfnDownResid => Some(Phase::Ffn),
            Phase::FinalNorm => Some(Phase::Head),
            _ => None,
        }
    }

    fn idx(self) -> usize {
        self as u8 as usize
    }
}

// ---------------------------------------------------------------------------
// Level gate
// ---------------------------------------------------------------------------

/// Profiler verbosity: 0 = off, 1 = depth-0 phases, 2 = depth-0 + depth-1.
///
/// Parsed from `LUMEN_CUDA_PROFILE` exactly once and cached, so the hot path
/// never calls `getenv`.
#[inline]
pub fn level() -> u8 {
    static CACHED: OnceLock<u8> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = match std::env::var("LUMEN_CUDA_PROFILE") {
            Ok(v) => v,
            Err(_) => return 0,
        };
        let lvl = match raw.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" | "no" => 0,
            "1" | "on" | "true" | "yes" | "coarse" => 1,
            "2" | "fine" => 2,
            "cupti" => {
                eprintln!(
                    "[PROFILE] note LUMEN_CUDA_PROFILE=cupti -- in-process event profiler is OFF; \
                     use the out-of-process CUPTI injection library (tools/cupti-inject/README.md)"
                );
                0
            }
            other => {
                eprintln!(
                    "[PROFILE] warn unrecognized LUMEN_CUDA_PROFILE={other:?} -- profiling disabled \
                     (expected 0|1|2|cupti)"
                );
                0
            }
        };
        if lvl > 0 {
            emit_environment_warnings();
        }
        lvl
    })
}

/// True when any profiling level is active.
#[inline]
pub fn enabled() -> bool {
    level() > 0
}

/// Warn about env vars that inject blocking device-to-host copies into the
/// decode path. Those readbacks serialize the stream mid-phase and make every
/// number in the table meaningless, so a profiling run must not set them.
fn emit_environment_warnings() {
    for var in ["LUMEN_XCHK", "LUMEN_MOE_PROBE", "LUMEN_CUDA_GDN_SUBSTAGE_TIMING"] {
        if std::env::var(var).is_ok() {
            eprintln!(
                "[PROFILE] warn {var} is set -- it injects blocking readbacks or forced syncs into \
                 the decode path; phase spans from this run are NOT trustworthy"
            );
        }
    }
    if let Ok(v) = std::env::var("LUMEN_CUDA_DECODE_DELAY_US") {
        if v.trim() != "0" && !v.trim().is_empty() {
            eprintln!(
                "[PROFILE] warn LUMEN_CUDA_DECODE_DELAY_US={v} -- a host sleep runs inside the \
                 per-token wall clock; host_outside_span_us is inflated by it"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Event pool
// ---------------------------------------------------------------------------

/// `CUevent` is a raw pointer, so it is not `Send`/`Sync` by default. CUDA
/// event handles are context-scoped, not thread-scoped, and every access here
/// is serialized by the profiler's own `Mutex`.
struct EventHandle(cuda_sys::CUevent);
unsafe impl Send for EventHandle {}
unsafe impl Sync for EventHandle {}

/// One completed begin/end pair awaiting an `elapsed` read.
struct Bracket {
    phase: u8,
    start: u32,
    end: u32,
}

#[derive(Default)]
struct Health {
    /// Brackets still open when the token ended (an early return skipped a
    /// close). Their time lands in `gpu_unattributed`.
    unclosed: u64,
    /// `end` called for a phase that was not the innermost open bracket.
    nest_errors: u64,
    /// `cuEventCreate` / `cuEventRecord` / `cuEventElapsedTime` failures.
    event_errors: u64,
    /// Largest number of events a single token needed.
    pool_high_water: usize,
    /// `begin` calls that arrived while no decode token was open, and were
    /// therefore not recorded. `compute_layer_gpu` is reachable from the
    /// `ComputeBackend::compute_layer` trait method (the streaming / per-layer
    /// path) as well as from the decode loop; that work is real but is not part
    /// of a decode token, so it is dropped rather than folded into the next
    /// token's totals. A non-zero count here means such calls happened.
    outside_token: u64,
}

#[derive(Default)]
struct Accum {
    /// Total GPU-timeline span per phase, microseconds.
    span_us: [f64; PHASE_COUNT],
    /// Bracket count per phase.
    calls: [u64; PHASE_COUNT],
    /// Per-token samples, one entry per settled token.
    wall_us: Vec<f64>,
    gpu_span_us: Vec<f64>,
    attributed_us: Vec<f64>,
}

impl Accum {
    fn tokens(&self) -> usize {
        self.wall_us.len()
    }
}

struct Profiler {
    /// Reused across tokens; `cursor` resets to 0 each token.
    pool: Vec<EventHandle>,
    cursor: usize,
    /// Open brackets, innermost last: (phase, start event index).
    open: Vec<(u8, u32)>,
    closed: Vec<Bracket>,
    token_start: Option<u32>,
    token_end: Option<u32>,
    wall_start: Option<Instant>,
    /// True between `token_begin` and `token_settle`. Phase brackets are only
    /// recorded while a token is open, so the bracketed functions being
    /// reachable from non-decode call paths cannot contaminate a token.
    in_token: bool,
    /// Set once `token_end` has recorded its event; cleared by `token_settle`.
    pending: bool,
    accum: Accum,
    health: Health,
}

impl Profiler {
    fn new() -> Self {
        Self {
            pool: Vec::new(),
            cursor: 0,
            open: Vec::new(),
            closed: Vec::new(),
            token_start: None,
            token_end: None,
            wall_start: None,
            in_token: false,
            pending: false,
            accum: Accum::default(),
            health: Health::default(),
        }
    }

    /// Take the next pool slot, growing the pool on first use of that slot.
    fn take_event(&mut self) -> Option<u32> {
        if self.cursor == self.pool.len() {
            match cu_event::create(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT) {
                Ok(e) => self.pool.push(EventHandle(e)),
                Err(_) => {
                    self.health.event_errors += 1;
                    return None;
                }
            }
        }
        let idx = self.cursor as u32;
        self.cursor += 1;
        Some(idx)
    }

    /// Take a slot and record it on `stream`.
    fn record(&mut self, stream: &CudaStream) -> Option<u32> {
        let idx = self.take_event()?;
        let ev = self.pool[idx as usize].0;
        // SAFETY: `ev` came from `cu_event::create` and is never destroyed
        // before process exit; `stream` is the live decode stream.
        if unsafe { cu_event::record(ev, stream.cu_stream()) }.is_err() {
            self.health.event_errors += 1;
            // Roll the cursor back so the slot is reused rather than leaked.
            self.cursor -= 1;
            return None;
        }
        Some(idx)
    }

    fn elapsed_us(&mut self, start: u32, end: u32) -> Option<f64> {
        let a = self.pool[start as usize].0;
        let b = self.pool[end as usize].0;
        // SAFETY: both events were created by `cu_event::create`, recorded on
        // the same stream, and the caller has synchronized that stream.
        match unsafe { cu_event::elapsed(a, b) } {
            Ok(ms) => Some(ms as f64 * 1000.0),
            Err(_) => {
                self.health.event_errors += 1;
                None
            }
        }
    }

    /// Fold this token's brackets into the accumulator and reset for the next.
    ///
    /// The caller must have synchronized the decode stream first.
    fn settle(&mut self) {
        // Any bracket still open lost its close to an early return. Count it
        // and drop its span: the time is not lost, it reappears in
        // `gpu_unattributed` because the token bracket still covers it.
        self.health.unclosed += self.open.len() as u64;
        self.open.clear();

        let mut attributed = 0.0f64;
        let closed = std::mem::take(&mut self.closed);
        for b in &closed {
            if let Some(us) = self.elapsed_us(b.start, b.end) {
                let i = b.phase as usize;
                self.accum.span_us[i] += us;
                self.accum.calls[i] += 1;
                if ALL_PHASES[i].depth() == 0 {
                    attributed += us;
                }
            }
        }
        self.closed = closed;
        self.closed.clear();

        let gpu_span = match (self.token_start, self.token_end) {
            (Some(a), Some(b)) => self.elapsed_us(a, b).unwrap_or(0.0),
            _ => 0.0,
        };
        let wall = self
            .wall_start
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or(0.0);

        self.accum.wall_us.push(wall);
        self.accum.gpu_span_us.push(gpu_span);
        self.accum.attributed_us.push(attributed);

        self.health.pool_high_water = self.health.pool_high_water.max(self.cursor);
        self.cursor = 0;
        self.token_start = None;
        self.token_end = None;
        self.wall_start = None;
        self.in_token = false;
        self.pending = false;
    }
}

fn state() -> &'static Mutex<Profiler> {
    static STATE: OnceLock<Mutex<Profiler>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(Profiler::new()))
}

/// Run `f` against the profiler, ignoring a poisoned lock (a panic elsewhere
/// must not turn a diagnostic into a second panic).
fn with<R>(f: impl FnOnce(&mut Profiler) -> R) -> Option<R> {
    match state().lock() {
        Ok(mut g) => Some(f(&mut g)),
        Err(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Instrumentation entry points
// ---------------------------------------------------------------------------

/// Open a phase bracket. No-op unless the phase's depth is within the level.
#[inline]
pub fn begin(phase: Phase, stream: &CudaStream) {
    let lvl = level();
    if lvl == 0 || phase.depth() >= lvl {
        return;
    }
    with(|p| {
        if !p.in_token {
            p.health.outside_token += 1;
            return;
        }
        if let Some(idx) = p.record(stream) {
            p.open.push((phase as u8, idx));
        }
    });
}

/// Close the innermost phase bracket, which must be `phase`.
#[inline]
pub fn end(phase: Phase, stream: &CudaStream) {
    let lvl = level();
    if lvl == 0 || phase.depth() >= lvl {
        return;
    }
    with(|p| {
        if !p.in_token {
            // Symmetric with `begin`: no bracket was opened, so closing one
            // here is not a nesting error.
            return;
        }
        match p.open.last().copied() {
            Some((open_phase, start)) if open_phase == phase as u8 => {
                p.open.pop();
                if let Some(end_idx) = p.record(stream) {
                    p.closed.push(Bracket {
                        phase: phase as u8,
                        start,
                        end: end_idx,
                    });
                }
            }
            _ => {
                // Mis-nesting: a close arrived out of order. Record it rather
                // than guessing, so the health line exposes the bug.
                p.health.nest_errors += 1;
            }
        }
    });
}

/// Start a decode token: reset the per-token cursor, take the wall clock, and
/// record the opening token event.
///
/// If the previous token never settled (a `?` propagated out of the decode
/// path), this synchronizes that token's closing event and settles it first so
/// the accumulator never mixes two tokens.
#[inline]
pub fn token_begin(stream: &CudaStream) {
    if level() == 0 {
        return;
    }
    with(|p| {
        if p.pending {
            if let Some(end_idx) = p.token_end {
                let ev = p.pool[end_idx as usize].0;
                // SAFETY: `ev` is a live event recorded on the decode stream.
                let _ = unsafe { cu_event::synchronize(ev) };
            }
            p.settle();
        } else if p.cursor != 0 || !p.open.is_empty() {
            // A token was abandoned before `token_end`. Discard its partial
            // brackets; there is no completed closing event to time against.
            p.health.unclosed += p.open.len() as u64;
            p.open.clear();
            p.closed.clear();
            p.cursor = 0;
        }
        p.in_token = true;
        p.wall_start = Some(Instant::now());
        p.token_start = p.record(stream);
    });
}

/// Record the closing token event. Call this after the last decode launch and
/// BEFORE the stream synchronize, so the event times the GPU work and not the
/// host's sync call.
#[inline]
pub fn token_end(stream: &CudaStream) {
    if level() == 0 {
        return;
    }
    with(|p| {
        p.token_end = p.record(stream);
        p.pending = true;
    });
}

/// Fold the token into the accumulator. Call this AFTER the decode path's
/// existing stream synchronize; it adds no synchronization of its own.
#[inline]
pub fn token_settle() {
    if level() == 0 {
        return;
    }
    with(|p| {
        if p.pending {
            p.settle();
        }
    });
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// One phase row of a [`Summary`].
#[derive(Clone, Debug)]
pub struct PhaseRow {
    pub phase: Phase,
    pub calls: u64,
    pub total_us: f64,
}

/// Aggregated profile over the tokens seen since the last reset.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub tokens: usize,
    pub rows: Vec<PhaseRow>,
    pub wall_us_mean: f64,
    pub wall_us_p50: f64,
    pub gpu_span_us_mean: f64,
    pub gpu_span_us_p50: f64,
    pub attributed_us_mean: f64,
    pub attributed_us_p50: f64,
    pub unclosed: u64,
    pub nest_errors: u64,
    pub event_errors: u64,
    pub pool_high_water: usize,
    pub outside_token: u64,
}

impl Summary {
    /// GPU-timeline time inside the token that no depth-0 phase claimed:
    /// inter-phase submission gaps plus any unbracketed launch.
    pub fn gpu_unattributed_us(&self) -> f64 {
        (self.gpu_span_us_mean - self.attributed_us_mean).max(0.0)
    }

    /// Host time outside the token's GPU span: submission latency, the stream
    /// synchronize, and the argmax readback.
    pub fn host_outside_span_us(&self) -> f64 {
        (self.wall_us_mean - self.gpu_span_us_mean).max(0.0)
    }

    fn per_token(&self, total: f64) -> f64 {
        if self.tokens == 0 {
            0.0
        } else {
            total / self.tokens as f64
        }
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn p50(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        0.5 * (v[mid - 1] + v[mid])
    } else {
        v[mid]
    }
}

/// Snapshot the accumulator without clearing it.
pub fn summary() -> Summary {
    with(|p| {
        let mut rows = Vec::with_capacity(PHASE_COUNT);
        for ph in ALL_PHASES {
            let i = ph.idx();
            if p.accum.calls[i] == 0 {
                continue;
            }
            rows.push(PhaseRow {
                phase: ph,
                calls: p.accum.calls[i],
                total_us: p.accum.span_us[i],
            });
        }
        Summary {
            tokens: p.accum.tokens(),
            rows,
            wall_us_mean: mean(&p.accum.wall_us),
            wall_us_p50: p50(&p.accum.wall_us),
            gpu_span_us_mean: mean(&p.accum.gpu_span_us),
            gpu_span_us_p50: p50(&p.accum.gpu_span_us),
            attributed_us_mean: mean(&p.accum.attributed_us),
            attributed_us_p50: p50(&p.accum.attributed_us),
            unclosed: p.health.unclosed,
            nest_errors: p.health.nest_errors,
            event_errors: p.health.event_errors,
            pool_high_water: p.health.pool_high_water,
            outside_token: p.health.outside_token,
        }
    })
    .unwrap_or_default()
}

/// Clear per-phase and per-token accumulation. The event pool and the health
/// counters survive, so pool reuse continues and defects stay visible.
pub fn reset() {
    let _ = with(|p| {
        p.accum = Accum::default();
    });
}

/// Emit the `[PROFILE]` block to stderr and clear the accumulator.
///
/// `label` identifies the run segment (for example a decode-sequence index) so
/// warmup segments can be told apart from measured ones. No-op when disabled or
/// when no token has been recorded.
pub fn print_and_reset(label: &str) {
    if level() == 0 {
        return;
    }
    let s = summary();
    if s.tokens == 0 {
        return;
    }
    write_summary(&mut std::io::stderr(), label, &s);
    reset();
}

/// Write the `[PROFILE]` block. Every line is `key=value` space separated
/// after a fixed `[PROFILE] <kind>` prefix, so the output greps and parses
/// without a schema.
pub fn write_summary<W: std::io::Write>(w: &mut W, label: &str, s: &Summary) {
    let wall = s.wall_us_mean.max(1e-9);
    let attributed = s.attributed_us_mean.max(1e-9);

    let _ = writeln!(
        w,
        "[PROFILE] meta version=1 level={} label={} tokens={}",
        level(),
        label,
        s.tokens
    );
    let _ = writeln!(
        w,
        "[PROFILE] token wall_us_mean={:.3} wall_us_p50={:.3} gpu_span_us_mean={:.3} \
         gpu_span_us_p50={:.3} attributed_us_mean={:.3} attributed_us_p50={:.3}",
        s.wall_us_mean,
        s.wall_us_p50,
        s.gpu_span_us_mean,
        s.gpu_span_us_p50,
        s.attributed_us_mean,
        s.attributed_us_p50,
    );
    let unattributed = s.gpu_unattributed_us();
    let host_gap = s.host_outside_span_us();
    let _ = writeln!(
        w,
        "[PROFILE] residual gpu_unattributed_us={:.3} gpu_unattributed_pct_of_wall={:.2} \
         host_outside_span_us={:.3} host_outside_span_pct_of_wall={:.2}",
        unattributed,
        100.0 * unattributed / wall,
        host_gap,
        100.0 * host_gap / wall,
    );

    for row in &s.rows {
        let per_tok = s.per_token(row.total_us);
        let calls_per_tok = if s.tokens == 0 {
            0.0
        } else {
            row.calls as f64 / s.tokens as f64
        };
        let parent = row.phase.parent().map(|p| p.name()).unwrap_or("-");
        let _ = writeln!(
            w,
            "[PROFILE] phase depth={} name={} parent={} calls_per_token={:.2} us_per_token={:.3} \
             pct_of_wall={:.2} pct_of_attributed={:.2}",
            row.phase.depth(),
            row.phase.name(),
            parent,
            calls_per_tok,
            per_tok,
            100.0 * per_tok / wall,
            100.0 * per_tok / attributed,
        );
    }

    // lm_head is not directly bracketable (six early returns in the dispatch
    // chain), so report it as Head minus FinalNorm when both are present.
    let head = s
        .rows
        .iter()
        .find(|r| r.phase == Phase::Head)
        .map(|r| s.per_token(r.total_us));
    let final_norm = s
        .rows
        .iter()
        .find(|r| r.phase == Phase::FinalNorm)
        .map(|r| s.per_token(r.total_us));
    if let (Some(h), Some(fnorm)) = (head, final_norm) {
        let lm = (h - fnorm).max(0.0);
        let _ = writeln!(
            w,
            "[PROFILE] derived name=lm_head from=head-final_norm us_per_token={:.3} \
             pct_of_wall={:.2}",
            lm,
            100.0 * lm / wall,
        );
    }

    let _ = writeln!(
        w,
        "[PROFILE] health unclosed_brackets={} nest_errors={} event_errors={} \
         outside_token_brackets={} pool_high_water={} brackets_per_token={:.1}",
        s.unclosed,
        s.nest_errors,
        s.event_errors,
        s.outside_token,
        s.pool_high_water,
        s.rows.iter().map(|r| r.calls).sum::<u64>() as f64 / s.tokens.max(1) as f64,
    );
    let _ = writeln!(w, "[PROFILE] end label={label}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_zero_phases_partition_the_token() {
        let d0: Vec<&str> = ALL_PHASES
            .iter()
            .filter(|p| p.depth() == 0)
            .map(|p| p.name())
            .collect();
        assert_eq!(
            d0,
            vec![
                "embed",
                "gdn_attn",
                "full_attn",
                "moe_ffn",
                "ffn",
                "layer_commit",
                "head",
                "argmax",
            ]
        );
    }

    #[test]
    fn depth_one_phases_all_declare_a_depth_zero_parent() {
        for p in ALL_PHASES {
            match p.depth() {
                0 => assert_eq!(p.parent(), None, "{} is depth 0", p.name()),
                1 => {
                    let parent = p.parent().unwrap_or_else(|| panic!("{} has no parent", p.name()));
                    assert_eq!(parent.depth(), 0, "{} parent must be depth 0", p.name());
                }
                d => panic!("{} has unsupported depth {d}", p.name()),
            }
        }
    }

    #[test]
    fn phase_discriminants_match_all_phases_order() {
        for (i, p) in ALL_PHASES.iter().enumerate() {
            assert_eq!(p.idx(), i, "{} discriminant must equal its index", p.name());
        }
        assert_eq!(ALL_PHASES.len(), PHASE_COUNT);
    }

    #[test]
    fn phase_names_are_unique() {
        let mut names: Vec<&str> = ALL_PHASES.iter().map(|p| p.name()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate phase name");
    }

    #[test]
    fn stats_helpers() {
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(p50(&[]), 0.0);
        assert!((mean(&[10.0, 20.0, 30.0]) - 20.0).abs() < 1e-9);
        assert!((p50(&[10.0, 20.0, 30.0]) - 20.0).abs() < 1e-9);
        assert!((p50(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-9);
        // p50 must not depend on input order.
        assert!((p50(&[4.0, 1.0, 3.0, 2.0]) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn residuals_are_clamped_and_never_negative() {
        let s = Summary {
            tokens: 4,
            wall_us_mean: 100.0,
            gpu_span_us_mean: 120.0,
            attributed_us_mean: 130.0,
            ..Default::default()
        };
        assert_eq!(s.gpu_unattributed_us(), 0.0);
        assert_eq!(s.host_outside_span_us(), 0.0);
    }

    #[test]
    fn residuals_decompose_the_token() {
        let s = Summary {
            tokens: 10,
            wall_us_mean: 1000.0,
            gpu_span_us_mean: 900.0,
            attributed_us_mean: 700.0,
            ..Default::default()
        };
        assert!((s.gpu_unattributed_us() - 200.0).abs() < 1e-9);
        assert!((s.host_outside_span_us() - 100.0).abs() < 1e-9);
        // attributed + unattributed + host gap == wall, exactly.
        let recomposed =
            s.attributed_us_mean + s.gpu_unattributed_us() + s.host_outside_span_us();
        assert!((recomposed - s.wall_us_mean).abs() < 1e-9);
    }

    #[test]
    fn summary_writes_parseable_lines_and_derives_lm_head() {
        let s = Summary {
            tokens: 2,
            rows: vec![
                PhaseRow {
                    phase: Phase::Head,
                    calls: 2,
                    total_us: 200.0,
                },
                PhaseRow {
                    phase: Phase::FinalNorm,
                    calls: 2,
                    total_us: 40.0,
                },
            ],
            wall_us_mean: 500.0,
            gpu_span_us_mean: 400.0,
            attributed_us_mean: 300.0,
            ..Default::default()
        };
        let mut buf: Vec<u8> = Vec::new();
        write_summary(&mut buf, "unit", &s);
        let out = String::from_utf8(buf).expect("utf8");
        for line in out.lines() {
            assert!(line.starts_with("[PROFILE] "), "unprefixed line: {line}");
        }
        assert!(out.contains("name=head "), "{out}");
        // head = 200/2 = 100 us/token, final_norm = 40/2 = 20 -> lm_head = 80.
        assert!(out.contains("name=lm_head from=head-final_norm us_per_token=80.000"), "{out}");
        assert!(out.contains("label=unit"), "{out}");
    }

    #[test]
    fn summary_omits_lm_head_when_final_norm_missing() {
        let s = Summary {
            tokens: 1,
            rows: vec![PhaseRow {
                phase: Phase::Head,
                calls: 1,
                total_us: 10.0,
            }],
            wall_us_mean: 20.0,
            ..Default::default()
        };
        let mut buf: Vec<u8> = Vec::new();
        write_summary(&mut buf, "unit", &s);
        let out = String::from_utf8(buf).expect("utf8");
        assert!(!out.contains("lm_head"), "{out}");
    }

    /// The output contract downstream analysis depends on: after the
    /// `[PROFILE] <kind>` prefix, every whitespace-separated token is exactly
    /// one `key=value` pair. A bare token would silently break any
    /// `awk`/`split('=')` parser.
    #[test]
    fn every_output_token_is_a_key_value_pair() {
        let s = Summary {
            tokens: 3,
            rows: ALL_PHASES
                .iter()
                .enumerate()
                .map(|(i, p)| PhaseRow {
                    phase: *p,
                    calls: 3 * (i as u64 + 1),
                    total_us: 10.0 * (i as f64 + 1.0),
                })
                .collect(),
            wall_us_mean: 8000.0,
            wall_us_p50: 7990.0,
            gpu_span_us_mean: 7500.0,
            gpu_span_us_p50: 7490.0,
            attributed_us_mean: 6800.0,
            attributed_us_p50: 6790.0,
            unclosed: 0,
            nest_errors: 0,
            event_errors: 0,
            pool_high_water: 634,
            outside_token: 0,
        };
        let mut buf: Vec<u8> = Vec::new();
        write_summary(&mut buf, "contract", &s);
        let out = String::from_utf8(buf).expect("utf8");

        let mut kinds = Vec::new();
        for line in out.lines() {
            let rest = line
                .strip_prefix("[PROFILE] ")
                .unwrap_or_else(|| panic!("missing prefix: {line}"));
            let mut toks = rest.split_whitespace();
            let kind = toks.next().unwrap_or_else(|| panic!("no kind: {line}"));
            assert!(
                !kind.contains('='),
                "first token must be a bare kind, got {kind:?} in {line}"
            );
            kinds.push(kind.to_string());
            for t in toks {
                let mut it = t.splitn(2, '=');
                let k = it.next().unwrap_or("");
                let v = it.next();
                assert!(!k.is_empty(), "empty key in {line}");
                let v = v.unwrap_or_else(|| panic!("token {t:?} is not key=value in {line}"));
                assert!(!v.is_empty(), "empty value for {k} in {line}");
                assert!(!v.contains('='), "value contains '=': {t:?} in {line}");
            }
        }

        // Every phase with calls > 0 must produce exactly one phase row.
        let phase_rows = kinds.iter().filter(|k| *k == "phase").count();
        assert_eq!(phase_rows, PHASE_COUNT, "one row per phase expected");
        assert_eq!(kinds.first().map(String::as_str), Some("meta"));
        assert_eq!(kinds.last().map(String::as_str), Some("end"));
        assert_eq!(kinds.iter().filter(|k| *k == "token").count(), 1);
        assert_eq!(kinds.iter().filter(|k| *k == "residual").count(), 1);
        assert_eq!(kinds.iter().filter(|k| *k == "health").count(), 1);
        // No NaN/inf can reach a parser.
        assert!(!out.contains("NaN") && !out.contains("inf"), "{out}");
    }

    #[test]
    fn empty_summary_still_writes_a_wellformed_block() {
        let s = Summary::default();
        let mut buf: Vec<u8> = Vec::new();
        write_summary(&mut buf, "empty", &s);
        let out = String::from_utf8(buf).expect("utf8");
        assert!(out.contains("[PROFILE] meta "));
        assert!(out.contains("[PROFILE] end label=empty"));
        // No division-by-zero artefacts.
        assert!(!out.contains("NaN"), "{out}");
        assert!(!out.contains("inf"), "{out}");
    }
}
