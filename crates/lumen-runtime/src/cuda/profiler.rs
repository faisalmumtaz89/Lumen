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
//! * `cupti` -- the Rust event profiler stays OFF; see
//!   `tools/cupti-inject/README.md` for the out-of-process CUPTI mode.
//!
//! The overhead of either level is UNMEASURED. Establish it with an A/B (flag
//! off vs on, same weights, same prompt) before quoting any absolute number,
//! and compare level 1 against level 2 -- their disagreement is the
//! measurement of what the depth-1 brackets cost.
//!
//! # Segment boundaries
//!
//! One `[PROFILE]` block is emitted per `Engine::generate` call, via the
//! [`SegmentGuard`] returned by [`begin_segment`]. It is a drop guard rather
//! than a post-loop statement for a specific reason: the decode loop propagates
//! errors with `?`, and every token that already completed has been folded into
//! the accumulator by its own [`token_settle`]. A statement after the loop is
//! skipped on error, and those orphaned tokens would then be reported inside the
//! NEXT segment's block under a label implying they belonged to it. Dropping
//! happens on every exit path, so a failed segment reports the tokens it did
//! complete, under its own label, and leaves nothing behind.
//!
//! # Concurrency
//!
//! The profiler is a single process-wide singleton. Today that is safe because
//! each `CudaBackend` holds its own `state` mutex across an entire decode call,
//! serializing all bracket calls for that instance. It is NOT safe by
//! construction: two `CudaBackend` instances decoding concurrently (multi-model
//! or multi-GPU in one process) would interleave brackets into the same stack.
//! That corrupts both instances' numbers rather than crashing, but it is
//! detectable -- it shows up as `nest_errors` on the health line, which is
//! exactly why that counter is reported.
//!
//! # Verified coverage and known blind spots
//!
//! An independent launch-site census of the decode path (186 sites) found every
//! kernel launch, every `launch_*` helper, every cuBLAS call, and every
//! `memcpy_dtod` inside exactly one depth-0 phase. The depth-1 children tile
//! their parent's launch set exactly for `full_attn` (7 children), `gdn_attn`
//! (4) and `ffn` (2) -- the parent-minus-children difference there is
//! inter-phase submission gap, not hidden work.
//!
//! The blind spots that remain, stated so they are not rediscovered as
//! surprises:
//!
//! * **`head` has one depth-1 child.** The lm_head dispatch chain is 26 of the
//!   27 launches in `compute_final_gpu` and has ten early returns, so it is
//!   reported as the derived `head - final_norm` rather than bracketed. That
//!   derived value is exactly the lm_head cost, but level 2 buys no resolution
//!   *within* lm_head (activation quantize vs. matvec are not separable).
//! * **Three D2H readbacks sit outside the token bracket** because they run
//!   after `token_end`: the full-vocab logits copy on the sampling path
//!   (~1 MB/token), the 4-byte argmax copy on the greedy path, and an
//!   `LUMEN_XCHK` probe. They are real time and appear in
//!   `host_outside_span_us`, never in the phase table.
//! * **`moe.rs` carries no brackets and contains its own `synchronize()`
//!   calls.** On a MoE model the `moe_ffn` phase therefore spans forced host
//!   round-trips and is not comparable to the other phases. Dense models never
//!   enter it.
//! * **Error paths leak their brackets.** Roughly 250 `?` sites and five
//!   `return Err` sites lie inside brackets. Every one aborts the token, so
//!   `token_settle` is never reached and `token_begin` discards the partial
//!   token, counting it in both `unclosed` and `abandoned_tokens`. Nothing is
//!   read off an unsynchronized event, so a leak can never contaminate a later
//!   token -- but the whole token is dropped rather than partially salvaged,
//!   including phases that had already closed cleanly.
//! * **`gpu_unattributed_us` conflates two causes** with opposite remedies:
//!   inter-phase submission gap ("fuse or overlap") and a launch covered by no
//!   bracket ("go add a bracket"). It cannot tell you which. The per-parent
//!   `uncovered` lines narrow it -- a residual that GROWS inside one depth-0
//!   parent after a code change points at a new unbracketed launch in that
//!   parent -- but treat a large global residual as "investigate", never as
//!   evidence for a specific lever.
//!
//! Read the `health` lines before anything else. `scope=lifetime` is
//! process-cumulative and `scope=segment` is this block only; a defect that
//! appeared in an earlier segment stays visible in the lifetime line forever,
//! which is why both are printed.
//!
//! # Reading the numbers
//!
//! Prefer `p50` over `mean` for the per-token quantities. The first decode
//! token of a run pays one-time costs that land inside a bracket and are not
//! representative:
//!
//! * `ensure_gdn_scratch` allocates every GDN state and conv-state buffer on
//!   the first GDN layer of the first token, inside the `gdn_attn` bracket.
//! * kernel PTX may still be resolving on first dispatch.
//!
//! Because each `generate` call emits its own `[PROFILE]` block with its own
//! label, a benchmark's warmup sequences absorb this and the measured
//! sequences are clean. A single-generation run does not have that luxury, so
//! read `p50`.
//!
//! Per-phase figures are `us_per_token` totals, which have no per-token
//! distribution attached -- a phase's total is summed across all its calls in
//! all tokens and then divided by the token count. A phase whose cost is
//! bimodal will not reveal that here.
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

/// Defect counters. Every one of these should read 0 on a trustworthy run.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct HealthCounts {
    /// Brackets still open when the token ended (an early return skipped a
    /// close). The token is discarded, so this never corrupts another token --
    /// but the token's data is lost, not partially salvaged.
    pub unclosed: u64,
    /// `end` called for a phase that was not the innermost open bracket. Also
    /// the signal that two `CudaBackend` instances are interleaving into this
    /// process-wide profiler (see the module docs).
    pub nest_errors: u64,
    /// `cuEventCreate` / `cuEventRecord` / `cuEventElapsedTime` failures.
    pub event_errors: u64,
    /// `begin` calls that arrived while no decode token was open, and were
    /// therefore not recorded. `compute_layer_gpu` is reachable from the
    /// `ComputeBackend::compute_layer` trait method (the streaming / per-layer
    /// path) as well as from the decode loop; that work is real but is not part
    /// of a decode token, so it is dropped rather than folded into the next
    /// token's totals. A non-zero count here means such calls happened.
    pub outside_token: u64,
    /// Tokens whose depth-0 phase sum EXCEEDED the whole-token GPU span by more
    /// than `RESIDUAL_EPSILON_US`.
    ///
    /// Phases nest inside the token bracket and do not overlap, so this is
    /// mathematically impossible beyond event-timer quantization. A non-zero
    /// count means the ordering assumption is broken somewhere and the residual
    /// arithmetic cannot be trusted. Without this counter the `.max(0.0)` clamp
    /// on the reported residuals would silently absorb exactly that bug.
    pub negative_residual: u64,
    /// Tokens abandoned before `token_end` (an error propagated mid-token).
    pub abandoned_tokens: u64,
    /// Largest number of events a single token needed.
    pub pool_high_water: usize,
}

impl HealthCounts {
    /// Field-wise difference, for reporting "since the last print" alongside
    /// the process-lifetime totals.
    fn since(&self, base: &HealthCounts) -> HealthCounts {
        HealthCounts {
            unclosed: self.unclosed.saturating_sub(base.unclosed),
            nest_errors: self.nest_errors.saturating_sub(base.nest_errors),
            event_errors: self.event_errors.saturating_sub(base.event_errors),
            outside_token: self.outside_token.saturating_sub(base.outside_token),
            negative_residual: self.negative_residual.saturating_sub(base.negative_residual),
            abandoned_tokens: self.abandoned_tokens.saturating_sub(base.abandoned_tokens),
            pool_high_water: self.pool_high_water,
        }
    }

    /// True when every defect counter is zero.
    pub fn is_clean(&self) -> bool {
        self.unclosed == 0
            && self.nest_errors == 0
            && self.event_errors == 0
            && self.outside_token == 0
            && self.negative_residual == 0
            && self.abandoned_tokens == 0
    }
}

/// Tolerance for the "phases cannot exceed the token span" check. CUDA event
/// timestamps quantize at roughly half a microsecond, and a token accumulates
/// ~100-300 brackets, so a few microseconds of accumulated rounding is benign.
const RESIDUAL_EPSILON_US: f64 = 4.0;

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
    /// Process-lifetime defect counters. Deliberately NOT cleared by `reset`,
    /// so a defect cannot be erased by a segment boundary.
    health: HealthCounts,
    /// Snapshot of `health` at the last print, so each `[PROFILE]` block can
    /// report both the lifetime totals and the delta for its own segment.
    /// Without the delta, a defect from segment 0 reads identically in every
    /// later segment's health line forever.
    health_baseline: HealthCounts,
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
            health: HealthCounts::default(),
            health_baseline: HealthCounts::default(),
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

        // Phases nest inside the token bracket and never overlap, so the phase
        // sum cannot exceed the token span by more than timer quantization.
        // Detect the violation PER TOKEN, where it is unambiguous -- checking
        // only the reported means would let a systematic ordering bug hide
        // behind the `.max(0.0)` clamp in `Summary`.
        if gpu_span > 0.0 && attributed > gpu_span + RESIDUAL_EPSILON_US {
            self.health.negative_residual += 1;
        }

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
            // A token was abandoned before `token_end` (an error propagated
            // mid-token). Discard its partial brackets: there is no completed
            // closing event to time against, and reading elapsed times off
            // events that may not have completed would produce garbage. The
            // whole token is dropped rather than partially salvaged, which is
            // why this is counted separately from `unclosed`.
            p.health.unclosed += p.open.len() as u64;
            p.health.abandoned_tokens += 1;
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
    /// Process-lifetime defect counters.
    pub health: HealthCounts,
    /// Defect counters accrued since the previous print.
    pub health_delta: HealthCounts,
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

    /// Per-token microseconds for one phase.
    fn phase_us_per_token(&self, phase: Phase) -> Option<f64> {
        self.rows
            .iter()
            .find(|r| r.phase == phase)
            .map(|r| self.per_token(r.total_us))
    }

    /// Time inside a depth-0 phase that none of its depth-1 children claimed.
    ///
    /// Returns `None` when the phase has no children recorded (level 1, or a
    /// phase with no children at all). This localizes what
    /// `gpu_unattributed_us` can only report globally: a growing per-parent
    /// residual points at a launch inside that parent covered by no child.
    pub fn uncovered_in(&self, parent: Phase) -> Option<f64> {
        let total = self.phase_us_per_token(parent)?;
        let mut children = 0.0;
        let mut any = false;
        for row in &self.rows {
            if row.phase.parent() == Some(parent) {
                any = true;
                children += self.per_token(row.total_us);
            }
        }
        if !any {
            return None;
        }
        Some((total - children).max(0.0))
    }

    fn per_token(&self, total: f64) -> f64 {
        if self.tokens == 0 {
            0.0
        } else {
            total / self.tokens as f64
        }
    }
}

/// Format `num/den` as a percentage, or `n/a` when the denominator is not
/// meaningfully positive.
///
/// Flooring the denominator at an epsilon instead would emit a finite but
/// absurd figure (a zero `attributed_us_mean` produced
/// `pct_of_attributed=1000000000000.00`), which a parser would happily ingest
/// as a real percentage. `n/a` cannot be mistaken for data.
fn pct(num: f64, den: f64) -> String {
    if den <= 1e-6 || !den.is_finite() || !num.is_finite() {
        "n/a".to_string()
    } else {
        format!("{:.2}", 100.0 * num / den)
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
            health: p.health,
            health_delta: p.health.since(&p.health_baseline),
        }
    })
    .unwrap_or_default()
}

/// Clear per-phase and per-token accumulation, and rebase the health delta.
///
/// The event pool and the lifetime health counters survive, so pool reuse
/// continues and a defect cannot be erased by a segment boundary.
pub fn reset() {
    let _ = with(|p| {
        p.accum = Accum::default();
        p.health_baseline = p.health;
    });
}

/// RAII guard that flushes one decode segment on drop.
///
/// This exists because flushing at the end of the decode loop is not enough.
/// `Engine::generate` propagates decode errors with `?`, so a statement placed
/// after the loop is skipped on error -- while the tokens that already
/// completed have each been folded into the accumulator by their own
/// `token_settle`. Those orphans would then be silently reported inside the
/// NEXT successful segment's block, under a label implying they belonged to it.
///
/// Dropping is the one thing that happens on every exit path, so the flush
/// lives here. A failed segment reports the tokens it did complete, under its
/// own label, and leaves nothing behind.
pub struct SegmentGuard {
    seq: u64,
}

impl Drop for SegmentGuard {
    fn drop(&mut self) {
        if level() == 0 {
            return;
        }
        print_and_reset(&format!("gen{}", self.seq));
        // print_and_reset only resets when it printed (tokens > 0); force the
        // rebase so an empty segment cannot carry state forward either.
        reset();
    }
}

/// Open a decode segment. The returned guard must be held for the whole
/// segment; it prints and clears on drop. No-op when profiling is disabled.
pub fn begin_segment() -> SegmentGuard {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if level() > 0 {
        // Defensive: if a previous guard could not run (process teardown, a
        // panic with abort), discard whatever it left rather than attributing
        // it here.
        let _ = with(|p| {
            if p.accum.tokens() > 0 {
                p.accum = Accum::default();
            }
            p.health_baseline = p.health;
        });
    }
    SegmentGuard { seq }
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
    let wall = s.wall_us_mean;
    let attributed = s.attributed_us_mean;

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
        "[PROFILE] residual gpu_unattributed_us={:.3} gpu_unattributed_pct_of_wall={} \
         host_outside_span_us={:.3} host_outside_span_pct_of_wall={}",
        unattributed,
        pct(unattributed, wall),
        host_gap,
        pct(host_gap, wall),
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
             pct_of_wall={} pct_of_attributed={}",
            row.phase.depth(),
            row.phase.name(),
            parent,
            calls_per_tok,
            per_tok,
            pct(per_tok, wall),
            pct(per_tok, attributed),
        );
    }

    // Per-parent residual: time inside a depth-0 phase claimed by none of its
    // depth-1 children. For a fully tiled parent this is submission gap; if it
    // GROWS after a code change, a new launch went in uncovered. This is the
    // only handle the tool offers on that distinction -- the global
    // `gpu_unattributed_us` conflates the two causes and cannot separate them.
    for parent in ALL_PHASES.iter().filter(|p| p.depth() == 0) {
        if let Some(uncovered) = s.uncovered_in(*parent) {
            let total = s.phase_us_per_token(*parent).unwrap_or(0.0);
            let _ = writeln!(
                w,
                "[PROFILE] uncovered parent={} us_per_token={:.3} pct_of_parent={}",
                parent.name(),
                uncovered,
                pct(uncovered, total),
            );
        }
    }

    // lm_head is not directly bracketable (ten early returns in the dispatch
    // chain), so report it as Head minus FinalNorm. When the F16 fast path
    // returns before the final RMSNorm, FinalNorm never runs and the difference
    // is undefined -- say so explicitly rather than omitting the row, so a
    // missing lm_head figure cannot be mistaken for a zero-cost lm_head.
    let head = s.phase_us_per_token(Phase::Head);
    let final_norm = s.phase_us_per_token(Phase::FinalNorm);
    match (head, final_norm) {
        (Some(h), Some(fnorm)) => {
            let lm = (h - fnorm).max(0.0);
            let _ = writeln!(
                w,
                "[PROFILE] derived name=lm_head from=head-final_norm us_per_token={:.3} \
                 pct_of_wall={}",
                lm,
                pct(lm, wall),
            );
        }
        (Some(_), None) => {
            let _ = writeln!(
                w,
                "[PROFILE] derived name=lm_head status=unavailable \
                 reason=final_norm_never_ran note=f16_fast_path_or_level_1"
            );
        }
        _ => {}
    }

    let brackets_per_token =
        s.rows.iter().map(|r| r.calls).sum::<u64>() as f64 / s.tokens.max(1) as f64;
    // Lifetime totals, then this segment's delta. The lifetime line alone would
    // make a defect from segment 0 read identically in every later segment.
    let _ = writeln!(
        w,
        "[PROFILE] health scope=lifetime clean={} unclosed_brackets={} nest_errors={} \
         event_errors={} outside_token_brackets={} negative_residual_tokens={} \
         abandoned_tokens={} pool_high_water={} brackets_per_token={:.1}",
        s.health.is_clean(),
        s.health.unclosed,
        s.health.nest_errors,
        s.health.event_errors,
        s.health.outside_token,
        s.health.negative_residual,
        s.health.abandoned_tokens,
        s.health.pool_high_water,
        brackets_per_token,
    );
    let _ = writeln!(
        w,
        "[PROFILE] health scope=segment clean={} unclosed_brackets={} nest_errors={} \
         event_errors={} outside_token_brackets={} negative_residual_tokens={} \
         abandoned_tokens={}",
        s.health_delta.is_clean(),
        s.health_delta.unclosed,
        s.health_delta.nest_errors,
        s.health_delta.event_errors,
        s.health_delta.outside_token,
        s.health_delta.negative_residual,
        s.health_delta.abandoned_tokens,
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

    /// The default-OFF contract, to the extent it is testable without a GPU.
    ///
    /// `begin`/`end`/`token_begin`/`token_end` need a live `CudaStream`, which
    /// cannot be constructed on a machine without an NVIDIA device, so this
    /// covers only the gate itself plus the two stream-free entry points. The
    /// gate is what every other entry point checks first, so proving it reads 0
    /// and that the stream-free paths are inert is the meaningful part; the
    /// remainder is verified by inspection (each begins with the same
    /// `level() == 0` early return).
    #[test]
    fn disabled_by_default_and_stream_free_entry_points_are_inert() {
        if std::env::var("LUMEN_CUDA_PROFILE").is_ok() {
            // An operator running the suite with the flag exported would
            // otherwise see a spurious failure. The contract under test is the
            // default, so skip rather than assert the wrong thing.
            return;
        }
        assert_eq!(level(), 0, "must be disabled when the env var is unset");
        assert!(!enabled());

        // Neither of these may panic, print, or accumulate state.
        token_settle();
        print_and_reset("must-not-print");

        let s = summary();
        assert_eq!(s.tokens, 0);
        assert!(s.rows.is_empty());
        assert!(s.health.is_clean());
        assert!(s.health_delta.is_clean());
        assert_eq!(
            s.health.pool_high_water, 0,
            "no event may be allocated when off"
        );
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
            health: HealthCounts {
                pool_high_water: 634,
                ..Default::default()
            },
            health_delta: HealthCounts::default(),
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
        assert_eq!(kinds.iter().filter(|k| *k == "health").count(), 2);
        // No NaN/inf can reach a parser.
        assert!(!out.contains("NaN") && !out.contains("inf"), "{out}");
    }

    /// Renders a representative `[PROFILE]` block so the output format can be
    /// inspected without a GPU. Ignored by default because it is a
    /// documentation aid, not an assertion:
    ///
    /// ```text
    /// cargo test -p lumen-runtime --features cuda --lib \
    ///     cuda::profiler::tests::show_profile_output_format -- --ignored --nocapture
    /// ```
    ///
    /// Shape numbers match Qwen3.5-9B (24 GDN + 8 full-attn); the timings are
    /// invented and must not be quoted as measurements.
    #[test]
    #[ignore = "documentation aid: run with --ignored --nocapture to see the output format"]
    fn show_profile_output_format() {
        // (phase, calls/token, us/token) -- fabricated, illustrative only.
        let shape: &[(Phase, u64, f64)] = &[
            (Phase::Embed, 1, 3.1),
            (Phase::GdnAttn, 24, 3980.0),
            (Phase::FullAttn, 8, 1520.0),
            (Phase::Ffn, 32, 2110.0),
            (Phase::LayerCommit, 32, 96.0),
            (Phase::Head, 1, 640.0),
            (Phase::Argmax, 1, 21.0),
            (Phase::GdnQkv, 24, 1450.0),
            (Phase::GdnConvRecur, 24, 1900.0),
            (Phase::GdnOut, 24, 520.0),
            (Phase::GdnGlue, 24, 60.0),
            (Phase::AttnQkv, 8, 610.0),
            (Phase::AttnQkNorm, 8, 90.0),
            (Phase::AttnRope, 8, 62.0),
            (Phase::AttnKvWrite, 8, 74.0),
            (Phase::AttnCore, 8, 300.0),
            (Phase::AttnWo, 8, 340.0),
            (Phase::FfnGateUp, 32, 1290.0),
            (Phase::FfnDownResid, 32, 790.0),
            (Phase::FinalNorm, 1, 8.0),
        ];
        let tokens = 128usize;
        let attributed: f64 = shape
            .iter()
            .filter(|(p, _, _)| p.depth() == 0)
            .map(|(_, _, us)| us)
            .sum();
        let s = Summary {
            tokens,
            rows: shape
                .iter()
                .map(|(p, calls, us)| PhaseRow {
                    phase: *p,
                    calls: calls * tokens as u64,
                    total_us: us * tokens as f64,
                })
                .collect(),
            wall_us_mean: 9120.0,
            wall_us_p50: 9105.0,
            gpu_span_us_mean: 8790.0,
            gpu_span_us_p50: 8781.0,
            attributed_us_mean: attributed,
            attributed_us_p50: attributed,
            health: HealthCounts {
                pool_high_water: 634,
                ..Default::default()
            },
            health_delta: HealthCounts::default(),
        };
        let mut buf: Vec<u8> = Vec::new();
        write_summary(&mut buf, "gen1", &s);
        print!("{}", String::from_utf8(buf).expect("utf8"));
    }

    /// A depth-0 phase whose children do not account for all of its time must
    /// report the shortfall, and it must be attributed to the right parent.
    #[test]
    fn uncovered_localizes_the_shortfall_per_parent() {
        let s = Summary {
            tokens: 2,
            rows: vec![
                // full_attn 1000/token, children sum to 900 -> 100 uncovered.
                PhaseRow { phase: Phase::FullAttn, calls: 2, total_us: 2000.0 },
                PhaseRow { phase: Phase::AttnQkv, calls: 2, total_us: 1200.0 },
                PhaseRow { phase: Phase::AttnCore, calls: 2, total_us: 600.0 },
                // ffn 500/token, children sum to 500 -> 0 uncovered (tiled).
                PhaseRow { phase: Phase::Ffn, calls: 2, total_us: 1000.0 },
                PhaseRow { phase: Phase::FfnGateUp, calls: 2, total_us: 600.0 },
                PhaseRow { phase: Phase::FfnDownResid, calls: 2, total_us: 400.0 },
                // argmax has no children at all -> None, not Some(0).
                PhaseRow { phase: Phase::Argmax, calls: 2, total_us: 40.0 },
            ],
            wall_us_mean: 2000.0,
            ..Default::default()
        };
        assert_eq!(s.uncovered_in(Phase::FullAttn), Some(100.0));
        assert_eq!(s.uncovered_in(Phase::Ffn), Some(0.0));
        assert_eq!(
            s.uncovered_in(Phase::Argmax),
            None,
            "a phase with no recorded children must not report a shortfall"
        );
        assert_eq!(
            s.uncovered_in(Phase::GdnAttn),
            None,
            "a phase that never ran must not report a shortfall"
        );

        let mut buf: Vec<u8> = Vec::new();
        write_summary(&mut buf, "unit", &s);
        let out = String::from_utf8(buf).expect("utf8");
        assert!(
            out.contains("uncovered parent=full_attn us_per_token=100.000 pct_of_parent=10.00"),
            "{out}"
        );
        assert!(out.contains("uncovered parent=ffn us_per_token=0.000"), "{out}");
        assert!(!out.contains("parent=argmax us_per_token"), "{out}");
    }

    /// lm_head must never be silently absent: when `final_norm` did not run the
    /// difference is undefined, and an omitted row could be misread as a
    /// zero-cost lm_head.
    #[test]
    fn lm_head_unavailability_is_stated_not_omitted() {
        let s = Summary {
            tokens: 1,
            rows: vec![PhaseRow { phase: Phase::Head, calls: 1, total_us: 500.0 }],
            wall_us_mean: 1000.0,
            ..Default::default()
        };
        let mut buf: Vec<u8> = Vec::new();
        write_summary(&mut buf, "unit", &s);
        let out = String::from_utf8(buf).expect("utf8");
        assert!(
            out.contains("name=lm_head status=unavailable reason=final_norm_never_ran"),
            "{out}"
        );
        assert!(!out.contains("from=head-final_norm"), "{out}");
    }

    /// Both health scopes must be emitted, and the segment delta must be
    /// independent of the lifetime totals.
    #[test]
    fn health_reports_lifetime_and_segment_separately() {
        let s = Summary {
            tokens: 1,
            rows: vec![PhaseRow { phase: Phase::Embed, calls: 1, total_us: 5.0 }],
            wall_us_mean: 100.0,
            health: HealthCounts {
                unclosed: 7,
                nest_errors: 2,
                negative_residual: 1,
                abandoned_tokens: 3,
                pool_high_water: 200,
                ..Default::default()
            },
            // This segment contributed only one of the seven unclosed brackets.
            health_delta: HealthCounts { unclosed: 1, ..Default::default() },
            ..Default::default()
        };
        let mut buf: Vec<u8> = Vec::new();
        write_summary(&mut buf, "unit", &s);
        let out = String::from_utf8(buf).expect("utf8");
        assert!(
            out.contains("health scope=lifetime clean=false unclosed_brackets=7 nest_errors=2"),
            "{out}"
        );
        assert!(out.contains("negative_residual_tokens=1"), "{out}");
        assert!(out.contains("abandoned_tokens=3"), "{out}");
        assert!(
            out.contains("health scope=segment clean=false unclosed_brackets=1 nest_errors=0"),
            "{out}"
        );
    }

    #[test]
    fn health_since_is_fieldwise_and_saturating() {
        let base = HealthCounts {
            unclosed: 5,
            nest_errors: 1,
            event_errors: 2,
            outside_token: 3,
            negative_residual: 4,
            abandoned_tokens: 6,
            pool_high_water: 10,
        };
        let now = HealthCounts {
            unclosed: 9,
            nest_errors: 1,
            event_errors: 2,
            outside_token: 5,
            negative_residual: 4,
            abandoned_tokens: 6,
            pool_high_water: 42,
        };
        let d = now.since(&base);
        assert_eq!(d.unclosed, 4);
        assert_eq!(d.nest_errors, 0);
        assert_eq!(d.outside_token, 2);
        assert!(d.is_clean() == false);
        // pool_high_water is a watermark, not a count: carried, not differenced.
        assert_eq!(d.pool_high_water, 42);
        // A baseline ahead of the current value (can only happen if health were
        // ever reset) must saturate to 0, not underflow-panic.
        assert_eq!(base.since(&now).unclosed, 0);
        assert!(HealthCounts::default().is_clean());
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
