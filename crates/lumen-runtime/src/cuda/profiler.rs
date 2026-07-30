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
    /// FFN down projection and the final residual add. Merged because two arms
    /// fold the residual into the down store and return early.
    FfnDownResid = 20,
    /// Final RMSNorm.
    FinalNorm = 21,

    // ---- depth 1: inside Head (siblings of FinalNorm) --------------------
    /// lm_head activation quantize (Q8_1), when the head runs on int8.
    HeadQuant = 22,
    /// The lm_head projection itself. Bracketed directly now that the
    /// dispatch chain is hoisted behind `dispatch_output_proj`, so
    /// `head == final_norm + head_quant + head_mv` is an identity rather than
    /// a subtraction.
    HeadMv = 23,

    // ---- depth 2: inside AttnQkv ----------------------------------------
    /// `wq` (fused Q+gate) on a Q6_K-coupled full-attention layer. Split from
    /// the Q4 twin because on 9B-Q4 only `attn_q` is Q6_K, on 4 of the 8
    /// full-attention layers, and the coupling guard drags the natively-Q4
    /// `wk`/`wv` onto the same F16 path with it. Campaign lever C1/C1b.
    AttnWqQ6k = 24,
    /// `wq` on a natively-Q4 full-attention layer.
    AttnWqQ4 = 25,
    /// `wk`/`wv` on a Q6_K-coupled layer -- natively Q4_0 weights dragged onto
    /// the coupled path. Campaign lever C2.
    AttnWkvQ6k = 26,
    /// `wk`/`wv` on a natively-Q4 layer.
    AttnWkvQ4 = 27,
    /// QKV activation quantize (shared by wq/wk/wv when the plan is Q8_1).
    AttnQkvQuant = 28,

    // ---- depth 2: inside AttnWo ------------------------------------------
    /// `wo` activation quantize.
    AttnWoQuant = 29,
    /// `wo` projection (folds the post-attention residual).
    AttnWoMv = 30,

    // ---- depth 2: inside Ffn (gate/up group) ----------------------------
    /// FFN activation quantize. Shared by gate and up when both are int8, so
    /// `calls_per_token` below 2x the matvec count is the shared-quantize
    /// optimization working; equal to it means a duplicate.
    FfnActQuant = 31,
    /// FFN gate projection.
    FfnGateMv = 32,
    /// FFN up projection.
    FfnUpMv = 33,
    /// Fused gate+up in one launch (label "gate_up"). Cannot be split into
    /// gate and up -- it is one kernel.
    FfnGateUpFused = 34,
    /// SwiGLU, when it is a standalone launch rather than fused into the
    /// gate/up or down kernel.
    FfnSwiglu = 35,

    // ---- depth 2: inside Ffn (down group) -------------------------------
    /// FFN down activation quantize.
    FfnDownQuant = 36,
    /// FFN down projection.
    FfnDownMv = 37,
    /// Final residual add, when not folded into the down store.
    FfnResidual = 38,

    // ---- depth 2: inside GdnQkv -----------------------------------------
    /// RMSNorm feeding the GDN projections.
    GdnNorm = 39,
    /// GDN fused qkv in_proj.
    GdnQkvMv = 40,
    /// GDN attn-gate projection (the z of silu(z)).
    GdnGateMv = 41,
    /// GDN ssm_alpha projection.
    GdnAlphaMv = 42,
    /// GDN ssm_beta projection.
    GdnBetaMv = 43,
    /// GDN activation quantize. alpha and beta each re-quantize the SAME
    /// `normed` buffer on the shipped plan, so this is where that duplicate
    /// shows up as calls_per_token.
    GdnActQuant = 44,

    // ---- depth 2: inside GdnConvRecur -----------------------------------
    /// Causal conv1d over the conv ring.
    GdnConv = 45,
    /// Recurrent delta-rule state update (gates, L2 norm, state update).
    GdnRecur = 46,

    // ---- depth 2: inside GdnOut -----------------------------------------
    /// Output RMSNorm + silu(z) gating.
    GdnNormGate = 47,
    /// `ssm_out` activation quantize. Named for the tensor, NOT a family:
    /// ssm_out is Q8Raw/int8 and sits OUTSIDE the Q4 activation plan.
    SsmOutQuant = 48,
    /// `ssm_out` projection. Same naming rationale as above.
    SsmOutMv = 49,
}

/// Number of [`Phase`] variants.
pub const PHASE_COUNT: usize = 50;

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
    Phase::HeadQuant,
    Phase::HeadMv,
    Phase::AttnWqQ6k,
    Phase::AttnWqQ4,
    Phase::AttnWkvQ6k,
    Phase::AttnWkvQ4,
    Phase::AttnQkvQuant,
    Phase::AttnWoQuant,
    Phase::AttnWoMv,
    Phase::FfnActQuant,
    Phase::FfnGateMv,
    Phase::FfnUpMv,
    Phase::FfnGateUpFused,
    Phase::FfnSwiglu,
    Phase::FfnDownQuant,
    Phase::FfnDownMv,
    Phase::FfnResidual,
    Phase::GdnNorm,
    Phase::GdnQkvMv,
    Phase::GdnGateMv,
    Phase::GdnAlphaMv,
    Phase::GdnBetaMv,
    Phase::GdnActQuant,
    Phase::GdnConv,
    Phase::GdnRecur,
    Phase::GdnNormGate,
    Phase::SsmOutQuant,
    Phase::SsmOutMv,
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
            Phase::HeadQuant => "head_quant",
            Phase::HeadMv => "head_mv",
            Phase::AttnWqQ6k => "attn_wq_q6k",
            Phase::AttnWqQ4 => "attn_wq_q4",
            Phase::AttnWkvQ6k => "attn_wkv_q6k",
            Phase::AttnWkvQ4 => "attn_wkv_q4",
            Phase::AttnQkvQuant => "attn_qkv_quant",
            Phase::AttnWoQuant => "attn_wo_quant",
            Phase::AttnWoMv => "attn_wo_mv",
            Phase::FfnActQuant => "ffn_act_quant",
            Phase::FfnGateMv => "ffn_gate_mv",
            Phase::FfnUpMv => "ffn_up_mv",
            Phase::FfnGateUpFused => "ffn_gate_up_fused",
            Phase::FfnSwiglu => "ffn_swiglu",
            Phase::FfnDownQuant => "ffn_down_quant",
            Phase::FfnDownMv => "ffn_down_mv",
            Phase::FfnResidual => "ffn_residual",
            Phase::GdnNorm => "gdn_norm",
            Phase::GdnQkvMv => "gdn_qkv_mv",
            Phase::GdnGateMv => "gdn_gate_mv",
            Phase::GdnAlphaMv => "gdn_alpha_mv",
            Phase::GdnBetaMv => "gdn_beta_mv",
            Phase::GdnActQuant => "gdn_act_quant",
            Phase::GdnConv => "gdn_conv",
            Phase::GdnRecur => "gdn_recur",
            Phase::GdnNormGate => "gdn_norm_gate",
            Phase::SsmOutQuant => "ssm_out_quant",
            Phase::SsmOutMv => "ssm_out_mv",
        }
    }

    /// 0 = token partition (level 1). 1 = coarse group. 2 = per-surface leaf.
    /// Depths 1 and 2 both require level 2.
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
            Phase::AttnQkv
            | Phase::AttnQkNorm
            | Phase::AttnBias
            | Phase::AttnRope
            | Phase::AttnKvWrite
            | Phase::AttnCore
            | Phase::AttnWo
            | Phase::GdnQkv
            | Phase::GdnConvRecur
            | Phase::GdnOut
            | Phase::GdnGlue
            | Phase::FfnGateUp
            | Phase::FfnDownResid
            | Phase::FinalNorm
            | Phase::HeadQuant
            | Phase::HeadMv => 1,
            _ => 2,
        }
    }

    /// True when this phase records at profiler level `lvl`.
    ///
    /// Level 1 is depth-0 only, which keeps level-1 output byte-identical to
    /// the pre-depth-2 instrument. Level 2 records every depth.
    pub fn included_at(self, lvl: u8) -> bool {
        lvl > 0 && (self.depth() == 0 || lvl >= 2)
    }

    /// The phase one level up that this phase refines, if any.
    pub fn parent(self) -> Option<Phase> {
        match self {
            // depth 1 -> depth 0
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
            Phase::FinalNorm | Phase::HeadQuant | Phase::HeadMv => Some(Phase::Head),
            // depth 2 -> depth 1
            Phase::AttnWqQ6k
            | Phase::AttnWqQ4
            | Phase::AttnWkvQ6k
            | Phase::AttnWkvQ4
            | Phase::AttnQkvQuant => Some(Phase::AttnQkv),
            Phase::AttnWoQuant | Phase::AttnWoMv => Some(Phase::AttnWo),
            Phase::FfnActQuant
            | Phase::FfnGateMv
            | Phase::FfnUpMv
            | Phase::FfnGateUpFused
            | Phase::FfnSwiglu => Some(Phase::FfnGateUp),
            Phase::FfnDownQuant | Phase::FfnDownMv | Phase::FfnResidual => {
                Some(Phase::FfnDownResid)
            }
            Phase::GdnNorm
            | Phase::GdnQkvMv
            | Phase::GdnGateMv
            | Phase::GdnAlphaMv
            | Phase::GdnBetaMv
            | Phase::GdnActQuant => Some(Phase::GdnQkv),
            Phase::GdnConv | Phase::GdnRecur => Some(Phase::GdnConvRecur),
            Phase::GdnNormGate | Phase::SsmOutQuant | Phase::SsmOutMv => Some(Phase::GdnOut),
            _ => None,
        }
    }

    /// True for phases that sit physically inside a larger bracket but whose
    /// span must NOT be attributed to it.
    ///
    /// The activation quantizes: they run inside the matvec helper's surface
    /// bracket, but they move activation bytes, not weight bytes, so leaving
    /// them in would dilute the matvec's effective bandwidth and double-count
    /// them inside the parent group.
    pub fn excludes_span_from_parent(self) -> bool {
        matches!(
            self,
            Phase::AttnQkvQuant
                | Phase::AttnWoQuant
                | Phase::FfnActQuant
                | Phase::FfnDownQuant
                | Phase::GdnActQuant
                | Phase::SsmOutQuant
                | Phase::HeadQuant
        )
    }

    /// True for phases whose reported bytes are weight bytes streamed by a
    /// matvec, so an effective GB/s is meaningful.
    pub fn is_matvec(self) -> bool {
        matches!(
            self,
            Phase::AttnWqQ6k
                | Phase::AttnWqQ4
                | Phase::AttnWkvQ6k
                | Phase::AttnWkvQ4
                | Phase::AttnWoMv
                | Phase::FfnGateMv
                | Phase::FfnUpMv
                | Phase::FfnGateUpFused
                | Phase::FfnDownMv
                | Phase::GdnQkvMv
                | Phase::GdnGateMv
                | Phase::GdnAlphaMv
                | Phase::GdnBetaMv
                | Phase::SsmOutMv
                | Phase::HeadMv
        )
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
    /// Bytes declared at `begin`; folded into the phase total at settle so an
    /// abandoned token contributes neither time nor bytes.
    bytes: u64,
    /// Unique within the token, so a child can name its enclosing bracket
    /// without depending on `Vec` positions.
    id: u64,
    /// When set, this bracket's span is SUBTRACTED from the enclosing bracket
    /// with that id before the enclosing phase accumulates.
    ///
    /// This is what lets an activation quantize be timed separately while
    /// sitting physically inside the matvec helper's surface bracket. Without
    /// it the quantize would be counted twice inside the parent group, and the
    /// matvec's reported bandwidth would be diluted by quantize time that moved
    /// no weight bytes. With it, a matvec surface reports NET matvec time, which
    /// is the only figure from which an honest effective GB/s can be derived.
    exclude_from: Option<u64>,
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

struct Accum {
    /// Total GPU-timeline span per phase, microseconds.
    span_us: [f64; PHASE_COUNT],
    /// Bracket count per phase.
    calls: [u64; PHASE_COUNT],
    /// Bytes the phase declared it would move, summed. Weight bytes for a
    /// matvec; 0 for phases that declare nothing.
    bytes: [u64; PHASE_COUNT],
    /// Per-token samples, one entry per settled token.
    wall_us: Vec<f64>,
    gpu_span_us: Vec<f64>,
    attributed_us: Vec<f64>,
}

/// Manual `Default`: `#[derive(Default)]` only covers arrays up to 32
/// elements and `PHASE_COUNT` is 50.
impl Default for Accum {
    fn default() -> Self {
        Self {
            span_us: [0.0; PHASE_COUNT],
            calls: [0; PHASE_COUNT],
            bytes: [0; PHASE_COUNT],
            wall_us: Vec::new(),
            gpu_span_us: Vec::new(),
            attributed_us: Vec::new(),
        }
    }
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
    /// Open brackets, innermost last: (phase, start event index, bytes, id).
    open: Vec<(u8, u32, u64, u64)>,
    /// Monotonic bracket id within the token.
    next_id: u64,
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
            next_id: 0,
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

        let closed = std::mem::take(&mut self.closed);

        // Pass 1: measure every bracket, and total up what each parent must
        // have netted out.
        let mut spans: Vec<(usize, f64, u64, u64)> = Vec::with_capacity(closed.len());
        let mut excluded: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
        for b in &closed {
            let Some(us) = self.elapsed_us(b.start, b.end) else {
                continue;
            };
            if let Some(pid) = b.exclude_from {
                *excluded.entry(pid).or_insert(0.0) += us;
            }
            spans.push((b.phase as usize, us, b.bytes, b.id));
        }

        // Pass 2: accumulate, subtracting each parent's excluded children.
        let mut attributed = 0.0f64;
        for (i, us, bytes, id) in spans {
            let net = match excluded.get(&id) {
                // Clamp at 0: a child cannot legitimately outlast its parent,
                // and the per-token residual check below catches the case where
                // the arithmetic goes wrong.
                Some(sub) => (us - sub).max(0.0),
                None => us,
            };
            self.accum.span_us[i] += net;
            self.accum.calls[i] += 1;
            self.accum.bytes[i] = self.accum.bytes[i].saturating_add(bytes);
            if ALL_PHASES[i].depth() == 0 {
                attributed += net;
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
        self.next_id = 0;
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
    if !phase.included_at(lvl) {
        return;
    }
    with(|p| {
        if !p.in_token {
            p.health.outside_token += 1;
            return;
        }
        if let Some(idx) = p.record(stream) {
            let id = p.next_id;
            p.next_id += 1;
            p.open.push((phase as u8, idx, 0, id));
        }
    });
}

/// Open a phase bracket that declares `bytes` of traffic, so the report can
/// derive an effective bandwidth for it.
#[inline]
pub fn begin_bytes(phase: Phase, bytes: u64, stream: &CudaStream) {
    if !phase.included_at(level()) {
        return;
    }
    with(|p| {
        if !p.in_token {
            p.health.outside_token += 1;
            return;
        }
        if let Some(idx) = p.record(stream) {
            let id = p.next_id;
            p.next_id += 1;
            p.open.push((phase as u8, idx, bytes, id));
        }
    });
}

/// Close the innermost phase bracket, which must be `phase`.
#[inline]
pub fn end(phase: Phase, stream: &CudaStream) {
    let lvl = level();
    if !phase.included_at(lvl) {
        return;
    }
    with(|p| {
        if !p.in_token {
            // Symmetric with `begin`: no bracket was opened, so closing one
            // here is not a nesting error.
            return;
        }
        match p.open.last().copied() {
            Some((open_phase, start, bytes, id)) if open_phase == phase as u8 => {
                p.open.pop();
                // The bracket now on top of the stack is this one's enclosing
                // bracket. A quantize phase names it so its span is netted out.
                let exclude_from = if phase.excludes_span_from_parent() {
                    p.open.last().map(|(_, _, _, pid)| *pid)
                } else {
                    None
                };
                if let Some(end_idx) = p.record(stream) {
                    p.closed.push(Bracket {
                        phase: phase as u8,
                        start,
                        end: end_idx,
                        bytes,
                        id,
                        exclude_from,
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
// Per-surface brackets keyed on the dispatch label
// ---------------------------------------------------------------------------

/// Which full-attention family the layer currently being decoded belongs to.
///
/// On Qwen3.5-9B Q4_0 the only K-quant layer tensor is `attn_q`, Q6_K on 4 of
/// the 8 full-attention layers. The dispatch guard for those layers also drags
/// the natively-Q4 `wk`/`wv` onto the coupled path, so the split is a property
/// of the LAYER's dispatch branch, not of each tensor's own format. Set once
/// per full-attention layer from the same condition the dispatcher uses, so the
/// bracket boundary and the branch boundary coincide by construction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AttnFamily {
    /// `wq` is Q6_K-derived (F32 by default, `Q6KRaw` under C1/C1b).
    Q6kCoupled,
    /// `wq` is natively Q4.
    NativeQ4,
}

/// Current attention family, consulted by the label->phase map.
///
/// A plain global rather than thread-local: decode is single-threaded per
/// backend and every bracket call is already serialized by the profiler mutex.
/// Defaults to `NativeQ4` so a missing `set_attn_family` cannot silently
/// attribute Q4 work to the Q6_K family -- it would show as an implausible
/// zero on the Q6_K rows instead, which is visible.
static ATTN_FAMILY: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(1);

/// Record which full-attention family the current layer takes. No-op when off.
#[inline]
pub fn set_attn_family(family: AttnFamily) {
    if level() == 0 {
        return;
    }
    let v = match family {
        AttnFamily::Q6kCoupled => 0u8,
        AttnFamily::NativeQ4 => 1u8,
    };
    ATTN_FAMILY.store(v, std::sync::atomic::Ordering::Relaxed);
}

fn attn_family() -> AttnFamily {
    if ATTN_FAMILY.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        AttnFamily::Q6kCoupled
    } else {
        AttnFamily::NativeQ4
    }
}

/// Map a dispatch label to its matvec phase.
///
/// The labels are already threaded through every matvec helper, which is what
/// makes per-surface bracketing possible without touching ~98 call sites. Two
/// labels name a launch that covers two surfaces and therefore cannot be
/// split: `"kv"` (wk and wv in one batched GEMM) and `"gate_up"` (gate and up
/// in one kernel). Those get their own phase named for what the launch is,
/// rather than being attributed to one of the two surfaces.
pub fn matvec_phase_for_label(label: &str) -> Option<Phase> {
    let q6k = attn_family() == AttnFamily::Q6kCoupled;
    Some(match label {
        "wq" => {
            if q6k {
                Phase::AttnWqQ6k
            } else {
                Phase::AttnWqQ4
            }
        }
        // "kv" is wk+wv fused into one batched launch -- inseparable.
        "wk" | "wv" | "kv" => {
            if q6k {
                Phase::AttnWkvQ6k
            } else {
                Phase::AttnWkvQ4
            }
        }
        "wo" => Phase::AttnWoMv,
        "gate" => Phase::FfnGateMv,
        "up" => Phase::FfnUpMv,
        "gate_up" => Phase::FfnGateUpFused,
        "down" => Phase::FfnDownMv,
        "gdn_qkv" => Phase::GdnQkvMv,
        "gdn_gate" => Phase::GdnGateMv,
        "gdn_alpha" | "gdn_alpha_f16" => Phase::GdnAlphaMv,
        "gdn_beta" | "gdn_beta_f16" => Phase::GdnBetaMv,
        // ssm_out is Q8Raw/int8 and sits OUTSIDE the Q4 activation plan, so it
        // is labelled by its own tensor name and never folded into a family.
        "gdn_ssm_out" => Phase::SsmOutMv,
        "output_proj" | "head" => Phase::HeadMv,
        _ => return None,
    })
}

/// Map a dispatch label to the activation-quantize phase that feeds it.
///
/// Covers both spellings: the caller-side `launch_quantize_input_q8_1` labels
/// ("qkv", "wo split", "ffn gate_up", "down split", ...) and the matvec labels,
/// because on the int8 arms the quantize happens inside the matvec helper under
/// the matvec's own label.
pub fn quant_phase_for_label(label: &str) -> Option<Phase> {
    let q6k = attn_family() == AttnFamily::Q6kCoupled;
    let _ = q6k;
    Some(match label {
        "qkv" | "wq" | "wk" | "wv" | "kv" => Phase::AttnQkvQuant,
        "wo" | "wo split" => Phase::AttnWoQuant,
        "gate" | "up" | "gate_up" | "ffn gate_up" => Phase::FfnActQuant,
        "down" | "down split" | "down split (sep swiglu)" => Phase::FfnDownQuant,
        "gdn_qkv" | "gdn_gate" | "gdn_alpha" | "gdn_beta" | "gdn_alpha_f16"
        | "gdn_beta_f16" => Phase::GdnActQuant,
        "gdn_ssm_out" => Phase::SsmOutQuant,
        "output_proj" | "head" => Phase::HeadQuant,
        _ => return None,
    })
}

/// True when `phase` already has an open bracket in this token.
///
/// Guards against re-entrancy double-counting: `launch_matvec_preq8_1_split`
/// tail-calls `launch_matvec_preq8_1` when the split sibling or kernel is
/// absent, and both carry a surface guard for the SAME label. Without this
/// check the surface would be counted twice for such a call. Checking the whole
/// open stack rather than just its top makes it robust to deeper nesting.
fn phase_already_open(phase: Phase) -> bool {
    with(|p| p.open.iter().any(|(ph, _, _, _)| *ph == phase as u8)).unwrap_or(false)
}

/// RAII bracket. Records `end` on drop, so it survives every early return and
/// every `?` in the helper it guards.
///
/// `launch_matvec_ext` has 23 `return` statements and `launch_matvec_residual`
/// has 15; closing by hand would be 38 edits where a single miss leaks a
/// bracket permanently.
pub struct SurfaceGuard<'a> {
    phase: Option<Phase>,
    stream: &'a CudaStream,
}

impl Drop for SurfaceGuard<'_> {
    fn drop(&mut self) {
        if let Some(p) = self.phase {
            end(p, self.stream);
        }
    }
}

/// Open a per-surface matvec bracket for `label`, declaring `bytes` of weight
/// traffic. Returns an inert guard when profiling is off or the label is not a
/// known matvec surface.
#[inline]
pub fn matvec_surface<'a>(
    label: &str,
    bytes: u64,
    stream: &'a CudaStream,
) -> SurfaceGuard<'a> {
    if level() < 2 {
        return SurfaceGuard {
            phase: None,
            stream,
        };
    }
    match matvec_phase_for_label(label) {
        // Re-entrant call for a surface already being timed: stay inert so the
        // outer bracket remains the single measurement of it.
        Some(ph) if phase_already_open(ph) => SurfaceGuard {
            phase: None,
            stream,
        },
        Some(ph) => {
            begin_bytes(ph, bytes, stream);
            SurfaceGuard {
                phase: Some(ph),
                stream,
            }
        }
        None => SurfaceGuard {
            phase: None,
            stream,
        },
    }
}

/// Open an activation-quantize bracket for `label`.
#[inline]
pub fn quant_surface<'a>(label: &str, bytes: u64, stream: &'a CudaStream) -> SurfaceGuard<'a> {
    if level() < 2 {
        return SurfaceGuard {
            phase: None,
            stream,
        };
    }
    match quant_phase_for_label(label) {
        // Re-entrant call for a surface already being timed: stay inert so the
        // outer bracket remains the single measurement of it.
        Some(ph) if phase_already_open(ph) => SurfaceGuard {
            phase: None,
            stream,
        },
        Some(ph) => {
            begin_bytes(ph, bytes, stream);
            SurfaceGuard {
                phase: Some(ph),
                stream,
            }
        }
        None => SurfaceGuard {
            phase: None,
            stream,
        },
    }
}

/// Open a bracket for a named phase declaring `bytes` of traffic, as a guard.
///
/// The guard form matters where a region has many exits: the lm_head dispatch
/// chain has nine early returns, and `Drop` fires on all of them plus the tail,
/// so the region is measured directly instead of being derived by subtraction
/// -- with no code motion and no risk of missing an exit.
#[inline]
pub fn guard_bytes<'a>(phase: Phase, bytes: u64, stream: &'a CudaStream) -> SurfaceGuard<'a> {
    if !phase.included_at(level()) {
        return SurfaceGuard {
            phase: None,
            stream,
        };
    }
    begin_bytes(phase, bytes, stream);
    SurfaceGuard {
        phase: Some(phase),
        stream,
    }
}

/// Open a bracket for a named phase, as a guard. For straight-line regions
/// where a manual `end` would still be correct but a guard is tidier.
#[inline]
pub fn guard<'a>(phase: Phase, stream: &'a CudaStream) -> SurfaceGuard<'a> {
    if !phase.included_at(level()) {
        return SurfaceGuard {
            phase: None,
            stream,
        };
    }
    begin(phase, stream);
    SurfaceGuard {
        phase: Some(phase),
        stream,
    }
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
    pub total_bytes: u64,
}

impl PhaseRow {
    /// Effective bandwidth over this phase's accumulated span, GB/s (10^9).
    /// `None` when the phase declared no bytes or took no measurable time.
    pub fn effective_gbps(&self) -> Option<f64> {
        if self.total_bytes == 0 || self.total_us <= 0.0 {
            return None;
        }
        Some(self.total_bytes as f64 / (self.total_us * 1e3))
    }
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

    /// Time inside a phase that none of its DIRECT children claimed.
    ///
    /// Generalized over depth: with depth-2 leaves present this yields a
    /// residual at every level, so an unbracketed launch is localized to the
    /// tightest enclosing bracket rather than only to a top-level phase.
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
                total_bytes: p.accum.bytes[i],
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
        // Bytes and effective bandwidth only where the phase declared traffic.
        // "n/a" rather than 0 so a phase that simply does not declare bytes is
        // never read as a phase that moved none.
        let bytes_per_tok = if row.total_bytes == 0 {
            "n/a".to_string()
        } else {
            format!("{:.0}", row.total_bytes as f64 / s.tokens.max(1) as f64)
        };
        let gbps = match row.effective_gbps() {
            Some(v) => format!("{v:.1}"),
            None => "n/a".to_string(),
        };
        let _ = writeln!(
            w,
            "[PROFILE] phase depth={} name={} parent={} calls_per_token={:.2} us_per_token={:.3} \
             pct_of_wall={} pct_of_attributed={} bytes_per_token={} eff_gbps={}",
            row.phase.depth(),
            row.phase.name(),
            parent,
            calls_per_tok,
            per_tok,
            pct(per_tok, wall),
            pct(per_tok, attributed),
            bytes_per_tok,
            gbps,
        );
    }

    // Per-parent residual: time inside a depth-0 phase claimed by none of its
    // depth-1 children. For a fully tiled parent this is submission gap; if it
    // GROWS after a code change, a new launch went in uncovered. This is the
    // only handle the tool offers on that distinction -- the global
    // `gpu_unattributed_us` conflates the two causes and cannot separate them.
    for parent in ALL_PHASES.iter() {
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

    // `head_mv` is now bracketed directly (the dispatch chain was hoisted
    // behind `dispatch_output_proj`), so this derived figure is kept only as a
    // CROSS-CHECK: it should agree with the measured `head_mv` row. A
    // disagreement means the head chain grew an exit that skips the bracket.
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

    /// The depth ladder must be exactly consistent: depth 0 has no parent, and
    /// every deeper phase's parent sits exactly one level shallower. This is
    /// what makes `uncovered_in` a correct residual at every level.
    #[test]
    fn depth_ladder_is_consistent() {
        for p in ALL_PHASES {
            match p.depth() {
                0 => assert_eq!(p.parent(), None, "{} is depth 0", p.name()),
                d => {
                    let parent = p
                        .parent()
                        .unwrap_or_else(|| panic!("{} (depth {d}) has no parent", p.name()));
                    assert_eq!(
                        parent.depth(),
                        d - 1,
                        "{} is depth {d} so its parent {} must be depth {}",
                        p.name(),
                        parent.name(),
                        d - 1
                    );
                }
            }
        }
    }

    /// Level 1 must record depth-0 only, so level-1 output stays identical to
    /// the pre-depth-2 instrument. Level 2 records everything.
    #[test]
    fn level_gating_matches_the_depth_ladder() {
        for p in ALL_PHASES {
            assert!(!p.included_at(0), "{} must be off at level 0", p.name());
            assert_eq!(
                p.included_at(1),
                p.depth() == 0,
                "{} at level 1: only depth-0 phases record",
                p.name()
            );
            assert!(p.included_at(2), "{} must record at level 2", p.name());
        }
    }

    /// Every activation-quantize phase must net its span out of the enclosing
    /// bracket, or the matvec it sits inside would report a diluted bandwidth
    /// and the parent group would double-count it.
    #[test]
    fn quantize_phases_exclude_their_span_from_the_parent() {
        let quant: Vec<&str> = ALL_PHASES
            .iter()
            .filter(|p| p.excludes_span_from_parent())
            .map(|p| p.name())
            .collect();
        assert_eq!(
            quant,
            vec![
                "head_quant",
                "attn_qkv_quant",
                "attn_wo_quant",
                "ffn_act_quant",
                "ffn_down_quant",
                "gdn_act_quant",
                "ssm_out_quant",
            ]
        );
        // A quantize phase must never also be counted as a matvec.
        for p in ALL_PHASES {
            assert!(
                !(p.excludes_span_from_parent() && p.is_matvec()),
                "{} cannot be both a quantize and a matvec",
                p.name()
            );
        }
    }

    /// The label map is what makes per-surface bracketing possible without
    /// touching ~98 call sites, so its coverage is load-bearing.
    #[test]
    fn every_dispatch_label_maps_to_a_surface() {
        // Labels observed in the tree, with the surface each must land in.
        let cases: &[(&str, Phase)] = &[
            ("wo", Phase::AttnWoMv),
            ("gate", Phase::FfnGateMv),
            ("up", Phase::FfnUpMv),
            ("gate_up", Phase::FfnGateUpFused),
            ("down", Phase::FfnDownMv),
            ("gdn_qkv", Phase::GdnQkvMv),
            ("gdn_gate", Phase::GdnGateMv),
            ("gdn_alpha", Phase::GdnAlphaMv),
            ("gdn_alpha_f16", Phase::GdnAlphaMv),
            ("gdn_beta", Phase::GdnBetaMv),
            ("gdn_beta_f16", Phase::GdnBetaMv),
            ("gdn_ssm_out", Phase::SsmOutMv),
            ("output_proj", Phase::HeadMv),
            ("head", Phase::HeadMv),
        ];
        for (label, want) in cases {
            assert_eq!(
                matvec_phase_for_label(label),
                Some(*want),
                "label {label:?}"
            );
        }
        // wq / wk / wv / kv are family-dependent and covered separately.
        for l in ["wq", "wk", "wv", "kv"] {
            assert!(matvec_phase_for_label(l).is_some(), "label {l:?}");
        }
        // A label that is not a matvec surface must map to nothing rather than
        // being silently folded into a neighbour.
        for l in ["attn F16", "ffn HGEMV", "rms_scale", "swiglu", ""] {
            assert_eq!(matvec_phase_for_label(l), None, "label {l:?}");
        }
        // ssm_out is named for its tensor, never folded into a family.
        assert_eq!(matvec_phase_for_label("gdn_ssm_out"), Some(Phase::SsmOutMv));
        assert_eq!(quant_phase_for_label("gdn_ssm_out"), Some(Phase::SsmOutQuant));
        assert_eq!(Phase::SsmOutMv.parent(), Some(Phase::GdnOut));
    }

    /// The qkv family split must key off the attention family, and only qkv --
    /// wo/rope/attention are format-independent and must NOT split.
    #[test]
    fn attn_family_splits_qkv_only() {
        set_attn_family(AttnFamily::Q6kCoupled);
        // With the profiler disabled `set_attn_family` is a no-op, so this test
        // asserts the mapping function directly rather than the global.
        let q6k_view = |l: &str| matvec_phase_for_label(l);
        // wo does not split regardless of family.
        assert_eq!(q6k_view("wo"), Some(Phase::AttnWoMv));
        // The two qkv families are distinct phases with the same parent.
        assert_ne!(Phase::AttnWqQ6k, Phase::AttnWqQ4);
        assert_eq!(Phase::AttnWqQ6k.parent(), Some(Phase::AttnQkv));
        assert_eq!(Phase::AttnWqQ4.parent(), Some(Phase::AttnQkv));
        assert_eq!(Phase::AttnWkvQ6k.parent(), Some(Phase::AttnQkv));
        assert_eq!(Phase::AttnWkvQ4.parent(), Some(Phase::AttnQkv));
    }

    /// Effective bandwidth must be derived only where bytes were declared, and
    /// must be arithmetically correct.
    /// A re-entrant surface guard must be inert, or the split->non-split
    /// tail-calls in the preq8_1 family would count the surface twice.
    #[test]
    fn reentrant_surface_guard_is_inert() {
        // With the profiler disabled `phase_already_open` sees an empty stack,
        // so this asserts the mechanism's logic directly against a synthetic
        // stack rather than relying on a live decode.
        assert!(!phase_already_open(Phase::FfnDownMv));
        let _ = with(|p| {
            p.open.push((Phase::FfnDownMv as u8, 0, 0, 0));
        });
        assert!(
            phase_already_open(Phase::FfnDownMv),
            "an open bracket must be detected"
        );
        assert!(
            !phase_already_open(Phase::FfnGateMv),
            "a different phase must not be reported open"
        );
        // Restore, so no other test sees a dangling open bracket.
        let _ = with(|p| {
            p.open.clear();
        });
        assert!(!phase_already_open(Phase::FfnDownMv));
    }

    #[test]
    fn effective_gbps_is_correct_and_absent_without_bytes() {
        // 1 GB over 1000 us = 1000 GB/s.
        let r = PhaseRow {
            phase: Phase::FfnDownMv,
            calls: 1,
            total_us: 1000.0,
            total_bytes: 1_000_000_000,
        };
        let g = r.effective_gbps().expect("bytes declared");
        assert!((g - 1000.0).abs() < 1e-6, "got {g}");

        // No bytes -> no bandwidth, rather than 0 GB/s.
        let r0 = PhaseRow {
            phase: Phase::FfnSwiglu,
            calls: 1,
            total_us: 10.0,
            total_bytes: 0,
        };
        assert_eq!(r0.effective_gbps(), None);

        // No time -> no bandwidth (avoids a divide-by-zero infinity).
        let rt = PhaseRow {
            phase: Phase::FfnDownMv,
            calls: 1,
            total_us: 0.0,
            total_bytes: 1_000,
        };
        assert_eq!(rt.effective_gbps(), None);
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
                    total_bytes: 0,
                },
                PhaseRow {
                    phase: Phase::FinalNorm,
                    calls: 2,
                    total_us: 40.0,
                    total_bytes: 0,
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
                    total_bytes: if p.is_matvec() { 1_000_000 } else { 0 },
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
    /// Renders a representative level-2 `[PROFILE]` block so the output format
    /// can be inspected without a GPU.
    ///
    /// ```text
    /// cargo test -p lumen-runtime --features cuda --lib \
    ///     cuda::profiler::tests::show_profile_output_format -- --ignored --nocapture
    /// ```
    ///
    /// Shape and byte counts are the real Qwen3.5-9B Q4_0 values; the TIMINGS
    /// are invented and must not be quoted as measurements.
    #[test]
    #[ignore = "documentation aid: run with --ignored --nocapture to see the output format"]
    fn show_profile_output_format() {
        // (phase, calls/token, us/token, bytes/call)
        let shape: &[(Phase, u64, f64, u64)] = &[
            (Phase::Embed, 1, 3.1, 0),
            (Phase::GdnAttn, 24, 3980.0, 0),
            (Phase::FullAttn, 8, 1520.0, 0),
            (Phase::Ffn, 32, 2110.0, 0),
            (Phase::LayerCommit, 32, 96.0, 0),
            (Phase::Head, 1, 640.0, 0),
            (Phase::Argmax, 1, 21.0, 0),
            // full_attn children: 4 layers coupled, 4 native.
            (Phase::AttnQkv, 8, 700.0, 0),
            (Phase::AttnWqQ6k, 4, 300.0, 67_108_864),
            (Phase::AttnWqQ4, 4, 120.0, 18_874_368),
            (Phase::AttnWkvQ6k, 4, 60.0, 16_777_216),
            (Phase::AttnWkvQ4, 4, 30.0, 4_718_592),
            (Phase::AttnQkvQuant, 8, 24.0, 0),
            (Phase::AttnCore, 8, 300.0, 0),
            (Phase::AttnWo, 8, 340.0, 0),
            (Phase::AttnWoMv, 8, 330.0, 9_437_184),
            // ffn children
            (Phase::FfnGateUp, 32, 1290.0, 0),
            (Phase::FfnActQuant, 32, 40.0, 0),
            (Phase::FfnGateMv, 32, 600.0, 28_311_552),
            (Phase::FfnUpMv, 32, 600.0, 28_311_552),
            (Phase::FfnSwiglu, 32, 30.0, 0),
            (Phase::FfnDownResid, 32, 790.0, 0),
            (Phase::FfnDownQuant, 32, 35.0, 0),
            (Phase::FfnDownMv, 32, 700.0, 28_311_552),
            (Phase::FfnResidual, 32, 20.0, 0),
            // gdn children
            (Phase::GdnQkv, 24, 1450.0, 0),
            (Phase::GdnNorm, 24, 20.0, 0),
            (Phase::GdnQkvMv, 24, 700.0, 18_874_368),
            (Phase::GdnGateMv, 24, 380.0, 9_437_184),
            (Phase::GdnAlphaMv, 24, 40.0, 139_264),
            (Phase::GdnBetaMv, 24, 40.0, 139_264),
            (Phase::GdnActQuant, 48, 60.0, 0),
            (Phase::GdnConvRecur, 24, 1900.0, 0),
            (Phase::GdnConv, 24, 300.0, 0),
            (Phase::GdnRecur, 24, 1560.0, 0),
            (Phase::GdnOut, 24, 520.0, 0),
            (Phase::GdnNormGate, 24, 90.0, 0),
            (Phase::SsmOutQuant, 24, 25.0, 0),
            (Phase::SsmOutMv, 24, 390.0, 17_825_792),
            (Phase::GdnGlue, 24, 60.0, 0),
            // head children
            (Phase::FinalNorm, 1, 8.0, 0),
            (Phase::HeadMv, 1, 620.0, 1_080_688_640),
        ];
        let tokens = 128usize;
        let attributed: f64 = shape
            .iter()
            .filter(|(p, _, _, _)| p.depth() == 0)
            .map(|(_, _, us, _)| us)
            .sum();
        let s = Summary {
            tokens,
            rows: shape
                .iter()
                .map(|(p, calls, us, bpc)| PhaseRow {
                    phase: *p,
                    calls: calls * tokens as u64,
                    total_us: us * tokens as f64,
                    total_bytes: bpc * calls * tokens as u64,
                })
                .collect(),
            wall_us_mean: 9120.0,
            wall_us_p50: 9105.0,
            gpu_span_us_mean: 8790.0,
            gpu_span_us_p50: 8781.0,
            attributed_us_mean: attributed,
            attributed_us_p50: attributed,
            health: HealthCounts {
                pool_high_water: 1180,
                ..Default::default()
            },
            health_delta: HealthCounts::default(),
        };
        let mut buf: Vec<u8> = Vec::new();
        write_summary(&mut buf, "gen1", &s);
        print!("{}", String::from_utf8(buf).expect("utf8"));
    }

    #[test]
    fn uncovered_localizes_the_shortfall_per_parent() {
        let s = Summary {
            tokens: 2,
            rows: vec![
                // full_attn 1000/token, children sum to 900 -> 100 uncovered.
                PhaseRow { phase: Phase::FullAttn, calls: 2, total_us: 2000.0, total_bytes: 0 },
                PhaseRow { phase: Phase::AttnQkv, calls: 2, total_us: 1200.0, total_bytes: 0 },
                PhaseRow { phase: Phase::AttnCore, calls: 2, total_us: 600.0, total_bytes: 0 },
                // ffn 500/token, children sum to 500 -> 0 uncovered (tiled).
                PhaseRow { phase: Phase::Ffn, calls: 2, total_us: 1000.0, total_bytes: 0 },
                PhaseRow { phase: Phase::FfnGateUp, calls: 2, total_us: 600.0, total_bytes: 0 },
                PhaseRow { phase: Phase::FfnDownResid, calls: 2, total_us: 400.0, total_bytes: 0 },
                // argmax has no children at all -> None, not Some(0).
                PhaseRow { phase: Phase::Argmax, calls: 2, total_us: 40.0, total_bytes: 0 },
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
            rows: vec![PhaseRow { phase: Phase::Head, calls: 1, total_us: 500.0, total_bytes: 0 }],
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
            rows: vec![PhaseRow { phase: Phase::Embed, calls: 1, total_us: 5.0, total_bytes: 0 }],
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
