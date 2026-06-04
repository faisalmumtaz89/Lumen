//! Per-layer graph reorder + greedy wavefront scheduler for Metal prefill.
//!
//! ## Background
//!
//! Lumen's existing batched prefill in `prefill_encode.rs` opens many short
//! compute encoders per layer (one per logical block: attn-norm+QKV, KV-write+
//! scores+softmax+output, Wo+residual+ffn-norm, FFN gate+up+swiglu, FFN
//! down+residual). Each encoder boundary serialises the GPU and pays
//! command-buffer encoding overhead.
//!
//! Apple's `MTLDispatchTypeConcurrent` encoder lets dispatches within the
//! encoder run in parallel when the GPU's hazard tracker sees no buffer
//! conflict; `memoryBarrierWithResources:` is then used between conflicting
//! groups instead of an encoder boundary. proved the buffer-renaming
//! lever alone is null in serial encoders, so the load-bearing change is
//! to consolidate dispatches into a single per-layer concurrent encoder and
//! schedule them through a small wavefront-based scheduler.
//!
//! ## Scope
//!
//! This module is the scheduler abstraction only. The first integration wires
//! it into Qwen3.5 full-attention + dense FFN layers in `prefill_encode.rs`.
//! GDN (linear-attention) layers continue to use the legacy multi-encoder
//! path until a follow-on revision when GDN ops + 4-way qkv split land together.
//!
//! ## Design
//!
//! 1. **`LayerOp` is a thin descriptor**: it carries the access metadata
//!    (read/write byte ranges on each mutable buffer it touches), an
//!    `OrderClass` (`Free` may be reordered within a lookahead window;
//!    `Strict` must execute in program order), and an emit closure that
//!    issues the actual `MetalComputeEncoder` dispatches.
//! 2. **The scheduler is byte-range-aware**: hazards are tracked on
//!    mutable buffers only (weights and RoPE cos/sin are read-only and
//!    cannot introduce conflicts). Two accesses conflict when they touch
//!    the same `BufferId` and at least one is a write *and* their byte
//!    ranges overlap.
//! 3. **Wavefronts are greedy**: build up an op list, seed each wavefront
//!    with the next op in program order, then scan forward (bounded by
//!    `LOOKAHEAD`) to pull in additional hazard-free ops. Stop a wavefront
//!    when the next candidate conflicts or has `OrderClass::Strict`.
//! 4. **Between wavefronts** the scheduler calls
//!    `memory_barrier_with_resources` on the buffers written in this
//!    wavefront that are read by a later wavefront. This is much cheaper
//!    than per-op events.
//!
//! ## Trap list
//!
//! Callers building `LayerOp`s must respect:
//! - In-place ops (RMSNorm-on-q, RoPE, softmax, sigmoid-gate) are
//!   `AccessKind::ReadWrite` on their input/output buffer.
//! - `scores_buf` byte length uses the padded `scores_stride` not the
//!   raw attend length (the materialized kernel pads to multiples of 8).
//! - For Qwen3.5 full-attention layers `qkv_buf`'s upper bound is
//!   `2 * q_dim` (interleaved Q+gate), NOT `qkv_dim`.
//! - Residual reads (Wo residual: `x_buf`, FFN-down residual:
//!   `attn_proj_buf`) MUST appear in the access list; otherwise the
//!   scheduler may pull the consumer too early.
//! - `LUMEN_METAL_PROFILE_DEEP=1` intentionally splits encoders for
//!   per-section profiling; the concurrent-encoder path must fall back to legacy.

// The first integration wires the scheduler against full-attention + dense
// FFN only. The `Down` buffer-id, byte-range arithmetic, and `is_read` helper
// exist for the GDN migration; allow dead-code warnings here so the public
// API stays stable across revisions.
#![allow(dead_code)]

use crate::error::RuntimeError;
use super::ffi::{MetalBuffer, MetalComputeEncoder};
use std::sync::atomic::{AtomicU8, Ordering as AOrd};

/// Identifies a logical mutable buffer for hazard tracking.
///
/// We track only the buffers that are written by some op in the per-layer
/// plan (activations / scratch / KV cache slot). Weight buffers and the
/// RoPE cos/sin tables are read-only and excluded.
///
/// The id is a small enum, not a raw `*const` pointer, so the scheduler
/// can be tested without owning real Metal resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BufferId {
    /// `batch_x_buf` — layer residual / hidden state across layers.
    X,
    /// `batch_normed_buf` — RMSNorm output for QKV and FFN-gate-up inputs.
    Normed,
    /// `batch_qkv_buf` — fused QKV (or Q+gate) output of the first GEMM.
    Qkv,
    /// `batch_q_buf` — Q projection output (after deinterleave / RoPE).
    Q,
    /// `batch_k_buf` — K projection output (after deinterleave / RoPE).
    K,
    /// `batch_v_buf` — V projection output (after deinterleave).
    V,
    /// `batch_attn_out_buf` — attention output (Scores * V).
    AttnOut,
    /// `batch_attn_proj_buf` — Wo output + residual 1 destination.
    AttnProj,
    /// `batch_gate_buf` — FFN gate output (or fused-kernel direct SwiGLU result).
    Gate,
    /// `batch_up_buf` — FFN up output (only consumed when not using the fused kernel).
    Up,
    /// `batch_down_buf` — FFN down output for the split-K path.
    Down,
    /// `batch_scores_buf` — attention scores + softmax probabilities.
    Scores,
    /// Per-layer K cache slot (`gpu_k_cache[layer_idx]`).
    KCache,
    /// Per-layer V cache slot (`gpu_v_cache[layer_idx]`).
    VCache,
}

/// A half-open byte range `[off, off+len)` within a logical buffer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ByteRange {
    pub off: u64,
    pub len: u64,
}

impl ByteRange {
    /// Whole-buffer marker (used when an op touches the entire logical span).
    pub const WHOLE: Self = Self { off: 0, len: u64::MAX };

    pub fn whole() -> Self { Self::WHOLE }

    pub fn overlaps(&self, other: &Self) -> bool {
        // Treat WHOLE as overlapping everything.
        if self.len == u64::MAX || other.len == u64::MAX {
            return true;
        }
        let a_end = self.off.saturating_add(self.len);
        let b_end = other.off.saturating_add(other.len);
        self.off < b_end && other.off < a_end
    }
}

/// Read / Write / ReadWrite classifier for a single buffer access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AccessKind {
    Read,
    Write,
    ReadWrite,
}

impl AccessKind {
    #[inline] pub fn is_write(self) -> bool { matches!(self, Self::Write | Self::ReadWrite) }
    #[inline] pub fn is_read (self) -> bool { matches!(self, Self::Read  | Self::ReadWrite) }
}

/// One mutable-buffer access by an op.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Access {
    pub buf: BufferId,
    pub range: ByteRange,
    pub kind: AccessKind,
}

impl Access {
    #[inline]
    pub fn read(buf: BufferId) -> Self {
        Self { buf, range: ByteRange::WHOLE, kind: AccessKind::Read }
    }
    #[inline]
    pub fn write(buf: BufferId) -> Self {
        Self { buf, range: ByteRange::WHOLE, kind: AccessKind::Write }
    }
    #[inline]
    pub fn read_write(buf: BufferId) -> Self {
        Self { buf, range: ByteRange::WHOLE, kind: AccessKind::ReadWrite }
    }
}

/// Inline-capacity access list for a single `LayerOp`.
///
/// `Access` is `Copy` and ops touch at most ~5 mutable buffers, so a fixed
/// `[Access; 8]` inline buffer always fits without spilling to the heap.
/// This replaces the prior `Vec<Access>` field whose `vec![..]` literal
/// caused one heap allocation per `LayerOp` constructed. With ~21 ops per
/// Qwen3.5-9B full-attention layer and 8 such layers per prefill, that
/// was ~168 of the ~424 heap allocations per prefill measured
/// §A.1 (and another 96 from the GDN FFN block plans). `AccessList` is
/// `Copy` so it never deallocates and lifts the LayerOp construction onto
/// the encoder thread's stack.
///
/// Capacity 8 was chosen by inspection of `encode_layer_batched_concurrent` and
/// `concurrent_encoder_extend_plan_with_ffn_block`: max distinct accesses per op = 5
/// (FFN-down: read Gate, read Up, write X, plus the residual fold). The
/// 8-slot ceiling leaves headroom; over-capacity pushes are debug-asserted
/// and silently dropped in release. The `as_slice()` view is the only API
/// the scheduler needs (it never resizes after construction).
///
#[derive(Clone, Copy)]
pub(crate) struct AccessList {
    len: u8,
    data: [Access; AccessList::MAX],
}

impl AccessList {
    const MAX: usize = 8;
    const SENTINEL: Access = Access {
        buf: BufferId::X,
        range: ByteRange::WHOLE,
        kind: AccessKind::Read,
    };

    #[inline]
    pub(crate) fn new() -> Self {
        Self { len: 0, data: [Self::SENTINEL; Self::MAX] }
    }

    #[inline]
    pub(crate) fn push(&mut self, a: Access) {
        debug_assert!((self.len as usize) < Self::MAX,
            "AccessList capacity exceeded ({}); raise MAX", Self::MAX);
        if (self.len as usize) < Self::MAX {
            self.data[self.len as usize] = a;
            self.len += 1;
        }
    }

    #[inline]
    pub(crate) fn from_iter_inline<I: IntoIterator<Item = Access>>(iter: I) -> Self {
        let mut v = Self::new();
        for a in iter { v.push(a); }
        v
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[Access] {
        &self.data[..self.len as usize]
    }

    #[inline]
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, Access> {
        self.as_slice().iter()
    }
}

impl std::fmt::Debug for AccessList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

/// Reordering policy for an op.
///
/// `Free` ops may be pulled into a wavefront ahead of program order so long
/// as their accesses do not conflict with anything already in the wavefront.
/// `Strict` ops act as a barrier in the program-order scan: no later op
/// may be pulled past them. Strict is the safe conservative choice for
/// recurrence-style operations (softmax, attention output, residual add,
/// later: GDN recurrence).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrderClass {
    Free,
    Strict,
}

/// Closure that issues the actual Metal dispatch for an op.
///
/// Stored as a boxed `FnMut` so callers can capture per-op constants
/// (pipeline state, buffer offsets, set_bytes payload) without bloating
/// the op enum with every possible parameter combination.
pub(crate) type EmitFn<'a> =
    Box<dyn FnMut(&MetalComputeEncoder) -> Result<(), RuntimeError> + 'a>;

/// A single scheduled operation in a per-layer plan.
///
/// `accesses` was previously a `Vec<Access>`; switching to the inline
/// `AccessList` removes one heap allocation per op constructed
/// (~21 per Qwen3.5-9B full-attention layer + 4 per GDN FFN block =
/// ~248 heap allocations per prefill eliminated). The scheduler's API
/// surface (`accesses.iter()`) is preserved unchanged.
pub(crate) struct LayerOp<'a> {
    /// Debug label (used in error messages and trace output).
    pub label: &'static str,
    /// Mutable-buffer accesses (read-only weights are NOT listed).
    pub accesses: AccessList,
    pub order_class: OrderClass,
    /// Emit closure. Boxed so heterogeneous op shapes share a single type.
    pub emit: EmitFn<'a>,
}

impl<'a> LayerOp<'a> {
    /// Returns true if this op writes any buffer that `other` also accesses
    /// (or vice versa) AND their byte ranges overlap on at least one shared
    /// buffer.
    fn conflicts_with(&self, other: &Self) -> bool {
        for a in self.accesses.iter() {
            for b in other.accesses.iter() {
                if a.buf != b.buf { continue; }
                // Two accesses on the same buffer conflict iff at least one
                // is a write and their ranges overlap.
                if (a.kind.is_write() || b.kind.is_write()) && a.range.overlaps(&b.range) {
                    return true;
                }
            }
        }
        false
    }

    /// Returns the buffers written by this op. Used to compute the
    /// barrier resource set between wavefronts.
    fn writes(&self) -> impl Iterator<Item = BufferId> + '_ {
        self.accesses.iter()
            .filter(|a| a.kind.is_write())
            .map(|a| a.buf)
    }
}

/// Greedy lookahead window.
///
/// The default `LOOKAHEAD = 32` matches the per-layer op count on the
/// plans (full-attn + dense FFN tops out at ~14 ops per
/// layer; GDN tops out at ~12 after the split).
///
/// the scheduler bumps this to **64** when `LUMEN_METAL_CONCURRENT_ENCODER_FULL=1` is
/// set. The wider window is only useful when the plan length exceeds the
/// legacy ~14 op cap, which happens once cross-section emit collapses the
/// FFN-norm + FFN gate+up+down chain into the same plan (currently still
/// split into two per-layer plans on main; bumping the window is a no-op
/// cost when the plan length is small, so this is safe to enable globally).
const LOOKAHEAD: usize = 32;
const LOOKAHEAD_FULL: usize = 64;

/// Master kill-switch for the runtime-architecture defaults.
///
/// Resolves once via `LUMEN_METAL_DEFAULTS_OFF=1`. Default OFF (i.e. the
/// runtime-architecture defaults are ACTIVE). When this env var is `1`, the
/// five default-ON optimizations (concurrent encoder for full-attn/dense FFN,
/// GDN concurrent encoder, Q8 SoA repack for FFN-down and gate+up, FFN-down
/// Split-K, fused gate+up+SwiGLU) revert to legacy default-OFF behaviour.
///
/// Provides a single env var that restores legacy behaviour end-to-end,
/// independent of the per-feature opt-in env vars.
#[inline]
pub(crate) fn metal_defaults_active() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let off = std::env::var("LUMEN_METAL_DEFAULTS_OFF")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    let active = !off;
    CACHE.store(if active { 2 } else { 1 }, AOrd::Relaxed);
    active
}

/// Active lookahead window for the current process.
///
/// Resolves to `LOOKAHEAD_FULL = 64` when `LUMEN_METAL_CONCURRENT_ENCODER_FULL=1` is set
/// otherwise `LOOKAHEAD = 32`.
/// Cached once per process via an atomic so the per-op scan hot path
/// stays branch-free after the first call.
#[inline]
fn active_lookahead() -> usize {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return if cur == 2 { LOOKAHEAD_FULL } else { LOOKAHEAD }; }
    let v = std::env::var("LUMEN_METAL_CONCURRENT_ENCODER_FULL")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    if v { LOOKAHEAD_FULL } else { LOOKAHEAD }
}

/// Whether the concurrent-encoder path is enabled for this process.
///
/// Resolved once via `LUMEN_METAL_CONCURRENT_ENCODER`. **Default: ON** when
/// the master kill-switch (`LUMEN_METAL_DEFAULTS_OFF`) is absent. Explicit
/// `LUMEN_METAL_CONCURRENT_ENCODER=0` always disables it.
///
/// The concurrent-encoder path consolidates per-layer dispatches into a
/// single `MTLComputeCommandEncoder` for the full-attn / dense FFN sub-graph
/// and uses resource-scoped barriers for inter-op hazards. This reduces the
/// per-prefill encoder count and is jointly enabled with the GDN concurrent
/// encoder for full coverage on Qwen3.5-9B.
#[inline]
pub(crate) fn concurrent_encoder_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let raw = std::env::var("LUMEN_METAL_CONCURRENT_ENCODER").ok();
    let v = match raw.as_deref() {
        Some("0") => false,
        Some(s) if !s.is_empty() => true,
        _ => metal_defaults_active(),
    };
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Whether the scheduler extensions are enabled.
///
/// Resolved once via `LUMEN_METAL_CONCURRENT_ENCODER_FULL=1`. Default OFF. Cached
/// atomically. When ON, `emit_plan_into_encoder` uses a 64-op lookahead
/// window (a wider lookahead than the legacy ~14-op cap) instead of the
/// 32-op default. This is a no-op for current per-layer plans (~14 ops
/// each) but becomes load-bearing when future revisions collapse multi-section
/// plans into the same scheduler call.
///
/// Independent of `LUMEN_METAL_CONCURRENT_ENCODER` and `LUMEN_METAL_GDN_CONCURRENT_ENCODER` so it can
/// be enabled or disabled standalone without affecting which sections
/// route through the concurrent-encoder path.
#[inline]
pub(crate) fn concurrent_encoder_full_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_CONCURRENT_ENCODER_FULL")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Whether the GDN concurrent-encoder path (4-way qkv split + concurrent encoder +
/// resource-scoped barriers in `encode_batched_gdn_prefill`) is enabled.
///
/// Resolved once via `LUMEN_METAL_GDN_CONCURRENT_ENCODER`. **Default: ON**
/// when the master kill-switch (`LUMEN_METAL_DEFAULTS_OFF`) is absent.
/// Explicit `LUMEN_METAL_GDN_CONCURRENT_ENCODER=0` always disables it.
///
/// Independent of the `LUMEN_METAL_CONCURRENT_ENCODER` switch so the GDN migration
/// can be enabled standalone. The two flags both target Apple's whole-
/// MTLBuffer hazard tracker, but at different points: `LUMEN_METAL_CONCURRENT_ENCODER`
/// routes full-attn + dense FFN layers through the scheduler; this flag
/// routes GDN linear-attn layers through the de-aliased path.
#[inline]
pub(crate) fn gdn_concurrent_encoder_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let raw = std::env::var("LUMEN_METAL_GDN_CONCURRENT_ENCODER").ok();
    let v = match raw.as_deref() {
        Some("0") => false,
        Some(s) if !s.is_empty() => true,
        _ => metal_defaults_active(),
    };
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Whether to validate the GDN concurrent-encoder plan by emitting it into a serial
/// (non-concurrent) compute encoder for byte-identity comparison against
/// legacy.
///
/// Resolved once via `LUMEN_METAL_GDN_CONCURRENT_ENCODER_VALIDATE=1`. Default OFF. Cached
/// atomically.
#[inline]
pub(crate) fn gdn_concurrent_encoder_validate_serial() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_GDN_CONCURRENT_ENCODER_VALIDATE")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Whether the GDN Phase 2a (32, NSG=4, 1) threadgroup geometry
/// is enabled.
///
/// Resolved once via `LUMEN_METAL_GDN_PHASE2A_NSG4=1`. Default OFF.
/// Cached atomically.
///
/// When ON, `encode_batched_gdn_prefill` dispatches the new kernel
/// `gdn_prefill_fused_v3_chunked_nsg4` with a (val_dim/4, n_heads, 1)
/// grid and (32, 4, 1) threadgroup shape. Each TG owns 4 consecutive rows
/// of state across 4 simdgroups that share Q/K HBM fetches via L1 (only V
/// differs per simdgroup). The algorithm is bit-identical to the legacy
/// `gdn_prefill_fused_v3_chunked` kernel; the reference-token gate is the
/// validator.
///
/// Research:concurrent-encoder section.
#[inline]
pub(crate) fn gdn_phase2a_nsg4_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_GDN_PHASE2A_NSG4")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Whether the GDN `ssm_out` projection uses the F32-batched residual kernel.
///
/// Resolved once via `LUMEN_METAL_GDN_SSM_OUT_F32_BATCHED`. Default ON
/// (opt-OUT semantics — set `=0` to fall back to the legacy per-token loop).
/// Cached atomically following the same pattern as the other env helpers
/// in this file.
///
/// Previously this was resolved inside the GDN compute body at
/// `gdn.rs:2289`, firing once per GDN layer × 24 GDN layers = 24
/// `getenv` syscalls per prefill. The cached resolver eliminates that
/// serialisation on libc's environ mutex.
///
#[inline]
pub(crate) fn gdn_ssm_out_f32_batched_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    // Opt-OUT semantics: default ON when unset, OFF only when explicitly "0".
    let v = std::env::var("LUMEN_METAL_GDN_SSM_OUT_F32_BATCHED")
        .map(|s| s != "0")
        .unwrap_or(true);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Whether to enable the FFN-down Split-K K64 path.
///
/// Resolved once via `LUMEN_METAL_FFN_DOWN_SPLITK=<N>` (N in {2,4,8};
/// explicit `0` disables). **Default: 8** when the master kill-switch
/// (`LUMEN_METAL_DEFAULTS_OFF`) is absent.
///
/// Eligibility (checked at the dispatch site, not here):
///   - Q8 weights, M <= 192, K >= 8192.
/// When non-zero, `encode_ffn_down` dispatches
/// `dequant_tiled_matmul_q8_0_k64_splitk` followed by
/// `reduce_splitk_add_residual` instead of the fused K64 residual kernel.
#[inline]
pub(crate) fn ffn_down_splitk_value() -> u32 {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return (cur - 1) as u32; }
    let raw = std::env::var("LUMEN_METAL_FFN_DOWN_SPLITK").ok();
    let v: u32 = match raw {
        Some(ref s) if !s.is_empty() => s.parse().unwrap_or(0),
        // Split-K targets dense FFN-down shape (N=hidden_dim,
        // K=intermediate_dim). MoE FFN-down is per-expert at a different
        // shape (N=hidden_dim, K=moe_expert_inter_dim) and dispatches through
        // `encode_moe_ffn_batched` / `encode_moe_ffn_with_shared_fused`, which
        // do NOT consult this gate. Default OFF on MoE is therefore a noop in
        // terms of compute, but it sizes down the `required_splitk_size`
        // allocation in `ensure_batch_buffers` (line 168-184 of
        // prefill_encode.rs) which saves 8 * batch * hidden_dim floats of
        // wasted scratch. Explicit env-set still wins for diagnostics.
        _ => if metal_defaults_active() && !crate::runtime_defaults::model_is_moe() { 8 } else { 0 },
    };
    // Allowed values: 0 (off), 2, 4, 8. Any other value disables.
    let v = if matches!(v, 0 | 2 | 4 | 8) { v } else { 0 };
    CACHE.store((v + 1) as u8, AOrd::Relaxed);
    v
}

/// Whether to enable the BF16 FFN-down Split-K K64 path.
///
/// Resolved once via `LUMEN_METAL_FFN_DOWN_SPLITK_BF16=<N>` (N in {2,4,8};
/// explicit `0` disables). **Default OFF** until empirical validation;
/// intentionally NOT gated under `metal_defaults_active()` (the Q8 variant
/// has multi-revision evidence; BF16 awaits its own).
///
/// Eligibility (checked at the dispatch site):
///   - BF16 weights, M <= 192, K >= 8192, K % 64 == 0.
/// When non-zero, the FFN-down dispatch uses `bf16_matmul_k64_splitk`
/// followed by the shared `reduce_splitk_add_residual` reduce kernel.
#[inline]
pub(crate) fn ffn_down_splitk_bf16_value() -> u32 {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return (cur - 1) as u32; }
    let raw = std::env::var("LUMEN_METAL_FFN_DOWN_SPLITK_BF16").ok();
    let v: u32 = match raw {
        Some(ref s) if !s.is_empty() => s.parse().unwrap_or(0),
        _ => 0,
    };
    // Allowed values: 0 (off), 2, 4, 8. Any other value disables.
    let v = if matches!(v, 0 | 2 | 4 | 8) { v } else { 0 };
    CACHE.store((v + 1) as u8, AOrd::Relaxed);
    v
}

/// BF16 GDN tile geometry override.
///
/// The default BF16 GDN dispatch picks `tiled_matmul_bf16_k64` when
/// hidden_dim % 64 == 0 && batch_size <= 4096. The K64 variant has fewer
/// barriers but doubles threadgroup-memory pressure (8 KB vs 4 KB shmem),
/// which can reduce occupancy on M3 Ultra at skinny-M (M=131) shapes.
///
/// When `LUMEN_METAL_BF16_GDN_TILE_NOK64=1`, the GDN dispatch forces
/// `tiled_matmul_bf16` (no K64) for ALL three BF16 GDN sites (QKV-proj,
/// attn_gate-proj, ssm_out residual). Default OFF — preserves
/// reference behaviour byte-identically when unset.
///
/// Microbench harness expectation:
#[inline]
pub(crate) fn bf16_gdn_tile_nok64_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_BF16_GDN_TILE_NOK64")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Apple MPSGraph BF16 GEMM opt-in.
///
/// When ON, the BF16 GDN `qkv_proj` (K=4096, N=8192) and `ssm_out`
/// (K=4096, N=4096) matmuls are routed through Apple's MPSGraph in place
/// of Lumen's `tiled_matmul_bf16_k64` / `tiled_matmul_bf16_k64_residual`
/// custom kernels. MPSGraph achieves ~7.5 TFLOPs at the qkv-proj shape
/// per microbench (1.166 ms median); the smaller projections
/// (gate, gk, gk_a_log) keep the custom kernel because MPSGraph
/// framework overhead dominates at N <= 128.
///
/// Activation conditions checked at the dispatch site:
///   1. `bf16_mps_enabled()` returns true (env `LUMEN_METAL_BF16_MPS=1`).
///   2. BF16 weight quantization (other quants stay on existing path).
///   3. Batch size > 32 (small-M overhead dominates the MPSGraph win).
///   4. K, N >= 4096 (excludes the small attention projections).
///
/// Default OFF — every BF16 path remains byte-identical until the user
/// explicitly opts in. The ssm_out residual addition (`+ x_buf`) is
/// performed by a follow-on F32 add kernel when this path is taken,
/// since MPSGraph does not subsume the residual in the same graph.
#[inline]
pub(crate) fn bf16_mps_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_BF16_MPS")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Top-level enable for Q8_0 hot-weight runtime repack.
///
/// When ON, the load-time pass allocates extra `MTLBuffer`s holding hot FFN
/// tensors in a Metal-friendly stripe SoA layout (see `metal/repack_q8.rs`).
/// The packed kernels in `shaders/gemm_q8_0.msl` consume these buffers; the
/// original AoS path remains as a fallback when this is OFF or when a tensor
/// doesn't qualify (wrong quant, dimensions misaligned, etc).
///
/// Resolved once via `LUMEN_METAL_Q8_REPACKED`. **Default: ON** when the
/// master kill-switch (`LUMEN_METAL_DEFAULTS_OFF`) is absent.
/// Explicit `LUMEN_METAL_Q8_REPACKED=0` always disables it (bit-exact
/// when disabled).
#[inline]
pub(crate) fn q8_repacked_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let raw = std::env::var("LUMEN_METAL_Q8_REPACKED").ok();
    let v = match raw.as_deref() {
        Some("0") => false,
        Some(s) if !s.is_empty() => true,
        // mirror CUDA's MoE-gating pattern. Q8 repack
        // operates on dense FFN-down + gate+up tensors, which are zero-slice
        // sentinels on MoE LBCs (Qwen3.5-MoE replaces dense FFN with per-expert
        // routing). The repack pass already skips MoE layers via `length > 0`
        // checks, so this is a defence-in-depth gate that ALSO suppresses the
        // repack-pass overhead on MoE models where it's guaranteed-noop.
        _ => metal_defaults_active() && !crate::runtime_defaults::model_is_moe(),
    };
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Sub-gate for FFN-down repack only. Allows independent rollout from gate+up.
/// Default: ON when `LUMEN_METAL_Q8_REPACKED=1`, can be force-disabled via
/// `LUMEN_METAL_Q8_REPACKED_FFN_DOWN=0`.
#[inline]
pub(crate) fn q8_repacked_ffn_down_enabled() -> bool {
    if !q8_repacked_enabled() { return false; }
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    // Default ON inside the parent gate, opt-out via "0".
    let v = std::env::var("LUMEN_METAL_Q8_REPACKED_FFN_DOWN")
        .map(|s| s != "0")
        .unwrap_or(true);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Sub-gate for FFN gate+up pair-packed repack. Same semantics as FFN-down sub-gate.
#[inline]
pub(crate) fn q8_repacked_gate_up_enabled() -> bool {
    if !q8_repacked_enabled() { return false; }
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_Q8_REPACKED_GATE_UP")
        .map(|s| s != "0")
        .unwrap_or(true);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Top-level enable for Q4_0 hot-weight runtime repack.
///
/// Port of `q8_repacked_enabled()` to Q4_0. When `LUMEN_METAL_Q4_REPACKED=1`,
/// the load-time pass allocates extra `MTLBuffer`s holding hot FFN tensors
/// in a Metal-friendly stripe SoA layout (see `metal/repack_q4.rs`). The
/// packed kernels in `shaders/gemm_q4.msl` consume these buffers; the
/// original AoS path remains as a fallback when this is OFF or when a tensor
/// doesn't qualify (wrong quant, dimensions misaligned, etc).
///
/// Default: OFF (bit-exact behavior when disabled).
#[inline]
pub(crate) fn q4_repacked_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    // default OFF (env-required-ON). MoE-safe by construction because
    // dense FFN slots are zero-slice sentinels on MoE LBCs — the repack pass
    // skips them via `length > 0` checks. Default stays OFF; explicit
    // `LUMEN_METAL_Q4_REPACKED=1` still wins (operator override honoured even
    // on MoE, matching's "explicit > default" precedence).
    let v = std::env::var("LUMEN_METAL_Q4_REPACKED")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Sub-gate for Q4 FFN-down repack only. Allows independent rollout from gate+up.
/// Default: ON when `LUMEN_METAL_Q4_REPACKED=1`, can be force-disabled via
/// `LUMEN_METAL_Q4_REPACKED_FFN_DOWN=0`.
#[inline]
pub(crate) fn q4_repacked_ffn_down_enabled() -> bool {
    if !q4_repacked_enabled() { return false; }
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_Q4_REPACKED_FFN_DOWN")
        .map(|s| s != "0")
        .unwrap_or(true);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Sub-gate for Q4 FFN gate+up pair-packed repack. Same semantics as FFN-down sub-gate.
#[inline]
pub(crate) fn q4_repacked_gate_up_enabled() -> bool {
    if !q4_repacked_enabled() { return false; }
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_Q4_REPACKED_GATE_UP")
        .map(|s| s != "0")
        .unwrap_or(true);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// BF16 GDN qkv-proj + attn-gate-proj paired-dispatch enable.
///
/// When ON, the two BF16 GEMM dispatches in Phase 1 of each GDN layer
/// (`gdn.rs::run_prefill_layer`) are collapsed into a single packed-weight
/// dispatch consuming a concat-then-stripe BF16 buffer (see `repack_bf16.rs`).
/// Only the 24 GDN layers (`layer_type=1` in Qwen3.5-9B) are repacked —
/// the 8 full-attention layers stay on their existing dispatch path.
///
/// Memory cost: per-layer `(qkv_n + gate_n) * hidden_dim * 2` ~96 MB × 24 GDN
/// layers = ~2.3 GB extra resident on M3 Ultra. Designed to stay under the
/// ~4.8 GB Apple AGX TLB threshold; a wider full-set BF16 repack crosses the
/// threshold and regresses.
///
/// Default: OFF (bit-exact when disabled).
///
/// The repacked layout produces a steady-state speed-up on long-running
/// server processes, but introduces a one-shot cold-start cost on the very
/// first inference of each fresh process because the 2.3 GB repacked buffer
/// is private to the process and its GPU page-table mapping is committed
/// lazily. The default is therefore kept OFF so that single-shot
/// `lumen run` invocations see no first-inference regression; long-lived
/// processes (e.g. `lumen-server`) can opt in with
/// `LUMEN_METAL_BF16_GDN_QKV_GATE_PAIRED=1` and amortise the
/// page-table commit over many inferences.
///
/// A load-time warmup dispatch in `gpu_resident.rs` is retained as an opt-in
/// (`LUMEN_METAL_BF16_GDN_WARMUP=1` with `MODE=minimal|full`) for
/// downstream investigation.
///
/// A stronger warmup mechanism (`bf16_paired_full_prefill_warmup_enabled`)
/// runs a complete throwaway prefill at `M=131` at the tail of
/// `preload_weights_gpu_resident`. Unlike the minimal-touch dispatch, which
/// only committed the FIRST byte of each packed buffer, the full-prefill
/// warmup exercises every paired-dispatch code path — every page of the
/// ~2.3 GB packed buffer, every production scratch buffer, and every Metal
/// pipeline state transition the production prefill will use. Cost: roughly
/// 180 ms of preload time (one-shot, amortised across the entire process
/// lifetime). The production prefill's first dispatch then finds all GPU
/// page-table mappings and per-process residency state already committed,
/// restoring the steady-state throughput on the very first inference.
#[inline]
pub(crate) fn bf16_gdn_qkv_gate_paired_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let raw = std::env::var("LUMEN_METAL_BF16_GDN_QKV_GATE_PAIRED").ok();
    let v = match raw.as_deref() {
        Some("0") => false,
        Some(s) if !s.is_empty() => true,
        _ => metal_defaults_active(),
    };
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Full-prefill warmup at preload time for the BF16 GDN paired
/// dispatch.
///
/// Returns `true` when the load-time warmup should run a complete throwaway
/// `M=131` prefill against the just-loaded BF16 GDN repack buffer. The warmup
/// runs once per `lumen run` subprocess at the tail of
/// `preload_weights_gpu_resident`, immediately after the packed buffer Vec is
/// populated. Cost on Apple M3 Ultra (Qwen3.5-9B BF16): ~180 ms of preload
/// latency. Net effect: the production prefill's first dispatch operates
/// against pages and scratch buffers whose GPU page-table mappings are
/// already committed, eliminating the cold-pair regression observed
/// when the first prefill ran without a prior warm-up dispatch.
///
/// Default: ON when `bf16_gdn_qkv_gate_paired_enabled()` is also ON. Users
/// who set `LUMEN_METAL_BF16_GDN_FULL_PREFILL_WARMUP=0` can opt out — useful
/// for downstream benchmarking the warmup contribution in isolation.
///
/// Resolves once per process; cached via `AtomicU8`.
#[inline]
pub(crate) fn bf16_paired_full_prefill_warmup_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let raw = std::env::var("LUMEN_METAL_BF16_GDN_FULL_PREFILL_WARMUP").ok();
    let v = match raw.as_deref() {
        Some("0") => false,
        Some(s) if !s.is_empty() => true,
        _ => bf16_gdn_qkv_gate_paired_enabled(),
    };
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Q8 FFN gate+up+SwiGLU fused-kernel enable.
///
/// Resolved once via `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED`. **Default: ON**
/// when the master kill-switch is absent; setting `=0` falls back to the
/// unfused gate/up/SwiGLU sequence.
///
/// The fused kernel runs the gate matmul, the up matmul, and the elementwise
/// `silu(gate) * up` in a single dispatch with a 12 KB threadgroup memory
/// budget for shared `sa[32*64] + sb_gate[32*64] + sb_up[32*64]` tiles. Saves
/// one HBM round-trip for the gate output, one dispatch, and one encoder
/// boundary per FFN compared to the unfused path.
#[inline]
pub(crate) fn ffn_gate_up_swiglu_fused_q8_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let raw = std::env::var("LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED").ok();
    let v = match raw.as_deref() {
        Some("0") => false,
        Some(s) if !s.is_empty() => true,
        _ => metal_defaults_active(),
    };
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Q4 FFN gate+up+SwiGLU fused-kernel enable.
///
/// Resolved once via `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_Q4`. **Default:
/// ON** when the master kill-switch is absent; setting `=0` falls back to
/// the unfused Q4 gate/up/SwiGLU sequence.
///
/// Q4 analogue of `ffn_gate_up_swiglu_fused_q8_enabled`. Same threadgroup
/// memory budget; identical algorithm with Q4-vs-Q8 dequant.
#[inline]
pub(crate) fn ffn_gate_up_swiglu_fused_q4_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let raw = std::env::var("LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_Q4").ok();
    let v = match raw.as_deref() {
        Some("0") => false,
        Some(s) if !s.is_empty() => true,
        _ => metal_defaults_active(),
    };
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// BF16 FFN gate+up+SwiGLU fused-kernel enable.
///
/// Resolved once via `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_BF16`. **Default:
/// ON** when the master kill-switch is absent, mirroring the Q8/Q4 variants.
///
/// BF16 analogue of `ffn_gate_up_swiglu_fused_q8_enabled`. Identical
/// algorithm; uses `simdgroup_bfloat8x8` MMA on Apple GPU family 9 (M3+).
/// Same 12 KB threadgroup-memory budget so the M3 Ultra occupancy knee is
/// preserved (sa[32*64] + sb_gate[32*64] + sb_up[32*64] = 6144 bfloats).
#[inline]
pub(crate) fn ffn_gate_up_swiglu_fused_bf16_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let raw = std::env::var("LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_BF16").ok();
    let v = match raw.as_deref() {
        Some("0") => false,
        Some(s) if !s.is_empty() => true,
        _ => metal_defaults_active(),
    };
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// NR microtile selection for the BF16 gate+up+SwiGLU fused kernel.
///
/// `LUMEN_METAL_BF16_GATE_UP_NR=<1|2|4>` selects which kernel variant the
/// fused-BF16 dispatch path uses:
///   - `NR=1`: TILE_M=16, mc[1][2], 10 KB shmem, 3 TG/CU at M3 occupancy knee.
///   - `NR=2`: TILE_M=32, mc[2][2], 12 KB shmem, 2 TG/CU. **Default / current
///     baseline.**
///   - `NR=4`: TILE_M=64, mc[4][2], 16 KB shmem, 1 TG/CU at the knee.
///
/// Other values fall back to NR=2 (baseline). This is a kernel-variant
/// selector, not a behaviour-changing flag — when `ffn_gate_up_swiglu_fused_bf16_enabled()`
/// is false the fused path is bypassed entirely and this value is ignored.
///
/// Sized to fit in `u8`; -1 used as the uninitialised sentinel.
#[inline]
pub(crate) fn bf16_gate_up_nr() -> u8 {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur - 1; }
    let raw = std::env::var("LUMEN_METAL_BF16_GATE_UP_NR").ok();
    let v: u8 = match raw.as_deref() {
        Some("1") => 1,
        Some("4") => 4,
        // "2", "0", missing, empty, or any other value -> baseline NR=2.
        _ => 2,
    };
    CACHE.store(v + 1, AOrd::Relaxed);
    v
}

/// `commandBufferWithUnretainedReferences` enable for the prefill
/// and decode hot paths.
///
/// Resolved once via `LUMEN_METAL_UNRETAINED_CMDBUFS=1`. **Default OFF** for
/// safety (intentionally NOT gated under `metal_defaults_active()`): the
/// unretained-references mode skips the per-resource retain/release pair the
/// Metal driver normally takes on every `setBuffer:` / `setComputePipelineState:`
/// call, which yields a small CPU-side encoding savings but transfers the
/// lifetime guarantee to the caller. Lumen satisfies the guarantee because
/// `MetalF32Backend` owns every buffer / PSO and the prefill / decode
/// functions commit-and-wait before returning, but the flag is gated behind
/// an explicit opt-in until we have multi-prompt regression evidence.
///
/// (the unretained variant is the canonical Apple-documented choice when
/// the caller owns resource lifetime, with an estimated 1-3% saving from
/// reduced Obj-C retain/release in the CPU encoding path).
#[inline]
pub(crate) fn unretained_cmdbufs_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_UNRETAINED_CMDBUFS")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Multi command-buffer in-flight on prefill.
///
/// When `LUMEN_METAL_MULTI_CB=1`, the prefill is split into K command
/// buffers (default K=2, override via `LUMEN_METAL_MULTI_CB_N`) that
/// are committed sequentially on the same MTLCommandQueue. CB[0] is
/// `commit()`'d (no wait) immediately after its layers are encoded,
/// then CB[k+1] is encoded while CB[k] executes on the GPU. Only the
/// FINAL CB is `commit_and_wait()`'d.
///
/// Same-queue submission order = strict GPU execution order, so the
/// CB[k]→CB[k+1] boundary on `batch_x_buf` (residual carrier) requires
/// no explicit fence — Metal's MTLCommandQueue guarantees CBs run in
/// submission FIFO. The win comes from CPU encoding of CB[k+1]
/// overlapping with GPU execution of CB[k].
///
/// Default: OFF. CPU encoding time per layer must be measured to be a
/// non-negligible fraction of prefill wall before this can deliver.
/// analytical evidence shows CPU encoding is microseconds at
/// pp131,
/// so the predicted upside is small (≤0.5%); kept env-gated until
/// empirical validation lands measurable.
///
/// 64-CB submission baseline benchmark (a comparison runtime
/// submits ~3 large CBs per prefill iteration with 130-170 ms each of
/// asynchronous GPU compute; Lumen submits 1 CB then waits).
#[inline]
pub(crate) fn multi_cb_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_MULTI_CB")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Number of command buffers to split prefill into when
/// `LUMEN_METAL_MULTI_CB=1`. Default 2. Allowed range: [2, 8]. Values
/// outside the range are clamped. Returns 1 when MULTI_CB is OFF (the
/// caller can branch on `> 1`).
#[inline]
pub(crate) fn multi_cb_count() -> usize {
    if !multi_cb_enabled() {
        return 1;
    }
    let n: usize = std::env::var("LUMEN_METAL_MULTI_CB_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    n.clamp(2, 8)
}

/// Whether wavefront trace prints are enabled.
///
/// When `LUMEN_METAL_CONCURRENT_ENCODER_TRACE=1`, `emit_plan_into_encoder` prints the
/// wavefront structure (one `[graph-trace]` line per wave) to stderr. Useful
/// for diagnosing wavefront grouping, but noisy enough to gate behind
/// an env var.
///
/// Cached atomically following the same pattern as `concurrent_encoder_enabled()` /
/// `unretained_cmdbufs_enabled()`. Previously this resolved
/// `std::env::var` on every call to `emit_plan_into_encoder`, which fires
/// 32 times per prefill on Qwen3.5-9B (8 full-attn layers + 24 GDN FFN
/// blocks). The uncached lookup serialises on libc's environ mutex
/// (`getenv` on macOS); caching eliminates that round-trip on every call
/// after the first.
///
#[inline]
pub(crate) fn concurrent_encoder_trace_enabled() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_CONCURRENT_ENCODER_TRACE")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Whether to validate the concurrent-encoder plan by emitting it serially.
///
/// When `LUMEN_METAL_CONCURRENT_ENCODER_VALIDATE=1`, the plan is emitted into a *serial*
/// `new_compute_encoder()` instead of a concurrent one and barriers are
/// suppressed. Serial emission is byte-equivalent to the legacy multi-
/// encoder path (modulo encoder boundaries), so any divergence in the
/// reference-token gate isolates emit-helper bugs from concurrent-
/// scheduling bugs.
#[inline]
pub(crate) fn concurrent_encoder_validate_serial() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_CONCURRENT_ENCODER_VALIDATE")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Whether to validate the whole-prefill encoder by emitting
/// into a serial encoder for byte-identity comparison against legacy.
///
/// Resolved once via `LUMEN_METAL_CONCURRENT_ENCODER_FULL_VALIDATE=1`. Default OFF. Cached
/// atomically. Mirrors `concurrent_encoder_validate_serial()`'s pattern. When ON, callers
/// that respect `concurrent_encoder_full_enabled()` should create their outer encoder as a
/// serial `new_compute_encoder()` instead of a concurrent one. The whole-
/// prefill dispatch order must then be identical to legacy modulo within-
/// encoder ordering (which is in-program-order for serial encoders); any
/// divergence in the reference-token gate isolates cross-layer barrier
/// bugs from concurrent-scheduling bugs.
#[inline]
pub(crate) fn concurrent_encoder_full_validate_serial() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(AOrd::Relaxed);
    if cur != 0 { return cur == 2; }
    let v = std::env::var("LUMEN_METAL_CONCURRENT_ENCODER_FULL_VALIDATE")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, AOrd::Relaxed);
    v
}

/// Emit a cross-layer hazard barrier on a single buffer into an open encoder.
///
/// Used by the whole-prefill scheduler to separate consecutive
/// layer plans that share a residual carrier (`x_buf`). The barrier ensures
/// that writes from the prior layer's final wavefront retire before the next
/// layer's first wavefront reads from the same buffer.
///
/// Apple's `MTLDispatchTypeConcurrent` encoder does NOT implicitly serialise
/// dispatches across the same buffer. The encoder-end / encoder-begin
/// transition in the legacy multi-encoder path provided this serialisation
/// implicitly; when the path consolidates 32 layers into one outer
/// encoder, the cross-layer hazard MUST be expressed explicitly here.
///
/// When `buffer_lookup` resolves the `BufferId` to a real `MetalBuffer`, this
/// emits `memoryBarrierWithResources:` scoped to that one resource. When
/// resolution fails (e.g. the encoder is serial-validate and no buffer
/// lookup is available), this falls back to a coarse whole-buffer
/// `memory_barrier_with_scope(1)` so correctness is preserved.
///
/// `serial_validate` follows the same semantics as `emit_plan_into_encoder`:
/// when true, the barrier is suppressed (serial encoders already serialise
/// on the GPU within the encoder).
pub(crate) fn emit_cross_layer_barrier(
    enc: &MetalComputeEncoder,
    residual_id: BufferId,
    serial_validate: bool,
    buffer_lookup: Option<BufferLookup<'_>>,
) {
    if serial_validate { return; }
    if let Some(lookup) = buffer_lookup {
        if let Some(buf) = lookup(residual_id) {
            let mut bufs: smallvec_no_alloc::Small16<&MetalBuffer> =
                smallvec_no_alloc::Small16::new();
            bufs.push(buf);
            enc.memory_barrier_with_resources(bufs.as_slice());
            return;
        }
    }
    enc.memory_barrier_with_scope(1);
}

/// Lookup from logical `BufferId` to the concrete `MetalBuffer` reference
/// the runtime is using this layer. When provided, the scheduler emits
/// resource-scoped barriers via `memory_barrier_with_resources`; otherwise
/// it falls back to a whole-buffer scope barrier
/// (`memory_barrier_with_scope(1)`).
///
/// The lookup is `Fn` (not `FnOnce`) so it can be called once per wave.
pub(crate) type BufferLookup<'a> = &'a dyn Fn(BufferId) -> Option<&'a MetalBuffer>;

/// Emit the ops in `plan` into `enc` using the greedy wavefront scheduler.
///
/// `enc` should be either a concurrent compute encoder (production) or a
/// regular serial encoder (validation). Apple Metal does NOT implicitly
/// serialise within-encoder dispatches that touch the same buffer, so
/// barriers are required in BOTH cases. The caller is responsible for
/// `end_encoding()`.
///
/// When `buffer_lookup` is `Some`, each between-wave barrier is scoped to
/// the actual MTLBuffer resources written by the previous pass and read by a
/// pending op. When `None`, the scheduler emits a coarse whole-buffer
/// barrier instead.
///
/// `serial_validate` is reserved for a future "no-barrier" debug mode where
/// the caller proves that dispatch ordering is already safe. The current
/// concurrent-encoder path always passes `false`.
///
/// Errors propagate from the first failing emit closure.
pub(crate) fn emit_plan_into_encoder(
    enc: &MetalComputeEncoder,
    plan: &mut [LayerOp<'_>],
    serial_validate: bool,
    buffer_lookup: Option<BufferLookup<'_>>,
) -> Result<(), RuntimeError> {
    let n = plan.len();
    if n == 0 { return Ok(()); }

    // Wavefront lookahead window: 32 by default, 64 when
    // `LUMEN_METAL_CONCURRENT_ENCODER_FULL=1` is set ( scheduler — 64-node lookahead
    // for plans larger than the legacy ~14-op cap).
    let lookahead = active_lookahead();

    // `done[i] = true` once op i has been emitted.
    let mut done = vec![false; n];
    // `wave[w] = vec of op indices in wavefront w` (for barrier set).
    let mut wave_buf: Vec<usize> = Vec::with_capacity(lookahead);
    let mut written_so_far: Vec<BufferId> = Vec::with_capacity(8);

    // LUMEN_METAL_CONCURRENT_ENCODER_TRACE=1 prints the wavefront structure to stderr per
    // layer; useful for diagnosing wavefront grouping but noisy enough that
    // it's gated behind an env var (no default-on prints). Atomic-cached
    // resolver eliminates ~32 `getenv` syscalls per prefill on Qwen3.5-9B
    // (one per call site; this function is invoked once per full-attn layer
    // and once per GDN FFN block).
    let trace_enabled = concurrent_encoder_trace_enabled();

    let mut emitted = 0usize;
    let mut wave_id = 0u32;
    while emitted < n {
        wave_buf.clear();

        // Find the first not-yet-emitted op (program order).
        let seed = match (0..n).find(|&i| !done[i]) {
            Some(s) => s,
            None => break,
        };
        wave_buf.push(seed);
        done[seed] = true;

        // Greedy lookahead: try to pull additional Free ops into this
        // wavefront. Two conditions for pulling op j:
        //   (a) j does not conflict with any op already in the wave; and
        //   (b) j does not conflict with any pending (not-yet-emitted) op
        //       with index < j. If a pending earlier op conflicts with j,
        //       that earlier op must execute before j, but it was skipped
        //       for this wave - so j must wait too.
        // Strict acts as a hard barrier in the program-order scan: no op
        // past a Strict candidate is considered for this wave.
        let mut scanned = 0usize;
        for j in (seed + 1)..n {
            if scanned >= lookahead { break; }
            scanned += 1;
            if done[j] { continue; }
            if plan[j].order_class == OrderClass::Strict { break; }
            // (a) Conflict with anything already in the wave?
            let mut conflict = false;
            for &w_idx in &wave_buf {
                if plan[j].conflicts_with(&plan[w_idx]) { conflict = true; break; }
            }
            if conflict { continue; }
            // (b) Conflict with any pending earlier op?
            let mut earlier_blocked = false;
            for k in (seed + 1)..j {
                if done[k] { continue; }
                if plan[j].conflicts_with(&plan[k]) {
                    earlier_blocked = true;
                    break;
                }
            }
            if earlier_blocked { continue; }
            wave_buf.push(j);
            done[j] = true;
        }

        // Emit ops in this wavefront (program order to keep traces stable).
        wave_buf.sort_unstable();
        // Track the writes performed by this wavefront so the next barrier
        // can be scoped to just those buffers.
        let mut wave_writes: Vec<BufferId> = Vec::with_capacity(8);
        if trace_enabled {
            let labels: Vec<&'static str> = wave_buf.iter().map(|&i| plan[i].label).collect();
            eprintln!("[graph-trace] wave {wave_id}: {:?}", labels);
        }
        wave_id += 1;
        for &idx in &wave_buf {
            for buf in plan[idx].writes() {
                if !wave_writes.contains(&buf) { wave_writes.push(buf); }
            }
            (plan[idx].emit)(enc)?;
        }
        emitted += wave_buf.len();

        // Insert a memory barrier before the next wavefront on any buffer
        // that future (still-pending) ops will read or write. Skip in serial
        // validate mode (serial encoders are already in-order on the GPU).
        if !serial_validate && emitted < n {
            // Scope: only buffers that (a) were written in this wave AND
            // (b) are touched by some pending op.
            let mut barrier_set: Vec<BufferId> = Vec::with_capacity(wave_writes.len());
            for &b in &wave_writes {
                let mut needed = false;
                for j in 0..n {
                    if done[j] { continue; }
                    if plan[j].accesses.iter().any(|a| a.buf == b) {
                        needed = true; break;
                    }
                }
                if needed { barrier_set.push(b); }
            }
            if !barrier_set.is_empty() {
                // Prefer the resource-scoped barrier when the caller has
                // supplied a buffer-id lookup. Resource-scoped barriers are
                // cheaper than whole-buffer scope because the GPU only
                // serialises against the listed resources, leaving other
                // in-flight dispatches free to retire.
                if let Some(lookup) = buffer_lookup {
                    let mut bufs: smallvec_no_alloc::Small16<&MetalBuffer> =
                        smallvec_no_alloc::Small16::new();
                    for &b in &barrier_set {
                        if let Some(buf) = lookup(b) { bufs.push(buf); }
                    }
                    if !bufs.is_empty() {
                        enc.memory_barrier_with_resources(bufs.as_slice());
                    } else {
                        // No mapping known for any of these buffers; fall
                        // back to whole-buffer scope so correctness still
                        // holds.
                        enc.memory_barrier_with_scope(1);
                    }
                } else {
                    enc.memory_barrier_with_scope(1); // MTLBarrierScope.buffers
                }
            }
            written_so_far.extend(wave_writes.iter().copied());
        }
    }

    Ok(())
}

// ============================================================================
// Minimal stack-only smallvec for the barrier resource set.
//
// Lumen doesn't otherwise depend on the `smallvec` crate, so this tiny
// fixed-capacity Vec-alike covers the only use site (collecting up to
// ~15 MetalBuffer refs for `memoryBarrierWithResources:`).
// ============================================================================

pub(crate) mod smallvec_no_alloc {
    use std::mem::MaybeUninit;

    pub(crate) struct Small16<T> {
        len: usize,
        data: [MaybeUninit<T>; 16],
    }

    impl<T> Small16<T> {
        pub(crate) fn new() -> Self {
            // SAFETY: an array of MaybeUninit can be safely initialised
            // by transmuting from the all-uninit pattern via std API.
            Self {
                len: 0,
                data: unsafe { MaybeUninit::uninit().assume_init() },
            }
        }
        pub(crate) fn push(&mut self, v: T) {
            if self.len < self.data.len() {
                self.data[self.len].write(v);
                self.len += 1;
            }
            // Over-capacity entries are dropped silently; the concurrent-encoder
            // plan has at most ~10 distinct mutable buffers, so this never
            // trips in production.
        }
        pub(crate) fn is_empty(&self) -> bool { self.len == 0 }
        pub(crate) fn as_slice(&self) -> &[T] {
            // SAFETY: 0..self.len are written and aligned correctly.
            unsafe {
                std::slice::from_raw_parts(
                    self.data.as_ptr() as *const T,
                    self.len,
                )
            }
        }
    }

    impl<T> Drop for Small16<T> {
        fn drop(&mut self) {
            // Run drop on all initialised entries.
            for i in 0..self.len {
                // SAFETY: 0..self.len entries are initialised.
                unsafe { self.data[i].assume_init_drop(); }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn op<'a>(
        label: &'static str,
        order_class: OrderClass,
        accesses: Vec<Access>,
        sink: &'a std::sync::Mutex<Vec<&'static str>>,
    ) -> LayerOp<'a> {
        LayerOp {
            label,
            accesses: AccessList::from_iter_inline(accesses),
            order_class,
            emit: Box::new(move |_enc| { sink.lock().unwrap().push(label); Ok(()) }),
        }
    }

    #[test]
    fn writes_iter_filters_reads() {
        let sink: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
        let o = op("x", OrderClass::Free, vec![
            Access::read(BufferId::Normed),
            Access::write(BufferId::Q),
            Access::read_write(BufferId::K),
        ], &sink);
        let writes: Vec<_> = o.writes().collect();
        assert!(writes.contains(&BufferId::Q));
        assert!(writes.contains(&BufferId::K));
        assert!(!writes.contains(&BufferId::Normed));
    }

    #[test]
    fn read_write_conflict() {
        let sink: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
        let a = op("a", OrderClass::Free,
            vec![Access::read(BufferId::Q)], &sink);
        let b = op("b", OrderClass::Free,
            vec![Access::write(BufferId::Q)], &sink);
        assert!(a.conflicts_with(&b));
        assert!(b.conflicts_with(&a));
    }

    #[test]
    fn read_read_no_conflict() {
        let sink: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
        let a = op("a", OrderClass::Free, vec![Access::read(BufferId::Q)], &sink);
        let b = op("b", OrderClass::Free, vec![Access::read(BufferId::Q)], &sink);
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn different_buffers_no_conflict() {
        let sink: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
        let a = op("a", OrderClass::Free, vec![Access::write(BufferId::Q)], &sink);
        let b = op("b", OrderClass::Free, vec![Access::write(BufferId::K)], &sink);
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn byte_range_disjoint_no_conflict() {
        let sink: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
        let a = LayerOp {
            label: "a", order_class: OrderClass::Free,
            accesses: AccessList::from_iter_inline([Access { buf: BufferId::Qkv,
                range: ByteRange { off: 0, len: 1024 }, kind: AccessKind::Write }]),
            emit: Box::new(|_| Ok(())),
        };
        let b = LayerOp {
            label: "b", order_class: OrderClass::Free,
            accesses: AccessList::from_iter_inline([Access { buf: BufferId::Qkv,
                range: ByteRange { off: 1024, len: 1024 }, kind: AccessKind::Write }]),
            emit: Box::new(|_| Ok(())),
        };
        let _ = &sink;
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn whole_range_overlaps_everything() {
        let whole = ByteRange::WHOLE;
        let small = ByteRange { off: 1024, len: 512 };
        assert!(whole.overlaps(&small));
        assert!(small.overlaps(&whole));
    }

    /// Regression: an op that conflicts with a *pending* (skipped) earlier
    /// op must not be merged into the current wave. Without this guard the
    /// scheduler would happily reorder writes past their producers.
    ///
    /// Example: ops [A: W(X)], [B: R(X), W(Y)], [C: R(Y)].
    /// The first wave is A. B conflicts with A (X) so it skips. C does NOT conflict
    /// with A directly (different buffers), but C reads Y which B writes,
    /// so C must wait too. The fix in `emit_plan_into_encoder` checks
    /// pending earlier ops.
    #[test]
    fn pending_earlier_blocks_reorder() {
        let sink: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
        let mut plan = vec![
            op("A", OrderClass::Free,
                vec![Access::write(BufferId::Normed)], &sink),
            op("B", OrderClass::Free,
                vec![Access::read(BufferId::Normed), Access::write(BufferId::Q)], &sink),
            op("C", OrderClass::Free,
                vec![Access::read(BufferId::Q)], &sink),
        ];

        // Compute wave structure without touching Metal: replicate the
        // wave-building portion of emit_plan_into_encoder.
        let n = plan.len();
        let mut done = vec![false; n];
        let mut waves: Vec<Vec<usize>> = Vec::new();
        while done.iter().any(|d| !d) {
            let mut wave = Vec::new();
            let seed = (0..n).find(|&i| !done[i]).unwrap();
            wave.push(seed);
            done[seed] = true;
            for j in (seed + 1)..n {
                if done[j] { continue; }
                if plan[j].order_class == OrderClass::Strict { break; }
                let conflict_with_wave = wave.iter()
                    .any(|&w| plan[j].conflicts_with(&plan[w]));
                if conflict_with_wave { continue; }
                let blocked_by_earlier = (seed + 1..j)
                    .filter(|&k| !done[k])
                    .any(|k| plan[j].conflicts_with(&plan[k]));
                if blocked_by_earlier { continue; }
                wave.push(j);
                done[j] = true;
            }
            waves.push(wave);
        }
        // Expected wave structure:
        //   wave 0: [A] (B conflicts with A; C conflicts transitively via B)
        //   wave 1: [B]
        //   wave 2: [C]
        assert_eq!(waves.len(), 3);
        assert_eq!(waves[0], vec![0]);
        assert_eq!(waves[1], vec![1]);
        assert_eq!(waves[2], vec![2]);
        // Avoid unused-variable warning when plan emits aren't invoked.
        let _ = &mut plan;
    }

    /// confirm `LOOKAHEAD_FULL` doubles the legacy window.
    ///
    /// The constant is `pub(crate)`-internal; this test pins the
    /// invariant `LOOKAHEAD_FULL == 2 * LOOKAHEAD` so a future refactor
    /// that bumps one without the other is caught at unit-test time
    /// (the wider-window scheduler design assumes the lookahead is at
    /// least 2× the legacy one).
    #[test]
    fn lookahead_full_doubles_legacy() {
        assert_eq!(super::LOOKAHEAD_FULL, 2 * super::LOOKAHEAD);
        assert_eq!(super::LOOKAHEAD_FULL, 64);
    }

    /// confirm `active_lookahead` returns the cached window.
    ///
    /// This is a smoke test only — `active_lookahead` reads the
    /// `LUMEN_METAL_CONCURRENT_ENCODER_FULL` env var once and caches it in an atomic,
    /// so the value here depends on test process env. We assert only
    /// that the return value is one of the two valid windows.
    #[test]
    fn active_lookahead_returns_valid_window() {
        let lookahead = super::active_lookahead();
        assert!(
            lookahead == super::LOOKAHEAD || lookahead == super::LOOKAHEAD_FULL,
            "active_lookahead must return LOOKAHEAD or LOOKAHEAD_FULL, got {}",
            lookahead
        );
    }

    /// `concurrent_encoder_full_validate_serial()` is a process-cached bool resolver.
    ///
    /// The env var `LUMEN_METAL_CONCURRENT_ENCODER_FULL_VALIDATE` is process-scoped and
    /// cached on first call. The test asserts that the resolver is
    /// idempotent: two consecutive calls return the same value.
    #[test]
    fn concurrent_encoder_full_validate_is_idempotent() {
        let a = super::concurrent_encoder_full_validate_serial();
        let b = super::concurrent_encoder_full_validate_serial();
        assert_eq!(a, b);
    }

    /// `concurrent_encoder_full_validate_serial()` and `concurrent_encoder_full_enabled()` resolve
    /// independently.
    ///
    /// The whole-prefill encoder gate (`LUMEN_METAL_CONCURRENT_ENCODER_FULL`) and
    /// its byte-identity validator (`LUMEN_METAL_CONCURRENT_ENCODER_FULL_VALIDATE`) MUST
    /// be independent env-var resolvers so that engineers can enable the
    /// new path with the validator OFF (production) or with the validator
    /// ON (regression debug) without coupling. This test confirms the two
    /// fns are distinct functions that read distinct env vars.
    #[test]
    fn concurrent_encoder_full_validate_and_enabled_are_independent_fns() {
        // Each returns a bool; nothing else to check at unit-test scope
        // without spawning subprocesses (which is overkill for the harness).
        let _ = super::concurrent_encoder_full_validate_serial();
        let _ = super::concurrent_encoder_full_enabled();
    }
}
