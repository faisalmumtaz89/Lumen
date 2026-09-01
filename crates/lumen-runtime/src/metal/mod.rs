//! Metal GPU F32 compute backend for Apple Silicon.
//!
//! Metal inference is active and validated against the CPU and HF FP16
//! references. The default-on Metal opt-ins deliver Q8 prefill at 0.95× the
//! llama.cpp baseline, BF16 prefill at 0.96× the baseline, and decode that
//! beats the baseline on Q8/Q4.
//!
//! The `qwen35moe` Metal-only forward path is retained as a reference for
//! the future CUDA MoE implementation.
//!
//! ---
//!
//! Implements `ComputeBackend` using Metal GPU compute shaders. On Apple Silicon
//! unified memory, weight data from mmap is already in GPU-accessible memory,
//! enabling zero-copy weight access via `MTLBuffer(bytesNoCopy:)`.
//!
//! Decode path: each `compute_layer` call encodes and executes GPU commands per
//! layer (async commit). Prefill path: ALL layers are encoded into a SINGLE
//! Metal command buffer with one commit_and_wait() at the end, eliminating
//! N-1 GPU-CPU sync barriers.
//!
//! # Performance characteristics
//!
//! - Matrix-vector multiply: GPU-parallelized across output rows
//! - RMSNorm: SIMD group reductions for fast sum-of-squares
//! - Attention: Scores computed in parallel, softmax on GPU, value accumulation parallel
//! - Activation buffers: Metal shared-mode buffers (CPU/GPU zero-copy)

mod decode_greedy;
pub(crate) mod decode_profile;
mod decode_single_cb;
pub(crate) mod ffi;
pub(crate) mod io;
pub(crate) mod shaders;
// disk-KV sync helpers (GPU<->CPU KV mirror + GDN state).
mod disk_sync;
mod gdn;
mod gpu_resident;
mod graph_reorder;
mod moe;
mod pipelines;
mod prefill;
mod prefill_encode;
pub(crate) mod profile;
// runtime Q8_0 hot-weight repack for FFN tensors (env-gated).
pub(crate) mod repack_q8;
// runtime Q4_0 hot-weight repack (Q4 port of), env-gated.
pub(crate) mod repack_q4;
// runtime BF16 GDN qkv+attn_gate concat-then-stripe repack
// (24-GDN-layer subset under the 4.8 GB Apple AGX TLB threshold), env-gated.
mod backend_impl;
pub(crate) mod repack_bf16;
pub(crate) mod types;
pub use types::*;

use self::ffi::{MTLSize, MetalBuffer, MetalCommandQueue, MetalDevice};
use self::io::MetalIOQueue;
use crate::error::RuntimeError;
use crate::expert::cache::ExpertLfuCache;
use crate::expert::profiler::ExpertActivationProfiler;
use crate::expert::reader::ExpertReader;
use crate::weight::cache::LayerView;
use lumen_format::quantization::QuantScheme;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::sync::OnceLock;

/// Page size for alignment checks (4 KiB on all Apple Silicon).
const PAGE_SIZE: usize = 4096;

/// Per-process cached resolver for `LUMEN_METAL_DECODE_DELAY_US`.
///
/// Falls through to `runtime_defaults::metal_decode_delay_us_default()` when
/// the env var is unset (0 — Metal greedy decode is deterministic at the
/// kernel level since the DET-001 fixes). An explicit env value
/// always wins so A/B benchmark drivers and CI can pin it. The `OnceLock`
/// cache keeps the hot decode path to a single integer load after the first
/// decode token of the process.
fn metal_decode_delay_us() -> u64 {
    static CACHED: OnceLock<u64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LUMEN_METAL_DECODE_DELAY_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or_else(crate::runtime_defaults::metal_decode_delay_us_default)
    })
}

/// Apply the decode-delay after a per-token `commit_and_wait()`.
///
/// When the resolved delay is `0` (the default) this is a single integer
/// load + branch with no syscall — the path stays bit-exact and the only
/// cost is one comparison. A non-zero value issues a CPU `thread::sleep` for
/// the configured number of microseconds — a diagnostic perturbation of GPU
/// scheduler timing, not a determinism guarantee (DET-001 is fixed at the
/// kernel level; a CPU sleep cannot make a within-token FP reduction
/// deterministic).
#[inline(always)]
fn maybe_apply_metal_decode_delay() {
    let delay_us = metal_decode_delay_us();
    if delay_us > 0 {
        std::thread::sleep(std::time::Duration::from_micros(delay_us));
    }
}

/// Opt-in switch for the GPU temperature sampler (Option A,
/// `LUMEN_METAL_GPU_SAMPLER=1`, default OFF). When set AND the active sampling
/// config is the subset the `gpu_sampler` kernel reproduces (temperature>0;
/// no top_k/top_p/min_p; the anti-restate byte-guard inactive — always true for
/// temperature>0), the lean decode pipeline finalizes the sampled token ON the
/// GPU (parity-matched to the CPU `sample_logits`) so temp>0 decode pipelines
/// exactly like the greedy argmax chain, removing the serial CPU-sampler +
/// logit-readback that caps the sampled decode rate. With the flag UNSET the
/// default CPU-sampler path is taken unchanged (byte-identical). Cached.
pub(crate) fn metal_gpu_sampler_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_GPU_SAMPLER")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_GPU_SAMPLER_EXACT=1` the GPU sampler uses the bit-exact
/// single-thread `gpu_sampler` kernel (sequential sum + walk, reproducing the
/// CPU accumulation order exactly bar the exp transcendental). Default OFF -> the
/// latency-hiding `gpu_sampler_fast` kernel (the production path). Cached. Used
/// only for parity validation / debugging the rare near-tie.
pub(crate) fn metal_gpu_sampler_exact_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_GPU_SAMPLER_EXACT")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// Maximum pass-1 tile count for the two-pass tiled argmax. Sizes the partials
/// scratch buffer ([ARGMAX_MAX_TILES] f32 + [ARGMAX_MAX_TILES] u32). 256 partials
/// reduce in a single 256-thread pass-2 TG.
pub(crate) const ARGMAX_MAX_TILES: usize = 256;

/// Pass-1 tile count for the two-pass tiled greedy argmax. 128 spreads the
/// 248320-wide vocab reduction across the GPU while keeping the <=256 partials
/// within a single 256-thread pass-2 threadgroup; token selection is
/// tile-count-invariant, so this only sets pass-1 parallelism. Must be
/// <= ARGMAX_MAX_TILES.
pub(crate) const TILED_ARGMAX_TILES: usize = 128;

/// The full Q4_0 fast-decode kernel stack — now the DEFAULT decode path (no flag).
/// Engages the complete validated Metal Q4_0 batch-1 decode acceleration on the
/// Qwen3.5 family: the decode-qmv weight layout (GDN QKV/gate, FFN gate/up + down,
/// lm_head, full-attn K/V), the half2 dequant-MAC matvec kernels with f16 native
/// scales and f32 per-block accumulation (sampling-correct), the 4-way amortized
/// F16 GDN decode state recurrence, the concurrent projection encoder (UltraFusion
/// die-saturation), and the lean async decode pipeline. Byte-identical to the
/// legacy path under greedy and correct under all sampling. The old
/// `LUMEN_METAL_Q4_FAST_DECODE` opt-in knob has been removed — the stack is
/// unconditionally on (the individual `LUMEN_METAL_*` levers it ORs into are now
/// always-true through this, and remain only for diagnostics).
pub(crate) fn q4_fast_decode_enabled() -> bool {
    true
}

/// Part of the default Q4_0 fast-decode stack (unconditional): the dense FFN
/// gate/up GEMV uses
/// the F16-SCALES kernel `qmv_q4_0_gate_up_swiglu_f16sc` with gate/up scale
/// buffers built as f16 (2 B/block instead of f32's 4 B). This streams 18 (not
/// 20) weight bytes per 32-value block = ~10% fewer bytes on the bandwidth-bound
/// dense FFN gate/up matvec. The f16 scale is the on-disk Q4_0 scale's native
/// precision (the f32 path widened it), so the result is byte-identical to the
/// f32-scale kernel. Cached.
pub(crate) fn q4_gateup_f16sc_enabled() -> bool {
    q4_gateup_h2math_enabled()
}

/// Part of the default Q4_0 fast-decode stack (unconditional): the dense FFN-down
/// GEMV uses the
/// F16-SCALES kernel `qmv_q4_0_residual_f16sc` with the per-block scale buffer
/// built as f16 (2 B/block instead of f32's 4 B). FFN-down is the longest-K
/// matvec (in=12288 -> 384 blocks/row), so this is the single largest per-token
/// scale stream: streaming 18 (not 20) weight bytes per 32-value block = ~10%
/// fewer bytes on the bandwidth-bound FFN-down matvec. The f16 scale is the
/// on-disk Q4_0 scale's native precision (the f32 decode-qmv layout widened it),
/// so the result is byte-identical to the f32-scale kernel. Cached.
pub(crate) fn q4_qmv_down_f16sc_enabled() -> bool {
    q4_down_h2math_enabled()
}

/// Part of the default Q4_0 fast-decode stack (unconditional — the dense FFN-down
/// GEMV always uses the half2-math kernel `qmv_q4_0_residual_f16sc_h2math`, on top
/// of the self-engaged FFN-down f16-scale path; see `q4_qmv_down_f16sc_enabled`).
/// x is staged as `half2` pairs and each per-32-block 16-term nibble*x dot is
/// accumulated in `half2` (two half FMAs per Apple GPU vector ALU slot), with the
/// cross-block reduction + sum-of-x + scale kept in f32.
///
/// RATIONALE: FFN-down is the longest-K matvec (in=12288 -> 384 blocks/row = the
/// most dequant FMAs per output row in the model), so doubling the per-nibble
/// unpack-MAC ALU width (half2 vs scalar half, which leaves half the vector ALU
/// idle) attacks the compute-bound 4-bit unpack that keeps Q4_0 decode below its
/// bandwidth roofline. NEAR-TIE, not guaranteed byte-identical (per-block
/// product+sum rounds to f16 mantissa); the f32 cross-block accumulation + f32 -8
/// fold + f32 scale bound the long-K drift.
pub(crate) fn q4_down_h2math_enabled() -> bool {
    true
}

/// Part of the default Q4_0 fast-decode stack (unconditional — the dense FFN
/// gate/up GEMV always uses the half2-math kernel
/// `qmv_q4_0_gate_up_swiglu_f16sc_h2math`, on top of the self-engaged gate/up
/// f16-scale path; see `q4_gateup_f16sc_enabled`). x is staged as `half2` pairs and
/// each per-32-block 16-term gate/up dot is accumulated in `half2` (two half FMAs
/// per Apple GPU vector ALU slot), with the cross-block reduction + sum-of-x +
/// scale + RMSNorm + SwiGLU kept in f32.
///
/// RATIONALE: the dominant FFN matvec is compute-bound on the 4-bit unpack and is
/// not washed out by the lean pipeline's overlap. Apple M-series GPU ALUs are
/// natively 16-bit-vectorized, so a `half2` FMA issues two half MACs in one ALU
/// slot — a scalar-`half` path leaves half the lane idle. Vectorizing the dequant
/// MAC into `half2` halves the per-block dequant-MAC instruction count on the
/// matvec that matters most. NEAR-TIE, not guaranteed byte-identical (the even/odd
/// half-lane partial-sum grouping differs from a scalar running sum); the f32
/// cross-block accumulation + f32 SwiGLU bound the drift to the class the scalar
/// twin kept answer-identical on the corpus. Takes priority over the other gate/up
/// variants.
pub(crate) fn q4_gateup_h2math_enabled() -> bool {
    true
}

/// Part of the default Q4_0 fast-decode stack (unconditional — the lm_head /
/// output-projection GEMV always uses `qmv_q4_0_rmsnorm_f16sc_h2math`, the
/// half2-vectorized rmsnorm matvec, on top of the self-engaged lm_head f16-scale
/// path; see `q4_lmhead_f16sc_enabled`, which builds the re-quantized Q4
/// output_proj scale buffer as f16). x is staged as `half` (packed two-per-`half2`),
/// each per-32-block 16-term nibble dot accumulates in `half2` (TWO half FMAs per
/// ALU slot), with the cross-block reduction + sum-of-x + scale + RMSNorm
/// sum-of-squares kept in f32.
///
/// RATIONALE: the lm_head (out=vocab, in=hidden=4096) is the single largest weight
/// tensor (the Q8 output_proj re-quantized to Q4 by default,
/// ~0.54 GB streamed/token). Packing its dequant-MAC into the native 16-bit-
/// vectorized ALU lane (which a scalar `half` path leaves half-idle) halves the
/// inner-loop arithmetic on the largest matvec. NEAR-TIE, not guaranteed
/// byte-identical (per-block product+sum rounds to f16 mantissa, half-lane
/// partial-sum grouping differs); the f32 cross-block accumulation bounds the drift
/// to the class the gate/up h2math twin kept answer-identical on the corpus.
pub(crate) fn q4_lmhead_h2math_enabled() -> bool {
    true
}

/// Part of the default Q4_0 fast-decode stack (unconditional — the GDN-layer
/// QKV-in-projection (`wq`, qkv_dim rows) AND attn_gate (q_dim rows) GEMVs always
/// use `qmv_q4_0_rmsnorm_f16sc_h2math`, the half2-vectorized rmsnorm matvec, on top
/// of the self-engaged GDN QKV-in-proj + attn_gate f16-scale path; see
/// `q4_proj_f16sc_enabled`, which builds both decode-qmv scale buffers as f16).
/// x is staged as `half2` pairs and each
/// per-32-block 16-term nibble dot accumulates in `half2` (TWO half FMAs per Apple
/// GPU vector ALU slot), with the cross-block reduction + sum-of-x + scale + RMSNorm
/// sum-of-squares kept in f32.
///
/// RATIONALE: with gate/up, lm_head and FFN-down already on half2, the GDN
/// QKV-in-proj + attn_gate (out=qkv_dim=8192 in-proj + q_dim=4096 gate,
/// in=hidden=4096, x 24 GDN layers — the majority of layers in the Qwen3.5-9B
/// GatedDeltaNet hybrid) is the largest remaining f32-math matvec stream. Packing
/// its dequant-MAC into the native 16-bit-vectorized ALU lane (which a scalar
/// `half` path leaves half-idle) halves the inner-loop dequant ALU on the
/// second-largest pool. SAME bindings + geometry as the f16sc kernel already wired
/// here (2 SG/TG, 4 rows/SG, out/8 TGs, 64 threads) — only the per-32-block dequant
/// MAC widens to half2. Takes precedence over the plain f16sc proj variant.
/// NEAR-TIE, not guaranteed byte-identical (the even/odd half-lane partial-sum
/// grouping differs from a scalar running sum); the f32 cross-block accumulation
/// bounds the drift to the class the gate/up/lm_head/down h2math twins kept
/// answer-identical on the corpus.
pub(crate) fn q4_proj_h2math_enabled() -> bool {
    true
}

/// Part of the default Q4_0 fast-decode stack (unconditional). The GDN `ssm_out`
/// output projection — which ships as Q8_0 on Qwen3.5-9B (so its per-token weight
/// stream is the LAST major Q8_0 stream on the GDN decode path: in=value_dim=4096,
/// out=hidden=2048, ~34 B / 32 weights over 24 GDN layers ≈ 0.21 GB/token) — is
/// RE-QUANTIZED Q8_0 -> Q4_0 at load into the **native GGUF Q4_0 block layout**
/// (2-byte f16 scale + 16 de-interleaved nibble bytes = 18 B / 32 weights) and
/// stored in a per-GDN-layer standalone weight buffer. The fused decode ssm_out
/// dispatch then binds that Q4 buffer and runs the EXISTING
/// `dequant_matmul_q4_0_silu_deferred_residual_copy_nr2` kernel (silu(gate)*x +
/// matvec + residual + copy, ALL in ONE dispatch) in place of the Q8 fused
/// `..._q8_0_..._nr2` kernel.
///
/// This is the CORRECTED form of the earlier qmv-layout ssm_out requant approach
/// (which requanted to the *qmv* sequential-nibble layout and routed through the
/// 3-dispatch `silu_elementwise_mul + qmv_q4_0_residual + residual_add_copy` path
/// — the byte saving was real but eaten by the two extra dispatches+barriers,
/// measured FLAT). By reusing the Q8 path's SINGLE fused kernel with a Q4 weight,
/// this captures the Q8->Q4 halving (~0.10 GB/token, ~1.9% of the 5.245 GB/tok Q4
/// stream) with ZERO added dispatch overhead. Deliberate precision tradeoff (NOT
/// byte-identical to the Q8 ssm_out path — Q8->Q4 requant rounding) validated by
/// the correctness gate to keep the answer (byte-identical OR near-tie); the
/// requant measured byte-identical on the corpus (1024/169/963/Paris unchanged).
pub(crate) fn q4_ssmout_nr2_enabled() -> bool {
    true
}

/// Part of the default Q4_0 fast-decode stack (unconditional; the Q4 lm_head
/// decode-qmv buffers re-quantized the Q8 output_proj):
/// the lm_head / output-projection GEMV uses the F16-SCALES kernel
/// `qmv_q4_0_rmsnorm_f16sc` with the per-block scale buffer built as f16 (2 B/block
/// instead of f32's 4 B). The lm_head is the single LARGEST weight tensor
/// (out=vocab), so its scale stream is the biggest single-tensor scale stream in
/// the model: streaming 18 (not 20) weight bytes per 32-value block = ~10% fewer
/// bytes on the bandwidth-bound lm_head matvec. The f16 scale is the on-disk Q4_0
/// scale's native precision (the f32 decode-qmv layout widened it), so the result
/// is byte-identical to the f32-scale kernel. Cached.
pub(crate) fn q4_lmhead_f16sc_enabled() -> bool {
    q4_lmhead_h2math_enabled()
}

/// Part of the default Q4_0 fast-decode stack (unconditional): the GDN-layer
/// QKV-in-projection (`wq`, qkv_dim rows) AND attn_gate (q_dim rows) GEMVs use
/// the F16-SCALES kernel `qmv_q4_0_rmsnorm_f16sc` with their per-block scale
/// buffers built as f16 (2 B/block instead of f32's 4 B). These two matvecs run
/// on EVERY GDN layer (the majority of layers in the Qwen3.5-9B GatedDeltaNet),
/// so their combined per-token scale stream (12288 rows x hidden/32 blocks) is
/// comparable in magnitude to the FFN-down scale stream: streaming 18 (not 20)
/// weight bytes per 32-value block = ~10% fewer bytes on these bandwidth-bound
/// matvecs. The f16 scale is the on-disk Q4_0 scale's native precision (the f32
/// decode-qmv layout widened it), so the result is byte-identical to the
/// f32-scale kernel. Cached.
pub(crate) fn q4_proj_f16sc_enabled() -> bool {
    q4_proj_h2math_enabled()
}

/// Part of the default Q4_0 fast-decode stack (unconditional). The full-attn
/// Q+gate/K/V projection cluster in the single-CB decode path is dispatched on a
/// CONCURRENT compute encoder (`MTLDispatchTypeConcurrent`) instead of the layer's
/// serial encoder, so Metal MAY distribute the three independent matvecs'
/// threadgroups across BOTH M3 Ultra UltraFusion dies (die-saturation lever). The
/// three projections read the SAME normed x and write DISJOINT buffers (qkv_buf /
/// k_buf / v_buf) with no shared state, so concurrent dispatch is byte-identical to
/// the serial dispatch (the recurrence/dependent ops stay on the serial encoder). A
/// resource-scoped barrier on (qkv_buf,k_buf,v_buf) is emitted before the
/// downstream DNA consumer. The GDN h_state recurrence is never made concurrent
/// (its determinism constraint is unrelated to these independent projections).
pub(crate) fn metal_concurrent_proj_enabled() -> bool {
    true
}

/// When `LUMEN_METAL_Q8_GDN_QKVGATE_2STREAM` is unset or != "0" (DEFAULT-ON, v0.5), the GDN
/// qkv in-proj + attn_gate Q8 matvecs are FUSED into `dequant_matmul_q8_0_qkv_gate_2stream`:
/// each thread accumulates BOTH dot products from the shared normed x (2 in-flight weight
/// streams/thread = more memory-level parallelism, +0.83% 27B-Q8 decode). BYTE-IDENTICAL to
/// the two separate 2sg dispatches. Set the env to "0" to disable.
pub(crate) fn metal_q8_gdn_qkvgate_2stream_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let on = std::env::var("LUMEN_METAL_Q8_GDN_QKVGATE_2STREAM")
        .map(|v| v != "0")
        .unwrap_or(true);
    CACHE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// Part of the default Q4_0 fast-decode stack (unconditional). The Tier-0 fused GDN decode
/// recurrence dispatches the 4-WAY VI-AMORTIZED `gdn_state_output_l2_sg_f16_h1_v4` kernel:
/// each threadgroup handles FOUR adjacent val_dim columns and computes the (vi-invariant)
/// Q/K L2-norm + Q/K device load ONCE, reusing it across all four. This cuts the per-token
/// redundant Q/K L2-norm ALU (the two simd_sum reductions + two sqrt + two reciprocal-
/// selects the reference grid recomputes val_dim=128x per head) AND the redundant Q/K
/// device reads on the GDN recurrence critical dependency path a further 2x vs the v2
/// kernel (4x vs reference). Grid quarters 4096 -> 1024 TGs (~13/core on the 80-core
/// M3 Ultra, still richly occupied). Takes precedence over the v2 / single-vi f16_h1 /
/// plain-f16 kernels when set; shares the SAME F16 h_state mirror (same lazy F32->F16
/// conversion). Output is BYTE-IDENTICAL to the single-vi/v2 kernels (same simd_sum
/// reduction tree, identical per-column recurrence arithmetic) and a near-tie vs F32
/// (F16 state rounding). Falls back to the f32 path when the mirror / v4 pipeline is
/// absent. Cached.
pub(crate) fn gdn_f16_state_h1_v4_enabled() -> bool {
    true
}

/// Parallel two-kernel MoE router (per-expert logits across the
/// grid + small top-k softmax) instead of the single-threadgroup serial router.
/// Measured +198% decode on Qwen3.5-MoE-35B-A3B Metal, byte-identical greedy
/// output, PRISTINE quality. **DEFAULT ON**; set
/// `LUMEN_METAL_MOE_ROUTER_PARALLEL=0` to force the legacy serial router.
/// Cached after first read.
pub(crate) fn moe_router_parallel_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    // Default ON: only the explicit "0"/"false"/"off" disables it.
    let v = match std::env::var("LUMEN_METAL_MOE_ROUTER_PARALLEL") {
        Ok(s) => !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")),
        Err(_) => true,
    };
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// Number of threadgroups to dispatch for the (replicated) MoE
/// router top-k softmax kernel. Default 1 (original single-TG behavior). Setting
/// N>1 dispatches N redundant TGs (all compute the identical top-k; only TG 0
/// writes) to keep the GPU occupied and avoid the 1-TG drain bubble on the serial
/// decode encoder. Env `LUMEN_METAL_MOE_ROUTER_TOPK_TGS=N`. Cached after first read.
pub(crate) fn moe_router_topk_tgs() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static CACHE: AtomicU32 = AtomicU32::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur;
    }
    let v = std::env::var("LUMEN_METAL_MOE_ROUTER_TOPK_TGS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1);
    CACHE.store(v, Ordering::Relaxed);
    v
}

/// Expert-grouped prefill MoE FFN (mul_mat_id style) instead of
/// Option B (all-experts-all-tokens). Measured +308% prefill (159→649 tok/s) on
/// Qwen3.5-MoE-35B-A3B Metal Q8, byte-identical greedy output vs Option B (10/10
/// prompts md5-equal), PRISTINE quality by equivalence. **DEFAULT ON**;
/// set `LUMEN_METAL_MOE_PREFILL_GROUPED=0` to force Option B.
/// Cached after first read. (Q8_0 experts only; non-Q8 layers fall back to
/// Option B regardless.)
pub(crate) fn moe_prefill_grouped_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = match std::env::var("LUMEN_METAL_MOE_PREFILL_GROUPED") {
        Ok(s) => !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")),
        Err(_) => true,
    };
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// Route-sort scatter mode for the grouped MoE prefill.
///   0 = serial   (legacy `moe_prefill_route_sort`, scatter on GPU thread 0)
///   1 = par      (`moe_prefill_route_sort_par`, one thread/expert scan)
///   2 = atomic   (`moe_prefill_route_sort_atomic`, fully-parallel
///                 atomic-cursor scatter — removes the 256x per-expert scan)
/// DEFAULT = atomic. Env `LUMEN_METAL_MOE_ROUTE_SORT=serial|par|atomic`. The
/// legacy `LUMEN_METAL_MOE_ROUTE_SORT_PAR=0` is honored as a serial override.
/// All three produce byte-identical final logits (atomic/par permute within an
/// expert segment, which cancels through gather->GEMM->scatter).
pub(crate) fn moe_route_sort_mode() -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(255);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 255 {
        return cur;
    }
    // Legacy kill switch takes priority: PAR=0 forces serial.
    let v = if matches!(
        std::env::var("LUMEN_METAL_MOE_ROUTE_SORT_PAR").as_deref(),
        Ok("0")
    ) {
        0u8
    } else {
        match std::env::var("LUMEN_METAL_MOE_ROUTE_SORT").as_deref() {
            Ok("serial") | Ok("0") => 0,
            Ok("par") | Ok("1") => 1,
            _ => 2, // atomic default
        }
    };
    CACHE.store(v, Ordering::Relaxed);
    v
}

/// Flattened work-tile-map dispatch for the grouped MoE GEMM.
/// The legacy grid is (N/32, max_m_tiles, num_experts) = ~19.5x over-subscribed
/// at batch=1239/256e (avg ~2 real m-tiles/expert vs max_m_tiles=39). The
/// tile-map path builds the exact non-empty (expert, m_tile) list on GPU and
/// dispatches (N/32, n_work_tiles_bound, 1). Byte-identical (only changes WHICH
/// TG computes which output tile; per-tile MMA order unchanged). DEFAULT ON,
/// kill switch `LUMEN_METAL_MOE_GEMM_TILEMAP=0`.
pub(crate) fn moe_gemm_tilemap_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = match std::env::var("LUMEN_METAL_MOE_GEMM_TILEMAP") {
        Ok(s) => !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")),
        Err(_) => true,
    };
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// Float4-vectorized grouped gather/scatter (128-bit copies,
/// 4x fewer threads; byte-exact). Requires hidden_dim % 4 == 0. DEFAULT ON,
/// kill switch `LUMEN_METAL_MOE_GATHER_VEC4=0`.
pub(crate) fn moe_gather_vec4_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = match std::env::var("LUMEN_METAL_MOE_GATHER_VEC4") {
        Ok(s) => !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")),
        Err(_) => true,
    };
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// Async expert prefetch state for one-layer lookahead.
///
/// After layer N's router produces expert_ids, a background thread pre-reads
/// those same experts for layer N+1. If layer N+1's router selects the same
/// experts (common due to routing locality), the prefetch result is used
/// directly, avoiding synchronous disk I/O.
struct PrefetchState {
    /// The target layer index for which experts were prefetched.
    target_layer: usize,
    /// The expert IDs that were prefetched (retained for diagnostics).
    #[allow(dead_code)]
    expert_ids: Vec<u32>,
    /// Join handle resolving to prefetched expert data.
    /// Each result corresponds to the expert_ids in order.
    handle: std::thread::JoinHandle<
        Vec<(
            u32,
            Result<
                (Vec<u8>, lumen_format::index::ExpertSlice),
                crate::expert::reader::ExpertReaderError,
            >,
        )>,
    >,
}

// ============================================================================
// MetalF32Backend
// ============================================================================

/// Metal GPU F32 compute backend.
///
/// Identical API to NaiveF32Backend and SimdF32Backend. The engine interacts
/// with it through `Box<dyn ComputeBackend>` with no awareness of the GPU.
pub struct MetalF32Backend {
    device: MetalDevice,
    queue: MetalCommandQueue,
    pipelines: Option<MetalPipelines>,

    // Global tensors (GPU buffers)
    embedding_buf: Option<MetalBuffer>,
    final_norm_buf: Option<MetalBuffer>,
    output_proj_buf: Option<MetalBuffer>,

    // CPU F32 copies of the global tensors (set via set_global_tensors).
    //
    // `final_norm` is always retained (tiny). For QUANTIZED models (Q8_0/Q4_0/
    // F16/BF16) `init()` builds the GPU embedding/output_proj buffers from the
    // raw native-quant bytes and then FREES `embedding`/`output_proj` (sets them
    // to empty Vecs) — every reachable Metal path reads the GPU buffers or the
    // raw bytes, never these F32 Vecs, so retaining them wasted ~4 GB. They stay
    // populated ONLY for non-quantized (pure-F32) models, where `init()` uploads
    // them directly and the embed_token CPU fallback may index `embedding`.
    embedding: Vec<f32>,
    final_norm: Vec<f32>,
    output_proj: Vec<f32>,
    /// Raw output_proj bytes for Q8_0 GPU dispatch (avoids CPU dequant).
    output_proj_raw: Option<Vec<u8>>,
    /// Quantization scheme of the output_proj tensor.
    output_proj_quant: QuantScheme,
    /// Raw embedding bytes for Q8_0/Q4_0 GPU dequant kernels.
    embedding_raw: Option<Vec<u8>>,
    /// Quantization scheme of the embedding tensor.
    embedding_quant: QuantScheme,
    /// Whether output_proj shares embedding storage (weight tying).
    weight_tying: bool,

    scratch: Mutex<Option<MetalScratch>>,
    cached_hidden_dim: usize,
    cached_vocab_size: usize,
    /// Attention-geometry expectations from init() hyperparams; lock-free
    /// so streaming layer-buffer creation can validate without the scratch
    /// mutex. None until init() runs.
    cached_attn_dims: Option<gpu_resident::AttnDims>,
    /// Header-declared MoE expert count (0 for dense), cached at init so
    /// streaming layer-buffer creation can reconcile it against each
    /// layer's expert bank without the scratch mutex.
    cached_moe_num_experts: usize,

    // ====================================================================
    // MoE expert caching infrastructure
    // ====================================================================
    // Only active for MoE models in streaming mode (non-GPU-resident).
    // Records expert activation patterns and caches hot experts to avoid
    // redundant SSD reads on subsequent tokens.
    /// Expert activation profiler: tracks per-(layer, expert) activation counts.
    /// Initialized when the model has num_experts > 0.
    expert_profiler: Option<Mutex<ExpertActivationProfiler>>,

    /// LFU cache for expert weights: keeps hot experts in RAM.
    /// Checked before loading from disk in the streaming MoE decode path.
    expert_cache: Option<Mutex<ExpertLfuCache>>,

    /// Direct byte-range reader for individual expert weights from LBC file.
    /// Used on cache misses to load only the needed expert (not the full layer blob).
    expert_reader: Option<Mutex<ExpertReader>>,

    /// Path to the LBC model file (stored for ExpertReader initialization).
    lbc_path: Option<PathBuf>,

    /// Number of profiling tokens remaining before triggering cache warm-up.
    /// When this reaches 0, `warm_from_profile()` is called to pre-populate the
    /// expert cache with the hottest experts observed during the profiling phase.
    /// Uses AtomicUsize for interior mutability (called from &self methods).
    profiling_tokens_remaining: AtomicUsize,
    /// Number of top-K experts per layer to cache during warmup.
    profiling_top_k: usize,
    /// Whether cache warmup has been completed.
    /// Uses AtomicBool for interior mutability (called from &self methods).
    warmup_complete: AtomicBool,

    // Cache-conditional routing bias
    // ====================================================================
    /// Bias magnitude for cache-conditional routing. When > 0.0, cached experts
    /// receive a logit boost of `cache_bias_lambda` before softmax in the MoE
    /// router, nudging borderline selections toward already-cached experts.
    /// Default 0.0 (disabled). Set via `configure_routing_bias()`.
    cache_bias_lambda: f32,

    // ====================================================================
    // MoE I/O instrumentation
    // ====================================================================
    /// Bytes of expert data loaded from disk via ExpertReader (Tier 2 misses).
    expert_bytes_from_disk: AtomicU64,
    /// Bytes of expert data served from ExpertLfuCache (Tier 1 + Tier 2 hits).
    expert_bytes_from_cache: AtomicU64,
    /// Bytes of expert data accessed via full layer blob fallback (Tier 3).
    expert_bytes_from_blob: AtomicU64,

    // ====================================================================
    // Option A dispatch
    // ====================================================================
    /// When true, MoE decode dispatches only the top-K selected expert FFNs
    /// instead of all num_experts (Option B). In streaming mode, expert_ids are
    /// available CPU-side after synchronous router readback. In
    /// GPU-resident mode, a two-CB split per MoE layer achieves the same
    /// selective dispatch. Default false (opt-in via
    /// `configure_option_a(true)`).
    use_option_a: bool,

    // ====================================================================
    // Async expert prefetching
    // ====================================================================
    /// One-layer lookahead prefetch handle. After layer N's router produces
    /// expert_ids, a background thread pre-reads the same experts for layer N+1
    /// from disk. At layer N+1, the prefetch result is checked before falling
    /// back to synchronous load. Only active when use_option_a is true.
    ///
    /// The handle contains: (target_layer, expert_ids, join_handle).
    /// The join_handle resolves to Vec<(expert_id, Result<(Vec<u8>, ExpertSlice)>)>.
    prefetch_handle: Mutex<Option<PrefetchState>>,

    // ====================================================================
    // Router diagnostics
    // ====================================================================
    /// When true, router debug readback is active: after each decode token,
    /// expert_ids and expert_weights are read back for all MoE layers and
    /// stored in `router_debug_log`.
    router_debug_enabled: bool,

    /// Accumulated per-layer routing stats from decode tokens.
    /// Only populated when `router_debug_enabled` is true.
    router_debug_log: Mutex<Vec<RouterLayerStats>>,

    // ====================================================================
    // Metal IO command queue for direct NVMe-to-GPU DMA
    // ====================================================================
    /// Metal IO command queue for direct file-to-GPU DMA transfers.
    /// Available on Metal 3 (M2+) with macOS 13+. When present, streaming
    /// expert loading bypasses CPU memory and loads directly from NVMe SSD
    /// into the Metal buffer. Falls back to pread + blit when None.
    metal_io_queue: Option<MetalIOQueue>,
}

impl MetalF32Backend {
    /// Ensure the persistent GDN state and shared GDN scratch buffers for
    /// `layer_idx` exist, returning its GDN index. The GPU-resident preload
    /// allocates these up front; the streaming decode and batched-prefill
    /// paths allocate lazily on first touch (layers are visited in order, so
    /// indices match the model's GDN layer sequence). Idempotent.
    pub(crate) fn ensure_gdn_layer_state(
        &self,
        s: &mut MetalScratch,
        layer_idx: usize,
    ) -> Result<usize, RuntimeError> {
        if let Some(idx) = s.gdn_layer_idx_map.get(layer_idx).copied().flatten() {
            if idx >= s.gdn_h_states.len() {
                return Err(RuntimeError::Compute(format!(
                    "GDN layer {layer_idx} maps to index {idx} but only {} \
                     state buffers exist (a prior preload failed partway)",
                    s.gdn_h_states.len()
                )));
            }
            return Ok(idx);
        }
        if s.gdn_conv_kernel_size == 0 {
            return Err(RuntimeError::Compute(
                "GDN conv_kernel is 0 (malformed LBC hyperparams)".into(),
            ));
        }
        let gdn_idx = s.gdn_h_states.len();

        // GDN dimensions differ from full-attention hyperparams; they come
        // from the resolved SSM dims (9B {32,16,128,4} default, or
        // 27B {48,16,128,4}) populated in init() from hyperparams.gdn_dims().
        let gdn_num_v_heads = s.gdn_num_v_heads; // ssm.time_step_rank
        let gdn_num_k_heads = s.gdn_num_k_heads; // ssm.group_count
        let gdn_head_dim = s.gdn_head_dim; // ssm.state_size
                                           // Fused QKV channels: 2 * (num_k_heads*head_dim) + num_v_heads*head_dim
                                           //   9B = 8192, 27B = 10240.
        let gdn_qkv_dim = 2 * gdn_num_k_heads * gdn_head_dim + gdn_num_v_heads * gdn_head_dim;
        // V / gate / output-projection width = num_v_heads * head_dim
        //   9B = 4096, 27B = 6144.
        let gdn_q_dim = gdn_num_v_heads * gdn_head_dim;
        let hidden_dim = s.hidden_dim;
        let conv_kernel_size = s.gdn_conv_kernel_size;
        let h_state_size = gdn_num_v_heads * gdn_head_dim * gdn_head_dim;
        let conv_state_size = (conv_kernel_size - 1) * gdn_qkv_dim;

        // Persistent h_state: F32 (4 B/elem).
        let h_buf = self
            .device
            .new_buffer(h_state_size * 4)
            .ok_or_else(|| RuntimeError::Compute("Failed to allocate GDN h_state".into()))?;
        h_buf.write_f32(&vec![0.0f32; h_state_size]);

        let c_buf = self
            .device
            .new_buffer(conv_state_size * 4)
            .ok_or_else(|| RuntimeError::Compute("Failed to allocate GDN conv_state".into()))?;
        c_buf.write_f32(&vec![0.0f32; conv_state_size]);

        // Stage the shared scratch bundle BEFORE mutating any state: every
        // allocation must succeed before anything is committed, so a failure
        // leaves the scratch exactly as it was.
        let scratch_bundle = if s.gdn_alpha_buf.is_none() {
            let alloc = |bytes: usize, what: &str| {
                self.device
                    .new_buffer(bytes)
                    .ok_or_else(|| RuntimeError::Compute(format!("Failed to allocate GDN {what}")))
            };
            Some((
                alloc(gdn_num_v_heads * 4, "alpha buf")?,
                alloc(gdn_num_v_heads * 4, "beta buf")?,
                alloc(gdn_q_dim * 4, "output buf")?,
                alloc(hidden_dim * 4, "ssm_proj buf")?,
                alloc(gdn_q_dim * 4, "gate sigmoid buf")?,
                alloc(gdn_q_dim * 4, "normed_out buf")?,
                alloc(gdn_num_v_heads * 4, "alpha_raw buf")?,
                alloc(gdn_num_v_heads * 4, "beta_raw buf")?,
                alloc(gdn_qkv_dim * 4, "qkv_conv buf")?,
            ))
        } else {
            None
        };

        // All allocations succeeded — commit state, then the map entry last.
        s.gdn_h_states.push(h_buf);
        // Length-sync the lazy F16 h_state mirror (filled on the first decode
        // touch by the default F16 decode recurrence).
        s.gdn_h_states_f16.push(std::cell::RefCell::new(None));
        s.gdn_conv_states.push(c_buf);
        s.gdn_conv_positions.push(0);
        s.gdn_num_layers = s.gdn_h_states.len();
        if let Some((alpha, beta, output, ssm_proj, gate_sig, normed, alpha_raw, beta_raw, qkv)) =
            scratch_bundle
        {
            s.gdn_alpha_buf = Some(alpha);
            s.gdn_beta_buf = Some(beta);
            s.gdn_output_buf = Some(output);
            s.gdn_ssm_proj_buf = Some(ssm_proj);
            s.gdn_gate_sigmoid_buf = Some(gate_sig);
            s.gdn_normed_out_buf = Some(normed);
            s.gdn_alpha_raw_buf = Some(alpha_raw);
            s.gdn_beta_raw_buf = Some(beta_raw);
            s.gdn_qkv_conv_buf = Some(qkv);
        }
        if s.gdn_layer_idx_map.len() <= layer_idx {
            s.gdn_layer_idx_map.resize(layer_idx + 1, None);
        }
        s.gdn_layer_idx_map[layer_idx] = Some(gdn_idx);
        Ok(gdn_idx)
    }

    /// Create a new Metal compute backend.
    ///
    /// Returns an error if Metal is not available on this system.
    pub fn new() -> Result<Self, RuntimeError> {
        let device = MetalDevice::system_default().ok_or_else(|| {
            RuntimeError::Compute("Metal GPU not available on this system".into())
        })?;

        let queue = device
            .new_command_queue()
            .ok_or_else(|| RuntimeError::Compute("Failed to create Metal command queue".into()))?;

        // Pick up LUMEN_METAL_PROFILE=1 if set in the environment. The
        // CLI's `--profile` flag also routes through `set_profile()`.
        profile::init_from_env();
        decode_profile::init_from_env();

        // Attempt to create a Metal IO command queue (Metal 3 / macOS 13+).
        // This enables direct NVMe-to-GPU DMA for streaming expert loading.
        let metal_io_queue = MetalIOQueue::new(&device);
        // MTLIOCommandQueue availability is observable via MetalF32Backend API if needed.

        Ok(Self {
            device,
            queue,
            pipelines: None,
            embedding_buf: None,
            final_norm_buf: None,
            output_proj_buf: None,
            embedding: Vec::new(),
            final_norm: Vec::new(),
            output_proj: Vec::new(),
            output_proj_raw: None,
            output_proj_quant: QuantScheme::F32,
            embedding_raw: None,
            embedding_quant: QuantScheme::F32,
            weight_tying: false,
            scratch: Mutex::new(None),
            cached_hidden_dim: 0,
            cached_attn_dims: None,
            cached_moe_num_experts: 0,
            cached_vocab_size: 0,
            expert_profiler: None,
            expert_cache: None,
            expert_reader: None,
            lbc_path: None,
            profiling_tokens_remaining: AtomicUsize::new(0),
            profiling_top_k: 0,
            warmup_complete: AtomicBool::new(false),
            cache_bias_lambda: 0.0,
            expert_bytes_from_disk: AtomicU64::new(0),
            expert_bytes_from_cache: AtomicU64::new(0),
            expert_bytes_from_blob: AtomicU64::new(0),
            use_option_a: false,
            prefetch_handle: Mutex::new(None),
            router_debug_enabled: false,
            router_debug_log: Mutex::new(Vec::new()),
            metal_io_queue,
        })
    }

    /// Returns whether expert cache warmup has been completed.
    pub fn is_warmup_complete(&self) -> bool {
        self.warmup_complete.load(Ordering::Relaxed)
    }

    /// Returns a snapshot of expert activation profiler statistics.
    /// Returns None if the model is not MoE or profiler is not initialized.
    pub fn expert_profiler_summary(&self) -> Option<crate::expert::profiler::ProfilerSummary> {
        self.expert_profiler
            .as_ref()
            .map(|p| p.lock().unwrap().summary())
    }

    /// Returns a snapshot of expert cache statistics.
    /// Returns None if expert caching is not configured.
    pub fn expert_cache_stats(&self) -> Option<crate::expert::cache::CacheStats> {
        self.expert_cache
            .as_ref()
            .map(|c| c.lock().unwrap().stats())
    }

    /// Returns cumulative MoE expert I/O byte counters.
    ///
    /// Returns `(bytes_from_disk, bytes_from_cache, bytes_from_blob)`:
    /// - `bytes_from_disk`: loaded via ExpertReader on cache miss (Tier 2)
    /// - `bytes_from_cache`: served from ExpertLfuCache (Tier 1 + Tier 2 hits)
    /// - `bytes_from_blob`: accessed via full layer blob fallback (Tier 3)
    pub fn expert_io_stats(&self) -> (u64, u64, u64) {
        (
            self.expert_bytes_from_disk.load(Ordering::Relaxed),
            self.expert_bytes_from_cache.load(Ordering::Relaxed),
            self.expert_bytes_from_blob.load(Ordering::Relaxed),
        )
    }

    /// Returns whether Metal IO DMA (MTLIOCommandQueue) is available.
    ///
    /// When true, streaming expert cache misses use direct NVMe-to-GPU DMA
    /// instead of pread + CPU copy.
    pub fn has_metal_io_queue(&self) -> bool {
        self.metal_io_queue.is_some()
    }

    /// Returns the accumulated router debug log and clears it.
    ///
    /// Each entry is a `RouterLayerStats` captured from one MoE layer during
    /// one decode token. The log contains entries for ALL MoE layers across
    /// ALL tokens decoded since the last call to this method (or since init).
    pub fn get_router_debug_log(&self) -> Vec<RouterLayerStats> {
        let mut log = self.router_debug_log.lock().unwrap();
        std::mem::take(&mut *log)
    }

    /// Set raw Q8_0 output projection bytes for GPU-native dequant-matmul.
    ///
    /// When called, compute_final() will use the fused dequant_matmul_q8_0
    /// kernel instead of matmul_f32, reducing bandwidth 3.76x.
    /// For quantized output_proj the F32 `self.output_proj` from
    /// set_global_tensors is dead and is freed in `init()` once the GPU
    /// buffer is built — weight tying reuses the embedding GPU buffer offset
    /// and never reads the F32 Vec.
    pub fn set_output_proj_q8(&mut self, raw_bytes: Vec<u8>, quant: QuantScheme) {
        self.output_proj_quant = quant;
        self.output_proj_raw = Some(raw_bytes);
    }

    /// Get the device name (for diagnostics).
    pub fn device_name(&self) -> String {
        self.device.name()
    }

    /// Current Metal-driver-reported allocated bytes for THIS
    /// backend's MTLDevice (in-process).
    ///
    /// Returns the same figure as `MTLDevice.currentAllocatedSize` — total
    /// bytes outstanding for all MTLBuffer / MTLTexture / MTLHeap objects
    /// the process holds. On Apple Silicon unified memory this is the
    /// authoritative measure of GPU residency; on the soak-harness host
    /// process it captures the lumen-server's own footprint (an external
    /// Swift probe would see its own process's allocations, not the
    /// server's).
    ///
    /// Used by the `/debug/memory_breakdown` server endpoint to
    /// distinguish "Metal driver state" growth from "Rust heap state"
    /// growth in the long-session RSS leak root-cause hunt.
    pub fn current_allocated_bytes(&self) -> u64 {
        self.device.current_allocated_size()
    }

    /// Upload f32 data to a GPU buffer.
    fn upload_f32(&self, data: &[f32]) -> Result<MetalBuffer, RuntimeError> {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        self.device
            .new_buffer_with_bytes(bytes)
            .ok_or_else(|| RuntimeError::Compute("Failed to create Metal buffer".into()))
    }

    /// Create a zero-copy MetalBuffer wrapping the entire layer blob.
    ///
    /// On Apple Silicon, mmap'd data is in unified memory shared between CPU and GPU.
    /// `MTLBuffer(bytesNoCopy:)` wraps it without copying -- the GPU accesses the same
    /// physical pages. Subtensors within the blob are accessed via buffer offsets in
    /// `set_buffer(&buf, offset, index)`.
    ///
    /// # Page alignment
    ///
    /// `bytesNoCopy` requires page-aligned pointers (4096 bytes on Apple Silicon).
    /// mmap'd data is always page-aligned. If the pointer is NOT page-aligned
    /// (heap-allocated LayerView from async provider), we fall back to
    /// `new_buffer_with_bytes` which copies.
    fn create_layer_buffer(&self, weights: &LayerView) -> Result<MetalBuffer, RuntimeError> {
        // Streaming / non-resident paths reach the dispatch kernels without
        // GPU-resident preload; this is their once-per-layer validation point.
        gpu_resident::validate_layer_quants(weights.layer_idx, &weights.subtensors)?;
        let dims = self.cached_attn_dims.ok_or_else(|| {
            RuntimeError::Compute("Metal attention dims not set: call init() first".into())
        })?;
        gpu_resident::validate_attention_dims(weights.layer_idx, &weights.subtensors, &dims)?;
        lumen_format::serving_rules::validate_expert_count(
            &weights.subtensors,
            self.cached_moe_num_experts,
        )
        .map_err(|e| RuntimeError::Compute(format!("layer {}: {e}", weights.layer_idx)))?;
        let blob = weights.as_bytes();
        let ptr = blob.as_ptr();
        let len = blob.len();

        if len == 0 {
            return self.device.new_buffer(4).ok_or_else(|| {
                RuntimeError::Compute("Failed to create empty layer buffer".into())
            });
        }

        // Check page alignment for zero-copy path
        if (ptr as usize) % PAGE_SIZE == 0 {
            // Page-aligned: use bytesNoCopy (zero-copy).
            // Round length up to page boundary as required by Metal.
            let aligned_len = (len + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

            // SAFETY: The LayerView's backing memory (mmap) outlives this buffer.
            // The engine holds a borrow on &dyn WeightProvider during generate(),
            // which keeps the mmap alive. The buffer is used only within this
            // compute_layer call and dropped before returning.
            let buf = unsafe {
                self.device
                    .new_buffer_no_copy(ptr as *mut c_void, aligned_len)
            };
            if let Some(buf) = buf {
                return Ok(buf);
            }
            // Fall through to copy path if bytesNoCopy fails (shouldn't happen
            // with page-aligned mmap, but defensive).
        }

        // Not page-aligned (heap data from async provider): copy.
        self.device.new_buffer_with_bytes(blob).ok_or_else(|| {
            RuntimeError::Compute("Failed to create layer buffer (copy fallback)".into())
        })
    }

    /// Create a Metal buffer covering only the non-expert portion of a
    /// MoE layer blob. This avoids page-faulting the expert byte range from mmap,
    /// since expert data will be served from the LFU cache instead.
    ///
    /// `non_expert_end` is the byte offset in the blob where expert data begins.
    /// The returned buffer covers `blob[0..non_expert_end]` (rounded up to page size).
    fn create_partial_layer_buffer(
        &self,
        weights: &LayerView,
        non_expert_end: usize,
    ) -> Result<MetalBuffer, RuntimeError> {
        gpu_resident::validate_layer_quants(weights.layer_idx, &weights.subtensors)?;
        let dims = self.cached_attn_dims.ok_or_else(|| {
            RuntimeError::Compute("Metal attention dims not set: call init() first".into())
        })?;
        gpu_resident::validate_attention_dims(weights.layer_idx, &weights.subtensors, &dims)?;
        lumen_format::serving_rules::validate_expert_count(
            &weights.subtensors,
            self.cached_moe_num_experts,
        )
        .map_err(|e| RuntimeError::Compute(format!("layer {}: {e}", weights.layer_idx)))?;
        let blob = weights.as_bytes();
        let ptr = blob.as_ptr();
        let len = non_expert_end.min(blob.len());

        if len == 0 {
            return self.device.new_buffer(4).ok_or_else(|| {
                RuntimeError::Compute("Failed to create empty partial layer buffer".into())
            });
        }

        // Check page alignment for zero-copy path
        if (ptr as usize) % PAGE_SIZE == 0 {
            // Round length up to page boundary as required by Metal.
            let aligned_len = (len + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
            // Ensure we don't exceed the blob's total length (page-rounded).
            let max_aligned = (blob.len() + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
            let aligned_len = aligned_len.min(max_aligned);

            let buf = unsafe {
                self.device
                    .new_buffer_no_copy(ptr as *mut c_void, aligned_len)
            };
            if let Some(buf) = buf {
                return Ok(buf);
            }
        }

        // Not page-aligned: copy only the non-expert portion.
        self.device
            .new_buffer_with_bytes(&blob[..len])
            .ok_or_else(|| {
                RuntimeError::Compute(
                    "Failed to create partial layer buffer (copy fallback)".into(),
                )
            })
    }

    /// Compute the byte offset where expert data begins in a MoE layer blob.
    ///
    /// Returns the end offset of the last non-expert tensor (attention weights,
    /// norms, router, biases). Everything before this offset is non-expert data;
    /// everything at or after it is expert data. If the layer has no experts,
    /// returns the full blob length.
    fn non_expert_byte_end(st: &lumen_format::index::SubtensorOffsets) -> usize {
        let mut end: u64 = 0;

        // Attention weights
        let slices = [&st.wq, &st.wk, &st.wv, &st.wo, &st.attn_norm, &st.ffn_norm];
        for s in &slices {
            let s_end = s.offset + s.length;
            if s_end > end {
                end = s_end;
            }
        }

        // Dense FFN weights (zero-length sentinels for MoE, but check anyway)
        for s in &[&st.w_gate, &st.w_up, &st.w_down] {
            let s_end = s.offset + s.length;
            if s_end > end {
                end = s_end;
            }
        }

        // Optional biases
        for opt in &[&st.bq, &st.bk, &st.bv] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        // Router weight (non-expert, always loaded)
        if let Some(ref s) = st.router_weight {
            let s_end = s.offset + s.length;
            if s_end > end {
                end = s_end;
            }
        }

        // Shared expert weights (always loaded, non-expert).
        // Qwen3.5-MoE has an always-active shared expert whose gate/up/down
        // weights live in the layer blob alongside attention/norm/router data.
        for opt in &[
            &st.shared_expert_gate,
            &st.shared_expert_up,
            &st.shared_expert_down,
        ] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        // Extended attention fields (always loaded, non-expert).
        // attn_gate, attn_post_norm are per-layer tensors used by hybrid models.
        for opt in &[&st.attn_gate, &st.attn_post_norm] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        // SSM / linear attention fields (always loaded, non-expert).
        // These are per-layer tensors for GatedDeltaNet hybrid layers.
        for opt in &[
            &st.ssm_a,
            &st.ssm_conv1d,
            &st.ssm_dt,
            &st.ssm_beta,
            &st.ssm_alpha,
            &st.ssm_norm,
            &st.ssm_out,
        ] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        // Per-head Q/K RMSNorm weights and shared expert gate input weight.
        for opt in &[&st.attn_q_norm, &st.attn_k_norm, &st.ffn_gate_inp_shexp] {
            if let Some(s) = opt {
                let s_end = s.offset + s.length;
                if s_end > end {
                    end = s_end;
                }
            }
        }

        end as usize
    }

    /// Dispatch a matmul_bytes_f32 kernel: out = W_bytes * x
    ///
    /// Note: Not used by the optimized compute_layer (which inlines encoding into
    /// batched command buffers with zero-copy offsets), but retained for testing
    /// and for potential use by future code paths.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_matmul_bytes(
        &self,
        pipelines: &MetalPipelines,
        w_bytes: &[u8],
        x_buf: &MetalBuffer,
        out_buf: &MetalBuffer,
        out_dim: usize,
        in_dim: usize,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        // Create a buffer wrapping the weight bytes (copy for safety)
        let w_buf = self.device.new_buffer_with_bytes(w_bytes).ok_or_else(|| {
            RuntimeError::Compute("Failed to create weight buffer for matmul".into())
        })?;

        let in_dim_u32 = in_dim as u32;

        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for matmul".into())
        })?;
        let enc = cmd.new_compute_encoder().ok_or_else(|| {
            RuntimeError::Compute("Failed to create compute encoder for matmul".into())
        })?;

        enc.set_pipeline_state(&pipelines.matmul_bytes_f32);
        enc.set_buffer(&w_buf, 0, 0);
        enc.set_buffer(x_buf, 0, 1);
        enc.set_buffer(out_buf, 0, 2);
        enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
        enc.dispatch_threadgroups(
            MTLSize::new(out_dim as u64, 1, 1),
            MTLSize::new(scratch.matmul_tg_size, 1, 1),
        );
        enc.end_encoding();
        cmd.commit_and_wait();

        Ok(())
    }

    /// Dispatch a dequant_matmul_q8_0 kernel: out = dequant(W_q8) * x
    ///
    /// The kernel performs fused Q8_0 dequantization and matrix-vector multiply.
    /// `in_dim` is the element count (not byte stride). The kernel computes the
    /// Q8_0 row byte stride internally from `in_dim`.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_matmul_q8_0(
        &self,
        pipelines: &MetalPipelines,
        w_bytes: &[u8],
        x_buf: &MetalBuffer,
        out_buf: &MetalBuffer,
        out_dim: usize,
        in_dim: usize,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        let w_buf = self.device.new_buffer_with_bytes(w_bytes).ok_or_else(|| {
            RuntimeError::Compute("Failed to create weight buffer for Q8_0 matmul".into())
        })?;

        let in_dim_u32 = in_dim as u32;

        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for Q8_0 matmul".into())
        })?;
        let enc = cmd.new_compute_encoder().ok_or_else(|| {
            RuntimeError::Compute("Failed to create compute encoder for Q8_0 matmul".into())
        })?;

        enc.set_pipeline_state(&pipelines.dequant_matmul_q8_0);
        enc.set_buffer(&w_buf, 0, 0);
        enc.set_buffer(x_buf, 0, 1);
        enc.set_buffer(out_buf, 0, 2);
        enc.set_bytes(&in_dim_u32.to_le_bytes(), 3);
        enc.dispatch_threadgroups(
            MTLSize::new(out_dim as u64, 1, 1),
            MTLSize::new(scratch.matmul_tg_size, 1, 1),
        );
        enc.end_encoding();
        cmd.commit_and_wait();

        Ok(())
    }

    /// Dispatch the appropriate matmul kernel based on quantization scheme.
    ///
    /// For Q8_0 weights, uses the fused `dequant_matmul_q8_0` kernel.
    /// For F32/unquantized weights, uses `matmul_bytes_f32` (cast uchar* to float*).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_matmul_for_quant(
        &self,
        pipelines: &MetalPipelines,
        w_bytes: &[u8],
        x_buf: &MetalBuffer,
        out_buf: &MetalBuffer,
        out_dim: usize,
        in_dim: usize,
        quant: QuantScheme,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        match quant {
            QuantScheme::Q8_0 => self
                .dispatch_matmul_q8_0(pipelines, w_bytes, x_buf, out_buf, out_dim, in_dim, scratch),
            _ => self.dispatch_matmul_bytes(
                pipelines, w_bytes, x_buf, out_buf, out_dim, in_dim, scratch,
            ),
        }
    }

    /// Dispatch rmsnorm_bytes kernel.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_rmsnorm_bytes(
        &self,
        pipelines: &MetalPipelines,
        x_buf: &MetalBuffer,
        w_bytes: &[u8],
        out_buf: &MetalBuffer,
        dim: usize,
        eps: f32,
        scratch: &MetalScratch,
    ) -> Result<(), RuntimeError> {
        let w_buf = self.device.new_buffer_with_bytes(w_bytes).ok_or_else(|| {
            RuntimeError::Compute("Failed to create weight buffer for rmsnorm".into())
        })?;
        let dim_u32 = dim as u32;

        let cmd = self.queue.new_command_buffer().ok_or_else(|| {
            RuntimeError::Compute("Failed to create command buffer for rmsnorm".into())
        })?;
        let enc = cmd.new_compute_encoder().ok_or_else(|| {
            RuntimeError::Compute("Failed to create compute encoder for rmsnorm".into())
        })?;

        enc.set_pipeline_state(&pipelines.rmsnorm_bytes);
        enc.set_buffer(x_buf, 0, 0);
        enc.set_buffer(&w_buf, 0, 1);
        enc.set_buffer(out_buf, 0, 2);
        enc.set_bytes(&dim_u32.to_le_bytes(), 3);
        enc.set_bytes(&eps.to_le_bytes(), 4);
        enc.dispatch_threadgroups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(scratch.norm_tg_size, 1, 1),
        );
        enc.end_encoding();
        cmd.commit_and_wait();

        Ok(())
    }
}

#[cfg(test)]
mod tests;
