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

pub(crate) mod ffi;
// Apple MPSGraph BF16 GEMM bindings (env-gated opt-in path).
// Provides `MpsGraphContext` + `encode_bf16_matmul_to_command_buffer`
// for the GDN qkv-proj and ssm-out matmuls when
// `LUMEN_METAL_BF16_MPS=1`. Default OFF.
mod decode_greedy;
pub(crate) mod decode_profile;
mod decode_single_cb;
pub(crate) mod io;
pub(crate) mod mps_graph_ffi;
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
// ssm_out_gemm microbench: standalone perf harness for the GDN hot-spot.
// Exposes a single `#[doc(hidden)] pub fn run_ssm_out_microbench` entry
// point; no production code path touches this module.
pub mod ssm_out_microbench;
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
/// the env var is unset (server → 50 µs, CLI → 0). An explicit env value
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
/// When the resolved delay is `0` this is a single integer load + branch with
/// no syscall — the path stays bit-exact and the only cost is one comparison.
/// When non-zero it issues a CPU `thread::sleep` for the configured number of
/// microseconds, which stabilises the GPU-scheduler wall-clock window that
/// otherwise lets a near-tie top-1/top-2 logit pair resolve differently across
/// repeated in-process decode calls (greedy-decode determinism guard).
#[inline(always)]
fn maybe_apply_metal_decode_delay() {
    let delay_us = metal_decode_delay_us();
    if delay_us > 0 {
        std::thread::sleep(std::time::Duration::from_micros(delay_us));
    }
}

/// Env-var gated opt-in: use the ggml-metal-ported Q8_0 × F32 GEMM kernel.
///
/// When `LUMEN_METAL_GEMM_GGML_PORT=1` (or any non-empty value), Lumen swaps
/// the Q8_0 prefill GEMM dispatch from the in-tree `dequant_tiled_matmul_q8_0_k64`
/// to the upstream-derived `kernel_mul_mm_q8_0_f32_ported` (see
/// `shaders/gemm_q8_0_ported.msl`). Default OFF.
///
/// Resolved once at startup (atomic) so dispatch sites don't pay an env lookup.
#[inline]
pub(crate) fn use_ggml_ported_q8_0_gemm() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    // 0 = unknown, 1 = false, 2 = true
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_GEMM_GGML_PORT")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_Q4_GATEUP_WIDE=1` (and the decode-qmv gate/up buffers
/// exist, i.e. `LUMEN_METAL_Q4_QMV_GATEUP=1`), the dense FFN gate/up GEMV uses
/// the WIDE-load kernel `rmsnorm_ffn_gate_up_swiglu_q4_0_wide` (256-thread,
/// uint4/128-bit aligned loads off the separated sequential-nibble layout)
/// instead of the 64-thread `qmv_q4_0_gate_up_swiglu`. Default OFF. Cached.
pub(crate) fn q4_gateup_wide_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_Q4_GATEUP_WIDE")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
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

/// Master switch for the four byte-identical F16-SCALES matvec paths
/// (`LUMEN_METAL_Q4_F16_SCALES_ALL=1`, default OFF). The gate/up, FFN-down,
/// lm_head, and GDN-QKV/gate f16-scale kernels each cut ~10% of their matvec's
/// per-block scale bytes and are byte-identical because the f16 scale is the
/// on-disk Q4_0 native precision the f32 decode-qmv layout widened (see the four
/// per-path getters below). This master ORs into all four per-path getters at
/// once so a single flag engages the full f16-scale weight-byte reduction across
/// every Q4 decode matvec stream. It does NOT touch the full-attn path
/// (`LUMEN_METAL_Q4_FULLATTN_F16SC`). Default OFF. Cached.
pub(crate) fn q4_f16_scales_all_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_Q4_F16_SCALES_ALL")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_Q4_GATEUP_F16SC=1` (and the decode-qmv gate/up buffers
/// exist, i.e. `LUMEN_METAL_Q4_QMV_GATEUP=1`), the dense FFN gate/up GEMV uses
/// the F16-SCALES kernel `qmv_q4_0_gate_up_swiglu_f16sc` with gate/up scale
/// buffers built as f16 (2 B/block instead of f32's 4 B). This streams 18 (not
/// 20) weight bytes per 32-value block = ~10% fewer bytes on the bandwidth-bound
/// dense FFN gate/up matvec. The f16 scale is the on-disk Q4_0 scale's native
/// precision (the f32 path widened it), so the result is byte-identical to the
/// f32-scale kernel. Also engaged by the master `LUMEN_METAL_Q4_F16_SCALES_ALL`.
/// Default OFF. Cached.
pub(crate) fn q4_gateup_f16sc_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = q4_f16_scales_all_enabled()
        || q4_gateup_h2math_enabled()
        || std::env::var("LUMEN_METAL_Q4_GATEUP_F16SC")
            .map(|s| !s.is_empty() && s != "0")
            .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_Q4_GATEUP_1SG=1` (and the dense FFN gate/up decode-qmv
/// buffers exist via `LUMEN_METAL_Q4_QMV_GATEUP=1`, AND the f16-scales gate/up
/// path is engaged so the f16 scale buffers are built — `LUMEN_METAL_Q4_GATEUP_F16SC`
/// or the master `LUMEN_METAL_Q4_F16_SCALES_ALL`), the dense FFN gate/up GEMV uses
/// the 1-SIMDGROUP-PER-THREADGROUP kernel `qmv_q4_0_gate_up_swiglu_f16sc_1sg`
/// (32 threads/TG, 4 rows/TG, dispatched over inter_dim/4 threadgroups) instead
/// of the 2-SG kernel (64 threads/TG, 8 rows/TG, inter_dim/8 TGs). This DOUBLES
/// the resident threadgroup count to deepen the wavefront queue and hide HBM
/// latency on the under-occupied gate/up matvec (~55-67% of bandwidth peak vs
/// lm_head's ~84%), while keeping the per-simdgroup 4-rows x-register-reuse intact.
/// Byte-identical to the f16sc kernel (same per-row math + same
/// per-simdgroup RMSNorm reduction; only the TG/SG partition changes). Perf lever,
/// default OFF. Cached.
pub(crate) fn q4_gateup_1sg_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_Q4_GATEUP_1SG")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_Q4_GATEUP_8ROW=1` (and the dense FFN gate/up decode-qmv
/// buffers exist via `LUMEN_METAL_Q4_QMV_GATEUP=1`, AND the f16-scales gate/up
/// path is engaged so the f16 scale buffers are built — `LUMEN_METAL_Q4_GATEUP_F16SC`
/// or the master `LUMEN_METAL_Q4_F16_SCALES_ALL`), the dense FFN gate/up GEMV uses
/// the 8-ROWS-PER-SIMDGROUP kernel `qmv_q4_0_gate_up_swiglu_f16sc_8row` (2 SG/TG,
/// 8 rows/SG = 16 rows/TG, dispatched over inter_dim/16 threadgroups) instead of
/// the 4-row f16sc kernel (8 rows/TG, inter_dim/8 TGs). This HALVES the threadgroup
/// count (9B: 1536 -> 768, still ~12.8/core on the 60-core M3 Ultra = richly
/// occupied) while DOUBLING x-register reuse: each register-staged normed-x value
/// (and the shared sumx + the RMSNorm ss accumulation) is reused across 8 rows x
/// 2 matrices = 16 MACs vs 8, lifting arithmetic intensity per fetched activation
/// byte. The gate/up matvec is over-subscribed (occupancy is NOT its limiter — the
/// 1SG MORE-TGs variant was flat) yet runs only ~55-67% of bandwidth peak vs
/// lm_head's ~84%; this attacks the per-byte-work side. Byte-identical per output
/// element to the f16sc kernel (each row's FP add order + simd_sum reduction are
/// unchanged; only the row<->simdgroup assignment differs). Requires
/// inter_dim % 16 == 0 (9B inter=12288 OK). Perf lever, default OFF. Cached.
pub(crate) fn q4_gateup_8row_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_Q4_GATEUP_8ROW")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_Q4_GATEUP_IL=1` (and the dense FFN gate/up decode-qmv buffers
/// exist via `LUMEN_METAL_Q4_QMV_GATEUP=1`), the dense FFN gate/up GEMV uses the
/// INTERLEAVED kernel `qmv_q4_0_gate_up_swiglu_il` reading ONE co-resident packed
/// buffer instead of four separate streams (w_gate, w_up, gate_scales, up_scales).
///
/// MECHANISM (roofline "coalescing — find the real limiter"): the f16sc gate/up
/// kernel issues, per 32-value block, FOUR distinct device-memory accesses from
/// FOUR buffers at four base addresses — gate nibbles, up nibbles, gate f16 scale,
/// up f16 scale. The dense FFN gate/up matvec is the DOMINANT per-token weight
/// stream (~1.81 GB/tok across 32 layers x 2 matrices) yet runs only ~55-67% of
/// bandwidth peak (vs lm_head's ~84%). Four interleaved streams under-utilise the
/// memory subsystem (more in-flight cache-line/TLB demand streams, weaker
/// prefetch) vs ONE contiguous stream. This packs, per output row, blocks
/// sequentially as `[half g_scale, half u_scale, 16B gate_nibbles, 16B up_nibbles]`
/// = 36 contiguous bytes/block, so each thread's per-block fetch is ONE coalesced
/// region. Same bytes moved, same nibble math, same -8 fold, same per-simdgroup
/// RMSNorm ss reduce + SwiGLU -> BYTE-IDENTICAL to qmv_q4_0_gate_up_swiglu_f16sc.
///
/// Engages only when the IL pipeline compiled AND the interleaved buffers were
/// built (preload_weights_gpu_resident, gated on this flag); otherwise the
/// dispatch falls back cleanly to the existing f16sc / 8row / default paths.
/// Requires in_dim % 512 == 0 and inter_dim % 8 == 0 (9B: in=4096, out=12288 OK).
/// Perf lever, default OFF. Cached.
pub(crate) fn q4_gateup_il_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_Q4_GATEUP_IL")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_Q4_QMV_DOWN_F16SC=1` (and the FFN-down decode-qmv buffers
/// exist, i.e. `LUMEN_METAL_Q4_QMV_DOWN=1`), the dense FFN-down GEMV uses the
/// F16-SCALES kernel `qmv_q4_0_residual_f16sc` with the per-block scale buffer
/// built as f16 (2 B/block instead of f32's 4 B). FFN-down is the longest-K
/// matvec (in=12288 -> 384 blocks/row), so this is the single largest per-token
/// scale stream: streaming 18 (not 20) weight bytes per 32-value block = ~10%
/// fewer bytes on the bandwidth-bound FFN-down matvec. The f16 scale is the
/// on-disk Q4_0 scale's native precision (the f32 decode-qmv layout widened it),
/// so the result is byte-identical to the f32-scale kernel. Default OFF. Cached.
pub(crate) fn q4_qmv_down_f16sc_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = q4_f16_scales_all_enabled()
        || q4_down_h2math_enabled()
        || std::env::var("LUMEN_METAL_Q4_QMV_DOWN_F16SC")
            .map(|s| !s.is_empty() && s != "0")
            .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
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

/// When `LUMEN_METAL_Q4_GATEUP_F16MATH=1` (and the dense FFN gate/up f16-scale path
/// is engaged — `q4_gateup_f16sc_enabled`, which the master `LUMEN_METAL_Q4_F16_SCALES_ALL`
/// turns on — AND the f16math kernel compiled), the dense FFN gate/up GEMV uses
/// `qmv_q4_0_gate_up_swiglu_f16sc_f16math`: x is staged as `half` and each per-32-block
/// dequant MAC (the 16-term nibble*x dot, for BOTH gate and up) is accumulated in
/// `half`, with the cross-block reduction + sum-of-x + scale + RMSNorm + SwiGLU kept
/// in f32.
///
/// RATIONALE: the same compute-bound-unpack argument as the FFN-down h2math path,
/// applied to the DOMINANT FFN matvec. The dense FFN gate/up is the largest single
/// matvec pool in decode (gate+up = 100M weights/layer x 32 dense layers, ~2x the
/// GDN qkv) and sits at the single-kernel bandwidth/compute ceiling, so halving the
/// per-nibble dequant ALU on the matvec that matters most is the remaining compute
/// lever. NEAR-TIE, not guaranteed byte-identical (per-block product+sum rounds to
/// f16 mantissa); the f32 cross-block accumulation + f32 SwiGLU bound the drift.
/// Default OFF. Cached.
pub(crate) fn q4_gateup_f16math_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_Q4_GATEUP_F16MATH")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
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
/// tensor (the Q8 output_proj re-quantized to Q4 under `LUMEN_METAL_Q4_QMV_LMHEAD`,
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

/// When `LUMEN_METAL_Q4_SSMOUT_REQUANT=1` (and `LUMEN_METAL_Q4_QMV_SSMOUT=1`), the
/// GDN `ssm_out` output projection — which ships as Q8_0 on Qwen3.5-9B (so the
/// native-Q4 ssm_out qmv path is otherwise INERT, building 0 buffers) — is
/// RE-QUANTIZED Q8_0 -> Q4_0 at load time and the decode-qmv buffers are built
/// from the Q4 result, so the existing `qmv_q4_0_residual` ssm_out dispatch
/// engages on all 24 GDN layers. ssm_out (in=value_dim, out=hidden) is the last
/// major per-token Q8_0 weight stream on the GDN decode path (~Q8 8.5 bits/weight
/// over 24 layers); halving it to Q4 (~4.5 bits) is a ~2% bytes-moved cut on the
/// bandwidth-bound decode. Like the lm_head Q8->Q4 requant
/// (`LUMEN_METAL_Q4_QMV_LMHEAD`), this is a deliberate precision tradeoff (NOT
/// byte-identical to the Q8 ssm_out NR2 path) — accepted only if the correctness
/// gate keeps the answer (byte-identical OR near-tie). Default OFF. Cached.
pub(crate) fn q4_ssmout_requant_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_Q4_SSMOUT_REQUANT")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
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
/// This is the CORRECTED form of the SUPERSEDED `LUMEN_METAL_Q4_SSMOUT_REQUANT`
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

/// When `LUMEN_METAL_Q4_LMHEAD_F16SC=1` (and the Q4 lm_head decode-qmv buffers
/// exist, i.e. `LUMEN_METAL_Q4_QMV_LMHEAD=1` re-quantized the Q8 output_proj),
/// the lm_head / output-projection GEMV uses the F16-SCALES kernel
/// `qmv_q4_0_rmsnorm_f16sc` with the per-block scale buffer built as f16 (2 B/block
/// instead of f32's 4 B). The lm_head is the single LARGEST weight tensor
/// (out=vocab), so its scale stream is the biggest single-tensor scale stream in
/// the model: streaming 18 (not 20) weight bytes per 32-value block = ~10% fewer
/// bytes on the bandwidth-bound lm_head matvec. The f16 scale is the on-disk Q4_0
/// scale's native precision (the f32 decode-qmv layout widened it), so the result
/// is byte-identical to the f32-scale kernel. Default OFF. Cached.
pub(crate) fn q4_lmhead_f16sc_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = q4_f16_scales_all_enabled()
        || q4_lmhead_h2math_enabled()
        || std::env::var("LUMEN_METAL_Q4_LMHEAD_F16SC")
            .map(|s| !s.is_empty() && s != "0")
            .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_Q4_PROJ_F16SC=1` (and the GDN decode-qmv QKV-in-proj +
/// attn_gate buffers exist, i.e. `LUMEN_METAL_Q4_QMV_PROJ=1`), the GDN-layer
/// QKV-in-projection (`wq`, qkv_dim rows) AND attn_gate (q_dim rows) GEMVs use
/// the F16-SCALES kernel `qmv_q4_0_rmsnorm_f16sc` with their per-block scale
/// buffers built as f16 (2 B/block instead of f32's 4 B). These two matvecs run
/// on EVERY GDN layer (the majority of layers in the Qwen3.5-9B GatedDeltaNet),
/// so their combined per-token scale stream (12288 rows x hidden/32 blocks) is
/// comparable in magnitude to the FFN-down scale stream: streaming 18 (not 20)
/// weight bytes per 32-value block = ~10% fewer bytes on these bandwidth-bound
/// matvecs. The f16 scale is the on-disk Q4_0 scale's native precision (the f32
/// decode-qmv layout widened it), so the result is byte-identical to the
/// f32-scale kernel. Default OFF. Cached.
pub(crate) fn q4_proj_f16sc_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = q4_f16_scales_all_enabled()
        || q4_proj_h2math_enabled()
        || std::env::var("LUMEN_METAL_Q4_PROJ_F16SC")
            .map(|s| !s.is_empty() && s != "0")
            .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_Q4_FULLATTN_F16SC=1` (and the full-attn decode-qmv buffers
/// exist, i.e. `LUMEN_METAL_Q4_QMV_PROJ=1` for Q+gate/Wo and
/// `LUMEN_METAL_Q4_QMV_KV=1` for K/V), the FULL-ATTENTION-layer projections —
/// Q+gate (`wq`, 2*q_dim rows), K (`wk`), V (`wv`) all via
/// `qmv_q4_0_rmsnorm_f16sc`, and the output projection Wo (`wo`) via
/// `qmv_q4_0_residual_f16sc` — read their per-block scale buffers built as f16
/// (2 B/block instead of f32's 4 B). This is the LAST remaining f32-scale matvec
/// stream: `LUMEN_METAL_Q4_PROJ_F16SC` already covers the 24 GDN layers, while
/// these run on the 8 full-attention layers of the Qwen3.5-9B GatedDeltaNet
/// hybrid. The f16 scale is the on-disk Q4_0 scale's native precision (the f32
/// decode-qmv layout widened it; the f16sc repack copies the on-disk f16 bytes
/// verbatim), so the result is BYTE-IDENTICAL to the f32-scale kernels — only the
/// streamed weight bytes drop from 20 to 18 per 32-value block (~10% fewer bytes
/// on these bandwidth-bound matvecs). Reuses the existing `qmv_q4_0_rmsnorm_f16sc`
/// and `qmv_q4_0_residual_f16sc` kernels (no new MSL). Default OFF. Cached.
pub(crate) fn q4_fullattn_f16sc_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_Q4_FULLATTN_F16SC")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_Q4_GATEUP_UNFUSED=1` (and the decode-qmv gate/up buffers
/// exist), the dense FFN gate/up runs as TWO separate single-matrix
/// `qmv_q4_0_rmsnorm` GEMVs (gate -> gate_buf, up -> up_buf) + a standalone
/// `swiglu`, instead of the fused dual-matrix kernel. The single-matrix qmv has
/// HALF the register pressure of the dual-matrix one (proven 84% peak on
/// lm_head), so it may beat the 8row on the dense FFN. Default OFF. Cached.
pub(crate) fn q4_gateup_unfused_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_Q4_GATEUP_UNFUSED")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_Q4_GATEUP_BAREQMV=1` (and the decode-qmv gate/up buffers
/// exist), the dense FFN gate/up runs as: RMSNorm the FFN input ONCE
/// (`rmsnorm_bytes`, attn_proj_buf -> normed_buf) -> a BARE single-matrix
/// `qmv_q4_0` GEMV on the pre-normed x for gate (-> gate_buf) -> a bare
/// `qmv_q4_0` for up (-> up_buf) -> a standalone `swiglu`. This removes the
/// redundant per-matrix RMSNorm that the fused `qmv_q4_0_rmsnorm` gate/up path
/// recomputes inside BOTH gate and up. Default OFF. Cached.
pub(crate) fn q4_gateup_bareqmv_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_Q4_GATEUP_BAREQMV")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_Q4_GATEUP_BAREQMV_256=1`, the bare-qmv gate/up path (see
/// `q4_gateup_bareqmv_enabled`) selects the 256-thread `qmv_q4_0_8sg` kernel
/// (8 simdgroups/threadgroup, K split across the 32 lanes) instead of the
/// 64-thread `qmv_q4_0`. Only consulted when BAREQMV is on. Default OFF. Cached.
pub(crate) fn q4_gateup_bareqmv_256_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_Q4_GATEUP_BAREQMV_256")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_Q4_QMV_DOWN_SPLITK=N` (N in {2,4,8}, default OFF/0), the Q4
/// FFN-down decode matvec uses the two-pass deterministic SPLIT-K kernels
/// (`qmv_q4_0_splitk_partial` + `qmv_q4_0_splitk_reduce`) with N K-slices instead
/// of the one-pass `qmv_q4_0_residual`. Splits the K=12288 contraction across N×
/// more threadgroups to raise memory-level parallelism for the row-starved down
/// projection. Requires the down qmv buffers (LUMEN_METAL_Q4_QMV_DOWN=1) AND
/// in_dim % (512*N) == 0. Returns the K-split count (0 = disabled). Perf lever,
/// default-OFF. NOT byte-identical (different FP reduction tree); quality-clean.
pub(crate) fn q4_qmv_down_splitk() -> u32 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return (cur - 1) as u32;
    }
    let n = std::env::var("LUMEN_METAL_Q4_QMV_DOWN_SPLITK")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&n| n == 2 || n == 4 || n == 8)
        .unwrap_or(0);
    CACHE.store((n as u8) + 1, Ordering::Relaxed);
    n
}

/// When `LUMEN_METAL_Q4_GATEUP_SPLITK=N` (N in {2,4}, default OFF/0), the Q4 dense
/// FFN gate/up projection uses a two-pass deterministic SPLIT-K: a pre-pass RMSNorm
/// of x, then `qmv_q4_0_splitk_partial` run TWICE (gate, up) over N K-slices each,
/// then `gateup_splitk_reduce_swiglu` (fixed-order reduce + SwiGLU). Splits the
/// K=4096 contraction across N× more threadgroups to raise memory-level parallelism
/// for the biggest decode pool. Requires the gate/up qmv buffers
/// (LUMEN_METAL_Q4_QMV_GATEUP=1) AND hidden % (512*N) == 0. Returns the K-split
/// count (0 = disabled). Perf lever, default-OFF. NOT byte-identical (different FP
/// reduction tree); quality-clean.
pub(crate) fn q4_gateup_splitk() -> u32 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return (cur - 1) as u32;
    }
    let n = std::env::var("LUMEN_METAL_Q4_GATEUP_SPLITK")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&n| n == 2 || n == 4)
        .unwrap_or(0);
    CACHE.store((n as u8) + 1, Ordering::Relaxed);
    n
}

/// When `LUMEN_METAL_Q4_SSMOUT_SPLITK=N` (N in {2,4,8}, default OFF/0), the Q4 GDN
/// ssm_out projection uses the two-pass deterministic SPLIT-K kernels
/// (`qmv_q4_0_splitk_partial` + `qmv_q4_0_splitk_reduce`, zero residual) with N
/// K-slices instead of the one-pass `qmv_q4_0_residual`. Splits the K=4096(q_dim)
/// contraction across N× more threadgroups to raise memory-level parallelism for
/// the row-starved ssm_out projection (out=2048, ~8.5 SG/core at N=1). Requires the
/// ssm_out qmv buffers (LUMEN_METAL_Q4_QMV_SSMOUT=1) AND in_dim % (512*N) == 0.
/// Returns the K-split count (0 = disabled). Perf lever, default-OFF. NOT
/// byte-identical (different FP reduction tree); quality-clean.
pub(crate) fn q4_ssmout_splitk() -> u32 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return (cur - 1) as u32;
    }
    let n = std::env::var("LUMEN_METAL_Q4_SSMOUT_SPLITK")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&n| n == 2 || n == 4 || n == 8)
        .unwrap_or(0);
    CACHE.store((n as u8) + 1, Ordering::Relaxed);
    n
}

/// When `LUMEN_METAL_Q4_KV_FUSE=1` (default OFF), the full-attn K and V projections
/// (each [kv_dim=1024, hidden=4096] Q4_0, ~4.3 SG/core separately) are computed by a
/// SINGLE fused qmv dispatch over the concatenated [2*kv_dim, hidden] weight (256 TGs
/// = ~8.5 SG/core), raising effective occupancy for the worst-occupancy matvecs in
/// the model. Requires the fused K|V qmv buffer (built at load under the same flag).
/// Byte-identical (same per-row accumulation, merged grid). Perf lever, default-OFF.
pub(crate) fn q4_kv_fuse_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let on = std::env::var("LUMEN_METAL_Q4_KV_FUSE")
        .map(|v| v == "1")
        .unwrap_or(false);
    CACHE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// When `LUMEN_METAL_Q4_QGATEKV_FUSE=1` (default OFF), the full-attn Q+gate, K AND
/// V projections (wq [qgate_dim=8192, hidden=4096], wk/wv [kv_dim=1024, hidden=4096],
/// all Q4_0) are computed by a SINGLE fused qmv dispatch over the concatenated
/// [qgate_dim + 2*kv_dim] row space (1280 TGs = ~42.7 SG/core for Qwen3.5-9B, well
/// past the ~24-32 occupancy knee that the separate dispatches and the K/V-only
/// fusion (~8.5 SG/core) cannot reach). Byte-identical to the three separate
/// qmv_q4_0_rmsnorm dispatches (same per-row RMSNorm + -8 fold + accumulation order,
/// merged grid). Setting this flag ALSO triggers the load-time build of the wq, wk
/// AND wv decode-qmv buffers (so the user need only set this ONE flag, not also
/// LUMEN_METAL_Q4_QMV_PROJ + LUMEN_METAL_Q4_QMV_KV). Perf lever, default-OFF.
pub(crate) fn q4_qgatekv_fuse_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let on = std::env::var("LUMEN_METAL_Q4_QGATEKV_FUSE")
        .map(|v| v == "1")
        .unwrap_or(false);
    CACHE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
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

/// When `LUMEN_METAL_FLASH_DECODE_ALWAYS=1` (default OFF), the full-attention decode
/// path uses the online-softmax `flash_decode_attention` + `flash_decode_reduce`
/// kernels at ALL KV lengths, instead of falling back to `multi_head_attention`
/// (MHA) below the 257 threshold. RATIONALE (diagnostic/perf lever): the MHA kernel
/// materializes the per-head score vector to DEVICE memory and re-reads it twice
/// (DRAM round-trip), and runs only `num_heads` (32) threadgroups = ~0.5 TG/core =
/// occupancy-starved. Flash keeps scores in registers (online softmax, no device
/// scratch) and splits over `num_tiles` TGs. NOTE: flash and MHA differ in FP
/// reduction order, so this is NOT byte-identical below 257 (the >=257 path already
/// uses flash in the byte-id baseline). A perf POC: measure first, then assess
/// quality. VALIDATED NEGATIVE: measured +6-7% slower on full_attn at the decode KV
/// lengths (<257), because single-tile flash adds a reduce dispatch + a partial-buffer
/// device round-trip that MHA's single dispatch avoids. Kept gated-off as an A/B
/// substrate; MHA is the optimal kernel for short-KV decode attention.
pub(crate) fn flash_decode_always_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let on = std::env::var("LUMEN_METAL_FLASH_DECODE_ALWAYS")
        .map(|v| v == "1")
        .unwrap_or(false);
    CACHE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// When `LUMEN_METAL_FUSED_SDPA_DECODE=1` (default OFF), the short-KV
/// (new_seq_len < 257) full-attention decode path uses
/// `fused_rope_kv_mha_tgscores` instead of `fused_rope_kv_mha`: the only
/// difference is that the per-head attention score vector lives in THREADGROUP
/// memory rather than the device `mha_scores_buf` scratch. The baseline kernel
/// touches that device score vector five times per head (write raw dot, read for
/// softmax-max, write exp, read for sum, write normalized, read for V-weighted
/// sum); for decode the vector is tiny (<=256 f32) and fits on-chip, so this
/// eliminates the transient score buffer's DRAM round-trips -- matching MLX's
/// vendor SDPA, which never round-trips a score buffer to device. The arithmetic,
/// thread ownership, softmax reduction, and V accumulation order are unchanged =>
/// byte-identical to the baseline; only the score storage location moves
/// device->threadgroup. Engages only on the `use_fused_rope_kv_mha` path
/// (standard RoPE, not NeoX, new_seq_len < 257).
pub(crate) fn fused_sdpa_decode_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let on = std::env::var("LUMEN_METAL_FUSED_SDPA_DECODE")
        .map(|v| v == "1")
        .unwrap_or(false);
    CACHE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// When `LUMEN_METAL_CONCURRENT_GATEUP=1` (default OFF, requires the decode-qmv
/// gate/up buffers), the dense FFN gate/up runs as: RMSNorm the FFN input ONCE
/// (`rmsnorm_bytes`, attn_proj_buf -> normed_buf) on the serial encoder, then a
/// pair of BARE single-matrix `qmv_q4_0` GEMVs (gate -> gate_buf, up -> up_buf)
/// dispatched on a CONCURRENT encoder, then a resource-scoped barrier on
/// (gate_buf, up_buf) before a standalone `swiglu`. The gate and up matvecs read
/// the SAME read-only normed x and write DISJOINT buffers, so concurrent dispatch
/// is byte-identical to running them serially (each matvec's internal
/// accumulation order is unchanged; only the inter-matvec ordering relaxes). This
/// is the die-saturation lever (`metal_concurrent_proj_enabled`) extended to the
/// dense FFN — the biggest decode GPU pool. The fused 8row gate/up kernel cannot
/// be split across dies; this un-fuses it into two independent matvecs whose
/// threadgroups Metal can spread across both UltraFusion dies. HONEST: gate/up
/// runs at ~51 simdgroups/core (past the occupancy knee), so the die-spread may
/// be marginal/flat where the projection clusters (4-17 SG/core, die-starved)
/// won big — measured empirically. Diagnostic/perf lever; default OFF.
pub(crate) fn metal_concurrent_gateup_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let on = std::env::var("LUMEN_METAL_CONCURRENT_GATEUP")
        .map(|v| v == "1")
        .unwrap_or(false);
    CACHE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// When `LUMEN_METAL_CONCURRENT_GATEUP_256=1` (only consulted when
/// `metal_concurrent_gateup_enabled`), the concurrent gate/up matvecs use the
/// 256-thread `qmv_q4_0_8sg` kernel (8 simdgroups/TG, K split across lanes)
/// instead of the 64-thread `qmv_q4_0`. The 256-thread variant has higher
/// per-matvec occupancy but fewer threadgroups to spread across dies; the 64-thread
/// variant has 4x more threadgroups (better die-fill) but lower per-matvec
/// occupancy. Measured empirically. Default OFF (64-thread). Cached.
pub(crate) fn metal_concurrent_gateup_256_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let on = std::env::var("LUMEN_METAL_CONCURRENT_GATEUP_256")
        .map(|v| v == "1")
        .unwrap_or(false);
    CACHE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// When `LUMEN_METAL_GDN_STATE_H1=1` (default OFF), the Tier-0 GDN decode state
/// update dispatches the `gdn_state_output_l2_sg_h1` ("h1" = one global write)
/// kernel instead of the reference `gdn_state_output_l2_sg`. The h1 variant is
/// MATHEMATICALLY IDENTICAL — same L2-norm, same decay/retrieval/delta-update
/// arithmetic order, same simd_sum reductions, same output — but it keeps the
/// decayed state in registers and elides the reference kernel's redundant first
/// `h_state` write (the decayed-state store at the top of the recurrence is a
/// DEAD STORE: it is immediately overwritten by the updated-state store and is
/// never read back from device memory, since `retrieval` reads the decayed
/// values from registers). Removing it halves the per-token `h_state` WRITE
/// traffic on the 24 GDN layers (~50 MB/token of pure redundant device writes,
/// ~1% of the Q4 token stream) with a BYTE-IDENTICAL final state and output.
/// f32 state only (it reads/writes `device float* h_state`, so it is independent
/// of the reduced-precision state flags and safe with the f32 prefill path).
/// Cached after first read.
pub(crate) fn metal_gdn_state_h1_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let on = std::env::var("LUMEN_METAL_GDN_STATE_H1")
        .map(|v| v == "1")
        .unwrap_or(false);
    CACHE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// When `LUMEN_METAL_GDN_F16_STATE_DECODE=1` (default OFF), the Tier-0 GDN decode
/// recurrence (`gdn_state_output_l2_sg`) is replaced by the `gdn_state_output_l2_sg_f16`
/// variant, which stores the persistent recurrent `h_state` in F16 instead of F32.
/// h_state is `[n_v_heads * head_dim * head_dim]` (9B: 32*128*128 = 512Ki f32 = 2 MB)
/// PER GDN layer, READ and WRITTEN every token x 24 GDN layers -> ~96 MB/token of
/// device h_state traffic (~1.9% of the ~5.2 GB/token Q4 stream). Halving the persisted
/// state to F16 cuts that to ~48 MB/token. The recurrence math stays F32-in-registers
/// (load+upcast, compute, downcast-on-store), so the ONLY numerical change is the
/// rounding of the persisted state -> a near-tie, NOT byte-identical (acceptable if the
/// answer is preserved + self-deterministic, exactly like the other reduced-precision
/// decode levers).
///
/// CRUCIAL SAFETY: the prefill GDN kernels (`gdn_prefill_fused_v3*`) write `h_state` as
/// `device float*`, so the persistent F32 buffer (`s.gdn_h_states`) is LEFT F32-allocated
/// and prefill is UNCHANGED (no corruption). When this flag is ON, the FIRST decode touch
/// of each GDN layer converts that layer's F32 state into a SEPARATE half-size F16 buffer
/// (`s.gdn_h_states_f16[idx]`, allocated + filled lazily by `gdn_state_f32_to_f16` — a
/// distinct dst buffer, so NO in-place read/write aliasing) and the F16 recurrence
/// reads/writes the F16 buffer thereafter. The one-time per-layer convert (24 tiny copies)
/// amortizes over the whole decode. Default OFF (the byte-identical F32 path). Cached.
pub(crate) fn gdn_f16_state_decode_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let on = std::env::var("LUMEN_METAL_GDN_F16_STATE_DECODE")
        .map(|v| v == "1")
        .unwrap_or(false);
    CACHE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// When `LUMEN_METAL_GDN_F16_STATE_H1=1` (default OFF), the GDN decode recurrence uses
/// the F16 persistent h_state (exactly like `LUMEN_METAL_GDN_F16_STATE_DECODE` — same
/// lazy F32->F16 mirror, half the device state R+W) BUT dispatches the
/// `gdn_state_output_l2_sg_f16_h1` kernel, which additionally ELIDES the redundant
/// decayed-state write-back (a dead store overwritten by the phase-2 updated store;
/// retrieval reads the decayed values from registers). Net vs the F32 2-write reference:
/// ONE F16 read + ONE F16 write per state row = 4x less h_state device traffic
/// (2x F16 width x 2x dropping the dead write). This flag IMPLIES the F16 mirror
/// (so it engages even when `GDN_F16_STATE_DECODE` is not set) and, when ON,
/// takes precedence over the plain F16 kernel. Falls back to the f32 path when the
/// mirror is absent / the f16_h1 pipeline did not compile. Output is byte-identical to
/// the plain F16 kernel (only the dead store is removed) and a near-tie vs F32 (F16
/// state rounding — answer-preserving + self-deterministic, like the other reduced-
/// precision decode levers). Cached.
pub(crate) fn gdn_f16_state_h1_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let on = std::env::var("LUMEN_METAL_GDN_F16_STATE_H1")
        .map(|v| v == "1")
        .unwrap_or(false);
    CACHE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// When `LUMEN_METAL_GDN_F16_STATE_H1_V2=1` (default OFF), the Tier-0 fused GDN decode
/// recurrence dispatches the VI-AMORTIZED `gdn_state_output_l2_sg_f16_h1_v2` kernel:
/// each threadgroup handles TWO adjacent val_dim columns and computes the (vi-invariant)
/// Q/K L2-norm + Q/K device load ONCE, reusing it across both columns. This halves the
/// per-token redundant Q/K L2-norm ALU (the two simd_sum reductions + two sqrt + two
/// reciprocal-selects that the reference grid recomputes val_dim=128x per head) AND the
/// redundant Q/K device reads on the GDN recurrence critical dependency path that the
/// lean async pipeline cannot hide. Grid halves 4096 -> 2048 TGs (still richly occupied
/// on the 80-core M3 Ultra). Requires the F16 h_state mirror (implies the same lazy
/// F32->F16 conversion as GDN_F16_STATE_H1, with which it shares the half-size mirror)
/// and takes precedence over the single-vi f16_h1 / plain-f16 kernels when set. Falls
/// back to the f32 path when the mirror / v2 pipeline is absent. Output is BYTE-IDENTICAL
/// to the single-vi `gdn_state_output_l2_sg_f16_h1` kernel (the Q/K norm uses the same
/// 32-lane simd_sum reduction tree and each column runs the identical recurrence
/// arithmetic) and a near-tie vs F32 (F16 state rounding). Cached.
pub(crate) fn gdn_f16_state_h1_v2_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let on = std::env::var("LUMEN_METAL_GDN_F16_STATE_H1_V2")
        .map(|v| v == "1")
        .unwrap_or(false);
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

/// A/B EXPERIMENT (default OFF): when set, the GDN qkv (+ attn_gate) Q4_0 decode
/// matvec uses the llama.cpp lane->block mapping variant `qmv_q4_0_rmsnorm_llamacpp`
/// instead of the reference `qmv_q4_0_rmsnorm`. Pure dispatch swap on the SAME
/// repacked buffers (requires LUMEN_METAL_Q4_QMV_PROJ=1 to build/select them); only
/// the memory coalescing pattern differs. Isolates the matvec mapping variable.
pub(crate) fn q4_qmv_proj_lcmap_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let on = std::env::var("LUMEN_METAL_Q4_QMV_PROJ_LCMAP")
        .map(|v| v == "1")
        .unwrap_or(false);
    CACHE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// Precision of the persistent GDN recurrent h_state buffer.
///
/// The default `gdn_state_output_l2_sg` recurrence stores h_state in F32 (2
/// MB/layer, read+written every token×layer). The reduced-precision variants
/// store it in bfloat or half (1 MB/layer), halving the dominant decode state
/// traffic. The recurrence math stays F32-in-registers in all cases; only the
/// persisted state is rounded. Selected at allocation AND dispatch time so the
/// buffer byte-size and the kernel pointer type stay consistent.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GdnStatePrecision {
    F32,
    Bf16,
    F16,
}

/// When `LUMEN_METAL_GDN_BF16_STATE=1` -> Bf16; `LUMEN_METAL_GDN_F16_STATE=1`
/// -> F16; otherwise F32. BF16 takes precedence if both are set. Default F32
/// (the existing byte-identical path). Cached after first read.
pub(crate) fn gdn_state_precision() -> GdnStatePrecision {
    use std::sync::atomic::{AtomicU8, Ordering};
    // 0 = uninit, 1 = F32, 2 = Bf16, 3 = F16
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return match cur {
            2 => GdnStatePrecision::Bf16,
            3 => GdnStatePrecision::F16,
            _ => GdnStatePrecision::F32,
        };
    }
    let env_on = |name: &str| {
        std::env::var(name)
            .map(|s| !s.is_empty() && s != "0")
            .unwrap_or(false)
    };
    let p = if env_on("LUMEN_METAL_GDN_BF16_STATE") {
        GdnStatePrecision::Bf16
    } else if env_on("LUMEN_METAL_GDN_F16_STATE") {
        GdnStatePrecision::F16
    } else {
        GdnStatePrecision::F32
    };
    CACHE.store(
        match p {
            GdnStatePrecision::Bf16 => 2,
            GdnStatePrecision::F16 => 3,
            GdnStatePrecision::F32 => 1,
        },
        Ordering::Relaxed,
    );
    p
}

/// When `LUMEN_METAL_MOE_DOWN_SGROW=1`, the mixed q8/q4 MoE
/// down+accum dispatch uses the one-simdgroup-per-row redesigned kernel
/// (`moe_batched_down_accum_shared_q8_0_se_q4_0_v2`). Default OFF → the
/// original kernel runs (byte-identical path). Cached after first read.
pub(crate) fn moe_down_sgrow_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_MOE_DOWN_SGROW")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
}

/// When `LUMEN_METAL_MOE_GATEUP_SGROW=1`, the routed
/// gate+up+swiglu dispatch uses the one-simdgroup-per-row redesigned kernel
/// (`moe_batched_gate_up_swiglu_q8_0_v2`). Default OFF → original kernel
/// (byte-identical path). Cached after first read.
pub(crate) fn moe_gateup_sgrow_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_MOE_GATEUP_SGROW")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
    CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
    v
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

/// Fused single-dispatch MoE router (logits + top-k in ONE
/// grid=num_experts dispatch via last-threadgroup reduction). Eliminates the
/// separate grid=1 top-k dispatch whose drain bubble was measured at ~6 ms/token
/// (39% of decode) on the serial GDN decode encoder. Default OFF until validated;
/// set `LUMEN_METAL_MOE_ROUTER_FUSED=1` to enable. Falls back to the two-kernel
/// parallel router when off or when the pipeline/counter is unavailable.
/// Cached after first read.
pub(crate) fn moe_router_fused_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 0 {
        return cur == 2;
    }
    let v = std::env::var("LUMEN_METAL_MOE_ROUTER_FUSED")
        .map(|s| !s.is_empty() && s != "0")
        .unwrap_or(false);
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

/// [diag] Returns which MoE sub-dispatch to SKIP for GPU-time attribution
/// (LUMEN_METAL_MOE_DIAG_SKIP=down→1, =gateup→2, else 0). Produces garbage
/// output when set; use only with the decode profiler to read GPU timings.
pub(crate) fn moe_diag_skip() -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(255);
    let cur = CACHE.load(Ordering::Relaxed);
    if cur != 255 {
        return cur;
    }
    let v = match std::env::var("LUMEN_METAL_MOE_DIAG_SKIP").ok().as_deref() {
        Some("down") => 1u8,
        Some("gateup") => 2u8,
        Some("shared") => 3u8,
        Some("gating") => 4u8,
        Some("router") => 5u8,
        Some("rtopk") => 6u8,   // skip only the top-k softmax kernel
        Some("rlogits") => 7u8, // skip only the per-expert logits kernel
        _ => 0u8,
    };
    CACHE.store(v, Ordering::Relaxed);
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
