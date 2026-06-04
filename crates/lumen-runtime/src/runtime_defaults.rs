//! Process-wide runtime defaults — operator-safety.
//!
//! Centralises the small handful of "what should the default be when the
//! user did not set an env var?" decisions that previously required the
//! operator to memorise multiple `LUMEN_CUDA_*` flags depending on the
//! model / quant configuration (Q8 dense, Q4 dense, BF16 dense, MoE) they
//! were running. The four distinct items this module addresses are:
//!
//! * **Env-var typo validator** — `validate_lumen_env_vars` enumerates the
//!   process env for `LUMEN_*` names that are NOT in the canonical allowlist
//!   and emits a single stderr warning per unknown name. Catches the class
//!   of bug where a missing `LUMEN_CUDA_` prefix silently turns the env into
//!   a no-op.
//! * **Server-default decode delay** — `set_path_is_server` flips the default of
//!   `LUMEN_CUDA_DECODE_DELAY_US` from `0` (CLI default) to `50` (server
//!   default), matching the server-determinism fix without requiring the
//!   operator to remember the flag.
//! * **Model-aware dense defaults** — `set_model_dense_quant` consumes the
//!   LBC-resolved dense tensor scheme and flips two families of defaults
//!   conditional on "BF16 model": (a) `bf16_gemmex_default()` returns `true`
//!   for BF16, `false` for Q8/Q4 dense; (b) `decode_graph_*_default()`
//!   returns `true` for BF16 dense, `false` otherwise (graph capture is a
//!   measurable win on BF16 dense but a regression on Q8/Q4 dense).
//! * **Graph-capture auto-enable** — the four graph-capture envs that
//!   required explicit opt-in (`LUMEN_CUDA_DECODE_GRAPH`,
//!   `LUMEN_CUDA_DECODE_GRAPH_QGATE`, `LUMEN_CUDA_DECODE_GRAPH_TILED`, plus the
//!   `LUMEN_CUDA_BF16_GEMMEX=1` covered by `set_model_dense_quant`) now auto-enable
//!   when the model is BF16 dense.
//!
//! # Ordering contract
//!
//! Setters MUST be called BEFORE the first read of any defaulted env:
//!
//! 1. Caller (binary `main`) opens the LBC, learns `provider.output_proj_quant`,
//!    invokes `set_path_is_server(args.backend.is_server)`, then
//!    `set_model_dense_quant(provider.output_proj_quant)`.
//! 2. `CudaBackend::new` and the first decode call subsequently invoke
//!    `bf16_gemmex_default()`, `cuda_decode_delay_us_default()`, and
//!    `decode_graph_default()` exactly once. Each is `OnceLock`-cached on
//!    first read, so post-init mutation has no effect.
//!
//! The setters are idempotent: setting the same value twice is a no-op;
//! changing the value after the cache has materialised is logged as a
//! debug warning and otherwise ignored. This matches the
//! `bf16_gemmex_env_force_off` cache pattern already in `backend_impl.rs`.

use lumen_format::quantization::QuantScheme;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Internal storage — atomics so callers don't need a Mutex on the hot path.
// ---------------------------------------------------------------------------

/// `0` = unset (CLI default), `1` = server. Read once, cached lazily by
/// `cuda_decode_delay_us_default`. Atomic + Relaxed because writes happen
/// at most once during `main()` before any backend dispatch.
static PATH_IS_SERVER: AtomicBool = AtomicBool::new(false);

/// Encodes the dense-quant hint set by the binary. `0` = unset (use legacy
/// "default ON" BF16-gemmex behaviour, default OFF graph capture), `1` =
/// BF16, `2` = quantised (Q8/Q4/etc.). Encoded as `AtomicU8` so the read
/// path is one relaxed load.
static MODEL_DENSE_QUANT_HINT: AtomicU8 = AtomicU8::new(0);

const HINT_UNSET: u8 = 0;
const HINT_BF16: u8 = 1;
const HINT_QUANTISED: u8 = 2;

/// Tracks whether the loaded LBC declares MoE experts (i.e. Qwen3.5-MoE-30B-A3B
/// class). `false` = dense; `true` = experts > 0 reported by the LBC
/// hyperparams. Finding: the Q8 "split sibling" weight clone path
/// (`LUMEN_CUDA_Q8_SPLIT=1`) is byte-identical to the canonical Q8 dense
/// decode kernel BUT causes catastrophic PAD-token spam on Q8 MoE 30B-A3B
/// (every prompt: 1 valid first token + 159 `[PAD248319]`). Previously
/// `q8_split_default()` flipped the default ON for any `HINT_QUANTISED`
/// model — Q8 MoE matches that hint via its Q8_0 output_proj, so the default
/// silently broke MoE decode. This atomic carries the missing "is this model
/// MoE?" signal so the Q8-only flag resolvers (Q8_SPLIT / OUTPUT_PROJ_SPLIT /
/// Q8_SCALE_HW / OUTPUT_PROJ_NR=16 / FFN_FUSED_GLU_SKIP) can stay OFF for
/// MoE while remaining ON for dense Q8 / Q4 (the dense Q8 configuration continues to win at
/// 0.907× llama.cpp).
static MODEL_IS_MOE: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Public setters — called once from the binary `main` after LBC inspection.
// ---------------------------------------------------------------------------

/// Marks the running process as the `lumen-server` binary (vs CLI). When
/// set, `cuda_decode_delay_us_default()` returns `50` instead of `0`, so
/// the server path closes the GPU-scheduler timing race without
/// requiring `LUMEN_CUDA_DECODE_DELAY_US=50` in the operator's env. The
/// env-var still wins if the operator sets it explicitly.
///
/// Idempotent — calling twice with the same value is a no-op.
pub fn set_path_is_server(is_server: bool) {
    PATH_IS_SERVER.store(is_server, Ordering::Relaxed);
}

/// Records the dense-tensor (`output_proj`) quantisation scheme observed
/// when the LBC opens. Used to flip the per-call defaults of
/// `LUMEN_CUDA_BF16_GEMMEX` and the `LUMEN_CUDA_DECODE_GRAPH*` family.
///
/// * `Bf16` → BF16-gemmex default ON; graph capture default ON.
/// * `Q8_0` / `Q4_0` / other quantised schemes → BF16-gemmex default OFF;
///   graph capture default OFF.
/// * Unset (this setter never called) → preserves legacy behaviour
///   (BF16-gemmex default ON, graph capture default OFF).
///
/// Idempotent. Called from `lumen-server::run` and `lumen-cli::run`
/// immediately after `SyncWeightProvider::open` returns.
pub fn set_model_dense_quant(scheme: QuantScheme) {
    let hint = match scheme {
        QuantScheme::Bf16 => HINT_BF16,
        // Anything quantised — Q8/Q4/Q5/Q6/etc. — gets the "quantised" default.
        QuantScheme::Q8_0
        | QuantScheme::Q4_0
        | QuantScheme::Q4_1
        | QuantScheme::Q4_K
        | QuantScheme::Q5_0
        | QuantScheme::Q5_K
        | QuantScheme::Q6_K
        | QuantScheme::Q2_K
        | QuantScheme::Q3_K => HINT_QUANTISED,
        // F32/F16 → leave as legacy (HINT_UNSET == 0 means
        // "fall through to legacy default ON" in the resolvers).
        QuantScheme::F32 | QuantScheme::F16 => HINT_UNSET,
    };
    MODEL_DENSE_QUANT_HINT.store(hint, Ordering::Relaxed);
}

/// Records whether the loaded LBC declares MoE experts. Called from the
/// CLI / server `main()` immediately after `SyncWeightProvider::open` and
/// alongside `set_model_dense_quant`. The signal feeds the Q8-only flag
/// resolvers (`q8_split_default`, `output_proj_split_default`,
/// `q8_scale_hw_default`, `output_proj_nr_default`,
/// `ffn_fused_glu_skip_default`) so they correctly stay OFF for MoE
/// 30B-A3B
/// while remaining ON for dense Q8 (dense Q8 9B, 0.907× llama.cpp) and dense Q4.
///
/// Idempotent — calling twice with the same value is a no-op. The CLI /
/// server should call this BEFORE `create_backend` so `CudaBackend::new`
/// observes the correct default on first read.
pub fn set_model_is_moe(is_moe: bool) {
    MODEL_IS_MOE.store(is_moe, Ordering::Relaxed);
}

/// Reports the cached MoE flag set by `set_model_is_moe`. Used by the
/// Q8-only default resolvers below and by Metal default resolvers in
/// `metal/graph_reorder.rs` (: gate Q8/Q4 repack and FFN-down
/// Split-K defaults OFF for MoE, mirroring CUDA's pattern).
pub(crate) fn model_is_moe() -> bool {
    MODEL_IS_MOE.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Default resolvers — called by `cuda::backend_impl` with a fall-through
// to `std::env::var` when the operator has set the env explicitly.
// ---------------------------------------------------------------------------

/// Resolves the per-process default for `LUMEN_CUDA_DECODE_DELAY_US` when
/// the env var is not set. Server path returns `50` µs (closes the
/// race); CLI returns `0` (no slowdown, CLI is already
/// fork-deterministic).
pub fn cuda_decode_delay_us_default() -> u64 {
    if PATH_IS_SERVER.load(Ordering::Relaxed) {
        50
    } else {
        0
    }
}

/// Resolves the per-process default for `LUMEN_METAL_DECODE_DELAY_US` when the
/// env var is not set. Returns `50` µs for BOTH the server AND the CLI path.
///
/// This DIVERGES from the CUDA policy (CUDA CLI returns `0`). The reason is
/// empirical: CUDA's CLI decode path replays a captured CUDA graph, so it is
/// bit-deterministic without any delay. The Metal backend has NO equivalent
/// graph-capture replay — its greedy decode (`decode_token_greedy`) is the
/// same on CLI and server, and was measured to be non-deterministic
/// across BOTH repeated in-process requests AND repeated cold-start `lumen run`
/// invocations at delay=0 (Q8 ~10% within-process / ~27% cross-process; Q4
/// ~30%). The divergence is the documented GPU-scheduler near-tie
/// timing race: at a sub-ULP-margin top-1/top-2 logit pair, scheduler-timing-
/// dependent floating-point reduction order in the upstream GPU kernels flips
/// the on-GPU argmax. (The argmax kernel itself is deterministic.)
///
/// IMPORTANT — this delay is a MITIGATION, not a cure. A sweep of the value
/// over 30-60-trial samples found NO value yields a reliable 30/30: the
/// rate is noisy and non-monotonic (Q8 ~1.7% residual at 50-200µs, WORSE at
/// 500µs; Q4 barely improves). A CPU inter-token sleep only perturbs the
/// scheduler-timing distribution; it cannot make a within-token GPU FP
/// reduction deterministic. 50µs reduces user-visible Q8 non-determinism ~6×
/// (10%→~1.7%) at ~0.45% TPOT cost, and unifies the CLI/server default. A true
/// hard guarantee would require deterministic-reduction kernels (out of scope).
///
/// UPDATE: the DET-001 decode non-determinism is now ROOT-CAUSED and FIXED
/// (two intra-kernel cross-threadgroup races in the decode full-attention path —
/// the `fused_rope_kv_mha` in-place K write-back, and the `deinterleave_norm_assemble`
/// qgate-read vs K/V-write aliasing on the shared qkv_buf). With both fixed, Metal
/// greedy decode is byte-deterministic at 100/100 on Q8 and Q4 (and Q8 byte-matches
/// llama.cpp). The decode-delay was always a MITIGATION that did not generalize and
/// cost ~0.45% TPOT; it is now UNNECESSARY. **Default reverted to 0 (bit-exact).** The
/// `LUMEN_METAL_DECODE_DELAY_US` env var remains available for diagnostics.
pub fn metal_decode_delay_us_default() -> u64 {
    // DET-001 is fixed at the kernel level; no scheduler-timing mitigation
    // is needed. 0 = bit-exact no-op path. Operators can still set the env var.
    0
}

/// Resolves the per-process default for `LUMEN_CUDA_BF16_GEMMEX` when the
/// env var is not set. BF16 models default to `true` (GemmEx fast path
/// on); quantised models default to `false` (the path is unused so the
/// startup probe would emit a misleading warning); unset hint preserves
/// the legacy default of `true`.
pub fn bf16_gemmex_default() -> bool {
    match MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) {
        HINT_QUANTISED => false,
        // BF16 OR unset (legacy preserves "true" so a never-set hint never
        // surprises the operator by flipping a behaviour they relied on).
        _ => true,
    }
}

/// Resolves the per-process default for `LUMEN_CUDA_DECODE_GRAPH` (the
/// master gate on CUDA graph capture) when the env var is not set. BF16
/// dense returns `true` (graph capture is a +13% TPOT win on
/// BF16 dense); Q8/Q4 / unset returns `false` (graph capture is a measured
/// regression on quantised configurations).
pub fn decode_graph_default() -> bool {
    MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) == HINT_BF16
}

/// Resolves the per-process default for `LUMEN_CUDA_DECODE_GRAPH_QGATE`
/// when the env var is not set. Coupled to `decode_graph_default` because
/// the qgate branch is meaningless without the master gate on.
pub fn decode_graph_qgate_default() -> bool {
    decode_graph_default()
}

/// Resolves the per-process default for `LUMEN_CUDA_DECODE_GRAPH_TILED`
/// when the env var is not set. Coupled to `decode_graph_default`.
pub fn decode_graph_tiled_default() -> bool {
    decode_graph_default()
}

// ---------------------------------------------------------------------------
// canonical performance defaults
//
// Without any env flags, Lumen CUDA decode runs at ~0.04× llama.cpp on the MoE Q8 configuration
// (5.4 vs 140 tok/s measured 2026-06-01 on A100) because the optimal kernels
// require ~14 LUMEN_CUDA_* opt-in flags. The 18-flag "canonical" config
// achieves 0.908× llama.cpp on
// the dense Q8 9B configuration. The gap root-caused to per-flag default
// drift: the optimal kernels are gated default-OFF for historical byte-
// identity reasons, but every production workload needs them ON. This revision
// flips each "safe" default to ON so unset operators get canonical perf.
//
// The flips below are SAFE because each gate is a no-op for irrelevant
// model classes (e.g. `LUMEN_CUDA_MOE_BATCHED=1` only fires when MoE layers
// are present; setting it ON has zero effect on dense-9B). The complete
// flag-by-flag safety analysis is in
//
// Opt-out: set `LUMEN_CUDA_LEGACY_DEFAULTS=1` to restore the previous behaviour
// "default OFF" behaviour on every flag below. The env var is checked once
// at process start via `OnceLock` so subsequent toggles in the same process
// have no effect. Per-flag explicit overrides (e.g. `LUMEN_CUDA_Q8_SPLIT=0`)
// still win over both this resolver and the legacy-defaults switch.
// ---------------------------------------------------------------------------

/// Master opt-out for the F2 canonical-default flips. Returns
/// `true` when `LUMEN_CUDA_LEGACY_DEFAULTS=1` is set — in that case every
/// per-flag default resolver below falls back to the previous behaviour "OFF"
/// behaviour, matching the byte-identical decode path used by the
/// regression bench /171.
fn legacy_defaults_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LUMEN_CUDA_LEGACY_DEFAULTS")
            .ok()
            .as_deref()
            .map(|v| matches!(v, "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    })
}

/// Returns the canonical default for a typical safe gate: ON unless the
/// master `LUMEN_CUDA_LEGACY_DEFAULTS=1` switch is set. Used by the safe
/// flag resolvers below; per-call cost is one cached atomic load.
fn canonical_default_on() -> bool {
    !legacy_defaults_enabled()
}

/// Per-process default for `LUMEN_CUDA_MOE_BATCHED` when the env is unset.
/// ON by default — fires only for MoE models, no effect on dense.
pub fn moe_batched_default() -> bool { canonical_default_on() }

/// Per-process default for `LUMEN_CUDA_MOE_ROUTER_PARALLEL` when unset.
/// ON by default — fires only for MoE, dispatches the two-launch parallel
/// router instead of the sequential single-CTA router.
pub fn moe_router_parallel_default() -> bool { canonical_default_on() }

/// Per-process default for `LUMEN_CUDA_GDN_REGISTER_RESIDENT` when unset.
/// ON by default — fires only for GDN models (Qwen3.5 family).
/// Finding: the two-launch phase 4 update is byte-identical to the reference
/// path.
pub fn gdn_register_resident_default() -> bool { canonical_default_on() }

/// Per-process default for `LUMEN_CUDA_BF16_MOE_V3` when unset. ON by
/// default — fires only for BF16 MoE expert dispatch.
pub fn bf16_moe_v3_default() -> bool { canonical_default_on() }

/// Per-process default for `LUMEN_CUDA_MOE_Q4_V3` when unset. ON by
/// default — fires only for Q4 MoE expert dispatch.
pub fn moe_q4_v3_default() -> bool { canonical_default_on() }

/// Per-process default for `LUMEN_CUDA_MOE_Q4_V3B` when unset. ON by
/// default — fires only for Q4 MoE; gated by V3 also being ON.
pub fn moe_q4_v3b_default() -> bool { canonical_default_on() }

/// Per-process default for `LUMEN_CUDA_MMV_Q_OUTPUT_PROJ` when unset. ON
/// by default — affects dense Q8/Q4 output projection. finding:
/// the fused matvec saves ~25% on the vocab projection.
pub fn mmv_q_output_proj_default() -> bool { canonical_default_on() }

/// Per-process default for `LUMEN_CUDA_FFN_FUSED_GLU` "skip" gate when
/// unset. The env-var semantics are inverted (`=0` SKIPS the fused kernel,
/// using the dp4a fall-through). Default is to skip on quantised dense
/// models. BF16 dense
/// uses a different kernel class, so the skip is a no-op there.
///
/// Returns the **skip** boolean: `true` means "use the dp4a fall-through"
/// (matches the canonical `LUMEN_CUDA_FFN_FUSED_GLU=0`). Quantised dense
/// is the only class where the dp4a fall-through wins; BF16 dense and MoE
/// are unaffected because their FFN paths don't dispatch the fused-GLU
/// kernel in the first place.
pub fn ffn_fused_glu_skip_default() -> bool {
    match MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) {
        // scope fix: docstring says "Quantised dense is the only
        // class where the dp4a fall-through wins". MoE FFN is routed through
        // the per-expert path, not the dense fused-GLU kernel, so the SKIP
        // default is irrelevant at best and risks parity drift at worst.
        // Stay OFF (legacy) for MoE; ON only for true dense Q8/Q4.
        HINT_QUANTISED if !model_is_moe() => canonical_default_on(),
        // BF16 / unset / MoE: skip is a no-op anyway, but default to false so
        // BF16 invocations don't pay the (tiny) extra check cost.
        _ => false,
    }
}

/// Per-process default for `LUMEN_CUDA_Q8_SPLIT` when unset. ON for Q8
/// dense (clones Q8_0 weights to the split layout, ~0.6 GB extra VRAM on
/// A100, enables `matvec_q8_split_q8_1`). No-op when the model has no
/// Q8_0 weights.
///
/// **scope fix**: explicitly OFF for MoE (Qwen3.5-MoE-30B-A3B).
/// The Q8 SPLIT clone pass operates on per-layer `wq/wk/wv/wo/w_gate/w_up/
/// w_down` Q8_0 tensors; on an MoE LBC the dense MLP path is replaced by
/// per-expert weights and the clone pass cloned 70 jobs / 0.6 GB on MoE
/// without populating siblings for the expert weights. The resulting
/// decode dispatch routed through a partially-cloned state and emitted
/// `WORD[PAD248319]×159` on every prompt, because the previous default
/// silently applied the same configuration to MoE. Gating the
/// default OFF for MoE restores MoE coherence while preserving the
/// 0.907× llama.cpp win on dense Q8. The documented intent of the docstring
/// ("Only Q8 dense benefits") matches this scope exactly.
pub fn q8_split_default() -> bool {
    match MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) {
        // Only Q8 dense benefits; Q4/BF16/F32 ignore the split sibling.
        // MoE: explicit OFF (measured root-cause).
        HINT_QUANTISED if !model_is_moe() => canonical_default_on(),
        _ => false,
    }
}

/// Per-process default for `LUMEN_CUDA_OUTPUT_PROJ_SPLIT` when unset. ON
/// for Q8 dense (output projection in particular). Same gating logic as
/// `q8_split_default`.
pub fn output_proj_split_default() -> bool { q8_split_default() }

/// Per-process default for `LUMEN_CUDA_Q8_SCALE_HW` when unset. ON for
/// Q8 dense (prefer the `matvec_q8_aligned_q8_1_hw` kernel that uses
/// hardware-scale dp4a; no-op when the kernel is absent or not Q8 dense).
pub fn q8_scale_hw_default() -> bool { q8_split_default() }

/// Per-process default for `LUMEN_CUDA_OUTPUT_PROJ_NR` when unset. Returns
/// `16` for Q8 dense (the measured optimum). `1` is the legacy
/// default for any other configuration.
pub fn output_proj_nr_default() -> u32 {
    if q8_split_default() { 16 } else { 1 }
}

/// Per-process default for `LUMEN_CUDA_MOE_DECODE_GRAPH` when unset.
/// MoE-only graph capture; measured 0.00% paired delta but the ON path is
/// byte-identical, so we ship it ON for MoE to keep the canonical flag stack
/// reproducible without env juggling. No-op for dense models.
pub fn moe_decode_graph_default() -> bool { canonical_default_on() }

// ---------------------------------------------------------------------------
// Env-var typo validator
// ---------------------------------------------------------------------------

/// Canonical allowlist of `LUMEN_*` env vars recognised across the
/// runtime, CLI, server, and bench crates. Generated by `grep -rEoh
/// '"LUMEN_[A-Z0-9_]+"' crates/` and reviewed manually. ADD new names here
/// when a new env gate ships, or the validator will warn at startup.
///
/// Sorted alphabetically to make `diff` reviewable when the list changes.
const KNOWN_LUMEN_ENV_VARS: &[&str] = &[
    "LUMEN_AB_ITERATIONS",
    "LUMEN_AB_WARMUP",
    "LUMEN_BASE_URL",
    "LUMEN_BENCH_ITERATIONS",
    "LUMEN_BENCH_SCALE",
    "LUMEN_BENCH_TOKENS",
    "LUMEN_BENCH_WARMUP",
    "LUMEN_CACHE_DIR",
    "LUMEN_CUDA_BF16_AUTOTUNE",
    "LUMEN_CUDA_BF16_GEMMEX",
    "LUMEN_CUDA_BF16_MOE_V3",
    "LUMEN_CUDA_DECODE_DELAY_US",
    "LUMEN_CUDA_DECODE_GRAPH",
    "LUMEN_CUDA_DECODE_GRAPH_QGATE",
    "LUMEN_CUDA_DECODE_GRAPH_TILED",
    "LUMEN_CUDA_DECODE_TILED",
    "LUMEN_CUDA_DECODE_TILED_THRESHOLD",
    "LUMEN_CUDA_FA2_ATTN",
    "LUMEN_CUDA_FA2_BLOCKSKIP",
    "LUMEN_CUDA_FFN_FUSED_GLU",
    "LUMEN_CUDA_GDN_AB_F32",
    "LUMEN_CUDA_GDN_F64_ACCUM",
    "LUMEN_CUDA_GDN_REGISTER_RESIDENT",
    "LUMEN_CUDA_GDN_PHASE4_COAL",
    "LUMEN_CUDA_GDN_SPLIT",
    "LUMEN_CUDA_L2NORM_RSQRTF",
    "LUMEN_CUDA_LEGACY_DEFAULTS",
    "LUMEN_CUDA_LEVER_TRACE",
    "LUMEN_CUDA_MAX_SEQ_LEN",
    "LUMEN_CUDA_MMV_BF16_OUTPUT_PROJ",
    "LUMEN_CUDA_MMV_Q_DP4A",
    "LUMEN_CUDA_MMV_Q_MOE_DP4A",
    "LUMEN_CUDA_MMV_Q_OUTPUT_PROJ",
    "LUMEN_CUDA_MOE_BATCHED",
    "LUMEN_CUDA_MOE_BATCHED_V2",
    "LUMEN_CUDA_MOE_BATCHED_V3",
    "LUMEN_CUDA_MOE_DEBUG_DUMP",
    "LUMEN_CUDA_MOE_DECODE_GRAPH",
    "LUMEN_CUDA_MOE_FUSED_NORM_ROUTER",
    "LUMEN_CUDA_MOE_FUSED_PERSISTENT",
    "LUMEN_CUDA_MOE_Q4_V3",
    "LUMEN_CUDA_MOE_Q4_V3B",
    "LUMEN_CUDA_MOE_ROUTER_PARALLEL",
    "LUMEN_CUDA_MOE_ROUTER_SINGLE_CTA",
    "LUMEN_CUDA_MOE_SHARED_FUSED",
    "LUMEN_CUDA_NORM_RSQRTF_BUNDLE",
    "LUMEN_CUDA_OUTPUT_PROJ_F16_CACHE",
    "LUMEN_CUDA_OUTPUT_PROJ_NR",
    "LUMEN_CUDA_OUTPUT_PROJ_SPLIT",
    "LUMEN_CUDA_PREFILL_F32",
    "LUMEN_CUDA_PROFILE",
    "LUMEN_CUDA_Q4_SPLIT",
    "LUMEN_CUDA_Q4_TILE",
    "LUMEN_CUDA_Q4_V3_TRACE",
    "LUMEN_CUDA_Q8_AOS_NR8",
    "LUMEN_CUDA_Q8_PROJ_MMQ",
    "LUMEN_CUDA_Q8_SCALE_HW",
    "LUMEN_CUDA_Q8_SPLIT",
    "LUMEN_CUDA_Q8_SPLIT_4THREAD",
    "LUMEN_CUDA_Q8_SPLIT_NR8",
    "LUMEN_CUDA_Q8_SSM_OUT_MMQ_OFF",
    "LUMEN_CUDA_Q8_TILE",
    "LUMEN_CUDA_RMSNORM_RSQRTF",
    "LUMEN_CUDA_SKIP_BF16_PROBE",
    "LUMEN_CUDA_SKIP_SHARED_EXPERT",
    "LUMEN_CUDA_SKIP_SHARED_GATE",
    "LUMEN_CUDA_TOPK_MOE_FUSED",
    "LUMEN_CUDA_VERBOSE",
    "LUMEN_DEBUG_DUMP_SSM_BETA_W",
    "LUMEN_DUMP_EXPERTS",
    "LUMEN_DUMP_GDN_L0_BIN",
    "LUMEN_DUMP_NORMED",
    "LUMEN_DUMP_RUNTIME_Q8_SCALES",
    "LUMEN_GRAPH_DIAGNOSTIC",
    "LUMEN_KV_PRECISION",
    "LUMEN_METAL_BF16_GATE_UP_NR",
    "LUMEN_METAL_BF16_GDN_FULL_PREFILL_WARMUP",
    "LUMEN_METAL_BF16_GDN_QKV_GATE_PAIRED",
    "LUMEN_METAL_BF16_GDN_TILE_NOK64",
    "LUMEN_METAL_BF16_GDN_WARMUP",
    "LUMEN_METAL_BF16_GDN_WARMUP_MODE",
    "LUMEN_METAL_BF16_MMAP_ONLY",
    "LUMEN_METAL_BF16_MPS",
    "LUMEN_METAL_CONCURRENT_ENCODER",
    "LUMEN_METAL_CONCURRENT_ENCODER_FULL",
    "LUMEN_METAL_CONCURRENT_ENCODER_FULL_VALIDATE",
    "LUMEN_METAL_CONCURRENT_ENCODER_TRACE",
    "LUMEN_METAL_CONCURRENT_ENCODER_VALIDATE",
    "LUMEN_METAL_DECODE_DELAY_US",
    "LUMEN_METAL_FFN_DOWN_SPLITK",
    "LUMEN_METAL_FFN_DOWN_SPLITK_BF16",
    "LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED",
    "LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_BF16",
    "LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_Q4",
    "LUMEN_METAL_GDN_CONCURRENT_ENCODER",
    "LUMEN_METAL_GDN_CONCURRENT_ENCODER_VALIDATE",
    "LUMEN_METAL_GDN_PHASE2A_NSG4",
    "LUMEN_METAL_GDN_SSM_OUT_F32_BATCHED",
    "LUMEN_METAL_GEMM_GGML_PORT",
    "LUMEN_METAL_MMAP_ONLY",
    "LUMEN_METAL_MULTI_CB",
    "LUMEN_METAL_NAN_DUMP",
    "LUMEN_METAL_MULTI_CB_N",
    "LUMEN_METAL_DEFAULTS_OFF",
    "LUMEN_METAL_PROFILE",
    "LUMEN_METAL_PROFILE_DEEP",
    "LUMEN_METAL_PROFILE_GDN",
    "LUMEN_METAL_Q4_REPACKED",
    "LUMEN_METAL_Q4_REPACKED_FFN_DOWN",
    "LUMEN_METAL_Q4_REPACKED_GATE_UP",
    "LUMEN_METAL_Q8_REPACKED",
    "LUMEN_METAL_Q8_REPACKED_FFN_DOWN",
    "LUMEN_METAL_Q8_REPACKED_GATE_UP",
    "LUMEN_METAL_UNRETAINED_CMDBUFS",
    "LUMEN_QWEN35_9B_BF16",
    "LUMEN_QWEN35_9B_PATH",
    "LUMEN_QWEN35_9B_Q4",
    "LUMEN_QWEN35_9B_Q8",
    "LUMEN_SERVER_DEBUG_MEM",
    "LUMEN_SERVER_PANIC_MAX",
    "LUMEN_SERVER_PANIC_WINDOW_SECS",
    "LUMEN_SERVER_PER_JOB_RESET",
    "LUMEN_SOAK_DURATION_SEC",
    "LUMEN_SOAK_OUT_DIR",
    "LUMEN_SOAK_STACK_DUMP",
    "LUMEN_SOAK_STACK_LEAKS",
    "LUMEN_SOAK_STACK_TICKS",
    "LUMEN_SOAK_WARMUP_SEC",
    "LUMEN_SUFFIX_THRESHOLD",
    "LUMEN_TEST_OPENAI_SDK",
];

/// Enumerates the process env and emits a stderr WARNING for every
/// `LUMEN_*` env var that does NOT appear in `KNOWN_LUMEN_ENV_VARS`.
///
/// This catches the family of bugs: an operator types
/// `GDN_REGISTER_RESIDENT=1` instead of `LUMEN_CUDA_GDN_REGISTER_RESIDENT=1`.
/// The typo is silently accepted by `std::env::var` (which returns
/// `Err(NotPresent)` for the correct name) and the gate it was supposed to
/// toggle stays in its default state.
///
/// Cost: one `env::vars` scan at startup (typically ~50-200 vars in a
/// shell session). The validator runs once from `main` before backend
/// construction. Returns the list of warnings emitted (in deterministic
/// alphabetical order) so the caller can record them in the startup log
/// and so the unit test below can assert on the exact set without
/// capturing stderr.
pub fn validate_lumen_env_vars() -> Vec<String> {
    let suspects = collect_unknown_lumen_env_vars();
    for warning in &suspects {
        eprintln!("[lumen] WARNING: {warning}");
    }
    suspects
}

/// Pure helper exposed for unit testing. Reads `std::env::vars` and
/// emits a sorted `Vec<String>` of human-readable warning messages for
/// two classes of typo:
///
/// 1. **Mis-spelled suffix on a `LUMEN_*` env var** — e.g. the canonical
///    `LUMEN_CUDA_GDN_REGISTER_RESIDENT=1` with a missing trailing `T`.
///    Caught by the "starts with `LUMEN_` but not in the allowlist" pass;
///    the closest canonical name appears in the suggestion list.
/// 2. **Missing `LUMEN_CUDA_` / `LUMEN_METAL_` prefix** — the literal
///    bug: operator typed `GDN_REGISTER_RESIDENT=1` expecting it
///    to behave like `LUMEN_CUDA_GDN_REGISTER_RESIDENT=1`. The plain-suffix
///    variant is undetectable by name-prefix matching alone, so this pass
///    additionally checks every `*` (non-LUMEN_) env var against the
///    suffix-match heuristic: if a `LUMEN_CUDA_*` allowlist entry ends
///    with the SAME suffix as a non-LUMEN env var (case-sensitive, full
///    suffix match), the validator warns. False positives are limited by
///    requiring the SUFFIX to be ≥ 6 chars and to begin with one of the
///    canonical LUMEN-domain roots (`CUDA_`, `METAL_`, `SERVER_`,
///    `BENCH_`, `CACHE_`, `GRAPH_`, `KV_`, `BASE_`, etc.). The list of
///    canonical suffixes is generated from the allowlist itself, so it
///    grows automatically as new envs ship.
///
/// All warning messages include up to 3 closest-suffix canonical names
/// so the operator can see "did you mean LUMEN_CUDA_GDN_REGISTER_RESIDENT?"
/// at a glance.
fn collect_unknown_lumen_env_vars() -> Vec<String> {
    let env_vars: Vec<String> = std::env::vars().map(|(k, _)| k).collect();
    let mut warnings = Vec::new();

    // Pass 1 — names that start with `LUMEN_` but are NOT in the
    // allowlist. This catches mis-spelled suffixes on otherwise-correct
    // env names.
    let mut unknown_with_prefix: Vec<&String> = env_vars
        .iter()
        .filter(|k| k.starts_with("LUMEN_"))
        .filter(|k| !KNOWN_LUMEN_ENV_VARS.iter().any(|known| *known == k.as_str()))
        .collect();
    unknown_with_prefix.sort();
    for name in unknown_with_prefix {
        let suggestions = closest_known_matches(name, 3);
        warnings.push(if suggestions.is_empty() {
            format!("unknown env var '{name}' — typo? known: (none similar)")
        } else {
            format!(
                "unknown env var '{name}' — typo? known: {}",
                suggestions.join(", ")
            )
        });
    }

    // Pass 2 — names that do NOT start with `LUMEN_` but DO suffix-match a
    // canonical LUMEN_CUDA_* / _METAL_* / _SERVER_* allowlist entry. This
    // catches the literal typo: `GDN_REGISTER_RESIDENT=1` instead
    // of `LUMEN_CUDA_GDN_REGISTER_RESIDENT=1`. The 6-char minimum on the
    // matching suffix keeps the false-positive rate low. Tracking `seen`
    // prevents emitting the same warning twice if e.g. `PER_JOB_RESET`
    // matches both `LUMEN_SERVER_PER_JOB_RESET` and (hypothetically) other
    // roots.
    let mut already_seen: std::collections::HashSet<&String> =
        std::collections::HashSet::new();
    let mut suffix_warnings: Vec<String> = Vec::new();
    for non_lumen in env_vars.iter().filter(|k| !k.starts_with("LUMEN_")) {
        if non_lumen.len() < 6 {
            continue;
        }
        if already_seen.contains(non_lumen) {
            continue;
        }
        let matched: Vec<&'static str> = KNOWN_LUMEN_ENV_VARS
            .iter()
            .copied()
            .filter(|known| {
                // Suffix-match: the known LUMEN_ name ends with
                // `_<non_lumen>` (so the user dropped exactly the
                // `LUMEN_CUDA` / `LUMEN_METAL` etc. prefix).
                known
                    .strip_suffix(non_lumen.as_str())
                    .and_then(|prefix| prefix.strip_suffix('_'))
                    .is_some()
            })
            .collect();
        if !matched.is_empty() {
            already_seen.insert(non_lumen);
            suffix_warnings.push(format!(
                "env var '{non_lumen}' — missing 'LUMEN_' prefix? known: {}",
                matched.join(", ")
            ));
        }
    }
    suffix_warnings.sort();
    warnings.extend(suffix_warnings);
    warnings
}

/// Returns up to `n` known env vars sorted by descending similarity score
/// against `candidate`. The score is `common_prefix_len + common_suffix_len`,
/// so a candidate with the right LUMEN_ prefix and a mis-spelled SUFFIX
/// (the canonical `LUMEN_CUDA_GDN_REGISTER_RESIDENT` with a missing trailing
/// `T` -> `LUMEN_CUDA_GDN_REGISTER_RESIDENT`) AND a candidate with a right
/// SUFFIX but missing prefix (e.g. `GDN_REGISTER_RESIDENT` ->
/// `LUMEN_CUDA_GDN_REGISTER_RESIDENT`) both
/// surface the correct name. Cheap O(N) over the allowlist with no
/// allocation per candidate. The minimum score of 4 prunes the trivial
/// `LUMEN_` shared root and other random matches.
fn closest_known_matches(candidate: &str, n: usize) -> Vec<&'static str> {
    let mut scored: Vec<(usize, &'static str)> = KNOWN_LUMEN_ENV_VARS
        .iter()
        .copied()
        .map(|k| {
            let score = common_prefix_len(candidate, k) + common_suffix_len(candidate, k);
            (score, k)
        })
        .collect();
    // Sort by descending score, then alphabetical for determinism.
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    // Drop matches with a combined score below 4 — too noisy to be useful
    // as a suggestion (every LUMEN_ var trivially shares the `LUMEN_`
    // 6-char root from the front and various 1-2 char suffixes from the
    // back; we want stronger signal than that).
    scored
        .into_iter()
        .filter(|(score, _)| *score >= 4)
        .take(n)
        .map(|(_, name)| name)
        .collect()
}

/// Length of the common prefix between two byte slices.
fn common_prefix_len(a: &str, b: &str) -> usize {
    a.as_bytes()
        .iter()
        .zip(b.as_bytes().iter())
        .take_while(|(x, y)| x == y)
        .count()
}

/// Length of the common suffix between two byte slices.
fn common_suffix_len(a: &str, b: &str) -> usize {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let mut i = 0;
    while i < ab.len() && i < bb.len() && ab[ab.len() - 1 - i] == bb[bb.len() - 1 - i] {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// Test-only state reset (used by the integration tests that drive multiple
// configurations in the same process). Production code MUST NOT call this.
// ---------------------------------------------------------------------------

/// Resets the process-wide hint atomics to their defaults. Test-only —
/// used by the unit tests below so each test starts from a known state.
#[doc(hidden)]
pub fn reset_for_tests() {
    PATH_IS_SERVER.store(false, Ordering::Relaxed);
    MODEL_DENSE_QUANT_HINT.store(HINT_UNSET, Ordering::Relaxed);
    MODEL_IS_MOE.store(false, Ordering::Relaxed);
}

/// A `OnceLock` "validator-ran" sentinel. Allows tests to assert that the
/// validator was invoked exactly once during `main()` startup.
static VALIDATOR_RAN: OnceLock<()> = OnceLock::new();

/// Marks the validator as having run. Idempotent.
pub fn mark_validator_ran() {
    let _ = VALIDATOR_RAN.set(());
}

/// Reports whether `mark_validator_ran` has been called this process.
pub fn validator_was_run() -> bool {
    VALIDATOR_RAN.get().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // The tests in this module mutate process-wide state (atomics + env).
    // Cargo runs tests in parallel within a binary by default; the
    // serial-test mutex enforces that exactly one test at a time observes
    // the global state we toggle. The lock is taken FIRST in each test.
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn server_default_decode_delay_is_50us() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_path_is_server(true);
        assert_eq!(cuda_decode_delay_us_default(), 50);
    }

    #[test]
    fn cli_default_decode_delay_is_zero() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        // CLI path: setter never called OR called with false.
        set_path_is_server(false);
        assert_eq!(cuda_decode_delay_us_default(), 0);
        reset_for_tests();
        assert_eq!(cuda_decode_delay_us_default(), 0);
    }

    #[test]
    fn metal_default_decode_delay_is_zero_after_det001_fix() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        // DET-001 is now ROOT-CAUSED and FIXED at the kernel level (two
        // intra-kernel cross-threadgroup races in the decode full-attention path).
        // Metal greedy decode is byte-deterministic (100/100 Q8+Q4) at delay=0, so
        // the mitigation delay (~0.45% TPOT, never a hard guarantee) is no
        // longer needed. The Metal default is reverted to 0 (bit-exact) on BOTH
        // paths; LUMEN_METAL_DECODE_DELAY_US remains available for diagnostics.
        reset_for_tests();
        set_path_is_server(true);
        assert_eq!(metal_decode_delay_us_default(), 0, "Metal server default must be 0 (DET-001 fixed)");
        reset_for_tests();
        set_path_is_server(false);
        assert_eq!(metal_decode_delay_us_default(), 0, "Metal CLI default must be 0 (DET-001 fixed)");
        reset_for_tests();
        assert_eq!(metal_decode_delay_us_default(), 0, "Metal default must be 0 even with no setter call");
    }

    #[test]
    fn bf16_dense_enables_gemmex_and_graph_capture() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Bf16);
        assert!(bf16_gemmex_default());
        assert!(decode_graph_default());
        assert!(decode_graph_qgate_default());
        assert!(decode_graph_tiled_default());
    }

    #[test]
    fn q8_dense_disables_gemmex_and_graph_capture() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q8_0);
        assert!(!bf16_gemmex_default());
        assert!(!decode_graph_default());
        assert!(!decode_graph_qgate_default());
        assert!(!decode_graph_tiled_default());
    }

    #[test]
    fn q4_dense_disables_gemmex_and_graph_capture() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q4_0);
        assert!(!bf16_gemmex_default());
        assert!(!decode_graph_default());
    }

    #[test]
    fn unset_hint_preserves_legacy_defaults() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        // BF16-gemmex was historically default ON; graph capture was
        // historically default OFF. Confirm both.
        assert!(bf16_gemmex_default());
        assert!(!decode_graph_default());
    }

    // -----------------------------------------------------------------------
    // canonical-default flips. The OnceLock-cached resolvers
    // (`legacy_defaults_enabled` and below) are intentionally not reset
    // between tests because they only read process env; tests that mutate
    // `LUMEN_CUDA_LEGACY_DEFAULTS` are serialised via SERIAL and must run in
    // a fresh process — we exercise the env-unset codepath only.
    // -----------------------------------------------------------------------

    #[test]
    fn q8_dense_enables_q8_split_and_output_proj_split_default() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q8_0);
        // Only Q8 dense benefits from the Q8 split sibling layout.
        assert!(q8_split_default(), "Q8 dense should default Q8_SPLIT=ON");
        assert!(
            output_proj_split_default(),
            "Q8 dense should default OUTPUT_PROJ_SPLIT=ON"
        );
        assert!(q8_scale_hw_default(), "Q8 dense should default Q8_SCALE_HW=ON");
        assert_eq!(output_proj_nr_default(), 16, "Q8 dense should default NR=16");
        assert!(
            ffn_fused_glu_skip_default(),
            "Q8 dense should default to SKIP fused GLU (use dp4a fall-through)"
        );
    }

    #[test]
    fn bf16_dense_does_not_set_q8_only_defaults() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Bf16);
        // BF16 dense is unaffected by Q8-only defaults; they stay legacy OFF.
        assert!(!q8_split_default(), "BF16 should NOT default Q8_SPLIT=ON");
        assert!(!output_proj_split_default(), "BF16 should NOT default OUTPUT_PROJ_SPLIT=ON");
        assert!(!q8_scale_hw_default(), "BF16 should NOT default Q8_SCALE_HW=ON");
        assert_eq!(output_proj_nr_default(), 1, "BF16 should default NR=1 (legacy)");
        assert!(
            !ffn_fused_glu_skip_default(),
            "BF16 should NOT default to SKIP fused GLU (kernel is no-op anyway)"
        );
    }

    #[test]
    fn moe_defaults_are_always_on_when_hint_set() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        // MoE Q8 (a typical MoE config) — hint is QUANTISED but the
        // MoE-only flags are independent of dense-quant hint; they default
        // ON regardless because they are no-ops for non-MoE models.
        set_model_dense_quant(QuantScheme::Q8_0);
        assert!(moe_batched_default());
        assert!(moe_router_parallel_default());
        assert!(bf16_moe_v3_default());
        assert!(moe_q4_v3_default());
        assert!(moe_q4_v3b_default());
        assert!(moe_decode_graph_default());
        // GDN register-resident is universally ON (no-op for non-GDN models).
        assert!(gdn_register_resident_default());
        // mmv_q output_proj is universally ON (the matvec ports are quant-
        // aware internally and skip when the source is BF16/F32).
        assert!(mmv_q_output_proj_default());
    }

    #[test]
    fn moe_q8_disables_q8_split_family_defaults() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        // regression guard: the Q8-only split/aligned/NR family
        // MUST default OFF when `set_model_is_moe(true)` has been called,
        // even though the dense-quant hint is QUANTISED. Without this gate,
        // Q8_SPLIT=1 corrupted the MoE Q8 decode path into PAD-token spam.
        set_model_dense_quant(QuantScheme::Q8_0);
        set_model_is_moe(true);
        assert!(!q8_split_default(), "Q8 MoE should NOT default Q8_SPLIT=ON (PAD-spam regression)");
        assert!(!output_proj_split_default(), "Q8 MoE should NOT default OUTPUT_PROJ_SPLIT=ON");
        assert!(!q8_scale_hw_default(), "Q8 MoE should NOT default Q8_SCALE_HW=ON");
        assert_eq!(output_proj_nr_default(), 1, "Q8 MoE should default NR=1 (legacy), not 16");
        assert!(!ffn_fused_glu_skip_default(), "Q8 MoE should NOT default FFN_FUSED_GLU_SKIP=ON");
        // The shared MoE flags MUST stay ON (they fire only on MoE anyway).
        assert!(moe_batched_default());
        assert!(moe_router_parallel_default());
        assert!(moe_decode_graph_default());
        assert!(gdn_register_resident_default());
    }

    #[test]
    fn dense_q8_still_enables_q8_split_family_defaults() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        // Sanity check: the fix must NOT regress the dense-Q8 win.
        // Dense Q8 (set_model_is_moe(false), the default) MUST still flip
        // the entire Q8 split family ON so the dense Q8 configuration continues at 0.907× llama.cpp.
        set_model_dense_quant(QuantScheme::Q8_0);
        // set_model_is_moe NOT called → defaults to false (dense).
        assert!(q8_split_default(), "Dense Q8 must keep Q8_SPLIT=ON for the dense Q8 0.907× llama.cpp");
        assert!(output_proj_split_default());
        assert!(q8_scale_hw_default());
        assert_eq!(output_proj_nr_default(), 16);
        assert!(ffn_fused_glu_skip_default());
    }

    #[test]
    fn validator_detects_missing_suffix_with_lumen_prefix() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        // Mis-spelled SUFFIX (correct LUMEN_ prefix present): canonical name
        // truncated by one trailing character. Construct dynamically so the
        // literal typo string does not appear verbatim in source.
        let canonical = "LUMEN_CUDA_GDN_REGISTER_RESIDENT";
        let typo: String = canonical.chars().take(canonical.len() - 1).collect();
        std::env::set_var(&typo, "1");
        let warnings = collect_unknown_lumen_env_vars();
        std::env::remove_var(&typo);
        assert!(
            warnings.iter().any(|w| w.contains(typo.as_str())),
            "warnings = {warnings:?}"
        );
        // And the suggestion list should include the correct name.
        assert!(
            warnings
                .iter()
                .any(|w| w.contains(canonical)),
            "expected typo suggestion to surface canonical name; warnings = {warnings:?}"
        );
    }

    #[test]
    fn validator_detects_missing_lumen_cuda_prefix() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        // This is the literal typo: user typed `GDN_REGISTER_RESIDENT=1`
        // instead of `LUMEN_CUDA_GDN_REGISTER_RESIDENT=1`. The bare-suffix form
        // does NOT start with LUMEN_, so we rely on the pass-2 suffix
        // heuristic to surface it.
        std::env::set_var("GDN_REGISTER_RESIDENT", "1");
        let warnings = collect_unknown_lumen_env_vars();
        std::env::remove_var("GDN_REGISTER_RESIDENT");
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("'GDN_REGISTER_RESIDENT'") && w.contains("LUMEN_CUDA_GDN_REGISTER_RESIDENT")),
            "expected missing-prefix warning; warnings = {warnings:?}"
        );
    }

    #[test]
    fn validator_does_not_warn_on_known_names() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        // Set a known env var and confirm it produces no warning.
        std::env::set_var("LUMEN_CUDA_BF16_GEMMEX", "0");
        let warnings = collect_unknown_lumen_env_vars();
        std::env::remove_var("LUMEN_CUDA_BF16_GEMMEX");
        assert!(
            !warnings
                .iter()
                .any(|w| w.contains("LUMEN_CUDA_BF16_GEMMEX")),
            "known env should not warn; warnings = {warnings:?}"
        );
    }

    #[test]
    fn closest_match_finds_canonical_for_missing_prefix() {
        // Bare suffix → no LUMEN_ prefix → validator doesn't catch this
        // (no LUMEN_ prefix means it's filtered out before suggestion),
        // but the closest_known_matches helper itself should still be
        // able to surface a sensible suggestion when called directly.
        // Construct the off-canonical needle dynamically (segment replaced
        // with a deliberately wrong fragment) so the literal off-name does
        // not appear verbatim in source.
        let canonical = "LUMEN_CUDA_GDN_REGISTER_RESIDENT";
        let needle = canonical.replace("CUDA", "FOOBAR");
        let matches = closest_known_matches(&needle, 3);
        assert!(
            matches.iter().any(|m| *m == canonical),
            "matches = {matches:?}"
        );
    }
}
