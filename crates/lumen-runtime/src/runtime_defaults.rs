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

/// Records the loaded model's transformer block count (9B = 32 layers,
/// 27B = 64). Called from the CLI / server alongside `set_model_dense_quant`.
/// This is the model-SIZE discriminator the per-class attention-precision
/// default needs: 9B and 27B are otherwise indistinguishable to the resolvers
/// (both dense + same quant hints). 0 = never set (legacy-safe fallback).
pub fn set_model_block_count(num_layers: u32) {
    MODEL_BLOCK_COUNT.store(num_layers, Ordering::Relaxed);
}

static MODEL_BLOCK_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub(crate) fn model_block_count() -> u32 {
    MODEL_BLOCK_COUNT.load(Ordering::Relaxed)
}

/// Per-class default for `LUMEN_CUDA_ATTN_PRECISE` (prefill WMMA attention
/// precision). `2` = pvf32 (exact-F32 P@V, F16 QK^T — heals the WMMA F16
/// P@V-mantissa token-flips at 94.6% of WMMA prefill perf); `0` = legacy
/// F16 WMMA.
///
/// Validated 2026-06-11 (N=3 byte-deterministic quality runs per cell):
/// pvf32 for MoE (all quants) + dense ≤32-layer (9B class) — strict wins
/// everywhere (9B-q8 and MoE cells all reach pristine quality gates; MoE-q8
/// long-form heals fully); default WMMA for the 27B class (64 layers) —
/// pvf32 measurably regresses 27B long-form output (bf16 verylong 3/3→1/3)
/// and softly regresses 27b-q8 shorts.
/// Unset block count (0, legacy callers) → conservative legacy WMMA.
/// `LUMEN_CUDA_ATTN_PRECISE=<0|1|2|3|4>` overrides either way.
pub fn attn_precise_default() -> u8 {
    let layers = model_block_count();
    if model_is_moe() || (layers > 0 && layers <= 32) { 2 } else { 0 }
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

/// True iff the loaded model is a MoE whose primary weights are BF16 (the
/// `MODEL_DENSE_QUANT_HINT` set at load from `output_proj_quant`). Used to gate
/// the GDN alpha/beta → F32-SGEMM fidelity lever ON for MoE BF16 ONLY: it kills
/// the BF16 arith-05 repetition that bf16-native alone leaves, but REGRESSES MoE
/// q8 (adds a DD-REP), so it must not fire on q8/q4 MoE models. Dense BF16
/// (`!model_is_moe()`) and all quantised models return false.
///
/// Referenced only from the CUDA prefill path (`cuda/prefill.rs`), which is
/// gated behind `#[cfg(feature = "cuda")]`. On the default/Metal build the
/// `cuda` module is excluded, leaving this function unreferenced; the
/// `allow(dead_code)` (applied only when `cuda` is OFF) silences the lint
/// without removing the definition, so a future non-cuda caller stays valid
/// and the cuda build is byte-identical (no attribute applied under `cuda`).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn model_is_moe_bf16() -> bool {
    model_is_moe() && MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) == HINT_BF16
}

/// Resolves the per-process default sampler `repetition_penalty` used by the
/// server wire layer (and the CLI's `--repeat-penalty` default) when the
/// operator does not set one explicitly.
///
/// * MoE (Qwen3.5-MoE-35B-A3B class) → `1.03`.
/// * Dense / unset → `1.05` (unchanged — preserves dense behaviour).
///
/// **The 1.08 band-aid is gone; the root cause is fixed.** The MoE q8
/// math-prompt ("Compute 17 times 23 …") computes *correct* products but pure
/// greedy decode used to fall into a near-tie "restate" attractor at the
/// post-`=` token ("17 x 20 = 17 x 20 = …", 4-gram rep ≥ 15, never reaching
/// 391). An elevated `1.08` MoE-only default masked it. The actual root cause
/// is the GatedDeltaNet (GDN) single-token DECODE recurrence, NOT the
/// decode-attention kernel: three structurally different decode-attention
/// kernels (single-block materialise-all, CUDA-graph single-block, FA2
/// split-K online softmax) ALL produced the identical loop, while running the
/// GDN delta-rule state update in F64 (`LUMEN_CUDA_GDN_F64_ACCUM`, now
/// default-ON for MoE via [`gdn_f64_accum_default`]) breaks it and reaches a
/// clean `340 + 51 = 391` at **pure greedy `rp = 1.0`** (A100, q8). The
/// `force_prefill`-clean observation that pointed at "the decode attention
/// kernel" was a mis-localisation: force_prefill rebatches the GDN recurrence.
///
/// **Why 1.03 for MoE (not 1.0, not the dense 1.05) — measured.** With the F64
/// fix the math near-tie now lands correctly at `rp = 1.0`, and a penalty
/// pushes it back OFF: A100 q8 sweep on the math prompt (greedy, temp 0) —
/// `rp = 1.0` → clean 391; `rp = 1.03` → clean 391; `rp = 1.05`
/// (windowed repeat-last-n 64 AND full-history) → CORRUPTED `17 x 20 = 140`,
/// `= 39`. So the dense 1.05 actively breaks MoE arithmetic and cannot be
/// reused. A small residual long-form repetition persists at `rp = 1.0`,
/// independent of the fix — the *sky* prompt loops on a "### N. Scattering"
/// tail at `rp = 1.0` in BOTH the F32 baseline (rep 7) and the F64 build
/// (rep 5). `rp = 1.03` is the empirically-found floor that BOTH preserves the
/// math (clean 391) AND renders the sky cleanly (rep 1). It is a generic
/// long-form guard, NOT a fix for the (now-fixed) math loop.
///
/// Dense keeps 1.05 (no GDN recurrence, not in the restate-loop regime, and
/// dense arithmetic is unaffected by 1.05). The env override
/// (`LUMEN_REPETITION_PENALTY` / `--repeat-penalty`) still wins. Operators who
/// want byte-pure greedy can pass `--repeat-penalty 1.0` and rely on the F64
/// fix for correct math.
pub fn repetition_penalty_default() -> f32 {
    if model_is_moe() {
        // MoE penalty windows are DISJOINT by quant (empirically mapped on
        // Qwen3.5-MoE-35B-A3B, A100; GDN-F64 default-ON lands the math at greedy):
        //   - q8/q4 math is penalty-SENSITIVE: rp>=1.05 penalizes legitimate
        //     digit repetition and corrupts arithmetic ("=39"); 1.03 is the
        //     floor that keeps math correct AND tames q8 long-form (sky rep=1).
        //   - bf16: RE-TUNED 2026-06-09 to 1.03 (was 1.06). The 1.06 was chosen
        //     for bf16 long-form back when bf16 ran the F16-cache FAST_16F GEMM;
        //     with the bf16-native (BF16+F32) MoE projection path now default-ON,
        //     1.06 CORRUPTS bf16 GQ arithmetic (conv-01 "2x100=20", arith-05
        //     misread → 13/15 FAIL) while long-form stays clean at 1.03 (GQ-004
        //     verylong 3/3 over 3072 tokens; GQ-001 14/15 PASS at 1.03).
        // All MoE quants now share 1.03: keeps the F64-fixed math correct
        // (rp>=1.05 corrupts it to "39") AND tames long-form repetition.
        match MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) {
            HINT_BF16 => 1.03,
            _ => 1.03, // q8 / q4 / other quantised MoE
        }
    } else {
        1.05 // dense unchanged
    }
}

/// Per-process default for the server-internal `frequency_penalty` (count-based:
/// `logit[t] -= frequency_penalty * count[t]`). Unlike `repetition_penalty`
/// (penalizes ANY previously-seen token, which corrupts short arithmetic where a
/// digit legitimately repeats once → the 1.03 floor), `frequency_penalty` scales
/// by occurrence COUNT, so a digit repeated once in short math is barely touched
/// while a phrase looped many times in long-form is strongly penalized.
/// `LUMEN_FREQUENCY_PENALTY` overrides. Stays 0.0 (no-op, byte-identical): the
/// 2026-06-09 GQ sweep {0.2,0.4,0.6} REJECTED a nonzero MoE default — it corrupts
/// short arithmetic (0.4 breaks arith-03, 0.6 breaks three) AND does not fix
/// verylong, because the verylong miss is a MODEL failure-to-terminate on long
/// greedy creative generation (the model writes a coherent story then degenerates
/// into a hallucinated fake-conversation tail), not token-frequency repetition.
/// Kept as an opt-in env lever only.
pub fn frequency_penalty_default() -> f32 {
    0.0
}

/// Process-wide default sampling `temperature` used by every surface (CLI
/// `--temperature`, server OpenAI `temperature`, server Anthropic
/// `temperature`) when the operator / client does NOT supply one.
///
/// **`0.7`** — the documented production value. An OpenAI-/Anthropic-style
/// endpoint defaults to *varied* output, and pure-greedy (`temperature 0` + no
/// penalty) deterministically loops on long-form generation, so a small
/// non-zero default keeps out-of-the-box serving coherent. This is the SINGLE
/// canonical no-temperature default; the CLI `--temperature` flag default and
/// both wire surfaces (`unwrap_or_else(default_temperature)`) source it here so
/// they cannot drift (previously the CLI defaulted `0.8` while both wire
/// surfaces used `0.7`, and the CLI help text contradicted its own example).
/// An explicit `temperature` (flag or request field) still wins; `0` selects
/// greedy decoding.
pub fn default_temperature() -> f32 {
    0.7
}

/// Resolves the effective server/CLI-internal `frequency_penalty` when the
/// operator does not pass an explicit flag / the client omits the field.
///
/// Precedence: `LUMEN_FREQUENCY_PENALTY` env (parsed `f32`, kept only when
/// `is_finite() && >= 0.0`) → [`frequency_penalty_default`] (`0.0`, no-op).
/// This is the ONLY place `LUMEN_FREQUENCY_PENALTY` is read; the server wire
/// (`wire::diag_frequency_penalty`) and the CLI (`run.rs`, when `--frequency-
/// penalty` is absent) both delegate here so the env is honoured IDENTICALLY on
/// every surface and is read in exactly one place.
pub fn frequency_penalty_resolved() -> f32 {
    std::env::var("LUMEN_FREQUENCY_PENALTY")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or_else(frequency_penalty_default)
}

/// Resolves the effective server/CLI-internal `repeat_last_n` (the recent-window
/// size for the repetition penalty) when the operator does not pass an explicit
/// flag / the client omits the field.
///
/// Precedence: `LUMEN_REPEAT_LAST_N` env (parsed `usize`) → `None` (the
/// production-identical full-history window). This is the ONLY place
/// `LUMEN_REPEAT_LAST_N` is read; the server wire (`wire::diag_repeat_last_n`)
/// and the CLI (`run.rs`, when `--repeat-last-n` is absent) both delegate here
/// so the env is honoured IDENTICALLY on every surface and is read in exactly
/// one place.
pub fn repeat_last_n_resolved() -> Option<usize> {
    std::env::var("LUMEN_REPEAT_LAST_N")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
}

// ---------------------------------------------------------------------------
// Reasoning ("thinking") control — the SINGLE shared source of truth used
// identically by every Lumen surface (CLI `apply_chat_template`, server
// OpenAI `render_chat_prompt`, server Anthropic `render_prompt`, and the
// `ReasoningExtractor` in `tooling`). Lives here in `lumen-runtime` rather
// than in `lumen-server::wire` because the CLI crate depends on
// `lumen-runtime` but NOT on `lumen-server`; co-locating it here is what
// makes the resolver a literally-shared implementation across all three
// surfaces (the hard consistency requirement) instead of three copies.
// ---------------------------------------------------------------------------

/// Process-wide default for chat "thinking" (reasoning trace) when neither a
/// per-request field nor the `LUMEN_CHAT_ENABLE_THINKING` env override is set.
///
/// **Default `false`** (no reasoning trace; the closed empty-`<think>` tail).
/// MoE and dense share the same default — reasoning is a per-request opt-in,
/// not a model property — so this is intentionally model-agnostic. With the
/// default in force, every surface emits the historical closed
/// `<think>\n\n</think>\n\n` prompt tail and performs NO reasoning extraction,
/// i.e. behaviour is byte-identical to the pre-reasoning-control codebase.
pub fn chat_enable_thinking_default() -> bool {
    false
}

/// Resolves whether chat "thinking" is enabled for a request, applying the
/// canonical precedence used by EVERY surface:
///
/// 1. `per_request` — an explicit per-request field (OpenAI `enable_thinking`
///    / `chat_template_kwargs.enable_thinking`, Anthropic `thinking.type`,
///    CLI `--think`) wins when present.
/// 2. `LUMEN_CHAT_ENABLE_THINKING` env override — applied only when the
///    request did not specify. Accepts `1`/`true`/`yes`/`on` (case-insensitive)
///    as ON and `0`/`false`/`no`/`off` as OFF; any other value is ignored and
///    falls through to the default.
/// 3. [`chat_enable_thinking_default`] (`false`).
///
/// This is the ONLY place the env var is consulted for the prompt tail — the
/// former OpenAI-inline `LUMEN_CHAT_ENABLE_THINKING == "1"` check is folded in
/// here so the three wire/CLI surfaces cannot drift.
pub fn resolve_enable_thinking(per_request: Option<bool>) -> bool {
    if let Some(v) = per_request {
        return v;
    }
    match std::env::var("LUMEN_CHAT_ENABLE_THINKING").ok().as_deref() {
        Some(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => chat_enable_thinking_default(),
        },
        None => chat_enable_thinking_default(),
    }
}

/// Process-wide default reasoning ("thinking") token budget used when a
/// request enables thinking but supplies no explicit `reasoning_budget`.
///
/// This is a SEPARATE budget from the answer `max_tokens` (industry-convergent
/// with Anthropic `thinking.budget_tokens` / Gemini `thinking_budget`) so the
/// answer is never starved by a long reasoning trace. **Part 4** (the decode
/// loop) enforces it via a forced-close; Parts 1-3 only carry it on the
/// request DTO / `JobRequest`. `2048` is a middle-of-the-road default that
/// fits a multi-step reasoning trace without unbounded runaway. The
/// budget is irrelevant (and unused) when thinking is disabled.
pub fn chat_reasoning_budget_default() -> usize {
    2048
}

/// The assistant prompt tail appended after `<|im_start|>assistant\n` for a
/// Qwen3.5-style ChatML template, selected by the resolved `enable_thinking`
/// flag. This is the SINGLE definition of the open-vs-closed `<think>` tail;
/// the CLI and both wire formats call it so they cannot diverge.
///
/// * `enable_thinking == false` → `"<think>\n\n</think>\n\n"` — the closed
///   empty-think block (Qwen3.5 `enable_thinking=false`): the model skips the
///   reasoning scratchpad and answers directly. This is the historical default
///   and is byte-identical to every surface's prior hardcoded string.
/// * `enable_thinking == true` → `"<think>\n"` — an OPEN think block
///   (Qwen3.5 `enable_thinking=true`): the model emits a reasoning trace which
///   the [`tooling::ReasoningExtractor`] then routes to `reasoning_content`.
pub fn think_prompt_tail(enable_thinking: bool) -> &'static str {
    if enable_thinking {
        "<think>\n"
    } else {
        "<think>\n\n</think>\n\n"
    }
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
///
/// **MoE models force graph OFF.** The captured decode graph bakes the
/// MoE routed-expert dispatch (router top-K → per-token expert kernels) and
/// replays the SAME experts every subsequent token, corrupting generation —
/// empirically Qwen3.5-MoE-35B-A3B at BF16 (the only config that turns graph
/// capture on by default) loops on the "sky" prompt and never reaches the math
/// answer, while `LUMEN_CUDA_DECODE_GRAPH=0` is fully coherent (sky correct,
/// math reaches 391). q8/q4 already run eager (graph default off) and route
/// experts correctly. Gate it OFF for MoE so BF16-dense keeps its graph win
/// while BF16-MoE runs the eager (correct) routed-expert path.
pub fn decode_graph_default() -> bool {
    // Validated 2026-06-11: the dense-bf16 graph default is
    // now OFF too — the captured-graph decode replay diverges from eager
    // decode (per-token state read from device pointers captured once),
    // driving medium-length generations into DD-REP and flipping short
    // answers (9b-bf16 13/15·5/8 → 14/15·7/8 with graph OFF, N=3 byte-id;
    // 27b-bf16 with graph-OFF+F64 = PERFECT 15/15·8/8·3/3 from 12/15·7/8·0/3).
    // Decode perf cost: −6.2% (68.89 → 64.63 tok/s, N=5) — correctness wins.
    // COUPLED with `gdn_f64_accum_default` (graph-ON + F64 = 0/3 catastrophic;
    // the two must flip together). No class defaults
    // graph ON anymore; `LUMEN_CUDA_DECODE_GRAPH=1` opts back in.
    false
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

/// Per-process default for `LUMEN_CUDA_GDN_F64_ACCUM` when the operator does
/// not set it explicitly.
///
/// * MoE GDN-hybrid (Qwen3.5-MoE-35B-A3B class) → ON.
/// * Dense / non-MoE → OFF (byte-identical to the historical default; dense
///   has no GDN delta-rule recurrence so the F64 kernels never dispatch
///   anyway — the gate is belt-and-suspenders).
///
/// **Why the MoE GDN model needs F64 accumulation.** The Qwen3.5-MoE-35B is a
/// GatedDeltaNet:full-attn ratio-3 hybrid. Its single-token DECODE path runs
/// the delta-rule recurrence
/// `s = alpha*s + k*((v - alpha*(s·k))*beta)` once per token, accumulating F32
/// rounding into the recurrent state `h_state`. Over a generation the F32 ULP
/// drift diverges from the batched-prefill GDN (which `force_prefill` rebuilds
/// from scratch each step), perturbing the *input* to the next full-attn layer
/// and flipping a near-tie at the post-`=` token. The 256-expert MoE router
/// amplifies that flip into the "17 x 20 = 17 x 20 = …" restate-loop on the
/// math prompt (4-gram rep 15-16, never reaches 391).
///
/// **Empirical isolation (A100, q8, pure greedy rp=1.0).** Three structurally
/// different decode-attention kernels — single-block materialise-all
/// (`attention_decode`), CUDA-graph single-block, and FA2 split-K online
/// softmax (`LUMEN_CUDA_FA2_ATTN=1`) — ALL produce the identical loop, ruling
/// the attention kernel OUT as the cause. Enabling F64 on the GDN phase-4
/// state update (`gdn_phase4_register_resident_f64accum`, the default
/// register-resident decode path) breaks the loop and reaches a clean,
/// arithmetically-correct `340 + 51 = 391` (4-gram rep ≤ 2). The
/// `force_prefill`-clean observation that previously pointed at "the decode
/// attention kernel" was a mis-localisation: force_prefill rebatches the GDN
/// recurrence, which is what it actually fixes.
///
/// The env override (`LUMEN_CUDA_GDN_F64_ACCUM=0/1`) still wins over this
/// default. F64 dispatches only the tiny per-head GDN state-update / norm-gate
/// kernels (not the MoE GEMMs), so the A100 decode-throughput cost is in the
/// noise.
pub fn gdn_f64_accum_default() -> bool {
    // MoE (original) + dense-bf16 (validated 2026-06-11): the
    // F32 GDN delta-rule decode recurrence accumulates ULP drift over long
    // generations into a repetition attractor on dense bf16; F64 heals it —
    // but ONLY with decode-graph OFF (graph-ON + F64 = 0/3 catastrophic:
    // the captured graph replays the F32 kernel and desyncs). COUPLED with
    // `decode_graph_default` — the two must flip together.
    model_is_moe() || MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) == HINT_BF16
}

/// Per-process default for `LUMEN_CUDA_GDN_DECODE_PROJ_MMQ` when the operator
/// does not set it explicitly.
///
/// * MoE GDN-hybrid (Qwen3.5-MoE-35B-A3B class) → ON.
/// * Dense / non-MoE → OFF (byte-identical to the historical decode path).
///
/// **Why the MoE GDN model needs the decode GDN projection to match prefill.**
/// The Qwen3.5-MoE-35B single-token DECODE path computes the GDN q/k/v/gate/
/// alpha/beta projections via the per-token pre-quantized-Q8_1 *tile* matvec
/// (`matvec_q8_0_q8_1` / split / tile kernels), whereas the batched PREFILL
/// path computes the SAME projections through `mmq_q8_0_batched` (the MMQ
/// INT8-dp4a kernel, which `LUMEN_CUDA_Q8_PROJ_MMQ` defaults ON for MoE).
/// These two kernels quantize the activation at a different granularity and
/// reduce in a different order, so the decode GDN output diverges from the
/// (llama-matching) prefill GDN output starting at layer 0. The 256-expert
/// top-K MoE router AMPLIFIES that sub-ULP divergence: at the math near-tie
/// (after id 44896 " multiplication") the decode path flips 7-of-8 expert
/// selections at L0, cascading through all 40 MoE layers and inflating the
/// final `logit[1633]` ("lication") by ~+17 vs prefill — the cosmetic
/// "multiplicationlication" doubling.
///
/// **Empirical isolation (A100, q8, temp 0, raw-token-id dumps).** Re-feeding
/// the identical 44-token prefix as a fresh PREFILL gives `margin(1633−1083) =
/// −19.62` (clean, argmax 1608, matches llama −18.42); the incremental DECODE
/// gives `+10.47` (picks 1633). The flip is 100% in the decode path. Routing
/// the decode GDN projections through the SAME `mmq_q8_0_batched` kernel
/// (batch = 1) the prefill uses aligns the projection numerics, so the router
/// selects the prefill expert set and the final logits match the clean
/// prefill.
///
/// Distinct from `LUMEN_CUDA_GDN_PHASE123_F64` (raising the *recurrence*
/// precision, which REGRESSED): this aligns the *projection GEMM* to the
/// prefill, it does not add precision. The env override
/// (`LUMEN_CUDA_GDN_DECODE_PROJ_MMQ=0/1`) wins.
///
/// **REFUTED / DEFAULT-OFF (2026-06-08).** Routing the decode GDN projection
/// through `mmq_q8_0_batched` at batch=1 did NOT match the batched (batch=N)
/// prefill MMQ numerics — empirically it made q8 math WORSE ("17×23 … = 39",
/// arithmetic broken) while the doubling persisted, i.e. the MMQ kernel has a
/// batch-dependent reduction so batch=1 ≠ batch=N. The projection is therefore
/// NOT the sole decode-vs-prefill divergence (the incremental-vs-batched GDN
/// recurrence also contributes). Kept as opt-in for the record; default OFF so
/// MoE q8 stays at its rep2/391 baseline. The real fix must make decode GDN
/// (projection AND recurrence) numerically equal prefill.
pub fn gdn_decode_proj_mmq_default() -> bool { false }

/// Per-process default for `LUMEN_CUDA_GDN_DECODE_AB_MMQ` (alpha/beta-only
/// decode-vs-prefill projection alignment) when the operator does not set it.
///
/// * MoE (Qwen3.5-MoE-35B-A3B class) → ON.
/// * Dense / non-MoE → OFF (dense decode stays byte-identical to history).
///
/// **What it fixes (empirically isolated, A100, 2026-06-08).** On the MoE
/// GDN-hybrid, the `ssm_alpha` / `ssm_beta` GDN gate-projection weights are
/// stored `Q8Raw` in *every* LBC quant (the GGUF source is F32; the MoE
/// converter force-requantizes them to Q8_0 — see
/// `lumen-convert/src/arch/gdn_gates.rs`). The single-token DECODE projects
/// them through the per-token Q8_1/dp4a `matvec_q8_0_q8_1` kernel, while the
/// batched PREFILL projects them through `mmq_q8_0_batched` (the MMQ
/// INT8→INT32→F32-scale path, default-ON for MoE Q8 via
/// `LUMEN_CUDA_Q8_PROJ_MMQ`). The two INT8 kernels differ in activation-quant
/// granularity and reduction order, so the decode alpha/beta output diverges
/// ~20% from the (llama-matching) prefill output **at layer 0** — where the
/// projection input is the bit-identical token embedding, so the delta is
/// *pure projection kernel*. (The qkv/gate projections were measured
/// bit-identical between decode and prefill — they are NOT the source.) The
/// 20% alpha/beta perturbation propagates through the GDN delta-rule
/// recurrence (`s = α·s + k·((v − α·(s·k))·β)`); the layer-0 GDN output
/// diverges ~28%; the 256-expert top-K router then flips 5-of-8 expert
/// selections at L0 and the error cascades through all 40 layers, derailing
/// greedy decode (sub-word doubling, false products, restate loops).
///
/// **The fix:** route the DECODE alpha/beta projection through the SAME
/// `mmq_q8_0_batched` kernel (batch = 1) the prefill uses, from the F32
/// RMSNorm output — matching the INT8 reduction order decode==prefill. The
/// MMQ kernel is per-token-independent (each `(token, row-group)` CUDA block
/// reads only its own `x_row`, with no cross-token coupling), so batch=1
/// is bit-identical to row 0 of batch=N — the projection delta at L0 goes to
/// zero. qkv/gate are left on their existing (bit-identical) decode paths.
///
/// **Distinct from the REFUTED `gdn_decode_proj_mmq`** (which MMQ'd *all four*
/// qkv/alpha/beta/gate and required all-four-Q8Raw so it never engaged for
/// bf16, and on q8 MMQ'd the large qkv projection — that broke q8 math). This
/// is alpha/beta-ONLY and engages for any quant whose alpha/beta are Q8Raw
/// (bf16, q8, q4). Distinct from `LUMEN_CUDA_GDN_PHASE123_F64` (raising
/// recurrence precision, which REGRESSED): this aligns the projection GEMM to
/// prefill, it does not add precision. The env override
/// (`LUMEN_CUDA_GDN_DECODE_AB_MMQ=0/1`) wins.
///
/// **MEASURED / DEFAULT-OFF (2026-06-08, A100, GQ-001).** Engagement CONFIRMED
/// (one-shot `[ABMMQENG]` probe: `gdn_decode_ab_mmq=true` for bf16, all guards
/// pass, `gdn_use_preq=false` so the bf16 `else` branch MMQs alpha/beta), but
/// the projection alignment is **necessary-but-not-sufficient** and net-zero
/// to net-negative on the suite:
///   * **bf16 11/15 → 11/15 (byte-identical** — the "multiplicationulation"
///     doubling and "10246" garble survive UNCHANGED even though decode
///     alpha/beta now go through the SAME MMQ as prefill). This proves the GDN
///     **recurrence** (incremental single-token vs batched scan — the diag's
///     2nd divergence; L0 output relD 28% ≫ the 20% projection relD) dominates
///     the residual divergence, NOT the projection. The defect is the diffuse
///     decode-vs-prefill GDN fidelity gap amplified by the 256-expert router,
///     not a single projection kernel.
///   * **q8 10/15 → 9/15 (REGRESSED** — 144/12 newly loops; 1000−37 newly
///     digit-spams "1000000…" / DD-CHARSPAM). The alignment shifted the q8
///     near-tie into a worse basin — same failure class as the refuted
///     `gdn_decode_proj_mmq` "=39" and `GDN_PHASE123_F64`.
///   * **q4 9/15 → 11/15 (IMPROVED** — 144/12 and 256+768 now clean).
/// Mixed across quants with a NEW catastrophic q8 mode ⟹ default OFF (no quant
/// regresses; all three stay at their baseline). Kept as an opt-in env lever
/// for the record (architecturally-correct prefill-matching projection; helps
/// q4 in isolation). The true fix requires aligning the decode GDN RECURRENCE
/// to prefill (the dominant remaining divergence), which prior work
/// (`GDN_PHASE123_F64`) found regresses — i.e. it is the documented diffuse
/// cross-engine numeric-fidelity problem, not closable by this projection
/// lever alone.
pub fn gdn_decode_ab_mmq_default() -> bool { false }

/// Per-process default for `LUMEN_CUDA_GDN_AB_F16` — route the GDN
/// `ssm_alpha` / `ssm_beta` projections through a pre-dequanted **F16**
/// cache and cuBLAS `cublasGemmEx` (HGEMV in decode, HGEMM in prefill) in
/// BOTH paths, MoE-gated.
///
/// The GDN `ssm_alpha` / `ssm_beta` weights are stored `Q8Raw` in every LBC
/// quant (the GGUF source is F32; the MoE converter force-requantizes them to
/// Q8_0). With the keeper Q8-prefill-MMQ default ON, the batched PREFILL
/// projects them via `mmq_q8_0_batched` (INT8 MMA) while the single-token
/// DECODE uses the per-token Q8_1/dp4a `matvec_q8_0_q8_1` tile matvec — a
/// DIFFERENT activation-quant granularity + INT8 reduction order. The
/// `[GDNPROJSS]` whole-buffer-sumsq probe at GDN L0 measured this as
/// alpha relD 19.45% / beta relD 20.96% decode-vs-prefill, while the
/// (F16/bf16) qkv + gate projections were 0.000% (BIT-IDENTICAL). The
/// 256-expert top-K router amplifies the ~20% alpha/beta divergence into a
/// 5-of-8 expert flip that cascades 40 layers and derails greedy decode.
///
/// This lever dequant the `Q8Raw` alpha/beta weights to an F16 cache once at
/// load (mirroring the existing GDN F16 weight-cache mechanism) and routes
/// BOTH decode (`cublasGemmEx` N=1, `CUDA_R_16F` × `CUDA_R_16F`,
/// `COMPUTE_32F_FAST_16F`) and prefill (`cublasGemmEx` N=batch, identical
/// dtypes/compute-type) through it — the EXACT recipe that makes qkv/gate
/// bit-identical. batch=1 == row 0 of batch=N under the same GEMM, so the L0
/// alpha/beta delta collapses to ~0% at its source. Distinct from the refuted
/// `gdn_decode_ab_mmq` (which used INT8 MMQ batch=1, found net-negative) and
/// from `GDN_PHASE123_F64` (recurrence precision, regressed).
///
/// MoE-default-ON (2026-06-09 GQ validation: the parity stack makes MoE q8/q4
/// PRISTINE and clears bf16 gross garble); dense byte-identical (gate requires
/// `model_is_moe()`). Set `LUMEN_CUDA_GDN_AB_F16=0|1` to override the per-model default.
pub fn gdn_ab_f16_default() -> bool { true }

/// Per-process default for `LUMEN_CUDA_GDN_RECUR_PREFILL_ORDER` — route the GDN
/// **decode** delta-rule state update through a variant whose arithmetic is
/// reordered to match the **prefill** batched-scan single-token step
/// bit-for-bit, MoE-gated.
///
/// Once the `ssm_alpha`/`ssm_beta` projection is made bit-identical
/// decode-vs-prefill (`LUMEN_CUDA_GDN_AB_F16`), the residual GDN L0 divergence
/// is the RECURRENCE: the decode F64 phase4 kernel
/// (`gdn_phase4_register_resident_f64accum`) and the prefill F64 scan
/// (`gdn_prefill_fused_v3_f64accum`) are the SAME precision (F64) and SAME
/// warp-reduce structure but apply the alpha decay in a DIFFERENT algebraic
/// order: prefill folds `alpha` into each state element BEFORE the K reduction
/// (`retrieval = SUM (alpha*s_r)*k_r`), while decode reduces the raw `S.k`
/// first and multiplies by `alpha` ONCE after (`delta = (v - alpha*SUM s_r*k_r)*beta`).
/// These are algebraically equal but round differently even in F64; the
/// 256-expert router amplifies the last-bit delta into expert flips. This
/// lever swaps in `gdn_phase4_register_resident_f64accum_prefillorder`, which
/// uses the prefill ordering exactly so the decode recurrence step == the
/// prefill recurrence step. Distinct from `GDN_PHASE123_F64` (which RAISES the
/// conv1d/L2 precision and REGRESSED) — this changes ONLY the delta-rule order,
/// matching prefill rather than exceeding it.
///
/// Default-OFF until proven net-positive on the GQ suite; dense byte-identical
/// (gate requires `model_is_moe()`). Set
/// `LUMEN_CUDA_GDN_RECUR_PREFILL_ORDER=0|1` to override the per-model default.
pub fn gdn_recur_prefill_order_default() -> bool { false }

/// Per-process default for `LUMEN_CUDA_GDN_PHASE123_ALIGN` — align the GDN
/// **decode** phase123 (conv1d + SiLU + L2-norm) to the **prefill** phase123
/// bit-for-bit, MoE-gated.
///
/// Source localization (after `LUMEN_CUDA_GDN_AB_F16` made the projection
/// bit-identical and phase4-reorder proved net-zero): the default MoE decode
/// runs the F32 `gdn_phase123_register_resident` (conv1d F32 + L2-norm F32),
/// while the MoE prefill runs `ssm_conv1d_silu_prefill` (conv1d F32) +
/// `l2_normalize_qk_strided_f64accum` (L2-norm **F64**, engaged because
/// `gdn_f64_accum_default()` = `model_is_moe()` = true). The conv1d is already
/// bit-identical (both F32, identical tap order); the SOLE phase123 divergence
/// is the L2-norm PRECISION/reduction (decode F32 vs prefill F64). This lever
/// swaps in `gdn_phase123_register_resident_alignl2`, which keeps conv1d in F32
/// (so it is NOT the regressed `GDN_PHASE123_F64`, which raised conv1d to F64
/// too) and computes the per-head L2-norm with the EXACT F64 reduction of the
/// prefill `l2_normalize_qk_strided_f64accum`. The decode q_norm/k_norm then
/// match a prefill of the same token bit-for-bit, collapsing the residual GDN
/// L0-output divergence (a_sumsq/x_sumsq relD ~27-28%) at its true source.
///
/// Default-OFF until proven net-positive on the GQ suite; dense byte-identical
/// (gate requires `model_is_moe()`). Run together with `LUMEN_CUDA_GDN_AB_F16=1`
/// (projection prerequisite). Set `LUMEN_CUDA_GDN_PHASE123_ALIGN=0|1` to
/// override the per-model default.
pub fn gdn_phase123_align_default() -> bool { false }

/// Per-process default for `LUMEN_CUDA_GDN_DECODE_VIA_PREFILL` — the combined
/// GDN-decode==GDN-prefill structural-parity lever (ALL GDN models).
///
/// All prior single-lever fixes (AB_F16 projection, phase123-align L2-norm,
/// phase4-reorder, F64-megakernel) each aligned ONE decode-vs-prefill
/// divergence and only SHUFFLED near-ties because the divergence is DIFFUSE.
/// This lever makes the WHOLE GDN decode recurrence block byte-equivalent to a
/// prefill of the same single position AT ONCE: for MoE GDN layers in decode it
/// dispatches the PREFILL fused kernels (`ssm_conv1d_silu_prefill` +
/// `gdn_compute_gates_batched` + `l2_normalize_qk_strided[_f64accum]` +
/// `gdn_prefill_fused_v3[_f64accum]` + `gdn_prefill_norm_gate[_f64accum]`) at
/// `T=1` on the single new token, carrying the persistent `h_state` /
/// `conv_state`, INSTEAD of the decode megakernel / register-resident phase4
/// recurrence (which compute a structurally different update — ~0.98%/step
/// `h_state` drift vs the prefill scan, NOT a precision artefact). Combined with
/// `LUMEN_CUDA_GDN_AB_F16=1` (alpha/beta projection → F16, collapsing the L0
/// ~20% projection divergence), GDN-decode == GDN-prefill BY CONSTRUCTION.
///
/// MoE-default-ON (2026-06-09 GQ validation: the parity stack makes MoE q8/q4
/// PRISTINE and clears bf16 gross garble). DENSE-default-ON for NON-BF16
/// quants since 2026-06-10 (validated N≥3 byte-deterministic):
/// the same per-step recurrence drift accumulates on dense over long
/// generations — 9B-q8 GQ-004 verylong 0/3 (deterministic N=3, stuck at token
/// cap in a DD-REP/CHARSPAM attractor) flips to 3/3 with clean EOS under
/// via-prefill ALONE (N=5 incl. 27B; 27b-q4 goes PRISTINE); decode tok/s flat
/// (-0.6%). AB_F16/CONVSTATE_PARITY stay MoE-only (dense ablation:
/// unnecessary; CONVSTATE-without-AB is harmful).
///
/// **DENSE BF16 ≥33-layer stays on the legacy decode path (ablation 2026-06-10):**
/// via-prefill-alone CORRUPTS dense bf16 — 9b-bf16 4/15·0/8·0/3 and 27b-bf16
/// 0✓/3✗ with every DD detector firing, vs the legacy path's sane-but-imperfect
/// 13/15·5/8·2/3. `LUMEN_CUDA_GDN_F64_ACCUM=1` does NOT rescue it (still 0/3),
/// so the interaction is not a precision artefact of the recurrence; the
/// difference vs MoE-bf16 (where via-prefill works) is the MoE-only projection
/// aligners (AB_F16/CONVSTATE) — root-cause investigation tracked separately.
/// Set `LUMEN_CUDA_GDN_DECODE_VIA_PREFILL=0|1` to override either way.
pub fn gdn_decode_via_prefill_default() -> bool {
    // Dense bf16 carve-back (validated 2026-06-11): on the
    // production binary (pvf32 per-class attention), via-prefill HEALS the
    // 9B-bf16 verylong attractor (1/3 -> 3/3, N=3 byte-identical, matches
    // llama.cpp bf16) — the earlier "via-prefill catastrophic on dense bf16"
    // result was measured on the pre-pvf32 binary and is OBSOLETE for the 9B
    // class. 27B-bf16 is the refutation boundary: via-prefill regresses its
    // pristine short gate (15/15 -> 14/15), so the bf16 enable is scoped to
    // the same <=32-layer class as `attn_precise_default()`. Unset block
    // count (legacy callers) keeps bf16 on the legacy decode path.
    let dense_bf16 = MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) == HINT_BF16;
    let small = { let l = model_block_count(); l > 0 && l <= 32 };
    model_is_moe() || !dense_bf16 || small
}

/// Per-process default for `LUMEN_CUDA_GDN_CONVSTATE_PARITY` — make the decode
/// GDN `conv_state` bit-match a true prefill of the same position (MoE-gated).
///
/// With `LUMEN_CUDA_GDN_DECODE_VIA_PREFILL=1` the decode conv1d runs the EXACT
/// prefill `ssm_conv1d_silu_prefill` kernel on the conv ring buffer, so the only
/// residual decode-vs-prefill divergence is the SINGLE new ring slot — the qkv
/// projection of the new token. The decode qkv projection uses a different
/// cuBLAS kernel/algo (N=1 GEMV: native-BF16 `cublasGemmEx` with the autotuned
/// `bf16_algo_for` algo / per-token Q8_1 dp4a / aligned-Q8 matvec) than the
/// prefill (N=batch GEMM via `launch_gemm_projection`: `CUBLAS_GEMM_DEFAULT_`
/// `TENSOR_OP` BF16 GemmEx for bf16, MMQ INT8/INT4 for q8/q4). That kernel-class
/// mismatch injects a ~0.0014% per-element qkv delta that the conv1d window
/// dot-product + SiLU amplify into a ~5% `conv_state` divergence at L0 — which
/// the 256-expert router then turns into expert-rank swaps and a mild
/// number-misread / arithmetic-slip degeneration (the genuine bf16 residual).
///
/// When ON, the decode GDN qkv projection (the buffer that feeds the conv ring,
/// `gdn.qkv_buf`) is computed via the SAME `launch_gemm_projection` path the
/// prefill uses, at `batch = 1` — same cuBLAS algo (DEFAULT_TENSOR_OP) for bf16,
/// same MMQ INT8/INT4 reduction for q8/q4 — exactly as `GDN_AB_F16` already does
/// for the alpha/beta projection. This collapses the new-slot qkv delta, so the
/// decode `conv_state` (whose carried-in slots are already prefill-written and
/// bit-identical) bit-matches a true prefill of that position; `h_state` and the
/// router then follow toward the prefill trajectory. Only the qkv projection is
/// rerouted (gate/alpha/beta keep their existing, already-aligned paths).
///
/// MoE-default-ON (2026-06-09 GQ validation: bit-identical decode conv_state
/// lifted MoE q8 to PRISTINE; q4 byte-identical via the Q4Raw exclusion); dense
/// byte-identical (gate requires `model_is_moe()`). Requires
/// `LUMEN_CUDA_GDN_DECODE_VIA_PREFILL` to be effective (it only matters for the
/// via-prefill conv consume). Set `LUMEN_CUDA_GDN_CONVSTATE_PARITY=0|1` to override.
pub fn gdn_convstate_parity_default() -> bool { true }

/// Per-process default for the greedy anti-degeneration guard
/// (`SamplingParams::anti_restate`).
///
/// * MoE (Qwen3.5-MoE-35B-A3B class) → ON.
/// * Dense / non-MoE → OFF (dense decode stays byte-identical to history).
///
/// **What it fixes.** On the MoE GDN-hybrid the per-quant CUDA decode forward
/// produces a single near-tie at one step where Lumen's top-1 logit is a
/// degenerate continuation of the just-emitted text that llama.cpp does NOT
/// pick: a sub-word doubling (" multiplication" → "lication", rendering
/// "multiplicationlication") and a short n-gram restate ("17 × 20 = 17 × 20",
/// the q4 "340 + 51 = 340 + 51" loop). This is a *near-tie flip*, not a
/// magnitude error — confirmed by bf16 (near-full precision) reproducing the
/// doubling and by F64 GDN accumulation shifting but not removing it. Because
/// the divergence is a sub-ULP logit-margin disagreement at a single greedy
/// step, no precision lever or repetition-penalty value removes it without
/// collateral arithmetic corruption (rp ≥ 1.05 breaks the math).
///
/// The guard is a deterministic, backend-agnostic veto applied AFTER the
/// argmax: it only fires on a genuine degenerate restatement and otherwise
/// returns the plain argmax unchanged, so it never perturbs coherent text and
/// is safe to default ON for MoE. The override (`LUMEN_ANTI_RESTATE=0/1`)
/// wins; operators who want byte-pure greedy can disable it.
pub fn anti_restate_default() -> bool {
    match std::env::var("LUMEN_ANTI_RESTATE").ok().as_deref() {
        Some("0" | "false" | "no" | "off") => false,
        Some(_) => true,
        // BF16 MoE ONLY. The anti-degeneration veto is INCOMPATIBLE with the
        // QUANTISED (q8/q4) MoE math path: token-level A/B on Qwen3.5-MoE-35B
        // (A100, temp 0, raw-token-id dumps) proves the q8/q4 "Compute 17×23"
        // greedy trajectory reaches the correct `…= 340 + 51 = 391` ONLY when
        // the veto is OFF. The veto's sub-word-doubling rule flips the single
        // token at the word "multiplication" (id 44896 → 1633 "…lication"
        // vs 2820); the 1633 branch carries BOTH the cosmetic doubling AND the
        // arithmetic that lands 391, while the vetoed 2820 branch deterministly
        // routes into a "17×20 = 17×20 = …" loop that never emits 391. The
        // loop/ngram id-level rules likewise veto the high-frequency digit /
        // space / operator tokens the answer needs, redirecting the bounded
        // fallback into the same loop. So for q8/q4 MoE every rule combination
        // REGRESSES a passing rep≤2 / 391 baseline into a non-terminating loop;
        // the documented baseline (veto OFF) is the correct, PASSING state and
        // its only blemish is a cosmetic doubling inside an English WORD, not
        // the arithmetic (391 is present and correct).
        //
        // BF16 MoE reaches 391 from a DIFFERENT basin whose token at "multipl-
        // ication" is not the vetoed near-tie, so there the veto cleanly removes
        // the doubling AND keeps 391 — a genuine win. BF16 is distinguishable
        // from q8/q4 by the dense-quant hint (`HINT_BF16` vs `HINT_QUANTISED`),
        // so no expert-quant probe is needed for this gate (the MMQ-Q4-default
        // gate DOES need one because q8/q4 share `HINT_QUANTISED`). Operators
        // who want the veto on a quantised MoE anyway can force `=1`.
        None => model_is_moe() && MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) == HINT_BF16,
    }
}

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
    "LUMEN_ANTI_RESTATE",
    "LUMEN_ANTI_RESTATE_SUBWORD",
    "LUMEN_ANTI_RESTATE_NGRAM",
    "LUMEN_ANTI_RESTATE_LOOP",
    "LUMEN_BASE_URL",
    "LUMEN_BENCH_ITERATIONS",
    "LUMEN_BENCH_SCALE",
    "LUMEN_BENCH_TOKENS",
    "LUMEN_BENCH_WARMUP",
    "LUMEN_CACHE_DIR",
    "LUMEN_CHAT_ENABLE_THINKING",
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
    "LUMEN_CUDA_GDN_AB_F16",
    "LUMEN_CUDA_GDN_AB_F32",
    "LUMEN_CUDA_GDN_CONVSTATE_PARITY",
    "LUMEN_CUDA_GDN_DECODE_AB_MMQ",
    "LUMEN_CUDA_GDN_DECODE_MEGAKERNEL_F64",
    "LUMEN_CUDA_GDN_DECODE_PROJ_MMQ",
    "LUMEN_CUDA_GDN_DECODE_VIA_PREFILL",
    "LUMEN_CUDA_GDN_F64_ACCUM",
    "LUMEN_CUDA_GDN_PHASE123_ALIGN",
    "LUMEN_CUDA_GDN_PHASE123_F64",
    "LUMEN_CUDA_GDN_RECUR_PREFILL_ORDER",
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
    "LUMEN_CUDA_MOE_BF16_NATIVE",
    "LUMEN_CUDA_MOE_DEBUG_DUMP",
    "LUMEN_CUDA_MOE_DECODE_F32",
    "LUMEN_CUDA_MOE_DECODE_F32_FFN",
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
    "LUMEN_FORCE_PREFILL_DECODE",
    "LUMEN_FREQUENCY_PENALTY",
    "LUMEN_GRAPH_DIAGNOSTIC",
    "LUMEN_KV_PRECISION",
    "LUMEN_CUDA_ATTN_PRECISE",
    "LUMEN_CUDA_ATTN_PRECISE_DBG",
    "LUMEN_LOGIT_DUMP",
    "LUMEN_LOGIT_DUMP_A",
    "LUMEN_LOGIT_DUMP_B",
    "LUMEN_METAL_ATTN_PRECISE",
    "LUMEN_METAL_ATTN_PRECISE_DBG",
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
    "LUMEN_MOE_PROBE",
    "LUMEN_QWEN35_9B_BF16",
    "LUMEN_QWEN35_9B_PATH",
    "LUMEN_QWEN35_9B_Q4",
    "LUMEN_QWEN35_9B_Q8",
    "LUMEN_REPEAT_LAST_N",
    "LUMEN_REPETITION_PENALTY",
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
    MODEL_BLOCK_COUNT.store(0, Ordering::Relaxed);
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
        // Validated 2026-06-11: graph capture is OFF for dense
        // bf16 too — the captured-graph decode replay diverges from eager
        // decode (DD-REP on medium gens; 27b-bf16 graph-OFF+F64 = PERFECT).
        assert!(!decode_graph_default());
        assert!(!decode_graph_qgate_default());
        assert!(!decode_graph_tiled_default());
    }

    #[test]
    fn attn_precise_default_per_class() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        // Validated 2026-06-11: pvf32 (2) for MoE + dense
        // <=32-layer (9B); legacy WMMA (0) for the 27B class and for legacy
        // callers that never set the block count.
        reset_for_tests();
        assert_eq!(attn_precise_default(), 0, "unset block count -> legacy WMMA");
        reset_for_tests();
        set_model_block_count(32);
        assert_eq!(attn_precise_default(), 2, "dense 9B (32 layers) -> pvf32");
        reset_for_tests();
        set_model_block_count(64);
        assert_eq!(attn_precise_default(), 0, "dense 27B (64 layers) -> legacy WMMA");
        reset_for_tests();
        set_model_block_count(64);
        set_model_is_moe(true);
        assert_eq!(attn_precise_default(), 2, "MoE -> pvf32 regardless of size");
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
    fn repetition_penalty_default_moe_per_quant_dense_1_05() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());

        // Dense keeps 1.05 (no GDN recurrence; arithmetic unaffected).
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q8_0);
        assert!(
            (repetition_penalty_default() - 1.05).abs() < f32::EPSILON,
            "dense Q8 keeps repetition_penalty default 1.05"
        );

        reset_for_tests();
        set_model_dense_quant(QuantScheme::Bf16);
        assert!(
            (repetition_penalty_default() - 1.05).abs() < f32::EPSILON,
            "dense BF16 keeps repetition_penalty default 1.05"
        );

        // MoE is PER-QUANT (the 1.08 band-aid is removed; GDN F64 fixes the math
        // loop at rp=1.0). All MoE quants → 1.03: the floor that preserves the
        // F64-fixed math (rp>=1.05 corrupts it to "39") while taming long-form.
        // bf16 was RE-TUNED 1.06→1.03 on 2026-06-09 (the bf16-native path makes
        // 1.03 sufficient for long-form — GQ-004 verylong 3/3 — and 1.06 was
        // corrupting bf16 GQ arithmetic).
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q8_0);
        set_model_is_moe(true);
        assert!(
            (repetition_penalty_default() - 1.03).abs() < f32::EPSILON,
            "MoE Q8 must default repetition_penalty to 1.03 (>=1.05 breaks math)"
        );
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Bf16);
        set_model_is_moe(true);
        assert!(
            (repetition_penalty_default() - 1.03).abs() < f32::EPSILON,
            "MoE BF16 must default repetition_penalty to 1.03 (1.06 corrupted GQ arithmetic; long-form clean at 1.03)"
        );

        // Unset (no setters): dense 1.05.
        reset_for_tests();
        assert!(
            (repetition_penalty_default() - 1.05).abs() < f32::EPSILON,
            "unset hint defaults to dense 1.05"
        );

        reset_for_tests();
    }

    #[test]
    fn gdn_f64_accum_default_is_moe_gated() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());

        // Dense (set_model_is_moe NOT called → false): OFF. Dense models have
        // no GDN delta-rule recurrence, so the F64 kernels never dispatch; the
        // gate is belt-and-suspenders and must stay OFF for byte-identity.
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q8_0);
        assert!(!gdn_f64_accum_default(), "dense Q8 must default GDN_F64_ACCUM OFF");

        // Validated 2026-06-11: dense BF16 now defaults F64 ON —
        // the F32 GDN delta-rule decode recurrence accumulates ULP drift into
        // a repetition attractor on long generations; F64 heals it (coupled
        // with decode-graph OFF).
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Bf16);
        assert!(gdn_f64_accum_default(), "dense BF16 must default GDN_F64_ACCUM ON (GAP-D)");

        // MoE (set_model_is_moe(true)): ON for both q8 and bf16 — F64 on the
        // GDN single-token state update removes the decode-vs-prefill ULP
        // drift that triggered the q8 greedy restate-loop.
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q8_0);
        set_model_is_moe(true);
        assert!(gdn_f64_accum_default(), "MoE Q8 must default GDN_F64_ACCUM ON");

        reset_for_tests();
        set_model_dense_quant(QuantScheme::Bf16);
        set_model_is_moe(true);
        assert!(gdn_f64_accum_default(), "MoE BF16 must default GDN_F64_ACCUM ON");

        // Unset (no setters): OFF (dense default).
        reset_for_tests();
        assert!(!gdn_f64_accum_default(), "unset hint defaults GDN_F64_ACCUM OFF");

        reset_for_tests();
    }

    #[test]
    fn gdn_decode_ab_mmq_default_is_off() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());

        // Default OFF for EVERY model class. Empirically (A100 GQ-001,
        // 2026-06-08) the alpha/beta projection→MMQ alignment is net-zero on
        // bf16 (recurrence dominates), REGRESSES q8 (new DD-CHARSPAM), and only
        // helps q4 — mixed across quants with a new catastrophic q8 mode, so it
        // stays OFF (no quant regresses). Kept as an opt-in env lever
        // (`LUMEN_CUDA_GDN_DECODE_AB_MMQ=1`). Must be OFF for dense AND MoE.
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q8_0);
        assert!(!gdn_decode_ab_mmq_default(), "dense Q8 must default GDN_DECODE_AB_MMQ OFF");

        reset_for_tests();
        set_model_dense_quant(QuantScheme::Bf16);
        assert!(!gdn_decode_ab_mmq_default(), "dense BF16 must default GDN_DECODE_AB_MMQ OFF");

        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q8_0);
        set_model_is_moe(true);
        assert!(!gdn_decode_ab_mmq_default(), "MoE Q8 must default GDN_DECODE_AB_MMQ OFF (regresses)");

        reset_for_tests();
        set_model_dense_quant(QuantScheme::Bf16);
        set_model_is_moe(true);
        assert!(!gdn_decode_ab_mmq_default(), "MoE BF16 must default GDN_DECODE_AB_MMQ OFF (net-zero)");

        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q4_0);
        set_model_is_moe(true);
        assert!(!gdn_decode_ab_mmq_default(), "MoE Q4 must default GDN_DECODE_AB_MMQ OFF (lever opt-in)");

        // Unset (no setters): OFF.
        reset_for_tests();
        assert!(!gdn_decode_ab_mmq_default(), "unset hint defaults GDN_DECODE_AB_MMQ OFF");

        reset_for_tests();
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
    fn gdn_phase123_align_default_is_off_and_env_is_known() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        // Default-OFF: the lever must not engage unless explicitly enabled
        // (the MoE gating is applied separately in backend_impl's resolver).
        assert!(
            !gdn_phase123_align_default(),
            "GDN_PHASE123_ALIGN must default OFF (net-positive-only rule)"
        );
        // The env var must be in the canonical allowlist so it does not
        // false-fire the unknown-LUMEN-var validator when set on the server.
        assert!(
            KNOWN_LUMEN_ENV_VARS.contains(&"LUMEN_CUDA_GDN_PHASE123_ALIGN"),
            "LUMEN_CUDA_GDN_PHASE123_ALIGN must be in KNOWN_LUMEN_ENV_VARS"
        );
        std::env::set_var("LUMEN_CUDA_GDN_PHASE123_ALIGN", "1");
        let warnings = collect_unknown_lumen_env_vars();
        std::env::remove_var("LUMEN_CUDA_GDN_PHASE123_ALIGN");
        assert!(
            !warnings
                .iter()
                .any(|w| w.contains("LUMEN_CUDA_GDN_PHASE123_ALIGN")),
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

    // ---- Reasoning ("thinking") control ----

    #[test]
    fn chat_enable_thinking_default_is_false() {
        // The default MUST be false so every surface stays byte-identical to
        // the pre-reasoning-control behaviour when nothing opts in.
        assert!(!chat_enable_thinking_default());
    }

    #[test]
    fn think_prompt_tail_open_vs_closed_strings_are_exact() {
        // These two literals are the SINGLE source of the open/closed tail
        // that the CLI, OpenAI, and Anthropic surfaces all append. Pin them
        // byte-for-byte; the closed form must match the historical hardcoded
        // string in every surface's prior implementation.
        assert_eq!(think_prompt_tail(false), "<think>\n\n</think>\n\n");
        assert_eq!(think_prompt_tail(true), "<think>\n");
    }

    #[test]
    fn resolve_enable_thinking_per_request_wins_over_env_and_default() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        // Save & restore the env var so we never leak global state to a
        // sibling test (the SERIAL lock makes this safe to mutate here).
        let saved = std::env::var("LUMEN_CHAT_ENABLE_THINKING").ok();

        // Per-request Some(_) is authoritative regardless of env.
        std::env::set_var("LUMEN_CHAT_ENABLE_THINKING", "1");
        assert!(!resolve_enable_thinking(Some(false)), "per-request false beats env=1");
        std::env::set_var("LUMEN_CHAT_ENABLE_THINKING", "0");
        assert!(resolve_enable_thinking(Some(true)), "per-request true beats env=0");

        match saved {
            Some(v) => std::env::set_var("LUMEN_CHAT_ENABLE_THINKING", v),
            None => std::env::remove_var("LUMEN_CHAT_ENABLE_THINKING"),
        }
    }

    #[test]
    fn resolve_enable_thinking_env_override_when_request_absent() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let saved = std::env::var("LUMEN_CHAT_ENABLE_THINKING").ok();

        // Env override applies only when per_request is None. Accept the
        // canonical truthy/falsy spellings; bogus values fall to the default.
        for truthy in ["1", "true", "yes", "on", "ON", "True"] {
            std::env::set_var("LUMEN_CHAT_ENABLE_THINKING", truthy);
            assert!(resolve_enable_thinking(None), "env '{truthy}' should enable");
        }
        for falsy in ["0", "false", "no", "off", "OFF"] {
            std::env::set_var("LUMEN_CHAT_ENABLE_THINKING", falsy);
            assert!(!resolve_enable_thinking(None), "env '{falsy}' should disable");
        }
        std::env::set_var("LUMEN_CHAT_ENABLE_THINKING", "garbage");
        assert_eq!(
            resolve_enable_thinking(None),
            chat_enable_thinking_default(),
            "unparseable env falls through to the default"
        );
        std::env::remove_var("LUMEN_CHAT_ENABLE_THINKING");
        assert_eq!(
            resolve_enable_thinking(None),
            chat_enable_thinking_default(),
            "absent env + absent request == default"
        );

        match saved {
            Some(v) => std::env::set_var("LUMEN_CHAT_ENABLE_THINKING", v),
            None => std::env::remove_var("LUMEN_CHAT_ENABLE_THINKING"),
        }
    }

    // ---- F3: canonical no-temperature default ----

    #[test]
    fn default_temperature_is_0_7() {
        // The SINGLE canonical no-temperature default sourced by the CLI flag
        // default and both wire surfaces. Pin it so the CLI and wire cannot
        // silently diverge again (the bug was CLI 0.8 vs wire 0.7).
        assert_eq!(default_temperature(), 0.7);
    }

    // ---- F1: shared env resolvers (read in exactly ONE place) ----

    #[test]
    fn frequency_penalty_resolved_env_precedence() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let saved = std::env::var("LUMEN_FREQUENCY_PENALTY").ok();

        // Absent env → the default (0.0, no-op).
        std::env::remove_var("LUMEN_FREQUENCY_PENALTY");
        assert_eq!(frequency_penalty_resolved(), frequency_penalty_default());
        assert_eq!(frequency_penalty_resolved(), 0.0);

        // A finite, >= 0.0 env value wins over the default.
        std::env::set_var("LUMEN_FREQUENCY_PENALTY", "0.4");
        assert_eq!(frequency_penalty_resolved(), 0.4);
        std::env::set_var("LUMEN_FREQUENCY_PENALTY", "0");
        assert_eq!(frequency_penalty_resolved(), 0.0);

        // Invalid / out-of-range values are rejected and fall through to the
        // default (the `is_finite() && >= 0.0` filter): negative, NaN, garbage.
        for bogus in ["-1.0", "NaN", "inf", "not-a-number", ""] {
            std::env::set_var("LUMEN_FREQUENCY_PENALTY", bogus);
            assert_eq!(
                frequency_penalty_resolved(),
                frequency_penalty_default(),
                "bogus env '{bogus}' must fall through to the default"
            );
        }

        match saved {
            Some(v) => std::env::set_var("LUMEN_FREQUENCY_PENALTY", v),
            None => std::env::remove_var("LUMEN_FREQUENCY_PENALTY"),
        }
    }

    #[test]
    fn repeat_last_n_resolved_env_precedence() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let saved = std::env::var("LUMEN_REPEAT_LAST_N").ok();

        // Absent env → None (full-history window, production-identical).
        std::env::remove_var("LUMEN_REPEAT_LAST_N");
        assert_eq!(repeat_last_n_resolved(), None);

        // A parseable usize env value wins.
        std::env::set_var("LUMEN_REPEAT_LAST_N", "64");
        assert_eq!(repeat_last_n_resolved(), Some(64));
        std::env::set_var("LUMEN_REPEAT_LAST_N", "0");
        assert_eq!(repeat_last_n_resolved(), Some(0));

        // Unparseable values (negative, float, garbage) fall through to None.
        for bogus in ["-1", "12.5", "garbage", ""] {
            std::env::set_var("LUMEN_REPEAT_LAST_N", bogus);
            assert_eq!(
                repeat_last_n_resolved(),
                None,
                "bogus env '{bogus}' must fall through to None"
            );
        }

        match saved {
            Some(v) => std::env::set_var("LUMEN_REPEAT_LAST_N", v),
            None => std::env::remove_var("LUMEN_REPEAT_LAST_N"),
        }
    }

    // ---- F1 + F2: allowlist membership (no false unknown-env typo warning) ----

    #[test]
    fn newly_documented_env_vars_are_in_allowlist_and_do_not_warn() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());

        // F1: LUMEN_FREQUENCY_PENALTY (honoured by both wire + CLI) and
        // F2: LUMEN_CUDA_MOE_BF16_NATIVE (documented bf16-MoE restore-path flag)
        // must be in the canonical allowlist so they do not false-fire the
        // unknown-LUMEN-var validator when an operator sets them.
        for name in ["LUMEN_FREQUENCY_PENALTY", "LUMEN_CUDA_MOE_BF16_NATIVE"] {
            assert!(
                KNOWN_LUMEN_ENV_VARS.contains(&name),
                "{name} must be in KNOWN_LUMEN_ENV_VARS"
            );
            let saved = std::env::var(name).ok();
            std::env::set_var(name, "1");
            let warnings = collect_unknown_lumen_env_vars();
            match saved {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
            assert!(
                !warnings.iter().any(|w| w.contains(name)),
                "known env '{name}' must not warn; warnings = {warnings:?}"
            );
        }
    }

    #[test]
    fn allowlist_members_do_not_warn_when_set() {
        // Completeness check that iterates the allowlist rather than hard-coding
        // a single var: EVERY canonical name, when present in the env, must be
        // recognised by the validator (pass-1 membership) so it never emits a
        // false unknown-env typo warning. This guards against a future entry
        // being added with a subtle mismatch (trailing whitespace, wrong case,
        // a stray character) that would slip the prefix-membership check.
        //
        // Unlike a self-membership assertion (which is a tautology — every
        // element trivially equals itself), this drives the REAL validator:
        // set each var, call `collect_unknown_lumen_env_vars()`, and assert no
        // emitted warning names it.
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        for name in KNOWN_LUMEN_ENV_VARS {
            let saved = std::env::var(name).ok();
            std::env::set_var(name, "1");
            let warnings = collect_unknown_lumen_env_vars();
            // Restore BEFORE asserting so a failure cannot leak this var into
            // the process env for sibling tests.
            match saved {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
            assert!(
                !warnings.iter().any(|w| w.contains(name)),
                "allowlist member '{name}' must not warn; warnings = {warnings:?}"
            );
        }
    }
}
