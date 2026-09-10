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
//!   LBC-resolved dense tensor scheme and flips the `bf16_gemmex_default()`
//!   default conditional on "BF16 model": returns `true` for BF16, `false`
//!   for Q8/Q4 dense.
//!
//! # Ordering contract
//!
//! Setters MUST be called BEFORE the first read of any defaulted env:
//!
//! 1. Caller (binary `main`) opens the LBC, learns `provider.output_proj_quant`,
//!    invokes `set_path_is_server(args.backend.is_server)`, then
//!    `set_model_dense_quant(provider.output_proj_quant)`.
//! 2. `CudaBackend::new` and the first decode call subsequently invoke
//!    `bf16_gemmex_default()` and `cuda_decode_delay_us_default()` exactly
//!    once. Each is `OnceLock`-cached on
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
/// "default ON" BF16-gemmex behaviour), `1` =
/// BF16, `2` = quantised (Q8/Q4/etc.). Encoded as `AtomicU8` so the read
/// path is one relaxed load.
static MODEL_DENSE_QUANT_HINT: AtomicU8 = AtomicU8::new(0);
/// The active CUDA device's compute-capability major, stored by the CUDA
/// backend at init before any lever default is resolved; 0 = unknown / not
/// CUDA. Some defaults were tuned on one architecture and are wrong on
/// another, so they read this.
///
/// Process-global, so it describes one device. A process that initialises
/// backends on two devices of different capability concurrently can have the
/// second store land between the first backend's store and its default
/// resolution, and that backend then reads the other device's capability.
/// Sequential initialisation, which is what the binaries do, is sound.
static DEVICE_CC_MAJOR: AtomicU8 = AtomicU8::new(0);

pub fn set_device_cc_major(major: u8) {
    DEVICE_CC_MAJOR.store(major, Ordering::Relaxed);
}

pub fn device_cc_major() -> u8 {
    DEVICE_CC_MAJOR.load(Ordering::Relaxed)
}

const HINT_UNSET: u8 = 0;
const HINT_BF16: u8 = 1;
const HINT_QUANTISED: u8 = 2;

/// Stores the EXACT **primary / bulk** model `QuantScheme` (the body
/// attention+FFN weight scheme, `lbc.header.quantization.scheme`), via
/// `QuantScheme::to_u8` (range 0..=12), so resolvers that must distinguish
/// *within* the "quantised" bucket (Q4_0 versus Q8_0) can do so.
///
/// **Why the PRIMARY scheme, not `output_proj_quant`.** The coarse
/// `MODEL_DENSE_QUANT_HINT` is fed from `output_proj_quant` (the lm_head),
/// which only needs the BF16-vs-quantised split for the GemmEx
/// resolvers. But GGUF keeps the lm_head at higher precision than the body:
/// the "27B-Q4_0" LBC has `output_proj_quant == Q8_0` — IDENTICAL to the
/// "27B-Q8_0" LBC's lm_head. So `output_proj_quant` CANNOT separate q4 from q8
/// (verified at runtime, `dump_quant_hint`: q4 primary=Q4_0/outproj=Q8_0;
/// q8 primary=Q8_0/outproj=Q8_0). The body scheme `header.quantization.scheme`
/// is the correct discriminator and is what this atomic carries. It is set by
/// the separate `set_model_primary_quant` (fed the bulk scheme) and read by
/// `model_dense_quant()`; it NEVER feeds the coarse-hint resolvers, so they
/// stay byte-identical. `255` = unset sentinel (no LBC opened, or a legacy
/// caller that never invoked the setter) — can never collide with a real
/// `to_u8` tag (max 12).
static MODEL_PRIMARY_QUANT_SCHEME: AtomicU8 = AtomicU8::new(QUANT_SCHEME_UNSET);

const QUANT_SCHEME_UNSET: u8 = 255;

/// Tracks whether the loaded LBC declares MoE experts (i.e. Qwen3.5-MoE-35B-A3B
/// class). `false` = dense; `true` = experts > 0 reported by the LBC
/// hyperparams. Finding: the Q8 "split sibling" weight clone path
/// (`LUMEN_CUDA_Q8_SPLIT=1`) is byte-identical to the canonical Q8 dense
/// decode kernel BUT causes catastrophic PAD-token spam on Q8 MoE 35B-A3B
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
/// when the LBC opens. Used to flip the per-call default of
/// `LUMEN_CUDA_BF16_GEMMEX`.
///
/// * `Bf16` → BF16-gemmex default ON.
/// * `Q8_0` / `Q4_0` / other quantised schemes → BF16-gemmex default OFF.
/// * Unset (this setter never called) → preserves legacy behaviour
///   (BF16-gemmex default ON).
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
        | QuantScheme::Q3_K
        | QuantScheme::CtInt4G32 => HINT_QUANTISED,
        // F32/F16 → leave as legacy (HINT_UNSET == 0 means
        // "fall through to legacy default ON" in the resolvers).
        QuantScheme::F32 | QuantScheme::F16 => HINT_UNSET,
    };
    MODEL_DENSE_QUANT_HINT.store(hint, Ordering::Relaxed);
}

/// Records the EXACT **primary / bulk** model quant scheme
/// (`lbc.header.quantization.scheme` — the body attention+FFN weight scheme),
/// which is what the per-quant resolvers read to distinguish Q4_0 from Q8_0.
/// This is a SEPARATE signal from
/// `set_model_dense_quant` (fed `output_proj_quant`): the lm_head is kept at
/// higher precision than the body in GGUF, so `output_proj_quant` is Q8_0 for
/// BOTH 27B-q4 and 27B-q8 and cannot tell them apart — the body scheme can.
///
/// Idempotent. Called from `lumen-server::run` and `lumen-cli::run`
/// immediately after `SyncWeightProvider::open` returns, alongside
/// `set_model_dense_quant` / `set_model_block_count`.
pub fn set_model_primary_quant(scheme: QuantScheme) {
    MODEL_PRIMARY_QUANT_SCHEME.store(scheme.to_u8(), Ordering::Relaxed);
}

/// Reports the EXACT primary/bulk model quant scheme recorded by
/// `set_model_primary_quant`, or `None` if the setter was never called (legacy
/// caller / no LBC opened). Used by the per-quant resolvers to distinguish
/// Q4_0 from Q8_0. One relaxed atomic
/// load + a `from_u8` decode.
pub(crate) fn model_dense_quant() -> Option<QuantScheme> {
    let tag = MODEL_PRIMARY_QUANT_SCHEME.load(Ordering::Relaxed);
    if tag == QUANT_SCHEME_UNSET {
        None
    } else {
        QuantScheme::from_u8(tag).ok()
    }
}

/// Public diagnostic wrapper over `model_dense_quant` for the
/// `dump_quant_hint` example (which lives outside the crate and so cannot see
/// the `pub(crate)` accessor). Behaviourally identical; not used on any hot
/// path.
pub fn model_dense_quant_pub() -> Option<QuantScheme> {
    model_dense_quant()
}

/// Records the loaded model's transformer block count (9B = 32 layers,
/// 27B = 64). Called from the CLI / server alongside `set_model_dense_quant`.
/// 9B and 27B are otherwise indistinguishable to the resolvers (both dense +
/// same quant hints); no shipped default depends on the count today, so it
/// is recorded for a class that later evidence splits by size. 0 = never set.
pub fn set_model_block_count(num_layers: u32) {
    MODEL_BLOCK_COUNT.store(num_layers, Ordering::Relaxed);
}

static MODEL_BLOCK_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub(crate) fn model_block_count() -> u32 {
    MODEL_BLOCK_COUNT.load(Ordering::Relaxed)
}

/// Records whether the loaded LBC declares MoE experts. Called from the
/// CLI / server `main()` immediately after `SyncWeightProvider::open` and
/// alongside `set_model_dense_quant`. The signal feeds the Q8-only flag
/// resolvers (`q8_split_default`, `output_proj_split_default`,
/// `q8_scale_hw_default`, `output_proj_nr_default`,
/// `ffn_fused_glu_skip_default`) so they correctly stay OFF for MoE
/// 35B-A3B
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
/// kernels (single-block materialise-all, a since-removed CUDA-graph
/// single-block variant, FA2 split-K online softmax) ALL produced the
/// identical loop, while running the
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
///   the [`ReasoningExtractor`](crate::tooling::ReasoningExtractor) then routes to `reasoning_content`.
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
/// the env var is not set. Server path returns `50` µs — an empirical
/// mitigation for observed cross-request decode non-determinism, not a
/// root-caused fix. CLI returns `0` (no slowdown; the observed
/// non-determinism was server-concurrency-specific).
pub fn cuda_decode_delay_us_default() -> u64 {
    if PATH_IS_SERVER.load(Ordering::Relaxed) {
        50
    } else {
        0
    }
}

/// Resolves the per-process default for `LUMEN_METAL_DECODE_DELAY_US` when the
/// env var is not set. Returns `0` for both server and CLI: the three
/// DET-001 intra-kernel cross-threadgroup races in the decode path are fixed
/// at the kernel level (see `tests/metal_greedy_determinism_test.rs` for the
/// enumeration and the repeated-run gate, and
/// `scripts/metal_determinism_regression.sh`), so no inter-token delay is
/// needed for them. A CPU sleep only perturbs the GPU scheduler-timing
/// distribution and cannot make a within-token FP reduction deterministic;
/// the env var remains available for diagnostics.
pub fn metal_decode_delay_us_default() -> u64 {
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

/// Public view of the master-rollback default (`LUMEN_CUDA_LEGACY_DEFAULTS`)
/// for optimization gates resolved outside this module.
pub fn canonical_default_on_pub() -> bool {
    canonical_default_on()
}

/// Per-process default for `LUMEN_CUDA_MOE_BATCHED` when the env is unset.
/// ON by default — fires only for MoE models, no effect on dense.
pub fn moe_batched_default() -> bool {
    canonical_default_on()
}

/// Per-process default for `LUMEN_CUDA_MOE_ROUTER_PARALLEL` when unset.
/// ON by default — fires only for MoE, dispatches the two-launch parallel
/// router instead of the sequential single-CTA router.
pub fn moe_router_parallel_default() -> bool {
    canonical_default_on()
}

/// Per-process default for `LUMEN_CUDA_GDN_REGISTER_RESIDENT` when unset.
/// ON by default — fires only for GDN models (Qwen3.5 family).
/// Finding: the two-launch phase 4 update is byte-identical to the reference
/// path.
pub fn gdn_register_resident_default() -> bool {
    canonical_default_on()
}

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
/// (`attention_decode`), a since-removed CUDA-graph single-block variant, and
/// FA2 split-K online softmax — ALL produced the identical loop, ruling
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
    // generations into a repetition attractor on dense bf16; F64 heals it.
    model_is_moe() || MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) == HINT_BF16
}

/// Per-process default for `LUMEN_CUDA_GDN_AB_F16` — route the GDN
/// `ssm_alpha` / `ssm_beta` projections through a pre-dequanted **F16**
/// cache and cuBLAS `cublasGemmEx` (HGEMV in decode, HGEMM in prefill) in
/// BOTH paths, MoE-gated.
///
/// The GDN `ssm_alpha` / `ssm_beta` weights are stored `Q8Raw` in default
/// conversions (the GGUF source is typically F32; the converter
/// force-requantizes them to Q8_0 — source-fidelity, HF-import, and
/// `--dequantize` non-Metal artifacts carry F32 gates and take the F32
/// route instead). With the keeper Q8-prefill-MMQ default ON, the batched PREFILL
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
pub fn gdn_ab_f16_default() -> bool {
    true
}

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
/// On for every model class: the per-step GDN decode recurrence drifts over
/// long generations into a repetition attractor, on MoE and dense models
/// alike, and running each decode step's recurrence through the prefill
/// kernel does not, at a flat decode rate. The dense BF16 classes were
/// measured last, alongside an F16 tensor-core prefill attention that has
/// since been removed; prefill attention is exact F32 throughout now.
/// Set `LUMEN_CUDA_GDN_DECODE_VIA_PREFILL=0|1` to override either way.
pub fn gdn_decode_via_prefill_default() -> bool {
    // Every term is on, so this returns true on every path; the factored form
    // names the classes that were measured separately. CUDA-only: the sole consumer is
    // `gdn_decode_via_prefill_enabled()` in `cuda/backend_impl.rs`; Metal runs
    // its own GDN decode path and never reads this resolver or the variable.
    let dense_bf16 = MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) == HINT_BF16;
    let small = {
        let l = model_block_count();
        l > 0 && l <= 32
    };
    model_is_moe() || !dense_bf16 || small || (dense_bf16 && !small)
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
pub fn gdn_convstate_parity_default() -> bool {
    true
}

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
pub fn bf16_moe_v3_default() -> bool {
    canonical_default_on()
}

/// Per-process default for `LUMEN_CUDA_MOE_Q4_V3` when unset. ON by
/// default — fires only for Q4 MoE expert dispatch.
pub fn moe_q4_v3_default() -> bool {
    canonical_default_on()
}

/// Per-process default for `LUMEN_CUDA_MOE_Q4_V3B` when unset. ON by
/// default — fires only for Q4 MoE; gated by V3 also being ON.
pub fn moe_q4_v3b_default() -> bool {
    canonical_default_on()
}

/// Per-process default for `LUMEN_CUDA_MMV_Q_OUTPUT_PROJ` when unset. ON
/// by default — affects dense Q8/Q4 output projection. finding:
/// the fused matvec saves ~25% on the vocab projection.
pub fn mmv_q_output_proj_default() -> bool {
    canonical_default_on()
}

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
/// A100 for the FFN set, enables `matvec_q8_split_q8_1`) and for BF16
/// dense (clones the converter's Q8-floored GDN `ssm_out` tensors —
/// 48 jobs / ~1.61 GB on 27B — see the BF16 arm below). No-op when the
/// model has no Q8_0 weights.
///
/// **scope fix**: explicitly OFF for MoE (Qwen3.5-MoE-35B-A3B).
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
        // Q8 dense: full clone set. MoE: explicit OFF (measured root-cause).
        HINT_QUANTISED if !model_is_moe() => canonical_default_on(),
        // BF16 dense: the converter Q8-floors the GDN `ssm_out` tensors, so
        // the clone pass enumerates exactly those 48 (all other weights are
        // Bf16Raw and skipped — census-verified: 48 jobs, 1.61 GB) and the
        // split mmvq route serves them (+0.462 ms/token engine ABBA on H100,
        // stream byte-identical to the raw mmvq route).
        HINT_BF16 if !model_is_moe() => canonical_default_on(),
        _ => false,
    }
}

/// Canonical gate for the output-projection companion defaults
/// (`OUTPUT_PROJ_SPLIT`, `Q8_SCALE_HW`, `OUTPUT_PROJ_NR`): quantized output
/// head, dense, canonical defaults on. Reads the coarse output-head hint —
/// it does NOT identify the body scheme — and deliberately excludes
/// `q8_split_default`'s BF16 arm above.
fn quantized_output_head_canonical() -> bool {
    MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) == HINT_QUANTISED
        && !model_is_moe()
        && canonical_default_on()
}

/// BF16-dense-body canonical gate for the BF16 decode levers
/// (`bf16_ab_q8bank_enabled`, `bf16_wo_nr1_enabled`). Keyed on the EXACT
/// primary/bulk scheme (`model_dense_quant`), not the coarse output-head
/// hint: these levers reroute BODY projections, and the body scheme is the
/// signal that cannot be confounded by a higher-precision lm_head.
fn bf16_dense_canonical() -> bool {
    matches!(model_dense_quant(), Some(QuantScheme::Bf16))
        && !model_is_moe()
        && canonical_default_on()
}

/// `LUMEN_CUDA_Q4_SPLIT_ATTN` (default ON): extend the Q4 split-clone pass
/// beyond the dense FFN set to the non-residual attention/GDN projections
/// (GDN fused QKV + gate, full-attention Wq/Wk/Wv). The dispatch sites
/// already prefer a split sibling when present and run the codegen-locked
/// kernel — output is byte-identical to the AoS route — so the setting only
/// widens which weights receive siblings. The clone pass itself skips
/// narrow-GDN configs (v_heads == 32) whose F32-activation dispatch cannot
/// consume the siblings. `=0` opts out.
pub fn q4_split_attn_enabled() -> bool {
    !matches!(std::env::var("LUMEN_CUDA_Q4_SPLIT_ATTN"), Ok(v) if v == "0")
}

/// `LUMEN_CUDA_Q8_SPLIT_ATTN` (default ON): extend the Q8 split-clone pass
/// beyond the dense FFN set to the non-residual attention/GDN projections
/// (GDN fused QKV + gate, full-attention Wq/Wk/Wv). Q8 twin of
/// `LUMEN_CUDA_Q4_SPLIT_ATTN`, scoped to wide-GDN models (v_heads != 32) —
/// the dispatch sites already prefer a Q8 split sibling when present; on
/// non-GDN and narrow-GDN models the clone pass leaves attention untouched
/// (see the clone-pass comment). `=0` opts out.
pub fn q8_split_attn_enabled() -> bool {
    !matches!(std::env::var("LUMEN_CUDA_Q8_SPLIT_ATTN"), Ok(v) if v == "0")
}

/// `LUMEN_CUDA_PROFILE_ATTN_LEAF`: with `LUMEN_CUDA_PROFILE=1`, additionally
/// brackets ONE full-attention sub-stage as the `attn_leaf` row in the profile
/// table. Values: `norm_q8`, `qkv`, `prep`, `attn_core`, `gate`, `wo`. One
/// leaf per run keeps the event stream small enough not to distort the span
/// it measures. Unset or unrecognized value: no leaf bracket. Caveats:
/// `norm_q8` and `qkv` are emitted only on the quantized/BF16 projection
/// branches (preq and non-preq) — F32/F16-cache branches and the fused-norm
/// failure fallback emit no row for those two, while `prep`, `attn_core`,
/// `gate`, and `wo` bracket the shared post-projection stages on every
/// branch. The leaf nests inside `full_attn`, so the profile summary's
/// TOTAL double-counts it — read the leaf row against `full_attn`, never
/// the total.
pub fn profile_attn_leaf() -> Option<&'static str> {
    static CACHE: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let v = std::env::var("LUMEN_CUDA_PROFILE_ATTN_LEAF").ok()?;
        ["norm_q8", "qkv", "prep", "attn_core", "gate", "wo"]
            .into_iter()
            .find(|s| *s == v)
    })
}

/// `LUMEN_CUDA_Q8_SPLIT_SSMOUT` (default ON): clone the GDN `ssm_out` Q8
/// weight into its per-row split sibling and dispatch it through the Q8
/// split family instead of the raw-layout route. `=0` opts out.
pub fn q8_split_ssmout_enabled() -> bool {
    !matches!(std::env::var("LUMEN_CUDA_Q8_SPLIT_SSMOUT"), Ok(v) if v == "0")
}

/// `LUMEN_CUDA_Q4_SPLIT_WO=1` (probe): clone the full-attention `wo` into its
/// Q4 split sibling, routing its decode through the residual-split kernel an
/// earlier campaign recorded as broken — exists to re-test that verdict on
/// current source. Default OFF.
pub fn q4_split_wo_probe_enabled() -> bool {
    matches!(std::env::var("LUMEN_CUDA_Q4_SPLIT_WO"), Ok(v) if v == "1")
}

/// `LUMEN_CUDA_ATTN_SPLITK` (model-aware default: ON for Q8_0- and
/// BF16-body dense models, OFF otherwise): route the full-attention decode
/// step through the split-K kernel pair (sequence-parallel: one CTA per
/// query head per chunk, plus a merge) instead of the one-CTA-per-head
/// tiled kernel. Lifts the occupancy ceiling on few-head models (27B: 24
/// CTAs -> 24 per chunk, plus a 24-CTA merge) and cuts each partial CTA's
/// sequence walk to its own chunk (total work stays linear in context
/// length). The chunk count follows the context length, one chunk per
/// [`attn_splitk_chunk_positions`] positions; where that count is 1 the
/// tiled kernel runs instead. Quality-equivalent near-tie — the cross-chunk
/// merge sums in a different order than the tiled kernel's progressive
/// rescale, and that order follows the chunk count. `=0` opts out, `=1`
/// forces on; unset resolves model-aware: ON for Q8_0-body and BF16-body
/// dense models (the classes the engine A/Bs + full GQ/DET gates banked, at
/// a fixed 4 chunks: Q8 +0.195 ms/tok on A100-SXM, BF16 +0.298 ms/tok on
/// H100), following the canonical-defaults master switch. Q4_0-body dense
/// models take the pair on compute capability 12.x as well: on the RTX 5090
/// the tiled route is the kernel that collapses at context (one CTA per
/// head, 24 CTAs on a 170-SM card) and the pair, context-scaled, was gated
/// there at +34.39 % at 1,300 tokens and +67 % at 2,600 with the 48-token
/// greedy output byte-identical to the tiled route's (r2-016; the 1,024-in /
/// 128-out board: 60.7 -> 79.4 tok/s). On every other capability Q4 bodies
/// stay on the tiled route: an earlier Q4 quality gate on the A100 failed
/// with the (then fixed 4-chunk) pair on, and nothing has been measured there
/// since. `=0` opts a Blackwell Q4 run back out.
pub fn attn_splitk_enabled() -> bool {
    match std::env::var("LUMEN_CUDA_ATTN_SPLITK") {
        Ok(v) if v == "0" => false,
        Ok(v) if v == "1" => true,
        _ => attn_splitk_default(),
    }
}

/// The model-aware default behind [`attn_splitk_enabled`]: Q8_0 and BF16
/// dense bodies everywhere, Q4_0 dense bodies on compute capability 12.x,
/// never MoE, all under the canonical-defaults master switch.
pub fn attn_splitk_default() -> bool {
    attn_splitk_default_for(
        model_dense_quant(),
        model_is_moe(),
        device_cc_major(),
        canonical_default_on(),
    )
}

/// [`attn_splitk_default`] with every input explicit (the process wrappers feed the globals;
/// tests feed values).
pub fn attn_splitk_default_for(
    quant: Option<QuantScheme>,
    moe: bool,
    cc_major: u8,
    canonical: bool,
) -> bool {
    if moe || !canonical {
        return false;
    }
    match quant {
        Some(QuantScheme::Q8_0) | Some(QuantScheme::Bf16) => true,
        Some(QuantScheme::Q4_0) => cc_major == 12,
        _ => false,
    }
}

/// Per-process default for `LUMEN_CUDA_NORM_CTA5_DUAL` when unset: ON for a
/// Q4_0 dense body on compute capability 12.x — the one cell it is measured
/// on (source-fidelity Qwen3.8-27B Q4_0, RTX 5090: +4.10 % decode, byte-
/// identical, r3-022/023/024; the device twin tests pin both kernels bitwise
/// to the single-block original at dims 2048/4096/5120/5152). OFF on every other
/// capability and body class because the launch shape is unmeasured there,
/// not because it is known to be slow. Follows the canonical-defaults
/// master switch; `=1` still forces it on anywhere.
pub fn norm_cta5_dual_default() -> bool {
    norm_cta5_dual_default_for(
        model_dense_quant(),
        model_is_moe(),
        device_cc_major(),
        canonical_default_on(),
    )
}

/// [`norm_cta5_dual_default`] with every input explicit.
pub fn norm_cta5_dual_default_for(
    quant: Option<QuantScheme>,
    moe: bool,
    cc_major: u8,
    canonical: bool,
) -> bool {
    !moe && matches!(quant, Some(QuantScheme::Q4_0)) && cc_major == 12 && canonical
}

/// The NVRTC target the tiled decode-attention kernel is compiled for when
/// `LUMEN_CUDA_ATTN_TILED_CODEGEN` is unset: `ptx120` (compute_120) on a
/// compute capability 12.x device whose NVRTC lists that target, else NVRTC's
/// default. Measured on the RTX 5090 with CUDA 13.3: the same source emits
/// 2,520 instructions at compute_120 against 3,632 at the default sm_75
/// target, +9.6 % decode at 1,300 tokens and +14.7 % at 2,600 on the tiled
/// route, byte-identical (r3-025/028); the compute_80 control was null, so
/// the gain is the target, not the recompile. Only this kernel: the GDN
/// kernels grow at compute_120, so the policy is per kernel, not per process.
/// And only the measured cell — a Q4_0 dense body — like the other two
/// promoted defaults: an MoE or Q8/BF16 model on the same card keeps NVRTC's
/// default target until it is gated there. Follows the canonical-defaults
/// master switch. Resolved at kernel compilation, after the CLI/server have
/// recorded the model's body class and before the backend records the
/// capability (which is why the capability is a parameter here).
pub fn attn_tiled_codegen_default(cc_major: u8, nvrtc_can_target_120: bool) -> &'static str {
    attn_tiled_codegen_default_for(
        model_dense_quant(),
        model_is_moe(),
        cc_major,
        nvrtc_can_target_120,
        canonical_default_on(),
    )
}

/// [`attn_tiled_codegen_default`] with every input explicit.
pub fn attn_tiled_codegen_default_for(
    quant: Option<QuantScheme>,
    moe: bool,
    cc_major: u8,
    nvrtc_can_target_120: bool,
    canonical: bool,
) -> &'static str {
    if !moe
        && matches!(quant, Some(QuantScheme::Q4_0))
        && cc_major == 12
        && nvrtc_can_target_120
        && canonical
    {
        "ptx120"
    } else {
        "default"
    }
}

/// `LUMEN_CUDA_FORCE_SCALAR_ATTN=1`: run the fused Q+gate prefill attention
/// on the one-warp-per-row scalar kernel, whatever the tiled route says: the
/// score-block sizing reads it and leaves the block unallocated. Read once.
pub fn force_scalar_attn_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("LUMEN_CUDA_FORCE_SCALAR_ATTN").as_deref() == Ok("1"))
}

/// `LUMEN_CUDA_ATTN_PREFILL_SGEMM=0`: kill-switch for the tiled prefill
/// attention — cuBLAS F32 SGEMM for Q·Kᵀ and P·V (strided-batched over each
/// KV head's query group) around an exact-F32 causal softmax, in query blocks
/// of at most 512 rows.
///
/// This is not a precision policy: it selects how the exact-F32 attention
/// is computed for prefills of 16 tokens or more. Shorter prefills and a
/// full-attention layer without per-head q/k norms (none that today's
/// converter produces) keep the scalar kernel whatever this returns; the
/// decode path and the non-CUDA backends are unaffected: the SGEMM route
/// serves the fused Q+gate prefill path only.
///
/// Exact F32 throughout like the one-warp-per-row scalar kernel it replaces —
/// no F16 operand anywhere — but a different evaluation order: the softmax row is normalised
/// before P·V and reduced over the whole row rather than online. Greedy output
/// can therefore differ from the scalar kernel where two candidates sit within
/// rounding of each other.
///
/// Default ON; unset follows `canonical_default_on`, so
/// `LUMEN_CUDA_LEGACY_DEFAULTS=1` rolls this back with every other CUDA
/// default. `=0` restores the scalar kernel for an exact A/B of the two
/// evaluation orders at a fixed precision mode.
pub fn attn_prefill_sgemm_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_ATTN_PREFILL_SGEMM") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// Target KV positions per split-K decode-attention chunk unless
/// `LUMEN_CUDA_ATTN_SPLITK_CHUNK` says otherwise: one of the kernel's
/// 128-position tiles. The value picks the chunk *count* — the context
/// length divided by it, rounded up, capped at the scratch bound — so the
/// span a chunk actually walks is the context divided by that count: 65 and
/// 64 at a context of 129, 119 at 1300, 384 at 12280, where the cap binds.
/// Measured on the RTX 5090 (Qwen3.8-27B, 330 to 2.6k tokens of context):
/// 128 beat 256 at every length.
pub const ATTN_SPLITK_CHUNK_POSITIONS: u32 = 128;

/// The split count the split-K pair used at every context through v0.24.0,
/// and what [`attn_splitk_scale_with_context`] returns to when opted out.
pub const ATTN_SPLITK_FIXED_CHUNKS: u32 = 4;

/// `LUMEN_CUDA_ATTN_SPLITK_GQA6=1`: serve the eligible full-attention decode
/// step with the GQA-shared split-K pair
/// (`attention_decode_splitk_partial_gqa6_f32` +
/// `attention_decode_splitk_merge_gqa6_f32`) instead of the per-query-head
/// pair. One CTA per (KV head, chunk) fetches each K and V row once for the
/// whole six-query-head group rather than once per query head, and every Q,
/// K and V read is a 16-byte load; the merge computes each chunk's rescale
/// once instead of once per output dimension.
///
/// OFF unless set. Eligibility is narrow — six query heads per KV head
/// (24/4, 12/2 and 6/1 alike), head_dim 256, and a context the chunk cap
/// covers — and every other shape
/// keeps its existing route, so the flag is a no-op elsewhere. The chunk
/// partition and both reduction orders differ from the per-query-head pair,
/// which makes the two a near-tie rather than byte-identical.
///
/// Read once: the loader consults it while building the kernel set and the
/// scratch allocator consults it at init.
pub fn attn_splitk_gqa6_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("LUMEN_CUDA_ATTN_SPLITK_GQA6").as_deref() == Ok("1"))
}

/// `LUMEN_CUDA_ATTN_SPLITK_SCALE` (default ON, canonical): size the split-K
/// decode-attention split count from the context. `=0` pins the fixed count
/// [`ATTN_SPLITK_FIXED_CHUNKS`] at every context, the configuration the
/// classes that take the pair by default (Q8_0- and BF16-body dense) were
/// gate-banked under; the scaled count merges a different number of chunks,
/// a near-tie numerics change on those classes. Follows the
/// canonical-defaults master switch.
pub fn attn_splitk_scale_with_context() -> bool {
    match std::env::var("LUMEN_CUDA_ATTN_SPLITK_SCALE") {
        Ok(v) if v == "0" => false,
        Ok(v) if v == "1" => true,
        _ => canonical_default_on(),
    }
}

/// `LUMEN_CUDA_ATTN_SPLITK_CHUNK`: target KV positions per split-K attention
/// chunk (default [`ATTN_SPLITK_CHUNK_POSITIONS`]). The chunk count is the
/// context length divided by this, rounded up and capped at the scratch
/// bound; a count of 1 means the tiled kernel runs instead. A value below 128
/// is raised to 128 (one kernel tile) and an unparseable value is the
/// default; either substitution is printed once, so a chunk size the operator
/// wrote and the runtime did not use never passes unnoticed.
pub fn attn_splitk_chunk_positions() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let (chunk, warning) = parse_attn_splitk_chunk(
            std::env::var("LUMEN_CUDA_ATTN_SPLITK_CHUNK")
                .ok()
                .as_deref(),
        );
        if let Some(warning) = warning {
            eprintln!("{warning}");
        }
        chunk
    })
}

/// Pure parser behind [`attn_splitk_chunk_positions`] (separated for unit
/// testing): the resolved chunk size and, when the operator's value was not
/// used verbatim, the one line the caller prints. A perf knob the runtime
/// rewrites silently reads as honoured and is not; a typo must still not
/// abort engine init.
fn parse_attn_splitk_chunk(raw: Option<&str>) -> (u32, Option<String>) {
    const ENV: &str = "LUMEN_CUDA_ATTN_SPLITK_CHUNK";
    let Some(raw) = raw else {
        return (ATTN_SPLITK_CHUNK_POSITIONS, None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return (
            ATTN_SPLITK_CHUNK_POSITIONS,
            Some(format!(
                "[CUDA] {ENV}='{raw}' is empty; using the default \
                 {ATTN_SPLITK_CHUNK_POSITIONS} KV positions per chunk"
            )),
        );
    }
    match trimmed.parse::<u32>() {
        Ok(v) if v >= ATTN_SPLITK_CHUNK_POSITIONS => (v, None),
        Ok(v) => (
            ATTN_SPLITK_CHUNK_POSITIONS,
            Some(format!(
                "[CUDA] {ENV}='{raw}' is below one kernel tile ({v} < \
                 {ATTN_SPLITK_CHUNK_POSITIONS}); raised to \
                 {ATTN_SPLITK_CHUNK_POSITIONS}"
            )),
        ),
        Err(e) => (
            ATTN_SPLITK_CHUNK_POSITIONS,
            Some(format!(
                "[CUDA] {ENV}='{raw}' is not a positive integer ({e}); using \
                 the default {ATTN_SPLITK_CHUNK_POSITIONS} KV positions per chunk"
            )),
        ),
    }
}

/// `LUMEN_CUDA_BF16_NR1` (default ON): route the broad BF16 decode matvecs
/// through the one-row/CTA `matvec_bf16_v4_nr1` kernel instead of the NR=2
/// `matvec_bf16_v4` (+0.303 ms/token engine ABBA on H100; leaf 18.369 vs
/// 19.085 weighted ms/token). BYTE-IDENTICAL to the NR=2 route — the per-row
/// F32 accumulation sequence is unchanged, only the CTA that computes it —
/// verified by 256-token greedy md5 equality on the live 27B artifact.
/// `=0` opts out; unset follows the canonical-defaults master switch.
pub fn bf16_nr1_enabled() -> bool {
    match std::env::var("LUMEN_CUDA_BF16_NR1") {
        Ok(v) if v == "0" => false,
        Ok(v) if v == "1" => true,
        _ => canonical_default_on(),
    }
}

/// `LUMEN_CUDA_BF16_FUSED_GLU` (default ON): on the GPU-resident decode
/// path, serve the BF16 dense FFN gate+up+SwiGLU with ONE fused kernel
/// (both dots off the shared F32 normed activation) instead of the separate
/// gate matvec + up matvec + swiglu_inplace sub-sequence (+0.510 ms/token
/// engine ABBA on H100; the non-resident streaming fallback keeps the
/// separate sequence).
/// BYTE-IDENTICAL to that separate custom-matvec route — verified by
/// 256-token greedy md5 equality on the live 27B artifact. Dispatch also
/// requires `LUMEN_CUDA_BF16_MATVEC` on (the identity baseline; `=0` there
/// restores the pre-existing BF16 fallback dispatch for the layer). `=0`
/// opts out; unset follows the canonical-defaults master switch.
pub fn bf16_fused_glu_enabled() -> bool {
    match std::env::var("LUMEN_CUDA_BF16_FUSED_GLU") {
        Ok(v) if v == "0" => false,
        Ok(v) if v == "1" => true,
        _ => canonical_default_on(),
    }
}

/// `LUMEN_CUDA_BF16_AB_Q8BANK` (default ON for BF16-dense): on the BF16 GDN
/// route, quantize the normed activation ONCE and serve the Q8-forced ssm
/// alpha+beta projections with the existing banked raw-route kernel —
/// replacing the two separate quantize+matvec pairs the generic dispatch
/// previously ran (the Q8 route's bank is unreachable here because BF16
/// qkv/gate disqualify its prequant predicate). +0.308 ms/token engine ABBA
/// on H100. Near-tie class (banked kernel; DET+GQ gate-banked). `=0` opts
/// out, `=1` forces on; unset resolves ON for BF16-body dense non-MoE ONLY —
/// the default converter path Q8-forces alpha/beta across quants
/// (source-fidelity artifacts may preserve F32 gates, which fail the Q8Raw
/// predicate and stay on the fallback), so without the body scope this arm
/// would also intercept the Q4/Q8 models' generic route.
pub fn bf16_ab_q8bank_enabled() -> bool {
    match std::env::var("LUMEN_CUDA_BF16_AB_Q8BANK") {
        Ok(v) if v == "0" => false,
        Ok(v) if v == "1" => true,
        _ => bf16_dense_canonical(),
    }
}

/// `LUMEN_CUDA_BF16_WO_NR1` (default ON for BF16-dense): serve the BF16
/// full-attention `wo` decode projection with the one-row residual matvec
/// (F32 activation read directly, BF16->F32 lossless weight upcast, F32
/// accumulate + residual add in one launch) instead of the cuBLAS chain
/// (residual dtod copy + F32->BF16 conversion + GemmEx beta=1). +0.186
/// ms/token engine ABBA on H100; keeps the activation in F32 (the GemmEx
/// route downcasts it to BF16), and the differing reduction order means
/// output is not guaranteed byte-identical (near-tie class, DET+GQ
/// gate-banked). `=0` opts out; unset resolves ON for BF16-body dense
/// non-MoE, following the canonical-defaults master switch. Dispatch also
/// requires `LUMEN_CUDA_BF16_MATVEC` on — `=0` there restores the
/// pre-existing BF16 fallback dispatch for the layer.
pub fn bf16_wo_nr1_enabled() -> bool {
    match std::env::var("LUMEN_CUDA_BF16_WO_NR1") {
        Ok(v) if v == "0" => false,
        Ok(v) if v == "1" => true,
        _ => bf16_dense_canonical(),
    }
}

/// `LUMEN_CUDA_CT4_EXACTK` (default ON; `=0` opts out): launch the CtInt4G32
/// decode matvec with a block size matched to the reduction depth instead of
/// the fixed 256. The K=5120 / K=6144 projection shapes have only 160 / 192
/// g32 blocks per row, so at 256 threads 37.5% / 25% of every CTA's warps
/// hold no work; the exact-K kernels (160 / 192 threads, reduction folding a
/// zero-padded 8-slot array) remove those idle warps with bit-identical
/// output. K=17408 (FFN down) keeps the 256-thread kernel.
fn parse_ct4_exactk(raw: Result<String, std::env::VarError>) -> bool {
    !matches!(raw, Ok(v) if v.trim() == "0")
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn ct4_exactk() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| parse_ct4_exactk(std::env::var("LUMEN_CUDA_CT4_EXACTK")))
}

/// `LUMEN_CUDA_CT4_DP4A` (default ON): serve imported CtInt4G32 weights via
/// the W4A8 dp4a decode kernel. `=0` dequantizes ALL of them to F16 at
/// upload and serves the existing F16 routes instead (W4A16-style reference,
/// ~3.2x the weight bytes). A comma-separated role list (e.g.
/// `=ssm_out,attn_gate`) forces F16 for those roles only; unknown role
/// names are a startup error. Naming any of wq/wk/wv selects all three:
/// the attention QKV path dispatches the trio together, so a partial F16
/// conversion there would mix incompatible routes.
///
/// `Ok(None)` = dp4a for everything (default); `Ok(Some(roles))` = force
/// F16 for the named roles (`"*"` = all); `Err` = malformed value.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn ct4_f16_roles() -> Result<Option<Vec<String>>, String> {
    match std::env::var("LUMEN_CUDA_CT4_DP4A") {
        Ok(raw) => parse_ct4_f16_roles(&raw),
        Err(_) => Ok(None),
    }
}

/// Pure parser behind [`ct4_f16_roles`] (separated for unit testing).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn parse_ct4_f16_roles(raw: &str) -> Result<Option<Vec<String>>, String> {
    const ROLES: [&str; 9] = [
        "wq",
        "wk",
        "wv",
        "wo",
        "w_gate",
        "w_up",
        "w_down",
        "ssm_out",
        "attn_gate",
    ];
    match raw.trim() {
        "" | "1" => Ok(None),
        "0" => Ok(Some(vec!["*".into()])),
        v => {
            let mut roles: Vec<String> = Vec::new();
            for r in v.split(',').map(str::trim).filter(|r| !r.is_empty()) {
                if !ROLES.contains(&r) {
                    return Err(format!(
                        "LUMEN_CUDA_CT4_DP4A: unknown role {r:?} (expected 0, 1, \
                         or a comma-separated subset of {ROLES:?})"
                    ));
                }
                if !roles.iter().any(|x| x == r) {
                    roles.push(r.to_owned());
                }
            }
            if roles.iter().any(|r| r == "wq" || r == "wk" || r == "wv") {
                for qkv in ["wq", "wk", "wv"] {
                    if !roles.iter().any(|x| x == qkv) {
                        roles.push(qkv.to_owned());
                    }
                }
            }
            Ok(if roles.is_empty() { None } else { Some(roles) })
        }
    }
}

/// `LUMEN_CUDA_Q8_SPLIT_WO=1` (probe): clone the full-attention `wo` into its
/// Q8 split sibling. Unlike the Q4 twin above, the Q8 residual-split route it
/// enables (`matvec_q8_split_q8_1_mmvq_residual`) is the same kernel already
/// shipping for the FFN down and folded ssm_out projections. Default OFF.
/// Consumed INSIDE the attention clone pass: it only takes effect on
/// wide-GDN models with `LUMEN_CUDA_Q8_SPLIT_ATTN` on — with the attention
/// clone off (or on non-wide-GDN models) `=1` clones nothing, so an A/B
/// probe there is a silent no-op, not a treatment arm.
pub fn q8_split_wo_probe_enabled() -> bool {
    matches!(std::env::var("LUMEN_CUDA_Q8_SPLIT_WO"), Ok(v) if v == "1")
}

/// `LUMEN_CUDA_GDN_NG_Q8` (default ON; `=0` opts out): the T=1 via-prefill norm-gate also
/// emits the Q8_1 quantization of its own output, eliding the separate
/// quantize launch before the ssm_out split matvec. Verbatim-cloned
/// arithmetic => bit-identical bytes; dense-F32 path only.
pub fn gdn_ng_q8_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_GDN_NG_Q8") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// `LUMEN_CUDA_GDN_P123_FUSE` (default ON; `=0` opts out): fuse the first three T=1
/// via-prefill GDN launches (conv+SiLU, gates, QK-L2) into one kernel.
/// Per-op arithmetic cloned verbatim => bit-identical. The F32 path uses
/// `gdn_decode_phase123_fused`; the F64-recurrence path uses the
/// `_f64norm` twin (phase-3 L2 in F64, bit-identical to its 3-launch chain).
pub fn gdn_p123_fuse_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_GDN_P123_FUSE") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// `LUMEN_CUDA_ROPE_TAB` (default ON; `=0` opts out): NeoX RoPE reads its cos/sin pairs from
/// a per-CTA shared table computed once (identical expression, identical
/// inputs => identical bits) instead of every thread recomputing
/// powf/cosf/sinf.
pub fn rope_tabled_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_ROPE_TAB") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// `LUMEN_CUDA_ATTN_PREP_FUSE` (default ON; `=0` opts out): the six-launch full-attention
/// prep chain (deinterleave, per-head Q/K norms, NeoX RoPE, K/V cache
/// appends) issues as ONE kernel. Per-value op sequences cloned verbatim =>
/// bit-identical; the region is CPU-launch-shadow bound, so the launch count
/// is the lever.
pub fn attn_prep_fuse_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_ATTN_PREP_FUSE") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// `LUMEN_CUDA_ATTN_BANK3` (default ON; `=0` opts out): full-attention wq/wk/wv issue as ONE
/// banked launch (virtual row concat via the 4-way kernel with an empty
/// fourth slot) off their shared Q8_1 input — two launch boundaries removed
/// and the tiny wk/wv grids (256 CTAs each) ride the wq grid's tail.
/// Bit-identical per row.
pub fn attn_bank3_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_ATTN_BANK3") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// `LUMEN_CUDA_Q4_V4LOAD` (default ON, GDN-bank-scoped; `=0` opts out):
/// the banked GDN launch loads the nibble
/// stream as one 128-bit uint4 instead of four u32 words (alignment-guarded
/// at dispatch). Integer loads are exact => bit-identical; gate-verified.
///
pub fn q4_v4load_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_Q4_V4LOAD") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// `LUMEN_CUDA_Q4_B160` (default ON; `=0` opts out): route the GDN banked
/// nb=160 launch through the 160-thread compile variant — every lane productive in
/// the K-loop instead of 160/256, ~+50% load-issuing lanes at full
/// occupancy. The dropped warps only folded exact +0.0 partials, so output
/// is expected bit-identical (byte-gate enforced).
pub fn q4_b160_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_Q4_B160") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// `LUMEN_CUDA_Q8_AB_BANK` (default ON; `=0` opts out): the GDN alpha+beta Q8Raw matvecs
/// (`[48,5120]` each, 24-CTA grids) issue as ONE banked launch. The banked
/// kernel duplicates the raw-route body verbatim but compiles under
/// fast-math, so equality vs the two-launch route is validated by output-equality tests
/// rather than assumed.
pub fn q8_ab_bank_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_Q8_AB_BANK") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// `LUMEN_CUDA_Q4_PROJ_BANK` (default ON; `=0` opts out): bank the GDN qkv + gate Q4 split
/// matvecs into ONE launch of `matvec_q4_split_q8_1_locked_banked` (both read
/// the same pre-quantized Q8_1 input; per-row math untouched, so output is
/// bit-identical to the two-launch route).
pub fn q4_proj_bank_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_Q4_PROJ_BANK") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// `LUMEN_CUDA_SSMOUT_RESID_FOLD` (default ON; `=0` opts out): when the ssm_out split route is active,
/// dispatch its residual variant so the projection writes `attn_proj = W*x +
/// x_gpu` directly and the per-layer residual_add_copy launch is skipped.
/// Byte-identical for normal-range activations; the folded add runs in a
/// flush-to-zero kernel, so a subnormal residual can differ in the last bit
/// from the separately-compiled legacy add.
///
pub fn ssmout_residual_fold_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_SSMOUT_RESID_FOLD") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// `LUMEN_CUDA_Q5K_SSMOUT=0`: kill-switch for the source-fidelity Q5_K
/// ssm_out decode route (falls back to the F16 image via HGEMV). Default ON.
pub fn q5k_ssmout_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_Q5K_SSMOUT") {
        Ok(v) => v != "0",
        Err(_) => true,
    })
}

/// `LUMEN_CUDA_Q4_1_DOWN=0`: kill-switch for the source-fidelity Q4_1
/// w_down decode route (falls back to the F16 image via HGEMV). Default ON.
pub fn q4_1_down_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_Q4_1_DOWN") {
        Ok(v) => v != "0",
        Err(_) => true,
    })
}

/// `LUMEN_CUDA_Q6K_HEAD=0`: kill-switch for the source-fidelity Q6_K output
/// head planes. When OFF the CUDA init skips the plane build and serves the
/// head from the provider's F32 dequant copy (SGEMV; ~5 GB extra VRAM —
/// debug/bisect only). Default ON.
pub fn q6k_head_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_Q6K_HEAD") {
        Ok(v) => v != "0",
        Err(_) => true,
    })
}

/// Default byte budget for the split-sibling clones: free VRAM minus the
/// activation slack. `free` is read in `preload_weights`, after `init` has
/// allocated the KV caches, so it is already net of KV; subtracting a KV
/// reserve here again would count it twice. The formula this replaces also
/// raised the budget to a 5.1 GB minimum, and that minimum was live: on a
/// 32 GB card with 2.76 GB free after the F16 caches it resolved 5.1 GB and,
/// with the other clones that follow, left the card at 0.14 GB free. A
/// minimum above `free - slack` spends the slack, so there is none now, and
/// the same card resolves 0.76 GB.
pub fn split_clone_budget_bytes(free: usize, slack: usize) -> usize {
    free.saturating_sub(slack)
}

/// The split-clone budget the clone passes consume, and whether it came from
/// the operator: an explicit `LUMEN_CUDA_*_SPLIT_BUDGET_GB` value that parses
/// as a finite number above zero is taken as gigabytes, uncapped, exactly as
/// before the free-memory default existed; anything else (unset, empty,
/// zero, negative, not a number) resolves to [`split_clone_budget_bytes`].
/// Pure, so the choice is pinned off-device; the call site supplies the
/// inputs (its free-memory reading, the slack, the raw override).
pub fn resolve_split_clone_budget_bytes(
    free: usize,
    slack: usize,
    override_raw: Option<&str>,
) -> (usize, bool) {
    if let Some(gb) = override_raw
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|gb| gb.is_finite() && *gb > 0.0)
    {
        return ((gb * 1_000_000_000.0) as usize, true);
    }
    (split_clone_budget_bytes(free, slack), false)
}

/// Whether a clone of `clone_bytes` may be made when `free` bytes remain and
/// `slack` must stay free for decode afterwards. The output-projection split
/// clone, made between the Q8 and Q4 sibling passes, goes through this, so it
/// cannot spend the decode slack. Two clones do not: the aligned Q8 and Q4
/// repacks of the output projection on a model without GDN layers, which every
/// model in the shipped registry has, so neither path is reached. On a 32 GB card
/// (Qwen3.8-27B Q4_0, 4096-token context) the sibling pass left 2.06 GB free;
/// with this 1.35 GB clone also made the card reached 32066 of 32607 MiB and
/// the first inference failed with CUDA_ERROR_OUT_OF_MEMORY. Skipping either
/// the sibling clones or this clone let the same run complete (with this one
/// skipped the peak was 30820 MiB): each clone class is necessary for the
/// failure and neither alone is sufficient, so every clone on the reached
/// paths is budgeted.
pub fn clone_fits(clone_bytes: u64, free: u64, slack: u64) -> bool {
    clone_bytes
        .checked_add(slack)
        .is_some_and(|need| need <= free)
}

/// Whether the output-projection split clone is made. A clone that would
/// spend the decode slack is skipped, and so is one decided on a failed
/// memory query: the clone is an optimisation, so skipping it costs nothing.
/// [`f16_cache_refusal`] goes the other way on a failed query and builds,
/// because a telemetry failure is not a memory shortage and a load that got
/// that far is refused only on a measured one. [`F16_CACHE_FORCE_ENV`]
/// overrides both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloneDecision {
    /// Make the clone.
    Proceed,
    /// Skip it: the clone plus the slack exceeds the free bytes.
    NoRoom {
        clone_bytes: u64,
        free: u64,
        slack: u64,
    },
    /// Skip it: the free-memory query failed and the clone is not forced.
    UnknownFree,
}

impl CloneDecision {
    /// Whether the clone goes ahead.
    pub fn proceed(self) -> bool {
        matches!(self, CloneDecision::Proceed)
    }

    /// The reason the clone is skipped, for the start-up log; `None` when it
    /// proceeds.
    pub fn skip_reason(self) -> Option<String> {
        let gb = |b: u64| b as f64 / 1.0e9;
        match self {
            CloneDecision::Proceed => None,
            CloneDecision::NoRoom { clone_bytes, free, slack } => Some(format!(
                "a {:.2} GB clone would leave less than the {:.2} GB decode slack of the {:.2} GB free \
                 (set {F16_CACHE_FORCE_ENV}=1 to make it anyway)",
                gb(clone_bytes),
                gb(slack),
                gb(free)
            )),
            CloneDecision::UnknownFree => Some(format!(
                "the free-memory query failed (set {F16_CACHE_FORCE_ENV}=1 to make it anyway)"
            )),
        }
    }
}

/// The decision for a clone of `clone_bytes` given what the memory query said;
/// see [`CloneDecision`] for the policy.
pub fn output_proj_clone_decision(
    clone_bytes: u64,
    free: FreeMemory,
    slack: u64,
    forced: bool,
) -> CloneDecision {
    if forced {
        return CloneDecision::Proceed;
    }
    match free {
        FreeMemory::Unknown => CloneDecision::UnknownFree,
        FreeMemory::Bytes(free) if clone_fits(clone_bytes, free, slack) => CloneDecision::Proceed,
        FreeMemory::Bytes(free) => CloneDecision::NoRoom {
            clone_bytes,
            free,
            slack,
        },
    }
}

/// Margin on top of the F16 dequant caches' predicted size. The sizing is a
/// lower bound (Qwen3.8-27B Q4_0 predicted 11.91 GB and allocated 11.91 GB
/// on one load and 12.01 GB on another), and this covers that measured
/// spread. It reserves nothing for the allocations that follow the caches,
/// the aligned Q8 repack on a model without GDN layers among them: a Tesla
/// T4 serving Qwen3.5-9B Q8_0 at an 8192-token context builds 3.36 GB of
/// caches into 3.80 GB free and then decodes with the 0.45 GB left, which a
/// 512 MiB reserve refused. Both figures are from loads that cached every
/// quantised projection, which is now [`F16_CACHE_ENV`]'s path; by default
/// the total the margin sits on covers the F32 projections and the quantised
/// K/V or up beside an F32 Q or gate.
pub const F16_CACHE_HEADROOM_BYTES: u64 = 128 * 1024 * 1024;

/// Set to a truthy value to build the F16 dequant caches even when
/// [`f16_cache_refusal`] would refuse. The refusal is a prediction; this is
/// the escape hatch when the prediction is wrong for a card.
pub const F16_CACHE_FORCE_ENV: &str = "LUMEN_CUDA_F16_CACHE_FORCE";

/// Set to a truthy value to build F16 dequant caches for the Q8_0 / Q4_0
/// projections of full-attention layers (Q/K/V/O and the FFN gate/up/down) as
/// well. By default a projection gets one when it is F32, or when it is a
/// K/V beside an F32 Q or an up beside an F32 gate: those are the copies
/// decode's HGEMV reads. Batched prefill dequantises each weight into scratch
/// per matmul so its arithmetic matches decode's, and never reads these
/// caches, so the other quantised copies served only the decode fallback
/// taken for input dimensions above 24576 or for a matvec kernel that failed
/// to load. Set it on a card where that fallback is the path taken, or to
/// restore the previous memory profile.
pub const F16_CACHE_ENV: &str = "LUMEN_CUDA_F16_CACHE";

/// `true` for the values the `LUMEN_CUDA_*` truthy flags accept; unset, and
/// anything outside that set, is off. Pure so the switch below is testable
/// without mutating the process environment.
fn env_truthy(value: Option<&str>) -> bool {
    matches!(
        value,
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

/// Whether [`F16_CACHE_ENV`] asks for the quantised projections' F16 dequant
/// caches. Read once, and announced only when it is on: the default is what
/// every load does, so it is not news.
pub fn f16_cache_for_quantised() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let on = env_truthy(std::env::var(F16_CACHE_ENV).ok().as_deref());
        if on {
            eprintln!(
                "[CUDA] {F16_CACHE_ENV}=ON: the quantised projections get F16 \
                 dequant caches too"
            );
        }
        on
    })
}

#[cfg(test)]
mod f16_cache_env_tests {
    use super::*;

    /// The documented truthy set, and what is outside it. The values are
    /// arguments, so proving the policy costs no process-environment
    /// mutation and the test is order-independent.
    #[test]
    fn env_truthy_accepts_only_the_documented_values() {
        for v in ["1", "true", "TRUE", "yes", "YES", "on", "ON"] {
            assert!(env_truthy(Some(v)), "{v} is documented as truthy");
        }
        for v in ["0", "", "2", "True", "off", "no", " 1", "1 "] {
            assert!(!env_truthy(Some(v)), "{v} is not in the truthy set");
        }
        assert!(!env_truthy(None), "unset is off");
    }
}

/// What the memory query said just before the F16 caches are built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreeMemory {
    /// `cuMemGetInfo` succeeded; the device has this many bytes free.
    Bytes(u64),
    /// `cuMemGetInfo` failed; the caches are built on the assumption that
    /// the load that got this far will fit, and the caller logs the failure.
    Unknown,
}

/// The refusal to build the F16 dequant caches when they cannot fit, or
/// `None` when they can, when nothing needs building, when the memory query
/// failed (a telemetry failure is not a memory shortage), or when
/// [`F16_CACHE_FORCE_ENV`] is set. `needed` is the byte total the caches will
/// allocate; the remaining arguments only make the message concrete.
pub fn f16_cache_refusal(
    needed: u64,
    free: FreeMemory,
    headroom: u64,
    forced: bool,
    attention_layers: usize,
    max_seq_len: usize,
    kv_bytes: u64,
) -> Option<String> {
    if needed == 0 || forced {
        return None;
    }
    let free = match free {
        FreeMemory::Bytes(b) => b,
        FreeMemory::Unknown => return None,
    };
    if let Some(total) = needed.checked_add(headroom) {
        if total <= free {
            return None;
        }
    }
    let gb = |b: u64| b as f64 / 1.0e9;
    Some(format!(
        "F16 dequant caches for {attention_layers} attention layers need {:.2} GB \
         (+{:.2} GB margin) but only {:.2} GB of device memory is free after the \
         weights and the {max_seq_len}-token KV cache ({:.2} GB). Lower --context-len, \
         use a smaller quantization, or set {F16_CACHE_FORCE_ENV}=1 to build them anyway.",
        gb(needed),
        gb(headroom),
        gb(free),
        gb(kv_bytes)
    ))
}

/// Per-process default for `LUMEN_CUDA_SOA_LOCKED` when the env is unset.
///
/// ON only for quantised dense models on a measured-good compute capability
/// (8.x and 9.x, where the codegen-locked Q4_0 split matvec was tuned:
/// word-load nibble stream, load-hoist, `.rn`-pinned epilogue,
/// bit-deterministic and faster than the unlocked split kernel). OFF on
/// every other capability, including 12.x, 10.x and a device whose
/// capability query failed (`device_cc_major() == 0`), because the kernel
/// is unmeasured there, not because it is known to be slow: on an RTX 5090,
/// in a configuration that fits, forcing the lever on does not reproduce
/// the 1.1 tok/s once attributed to it (75.4–75.5 tok/s either way, five
/// fresh processes per arm; whether the locked kernel executed in the
/// forced arm is not shown by that probe's logs). `=1` still forces it on.
/// The effect is gated downstream by Q4 split dispatch and the locked
/// kernel's presence (`matvec_q4_split_q8_1_locked`), so on a Q8/BF16 model
/// it is a no-op.
///
/// **MoE: explicit OFF.** `SOA_LOCKED` implies the Q4 split clone pass
/// (`repack_all_layers_q4_clone_to_split`), which populates the dense
/// `wq/wk/wv/wo/w_gate/w_up/w_down` siblings only. On an MoE LBC the dense
/// MLP is replaced by per-expert weights, so arming the clone there would
/// partially populate siblings exactly as the Q8 SPLIT pass did before its
/// MoE gate (PAD-token spam). Gating OFF for MoE mirrors `q8_split_default`.
pub fn soa_locked_default() -> bool {
    if !matches!(device_cc_major(), 8 | 9) {
        return false;
    }
    match MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) {
        // Q4 dense benefits (Q8/BF16/F32 lack the locked kernel → no-op).
        // MoE: explicit OFF (clone-pass hazard, mirrors q8_split_default).
        HINT_QUANTISED if !model_is_moe() => canonical_default_on(),
        _ => false,
    }
}

/// Per-process default for `LUMEN_CUDA_OUTPUT_PROJ_SPLIT` when unset. ON
/// for Q8 dense (output projection in particular). Gated by
/// `quantized_output_head_canonical` — deliberately NARROWER than `q8_split_default`,
/// which also turns on for BF16 dense.
pub fn output_proj_split_default() -> bool {
    quantized_output_head_canonical()
}

/// Per-process default for `LUMEN_CUDA_Q8_SCALE_HW` when unset. ON for
/// Q8 dense (prefer the `matvec_q8_aligned_q8_1_hw` kernel that uses
/// hardware-scale dp4a; no-op when the kernel is absent or not Q8 dense).
pub fn q8_scale_hw_default() -> bool {
    quantized_output_head_canonical()
}

/// Per-process default for `LUMEN_CUDA_OUTPUT_PROJ_NR` when unset. Returns
/// `16` for Q8 dense (the measured optimum). `1` is the legacy
/// default for any other configuration.
pub fn output_proj_nr_default() -> u32 {
    if quantized_output_head_canonical() {
        16
    } else {
        1
    }
}

// ---------------------------------------------------------------------------
// Env-var typo validator
// ---------------------------------------------------------------------------

/// Canonical allowlist of `LUMEN_*` env vars recognised across the
/// runtime, CLI, server, and bench crates. Generated by `grep -rEoh
/// '"LUMEN_[A-Z0-9_]+"' crates/` and reviewed manually. ADD new names here
/// when a new env gate ships, or the validator will warn at startup. Names
/// that only the repository's own scripts define belong in
/// [`KNOWN_LUMEN_TOOLING_ENV_VARS`] instead.
///
/// Sorted alphabetically to make `diff` reviewable when the list changes; a
/// test holds both lists to that order.
const KNOWN_LUMEN_ENV_VARS: &[&str] = &[
    "LUMEN_AB_ITERATIONS",
    "LUMEN_AB_WARMUP",
    "LUMEN_ANTI_RESTATE",
    "LUMEN_ANTI_RESTATE_LOOP",
    "LUMEN_ANTI_RESTATE_NGRAM",
    "LUMEN_ANTI_RESTATE_SUBWORD",
    "LUMEN_BASE_URL",
    "LUMEN_BENCH_ITERATIONS",
    "LUMEN_BENCH_MASK_EOG",
    "LUMEN_BENCH_SCALE",
    "LUMEN_BENCH_TOKENS",
    "LUMEN_BENCH_TOKEN_IDS",
    "LUMEN_BENCH_TOP2",
    "LUMEN_BENCH_WARMUP",
    "LUMEN_CACHE_DIR",
    "LUMEN_CHAT_ENABLE_THINKING",
    "LUMEN_CONVERT_KEEP_Q6K_OUTPUT",
    "LUMEN_CONVERT_SOURCE_FIDELITY",
    "LUMEN_CORR010_MODEL",
    "LUMEN_CUDA_ARGMAX_TILED",
    "LUMEN_CUDA_ATTN_BANK3",
    "LUMEN_CUDA_ATTN_PREFILL_SGEMM",
    "LUMEN_CUDA_ATTN_PREP_FUSE",
    "LUMEN_CUDA_ATTN_SPLITK",
    "LUMEN_CUDA_ATTN_SPLITK_CHUNK",
    "LUMEN_CUDA_ATTN_SPLITK_GQA6",
    "LUMEN_CUDA_ATTN_SPLITK_SCALE",
    "LUMEN_CUDA_ATTN_TILED_CODEGEN",
    "LUMEN_CUDA_BF16_AB_Q8BANK",
    "LUMEN_CUDA_BF16_AUTOTUNE",
    "LUMEN_CUDA_BF16_FUSED_GLU",
    "LUMEN_CUDA_BF16_GEMMEX",
    "LUMEN_CUDA_BF16_MATVEC",
    "LUMEN_CUDA_BF16_MOE_V3",
    "LUMEN_CUDA_BF16_NR1",
    "LUMEN_CUDA_BF16_WO_NR1",
    "LUMEN_CUDA_CT4_DP4A",
    "LUMEN_CUDA_CT4_EXACTK",
    "LUMEN_CUDA_DECODE_DELAY_US",
    "LUMEN_CUDA_DECODE_TILED",
    "LUMEN_CUDA_DECODE_TILED_THRESHOLD",
    "LUMEN_CUDA_F16_CACHE",
    "LUMEN_CUDA_F16_CACHE_FORCE",
    "LUMEN_CUDA_FFN_DIRECT_RESIDUAL",
    "LUMEN_CUDA_FFN_FUSED_GLU",
    "LUMEN_CUDA_FFN_GATE_UP_BANK",
    "LUMEN_CUDA_FORCE_SCALAR_ATTN",
    "LUMEN_CUDA_GDN_AB_F16",
    "LUMEN_CUDA_GDN_AB_F32",
    "LUMEN_CUDA_GDN_CONVSTATE_PARITY",
    "LUMEN_CUDA_GDN_DECODE_MEGAKERNEL_F64",
    "LUMEN_CUDA_GDN_DECODE_VIA_PREFILL",
    "LUMEN_CUDA_GDN_F64_ACCUM",
    "LUMEN_CUDA_GDN_NG_Q8",
    "LUMEN_CUDA_GDN_P123_FUSE",
    "LUMEN_CUDA_GDN_PREFILL_F64",
    "LUMEN_CUDA_GDN_REGISTER_RESIDENT",
    "LUMEN_CUDA_GDN_SKIP_DUP_QKV",
    "LUMEN_CUDA_GDN_SUBSTAGE_TIMING",
    "LUMEN_CUDA_GPU_SAMPLE",
    "LUMEN_CUDA_LEGACY_DEFAULTS",
    "LUMEN_CUDA_MAX_SEQ_LEN",
    "LUMEN_CUDA_MMV_BF16_OUTPUT_PROJ",
    "LUMEN_CUDA_MMV_Q_DP4A",
    "LUMEN_CUDA_MMV_Q_MOE_DP4A",
    "LUMEN_CUDA_MMV_Q_OUTPUT_PROJ",
    "LUMEN_CUDA_MOE_BATCHED",
    "LUMEN_CUDA_MOE_BATCHED_V2",
    "LUMEN_CUDA_MOE_BATCHED_V3",
    "LUMEN_CUDA_MOE_BF16_NATIVE",
    "LUMEN_CUDA_MOE_DECODE_F32",
    "LUMEN_CUDA_MOE_DECODE_F32_FFN",
    "LUMEN_CUDA_MOE_DOWN_TILED_F32ACT",
    "LUMEN_CUDA_MOE_FUSED_NORM_ROUTER",
    "LUMEN_CUDA_MOE_GATE_UP_W10",
    "LUMEN_CUDA_MOE_GROUPED_TILED",
    "LUMEN_CUDA_MOE_PREFILL_BATCHED",
    "LUMEN_CUDA_MOE_Q4_V3",
    "LUMEN_CUDA_MOE_Q4_V3B",
    "LUMEN_CUDA_MOE_RESIDUAL_Q8",
    "LUMEN_CUDA_MOE_ROUTER_PARALLEL",
    "LUMEN_CUDA_NORM_CTA5_DUAL",
    "LUMEN_CUDA_OUTPUT_PROJ_NR",
    "LUMEN_CUDA_OUTPUT_PROJ_SPLIT",
    "LUMEN_CUDA_PREFILL_F32",
    "LUMEN_CUDA_PROFILE",
    "LUMEN_CUDA_PROFILE_ATTN_LEAF",
    "LUMEN_CUDA_PTX_CACHE",
    "LUMEN_CUDA_PTX_CACHE_DIR",
    "LUMEN_CUDA_Q4_1_DOWN",
    "LUMEN_CUDA_Q4_B160",
    "LUMEN_CUDA_Q4_DOWN_NR1",
    "LUMEN_CUDA_Q4_F32ACT_KERNEL",
    "LUMEN_CUDA_Q4_MMVQ",
    "LUMEN_CUDA_Q4_PROJ_BANK",
    "LUMEN_CUDA_Q4_SPLIT",
    "LUMEN_CUDA_Q4_SPLIT_ATTN",
    "LUMEN_CUDA_Q4_SPLIT_BUDGET_GB",
    "LUMEN_CUDA_Q4_SPLIT_WO",
    "LUMEN_CUDA_Q4_V4LOAD",
    "LUMEN_CUDA_Q5K_SSMOUT",
    "LUMEN_CUDA_Q6K_HEAD",
    "LUMEN_CUDA_Q8_AB_BANK",
    "LUMEN_CUDA_Q8_MATVEC_FAST",
    "LUMEN_CUDA_Q8_MMVQ",
    "LUMEN_CUDA_Q8_PROJ_MMQ",
    "LUMEN_CUDA_Q8_SCALE_HW",
    "LUMEN_CUDA_Q8_SPLIT",
    "LUMEN_CUDA_Q8_SPLIT_ATTN",
    "LUMEN_CUDA_Q8_SPLIT_BUDGET_GB",
    "LUMEN_CUDA_Q8_SPLIT_SSMOUT",
    "LUMEN_CUDA_Q8_SPLIT_WO",
    "LUMEN_CUDA_ROPE_TAB",
    "LUMEN_CUDA_SHARED_FUSED_DECODE",
    "LUMEN_CUDA_SHARED_TILED",
    "LUMEN_CUDA_SKIP_BF16_PROBE",
    "LUMEN_CUDA_SOA_LOCKED",
    "LUMEN_CUDA_SSMOUT_RESID_FOLD",
    "LUMEN_CUDA_TOPK_MOE_FUSED",
    "LUMEN_CUDA_VERBOSE",
    "LUMEN_DUMP_EXPERTS",
    "LUMEN_DUMP_GDN_L0_BIN",
    "LUMEN_DUMP_NORMED",
    "LUMEN_FREQUENCY_PENALTY",
    "LUMEN_KV_PRECISION",
    "LUMEN_METAL_ATTN_PRECISE",
    "LUMEN_METAL_BF16_GATE_UP_NR",
    "LUMEN_METAL_BF16_GDN_FULL_PREFILL_WARMUP",
    "LUMEN_METAL_BF16_GDN_QKV_GATE_PAIRED",
    "LUMEN_METAL_BF16_MMAP_ONLY",
    "LUMEN_METAL_CB_SPLIT",
    "LUMEN_METAL_CONCURRENT_ENCODER",
    "LUMEN_METAL_CONCURRENT_ENCODER_VALIDATE",
    "LUMEN_METAL_DECODE_DELAY_US",
    "LUMEN_METAL_DECODE_GPUTIME",
    "LUMEN_METAL_DECODE_PROFILE",
    "LUMEN_METAL_DEFAULTS_OFF",
    "LUMEN_METAL_FFN_DOWN_SPLITK",
    "LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED",
    "LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_BF16",
    "LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_Q4",
    "LUMEN_METAL_GDN_CONCURRENT_ENCODER",
    "LUMEN_METAL_GDN_CONCURRENT_ENCODER_VALIDATE",
    "LUMEN_METAL_GDN_SSM_OUT_F32_BATCHED",
    "LUMEN_METAL_GPU_SAMPLER",
    "LUMEN_METAL_GPU_SAMPLER_EXACT",
    "LUMEN_METAL_GPU_SAMPLER_QUIET",
    "LUMEN_METAL_MMAP_ONLY",
    "LUMEN_METAL_MOE_GATHER_VEC4",
    "LUMEN_METAL_MOE_GEMM_TILEMAP",
    "LUMEN_METAL_MOE_PREFILL_GROUPED",
    "LUMEN_METAL_MOE_ROUTER_PARALLEL",
    "LUMEN_METAL_MOE_ROUTER_TOPK_TGS",
    "LUMEN_METAL_MOE_ROUTE_SORT",
    "LUMEN_METAL_MOE_ROUTE_SORT_PAR",
    "LUMEN_METAL_NAN_DUMP",
    "LUMEN_METAL_PREFILL_GPUTIME",
    "LUMEN_METAL_PROFILE",
    "LUMEN_METAL_Q8_GDN_QKVGATE_2STREAM",
    "LUMEN_METAL_Q8_REPACKED",
    "LUMEN_METAL_Q8_REPACKED_FFN_DOWN",
    "LUMEN_METAL_Q8_REPACKED_GATE_UP",
    "LUMEN_METAL_UNRETAINED_CMDBUFS",
    "LUMEN_MOE_PROBE",
    "LUMEN_PREFILL_TIMING",
    "LUMEN_QWEN35_9B_BF16",
    "LUMEN_QWEN35_9B_PATH",
    "LUMEN_QWEN35_9B_Q4",
    "LUMEN_QWEN35_9B_Q8",
    "LUMEN_REPEAT_LAST_N",
    "LUMEN_REPETITION_PENALTY",
    "LUMEN_SERVER_DEBUG_MEM",
    "LUMEN_SERVER_PANIC_MAX",
    "LUMEN_SERVER_PANIC_WINDOW_SECS",
    "LUMEN_SOAK_DURATION_SEC",
    "LUMEN_SOAK_OUT_DIR",
    "LUMEN_SOAK_STACK_DUMP",
    "LUMEN_SOAK_STACK_LEAKS",
    "LUMEN_SOAK_STACK_TICKS",
    "LUMEN_SOAK_WARMUP_SEC",
    "LUMEN_SPEC_DUMP_IDS",
    "LUMEN_SUFFIX_THRESHOLD",
    "LUMEN_TEST_OPENAI_SDK",
    "LUMEN_XCHK",
    "LUMEN_XCHK2",
];

/// `LUMEN_*` names the repository's own scripts, packaging and CI define and
/// consume themselves; the engine never reads them. They still reach an
/// engine process's environment, because the shell that exports them then
/// runs the binary (the release workflow exports `LUMEN_BIN` and
/// `LUMEN_SERVER_BIN` before the packaged binaries run; the installer's
/// `LUMEN_MODEL` and `LUMEN_QUANT` are inherited by its `lumen pull`), so the
/// typo validator must know them or every such run warns about a name that
/// is not a typo. A test derives the expected set from the scripts themselves
/// and fails on drift in either direction.
///
/// The cost is one near-miss the validator can no longer catch: a name here
/// that resembles an engine name (`LUMEN_CACHE_ROOT` beside
/// `LUMEN_CACHE_DIR`) is accepted as the tooling name it is.
///
/// Sorted alphabetically, like the list above.
const KNOWN_LUMEN_TOOLING_ENV_VARS: &[&str] = &[
    "LUMEN_ALLOW_INSECURE_BASE",
    "LUMEN_BIN",
    "LUMEN_CACHE_ROOT",
    "LUMEN_DET_MAXTOK",
    "LUMEN_DET_MODEL",
    "LUMEN_DET_PORT",
    "LUMEN_INSECURE_SKIP_CHECKSUM",
    "LUMEN_MODEL",
    "LUMEN_ONLY",
    "LUMEN_PREFIX",
    "LUMEN_QS_BACKEND",
    "LUMEN_QS_FAKE_FREE_GIB",
    "LUMEN_QS_HOST",
    "LUMEN_QS_MODEL",
    "LUMEN_QS_PORT",
    "LUMEN_QS_QUANT",
    "LUMEN_QS_VERBOSE",
    "LUMEN_QS_YES",
    "LUMEN_QUANT",
    "LUMEN_RELEASE_BASE",
    "LUMEN_ROOT",
    "LUMEN_SERVER_BIN",
    "LUMEN_TAG",
    "LUMEN_TEST_MODEL",
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

    // Pass 1 — names that start with `LUMEN_` but are neither an engine
    // env nor one of the repository's own tooling names. This catches
    // mis-spelled suffixes on otherwise-correct env names.
    let mut unknown_with_prefix: Vec<&String> = env_vars
        .iter()
        .filter(|k| k.starts_with("LUMEN_"))
        .filter(|k| {
            !KNOWN_LUMEN_ENV_VARS
                .iter()
                .chain(KNOWN_LUMEN_TOOLING_ENV_VARS.iter())
                .any(|known| *known == k.as_str())
        })
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
    // prevents emitting the same warning twice if a single suffix
    // matches more than one canonical root.
    let mut already_seen: std::collections::HashSet<&String> = std::collections::HashSet::new();
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

/// Resets the process-wide hint atomics to their defaults, with the device
/// capability set to a measured-good one (8) so the capability gate does not
/// mask the model-shape rules the tests exercise. Test-only — used by the
/// unit tests below so each test starts from a known state.
#[doc(hidden)]
pub fn reset_for_tests() {
    // A measured-good capability, so the capability gate does not mask the
    // model-shape rules the tests below exercise (an unknown capability
    // returns early, before the MoE arm is reached).
    DEVICE_CC_MAJOR.store(8, Ordering::Relaxed);
    PATH_IS_SERVER.store(false, Ordering::Relaxed);
    MODEL_DENSE_QUANT_HINT.store(HINT_UNSET, Ordering::Relaxed);
    MODEL_PRIMARY_QUANT_SCHEME.store(QUANT_SCHEME_UNSET, Ordering::Relaxed);
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

// ---------------------------------------------------------------------------
// Fixed-horizon bench surfaces (off unless set)
// ---------------------------------------------------------------------------

/// The masked-logit value: the most negative finite `f32`, which is what the
/// literal `-3.402823466e+38f` in `mask_logits_f32_min` rounds to (see
/// `cuda/shaders/argmax.cu`). The GPU and host mask write the same bits, so
/// the two selection paths can never disagree on a tie. Finite, not `-inf`,
/// so a masked logit stays a valid comparand in every reduction.
pub(crate) const EOG_MASK_SENTINEL: f32 = f32::MIN;

/// Upper bound on the number of masked ids: the device mask runs as one
/// thread block, one id per thread.
const EOG_MASK_MAX_IDS: usize = 1024;

/// `LUMEN_BENCH_MASK_EOG="id1,id2"` — end-of-generation token ids that a
/// greedy decode may never select, so a `max_tokens` request does not stop
/// at an end-of-generation token before the horizon: the "most likely
/// continuation, conditional on continuing" that a fixed-horizon quality
/// comparison across engines is defined on. A bench surface, not a serving
/// feature; unset means shipping behaviour and not one logit is touched.
///
/// Single source of truth for BOTH selection paths. The CUDA greedy path
/// masks on the device at `launch_argmax`; every host-side selection masks in
/// `engine::sample_token_with_state`. They must agree, so both read this. A
/// backend whose greedy selection runs on the device without the mask (Metal)
/// refuses to construct while the variable is set, rather than running
/// unmasked under the marker line.
///
/// # Fail-closed
///
/// A malformed value panics at first use, naming the variable. It does not
/// fall back to an unmasked run: a prompt that would not have ended within
/// the horizon anyway yields exactly N tokens with no end-of-generation id
/// either way, so nothing downstream can tell a masked run from an unmasked
/// one except the marker line. Refusing is the only reliable guard.
pub(crate) fn bench_mask_eog_ids() -> &'static [u32] {
    static IDS: OnceLock<Vec<u32>> = OnceLock::new();
    IDS.get_or_init(|| {
        let raw = match std::env::var("LUMEN_BENCH_MASK_EOG") {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        match parse_eog_mask_ids(&raw) {
            Ok(ids) => {
                if !ids.is_empty() {
                    // Unconditional: the line's absence is what shows a run
                    // was not masked.
                    eprintln!(
                        "[BENCH] MASK_EOG=ON ids=[{}] count={} (fixed-horizon protocol surface)",
                        ids.iter()
                            .map(|i| i.to_string())
                            .collect::<Vec<_>>()
                            .join(","),
                        ids.len()
                    );
                }
                ids
            }
            Err(e) => panic!("{e}"),
        }
    })
}

/// Exact-value parse for [`bench_mask_eog_ids`], separated from the
/// environment read so the accepted grammar is testable on its own.
///
/// Empty or all-whitespace is the empty set (mask off). Any other malformed
/// input is an error, never a silently smaller mask.
pub(crate) fn parse_eog_mask_ids(raw: &str) -> Result<Vec<u32>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let mut ids: Vec<u32> = Vec::new();
    for part in trimmed.split(',') {
        let t = part.trim();
        if t.is_empty() {
            return Err(format!(
                "LUMEN_BENCH_MASK_EOG={raw:?}: empty element (doubled or trailing comma). \
                 Refusing rather than running with a smaller mask than written."
            ));
        }
        let id: u32 = t.parse().map_err(|_| {
            format!(
                "LUMEN_BENCH_MASK_EOG={raw:?}: element {t:?} is not a u32 token id. \
                 Refusing rather than running unmasked."
            )
        })?;
        if ids.contains(&id) {
            return Err(format!(
                "LUMEN_BENCH_MASK_EOG={raw:?}: duplicate id {id}. Refusing so the marker \
                 line cannot disagree with the value."
            ));
        }
        ids.push(id);
        if ids.len() > EOG_MASK_MAX_IDS {
            return Err(format!(
                "LUMEN_BENCH_MASK_EOG={raw:?}: more than {EOG_MASK_MAX_IDS} ids."
            ));
        }
    }
    Ok(ids)
}

/// Refuse a mask that names an id outside the model's vocabulary. Both mask
/// implementations bounds-check and would otherwise skip such an id silently,
/// which is an unmasked id under the marker line. Called wherever the
/// vocabulary size first becomes known (backend init, server start).
pub fn check_eog_mask_vocab(vocab_size: usize) -> Result<(), String> {
    let ids = bench_mask_eog_ids();
    match ids.iter().find(|&&id| id as usize >= vocab_size) {
        None => Ok(()),
        Some(id) => Err(format!(
            "LUMEN_BENCH_MASK_EOG={}: id {id} is outside the vocabulary (size {vocab_size}). \
             Refusing rather than running with that id unmasked.",
            ids.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )),
    }
}

/// Apply the mask to a host logits slice, in place: the host twin of
/// `mask_logits_f32_min`, writing the same value. A no-op when the mask is
/// off.
pub(crate) fn apply_eog_mask(logits: &mut [f32]) {
    let ids = bench_mask_eog_ids();
    if !ids.is_empty() {
        mask_logits_in_place(logits, ids);
    }
}

/// Write [`EOG_MASK_SENTINEL`] at every `id` in `ids`. Out-of-range ids are
/// skipped, matching the kernel's bounds guard; [`check_eog_mask_vocab`] has
/// already refused such a mask before any decode.
fn mask_logits_in_place(logits: &mut [f32], ids: &[u32]) {
    for &id in ids {
        if let Some(slot) = logits.get_mut(id as usize) {
            *slot = EOG_MASK_SENTINEL;
        }
    }
}

/// `LUMEN_BENCH_TOP2=1`: a bench surface that records, for every generated token, the
/// argmax of the logits as the session received them and the runner-up with both logits
/// (the engine's own numbers; the selected token is the token-id record's entry). Off by default;
/// it moves greedy decode off the on-device argmax route onto the host-logits route,
/// which runs the same kernels and costs one vocabulary-sized copy per token. Implies
/// `LUMEN_BENCH_TOKEN_IDS`. Read once per process.
pub fn bench_top2_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = env_is_exactly_one("LUMEN_BENCH_TOP2");
        if on {
            eprintln!(
                "[BENCH] TOP2=ON: responses carry the argmax and the runner-up with \
                 both logits per generated token (greedy decode on the host-logits route)"
            );
        }
        on
    })
}

/// `LUMEN_BENCH_TOKEN_IDS=1` — `lumen-server` non-streaming responses
/// additionally carry the raw generated token-id array, the finish reason,
/// and the per-request EOS set, under a top-level `lumen_bench` object. A
/// bench surface for comparing end-of-generation behaviour across engines on
/// the ids themselves rather than on re-tokenised text; unset means shipping
/// responses, byte for byte.
///
/// Exact-value: `1` only. `true`, `on`, `0` and an empty value are all off,
/// so a bench surface cannot be armed by a truthy-looking typo.
pub fn bench_token_ids_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        // LUMEN_BENCH_TOP2 implies this surface (its record is aligned with the token ids), so
        // every guard that reads this — the streaming/stop-sequence refusal, the attach step —
        // sees the implication without each spelling it out.
        let on =
            env_is_exactly_one("LUMEN_BENCH_TOKEN_IDS") || env_is_exactly_one("LUMEN_BENCH_TOP2");
        if on {
            eprintln!(
                "[BENCH] TOKEN_IDS=ON: responses carry raw generated token ids + \
                 eos set (instrument-only surface)"
            );
        }
        on
    })
}

/// `true` only when `name` is set to exactly `1`: no trimming, no truthy words.
fn env_is_exactly_one(name: &str) -> bool {
    matches!(std::env::var(name), Ok(v) if v == "1")
}

#[cfg(test)]
mod fixed_horizon_bench_tests {
    use super::*;
    use crate::ENV_TEST_LOCK as SERIAL;

    /// Exact-value and fail-closed: empty means off and is legal; anything
    /// malformed is an error, never a smaller or empty mask.
    #[test]
    fn eog_mask_parse_is_exact_value_and_fail_closed() {
        assert_eq!(parse_eog_mask_ids("").unwrap(), Vec::<u32>::new());
        assert_eq!(parse_eog_mask_ids("   ").unwrap(), Vec::<u32>::new());
        assert_eq!(parse_eog_mask_ids("248046").unwrap(), vec![248046]);
        assert_eq!(
            parse_eog_mask_ids(" 248046 , 248044 ").unwrap(),
            vec![248046, 248044]
        );
        for bad in [
            "1,2x",
            "248O46",
            "248046,",
            "248046,,7",
            "-1",
            "1e3",
            "0x1",
            " ,",
            "248046 248044",
            "248046,248046",
        ] {
            let r = parse_eog_mask_ids(bad);
            assert!(r.is_err(), "{bad:?} must be rejected, got {r:?}");
            assert!(
                r.unwrap_err().starts_with("LUMEN_BENCH_MASK_EOG="),
                "every refusal names the variable"
            );
        }
        assert_eq!(parse_eog_mask_ids("0").unwrap(), vec![0]);
        let too_many = (0..=EOG_MASK_MAX_IDS as u32)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert!(
            parse_eog_mask_ids(&too_many).is_err(),
            "one thread block is the bound"
        );
        let at_bound = (0..EOG_MASK_MAX_IDS as u32)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(
            parse_eog_mask_ids(&at_bound).unwrap().len(),
            EOG_MASK_MAX_IDS
        );
    }

    /// The host mask must write the same bits the kernel writes.
    #[test]
    fn host_sentinel_matches_the_kernel_literal() {
        const SRC: &str = include_str!("cuda/shaders/argmax.cu");
        assert!(
            SRC.contains("logits[id] = -3.402823466e+38f;"),
            "argmax.cu no longer writes the literal this host path mirrors"
        );
        let kernel_literal: f32 = "-3.402823466e+38".parse().unwrap();
        assert_eq!(EOG_MASK_SENTINEL.to_bits(), kernel_literal.to_bits());
        assert!(EOG_MASK_SENTINEL.is_finite());
    }

    /// The host mask makes every masked id lose the argmax and touches no
    /// other slot; out-of-range ids are skipped, as in the kernel.
    #[test]
    fn host_mask_removes_masked_ids_and_nothing_else() {
        fn argmax(v: &[f32]) -> usize {
            v.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap()
        }
        let orig: Vec<f32> = (0..64).map(|i| (i as f32 * 7.0) % 11.0).collect();
        let mut logits = orig.clone();
        logits[7] = 100.0;
        logits[9] = 50.0;
        assert_eq!(argmax(&logits), 7);
        mask_logits_in_place(&mut logits, &[7, 9, 64, u32::MAX]);
        assert_ne!(argmax(&logits), 7);
        assert_ne!(argmax(&logits), 9);
        assert_eq!(logits[7].to_bits(), EOG_MASK_SENTINEL.to_bits());
        assert_eq!(logits[9].to_bits(), EOG_MASK_SENTINEL.to_bits());
        for i in (0..64).filter(|i| *i != 7 && *i != 9) {
            assert_eq!(logits[i], orig[i], "slot {i} must be untouched");
        }
        let again = logits.clone();
        mask_logits_in_place(&mut logits, &[7, 9]);
        assert_eq!(logits, again, "idempotent");
    }

    /// Mask off must be a true no-op on the logits buffer.
    #[test]
    fn mask_off_is_a_no_op() {
        if std::env::var("LUMEN_BENCH_MASK_EOG").is_ok() {
            eprintln!("skipping: LUMEN_BENCH_MASK_EOG is set in this environment");
            return;
        }
        let orig: Vec<f32> = (0..128).map(|i| i as f32 * 0.5 - 3.0).collect();
        let mut logits = orig.clone();
        apply_eog_mask(&mut logits);
        assert_eq!(logits, orig);
        assert!(
            check_eog_mask_vocab(1).is_ok(),
            "no mask, nothing to refuse"
        );
    }

    /// Both GPU argmax variants are dispatched inside the one function the
    /// mask is applied in, after the mask; the host sampler masks before it
    /// selects; and Metal, whose greedy argmax runs on the device without a
    /// mask, refuses to construct while the mask is set.
    #[test]
    fn every_selection_path_is_masked_or_refused() {
        const BE: &str = include_str!("cuda/backend_impl.rs");
        const EN: &str = include_str!("engine.rs");
        const MT: &str = include_str!("metal/mod.rs");
        assert_eq!(
            BE.matches("st.kernels.mask_logits_f32_min").count(),
            1,
            "the mask kernel handle is read exactly once, inside launch_argmax"
        );
        let la = BE.find("fn launch_argmax(").expect("launch_argmax");
        let rest = &BE[la..];
        let body_end = rest[1..]
            .find("\n    fn ")
            .map(|i| i + 1)
            .unwrap_or(rest.len());
        let body = &rest[..body_end];
        let mask_at = body
            .find("st.kernels.mask_logits_f32_min")
            .expect("mask inside launch_argmax");
        let tiled_at = body
            .find("argmax_f32_tile_phase1")
            .expect("tiled dispatch inside launch_argmax");
        let single_at = body
            .find("kernels.argmax_f32")
            .expect("single-block dispatch inside launch_argmax");
        assert!(
            mask_at < tiled_at && mask_at < single_at,
            "the mask precedes both dispatches"
        );
        assert_eq!(BE.matches("argmax_f32_tile_phase1.clone()").count(), 1);
        assert_eq!(
            BE.matches("launch_builder(&st.kernels.argmax_f32)").count(),
            1,
            "a second argmax dispatch site would bypass the mask"
        );

        let sts = EN.find("pub fn sample_token_with_state(").expect("sampler");
        let rest = &EN[sts..];
        let fn_end = rest.find("\n}\n").expect("fn end");
        let body = &rest[..fn_end];
        let mask_at = body
            .find("apply_eog_mask")
            .expect("the host sampler must mask");
        let sample_at = body.find("sample_logits").expect("sample_logits call");
        assert!(
            mask_at < sample_at,
            "the mask must be applied before selection"
        );

        let mn = MT
            .find("pub fn new() -> Result<Self, RuntimeError> {")
            .expect("Metal constructor");
        let head = &MT[mn..mn + 1200];
        assert!(
            head.contains("bench_mask_eog_ids()"),
            "the Metal constructor must refuse while the mask is set"
        );
    }

    /// An id outside the vocabulary is refused, naming the variable.
    #[test]
    fn out_of_vocabulary_id_is_refused() {
        // Exercised through the parse + bound helpers; the env-backed
        // resolver is covered by the live server run.
        let ids = parse_eog_mask_ids("5,248046").unwrap();
        let bad = ids.iter().find(|&&id| id as usize >= 248046);
        assert_eq!(bad, Some(&248046));
    }

    /// `=1` only: no presence-parse, no truthy words, no whitespace.
    #[test]
    fn token_ids_flag_is_exact_value_one_only() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let name = "ENGINE_TEST_EXACT_ONE_PROBE";
        for (v, want) in [
            ("1", true),
            ("0", false),
            ("true", false),
            ("on", false),
            ("yes", false),
            ("", false),
            (" 1", false),
            ("1 ", false),
            ("01", false),
        ] {
            std::env::set_var(name, v);
            assert_eq!(env_is_exactly_one(name), want, "value {v:?}");
        }
        std::env::remove_var(name);
        assert!(!env_is_exactly_one(name));
    }

    /// Off is shipping behaviour: the resolver reports false.
    #[test]
    fn token_ids_off_is_shipping_behaviour() {
        if std::env::var("LUMEN_BENCH_TOKEN_IDS").is_ok() {
            eprintln!("skipping: LUMEN_BENCH_TOKEN_IDS is set in this environment");
            return;
        }
        assert!(!bench_token_ids_enabled());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The tests in this module mutate process-wide state (atomics + env).
    // Cargo runs tests in parallel within a binary by default; the
    // crate-wide env lock enforces that exactly one test at a time observes
    // the global state we toggle (env mutation in ANY module races env reads
    // here, so the lock must be crate-global). Taken FIRST in each test.
    use crate::ENV_TEST_LOCK as SERIAL;

    #[test]
    fn ct4_role_parsing() {
        // Pure parser — no env access, no SERIAL lock needed.
        assert_eq!(parse_ct4_f16_roles("").unwrap(), None);
        assert_eq!(parse_ct4_f16_roles("1").unwrap(), None);
        assert_eq!(parse_ct4_f16_roles(" 0 ").unwrap(), Some(vec!["*".into()]));
        assert_eq!(
            parse_ct4_f16_roles("ssm_out, w_down").unwrap(),
            Some(vec!["ssm_out".into(), "w_down".into()])
        );
        // Naming any of q/k/v pulls in the whole trio (the QKV path
        // dispatches all three together).
        let qkv = parse_ct4_f16_roles("wk").unwrap().unwrap();
        for role in ["wk", "wq", "wv"] {
            assert!(qkv.iter().any(|r| r == role), "missing {role}");
        }
        // Typos and unknown roles are errors, not silent no-ops.
        assert!(parse_ct4_f16_roles("w_dwon").is_err());
        assert!(parse_ct4_f16_roles("false").is_err());
    }

    #[test]
    fn server_default_decode_delay_is_50us() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_path_is_server(true);
        assert_eq!(cuda_decode_delay_us_default(), 50);
        reset_for_tests();
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
        // The three DET-001 intra-kernel cross-threadgroup races in the
        // decode path are fixed at the kernel level (see
        // tests/metal_greedy_determinism_test.rs), so the mitigation delay
        // (~0.45% TPOT, never a hard guarantee) is no longer needed. The
        // Metal default is 0 (bit-exact) on BOTH paths;
        // LUMEN_METAL_DECODE_DELAY_US remains available for diagnostics.
        reset_for_tests();
        set_path_is_server(true);
        assert_eq!(
            metal_decode_delay_us_default(),
            0,
            "Metal server default must be 0 (DET-001 fixed)"
        );
        reset_for_tests();
        set_path_is_server(false);
        assert_eq!(
            metal_decode_delay_us_default(),
            0,
            "Metal CLI default must be 0 (DET-001 fixed)"
        );
        reset_for_tests();
        assert_eq!(
            metal_decode_delay_us_default(),
            0,
            "Metal default must be 0 even with no setter call"
        );
    }

    #[test]
    fn bf16_dense_enables_gemmex() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Bf16);
        assert!(bf16_gemmex_default());
    }

    #[test]
    fn gdn_decode_via_prefill_default_per_class() {
        // 2026-06-12 follow-up: the 27B-bf16 carve-OUT is removed, so
        // via-prefill is now ON for EVERY class. The bf16 dimension is driven by
        // the COARSE `MODEL_DENSE_QUANT_HINT` (set via `set_model_dense_quant`
        // from output_proj), NOT the primary-quant atomic — bf16 LBCs report
        // output_proj == Bf16, so the coarse hint is the right signal here.
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());

        // Unset (legacy caller, no LBC): non-bf16 hint -> ON.
        reset_for_tests();
        assert!(
            gdn_decode_via_prefill_default(),
            "unset hint (non-bf16) -> via-prefill ON"
        );

        // Dense non-bf16 (q8/q4), any layer count: ON (unchanged).
        reset_for_tests();
        set_model_block_count(64);
        set_model_dense_quant(QuantScheme::Q8_0);
        assert!(
            gdn_decode_via_prefill_default(),
            "dense 27B q8 -> via-prefill ON (unchanged)"
        );
        reset_for_tests();
        set_model_block_count(64);
        set_model_dense_quant(QuantScheme::Q4_0);
        assert!(
            gdn_decode_via_prefill_default(),
            "dense 27B q4 -> via-prefill ON (unchanged)"
        );

        // 9B bf16 (<=32 layers): ON — MUST stay ON byte-identically (validated
        // 9b-bf16 stack; this is the behavior the follow-up analysis must not change).
        reset_for_tests();
        set_model_block_count(32);
        set_model_dense_quant(QuantScheme::Bf16);
        assert!(
            gdn_decode_via_prefill_default(),
            "dense 9B bf16 (<=32 layers) -> via-prefill ON (PRESERVED, validated stack)"
        );

        // 27B bf16 (>32 layers): ON.
        reset_for_tests();
        set_model_block_count(64);
        set_model_dense_quant(QuantScheme::Bf16);
        assert!(
            gdn_decode_via_prefill_default(),
            "dense 27B bf16 (>32 layers) -> via-prefill ON (carve-out REMOVED)"
        );

        // MoE (any quant, any size): ON.
        reset_for_tests();
        set_model_block_count(64);
        set_model_is_moe(true);
        set_model_dense_quant(QuantScheme::Bf16); // MoE bf16 -> still ON (MoE wins)
        assert!(
            gdn_decode_via_prefill_default(),
            "MoE bf16 -> via-prefill ON"
        );
    }

    #[test]
    fn model_primary_quant_accessor_roundtrips() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        assert_eq!(model_dense_quant(), None, "unset -> None");
        // The output_proj setter must NOT populate the primary-scheme accessor.
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q8_0);
        assert_eq!(
            model_dense_quant(),
            None,
            "set_model_dense_quant (output_proj) must not feed the primary-quant accessor"
        );
        for scheme in [
            QuantScheme::Q4_0,
            QuantScheme::Q8_0,
            QuantScheme::Bf16,
            QuantScheme::Q4_K,
            QuantScheme::F32,
        ] {
            reset_for_tests();
            set_model_primary_quant(scheme);
            assert_eq!(
                model_dense_quant(),
                Some(scheme),
                "primary scheme must round-trip for {scheme:?}"
            );
        }
    }

    #[test]
    fn q8_dense_disables_gemmex() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q8_0);
        assert!(!bf16_gemmex_default());
    }

    #[test]
    fn q4_dense_disables_gemmex() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q4_0);
        assert!(!bf16_gemmex_default());
    }

    #[test]
    fn unset_hint_preserves_legacy_defaults() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        // BF16-gemmex was historically default ON.
        assert!(bf16_gemmex_default());
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
        assert!(
            q8_scale_hw_default(),
            "Q8 dense should default Q8_SCALE_HW=ON"
        );
        assert_eq!(
            output_proj_nr_default(),
            16,
            "Q8 dense should default NR=16"
        );
        assert!(
            ffn_fused_glu_skip_default(),
            "Q8 dense should default to SKIP fused GLU (use dp4a fall-through)"
        );
    }

    #[test]
    fn bf16_dense_q8_default_scope() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Bf16);
        // BF16 dense keeps the Q8-only defaults OFF with ONE exception:
        // Q8_SPLIT defaults ON — the converter Q8-floors the GDN ssm_out
        // tensors, so the clone pass serves exactly those via the split
        // route (48 jobs on 27B).
        assert!(
            q8_split_default(),
            "BF16 dense should default Q8_SPLIT=ON (Q8-floored ssm_out set)"
        );
        assert!(
            !output_proj_split_default(),
            "BF16 should NOT default OUTPUT_PROJ_SPLIT=ON"
        );
        assert!(
            !q8_scale_hw_default(),
            "BF16 should NOT default Q8_SCALE_HW=ON"
        );
        assert_eq!(
            output_proj_nr_default(),
            1,
            "BF16 should default NR=1 (legacy)"
        );
        assert!(
            !ffn_fused_glu_skip_default(),
            "BF16 should NOT default to SKIP fused GLU (kernel is no-op anyway)"
        );
    }

    #[test]
    fn bf16_body_levers_key_on_primary_quant() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        // The BF16 body levers key on the PRIMARY (bulk) scheme, not the
        // coarse output-head hint: the converter Q8-forces GDN alpha/beta on
        // EVERY quant, so a body-scope miss would silently reroute the Q4/Q8
        // models' generic alpha/beta path.
        set_model_primary_quant(QuantScheme::Bf16);
        assert!(
            bf16_ab_q8bank_enabled(),
            "BF16 body should default AB_Q8BANK=ON"
        );
        assert!(bf16_wo_nr1_enabled(), "BF16 body should default WO_NR1=ON");

        reset_for_tests();
        set_model_primary_quant(QuantScheme::Q4_0);
        assert!(
            !bf16_ab_q8bank_enabled(),
            "Q4 body must NOT default AB_Q8BANK=ON (precision-fragile route)"
        );
        assert!(!bf16_wo_nr1_enabled(), "Q4 body must NOT default WO_NR1=ON");

        reset_for_tests();
        set_model_primary_quant(QuantScheme::Q8_0);
        assert!(
            !bf16_ab_q8bank_enabled(),
            "Q8 body must NOT default AB_Q8BANK=ON"
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
        assert!(
            !q8_split_default(),
            "Q8 MoE should NOT default Q8_SPLIT=ON (PAD-spam regression)"
        );
        assert!(
            !output_proj_split_default(),
            "Q8 MoE should NOT default OUTPUT_PROJ_SPLIT=ON"
        );
        assert!(
            !q8_scale_hw_default(),
            "Q8 MoE should NOT default Q8_SCALE_HW=ON"
        );
        assert_eq!(
            output_proj_nr_default(),
            1,
            "Q8 MoE should default NR=1 (legacy), not 16"
        );
        assert!(
            !ffn_fused_glu_skip_default(),
            "Q8 MoE should NOT default FFN_FUSED_GLU_SKIP=ON"
        );
        // The shared MoE flags MUST stay ON (they fire only on MoE anyway).
        assert!(moe_batched_default());
        assert!(moe_router_parallel_default());
        assert!(gdn_register_resident_default());
    }

    #[test]
    fn quantised_dense_enables_soa_locked_default() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        // Q4 dense (lm_head often Q8_0 → HINT_QUANTISED) defaults SOA_LOCKED ON
        // on a measured-good capability (the A100 the kernel was tuned on).
        set_model_dense_quant(QuantScheme::Q8_0);
        set_device_cc_major(8);
        assert!(
            soa_locked_default(),
            "quantised dense on cc 8.x should default SOA_LOCKED=ON"
        );
    }

    #[test]
    fn split_clone_budget_is_free_minus_the_slack_and_nothing_else() {
        let gb = |x: f64| (x * 1e9) as usize;
        // A100 after a 27B-Q8 load: 46 GB free, 2 GB slack -> 44 GB.
        assert_eq!(split_clone_budget_bytes(gb(46.0), gb(2.0)), gb(44.0));
        // A 32 GB card with 2.76 GB free after the F16 caches -> 0.76 GB.
        assert_eq!(split_clone_budget_bytes(gb(2.76), gb(2.0)), gb(0.76));
        // Less than the slack: nothing. (Kills a mutant that subtracts any other
        // amount, or that adds a floor back: 1.5 - 2.0 must be 0, not 5.1 or 1.5.)
        assert_eq!(split_clone_budget_bytes(gb(1.5), gb(2.0)), 0);
        assert_eq!(split_clone_budget_bytes(0, gb(2.0)), 0);
        assert_eq!(split_clone_budget_bytes(gb(2.0), gb(2.0)), 0);
    }

    #[test]
    fn the_budget_selection_takes_a_positive_override_and_nothing_else() {
        let gb = |x: f64| (x * 1e9) as usize;
        // An explicit override is taken verbatim, uncapped, and reported as such.
        assert_eq!(
            resolve_split_clone_budget_bytes(gb(2.76), gb(2.0), Some("3")),
            (gb(3.0), true)
        );
        assert_eq!(
            resolve_split_clone_budget_bytes(gb(2.76), gb(2.0), Some(" 1.5 ")),
            (1_500_000_000, true)
        );
        // Everything that is not a finite positive number falls back to the
        // free-minus-slack default. (Kills a mutant that returns usize::MAX or
        // the raw free figure on the default path.)
        for raw in [
            None,
            Some(""),
            Some("0"),
            Some("-1"),
            Some("nan"),
            Some("inf"),
            Some("abc"),
        ] {
            assert_eq!(
                resolve_split_clone_budget_bytes(gb(2.76), gb(2.0), raw),
                (gb(0.76), false),
                "{raw:?}"
            );
        }
        assert_eq!(
            resolve_split_clone_budget_bytes(gb(1.5), gb(2.0), Some("x")),
            (0, false)
        );
    }

    #[test]
    fn a_clone_must_leave_the_slack_free() {
        let gb = |x: f64| (x * 1e9) as u64;
        // The 5090 case: 2.06 GB free, 1.35 GB clone, 2 GB slack -> refused.
        assert!(!clone_fits(gb(1.35), gb(2.06), gb(2.0)));
        // Exactly enough is enough.
        assert!(clone_fits(gb(1.35), gb(3.35), gb(2.0)));
        assert!(!clone_fits(gb(1.35), gb(3.34), gb(2.0)));
        // Overflow is a refusal, not a wrap.
        assert!(!clone_fits(u64::MAX, u64::MAX, 1));
    }

    #[test]
    fn output_proj_clone_is_skipped_unless_it_fits_or_is_forced() {
        let gb = |x: f64| (x * 1e9) as u64;
        let b = FreeMemory::Bytes;
        // The 5090 case: 2.06 GB free, 1.35 GB clone, 2 GB slack -> skipped, and the
        // reason names all three numbers.
        let d = output_proj_clone_decision(gb(1.35), b(gb(2.06)), gb(2.0), false);
        assert!(!d.proceed());
        let reason = d.skip_reason().expect("skipped clones give a reason");
        for needle in ["1.35 GB", "2.00 GB", "2.06 GB", F16_CACHE_FORCE_ENV] {
            assert!(reason.contains(needle), "missing {needle:?} in {reason}");
        }
        // Room enough: proceeds, no reason.
        let d = output_proj_clone_decision(gb(1.35), b(gb(3.35)), gb(2.0), false);
        assert!(d.proceed());
        assert_eq!(d.skip_reason(), None);
        // A failed memory query is no room, not a licence: skipped (fail-closed).
        assert!(
            !output_proj_clone_decision(gb(1.35), FreeMemory::Unknown, gb(2.0), false).proceed()
        );
        // The override proceeds regardless of either.
        assert!(output_proj_clone_decision(gb(1.35), b(0), gb(2.0), true).proceed());
        assert!(output_proj_clone_decision(gb(1.35), FreeMemory::Unknown, gb(2.0), true).proceed());
    }

    #[test]
    fn f16_cache_refusal_fits_when_needed_plus_headroom_is_free() {
        let b = FreeMemory::Bytes;
        assert_eq!(f16_cache_refusal(10, b(90), 80, false, 16, 2048, 1), None);
        assert!(f16_cache_refusal(10, b(90), 81, false, 16, 2048, 1).is_some());
    }

    #[test]
    fn f16_cache_refusal_accepts_the_t4_that_serves_with_half_a_gigabyte_left() {
        // Tesla T4, Qwen3.5-9B Q8_0, 8192-token context: 3.36 GB of caches into
        // 3.80 GB free, then a passing determinism and coherence run. The margin
        // must not turn that into a refusal (a 512 MiB reserve did), and it is
        // the measured sizing spread, not a reserve for what comes after.
        assert_eq!(F16_CACHE_HEADROOM_BYTES, 128 * 1024 * 1024);
        let gb = |x: f64| (x * 1e9) as u64;
        assert_eq!(
            f16_cache_refusal(
                gb(3.36),
                FreeMemory::Bytes(gb(3.80)),
                F16_CACHE_HEADROOM_BYTES,
                false,
                8,
                8192,
                gb(2.15),
            ),
            None
        );
        // And the margin is still a margin: caches that fit only by eating into
        // it are refused (this passes with no margin at all, so a zeroed constant
        // fails here).
        assert!(f16_cache_refusal(
            gb(3.80) - 64 * 1024 * 1024,
            FreeMemory::Bytes(gb(3.80)),
            F16_CACHE_HEADROOM_BYTES,
            false,
            8,
            8192,
            gb(2.15),
        )
        .is_some());
    }

    #[test]
    fn f16_cache_refusal_never_fires_for_nothing_to_build() {
        // BF16 / F16 models build no caches: no refusal even at zero free.
        assert_eq!(
            f16_cache_refusal(0, FreeMemory::Bytes(0), 1, false, 0, 8192, 0),
            None
        );
    }

    #[test]
    fn f16_cache_refusal_treats_a_failed_memory_query_as_unknown_not_zero() {
        assert_eq!(
            f16_cache_refusal(10, FreeMemory::Unknown, 1, false, 16, 8192, 0),
            None
        );
    }

    #[test]
    fn f16_cache_refusal_is_overridden_by_force() {
        assert_eq!(
            f16_cache_refusal(10, FreeMemory::Bytes(0), 1, true, 16, 8192, 0),
            None
        );
    }

    #[test]
    fn f16_cache_refusal_names_the_levers_and_the_numbers() {
        let msg = f16_cache_refusal(
            13_500_000_000,
            FreeMemory::Bytes(11_450_000_000),
            F16_CACHE_HEADROOM_BYTES,
            false,
            16,
            8192,
            4_290_000_000,
        )
        .expect("13.5 GB cannot fit in 11.45 GB");
        for needle in [
            "16 attention layers",
            "13.50 GB",
            "11.45 GB",
            "8192-token",
            "4.29 GB",
            "--context-len",
            F16_CACHE_FORCE_ENV,
        ] {
            assert!(msg.contains(needle), "missing {needle:?} in {msg}");
        }
    }

    #[test]
    fn f16_cache_refusal_does_not_overflow_on_huge_need() {
        assert!(f16_cache_refusal(
            u64::MAX - 1,
            FreeMemory::Bytes(u64::MAX - 1),
            2,
            false,
            1,
            1,
            0
        )
        .is_some());
    }

    #[test]
    fn only_measured_good_capabilities_keep_soa_locked_on() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q4_0);
        set_model_is_moe(false);
        set_device_cc_major(8);
        assert!(soa_locked_default(), "cc 8.x (A100) keeps it ON");
        set_device_cc_major(9);
        assert!(soa_locked_default(), "cc 9.x (H100) keeps it ON");
        for cc in [0u8, 7, 10, 11, 12, 13] {
            set_device_cc_major(cc);
            assert!(
                !soa_locked_default(),
                "cc {cc}.x is not measured-good (0 = query failed): OFF"
            );
        }
        reset_for_tests();
    }

    #[test]
    fn q4_dense_takes_split_k_only_on_blackwell() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_primary_quant(QuantScheme::Q4_0);
        set_model_is_moe(false);
        set_device_cc_major(12);
        assert!(
            attn_splitk_default(),
            "Q4_0 dense on cc 12.x: the pair is the measured route"
        );
        for cc in [0u8, 7, 8, 9, 10, 11, 13] {
            set_device_cc_major(cc);
            assert!(
                !attn_splitk_default(),
                "Q4_0 dense on cc {cc}.x is unmeasured: tiled"
            );
        }
        set_device_cc_major(12);
        set_model_is_moe(true);
        assert!(
            !attn_splitk_default(),
            "MoE never takes the pair by default"
        );
        set_model_is_moe(false);
        set_model_primary_quant(QuantScheme::Q8_0);
        set_device_cc_major(8);
        assert!(
            attn_splitk_default(),
            "Q8_0 dense keeps the pair on every capability"
        );
        reset_for_tests();
    }

    #[test]
    fn norm_dual_defaults_on_for_the_measured_cell_only() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_primary_quant(QuantScheme::Q4_0);
        set_model_is_moe(false);
        set_device_cc_major(12);
        assert!(
            norm_cta5_dual_default(),
            "Q4_0 dense on cc 12.x: measured +4.10 %"
        );
        for cc in [0u8, 8, 9, 10] {
            set_device_cc_major(cc);
            assert!(!norm_cta5_dual_default(), "cc {cc}.x is unmeasured: OFF");
        }
        set_device_cc_major(12);
        set_model_primary_quant(QuantScheme::Q8_0);
        assert!(!norm_cta5_dual_default(), "Q8_0 body is unmeasured: OFF");
        set_model_primary_quant(QuantScheme::Q4_0);
        set_model_is_moe(true);
        assert!(!norm_cta5_dual_default(), "MoE: OFF");
        reset_for_tests();
    }

    #[test]
    fn tiled_codegen_defaults_to_compute_120_only_for_the_measured_cell() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        set_model_primary_quant(QuantScheme::Q4_0);
        set_model_is_moe(false);
        assert_eq!(attn_tiled_codegen_default(12, true), "ptx120");
        assert_eq!(
            attn_tiled_codegen_default(12, false),
            "default",
            "NVRTC without the target"
        );
        for cc in [0u8, 8, 9, 10, 13] {
            assert_eq!(attn_tiled_codegen_default(cc, true), "default", "cc {cc}.x");
        }
        set_model_is_moe(true);
        assert_eq!(
            attn_tiled_codegen_default(12, true),
            "default",
            "MoE keeps NVRTC's default target"
        );
        set_model_is_moe(false);
        set_model_primary_quant(QuantScheme::Q8_0);
        assert_eq!(
            attn_tiled_codegen_default(12, true),
            "default",
            "Q8_0 body is unmeasured"
        );
        reset_for_tests();
    }

    #[test]
    fn legacy_defaults_switch_off_every_promoted_default() {
        // Through the explicit-input resolvers: the process-wide legacy cache cannot be toggled
        // inside one test process without leaking into its siblings.
        let q4 = Some(QuantScheme::Q4_0);
        assert!(attn_splitk_default_for(q4, false, 12, true));
        assert!(
            !attn_splitk_default_for(q4, false, 12, false),
            "legacy switch: split-K off"
        );
        assert!(norm_cta5_dual_default_for(q4, false, 12, true));
        assert!(
            !norm_cta5_dual_default_for(q4, false, 12, false),
            "legacy switch: dual norm off"
        );
        assert_eq!(
            attn_tiled_codegen_default_for(q4, false, 12, true, true),
            "ptx120"
        );
        assert_eq!(
            attn_tiled_codegen_default_for(q4, false, 12, true, false),
            "default",
            "legacy switch: default target"
        );
        assert!(
            !attn_splitk_default_for(Some(QuantScheme::Q8_0), false, 8, false),
            "legacy switch: Q8 pair off too"
        );
    }

    #[test]
    fn moe_disables_soa_locked_default() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        // regression guard: SOA_LOCKED MUST default OFF on MoE even though the
        // dense-quant hint is QUANTISED — the Q4 split clone pass populates only
        // dense siblings and would PAD-spam an MoE decode (same class as the
        // Q8_SPLIT MoE regression).
        set_model_dense_quant(QuantScheme::Q8_0);
        set_model_is_moe(true);
        // On a measured-good capability, so the MoE arm itself is what decides.
        set_device_cc_major(8);
        assert!(
            !soa_locked_default(),
            "MoE should NOT default SOA_LOCKED=ON (clone-pass / PAD-spam regression)"
        );
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
        assert!(
            q8_split_default(),
            "Dense Q8 must keep Q8_SPLIT=ON for the dense Q8 0.907× llama.cpp"
        );
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
        assert!(
            !gdn_f64_accum_default(),
            "dense Q8 must default GDN_F64_ACCUM OFF"
        );

        // Validated 2026-06-11: dense BF16 now defaults F64 ON —
        // the F32 GDN delta-rule decode recurrence accumulates ULP drift into
        // a repetition attractor on long generations; F64 heals it (measured
        // with the since-removed decode-graph path OFF).
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Bf16);
        assert!(
            gdn_f64_accum_default(),
            "dense BF16 must default GDN_F64_ACCUM ON (GAP-D)"
        );

        // MoE (set_model_is_moe(true)): ON for both q8 and bf16 — F64 on the
        // GDN single-token state update removes the decode-vs-prefill ULP
        // drift that triggered the q8 greedy restate-loop.
        reset_for_tests();
        set_model_dense_quant(QuantScheme::Q8_0);
        set_model_is_moe(true);
        assert!(
            gdn_f64_accum_default(),
            "MoE Q8 must default GDN_F64_ACCUM ON"
        );

        reset_for_tests();
        set_model_dense_quant(QuantScheme::Bf16);
        set_model_is_moe(true);
        assert!(
            gdn_f64_accum_default(),
            "MoE BF16 must default GDN_F64_ACCUM ON"
        );

        // Unset (no setters): OFF (dense default).
        reset_for_tests();
        assert!(
            !gdn_f64_accum_default(),
            "unset hint defaults GDN_F64_ACCUM OFF"
        );

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
            warnings.iter().any(|w| w.contains(canonical)),
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
                .any(|w| w.contains("'GDN_REGISTER_RESIDENT'")
                    && w.contains("LUMEN_CUDA_GDN_REGISTER_RESIDENT")),
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
        assert!(
            !resolve_enable_thinking(Some(false)),
            "per-request false beats env=1"
        );
        std::env::set_var("LUMEN_CHAT_ENABLE_THINKING", "0");
        assert!(
            resolve_enable_thinking(Some(true)),
            "per-request true beats env=0"
        );

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
            assert!(
                resolve_enable_thinking(None),
                "env '{truthy}' should enable"
            );
        }
        for falsy in ["0", "false", "no", "off", "OFF"] {
            std::env::set_var("LUMEN_CHAT_ENABLE_THINKING", falsy);
            assert!(
                !resolve_enable_thinking(None),
                "env '{falsy}' should disable"
            );
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

    #[test]
    fn attn_splitk_chunk_parse_warns_on_every_substituted_value() {
        // Unset and any value of one kernel tile or more are taken verbatim
        // and say nothing; every substitution (empty, sub-tile, negative,
        // float, garbage) resolves to the default and names the variable.
        for (raw, want) in [
            (None, ATTN_SPLITK_CHUNK_POSITIONS),
            (Some("128"), 128),
            (Some(" 256 "), 256),
            (Some("4294967295"), u32::MAX),
        ] {
            assert_eq!(parse_attn_splitk_chunk(raw), (want, None), "raw={raw:?}");
        }
        for raw in ["", "   ", "0", "1", "127", "-1", "12.5", "abc", "1e9"] {
            let (chunk, warning) = parse_attn_splitk_chunk(Some(raw));
            assert_eq!(chunk, ATTN_SPLITK_CHUNK_POSITIONS, "raw={raw:?}");
            let warning = warning.unwrap_or_else(|| panic!("raw={raw:?} must warn"));
            assert!(
                warning.contains("LUMEN_CUDA_ATTN_SPLITK_CHUNK"),
                "{warning}"
            );
        }
    }

    #[test]
    fn attn_splitk_chunk_default_is_one_kernel_tile() {
        assert_eq!(ATTN_SPLITK_CHUNK_POSITIONS, 128);
    }

    // ---- F1 + F2: allowlist membership (no false unknown-env typo warning) ----

    #[test]
    fn ct4_exactk_parses_strict_one_only() {
        // LUMEN_CUDA_CT4_EXACTK is default ON with a strict `=0` escape hatch:
        // only an explicit `0` (whitespace-tolerant) disables the shipping
        // exact-K route; unset, empty, and any other value stay ON.
        use std::env::VarError;
        for (raw, want) in [
            (Err(VarError::NotPresent), true),
            (Ok(String::new()), true),
            (Ok("0".into()), false),
            (Ok(" 0 ".into()), false),
            (Ok("off".into()), true),
            (Ok("false".into()), true),
            (Ok("1".into()), true),
            (Ok(" 1 ".into()), true),
        ] {
            assert_eq!(parse_ct4_exactk(raw.clone()), want, "raw={raw:?}");
        }
    }

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

    /// Every `LUMEN_*` token the repository's scripts, packaging and CI
    /// define, read from the checkout itself. `None` when the crate is built
    /// outside a checkout (a packaged crate has no `scripts/`), and the test
    /// that uses it skips then rather than pass vacuously.
    fn lumen_names_defined_by_the_scripts() -> Option<std::collections::BTreeSet<String>> {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let roots = ["scripts", "packaging", ".github", "bench"];
        if !roots.iter().all(|r| repo.join(r).is_dir()) {
            return None;
        }
        let mut names = std::collections::BTreeSet::new();
        let mut stack: Vec<std::path::PathBuf> = roots.iter().map(|r| repo.join(r)).collect();
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("readable repository directory") {
                let path = entry.expect("directory entry").path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                // Scripts, workflow files, Dockerfiles and templates only: a
                // markdown file may quote a name that no longer exists.
                let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                let is_script = file_name == "Dockerfile"
                    || matches!(
                        path.extension().and_then(|e| e.to_str()),
                        Some("sh" | "py" | "yml" | "yaml" | "toml" | "in")
                    );
                if !is_script {
                    continue;
                }
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                let bytes = text.as_bytes();
                let mut i = 0;
                while let Some(off) = text[i..].find("LUMEN_") {
                    let start = i + off;
                    let preceded_by_name_char = start > 0
                        && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_');
                    let mut end = start;
                    while end < bytes.len()
                        && (bytes[end].is_ascii_uppercase()
                            || bytes[end].is_ascii_digit()
                            || bytes[end] == b'_')
                    {
                        end += 1;
                    }
                    // A token written as a prefix (`LUMEN_METAL_DET_*`) names a
                    // family, not a variable.
                    if !preceded_by_name_char && !text[start..end].ends_with('_') {
                        names.insert(text[start..end].to_string());
                    }
                    i = end;
                }
            }
        }
        Some(names)
    }

    #[test]
    fn both_allowlists_are_sorted_and_disjoint() {
        for (list, name) in [
            (KNOWN_LUMEN_ENV_VARS, "KNOWN_LUMEN_ENV_VARS"),
            (KNOWN_LUMEN_TOOLING_ENV_VARS, "KNOWN_LUMEN_TOOLING_ENV_VARS"),
        ] {
            let unsorted: Vec<&[&str]> = list.windows(2).filter(|w| w[0] >= w[1]).collect();
            assert!(
                unsorted.is_empty(),
                "{name} is out of order at {unsorted:?}"
            );
        }
        let engine: std::collections::BTreeSet<&str> =
            KNOWN_LUMEN_ENV_VARS.iter().copied().collect();
        let shared: Vec<&&str> = KNOWN_LUMEN_TOOLING_ENV_VARS
            .iter()
            .filter(|n| engine.contains(*n))
            .collect();
        assert!(shared.is_empty(), "names on both lists: {shared:?}");
    }

    #[test]
    fn tooling_names_are_exactly_the_ones_the_scripts_define() {
        let Some(defined) = lumen_names_defined_by_the_scripts() else {
            eprintln!("skipped: not built inside a checkout");
            return;
        };
        let engine: std::collections::BTreeSet<&str> =
            KNOWN_LUMEN_ENV_VARS.iter().copied().collect();
        let tooling: std::collections::BTreeSet<&str> =
            KNOWN_LUMEN_TOOLING_ENV_VARS.iter().copied().collect();
        // A script-defined name the engine does not read must be on the tooling
        // list, or every run under that script warns about a name that is not
        // a typo.
        let missing: Vec<&String> = defined
            .iter()
            .filter(|n| !engine.contains(n.as_str()) && !tooling.contains(n.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "script-defined names the validator would flag: {missing:?}"
        );
        // And the tooling list may not outlive the scripts that justify it.
        let stale: Vec<&&str> = tooling.iter().filter(|n| !defined.contains(**n)).collect();
        assert!(
            stale.is_empty(),
            "tooling names no script defines any more: {stale:?}"
        );
    }

    #[test]
    fn tooling_names_do_not_warn_when_set() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        for name in [
            "LUMEN_BIN",
            "LUMEN_SERVER_BIN",
            "LUMEN_DET_MODEL",
            "LUMEN_QS_MODEL",
        ] {
            let saved = std::env::var(name).ok();
            std::env::set_var(name, "1");
            let warnings = collect_unknown_lumen_env_vars();
            match saved {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
            assert!(
                !warnings.iter().any(|w| w.contains(name)),
                "tooling name '{name}' must not warn; warnings = {warnings:?}"
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

    // ---- Reverse coverage: every READ env var is in the allowlist ----

    /// Every `LUMEN_*` env var **read** at runtime in `crates/` (via
    /// `std::env::var` / the `env_*` helpers). The two tests above prove the
    /// forward direction (`allowlist ⇒ no-warn`); this static proves the
    /// *reverse* — `reads ⊆ allowlist` — which is the direction that actually
    /// prevents the startup false-warn defect: a flag that is read but NOT
    /// allowlisted makes `validate_lumen_env_vars()` emit a spurious
    /// "unknown LUMEN var — typo?" warning the moment an operator sets it.
    ///
    /// REGENERATE (from repo root) with the campaign one-liner:
    ///   grep -rhoE '"LUMEN_[A-Z0-9_]+"' crates --include='*.rs' | tr -d '"' | sort -u
    /// then drop `LUMEN_BUILD_VERSION` — it is a compile-time `option_env!`
    /// baked in at build time, never present in the runtime process env, so it
    /// is intentionally NOT a runtime allowlist member.
    static READ_SITE_LUMEN_ENV_VARS: &[&str] = &[
        "LUMEN_AB_ITERATIONS",
        "LUMEN_AB_WARMUP",
        "LUMEN_ANTI_RESTATE",
        "LUMEN_ANTI_RESTATE_LOOP",
        "LUMEN_ANTI_RESTATE_NGRAM",
        "LUMEN_ANTI_RESTATE_SUBWORD",
        "LUMEN_BASE_URL",
        "LUMEN_BENCH_ITERATIONS",
        "LUMEN_BENCH_MASK_EOG",
        "LUMEN_BENCH_SCALE",
        "LUMEN_BENCH_TOKENS",
        "LUMEN_BENCH_TOKEN_IDS",
        "LUMEN_BENCH_TOP2",
        "LUMEN_BENCH_WARMUP",
        "LUMEN_CACHE_DIR",
        "LUMEN_CHAT_ENABLE_THINKING",
        "LUMEN_CORR010_MODEL",
        "LUMEN_CUDA_ARGMAX_TILED",
        "LUMEN_CUDA_ATTN_BANK3",
        "LUMEN_CUDA_ATTN_PREFILL_SGEMM",
        "LUMEN_CUDA_ATTN_PREP_FUSE",
        "LUMEN_CUDA_ATTN_SPLITK",
        "LUMEN_CUDA_ATTN_SPLITK_CHUNK",
        "LUMEN_CUDA_ATTN_SPLITK_GQA6",
        "LUMEN_CUDA_ATTN_SPLITK_SCALE",
        "LUMEN_CUDA_ATTN_TILED_CODEGEN",
        "LUMEN_CUDA_F16_CACHE",
        "LUMEN_CUDA_F16_CACHE_FORCE",
        "LUMEN_CUDA_FFN_DIRECT_RESIDUAL",
        "LUMEN_CUDA_FFN_GATE_UP_BANK",
        "LUMEN_CUDA_Q4_DOWN_NR1",
        "LUMEN_CUDA_BF16_AB_Q8BANK",
        "LUMEN_CUDA_BF16_AUTOTUNE",
        "LUMEN_CUDA_BF16_FUSED_GLU",
        "LUMEN_CUDA_BF16_GEMMEX",
        "LUMEN_CUDA_BF16_MATVEC",
        "LUMEN_CUDA_BF16_MOE_V3",
        "LUMEN_CUDA_BF16_NR1",
        "LUMEN_CUDA_BF16_WO_NR1",
        "LUMEN_CUDA_CT4_DP4A",
        "LUMEN_CUDA_CT4_EXACTK",
        "LUMEN_CUDA_DECODE_DELAY_US",
        "LUMEN_CUDA_DECODE_TILED",
        "LUMEN_CUDA_DECODE_TILED_THRESHOLD",
        "LUMEN_CUDA_FFN_FUSED_GLU",
        "LUMEN_CUDA_FORCE_SCALAR_ATTN",
        "LUMEN_CUDA_GDN_AB_F16",
        "LUMEN_CUDA_GDN_AB_F32",
        "LUMEN_CUDA_GDN_CONVSTATE_PARITY",
        "LUMEN_CUDA_GDN_DECODE_MEGAKERNEL_F64",
        "LUMEN_CUDA_GDN_DECODE_VIA_PREFILL",
        "LUMEN_CUDA_GDN_F64_ACCUM",
        "LUMEN_CUDA_GDN_NG_Q8",
        "LUMEN_CUDA_GDN_P123_FUSE",
        "LUMEN_CUDA_GDN_PREFILL_F64",
        "LUMEN_CUDA_GDN_REGISTER_RESIDENT",
        "LUMEN_CUDA_GDN_SKIP_DUP_QKV",
        "LUMEN_CUDA_GDN_SUBSTAGE_TIMING",
        "LUMEN_CUDA_GPU_SAMPLE",
        "LUMEN_CUDA_LEGACY_DEFAULTS",
        "LUMEN_CUDA_MAX_SEQ_LEN",
        "LUMEN_CUDA_MMV_BF16_OUTPUT_PROJ",
        "LUMEN_CUDA_MMV_Q_DP4A",
        "LUMEN_CUDA_MMV_Q_MOE_DP4A",
        "LUMEN_CUDA_MMV_Q_OUTPUT_PROJ",
        "LUMEN_CUDA_MOE_BATCHED",
        "LUMEN_CUDA_MOE_BATCHED_V2",
        "LUMEN_CUDA_MOE_BATCHED_V3",
        "LUMEN_CUDA_MOE_BF16_NATIVE",
        "LUMEN_CUDA_MOE_DECODE_F32",
        "LUMEN_CUDA_MOE_DECODE_F32_FFN",
        "LUMEN_CUDA_MOE_DOWN_TILED_F32ACT",
        "LUMEN_CUDA_MOE_FUSED_NORM_ROUTER",
        "LUMEN_CUDA_MOE_GATE_UP_W10",
        "LUMEN_CUDA_MOE_GROUPED_TILED",
        "LUMEN_CUDA_MOE_PREFILL_BATCHED",
        "LUMEN_CUDA_MOE_Q4_V3",
        "LUMEN_CUDA_MOE_Q4_V3B",
        "LUMEN_CUDA_MOE_RESIDUAL_Q8",
        "LUMEN_CUDA_MOE_ROUTER_PARALLEL",
        "LUMEN_CUDA_NORM_CTA5_DUAL",
        "LUMEN_CUDA_OUTPUT_PROJ_NR",
        "LUMEN_CUDA_OUTPUT_PROJ_SPLIT",
        "LUMEN_CUDA_PREFILL_F32",
        "LUMEN_CUDA_PROFILE",
        "LUMEN_CUDA_PROFILE_ATTN_LEAF",
        "LUMEN_CUDA_PTX_CACHE",
        "LUMEN_CUDA_PTX_CACHE_DIR",
        "LUMEN_CUDA_Q4_B160",
        "LUMEN_CUDA_Q4_F32ACT_KERNEL",
        "LUMEN_CUDA_Q4_MMVQ",
        "LUMEN_CUDA_Q4_PROJ_BANK",
        "LUMEN_CUDA_Q4_SPLIT",
        "LUMEN_CUDA_Q4_SPLIT_ATTN",
        "LUMEN_CUDA_Q4_SPLIT_BUDGET_GB",
        "LUMEN_CUDA_Q4_SPLIT_WO",
        "LUMEN_CUDA_Q4_V4LOAD",
        "LUMEN_CUDA_Q8_AB_BANK",
        "LUMEN_CUDA_Q8_MATVEC_FAST",
        "LUMEN_CUDA_Q8_MMVQ",
        "LUMEN_CUDA_Q8_PROJ_MMQ",
        "LUMEN_CUDA_Q8_SCALE_HW",
        "LUMEN_CUDA_Q8_SPLIT",
        "LUMEN_CUDA_Q8_SPLIT_ATTN",
        "LUMEN_CUDA_Q8_SPLIT_BUDGET_GB",
        "LUMEN_CUDA_Q8_SPLIT_SSMOUT",
        "LUMEN_CUDA_Q8_SPLIT_WO",
        "LUMEN_CUDA_ROPE_TAB",
        "LUMEN_CUDA_SHARED_FUSED_DECODE",
        "LUMEN_CUDA_SHARED_TILED",
        "LUMEN_CUDA_SKIP_BF16_PROBE",
        "LUMEN_CUDA_SOA_LOCKED",
        "LUMEN_CUDA_SSMOUT_RESID_FOLD",
        "LUMEN_CUDA_TOPK_MOE_FUSED",
        "LUMEN_CUDA_VERBOSE",
        "LUMEN_DUMP_EXPERTS",
        "LUMEN_DUMP_GDN_L0_BIN",
        "LUMEN_DUMP_NORMED",
        "LUMEN_FREQUENCY_PENALTY",
        "LUMEN_KV_PRECISION",
        "LUMEN_METAL_ATTN_PRECISE",
        "LUMEN_METAL_BF16_GATE_UP_NR",
        "LUMEN_METAL_BF16_GDN_FULL_PREFILL_WARMUP",
        "LUMEN_METAL_BF16_GDN_QKV_GATE_PAIRED",
        "LUMEN_METAL_BF16_MMAP_ONLY",
        "LUMEN_METAL_CB_SPLIT",
        "LUMEN_METAL_CONCURRENT_ENCODER",
        "LUMEN_METAL_CONCURRENT_ENCODER_VALIDATE",
        "LUMEN_METAL_DECODE_DELAY_US",
        "LUMEN_METAL_DECODE_GPUTIME",
        "LUMEN_METAL_DECODE_PROFILE",
        "LUMEN_METAL_DEFAULTS_OFF",
        "LUMEN_METAL_FFN_DOWN_SPLITK",
        "LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED",
        "LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_BF16",
        "LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_Q4",
        "LUMEN_METAL_GDN_CONCURRENT_ENCODER",
        "LUMEN_METAL_GDN_CONCURRENT_ENCODER_VALIDATE",
        "LUMEN_METAL_GDN_SSM_OUT_F32_BATCHED",
        "LUMEN_METAL_GPU_SAMPLER",
        "LUMEN_METAL_GPU_SAMPLER_EXACT",
        "LUMEN_METAL_GPU_SAMPLER_QUIET",
        "LUMEN_METAL_MMAP_ONLY",
        "LUMEN_METAL_MOE_GATHER_VEC4",
        "LUMEN_METAL_MOE_GEMM_TILEMAP",
        "LUMEN_METAL_MOE_PREFILL_GROUPED",
        "LUMEN_METAL_MOE_ROUTER_PARALLEL",
        "LUMEN_METAL_MOE_ROUTER_TOPK_TGS",
        "LUMEN_METAL_MOE_ROUTE_SORT",
        "LUMEN_METAL_MOE_ROUTE_SORT_PAR",
        "LUMEN_METAL_NAN_DUMP",
        "LUMEN_METAL_PREFILL_GPUTIME",
        "LUMEN_METAL_PROFILE",
        "LUMEN_METAL_Q8_GDN_QKVGATE_2STREAM",
        "LUMEN_METAL_Q8_REPACKED",
        "LUMEN_METAL_Q8_REPACKED_FFN_DOWN",
        "LUMEN_METAL_Q8_REPACKED_GATE_UP",
        "LUMEN_METAL_UNRETAINED_CMDBUFS",
        "LUMEN_MOE_PROBE",
        "LUMEN_PREFILL_TIMING",
        "LUMEN_QWEN35_9B_BF16",
        "LUMEN_QWEN35_9B_PATH",
        "LUMEN_QWEN35_9B_Q4",
        "LUMEN_QWEN35_9B_Q8",
        "LUMEN_REPEAT_LAST_N",
        "LUMEN_REPETITION_PENALTY",
        "LUMEN_SERVER_DEBUG_MEM",
        "LUMEN_SERVER_PANIC_MAX",
        "LUMEN_SERVER_PANIC_WINDOW_SECS",
        "LUMEN_SOAK_DURATION_SEC",
        "LUMEN_SOAK_OUT_DIR",
        "LUMEN_SOAK_STACK_DUMP",
        "LUMEN_SOAK_STACK_LEAKS",
        "LUMEN_SOAK_STACK_TICKS",
        "LUMEN_SOAK_WARMUP_SEC",
        "LUMEN_SPEC_DUMP_IDS",
        "LUMEN_SUFFIX_THRESHOLD",
        "LUMEN_TEST_OPENAI_SDK",
        "LUMEN_XCHK",
        "LUMEN_XCHK2",
    ];

    #[test]
    fn all_read_env_vars_are_registered() {
        // Reverse-registry invariant: `reads ⊆ KNOWN_LUMEN_ENV_VARS`. If this
        // fails, a newly-added env read is missing from the allowlist — add it
        // to KNOWN_LUMEN_ENV_VARS (or remove the read). Regenerate the array
        // above with the one-liner in its doc comment.
        for name in READ_SITE_LUMEN_ENV_VARS {
            assert!(
                KNOWN_LUMEN_ENV_VARS.contains(name),
                "read-but-unregistered LUMEN env var '{name}': it is read in \
                 crates/ but absent from KNOWN_LUMEN_ENV_VARS, so it would \
                 false-warn at startup. Add it to the allowlist."
            );
        }
    }
}

/// `LUMEN_CUDA_FFN_DIRECT_RESIDUAL` (default ON; `=0` opts out): the
/// FFN down projection folds its residual into its own store and writes
/// `x_gpu` directly, eliding both the `residual_add` launch and the decode
/// loop's layer-commit D2D copy (2 commands x 64 layers per token). On the
/// validated Q4/Q8 split routes the residual add is the same explicitly
/// pinned single `add.rn.f32` the separate launch performs, so output bytes
/// are unchanged; routes without an eligible residual sibling keep the
/// separate tail.
pub fn ffn_direct_residual() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_FFN_DIRECT_RESIDUAL") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// `LUMEN_CUDA_FFN_GATE_UP_BANK` (default ON; `=0` opts out): FFN
/// gate and up projections issue as ONE banked launch off the shared Q8_1
/// input (baseline 256-thread kernel; B160/V4 variants are GDN-only wins and
/// measured FFN regressions). Bit-identical per row to the two-launch
/// route.
pub fn ffn_gate_up_bank() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_FFN_GATE_UP_BANK") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// `LUMEN_CUDA_Q4_DOWN_NR1` (default ON; `=0` opts out): the FFN down
/// projection's locked Q4 split matvec runs one row per CTA (grid = out_dim)
/// instead of four. Byte-identical per row — the grid mapping is the only
/// change; the short-N long-K down shape underfills the NR=4 grid.
pub fn q4_down_nr1() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("LUMEN_CUDA_Q4_DOWN_NR1") {
        Ok(v) => v != "0",
        Err(_) => canonical_default_on(),
    })
}

/// The one place the CUDA decode matvec route line is spelled.
///
/// It lives here rather than beside the dispatches it describes because it is a
/// pure string function and the `cuda` module is feature-gated: the format is a
/// contract with whatever reads a run's log back, and a contract whose test only
/// compiles under `--features cuda` is one the default test run never checks.
///
/// The shape: a kernel name, then `: ACTIVE`, then the site the route was first
/// taken at and the matvec's dimensions. Nothing else on the line, and no
/// wording that reads as a route the run declined.
pub fn matvec_route_line(kernel: &str, label: &str, out_dim: usize, in_dim: usize) -> String {
    format!("[CUDA] {kernel}: ACTIVE (first at {label}, out={out_dim}, in={in_dim})")
}

#[cfg(test)]
mod matvec_route_line_tests {
    //! The route line is read by a parser outside this repo: it accepts
    //! `[CUDA] <kernel>: ACTIVE` for a kernel token from a known family and
    //! rejects the line outright on any of the loader's own qualifiers.

    use super::matvec_route_line;

    #[test]
    fn names_the_kernel_the_site_and_the_shape() {
        assert_eq!(
            matvec_route_line("matvec_q4_split_q8_1", "gate", 17408, 5120),
            "[CUDA] matvec_q4_split_q8_1: ACTIVE (first at gate, out=17408, in=5120)"
        );
    }

    #[test]
    fn carries_no_qualifier_that_voids_the_line() {
        let line = matvec_route_line("matvec_q5k_split_q8_1_residual", "gdn_ssm_out", 5120, 4096);
        assert!(line.starts_with("[CUDA] matvec_"), "{line}");
        for voided in [
            " set but ",
            " unrecognized",
            " defaults OFF",
            " clone skipped: ",
        ] {
            assert!(!line.contains(voided), "{line} carries {voided:?}");
        }
    }
}
