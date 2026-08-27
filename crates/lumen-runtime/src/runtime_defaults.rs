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

const HINT_UNSET: u8 = 0;
const HINT_BF16: u8 = 1;
const HINT_QUANTISED: u8 = 2;

/// Stores the EXACT **primary / bulk** model `QuantScheme` (the body
/// attention+FFN weight scheme, `lbc.header.quantization.scheme`), via
/// `QuantScheme::to_u8` (range 0..=12), so resolvers that must distinguish
/// *within* the "quantised" bucket (e.g. Q4_0-vs-Q8_0 attention-precision
/// tuning) can do so.
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
/// which is what `attn_precise_default()` reads to distinguish Q4_0 from Q8_0
/// within the 27B (64-layer) dense class. This is a SEPARATE signal from
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
/// caller / no LBC opened). Used by `attn_precise_default()` to distinguish
/// Q4_0 from Q8_0 within the 27B (64-layer) dense class. One relaxed atomic
/// load + a `from_u8` decode.
pub(crate) fn model_dense_quant() -> Option<QuantScheme> {
    let tag = MODEL_PRIMARY_QUANT_SCHEME.load(Ordering::Relaxed);
    if tag == QUANT_SCHEME_UNSET {
        None
    } else {
        QuantScheme::from_u8(tag).ok()
    }
}

/// Public diagnostic wrapper over [`model_dense_quant`] for the
/// `dump_quant_hint` example (which lives outside the crate and so cannot see
/// the `pub(crate)` accessor). Behaviourally identical; not used on any hot
/// path.
pub fn model_dense_quant_pub() -> Option<QuantScheme> {
    model_dense_quant()
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
/// precision). Mode map for the batch-≥16 WMMA full-attention prefill kernel:
/// `0` = legacy F16 WMMA (both QK^T and P@V rounded to F16 operands);
/// `1` = qkf32 (exact-F32 QK^T, F16 P@V); `2` = pvf32 (F16 QK^T, exact-F32
/// P@V); `3` = scalar (exact-F32 QK^T **and** exact-F32 P@V — the F32 Br=4
/// scalar kernel, both dispatch sites route mode 3 there); `4` = split
/// (hi/lo tensor-core approximation, unqualified).
///
/// **Ratified default (2026-07-22): `3` (scalar) for every supported
/// production class.** codex-sol RCA (F32-golden L3 trace) proved the
/// batch-≥16 WMMA prefill has TWO independent precision carriers, and the
/// prior pvf32 default only closed one of them:
///
///   * **P@V F16 carrier** — closed by AP=2 (pvf32). This is the GQ-014
///     multi-turn heal documented below; exact-F32 P@V is *required* and must
///     be preserved.
///   * **QK^T F16 carrier** — closed only by exact-F32 QK^T. The F16 QK^T
///     score matmul rounds Q/K operands to half BEFORE the (already-F32)
///     accumulator, discarding mantissa bits that flip score ordering. On the
///     cuda/9B/Q8_0 golden this flips **case-08** (Lumen emits `35`; the
///     HF-F32 golden and llama.cpp `-fa 0` both emit the correct `28`). The
///     L3 attention distribution — not global hidden-state L2 — is the
///     decisive signal: under the F16-QK default the L3 top-token/mass
///     diverges from the golden (~0.193 trace delta); AP=3 collapses it.
///
/// Why `3` and not `1` (qkf32): AP=1 closes the QK^T carrier but REOPENS the
/// F16 P@V hole that AP=2 was introduced to fix, so it regresses GQ-014 (the
/// RCA lever bisect below already recorded "AP=1 QK^T-only does NOT heal"
/// GQ-014). AP=3 makes BOTH matmuls exact-F32, so it is the only existing
/// mode structurally guaranteed to satisfy both carriers simultaneously. The
/// cost is prefill-only (decode uses a separate F32 tiled kernel); the ≥0.95
/// perf board is re-measured against this corrected default, not the retired
/// F16-QK numbers.
///
/// Scope: the QK^T defect is weight-quant-independent — it is intrinsic to the
/// F16 operand rounding in the WMMA score matmul — so the default broadens to
/// EVERY class routed through this kernel (9B, 27B, MoE × Q4/Q8/BF16), not
/// just 9B-Q8. Legacy callers that never set the block count still get
/// conservative WMMA (`0`).
///
/// ---
/// **Historical RCA (retained — this is WHY exact-F32 P@V must be preserved,
/// i.e. why the default is AP=3 and not AP=1):**
///
/// Validated 2026-06-11 (N=3 byte-deterministic quality runs per cell):
/// pvf32 for MoE (all quants) + dense ≤32-layer (9B class) — strict wins
/// everywhere (9B-q8 and MoE cells all reach pristine quality gates; MoE-q8
/// long-form heals fully); default WMMA for the 27B class (64 layers) —
/// pvf32 measurably regressed 27B long-form output on bf16/q8 (bf16 verylong
/// 3/3→1/3, q8 shorts softened) when the default keyed purely on layer count.
///
/// **2026-06-12 (GQ-014 multi-turn fidelity fix):**
/// the 27B layer-count carve-out was too coarse — it lumped all 27B quants
/// onto legacy all-F16 WMMA, but that path FAILS the multi-turn gate (GQ-014)
/// on the quantised 27B cells: an F16-WMMA near-tie flip early in a longer
/// (multi-turn) prefill derails the conversation (27b-q4 4/8, 27b-q8 6/8).
/// The exact-F32 P@V (AP=2) was decided empirically per quant (reference GPU,
/// N≥3, runtime evidence; full validation matrix). AP=3 inherits this exact
/// P@V unchanged and adds exact QK^T on top:
///
///   * 27B (64-layer) dense **Q4_0**: pvf32 heals GQ-014
///     4/8→8/8 with ZERO single-prompt regression (15/15·7/8·3/3 →
///     15/15·8/8·3/3, GQ-002 even improves +1), N=3 cross-process
///     byte-identical. Re-confirmed here. **THE FIX.**
///   * 27B (64-layer) dense **Q8_0**. **2026-06-12
///     re-classification**: the prior
///     branch excluded q8 because pvf32 appeared to regress GQ-001
///     (short-arith-05) + GQ-004 (vlong-explain-01) DD-REP. Further analysis
///     PROVED both are DETECTOR FALSE-POSITIVES on gold-standard outputs
///     (full-text: short-arith-05 = `963` correct, finish=stop, the DD-REP
///     fires on 3 CORRECT borrowing lines where the n=4 window clips one token
///     before the diverging content word; vlong-explain-01 = coherent
///     finish=stop, word-granularity dd_rep PASS 0.991). Cross-backend
///     corroboration: Metal 27b-q8 fires the SAME short-arith-05 DD-REP while
///     passing GQ-014 8/8 → detector-sensitivity, not a CUDA defect. The lever
///     bisect on the minimal failing prefix isolates the carrier as
///     prefill-attention P@V F16 mantissa (AP=1 QK^T-only does NOT heal, AP=2
///     P@V-exact DOES). With the harness-only detector calibration that lands
///     alongside this change, q8 → 15/15·8/8·3/3 + GQ-014 8/8. **THE FIX (q8).**
///   * 27B (64-layer) dense **BF16**, **paired with via-prefill
///     ON** (see `gdn_decode_via_prefill_default`, whose 27B-bf16 carve-out is
///     removed alongside this). bf16 has TWO coupled carriers: prefill
///     attention P@V F16 (healed by AP=2) + GDN decode-recurrence per-step
///     drift over long generations (healed by via-prefill ON). The prior
///     branch's "AP=2 wrecks bf16 verylong 3/3→1/3" was part detector-FP, part
///     a GENUINE long-form stutter (`Moka, Moka, …` ×20) that via-prefill ON
///     eliminates. AP=2 ALONE leaves the stutter; via-prefill ALONE leaves the
///     t3 prefill-attention near-tie; ONLY the COMBINATION heals every
///     symptom — exactly the validated 9b-bf16 stack. With the full stack:
///     GQ-014 4/8→8/8 N=3, GQ-004 3/3 (stutter gone). One honest cost: GQ-002
///     8/8→7/8 (med-reason-02, a verbosity truncation near-tie — 391 still
///     computed; validated against the LC reference-hardness check).
///     **THE FIX (bf16).**
///   * MoE (any quant) + dense ≤32-layer (9B): exact-F32 P@V, unchanged.
///
/// Unset block count (0, legacy callers) → conservative legacy WMMA.
/// CUDA-only: the sole consumers are the two `flash_attention_wmma_*` dispatch
/// sites in `cuda/backend_impl.rs` (both `#[cfg(feature = "cuda")]`). Metal
/// reads its OWN env (`LUMEN_METAL_ATTN_PRECISE`) in `metal/prefill_encode.rs`
/// and never calls this function, so this default is a no-op on the Metal/CPU
/// build. `LUMEN_CUDA_ATTN_PRECISE=<0|1|2|3|4>` overrides either way.
pub fn attn_precise_default() -> u8 {
    let layers = model_block_count();
    // MoE (any quant) + dense 9B class (≤32 layers): ratified AP=3 (scalar).
    // AP=3 = exact-F32 QK^T AND exact-F32 P@V. It keeps the exact P@V that
    // heals GQ-014 (see below) and ADDS exact QK^T to close the F16-QK score
    // carrier that flipped case-08 (cuda/9B/Q8_0: F16-QK emits 35, golden 28).
    if model_is_moe() || (layers > 0 && layers <= 32) {
        return 3;
    }
    // 27B (64-layer) dense class: AP=3 (scalar). Exact P@V heals the GQ-014
    // multi-turn F16-WMMA near-tie flip (the per-QUANT bisect below), and the
    // added exact QK^T closes the quant-independent F16-QK score carrier. The
    // 2026-06-12 follow-up already proved exact P@V is correct for ALL three
    // 27B quants; AP=3 preserves it and layers exact QK^T on top:
    //   * Q4_0 → 3 (scalar). Exact P@V re-confirmed N=3: GQ-014 4/8→8/8 with
    //     ZERO single-prompt regression. AP=3 adds exact QK^T (no P@V change).
    //   * Q8_0 → 3 (scalar). The P@V carrier is prefill-attention P@V F16
    //     mantissa (lever bisect: AP=1 QK^T-only does NOT heal GQ-014, AP=2
    //     P@V-exact DOES; GQ-014 6/8→8/8 N=3) — AP=3 keeps that exact P@V and
    //     ALSO makes QK^T exact. The two prior "regressions" that excluded q8
    //     (GQ-001 short-arith-05 + GQ-004 vlong-explain-01 DD-REP) were proven
    //     DETECTOR FALSE-POSITIVES on gold-standard outputs (full-text: 963
    //     correct, finish=stop; coherent DNS explanation) — fixed by the
    //     harness-only detector calibration that lands with this change.
    //   * Bf16 → 3 (scalar) AND `gdn_decode_via_prefill_default` carved back IN
    //     for the 27B class (see that fn). bf16 has TWO coupled carriers:
    //     prefill-attention P@V F16 (healed by exact P@V) + GDN decode-recurrence
    //     per-step drift over long gens (healed by via-prefill ON). ONLY the
    //     combination heals every symptom — exactly the validated 9b-bf16 stack
    //     (exact P@V + via-prefill, both ON). GQ-014 4/8→8/8 N=3, GQ-004 Moka
    //     stutter eliminated. The earlier "AP=2 wrecks bf16 verylong 3/3→1/3"
    //     was part detector-FP, part a genuine stutter that via-prefill ON
    //     removes; with the FULL stack GQ-004 is 3/3.
    // NOTE: AP=3 is NOT AP=1 (qkf32). AP=1 would close QK^T but REOPEN the F16
    // P@V hole above and regress GQ-014; only AP=3 satisfies both carriers.
    // Other quants / unset → conservative WMMA.
    if layers > 32 {
        match model_dense_quant() {
            Some(QuantScheme::Q4_0) | Some(QuantScheme::Q8_0) | Some(QuantScheme::Bf16) => {
                return 3;
            }
            _ => {}
        }
    }
    0
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
/// **DENSE BF16 ≥33-layer NOW ON (2026-06-12 follow-up analysis, supersedes the
/// 2026-06-10 ablation):** the old ablation found via-prefill-alone CORRUPTS
/// dense bf16 (9b-bf16 4/15·0/8·0/3, 27b-bf16 0✓/3✗) — but that was measured
/// "alone" on the PRE-pvf32 binary. The follow-up analysis proved via-prefill must be
/// paired with AP=2 for the 27B-bf16 class: AP=2 alone leaves a genuine
/// long-form Moka stutter; via-prefill alone leaves the prefill-attention
/// near-tie; the COMBINATION (mirroring the validated 9b-bf16 stack) heals
/// GQ-014 4/8→8/8 (N=3) AND restores GQ-004 verylong to 3/3. The 27B-bf16
/// carve-out is removed accordingly (see the resolver body). 9B-bf16 was
/// already ON via the ≤32 carve-back and is byte-unchanged.
/// Set `LUMEN_CUDA_GDN_DECODE_VIA_PREFILL=0|1` to override either way.
pub fn gdn_decode_via_prefill_default() -> bool {
    // 27B-bf16 carve-OUT REMOVED (2026-06-12 follow-up analysis).
    // via-prefill is now ON for ALL
    // classes — MoE, every dense quant, and dense bf16 at any layer count.
    //
    // History of this predicate:
    //   * MoE + dense non-bf16: always ON (validated; the per-step GDN
    //     recurrence drift accumulates into a DD-REP/CHARSPAM attractor over
    //     long gens — 9B-q8 GQ-004 0/3→3/3 under via-prefill alone).
    //   * 9B-bf16 (≤32 layers): ON since the 2026-06-11 carve-back (on the
    //     pvf32 binary, via-prefill HEALS the 9B-bf16 verylong attractor
    //     1/3→3/3, N=3 byte-identical, matches llama.cpp bf16).
    //   * 27B-bf16 (>32 layers): was the SOLE carve-OUT (returned false),
    //     because on the PRE-pvf32 binary via-prefill ALONE regressed it. The
    //     follow-up analysis proved that boundary OBSOLETE: paired with AP=2 (which
    //     this branch now sets for 27B-bf16, see `attn_precise_default`),
    //     via-prefill ON is REQUIRED — it removes the genuine long-form Moka
    //     stutter (GQ-004 back to 3/3) and the `<think>` reasoning-leak entry,
    //     while AP=2 fixes the prefill-attention near-ties. Both levers are
    //     load-bearing and independent; together they mirror the validated
    //     9b-bf16 winning stack and take 27b-bf16 GQ-014 4/8→8/8 (N=3). The
    //     prior "short 15/15→14/15" regression was measured at AP=0+viapre;
    //     with AP=2 the 27b-bf16 short gate is pristine 15/15.
    //
    // The predicate is therefore now unconditional ON. The branches below are
    // retained as documentation of the (now-collapsed) per-class structure so
    // future evidence can re-split a class back OUT without reconstructing the
    // reasoning. CUDA-only: the SOLE consumer is `gdn_decode_via_prefill_`
    // `enabled()` in `cuda/backend_impl.rs` (`#[cfg(feature = "cuda")]`); Metal
    // runs its OWN GDN decode path (`metal/gdn.rs` megakernel/dual-gates tiers)
    // and never reads this resolver or `LUMEN_CUDA_GDN_DECODE_VIA_PREFILL`, so
    // removing the carve-out is a provable no-op on the Metal/CPU build.
    // `LUMEN_CUDA_GDN_DECODE_VIA_PREFILL=0|1` overrides either way.
    let dense_bf16 = MODEL_DENSE_QUANT_HINT.load(Ordering::Relaxed) == HINT_BF16;
    let small = {
        let l = model_block_count();
        l > 0 && l <= 32
    };
    // 27B-bf16 large carve-out removed: `|| (dense_bf16 && large)` folded in,
    // making the expression total. Kept factored for readability + future
    // re-split.
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
/// step through the split-K kernel pair (sequence-parallel: heads x 4 chunks
/// + merge) instead of the one-CTA-per-head tiled kernel. Lifts the
/// occupancy ceiling on few-head models (27B: 24 CTAs -> 96 + 24) and cuts
/// each partial CTA's sequence walk to ~1/4 (total work stays linear in
/// context length). Quality-equivalent
/// near-tie — the cross-chunk merge sums in a different order than the
/// tiled kernel's progressive rescale. `=0` opts out, `=1` forces on; unset
/// resolves model-aware: ON for Q8_0-body and BF16-body dense models (the
/// classes the engine A/Bs + full GQ/DET gates banked: Q8 +0.195 ms/tok on
/// A100-SXM, BF16 +0.298 ms/tok on H100), following the canonical-defaults
/// master switch. Q4-body models stay on the tiled route — the gates run
/// FAILED Q4 quality with split-K on (near-tie perturbation lands on the
/// noisier quant), so widening further is a separate, gated decision.
pub fn attn_splitk_enabled() -> bool {
    match std::env::var("LUMEN_CUDA_ATTN_SPLITK") {
        Ok(v) if v == "0" => false,
        Ok(v) if v == "1" => true,
        _ => {
            matches!(
                model_dense_quant(),
                Some(QuantScheme::Q8_0) | Some(QuantScheme::Bf16)
            ) && !model_is_moe()
                && canonical_default_on()
        }
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
/// ([48,5120] each, 24-CTA grids) issue as ONE banked launch. The banked
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

/// Per-process default for `LUMEN_CUDA_SOA_LOCKED` when the env is unset.
/// ON for quantised dense (the codegen-locked Q4_0 split matvec: word-load
/// nibble stream + load-hoist + `.rn`-pinned epilogue, bit-deterministic and
/// faster than the unlocked split kernel). The effect is gated downstream by
/// Q4 split-dispatch + locked-kernel presence (`matvec_q4_split_q8_1_locked`),
/// so on a Q8/BF16 model the locked kernel is absent and this is a no-op.
///
/// **MoE: explicit OFF.** `SOA_LOCKED` implies the Q4 split clone pass
/// (`repack_all_layers_q4_clone_to_split`), which populates the dense
/// `wq/wk/wv/wo/w_gate/w_up/w_down` siblings only. On an MoE LBC the dense MLP
/// is replaced by per-expert weights, so arming the clone there would partially
/// populate siblings exactly as the Q8 SPLIT pass did before its MoE gate
/// (PAD-token spam). Gating OFF for MoE mirrors `q8_split_default` and keeps the
/// Q4-dense decode win without touching the MoE path.
pub fn soa_locked_default() -> bool {
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
/// when a new env gate ships, or the validator will warn at startup.
///
/// Sorted alphabetically to make `diff` reviewable when the list changes.
const KNOWN_LUMEN_ENV_VARS: &[&str] = &[
    "LUMEN_AB_ITERATIONS",
    "LUMEN_AB_WARMUP",
    "LUMEN_ANTI_RESTATE",
    "LUMEN_ANTI_RESTATE_LOOP",
    "LUMEN_ANTI_RESTATE_NGRAM",
    "LUMEN_ANTI_RESTATE_SUBWORD",
    "LUMEN_BASE_URL",
    "LUMEN_BENCH_ITERATIONS",
    "LUMEN_BENCH_SCALE",
    "LUMEN_BENCH_TOKENS",
    "LUMEN_BENCH_WARMUP",
    "LUMEN_CACHE_DIR",
    "LUMEN_CHAT_ENABLE_THINKING",
    "LUMEN_CONVERT_KEEP_Q6K_OUTPUT",
    "LUMEN_CONVERT_SOURCE_FIDELITY",
    "LUMEN_CORR010_MODEL",
    "LUMEN_CUDA_Q4_1_DOWN",
    "LUMEN_CUDA_Q5K_SSMOUT",
    "LUMEN_CUDA_Q6K_HEAD",
    "LUMEN_CUDA_ARGMAX_TILED",
    "LUMEN_CUDA_ATTN_BANK3",
    "LUMEN_CUDA_ATTN_PRECISE",
    "LUMEN_CUDA_ATTN_PRECISE_DBG",
    "LUMEN_CUDA_ATTN_PREP_FUSE",
    "LUMEN_CUDA_ATTN_SPLITK",
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
        .filter(|k| {
            !KNOWN_LUMEN_ENV_VARS
                .iter()
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

/// Resets the process-wide hint atomics to their defaults. Test-only —
/// used by the unit tests below so each test starts from a known state.
#[doc(hidden)]
pub fn reset_for_tests() {
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
    fn attn_precise_default_per_class() {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        // Ratified 2026-07-22: AP=3 (scalar — exact-F32 QK^T AND exact-F32 P@V)
        // for EVERY supported production class (MoE + dense 9B + dense 27B ×
        // Q4/Q8/BF16). AP=3 keeps the exact P@V that heals GQ-014 (2026-06-11 /
        // 2026-06-12 evidence below) and ADDS exact QK^T to close the
        // quant-independent F16-QK score carrier that flipped case-08 (cuda/9B/
        // Q8_0: F16-QK emits 35, HF-F32 golden 28). It is NOT AP=1 (qkf32),
        // which would reopen the F16 P@V hole and regress GQ-014. Legacy callers
        // that never set the block count still get conservative WMMA (0). The
        // per-quant `match` arm is kept so a class can be re-split on future
        // evidence.
        reset_for_tests();
        assert_eq!(
            attn_precise_default(),
            0,
            "unset block count -> legacy WMMA"
        );

        // 9B (32-layer) dense: AP=3 scalar regardless of quant.
        reset_for_tests();
        set_model_block_count(32);
        assert_eq!(
            attn_precise_default(),
            3,
            "dense 9B (32 layers) -> scalar (AP=3)"
        );
        reset_for_tests();
        set_model_block_count(32);
        set_model_primary_quant(QuantScheme::Bf16);
        assert_eq!(
            attn_precise_default(),
            3,
            "dense 9B bf16 -> scalar AP=3 (size wins)"
        );

        // 27B (64-layer) dense, per-quant discrimination — keyed on the PRIMARY
        // (bulk) scheme, NOT output_proj (which is Q8_0 for both q4 and q8).
        reset_for_tests();
        set_model_block_count(64);
        // No primary-quant set (legacy 27B caller) -> conservative legacy WMMA.
        assert_eq!(
            attn_precise_default(),
            0,
            "dense 27B, quant unset -> legacy WMMA"
        );
        reset_for_tests();
        set_model_block_count(64);
        set_model_primary_quant(QuantScheme::Q4_0);
        assert_eq!(
            attn_precise_default(),
            3,
            "dense 27B Q4_0 -> scalar AP=3 (exact P@V GQ-014 heal + exact QK^T)"
        );
        reset_for_tests();
        set_model_block_count(64);
        set_model_primary_quant(QuantScheme::Q8_0);
        assert_eq!(attn_precise_default(), 3, "dense 27B Q8_0 -> scalar AP=3 (exact P@V GQ-014 heal + exact QK^T; prior regressions were detector false-positives)");
        reset_for_tests();
        set_model_block_count(64);
        set_model_primary_quant(QuantScheme::Bf16);
        assert_eq!(
            attn_precise_default(),
            3,
            "dense 27B bf16 -> scalar AP=3 (exact P@V + exact QK^T, paired with via-prefill ON)"
        );
        // The crux of the 2026-06-12 root-cause: output_proj (Q8_0) must NOT be
        // what drives this — only the primary bulk scheme. Set the coarse
        // output_proj hint to Q8_0 (as a real q4 LBC does) WITHOUT a primary
        // scheme: must stay legacy WMMA (no q4 signal present).
        reset_for_tests();
        set_model_block_count(64);
        set_model_dense_quant(QuantScheme::Q8_0); // output_proj hint only
        assert_eq!(
            attn_precise_default(),
            0,
            "dense 27B with only output_proj=Q8_0 hint (no primary) -> legacy WMMA"
        );

        // MoE: AP=3 scalar regardless of size or quant.
        reset_for_tests();
        set_model_block_count(64);
        set_model_is_moe(true);
        assert_eq!(
            attn_precise_default(),
            3,
            "MoE -> scalar AP=3 regardless of size"
        );
        reset_for_tests();
        set_model_block_count(64);
        set_model_is_moe(true);
        set_model_primary_quant(QuantScheme::Bf16);
        assert_eq!(
            attn_precise_default(),
            3,
            "MoE bf16 -> scalar AP=3 (MoE wins over bf16 carve-out)"
        );
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

        // 27B bf16 (>32 layers): ON — THE CHANGE. Previously the sole carve-OUT
        // (returned false); follow-up analysis proved it must be ON, paired with AP=2.
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
        // Q4 dense (lm_head often Q8_0 → HINT_QUANTISED) defaults SOA_LOCKED ON.
        set_model_dense_quant(QuantScheme::Q8_0);
        assert!(
            soa_locked_default(),
            "quantised dense should default SOA_LOCKED=ON"
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
        "LUMEN_BENCH_SCALE",
        "LUMEN_BENCH_TOKENS",
        "LUMEN_BENCH_WARMUP",
        "LUMEN_CACHE_DIR",
        "LUMEN_CHAT_ENABLE_THINKING",
        "LUMEN_CORR010_MODEL",
        "LUMEN_CUDA_ARGMAX_TILED",
        "LUMEN_CUDA_ATTN_BANK3",
        "LUMEN_CUDA_ATTN_PRECISE",
        "LUMEN_CUDA_ATTN_PRECISE_DBG",
        "LUMEN_CUDA_ATTN_PREP_FUSE",
        "LUMEN_CUDA_ATTN_SPLITK",
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
