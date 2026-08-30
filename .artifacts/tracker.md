# Lumen engineering tracker

Source of truth for open items and verified-latent findings. Append-only per
item; close with the shipping release.

## Open

- **MODE-1 · Bimodal Metal GPU speed state (M3 Ultra)** — ~40 vs ~22 tok/s on
  27B-Q4, whole-GPU (prefill degrades 2.82×, decode 1.63×). Refuted causes:
  engine version, thermal-after-idle, contention, silent GDN PSO fallback
  (6 informative full-stderr logs, zero warnings), Q4 page-cache residency,
  CPU-load ramp. Remaining suspect after PERF-Q8 was disentangled (below):
  GPU power/performance governor — needs sudo powermetrics --samplers
  gpu_power captured in both states.
- **PERF-Q8-METAL · a ~9% Q8-specific decode deficit, cause unattributed** —
  2026-08-29 bracketed experiment: Q4 batteries fast all session (CV<1%)
  bound a Q8-specific deficit that survives same-session normalisation
  (board Q8/Q4 0.614 vs probe 0.556). The bimodal slow mode is excluded;
  the page cache is excluded for DECODE time (prewarm left decode at
  21.3, CV 1.2%) — but the prewarm `cat` ran at disk rate (5.75 GB/s), so
  full residency was not itself measured, and a **33% Q8-specific
  non-decode overhead survived the prewarm** (6.0 s/seq vs Q4's 0.6;
  board-era 2.9%) and is unexplained — the most concrete open signal.
  Cause unattributed: no Q8 cross-version A/B exists yet (board v0.11.0
  vs probe v0.15.0 confound version/session/binary). Next: same-session
  battery-patched v0.11.0 vs v0.17.0 A/B, and instrument the non-decode
  window. The published 1.15× stays withdrawn.

## Closed this round

- **C2 · CLOSED (fixed)** — the single-launch QKV path was reachable after
  all: `has_qgate_fusion` is tensor presence (`attn_q_norm.is_some()`), the
  LBC carries no architecture field, and `attn_q_norm` is optional — a
  `qwen35`-declared GGUF omitting it converted cleanly onto the unguarded
  path (reproduced). The loader now requires uniform, contiguous wq/wk/wv
  on full-attention layers without per-head Q/K norms. Loader-only and
  Metal-scoped: the converter still emits such layouts, and the CUDA
  sibling predicate is a separate open item (QKV-SHAPE below).
- **Shexp float serving — CLOSED (fixed)** — the fused fallback gained its
  Bf16 arm and the missing barrier before SwiGLU; the fallback's numerics
  are proven against a CPU reference on real Metal, and the test now uses
  a concurrent compute encoder so the barrier is load-bearing (a passing
  run cannot *prove* the race absent — the barrier's necessity rests on
  the Metal ordering model, the test keeps it exercised);
  `shared_expert_down_buf` is now sized `hidden.max(se_inter)` floats so
  the fallback's gate/up intermediate cannot overrun it. The load guard
  admits gate/up pairs uniform in {Q8_0, Q4_0, F16, Bf16, F32} with down
  independent — matching kernel truth — so the over-rejection of float
  shared experts is gone. The single-encoder MoE selector
  (`encode_moe_ffn_with_shared_fused`) now also keys on the gate quant,
  routing float gate/up pairs to the non-fused fallback instead of the
  fused path's error arm (mutation-proven regression test). The raw
  encoder's float-gate fallback (`encode_shared_expert_ffn_decode_raw`)
  gained the same full down-quant dispatch as the fused variant — its old
  F16/Bf16-else-F32 match read Q8_0/Q4_0 down bytes as f32
  (mutation-proven: reverting yields NaN vs the CPU reference).

- **GDN-F16-PREFILL · CLOSED (guarded)** — the Metal GDN prefill
  projections (attn_qkv in-projection, attn_gate, ssm_out) dispatch on
  Q8_0/Bf16/Q4_0 arms with a per-token F32 fallback: F16 weights
  (loader-admitted) were read as f32, silently corrupting prefill while
  decode has F16 arms — broken end-to-end since any run prefills first.
  The loader now rejects F16 on those three tensors for GDN layers with a
  clear re-convert error (C2 precedent: the loader must not rely on what
  the converter happens to emit). Adding real F16 prefill arms would
  supersede the guard.
- **CUDA-HEAD-ROWS · CLOSED (fixed)** — the round-6 raw-length validation
  for the output head checked flattened `vocab*hidden` block divisibility
  only; the head matvec kernels lay blocks per row, so `hidden_dim` itself
  must be block-aligned (Q8_0/Q4_0: 32, Q6_K: 256). A total-aligned but
  row-misaligned head (e.g. hidden=48 under Q8_0) passed validation and
  the kernels misindexed. `validate_output_head_row_alignment` now rejects
  it (unit-tested with the exact counterexample); the embedding keeps the
  flattened check — its lookup kernels tolerate blocks crossing rows.

- **GDN-GATE-F32 · CLOSED (guarded)** — the Metal GDN decode F32-gate
  fallback reads `normed_buf`, which only the F32 QKV route writes; the
  fused Q4_0/F16/Bf16 QKV routes RMSNorm inline from `x_buf` and leave it
  stale, so gate-F32 next to a fused QKV route computed silently wrong
  output (validator's Q8-parity rule was blind to it). The loader now
  rejects F32 attn_gate unless attn_qkv is also F32; F32/F32 dequantized
  models stay loadable. The companion race claim was CONFIRMED after an
  initial wrong refutation (the callers' serial-encoder choice is
  irrelevant: `encode_gdn_layer_decode_fused` ends the caller's encoder
  and opens its own CONCURRENT projection cluster, gdn.rs:~835, with no
  barrier before :1174): the F32 QKV fallback's RMSNorm write of
  `normed_buf` was unordered against the QKV matvec and F32-gate reads.
  Fixed with `memory_barrier_with_resources(&[&normed_buf])` after the
  RMSNorm dispatch, making the admitted F32/F32 combination sound.
- **METAL-HEAD-ROWS · CLOSED (fixed)** — Metal had the same output-head
  row bug fixed on CUDA: the head kernels stride rows by
  `hidden_dim / 32` blocks, and the raw Q8_0/Q4_0 head upload had no
  row-alignment check. `init` now rejects Q8_0/Q4_0 heads whose
  `hidden_dim % 32 != 0` (mirror of `validate_output_head_row_alignment`).
- **EMBED-U32-CAP · CLOSED (guarded)** — the CUDA and Metal embed/head
  kernels compute element and byte offsets in 32-bit
  (`token_id * hidden_dim + gid`, `block_idx * block_bytes`), wrapping
  past 2^32 with no loader cap. Both backends now reject globals whose
  element count or byte length exceeds 2^32 (exact boundary: counts up
  to 2^32 are fine, max index 2^32-1) — `raw_global_expected_len` on
  CUDA, `validate_raw_global` in Metal `init`, which now also checks the
  raw byte length equals the scheme's packed size (a truncated blob
  would send the shaders past the buffer; CUDA already had the equality
  check). All shipped models sit ≥1.6x below the cap (F16/Bf16
  bytes are the tightest at 1.69x; element count ≥3.3x). Reviewer ask for per-scheme/per-backend max-index analysis
  REJECTED as over-engineering: the uniform cap only bites at ≥2^32
  elements, no such model exists, and it errs conservative (rejects,
  never admits a wrap).

## Verified-latent (guard exists or path unreachable; do not fix unprompted)

- **C5b · Metal Q4_1 MoE expert kernels unreachable** — `named_slices()`
  covers experts and the layer-tensor allowlist rejects Q4_1, so the expert
  kernels cannot be reached; the converter force-requants Q4_1 experts to
  Q4_0. NOTE: that allowlist is NOT dead code — it is the live guard for
  non-expert Q4_1 under `--target generic` (Q4_1 is preserved there only
  when `LUMEN_CONVERT_SOURCE_FIDELITY=1`; the default requants it). Only
  the Q4_1 expert *kernels* are unreachable.
- **disk_sync GDN allocator** — `ensure_gdn_storage_for_layout`
  (`metal/disk_sync.rs:~474`) writes neither `gdn_layer_idx_map` nor
  `gdn_h_states_f16`; reachable only via `--session-resume` on a
  non-resident Metal run, which errors at decode before generation (tested:
  session file unmodified). Contain by rejecting a recurrent section when
  `expected_gdn_layout` is None if session-resume ever gains a Metal
  streaming mode.
- **Metal caps() hardcodes gpu_resident=true** — makes `--no-gpu-resident` /
  `--async` decode (and the entire expert-streaming CLI surface) dead-end on
  Metal with a clean error after a working prefill; `is_gpu_resident()` has
  zero call sites. Fix is dynamic caps like CUDA's — a behavior change
  needing its own validation round.
- **Convert-side %32 assert convention** — 19 sibling `assert!` sites on
  block divisibility across the converters (18 production; 1 is the
  `#[cfg(test)]` Q3_K encoder at `dequant.rs:1018`). Each runs on the
  convert path for its architecture and source-quant branch — a dense
  convert never executes the MoE converter's 8, and quant-specific arms
  gate several more. Well-formed GGUFs pass; a malformed one panics
  instead of returning `ConvertError`. Converting the class to
  `ConvertError` (and `checked_mul` for hostile headers per
  `format/quantization.rs` precedent) is a single sweep when touched next.
- **M-Z-DET-BASELINE** — the v0.14.0 DET-001 gates file is a transcription
  (bare md5 fields); byte-stability claims should cite the recorded
  v0.10.0/v0.11.0 boards or the raw v0.15.0/v0.16.0 tails instead.
- **X3 · CUDA expert offsets lack a per-slice extent check** — kernels read
  expert tensors at offsets with sizes derived from dims, not slice
  lengths; a hand-built LBC whose expert slice lies about its length could
  read past the layer blob (single-reviewer finding; converter output
  cannot produce it). `LayerIndex::validate` would bound offset+length but
  has zero production call sites, so it is no mitigation today. Guard
  needs dims plumbed into `build_moe_meta` — do with the next CUDA MoE
  change.
- **QKV-SHAPE · fused-QKV shape-assumption class not fully closed** — the
  new Metal guard closes the quant/contiguity instance, not the class.
  Remaining members: (a) no row-count validation — the closer is asserting
  wq rows == q_dim (or 2*q_dim when `attn_q_norm` is present) and wk/wv ==
  kv_dim, derived from slice length / bytes-per-row; blocked in
  `create_layer_buffer` by the scratch-buffer lock already being held
  (naive re-lock deadlocks — needs the dims passed in, not re-derived);
  (b) CUDA sibling predicate `has_qgate_fusion = lw.attn_q_norm.is_some()`
  (`cuda/backend_impl.rs:~2258`) has no equivalent guard; (c) Metal
  option-A fused QKV (`metal/moe.rs:2554`) gates only on
  `gdn_layer_idx.is_none()`; (d) the converter still emits
  guard-rejected layouts silently — a convert-time check would fail fast.
  Do as one change with dims plumbed through.
- **TEMP-PATH · fixed-name staging paths remain** — the PID fix covered
  the 5 session/provider sites only. The verified remaining production
  instance: `lumen-cli/download.rs:171` builds `{filename}.part` (and
  `.sha256`) with fixed names in the shared model cache — two concurrent
  `lumen run` downloads of the same file clobber each other's `.part`
  before the atomic rename at `:280`. Most other fixed-name temp sites
  found in the sweep are `#[cfg(test)]` fixtures (risk = parallel
  `cargo test` flakiness only, e.g. `storage/sync.rs:99,123`,
  `download.rs:453/471/502`). Add `std::process::id()` to the download
  staging names when touched next.
- **DENSE-F16-STREAM · Metal streaming dense-FFN gate/up F16/Bf16 arm
  missing** — `metal/backend_impl.rs:~2536` (`compute_layer` dense FFN)
  dispatches gate/up on Q8_0/Q4_0-else-`matmul_bytes_f32`; F16/Bf16 would
  be read as f32. Unreachable in production: every `forward_pass` /
  `compute_layer` call site is gated behind `!caps.batched_prefill` or
  `!caps.gpu_resident`, and Metal hardcodes both true (the same
  containment as the caps() entry above — fix together if caps ever go
  dynamic). The same function's down projection has the float arms.
- **Q4_1-LATENT · MoE gate+up `_` arms select the Q4_0 pipeline** —
  `metal/moe.rs:1246,:1401` catch-all arms would run Q4_1 weights on the
  Q4_0 kernel, and the `has_gate_kernel` guards
  (`decode_greedy.rs:~2035`, `decode_single_cb.rs:~1702`) explicitly
  admit Q4_1. Latent only because the load allowlist rejects Q4_1
  layer-wide; goes live immediately if Q4_1 is ever re-admitted (pair
  with C5b above).
- **Option-A output-proj catch-all** — `metal/moe.rs:~3282` catches
  F16/Bf16 in an `_` arm inside a function documented unreachable on
  every production path; dead code, do not fix unprompted.
- **GDN test-model generator lacks `attn_gate`** — `generate_test_model_q8_0_gdn`
  builds GDN layers without the fused gate, so runtime tests over it stop at
  a clean "missing attn_gate_off" shape error; extending the generator would
  let the streaming-wiring test assert full prefill output.
