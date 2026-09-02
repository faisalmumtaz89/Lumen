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

- **R8-ISSUE-13 · 27B + MoE load-coverage — CLOSED (empirically proven, both models)**
  — the round-8 loader changes were end-to-end validated by DOWNLOADING,
  CONVERTING, and LOADING the two real models #13 named, on Metal (serial,
  MMAP_ONLY, current round-8 binary), and checking output coherence — no
  config-twin assumption:
    - **Qwen3.6-27B Q4_0** (64 layers, hidden=5120, inter=17408, 48 GDN + full-attn,
      Q6_K→Q8_0 upcast, ssm_out 48/48): "What is the capital of France?" →
      "The capital of France is Paris." (exit 0). Log:
      evidence-v0220/issue13/qwen3-6-27b-metal-load.log.
    - **Qwen3.5-MoE-35B-A3B Q4_0** (40 layers, 256 experts top-8, shexp requant
      Q4_0): "17 × 20? + capital of France" → "340, Paris" — the MoE arithmetic
      is CORRECT (340, not the corrupted "39" signature that a broken MoE emits),
      and attention/recall coherent (exit 0). Log:
      evidence-v0220/issue13/qwen3-5-moe-35b-a3b-metal-load.log.
  Also earlier confirmed on the cached 9B GDN. So `validate_layer_quants` wiring,
  the universal `ssm_out` check, `LayerIndex::validate`, presence/geometry guards,
  and the MoE expert/router path all load and serve real 9B/27B/MoE-35B models
  coherently on Metal.
- **R8-ISSUE-11 · 0/40 treatment claim now backed by retained per-run records — CLOSED (evidence)**
  — the concurrency-fix "0/40 failures" headline was a bare aggregate beside a
  correction that had given the CONTROL per-run records — an asymmetry. Added a
  round-8 addendum to `evidence-v0190/n2-concurrency-proof.txt` citing retained
  per-process exit records two ways: the control section's own fixed-binary rows
  (0/100 each in `n2-control-runs-main.txt`) and a fresh 0/200 re-run on the
  current binary via the same `n2-control-harness.sh`
  (`evidence-v0220/issue11/n2-treatment-rerun-*.txt`, sha recorded). Reworded the
  shipped CHANGELOG [0.21.0] concurrency bullet to rest on the retained records
  (0 failures fixed vs 14/100 & 13/100 pre-fix) instead of the un-recorded "forty
  runs" figure. No code (CHANGELOG + evidence only).
- **R8-ISSUE-12 · convert-log exit codes + MoE-geometry source provenance — CLOSED (evidence)**
  — regenerated the four category-2 convert logs WITH explicit exit codes on the
  current binary (`evidence-v0220/issue12/convert-logs-with-exitcodes-*.txt`):
  c2q/c2b/c2n1 exit 1 with the same convert-gate rejections as v0.19.0 (Q+gate
  bias, wq geometry, no-FFN-pre-norm), c2f16 exit 0 — the round-8 binary
  reproduces the gate behaviour. Added source provenance to
  `evidence-v0190/moe-geometry-verification.md`: sha256 of the retained
  `moe-gguf-header-16MB.bin` (232ee271…) + the tensor-info offsets/dims/types of
  every cited tensor (blk.0.attn_qkv, ssm_conv1d, blk.3.attn_q/k/v), re-parsed
  round-8 and byte-matching the claims. Evidence only.
- **R8-ISSUE-9-SSMOUT · generic-target `ssm_out` geometry now validated at convert — CLOSED (fixed)**
  — round-8 review (codex) found `validate_layer_plan` checked `ssm_out` geometry
  only inside its Metal-only branch [erratum 2026-09-03: this check was itself new in 2c12c1c — see REMEDIATION-HISTORY], while the converter sizes `ssm_out` straight
  from the source GGUF's element count — so a malformed-source GGUF converted with
  `--target generic` produced a wrong-geometry `ssm_out` LBC that no convert-time
  check caught (CUDA/CPU would then read it at the wrong geometry; only the Metal
  load guard would catch it) [erratum 2026-09-03: the Metal guard did not check `ssm_out` at 6914bd4 either — see REMEDIATION-HISTORY]. Hoisted `validate_projection_geometry("ssm_out", …,
  gdn_v_dim, [hidden])` out of the `if metal` block into the universal projection [erratum 2026-09-03: an addition, not a hoist — see REMEDIATION-HISTORY]
  checks (serving_rules.rs, beside the existing universal `attn_gate` check), so
  every target now rejects an inconsistent `ssm_out` at convert. Well-formed
  sources unaffected (convert 199 / format 82 still green). MUTATION-PROVEN by
  `generic_convert_rejects_wrong_ssm_out_geometry` (metal_target_lockstep.rs):
  a GDN GGUF with a doubled-width `ssm_out` is refused at `--target generic`
  conversion; removing the universal check lets the conversion succeed and fails
  the test.
- **R8-ISSUE-9-QUANT-WIRING · Metal K-quant load guard now mutation-pinned — CLOSED (fixed)**
  — round-8 review (codex) showed `validate_layer_quants` is a benign-reachable
  CORRECTNESS guard, not a test-integrity residual: a `--target generic` K-quant
  MoE artifact (legitimate for CUDA/CPU; the macOS cache falls back to it) would,
  without the guard, feed K-quant bytes to Metal's F32-reading dispatch → silent
  gibberish (self-documented at cache.rs and the validator). Its wiring at both
  independent first-load entry sites — resident preload (gpu_resident.rs:209) and
  the streaming `create_layer_buffer` reached via prefill (mod.rs:993) — is now
  mutation-pinned by `metal_kquant_dense_tensor_rejected_at_both_load_sites`
  (metal/tests/basic.rs): a Q2_K dense FFN tensor is refused at each site with
  the unsupported-quant remedy. MUTATION-PROVEN: removing either call site fails
  exactly the matching assertion (209 → silent-accept; 993 → all-zeros gibberish),
  independently; both files byte-identical after restore. A third
  `validate_layer_quants` site exists — `create_partial_layer_buffer`
  (mod.rs:1051, MoE partial-decode) — deliberately not pinned because it is
  unreachable before 993/209 reject: partial decode requires all experts already
  LFU-cached, which only happens post-warmup after prior full decodes that each
  passed the pinned sites, so a K-quant expert model dies at 993/209 on its first
  MoE-layer decode and 1051 is only a downstream re-check. The K-quant MoE-EXPERT
  case rides the same guard: `validate_layer_quants` loops `named_slices()` (which
  includes `expert[i].gate/up/down`), so the dense-fixture pin covers the expert
  path transitively. The F32 `ssm_alpha`/`ssm_beta` gate check is a branch inside
  the SAME `validate_layer_quants`, so it is pinned-by-composition here too; only
  the attention-geometry and expert-count load-site wiring stays ledgered — see
  R8-ISSUE-9 in the verified-latent section for the corrected convert-containment
  rationale. Runtime 652, fmt clean.
- **R8-ISSUE-8 · download staging edges — CLOSED (2 fixed, 2 ledgered)**
  — `lumen-cli/src/download.rs` HF downloader. FIXED: (1) the create→guard
  window where an `fstat` failure leaked the staging `.part` — folded the
  `fstat` into `create_exclusive_staging`, which now returns `(path, fd,
  (dev,ino))` so the cleanup guard is armed from the fd's own inode with no
  second stat and no unguarded path-delete; an fstat failure (near-impossible
  on a fresh held fd) leaves the `.part` for `reclaim_stale_parts` to sweep
  after process exit (bounded, self-healing). (4) no-Content-Length truncation
  could publish unverified bytes — extracted `verify_complete_transfer` which
  is fail-closed on an unknown length (HF always reports one via GET or the
  HEAD fallback, so this never fires on a real pull; matches HF's own
  downloader), with a unit pin. LEDGERED: (2) the sidecar is written after the
  atomic rename, so a crash can leave a final file without a current sidecar
  indefinitely — harmless, the sidecar is write-only metadata NO load path
  reads (both paths check only file-exists + nonempty; `verify_sha256` has
  zero production callers), and correctness rests on the rename publishing
  only size+hash-verified bytes; (3) symlinked-cache TOCTOU — weak threat
  model under the default private cache (an attacker with cache-dir write
  already owns it), a documented residual only under a shared attacker-writable
  `LUMEN_CACHE_DIR`. Also fixed a stale doc comment (claimed sidecar-before-
  rename). Karpathy PASS (code + restructure); codex confirmed code-safety on
  every point (its residual notes were comment-accuracy, now corrected). CLI
  75/75, fmt clean.
- **R8-ISSUE-7 · CUDA preload validated MoE meta after building it — CLOSED (fixed)** [erratum 2026-09-03: this "after" ordering is intra-PR, never merged history — see REMEDIATION-HISTORY]
  — the GPU-resident preload built and stored per-layer MoE meta tables
  (`build_moe_meta` + `build_batched_offsets`, writing the persistent
  `moe_meta_cache`/`moe_batched_offsets` + device allocations) before the
  full validator set ran; expert-count and attn-extent checks lived only
  inside `upload_layer_weights`, which followed. [erratum 2026-09-03: both validators and those call sites are new in 2c12c1c — see REMEDIATION-HISTORY] A header passing bounds but
  failing those checks left a partial meta table behind before the load [erratum 2026-09-03: intra-PR state — see REMEDIATION-HISTORY]
  errored. The preload comment already promised "validate before ANY
  per-layer GPU work (meta tables included)" but only ran the bounds
  validator there; completed the contract by hoisting `validate_expert_count`
  + `validate_attn_vector_extents` ahead of `build_moe_meta` [erratum 2026-09-03: an addition, not a hoist — see REMEDIATION-HISTORY]
  (backend_impl.rs preload loop). Behavior-neutral on valid artifacts (same
  read-only serving_rules validators, convert-time gate already runs all
  three); a rejected header now aborts before any persistent/device write.
  Residual (failed preload) was and stays fail-safe — the half-built
  CudaState is dropped, never dispatched. Dual PASS (karpathy + codex). GPU
  load path itself deferred to the release production checklist.
- **R8-ISSUE-6 · LayerIndex bounds unwired + field divergence — CLOSED (fixed)**
  — `LayerIndex::validate` (offset+length ≤ blob) had zero production call
  sites, so a malformed `.lbc` whose sub-tensor offset ran past its blob was
  accepted at load and the Metal resident loader bound the raw offset into
  the device buffer (out-of-bounds device read). Wired `idx.validate(i)?`
  into the single `parse_lbc` layer loop (reader.rs:365-373), upstream of
  every `LbcFile` consumer. Adversarial review then found `validate`'s
  hand-rolled slice list omitted three loader-consumed fields
  (`attn_q_norm`, `attn_k_norm`, `ffn_gate_inp_shexp`) that Metal binds raw
  (gpu_resident.rs:299-308); root-caused by rewriting `validate` to iterate
  `SubtensorOffsets::named_slices()` (the single authoritative enumeration),
  so the field set cannot diverge again. `LayerOutOfBounds.tensor_name`
  became `String` to carry owned names. Pins: parse-time rejection
  (reader.rs), per-field rejection of the three, fully-populated-layer pass;
  all three mutation-proven. Dual PASS (karpathy + codex).
- **R8-ISSUE-5 · GDN pair-force wiring only partially pinned — CLOSED (fixed)**
  — the F16/split pair-force divisibility guard (`pair_forced_q8_slice`,
  rejects non-%32 element counts before the quantizer panics) is called at
  four convert sites (dense/MoE × qkv/gate); only dense-qkv had an E2E
  wiring pin, leaving the other three revertible to an inline `assert!`
  undetected. Parametrized the convert-E2E test over all four sites [erratum 2026-09-03: no pin existed at 6914bd4; all four are new — see REMEDIATION-HISTORY]
  (metal_target_lockstep.rs); each independently mutation-proven (reverting
  one site fails only its own test; files byte-identical after restore).
  Dual PASS (karpathy + codex r3).
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
  the fallback's gate/up intermediate cannot overrun it — for shipped
  MoE geometry (se_inter 512 < hidden 2048) this is byte-identical to
  the previous `hidden * 4`; the resize guards non-shipped geometries
  only. The load guard
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
  past 2^32 with no loader cap. Both backends now reject raw quantized
  globals whose
  element count or byte length exceeds 2^32 (the plain-F32 upload path
  sits outside these functions and remains uncapped — see the
  Verified-latent entry below) (exact boundary: counts up
  to 2^32 are fine, max index 2^32-1) — `raw_global_expected_len` on
  CUDA, `validate_raw_global` in Metal `init`, which now also checks the
  raw byte length equals the scheme's packed size (a truncated blob
  would send the shaders past the buffer; CUDA already had the equality
  check). All shipped models sit ≥1.6x below the cap (F16/Bf16
  bytes are the tightest at 1.69x; element count ≥3.3x). Reviewer ask for per-scheme/per-backend max-index analysis
  REJECTED as over-engineering: the uniform cap only bites at ≥2^32
  elements, no such model exists, and it errs conservative (rejects,
  never admits a wrap).

- **N3 · CLOSED (round 7, Category 2)** — Q+gate bias policy divergence:
  CUDA served QKV biases on Q+gate layers via its unfused path while
  Metal rejected the same artifact. Decided once for both backends:
  REJECT (no shipped model emits the combination — qwen35 has no biases,
  qwen2 has biases but never per-head Q/K norms; Metal has no kernel for
  it and fail-closed consistency wins). CUDA now rejects in
  `validate_mandatory_presence`; qwen2-style biased attention WITHOUT
  per-head norms stays served (unit-tested both ways). The converters
  refuse to emit the combination (UnsupportedModel).
- **N4 · CLOSED STRUCTURALLY (round 7, Category 2)** — the
  converter/loader contract is now enforced by construction for the
  covered GGUF layer-plan rules: the loaders'
  shared validation rules moved to `lumen_format::serving_rules` (the
  Metal and CUDA loaders call thin wrappers into the same module —
  CUDA for presence/geometry/slices/expert-bank/conv1d, Metal for the
  allowlist/pairing and attention-geometry set, which re-run the
  FFN-pre-norm/slice/expert-bank rules internally; backend-local rules
  such as CUDA's CtInt4G32 geometry and F32-only norm/bias checks, and
  both backends' global-tensor rules, remain outside the module), and
  `convert_gguf` runs `validate_layer_plan` — presence,
  per-tensor projection geometry, and (Metal target) the full Metal rule
  set — over every PLANNED layer's `SubtensorOffsets` before any byte is
  written. Because the gate reads planned `QuantScheme`s (after
  requant/upcast/pair-force), the convert-side shadow mappings of source
  `GgmlType`s are deleted [erratum 2026-09-03: nothing was deleted — see REMEDIATION-HISTORY], retiring the entire divergence class two
  review rounds of one-by-one fixes kept refilling (adversarial reviews
  found 7 then 5 instances; the composite closes all 12 plus the
  headerless-GDN residual — validated against the same
  declared-or-QWEN35_9B default the loaders use). Round-8 reviewer pass
  fixed the FILING: universal rules (zero-length optionals — the moved
  19-entry `validate_layer_slices` — FFN pre-norm, expert-bank
  uniformity via the shared `validate_expert_bank` that `build_moe_meta`
  now delegates to, and attn_gate geometry [hidden→v_dim]) run
  UNCONDITIONALLY, not inside the Metal branch; probes refuse on both
  targets (`e2e-cat2-provenance.txt` shows metal+generic legs). In-repo
  negative tests cover all four refusal classes
  (`contract_gate_refuses_loader_rejected_plans`, mutation-proven:
  disabling the gate fails the test). Fixer passes
  (K-quant/Q5 upcast, GDN pair force incl. F16 and F32-split, Q4_1
  requant) run in the planners BEFORE the gate, so legitimate sources
  land on servable schemes; anything a fixer misses is refused with the
  loader's own message prefixed "planned artifact would be refused at
  load". Probes: c2q/c2b/c2n1 refused (three distinct rule classes),
  c2f16 force-converts and serves, 9B+27B GGUFs regression-convert clean
  (`evidence-v0190/e2e-cat2-provenance.txt`). Remedy string fixed
  (`lumen convert --target metal`). Residuals: the HF importer
  (`convert_hf.rs`) builds correct geometry by construction and is not
  yet routed through the gate; generic-target artifacts satisfy the CUDA
  rules only (Metal's allowlist refuses K-quants at load with a
  re-convert message — by design). Lockstep/q6k test fixtures now
  declare GDN dims coherent with EVERY tensor (nv=8,nk=2,gh=8: v_dim
  64 = attn_gate/ssm_out/nh, qkv rows 96 — the gate caught fixtures whose
  declared dims their own tensors contradicted, twice). Known limits,
  stated: byte-length geometry is blind to transposition (neither loader
  catches that either); CtInt4G32 skips the generic geometry table (its
  ct4-specific check runs at CUDA load); the gate validates the PLAN —
  only the writer's blob-length equality ties plan to bytes; the HF
  importer bypasses the gate (its own shape asserts are stronger).
  Round-9 (codex r5): the fixture repair exposed a LIVE bug — no
  validator anywhere checked `ssm_conv1d` length while both backends
  index `qkv_rows x conv_kernel` F32s from it (`cuda/shaders/gdn.cu:48`,
  `metal/gdn.rs:221`): a short conv1d = CUDA OOB read / Metal silent
  read of the next layer's bytes. Closed universally:
  `serving_rules::validate_gdn_conv1d` (exact byte length) called by the
  Metal loader (validate_attention_dims), the CUDA loader
  (upload_layer_weights), and the convert gate (universal section) —
  negative-tested at unit level; the rule immediately caught (a) the
  format test-model generator's short conv, (b) two lockstep fixtures,
  and (c) the round-6 c2 probe family's own conv1d (those artifacts
  would have OOB'd at dispatch — regenerated coherent).
  Follow-ups carried, not lost: direct unit tests for
  `serving_rules.rs` itself (812 lines, currently covered indirectly
  through the loader wrappers + the mutation-proven convert gate test);
  tightening two role-agnostic negative-test needles; deriving fixture
  constants (96, nh) from the declared ssm.* keys instead of hardcoding.
- **REMEDIATION-HISTORY · pattern named (round 9): remediation history invented
  for newly written code** — ledger entries narrate a fix as moving, hoisting,
  deleting, replacing or re-ordering something, when git shows the cited commit
  only ADDED the code and no prior state existed in merged history. Six dated
  errata (A–F, 2026-09-03) stand corrected here, decided against 6914bd4 (the
  parent of 2c12c1c) and the round-7 commits 8ca6834 / 33ace97; every affected
  locus carries an inline pointer back to this entry and its original wording is
  retained. In every instance the SUBSTANCE — the gap and the fix — is real; only
  the history was invented.
  Erratum A (R8-ISSUE-7 — its title and consequence included — and the
  R8-ISSUE-9 call-site inventory): `validate_expert_count` and
  `validate_attn_vector_extents` have zero hits at 6914bd4. 2c12c1c defined both
  in serving_rules.rs (`git diff -U0 6914bd4 2c12c1c --
  crates/lumen-format/src/serving_rules.rs`: `validate_attn_vector_extents` at
  `@@ -618,0 +649,49`, `validate_expert_count` at `@@ -800,0 +926,27`), added
  their `upload_layer_weights` call sites as a pure `+10` hunk
  (crates/lumen-runtime/src/cuda/gpu_buffers.rs `@@ -1156`), and added the
  resident-preload calls before `build_moe_meta` in the same commit. So nothing
  pre-existed to hoist, and the "validated MoE meta AFTER building it" ordering
  never existed in merged history — it describes a state between iterations of
  the one squashed PR. Correct statement: added at both the resident preload and
  `upload_layer_weights`. The preload wiring is the load-site wiring R8-ISSUE-9
  ledgers as NOT mutation-pinned; the mutation-pinned tests belong to B and F.
  Erratum B (R8-ISSUE-9-SSMOUT; the R8-ISSUE-9 containment note; QKV-SHAPE (d)):
  serving_rules.rs / `validate_layer_plan` had no `ssm_out` geometry check at
  6914bd4 (the symbol appears only in slice-name lists there). Rejecting
  `ssm_out` geometry checks did exist ELSEWHERE and are separate from this
  gate: CUDA's CtInt4G32 upload path, `upload_projection_tensor(…, "ssm_out",
  …)` (unchanged — 2c12c1c only retyped that arm's `in_dim` arithmetic), and
  the HF-import converter's `linear_attn.out_proj` shape checks in
  convert_hf.rs (`(n, k) != (hidden, v_rows)` and the Bf16 shape test), which
  concern the HF path, not GGUF conversion; Metal's gpu_resident.rs
  compared `ssm_out.length == hidden × q8_row_bytes` only to gate an optional
  Q8→Q4 requant buffer — `None` on mismatch, never a rejection. 2c12c1c's
  serving_rules diff for `ssm_out` is all `+` lines: the Metal-branch
  `expect("ssm_out", …)` and the universal `validate_projection_geometry("ssm_out",
  …)` were BOTH added, nothing removed, and the Metal-branch check still exists.
  So the entry's premise "checked only inside its Metal-only branch" and "only the
  Metal load guard would catch it" describe the state within 2c12c1c, not the
  prior release. Correct statement: added a universal check alongside a
  Metal-branch check introduced in the same commit.
  Erratum C (the round-7 gate entry, re 8ca6834): "the convert-side shadow
  mappings of source `GgmlType`s are deleted" — `git diff -U0 8ca6834^ 8ca6834 --
  crates/lumen-convert/src | grep '^-'` removes no `GgmlType` mapping code (one
  pattern replacement and two diagnostic strings). The source-type predicates
  still exist and still drive planning — `metal_gdn_pair_forces_q8` has four call
  sites at HEAD, and 8ca6834 EXPANDED it (two `is_f16`/`is_f32` closures and
  three applications, five `+` lines). Correct statement: the mappings were not
  deleted; the gate validates every planned layer against the loader's own rules,
  and that — not any deletion — is what retires the divergence class.
  Erratum D (the staging entry, re 33ace97): "reclamation … moved back ABOVE the
  cache-hit return" — `reclaim_stale_parts` has zero hits at 33ace97^; the
  function and its call were added in 33ace97. Correct statement: added above
  the cache-hit return.
  Erratum E (the same entry, re 33ace97 — three sentences): "the Drop guard
  deleting by pathname … the guard now captures our inode" implies a prior
  pathname-deleting guard, "identity is (dev, ino) not ino alone" implies a prior
  ino-only identity, and "a `kill` subprocess also printed to the console"
  implies a prior subprocess liveness check. `StagingGuard`, `.dev()`, `.ino()`
  and `libc::kill` all have zero hits at 33ace97^ and no revision on any branch
  contains a `kill` subprocess — all were introduced together in 33ace97.
  Correct statement: the guard was introduced with (dev, ino) identity and
  `libc::kill` liveness from the start.
  Erratum F (R8-ISSUE-5): "only dense-qkv had an E2E wiring pin … Parametrized
  the convert-E2E test over all four sites" — `pair_forced_q8_slice` has zero
  hits at 6914bd4 and the `PairForceSite` enum plus all four tests are pure `+`
  in 2c12c1c; no prior pin existed to parametrize. The other half of that entry
  stands: the inline `assert!` sites in qwen35.rs WERE replaced (two `-` hunks).
  Correct statement: four convert-E2E pins added, one per site.
  Intra-PR narrative — states that existed only between iterations of one
  squashed PR and leave no trace in the merged diff — is now marked as such where
  it appears: the write-only/EBADF staging state (the parent hashed by pathname),
  and the two "reverted" hunks in the temp-path entry. Prior instances of this
  class, retracted below near the d0960ce note: the invented ledger-entry claim
  and the d0960ce "hoisted above build_moe_meta" claim. Claims that survive the
  same test and stand: round 7's rules "moved to serving_rules" (8ca6834's
  loader-side diff is deletion-dominated — gpu_resident.rs −497/+35,
  gpu_buffers.rs −164/+32 — as the loaders' own rule copies, their optional-slice
  lists included, were replaced by calls into serving_rules) and "the sidecar
  write moved AFTER the rename" (33ace97: write precedes rename at its parent,
  follows it after). Round labels in this ledger count review passes, not
  releases, and collide: 8ca6834 (2026-08-31) carries both "round 7" and
  "Round-9 (codex r5)"; 2c12c1c (2026-09-02) carries both "round 8" and
  "round-9" — substance unaffected. Policy: hoisted / moved / removed / deleted /
  replaced / re-ordered is stated only when the cited commit carries the matching
  `-` hunk; "restored" cites the commit that removed it; intra-PR sequence is
  labelled intra-PR; anything else is stated as an addition.
- **ROUNDING · pattern named and closed (round 7, Category 4)** — three
  historically published ratios rounded TOWARD Lumen: 0.892 (true
  0.891), 0.727 (true 0.726), 1.15 (true 1.145; the v0.11.0-metal
  board's own BOARD.md:31 records 1.145). All three were corrected
  (0.891 published at 51c3916 with its 0.892 changelog residue corrected
  at 2227714; 0.726 at 2a99719) or withdrawn (the 1.15× row) in
  earlier rounds; BOTH retained files that still carried old digits now
  bear dated errata with the exact arithmetic —
  `~/lumen-bench-out/benchmark-cuda-final.md` (0.727→0.726, plus its
  summary prose restated to exact three-decimal figures) and
  `Lumen-Workbench/benchmark/history/20260716T175200Z__vr2-9cell-battery/BOARD.md`
  (0.892→0.891). A round-7 independent audit of 70 published ratio
  derivations found zero further instances (65 exact, 5 rounded down;
  source: LUMEN-V0180-FINAL-CONCLUSION-2026-08-30.md:36 — the itemized
  70-row list is NOT retained, only that one-line summary, so the count
  carries that hedge); a separate claims-panel pass counted ~17 of ~26
  published ratios with no retained artifact (same report, :78).
  Current policy: exact or conservative rounding only, checked
  per-sentence before any figure ships. The Metal 1.30× attribution is
  relabeled in docs/support.md (same GGUF, sequential per-engine runs —
  `benchmark/history/20260824T090000Z__v0.11.0-metal/board-lite.json`
  records co_located: false on all 9 engine comparisons across its 3
  cells; its per-cell raw JSON carries lumen_version v0.11.0-dirty, a
  provenance nit on an artifact whose numbers were re-derived
  independently), and an evidence-policy
  note above the CUDA matrix covers both matrices (the Metal section
  carries a one-line pointer to it). NOTE: an earlier resolution doc
  claimed this ledger entry existed before it did — that false
  action-claim is part of the round-7 record and drove the
  claim-classification discipline (code/artifact/action) now in force.
  A second false action-claim from the same era is retracted here: the
  v0.17.0 resolution's "hoisted above build_moe_meta" implied a
  pre-existing call that was moved, but `validate_layer_slices` and
  BOTH its call sites were new in d0960ce — nothing existed to hoist
  (the v0.18.0 CHANGELOG bullet now carries a dated correction, and the
  retained resolution doc an appended one);
  with both errata now written, both halves of the reviewer's R8
  refutation are closed.
- **N1 · CLOSED (round 7)** — `ffn_norm` zero-sentinel with
  `attn_post_norm` absent produced a present zero-length norm buffer on
  CUDA (`unwrap_or` fell back to the sentinel itself) and an offset-0 F32
  misread on Metal (`map_or(0)`). Both loaders now reject the combination,
  keyed on `attn_post_norm`'s absence — NEVER on the zero sentinel alone,
  which is legitimate on every shipped GDN/MoE layer (the brick trap the
  round-7 report warned about). Repro: `evidence-v0190/c2n1.{gguf,lbc}`
  (loaded silently on v0.18.0; clean error now).
- **TEMP-PATH / N2 · CLOSED (round 7, Category 3)** — every temp-path
  collision site is PID-disambiguated: the production `.part` staging
  name in `download.rs` (the clobber window before the atomic rename;
  the `.sha256` sidecar deliberately keeps its stable name — persistent
  cache metadata), the two `storage/sync.rs` fixtures the round-7 report
  reproduced failing 30/40 and 22/40 under concurrent pairs, 5
  counter-only src siblings (incl. convert.rs/sharded.rs), 3 fixed-name
  download test dirs, and 23 integration-test dirs — 21 changed plus 2
  that turned out already-PID-safe and were reverted [2026-09-03: intra-PR, net-zero in the merged diff] (counters
  disambiguate threads within a process, never across processes).
  Round-final (codex fresh pass): PIDs are NOT unique across PID
  namespaces — two containers sharing a cache volume can both be PID 1,
  resurrecting the truncation race with a silently partial FINAL file.
  Closed with exclusive creation: staging is `{filename}.{pid}-{nonce}`
  opened with `create_new` (O_EXCL) and collision-retried — the
  filesystem, not the name, is the arbiter (no lockfile needed); same
  treatment in the bench model-cache generator. Reclamation parses both
  name forms and moved back ABOVE the cache-hit return [erratum 2026-09-03: added, not moved — see REMEDIATION-HISTORY] (a SIGKILLed
  loser's litter would otherwise never be reclaimed once the winner
  published; the scan is cheap now that liveness is one libc::kill).
  Adversarial round outcomes folded in: the `.part` staging gained a
  Drop guard (every error path after the guard is armed cleans our own
  PID-named file; one fallible fstat sits between the exclusive open
  and arming) and
  PID-liveness-checked stale-part reclamation at entry (never deletes
  staging it OBSERVES locally-live and actively-written — the exact
  final rule, with its foreign-namespace, legacy-name, and
  sample-to-unlink caveats, is at the end of this entry; pure
  age/name-based deletion of fresh files would
  reintroduce the bug); the sidecar write moved AFTER the rename with an honest comment
  (shared last-writer-wins by design, content identical per URL,
  write-only in production — `verify_sha256` has no production caller);
  the reviewer-identified bench MODEL-CACHE hazard fixed
  (`runner.rs` generated `bench_{size}.lbc` directly into the
  existence-as-readiness path — now PID-staged + atomic rename); two
  double-PID test hunks reverted [2026-09-03: intra-PR, net-zero in the merged diff]. Counts corrected: 5 counter-only src
  siblings (incl. convert.rs/sharded.rs), 23 integration-test sites of
  which 2 were already PID-safe (reverted [2026-09-03: intra-PR, net-zero in the merged diff]). Inherent residual, safe
  direction: reclaim's liveness check is namespace-relative — across
  containers it can keep another namespace's dead litter, or unlink a
  live foreign-namespace staging file, which costs that download a
  failed rename (an explicit retry diagnostic when the pre-rename
  identity check observes the reclamation, else a plain rename error),
  never corruption (O_EXCL guarantees creation exclusivity; for
  current PID-named writers on a filesystem honoring O_EXCL and atomic
  rename, the only
  concurrency-induced partial-final path is the ledgered microsecond
  check-to-rename window
  below, which additionally requires a same-PID-same-nonce name reuse;
  a nonconforming network mount is a second concurrency-induced path —
  it voids the O_EXCL arbitration itself, resurrecting the
  shared-staging race — ledgered below and declared unsupported;
  cross-namespace liveness needs a lease, correctly
  rejected). Non-concurrency partial-final paths, ledgered: a transport
  where neither HEAD nor GET yields a Content-Length skips the
  completeness check, so a close-delimited truncated response would be
  hashed and published as-is; and the publish is not fsync-durable, so
  a power loss straddling the rename can surface a partial or empty
  final. Codex's ask for a
  per-cache-key lockfile REJECTED as over-engineering with rationale:
  rename is atomic (final is always complete bytes), sidecar content is
  deterministic per URL and write-only in production; the theoretical
  file-A/hash-B window under a mutable upstream ref is ledgered.
  Empirical proof: 20 concurrent pairs (40 runs) of the two
  previously-failing tests, 0 failures. CORRECTION (2026-08-31): the
  same-session control originally cited here (old name 20/40, PID name
  0/40) was never retained and is withdrawn; the retained replacement
  runs the RELEASED v0.20.0 tag's test binary against main under 4-way
  contention with per-process exit records: v0.20.0 14/100 and 13/100
  failures vs main 0/100 and 0/100 (harness script, binary sha256s,
  per-run lines, and the corrected 3-in-25-rounds ~12% residual bound
  all in `evidence-v0190/n2-concurrency-proof.txt` and its companion
  n2-control-* files). Remaining fixed names are deliberate
  user-facing locations, read-only fixture constants, and env-var
  round-trip strings that never touch disk — none collision-prone.
  Review-round D-items folded: libc::kill liveness (EPERM = alive under
  another user, keep; a `kill` subprocess also printed to the console [erratum 2026-09-03: no subprocess kill ever existed; libc::kill was added in 33ace97 — see REMEDIATION-HISTORY]),
  legacy `{filename}.part` litter reclaimed when >1h stale, runner.rs
  rename-failure cleanup, and explicit post-finish flush in the model
  generators (BufWriter Drop swallows I/O errors). FINAL deletion rule
  after the namespace round: reclaim runs ABOVE the cache-hit return (a
  SIGKILLed loser's litter must remain reclaimable after the winner
  publishes) and deletes a PID-named staging file only on stale-mtime
  (>60s grace — a live writer
  in ANY namespace refreshes mtime every chunk) AND (ESRCH locally OR
  >24h stale) — pure pid-liveness is namespace-local and could ESRCH a
  live foreign container's writer; legacy fixed-name `{filename}.part`
  files carry no PID and are reclaimed on mtime age alone (>1h); in
  both forms the staleness/liveness sample and the unlink are separate
  steps, so a writer resuming inside that window loses its staging —
  for the PID form that download then fails without publishing
  (explicit retry message at the identity check, else a plain rename
  error — except through the same-PID-same-nonce check-to-rename reuse
  window ledgered below); the legacy form belongs to pre-0.21 binaries
  that hash and rename by pathname — such a writer fails at hash-open
  when the unlink precedes its reopen, at rename when it follows, and
  if another old binary has recreated the fixed name it can instead
  hash or rename the in-progress replacement and publish a partial
  final: the pre-0.21 shared-name race, which reclamation cannot
  prevent (old binaries truncate-create the fixed name regardless);
  the legacy branch simply removes >1h-stale litter left by those
  binaries. Non-UTF-8 cache dirs handled (staging
  paths built by join, never lossy string mangling). NFS/FUSE/SMB caveat
  ledgered: O_EXCL atomicity holds on NFSv3+/kernel 2.6+; nonconforming
  network filesystems cannot provide the guarantee — sharing a model
  cache over such mounts is unsupported. Codex r5's two deeper races
  also closed: (i) hashing/renaming by PATHNAME could, after an unlink,
  read a REUSED name's in-progress bytes and publish a partial final —
  the download now hashes through its own fd and verifies fd-inode ==
  path-inode before the rename (mismatch = disarm guard + clean
  retryable error; the residual TOCTOU between check and rename is
  microseconds and additionally requires a same-pid-same-nonce name
  reuse — ledgered); (ii) the production wrappers' cache-hit
  short-circuits bypassed download_gguf entirely, so post-publish
  reclamation never ran — both wrappers now call the exported
  reclaim_stale_parts before their cache-hit returns. Codex r6 then
  caught (i) a HARD bug in the fd-hash change itself — staging was
  opened write-only, so the same-fd hash read returned EBADF [erratum 2026-09-03: intra-PR state — the parent hashed by pathname — see REMEDIATION-HISTORY] and every
  cold download would have failed (fixed with .read(true); regression
  test exclusive_staging_write_then_hash_via_same_fd covers the exact
  create/write/seek/hash flow); and (ii) the Drop guard deleting by
  pathname could remove a stranger's reused-name file on our error
  paths — the guard now captures our inode at creation [erratum 2026-09-03: no prior guard existed; introduced with inode identity — see REMEDIATION-HISTORY] and removes only
  when the path still resolves to it. r7 hardening: identity is
  (dev, ino) not ino alone [erratum 2026-09-03: `.dev()` and `.ino()` were added together in 33ace97; no ino-only version shipped — see REMEDIATION-HISTORY], the fd stays open through the rename
  (prevents inode recycling from blurring the just-verified identity),
  and the regression test is feature-gated so no-default-features builds
  compile. Drop's own metadata-to-remove_file window remains a
  microsecond-class TOCTOU — absolute closure needs serialized cleanup,
  deliberately out of scope; every misfire direction is
  keep-not-delete except that window, and it additionally requires a
  same-pid-same-nonce name reuse. The guard's and the rename path's
  (dev,ino) identity re-checks are reasoned, not tested: no guard-reuse
  regression test exists (r7 review finding, accepted).

## Verified-latent / accepted residuals (each entry states its own containment — a guard, unreachability, or an accepted exposure; do not fix unprompted)

- **HOSTILE-HEADER-KERNEL-CAPS · residual u32/aggregate exposures under
  bounded-but-hostile headers (round-9 reviews; EMBED-U32-CAP posture:
  no real artifact reaches any of them, fix when touched)** — the
  round-9 hyperparam bounds gate (fields <= 2^15, seq <= 2^20, experts
  <= 256, layers <= 2^12, GQA divisibility, conv_kernel >= 2) makes all
  u32 DIMENSION arithmetic total, but deeper products can still exceed
  narrower consumers on hostile headers inside the bounds. Reachability
  varies per item — stated individually rather than as one blanket
  claim: (a) KV index
  bases computed in u32 (`kv_cache.cu` head base = head x seq x
  head_dim can pass 2^32) — needs a multi-GiB KV allocation to reach;
  (b) GDN h-state size v_heads x head_dim^2
  (to 2^45) narrowed `as u32` in `metal/gdn.rs` while the grid
  uses the wide value; (c) the fused GDN kernels process exactly 32
  lanes x 4 values = 128 per head, and there is NO production
  head_dim==128 check (only a `#[cfg(test)]` assert) — a declared GDN
  head_dim != 128 is an out-of-bounds READ AND WRITE on both backends;
  it needs only a small (<1 MiB) hostile-header artifact, no large
  body — the sharpest of these, though every real artifact declares
  128; (d) CUDA dynamic prefill dispatch u32
  truncation (`cuda/prefill.rs`) — reachable via a large batch/seq on a
  NON-hostile header (shipped 27B + long prompt), the one item needing
  no crafted header; (e) KV-total accounting products in
  `kv/mod.rs` can wrap u64 at bounded maxima (real configs
  allocate-or-abort long before). Line numbers deliberately omitted (they
  drift); grep the named symbols.
- **MOE-EXPERT-COUNT-WIRING-TEST · the header/bank reconciliation rule
  is not runtime-wiring-pinned** — round-9 added
  `validate_expert_count` (header num_experts == per-layer expert bank
  length for MoE layers) and wired it at all four load sites: CUDA
  `upload_layer_weights`, Metal resident preload, and both Metal
  streaming buffer paths (`create_layer_buffer`,
  `create_partial_layer_buffer`), plus the convert gate. The RULE is
  mutation-pinned (unit test, both mismatch directions + exemptions).
  What is NOT pinned by a runtime test is the WIRING at the loader
  sites: a full MoE LBC that is valid in every OTHER respect (attention
  geometry, FFN, expert-bank uniformity, conv1d, ...) but declares the
  wrong expert count would have to be hand-constructed to pass fifteen
  prior rules before reaching this check — disproportionate fixture
  surface (round-6 T1 precedent). Containment: the converter builds the
  per-layer bank as exactly `num_experts` entries (the same value that
  becomes the header), so a Lumen-produced artifact can never mismatch
  (verified by both round-9 reviewers against the Qwen3.5-MoE config);
  the guard only bites a hand-built LBC. Add a MoE-fixture wiring test
  when a MoE test-model generator exists.
- **R8-ISSUE-9 · load-point guard WIRING not mutation-pinned (same class as
  MOE-EXPERT-COUNT-WIRING-TEST above)** — servelumen round-8 observed that
  commenting out the load-site guard calls leaves the suite green. Full call-site
  inventory (13 = 9 Metal + 4 CUDA, broader than the 5 the finding named): Metal
  `validate_layer_quants` + `validate_attention_dims` + `validate_expert_count`
  at `create_layer_buffer` (mod.rs:993/997/998), `create_partial_layer_buffer`
  (mod.rs:1051/1055/1056), and resident preload (gpu_resident.rs:209/210/211);
  CUDA `validate_expert_count` + `validate_attn_vector_extents` in
  `upload_layer_weights` (gpu_buffers.rs:1159/1161) AND the sibling pair hoisted [erratum 2026-09-03: an addition, not a hoist — see REMEDIATION-HISTORY]
  into the resident preload for R8-ISSUE-7 (backend_impl.rs:18877/18882). The
  guard RULES are mutation-pinned (exhaustive `validate_attention_dims` /
  `validate_mandatory_presence` / `validate_attn_vector_extents` /
  `validate_expert_count` unit tests, both directions + exemptions), and the
  CONVERT-side gate that produces every shipped artifact IS runtime
  mutation-pinned (Issue 5's `metal_target_lockstep` four-site pins +
  `validate_layer_plan`). What is not pinned is the LOAD-site wiring: exercising
  it needs the GPU load path (Metal M3-serial / CUDA A100) plus a fixture that
  passes all prior rules — the disproportionate surface documented above.
  Containment (verified by round-8 review): no correctly-converted Lumen artifact
  trips any of these — `validate_expert_count`/`validate_attn_vector_extents` run
  at convert for EVERY target (serving_rules.rs:974/971 [citation corrected 2026-09-03], outside the `if metal`
  block); Metal `validate_attention_dims` byte-geometry is caught universally at
  convert by the byte-identical `validate_projection_geometry`; the GDN
  wk/wv-empty sentinel is a converter-guaranteed invariant; and `ssm_out`
  geometry is now validated universally at convert too (round-8 fix — hoisted out [erratum 2026-09-03: an addition, not a hoist — see REMEDIATION-HISTORY]
  of the Metal-only branch, mutation-pinned by
  `generic_convert_rejects_wrong_ssm_out_geometry`), closing the earlier
  generic/CUDA asymmetry codex found. CORRECTION (round-8 codex, verified):
  `validate_layer_quants` is NOT a clean-reject test-integrity guard — it is a
  benign-reachable CORRECTNESS guard. A `--target generic` K-quant MoE artifact is
  legitimate for CUDA/CPU and the macOS cache falls back to it when no `-metal`
  variant exists (cache.rs documents the "incoherent output" outcome); without the
  guard the Metal dense/MoE dispatch feeds K-quant bytes to an F32-reading pipeline
  → silent gibberish. That guard is now MUTATION-PINNED at both independent
  first-load entry sites (resident preload gpu_resident.rs:209, streaming
  create_layer_buffer mod.rs:993) — see the closed entry `R8-ISSUE-9-QUANT-WIRING`
  above (`metal_kquant_dense_tensor_rejected_at_both_load_sites`). The F32
  `ssm_alpha`/`ssm_beta` gate check (also benign-reachable via `--target generic
  --dequantize`) is a BRANCH INSIDE `validate_layer_quants` (serving_rules.rs:368),
  so it is PINNED BY COMPOSITION: the whole-function wiring is pinned at 209/993 by
  the K-quant test, and the predicate itself is unit-pinned (gpu_resident.rs:2224)
  — NOT merely defended-in-depth. What REMAINS ledgered is only the
  attention-GEOMETRY and expert-COUNT load-site wiring (both convert-gate-contained
  above): a regression removing those load calls would only matter for a hostile
  hand-built LBC, since a converter-produced artifact is already rejected at the
  mutation-pinned convert gate. Backstop for those is point-in-time only — e2e load
  logs prove TODAY's wiring; the coherence-gated production checklist does NOT catch
  guard REMOVAL (a good model trips no guard, so it loads coherently either way).
  A type-state refactor making `LayerWeightsGpu` unconstructible without validation
  would close the remainder; deferred as disproportionate for convert-contained
  guards.
- **READER-FROMBYTES-OVERFLOW (pre-existing, debug-only)** — `LbcFile::from_bytes`
  has a tokenizer-range overflow that can `panic` in debug builds; a SEPARATE path
  from `open()` (whose post-`checked_add` `needed:` provably cannot overflow after
  the round-8 hardening). Surfaced by codex during the round-8 full-diff review;
  the diff does not touch `from_bytes`. Release builds wrap; harden with a
  `checked_add` when `from_bytes` is next touched.
- **READER-OPEN-READTOEND-FALLBACK (pre-existing)** — `parse_lbc`/reader.rs:82: on
  a `checked_*` overflow the size falls to `usize::MAX` → the `read_to_end`
  fallback, which reads the ACTUAL on-disk file (bounded by real file length), not
  the header's claimed size — no header-driven amplification. Predates round 8;
  the round-8 `checked_*` strengthened it (the old release-wrap could under-read and
  mask the parse error). Docstring caveat only; noted by codex, not a regression.
- **DISK-RESUME-RECURRENT · session-resume recurrent layout
  deserialization** — `RecurrentState::zeroed` in `kv/disk.rs` allocates
  from on-disk GDN dims BEFORE the CRC check and before comparing them
  to the live layout (CUDA's `gdn_layout()` returns None and skips the
  comparison entirely), and
  `--session-resume` accepts an arbitrary path with keyless CRC32 (not
  a trust boundary); CUDA `gdn_layout()` returns None and skips the
  layout comparison entirely. Hostile session files only; fix when the
  resume path is next touched.
- **F32-GATE-STREAMING · mode-level over-reject, contained by F1** —
  the GDN F32-gate load guard (GDN-GATE-F32 above) is backend-wide, but
  the Metal serial streaming decoder always writes `normed_buf`, so the
  F32-gate-next-to-fused-QKV combination it rejects would be valid
  there. Streaming decode is unreachable while `caps()` hardcodes
  gpu_resident=true (the caps() entry below), so no live impact; if
  streaming ever becomes reachable, scope the guard to the resident
  path. (Round-7 report §3 item 5, accepted as stated.)
- **PROVIDER-SYNC-F32-FALLBACK** — `weight/provider_sync.rs:433-437`
  (and the sibling `read_output_proj_global`) reinterpret an
  unrecognized-format global buffer as F32 ("backward compat") instead
  of erroring — when the byte length is four-byte-aligned; a
  NON-aligned unrecognized buffer (e.g. a 256-element Q3_K global at
  110 bytes) PANICS in `bytes_to_f32`'s alignment assert rather than
  failing cleanly (r6 review find; hand-built artifacts only).
  SHARPER SUB-CASE (2026-08-31, r5 review find): the Q4_0 detection
  branches are length-only (the F16/Bf16 arms and the output-proj Q6_K
  arm consult the header), and Q4_K's packed length collides EXACTLY
  with Q4_0's
  (144 bytes per 256 elements — the repo's own `ggml_byte_size_for`
  test asserts both), so a hand-built Q4_K global would be
  misclassified as Q4_0 and forwarded RAW through the setters —
  silently wrong dequantization, not an F32 fallback. The full
  fixed-layout sweep yields three equal-length pairs (Q4_0/Q4_K 144,
  Q5_0/Q5_K 176, F16/Bf16 512 per 256 elements), but only Q4_0/Q4_K
  changes raw-forwarding behavior: F16/Bf16 are header-disambiguated
  and Q5_0/Q5_K both miss every recognized detector length.
  Containment: the converter never emits a COLLIDING K-quant global —
  K-quant heads requantize to Q8_0/Q4_0 by default, and non-Metal
  fidelity mode (`LUMEN_CONVERT_SOURCE_FIDELITY=1` or
  `LUMEN_CONVERT_KEEP_Q6K_OUTPUT=1`) may preserve a Q6_K head, whose
  210-byte block collides with nothing — so the raw-forwarding path
  needs a hand-built LBC. Fix when touched: scheme-aware or
  fail-closed detectors (covering the Q5_0/Q5_K pair while there) + a
  Q4_K/Q4_0 collision test.
  Round-7 report §4 A2(c) second half, accepted: unchanged since
  v0.18.0, contained by the conversion-side scheme gating; a fail-closed
  arm is the fix when touched.
- **PLAIN-F32-GLOBAL-CAP** — the u32 element/byte caps cover the raw
  quantized global paths only (EMBED-U32-CAP above); plain-F32 globals
  upload via a separate path with no cap on either backend. Containment
  is thinner than the raw paths' 1.6x margin — and absent for the
  largest shipped geometry: a dequantized 27B-scale F32 global
  (248320 x 5120 x 4 = 5,085,593,600 bytes) is already 1.18x PAST the
  2^32 byte boundary on this uncapped path (reachable via
  full-dequantize conversion); the 9B-scale one (248320 x 4096 x 4 =
  4,068,474,880) sits ~5% below it. Extending the cap to the F32 upload
  path is the fix when touched. Round-7 report §4 A2(f), accepted as
  stated.

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
  cannot produce it). As of R8-ISSUE-6, `LayerIndex::validate` now runs at
  parse and bounds every expert slice's offset+length against the layer
  blob — so a slice that lies about its length past the blob is already
  rejected at load. What remains is the narrower case of a length that
  fits the blob but overruns the dims the CUDA kernel derives; that guard
  still needs dims plumbed into `build_moe_meta` — do with the next CUDA
  MoE change.
- **QKV-SHAPE · latent class members (b)-(e) remain, (f) narrowed; instance + (a) closed round 7** — row-count validation now
  enforced at load on BOTH backends: Metal `validate_attention_dims`
  (wq == q_dim / 2*q_dim by `attn_q_norm` presence / declared-GDN
  in-projection rows; wk/wv == kv_dim on full attention, empty on GDN;
  dims passed lock-free via `cached_attn_dims`, honoring the deadlock
  constraint) at all three load paths; CUDA `validate_projection_geometry`
  in `upload_projection_tensor` with presence-aware allowed row counts,
  covering Q5_0 and all five K-quants (which reach CUDA verbatim under
  `--target cuda`) with non-block-multiple widths failing closed, plus
  `validate_mandatory_presence` closing zero-length suppression for wq,
  full-attention wk/wv/wo, and the dense FFN trio (MoE = router AND
  non-empty experts; half-declared MoE is rejected).
  GDN expectations use declared header dims when present, else the
  documented QWEN35_9B compatibility default — the same default the
  kernels dispatch on, so headerless 9B-era artifacts still load; a
  defaulted mismatch fails with observed-vs-expected plus a NOTE naming
  the missing `ssm.*` keys. Canonical repro artifacts (uniform,
  contiguous, fused-geometry wq; zero-length wk; both loaded silently on
  v0.18.0) now rejected: `evidence-v0190/c2{b,c}.{gguf,lbc}` +
  `gen_c2_variants.py`. Remaining class members, recorded honestly:
  (a) CLOSED in Category 2: both converters now validate full-attention
  attn_q/k/v dims against the same presence rule the loaders use
  (TensorShapeMismatch at convert; canonical-C2 sources refuse to
  convert), refuse Q+gate+bias and missing-both-pre-norms sources
  (UnsupportedModel), and force F16 GDN qkv/gate pairs to Q8_0 on the
  Metal target (the prefill has no F16 arm; Bf16 passes) — probes
  c2q/c2b/c2n1 refuse at convert, c2f16 converts forced-Q8; real 27B
  and 9B GGUFs still convert successfully under the gate and serve
  coherently — with the planner's normal transformations, and no
  byte-identity check against pre-fix output was run
  (evidence-v0190/convert-*-post*.log);
  (b) narrowed in round 8: Metal's `validate_attention_dims` now runs
  `validate_mandatory_presence` (wo/FFN-trio presence, MoE
  half-declaration, bias policy) and validates wo geometry
  (hidden rows x q_dim width) at all three load paths — the dense FFN
  trio still has no Metal GEOMETRY check
  (CUDA covers it; `gpu_resident.rs:~1828` even derives a qmv repack
  row count FROM the buffer); (c) GDN dims are pinned through their
  aggregates, not their decomposition (narrowed twice 2026-08-31:
  `ssm_conv1d` IS cross-checked against the aggregate `hp.gdn` geometry
  by `validate_gdn_conv1d` on both loaders and at the convert gate, and
  a present `attn_gate` is checked against v_heads x head_dim on CUDA
  and at the gate — so {32,16,128} vs {48,8,128}, same row sum but
  different gate product, IS distinguishable there; the true residual
  is decompositions sharing BOTH aggregates, e.g. {32,16,128} vs
  {64,32,64} — row sum 8192 and gate product 4096 alike — plus Metal
  loading and gate-absent artifacts, where only the sum is pinned);
  (d) the CUDA LOADER has no non-CtInt4G32 `ssm_out` geometry check, but the
  convert gate now validates `ssm_out` (hidden rows x gdn_v_dim width) for
  EVERY target — round 8 hoisted it out of the Metal-only branch into the [erratum 2026-09-03: an addition, not a hoist — see REMEDIATION-HISTORY]
  universal projection checks in `validate_layer_plan`
  (`validate_projection_geometry("ssm_out", …)`, mutation-pinned by
  `generic_convert_rejects_wrong_ssm_out_geometry`), so a wrong-geometry
  `--target generic`/CUDA artifact is rejected at convert rather than read at
  the wrong geometry downstream; (e) byte-length checks are inherently blind
  to transposition/permutation and to the F32-vs-2xF16 length collision —
  not closable this way, stated as a limit; (f) narrowed in Category 2: the real MoE GGUF's
  attention geometry is now verified at the header level (16 MiB ranged
  fetch of bartowski Q4_0; attn_q [2048,8192] = 2*q_dim WITH q_norm,
  k/v = kv_dim — `evidence-v0190/moe-geometry-verification.md`); expert
  and FFN tensors remain covered by converter-source reasoning plus the
  CI Modal matrix only. The old entry's member (c),
  `metal/moe.rs:2554` option-A, is covered transitively (all its weight
  paths pass a validated load point) and the route is documented dead
  outside tests.
- **Pre-existing lint (not CI-gated)** — `runtime_defaults.rs:1940` trips
  clippy `items_after_test_module` under `--all-targets` (CI's gate,
  `ci.yml:50`, does not use that flag). Out of round-7 scope; sweep when
  touched.
- **GDN fixture ssm_* sizes still hidden-derived (attention geometry fixed round 7)** —
  `generate_test_model_q8_0_gdn` wrote GDN layers with separate non-empty
  wk/wv/wo and no declared GDN dims (so `gdn_dims()` silently defaulted
  to 9B geometry). The round-7 loader validation caught it; the fixture's
  ATTENTION geometry now mirrors the converter contract (fused wq, empty
  wk/wv/wo, declared dims consistent with q_dim), and since round 8 its
  ssm_out is v-dim-consistent (hidden x gdn_v_dim — the new Metal
  ssm_out rule forced it). Its other ssm_* sizes remain
  hidden/head_dim-derived (no rule reads them), and it
  still omits `attn_gate` — the pre-existing generator gap below.
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
