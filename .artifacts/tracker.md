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
  `GgmlType`s are deleted, retiring the entire divergence class two
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
  that turned out already-PID-safe and were reverted (counters
  disambiguate threads within a process, never across processes).
  Round-final (codex fresh pass): PIDs are NOT unique across PID
  namespaces — two containers sharing a cache volume can both be PID 1,
  resurrecting the truncation race with a silently partial FINAL file.
  Closed with exclusive creation: staging is `{filename}.{pid}-{nonce}`
  opened with `create_new` (O_EXCL) and collision-retried — the
  filesystem, not the name, is the arbiter (no lockfile needed); same
  treatment in the bench model-cache generator. Reclamation parses both
  name forms and moved back ABOVE the cache-hit return (a SIGKILLed
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
  double-PID test hunks reverted. Counts corrected: 5 counter-only src
  siblings (incl. convert.rs/sharded.rs), 23 integration-test sites of
  which 2 were already PID-safe (reverted). Inherent residual, safe
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
  another user, keep; a `kill` subprocess also printed to the console),
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
  opened write-only, so the same-fd hash read returned EBADF and every
  cold download would have failed (fixed with .read(true); regression
  test exclusive_staging_write_then_hash_via_same_fd covers the exact
  create/write/seek/hash flow); and (ii) the Drop guard deleting by
  pathname could remove a stranger's reused-name file on our error
  paths — the guard now captures our inode at creation and removes only
  when the path still resolves to it. r7 hardening: identity is
  (dev, ino) not ino alone, the fd stays open through the rename
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

- **F32-GATE-STREAMING · mode-level over-reject, contained by F1** —
  the GDN F32-gate load guard (GDN-GATE-F32 above) is backend-wide, but
  the Metal serial streaming decoder always writes `normed_buf`, so the
  F32-gate-next-to-fused-QKV combination it rejects would be valid
  there. Streaming decode is unreachable while `caps()` hardcodes
  gpu_resident=true (the caps() entry below), so no live impact; if
  streaming ever becomes reachable, scope the guard to the resident
  path. (Round-7 report §3 item 5, accepted as stated.)
- **PROVIDER-SYNC-F32-FALLBACK** — `weight/provider_sync.rs:433-437`
  (and the sibling `read_output_proj_global`) reinterpret any
  unrecognized-format global buffer as F32 ("backward compat") instead
  of erroring; a wrong-sized buffer is silently read as F32 data.
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
  cannot produce it). `LayerIndex::validate` would bound offset+length but
  has zero production call sites, so it is no mitigation today. Guard
  needs dims plumbed into `build_moe_meta` — do with the next CUDA MoE
  change.
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
  (b) Metal checks wq/wk/wv
  only — wo and the dense FFN trio have no Metal geometry/presence check
  (CUDA covers them; `gpu_resident.rs:~1828` even derives a qmv repack
  row count FROM the buffer); (c) GDN dims are pinned only through their
  SUM — {32,16,128} and {48,8,128} both give 8192 rows, and no ssm_*
  tensor is cross-checked against `hp.gdn`; (d) non-CtInt4G32 `ssm_out`
  has no CUDA geometry check; (e) byte-length checks are inherently blind
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
  wk/wv/wo, declared dims consistent with q_dim). Its ssm_* tensor sizes
  remain hidden/head_dim-derived rather than GDN-dim-consistent, and it
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
