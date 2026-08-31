# Changelog

All notable changes to Lumen are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once
`0.1.0` is published.

## [Unreleased]

## [0.20.0] — 2026-08-31

### Fixed

- **A missing GDN conv-weight length check was an out-of-bounds read**:
  both backends index `qkv_rows x conv_kernel` F32 values from
  `ssm_conv1d` without consulting the slice length — a short tensor was
  an out-of-bounds device read on CUDA and a silent read of adjacent
  tensor bytes on Metal. An exact-length rule is now enforced at both
  loaders and at conversion. Real sources carry exactly `qkv_rows` conv
  channels — 9B and 27B verified end-to-end by conversion and serving
  under the new rule, MoE from GGUF header dims; the rule immediately
  caught incoherent test fixtures, including this campaign's own earlier
  probe artifacts.
- **GGUF conversion no longer emits artifacts the target backend's
  loader would refuse**: the loaders' validation rules moved to a shared
  module (`lumen_format::serving_rules`) that both backends call, and
  conversion now runs the same rules over every planned layer — on the
  planned quantization schemes, after requant/upcast decisions — before
  any byte is written. Sources that would produce a loader-refused
  artifact fail at conversion with the loader's own message. Covered
  classes: mandatory tensor presence, attention/FFN/conv geometry,
  zero-length optional and expert tensors, missing FFN pre-norms,
  expert-bank uniformity, Q+gate bias sets, and (Metal target) the
  scheme allowlist and pairing rules. Known limits: the Hugging Face
  import path does not run the gate; `--target generic` applies only the
  CUDA rule set, so such artifacts can still be refused by the Metal
  loader; the gate validates the planned layout rather than the written
  bytes; and `CtInt4G32` slices skip geometry validation.
- **One QKV-bias policy on both backends**: Q+gate attention layers
  (per-head Q/K norms) reject bias sets everywhere — previously CUDA
  served them through an unfused path while Metal refused the same file.
  Biased attention without per-head norms remains served.
- **The GDN quantization pair force covers F16 and F32-split sources**
  (the Metal prefill has no F16 arm; a lone F32 gate reads a buffer only
  the F32 QKV route writes), and one load-error message recommended a
  flag that does not exist (`--quant bf16`) — it now names the real
  command.
## [0.19.0] — 2026-08-31

### Fixed

- **Attention projections are validated dimensionally at load** (both
  backends): the v0.18.0 guard checked quant uniformity and contiguity but
  never dimensions, so a uniform, contiguous `wq` with fused-Q+gate
  geometry on a full-attention layer without per-head norms was still
  loaded and generated rather than being rejected — and a zero-length
  `wk` suppressed
  the guard entirely. Loaders now require the exact packed byte length the
  dispatch implies: `wq` = GDN in-projection rows on GDN layers (Metal
  additionally requires `wk`/`wv` empty there), else `2*q_dim` with
  per-head norms, else `q_dim`; `wk`/`wv` = exactly `kv_dim` rows on
  full attention. Kernels derive dimensions from hyperparams, so byte
  length is the load-time observable for this class of wrong-geometry
  artifact.
- **CUDA projection geometry covers every fixed-layout scheme** — the
  enforcement its doc contract already claimed: Q5_0 and all five
  K-quants (which reach CUDA verbatim under `--target cuda`) now get
  exact row checks; non-block-multiple widths and `in_dim 0` fail closed
  instead of skipping.
- **Zero-length mandatory tensors are rejected** (CUDA): `wq` on every
  layer, `wk`/`wv`/`wo` on full attention, and the dense FFN trio on
  non-MoE layers — where MoE means router AND a non-empty expert bank;
  a half-declared MoE layer is rejected rather than silently exempted.
- **A missing FFN pre-norm is a load error, not a silent misread** (both
  backends): `ffn_norm` zero-sentinel with `attn_post_norm` absent
  previously produced a present zero-length norm buffer on CUDA and an
  offset-0 F32 misread on Metal. The rejection keys on `attn_post_norm`'s
  absence — never on the zero sentinel, which is legitimate on every
  shipped GDN/MoE layer.
- **GDN validation honors the documented compatibility default**:
  declared header dims when present, else the QWEN35_9B default the
  kernels already dispatch on; a defaulted mismatch errors with a NOTE
  naming the missing `ssm.*` keys.
- The Q8_0 GDN test-model generator violated the converter contract
  (separate non-empty `wk`/`wv`/`wo` on its GDN layer, no declared GDN
  dims); its attention geometry now mirrors the contract.

## [0.18.0] — 2026-08-30

### Fixed

- **The single-launch fused QKV path is guarded at load** (Metal): full
  attention layers without per-head Q/K norms dispatch one uniform-width
  kernel over wq/wk/wv, which requires uniform quantization and
  byte-contiguity — a converted model violating either would previously
  load and generate silently wrong output. That the unguarded path is
  reachable at all was reproduced: a `qwen35`-declared GGUF omitting
  `attn_q_norm` converts cleanly onto it. The loader now rejects such
  layouts with a reconversion remedy.
- **Float shared experts are served correctly end-to-end** (Metal): the
  load guard now admits gate/up pairs uniform in {Q8_0, Q4_0, F16, Bf16,
  F32} with the down projection independent — matching what the kernels
  dispatch — instead of over-rejecting float trios; the fused fallback
  gained its missing Bf16 arm and a memory barrier before SwiGLU
  (numerics checked against a CPU reference; the test encodes on a
  concurrent compute encoder so the barrier is load-bearing there); the
  single-encoder MoE selector routes float gate/up pairs to the non-fused
  fallback instead of an error arm; the raw encoder's float-gate fallback
  gained the full down-quant dispatch (Q8_0/Q4_0 down weights were
  previously read as raw f32 bytes); and the shared-expert down scratch
  buffer is sized `max(hidden_dim, shared_expert_inter_dim)` so the float
  fallback's gate/up intermediate cannot overrun it.
- **GDN load validation closes two silent-corruption classes** (Metal):
  F16 weights on the GDN prefill projections (attn_qkv, attn_gate,
  ssm_out) — which have no F16 kernel arm and were read as f32 — are
  rejected at load, as is an F32 attn_gate next to a non-F32 QKV route
  (the F32-gate decode fallback reads a normalization buffer only the F32
  QKV route writes). The F32 QKV route itself gained the memory barrier
  its concurrent projection cluster was missing.
- **Output-head raw weights are validated per row, not just in total**
  (both backends): the head matvec kernels lay quantization blocks out per
  row, so `hidden_dim` itself must be block-aligned — a total-aligned but
  row-misaligned head previously uploaded cleanly and misindexed every row
  past the first. CUDA additionally validates this for Q6_K heads.
- **Global tensors are capped to the kernels' 32-bit indexing** (both
  backends): embedding and output-head kernels compute element and byte
  offsets in 32-bit; element counts or packed byte lengths past 2^32 are
  now rejected at load instead of wrapping. Metal additionally validates
  raw globals against the scheme's exact packed length (CUDA already did).
- **CUDA per-layer slice validation runs before any GPU work**: the
  zero-length checks are hoisted above MoE metadata table construction at
  preload; the output head and embedding raw-length validation rejects
  non-block-multiple element counts with checked arithmetic.
- **Session temp files are process-unique**: five session/provider staging
  paths gained `std::process::id()` suffixes, so concurrent processes no
  longer clobber each other's session staging files (model-download
  staging still uses fixed names; tracked separately).

## [0.17.0] — 2026-08-29

### Fixed

- **CUDA MoE loading validates the expert bank** the way Metal already does:
  a within-expert gate/up quant split, or any expert whose schemes diverge
  from expert 0's (including the up projection, which was previously never
  checked at all), is rejected at load with the reconversion remedy instead
  of being decoded at the wrong stride.
- **Metal load validation closes the remaining malformed-input classes**:
  zero-length optional tensors (which previously passed every check as
  "absent" and then crashed or misread at dispatch), incomplete QKV bias
  sets on quantized projections (the fused bias kernels silently dropped
  partial sets; F32 projections keep applying partial sets independently),
  and non-F32 biases. CUDA gains the matching non-F32 bias rejection
  instead of silently dropping the tensor.
- **GDN state bookkeeping is atomic**: the layer-to-state map is committed
  only after every allocation succeeds, on both the resident and streaming
  paths, and a mapped-but-unbacked index reports a clean error instead of
  indexing out of bounds.
- **`--metal --no-gpu-resident` no longer crashes on GDN models.** The
  batched prefill path never allocated the GDN recurrent state (index
  panic); allocation now happens on first touch via one shared, idempotent
  path also used by streaming decode, and the resident preload records the
  layer-to-state mapping it always implied. Resident output is
  byte-identical; streaming prefill now works, with decode reporting the
  pre-existing GPU-resident requirement as a clean error. Also cures the
  same crash under `--metal --async`.
- **Metal's shared-expert fused dispatch can no longer silently drop the
  up-projection.** The float arms selected plain matvec shaders while
  binding a fused-kernel buffer; those arms now error instead (such
  shared-expert schemes are rejected at load anyway).
- **CUDA `init` validates raw embedding/head byte lengths** against the
  scheme's block layout, and **prefill rejects out-of-range token ids**
  before any GPU work (previously an out-of-bounds device read via
  `--tokens` or a library caller).
- Malformed GDN hyperparams (`conv_kernel = 0`) are rejected on both the
  resident and streaming allocation paths.

### Changed

- The Metal Qwen3.8-27B Q8_0 decode claim (1.15× llama.cpp) is withdrawn
  pending re-measurement: bracketed audited batteries reproduce a
  Q8-specific deficit with machine state and file paging both excluded.
  The 9B BF16 ratio is corrected to 0.726× (rounding).
- The converter's six SSM tensors are planned and written from one shared
  decision (`SsmForm`), removing the triple-mirrored planner/writer logic
  that had desynced twice. Output is byte-identical (448-case differential
  vs the previous release).

## [0.16.0] — 2026-08-28

### Fixed

- **The converter no longer produces Metal-target files its own loader
  rejects.** Norm tensors are always written F32 (both backends read norm
  weights as F32 and refuse anything else at load; a non-F32-norm source
  GGUF previously converted silently and then failed to load). On the Metal
  target, a GDN layer's `attn_qkv`/`attn_gate` pair is written uniformly:
  when a per-tensor upcast (or a mixed source, including Q8_1) would split
  the pair on the Q8_0 axis, both tensors are written Q8_0, with a notice
  printed. The loader's mismatch remedy now recommends the plain
  re-conversion that actually works for dense and MoE models.
- The registry no longer advertises an F16 quantization no model ships;
  `lumen pull` help text matches.

### Changed

- Metal validation suite metadata records the tested binary's path, sha256,
  and engine commit; the release-gates harness stamps the correct release
  tag inside its cloud container.

## [0.15.0] — 2026-08-28

### Fixed

- **Metal load validation now runs on every execution path and covers the
  tensor classes the dispatch reads with a hardcoded scheme.** The
  layer-quant checks moved into one validator that also runs on streaming /
  non-resident loading (previously unguarded), and gained new checks:
  `ssm_alpha`/`ssm_beta` must be Q8_0 (the Metal GDN gate pipelines read
  them as Q8_0 only); expert banks must share one quant scheme (dispatch
  applies expert 0's schemes to all); shared-expert tensors must form a
  complete gate/up/down set on the fused-kernel quants; and
  norm/router/SSM-scalar tensors must be F32 (every Metal shader reads
  them as F32; CUDA already rejected these at load).
- **Weight-tied models load correctly on Metal's staged (non-mmap) path
  when the output head and embedding are stored in different
  representations.** The tied-head alias now applies only when the two
  representations match; previously a tied BF16 model computed wrong
  logits on that path (the BF16-typed head shader read F32 bytes).
- **`--dequantize` works on dense GDN models again.** The converter's
  planner and writer disagreed on `ssm_alpha`/`ssm_beta` under
  `--dequantize`, aborting the conversion for every real Qwen3.5/3.8 GGUF.
  Non-Metal targets now write true F32 gates (CUDA serves them); the Metal
  target keeps the Q8_0 force so the output stays loadable. Dequantized
  MoE expert banks are served by Metal's per-expert float path.
  **Disclosure:** before v0.15.0, an LBC whose `ssm_alpha`/`ssm_beta` were
  stored F32 loaded and computed **silently wrong output** on Metal (F32
  gate bytes read as Q8_0 blocks). Two conversion paths produced such
  files: `--dequantize --target metal` on a GGUF whose gates were already
  Q8_0, and `LUMEN_CONVERT_SOURCE_FIDELITY=1` on a non-Metal target with
  the F32-source gates real GGUFs ship (most `--dequantize` combinations
  instead aborted at conversion). If you ran either output on Metal with
  an earlier release, re-convert with `--target metal`.

### Changed

- Benchmark and capacity documentation scopes each provenance claim to what
  its artifact records: BF16 battery cells are separate same-GPU H100
  batteries (not co-located), the llama.cpp build/protocol attestation
  applies to the co-located A100 cells only, the A100-80GB BF16-MoE fit is
  stated as unverified (conflicting unretained records), and the
  Qwen3.6-27B CUDA rows carry the retained record (0.891×/0.820×).

## [0.14.0] — 2026-08-27

### Fixed

- **Metal rejects converter-producible mixed-quant tensor pairs at load**
  instead of computing silently wrong output: dense `ffn_gate`/`ffn_up`,
  per-expert gate/up, and (on GDN layers) `attn_qkv`/`attn_gate` splits in
  either direction. All were reachable through the converter's per-tensor
  K-quant upcasts; every rejection names the remedy.
- **`--requant` on MoE models is refused with a clear error.** It previously
  did nothing to layer tensors while stamping the requested scheme into the
  LBC header and printing a success line.
- Stale comments, wrong flag-default claims, and dead rationales corrected
  across the CUDA and converter code.

### Added

- **CUDA raw-BF16 embedding path** (decode + batched prefill, new
  `embed_batch_bf16` kernel): BF16 models no longer pay an F32-materialized
  embedding on CUDA (~2 GB less GPU memory). A100-validated byte-identical
  to the previous path. Metal's two catch-all embed sites gained the
  F16/BF16 arms the other dispatch sites already had.

### Changed

- **Published benchmark and capacity documentation now carries only
  retained-artifact provenance.** The unretained 2026-06-02 dataset is
  marked as such, and the retained 2026-07-16 battery (co-located A100
  quant cells; separate same-GPU H100 BF16 batteries) is the
  published record — notably dense-9B Q8 0.970× / Q4 0.979× vs llama.cpp
  (better than the withdrawn figures) and MoE BF16 0.575× on H100 (the
  earlier 0.902× "production-recommended" claim is withdrawn). VRAM figures
  are restated in MiB with sources.
- CI compile-checks all CUDA test targets, including `lumen-server`'s.

## [0.13.0] — 2026-08-26

### Fixed

- **Accelerate prefill (`--accelerate`) no longer crashes or reads out of
  bounds**: the backend refuses models whose position encoding it cannot
  compute (NeoX-layout, partial-RoPE, RoPE-scaled) at construction, and
  validates every weight plane's quant scheme, byte length, and alignment
  before any `f32` reinterpret. What used to panic (or silently compute on
  out-of-bounds memory) is now a clean, actionable error.
- **CUDA Q5_0 host dequantization** used the wrong nibble/high-bit layout
  (30 of every 32 values corrupted on that path). Fixed to the GGML
  reference layout; latent — no shipped conversion produced Q5_0 CUDA LBCs.
- **`--target metal` conversions of Q5_0 GGUFs** produced files the Metal
  backend then refused to load. Q5_0 layer tensors now upcast to Q8_0 at
  convert time, like the K-quants.
- **Metal resident load rejects dense Q4_1 layer tensors** (only reachable
  from `--target generic` source-fidelity files) instead of silently
  misreading them; the error names the reconversion remedy.
- **Metal-target conversions no longer keep a Q6_K output head** under
  `LUMEN_CONVERT_SOURCE_FIDELITY=1` / `LUMEN_CONVERT_KEEP_Q6K_OUTPUT=1`
  (Metal would serve it through the slow F32-dequant fallback); the head
  is requantized to the fast Q8_0 path, matching every other fidelity gate.
- A racy provider test and a cross-module environment-variable race in the
  test suite (per-module locks replaced by one crate-wide env test lock).

### Changed

- **`lumen run <model>` with a bare registry name** now selects the registry
  default quant when its LBC is cached, or the sole cached quant when
  exactly one exists, instead of always exiting with a list. A bare name
  never starts a download.
- **The installer probes the CUDA userland** (libcuda/libnvrtc/libcublas)
  on NVIDIA hosts and prints an actionable warning when libraries are
  missing, instead of installing binaries that fail on first inference.
- **`THIRD_PARTY_NOTICES.md`** (MLX, llama.cpp/ggml kernel-port notices,
  Qwen chat-template attribution, crate license families) now ships in the
  release tarballs, the Homebrew keg, the installer's `share/doc/lumen`,
  and the Docker image; release staging fails if the file is missing.
- Comments and documentation describing removed machinery (CUDA-graph
  capture, stale K-quant/MXFP4 conversion claims, 30B→35B MoE naming) were
  corrected tree-wide; dead graph-era fields and the phantom
  `LUMEN_GRAPH_DIAGNOSTIC` env var were removed.
- CI now executes the `lumen-runtime` test suite (it was compiled but
  never run) and compile-checks every CUDA test target.

## [0.12.1] — 2026-08-25

### Changed

- `LUMEN_CUDA_CT4_EXACTK` is now **on by default** — the exact-K CtInt4G32
  decode launch (bit-identical, measured +11.3% decode on Qwen3.8-27B ct4)
  ships as the standard route. An explicit `LUMEN_CUDA_CT4_EXACTK=0` restores
  the fixed-256 launch.

## [0.12.0] — 2026-08-25

### Added

- **`LUMEN_CUDA_CT4_EXACTK`** (default OFF): exact-K launch geometry for the
  CtInt4G32 decode matvec. The K=5120 / K=6144 projection shapes carry only
  160 / 192 g32 blocks per row, so the fixed 256-thread kernel idled 37.5% /
  25% of every CTA's warps; with the flag set those shapes launch 160- /
  192-thread kernels whose reduction folds a zero-padded 8-slot array —
  bit-identical output (greedy decode byte-identical to the v0.11.0 release
  anchor), measured **+11.3% decode** on Qwen3.8-27B CtInt4G32 (A100,
  50.5 → 56.7 tok/s). K=17408 keeps the 256-thread kernel.

## [0.11.1] — 2026-08-24

### Fixed

- Merged GGUFs produced by `llama-gguf-split --merge` (which retain the
  `split.*` metadata keys with `split.count = 0`) are now accepted: a zero
  declaration is the merge tool's "no longer split" marker, not a claim of
  zero shards. The authoritative declared count is the first non-zero
  declaration across the set, so a zeroed shard 0 cannot disable the
  sibling-count checks on multi-file shard sets.

## [0.11.0] — 2026-08-21

### Added

- **Hugging Face compressed-tensors import** (`lumen convert --from-hf <dir>`):
  converts pack-quantized INT4 group-32 checkpoints (dense qwen35-family
  models, indexed sharded safetensors) to LBC as the new `CtInt4G32` scheme,
  preserving every quantized value exactly. Served by new CUDA W4A8 dp4a
  decode kernels (SM80+) with an F16-dequant prefill path;
  `LUMEN_CUDA_CT4_DP4A=0` selects an F16 reference route. A donor GGUF of the
  same model supplies tokenizer and hyperparameter metadata only.

### Fixed

- First token after prefill could be wrong for models whose output head is
  served from a raw BF16 buffer (the host finalizer lacked a BF16 arm).
- `<|endoftext|>` is now honored as an alternate end-of-sequence token when
  the vocabulary marks it special, matching the model's declared generation
  config.

## [0.10.0] — 2026-08-20

### Added

- **Split-K decode attention** (`LUMEN_CUDA_ATTN_SPLITK`). A sequence-parallel
  kernel pair (per-head KV walk split across 4 partial CTAs plus an
  online-softmax merge) that lifts the one-CTA-per-head occupancy ceiling of
  the tiled decode-attention kernel on few-head models. Model-aware default:
  ON for Q8_0- and BF16-body dense models; other models keep the tiled route.
- **Q8 split-clone attention coverage** (`LUMEN_CUDA_Q8_SPLIT_ATTN`). The Q8
  raw+split clone pass now also builds SoA siblings for the GDN qkv/gate and
  full-attention Wq/Wk/Wv projections on wide-GDN models, serving them through
  the existing split mmvq kernel family.
- **BF16 decode kernels.** Four new default-ON routes for BF16-body dense
  models, each with a documented kill-switch: a one-row-per-CTA GEMV
  (`LUMEN_CUDA_BF16_NR1`, byte-identical to the previous blocking), a fused
  gate+up+SwiGLU FFN kernel (`LUMEN_CUDA_BF16_FUSED_GLU`, byte-identical to
  the separate sequence), a banked route for the Q8-converted GDN alpha/beta
  projections (`LUMEN_CUDA_BF16_AB_Q8BANK`), and a one-launch residual matvec
  for the attention output projection (`LUMEN_CUDA_BF16_WO_NR1`) that keeps
  the activation in F32. The Q8 split-clone pass also serves the
  converter-forced Q8 GDN `ssm_out` tensors on BF16 models.
- **Fused GDN phase-1/2/3 under F64 recurrence.** A twin of the fused
  conv+gates+L2-norm decode kernel whose normalization accumulates in F64,
  bit-identical to the previous three-launch chain, re-enabling the fusion in
  F64-recurrence mode.
- `LUMEN_CUDA_PROFILE_ATTN_LEAF` diagnostic: per-sub-stage bracketing of the
  full-attention block under the existing profiler.

Quality, sampling, determinism, and tool-calling gates pass on this build for
Q4/Q8/BF16; the Q4 route is byte-identical to the v0.9.0 certified baseline.

## [0.9.0] — 2026-08-18

### Added

- **Source-fidelity conversion** (`LUMEN_CONVERT_SOURCE_FIDELITY=1`). The
  converter preserves tensors in the exact format the source GGUF stores —
  Q6_K output head, K-quant `ssm_out` (Q5_K in Q4_0-preset files), Q4_1
  layer tensors, and F32 GDN `ssm_alpha`/`ssm_beta` gates — instead of
  requantizing them, and the CUDA runtime serves each of these formats
  natively (dedicated split-plane dp4a kernels for the Q6_K head and Q5_K
  `ssm_out`, a Q4_1 dp4a kernel with exact min-term handling, and a banked
  F32 gates kernel that keeps the fused GDN projection route). Per-route
  fall-back switches are documented in `docs/environment-variables.md`.
  Artifacts without these formats dispatch byte-identically to v0.8.0; the
  keeps are excluded on the Metal target.

### Fixed

- **Q6_K dequantization band order.** Both host Q6_K dequant
  implementations read two of the four bands per 128-element half in the
  wrong order relative to the ggml reference — every artifact whose
  `output.weight` was requantized from a Q6_K source (all Q4_0-preset
  conversions, CUDA and Metal targets) shipped an output head with
  misassembled weights. Fixed in both copies with new permutation-sensitive
  reference tests (the prior uniform-block tests could not detect ordering
  errors). **Affected artifacts should be reconverted**; greedy
  determinism baselines for those cells are re-established.
- The CLI and server raw-head allow-lists now admit a Q6_K output head on
  the CUDA backend, and the weight providers recognize it by header quant
  (previously the raw head was silently dropped and misinterpreted).
- The Q4_1 host dequantization used pairwise nibble order instead of ggml's
  de-interleaved layout (dormant until source fidelity made it
  load-bearing).

### Validation

- Full production checklist on the release build: unit suites, quality,
  sampling, determinism (50/50 byte-identical greedy), GQ-014 multi-turn,
  and tool-calling gates pass for the new and existing artifacts; decode
  throughput at parity with v0.8.0.

## [0.8.0] — 2026-08-17

### Changed

- **CUDA FFN decode tail restructuring** (dense GDN models). The down
  projection folds its residual into its own store and writes the layer
  output buffer directly — eliding the separate residual-add launch and the
  per-layer commit copy — and the gate/up projections issue as one banked
  launch off their shared quantized input. ≈ +4% decode wall on
  Qwen3.8-27B Q4_0 and ≈ +2% on Q8_0 (A100). Byte-identical to the v0.7.0
  certified baselines (determinism-hash equality at n=50 on both 27B
  quants; Qwen3.5-9B and the MoE model verified byte-identical end to end;
  quality, sampling, tool-calling and KV-cache-equivalence gates pass; 1-hour
  server soak clean). Both changes carry documented `=0` kill-switches
  respecting `LUMEN_CUDA_LEGACY_DEFAULTS` (see
  `docs/environment-variables.md`).
- Layer-output placement on the CUDA decode path is now a typed contract
  shared by all callers, fixing a pre-existing stale-read on the public
  layer-compute path (dense layers whose result had not been committed could
  be read back one layer stale).
- A one-shot verbose route census names the FFN down-projection dispatch
  branch actually taken (`LUMEN_CUDA_VERBOSE=1`), making dispatch-level
  changes verifiable against the live route.

## [0.7.0] — 2026-08-17

### Changed

- **CUDA decode performance round for the dense GDN models** (Qwen3.8-27B /
  Qwen3.6-27B; Qwen3.5-9B unaffected and verified byte-identical). The
  per-token launch chain shrinks on both the GDN and full-attention
  sections: the GDN prep launches fuse into one kernel, the norm-gate emits
  its own quantized blocks, the residual add folds into the output
  projection, the qkv+gate / alpha+beta / attention wq+wk+wv projections
  each issue as one banked launch off their shared quantized input, the
  six-launch attention prep chain (deinterleave, per-head norms, RoPE, K/V
  appends) fuses into one kernel, and the greedy argmax goes two-phase.
  ≈ +3% decode wall on 27B Q4/Q8. Every route is byte-identical to the
  v0.6.0 certified baselines (determinism-hash equality at n=50, golden
  continuity, quality gates on all five validated cells) and each
  optimization has a documented `=0` kill-switch respecting
  `LUMEN_CUDA_LEGACY_DEFAULTS` (see `docs/environment-variables.md`).
- The full-attention sigmoid gate runs in place unconditionally, removing a
  temp-buffer write and a device copy per full-attention layer.

### Removed

- Experimental probe surface that measured flat or negative (kernel
  variants, an unreachable dispatch path, and instrumentation hooks whose
  target kernels no longer run under shipping defaults), after an
  adversarial two-reviewer evaluation. Implementations remain available
  under `probe/*` git tags.

## [0.6.0] — 2026-08-15

### Added

- **Qwen3.8-27B day-zero support** (`qwen3.8-27b`, Q8_0 / Q4_0 / BF16). The
  model ships the same `qwen35` (GatedDeltaNet, dense) GGUF architecture and
  shapes as Qwen3.6-27B, so all existing 27B kernels and optimizations apply
  directly. The revised Qwen3.8 chat template (`reasoning_effort` system
  preamble, `preserve_thinking`, tojson argument serialization) renders
  byte-identical to HuggingFace's `render_jinja_template` across a 58-shape
  conformance corpus (fixtures + test included). Validated end-to-end on both
  backends: quality suites, DET-001 byte-determinism (50/50 on all five
  runnable cells), multi-turn GQ-014 (8/8 with replay determinism and
  cold-respawn cache-equivalence), and the native tool-calling battery.
  Decode vs llama.cpp b10032 on identical weights: Metal Q4 1.30× / Q8 1.15×
  (M3 Ultra), CUDA Q4 0.93× / Q8 1.02× (A100), BF16 0.87× (H100).
  *(2026-08-29 correction: the Metal Q8 1.15× is withdrawn pending
  re-measurement — see docs/support.md.)*
- `ComputeBackend::reconcile_speculative_tail`: lets the session detect that
  a pipelined decode path already processed the trailing token device-side
  (Metal's lean greedy pipeline keeps one speculative command buffer in
  flight) so warm continuations advance the KV cursor instead of feeding the
  token twice.

### Fixed

- **Warm-session KV lag (all backends).** A completed generation left the
  final sampled token in the transcript without its forward pass
  (`kv.seq_len() == tokens.len() - 1`). A follow-up request that extended
  the session's exact token history then either failed with an HTTP 500
  (`prefill_from` start-position assert, batched path) or silently wrote
  every appended token one KV slot early, producing wrong output
  (short-suffix path). The session now reconciles the un-fed tail at every
  append entry — suffix prefill, warm `extend` (including the server's
  forced-`</think>` injection), and the macOS external-prefill path — and
  reports the repair in `processed_tokens` / prefill timing.
- GDN warm appends now advance token-by-token: the batched prefill resume
  does not continue live recurrent state (h_state / conv_state) mid-stream.
- A sampling-route switch over a live speculative pipeline (e.g. a greedy
  request followed by a temperature request reusing the same session) now
  cold-rebuilds instead of double-advancing GDN recurrent state.
- Quickstart accepted neither `qwen3.6-27b` nor `qwen3.8-27b`; both are now
  in the model catalog, name validation, and help text.

### Quality gates

- DD-SPAM assesses raw units first (unchanged strictness) and applies a
  bounded markdown-table-line exemption — 2–16 pipes per line with
  Unicode-aware cell content or an alignment row — only when the dominant
  unit is table scaffolding, fixing a false FAIL on a legitimate
  table-closing answer while adversarial glyph streams still fire.
- `window_detectors` windows keep their original 256/128 word membership but
  are sliced from the source text by character spans, preserving line
  structure for line-aware detectors.

## [0.5.0] — 2026-07-25

### Added

- **Native Qwen3.5 tool-calling.** The server renders the model's embedded chat
  template and parses its native `<function=…><parameter=…>` tool-call protocol
  (the legacy JSON-in-`<tool_call>` form is still accepted), on both the OpenAI
  (`/v1/chat/completions`) and Anthropic (`/v1/messages`) wire APIs, in default
  and thinking modes.
- OpenAI `stream_options.include_usage`: streaming chat completions emit a final
  usage chunk (prompt/completion token counts) before `[DONE]` when requested.

### Fixed

- Corrected a wrong long-prompt answer caused by low-precision attention scores:
  prefill attention now defaults to exact-F32 (QK^T and P@V). The change is
  decode-neutral — its cost is confined to prefill.
- Anthropic streaming now reports `stop_reason: "tool_use"` on tool-call turns.
- `/v1/completions` returns the decoded model text verbatim (it no longer strips
  tool-call blocks from the raw stream).
- `top_k=1` sampling breaks exact ties to the lowest-index token, matching greedy
  argmax.

### Performance

- CUDA and Metal decode-throughput optimizations across the dense, quantized, and
  MoE paths, plus a byte-identical duplicate-QKV-projection elimination.

### Removed

- Retired dead/no-op diagnostic environment flags and their unused kernels
  (env-flag surface cleanup).

## [0.4.0] — 2026-07-08

### Performance
- Metal decode fast-paths, now unconditional (no flags): two-pass tiled GPU argmax,
  fused attention bookend (deinterleave + norm + RoPE + KV-write in one dispatch),
  fused attention-output glue (gate + output projection + residual), row-interleaved
  dense-FFN gate/up kernel, and a vectorized BF16 decode matvec.
  Measured on M3 Ultra (Qwen3.5-9B): **+7% Q4_0 decode, +13% BF16 decode**,
  +8% long-context decode via a corrected flash-decode threshold (K ≤ 512 now uses
  the exact-MHA kernel, which is faster in that band).

### Fixed
- ThreadPool teardown lost-wakeup race that could hang process exit (and a latent
  use-after-free path when a worker raced a dying pool); test suite now runs with
  zero skips.
- Flash-decode threshold mistune that routed K ∈ [257, 512] contexts to the slower
  kernel.

## [0.3.0] — 2026-06-27

### Added

- `lumen-server` accepts the model as a positional `model:quant` argument (e.g. `lumen-server qwen3.5-9b:q4_0`).

### Changed

- CUDA Q4_0 single-stream decode is ~45% faster — Qwen3.5-9B Q4_0 reaches ~148 tok/s on A100 (**0.97× llama.cpp**, up from 0.64×). The Q4_0 split matvec uses a codegen-locked structure-of-arrays layout (word-load nibble streaming + load-hoist) that stays byte-identical to the reference path, and greedy decode computes the argmax on the GPU. Both are default-on; validated across all supported models × quants on greedy and sampling.
- The one-command installer UX was overhauled for smoother cross-platform setup.

## [0.2.0] — 2026-06-22

### Changed

- Metal Q4_0 decode is unified onto a single, sampling-correct path with the optimized matvec stack as the default (qmv projections, Q4_0 lm_head, projection fusion, concurrent dispatch, and a lean single-queue async pipeline). Greedy decode is byte-deterministic; temperature > 0 runs a GPU temperature sampler, falling back to the CPU sampler for sampling options it does not implement.

### Fixed

- Sampled (temperature > 0) Metal Q4_0 decode could produce degraded output under the previous experimental fast path. The matvec kernels now accumulate in f32 per block and the sampled path uses correct scale-type wiring, so sampled output is fluent and matches the CPU sampler.

### Removed

- The experimental `LUMEN_METAL_Q4_FAST_DECODE` flag (its optimized stack is now the default) and the superseded decode pipeline variants, plus unused feature getters and unreachable shaders.

## [0.1.0] — 2026-06-15

### Production-ready (2026-06-02)

CUDA (NVIDIA, compute capability 8.0+ — Ampere / Hopper; benchmarked on A100-80GB):

- Qwen3.5-9B dense at Q8_0 (**0.91× llama.cpp** decode), Q4_0 (0.64× llama.cpp), BF16 (**0.93–0.94× llama.cpp** decode) *(2026-08-26 correction: artifacts not retained; retained record — Q8 0.970×, Q4 0.979× on A100 co-located, BF16 0.726× on H100 in separate per-engine batteries)*
- Qwen3.5-MoE-35B-A3B at Q8_0 (0.584× llama.cpp), Q4_0 (0.674× llama.cpp), and **BF16 (0.902× llama.cpp, production-recommended)** *(2026-08-26 correction: these ratios have no retained measurement artifacts and the production recommendation is withdrawn — under the retained record BF16 is the lowest MoE ratio; the retained record is 0.567×/0.598× on A100 co-located and BF16 0.575× on H100 in separate per-engine batteries — see bench/RESULTS.md)*
- Validated end-to-end on the full models × quants matrix against llama.cpp

Metal (Apple Silicon, M-series; benchmarked on M3 Ultra):

- Qwen3.5-9B dense at Q8_0 (**0.98× llama.cpp** decode), Q4_0 (**1.17× llama.cpp** decode — beats llama.cpp), BF16 (0.83× llama.cpp decode)
- Qwen3.5-MoE-35B-A3B at Q8_0 + Q4_0 functional (sole provider on Apple Silicon — llama.cpp build 8680 cannot load this architecture; requires `LUMEN_METAL_MMAP_ONLY=1`)
- Validated end-to-end on the full models × quants matrix against llama.cpp

### Features

- Hybrid GDN linear-attention + dense FFN forward pass (L=32 layers, 24 GDN + 8 full-attention)
- MoE forward pass with top-K expert dispatch
- Flash Attention 2 prefill (CUDA, `LUMEN_CUDA_FA2_BLOCKSKIP=1`)
- Long-context decode beyond 40,950 tokens via tiled streaming-softmax (default-on as of `ATTN_DECODE_TILED_DEFAULT_THRESHOLD=0`)
- F16 KV cache (Metal); F32 KV cache (CUDA, CPU)
- Disk-persistent KV cache with eviction policy (`--kv-disk-dir`, `--kv-disk-space-mb`)
- Session save/resume with suffix-prefill cache (a cache-hit turn skips reprocessing the shared prefix; cache-reuse throughput is not part of the published benchmark suite)
- HTTP server (`lumen-server`) with OpenAI + Anthropic wire formats, SSE streaming, template-driven tool-call parser (v1 ships the Qwen3.5 `<tool_call>` marker pattern)
- Per-request reasoning / extended-thinking control (default OFF): OpenAI `enable_thinking` (+ vLLM `chat_template_kwargs.enable_thinking`) with `delta.reasoning_content`, Anthropic `thinking.type` with a `thinking` content block, CLI `--think`, and a separate `reasoning_budget` (distinct from `max_tokens`); `LUMEN_CHAT_ENABLE_THINKING` overrides the default (see `docs/server.md` and `.artifacts/REASONING-CONTROL-DESIGN.md`)
- BPE tokenizer embedded in LBC v3 (no Python at runtime)
- GGUF → LBC converter supporting K-quants (Q4_K, Q5_K, Q6_K, Q2_K, Q3_K) and MXFP4 via dequant on import
- `Configuration precedence: CLI flag > env var > built-in default` documented end-to-end

### Known limitations

- Concurrent CLI bursts (≥4) per GPU are unsupported by design — use `lumen-server`
- Q8 / Q4 prefill × llama.cpp ratios are structurally below 1.0 on the current NVRTC compute_61 stack
- PURE-greedy long-form generation (≥512 tokens) deterministically loops — use `--temperature 0.7` or, on DENSE models, `--repetition-penalty 1.05 --repeat-last-n 64` (when omitted the server/CLI apply a model-aware penalty: 1.05 dense / 1.03 MoE — MoE must stay ≤ 1.03 or arithmetic corrupts)
- BF16 MoE-35B-A3B requires a dedicated 80 GB+ GPU (peak VRAM 72,475 MiB, measured on H100)
- `lumen-server` Authorization / CORS / per-request timeout are not implemented; deploy behind a reverse proxy
- BF16-dense / Q8-MoE / Q4-MoE on Metal require `LUMEN_METAL_MMAP_ONLY=1` (M3 Ultra 96 GB residency budget)

### History

For pre-`0.1.0` commit-level history see the git log. Notable cumulative work:

- Extensive optimization across the CUDA and Metal kernel paths
- Removed 1385 LoC of non-Qwen3.5 architecture support to focus the v1 surface
- Ten-configuration CUDA gold-standard validation against llama.cpp
- Production-readiness audit across 10 operational dimensions (2026-05-29)
- CUDA final validation (2026-06-02): full models × quants matrix verified on A100-80GB
- Metal final validation (2026-06-02): full models × quants matrix verified on M3 Ultra, head-to-head vs llama-bench build 8680

### Documentation

- Documentation pass (2026-06-02): added the `docs/` tree, `CONTRIBUTING.md`, `SECURITY.md`, and `CHANGELOG.md`; fixed README hero numbers and the vLLM prefill ratio (2.29× → 2.62×).

[unreleased]: https://github.com/faisalmumtaz89/Lumen/compare/v0.12.1...HEAD
[0.17.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.12.1...v0.13.0
[0.12.1]: https://github.com/faisalmumtaz89/Lumen/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/faisalmumtaz89/Lumen/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/faisalmumtaz89/Lumen/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/faisalmumtaz89/Lumen/releases/tag/v0.1.0
