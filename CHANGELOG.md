# Changelog

All notable changes to Lumen are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once
`0.1.0` is published.

## [Unreleased]

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

- Qwen3.5-9B dense at Q8_0 (**0.91× llama.cpp** decode), Q4_0 (0.64× llama.cpp), BF16 (**0.93–0.94× llama.cpp** decode)
- Qwen3.5-MoE-35B-A3B (architecture-truthful active params: 30B-A3B) at Q8_0 (0.584× llama.cpp), Q4_0 (0.674× llama.cpp), and **BF16 (0.902× llama.cpp, production-recommended)**
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
- BF16 MoE-30B-A3B requires a dedicated 80 GB+ GPU (peak VRAM 72.4 GB)
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

[unreleased]: https://github.com/faisalmumtaz89/Lumen/compare/v0.12.0...HEAD
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
