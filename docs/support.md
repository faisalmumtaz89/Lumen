# Lumen Model Support Matrix

This page is the source of truth for what is currently **verified-against-llama.cpp** end-to-end. Lumen runs LLM inference in Rust for Apple Silicon and NVIDIA CUDA; v1 (current) verifies the Qwen3.5 family plus the Qwen3.6-27B and Qwen3.8-27B dense models; additional model families are planned. Architectures outside the v1 set (llama, mistral, qwen2, phi, gemma) are currently rejected at GGUF conversion because they have not yet been gated end-to-end on this runtime.

## What is verified

Each backend (CUDA + Metal) is validated end-to-end against llama.cpp per the matrices below; validation dates are per row, and cells marked N/A are excluded by capacity policy rather than validated.

### CUDA (NVIDIA, compute capability 8.0+)

The ratios in this table were measured on an A100-80GB (27B-class BF16 cells on H100 — sm_80 routes BF16 through F32 and cannot hold them; the MoE BF16 record is also H100-measured); see [`bench/RESULTS.md`](../bench/RESULTS.md) for the rig and full numbers.

> **Scope of the current release's verification.** Decode attention now runs one kernel on every
> model and context (see the CHANGELOG), and that change was verified on an RTX 5090 (compute
> capability 12.0) only: Qwen3.8-27B Q4_0 (session boards at six context shapes, the long-context
> quality record on both KV stores), Qwen3.5-9B Q4_0 (a paired gate against the previous route at
> three context lengths, the quality record) and Qwen3.5-MoE-35B-A3B Q4_0 (the quality record).
> The decode-attention route changed for every CUDA model in this release (the MoE and the
> Q4_0 dense models on Ampere/Hopper took the tiled kernel before; every model takes the one
> kernel now) and was measured on the RTX 5090 only. Every A100/H100 ratio in the table below is
> a **historical record** taken on an earlier release's decode-attention routes; it was not
> re-measured on this release, and the ratio it states should be read as that release's. What
> this release verified on the RTX 5090:

| Model | Quant | RTX 5090 decode, tok/s (2026-09-11) | Verified by |
|-------|-------|------:|---|
| Qwen3.8-27B dense | Q4_0 | 85.4 / 83.9 / 81.8 / 78.6 at 1,024 / 3,072 / 6,144 / 12,288 tokens of context (85.7 / 84.7 / 83.3 / 81.3 on the 16-bit KV store) | session boards (five runs, CV ≤ 0.2 %) on both stores; DET-001 50/50 on both stores at 1,335 and 6,153 keys, equal to the registered references; the long-context quality record (5 contexts, both stores) bit-identical to the previous build's on every item |
| Qwen3.5-9B dense | Q4_0 | 171.6 / 167.1 / 159.9 at 3,072 / 6,144 / 12,288 | a paired gate against the previous route (+3.8 / +5.9 / +10.9 %, two runs); DET-001 50/50 on both stores identical to the previous route's at 6,153 keys; in the long-context quality record (42 items) the F32 store parts from the previous build's on six items, every one inside the declared near-tie margin (the same runner-up on both builds, margins ≤ 0.25 logits); the 16-bit store parts from the previous build's 16-bit route once outside that margin, at 12k keys (margins 0.034 / 0.015), adjudicated by a float64 replay of the real activations at the parting positions (this kernel's half route 2.6e-7 relative L2 against the previous half route's 6.1e-7, each against a float64 reference on the half-rounded inputs it read) |
| Qwen3.5-MoE-35B-A3B | Q4_0 | 213.6 / 208.3 / 200.6 at 3,072 / 6,144 / 12,288 (the previous route: 80.2 / 49.8 / 28.3) | a paired gate against the previous route (+166 / +318 / +599 %, its kernel arm at the budget-derived target of 256 that preceded the shipped 128; the rates in the previous column are the shipped target's, from the policy check that set it); DET-001 50/50 on both stores, both identical to the previous route's digest at 6,153 keys; the long-context quality record (48 items at four contexts, two of them past the one-tile bound) fails its strict first-flip rule on one prompt on the F32 store (token 25: the runner-up is the same token on both builds at margins of 0.033 and 0.024 logits but is not the other's choice) and on two prompts of the 16-bit store against this build's F32 store (the same prompt at token 25, and one past the one-tile bound at token 245, a mutual runner-up at a margin of 0.29 logits); the 16-bit store never parts from the previous build's 16-bit route outside the near-tie margin. Both partings adjudicated by a float64 replay of the real activations at the parting positions: this kernel 4.7e-7 and 3.3e-7 relative L2 on the F32 store against the previous route's 6.3e-7 and 9.3e-7, and 2.9e-7 and 3.9e-7 on the 16-bit store against the previous 16-bit route's 6.7e-7 and 6.9e-7, each store against a float64 reference on the inputs it read |

**Status reflects functional verification** (correctness, robustness, and determinism gates), not decode-speed parity: a cell can be production-ready while decoding slower than llama.cpp on the same hardware — the ratio column carries the observed record. Cells below ~0.95× are open performance targets.

> **Evidence policy (both matrices below):** ratios marked "retained
> record" have measurement artifacts on disk that reproduce the digit; a
> 2026-08-30 audit of 70 published ratio derivations found every
> artifact-backed figure exact or rounded conservatively (zero rounded in
> Lumen's favor). Rows WITHOUT a "retained" label state observed
> historical measurements whose raw artifacts were not kept — the same
> review separately counted roughly seventeen such ratios, out of about
> twenty-six published, across both matrices; where a board does exist
> for one of them it reads more favourably than the published figure
> (stale-conservative, not slanted). Figures whose artifacts were found
> wrong were withdrawn rather than restated. Three *previously* published
> figures rounded toward Lumen (0.892, 0.727, 1.15 for true 0.891, 0.726,
> 1.145); all three were corrected (0.891, 0.726) or withdrawn (1.15×) in
> earlier rounds — which is why the seventy re-derived current rows
> contain none — and the two retained files still carrying old digits now
> bear dated errata.

| Model | Quant | Status | × llama.cpp decode (canonical) | Notes |
|-------|-------|--------|------:|---|
| Qwen3.5-9B dense | Q8_0 | Production-ready | **0.970× llama.cpp** (retained co-located A100 record: 114.1 vs 117.6) | All robustness and correctness gates pass |
| Qwen3.5-9B dense | Q4_0 | Production-ready | **0.979× llama.cpp** (retained co-located A100 record: 146.6 vs 149.8) | All functional gates pass |
| Qwen3.5-9B dense | BF16 | Production-ready | 0.726× llama.cpp (retained same-GPU H100 record, separate per-engine batteries: 106.5 vs 146.6; earlier 0.93–0.94× not retained) | Highest-precision |
| Qwen3.5-MoE-35B-A3B | Q8_0 | Production-ready (functional) | 0.567× llama.cpp (retained co-located A100 record: 79.2 vs 139.7) | MoE_Q8_SPLIT=OFF default validated |
| Qwen3.5-MoE-35B-A3B | Q4_0 | Production-ready (functional) | 0.598× llama.cpp (retained co-located A100 record: 93.6 vs 156.5) | Same MoE setup path as Q8 MoE |
| Qwen3.5-MoE-35B-A3B | BF16 | Production-ready with caveats | 0.575× llama.cpp (retained same-GPU H100 record, separate per-engine batteries: 104.1 vs 181.1; the previously published 0.902× has no retained artifact) | Requires a dedicated H100/H200-class GPU (peak 72,475 MiB ≈ 70.8 GiB, H100-measured; the A100-80GB fit is unverified and A100 decode unmeasured) |
| Qwen3.6-27B dense | Q8_0 | Production-ready | 0.891× llama.cpp (retained 2026-07-16 co-located A100 record: 35.08 vs 39.35; see bench/RESULTS.md) | All quality gates pristine (2026-06-11 checklist; earlier 0.85× not artifact-retained) |
| Qwen3.6-27B dense | Q4_0 | Production-ready | 0.820× llama.cpp (retained 2026-07-16 co-located A100 record: 45.34 vs 55.32; see bench/RESULTS.md) | All quality gates pristine (earlier 0.66× not artifact-retained) |
| Qwen3.6-27B dense | BF16 | Production-ready (H100) | 0.818× llama.cpp (retained 2026-07-16 same-GPU H100 record, separate per-engine batteries: 40.4 vs 49.4; earlier 0.89× not artifact-retained) | All quality gates pass; shares the deterministic stray-first-token issue noted on the Qwen3.8-27B BF16 row |
| Qwen3.8-27B dense | Q8_0 | Production-ready | **1.02× llama.cpp** | All quality gates pristine + DET-001 50/50 (2026-08-14, A100; llama.cpp b10032 co-located, same GGUF) |
| Qwen3.8-27B dense | Q4_0 | Production-ready | 0.93× llama.cpp | All quality gates pristine + DET-001 50/50 (2026-08-14, A100) |
| Qwen3.8-27B dense | BF16 | Production-ready (H100 / sm_90) | 0.87× llama.cpp | All quality gates pass + DET-001 50/50 (2026-08-14, H100 — sm_90 native BF16). Known issue: a deterministic stray first token at BF16, shared with Qwen3.6-27B BF16 (tracked prefill-numerics issue) |
| Qwen3.8-27B dense | CtInt4G32 (HF import) | Production-ready, compatibility cell (SM80+) | — (no llama.cpp equivalent format) | Serves the community compressed-tensors INT4 g32 checkpoint byte-exactly (`lumen convert --from-hf`); quality + DET-001 50/50 verified on A100. W4A8 dp4a route — slower than engines with W4A16 4-bit kernels on the same bytes |

### Metal (Apple Silicon, M-series)

Benchmarked on an M3 Ultra; see [`bench/RESULTS.md`](../bench/RESULTS.md) for the rig and full numbers.

> The evidence policy stated above the CUDA matrix applies to this matrix as well.

| Model | Quant | Status | Decode × llama.cpp | Prefill × llama.cpp | Notes |
|-------|-------|--------|------:|------:|---|
| Qwen3.5-9B dense | Q8_0 | Production-ready (default) | **0.98×** | 0.95× | Cleared 0.9× decode gate |
| Qwen3.5-9B dense | Q4_0 | Production-ready | **1.02×** / **1.17×** (beats llama.cpp) | 0.88× | Below 0.9× prefill (structural) |
| Qwen3.5-9B dense | BF16 | Production-ready (functional) | 0.83× | 0.66× (up from 0.31×) | mmap zero-copy load (the default on Metal) |
| Qwen3.5-MoE-35B-A3B | Q8_0 | Production-ready (functional) | not retained | not retained | mmap zero-copy load (the default on Metal). Earlier 0.21×/0.09× ratios have no retained artifact (the cited bench records `none` for these cells — llama-bench 8680 could not load this arch); MoE perf on Metal is a known optimization target |
| Qwen3.5-MoE-35B-A3B | Q4_0 | Production-ready (functional) | not retained | not retained | Same mmap default; earlier 0.18×/0.08× ratios have no retained artifact; same MoE-perf caveat |
| Qwen3.6-27B dense | Q8_0 | Production-ready | **1.03× (beats llama.cpp)** | 0.86× | All quality gates pristine (2026-06-11) |
| Qwen3.6-27B dense | Q4_0 | Production-ready | 0.99× | 0.82× | All quality gates pristine |
| Qwen3.6-27B dense | BF16 | N/A on Metal | — | — | Same ~50 GiB capacity-margin arithmetic as the Qwen3.8-27B BF16 row below; validated on CUDA H100 instead |
| Qwen3.8-27B dense | Q8_0 | Production-ready (functional) | withdrawn — under re-measurement (the 2026-08-14 1.15× predates a machine-level bimodal-speed finding; audited re-runs read below it and are quarantined until the trigger is isolated) | — | All quality gates pristine + DET-001 50/50 (2026-08-14, M3 Ultra; llama.cpp b10032, same GGUF); prefill row pending |
| Qwen3.8-27B dense | Q4_0 | Production-ready | **1.30× (beats llama.cpp)** | — | All quality gates pristine + DET-001 50/50 (2026-08-14); ratio from the 2026-08-24 board (40.17 vs 30.57 = 1.314, published conservatively): same GGUF, sequential per-engine runs on the strictly-serial M3 Ultra — not co-resident processes; prefill row pending |
| Qwen3.8-27B dense | BF16 | N/A on Metal | — | — | ~50 GiB weights sit inside the capacity margin policy on the 96 GB test rig (same arithmetic as Qwen3.6-27B BF16); validated on CUDA H100 instead |

## What is not (yet) supported

| Class | Status | Why |
|---|---|---|
| llama / mistral / qwen2 / phi / gemma architectures | Currently rejected at conversion | Not yet on the verified-against-llama.cpp matrix. v1 scope decision; planned for future model-family releases. |
| A CUDA model outside the decode-attention kernel's domain (more than 8 query heads per KV head, or a head dimension other than 128 or 256) | Refused at CUDA init, with the shape named | The one decode-attention kernel is compiled per model for its group size and head dimension; every model in the registry is inside the domain. Such a model runs on the CPU or Metal backends |
| NVIDIA hardware below compute capability 8.0 (pre-Ampere) | Untested | The kernels are compiled at load for whatever target NVRTC offers; the current release is verified on an RTX 5090 (12.0), earlier releases on A100 / H100 (8.0 / 9.0); older cards may compile but are not gated |
| Apple Silicon outside the M-series tested configuration | Untested | The published Metal benchmarks were measured on an M3 Ultra |
| K-quants (Q4_K, Q5_K, Q6_K, Q2_K, Q3_K) at runtime | Backend-dependent at GGUF→LBC import. `--target metal` upcasts K-quant (and legacy Q5_0) layer tensors to Q8_0 and re-quantizes Q4_1 to Q4_0, unless an explicit `--requant`/`--dequantize` overrides. Generic/CUDA conversions carry K-quant layer planes verbatim and CUDA dequantizes them to F32 at load — except MoE shared-expert planes, which are always requantized to Q4_0. Two role-specific tensors are requantized by default and preserved with `LUMEN_CONVERT_SOURCE_FIDELITY=1` for dedicated CUDA kernels: a Q5_K `ssm_out` and a Q6_K output head | No general K-quant matmul kernels; CUDA kernels exist only for the two preserved roles |
| MXFP4 at runtime | No LBC representation: required MXFP4 layer tensors are rejected at conversion; optional tensors and MoE shared-expert planes are dequantized (to F32, or Q4_0 for shared-expert gate/up) | No MXFP4 runtime kernels |
| Batched serving (batch > 1 per request) | Not implemented | Single-stream decode is the optimization target |
| Speculative decoding / MTP heads | Filtered at conversion | Out of scope |

## Configurations that pass-or-fail at runtime

The full registry (canonical) is at `model_registry.toml`. `lumen models` prints the live set including disk-cached LBCs. Unsupported `(model, quant)` combinations are rejected with a clear error listing available alternatives.

## Reference numbers

The status tables above give each configuration's ratio against llama.cpp. Raw throughput (tok/s) for Lumen and the llama.cpp baselines it is measured against — including the cold-load vs warm-state split and the rigs the numbers were captured on — is in [`bench/RESULTS.md`](../bench/RESULTS.md). Benchmark methodology: [`bench/METHODOLOGY.md`](../bench/METHODOLOGY.md).
