# Lumen Model Support Matrix

This page is the source of truth for what is currently **verified-against-llama.cpp** end-to-end. Lumen runs LLM inference in Rust for Apple Silicon and NVIDIA CUDA; v1 (current) verifies the Qwen3.5 family plus the Qwen3.6-27B and Qwen3.8-27B dense models; additional model families are planned. Architectures outside the v1 set (llama, mistral, qwen2, phi, gemma) are currently rejected at GGUF conversion because they have not yet been gated end-to-end on this runtime.

## What is verified

Each backend (CUDA + Metal) is validated end-to-end against llama.cpp per the matrices below; validation dates are per row, and cells marked N/A are excluded by capacity policy rather than validated.

### CUDA (NVIDIA, compute capability 8.0+ — Ampere / Hopper)

Benchmarked on an A100-80GB (27B-class BF16 cells on H100 — sm_80 routes BF16 through F32 and cannot hold them; the MoE BF16 record is also H100-measured); see [`bench/RESULTS.md`](../bench/RESULTS.md) for the rig and full numbers.

**Status reflects functional verification** (correctness, robustness, and determinism gates), not decode-speed parity: a cell can be production-ready while decoding slower than llama.cpp on the same hardware — the ratio column carries the observed record. Cells below ~0.95× are open performance targets.

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

| Model | Quant | Status | Decode × llama.cpp | Prefill × llama.cpp | Notes |
|-------|-------|--------|------:|------:|---|
| Qwen3.5-9B dense | Q8_0 | Production-ready (default) | **0.98×** | 0.95× | Cleared 0.9× decode gate |
| Qwen3.5-9B dense | Q4_0 | Production-ready | **1.02×** / **1.17×** (beats llama.cpp) | 0.88× | Below 0.9× prefill (structural) |
| Qwen3.5-9B dense | BF16 | Production-ready (functional) | 0.83× | 0.66× (up from 0.31×) | mmap zero-copy load (the default on Metal) |
| Qwen3.5-MoE-35B-A3B | Q8_0 | Production-ready (functional) | 0.21× | 0.09× | mmap zero-copy load (the default on Metal). (llama.cpp build 8680 could not load this arch; current llama.cpp builds can — ratios vs a 2026-06-11 build. MoE perf on Metal is a known optimization target.) |
| Qwen3.5-MoE-35B-A3B | Q4_0 | Production-ready (functional) | 0.18× | 0.08× | Same mmap default; same MoE-perf caveat |
| Qwen3.6-27B dense | Q8_0 | Production-ready | **1.03× (beats llama.cpp)** | 0.86× | All quality gates pristine (2026-06-11) |
| Qwen3.6-27B dense | Q4_0 | Production-ready | 0.99× | 0.82× | All quality gates pristine |
| Qwen3.6-27B dense | BF16 | N/A on Metal | — | — | Same ~50 GiB capacity-margin arithmetic as the Qwen3.8-27B BF16 row below; validated on CUDA H100 instead |
| Qwen3.8-27B dense | Q8_0 | Production-ready (functional) | withdrawn — under re-measurement (the 2026-08-14 1.15× predates a machine-level bimodal-speed finding; audited re-runs read below it and are quarantined until the trigger is isolated) | — | All quality gates pristine + DET-001 50/50 (2026-08-14, M3 Ultra; llama.cpp b10032, same GGUF); prefill row pending |
| Qwen3.8-27B dense | Q4_0 | Production-ready | **1.30× (beats llama.cpp)** | — | All quality gates pristine + DET-001 50/50 (2026-08-14); prefill row pending |
| Qwen3.8-27B dense | BF16 | N/A on Metal | — | — | ~50 GiB weights sit inside the capacity margin policy on the 96 GB test rig (same arithmetic as Qwen3.6-27B BF16); validated on CUDA H100 instead |

## What is not (yet) supported

| Class | Status | Why |
|---|---|---|
| llama / mistral / qwen2 / phi / gemma architectures | Currently rejected at conversion | Not yet on the verified-against-llama.cpp matrix. v1 scope decision; planned for future model-family releases. |
| NVIDIA hardware below compute capability 8.0 (pre-Ampere) | Untested | Ampere/Hopper (e.g. A100, H100) is the kernel target; older cards may compile but are not gated |
| Apple Silicon outside the M-series tested configuration | Untested | The published Metal benchmarks were measured on an M3 Ultra |
| K-quants (Q4_K, Q5_K, Q6_K, Q2_K, Q3_K) at runtime | Backend-dependent at GGUF→LBC import. `--target metal` upcasts K-quant (and legacy Q5_0) layer tensors to Q8_0 and re-quantizes Q4_1 to Q4_0, unless an explicit `--requant`/`--dequantize` overrides. Generic/CUDA conversions carry K-quant layer planes verbatim and CUDA dequantizes them to F32 at load — except MoE shared-expert planes, which are always requantized to Q4_0. Two role-specific tensors are requantized by default and preserved with `LUMEN_CONVERT_SOURCE_FIDELITY=1` for dedicated CUDA kernels: a Q5_K `ssm_out` and a Q6_K output head | No general K-quant matmul kernels; CUDA kernels exist only for the two preserved roles |
| MXFP4 at runtime | No LBC representation: required MXFP4 layer tensors are rejected at conversion; optional tensors and MoE shared-expert planes are dequantized (to F32, or Q4_0 for shared-expert gate/up) | No MXFP4 runtime kernels |
| Batched serving (batch > 1 per request) | Not implemented | Single-stream decode is the optimization target |
| Speculative decoding / MTP heads | Filtered at conversion | Out of scope |

## Configurations that pass-or-fail at runtime

The full registry (canonical) is at `model_registry.toml`. `lumen models` prints the live set including disk-cached LBCs. Unsupported `(model, quant)` combinations are rejected with a clear error listing available alternatives.

## Reference numbers

The status tables above give each configuration's ratio against llama.cpp. Raw throughput (tok/s) for Lumen and the llama.cpp baselines it is measured against — including the cold-load vs warm-state split and the rigs the numbers were captured on — is in [`bench/RESULTS.md`](../bench/RESULTS.md). Benchmark methodology: [`bench/METHODOLOGY.md`](../bench/METHODOLOGY.md).
