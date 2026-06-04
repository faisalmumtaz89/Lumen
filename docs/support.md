# Lumen Model Support Matrix

This page is the source of truth for what is currently **verified-against-llama.cpp** end-to-end. Lumen runs LLM inference in Rust for Apple Silicon and NVIDIA CUDA; v1 (current) verifies the Qwen3.5 family and additional model families are planned. Architectures outside the v1 set (llama, mistral, qwen2, phi, gemma) are currently rejected at GGUF conversion because they have not yet been gated end-to-end on this runtime.

## What is verified

Each backend (CUDA + Metal) is validated end-to-end on the full models × quants matrix against llama.cpp (2026-06-02).

### CUDA (NVIDIA, compute capability 8.0+ — Ampere / Hopper)

Benchmarked on an A100-80GB; see [`bench/RESULTS.md`](../bench/RESULTS.md) for the rig and full numbers.

| Model | Quant | Status | × llama.cpp decode (canonical) | Notes |
|-------|-------|--------|------:|---|
| Qwen3.5-9B dense | Q8_0 | Production-ready | **0.91× llama.cpp** | All robustness and correctness gates pass |
| Qwen3.5-9B dense | Q4_0 | Production-ready (functional) | 0.64× llama.cpp | Below 0.9× perf gate (structural ceiling); all functional gates pass |
| Qwen3.5-9B dense | BF16 | Production-ready | **0.93–0.94× llama.cpp** | Highest-precision |
| Qwen3.5-MoE-35B-A3B | Q8_0 | Production-ready (functional) | 0.584× llama.cpp | MoE_Q8_SPLIT=OFF default validated |
| Qwen3.5-MoE-35B-A3B | Q4_0 | Production-ready (functional) | 0.674× llama.cpp | Same MoE setup path as Q8 MoE |
| Qwen3.5-MoE-35B-A3B | BF16 | Production-ready with caveats | 0.902× llama.cpp (recommended) | Requires dedicated 80 GB+ GPU (peak 72.4 GB) |

### Metal (Apple Silicon, M-series)

Benchmarked on an M3 Ultra; see [`bench/RESULTS.md`](../bench/RESULTS.md) for the rig and full numbers.

| Model | Quant | Status | Decode × llama.cpp | Prefill × llama.cpp | Notes |
|-------|-------|--------|------:|------:|---|
| Qwen3.5-9B dense | Q8_0 | Production-ready (default) | **0.98×** | 0.95× | Cleared 0.9× decode gate |
| Qwen3.5-9B dense | Q4_0 | Production-ready | **1.02×** / **1.17×** (beats llama.cpp) | 0.88× | Below 0.9× prefill (structural) |
| Qwen3.5-9B dense | BF16 | Production-ready (functional) | 0.83× | 0.66× (up from 0.31×) | Requires `LUMEN_METAL_MMAP_ONLY=1` |
| Qwen3.5-MoE-35B-A3B | Q8_0 | Production-ready (functional) | — | — | Requires `LUMEN_METAL_MMAP_ONLY=1`. llama.cpp build 8680 cannot load this arch — **Lumen is the sole provider on Apple Silicon**. |
| Qwen3.5-MoE-35B-A3B | Q4_0 | Production-ready (functional) | — | — | Same `MMAP_ONLY=1` requirement; sole provider on Apple Silicon |

## What is not (yet) supported

| Class | Status | Why |
|---|---|---|
| llama / mistral / qwen2 / phi / gemma architectures | Currently rejected at conversion | Not yet on the verified-against-llama.cpp matrix. v1 scope decision; planned for future model-family releases. |
| NVIDIA hardware below compute capability 8.0 (pre-Ampere) | Untested | Ampere/Hopper (e.g. A100, H100) is the kernel target; older cards may compile but are not gated |
| Apple Silicon outside the M-series tested configuration | Untested | The published Metal benchmarks were measured on an M3 Ultra |
| K-quants (Q4_K, Q5_K, Q6_K, Q2_K, Q3_K) at runtime | Converted to Q8_0 at GGUF→LBC import | No K-quant dispatch kernels |
| MXFP4 at runtime | Converted to Q8_0 at GGUF→LBC import | Same reason |
| Batched serving (batch > 1 per request) | Not implemented | Single-stream decode is the optimization target |
| Speculative decoding / MTP heads | Filtered at conversion | Out of scope |

## Configurations that pass-or-fail at runtime

The full registry (canonical) is at `model_registry.toml`. `lumen models` prints the live set including disk-cached LBCs. Unsupported `(model, quant)` combinations are rejected with a clear error listing available alternatives.

## Reference numbers

The status tables above give each configuration's ratio against llama.cpp. Raw throughput (tok/s) for Lumen and the llama.cpp baselines it is measured against — including the cold-load vs warm-state split and the rigs the numbers were captured on — is in [`bench/RESULTS.md`](../bench/RESULTS.md). Benchmark methodology: [`bench/METHODOLOGY.md`](../bench/METHODOLOGY.md).
