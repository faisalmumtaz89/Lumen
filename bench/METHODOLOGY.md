# Lumen Benchmark Methodology

## Overview

This document describes the methodology used for Lumen's published benchmark suite against llama.cpp (CUDA + Metal) and vLLM (CUDA).

The published benchmark suite is scoped to the v1 model family. Additional model families will be added to the suite as they ship.

- **Models measured (v1)**: Qwen3.5-9B (`qwen35`) and Qwen3.5-MoE 35B-A3B (`qwen35moe`; architecture-truthful active-parameter label is 30B-A3B)
- **Backends measured**: CUDA on NVIDIA A100-80GB (SM 80) and Metal on Apple Silicon M3 Ultra
- **Canonical headline configuration**: pp128 + gen128, batch=1 (the (M, G) envelope extends this — see "Prompt and Generation Sizes" below)

Architectures outside the v1 set and hardware classes outside the measured rigs are out of scope for the currently-published benchmark; they will be folded in as they are added to the verified-against-llama.cpp matrix.

## Engines

| Engine | Version Tracking | Weight Format |
|--------|-----------------|---------------|
| **Lumen** | Git commit hash | LBC (converted from GGUF inside the Modal container) |
| **llama.cpp** | Git commit hash + build number | GGUF (same file Lumen converts from) |
| **vLLM** | pip version | GGUF (BF16) |

All three engines use the **same v1 Qwen3.5-9B weight file** for a given quant. Lumen converts GGUF to LBC during container setup; llama.cpp and vLLM load the GGUF directly. As new model families are added to the v1+ matrix, parity runs will use the same shared-weight-file methodology.

**vLLM note**: vLLM's GGUF loader is used in BF16 only. vLLM is run at batch=1 to match the other engines. vLLM's strengths surface at larger batch sizes, which is not the configuration measured here.

## Hardware

| Field | Value |
|-------|-------|
| Platform | Remote A100 cluster (canonical); Modal (deprecated for benchmarking) |
| GPU | NVIDIA A100-80GB (PCIe canonical for parity baselines; SXM4 used for long-context and soak scenarios where annotated) |
| CUDA | 12.6.3 |
| Compute capability | SM 80 |

**Infrastructure note:** Lumen pivoted from Modal to a remote 4× A100 cluster as the primary CUDA execution channel on 2026-05-25.

**Metal note:** Apple Silicon benchmarks run via `bench/run_bench.sh` on Mac Studio M3 Ultra 96 GB against `llama-bench` (build 8680) and, when present, MLX.

## Model (v1, current benchmark)

The published headline benchmark targets the v1 Qwen3.5-9B cell. Future model families will get their own entries against the same harness.

| Field | Value |
|-------|-------|
| Name | Qwen3.5-9B |
| Architecture | `qwen35` (GDN hybrid + dense FFN) |
| Parameters | 9B |
| Quants tested | Q8_0, Q4_0, BF16 |
| GGUF source | `bartowski/Qwen_Qwen3.5-9B-GGUF` |
| BF16 derivation | `convert_hf_to_gguf.py --outtype bf16` on the HuggingFace fp32 release |

## Metrics

| Metric | Definition | Unit |
|--------|-----------|------|
| **Decode throughput** | Generated tokens per second (excludes first token) | tok/s |
| **Prefill throughput** | Prompt tokens processed per second | tok/s |

Timing excludes tokenization and detokenization.

## Prompt and Generation Sizes

| Config | Prompt tokens | Generation tokens |
|--------|:------------:|:-----------------:|
| pp128 + gen128 | 128 | 128 |

The headline summary in [RESULTS.md](RESULTS.md) is pinned to (M, G) = (128, 128) at batch=1. The full decode envelope is parameterized by BOTH M (context length) AND G (output length post-warmup) — see "Decode envelope" in [RESULTS.md](RESULTS.md). Decode rate is sensitive to both axes:

### Multi-workload pattern protocol

In addition to the canonical pp128+gen128 anchor, the workload-stability matrix measures 5 realistic workload patterns:

| Workload | Prompt tokens (after BPE) | Gen tokens | Purpose |
|----------|---------------------------|-----------:|---------|
| Short Q&A           | ~50  | 150 | Chat-style interactions, low latency |
| Medium completion   | ~500 | 500 | Document summarization, structured outputs |
| Long-form           | ~200 | 1500 | Creative writing, long-context reasoning |
| Code generation     | ~100 | 800 | Structured Rust/Python/C output |
| Multi-turn          | ~500 (3-turn) | 200 ea | Chat sessions, KV reuse scenarios |

**Workload-stability gate**: decode tok/s range across the 5 workloads must be ≤15% within a quant. Empirical results at: Q8 9.1%, Q4 9.6%, BF16 10.1% — all PASS. Workload-weighted means are reported in [RESULTS.md](RESULTS.md) §"TL;DR — production-realistic numbers".



- **M (context-length axis)** drives per-step KV-scan cost; decode tok/s falls roughly linearly with M at fixed G in the long-context band.
- **G (gen-length axis)** drives a tail-growth effect post-warmup: measured a **-39% Q8 decode drop** going from G = 128 to G = 512 at fixed M = 128 (85.8 -> 52.6 tok/s). An envelope anchored on gen128 alone misleads consumers of long-output (streaming chat, code generation) production workloads.

When citing decode tok/s, ALWAYS report the (M, G) pair, not M alone. The bench JSON schema records `prompt_len` and `gen_len` as separate result-row fields so downstream readers can re-compute envelopes by either axis (see "Bench report JSON schema" below).

## Trial Protocol

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Warmup runs | 1 | Prime GPU caches, JIT, cuBLAS workspace, CUDA graph capture |
| Measured runs | 5 | Industry standard (llama-bench, vLLM benchmarks) |
| Batch size | 1 | Single-stream latency-oriented workload |

### Execution

Each engine runs in the same A100 container with the same GPU. CUDA graphs:

- **Lumen** captures a CUDA graph during the warmup phase and replays it for measured tokens.
- **llama.cpp** uses its own CUDA graph mechanism.
- **vLLM** is run with `enforce_eager=True` (CUDA graphs disabled in vLLM).

This asymmetry is inherent to each engine's architecture and is preserved in published numbers.

### Post-warmup measurement

For long-running soak scenarios and multi-quant comparisons, the canonical methodology is **post-warmup linear regression**: the initial transient samples are discarded from the slope computation, so only steady-state behavior is graded. This applies to RSS / VRAM / FD / TPS drift gates equally and yields stable signal independent of the initial transient.

### Paired A/B (multi-quant comparison)

When comparing multiple environment-variable opt-in combinations on the same hardware in the same session, runs alternate `A, B, A, B, ...` and paired delta is computed per A-B pair. This methodology:

- removes hardware-instance drift (A100 PCIe vs SXM4 variability per instance),
- removes inter-trial thermal drift,
- yields a tight confidence interval at relatively low trial counts (10 pairs ~ 0.05% noise floor).

The `+1.5%` ACCEPT gate used in Lumen perf evaluation is derived empirically from paired-AB noise-floor estimation.

### Bandwidth-floor cross-check

For each quant + sequence length, the theoretical bandwidth floor is computed as `(KV bytes read + active weight bytes) / A100 HBM bandwidth (2 TB/s peak, ~1.55 TB/s sustained)`. Measured prefill tok/s is then compared against this floor. The bandwidth floor is the lower bound; achieving >= 60% of the floor is the heuristic "kernel-arithmetic-not-the-bottleneck" sanity check. The current published long-context bandwidth-floor table covers the v1 Qwen3.5-9B cell at 4K-131K — see `bench/RESULTS.md`.

### Instance freshness

Each remote-A100 invocation gets a fresh GPU container. Cross-invocation re-runs establish reproducibility. Modal invocations (deprecated) similarly allocated fresh containers per `modal run`.

## Statistical Reporting

| Statistic | Role |
|-----------|------|
| **Median** | Primary reported value |
| **Stddev** | Reported in detail tables when available |

Individual per-run measurements are preserved in `bench/results/<timestamp>/results.json`.

## Bench report JSON schema

The bench harness in `bench/run_bench.sh` writes `bench/results/<timestamp>/results.json` and a matching `results.md`. Each `results` row captures BOTH the prompt-length (M) and the generation-length (G) axis as separate fields, so downstream consumers can re-aggregate the envelope on either axis without ambiguity. This is the corrective for the C-17 methodology gap (envelope rows that cited only M and silently dropped G implicitly anchored on G = 128).

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | Run timestamp `YYYY-MM-DD_HHMMSS` (matches the parent directory) |
| `hardware` | string | Auto-detected host description (CPU brand, GPU cores, total RAM) |
| `runs_per_config` | int | Measured runs per (engine, model, quant, M, G) cell |
| `warmup_runs` | int | Warmup runs discarded before measurement starts |
| `prompt_lengths` | int| All M values exercised in this run (the M axis of the envelope) |
| `gen_lengths` | int| All G values exercised in this run (the G axis of the envelope) |
| `results` | object| One entry per (engine, model, quant, M, G) cell (see below) |

### Per-result-row fields

| Field | Type | Description |
|-------|------|-------------|
| `engine` | string | `lumen` / `mlx` / `llamacpp` (one row per engine per cell) |
| `model_file` | string | Basename of the model weights file (LBC / safetensors-dir / GGUF) |
| `model_name` | string | Human-readable model identifier (e.g. `Qwen3.5 9B`) |
| `quant` | string | Quantization label (`Q8_0`, `Q4_0`, `BF16`, ...) |
| `prompt_len` | int | **M (context length).** Prompt token count for this cell. |
| `gen_len` | int | **G (generation length).** Tokens decoded post-prefill for this cell. |
| `prefill_tps_median` | float \| null | Prefill throughput (tok/s), median over `runs_per_config` measured runs. Null when unsupported or all runs failed. |
| `prefill_tps_stddev` | float \| null | Prefill throughput stddev across measured runs. Null mirrors `prefill_tps_median`. |
| `decode_tps_median` | float \| null | Decode throughput (tok/s) for this (M, G) cell, median over `runs_per_config` measured runs. Null when unsupported or all runs failed. |
| `decode_tps_stddev` | float \| null | Decode throughput stddev across measured runs. Null mirrors `decode_tps_median`. |
| `unsupported` | bool | True when the (engine, model, quant) combination is structurally not supported (e.g. llama.cpp on Qwen3.5-9B's GDN layers); both throughput fields will be null. |

### (M, G) preservation invariant

`prompt_len` and `gen_len` are recorded as **separate** integer fields on every row. Tools that aggregate over the JSON MUST group by `(model_name, quant, prompt_len, gen_len)` jointly, not by `(model_name, quant, prompt_len)` alone — collapsing the G axis is the bug that C-17 corrects.

Determinism across G is verified separately in the release-level harnesses (`scripts/prompt_length_bench.sh` records a per-G `first-16 generated tokens` line so equivalence across G can be byte-checked at fixed seed).

## Comparison Ratios

```
Ratio = Lumen / baseline
```

- **Bold** when ratio `>= 1.00` (Lumen at least as fast as baseline)
- Plain when ratio `< 1.00`
- `n/a` when the baseline engine does not support this configuration
- `--` when the run failed or produced no result

## Required env-vars for full performance

Lumen exposes several CUDA optimizations as environment variables. The **canonical production stack is 12 flags total** (8 default-ON opt-out flags + 4 load-bearing perf levers). All default to ON except `LUMEN_CUDA_BF16_GEMMEX` (which must be `0`), so setting them explicitly is idempotent.

### canonical production env stack (CUDA, MoE-30B-A3B, BF16 0.9× llama.cpp gate)

```bash
# canonical 8-flag stack (default-ON; opt-out=0):
LUMEN_CUDA_MOE_BATCHED=1                  # default ON
LUMEN_CUDA_MOE_ROUTER_SINGLE_CTA=1        # default ON
LUMEN_CUDA_MOE_ROUTER_PARALLEL=1          # default ON
LUMEN_CUDA_GDN_REGISTER_RESIDENT=1                 # default ON — +9.4% Q8, +10.3% Q4
LUMEN_CUDA_BF16_GEMMEX=0                  # MUST be 0 for BF16 P3 correctness
LUMEN_CUDA_BF16_MOE_V3=1                  # default ON
LUMEN_CUDA_MOE_Q4_V3=1                    # default ON
LUMEN_CUDA_MOE_Q4_V3B=1                   # default ON

# Canonical defaults (the 4 load-bearing perf levers):
LUMEN_CUDA_MMV_BF16_OUTPUT_PROJ=1         # default ON — load-bearing for BF16 0.902× llama.cpp
LUMEN_CUDA_TOPK_MOE_FUSED=1               # default ON — +6-8% all 3 MoE quants
LUMEN_CUDA_MMV_Q_DP4A=1                   # default ON — +7.1% Q8, +6.3% Q4
LUMEN_CUDA_MMV_Q_MOE_DP4A=1               # default ON — +11.7% Q4
```

**The 4 canonical defaults (`LUMEN_CUDA_TOPK_MOE_FUSED`, `LUMEN_CUDA_MMV_Q_DP4A`, `LUMEN_CUDA_MMV_Q_MOE_DP4A`, plus the `LUMEN_CUDA_MMV_BF16_OUTPUT_PROJ`) each toggle a specific code path; all default to ON so that out-of-the-box runs reproduce the published gate-clear numbers. To A/B against any prior baseline, set the corresponding gate to `=0`.**

### Optional opt-ins (kept env-gated, default OFF)

| Env var | Default | Path it toggles | Measured gain | Why default-OFF |
|---------|---------|-----------------|--------------:|------------------|
| `LUMEN_CUDA_MMV_Q_OUTPUT_PROJ=1` | OFF | T2-retry Q8/Q4 output_proj llama.cpp port | +1.5% / -1.6% (noise) | Within noise floor; kept opt-in for byte-identity preservation on Q8/Q4 vocab head |
| `LUMEN_CUDA_MOE_BATCHED_V4=1` | OFF | V4 MoE FFN cooperative-CTA register-budget-reduced kernels | +16.8% Q8 standalone | Not integrated to main; ~450 LoC dispatch wiring deferred to a future release |
| `LUMEN_CUDA_FA2_ATTN=1` | OFF | FA2 attention decode port | -2.7% Q8 | Per-call HBM alloc dominates at batch=1 |
| `LUMEN_CUDA_GDN_SPLIT=1` | OFF | Split layout for GDN tensors (Q4 only) | +2.6% Q4 decode | Q8 + GDN_SPLIT does not fit in 80 GB VRAM |
| `LUMEN_CUDA_Q8_SPLIT=1` | OFF | Raw + split layout for Q8_0 weights | +4.5% Q8 decode |  dense-9B opt-in; superseded by MMV_Q_DP4A for MoE |
| `LUMEN_CUDA_Q4_SPLIT=1` | OFF | Raw + split layout for Q4_0 weights | +9.0% Q4 decode |  dense-9B opt-in; superseded by MMV_Q_MOE_DP4A for MoE |
| `LUMEN_CUDA_OUTPUT_PROJ_SPLIT=1` | OFF | Big-NR variant for the 1 GB output projection tensor | +7.7% Q8 decode (alone) | Dense-9B path; output_proj llama.cpp port is the MoE-class equivalent |
| `LUMEN_CUDA_OUTPUT_PROJ_NR=16` | OFF | Override `NR` from default 32 to 16 for the output projection | +1.6% Q8 decode | Dense-9B path |
| `LUMEN_CUDA_Q8_SCALE_HW=1` | OFF | Native `LDG.E.U16` scale fetch in the Q8 matvec path | +0.4% Q8 decode | Below ±1.5% gate threshold |
| `LUMEN_CUDA_DECODE_DELAY_US=<N>` | `0` (CLI/bench) | CPU sleep (µs) after the per-decode-step `device.synchronize()`; serializes inter-step submission to close the GPU-scheduler timing race that surfaces as MoE Q4 **server**-only non-determinism. `lumen-server` applies `50` automatically. | n/a — determinism knob, not a perf lever | CLI / Q8 / BF16 are deterministic without it. Cost ≤1% TPOT per acceptance gate. |

### Opt-out matrix (rollback)

| Scenario | env vars to set | Effect |
|----------|-----------------|--------|
| Roll back → (BF16 only) | `LUMEN_CUDA_TOPK_MOE_FUSED=0 LUMEN_CUDA_MMV_Q_DP4A=0 LUMEN_CUDA_MMV_Q_MOE_DP4A=0` | BF16 86.1 = 0.849× llama.cpp; Q8 71.8; Q4 80.9 |
| Roll back to baseline | + `LUMEN_CUDA_MMV_BF16_OUTPUT_PROJ=0` | BF16 83.8; Q8 71.8; Q4 80.9 |
| Q4-only opt-out (preserves Q8 + BF16 gains) | `LUMEN_CUDA_MMV_Q_MOE_DP4A=0` | Q4 drops ~94 (loses +11.7%); BF16 / Q8 unchanged |

### Legacy Modal full-perf invocation (DEPRECATED — retained for historical re-runs)

> **Warning:** the Modal invocation block below targets dense-9B and uses the legacy opt-in set. It is **NOT** the current production stack. For current main, the 12-flag stack documented above is **default-ON** and out-of-the-box `lumen run` reproduces the published numbers. Critically, the legacy block below explicitly sets `LUMEN_CUDA_BF16_GEMMEX=1` (the legacy default); the **current production requirement is `LUMEN_CUDA_BF16_GEMMEX=0`** for BF16 P3 correctness on MoE. Do not copy this stanza into new deployments.

```
modal run modal/bench_real_models.py --models qwen3.5-9b --quants q8_0,q4_0 \
  --lumen-env "LUMEN_CUDA_GDN_SPLIT=1,LUMEN_CUDA_Q8_SPLIT=1,LUMEN_CUDA_Q4_SPLIT=1,\
LUMEN_CUDA_OUTPUT_PROJ_SPLIT=1,LUMEN_CUDA_Q8_SCALE_HW=1,LUMEN_CUDA_GDN_REGISTER_RESIDENT=1,\
LUMEN_CUDA_OUTPUT_PROJ_NR=16,LUMEN_CUDA_BF16_GEMMEX=1"   # legacy value; current prod = 0
```

For BF16 specifically (legacy):

```
modal run modal/bf16_industry_bench.py
```

Note: `bf16_industry_bench.py` exercises the legacy default-on cuBLAS GemmEx prefill path. Per the current production requirement is `LUMEN_CUDA_BF16_GEMMEX=0`.

## Reproducibility

Each benchmark report includes:

| Field | Example |
|-------|---------|
| Lumen commit | `abc1234` |
| Rust version | `1.93.1` |
| llama.cpp commit | `0ec191e` |
| vLLM version | `0.21.0` |
| transformers version | `5.5.4` |
| GPU | `NVIDIA A100-80GB PCIe` |
| CUDA | `12.6.3` |
| Date | `2026-06-02` (RESULTS.md last-updated; see [RESULTS.md](RESULTS.md)) |
| Bench script (CUDA) | remote-A100 harness (canonical); `modal run modal/bench_real_models.py` is deprecated and retained for historical re-runs |
| Bench script (Metal) | `bench/run_bench.sh` on Mac Studio M3 Ultra |

## Build Configuration

| Engine | Build flags |
|--------|------------|
| Lumen | `cargo build --release --features cuda -p lumen-cli` |
| llama.cpp | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80-real -DCMAKE_BUILD_TYPE=Release` |
| vLLM | `enforce_eager=True`, `VLLM_USE_FLASHINFER_SAMPLER=0`, batch=1 |

## What this methodology does NOT cover

- Architectures beyond v1's `qwen35` dense + `qwen35moe` (the published parity table above is dense-only). Future model families enter the matrix as they ship.
- GPUs other than A100-80GB
- Batched serving throughput (multiple concurrent requests)
- Time-to-first-token (TTFT) as a separate latency metric (prefill throughput is reported, which implicitly measures prompt processing time)
- Accuracy/quality comparisons (all engines produce equivalent outputs for the same quantization on the cross-engine reference matrix; quality is validated separately)
- GPU memory usage comparisons (covered in soak harnesses)
- Power consumption
- Multi-turn / KV cache reuse performance beyond the suffix-prefill cache hit (Cell J)

Long-context performance >4K is covered separately in the long-context bench results; see [RESULTS.md](RESULTS.md) for the published 64K / 98K / 131K numbers.

These may be added in future methodology revisions.
