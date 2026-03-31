# Lumen Benchmark Methodology

## Overview

This document describes the methodology used for Lumen's benchmarks against llama.cpp, MLX, and vLLM.

## Engines

| Engine | Version Tracking | Weight Format |
|--------|-----------------|---------------|
| **Lumen** | Git commit hash | LBC (converted from GGUF) |
| **llama.cpp** | Git commit hash + build number | GGUF (same file as Lumen's source) |
| **MLX** | mlx-lm pip version + mlx version | MLX format (converted from HuggingFace) |
| **vLLM** | pip version | HuggingFace (FP16 only) |

All engines use the **same model weights** where format permits. Lumen and llama.cpp share the same GGUF source file (bit-identical weights). MLX uses its own quantization format converted from HuggingFace. Note: MLX 4-bit uses affine quantization (group_size=64) while GGML Q4_0 uses symmetric quantization (group_size=32) — these are not the same encoding. Q8_0 is effectively equivalent across formats.

## Hardware

### Metal (Apple Silicon)

| Field | Value |
|-------|-------|
| Machine | Mac Studio |
| Chip | Apple M3 Ultra |
| CPU | 28 cores (20P + 8E) |
| GPU | 60 cores |
| Memory | 96 GB unified |
| Memory BW | 819 GB/s |
| OS | macOS (report exact version) |
| Power Mode | High Power (not Low Power) |

### CUDA (NVIDIA)

| Field | Value |
|-------|-------|
| Platform | Modal AI (serverless) |
| GPU | NVIDIA A100 80GB |
| Note | Modal allocates either SXM4 or PCIe variants. Report the actual variant from `nvidia-smi`. |

## Models

### Standard Dense (benchmarked on both Metal and CUDA)

| Model | Architecture | Params | Quants Tested |
|-------|-------------|-------:|--------------|
| Qwen2.5 3B | qwen2 | 3B | F16, Q8_0, Q4_0 |
| Qwen2.5 7B | qwen2 | 7B | F16, Q8_0, Q4_0 |
| Llama 3.1 8B | llama | 8B | F16, Q8_0, Q4_0 |
| Qwen2.5 14B | qwen2 | 14B | F16, Q8_0, Q4_0 |

### GDN Hybrid (CUDA and Metal)

| Model | Architecture | Params | Quants Tested |
|-------|-------------|-------:|--------------|
| Qwen3.5 9B | qwen35 | 9B | Q8_0, Q4_0 |

### Small Model (Metal only)

| Model | Architecture | Params | Quants Tested |
|-------|-------------|-------:|--------------|
| TinyLlama 1.1B | llama | 1.1B | F16, Q8_0, Q4_0 |

## Metrics

| Metric | Definition | Unit |
|--------|-----------|------|
| **Decode throughput** | Generated tokens per second (excludes first token) | tok/s |
| **Prefill throughput** | Prompt tokens processed per second | tok/s |

Timing excludes tokenization and detokenization.

## Prompt and Generation Sizes

| Config | Prompt tokens | Generation tokens | Purpose |
|--------|:------------:|:-----------------:|---------|
| pp128 + gen128 | 128 | 128 | Primary comparison metric |

All published summary numbers use pp128 + gen128.

## Trial Protocol

### Runs

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Warmup runs | 1 | Prime GPU caches, JIT, cuBLAS workspace |
| Measured runs | 5 | Industry standard (llama.cpp, MLX default) |
| MLX internal trials | 5 | mlx_lm.benchmark default |

### Execution Order (Metal)

1. **MLX first** on a cold GPU (most sensitive to thermal state)
2. **llama.cpp second**
3. **Lumen third**

This ordering is conservative — Lumen runs on the warmest GPU.

### Execution Order (CUDA)

Each engine runs in a separate Modal container. The report records each engine's GPU variant from `nvidia-smi`. If variants differ across engines (e.g., one gets SXM4, another gets PCIe), the run is flagged as "mixed hardware."

**CUDA graphs:** Lumen captures a CUDA graph during the warmup phase and replays it for all measured tokens. vLLM is run with `enforce_eager=True` (CUDA graphs disabled). llama.cpp uses its own CUDA graph mechanism. This asymmetry is inherent to each engine's architecture.

### Cooldown

| Between | Duration | Rationale |
|---------|:--------:|-----------|
| Models | 30 seconds | Allow GPU thermal recovery between different model sizes |
| Configs (within a model) | 5 seconds | Brief pause between quant/prompt-length variations |
| Trials (within a config) | 0 seconds | Back-to-back for sustained throughput measurement |

### Thermal Monitoring (Metal)

Before each model's first trial, log GPU thermal state via `pmset -g therm`. If thermal pressure is elevated, insert additional 30-second cooldown and re-check. Discard any run where GPU frequency drops below 90% of observed peak.

Note: `pmset` thermal monitoring has limited granularity — it may not detect moderate frequency throttling. Close GPU-intensive background applications before benchmarking.

### Instance Freshness (CUDA)

Each `modal run` invocation gets fresh GPU containers. Multiple benchmark runs on separate invocations establish cross-instance reproducibility.

## Statistical Reporting

| Statistic | Role |
|-----------|------|
| **Median** | Primary reported value |
| **Stddev** | Shown as `+-` in detail tables |
| **Min / Max** | Recorded in raw data |

If coefficient of variation (stddev/mean) exceeds 5%, the result is flagged as high-variance.

## Comparison Ratios

```
Ratio = Lumen / baseline
```

- **Bold** when ratio > 1.00 (Lumen is faster)
- Plain when ratio <= 1.00 (Lumen is slower or equal)
- `n/a` when the baseline engine does not support this model/quant
- `--` when the run failed or produced no result

## Report Structure

Every benchmark run produces:

### 1. `results.md` — Human-readable report

```
# Lumen Benchmark Report
# [timestamp]

## Environment
[Hardware, OS, engine versions, Lumen commit]

## Methodology
[Brief: runs, warmup, cooldown, ordering, statistic]

## Summary — Decode (pp128 + gen128)
[One table: all models × quants, Lumen vs baselines with ratios]

## Summary — Prefill (pp128 + gen128)
[Same structure for prefill]

## Detailed Results
[Per model/quant tables with stddev]

## Lumen Only
[Models/quants where no same-format baseline exists]

## Models
[File names, sizes, sources for reproducibility]
```

### 2. `results.json` — Machine-readable data

Contains per-trial raw measurements, engine versions, hardware info, and model metadata.

### 3. `raw/` — Individual run output

Raw stdout/stderr from each engine invocation.

## Reproducibility Requirements

Each benchmark report includes:

| Field | Example |
|-------|---------|
| Lumen commit | `abc1234` |
| Rust version | `1.85.0` |
| llama.cpp version | `build 6729d49` |
| MLX version | `mlx-lm 0.30.7` |
| vLLM version | `0.8.4` |
| OS version | `macOS 26.3.0` |
| GPU | `Apple M3 Ultra 60c / A100-SXM4-80GB` |
| Date | `2026-03-30` |
| Benchmark script | `bench/run_bench.sh` (Metal) or `modal run modal/bench_real_models.py` (CUDA) |

## Model Provenance

For reproducibility, each benchmark report includes the source of every model file:

| Field | Example |
|-------|---------|
| GGUF source | `bartowski/Qwen2.5-7B-Instruct-GGUF` on HuggingFace |
| GGUF filename | `Qwen2.5-7B-Instruct-Q8_0.gguf` |
| LBC conversion | `lumen convert --input <gguf> --output <lbc>` |
| MLX source | `mlx-community/Qwen2.5-7B-Instruct-8bit` or `mlx_lm.convert` |

## Build Configuration

| Engine | Build flags |
|--------|------------|
| Lumen | `cargo build --release` (Metal) or `cargo build --release --features cuda` (CUDA) |
| llama.cpp | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80-real -DCMAKE_BUILD_TYPE=Release` |
| vLLM | `enforce_eager=True` (CUDA graphs disabled in vLLM) |
| MLX | `mlx_lm.benchmark` defaults |

## What This Methodology Does NOT Cover

- **Batch serving throughput** (multiple concurrent requests)
- **Time-to-first-token (TTFT)** as a separate latency metric (prefill throughput is reported, which implicitly measures prompt processing time)
- **Long-context** performance (>2048 tokens)
- **Accuracy/quality** comparisons (all engines produce equivalent outputs for the same quantization)
- **GPU memory usage** comparisons
- **Power consumption**
- **Multi-turn / KV cache reuse** performance

These may be added in future methodology revisions.
