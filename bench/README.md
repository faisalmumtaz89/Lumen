# Lumen Benchmark Suite

Reproducible benchmarking for Lumen — **LLM inference in Rust, for Apple Silicon and NVIDIA CUDA** — measured head-to-head against llama.cpp (CUDA + Metal) and vLLM (CUDA only).

This directory contains the perf-benchmarking suite. Methodology lives in [METHODOLOGY.md](METHODOLOGY.md); results live in [RESULTS.md](RESULTS.md); the run harness is `run_bench.sh`.

The published benchmark suite is currently scoped to Lumen's v1 model family on two accelerator families. Additional model families will be added to the suite as they ship.

- **Primary model (v1)**: Qwen3.5-9B (`qwen35` architecture — GDN hybrid + dense FFN)
- **Secondary model (v1)**: Qwen3.5-MoE 35B-A3B (`qwen35moe`; architecture-truthful active-parameter label is 30B-A3B) — see [RESULTS.md](RESULTS.md)
- **Backends measured**: CUDA on NVIDIA A100-80GB (SM 80) and Metal on Apple Silicon M3 Ultra

Architectures outside the v1 set are out of scope for the currently-published benchmark.

## Quick Start

### Metal (canonical local harness — Apple Silicon)

`run_bench.sh` is a macOS / Apple-Silicon-only harness that runs Lumen vs `llama-bench` (and, when available, MLX) head-to-head. It does **not** orchestrate Modal or vLLM; CUDA + vLLM runs use the legacy Modal path below.

```bash
./run_bench.sh                  # full envelope (~12 cells)
./run_bench.sh --quick          # smaller subset
./run_bench.sh --lumen-only     # skip llama-bench / MLX baselines
```

Prerequisites: macOS on Apple Silicon, `/opt/homebrew/bin/llama-bench`, optionally an MLX venv at `~/.venvs/mlx-bench`, and Lumen built with `cargo build --release -p lumen-cli` (LBCs auto-cache under `/tmp/lumen-bench`).

### CUDA (Modal — legacy, retained for historical re-runs)

```bash
# Q8_0 and Q4_0 head-to-head (Lumen vs llama.cpp vs vLLM)
modal run modal/bench_real_models.py --models qwen3.5-9b --quants q8_0,q4_0

# BF16 head-to-head (separate script — uses bf16-converted GGUF)
modal run modal/bf16_industry_bench.py
```

Note: the Modal scripts (`modal/bench_real_models.py`, `modal/bf16_industry_bench.py`) are referenced by [METHODOLOGY.md](METHODOLOGY.md) but are no longer maintained in-tree; canonical CUDA numbers in [RESULTS.md](RESULTS.md) were captured directly on the remote A100 cluster.

For the v1 Qwen3.5-9B cell, Lumen, llama.cpp, and vLLM all consume weights from `bartowski/Qwen_Qwen3.5-9B-GGUF` (BF16 weights generated locally via `convert_hf_to_gguf.py --outtype bf16`).

## Prerequisites

- **Lumen**: Rust toolchain (`cargo build --release --features cuda -p lumen-cli` for CUDA; `cargo build --release -p lumen-cli` for Metal)
- **CUDA hardware harness**: remote A100 cluster (canonical) — see [METHODOLOGY.md](METHODOLOGY.md) §Hardware
- **Metal hardware harness**: Mac Studio M3 Ultra 96 GB (canonical) — `run_bench.sh` is the entry point
- **GGUF weights (v1)**: pulled from `bartowski/Qwen_Qwen3.5-9B-GGUF`; the LBC conversion runs once per container / once per Mac cache. Future model families will reference their own GGUF sources.

## Full-performance invocation (CUDA)

The Lumen production env stack is **default-ON**.

The 12-flag canonical env stack is documented at [METHODOLOGY.md § Required env-vars for full performance](METHODOLOGY.md#required-env-vars-for-full-performance). Critically `LUMEN_CUDA_BF16_GEMMEX=0` is required for BF16 P3 correctness on MoE.

## Reproducing a single configuration

`run_bench.sh` (Metal) flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--quick` | (off) | Smaller subset (single M, single G, fewer trials) |
| `--lumen-only` | (off) | Skip llama-bench and MLX baselines |
| `--models <filter>` | all | Filter by registry name fragment |
| `--prompt-lengths "L L L"` | `32 128 512 1024` | M axis |
| `--gen-lengths "L L L"` | `32 128 256` | G axis |
| `RUNS` env var | 5 | Measured runs per cell |
| `WARMUP_RUNS` env var | 1 | Warmup runs discarded |

CUDA-via-Modal (legacy) flag list is preserved in `modal/bench_real_models.py` source.

## Output

`run_bench.sh` writes per-run results under `bench/results/<timestamp>/`:

```
bench/results/
  <timestamp>/
    results.md       # human-readable summary
    results.json     # machine-parseable JSON (schema: METHODOLOGY.md §"Bench report JSON schema")
  archive/           # archived legacy results preserved here
```

`bench/results/` is git-tracked only for the small summary text files; per-run raw directories are gitignored.

## Report Format

The canonical results live in [RESULTS.md](RESULTS.md). Each row is `Quant × Metric` with engine columns (Lumen, llama.cpp, vLLM where applicable). Ratios are reported as `Lumen / baseline` — values `>= 1.00` (bold) mean Lumen is at least as fast as the baseline.

## What this suite does NOT cover

- Architectures other than v1's `qwen35` and `qwen35moe` (additional model families are on the roadmap)
- Hardware below NVIDIA SM 8.0 / Apple Silicon M3 Ultra
- Batched serving throughput (batch > 1 per request)
- TTFT as a separate metric (prefill throughput is reported)
- Power consumption

See [METHODOLOGY.md](METHODOLOGY.md) for the full set of methodology decisions and the [archive](results/archive/) for historical results from earlier scopes.
