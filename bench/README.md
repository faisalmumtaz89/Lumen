# Lumen Benchmark Suite

Automated, reproducible benchmarking for Lumen inference engine vs MLX and llama.cpp on Apple Silicon.

## Quick Start

```bash
# Full suite (all models, all configs, ~45 min)
./bench/run_bench.sh

# Quick check (pp128+gen128 only, 3 runs, ~10 min)
./bench/run_bench.sh --quick

# Lumen only (skip MLX and llama.cpp)
./bench/run_bench.sh --lumen-only

# Specific model(s)
./bench/run_bench.sh --models "llama-8b"
./bench/run_bench.sh --models "llama-8b,qwen35-9b"
```

## Prerequisites

- **Lumen**: Rust toolchain (`cargo build --release -p lumen-cli`)
- **MLX**: Python venv at `~/.venvs/mlx-bench/` with `mlx-lm` installed
- **llama.cpp** (optional): `llama-bench` at `/opt/homebrew/bin/llama-bench` + GGUF files; auto-skipped if not found
- **Models**: `.lbc` files in `/tmp/lumen-bench/` (configurable with `--bench-dir`)
- MLX models auto-discovered from HuggingFace cache and `/tmp/lumen-bench/`

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--quick` | off | pp128+gen128 only, 3 runs |
| `--lumen-only` | off | Skip MLX and llama.cpp benchmarks |
| `--skip-llamacpp` | off | Skip llama.cpp benchmarks only |
| `--no-build` | off | Skip `cargo build` step |
| `--models FILTER` | all | Comma-separated substrings to match LBC filenames |
| `--runs N` | 5 | Measured runs per config |
| `--warmup N` | 1 | Warmup runs (discarded) |
| `--cooldown N` | 30 | Seconds between model switches |
| `--config-cooldown N` | 5 | Seconds between config changes within a model |
| `--prompt-lengths "L..."` | "32 128 512 1024" | Space-separated prompt token counts |
| `--gen-lengths "L..."` | "32 128 256" | Space-separated generation token counts |
| `--bench-dir DIR` | /tmp/lumen-bench | Directory containing `.lbc` model files |

## Methodology

1. **Execution order**: MLX first (cold GPU), then llama.cpp, then Lumen -- avoids thermal bias
2. MLX uses `-n 5` (5 internal trials per run) for stable measurements
3. Warmup runs are discarded before measured runs begin
4. Statistics: **median** and **stddev** across measured runs
5. Cooldown period between model switches to prevent thermal throttling
6. Ctrl+C generates partial results from whatever has completed

## Output

Results are saved to `bench/results/YYYY-MM-DD_HHMMSS/` with a symlink at `bench/results/latest/`.

```
bench/results/latest/
  results.md         # Markdown report
  results.json       # Machine-parseable JSON
  results_raw.txt    # Pipe-delimited raw data
  raw/               # Individual run logs
```

`bench/results/` is gitignored.

## Report Format

The report is structured around **Lumen vs baselines** (MLX and llama.cpp where available), not model-to-model comparison.

### Summary Table

One row per model+quant at the canonical pp128+gen128 config, showing decode and prefill with ratios against each available baseline:

```
| Model        | Quant | Lumen Decode | MLX Decode | vs MLX    | LC Decode | vs LC     | Lumen Prefill | ...
|--------------|:-----:|-------------:|-----------:|:---------:|----------:|:---------:|--------------:|----
| Llama 3.1 8B | Q8_0  | 73           | 80         | 0.92x     | 68        | **1.08x** | 836           | ...
| Qwen3.5 9B   | Q8_0  | 56           | 70         | 0.80x     | n/a       | n/a       | 336           | ...
```

- **Ratio = Lumen / baseline**. Bold when >= 1.00 (Lumen faster), plain when < 1.00 (Lumen slower).
- Same-quant comparisons only. llama.cpp columns shown when models are supported (n/a otherwise).

### Detail Tables

Decode and prefill detail tables show every config (prompt length x generation length) grouped per model, with sub-rows for each available baseline:

```
| Model           | Quant | pp32+g32 | pp128+g128 | ...  |
|-----------------|:-----:|--------: |----------: |-----:|
| Llama 3.1 8B    | Q8_0  | 76       | 73         | ...  |   <- Lumen
| ^(MLX Q8_0)     |       | 82       | 80         | ...  |   <- MLX baseline
| ^(vs MLX)       |       | 0.93x    | 0.92x      | ...  |   <- Lumen/MLX
| ^(llama.cpp Q8_0)|      | 70       | 68         | ...  |   <- llama.cpp baseline
| ^(vs LC)        |       | **1.09x**| **1.08x**  | ...  |   <- Lumen/llama.cpp
```

### Lumen-Only Mode

When `--lumen-only` is used, the report shows raw Lumen decode and prefill tables without baseline comparison.

## CUDA Benchmarks

CUDA benchmarks run on NVIDIA A100-80GB via [Modal](https://modal.com). The CUDA benchmark script is not included in this repository (it requires Modal credentials and infrastructure). See [METHODOLOGY.md](METHODOLOGY.md) for the CUDA benchmark methodology, execution order, and hardware details.
