# Lumen Benchmark Suite

Automated, reproducible benchmarking for Lumen inference engine vs MLX and llama.cpp on Apple Silicon.

## Quick Start

```bash
# Full suite (all models, all configs, ~45 min)
./bench/run_bench.sh

# Quick check (pp128+gen128 only, 3 runs, ~10 min)
./bench/run_bench.sh --quick

# Lumen only (skip MLX comparison)
./bench/run_bench.sh --lumen-only

# Specific model(s)
./bench/run_bench.sh --models "llama-8b"
./bench/run_bench.sh --models "llama-8b,qwen35-9b"
```

## Prerequisites

- **Lumen**: Rust toolchain (`cargo build --release -p lumen-cli`)
- **MLX**: Python venv at `~/.venvs/mlx-bench/` with `mlx-lm` installed
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

1. **MLX runs first** on a cold GPU to avoid thermal bias from Lumen runs
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

The report is structured around **Lumen vs MLX comparison**, not model-to-model comparison.

### Summary Table

One row per model+quant at the canonical pp128+gen128 config, showing decode and prefill with Lumen/MLX ratio:

```
| Model           | Quant | Lumen Decode | MLX Decode | Ratio    |
|-----------------|:-----:|-------------:|-----------:|:--------:|
| Llama 3.1 8B    | Q4_0  | 98           | 80         | **1.23x**|
| Llama 3.1 8B    | Q8_0  | 73           | 80         | 0.92x   |
| Qwen3.5 9B      | Q8_0  | 56           | 70         | 0.80x   |
```

- **Ratio = Lumen / baseline**. Bold when >= 1.00 (Lumen faster), plain when < 1.00 (Lumen slower).
- Same-quant comparisons where available. Includes llama.cpp when models are supported.

### Detail Tables

Decode and prefill detail tables show every config (prompt length x generation length) grouped per model, with three rows each:

```
| Model           | Quant | pp32+g32 | pp128+g128 | ...  |
|-----------------|:-----:|--------: |----------: |-----:|
| Llama 3.1 8B    | Q4_0  | 106      | 98         | ...  |   <- Lumen
| ^(MLX Q8)       |       | 82       | 80         | ...  |   <- MLX baseline
| ^(ratio)        |       | **1.30x**| **1.23x**  | ...  |   <- Lumen/MLX
```

### Lumen-Only Mode

When `--lumen-only` is used, the report shows raw Lumen decode and prefill tables without MLX comparison.
