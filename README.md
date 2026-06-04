# Lumen

[Getting started](docs/getting-started.md) ·
[Models](docs/support.md) ·
[Server](docs/server.md) ·
[Production](docs/production.md) ·
[Bench](bench/RESULTS.md) ·
[Contributing](CONTRIBUTING.md) ·
[Changelog](CHANGELOG.md)

LLM inference in Rust, for **Apple Silicon** and **NVIDIA CUDA**.

Built from scratch in Rust with zero ML dependencies — no PyTorch, no ONNX, no Python runtime. Native CUDA C and Metal MSL kernels, native tokenizer, native model format. A single binary that downloads a model, runs GPU-resident inference, and prints text. v1 (current) ships Qwen3.5; additional model families are planned.

> **Status:** Production-ready for the currently-shipped Qwen3.5 family (dense 9B and MoE-30B-A3B) on NVIDIA CUDA (compute capability 8.0+) and Apple Silicon (M-series), with decode throughput competitive with llama.cpp on the benchmarked configs — see [benchmarks](bench/RESULTS.md). The public API and the binary `.lbc` format are not yet stable. **Read [Production deployment](docs/production.md) before deploying.**

## Supported models

The runtime is built around two architecture classes (a dense GDN-hybrid and an MoE GDN-hybrid). v1 ships verified support for the Qwen3.5 family; additional model families are planned in future releases.

### v1 (current) — verified-against-llama.cpp

| Name | Architecture | Parameters | Quants (registry) | GGUF source |
|------|--------------|------------|--------|-------------|
| `qwen3.5-9b` | Qwen3.5 (GDN hybrid + dense FFN) | 9B | Q8_0 in the registry. A Q4_0 LBC is reachable out-of-registry via `lumen convert --requant q4_0` (`--requant` accepts `q4_0` / `q8_0`). | [`bartowski/Qwen_Qwen3.5-9B-GGUF`](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF) |
| `qwen3.5-moe-35b-a3b` | Qwen3.5 MoE (GDN hybrid + sparse FFN) | 30B total / 3B active (registry label retains `35b-a3b`; architecture-truthful active-parameter count is 30B) | Q8_0, Q4_0, BF16 | [`bartowski/Qwen_Qwen3.5-35B-A3B-GGUF`](https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF) |

**Registry source of truth**: `model_registry.toml`. Run `lumen models` for current cells, aliases (e.g. `qwen3.5-moe`), and disk-cached LBCs. Unsupported `(model, quant)` combinations are rejected with a clear error message listing available alternatives.

Architectures outside the v1 set (llama, mistral, qwen2, etc.) are currently rejected at conversion — they are not yet verified-against-llama.cpp on this runtime. K-quant and MXFP4 GGUFs are accepted and converted on import. Additional model families are on the roadmap; see [`docs/support.md`](docs/support.md) for the live verified-against-llama.cpp matrix.

## Supported backends

| Backend | Hardware | Status |
|---------|----------|--------|
| **CUDA** | NVIDIA, compute capability 8.0+ (Ampere / Hopper — e.g. A100, H100) | Production-ready |
| **Metal** | Apple Silicon (M-series) | Production-ready |
| **CPU** | Scalar reference + SIMD NEON | Correctness reference; not optimized for throughput |

CUDA is enabled at build time with `--features cuda`. Metal compiles automatically on macOS targets.

Hardware below NVIDIA compute capability 8.0, or Apple Silicon outside the M-series, is untested — kernels may compile and run, but no correctness/performance gate covers them.

## Quick start

**One command, clone to running server** — detects your backend (Metal/CUDA/CPU), builds the binaries, downloads + converts a model, and starts the OpenAI-compatible server with copy-paste `curl` examples:

```bash
./scripts/quickstart.sh            # interactive; pick a model and go
./scripts/quickstart.sh --yes      # non-interactive (defaults: qwen3.5-9b q8_0)
./scripts/quickstart.sh --help     # all flags (--model, --quant, --backend, --port, --dry-run, …)
```

It auto-pulls the downloadable combos (`qwen3.5-9b:q8_0`, `qwen3.5-moe-35b-a3b:{q4_0,q8_0}`) and checks free disk against the true download+convert peak; any other model/quant (including the ~70 GB 2-shard `qwen3.5-moe-35b-a3b:bf16`) is refused with instructions to prepare it via `lumen convert` and re-run with `--model <path.lbc>`.

Or do it by hand:

```bash
# Install (CUDA on Linux)
cargo install --path crates/lumen-cli --features cuda

# Install (Metal on macOS)
cargo install --path crates/lumen-cli

# Auto-downloads ~10 GB on first use, converts GGUF -> LBC, caches, runs
lumen run qwen3.5-9b:q8_0 "Write a haiku about Rust"

# MoE — Q8_0 and Q4_0 auto-download from the registry
lumen run qwen3.5-moe-35b-a3b:q8_0 "Hello"
lumen run qwen3.5-moe-35b-a3b:q4_0 "Explain quantum computing in one paragraph"
```

A Q4_0 LBC for `qwen3.5-9b` (not published in the registry today) is produced from the registry's Q8_0 source via `lumen convert --requant q4_0`. `--requant` accepts `q4_0` and `q8_0`.

The MoE BF16 build (~70 GB, highest quality) is a 2-shard split GGUF that is **not auto-downloaded** today; fetch the shards manually, convert with `lumen convert`, and run with `--model <path.lbc>`.

The backend is auto-detected: macOS → Metal, Linux with a CUDA device → CUDA, else SIMD CPU. Force a backend with `--cuda`, `--metal`, or `--simd`. Pick a CUDA device with `--cuda-device <n>`.

## CLI

```bash
# Download / pre-convert without running
lumen pull qwen3.5-9b:q8_0

# List cached + available models
lumen models

# Raw token mode
lumen run --model /path/to/model.lbc --tokens "1 2 3" --max-tokens 128 --cuda

# Convert GGUF -> LBC manually (with optional requant)
lumen convert --input model.gguf --output model.lbc
lumen convert --input model.gguf --output model.lbc --requant q4_0
```

| Flag | Description |
|------|-------------|
| `--system <text>` | System prompt |
| `--max-tokens <n>` | Tokens to generate (default: unlimited, stops at EOS) |
| `--temperature <f>` | Sampling temperature (0 = greedy, default 0.8) |
| `--top-p` / `--top-k` / `--min-p` | Nucleus / top-K / min-prob cutoffs |
| `--repetition-penalty` / `--presence-penalty` / `--frequency-penalty` | Sampling penalties |
| `--seed <n>` | Random seed for reproducibility |
| `--cuda` / `--metal` / `--simd` | Force a backend |
| `--cuda-device <n>` | CUDA device ordinal (default 0) |
| `--context-len <n>` | KV cache size (auto-sized by default) |
| `--kv-disk-dir <path>` | Directory for disk-persistent KV cache |
| `--session-save <p>` / `--session-resume <p>` | Persist / restore a Session across runs (Metal today) |
| `--no-gpu-resident` | Stream weights from disk instead of GPU memory |
| `--verbose` | Show diagnostics and metrics |
| `--profile` | Per-operation timing breakdown |

**Configuration precedence:** CLI flag > environment variable > built-in default. For example, `--kv-precision f32` overrides `LUMEN_KV_PRECISION=f16`; with neither set the per-backend default applies (Metal `f16`, CUDA/CPU `f32`).

Full reference: `lumen run --help`.

## HTTP server

`lumen-server` ships both a library crate and an opt-in standalone binary that exposes an axum-based router with OpenAI and Anthropic wire formats. **Use this (not repeated `lumen run`) for any concurrent-client deployment** — see [Production deployment](#production-deployment) for the rationale.

```bash
# Pre-download a model (one-time, ~10 GB for Qwen3.5-9B Q8_0)
lumen pull qwen3.5-9b:q8_0

# Boot the server (Metal on macOS, CUDA on Linux with --features cuda)
cargo run --release --bin lumen-server --features bin -- \
  --model qwen3.5-9b --quant q8_0 --port 8000

# Or against an explicit LBC path
cargo run --release --bin lumen-server --features bin -- \
  --model /path/to/qwen3-5-9b-Q8_0.lbc

# Try it
curl http://localhost:8000/v1/models
```

The bin is gated behind the `bin` Cargo feature so library embedders that wire their own tokenizer / backend keep the lumen-server dep graph minimal. `lumen-server --help` lists all flags.

Custom embedders own the tokenizer and weight provider; the runtime owns the GPU.

```rust
use lumen_server::{build_router, EngineWorker};

let handle = EngineWorker::spawn(
    runtime_config,
    hyperparams,
    backend,
    weights,
    tokenizer,
    model_info,
    /* inbox_size */ 32,
);
let router = build_router(handle);
axum::serve(listener, router).await?;
```

Real signatures: `EngineWorker::spawn(config, hyperparams, backend, weights, tokenizer, model_info, inbox_size) -> EngineHandle` (`crates/lumen-server/src/engine.rs`) and `build_router(engine: EngineHandle)` (`crates/lumen-server/src/router.rs`). See `crates/lumen-server/tests/server_integration.rs` for a fully wired reference.

Endpoints:

```
POST /v1/chat/completions   # OpenAI-compatible, SSE streaming
POST /v1/completions        # OpenAI-compatible
POST /v1/messages           # Anthropic-compatible, SSE streaming
```

Both wire formats support SSE streaming. The tool-call parser is template-driven; v1 (current) ships the Qwen3.5 `<tool_call>` / `</tool_call>` marker pattern with streaming partial-marker hold-back, and additional templates are straightforward to register as more model families ship. Reference embedder: `crates/lumen-server/tests/server_integration.rs`. Reference binary: `crates/lumen-server/src/bin/lumen-server.rs`.

## Architecture

```
lumen-format      LBC binary format, quantization descriptors, test model generators
lumen-convert     GGUF -> LBC converter (v1 architectures: qwen35 dense, qwen35moe MoE)
lumen-runtime     CUDA backend (200+ NVRTC kernels across ~34 families), Metal backend (MSL shaders),
                  naive CPU + SIMD NEON references, KV cache (memory + disk),
                  GDN recurrent state, sampling, sessions, suffix prefill
lumen-server      axum HTTP server: OpenAI + Anthropic SSE endpoints, tool calling
lumen-bench       benchmark harness with JSON + table output
lumen-cli         CLI: built-in BPE tokenizer, model registry, HuggingFace downloader
```

**Forward pass.** v1's shipped instances (Qwen3.5-9B dense, Qwen3.5-MoE) share an L=32 layer stack of hybrid GDN linear-attention layers (24) interleaved with full-attention layers (8), with a fused gate+up+SwiGLU+down FFN (dense) or top-k expert dispatch (MoE).

For the forward-pass details, the `.lbc` on-disk format, and suffix-prefill cache reuse, see [`docs/architecture.md`](docs/architecture.md).

## Performance

Source: [`bench/RESULTS.md`](bench/RESULTS.md). Methodology: [`bench/METHODOLOGY.md`](bench/METHODOLOGY.md). 5-trial median, batch=1, `--temperature 0 --seed 42`.

### Performance summary (2026-06-02)

| Hardware | Model | Quant | Decode × llama.cpp | Prefill × llama.cpp |
|---|---|---|---|---|
| A100-80GB PCIe | Qwen3.5-9B dense | Q8_0 | **0.91×** | (structural < 1.0×) |
| A100-80GB PCIe | Qwen3.5-9B dense | BF16 | **0.93–0.94×** | (structural < 1.0×) |
| A100-80GB PCIe | Qwen3.5-9B dense | Q4_0 | 0.64× | (structural < 1.0×) |
| A100-80GB PCIe | MoE-30B-A3B | BF16 (recommended) | **0.902×** | (structural < 1.0×) |
| A100-80GB PCIe | MoE-30B-A3B | Q8_0 | 0.584× | — |
| A100-80GB PCIe | MoE-30B-A3B | Q4_0 | 0.674× | — |
| M3 Ultra | Qwen3.5-9B dense | Q8_0 | **0.98×** | **0.95×** |
| M3 Ultra | Qwen3.5-9B dense | Q4_0 | **1.02×** / **1.17×** (beats llama.cpp) | 0.88× |
| M3 Ultra | Qwen3.5-9B dense | BF16 | 0.83× | 0.66× |
| M3 Ultra | MoE-30B-A3B (Q8/Q4) | — | functional, Lumen is **sole provider on Apple Silicon** (llama-bench 8680 cannot load GDN MoE) | — |

**Workload-weighted decode** (mean across short/medium/long/code/multi-turn): Q8 76.4 = 0.551× llama.cpp, Q4 99.5 = 0.640× llama.cpp, BF16 87.4 = 0.883× llama.cpp. Realistic workload-weighted decode runs 3-7% below the canonical 5-trial median.

**Prefill ratio versus llama.cpp** is structurally below 1.0× across all CUDA cells (NVRTC compute_61 PTX ISA + non-monolithic encoder ceiling). Lumen wins on decode latency at batch=1.

BF16 MoE prefill uses an llama.cpp-port mmvf kernel; **`LUMEN_CUDA_BF16_GEMMEX=0`** is required for BF16 P3 correctness on MoE. See [bench/METHODOLOGY.md](bench/METHODOLOGY.md) for the full 12-flag canonical env stack.

### Long context

Tiled streaming-softmax decode kernel is the default for all sequence lengths (`ATTN_DECODE_TILED_DEFAULT_THRESHOLD = 0` at `crates/lumen-runtime/src/cuda/decode.rs`). To force the legacy single-block path, set `LUMEN_CUDA_DECODE_TILED_THRESHOLD=4294967295`. End-to-end decode at 65,536 tokens validated on A100-80GB across Q8_0 / Q4_0 / BF16; 98,304-token decode validated for Q4_0 / Q8_0. The GDN prefill grid-Y limit at 98K+ tokens was closed by sub-batched dispatch.

## Comparison & benchmarks

The `bench/` directory contains the canonical benchmark harness. The published benchmark suite is currently scoped to the v1 model family (Qwen3.5); benchmarks for additional model families will be added as they ship. See [`bench/RESULTS.md`](bench/RESULTS.md) for the full numbers and [`bench/METHODOLOGY.md`](bench/METHODOLOGY.md) for the methodology. On the measured Qwen3.5-9B BF16 cell at batch=1, Lumen's decode is **3.05× vLLM 0.21.0** and prefill is **2.62× vLLM** (`bench/RESULTS.md` § "BF16 — Lumen vs vLLM 0.21.0"). llama.cpp ratios are in the summary table above. vLLM's strengths surface at larger batch sizes, which is not the configuration measured here.

## GPU memory

| Quant | Qwen3.5-9B (peak VRAM) | Qwen3.5-MoE-30B-A3B (peak VRAM, 5-trial) |
|-------|-----------------------:|-------------------------------------------:|
| Q4_0  | ~5.1 GB                | **24.1 GB** (LBC 20.7 GB)                  |
| Q8_0  | ~10.0 GB / ~22.9 GB (with cuBLAS workspace + cache) | **54.3 GB** (LBC 37.6 GB) |
| BF16  | ~17.8 GB               | **72.4 GB** (LBC 69.7 GB) — see warning below |

Qwen3.5-MoE-30B-A3B at all three quants fits on a single A100-80GB (validated).

**BF16 MoE-30B-A3B headroom warning**: peak 72.4 GB on 80 GB A100 leaves only ~7.6 GB headroom. Any concurrent process consuming >5 GB can race `cuMemAlloc` and cause OOM mid-upload. **In a multi-tenant deployment, BF16 MoE requires a dedicated 80 GB+ GPU reservation.** No co-tenant workloads. For shared-GPU deployments, use Q8 (54 GB peak) or Q4 (24 GB peak).

KV cache is auto-sized to fit remaining VRAM; `--context-len` overrides. KV growth is bit-perfect to the theoretical formula: `max_seq_len × num_layers × num_kv_heads × head_dim × 4 (F32) × 2 (K+V)`.

## Production deployment

Full operator runbook lives in [`docs/production.md`](docs/production.md). Key requirements before deploying:

- For concurrent clients deploy `lumen-server`, not repeated `lumen run` invocations (the CLI cold-loads ~60–120 s per call; 16-client burst measured 82.4% timeout rate).
- **BF16 MoE-30B-A3B requires a dedicated 80 GB+ GPU** — peak VRAM 72.4 GB on 80 GB A100. No co-tenant workloads.
- For multilingual / long-form prompts pass `--max-tokens 512` minimum (chat template opens `<think>…</think>`).
- PURE-greedy (`--temperature 0`) on long generations (≥512 tokens) deterministically loops; use `--temperature 0.7` OR `--repetition-penalty 1.05 --repeat-last-n 64`.
- Canonical env stack (CUDA, MoE-30B-A3B BF16 0.9× llama.cpp gate): all 12 flags default-ON. Critically **`LUMEN_CUDA_BF16_GEMMEX=0`** is required for BF16 P3 correctness. Full stack in [`bench/METHODOLOGY.md`](bench/METHODOLOGY.md).
- LBC format compatibility: `LBC_VERSION = 3`. Rebuild LBCs after major Lumen upgrades via `lumen convert` or `lumen pull`.

See [`docs/production.md`](docs/production.md) for serving-mode selection, GPU-reservation policy per quant, and known limitations.

## Build

**Requirements:**
- Rust 1.75+
- For CUDA: an NVIDIA GPU + driver (no build-time CUDA SDK needed; kernels compile at runtime via NVRTC)
- For Metal: macOS on Apple Silicon (no extra deps; ships with the OS)

```bash
# CUDA (Linux)
cargo build --release --features cuda

# Metal (macOS)
cargo build --release

# Workspace tests (no GPU required for the CPU reference suite)
cargo test --workspace --release
```

`cargo install --path crates/lumen-cli --features cuda` produces a single static `lumen` binary.

## Tests

```bash
cargo test --workspace --release
```

The CPU reference suite (`lumen-convert`, `lumen-cli`, `lumen-format`) runs without a GPU; the `lumen-runtime` GPU paths require the matching backend (Apple Silicon for Metal, an NVIDIA GPU + `--features cuda` for CUDA) and SKIP without the driver. CI runs the workspace minus the GPU-required tests. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the build-and-test workflow.

## Status

Production-ready on both NVIDIA CUDA (compute capability 8.0+) and Apple Silicon (M-series) for the Qwen3.5 family. Decode throughput relative to llama.cpp is reported in [benchmarks](bench/RESULTS.md). The public API and the binary `.lbc` format are not yet stable.

Roadmap and history: [`CHANGELOG.md`](CHANGELOG.md).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
