# Lumen

[Getting started](docs/getting-started.md) ·
[Models](docs/support.md) ·
[Server](docs/server.md) ·
[Production](docs/production.md) ·
[Benchmarks](bench/RESULTS.md) ·
[Releases](RELEASING.md) ·
[Changelog](CHANGELOG.md)

**LLM inference in Rust, for Apple Silicon and NVIDIA CUDA.**

A single binary that downloads a model, runs it GPU-resident, and prints text — built from scratch with zero ML dependencies (no PyTorch, no ONNX, no Python), native CUDA C and Metal kernels, a native tokenizer, and a native model format.

```bash
lumen run qwen3.5-9b:q8_0 "Write a haiku about Rust"
```

That one command downloads the model on first use, converts it, picks your backend (Metal on Apple Silicon, CUDA on NVIDIA), and streams tokens.

> **Status:** Production-ready for the shipped Qwen3.5 / Qwen3.6 / Qwen3.8 models (dense 9B, dense 27B, and MoE-30B-A3B) on NVIDIA CUDA (compute capability 8.0+) and Apple Silicon (M-series). The public API and the binary `.lbc` format are not yet stable — **read [Production deployment](docs/production.md) before deploying.**

## Quick start

Get Lumen one of two ways, then run.

**Option A — pre-built binary** (no Rust toolchain). One command detects your platform (macOS → Metal, Linux x86_64 + NVIDIA → CUDA), installs the matching validated binary, and helps you set up a model:

```bash
curl -fsSL https://servelumen.com/install.sh | bash
```

**Option B — build from source** (Rust toolchain):

```bash
git clone https://github.com/faisalmumtaz89/Lumen && cd Lumen
cargo install --path crates/lumen-cli                  # Apple Silicon (Metal)
cargo install --path crates/lumen-cli --features cuda   # NVIDIA Linux (CUDA)
```

(That installs the `lumen` CLI. For the `lumen-server` binary too: `cargo install --path crates/lumen-server --features bin` — append `,cuda` on NVIDIA. Option A installs both.)

**Run** — the model auto-downloads + converts on first use, then runs GPU-resident on your backend (Metal on Apple Silicon, CUDA on NVIDIA):

```bash
lumen run qwen3.5-9b:q8_0 "Write a haiku about Rust"
lumen run qwen3.5-moe:q4_0 "Explain quantum computing in one paragraph"   # mixture-of-experts
```

More install paths (Docker, the one-command `clone → running server` script): **[Getting started](docs/getting-started.md)**.

## What it is

- **One self-contained binary, zero ML dependencies** — native CUDA C and Metal MSL kernels, a native BPE tokenizer, and the native `.lbc` model format, all in Rust. No PyTorch, no ONNX, no Python runtime.
- **No build-time CUDA SDK** — kernels JIT-compile at runtime via NVRTC, so one CUDA build runs on any compute-capability-8.0+ device (driver-only).
- **Download → convert → run** in a single command; weights stay GPU-resident for fast batch-1 decode.
- **OpenAI- and Anthropic-compatible HTTP server** with SSE streaming and template-driven tool calls; optional per-request reasoning / extended thinking.
- **Runs on NVIDIA cc 8.0+ (Ampere / Hopper) and Apple Silicon (M-series)**; a scalar + SIMD CPU path is the correctness reference.
- **Tuned for interactive serving** — single-stream, GPU-resident decode latency, not large-batch throughput.

## Supported models & hardware

v1 (current) verifies the Qwen3.5 family and the Qwen3.6-27B / Qwen3.8-27B dense models end-to-end; more model families are planned.

| Model | Architecture | Parameters | Quants |
|-------|--------------|------------|--------|
| `qwen3.5-9b` | Dense GDN-hybrid | 9B | Q8_0, Q4_0, BF16 |
| `qwen3.6-27b` | Dense GDN-hybrid | 27B | Q8_0, Q4_0, BF16 |
| `qwen3.8-27b` | Dense GDN-hybrid | 27B | Q8_0, Q4_0, BF16 |
| `qwen3.5-moe` | MoE GDN-hybrid | 30B total / 3B active | Q8_0, Q4_0, BF16 |

| Backend | Hardware | Status |
|---------|----------|--------|
| **CUDA** | NVIDIA, compute capability 8.0+ (e.g. A100, H100) | Production-ready |
| **Metal** | Apple Silicon (M-series) | Production-ready |
| **CPU** | Scalar reference + SIMD NEON | Correctness reference, not throughput-optimized |

`lumen models` lists what is available and disk-cached. Live support matrix and per-config verification status: **[docs/support.md](docs/support.md)**.

## HTTP server

For concurrent clients, run the long-lived server (not repeated `lumen run`):

```bash
lumen pull qwen3.5-9b:q8_0
lumen-server qwen3.5-9b:q8_0                          # defaults: port 8000, auto backend
# (explicit form: lumen-server --model qwen3.5-9b --quant q8_0 --port 8000)
curl http://localhost:8000/v1/models
```

```text
POST /v1/chat/completions   # OpenAI-compatible, SSE streaming
POST /v1/completions        # OpenAI-compatible
POST /v1/messages           # Anthropic-compatible, SSE streaming
```

Wire formats, reasoning / extended thinking, sampling & reproducibility, and embedding the engine as a library: **[docs/server.md](docs/server.md)**.

## Performance

Lumen optimizes for **batch-1, GPU-resident decode latency** — single-stream interactive serving.

Workload-weighted decode (Qwen3.5-MoE-30B-A3B, A100-80GB, mean across short / medium / long / code / multi-turn):

| Quant | Decode (tok/s) |
|-------|---------------:|
| Q8_0  | 76.4 |
| Q4_0  | 99.5 |
| BF16  | 87.4 |

Full per-cell decode + prefill numbers (every model × quant, on A100-80GB and M3 Ultra), methodology, and baseline comparisons: **[bench/RESULTS.md](bench/RESULTS.md)**. Long-context decode is validated to 65K+ tokens.

## Architecture

```text
lumen-format      LBC binary format, quantization descriptors, test model generators
lumen-convert     GGUF -> LBC converter (qwen35 dense, qwen35moe MoE)
lumen-runtime     CUDA backend (200+ NVRTC kernels), Metal backend (MSL shaders),
                  CPU + SIMD NEON references, KV cache (memory + disk),
                  GDN recurrent state, sampling, sessions, suffix prefill
lumen-server      axum HTTP server: OpenAI + Anthropic SSE endpoints, tool calling
lumen-bench       benchmark harness with JSON + table output
lumen-cli         CLI: built-in BPE tokenizer, model registry, HuggingFace downloader
```

Shipped instances share an L=32 stack of GDN linear-attention layers interleaved with full-attention layers, with a fused gate+up+SwiGLU+down FFN (dense) or top-k expert dispatch (MoE). Forward-pass details, the `.lbc` on-disk format, and suffix-prefill cache reuse: **[docs/architecture.md](docs/architecture.md)**.

## Building & testing

For development — build the workspace and run the test suite (the install commands are in [Quick start](#quick-start) above):

```bash
cargo build --release                  # Metal (macOS)
cargo build --release --features cuda  # CUDA (Linux)
cargo test --workspace --release       # CPU reference suite needs no GPU
```

Rust is pinned via `rust-toolchain.toml`; CUDA needs `libnvrtc` + `libcublas` present at run time (no build-time SDK). Dev workflow: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Documentation

- [Getting started](docs/getting-started.md) — install, pull, run (binaries, Docker, source)
- [CLI reference](docs/cli.md) — all subcommands and flags (or `lumen run --help`)
- [HTTP server](docs/server.md) — endpoints, reasoning, library embedding
- [Model support](docs/support.md) — live support matrix and verification status
- [Production deployment](docs/production.md) — serving mode, GPU sizing, known limitations
- [Environment variables](docs/environment-variables.md) — all `LUMEN_*` flags
- [Benchmarks](bench/RESULTS.md) — per-cell numbers and methodology
- [Releases & versioning](RELEASING.md) · [Changelog](CHANGELOG.md)

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
