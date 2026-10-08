<p align="center">
  <img src="docs/assets/lumen.png" alt="Lumen" width="720">
</p>

# Lumen

[servelumen.com](https://servelumen.com) ·
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
lumen run qwen3.5-9b:q8_0 "Write a haiku about light"
```

That one command downloads the model on first use, converts it, picks your backend (Metal on Apple Silicon, CUDA on NVIDIA), and streams tokens.

> **Status:** Serves the Qwen3.5 / Qwen3.8 models (dense 9B, dense 27B, and MoE-35B-A3B) on NVIDIA CUDA (compute capability 8.0+) and Apple Silicon (M-series); each model, format and GPU's status is in [Model support](docs/support.md). The public API and the binary `.lbc` format are not yet stable — **read [Production deployment](docs/production.md) before deploying.**

## Quick start

Get Lumen one of two ways, then run.

**Option A — pre-built binary** (no Rust toolchain). One command detects your platform (macOS → Metal, Linux x86_64 + NVIDIA → CUDA) and installs the matching validated binary; models download on first use (`--model <name>` downloads one during the install):

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
lumen run qwen3.5-9b:q8_0 "Write a haiku about light"
lumen run qwen3.5-moe:q4_0 "Explain quantum computing in one paragraph"   # mixture-of-experts
```

More on installing, pulling and running: **[Getting started](docs/getting-started.md)**.

## What it is

- **One self-contained binary, zero ML dependencies** — native CUDA C and Metal MSL kernels, a native BPE tokenizer, and the native `.lbc` model format, all in Rust. No PyTorch, no ONNX, no Python runtime.
- **No build-time CUDA SDK** — kernels JIT-compile at runtime via NVRTC, so one CUDA build runs on any compute-capability-8.0+ device.
- **Download → convert → run** in a single command; weights stay GPU-resident for fast batch-1 decode.
- **OpenAI- and Anthropic-compatible HTTP server** with SSE streaming and template-driven tool calls; optional per-request reasoning / extended thinking.
- **Text to image** with Qwen-Image-2.1 on CUDA, from a separate image-only server.
- **Runs on NVIDIA cc 8.0+ (Ampere, Hopper, Blackwell RTX 5090) and Apple Silicon (M-series)**; a scalar + SIMD CPU path is the correctness reference.
- **Tuned for interactive serving** — single-stream, GPU-resident decode latency, not large-batch throughput.

## Supported models & hardware

v1 (current) verifies the Qwen3.5 family and the Qwen3.8-27B dense model end-to-end; more model families are planned.

| Model | Architecture | Parameters | Quants |
|-------|--------------|------------|--------|
| `qwen3.5-9b` | Dense GDN-hybrid | 9B | Q8_0, Q4_0, BF16 |
| `qwen3.8-27b` | Dense GDN-hybrid | 27B | Q8_0, Q4_0, BF16, Q4_K_M (16.2 GiB download), Q5_K_M (19.5 GiB download) |
| `qwen3.5-moe` | MoE GDN-hybrid | 35B total / 3B active | Q8_0, Q4_0, BF16 |

The two K-quant cells are served as stored on CUDA only. On Apple Silicon (the Metal
conversion target) the converter converts them exactly as it always has — every K-quant
layer plane upcast to Q8_0, the head re-quantised, a K-quant embedding dequantised to F32 —
so the artifact comes out larger than the `Q8_0` one while carrying the source's coarser
precision; prefer `Q8_0` or `Q4_0` there.

The 27B also runs from Hugging Face checkpoints imported as stored with `lumen convert
--from-hf` ([how](docs/lbc-format.md)), on CUDA only: NVIDIA ModelOpt NVFP4 + FP8, and
compressed-tensors INT4 (group 32).

| Backend | Hardware | Formats |
|---------|----------|--------|
| **CUDA** | NVIDIA, compute capability 8.0+ (e.g. A100, H100, RTX 5090) | Every format above; the BF16 27B and MoE need an 80 GB H100-class GPU |
| **Metal** | Apple Silicon (M-series) | Q8_0 and Q4_0 for every model, BF16 for the 9B |
| **CPU** | Scalar reference + SIMD NEON | Correctness reference, not throughput-optimized |

Image: Qwen-Image-2.1 on CUDA — `lumen pull qwen-image` downloads and converts it, `lumen image "<prompt>"` makes a picture, `lumen-server qwen-image` serves it — **[docs/image-generation.md](docs/image-generation.md)**.

`lumen models` lists what is available and disk-cached. Per-model, per-format status and verification: **[docs/support.md](docs/support.md)**.

## HTTP server

For concurrent clients, run the long-lived server (not repeated `lumen run`):

```bash
lumen pull qwen3.5-9b:q8_0                            # optional: the server fetches a missing model
lumen-server qwen3.5-9b:q8_0                          # defaults: port 8000, auto backend
# (explicit form: lumen-server --model qwen3.5-9b --quant q8_0 --port 8000)
curl http://localhost:8000/v1/models
```

```text
POST /v1/chat/completions   # OpenAI-compatible, SSE streaming
POST /v1/completions        # OpenAI-compatible, SSE streaming
POST /v1/messages           # Anthropic-compatible, SSE streaming
```

Text to image runs as its own server: `lumen pull qwen-image`, then `lumen-server qwen-image` (a `--features image` build, which the Linux/CUDA release is) serves `POST /v1/images/generations` ([docs/image-generation.md](docs/image-generation.md)).

On CUDA, `--kv-precision bf16` halves the KV cache's memory for long contexts. Wire formats, reasoning / extended thinking, sampling & reproducibility, and embedding the engine as a library: **[docs/server.md](docs/server.md)**.

## Performance

Lumen optimizes for **batch-1, GPU-resident decode latency** — single-stream interactive serving.

Retained decode record (Qwen3.5-MoE-35B-A3B, 5 runs per engine; Q8/Q4
co-located — both engines in one A100 container; BF16 measured in separate
same-GPU H100 batteries):

| Quant | GPU | Decode (tok/s) | × llama.cpp |
|-------|-----|---------------:|------------:|
| Q8_0  | A100-80GB | 79.2 | 0.567× |
| Q4_0  | A100-80GB | 93.6 | 0.598× |
| BF16  | H100 | 104.1 | 0.575× |

Per-cell decode + prefill numbers for the 9B and the MoE (A100-80GB, M3 Ultra), methodology, and baseline comparisons: **[bench/RESULTS.md](bench/RESULTS.md)**; recorded RTX 5090 numbers: **[docs/support.md](docs/support.md)**.

## Architecture

```text
lumen-format      LBC binary format, quantization descriptors, test model generators
lumen-convert     GGUF and Hugging Face checkpoint -> LBC converter (qwen35, qwen35moe)
lumen-runtime     CUDA backend (200+ NVRTC kernels), Metal backend (MSL shaders),
                  CPU + SIMD NEON references, KV cache (memory + disk),
                  GDN recurrent state, sampling, sessions, suffix prefill
lumen-server      axum HTTP server: OpenAI + Anthropic SSE endpoints, tool calling
lumen-bench       benchmark harness with JSON + table output
lumen-cli         CLI: built-in BPE tokenizer, model registry, HuggingFace downloader
lumen-image       Qwen-Image-2.1 text-to-image (CUDA) and the lbi-convert converter
```

The shipped models interleave GDN linear-attention layers with full-attention layers (32 layers in the 9B, 40 in the MoE, 64 in the 27B), with a SwiGLU FFN (dense) or top-k expert dispatch (MoE). Forward-pass details, the `.lbc` on-disk format, and suffix-prefill cache reuse: **[docs/architecture.md](docs/architecture.md)**.

## Building & testing

For development — build the workspace and run the test suite (the install commands are in [Quick start](#quick-start) above):

```bash
cargo build --release                  # Metal (macOS)
cargo build --release --features cuda  # CUDA (Linux)
cargo test --workspace --release       # CPU reference suite needs no GPU
```

Rust is pinned via `rust-toolchain.toml`; CUDA needs `libnvrtc` + `libcublas` present at run time (no build-time SDK). Dev workflow: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Documentation

- [Getting started](docs/getting-started.md) — install, pull, run (binaries, source)
- [CLI reference](docs/cli.md) — all subcommands and flags (or `lumen run --help`)
- [HTTP server](docs/server.md) — endpoints, reasoning, library embedding
- [Image generation](docs/image-generation.md) — Qwen-Image-2.1 from an image-only server
- [Model support](docs/support.md) — live support matrix and verification status
- [Production deployment](docs/production.md) — serving mode, GPU sizing, known limitations
- [Environment variables](docs/environment-variables.md) — `LUMEN_*` runtime flags
- [Benchmarks](bench/RESULTS.md) — per-cell numbers and methodology
- [Releases & versioning](RELEASING.md) · [Changelog](CHANGELOG.md)

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. Third-party notices for ported kernel code: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
