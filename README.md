# Lumen

LLM inference engine in Rust with Metal and CUDA GPU backends.

Lumen is built from scratch with zero external ML dependencies — no PyTorch, no ONNX, no Python runtime. Every kernel, every tokenizer, every format parser is native Rust and GPU shader code. The result is a single binary that downloads a model, tokenizes your prompt, runs GPU-resident inference, and prints text.

> **Status:** Active development. Produces correct output across 575+ tests. API and binary format not yet stable.

## Features

- **GPU-resident inference** — all weights stay in GPU memory, zero per-token host/device transfers
- **Metal** (178 kernels) and **CUDA** (157 kernels) backends, compiled at runtime from embedded source
- **Built-in BPE tokenizer** — reads vocabulary from GGUF, embeds in LBC format, no Python needed
- **Auto-download** — `lumen run model:quant "prompt"` fetches from HuggingFace, converts, caches, runs
- Custom **LBC** binary format with 128 KiB aligned blobs for zero-copy GPU mapping
- 8 model architectures including GatedDeltaNet linear attention (Qwen3.5)
- F16, Q8_0, Q4_0, Q4_1 runtime quantization with automatic K-quant/MXFP4 conversion
- Mixture of Experts with LFU expert cache and parallel NVMe streaming (Metal)
- Zero external ML dependencies — single `cargo install`, nothing else

## Quick Start

```bash
# Install
cargo install --path crates/lumen-cli

# Run (auto-downloads model on first use)
lumen run qwen2.5-3b:q8_0 "What is the meaning of life?"
```

Backend is auto-detected (Metal on macOS, CUDA on Linux with NVIDIA GPU, CPU otherwise). To choose explicitly: `--metal`, `--cuda`, or `--simd`.

## Supported Models

| Architecture | Example Models | Metal | CUDA |
|-------------|----------------|:-----:|:----:|
| `llama` | LLaMA 2/3, TinyLlama, CodeLlama | Yes | Yes |
| `mistral` | Mistral 7B | Yes | Yes |
| `qwen2` | Qwen2, Qwen2.5 | Yes | Yes |
| `qwen35` | Qwen3.5 9B (GatedDeltaNet) | Yes | Yes |
| `qwen35moe` | Qwen3.5 35B-A3B (MoE) | Yes | -- |
| `internlm2` | InternLM2 | Yes | Yes |
| `exaone` | EXAONE | Yes | Yes |
| `xverse` | XVERSE | Yes | Yes |

**Quantization:** F16, Q8_0, Q4_0, Q4_1 (Metal). K-quant and MXFP4 models are automatically converted during import (see [LBC Format](#lbc-format)).

### Model Registry

Use `name:quant` syntax to refer to models. Available names:

| Name | Model | Quants |
|------|-------|--------|
| `qwen2.5-3b` | Qwen2.5 3B Instruct | F16, Q8_0, Q4_0 |
| `qwen2.5-7b` | Qwen2.5 7B Instruct | Q8_0, Q4_0 |
| `qwen2.5-14b` | Qwen2.5 14B Instruct | Q8_0, Q4_0 |
| `llama-8b` | Llama 3.1 8B Instruct | F16, Q8_0 |
| `mistral-7b` | Mistral 7B Instruct v0.3 | Q8_0 |
| `qwen3.5-9b` | Qwen3.5 9B | Q8_0 |
| `tinyllama` | TinyLlama 1.1B Chat | Q8_0, Q4_0 |

```bash
lumen run tinyllama:q4_0 "Write a haiku about Rust"
lumen run llama-8b:q8_0 "Explain quantum computing" --max-tokens 200
lumen models   # list cached and available models
lumen pull qwen2.5-7b:q8_0   # download without running
```

## CLI

```bash
# Raw token mode (for benchmarking and debugging)
lumen run --model model.lbc --tokens "1 2 3" --max-tokens 128 --metal
lumen run --model model.lbc --tokens "1 2 3" --cuda --cuda-device 1

# Convert GGUF to LBC manually
lumen convert --input model.gguf --output model.lbc
lumen convert --input model.gguf --output model.lbc --requant q4_0
```

| Flag | Description |
|------|-------------|
| `--system <text>` | System prompt |
| `--max-tokens <n>` | Tokens to generate (default: unlimited, stops at EOS) |
| `--temperature <f>` | Sampling temperature, 0 = greedy (default: 0.8) |
| `--seed <n>` | Random seed for reproducibility (default: random) |
| `--metal` | Force Metal GPU backend (macOS) |
| `--cuda` | Force CUDA GPU backend (NVIDIA) |
| `--cuda-device <n>` | CUDA device ordinal (default: 0) |
| `--simd` | CPU backend (ARM NEON SIMD) |
| `--context-len <n>` | KV cache size (auto-sized by default) |
| `--no-gpu-resident` | Stream weights from disk instead of GPU memory |
| `--verbose` | Show diagnostics, metrics, and model info |
| `--profile` | Per-operation timing breakdown |

## LBC Format

Lumen uses its own Layer-Blob Container (`.lbc`) format. LBC stores weights in large contiguous blobs with fixed byte offsets, allowing the runtime to seek directly to any tensor without scanning metadata.

**Benefits:**

- **Aligned for Direct I/O:** Layer blobs are perfectly aligned to 128 KiB boundaries, enabling zero-copy memory mapping and direct hardware reads from SSDs without padding overhead.
- **Zero-Copy Loading:** The entire file is memory-mapped. Accessing a layer is instantaneous (a simple pointer clone), with windowed prefetching allowing models to stream effectively even if they exceed available RAM.
- **Fast MoE Routing:** Sub-layer indexing allows loading individual experts by their absolute byte position in parallel without reading the full layer.
- **Streaming Conversion:** Layer sizes are deterministic, allowing conversion in constant memory without holding the full model. 
- **Per-Tensor Quantization:** Mixes multiple quantization schemes (F16, Q8_0, Q4_K, etc.) securely in the same file with a CRC32-validated header and backward-compatible sections.

The converter handles all GGUF v2/v3 models, streams one layer at a time, and automatically dequantizes K-quant tensors (Q4_K, Q5_K, Q6_K, Q2_K, Q3_K) and MXFP4 to runtime-supported formats.

## Performance

Decode throughput (tok/s) on Q8_0 models. pp128 + gen128. Same weights across all engines.

**CUDA** (NVIDIA A100-80GB):

| Model | Lumen | llama.cpp |
|-------|------:|----------:|
| Qwen2.5 3B | **228** | 214 |
| Qwen2.5 7B | **153** | 136 |
| Llama 3.1 8B | **141** | 131 |
| Qwen2.5 14B | **79** | 71 |

**Metal** (Apple M3 Ultra, 96 GB):

| Model | Lumen | llama.cpp | MLX |
|-------|------:|----------:|----:|
| TinyLlama 1.1B | **307** | 221 | 422 |
| Llama 3.1 8B | **73** | 64 | 76 |
| Llama 3.1 8B Q4_0 | **97** | 92 | -- |

Full results with hardware and engine versions: [`bench/RESULTS.md`](bench/RESULTS.md). Methodology: [`bench/METHODOLOGY.md`](bench/METHODOLOGY.md).

### GPU Memory (LBC file size ≈ GPU-resident memory)

| Model | Q4_0 | Q8_0 | F16 |
|-------|-----:|-----:|----:|
| TinyLlama 1.1B | 0.6 GB | 1.1 GB | 2.1 GB |
| Qwen2.5 3B | 2.6 GB | 3.1 GB | 5.8 GB |
| Llama 3.1 8B | -- | 8.0 GB | -- |
| Qwen2.5 7B | -- | 7.5 GB | -- |
| Qwen3.5 9B | -- | 10.0 GB | -- |

## Building from Source

**Requirements:** Rust 1.75+. macOS for Metal. NVIDIA GPU + driver for CUDA (no build-time SDK needed).

```bash
cargo build --release                    # Metal + CPU
cargo build --release --features cuda    # + CUDA
cargo test --workspace --release         # run tests
```

Kernels are compiled at runtime from embedded source — no external shader files or build steps.

## Architecture

```
lumen-format      LBC binary format, quantization descriptors, test model generators
lumen-convert     GGUF-to-LBC converter, 8 architecture mappings, 10 dequant kernels
lumen-runtime     CUDA backend, Metal backend (178 kernels), SIMD backend,
                  KV cache, GDN state, MoE expert cache, weight providers
lumen-bench       Benchmark harness with JSON + table output
lumen-cli         CLI with built-in BPE tokenizer, model registry, HuggingFace downloader
```

## Contributing

Contributions are welcome. The codebase is pure Rust with GPU shaders embedded as string constants — no external build tools or shader compilers needed. Run `cargo test --release --lib` to verify changes. See open issues for areas that need work.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
