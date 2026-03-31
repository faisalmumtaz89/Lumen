# Lumen

LLM inference engine in Rust with Metal and CUDA GPU backends.

> **Status:** Active development. Produces correct output across 500+ tests. API and binary format not yet stable.

## Features

- **Metal** (Apple Silicon) and **CUDA** (NVIDIA) GPU backends
- GPU-resident inference — all weights in GPU memory, zero per-token transfers
- Custom **LBC** binary format optimized for zero-copy GPU mapping and MoE streaming
- GGUF model import with automatic quantization format conversion
- F16, Q8_0, and Q4_0 runtime quantization
- 8 model architectures including GatedDeltaNet linear attention (Qwen3.5)
- Mixture of Experts support (Metal)
- Zero external ML dependencies

## Quick Start

```bash
# Build (Metal + CPU backends)
cargo build --release

# Build with CUDA
cargo build --release --features cuda

# Convert GGUF to LBC
./target/release/lumen convert --input model.gguf --output model.lbc

# Run inference
./target/release/lumen run --model model.lbc --tokens "1 2 3" --max-tokens 128 --metal
./target/release/lumen run --model model.lbc --tokens "1 2 3" --max-tokens 128 --cuda
```

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

**Quantization:** F16, Q8_0, Q4_0, Q4_1 (Metal). The GGUF converter handles K-quant formats (Q4_K, Q5_K, Q6_K, Q2_K, Q3_K) and MXFP4 via automatic dequantization and requantization.

## LBC Format

Lumen uses its own Layer-Blob Container (`.lbc`) format instead of reading GGUF directly. While GGUF requires full-header string parsing to locate individual tensors, LBC stores weights in large contiguous blobs with fixed byte offsets, allowing the runtime to seek instantly without scanning metadata.

```bash
lumen convert --input model.gguf --output model.lbc
lumen convert --input model.gguf --output model.lbc --requant q4_0   # requantize during conversion
```

**Benefits of LBC:**

- **Aligned for Direct I/O:** Layer blobs are perfectly aligned to 128 KiB boundaries, enabling zero-copy memory mapping and direct hardware reads from SSDs without padding overhead.
- **Zero-Copy Loading:** The entire file is memory-mapped. Accessing a layer is instantaneous (a simple pointer clone), with windowed prefetching allowing models to stream effectively even if they exceed available RAM.
- **Fast MoE Routing:** Sub-layer indexing allows loading individual experts by their absolute byte position in parallel without reading the full layer.
- **Streaming Conversion:** Layer sizes are deterministic, allowing conversion in constant memory without holding the full model. 
- **Per-Tensor Quantization:** Mixes multiple quantization schemes (F16, Q8_0, Q4_K, etc.) securely in the same file with a CRC32-validated header and backward-compatible sections.

The converter handles all GGUF v2/v3 models, streams one layer at a time, and automatically dequantizes K-quant tensors (Q4_K, Q5_K, Q6_K, Q2_K, Q3_K) and MXFP4 to runtime-supported formats.

## Performance

Decode and prefill throughput (tok/s). pp128 + gen128. Same model weights across all engines.

### CUDA (NVIDIA A100-SXM4-80GB)

#### Qwen2.5 3B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 153 | 169 | 61 |
| Q8_0 | 225 | 205 | N/A |
| Q4_0 | 217 | 246 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 6,454 | 8,168 | 7,588 |
| Q8_0 | 6,471 | 4,037 | N/A |
| Q4_0 | 6,486 | 3,992 | N/A |

#### Qwen2.5 7B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 95 | 95 | 73 |
| Q8_0 | 153 | 135 | N/A |
| Q4_0 | 196 | 184 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 5,204 | 4,632 | 8,536 |
| Q8_0 | 5,948 | 3,245 | N/A |
| Q4_0 | 5,202 | 2,975 | N/A |

#### Llama 3.1 8B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 89 | 91 | 67 |
| Q8_0 | 142 | 131 | N/A |
| Q4_0 | 178 | 169 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 4,847 | 4,275 | 7,607 |
| Q8_0 | 5,556 | 3,023 | N/A |
| Q4_0 | 4,812 | 2,947 | N/A |

#### Qwen2.5 14B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 51 | 50 | 44 |
| Q8_0 | 79 | 71 | N/A |
| Q4_0 | 100 | 98 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 2,989 | 2,875 | 5,447 |
| Q8_0 | 3,432 | 1,891 | N/A |
| Q4_0 | 3,413 | 1,948 | N/A |

#### Qwen3.5 9B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| Q8_0 | 59 | 114 | N/A |
| Q4_0 | 70 | 140 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| Q8_0 | 324 | 2,611 | N/A |
| Q4_0 | 316 | 2,806 | N/A |

### Metal (Apple M3 Ultra 96 GB, median of 3 runs)

#### TinyLlama 1.1B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | MLX |
|:-----:|------:|----------:|----:|
| F16 | 194 | 184 | 239 |
| Q8_0 | 305 | 225 | 449 |
| Q4_0 | 319 | 246 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | MLX |
|:-----:|------:|----------:|----:|
| F16 | 4,028 | 5,249 | 4,189 |
| Q8_0 | 4,911 | 5,053 | 4,838 |
| Q4_0 | 4,525 | 5,201 | N/A |

#### Llama 3.1 8B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | MLX |
|:-----:|------:|----------:|----:|
| F16 | 40 | 42 | N/A |
| Q8_0 | 73 | 67 | 79 |
| Q4_0 | 98 | 95 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | MLX |
|:-----:|------:|----------:|----:|
| F16 | 766 | 1,050 | N/A |
| Q8_0 | 838 | 1,003 | 906 |
| Q4_0 | 796 | 1,028 | N/A |

#### Qwen3.5 9B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | MLX |
|:-----:|------:|----------:|----:|
| Q8_0 | 57 | N/A | 92 |
| Q4_0 | 68 | N/A | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | MLX |
|:-----:|------:|----------:|----:|
| Q8_0 | 343 | N/A | 747 |
| Q4_0 | 332 | N/A | N/A |

vLLM: F16 only (GGUF quantized path is experimental — see report). MLX Q4_0: different encoding, not comparable. llama.cpp: does not support Qwen3.5 on Metal.

Methodology in [`bench/METHODOLOGY.md`](bench/METHODOLOGY.md). Full report in [`bench/BENCHMARK_REPORT.md`](bench/BENCHMARK_REPORT.md).

## Building from Source

**Requirements:** Rust 1.75+. macOS for Metal. NVIDIA GPU + driver for CUDA (no build-time SDK needed).

```bash
cargo build --release                    # Metal + CPU
cargo build --release --features cuda    # + CUDA
cargo test --workspace --release         # run tests
```

Kernels are compiled at runtime from embedded source — no external shader files or build steps.

## CLI

```bash
lumen run --model model.lbc --tokens "1 2 3" --max-tokens 128 --metal
lumen run --model model.lbc --tokens "1 2 3" --max-tokens 128 --cuda
lumen run --model model.lbc --tokens "1 2 3" --cuda --cuda-device 1
```

| Flag | Description |
|------|-------------|
| `--metal` | Metal GPU backend (macOS) |
| `--cuda` | CUDA GPU backend (NVIDIA) |
| `--cuda-device <n>` | CUDA device ordinal (default: 0) |
| `--simd` | ARM NEON SIMD backend |
| `--max-tokens <n>` | Tokens to generate (default: 10) |
| `--temperature <f>` | Sampling temperature, 0 = greedy (default: 1.0) |
| `--context-len <n>` | KV cache size (auto-sized by default) |
| `--no-gpu-resident` | Stream weights from disk instead of GPU-resident |
| `--profile` | Print timing breakdown |

## Architecture

```
lumen-format      LBC binary format, quantization descriptors, test model generators
lumen-convert     GGUF-to-LBC converter, 8 architecture mappings, 10 dequant kernels
lumen-runtime     CUDA backend, Metal backend (178 kernels), SIMD backend,
                  KV cache, GDN state, MoE expert cache, weight providers
lumen-bench       Benchmark harness with JSON + table output
lumen-cli         CLI entry point
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
