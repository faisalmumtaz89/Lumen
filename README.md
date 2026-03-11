# Lumen

GPU-resident LLM inference engine in Rust.

> **Status:** Lumen is under active development targeting Apple Silicon. It produces correct output, passes 500+ tests, and achieves competitive performance with MLX and llama.cpp on supported models. However, the Rust API, LBC binary format, and CLI interface are not yet stable and may change between releases.

## Quick Start

```bash
cargo build --release
./target/release/lumen convert --input model.gguf --output model.lbc
./target/release/lumen run --model model.lbc --tokens "1 2 3" --max-tokens 128 --metal
```

## Description

Lumen runs large language models on Apple Silicon GPUs with hand-written Metal compute shaders, a custom binary format, and zero external ML dependencies. The entire stack -- 67,000 lines of Rust and 17,000 lines of MSL -- is built from scratch.

- 178 Metal GPU kernels, hand-tuned for Apple Silicon
- Zero external ML dependencies (only `thiserror` and `libc`)
- GPU-resident inference with no per-token CPU-GPU transfers
- First open-source GatedDeltaNet (linear attention) implementation on Metal GPU
- 8 model architectures, 4 runtime quantization formats, 10 dequantization kernels

<details open>
<summary><h2>Benchmarks</h2></summary>

Apple M3 Ultra, 96 GB unified memory, 60 GPU cores. Prompt 128 tokens, generation 128 tokens. Methodology: 3 measured runs + 1 warmup per configuration; MLX runs first on a cold GPU; MLX uses 5 internal trials per run. Full results in [`bench/results/`](bench/results/).

### Decode throughput (tok/s, higher is better)

| Model | Quant | Lumen | MLX | llama.cpp | vs MLX | vs llama.cpp |
|-------|:-----:|------:|----:|----------:|:------:|:------------:|
| TinyLlama 1.1B | F16 | 193 | 231 | 176 | 0.84x | **1.10x** |
| TinyLlama 1.1B | Q8_0 | 314 | 331 | 221 | 0.95x | **1.42x** |
| TinyLlama 1.1B | Q4_0 | 322 | 430 | 247 | 0.75x | **1.31x** |
| Llama 3.1 8B | F16 | 37.3 | 42.5 | 41.3 | 0.88x | 0.90x |
| Llama 3.1 8B | Q8_0 | 71.6 | 76.4 | 65.8 | 0.94x | **1.09x** |
| Llama 3.1 8B | Q4_0 | 95.7 | 124 | 92.8 | 0.77x | **1.03x** |
| Qwen3.5 9B | F16 | 32.2 | 9.1 | -- | **3.54x** | -- |
| Qwen3.5 9B | Q8_0 | 56.1 | 58.6 | -- | 0.96x | -- |
| Qwen3.5 9B | Q4_0 | 65.9 | 88.1 | -- | 0.75x | -- |

### Prefill throughput (tok/s, higher is better)

| Model | Quant | Lumen | MLX | llama.cpp | vs MLX | vs llama.cpp |
|-------|:-----:|------:|----:|----------:|:------:|:------------:|
| TinyLlama 1.1B | F16 | 4,230 | 4,081 | 5,048 | **1.04x** | 0.84x |
| TinyLlama 1.1B | Q8_0 | 4,794 | 4,528 | 4,920 | **1.06x** | 0.97x |
| TinyLlama 1.1B | Q4_0 | 4,631 | 4,662 | 5,140 | 0.99x | 0.90x |
| Llama 3.1 8B | F16 | 758 | 706 | 1,028 | **1.07x** | 0.74x |
| Llama 3.1 8B | Q8_0 | 821 | 884 | 977 | 0.93x | 0.84x |
| Llama 3.1 8B | Q4_0 | 780 | 900 | 1,003 | 0.87x | 0.78x |

**Key takeaways:**

- Beats llama.cpp decode throughput on 5 of 6 comparable configurations (up to 1.42x).
- Within 5-25% of MLX on standard transformer decode, and 3.5x faster on Qwen3.5 F16 where native GatedDeltaNet kernels outperform MLX's generic fallback.
- Qwen3.5 results use GatedDeltaNet (linear attention) layers; llama.cpp does not support this architecture.

</details>

<details>
<summary><h2>Supported Models</h2></summary>

### Architectures

| Architecture | Models | Attention | FFN |
|-------------|--------|-----------|-----|
| `llama` | LLaMA 2/3, TinyLlama, CodeLlama | GQA + RoPE | SwiGLU |
| `mistral` | Mistral 7B | GQA + RoPE | SwiGLU |
| `qwen2` | Qwen2 | GQA + RoPE | SwiGLU |
| `qwen35` | Qwen3.5 9B (dense) | Hybrid GDN + softmax, partial RoPE | SwiGLU |
| `qwen35moe` | Qwen3.5 35B-A3B (MoE) | Hybrid GDN + softmax, partial RoPE | MoE (256 experts, top-8) |
| `internlm2` | InternLM2 | GQA + RoPE | SwiGLU |
| `exaone` | EXAONE | GQA + RoPE | SwiGLU |
| `xverse` | XVERSE | GQA + RoPE | SwiGLU |

</details>

<details>
<summary><h2>Quantization</h2></summary>

### Runtime formats (Metal GPU)

| Format | Bits/weight | Block size | Metal decode | Metal prefill |
|--------|:-----------:|:----------:|:------------:|:-------------:|
| F16 | 16 | -- | Yes | Yes |
| Q8_0 | 8 | 32 | Yes | Yes |
| Q4_0 | 4 | 32 | Yes | Yes |
| Q4_1 | 5.0 | 32 | Yes | Yes |

### Converter dequantization

The GGUF converter additionally supports dequantization from Q5_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, and MXFP4 (10 dequantization kernels total). K-quant tensors (e.g. Q6_K output weights) are automatically requantized to the target format during conversion.

</details>

<details>
<summary><h2>Features</h2></summary>

<details>
<summary>Metal GPU Backend</summary>

- 178 compute kernels in hand-written MSL, compiled at runtime
- GPU-resident inference: all weights in `StorageModePrivate` Metal buffers
- Fused kernels: RMSNorm+GEMM, RoPE+KV-write, gate+up+SwiGLU, flash decode + parallel reduce
- Tiled GEMM for batched prefill with k64 loop optimization
- f16 KV cache to halve attention memory bandwidth
- Flash decode with configurable tile threshold (MHA for short sequences, flash decode for long)
- Concurrent command encoder dispatch for non-dependent operations

</details>

<details>
<summary>GatedDeltaNet (Linear Attention)</summary>

- Full GDN decode pipeline: conv1d, L2 normalize, sigmoid gates, delta-net state update, output gating
- Simdgroup-parallel state kernel (4096 threadgroups of 32 threads, zero threadgroup barriers)
- Transposed h-state layout for coalesced memory access
- Tiled GEMM for GDN prefill (alpha/beta gates, chunked state accumulation)
- Hybrid attention: GDN layers + interleaved full softmax attention layers in the same model

</details>

<details>
<summary>Mixture of Experts (MoE)</summary>

- Batched MoE kernels: fused gate+up+SwiGLU and down+accumulate per expert
- Shared expert support (separate inter dimension)
- O(1) LFU expert cache for SSD-streaming mode
- Expert activation profiler with per-layer entropy tracking
- Optional routing bias for cache-conditional expert selection

</details>

<details>
<summary>GGUF Converter</summary>

- GGUF v2 and v3 parser, zero external dependencies
- 10 dequantization kernels: Q8_0, Q4_0, Q4_1, Q5_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, MXFP4
- Streaming LBC writer with O(1 layer) peak memory
- Optional requantization during conversion (`--requant q4_0`)

</details>

<details>
<summary>LBC Binary Format</summary>

- Custom binary format: `[Header] [LayerIndex] [ExpertIndex?] [Globals] [LayerBlobs]`
- 128 KiB-aligned blobs for direct I/O compatibility
- CRC32 integrity check on header (IEEE polynomial, hand-rolled)
- Expert-granular indexing for MoE models (per-expert byte offsets within layer blobs)

</details>

<details>
<summary>Compute Backends</summary>

| Backend | Target | Description |
|---------|--------|-------------|
| **Metal GPU** | macOS (Apple Silicon) | 178 MSL kernels, GPU-resident, batched prefill, flash decode |
| **ARM NEON SIMD** | aarch64 | Multi-threaded matmul with 4-accumulator FMA pattern, Q8_0 support |
| **Naive scalar** | All platforms | Ground-truth reference implementation, zero `unsafe` in math |
| **Accelerate** | macOS | AMX-backed batched prefill via Apple's BLAS |

</details>

<details>
<summary>Storage and Weight Providers</summary>

| Component | Description |
|-----------|-------------|
| **Mmap provider** | Zero-copy `LayerView` via `mmap(MAP_PRIVATE)`, windowed prefetch with `madvise` |
| **Async provider** | Background I/O thread, lock-free atomic slot states, spin-wait with backoff |
| **Sync provider** | `pread`-based sequential reads, mutex-protected cache |
| **Page cache control** | `madvise(WILLNEED/DONTNEED)`, `mincore` residency checks, `purge_file_cache` |

</details>

</details>

## CLI Reference

```
lumen <COMMAND> [OPTIONS]

COMMANDS:
    run                   Run inference on a model
    convert               Convert a GGUF model to LBC format
    bench                 Run benchmarks
    generate-test-model   Generate a synthetic model for testing
    purge                 Evict a model file from the OS page cache
    help                  Print help
```

### `lumen run`

```bash
lumen run --model model.lbc --tokens "1 2 3" --max-tokens 128 --metal
```

| Flag | Description |
|------|-------------|
| `--model <path>` | Path to LBC model file (required) |
| `--tokens <ids>` | Space-separated input token IDs (required) |
| `--max-tokens <n>` | Maximum tokens to generate (default: 10) |
| `--temperature <f>` | Sampling temperature, 0 = greedy (default: 1.0) |
| `--seed <n>` | Random seed (default: 42) |
| `--metal` | Use Metal GPU backend (macOS) |
| `--simd` | Use ARM NEON SIMD backend |
| `--gpu-resident` | Pre-load all weights into GPU buffers (default with `--metal`) |
| `--no-gpu-resident` | Use SSD-streaming mode instead of GPU-resident |
| `--context-len <n>` | Max context length for KV cache (auto-sized by default) |
| `--profile` | Print per-operation timing breakdown |

### `lumen convert`

```bash
lumen convert --input model.gguf --output model.lbc
lumen convert --input model.gguf --requant q4_0    # requantize to Q4_0
lumen convert --input model.gguf --dequantize       # all tensors to F32
```

## Architecture

Five crates in a Cargo workspace:

```
lumen-format      LBC file format: header, index, reader, streaming writer,
                  CRC32, quantization descriptors, test model generators

lumen-convert     GGUF-to-LBC converter with 10 dequantization kernels,
                  architecture-specific tensor mapping (8 architectures)

lumen-runtime     Inference engine core: Metal GPU backend (178 kernels),
                  ARM NEON SIMD backend, naive scalar backend, Accelerate
                  integration, KV cache, GDN state, MoE expert cache,
                  weight providers, pipeline scheduler, telemetry

lumen-bench       Benchmark harness: preset suites, configurable backends,
                  warm/cold start, JSON + table output

lumen-cli         Binary entry point: run, convert, bench, generate-test-model,
                  purge, help
```

## Numerical Stability

Three invariants enforced across all compute backends:

1. **Softmax**: subtract max before `exp` (prevents overflow)
2. **RMSNorm**: epsilon inside `sqrt`: `1.0 / (mean_sq + eps).sqrt()`
3. **Attention**: scale queries by `1.0 / sqrt(head_dim)`

## Building from Source

### Requirements

- Rust 1.75+ (2021 edition)
- macOS for Metal GPU and Accelerate backends (optional)
- aarch64 for NEON SIMD backend (optional; scalar backend works on all platforms)
- A GGUF model file, or use the built-in test model generator

### Build

```bash
cargo build --release
```

The binary is at `target/release/lumen`. Metal shaders are compiled at runtime from embedded MSL source -- no external shader files or build steps required.

### Run Tests

```bash
cargo test --workspace --release
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
