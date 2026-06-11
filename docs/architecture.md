# Architecture Overview

This page is the entry point for contributors. For deep dives, follow the links into source.

Lumen is an LLM inference engine written from scratch in Rust for Apple Silicon and NVIDIA CUDA. The runtime, KV cache, sampling, server protocols, OOM guard, and tokenizer pipeline are written against a generic transformer surface — new families plug in as new entries in the converter and registry without rewriting the kernel surface. The v1 architecture target is transformer + GDN-hybrid (dense or MoE FFN); v1 ships Qwen3.5-9B dense and Qwen3.5-MoE, and additional model families are planned.

## Crate layout

```text
lumen-format      LBC binary format, quantization descriptors, test model generators
lumen-convert     GGUF -> LBC converter (v1: qwen35 dense, qwen35moe MoE; additional
                  families planned)
lumen-runtime     CUDA backend (200+ NVRTC kernels across ~34 families), Metal backend (MSL shaders),
                  naive CPU + SIMD NEON references, KV cache (memory + disk),
                  GDN recurrent state, sampling, sessions, suffix prefill
lumen-server      axum HTTP server: OpenAI + Anthropic SSE endpoints, tool calling
lumen-bench       benchmark harness with JSON + table output
lumen-cli         CLI: built-in BPE tokenizer, model registry, HuggingFace downloader
```

## Forward pass

The forward-pass surface is transformer + GDN-hybrid (dense or MoE FFN). v1's shipped instances (Qwen3.5-9B dense, Qwen3.5-MoE) share an L=32 layer stack of hybrid GDN linear-attention layers (24 layers) interleaved with full-attention layers (8 layers). The same layer-stack contract applies to future model families that fall in this architecture class.

- **Dense FFN**: fused gate + up + SwiGLU + down kernel
- **MoE FFN**: routes the top-K experts per token through stacked gate + up + SwiGLU + down kernels
- **Long-context decode**: tiled streaming-softmax attention kernel that runs past the 40,950-token shared-memory ceiling (default for all sequence lengths; `ATTN_DECODE_TILED_DEFAULT_THRESHOLD = 0`)

## LBC binary format

Lumen's Layer-Blob Container (`.lbc`) format:

- 128 KiB-aligned blobs
- CRC32 header
- Zero-copy mmap
- Per-tensor quantization (mix BF16 / Q8_0 / Q4_0 in one file)
- Current version: `LBC_VERSION = 4`

The converter accepts GGUF v2/v3, streams one layer at a time, dequantizes K-quants (Q4_K, Q5_K, Q6_K, Q2_K, Q3_K) and MXFP4 to runtime-supported formats, and filters the MTP (Next-N) head.

## Suffix prefill

Each `Session` records its prompt history. On the next turn, `Session::extend_with_cache` reuses the longest shared prefix from the prior turn's KV cache and only recomputes the new suffix, so a cache-hit turn skips reprocessing the shared prefix and is substantially faster than a cold prefill. (Cache-reuse throughput is not part of the published benchmark suite.)

## Kernel surface

| Backend | Source root | Notable kernels |
|---|---|---|
| CUDA | `crates/lumen-runtime/src/cuda/` | `decode.rs` (tiled streaming-softmax decode), `prefill.rs` (FA2 prefill), `shaders/` (NVRTC kernels), `backend_impl.rs` (~16K LoC dispatch) |
| Metal | `crates/lumen-runtime/src/metal/` | `gdn.rs`, `moe.rs`, `prefill.rs`, `decode_*.rs`, `shaders/*.msl` |
| CPU | `crates/lumen-runtime/src/compute/cpu_naive.rs` + `crates/lumen-runtime/src/accelerate/` | Scalar reference + SIMD NEON |

## Per-backend status

Both backends (CUDA on NVIDIA compute-capability 8.0+, Metal on Apple Silicon M-series) are validated end-to-end against llama.cpp. The live verification matrix is in [`docs/support.md`](support.md); the benchmark rigs (an A100-80GB and an M3 Ultra) are recorded in [`bench/RESULTS.md`](../bench/RESULTS.md).

## Further reading

For benchmark methodology and the gate-driven optimization process, see [`bench/METHODOLOGY.md`](../bench/METHODOLOGY.md).
