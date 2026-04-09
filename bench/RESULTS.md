# Benchmark Results

Methodology: [METHODOLOGY.md](METHODOLOGY.md). How to reproduce: [README.md](README.md).

---

## Metal — Apple M3 Ultra

| Field | Value |
|-------|-------|
| Date | 2026-04-09 |
| Lumen | commit `57234e5` |
| Hardware | Apple M3 Ultra, 60 GPU cores, 96 GB unified, 819 GB/s |
| macOS | 26.3.1 |
| Rust | 1.93.1 |
| llama.cpp | build 8680 (Homebrew, Metal backend) |
| MLX | mlx-lm 0.30.7, mlx 0.31.0 |
| Config | pp128 + gen128, 3 runs, 1 warmup, median reported |

### Decode (tok/s)

| Model | Quant | Lumen | llama.cpp | MLX | vs LC | vs MLX |
|:------|:-----:|------:|----------:|----:|:-----:|:------:|
| TinyLlama 1.1B | F16 | 180 | 171 | 230 | **1.05x** | 0.78x |
| TinyLlama 1.1B | Q8_0 | 307 | 221 | 422 | **1.39x** | 0.73x |
| TinyLlama 1.1B | Q4_0 | 325 | 253 | — | **1.28x** | — |
| Llama 3.1 8B | F16 | 39.6 | 40.2 | — | 0.99x | — |
| Llama 3.1 8B | Q8_0 | 72.7 | 64.2 | 75.8 | **1.13x** | 0.96x |
| Llama 3.1 8B | Q4_0 | 97.0 | 92.2 | — | **1.05x** | — |
| Qwen3.5 9B | Q8_0 | 53.0 | — | 87.0 | — | 0.61x |
| Qwen3.5 9B | Q4_0 | 78.5 | — | — | — | — |

### Prefill (tok/s)

| Model | Quant | Lumen | llama.cpp | MLX | vs LC | vs MLX |
|:------|:-----:|------:|----------:|----:|:-----:|:------:|
| TinyLlama 1.1B | F16 | 3,945 | 5,061 | 3,959 | 0.78x | 1.00x |
| TinyLlama 1.1B | Q8_0 | 4,549 | 4,636 | 4,496 | 0.98x | **1.01x** |
| TinyLlama 1.1B | Q4_0 | 4,448 | 4,821 | — | 0.92x | — |
| Llama 3.1 8B | F16 | 527 | 995 | — | 0.53x | — |
| Llama 3.1 8B | Q8_0 | 826 | 964 | 876 | 0.86x | 0.94x |
| Llama 3.1 8B | Q4_0 | 781 | 988 | — | 0.79x | — |
| Qwen3.5 9B | Q8_0 | 322 | — | 730 | — | 0.44x |
| Qwen3.5 9B | Q4_0 | 719 | — | — | — | — |

### Lumen Only (no baseline available)

| Model | Quant | Decode | Prefill |
|:------|:-----:|-------:|--------:|
| Qwen2.5 3B | F16 | 79.1 | 1,820 |
| Qwen2.5 3B | Q8_0 | 126 | 1,856 |
| Qwen2.5 3B | Q4_0 | 124 | 1,710 |
| Qwen2.5 7B | Q8_0 | 71.1 | 841 |
| Qwen2.5 7B | Q4_0 | 88.1 | 815 |
| Qwen2.5 14B | Q8_0 | 40.5 | 368 |
| Qwen2.5 14B | Q4_0 | 57.2 | 421 |
| Mistral 7B | Q8_0 | 77.3 | 833 |

---

## CUDA — NVIDIA A100-80GB

| Field | Value |
|-------|-------|
| Date | 2026-04-09 |
| Lumen | commit `57234e5` |
| Hardware | NVIDIA A100-80GB (Modal; SXM4 or PCIe assigned per run) |
| CUDA | 12.6.3 |
| Rust | 1.93.1 |
| llama.cpp | commit `0ec191e` (co-located, same GPU as Lumen) |
| vLLM | 0.8.4 (FP16 only, separate container) |
| Config | pp128 + gen128, 5 trials, 1 warmup, median reported |

### Decode (tok/s)

| Model | Quant | Lumen | llama.cpp | vLLM | vs LC | vs vLLM |
|:------|:-----:|------:|----------:|-----:|:-----:|:-------:|
| Qwen2.5 3B | F16 | 150 | 175 | 51.3 | 0.85x | **2.91x** |
| Qwen2.5 3B | Q8_0 | 223 | 213 | — | **1.05x** | — |
| Qwen2.5 3B | Q4_0 | 199 | 260 | — | 0.77x | — |
| Qwen2.5 7B | F16 | 93.7 | 97.0 | 58.6 | 0.97x | **1.60x** |
| Qwen2.5 7B | Q8_0 | 150 | 137 | — | **1.09x** | — |
| Qwen2.5 7B | Q4_0 | 183 | 189 | — | 0.97x | — |
| Llama 3.1 8B | F16 | 88.6 | 93.2 | 52.3 | 0.95x | **1.69x** |
| Llama 3.1 8B | Q8_0 | 142 | 135 | — | **1.05x** | — |
| Llama 3.1 8B | Q4_0 | 174 | 175 | — | 0.99x | — |
| Qwen2.5 14B | F16 | 49.8 | 50.6 | 35.7 | 0.98x | **1.40x** |
| Qwen2.5 14B | Q8_0 | 77.8 | 73.0 | — | **1.07x** | — |
| Qwen2.5 14B | Q4_0 | 108 | 101 | — | **1.07x** | — |
| Qwen3.5 9B | Q8_0 | 68.1 | 118 | — | 0.58x | — |
| Qwen3.5 9B | Q4_0 | 76.2 | 146 | — | 0.52x | — |

### Prefill (tok/s)

Batched prefill is only enabled for Qwen3.5 (GDN architecture). Standard models use per-token prefill on CUDA.

| Model | Quant | Lumen | llama.cpp | vs LC |
|:------|:-----:|------:|----------:|:-----:|
| Qwen3.5 9B | Q8_0 | 2,071 | 2,875 | 0.72x |
| Qwen3.5 9B | Q4_0 | 2,132 | 2,732 | 0.78x |
