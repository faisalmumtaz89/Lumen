# Lumen Benchmark Report v2

Generated: 2026-03-31

## Environment

### CUDA
- GPU: NVIDIA A100-SXM4-80GB
- Runs: 1 (5 trials per config)
- Engines: Lumen (lumen 0.1.0), llama.cpp (6729d49), vLLM 0.8.4 (HuggingFace FP16)

### Metal
- Hardware: Apple M3 Ultra (60 GPU cores, 96 GB, 819 GB/s)
- Runs: 3 (5 trials per config per run)
- Engines: Lumen (LBC), llama.cpp (GGUF), MLX (mlx-lm 0.30.7)

## Methodology

All benchmarks use **pp128+gen128** (128-token prompt, 128-token generation). The reported statistic is the **median** of multiple trials. CUDA results come from a single run with 5 internal trials per configuration. Metal results are the **median across 3 independent runs**, each containing 5 internal trials. Metal benchmarks include a thermal guard (30s model cooldown, 5s config cooldown) to minimize thermal throttling variance.

## CUDA Results

### Qwen2.5 3B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 153 | 169 | 61.3 |
| Q8_0 | 225 | 205 | N/A |
| Q4_0 | 217 | 246 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 6,454 | 8,168 | 7,588 |
| Q8_0 | 6,471 | 4,037 | N/A |
| Q4_0 | 6,486 | 3,992 | N/A |

### Qwen2.5 7B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 94.7 | 95.5 | 72.9 |
| Q8_0 | 153 | 135 | N/A |
| Q4_0 | 196 | 184 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 5,204 | 4,632 | 8,536 |
| Q8_0 | 5,948 | 3,245 | N/A |
| Q4_0 | 5,202 | 2,975 | N/A |

### Llama 3.1 8B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 88.8 | 91.1 | 67.2 |
| Q8_0 | 142 | 131 | N/A |
| Q4_0 | 178 | 169 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 4,847 | 4,275 | 7,607 |
| Q8_0 | 5,556 | 3,023 | N/A |
| Q4_0 | 4,812 | 2,947 | N/A |

### Qwen2.5 14B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 50.6 | 49.8 | 44.1 |
| Q8_0 | 79.1 | 70.7 | N/A |
| Q4_0 | 100 | 97.9 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| F16 | 2,989 | 2,875 | 5,447 |
| Q8_0 | 3,432 | 1,891 | N/A |
| Q4_0 | 3,413 | 1,948 | N/A |

### Qwen3.5 9B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| Q8_0 | 58.9 | 114 | N/A |
| Q4_0 | 69.9 | 140 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | vLLM |
|:-----:|------:|----------:|-----:|
| Q8_0 | 324 | 2,611 | N/A |
| Q4_0 | 316 | 2,806 | N/A |

## vLLM GGUF (Experimental)

vLLM 0.8.4 supports GGUF Q8_0/Q4_0 via its V0 engine, but this path is [experimental and under-optimized](https://docs.vllm.ai/en/v0.8.4/features/quantization/gguf.html). GGUF decode throughput is no faster than FP16 (no quantization benefit), and GGUF prefill is 3-15x slower than FP16. These numbers are included for completeness but do not represent vLLM's production performance. Hardware: A100 PCIe (separate run from the SXM4 data above).

| Model | Quant | vLLM Decode | vLLM Prefill | vs vLLM F16 Dec | vs vLLM F16 PP |
|-------|:-----:|------------:|-------------:|----------------:|---------------:|
| Qwen2.5 3B | F16 | 54.9 | 6,630 | — | — |
| Qwen2.5 3B | Q8_0 | 52.8 | 1,599 | 0.96x | 0.24x |
| Qwen2.5 3B | Q4_0 | 51.8 | 2,344 | 0.94x | 0.35x |
| Qwen2.5 7B | F16 | 66.6 | 7,748 | — | — |
| Qwen2.5 7B | Q8_0 | 63.9 | 665 | 0.96x | 0.09x |
| Qwen2.5 7B | Q4_0 | 59.4 | 979 | 0.89x | 0.13x |
| Llama 3.1 8B | F16 | 54.4 | 6,306 | — | — |
| Llama 3.1 8B | Q8_0 | 60.2 | 596 | 1.11x | 0.09x |
| Llama 3.1 8B | Q4_0 | 59.5 | 918 | 1.09x | 0.15x |
| Qwen2.5 14B | F16 | 40.2 | 5,052 | — | — |
| Qwen2.5 14B | Q8_0 | 39.9 | 329 | 0.99x | 0.07x |
| Qwen2.5 14B | Q4_0 | 40.0 | 496 | 0.99x | 0.10x |

Qwen3.5 9B: GGUF architecture `qwen35` not supported by vLLM 0.8.4.

## Metal Results

### TinyLlama 1.1B

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

### Llama 3.1 8B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | MLX |
|:-----:|------:|----------:|----:|
| F16 | 40.2 | 42.3 | N/A |
| Q8_0 | 73.3 | 67.4 | 79.2 |
| Q4_0 | 97.6 | 95.4 | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | MLX |
|:-----:|------:|----------:|----:|
| F16 | 766 | 1,050 | N/A |
| Q8_0 | 838 | 1,003 | 906 |
| Q4_0 | 796 | 1,028 | N/A |

### Qwen3.5 9B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp | MLX |
|:-----:|------:|----------:|----:|
| Q8_0 | 57.0 | N/A | 91.8 |
| Q4_0 | 67.7 | N/A | N/A |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp | MLX |
|:-----:|------:|----------:|----:|
| Q8_0 | 343 | N/A | 747 |
| Q4_0 | 332 | N/A | N/A |

## Metal Reproducibility

Per-run values (R1/R2/R3) and coefficient of variation (CV%) across the 3 Metal runs.

| Model | Quant | Engine | Metric | R1 | R2 | R3 | Median | CV% |
|-------|:-----:|--------|:------:|---:|---:|---:|-------:|----:|
| TinyLlama 1.1B | F16 | llama.cpp | decode | 185 | 183 | 184 | 184 | 0.4 |
| TinyLlama 1.1B | F16 | llama.cpp | prefill | 5,249 | 5,254 | 5,242 | 5,249 | 0.1 |
| TinyLlama 1.1B | F16 | Lumen | decode | 194 | 194 | 194 | 194 | 0.1 |
| TinyLlama 1.1B | F16 | Lumen | prefill | 4,028 | 4,088 | 3,759 | 4,028 | 4.4 |
| TinyLlama 1.1B | F16 | MLX | decode | 238 | 239 | 240 | 239 | 0.3 |
| TinyLlama 1.1B | F16 | MLX | prefill | 4,186 | 4,189 | 4,197 | 4,189 | 0.1 |
| TinyLlama 1.1B | Q8_0 | llama.cpp | decode | 226 | 225 | 225 | 225 | 0.2 |
| TinyLlama 1.1B | Q8_0 | llama.cpp | prefill | 5,061 | 5,053 | 5,042 | 5,053 | 0.2 |
| TinyLlama 1.1B | Q8_0 | Lumen | decode | 302 | 308 | 305 | 305 | 1.0 |
| TinyLlama 1.1B | Q8_0 | Lumen | prefill | 4,926 | 4,778 | 4,911 | 4,911 | 1.7 |
| TinyLlama 1.1B | Q8_0 | MLX | decode | 449 | 450 | 449 | 449 | 0.1 |
| TinyLlama 1.1B | Q8_0 | MLX | prefill | 4,838 | 4,841 | 4,824 | 4,838 | 0.2 |
| TinyLlama 1.1B | Q4_0 | llama.cpp | decode | 246 | 265 | 246 | 246 | 4.4 |
| TinyLlama 1.1B | Q4_0 | llama.cpp | prefill | 5,167 | 5,201 | 5,213 | 5,201 | 0.5 |
| TinyLlama 1.1B | Q4_0 | Lumen | decode | 318 | 322 | 319 | 319 | 0.7 |
| TinyLlama 1.1B | Q4_0 | Lumen | prefill | 4,460 | 4,549 | 4,525 | 4,525 | 1.0 |
| Llama 3.1 8B | F16 | llama.cpp | decode | 42.2 | 42.3 | 42.5 | 42.3 | 0.4 |
| Llama 3.1 8B | F16 | llama.cpp | prefill | 1,050 | 1,052 | 1,050 | 1,050 | 0.1 |
| Llama 3.1 8B | F16 | Lumen | decode | 40.3 | 40.2 | 40.2 | 40.2 | 0.1 |
| Llama 3.1 8B | F16 | Lumen | prefill | 772 | 570 | 766 | 766 | 16.3 |
| Llama 3.1 8B | Q8_0 | llama.cpp | decode | 67.4 | 67.9 | 67.2 | 67.4 | 0.5 |
| Llama 3.1 8B | Q8_0 | llama.cpp | prefill | 1,004 | 1,003 | 1,002 | 1,003 | 0.1 |
| Llama 3.1 8B | Q8_0 | Lumen | decode | 73.3 | 73.7 | 73.3 | 73.3 | 0.3 |
| Llama 3.1 8B | Q8_0 | Lumen | prefill | 838 | 838 | 836 | 838 | 0.1 |
| Llama 3.1 8B | Q8_0 | MLX | decode | 79.2 | 79.2 | 79.2 | 79.2 | 0.0 |
| Llama 3.1 8B | Q8_0 | MLX | prefill | 905 | 906 | 907 | 906 | 0.2 |
| Llama 3.1 8B | Q4_0 | llama.cpp | decode | 95.2 | 96.6 | 95.4 | 95.4 | 0.8 |
| Llama 3.1 8B | Q4_0 | llama.cpp | prefill | 1,026 | 1,028 | 1,028 | 1,028 | 0.1 |
| Llama 3.1 8B | Q4_0 | Lumen | decode | 97.7 | 97.6 | 97.3 | 97.6 | 0.2 |
| Llama 3.1 8B | Q4_0 | Lumen | prefill | 796 | 798 | 796 | 796 | 0.1 |
| Qwen3.5 9B | Q8_0 | Lumen | decode | 57.1 | 57.0 | 57.0 | 57.0 | 0.1 |
| Qwen3.5 9B | Q8_0 | Lumen | prefill | 343 | 343 | 343 | 343 | 0.1 |
| Qwen3.5 9B | Q8_0 | MLX | decode | 91.8 | 91.6 | 92.0 | 91.8 | 0.2 |
| Qwen3.5 9B | Q8_0 | MLX | prefill | 738 | 747 | 750 | 747 | 0.8 |
| Qwen3.5 9B | Q4_0 | Lumen | decode | 67.7 | 67.8 | 67.5 | 67.7 | 0.2 |
| Qwen3.5 9B | Q4_0 | Lumen | prefill | 334 | 332 | 332 | 332 | 0.3 |

> **High-variance flag:** Llama 3.1 8B F16 Lumen prefill has CV=16.3% (R2=570 is an outlier vs R1=772, R3=766). The median (766) is robust but this measurement has elevated run-to-run variance.
