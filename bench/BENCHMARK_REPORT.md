# Lumen Benchmark Report v2

Generated: 2026-04-04

## Environment

### CUDA
- GPU: NVIDIA A100 80GB PCIe
- Runs: 1 (5 trials per config)
- Engines: Lumen (lumen 0.1.0), llama.cpp (d006858)

### Metal
- Hardware: Apple M3 Ultra (60 GPU cores, 96 GB, 819 GB/s)
- Runs: 3 (5 trials per config per run)
- Engines: Lumen (LBC), llama.cpp (GGUF), MLX (mlx-lm 0.30.7)

## Methodology

All benchmarks use **pp128+gen128** (128-token prompt, 128-token generation). The reported statistic is the **median** of multiple trials. CUDA results come from a single run with 5 internal trials per configuration. Metal results are the **median across 3 independent runs**, each containing 5 internal trials. Metal benchmarks include a thermal guard (30s model cooldown, 5s config cooldown) to minimize thermal throttling variance.

## CUDA Results

### Qwen2.5 3B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp |
|:-----:|------:|----------:|
| F16 | 152 | 174 |
| Q8_0 | 228 | 214 |
| Q4_0 | 213 | 259 |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp |
|:-----:|------:|----------:|
| F16 | 6,396 | 9,023 |
| Q8_0 | 7,743 | 5,031 |
| Q4_0 | 6,435 | 5,024 |

### Qwen2.5 7B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp |
|:-----:|------:|----------:|
| F16 | 94.7 | 95.2 |
| Q8_0 | 153 | 136 |
| Q4_0 | 194 | 189 |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp |
|:-----:|------:|----------:|
| F16 | 5,166 | 5,702 |
| Q8_0 | 5,426 | 3,673 |
| Q4_0 | 5,164 | 3,765 |

### Llama 3.1 8B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp |
|:-----:|------:|----------:|
| F16 | 88.6 | 93.0 |
| Q8_0 | 141 | 131 |
| Q4_0 | 180 | 175 |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp |
|:-----:|------:|----------:|
| F16 | 4,812 | 4,478 |
| Q8_0 | 5,486 | 3,308 |
| Q4_0 | 4,923 | 3,354 |

### Qwen2.5 14B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp |
|:-----:|------:|----------:|
| F16 | 50.6 | 49.6 |
| Q8_0 | 79.1 | 70.6 |
| Q4_0 | 99.5 | 99.2 |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp |
|:-----:|------:|----------:|
| F16 | 3,215 | 3,183 |
| Q8_0 | 3,415 | 1,989 |
| Q4_0 | 3,410 | 2,032 |

### Qwen3.5 9B

**Decode (tok/s):**

| Quant | Lumen | llama.cpp |
|:-----:|------:|----------:|
| Q8_0 | 62.6 | 116 |
| Q4_0 | 65.9 | 146 |

**Prefill (tok/s):**

| Quant | Lumen | llama.cpp |
|:-----:|------:|----------:|
| Q8_0 | 2,367 | 3,031 |
| Q4_0 | 2,196 | 3,089 |

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
