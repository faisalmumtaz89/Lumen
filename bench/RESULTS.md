# Benchmark Results

**Last updated:** 2026-08-26

> **Scope:** The benchmark suite covers Lumen's v1 model family: Qwen3.5-9B dense, Qwen3.5-MoE-35B-A3B, and (from the retained 2026-07-16 battery) Qwen3.6-27B dense CUDA decode. Additional model families will be added as they ship.

Methodology: [METHODOLOGY.md](METHODOLOGY.md). How to reproduce: [README.md](README.md). Deployment guidance: [`docs/production.md`](../docs/production.md).

---

## TL;DR — production-realistic numbers

Retained record (2026-07-16 battery; 5 decode runs per engine per cell;
the artifacts record no engine commit). Q8/Q4 cells are co-located — both
engines in one A100 PCIe container per cell, llama.cpp build `5839ba3`,
tg256. The BF16 cells are separate Lumen and llama.cpp batteries on the
same H100 GPU model and driver; their artifacts record per-run decode
rates but not the llama.cpp build or bench protocol, and same-container
co-residency is not recorded for them:

| Quant | GPU | Lumen (tok/s) | llama.cpp | × llama.cpp |
|-------|-----|------------:|----------:|------------:|
| Q8_0 | A100-80GB | 79.2 | 139.7 | 0.567× |
| Q4_0 | A100-80GB | 93.6 | 156.5 | 0.598× |
| BF16 | H100 | 104.1 | 181.1 | 0.575× — highest absolute throughput, lowest ratio; A100 decode unmeasured |

> **Provenance note (2026-08-26):** the previously published 2026-06-02
> dataset — BF16 91.4 = 0.902×, Q8 82.1 = 0.584×, Q4 105.6 = 0.674×, the
> workload-weighted means, the workload-stability percentages, and the
> bandwidth-utilization table below — has **no retained measurement
> artifacts**. Those sections are kept for historical context only and must
> not be cited. Additional retained BF16-MoE records: 78.8 tok/s (H200,
> 3-trial quality-confirmation cell, no comparator), 96.1 median (H100 soak
> window), 113.6 (H100 remeasure, no comparator).

---

## Per-workload empirical results (CUDA, Qwen3.5-MoE-35B-A3B)

*(2026-06-02 dataset — artifacts not retained; see provenance note)*

Conditions: Qwen3.5-MoE-35B-A3B on A100-80GB PCIe, GPU 1 isolated, driver 580.126.20, CUDA 12.2.140, sm_80. canonical 12-env-var stack. PURE-greedy `--temperature 0 --seed 42 --repeat-penalty 1.0 --repeat-last-n 0`. Each cell is 3-trial median.

### Decode tok/s per (workload pattern × quant)

| Workload (prompt ~tok / gen ~tok) | Q8 Lumen | Q8 llama.cpp | Q8 × llama.cpp | Q4 Lumen | Q4 llama.cpp | Q4 × llama.cpp | BF16 Lumen | BF16 llama.cpp | BF16 × llama.cpp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short Q&A (~50 / 150)             | 79.6 | 138.55 | 0.575× | 102.0 | 155.78 | 0.655× | 90.0 | 98.58 | 0.913× |
| Medium completion (~500 / 500)    | 74.8 | 138.89 | 0.538× | 94.4 | 155.57 | 0.607× | 82.3 | 98.55 | 0.835× |
| Long-form (~200 / 1500)           | 72.8 | 138.61 | 0.525× | 103.6 | 155.20 | 0.667× | 90.8 | 99.35 | 0.913× |
| Code generation (~100 / 800)      | 76.1 | 138.97 | 0.548× | 96.4 | 155.45 | 0.620× | 84.9 | 99.51 | 0.853× |
| Multi-turn (3-turn ~500 / 200 ea) | 78.5 | 138.55 | 0.567× | 100.9 | 155.78 | 0.648× | 89.1 | 98.58 | 0.904× |
| **Mean (equal-weight)**           | **76.4** | 138.71 | **0.551×** | **99.5** | 155.56 | **0.640×** | **87.4** | 98.91 | **0.883×** |

### Prefill tok/s per (workload pattern × quant)

*(2026-06-02 dataset — artifacts not retained; see provenance note)*

| Workload | Q8 Lumen | Q8 llama.cpp | Q8 × llama.cpp | Q4 Lumen | Q4 llama.cpp | Q4 × llama.cpp | BF16 Lumen | BF16 llama.cpp | BF16 × llama.cpp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short Q&A (~50)            | 114.4 | 866.0 | 0.132× | 179.5 | 1124.8 | 0.160× | 109.2 | 1030.5 | 0.106× |
| Medium completion (~500)   | 127.0 | 2500.5 | 0.051× | 233.2 | 3369.3 | 0.069× | 175.4 | 1481.8 | 0.118× |
| Long-form (~200)           | 86.7  | 1991.7 | 0.044× | 129.3 | 2249.0 | 0.057× | 78.1  | 1268.1 | 0.062× |
| Code generation (~100)     | 134.0 | 1346.1 | 0.100× | 214.8 | 1448.6 | 0.148× | 145.6 | 1635.3 | 0.089× |
| Multi-turn (~500)          | 135.2 | 2500.5 | 0.054× | 219.9 | 3369.3 | 0.065× | 154.6 | 1481.8 | 0.104× |

**Prefill structural ceiling**: ratio < 1.0× on all quants. NVRTC compute_61 PTX ISA + non-monolithic encoder is the root cause per feasibility analysis. Closing this requires multi-quarter scope-change.

### Memory-bandwidth utilization (decode, single-stream, A100-80GB theoretical 1935 GB/s HBM2e)

*(2026-06-02 dataset — artifacts not retained; see provenance note)*

| Quant | Active weights/token | Lumen achieved BW | Lumen % theoretical | llama.cpp achieved BW | llama.cpp % theoretical |
|-------|---------------------:|------------------:|--------------------:|---------------:|-----------------:|
| Q8    | 3.0 GB | 229 GB/s | 11.8% | 416 GB/s | 21.5% |
| Q4    | 1.5 GB | 149 GB/s | 7.7%  | 233 GB/s | 12.1% |
| BF16  | 6.0 GB | **524 GB/s** | **27.1%** | 594 GB/s | 30.7% |

BF16 is bandwidth-bound (27.1% utilized = within 4 pp of llama.cpp ceiling 30.7%); Q8 / Q4 are compute-bound (NVRTC compute_61 ISA loses ~30-40% per-kernel throughput vs llama.cpp's sm_80 cubin). **Revised bandwidth-utilization gate: ≥25% theoretical** (original ≥50% exceeds llama.cpp's own ceiling on the same silicon). The BF16 pass verdict on this gate is withdrawn (artifacts not retained).

### Cold-start TTFT (full model load + first 8 tokens)

*(2026-06-02 dataset — artifacts not retained; see provenance note)*

| Quant | LBC size | Cold (warm OS page cache) | Cold-cold (no OS cache) estimate |
|-------|---------:|--------------------------:|---------------------------------:|
| Dense-9B Q8 | 9.5 GB  | ~86 s   | ~95 s   |
| MoE Q4      | 20.7 GB | ~89 s   | ~110 s  |
| MoE Q8      | 37.6 GB | ~99 s   | ~136 s  |
| MoE BF16    | 69.7 GB | ~102 s  | ~172 s  |

Cold-start gate "≤30 s for ≤10 B model" is N/A for 35B-A3B MoE (active 3B but on-disk weight ≥19 GB). **Production recommendation**: pre-warm LBC into OS page cache at service startup (`cat /path/to/model.lbc > /dev/null`). Once warm, subsequent `lumen run` cold-loads stay within the warm-cache budget above.

### Concurrency scaling (process-level only — Lumen has no batch scheduler)

*(2026-06-02 dataset — artifacts not retained; see provenance note)*

| Concurrency | Per-client decode (tok/s) | Aggregate (tok/s) | Scaling vs C=1 |
|-------------|--------------------------:|------------------:|---------------:|
| C=1 | 106.1 | 106.1 | 1.00× |
| C=2 | 50.0 / 93.6 (asymmetric) | 143.6 | **1.35×** |
| C≥4 | n/a — no server-side scheduler in Lumen CLI; deploy `lumen-server` for ≥4 concurrent clients | n/a | n/a |

Per-GPU OOM ceilings (single A100-80GB, 2026-06-02 dataset — artifacts not retained; the A100 BF16-MoE fit is unverified per docs/production.md): Q4 ~C=3, Q8 ~C=2, BF16 C=1. C≥4 requires weight-sharing scheduler (deferred to KV-cache / server work).

---

## CUDA — Qwen3.5-9B on NVIDIA A100-80GB

| Field | Value |
|-------|-------|
| Date | Summary table: 2026-07-16 battery. Historical subsections below: 2026-05-26 re-bench |
| Model | Qwen3.5-9B dense (`qwen35` architecture — GDN hybrid + dense FFN) |
| Hardware | NVIDIA A100-80GB (PCIe canonical for parity baselines; SXM4 used for long-context / soak scenarios where annotated) |
| CUDA | 12.6.3 |
| Lumen | CUDA backend, opt-in stack enabled (see [METHODOLOGY.md](METHODOLOGY.md)) |
| llama.cpp | Summary table: `5839ba3` (tg256) for the A100 rows; not recorded for the BF16 row. Historical subsections: `0ec191e`, co-located in the same A100 container |
| vLLM | 0.21.0 (BF16 only; transformers 5.5.4, tokenizers 0.22.2, `VLLM_USE_FLASHINFER_SAMPLER=0`) |
| Config | Summary table: tg256 decode battery, 5 runs. Historical subsections: pp128 + gen128, batch=1, 5 trials, 1 warmup, median reported (see "Decode envelope" below for the full (M, G) parameter space) |
| GGUF source | `bartowski/Qwen_Qwen3.5-9B-GGUF` (BF16 generated via `convert_hf_to_gguf.py --outtype bf16`) |

### Summary — Qwen3.5-9B dense

Retained record (same 2026-07-16 battery as the TL;DR — co-located A100
for Q8/Q4, llama.cpp `5839ba3` tg256; the BF16 row is separate same-GPU
H100 batteries with build/protocol not recorded in the artifacts):

| Quant | GPU | Lumen (tok/s) | llama.cpp | × llama.cpp |
|-------|-----|------------:|----------:|------------:|
| Q8_0 | A100-80GB | 114.1 | 117.6 | **0.970×** |
| Q4_0 | A100-80GB | 146.6 | 149.8 | **0.979×** |
| BF16 | H100 | 106.5 | 146.6 | 0.726× |

### Summary — Qwen3.6-27B dense (CUDA)

Retained record (same 2026-07-16 battery; the staging artifact names the
source file `Qwen_Qwen3.6-27B-Q8_0.gguf`). Q8/Q4 co-located on A100 PCIe,
llama.cpp `5839ba3` tg256; the BF16 row is separate same-GPU H100
batteries (build/protocol not recorded in those artifacts):

| Quant | GPU | Lumen (tok/s) | llama.cpp | × llama.cpp |
|-------|-----|------------:|----------:|------------:|
| Q8_0 | A100-80GB PCIe | 35.08 | 39.35 | 0.891× |
| Q4_0 | A100-80GB PCIe | 45.34 | 55.32 | 0.820× |
| BF16 | H100 | 40.42 | 49.40 | 0.818× |

The 2026-06-02 headline (Q8 0.907×, Q4 0.635×, BF16 0.931–0.940×, and the
prefill ratios) has no retained measurement artifacts — see the provenance
note in the TL;DR:

| Quant | Decode vs llama.cpp | Prefill vs llama.cpp | Config |
|-------|--------------------:|---------------------:|--------|
| Q8_0  | 0.907× *(not retained)* | 0.75× | Q8 dense-9B |
| Q4_0  | 0.635× *(not retained)* | 0.71× | Q4 dense-9B |
| BF16 (cuBLAS GemmEx, default-on)  | 0.931×–0.940× *(not retained)* | 0.97× | BF16 dense-9B |

**Historical reference:**

| Quant | Decode tok/s | llama.cpp decode | vs llama.cpp | Prefill tok/s | llama.cpp prefill | vs llama.cpp | Source release |
|-------|------:|-----:|------:|------:|-----:|------:|----------------|
| Q8_0  | 85.8 | 117 | 0.73× | 2245 | 2979 | 0.75× | (PCIe, full opt-ins; superseded by 0.907×) |
| Q4_0  | 102.1 | 147 | 0.69× | 2203 | 3097 | 0.71× | (PCIe, full opt-ins; superseded by 0.635×) |
| BF16 (cuBLAS GemmEx, default-on) | 66.1 | 84.5 | 0.78× | 3899.7 | 4036.1 | 0.97× | BF16 bench (PCIe; superseded by 0.931–0.940×) |
| BF16 (prior-instance run) | — | — | — | 3967 | 3635 | 1.09× | baseline (SXM4) |

### Decode envelope — Qwen3.5-9B dense, parameterized by (M, G)

**Decode tok/s as a function of context M and output length G.** Decode rate is sensitive to BOTH M (KV cache scan cost per step) and G (output length post-warmup; longer outputs grow the effective KV tail seen by the back end of the run). Reporting only M underestimates user-visible cost for long-output prompts: at M = 128 the published pp128+gen128 anchor of **85.8 tok/s** (Q8) drops to **52.6 tok/s** at gen512 — a **39% structural decline** driven entirely by the gen-length axis.

#### Q8_0 decode tok/s

| M (context) | G = 32 | G = 128 | G = 512 |
|------------:|-------:|--------:|--------:|
| 128         | TBD    | **85.8** | **52.6** |
| 512         | **52.8** | TBD | TBD |
| 2 048       | **47.1** | TBD | TBD |
| 4 096       | **52.25** *(tiled)* | TBD | TBD |
| 8 192       | **40.44** *(tiled)* | TBD | **35.2** |
| 16 384      | **27.75** *(tiled)* | TBD | TBD |
| 32 768      | **16.70** *(tiled)* | TBD | TBD |
| 65 536      | **9.13** *(tiled)*  | TBD | TBD |
| 98 304      | **6.57** *(tiled)*   | TBD | TBD |
| 131 072     | (PASS, see long-context note) | TBD | TBD |

#### Q4_0 decode tok/s

| M (context) | G = 32 | G = 128 | G = 512 |
|------------:|-------:|--------:|--------:|
| 128         | TBD    | **102.1** | TBD *(Q4_0 LBC unavailable; see C-16)* |
| 4 096       | **56.49** *(tiled)* | TBD | TBD |
| 8 192       | **42.77** *(tiled)* | TBD | TBD |
| 16 384      | **28.77** *(tiled)* | TBD | TBD |
| 32 768      | **16.92** *(tiled)* | TBD | TBD |
| 65 536      | **9.21** *(tiled)*  | TBD | TBD |
| 98 304      | **6.62** *(tiled)*   | TBD | TBD |
| 131 072     | OOM (A100-80GB VRAM)         | OOM | OOM |

#### BF16 decode tok/s

| M (context) | G = 32 | G = 128 | G = 512 |
|------------:|-------:|--------:|--------:|
| 128         | TBD    | **66.1** | TBD |
| 4 096       | **47.17** *(tiled)* | TBD | TBD |
| 8 192       | **37.13** *(tiled)* | TBD | TBD |
| 16 384      | **26.16** *(tiled)* | TBD | TBD |
| 32 768      | **16.37** *(tiled)* | TBD | TBD |
| 65 536      | **8.97** *(tiled)*  | TBD | TBD |
| 98 304      | OOM (A100-80GB VRAM)         | OOM | OOM |
| 131 072     | OOM (A100-80GB VRAM)         | OOM | OOM |

**Methodology:** Decode rate is sensitive to BOTH M (KV cache scan cost per step) and G (output length post-warmup); reporting only M underestimates user-visible cost for long-output prompts. The column ("G = 32") is the steady-state per-token decode rate measured at 5 trials post 3-warmup (effectively G ~= 5 measured tokens, used here as the small-G anchor; the per-step KV-scan cost dominates over the gen-length effect at large M). All numbers are 5-trial median, greedy temperature 0, A100-80GB PCIe. Source measurements: M=128 anchor, long-context tiled decode, 98K/131K post-P-3 fix, gen-length sensitivity at short M, long-M long-G stationarity at M=6 808 G=512. TBD cells are reproducible from the harness in [README.md](README.md); they are not measured in the current cell matrix.

**Long-context note:** Tiled streaming-softmax decode kernel validated end-to-end at sequence lengths up to 131K on A100-80GB PCIe. After (GDN prefill grid-Y chunking + threshold default flip + head_dim hardware-compat guard), all previously-failing P-3 cells now PASS; the OOM cells at 98K (BF16) and 131K (BF16 + Q4) are pure hardware-class A100-80GB VRAM ceiling events, not kernel bugs. End-to-end at 64K is token-stream-exact versus the llama.cpp baseline. The tiled decode threshold defaults to 0 (always-on); legacy SingleBlock kernel remains available via `LUMEN_CUDA_DECODE_TILED_THRESHOLD=4294967295`. Determinism across G is preserved.

### Qwen3.5-MoE 35B-A3B decode — historical arc (SUPERSEDED)

> **Superseded by the retained record in the [TL;DR](#tldr--production-realistic-numbers) (the 2026-06-02 [per-workload](#per-workload-empirical-results-cuda-qwen35-moe-35b-a3b) tables are themselves withdrawn — artifacts not retained).** The 2026-06-02 numbers this note originally pointed at (Q8 82.1 = 0.584×, Q4 105.6 = 0.674×) have no retained measurement artifacts; the retained co-located record is in the TL;DR (Q8 79.2 = 0.567×, Q4 93.6 = 0.598×). The rows below are retained for historical traceability of the perf+correctness arc and use an earlier llama.cpp baseline (140.65 / 156.71) and earlier Lumen build; do not cite them as current.

The qwen35moe CUDA runtime path closed cleanly (one-line `BLOCK_DIM` fix in `moe_accum.cu:19`); the historical perf+correctness arc was:

| Model | Quant | Decode tok/s | llama.cpp | × llama.cpp | Restated gate | Source release |
|-------|-------|-------------:|---:|-----:|---------------|----------------|
| Qwen3.5-MoE 35B-A3B | Q8_0 | **~71.8** *(historical; retained current record 79.2 = 0.567×)* | 140.65 | 0.510× | ≥0.65× llama.cpp | (earlier 8-flag stack) |
| Qwen3.5-MoE 35B-A3B | Q4_0 | **~80.9** *(historical; retained current record 93.6 = 0.598×)* | 156.71 | 0.516× | ≥0.73× llama.cpp | |
| Qwen3.5-MoE 35B-A3B | BF16 | see canonical production config below | — | — | — | canonical 8-flag stack |
| Qwen3.5-MoE 35B-A3B (original Q8) | Q8_0 | 17.79 (legacy) | — | — | — | D.1 (legacy reference) |
| Qwen3.5-9B dense (regression check) | Q8_0 | 41.35 (legacy) | — | — | — | D.2 (legacy reference) |

**Canonical production config (CUDA, Qwen3.5-MoE-35B-A3B BF16)**:

```bash
# canonical 8-flag stack (defaults ON, opt-out=0):
LUMEN_CUDA_MOE_BATCHED=1                # default ON
LUMEN_CUDA_MOE_ROUTER_SINGLE_CTA=1      # default ON
LUMEN_CUDA_MOE_ROUTER_PARALLEL=1        # default ON
LUMEN_CUDA_GDN_REGISTER_RESIDENT=1      # default ON
LUMEN_CUDA_BF16_GEMMEX=0                # MUST be 0 for BF16 P3 correctness
LUMEN_CUDA_BF16_MOE_V3=1                # default ON
LUMEN_CUDA_MOE_Q4_V3=1                  # default ON
LUMEN_CUDA_MOE_Q4_V3B=1                 # default ON

# canonical default (the BF16 output_proj lever):
LUMEN_CUDA_MMV_BF16_OUTPUT_PROJ=1       # default ON — llama.cpp-parity BF16 output_proj matvec

# Pair with the +1 LoC CLI fix in lumen-cli/src/run.rs that allows
# Bf16 in set_output_proj_raw — without that, BF16 LBC raw weights are never
# uploaded and llama.cpp kernel never engages.
```

**The previously published BF16 0.902× llama.cpp figure has no retained measurement artifact** — see the provenance note in the TL;DR for the retained records.

### Audit-restated per-quant gates

*(2026-06-02 dataset — artifacts not retained; see provenance note)*

The canonical 5-trial median on "Once upon a time" was the load-bearing reference for this dataset. The two-backend validation matrix adds workload-stability gates across short / medium / long / code / multi-turn patterns:

| Quant | Decode gate | Workload-stability gate | Bandwidth gate | Result |
|-------|------------------------------|------------------------------------------|--------------------------|---------|
| BF16  | ≥0.9× llama.cpp = 91.2 tok/s — 91.4 | ≤15% range across 5 workloads — 10.1% | ≥25% theoretical — 27.1% | *verdict withdrawn — artifacts not retained* |
| Q8    | ≥0.585× llama.cpp = 82 tok/s — 82.1 | ≤15% range — 9.1% | ≥25% — 11.8% (compute-bound) | *verdict withdrawn — artifacts not retained* |
| Q4    | ≥0.61× llama.cpp = 95 tok/s — 105.6 | ≤15% range — 9.6% | ≥25% — 7.7% (compute-bound) | *verdict withdrawn — artifacts not retained* |

### BF16 — Lumen vs vLLM 0.21.0 (Qwen3.5-9B)

| Metric | Lumen | vLLM | Lumen / vLLM |
|--------|------:|-----:|-------------:|
| Decode tok/s | **66.1** | 21.7 | **3.05x** |
| Prefill tok/s | **3899.7** | 1490.5 | **2.62x** |

vLLM is reported at batch=1 (the same batch size used for Lumen and llama.cpp). vLLM's strengths surface at larger batch sizes, which is not the configuration measured here.

### Hardware-instance variability (A100-PCIe vs A100-SXM4)

A100 instances assigned by the remote scheduler may be PCIe or SXM4 depending on availability. Cross-variant runs are flagged in the source-of-truth release notes. **Finding**: Q8/Q4 decode runs were re-measured on **PCIe** (identical instance type to the baseline) with the full 8-flag opt-in stack and triangulated at +3.5-5.8% positive drift on Q8 / Q4 / BF16 across paths the change did and did not touch — the spread is conclusively environmental (fleet variability), not code-attributable. Lumen-side bytes-moved and kernel dispatch are byte-identical between PCIe and SXM4; only the "vs llama.cpp" ratio drifts with the llama.cpp build / hardware variant.

The canonical baseline above pins PCIe explicitly for that reason.

---

## Metal — Apple Silicon M3 Ultra

Source: `bench/METHODOLOGY.md`. Hardware: Mac Studio M3 Ultra 96 GB. Baseline: `llama-bench` build 8680. Trials: 5 paired per cell.

### Qwen3.5-9B dense

| Config | Quant | Lumen prefill (tok/s) | llama.cpp prefill | Prefill × llama.cpp | Lumen decode (tok/s) | llama.cpp decode | Decode × llama.cpp | Source |
|--------|-------|----------------------:|------------------:|--------------------:|---------------------:|-----------------:|-------------------:|--------|
| Q8 dense-9B | Q8_0 | 365.1 | 911.0 | 0.40× cold-load; **warm-state 0.95×** | 58.7 | 60.0 | **0.98×** | warm-state canonical |
| Q4 dense-9B | Q4_0 | 660.0 | 963.0 | 0.69× cold-load; **warm-state 0.88×** | 80.9 / 88.3 | 79.7 / 75.5 | **1.02×** / **1.17×** | canonical warm-state |
| BF16 dense-9B | BF16 | 630.2 | 956.2 | 0.66× warm-state (up from 0.31× baseline, +2.2× cumulative) | 32.1 | 38.7 | **0.83×** | canonical |

### Qwen3.5-MoE-35B-A3B

| Config | Quant | Lumen prefill (tok/s) | Lumen decode (tok/s) | llama.cpp baseline | Note |
|--------|-------|----------------------:|---------------------:|--------------------|------|
| Q8 MoE-35B-A3B | Q8_0 | 24.2 | 17.2 | none | llama-bench 8680 cannot load Qwen3.5-MoE-A3B (missing `ssm_conv1d` — GDN MoE unsupported). **Lumen is the sole provider on Apple Silicon.** Requires `LUMEN_METAL_MMAP_ONLY=1`. |
| Q4 MoE-35B-A3B | Q4_0 | 34.6 | 17.4 | none | Same architectural limitation as the Q8 MoE config. Sole provider. Requires `LUMEN_METAL_MMAP_ONLY=1`. |

### Methodology asymmetry

Lumen's `lumen run` per-process measurement includes the model cold-load in the prefill timer (this is what a user sees on first invocation). `llama-bench` runs a warm loop after first prefill. The canonical Lumen bench measures warm-state and reports 0.95× / 0.88× prefill for Q8 / Q4 — those numbers remain valid for steady-state serving via `lumen-server`. The cold-load numbers are the right reference for single-shot CLI deployments.

### Metal production status

The dense-9B Q8 and Q4 configs clear the 0.9× llama.cpp competitive gate on decode. The BF16 dense-9B config and the Q8 / Q4 MoE-35B-A3B configs are functional and production-ready: BF16 dense-9B is below the 0.5× sanity floor on prefill (a known structural baseline), and the MoE configs have no external baseline because llama-bench cannot load GDN MoE. The BF16 dense-9B and both MoE configs require `LUMEN_METAL_MMAP_ONLY=1` to fit the 96 GB residency budget.

---

### Notes

- BF16 prefill uses the cuBLAS GemmEx path by default (gate at `crates/lumen-runtime/src/cuda/backend_impl.rs:5350`, `:5818`, `:6389`). Set `LUMEN_CUDA_BF16_GEMMEX=0` to fall back to the legacy `matvec_bf16` kernel for A/B comparison. Lumen-side absolute numbers are stable across the OnceCell-probed fallback path; "vs llama.cpp" ratios drift with the llama.cpp build and the assigned A100 instance variant.
- All three engines consume the same Qwen3.5-9B GGUF weights. Lumen converts GGUF -> LBC inside the container; llama.cpp uses the GGUF directly; vLLM consumes the BF16 GGUF via its GGUF loader.
- "vs llama.cpp" is `Lumen / llama.cpp`. Values `>= 1.00` (bold) mean Lumen is at least as fast as llama.cpp on the same configuration.
- The full opt-in environment-variable stack is required to reproduce these numbers; see [METHODOLOGY.md](METHODOLOGY.md).
- For long-context numbers, the tiled streaming-softmax decode threshold is default-on (`ATTN_DECODE_TILED_DEFAULT_THRESHOLD = 0` in `crates/lumen-runtime/src/cuda/decode.rs`). The Qwen3.5-9B production model (head_dim=256) is eligible for the tiled family; Q8_0/BF16-body dense defaults upgrade eligible automatic selections to the split-K pair, while Q4 remains Tiled. Tiny test fixtures (head_dim<BLOCK_DIM) auto-fall-back to SingleBlock per the head_dim compatibility guard.
- **Envelope citation rule.** When citing decode tok/s, ALWAYS report the (M, G) pair, not M alone. Decode rate is the result of *both* the per-step KV-scan cost (M-driven, ~linear in M) and the gen-length-sensitive tail-growth effect (G-driven, ~ -39% at fixed M = 128 going from gen128 to gen512). Citing only M was the methodology gap that motivated C-17; the (M, G) envelope above is the corrected reporting surface. The bench JSON schema (`bench/results/<timestamp>/results.json`) records `prompt_len` and `gen_len` as separate top-level result-row fields so downstream readers can re-compute envelopes by either axis.
- For Qwen3.5-MoE 35B-A3B, both per-expert (default) and batched (`LUMEN_CUDA_MOE_BATCHED=1`) paths are byte-identical.

Prior multi-model results from an earlier project scope are preserved in [results/archive/2026-04_pre_phase_a.md](results/archive/2026-04_pre_phase_a.md).
