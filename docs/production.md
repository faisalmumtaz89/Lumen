# Production Deployment

Read this before deploying Lumen to production. Source: the two-backend (CUDA + Metal) validation matrix, verified end-to-end against llama.cpp (2026-06-02).

## Choose the right serving mode

| Mode | Use when | Cold-start | Concurrent clients (per GPU) | Validated |
|------|----------|-----------|------------------------------|-----------|
| `lumen run` (CLI per-process) | Single request, batch jobs, dev/test | 60–120 s (model load every invocation) | **1–2 max** | stability soak |
| `lumen-server` (long-running) | Concurrent clients, interactive serving | One-time at boot (~90 s warm including autotune) | **8–32** (single-tenant Session) | streaming / concurrency |

The CLI cold-loads weights on every invocation. A 16-client concurrent burst against `lumen run` measured **82.4 % timeout rate**. For any concurrent serving, use `lumen-server`.

## Production-readiness checklist

Operational policy items required before production deployment:

1. **GPU reservation**:
   - BF16 MoE-30B-A3B → dedicated 80 GB+ GPU (A100-80GB, H100, MI300). No co-tenant workloads.
   - Q8 MoE → 56 GB free GPU minimum (A100-80GB shared is OK if peer load < 20 GB).
   - Q4 MoE → 30 GB free GPU minimum (A100-40GB OK).
   - Dense-9B → 24 GB free GPU minimum (A100-40GB / L40S / 3090 / 4090 OK).
2. **Pre-warm LBC into OS page cache** at service start: `cat /path/to/model.lbc > /dev/null` before accepting traffic. Avoids 30–70 s NVMe penalty on first request.
3. **For concurrent clients**: deploy `lumen-server` (NOT repeated `lumen run` invocations).
4. **For multilingual prompts**: pass `--max-tokens 512` minimum. Below ~256 the model may burn the full budget in `<think>...</think>` reasoning before producing the answer in the target language.
5. **For long-form generation (≥ 512 tokens)**: avoid PURE-greedy (`--temperature 0` + no penalty) — deterministically loops. Use sampling (`--temperature 0.7`) OR `--repetition-penalty 1.05 --repeat-last-n 64`.
6. **Pin `--context-len`** for BF16 deployments. The BF16 mmvf kernel produces different first-token argmax at different KV-cache layout sizes. Fix at a single value (e.g. `--context-len 8192`) per deployment.
7. **Canonical env stack**: the 12-flag CUDA production stack is **default-ON**, so out-of-the-box `lumen run` reproduces the published gate-clear numbers. The one value you must not change is `LUMEN_CUDA_BF16_GEMMEX=0` (the explicit value required for BF16 P3 correctness on MoE). Full annotated stack with per-flag gains: [`bench/METHODOLOGY.md`](../bench/METHODOLOGY.md#required-env-vars-for-full-performance).
8. **Metal BF16-dense / Q8-MoE / Q4-MoE require `LUMEN_METAL_MMAP_ONLY=1`** to fit in the M3 Ultra 96 GB residency budget. This is a documented operating requirement, not a defect.
9. **CUDA driver / CUDA runtime**: validated on driver 580.126.20, CUDA 12.2.140, sm_80 (A100). NVRTC compiles kernels at runtime; no build-time CUDA SDK required.
10. **LBC format compatibility**: current `LBC_VERSION = 3`. Reader rejects newer-than-current with `UnsupportedVersion`; backward-compat for v1/v2 is in the code path but unverified at runtime. **Policy: rebuild LBCs after major Lumen upgrades** via `lumen convert` or `lumen pull --quant <scheme>`.

## Known limitations (will NOT be fixed in v0.1.0)

- **Concurrency C ≥ 4 per GPU under CLI mode**: structurally unsupported; cold-start contention dominates. Use `lumen-server` instead.
- **Prefill × llama.cpp ratio is structurally below 1.0** at all quants on the current NVRTC compute_61 + non-monolithic-encoder stack.
- **Q8 / Q4 decode × llama.cpp ratio is structurally locked at 0.584× / 0.674×** on CUDA MoE-30B-A3B. **For llama.cpp-equivalent throughput on MoE-30B-A3B, deploy BF16** (0.902× llama.cpp empirical, requires 80 GB+ GPU).
- **PURE-greedy long-form (≥ 512 tokens)** deterministically loops on all 4 quants. Use sampling or repetition penalty in production.
- **`lumen-server` mid-stream client disconnect** can wedge the engine worker. Pending fix; work around with a reverse-proxy that buffers SSE responses.
- **`lumen-server` Authorization / CORS / per-request timeout** are not implemented; deploy behind a reverse proxy that enforces auth, CORS, and request deadlines.
- **Lumen chat template forces `<think>\n` open for the v1 Qwen3.5 family**, the `--prompt` path on `lumen run` bypasses the template; for server use, the production behavior is the chat-templated path. Future model families will register their own chat templates without changing the dispatch layer.

## GPU memory peaks

| Quant | Qwen3.5-9B (peak VRAM) | Qwen3.5-MoE-30B-A3B (peak VRAM, 5-trial) |
|-------|-----------------------:|-------------------------------------------------------:|
| Q4_0  | ~5.1 GB                | **24.1 GB** (LBC 20.7 GB)                              |
| Q8_0  | ~10.0 GB / ~22.9 GB (with cuBLAS workspace + cache) | **54.3 GB** (LBC 37.6 GB) |
| BF16  | ~17.8 GB               | **72.4 GB** (LBC 69.7 GB)                              |

Qwen3.5-MoE-30B-A3B at all three quants fits on a single A100-80GB (validated under the models-and-quants matrix audit).

**BF16 MoE-30B-A3B headroom warning**: peak 72.4 GB on 80 GB A100 leaves only ~7.6 GB headroom. Any concurrent process consuming > 5 GB can race `cuMemAlloc` and cause OOM mid-upload. In a multi-tenant deployment, BF16 MoE requires a dedicated 80 GB+ GPU reservation. No co-tenant workloads. For shared-GPU deployments, use Q8 (54 GB peak) or Q4 (24 GB peak).

KV cache is auto-sized to fit remaining VRAM; `--context-len` overrides. KV growth is bit-perfect to the theoretical formula: `max_seq_len × num_layers × num_kv_heads × head_dim × 4 (F32) × 2 (K + V)`.

## Operational caveats summary

The matrix below summarizes the validation state across operational dimensions and the caveats that matter at deploy time.

| Dimension | State |
|-----------|-------|
| Models × Quants matrix | Validated (BF16 MoE needs an 80 GB+ GPU) |
| Correctness suite | Greedy parity differs from llama.cpp (root cause: chat template) |
| KV cache & memory | Validated (single-tenant) |
| Long-form generation | PURE-greedy loops; BF16 first-token argmax is context-length-sensitive — pin `--context-len` |
| Streaming & server protocol | Mid-stream disconnect can wedge the worker |
| Concurrency & multi-request | Validated (no 503 + Retry-After backpressure header) |
| Stability & soak | Validated (CLI per-process); a 16-client burst against `lumen run` fails by design — use `lumen-server` |
| Error handling & edge cases | Four protocol-completeness gaps remain; deploy behind a reverse proxy |
| Determinism & reproducibility | Validated — kernels byte-deterministic at a fixed seed; server + CLI randomize the seed by default, so pin `seed` / `--seed` (or `temperature 0`) to reproduce |
| Perf parity vs llama.cpp | BF16 0.90×; Q8 / Q4 at their structural gates |
