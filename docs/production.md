# Production Deployment

Read this before deploying Lumen to production. Source: the two-backend (CUDA + Metal) validation matrix, verified end-to-end against llama.cpp; throughput and capacity figures cite their per-claim provenance below (the retained records are the 2026-07-16 battery and the A100/H100 soaks).

## Choose the right serving mode

| Mode | Use when | Cold-start | Concurrent clients (per GPU) | Validated |
|------|----------|-----------|------------------------------|-----------|
| `lumen run` (CLI per-process) | Single request, batch jobs, dev/test | 60–120 s (model load every invocation) | **1–2 max** | stability soak |
| `lumen-server` (long-running) | Concurrent clients, interactive serving | One-time at boot (~90 s warm including autotune) | **8–32** (single-tenant Session) | streaming / concurrency |

The CLI cold-loads weights on every invocation. A 16-client concurrent burst against `lumen run` measured **82.4 % timeout rate**. For any concurrent serving, use `lumen-server`.

## Production-readiness checklist

Operational policy items required before production deployment:

1. **GPU reservation**:
   - BF16 MoE-35B-A3B → dedicated H100/H200-class GPU (validated there; the A100-80GB fit is unverified — see the headroom warning below). No co-tenant workloads.
   - Q8 MoE → ≥62 GiB free GPU minimum (measured peak 61,089 MiB; A100-80GB shared is OK if peer load < 17 GiB).
   - Q4 MoE → 30 GB free GPU minimum (A100-40GB OK).
   - Dense-9B → 24 GB free GPU minimum (A100-40GB / L40S / 3090 / 4090 OK).
2. **Pre-warm LBC into OS page cache** at service start: `cat /path/to/model.lbc > /dev/null` before accepting traffic. Avoids 30–70 s NVMe penalty on first request.
3. **For concurrent clients**: deploy `lumen-server` (NOT repeated `lumen run` invocations).
4. **For multilingual prompts**: pass `--max-tokens 512` minimum. Below ~256 the model may burn the full budget in `<think>...</think>` reasoning before producing the answer in the target language.
5. **For long-form generation (≥ 512 tokens)**: avoid PURE-greedy (`--temperature 0` + no penalty) — deterministically loops. Use sampling (`--temperature 0.7`) OR, **on DENSE models only**, `--repetition-penalty 1.05 --repeat-last-n 64`. When `--repetition-penalty` is omitted the server/CLI apply a **model-aware** default (1.05 dense / **1.03 MoE**, resolved by `runtime_defaults::repetition_penalty_default`); **MoE must stay ≤ 1.03** — a penalty of 1.05+ corrupts MoE arithmetic ("17 × 20 = … = 39"). Leave the flag unset on MoE so the 1.03 default applies.
6. **Pin `--context-len`** for BF16 deployments. The BF16 mmvf kernel produces different first-token argmax at different KV-cache layout sizes. Fix at a single value (e.g. `--context-len 8192`) per deployment.
7. **Canonical env stack**: the 12-flag CUDA production stack is **default-ON**, so out-of-the-box `lumen run` uses the canonical flag stack. The one value you must not change is `LUMEN_CUDA_BF16_GEMMEX=0` (the explicit value required for BF16 P3 correctness on MoE). Full annotated stack with per-flag gains: [`bench/METHODOLOGY.md`](../bench/METHODOLOGY.md#required-env-vars-for-full-performance).
8. **Metal BF16-dense / Q8-MoE / Q4-MoE require `LUMEN_METAL_MMAP_ONLY=1`** to fit in the M3 Ultra 96 GB residency budget. This is a documented operating requirement, not a defect.
9. **CUDA driver / CUDA runtime**: validated on driver 580.126.20, CUDA 12.2.140, sm_80 (A100). NVRTC compiles kernels at runtime; no build-time CUDA SDK required.
10. **LBC format compatibility**: current `LBC_VERSION = 4`. Reader rejects newer-than-current with `UnsupportedVersion`; backward-compat for v1/v2 is in the code path but unverified at runtime. **Policy: rebuild LBCs after major Lumen upgrades** via `lumen convert` or `lumen pull --quant <scheme>`.

## Known limitations (will NOT be fixed in v0.1.0)

- **Concurrency C ≥ 4 per GPU under CLI mode**: structurally unsupported; cold-start contention dominates. Use `lumen-server` instead.
- **Prefill × llama.cpp ratio is structurally below 1.0** at all quants on the current NVRTC compute_61 + non-monolithic-encoder stack.
- **MoE-35B-A3B decode vs llama.cpp — retained record**: Q8 0.567× and Q4 0.598× on A100 (co-located); BF16 0.575× on H100 (separate per-engine batteries; 104.1 vs 181.1 tok/s — the highest absolute throughput of the three but the lowest ratio; A100 decode unmeasured). The previously published 0.902×/0.584×/0.674× figures have no retained measurement artifacts.
- **PURE-greedy long-form (≥ 512 tokens)** deterministically loops on all 4 quants. Use sampling or repetition penalty in production.
- **`lumen-server` mid-stream client disconnect** can wedge the engine worker. Pending fix; work around with a reverse-proxy that buffers SSE responses.
- **`lumen-server` Authorization / CORS / per-request timeout** are not implemented; deploy behind a reverse proxy that enforces auth, CORS, and request deadlines.
- **Lumen chat template forces `<think>\n` open for the v1 Qwen3.5 family**, the `--prompt` path on `lumen run` bypasses the template; for server use, the production behavior is the chat-templated path. Future model families will register their own chat templates without changing the dispatch layer.

## GPU memory peaks

| Quant | Qwen3.5-9B (peak VRAM) | Qwen3.5-MoE-35B-A3B (peak VRAM, 5-trial) |
|-------|-----------------------:|-------------------------------------------------------:|
| Q4_0  | ~5.1 GB                | **24.1 GB** *(artifact not retained; LBC 20.7 GB)*     |
| Q8_0  | ~10.0 GB / ~22.9 GB (with cuBLAS workspace + cache) | **61,089 MiB ≈ 59.7 GiB** (A100 soak; LBC 37.6 GB) |
| BF16  | ~17.8 GB               | **72,475 MiB ≈ 70.8 GiB** (H100 soak; LBC 69.7 GB)     |

Qwen3.5-MoE-35B-A3B loads at Q4_0 and Q8_0 on a single A100-80GB (measured). The BF16 cell has **no retained A100 load artifact**: the retained BF16 measurements (decode battery, 1-hour soak) are from H100/H200, and the unretained records conflict — a withdrawn 2026-06-02 dataset recorded an A100 load, while a later validation note recorded the load exceeding A100-80GB. Treat A100 BF16 as unvalidated. The BF16 peak-VRAM figure above was measured on H100; the A100 estimate is ≈74,955 MiB ≈ 73.2 GiB — the H100 peak plus ≈2,480 MiB for sm_80's F32 upcast of the non-expert BF16 projection set (source-derived estimate ≈1.03e9 elements × 2 extra bytes ≈ 1,965 MiB; the embedding and output head upload raw BF16 on every architecture and are excluded) and an estimated ≈515 MiB of sm_80-only F16 prefill caches. An independently derived estimate of the same delta gives ≈2,575 MiB — within 4% of this one.

**BF16 MoE-35B-A3B headroom warning**: the H100-measured peak is 72,475 MiB; the A100 estimate is ≈74,955 MiB against the 81,152 MiB an A100-80GB PCIe reports, leaving ≈6.0 GiB estimated nominal headroom — an upper bound (CUDA context and driver reserve consume several hundred MiB before any allocation), and the conflicting unretained load records above mean even that margin is unconfirmed. Any concurrent process consuming a few GiB can race `cuMemAlloc` and cause OOM mid-upload. Deploy BF16 MoE on validated H100/H200-class hardware. No co-tenant workloads. For shared-GPU deployments, use Q8 (61,089 MiB ≈ 59.7 GiB peak) or Q4 (24.1 GB peak).

KV cache is sized to `--context-len` (the server defaults to 8192; the CLI sizes it to the prompt plus generation plus headroom). On CUDA the F16 dequant caches are then checked against the memory left: a load whose caches plus a 512 MiB headroom exceed free memory is refused with the bytes needed and free, so lower `--context-len` or use a smaller quantization (`LUMEN_CUDA_F16_CACHE_FORCE=1` overrides the check). KV growth is bit-perfect to the theoretical formula: `max_seq_len × num_layers × num_kv_heads × head_dim × 4 (F32) × 2 (K + V)`.

## Operational caveats summary

The matrix below summarizes the validation state across operational dimensions and the caveats that matter at deploy time.

| Dimension | State |
|-----------|-------|
| Models × Quants matrix | Validated — Q8/Q4 throughput on A100; BF16 decode measured on H100/H200 only (A100-80GB BF16 MoE unvalidated) |
| Correctness suite | Greedy parity differs from llama.cpp (root cause: chat template) |
| KV cache & memory | Validated (single-tenant) |
| Long-form generation | PURE-greedy loops; BF16 first-token argmax is context-length-sensitive — pin `--context-len` |
| Streaming & server protocol | Mid-stream disconnect can wedge the worker |
| Concurrency & multi-request | Validated (no 503 + Retry-After backpressure header) |
| Stability & soak | Validated (CLI per-process); a 16-client burst against `lumen run` fails by design — use `lumen-server` |
| Error handling & edge cases | Four protocol-completeness gaps remain; deploy behind a reverse proxy |
| Determinism & reproducibility | Validated — kernels byte-deterministic at a fixed seed; server + CLI randomize the seed by default, so pin `seed` / `--seed` (or `temperature 0`) to reproduce |
| Perf vs llama.cpp | Retained record: Q8 0.567× / Q4 0.598× (A100, co-located); BF16 0.575× (H100, separate per-engine batteries) |
