# Environment Variables

Lumen exposes ~130 `LUMEN_*` environment variables. The canonical source-of-truth is the `KNOWN_LUMEN_ENV_VARS` array in [`crates/lumen-runtime/src/runtime_defaults.rs`](../crates/lumen-runtime/src/runtime_defaults.rs).

**Configuration precedence**: CLI flag > environment variable > built-in default. For example, `--kv-precision f32` overrides `LUMEN_KV_PRECISION=f16`; with neither set the per-backend default applies (Metal `f16`, CUDA / CPU `f32`).

## Production canonical stack (CUDA)

The 12-flag CUDA production stack is **default-ON** (set any to `=0` to opt out). The annotated stack with per-flag rationale and gains lives in [`bench/METHODOLOGY.md` § "Required env-vars for full performance"](../bench/METHODOLOGY.md#required-env-vars-for-full-performance) — the single source of truth. Every flag in that stack also appears individually in the [CUDA — production perf levers](#cuda--production-perf-levers-default-on) table below. The one value you must not change is `LUMEN_CUDA_BF16_GEMMEX=0` (required for BF16 P3 correctness on MoE).

## Full catalog (alphabetical, 1-line description)

### Bench harness (general)

| Variable | Description |
|---|---|
| `LUMEN_AB_ITERATIONS` | Paired A/B iteration count for bench harness |
| `LUMEN_AB_WARMUP` | Warmup iterations before A/B measurement starts |
| `LUMEN_BASE_URL` | Override server base URL in client-side tests |
| `LUMEN_BENCH_ITERATIONS` | Measured iteration count for the in-tree bench harness |
| `LUMEN_BENCH_SCALE` | Scale factor for synthetic bench workload sizing |
| `LUMEN_BENCH_TOKENS` | Token budget for bench-driven decode runs |
| `LUMEN_BENCH_WARMUP` | Warmup iteration count before measurement starts |
| `LUMEN_CACHE_DIR` | Override LBC / GGUF cache directory (default `~/.cache/lumen`) |
| `LUMEN_TEST_OPENAI_SDK` | Gate that runs the OpenAI-SDK contract tests when set |

### CUDA — production perf levers (default-ON)

| Variable | Description |
|---|---|
| `LUMEN_CUDA_BF16_GEMMEX` | BF16 GemmEx path. **MUST be 0** for BF16 P3 correctness on MoE. |
| `LUMEN_CUDA_BF16_MOE_V3` | BF16 MoE V3 path; default-ON |
| `LUMEN_CUDA_GDN_REGISTER_RESIDENT` | Register-resident GDN two-launch dispatch; default-ON (+9.4% Q8, +10.3% Q4) |
| `LUMEN_CUDA_MMV_BF16_OUTPUT_PROJ` | BF16 output_proj llama.cpp-parity port; load-bearing for 0.902× llama.cpp |
| `LUMEN_CUDA_MMV_Q_DP4A` | dp4a dense matvec; +7.1% Q8, +6.3% Q4 |
| `LUMEN_CUDA_MMV_Q_MOE_DP4A` | MoE batched dp4a matvec; +11.7% Q4 |
| `LUMEN_CUDA_MOE_BATCHED` | batched MoE FFN dispatch; default-ON |
| `LUMEN_CUDA_MOE_Q4_V3` | MoE Q4 V3 path; default-ON |
| `LUMEN_CUDA_MOE_Q4_V3B` | MoE Q4 V3B variant; default-ON |
| `LUMEN_CUDA_MOE_ROUTER_PARALLEL` | parallel topK MoE router; default-ON |
| `LUMEN_CUDA_MOE_ROUTER_SINGLE_CTA` | single-CTA topK MoE router; default-ON |
| `LUMEN_CUDA_TOPK_MOE_FUSED` | fused topK MoE router (sigmoid+top-K+renorm in one kernel); +6-8% all MoE quants |

### CUDA — opt-in perf experiments (default OFF)

| Variable | Description |
|---|---|
| `LUMEN_CUDA_BF16_AUTOTUNE` | BF16 autotune sweep (off in production) |
| `LUMEN_CUDA_DECODE_GRAPH` | CUDA graph capture for decode (experimental) |
| `LUMEN_CUDA_DECODE_GRAPH_QGATE` | CUDA graph + Q-gate fusion variant |
| `LUMEN_CUDA_DECODE_GRAPH_TILED` | CUDA graph for tiled decode kernel |
| `LUMEN_CUDA_DECODE_TILED` | Force tiled streaming-softmax decode kernel |
| `LUMEN_CUDA_DECODE_TILED_THRESHOLD` | Token threshold above which tiled kernel engages (default 0 = always-on); set to `4294967295` to force single-block path |
| `LUMEN_CUDA_FA2_ATTN` | Flash Attention 2 decode (per-call HBM alloc dominates at batch=1) |
| `LUMEN_CUDA_FA2_BLOCKSKIP` | FA2 prefill with block-skip causal kernel |
| `LUMEN_CUDA_FFN_FUSED_GLU` | Fused gate+up+SwiGLU FFN variant |
| `LUMEN_CUDA_GDN_AB_F32` | F32 A-tile / B-tile accumulators in GDN |
| `LUMEN_CUDA_GDN_F64_ACCUM` | F64 GDN delta-rule accumulator; **default-ON for MoE** — the decode-vs-prefill parity fix that lands MoE arithmetic at greedy; no-op/OFF for dense (see [MoE correctness defaults](#cuda--moe-correctness-defaults-auto-on-for-moe-no-op-for-dense)) |
| `LUMEN_CUDA_GDN_PHASE4_COAL` | GDN phase-4 coalesced dispatch |
| `LUMEN_CUDA_GDN_SPLIT` | Split layout for GDN tensors (Q4; +2.6% Q4 decode) |
| `LUMEN_CUDA_L2NORM_RSQRTF` | Two-step rsqrtf L2 norm variant (`rsqrtf(fmaxf(ss,eps^2))`, one HW op) |
| `LUMEN_CUDA_MMV_Q_OUTPUT_PROJ` | Q8/Q4 output_proj llama.cpp-parity port (within noise floor) |
| `LUMEN_CUDA_MOE_BATCHED_V2` / `_V3` | MoE batched FFN variants V2 / V3 |
| `LUMEN_CUDA_MOE_DECODE_GRAPH` | CUDA graph capture for MoE decode |
| `LUMEN_CUDA_MOE_FUSED_NORM_ROUTER` | Fused norm + router kernel |
| `LUMEN_CUDA_MOE_FUSED_PERSISTENT` | Persistent fused MoE kernel |
| `LUMEN_CUDA_MOE_SHARED_FUSED` | Fused shared-expert path |
| `LUMEN_CUDA_NORM_RSQRTF_BUNDLE` | Umbrella gate: enables L2NORM_RSQRTF + RMSNORM_RSQRTF + GDN_AB_F32 together (rsqrtf-norm bundle) |
| `LUMEN_CUDA_OUTPUT_PROJ_F16_CACHE` | F16 cache for output_proj |
| `LUMEN_CUDA_OUTPUT_PROJ_NR` | Override NR for output projection (16 / 32) |
| `LUMEN_CUDA_OUTPUT_PROJ_SPLIT` | Split-K layout for 1 GB output projection |
| `LUMEN_CUDA_PREFILL_F32` | Force F32 prefill path |
| `LUMEN_CUDA_Q4_SPLIT` | Raw + split layout for Q4_0 weights (+9.0% Q4 decode) |
| `LUMEN_CUDA_Q4_TILE` | Q4 matmul tile size override |
| `LUMEN_CUDA_Q8_AOS_NR8` / `LUMEN_CUDA_Q8_SPLIT_NR8` | Q8 NR=8 dp4a-mmvq variants (AoS / SPLIT layouts) |
| `LUMEN_CUDA_Q8_SPLIT_4THREAD` | Q8 SPLIT 4-threads-per-block dp4a-mmvq variant |
| `LUMEN_CUDA_Q8_PROJ_MMQ` | Q8 projection via MMQ |
| `LUMEN_CUDA_Q8_SCALE_HW` | Native `LDG.E.U16` scale fetch for Q8 matvec (+0.4% Q8 decode) |
| `LUMEN_CUDA_Q8_SPLIT` | Raw + split layout for Q8_0 weights (+4.5% Q8 decode) |
| `LUMEN_CUDA_Q8_SSM_OUT_MMQ_OFF` | Disable MMQ for Q8 SSM-out |
| `LUMEN_CUDA_Q8_TILE` | Q8 matmul tile size override |
| `LUMEN_CUDA_RMSNORM_RSQRTF` | rsqrtf + block-wide warp-shuffle RMSNorm kernel |
| `LUMEN_CUDA_SKIP_BF16_PROBE` | Skip BF16 GemmEx capability probe at startup |
| `LUMEN_CUDA_SKIP_SHARED_EXPERT` / `_GATE` | Skip shared-expert path / gate (debug) |

### CUDA — MoE correctness defaults (auto-ON for MoE, no-op for dense)

These flags fix the MoE GDN decode-vs-prefill divergence that the 256-expert
router amplifies into garble. Each is **default-ON only for MoE models** (gated
on `model_is_moe()`); for dense models the gate forces them OFF so dense output
stays **byte-identical** to history regardless of the env (`LUMEN_ANTI_RESTATE`
narrows further to BF16 MoE only — see its row). Source of truth:
`runtime_defaults` (`gdn_ab_f16_default`, `gdn_decode_via_prefill_default`,
`gdn_convstate_parity_default`, `gdn_f64_accum_default`, `anti_restate_default`).
Set any to `=0` to opt out on MoE, `=1` to force on.

| Variable | Description |
|---|---|
| `LUMEN_CUDA_GDN_AB_F16` | Route the GDN `ssm_alpha`/`ssm_beta` projections through the same F16-cache `cublasGemmEx` recipe in BOTH decode and prefill, making alpha/beta bit-identical decode-vs-prefill; **default-ON for MoE**, no-op for dense |
| `LUMEN_CUDA_GDN_DECODE_VIA_PREFILL` | Run the whole MoE GDN decode recurrence block (conv1d + gates + L2-norm + delta-rule + norm-gate) through the PREFILL fused kernels at `T=1`, so GDN-decode == GDN-prefill by construction; **default-ON for MoE**, no-op for dense |
| `LUMEN_CUDA_GDN_CONVSTATE_PARITY` | Compute the decode GDN qkv projection via the prefill `launch_gemm_projection` path at `batch=1` so the new conv-ring slot's `conv_state` bit-matches a true prefill; **default-ON for MoE** (requires `GDN_DECODE_VIA_PREFILL`), no-op for dense |
| `LUMEN_CUDA_GDN_F64_ACCUM` | F64 GDN delta-rule accumulator — the foundational decode-vs-prefill parity fix that lands MoE arithmetic at greedy; **default-ON for MoE**, no-op/OFF for dense |
| `LUMEN_ANTI_RESTATE` | Deterministic greedy anti-degeneration veto applied after argmax; suppresses a single near-tie sub-word doubling / n-gram restate the MoE decode picks but llama.cpp does not. **Default-ON for BF16 MoE only** (the resolver keeps it effectively off for q8/q4 MoE, whose math basin needs the un-vetoed token); OFF for dense |

### CUDA — diagnostics & legacy

| Variable | Description |
|---|---|
| `LUMEN_CUDA_DECODE_DELAY_US` | CPU sleep (µs) after `device.synchronize()` per decode step; serializes inter-step submission to close a GPU-scheduler timing race for MoE Q4 server determinism. Default is `50` on `lumen-server` (applied automatically) and `0` on the `lumen run` CLI, which is deterministic without it. |
| `LUMEN_CUDA_LEGACY_DEFAULTS` | Roll back to legacy default values |
| `LUMEN_CUDA_LEVER_TRACE` | Trace which perf levers fire at runtime |
| `LUMEN_CUDA_MAX_SEQ_LEN` | Cap KV-cache max sequence length |
| `LUMEN_CUDA_MOE_DEBUG_DUMP` | Dump MoE intermediate tensors |
| `LUMEN_CUDA_PROFILE` | Enable CUDA-side profiling |
| `LUMEN_CUDA_Q4_V3_TRACE` | Trace Q4 V3 kernel dispatch |
| `LUMEN_CUDA_VERBOSE` | Verbose CUDA backend logging |

### Metal — perf levers (default-ON, opt-out via `LUMEN_METAL_DEFAULTS_OFF=1`)

| Variable | Description |
|---|---|
| `LUMEN_METAL_CONCURRENT_ENCODER` | concurrent-encoder + 4-way QKV split for GDN; default-ON |
| `LUMEN_METAL_GDN_CONCURRENT_ENCODER` | GDN concurrent-encoder dispatch variant (paired with above) |
| `LUMEN_METAL_GDN_CONCURRENT_ENCODER_VALIDATE` | Validate the GDN concurrent-encoder path against the legacy path |
| `LUMEN_METAL_Q8_REPACKED` | runtime Q8 hot-weight repack (+6.89% prefill); default-ON |
| `LUMEN_METAL_Q8_REPACKED_FFN_DOWN` / `_GATE_UP` | Sub-gates for Q8 repack on FFN-down / gate+up |
| `LUMEN_METAL_FFN_DOWN_SPLITK` | FFN-down K=8 Split-K (+1.89% Q8 prefill) |
| `LUMEN_METAL_FFN_DOWN_SPLITK_BF16` | BF16 variant of Split-K FFN-down |
| `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED` | fused FFN gate+up+SwiGLU kernel (+2.6%) |
| `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_Q4` / `_BF16` | Q4 / BF16 ports of the fused kernel |
| `LUMEN_METAL_DEFAULTS_OFF` | Master kill-switch: revert all default-ON levers |

### Metal — opt-in experiments (default OFF)

| Variable | Description |
|---|---|
| `LUMEN_METAL_BF16_GATE_UP_NR` | BF16 gate+up NR override |
| `LUMEN_METAL_BF16_GDN_FULL_PREFILL_WARMUP` / `_QKV_GATE_PAIRED` / `_TILE_NOK64` / `_WARMUP` / `_WARMUP_MODE` | BF16 GDN prefill experiments |
| `LUMEN_METAL_BF16_MMAP_ONLY` | mmap-only loader for BF16 (parity with `LUMEN_METAL_MMAP_ONLY`) |
| `LUMEN_METAL_BF16_MPS` | Use MPS Graph for BF16 |
| `LUMEN_METAL_CONCURRENT_ENCODER_FULL` / `_TRACE` / `_VALIDATE` / `_FULL_VALIDATE` | LOOKAHEAD=64 + concurrent encoder validation |
| `LUMEN_METAL_GDN_PHASE2A_NSG4` | (32, NSG=4, 1) TG geometry — empirically refuted on M3 Ultra |
| `LUMEN_METAL_GDN_SSM_OUT_F32_BATCHED` | F32 batched SSM-out path |
| `LUMEN_METAL_GEMM_GGML_PORT` | ggml-metal ported Q8_0 GEMM variant |
| `LUMEN_METAL_MULTI_CB` | multi command-buffer split (`_N` selects K∈{2,3,4,8}) |
| `LUMEN_METAL_MULTI_CB_N` | Multi-CB K parameter |
| `LUMEN_METAL_Q4_REPACKED` / `_FFN_DOWN` / `_GATE_UP` | Q4 repack (NULL on M3 Ultra; off by default) |
| `LUMEN_METAL_UNRETAINED_CMDBUFS` | Use unretained command-buffer allocation |

### Metal — required operating env (BF16 dense / Q8 MoE / Q4 MoE)

| Variable | Description |
|---|---|
| `LUMEN_METAL_MMAP_ONLY` | **Required** for Metal BF16 dense, Q8 MoE, and Q4 MoE on M3 Ultra 96 GB to avoid OOM-kill during model load |

### Metal — diagnostics

| Variable | Description |
|---|---|
| `LUMEN_METAL_DECODE_DELAY_US` | CPU sleep (µs) after the per-token `commit_and_wait()` in greedy decode. Default `0` (bit-exact no-op; the decode non-determinism it once mitigated is now fixed at the kernel level, so the delay is unnecessary). Retained for diagnostics — set e.g. `=50` to re-introduce the inter-token pause when investigating GPU-scheduler timing. |
| `LUMEN_METAL_NAN_DUMP` | Dump tensors when a NaN is detected |
| `LUMEN_METAL_PROFILE` / `_DEEP` / `_GDN` | Metal-side profiling levels |

### KV cache

| Variable | Description |
|---|---|
| `LUMEN_KV_PRECISION` | KV cache precision (`f16` / `f32`). Per-backend defaults: Metal `f16`, CUDA / CPU `f32` |

### Sessions / suffix prefill

| Variable | Description |
|---|---|
| `LUMEN_SUFFIX_THRESHOLD` | Minimum shared-prefix length to attempt suffix prefill |

### Diagnostics / dumps (off in production)

| Variable | Description |
|---|---|
| `LUMEN_DEBUG_DUMP_SSM_BETA_W` | Dump SSM β weight tensor |
| `LUMEN_DUMP_EXPERTS` | Dump per-expert routing decisions |
| `LUMEN_DUMP_GDN_L0_BIN` | Dump GDN layer-0 binaries |
| `LUMEN_DUMP_NORMED` | Dump post-RMSNorm activations |
| `LUMEN_DUMP_RUNTIME_Q8_SCALES` | Dump runtime Q8 scale factors |
| `LUMEN_GRAPH_DIAGNOSTIC` | Emit CUDA-graph capture diagnostics |

### Server (`lumen-server`)

| Variable | Description |
|---|---|
| `LUMEN_SERVER_DEBUG_MEM` | Enable server memory-tracking instrumentation |
| `LUMEN_SERVER_PANIC_MAX` | Max panic count before server self-terminates |
| `LUMEN_SERVER_PANIC_WINDOW_SECS` | Window size (seconds) for `PANIC_MAX` counter |
| `LUMEN_SERVER_PER_JOB_RESET` | Reset KV cache between jobs (multi-tenant) |

### Soak harness

| Variable | Description |
|---|---|
| `LUMEN_SOAK_DURATION_SEC` | Soak test duration |
| `LUMEN_SOAK_OUT_DIR` | Soak output directory |
| `LUMEN_SOAK_STACK_DUMP` / `_LEAKS` / `_TICKS` | Stack-dump diagnostics for leak hunting |
| `LUMEN_SOAK_WARMUP_SEC` | Post-warmup window for soak-regression measurement (default 300 s) |

### Test-fixture overrides (per model family)

v1 ships fixture-path overrides for Qwen3.5-9B. As new model families ship, additional fixture-path env vars will be added alongside under the same naming convention.

| Variable | Description |
|---|---|
| `LUMEN_QWEN35_9B_PATH` / `_Q8` / `_Q4` / `_BF16` | Override Qwen3.5-9B LBC path / quant fixture for tests (v1 fixture set) |

