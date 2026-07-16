# Environment Variables

Lumen reads its runtime configuration from `LUMEN_*` environment variables. The
canonical source-of-truth for the recognised set is the `KNOWN_LUMEN_ENV_VARS`
allowlist in
[`crates/lumen-runtime/src/runtime_defaults.rs`](../crates/lumen-runtime/src/runtime_defaults.rs);
at startup `validate_lumen_env_vars()` warns about any `LUMEN_*` var that is set
but not on that list (typo protection). Two unit tests keep this document
honest: `allowlist_members_do_not_warn_when_set` (allowlist ⇒ no warning) and
`all_read_env_vars_are_registered` (every var **read** in the code is on the
allowlist — no read-but-unregistered var can false-warn).

**Configuration precedence:** CLI flag > environment variable > built-in
default. For example `--kv-precision f32` overrides `LUMEN_KV_PRECISION=f16`;
with neither set the per-backend default applies (Metal `f16`, CUDA / CPU
`f32`).

## Categories

Every variable below is tagged with one category:

| Category | Meaning | Rule of thumb |
|---|---|---|
| **kill-switch** | A shipped default-ON optimization with an off-switch. Toggling only chooses between two byte-identical (or measured-equivalent) code paths. | Leave at the default; set `=0` only to A/B a suspected regression. |
| **config** | An operator-tunable value or opt-in path. | Change deliberately for the documented effect. |
| **diagnostic** | Off by default; when enabled it prints/dumps or times something. Default state is byte-identical to unset. | Set only while investigating; unset in production. |
| **test-fixture** | Read only by tests / benches, never by the shipped `lumen` / `lumen-server` runtime path. | Ignore in production; used by the harness. |

## Production canonical stack (CUDA)

The CUDA production stack is **default-ON** — you get full performance with no
env set. The annotated stack with per-flag gains lives in
[`bench/METHODOLOGY.md` § "Required env-vars for full performance"](../bench/METHODOLOGY.md#required-env-vars-for-full-performance),
the single source of truth for benchmarking. The one value you must not change
is `LUMEN_CUDA_BF16_GEMMEX=0` (required for BF16 P3 correctness on MoE).

---

## General runtime

| Variable | Default | Category | Effect | When to touch |
|---|---|---|---|---|
| `LUMEN_KV_PRECISION` | backend: Metal `f16`, CUDA/CPU `f32` | config | KV-cache precision (`f16`/`f32`); invalid value warns + falls back. CLI `--kv-precision` wins. **Metal is F16-only for the KV cache** — `f32` is rejected with a clear error (GPU-resident KV buffers are allocated F16); use `f32` only on CUDA/CPU. | `f16` on CUDA to halve KV memory; `f32` for a numerical-parity check on CUDA/CPU. |
| `LUMEN_SUFFIX_THRESHOLD` | built-in `DEFAULT_SUFFIX_THRESHOLD` | config | Minimum shared-prefix length before suffix-prefill reuse engages (positive-int; warns + falls back on invalid). | Raise to suppress suffix reuse; lower to reuse shorter shared prefixes. |
| `LUMEN_REPEAT_LAST_N` | unset = full-history window | config | Finite recent-token window for the repetition penalty (e.g. `64`). Byte-identical to history-wide at default. | Set a finite window to bound repetition-penalty scope. |
| `LUMEN_REPETITION_PENALTY` | `1.05` dense / `1.03` MoE | config | Repetition penalty (server wire); bounds-checked (finite, `>0`). `=1.0` disables. | Tune output repetition; `=1.0` for pure greedy. |
| `LUMEN_FREQUENCY_PENALTY` | `0.0` (no-op) | config | OpenAI-style frequency penalty (finite, `≥0`); `0.0` is byte-identical. | Set `>0` to penalise frequent tokens. |
| `LUMEN_CHAT_ENABLE_THINKING` | `false` | config | Emits the model's `<think>` block in chat formatting; bad values fall back to default. | Enable to surface reasoning traces. |
| `LUMEN_CACHE_DIR` | `dirs::cache_dir()/lumen` or `~/.cache/lumen` | config | Override LBC / GGUF download cache directory (empty-string guarded). | Point at a larger / faster volume for model artifacts. |
| `LUMEN_BUILD_VERSION` | falls back to `CARGO_PKG_VERSION` | config | **Compile-time** `option_env!` stamped into the version banner. Not a runtime var (intentionally not in the runtime allowlist). | Set at build time to embed a custom build tag. |

## Sampling determinism / anti-degeneration (CUDA MoE)

| Variable | Default | Category | Effect | When to touch |
|---|---|---|---|---|
| `LUMEN_ANTI_RESTATE` | ON for **BF16 MoE** only; OFF dense & q8/q4 MoE | kill-switch | Deterministic greedy veto after argmax; suppresses a single near-tie sub-word doubling / n-gram restate the BF16-MoE decode picks but llama.cpp does not. | Leave as-is; `=0` to observe the un-vetoed token, `=1` to force on. |
| `LUMEN_ANTI_RESTATE_SUBWORD` | ON (while veto active) | kill-switch | Enables the sub-word-doubling arm of the anti-restate veto. | `=0` to isolate one veto rule during triage. |
| `LUMEN_ANTI_RESTATE_NGRAM` | ON (while veto active) | kill-switch | Enables the n-gram-restate arm of the veto. | `=0` to isolate one veto rule during triage. |
| `LUMEN_ANTI_RESTATE_LOOP` | ON (while veto active) | kill-switch | Enables the loop-restate arm of the veto. | `=0` to isolate one veto rule during triage. |

---

## CUDA — production kill-switches (default-ON)

Set any to `=0` to opt out. Model-aware defaults are noted per row.

| Variable | Default | Category | Effect | When to touch |
|---|---|---|---|---|
| `LUMEN_CUDA_BF16_GEMMEX` | model-aware: BF16 ON, quant OFF | kill-switch | BF16 GemmEx path. **MUST be `0`** for BF16 P3 correctness on MoE. | Leave default; never force on for MoE. |
| `LUMEN_CUDA_BF16_MOE_V3` | ON (no-op on dense/Q8/Q4) | kill-switch | Canonical BF16 MoE V3 FFN path. | `=0` only to A/B the legacy BF16 MoE path. |
| `LUMEN_CUDA_BF16_AUTOTUNE` | ON | kill-switch | BF16 GEMM autotune selection. | `=0` to pin a fixed BF16 GEMM config. |
| `LUMEN_CUDA_GDN_REGISTER_RESIDENT` | ON (GDN models) | kill-switch | Register-resident GDN two-launch dispatch (+9.4% Q8, +10.3% Q4). | `=0` to A/B the legacy GDN dispatch. |
| `LUMEN_CUDA_MMV_BF16_OUTPUT_PROJ` | ON | kill-switch | BF16 output_proj llama.cpp-parity matvec; load-bearing for 0.902× llama.cpp. | `=0` only to A/B output_proj. |
| `LUMEN_CUDA_MMV_Q_DP4A` | ON | kill-switch | dp4a dense matvec (+7.1% Q8, +6.3% Q4). | `=0` to fall back to the scalar matvec. |
| `LUMEN_CUDA_MMV_Q_MOE_DP4A` | ON dense; OFF for MoE (model-aware) | kill-switch | MoE batched dp4a matvec (+11.7% Q4). | `=1` to force on for MoE experimentation. |
| `LUMEN_CUDA_MMV_Q_OUTPUT_PROJ` | ON (canonical) | kill-switch | Q8/Q4 output_proj llama.cpp-parity port (within noise floor). | `=0` to A/B output_proj. |
| `LUMEN_CUDA_FFN_FUSED_GLU` | model-aware (quant-dense uses dp4a fall-through) | kill-switch | Fused gate+up+SwiGLU FFN. | `=1`/`=0` to A/B the fused vs. split FFN. |
| `LUMEN_CUDA_MOE_BATCHED` | ON | kill-switch | Batched MoE FFN dispatch. | `=0` to A/B against per-expert dispatch. |
| `LUMEN_CUDA_MOE_BATCHED_V2` | ON (under `MOE_BATCHED`) | kill-switch | V2 batched MoE FFN variant. | `=0` to drop to V1 batching. |
| `LUMEN_CUDA_MOE_BATCHED_V3` | ON (under V2) | kill-switch | V3 batched MoE FFN variant. | `=0` to drop to V2 batching. |
| `LUMEN_CUDA_MOE_Q4_V3` | ON (canonical) | kill-switch | MoE Q4 V3 FFN path. | `=0` to A/B the legacy MoE Q4 path. |
| `LUMEN_CUDA_MOE_Q4_V3B` | ON (effective under V3) | kill-switch | MoE Q4 V3B refinement. | `=0` to drop the V3B refinement. |
| `LUMEN_CUDA_MOE_ROUTER_PARALLEL` | ON | kill-switch | Parallel top-K MoE router. | `=0` to A/B the serial router. |
| `LUMEN_CUDA_TOPK_MOE_FUSED` | ON | kill-switch | Fused sigmoid+top-K+renorm router (+6–8% all MoE quants). | `=0` to A/B the unfused router. |
| `LUMEN_CUDA_MOE_FUSED_NORM_ROUTER` | ON | kill-switch | Fused norm + router kernel. | `=0` to A/B the split norm/router. |
| `LUMEN_CUDA_MOE_BF16_NATIVE` | ON (MoE-bf16 raw path) | kill-switch | Native BF16 MoE (no F32 upcast). | `=0` to force the F32-upcast MoE path. |
| `LUMEN_CUDA_MOE_DOWN_TILED_F32ACT` | ON (under `MOE_GROUPED_TILED`) | kill-switch | F32-accumulated tiled MoE down-projection. | `=0` to A/B the down-projection accumulator. |
| `LUMEN_CUDA_GPU_SAMPLE` | ON | kill-switch | GPU-side argmax/sampling (avoids a DtoH per token). | `=0` to sample on the host. |
| `LUMEN_CUDA_PTX_CACHE` | ON | kill-switch | Cache compiled PTX between runs. | `=0` to force a cold recompile. |
| `LUMEN_CUDA_Q8_SPLIT` | ON (Q8-dense); OFF Q4/BF16/MoE | kill-switch | Raw+split layout for Q8_0 weights (+4.5% Q8 decode). | `=0` to A/B the packed Q8 layout. |
| `LUMEN_CUDA_Q8_SCALE_HW` | ON (Q8-dense) | kill-switch | Native `LDG.E.U16` scale fetch for Q8 matvec (+0.4% Q8 decode). | `=0` to A/B the scalar scale fetch. |
| `LUMEN_CUDA_OUTPUT_PROJ_SPLIT` | ON (Q8-dense); OFF else | kill-switch | Split-K layout for the ~1 GB output projection. | `=0` to A/B the packed output_proj. |
| `LUMEN_CUDA_OUTPUT_PROJ_NR` | `16` (Q8-dense) / `1` else | config | Rows-per-thread (NR) for output_proj; unrecognised → warn + `32`. | Tune output_proj occupancy on new GPUs. |
| `LUMEN_CUDA_SOA_LOCKED` | ON (Q4-dense); OFF else | kill-switch | Codegen-locked SoA Q4 matvec (word-load reorder; +30% path). | `=0` to A/B the AoS Q4 matvec. |
| `LUMEN_CUDA_Q4_SPLIT` | armed by `SOA_LOCKED` for Q4-dense (env A/B only) | kill-switch | Raw+split layout for Q4_0 weights (+9.0% Q4 decode); load-bearing under SoA-lock. | Leave as-is; the env read is A/B-only. |
| `LUMEN_CUDA_Q8_PROJ_MMQ` | ON for MoE, OFF for dense (model-aware) | kill-switch | Q8 projection via MMQ. | `=0`/`=1` to A/B MMQ vs. matvec projection. |
| `LUMEN_CUDA_SHARED_TILED` | ON (only `0/false/no` disables) | kill-switch | Shared-memory tiled path for the MoE shared expert. | `=0` to A/B the untiled shared expert. |
| `LUMEN_CUDA_MOE_GATE_UP_W10` | ON, Q8 MoE (only `0/false/no` disables) | kill-switch | Wide-M IMMA gate+up MoE prefill kernel (+9.30% Q8 MoE prefill, PRISTINE ×3); member of the 3.78× MoE-35B prefill combo. Engages only for Q8 experts. Quality-equivalent to the per-token path (int8 activation prequant — **not** byte-identical). No-op on dense. | `=0` to disable the W10 gate+up path. |
| `LUMEN_CUDA_MOE_GROUPED_TILED` | ON, MoE (only `0/false/no` disables) | kill-switch | Grouped-tiled MoE prefill GEMM; parent of the tiled-prefill stack (`MOE_DOWN_TILED_F32ACT`, `SHARED_TILED`). Quality-equivalent to the per-column path (grouped F32 reduction reorder — **not** byte-identical; x_sumsq-oracle + GQ-PRISTINE gated). No-op on dense. | `=0` to A/B the per-column MoE prefill. |
| `LUMEN_CUDA_MOE_PREFILL_BATCHED` | ON, MoE (only `0/false/no` disables) | kill-switch | Batched/grouped MoE prefill dispatch (single grouped GEMM vs the per-token loop); combo member. Quality-equivalent, **not** byte-identical. No-op on dense. | `=0` to A/B the per-token MoE prefill loop. |
| `LUMEN_CUDA_SHARED_FUSED_DECODE` | ON, MoE (only `0/false/no` disables) | kill-switch | Fused shared-expert FFN decode (batch=1-native fused GLU/down kernels, 6→3 launches/layer; +8.4% MoE-Q4 decode, byte-identical). No-op on dense. | `=0` to A/B the naive shared-expert decode path. |

## CUDA — MoE correctness defaults (auto-ON for MoE, no-op for dense)

These fix the MoE GDN decode-vs-prefill divergence that the 256-expert router
amplifies into garble. Each is **default-ON only for MoE** (gated on
`model_is_moe()`); for dense models the gate forces them off so dense output
stays byte-identical to history. Set any to `=0` to opt out on MoE.

| Variable | Default | Category | Effect | When to touch |
|---|---|---|---|---|
| `LUMEN_CUDA_GDN_AB_F16` | ON for MoE; dense byte-identical | kill-switch | Routes GDN `ssm_alpha`/`ssm_beta` through the same F16-cache GemmEx in decode + prefill so α/β are bit-identical across the two. | `=0` only to prove a decode-vs-prefill α/β divergence. |
| `LUMEN_CUDA_GDN_DECODE_VIA_PREFILL` | ON (all GDN classes) | kill-switch | Runs the whole MoE GDN decode recurrence through the prefill fused kernels at `T=1`, so GDN-decode == GDN-prefill by construction. | `=0` to isolate the decode-only GDN kernels. |
| `LUMEN_CUDA_GDN_CONVSTATE_PARITY` | ON for MoE (needs `VIA_PREFILL`) | kill-switch | Computes the decode GDN qkv projection via the prefill path at `batch=1` so the conv-ring `conv_state` bit-matches a true prefill. | `=0` to prove a conv-state divergence. |
| `LUMEN_CUDA_GDN_F64_ACCUM` | ON for MoE + dense-bf16; OFF dense-non-bf16 | kill-switch | F64 GDN delta-rule accumulator — the foundational decode-vs-prefill parity fix landing MoE arithmetic at greedy. | `=0` to measure the F32-accum error. |
| `LUMEN_CUDA_GDN_DECODE_MEGAKERNEL_F64` | ON for MoE + dense-bf16 | kill-switch | F64 accumulation inside the fused GDN decode megakernel. | `=0` to A/B the F32 megakernel. |
| `LUMEN_CUDA_GDN_PREFILL_F64` | `false` (F32) for MoE; non-MoE inherits `F64_ACCUM` | config | F64 GDN prefill accumulator (MoE keeps F32 for the floor win). | `=1` to force F64 prefill for a parity check. |
| `LUMEN_CUDA_GDN_AB_F32` | OFF | config | F32 A-tile / B-tile accumulators in the GDN α/β projection. | `=1` to A/B GDN α/β accumulator precision. |

## CUDA — config knobs & opt-in paths

| Variable | Default | Category | Effect | When to touch |
|---|---|---|---|---|
| `LUMEN_CUDA_DECODE_DELAY_US` | `0` (CLI) / `50` (server) | config | CPU sleep (µs) after `device.synchronize()` per decode step; serialises inter-step submission to close a GPU-scheduler timing race for MoE-Q4 server determinism. Unparseable → default; no upper clamp. | Raise on a server if MoE-Q4 determinism drifts; `0` on the deterministic CLI. |
| `LUMEN_CUDA_MAX_SEQ_LEN` | unset = model max | config | Cap KV-cache max sequence length. `=0` is treated as **no cap** (a 0-length KV cache previously faulted the GPU with an illegal address; guarded 2026-07-14); non-numeric ignored. | Set to bound KV memory for long-context models. |
| `LUMEN_CUDA_ATTN_PRECISE` | numeric, model-aware: `2` (P@V-F32) for MoE + dense ≤32-layer, `0` (legacy F16 WMMA) for 27B-other | config | Attention accumulation precision. | Force `2` when debugging attention numerics. |
| `LUMEN_CUDA_DECODE_TILED` | OFF (force-mode opt-in) | config | Force the tiled streaming-softmax decode kernel. | `=1` to exercise the tiled decode kernel. |
| `LUMEN_CUDA_DECODE_TILED_THRESHOLD` | `0` (tiled-always) | config | Token threshold above which the tiled kernel engages; `4294967295` forces the single-block path. | Tune the tiled/single-block crossover. |
| `LUMEN_CUDA_PREFILL_F32` | OFF | config | Force the F32 prefill path. | `=1` to debug F16 prefill drift. |
| `LUMEN_CUDA_SKIP_BF16_PROBE` | OFF | config | Skip the BF16 GemmEx capability probe at startup. | `=1` under sanitizers / to skip the probe. |
| `LUMEN_CUDA_PTX_CACHE_DIR` | `$LUMEN_CACHE_DIR/ptx` / XDG fallback | config | Override the compiled-PTX cache directory (empty-string guarded). | Point PTX cache at a writable path. |
| `LUMEN_CUDA_LEGACY_DEFAULTS` | OFF (canonical defaults ON) | config | Roll every CUDA default back to its legacy value. | `=1` to reproduce a pre-optimization baseline. |
| `LUMEN_CUDA_MOE_DECODE_F32` | OFF | config | F32 MoE decode path. Retained pending the open BF16-MoE decode RCA. | `=1` only for the BF16-MoE decode investigation. |
| `LUMEN_CUDA_MOE_DECODE_F32_FFN` | OFF | config | F32 MoE decode FFN path (pairs with `MOE_DECODE_F32`). Retained pending the open BF16-MoE decode RCA. | `=1` only for the BF16-MoE decode investigation. |

## CUDA — diagnostics

| Variable | Default | Category | Effect | When to touch |
|---|---|---|---|---|
| `LUMEN_CUDA_PROFILE` | OFF | diagnostic | Per-kernel CUDA-side profiling. | Primary CUDA perf field-triage. |
| `LUMEN_CUDA_VERBOSE` | OFF | diagnostic | Verbose CUDA backend logging (kernel-load failures). | Diagnose kernel-load / driver issues. |
| `LUMEN_CUDA_GDN_SUBSTAGE_TIMING` | OFF | diagnostic | Per-substage GDN timing. | Attribute time inside the GDN block. |
| `LUMEN_CUDA_ATTN_PRECISE_DBG` | OFF | diagnostic | One-shot attention-precision engagement print. | Confirm which attention path engaged. |
| `LUMEN_CUDA_FORCE_SCALAR_ATTN` | OFF | diagnostic | Force the scalar attention kernel (correctness reference). | Compare WMMA vs. scalar attention output. |
| `LUMEN_GRAPH_DIAGNOSTIC` | `false` | diagnostic | Emit CUDA-graph capture diagnostics. | The only lever to debug the crash-prone graph path. |

---

## Metal — production kill-switches (default-ON)

The Metal default-ON stack reverts wholesale via `LUMEN_METAL_DEFAULTS_OFF=1`;
individual switches also accept `=0`.

| Variable | Default | Category | Effect | When to touch |
|---|---|---|---|---|
| `LUMEN_METAL_DEFAULTS_OFF` | OFF (defaults ACTIVE) | kill-switch | Master switch: reverts all default-ON Metal optimizations to legacy. | `=1` to bisect the whole default-ON stack. |
| `LUMEN_METAL_CONCURRENT_ENCODER` | ON | kill-switch | Concurrent-encoder + 4-way QKV split (prefill). | `=0` to A/B the per-op encoder. |
| `LUMEN_METAL_GDN_CONCURRENT_ENCODER` | ON | kill-switch | GDN concurrent-encoder dispatch (prefill). | `=0` to A/B the per-op GDN encoder. |
| `LUMEN_METAL_CB_SPLIT` | ON (metal-R10) | kill-switch | AUTO per-token command-buffer split at full-attn island close (27B +2.4%, MoE-35B +5.6%); no-ops <73 encoders (e.g. 9B). Byte-identical. | `=0` to A/B the single-CB decode. |
| `LUMEN_METAL_FFN_DOWN_SPLITK` | `8` (dense, non-MoE) | kill-switch | FFN-down K-split factor (+1.89% Q8 prefill); bounds `{0,2,4,8}`. | Set `0` to disable, or `2`/`4` to A/B the split. |
| `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED` | ON | kill-switch | Fused FFN gate+up+SwiGLU kernel (+2.6%, Q8 prefill). | `=0` to A/B the unfused FFN. |
| `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_Q4` | ON | kill-switch | Q4 port of the fused gate+up+SwiGLU kernel. | `=0` to A/B unfused Q4 FFN. |
| `LUMEN_METAL_FFN_GATE_UP_SWIGLU_FUSED_BF16` | ON | kill-switch | BF16 port of the fused gate+up+SwiGLU kernel. | `=0` to A/B unfused BF16 FFN. |
| `LUMEN_METAL_BF16_GDN_QKV_GATE_PAIRED` | ON | kill-switch | Paired BF16 GDN QKV+gate dispatch (prefill win). | `=0` to A/B the unpaired dispatch. |
| `LUMEN_METAL_BF16_GDN_FULL_PREFILL_WARMUP` | ON (when PAIRED on) | kill-switch | Load-time full M=131 BF16 GDN warmup (steady-state effect). | `=0` to skip the load-time warmup. |
| `LUMEN_METAL_Q8_REPACKED` | ON | kill-switch | Runtime Q8 hot-weight repack (+6.89% prefill). | `=0` to A/B the un-repacked Q8 weights. |
| `LUMEN_METAL_Q8_REPACKED_FFN_DOWN` | ON | kill-switch | Q8 repack sub-gate for FFN-down. | `=0` to exclude FFN-down from repack. |
| `LUMEN_METAL_Q8_REPACKED_GATE_UP` | ON | kill-switch | Q8 repack sub-gate for gate+up. | `=0` to exclude gate+up from repack. |
| `LUMEN_METAL_Q8_GDN_QKVGATE_2STREAM` | ON (v0.5) | kill-switch | Per-thread 2-stream Q8 GDN qkv+gate matvec (+0.8% 27B-Q8 decode; byte-identical). | `=0` to A/B the single-stream matvec. |
| `LUMEN_METAL_MMAP_ONLY` | ON (wrapper auto-enables) | kill-switch | No-copy `newBufferWithBytesNoCopy` model load — **required** for BF16 dense / Q8 MoE / Q4 MoE on M3 Ultra 96 GB to avoid OOM at load. | `=0` only on high-memory boxes to force the copy loader. |
| `LUMEN_METAL_GDN_SSM_OUT_F32_BATCHED` | ON (opt-out) | kill-switch | F32 batched GDN ssm-out path. | `=0` to revert to the non-batched ssm-out. |
| `LUMEN_METAL_MOE_PREFILL_GROUPED` | ON | kill-switch | Grouped MoE prefill dispatch (`=0` selects Option B). | `=0` to A/B the alternate MoE prefill. |
| `LUMEN_METAL_MOE_ROUTER_PARALLEL` | ON | kill-switch | Parallel MoE router (`=0` → serial). The parallel and serial routers diverge at greedy near-ties: the `=0` token stream is coherent but **not byte-identical** to the default. | `=0` to A/B the serial router. |
| `LUMEN_METAL_MOE_GEMM_TILEMAP` | ON | kill-switch | Work-tile-map dispatch for the grouped MoE GEMM (byte-identical; only changes which TG computes which tile). | `=0` to A/B the legacy over-subscribed grid. |
| `LUMEN_METAL_MOE_GATHER_VEC4` | ON | kill-switch | Float4-vectorized grouped gather/scatter (byte-exact; needs `hidden_dim % 4 == 0`). | `=0` to A/B the scalar gather. |
| `LUMEN_METAL_MOE_ROUTE_SORT` | `atomic` | kill-switch | MoE route-sort strategy: `serial`/`0`, `par`/`1`, else atomic. | Set `serial`/`par` to A/B route-sort strategies. |
| `LUMEN_METAL_MOE_ROUTE_SORT_PAR` | unset (atomic) | kill-switch | Legacy MoE route-sort kill switch; `=0` forces serial (takes priority over `MOE_ROUTE_SORT`). | `=0` to force the serial route-sort. |

## Metal — config knobs & opt-in paths

| Variable | Default | Category | Effect | When to touch |
|---|---|---|---|---|
| `LUMEN_METAL_ATTN_PRECISE` | OFF (F16 tiled P@V, byte-identical) | config | `=2` selects exact-F32 scalar P@V attention (correctness field-triage). | `=2` to rule out a P@V precision defect. |
| `LUMEN_METAL_BF16_GATE_UP_NR` | `2` (TILE_M=32) | config | BF16 gate+up microtile geometry; clamps to `{1,2,4}` → `2`. | Tune BF16 gate+up prefill tiling. |
| `LUMEN_METAL_BF16_MMAP_ONLY` | OFF | config | BF16-specific alias of `LUMEN_METAL_MMAP_ONLY` (back-compat). | Use for BF16-only no-copy loading. |
| `LUMEN_METAL_DECODE_DELAY_US` | `0` (no-op; DET-001 fixed) | config | CPU sleep (µs) after the per-token `commit_and_wait()`; the non-determinism it mitigated is now fixed at the kernel level. | `=50` to re-introduce the pause when investigating scheduler timing. |
| `LUMEN_METAL_UNRETAINED_CMDBUFS` | OFF | config | Unretained command-buffer allocation (Apple-documented opt-in). | `=1` to reduce command-buffer retain traffic. |
| `LUMEN_METAL_MOE_ROUTER_TOPK_TGS` | `1` (no-op) | config | Threadgroup-size selector for the MoE top-K router. | Leave at `1`; a tuning knob for the router TGS. |
| `LUMEN_METAL_GPU_SAMPLER` | OFF | config | GPU-side sampler for Metal decode. | `=1` to sample on the GPU. |
| `LUMEN_METAL_GPU_SAMPLER_EXACT` | OFF | config | Exact-mode variant of the GPU sampler. | `=1` with `GPU_SAMPLER` for exact sampling. |
| `LUMEN_METAL_GPU_SAMPLER_QUIET` | OFF | config | Suppress GPU-sampler logging. | `=1` to quiet sampler diagnostics. |

## Metal — diagnostics & validators

| Variable | Default | Category | Effect | When to touch |
|---|---|---|---|---|
| `LUMEN_METAL_CONCURRENT_ENCODER_VALIDATE` | OFF | diagnostic | Re-emits the concurrent-encoder plan into a serial encoder for a byte-identity check. | Set to validate the default-ON concurrent encoder. |
| `LUMEN_METAL_GDN_CONCURRENT_ENCODER_VALIDATE` | OFF | diagnostic | Byte-identity validator for the GDN concurrent encoder. | Set to validate the default-ON GDN encoder. |
| `LUMEN_METAL_DECODE_PROFILE` | OFF | diagnostic | Per-section CPU-side GPU timing in decode. | Primary decode hot-section triage. |
| `LUMEN_METAL_DECODE_GPUTIME` | OFF | diagnostic | Per-token GPU-busy-vs-wall accounting in decode. | Primary decode GPU-time triage. |
| `LUMEN_METAL_PREFILL_GPUTIME` | OFF | diagnostic | Per-section GPU-busy accounting in prefill. | Prefill GPU-time triage. |
| `LUMEN_METAL_PROFILE` | OFF | diagnostic | Metal-side profiling. | General Metal perf triage. |
| `LUMEN_METAL_NAN_DUMP` | OFF | diagnostic | Dump tensors when a NaN is detected. | Set when hunting a NaN source. |

---

## Server (`lumen-server`)

| Variable | Default | Category | Effect | When to touch |
|---|---|---|---|---|
| `LUMEN_SERVER_DEBUG_MEM` | OFF (endpoint 404) | diagnostic | Exposes the memory-breakdown JSON endpoint + soak breakdown logging (uncached for live toggle). | Enable for long-run server memory triage. |
| `LUMEN_SERVER_PANIC_MAX` | `3` panics/window | config | Max engine panics in the window before the health circuit-breaker trips (`0` ⇒ unhealthy on first panic). | Tune recovery aggressiveness. |
| `LUMEN_SERVER_PANIC_WINDOW_SECS` | `60` s | config | Sliding-window length for the panic counter (`0` ⇒ degenerate window). | Tune the fail-fast window. |

## Diagnostics / dumps (general, off in production)

| Variable | Default | Category | Effect | When to touch |
|---|---|---|---|---|
| `LUMEN_MOE_PROBE` | OFF (byte-identical) | diagnostic | One-time GDNSTATE / decode-vs-prefill / router `eprintln` diagnostics across the CUDA MoE path. Genuine triage value for the recurring GDN decode-vs-prefill divergence class (still open for BF16 MoE). | Set when localising a CUDA MoE decode-vs-prefill divergence. |
| `LUMEN_XCHK` | OFF (byte-identical) | diagnostic | Cross-backend per-op forensic probe (sumsq trajectories + prompt-id echo). | Central Metal-vs-CUDA byte-identity instrument. |
| `LUMEN_XCHK2` | OFF (byte-identical) | diagnostic | Single-CB-path forensic variant of `XCHK`. | Forensic hook on the default lean decode path. |
| `LUMEN_SPEC_DUMP_IDS` | OFF (stderr only) | diagnostic | Dumps raw generated token ids pre-filter (spec-decode acceptance / prefix feed). | Set for raw-id correctness forensics. |
| `LUMEN_PREFILL_TIMING` | OFF | diagnostic | Prints prefill tok/s (server; MoE-prefill vs llama-bench pp). | Measure MoE prefill throughput. |
| `LUMEN_DUMP_EXPERTS` | unset (no-op) | diagnostic | Dump per-expert routing decisions (adds a DtoH sync when set). | Set for MoE routing triage. |
| `LUMEN_DUMP_GDN_L0_BIN` | unset (no writes) | diagnostic | Dump GDN layer-0 full-precision binaries. **Value is a directory path** (or `=all` for every GDN layer), *not* a boolean — set it to an existing writable directory, e.g. `/tmp/gdndump`. | Set to a dir for GDN correctness triage. |
| `LUMEN_DUMP_NORMED` | unset (no-op) | diagnostic | Dump post-RMSNorm activations. | Set to inspect normed activations. |

---

## Test-fixtures (test-only — not read by the shipped runtime)

These are read only by the test / bench / soak harnesses. They are on the
startup allowlist (so they never false-warn) but have **no effect on the
`lumen` / `lumen-server` runtime path**. Ignore them in production.

| Variable | Default | Effect (test harness) |
|---|---|---|
| `LUMEN_QWEN35_9B_PATH` | unset → test skips | Path to a Qwen3.5-9B `.lbc` for KV-resume + CLI matrix tests. |
| `LUMEN_QWEN35_9B_Q8` | unset → test skips | Q8_0 9B `.lbc` fixture for CLI + soak tests. |
| `LUMEN_QWEN35_9B_Q4` | unset → test skips | Q4_0 9B `.lbc` fixture for the CLI matrix test. |
| `LUMEN_QWEN35_9B_BF16` | unset → test skips | BF16 9B `.lbc` fixture for the gated CLI matrix test. |
| `LUMEN_CORR010_MODEL` | unset → test skips / default model | Model override for the CORR-010 KV-cache-equivalence test. |
| `LUMEN_AB_ITERATIONS` | `10000` | Paired A/B measured iteration count. |
| `LUMEN_AB_WARMUP` | `100` | Warmup iterations before A/B measurement. |
| `LUMEN_BENCH_ITERATIONS` | `10000` | Measured iteration count for the in-tree bench harness. |
| `LUMEN_BENCH_SCALE` | `tiny` | Synthetic bench workload scale. |
| `LUMEN_BENCH_TOKENS` | `50` | Token budget for bench-driven decode runs. |
| `LUMEN_BENCH_WARMUP` | `5` | Warmup iteration count before measurement. |
| `LUMEN_BASE_URL` | unset (built from bound port) | Override server base URL in client-side tests. |
| `LUMEN_SOAK_DURATION_SEC` | `300`/`1800` | Soak run duration. |
| `LUMEN_SOAK_WARMUP_SEC` | `300` s | Post-warmup window before soak-regression sampling. |
| `LUMEN_SOAK_OUT_DIR` | unset → default dir | Output directory for soak jsonl artifacts. |
| `LUMEN_SOAK_STACK_DUMP` | OFF | Enable jemalloc heap/stack snapshot dumps. |
| `LUMEN_SOAK_STACK_LEAKS` | OFF | Opt-in jemalloc leak report. |
| `LUMEN_SOAK_STACK_TICKS` | built-in cadence | Snapshot cadence (secs) for the soak leak harness. |
| `LUMEN_TEST_OPENAI_SDK` | unset → test skipped | Gate that runs the real OpenAI-SDK round-trip test. |
