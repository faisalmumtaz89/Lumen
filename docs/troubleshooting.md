# Troubleshooting

## `lumen run` hangs at "Loading"

Most common cause: the LBC is not yet in OS page cache and the disk read is dominating. Pre-warm at service start:

```bash
cat /path/to/model.lbc > /dev/null
```

A 9.5 GB Q8 dense-9B model takes ~86 s with warm OS cache and up to ~95 s cold-cold. Multi-GB MoE models take longer (~99-102 s warm, ~136-172 s cold-cold for BF16). See [`bench/RESULTS.md`](../bench/RESULTS.md) § "Cold-start TTFT" for the full table.

## OOM during model load

- **BF16 MoE-30B-A3B**: peak 72.4 GB on a 80 GB A100 — requires a dedicated GPU, no co-tenant workloads.
- **Metal BF16 dense / Q8 MoE / Q4 MoE**: set `LUMEN_METAL_MMAP_ONLY=1`. The 96 GB residency budget on M3 Ultra cannot fit the LBC AND a full eager allocation; mmap-only is the documented operating mode.
- **CUDA + concurrent process**: a peer consuming > 5 GB can race `cuMemAlloc` and cause OOM mid-upload. Reserve the GPU.

## Generated text loops on long prompts

PURE-greedy (`--temperature 0` + no repetition penalty) deterministically loops beyond ~512 tokens on all 4 quants. Use one of:

```bash
--temperature 0.7                        # sample instead of greedy
# OR (DENSE models only)
--repetition-penalty 1.05 --repeat-last-n 64    # cap repetition explicitly
```

When `--repetition-penalty` is omitted, the server/CLI apply a **model-aware** default (1.05 dense / **1.03 MoE**, resolved by `runtime_defaults::repetition_penalty_default`). The `1.05` recommendation above is for **DENSE models only** — MoE (Qwen3.5-MoE class) must stay **≤ 1.03**, because a penalty of 1.05+ corrupts MoE arithmetic ("17 × 20 = … = 39"). On MoE, leave `--repetition-penalty` unset so the 1.03 default applies.

## Multilingual / long-form prompt returns empty or truncated output

For v1's Qwen3.5 family the chat template opens a `<think>...</think>` block (per the Qwen3.5 spec). With `--max-tokens 256` or lower the full budget may be consumed inside `<think>...</think>` before the model produces the answer in the target language. Use `--max-tokens 512` minimum for multilingual prompts. Future model families may register chat templates with different reasoning blocks; tune `--max-tokens` accordingly.

## BF16 first-token output differs across runs

The BF16 mmvf kernel produces different first-token argmax at different KV-cache layout sizes per deployment.

## Decode output differs across identical runs

**Most common cause — no seed pinned.** By default the server and `lumen run` sample with a **random seed per request/run** (the OpenAI/llama.cpp convention), so the same prompt returns different text each time. This is expected, not a bug. To reproduce a specific output:

- **Server**: pass an explicit `"seed": <n>` (OpenAI `/v1/chat/completions`, `/v1/completions`) — same seed + params ⇒ identical output. (Anthropic `/v1/messages` has no seed field, so it is always randomized.)
- **CLI**: pass `--seed <n>`, or `--temperature 0` for greedy/argmax (deterministic regardless of seed).

**At a fixed seed, the kernels are byte-deterministic** on both backends:

- **Metal** — the GPU races that once broke this are fixed in the kernels. No knob is required; `LUMEN_METAL_DECODE_DELAY_US` defaults to `0`.
- **CUDA** — `lumen-server` applies `LUMEN_CUDA_DECODE_DELAY_US=50` automatically to serialize inter-step decode submission for MoE Q4; `lumen run` (CLI) defaults to `0` and is deterministic without it. If you see run-to-run variation on a CUDA server *with a pinned seed*, confirm the delay has not been overridden to `0`.

## Concurrent `lumen run` invocations time out

A 16-client concurrent burst against the per-process CLI measured 82.4 % timeout rate. The CLI cold-loads weights every invocation; the cold-start contention dominates. Use `lumen-server` for any C ≥ 2 deployment.

## `lumen-server` SSE stream wedges after a client disconnect

Mid-stream client disconnect can wedge the engine worker that buffers SSE responses and absorbs disconnect signals before they reach `lumen-server`. Fix pending.

## "Unsupported quantization" error

Run `lumen models` to see registry-declared cells. All three Qwen3.5-9B quants (Q8_0, Q4_0, BF16) are directly downloadable from the registry — use `lumen pull qwen3.5-9b:<quant>`. Avoid deriving Q4_0 via `lumen convert --requant q4_0` from a Q8_0 source: the requant route double-quantizes and is quality-broken on the CUDA target (see model_registry.toml notes); the registry's direct Q4_0 is the supported path.

## "Unsupported architecture" error at conversion

The converter currently accepts the v1 architecture set (`qwen35` and `qwen35moe`); architectures outside the v1 set are rejected at conversion because they have not yet been gated end-to-end on this runtime. Additional model families are planned — see [`docs/support.md`](support.md) for the live verified-against-llama.cpp matrix.

## `LBC_VERSION` mismatch

`LBC_VERSION = 4` is current. The reader rejects newer-than-current with `UnsupportedVersion`; backward-compat for v1 / v2 is in the code path but unverified at runtime. Policy: rebuild LBCs after major Lumen upgrades via `lumen convert` or `lumen pull --quant <scheme>`.
