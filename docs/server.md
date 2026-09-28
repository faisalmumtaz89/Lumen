# HTTP Server (`lumen-server`)

The serving front-end for Lumen — **LLM inference in Rust, for Apple Silicon and NVIDIA CUDA**.

`lumen-server` ships as both a library crate and an opt-in standalone binary. It exposes an axum-based router with OpenAI and Anthropic wire formats and SSE streaming. Wire protocols, SSE, and the routing layer are model-agnostic; the tool-call parser is template-driven and v1 ships the Qwen3.5 marker pattern, with additional templates pluggable as future model families ship.

For concurrent-client deployments use `lumen-server` (not repeated `lumen run`) — see [`docs/production.md`](production.md) for rationale.

## Run the standalone binary

```bash
# Pre-download a model (one-time, ~10 GB for Qwen3.5-9B Q8_0)
lumen pull qwen3.5-9b:q8_0

# Boot the server. The model is positional (model:quant); port defaults to 8000
# and the backend auto-detects (Metal on macOS, CUDA on Linux).
lumen-server qwen3.5-9b:q8_0

# Equivalent explicit-flag form (and how to override port / backend)
lumen-server --model qwen3.5-9b --quant q8_0 --port 8000

# Try it
curl http://localhost:8000/v1/models
```

Building from source instead of the prebuilt binary? Prefix with
`cargo run --release --bin lumen-server --features bin --`, e.g.
`cargo run --release --bin lumen-server --features bin -- qwen3.5-9b:q8_0`
(append `,cuda` to `--features` on NVIDIA). An explicit `.lbc` path also works in
place of `model:quant`: `lumen-server /path/to/qwen3-5-9b-Q8_0.lbc`.

The bin is gated behind the `bin` Cargo feature so library embedders that wire their own tokenizer / backend keep the `lumen-server` dep graph minimal. `lumen-server --help` lists all flags.

## Endpoints

```text
POST /v1/chat/completions   # OpenAI-compatible, SSE streaming
POST /v1/completions        # OpenAI-compatible
POST /v1/messages           # Anthropic-compatible, SSE streaming
GET  /v1/models             # Model list
POST /v1/images/generations # Text to image (`--features image` builds with
                            # LUMEN_IMAGE_LBI and LUMEN_IMAGE_CKPT set)
```

The image endpoint — conversion, limits, request shape and how it shares a device with the text model — is described in [image-generation.md](image-generation.md).

`/v1/messages` ignores fields it does not use, such as `metadata`, so Anthropic clients that attach them are served. A structured-output request (`output_config.format`) is refused with a 400, since decoding cannot be constrained to a schema. `/v1/chat/completions` and `/v1/completions` refuse an unknown field with a 400.

A system message after the start of the conversation (`role: "system"` inside `/v1/messages` `messages`, or any but the first message of `/v1/chat/completions`) stays where it was sent, so its instructions apply from that point on and the prompt before it is unchanged. It renders the way the model's chat template renders it; a template that only accepts a system message at the start (Qwen3.5, Qwen3.8) gets it as a system turn in that template's own framing. Lumen first renders a probe conversation with the template and refuses the request with a 400 when the probe shows the template moving such a message to the start, dropping it or rendering it as another role.

`/v1/completions` takes `prompt` as a string, an array of strings (concatenated), or an array of token ids. Token ids reach the model unchanged, with no tokenization and no special tokens added, so a client can send the exact ids it measured elsewhere. An id at or above the model's vocabulary size, an array mixing strings with ids or holding anything else, or a prompt that resolves to no tokens, is refused with a 400.

Both wire formats support SSE streaming. Tool-call parsing is template-driven: v1 (current) ships the Qwen3.5 `<tool_call>` / `</tool_call>` marker pattern with a streaming parser that uses partial-marker hold-back; additional templates are registered as new model families ship. Reference embedder: [`crates/lumen-server/tests/server_integration.rs`](../crates/lumen-server/tests/server_integration.rs). Reference binary: [`crates/lumen-server/src/bin/lumen-server.rs`](../crates/lumen-server/src/bin/lumen-server.rs).

## Sampling & reproducibility

Sampling defaults are resolved per request from the wire payload:

| Field | Default when omitted | Notes |
|---|---|---|
| `temperature` | `0.7` | `0` = greedy (argmax) |
| `seed` | **fresh random per request** | so identical requests **vary** |
| `ignore_eos` | `false` | `true` keeps decoding past the model's end-of-sequence tokens, which then render nothing; `max_tokens`, `stop` strings and the context limit still end the answer. OpenAI `/v1/chat/completions` and `/v1/completions` |
| `repetition_penalty` | `1.05` dense / `1.03` MoE (model-aware; `LUMEN_REPETITION_PENALTY` overrides) | server-internal, not exposed in the OpenAI/Anthropic request schema; source of truth `runtime_defaults::repetition_penalty_default` (MoE capped at 1.03 — 1.05+ corrupts MoE arithmetic) |

The server follows the OpenAI / llama.cpp convention: **with no `seed`, every request samples from a fresh random seed, so the same prompt returns different text each time.** For reproducible output, pass an explicit `seed` (OpenAI `/v1/chat/completions` and `/v1/completions`):

```bash
# Reproducible: same seed + params => identical output. Repeat to confirm.
curl -fsS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Hi"}],"seed":42,"temperature":0.7}'
```

- `temperature: 0` is greedy/deterministic **regardless of seed**.
- The Anthropic `/v1/messages` schema has no `seed` field, so it always uses a fresh random seed (matching the upstream Anthropic API).
- These are two independent properties: the **kernels** are byte-deterministic for fixed inputs (same seed + params ⇒ same tokens; for an NVFP4/FP8 artifact on CUDA, while its measured GEMM plan table stays the same and the request does not continue the one the server ran before it, prompt and reply — see [troubleshooting.md](troubleshooting.md)), while **sampling output varies by default** because the seed is randomized per request. Pin the `seed` to combine both into reproducible output.

## Reasoning / extended thinking

The Qwen3.5 family supports an optional `<think>...</think>` reasoning trace. Lumen exposes a per-request reasoning toggle across all surfaces. **Thinking is OFF by default** (the assistant prompt opens a closed empty-think block, `<think>\n\n</think>\n\n`, so the model answers directly). When enabled, the open `<think>\n` tail is emitted, the model produces a reasoning trace, and Lumen routes that trace to a separate field — it is never mixed into the answer text.

The reasoning budget is **separate from `max_tokens`** (industry-convergent with Anthropic `thinking.budget_tokens` / Gemini `thinking_budget`), so a long reasoning trace never starves the answer. Default reasoning budget is `2048` tokens (`runtime_defaults::chat_reasoning_budget_default`).

**Precedence** (resolved by `runtime_defaults::resolve_enable_thinking`): explicit per-request field → `LUMEN_CHAT_ENABLE_THINKING` env override → process default (OFF).

| Surface | Enable thinking | Separate budget | Reasoning output |
|---|---|---|---|
| OpenAI `/v1/chat/completions` | top-level `enable_thinking: true`, or vLLM/SGLang-compatible `chat_template_kwargs: {"enable_thinking": true}` (top-level wins), or `reasoning_effort` other than `none` | `reasoning_budget` | streamed as `delta.reasoning_content`; non-stream as `message.reasoning_content` (omitted when empty) |
| Anthropic `/v1/messages` | `thinking: {"type": "enabled"}` or `{"type": "adaptive"}` (`"disabled"` = off, absent = the default; any other type is refused with a 400) | `thinking.budget_tokens` → `reasoning_budget` | a `{"type": "thinking", "thinking": ...}` content block; streamed via `thinking_delta` |
| CLI `lumen run` | `--think` (and `--no-think` forces off, overriding the env var) | — | reasoning printed to stderr, answer to stdout |

```bash
# OpenAI: enable reasoning, cap the trace at 1024 tokens, answer budget separate
curl -fsS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Plan a 3-day trip"}],"enable_thinking":true,"reasoning_budget":1024,"max_tokens":512}'

# vLLM-compatible form (chat_template_kwargs)
#   "chat_template_kwargs": {"enable_thinking": true}

# Anthropic: enable extended thinking with a 1024-token budget
curl -fsS http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-9b","max_tokens":512,"messages":[{"role":"user","content":"Plan a 3-day trip"}],"thinking":{"type":"enabled","budget_tokens":1024}}'
```

A reasoning effort sets how much the model reasons while thinking is on: `output_config.effort` on `/v1/messages` and `reasoning_effort` on `/v1/chat/completions`. The levels both APIs share map the same way: `low` and `medium` reach the chat template as its `reasoning_effort` (Qwen3.8 instructs brief reasoning at `low`), while `high`, `xhigh`, `max` and an absent effort keep the template's default (`xhigh` on Qwen3.8). OpenAI's `minimal` runs as `low`. On `/v1/chat/completions` an effort also asks for reasoning, and `none` turns it off, unless `enable_thinking` or `chat_template_kwargs.enable_thinking` is given. Any other value is refused with a 400.

`LUMEN_CHAT_ENABLE_THINKING=1` (accepts `1`/`true`/`yes`/`on`; `0`/`false`/`no`/`off` for off) flips the default for requests that do not specify the toggle.

## Fault injection (test builds)

`cargo build -p lumen-server --features bin,fault-injection` compiles two hooks into the engine worker for
recovery and timing tests; a build without the feature has none of this code.

| Variable | Effect |
|---|---|
| `LUMEN_FAULT_PANIC_AT` | `prefill` or `decode:<n>`: the worker panics at that point of the first job that reaches it (`decode:<n>` is before the token after `n` generated ones), once per process. The client gets the supervisor's error, `engine recovered from panic: …` — HTTP 500 for a non-streaming request, an `error` event on a stream — and the next request is served. A malformed value is a startup error. |
| `LUMEN_FAULT_PERTURB_US` | `<max>`: every job sleeps a random `0..=max` microseconds before its prefill and before every decode step. |

## Embed as a library

Custom embedders own the tokenizer and weight provider; the runtime owns the GPU.

```rust
use lumen_server::{build_router, EngineWorker};

let handle = EngineWorker::spawn(
    runtime_config,
    hyperparams,
    backend,
    weights,
    tokenizer,
    model_info,
    /* inbox_size */ 32,
);
let router = build_router(handle);
axum::serve(listener, router).await?;
```

`EngineWorker::spawn` signature: `(config, hyperparams, backend, weights, tokenizer, model_info, inbox_size) -> EngineHandle` ([`crates/lumen-server/src/engine.rs`](../crates/lumen-server/src/engine.rs)). `build_router(engine: EngineHandle)` returns the configured axum `Router` ([`crates/lumen-server/src/router.rs`](../crates/lumen-server/src/router.rs)). `AppState` is constructed internally by `build_router` from the handle — embedders pass the handle directly.

## Known limitations

- **Authorization / CORS / per-request timeout** are not implemented; deploy behind a reverse proxy that enforces auth, CORS, and request deadlines.
- **Mid-stream client disconnect** can wedge the engine worker. Pending fix; work around with a reverse-proxy that buffers SSE responses.
