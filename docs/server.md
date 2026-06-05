# HTTP Server (`lumen-server`)

The serving front-end for Lumen — **LLM inference in Rust, for Apple Silicon and NVIDIA CUDA**.

`lumen-server` ships as both a library crate and an opt-in standalone binary. It exposes an axum-based router with OpenAI and Anthropic wire formats and SSE streaming. Wire protocols, SSE, and the routing layer are model-agnostic; the tool-call parser is template-driven and v1 ships the Qwen3.5 marker pattern, with additional templates pluggable as future model families ship.

For concurrent-client deployments use `lumen-server` (not repeated `lumen run`) — see [`docs/production.md`](production.md) for rationale.

## Run the standalone binary

```bash
# Pre-download a model (one-time, ~10 GB for Qwen3.5-9B Q8_0)
lumen pull qwen3.5-9b:q8_0

# Boot the server (Metal on macOS, CUDA on Linux with --features cuda)
cargo run --release --bin lumen-server --features bin -- \
  --model qwen3.5-9b --quant q8_0 --port 8000

# Or against an explicit LBC path
cargo run --release --bin lumen-server --features bin -- \
  --model /path/to/qwen3-5-9b-Q8_0.lbc

# Try it
curl http://localhost:8000/v1/models
```

The bin is gated behind the `bin` Cargo feature so library embedders that wire their own tokenizer / backend keep the `lumen-server` dep graph minimal. `lumen-server --help` lists all flags.

## Endpoints

```
POST /v1/chat/completions   # OpenAI-compatible, SSE streaming
POST /v1/completions        # OpenAI-compatible
POST /v1/messages           # Anthropic-compatible, SSE streaming
GET  /v1/models             # Model list
```

Both wire formats support SSE streaming. Tool-call parsing is template-driven: v1 (current) ships the Qwen3.5 `<tool_call>` / `</tool_call>` marker pattern with a streaming parser that uses partial-marker hold-back; additional templates are registered as new model families ship. Reference embedder: [`crates/lumen-server/tests/server_integration.rs`](../crates/lumen-server/tests/server_integration.rs). Reference binary: [`crates/lumen-server/src/bin/lumen-server.rs`](../crates/lumen-server/src/bin/lumen-server.rs).

## Sampling & reproducibility

Sampling defaults are resolved per request from the wire payload:

| Field | Default when omitted | Notes |
|---|---|---|
| `temperature` | `0.7` | `0` = greedy (argmax) |
| `seed` | **fresh random per request** | so identical requests **vary** |
| `repetition_penalty` | `1.05` (server-internal) | not exposed in the OpenAI/Anthropic request schema |

The server follows the OpenAI / llama.cpp convention: **with no `seed`, every request samples from a fresh random seed, so the same prompt returns different text each time.** For reproducible output, pass an explicit `seed` (OpenAI `/v1/chat/completions` and `/v1/completions`):

```bash
# Reproducible: same seed + params => identical output. Repeat to confirm.
curl -fsS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Hi"}],"seed":42,"temperature":0.7}'
```

- `temperature: 0` is greedy/deterministic **regardless of seed**.
- The Anthropic `/v1/messages` schema has no `seed` field, so it always uses a fresh random seed (matching the upstream Anthropic API).
- These are two independent properties: the **kernels** are byte-deterministic for fixed inputs (same seed + params ⇒ same tokens), while **sampling output varies by default** because the seed is randomized per request. Pin the `seed` to combine both into reproducible output.

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
