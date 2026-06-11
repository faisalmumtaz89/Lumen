//! Wire-format encoders.
//!
//! Each submodule owns the request DTO, the SSE state machine, and the
//! non-streaming response shape for one external API.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::error::ServerError;

pub mod anthropic;
pub mod openai;

/// Validate that a message `content` value is a shape both wire surfaces
/// accept, then flatten it to prompt text — the SINGLE content-parts
/// flattener routed through by BOTH OpenAI and Anthropic (replacing the two
/// divergent `content_to_string` copies that recognized different key sets).
///
/// Accepted shapes (mirrors OpenAI's `string | array`):
/// - `String` → returned verbatim.
/// - `Null` → empty string (an absent/optional content field).
/// - `Array` → each element flattened and concatenated: a bare string element
///   is appended as-is; a content-part object contributes ONLY its `text`
///   field (the single recognized key). Any other element kind (number, bool,
///   nested array, object without `text`) contributes nothing.
///
/// ROBUST-007 numeric-type-guard: a bare number/bool (or any non
/// string/array/null scalar) at the top level is REJECTED as a 400 rather
/// than silently coerced via `Value::to_string`. `param` localizes the
/// offending field for the error envelope. Applied identically on both
/// surfaces so a number `content` 400s on `/v1/messages` exactly as it does
/// on `/v1/chat/completions`.
///
/// NOTE on the dropped Anthropic `content` fallback: the prior Anthropic copy
/// ALSO accepted `{"content": "..."}` content-part objects, so the same
/// content array yielded different prompt text per surface. We drop that
/// fallback for byte-parity — content-part objects use `text`. Tool-result
/// blocks (`{type:"tool_result", content:...}`) are handled by the dedicated
/// tool-turn renderer (see `render_tool_turns`), not by this text flattener.
pub(crate) fn flatten_content(content: &Value, param: &str) -> Result<String, ServerError> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(arr) => {
            let mut out = String::new();
            for piece in arr {
                if let Some(s) = piece.as_str() {
                    out.push_str(s);
                } else if let Some(obj) = piece.as_object() {
                    if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                        out.push_str(text);
                    }
                }
            }
            Ok(out)
        }
        // ROBUST-007: a bare number/bool/etc. is not a valid content value.
        _ => Err(ServerError::bad_request_field(
            "message 'content' must be a string or a content-parts array",
            param,
            "invalid_type",
        )),
    }
}

/// Per-request random seed for sampling when the client does not supply one.
///
/// An OpenAI-/Anthropic-compatible endpoint returns *varied* output by default
/// (reproducibility is opt-in via an explicit `seed`), so an omitted seed must
/// resolve to a fresh value per request rather than a fixed constant.
///
/// A monotonic counter guarantees every request in this process gets a distinct
/// seed — even under concurrent same-nanosecond bursts, which the wall clock
/// alone cannot — and a one-time wall-clock offset makes the sequence differ
/// across process restarts. No bit-mixing is done here: the seed is avalanched
/// downstream by `Xorshift64::new` (`lumen_runtime::sampling`), so distinct
/// inputs are sufficient for distinct, well-separated RNG streams.
pub(crate) fn next_random_seed() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    static START: OnceLock<u64> = OnceLock::new();
    let start = *START.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    });
    start.wrapping_add(COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Server-internal repetition penalty, model-aware. When the operator does
/// not set `LUMEN_REPETITION_PENALTY` explicitly, the default is resolved by
/// [`lumen_runtime::runtime_defaults::repetition_penalty_default`]: `1.03`
/// for MoE (Qwen3.5-MoE-35B-A3B class, all quants), `1.05` for dense / unset.
///
/// The historical 1.08/1.10 MoE band-aid is gone; the root cause was fixed in
/// the GDN decode path (the F64 delta-rule accumulator + the decode-vs-prefill
/// parity stack, all default-ON for MoE), so the math near-tie lands at greedy
/// without a heavy penalty. The MoE value is now CAPPED at 1.03: a penalty of
/// 1.05 or higher penalizes legitimate digit repetition and CORRUPTS MoE
/// arithmetic (the matrix-proven "17 x 20 = … = 39" at 1.05), while 1.03 is the
/// floor that keeps the F64-fixed math correct AND tames MoE long-form
/// repetition. Dense keeps 1.05 (no GDN recurrence, arithmetic unaffected).
/// See `repetition_penalty_default` for the full rationale and the lever-sweep
/// evidence — it is the single source of truth for this value.
///
/// The env var `LUMEN_REPETITION_PENALTY=<f32>` still overrides this default
/// (e.g. for diagnostics or to restore pure-greedy with `=1.0`).
pub(crate) fn diag_repetition_penalty() -> f32 {
    std::env::var("LUMEN_REPETITION_PENALTY")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or_else(lumen_runtime::runtime_defaults::repetition_penalty_default)
}

/// Server-internal frequency penalty (count-based: `logit[t] -= freq * count[t]`).
/// Complements `diag_repetition_penalty`: the repetition penalty floor is kept low
/// (1.03 MoE) so short arithmetic isn't corrupted, but that leaves q8/q4 long-form
/// (verylong) prone to repetition/loops; the count-scaled frequency penalty damps
/// those without touching short generations. Default resolved by
/// `runtime_defaults::frequency_penalty_default` (0.0 = no-op until the GQ sweep
/// fixes the MoE value). `LUMEN_FREQUENCY_PENALTY=<f32>` overrides (`=0.0` no-op).
///
/// Delegates to [`runtime_defaults::frequency_penalty_resolved`] so the
/// `LUMEN_FREQUENCY_PENALTY` env is read in exactly ONE place and the CLI
/// (when `--frequency-penalty` is absent) honours it identically.
pub(crate) fn diag_frequency_penalty() -> f32 {
    lumen_runtime::runtime_defaults::frequency_penalty_resolved()
}

/// DIAGNOSTIC (default None = full-history window, production-identical):
/// server-internal repeat-penalty window. Overridable via
/// `LUMEN_REPEAT_LAST_N=<usize>` to probe whether a finite recent-window
/// penalty (llama.cpp default 64) changes the q8 loop behaviour.
///
/// Delegates to [`runtime_defaults::repeat_last_n_resolved`] so the
/// `LUMEN_REPEAT_LAST_N` env is read in exactly ONE place and the CLI
/// (when `--repeat-last-n` is absent) honours it identically.
pub(crate) fn diag_repeat_last_n() -> Option<usize> {
    lumen_runtime::runtime_defaults::repeat_last_n_resolved()
}

/// Resolves the greedy anti-degeneration guard flag for a server request.
///
/// Delegates to [`runtime_defaults::anti_restate_default`] (ON for MoE, OFF
/// for dense, `LUMEN_ANTI_RESTATE=0/1` override). Centralised here so the
/// three wire constructors (OpenAI chat + completions, Anthropic messages)
/// stay in sync.
pub(crate) fn diag_anti_restate() -> bool {
    lumen_runtime::runtime_defaults::anti_restate_default()
}

/// Resolves whether chat "thinking" (reasoning trace) is enabled for a server
/// request. The SINGLE server-side entry point — both wire formats (OpenAI
/// `ChatCompletionRequest::resolve_thinking`, Anthropic
/// `MessagesRequest::resolve_thinking`) route through here so the OpenAI and
/// Anthropic surfaces cannot diverge.
///
/// Thin delegate to the cross-crate canonical
/// [`lumen_runtime::runtime_defaults::resolve_enable_thinking`] (precedence:
/// per-request field → `LUMEN_CHAT_ENABLE_THINKING` env override → default
/// `false`). The logic lives in `lumen-runtime` so the CLI — which cannot
/// depend on `lumen-server` — shares the exact same implementation; this
/// wrapper just keeps the server's wire layer consistent with the
/// `diag_*`-resolver pattern above (all server-internal request defaults
/// resolved in one module).
pub(crate) fn resolve_enable_thinking(per_request: Option<bool>) -> bool {
    lumen_runtime::runtime_defaults::resolve_enable_thinking(per_request)
}

/// Monotonic per-process sequence used to keep response `id`s unique even when
/// several requests share the same `created`/clock value (sub-second burst).
pub(crate) fn next_response_seq() -> u64 {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed)
}

/// Synchronous oversize-prompt guard, shared by every `into_job`.
///
/// Returns a 400 `context_length_exceeded` BEFORE the handler opens the 200 OK
/// / SSE stream when the tokenized prompt is longer than the model's context
/// window. This fixes the streaming surface's success-then-error-frame
/// behaviour (the worker's backstop guard at `engine.rs::run_job` can only
/// emit a mid-stream error after headers are already sent) and turns the
/// non-streaming 500 into a clean 400.
///
/// The message mirrors the worker guard's format byte-for-byte (same
/// `"prompt is N tokens but server max_seq_len is M; ..."` text) so a client
/// sees an identical body whichever guard fires; the wire `param`/`code`
/// (`context_length_exceeded`) come from the shared classifier. The worker
/// guard is retained as a backstop for any path that bypasses `into_job`.
///
/// `context_length == 0` is treated as "unknown / unconfigured" and skips the
/// check (defensive: a misconfigured 0 must never reject every request).
pub(crate) fn check_prompt_length(
    prompt_tokens: usize,
    context_length: usize,
) -> Result<(), ServerError> {
    if context_length > 0 && prompt_tokens > context_length {
        return Err(ServerError::classify_runtime(format!(
            "prompt is {prompt_tokens} tokens but server max_seq_len is {context_length}; \
             reduce prompt or restart with a larger --context-len",
        )));
    }
    Ok(())
}

/// CLI-parity zero-normalization for the additive penalties
/// (presence_penalty / frequency_penalty). The CLI maps an explicit
/// `--frequency-penalty 0` / `--presence-penalty 0` to `None` (a no-op) at
/// `run.rs:357,368`; the wire surfaces must do the same so an all-zero
/// request stays byte-identical to the no-penalty default path. A non-finite
/// value (NaN/inf) also normalizes to `None` so a junk float can never reach
/// the sampler. `None` (field omitted) passes through unchanged.
pub(crate) fn normalize_zero_penalty(v: Option<f32>) -> Option<f32> {
    v.filter(|p| p.is_finite() && *p != 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn next_random_seed_unique_across_threads() {
        // The counter's raison d'être: concurrent callers must never share a
        // seed. Exercise the atomic RMW under real contention — 8 threads x 20k.
        use std::thread;
        let (threads, per) = (8usize, 20_000usize);
        let handles: Vec<_> = (0..threads)
            .map(|_| thread::spawn(move || (0..per).map(|_| next_random_seed()).collect::<Vec<u64>>()))
            .collect();
        let mut seen = HashSet::with_capacity(threads * per);
        for h in handles {
            for s in h.join().unwrap() {
                assert!(seen.insert(s), "duplicate seed across threads");
            }
        }
        assert_eq!(seen.len(), threads * per);
    }

    #[test]
    fn next_response_seq_is_strictly_unique() {
        // Response ids must never collide even under sub-second concurrent burst.
        let n = 50_000;
        let mut seen = HashSet::with_capacity(n);
        for _ in 0..n {
            assert!(seen.insert(next_response_seq()), "duplicate response seq");
        }
        assert_eq!(seen.len(), n);
    }

    // ---- F5: shared zero-normalization (CLI parity) ----

    #[test]
    fn normalize_zero_penalty_matches_cli() {
        assert_eq!(normalize_zero_penalty(None), None, "omitted -> None");
        assert_eq!(normalize_zero_penalty(Some(0.0)), None, "explicit 0 -> None (CLI parity)");
        assert_eq!(normalize_zero_penalty(Some(-0.0)), None, "negative-zero -> None");
        assert_eq!(normalize_zero_penalty(Some(0.7)), Some(0.7), "non-zero passes through");
        assert_eq!(normalize_zero_penalty(Some(f32::NAN)), None, "NaN -> None (junk guard)");
        assert_eq!(normalize_zero_penalty(Some(f32::INFINITY)), None, "inf -> None");
    }

    // ---- F16(a): runtime-error classifier ----

    #[test]
    fn classify_runtime_oversize_sentinel_is_400() {
        // The proactive prompt-length guard's message.
        let e = ServerError::classify_runtime(
            "prompt is 9000 tokens but server max_seq_len is 8192; reduce prompt",
        );
        match e {
            ServerError::BadRequest { code, .. } => {
                assert_eq!(code.as_deref(), Some("context_length_exceeded"));
            }
            other => panic!("expected 400 BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn classify_runtime_kv_overflow_sentinel_is_400() {
        // The KV-overflow formatter's message ("would exceed max_seq_len").
        let e = ServerError::classify_runtime("decode: token would exceed max_seq_len 8192");
        assert!(matches!(
            e,
            ServerError::BadRequest { ref code, .. } if code.as_deref() == Some("context_length_exceeded")
        ));
    }

    #[test]
    fn classify_runtime_empty_prompt_is_400() {
        let e = ServerError::classify_runtime("prompt is empty");
        assert!(matches!(
            e,
            ServerError::BadRequest { ref code, .. } if code.as_deref() == Some("empty_prompt")
        ));
    }

    #[test]
    fn classify_runtime_compute_error_stays_500() {
        // A genuine compute/IO failure must NOT be downgraded to 400.
        let e = ServerError::classify_runtime("compute: matmul kernel returned NaN");
        assert!(matches!(e, ServerError::Runtime(_)), "compute error stays Runtime (500)");
    }

    #[test]
    fn check_prompt_length_guard() {
        assert!(check_prompt_length(100, 4096).is_ok(), "under window OK");
        assert!(check_prompt_length(4096, 4096).is_ok(), "exactly at window OK");
        assert!(check_prompt_length(4097, 4096).is_err(), "over window 400");
        assert!(check_prompt_length(99999, 0).is_ok(), "context 0 = unknown, skip");
    }

    // ---- F7: ONE shared content-parts flattener, single recognized key set ----

    #[test]
    fn flatten_content_recognizes_only_text_key_no_anthropic_content_fallback() {
        // A content-part object uses `text`; the prior Anthropic-only `content`
        // fallback is dropped for byte-parity, so a `{content:...}` part
        // contributes nothing.
        let v = serde_json::json!([
            {"type": "text", "text": "hello "},
            "world",
            {"type": "text", "content": "DROPPED"}
        ]);
        assert_eq!(flatten_content(&v, "messages.content").unwrap(), "hello world");
    }

    #[test]
    fn flatten_content_string_and_null() {
        assert_eq!(flatten_content(&serde_json::json!("hi"), "p").unwrap(), "hi");
        assert_eq!(flatten_content(&Value::Null, "p").unwrap(), "");
    }

    #[test]
    fn flatten_content_numeric_is_robust007_400() {
        let err = flatten_content(&serde_json::json!(42), "messages.content").unwrap_err();
        match err {
            ServerError::BadRequest { code, param, .. } => {
                assert_eq!(code.as_deref(), Some("invalid_type"));
                assert_eq!(param.as_deref(), Some("messages.content"));
            }
            other => panic!("expected 400, got {other:?}"),
        }
        assert!(flatten_content(&serde_json::json!(true), "p").is_err(), "bool also 400");
    }

    #[test]
    fn flatten_content_same_string_on_a_mixed_array_for_both_surfaces() {
        // The same content array must flatten to the SAME string regardless of
        // surface — this is the whole point of the shared flattener. Both
        // surfaces call `flatten_content`, so we assert the canonical result
        // here; the per-surface end-to-end byte parity is covered by the
        // cross-surface tool-turn golden test below.
        let mixed = serde_json::json!([
            "a",
            {"type": "text", "text": "b"},
            {"type": "image_url", "image_url": {"url": "x"}}, // no `text` -> contributes nothing
            {"type": "text", "text": "c"}
        ]);
        assert_eq!(flatten_content(&mixed, "messages.content").unwrap(), "abc");
    }

    // ---- F6/F7: cross-surface byte-identical tool transcript golden test ----

    /// Decode a byte-faithful `IdentityByteTokenizer` token stream back to the
    /// rendered prompt string (1 token == 1 byte) so we can compare the EXACT
    /// prompt text each surface produces from an equivalent tool round-trip.
    fn decode_prompt(tokens: &[u32]) -> String {
        String::from_utf8(tokens.iter().map(|t| (*t & 0xff) as u8).collect()).unwrap()
    }

    #[tokio::test]
    async fn openai_and_anthropic_render_byte_identical_tool_transcript() {
        use crate::engine::EngineHandle;
        use crate::wire::anthropic::MessagesRequest;
        use crate::wire::openai::ChatCompletionRequest;

        let engine = EngineHandle::new_for_test(8192);

        // OpenAI shape: assistant carries top-level `tool_calls` (arguments is a
        // JSON *string*); the tool result is a separate `role:"tool"` message.
        //
        // The OpenAI surface forwards the client's `arguments` string verbatim
        // (it is already on-wire JSON text — the wire layer must NOT rewrite
        // client formatting). The Anthropic `input` *object* is serialized via
        // `serde_json::Value::to_string()` (COMPACT, no spaces). So for the two
        // surfaces to render byte-identical transcripts the equivalent OpenAI
        // arguments string must be in the SAME compact form — `{"city":"Paris"}`
        // — which is exactly what an Anthropic client's object serializes to.
        let openai_body = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": "Let me check.", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {
                        "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"
                    }}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18}"}
            ]
        });

        // Anthropic shape: assistant carries a `tool_use` content block
        // (`input` is a JSON *object*); the tool result is a `tool_result`
        // content block inside the next user message.
        let anthropic_body = serde_json::json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "name": "get_weather", "input": {"city": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "content": "{\"temp\": 18}"}
                ]}
            ]
        });

        let openai_req: ChatCompletionRequest = serde_json::from_value(openai_body).unwrap();
        let anthropic_req: MessagesRequest = serde_json::from_value(anthropic_body).unwrap();

        let openai_prompt = decode_prompt(&openai_req.into_job(&engine).unwrap().prompt_tokens);
        let anthropic_prompt =
            decode_prompt(&anthropic_req.into_job(&engine).unwrap().prompt_tokens);

        assert_eq!(
            openai_prompt, anthropic_prompt,
            "OpenAI and Anthropic must render byte-identical tool transcripts\n\
             OPENAI:\n{openai_prompt}\nANTHROPIC:\n{anthropic_prompt}"
        );
        // Sanity: the transcript actually contains the round-trip markers.
        assert!(openai_prompt.contains("<tool_call>"), "tool_call present");
        assert!(openai_prompt.contains("<tool_response>"), "tool_response present");
        assert!(openai_prompt.contains("{\"city\":\"Paris\"}"), "arguments reconciled (compact)");
    }
}
