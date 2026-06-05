//! OpenAI-compatible endpoints: `/v1/chat/completions` + `/v1/completions`.
//!
//! References:
//! - https://platform.openai.com/docs/api-reference/chat
//! - https://platform.openai.com/docs/api-reference/completions
//!
//! the top-level request bodies (`ChatCompletionRequest`,
//! `CompletionRequest`) carry `#[serde(deny_unknown_fields)]` to match
//! the OpenAI schema's `additionalProperties: false` contract. Unknown
//! top-level fields trigger HTTP 400 + OpenAI envelope with
//! `code="unknown_field"` via the [`crate::router::OpenAiJson`] extractor.
//! Inner DTOs (messages, tool defs, tool calls) keep the original
//! permissive behavior so forward-compatible client extras still pass.

use axum::body::Body;
use lumen_runtime::engine::SamplingParams;
use lumen_runtime::tooling::{compose_system_with_tools, ToolSchema};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::engine::{EngineHandle, FinishReason, JobRequest, JobResponseChannel, TokenEvent};
use crate::error::ServerError;
use crate::sse::SseSafeEmitter;
use crate::tokenstop::StopMatcher;

// ----------------------------- Request DTOs -----------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Value,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<AssistantToolCall>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssistantToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: AssistantToolCallFn,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssistantToolCallFn {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub def_type: String,
    pub function: ToolDefFunction,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolDefFunction {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub tools: Vec<ToolDef>,
}

impl ChatCompletionRequest {
    pub fn into_job(self, engine: &EngineHandle) -> Result<JobRequest, ServerError> {
        let prompt = render_chat_prompt(&self.messages, &self.tools)?;
        let prompt_tokens = engine.tokenize_for_request(&prompt);
        let stop_text = parse_stop_field(self.stop);
        let eos = engine.eos_tokens_for_request();
        let max_tokens = self.max_tokens.unwrap_or(256);
        // server-internal sampler defaults aligned with CLI's
        // production defaults (`--repeat-penalty 1.05`). The
        // OpenAI API surface is preserved: the `repetition_penalty` field
        // is NOT in the request schema (OpenAI does not expose it) so this
        // default applies only on the server-internal codepath. When the
        // client explicitly sends `temperature=0`, greedy decoding takes
        // over and the repetition penalty is a no-op for argmax anyway.
        //
        // An omitted `seed` resolves to a fresh per-request random seed (the
        // OpenAI/llama.cpp convention) so identical requests vary; pass an
        // explicit `seed` for reproducible output.
        let sampling = SamplingParams {
            temperature: self.temperature.unwrap_or(0.7),
            seed: Some(self.seed.unwrap_or_else(super::next_random_seed)),
            repetition_penalty: Some(1.05),
            ..Default::default()
        };
        Ok(JobRequest {
            prompt_tokens,
            max_tokens,
            stop_text,
            eos_token_ids: eos,
            sampling,
            suffix_threshold: lumen_runtime::session::Session::DEFAULT_SUFFIX_THRESHOLD,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: Value,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Value>,
}

impl CompletionRequest {
    pub fn into_job(self, engine: &EngineHandle) -> Result<JobRequest, ServerError> {
        let text = match self.prompt {
            Value::String(s) => s,
            Value::Array(arr) => arr
                .into_iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
                .join(""),
            _ => {
                return Err(ServerError::bad_request_field(
                    "prompt must be string or array",
                    "prompt",
                    "invalid_type",
                ))
            }
        };
        let prompt_tokens = engine.tokenize_for_request(&text);
        let stop_text = parse_stop_field(self.stop);
        let eos = engine.eos_tokens_for_request();
        let max_tokens = self.max_tokens.unwrap_or(256);
        // server-internal sampler defaults (see
        // ChatCompletionRequest::into_job for the full rationale). An omitted
        // `seed` resolves to a fresh per-request random seed; pass an explicit
        // `seed` for reproducible output.
        let sampling = SamplingParams {
            temperature: self.temperature.unwrap_or(0.7),
            seed: Some(self.seed.unwrap_or_else(super::next_random_seed)),
            repetition_penalty: Some(1.05),
            ..Default::default()
        };
        Ok(JobRequest {
            prompt_tokens,
            max_tokens,
            stop_text,
            eos_token_ids: eos,
            sampling,
            suffix_threshold: lumen_runtime::session::Session::DEFAULT_SUFFIX_THRESHOLD,
        })
    }
}

fn parse_stop_field(v: Option<Value>) -> Vec<String> {
    match v {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s],
        Some(Value::Array(arr)) => arr
            .into_iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s),
                _ => None,
            })
            .collect(),
        Some(_) => Vec::new(),
    }
}

/// Render a chat-completion request as a single prompt string.
///
/// We do NOT apply the model's chat template here; that is the tokenizer's
/// concern (different models render differently). We emit a stable
/// intermediate form -- system, user, assistant, tool turns separated by
/// the Qwen3.5 ChatML markers -- that maps cleanly through any ChatML
/// tokenizer (Qwen2.5, Qwen3.5, others). If the embedder hands us a
/// custom tokenizer with `apply_chat_template`, the worker can override
/// this in a later iteration.
///
/// emit the canonical Qwen3.5 `enable_thinking=false` empty-think
/// tail (`<think>\n\n</think>\n\n`) so server output matches the CLI's
/// post- chat template. Previously the wire layer ended the prompt
/// at `<|im_start|>assistant\n` with no think block, causing Qwen3.5 to
/// either enter open-think metacommentary (gen budget burn) OR diverge from
/// the CLI's enable_thinking=false canonical pattern. The closed empty-think
/// tail is a no-op for non-Qwen3.5 ChatML models that do not treat
/// `<think>`/`</think>` as special tokens — they render as literal text and
/// are stripped by the wire layer's StopMatcher / SseSafeEmitter only on
/// Qwen3.5 (because only Qwen3.5's special-token map contains them).
fn render_chat_prompt(messages: &[ChatMessage], tools: &[ToolDef]) -> Result<String, ServerError> {
    let mut system: Option<String> = None;
    let mut transcript = String::new();
    for m in messages.iter() {
        // ROBUST-007: `content` must be a string or a content-parts array (or
        // null). A bare number/bool would otherwise be silently coerced by
        // `content_to_string`'s `other => to_string()` arm and accepted (HTTP
        // 200); OpenAI requires string|array, so reject it as a 400 instead.
        match &m.content {
            Value::String(_) | Value::Array(_) | Value::Null => {}
            _ => {
                return Err(ServerError::bad_request_field(
                    "message 'content' must be a string or a content-parts array",
                    "messages.content",
                    "invalid_type",
                ))
            }
        }
        match m.role.as_str() {
            "system" => system = Some(content_to_string(&m.content)),
            "user" => {
                transcript.push_str("<|im_start|>user\n");
                transcript.push_str(&content_to_string(&m.content));
                transcript.push_str("<|im_end|>\n");
            }
            "assistant" => {
                transcript.push_str("<|im_start|>assistant\n");
                transcript.push_str(&content_to_string(&m.content));
                for tc in &m.tool_calls {
                    transcript.push_str("\n<tool_call>\n{\"name\": \"");
                    transcript.push_str(&tc.function.name);
                    transcript.push_str("\", \"arguments\": ");
                    transcript.push_str(&tc.function.arguments);
                    transcript.push_str("}\n</tool_call>");
                }
                transcript.push_str("<|im_end|>\n");
            }
            "tool" => {
                transcript.push_str("<|im_start|>user\n<tool_response>\n");
                transcript.push_str(&content_to_string(&m.content));
                transcript.push_str("\n</tool_response><|im_end|>\n");
            }
            other => {
                return Err(ServerError::bad_request_field(
                    format!("unknown message role: {other}"),
                    "messages[].role",
                    "invalid_value",
                ));
            }
        }
    }

    let tool_schemas: Vec<ToolSchema> = tools
        .iter()
        .map(|t| ToolSchema {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            parameters_json_schema: serde_json::to_string(&t.function.parameters)
                .unwrap_or_else(|_| "{}".into()),
        })
        .collect();
    let final_system = compose_system_with_tools(system.as_deref(), &tool_schemas);

    let mut prompt = String::new();
    if !final_system.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(&final_system);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str(&transcript);
    prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    Ok(prompt)
}

fn content_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
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
            out
        }
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

// ----------------------------- SSE streaming chat ------------------------

fn sse_frame(payload: &str) -> Vec<u8> {
    let mut buf = String::with_capacity(payload.len() + 8);
    buf.push_str("data: ");
    buf.push_str(payload);
    buf.push_str("\n\n");
    buf.into_bytes()
}

fn sse_done() -> Vec<u8> {
    b"data: [DONE]\n\n".to_vec()
}

/// Convert a `mpsc::Receiver<Vec<u8>>` into a `Body` whose chunks are the
/// raw SSE frames. The receiver is filled by a background task that drives
/// the streaming state machine.
fn body_from_byte_stream(rx: tokio::sync::mpsc::Receiver<Vec<u8>>) -> Body {
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (
            Ok::<bytes::Bytes, std::io::Error>(bytes::Bytes::from(chunk)),
            rx,
        ))
    });
    Body::from_stream(stream)
}

pub fn stream_chat(rx: JobResponseChannel, model: String, created: u64) -> Body {
    let (tx, body_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    tokio::spawn(drive_chat_stream(rx, tx, model, created, true));
    body_from_byte_stream(body_rx)
}

pub fn stream_completion(rx: JobResponseChannel, model: String, created: u64) -> Body {
    let (tx, body_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    tokio::spawn(drive_chat_stream(rx, tx, model, created, false));
    body_from_byte_stream(body_rx)
}

async fn drive_chat_stream(
    mut rx: JobResponseChannel,
    tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    model: String,
    created: u64,
    chat: bool,
) {
    let id = format!("chatcmpl-lumen-{created:x}-{:x}", super::next_response_seq());
    let mut emitter = SseSafeEmitter::new();
    let mut stop_matcher = StopMatcher::new(Vec::new());
    let mut finish_reason: Option<FinishReason> = None;
    let mut tool_call_index = 0usize;
    let mut emitted_any_tool_call = false;

    if chat {
        let head = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant" },
                "finish_reason": null
            }],
        });
        if tx.send(sse_frame(&head.to_string())).await.is_err() {
            return;
        }
    }

    while let Some(evt) = rx.recv().await {
        match evt {
            TokenEvent::PrefillDone { .. } => {}
            TokenEvent::Token { delta_text, .. } => {
                let delta = emitter.push(&delta_text);
                let (safe_text, hit_stop) = stop_matcher.push(&delta.text);
                if !safe_text.is_empty() {
                    let frame = if chat {
                        json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": safe_text },
                                "finish_reason": null
                            }],
                        })
                    } else {
                        json!({
                            "id": id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "text": safe_text,
                                "finish_reason": null,
                            }],
                        })
                    };
                    if tx.send(sse_frame(&frame.to_string())).await.is_err() {
                        return;
                    }
                }
                for tc in delta.tool_calls {
                    tool_call_index += 1;
                    emitted_any_tool_call = true;
                    if chat {
                        let frame = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": tool_call_index - 1,
                                        "id": format!("call_lumen_{}", tool_call_index),
                                        "type": "function",
                                        "function": {
                                            "name": tc.name,
                                            "arguments": tc.arguments_json,
                                        }
                                    }]
                                },
                                "finish_reason": null,
                            }],
                        });
                        if tx.send(sse_frame(&frame.to_string())).await.is_err() {
                            return;
                        }
                    }
                }
                if hit_stop {
                    finish_reason = Some(FinishReason::Stop);
                    break;
                }
            }
            TokenEvent::Done { finish_reason: fr, .. } => {
                finish_reason = Some(fr);
                break;
            }
            TokenEvent::Error(msg) => {
                let err = json!({"error": { "message": msg, "type": "api_error" }});
                let _ = tx.send(sse_frame(&err.to_string())).await;
                let _ = tx.send(sse_done()).await;
                return;
            }
        }
    }

    let (residual, _incomplete) = emitter.finish();
    if !residual.text.is_empty() {
        let frame = if chat {
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": { "content": residual.text },
                    "finish_reason": null
                }],
            })
        } else {
            json!({
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": residual.text,
                    "finish_reason": null,
                }],
            })
        };
        if tx.send(sse_frame(&frame.to_string())).await.is_err() {
            return;
        }
    }
    let reason = match finish_reason {
        Some(FinishReason::Stop) if emitted_any_tool_call => FinishReason::ToolCalls,
        Some(r) => r,
        None => FinishReason::Stop,
    };
    let tail = if chat {
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": reason.as_openai(),
            }],
        })
    } else {
        json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "text": "",
                "finish_reason": reason.as_openai(),
            }],
        })
    };
    let _ = tx.send(sse_frame(&tail.to_string())).await;
    let _ = tx.send(sse_done()).await;
}

// ----------------------------- Non-streaming chat ------------------------

pub async fn collect_chat(
    mut rx: JobResponseChannel,
    model: String,
    created: u64,
) -> Result<Value, ServerError> {
    let mut emitter = SseSafeEmitter::new();
    let mut stop_matcher = StopMatcher::new(Vec::new());
    let mut content = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut finish = FinishReason::Stop;

    while let Some(evt) = rx.recv().await {
        match evt {
            TokenEvent::PrefillDone { .. } => {}
            TokenEvent::Token { delta_text, .. } => {
                let delta = emitter.push(&delta_text);
                let (safe_text, hit_stop) = stop_matcher.push(&delta.text);
                content.push_str(&safe_text);
                for tc in delta.tool_calls {
                    tool_calls.push(json!({
                        "id": format!("call_lumen_{}", tool_calls.len() + 1),
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments_json,
                        }
                    }));
                }
                if hit_stop {
                    break;
                }
            }
            TokenEvent::Done { finish_reason, prompt_tokens: p, completion_tokens: c } => {
                finish = finish_reason;
                prompt_tokens = p;
                completion_tokens = c;
                break;
            }
            TokenEvent::Error(msg) => return Err(ServerError::Runtime(msg)),
        }
    }
    let (residual, _) = emitter.finish();
    content.push_str(&residual.text);

    if !tool_calls.is_empty() && finish == FinishReason::Stop {
        finish = FinishReason::ToolCalls;
    }

    let msg = if tool_calls.is_empty() {
        json!({ "role": "assistant", "content": content })
    } else {
        json!({
            "role": "assistant",
            "content": if content.is_empty() { Value::Null } else { Value::String(content) },
            "tool_calls": tool_calls,
        })
    };
    Ok(json!({
        "id": format!("chatcmpl-lumen-{created:x}-{:x}", super::next_response_seq()),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": finish.as_openai(),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    }))
}

/// Public test helper: build a chat-completion JSON from a sequence of
/// `TokenEvent`s without booting a real engine. Exposed via the
/// `Tokenize`-free path for unit testing of the wire layer's behavior on
/// tool-call-bearing token streams.
///
/// wraps the raw `mpsc::Receiver` in an unpooled `PooledReceiver`
/// (pool=None) so the test helper's signature matches the real
/// `collect_chat` (which now expects `JobResponseChannel`).
#[cfg(any(test, doctest))]
pub async fn collect_chat_from_events(
    events: Vec<TokenEvent>,
    model: String,
    created: u64,
) -> Result<Value, ServerError> {
    let (tx, rx) = tokio::sync::mpsc::channel(events.len().max(1));
    let return_sender = tx.clone();
    for e in events {
        tx.send(e).await.unwrap();
    }
    drop(tx);
    // test helper; no cancellation guard needed (no live
    // worker, no client-disconnect path).
    let pooled = crate::engine::PooledReceiver::new(rx, return_sender, None, 0, None);
    collect_chat(pooled, model, created).await
}

pub async fn collect_completion(
    mut rx: JobResponseChannel,
    model: String,
    created: u64,
) -> Result<Value, ServerError> {
    let mut emitter = SseSafeEmitter::new();
    let mut stop_matcher = StopMatcher::new(Vec::new());
    let mut text = String::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut finish = FinishReason::Stop;

    while let Some(evt) = rx.recv().await {
        match evt {
            TokenEvent::PrefillDone { .. } => {}
            TokenEvent::Token { delta_text, .. } => {
                let delta = emitter.push(&delta_text);
                let (safe_text, hit_stop) = stop_matcher.push(&delta.text);
                text.push_str(&safe_text);
                if hit_stop {
                    break;
                }
            }
            TokenEvent::Done { finish_reason, prompt_tokens: p, completion_tokens: c } => {
                finish = finish_reason;
                prompt_tokens = p;
                completion_tokens = c;
                break;
            }
            TokenEvent::Error(msg) => return Err(ServerError::Runtime(msg)),
        }
    }
    let (residual, _) = emitter.finish();
    text.push_str(&residual.text);

    Ok(json!({
        "id": format!("cmpl-lumen-{created:x}-{:x}", super::next_response_seq()),
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish.as_openai(),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_runtime::tooling::Qwen35Renderer;

    fn tok(text: &str) -> TokenEvent {
        TokenEvent::Token { token_id: 0, delta_text: text.to_string() }
    }

    #[tokio::test]
    async fn collect_chat_aggregates_text_and_tool_calls() {
        let call = Qwen35Renderer::render_one_call("get_weather", "{\"city\": \"Paris\"}");
        let events = vec![
            tok("Sure. "),
            tok(&call),
            tok(" The weather is sunny."),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 3,
                completion_tokens: 12,
            },
        ];
        let resp = collect_chat_from_events(events, "test-model".into(), 1234).await.unwrap();
        // Tool calls present -> finish_reason becomes tool_calls automatically.
        assert_eq!(resp["choices"][0]["finish_reason"], "tool_calls");
        let tcs = &resp["choices"][0]["message"]["tool_calls"];
        assert!(tcs.is_array());
        assert_eq!(tcs[0]["function"]["name"], "get_weather");
        // Text content includes the parts surrounding the tool call.
        let content = resp["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(content.contains("Sure."));
        assert!(content.contains("sunny"));
    }

    #[tokio::test]
    async fn collect_chat_marker_split_across_tokens() {
        // Split the marker across two token deltas: only `<tool` and then
        // `_call>\n...` -- the parser must NOT leak `<tool` to the wire.
        let events = vec![
            tok("Calling <tool"),
            tok("_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 2,
            },
        ];
        let resp = collect_chat_from_events(events, "test".into(), 1).await.unwrap();
        let content = resp["choices"][0]["message"]["content"].as_str().unwrap();
        // The literal "<tool" must not appear in user-visible content.
        assert!(!content.contains("<tool"));
        assert_eq!(resp["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "f");
    }

    #[tokio::test]
    async fn collect_chat_max_tokens_finish_reason() {
        let events = vec![
            tok("hello"),
            TokenEvent::Done {
                finish_reason: FinishReason::Length,
                prompt_tokens: 1,
                completion_tokens: 5,
            },
        ];
        let resp = collect_chat_from_events(events, "test".into(), 1).await.unwrap();
        assert_eq!(resp["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn collect_chat_passes_through_error_event() {
        let events = vec![TokenEvent::Error("model exploded".into())];
        let r = collect_chat_from_events(events, "test".into(), 1).await;
        assert!(r.is_err());
    }

    // ------------------------------------------------------------------
    // server `render_chat_prompt` must emit the Qwen3.5
    // `enable_thinking=false` empty-think tail (`<think>\n\n</think>\n\n`)
    // so the server's rendered prompt matches the CLI's
    // `apply_chat_template_with_system` post- output.
    // ------------------------------------------------------------------

    fn user_msg(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: Value::String(text.into()),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    fn system_msg(text: &str) -> ChatMessage {
        ChatMessage {
            role: "system".into(),
            content: Value::String(text.into()),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    fn assistant_msg(text: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            content: Value::String(text.into()),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    #[test]
    fn render_chat_prompt_user_only_emits_closed_think() {
        let messages = vec![user_msg("Hello")];
        let out = render_chat_prompt(&messages, &[]).unwrap();
        // CLI's `apply_chat_template_with_system("Hello", None)` for qwen35
        // post- produces exactly this string (see crates/lumen-cli
        // /src/tokenize.rs:273-292).
        let expected =
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        assert_eq!(out, expected, "render_chat_prompt user-only != CLI output");
    }

    #[test]
    fn render_chat_prompt_system_plus_user_emits_closed_think() {
        let messages = vec![system_msg("You are helpful."), user_msg("Hi")];
        let out = render_chat_prompt(&messages, &[]).unwrap();
        let expected = "<|im_start|>system\nYou are helpful.<|im_end|>\n\
                        <|im_start|>user\nHi<|im_end|>\n\
                        <|im_start|>assistant\n<think>\n\n</think>\n\n";
        assert_eq!(out, expected, "render_chat_prompt system+user != CLI output");
    }

    #[test]
    fn render_chat_prompt_multi_turn_emits_closed_think_only_at_tail() {
        // Three-turn: user, assistant, user. The empty-think tail must
        // appear ONLY at the final assistant prefix, NOT at the previous
        // assistant turn (which carries real content).
        let messages = vec![
            user_msg("Q1"),
            assistant_msg("A1"),
            user_msg("Q2"),
        ];
        let out = render_chat_prompt(&messages, &[]).unwrap();
        let expected = "<|im_start|>user\nQ1<|im_end|>\n\
                        <|im_start|>assistant\nA1<|im_end|>\n\
                        <|im_start|>user\nQ2<|im_end|>\n\
                        <|im_start|>assistant\n<think>\n\n</think>\n\n";
        assert_eq!(out, expected, "multi-turn render did not match CLI shape");
        // Defensive: the empty-think substring should occur exactly once.
        let count = out.matches("<think>\n\n</think>").count();
        assert_eq!(count, 1, "empty-think tail must appear exactly once");
    }

    #[test]
    fn render_chat_prompt_matches_cli_for_qwen35_user_only() {
        // Cross-check against CLI's apply_chat_template_with_system. We
        // can't construct a real BpeTokenizer here without a model fixture,
        // so we mirror the exact format string from tokenize.rs:288.
        // (The unit test in tokenize.rs guards the CLI side; this guards
        // the server side; they must produce byte-identical strings.)
        let messages = vec![user_msg("Hello")];
        let server_out = render_chat_prompt(&messages, &[]).unwrap();
        let cli_out = format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt = "Hello"
        );
        assert_eq!(server_out, cli_out, "server render must match CLI render byte-for-byte");
    }
}

