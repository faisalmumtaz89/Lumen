//! Anthropic-compatible `/v1/messages` endpoint.
//!
//! Reference: https://docs.anthropic.com/en/api/messages
//!
//! Streaming format uses Anthropic typed events:
//!
//! ```text
//! event: message_start
//! data: {...}
//!
//! event: content_block_start
//! data: {...}
//!
//! event: content_block_delta
//! data: {...}
//!
//! event: content_block_stop
//! data: {...}
//!
//! event: message_delta
//! data: {...}
//!
//! event: message_stop
//! data: {...}
//! ```

use axum::body::Body;
use lumen_runtime::engine::SamplingParams;
use lumen_runtime::tooling::{compose_system_with_tools, ToolSchema};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::engine::{EngineHandle, FinishReason, JobRequest, JobResponseChannel, TokenEvent};
use crate::error::ServerError;
use crate::sse::SseSafeEmitter;
use crate::tokenstop::StopMatcher;

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub input_schema: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: usize,
    #[serde(default)]
    pub system: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub tools: Vec<AnthropicTool>,
}

impl MessagesRequest {
    pub fn into_job(self, engine: &EngineHandle) -> Result<JobRequest, ServerError> {
        let system_text = self.system.map(|v| content_to_string(&v));
        let tool_schemas: Vec<ToolSchema> = self
            .tools
            .iter()
            .map(|t| ToolSchema {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters_json_schema: serde_json::to_string(&t.input_schema)
                    .unwrap_or_else(|_| "{}".into()),
            })
            .collect();
        let final_system = compose_system_with_tools(system_text.as_deref(), &tool_schemas);
        let prompt = render_prompt(&final_system, &self.messages)?;
        let prompt_tokens = engine.tokenize_for_request(&prompt);
        let eos = engine.eos_tokens_for_request();
        // server-internal sampler defaults aligned with CLI's
        // production defaults. Anthropic Messages API does not expose
        // repetition_penalty or seed in its request schema, so the defaults
        // apply server-internal only: every request gets a fresh random seed,
        // so identical requests vary (matching the real Anthropic API's
        // non-deterministic behavior).
        let sampling = SamplingParams {
            temperature: self.temperature.unwrap_or(0.7),
            seed: Some(super::next_random_seed()),
            repetition_penalty: Some(1.05),
            ..Default::default()
        };
        Ok(JobRequest {
            prompt_tokens,
            max_tokens: self.max_tokens,
            stop_text: self.stop_sequences,
            eos_token_ids: eos,
            sampling,
            suffix_threshold: lumen_runtime::session::Session::DEFAULT_SUFFIX_THRESHOLD,
        })
    }
}

fn render_prompt(system: &str, messages: &[AnthropicMessage]) -> Result<String, ServerError> {
    let mut prompt = String::new();
    if !system.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system);
        prompt.push_str("<|im_end|>\n");
    }
    for m in messages {
        match m.role.as_str() {
            "user" => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(&content_to_string(&m.content));
                prompt.push_str("<|im_end|>\n");
            }
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                prompt.push_str(&content_to_string(&m.content));
                prompt.push_str("<|im_end|>\n");
            }
            other => {
                return Err(ServerError::bad_request_field(
                    format!("unknown anthropic role: {other}"),
                    "messages[].role",
                    "invalid_value",
                ));
            }
        }
    }
    // emit the canonical Qwen3.5 `enable_thinking=false`
    // empty-think tail so the Anthropic-compat path matches CLI + OpenAI
    // server behavior. See render_chat_prompt in wire/openai.rs for the
    // full rationale.
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
                    // Anthropic content blocks: {"type": "text", "text": "..."}
                    if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                        out.push_str(text);
                    } else if let Some(text) = obj.get("content").and_then(|v| v.as_str()) {
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

// ----------------------------- SSE streaming ----------------------------

fn sse_event(event: &str, payload: &str) -> Vec<u8> {
    let mut buf = String::with_capacity(payload.len() + event.len() + 16);
    buf.push_str("event: ");
    buf.push_str(event);
    buf.push('\n');
    buf.push_str("data: ");
    buf.push_str(payload);
    buf.push_str("\n\n");
    buf.into_bytes()
}

fn body_from_byte_stream(rx: mpsc::Receiver<Vec<u8>>) -> Body {
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (
            Ok::<bytes::Bytes, std::io::Error>(bytes::Bytes::from(chunk)),
            rx,
        ))
    });
    Body::from_stream(stream)
}

pub fn stream_messages(rx: JobResponseChannel, model: String) -> Body {
    let (tx, body_rx) = mpsc::channel::<Vec<u8>>(64);
    tokio::spawn(drive_messages_stream(rx, tx, model));
    body_from_byte_stream(body_rx)
}

async fn drive_messages_stream(
    mut rx: JobResponseChannel,
    tx: mpsc::Sender<Vec<u8>>,
    model: String,
) {
    let msg_id = format!("msg_lumen_{:x}-{:x}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64).unwrap_or(0), super::next_response_seq());
    let mut emitter = SseSafeEmitter::new();
    let mut stop_matcher = StopMatcher::new(Vec::new());
    let mut finish_reason: Option<FinishReason> = None;
    let mut text_block_open = false;
    let mut block_count = 0usize;
    let mut input_tokens = 0usize;
    let mut output_tokens = 0usize;

    // message_start
    let start = json!({
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": null,
            "stop_sequence": null,
            "usage": { "input_tokens": 0, "output_tokens": 0 }
        }
    });
    if tx.send(sse_event("message_start", &start.to_string())).await.is_err() {
        return;
    }

    while let Some(evt) = rx.recv().await {
        match evt {
            TokenEvent::PrefillDone { .. } => {}
            TokenEvent::Token { delta_text, .. } => {
                let delta = emitter.push(&delta_text);
                let (safe_text, hit_stop) = stop_matcher.push(&delta.text);
                if !safe_text.is_empty() {
                    if !text_block_open {
                        // Open the first text content block.
                        let b = json!({
                            "type": "content_block_start",
                            "index": block_count,
                            "content_block": { "type": "text", "text": "" }
                        });
                        if tx.send(sse_event("content_block_start", &b.to_string())).await.is_err() {
                            return;
                        }
                        text_block_open = true;
                    }
                    let d = json!({
                        "type": "content_block_delta",
                        "index": block_count,
                        "delta": { "type": "text_delta", "text": safe_text }
                    });
                    if tx.send(sse_event("content_block_delta", &d.to_string())).await.is_err() {
                        return;
                    }
                }
                for tc in delta.tool_calls {
                    // Close any open text block before opening the tool block.
                    if text_block_open {
                        let s = json!({ "type": "content_block_stop", "index": block_count });
                        if tx.send(sse_event("content_block_stop", &s.to_string())).await.is_err() {
                            return;
                        }
                        text_block_open = false;
                        block_count += 1;
                    }
                    let idx = block_count;
                    let start_block = json!({
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": format!("toolu_lumen_{}", idx),
                            "name": tc.name,
                            "input": {},
                        }
                    });
                    if tx.send(sse_event("content_block_start", &start_block.to_string())).await.is_err() {
                        return;
                    }
                    // Emit the JSON arguments as a single input_json_delta.
                    let d = json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": { "type": "input_json_delta", "partial_json": tc.arguments_json }
                    });
                    if tx.send(sse_event("content_block_delta", &d.to_string())).await.is_err() {
                        return;
                    }
                    let stop_block = json!({ "type": "content_block_stop", "index": idx });
                    if tx.send(sse_event("content_block_stop", &stop_block.to_string())).await.is_err() {
                        return;
                    }
                    block_count += 1;
                }
                if hit_stop {
                    finish_reason = Some(FinishReason::Stop);
                    break;
                }
            }
            TokenEvent::Done { finish_reason: fr, prompt_tokens, completion_tokens } => {
                finish_reason = Some(fr);
                input_tokens = prompt_tokens;
                output_tokens = completion_tokens;
                break;
            }
            TokenEvent::Error(msg) => {
                let err = json!({"type": "error", "error": { "type": "api_error", "message": msg }});
                let _ = tx.send(sse_event("error", &err.to_string())).await;
                return;
            }
        }
    }
    // Flush residual.
    let (residual, _incomplete) = emitter.finish();
    if !residual.text.is_empty() {
        if !text_block_open {
            let b = json!({
                "type": "content_block_start",
                "index": block_count,
                "content_block": { "type": "text", "text": "" }
            });
            let _ = tx.send(sse_event("content_block_start", &b.to_string())).await;
            text_block_open = true;
        }
        let d = json!({
            "type": "content_block_delta",
            "index": block_count,
            "delta": { "type": "text_delta", "text": residual.text }
        });
        let _ = tx.send(sse_event("content_block_delta", &d.to_string())).await;
    }
    if text_block_open {
        let s = json!({ "type": "content_block_stop", "index": block_count });
        let _ = tx.send(sse_event("content_block_stop", &s.to_string())).await;
    }
    let reason = finish_reason.unwrap_or(FinishReason::Stop);
    let delta_msg = json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": reason.as_anthropic(),
            "stop_sequence": null,
        },
        "usage": { "output_tokens": output_tokens, "input_tokens": input_tokens }
    });
    let _ = tx.send(sse_event("message_delta", &delta_msg.to_string())).await;
    let _ = tx.send(sse_event("message_stop", "{\"type\":\"message_stop\"}")).await;
}

// ----------------------------- Non-streaming ----------------------------

pub async fn collect_messages(
    mut rx: JobResponseChannel,
    model: String,
) -> Result<Value, ServerError> {
    let mut emitter = SseSafeEmitter::new();
    let mut stop_matcher = StopMatcher::new(Vec::new());
    let mut text = String::new();
    let mut tool_blocks: Vec<Value> = Vec::new();
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
                for tc in delta.tool_calls {
                    tool_blocks.push(json!({
                        "type": "tool_use",
                        "id": format!("toolu_lumen_{}", tool_blocks.len() + 1),
                        "name": tc.name,
                        "input": serde_json::from_str::<Value>(&tc.arguments_json)
                            .unwrap_or_else(|_| Value::String(tc.arguments_json)),
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
    text.push_str(&residual.text);

    if !tool_blocks.is_empty() && finish == FinishReason::Stop {
        finish = FinishReason::ToolCalls;
    }

    let mut content_blocks: Vec<Value> = Vec::new();
    if !text.is_empty() {
        content_blocks.push(json!({"type": "text", "text": text}));
    }
    content_blocks.extend(tool_blocks);

    Ok(json!({
        "id": format!("msg_lumen_{:x}-{:x}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64).unwrap_or(0), super::next_response_seq()),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": finish.as_anthropic(),
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        }
    }))
}
