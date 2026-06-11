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

/// Anthropic extended-thinking config: `{"type": "enabled"|"disabled",
/// "budget_tokens": N}`. Permissive (not `deny_unknown_fields`) so future
/// Anthropic keys pass through. `type == "enabled"` turns reasoning on;
/// anything else (including `"disabled"`) turns it off.
#[derive(Debug, Clone, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(default)]
    pub budget_tokens: Option<usize>,
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
    /// Anthropic-valid sampler subset. The Messages API exposes `top_p` and
    /// `top_k` (NOT presence/frequency penalties); both are honored on the
    /// CLI and were previously HTTP-400-rejected here by `deny_unknown_fields`.
    /// `None` (omitted) leaves the sampler default untouched.
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub tools: Vec<AnthropicTool>,
    /// Anthropic extended-thinking control. `Some({type:"enabled"})` opens the
    /// `<think>` block (reasoning surfaced as a `thinking` content block);
    /// `Some({type:"disabled"})` forces it closed; `None` defers to the
    /// `LUMEN_CHAT_ENABLE_THINKING` env override then the process default.
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
}

impl MessagesRequest {
    /// Resolve the per-request reasoning toggle via the single shared resolver.
    /// The Anthropic `thinking.type == "enabled"` maps to `Some(true)`, any
    /// other explicit value to `Some(false)`, and an absent `thinking` field to
    /// `None` (defer to env/default).
    pub fn resolve_thinking(&self) -> bool {
        let per_request = self
            .thinking
            .as_ref()
            .map(|t| t.thinking_type == "enabled");
        super::resolve_enable_thinking(per_request)
    }

    pub fn into_job(self, engine: &EngineHandle) -> Result<JobRequest, ServerError> {
        let enable_thinking = self.resolve_thinking();
        // Per-request reasoning budget (Anthropic `thinking.budget_tokens`);
        // falls back to the shared default. Carried for Part 4 (decode loop).
        let reasoning_budget = self
            .thinking
            .as_ref()
            .and_then(|t| t.budget_tokens)
            .unwrap_or_else(lumen_runtime::runtime_defaults::chat_reasoning_budget_default);
        // Flatten the system field through the SAME shared helper as message
        // content (ROBUST-007 guard + single recognized key set), so a number
        // `system` 400s identically to a number `content`.
        let system_text = match &self.system {
            Some(v) => Some(super::flatten_content(v, "system")?),
            None => None,
        };
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
        let prompt = render_prompt(&final_system, &self.messages, enable_thinking)?;
        let prompt_tokens = engine.tokenize_for_request(&prompt);
        // Synchronous oversize guard: 400 BEFORE the 200/SSE stream opens.
        super::check_prompt_length(prompt_tokens.len(), engine.context_length())?;
        let eos = engine.eos_tokens_for_request();
        // server-internal sampler defaults aligned with CLI's
        // production defaults. Anthropic Messages API does not expose
        // repetition_penalty or seed in its request schema, so the defaults
        // apply server-internal only: every request gets a fresh random seed,
        // so identical requests vary (matching the real Anthropic API's
        // non-deterministic behavior).
        let sampling = SamplingParams {
            temperature: self
                .temperature
                .unwrap_or_else(lumen_runtime::runtime_defaults::default_temperature),
            seed: Some(super::next_random_seed()),
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: Some(super::diag_repetition_penalty()),
            frequency_penalty: Some(super::diag_frequency_penalty()),
            repeat_last_n: super::diag_repeat_last_n(),
            anti_restate: super::diag_anti_restate(),
            ..Default::default()
        };
        Ok(JobRequest {
            prompt_tokens,
            max_tokens: self.max_tokens,
            stop_text: self.stop_sequences,
            eos_token_ids: eos,
            sampling,
            suffix_threshold: lumen_runtime::session::Session::DEFAULT_SUFFIX_THRESHOLD,
            enable_thinking,
            reasoning_budget,
        })
    }
}

fn render_prompt(
    system: &str,
    messages: &[AnthropicMessage],
    enable_thinking: bool,
) -> Result<String, ServerError> {
    let mut prompt = String::new();
    if !system.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system);
        prompt.push_str("<|im_end|>\n");
    }
    for m in messages {
        match m.role.as_str() {
            "user" => render_user_turn(&mut prompt, &m.content)?,
            "assistant" => render_assistant_turn(&mut prompt, &m.content)?,
            other => {
                return Err(ServerError::bad_request_field(
                    format!("unknown anthropic role: {other}"),
                    "messages[].role",
                    "invalid_value",
                ));
            }
        }
    }
    // Open vs closed `<think>` tail from the single shared helper, matching
    // the CLI + OpenAI server behaviour exactly (closed when reasoning is
    // off — the default — so this path is byte-identical to before). See
    // render_chat_prompt in wire/openai.rs for the full rationale.
    prompt.push_str("<|im_start|>assistant\n");
    prompt.push_str(lumen_runtime::runtime_defaults::think_prompt_tail(enable_thinking));
    Ok(prompt)
}

/// Render an Anthropic `user` message. A user turn may carry plain text AND
/// `tool_result` content blocks (Anthropic models a tool result as a block
/// inside the user message, where OpenAI uses a separate `role:"tool"`
/// message). Each `tool_result` block is re-rendered as the shared ChatML
/// `<tool_response>` turn so the on-wire transcript is byte-identical to the
/// OpenAI surface's; any remaining text is emitted as a normal user turn.
///
/// String / null content (no tool blocks) renders exactly as before:
/// `<|im_start|>user\n{text}<|im_end|>\n`.
fn render_user_turn(prompt: &mut String, content: &Value) -> Result<(), ServerError> {
    // The common case (string / content-parts WITHOUT tool_result) is a plain
    // user turn; `partition_tool_result_blocks` returns the flattened text and
    // the ordered tool-result contents.
    let (text, tool_results) = partition_tool_result_blocks(content, "messages.content")?;
    // Tool results precede any trailing user text, mirroring the
    // assistant-then-tool ordering OpenAI produces (tool result is its own
    // turn there, emitted before the next user message).
    for tr in &tool_results {
        prompt.push_str(&lumen_runtime::tooling::render_tool_response_turn(tr));
    }
    // Only emit a user turn when there is text OR there were no tool results
    // at all (so an empty plain user message still renders, byte-identical to
    // before). A user message that is PURELY tool_result blocks emits no
    // stray empty `<|im_start|>user\n<|im_end|>` turn.
    if !text.is_empty() || tool_results.is_empty() {
        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(&text);
        prompt.push_str("<|im_end|>\n");
    }
    Ok(())
}

/// Render an Anthropic `assistant` message. An assistant turn may carry text
/// AND `tool_use` content blocks (`{type:"tool_use", name, input}`). Text is
/// flattened first, then each `tool_use` block is re-rendered through the
/// shared tool-call helper so the transcript is byte-identical to the OpenAI
/// surface. The Anthropic `input` *object* is serialized to the same on-wire
/// `{name, arguments:<json>}` Qwen form the OpenAI string `arguments` produces.
fn render_assistant_turn(prompt: &mut String, content: &Value) -> Result<(), ServerError> {
    let (text, tool_uses) = partition_tool_use_blocks(content, "messages.content")?;
    prompt.push_str("<|im_start|>assistant\n");
    prompt.push_str(&text);
    for (name, arguments_json) in &tool_uses {
        prompt.push_str(&lumen_runtime::tooling::render_assistant_tool_call_segment(
            name,
            arguments_json,
        ));
    }
    prompt.push_str("<|im_end|>\n");
    Ok(())
}

/// Walk an assistant content value, returning `(flattened_text, tool_uses)`
/// where each tool_use is `(name, arguments_json)`. Recognizes
/// `{type:"tool_use", name, input}` blocks; the `input` object is serialized
/// to a JSON string (the on-wire `arguments` form). Bare-string and
/// `{type:"text", text}` parts flatten into the text via the SAME key set as
/// the shared `flatten_content`. A bare number/bool top-level content 400s
/// (ROBUST-007), consistent with the text flattener.
fn partition_tool_use_blocks(
    content: &Value,
    param: &str,
) -> Result<(String, Vec<(String, String)>), ServerError> {
    match content {
        // No typed blocks possible in a string/null: reuse the shared text
        // flattener verbatim (also enforces ROBUST-007 on scalars).
        Value::String(_) | Value::Null => {
            Ok((super::flatten_content(content, param)?, Vec::new()))
        }
        Value::Array(arr) => {
            let mut text = String::new();
            let mut tool_uses = Vec::new();
            for piece in arr {
                if let Some(s) = piece.as_str() {
                    text.push_str(s);
                } else if let Some(obj) = piece.as_object() {
                    match obj.get("type").and_then(|v| v.as_str()) {
                        Some("tool_use") => {
                            let name = obj
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            // `input` is a JSON object on the wire; serialize to
                            // the raw JSON string the call body expects. Absent
                            // input -> empty object, matching an empty-args call.
                            let arguments_json = obj
                                .get("input")
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "{}".to_string());
                            tool_uses.push((name, arguments_json));
                        }
                        // text part (or any other block that carries `text`).
                        _ => {
                            if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                                text.push_str(t);
                            }
                        }
                    }
                }
            }
            Ok((text, tool_uses))
        }
        _ => Err(ServerError::bad_request_field(
            "message 'content' must be a string or a content-parts array",
            param,
            "invalid_type",
        )),
    }
}

/// Walk a user content value, returning `(flattened_text, tool_result_contents)`.
/// Recognizes `{type:"tool_result", content}` blocks; the inner `content`
/// (string OR content-parts array) is flattened through the shared
/// `flatten_content`. Non-tool_result parts flatten into the text via the same
/// key set. A bare number/bool top-level content 400s (ROBUST-007).
fn partition_tool_result_blocks(
    content: &Value,
    param: &str,
) -> Result<(String, Vec<String>), ServerError> {
    match content {
        Value::String(_) | Value::Null => {
            Ok((super::flatten_content(content, param)?, Vec::new()))
        }
        Value::Array(arr) => {
            let mut text = String::new();
            let mut tool_results = Vec::new();
            for piece in arr {
                if let Some(s) = piece.as_str() {
                    text.push_str(s);
                } else if let Some(obj) = piece.as_object() {
                    match obj.get("type").and_then(|v| v.as_str()) {
                        Some("tool_result") => {
                            // Inner content is itself string | content-parts;
                            // flatten through the shared helper (ROBUST-007 +
                            // single key set). Absent inner content -> empty.
                            let inner = match obj.get("content") {
                                Some(c) => super::flatten_content(c, param)?,
                                None => String::new(),
                            };
                            tool_results.push(inner);
                        }
                        _ => {
                            if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                                text.push_str(t);
                            }
                        }
                    }
                }
            }
            Ok((text, tool_results))
        }
        _ => Err(ServerError::bad_request_field(
            "message 'content' must be a string or a content-parts array",
            param,
            "invalid_type",
        )),
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

pub fn stream_messages(
    rx: JobResponseChannel,
    model: String,
    thinking: bool,
    stop: Vec<String>,
) -> Body {
    let (tx, body_rx) = mpsc::channel::<Vec<u8>>(64);
    tokio::spawn(drive_messages_stream(rx, tx, model, thinking, stop));
    body_from_byte_stream(body_rx)
}

async fn drive_messages_stream(
    mut rx: JobResponseChannel,
    tx: mpsc::Sender<Vec<u8>>,
    model: String,
    thinking: bool,
    stop: Vec<String>,
) {
    let msg_id = format!("msg_lumen_{:x}-{:x}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64).unwrap_or(0), super::next_response_seq());
    let mut emitter = SseSafeEmitter::new(thinking);
    // F4: seed the streaming stop matcher from `stop_sequences`. Empty =>
    // verbatim passthrough (byte-identical); see the OpenAI `drive_chat_stream`
    // note for the worker/wire division of labour.
    let mut stop_matcher = StopMatcher::new(stop);
    let mut finish_reason: Option<FinishReason> = None;
    // `thinking` content block (Anthropic extended-thinking). Opened lazily on
    // the first reasoning delta, closed before the first text/tool block. Stays
    // closed on the thinking-off default path (no reasoning ever arrives).
    let mut thinking_block_open = false;
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
                // Reasoning trace -> a `thinking` content block, emitted BEFORE
                // any text/tool block. Opens lazily; never fires on the
                // thinking-off default path (delta.reasoning stays empty).
                if !delta.reasoning.is_empty() {
                    if !thinking_block_open {
                        let b = json!({
                            "type": "content_block_start",
                            "index": block_count,
                            "content_block": { "type": "thinking", "thinking": "" }
                        });
                        if tx.send(sse_event("content_block_start", &b.to_string())).await.is_err() {
                            return;
                        }
                        thinking_block_open = true;
                    }
                    let d = json!({
                        "type": "content_block_delta",
                        "index": block_count,
                        "delta": { "type": "thinking_delta", "thinking": delta.reasoning }
                    });
                    if tx.send(sse_event("content_block_delta", &d.to_string())).await.is_err() {
                        return;
                    }
                }
                let (safe_text, hit_stop) = stop_matcher.push(&delta.text);
                if !safe_text.is_empty() {
                    // The reasoning block (if any) must close before answer text.
                    if thinking_block_open {
                        let s = json!({ "type": "content_block_stop", "index": block_count });
                        if tx.send(sse_event("content_block_stop", &s.to_string())).await.is_err() {
                            return;
                        }
                        thinking_block_open = false;
                        block_count += 1;
                    }
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
                    // Close the reasoning block before the first tool block too
                    // (reasoning precedes all answer content).
                    if thinking_block_open {
                        let s = json!({ "type": "content_block_stop", "index": block_count });
                        if tx.send(sse_event("content_block_stop", &s.to_string())).await.is_err() {
                            return;
                        }
                        thinking_block_open = false;
                        block_count += 1;
                    }
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
                    // Wire-side stop (redundant safety net). Report StopSequence
                    // so Anthropic renders "stop_sequence".
                    finish_reason = Some(FinishReason::StopSequence);
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
    // Residual reasoning -> the (possibly still-open) thinking block. Empty on
    // the thinking-off default path.
    if !residual.reasoning.is_empty() {
        if !thinking_block_open {
            let b = json!({
                "type": "content_block_start",
                "index": block_count,
                "content_block": { "type": "thinking", "thinking": "" }
            });
            let _ = tx.send(sse_event("content_block_start", &b.to_string())).await;
            thinking_block_open = true;
        }
        let d = json!({
            "type": "content_block_delta",
            "index": block_count,
            "delta": { "type": "thinking_delta", "thinking": residual.reasoning }
        });
        let _ = tx.send(sse_event("content_block_delta", &d.to_string())).await;
    }
    // Drop the residual answer text once a stop sequence fired (post-stop
    // content); otherwise route the emitter residual through the stop matcher
    // (catches a stop straddling the emitter's held tail) and drain it. Empty
    // stop => `residual.text` verbatim + nothing, byte-identical to before.
    let stopped_by_sequence = finish_reason == Some(FinishReason::StopSequence);
    let final_text = if stopped_by_sequence {
        String::new()
    } else {
        let (residual_safe, _) = if stop_matcher.is_active() {
            stop_matcher.push(&residual.text)
        } else {
            (residual.text.clone(), false)
        };
        let mut t = residual_safe;
        t.push_str(&stop_matcher.finish());
        t
    };
    if !final_text.is_empty() {
        // Close the reasoning block before residual answer text.
        if thinking_block_open {
            let s = json!({ "type": "content_block_stop", "index": block_count });
            let _ = tx.send(sse_event("content_block_stop", &s.to_string())).await;
            thinking_block_open = false;
            block_count += 1;
        }
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
            "delta": { "type": "text_delta", "text": final_text }
        });
        let _ = tx.send(sse_event("content_block_delta", &d.to_string())).await;
    }
    // Close whichever block is still open (thinking-only reply, or text).
    if thinking_block_open || text_block_open {
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
    thinking: bool,
    stop: Vec<String>,
) -> Result<Value, ServerError> {
    let mut emitter = SseSafeEmitter::new(thinking);
    // F4: seed from `stop_sequences`. Empty => verbatim, byte-identical.
    let mut stop_matcher = StopMatcher::new(stop);
    let mut text = String::new();
    // Reasoning trace accumulated separately; surfaced as a `thinking` content
    // block placed BEFORE the text block. Empty on the thinking-off path.
    let mut reasoning = String::new();
    let mut tool_blocks: Vec<Value> = Vec::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut finish = FinishReason::Stop;

    while let Some(evt) = rx.recv().await {
        match evt {
            TokenEvent::PrefillDone { .. } => {}
            TokenEvent::Token { delta_text, .. } => {
                let delta = emitter.push(&delta_text);
                reasoning.push_str(&delta.reasoning);
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
                    // Wire-side stop match: report the Anthropic-correct
                    // stop_sequence reason (overrides the default end_turn).
                    finish = FinishReason::StopSequence;
                    break;
                }
            }
            TokenEvent::Done { finish_reason, prompt_tokens: p, completion_tokens: c } => {
                finish = finish_reason;
                prompt_tokens = p;
                completion_tokens = c;
                break;
            }
            // Classify: an oversize / empty-prompt runtime error from the
            // worker is a client 400 (context_length_exceeded), not a 500.
            TokenEvent::Error(msg) => return Err(ServerError::classify_runtime(msg)),
        }
    }
    let (residual, _) = emitter.finish();
    reasoning.push_str(&residual.reasoning);
    // Drop the residual answer text once a stop sequence fired (post-stop
    // content); otherwise stop-match + drain the emitter residual. Empty stop =>
    // verbatim append, byte-identical.
    if finish != FinishReason::StopSequence {
        let (residual_safe, _) = if stop_matcher.is_active() {
            stop_matcher.push(&residual.text)
        } else {
            (residual.text.clone(), false)
        };
        text.push_str(&residual_safe);
        text.push_str(&stop_matcher.finish());
    }

    if !tool_blocks.is_empty() && finish == FinishReason::Stop {
        finish = FinishReason::ToolCalls;
    }

    let mut content_blocks: Vec<Value> = Vec::new();
    // Thinking block first (when non-empty), then text, then tool blocks —
    // matching the streaming order. Omitted entirely on the thinking-off path.
    if !reasoning.is_empty() {
        content_blocks.push(json!({"type": "thinking", "thinking": reasoning}));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str) -> TokenEvent {
        TokenEvent::Token { token_id: 0, delta_text: text.to_string() }
    }

    /// Build an unpooled `JobResponseChannel` from a fixed event list (mirrors
    /// the OpenAI `collect_chat_from_events` helper).
    async fn collect_messages_from_events(
        events: Vec<TokenEvent>,
        thinking: bool,
    ) -> Value {
        collect_messages_from_events_with_stop(events, thinking, Vec::new()).await
    }

    async fn collect_messages_from_events_with_stop(
        events: Vec<TokenEvent>,
        thinking: bool,
        stop: Vec<String>,
    ) -> Value {
        let (tx, rx) = mpsc::channel(events.len().max(1));
        let return_sender = tx.clone();
        for e in events {
            tx.send(e).await.unwrap();
        }
        drop(tx);
        let pooled = crate::engine::PooledReceiver::new(rx, return_sender, None, 0, None);
        collect_messages(pooled, "test".into(), thinking, stop).await.unwrap()
    }

    fn user(text: &str) -> AnthropicMessage {
        AnthropicMessage { role: "user".into(), content: Value::String(text.into()) }
    }

    #[test]
    fn render_prompt_closed_think_tail_when_disabled() {
        // Reasoning off (default) MUST emit the closed empty-think tail —
        // byte-identical to the prior hardcoded Anthropic render AND to the
        // OpenAI / CLI user-only output.
        let out = render_prompt("", &[user("Hi")], false).unwrap();
        let expected =
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        assert_eq!(out, expected);
    }

    #[test]
    fn render_prompt_open_think_tail_when_enabled() {
        let out = render_prompt("", &[user("Hi")], true).unwrap();
        let expected = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        assert_eq!(out, expected);
    }

    #[test]
    fn render_prompt_with_system_closed_think_when_disabled() {
        let out = render_prompt("Sys", &[user("Hi")], false).unwrap();
        let expected = "<|im_start|>system\nSys<|im_end|>\n\
                        <|im_start|>user\nHi<|im_end|>\n\
                        <|im_start|>assistant\n<think>\n\n</think>\n\n";
        assert_eq!(out, expected);
    }

    #[test]
    fn resolve_thinking_maps_anthropic_config() {
        // type=="enabled" -> true; type=="disabled" -> false. Absent -> env/
        // default (false here, no env set in the test process).
        let enabled = MessagesRequest {
            model: "m".into(), messages: vec![], max_tokens: 1, system: None,
            temperature: None, top_p: None, top_k: None, stream: None,
            stop_sequences: vec![], tools: vec![],
            thinking: Some(ThinkingConfig { thinking_type: "enabled".into(), budget_tokens: None }),
        };
        assert!(enabled.resolve_thinking());
        let disabled = MessagesRequest {
            thinking: Some(ThinkingConfig { thinking_type: "disabled".into(), budget_tokens: None }),
            ..enabled.clone()
        };
        assert!(!disabled.resolve_thinking());
    }

    #[tokio::test]
    async fn collect_messages_thinking_off_has_no_thinking_block() {
        // Byte-identity guard: thinking off => NO thinking content block even
        // when the model literally emits </think>.
        let events = vec![
            tok("answer </think> still answer"),
            TokenEvent::Done { finish_reason: FinishReason::Stop, prompt_tokens: 1, completion_tokens: 4 },
        ];
        let resp = collect_messages_from_events(events, false).await;
        let blocks = resp["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "answer </think> still answer");
        assert!(blocks.iter().all(|b| b["type"] != "thinking"));
    }

    #[tokio::test]
    async fn collect_messages_thinking_on_emits_thinking_block_before_text() {
        let events = vec![
            tok("reasoning here</think>The answer."),
            TokenEvent::Done { finish_reason: FinishReason::Stop, prompt_tokens: 1, completion_tokens: 5 },
        ];
        let resp = collect_messages_from_events(events, true).await;
        let blocks = resp["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["thinking"], "reasoning here");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], "The answer.");
    }

    // ---- F4: wire-side stop matcher seeding (non-streaming messages) ----

    /// A seeded stop matcher strips the matched bytes and reports the
    /// Anthropic-correct `stop_reason:"stop_sequence"`.
    #[tokio::test]
    async fn collect_messages_wire_stop_truncates_and_reports_stop_sequence() {
        let events = vec![
            tok("visible HALT hidden"),
            TokenEvent::Done { finish_reason: FinishReason::Length, prompt_tokens: 1, completion_tokens: 3 },
        ];
        let resp =
            collect_messages_from_events_with_stop(events, false, vec!["HALT".into()]).await;
        let blocks = resp["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "visible ");
        assert_eq!(
            resp["stop_reason"], "stop_sequence",
            "a matched stop sequence must report stop_reason:stop_sequence"
        );
    }

    /// EMPTY stop list => inert matcher, full text verbatim, worker reason kept.
    #[tokio::test]
    async fn collect_messages_wire_empty_stop_is_byte_identical() {
        let full = "visible HALT hidden";
        let events = vec![
            tok(full),
            TokenEvent::Done { finish_reason: FinishReason::Length, prompt_tokens: 1, completion_tokens: 3 },
        ];
        let resp = collect_messages_from_events_with_stop(events, false, Vec::new()).await;
        let blocks = resp["content"].as_array().unwrap();
        assert_eq!(blocks[0]["text"], full, "empty stop passes full text through verbatim");
        assert_eq!(resp["stop_reason"], "max_tokens");
    }

    /// A stop straddling two token deltas is caught by the wire window buffer.
    #[tokio::test]
    async fn collect_messages_wire_stop_straddles_token_deltas() {
        let events = vec![
            tok("alpha HA"),
            tok("LT omega"),
            TokenEvent::Done { finish_reason: FinishReason::Length, prompt_tokens: 1, completion_tokens: 4 },
        ];
        let resp =
            collect_messages_from_events_with_stop(events, false, vec!["HALT".into()]).await;
        let blocks = resp["content"].as_array().unwrap();
        assert_eq!(blocks[0]["text"], "alpha ");
        assert_eq!(resp["stop_reason"], "stop_sequence");
    }

    // ---- F5: Anthropic-valid sampler subset (top_p, top_k) ----

    #[test]
    fn messages_request_with_top_p_top_k_deserializes_not_400() {
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "top_p": 0.9,
            "top_k": 50
        });
        let req: MessagesRequest = serde_json::from_value(body)
            .expect("Anthropic top_p/top_k must deserialize, not 400");
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.top_k, Some(50));
    }

    #[tokio::test]
    async fn messages_top_p_top_k_reach_sampling_params_via_into_job() {
        let engine = EngineHandle::new_for_test(4096);
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "top_p": 0.9, "top_k": 50
        });
        let req: MessagesRequest = serde_json::from_value(body).unwrap();
        let job = req.into_job(&engine).unwrap();
        assert_eq!(job.sampling.top_p, Some(0.9));
        assert_eq!(job.sampling.top_k, Some(50));
    }

    #[test]
    fn messages_unknown_field_still_400s_deny_unknown_fields_intact() {
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "definitely_not_a_field": 1
        });
        let r: Result<MessagesRequest, _> = serde_json::from_value(body);
        assert!(r.is_err(), "unknown top-level field must still be rejected");
    }

    // ---- F16(b): Anthropic synchronous oversize-prompt guard ----

    #[tokio::test]
    async fn messages_oversize_prompt_returns_400() {
        let engine = EngineHandle::new_for_test(8);
        let long = "x".repeat(500);
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": long}],
            "max_tokens": 16
        });
        let req: MessagesRequest = serde_json::from_value(body).unwrap();
        let err = req.into_job(&engine).expect_err("oversize prompt must 400");
        match err {
            ServerError::BadRequest { code, .. } => {
                assert_eq!(code.as_deref(), Some("context_length_exceeded"));
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    // ---- F7: shared content-parts flattener (number content -> 400) ----

    #[tokio::test]
    async fn messages_numeric_content_is_rejected_robust007() {
        // The OpenAI ROBUST-007 guard now also applies on the Anthropic path:
        // a bare-number content must 400, not be coerced via to_string().
        let engine = EngineHandle::new_for_test(4096);
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": 42}],
            "max_tokens": 16
        });
        let req: MessagesRequest = serde_json::from_value(body).unwrap();
        let err = req.into_job(&engine).expect_err("numeric content must 400");
        match err {
            ServerError::BadRequest { code, .. } => {
                assert_eq!(code.as_deref(), Some("invalid_type"));
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    // ---- F6: Anthropic now CONSUMES a tool round-trip (renders the markers) ----

    #[test]
    fn anthropic_renders_tool_use_and_tool_result_blocks() {
        // assistant {type:tool_use,name,input} -> <tool_call> segment;
        // user {type:tool_result,content} -> <tool_response> turn. Previously
        // these blocks were dropped entirely.
        let messages = vec![
            AnthropicMessage {
                role: "assistant".into(),
                content: serde_json::json!([
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "name": "get_weather", "input": {"city": "Paris"}}
                ]),
            },
            AnthropicMessage {
                role: "user".into(),
                content: serde_json::json!([
                    {"type": "tool_result", "content": "{\"temp\": 18}"}
                ]),
            },
        ];
        let out = render_prompt("", &messages, false).unwrap();
        assert!(out.contains("<tool_call>"), "tool_use must render a tool_call: {out}");
        assert!(out.contains("get_weather"), "tool name present");
        // `input` object serializes COMPACT via serde_json::Value::to_string().
        assert!(out.contains("{\"city\":\"Paris\"}"), "input serialized as compact arguments");
        assert!(out.contains("<tool_response>"), "tool_result must render a tool_response");
        assert!(out.contains("{\"temp\": 18}"), "tool result content present");
    }
}
