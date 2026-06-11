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

/// vLLM-/SGLang-compatible `chat_template_kwargs`. The only field Lumen reads
/// is `enable_thinking`; any other keys pass through and are ignored (the
/// struct is permissive — NOT `deny_unknown_fields` — so forward-compatible
/// extras don't 400). This mirrors the vLLM OpenAI server, which accepts
/// `{"chat_template_kwargs": {"enable_thinking": false}}` to toggle the
/// Qwen3.5 reasoning block.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatTemplateKwargs {
    #[serde(default)]
    pub enable_thinking: Option<bool>,
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
    /// Nucleus-sampling cutoff (OpenAI `top_p`). Honored on the CLI today but
    /// previously HTTP-400-rejected here by `deny_unknown_fields`. `None`
    /// (omitted) leaves the sampler default untouched.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k logit cut. Not a standard OpenAI field, but vLLM/llama.cpp accept
    /// it and the CLI honors it; carried here for surface parity.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Min-p relative cutoff. As with `top_k`, CLI-honored and accepted by
    /// vLLM/llama.cpp.
    #[serde(default)]
    pub min_p: Option<f32>,
    /// OpenAI `presence_penalty`. Zero-normalized to `None` (CLI parity) so an
    /// explicit `0` stays a no-op identical to the default path.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// OpenAI `frequency_penalty`. Zero-normalized to `None` (CLI parity);
    /// when supplied non-zero it overrides the server-internal
    /// `diag_frequency_penalty` default.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub tools: Vec<ToolDef>,
    /// Per-request reasoning toggle. `Some(true)` opens the `<think>` block so
    /// the model emits a reasoning trace (surfaced as `reasoning_content`);
    /// `Some(false)` forces the closed empty-think tail. `None` defers to the
    /// `LUMEN_CHAT_ENABLE_THINKING` env override, then the process default
    /// (`false`). Resolved via the shared
    /// [`lumen_runtime::runtime_defaults::resolve_enable_thinking`].
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// Separate reasoning-token budget (industry-convergent with Anthropic
    /// `thinking.budget_tokens` / Gemini `thinking_budget`). Carried on the
    /// request DTO now; the decode-loop enforcement is Part 4 (separate work).
    #[serde(default)]
    pub reasoning_budget: Option<usize>,
    /// vLLM-compatible `{"chat_template_kwargs": {"enable_thinking": ...}}`.
    /// The top-level `enable_thinking` field wins when both are present.
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
}

impl ChatCompletionRequest {
    /// Resolve the per-request reasoning toggle using the single shared
    /// resolver. Precedence: top-level `enable_thinking` → vLLM
    /// `chat_template_kwargs.enable_thinking` → env override → default. The
    /// per-request `Option` is collapsed here, then handed to the one resolver
    /// so the env/default fall-through is identical to every other surface.
    pub fn resolve_thinking(&self) -> bool {
        let per_request = self.enable_thinking.or_else(|| {
            self.chat_template_kwargs
                .as_ref()
                .and_then(|k| k.enable_thinking)
        });
        super::resolve_enable_thinking(per_request)
    }

    pub fn into_job(self, engine: &EngineHandle) -> Result<JobRequest, ServerError> {
        // ROBUST-007 (2026-06-11 checklist): out-of-range sampler params and
        // empty `messages` must 400 like other malformed fields, not be
        // silently accepted/clamped.
        validate_sampler_ranges(self.temperature, self.top_p)?;
        if self.messages.is_empty() {
            return Err(ServerError::bad_request_field(
                "messages must be a non-empty array",
                "messages",
                "invalid_value",
            ));
        }
        let enable_thinking = self.resolve_thinking();
        let prompt = render_chat_prompt(&self.messages, &self.tools, enable_thinking)?;
        let prompt_tokens = engine.tokenize_for_request(&prompt);
        // Synchronous oversize guard: 400 BEFORE the 200/SSE stream opens.
        super::check_prompt_length(prompt_tokens.len(), engine.context_length())?;
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
        // Client-supplied additive penalties are zero-normalized to `None`
        // (CLI parity, run.rs:357,368): an explicit `0` is a no-op identical
        // to the default path. `frequency_penalty` OVERRIDES the
        // server-internal `diag_frequency_penalty` default ONLY when the
        // client sends a non-zero value; otherwise the default stands so the
        // all-zero / omitted request stays byte-identical to today.
        let presence_penalty = super::normalize_zero_penalty(self.presence_penalty);
        let frequency_penalty = super::normalize_zero_penalty(self.frequency_penalty)
            .unwrap_or_else(super::diag_frequency_penalty);
        let sampling = SamplingParams {
            temperature: self
                .temperature
                .unwrap_or_else(lumen_runtime::runtime_defaults::default_temperature),
            seed: Some(self.seed.unwrap_or_else(super::next_random_seed)),
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repetition_penalty: Some(super::diag_repetition_penalty()),
            presence_penalty,
            frequency_penalty: Some(frequency_penalty),
            repeat_last_n: super::diag_repeat_last_n(),
            anti_restate: super::diag_anti_restate(),
            ..Default::default()
        };
        Ok(JobRequest {
            prompt_tokens,
            max_tokens,
            stop_text,
            eos_token_ids: eos,
            sampling,
            suffix_threshold: lumen_runtime::session::Session::DEFAULT_SUFFIX_THRESHOLD,
            enable_thinking,
            reasoning_budget: self
                .reasoning_budget
                .unwrap_or_else(lumen_runtime::runtime_defaults::chat_reasoning_budget_default),
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
    /// Mirror of the OpenAI-valid sampler set carried on chat completions
    /// (see `ChatCompletionRequest`): honored on the CLI, previously
    /// 400-rejected here by `deny_unknown_fields`. Same zero-normalization /
    /// override semantics as the chat path.
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Value>,
}

impl CompletionRequest {
    pub fn into_job(self, engine: &EngineHandle) -> Result<JobRequest, ServerError> {
        // ROBUST-007: same sampler-range guard as the chat endpoint.
        validate_sampler_ranges(self.temperature, self.top_p)?;
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
        // Synchronous oversize guard: 400 BEFORE the 200/SSE stream opens.
        super::check_prompt_length(prompt_tokens.len(), engine.context_length())?;
        let stop_text = parse_stop_field(self.stop);
        let eos = engine.eos_tokens_for_request();
        let max_tokens = self.max_tokens.unwrap_or(256);
        // server-internal sampler defaults (see
        // ChatCompletionRequest::into_job for the full rationale). An omitted
        // `seed` resolves to a fresh per-request random seed; pass an explicit
        // `seed` for reproducible output.
        // Same CLI-parity zero-normalization + frequency-penalty override as
        // the chat path (see `ChatCompletionRequest::into_job`).
        let presence_penalty = super::normalize_zero_penalty(self.presence_penalty);
        let frequency_penalty = super::normalize_zero_penalty(self.frequency_penalty)
            .unwrap_or_else(super::diag_frequency_penalty);
        let sampling = SamplingParams {
            temperature: self
                .temperature
                .unwrap_or_else(lumen_runtime::runtime_defaults::default_temperature),
            seed: Some(self.seed.unwrap_or_else(super::next_random_seed)),
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repetition_penalty: Some(super::diag_repetition_penalty()),
            presence_penalty,
            frequency_penalty: Some(frequency_penalty),
            repeat_last_n: super::diag_repeat_last_n(),
            anti_restate: super::diag_anti_restate(),
            ..Default::default()
        };
        Ok(JobRequest {
            prompt_tokens,
            max_tokens,
            stop_text,
            eos_token_ids: eos,
            sampling,
            suffix_threshold: lumen_runtime::session::Session::DEFAULT_SUFFIX_THRESHOLD,
            // Legacy text-completions have no chat template / `<think>` block,
            // so reasoning is never enabled on this path.
            enable_thinking: false,
            reasoning_budget: 0,
        })
    }
}

/// ROBUST-007 (2026-06-11 production checklist): reject out-of-range sampler
/// parameters with HTTP 400 (OpenAI-spec ranges: `temperature` in [0, 2],
/// `top_p` in [0, 1]) instead of silently accepting/clamping. NaN rejected.
fn validate_sampler_ranges(
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> Result<(), ServerError> {
    if let Some(t) = temperature {
        if !t.is_finite() || !(0.0..=2.0).contains(&t) {
            return Err(ServerError::bad_request_field(
                "temperature must be between 0 and 2",
                "temperature",
                "invalid_value",
            ));
        }
    }
    if let Some(p) = top_p {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(ServerError::bad_request_field(
                "top_p must be between 0 and 1",
                "top_p",
                "invalid_value",
            ));
        }
    }
    Ok(())
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
/// Emit the Qwen3.5 assistant prompt tail selected by `enable_thinking`:
/// the closed empty-think tail (`<think>\n\n</think>\n\n`) when `false` so the
/// model answers directly (matching the CLI's `enable_thinking=false` chat
/// template), or the OPEN `<think>\n` tail when `true` so the model emits a
/// reasoning trace (surfaced as `reasoning_content` by the
/// [`crate::sse::SseSafeEmitter`]). The open/closed string is chosen by the
/// single shared [`lumen_runtime::runtime_defaults::think_prompt_tail`] helper
/// so the CLI, OpenAI, and Anthropic surfaces cannot drift.
///
/// The tail is a no-op for non-Qwen3.5 ChatML models that do not treat
/// `<think>`/`</think>` as special tokens — they render as literal text and
/// are stripped by the wire layer's StopMatcher / SseSafeEmitter only on
/// Qwen3.5 (because only Qwen3.5's special-token map contains them).
fn render_chat_prompt(
    messages: &[ChatMessage],
    tools: &[ToolDef],
    enable_thinking: bool,
) -> Result<String, ServerError> {
    let mut system: Option<String> = None;
    let mut transcript = String::new();
    for m in messages.iter() {
        // `flatten_content` enforces the ROBUST-007 numeric-type-guard (a bare
        // number/bool `content` 400s instead of being coerced) AND flattens
        // content-parts via the single shared key set — the SAME helper the
        // Anthropic surface uses, so the two cannot diverge.
        match m.role.as_str() {
            "system" => system = Some(super::flatten_content(&m.content, "messages.content")?),
            "user" => {
                transcript.push_str("<|im_start|>user\n");
                transcript.push_str(&super::flatten_content(&m.content, "messages.content")?);
                transcript.push_str("<|im_end|>\n");
            }
            "assistant" => {
                transcript.push_str("<|im_start|>assistant\n");
                transcript.push_str(&super::flatten_content(&m.content, "messages.content")?);
                // Tool calls render through the shared tooling helper so the
                // assistant tool-call transcript is byte-identical to the one
                // the Anthropic surface emits for an equivalent round-trip.
                for tc in &m.tool_calls {
                    transcript.push_str(&lumen_runtime::tooling::render_assistant_tool_call_segment(
                        &tc.function.name,
                        &tc.function.arguments,
                    ));
                }
                transcript.push_str("<|im_end|>\n");
            }
            "tool" => {
                // Shared tool-response turn (single source of truth in tooling).
                transcript.push_str(&lumen_runtime::tooling::render_tool_response_turn(
                    &super::flatten_content(&m.content, "messages.content")?,
                ));
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
    // Open vs closed `<think>` tail comes from the single shared resolver/
    // helper (see `ChatCompletionRequest::resolve_thinking`). The former
    // OpenAI-only inline `LUMEN_CHAT_ENABLE_THINKING == "1"` check is GONE —
    // the env override now lives in `resolve_enable_thinking` so the CLI and
    // both wire formats honour it identically.
    prompt.push_str("<|im_start|>assistant\n");
    prompt.push_str(lumen_runtime::runtime_defaults::think_prompt_tail(enable_thinking));
    Ok(prompt)
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

pub fn stream_chat(
    rx: JobResponseChannel,
    model: String,
    created: u64,
    thinking: bool,
    stop: Vec<String>,
) -> Body {
    let (tx, body_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    tokio::spawn(drive_chat_stream(rx, tx, model, created, true, thinking, stop));
    body_from_byte_stream(body_rx)
}

pub fn stream_completion(
    rx: JobResponseChannel,
    model: String,
    created: u64,
    stop: Vec<String>,
) -> Body {
    let (tx, body_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    // Legacy completions have no chat template / `<think>` block: thinking is
    // always off, so the emitter's reasoning stage is a passthrough.
    tokio::spawn(drive_chat_stream(rx, tx, model, created, false, false, stop));
    body_from_byte_stream(body_rx)
}

async fn drive_chat_stream(
    mut rx: JobResponseChannel,
    tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    model: String,
    created: u64,
    chat: bool,
    thinking: bool,
    stop: Vec<String>,
) {
    let id = format!("chatcmpl-lumen-{created:x}-{:x}", super::next_response_seq());
    let mut emitter = SseSafeEmitter::new(thinking);
    // F4: seed the streaming stop matcher from the request stop list. The
    // worker already truncates generation at the stop string (and reports
    // `FinishReason::StopSequence`, which it forwards via `TokenEvent::Done`);
    // this wire-side matcher is the redundant safety net that strips any stop
    // bytes the worker forwarded and keeps the OpenAI semantics (matched bytes
    // never reach the client). When `stop` is empty the matcher passes every
    // fragment through verbatim — byte-identical.
    let mut stop_matcher = StopMatcher::new(stop);
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
                // Reasoning trace (chat only): emit `delta.reasoning_content`
                // chunks BEFORE answer content. The trace bypasses the stop
                // matcher (stop sequences apply to the answer, not the trace).
                // `delta.reasoning` is always empty when thinking is off, so
                // this block never fires on the default path.
                if chat && !delta.reasoning.is_empty() {
                    let frame = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": { "reasoning_content": delta.reasoning },
                            "finish_reason": null
                        }],
                    });
                    if tx.send(sse_frame(&frame.to_string())).await.is_err() {
                        return;
                    }
                }
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
                    // Wire-side stop (redundant safety net; the worker normally
                    // hits it first and sends Done{StopSequence}). Report
                    // StopSequence so OpenAI renders "stop".
                    finish_reason = Some(FinishReason::StopSequence);
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
    // Flush any residual reasoning trace (chat only) before the residual
    // answer text. Empty on the thinking-off default path.
    if chat && !residual.reasoning.is_empty() {
        let frame = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": residual.reasoning },
                "finish_reason": null
            }],
        });
        if tx.send(sse_frame(&frame.to_string())).await.is_err() {
            return;
        }
    }
    // Residual ANSWER text. Once a stop sequence has fired, EVERYTHING from the
    // stop onward is dropped — including any tail the emitter was still holding
    // (which is post-stop content) — so we skip the residual entirely in that
    // case. Otherwise route the emitter residual through the stop matcher (a
    // stop could straddle the emitter's held tail) and drain the matcher. With
    // an empty stop list this is `(residual.text, "")`: byte-identical.
    let stopped_by_sequence = finish_reason == Some(FinishReason::StopSequence);
    let final_content = if stopped_by_sequence {
        String::new()
    } else {
        let (residual_safe, _residual_hit) = if stop_matcher.is_active() {
            stop_matcher.push(&residual.text)
        } else {
            (residual.text.clone(), false)
        };
        let mut c = residual_safe;
        c.push_str(&stop_matcher.finish());
        c
    };
    if !final_content.is_empty() {
        let frame = if chat {
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": { "content": final_content },
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
                    "text": final_content,
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
    thinking: bool,
    stop: Vec<String>,
) -> Result<Value, ServerError> {
    let mut emitter = SseSafeEmitter::new(thinking);
    // F4: seed from the request stop list (see `drive_chat_stream`). Empty =>
    // verbatim passthrough, byte-identical to the pre-F4 response.
    let mut stop_matcher = StopMatcher::new(stop);
    let mut content = String::new();
    // Reasoning trace accumulated separately from `content`; surfaced as
    // `reasoning_content` (omitted when empty, i.e. on the thinking-off path).
    let mut reasoning = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
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
                    // Wire-side stop match -> StopSequence (renders OpenAI "stop").
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
    // Once a stop sequence fired, drop the residual answer text (post-stop
    // content). Otherwise pass the emitter residual through the stop matcher
    // (catches a stop straddling the emitter's held tail) and drain it. Empty
    // stop => appends `residual.text` verbatim + nothing, byte-identical.
    if finish != FinishReason::StopSequence {
        let (residual_safe, _) = if stop_matcher.is_active() {
            stop_matcher.push(&residual.text)
        } else {
            (residual.text.clone(), false)
        };
        content.push_str(&residual_safe);
        content.push_str(&stop_matcher.finish());
    }

    if !tool_calls.is_empty() && finish == FinishReason::Stop {
        finish = FinishReason::ToolCalls;
    }

    let mut msg = if tool_calls.is_empty() {
        json!({ "role": "assistant", "content": content })
    } else {
        json!({
            "role": "assistant",
            "content": if content.is_empty() { Value::Null } else { Value::String(content) },
            "tool_calls": tool_calls,
        })
    };
    // Attach the reasoning trace as `reasoning_content`, OMITTED when empty so
    // the thinking-off default response is byte-identical to before.
    if !reasoning.is_empty() {
        if let Value::Object(ref mut map) = msg {
            map.insert("reasoning_content".to_string(), Value::String(reasoning));
        }
    }
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
/// `thinking` selects whether the emitter splits a `<think>` trace into
/// `reasoning_content` (mirrors a request with `enable_thinking=true`).
#[cfg(any(test, doctest))]
pub async fn collect_chat_from_events(
    events: Vec<TokenEvent>,
    model: String,
    created: u64,
    thinking: bool,
) -> Result<Value, ServerError> {
    collect_chat_from_events_with_stop(events, model, created, thinking, Vec::new()).await
}

/// `collect_chat_from_events` with an explicit stop list, so F4 stop-truncation
/// tests can exercise the seeded wire-side matcher without a live engine.
#[cfg(any(test, doctest))]
pub async fn collect_chat_from_events_with_stop(
    events: Vec<TokenEvent>,
    model: String,
    created: u64,
    thinking: bool,
    stop: Vec<String>,
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
    collect_chat(pooled, model, created, thinking, stop).await
}

pub async fn collect_completion(
    mut rx: JobResponseChannel,
    model: String,
    created: u64,
    stop: Vec<String>,
) -> Result<Value, ServerError> {
    // Legacy completions have no `<think>` block: thinking off (passthrough).
    let mut emitter = SseSafeEmitter::new(false);
    // F4: seed from the request stop list. Empty => verbatim, byte-identical.
    let mut stop_matcher = StopMatcher::new(stop);
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
                    // Wire-side stop match -> StopSequence (renders OpenAI "stop").
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
    if finish != FinishReason::StopSequence {
        let (residual_safe, _) = if stop_matcher.is_active() {
            stop_matcher.push(&residual.text)
        } else {
            (residual.text.clone(), false)
        };
        text.push_str(&residual_safe);
        text.push_str(&stop_matcher.finish());
    }

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
        let resp = collect_chat_from_events(events, "test-model".into(), 1234, false).await.unwrap();
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
        let resp = collect_chat_from_events(events, "test".into(), 1, false).await.unwrap();
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
        let resp = collect_chat_from_events(events, "test".into(), 1, false).await.unwrap();
        assert_eq!(resp["choices"][0]["finish_reason"], "length");
    }

    // ---- F4: wire-side stop matcher seeding (non-streaming chat) ----

    /// A seeded stop matcher strips the matched bytes (and everything after) and
    /// renders `finish_reason:"stop"`. Models the case where the worker forwarded
    /// text containing the stop (the wire net catches it); the worker's own
    /// truncation is covered by the engine-direct integration tests.
    #[tokio::test]
    async fn collect_chat_wire_stop_truncates_and_reports_stop() {
        let events = vec![
            tok("keep this STOP drop this"),
            TokenEvent::Done {
                finish_reason: FinishReason::Length,
                prompt_tokens: 1,
                completion_tokens: 9,
            },
        ];
        let resp = collect_chat_from_events_with_stop(
            events,
            "test".into(),
            1,
            false,
            vec!["STOP".into()],
        )
        .await
        .unwrap();
        assert_eq!(resp["choices"][0]["message"]["content"], "keep this ");
        // Wire-side stop overrides the worker's Length: a matched stop string wins.
        assert_eq!(resp["choices"][0]["finish_reason"], "stop");
    }

    /// EMPTY stop list => the seeded matcher is inert and the content is the full
    /// text byte-for-byte, with the worker's finish reason untouched. This is the
    /// wire-layer mirror of the engine-direct byte-identity invariant.
    #[tokio::test]
    async fn collect_chat_wire_empty_stop_is_byte_identical() {
        let full = "keep this STOP drop this";
        let events = vec![
            tok(full),
            TokenEvent::Done {
                finish_reason: FinishReason::Length,
                prompt_tokens: 1,
                completion_tokens: 9,
            },
        ];
        // Same events, empty stop list.
        let resp = collect_chat_from_events_with_stop(events, "test".into(), 1, false, Vec::new())
            .await
            .unwrap();
        assert_eq!(
            resp["choices"][0]["message"]["content"], full,
            "empty stop must pass the full text through verbatim"
        );
        assert_eq!(resp["choices"][0]["finish_reason"], "length");
    }

    /// Post-stop content the EMITTER might hold (a trailing `<` that looks like a
    /// possible `<tool_call>` prefix) must NOT leak after a stop hit. Guards the
    /// residual-drop rule in `collect_chat`.
    #[tokio::test]
    async fn collect_chat_wire_no_post_stop_residual_leak() {
        let events = vec![
            // The whole answer arrives in one delta; "STOP" is mid-string and the
            // text ends with a `<` that the emitter would otherwise hold back.
            tok("visible STOP hidden <"),
            TokenEvent::Done {
                finish_reason: FinishReason::Length,
                prompt_tokens: 1,
                completion_tokens: 4,
            },
        ];
        let resp = collect_chat_from_events_with_stop(
            events,
            "test".into(),
            1,
            false,
            vec!["STOP".into()],
        )
        .await
        .unwrap();
        assert_eq!(
            resp["choices"][0]["message"]["content"], "visible ",
            "no post-stop bytes (not even an emitter-held trailing '<') may leak"
        );
        assert_eq!(resp["choices"][0]["finish_reason"], "stop");
    }

    /// A stop that straddles two token deltas is still caught by the wire matcher
    /// (window buffering), proving the seeded matcher handles split fragments.
    #[tokio::test]
    async fn collect_chat_wire_stop_straddles_token_deltas() {
        let events = vec![
            tok("alpha ST"),
            tok("OP omega"),
            TokenEvent::Done {
                finish_reason: FinishReason::Length,
                prompt_tokens: 1,
                completion_tokens: 4,
            },
        ];
        let resp = collect_chat_from_events_with_stop(
            events,
            "test".into(),
            1,
            false,
            vec!["STOP".into()],
        )
        .await
        .unwrap();
        assert_eq!(resp["choices"][0]["message"]["content"], "alpha ");
        assert_eq!(resp["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn collect_chat_passes_through_error_event() {
        let events = vec![TokenEvent::Error("model exploded".into())];
        let r = collect_chat_from_events(events, "test".into(), 1, false).await;
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
    fn render_chat_prompt_user_only_emits_closed_think_when_disabled() {
        // enable_thinking=false (the default) MUST emit the closed empty-think
        // tail, byte-identical to the pre-reasoning-control behaviour.
        let messages = vec![user_msg("Hello")];
        let out = render_chat_prompt(&messages, &[], false).unwrap();
        // CLI's `apply_chat_template_with_system("Hello", None)` for qwen35
        // post- produces exactly this string (see crates/lumen-cli
        // /src/tokenize.rs:273-292).
        let expected =
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        assert_eq!(out, expected, "render_chat_prompt user-only != CLI output");
    }

    #[test]
    fn render_chat_prompt_user_only_emits_open_think_when_enabled() {
        // enable_thinking=true MUST emit the OPEN `<think>\n` tail so the
        // model produces a reasoning trace.
        let messages = vec![user_msg("Hello")];
        let out = render_chat_prompt(&messages, &[], true).unwrap();
        let expected = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n";
        assert_eq!(out, expected, "render_chat_prompt user-only enabled != open think tail");
    }

    #[test]
    fn render_chat_prompt_system_plus_user_emits_closed_think_when_disabled() {
        let messages = vec![system_msg("You are helpful."), user_msg("Hi")];
        let out = render_chat_prompt(&messages, &[], false).unwrap();
        let expected = "<|im_start|>system\nYou are helpful.<|im_end|>\n\
                        <|im_start|>user\nHi<|im_end|>\n\
                        <|im_start|>assistant\n<think>\n\n</think>\n\n";
        assert_eq!(out, expected, "render_chat_prompt system+user != CLI output");
    }

    #[test]
    fn render_chat_prompt_system_plus_user_emits_open_think_when_enabled() {
        let messages = vec![system_msg("You are helpful."), user_msg("Hi")];
        let out = render_chat_prompt(&messages, &[], true).unwrap();
        let expected = "<|im_start|>system\nYou are helpful.<|im_end|>\n\
                        <|im_start|>user\nHi<|im_end|>\n\
                        <|im_start|>assistant\n<think>\n";
        assert_eq!(out, expected, "render_chat_prompt system+user enabled != open think tail");
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
        let out = render_chat_prompt(&messages, &[], false).unwrap();
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
        let server_out = render_chat_prompt(&messages, &[], false).unwrap();
        let cli_out = format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt = "Hello"
        );
        assert_eq!(server_out, cli_out, "server render must match CLI render byte-for-byte");
    }

    // ---- reasoning_content extraction (non-stream collect_chat) ----

    #[tokio::test]
    async fn collect_chat_thinking_off_has_no_reasoning_content() {
        // Byte-identity guard: thinking off => the message has NO
        // reasoning_content key even if the model literally emits </think>.
        let events = vec![
            tok("plain answer </think> still answer"),
            TokenEvent::Done { finish_reason: FinishReason::Stop, prompt_tokens: 1, completion_tokens: 4 },
        ];
        let resp = collect_chat_from_events(events, "test".into(), 1, false).await.unwrap();
        let msg = &resp["choices"][0]["message"];
        assert!(msg.get("reasoning_content").is_none(), "no reasoning_content when thinking off");
        assert_eq!(msg["content"], "plain answer </think> still answer");
    }

    #[tokio::test]
    async fn collect_chat_thinking_on_splits_reasoning_content() {
        let events = vec![
            tok("let me think"),
            tok(" carefully</think>The answer is 42."),
            TokenEvent::Done { finish_reason: FinishReason::Stop, prompt_tokens: 1, completion_tokens: 6 },
        ];
        let resp = collect_chat_from_events(events, "test".into(), 1, true).await.unwrap();
        let msg = &resp["choices"][0]["message"];
        assert_eq!(msg["reasoning_content"], "let me think carefully");
        assert_eq!(msg["content"], "The answer is 42.");
    }

    // ---- F5: standard sampler params accepted (no 400) + reach SamplingParams ----

    #[test]
    fn chat_request_with_sampler_params_deserializes_not_400() {
        // Previously these fields tripped `deny_unknown_fields` -> HTTP 400.
        // They must now deserialize cleanly onto the DTO.
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.7
        });
        let req: ChatCompletionRequest = serde_json::from_value(body)
            .expect("sampler params must deserialize, not 400");
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.top_k, Some(40));
        assert_eq!(req.min_p, Some(0.05));
        assert_eq!(req.presence_penalty, Some(0.5));
        assert_eq!(req.frequency_penalty, Some(0.7));
    }

    #[tokio::test]
    async fn chat_sampler_params_reach_sampling_params_via_into_job() {
        let engine = EngineHandle::new_for_test(4096);
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.9, "top_k": 40, "min_p": 0.05,
            "presence_penalty": 0.5, "frequency_penalty": 0.7
        });
        let req: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        let job = req.into_job(&engine).unwrap();
        assert_eq!(job.sampling.top_p, Some(0.9));
        assert_eq!(job.sampling.top_k, Some(40));
        assert_eq!(job.sampling.min_p, Some(0.05));
        assert_eq!(job.sampling.presence_penalty, Some(0.5));
        // A supplied non-zero frequency_penalty OVERRIDES the diag default.
        assert_eq!(job.sampling.frequency_penalty, Some(0.7));
    }

    #[tokio::test]
    async fn chat_all_zero_penalties_normalize_to_default_path() {
        // CLI parity: presence/frequency == 0.0 -> None (no-op). The all-zero
        // request must be byte-identical to omitting the fields: presence_penalty
        // None, frequency_penalty == the server-internal diag default.
        let engine = EngineHandle::new_for_test(4096);
        let zero_body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "presence_penalty": 0.0, "frequency_penalty": 0.0
        });
        let omitted_body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let zero_req: ChatCompletionRequest = serde_json::from_value(zero_body).unwrap();
        let omitted_req: ChatCompletionRequest = serde_json::from_value(omitted_body).unwrap();
        let zero_job = zero_req.into_job(&engine).unwrap();
        let omitted_job = omitted_req.into_job(&engine).unwrap();
        assert_eq!(zero_job.sampling.presence_penalty, None, "zero presence -> None");
        assert_eq!(
            zero_job.sampling.frequency_penalty, omitted_job.sampling.frequency_penalty,
            "all-zero freq penalty must equal the omitted (diag-default) path"
        );
        assert_eq!(zero_job.sampling.top_p, None);
        assert_eq!(zero_job.sampling.top_k, None);
    }

    #[test]
    fn completion_request_with_sampler_params_deserializes_not_400() {
        let body = serde_json::json!({
            "model": "m",
            "prompt": "hi",
            "top_p": 0.8, "presence_penalty": 0.3, "frequency_penalty": 0.4
        });
        let req: CompletionRequest = serde_json::from_value(body)
            .expect("completion sampler params must deserialize, not 400");
        assert_eq!(req.top_p, Some(0.8));
        assert_eq!(req.presence_penalty, Some(0.3));
        assert_eq!(req.frequency_penalty, Some(0.4));
    }

    #[test]
    fn unknown_field_still_400s_deny_unknown_fields_intact() {
        // Guard: adding the sampler fields must NOT loosen deny_unknown_fields.
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "definitely_not_a_field": 1
        });
        let r: Result<ChatCompletionRequest, _> = serde_json::from_value(body);
        assert!(r.is_err(), "unknown top-level field must still be rejected");
    }

    // ---- F16(b): synchronous oversize-prompt guard returns 400 in into_job ----

    #[tokio::test]
    async fn chat_oversize_prompt_returns_400_before_submit() {
        // context_length is tiny; the IdentityByteTokenizer maps 1 byte -> 1
        // token, so a long content overflows it. into_job must 400 (NOT 500,
        // NOT 200) BEFORE any stream/submit.
        let engine = EngineHandle::new_for_test(8);
        let long = "x".repeat(500);
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": long}]
        });
        let req: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        let err = req.into_job(&engine).expect_err("oversize prompt must 400");
        match err {
            ServerError::BadRequest { code, message, .. } => {
                assert_eq!(code.as_deref(), Some("context_length_exceeded"));
                assert!(message.contains("max_seq_len is"), "msg carries the sentinel: {message}");
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_within_context_length_is_ok() {
        // A short prompt under the window must NOT be rejected by the guard.
        let engine = EngineHandle::new_for_test(4096);
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        assert!(req.into_job(&engine).is_ok(), "short prompt must pass the guard");
    }
}

