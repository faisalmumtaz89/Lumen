//! Anthropic-compatible `/v1/messages` endpoint.
//!
//! Reference: <https://docs.anthropic.com/en/api/messages>
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
use lumen_runtime::tooling::{
    compose_system_with_tools, StreamEvent, ToolSchema, ToolSchemas, ToolStreamEvent,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::engine::{EngineHandle, FinishReason, JobRequest, JobResponseChannel, TokenEvent};
use crate::error::ServerError;
use crate::sse::{ReplyTools, SseSafeEmitter};
use crate::tokenstop::StopMatcher;

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTool {
    /// `custom` (the default) is a tool the client defines and runs. Other
    /// types are Anthropic-defined tools (web search, code execution, bash,
    /// text editor, ...), whose schemas this server does not know; they are
    /// refused.
    #[serde(default, rename = "type")]
    pub tool_type: Option<String>,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub input_schema: Value,
}

/// Anthropic extended-thinking config: `{"type": "enabled"|"adaptive"|"disabled",
/// "budget_tokens": N}`. Other keys, such as `display`, are ignored. An unknown
/// `type` is refused rather than read as off.
#[derive(Debug, Clone, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: ThinkingType,
    #[serde(default)]
    pub budget_tokens: Option<usize>,
}

/// `thinking.type`. `Adaptive` leaves the decision to think to the model; the
/// model always opens a reasoning block, so it is served as `Enabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
pub enum ThinkingType {
    Enabled,
    Adaptive,
    Disabled,
}

impl TryFrom<String> for ThinkingType {
    type Error = String;

    fn try_from(value: String) -> Result<Self, String> {
        match value.as_str() {
            "enabled" => Ok(Self::Enabled),
            "adaptive" => Ok(Self::Adaptive),
            "disabled" => Ok(Self::Disabled),
            _ => Err("unknown thinking type, expected `enabled`, `adaptive` or `disabled`".into()),
        }
    }
}

/// A `/v1/messages` request. Fields this server does not use (`metadata`,
/// `context_management`, and the others Anthropic clients attach) are ignored,
/// not rejected; a structured-output request (`output_config.format`) is
/// refused.
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    /// Required by `/v1/messages` (see [`Self::into_job`]); not by
    /// `/v1/messages/count_tokens`.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub system: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Anthropic-valid sampler subset. The Messages API exposes `top_p` and
    /// `top_k` (NOT presence/frequency penalties); both are honored as on the
    /// CLI. `None` (omitted) leaves the sampler default untouched.
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
    /// Anthropic extended-thinking control. `Some({type:"enabled"})` or
    /// `Some({type:"adaptive"})` opens the `<think>` block (reasoning surfaced
    /// as a `thinking` content block);
    /// `Some({type:"disabled"})` forces it closed; `None` defers to the
    /// `LUMEN_CHAT_ENABLE_THINKING` env override then the process default.
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    /// Read for `effort` (see `Self::reasoning_effort`) and `format`:
    /// decoding cannot be constrained to a schema, so a non-null `format` is
    /// refused rather than answered in free text. Other keys are ignored like
    /// unknown top-level fields; anything but an object is refused.
    #[serde(default)]
    pub output_config: Option<serde_json::Map<String, Value>>,
    /// `{"type": "auto" | "any" | "none"}` or `{"type": "tool", "name": ...}`;
    /// see `Self::tool_choice`.
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// Every field the request does not declare; see `MESSAGES_UNSUPPORTED`.
    #[serde(flatten)]
    pub other: serde_json::Map<String, Value>,
}

/// `/v1/messages` fields that must stay at their default. `tool_choice` and
/// tool types are checked separately (see [`MessagesRequest::into_job`]).
const MESSAGES_UNSUPPORTED: &[super::Unsupported] = &[
    super::Unsupported {
        field: "output_format",
        accepts: |_| false,
        refused: "structured output",
    },
    super::Unsupported {
        field: "mcp_servers",
        accepts: |v| v.as_array().is_some_and(|a| a.is_empty()),
        refused: "MCP servers",
    },
    super::Unsupported {
        field: "container",
        accepts: |_| false,
        refused: "a code execution container",
    },
];

impl MessagesRequest {
    /// Resolve the per-request reasoning toggle via the single shared resolver.
    /// `thinking.type` `enabled` or `adaptive` maps to `Some(true)`, `disabled`
    /// to `Some(false)`, and an absent `thinking` field to `None` (defer to
    /// env/default).
    pub fn resolve_thinking(&self) -> bool {
        let per_request = self
            .thinking
            .as_ref()
            .map(|t| t.thinking_type != ThinkingType::Disabled);
        super::resolve_enable_thinking(per_request)
    }

    /// `tool_choice.disable_parallel_tool_use`: at most one tool call.
    fn single_tool_call(&self) -> bool {
        self.tool_choice
            .as_ref()
            .and_then(|c| c.get("disable_parallel_tool_use"))
            == Some(&Value::Bool(true))
    }

    fn tool_choice(&self) -> Result<super::ToolChoice, ServerError> {
        use super::ToolChoice;
        let choice = self.tool_choice.as_ref().unwrap_or(&Value::Null);
        if choice.is_null() {
            return Ok(ToolChoice::Auto);
        }
        if choice
            .get("disable_parallel_tool_use")
            .is_some_and(|d| !d.is_null() && !d.is_boolean())
        {
            return Err(ServerError::bad_request_field(
                "tool_choice.disable_parallel_tool_use must be a boolean",
                "tool_choice.disable_parallel_tool_use",
                "invalid_type",
            ));
        }
        match (choice["type"].as_str(), choice["name"].as_str()) {
            (Some("auto"), _) => Ok(ToolChoice::Auto),
            (Some("none"), _) => Ok(ToolChoice::None),
            (Some("any"), _) => Ok(ToolChoice::Required),
            (Some("tool"), Some(name)) => Ok(ToolChoice::Named(name.into())),
            _ => Err(ServerError::bad_request_field(
                "tool_choice must be {\"type\": \"auto\" | \"any\" | \"none\"} or {\"type\": \"tool\", \"name\": ...}",
                "tool_choice",
                "invalid_value",
            )),
        }
    }

    /// The tool calls the reply may carry, for the collectors. Taken before
    /// `into_job` consumes the request; a malformed `tool_choice` reads as
    /// `auto` here because `into_job` refuses it.
    pub fn reply_tools(&self) -> ReplyTools {
        self.tool_choice()
            .unwrap_or(super::ToolChoice::Auto)
            .reply_tools(tool_schemas(&self.tools), self.single_tool_call())
    }

    /// The prompt for this request and the text the reply starts with, which
    /// also ends the prompt (see [`super::ToolChoice::response_prefix`]).
    pub(crate) fn prompt(
        &self,
        chat_template: Option<&str>,
    ) -> Result<(String, String), ServerError> {
        let reasoning_effort = self.reasoning_effort()?;
        let enable_thinking = self.resolve_thinking();
        let tool_choice = self.tool_choice()?;
        tool_choice.check(
            self.tools.iter().map(|t| t.name.as_str()),
            enable_thinking,
            chat_template.is_some(),
        )?;
        let tools: &[AnthropicTool] = if tool_choice.offers_tools() {
            &self.tools
        } else {
            &[]
        };
        // Flatten the system field through the SAME shared helper as message
        // content (ROBUST-007 guard + single recognized key set), so a number
        // `system` 400s identically to a number `content`.
        let system_text = match &self.system {
            Some(v) => Some(super::flatten_content(v, "system")?),
            None => None,
        };
        // Prefer the model's embedded template (native tool-calling protocol),
        // shared with the CLI and OpenAI surfaces so the three cannot drift.
        // Fall back to the hard-coded ChatML transcript (compose the tools into
        // the system message) when no template is embedded.
        let mut prompt = match chat_template {
            Some(tmpl) => render_prompt_templated(
                system_text.as_deref(),
                &self.messages,
                tools,
                enable_thinking,
                reasoning_effort,
                tmpl,
            )?,
            None => {
                let tool_schemas: Vec<ToolSchema> = tools
                    .iter()
                    .map(|t| ToolSchema {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters_json_schema: serde_json::to_string(&t.input_schema)
                            .unwrap_or_else(|_| "{}".into()),
                    })
                    .collect();
                let final_system = compose_system_with_tools(system_text.as_deref(), &tool_schemas);
                render_prompt(&final_system, &self.messages, enable_thinking)?
            }
        };
        let response_prefix = tool_choice.response_prefix();
        prompt.push_str(&response_prefix);
        Ok((prompt, response_prefix))
    }

    /// `output_config.effort` (`low`, `medium`, `high`, `xhigh` or `max`) as
    /// the chat template's `reasoning_effort`, mapped by the shared
    /// [`super::template_reasoning_effort`]. Any other value is refused.
    fn reasoning_effort(&self) -> Result<Option<&'static str>, ServerError> {
        const PARAM: &str = "output_config.effort";
        const LEVELS: &str = "`low`, `medium`, `high`, `xhigh` or `max`";
        let effort = self.output_config.as_ref().and_then(|c| c.get("effort"));
        match super::effort_level(effort, PARAM, LEVELS)? {
            None => Ok(None),
            Some(level) => super::template_reasoning_effort(level).ok_or_else(|| {
                ServerError::bad_request_field(
                    format!("{PARAM} must be one of {LEVELS}"),
                    PARAM,
                    "invalid_value",
                )
            }),
        }
    }

    /// Refuse what this request asks for that the server cannot produce;
    /// shared by `/v1/messages` and `/v1/messages/count_tokens`. The prompt's
    /// length is `into_job`'s to check: a count of a prompt longer than the
    /// context is still a count.
    fn check(&self) -> Result<(), ServerError> {
        super::refuse_unsupported(&self.other, MESSAGES_UNSUPPORTED)?;
        if self
            .tools
            .iter()
            .any(|t| t.tool_type.as_deref().is_some_and(|ty| ty != "custom"))
        {
            return Err(super::unsupported(
                "tools[].type",
                "a tool type other than `custom`",
            ));
        }
        if self
            .output_config
            .as_ref()
            .and_then(|c| c.get("format"))
            .is_some_and(|f| !f.is_null())
        {
            return Err(super::unsupported(
                "output_config.format",
                "structured output",
            ));
        }
        Ok(())
    }

    /// The input tokens `/v1/messages` would run for this request: its prompt,
    /// tokenized. Served as `/v1/messages/count_tokens`.
    pub fn count_tokens(&self, engine: &EngineHandle) -> Result<usize, ServerError> {
        self.check()?;
        let (prompt, _) = self.prompt(engine.chat_template())?;
        Ok(engine.tokenize_for_request(&prompt).len())
    }

    pub fn into_job(self, engine: &EngineHandle) -> Result<JobRequest, ServerError> {
        let max_tokens = self.max_tokens.ok_or_else(|| {
            ServerError::bad_request_field(
                "missing required field: `max_tokens`",
                "max_tokens",
                "missing_field",
            )
        })?;
        self.check()?;
        let (prompt, response_prefix) = self.prompt(engine.chat_template())?;
        let enable_thinking = self.resolve_thinking();
        // Reasoning-token cap within `max_tokens` (Anthropic
        // `thinking.budget_tokens`); falls back to the shared default.
        let reasoning_budget = self
            .thinking
            .as_ref()
            .and_then(|t| t.budget_tokens)
            .unwrap_or_else(lumen_runtime::runtime_defaults::chat_reasoning_budget_default);
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
        let temperature = self
            .temperature
            .unwrap_or_else(lumen_runtime::runtime_defaults::default_temperature);
        let sampling = SamplingParams {
            temperature,
            seed: Some(super::next_random_seed()),
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: Some(super::diag_repetition_penalty(temperature)),
            frequency_penalty: Some(super::diag_frequency_penalty()),
            repeat_last_n: super::diag_repeat_last_n(),
            anti_restate: super::diag_anti_restate(),
            ..Default::default()
        };
        Ok(JobRequest {
            prompt_tokens,
            max_tokens,
            stop_text: self.stop_sequences,
            eos_token_ids: eos,
            ignore_eos: false,
            sampling,
            suffix_threshold: lumen_runtime::session::Session::DEFAULT_SUFFIX_THRESHOLD,
            enable_thinking,
            reasoning_budget,
            response_prefix,
        })
    }
}

/// Build the runtime [`ToolSchemas`] from Anthropic tool defs (typing native
/// `<parameter>` values in the collectors, mirroring the OpenAI surface).
fn tool_schemas(tools: &[AnthropicTool]) -> ToolSchemas {
    let schemas: Vec<ToolSchema> = tools
        .iter()
        .map(|t| ToolSchema {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters_json_schema: serde_json::to_string(&t.input_schema)
                .unwrap_or_else(|_| "{}".into()),
        })
        .collect();
    ToolSchemas::from_tools(&schemas)
}

/// Render the Anthropic request through the model's embedded Jinja template.
/// Converts Anthropic's block-structured messages into the template's message
/// shape — `tool_result` blocks become `role:"tool"` messages (the template
/// groups consecutive ones into a single user turn), `tool_use` blocks become
/// assistant `tool_calls` with the `input` object as `arguments` — then defers
/// to the shared renderer so the transcript is byte-identical to the OpenAI
/// surface's for an equivalent round-trip.
fn render_prompt_templated(
    system: Option<&str>,
    messages: &[AnthropicMessage],
    tools: &[AnthropicTool],
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
    template: &str,
) -> Result<String, ServerError> {
    let mut msgs: Vec<Value> = Vec::with_capacity(messages.len() + 1);
    if let Some(s) = system {
        if !s.is_empty() {
            msgs.push(json!({"role": "system", "content": s}));
        }
    }
    for m in messages {
        match m.role.as_str() {
            "user" => {
                let (text, tool_results) =
                    partition_tool_result_blocks(&m.content, "messages.content")?;
                for tr in &tool_results {
                    msgs.push(json!({"role": "tool", "content": tr}));
                }
                if !text.is_empty() || tool_results.is_empty() {
                    msgs.push(json!({"role": "user", "content": text}));
                }
            }
            "assistant" => {
                let (text, reasoning, tool_uses) =
                    partition_tool_use_blocks(&m.content, "messages.content")?;
                let mut obj = serde_json::Map::new();
                obj.insert("role".into(), json!("assistant"));
                obj.insert("content".into(), json!(text));
                // Passed even when empty: a template that finds none looks for
                // reasoning inside the content instead (Qwen3.5 splits it at
                // `</think>`).
                if let Some(reasoning) = reasoning {
                    obj.insert("reasoning_content".into(), json!(reasoning));
                }
                if !tool_uses.is_empty() {
                    let calls: Vec<Value> = tool_uses
                        .iter()
                        .map(|(name, arguments_json)| {
                            let args = serde_json::from_str::<Value>(arguments_json)
                                .ok()
                                .filter(Value::is_object)
                                .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                            json!({"type": "function", "function": {"name": name, "arguments": args}})
                        })
                        .collect();
                    obj.insert("tool_calls".into(), Value::Array(calls));
                }
                msgs.push(Value::Object(obj));
            }
            "system" => msgs.push(json!({
                "role": "system",
                "content": super::flatten_content(&m.content, "messages.content")?,
            })),
            other => {
                return Err(ServerError::bad_request_field(
                    format!("unknown anthropic role: {other}"),
                    "messages[].role",
                    "invalid_value",
                ));
            }
        }
    }
    let tools_json: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            })
        })
        .collect();
    lumen_runtime::chat_template::render_chat_prompt(
        template,
        &Value::Array(msgs),
        &Value::Array(tools_json),
        true,
        enable_thinking,
        reasoning_effort,
    )
    .map_err(|e| ServerError::bad_request(format!("chat template render failed: {e}")))
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
            // A system message inside `messages` stays where it was sent.
            "system" => {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(&super::flatten_content(&m.content, "messages.content")?);
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
    // Open vs closed `<think>` tail from the single shared helper, matching
    // the CLI + OpenAI server behaviour exactly (closed when reasoning is
    // off — the default — so this path is byte-identical to before). See
    // render_chat_prompt in wire/openai.rs for the full rationale.
    prompt.push_str("<|im_start|>assistant\n");
    prompt.push_str(lumen_runtime::runtime_defaults::think_prompt_tail(
        enable_thinking,
    ));
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
    let (text, _, tool_uses) = partition_tool_use_blocks(content, "messages.content")?;
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

/// Walk an assistant content value, returning `(flattened_text, reasoning,
/// tool_uses)` where reasoning joins the `{type:"thinking", thinking}` blocks
/// (`None` without one) and each tool_use is `(name, arguments_json)`.
/// Recognizes
/// `{type:"tool_use", name, input}` blocks; the `input` object is serialized
/// to a JSON string (the on-wire `arguments` form). Bare-string and
/// `{type:"text", text}` parts flatten into the text via the SAME key set as
/// the shared `flatten_content`. A bare number/bool top-level content 400s
/// (ROBUST-007), consistent with the text flattener.
fn partition_tool_use_blocks(
    content: &Value,
    param: &str,
) -> Result<(String, Option<String>, Vec<(String, String)>), ServerError> {
    match content {
        // No typed blocks possible in a string/null: reuse the shared text
        // flattener verbatim (also enforces ROBUST-007 on scalars).
        Value::String(_) | Value::Null => {
            Ok((super::flatten_content(content, param)?, None, Vec::new()))
        }
        Value::Array(arr) => {
            let mut text = String::new();
            let mut reasoning: Option<String> = None;
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
                        // The turn's reasoning, which the chat template renders
                        // back into the turn.
                        Some("thinking") => {
                            let t = obj.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                            reasoning.get_or_insert_with(String::new).push_str(t);
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
            Ok((text, reasoning, tool_uses))
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
        Value::String(_) | Value::Null => Ok((super::flatten_content(content, param)?, Vec::new())),
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
        rx.recv().await.map(|chunk| {
            (
                Ok::<bytes::Bytes, std::io::Error>(bytes::Bytes::from(chunk)),
                rx,
            )
        })
    });
    Body::from_stream(stream)
}

pub fn stream_messages(
    rx: JobResponseChannel,
    model: String,
    thinking: bool,
    stop: Vec<String>,
    tools: ReplyTools,
) -> Body {
    let (tx, body_rx) = mpsc::channel::<Vec<u8>>(64);
    tokio::spawn(drive_messages_stream(rx, tx, model, thinking, stop, tools));
    body_from_byte_stream(body_rx)
}

/// Idle-keepalive interval: emit a ping after this long with no wire output.
const PING_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

/// One step of the Anthropic content-block stream, in emission order. The driver
/// turns each [`crate::sse::EmitDelta`] (and the end-of-stream flush) into an
/// ordered list of these and plays them through [`BlockState`], so the live loop
/// and the flush share ONE state machine for opening, filling, and closing the
/// thinking / text / tool_use content blocks.
enum BlockStep {
    Reasoning(String),
    Text(String),
    Tool(ToolStreamEvent),
}

/// Open/closed state of the single in-flight Anthropic content block. At most one
/// of thinking / text / tool_use is open at a time; opening a new block (or the
/// end-of-stream flush) closes the current one via `close_current` and advances
/// `index`, so indices are always unique and every start has one stop.
#[derive(Default)]
struct BlockState {
    index: usize,
    thinking_open: bool,
    text_open: bool,
    tool_open: bool,
    /// Set once any tool_use block opens, so the terminal stop_reason can be
    /// upgraded Stop -> tool_use (matching the non-streaming Anthropic path).
    emitted_tool: bool,
    /// Set when a tool block is closed WITHOUT its own `End` (a call cut off
    /// mid-input). Persists across later blocks so the terminal stop_reason stays
    /// max_tokens even when a well-formed call follows the truncated one.
    truncated: bool,
}

/// Send one SSE event; `Err(())` signals the client hung up (the driver returns).
async fn send_block(tx: &mpsc::Sender<Vec<u8>>, event: &str, data: &Value) -> Result<(), ()> {
    tx.send(sse_event(event, &data.to_string()))
        .await
        .map_err(|_| ())
}

/// Close whichever content block is open (thinking, text, or tool_use) before a
/// different block opens or the stream ends, advancing `index`. A tool_use block
/// closed here was abandoned without its own `End` (cut off mid-input), so the turn
/// is marked `truncated` — this is what keeps indices unique and stop_reason
/// correct when a truncated call is followed by more text or another call.
async fn close_current(tx: &mpsc::Sender<Vec<u8>>, st: &mut BlockState) -> Result<(), ()> {
    if st.thinking_open || st.text_open || st.tool_open {
        let s = json!({ "type": "content_block_stop", "index": st.index });
        send_block(tx, "content_block_stop", &s).await?;
        st.index += 1;
        if st.tool_open {
            st.truncated = true;
        }
        st.thinking_open = false;
        st.text_open = false;
        st.tool_open = false;
    }
    Ok(())
}

/// Play one [`BlockStep`] through the content-block state machine.
async fn emit_block_step(
    tx: &mpsc::Sender<Vec<u8>>,
    st: &mut BlockState,
    step: BlockStep,
) -> Result<(), ()> {
    match step {
        BlockStep::Reasoning(text) => {
            if !st.thinking_open {
                let b = json!({ "type": "content_block_start", "index": st.index,
                    "content_block": { "type": "thinking", "thinking": "" } });
                send_block(tx, "content_block_start", &b).await?;
                st.thinking_open = true;
            }
            let d = json!({ "type": "content_block_delta", "index": st.index,
                "delta": { "type": "thinking_delta", "thinking": text } });
            send_block(tx, "content_block_delta", &d).await?;
        }
        BlockStep::Text(text) => {
            // Append to an already-open text block (one content_block with many
            // text_deltas, as the real Anthropic API does). Close a thinking block,
            // or a tool block abandoned mid-input, before opening the text block —
            // the latter keeps text off a truncated call's index.
            if st.thinking_open || st.tool_open {
                close_current(tx, st).await?;
            }
            if !st.text_open {
                let b = json!({ "type": "content_block_start", "index": st.index,
                    "content_block": { "type": "text", "text": "" } });
                send_block(tx, "content_block_start", &b).await?;
                st.text_open = true;
            }
            let d = json!({ "type": "content_block_delta", "index": st.index,
                "delta": { "type": "text_delta", "text": text } });
            send_block(tx, "content_block_delta", &d).await?;
        }
        BlockStep::Tool(ToolStreamEvent::Start { name }) => {
            st.emitted_tool = true;
            close_current(tx, st).await?;
            let b = json!({ "type": "content_block_start", "index": st.index,
                "content_block": { "type": "tool_use", "id": super::tool_call_id("toolu"),
                    "name": name, "input": {} } });
            send_block(tx, "content_block_start", &b).await?;
            st.tool_open = true;
        }
        BlockStep::Tool(ToolStreamEvent::ArgJsonDelta { partial_json }) => {
            let d = json!({ "type": "content_block_delta", "index": st.index,
                "delta": { "type": "input_json_delta", "partial_json": partial_json } });
            send_block(tx, "content_block_delta", &d).await?;
        }
        BlockStep::Tool(ToolStreamEvent::End) => {
            let s = json!({ "type": "content_block_stop", "index": st.index });
            send_block(tx, "content_block_stop", &s).await?;
            st.index += 1;
            st.tool_open = false;
        }
    }
    Ok(())
}

async fn drive_messages_stream(
    mut rx: JobResponseChannel,
    tx: mpsc::Sender<Vec<u8>>,
    model: String,
    thinking: bool,
    stop: Vec<String>,
    tools: ReplyTools,
) {
    let msg_id = format!(
        "msg_lumen_{:x}-{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0),
        super::next_response_seq()
    );
    let mut emitter = SseSafeEmitter::with_tools(thinking, tools);
    // F4: seed the streaming stop matcher from `stop_sequences`. Empty =>
    // verbatim passthrough (byte-identical); see the OpenAI `drive_chat_stream`
    // note for the worker/wire division of labour.
    let mut stop_matcher = StopMatcher::new(stop);
    let mut finish_reason: Option<FinishReason> = None;
    // Content-block state machine for the thinking / text / tool_use blocks. The
    // same `BlockState` drives the live loop and the end-of-stream flush, so a
    // tool_use block whose input spans feeds opens, fills, and closes coherently.
    let mut st = BlockState::default();
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
    if tx
        .send(sse_event("message_start", &start.to_string()))
        .await
        .is_err()
    {
        return;
    }

    // Keepalive clock. The ping fires after PING_INTERVAL of WIRE inactivity (no
    // frame emitted), not token inactivity: a long buffered argument receives
    // tokens while emitting nothing and must still be kept alive. `last_emit`
    // advances only when a frame goes out, so the deadline tracks the wire.
    let mut last_emit = tokio::time::Instant::now();
    loop {
        let evt = tokio::select! {
            maybe = rx.recv() => match maybe {
                Some(evt) => evt,
                None => break,
            },
            _ = tokio::time::sleep_until(last_emit + PING_INTERVAL) => {
                // Keepalive during a long prefill/queue wait before the first token,
                // or a long buffered argument mid-generation: the Anthropic
                // protocol's `event: ping`. Incremental streaming covers gaps where
                // frames flow; this covers gaps where they do not.
                if tx.send(sse_event("ping", "{\"type\":\"ping\"}")).await.is_err() {
                    return;
                }
                last_emit = tokio::time::Instant::now();
                continue;
            }
        };
        match evt {
            TokenEvent::PrefillDone { .. } => {}
            // Bench surface: the router refuses streaming requests while it is
            // armed, so this is unreachable in practice; if it does arrive,
            // end the stream with an error rather than drop it silently.
            TokenEvent::BenchTokenIds { .. } => {
                let err = json!({"type": "error", "error": { "type": "api_error",
                    "message": "LUMEN_BENCH_TOKEN_IDS is not supported on streaming \
responses; use stream=false" }});
                let _ = tx.send(sse_event("error", &err.to_string())).await;
                return;
            }
            TokenEvent::Token { delta_text, .. } => {
                let delta = emitter.push(&delta_text);
                // A token that emits nothing (a buffered argument accumulating) must
                // not reset the keepalive clock — only real wire output does.
                let mut emitted = false;
                // Reasoning precedes all content (the extractor is one-way).
                if !delta.reasoning.is_empty() {
                    if emit_block_step(&tx, &mut st, BlockStep::Reasoning(delta.reasoning))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    emitted = true;
                }
                // Play the content events in SOURCE order: text through the stop
                // matcher, tool events straight through. Keeping the parser's order is
                // what lets text before / between / after a tool call land in its own
                // block at the right place — no reordering heuristic, so a truncated
                // call followed by text and another call cannot corrupt the blocks.
                let mut hit_stop = false;
                for ev in delta.events {
                    match ev {
                        StreamEvent::Text(t) => {
                            let (safe_text, hit) = stop_matcher.push(&t);
                            if !safe_text.is_empty() {
                                if emit_block_step(&tx, &mut st, BlockStep::Text(safe_text))
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                                emitted = true;
                            }
                            if hit {
                                hit_stop = true;
                                break;
                            }
                        }
                        StreamEvent::Tool(te) => {
                            if emit_block_step(&tx, &mut st, BlockStep::Tool(te))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            emitted = true;
                        }
                    }
                }
                // Advance the clock AFTER the frames are sent: a send that blocks on
                // backpressure must not leave a stale (early) deadline behind it.
                if emitted {
                    last_emit = tokio::time::Instant::now();
                }
                if hit_stop {
                    // Wire-side stop (redundant safety net). Report StopSequence
                    // so Anthropic renders "stop_sequence".
                    finish_reason = Some(FinishReason::StopSequence);
                    break;
                }
            }
            TokenEvent::Done {
                finish_reason: fr,
                prompt_tokens,
                completion_tokens,
            } => {
                finish_reason = Some(fr);
                input_tokens = prompt_tokens;
                output_tokens = completion_tokens;
                break;
            }
            TokenEvent::Error(msg) => {
                let err =
                    json!({"type": "error", "error": { "type": "api_error", "message": msg }});
                let _ = tx.send(sse_event("error", &err.to_string())).await;
                return;
            }
        }
    }
    // Flush residual: reasoning, then any tool events that close in the flush,
    // then residual answer text -- same ordering and state machine as the loop.
    // Errors are ignored here: the stream is ending.
    let (residual, incomplete) = emitter.finish();
    let stopped_by_sequence = finish_reason == Some(FinishReason::StopSequence);
    if !residual.reasoning.is_empty() {
        let _ = emit_block_step(&tx, &mut st, BlockStep::Reasoning(residual.reasoning)).await;
    }
    // Play the residual content in SOURCE order, through the same state machine as
    // the loop. Once a stop sequence fired, residual content is post-stop and
    // dropped; otherwise text passes through the stop matcher (catching a stop
    // straddling the held tail) and tool events close any call that finished in the
    // flush.
    if !stopped_by_sequence {
        for ev in residual.events {
            match ev {
                StreamEvent::Text(t) => {
                    let safe = if stop_matcher.is_active() {
                        stop_matcher.push(&t).0
                    } else {
                        t
                    };
                    if !safe.is_empty() {
                        let _ = emit_block_step(&tx, &mut st, BlockStep::Text(safe)).await;
                    }
                }
                StreamEvent::Tool(te) => {
                    let _ = emit_block_step(&tx, &mut st, BlockStep::Tool(te)).await;
                }
            }
        }
    }
    // Trailing text: the stop matcher's held tail (now safe), dropped once a stop
    // fired. The cut-off tool body is appended below.
    let mut final_text = if stopped_by_sequence {
        String::new()
    } else {
        stop_matcher.finish()
    };
    // A tool call cut off mid-body: a NATIVE one already streamed its partial
    // input into the still-open tool block (closed below; never duplicated as
    // text); a legacy / pre-`<function=>` one streamed nothing, so its body is
    // surfaced as answer text so it is never silently lost. Either way the finish
    // reason is forced to Length so the client sees a truncation and continues.
    // `tool_open` is taken to be THIS incomplete call's own block, which holds for
    // well-formed output and any single truncated call; the trained format never
    // emits the one shape that breaks it (a native call that reaches `</tool_call>`
    // with a parameter left open, keeping the block open across a following call).
    if let Some(body) = &incomplete {
        if !st.tool_open {
            final_text.push_str(body);
        }
    }
    if !final_text.is_empty() {
        let _ = emit_block_step(&tx, &mut st, BlockStep::Text(final_text)).await;
    }
    // Close whichever block is still open (thinking-only reply, text, or a tool
    // block cut off mid-input); closing a tool block here sets `truncated`.
    let _ = close_current(&tx, &mut st).await;
    // A tool-call turn ends with the worker's natural `Stop`; upgrade it to
    // `ToolCalls` so the terminal message_delta reports stop_reason "tool_use"
    // (matching the OpenAI streaming + non-streaming Anthropic paths). A turn
    // truncated mid tool call -> max_tokens (via Length): either the parser
    // reported an incomplete body, OR some tool call was abandoned without its
    // `End` (`truncated`) — never report that partial, invalid-JSON call as a
    // clean stop, even when a well-formed call follows it.
    let reason = if incomplete.is_some() || st.truncated {
        FinishReason::Length
    } else {
        match finish_reason {
            Some(FinishReason::Stop) if st.emitted_tool => FinishReason::ToolCalls,
            Some(r) => r,
            None => FinishReason::Stop,
        }
    };
    let delta_msg = json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": reason.as_anthropic(),
            "stop_sequence": null,
        },
        "usage": { "output_tokens": output_tokens, "input_tokens": input_tokens }
    });
    let _ = tx
        .send(sse_event("message_delta", &delta_msg.to_string()))
        .await;
    let _ = tx
        .send(sse_event("message_stop", "{\"type\":\"message_stop\"}"))
        .await;
}

// ----------------------------- Non-streaming ----------------------------

pub async fn collect_messages(
    mut rx: JobResponseChannel,
    model: String,
    thinking: bool,
    stop: Vec<String>,
    tools: ReplyTools,
) -> Result<Value, ServerError> {
    let mut emitter = SseSafeEmitter::with_tools(thinking, tools);
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
    // Bench surface (LUMEN_BENCH_TOKEN_IDS): (generated ids, eos set).
    let mut bench_ids: Option<super::openai::BenchRecord> = None;

    while let Some(evt) = rx.recv().await {
        match evt {
            TokenEvent::PrefillDone { .. } => {}
            TokenEvent::BenchTokenIds {
                generated_token_ids,
                eos_token_ids,
                top2,
            } => {
                bench_ids = Some((generated_token_ids, eos_token_ids, top2));
            }
            TokenEvent::Token { delta_text, .. } => {
                let delta = emitter.push(&delta_text);
                reasoning.push_str(&delta.reasoning);
                let (safe_text, hit_stop) = stop_matcher.push(&delta.text());
                text.push_str(&safe_text);
                for tc in delta.tool_calls {
                    tool_blocks.push(json!({
                        "type": "tool_use",
                        "id": super::tool_call_id("toolu"),
                        "name": tc.name,
                        "input": serde_json::from_str::<Value>(&tc.arguments_json)
                            .unwrap_or(Value::String(tc.arguments_json)),
                    }));
                }
                if hit_stop {
                    // Wire-side stop match: report the Anthropic-correct
                    // stop_sequence reason (overrides the default end_turn).
                    finish = FinishReason::StopSequence;
                    break;
                }
            }
            TokenEvent::Done {
                finish_reason,
                prompt_tokens: p,
                completion_tokens: c,
            } => {
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
    let (residual, incomplete) = emitter.finish();
    reasoning.push_str(&residual.reasoning);
    // Drop the residual answer text once a stop sequence fired (post-stop
    // content); otherwise stop-match + drain the emitter residual. Empty stop =>
    // verbatim append, byte-identical.
    if finish != FinishReason::StopSequence {
        let (residual_safe, _) = if stop_matcher.is_active() {
            stop_matcher.push(&residual.text())
        } else {
            (residual.text(), false)
        };
        text.push_str(&residual_safe);
        text.push_str(&stop_matcher.finish());
    }
    // surface a tool call cut off inside its body as answer text (never drop it).
    if let Some(body) = &incomplete {
        text.push_str(body);
    }

    if !tool_blocks.is_empty() && finish == FinishReason::Stop {
        finish = FinishReason::ToolCalls;
    }
    // an incomplete tool call means the turn was truncated -> report max_tokens
    // (via Length) so the client continues instead of trusting a clean stop.
    if incomplete.is_some() {
        finish = FinishReason::Length;
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

    let mut body = json!({
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
    });
    // The same surface as the OpenAI routes, so no route drops the ids silently.
    super::openai::attach_bench_token_ids(&mut body, bench_ids, finish)?;
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str) -> TokenEvent {
        TokenEvent::Token {
            token_id: 0,
            delta_text: text.to_string(),
        }
    }

    /// Build an unpooled `JobResponseChannel` from a fixed event list (mirrors
    /// the OpenAI `collect_chat_from_events` helper).
    async fn collect_messages_from_events(events: Vec<TokenEvent>, thinking: bool) -> Value {
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
        // Test helper exercises the legacy JSON tool-call path (schemaless); the
        // schema-aware native path is covered by the runtime tests + Modal §2D.
        collect_messages(pooled, "test".into(), thinking, stop, ReplyTools::default())
            .await
            .unwrap()
    }

    fn user(text: &str) -> AnthropicMessage {
        AnthropicMessage {
            role: "user".into(),
            content: Value::String(text.into()),
        }
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

    fn system(text: &str) -> AnthropicMessage {
        AnthropicMessage {
            role: "system".into(),
            content: Value::String(text.into()),
        }
    }

    #[test]
    fn system_message_renders_where_it_was_sent() {
        let template =
            include_str!("../../../lumen-runtime/tests/fixtures/qwen38_chat_template.jinja");
        let messages = [
            user("Hi"),
            system("Answer in French."),
            AnthropicMessage {
                role: "assistant".into(),
                content: serde_json::json!([
                    {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}}
                ]),
            },
            AnthropicMessage {
                role: "user".into(),
                content: serde_json::json!([
                    {"type": "tool_result", "tool_use_id": "t1", "content": "18C"},
                    {"type": "text", "text": "And now?"}
                ]),
            },
            AnthropicMessage {
                role: "system".into(),
                content: serde_json::json!([{"type": "text", "text": "Be brief."}]),
            },
        ];
        let out =
            render_prompt_templated(Some("Sys"), &messages, &[], false, None, template).unwrap();
        assert!(
            out.starts_with("<|im_start|>system\nSys<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>system\nAnswer in French.<|im_end|>\n<|im_start|>assistant\n"),
            "{out}"
        );
        assert!(
            out.ends_with("<|im_start|>user\nAnd now?<|im_end|>\n<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "{out}"
        );

        let manual = render_prompt("Sys", &[user("Hi"), system("Be brief.")], false).unwrap();
        assert_eq!(
            manual,
            "<|im_start|>system\nSys<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn resolve_thinking_maps_anthropic_config() {
        // type=="enabled" -> true; type=="disabled" -> false. Absent -> env/
        // default (false here, no env set in the test process).
        let enabled = MessagesRequest {
            model: "m".into(),
            messages: vec![],
            max_tokens: Some(1),
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
            stop_sequences: vec![],
            tools: vec![],
            thinking: Some(ThinkingConfig {
                thinking_type: ThinkingType::Enabled,
                budget_tokens: None,
            }),
            output_config: None,
            tool_choice: None,
            other: Default::default(),
        };
        assert!(enabled.resolve_thinking());
        let disabled = MessagesRequest {
            thinking: Some(ThinkingConfig {
                thinking_type: ThinkingType::Disabled,
                budget_tokens: None,
            }),
            ..enabled.clone()
        };
        assert!(!disabled.resolve_thinking());
    }

    #[test]
    fn adaptive_thinking_is_on_and_an_unknown_type_is_refused() {
        let request = |thinking: Value| {
            serde_json::from_value::<MessagesRequest>(json!({
                "model": "m", "max_tokens": 1, "messages": [], "thinking": thinking,
            }))
        };
        let adaptive = request(json!({"type": "adaptive", "display": "omitted"})).unwrap();
        assert!(adaptive.resolve_thinking());
        assert!(request(json!({"type": "enabled", "budget_tokens": 1024}))
            .unwrap()
            .resolve_thinking());
        assert!(!request(json!({"type": "disabled"}))
            .unwrap()
            .resolve_thinking());
        let unknown = request(json!({"type": "sometimes"}))
            .unwrap_err()
            .to_string();
        assert!(
            unknown.contains("unknown thinking type, expected") && !unknown.contains("sometimes"),
            "{unknown}"
        );
        for not_a_string in [json!(1), json!(null), json!({"adaptive": null})] {
            let err = request(json!({"type": not_a_string}))
                .unwrap_err()
                .to_string();
            assert!(
                err.starts_with("invalid type:") && err.contains("expected a string"),
                "{err}"
            );
        }
    }

    #[tokio::test]
    async fn collect_messages_thinking_off_has_no_thinking_block() {
        // Byte-identity guard: thinking off => NO thinking content block even
        // when the model literally emits </think>.
        let events = vec![
            tok("answer </think> still answer"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 4,
            },
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
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 5,
            },
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
            TokenEvent::Done {
                finish_reason: FinishReason::Length,
                prompt_tokens: 1,
                completion_tokens: 3,
            },
        ];
        let resp = collect_messages_from_events_with_stop(events, false, vec!["HALT".into()]).await;
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
            TokenEvent::Done {
                finish_reason: FinishReason::Length,
                prompt_tokens: 1,
                completion_tokens: 3,
            },
        ];
        let resp = collect_messages_from_events_with_stop(events, false, Vec::new()).await;
        let blocks = resp["content"].as_array().unwrap();
        assert_eq!(
            blocks[0]["text"], full,
            "empty stop passes full text through verbatim"
        );
        assert_eq!(resp["stop_reason"], "max_tokens");
    }

    /// A stop straddling two token deltas is caught by the wire window buffer.
    #[tokio::test]
    async fn collect_messages_wire_stop_straddles_token_deltas() {
        let events = vec![
            tok("alpha HA"),
            tok("LT omega"),
            TokenEvent::Done {
                finish_reason: FinishReason::Length,
                prompt_tokens: 1,
                completion_tokens: 4,
            },
        ];
        let resp = collect_messages_from_events_with_stop(events, false, vec!["HALT".into()]).await;
        let blocks = resp["content"].as_array().unwrap();
        assert_eq!(blocks[0]["text"], "alpha ");
        assert_eq!(resp["stop_reason"], "stop_sequence");
    }

    // ---- Streaming terminal stop_reason (tool_use vs end_turn) ----

    /// Drive `drive_messages_stream` over a fixed event list and return the raw
    /// SSE byte stream as a UTF-8 string (mirrors the non-streaming helper).
    async fn stream_messages_to_string(
        events: Vec<TokenEvent>,
        thinking: bool,
        stop: Vec<String>,
    ) -> String {
        stream_messages_to_string_tools(events, thinking, stop, ReplyTools::default()).await
    }

    async fn stream_messages_to_string_tools(
        events: Vec<TokenEvent>,
        thinking: bool,
        stop: Vec<String>,
        tools: ReplyTools,
    ) -> String {
        let (tx, rx) = mpsc::channel(events.len().max(1));
        let return_sender = tx.clone();
        for e in events {
            tx.send(e).await.unwrap();
        }
        drop(tx);
        let pooled = crate::engine::PooledReceiver::new(rx, return_sender, None, 0, None);
        let (body_tx, mut body_rx) = mpsc::channel::<Vec<u8>>(256);
        tokio::spawn(drive_messages_stream(
            pooled,
            body_tx,
            "test".into(),
            thinking,
            stop,
            tools,
        ));
        let mut out = String::new();
        while let Some(chunk) = body_rx.recv().await {
            out.push_str(&String::from_utf8_lossy(&chunk));
        }
        out
    }

    /// A `ReplyTools` with one `string`-typed parameter so the native streamer
    /// takes the incremental `StreamString` path. Without a schema the value is
    /// `BufferTyped` — held until the close — so it never exercises char streaming.
    fn string_param_tools(func: &str, param: &str) -> ReplyTools {
        let schema = ToolSchema {
            name: func.into(),
            description: String::new(),
            parameters_json_schema: format!(
                "{{\"type\":\"object\",\"properties\":{{\"{param}\":{{\"type\":\"string\"}}}}}}"
            ),
        };
        ReplyTools {
            schemas: std::sync::Arc::new(ToolSchemas::from_tools(&[schema])),
            ..ReplyTools::default()
        }
    }

    /// Pull `delta.stop_reason` from the terminal `message_delta` SSE event.
    fn stream_stop_reason(sse: &str) -> String {
        for block in sse.split("\n\n") {
            if block.contains("event: message_delta") {
                let data = block
                    .lines()
                    .find_map(|l| l.strip_prefix("data: "))
                    .expect("message_delta must carry a data line");
                let v: Value = serde_json::from_str(data).unwrap();
                return v["delta"]["stop_reason"].as_str().unwrap().to_string();
            }
        }
        panic!("no message_delta event in stream: {sse}");
    }

    #[tokio::test]
    async fn collect_messages_incomplete_tool_call_surfaces_partial_as_max_tokens() {
        // a tool call cut off inside its body (EOS before </tool_call>) must surface
        // the partial body as a text block and report max_tokens, not drop it as a clean
        // end_turn.
        let events = vec![
            tok("<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"content\": \"fn main() {"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 20,
            },
        ];
        let resp = collect_messages_from_events(events, false).await;
        assert_eq!(resp["stop_reason"], "max_tokens");
        let blocks = resp["content"].as_array().unwrap();
        let text: String = blocks
            .iter()
            .filter(|b| b["type"] == "text")
            .filter_map(|b| b["text"].as_str())
            .collect();
        assert!(
            text.contains("fn main()"),
            "partial surfaced as text: {text:?}"
        );
        assert!(
            !blocks.iter().any(|b| b["type"] == "tool_use"),
            "an incomplete call is not a complete tool_use"
        );
    }

    #[tokio::test]
    async fn stream_messages_incomplete_tool_call_surfaces_partial_as_max_tokens() {
        // The streaming analogue: the partial body is emitted as a text delta and the
        // terminal message_delta reports stop_reason max_tokens.
        let events = vec![
            tok("<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"content\": \"fn main() {"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 20,
            },
        ];
        let sse = stream_messages_to_string(events, false, Vec::new()).await;
        assert_eq!(stream_stop_reason(&sse), "max_tokens");
        assert!(
            sse.contains("fn main()"),
            "partial surfaced in stream: {sse}"
        );
    }

    /// A NATIVE tool call's input streams as it is generated — many
    /// `input_json_delta` frames across feeds, not one buffered frame at close —
    /// and the concatenated `partial_json` is valid JSON with the generated value.
    /// This keeps a client fed during a long tool argument instead of going silent
    /// until the call closes (which can trip a client's idle watchdog).
    #[tokio::test]
    async fn stream_messages_native_tool_input_streams_incrementally() {
        let events = vec![
            tok("<tool_call>\n<function=write_file>\n<parameter=content>\n"),
            tok("line one\n"),
            tok("line two\n"),
            tok("line three\n"),
            tok("</parameter>\n</function>\n</tool_call>"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 3,
                completion_tokens: 12,
            },
        ];
        // A `string` schema so `content` streams incrementally via StreamString —
        // the path this test exists to exercise (a schemaless value would buffer).
        let sse = stream_messages_to_string_tools(
            events,
            false,
            Vec::new(),
            string_param_tools("write_file", "content"),
        )
        .await;
        assert_eq!(
            sse.matches("\"type\":\"tool_use\"").count(),
            1,
            "exactly one tool_use block: {sse}"
        );
        let mut args = String::new();
        let mut parts: Vec<String> = Vec::new();
        for block in sse.split("\n\n") {
            if !block.contains("event: content_block_delta") {
                continue;
            }
            let data = block
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .unwrap();
            let v: Value = serde_json::from_str(data).unwrap();
            if v["delta"]["type"] == "input_json_delta" {
                let p = v["delta"]["partial_json"].as_str().unwrap().to_string();
                args.push_str(&p);
                parts.push(p);
            }
        }
        assert!(
            parts.len() > 1,
            "tool input must stream incrementally, got {} frame(s): {sse}",
            parts.len()
        );
        // The value must stream ACROSS frames, not arrive whole in one (which a
        // buffered path would do): early and late value bytes land in different
        // frames. This is the timing guarantee, not just the framing count.
        assert!(
            !parts
                .iter()
                .any(|p| p.contains("line one") && p.contains("line three")),
            "value must stream across frames, not arrive whole: {parts:?}"
        );
        let parsed: Value =
            serde_json::from_str(&args).expect("concatenated partial_json must be valid JSON");
        assert_eq!(parsed["content"], "line one\nline two\nline three");
        assert_eq!(stream_stop_reason(&sse), "tool_use");
    }

    /// A NATIVE tool call cut off mid-input (EOS before `</tool_call>`): the opened
    /// tool_use block is closed in the flush, the partial arguments already sent as
    /// `input_json_delta` are NOT also duplicated as a text block, and the turn
    /// reports max_tokens. (The legacy-JSON analogue above surfaces as text because
    /// nothing was streamed yet.)
    #[tokio::test]
    async fn stream_messages_native_incomplete_tool_call_closes_block_as_max_tokens() {
        let events = vec![
            tok("<tool_call>\n<function=write_file>\n<parameter=content>\nfn main() {"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 20,
            },
        ];
        // String schema so the partial `content` streams (StreamString) before the
        // cut-off, exercising the open-tool-block truncation path.
        let sse = stream_messages_to_string_tools(
            events,
            false,
            Vec::new(),
            string_param_tools("write_file", "content"),
        )
        .await;
        assert_eq!(stream_stop_reason(&sse), "max_tokens");
        assert_eq!(
            sse.matches("\"type\":\"tool_use\"").count(),
            1,
            "the opened tool_use block is present: {sse}"
        );
        assert!(
            sse.contains("\"type\":\"input_json_delta\""),
            "the partial was streamed as tool input: {sse}"
        );
        assert!(
            !sse.contains("\"type\":\"text_delta\""),
            "a native partial must not also surface as text: {sse}"
        );
        assert_eq!(
            sse.matches("\"type\":\"content_block_start\"").count(),
            sse.matches("\"type\":\"content_block_stop\"").count(),
            "every opened content block must be closed: {sse}"
        );
    }

    /// Plain multi-token text streams as ONE text content block with many
    /// text_deltas (as the real Anthropic API does), not one block per token. A
    /// standard client reading `content[0].text` must see the whole answer, not
    /// just the first token — the regression the BlockState refactor introduced.
    #[tokio::test]
    async fn stream_messages_multi_token_text_is_one_block() {
        let events = vec![
            tok("Hello"),
            tok(" world"),
            tok(" again"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 3,
            },
        ];
        let sse = stream_messages_to_string(events, false, Vec::new()).await;
        assert_eq!(
            sse.matches("\"type\":\"text\"").count(),
            1,
            "exactly one text content block: {sse}"
        );
        assert_eq!(
            sse.matches("\"type\":\"content_block_start\"").count(),
            1,
            "one content_block_start: {sse}"
        );
        assert_eq!(
            sse.matches("\"type\":\"text_delta\"").count(),
            3,
            "all three deltas land in that one block: {sse}"
        );
    }

    /// Text that sits between two tool calls — all arriving in ONE feed while a
    /// tool block is still open from the previous feed — keeps source order and
    /// unique block indices: tool_use(0), text(1), tool_use(2). The pre-fix bug
    /// opened the text block at the second call's index and emitted an orphan stop.
    #[tokio::test]
    async fn stream_messages_text_between_two_calls_in_one_feed_keeps_order() {
        let events = vec![
            tok("<tool_call>\n<function=f>\n"),
            tok("</function>\n</tool_call>between<tool_call>\n<function=g>\n</function>\n</tool_call>"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 8,
            },
        ];
        let sse = stream_messages_to_string(events, false, Vec::new()).await;
        let mut starts: Vec<(u64, String)> = Vec::new();
        let mut stops: Vec<u64> = Vec::new();
        for block in sse.split("\n\n") {
            let Some(data) = block.lines().find_map(|l| l.strip_prefix("data: ")) else {
                continue;
            };
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            match v["type"].as_str() {
                Some("content_block_start") => starts.push((
                    v["index"].as_u64().unwrap(),
                    v["content_block"]["type"].as_str().unwrap().to_string(),
                )),
                Some("content_block_stop") => stops.push(v["index"].as_u64().unwrap()),
                _ => {}
            }
        }
        let indices: Vec<u64> = starts.iter().map(|(i, _)| *i).collect();
        let types: Vec<&str> = starts.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(
            indices,
            vec![0, 1, 2],
            "blocks open at unique indices 0,1,2: {starts:?}"
        );
        assert_eq!(
            types,
            vec!["tool_use", "text", "tool_use"],
            "f, the between-text, then g — in source order: {starts:?}"
        );
        stops.sort_unstable();
        assert_eq!(
            stops,
            vec![0, 1, 2],
            "each block closed once, no orphan stop: {sse}"
        );
    }

    /// A tool call whose outer `</tool_call>` ARRIVES but whose parameter never
    /// closed: the buffered parse finalizes an empty call, so `incomplete` is None —
    /// this truncation is caught ONLY by the open-tool-block / `truncated` guard.
    /// Must report max_tokens, one closed tool_use block, and never duplicate the
    /// partial as text. (Regression guard for the load-bearing `truncated` branch.)
    #[tokio::test]
    async fn stream_messages_param_unclosed_with_outer_close_is_max_tokens() {
        let events = vec![
            tok("<tool_call>\n<function=f>\n<parameter=x>\n1</tool_call>"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 6,
            },
        ];
        let sse = stream_messages_to_string_tools(
            events,
            false,
            Vec::new(),
            string_param_tools("f", "x"),
        )
        .await;
        assert_eq!(stream_stop_reason(&sse), "max_tokens", "{sse}");
        assert_eq!(sse.matches("\"type\":\"tool_use\"").count(), 1, "{sse}");
        assert!(
            !sse.contains("\"type\":\"text_delta\""),
            "the partial must not be duplicated as text: {sse}"
        );
        assert_eq!(
            sse.matches("\"type\":\"content_block_start\"").count(),
            sse.matches("\"type\":\"content_block_stop\"").count(),
            "the block is closed: {sse}"
        );
    }

    /// A truncated call (param unclosed, outer `</tool_call>` present) FOLLOWED by a
    /// complete call: both tool_use blocks must get unique, closed indices (no reuse,
    /// no orphan stop), and the turn must still report max_tokens — the truncation
    /// signal persists past the well-formed call that follows it.
    #[tokio::test]
    async fn stream_messages_truncated_call_then_complete_call_stays_valid() {
        let events = vec![
            tok("<tool_call>\n<function=f>\n<parameter=x>\n1</tool_call>"),
            tok("<tool_call>\n<function=g>\n</function>\n</tool_call>"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 8,
            },
        ];
        let sse = stream_messages_to_string_tools(
            events,
            false,
            Vec::new(),
            string_param_tools("f", "x"),
        )
        .await;
        assert_eq!(
            stream_stop_reason(&sse),
            "max_tokens",
            "truncation persists past the complete call: {sse}"
        );
        let mut starts: Vec<u64> = Vec::new();
        let mut stops: Vec<u64> = Vec::new();
        let mut tool_uses = 0;
        for block in sse.split("\n\n") {
            let Some(data) = block.lines().find_map(|l| l.strip_prefix("data: ")) else {
                continue;
            };
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            match v["type"].as_str() {
                Some("content_block_start") => {
                    starts.push(v["index"].as_u64().unwrap());
                    if v["content_block"]["type"] == "tool_use" {
                        tool_uses += 1;
                    }
                }
                Some("content_block_stop") => stops.push(v["index"].as_u64().unwrap()),
                _ => {}
            }
        }
        assert_eq!(tool_uses, 2, "two tool_use blocks: {sse}");
        let mut uniq = starts.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(
            uniq.len(),
            starts.len(),
            "indices unique (no reuse): {starts:?}"
        );
        stops.sort_unstable();
        assert_eq!(
            stops, uniq,
            "every opened block closed once, no orphan: {sse}"
        );
    }

    /// A TRUNCATED call, then intervening text, then another call that SPANS feeds —
    /// the hardest ordering case. Source-ordered events keep "between" between f and g,
    /// so f closes (truncated), the text gets its OWN block, and g opens fresh and
    /// streams its full input: unique closed indices, no JSON in the text block, no
    /// orphan stop, max_tokens for the truncated f. The pre-rework heuristic emitted
    /// Start(g) before the text and corrupted the blocks (JSON in the text block, an
    /// orphan stop) — this is the regression guard for that fix.
    #[tokio::test]
    async fn stream_messages_truncated_then_text_then_spanning_call_stays_valid() {
        let mk = |n: &str, p: &str| ToolSchema {
            name: n.into(),
            description: String::new(),
            parameters_json_schema: format!(
                "{{\"type\":\"object\",\"properties\":{{\"{p}\":{{\"type\":\"string\"}}}}}}"
            ),
        };
        let tools = ReplyTools {
            schemas: std::sync::Arc::new(ToolSchemas::from_tools(&[mk("f", "x"), mk("g", "y")])),
            ..ReplyTools::default()
        };
        let events = vec![
            tok("<tool_call>\n<function=f>\n<parameter=x>\n1</tool_call>"),
            tok("between<tool_call>\n<function=g>\n<parameter=y>\n2"),
            tok("\n</parameter>\n</function>\n</tool_call>"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 12,
            },
        ];
        let sse = stream_messages_to_string_tools(events, false, Vec::new(), tools).await;

        let mut starts: Vec<(u64, String)> = Vec::new();
        let mut stops: Vec<u64> = Vec::new();
        let mut text_by_index: std::collections::BTreeMap<u64, String> = Default::default();
        let mut input_by_index: std::collections::BTreeMap<u64, String> = Default::default();
        for block in sse.split("\n\n") {
            let Some(data) = block.lines().find_map(|l| l.strip_prefix("data: ")) else {
                continue;
            };
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            match v["type"].as_str() {
                Some("content_block_start") => starts.push((
                    v["index"].as_u64().unwrap(),
                    v["content_block"]["type"].as_str().unwrap().to_string(),
                )),
                Some("content_block_stop") => stops.push(v["index"].as_u64().unwrap()),
                Some("content_block_delta") => {
                    let i = v["index"].as_u64().unwrap();
                    if let Some(t) = v["delta"]["text"].as_str() {
                        text_by_index.entry(i).or_default().push_str(t);
                    }
                    if let Some(j) = v["delta"]["partial_json"].as_str() {
                        input_by_index.entry(i).or_default().push_str(j);
                    }
                }
                _ => {}
            }
        }
        let indices: Vec<u64> = starts.iter().map(|(i, _)| *i).collect();
        let types: Vec<&str> = starts.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(
            indices,
            vec![0, 1, 2],
            "unique indices f / text / g: {starts:?}"
        );
        assert_eq!(
            types,
            vec!["tool_use", "text", "tool_use"],
            "source order f, between-text, g: {starts:?}"
        );
        stops.sort_unstable();
        assert_eq!(
            stops,
            vec![0, 1, 2],
            "each block closed once, no orphan stop: {sse}"
        );
        assert_eq!(
            text_by_index.get(&1).map(String::as_str),
            Some("between"),
            "the between-text is in the TEXT block: {sse}"
        );
        assert_eq!(
            input_by_index.get(&2).map(String::as_str),
            Some("{\"y\":\"2\"}"),
            "g streams its full, valid input in its OWN block: {sse}"
        );
        assert!(
            !input_by_index.contains_key(&1),
            "no tool JSON leaked into the text block: {sse}"
        );
        assert_eq!(
            stream_stop_reason(&sse),
            "max_tokens",
            "the truncated f forces max_tokens: {sse}"
        );
    }

    /// The Anthropic protocol carries usage on the terminal `message_delta`.
    /// Verify the streaming path actually populates it from the worker's Done
    /// counts (input_tokens / output_tokens) rather than zeros — the streamed
    /// analogue of the non-streaming `usage` object.
    #[tokio::test]
    async fn stream_messages_delta_reports_usage_counts() {
        let sse = stream_messages_to_string(
            vec![
                TokenEvent::Token {
                    token_id: 0,
                    delta_text: "hello".into(),
                },
                TokenEvent::Done {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: 7,
                    completion_tokens: 3,
                },
            ],
            false,
            Vec::new(),
        )
        .await;
        for block in sse.split("\n\n") {
            if block.contains("event: message_delta") {
                let data = block
                    .lines()
                    .find_map(|l| l.strip_prefix("data: "))
                    .expect("message_delta must carry a data line");
                let v: Value = serde_json::from_str(data).unwrap();
                assert_eq!(v["usage"]["input_tokens"], 7, "input_tokens: {data}");
                assert_eq!(v["usage"]["output_tokens"], 3, "output_tokens: {data}");
                return;
            }
        }
        panic!("no message_delta event in stream: {sse}");
    }

    /// The idle keepalive: when no token arrives for the ping interval — a long
    /// prefill or queue wait before the first token — the driver emits
    /// `event: ping` so the client's idle watchdog does not abort the turn. The
    /// incremental streaming covers every in-generation gap; this is the one gap
    /// (before generation starts) that pings cover. Virtual time auto-advances to
    /// the pending timer, so the test does not actually wait.
    #[tokio::test(start_paused = true)]
    async fn stream_messages_emits_ping_during_a_long_idle_gap() {
        let (tx, rx) = mpsc::channel(8);
        let return_sender = tx.clone();
        let pooled = crate::engine::PooledReceiver::new(rx, return_sender, None, 0, None);
        let (body_tx, mut body_rx) = mpsc::channel::<Vec<u8>>(256);
        let driver = tokio::spawn(drive_messages_stream(
            pooled,
            body_tx,
            "test".into(),
            false,
            Vec::new(),
            ReplyTools::default(),
        ));
        // message_start goes out immediately; with no token queued the driver then
        // parks on the select and virtual time auto-advances to fire the ping.
        let start = body_rx.recv().await.unwrap();
        assert!(String::from_utf8_lossy(&start).contains("message_start"));
        let ping = body_rx.recv().await.unwrap();
        assert!(
            String::from_utf8_lossy(&ping).contains("event: ping"),
            "a long idle gap before the first token must emit a keepalive ping"
        );
        // A token + Done then close the turn cleanly (once events flow the ready
        // token wins the select without advancing time, so no further pings).
        tx.send(tok("hi")).await.unwrap();
        tx.send(TokenEvent::Done {
            finish_reason: FinishReason::Stop,
            prompt_tokens: 1,
            completion_tokens: 1,
        })
        .await
        .unwrap();
        drop(tx);
        let mut rest = String::new();
        while let Some(chunk) = body_rx.recv().await {
            rest.push_str(&String::from_utf8_lossy(&chunk));
        }
        assert!(
            rest.contains("message_stop"),
            "turn completes after the ping: {rest}"
        );
        driver.await.unwrap();
    }

    /// The keepalive measures WIRE output, not token arrival: a token that emits
    /// nothing (here a held-back tool-marker prefix) must not reset the clock, so a
    /// ping still fires. This is the long-buffered-argument case — tokens flow while
    /// nothing reaches the wire.
    #[tokio::test(start_paused = true)]
    async fn stream_messages_ping_fires_while_a_token_is_held_back() {
        let (tx, rx) = mpsc::channel(8);
        let return_sender = tx.clone();
        let pooled = crate::engine::PooledReceiver::new(rx, return_sender, None, 0, None);
        let (body_tx, mut body_rx) = mpsc::channel::<Vec<u8>>(256);
        let driver = tokio::spawn(drive_messages_stream(
            pooled,
            body_tx,
            "test".into(),
            false,
            Vec::new(),
            ReplyTools::default(),
        ));
        let start = body_rx.recv().await.unwrap();
        assert!(String::from_utf8_lossy(&start).contains("message_start"));
        // A partial open-marker is held back and emits no wire frame.
        tx.send(tok("<tool")).await.unwrap();
        // The held-back token did not reset the clock, so a ping still fires.
        let ping = body_rx.recv().await.unwrap();
        assert!(
            String::from_utf8_lossy(&ping).contains("event: ping"),
            "a non-emitting token must not suppress the keepalive: {}",
            String::from_utf8_lossy(&ping)
        );
        // Terminate via Done: the driver holds a return-sender clone, so dropping
        // the test's sender alone never closes its input channel.
        tx.send(TokenEvent::Done {
            finish_reason: FinishReason::Stop,
            prompt_tokens: 1,
            completion_tokens: 1,
        })
        .await
        .unwrap();
        drop(tx);
        let mut rest = String::new();
        while let Some(chunk) = body_rx.recv().await {
            rest.push_str(&String::from_utf8_lossy(&chunk));
        }
        // The held "<tool" turned out not to be a marker; it flushes as text.
        assert!(
            rest.contains("message_stop"),
            "turn still completes: {rest}"
        );
        driver.await.unwrap();
    }

    /// A streamed tool-call turn must report the terminal stop_reason
    /// `tool_use` (the worker reports a natural `Stop`; the wire layer upgrades
    /// it — matching the OpenAI streaming + non-streaming Anthropic paths).
    /// Regression guard: before the fix this reported "end_turn".
    #[tokio::test]
    async fn stream_messages_tool_call_reports_tool_use_stop_reason() {
        use lumen_runtime::tooling::Qwen35Renderer;
        let call = Qwen35Renderer::render_one_call("get_weather", "{\"city\": \"Paris\"}");
        let events = vec![
            tok("Let me check. "),
            tok(&call),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 3,
                completion_tokens: 8,
            },
        ];
        let sse = stream_messages_to_string(events, false, Vec::new()).await;
        assert!(
            sse.contains("\"type\":\"tool_use\""),
            "stream must carry a tool_use content block: {sse}"
        );
        assert_eq!(
            stream_stop_reason(&sse),
            "tool_use",
            "a streamed tool-call turn must report stop_reason:tool_use"
        );
    }

    /// Tool-use ids are unique across responses as well as within one: a
    /// client keeps every earlier turn's ids in its history, and one that
    /// meets an id twice drops a call.
    #[tokio::test]
    async fn tool_use_ids_are_unique_across_responses() {
        use lumen_runtime::tooling::Qwen35Renderer;
        let call = Qwen35Renderer::render_one_call("get_weather", "{\"city\": \"Paris\"}");
        let events = || {
            vec![
                tok(&call),
                tok(&call),
                TokenEvent::Done {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: 3,
                    completion_tokens: 16,
                },
            ]
        };
        let mut ids = Vec::new();
        for _ in 0..2 {
            let body = collect_messages_from_events(events(), false).await;
            for b in body["content"].as_array().unwrap() {
                if b["type"] == "tool_use" {
                    ids.push(b["id"].as_str().unwrap().to_string());
                }
            }
            let sse = stream_messages_to_string(events(), false, Vec::new()).await;
            for part in sse.split("\"id\":\"toolu_").skip(1) {
                ids.push(format!("toolu_{}", part.split('"').next().unwrap()));
            }
        }
        assert_eq!(ids.len(), 8, "{ids:?}");
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "{ids:?}");
    }

    /// A plain (no-tool) streamed turn is unchanged: stop_reason `end_turn`.
    #[tokio::test]
    async fn stream_messages_normal_turn_reports_end_turn_stop_reason() {
        let events = vec![
            tok("Just a plain answer."),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 3,
                completion_tokens: 4,
            },
        ];
        let sse = stream_messages_to_string(events, false, Vec::new()).await;
        assert!(
            !sse.contains("\"type\":\"tool_use\""),
            "a normal turn must not emit a tool_use block: {sse}"
        );
        assert_eq!(
            stream_stop_reason(&sse),
            "end_turn",
            "a normal turn must keep stop_reason:end_turn"
        );
    }

    /// The upgrade is conservative (mirrors OpenAI): a non-`Stop` worker reason
    /// is never rewritten, even with a tool block present. `Length` stays
    /// `max_tokens`.
    #[tokio::test]
    async fn stream_messages_tool_call_under_length_keeps_max_tokens() {
        use lumen_runtime::tooling::Qwen35Renderer;
        let call = Qwen35Renderer::render_one_call("get_weather", "{\"city\": \"Paris\"}");
        let events = vec![
            tok(&call),
            TokenEvent::Done {
                finish_reason: FinishReason::Length,
                prompt_tokens: 3,
                completion_tokens: 8,
            },
        ];
        let sse = stream_messages_to_string(events, false, Vec::new()).await;
        assert_eq!(
            stream_stop_reason(&sse),
            "max_tokens",
            "a Length finish must not be upgraded to tool_use"
        );
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
        let req: MessagesRequest =
            serde_json::from_value(body).expect("Anthropic top_p/top_k must deserialize, not 400");
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

    #[tokio::test]
    async fn messages_unknown_top_level_fields_are_ignored() {
        let engine = EngineHandle::new_for_test(4096);
        let plain = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16
        });
        let mut extended = plain.clone();
        for (k, v) in [
            ("metadata", json!({"user_id": "u"})),
            ("output_config", json!({"effort": "low"})),
            (
                "context_management",
                json!({"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}),
            ),
            ("safeguards", json!([{"type": "dangerous_tool_use"}])),
            ("definitely_not_a_field", json!(1)),
        ] {
            extended[k] = v;
        }
        let plain: MessagesRequest = serde_json::from_value(plain).unwrap();
        let extended: MessagesRequest = serde_json::from_value(extended)
            .expect("unknown top-level fields must be ignored, not rejected");
        let (plain, extended) = (
            plain.into_job(&engine).unwrap(),
            extended.into_job(&engine).unwrap(),
        );
        assert_eq!(plain.prompt_tokens, extended.prompt_tokens);
        assert_eq!(plain.max_tokens, extended.max_tokens);
    }

    #[test]
    fn effort_reaches_the_template_as_reasoning_effort() {
        let template =
            include_str!("../../../lumen-runtime/tests/fixtures/qwen38_chat_template.jinja");
        let request = |output_config: Value| -> MessagesRequest {
            serde_json::from_value(json!({
                "model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "adaptive"}, "output_config": output_config,
            }))
            .unwrap()
        };
        let preamble = |output_config: Value| {
            let req = request(output_config);
            let effort = req.reasoning_effort().unwrap();
            let prompt =
                render_prompt_templated(None, &req.messages, &[], true, effort, template).unwrap();
            prompt
                .split("<|im_end|>")
                .next()
                .filter(|head| head.starts_with("<|im_start|>system\n"))
                .map(str::to_owned)
        };
        let xhigh = preamble(json!(null));
        assert!(xhigh
            .as_deref()
            .unwrap()
            .contains("Reasoning effort is set to xhigh"));
        for level in ["high", "xhigh", "max"] {
            assert_eq!(preamble(json!({"effort": level})), xhigh, "{level}");
        }
        assert!(preamble(json!({"effort": "low"}))
            .unwrap()
            .contains("Reasoning effort is set to low"));
        assert_eq!(preamble(json!({"effort": "medium"})), None);
        for bad in [json!("minimal"), json!(1)] {
            match request(json!({"effort": bad})).reasoning_effort() {
                Err(ServerError::BadRequest { param, .. }) => {
                    assert_eq!(param.as_deref(), Some("output_config.effort"))
                }
                other => panic!("{bad}: expected a 400, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn count_tokens_is_the_prompt_messages_would_run() {
        let engine = EngineHandle::new_for_test(8192);
        let body = json!({
            "model": "m",
            "system": "Be brief.",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "R"},
                    {"type": "text", "text": "hello"},
                ]},
                {"role": "user", "content": "weather?"},
            ],
            "tools": [{"name": "f", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "none"},
            "thinking": {"type": "disabled"},
        });
        let counted = serde_json::from_value::<MessagesRequest>(body.clone())
            .unwrap()
            .count_tokens(&engine)
            .unwrap();
        let mut run = body;
        run["max_tokens"] = json!(16);
        let job = serde_json::from_value::<MessagesRequest>(run)
            .unwrap()
            .into_job(&engine)
            .unwrap();
        assert_eq!(counted, job.prompt_tokens.len());
        // A request `/v1/messages` refuses is refused here too.
        let refused = serde_json::from_value::<MessagesRequest>(json!({
            "model": "m", "messages": [], "mcp_servers": [{"type": "url", "url": "https://x"}],
        }))
        .unwrap()
        .count_tokens(&engine);
        assert!(matches!(refused, Err(ServerError::BadRequest { .. })));
    }

    #[tokio::test]
    async fn messages_refuse_what_they_cannot_produce() {
        let engine = EngineHandle::new_for_test(4096);
        let job = |extra: Value| {
            let mut body = json!({
                "model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}],
            });
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            serde_json::from_value::<MessagesRequest>(body)
                .expect("the request parses")
                .into_job(&engine)
                .map_err(|e| match e {
                    ServerError::BadRequest { param, code, .. } => (param, code),
                    other => panic!("expected a 400, got {other:?}"),
                })
        };
        let plain = job(json!({})).unwrap();
        let tool = json!({"name": "f", "input_schema": {"type": "object"}});
        let (param, code) =
            job(json!({"tool_choice": {"type": "auto", "disable_parallel_tool_use": "true"}}))
                .unwrap_err();
        assert_eq!(
            (param.as_deref(), code.as_deref()),
            (
                Some("tool_choice.disable_parallel_tool_use"),
                Some("invalid_type")
            )
        );
        let single = |choice: Value| {
            serde_json::from_value::<MessagesRequest>(json!({
                "model": "m", "max_tokens": 16, "messages": [], "tool_choice": choice,
            }))
            .unwrap()
            .reply_tools()
            .single
        };
        assert!(single(
            json!({"type": "auto", "disable_parallel_tool_use": true})
        ));
        assert!(!single(
            json!({"type": "auto", "disable_parallel_tool_use": false})
        ));
        assert!(!single(json!({"type": "auto"})));
        for accepted in [
            json!({"tool_choice": {"type": "auto"}, "mcp_servers": [], "container": null}),
            json!({"tool_choice": {"type": "auto", "disable_parallel_tool_use": false}}),
            json!({"tool_choice": {"type": "none"}}),
            json!({"tool_choice": {"type": "auto", "disable_parallel_tool_use": true}}),
        ] {
            assert_eq!(job(accepted).unwrap().prompt_tokens, plain.prompt_tokens);
        }
        let custom = json!({"type": "custom", "name": "f", "input_schema": {"type": "object"}});
        assert_eq!(
            job(json!({"tools": [custom]})).unwrap().prompt_tokens,
            job(json!({"tools": [tool]})).unwrap().prompt_tokens
        );
        // Forcing a call needs the model's chat template, which this engine
        // lacks (see `tool_choice_matches_the_messages_endpoint`).
        for (extra, field) in [
            (
                json!({"tool_choice": {"type": "any"}, "tools": [tool.clone()]}),
                "tool_choice",
            ),
            (
                json!({"tool_choice": {"type": "tool", "name": "f"}}),
                "tool_choice",
            ),
            (
                json!({"mcp_servers": [{"type": "url", "url": "https://x", "name": "x"}]}),
                "mcp_servers",
            ),
            (json!({"container": "c"}), "container"),
            (
                json!({"output_format": {"type": "json_schema"}}),
                "output_format",
            ),
            (
                json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
                "tools[].type",
            ),
        ] {
            let (param, code) = job(extra.clone()).unwrap_err();
            assert_eq!(param.as_deref(), Some(field), "{extra}");
            assert_eq!(code.as_deref(), Some("invalid_value"), "{extra}");
        }
    }

    #[tokio::test]
    async fn messages_output_format_is_refused_not_ignored() {
        let engine = EngineHandle::new_for_test(4096);
        let request = |output_config: Value| -> MessagesRequest {
            serde_json::from_value(json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "output_config": output_config
            }))
            .unwrap()
        };
        for format in [
            json!({"type": "json_schema", "schema": {"type": "object"}}),
            json!("json"),
            json!({}),
        ] {
            let err = request(json!({"effort": "low", "format": format}))
                .into_job(&engine)
                .unwrap_err();
            let resp = axum::response::IntoResponse::into_response(err);
            assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
            let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["error"]["param"], "output_config.format");
            assert_eq!(body["error"]["code"], "invalid_value");
        }
        for accepted in [
            json!(null),
            json!({"effort": "low"}),
            json!({"format": null}),
        ] {
            assert!(request(accepted).into_job(&engine).is_ok());
        }
        for not_an_object in [json!("low"), json!([]), json!(["low", null])] {
            let body = json!({
                "model": "m", "max_tokens": 16, "messages": [], "output_config": not_an_object
            });
            let err = serde_json::from_value::<MessagesRequest>(body)
                .unwrap_err()
                .to_string();
            assert!(err.contains("expected a map"), "{err}");
        }
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
        assert!(
            out.contains("<tool_call>"),
            "tool_use must render a tool_call: {out}"
        );
        assert!(out.contains("get_weather"), "tool name present");
        // `input` object serializes COMPACT via serde_json::Value::to_string().
        assert!(
            out.contains("{\"city\":\"Paris\"}"),
            "input serialized as compact arguments"
        );
        assert!(
            out.contains("<tool_response>"),
            "tool_result must render a tool_response"
        );
        assert!(
            out.contains("{\"temp\": 18}"),
            "tool result content present"
        );
    }
}
