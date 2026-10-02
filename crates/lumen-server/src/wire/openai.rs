//! OpenAI-compatible endpoints: `/v1/chat/completions` + `/v1/completions`.
//!
//! References:
//! - <https://platform.openai.com/docs/api-reference/chat>
//! - <https://platform.openai.com/docs/api-reference/completions>
//!
//! A field a request body does not declare is ignored, as on `/v1/messages`,
//! unless it asks for output this server cannot produce: those are listed in
//! `OPENAI_UNSUPPORTED`, `CHAT_UNSUPPORTED` and `COMPLETION_UNSUPPORTED`
//! and refused with a 400 (see `super::refuse_unsupported`).

use axum::body::Body;
use lumen_runtime::engine::SamplingParams;
use lumen_runtime::tooling::{
    compose_system_with_tools, StreamEvent, ToolSchema, ToolSchemas, ToolStreamEvent,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::engine::{EngineHandle, FinishReason, JobRequest, JobResponseChannel, TokenEvent};
use crate::error::ServerError;
use crate::sse::{ReplyTools, SseSafeEmitter};
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
    /// An earlier assistant turn's reasoning, as this server returns it; the
    /// chat template renders it back into the turn.
    #[serde(default)]
    pub reasoning_content: Option<String>,
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
/// is `enable_thinking`; any other keys are ignored. This mirrors the vLLM
/// OpenAI server, which accepts
/// `{"chat_template_kwargs": {"enable_thinking": false}}` to toggle the
/// Qwen3.5 reasoning block.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatTemplateKwargs {
    #[serde(default)]
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// The newer name for `max_tokens`; wins when both are given.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Nucleus-sampling cutoff (OpenAI `top_p`), honored as on the CLI. `None`
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
    /// OpenAI `stream_options` (consulted on the streaming path only). The one
    /// field Lumen reads is `include_usage`: when true, the stream emits ONE
    /// final chunk with empty `choices` and the `usage` totals BEFORE
    /// `data: [DONE]` (the OpenAI contract); absent or false, no usage chunk
    /// is emitted — byte-identical to the historical stream shape.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub stop: Option<Value>,
    /// Keep decoding past the model's end-of-sequence tokens (which then
    /// render nothing) until `max_tokens` or another stop. Off by default.
    #[serde(default)]
    pub ignore_eos: bool,
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
    /// Reasoning-token cap within `max_tokens` (like Anthropic
    /// `thinking.budget_tokens`); the shared default when absent.
    #[serde(default)]
    pub reasoning_budget: Option<usize>,
    /// vLLM-compatible `{"chat_template_kwargs": {"enable_thinking": ...}}`.
    /// The top-level `enable_thinking` field wins when both are present.
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// OpenAI reasoning effort: `none`, `minimal`, `low`, `medium`, `high`,
    /// `xhigh` or `max`. See `Self::reasoning_effort` and
    /// [`Self::resolve_thinking`].
    #[serde(default)]
    pub reasoning_effort: Option<Value>,
    /// `none`, `auto`, `required` or `{"type": "function", "function":
    /// {"name": ...}}`; see `Self::tool_choice`.
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// `false`: at most one tool call per reply. A boolean; see
    /// [`Self::into_job`].
    #[serde(default)]
    pub parallel_tool_calls: Option<Value>,
    /// Every field the request does not declare; see `CHAT_UNSUPPORTED`.
    #[serde(flatten)]
    pub other: serde_json::Map<String, Value>,
}

/// Fields both OpenAI endpoints refuse unless left at their default: the ones
/// the two share, then the extensions other OpenAI-compatible servers honour,
/// since a client sending them expects them to apply.
const OPENAI_UNSUPPORTED: &[super::Unsupported] = &[
    super::Unsupported {
        field: "n",
        accepts: |v| super::is_number(v, 1.0),
        refused: "more than one choice",
    },
    super::Unsupported {
        field: "best_of",
        accepts: |v| super::is_number(v, 1.0),
        refused: "choosing among several completions",
    },
    super::Unsupported {
        field: "use_beam_search",
        accepts: |v| *v == false,
        refused: "beam search",
    },
    super::Unsupported {
        field: "length_penalty",
        accepts: |v| super::is_number(v, 1.0),
        refused: "a length penalty",
    },
    super::Unsupported {
        field: "echo",
        accepts: |v| *v == false,
        refused: "echoing the prompt",
    },
    super::Unsupported {
        field: "logit_bias",
        accepts: super::is_zero_bias,
        refused: "biasing token probabilities",
    },
    super::Unsupported {
        field: "prompt_logprobs",
        accepts: |_| false,
        refused: "returning log probabilities",
    },
    super::Unsupported {
        field: "return_tokens_as_token_ids",
        accepts: |v| *v == false,
        refused: "returning token ids",
    },
    super::Unsupported {
        field: "response_format",
        accepts: |v| v["type"] == "text",
        refused: "structured output",
    },
    super::Unsupported {
        field: "structural_tag",
        accepts: |_| false,
        refused: "constrained output",
    },
    super::Unsupported {
        field: "guided_json",
        accepts: |_| false,
        refused: "constrained output",
    },
    super::Unsupported {
        field: "guided_choice",
        accepts: |_| false,
        refused: "constrained output",
    },
    super::Unsupported {
        field: "guided_regex",
        accepts: |_| false,
        refused: "constrained output",
    },
    super::Unsupported {
        field: "guided_grammar",
        accepts: |_| false,
        refused: "constrained output",
    },
    super::Unsupported {
        field: "structured_outputs",
        accepts: |_| false,
        refused: "constrained output",
    },
    super::Unsupported {
        field: "json_schema",
        accepts: |_| false,
        refused: "constrained output",
    },
    super::Unsupported {
        field: "regex",
        accepts: |_| false,
        refused: "constrained output",
    },
    super::Unsupported {
        field: "ebnf",
        accepts: |_| false,
        refused: "constrained output",
    },
    super::Unsupported {
        field: "grammar",
        accepts: |_| false,
        refused: "constrained output",
    },
    super::Unsupported {
        field: "min_tokens",
        accepts: |v| super::is_number(v, 0.0),
        refused: "a minimum length",
    },
    super::Unsupported {
        field: "stop_token_ids",
        accepts: |v| v.as_array().is_some_and(|a| a.is_empty()),
        refused: "stop token ids",
    },
    super::Unsupported {
        field: "include_stop_str_in_output",
        accepts: |v| *v == false,
        refused: "keeping the stop string",
    },
    super::Unsupported {
        field: "no_stop_trim",
        accepts: |v| *v == false,
        refused: "keeping the stop string",
    },
    super::Unsupported {
        field: "repetition_penalty",
        accepts: |_| false,
        refused: "a per-request repetition penalty",
    },
    super::Unsupported {
        field: "bad_words",
        accepts: |v| v.as_array().is_some_and(|a| a.is_empty()),
        refused: "banned words",
    },
    super::Unsupported {
        field: "allowed_token_ids",
        accepts: |v| v.as_array().is_some_and(|a| a.is_empty()),
        refused: "restricting the vocabulary",
    },
    super::Unsupported {
        field: "logits_processors",
        accepts: |_| false,
        refused: "logits processors",
    },
    super::Unsupported {
        field: "custom_logit_processor",
        accepts: |_| false,
        refused: "logits processors",
    },
    super::Unsupported {
        field: "truncate_prompt_tokens",
        accepts: |_| false,
        refused: "truncating the prompt",
    },
    super::Unsupported {
        field: "skip_special_tokens",
        accepts: |v| *v == true,
        refused: "keeping special tokens in the text",
    },
    super::Unsupported {
        field: "spaces_between_special_tokens",
        accepts: |v| *v == true,
        refused: "changing special-token spacing",
    },
    super::Unsupported {
        field: "lora_path",
        accepts: |_| false,
        refused: "LoRA adapters",
    },
    super::Unsupported {
        field: "reasoning",
        accepts: |_| false,
        refused: "a `reasoning` object (use `reasoning_effort`)",
    },
];

/// Chat fields that must stay at their default, besides [`OPENAI_UNSUPPORTED`].
const CHAT_UNSUPPORTED: &[super::Unsupported] = &[
    super::Unsupported {
        field: "logprobs",
        accepts: |v| *v == false,
        refused: "returning log probabilities",
    },
    super::Unsupported {
        field: "top_logprobs",
        accepts: |v| super::is_number(v, 0.0),
        refused: "returning log probabilities",
    },
    super::Unsupported {
        field: "verbosity",
        accepts: |v| v == "medium",
        refused: "a verbosity other than `medium`",
    },
    super::Unsupported {
        field: "moderation",
        accepts: |_| false,
        refused: "moderation",
    },
    super::Unsupported {
        field: "functions",
        accepts: |v| v.as_array().is_some_and(|a| a.is_empty()),
        refused: "legacy function calling (use `tools`)",
    },
    super::Unsupported {
        field: "function_call",
        accepts: |v| v == "none" || v == "auto",
        refused: "legacy function calling (use `tools`)",
    },
    super::Unsupported {
        field: "modalities",
        accepts: |v| v.as_array().is_some_and(|a| a.iter().all(|m| m == "text")),
        refused: "output other than text",
    },
    super::Unsupported {
        field: "audio",
        accepts: |_| false,
        refused: "audio output",
    },
    super::Unsupported {
        field: "web_search_options",
        accepts: |_| false,
        refused: "web search",
    },
    super::Unsupported {
        field: "add_generation_prompt",
        accepts: |v| *v == true,
        refused: "rendering without the assistant prompt",
    },
    super::Unsupported {
        field: "continue_final_message",
        accepts: |v| *v == false,
        refused: "continuing the last message",
    },
    super::Unsupported {
        field: "add_special_tokens",
        accepts: |v| *v == false,
        refused: "adding special tokens to the prompt",
    },
    super::Unsupported {
        field: "chat_template",
        accepts: |_| false,
        refused: "a request chat template",
    },
    super::Unsupported {
        field: "documents",
        accepts: |_| false,
        refused: "documents",
    },
    super::Unsupported {
        field: "mm_processor_kwargs",
        accepts: |_| false,
        refused: "multimodal input",
    },
    super::Unsupported {
        field: "separate_reasoning",
        accepts: |v| *v == true,
        refused: "reasoning mixed into the answer",
    },
    super::Unsupported {
        field: "stream_reasoning",
        accepts: |v| *v == true,
        refused: "withholding streamed reasoning",
    },
    super::Unsupported {
        field: "return_hidden_states",
        accepts: |v| *v == false,
        refused: "returning hidden states",
    },
];

/// Completion fields that must stay at their default, besides
/// [`OPENAI_UNSUPPORTED`].
const COMPLETION_UNSUPPORTED: &[super::Unsupported] = &[
    super::Unsupported {
        field: "suffix",
        accepts: |v| v == "",
        refused: "a suffix after the completion",
    },
    super::Unsupported {
        field: "logprobs",
        accepts: |_| false,
        refused: "returning log probabilities",
    },
];

/// OpenAI `stream_options` object (streaming requests only). Other keys are
/// ignored like unknown request fields.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct StreamOptions {
    /// When true, emit the final usage chunk (empty `choices` + `usage`)
    /// before `data: [DONE]`.
    #[serde(default)]
    pub include_usage: Option<bool>,
}

impl ChatCompletionRequest {
    /// True when the streaming client asked for the final usage chunk
    /// (`stream_options: {"include_usage": true}`). Consulted only on the
    /// streaming path; the non-streaming response always carries `usage`.
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|o| o.include_usage)
            .unwrap_or(false)
    }

    /// Resolve the per-request reasoning toggle using the single shared
    /// resolver. Precedence: top-level `enable_thinking` → vLLM
    /// `chat_template_kwargs.enable_thinking` → `reasoning_effort` (`none`
    /// means no reasoning, any other level asks for it) → env override →
    /// default. The per-request `Option` is collapsed here, then handed to the
    /// one resolver so the env/default fall-through is identical to every
    /// other surface.
    pub fn resolve_thinking(&self) -> bool {
        let per_request = self
            .enable_thinking
            .or_else(|| {
                self.chat_template_kwargs
                    .as_ref()
                    .and_then(|k| k.enable_thinking)
            })
            .or_else(|| {
                self.reasoning_effort
                    .as_ref()
                    .and_then(Value::as_str)
                    .map(|e| e != "none")
            });
        super::resolve_enable_thinking(per_request)
    }

    fn tool_choice(&self) -> Result<super::ToolChoice, ServerError> {
        use super::ToolChoice;
        let choice = self.tool_choice.as_ref().unwrap_or(&Value::Null);
        match choice.as_str() {
            _ if choice.is_null() => Ok(ToolChoice::Auto),
            Some("auto") => Ok(ToolChoice::Auto),
            Some("none") => Ok(ToolChoice::None),
            Some("required") => Ok(ToolChoice::Required),
            _ => match (&choice["type"], choice["function"]["name"].as_str()) {
                (t, Some(name)) if t == "function" => Ok(ToolChoice::Named(name.into())),
                _ => Err(ServerError::bad_request_field(
                    "tool_choice must be `none`, `auto`, `required` or {\"type\": \"function\", \"function\": {\"name\": ...}}",
                    "tool_choice",
                    "invalid_value",
                )),
            },
        }
    }

    /// The tool calls the reply may carry, for the collectors. Taken before
    /// `into_job` consumes the request; a malformed `tool_choice` reads as
    /// `auto` here because `into_job` refuses it.
    pub fn reply_tools(&self) -> ReplyTools {
        self.tool_choice()
            .unwrap_or(super::ToolChoice::Auto)
            .reply_tools(
                tool_schemas(&self.tools),
                self.parallel_tool_calls == Some(Value::Bool(false)),
            )
    }

    /// The prompt for this request and the text the reply starts with, which
    /// also ends the prompt (see [`super::ToolChoice::response_prefix`]).
    fn prompt(&self, chat_template: Option<&str>) -> Result<(String, String), ServerError> {
        let reasoning_effort = self.reasoning_effort()?;
        let enable_thinking = self.resolve_thinking();
        let tool_choice = self.tool_choice()?;
        tool_choice.check(
            self.tools.iter().map(|t| t.function.name.as_str()),
            enable_thinking,
            chat_template.is_some(),
        )?;
        let tools: &[ToolDef] = if tool_choice.offers_tools() {
            &self.tools
        } else {
            &[]
        };
        let mut prompt = render_chat_prompt(
            &self.messages,
            tools,
            enable_thinking,
            chat_template,
            reasoning_effort,
        )?;
        let response_prefix = tool_choice.response_prefix();
        prompt.push_str(&response_prefix);
        Ok((prompt, response_prefix))
    }

    /// `reasoning_effort` as the chat template's `reasoning_effort`: the
    /// levels shared with `/v1/messages` map through
    /// [`super::template_reasoning_effort`], `minimal` runs as `low` (the
    /// closest level a template offers) and `none` needs none (thinking is off,
    /// see [`Self::resolve_thinking`]). Any other value is refused.
    fn reasoning_effort(&self) -> Result<Option<&'static str>, ServerError> {
        const PARAM: &str = "reasoning_effort";
        const LEVELS: &str = "`none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`";
        match super::effort_level(self.reasoning_effort.as_ref(), PARAM, LEVELS)? {
            None | Some("none") => Ok(None),
            Some("minimal") => Ok(Some("low")),
            Some(level) => super::template_reasoning_effort(level).ok_or_else(|| {
                ServerError::bad_request_field(
                    format!("{PARAM} must be one of {LEVELS}"),
                    PARAM,
                    "invalid_value",
                )
            }),
        }
    }

    pub fn into_job(self, engine: &EngineHandle) -> Result<JobRequest, ServerError> {
        // ROBUST-007 (2026-06-11 checklist): out-of-range sampler params and
        // empty `messages` must 400 like other malformed fields, not be
        // silently accepted/clamped.
        super::refuse_unsupported(&self.other, CHAT_UNSUPPORTED)?;
        super::refuse_unsupported(&self.other, OPENAI_UNSUPPORTED)?;
        if self
            .parallel_tool_calls
            .as_ref()
            .is_some_and(|p| !p.is_null() && !p.is_boolean())
        {
            return Err(ServerError::bad_request_field(
                "parallel_tool_calls must be a boolean",
                "parallel_tool_calls",
                "invalid_type",
            ));
        }
        validate_sampler_ranges(self.temperature, self.top_p)?;
        if self.messages.is_empty() {
            return Err(ServerError::bad_request_field(
                "messages must be a non-empty array",
                "messages",
                "invalid_value",
            ));
        }
        let (prompt, response_prefix) = self.prompt(engine.chat_template())?;
        let enable_thinking = self.resolve_thinking();
        let prompt_tokens = engine.tokenize_for_request(&prompt);
        // Synchronous oversize guard: 400 BEFORE the 200/SSE stream opens.
        super::check_prompt_length(prompt_tokens.len(), engine.context_length())?;
        let stop_text = parse_stop_field(self.stop);
        let eos = engine.eos_tokens_for_request();
        // Reasoning included; absent, only the context window bounds it.
        let max_tokens = self
            .max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(usize::MAX);
        // server-internal sampler defaults aligned with CLI's
        // production defaults (`--repeat-penalty 1.05`). The
        // OpenAI API surface is preserved: the `repetition_penalty` field
        // is NOT in the request schema (OpenAI does not expose it) so this
        // default applies only on the server-internal codepath, and only when
        // sampling: at greedy decoding (`temperature <= 0.0`) a repetition penalty
        // would reshape the logits BEFORE the argmax and change the chosen token,
        // so `diag_repetition_penalty` defaults it to 1.0 there (unless the env
        // override is set). See its doc for the resolution order and the tradeoff.
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
        let temperature = self
            .temperature
            .unwrap_or_else(lumen_runtime::runtime_defaults::default_temperature);
        let sampling = SamplingParams {
            temperature,
            seed: Some(self.seed.unwrap_or_else(super::next_random_seed)),
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repetition_penalty: Some(super::diag_repetition_penalty(temperature)),
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
            ignore_eos: self.ignore_eos,
            sampling,
            suffix_threshold: lumen_runtime::session::Session::DEFAULT_SUFFIX_THRESHOLD,
            enable_thinking,
            reasoning_budget: self
                .reasoning_budget
                .unwrap_or_else(lumen_runtime::runtime_defaults::chat_reasoning_budget_default),
            response_prefix,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
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
    /// (see `ChatCompletionRequest`), honored as on the CLI. Same
    /// zero-normalization / override semantics as the chat path.
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
    /// Keep decoding past the model's end-of-sequence tokens (which then
    /// render nothing) until `max_tokens` or another stop. Off by default.
    #[serde(default)]
    pub ignore_eos: bool,
    /// Streaming only: `include_usage` adds the final usage chunk, as on chat.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Every field the request does not declare; see `COMPLETION_UNSUPPORTED`.
    #[serde(flatten)]
    pub other: serde_json::Map<String, Value>,
}

impl CompletionRequest {
    /// True when the streaming client asked for the final usage chunk.
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|o| o.include_usage)
            .unwrap_or(false)
    }

    pub fn into_job(self, engine: &EngineHandle) -> Result<JobRequest, ServerError> {
        super::refuse_unsupported(&self.other, COMPLETION_UNSUPPORTED)?;
        super::refuse_unsupported(&self.other, OPENAI_UNSUPPORTED)?;
        // ROBUST-007: same sampler-range guard as the chat endpoint.
        validate_sampler_ranges(self.temperature, self.top_p)?;
        let prompt_tokens = completion_prompt_tokens(self.prompt, engine)?;
        // The engine treats an empty prompt as "continue what is loaded", which
        // here is the previous request's context.
        if prompt_tokens.is_empty() {
            return Err(ServerError::bad_request_field(
                "prompt must not be empty",
                "prompt",
                "invalid_value",
            ));
        }
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
        let temperature = self
            .temperature
            .unwrap_or_else(lumen_runtime::runtime_defaults::default_temperature);
        let sampling = SamplingParams {
            temperature,
            seed: Some(self.seed.unwrap_or_else(super::next_random_seed)),
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repetition_penalty: Some(super::diag_repetition_penalty(temperature)),
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
            ignore_eos: self.ignore_eos,
            sampling,
            suffix_threshold: lumen_runtime::session::Session::DEFAULT_SUFFIX_THRESHOLD,
            // Legacy text-completions have no chat template / `<think>` block,
            // so reasoning is never enabled on this path.
            enable_thinking: false,
            reasoning_budget: 0,
            response_prefix: String::new(),
        })
    }
}

/// Resolve a legacy-completions `prompt` to token ids. A string, or an array
/// of strings (concatenated), is tokenized; an array of integers is taken as
/// token ids and reaches the engine unchanged, each checked against the
/// model's vocabulary. An array mixing the two, or holding any other value
/// (a nested array, a float, a negative number), is refused rather than
/// partly ignored.
fn completion_prompt_tokens(prompt: Value, engine: &EngineHandle) -> Result<Vec<u32>, ServerError> {
    let invalid = || {
        ServerError::bad_request_field(
            "prompt must be a string, an array of strings, or an array of token ids",
            "prompt",
            "invalid_type",
        )
    };
    let arr = match prompt {
        Value::String(s) => return Ok(engine.tokenize_for_request(&s)),
        Value::Array(arr) => arr,
        _ => return Err(invalid()),
    };
    if arr.iter().all(Value::is_string) {
        let text: String = arr.iter().filter_map(Value::as_str).collect();
        return Ok(engine.tokenize_for_request(&text));
    }
    let vocab_size = engine.vocab_size();
    arr.iter()
        .map(|v| {
            let id = v
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(invalid)?;
            if id as usize >= vocab_size {
                return Err(ServerError::bad_request_field(
                    format!(
                        "prompt token id {id} is out of range for a vocabulary of {vocab_size}"
                    ),
                    "prompt",
                    "invalid_value",
                ));
            }
            Ok(id)
        })
        .collect()
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

/// Build the runtime [`ToolSchemas`] (function -> parameter -> JSON-Schema type)
/// the native tool-call parser needs, from the OpenAI tool definitions on a
/// request. Mirrors the `ToolSchema` conversion the manual render uses.
fn tool_schemas(tools: &[ToolDef]) -> ToolSchemas {
    let schemas: Vec<ToolSchema> = tools
        .iter()
        .map(|t| ToolSchema {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            parameters_json_schema: serde_json::to_string(&t.function.parameters)
                .unwrap_or_else(|_| "{}".into()),
        })
        .collect();
    ToolSchemas::from_tools(&schemas)
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
    chat_template: Option<&str>,
    reasoning_effort: Option<&str>,
) -> Result<String, ServerError> {
    // Prefer the model's EMBEDDED chat template (Qwen3.5's native tool-calling
    // protocol) rendered via the shared Jinja engine — the SAME renderer the CLI
    // uses, so the two cannot drift. Falls back to the hard-coded ChatML
    // transcript below when no template is embedded (older LBCs / synthetic test
    // tokenizers), keeping those paths byte-identical to today.
    if let Some(template) = chat_template {
        return render_chat_prompt_templated(
            messages,
            tools,
            enable_thinking,
            reasoning_effort,
            template,
        );
    }
    render_chat_prompt_manual(messages, tools, enable_thinking)
}

/// Render `messages` + `tools` through the model's embedded Jinja template.
///
/// Builds the render context in the shape the template consumes: message
/// `content` is flattened to a string via the shared [`super::flatten_content`]
/// (preserving the ROBUST-007 numeric-content 400), assistant `tool_calls` are
/// mapped to `{function: {name, arguments}}` with the OpenAI on-wire arguments
/// JSON STRING parsed back into an object (the template iterates it with
/// `|items`), and each tool is the OpenAI function-tool object. A template
/// `raise_exception` (e.g. "No user query found", "System message must be at the
/// beginning") surfaces as a 400.
fn render_chat_prompt_templated(
    messages: &[ChatMessage],
    tools: &[ToolDef],
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
    template: &str,
) -> Result<String, ServerError> {
    let mut msgs: Vec<Value> = Vec::with_capacity(messages.len());
    for m in messages {
        let content = super::flatten_content(&m.content, "messages.content")?;
        let mut obj = serde_json::Map::new();
        obj.insert("role".into(), Value::String(m.role.clone()));
        obj.insert("content".into(), Value::String(content));
        // Passed even when empty: a template that finds none looks for
        // reasoning inside the content instead (Qwen3.5 splits it at `</think>`).
        if let Some(reasoning) = m
            .reasoning_content
            .as_ref()
            .filter(|_| m.role == "assistant")
        {
            obj.insert("reasoning_content".into(), Value::String(reasoning.clone()));
        }
        if !m.tool_calls.is_empty() {
            let calls: Vec<Value> = m
                .tool_calls
                .iter()
                .map(|tc| {
                    // OpenAI carries arguments as a JSON string; the template
                    // needs a mapping (`arguments | items`), so parse it back.
                    let args = serde_json::from_str::<Value>(&tc.function.arguments)
                        .ok()
                        .filter(Value::is_object)
                        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                    json!({
                        "id": tc.id,
                        "type": tc.call_type,
                        "function": {"name": tc.function.name, "arguments": args},
                    })
                })
                .collect();
            obj.insert("tool_calls".into(), Value::Array(calls));
        }
        msgs.push(Value::Object(obj));
    }
    let tools_json: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "type": t.def_type,
                "function": {
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters,
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

/// Hard-coded ChatML transcript, retained as the fallback for tokenizers with no
/// embedded template. Byte-identical to the pre-embedded-template behaviour.
fn render_chat_prompt_manual(
    messages: &[ChatMessage],
    tools: &[ToolDef],
    enable_thinking: bool,
) -> Result<String, ServerError> {
    let mut system: Option<String> = None;
    let mut transcript = String::new();
    for (i, m) in messages.iter().enumerate() {
        // `flatten_content` enforces the ROBUST-007 numeric-type-guard (a bare
        // number/bool `content` 400s instead of being coerced) AND flattens
        // content-parts via the single shared key set — the SAME helper the
        // Anthropic surface uses, so the two cannot diverge.
        match m.role.as_str() {
            "system" if i == 0 => {
                system = Some(super::flatten_content(&m.content, "messages.content")?)
            }
            // A later system message stays where it was sent.
            "system" => {
                transcript.push_str("<|im_start|>system\n");
                transcript.push_str(&super::flatten_content(&m.content, "messages.content")?);
                transcript.push_str("<|im_end|>\n");
            }
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
                    transcript.push_str(
                        &lumen_runtime::tooling::render_assistant_tool_call_segment(
                            &tc.function.name,
                            &tc.function.arguments,
                        ),
                    );
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
    prompt.push_str(lumen_runtime::runtime_defaults::think_prompt_tail(
        enable_thinking,
    ));
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
        rx.recv().await.map(|chunk| {
            (
                Ok::<bytes::Bytes, std::io::Error>(bytes::Bytes::from(chunk)),
                rx,
            )
        })
    });
    Body::from_stream(stream)
}

pub fn stream_chat(
    rx: JobResponseChannel,
    model: String,
    created: u64,
    thinking: bool,
    stop: Vec<String>,
    tools: ReplyTools,
    include_usage: bool,
) -> Body {
    let (tx, body_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    tokio::spawn(drive_chat_stream(
        rx,
        tx,
        model,
        created,
        true,
        thinking,
        stop,
        tools,
        include_usage,
    ));
    body_from_byte_stream(body_rx)
}

pub fn stream_completion(
    rx: JobResponseChannel,
    model: String,
    created: u64,
    stop: Vec<String>,
    include_usage: bool,
) -> Body {
    let (tx, body_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    // Legacy completions have no chat template / `<think>` block: thinking is
    // always off, so the emitter's reasoning stage is a passthrough. No tools
    // are advertised on the legacy surface, so the parser is schemaless.
    tokio::spawn(drive_chat_stream(
        rx,
        tx,
        model,
        created,
        false,
        false,
        stop,
        ReplyTools::default(),
        include_usage,
    ));
    body_from_byte_stream(body_rx)
}

/// Idle-keepalive interval: emit a ping after this long with no wire output.
const PING_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

async fn drive_chat_stream(
    mut rx: JobResponseChannel,
    tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    model: String,
    created: u64,
    chat: bool,
    thinking: bool,
    stop: Vec<String>,
    tools: ReplyTools,
    include_usage: bool,
) {
    let id = format!(
        "chatcmpl-lumen-{created:x}-{:x}",
        super::next_response_seq()
    );
    let mut emitter = SseSafeEmitter::with_tools(thinking, tools);
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
    // True between a streamed tool call's first chunk (its name) and the end of
    // its arguments. If the stream ends while true the call was cut off mid-input:
    // its partial arguments already went out, so the flush must NOT also surface
    // the body as content (that would duplicate it).
    let mut tool_open = false;
    // Set when a tool call is abandoned without its `End` (cut off mid-input):
    // either a new call's first chunk arrives while one is still open, or the
    // stream ends with `tool_open`. Persists so a truncated call followed by a
    // well-formed one still reports finish_reason "length".
    let mut truncated_tool_call = false;
    // Usage totals from the worker's `Done` event, reported in the final usage
    // chunk when `stream_options.include_usage` was requested. On the rare
    // wire-side stop path (the redundant net fires before the worker's Done)
    // they remain 0 — the worker normally enforces stops first.
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;

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

    // Keepalive clock: fire after PING_INTERVAL of WIRE inactivity (no frame
    // emitted), not token inactivity, so a long buffered argument — tokens
    // arriving while nothing is emitted — is still kept alive. `last_emit` advances
    // only when a frame goes out.
    let mut last_emit = tokio::time::Instant::now();
    loop {
        let evt = tokio::select! {
            maybe = rx.recv() => match maybe {
                Some(evt) => evt,
                None => break,
            },
            _ = tokio::time::sleep_until(last_emit + PING_INTERVAL) => {
                // Keepalive during a long prefill/queue wait before the first token,
                // or a long buffered argument mid-generation: an SSE comment, which
                // every EventSource client ignores.
                if tx.send(b": ping\n\n".to_vec()).await.is_err() {
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
                let err = json!({"error": {
                    "message": "LUMEN_BENCH_TOKEN_IDS is not supported on streaming responses; use stream=false",
                    "type": "api_error" }});
                let _ = tx.send(sse_frame(&err.to_string())).await;
                let _ = tx.send(sse_done()).await;
                return;
            }
            TokenEvent::Token { delta_text, .. } => {
                // RAW passthrough for /v1/completions (`chat == false`): the
                // legacy surface returns the decoded model text VERBATIM — no
                // reasoning split and no tool-call parsing/stripping — so tool
                // markers (`<tool_call>` / `<function=`) reach the client and
                // the endpoint remains a faithful raw-emission oracle for the
                // pre-parser model output. Stop matching still applies.
                // `/v1/chat/completions` (`chat == true`) is unchanged.
                let delta = if chat {
                    emitter.push(&delta_text)
                } else {
                    crate::sse::EmitDelta {
                        reasoning: String::new(),
                        events: vec![StreamEvent::Text(delta_text)],
                        tool_calls: Vec::new(),
                    }
                };
                // OpenAI reassembles `content` and indexed `tool_calls` independently,
                // so the plain-text and tool-event views of the ordered delta are the
                // two wire channels; computed once each.
                let answer_text = delta.text();
                let tool_events = delta.tool_stream();
                let (safe_text, hit_stop) = stop_matcher.push(&answer_text);
                // Wire-inactivity keepalive clock: a token that emits at least one
                // frame resets it; a purely buffering token (empty reasoning, text,
                // and tool stream) does not, so a long buffered argument still
                // receives keepalives.
                let will_emit = if chat {
                    !delta.reasoning.is_empty() || !safe_text.is_empty() || !tool_events.is_empty()
                } else {
                    !safe_text.is_empty()
                };
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
                // Stream the tool call's input incrementally: one opening chunk
                // carrying the index/id/type/name and an empty `arguments`, then an
                // arguments-only delta per fragment as it is generated (the OpenAI
                // streaming tool-call contract). `tool_stream` is empty on the raw
                // `/v1/completions` path, so this never fires there.
                for ev in tool_events {
                    match ev {
                        ToolStreamEvent::Start { name } => {
                            if tool_open {
                                // The previous call never emitted its End — truncated.
                                truncated_tool_call = true;
                            }
                            tool_call_index += 1;
                            emitted_any_tool_call = true;
                            tool_open = true;
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
                                            "id": super::tool_call_id("call"),
                                            "type": "function",
                                            "function": { "name": name, "arguments": "" }
                                        }]
                                    },
                                    "finish_reason": null,
                                }],
                            });
                            if tx.send(sse_frame(&frame.to_string())).await.is_err() {
                                return;
                            }
                        }
                        ToolStreamEvent::ArgJsonDelta { partial_json } => {
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
                                            "function": { "arguments": partial_json }
                                        }]
                                    },
                                    "finish_reason": null,
                                }],
                            });
                            if tx.send(sse_frame(&frame.to_string())).await.is_err() {
                                return;
                            }
                        }
                        ToolStreamEvent::End => {
                            tool_open = false;
                        }
                    }
                }
                // Advance the keepalive clock AFTER the frames are sent (a blocked
                // backpressured send must not leave a stale early deadline).
                if will_emit {
                    last_emit = tokio::time::Instant::now();
                }
                if hit_stop {
                    // Wire-side stop (redundant safety net; the worker normally
                    // hits it first and sends Done{StopSequence}). Report
                    // StopSequence so OpenAI renders "stop".
                    finish_reason = Some(FinishReason::StopSequence);
                    break;
                }
            }
            TokenEvent::Done {
                finish_reason: fr,
                prompt_tokens: p,
                completion_tokens: c,
            } => {
                finish_reason = Some(fr);
                prompt_tokens = p;
                completion_tokens = c;
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

    let (residual, incomplete) = emitter.finish();
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
    let mut final_content = if stopped_by_sequence {
        String::new()
    } else {
        let (residual_safe, _residual_hit) = if stop_matcher.is_active() {
            stop_matcher.push(&residual.text())
        } else {
            (residual.text(), false)
        };
        let mut c = residual_safe;
        c.push_str(&stop_matcher.finish());
        c
    };
    // A tool call cut off mid-body: a NATIVE one already streamed its partial
    // arguments (tool_open), so its body must NOT be re-surfaced as content; a
    // legacy / pre-`<function=>` one streamed nothing, so its body is surfaced as
    // content so it is never lost. Either way the finish reason below is forced to
    // Length ("length") so the client sees a truncation and continues.
    // `tool_open` is taken to be THIS incomplete call's own block, which holds for
    // well-formed output and any single truncated call; the trained format never
    // emits the one shape that breaks it (a native call that reaches `</tool_call>`
    // with a parameter left open, keeping the block open across a following call).
    if let Some(body) = &incomplete {
        if !tool_open {
            final_content.push_str(body);
        }
    }
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
    // A turn truncated mid tool call -> "length" (via Length): either the parser
    // reported an incomplete body, OR a tool call is still open (`tool_open` — its
    // outer `</tool_call>` arrived but a parameter never closed, so no End was
    // emitted). Never report that partial, invalid-JSON call as a clean tool_calls.
    let reason = if incomplete.is_some() || tool_open || truncated_tool_call {
        FinishReason::Length
    } else {
        match finish_reason {
            Some(FinishReason::Stop) if emitted_any_tool_call => FinishReason::ToolCalls,
            Some(r) => r,
            None => FinishReason::Stop,
        }
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
    // OpenAI `stream_options.include_usage` contract: when the client
    // requested it, ONE extra chunk with empty `choices` and the usage totals
    // goes out AFTER the finish chunk and BEFORE `data: [DONE]`. When not
    // requested, nothing is emitted here — the stream stays byte-identical to
    // the historical shape.
    if include_usage {
        let usage = json!({
            "id": id,
            "object": if chat { "chat.completion.chunk" } else { "text_completion" },
            "created": created,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        });
        if tx.send(sse_frame(&usage.to_string())).await.is_err() {
            return;
        }
    }
    let _ = tx.send(sse_done()).await;
}

// ----------------------------- Non-streaming chat ------------------------

pub async fn collect_chat(
    mut rx: JobResponseChannel,
    model: String,
    created: u64,
    thinking: bool,
    stop: Vec<String>,
    tools: ReplyTools,
) -> Result<Value, ServerError> {
    let mut emitter = SseSafeEmitter::with_tools(thinking, tools);
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
    // Bench surface (LUMEN_BENCH_TOKEN_IDS): (generated ids, eos set).
    let mut bench_ids: Option<BenchRecord> = None;

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
                content.push_str(&safe_text);
                for tc in delta.tool_calls {
                    tool_calls.push(json!({
                        "id": super::tool_call_id("call"),
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
    // Once a stop sequence fired, drop the residual answer text (post-stop
    // content). Otherwise pass the emitter residual through the stop matcher
    // (catches a stop straddling the emitter's held tail) and drain it. Empty
    // stop => appends `residual.text` verbatim + nothing, byte-identical.
    if finish != FinishReason::StopSequence {
        let (residual_safe, _) = if stop_matcher.is_active() {
            stop_matcher.push(&residual.text())
        } else {
            (residual.text(), false)
        };
        content.push_str(&residual_safe);
        content.push_str(&stop_matcher.finish());
    }
    // surface a tool call cut off inside its body as content (never drop it).
    if let Some(body) = &incomplete {
        content.push_str(body);
    }

    if !tool_calls.is_empty() && finish == FinishReason::Stop {
        finish = FinishReason::ToolCalls;
    }
    // an incomplete tool call means the turn was truncated -> report "length"
    // (via Length) so the client continues instead of trusting a clean stop.
    if incomplete.is_some() {
        finish = FinishReason::Length;
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
    let mut body = json!({
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
    });
    attach_bench_token_ids(&mut body, bench_ids, finish)?;
    Ok(body)
}

/// The bench record the engine emits: generated ids, the EOS set, and under
/// `LUMEN_BENCH_TOP2` the per-token top-2 entries.
pub(crate) type BenchRecord = (Vec<u32>, Vec<u32>, Vec<lumen_runtime::session::BenchTop2>);

/// Attach the bench token-id surface (`LUMEN_BENCH_TOKEN_IDS=1`) to a finished
/// response body as a top-level `lumen_bench` object. `finish_reason` uses
/// the OpenAI vocabulary on every route, so the object reads the same on
/// `/v1/messages` as on `/v1/chat/completions`.
///
/// Fail-closed: when the surface is armed the ids MUST be present. If the engine
/// emitted none on this path, the request fails rather than returning a body
/// that silently lacks the surface. When the flag is off this is a no-op and
/// the body is byte-identical.
pub(crate) fn attach_bench_token_ids(
    body: &mut serde_json::Value,
    bench_ids: Option<BenchRecord>,
    finish: FinishReason,
) -> Result<(), ServerError> {
    attach_bench_token_ids_if(
        lumen_runtime::runtime_defaults::bench_token_ids_enabled()
            || lumen_runtime::runtime_defaults::bench_top2_enabled(),
        body,
        bench_ids,
        finish,
    )
}

/// [`attach_bench_token_ids`] with the flag passed in, so the armed path is
/// testable without touching the process environment.
fn attach_bench_token_ids_if(
    enabled: bool,
    body: &mut serde_json::Value,
    bench_ids: Option<BenchRecord>,
    finish: FinishReason,
) -> Result<(), ServerError> {
    if !enabled {
        return Ok(());
    }
    let Some((generated, eos, top2)) = bench_ids else {
        return Err(ServerError::Internal(
            "LUMEN_BENCH_TOKEN_IDS=1 but the engine emitted no token-id record on \
             this path. Refusing to return a response without the requested surface."
                .to_string(),
        ));
    };
    let obj = body
        .as_object_mut()
        .expect("response body is a JSON object");
    let mut bench = json!({
        "generated_token_ids": generated,
        "generated_token_count": generated.len(),
        "finish_reason": finish.as_openai(),
        "eos_token_ids": eos,
    });
    if !top2.is_empty() {
        // [argmax, logit, runner_up, runner_up_logit] per generated token, in order; the
        // argmax is of the raw logits, the selected token is generated_token_ids[i].
        bench["top2"] = json!(top2
            .iter()
            .map(|t| json!([t.argmax, t.logit, t.runner_up, t.runner_up_logit]))
            .collect::<Vec<_>>());
    }
    obj.insert("lumen_bench".to_string(), bench);
    Ok(())
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
    // Test helper exercises the legacy JSON tool-call path, which parses without
    // schemas; the schema-aware native path is covered by the runtime tooling
    // tests and the Modal §2D gate.
    collect_chat(
        pooled,
        model,
        created,
        thinking,
        stop,
        ReplyTools::default(),
    )
    .await
}

pub async fn collect_completion(
    mut rx: JobResponseChannel,
    model: String,
    created: u64,
    stop: Vec<String>,
) -> Result<Value, ServerError> {
    // RAW passthrough: the legacy completions surface returns the decoded
    // model text VERBATIM — no SseSafeEmitter reasoning/tool stage (which
    // would strip `<tool_call>` / `<function=` blocks) — so the endpoint is a
    // faithful raw-emission oracle for the pre-parser model output.
    // F4: seed from the request stop list. Empty => verbatim, byte-identical.
    let mut stop_matcher = StopMatcher::new(stop);
    let mut text = String::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut finish = FinishReason::Stop;
    // Bench surface (LUMEN_BENCH_TOKEN_IDS): (generated ids, eos set).
    let mut bench_ids: Option<BenchRecord> = None;

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
                let (safe_text, hit_stop) = stop_matcher.push(&delta_text);
                text.push_str(&safe_text);
                if hit_stop {
                    // Wire-side stop match -> StopSequence (renders OpenAI "stop").
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
    if finish != FinishReason::StopSequence {
        // Only the stop matcher can be holding bytes (a partial stop-sequence
        // prefix); with no stop list this is empty — byte-identical verbatim.
        text.push_str(&stop_matcher.finish());
    }

    let mut body = json!({
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
    });
    attach_bench_token_ids(&mut body, bench_ids, finish)?;
    Ok(body)
}

#[cfg(test)]
mod bench_token_ids_surface_tests {
    use super::*;

    /// The decode loop `break`s on EOS without emitting a `Token` event, so a
    /// collector fed by `Token` events would lose the terminating id. The
    /// record is therefore taken inside the engine loop, before the EOS check.
    #[test]
    fn ids_are_recorded_before_the_eos_break() {
        const ENG: &str = include_str!("../engine.rs");
        let push = ENG
            .find("bench_token_ids.push(token_id)")
            .expect("engine must record sampled ids in the decode loop");
        let eos_check = ENG
            .find("request.eos_token_ids.contains(&token_id)")
            .expect("engine must have the EOS check");
        assert!(push < eos_check, "the id record must precede the EOS check");
    }

    /// The top-2 entries surface as `[argmax, logit, runner_up, runner_up_logit]`
    /// arrays, and an empty record adds no key.
    #[test]
    fn top2_entries_surface_as_arrays_when_present() {
        let mut body = json!({"id": "x"});
        let t = lumen_runtime::session::BenchTop2 {
            argmax: 7,
            logit: 1.5,
            runner_up: 9,
            runner_up_logit: 1.25,
        };
        attach_bench_token_ids_if(
            true,
            &mut body,
            Some((vec![7], vec![2], vec![t])),
            FinishReason::Stop,
        )
        .unwrap();
        assert_eq!(body["lumen_bench"]["top2"], json!([[7, 1.5, 9, 1.25]]));
        let mut plain = json!({"id": "x"});
        attach_bench_token_ids_if(
            true,
            &mut plain,
            Some((vec![7], vec![2], vec![])),
            FinishReason::Stop,
        )
        .unwrap();
        assert!(
            plain["lumen_bench"].get("top2").is_none(),
            "no entries, no key"
        );
    }

    /// Off leaves the body byte-identical; armed without ids fails rather
    /// than returning a body without the surface.
    #[test]
    fn off_is_a_no_op_and_armed_without_ids_fails() {
        let mut body = json!({"a": 1});
        let before = body.clone();
        attach_bench_token_ids_if(false, &mut body, None, FinishReason::Stop)
            .expect("off never fails");
        assert_eq!(body, before);
        attach_bench_token_ids_if(
            false,
            &mut body,
            Some((vec![1], vec![2], vec![])),
            FinishReason::Stop,
        )
        .expect("off ignores ids");
        assert_eq!(body, before, "off never writes, even with ids present");
        let err = attach_bench_token_ids_if(true, &mut body, None, FinishReason::Stop)
            .expect_err("armed without ids must fail");
        assert!(
            matches!(err, ServerError::Internal(ref m) if m.starts_with("LUMEN_BENCH_TOKEN_IDS=1"))
        );
        assert_eq!(body, before, "a refused attach leaves the body untouched");
    }

    /// The emitted shape: exactly the four keys, the array with the
    /// terminator last, its length, the OpenAI finish vocabulary, the EOS set.
    #[test]
    fn armed_attach_emits_the_documented_schema() {
        let mut body = json!({"object": "chat.completion"});
        attach_bench_token_ids_if(
            true,
            &mut body,
            Some((vec![9, 42, 248046], vec![248046, 248044], vec![])),
            FinishReason::Stop,
        )
        .unwrap();
        assert_eq!(body["object"], "chat.completion", "existing keys untouched");
        let b = body["lumen_bench"]
            .as_object()
            .expect("top-level lumen_bench object");
        let mut keys: Vec<&str> = b.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "eos_token_ids",
                "finish_reason",
                "generated_token_count",
                "generated_token_ids"
            ]
        );
        assert_eq!(b["generated_token_ids"], json!([9, 42, 248046]));
        assert_eq!(b["generated_token_count"], 3);
        assert_eq!(b["finish_reason"], "stop");
        assert_eq!(b["eos_token_ids"], json!([248046, 248044]));

        let mut body = json!({});
        attach_bench_token_ids_if(
            true,
            &mut body,
            Some((vec![], vec![], vec![])),
            FinishReason::Length,
        )
        .unwrap();
        assert_eq!(body["lumen_bench"]["finish_reason"], "length");
        assert_eq!(body["lumen_bench"]["generated_token_count"], 0);
    }

    /// The guard refuses what the surface cannot serve, and only when armed.
    #[test]
    fn guard_refuses_streaming_and_stop_sequences_only_when_armed() {
        if lumen_runtime::runtime_defaults::bench_token_ids_enabled() {
            let e = super::super::bench_token_ids_guard(true, &[]).expect_err("streaming refused");
            assert!(
                matches!(e, ServerError::BadRequest { ref param, .. } if param.as_deref() == Some("stream"))
            );
            let e = super::super::bench_token_ids_guard(false, &["x".into()])
                .expect_err("stop refused");
            assert!(
                matches!(e, ServerError::BadRequest { ref param, .. } if param.as_deref() == Some("stop"))
            );
            super::super::bench_token_ids_guard(false, &[]).expect("plain non-streaming allowed");
        } else {
            super::super::bench_token_ids_guard(true, &["x".into()]).expect("off never refuses");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_runtime::tooling::Qwen35Renderer;

    fn tok(text: &str) -> TokenEvent {
        TokenEvent::Token {
            token_id: 0,
            delta_text: text.to_string(),
        }
    }

    /// Drive `drive_chat_stream` over a fixed event list and return the raw
    /// SSE byte stream as a UTF-8 string (mirrors the Anthropic helper).
    async fn stream_openai_to_string(
        events: Vec<TokenEvent>,
        chat: bool,
        include_usage: bool,
    ) -> String {
        stream_openai_to_string_tools(events, chat, include_usage, ReplyTools::default()).await
    }

    /// A `ReplyTools` with one `string`-typed parameter so the native streamer
    /// takes the incremental `StreamString` path (a schemaless value buffers).
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

    async fn stream_openai_to_string_tools(
        events: Vec<TokenEvent>,
        chat: bool,
        include_usage: bool,
        tools: ReplyTools,
    ) -> String {
        let (tx, rx) = tokio::sync::mpsc::channel(events.len().max(1));
        let return_sender = tx.clone();
        for e in events {
            tx.send(e).await.unwrap();
        }
        drop(tx);
        let pooled = crate::engine::PooledReceiver::new(rx, return_sender, None, 0, None);
        let (body_tx, mut body_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);
        tokio::spawn(drive_chat_stream(
            pooled,
            body_tx,
            "test-model".into(),
            1234,
            chat,
            false,
            Vec::new(),
            tools,
            include_usage,
        ));
        let mut out = String::new();
        while let Some(chunk) = body_rx.recv().await {
            out.push_str(&String::from_utf8_lossy(&chunk));
        }
        out
    }

    /// On the OpenAI wire a NATIVE tool call's arguments stream as many
    /// `function.arguments` deltas — one opening chunk carrying the name and an
    /// empty `arguments`, then arguments-only deltas — and the concatenation is
    /// valid JSON with the generated value, not one buffered frame at close.
    #[tokio::test]
    async fn stream_chat_native_tool_arguments_stream_incrementally() {
        let events = vec![
            tok("<tool_call>\n<function=write_file>\n<parameter=content>\n"),
            tok("line one\n"),
            tok("line two\n"),
            tok("</parameter>\n</function>\n</tool_call>"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 3,
                completion_tokens: 9,
            },
        ];
        // String schema so `content` streams via StreamString (a schemaless value
        // buffers to one frame, defeating the point of this test).
        let sse = stream_openai_to_string_tools(
            events,
            true,
            false,
            string_param_tools("write_file", "content"),
        )
        .await;
        let mut name = String::new();
        let mut args = String::new();
        let mut arg_strings: Vec<String> = Vec::new();
        for frame in sse.split("\n\n") {
            let data = match frame.lines().find_map(|l| l.strip_prefix("data: ")) {
                Some(d) if d != "[DONE]" => d,
                _ => continue,
            };
            let v: serde_json::Value = serde_json::from_str(data).unwrap();
            let tc = match v["choices"][0]["delta"]["tool_calls"].get(0) {
                Some(tc) => tc.clone(),
                None => continue,
            };
            if let Some(n) = tc["function"]["name"].as_str() {
                name.push_str(n);
            }
            if let Some(a) = tc["function"]["arguments"].as_str() {
                if !a.is_empty() {
                    arg_strings.push(a.to_string());
                }
                args.push_str(a);
            }
        }
        assert_eq!(
            name, "write_file",
            "name is sent once on the opening chunk: {sse}"
        );
        assert!(
            arg_strings.len() > 1,
            "arguments must stream incrementally, got {} frame(s): {sse}",
            arg_strings.len()
        );
        // The VALUE itself must split across frames (StreamString), not arrive whole in
        // one frame: a buffered value would yield a `{"content":"`, the whole value,
        // and a `}` — three frames, but with both halves together in one. Asserting no
        // single frame carries both halves is what distinguishes true char streaming.
        assert!(
            !arg_strings
                .iter()
                .any(|a| a.contains("line one") && a.contains("line two")),
            "the value must stream in pieces, not arrive whole in one frame: {arg_strings:?}"
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&args).expect("concatenated arguments must be valid JSON");
        assert_eq!(parsed["content"], "line one\nline two");
        assert!(
            sse.contains("\"finish_reason\":\"tool_calls\""),
            "a tool-call turn reports finish_reason tool_calls: {sse}"
        );
    }

    /// A native tool call cut off mid-argument: the partial `function.arguments`
    /// already streamed must NOT be re-surfaced as message content, and the turn
    /// reports finish_reason "length" (the OpenAI analogue of max_tokens), never a
    /// clean "tool_calls" over invalid JSON.
    #[tokio::test]
    async fn stream_chat_native_incomplete_tool_call_is_length() {
        let events = vec![
            tok("<tool_call>\n<function=write_file>\n<parameter=content>\nfn main() {"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 20,
            },
        ];
        let sse = stream_openai_to_string_tools(
            events,
            true,
            false,
            string_param_tools("write_file", "content"),
        )
        .await;
        assert!(
            sse.contains("\"finish_reason\":\"length\""),
            "a cut-off tool call reports length: {sse}"
        );
        assert!(
            sse.contains("\"arguments\""),
            "the partial streamed as tool arguments: {sse}"
        );
        assert!(
            !sse.contains("\"content\":\""),
            "nothing is surfaced as message content: {sse}"
        );
    }

    /// A truncated tool call followed by a complete one must report finish_reason
    /// "length": the complete call clears `tool_open`, so `truncated_tool_call` is
    /// what carries the truncation signal to the terminal frame.
    #[tokio::test]
    async fn stream_chat_truncated_call_then_complete_call_is_length() {
        let events = vec![
            tok("<tool_call>\n<function=f>\n<parameter=x>\n1</tool_call>"),
            tok("<tool_call>\n<function=g>\n</function>\n</tool_call>"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 8,
            },
        ];
        let sse =
            stream_openai_to_string_tools(events, true, false, string_param_tools("f", "x")).await;
        assert!(
            sse.contains("\"finish_reason\":\"length\""),
            "truncation persists past the complete call: {sse}"
        );
    }

    /// The idle keepalive: with no token for the ping interval (a long prefill or
    /// queue wait before the first token) the driver emits an SSE comment, which
    /// every EventSource client ignores, so a slow-to-start turn is not aborted.
    /// Virtual time auto-advances to the pending timer, so the test does not wait.
    #[tokio::test(start_paused = true)]
    async fn stream_chat_emits_ping_during_a_long_idle_gap() {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let return_sender = tx.clone();
        let pooled = crate::engine::PooledReceiver::new(rx, return_sender, None, 0, None);
        let (body_tx, mut body_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);
        let driver = tokio::spawn(drive_chat_stream(
            pooled,
            body_tx,
            "m".into(),
            1234,
            true,
            false,
            Vec::new(),
            ReplyTools::default(),
            false,
        ));
        // The role head chunk goes out immediately; with no token queued the driver
        // parks on the select and virtual time auto-advances to fire the keepalive.
        let head = body_rx.recv().await.unwrap();
        assert!(String::from_utf8_lossy(&head).contains("role"));
        let ping = body_rx.recv().await.unwrap();
        assert!(
            String::from_utf8_lossy(&ping).contains(": ping"),
            "a long idle gap must emit an SSE-comment keepalive"
        );
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
            rest.contains("[DONE]"),
            "turn completes after the ping: {rest}"
        );
        driver.await.unwrap();
    }

    /// `stream_options.include_usage: true` -> exactly ONE extra chunk with
    /// empty `choices` + the usage totals, positioned as the LAST data frame
    /// before `data: [DONE]` (the OpenAI streaming contract).
    #[tokio::test]
    async fn stream_usage_chunk_present_when_requested() {
        for (chat, object) in [(true, "chat.completion.chunk"), (false, "text_completion")] {
            let events = vec![
                tok("hi"),
                TokenEvent::Done {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: 7,
                    completion_tokens: 3,
                },
            ];
            assert_usage_chunk(&stream_openai_to_string(events, chat, true).await, object);
        }
    }

    fn assert_usage_chunk(sse: &str, object: &str) {
        let frames: Vec<&str> = sse
            .split("\n\n")
            .filter_map(|b| b.trim().strip_prefix("data: "))
            .collect();
        assert_eq!(*frames.last().unwrap(), "[DONE]");
        let usage_frame = frames[frames.len() - 2];
        let v: Value = serde_json::from_str(usage_frame).unwrap();
        assert_eq!(
            v["choices"].as_array().unwrap().len(),
            0,
            "usage chunk must carry empty choices: {usage_frame}"
        );
        assert_eq!(v["object"], object);
        assert_eq!(v["usage"]["prompt_tokens"], 7);
        assert_eq!(v["usage"]["completion_tokens"], 3);
        assert_eq!(v["usage"]["total_tokens"], 10);
        let n = frames.iter().filter(|f| f.contains("\"usage\"")).count();
        assert_eq!(n, 1, "exactly one usage chunk: {sse}");
    }

    /// /v1/completions is a RAW surface: tool markers pass through VERBATIM
    /// (non-streaming). Before the fix the SseSafeEmitter stripped the
    /// `<tool_call>` block out of `choices[0].text`.
    #[tokio::test]
    async fn collect_completion_passes_tool_markers_verbatim() {
        let raw = "prefix <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call> suffix";
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let return_sender = tx.clone();
        // Split mid-marker to prove nothing is held back or stripped across
        // chunk boundaries on the raw path.
        for part in [
            "prefix <tool_",
            "call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_",
            "call> suffix",
        ] {
            tx.send(tok(part)).await.unwrap();
        }
        tx.send(TokenEvent::Done {
            finish_reason: FinishReason::Stop,
            prompt_tokens: 4,
            completion_tokens: 9,
        })
        .await
        .unwrap();
        drop(tx);
        let pooled = crate::engine::PooledReceiver::new(rx, return_sender, None, 0, None);
        let resp = collect_completion(pooled, "test-model".into(), 1234, Vec::new())
            .await
            .unwrap();
        assert_eq!(resp["choices"][0]["text"].as_str().unwrap(), raw);
        assert_eq!(resp["usage"]["total_tokens"], 13);
    }

    /// Streaming /v1/completions: the concatenated `choices[0].text` deltas
    /// equal the verbatim model text, tool markers included (no stripping, no
    /// held-back marker prefixes).
    #[tokio::test]
    async fn stream_completion_passes_tool_markers_verbatim() {
        let raw = "prefix <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call> suffix";
        let events = vec![
            tok("prefix <tool_"),
            tok("call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_"),
            tok("call> suffix"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 4,
                completion_tokens: 9,
            },
        ];
        let sse = stream_openai_to_string(events, false, false).await;
        let mut text = String::new();
        for frame in sse
            .split("\n\n")
            .filter_map(|b| b.trim().strip_prefix("data: "))
        {
            if frame == "[DONE]" {
                continue;
            }
            let v: Value = serde_json::from_str(frame).unwrap();
            if let Some(t) = v["choices"][0]["text"].as_str() {
                text.push_str(t);
            }
        }
        assert_eq!(text, raw);
        assert!(
            !sse.contains("\"usage\""),
            "legacy surface has no usage chunk"
        );
    }

    /// Absent `stream_options.include_usage`, NO usage chunk is emitted — the
    /// stream stays byte-identical to the historical shape (matching OpenAI).
    #[tokio::test]
    async fn stream_chat_no_usage_chunk_by_default() {
        let events = vec![
            tok("hi"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 7,
                completion_tokens: 3,
            },
        ];
        let sse = stream_openai_to_string(events, true, false).await;
        assert!(
            !sse.contains("\"usage\""),
            "no usage chunk absent the flag: {sse}"
        );
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
        let resp = collect_chat_from_events(events, "test-model".into(), 1234, false)
            .await
            .unwrap();
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

    /// Tool-call ids are unique across responses as well as within one: a
    /// client keeps every earlier turn's ids in its history.
    #[tokio::test]
    async fn tool_call_ids_are_unique_across_responses() {
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
            let resp = collect_chat_from_events(events(), "m".into(), 1, false)
                .await
                .unwrap();
            for tc in resp["choices"][0]["message"]["tool_calls"]
                .as_array()
                .unwrap()
            {
                ids.push(tc["id"].as_str().unwrap().to_string());
            }
            let sse = stream_openai_to_string(events(), true, false).await;
            for part in sse.split("\"id\":\"call_").skip(1) {
                ids.push(format!("call_{}", part.split('"').next().unwrap()));
            }
        }
        assert_eq!(ids.len(), 8, "{ids:?}");
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "{ids:?}");
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
        let resp = collect_chat_from_events(events, "test".into(), 1, false)
            .await
            .unwrap();
        let content = resp["choices"][0]["message"]["content"].as_str().unwrap();
        // The literal "<tool" must not appear in user-visible content.
        assert!(!content.contains("<tool"));
        assert_eq!(
            resp["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "f"
        );
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
        let resp = collect_chat_from_events(events, "test".into(), 1, false)
            .await
            .unwrap();
        assert_eq!(resp["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn collect_chat_incomplete_tool_call_surfaces_partial_as_length() {
        // a tool call cut off inside its body (EOS before </tool_call>) must surface
        // the partial body as content and report "length", not drop it as a clean "stop".
        let events = vec![
            tok("<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"content\": \"fn main() {"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 20,
            },
        ];
        let resp = collect_chat_from_events(events, "test".into(), 1, false)
            .await
            .unwrap();
        assert_eq!(resp["choices"][0]["finish_reason"], "length");
        let content = resp["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(
            content.contains("fn main()"),
            "partial surfaced as content: {content:?}"
        );
        assert!(
            resp["choices"][0]["message"]["tool_calls"].is_null(),
            "an incomplete call is not a complete tool_call"
        );
    }

    #[tokio::test]
    async fn stream_chat_incomplete_tool_call_surfaces_partial_as_length() {
        // The streaming analogue: the partial body is emitted as a content delta and the
        // terminal chunk reports finish_reason "length".
        let events = vec![
            tok("<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"content\": \"fn main() {"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 20,
            },
        ];
        let sse = stream_openai_to_string(events, true, false).await;
        assert!(
            sse.contains("fn main()"),
            "partial surfaced in stream: {sse}"
        );
        assert!(
            sse.contains("\"finish_reason\":\"length\""),
            "length reason in stream: {sse}"
        );
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
            reasoning_content: None,
            role: "user".into(),
            content: Value::String(text.into()),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    fn system_msg(text: &str) -> ChatMessage {
        ChatMessage {
            reasoning_content: None,
            role: "system".into(),
            content: Value::String(text.into()),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    fn assistant_msg(text: &str) -> ChatMessage {
        ChatMessage {
            reasoning_content: None,
            role: "assistant".into(),
            content: Value::String(text.into()),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    fn assistant_tool_call_msg(name: &str, arguments: &str) -> ChatMessage {
        ChatMessage {
            reasoning_content: None,
            role: "assistant".into(),
            content: Value::String(String::new()),
            tool_call_id: None,
            tool_calls: vec![AssistantToolCall {
                id: "call_1".into(),
                call_type: "function".into(),
                function: AssistantToolCallFn {
                    name: name.into(),
                    arguments: arguments.into(),
                },
            }],
        }
    }

    fn tool_msg(content: &str) -> ChatMessage {
        ChatMessage {
            reasoning_content: None,
            role: "tool".into(),
            content: Value::String(content.into()),
            tool_call_id: Some("call_1".into()),
            tool_calls: Vec::new(),
        }
    }

    fn weather_tool_def() -> ToolDef {
        ToolDef {
            def_type: "function".into(),
            function: ToolDefFunction {
                name: "get_weather".into(),
                description: "Get weather.".into(),
                parameters: json!({"type":"object","properties":{"city":{"type":"string"}}}),
            },
        }
    }

    #[test]
    fn templated_render_parses_tool_call_arguments_into_object_for_items() {
        // The template iterates `tool_calls[].function.arguments | items`, so the
        // server MUST parse the OpenAI on-wire arguments JSON STRING into an
        // object; if it passed the raw string this render would error on `|items`.
        let tmpl = "{%- for m in messages %}{%- if m.tool_calls %}{%- for tc in m.tool_calls %}{%- set f = tc.function %}fn={{ f.name }}{%- for k, v in f.arguments | items %} {{ k }}={{ v }}{%- endfor %}{%- endfor %}{%- endif %}{%- endfor %}";
        let messages = vec![assistant_tool_call_msg(
            "get_weather",
            r#"{"city": "Riyadh", "unit": "celsius"}"#,
        )];
        let out = render_chat_prompt(&messages, &[], false, Some(tmpl), None).unwrap();
        assert_eq!(out, "fn=get_weather city=Riyadh unit=celsius");
    }

    #[test]
    fn templated_render_groups_consecutive_tool_results_via_adjacent_loop_items() {
        // Two consecutive `tool` messages must collapse into ONE user turn
        // (opened before the first, closed after the last) — the template does
        // this with loop.previtem / loop.nextitem, which requires the
        // adjacent_loop_items feature the shared renderer enables.
        let tmpl = "{%- for m in messages %}{%- if m.role == 'tool' %}{%- if loop.previtem and loop.previtem.role != 'tool' %}<user>{%- endif %}[{{ m.content }}]{%- if loop.last or loop.nextitem.role != 'tool' %}</user>{%- endif %}{%- else %}<{{ m.role }}>{{ m.content }}{%- endif %}{%- endfor %}";
        let messages = vec![
            user_msg("hi"),
            tool_msg("A"),
            tool_msg("B"),
            assistant_msg("done"),
        ];
        let out = render_chat_prompt(&messages, &[], false, Some(tmpl), None).unwrap();
        assert_eq!(out, "<user>hi<user>[A][B]</user><assistant>done");
    }

    #[test]
    fn templated_render_dispatches_and_builds_tool_context() {
        // A tool is advertised: the templated path must expose `tools` (non-empty)
        // and the flattened user content to the template.
        let tmpl = "{%- if tools %}TOOLS={{ tools[0].function.name }}{%- endif %} U={{ messages[-1].content }}";
        let messages = vec![user_msg("weather?")];
        let out =
            render_chat_prompt(&messages, &[weather_tool_def()], false, Some(tmpl), None).unwrap();
        assert_eq!(out, "TOOLS=get_weather U=weather?");
    }

    #[test]
    fn render_chat_prompt_user_only_emits_closed_think_when_disabled() {
        // enable_thinking=false (the default) MUST emit the closed empty-think
        // tail, byte-identical to the pre-reasoning-control behaviour.
        let messages = vec![user_msg("Hello")];
        let out = render_chat_prompt(&messages, &[], false, None, None).unwrap();
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
        let out = render_chat_prompt(&messages, &[], true, None, None).unwrap();
        let expected = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n";
        assert_eq!(
            out, expected,
            "render_chat_prompt user-only enabled != open think tail"
        );
    }

    #[test]
    fn render_chat_prompt_system_plus_user_emits_closed_think_when_disabled() {
        let messages = vec![system_msg("You are helpful."), user_msg("Hi")];
        let out = render_chat_prompt(&messages, &[], false, None, None).unwrap();
        let expected = "<|im_start|>system\nYou are helpful.<|im_end|>\n\
                        <|im_start|>user\nHi<|im_end|>\n\
                        <|im_start|>assistant\n<think>\n\n</think>\n\n";
        assert_eq!(
            out, expected,
            "render_chat_prompt system+user != CLI output"
        );
    }

    #[test]
    fn render_chat_prompt_system_plus_user_emits_open_think_when_enabled() {
        let messages = vec![system_msg("You are helpful."), user_msg("Hi")];
        let out = render_chat_prompt(&messages, &[], true, None, None).unwrap();
        let expected = "<|im_start|>system\nYou are helpful.<|im_end|>\n\
                        <|im_start|>user\nHi<|im_end|>\n\
                        <|im_start|>assistant\n<think>\n";
        assert_eq!(
            out, expected,
            "render_chat_prompt system+user enabled != open think tail"
        );
    }

    #[test]
    fn earlier_reasoning_renders_back_on_both_endpoints() {
        let template =
            include_str!("../../../lumen-runtime/tests/fixtures/qwen38_chat_template.jinja");
        let chat = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
                {"role": "user", "content": "Q2"},
            ],
        }))
        .unwrap()
        .prompt(Some(template))
        .unwrap()
        .0;
        let messages = serde_json::from_value::<crate::wire::anthropic::MessagesRequest>(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "R1", "signature": "s"},
                    {"type": "text", "text": "A1"},
                ]},
                {"role": "user", "content": "Q2"},
            ],
        }))
        .unwrap()
        .prompt(Some(template))
        .unwrap()
        .0;
        assert_eq!(chat, messages);
        assert!(
            chat.contains("<|im_start|>assistant\n<think>\nR1\n</think>\n\nA1<|im_end|>"),
            "{chat}"
        );
        // Explicitly empty reasoning still reaches the template, so Qwen3.5
        // does not go looking for reasoning inside the answer.
        let qwen35 =
            include_str!("../../../lumen-runtime/tests/fixtures/qwen35_chat_template.jinja");
        let answer = "X</think>M</think>Y";
        let chat = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": answer, "reasoning_content": ""},
                {"role": "user", "content": "Q2"},
            ],
        }))
        .unwrap()
        .prompt(Some(qwen35))
        .unwrap()
        .0;
        let messages = serde_json::from_value::<crate::wire::anthropic::MessagesRequest>(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "", "signature": "s"},
                    {"type": "text", "text": answer},
                ]},
                {"role": "user", "content": "Q2"},
            ],
        }))
        .unwrap()
        .prompt(Some(qwen35))
        .unwrap()
        .0;
        assert_eq!(chat, messages);
        assert!(
            chat.contains(&format!("<|im_start|>assistant\n{answer}<|im_end|>")),
            "{chat}"
        );
    }

    #[test]
    fn tool_choice_matches_the_messages_endpoint() {
        let template =
            include_str!("../../../lumen-runtime/tests/fixtures/qwen38_chat_template.jinja");
        let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let with = |mut body: Value, extra: Value| {
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            body
        };
        let chat = |choice: Value, extra: Value| {
            let body = json!({
                "model": "m", "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "description": "Weather", "parameters": schema}}],
                "tool_choice": choice,
            });
            serde_json::from_value::<ChatCompletionRequest>(with(body, extra))
                .unwrap()
                .prompt(Some(template))
        };
        let messages = |choice: Value, extra: Value| {
            let body = json!({
                "model": "m", "max_tokens": 16,
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [{"name": "get_weather", "description": "Weather", "input_schema": schema}],
                "tool_choice": choice,
            });
            serde_json::from_value::<crate::wire::anthropic::MessagesRequest>(with(body, extra))
                .unwrap()
                .prompt(Some(template))
        };
        let named = json!({"type": "function", "function": {"name": "get_weather"}});
        for (openai, anthropic, prefix) in [
            (json!("auto"), json!({"type": "auto"}), ""),
            (json!("none"), json!({"type": "none"}), ""),
            (json!("required"), json!({"type": "any"}), "<tool_call>\n"),
            (
                named.clone(),
                json!({"type": "tool", "name": "get_weather"}),
                "<tool_call>\n<function=get_weather>\n",
            ),
        ] {
            let (prompt, response_prefix) = chat(openai.clone(), json!({})).unwrap();
            assert_eq!(
                (prompt.clone(), response_prefix.clone()),
                messages(anthropic, json!({})).unwrap(),
                "{openai}"
            );
            assert_eq!(response_prefix, prefix);
            assert!(prompt.ends_with(&format!("<think>\n\n</think>\n\n{prefix}")));
        }
        // `none` offers no tools; a forced choice keeps every tool offered.
        let (none, _) = chat(json!("none"), json!({})).unwrap();
        assert!(!none.contains("<tools>"));
        assert!(chat(named.clone(), json!({}))
            .unwrap()
            .0
            .contains("<tools>"));
        // Refused on both: a tool not offered, and a forced call while thinking.
        let unknown = json!({"type": "function", "function": {"name": "nope"}});
        let refusals = [
            chat(unknown, json!({})).map(|_| ()),
            messages(json!({"type": "tool", "name": "nope"}), json!({})).map(|_| ()),
            chat(json!("required"), json!({"enable_thinking": true})).map(|_| ()),
            messages(
                json!({"type": "any"}),
                json!({"thinking": {"type": "enabled"}}),
            )
            .map(|_| ()),
            chat(json!("required"), json!({"tools": []})).map(|_| ()),
            chat(json!("sometimes"), json!({})).map(|_| ()),
            chat(
                json!({"type": "function", "function": {"name": ""}}),
                json!({"tools": [{"type": "function", "function": {"name": "", "parameters": {}}}]}),
            )
            .map(|_| ()),
            chat(
                json!({"type": "function", "function": {"name": "get weather"}}),
                json!({"tools": [{"type": "function", "function": {"name": "get weather", "parameters": {}}}]}),
            )
            .map(|_| ()),
        ];
        for refusal in refusals {
            match refusal {
                Err(ServerError::BadRequest { param, .. }) => {
                    assert_eq!(param.as_deref(), Some("tool_choice"))
                }
                other => panic!("expected a 400 naming tool_choice, got {other:?}"),
            }
        }
        // Without the model's template the opener's protocol is unknown.
        let body = json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            "tool_choice": named,
        });
        assert!(serde_json::from_value::<ChatCompletionRequest>(body)
            .unwrap()
            .prompt(None)
            .is_err());
    }

    #[test]
    fn reasoning_effort_matches_the_messages_endpoint() {
        let template =
            include_str!("../../../lumen-runtime/tests/fixtures/qwen38_chat_template.jinja");
        let request = |extra: Value| -> ChatCompletionRequest {
            let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            serde_json::from_value(body).unwrap()
        };
        let prompt = |extra: Value| request(extra).prompt(Some(template)).unwrap().0;
        let anthropic = |output_config: Value| {
            let req: crate::wire::anthropic::MessagesRequest = serde_json::from_value(json!({
                "model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "enabled"}, "output_config": output_config,
            }))
            .unwrap();
            req.prompt(Some(template)).unwrap().0
        };
        // Each level shared with /v1/messages renders the same prompt there.
        for level in ["low", "medium", "high", "xhigh", "max"] {
            assert_eq!(
                prompt(json!({"reasoning_effort": level})),
                anthropic(json!({"effort": level})),
                "{level}"
            );
        }
        // Any level but `none` asks for reasoning; `minimal` runs as `low`.
        assert_eq!(
            prompt(json!({"reasoning_effort": "minimal"})),
            prompt(json!({"reasoning_effort": "low"}))
        );
        assert_eq!(
            prompt(json!({"reasoning_effort": "none"})),
            prompt(json!({"enable_thinking": false}))
        );
        // An explicit toggle wins over the effort.
        assert_eq!(
            prompt(json!({"reasoning_effort": "low", "enable_thinking": false})),
            prompt(json!({"enable_thinking": false}))
        );
        assert_eq!(
            prompt(json!({"reasoning_effort": "none", "enable_thinking": true})),
            prompt(json!({"enable_thinking": true}))
        );
        for (bad, code) in [
            (json!("extreme"), "invalid_value"),
            (json!(3), "invalid_type"),
            (json!(["low"]), "invalid_type"),
        ] {
            match request(json!({"reasoning_effort": bad})).reasoning_effort() {
                Err(ServerError::BadRequest {
                    param, code: got, ..
                }) => {
                    assert_eq!(param.as_deref(), Some("reasoning_effort"));
                    assert_eq!(got.as_deref(), Some(code), "{bad}");
                }
                other => panic!("{bad}: expected a 400, got {other:?}"),
            }
        }
    }

    #[test]
    fn render_chat_prompt_keeps_later_system_where_it_was_sent() {
        let messages = vec![
            system_msg("Sys"),
            user_msg("Q1"),
            system_msg("Be brief."),
            assistant_msg("A1"),
            user_msg("Q2"),
        ];
        let expected = "<|im_start|>system\nSys<|im_end|>\n\
                        <|im_start|>user\nQ1<|im_end|>\n\
                        <|im_start|>system\nBe brief.<|im_end|>\n\
                        <|im_start|>assistant\nA1<|im_end|>\n\
                        <|im_start|>user\nQ2<|im_end|>\n\
                        <|im_start|>assistant\n<think>\n\n</think>\n\n";
        assert_eq!(
            render_chat_prompt(&messages, &[], false, None, None).unwrap(),
            expected
        );
        let template =
            include_str!("../../../lumen-runtime/tests/fixtures/qwen38_chat_template.jinja");
        // The Qwen3.8 template keeps (empty) reasoning on earlier assistant turns.
        assert_eq!(
            render_chat_prompt(&messages, &[], false, Some(template), None).unwrap(),
            expected.replace("assistant\nA1", "assistant\n<think>\n\n</think>\n\nA1")
        );
    }

    #[test]
    fn render_chat_prompt_multi_turn_emits_closed_think_only_at_tail() {
        // Three-turn: user, assistant, user. The empty-think tail must
        // appear ONLY at the final assistant prefix, NOT at the previous
        // assistant turn (which carries real content).
        let messages = vec![user_msg("Q1"), assistant_msg("A1"), user_msg("Q2")];
        let out = render_chat_prompt(&messages, &[], false, None, None).unwrap();
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
        let server_out = render_chat_prompt(&messages, &[], false, None, None).unwrap();
        let cli_out = format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt = "Hello"
        );
        assert_eq!(
            server_out, cli_out,
            "server render must match CLI render byte-for-byte"
        );
    }

    // ---- reasoning_content extraction (non-stream collect_chat) ----

    #[tokio::test]
    async fn collect_chat_thinking_off_has_no_reasoning_content() {
        // Byte-identity guard: thinking off => the message has NO
        // reasoning_content key even if the model literally emits </think>.
        let events = vec![
            tok("plain answer </think> still answer"),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 4,
            },
        ];
        let resp = collect_chat_from_events(events, "test".into(), 1, false)
            .await
            .unwrap();
        let msg = &resp["choices"][0]["message"];
        assert!(
            msg.get("reasoning_content").is_none(),
            "no reasoning_content when thinking off"
        );
        assert_eq!(msg["content"], "plain answer </think> still answer");
    }

    #[tokio::test]
    async fn collect_chat_thinking_on_splits_reasoning_content() {
        let events = vec![
            tok("let me think"),
            tok(" carefully</think>The answer is 42."),
            TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 6,
            },
        ];
        let resp = collect_chat_from_events(events, "test".into(), 1, true)
            .await
            .unwrap();
        let msg = &resp["choices"][0]["message"];
        assert_eq!(msg["reasoning_content"], "let me think carefully");
        assert_eq!(msg["content"], "The answer is 42.");
    }

    // ---- F5: standard sampler params accepted (no 400) + reach SamplingParams ----

    #[test]
    fn chat_request_with_sampler_params_deserializes_not_400() {
        // These fields must deserialize cleanly onto the DTO.
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.7
        });
        let req: ChatCompletionRequest =
            serde_json::from_value(body).expect("sampler params must deserialize, not 400");
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
        assert_eq!(
            zero_job.sampling.presence_penalty, None,
            "zero presence -> None"
        );
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

    /// Build a job from `base` plus `extra`, or the 400's param and code.
    fn job_or_param<R: serde::de::DeserializeOwned>(
        base: Value,
        extra: Value,
        into_job: impl Fn(R, &EngineHandle) -> Result<JobRequest, ServerError>,
    ) -> Result<JobRequest, (Option<String>, Option<String>)> {
        let engine = EngineHandle::new_for_test(4096);
        let mut body = base;
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let req: R = serde_json::from_value(body).expect("the request parses");
        into_job(req, &engine).map_err(|e| match e {
            ServerError::BadRequest { param, code, .. } => (param, code),
            other => panic!("expected a 400, got {other:?}"),
        })
    }

    #[test]
    fn chat_ignores_unused_fields_and_refuses_unproducible_ones() {
        let base = || json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        let job = |extra: Value| job_or_param(base(), extra, ChatCompletionRequest::into_job);
        let plain = job(json!({})).unwrap();
        // Fields this server does not use, and unproducible ones at their
        // defaults, leave the job as it was.
        let ignored = job(json!({
            "store": true, "metadata": {"k": "v"}, "user": "u", "safety_identifier": "s",
            "service_tier": "auto", "prompt_cache_key": "k", "parallel_tool_calls": true,
            "prediction": {"type": "content", "content": "x"}, "verbosity": "medium",
            "definitely_not_a_field": 1, "n": 1.0, "logprobs": false, "top_logprobs": 0,
            "logit_bias": {"42": 0}, "response_format": {"type": "text", "strict": false},
            "tool_choice": "none", "function_call": "none", "functions": [], "modalities": [],
            "audio": null, "echo": false, "best_of": 1, "use_beam_search": false,
            "skip_special_tokens": true, "add_generation_prompt": true, "min_tokens": 0,
            "stop_token_ids": [], "separate_reasoning": true, "priority": 0, "request_id": "r",
            "stream_options": {"include_usage": false, "include_obfuscation": false},
        }))
        .unwrap();
        assert_eq!(ignored.prompt_tokens, plain.prompt_tokens);
        assert_eq!(ignored.max_tokens, plain.max_tokens);
        // Without a limit only the context window bounds the reply.
        assert_eq!(plain.max_tokens, usize::MAX);
        // The newer name for max_tokens is honoured, and wins.
        assert_eq!(
            job(json!({"max_completion_tokens": 7})).unwrap().max_tokens,
            7
        );
        assert_eq!(
            job(json!({"max_tokens": 9, "max_completion_tokens": 7}))
                .unwrap()
                .max_tokens,
            7
        );
        for (field, value) in [
            ("n", json!(2)),
            ("response_format", json!({"type": "json_object"})),
            ("logprobs", json!(true)),
            ("top_logprobs", json!(3)),
            ("logit_bias", json!({"42": 5})),
            ("functions", json!([{"name": "f", "parameters": {}}])),
            ("function_call", json!({"name": "f"})),
            ("tool_choice", json!("required")),
            ("modalities", json!(["text", "audio"])),
            ("audio", json!({"voice": "alloy", "format": "wav"})),
            ("web_search_options", json!({})),
            ("verbosity", json!("low")),
            ("moderation", json!({"model": "omni-moderation-latest"})),
            ("guided_json", json!({"type": "object"})),
            ("guided_choice", json!(["yes", "no"])),
            ("structured_outputs", json!({"regex": "a+"})),
            ("grammar", json!("root ::= \"a\"")),
            ("structural_tag", json!("{}")),
            ("echo", json!(true)),
            ("prompt_logprobs", json!(1)),
            ("min_tokens", json!(5)),
            ("stop_token_ids", json!([1])),
            ("repetition_penalty", json!(1.1)),
            ("bad_words", json!(["x"])),
            ("add_generation_prompt", json!(false)),
            ("chat_template", json!("{{ messages }}")),
            ("lora_path", json!("adapter")),
            ("reasoning", json!({"effort": "high"})),
            ("skip_special_tokens", json!(false)),
        ] {
            let (param, code) = job(json!({ field: value })).unwrap_err();
            assert_eq!(param.as_deref(), Some(field));
            assert_eq!(code.as_deref(), Some("invalid_value"), "{field}");
        }
        // `parallel_tool_calls: false` limits the reply to one call.
        let single = |extra: Value| {
            let mut body = base();
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            serde_json::from_value::<ChatCompletionRequest>(body)
                .unwrap()
                .reply_tools()
                .single
        };
        for bad in [json!("false"), json!(0), json!({})] {
            let (param, code) = job(json!({"parallel_tool_calls": bad})).unwrap_err();
            assert_eq!(
                (param.as_deref(), code.as_deref()),
                (Some("parallel_tool_calls"), Some("invalid_type"))
            );
        }
        assert!(single(json!({"parallel_tool_calls": false})));
        assert!(!single(json!({"parallel_tool_calls": true})));
        assert!(!single(json!({})));
        // `none` with tools offers none of them (see
        // `tool_choice_matches_the_messages_endpoint`).
        let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {}}}]);
        let none = job(json!({"tool_choice": "none", "tools": tools})).unwrap();
        assert_eq!(none.prompt_tokens, plain.prompt_tokens);
    }

    #[test]
    fn completions_ignore_unused_fields_and_refuse_unproducible_ones() {
        let base = || json!({"model": "m", "prompt": "hi"});
        let job = |extra: Value| job_or_param(base(), extra, CompletionRequest::into_job);
        let plain = job(json!({})).unwrap();
        let ignored = job(json!({
            "user": "u", "definitely_not_a_field": 1, "n": 1, "best_of": 1.0, "echo": false,
            "suffix": "", "logit_bias": {}, "logprobs": null,
        }))
        .unwrap();
        assert_eq!(ignored.prompt_tokens, plain.prompt_tokens);
        assert_eq!(ignored.max_tokens, plain.max_tokens);
        for (field, value) in [
            ("n", json!(2)),
            ("best_of", json!(3)),
            ("echo", json!(true)),
            ("suffix", json!("tail")),
            ("logit_bias", json!({"42": 5})),
            ("logprobs", json!(2)),
            ("guided_regex", json!("a+")),
            ("response_format", json!({"type": "json_object"})),
            ("min_tokens", json!(5)),
        ] {
            let (param, code) = job(json!({ field: value })).unwrap_err();
            assert_eq!(param.as_deref(), Some(field));
            assert_eq!(code.as_deref(), Some("invalid_value"), "{field}");
        }
    }

    /// `ignore_eos` reaches the job on both endpoints, off by default, and
    /// leaves the EOS set itself in place: the engine still needs it to know
    /// which tokens to skip.
    #[test]
    fn ignore_eos_reaches_the_job_and_defaults_off() {
        let engine = EngineHandle::new_for_test(512);
        let eos = engine.eos_tokens_for_request();
        assert!(!eos.is_empty(), "the test engine must have EOS ids to drop");
        let chat = |extra: serde_json::Value| {
            let mut body = serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}]
            });
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            serde_json::from_value::<ChatCompletionRequest>(body)
                .unwrap()
                .into_job(&engine)
                .unwrap()
        };
        let completion = |extra: serde_json::Value| {
            let mut body = serde_json::json!({"model": "m", "prompt": "hi"});
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            serde_json::from_value::<CompletionRequest>(body)
                .unwrap()
                .into_job(&engine)
                .unwrap()
        };
        let job = chat(serde_json::json!({}));
        assert!(!job.ignore_eos);
        assert_eq!(job.eos_token_ids, eos);
        assert!(!chat(serde_json::json!({"ignore_eos": false})).ignore_eos);
        let job = chat(serde_json::json!({"ignore_eos": true}));
        assert!(job.ignore_eos);
        assert_eq!(job.eos_token_ids, eos);
        assert!(!completion(serde_json::json!({})).ignore_eos);
        assert!(completion(serde_json::json!({"ignore_eos": true})).ignore_eos);
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
                assert!(
                    message.contains("max_seq_len is"),
                    "msg carries the sentinel: {message}"
                );
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    fn completion_job(prompt: Value) -> Result<JobRequest, ServerError> {
        let engine = EngineHandle::new_for_test(4096);
        serde_json::from_value::<CompletionRequest>(json!({"model": "m", "prompt": prompt}))
            .unwrap()
            .into_job(&engine)
    }

    fn assert_prompt_error(prompt: Value, want_code: &str) {
        match completion_job(prompt.clone()) {
            Err(ServerError::BadRequest { code, param, .. }) => {
                assert_eq!(code.as_deref(), Some(want_code), "prompt {prompt}");
                assert_eq!(param.as_deref(), Some("prompt"), "prompt {prompt}");
            }
            other => panic!("prompt {prompt}: expected a 400, got {other:?}"),
        }
    }

    #[test]
    fn completion_token_id_prompt_reaches_the_job_unchanged() {
        // The test engine's vocabulary is 256 ids; 0 and 255 are its edges.
        let ids = [0u32, 104, 105, 255, 7, 7];
        let job = completion_job(json!(ids)).unwrap();
        assert_eq!(job.prompt_tokens, ids);
        // Ids are not re-tokenized: [104, 105] is the byte tokenizer's "hi".
        assert_eq!(
            completion_job(json!([104, 105])).unwrap().prompt_tokens,
            completion_job(json!("hi")).unwrap().prompt_tokens
        );
    }

    #[test]
    fn completion_string_prompts_still_tokenize() {
        assert_eq!(completion_job(json!("ab")).unwrap().prompt_tokens, [97, 98]);
        assert_eq!(
            completion_job(json!(["a", "b"])).unwrap().prompt_tokens,
            [97, 98]
        );
    }

    #[test]
    fn completion_prompt_outside_the_accepted_forms_is_refused() {
        assert_prompt_error(json!([256]), "invalid_value");
        assert_prompt_error(json!(""), "invalid_value");
        assert_prompt_error(json!([]), "invalid_value");
        assert_prompt_error(json!([1, 2, 4_294_967_295u64]), "invalid_value");
        for bad in [
            json!([1, "a"]),
            json!(["a", 1]),
            json!([[1, 2]]),
            json!([-1]),
            json!([1.5]),
            json!([4_294_967_296u64]),
            json!([null]),
            json!(5),
            json!(null),
            json!({"text": "a"}),
        ] {
            assert_prompt_error(bad, "invalid_type");
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
        assert!(
            req.into_job(&engine).is_ok(),
            "short prompt must pass the guard"
        );
    }
}
