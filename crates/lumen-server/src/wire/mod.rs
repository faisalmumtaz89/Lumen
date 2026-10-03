//! Wire-format encoders.
//!
//! Each submodule owns the request DTO, the SSE state machine, and the
//! non-streaming response shape for one external API.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use lumen_runtime::tooling::ToolSchemas;
use serde_json::Value;

use crate::error::ServerError;
use crate::sse::ReplyTools;

pub mod anthropic;
pub mod image;
pub mod openai;

/// Refuse, before the job is submitted, a request the bench token-id surface
/// (`LUMEN_BENCH_TOKEN_IDS=1`) cannot serve: a streaming response has no body
/// to carry the `lumen_bench` object, and a stop sequence ends the response at
/// the wire layer before the engine's token-id record arrives. Both would
/// otherwise be served and then fail, or be served without the surface. A
/// no-op when the flag is off.
pub(crate) fn bench_token_ids_guard(stream: bool, stop_text: &[String]) -> Result<(), ServerError> {
    if !lumen_runtime::runtime_defaults::bench_token_ids_enabled() {
        return Ok(());
    }
    if stream {
        return Err(ServerError::BadRequest {
            message:
                "LUMEN_BENCH_TOKEN_IDS is not supported on streaming responses; use stream=false"
                    .to_string(),
            param: Some("stream".to_string()),
            code: Some("unsupported_with_bench_token_ids".to_string()),
        });
    }
    if !stop_text.is_empty() {
        return Err(ServerError::BadRequest {
            message: "LUMEN_BENCH_TOKEN_IDS is not supported with stop sequences".to_string(),
            param: Some("stop".to_string()),
            code: Some("unsupported_with_bench_token_ids".to_string()),
        });
    }
    Ok(())
}

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

/// Server-internal repetition penalty, model-aware and greedy-aware. Resolved
/// per request from the effective `temperature`, in this order:
///
/// 1. `LUMEN_REPETITION_PENALTY=<f32>` (finite, > 0) always wins — diagnostics,
///    or restoring the previous default (`=1.05` dense, `=1.03` MoE).
/// 2. At greedy decoding (`temperature <= 0.0`, the sampler's own greedy switch)
///    the default repetition penalty is `1.0`: greedy then selects the argmax of
///    the model's own logits, not penalty-shifted ones (a non-unit penalty
///    reshapes the logits BEFORE the argmax in `sampling::sample_logits` and
///    changes which token wins). This is the model's greedy output, matching
///    the common `temperature = 0` convention; the GDN F64 decode fix keeps the
///    arithmetic correct at greedy without the penalty.
///
///    Tradeoff: the penalty's *other* role was damping long-form repetition, which
///    pure greedy gives up — long generations can repeat or loop (dense and q8/q4
///    MoE have no other default guard; only BF16 MoE keeps `anti_restate`). To
///    restore taming set `LUMEN_REPETITION_PENALTY` (process-global, all requests),
///    or use `temperature > 0` (per request).
///    "Greedy" here means no *repetition* penalty: an active `frequency_penalty`
///    or `anti_restate` still shapes the selection.
/// 3. Otherwise (sampling, `temperature > 0`) the model-aware default from
///    [`lumen_runtime::runtime_defaults::repetition_penalty_default`]: `1.05`
///    dense, `1.03` MoE (Qwen3.5-MoE-35B-A3B class, all quants).
///
/// The MoE 1.03 cap and the dense 1.05 rationale (the historical 1.08/1.10 MoE
/// band-aid is gone; 1.05+ corrupts MoE digit arithmetic — the matrix-proven
/// "17 x 20 = … = 39") live in `repetition_penalty_default`, the single source
/// of truth for the sampling-temperature default.
pub(crate) fn diag_repetition_penalty(temperature: f32) -> f32 {
    repetition_penalty_for(
        std::env::var("LUMEN_REPETITION_PENALTY")
            .ok()
            .and_then(|v| v.parse::<f32>().ok()),
        temperature,
    )
}

/// Pure resolution core for [`diag_repetition_penalty`] (the
/// `LUMEN_REPETITION_PENALTY` env is already parsed into `env_override`). Split
/// out so the resolution order is unit-tested without touching process env. A
/// non-finite or `<= 0` `env_override` is ignored.
fn repetition_penalty_for(env_override: Option<f32>, temperature: f32) -> f32 {
    if let Some(env) = env_override.filter(|v| v.is_finite() && *v > 0.0) {
        return env;
    }
    if temperature <= 0.0 {
        return 1.0;
    }
    lumen_runtime::runtime_defaults::repetition_penalty_default()
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

/// A request's reasoning effort level, the one both APIs share (Anthropic
/// `output_config.effort`, OpenAI `reasoning_effort`), as the chat template's
/// `reasoning_effort`. `low` and `medium` pass through; `high`, `xhigh` and
/// `max` keep the template's default, which is its highest level (`xhigh` on
/// Qwen3.8), so the variable is left unset. `None` for any other level.
pub(crate) fn template_reasoning_effort(level: &str) -> Option<Option<&'static str>> {
    match level {
        "low" => Some(Some("low")),
        "medium" => Some(Some("medium")),
        "high" | "xhigh" | "max" => Some(None),
        _ => None,
    }
}

/// A request's tool choice, as both chat APIs express it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolChoice {
    /// The model decides.
    Auto,
    /// No tool call: the prompt offers no tools.
    None,
    /// A call to some offered tool.
    Required,
    /// A call to this tool.
    Named(String),
}

impl ToolChoice {
    /// Refuse a choice the request cannot honour: a named tool it does not
    /// offer or whose name is not one both APIs allow (1 to 64 letters, digits,
    /// `_`, `-`; the name goes into the prompt, see [`Self::response_prefix`]), a
    /// required call with no tools, or a forced call while thinking is on (the
    /// reply opens with the call) or without the model's chat template, whose
    /// tool-call protocol the opener follows.
    pub(crate) fn check<'a>(
        &self,
        mut tools: impl Iterator<Item = &'a str>,
        thinking: bool,
        templated: bool,
    ) -> Result<(), ServerError> {
        let refusal = match self {
            Self::Auto | Self::None => return Ok(()),
            Self::Required if tools.next().is_none() => {
                "tool_choice requires a tool call but `tools` is empty"
            }
            Self::Named(name)
                if name.is_empty()
                    || name.len() > 64
                    || !name
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') =>
            {
                "tool_choice names a tool whose name is not 1 to 64 letters, digits, `_` or `-`"
            }
            Self::Named(name) if !tools.any(|t| t == name) => {
                "tool_choice names no tool that `tools` offers"
            }
            _ if thinking => "tool_choice cannot force a tool call while thinking is on",
            _ if !templated => {
                "tool_choice cannot force a tool call without the model's chat template"
            }
            _ => return Ok(()),
        };
        Err(ServerError::bad_request_field(
            refusal,
            "tool_choice",
            "invalid_value",
        ))
    }

    /// Whether the prompt offers the tools: not for [`Self::None`].
    pub(crate) fn offers_tools(&self) -> bool {
        *self != Self::None
    }

    /// The tool calls a reply may carry: none under [`Self::None`] (tool-call
    /// markup stays text), only calls to the named tool under
    /// [`Self::Named`], and at most one when `single`.
    pub(crate) fn reply_tools(&self, schemas: ToolSchemas, single: bool) -> ReplyTools {
        ReplyTools {
            schemas: Arc::new(schemas),
            parsed: *self != Self::None,
            only: match self {
                Self::Named(name) => Some(name.clone()),
                _ => None,
            },
            single,
        }
    }

    /// The text a forced reply starts with: the model's tool-call opener, with
    /// the tool's name when one is named. It ends the prompt, so the model
    /// continues the call, and is reported as the start of the reply.
    pub(crate) fn response_prefix(&self) -> String {
        use lumen_runtime::tooling::forced_tool_call_prefix;
        match self {
            Self::Auto | Self::None => String::new(),
            Self::Required => forced_tool_call_prefix(None),
            Self::Named(name) => forced_tool_call_prefix(Some(name)),
        }
    }
}

/// A request field that must stay at its default, because any other value asks
/// for output this server cannot produce. `accepts` says whether a value asks
/// for nothing more than the default does; `refused` names what it asks for.
pub(crate) struct Unsupported {
    pub field: &'static str,
    pub accepts: fn(&Value) -> bool,
    pub refused: &'static str,
}

/// Whether `value` is the number `n`, however it is written (`1`, `1.0`).
pub(crate) fn is_number(value: &Value, n: f64) -> bool {
    value.as_f64() == Some(n)
}

/// Whether a `logit_bias` map biases nothing.
pub(crate) fn is_zero_bias(value: &Value) -> bool {
    value
        .as_object()
        .is_some_and(|m| m.values().all(|b| is_number(b, 0.0)))
}

/// The 400 for a field whose value asks for something this server cannot do.
pub(crate) fn unsupported(field: &str, refused: &str) -> ServerError {
    ServerError::bad_request_field(
        format!("{field}: {refused} is not supported"),
        field,
        "invalid_value",
    )
}

/// Every endpoint ignores a field it does not use, as long as ignoring it cannot
/// change the answer. The fields in `table` could, so a value other than null or
/// one they accept is refused rather than silently dropped. `other` holds the
/// fields the request does not declare.
pub(crate) fn refuse_unsupported(
    other: &serde_json::Map<String, Value>,
    table: &[Unsupported],
) -> Result<(), ServerError> {
    for u in table {
        if let Some(value) = other.get(u.field) {
            if !value.is_null() && !(u.accepts)(value) {
                return Err(unsupported(u.field, u.refused));
            }
        }
    }
    Ok(())
}

/// A request's reasoning effort field as its level: `None` when absent or null,
/// a 400 naming `param` when it is not a string. The level itself is checked by
/// the caller, since each API lists its own.
pub(crate) fn effort_level<'a>(
    value: Option<&'a Value>,
    param: &str,
    levels: &str,
) -> Result<Option<&'a str>, ServerError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(level)) => Ok(Some(level)),
        Some(_) => Err(ServerError::bad_request_field(
            format!("{param} must be one of {levels}"),
            param,
            "invalid_type",
        )),
    }
}

/// Tool-call ids: a 64-bit namespace drawn at random when the generator is
/// made, then a count. Clients keep every earlier turn's ids in their history
/// and treat a repeated one as the same call, and a conversation outlives the
/// server process that answered its earlier turns, so an id must not repeat
/// within a process (the count, until it wraps after 2^64 ids) nor across
/// processes: two independently started processes draw the same namespace
/// with probability about 2^-64, given the OS's random source. No clock is
/// read.
pub(crate) struct ToolCallIds {
    namespace: u64,
    count: AtomicU64,
}

impl ToolCallIds {
    pub(crate) fn new() -> Self {
        use std::hash::{BuildHasher, Hasher};
        Self {
            // The standard library's hasher keys are drawn from the OS's
            // random source; an empty hash under them is a random 64-bit value.
            namespace: std::collections::hash_map::RandomState::new()
                .build_hasher()
                .finish(),
            count: AtomicU64::new(0),
        }
    }

    /// The next id under `prefix` (`toolu` for Anthropic, `call` for OpenAI).
    pub(crate) fn next(&self, prefix: &str) -> String {
        let n = self.count.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}_lumen_{:016x}{n:x}", self.namespace)
    }
}

/// The server's tool-call id: [`ToolCallIds`] made once per process.
pub(crate) fn tool_call_id(prefix: &str) -> String {
    static IDS: OnceLock<ToolCallIds> = OnceLock::new();
    IDS.get_or_init(ToolCallIds::new).next(prefix)
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
    fn repetition_penalty_env_override_wins_even_at_greedy() {
        // A set env value (finite, > 0) always wins, including at temperature 0,
        // which restores the previous default behaviour (`=1.05` at temp 0).
        assert_eq!(repetition_penalty_for(Some(1.05), 0.0), 1.05);
        assert_eq!(repetition_penalty_for(Some(1.05), 0.7), 1.05);
        assert_eq!(repetition_penalty_for(Some(2.0), 0.0), 2.0);
    }

    #[test]
    fn repetition_penalty_is_unity_at_greedy_without_env() {
        // `temperature <= 0.0` is the sampler's greedy switch; pure greedy applies
        // no penalty so the argmax is over the raw logits.
        assert_eq!(repetition_penalty_for(None, 0.0), 1.0);
        assert_eq!(repetition_penalty_for(None, -0.5), 1.0);
    }

    #[test]
    fn repetition_penalty_uses_model_default_when_sampling() {
        // `temperature > 0` keeps the model-aware default (unchanged behaviour).
        // Pin the omitted-temperature fallback (`default_temperature`, 0.7): an
        // omitted temperature must stay penalized, not become pure greedy — guards
        // the wire constructors' `unwrap_or_else(default_temperature)`.
        assert_eq!(
            repetition_penalty_for(None, lumen_runtime::runtime_defaults::default_temperature()),
            lumen_runtime::runtime_defaults::repetition_penalty_default()
        );
        assert_eq!(
            repetition_penalty_for(None, 0.7),
            lumen_runtime::runtime_defaults::repetition_penalty_default()
        );
    }

    #[test]
    fn repetition_penalty_rejects_nonpositive_or_nonfinite_env() {
        // A rejected env override falls through to the temperature logic.
        assert_eq!(repetition_penalty_for(Some(0.0), 0.0), 1.0);
        assert_eq!(repetition_penalty_for(Some(-1.0), 0.0), 1.0);
        assert_eq!(repetition_penalty_for(Some(f32::NAN), 0.0), 1.0);
        assert_eq!(
            repetition_penalty_for(Some(-1.0), 0.7),
            lumen_runtime::runtime_defaults::repetition_penalty_default()
        );
    }

    #[test]
    fn repetition_penalty_greedy_boundary_matches_the_sampler() {
        // The `<= 0.0` cutoff must agree with the sampler's greedy switch on edge
        // values. (serde rejects NaN/Inf in request JSON, so these reach the
        // resolver only defensively, but it must stay correct for all of them.)
        let default = lumen_runtime::runtime_defaults::repetition_penalty_default();
        assert_eq!(repetition_penalty_for(None, -0.0), 1.0); // signed zero -> greedy
        assert_eq!(repetition_penalty_for(None, f32::NEG_INFINITY), 1.0); // -inf -> greedy
        assert_eq!(repetition_penalty_for(None, f32::from_bits(1)), default); // subnormal > 0 -> sampling
        assert_eq!(repetition_penalty_for(None, f32::NAN), default); // NaN -> not greedy -> default
    }

    #[test]
    fn tool_call_ids_do_not_repeat_within_or_across_generators() {
        // Two generators stand for two server processes; ids are drawn from
        // both under contention and must all differ.
        use std::collections::HashSet;
        use std::sync::Arc;
        use std::thread;
        let (a, b) = (Arc::new(ToolCallIds::new()), Arc::new(ToolCallIds::new()));
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let g = if t % 2 == 0 { a.clone() } else { b.clone() };
                thread::spawn(move || (0..5_000).map(|_| g.next("toolu")).collect::<Vec<_>>())
            })
            .collect();
        let ids: Vec<String> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        let unique: HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), 40_000);
        assert!(ids.iter().all(|id| id.starts_with("toolu_lumen_")
            && id["toolu_lumen_".len()..]
                .chars()
                .all(|c| c.is_ascii_hexdigit())));
        assert_ne!(a.namespace, b.namespace);
    }

    #[test]
    fn next_random_seed_unique_across_threads() {
        // The counter's raison d'être: concurrent callers must never share a
        // seed. Exercise the atomic RMW under real contention — 8 threads x 20k.
        use std::thread;
        let (threads, per) = (8usize, 20_000usize);
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                thread::spawn(move || (0..per).map(|_| next_random_seed()).collect::<Vec<u64>>())
            })
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
        assert_eq!(
            normalize_zero_penalty(Some(0.0)),
            None,
            "explicit 0 -> None (CLI parity)"
        );
        assert_eq!(
            normalize_zero_penalty(Some(-0.0)),
            None,
            "negative-zero -> None"
        );
        assert_eq!(
            normalize_zero_penalty(Some(0.7)),
            Some(0.7),
            "non-zero passes through"
        );
        assert_eq!(
            normalize_zero_penalty(Some(f32::NAN)),
            None,
            "NaN -> None (junk guard)"
        );
        assert_eq!(
            normalize_zero_penalty(Some(f32::INFINITY)),
            None,
            "inf -> None"
        );
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
        assert!(
            matches!(e, ServerError::Runtime(_)),
            "compute error stays Runtime (500)"
        );
    }

    #[test]
    fn check_prompt_length_guard() {
        assert!(check_prompt_length(100, 4096).is_ok(), "under window OK");
        assert!(
            check_prompt_length(4096, 4096).is_ok(),
            "exactly at window OK"
        );
        assert!(check_prompt_length(4097, 4096).is_err(), "over window 400");
        assert!(
            check_prompt_length(99999, 0).is_ok(),
            "context 0 = unknown, skip"
        );
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
        assert_eq!(
            flatten_content(&v, "messages.content").unwrap(),
            "hello world"
        );
    }

    #[test]
    fn flatten_content_string_and_null() {
        assert_eq!(
            flatten_content(&serde_json::json!("hi"), "p").unwrap(),
            "hi"
        );
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
        assert!(
            flatten_content(&serde_json::json!(true), "p").is_err(),
            "bool also 400"
        );
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
        assert!(
            openai_prompt.contains("<tool_response>"),
            "tool_response present"
        );
        assert!(
            openai_prompt.contains("{\"city\":\"Paris\"}"),
            "arguments reconciled (compact)"
        );
    }
}
