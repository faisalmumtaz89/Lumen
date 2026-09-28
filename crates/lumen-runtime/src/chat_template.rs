//! Shared chat-template renderer.
//!
//! The model's chat template is embedded in the LBC file (`tokenizer.chat_template`
//! in the source GGUF). For Qwen3.5 that template is authored in Jinja2 and encodes
//! the model's NATIVE tool-calling protocol — the `<function=NAME><parameter=NAME>`
//! XML block wrapped in `<tool_call>...</tool_call>`, the grouped-`<tool_response>`
//! turn shape, the `<tools>`/`<IMPORTANT>` system preamble, and the `<think>` tail.
//!
//! The engine historically hard-coded a ChatML string in two independent places
//! (the CLI's `apply_chat_template_with_system` and the server's
//! `render_chat_prompt`), which (a) advertised the OLD `<tool_call>{"name",
//! "arguments"}` JSON protocol the default-mode model no longer emits, and
//! (b) drifted from the pinned template (trailing-whitespace differences).
//!
//! This module renders the EMBEDDED template through a real Jinja engine
//! ([`minijinja`]) so there is exactly ONE renderer both surfaces call. To match
//! the reference jinja2 (HuggingFace's `apply_chat_template`, an
//! `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`)
//! byte-for-byte we:
//!   * enable `trim_blocks` + `lstrip_blocks`,
//!   * register `minijinja_contrib::pycompat` so Python str methods the template
//!     uses (`.split`, `.startswith`, `.endswith`, `.rstrip`, `.lstrip`) work,
//!   * register `raise_exception` (the template calls it on malformed input),
//! and validate the result against the pinned jinja2 via a token-ID equivalence
//! oracle (§2H / §2D of the validation harness).

use std::sync::{Arc, Mutex};

use minijinja::{Environment, Value};
use serde::Serialize;

/// A `serde_json` formatter that reproduces Python's `json.dumps` DEFAULT
/// separators — `", "` between items and `": "` after a key — which is what
/// HuggingFace's `tojson` filter uses (`json.dumps(ensure_ascii=False)`).
/// serde_json's own compact formatter uses `","`/`":"` (no spaces), so the
/// template's `tool | tojson` / `args_value | tojson` output would otherwise
/// diverge from the reference by exactly the missing spaces. Non-ASCII is left
/// unescaped (serde_json's default == `ensure_ascii=False`).
struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

/// `tojson` filter matching HuggingFace's (`json.dumps(ensure_ascii=False)`):
/// Python default separators, non-ASCII preserved, keys in insertion order
/// (guaranteed by `serde_json`'s `preserve_order`). minijinja has no built-in
/// `tojson` without its `json` feature, and even that would use compact
/// separators — so we register this instead to be byte-faithful to the pinned
/// template. Result is marked safe (the template is not autoescaped, but jinja2's
/// `tojson` returns markup, so we mirror that).
fn tojson_filter(value: Value) -> Result<Value, minijinja::Error> {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyJsonFormatter);
    value.serialize(&mut ser).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson: {e}"),
        )
    })?;
    let s = String::from_utf8(buf).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson utf8: {e}"),
        )
    })?;
    Ok(Value::from_safe_string(s))
}

/// `string` filter matching Python's `str()` for the value kinds the template
/// feeds it. The Qwen3.5 template renders a scalar tool-call argument as
/// `args_value | string` (the non-mapping / non-sequence branch), and the
/// reference jinja2 runs Python `str()`: a bool becomes `True`/`False` (capital),
/// `None` becomes `None`. minijinja's built-in `string` lowercases booleans
/// (`true`/`false`), which would diverge from the pinned template whenever a
/// re-rendered assistant tool-call carries a boolean argument. Numbers and
/// strings already match Python, so they fall through to the default rendering.
fn string_filter(value: Value) -> Value {
    use minijinja::value::ValueKind;
    match value.kind() {
        ValueKind::Bool => Value::from(if value.is_true() { "True" } else { "False" }),
        ValueKind::None | ValueKind::Undefined => Value::from("None"),
        _ => Value::from(value.to_string()),
    }
}

/// Failure rendering a chat template.
#[derive(Debug, Clone)]
pub enum ChatTemplateError {
    /// The template source failed to compile.
    Compile(String),
    /// Rendering failed — malformed messages (`raise_exception`), an unknown
    /// method/filter, or a type error. Carries minijinja's detailed chain.
    Render(String),
}

impl std::fmt::Display for ChatTemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatTemplateError::Compile(m) => write!(f, "chat template compile error: {m}"),
            ChatTemplateError::Render(m) => write!(f, "chat template render error: {m}"),
        }
    }
}

impl std::error::Error for ChatTemplateError {}

/// The template calls `raise_exception('...')` when messages violate its
/// invariants (no user query, system-not-first, images in a system message,
/// unexpected content/role). We surface that as a render error rather than
/// panicking, matching jinja2's `TemplateError`.
fn raise_exception(msg: String) -> Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

/// Build the environment once per render. minijinja environments are cheap to
/// construct (no global registry); the cost is the single template compile,
/// which is dwarfed by prefill/decode. A shared build here keeps the CLI and
/// server byte-identical because they run the SAME configuration.
///
/// With `marking`, `raise_exception` does not fail: it records its message and
/// emits the marker where the template called it (see [`Marking`]).
fn build_env(
    template_src: &str,
    marking: Option<(String, Arc<Mutex<Vec<String>>>)>,
) -> Result<Environment<'_>, ChatTemplateError> {
    let mut env = Environment::new();
    // HuggingFace renders chat templates with trim_blocks + lstrip_blocks; the
    // Qwen3.5 template's whitespace layout depends on both being on.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // Python-compatible str/dict methods (.split/.startswith/.rstrip/... and
    // list indexing) the HF template relies on.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    match marking {
        None => env.add_function("raise_exception", raise_exception),
        Some((marker, raised)) => env.add_function("raise_exception", move |msg: String| {
            raised.lock().unwrap().push(msg);
            Value::from_safe_string(marker.clone())
        }),
    }
    // HuggingFace-faithful `tojson` (see `tojson_filter`). Overrides any builtin.
    env.add_filter("tojson", tojson_filter);
    // Python-`str()`-faithful `string` (bool -> `True`/`False`); see `string_filter`.
    env.add_filter("string", string_filter);
    env.add_template("chat", template_src)
        .map_err(|e| ChatTemplateError::Compile(format!("{e:#}")))?;
    Ok(env)
}

/// Render a chat prompt by applying the model's embedded Jinja `template_src`
/// to `messages` + `tools`.
///
/// `messages` is a JSON array of message objects in the shape the template
/// consumes: `{role, content, ...}`. For assistant turns that carry tool calls,
/// each `tool_calls[].function.arguments` MUST be a JSON OBJECT (not the OpenAI
/// on-wire JSON string) because the template iterates it with `|items`; callers
/// parse the arguments string before building the context (see
/// `lumen-server`'s `render_chat_prompt`). `tools` is a JSON array of the
/// OpenAI function-tool objects (or an empty array / null when there are none —
/// the template treats an empty list as "no tools").
///
/// `add_generation_prompt` appends the assistant tail; `enable_thinking`
/// selects the closed empty-`<think>` tail (`false`, the default) or the open
/// `<think>` tail (`true`), matching the template's `enable_thinking` branch.
/// `reasoning_effort`, when given, is passed to the template as
/// `reasoning_effort` (Qwen3.8 reads `low`, `medium` or `xhigh` while thinking
/// is on); `None` leaves the variable undefined, so the template's default holds.
///
/// A `system` message after the first message stays where it was sent, so the
/// prompt before it is unchanged and instructions apply from that point on. A
/// template that renders it in place as a system turn is used as is. A template
/// that rejects it where it occurs (`raise_exception` at its position, as
/// Qwen3.5 and Qwen3.8 do) gets it as a system turn in the template's own
/// framing at that position. Either behaviour is first confirmed on a probe
/// conversation, and a template the probe shows doing anything else, for
/// example moving the message to the top, dropping it or rendering it as
/// another role, is a render error. The probe covers one conversation without
/// tools; a template that changes this behaviour for other conversations is
/// not detected.
pub fn render_chat_prompt(
    template_src: &str,
    messages: &serde_json::Value,
    tools: &serde_json::Value,
    add_generation_prompt: bool,
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
) -> Result<String, ChatTemplateError> {
    let env = build_env(template_src, None)?;
    let rendered = render(
        &env,
        messages,
        tools,
        add_generation_prompt,
        enable_thinking,
        reasoning_effort,
    );
    let later_systems = later_system_contents(messages);
    if later_systems.is_empty() {
        return rendered;
    }
    let probe = probe_messages(true);
    let no_tools = serde_json::Value::Array(Vec::new());
    match rendered {
        Ok(prompt) => {
            let in_place = render(&env, &probe, &no_tools, false, enable_thinking, None)
                .is_ok_and(|p| keeps_probe_systems_in_place(&env, &p, enable_thinking));
            if in_place {
                Ok(prompt)
            } else {
                Err(ChatTemplateError::Render(
                    "this chat template does not keep a system message where it was sent".into(),
                ))
            }
        }
        Err(rejected) => {
            let marking = Marking::new(template_src).ok_or_else(|| rejected.clone())?;
            // The probe's rejections must all carry one message, and the
            // request's rejections must all carry that same message.
            let framed = marking
                .render(&env, &probe, &no_tools, false, enable_thinking, None)
                .filter(|(p, raised)| {
                    raised.iter().all(|m| *m == raised[0])
                        && keeps_probe_systems_in_place(&env, p, enable_thinking)
                })
                .and_then(|(_, probe_raised)| {
                    marking
                        .render(
                            &env,
                            messages,
                            tools,
                            add_generation_prompt,
                            enable_thinking,
                            reasoning_effort,
                        )
                        .filter(|(_, raised)| raised.iter().all(|m| *m == probe_raised[0]))
                });
            framed.map(|(prompt, _)| prompt).ok_or(rejected)
        }
    }
}

fn render(
    env: &Environment<'_>,
    messages: &serde_json::Value,
    tools: &serde_json::Value,
    add_generation_prompt: bool,
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
) -> Result<String, ChatTemplateError> {
    let tmpl = env
        .get_template("chat")
        .map_err(|e| ChatTemplateError::Compile(format!("{e:#}")))?;
    let effort = match reasoning_effort {
        Some(effort) => minijinja::context! { reasoning_effort => effort },
        None => minijinja::context! {},
    };
    let ctx = minijinja::context! {
        messages => Value::from_serialize(messages),
        tools => Value::from_serialize(tools),
        add_generation_prompt => add_generation_prompt,
        enable_thinking => enable_thinking,
        ..effort
    };
    tmpl.render(ctx)
        .map_err(|e| ChatTemplateError::Render(format!("{e:#}")))
}

/// The content of every `system` message after the first message.
fn later_system_contents(messages: &serde_json::Value) -> Vec<&serde_json::Value> {
    messages
        .as_array()
        .into_iter()
        .flatten()
        .skip(1)
        .filter(|m| m["role"] == "system")
        .map(|m| &m["content"])
        .collect()
}

/// Probe conversation with distinct texts: a later system message between an
/// assistant turn and a user turn, and one at the end.
const PROBE: [(&str, &str); 6] = [
    ("system", "lumen probe system 0"),
    ("user", "lumen probe user 0"),
    ("assistant", "lumen probe assistant 0"),
    ("system", "lumen probe system 1"),
    ("user", "lumen probe user 1"),
    ("system", "lumen probe system 2"),
];

fn probe_messages(with_later_systems: bool) -> serde_json::Value {
    PROBE
        .iter()
        .enumerate()
        .filter(|(i, (role, _))| with_later_systems || *i == 0 || *role != "system")
        .map(|(_, (role, content))| serde_json::json!({"role": role, "content": content}))
        .collect()
}

/// Whether `rendered`, the probe conversation's render, is the probe without its
/// later system messages plus each one's system turn at its own position: the
/// first after the assistant turn and before the next user turn, the second at
/// the end.
fn keeps_probe_systems_in_place(
    env: &Environment<'_>,
    rendered: &str,
    enable_thinking: bool,
) -> bool {
    let no_tools = serde_json::Value::Array(Vec::new());
    let (Ok(without), Some(first), Some(last)) = (
        render(
            env,
            &probe_messages(false),
            &no_tools,
            false,
            enable_thinking,
            None,
        ),
        system_turn(env, &PROBE[3].1.into()),
        system_turn(env, &PROBE[5].1.into()),
    ) else {
        return false;
    };
    let Some(rest) = rendered.strip_suffix(&last) else {
        return false;
    };
    let Some(at) = rest.find(&first) else {
        return false;
    };
    rest.matches(&first).count() == 1
        && format!("{}{}", &rest[..at], &rest[at + first.len()..]) == without
        && rest
            .find(PROBE[2].1)
            .is_some_and(|assistant| assistant < at)
        && rest.find(PROBE[4].1).is_some_and(|user| user > at)
}

/// An environment whose `raise_exception` records its message and emits a
/// random marker instead of failing, so each rejected later system message
/// leaves a marker at its own position.
struct Marking<'s> {
    env: Environment<'s>,
    marker: String,
    raised: Arc<Mutex<Vec<String>>>,
}

impl<'s> Marking<'s> {
    fn new(template_src: &'s str) -> Option<Self> {
        use std::hash::{BuildHasher, Hasher};
        let marker = format!(
            "\u{1}lumen-system-{:016x}\u{1}",
            std::collections::hash_map::RandomState::new()
                .build_hasher()
                .finish()
        );
        let raised = Arc::new(Mutex::new(Vec::new()));
        let env = build_env(template_src, Some((marker.clone(), raised.clone()))).ok()?;
        Some(Self {
            env,
            marker,
            raised,
        })
    }

    /// Render `messages` and replace each marker with the system turn of the
    /// later system message at that position, returning the prompt and the
    /// messages `raise_exception` was called with. `None` unless there is
    /// exactly one call, and one marker, per later system message.
    fn render(
        &self,
        env: &Environment<'_>,
        messages: &serde_json::Value,
        tools: &serde_json::Value,
        add_generation_prompt: bool,
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
    ) -> Option<(String, Vec<String>)> {
        self.raised.lock().unwrap().clear();
        let marked = render(
            &self.env,
            messages,
            tools,
            add_generation_prompt,
            enable_thinking,
            reasoning_effort,
        )
        .ok()?;
        let raised = std::mem::take(&mut *self.raised.lock().unwrap());
        let systems = later_system_contents(messages);
        let pieces: Vec<&str> = marked.split(self.marker.as_str()).collect();
        if raised.len() != systems.len() || pieces.len() != systems.len() + 1 {
            return None;
        }
        let mut prompt = String::with_capacity(marked.len());
        prompt.push_str(pieces[0]);
        for (content, piece) in systems.iter().zip(&pieces[1..]) {
            prompt.push_str(&system_turn(env, content)?);
            prompt.push_str(piece);
        }
        Some((prompt, raised))
    }
}

/// `content` rendered as the template's leading system turn: the conversation
/// `[system, user]` minus the conversation `[user]`, with no tools and thinking
/// off, so the turn carries none of the template's other leading-turn text.
fn system_turn(env: &Environment<'_>, content: &serde_json::Value) -> Option<String> {
    let user = serde_json::json!({"role": "user", "content": PROBE[1].1});
    let no_tools = serde_json::Value::Array(Vec::new());
    let with = render(
        env,
        &serde_json::json!([{"role": "system", "content": content}, user]),
        &no_tools,
        false,
        false,
        None,
    )
    .ok()?;
    let without = render(
        env,
        &serde_json::json!([user]),
        &no_tools,
        false,
        false,
        None,
    )
    .ok()?;
    with.strip_suffix(without.as_str()).map(str::to_owned)
}

/// Convenience for the single-turn CLI path: render an optional system message
/// plus one user message with the embedded template. Equivalent to building the
/// `[{system?}, {user}]` array and calling [`render_chat_prompt`] with no tools.
pub fn render_single_turn(
    template_src: &str,
    system: Option<&str>,
    user: &str,
    enable_thinking: bool,
) -> Result<String, ChatTemplateError> {
    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(serde_json::json!({"role": "system", "content": sys}));
    }
    messages.push(serde_json::json!({"role": "user", "content": user}));
    render_chat_prompt(
        template_src,
        &serde_json::Value::Array(messages),
        &serde_json::Value::Array(Vec::new()),
        true,
        enable_thinking,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal ChatML template exercising the core constructs the real Qwen3.5
    // template uses: reverse-slice loop, namespace, adjacent-loop-item access,
    // trim, and the enable_thinking tail. Kept tiny so these unit tests need no
    // model file; the full embedded-template byte-parity is proven by the
    // harness token-ID equivalence oracle (§2H / §2D).
    const MINI_TMPL: &str = "\
{%- for message in messages %}\
{{- '<|im_start|>' + message.role + '\n' + (message.content | trim) + '<|im_end|>' + '\n' }}\
{%- endfor %}\
{%- if add_generation_prompt %}\
{{- '<|im_start|>assistant\n' }}\
{%- if enable_thinking is defined and enable_thinking is false %}\
{{- '<think>\n\n</think>\n\n' }}\
{%- else %}\
{{- '<think>\n' }}\
{%- endif %}\
{%- endif %}";

    #[test]
    fn single_turn_closed_think_when_disabled() {
        let out = render_single_turn(MINI_TMPL, None, "Hello", false).unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn single_turn_open_think_when_enabled() {
        let out = render_single_turn(MINI_TMPL, None, "Hello", true).unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn single_turn_trims_user_content() {
        // `| trim` strips leading/trailing whitespace — the exact behaviour the
        // hard-coded renderer lacked (§2H u_code / u_whitespace).
        let out = render_single_turn(MINI_TMPL, None, "   spaced   ", false).unwrap();
        assert!(
            out.starts_with("<|im_start|>user\nspaced<|im_end|>\n"),
            "got: {out:?}"
        );
    }

    #[test]
    fn system_plus_user() {
        let out = render_single_turn(MINI_TMPL, Some("Sys"), "Hi", true).unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nSys<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn raise_exception_surfaces_as_render_error() {
        let out = render_chat_prompt(
            "{{- raise_exception('boom') }}",
            &serde_json::Value::Array(vec![]),
            &serde_json::Value::Array(vec![]),
            false,
            false,
            None,
        );
        match out {
            Err(ChatTemplateError::Render(m)) => assert!(m.contains("boom"), "got: {m}"),
            other => panic!("expected render error, got {other:?}"),
        }
    }

    #[test]
    fn pycompat_methods_available() {
        // `.startswith` / `.split` come from minijinja-contrib pycompat; the real
        // Qwen3.5 template uses them (tool-response detection, </think> split).
        let tmpl = "{{- 'yes' if messages[0].content.startswith('<tool_response>') else 'no' }}";
        let msgs =
            serde_json::json!([{"role": "user", "content": "<tool_response>x</tool_response>"}]);
        let out = render_chat_prompt(
            tmpl,
            &msgs,
            &serde_json::Value::Array(vec![]),
            false,
            false,
            None,
        )
        .unwrap();
        assert_eq!(out, "yes");
    }

    // ---- Byte-identity conformance vs the reference jinja2 (§6 render oracle) ----
    //
    // These are the load-bearing tests: they render the ACTUAL pinned Qwen3.5
    // embedded chat_template (fixture, sha256
    // a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715) against a
    // corpus whose `expected` strings were produced by HuggingFace transformers'
    // `render_jinja_template` (exactly what `AutoTokenizer.apply_chat_template`
    // calls — the §2H gate oracle). Byte-identity here == token-id identity for
    // any tokenizer, which is the token-ID equivalence oracle §6 mandates,
    // covering tool advertisement, native `<function=/<parameter=` history,
    // grouped `<tool_response>` turns, nested/array/bool/number/special-char
    // args, multi-tool, and thinking on/off.
    const REAL_TEMPLATE: &str = include_str!("../tests/fixtures/qwen35_chat_template.jinja");
    const REFERENCE_CORPUS: &str = include_str!("../tests/fixtures/qwen35_render_reference.json");

    // Qwen3.8 ships a revised embedded template (fixture, sha256
    // 701ba13a085c0c1b5e05414dec1aa3069904f962beee36f0899e441720b83974): it
    // injects a `reasoning_effort` system preamble (default xhigh), preserves
    // historical `<think>` content by default (`preserve_thinking`), skips
    // empty-string tool arguments, and serializes non-string scalar args via
    // `tojson`. Its corpus re-renders the full Qwen3.5 shape set plus shapes
    // for those new paths through the same HF `render_jinja_template` oracle.
    const QWEN38_TEMPLATE: &str = include_str!("../tests/fixtures/qwen38_chat_template.jinja");
    const QWEN38_CORPUS: &str = include_str!("../tests/fixtures/qwen38_render_reference.json");

    #[test]
    fn embedded_template_byte_identical_to_reference_jinja2() {
        assert_corpus_byte_identical(REAL_TEMPLATE, REFERENCE_CORPUS);
    }

    #[test]
    fn qwen38_template_byte_identical_to_reference_jinja2() {
        assert_corpus_byte_identical(QWEN38_TEMPLATE, QWEN38_CORPUS);
    }

    /// The template with its non-first-system rejection replaced by a system
    /// turn in its own framing: the reference for a later system message.
    fn inline_system_reference(template: &str) -> String {
        let rejection = "{{- raise_exception('System message must be at the beginning.') }}";
        assert_eq!(template.matches(rejection).count(), 1);
        template.replace(
            rejection,
            "{{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}",
        )
    }

    fn weather_tool() -> serde_json::Value {
        serde_json::json!([{"type": "function", "function": {
            "name": "get_weather", "description": "Weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}])
    }

    /// Conversations with later system messages: after a user turn, last, after
    /// tool results that are followed by more tool results, directly after the
    /// leading system message, several, and before a later user query (which
    /// changes how Qwen3.5 renders earlier assistant turns).
    fn later_system_conversations() -> Vec<serde_json::Value> {
        let call = serde_json::json!([{"type": "function", "function": {
            "name": "get_weather", "arguments": {"city": "Paris"}}}]);
        vec![
            serde_json::json!([
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": "  From now on answer in French.  "},
                {"role": "assistant", "content": "Bonjour"},
                {"role": "user", "content": "Weather?"},
                {"role": "system", "content": "The date is 2026-09-28."},
            ]),
            serde_json::json!([
                {"role": "user", "content": "Weather in Paris and Rome?"},
                {"role": "assistant", "content": "", "tool_calls": call},
                {"role": "tool", "content": "18C"},
                {"role": "system", "content": "Tool output may be stale."},
                {"role": "tool", "content": "21C"},
                {"role": "assistant", "content": "Paris 18C, Rome 21C."},
                {"role": "user", "content": "Thanks"},
            ]),
            serde_json::json!([
                {"role": "system", "content": "First."},
                {"role": "system", "content": "Second."},
                {"role": "user", "content": "Go"},
            ]),
            serde_json::json!([
                {"role": "user", "content": "Plan"},
                {"role": "assistant", "content": "", "tool_calls": call},
                {"role": "tool", "content": "18C"},
                {"role": "system", "content": "Be brief."},
                {"role": "assistant", "content": "It is 18C."},
                {"role": "user", "content": "And tomorrow?"},
            ]),
        ]
    }

    fn assert_later_system_in_template_framing(template: &str) {
        let reference = inline_system_reference(template);
        let no_tools = serde_json::json!([]);
        for messages in later_system_conversations() {
            for tools in [&no_tools, &weather_tool()] {
                for thinking in [false, true] {
                    let want =
                        render_chat_prompt(&reference, &messages, tools, true, thinking, None)
                            .expect("reference renders");
                    let got = render_chat_prompt(template, &messages, tools, true, thinking, None)
                        .unwrap_or_else(|e| panic!("{messages}: {e}"));
                    assert_eq!(got, want, "{messages} thinking={thinking}");
                }
            }
        }
    }

    #[test]
    fn qwen38_later_system_renders_in_place_in_template_framing() {
        assert_later_system_in_template_framing(QWEN38_TEMPLATE);
    }

    #[test]
    fn qwen35_later_system_renders_in_place_in_template_framing() {
        assert_later_system_in_template_framing(REAL_TEMPLATE);
    }

    #[test]
    fn later_system_keeps_the_prompt_before_it() {
        let messages = serde_json::json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Weather?"},
        ]);
        let mut with_notice = messages.as_array().unwrap().clone();
        with_notice.push(serde_json::json!({"role": "system", "content": "Be brief."}));
        let before = render_chat_prompt(
            QWEN38_TEMPLATE,
            &messages,
            &weather_tool(),
            false,
            true,
            None,
        )
        .unwrap();
        let after = render_chat_prompt(
            QWEN38_TEMPLATE,
            &serde_json::Value::Array(with_notice),
            &weather_tool(),
            true,
            true,
            None,
        )
        .unwrap();
        assert_eq!(
            after,
            format!(
                "{before}<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        );
    }

    #[test]
    fn other_rejections_still_fail_with_a_later_system() {
        // No user query at all: the template's own error, not a framed render.
        let messages = serde_json::json!([
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
        ]);
        match render_chat_prompt(
            QWEN38_TEMPLATE,
            &messages,
            &serde_json::json!([]),
            true,
            false,
            None,
        ) {
            Err(ChatTemplateError::Render(m)) => {
                assert!(m.contains("No user query found"), "got: {m}")
            }
            other => panic!("expected the template's rejection, got {other:?}"),
        }
    }

    #[test]
    fn template_that_renders_later_system_in_place_is_used_as_is() {
        let messages = serde_json::json!([
            {"role": "system", "content": "A"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "B"},
        ]);
        let out = render_chat_prompt(
            MINI_TMPL,
            &messages,
            &serde_json::json!([]),
            false,
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nA<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>system\nB<|im_end|>\n"
        );
    }

    #[test]
    fn template_that_moves_or_drops_later_system_is_refused() {
        let hoists = "{%- for m in messages if m.role == 'system' %}[{{ m.content }}]{%- endfor %}\
                      {%- for m in messages if m.role != 'system' %}<{{ m.content }}>{%- endfor %}";
        let drops = "{%- for m in messages %}{%- if m.role != 'system' or loop.first %}<{{ m.content }}>{%- endif %}{%- endfor %}";
        let messages = serde_json::json!([
            {"role": "system", "content": "A"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "B"},
        ]);
        for tmpl in [hoists, drops] {
            // Without a later system message the template renders normally.
            let first_two = serde_json::Value::Array(messages.as_array().unwrap()[..2].to_vec());
            assert!(render_chat_prompt(
                tmpl,
                &first_two,
                &serde_json::json!([]),
                false,
                false,
                None
            )
            .is_ok());
            match render_chat_prompt(tmpl, &messages, &serde_json::json!([]), false, false, None) {
                Err(ChatTemplateError::Render(m)) => {
                    assert!(m.contains("where it was sent"), "got: {m}")
                }
                other => panic!("{tmpl}: expected refusal, got {other:?}"),
            }
        }
    }

    #[test]
    fn template_that_misplaces_later_system_only_sometimes_is_refused() {
        const TURN: &str =
            "{%- macro turn(m) %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{%- endmacro %}";
        let as_user = format!(
            "{TURN}{{%- for m in messages %}}{{%- if m.role == 'system' and not loop.first %}}\
             <user>{{{{ m.content }}}}</user>{{%- else %}}{{{{ turn(m) }}}}{{%- endif %}}{{%- endfor %}}"
        );
        let drops_last = format!(
            "{TURN}{{%- for m in messages %}}{{%- if not (m.role == 'system' and loop.last) %}}\
             {{{{ turn(m) }}}}{{%- endif %}}{{%- endfor %}}"
        );
        // Rejects later system messages in a validation pass before the turns.
        let validates_first = format!(
            "{TURN}{{%- for m in messages[:2] %}}{{{{ turn(m) }}}}{{%- endfor %}}\
             {{%- for m in messages[2:] if m.role == 'system' %}}{{{{ raise_exception('System message must be at the beginning.') }}}}{{%- endfor %}}\
             {{%- for m in messages[2:] if m.role != 'system' %}}{{{{ turn(m) }}}}{{%- endfor %}}"
        );
        let cases = [
            (
                as_user,
                serde_json::json!([
                    {"role": "user", "content": "Q"},
                    {"role": "system", "content": "N"},
                ]),
            ),
            (
                drops_last,
                serde_json::json!([
                    {"role": "user", "content": "Q"},
                    {"role": "system", "content": "N"},
                ]),
            ),
            (
                validates_first,
                serde_json::json!([
                    {"role": "system", "content": "A"},
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "R"},
                    {"role": "system", "content": "N"},
                    {"role": "user", "content": "Q2"},
                ]),
            ),
        ];
        for (tmpl, messages) in cases {
            let out =
                render_chat_prompt(&tmpl, &messages, &serde_json::json!([]), false, false, None);
            assert!(out.is_err(), "{tmpl}: expected refusal, got {out:?}");
        }
    }

    #[test]
    fn unrelated_rejection_is_not_taken_for_a_later_system() {
        let tmpl = format!(
            "{{%- if tools %}}{{%- set ignored = raise_exception('Unsupported tools.') %}}{{%- endif %}}{QWEN38_TEMPLATE}"
        );
        let messages = serde_json::json!([
            {"role": "user", "content": "Q"},
            {"role": "system", "content": "N"},
        ]);
        // Without tools the later system renders in place.
        assert!(
            render_chat_prompt(&tmpl, &messages, &serde_json::json!([]), true, false, None).is_ok()
        );
        match render_chat_prompt(&tmpl, &messages, &weather_tool(), true, false, None) {
            Err(ChatTemplateError::Render(m)) => {
                assert!(m.contains("Unsupported tools."), "got: {m}")
            }
            other => panic!("expected the template's rejection, got {other:?}"),
        }
    }

    fn assert_corpus_byte_identical(template: &str, corpus_json: &str) {
        let corpus: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(corpus_json).expect("parse reference corpus");
        let mut failures = Vec::new();
        for (name, rec) in &corpus {
            let messages = &rec["messages"];
            let tools = &rec["tools"];
            let enable_thinking = rec["enable_thinking"].as_bool().unwrap_or(false);
            let add_generation_prompt = rec["add_generation_prompt"].as_bool().unwrap_or(true);
            let expected = rec["expected"].as_str().expect("expected string");
            match render_chat_prompt(
                template,
                messages,
                tools,
                add_generation_prompt,
                enable_thinking,
                None,
            ) {
                Ok(got) if got == expected => {}
                Ok(got) => {
                    // Show the first differing byte for a precise diagnostic.
                    let at = got
                        .bytes()
                        .zip(expected.bytes())
                        .position(|(a, b)| a != b)
                        .unwrap_or(got.len().min(expected.len()));
                    failures.push(format!(
                        "[{name}] mismatch at byte {at}:\n  exp: {:?}\n  got: {:?}",
                        &expected[at.saturating_sub(20)..(at + 40).min(expected.len())],
                        &got[at.saturating_sub(20)..(at + 40).min(got.len())],
                    ));
                }
                Err(e) => failures.push(format!("[{name}] render error: {e}")),
            }
        }
        assert!(
            failures.is_empty(),
            "{} / {} shapes diverge from reference jinja2:\n{}",
            failures.len(),
            corpus.len(),
            failures.join("\n")
        );
    }
}
