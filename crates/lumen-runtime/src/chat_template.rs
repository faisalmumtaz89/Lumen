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
fn build_env(template_src: &str) -> Result<Environment<'_>, ChatTemplateError> {
    let mut env = Environment::new();
    // HuggingFace renders chat templates with trim_blocks + lstrip_blocks; the
    // Qwen3.5 template's whitespace layout depends on both being on.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // Python-compatible str/dict methods (.split/.startswith/.rstrip/... and
    // list indexing) the HF template relies on.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", raise_exception);
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
pub fn render_chat_prompt(
    template_src: &str,
    messages: &serde_json::Value,
    tools: &serde_json::Value,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<String, ChatTemplateError> {
    let env = build_env(template_src)?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| ChatTemplateError::Compile(format!("{e:#}")))?;
    let ctx = minijinja::context! {
        messages => Value::from_serialize(messages),
        tools => Value::from_serialize(tools),
        add_generation_prompt => add_generation_prompt,
        enable_thinking => enable_thinking,
    };
    tmpl.render(ctx)
        .map_err(|e| ChatTemplateError::Render(format!("{e:#}")))
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
        let out = render_chat_prompt(tmpl, &msgs, &serde_json::Value::Array(vec![]), false, false)
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
