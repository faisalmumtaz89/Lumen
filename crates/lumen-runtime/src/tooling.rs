//! Tool calling: schema, rendering, streaming state machine, final parser.
//!
//! Lumen's runtime is model-architecture aware but server-protocol agnostic.
//! This module owns the model-side contract -- "how does the model emit a
//! tool call, and how do we recover a structured call from its token stream?"
//! -- and lets the server (or any other host) translate to the wire format
//! its API requires (OpenAI's `tool_calls`, Anthropic's `tool_use`, etc.).
//!
//! # Supported model family
//!
//! Qwen3 / Qwen3.5 ChatML with native `<tool_call>...</tool_call>` markers.
//! The chat template renders the tool schema list into the system message
//! between `<tools>` and `</tools>`; the assistant emits tool calls as
//! `<tool_call>\n{"name": "fn", "arguments": {...}}\n</tool_call>` blocks.
//!
//! Other architectures (Llama-3, Mistral, Phi-3, ...) ship different
//! formats. New `Renderer` / `Parser` pairs would live next to this one.
//!
//! # Design
//!
//! - [`ToolSchema`] is the structural type the caller hands us. It is
//!   serialized into the prompt by [`Renderer::render_tools_block`].
//! - [`StreamingParser`] consumes the model's emitted text incrementally.
//!   It distinguishes three states: "outside any call", "inside a call's
//!   JSON body", and "uncertain -- holding back a possible marker
//!   prefix". The hold-back is the ds4-style trick that keeps a
//!   straddled marker (`<tool_` arriving in one SSE chunk, `call>` in
//!   the next) from leaking half-emitted text to the client.
//! - [`parse_final`] is the equivalent batch parser for non-streaming
//!   completions; it scans the full assistant message and returns the
//!   plain-text portion and the list of structured calls.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Schema types
// ---------------------------------------------------------------------------

/// A single tool the assistant may call.
///
/// Mirrors the OpenAI / Anthropic function-call shape that callers will
/// translate from at the wire boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSchema {
    /// The function name the model will use in `tool_call.name`.
    pub name: String,

    /// Free-form natural-language description shown to the model.
    pub description: String,

    /// JSON Schema describing the parameter object. Stored as raw text so
    /// arbitrary JSON Schema features pass through unchanged.
    pub parameters_json_schema: String,
}

/// A parsed tool call extracted from the assistant's output.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    /// Function name the model selected.
    pub name: String,

    /// Argument object as raw JSON text. Callers parse / validate this
    /// against the schema in whatever way their wire format requires.
    pub arguments_json: String,
}

// ---------------------------------------------------------------------------
// Markers
// ---------------------------------------------------------------------------

/// The literal opening marker the Qwen3.5 chat template tells the model to
/// emit before a tool-call JSON object.
pub const TOOL_CALL_OPEN: &str = "<tool_call>";

/// The literal closing marker.
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Renderer for Qwen3.5 tool definitions.
///
/// Produces the text that gets injected into the system message between
/// `<tools>` and `</tools>` per Qwen3.5's chat template, plus a per-call
/// helper used by tests / mock pipelines that need to round-trip a known
/// call through the streaming parser.
pub struct Qwen35Renderer;

impl Qwen35Renderer {
    /// Render the tool-list block. Returns text intended to be wrapped in
    /// the model's `<tools>...</tools>` envelope at the call site (we don't
    /// emit the envelope here because chat-template rendering happens in
    /// the tokenizer layer, which already knows the per-model envelope).
    ///
    /// Format (one JSON object per line) matches Qwen3.5's published
    /// template exactly:
    ///
    /// ```text
    /// {"type": "function", "function": {"name": "...", "description": "...", "parameters": <schema>}}
    /// {"type": "function", "function": {...}}
    /// ```
    pub fn render_tools_block(tools: &[ToolSchema]) -> String {
        let mut out = String::new();
        for t in tools {
            out.push_str("{\"type\": \"function\", \"function\": ");
            out.push_str("{\"name\": ");
            json_string_into(&t.name, &mut out);
            out.push_str(", \"description\": ");
            json_string_into(&t.description, &mut out);
            out.push_str(", \"parameters\": ");
            // parameters is already JSON; emit raw.
            out.push_str(&t.parameters_json_schema);
            out.push_str("}}\n");
        }
        out
    }

    /// Convenience for tests: produce the exact text the model is expected
    /// to emit when calling `name` with `arguments` (a JSON string).
    pub fn render_one_call(name: &str, arguments_json: &str) -> String {
        let mut out = String::with_capacity(64 + arguments_json.len());
        out.push_str(TOOL_CALL_OPEN);
        out.push('\n');
        out.push_str("{\"name\": ");
        json_string_into(name, &mut out);
        out.push_str(", \"arguments\": ");
        out.push_str(arguments_json);
        out.push_str("}\n");
        out.push_str(TOOL_CALL_CLOSE);
        out
    }
}

/// JSON-escape `s` and write `"escaped"` into `out`. We hand-roll this to
/// avoid pulling in serde for the runtime crate. RFC 8259 §7.
fn json_string_into(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ---------------------------------------------------------------------------
// Streaming parser
// ---------------------------------------------------------------------------

/// What the streaming parser produces from one `feed` call.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StreamingDelta {
    /// Plain assistant text that is safe to forward to the client right now.
    ///
    /// "Safe" means: no possible prefix of a tool-call open marker is being
    /// held back. The text is byte-exact: concatenating every delta's
    /// `text` field reconstructs the full assistant content stripped of
    /// tool-call markers and their inner JSON bodies.
    pub text: String,

    /// Tool calls fully parsed during this feed call. Each call has its
    /// closing marker observed; the JSON body is captured verbatim.
    pub tool_calls: Vec<ParsedToolCall>,
}

/// State of the streaming tool-call parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParserMode {
    /// Outside any tool call. May be holding back a partial open marker.
    Outside,
    /// Inside a tool call: accumulating the JSON body until `</tool_call>`.
    InsideCall,
}

/// Streaming state machine for Qwen3.5 `<tool_call>...</tool_call>` markers.
///
/// # Contract
///
/// - The same instance is fed the assistant's tokens as they decode.
/// - `feed` returns a [`StreamingDelta`] describing what the client should
///   see right now and what tool calls have been finalized.
/// - When the stream ends, the caller MUST call `finish()`, which flushes
///   any held-back text the parser was uncertain about. If `finish` reports
///   a non-empty `incomplete_tool_call`, the model emitted an unclosed
///   `<tool_call>` -- a programming or sampling error the caller is free
///   to surface.
#[derive(Debug, Clone)]
pub struct StreamingParser {
    mode: ParserMode,
    /// Pending text we cannot yet emit because it MIGHT extend into the
    /// open marker. Always shorter than `TOOL_CALL_OPEN`.
    held_back: String,
    /// Body of the call we're currently parsing (only in `InsideCall`).
    current_body: String,
}

/// What [`StreamingParser::finish`] returns.
#[derive(Debug, Default)]
pub struct StreamingFinish {
    /// Any text still held back at stream end. The parser couldn't have
    /// completed a tool-call marker from it, so it is safe to emit.
    pub flushed_text: String,
    /// If the stream ended inside a `<tool_call>` body, the partial body
    /// is returned here so callers can decide what to do.
    pub incomplete_tool_call: Option<String>,
}

impl Default for StreamingParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingParser {
    pub fn new() -> Self {
        Self {
            mode: ParserMode::Outside,
            held_back: String::new(),
            current_body: String::new(),
        }
    }

    /// Feed the next chunk of decoded assistant text into the parser.
    /// Returns the emit-safe text and any newly completed tool calls.
    pub fn feed(&mut self, chunk: &str) -> StreamingDelta {
        let mut delta = StreamingDelta::default();
        if chunk.is_empty() {
            return delta;
        }

        // We process character-by-character to keep marker matching simple
        // and correct under arbitrary chunk boundaries. The total work is
        // O(len(chunk)) per call.
        let mut input = String::with_capacity(self.held_back.len() + chunk.len());
        input.push_str(&self.held_back);
        input.push_str(chunk);
        self.held_back.clear();

        match self.mode {
            ParserMode::Outside => self.feed_outside(&input, &mut delta),
            ParserMode::InsideCall => self.feed_inside(&input, &mut delta),
        }

        delta
    }

    fn feed_outside(&mut self, input: &str, delta: &mut StreamingDelta) {
        // Scan for a full open marker. Emit everything before it. Whatever
        // tail of `input` MIGHT be the start of an open marker becomes
        // `held_back`.
        let bytes = input.as_bytes();
        let n = bytes.len();
        let marker = TOOL_CALL_OPEN.as_bytes();
        let m = marker.len();

        if n == 0 {
            return;
        }

        if let Some(pos) = find_subslice(bytes, marker) {
            // Emit [0, pos) as safe text.
            delta.text.push_str(&input[..pos]);
            // Transition into the call body. We DROP the open marker
            // itself from the output stream.
            self.mode = ParserMode::InsideCall;
            self.current_body.clear();
            let body_start = pos + m;
            // Continue parsing the remainder as inside-call.
            let remainder = &input[body_start..];
            self.feed_inside(remainder, delta);
        } else {
            // No full marker. Determine the largest suffix of the
            // remaining input that COULD be the start of a marker, hold
            // it back, and emit the rest as safe text.
            let hold_len = longest_marker_prefix(bytes, marker);
            let safe_end_in_input = n - hold_len;
            delta.text.push_str(&input[..safe_end_in_input]);
            self.held_back.push_str(&input[safe_end_in_input..]);
        }
    }

    fn feed_inside(&mut self, input: &str, delta: &mut StreamingDelta) {
        // Inside a call: scan for the close marker. Everything before it is
        // appended to the current body. After we see the close marker, we
        // finalize the call (parse it) and flip back to Outside, recursing
        // on the remainder.
        let close = TOOL_CALL_CLOSE.as_bytes();
        let bytes = input.as_bytes();
        if let Some(pos) = find_subslice(bytes, close) {
            // body grows by input[..pos]
            self.current_body.push_str(&input[..pos]);
            // Finalize.
            if let Some(call) = parse_call_body(&self.current_body) {
                delta.tool_calls.push(call);
            }
            self.current_body.clear();
            self.mode = ParserMode::Outside;
            // Continue with the tail (could contain plain text or another
            // tool call).
            let tail_start = pos + close.len();
            let tail = &input[tail_start..];
            if !tail.is_empty() {
                self.feed_outside(tail, delta);
            }
        } else {
            // Hold back the longest suffix that COULD be the start of the
            // close marker so we don't truncate it across feeds.
            let hold_len = longest_marker_prefix(bytes, close);
            let safe_end = bytes.len() - hold_len;
            self.current_body.push_str(&input[..safe_end]);
            self.held_back.push_str(&input[safe_end..]);
        }
    }

    /// Stream done -- flush any held-back text.
    pub fn finish(mut self) -> StreamingFinish {
        let mut out = StreamingFinish::default();
        if !self.held_back.is_empty() {
            // Outside-mode held-back is plain text; inside-mode held-back
            // belongs to the body.
            match self.mode {
                ParserMode::Outside => out.flushed_text.push_str(&self.held_back),
                ParserMode::InsideCall => self.current_body.push_str(&self.held_back),
            }
            self.held_back.clear();
        }
        if self.mode == ParserMode::InsideCall && !self.current_body.is_empty() {
            out.incomplete_tool_call = Some(self.current_body);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Non-streaming parser
// ---------------------------------------------------------------------------

/// What the batch parser returns: the plain text content with markers
/// stripped, plus the parsed tool calls in order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParsedAssistant {
    pub content: String,
    pub tool_calls: Vec<ParsedToolCall>,
}

/// Parse a full assistant message, stripping `<tool_call>...</tool_call>`
/// blocks into the `tool_calls` list. Bytes between blocks are concatenated
/// into `content` verbatim.
pub fn parse_final(assistant_text: &str) -> ParsedAssistant {
    let mut out = ParsedAssistant::default();
    let mut p = StreamingParser::new();
    let delta = p.feed(assistant_text);
    out.content.push_str(&delta.text);
    out.tool_calls.extend(delta.tool_calls);
    let fin = p.finish();
    out.content.push_str(&fin.flushed_text);
    // An unclosed `<tool_call>` block produces no parsed call; the
    // surviving body is dropped. This matches what every reference
    // implementation does for malformed tool emissions.
    out
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Naive substring search. Marker lengths are small (~11 bytes); a memmem
/// crate would be overkill.
fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    let n = needle.len();
    let last = hay.len() - n;
    for i in 0..=last {
        if &hay[i..i + n] == needle {
            return Some(i);
        }
    }
    None
}

/// Returns the length of the longest suffix of `tail` that equals a prefix
/// of `marker`. Used to hold back ambiguous chunk tails -- if `tail` ends
/// with `<tool_`, we must not emit those 6 bytes yet because the next
/// chunk might complete the marker.
///
/// O(min(tail.len(), marker.len()) ** 2) which is fine for tiny markers.
fn longest_marker_prefix(tail: &[u8], marker: &[u8]) -> usize {
    let max = tail.len().min(marker.len().saturating_sub(1));
    for k in (1..=max).rev() {
        if &tail[tail.len() - k..] == &marker[..k] {
            return k;
        }
    }
    0
}

/// Parse a tool-call body: expects JSON of the shape
/// `{"name": "...", "arguments": ...}`. Returns None on malformed input.
///
/// We hand-roll this minimal parser instead of pulling in serde. The Qwen3.5
/// chat template fixes the exact shape; anything else is a model emission
/// bug that we surface to the caller via "no parsed call".
fn parse_call_body(body: &str) -> Option<ParsedToolCall> {
    let trimmed = body.trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return None;
    }
    let name = extract_json_string_field(trimmed, "name")?;
    let arguments_json = extract_json_value_field(trimmed, "arguments")?;
    Some(ParsedToolCall { name, arguments_json })
}

/// Extract `"key": "string-value"` from a JSON object body. Returns the
/// unescaped string. None if the key is absent or the value is not a string.
fn extract_json_string_field(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let after = find_unescaped(body, &needle)?;
    let rest = body[after..].trim_start_matches(|c: char| c.is_whitespace() || c == ':');
    if !rest.starts_with('"') {
        return None;
    }
    let inner = &rest[1..];
    let mut out = String::with_capacity(inner.len());
    let mut iter = inner.chars();
    while let Some(c) = iter.next() {
        match c {
            '"' => return Some(out),
            '\\' => match iter.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'b' => out.push('\u{08}'),
                'f' => out.push('\u{0C}'),
                'u' => {
                    let hex: String = (&mut iter).take(4).collect();
                    if hex.len() != 4 {
                        return None;
                    }
                    let cp = u32::from_str_radix(&hex, 16).ok()?;
                    out.push(char::from_u32(cp)?);
                }
                _ => return None,
            },
            c => out.push(c),
        }
    }
    None
}

/// Extract the raw text of a JSON value associated with `key`. Returns the
/// substring covering exactly one JSON value (object, array, string, number,
/// boolean, or null) starting at the colon after the key.
fn extract_json_value_field(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let after = find_unescaped(body, &needle)?;
    let rest = body[after..].trim_start_matches(|c: char| c.is_whitespace() || c == ':');
    let len = json_value_len(rest)?;
    Some(rest[..len].to_string())
}

/// Find the byte offset just past the first non-escaped occurrence of
/// `needle` in `body`. None if not present.
fn find_unescaped(body: &str, needle: &str) -> Option<usize> {
    let nb = needle.as_bytes();
    let hb = body.as_bytes();
    let mut i = 0;
    while i + nb.len() <= hb.len() {
        if &hb[i..i + nb.len()] == nb {
            // Check the preceding byte is not a backslash (within a string
            // value an escaped quote would precede an embedded key-like
            // sequence -- this is paranoid but cheap).
            let escaped = i > 0 && hb[i - 1] == b'\\';
            if !escaped {
                return Some(i + nb.len());
            }
        }
        i += 1;
    }
    None
}

/// Returns the byte length of a single JSON value starting at the beginning
/// of `s`. Skips leading whitespace. Returns None on malformed input.
fn json_value_len(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i == bytes.len() {
        return None;
    }
    let start = i;
    let c = bytes[i] as char;
    match c {
        '{' | '[' => {
            let close = if c == '{' { b'}' } else { b']' };
            let open = bytes[i];
            let mut depth = 1usize;
            let mut in_str = false;
            let mut escape = false;
            i += 1;
            while i < bytes.len() {
                let b = bytes[i];
                if in_str {
                    if escape {
                        escape = false;
                    } else if b == b'\\' {
                        escape = true;
                    } else if b == b'"' {
                        in_str = false;
                    }
                } else {
                    if b == b'"' {
                        in_str = true;
                    } else if b == open {
                        depth += 1;
                    } else if b == close {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1 - start);
                        }
                    }
                }
                i += 1;
            }
            None
        }
        '"' => {
            let mut escape = false;
            i += 1;
            while i < bytes.len() {
                let b = bytes[i];
                if escape {
                    escape = false;
                } else if b == b'\\' {
                    escape = true;
                } else if b == b'"' {
                    return Some(i + 1 - start);
                }
                i += 1;
            }
            None
        }
        c if c == '-' || c.is_ascii_digit() => {
            while i < bytes.len() {
                let b = bytes[i];
                let cb = b as char;
                if cb == '-' || cb == '+' || cb == '.' || cb.is_ascii_digit() || cb == 'e' || cb == 'E' {
                    i += 1;
                } else {
                    break;
                }
            }
            Some(i - start)
        }
        't' if s[i..].starts_with("true") => Some(4),
        'f' if s[i..].starts_with("false") => Some(5),
        'n' if s[i..].starts_with("null") => Some(4),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Public helper: rendering a tool-augmented chat prompt
// ---------------------------------------------------------------------------

/// Build the Qwen3.5 system-message addition that advertises `tools` to the
/// model. The caller composes this with whatever base system content they
/// want. The return value is *just the tool block*, including the
/// `<tools>...</tools>` envelope and the trailing instruction the official
/// template carries.
pub fn qwen35_system_tool_block(tools: &[ToolSchema]) -> String {
    let body = Qwen35Renderer::render_tools_block(tools);
    // The Qwen3.5 official template wraps the JSON list in `<tools>...</tools>`
    // and follows with a brief usage instruction. Keep this body close to
    // the published reference so model behavior remains predictable.
    let mut out = String::with_capacity(256 + body.len());
    out.push_str("\n\n# Tools\n\n");
    out.push_str(
        "You may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n\n");
    out.push_str("<tools>\n");
    out.push_str(&body);
    out.push_str("</tools>\n\n");
    out.push_str(
        "For each function call, return a json object with function name and \
         arguments within <tool_call></tool_call> XML tags:\n\n");
    out.push_str("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
    out
}

/// Convenience: build an envelope-aware system message that combines a base
/// system prompt with a tool-block. Returns the full string ready to be
/// passed to `apply_chat_template_with_system`.
pub fn compose_system_with_tools(base_system: Option<&str>, tools: &[ToolSchema]) -> String {
    let base = base_system.unwrap_or("");
    if tools.is_empty() {
        return base.to_string();
    }
    let mut s = String::new();
    s.push_str(base);
    s.push_str(&qwen35_system_tool_block(tools));
    s
}

// ---------------------------------------------------------------------------
// Tool result helper
// ---------------------------------------------------------------------------

/// Format a tool result the way Qwen3.5 expects it back: a user message
/// containing `<tool_response>...</tool_response>`. Multi-result calls
/// concatenate the blocks in order.
pub fn format_tool_responses(results: &[ToolResult<'_>]) -> String {
    let mut out = String::new();
    for r in results {
        out.push_str("<tool_response>\n");
        out.push_str(r.content);
        out.push_str("\n</tool_response>\n");
    }
    out
}

/// A single tool execution result. `content` is the raw JSON or text string
/// the tool produced; the caller chooses the encoding.
#[derive(Debug, Clone, Copy)]
pub struct ToolResult<'a> {
    pub tool_name: &'a str,
    pub content: &'a str,
}

// ---------------------------------------------------------------------------
// Utility: schema map for callers that look up by name
// ---------------------------------------------------------------------------

/// Helper for callers that need O(1) name -> schema lookup when dispatching
/// parsed tool calls. Keeps `ToolSchema` itself a plain data record.
pub fn build_schema_map(tools: &[ToolSchema]) -> HashMap<String, ToolSchema> {
    tools.iter().map(|t| (t.name.clone(), t.clone())).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tool() -> ToolSchema {
        ToolSchema {
            name: "get_weather".into(),
            description: "Get current weather for a city.".into(),
            parameters_json_schema:
                "{\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \
                 \"required\": [\"city\"]}".into(),
        }
    }

    fn calc_tool() -> ToolSchema {
        ToolSchema {
            name: "calc".into(),
            description: "Evaluate a math expression.".into(),
            parameters_json_schema:
                "{\"type\": \"object\", \"properties\": {\"expr\": {\"type\": \"string\"}}}".into(),
        }
    }

    // ---- Renderer ----

    #[test]
    fn renderer_emits_one_function_per_line() {
        let block = Qwen35Renderer::render_tools_block(&[weather_tool(), calc_tool()]);
        assert!(block.contains("\"name\": \"get_weather\""));
        assert!(block.contains("\"name\": \"calc\""));
        assert_eq!(block.matches('\n').count(), 2, "one trailing newline per tool");
    }

    #[test]
    fn renderer_escapes_special_chars_in_description() {
        let t = ToolSchema {
            name: "echo".into(),
            description: "echoes a \"quoted\"\nstring".into(),
            parameters_json_schema: "{}".into(),
        };
        let block = Qwen35Renderer::render_tools_block(&[t]);
        assert!(block.contains("\\\"quoted\\\""));
        assert!(block.contains("\\n"));
    }

    #[test]
    fn render_one_call_produces_well_formed_emission() {
        let s = Qwen35Renderer::render_one_call("get_weather", "{\"city\": \"Paris\"}");
        assert!(s.starts_with("<tool_call>\n"));
        assert!(s.ends_with("</tool_call>"));
        assert!(s.contains("\"name\": \"get_weather\""));
    }

    // ---- Streaming parser: structural cases ----

    #[test]
    fn streaming_no_tool_calls_passes_through() {
        let mut p = StreamingParser::new();
        let delta = p.feed("hello world");
        assert_eq!(delta.text, "hello world");
        assert!(delta.tool_calls.is_empty());
        let fin = p.finish();
        assert!(fin.flushed_text.is_empty());
    }

    #[test]
    fn streaming_complete_call_in_one_chunk() {
        let call = Qwen35Renderer::render_one_call("get_weather", "{\"city\": \"Paris\"}");
        let chunk = format!("Sure. {call} The weather is sunny.");
        let mut p = StreamingParser::new();
        let delta = p.feed(&chunk);
        assert_eq!(delta.text, "Sure.  The weather is sunny.");
        assert_eq!(delta.tool_calls.len(), 1);
        assert_eq!(delta.tool_calls[0].name, "get_weather");
        assert_eq!(delta.tool_calls[0].arguments_json, "{\"city\": \"Paris\"}");
    }

    #[test]
    fn streaming_open_marker_split_across_chunks() {
        // "<tool" arrives in one chunk, "_call>{...}</tool_call>" in the next.
        let mut p = StreamingParser::new();
        let d1 = p.feed("Calling: <tool");
        assert_eq!(d1.text, "Calling: ", "the <tool prefix must be held back");
        assert!(d1.tool_calls.is_empty());

        let d2 = p.feed("_call>\n{\"name\": \"calc\", \"arguments\": {\"expr\": \"1+1\"}}\n</tool_call> done");
        assert_eq!(d2.tool_calls.len(), 1);
        assert_eq!(d2.tool_calls[0].name, "calc");
        assert_eq!(d2.text, " done");
    }

    #[test]
    fn streaming_close_marker_split_across_chunks() {
        let mut p = StreamingParser::new();
        let _ = p.feed("<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"expr\": \"2*3\"}}\n</tool_ca");
        let d = p.feed("ll> finished");
        assert_eq!(d.tool_calls.len(), 1);
        assert_eq!(d.tool_calls[0].name, "calc");
        assert_eq!(d.text, " finished");
    }

    #[test]
    fn streaming_two_consecutive_calls_in_one_chunk() {
        let c1 = Qwen35Renderer::render_one_call("a", "{}");
        let c2 = Qwen35Renderer::render_one_call("b", "{\"x\": 1}");
        let chunk = format!("{c1} mid {c2} end");
        let mut p = StreamingParser::new();
        let delta = p.feed(&chunk);
        assert_eq!(delta.tool_calls.len(), 2);
        assert_eq!(delta.tool_calls[0].name, "a");
        assert_eq!(delta.tool_calls[1].name, "b");
        assert_eq!(delta.text, " mid  end");
    }

    #[test]
    fn streaming_finish_reports_incomplete_call() {
        let mut p = StreamingParser::new();
        let _ = p.feed("<tool_call>\n{\"name\": \"x\", \"arguments\":");
        let fin = p.finish();
        assert!(fin.incomplete_tool_call.is_some());
    }

    #[test]
    fn streaming_byte_for_byte_emission_recoverable() {
        // Emit a long sequence one character at a time and verify the
        // assembled output matches the result of a single-shot feed.
        let full = "Hi! <tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call> Tail.";
        let mut p = StreamingParser::new();
        let mut text_acc = String::new();
        let mut calls_acc = Vec::new();
        for ch in full.chars() {
            let buf = ch.to_string();
            let d = p.feed(&buf);
            text_acc.push_str(&d.text);
            calls_acc.extend(d.tool_calls);
        }
        let fin = p.finish();
        text_acc.push_str(&fin.flushed_text);
        assert_eq!(text_acc, "Hi!  Tail.");
        assert_eq!(calls_acc.len(), 1);
        assert_eq!(calls_acc[0].name, "f");
    }

    // ---- Non-streaming parser ----

    #[test]
    fn final_parser_round_trip() {
        let call = Qwen35Renderer::render_one_call("calc", "{\"expr\": \"7-2\"}");
        let msg = format!("Let me compute that. {call} Done.");
        let parsed = parse_final(&msg);
        assert_eq!(parsed.content, "Let me compute that.  Done.");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "calc");
        assert_eq!(parsed.tool_calls[0].arguments_json, "{\"expr\": \"7-2\"}");
    }

    #[test]
    fn final_parser_handles_no_calls() {
        let parsed = parse_final("Just a plain message.");
        assert_eq!(parsed.content, "Just a plain message.");
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn final_parser_dropping_malformed_call() {
        // Open marker but no close: drop the body silently.
        let parsed = parse_final("Hello <tool_call>\nbroken");
        assert_eq!(parsed.content, "Hello ");
        assert!(parsed.tool_calls.is_empty());
    }

    // ---- System composition ----

    #[test]
    fn compose_system_with_tools_appends_block() {
        let tools = vec![weather_tool()];
        let s = compose_system_with_tools(Some("You are helpful."), &tools);
        assert!(s.starts_with("You are helpful."));
        assert!(s.contains("<tools>"));
        assert!(s.contains("get_weather"));
        assert!(s.contains("</tools>"));
        assert!(s.contains("<tool_call>"));
    }

    #[test]
    fn compose_system_with_tools_empty_passes_through() {
        let s = compose_system_with_tools(Some("base"), &[]);
        assert_eq!(s, "base");
    }

    // ---- Helpers ----

    #[test]
    fn schema_map_builds_lookup() {
        let map = build_schema_map(&[weather_tool(), calc_tool()]);
        assert!(map.contains_key("get_weather"));
        assert!(map.contains_key("calc"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn tool_response_block_is_valid_xml_envelope() {
        let s = format_tool_responses(&[
            ToolResult { tool_name: "calc", content: "{\"value\": 5}" },
            ToolResult { tool_name: "calc", content: "{\"value\": 7}" },
        ]);
        let count = s.matches("<tool_response>").count();
        assert_eq!(count, 2);
        assert!(s.contains("\"value\": 5"));
        assert!(s.contains("\"value\": 7"));
    }

    // ---- JSON helper unit tests ----

    #[test]
    fn json_value_len_handles_nested_objects() {
        let s = "{\"a\": {\"b\": [1, 2, 3]}, \"c\": \"x\"}rest";
        let len = json_value_len(s).unwrap();
        assert_eq!(&s[..len], "{\"a\": {\"b\": [1, 2, 3]}, \"c\": \"x\"}");
    }

    #[test]
    fn json_value_len_string_with_escaped_quote() {
        let s = "\"he said \\\"hi\\\"\"after";
        let len = json_value_len(s).unwrap();
        assert_eq!(&s[..len], "\"he said \\\"hi\\\"\"");
    }

    #[test]
    fn extract_string_field_unescapes() {
        let body = "{\"name\": \"with \\\"q\\\"\", \"arguments\": {}}";
        let v = extract_json_string_field(body, "name").unwrap();
        assert_eq!(v, "with \"q\"");
    }
}
