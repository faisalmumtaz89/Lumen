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
//! Qwen3.5 ChatML with `<tool_call>...</tool_call>` markers. Inside the markers
//! the model emits its NATIVE protocol — a `<function=NAME>` block whose
//! `<parameter=NAME>value</parameter>` children carry each argument as raw text
//! (a scalar `str()`-ified, an object/array `tojson`-ed). The parser
//! reconstructs correctly-typed JSON arguments from the advertised
//! [`ToolSchemas`]. The OLDER `<tool_call>\n{"name","arguments"}\n</tool_call>`
//! JSON body is still accepted (retained backward-compat) so historical callers
//! and the legacy JSON emissions keep parsing; the body parser dispatches on
//! whether the block opens with `{` (JSON) or `<function=` (native).
//!
//! The tool schema list and the native-protocol instructions live in the
//! system message; the engine renders them from the model's EMBEDDED chat
//! template (see `lumen_runtime::chat_template`), NOT from a hand-rolled string
//! here — the render helpers below are retained only for the legacy JSON path
//! and its tests.
//!
//! Other architectures (Llama-3, Mistral, Phi-3, ...) ship different
//! formats. New `Renderer` / `Parser` pairs would live next to this one.
//!
//! # Design
//!
//! - [`ToolSchema`] is the structural type the caller hands us. It is
//!   serialized into the prompt by [`Qwen35Renderer::render_tools_block`].
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
use std::sync::Arc;

use serde_json::Value as JsonValue;

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

/// Per-tool parameter type table used to reconstruct native-protocol tool
/// calls. In the Qwen3.5 native protocol every `<parameter=NAME>` value is
/// emitted as raw text (a scalar is `str()`-ified, an object/array is
/// `tojson`-ed), so recovering correctly-typed JSON arguments requires the
/// declared JSON-Schema `type` of each parameter — a string `"35"` and a number
/// `35` are indistinguishable in the token stream without it.
#[derive(Debug, Clone, Default)]
pub struct ToolSchemas {
    /// function name -> (parameter name -> JSON Schema `type`).
    params: HashMap<String, HashMap<String, String>>,
}

impl ToolSchemas {
    /// Build the type table from the advertised tools. A tool whose
    /// `parameters_json_schema` does not parse contributes an empty entry;
    /// parameters missing a `type` are simply absent (parsed heuristically).
    pub fn from_tools(tools: &[ToolSchema]) -> Self {
        let mut params = HashMap::with_capacity(tools.len());
        for t in tools {
            let mut pmap = HashMap::new();
            if let Ok(schema) = serde_json::from_str::<JsonValue>(&t.parameters_json_schema) {
                if let Some(props) = schema.get("properties").and_then(JsonValue::as_object) {
                    for (name, spec) in props {
                        if let Some(ty) = schema_scalar_type(spec) {
                            pmap.insert(name.clone(), ty);
                        }
                    }
                }
            }
            params.insert(t.name.clone(), pmap);
        }
        ToolSchemas { params }
    }

    /// The declared JSON-Schema `type` of `param` on `func`, if known.
    fn param_type(&self, func: &str, param: &str) -> Option<&str> {
        self.params
            .get(func)
            .and_then(|m| m.get(param))
            .map(String::as_str)
    }

    /// True when no tool schema is known (the parser falls back to the JSON
    /// value heuristic for every parameter).
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

/// Extract a scalar JSON-Schema `type` for coercion. Accepts either a plain
/// string type (`"type": "string"`) or JSON-Schema's array-of-types union
/// (`"type": ["string", "null"]` — a *nullable* parameter, common in real
/// tool / MCP schemas), returning the first non-`"null"` member. `anyOf` /
/// `oneOf` / `$ref` and a missing `type` return `None`, so those parameters
/// fall to the value heuristic (unchanged behaviour). Resolving the union to
/// its non-null member is what lets a nullable-`string` param coerce a bare
/// `35` to the string `"35"` instead of the JSON number `35`.
fn schema_scalar_type(spec: &JsonValue) -> Option<String> {
    match spec.get("type") {
        Some(JsonValue::String(s)) => Some(s.clone()),
        Some(JsonValue::Array(members)) => members
            .iter()
            .filter_map(JsonValue::as_str)
            .find(|s| *s != "null")
            .map(str::to_string),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Markers
// ---------------------------------------------------------------------------

/// The literal opening marker the Qwen3.5 chat template tells the model to
/// emit before a tool-call JSON object.
pub const TOOL_CALL_OPEN: &str = "<tool_call>";

/// The literal closing marker.
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// The start of a tool call in the native protocol, through the function name
/// when one is given: a reply forced to call a tool continues from here. It
/// ends where a token does (after a newline), so the model's next token is the
/// one it would produce there itself.
pub fn forced_tool_call_prefix(name: Option<&str>) -> String {
    match name {
        Some(name) => format!("{TOOL_CALL_OPEN}\n{FUNCTION_OPEN}{name}>\n"),
        None => format!("{TOOL_CALL_OPEN}\n"),
    }
}

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

    /// Render the exact ChatML text the model emits when calling `name` with
    /// `arguments` (a JSON string). This is the shared primitive used by both
    /// the wire tool-call renderer [`render_assistant_tool_call_segment`] (the
    /// OpenAI and Anthropic surfaces) and the tooling tests; it is a
    /// load-bearing production codepath, not a test-only convenience.
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
    json_escape_into(s, out);
    out.push('"');
}

/// JSON-escape `s` WITHOUT the surrounding quotes, appending to `out`. The
/// escaping is per-character and matches serde_json / [`json_string_into`]
/// (RFC 8259 §7), so a string escaped in pieces on character boundaries
/// concatenates to the same bytes as escaping it whole. The incremental
/// tool-call streamer relies on that property to stream a string argument value
/// chunk-by-chunk and still match the buffered path's JSON byte-for-byte.
fn json_escape_into(s: &str, out: &mut String) {
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
}

// ---------------------------------------------------------------------------
// Streaming parser
// ---------------------------------------------------------------------------

/// One incremental tool-call event, emitted while a NATIVE (`<function=…>`)
/// tool-call body streams, for wire formats that stream tool input (Anthropic
/// `input_json_delta`, OpenAI `tool_calls` argument deltas). Between a `Start`
/// and its `End`, the concatenation of every `ArgJsonDelta.partial_json` equals
/// the buffered path's `ParsedToolCall.arguments_json` byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolStreamEvent {
    /// A tool call opened. `name` is the function name (now known).
    Start { name: String },
    /// A fragment of the arguments JSON object, already framed and escaped.
    ArgJsonDelta { partial_json: String },
    /// The tool call closed; its arguments JSON is now complete and valid.
    End,
}

/// One ordered item in a `feed` call's output: plain user-visible text, or a
/// tool-call event. Keeping text and tool events in ONE source-ordered list is
/// what lets the wire layer preserve their exact interleaving (text before,
/// between, or after tool calls) instead of trying to reconstruct it from separate
/// fields — the reconstruction that could not be made correct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// Plain assistant text safe to forward now (no held-back marker prefix).
    Text(String),
    /// A native tool-call event (`Start` / `ArgJsonDelta` / `End`), or a legacy
    /// body's atomic `Start` + arguments + `End`.
    Tool(ToolStreamEvent),
}

/// What the streaming parser produces from one `feed` call: the content as an
/// ordered [`StreamEvent`] list, plus any tool calls finalized this feed.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StreamingDelta {
    /// Content in SOURCE order: `Text` fragments interleaved with tool-call events.
    /// Concatenating every `Text` across all deltas reconstructs the full assistant
    /// content with tool-call markers and bodies removed; the tool events stream a
    /// call's input incrementally (native bodies) or atomically (legacy bodies).
    pub events: Vec<StreamEvent>,

    /// Tool calls fully parsed during this feed call (closing marker observed, JSON
    /// body captured verbatim). Used by the NON-STREAMING consumers (collect_*),
    /// which emit one block at the close; streaming consumers use `events`.
    pub tool_calls: Vec<ParsedToolCall>,
}

impl StreamingDelta {
    /// Concatenate the `Text` events into plain answer text — the view the
    /// non-streaming aggregate and the stop matcher want.
    pub fn text(&self) -> String {
        let mut s = String::new();
        for ev in &self.events {
            if let StreamEvent::Text(t) = ev {
                s.push_str(t);
            }
        }
        s
    }

    /// The tool-call events, in order — the view the byte-identity tests want.
    pub fn tool_stream(&self) -> Vec<ToolStreamEvent> {
        self.events
            .iter()
            .filter_map(|ev| match ev {
                StreamEvent::Tool(te) => Some(te.clone()),
                StreamEvent::Text(_) => None,
            })
            .collect()
    }
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
    /// Tool schemas used to type the native protocol's `<parameter>` values.
    /// `None` (the default / schemaless constructor) parses native scalars
    /// heuristically and legacy JSON bodies exactly. Shared (`Arc`) because the
    /// same schema set is handed to every request's parser.
    schemas: Option<Arc<ToolSchemas>>,
    /// Incremental streamer for native tool-call bodies; emits the
    /// `StreamingDelta::tool_stream` events in parallel with the buffered parse,
    /// reset per tool call. Byte-identical to the buffered `tool_calls` output.
    native: NativeStreamer,
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
            schemas: None,
            native: NativeStreamer::new(None),
        }
    }

    /// Construct a parser that types native `<parameter>` values by `schemas`.
    /// The server builds one `ToolSchemas` per request (from the advertised
    /// tools) and hands it here so the streaming reconstruction is schema-aware
    /// and byte-identical to the non-streaming [`parse_final_with_schemas`].
    pub fn with_schemas(schemas: Arc<ToolSchemas>) -> Self {
        Self {
            mode: ParserMode::Outside,
            held_back: String::new(),
            current_body: String::new(),
            native: NativeStreamer::new(Some(Arc::clone(&schemas))),
            schemas: Some(schemas),
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
            // Emit [0, pos) as safe text, ordered before this call's tool events.
            if pos > 0 {
                delta
                    .events
                    .push(StreamEvent::Text(input[..pos].to_string()));
            }
            // Transition into the call body. We DROP the open marker
            // itself from the output stream.
            self.mode = ParserMode::InsideCall;
            self.current_body.clear();
            self.native = NativeStreamer::new(self.schemas.clone());
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
            if safe_end_in_input > 0 {
                delta
                    .events
                    .push(StreamEvent::Text(input[..safe_end_in_input].to_string()));
            }
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
            // Stream the same body bytes incrementally (native bodies only), then
            // close the streamed object at end-of-body for the rare well-formed
            // call that omits `</function>`. Collect into a local vec so the tool
            // events append to `events` in source order (after any preceding text).
            let mut tool_evs = Vec::new();
            self.native.feed(&input[..pos], &mut tool_evs);
            self.native.end_of_body(&mut tool_evs);
            // Finalize. Native `<function=>` bodies are typed by the request's
            // schemas; legacy JSON bodies ignore them.
            if let Some(call) = parse_call_body(&self.current_body, self.schemas.as_deref()) {
                // A native `<function=>` body already streamed its input
                // incrementally above; a legacy JSON body streams nothing, so
                // surface it as one atomic Start + arguments + End. Detect legacy
                // by the SAME rule `parse_call_body` dispatches on — the first
                // non-whitespace byte is `{` — so a `<function=>` appearing inside a
                // legacy JSON string value neither suppresses the real call nor
                // double-emits (the streamer is inert on legacy bodies).
                if self.current_body.trim_start().starts_with('{') {
                    tool_evs.push(ToolStreamEvent::Start {
                        name: call.name.clone(),
                    });
                    tool_evs.push(ToolStreamEvent::ArgJsonDelta {
                        partial_json: call.arguments_json.clone(),
                    });
                    tool_evs.push(ToolStreamEvent::End);
                }
                delta.tool_calls.push(call);
            }
            delta
                .events
                .extend(tool_evs.into_iter().map(StreamEvent::Tool));
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
            let mut tool_evs = Vec::new();
            self.native.feed(&input[..safe_end], &mut tool_evs);
            delta
                .events
                .extend(tool_evs.into_iter().map(StreamEvent::Tool));
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
// Reasoning ("thinking") extraction
// ---------------------------------------------------------------------------

/// The literal closing marker of a Qwen3.5 reasoning block. The opening
/// `<think>` is part of the *prompt* tail (see
/// `runtime_defaults::think_prompt_tail`); the assistant's emitted stream
/// therefore begins *inside* the reasoning block (when thinking is enabled)
/// and the first thing we must detect in its output is this close marker.
pub const THINK_CLOSE: &str = "</think>";

/// What [`ReasoningExtractor::feed`] / [`ReasoningExtractor::finish`] produce
/// from one chunk: the portion that belongs to the reasoning trace and the
/// portion that belongs to the user-visible answer.
///
/// Invariant: concatenating every delta's `reasoning` reconstructs the full
/// reasoning trace (the text the model emitted before `</think>`, with the
/// marker itself dropped); concatenating every delta's `content` reconstructs
/// the answer (everything after `</think>`). With thinking disabled, all input
/// flows to `content` and `reasoning` is always empty.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReasoningDelta {
    /// Text belonging to the reasoning trace (Qwen3.5 `<think>...</think>`
    /// interior), safe to forward right now. Routed to `reasoning_content`
    /// (OpenAI), a `thinking` block (Anthropic), or a labelled stderr section
    /// (CLI) by the caller.
    pub reasoning: String,

    /// User-visible answer text (everything after the reasoning block), safe
    /// to forward right now.
    pub content: String,
}

/// Streaming splitter that separates a model's reasoning trace from its
/// answer, mirroring [`StreamingParser`]'s hold-back state machine.
///
/// # Why it mirrors `StreamingParser`
///
/// The reasoning/answer boundary is exactly the tool-call problem rotated 90°:
/// a single marker (`</think>` here, `<tool_call>` there) can straddle two SSE
/// chunks, so a partial `</thi` arriving at the end of one feed must NOT be
/// emitted as reasoning until the next feed confirms whether it completes into
/// the marker. We reuse the same `find_subslice` / `longest_marker_prefix`
/// hold-back primitives so the two state machines have identical
/// chunk-boundary semantics.
///
/// # Outermost-first composition
///
/// Reasoning is the OUTERMOST structure: the trace is split off FIRST, then
/// the answer text is handed to the tool-call [`StreamingParser`] (see
/// `lumen-server::sse::SseSafeEmitter`). Tool calls live in the answer, never
/// in the reasoning trace, so this ordering is correct for Qwen3.5.
///
/// # Contract
///
/// - `new(thinking)` starts the machine `in_reasoning = thinking`. When
///   `thinking == false` the extractor is a pure passthrough: `feed` returns
///   all input as `content`, `reasoning` always empty, NO bytes are ever held
///   back — byte-identical to feeding the same text straight through.
/// - The same instance is fed the assistant's decoded text as it streams.
/// - At end-of-stream the caller MUST call [`finish`](Self::finish) to flush
///   any held-back partial marker (which, having not completed, is real text).
#[derive(Debug, Clone)]
pub struct ReasoningExtractor {
    /// True while we are still inside the reasoning block (before `</think>`).
    /// Initialised to the request's `thinking` flag.
    in_reasoning: bool,
    /// Pending reasoning text we cannot yet emit because it MIGHT extend into
    /// the `</think>` close marker. Always shorter than `THINK_CLOSE`. Only
    /// ever non-empty while `in_reasoning` is true.
    held_back: String,
}

impl ReasoningExtractor {
    /// Construct an extractor. `thinking` is the resolved per-request flag
    /// (see `runtime_defaults::resolve_enable_thinking`): `true` means the
    /// prompt opened a `<think>` block so the stream starts in reasoning;
    /// `false` means the closed empty-think tail was used so there is no
    /// reasoning and this is a passthrough.
    pub fn new(thinking: bool) -> Self {
        Self {
            in_reasoning: thinking,
            held_back: String::new(),
        }
    }

    /// Feed the next chunk of decoded assistant text. Returns the
    /// reasoning/answer split that is safe to forward right now.
    pub fn feed(&mut self, chunk: &str) -> ReasoningDelta {
        let mut delta = ReasoningDelta::default();
        if chunk.is_empty() {
            return delta;
        }

        if !self.in_reasoning {
            // Passthrough: everything is answer content. No marker to scan
            // for, nothing held back — byte-identical to a raw forward.
            delta.content.push_str(chunk);
            return delta;
        }

        // Inside the reasoning block: prepend whatever partial-marker tail we
        // held back last time, then scan for the `</think>` close marker.
        let mut input = String::with_capacity(self.held_back.len() + chunk.len());
        input.push_str(&self.held_back);
        input.push_str(chunk);
        self.held_back.clear();

        let bytes = input.as_bytes();
        let marker = THINK_CLOSE.as_bytes();
        if let Some(pos) = find_subslice(bytes, marker) {
            // Reasoning ends here. Text before the marker is reasoning; DROP
            // the marker itself; everything after is answer content.
            delta.reasoning.push_str(&input[..pos]);
            self.in_reasoning = false;
            let tail = &input[pos + marker.len()..];
            delta.content.push_str(tail);
        } else {
            // No full marker. Hold back the longest suffix that COULD be the
            // start of `</think>` so we never truncate it across feeds; emit
            // the rest as reasoning.
            let hold_len = longest_marker_prefix(bytes, marker);
            let safe_end = bytes.len() - hold_len;
            delta.reasoning.push_str(&input[..safe_end]);
            self.held_back.push_str(&input[safe_end..]);
        }
        delta
    }

    /// Flush at end-of-stream. Any text still held back never completed into
    /// `</think>`, so it is genuine reasoning text and is emitted as such.
    /// (If the stream already left the reasoning block, `held_back` is empty
    /// and the returned delta is empty.)
    pub fn finish(mut self) -> ReasoningDelta {
        let mut delta = ReasoningDelta::default();
        if !self.held_back.is_empty() {
            // We can only be holding back while still in_reasoning (the
            // passthrough and post-marker paths never populate held_back), so
            // the residue belongs to the reasoning trace.
            delta.reasoning.push_str(&self.held_back);
            self.held_back.clear();
        }
        delta
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
///
/// Schemaless: native `<parameter>` scalars are typed heuristically (fine for
/// the CLI, which advertises no tools, and for legacy JSON bodies which arrive
/// already typed). Use [`parse_final_with_schemas`] on the server tool path so
/// scalar values are typed by the advertised schema.
pub fn parse_final(assistant_text: &str) -> ParsedAssistant {
    run_final(assistant_text, StreamingParser::new())
}

/// Schema-aware [`parse_final`]: types native `<parameter>` values by the
/// advertised tool schemas so the batch (non-streaming) reconstruction is
/// byte-identical to the streaming one built with the same `schemas`.
pub fn parse_final_with_schemas(
    assistant_text: &str,
    schemas: Arc<ToolSchemas>,
) -> ParsedAssistant {
    run_final(assistant_text, StreamingParser::with_schemas(schemas))
}

fn run_final(assistant_text: &str, mut p: StreamingParser) -> ParsedAssistant {
    let mut out = ParsedAssistant::default();
    let delta = p.feed(assistant_text);
    out.content.push_str(&delta.text());
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

/// Native-protocol markers. The assistant emits, inside `<tool_call>`:
/// `<function=NAME>` then one `<parameter=NAME>\nVALUE\n</parameter>` per
/// argument, then `</function>`.
const FUNCTION_OPEN: &str = "<function=";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

/// Parse the body between `<tool_call>` and `</tool_call>`. Dispatches on the
/// first non-whitespace byte: `{` is the LEGACY JSON protocol (retained
/// backward-compat), `<function=` is the Qwen3.5 NATIVE protocol. `schemas`
/// types the native path's scalar values; it is unused by the JSON path (whose
/// arguments arrive already typed). Returns None on a body that is neither.
fn parse_call_body(body: &str, schemas: Option<&ToolSchemas>) -> Option<ParsedToolCall> {
    let trimmed = body.trim();
    if trimmed.starts_with('{') {
        return parse_json_call_body(trimmed);
    }
    if trimmed.contains(FUNCTION_OPEN) {
        return parse_native_call_body(trimmed, schemas);
    }
    None
}

/// Parse the LEGACY JSON tool-call body `{"name": "...", "arguments": ...}`.
/// Retained so existing callers, the server's own round-trip re-render, and any
/// model that still emits JSON keep working. Returns None on malformed input.
fn parse_json_call_body(trimmed: &str) -> Option<ParsedToolCall> {
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return None;
    }
    let name = extract_json_string_field(trimmed, "name")?;
    let arguments_json = extract_json_value_field(trimmed, "arguments")?;
    Some(ParsedToolCall {
        name,
        arguments_json,
    })
}

/// Parse the Qwen3.5 NATIVE tool-call body:
///
/// ```text
/// <function=get_weather>
/// <parameter=city>
/// Riyadh
/// </parameter>
/// <parameter=unit>
/// celsius
/// </parameter>
/// </function>
/// ```
///
/// Reconstructs a JSON arguments object, coercing each `<parameter>` value by
/// its declared schema type (objects/arrays JSON-parsed, numbers/booleans
/// typed, strings kept verbatim); parameters with no known type fall back to a
/// JSON-value heuristic. The arguments object preserves the emitted parameter
/// order. Returns None when no `<function=NAME>` opener is present.
fn parse_native_call_body(body: &str, schemas: Option<&ToolSchemas>) -> Option<ParsedToolCall> {
    let fstart = body.find(FUNCTION_OPEN)? + FUNCTION_OPEN.len();
    let after_name = &body[fstart..];
    let name_end = after_name.find('>')?;
    let name = after_name[..name_end].trim().to_string();

    // `serde_json::Map` preserves insertion order under the `preserve_order`
    // feature (enabled crate-wide), so the reconstructed arguments keep the
    // model's emitted parameter order.
    let mut map = serde_json::Map::new();
    let mut rest = &after_name[name_end + 1..];
    // Known limitation (RISK-2): a `<parameter>` value that itself contains the
    // literal `</parameter>` truncates that parameter early — IDENTICALLY on the
    // batch and streaming paths (both stop the value at `</parameter>`), so the
    // streaming==batch guarantee holds for it. A value containing the OUTER
    // `</tool_call>` marker is the one case the two paths do NOT agree on: the batch
    // path is handed a body already cut at `</tool_call>` and drops the unclosed
    // parameter (`{}`), while the streamer is mid-value and emits the partial input
    // with no `End` (the wire then reports a truncated turn). The trained Qwen3.5
    // format never emits either marker inside a value, and in the `</tool_call>` case
    // the stream still degrades safely (truncation signalled, block integrity intact,
    // nothing lost on the client's terms). Not rewritten to a full nested parser:
    // that carries regression risk for no observed benefit.
    while let Some(p) = rest.find(PARAM_OPEN) {
        let after_open = &rest[p + PARAM_OPEN.len()..];
        let Some(pname_end) = after_open.find('>') else {
            break;
        };
        let pname = after_open[..pname_end].trim().to_string();
        let value_region = &after_open[pname_end + 1..];
        let Some(close) = value_region.find(PARAM_CLOSE) else {
            break;
        };
        // The template wraps the value as `>\nVALUE\n</parameter>`; strip the one
        // leading and one trailing newline it adds, preserving VALUE's interior.
        let raw = &value_region[..close];
        let value = strip_one_surrounding_newline(raw);
        let coerced = coerce_param_value(value, schemas.and_then(|s| s.param_type(&name, &pname)));
        map.insert(pname, coerced);
        rest = &value_region[close + PARAM_CLOSE.len()..];
    }

    Some(ParsedToolCall {
        name,
        arguments_json: JsonValue::Object(map).to_string(),
    })
}

/// A newline immediately followed by the parameter close. The template puts one
/// `\n` right before `</parameter>` and the buffered parser strips it, so while
/// streaming a string value this whole suffix must be held back until the close
/// is ruled out — otherwise that trailing newline leaks into the value.
const NL_PARAM_CLOSE: &str = "\n</parameter>";

/// Round `i` down to a UTF-8 character boundary of `s`, so a held-back tail never
/// splits a multi-byte character (`feed` always receives valid UTF-8).
fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Phase of [`NativeStreamer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeState {
    SeekFunction,
    ReadFuncName,
    SeekParamOrEnd,
    ReadParamName,
    StreamString,
    BufferTyped,
    /// The body is legacy JSON (first non-whitespace byte `{`), which
    /// `parse_call_body` routes to the JSON parser. The streamer emits nothing;
    /// the legacy atomic triple is produced at finalize.
    NotNative,
    Done,
}

/// Streams a NATIVE (`<function=NAME><parameter=X>…</parameter>…</function>`)
/// tool-call body incrementally as [`ToolStreamEvent`]s. Between the `Start` and
/// `End`, the concatenation of every `ArgJsonDelta.partial_json` equals the
/// buffered [`parse_native_call_body`] `arguments_json` byte-for-byte: it reuses
/// the same [`json_string_into`]/[`json_escape_into`] escaping, the same
/// [`strip_one_surrounding_newline`] trim, and the same [`coerce_param_value`]
/// typing. Only `string`-typed parameters stream their value char-by-char; every
/// other type, and every value with no declared type, is buffered to its close
/// and coerced exactly as the buffered path (those values are short). A legacy
/// JSON body never contains `<function=`, so this emits nothing for it.
#[derive(Debug, Clone)]
struct NativeStreamer {
    state: NativeState,
    hold: String,
    schemas: Option<Arc<ToolSchemas>>,
    func: String,
    param: String,
    first_param: bool,
    leading_nl_pending: bool,
    /// Latched once the body's first non-whitespace byte proves it NATIVE (not a
    /// legacy `{`): later chunks are no longer re-classified, so a non-`{` first
    /// chunk followed by a `{` chunk cannot flip the streamer to `NotNative`.
    committed_native: bool,
    /// Accumulator for a `BufferTyped` value (typed / composite parameters, which
    /// cannot stream and must be coerced whole). Appended across feeds so the hold
    /// stays bounded — O(n) total, not O(n^2).
    typed_buf: String,
}

impl NativeStreamer {
    fn new(schemas: Option<Arc<ToolSchemas>>) -> Self {
        Self {
            state: NativeState::SeekFunction,
            hold: String::new(),
            schemas,
            func: String::new(),
            param: String::new(),
            first_param: true,
            leading_nl_pending: false,
            committed_native: false,
            typed_buf: String::new(),
        }
    }

    fn param_type(&self) -> Option<String> {
        self.schemas
            .as_ref()
            .and_then(|s| s.param_type(&self.func, &self.param))
            .map(str::to_string)
    }

    /// Longest suffix of `s` that is a prefix of `marker` (markers are ASCII).
    fn marker_tail(s: &str, marker: &str) -> usize {
        longest_marker_prefix(s.as_bytes(), marker.as_bytes())
    }

    /// Feed the next body text; append any events to `out`.
    fn feed(&mut self, chunk: &str, out: &mut Vec<ToolStreamEvent>) {
        let mut input = std::mem::take(&mut self.hold);
        input.push_str(chunk);
        let mut rest = input.as_str();
        loop {
            match self.state {
                NativeState::SeekFunction => {
                    // Native-vs-legacy detection matching `parse_call_body`: a body
                    // whose first non-whitespace byte is `{` is LEGACY JSON — the
                    // streamer stays inert (the legacy atomic triple is emitted at
                    // finalize). Any other body is native, so seek its `<function=`
                    // opener. This also stops the streamer from matching a
                    // `<function=` that merely appears inside a legacy JSON string.
                    // The decision is LATCHED at the first non-whitespace byte
                    // (`committed_native`): once a non-`{` body is seen, a later chunk
                    // starting with `{` cannot flip it (the body's true start wins,
                    // as `parse_call_body` dispatches on the whole trimmed body).
                    if !self.committed_native {
                        let trimmed = rest.trim_start();
                        if trimmed.is_empty() {
                            // Leading whitespace before the first non-ws byte is
                            // insignificant (it precedes `<function=` / `{`, which both
                            // the streamer and `parse_call_body` skip or trim), so
                            // DISCARD it rather than re-holding the whole prefix every
                            // feed — the latter is O(n^2) for a long whitespace run.
                            return;
                        }
                        if trimmed.starts_with('{') {
                            self.state = NativeState::NotNative;
                            return;
                        }
                        self.committed_native = true;
                    }
                    match rest.find(FUNCTION_OPEN) {
                        Some(p) => {
                            rest = &rest[p + FUNCTION_OPEN.len()..];
                            self.state = NativeState::ReadFuncName;
                        }
                        None => {
                            let keep = Self::marker_tail(rest, FUNCTION_OPEN);
                            self.hold = rest[rest.len() - keep..].to_string();
                            return;
                        }
                    }
                }
                NativeState::ReadFuncName => match rest.find('>') {
                    Some(e) => {
                        self.func = rest[..e].trim().to_string();
                        out.push(ToolStreamEvent::Start {
                            name: self.func.clone(),
                        });
                        rest = &rest[e + 1..];
                        self.state = NativeState::SeekParamOrEnd;
                    }
                    None => {
                        self.hold = rest.to_string();
                        return;
                    }
                },
                NativeState::SeekParamOrEnd => match rest.find(PARAM_OPEN) {
                    Some(p) => {
                        rest = &rest[p + PARAM_OPEN.len()..];
                        self.state = NativeState::ReadParamName;
                    }
                    None => {
                        // The buffered parser scans every `<parameter=` to the end of
                        // the body and never treats `</function>` as a terminator (it
                        // is skipped like any other between-parameter text). Match
                        // that: hold only a possible `<parameter=` prefix and let
                        // `end_of_body` (the outer `</tool_call>`) close the object.
                        let keep = Self::marker_tail(rest, PARAM_OPEN);
                        self.hold = rest[rest.len() - keep..].to_string();
                        return;
                    }
                },
                NativeState::ReadParamName => match rest.find('>') {
                    Some(e) => {
                        self.param = rest[..e].trim().to_string();
                        let mut prefix = String::from(if self.first_param { "{" } else { "," });
                        json_string_into(&self.param, &mut prefix);
                        prefix.push(':');
                        self.first_param = false;
                        rest = &rest[e + 1..];
                        if self.param_type().as_deref() == Some("string") {
                            prefix.push('"');
                            out.push(ToolStreamEvent::ArgJsonDelta {
                                partial_json: prefix,
                            });
                            self.leading_nl_pending = true;
                            self.state = NativeState::StreamString;
                        } else {
                            out.push(ToolStreamEvent::ArgJsonDelta {
                                partial_json: prefix,
                            });
                            self.state = NativeState::BufferTyped;
                        }
                    }
                    None => {
                        self.hold = rest.to_string();
                        return;
                    }
                },
                NativeState::StreamString => {
                    if self.leading_nl_pending && !rest.is_empty() {
                        rest = rest.strip_prefix('\n').unwrap_or(rest);
                        self.leading_nl_pending = false;
                    }
                    match rest.find(PARAM_CLOSE) {
                        Some(p) => {
                            // Strip the one trailing '\n' the template adds (parity
                            // with `strip_one_surrounding_newline`; the leading one
                            // was dropped on entry), then close the JSON string.
                            let val = rest[..p].strip_suffix('\n').unwrap_or(&rest[..p]);
                            let mut d = String::new();
                            json_escape_into(val, &mut d);
                            d.push('"');
                            out.push(ToolStreamEvent::ArgJsonDelta { partial_json: d });
                            rest = &rest[p + PARAM_CLOSE.len()..];
                            self.state = NativeState::SeekParamOrEnd;
                        }
                        None => {
                            // Hold a possible `</parameter>` prefix, OR a `\n`
                            // followed by a `</parameter>` prefix: a newline right
                            // before the close is the trailing one to strip, so it
                            // must not be emitted until the close is ruled out.
                            let keep = Self::marker_tail(rest, PARAM_CLOSE)
                                .max(Self::marker_tail(rest, NL_PARAM_CLOSE));
                            let safe = floor_char_boundary(rest, rest.len() - keep);
                            if safe > 0 {
                                let mut d = String::new();
                                json_escape_into(&rest[..safe], &mut d);
                                out.push(ToolStreamEvent::ArgJsonDelta { partial_json: d });
                            }
                            self.hold = rest[safe..].to_string();
                            return;
                        }
                    }
                }
                NativeState::BufferTyped => match rest.find(PARAM_CLOSE) {
                    Some(p) => {
                        // Close: the full value is the accumulator plus this chunk's
                        // head. Coerce it exactly as the buffered path.
                        self.typed_buf.push_str(&rest[..p]);
                        let val = strip_one_surrounding_newline(&self.typed_buf);
                        let coerced = coerce_param_value(val, self.param_type().as_deref());
                        out.push(ToolStreamEvent::ArgJsonDelta {
                            partial_json: coerced.to_string(),
                        });
                        self.typed_buf.clear();
                        rest = &rest[p + PARAM_CLOSE.len()..];
                        self.state = NativeState::SeekParamOrEnd;
                    }
                    None => {
                        // Accumulate all but a possible `</parameter>` prefix into the
                        // dedicated buffer; hold only that bounded suffix. This keeps
                        // a long typed/composite value O(n) total instead of
                        // re-copying the whole value into `hold` every feed (O(n^2)).
                        let keep = Self::marker_tail(rest, PARAM_CLOSE);
                        let safe = floor_char_boundary(rest, rest.len() - keep);
                        self.typed_buf.push_str(&rest[..safe]);
                        self.hold = rest[safe..].to_string();
                        return;
                    }
                },
                NativeState::NotNative | NativeState::Done => return,
            }
            if rest.is_empty() {
                return;
            }
        }
    }

    /// The body ended (the outer `</tool_call>` was reached). If the streamer is
    /// cleanly between parameters, close the arguments object (`{}` when empty,
    /// else `}`) and emit `End`. This is the normal close: `</function>` is NOT a
    /// terminator (matching the buffered parser, which scans every `<parameter=`
    /// to the end of the body), so a well-formed call reaches here in
    /// `SeekParamOrEnd` with its `</function>` already skipped. A streamer still
    /// mid-value (or still seeking the function) means the call was cut off: it
    /// emits no `End`, and the caller surfaces the truncation (a tool block still
    /// open at end of stream becomes max_tokens).
    fn end_of_body(&mut self, out: &mut Vec<ToolStreamEvent>) {
        if self.state == NativeState::SeekParamOrEnd {
            out.push(ToolStreamEvent::ArgJsonDelta {
                partial_json: if self.first_param { "{}" } else { "}" }.to_string(),
            });
            out.push(ToolStreamEvent::End);
            self.state = NativeState::Done;
        }
    }
}

/// Strip exactly one leading and one trailing `\n` (the template's per-value
/// framing), leaving any interior newlines of a multi-line value intact.
fn strip_one_surrounding_newline(s: &str) -> &str {
    let s = s.strip_prefix('\n').unwrap_or(s);
    s.strip_suffix('\n').unwrap_or(s)
}

/// Coerce a native `<parameter>` string value into a typed JSON value using the
/// declared schema `param_type`. Objects/arrays are JSON-parsed (they were
/// `tojson`-ed on the way out); numbers/booleans are typed; strings are kept
/// verbatim. When the schema is unknown the value is accepted as JSON only if it
/// parses to a non-string composite/scalar, else treated as a string — so a
/// bare `celsius` stays a string while `{...}`/`42`/`true` are typed.
fn coerce_param_value(value: &str, param_type: Option<&str>) -> JsonValue {
    let as_json = || serde_json::from_str::<JsonValue>(value.trim());
    match param_type {
        Some("string") => JsonValue::String(value.to_string()),
        Some("integer") | Some("number") => as_json()
            .ok()
            .filter(JsonValue::is_number)
            .unwrap_or_else(|| JsonValue::String(value.to_string())),
        // The Qwen3.5 template renders a historical boolean argument with Python
        // `str()` — `True` / `False` (capitalized) — so the model is trained to
        // EMIT that form, not JSON `true`/`false`. Accept both casings; anything
        // else stays a string. (This is the parse-side mirror of the renderer's
        // Python-`str()`-faithful `string` filter — see chat_template.rs.)
        Some("boolean") => match value.trim() {
            "true" | "True" => JsonValue::Bool(true),
            "false" | "False" => JsonValue::Bool(false),
            _ => JsonValue::String(value.to_string()),
        },
        Some("object") | Some("array") => {
            as_json().unwrap_or_else(|_| JsonValue::String(value.to_string()))
        }
        _ => match as_json() {
            Ok(v) if v.is_object() || v.is_array() || v.is_number() || v.is_boolean() => v,
            _ => JsonValue::String(value.to_string()),
        },
    }
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
                if cb == '-'
                    || cb == '+'
                    || cb == '.'
                    || cb.is_ascii_digit()
                    || cb == 'e'
                    || cb == 'E'
                {
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
         You are provided with function signatures within <tools></tools> XML tags:\n\n",
    );
    out.push_str("<tools>\n");
    out.push_str(&body);
    out.push_str("</tools>\n\n");
    out.push_str(
        "For each function call, return a json object with function name and \
         arguments within <tool_call></tool_call> XML tags:\n\n",
    );
    out.push_str(
        "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
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
// Wire-level ChatML tool-turn rendering (SINGLE source of truth)
// ---------------------------------------------------------------------------
//
// Both wire surfaces (OpenAI `/v1/chat/completions` and Anthropic
// `/v1/messages`) must CONSUME a tool round-trip and re-render it into the
// exact same Qwen3.5 ChatML transcript so the model sees a byte-identical
// prompt regardless of which API the caller used. These two helpers are that
// single source of truth: the OpenAI path used to inline the byte sequences
// and the Anthropic path dropped tool blocks entirely. Routing both through
// here makes their tool transcripts byte-identical by construction.

/// Render one assistant tool-call as the ChatML segment that is appended to
/// the assistant turn (AFTER any assistant text content, BEFORE the closing
/// `<|im_end|>`): a leading newline, then the `<tool_call>...</tool_call>`
/// block from [`Qwen35Renderer::render_one_call`].
///
/// `arguments_json` is the raw on-wire JSON value for the call arguments
/// (OpenAI carries it as a JSON *string*; Anthropic carries an `input`
/// *object* the caller serializes to JSON first). The name is JSON-escaped by
/// `render_one_call`, so a name containing a quote can never break the block.
///
/// Output (for `name="get_weather"`, `arguments_json="{\"city\": \"Paris\"}"`):
/// ```text
/// \n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>
/// ```
pub fn render_assistant_tool_call_segment(name: &str, arguments_json: &str) -> String {
    let mut out = String::with_capacity(1 + 64 + arguments_json.len());
    out.push('\n');
    out.push_str(&Qwen35Renderer::render_one_call(name, arguments_json));
    out
}

/// Render a tool-result turn as a full ChatML user message wrapping a single
/// `<tool_response>` block. This is the consume side of the round-trip
/// (matching how the model is taught to emit calls and read responses).
///
/// Output (for `content="{\"temp\": 18}"`):
/// ```text
/// <|im_start|>user\n<tool_response>\n{"temp": 18}\n</tool_response><|im_end|>\n
/// ```
///
/// Note the byte layout deliberately matches the OpenAI surface's prior inline
/// form (no newline between `</tool_response>` and `<|im_end|>`), NOT
/// [`format_tool_responses`] (which appends a trailing newline and is used on
/// a different, non-wire codepath); changing that helper would ripple into
/// unrelated callers.
pub fn render_tool_response_turn(content: &str) -> String {
    let mut out = String::with_capacity(48 + content.len());
    out.push_str("<|im_start|>user\n<tool_response>\n");
    out.push_str(content);
    out.push_str("\n</tool_response><|im_end|>\n");
    out
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
                 \"required\": [\"city\"]}"
                    .into(),
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
        assert_eq!(
            block.matches('\n').count(),
            2,
            "one trailing newline per tool"
        );
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
        assert_eq!(delta.text(), "hello world");
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
        assert_eq!(delta.text(), "Sure.  The weather is sunny.");
        assert_eq!(delta.tool_calls.len(), 1);
        assert_eq!(delta.tool_calls[0].name, "get_weather");
        assert_eq!(delta.tool_calls[0].arguments_json, "{\"city\": \"Paris\"}");
    }

    #[test]
    fn streaming_open_marker_split_across_chunks() {
        // "<tool" arrives in one chunk, "_call>{...}</tool_call>" in the next.
        let mut p = StreamingParser::new();
        let d1 = p.feed("Calling: <tool");
        assert_eq!(d1.text(), "Calling: ", "the <tool prefix must be held back");
        assert!(d1.tool_calls.is_empty());

        let d2 = p.feed(
            "_call>\n{\"name\": \"calc\", \"arguments\": {\"expr\": \"1+1\"}}\n</tool_call> done",
        );
        assert_eq!(d2.tool_calls.len(), 1);
        assert_eq!(d2.tool_calls[0].name, "calc");
        assert_eq!(d2.text(), " done");
    }

    #[test]
    fn streaming_close_marker_split_across_chunks() {
        let mut p = StreamingParser::new();
        let _ = p.feed(
            "<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"expr\": \"2*3\"}}\n</tool_ca",
        );
        let d = p.feed("ll> finished");
        assert_eq!(d.tool_calls.len(), 1);
        assert_eq!(d.tool_calls[0].name, "calc");
        assert_eq!(d.text(), " finished");
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
        assert_eq!(delta.text(), " mid  end");
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
        let full =
            "Hi! <tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call> Tail.";
        let mut p = StreamingParser::new();
        let mut text_acc = String::new();
        let mut calls_acc = Vec::new();
        for ch in full.chars() {
            let buf = ch.to_string();
            let d = p.feed(&buf);
            text_acc.push_str(&d.text());
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
            ToolResult {
                tool_name: "calc",
                content: "{\"value\": 5}",
            },
            ToolResult {
                tool_name: "calc",
                content: "{\"value\": 7}",
            },
        ]);
        let count = s.matches("<tool_response>").count();
        assert_eq!(count, 2);
        assert!(s.contains("\"value\": 5"));
        assert!(s.contains("\"value\": 7"));
    }

    // ---- Wire-level ChatML tool-turn rendering (shared by both surfaces) ----

    #[test]
    fn assistant_tool_call_segment_is_leading_nl_plus_render_one_call() {
        // The segment is exactly a leading '\n' followed by render_one_call —
        // the byte sequence the OpenAI surface used to inline, with the name
        // now JSON-escaped.
        let seg = render_assistant_tool_call_segment("get_weather", "{\"city\": \"Paris\"}");
        assert_eq!(
            seg,
            "\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"
        );
        // Equivalence to the underlying primitive.
        assert_eq!(
            seg,
            format!(
                "\n{}",
                Qwen35Renderer::render_one_call("get_weather", "{\"city\": \"Paris\"}")
            )
        );
    }

    #[test]
    fn tool_response_turn_matches_openai_inline_byte_layout() {
        // No newline between </tool_response> and <|im_end|>, trailing newline
        // after <|im_end|> — exactly the OpenAI surface's prior inline form.
        let turn = render_tool_response_turn("{\"temp\": 18}");
        assert_eq!(
            turn,
            "<|im_start|>user\n<tool_response>\n{\"temp\": 18}\n</tool_response><|im_end|>\n"
        );
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

    // ---- ReasoningExtractor ----
    //
    // These mirror the StreamingParser tests above: same structural cases
    // (one-shot, marker split across chunks, char-by-char recoverability)
    // plus the thinking=false passthrough that has no StreamingParser analogue.

    #[test]
    fn reasoning_disabled_is_pure_passthrough() {
        // thinking=false: everything is content, reasoning always empty, and
        // nothing is held back even when the answer text itself contains a
        // literal `</think>` (there is no reasoning block to close).
        let mut r = ReasoningExtractor::new(false);
        let d = r.feed("plain answer with a </think> literal");
        assert_eq!(d.reasoning, "");
        assert_eq!(d.content, "plain answer with a </think> literal");
        let f = r.finish();
        assert_eq!(f.reasoning, "");
        assert_eq!(f.content, "");
    }

    #[test]
    fn reasoning_enabled_splits_in_one_chunk() {
        let mut r = ReasoningExtractor::new(true);
        let d = r.feed("let me think step by step</think>The answer is 42.");
        assert_eq!(d.reasoning, "let me think step by step");
        assert_eq!(d.content, "The answer is 42.");
        // After the marker we are in content; the close marker is dropped.
        let d2 = r.feed(" More answer.");
        assert_eq!(d2.reasoning, "");
        assert_eq!(d2.content, " More answer.");
        let f = r.finish();
        assert_eq!(f.reasoning, "");
        assert_eq!(f.content, "");
    }

    #[test]
    fn reasoning_close_marker_split_across_chunks() {
        // "</thi" arrives in one chunk, "nk>answer" in the next. The partial
        // "</thi" must be held back (NOT emitted as reasoning) until resolved.
        let mut r = ReasoningExtractor::new(true);
        let d1 = r.feed("reasoning text</thi");
        assert_eq!(d1.reasoning, "reasoning text", "partial marker held back");
        assert_eq!(d1.content, "");
        let d2 = r.feed("nk>visible answer");
        assert_eq!(d2.reasoning, "");
        assert_eq!(d2.content, "visible answer");
    }

    #[test]
    fn reasoning_partial_marker_that_is_not_the_marker_is_flushed_as_reasoning() {
        // A held-back partial that turns out to be ordinary reasoning text
        // (e.g. "</thinking" — a different word) must be re-emitted to
        // reasoning, never lost. We feed "</thi" (held back) then "s" which
        // makes "</this" — not the marker — so all of it is reasoning.
        let mut r = ReasoningExtractor::new(true);
        let d1 = r.feed("note </thi");
        assert_eq!(d1.reasoning, "note ");
        let d2 = r.feed("s is reasoning");
        assert_eq!(d2.reasoning, "</this is reasoning");
        assert_eq!(d2.content, "");
        // Never closed -> finish flushes any residue (none here).
        let f = r.finish();
        assert_eq!(f.reasoning, "");
    }

    #[test]
    fn reasoning_finish_flushes_held_partial_marker_as_text() {
        // Stream ends mid-partial-marker: the held-back "</thi" never
        // completed, so finish() emits it as reasoning text.
        let mut r = ReasoningExtractor::new(true);
        let d = r.feed("thinking</thi");
        assert_eq!(d.reasoning, "thinking");
        let f = r.finish();
        assert_eq!(f.reasoning, "</thi");
        assert_eq!(f.content, "");
    }

    #[test]
    fn reasoning_no_close_marker_all_reasoning() {
        // The whole stream is reasoning (model never emitted </think>): every
        // byte is reasoning, content stays empty.
        let mut r = ReasoningExtractor::new(true);
        let d = r.feed("still thinking and thinking");
        assert_eq!(d.reasoning, "still thinking and thinking");
        assert_eq!(d.content, "");
        let f = r.finish();
        assert_eq!(f.reasoning, "");
    }

    #[test]
    fn reasoning_byte_for_byte_emission_recoverable() {
        // Feed one char at a time and verify the assembled reasoning/content
        // match a single-shot split — the same recoverability guarantee the
        // StreamingParser test asserts for tool calls.
        let full = "chain of thought here</think>final answer text";
        let mut r = ReasoningExtractor::new(true);
        let mut reasoning_acc = String::new();
        let mut content_acc = String::new();
        for ch in full.chars() {
            let buf = ch.to_string();
            let d = r.feed(&buf);
            reasoning_acc.push_str(&d.reasoning);
            content_acc.push_str(&d.content);
        }
        let f = r.finish();
        reasoning_acc.push_str(&f.reasoning);
        content_acc.push_str(&f.content);
        assert_eq!(reasoning_acc, "chain of thought here");
        assert_eq!(content_acc, "final answer text");
    }

    #[test]
    fn reasoning_empty_marker_immediately_at_start() {
        // Edge: the model emits </think> with no reasoning content at all
        // (empty trace), then the answer. reasoning is empty, content is all.
        let mut r = ReasoningExtractor::new(true);
        let d = r.feed("</think>direct answer");
        assert_eq!(d.reasoning, "");
        assert_eq!(d.content, "direct answer");
    }

    // ---- Native `<function=..><parameter=..>` protocol (Qwen3.5) ----

    fn schedule_tool() -> ToolSchema {
        ToolSchema {
            name: "schedule_event".into(),
            description: "Create a calendar event.".into(),
            parameters_json_schema: r#"{"type":"object","properties":{
                "title":{"type":"string"},
                "date":{"type":"string"},
                "time":{"type":"object"},
                "attendees":{"type":"array"},
                "count":{"type":"integer"},
                "all_day":{"type":"boolean"}
            }}"#
            .into(),
        }
    }

    /// Build a native-protocol emission for `name` with ordered `(param, value)`
    /// pairs, exactly as the Qwen3.5 template teaches the model to emit it.
    fn native_call(name: &str, params: &[(&str, &str)]) -> String {
        let mut s = String::from("<tool_call>\n<function=");
        s.push_str(name);
        s.push_str(">\n");
        for (p, v) in params {
            s.push_str("<parameter=");
            s.push_str(p);
            s.push_str(">\n");
            s.push_str(v);
            s.push_str("\n</parameter>\n");
        }
        s.push_str("</function>\n</tool_call>");
        s
    }

    #[test]
    fn native_scalar_values_typed_by_schema() {
        // string stays string (even numeric-looking date), number/boolean typed.
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let emission = native_call(
            "schedule_event",
            &[
                ("title", "Team Sync"),
                ("date", "2026-08-01"),
                ("count", "3"),
                ("all_day", "true"),
            ],
        );
        let parsed = parse_final_with_schemas(&emission, schemas);
        assert_eq!(parsed.content, "");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "schedule_event");
        assert_eq!(
            parsed.tool_calls[0].arguments_json,
            r#"{"title":"Team Sync","date":"2026-08-01","count":3,"all_day":true}"#
        );
    }

    #[test]
    fn native_string_type_keeps_numeric_looking_value_a_string() {
        // A `string`-typed parameter whose value looks like a number must stay a
        // JSON string — the whole reason the parser is schema-aware.
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let emission = native_call("schedule_event", &[("date", "2026")]);
        let parsed = parse_final_with_schemas(&emission, schemas);
        assert_eq!(parsed.tool_calls[0].arguments_json, r#"{"date":"2026"}"#);
    }

    /// Concatenate the streaming parser's `tool_stream` argument deltas when the
    /// full `<tool_call>…</tool_call>` emission is fed as the given chunks.
    fn stream_chunks(chunks: &[&str], schemas: Arc<ToolSchemas>) -> String {
        let mut p = StreamingParser::with_schemas(schemas);
        let mut evs = Vec::new();
        for c in chunks {
            evs.extend(p.feed(c).tool_stream());
        }
        let _ = p.finish();
        let mut out = String::new();
        let mut inside = false;
        for e in evs {
            match e {
                ToolStreamEvent::Start { .. } => inside = true,
                ToolStreamEvent::ArgJsonDelta { partial_json } if inside => {
                    out.push_str(&partial_json)
                }
                ToolStreamEvent::ArgJsonDelta { .. } => {}
                ToolStreamEvent::End => inside = false,
            }
        }
        out
    }

    /// Split `s` into random-length (1..=5) char-boundary chunks, driven by a
    /// seeded LCG so any failure reproduces.
    fn random_char_chunks(s: &str, rng: &mut u64) -> Vec<String> {
        let chars: Vec<char> = s.chars().collect();
        let mut chunks = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let len = 1 + ((*rng >> 33) as usize % 5);
            let end = (i + len).min(chars.len());
            chunks.push(chars[i..end].iter().collect());
            i = end;
        }
        chunks
    }

    /// The streaming path is DEFINED as "byte-identical to the buffered parse, just
    /// incremental": the concatenated `tool_stream` argument deltas must equal the
    /// buffered `arguments_json` for the same call — fed whole-text, char-by-char,
    /// at EVERY two-way split point, and across 200 seeded-random chunkings — over
    /// an adversarial corpus (escaping, CRLF/tab, multibyte, a `</parameter`
    /// substring, empty values, typed scalars + composites). Char-boundary chunking
    /// at any split point is the true worst case: `SseEmitter::push` buffers partial
    /// UTF-8 in `pending_bytes` and feeds the parser only complete codepoints via
    /// `drain_complete_utf8`, so the parser never receives a split multi-byte char.
    #[test]
    fn streaming_tool_args_byte_identical_to_buffered() {
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let cases: Vec<Vec<(&str, &str)>> = vec![
            vec![("title", "Team Sync")],
            vec![("title", "")],
            vec![("title", "quote \" and backslash \\ end")],
            vec![("title", "carriage\r\nreturn\tand tab")],
            vec![("title", "emoji 😀 and CJK 日本語 end")],
            vec![("title", "has a </parameter but no close here")],
            vec![("title", "multi\nline\nvalue\n")],
            vec![("date", "2026-08-01"), ("count", "3"), ("all_day", "true")],
            vec![
                ("time", r#"{"start":"14:00","end":"15:00"}"#),
                ("attendees", r#"["Omar","Layla"]"#),
            ],
            vec![
                ("title", "Sync"),
                ("date", "2026"),
                ("count", "7"),
                ("all_day", "False"),
            ],
        ];
        for (i, params) in cases.iter().enumerate() {
            let emission = native_call("schedule_event", params);
            let buffered = parse_final_with_schemas(&emission, schemas.clone()).tool_calls[0]
                .arguments_json
                .clone();
            let chars: Vec<char> = emission.chars().collect();

            assert_eq!(
                stream_chunks(&[&emission], schemas.clone()),
                buffered,
                "case {i} whole"
            );

            let singles: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
            let refs: Vec<&str> = singles.iter().map(String::as_str).collect();
            assert_eq!(
                stream_chunks(&refs, schemas.clone()),
                buffered,
                "case {i} char"
            );

            for sp in 1..chars.len() {
                let a: String = chars[..sp].iter().collect();
                let b: String = chars[sp..].iter().collect();
                assert_eq!(
                    stream_chunks(&[&a, &b], schemas.clone()),
                    buffered,
                    "case {i} 2-split@{sp}"
                );
            }

            let mut rng = 0x5eed_u64 ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for _ in 0..200 {
                let parts = random_char_chunks(&emission, &mut rng);
                let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
                assert_eq!(
                    stream_chunks(&refs, schemas.clone()),
                    buffered,
                    "case {i} random"
                );
            }
        }
    }

    /// A legacy JSON body carries no incremental native events, so the parser
    /// surfaces it to `tool_stream` as exactly one atomic triple — Start, the
    /// full arguments, End — whose payload is byte-identical to the finalized
    /// call. This is what lets the wire layer stream BOTH protocols from
    /// `tool_stream` alone (never falling back to the buffered `tool_calls`).
    #[test]
    fn legacy_json_call_streams_as_one_atomic_triple() {
        let emission = "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n</tool_call>";
        let mut p = StreamingParser::new();
        let delta = p.feed(emission);
        let _ = p.finish();
        assert_eq!(delta.tool_calls.len(), 1, "one legacy call finalized");
        let args = delta.tool_calls[0].arguments_json.clone();
        assert_eq!(
            delta.tool_stream(),
            vec![
                ToolStreamEvent::Start { name: "f".into() },
                ToolStreamEvent::ArgJsonDelta { partial_json: args },
                ToolStreamEvent::End,
            ],
            "legacy body must stream as one atomic Start+args+End"
        );
    }

    /// A legacy JSON body whose argument value contains the literal native marker
    /// `<function=g>` must still stream as the real call (keyed on the body's
    /// leading `{`, like `parse_call_body`), never as a phantom `g` call: the
    /// streamer stays inert on legacy bodies and the atomic triple carries `f`.
    #[test]
    fn legacy_body_with_embedded_function_marker_streams_the_real_call() {
        let emission = "<tool_call>\n{\"name\": \"f\", \"arguments\": \
                        {\"x\": \"<function=g></function>\"}}\n</tool_call>";
        // Fed whole, and split so the leading `{` lands in the second chunk.
        for chunks in [vec![emission], vec!["<tool_call>\n", &emission[12..]]] {
            let mut p = StreamingParser::new();
            let mut calls = Vec::new();
            let mut stream = Vec::new();
            for c in &chunks {
                let d = p.feed(c);
                stream.extend(d.tool_stream());
                calls.extend(d.tool_calls);
            }
            let _ = p.finish();
            assert_eq!(calls.len(), 1, "exactly one call");
            assert_eq!(calls[0].name, "f", "the real call, not the embedded marker");
            assert_eq!(
                stream,
                vec![
                    ToolStreamEvent::Start { name: "f".into() },
                    ToolStreamEvent::ArgJsonDelta {
                        partial_json: calls[0].arguments_json.clone(),
                    },
                    ToolStreamEvent::End,
                ],
                "no phantom `g`"
            );
        }
    }

    /// A value whose raw text contains `</parameter></function><parameter=…>` is
    /// truncated at the first `</parameter>` on BOTH paths; the streamer then scans
    /// PAST the `</function>` to the next parameter, exactly as the buffered parser
    /// does (it never treats `</function>` as a terminator). Streamed args are
    /// byte-identical to the buffered parse, and the later parameter is captured.
    #[test]
    fn embedded_function_close_does_not_terminate_param_scan() {
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let emission = "<tool_call>\n<function=schedule_event>\n<parameter=title>\n\
                        A</parameter></function><parameter=date>\nB\n</parameter>\n\
                        </function>\n</tool_call>";
        let buffered = parse_final_with_schemas(emission, schemas.clone()).tool_calls[0]
            .arguments_json
            .clone();
        assert_eq!(stream_chunks(&[emission], schemas.clone()), buffered);
        assert!(
            buffered.contains("title"),
            "both params captured: {buffered}"
        );
        assert!(
            buffered.contains("date"),
            "scan continued past </function>: {buffered}"
        );
    }

    /// A native call cut off mid-value (no outer `</tool_call>`) streams its Start
    /// and partial args but NO End, and `finish` reports the incomplete body — so
    /// the wire layer sees a still-open call (→ max_tokens), never a clean close.
    #[test]
    fn native_truncation_mid_value_emits_no_end() {
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let mut p = StreamingParser::with_schemas(schemas);
        let delta = p.feed("<tool_call>\n<function=schedule_event>\n<parameter=title>\nTeam");
        let fin = p.finish();
        assert!(
            matches!(
                delta.tool_stream().first(),
                Some(ToolStreamEvent::Start { .. })
            ),
            "the call opened"
        );
        assert!(
            !delta.tool_stream().contains(&ToolStreamEvent::End),
            "a cut-off call emits no End"
        );
        assert!(
            fin.incomplete_tool_call.is_some(),
            "finish reports the incomplete body"
        );
    }

    /// Duplicate parameter names are the one shape where streamed bytes differ
    /// from the buffered parse: the buffered parser keeps the last value (map
    /// insert) while the stream emits each occurrence. Both are VALID JSON that
    /// `serde_json` (and the standard last-wins semantics the server and typical
    /// clients use) resolve to the SAME object, and the trained template never
    /// emits duplicate names. This locks that equivalence under serde_json; a
    /// receiver that preserves duplicate keys (e.g. an object-pairs hook) is out of
    /// scope because the model never produces this shape.
    #[test]
    fn duplicate_param_names_stream_valid_json_with_same_object() {
        let emission = "<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n\
                        <parameter=x>\n2\n</parameter>\n</function>\n</tool_call>";
        let mut p = StreamingParser::new();
        let delta = p.feed(emission);
        let _ = p.finish();
        let buffered = &delta.tool_calls[0].arguments_json;
        let streamed: String = delta
            .tool_stream()
            .iter()
            .filter_map(|e| match e {
                ToolStreamEvent::ArgJsonDelta { partial_json } => Some(partial_json.as_str()),
                _ => None,
            })
            .collect();
        let bv: JsonValue = serde_json::from_str(buffered).unwrap();
        let sv: JsonValue = serde_json::from_str(&streamed).unwrap();
        assert_eq!(bv, sv, "streamed={streamed} buffered={buffered}");
    }

    #[test]
    fn native_nested_object_and_array_json_parsed() {
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let emission = native_call(
            "schedule_event",
            &[
                ("time", r#"{"start": "14:00", "end": "15:00"}"#),
                ("attendees", r#"["Omar", "Layla"]"#),
            ],
        );
        let parsed = parse_final_with_schemas(&emission, schemas);
        // Nested composites are reparsed to real JSON (compact re-serialization).
        assert_eq!(
            parsed.tool_calls[0].arguments_json,
            r#"{"time":{"start":"14:00","end":"15:00"},"attendees":["Omar","Layla"]}"#
        );
    }

    #[test]
    fn native_multiline_value_interior_newlines_preserved() {
        let schemas = Arc::new(ToolSchemas::from_tools(&[ToolSchema {
            name: "send_message".into(),
            description: "send".into(),
            parameters_json_schema: r#"{"type":"object","properties":{"body":{"type":"string"}}}"#
                .into(),
        }]));
        let emission = native_call("send_message", &[("body", "line one\nline two")]);
        let parsed = parse_final_with_schemas(&emission, schemas);
        assert_eq!(
            parsed.tool_calls[0].arguments_json,
            "{\"body\":\"line one\\nline two\"}"
        );
    }

    #[test]
    fn native_schemaless_heuristic_types_composites_and_scalars() {
        // With no schema: `{...}`/`42`/`true` become typed, a bare word stays a
        // string. (The CLI path is schemaless; the server path is schema-aware.)
        let emission = native_call(
            "f",
            &[
                ("obj", "{\"a\": 1}"),
                ("n", "42"),
                ("b", "true"),
                ("s", "hello"),
            ],
        );
        let parsed = parse_final(&emission);
        assert_eq!(
            parsed.tool_calls[0].arguments_json,
            r#"{"obj":{"a":1},"n":42,"b":true,"s":"hello"}"#
        );
    }

    #[test]
    fn native_two_consecutive_calls() {
        let schemas = Arc::new(ToolSchemas::from_tools(&[ToolSchema {
            name: "get_weather".into(),
            description: "w".into(),
            parameters_json_schema: r#"{"type":"object","properties":{"city":{"type":"string"}}}"#
                .into(),
        }]));
        let a = native_call("get_weather", &[("city", "Riyadh")]);
        let b = native_call("get_weather", &[("city", "Jeddah")]);
        let emission = format!("{a}\n{b}");
        let parsed = parse_final_with_schemas(&emission, schemas);
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].arguments_json, r#"{"city":"Riyadh"}"#);
        assert_eq!(parsed.tool_calls[1].arguments_json, r#"{"city":"Jeddah"}"#);
    }

    #[test]
    fn native_streaming_equals_batch_char_by_char() {
        // The §2D stream_eq_nonstream guarantee: feeding the native emission one
        // char at a time reconstructs the SAME call as a single-shot batch parse.
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let emission = format!(
            "Sure, scheduling now. {} Done.",
            native_call(
                "schedule_event",
                &[
                    ("title", "Sync"),
                    ("count", "2"),
                    ("time", r#"{"start": "09:00"}"#)
                ],
            )
        );

        let batch = parse_final_with_schemas(&emission, schemas.clone());

        let mut p = StreamingParser::with_schemas(schemas);
        let mut text = String::new();
        let mut calls = Vec::new();
        for ch in emission.chars() {
            let d = p.feed(&ch.to_string());
            text.push_str(&d.text());
            calls.extend(d.tool_calls);
        }
        let fin = p.finish();
        text.push_str(&fin.flushed_text);

        assert_eq!(calls, batch.tool_calls, "streaming calls must equal batch");
        assert_eq!(text, batch.content, "streaming text must equal batch");
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].arguments_json,
            r#"{"title":"Sync","count":2,"time":{"start":"09:00"}}"#
        );
    }

    #[test]
    fn native_classification_latches_across_a_non_brace_preamble_chunk() {
        // Regression: a body fed as a non-`{` preamble chunk and then a `{` chunk
        // must stay NATIVE and stream its call. The native-vs-legacy decision is
        // latched at the body's first non-whitespace byte, so chunk 2's leading `{`
        // cannot reclassify the body as legacy and drop the call from the stream
        // (the buffered parse, which dispatches on the whole trimmed body, keeps it).
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let full = "<tool_call>\nnote{<function=schedule_event>\n\
                    <parameter=title>\nHi\n</parameter>\n</function>\n</tool_call>";
        let buffered = parse_final_with_schemas(full, schemas.clone()).tool_calls[0]
            .arguments_json
            .clone();
        let streamed = stream_chunks(
            &[
                "<tool_call>\nnote",
                "{<function=schedule_event>\n<parameter=title>\n\
                 Hi\n</parameter>\n</function>\n</tool_call>",
            ],
            schemas,
        );
        // Before latching this was "" (the call was dropped from the stream).
        assert_eq!(
            streamed, buffered,
            "latched native body must stream the call"
        );
        assert_eq!(buffered, r#"{"title":"Hi"}"#);
    }

    #[test]
    fn large_typed_array_streams_byte_identical_across_small_chunks() {
        // A long typed (array) value is buffered to its close and coerced whole; it
        // must accumulate across many feeds into the SAME JSON as the buffered parse.
        // The accumulator appends into a dedicated buffer and holds only a bounded
        // `</parameter>`-prefix suffix, so this stays O(n) rather than re-copying the
        // whole value every feed — char-by-char here drives one feed per character.
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let arr = format!(
            "[{}]",
            (0..200)
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let emission = native_call("schedule_event", &[("attendees", &arr)]);
        let buffered = parse_final_with_schemas(&emission, schemas.clone()).tool_calls[0]
            .arguments_json
            .clone();
        let chars: Vec<String> = emission.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
        let streamed = stream_chunks(&refs, schemas);
        assert_eq!(streamed, buffered, "typed array must stream byte-identical");
        assert!(buffered.starts_with(r#"{"attendees":[0,1,2,"#));
    }

    #[test]
    fn whitespace_prefix_before_function_streams_the_call() {
        // Insignificant leading whitespace inside `<tool_call>` before `<function=`
        // is DISCARDED each feed (not re-held, which would be O(n^2) for a long run);
        // the call must still stream byte-identically to the buffered parse. The
        // whitespace is split into its own chunks to drive the discard path repeatedly.
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let full = "<tool_call>\n   \n  \t <function=schedule_event>\n\
                    <parameter=title>\nHi\n</parameter>\n</function>\n</tool_call>";
        let buffered = parse_final_with_schemas(full, schemas.clone()).tool_calls[0]
            .arguments_json
            .clone();
        let streamed = stream_chunks(
            &[
                "<tool_call>\n",
                "   ",
                "  \t ",
                "<function=schedule_event>\n<parameter=title>\n\
                 Hi\n</parameter>\n</function>\n</tool_call>",
            ],
            schemas,
        );
        assert_eq!(
            streamed, buffered,
            "whitespace prefix must not change the call"
        );
        assert_eq!(buffered, r#"{"title":"Hi"}"#);
    }

    #[test]
    fn native_negative_plain_answer_is_not_a_tool_call() {
        // The negative case (§2D case 10): a plain answer with no `<tool_call>`
        // marker yields zero calls and verbatim content.
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let parsed = parse_final_with_schemas("hello", schemas);
        assert_eq!(parsed.content, "hello");
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn native_boolean_python_capitalized_coerces_to_bool() {
        // ADVERSARIAL: the embedded template renders a boolean argument with
        // Python `str()` (`True`/`False`), so a model trained on it emits the
        // CAPITALIZED form. A schema-typed `boolean` MUST coerce both `True`
        // and `False` to real JSON booleans, else `args_schema_valid` (§2D)
        // rejects a correct call. Lowercase `true`/`false` must also still work.
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let cap = native_call("schedule_event", &[("title", "Sync"), ("all_day", "True")]);
        let p = parse_final_with_schemas(&cap, schemas.clone());
        assert_eq!(
            p.tool_calls[0].arguments_json,
            r#"{"title":"Sync","all_day":true}"#
        );

        let capf = native_call("schedule_event", &[("all_day", "False")]);
        let pf = parse_final_with_schemas(&capf, schemas.clone());
        assert_eq!(pf.tool_calls[0].arguments_json, r#"{"all_day":false}"#);

        let low = native_call("schedule_event", &[("all_day", "false")]);
        let pl = parse_final_with_schemas(&low, schemas);
        assert_eq!(pl.tool_calls[0].arguments_json, r#"{"all_day":false}"#);
    }

    #[test]
    fn legacy_json_body_still_parses_backward_compat() {
        // §1 step 5: the OLD Qwen2.5-era `<tool_call>{"name","arguments"}` JSON
        // body MUST keep parsing so existing callers / the server round-trip
        // re-render / §2G legacy paths are not broken.
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let legacy = "<tool_call>\n{\"name\": \"schedule_event\", \"arguments\": {\"title\": \"Sync\", \"count\": 3}}\n</tool_call>";
        let p = parse_final_with_schemas(legacy, schemas);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].name, "schedule_event");
        assert_eq!(
            p.tool_calls[0].arguments_json,
            "{\"title\": \"Sync\", \"count\": 3}"
        );
    }

    #[test]
    fn native_marker_split_across_stream_chunks_reconstructs() {
        // ADVERSARIAL streaming: the `<tool_call>` open and `</tool_call>` close
        // markers straddle chunk boundaries mid-`<function>` block; the parser
        // must still reconstruct exactly one schema-typed call.
        let schemas = Arc::new(ToolSchemas::from_tools(&[schedule_tool()]));
        let mut p = StreamingParser::with_schemas(schemas);
        let mut calls = Vec::new();
        for chunk in [
            "Working on it <too",
            "l_call>\n<function=schedule_e",
            "vent>\n<parameter=title>\nSt",
            "andup\n</parameter>\n<parameter=count>\n5\n</para",
            "meter>\n</function>\n</tool_",
            "call> done",
        ] {
            calls.extend(p.feed(chunk).tool_calls);
        }
        let _ = p.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "schedule_event");
        assert_eq!(calls[0].arguments_json, r#"{"title":"Standup","count":5}"#);
    }

    #[test]
    fn nullable_string_param_coerces_numericlike_to_string() {
        // JSON-Schema union `["string","null"]` (a nullable string — common in
        // real tool/MCP schemas) must be treated as `string`, so a bare `35`
        // stays the string "35", not the JSON number 35 (the schema-aware
        // guarantee; before the union-type fix this fell to the heuristic).
        let tool = ToolSchema {
            name: "f".into(),
            description: String::new(),
            parameters_json_schema:
                r#"{"type":"object","properties":{"code":{"type":["string","null"]}}}"#.into(),
        };
        let schemas = ToolSchemas::from_tools(&[tool]);
        let call = parse_native_call_body(
            "<function=f>\n<parameter=code>\n35\n</parameter>\n</function>",
            Some(&schemas),
        )
        .unwrap();
        assert_eq!(call.name, "f");
        assert_eq!(call.arguments_json, r#"{"code":"35"}"#);
    }

    #[test]
    fn nullable_integer_param_coerces_to_number() {
        // `["null","integer"]` resolves to `integer`, so the value is a number.
        let tool = ToolSchema {
            name: "f".into(),
            description: String::new(),
            parameters_json_schema:
                r#"{"type":"object","properties":{"n":{"type":["null","integer"]}}}"#.into(),
        };
        let schemas = ToolSchemas::from_tools(&[tool]);
        let call = parse_native_call_body(
            "<function=f>\n<parameter=n>\n35\n</parameter>\n</function>",
            Some(&schemas),
        )
        .unwrap();
        assert_eq!(call.arguments_json, r#"{"n":35}"#);
    }
}
