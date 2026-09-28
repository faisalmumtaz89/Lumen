//! SSE "safe emit" cursor.
//!
//! The decoder yields a [`crate::engine::TokenEvent::Token`] every time the
//! model produces a token. The decoded `delta_text` is opaque text from the
//! tokenizer; depending on the BPE configuration it may end on a partial
//! UTF-8 codepoint (e.g. half of a CJK character), and it may contain the
//! start of a `<tool_call>` marker that the model is about to finish on
//! the next token.
//!
//! [`SseSafeEmitter`] sits between the worker stream and the wire
//! encoders. It returns text only when:
//!
//! 1. The buffered bytes end on a valid UTF-8 boundary.
//! 2. The tool-call streaming parser has decided the buffered text is
//!    user-visible (not held back as a possible marker prefix).
//!
//! In return, it surfaces structured tool-call deltas as they finalize.

use std::sync::Arc;

use lumen_runtime::tooling::{
    ParsedToolCall, ReasoningExtractor, StreamingFinish, StreamingParser, ToolSchemas,
};

/// The output of a single [`SseSafeEmitter::push`] call.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct EmitDelta {
    /// Reasoning-trace text safe to forward right now. Non-empty only while
    /// the request enabled thinking AND the model is still inside the
    /// `<think>...</think>` block. The caller routes this to
    /// `reasoning_content` (OpenAI), a `thinking` content block (Anthropic),
    /// or a labelled CLI section — NEVER into the user-visible answer.
    pub reasoning: String,

    /// Plain user-visible text safe to forward right now.
    pub text: String,

    /// Tool calls that finalized in this push.
    pub tool_calls: Vec<ParsedToolCall>,
}

/// The tool calls a reply may carry. `schemas` types native `<parameter>`
/// values by the request's tools; the request's tool choice decides which
/// calls count.
#[derive(Debug, Clone)]
pub struct ReplyTools {
    pub schemas: Arc<ToolSchemas>,
    /// Whether tool calls are parsed out of the reply at all. Not under
    /// `tool_choice: none`, where tool-call markup stays plain text.
    pub parsed: bool,
    /// Under a named tool choice, the only tool a call may name; calls to any
    /// other tool are dropped.
    pub only: Option<String>,
    /// At most one tool call (OpenAI `parallel_tool_calls: false`, Anthropic
    /// `disable_parallel_tool_use: true`); calls after the first are dropped.
    pub single: bool,
}

impl Default for ReplyTools {
    fn default() -> Self {
        Self {
            schemas: Arc::default(),
            parsed: true,
            only: None,
            single: false,
        }
    }
}

/// Parse tool calls out of answer text, keeping the calls the reply may carry;
/// with parsing off the text passes through unchanged.
fn parse_answer(
    parser: &mut StreamingParser,
    tools: &ReplyTools,
    calls: &mut usize,
    answer: &str,
) -> (String, Vec<ParsedToolCall>) {
    if !tools.parsed {
        return (answer.to_string(), Vec::new());
    }
    let mut parsed = parser.feed(answer);
    if let Some(only) = &tools.only {
        parsed.tool_calls.retain(|c| c.name == *only);
    }
    if tools.single {
        // Room for one call over the whole reply.
        parsed.tool_calls.truncate(1usize.saturating_sub(*calls));
    }
    *calls += parsed.tool_calls.len();
    (parsed.text, parsed.tool_calls)
}

/// Buffers decoded token fragments until they are safe to emit on the wire.
pub struct SseSafeEmitter {
    /// Bytes that arrived but didn't terminate on a UTF-8 boundary.
    pending_bytes: Vec<u8>,
    /// Reasoning/answer splitter. Runs FIRST (outermost): it strips the
    /// `<think>...</think>` trace off the front so only answer text reaches
    /// the tool-call parser. A passthrough (no held-back bytes, no effect)
    /// when the request did not enable thinking.
    reasoning: ReasoningExtractor,
    /// Tool-call streaming parser. Sees only post-reasoning answer text.
    parser: StreamingParser,
    /// Which parsed tool calls the reply keeps.
    tools: ReplyTools,
    /// Tool calls kept so far.
    calls: usize,
}

impl Default for SseSafeEmitter {
    fn default() -> Self {
        // Default = thinking disabled: the reasoning extractor is a pure
        // passthrough, so the emitter is byte-identical to the pre-reasoning
        // behaviour. `SseSafeEmitter::new(thinking)` is the production path.
        Self {
            pending_bytes: Vec::new(),
            reasoning: ReasoningExtractor::new(false),
            parser: StreamingParser::new(),
            tools: ReplyTools::default(),
            calls: 0,
        }
    }
}

impl SseSafeEmitter {
    /// Construct an emitter for a request whose reasoning flag is `thinking`
    /// (resolved via `runtime_defaults::resolve_enable_thinking`). When
    /// `thinking == false` the reasoning stage is an exact passthrough and the
    /// emitter behaves byte-for-byte like the historical tool-call-only one.
    pub fn new(thinking: bool) -> Self {
        Self {
            pending_bytes: Vec::new(),
            reasoning: ReasoningExtractor::new(thinking),
            parser: StreamingParser::new(),
            tools: ReplyTools::default(),
            calls: 0,
        }
    }

    /// Construct an emitter for a request's [`ReplyTools`]: its tool-call
    /// parser types native-protocol `<parameter>` values by the request's
    /// schemas, and only the calls its tool choice allows are kept. The chat
    /// and Anthropic surfaces use this so the streaming reconstruction is
    /// schema-aware (and byte-identical to the aggregated non-streaming result,
    /// which flows through the same emitter). Legacy `/v1/completions` and the
    /// unit tests keep the schemaless [`new`](Self::new).
    pub fn with_tools(thinking: bool, tools: ReplyTools) -> Self {
        Self {
            pending_bytes: Vec::new(),
            reasoning: ReasoningExtractor::new(thinking),
            parser: StreamingParser::with_schemas(tools.schemas.clone()),
            tools,
            calls: 0,
        }
    }

    /// Run a slice of decode-safe answer/reasoning text through the reasoning
    /// splitter and then the tool-call parser, folding the result into an
    /// [`EmitDelta`]. Reasoning text bypasses the tool parser entirely
    /// (reasoning never contains tool calls); only answer content is parsed.
    fn process_safe_text(&mut self, safe_text: &str) -> EmitDelta {
        let split = self.reasoning.feed(safe_text);
        let (text, tool_calls) = parse_answer(
            &mut self.parser,
            &self.tools,
            &mut self.calls,
            &split.content,
        );
        EmitDelta {
            reasoning: split.reasoning,
            text,
            tool_calls,
        }
    }

    /// Push the next decoded fragment. Returns the emit-safe text and any
    /// finalized tool calls. The fragment is text-typed because the
    /// tokenizer already produced UTF-8 bytes; this method is conservative
    /// about partial-codepoint cases for tokenizers whose
    /// `decode_incremental` returns mid-codepoint slices.
    pub fn push(&mut self, fragment: &str) -> EmitDelta {
        self.pending_bytes.extend_from_slice(fragment.as_bytes());
        let safe_text = self.drain_complete_utf8();
        if safe_text.is_empty() {
            return EmitDelta::default();
        }
        self.process_safe_text(&safe_text)
    }

    /// Flush the emitter at end-of-stream. Returns whatever held-back text
    /// the reasoning splitter / tool parser were sitting on (now guaranteed
    /// safe to emit), plus a flag indicating whether a tool-call body was
    /// incomplete.
    pub fn finish(mut self) -> (EmitDelta, Option<String>) {
        // First, force-flush any partial UTF-8 bytes -- at this point we
        // know no more bytes are coming, so the only safe thing is to
        // append U+FFFD for incomplete sequences.
        let trailing_bytes = std::mem::take(&mut self.pending_bytes);
        let trailing = String::from_utf8_lossy(&trailing_bytes).into_owned();

        // Push the trailing bytes through the reasoning splitter, then flush
        // it: any held-back partial `</think>` that never completed is real
        // reasoning text. All answer `content` (from both the final feed and
        // the flush) goes on through the tool parser before it, too, is
        // finished, so a tool-call marker straddling the very end is recovered.
        let split = self.reasoning.feed(&trailing);
        let reasoning_fin = self.reasoning.finish();
        // `reasoning_fin.content` is empty by the extractor's contract (it
        // only ever holds back while still inside reasoning, and that residue
        // is reasoning text); concatenate defensively so the flow is total.
        let mut answer_tail = String::new();
        answer_tail.push_str(&split.content);
        answer_tail.push_str(&reasoning_fin.content);
        let (text, tool_calls) =
            parse_answer(&mut self.parser, &self.tools, &mut self.calls, &answer_tail);
        let fin: StreamingFinish = if self.tools.parsed {
            self.parser.finish()
        } else {
            StreamingFinish::default()
        };

        let mut delta = EmitDelta::default();
        delta.reasoning.push_str(&split.reasoning);
        delta.reasoning.push_str(&reasoning_fin.reasoning);
        delta.text.push_str(&text);
        delta.text.push_str(&fin.flushed_text);
        delta.tool_calls.extend(tool_calls);

        (delta, fin.incomplete_tool_call)
    }

    /// Test-only entry point that lets the buffer-boundary tests inject
    /// raw bytes (potentially a partial UTF-8 codepoint) without going
    /// through `&str`. Behavior is otherwise identical to `push`.
    #[cfg(test)]
    pub(crate) fn push_raw_bytes_for_test(&mut self, bytes: &[u8]) -> EmitDelta {
        self.pending_bytes.extend_from_slice(bytes);
        let safe_text = self.drain_complete_utf8();
        if safe_text.is_empty() {
            return EmitDelta::default();
        }
        self.process_safe_text(&safe_text)
    }

    /// Drain the longest valid UTF-8 prefix from `pending_bytes` and
    /// return it as an owned string. Bytes that form a partial codepoint
    /// are left in the buffer for the next push.
    fn drain_complete_utf8(&mut self) -> String {
        if self.pending_bytes.is_empty() {
            return String::new();
        }
        match std::str::from_utf8(&self.pending_bytes) {
            Ok(_) => {
                // Whole buffer is valid UTF-8; drain it.
                let bytes = std::mem::take(&mut self.pending_bytes);
                String::from_utf8(bytes).unwrap_or_default()
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid == 0 {
                    return String::new();
                }
                let head: Vec<u8> = self.pending_bytes.drain(..valid).collect();
                String::from_utf8(head).unwrap_or_default()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: every test below that builds `SseSafeEmitter::new(false)` is
    // simultaneously the byte-identity guard for the thinking-OFF default:
    // with the reasoning extractor in passthrough, `reasoning` is always
    // empty and `text`/`tool_calls` match the pre-reasoning-control behaviour.

    #[test]
    fn passes_through_plain_ascii_in_one_call() {
        let mut e = SseSafeEmitter::new(false);
        let d = e.push("hello world");
        assert_eq!(d.text, "hello world");
        assert_eq!(d.reasoning, "", "thinking-off must never produce reasoning");
        assert!(d.tool_calls.is_empty());
    }

    #[test]
    fn buffers_partial_utf8_until_codepoint_completes() {
        // 0xE6 0x97 0xA5 = 日 (U+65E5). Split a fully-valid string and push
        // the halves separately so we exercise the byte-boundary buffering
        // logic without ever constructing an invalid `&str` (UB-clean).
        let mut e = SseSafeEmitter::new(false);
        let chars = "日"; // 3 bytes
        let bytes = chars.as_bytes();
        // Push first 2 bytes by routing through pending_bytes directly.
        e.push_raw_bytes_for_test(&bytes[..2]);
        let d1 = e.push_raw_bytes_for_test(&[]);
        assert_eq!(d1.text, "", "should hold partial codepoint");
        let d2 = e.push_raw_bytes_for_test(&bytes[2..]);
        assert_eq!(d2.text, "日");
    }

    #[test]
    fn holds_back_partial_tool_call_marker() {
        let mut e = SseSafeEmitter::new(false);
        let d = e.push("Calling <tool");
        assert_eq!(d.text, "Calling ", "must hold the marker prefix");
        let d2 = e.push("_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call> end");
        assert_eq!(d2.text, " end");
        assert_eq!(d2.tool_calls.len(), 1);
        assert_eq!(d2.tool_calls[0].name, "f");
    }

    #[test]
    fn finish_flushes_held_text() {
        let mut e = SseSafeEmitter::new(false);
        let d_push = e.push("partial <to");
        // "partial " is safe, "<to" is held back.
        assert_eq!(d_push.text, "partial ");
        let (d_finish, incomplete) = e.finish();
        // The flush emits the held-back fragment that turned out NOT to be
        // a tool-call marker.
        assert_eq!(d_finish.text, "<to");
        assert!(incomplete.is_none());
    }

    #[test]
    fn finish_reports_incomplete_tool_call() {
        let mut e = SseSafeEmitter::new(false);
        let _ = e.push("<tool_call>\n{\"name\": \"x\"");
        let (_d, incomplete) = e.finish();
        assert!(incomplete.is_some());
    }

    #[test]
    fn default_emitter_is_thinking_off() {
        // The Default impl must equal new(false): the reasoning extractor is a
        // passthrough so a `<think>` boundary in the stream is NOT honoured.
        let mut e = SseSafeEmitter::default();
        let d = e.push("text</think>more");
        assert_eq!(d.reasoning, "");
        assert_eq!(d.text, "text</think>more");
    }

    // ---- Reasoning extraction (thinking ON) ----

    #[test]
    fn thinking_on_splits_reasoning_then_content() {
        let mut e = SseSafeEmitter::new(true);
        let d = e.push("let me reason</think>The answer.");
        assert_eq!(d.reasoning, "let me reason");
        assert_eq!(d.text, "The answer.");
        assert!(d.tool_calls.is_empty());
    }

    #[test]
    fn thinking_on_reasoning_then_content_with_tool_call() {
        // Reasoning is outermost: the trace is stripped first, then the
        // post-</think> answer is parsed for tool calls.
        let mut e = SseSafeEmitter::new(true);
        let d = e.push("thinking about weather</think>Sure. ");
        assert_eq!(d.reasoning, "thinking about weather");
        assert_eq!(d.text, "Sure. ");
        let d2 = e.push("<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call> done");
        assert_eq!(d2.reasoning, "");
        assert_eq!(d2.text, " done");
        assert_eq!(d2.tool_calls.len(), 1);
        assert_eq!(d2.tool_calls[0].name, "f");
    }

    #[test]
    fn thinking_on_close_marker_split_across_pushes() {
        let mut e = SseSafeEmitter::new(true);
        let d1 = e.push("reasoning</thi");
        assert_eq!(d1.reasoning, "reasoning", "partial </think> held back");
        assert_eq!(d1.text, "");
        let d2 = e.push("nk>answer");
        assert_eq!(d2.reasoning, "");
        assert_eq!(d2.text, "answer");
    }

    #[test]
    fn thinking_on_unclosed_reasoning_flushed_on_finish() {
        // Model never emits </think>: all text is reasoning, finish flushes
        // the held-back partial marker as reasoning, content stays empty.
        let mut e = SseSafeEmitter::new(true);
        let d = e.push("still reasoning</thi");
        assert_eq!(d.reasoning, "still reasoning");
        let (fin, incomplete) = e.finish();
        assert_eq!(fin.reasoning, "</thi");
        assert_eq!(fin.text, "");
        assert!(incomplete.is_none());
    }

    fn tools_with(parsed: bool, only: Option<&str>) -> ReplyTools {
        ReplyTools {
            parsed,
            only: only.map(str::to_owned),
            ..ReplyTools::default()
        }
    }

    const TWO_CALLS: &str = "<tool_call>\n<function=f>\n</function>\n</tool_call>\
                             <tool_call>\n<function=g>\n</function>\n</tool_call>";

    #[test]
    fn a_reply_that_may_call_no_tool_keeps_tool_markup_as_text() {
        let mut e = SseSafeEmitter::with_tools(false, tools_with(false, None));
        let d = e.push(TWO_CALLS);
        let (fin, incomplete) = e.finish();
        assert!(d.tool_calls.is_empty() && fin.tool_calls.is_empty());
        assert_eq!(format!("{}{}", d.text, fin.text), TWO_CALLS);
        assert_eq!(incomplete, None);
    }

    #[test]
    fn a_named_tool_choice_keeps_only_calls_to_that_tool() {
        let mut e = SseSafeEmitter::with_tools(false, tools_with(true, Some("f")));
        let d = e.push(TWO_CALLS);
        let (fin, _) = e.finish();
        let names: Vec<_> = d
            .tool_calls
            .iter()
            .chain(&fin.tool_calls)
            .map(|c| c.name.clone())
            .collect();
        assert_eq!(names, ["f"]);
    }

    #[test]
    fn a_single_call_reply_keeps_only_its_first_call_across_pushes() {
        let tools = ReplyTools {
            single: true,
            ..ReplyTools::default()
        };
        let mut e = SseSafeEmitter::with_tools(false, tools);
        let first = e.push(TWO_CALLS);
        let later = e.push("<tool_call>\n<function=h>\n</function>\n</tool_call>");
        let (fin, _) = e.finish();
        let names: Vec<_> = [first, later, fin]
            .iter()
            .flat_map(|d| d.tool_calls.iter().map(|c| c.name.clone()))
            .collect();
        assert_eq!(names, ["f"]);
    }
}
