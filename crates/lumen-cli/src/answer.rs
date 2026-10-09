//! The answer `lumen run` prints, written while it is generated.
//!
//! Each generated token is decoded on arrival and passed through the same
//! incremental stages the server streams with: the reasoning split (thinking
//! on), the tool-call parser and the `--stop` matcher. The answer goes to
//! stdout and is flushed per token; reasoning and tool calls go to stderr.
//! Concatenated, the output is what decoding the whole token list at once
//! gives.

use std::io::Write;

use lumen_runtime::tokenstop::StopMatcher;
use lumen_runtime::tooling::{ParsedToolCall, ReasoningExtractor, StreamingParser};

use crate::tokenize::BpeTokenizer;

pub(crate) struct AnswerPrinter<'a, O: Write, E: Write> {
    tokenizer: Option<&'a BpeTokenizer>,
    out: O,
    err: E,
    /// Bytes of a character whose remaining bytes have not arrived yet.
    pending: Vec<u8>,
    /// The reasoning split, when thinking is enabled.
    reasoning: Option<ReasoningExtractor>,
    /// A `[reasoning]` line has been started on stderr and not yet ended.
    reasoning_open: bool,
    tools: StreamingParser,
    stop: StopMatcher,
    /// A `--stop` sequence matched; nothing more of the answer is printed.
    stopped: bool,
}

impl<'a> AnswerPrinter<'a, std::io::Stdout, std::io::Stderr> {
    pub(crate) fn new(
        tokenizer: Option<&'a BpeTokenizer>,
        enable_thinking: bool,
        stops: Vec<String>,
    ) -> Self {
        Self::with_writers(
            tokenizer,
            enable_thinking,
            stops,
            std::io::stdout(),
            std::io::stderr(),
        )
    }
}

impl<'a, O: Write, E: Write> AnswerPrinter<'a, O, E> {
    fn with_writers(
        tokenizer: Option<&'a BpeTokenizer>,
        enable_thinking: bool,
        stops: Vec<String>,
        out: O,
        err: E,
    ) -> Self {
        Self {
            tokenizer,
            out,
            err,
            pending: Vec::new(),
            reasoning: enable_thinking.then(|| ReasoningExtractor::new(true)),
            reasoning_open: false,
            tools: StreamingParser::new(),
            stop: StopMatcher::new(stops),
            stopped: false,
        }
    }

    /// Print what token `id` completes. Stop tokens print nothing.
    pub(crate) fn token(&mut self, id: u32) {
        let Some(tok) = self.tokenizer else {
            return;
        };
        if tok.stop_token_ids.contains(&id) {
            return;
        }
        self.pending.extend_from_slice(&tok.decode_bytes(&[id]));
        let text = take_decodable(&mut self.pending);
        if !text.is_empty() {
            self.text(&text);
        }
    }

    /// Print what is left once generation has ended, then the final newline.
    /// `tokens` is the whole generated list, dumped to stderr under
    /// `LUMEN_SPEC_DUMP_IDS=1`.
    pub(crate) fn finish(mut self, tokens: &[u32]) {
        if std::env::var("LUMEN_SPEC_DUMP_IDS").as_deref() == Ok("1") {
            let _ = writeln!(
                self.err,
                "[SPEC_DUMP_IDS] raw_count={} ids={tokens:?}",
                tokens.len()
            );
        }
        if self.tokenizer.is_none() {
            return;
        }
        if !self.pending.is_empty() {
            let rest = String::from_utf8_lossy(&std::mem::take(&mut self.pending)).into_owned();
            self.text(&rest);
        }
        if let Some(reasoning) = self.reasoning.take() {
            let tail = reasoning.finish();
            self.reason(&tail.reasoning);
            self.content(&tail.content);
        }
        let tools = std::mem::replace(&mut self.tools, StreamingParser::new());
        let fin = tools.finish();
        self.answer(&fin.flushed_text);
        let stop = std::mem::replace(&mut self.stop, StopMatcher::new(Vec::new()));
        if !self.stopped {
            let held = stop.finish();
            self.write_answer(&held);
        }
        self.end_reasoning();
        let _ = writeln!(self.out);
        let _ = self.out.flush();
    }

    fn text(&mut self, text: &str) {
        let content = match self.reasoning.as_mut() {
            Some(reasoning) => {
                let delta = reasoning.feed(text);
                self.reason(&delta.reasoning);
                delta.content
            }
            None => text.to_string(),
        };
        self.content(&content);
    }

    fn content(&mut self, content: &str) {
        if content.is_empty() {
            return;
        }
        let delta = self.tools.feed(content);
        for call in &delta.tool_calls {
            self.tool_call(call);
        }
        self.answer(&delta.text());
    }

    fn answer(&mut self, text: &str) {
        if self.stopped || text.is_empty() {
            return;
        }
        let (safe, hit) = self.stop.push(text);
        self.write_answer(&safe);
        self.stopped = hit;
    }

    fn write_answer(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let _ = self.out.write_all(text.as_bytes());
        let _ = self.out.flush();
    }

    fn reason(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if !self.reasoning_open {
            let _ = write!(self.err, "[reasoning] ");
            self.reasoning_open = true;
        }
        let _ = write!(self.err, "{text}");
    }

    fn end_reasoning(&mut self) {
        if self.reasoning_open {
            let _ = writeln!(self.err);
            self.reasoning_open = false;
        }
    }

    fn tool_call(&mut self, call: &ParsedToolCall) {
        self.end_reasoning();
        let _ = writeln!(
            self.err,
            "[tool_call] {}({})",
            call.name, call.arguments_json
        );
    }
}

/// Take the decodable front of `bytes`: every complete character, with each
/// invalid sequence replaced by U+FFFD where it stands (as a lossy decode of
/// the whole stream does), leaving only the start of a character that later
/// bytes may complete.
fn take_decodable(bytes: &mut Vec<u8>) -> String {
    let mut out = String::new();
    let mut start = 0;
    loop {
        match std::str::from_utf8(&bytes[start..]) {
            Ok(rest) => {
                out.push_str(rest);
                start = bytes.len();
                break;
            }
            Err(e) => {
                let valid = start + e.valid_up_to();
                out.push_str(std::str::from_utf8(&bytes[start..valid]).unwrap_or_default());
                match e.error_len() {
                    Some(len) => {
                        out.push(char::REPLACEMENT_CHARACTER);
                        start = valid + len;
                    }
                    None => {
                        start = valid;
                        break;
                    }
                }
            }
        }
    }
    bytes.drain(..start);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenize::build_byte_to_unicode;

    const STOP_ID: u32 = 256;

    /// A byte-level tokenizer: id `b` (0..256) decodes to byte `b`, and
    /// `STOP_ID` is its end-of-turn token. Any byte sequence, valid UTF-8 or
    /// not, can be fed one byte per token.
    fn byte_tokenizer() -> BpeTokenizer {
        let b2u = build_byte_to_unicode();
        let mut tokens: Vec<String> = (0..256).map(|b| b2u[b].to_string()).collect();
        tokens.push("<|im_end|>".into());
        let data = lumen_convert::tokenizer_data::TokenizerData {
            model_type: "gpt2".into(),
            pre_tokenizer: "qwen35".into(),
            token_types: vec![1; tokens.len()],
            scores: vec![0.0; tokens.len()],
            tokens,
            merges: Vec::new(),
            bos_token_id: STOP_ID,
            eos_token_id: STOP_ID,
            pad_token_id: None,
            add_bos_token: false,
            add_eos_token: false,
            add_space_prefix: false,
            chat_template: None,
        };
        BpeTokenizer::from_tokenizer_data(&data)
    }

    fn ids(bytes: &[u8]) -> Vec<u32> {
        bytes.iter().map(|&b| b as u32).collect()
    }

    /// What `lumen run` printed before it streamed: decode the whole list
    /// (stop ids removed), split the reasoning, strip tool calls, cut at the
    /// earliest-starting stop, then one `println`.
    fn batch(
        tok: &BpeTokenizer,
        tokens: &[u32],
        thinking: bool,
        stops: &[String],
    ) -> (String, String) {
        let clean: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|t| !tok.stop_token_ids.contains(t))
            .collect();
        let text = tok.decode(&clean);
        let mut err = String::new();
        let content = if thinking {
            let mut extractor = ReasoningExtractor::new(true);
            let mut delta = extractor.feed(&text);
            let tail = extractor.finish();
            delta.reasoning.push_str(&tail.reasoning);
            delta.content.push_str(&tail.content);
            if !delta.reasoning.is_empty() {
                err.push_str(&format!("[reasoning] {}\n", delta.reasoning));
            }
            delta.content
        } else {
            text
        };
        let parsed = lumen_runtime::tooling::parse_final(&content);
        for call in &parsed.tool_calls {
            err.push_str(&format!(
                "[tool_call] {}({})\n",
                call.name, call.arguments_json
            ));
        }
        let mut cut = None;
        for s in stops {
            if let Some(pos) = parsed.content.find(s.as_str()) {
                cut = Some(cut.map_or(pos, |c: usize| c.min(pos)));
            }
        }
        let answer = &parsed.content[..cut.unwrap_or(parsed.content.len())];
        (format!("{answer}\n"), err)
    }

    /// What the streaming printer writes for `tokens`, fed one at a time.
    fn streamed(
        tok: &BpeTokenizer,
        tokens: &[u32],
        thinking: bool,
        stops: &[String],
    ) -> (String, String) {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut printer =
            AnswerPrinter::with_writers(Some(tok), thinking, stops.to_vec(), &mut out, &mut err);
        for &id in tokens {
            printer.token(id);
        }
        printer.finish(tokens);
        (
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    fn same_as_batch(tokens: &[u32], thinking: bool, stops: &[&str]) {
        let tok = byte_tokenizer();
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            streamed(&tok, tokens, thinking, &stops),
            batch(&tok, tokens, thinking, &stops),
            "tokens {tokens:?}"
        );
    }

    #[test]
    fn plain_text_prints_as_the_batch_decode_did() {
        same_as_batch(&ids(b"Hello, world."), false, &[]);
    }

    #[test]
    fn a_character_split_across_tokens_prints_whole() {
        same_as_batch(&ids("caf\u{e9} \u{1f600}!".as_bytes()), false, &[]);
    }

    #[test]
    fn stop_tokens_anywhere_print_nothing() {
        let mut tokens = ids(b"one ");
        tokens.push(STOP_ID);
        tokens.extend(ids(b"two"));
        tokens.push(STOP_ID);
        same_as_batch(&tokens, false, &[]);
    }

    #[test]
    fn invalid_bytes_are_replaced_where_they_stand() {
        // A stray continuation byte mid-stream, and a four-byte lead followed
        // by ASCII: each U+FFFD lands where the batch decode puts it.
        same_as_batch(&ids(b"a\x80b"), false, &[]);
        same_as_batch(&ids(b"x\xf0y z"), false, &[]);
        same_as_batch(&ids(b"x\xf0\x9f\x98y"), false, &[]);
    }

    #[test]
    fn an_unfinished_character_at_the_end_prints_as_the_batch_decode_did() {
        same_as_batch(&ids(b"end \xe2\x82"), false, &[]);
    }

    #[test]
    fn a_stop_sequence_inside_one_token_or_across_tokens_cuts_the_answer() {
        same_as_batch(&ids(b"keep this STOP drop this"), false, &["STOP"]);
        same_as_batch(&ids(b"aaa END bbb STOP ccc"), false, &["STOP", "END"]);
        same_as_batch(&ids(b"no match here"), false, &["ZZZ"]);
    }

    #[test]
    fn overlapping_stops_cut_at_the_first_to_complete() {
        // The one change from the batch printer: with stops "abcd" and "c",
        // "c" completes first, so the answer is "ab" (the server's rule); the
        // batch printer cut at the earliest start and printed nothing.
        let tok = byte_tokenizer();
        let stops = vec!["abcd".to_string(), "c".to_string()];
        let (out, err) = streamed(&tok, &ids(b"abcd"), false, &stops);
        assert_eq!(out, "ab\n");
        assert_eq!(err, "");
    }

    #[test]
    fn tool_calls_go_to_stderr_as_before() {
        same_as_batch(
            &ids(br#"Checking.<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call> Done."#),
            false,
            &[],
        );
    }

    #[test]
    fn reasoning_goes_to_stderr_and_the_answer_to_stdout() {
        same_as_batch(
            &ids(b"weighing it up</think>\n\nThe answer is 4."),
            true,
            &[],
        );
        same_as_batch(
            &ids(br#"plan</think>Calling.<tool_call>{"name": "f", "arguments": {}}</tool_call>"#),
            true,
            &[],
        );
        same_as_batch(&ids(b"no reasoning close at all"), true, &[]);
    }

    #[test]
    fn each_complete_piece_is_written_before_the_next_token() {
        let tok = byte_tokenizer();
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut printer =
            AnswerPrinter::with_writers(Some(&tok), false, Vec::new(), &mut out, &mut err);
        for id in ids(b"Hi") {
            printer.token(id);
        }
        printer.token(0xc3);
        drop(printer);
        assert_eq!(
            out, b"Hi",
            "the completed text is out; the half character is held"
        );
    }

    #[test]
    fn an_invalid_byte_does_not_hold_back_what_follows() {
        let tok = byte_tokenizer();
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut printer =
            AnswerPrinter::with_writers(Some(&tok), false, Vec::new(), &mut out, &mut err);
        for id in ids(b"a\x80b") {
            printer.token(id);
        }
        drop(printer);
        assert_eq!(String::from_utf8(out).unwrap(), "a\u{fffd}b");
    }
}
