//! Mechanical checklist harness for tokenizer gates not covered by the
//! fixture-only `test_qwen35_tokenizer` unit test.
//!
//! Covers: CORR-009 (special-token handling), TOK-005 (empty/whitespace at the
//! tokenizer level), TOK-006 (byte-fallback 0x01-0xFF), TOK-008 (streaming-decode
//! alignment — replicates the PRODUCTION `BpeTokenizerAdapter::decode_incremental`
//! algorithm from `crates/lumen-server/src/bin/lumen-server.rs`).
//!
//! Each test loads the real Qwen3.5-9B GGUF tokenizer (the production model) and
//! reduces every claim to a mechanical predicate per AH-1. Run with:
//!   cargo test --release -p lumen-cli --test tok_checklist_harness -- --ignored --nocapture

use lumen_cli::tokenize::BpeTokenizer;
use unicode_normalization::UnicodeNormalization;

const QWEN35_GGUF: &str = "/tmp/lumen-bench/Qwen_Qwen3.5-9B-Q8_0.gguf";

fn load() -> BpeTokenizer {
    let data = std::fs::read(QWEN35_GGUF)
        .unwrap_or_else(|e| panic!("read GGUF {QWEN35_GGUF}: {e}"));
    let gguf = lumen_convert::gguf::GgufFile::parse(&mut data.as_slice())
        .unwrap_or_else(|e| panic!("parse GGUF: {e}"));
    let tok = lumen_convert::tokenizer_data::extract_tokenizer(&gguf)
        .unwrap_or_else(|| panic!("no tokenizer data in GGUF"));
    BpeTokenizer::from_tokenizer_data(&tok)
}

/// PRODUCTION streaming decode, copied verbatim from
/// `BpeTokenizerAdapter::decode_incremental` (lumen-server.rs:349).
/// Buffers bytes across tokens, flushing only complete UTF-8 prefixes.
fn decode_incremental(tok: &BpeTokenizer, state: &mut Vec<u8>, token_id: u32) -> String {
    let frag_bytes = tok.decode_bytes(&[token_id]);
    state.extend_from_slice(&frag_bytes);
    match std::str::from_utf8(state) {
        Ok(_) => {
            let bytes = std::mem::take(state);
            String::from_utf8(bytes).unwrap_or_default()
        }
        Err(e) => {
            let valid = e.valid_up_to();
            if valid == 0 {
                String::new()
            } else {
                let head = state[..valid].to_vec();
                let tail = state[valid..].to_vec();
                *state = tail;
                String::from_utf8(head).unwrap_or_default()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CORR-009: Special-token handling
// ---------------------------------------------------------------------------
//   (a) encode("<literal>") == [special_id]
//   (b) decode([special_id]) == literal (no suppression in raw tokenizer)
//   (c) plain text containing the literal round-trips to the literal bytes
#[test]
#[ignore]
fn corr009_special_token_handling() {
    let tok = load();
    // Special tokens present in the Qwen3.5 vocab.
    let specials = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<think>",
        "</think>",
    ];
    let mut fail = Vec::new();
    let mut checked = 0usize;
    for s in &specials {
        let Some(id) = tok.token_to_id(s) else {
            // Token genuinely absent from this vocab — skip, do not fail.
            continue;
        };
        checked += 1;
        // (a) literal encodes to exactly the single special id.
        let enc = tok.encode(s);
        if enc != vec![id] {
            fail.push(format!("(a) encode({s:?}) = {enc:?}, want [{id}]"));
        }
        // (b) decode of the special id reproduces the literal.
        let dec = tok.decode(&[id]);
        if dec != *s {
            fail.push(format!("(b) decode([{id}]) = {dec:?}, want {s:?}"));
        }
        // (c) the literal embedded in plain text round-trips to the literal bytes.
        let wrapped = format!("a{s}b");
        let rt = tok.decode(&tok.encode(&wrapped));
        if rt != wrapped {
            fail.push(format!("(c) roundtrip({wrapped:?}) = {rt:?}"));
        }
    }
    assert!(checked >= 3, "expected >=3 special tokens in vocab, checked {checked}");
    assert!(fail.is_empty(), "CORR-009 failures ({}/{checked} special tokens):\n{}",
        fail.len(), fail.join("\n"));
    println!("CORR-009 PASS: {checked} special tokens, all 3 properties (a/b/c) hold");
}

// ---------------------------------------------------------------------------
// TOK-005 (tokenizer-level): empty / whitespace inputs do not crash and
// round-trip. (The full generation/timeout predicate is a server gate.)
// ---------------------------------------------------------------------------
#[test]
#[ignore]
fn tok005_empty_and_whitespace_tokenizer_level() {
    let tok = load();
    let cases = ["", " ", "\n\n", "\t", "   "];
    let mut fail = Vec::new();
    for c in &cases {
        let ids = tok.encode(c); // must not panic
        let dec = tok.decode(&ids);
        // NFC round-trip (Qwen applies NFC before tokenization).
        let want: String = c.nfc().collect();
        let got: String = dec.nfc().collect();
        if got != want {
            fail.push(format!("{c:?}: encode->{ids:?}->decode {got:?} != {want:?}"));
        }
    }
    assert!(fail.is_empty(), "TOK-005 tokenizer-level failures:\n{}", fail.join("\n"));
    println!("TOK-005 PASS (tokenizer-level): {}/{} empty/whitespace round-trip; no panic",
        cases.len(), cases.len());
}

// ---------------------------------------------------------------------------
// TOK-006: Byte-fallback safety. A string containing every byte 0x01..=0xFF
// encodes without panic and round-trips byte-identically.
// ---------------------------------------------------------------------------
#[test]
#[ignore]
fn tok006_byte_fallback_all_bytes() {
    let tok = load();
    // Build a UTF-8 string covering 0x01..=0xFF. Bytes 0x01-0x7F are 1-byte
    // codepoints; 0x80-0xFF are emitted as their own codepoints (Latin-1),
    // which the GPT-2 byte mapping must round-trip via the byte-to-unicode table.
    let s: String = (0x01u32..=0xFF).filter_map(char::from_u32).collect();
    // Sanity: all 255 codepoints present.
    assert_eq!(s.chars().count(), 255, "expected 255 codepoints");

    let ids = tok.encode(&s); // must not panic
    assert!(!ids.is_empty(), "encoding produced no tokens");
    let dec = tok.decode(&ids);

    // Qwen NFC-normalizes; compare NFC(decode) to NFC(input).
    let want: String = s.nfc().collect();
    let got: String = dec.nfc().collect();
    assert_eq!(
        got, want,
        "TOK-006 byte-fallback round-trip mismatch\n  ids[..16]={:?}\n  want_len={} got_len={}",
        &ids[..ids.len().min(16)], want.len(), got.len()
    );
    println!("TOK-006 PASS: 255 bytes (0x01-0xFF) -> {} tokens -> byte-identical NFC round-trip",
        ids.len());
}

// ---------------------------------------------------------------------------
// TOK-008: Streaming-decode alignment. Feeding token IDs one-at-a-time through
// the PRODUCTION incremental decoder and concatenating must equal batch decode.
// Catches partial-UTF-8 boundary bugs (multi-token codepoints, emoji, ZWJ, CJK).
// ---------------------------------------------------------------------------
#[test]
#[ignore]
fn tok008_streaming_decode_alignment() {
    let tok = load();
    // Inputs deliberately dense in multi-token / multi-byte codepoints.
    let inputs = [
        "Hello, world!",
        "café résumé",
        "北京市",
        "مرحبا",
        "🎉🔥💻",
        "👨\u{200d}👩\u{200d}👧\u{200d}👦", // family ZWJ
        "Hello 世界",
        "The quick brown fox jumps over the lazy dog.",
        "def foo(x):\n    return x * 2\n",
        "don't won't can't",
        "🇺🇸🇯🇵",       // regional-indicator flags
        "a\u{0301}e\u{0301}", // combining acute on a, e
    ];
    let mut mismatch = Vec::new();
    let mut ffr_total = 0usize; // total chunks that returned U+FFFD mid-stream
    let mut equal_count = 0usize;
    for inp in &inputs {
        let ids = tok.encode_with_special(inp);
        // Batch decode of the SAME ids.
        let batch = tok.decode(&ids);
        // Streaming decode via the production algorithm.
        let mut state: Vec<u8> = Vec::new();
        let mut streamed = String::new();
        for &id in &ids {
            let chunk = decode_incremental(&tok, &mut state, id);
            // AH-1: no chunk may contain a replacement char (mid-codepoint split).
            if chunk.contains('\u{FFFD}') {
                ffr_total += 1;
            }
            streamed.push_str(&chunk);
        }
        let eq = streamed == batch && state.is_empty();
        if eq {
            equal_count += 1;
        } else {
            mismatch.push(format!(
                "{inp:?}: streamed!=batch (unflushed={})\n    batch:    {batch:?}\n    streamed: {streamed:?}",
                state.len()
            ));
        }
    }
    println!(
        "TOK-008 RESULT: {}/{} inputs streamed==batch; {} chunk(s) contained U+FFFD across all inputs",
        equal_count, inputs.len(), ffr_total
    );
    for m in &mismatch {
        println!("  MISMATCH {m}");
    }
    // Pass criterion (TOK-008): concatenated streaming chunks byte-identical to
    // batch decode for EVERY input, and zero mid-codepoint U+FFFD chunks.
    assert!(
        mismatch.is_empty() && ffr_total == 0,
        "TOK-008 FAIL: {} mismatched input(s), {} U+FFFD chunk(s)",
        mismatch.len(),
        ffr_total
    );
    println!(
        "TOK-008 PASS: {} inputs, streamed==batch byte-identical, 0 U+FFFD chunks, buffer fully flushed",
        inputs.len()
    );
}
