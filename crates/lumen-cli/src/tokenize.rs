//! Native BPE tokenizer for Lumen.
//!
//! Supports two tokenizer families:
//! - **tiktoken-style BPE** (model="gpt2"): Llama-3, Qwen2.5, Qwen3.5
//!   Uses a pre-tokenizer regex, GPT-2 byte-to-unicode mapping, ranked merges.
//! - **SentencePiece BPE** (model="llama"): TinyLlama
//!   Uses scores as merge priorities, non-legacy prefix space, byte fallback.

use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

// ---------------------------------------------------------------------------
// GPT-2 byte-to-unicode mapping
// ---------------------------------------------------------------------------

/// Build the GPT-2 byte-to-unicode mapping.
/// Maps each byte value (0..256) to a unicode character.
/// Printable bytes map to themselves; others map to U+0100..U+013F.
fn build_byte_to_unicode() -> [char; 256] {
    let mut b2u = ['\0'; 256];
    let mut n: u32 = 0;
    // First pass: printable ranges map to themselves.
    for b in 0u16..256 {
        let byte = b as u8;
        if is_direct_byte(byte) {
            b2u[byte as usize] = byte as char;
        }
    }
    // Second pass: remaining bytes map to U+0100 + n.
    for b in 0u16..256 {
        let byte = b as u8;
        if !is_direct_byte(byte) {
            b2u[byte as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    b2u
}

fn is_direct_byte(b: u8) -> bool {
    (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b)
}

/// Build the reverse mapping: unicode char -> byte value.
fn build_unicode_to_byte() -> HashMap<char, u8> {
    let b2u = build_byte_to_unicode();
    let mut u2b = HashMap::with_capacity(256);
    for (b, &c) in b2u.iter().enumerate() {
        u2b.insert(c, b as u8);
    }
    u2b
}

// ---------------------------------------------------------------------------
// BpeTokenizer
// ---------------------------------------------------------------------------

/// A native BPE tokenizer that matches HuggingFace output token-for-token.
pub struct BpeTokenizer {
    /// Vocabulary: token_id -> token string.
    vocab: Vec<String>,
    /// Reverse mapping: token string -> token_id.
    token_to_id: HashMap<String, u32>,
    /// For tiktoken-style: merge pair -> rank (lower = higher priority).
    merge_ranks: HashMap<(String, String), usize>,
    /// For SPM: scores per token (higher = higher priority merge).
    scores: Vec<f32>,
    /// Pre-tokenizer regex pattern (compiled, supports lookahead via fancy-regex).
    pre_regex: Option<fancy_regex::Regex>,
    /// Tokenizer model type: "gpt2" or "llama".
    model_type: String,
    /// Pre-tokenizer identifier: "llama-bpe", "qwen2", "qwen35", "default".
    pre_tokenizer: String,
    /// Beginning-of-sequence token ID.
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token ID.
    pub eos_token_id: u32,
    /// Stop token IDs (includes eos + model-specific stop tokens).
    pub stop_token_ids: Vec<u32>,
    /// Whether to automatically prepend BOS.
    pub add_bos_token: bool,
    /// Whether to automatically append EOS.
    pub add_eos_token: bool,
    /// Whether to prepend a leading space (SPM).
    add_space_prefix: bool,
    /// GPT-2 byte-to-unicode mapping.
    byte_to_unicode: [char; 256],
    /// Reverse: unicode-to-byte.
    unicode_to_byte: HashMap<char, u8>,
    /// Special tokens: exact string -> token_id. Sorted longest-first.
    special_tokens: Vec<(String, u32)>,
}

/// Pre-tokenizer regex shared by llama-bpe, qwen2, and qwen35.
/// Uses negative lookahead `(?!\S)` which requires fancy-regex.
const PRETOKENIZER_REGEX: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";


impl BpeTokenizer {
    /// Build a tokenizer from extracted GGUF tokenizer data.
    pub fn from_tokenizer_data(data: &lumen_convert::tokenizer_data::TokenizerData) -> Self {
        let byte_to_unicode = build_byte_to_unicode();
        let unicode_to_byte = build_unicode_to_byte();

        // Build vocab and reverse mapping.
        let vocab = data.tokens.clone();
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (id, tok_str) in vocab.iter().enumerate() {
            token_to_id.insert(tok_str.clone(), id as u32);
        }

        // Build merge rank HashMap for O(1) lookup.
        let mut merge_ranks = HashMap::with_capacity(data.merges.len());
        for (rank, line) in data.merges.iter().enumerate() {
            if let Some((a, b)) = line.split_once(' ') {
                merge_ranks.insert((a.to_string(), b.to_string()), rank);
            }
        }

        let scores = data.scores.clone();

        // Compile pre-tokenizer regex for tiktoken-style models.
        let pre_regex = if data.model_type == "gpt2" {
            Some(
                fancy_regex::Regex::new(PRETOKENIZER_REGEX)
                    .expect("failed to compile pre-tokenizer regex"),
            )
        } else {
            None
        };

        // Determine BOS token ID.
        let bos_token_id = if data.bos_token_id == 0
            && data.model_type == "gpt2"
            && data.pre_tokenizer != "default"
        {
            // Qwen models: token 0 is "!" (not BOS). BOS is effectively None.
            if vocab.first().map(|s| s.as_str()) == Some("!") {
                None
            } else {
                Some(data.bos_token_id)
            }
        } else {
            Some(data.bos_token_id)
        };

        // Detect special tokens from vocabulary.
        let mut special_tokens = Vec::new();
        let mut stop_token_ids = vec![data.eos_token_id];

        // Scan for special token patterns.
        // Type 3 = control, type 4 = user_defined — both are special tokens.
        for (id, tok_str) in vocab.iter().enumerate() {
            let id = id as u32;
            let token_type = data
                .token_types
                .get(id as usize)
                .copied()
                .unwrap_or(1);
            let is_special = token_type == 3 || token_type == 4;
            if is_special && tok_str.starts_with('<') && tok_str.ends_with('>') {
                special_tokens.push((tok_str.clone(), id));
            }

            // Collect stop tokens.
            match tok_str.as_str() {
                "<|eot_id|>" | "<|im_end|>" => {
                    if !stop_token_ids.contains(&id) {
                        stop_token_ids.push(id);
                    }
                }
                _ => {}
            }
        }

        // For models without token_types (Qwen), scan by name.
        if special_tokens.is_empty() {
            for pattern in &[
                "<|endoftext|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|eot_id|>",
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<think>",
                "</think>",
            ] {
                if let Some(&id) = token_to_id.get(*pattern) {
                    special_tokens.push((pattern.to_string(), id));
                }
            }
        }

        // Sort special tokens by length descending for greedy matching.
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        BpeTokenizer {
            vocab,
            token_to_id,
            merge_ranks,
            scores,
            pre_regex,
            model_type: data.model_type.clone(),
            pre_tokenizer: data.pre_tokenizer.clone(),
            bos_token_id,
            eos_token_id: data.eos_token_id,
            stop_token_ids,
            add_bos_token: data.add_bos_token,
            add_eos_token: data.add_eos_token,
            add_space_prefix: data.add_space_prefix,
            byte_to_unicode,
            unicode_to_byte,
            special_tokens,
        }
    }

    /// Encode text into token IDs (without BOS/EOS -- raw BPE output).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        if self.model_type == "llama" {
            self.encode_spm(text)
        } else {
            self.encode_tiktoken(text)
        }
    }

    /// Encode with BOS/EOS tokens prepended/appended as configured.
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut ids = self.encode(text);
        if self.add_bos_token {
            if let Some(bos) = self.bos_token_id {
                ids.insert(0, bos);
            }
        }
        if self.add_eos_token {
            ids.push(self.eos_token_id);
        }
        ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        if self.model_type == "llama" {
            self.decode_spm(ids)
        } else {
            self.decode_tiktoken(ids)
        }
    }

    /// Apply chat template and return the full prompt string.
    pub fn apply_chat_template(&self, prompt: &str) -> String {
        self.apply_chat_template_with_system(prompt, None)
    }

    pub fn apply_chat_template_with_system(&self, prompt: &str, system: Option<&str>) -> String {
        if self.token_to_id.contains_key("<|begin_of_text|>") {
            // Llama-3 style
            let sys = system.unwrap_or(
                "Cutting Knowledge Date: December 2023\n\
                 Today Date: 26 Jul 2024\n\n"
            );
            format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
                 {sys}\
                 <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
                 {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        } else if self.pre_tokenizer == "qwen35" {
            // Qwen3.5: system optional, includes <think>
            if let Some(sys) = system {
                format!("<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n")
            } else {
                format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n")
            }
        } else if self.pre_tokenizer == "qwen2" {
            // Qwen2.5: ChatML with system message
            let sys = system.unwrap_or(
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            );
            format!(
                "<|im_start|>system\n\
                 {sys}<|im_end|>\n\
                 <|im_start|>user\n\
                 {prompt}<|im_end|>\n\
                 <|im_start|>assistant\n"
            )
        } else {
            // TinyLlama / generic
            if let Some(sys) = system {
                format!("<|system|>\n{sys}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n")
            } else {
                format!("<|user|>\n{prompt}</s>\n<|assistant|>\n")
            }
        }
    }

    // =========================================================================
    // tiktoken-style BPE (Llama-3, Qwen2.5, Qwen3.5)
    // =========================================================================

    fn encode_tiktoken(&self, text: &str) -> Vec<u32> {
        let re = self.pre_regex.as_ref().unwrap();
        let mut all_ids = Vec::new();

        // Qwen models apply NFC normalization before tokenization.
        let normalized;
        let text = if self.pre_tokenizer == "qwen2" || self.pre_tokenizer == "qwen35" {
            normalized = text.nfc().collect::<String>();
            normalized.as_str()
        } else {
            text
        };

        // Split text into segments: special tokens and normal text.
        let segments = self.split_on_special_tokens(text);

        for segment in segments {
            match segment {
                Segment::Special(id) => {
                    all_ids.push(id);
                }
                Segment::Text(chunk) => {
                    // Pre-tokenize: split using regex.
                    // fancy_regex::Regex::find_iter returns Result items.
                    let mut start = 0;
                    while start < chunk.len() {
                        match re.find_from_pos(&chunk, start) {
                            Ok(Some(mat)) => {
                                let piece = mat.as_str();
                                // Convert bytes to GPT-2 unicode.
                                let unicode_piece: String = piece
                                    .as_bytes()
                                    .iter()
                                    .map(|&b| self.byte_to_unicode[b as usize])
                                    .collect();

                                // Split into individual GPT-2 unicode chars.
                                let symbols: Vec<String> =
                                    unicode_piece.chars().map(|c| c.to_string()).collect();

                                // Apply BPE merges.
                                let merged = self.bpe_merge_tiktoken(&symbols);

                                // Map to token IDs.
                                for sym in &merged {
                                    if let Some(&id) = self.token_to_id.get(sym) {
                                        all_ids.push(id);
                                    }
                                }

                                start = mat.end();
                            }
                            Ok(None) => break,
                            Err(_) => break,
                        }
                    }
                }
            }
        }

        all_ids
    }

    /// Split text into segments of special tokens and normal text.
    fn split_on_special_tokens(&self, text: &str) -> Vec<Segment> {
        if self.special_tokens.is_empty() {
            return vec![Segment::Text(text.to_string())];
        }

        let mut segments = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Try to match a special token at the current position.
            let mut found = false;
            for (tok_str, tok_id) in &self.special_tokens {
                if remaining.starts_with(tok_str.as_str()) {
                    segments.push(Segment::Special(*tok_id));
                    remaining = &remaining[tok_str.len()..];
                    found = true;
                    break;
                }
            }
            if found {
                continue;
            }

            // Find the next special token occurrence.
            let mut next_pos = remaining.len();
            for (tok_str, _) in &self.special_tokens {
                if let Some(pos) = remaining.find(tok_str.as_str()) {
                    if pos < next_pos {
                        next_pos = pos;
                    }
                }
            }

            let normal = &remaining[..next_pos];
            if !normal.is_empty() {
                segments.push(Segment::Text(normal.to_string()));
            }
            remaining = &remaining[next_pos..];
        }

        segments
    }

    /// Apply BPE merges using ranked merge pairs (lower rank = higher priority).
    fn bpe_merge_tiktoken(&self, symbols: &[String]) -> Vec<String> {
        if symbols.len() <= 1 {
            return symbols.to_vec();
        }

        let mut work: Vec<String> = symbols.to_vec();

        loop {
            if work.len() <= 1 {
                break;
            }

            // Find the pair with the lowest merge rank.
            let mut best_rank = usize::MAX;
            let mut best_idx = usize::MAX;

            for i in 0..work.len() - 1 {
                let key = (work[i].clone(), work[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&key) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            // Merge the pair.
            let merged = format!("{}{}", work[best_idx], work[best_idx + 1]);
            work[best_idx] = merged;
            work.remove(best_idx + 1);
        }

        work
    }

    fn decode_tiktoken(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok_str) = self.vocab.get(id as usize) {
                for c in tok_str.chars() {
                    if let Some(&b) = self.unicode_to_byte.get(&c) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    // =========================================================================
    // SentencePiece BPE (TinyLlama)
    // =========================================================================

    fn encode_spm(&self, text: &str) -> Vec<u32> {
        let mut all_ids = Vec::new();

        // Split on special tokens first.
        let segments = self.split_on_special_tokens(text);
        let mut is_first_text = true;

        for segment in segments {
            match segment {
                Segment::Special(id) => {
                    all_ids.push(id);
                }
                Segment::Text(chunk) => {
                    let ids = self.encode_spm_chunk(&chunk, is_first_text);
                    all_ids.extend(ids);
                    is_first_text = false;
                }
            }
        }

        all_ids
    }

    /// Encode a single text chunk with SentencePiece BPE.
    /// `add_prefix`: whether to apply the dummy prefix space logic.
    fn encode_spm_chunk(&self, text: &str, add_prefix: bool) -> Vec<u32> {
        // Non-legacy HF LlamaTokenizer behavior:
        // Only the first text segment gets the dummy prefix space.
        // If add_space_prefix is true AND this is the first segment
        // AND text does NOT start with space or ▁, prepend a space.
        let text = if self.add_space_prefix && add_prefix {
            let starts_with_space =
                text.starts_with(' ') || text.starts_with('\u{2581}');
            if starts_with_space {
                text.to_string()
            } else {
                format!(" {text}")
            }
        } else {
            text.to_string()
        };

        // Replace all spaces with ▁ (U+2581).
        let text = text.replace(' ', "\u{2581}");

        // Split into individual characters (or byte fallback for unknown chars).
        let mut symbols: Vec<String> = Vec::new();
        for ch in text.chars() {
            let ch_str = ch.to_string();
            if self.token_to_id.contains_key(&ch_str) {
                symbols.push(ch_str);
            } else {
                // Byte fallback: encode as <0xHH> tokens.
                for b in ch_str.as_bytes() {
                    let hex_tok = format!("<0x{:02X}>", b);
                    symbols.push(hex_tok);
                }
            }
        }

        if symbols.is_empty() {
            return Vec::new();
        }

        // Apply BPE merges using scores.
        let merged = self.bpe_merge_spm(&symbols);

        // Map to token IDs.
        let mut ids = Vec::new();
        for sym in &merged {
            if let Some(&id) = self.token_to_id.get(sym) {
                ids.push(id);
            }
        }

        ids
    }

    /// Apply BPE merges for SentencePiece: repeatedly merge the adjacent pair
    /// whose merged token has the highest score. All vocab-present merges are
    /// valid; the score only determines priority (higher = merge first).
    fn bpe_merge_spm(&self, symbols: &[String]) -> Vec<String> {
        if symbols.len() <= 1 {
            return symbols.to_vec();
        }

        let mut work = symbols.to_vec();

        loop {
            if work.len() <= 1 {
                break;
            }

            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;

            for i in 0..work.len() - 1 {
                let merged_str = format!("{}{}", work[i], work[i + 1]);
                if let Some(&id) = self.token_to_id.get(&merged_str) {
                    let score = self
                        .scores
                        .get(id as usize)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY);
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            let merged = format!("{}{}", work[best_idx], work[best_idx + 1]);
            work[best_idx] = merged;
            work.remove(best_idx + 1);
        }

        work
    }

    fn decode_spm(&self, ids: &[u32]) -> String {
        let mut byte_buf: Vec<u8> = Vec::new();
        let mut result = String::new();

        // Flush accumulated byte-fallback bytes as UTF-8.
        let flush_bytes = |buf: &mut Vec<u8>, out: &mut String| {
            if !buf.is_empty() {
                out.push_str(&String::from_utf8_lossy(buf));
                buf.clear();
            }
        };

        for &id in ids {
            if let Some(tok_str) = self.vocab.get(id as usize) {
                // Byte fallback token: <0xHH>
                if tok_str.starts_with("<0x") && tok_str.ends_with('>') && tok_str.len() == 6 {
                    if let Ok(byte_val) = u8::from_str_radix(&tok_str[3..5], 16) {
                        byte_buf.push(byte_val);
                        continue;
                    }
                }
                // Non-byte token: flush any accumulated bytes first.
                flush_bytes(&mut byte_buf, &mut result);
                result.push_str(tok_str);
            }
        }
        // Flush any remaining bytes.
        flush_bytes(&mut byte_buf, &mut result);

        // Replace ▁ back to space.
        let result = result.replace('\u{2581}', " ");
        // Remove leading space if it was added.
        if self.add_space_prefix && result.starts_with(' ') {
            result[1..].to_string()
        } else {
            result
        }
    }

    // =========================================================================
    // Public accessors
    // =========================================================================

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Look up a token string by ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(|s| s.as_str())
    }

    /// Look up a token ID by string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
}

enum Segment {
    Special(u32),
    Text(String),
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixture format matching the JSON test files.
    #[derive(serde::Deserialize)]
    struct Fixture {
        model_key: String,
        #[allow(dead_code)]
        hf_model_id: String,
        #[allow(dead_code)]
        gguf_model_type: String,
        #[allow(dead_code)]
        gguf_pre: String,
        #[allow(dead_code)]
        vocab_size: u32,
        #[allow(dead_code)]
        bos_token_id: Option<u32>,
        #[allow(dead_code)]
        eos_token_id: Option<u32>,
        test_cases: Vec<TestCase>,
        chat_template_test: Option<ChatTemplateTest>,
    }

    #[derive(serde::Deserialize)]
    struct TestCase {
        id: String,
        input: String,
        token_ids: Vec<u32>,
        decoded: String,
        round_trip: bool,
        #[allow(dead_code)]
        num_tokens: usize,
    }

    #[derive(serde::Deserialize)]
    struct ChatTemplateTest {
        #[allow(dead_code)]
        messages: Vec<serde_json::Value>,
        expected_text: String,
        expected_token_ids: Vec<u32>,
    }

    /// Load a tokenizer from a real GGUF file on disk.
    fn load_tokenizer_from_gguf(path: &str) -> BpeTokenizer {
        let data = std::fs::read(path).unwrap_or_else(|e| {
            panic!("Failed to read GGUF file {path}: {e}");
        });
        let gguf = lumen_convert::gguf::GgufFile::parse(&mut data.as_slice())
            .unwrap_or_else(|e| panic!("Failed to parse GGUF {path}: {e}"));
        let tok_data = lumen_convert::tokenizer_data::extract_tokenizer(&gguf)
            .unwrap_or_else(|| panic!("No tokenizer data in GGUF {path}"));
        BpeTokenizer::from_tokenizer_data(&tok_data)
    }

    /// Load a fixture JSON file.
    fn load_fixture(json: &str) -> Fixture {
        serde_json::from_str(json).expect("Failed to parse fixture JSON")
    }

    // =========================================================================
    // GPT-2 byte-to-unicode mapping tests
    // =========================================================================

    #[test]
    fn byte_to_unicode_is_bijective() {
        let b2u = build_byte_to_unicode();
        let mut seen = std::collections::HashSet::new();
        for &c in &b2u {
            assert!(c != '\0', "null char in byte-to-unicode mapping");
            assert!(
                seen.insert(c),
                "duplicate char {c:?} in byte-to-unicode mapping"
            );
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn byte_to_unicode_printable_ascii() {
        let b2u = build_byte_to_unicode();
        for b in 33u8..=126 {
            assert_eq!(b2u[b as usize], b as char, "byte {b} should map to itself");
        }
    }

    #[test]
    fn byte_to_unicode_space_not_direct() {
        let b2u = build_byte_to_unicode();
        assert_ne!(b2u[32], ' ');
        assert!(b2u[32] as u32 >= 256);
    }

    #[test]
    fn unicode_to_byte_roundtrip() {
        let b2u = build_byte_to_unicode();
        let u2b = build_unicode_to_byte();
        for b in 0u8..=255 {
            let c = b2u[b as usize];
            assert_eq!(u2b[&c], b, "roundtrip failed for byte {b}");
        }
    }

    // =========================================================================
    // Fixture-driven tests against real GGUF files
    // =========================================================================

    const TINYLLAMA_GGUF: &str = "/tmp/lumen-bench/TinyLlama-1.1B-Chat-v1.0-f16.gguf";
    const LLAMA3_GGUF: &str = "/tmp/lumen-bench/Llama-3.1-8B-Instruct-f16.gguf";
    const QWEN25_GGUF: &str = "/tmp/lumen-bench/Qwen2.5-3B-Instruct-Q8_0.gguf";
    const QWEN35_GGUF: &str = "/tmp/lumen-bench/Qwen_Qwen3.5-9B-Q8_0.gguf";

    const TINYLLAMA_FIXTURE: &str =
        include_str!("../../../tests/fixtures/tokenizer_tinyllama.json");
    const LLAMA3_FIXTURE: &str =
        include_str!("../../../tests/fixtures/tokenizer_llama_3_1_8b.json");
    const QWEN25_FIXTURE: &str =
        include_str!("../../../tests/fixtures/tokenizer_qwen2_5_3b.json");
    const QWEN35_FIXTURE: &str =
        include_str!("../../../tests/fixtures/tokenizer_qwen3_5_9b.json");

    /// Run all test cases from a fixture against a loaded tokenizer.
    fn run_fixture_tests(tokenizer: &BpeTokenizer, fixture: &Fixture) {
        let mut encode_failures = Vec::new();
        let mut decode_failures = Vec::new();
        let mut roundtrip_failures = Vec::new();

        for tc in &fixture.test_cases {
            // Test encode.
            let got_ids = tokenizer.encode(&tc.input);
            if got_ids != tc.token_ids {
                encode_failures.push(format!(
                    "  ENCODE [{}] input={:?}\n    expected: {:?}\n    got:      {:?}",
                    tc.id, tc.input, tc.token_ids, got_ids
                ));
            }

            // Test decode (using fixture's token_ids, not our encode output).
            let got_decoded = tokenizer.decode(&tc.token_ids);
            if got_decoded != tc.decoded {
                decode_failures.push(format!(
                    "  DECODE [{}] ids={:?}\n    expected: {:?}\n    got:      {:?}",
                    tc.id, &tc.token_ids[..tc.token_ids.len().min(10)], tc.decoded, got_decoded
                ));
            }

            // Test round-trip: decode(encode(input)) produces the canonical decoded form.
            // Note: NFC normalization means decode(encode("cafe\u{0301}")) = "café" (NFC),
            // so we compare against the fixture's decoded value, not the original input.
            if tc.round_trip {
                let roundtrip = tokenizer.decode(&got_ids);
                if roundtrip != tc.decoded {
                    roundtrip_failures.push(format!(
                        "  ROUNDTRIP [{}] input={:?}\n    expected: {:?}\n    got: {:?}",
                        tc.id, tc.input, tc.decoded, roundtrip
                    ));
                }
            }
        }

        let total_failures =
            encode_failures.len() + decode_failures.len() + roundtrip_failures.len();
        if total_failures > 0 {
            let mut msg = format!(
                "Tokenizer {} failed {} checks:\n",
                fixture.model_key, total_failures
            );
            for f in &encode_failures { msg.push_str(f); msg.push('\n'); }
            for f in &decode_failures { msg.push_str(f); msg.push('\n'); }
            for f in &roundtrip_failures { msg.push_str(f); msg.push('\n'); }
            panic!("{msg}");
        }
    }

    /// Run chat template test from a fixture.
    fn run_chat_template_test(tokenizer: &BpeTokenizer, fixture: &Fixture) {
        if let Some(ref chat_test) = fixture.chat_template_test {
            let prompt = "Hello";
            let template_text = tokenizer.apply_chat_template(prompt);
            assert_eq!(
                template_text, chat_test.expected_text,
                "Chat template text mismatch for {}",
                fixture.model_key
            );

            // Verify tokenization of the template text.
            let got_ids = tokenizer.encode(&template_text);
            assert_eq!(
                got_ids, chat_test.expected_token_ids,
                "Chat template token IDs mismatch for {}.\ntext: {:?}\nexpected: {:?}\ngot: {:?}",
                fixture.model_key, template_text, chat_test.expected_token_ids, got_ids
            );
        }
    }

    /// Validate encode/decode/round-trip against HuggingFace fixtures.
    /// Requires GGUF model files at /tmp/lumen-bench/. Run with:
    ///   cargo test --release -p lumen-cli -- --ignored tokenize
    #[test]
    #[ignore]
    fn test_tinyllama_tokenizer() {
        let tokenizer = load_tokenizer_from_gguf(TINYLLAMA_GGUF);
        let fixture = load_fixture(TINYLLAMA_FIXTURE);
        run_fixture_tests(&tokenizer, &fixture);
    }

    #[test]
    #[ignore]
    fn test_llama3_tokenizer() {
        let tokenizer = load_tokenizer_from_gguf(LLAMA3_GGUF);
        let fixture = load_fixture(LLAMA3_FIXTURE);
        run_fixture_tests(&tokenizer, &fixture);
    }

    #[test]
    #[ignore]
    fn test_qwen25_tokenizer() {
        let tokenizer = load_tokenizer_from_gguf(QWEN25_GGUF);
        let fixture = load_fixture(QWEN25_FIXTURE);
        run_fixture_tests(&tokenizer, &fixture);
    }

    #[test]
    #[ignore]
    fn test_qwen35_tokenizer() {
        let tokenizer = load_tokenizer_from_gguf(QWEN35_GGUF);
        let fixture = load_fixture(QWEN35_FIXTURE);
        run_fixture_tests(&tokenizer, &fixture);
    }

    #[test]
    #[ignore]
    fn test_tinyllama_chat_template() {
        let tokenizer = load_tokenizer_from_gguf(TINYLLAMA_GGUF);
        let fixture = load_fixture(TINYLLAMA_FIXTURE);
        run_chat_template_test(&tokenizer, &fixture);
    }

    #[test]
    #[ignore]
    fn test_llama3_chat_template() {
        let tokenizer = load_tokenizer_from_gguf(LLAMA3_GGUF);
        let fixture = load_fixture(LLAMA3_FIXTURE);
        run_chat_template_test(&tokenizer, &fixture);
    }

    #[test]
    #[ignore]
    fn test_qwen25_chat_template() {
        let tokenizer = load_tokenizer_from_gguf(QWEN25_GGUF);
        let fixture = load_fixture(QWEN25_FIXTURE);
        run_chat_template_test(&tokenizer, &fixture);
    }

    #[test]
    #[ignore]
    fn test_qwen35_chat_template() {
        let tokenizer = load_tokenizer_from_gguf(QWEN35_GGUF);
        let fixture = load_fixture(QWEN35_FIXTURE);
        run_chat_template_test(&tokenizer, &fixture);
    }
}
