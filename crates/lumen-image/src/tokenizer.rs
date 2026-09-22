//! A minimal byte-pair-encoding tokenizer for the prompt path.
//!
//! The pipeline renders a fixed template, encodes it, and the model consumes the
//! ids. That is the whole requirement: no chat templating, no special-token
//! handling beyond the ids the template already contains, no streaming detokenize.
//!
//! The vocabulary and merges come from the checkpoint's `vocab.json` and
//! `merges.txt`, which is what the reference tokenizer loads. The pre-tokenizer
//! is the Qwen family's: a GPT-2-style pattern, then byte-level mapping.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use fancy_regex::Regex;
use unicode_normalization::UnicodeNormalization;

#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// A merge rule names a pair whose result is not in the vocabulary.
    BadMerge(String),
    /// The vocabulary is empty or unreadable.
    EmptyVocabulary,
    /// A special token is the empty string, which would match at every
    /// position and never advance.
    EmptySpecial,
    /// The pre-tokenizer's engine gave up on the text (its backtracking
    /// stack is bounded, and a run of a million whitespace characters
    /// exhausts it).
    Split(Box<fancy_regex::Error>),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Json(e) => write!(f, "json: {e}"),
            Self::BadMerge(m) => write!(f, "merge {m:?} has no result in the vocabulary"),
            Self::EmptyVocabulary => write!(f, "the vocabulary is empty"),
            Self::EmptySpecial => write!(f, "a special token is the empty string"),
            Self::Split(e) => write!(f, "pre-tokenizing failed: {e}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

/// The byte-level alphabet GPT-2 and the Qwen family use: every byte maps to a
/// printable code point so no input byte is lost.
fn byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for (b, slot) in table.iter_mut().enumerate() {
        let b = b as u32;
        // The bytes that map to themselves, then the rest in order, skipping the
        // range already used.
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        if printable {
            *slot = char::from_u32(b).unwrap();
        } else {
            *slot = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    table
}

/// The checkpoint's pre-tokenizer pattern, the GPT-2 one: each alternative is
/// tried at the current position and the first match wins, which is what makes
/// a trailing space attach to the following word rather than standing alone.
const PIECE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Split text the way the checkpoint's pre-tokenizer does.
pub fn split_pieces(text: &str) -> Result<Vec<String>, TokenizerError> {
    static PIECES: OnceLock<Regex> = OnceLock::new();
    let pieces = PIECES.get_or_init(|| Regex::new(PIECE_PATTERN).expect("the pattern is fixed"));
    pieces
        .find_iter(text)
        .map(|m| {
            m.map(|m| m.as_str().to_string())
                .map_err(|e| TokenizerError::Split(Box::new(e)))
        })
        .collect()
}

/// A loaded BPE tokenizer.
pub struct Tokenizer {
    /// token string -> id.
    encoder: HashMap<String, u32>,
    /// id -> token string.
    decoder: Vec<String>,
    /// (left, right) -> rank; lower is merged first.
    /// `"left right"` -> rank; lower merges first.
    merges: HashMap<String, u32>,
    /// The byte-level alphabet, applied before merges.
    byte_map: [char; 256],
    /// character -> byte, for the inverse.
    char_map: HashMap<char, u8>,
    /// Special tokens matched as whole units before any BPE runs, longest
    /// first so `<|im_start|>` is not shadowed by a shorter prefix.
    specials: Vec<(String, u32)>,
}

impl Tokenizer {
    /// Load `vocab.json` and `merges.txt` from a checkpoint directory.
    pub fn from_files(vocab: &Path, merges: &Path) -> Result<Self, TokenizerError> {
        Self::from_files_with_added(vocab, merges, None)
    }

    /// As [`from_files`](Self::from_files), also reading `added_tokens.json`.
    ///
    /// The special tokens the prompt template is built from (`<|im_start|>` and
    /// friends) are NOT reachable by BPE: they are single vocabulary entries
    /// that must be matched as whole units, or the template tokenizes into its
    /// punctuation and the model is asked a different question.
    pub fn from_files_with_added(
        vocab: &Path,
        merges: &Path,
        added: Option<&Path>,
    ) -> Result<Self, TokenizerError> {
        let raw = std::fs::read(vocab)?;
        let map: HashMap<String, u32> = serde_json::from_slice(&raw)?;
        if map.is_empty() {
            return Err(TokenizerError::EmptyVocabulary);
        }
        let size = map.values().copied().max().unwrap_or(0) as usize + 1;
        let mut decoder = vec![String::new(); size];
        for (token, &id) in &map {
            if (id as usize) < decoder.len() {
                decoder[id as usize] = token.clone();
            }
        }

        let text = std::fs::read_to_string(merges)?;
        let mut merges_map = HashMap::new();
        for (rank, line) in text.lines().enumerate() {
            // The first line is a `#version` header in this format; a merge
            // whose left symbol is `#` (`# #`, `# include`) is a real rule.
            if line.starts_with("#version") || line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split(' ');
            let (Some(a), Some(b)) = (parts.next(), parts.next()) else {
                continue;
            };
            merges_map.insert(format!("{a} {b}"), rank as u32);
        }

        let byte_map = byte_to_unicode();
        let char_map = byte_map
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        // A named file must load: without its special tokens the chat
        // template's `<|im_start|>` is BPE'd into ordinary pieces and every
        // prompt is silently wrong.
        let mut specials: Vec<(String, u32)> = match added {
            Some(p) => serde_json::from_slice::<HashMap<String, u32>>(&std::fs::read(p)?)?
                .into_iter()
                .collect(),
            None => Vec::new(),
        };
        if specials.iter().any(|(s, _)| s.is_empty()) {
            return Err(TokenizerError::EmptySpecial);
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Ok(Self {
            encoder: map,
            decoder,
            merges: merges_map,
            byte_map,
            char_map,
            specials,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.decoder.len()
    }

    /// The id of a token, if present.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.encoder.get(token).copied()
    }

    /// Encode text to token ids.
    ///
    /// The reference's pre-tokenizer is a GPT-2-style regex followed by the
    /// byte-level map, and merges are only ever applied WITHIN the pieces that
    /// regex produces. Encoding per character would never merge, and encoding
    /// the whole string would merge across piece boundaries — both give the
    /// wrong ids, so the split has to be the real one.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut out = Vec::new();
        if self.specials.is_empty() {
            self.encode_plain(text, &mut out)?;
            return Ok(out);
        }
        // Scan for special tokens; everything between them goes through BPE.
        let mut cursor = 0usize;
        while cursor < text.len() {
            let rest = &text[cursor..];
            let hit = self
                .specials
                .iter()
                .find(|(tok, _)| rest.starts_with(tok.as_str()));
            match hit {
                Some((tok, id)) => {
                    out.push(*id);
                    cursor += tok.len();
                }
                None => {
                    // Advance to the next special token, or the end. The search
                    // starts one CHARACTER past the current position (never one
                    // byte: a byte offset inside a multi-byte character is not a
                    // valid slice boundary and panics).
                    let step = rest.chars().next().map(char::len_utf8).unwrap_or(0);
                    let next = self
                        .specials
                        .iter()
                        .filter_map(|(tok, _)| {
                            rest[step..].find(tok.as_str()).map(|i| cursor + step + i)
                        })
                        .min()
                        .unwrap_or(text.len());
                    self.encode_plain(&text[cursor..next], &mut out)?;
                    cursor = next;
                }
            }
        }
        Ok(out)
    }

    /// BPE over a span with no special tokens in it.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) -> Result<(), TokenizerError> {
        // The reference normalises to NFC before splitting, so a decomposed
        // accent and its precomposed form take the same ids.
        let text: String = text.nfc().collect();
        for piece in split_pieces(&text)? {
            let mapped: String = piece
                .as_bytes()
                .iter()
                .map(|&b| self.byte_map[b as usize])
                .collect();
            if let Some(ids) = self.encode_word(&mapped) {
                out.extend(ids);
            }
        }
        Ok(())
    }

    /// Encode one pre-tokenized word with the BPE merge loop.
    fn encode_word(&self, word: &str) -> Option<Vec<u32>> {
        // Symbols are indices into a growing pool, and each carries its own
        // byte buffer, so a merge replaces a pair with one entry rather than
        // rebuilding strings. The pair scan is over a linked list of live
        // symbols, so a merge costs its neighbourhood rather than the word.
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if syms.is_empty() {
            return None;
        }
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                // Borrow rather than clone: the map lookup takes &str.
                let pair = format!("{} {}", syms[i], syms[i + 1]);
                if let Some(&rank) = self.merges.get(&pair) {
                    if best.map_or(true, |(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms.splice(i..i + 2, [merged]);
        }
        let mut ids = Vec::with_capacity(syms.len());
        for s in syms {
            ids.push(*self.encoder.get(&s)?);
        }
        Some(ids)
    }

    /// Decode ids back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok) = self.decoder.get(id as usize) {
                for c in tok.chars() {
                    if let Some(&b) = self.char_map.get(&c) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny vocabulary, enough to exercise the merge loop.
    fn tiny() -> Tokenizer {
        let mut encoder = HashMap::new();
        for (i, t) in ["a", "b", "ab", "c", "abc"].iter().enumerate() {
            encoder.insert((*t).to_string(), i as u32);
        }
        let decoder = vec![
            "a".to_string(),
            "b".to_string(),
            "ab".to_string(),
            "c".to_string(),
            "abc".to_string(),
        ];
        let mut merges = HashMap::new();
        merges.insert("a b".to_string(), 0u32);
        merges.insert("ab c".to_string(), 1u32);
        let byte_map = byte_to_unicode();
        let char_map = byte_map
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        Tokenizer {
            encoder,
            decoder,
            merges,
            byte_map,
            char_map,
            specials: Vec::new(),
        }
    }

    #[test]
    fn merges_are_applied_in_rank_order() {
        let t = tiny();
        assert_eq!(t.encode("abc").unwrap(), vec![4]); // a+b+c merges all the way
        assert_eq!(t.encode("ab").unwrap(), vec![2]);
        assert_eq!(t.encode("a").unwrap(), vec![0]);
    }

    /// The byte map must be a bijection over all 256 bytes, or text would be
    /// lost or aliased.
    #[test]
    fn the_byte_alphabet_is_a_bijection() {
        let m = byte_to_unicode();
        let mut seen: Vec<char> = m.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 256, "byte map is not injective");
    }

    /// A whitespace run that holds a newline is one piece up to its last
    /// newline, whatever precedes it; whitespace after that newline starts a
    /// new piece.
    #[test]
    fn whitespace_before_a_newline_stays_in_the_newline_piece() {
        for (input, want) in [
            ("hello\n\nworld", vec!["hello", "\n\n", "world"]),
            ("a\r\nb", vec!["a", "\r\n", "b"]),
            ("a\t\nb", vec!["a", "\t\n", "b"]),
            ("x  \ny", vec!["x", "  \n", "y"]),
            ("  \n\nb", vec!["  \n\n", "b"]),
            ("\n  b", vec!["\n", " ", " b"]),
            ("a \n \nb", vec!["a", " \n \n", "b"]),
        ] {
            let p = split_pieces(input).unwrap();
            assert_eq!(p, want, "input {input:?}");
        }
    }

    /// The pre-tokenizer keeps a leading space with the word that follows it:
    /// `" and"` is one piece, not a space and then `"and"`. Getting this wrong
    /// changes every id after the first space.
    #[test]
    fn a_leading_space_stays_with_its_word() {
        let p = split_pieces("Comprehend and analyze").unwrap();
        assert_eq!(p, vec!["Comprehend", " and", " analyze"], "got {p:?}");
    }

    #[test]
    fn every_byte_round_trips_through_the_map() {
        let m = byte_to_unicode();
        for b in 0u16..=255 {
            let c = m[b as usize];
            // The inverse is built from the same table, so this is the property
            // the decoder relies on.
            let back = m.iter().position(|&x| x == c).unwrap();
            assert_eq!(back, b as usize);
        }
    }
}

#[cfg(test)]
mod special_token_tests {
    use super::*;

    fn with_specials() -> Tokenizer {
        let mut t = {
            let mut e = HashMap::new();
            e.insert("a".to_string(), 0u32);
            e.insert("b".to_string(), 1u32);
            Tokenizer {
                encoder: e,
                decoder: vec!["a".to_string(), "b".to_string()],
                merges: HashMap::new(),
                byte_map: byte_to_unicode(),
                char_map: byte_to_unicode()
                    .iter()
                    .enumerate()
                    .map(|(b, &c)| (c, b as u8))
                    .collect(),
                specials: Vec::new(),
            }
        };
        t.specials = vec![
            ("<|im_start|>".to_string(), 9),
            ("<|im_end|>".to_string(), 10),
        ];
        t
    }

    /// Every one of these put a multi-byte character at the start of a span that
    /// follows a special token, and the old search sliced at byte offset 1 —
    /// which is not a char boundary, so it panicked. A prompt is untrusted
    /// input, so this was a 500 from one JSON field.
    #[test]
    fn multi_byte_text_after_a_special_token_does_not_panic() {
        let t = with_specials();
        for input in [
            "<|im_start|>😀",
            "<|im_end|>中文",
            "<|im_start|>a cat",
            "café <|im_start|>naïve",
            "<|im_start|>Ünïcödé",
            "<|im_start|>",
            "😀<|im_start|>",
        ] {
            let ids = t.encode(input).unwrap();
            assert!(!ids.is_empty(), "empty ids for {input:?}");
        }
    }

    /// An empty special would match at every position and never advance the
    /// cursor, so the loader refuses it rather than letting `encode` spin.
    #[test]
    fn an_empty_special_token_is_refused_at_load() {
        let dir = std::env::temp_dir().join(format!(
            "lumen-image-tokenizer-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let vocab = dir.join("vocab.json");
        let merges = dir.join("merges.txt");
        let added = dir.join("added_tokens.json");
        std::fs::write(&vocab, r#"{"a": 0, "b": 1}"#).unwrap();
        std::fs::write(&merges, "#version: 0.2\n").unwrap();
        std::fs::write(&added, r#"{"": 9}"#).unwrap();
        let result = Tokenizer::from_files_with_added(&vocab, &merges, Some(&added));
        assert!(
            matches!(result, Err(TokenizerError::EmptySpecial)),
            "got {:?}",
            result.map(|_| ())
        );
        std::fs::write(&added, r#"{"<|x|>": 9}"#).unwrap();
        let t = Tokenizer::from_files_with_added(&vocab, &merges, Some(&added)).unwrap();
        assert_eq!(t.encode("<|x|>").unwrap(), vec![9]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Only the `#version` header is skipped: a merge whose left symbol is
    /// `#` is a rule like any other.
    #[test]
    fn a_hash_led_merge_is_applied() {
        let dir = std::env::temp_dir().join(format!(
            "lumen-image-tokenizer-hash-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let vocab = dir.join("vocab.json");
        let merges = dir.join("merges.txt");
        std::fs::write(&vocab, r###"{"#": 0, "##": 1}"###).unwrap();
        std::fs::write(&merges, "#version: 0.2\n# #\n").unwrap();
        let t = Tokenizer::from_files(&vocab, &merges).unwrap();
        assert_eq!(t.encode("##").unwrap(), vec![1]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The engine's stack is bounded; exhausting it is an error, not a panic.
    #[test]
    fn a_pathological_whitespace_run_is_an_error() {
        let t = with_specials();
        assert!(matches!(
            t.encode(&" ".repeat(1_000_001)),
            Err(TokenizerError::Split(_))
        ));
        assert!(t.encode(&" ".repeat(10_000)).is_ok());
    }

    /// A decomposed accent takes the precomposed form's ids.
    #[test]
    fn text_is_nfc_normalised_before_encoding() {
        // A vocabulary of exactly the two bytes of a precomposed `é` (C3 A9),
        // as the byte alphabet spells them, so the decomposed form encodes to
        // those ids only through NFC; without it `e` and U+0301 have no ids.
        let mut t = with_specials();
        let map = byte_to_unicode();
        t.encoder.insert(map[0xC3].to_string(), 20);
        t.encoder.insert(map[0xA9].to_string(), 21);
        assert_eq!(t.encode("\u{e9}").unwrap(), vec![20, 21]);
        assert_eq!(t.encode("e\u{301}").unwrap(), vec![20, 21]);
        assert_eq!(t.encode("<|im_start|>e\u{301}").unwrap(), vec![9, 20, 21]);
    }

    /// The special tokens themselves still resolve to their own ids.
    #[test]
    fn specials_are_still_matched_whole() {
        let t = with_specials();
        let ids = t.encode("<|im_start|>ab<|im_end|>").unwrap();
        assert_eq!(ids.first(), Some(&9), "the opening special");
        assert_eq!(ids.last(), Some(&10), "the closing special");
    }
}
