//! Tokenizer data embedded in LBC v3 files.
//!
//! The tokenizer section is an optional appendix after the last layer blob.
//! Its offset, length, and CRC32 are stored in the v3 header extension.

use crate::FormatError;

/// Tokenizer data embedded in an LBC file (v3+).
#[derive(Debug, Clone)]
pub struct TokenizerSection {
    pub model_type: String,
    pub pre_tokenizer: String,
    pub tokens: Vec<String>,
    /// Token type IDs. Length may be 0 (absent) or vocab_size.
    pub token_types: Vec<u32>,
    /// Merge scores. Length may be 0 (BPE) or vocab_size (SPM).
    pub scores: Vec<f32>,
    pub merges: Vec<String>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: Option<u32>,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
    pub add_space_prefix: bool,
    pub chat_template: Option<String>,
}

impl TokenizerSection {
    /// Serialize to binary format (little-endian).
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64 * 1024);

        // model_type: u32_len + bytes
        write_string(&mut buf, &self.model_type);
        // pre_tokenizer: u32_len + bytes
        write_string(&mut buf, &self.pre_tokenizer);

        // vocab_size: u32
        let vocab_size = self.tokens.len() as u32;
        buf.extend_from_slice(&vocab_size.to_le_bytes());
        // num_merges: u32
        let num_merges = self.merges.len() as u32;
        buf.extend_from_slice(&num_merges.to_le_bytes());

        // bos_token_id: u32
        buf.extend_from_slice(&self.bos_token_id.to_le_bytes());
        // eos_token_id: u32
        buf.extend_from_slice(&self.eos_token_id.to_le_bytes());

        // has_pad_token: u8, pad_token_id: u32
        match self.pad_token_id {
            Some(id) => {
                buf.push(1);
                buf.extend_from_slice(&id.to_le_bytes());
            }
            None => {
                buf.push(0);
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
        }

        // add_bos_token: u8
        buf.push(self.add_bos_token as u8);
        // add_eos_token: u8
        buf.push(self.add_eos_token as u8);
        // add_space_prefix: u8
        buf.push(self.add_space_prefix as u8);

        // chat_template: u32_len + bytes (len=0 means absent)
        match &self.chat_template {
            Some(tmpl) => write_string(&mut buf, tmpl),
            None => buf.extend_from_slice(&0u32.to_le_bytes()),
        }

        // tokens: [u32_len + bytes] x vocab_size
        for token in &self.tokens {
            write_string(&mut buf, token);
        }

        // num_token_types: u32
        let num_token_types = self.token_types.len() as u32;
        buf.extend_from_slice(&num_token_types.to_le_bytes());
        // token_types: [u32] x num_token_types
        for &tt in &self.token_types {
            buf.extend_from_slice(&tt.to_le_bytes());
        }

        // num_scores: u32
        let num_scores = self.scores.len() as u32;
        buf.extend_from_slice(&num_scores.to_le_bytes());
        // scores: [f32] x num_scores
        for &s in &self.scores {
            buf.extend_from_slice(&s.to_le_bytes());
        }

        // merges: [u32_len + bytes] x num_merges
        for merge in &self.merges {
            write_string(&mut buf, merge);
        }

        buf
    }

    /// Parse from binary format (little-endian).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut pos = 0usize;

        let model_type = read_string(data, &mut pos)?;
        let pre_tokenizer = read_string(data, &mut pos)?;

        let vocab_size = read_u32(data, &mut pos)? as usize;
        let num_merges = read_u32(data, &mut pos)? as usize;

        // Sanity bounds: each token/merge needs at minimum 4 bytes (u32 length prefix).
        // Reject absurd values early to prevent OOM from Vec::with_capacity.
        let max_possible = data.len() / 4;
        if vocab_size > max_possible || num_merges > max_possible {
            return Err(FormatError::UnexpectedEof {
                needed: (vocab_size + num_merges) as u64 * 4,
                available: data.len() as u64,
            });
        }

        let bos_token_id = read_u32(data, &mut pos)?;
        let eos_token_id = read_u32(data, &mut pos)?;

        let has_pad = read_u8(data, &mut pos)?;
        let pad_raw = read_u32(data, &mut pos)?;
        let pad_token_id = if has_pad != 0 { Some(pad_raw) } else { None };

        let add_bos_token = read_u8(data, &mut pos)? != 0;
        let add_eos_token = read_u8(data, &mut pos)? != 0;
        let add_space_prefix = read_u8(data, &mut pos)? != 0;

        // chat_template: u32_len + bytes, len=0 means absent
        let tmpl_len = read_u32(data, &mut pos)? as usize;
        let chat_template = if tmpl_len == 0 {
            None
        } else {
            // Rewind pos by 4 to let read_string re-read the length
            pos -= 4;
            Some(read_string(data, &mut pos)?)
        };

        // tokens
        let mut tokens = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            tokens.push(read_string(data, &mut pos)?);
        }

        // token_types
        let num_token_types = read_u32(data, &mut pos)? as usize;
        if num_token_types > data.len() / 4 {
            return Err(FormatError::UnexpectedEof {
                needed: num_token_types as u64 * 4,
                available: (data.len() - pos) as u64,
            });
        }
        let mut token_types = Vec::with_capacity(num_token_types);
        for _ in 0..num_token_types {
            token_types.push(read_u32(data, &mut pos)?);
        }

        // scores
        let num_scores = read_u32(data, &mut pos)? as usize;
        if num_scores > data.len() / 4 {
            return Err(FormatError::UnexpectedEof {
                needed: num_scores as u64 * 4,
                available: (data.len() - pos) as u64,
            });
        }
        let mut scores = Vec::with_capacity(num_scores);
        for _ in 0..num_scores {
            scores.push(read_f32(data, &mut pos)?);
        }

        // merges
        let mut merges = Vec::with_capacity(num_merges);
        for _ in 0..num_merges {
            merges.push(read_string(data, &mut pos)?);
        }

        Ok(Self {
            model_type,
            pre_tokenizer,
            tokens,
            token_types,
            scores,
            merges,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            add_bos_token,
            add_eos_token,
            add_space_prefix,
            chat_template,
        })
    }
}

// ---------- Binary helpers ----------

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn ensure(data: &[u8], pos: usize, n: usize) -> Result<(), FormatError> {
    let remaining = data.len().saturating_sub(pos);
    if n > remaining {
        Err(FormatError::UnexpectedEof {
            needed: pos as u64 + n as u64,
            available: data.len() as u64,
        })
    } else {
        Ok(())
    }
}

fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, FormatError> {
    ensure(data, *pos, 1)?;
    let v = data[*pos];
    *pos += 1;
    Ok(v)
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, FormatError> {
    ensure(data, *pos, 4)?;
    let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn read_f32(data: &[u8], pos: &mut usize) -> Result<f32, FormatError> {
    ensure(data, *pos, 4)?;
    let v = f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String, FormatError> {
    let len = read_u32(data, pos)? as usize;
    ensure(data, *pos, len)?;
    let s = std::str::from_utf8(&data[*pos..*pos + len]).map_err(|e| {
        FormatError::UnsupportedQuantization(format!("invalid UTF-8 in tokenizer string: {e}"))
    })?;
    *pos += len;
    Ok(s.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tokenizer() -> TokenizerSection {
        TokenizerSection {
            model_type: "llama".to_string(),
            pre_tokenizer: "byte_level".to_string(),
            tokens: vec!["hello".into(), "world".into(), "<s>".into(), "</s>".into()],
            token_types: vec![1, 1, 3, 3],
            scores: vec![-1.0, -2.0, 0.0, 0.0],
            merges: vec!["h e".into(), "l o".into()],
            bos_token_id: 2,
            eos_token_id: 3,
            pad_token_id: Some(0),
            add_bos_token: true,
            add_eos_token: false,
            add_space_prefix: true,
            chat_template: Some("{% for msg in messages %}...{% endfor %}".to_string()),
        }
    }

    #[test]
    fn roundtrip_serialize_parse() {
        let tok = make_test_tokenizer();
        let bytes = tok.serialize();
        let parsed = TokenizerSection::parse(&bytes).unwrap();

        assert_eq!(parsed.model_type, tok.model_type);
        assert_eq!(parsed.pre_tokenizer, tok.pre_tokenizer);
        assert_eq!(parsed.tokens, tok.tokens);
        assert_eq!(parsed.token_types, tok.token_types);
        assert_eq!(parsed.scores, tok.scores);
        assert_eq!(parsed.merges, tok.merges);
        assert_eq!(parsed.bos_token_id, tok.bos_token_id);
        assert_eq!(parsed.eos_token_id, tok.eos_token_id);
        assert_eq!(parsed.pad_token_id, tok.pad_token_id);
        assert_eq!(parsed.add_bos_token, tok.add_bos_token);
        assert_eq!(parsed.add_eos_token, tok.add_eos_token);
        assert_eq!(parsed.add_space_prefix, tok.add_space_prefix);
        assert_eq!(parsed.chat_template, tok.chat_template);
    }

    #[test]
    fn roundtrip_no_pad_no_template() {
        let tok = TokenizerSection {
            model_type: "gpt2".to_string(),
            pre_tokenizer: "".to_string(),
            tokens: vec!["a".into(), "b".into()],
            token_types: vec![],
            scores: vec![],
            merges: vec!["a b".into()],
            bos_token_id: 0,
            eos_token_id: 1,
            pad_token_id: None,
            add_bos_token: false,
            add_eos_token: true,
            add_space_prefix: false,
            chat_template: None,
        };
        let bytes = tok.serialize();
        let parsed = TokenizerSection::parse(&bytes).unwrap();

        assert_eq!(parsed.pad_token_id, None);
        assert_eq!(parsed.chat_template, None);
        assert_eq!(parsed.token_types.len(), 0);
        assert_eq!(parsed.scores.len(), 0);
        assert_eq!(parsed.tokens, vec!["a", "b"]);
        assert_eq!(parsed.merges, vec!["a b"]);
    }

    #[test]
    fn roundtrip_empty_vocab() {
        let tok = TokenizerSection {
            model_type: "".to_string(),
            pre_tokenizer: "".to_string(),
            tokens: vec![],
            token_types: vec![],
            scores: vec![],
            merges: vec![],
            bos_token_id: 0,
            eos_token_id: 0,
            pad_token_id: None,
            add_bos_token: false,
            add_eos_token: false,
            add_space_prefix: false,
            chat_template: None,
        };
        let bytes = tok.serialize();
        let parsed = TokenizerSection::parse(&bytes).unwrap();

        assert_eq!(parsed.tokens.len(), 0);
        assert_eq!(parsed.merges.len(), 0);
        assert_eq!(parsed.scores.len(), 0);
        assert_eq!(parsed.token_types.len(), 0);
    }

    #[test]
    fn roundtrip_large_vocab() {
        let tokens: Vec<String> = (0..10000).map(|i| format!("token_{i}")).collect();
        let scores: Vec<f32> = (0..10000).map(|i| -(i as f32)).collect();
        let token_types: Vec<u32> = (0..10000).map(|i| if i < 3 { 3 } else { 1 }).collect();
        let merges: Vec<String> = (0..5000).map(|i| format!("tok_{} en_{}", i, i + 1)).collect();

        let tok = TokenizerSection {
            model_type: "llama".to_string(),
            pre_tokenizer: "byte_level".to_string(),
            tokens,
            token_types,
            scores,
            merges,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: Some(0),
            add_bos_token: true,
            add_eos_token: false,
            add_space_prefix: true,
            chat_template: Some("template".to_string()),
        };
        let bytes = tok.serialize();
        let parsed = TokenizerSection::parse(&bytes).unwrap();

        assert_eq!(parsed.tokens.len(), 10000);
        assert_eq!(parsed.scores.len(), 10000);
        assert_eq!(parsed.token_types.len(), 10000);
        assert_eq!(parsed.merges.len(), 5000);
        assert_eq!(parsed.tokens[9999], "token_9999");
        assert_eq!(parsed.scores[42], -42.0);
        assert_eq!(parsed.merges[4999], "tok_4999 en_5000");
    }

    #[test]
    fn parse_truncated_data() {
        let tok = make_test_tokenizer();
        let bytes = tok.serialize();
        // Truncate at various points
        for cut in [0, 4, 10, 30, bytes.len() / 2] {
            let result = TokenizerSection::parse(&bytes[..cut]);
            assert!(result.is_err(), "should fail with {} bytes", cut);
        }
    }

    #[test]
    fn roundtrip_unicode_tokens() {
        let tok = TokenizerSection {
            model_type: "llama".to_string(),
            pre_tokenizer: "".to_string(),
            tokens: vec![
                "\u{00e9}".into(),   // e-acute
                "\u{4e16}\u{754c}".into(), // Chinese: "world"
                "\u{1f600}".into(),  // emoji: grinning face
            ],
            token_types: vec![],
            scores: vec![],
            merges: vec![],
            bos_token_id: 0,
            eos_token_id: 0,
            pad_token_id: None,
            add_bos_token: false,
            add_eos_token: false,
            add_space_prefix: false,
            chat_template: None,
        };
        let bytes = tok.serialize();
        let parsed = TokenizerSection::parse(&bytes).unwrap();
        assert_eq!(parsed.tokens, tok.tokens);
    }
}
