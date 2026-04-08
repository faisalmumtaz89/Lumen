//! Tokenizer data extracted from GGUF files.

use crate::gguf::GgufFile;

/// All tokenizer metadata from a GGUF file, ready for embedding in LBC.
#[derive(Debug, Clone)]
pub struct TokenizerData {
    /// Tokenizer model type: "gpt2" (BPE/tiktoken) or "llama" (SentencePiece BPE).
    pub model_type: String,
    /// Pre-tokenizer identifier: "llama-bpe", "qwen2", "default", etc.
    pub pre_tokenizer: String,
    /// Full vocabulary — one string per token ID.
    pub tokens: Vec<String>,
    /// Token type per ID: 1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte.
    pub token_types: Vec<u32>,
    /// Merge priority scores (SentencePiece BPE models). Empty for tiktoken models.
    pub scores: Vec<f32>,
    /// BPE merge rules, e.g. "a b" meaning "merge 'a' and 'b'". Empty for score-only SPM.
    pub merges: Vec<String>,
    /// Beginning-of-sequence token ID.
    pub bos_token_id: u32,
    /// End-of-sequence token ID.
    pub eos_token_id: u32,
    /// Padding token ID (if any).
    pub pad_token_id: Option<u32>,
    /// Whether to automatically prepend BOS token.
    pub add_bos_token: bool,
    /// Whether to automatically append EOS token.
    pub add_eos_token: bool,
    /// Whether to prepend a leading space (SentencePiece models).
    pub add_space_prefix: bool,
    /// Chat template string (Jinja2 format, stored for future use).
    pub chat_template: Option<String>,
}

/// Extract tokenizer data from a GGUF file. Returns None if no tokenizer data is present.
pub fn extract_tokenizer(gguf: &GgufFile) -> Option<TokenizerData> {
    // The tokens array is the minimum requirement for a tokenizer.
    let token_strs = gguf.get_string_array("tokenizer.ggml.tokens")?;
    let tokens: Vec<String> = token_strs.into_iter().map(|s| s.to_owned()).collect();

    let model_type = gguf
        .get_string("tokenizer.ggml.model")
        .unwrap_or("gpt2")
        .to_owned();

    let pre_tokenizer = gguf
        .get_string("tokenizer.ggml.pre")
        .unwrap_or("default")
        .to_owned();

    let token_types = gguf
        .get_u32_array("tokenizer.ggml.token_type")
        .unwrap_or_default();

    let scores = gguf
        .get_f32_array("tokenizer.ggml.scores")
        .unwrap_or_default();

    let merges = gguf
        .get_string_array("tokenizer.ggml.merges")
        .map(|v| v.into_iter().map(|s| s.to_owned()).collect())
        .unwrap_or_default();

    let bos_token_id = gguf.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(0);
    let eos_token_id = gguf.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(0);
    let pad_token_id = gguf.get_u32("tokenizer.ggml.padding_token_id");

    // Boolean fields: use get_metadata + pattern match since there's no get_bool.
    let add_bos_token = match gguf.get_metadata("tokenizer.ggml.add_bos_token") {
        Some(crate::gguf::GgufValue::Bool(v)) => *v,
        _ => true, // default: add BOS
    };
    let add_eos_token = match gguf.get_metadata("tokenizer.ggml.add_eos_token") {
        Some(crate::gguf::GgufValue::Bool(v)) => *v,
        _ => false, // default: don't add EOS
    };
    let add_space_prefix = match gguf.get_metadata("tokenizer.ggml.add_space_prefix") {
        Some(crate::gguf::GgufValue::Bool(v)) => *v,
        _ => model_type == "llama", // SPM models default to adding prefix space
    };

    let chat_template = gguf
        .get_string("tokenizer.chat_template")
        .map(|s| s.to_owned());

    Some(TokenizerData {
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufBuilder;

    /// Build a minimal GGUF with only the tokens array (minimum for extraction).
    fn build_minimal_tokenizer_gguf() -> Vec<u8> {
        let mut builder = GgufBuilder::new();
        builder.add_string_array("tokenizer.ggml.tokens", &["<s>", "</s>", "hello"]);
        builder.build()
    }

    /// Build a full GGUF with all tokenizer fields populated.
    fn build_full_tokenizer_gguf() -> Vec<u8> {
        let mut builder = GgufBuilder::new();
        builder.add_string("tokenizer.ggml.model", "llama");
        builder.add_string("tokenizer.ggml.pre", "llama-bpe");
        builder.add_string_array(
            "tokenizer.ggml.tokens",
            &["<s>", "</s>", "<unk>", "hello", "world"],
        );
        builder.add_u32_array("tokenizer.ggml.token_type", &[3, 3, 2, 1, 1]);
        builder.add_f32_array("tokenizer.ggml.scores", &[0.0, 0.0, 0.0, -1.0, -2.0]);
        builder.add_string_array("tokenizer.ggml.merges", &["h e", "l l", "o _"]);
        builder.add_u32("tokenizer.ggml.bos_token_id", 0);
        builder.add_u32("tokenizer.ggml.eos_token_id", 1);
        builder.add_u32("tokenizer.ggml.padding_token_id", 2);
        builder.add_bool("tokenizer.ggml.add_bos_token", true);
        builder.add_bool("tokenizer.ggml.add_eos_token", true);
        builder.add_bool("tokenizer.ggml.add_space_prefix", false);
        builder.add_string("tokenizer.chat_template", "{{ bos_token }}{{ content }}");
        builder.build()
    }

    #[test]
    fn extract_minimal_tokenizer() {
        let bytes = build_minimal_tokenizer_gguf();
        let gguf = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        let tok = extract_tokenizer(&gguf).unwrap();

        assert_eq!(tok.tokens, vec!["<s>", "</s>", "hello"]);
        // Defaults when fields are missing.
        assert_eq!(tok.model_type, "gpt2");
        assert_eq!(tok.pre_tokenizer, "default");
        assert!(tok.token_types.is_empty());
        assert!(tok.scores.is_empty());
        assert!(tok.merges.is_empty());
        assert_eq!(tok.bos_token_id, 0);
        assert_eq!(tok.eos_token_id, 0);
        assert!(tok.pad_token_id.is_none());
        assert!(tok.add_bos_token); // default true
        assert!(!tok.add_eos_token); // default false
        assert!(!tok.add_space_prefix); // model_type=gpt2, default false
        assert!(tok.chat_template.is_none());
    }

    #[test]
    fn extract_returns_none_without_tokens() {
        let builder = GgufBuilder::new();
        let bytes = builder.build();
        let gguf = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        assert!(extract_tokenizer(&gguf).is_none());
    }

    #[test]
    fn extract_returns_none_with_only_model_type() {
        let mut builder = GgufBuilder::new();
        builder.add_string("tokenizer.ggml.model", "llama");
        let bytes = builder.build();
        let gguf = GgufFile::parse(&mut bytes.as_slice()).unwrap();

        assert!(extract_tokenizer(&gguf).is_none());
    }

    #[test]
    fn extract_full_tokenizer() {
        let bytes = build_full_tokenizer_gguf();
        let gguf = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        let tok = extract_tokenizer(&gguf).unwrap();

        assert_eq!(tok.model_type, "llama");
        assert_eq!(tok.pre_tokenizer, "llama-bpe");
        assert_eq!(
            tok.tokens,
            vec!["<s>", "</s>", "<unk>", "hello", "world"]
        );
        assert_eq!(tok.token_types, vec![3, 3, 2, 1, 1]);
        assert_eq!(tok.scores, vec![0.0, 0.0, 0.0, -1.0, -2.0]);
        assert_eq!(tok.merges, vec!["h e", "l l", "o _"]);
        assert_eq!(tok.bos_token_id, 0);
        assert_eq!(tok.eos_token_id, 1);
        assert_eq!(tok.pad_token_id, Some(2));
        assert!(tok.add_bos_token);
        assert!(tok.add_eos_token);
        assert!(!tok.add_space_prefix);
        assert_eq!(
            tok.chat_template.as_deref(),
            Some("{{ bos_token }}{{ content }}")
        );
    }

    #[test]
    fn extract_llama_model_defaults_space_prefix_true() {
        let mut builder = GgufBuilder::new();
        builder.add_string("tokenizer.ggml.model", "llama");
        builder.add_string_array("tokenizer.ggml.tokens", &["<s>", "</s>"]);
        // Don't set add_space_prefix -- should default to true for "llama" model.
        let bytes = builder.build();
        let gguf = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        let tok = extract_tokenizer(&gguf).unwrap();

        assert_eq!(tok.model_type, "llama");
        assert!(tok.add_space_prefix);
    }

    #[test]
    fn extract_gpt2_model_defaults_space_prefix_false() {
        let mut builder = GgufBuilder::new();
        builder.add_string("tokenizer.ggml.model", "gpt2");
        builder.add_string_array("tokenizer.ggml.tokens", &["<s>", "</s>"]);
        let bytes = builder.build();
        let gguf = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        let tok = extract_tokenizer(&gguf).unwrap();

        assert_eq!(tok.model_type, "gpt2");
        assert!(!tok.add_space_prefix);
    }

    #[test]
    fn extract_explicit_bools_override_defaults() {
        let mut builder = GgufBuilder::new();
        builder.add_string_array("tokenizer.ggml.tokens", &["a", "b"]);
        builder.add_bool("tokenizer.ggml.add_bos_token", false);
        builder.add_bool("tokenizer.ggml.add_eos_token", true);
        builder.add_bool("tokenizer.ggml.add_space_prefix", true);
        let bytes = builder.build();
        let gguf = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        let tok = extract_tokenizer(&gguf).unwrap();

        assert!(!tok.add_bos_token); // overridden from default true
        assert!(tok.add_eos_token); // overridden from default false
        assert!(tok.add_space_prefix); // explicit true, model_type=gpt2
    }

    #[test]
    fn extract_missing_optional_arrays() {
        let mut builder = GgufBuilder::new();
        builder.add_string("tokenizer.ggml.model", "llama");
        builder.add_string_array("tokenizer.ggml.tokens", &["<s>", "</s>", "x"]);
        // No scores, merges, or token_types set.
        let bytes = builder.build();
        let gguf = GgufFile::parse(&mut bytes.as_slice()).unwrap();
        let tok = extract_tokenizer(&gguf).unwrap();

        assert!(tok.token_types.is_empty());
        assert!(tok.scores.is_empty());
        assert!(tok.merges.is_empty());
    }
}
