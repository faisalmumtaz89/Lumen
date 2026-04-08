//! Model registry: resolve preset model names to GGUF sources and tokenizer IDs.
//!
//! Reads `model_registry.toml` (embedded at compile time) and provides lookup
//! by canonical key or alias.

use std::collections::HashMap;

/// Embedded registry TOML (compiled into the binary).
const REGISTRY_TOML: &str = include_str!("../../../model_registry.toml");

/// A resolved model entry from the registry.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub key: String,
    pub display_name: String,
    pub architecture: String,
    pub parameters: String,
    pub tokenizer: String,
    pub gguf_files: HashMap<String, GgufSource>,
}

/// Source location for a GGUF file on HuggingFace.
#[derive(Debug, Clone)]
pub struct GgufSource {
    pub repo: String,
    pub file: String,
}

/// The parsed model registry.
pub struct Registry {
    models: HashMap<String, ModelEntry>,
    aliases: HashMap<String, String>,
    default_quant: String,
}

impl Registry {
    /// Get the default quantization (e.g. "Q8_0").
    pub fn default_quant(&self) -> &str {
        &self.default_quant
    }

    /// Resolve a model name (canonical key or alias) to a ModelEntry.
    pub fn resolve(&self, name: &str) -> Option<&ModelEntry> {
        // Try direct lookup first.
        if let Some(entry) = self.models.get(name) {
            return Some(entry);
        }
        // Try alias.
        if let Some(canonical) = self.aliases.get(name) {
            return self.models.get(canonical.as_str());
        }
        None
    }

    /// Get the default model entry.
    pub fn default_model(&self) -> &ModelEntry {
        self.models.get("qwen2-5-3b").expect("default model qwen2-5-3b must exist in registry")
    }

    /// List all model entries (sorted by key).
    pub fn list(&self) -> Vec<&ModelEntry> {
        let mut entries: Vec<_> = self.models.values().collect();
        entries.sort_by_key(|e| &e.key);
        entries
    }

    /// Return all alias keys (for fuzzy-matching / suggestions).
    pub fn alias_keys(&self) -> Vec<&str> {
        self.aliases.keys().map(|s| s.as_str()).collect()
    }
}

/// Load and parse the embedded model registry.
pub fn load_registry() -> Registry {
    let table: toml::Table = REGISTRY_TOML.parse().expect("embedded model_registry.toml must be valid TOML");

    let meta = table.get("meta").and_then(|v| v.as_table()).expect("registry must have [meta]");
    let default_quant = meta.get("default_quant")
        .and_then(|v| v.as_str())
        .unwrap_or("Q8_0")
        .to_owned();

    // Parse models.
    let mut models = HashMap::new();
    if let Some(models_table) = table.get("models").and_then(|v| v.as_table()) {
        for (key, value) in models_table {
            let model_table = match value.as_table() {
                Some(t) => t,
                None => continue,
            };
            let display_name = model_table.get("display_name")
                .and_then(|v| v.as_str()).unwrap_or(key).to_owned();
            let architecture = model_table.get("architecture")
                .and_then(|v| v.as_str()).unwrap_or("").to_owned();
            let parameters = model_table.get("parameters")
                .and_then(|v| v.as_str()).unwrap_or("").to_owned();
            let tokenizer = model_table.get("tokenizer")
                .and_then(|v| v.as_str()).unwrap_or("").to_owned();

            let mut gguf_files = HashMap::new();
            if let Some(gguf_table) = model_table.get("gguf_files").and_then(|v| v.as_table()) {
                for (quant, source_val) in gguf_table {
                    if let Some(source_table) = source_val.as_table() {
                        let repo = source_table.get("repo")
                            .and_then(|v| v.as_str()).unwrap_or("").to_owned();
                        let file = source_table.get("file")
                            .and_then(|v| v.as_str()).unwrap_or("").to_owned();
                        gguf_files.insert(quant.clone(), GgufSource { repo, file });
                    }
                }
            }

            models.insert(key.clone(), ModelEntry {
                key: key.clone(),
                display_name,
                architecture,
                parameters,
                tokenizer,
                gguf_files,
            });
        }
    }

    // Parse aliases.
    let mut aliases = HashMap::new();
    if let Some(aliases_table) = table.get("aliases").and_then(|v| v.as_table()) {
        for (alias, target) in aliases_table {
            if let Some(target_str) = target.as_str() {
                aliases.insert(alias.clone(), target_str.to_owned());
            }
        }
    }

    Registry { models, aliases, default_quant }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_registry_succeeds() {
        let reg = load_registry();
        assert!(!reg.models.is_empty(), "registry must have at least one model");
        assert_eq!(reg.default_quant(), "Q8_0");
    }

    #[test]
    fn resolve_canonical_key() {
        let reg = load_registry();
        let entry = reg.resolve("qwen2-5-3b").expect("qwen2-5-3b must exist");
        assert_eq!(entry.display_name, "Qwen2.5 3B Instruct");
        assert_eq!(entry.architecture, "qwen2");
        assert_eq!(entry.tokenizer, "Qwen/Qwen2.5-3B-Instruct");
        assert!(entry.gguf_files.contains_key("Q8_0"));
    }

    #[test]
    fn resolve_alias() {
        let reg = load_registry();
        let entry = reg.resolve("llama-8b").expect("alias llama-8b must resolve");
        assert_eq!(entry.key, "llama-3-1-8b");
        assert_eq!(entry.display_name, "Llama 3.1 8B Instruct");
    }

    #[test]
    fn resolve_qwen_dot_alias() {
        let reg = load_registry();
        let entry = reg.resolve("qwen2.5-3b").expect("alias qwen2.5-3b must resolve");
        assert_eq!(entry.key, "qwen2-5-3b");
    }

    #[test]
    fn resolve_tinyllama_alias() {
        let reg = load_registry();
        let entry = reg.resolve("tinyllama").expect("alias tinyllama must resolve");
        assert_eq!(entry.key, "tinyllama-1-1b");
        assert_eq!(entry.architecture, "llama");
    }

    #[test]
    fn resolve_nonexistent_returns_none() {
        let reg = load_registry();
        assert!(reg.resolve("nonexistent-model").is_none());
    }

    #[test]
    fn default_model_is_qwen25_3b() {
        let reg = load_registry();
        let default = reg.default_model();
        assert_eq!(default.key, "qwen2-5-3b");
    }

    #[test]
    fn list_returns_all_models() {
        let reg = load_registry();
        let list = reg.list();
        // 7 models: llama-3-1-8b, mistral-7b-v0-3, qwen2-5-3b, qwen2-5-7b, qwen2-5-14b, qwen3-5-9b, tinyllama-1-1b
        assert_eq!(list.len(), 7, "expected 7 models, got {}", list.len());
    }

    #[test]
    fn tinyllama_has_gguf_files() {
        let reg = load_registry();
        let entry = reg.resolve("tinyllama-1-1b").expect("tinyllama must exist");
        assert!(entry.gguf_files.contains_key("Q8_0"));
        assert!(entry.gguf_files.contains_key("Q4_0"));
        let q8 = &entry.gguf_files["Q8_0"];
        assert!(q8.repo.contains("TinyLlama"));
        assert!(q8.file.contains("Q8_0"));
    }

    #[test]
    fn all_models_have_tokenizer() {
        let reg = load_registry();
        for entry in reg.list() {
            assert!(!entry.tokenizer.is_empty(), "model {} has no tokenizer", entry.key);
        }
    }
}
