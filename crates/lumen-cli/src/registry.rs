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

/// Source location for a GGUF (single-file or multi-shard) on HuggingFace.
///
/// `files` is the list of shard filenames in the repo. For single-file GGUFs
/// this is a one-element vector. For multi-shard GGUFs (e.g. BF16 Qwen3.5-MoE
/// 35B-A3B which ships as `*-00001-of-00002.gguf` + `*-00002-of-00002.gguf`)
/// every shard filename is listed here so the CLI can download all siblings
/// before invoking the converter.
///
/// The `file()` accessor returns the canonical first/primary shard filename
/// (whichever the converter should be pointed at -- the multi-shard reader
/// auto-discovers siblings from any one shard's path, but the convention is
/// to point at shard 1).
#[derive(Debug, Clone)]
pub struct GgufSource {
    pub repo: String,
    pub files: Vec<String>,
}

impl GgufSource {
    /// Returns the primary (first) shard filename. For single-file GGUFs this
    /// is the file itself; for multi-shard GGUFs it's the shard the converter
    /// should be pointed at (auto-discovery will locate siblings).
    pub fn file(&self) -> &str {
        self.files
            .first()
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// `true` iff this source contains more than one shard file.
    pub fn is_multi_shard(&self) -> bool {
        self.files.len() > 1
    }
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
        self.models.get("qwen3-5-9b").expect("default model qwen3-5-9b must exist in registry")
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
                        // Accept both single-file (`file = "name.gguf"`) and
                        // multi-shard (`files = ["shard1.gguf", "shard2.gguf"]`)
                        // forms. Multi-shard wins if both are present.
                        let files = if let Some(arr) = source_table.get("files")
                            .and_then(|v| v.as_array())
                        {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_owned()))
                                .collect::<Vec<String>>()
                        } else if let Some(single) = source_table.get("file").and_then(|v| v.as_str()) {
                            vec![single.to_owned()]
                        } else {
                            Vec::new()
                        };
                        if files.is_empty() {
                            // Skip malformed entries silently to preserve the
                            // permissive registry-loader contract.
                            continue;
                        }
                        gguf_files.insert(quant.clone(), GgufSource { repo, files });
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
        let entry = reg.resolve("qwen3-5-9b").expect("qwen3-5-9b must exist");
        assert_eq!(entry.display_name, "Qwen3.5 9B");
        assert_eq!(entry.architecture, "qwen35");
        assert_eq!(entry.tokenizer, "Qwen/Qwen3.5-9B");
        assert!(entry.gguf_files.contains_key("Q8_0"));
    }

    #[test]
    fn resolve_qwen35_dot_alias() {
        let reg = load_registry();
        let entry = reg.resolve("qwen3.5-9b").expect("alias qwen3.5-9b must resolve");
        assert_eq!(entry.key, "qwen3-5-9b");
        assert_eq!(entry.architecture, "qwen35");
    }

    #[test]
    fn resolve_nonexistent_returns_none() {
        let reg = load_registry();
        assert!(reg.resolve("nonexistent-model").is_none());
    }

    #[test]
    fn resolve_legacy_alias_returns_none() {
        // Regression test for the Qwen3.5-only registry: the legacy
        // aliases `llama-8b`, `tinyllama`, `qwen2.5-3b`, `mistral-7b`
        // must no longer resolve. Callers should receive `None` (and the
        // CLI will surface a clear "unsupported model" error).
        let reg = load_registry();
        for legacy in &["llama-8b", "tinyllama", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b", "mistral-7b"] {
            assert!(reg.resolve(legacy).is_none(), "{legacy} must not resolve");
        }
    }

    #[test]
    fn default_model_is_qwen35_9b() {
        let reg = load_registry();
        let default = reg.default_model();
        assert_eq!(default.key, "qwen3-5-9b");
        assert_eq!(default.architecture, "qwen35");
    }

    #[test]
    fn list_returns_supported_qwen35_models() {
        // Post- (2026-05-26): registry contains the Qwen3.5 family that
        // has a validated runtime path. Dense Qwen3.5-9B was the original
        // baseline; Qwen3.5-MoE-35B-A3B (qwen35moe arch) shipped with
        // after the batched-MoE correctness fix.
        let reg = load_registry();
        let list = reg.list();
        assert!(
            list.len() >= 1,
            "expected at least 1 model, got {}",
            list.len(),
        );
        let keys: Vec<&str> = list.iter().map(|e| e.key.as_str()).collect();
        assert!(keys.contains(&"qwen3-5-9b"), "dense baseline missing: {:?}", keys);
        // qwen35moe entry is asserted by `resolve_qwen35moe_canonical_key` below
        // so this list test simply enforces the family lock.
        let archs: std::collections::HashSet<&str> = list.iter().map(|e| e.architecture.as_str()).collect();
        for arch in &archs {
            assert!(
                matches!(*arch, "qwen35" | "qwen35moe"),
                "registry contains non-Qwen3.5 arch {}",
                arch,
            );
        }
    }

    #[test]
    fn resolve_qwen35moe_canonical_key() {
        // P2-1 ships qwen35moe runtime path. The 35B-A3B
        // checkpoint is the validated MoE production model.
        let reg = load_registry();
        let entry = reg
            .resolve("qwen3-5-moe-35b-a3b")
            .expect("qwen3-5-moe-35b-a3b must exist ");
        assert_eq!(entry.architecture, "qwen35moe");
        assert!(entry.gguf_files.contains_key("Q8_0"));
    }

    #[test]
    fn resolve_qwen35moe_dot_alias() {
        let reg = load_registry();
        let entry = reg
            .resolve("qwen3.5-moe-35b-a3b")
            .expect("alias qwen3.5-moe-35b-a3b must resolve");
        assert_eq!(entry.key, "qwen3-5-moe-35b-a3b");
        assert_eq!(entry.architecture, "qwen35moe");
    }

    #[test]
    fn qwen35_has_gguf_files() {
        let reg = load_registry();
        let entry = reg.resolve("qwen3-5-9b").expect("qwen3-5-9b must exist");
        assert!(entry.gguf_files.contains_key("Q8_0"));
        let q8 = &entry.gguf_files["Q8_0"];
        assert!(q8.repo.contains("Qwen3.5"), "Q8_0 repo should reference Qwen3.5: {}", q8.repo);
        assert!(q8.file().contains("Q8_0"));
        assert!(!q8.is_multi_shard(), "Q8_0 dense should be single-shard");
    }

    #[test]
    fn qwen35_moe_bf16_is_multi_shard() {
        // BF16 Qwen3.5-MoE 35B-A3B ships as a 2-shard split GGUF from HF
        // (the BF16 weights total ~68 GB, which exceeds the single-file
        // convention). The registry MUST declare both shard filenames so the
        // CLI can download them before invoking the multi-shard reader.
        let reg = load_registry();
        let entry = reg
            .resolve("qwen3-5-moe-35b-a3b")
            .expect("qwen3-5-moe-35b-a3b must exist");
        let bf16 = entry
            .gguf_files
            .get("BF16")
            .expect("BF16 quant must be declared for qwen3-5-moe-35b-a3b");
        assert!(bf16.is_multi_shard(), "BF16 entry must list multiple shard files");
        assert!(bf16.files.len() >= 2, "BF16 must have at least 2 shard files");
        for f in &bf16.files {
            // bartowski's repo uses lowercase `bf16` in the filename; other
            // providers may use uppercase. Accept either casing.
            let lower = f.to_ascii_lowercase();
            assert!(
                lower.contains("bf16") && lower.ends_with(".gguf"),
                "BF16 shard filename should look like a BF16 GGUF: {f}"
            );
            assert!(
                f.contains("-of-"),
                "BF16 shard filename should match the *-of-* pattern: {f}"
            );
        }
    }

    #[test]
    fn gguf_source_single_file_accessor() {
        // Sanity-check the file()/files accessors on a synthetic source.
        let src = GgufSource {
            repo: "test/repo".to_owned(),
            files: vec!["only.gguf".to_owned()],
        };
        assert_eq!(src.file(), "only.gguf");
        assert!(!src.is_multi_shard());
    }

    #[test]
    fn gguf_source_multi_shard_accessor() {
        let src = GgufSource {
            repo: "test/repo".to_owned(),
            files: vec![
                "shard-00001-of-00002.gguf".to_owned(),
                "shard-00002-of-00002.gguf".to_owned(),
            ],
        };
        assert!(src.is_multi_shard());
        // file() returns the first/primary shard.
        assert_eq!(src.file(), "shard-00001-of-00002.gguf");
    }

    #[test]
    fn all_models_have_tokenizer() {
        let reg = load_registry();
        for entry in reg.list() {
            assert!(!entry.tokenizer.is_empty(), "model {} has no tokenizer", entry.key);
        }
    }
}
