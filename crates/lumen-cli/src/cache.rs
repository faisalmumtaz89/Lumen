//! Model cache directory management.
//!
//! Manages a local cache at `~/.cache/lumen/` (or `$LUMEN_CACHE_DIR`) where
//! downloaded GGUF files and converted LBC files are stored.

use std::path::PathBuf;

/// Get the cache directory path.
///
/// Priority:
/// 1. `$LUMEN_CACHE_DIR` environment variable (if set and non-empty)
/// 2. `$XDG_CACHE_HOME/lumen/` (via `dirs::cache_dir()` if `dirs` feature is available)
/// 3. `~/.cache/lumen/` fallback
pub fn cache_dir() -> PathBuf {
    if let Ok(val) = std::env::var("LUMEN_CACHE_DIR") {
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }

    #[cfg(feature = "download")]
    {
        if let Some(base) = dirs::cache_dir() {
            return base.join("lumen");
        }
    }

    // Fallback: ~/.cache/lumen
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("lumen");
    }

    // Last resort
    PathBuf::from(".cache").join("lumen")
}

/// Check if a cached LBC file exists for a given model key and quant.
///
/// The cached path is `<cache_dir>/<key>-<quant>.lbc` (e.g. `qwen2-5-3b-Q8_0.lbc`).
/// Returns `Some(path)` if the file exists and is non-empty.
pub fn cached_lbc(key: &str, quant: &str) -> Option<PathBuf> {
    let path = lbc_path(key, quant);
    if path.is_file() {
        // Check non-empty to avoid stale zero-length files.
        if let Ok(meta) = std::fs::metadata(&path) {
            if meta.len() > 0 {
                return Some(path);
            }
        }
    }
    None
}

/// Check if a cached GGUF file exists.
///
/// Returns `Some(path)` if the file exists and is non-empty.
pub fn cached_gguf(filename: &str) -> Option<PathBuf> {
    let path = gguf_path(filename);
    if path.is_file() {
        if let Ok(meta) = std::fs::metadata(&path) {
            if meta.len() > 0 {
                return Some(path);
            }
        }
    }
    None
}

/// Get the path where an LBC file would be cached.
pub fn lbc_path(key: &str, quant: &str) -> PathBuf {
    cache_dir().join(format!("{key}-{quant}.lbc"))
}

/// Get the path where a GGUF file would be cached.
pub fn gguf_path(filename: &str) -> PathBuf {
    cache_dir().join(filename)
}

/// Ensure the cache directory exists.
///
/// Creates all parent directories as needed. Returns the cache directory path.
pub fn ensure_cache_dir() -> Result<PathBuf, String> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create cache directory {}: {e}", dir.display()))?;
    Ok(dir)
}

/// List all cached `.lbc` files with their display names and sizes.
///
/// Returns `(stem, path, size_bytes)` tuples sorted by stem.
pub fn list_cached() -> Vec<(String, PathBuf, u64)> {
    let dir = cache_dir();
    let mut entries = Vec::new();

    let read_dir = match std::fs::read_dir(&dir) {
        Ok(rd) => rd,
        Err(_) => return entries,
    };

    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("lbc") {
            let stem = path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_owned();
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            if size > 0 {
                entries.push((stem, path, size));
            }
        }
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

/// Format a byte count as a human-readable string (e.g. "4.2 GB").
pub fn format_size(bytes: u64) -> String {
    const GB: f64 = 1_073_741_824.0;
    const MB: f64 = 1_048_576.0;
    const KB: f64 = 1024.0;

    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.1} MB", b / MB)
    } else if b >= KB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_returns_valid_path() {
        // Note: other tests may set LUMEN_CACHE_DIR, so we just verify
        // the returned path is non-empty and absolute or contains "lumen".
        let dir = cache_dir();
        let dir_str = dir.to_string_lossy();
        assert!(!dir_str.is_empty(), "cache_dir must return non-empty path");
        assert!(
            dir_str.contains("lumen"),
            "cache_dir should contain 'lumen': {dir_str}"
        );
    }

    #[test]
    fn lumen_cache_dir_env_override() {
        // This test uses a unique env var value to avoid collision with other tests.
        let original = std::env::var("LUMEN_CACHE_DIR").ok();
        std::env::set_var("LUMEN_CACHE_DIR", "/tmp/lumen-test-cache-env-override");
        let dir = cache_dir();
        assert_eq!(dir, PathBuf::from("/tmp/lumen-test-cache-env-override"));
        // Restore.
        match original {
            Some(val) => std::env::set_var("LUMEN_CACHE_DIR", val),
            None => std::env::remove_var("LUMEN_CACHE_DIR"),
        }
    }

    #[test]
    fn cached_lbc_returns_none_for_nonexistent() {
        let original = std::env::var("LUMEN_CACHE_DIR").ok();
        std::env::set_var("LUMEN_CACHE_DIR", "/tmp/lumen-nonexistent-dir-for-test");
        assert!(cached_lbc("nonexistent-model", "Q8_0").is_none());
        match original {
            Some(val) => std::env::set_var("LUMEN_CACHE_DIR", val),
            None => std::env::remove_var("LUMEN_CACHE_DIR"),
        }
    }

    #[test]
    fn lbc_path_format() {
        let original = std::env::var("LUMEN_CACHE_DIR").ok();
        std::env::set_var("LUMEN_CACHE_DIR", "/tmp/lumen-test-cache");
        let path = lbc_path("qwen2-5-3b", "Q8_0");
        assert_eq!(path, PathBuf::from("/tmp/lumen-test-cache/qwen2-5-3b-Q8_0.lbc"));
        match original {
            Some(val) => std::env::set_var("LUMEN_CACHE_DIR", val),
            None => std::env::remove_var("LUMEN_CACHE_DIR"),
        }
    }

    #[test]
    fn gguf_path_format() {
        let original = std::env::var("LUMEN_CACHE_DIR").ok();
        std::env::set_var("LUMEN_CACHE_DIR", "/tmp/lumen-test-cache");
        let path = gguf_path("model.Q8_0.gguf");
        assert_eq!(path, PathBuf::from("/tmp/lumen-test-cache/model.Q8_0.gguf"));
        match original {
            Some(val) => std::env::set_var("LUMEN_CACHE_DIR", val),
            None => std::env::remove_var("LUMEN_CACHE_DIR"),
        }
    }

    #[test]
    fn format_size_gb() {
        assert_eq!(format_size(4_500_000_000), "4.2 GB");
    }

    #[test]
    fn format_size_mb() {
        assert_eq!(format_size(52_428_800), "50.0 MB");
    }

    #[test]
    fn format_size_kb() {
        assert_eq!(format_size(2048), "2.0 KB");
    }

    #[test]
    fn format_size_bytes() {
        assert_eq!(format_size(42), "42 B");
    }

    #[test]
    fn list_cached_empty_dir() {
        let original = std::env::var("LUMEN_CACHE_DIR").ok();
        std::env::set_var("LUMEN_CACHE_DIR", "/tmp/lumen-nonexistent-dir-for-test");
        let entries = list_cached();
        assert!(entries.is_empty());
        match original {
            Some(val) => std::env::set_var("LUMEN_CACHE_DIR", val),
            None => std::env::remove_var("LUMEN_CACHE_DIR"),
        }
    }
}
