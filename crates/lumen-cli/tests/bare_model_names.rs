//! A bare model name means the model's default quant in `lumen pull`,
//! `lumen run` and `lumen models`. Host-only: a cache of placeholder files and
//! a proxy setting the downloader refuses before any request.

#![cfg(feature = "download")]

use std::path::{Path, PathBuf};
use std::process::Command;

/// A fresh cache holding a placeholder for each of `files`.
fn cache(tag: &str, files: &[&str]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lumen-bare-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for file in files {
        std::fs::write(dir.join(file), b"x").unwrap();
    }
    dir
}

/// `lumen` with `args` and `cache` as its cache, standard input closed and no
/// network: the exit code and what it printed, standard output then standard
/// error. The cache is removed afterwards.
fn lumen(cache: &Path, args: &[&str]) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args(args)
        .env("LUMEN_CACHE_DIR", cache)
        .env("https_proxy", "ftp://127.0.0.1:9")
        .env("HTTPS_PROXY", "ftp://127.0.0.1:9")
        .env_remove("no_proxy")
        .env_remove("NO_PROXY")
        .output()
        .expect("run lumen");
    let _ = std::fs::remove_dir_all(cache);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr),
    )
}

#[test]
fn a_bare_pull_takes_the_default_quant() {
    let dir = cache("pull", &["qwen3-8-27b-Q4_0.lbc"]);
    let (code, out) = lumen(&dir, &["pull", "qwen3.8-27b"]);
    assert_eq!(code, Some(0), "{out}");
    assert!(out.contains("Already cached:"), "{out}");
    assert!(out.contains("qwen3-8-27b-Q4_0.lbc"), "{out}");
}

#[test]
fn a_bare_run_names_a_downloaded_quant_before_downloading_the_default() {
    let dir = cache("run", &["qwen3-8-27b-Q8_0.lbc"]);
    let (code, out) = lumen(&dir, &["run", "qwen3.8-27b", "hi"]);
    assert_eq!(code, Some(1), "{out}");
    let notice = out
        .find("qwen3.8-27b:q8_0 is already downloaded (run it with: lumen run qwen3.8-27b:q8_0")
        .unwrap_or_else(|| panic!("no notice: {out}"));
    let failed = out
        .find("Download failed")
        .unwrap_or_else(|| panic!("no download: {out}"));
    assert!(notice < failed, "the notice comes first: {out}");
    assert!(
        out.contains("downloading the default, qwen3.8-27b:q4_0"),
        "{out}"
    );
}

#[test]
fn lumen_models_marks_each_default_with_models_cached() {
    let dir = cache("models", &["qwen3-8-27b-Q4_0.lbc", "qwen3-8-27b-Q8_0.lbc"]);
    let (code, out) = lumen(&dir, &["models"]);
    assert_eq!(code, Some(0), "{out}");
    let line = |needle: &str| {
        out.lines()
            .find(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("no {needle}: {out}"))
    };
    assert!(line("qwen3-8-27b-Q4_0").ends_with("(default)"), "{out}");
    assert!(!line("qwen3-8-27b-Q8_0").contains("(default)"), "{out}");
    assert!(line("Qwen3.5 9B Q8_0").ends_with("(default)"), "{out}");
    assert!(
        line("Qwen3.5 MoE 35B-A3B Q4_0").ends_with("(default)"),
        "{out}"
    );
    assert_eq!(out.matches("(default)").count(), 3, "{out}");
}

#[test]
fn a_pull_takes_the_cuda_device_its_download_is_checked_against() {
    let (code, out) = lumen(
        &cache("device", &[]),
        &["pull", "qwen3.5-9b", "--cuda-device", "1"],
    );
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("Download failed"), "the pull goes on: {out}");
    let (code, out) = lumen(
        &cache("device-bad", &[]),
        &["pull", "qwen3.5-9b", "--cuda-device", "x"],
    );
    assert_eq!(code, Some(1), "{out}");
    assert!(
        out.contains("--cuda-device must be a non-negative integer, got: x"),
        "{out}"
    );
}
