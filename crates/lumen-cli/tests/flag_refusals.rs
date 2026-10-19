//! A run the arguments, the build or this machine rule out is refused before
//! the model is resolved, so nothing is downloaded: the cache stays empty and
//! no download starts (a proxy that refuses every connection would make one
//! fail with another message). Host-only; no network, no model, no GPU.

use std::path::Path;
use std::process::Command;

/// `lumen run` with `args` against an empty cache: the exit code, what it
/// printed, and the files the cache holds afterwards.
fn run(tag: &str, args: &[&str]) -> (Option<i32>, String, usize) {
    let cache =
        std::env::temp_dir().join(format!("lumen-flag-refusals-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    std::fs::create_dir_all(&cache).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .arg("run")
        .args(args)
        .env("LUMEN_CACHE_DIR", &cache)
        .env("https_proxy", "ftp://127.0.0.1:9")
        .env("HTTPS_PROXY", "ftp://127.0.0.1:9")
        .env_remove("LUMEN_KV_PRECISION")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run lumen");
    let files = count_files(&cache);
    let _ = std::fs::remove_dir_all(&cache);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr),
        files,
    )
}

fn count_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| {
                    if e.path().is_dir() {
                        count_files(&e.path())
                    } else {
                        1
                    }
                })
                .sum()
        })
        .unwrap_or(0)
}

/// `args` are refused with exit code `code` and a message containing
/// `message`, before anything is downloaded.
fn refused(tag: &str, args: &[&str], code: i32, message: &str) {
    let (got, output, files) = run(tag, args);
    assert_eq!(got, Some(code), "{args:?}: {output}");
    assert!(output.contains(message), "{args:?}: {output}");
    assert!(
        !output.contains("Download"),
        "{args:?} started a download: {output}"
    );
    assert_eq!(files, 0, "{args:?} left files in the cache: {output}");
}

#[test]
fn a_missing_or_empty_prompt_is_refused_before_the_download() {
    refused("none", &["qwen3.5-9b:q4_0"], 1, "No prompt provided.");
    refused(
        "empty",
        &["qwen3.5-9b:q4_0", "--prompt", ""],
        1,
        "--prompt must not be empty",
    );
    refused(
        "tokens",
        &["qwen3.5-9b:q4_0", "--tokens", "1 x 3"],
        1,
        "token IDs must be non-negative integers, got: x",
    );
    refused(
        "blank-tokens",
        &["qwen3.5-9b:q4_0", "--tokens", " "],
        1,
        "--tokens must contain at least one token ID",
    );
}

#[test]
fn a_kv_precision_the_backend_cannot_store_is_refused_before_the_download() {
    refused(
        "cpu-bf16",
        &["qwen3.5-9b:q4_0", "--simd", "--kv-precision", "bf16", "hi"],
        1,
        "the CPU backends store the KV cache as F32 or F16 (requested Bf16)",
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_on_a_build_without_it_is_refused_before_the_download() {
    refused(
        "cuda",
        &["qwen3.5-9b:q4_0", "--cuda", "hi"],
        1,
        "--cuda requires building with --features cuda",
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_and_accelerate_off_macos_are_refused_before_the_download() {
    refused(
        "metal",
        &["qwen3.5-9b:q4_0", "--metal", "hi"],
        1,
        "--metal is only supported on macOS",
    );
    refused(
        "accelerate",
        &["qwen3.5-9b:q4_0", "--accelerate", "hi"],
        1,
        "--accelerate is only supported on macOS",
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_runs_it_cannot_serve_are_refused_before_the_download() {
    refused(
        "metal-f32",
        &["qwen3.5-9b:q4_0", "--metal", "--kv-precision", "f32", "hi"],
        1,
        "Metal backend KV cache is F16-only",
    );
    for mode in ["--async", "--no-gpu-resident", "--streaming"] {
        refused(
            mode.trim_start_matches('-'),
            &["qwen3.5-9b:q4_0", "--metal", mode, "hi"],
            1,
            "linear-attention (GDN) layers run only from GPU-resident weights",
        );
    }
    refused(
        "accelerate-session",
        &[
            "qwen3.5-9b:q4_0",
            "--accelerate",
            "--session-save",
            "/tmp/lumen-flag-refusals-s.bin",
            "hi",
        ],
        2,
        "is not supported with --accelerate",
    );
}
