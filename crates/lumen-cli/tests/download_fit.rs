//! A CUDA build refuses a download whose weights the GPU it would run on
//! cannot hold; where it cannot tell (no CUDA device, or a size it cannot
//! learn, as offline) it goes on to the download, which fails or succeeds as
//! it always has.

#![cfg(all(feature = "download", feature = "cuda"))]

use std::process::Command;

#[test]
fn a_pull_whose_size_cannot_be_told_goes_on_to_the_download() {
    let cache = std::env::temp_dir().join(format!("lumen-download-fit-{}", std::process::id()));
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args(["pull", "qwen3.8-27b:bf16", "--yes"])
        .env("LUMEN_CACHE_DIR", &cache)
        .env("https_proxy", "ftp://127.0.0.1:9")
        .env("HTTPS_PROXY", "ftp://127.0.0.1:9")
        .env_remove("no_proxy")
        .env_remove("NO_PROXY")
        .output()
        .expect("run lumen");
    let _ = std::fs::remove_dir_all(&cache);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("Download failed"), "{stderr}");
    assert!(!stderr.contains("cannot run there"), "{stderr}");
}
