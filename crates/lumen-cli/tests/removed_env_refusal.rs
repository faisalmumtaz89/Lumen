//! The binary refuses to start when an environment name an earlier release
//! read, and this one does not, is set: exit code 2, the name and the remedy on
//! stderr, before any model is opened. Host-only; no model, no GPU.

use std::process::Command;

fn lumen() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lumen"))
}

#[test]
fn a_removed_env_name_refuses_startup_with_its_remedy() {
    let out = lumen()
        .arg("--version")
        .env("LUMEN_CUDA_ATTN_SPLITK", "1")
        .output()
        .expect("run lumen");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr: {stderr}");
    assert!(
        stderr.contains("LUMEN_CUDA_ATTN_SPLITK is set but this release does not read it"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("there is no other route to switch on or off. Unset it."),
        "the remedy is missing: {stderr}"
    );
    assert!(stderr.contains("refusing to start"), "stderr: {stderr}");
}

#[test]
fn a_removed_env_name_set_to_zero_refuses_too() {
    let out = lumen()
        .arg("--version")
        .env("LUMEN_CUDA_DECODE_TILED_THRESHOLD", "0")
        .output()
        .expect("run lumen");
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn the_live_knobs_start_the_binary_without_a_warning() {
    let out = lumen()
        .arg("--version")
        .env("LUMEN_CUDA_ATTN_ONE_TILE", "176")
        .env("LUMEN_CUDA_ATTN_TARGET", "128")
        .env("LUMEN_CUDA_ATTN_CODEGEN", "default")
        // cargo's own test environment carries names the validator flags
        // (LUMEN_BUILD_VERSION, OUT_DIR); they are not what this test is about.
        .env_remove("LUMEN_BUILD_VERSION")
        .env_remove("OUT_DIR")
        .output()
        .expect("run lumen");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    for knob in [
        "LUMEN_CUDA_ATTN_ONE_TILE",
        "LUMEN_CUDA_ATTN_TARGET",
        "LUMEN_CUDA_ATTN_CODEGEN",
    ] {
        assert!(!stderr.contains(knob), "{knob} was warned about: {stderr}");
    }
    assert!(!stderr.contains("does not read it"), "stderr: {stderr}");
}
