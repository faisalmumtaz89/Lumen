//! `lumen-server qwen-image` and the image settings are refused where they do
//! not apply, before any model is opened or device touched. Host-only. The
//! binary needs the `bin` feature; with `image` the endpoint's own refusals
//! are checked, without it the refusal to serve images at all.

use std::process::{Command, Output};

/// `lumen-server` with `args` and `env`, and `cache` as its cache: the exit
/// code and what it printed to standard error.
#[allow(clippy::assertions_on_constants)]
fn server_in(
    cache: &std::path::Path,
    args: &[&str],
    env: &[(&str, &str)],
) -> (Option<i32>, String) {
    assert!(
        cfg!(feature = "bin"),
        "this test spawns the server binary: run with --features lumen-server/bin"
    );
    let out: Output = Command::new(env!("CARGO_BIN_EXE_lumen-server"))
        .args(args)
        .env("LUMEN_CACHE_DIR", cache)
        .envs(env.iter().copied())
        .output()
        .expect("run lumen-server");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// [`server_in`] with an empty cache of its own.
fn server(args: &[&str], env: &[(&str, &str)]) -> (Option<i32>, String) {
    let cache = std::env::temp_dir().join(format!(
        "lumen-server-image-refusals-{}-{}",
        std::process::id(),
        args.join("-").replace(['/', ':'], "_")
    ));
    let result = server_in(&cache, args, env);
    let _ = std::fs::remove_dir_all(&cache);
    result
}

#[cfg(feature = "image")]
#[test]
fn the_image_model_by_name_is_refused_when_misused_or_not_downloaded() {
    for (args, env, expected) in [
        (
            &["qwen-image:q8_0"][..],
            &[][..],
            "comes in one form and takes no quantization",
        ),
        (
            &["--model", "qwen-image:q8_0"][..],
            &[][..],
            "comes in one form and takes no quantization",
        ),
        (
            &["qwen-image"][..],
            &[("LUMEN_IMAGE_LBI", "/x"), ("LUMEN_IMAGE_CKPT", "/y")][..],
            "pass one or the other",
        ),
        (
            &["qwen-image"][..],
            &[][..],
            "Run `lumen pull qwen-image` first.",
        ),
        (
            &["qwen-image"][..],
            &[("LUMEN_IMAGE_MODEL_ID", "x")][..],
            "Run `lumen pull qwen-image` first.",
        ),
        (
            &["--model", "qwen-image:"][..],
            &[][..],
            "Run `lumen pull qwen-image` first.",
        ),
        (
            &["qwen3.5-9b"][..],
            &[("LUMEN_IMAGE_MODEL_ID", "x")][..],
            "nor the image model's name (lumen-server qwen-image) is",
        ),
    ] {
        let (code, stderr) = server(args, env);
        assert_ne!(code, Some(0), "{args:?} {env:?}: {stderr}");
        assert!(stderr.contains(expected), "{args:?} {env:?}: {stderr}");
    }
}

#[cfg(feature = "image")]
#[test]
fn the_image_settings_apply_to_a_checkpoint_converted_by_hand() {
    let (code, stderr) = server(
        &[],
        &[
            ("LUMEN_IMAGE_LBI", "/nonexistent/lbi"),
            ("LUMEN_IMAGE_CKPT", "/nonexistent/ckpt"),
            ("LUMEN_IMAGE_DEVICE", "cpu"),
        ],
    );
    assert_ne!(code, Some(0), "{stderr}");
    assert!(!stderr.contains("is set, but neither"), "{stderr}");
    assert!(
        stderr.contains("/nonexistent/lbi must hold transformer.lbi"),
        "the start-up check is reached: {stderr}"
    );
}

#[cfg(feature = "image")]
#[test]
fn the_image_model_by_name_is_read_from_the_directory_lumen_pull_fills() {
    let cache =
        std::env::temp_dir().join(format!("lumen-server-image-cache-{}", std::process::id()));
    let lbi = cache.join("qwen-image-2-1/lbi");
    std::fs::create_dir_all(&lbi).unwrap();
    let (code, stderr) = server_in(&cache, &["qwen-image"], &[("LUMEN_IMAGE_DEVICE", "cpu")]);
    let _ = std::fs::remove_dir_all(&cache);
    assert_ne!(code, Some(0), "{stderr}");
    assert!(
        stderr.contains(&format!("{}:", lbi.join("text_encoder.lbi").display())),
        "the converted files are checked first: {stderr}"
    );
    assert!(
        stderr.contains(&format!(
            "{} must hold transformer.lbi, vae.lbi and text_encoder.lbi",
            lbi.display()
        )),
        "{stderr}"
    );
}

#[cfg(not(feature = "image"))]
#[test]
fn a_server_without_the_image_endpoint_refuses_the_image_model() {
    let (code, stderr) = server(&["qwen-image"], &[]);
    assert_ne!(code, Some(0), "{stderr}");
    assert!(
        stderr.contains("makes images, and this lumen-server was built without the image endpoint"),
        "{stderr}"
    );
}
