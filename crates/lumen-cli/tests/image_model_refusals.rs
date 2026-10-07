//! The image model is listed as the one form it comes in and refused where
//! it does not belong, before anything is downloaded or loaded. Host-only; no
//! network, no model, no GPU.

use std::path::Path;
use std::process::Command;

/// `lumen` with `args` and `cache` as its cache: the exit code and what it
/// printed, standard output then standard error.
fn lumen_in(cache: &Path, args: &[&str]) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args(args)
        .env("LUMEN_CACHE_DIR", cache)
        .output()
        .expect("run lumen");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr),
    )
}

fn lumen(args: &[&str]) -> (Option<i32>, String) {
    let cache = std::env::temp_dir().join(format!("lumen-image-refusals-{}", std::process::id()));
    let result = lumen_in(&cache, args);
    let _ = std::fs::remove_dir_all(&cache);
    result
}

#[test]
fn lumen_run_points_the_image_model_to_the_server() {
    let (code, output) = lumen(&["run", "qwen-image", "hello"]);
    assert_eq!(code, Some(1), "{output}");
    assert!(
        output.contains("makes images, not text. Serve it with: lumen-server qwen-image"),
        "{output}"
    );
}

#[test]
fn the_image_model_takes_no_quantization() {
    for args in [
        &["pull", "qwen-image:q8_0"][..],
        &["pull", "qwen-image", "--quant", "Q4_0"][..],
    ] {
        let (code, output) = lumen(args);
        assert_eq!(code, Some(1), "{args:?}: {output}");
        assert!(
            output.contains("comes in one form and takes no quantization"),
            "{args:?}: {output}"
        );
    }
}

#[test]
fn lumen_run_suggests_only_text_models() {
    let (code, output) = lumen(&["run", "qwen-imag", "hello"]);
    assert_eq!(code, Some(1), "{output}");
    assert!(output.contains("unknown model 'qwen-imag'"), "{output}");
    assert!(!output.contains("Qwen-Image-2.1"), "{output}");
}

#[cfg(all(feature = "download", not(feature = "cuda")))]
#[test]
fn a_lumen_without_cuda_refuses_to_make_a_picture() {
    let (code, stderr) = lumen_offline(&["image", "A red apple"]);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(
        stderr.contains("makes images on NVIDIA CUDA, and this lumen was built without CUDA."),
        "{stderr}"
    );
}

#[cfg(all(feature = "download", not(feature = "cuda")))]
#[test]
fn a_lumen_without_cuda_refuses_to_pull_the_image_model() {
    // Without --yes and with standard input closed, a lumen that went on would
    // stop at the prompt rather than download.
    let (code, output) = lumen(&["pull", "qwen-image"]);
    assert_eq!(code, Some(1), "{output}");
    assert!(
        output.contains("this lumen was built without CUDA; nothing was downloaded"),
        "{output}"
    );
}

#[test]
fn the_image_model_is_listed_as_one_form_and_only_where_it_can_be_used() {
    let (_, output) = lumen(&["run"]);
    assert!(!output.contains("qwen-image"), "{output}");
    let (_, output) = lumen(&["pull"]);
    assert!(output.contains("\n  qwen-image-2-1\n"), "{output}");
    let (_, output) = lumen(&["models"]);
    assert!(
        output.contains("qwen-image-2-1       Qwen-Image-2.1 (text to image)"),
        "{output}"
    );

    // With a text model cached, the image model is offered for download
    // until it is cached itself, then listed as cached.
    let cache = std::env::temp_dir().join(format!("lumen-image-listed-{}", std::process::id()));
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::write(cache.join("qwen3-5-9b-Q8_0.lbc"), b"bytes").unwrap();
    let (_, output) = lumen_in(&cache, &["models"]);
    let (_, available) = output.split_once("Available to download").unwrap();
    assert!(
        available.contains("qwen-image-2-1       Qwen-Image-2.1 text to image"),
        "{output}"
    );
    for file in [
        "lbi/transformer.lbi",
        "lbi/vae.lbi",
        "lbi/text_encoder.lbi",
        "processor/vocab.json",
        "processor/merges.txt",
        "processor/added_tokens.json",
    ] {
        let path = cache.join("qwen-image-2-1").join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"bytes").unwrap();
    }
    let (_, output) = lumen_in(&cache, &["models"]);
    std::fs::remove_dir_all(&cache).ok();
    let (cached, available) = output.split_once("Available to download").unwrap();
    assert!(
        cached.contains(&format!("  {:<40} 15 B", "qwen-image-2-1")),
        "the three converted files' size: {output}"
    );
    assert!(!available.contains("qwen-image-2-1"), "{output}");
}

/// `lumen` with `args`, standard input closed and a proxy setting the
/// downloader refuses before any request, so a download, were one started,
/// fails at once without reaching the network.
#[cfg(feature = "download")]
fn lumen_offline(args: &[&str]) -> (Option<i32>, String) {
    let cache = std::env::temp_dir().join(format!(
        "lumen-image-offline-{}-{}",
        std::process::id(),
        args.len()
    ));
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args(args)
        .env("LUMEN_CACHE_DIR", &cache)
        .env("https_proxy", "ftp://127.0.0.1:9")
        .env("HTTPS_PROXY", "ftp://127.0.0.1:9")
        .env_remove("no_proxy")
        .env_remove("NO_PROXY")
        .output()
        .expect("run lumen");
    let _ = std::fs::remove_dir_all(&cache);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[cfg(all(feature = "download", feature = "cuda"))]
#[test]
fn a_lumen_with_cuda_asks_before_it_downloads_the_image_model() {
    let (code, stderr) = lumen_offline(&["pull", "qwen-image"]);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(
        stderr.contains("Download Qwen-Image-2.1 from Qwen/Qwen-Image-2.1 ("),
        "{stderr}"
    );
    assert!(
        stderr.contains("no answer: standard input is closed; pass --yes"),
        "{stderr}"
    );
}

#[cfg(all(feature = "download", feature = "cuda"))]
#[test]
fn a_lumen_with_cuda_downloads_without_asking_given_yes() {
    let (code, stderr) = lumen_offline(&["pull", "qwen-image", "--yes"]);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(!stderr.contains("[Y/n]"), "{stderr}");
    // It goes on to the download, or, on a disk without room for the model,
    // to the refusal that comes before it.
    assert!(
        stderr.contains("Downloading Qwen-Image-2.1 from Qwen/Qwen-Image-2.1 (")
            || stderr.contains("Qwen-Image-2.1 needs"),
        "{stderr}"
    );
}

#[cfg(all(feature = "download", feature = "cuda"))]
#[test]
fn a_first_picture_downloads_without_asking_only_where_the_device_can_run_it() {
    let device = lumen_image::pipeline::check_device();
    let dir = std::env::temp_dir().join(format!("lumen-image-first-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("apple.png");
    let (code, stderr) = lumen_offline(&["image", "A red apple", "-o", path.to_str().unwrap()]);
    let left: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(!stderr.contains("[Y/n]"), "{stderr}");
    match device {
        Err(e) => {
            assert!(stderr.contains(&e.to_string()), "{stderr}");
            assert!(!stderr.contains("Downloading"), "{stderr}");
        }
        // It goes on to the download, or, on a disk without room for the
        // model, to the refusal that comes before it.
        Ok(()) => assert!(
            stderr.contains("Downloading Qwen-Image-2.1 from Qwen/Qwen-Image-2.1 (")
                || stderr.contains("Qwen-Image-2.1 needs"),
            "{stderr}"
        ),
    }
    // A run that fails before it has a picture leaves nothing behind.
    assert!(left.is_empty(), "{left:?}");
}
