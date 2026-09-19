//! A conversion that does not run to completion must leave nothing at the output path:
//! the CLI converts straight into the model cache, and the cache serves any non-empty
//! file at that path as a finished artifact, so a half-written `.lbc` left behind is
//! served for every later run.
mod common;

use lumen_convert::convert::convert_gguf_bytes_to_lbc;
use std::path::{Path, PathBuf};

/// The sibling the conversion writes into before it renames onto `lbc_path`.
fn tmp_path(lbc_path: &Path) -> PathBuf {
    let name = lbc_path.file_name().unwrap().to_string_lossy().into_owned();
    lbc_path.with_file_name(format!("{name}.tmp.{}", std::process::id()))
}

/// Every entry beside `lbc_path` whose name extends it: whatever a conversion writes
/// under an interim name lands here, so this does not depend on that name.
fn siblings(lbc_path: &Path) -> Vec<String> {
    let stem = lbc_path.file_name().unwrap().to_string_lossy().into_owned();
    std::fs::read_dir(lbc_path.parent().unwrap())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(&stem) && *n != stem)
        .collect()
}

fn case(tag: &str) -> (Vec<u8>, PathBuf) {
    let out = std::env::temp_dir().join(format!("interrupted_{tag}_{}.lbc", std::process::id()));
    std::fs::remove_file(&out).ok();
    std::fs::remove_dir_all(&out).ok();
    for name in siblings(&out) {
        std::fs::remove_file(out.with_file_name(name)).ok();
    }
    (common::build(common::GATE, common::GATE), out)
}

/// A source truncated inside its tensor data plans and opens fine — the layer blobs are
/// read after the output file is open — so the failure lands mid-write.
#[test]
fn a_conversion_failing_mid_write_leaves_no_artifact() {
    let (gguf, out) = case("midwrite");

    let err = convert_gguf_bytes_to_lbc(&gguf[..gguf.len() / 2], &out, &common::generic())
        .expect_err("a truncated source cannot convert");
    assert!(err.to_string().contains("fewer are available"), "{err}");
    assert!(
        !out.exists(),
        "a failed conversion left a file at the output path",
    );
    assert_eq!(
        siblings(&out),
        Vec::<String>::new(),
        "a failed conversion left an interim file",
    );
}

/// The artifact is complete and flushed before the rename that publishes it, so a
/// failure at that last step must still leave neither an artifact nor a temp file. A
/// directory at the output path makes the rename fail with everything else done.
#[test]
fn a_failure_after_the_artifact_is_written_leaves_no_temp_file() {
    let (gguf, out) = case("publish");
    std::fs::create_dir_all(out.join("occupied")).unwrap();

    let err = convert_gguf_bytes_to_lbc(&gguf, &out, &common::generic())
        .expect_err("the artifact cannot be renamed onto a directory");
    assert_eq!(
        siblings(&out),
        Vec::<String>::new(),
        "a failure after the write left an interim file: {err}",
    );
    assert!(out.is_dir(), "the output path was replaced");
    std::fs::remove_dir_all(&out).ok();
}

/// The temp file is opened with `create_new`, so an existing path there — a symlink
/// planted by another process, or a concurrent conversion's own temp file — refuses the
/// conversion instead of writing through it.
#[test]
fn a_pre_existing_path_at_the_temp_name_is_refused() {
    let (gguf, out) = case("planted");
    let tmp = tmp_path(&out);
    let target = out.with_extension("planted-target");
    std::fs::remove_file(&target).ok();
    std::os::unix::fs::symlink(&target, &tmp).unwrap();

    let err = convert_gguf_bytes_to_lbc(&gguf, &out, &common::generic())
        .expect_err("a temp path that already exists must refuse the conversion");
    assert!(err.to_string().contains("exists"), "{err}");
    assert!(!target.exists(), "the conversion wrote through the symlink");
    assert!(!out.exists(), "the conversion published an artifact");
    assert!(
        std::fs::symlink_metadata(&tmp).is_ok(),
        "the conversion removed a path it did not create",
    );
    std::fs::remove_file(&tmp).ok();
}

/// Two outputs whose names differ only after a dot — the registry keys carry them, as
/// `qwen3.5-9b-q8_0` and `qwen3.8-27b-q4_0` do — must not share a temp path, or the
/// second conversion in a process refuses on the first one's file.
#[test]
fn outputs_differing_after_a_dot_do_not_share_a_temp_path() {
    let gguf = common::build(common::GATE, common::GATE);
    let dir = std::env::temp_dir().join(format!("interrupted_dotted_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let first = dir.join("qwen3.5-9b-q8_0");
    let second = dir.join("qwen3.8-27b-q4_0");
    // The two temp paths differ only because the suffix extends the whole file name:
    // a scheme that replaced the extension would give both the same one.
    assert_ne!(tmp_path(&first), tmp_path(&second));
    // And the conversion really writes at that name — planting a file there stops it.
    std::fs::write(tmp_path(&first), b"planted").unwrap();
    convert_gguf_bytes_to_lbc(&gguf, &first, &common::generic())
        .expect_err("the conversion did not write at the temp path this name gives");
    std::fs::remove_file(tmp_path(&first)).unwrap();

    for out in [&first, &second] {
        convert_gguf_bytes_to_lbc(&gguf, out, &common::generic())
            .unwrap_or_else(|e| panic!("{}: {e}", out.display()));
        assert_eq!(
            siblings(out),
            Vec::<String>::new(),
            "an interim file outlived the conversion",
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// And a conversion that succeeds leaves the artifact at the output path with no temp
/// file beside it.
#[test]
fn a_successful_conversion_leaves_only_the_artifact() {
    let (gguf, out) = case("completed");

    convert_gguf_bytes_to_lbc(&gguf, &out, &common::generic()).unwrap();
    assert!(out.metadata().unwrap().len() > 0, "artifact is empty");
    assert_eq!(
        siblings(&out),
        Vec::<String>::new(),
        "an interim file outlived the conversion",
    );
    std::fs::remove_file(&out).ok();
}
