//! `convert_checkpoint` on a small checkpoint: every component converts,
//! staging files left by a dead conversion are removed first, and a
//! conversion that fails leaves no staging files behind.

use std::path::{Path, PathBuf};

use lumen_image::convert::{convert_checkpoint, ConvertError, COMPONENTS};
use lumen_image::lbi::staging_paths;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lumen-convert-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One single-file component: `config.json` and a safetensors holding one
/// tensor of `dtype` with four data bytes.
fn write_component(ckpt: &Path, component: &str, dtype: &str) {
    let dir = ckpt.join(component);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), br#"{"probe":true}"#).unwrap();
    let header = format!(r#"{{"w":{{"dtype":"{dtype}","shape":[1],"data_offsets":[0,4]}}}}"#);
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[1, 2, 3, 4]);
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn every_component_converts_and_stale_staging_files_are_removed_first() {
    let root = scratch("ok");
    let ckpt = root.join("ckpt");
    for component in COMPONENTS {
        write_component(&ckpt, component, "F32");
    }
    let out = root.join("lbi");
    std::fs::create_dir_all(&out).unwrap();
    // What a conversion that died mid-way leaves behind.
    let (part, tmp) = staging_paths(&out.join("text_encoder.lbi"));
    std::fs::write(&part, b"stale").unwrap();
    std::fs::write(&tmp, b"stale").unwrap();

    let reports = convert_checkpoint(&ckpt, &out).unwrap();

    assert_eq!(
        reports
            .iter()
            .map(|r| r.component.as_str())
            .collect::<Vec<_>>(),
        COMPONENTS
    );
    assert!(reports
        .iter()
        .all(|r| r.tensor_count == 1 && r.total_bytes == 4));
    assert_eq!(
        names(&out),
        ["text_encoder.lbi", "transformer.lbi", "vae.lbi"],
        "only the three containers remain"
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn a_failed_conversion_keeps_earlier_containers_and_no_staging_files() {
    let root = scratch("fail");
    let ckpt = root.join("ckpt");
    write_component(&ckpt, "transformer", "F32");
    write_component(&ckpt, "vae", "F32");
    write_component(&ckpt, "text_encoder", "I8");
    let out = root.join("lbi");

    let err = convert_checkpoint(&ckpt, &out).unwrap_err();

    assert!(
        matches!(err, ConvertError::UnsupportedDtype { .. }),
        "got {err}"
    );
    assert_eq!(
        names(&out),
        ["transformer.lbi", "vae.lbi"],
        "the components converted before the failure stay, nothing else"
    );
    std::fs::remove_dir_all(&root).ok();
}
