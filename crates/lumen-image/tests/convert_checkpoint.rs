//! `convert_checkpoint`, and the `lbi-convert` command built on it, on a small
//! checkpoint: every component converts, a component in shards from its
//! index, staging files left by a dead conversion are removed first, and a
//! conversion that fails says which component and leaves no staging files.

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
    write_shard(&dir.join("model.safetensors"), "w", dtype);
}

/// A safetensors at `path` holding one tensor `name` of `dtype` with four
/// data bytes.
fn write_shard(path: &Path, name: &str, dtype: &str) {
    let header = format!(r#"{{"{name}":{{"dtype":"{dtype}","shape":[1],"data_offsets":[0,4]}}}}"#);
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[1, 2, 3, 4]);
    std::fs::write(path, bytes).unwrap();
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
fn stale_staging_files_are_removed_before_anything_is_converted() {
    let root = scratch("stale");
    let ckpt = root.join("ckpt");
    // The first component fails, so the later ones are never written: only
    // the sweep at the start can remove what a dead conversion left for them.
    write_component(&ckpt, "transformer", "I8");
    write_component(&ckpt, "vae", "F32");
    write_component(&ckpt, "text_encoder", "F32");
    let out = root.join("lbi");
    std::fs::create_dir_all(&out).unwrap();
    for component in ["vae", "text_encoder"] {
        let (part, tmp) = staging_paths(&out.join(format!("{component}.lbi")));
        std::fs::write(&part, b"stale").unwrap();
        std::fs::write(&tmp, b"stale").unwrap();
    }

    convert_checkpoint(&ckpt, &out, |_| {}).unwrap_err();

    assert!(names(&out).is_empty(), "left behind: {:?}", names(&out));
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn lbi_convert_reports_each_component_and_writes_the_three_containers() {
    let root = scratch("command-ok");
    let ckpt = root.join("ckpt");
    for component in COMPONENTS {
        write_component(&ckpt, component, "F32");
    }
    let out = root.join("lbi");

    let run = std::process::Command::new(env!("CARGO_BIN_EXE_lbi-convert"))
        .arg(&ckpt)
        .arg(&out)
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "{stdout}");
    for (line, component) in lines.iter().zip(COMPONENTS) {
        assert!(
            line.starts_with(component) && line.contains("1 tensors"),
            "{line}"
        );
    }
    assert_eq!(
        names(&out),
        ["text_encoder.lbi", "transformer.lbi", "vae.lbi"]
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn lbi_convert_removes_a_dead_conversions_staging_files_and_names_what_failed() {
    let root = scratch("command");
    let ckpt = root.join("ckpt");
    write_component(&ckpt, "transformer", "I8");
    write_component(&ckpt, "vae", "F32");
    write_component(&ckpt, "text_encoder", "F32");
    let out = root.join("lbi");
    std::fs::create_dir_all(&out).unwrap();
    let (part, tmp) = staging_paths(&out.join("vae.lbi"));
    std::fs::write(&part, b"stale").unwrap();
    std::fs::write(&tmp, b"stale").unwrap();

    let run = std::process::Command::new(env!("CARGO_BIN_EXE_lbi-convert"))
        .arg(&ckpt)
        .arg(&out)
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success(), "{stderr}");
    assert!(stderr.contains("convert transformer"), "{stderr}");
    assert!(names(&out).is_empty(), "left behind: {:?}", names(&out));
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn every_component_converts_and_only_the_containers_remain() {
    let root = scratch("ok");
    let ckpt = root.join("ckpt");
    for component in COMPONENTS {
        write_component(&ckpt, component, "F32");
    }
    let out = root.join("lbi");

    let mut reports = Vec::new();
    convert_checkpoint(&ckpt, &out, |r| reports.push(r.clone())).unwrap();

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
fn a_component_in_shards_converts_from_its_index() {
    let root = scratch("sharded");
    let ckpt = root.join("ckpt");
    let dir = ckpt.join("transformer");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), br#"{"probe":true}"#).unwrap();
    write_shard(&dir.join("model-00001-of-00002.safetensors"), "a", "F32");
    write_shard(&dir.join("model-00002-of-00002.safetensors"), "b", "F32");
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
    )
    .unwrap();
    write_component(&ckpt, "vae", "F32");
    write_component(&ckpt, "text_encoder", "F32");
    let out = root.join("lbi");

    let mut reports = Vec::new();
    convert_checkpoint(&ckpt, &out, |r| reports.push(r.clone())).unwrap();

    assert_eq!(reports[0].component, "transformer");
    assert_eq!((reports[0].tensor_count, reports[0].total_bytes), (2, 8));
    let file = lumen_image::lbi::LbiFile::open(&out.join("transformer.lbi")).unwrap();
    assert_eq!(file.len(), 2, "both shards' tensors");
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

    let mut done = Vec::new();
    let err = convert_checkpoint(&ckpt, &out, |r| done.push(r.component.clone())).unwrap_err();
    assert_eq!(
        done,
        ["transformer", "vae"],
        "each finished component is reported as it finishes"
    );

    assert!(
        matches!(
            &err,
            ConvertError::Component { component, error }
                if component == "text_encoder"
                    && matches!(**error, ConvertError::UnsupportedDtype { .. })
        ),
        "got {err}"
    );
    assert!(
        err.to_string().starts_with("convert text_encoder: "),
        "{err}"
    );
    assert_eq!(
        names(&out),
        ["transformer.lbi", "vae.lbi"],
        "the components converted before the failure stay, nothing else"
    );
    std::fs::remove_dir_all(&root).ok();
}
