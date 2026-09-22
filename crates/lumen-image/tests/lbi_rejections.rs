//! Malformed-input rejections for the `.lbi` reader.
//!
//! Every case here builds a corrupt container from a valid one and requires the
//! reader to reject it. These are the reader-side counterpart to the writer
//! tests in `lbi_roundtrip.rs`; the earlier suite tested the writer and the
//! report wrongly described that as reader coverage.

use lumen_format::QuantScheme;
use lumen_image::lbi::{LbiError, LbiFile, LbiWriter};

const ENTRY_BYTES: usize = 60;

fn tmp(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("lbi-rej-{}-{name}", std::process::id()));
    p
}

/// A valid one-tensor file, returned as raw bytes with the layout offsets.
struct Built {
    path: std::path::PathBuf,
    bytes: Vec<u8>,
    names_off: usize,
    names_len: usize,
    index_off: usize,
    blob_off: usize,
}

fn built(name: &str) -> Built {
    let path = tmp(name);
    let _ = std::fs::remove_file(&path);
    let mut w = LbiWriter::create(&path, serde_json::json!({"a": 1})).unwrap();
    let data: Vec<u8> = (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    w.append("w", &[4, 4], QuantScheme::F32, &data).unwrap();
    w.finish().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let names_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let config_len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let names_off = 20;
    let index_off = names_off + names_len + 4 + config_len;
    let entries = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let blob_off = (index_off + entries * ENTRY_BYTES).div_ceil(64) * 64;
    assert!(bytes.len() >= blob_off + 64, "fixture blob region is short");
    Built {
        path,
        bytes,
        names_off,
        names_len,
        index_off,
        blob_off,
    }
}

fn write_and_open(name: &str, bytes: &[u8]) -> Result<usize, LbiError> {
    let p = tmp(name);
    std::fs::write(&p, bytes).unwrap();
    let r = LbiFile::open(&p).map(|f| f.len());
    let _ = std::fs::remove_file(&p);
    r
}

#[test]
fn bad_version_is_rejected() {
    let b = built("version");
    let mut v = b.bytes.clone();
    v[4..8].copy_from_slice(&99u32.to_le_bytes());
    assert!(matches!(
        write_and_open("version2", &v),
        Err(LbiError::UnsupportedVersion(99))
    ));
    let _ = std::fs::remove_file(b.path);
}

#[test]
fn tensor_count_above_the_limit_is_rejected() {
    let b = built("count");
    let mut v = b.bytes.clone();
    v[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        write_and_open("count2", &v),
        Err(LbiError::TooManyTensors(_))
    ));
    let _ = std::fs::remove_file(b.path);
}

#[test]
fn oversized_name_section_is_rejected() {
    let b = built("names");
    let mut v = b.bytes.clone();
    v[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        write_and_open("names2", &v),
        Err(LbiError::NameSectionTooLarge(_))
    ));
    let _ = std::fs::remove_file(b.path);
}

#[test]
fn name_offset_outside_the_name_section_is_rejected() {
    let b = built("nameoff");
    let mut v = b.bytes.clone();
    // Point the first entry's name past the end of the section.
    v[b.index_off..b.index_off + 4].copy_from_slice(&(b.names_len as u32).to_le_bytes());
    v[b.index_off + 4..b.index_off + 8].copy_from_slice(&8u32.to_le_bytes());
    assert!(matches!(
        write_and_open("nameoff2", &v),
        Err(LbiError::Truncated { .. })
    ));
    let _ = std::fs::remove_file(b.path);
    assert_eq!(b.names_off, 20);
}

#[test]
fn unknown_quantization_tag_is_rejected() {
    let b = built("tag");
    let mut v = b.bytes.clone();
    let quant_at = b.index_off + 12 + 24;
    v[quant_at] = 250;
    assert!(matches!(
        write_and_open("tag2", &v),
        Err(LbiError::UnknownQuantTag(250))
    ));
    let _ = std::fs::remove_file(b.path);
}

#[test]
fn rank_above_the_index_budget_is_rejected() {
    let b = built("rank");
    let mut v = b.bytes.clone();
    v[b.index_off + 8] = 200;
    assert!(matches!(
        write_and_open("rank2", &v),
        Err(LbiError::RankTooLarge { .. })
    ));
    let _ = std::fs::remove_file(b.path);
}

#[test]
fn config_length_prefix_mismatch_is_rejected() {
    let b = built("cfgmismatch");
    let mut v = b.bytes.clone();
    let cfg_prefix = b.names_off + b.names_len;
    v[cfg_prefix..cfg_prefix + 4].copy_from_slice(&7u32.to_le_bytes());
    // The prefix must be changed to something the header does NOT already say,
    // or the mutation is a no-op.
    let hdr_cfg_len = u32::from_le_bytes(v[16..20].try_into().unwrap());
    let bogus = hdr_cfg_len ^ 0x1;
    v[cfg_prefix..cfg_prefix + 4].copy_from_slice(&bogus.to_le_bytes());
    assert!(matches!(
        write_and_open("cfgmismatch2", &v),
        Err(LbiError::Truncated { .. })
    ));
    let _ = std::fs::remove_file(b.path);
}

#[test]
fn misaligned_tensor_offset_is_rejected() {
    let b = built("misalign");
    let mut v = b.bytes.clone();
    let off_at = b.index_off + 12 + 24 + 8;
    v[off_at..off_at + 8].copy_from_slice(&1u64.to_le_bytes());
    assert!(matches!(
        write_and_open("misalign2", &v),
        Err(LbiError::MisalignedOffset { .. })
    ));
    let _ = std::fs::remove_file(b.path);
}

/// An offset near the top of the range must not wrap into a valid-looking
/// window when the absolute range is computed.
#[test]
fn wrapping_tensor_offset_is_rejected() {
    let b = built("wrap");
    let mut v = b.bytes.clone();
    let off_at = b.index_off + 12 + 24 + 8;
    v[off_at..off_at + 8].copy_from_slice(&(u64::MAX - 63).to_le_bytes());
    v[off_at + 8..off_at + 16].copy_from_slice(&64u64.to_le_bytes());
    assert!(
        matches!(write_and_open("wrap2", &v), Err(LbiError::Truncated { .. })),
        "a wrapping offset must be rejected, not resolved to an in-file range"
    );
    let _ = std::fs::remove_file(b.path);
    assert!(b.blob_off > 0);
}

#[test]
fn blob_shorter_than_the_index_claims_is_rejected() {
    let b = built("short");
    // Cut the blob region off entirely.
    let v = b.bytes[..b.blob_off].to_vec();
    assert!(matches!(
        write_and_open("short2", &v),
        Err(LbiError::Truncated { .. })
    ));
    let _ = std::fs::remove_file(b.path);
}

#[test]
fn header_shorter_than_the_fixed_prefix_is_rejected() {
    for cut in [0usize, 4, 12, 19] {
        assert!(
            matches!(
                write_and_open(&format!("hdr{cut}"), &vec![0u8; cut]),
                Err(LbiError::Truncated { .. })
            ),
            "a {cut}-byte file cannot hold the header"
        );
    }
}

#[test]
fn a_shape_above_the_wire_limit_is_rejected_by_the_writer() {
    let path = tmp("bigdim");
    let _ = std::fs::remove_file(&path);
    let mut w = LbiWriter::create(&path, serde_json::json!({})).unwrap();
    // A dimension that does not fit u32 must be refused, not truncated.
    let err = w
        .append("w", &[0, 4_294_967_296], QuantScheme::F32, &[])
        .unwrap_err();
    assert!(matches!(err, LbiError::ShapeTooLarge { .. }), "got {err:?}");
}

#[test]
fn a_byte_count_that_overflows_is_rejected_by_the_writer() {
    // (u32::MAX)^2 elements fits u64, but at 4 bytes each it does not.
    let path = tmp("overflow");
    let _ = std::fs::remove_file(&path);
    let mut w = LbiWriter::create(&path, serde_json::json!({})).unwrap();
    let err = w
        .append("w", &[4_294_967_295, 4_294_967_295], QuantScheme::F32, &[])
        .unwrap_err();
    assert!(matches!(err, LbiError::ShapeTooLarge { .. }), "got {err:?}");
}

#[test]
fn an_element_count_that_overflows_is_rejected_by_the_writer() {
    // Every dimension is wire-valid (below u32::MAX), so the dimension guard
    // passes and the failure must come from the element-count multiplication:
    // (1 << 22)^3 is 2^66.
    let path = tmp("elemoverflow");
    let _ = std::fs::remove_file(&path);
    let mut w = LbiWriter::create(&path, serde_json::json!({})).unwrap();
    let dim = 1u64 << 22;
    let err = w
        .append("w", &[dim, dim, dim], QuantScheme::Bf16, &[])
        .unwrap_err();
    assert!(matches!(err, LbiError::ShapeTooLarge { .. }), "got {err:?}");
}

#[test]
fn an_empty_payload_for_a_nonzero_shape_is_rejected() {
    // Guards the zero-tensor path: length must match the declared shape.
    let path = tmp("emptyfornonzero");
    let _ = std::fs::remove_file(&path);
    let mut w = LbiWriter::create(&path, serde_json::json!({})).unwrap();
    let err = w.append("w", &[4], QuantScheme::F32, &[]).unwrap_err();
    assert!(
        matches!(err, LbiError::LengthMismatch { .. }),
        "got {err:?}"
    );
}

/// The writer must not produce a file its own reader would refuse.
#[test]
fn an_oversized_config_is_refused_by_the_writer() {
    let path = tmp("bigcfg");
    let _ = std::fs::remove_file(&path);
    let big = "x".repeat((16 << 20) + 1024);
    let mut w = LbiWriter::create(&path, serde_json::json!({ "pad": big })).unwrap();
    w.append("w", &[1], QuantScheme::F32, &[0u8; 4]).unwrap();
    let err = w.finish().unwrap_err();
    assert!(matches!(err, LbiError::ConfigTooLarge(_)), "got {err:?}");
}

#[test]
fn a_valid_file_still_opens_after_all_these() {
    let b = built("control");
    let f = LbiFile::open(&b.path).unwrap();
    assert_eq!(f.len(), 1);
    assert_eq!(f.get("w").unwrap().shape, vec![4, 4]);
    let _ = std::fs::remove_file(b.path);
}

/// A safetensors header that declares the same tensor twice must be rejected:
/// `serde_json::Value` would keep only the last entry and silently drop the
/// first payload.
#[test]
fn a_safetensors_header_with_a_repeated_tensor_key_is_rejected() {
    use lumen_image::safetensors::SafetensorsFile;

    let dir = std::env::temp_dir();
    let path = dir.join(format!("st-dup-{}.safetensors", std::process::id()));
    // Hand-built JSON: the same key twice, with different byte ranges.
    let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
                      "w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&[0u8; 8]);
    std::fs::write(&path, &bytes).unwrap();

    let err = SafetensorsFile::open(&path).unwrap_err();
    assert!(
        err.to_string().contains("more than once"),
        "a repeated tensor key must be reported, got {err}"
    );
    let _ = std::fs::remove_file(path);
}

/// The shard index must reject a `weight_map` naming a tensor twice.
#[test]
fn a_weight_map_with_a_repeated_tensor_key_is_rejected() {
    use lumen_image::shard_index::ShardIndex;

    let dir = std::env::temp_dir().join(format!("idx-dup-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let index = br#"{"weight_map":{"a":"s.safetensors","a":"t.safetensors"}}"#;
    std::fs::write(dir.join("model.safetensors.index.json"), index).unwrap();

    let err = ShardIndex::open_dir(&dir).unwrap_err();
    assert!(
        err.to_string().contains("more than once"),
        "a repeated weight_map key must be reported, got {err}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A shard named in a subdirectory must not make a top-level shard of the same
/// file name look referenced: they are different files, and the top-level one
/// holding an unindexed tensor would otherwise be dropped silently.
#[test]
fn a_subdirectory_shard_does_not_mask_a_toplevel_one() {
    use lumen_image::shard_index::ShardIndex;

    let dir = std::env::temp_dir().join(format!("idx-sub-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("nested")).unwrap();

    // The indexed shard lives in nested/; a separate top-level shard exists too.
    let header = |name: &str| {
        let h = format!(r#"{{"{name}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#);
        let mut b = (h.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(h.as_bytes());
        b.extend_from_slice(&[0u8; 4]);
        b
    };
    std::fs::write(dir.join("nested/model.safetensors"), header("a")).unwrap();
    std::fs::write(dir.join("model.safetensors"), header("b")).unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        br#"{"weight_map":{"a":"nested/model.safetensors"}}"#,
    )
    .unwrap();

    let index = ShardIndex::open_dir(&dir).unwrap();
    let cross = index.cross_check().unwrap();
    assert!(
        !cross.is_clean(),
        "a top-level model.safetensors holding an unindexed tensor must be reported"
    );
    // Exactly the top-level shard is unreferenced. The reported path is
    // absolute, so compare the file name rather than the whole string.
    assert_eq!(
        cross.unreferenced.len(),
        1,
        "expected only the top-level shard, got {:?}",
        cross.unreferenced
    );
    assert_eq!(
        std::path::Path::new(&cross.unreferenced[0])
            .file_name()
            .and_then(|f| f.to_str()),
        Some("model.safetensors")
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A backslash is an ordinary filename character on Unix, so a file named
/// `nested\model.safetensors` is NOT the same file as `nested/model.safetensors`.
/// Treating them as one hid an unindexed shard.
#[test]
fn a_literal_backslash_filename_is_not_a_path_separator() {
    use lumen_image::shard_index::ShardIndex;

    let dir = std::env::temp_dir().join(format!("idx-bslash-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("nested")).unwrap();

    let header = |name: &str| {
        let h = format!(r#"{{"{name}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#);
        let mut b = (h.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(h.as_bytes());
        b.extend_from_slice(&[0u8; 4]);
        b
    };
    std::fs::write(dir.join("nested/model.safetensors"), header("a")).unwrap();
    // A single file whose name contains a backslash.
    std::fs::write(dir.join("nested\\model.safetensors"), header("b")).unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        br#"{"weight_map":{"a":"nested/model.safetensors"}}"#,
    )
    .unwrap();

    let index = ShardIndex::open_dir(&dir).unwrap();
    let cross = index.cross_check().unwrap();
    assert!(
        !cross.is_clean(),
        "the backslash-named shard was treated as the indexed one: {cross:?}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// Equivalent spellings of the same path must all count as referenced, so a
/// legitimate index is not rejected for its spelling.
#[test]
fn equivalent_path_spellings_are_recognised() {
    use lumen_image::shard_index::ShardIndex;

    let dir = std::env::temp_dir().join(format!("idx-equiv-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let h = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut b = (h.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(h);
    b.extend_from_slice(&[0u8; 4]);
    std::fs::write(dir.join("model.safetensors"), &b).unwrap();

    for spelling in [
        "./model.safetensors",
        "././model.safetensors",
        "model.safetensors",
    ] {
        let idx = format!(r#"{{"weight_map":{{"a":"{spelling}"}}}}"#);
        std::fs::write(dir.join("model.safetensors.index.json"), idx).unwrap();
        let index = ShardIndex::open_dir(&dir).unwrap();
        let cross = index.cross_check().unwrap();
        assert!(
            cross.is_clean(),
            "spelling {spelling} should be recognised as the same file: {cross:?}"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// A shard reachable only through a symlinked directory is still a shard. A
/// walk that does not follow directory links would miss an unindexed tensor.
#[cfg(unix)]
#[test]
fn a_symlinked_directory_is_walked() {
    use lumen_image::shard_index::ShardIndex;

    let root = std::env::temp_dir().join(format!("idx-symdir-{}", std::process::id()));
    let parts = std::env::temp_dir().join(format!("idx-symdir-parts-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&parts);
    std::fs::create_dir_all(root.join("transformer")).unwrap();
    std::fs::create_dir_all(&parts).unwrap();

    let header = |name: &str| {
        let h = format!(r#"{{"{name}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#);
        let mut b = (h.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(h.as_bytes());
        b.extend_from_slice(&[0u8; 4]);
        b
    };
    std::fs::write(parts.join("a.safetensors"), header("a")).unwrap();
    std::fs::write(parts.join("b.safetensors"), header("b")).unwrap();
    std::os::unix::fs::symlink(&parts, root.join("transformer/parts")).unwrap();
    std::fs::write(
        root.join("transformer/model.safetensors.index.json"),
        br#"{"weight_map":{"a":"parts/a.safetensors"}}"#,
    )
    .unwrap();

    let index = ShardIndex::open_dir(&root.join("transformer")).unwrap();
    let cross = index.cross_check().unwrap();
    assert!(
        !cross.is_clean(),
        "b lives behind a symlinked directory and must be reported: {cross:?}"
    );
    assert!(
        cross
            .unreferenced
            .iter()
            .any(|u| u.ends_with("b.safetensors")),
        "expected b.safetensors unreferenced, got {:?}",
        cross.unreferenced
    );
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&parts);
}

/// The single-shard path (no index) must refuse a component directory holding
/// more than one shard, wherever the extra one lives.
#[test]
fn a_second_untracked_shard_is_refused_without_an_index() {
    use lumen_image::shard_index::ShardIndex;

    let dir = std::env::temp_dir().join(format!("idx-multi-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("nested")).unwrap();

    let header = |name: &str| {
        let h = format!(r#"{{"{name}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#);
        let mut b = (h.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(h.as_bytes());
        b.extend_from_slice(&[0u8; 4]);
        b
    };
    std::fs::write(dir.join("model.safetensors"), header("a")).unwrap();
    std::fs::write(dir.join("nested/extra.safetensors"), header("b")).unwrap();

    let err = ShardIndex::single_shard(&dir).unwrap_err();
    assert!(
        err.to_string().contains("expected one shard"),
        "a hidden second shard must be refused, got {err}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// One file, referenced by two spellings that resolve to the same file, is not
/// a duplicate. Comparing names as strings would wrongly reject it.
#[test]
fn two_spellings_of_one_shard_are_not_a_duplicate() {
    use lumen_image::shard_index::ShardIndex;

    let dir = std::env::temp_dir().join(format!("idx-alias-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let h = (br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#).to_vec();
    let mut b = (h.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(&h);
    b.extend_from_slice(&[0u8; 8]);
    std::fs::write(dir.join("model.safetensors"), &b).unwrap();
    // Both tensors live in one file, named twice with different spellings.
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        br#"{"weight_map":{"a":"model.safetensors","b":"./model.safetensors"}}"#,
    )
    .unwrap();

    let index = ShardIndex::open_dir(&dir).unwrap();
    let cross = index.cross_check().unwrap();
    assert!(
        cross.duplicated.is_empty(),
        "equivalent spellings must not read as a duplicate: {:?}",
        cross.duplicated
    );
    assert!(cross.is_clean(), "expected a clean check, got {cross:?}");
    let _ = std::fs::remove_dir_all(dir);
}

/// One physical shard reachable under two names — a hard link — is one shard.
/// Treating the names as two files both opens the same tensors twice and
/// reports a duplicate that is not there.
#[test]
fn a_hard_link_alias_is_one_shard() {
    use lumen_image::shard_index::ShardIndex;

    let dir = std::env::temp_dir().join(format!("idx-hard-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let h = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    let mut b = (h.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(h);
    b.extend_from_slice(&[0u8; 8]);
    std::fs::write(dir.join("model.safetensors"), &b).unwrap();
    std::fs::hard_link(dir.join("model.safetensors"), dir.join("alias.safetensors")).unwrap();
    // Both tensors live in the one physical file, named two ways.
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        br#"{"weight_map":{"a":"model.safetensors","b":"alias.safetensors"}}"#,
    )
    .unwrap();

    let index = ShardIndex::open_dir(&dir).unwrap();
    let cross = index.cross_check().unwrap();
    assert!(
        cross.is_clean(),
        "a hard link is one shard and must not read as a duplicate: {cross:?}"
    );

    // With no index, the same directory is one shard, not two.
    std::fs::remove_file(dir.join("model.safetensors.index.json")).unwrap();
    let one = ShardIndex::single_shard(&dir).unwrap();
    assert!(one.exists());
    let _ = std::fs::remove_dir_all(dir);
}

/// A tensor assigned to the wrong file must be reported, even though every
/// declared tensor exists somewhere and no name is repeated.
#[test]
fn a_tensor_placed_in_the_wrong_shard_is_reported() {
    use lumen_image::shard_index::ShardIndex;

    let dir = std::env::temp_dir().join(format!("idx-swap-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let one = |name: &str| {
        let h = format!(r#"{{"{name}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#);
        let mut b = (h.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(h.as_bytes());
        b.extend_from_slice(&[0u8; 4]);
        b
    };
    // a lives in a.safetensors, b in b.safetensors.
    std::fs::write(dir.join("a.safetensors"), one("a")).unwrap();
    std::fs::write(dir.join("b.safetensors"), one("b")).unwrap();
    // The index swaps them.
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        br#"{"weight_map":{"a":"b.safetensors","b":"a.safetensors"}}"#,
    )
    .unwrap();

    let index = ShardIndex::open_dir(&dir).unwrap();
    let cross = index.cross_check().unwrap();
    assert!(
        !cross.is_clean(),
        "a swapped assignment must not read as clean: {cross:?}"
    );
    // Two discrepancies per file: the tensor declared for it is absent, and the
    // tensor present in it is not declared for it.
    assert_eq!(
        cross.misplaced.len(),
        4,
        "both files must report both directions: {:?}",
        cross.misplaced
    );
    let _ = std::fs::remove_dir_all(dir);
}
