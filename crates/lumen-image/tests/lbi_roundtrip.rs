//! Round-trip and rejection tests for the `.lbi` container.
//!
//! The load-bearing property is that stored bytes come back identical. These
//! tests assert that on F32 and BF16, and — just as importantly — that a
//! one-byte change is *detected*, so the round-trip check cannot pass
//! vacuously.

use std::io::Write;

use lumen_format::QuantScheme;
use lumen_image::lbi::{LbiError, LbiFile, LbiWriter};

fn tmp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("lbi-test-{}-{}", std::process::id(), name));
    p
}

fn write_file(
    name: &str,
    tensors: &[(&str, Vec<u64>, QuantScheme, Vec<u8>)],
) -> std::path::PathBuf {
    let path = tmp_path(name);
    let _ = std::fs::remove_file(&path);
    let mut w = LbiWriter::create(&path, serde_json::json!({"probe": true})).unwrap();
    for (n, shape, q, data) in tensors {
        w.append(n, shape, *q, data).unwrap();
    }
    w.finish().unwrap();
    path
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        let bits = (v.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

#[test]
fn f32_round_trips_bit_exact() {
    let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.37).collect();
    let mut data = Vec::new();
    for v in &values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let path = write_file("f32", &[("w", vec![8, 8], QuantScheme::F32, data.clone())]);

    let f = LbiFile::open(&path).unwrap();
    assert_eq!(f.len(), 1);
    let got = f.tensor_bytes("w").unwrap();
    assert_eq!(got, &data[..], "stored bytes must be identical");
    assert_eq!(f.read_f32("w").unwrap(), values);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn bf16_round_trips_bit_exact() {
    let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
    let data = bf16_bytes(&values);
    let path = write_file("bf16", &[("w", vec![64], QuantScheme::Bf16, data.clone())]);

    let f = LbiFile::open(&path).unwrap();
    assert_eq!(f.tensor_bytes("w").unwrap(), &data[..]);
    // Decoding is exact for values already representable in bf16.
    let decoded = f.read_f32("w").unwrap();
    for (a, b) in decoded.iter().zip(&values) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn shapes_and_names_survive() {
    let data = vec![0u8; 2 * 3 * 4 * 4]; // 2x3x4 f32
    let path = write_file(
        "shapes",
        &[(
            "transformer.blocks.0.conv.weight",
            vec![2, 3, 4],
            QuantScheme::F32,
            data,
        )],
    );
    let f = LbiFile::open(&path).unwrap();
    let e = f.get("transformer.blocks.0.conv.weight").unwrap();
    assert_eq!(e.shape, vec![2, 3, 4]);
    assert_eq!(e.num_elements().unwrap(), 24);
    std::fs::remove_file(path).unwrap();
}

/// The negative control: change one byte of the payload and the read must differ.
///
/// Without this, "the bytes match" could hold for a reader that ignores the blob
/// region entirely.
#[test]
fn a_changed_payload_byte_is_detected() {
    let values: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mut data = Vec::new();
    for v in &values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let path = write_file("negcontrol", &[("w", vec![16], QuantScheme::F32, data)]);

    let before = LbiFile::open(&path).unwrap().read_f32("w").unwrap();
    assert_eq!(before, values);

    // Flip the last byte of the file (inside the blob region).
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    let mut fh = std::fs::File::create(&path).unwrap();
    fh.write_all(&bytes).unwrap();
    fh.sync_all().unwrap();
    drop(fh);

    let after = LbiFile::open(&path).unwrap().read_f32("w").unwrap();
    assert_ne!(
        after, before,
        "a flipped payload byte must change the decoded tensor"
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn wrong_length_is_rejected() {
    let path = tmp_path("badlen");
    let _ = std::fs::remove_file(&path);
    let mut w = LbiWriter::create(&path, serde_json::json!({})).unwrap();
    // 4 f32 elements need 16 bytes, not 12.
    let err = w
        .append("w", &[4], QuantScheme::F32, &[0u8; 12])
        .unwrap_err();
    match err {
        LbiError::LengthMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, 16);
            assert_eq!(actual, 12);
        }
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn duplicate_name_is_rejected() {
    let path = tmp_path("dup");
    let _ = std::fs::remove_file(&path);
    let mut w = LbiWriter::create(&path, serde_json::json!({})).unwrap();
    w.append("w", &[2], QuantScheme::F32, &[0u8; 8]).unwrap();
    let err = w
        .append("w", &[2], QuantScheme::F32, &[0u8; 8])
        .unwrap_err();
    assert!(matches!(err, LbiError::DuplicateTensor(_)), "got {err:?}");
}

#[test]
fn bad_magic_is_rejected() {
    let path = tmp_path("magic");
    // At least a full header, so the magic is what fails and not the length.
    let mut bytes = vec![0u8; 64];
    bytes[0..4].copy_from_slice(b"XXXX");
    bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let err = LbiFile::open(&path).unwrap_err();
    assert!(matches!(err, LbiError::BadMagic(_)), "got {err:?}");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn truncated_file_is_rejected() {
    let data = vec![0u8; 32];
    let path = write_file("trunc", &[("w", vec![8], QuantScheme::F32, data)]);
    let full = std::fs::read(&path).unwrap();
    for cut in [8usize, 20, full.len() / 2, full.len() - 1] {
        let part = tmp_path(&format!("trunc{cut}"));
        std::fs::write(&part, &full[..cut]).unwrap();
        assert!(
            LbiFile::open(&part).is_err(),
            "a file truncated to {cut} bytes must not open"
        );
        std::fs::remove_file(part).unwrap();
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn unresolved_scheme_is_rejected() {
    let path = tmp_path("scheme");
    let _ = std::fs::remove_file(&path);
    let mut w = LbiWriter::create(&path, serde_json::json!({})).unwrap();
    let err = w
        .append("w", &[32], QuantScheme::Q8_0, &[0u8; 34])
        .unwrap_err();
    assert!(matches!(err, LbiError::UnsupportedScheme(_)), "got {err:?}");
}

#[test]
fn empty_file_opens() {
    let path = tmp_path("empty");
    let _ = std::fs::remove_file(&path);
    LbiWriter::create(&path, serde_json::json!({"k": 1}))
        .unwrap()
        .finish()
        .unwrap();
    let f = LbiFile::open(&path).unwrap();
    assert_eq!(f.len(), 0);
    assert!(f.is_empty());
    assert_eq!(f.config()["k"], 1);
    std::fs::remove_file(path).unwrap();
}

/// Regression: a tensor whose size leaves the cursor unaligned shifts the next
/// one, and the entry must point at the data rather than at the alignment pad.
///
/// The first version recorded the pre-padding cursor, so every tensor that
/// landed after a non-64-byte-multiple neighbour read back as zeros.
#[test]
fn tensors_after_unaligned_neighbours_read_back() {
    // 1 + 4 + 16 + 64 elements at 4 bytes: 4, 16, 64, 256 bytes.
    // Sizes 4 and 16 and 64 keep the cursor aligned only by luck; sizes that
    // leave a remainder force the pad path.
    let sizes = [3usize, 5, 1, 7, 33]; // bytes 12, 20, 4, 28, 132 — all unaligned
    let mut tensors = Vec::new();
    let mut data = Vec::new();
    for (i, n) in sizes.iter().enumerate() {
        let name = format!("t{i}");
        let vals: Vec<f32> = (0..*n).map(|k| (k as f32 + i as f32) + 0.25).collect();
        let mut b = Vec::new();
        for v in &vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
        tensors.push((name, vec![*n as u64], QuantScheme::F32, b.clone()));
        data.push(vals);
    }
    let refs: Vec<(&str, Vec<u64>, QuantScheme, Vec<u8>)> = tensors
        .iter()
        .map(|(n, s, q, b)| (n.as_str(), s.clone(), *q, b.clone()))
        .collect();
    let path = write_file("unaligned", &refs);

    let f = LbiFile::open(&path).unwrap();
    for (i, vals) in data.iter().enumerate() {
        let got = f.read_f32(&format!("t{i}")).unwrap();
        assert_eq!(&got, vals, "tensor t{i} did not survive the alignment pad");
    }
    std::fs::remove_file(path).unwrap();
}

/// A file open for reading stays whole while a new one is written to its path:
/// the writer assembles beside it and renames only in `finish`, so a reader's
/// mapping never sees a truncated file (reading one past its new end faults).
#[test]
fn rewriting_a_path_leaves_an_open_file_whole() {
    let old: Vec<u8> = (0..4096u32)
        .flat_map(|v| (v as f32).to_le_bytes())
        .collect();
    let path = write_file(
        "rewrite",
        &[("w", vec![4096], QuantScheme::F32, old.clone())],
    );
    let reader = LbiFile::open(&path).unwrap();

    let mut w = LbiWriter::create(&path, serde_json::json!({"probe": 2})).unwrap();
    assert_eq!(reader.tensor_bytes("w").unwrap(), &old[..], "while writing");
    w.append("w", &[2], QuantScheme::F32, &[0u8; 8]).unwrap();
    assert_eq!(
        LbiFile::open(&path).unwrap().config()["probe"],
        true,
        "the path names the old file until finish"
    );
    w.finish().unwrap();
    assert_eq!(
        reader.tensor_bytes("w").unwrap(),
        &old[..],
        "after the rename"
    );

    let new = LbiFile::open(&path).unwrap();
    assert_eq!(new.config()["probe"], 2);
    assert_eq!(new.tensor_bytes("w").unwrap(), &[0u8; 8][..]);
    assert!(!path.with_extension("lbi.part").exists());
    std::fs::remove_file(path).unwrap();
}
