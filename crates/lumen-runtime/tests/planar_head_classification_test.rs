//! Output-head and embedding classification is header-tag-first, so a planar (NVFP4 or FP8) plane cannot be
//! misread by the byte-length cascade, and a planar plane of the wrong length is refused by name.
//!
//! The hazard this pins: an NVFP4 head's packed weights and block scales take `n/2 + n/16` bytes, exactly
//! Q4_0's `(n/32) * 18` for any `n` divisible by 32, and the stored plane is 4 bytes longer for its F32
//! global scale (8 with the head's activation scale after it). A length-first cascade therefore MISSES the
//! Q4_0 arm (the lengths differ) and ACCEPTS the F32 fallback (the length is a multiple of 4), producing a
//! head of the wrong element count — silently wrong numbers from a file that looks valid. The header tag
//! must decide.
//!
//! Each case is driven through the production classifiers (`read_output_proj_global`,
//! `read_embedding_global`), so what is tested is the decision the loader makes, not a restatement of it.
//!
//! These are pure functions over bytes, so no GPU is needed:
//!   cargo test --release -p lumen-runtime --test planar_head_classification_test

use lumen_format::{Nvfp4Planes, QuantScheme};
use lumen_runtime::weight::provider_sync::{read_embedding_global, read_output_proj_global};

/// Small enough to build in a test; the byte lengths collide as they do at any size (NVFP4 = n/2 + n/16 ==
/// Q4_0's (n/32)*18 for any n divisible by 32).
const VOCAB: usize = 320;
const HIDDEN: usize = 512;

fn n_elements() -> usize {
    VOCAB * HIDDEN
}

/// The head's planes, weight | block_scale | global_scale(4B), without the optional activation scale.
fn head_planes() -> usize {
    Nvfp4Planes::for_shape(VOCAB as u64, HIDDEN as u64)
        .unwrap()
        .total_bytes() as usize
}

#[test]
fn an_nvfp4_head_is_classified_by_header_and_keeps_its_bytes() {
    // The planes alone, and the planes followed by the head's 4-byte activation scale.
    for len in [head_planes(), head_planes() + 4] {
        let plane = vec![0xABu8; len];
        let (f32_data, raw, quant) =
            read_output_proj_global(plane.clone(), VOCAB, HIDDEN, QuantScheme::Nvfp4)
                .unwrap_or_else(|e| panic!("planar head of {len} bytes: {e}"));
        assert_eq!(
            quant,
            QuantScheme::Nvfp4,
            "the header tag must decide the scheme"
        );
        assert_eq!(
            raw, plane,
            "the stored bytes are kept as they are for the CUDA head kernel"
        );
        // No F32 form exists for this scheme here, so the F32 copy is EMPTY rather than a wrong reading.
        assert!(
            f32_data.is_empty(),
            "an NVFP4 head must not produce an F32 copy: {} elements would be a misread",
            f32_data.len()
        );
    }
}

#[test]
fn the_q4_0_length_plane_is_still_classified_by_header() {
    // `n/2 + n/16` is Q4_0's byte count exactly. Without the header tag this would be read as Q4_0. With
    // the tag it is refused as short (the global scale is missing), by name, and never read as Q4_0.
    let n = n_elements();
    let q4_len = (n / 32) * 18;
    assert_eq!(
        n / 2 + n / 16,
        q4_len,
        "the collision must be real for this test to mean anything"
    );
    let err = read_output_proj_global(vec![0x11u8; q4_len], VOCAB, HIDDEN, QuantScheme::Nvfp4)
        .expect_err("a Q4_0-length NVFP4 head is short and must be refused");
    assert!(format!("{err}").contains("Nvfp4"), "refused by name: {err}");
    // And the SAME bytes under a Q4_0 header really do classify as Q4_0, so the test above is not vacuous:
    // the difference is the header, not the length.
    let (_f32, _raw, quant) =
        read_output_proj_global(vec![0x11u8; q4_len], VOCAB, HIDDEN, QuantScheme::Q4_0)
            .expect("Q4_0 header");
    assert_eq!(
        quant,
        QuantScheme::Q4_0,
        "the same bytes under a Q4_0 header are Q4_0"
    );
}

#[test]
fn a_truncated_plane_is_refused_by_name() {
    for short_by in [1usize, 4, 100] {
        let len = head_planes() - short_by;
        let err = read_output_proj_global(vec![0x11u8; len], VOCAB, HIDDEN, QuantScheme::Nvfp4)
            .expect_err("a truncated NVFP4 head must be refused");
        let msg = format!("{err}");
        assert!(msg.contains("Nvfp4"), "refused by name: {msg}");
        assert!(
            msg.contains(&format!("is {len} bytes"))
                && msg.contains(&format!(
                    "exactly {}, or {}",
                    head_planes(),
                    head_planes() + 4
                )),
            "and stating the length and the two it may have: {msg}"
        );
    }
}

#[test]
fn an_over_long_plane_is_refused() {
    // The converter writes the planes, then at most the head's 4-byte activation scale, with no padding, so
    // any other length past the planes is a malformed plane and is refused by name.
    for extra in [1usize, 3, 5, 8, 64] {
        let len = head_planes() + extra;
        let err = read_output_proj_global(vec![0x11u8; len], VOCAB, HIDDEN, QuantScheme::Nvfp4)
            .expect_err("an over-long NVFP4 head must be refused");
        let msg = format!("{err}");
        assert!(msg.contains("Nvfp4"), "refused by name: {msg}");
        assert!(
            msg.contains(&format!("is {len} bytes"))
                && msg.contains(&format!(
                    "exactly {}, or {}",
                    head_planes(),
                    head_planes() + 4
                )),
            "and stating the length and the two it may have: {msg}"
        );
    }
}

#[test]
fn an_fp8_head_is_refused_by_name_at_any_length() {
    // No backend serves an FP8 output head: short or exact, the plane is refused by name rather than left
    // to the length cascade.
    let n = n_elements();
    for len in [n + 3, n + 4] {
        let err = read_output_proj_global(vec![0x38u8; len], VOCAB, HIDDEN, QuantScheme::Fp8E4M3)
            .expect_err("an FP8 head must be refused");
        assert!(
            format!("{err}").contains("Fp8E4M3"),
            "refused by name: {err}"
        );
    }
}

#[test]
fn a_planar_embedding_is_refused_by_name() {
    // No backend gathers an embedding from a planar scheme, so a well-formed plane is refused by name and
    // never read as F32.
    let n = n_elements();
    for (quant, len) in [
        (QuantScheme::Nvfp4, head_planes()),
        (QuantScheme::Fp8E4M3, n + 4),
    ] {
        let err = read_embedding_global(vec![0x5Au8; len], VOCAB, HIDDEN, quant)
            .expect_err("a planar embedding must be refused");
        assert!(
            format!("{err}").contains(&format!("{quant:?}")),
            "refused by name: {err}"
        );
    }
}

#[test]
fn a_normal_f32_head_still_takes_the_f32_path() {
    // A scheme the cascade already handled must be unaffected by the planar branch.
    let n = n_elements();
    let vals: Vec<f32> = (0..n).map(|k| (k % 97) as f32 * 0.5 - 24.0).collect();
    let mut bytes = Vec::with_capacity(n * 4);
    for v in &vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let (f32_data, _raw, quant) =
        read_output_proj_global(bytes, VOCAB, HIDDEN, QuantScheme::F32).expect("f32 head");
    assert_eq!(quant, QuantScheme::F32);
    assert_eq!(f32_data.len(), n, "the F32 head is read as before");
    assert_eq!(f32_data[10], vals[10], "and its values are unchanged");
}
