//! What a K-quant source preserves by default has to be the plane the runtime reads.
//! The GDN gate projections read `ssm_alpha` / `ssm_beta` at `num_v_heads x hidden`
//! and refuse a buffer shorter than that at the first token; a gate stored at any
//! other extent is not the plane the projection reads. Either way the conversion takes
//! the Q8_0 gate 0.31.0's default wrote for it rather than keeping the F32 plane. The
//! explicit `LUMEN_CONVERT_SOURCE_FIDELITY` switch is outside this rule and unchanged
//! — `kquant_gate_extent_fidelity.rs` pins it — so these fixtures set no environment.
//!
//! GGUF sizes a tensor from its flattened element count, so a gate of any extent is a
//! file the converter reads without complaint — the source of the first two fixtures.
//! No GDN export stores a gate of another extent, so no shipped file converts
//! differently; the pins are the guard against the default widening to one that does.
//!
//! The pins were derived by building these same fixtures against the 0.31.0 converter
//! (release commit `1958662`, `crates/lumen-convert` unmodified) in a throwaway
//! worktree and hashing the gate planes of its artifact.
mod common;

use common::{build, convert_gates, generic, GATE};
use lumen_format::quantization::QuantScheme;

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The 0.31.0 gate planes of the two mis-sized fixtures: the F32 source gate
/// requantised to Q8_0, 1 054 bytes at 992 elements and 2 176 at 2 048.
const SHORT_GATE_0_31_0: &str = "35289a52c75b282b162b9a91a9fc12dc55af548ff4883fc79ac8fe989299148a";
const LONG_GATE_0_31_0: &str = "4b4abb2e455516f1c8481424540485291a39df18d6e2e43c9d1f3f97a2991d04";

#[test]
fn a_gate_shorter_than_the_projection_reads_takes_the_0_31_0_plane() {
    let (primary, alpha, beta) = convert_gates("short", &build(GATE - 32, GATE), &generic());
    assert_eq!(
        primary,
        QuantScheme::Q4_K,
        "fixture is not a K-quant source"
    );
    assert_eq!(
        alpha.0,
        QuantScheme::Q8_0,
        "a gate the projection cannot read whole was carried verbatim"
    );
    assert_eq!(alpha.1.len(), 1_054, "the 0.31.0 gate is 1 054 bytes");
    assert_eq!(
        sha256_hex(&alpha.1),
        SHORT_GATE_0_31_0,
        "the ssm_alpha plane moved from 0.31.0"
    );
    assert_eq!(
        beta.0,
        QuantScheme::F32,
        "the well-sized ssm_beta beside it was not kept"
    );
}

#[test]
fn a_gate_longer_than_the_projection_reads_takes_the_0_31_0_plane() {
    let (primary, alpha, beta) = convert_gates("long", &build(GATE, 2 * GATE), &generic());
    assert_eq!(
        primary,
        QuantScheme::Q4_K,
        "fixture is not a K-quant source"
    );
    assert_eq!(
        beta.0,
        QuantScheme::Q8_0,
        "a gate longer than the projection reads was carried verbatim"
    );
    assert_eq!(beta.1.len(), 2_176, "the 0.31.0 gate is 2 176 bytes");
    assert_eq!(
        sha256_hex(&beta.1),
        LONG_GATE_0_31_0,
        "the ssm_beta plane moved from 0.31.0"
    );
    assert_eq!(
        alpha.0,
        QuantScheme::F32,
        "the well-sized ssm_alpha beside it was not kept"
    );
}

/// The extent the rule is asked about is `num_v_heads x hidden`, and a gate stored at
/// it is the plane the projection reads: kept as the source F32 bytes.
#[test]
fn gates_at_the_extent_the_projection_reads_are_kept_as_stored() {
    let (_, alpha, beta) = convert_gates("exact", &build(GATE, GATE), &generic());
    let source: Vec<u8> = (0..GATE).flat_map(|_| 0.02f32.to_le_bytes()).collect();
    assert_eq!(
        alpha.0,
        QuantScheme::F32,
        "the F32 ssm_alpha was requantised"
    );
    assert_eq!(beta.0, QuantScheme::F32, "the F32 ssm_beta was requantised");
    assert_eq!(
        alpha.1, source,
        "the kept ssm_alpha is not the source bytes"
    );
    assert_eq!(beta.1, source, "the kept ssm_beta is not the source bytes");
}
