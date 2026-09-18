//! The explicit `LUMEN_CONVERT_SOURCE_FIDELITY` switch is outside the servability rule
//! the K-quant source default applies to the F32 GDN gates: it keeps a gate of any
//! stored extent, which is 0.31.0's answer under the same switch. Its own test binary,
//! because it sets a process-wide environment variable that the default fixtures in
//! `kquant_gate_extent.rs` must not see.
mod common;

use common::{build, convert_gates, generic, GATE};
use lumen_format::quantization::QuantScheme;

#[test]
fn the_explicit_switch_keeps_a_gate_of_any_extent() {
    std::env::set_var("LUMEN_CONVERT_SOURCE_FIDELITY", "1");
    let (_, alpha, beta) = convert_gates("fidelity", &build(GATE - 32, GATE), &generic());
    std::env::remove_var("LUMEN_CONVERT_SOURCE_FIDELITY");
    let source: Vec<u8> = (0..GATE - 32).flat_map(|_| 0.02f32.to_le_bytes()).collect();
    assert_eq!(
        alpha.0,
        QuantScheme::F32,
        "the switch stopped keeping a short gate"
    );
    assert_eq!(alpha.1.len(), 3_968, "0.31.0 kept 992 F32 elements");
    assert_eq!(
        alpha.1, source,
        "the kept ssm_alpha is not the source bytes"
    );
    assert_eq!(
        beta.0,
        QuantScheme::F32,
        "the well-sized gate was requantised"
    );
}
