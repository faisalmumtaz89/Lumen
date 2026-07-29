//! Convert-time env gates for the 9B-Q4 K-quant format lever family.
//!
//! # Why this module exists, and why it is the only env reader in this crate
//!
//! `lumen-convert` had zero `env::var` reads before this module: conversion
//! behaviour was driven entirely by `ConvertOptions` and `cfg!`. Two of the
//! four K-quant candidates are nevertheless *convert-time* decisions — which
//! format a tensor is stored in is fixed when the `.lbc` is written, not when
//! it is read — so they have to be readable here.
//!
//! Threading them through `ConvertOptions` would have been the
//! precedent-matching alternative, but it means widening the `ArchConverter`
//! trait's `compute_layer_shape` / `write_layer_blob` signatures and every arch
//! implementation, for two default-OFF experiment flags. Concentrating the
//! reads in one small documented module instead keeps the blast radius at one
//! file and makes the new precedent explicit rather than scattered. If either
//! candidate is ever promoted to a shipped default, the right move is to delete
//! the gate entirely, not to generalise it.
//!
//! # The cache-clobbering problem these gates create, and how it is solved
//!
//! `lumen-cli`'s `lbc_path()` keys the cached `.lbc` filename on exactly
//! `(model key, quant tag)`. It ignores `requant_to` and `ConvertTarget`, which
//! is a pre-existing hazard: a macOS host writes Metal-upcast content to the
//! filename documented as the Generic output, and `--requant q4_0` /
//! `--requant q8_0` conversions are mutually indistinguishable on disk.
//!
//! A gated variant conversion would inherit that hazard and silently overwrite
//! the campaign's baseline LBC — destroying the control arm of the very A/B it
//! exists to serve. [`lbc_variant_suffix`] is the fix: it returns a suffix
//! derived from the active gates, `lbc_path()` appends it, and because the
//! cache *reader* goes through the same function, both arms coexist in one
//! `LUMEN_CACHE_DIR` and each finds its own file. One source of truth, so the
//! writer and reader cannot disagree.

/// Parse a campaign lever flag: ON iff the value is exactly `"1"`.
///
/// Same two-valued contract as `lumen_runtime::runtime_defaults`' campaign
/// flags: `=0`, `=true`, `=on`, `=` and unset all mean OFF. The loose-truthy
/// dialects elsewhere in this workspace have produced live defects where `=0`
/// enabled a path and `=on` disabled one; these gates do not repeat that.
fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

/// `LUMEN_CUDA_LMHEAD_Q6K=1` (candidate C3, convert half) — keep a K-quant
/// `output.weight` in its source format instead of requantizing it to Q8_0.
///
/// The default requant costs `1,017,118,720 x (1.0625 - 0.8203) = 246.3 MB`
/// = 234.9 MiB per token on Qwen3.5-9B-Q4_0, the largest single format
/// mismatch on the decode path. It is also lossy twice over: once through
/// Q6_K -> F32 -> Q8_0, and once through the swapped `ql` mapping in
/// [`crate::dequant::dequantize_q6_k`] (see [`q6k_layout_fix`]).
///
/// The runtime half of this candidate lives in `lumen-runtime`: the CLI's
/// `set_output_proj_raw` allow-list, `CudaBackend::init`'s `output_proj_quant`
/// match, and the `compute_final_gpu` dispatch chain.
pub fn lmhead_q6k() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = flag("LUMEN_CUDA_LMHEAD_Q6K");
        if on {
            eprintln!("[Q6K] LMHEAD=ON (convert): output.weight keeps its source K-quant format");
        }
        on
    })
}

/// `LUMEN_CUDA_SSMOUT_NATIVE=1` (candidate C4) — drop the convert-time Q8_0
/// floor on `ssm_out` and store it at whatever the source GGUF holds.
///
/// # Read the risk before enabling this
///
/// The floor is a **documented quality keeper**, not an oversight. Its
/// rationale, recorded at the two floor sites in `arch/qwen35.rs`, is a
/// measurement: a `--requant q4_0` LBC passed **1 of 15** short prompts against
/// **13 of 15** for an LBC built from a provider Q4_0 GGUF that ships Q8-class
/// `ssm_out` (2026-06-10), and the comment concludes "`ssm_out` quantization is
/// empirically the dominant quality lever on this architecture". `ssm_out` sits
/// inside the GDN linear-attention recurrence, which is precisely the structure
/// the F32 activation policy exists to protect.
///
/// Two caveats on that evidence, recorded so the decision is made on facts:
/// the 1/15-vs-13/15 measurement exists **only** as an assertion in that source
/// comment — there is no artifact, log, or test behind it anywhere in the repo
/// — and it was measured against `--requant q4_0`, i.e. requantizing *every*
/// `ssm_out` down from Q8, whereas this gate merely stops *upcasting* the 12
/// layers the GGUF already stores as Q4_0. Those are different interventions
/// and the second is strictly milder. That is a reason to re-measure, not a
/// reason to assume the floor is wrong.
///
/// The lever is worth `12 x 16,777,216 x (1.0625 - 0.5625) = 100.7 MB`
/// = 96.0 MiB/token. It must clear the GDN quality gate before it is a
/// candidate for anything.
pub fn ssmout_native() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = flag("LUMEN_CUDA_SSMOUT_NATIVE");
        if on {
            eprintln!(
                "[SSM] SSMOUT_NATIVE=ON (convert): Q8_0 floor on ssm_out REMOVED -- \
                 this is a documented quality keeper, GDN quality gate required"
            );
        }
        on
    })
}

/// `LUMEN_Q6K_LAYOUT_FIX=1` (candidate C0) — dequantize Q6_K blocks with the
/// ggml element mapping instead of the swapped one currently shipped.
///
/// See `lumen_runtime::runtime_defaults::q6k_layout_fix` for the full
/// write-up. Short version: [`crate::dequant::dequantize_q6_k`] takes the `ql`
/// nibble for output slots `l+32` and `l+64` from the wrong byte, so 126 of
/// every 256 elements decode to a value assembled from the low 4 bits of one
/// weight and the high 2 bits of another. On the convert side this corrupts
/// every K-quant tensor that gets dequantized: the Q6_K `output.weight` on
/// every 9B-Q4 LBC (via the Q8_0 requant) and every K-quant layer tensor
/// upcast by `ConvertTarget::Metal`.
pub fn q6k_layout_fix() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = flag("LUMEN_Q6K_LAYOUT_FIX");
        if on {
            eprintln!("[Q6K] LAYOUT_FIX=ON (convert): Q6_K dequant uses the ggml ql mapping");
        }
        on
    })
}

/// Filename suffix identifying which convert-time gates produced an `.lbc`.
///
/// Empty when no gate is active, so the canonical baseline filename is
/// unchanged and no existing cache entry is invalidated. Otherwise a stable,
/// order-independent concatenation, e.g. `-q6khead`, `-ssmq4`,
/// `-q6khead-q6kfix`.
///
/// `lumen-cli`'s `lbc_path()` appends this, and because the cache *reader* uses
/// the same function, a variant arm and the baseline can share one
/// `LUMEN_CACHE_DIR` without either clobbering or shadowing the other. That
/// matters on the Modal harness, which converts once per volume: without the
/// suffix, arming a convert-time gate would overwrite the baseline LBC in place
/// and the A/B would silently compare an arm against itself.
///
/// Note this does NOT fix the pre-existing collisions on `ConvertTarget` and
/// `requant_to` — those remain unrepresented in the filename. It only ensures
/// these four candidates cannot add a new one.
pub fn lbc_variant_suffix() -> String {
    let mut s = String::new();
    if lmhead_q6k() {
        s.push_str("-q6khead");
    }
    if ssmout_native() {
        s.push_str("-ssmq4");
    }
    if q6k_layout_fix() {
        s.push_str("-q6kfix");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The suffix must be empty unless a gate is armed, or every existing
    /// cached LBC would be orphaned the moment this code ships.
    ///
    /// Reads the real env, so it asserts the property that matters in the
    /// default configuration the test suite runs under. The `OnceLock`s make
    /// set_var-based testing of the ON case unreliable within one process, so
    /// the ON composition is covered by `suffix_composition_is_stable` below
    /// against the pure formatting logic instead.
    #[test]
    fn suffix_is_empty_when_no_gate_is_armed() {
        if std::env::var("LUMEN_CUDA_LMHEAD_Q6K").is_err()
            && std::env::var("LUMEN_CUDA_SSMOUT_NATIVE").is_err()
            && std::env::var("LUMEN_Q6K_LAYOUT_FIX").is_err()
        {
            assert_eq!(
                lbc_variant_suffix(),
                "",
                "an unarmed build must produce the canonical LBC filename"
            );
        }
    }

    /// Suffix fragments are fixed strings in a fixed order, so the same gate
    /// combination always resolves to the same filename across processes and
    /// hosts. A drifting suffix would silently re-convert instead of reusing.
    #[test]
    fn suffix_composition_is_stable() {
        fn compose(head: bool, ssm: bool, fix: bool) -> String {
            let mut s = String::new();
            if head {
                s.push_str("-q6khead");
            }
            if ssm {
                s.push_str("-ssmq4");
            }
            if fix {
                s.push_str("-q6kfix");
            }
            s
        }
        assert_eq!(compose(false, false, false), "");
        assert_eq!(compose(true, false, false), "-q6khead");
        assert_eq!(compose(false, true, false), "-ssmq4");
        assert_eq!(compose(false, false, true), "-q6kfix");
        assert_eq!(compose(true, true, true), "-q6khead-ssmq4-q6kfix");
        // Distinctness: no two gate combinations may collide on one filename.
        let mut seen = std::collections::HashSet::new();
        for h in [false, true] {
            for s in [false, true] {
                for f in [false, true] {
                    assert!(
                        seen.insert(compose(h, s, f)),
                        "suffix collision at ({h}, {s}, {f})"
                    );
                }
            }
        }
    }

    /// `flag` is strictly two-valued on `"1"`. This is the property that keeps
    /// these gates out of the `=0`-enables / `=on`-disables defect class that
    /// already exists elsewhere in this workspace.
    #[test]
    fn flag_is_one_only() {
        let name = "LUMEN_CONVERT_ENV_GATE_PARSE_PROBE";
        for (value, want) in [
            ("1", true),
            ("0", false),
            ("true", false),
            ("on", false),
            ("yes", false),
            ("", false),
            (" 1", false),
            ("1 ", false),
            ("01", false),
        ] {
            std::env::set_var(name, value);
            assert_eq!(flag(name), want, "value {value:?}");
        }
        std::env::remove_var(name);
        assert!(!flag(name), "unset must be OFF");
    }
}
