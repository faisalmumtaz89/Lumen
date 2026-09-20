//! An artifact whose weights are a scheme with no serving kernels is refused
//! at admission, by name, whatever backend was asked for — and an artifact
//! whose schemes do serve is not touched by that rule.
//!
//! The refusal is the point where the CLI stops: it happens before a weight
//! provider opens, so it holds on a machine with no GPU at all.

use std::path::{Path, PathBuf};
use std::process::Command;

use lumen_convert::test_checkpoint::{self, Modules};
use lumen_format::serving_rules::{scheme_has_no_serving_kernels, unservable_scheme};
use lumen_format::QuantScheme;

/// `lumen run` against an artifact, with the CPU backend so the test needs no
/// GPU. Returns (exit code, stderr).
fn run_cli(artifact: &Path, extra: &[&str]) -> (Option<i32>, String) {
    // `--tokens` rather than `--prompt`: the synthetic artifact's vocabulary
    // is 32 placeholder tokens, and the tokenizer runs before admission.
    let mut args = vec![
        "run",
        "--model",
        artifact.to_str().unwrap(),
        "--tokens",
        "1",
        "--max-tokens",
        "1",
    ];
    args.extend_from_slice(extra);
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args(&args)
        .output()
        .expect("spawn lumen");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn workdir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("lumen-admission-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn an_nvfp4_artifact_is_refused_by_name_on_every_backend() {
    let dir = workdir("nvfp4");
    let artifact = test_checkpoint::write_artifact(&dir, Modules::Nvfp4AndFp8)
        .expect("convert the synthetic ModelOpt checkpoint");
    // The rule's own answer for this artifact, before the binary is asked.
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(unservable_scheme(&lbc), Some(QuantScheme::Nvfp4));

    // Every backend selection the CLI offers on this host, plus the default.
    for extra in [&[][..], &["--simd"][..], &["--sync"][..], &["--async"][..]] {
        let (code, stderr) = run_cli(&artifact, extra);
        assert_eq!(code, Some(1), "backend {extra:?}: exit code\n{stderr}");
        assert!(
            stderr.contains("Nvfp4") && stderr.contains("no serving kernels for this scheme yet"),
            "backend {extra:?}: refusal does not name the scheme:\n{stderr}"
        );
        // It stopped AT admission: no weight provider opened, so nothing
        // downstream of it ran.
        assert!(
            !stderr.contains("Loading") && !stderr.contains("simd_kernels"),
            "backend {extra:?}: the run went past admission:\n{stderr}"
        );
    }
}

#[test]
fn an_fp8_artifact_is_refused_by_name() {
    let dir = workdir("fp8");
    let artifact = test_checkpoint::write_artifact(&dir, Modules::Fp8Only)
        .expect("convert the synthetic ModelOpt checkpoint");
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(lbc.header.quantization.scheme, QuantScheme::Fp8E4M3);
    assert_eq!(unservable_scheme(&lbc), Some(QuantScheme::Fp8E4M3));

    let (code, stderr) = run_cli(&artifact, &["--simd"]);
    assert_eq!(code, Some(1), "exit code\n{stderr}");
    assert!(
        stderr.contains("Fp8E4M3") && stderr.contains("no serving kernels for this scheme yet"),
        "the refusal does not name the scheme:\n{stderr}"
    );
}

#[test]
fn one_unservable_layer_slice_is_refused_by_name() {
    // The header's own scheme serves, so only the per-slice scan can find
    // the one projection that does not.
    let dir = workdir("mixed");
    let artifact = test_checkpoint::write_artifact(&dir, Modules::Int4WithOneFp8)
        .expect("convert the synthetic compressed-tensors checkpoint");
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(lbc.header.quantization.scheme, QuantScheme::CtInt4G32);
    assert!(!scheme_has_no_serving_kernels(QuantScheme::CtInt4G32));
    assert_eq!(unservable_scheme(&lbc), Some(QuantScheme::Fp8E4M3));

    let (code, stderr) = run_cli(&artifact, &["--simd"]);
    assert_eq!(code, Some(1), "exit code\n{stderr}");
    assert!(
        stderr.contains("Fp8E4M3") && stderr.contains("no serving kernels for this scheme yet"),
        "the refusal does not name the scheme:\n{stderr}"
    );
}

#[test]
fn an_existing_scheme_still_passes_admission() {
    // The control for the rule above: a Q4_0 artifact must reach the same
    // code path and NOT be refused by it. Whatever happens afterwards is the
    // behaviour this PR leaves alone.
    let dir = workdir("q4");
    let artifact = test_checkpoint::write_q4_0_artifact(&dir);
    let (_, stderr) = run_cli(&artifact, &["--simd"]);
    assert!(
        !stderr.contains("no serving kernels for this scheme yet"),
        "a Q4_0 artifact was refused by the planar rule:\n{stderr}"
    );
    // And the rule itself says so, on the same artifact the CLI just ran:
    // the scan finds nothing unservable, so admission passes it through.
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(unservable_scheme(&lbc), None);
}
