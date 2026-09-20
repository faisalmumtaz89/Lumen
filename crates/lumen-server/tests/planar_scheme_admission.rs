//! The server refuses an artifact whose weights are a scheme with no serving
//! kernels, by name, before the weight provider opens — and starts its
//! admission unchanged for an artifact whose schemes do serve.
//!
//! Host-only, no GPU: the refusal is decided from the header and index.

use std::path::Path;
use std::process::Command;

use lumen_convert::test_checkpoint::{self, Modules};
use lumen_format::serving_rules::{scheme_has_no_serving_kernels, unservable_scheme};
use lumen_format::QuantScheme;

/// Start the server on `artifact` and return its stderr. `--port 0` keeps a
/// successful start from binding a fixed port.
fn run_server(artifact: &Path, backend: &str) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_lumen-server"))
        .args([
            "--model",
            artifact.to_str().unwrap(),
            "--backend",
            backend,
            "--port",
            "0",
        ])
        .output()
        .expect("run lumen-server");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn workdir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lumen-admission-server-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn an_nvfp4_artifact_is_refused_by_name_at_startup() {
    assert!(
        cfg!(feature = "bin"),
        "this test spawns the server binary: run with --features lumen-server/bin"
    );
    let dir = workdir("nvfp4");
    let artifact = test_checkpoint::write_artifact(&dir, Modules::Nvfp4AndFp8)
        .expect("convert the synthetic ModelOpt checkpoint");

    // The rule's own answer for this artifact, before the binary is asked.
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(unservable_scheme(&lbc), Some(QuantScheme::Nvfp4));

    // The named backend and the one the server picks for itself.
    for backend in ["cpu", "auto"] {
        let (code, stderr) = run_server(&artifact, backend);
        assert_ne!(
            code,
            Some(0),
            "backend {backend}: the server started: {stderr}"
        );
        assert!(
            stderr.contains("Nvfp4") && stderr.contains("no serving kernels for this scheme yet"),
            "backend {backend}: the refusal does not name the scheme:\n{stderr}"
        );
        // It stopped at admission: the provider never opened, so the server
        // never bound a port.
        assert!(
            !stderr.contains("listening"),
            "backend {backend}: the server got past admission:\n{stderr}"
        );
    }
}

#[test]
fn an_fp8_artifact_is_refused_by_name_at_startup() {
    assert!(
        cfg!(feature = "bin"),
        "this test spawns the server binary: run with --features lumen-server/bin"
    );
    let dir = workdir("fp8");
    let artifact = test_checkpoint::write_artifact(&dir, Modules::Fp8Only)
        .expect("convert the synthetic ModelOpt checkpoint");
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(lbc.header.quantization.scheme, QuantScheme::Fp8E4M3);
    assert_eq!(unservable_scheme(&lbc), Some(QuantScheme::Fp8E4M3));

    let (code, stderr) = run_server(&artifact, "cpu");
    assert_ne!(code, Some(0), "the server started: {stderr}");
    assert!(
        stderr.contains("Fp8E4M3") && stderr.contains("no serving kernels for this scheme yet"),
        "the refusal does not name the scheme:\n{stderr}"
    );
}

#[test]
fn a_planar_layer_slice_under_a_servable_primary_is_refused_by_name_at_startup() {
    assert!(
        cfg!(feature = "bin"),
        "this test spawns the server binary: run with --features lumen-server/bin"
    );
    // The header's own scheme serves, so the refusal rests entirely on the
    // per-slice scan past it.
    let dir = workdir("mixed");
    let artifact = test_checkpoint::write_planar_slice_artifact(&dir);

    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(lbc.header.quantization.scheme, QuantScheme::Q4_0);
    assert!(!scheme_has_no_serving_kernels(QuantScheme::Q4_0));
    assert_eq!(
        lbc.layer_indices[0].subtensors.w_gate.quant,
        QuantScheme::Fp8E4M3
    );
    assert_eq!(unservable_scheme(&lbc), Some(QuantScheme::Fp8E4M3));

    let (code, stderr) = run_server(&artifact, "cpu");
    assert_ne!(code, Some(0), "the server started: {stderr}");
    assert!(
        stderr.contains("Fp8E4M3") && stderr.contains("no serving kernels for this scheme yet"),
        "the refusal does not name the scheme:\n{stderr}"
    );
    assert!(
        !stderr.contains("listening"),
        "the server got past admission:\n{stderr}"
    );
}

#[test]
fn an_existing_scheme_still_passes_admission() {
    assert!(
        cfg!(feature = "bin"),
        "this test spawns the server binary: run with --features lumen-server/bin"
    );
    let dir = workdir("q4");
    let artifact = test_checkpoint::write_q4_0_artifact(&dir);
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(unservable_scheme(&lbc), None);

    let (_, stderr) = run_server(&artifact, "cpu");
    assert!(
        !stderr.contains("no serving kernels for this scheme yet"),
        "a Q4_0 artifact was refused by the planar rule:\n{stderr}"
    );
}
