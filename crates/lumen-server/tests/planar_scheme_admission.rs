//! The server refuses an artifact whose weights are a scheme with no serving
//! kernels, by name, before the weight provider opens — and starts its
//! admission unchanged for an artifact whose schemes do serve.
//!
//! Host-only, no GPU: the refusal is decided from what `LbcFile::open`
//! parsed, before any weight provider opens.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use lumen_convert::test_checkpoint::{self, Modules, Tokenizer};
use lumen_format::serving_rules::{scheme_has_no_serving_kernels, unservable_scheme};
use lumen_format::QuantScheme;

/// How long the server is given to refuse and exit. A refusal is decided
/// before any weight provider opens, so it is immediate; this bound only fires
/// when the server does not refuse at all, in which case the test fails here
/// instead of leaving a started server running until CI's own limit.
const REFUSAL_TIMEOUT: Duration = Duration::from_secs(60);

/// Start the server on `artifact` and return its stderr. `--port 0` keeps a
/// successful start from binding a fixed port. The binary needs the `bin`
/// feature, and Cargo names its path whether or not the feature built it, so
/// the feature is checked rather than the path trusted: without it the test
/// fails instead of passing against a stale or absent binary.
///
/// The wait is bounded: a server still running at `REFUSAL_TIMEOUT` is killed
/// and this panics with its stderr, so a server that fails to refuse is a red
/// test rather than a hang. A refusal writes a few hundred bytes to stderr and
/// exits well inside the bound.
fn run_server(artifact: &Path, backend: &str) -> (Option<i32>, String) {
    if !cfg!(feature = "bin") {
        panic!("this test spawns the server binary: run with --features lumen-server/bin");
    }
    let mut child = Command::new(env!("CARGO_BIN_EXE_lumen-server"))
        .args([
            "--model",
            artifact.to_str().unwrap(),
            "--backend",
            backend,
            "--port",
            "0",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn lumen-server");

    let deadline = Instant::now() + REFUSAL_TIMEOUT;
    loop {
        if child.try_wait().expect("wait for lumen-server").is_some() {
            break;
        }
        if Instant::now() >= deadline {
            child.kill().expect("kill lumen-server");
            let status = child.wait().expect("reap lumen-server");
            let mut pipe = child.stderr.take().expect("capture server stderr");
            let mut stderr = Vec::new();
            pipe.read_to_end(&mut stderr).expect("read server stderr");
            panic!(
                "the server did not refuse (backend {backend}); it was killed after {}s with \
                 status {status:?}:\n{}",
                REFUSAL_TIMEOUT.as_secs(),
                String::from_utf8_lossy(&stderr)
            );
        }
        thread::sleep(Duration::from_millis(20));
    }

    let out = child
        .wait_with_output()
        .expect("collect lumen-server output");
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
    // The header's own scheme serves, so the refusal rests entirely on the
    // per-slice scan past it.
    let dir = workdir("mixed");
    let artifact = test_checkpoint::write_planar_slice_artifact(
        &dir,
        Tokenizer::Embedded,
        QuantScheme::Fp8E4M3,
    );

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
fn an_unservable_artifact_with_no_tokenizer_is_refused_by_scheme_at_startup() {
    // The refusal is decided from what `LbcFile::open` parsed — the header,
    // the index and the tokenizer section — before the tokenizer is built.
    // An artifact with no section parses with none, so it is still refused
    // for its scheme, not for the tokenizer it lacks.
    let dir = workdir("no-tokenizer");
    let artifact =
        test_checkpoint::write_planar_slice_artifact(&dir, Tokenizer::Absent, QuantScheme::Fp8E4M3);
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert!(lbc.tokenizer.is_none());
    assert_eq!(unservable_scheme(&lbc), Some(QuantScheme::Fp8E4M3));

    let (code, stderr) = run_server(&artifact, "cpu");
    assert_ne!(code, Some(0), "the server started: {stderr}");
    assert!(
        stderr.contains("Fp8E4M3") && stderr.contains("no serving kernels for this scheme yet"),
        "the refusal does not name the scheme:\n{stderr}"
    );
    assert!(
        !stderr.contains("no embedded tokenizer"),
        "the missing-tokenizer refusal fired instead of the scheme refusal:\n{stderr}"
    );
}

#[test]
fn an_existing_scheme_still_passes_admission() {
    let dir = workdir("q4");
    let artifact = test_checkpoint::write_q4_0_artifact(&dir, Tokenizer::Absent);
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(unservable_scheme(&lbc), None);

    let (_, stderr) = run_server(&artifact, "cpu");
    assert!(
        !stderr.contains("no serving kernels for this scheme yet"),
        "a Q4_0 artifact was refused by the planar rule:\n{stderr}"
    );
}
