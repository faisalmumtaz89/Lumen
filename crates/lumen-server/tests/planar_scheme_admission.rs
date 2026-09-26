//! Scheme admission at `lumen-server` startup.
//!
//! The server refuses an artifact whose weights use a scheme with no serving
//! kernels on its backend, by name, before the weight provider opens, and
//! leaves an artifact whose schemes do serve to start as before. Each case
//! first asks the rule itself (`lumen_format::serving_rules::unservable_scheme`)
//! for its answer on the synthetic artifact, then starts the server binary and
//! checks its exit code and stderr against that answer.
//!
//! Host-only, no GPU: the refusal is decided from what `LbcFile::open`
//! parsed, before any weight provider opens. The binary needs the `bin`
//! feature:
//!
//! ```text
//! cargo test -p lumen-server --features lumen-server/bin --test planar_scheme_admission
//! ```

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use lumen_convert::test_checkpoint::{self, Modules, Tokenizer};
use lumen_format::serving_rules::{
    scheme_has_no_serving_kernels, unservable_scheme, ServingBackend,
};
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
    run_server_with_env(artifact, backend, &[])
}

/// [`run_server`] with extra environment variables set for the server.
fn run_server_with_env(
    artifact: &Path,
    backend: &str,
    env: &[(&str, &str)],
) -> (Option<i32>, String) {
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
        .envs(env.iter().copied())
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

/// A fresh directory for one case's files, removed when the returned guard drops.
fn workdir(tag: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("lumen-admission-server-{tag}-"))
        .tempdir()
        .unwrap()
}

#[test]
fn an_nvfp4_artifact_is_refused_by_name_at_startup() {
    let dir = workdir("nvfp4");
    let artifact = test_checkpoint::write_artifact(dir.path(), Modules::Nvfp4AndFp8)
        .expect("convert the synthetic ModelOpt checkpoint");

    // The rule's own answer for this artifact, before the binary is asked.
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(
        unservable_scheme(&lbc, ServingBackend::Cpu),
        Some(QuantScheme::Nvfp4)
    );

    // The named backend, and, when this build has no CUDA backend, the one the server picks for itself
    // (with CUDA, `auto` selects it and the artifact is served).
    let backends: &[&str] = if cfg!(feature = "cuda") {
        &["cpu"]
    } else {
        &["cpu", "auto"]
    };
    for &backend in backends {
        let (code, stderr) = run_server(&artifact, backend);
        assert_ne!(
            code,
            Some(0),
            "backend {backend}: the server started: {stderr}"
        );
        assert!(
            stderr.contains("Nvfp4") && stderr.contains("is served on CUDA only"),
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

/// `--backend cuda` in a build without CUDA is refused for the build before scheme admission: with
/// `LUMEN_CUDA_NVFP4=0` set, admission would otherwise refuse the artifact by the switch's name.
#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_without_the_cuda_build_is_refused_for_the_build() {
    let dir = workdir("nocuda");
    let artifact = test_checkpoint::write_artifact(dir.path(), Modules::Nvfp4AndFp8)
        .expect("convert the synthetic ModelOpt checkpoint");
    let (code, stderr) = run_server_with_env(&artifact, "cuda", &[("LUMEN_CUDA_NVFP4", "0")]);
    assert_ne!(code, Some(0), "the server started: {stderr}");
    assert!(
        stderr.contains("--backend cuda requires building with --features cuda")
            && !stderr.contains("is served on CUDA only"),
        "the refusal does not name the build:\n{stderr}"
    );
}

/// `LUMEN_CUDA_NVFP4=0` reaches admission: the server publishes the switch
/// before asking the rule, so a CUDA start refuses the artifact by the
/// switch's name. The refusal comes before the CUDA backend is constructed, so
/// no device is needed.
#[cfg(feature = "cuda")]
#[test]
fn the_kill_switch_refuses_an_nvfp4_artifact_on_cuda_at_startup() {
    let dir = workdir("kill-switch");
    let artifact = test_checkpoint::write_artifact(dir.path(), Modules::Nvfp4AndFp8)
        .expect("convert the synthetic ModelOpt checkpoint");
    let (code, stderr) = run_server_with_env(&artifact, "cuda", &[("LUMEN_CUDA_NVFP4", "0")]);
    assert_ne!(code, Some(0), "the server started: {stderr}");
    assert!(
        stderr.contains(
            "this model (Nvfp4) needs the CUDA kernels for this scheme, which LUMEN_CUDA_NVFP4=0 disables"
        ),
        "the refusal does not name the scheme and the switch:\n{stderr}"
    );
    assert!(
        !stderr.contains("listening"),
        "the server got past admission:\n{stderr}"
    );
}

#[test]
fn an_fp8_artifact_is_refused_by_name_at_startup() {
    let dir = workdir("fp8");
    let artifact = test_checkpoint::write_artifact(dir.path(), Modules::Fp8Only)
        .expect("convert the synthetic ModelOpt checkpoint");
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(lbc.header.quantization.scheme, QuantScheme::Fp8E4M3);
    assert_eq!(
        unservable_scheme(&lbc, ServingBackend::Cpu),
        Some(QuantScheme::Fp8E4M3)
    );

    let (code, stderr) = run_server(&artifact, "cpu");
    assert_ne!(code, Some(0), "the server started: {stderr}");
    assert!(
        stderr.contains("Fp8E4M3") && stderr.contains("is served on CUDA only"),
        "the refusal does not name the scheme:\n{stderr}"
    );
}

#[test]
fn a_planar_layer_slice_under_a_servable_primary_is_refused_by_name_at_startup() {
    // The header's own scheme serves, so the refusal rests entirely on the
    // per-slice scan past it.
    let dir = workdir("mixed");
    let artifact = test_checkpoint::write_planar_slice_artifact(
        dir.path(),
        Tokenizer::Embedded,
        QuantScheme::Fp8E4M3,
    );

    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(lbc.header.quantization.scheme, QuantScheme::Q4_0);
    assert!(!scheme_has_no_serving_kernels(
        QuantScheme::Q4_0,
        ServingBackend::Cpu
    ));
    assert_eq!(
        lbc.layer_indices[0].subtensors.w_gate.quant,
        QuantScheme::Fp8E4M3
    );
    assert_eq!(
        unservable_scheme(&lbc, ServingBackend::Cpu),
        Some(QuantScheme::Fp8E4M3)
    );

    let (code, stderr) = run_server(&artifact, "cpu");
    assert_ne!(code, Some(0), "the server started: {stderr}");
    assert!(
        stderr.contains("Fp8E4M3") && stderr.contains("is served on CUDA only"),
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
    let artifact = test_checkpoint::write_planar_slice_artifact(
        dir.path(),
        Tokenizer::Absent,
        QuantScheme::Fp8E4M3,
    );
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert!(lbc.tokenizer.is_none());
    assert_eq!(
        unservable_scheme(&lbc, ServingBackend::Cpu),
        Some(QuantScheme::Fp8E4M3)
    );

    let (code, stderr) = run_server(&artifact, "cpu");
    assert_ne!(code, Some(0), "the server started: {stderr}");
    assert!(
        stderr.contains("Fp8E4M3") && stderr.contains("is served on CUDA only"),
        "the refusal does not name the scheme:\n{stderr}"
    );
    assert!(
        !stderr.contains("no embedded tokenizer"),
        "the missing-tokenizer refusal fired instead of the scheme refusal:\n{stderr}"
    );
}

#[test]
fn an_existing_scheme_still_passes_admission() {
    // The artifact has no tokenizer section, so the step after admission —
    // building the tokenizer — refuses it. Reaching that refusal is what
    // shows admission let the artifact through.
    let dir = workdir("q4");
    let artifact = test_checkpoint::write_q4_0_artifact(dir.path(), Tokenizer::Absent);
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(unservable_scheme(&lbc, ServingBackend::Cpu), None);

    let (code, stderr) = run_server(&artifact, "cpu");
    assert_ne!(
        code,
        Some(0),
        "the server started without a tokenizer: {stderr}"
    );
    assert!(
        stderr.contains("no embedded tokenizer"),
        "the server did not reach the step after admission:\n{stderr}"
    );
    assert!(
        !stderr.contains("is served on CUDA only"),
        "a Q4_0 artifact was refused by the planar rule:\n{stderr}"
    );
}
