//! Scheme admission in `lumen run` and the benchmark runner.
//!
//! An artifact whose weights use a scheme with no serving kernels on the
//! selected backend is refused at admission, by name, and an artifact whose
//! schemes do serve is not touched by that rule. Each case first asks the
//! rule itself (`lumen_format::serving_rules::unservable_scheme`) for its
//! answer on the artifact, then runs the `lumen` binary (or the benchmark
//! runner) and checks its exit code and stderr against that answer. The artifacts are
//! synthetic: a ModelOpt checkpoint converted by `lumen_convert`, or a Q4_0
//! artifact with one layer slice retagged.
//!
//! The refusal happens before a weight provider opens, so every case runs on
//! a machine with no GPU:
//!
//! ```text
//! cargo test -p lumen-cli --test planar_scheme_admission
//! ```

use std::path::Path;
use std::process::Command;

use lumen_convert::test_checkpoint::{self, Modules, Tokenizer};
use lumen_format::serving_rules::{
    scheme_has_no_serving_kernels, unservable_scheme, ServingBackend,
};
use lumen_format::QuantScheme;

/// `lumen run` against an artifact, with the CPU backend so the test needs no
/// GPU. Returns (exit code, stderr).
fn run_cli(artifact: &Path, extra: &[&str]) -> (Option<i32>, String) {
    // `--tokens` rather than `--prompt`: the synthetic artifact's vocabulary
    // is 32 placeholder tokens, no text to tokenize against.
    run_cli_with(artifact, &["--tokens", "1"], extra)
}

/// `lumen run` with the prompt given as text, so the tokenizer is on the
/// path the run takes.
fn run_cli_prompt(artifact: &Path, extra: &[&str]) -> (Option<i32>, String) {
    run_cli_with(artifact, &["--prompt", "hi"], extra)
}

fn run_cli_with(artifact: &Path, input: &[&str], extra: &[&str]) -> (Option<i32>, String) {
    let mut args = vec!["run", "--model", artifact.to_str().unwrap()];
    args.extend_from_slice(input);
    args.extend_from_slice(&["--max-tokens", "1"]);
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

/// A fresh directory for one case's files, removed when the returned guard drops.
fn workdir(tag: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("lumen-admission-cli-{tag}-"))
        .tempdir()
        .unwrap()
}

#[test]
fn an_nvfp4_artifact_is_refused_by_name_on_the_backends_that_cannot_serve_it() {
    let dir = workdir("nvfp4");
    let artifact = test_checkpoint::write_artifact(dir.path(), Modules::Nvfp4AndFp8)
        .expect("convert the synthetic ModelOpt checkpoint");
    // The rule's own answer for this artifact, before the binary is asked.
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(
        unservable_scheme(&lbc, ServingBackend::Cpu),
        Some(QuantScheme::Nvfp4)
    );

    // The selections that run anywhere: the SIMD CPU backend, and, in a build
    // without CUDA, no flag (the mmap provider and the CLI's own choice of
    // backend), `--streaming` (the same provider with the weights left out of
    // GPU residency) and the synchronous/asynchronous providers. A CUDA build serves it with
    // `--cuda`, and with no flag or `--sync` on a machine with an NVIDIA GPU
    // (`--streaming` fails there instead: its GDN layers need GPU-resident
    // weights), so this set is asked only without CUDA. `--metal` is not
    // here because off macOS it is refused before the scheme is read.
    let cpu_only: &[&[&str]] = &[&["--simd"]];
    let own_choice: &[&[&str]] = &[&[], &["--streaming"], &["--sync"], &["--async"]];
    let selections = cpu_only.iter().chain(if cfg!(feature = "cuda") {
        &[][..]
    } else {
        own_choice
    });
    for &extra in selections {
        let (code, stderr) = run_cli(&artifact, extra);
        assert_eq!(code, Some(1), "backend {extra:?}: exit code\n{stderr}");
        assert!(
            stderr.contains("Nvfp4") && stderr.contains("is served on CUDA only"),
            "backend {extra:?}: refusal does not name the scheme:\n{stderr}"
        );
        // It stopped AT admission: nothing was printed after the refusal.
        assert!(
            stderr
                .lines()
                .last()
                .is_some_and(|line| line.contains("is served on CUDA only")),
            "backend {extra:?}: the run went past admission:\n{stderr}"
        );
    }
}

/// `--cuda` in a build without CUDA is refused for the build before scheme admission: with
/// `LUMEN_CUDA_NVFP4=0` set, admission would otherwise refuse the artifact by the switch's name.
#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_without_the_cuda_build_is_refused_for_the_build() {
    let dir = workdir("nocuda");
    let artifact = test_checkpoint::write_artifact(dir.path(), Modules::Nvfp4AndFp8)
        .expect("convert the synthetic ModelOpt checkpoint");
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args([
            "run",
            "--model",
            artifact.to_str().unwrap(),
            "--tokens",
            "1",
            "--max-tokens",
            "1",
            "--cuda",
        ])
        .env("LUMEN_CUDA_NVFP4", "0")
        .output()
        .expect("spawn lumen");
    let (code, stderr) = (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    assert_eq!(code, Some(1), "exit code\n{stderr}");
    assert!(
        stderr.contains("--cuda requires building with --features cuda")
            && !stderr.contains("is served on CUDA only"),
        "the refusal does not name the build:\n{stderr}"
    );
}

/// `LUMEN_CUDA_NVFP4=0` reaches admission: the CLI publishes the switch before
/// asking the rule, so a CUDA run refuses the artifact by the switch's name.
/// The refusal comes before the CUDA backend is constructed, but `--cuda`
/// first checks the device ordinal against the driver, so this needs one:
///
/// ```text
/// cargo test --release -p lumen-cli --features cuda --test planar_scheme_admission -- --ignored
/// ```
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA device; run with --ignored"]
fn the_kill_switch_refuses_an_nvfp4_artifact_on_cuda_before_the_backend_is_built() {
    let dir = workdir("kill-switch");
    let artifact = test_checkpoint::write_artifact(dir.path(), Modules::Nvfp4AndFp8)
        .expect("convert the synthetic ModelOpt checkpoint");
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args([
            "run",
            "--model",
            artifact.to_str().unwrap(),
            "--tokens",
            "1",
            "--max-tokens",
            "1",
            "--cuda",
        ])
        .env("LUMEN_CUDA_NVFP4", "0")
        .output()
        .expect("spawn lumen");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "exit code\n{stderr}");
    // The refusal is all the run printed, apart from the binary's warnings
    // about the test harness's environment: nothing past admission ran.
    let printed: Vec<&str> = stderr
        .lines()
        .filter(|line| !line.starts_with("[lumen] WARNING: "))
        .collect();
    assert_eq!(
        printed,
        ["Error: this model (Nvfp4) needs the CUDA kernels for this scheme, which LUMEN_CUDA_NVFP4=0 disables"],
        "the run was not refused at admission by the switch:\n{stderr}"
    );
}

#[test]
fn an_fp8_artifact_is_refused_by_name() {
    let dir = workdir("fp8");
    let artifact = test_checkpoint::write_artifact(dir.path(), Modules::Fp8Only)
        .expect("convert the synthetic ModelOpt checkpoint");
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(lbc.header.quantization.scheme, QuantScheme::Fp8E4M3);
    assert_eq!(
        unservable_scheme(&lbc, ServingBackend::Cpu),
        Some(QuantScheme::Fp8E4M3)
    );

    let (code, stderr) = run_cli(&artifact, &["--simd"]);
    assert_eq!(code, Some(1), "exit code\n{stderr}");
    assert!(
        stderr.contains("Fp8E4M3") && stderr.contains("is served on CUDA only"),
        "the refusal does not name the scheme:\n{stderr}"
    );
}

#[test]
fn a_planar_layer_slice_under_a_servable_primary_is_refused_by_name() {
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

    let (code, stderr) = run_cli(&artifact, &["--simd"]);
    assert_eq!(code, Some(1), "exit code\n{stderr}");
    assert!(
        stderr.contains("Fp8E4M3") && stderr.contains("is served on CUDA only"),
        "the refusal does not name the scheme:\n{stderr}"
    );
}

#[test]
fn a_ct_int4_slice_is_turned_away_by_the_check_after_admission() {
    // CtInt4G32 has CUDA kernels, so admission passes it and the check after
    // it decides: on a build without the cuda feature that check names the
    // feature; with it, the backend the run was not given.
    let dir = workdir("ct-int4");
    let artifact = test_checkpoint::write_planar_slice_artifact(
        dir.path(),
        Tokenizer::Absent,
        QuantScheme::CtInt4G32,
    );
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(unservable_scheme(&lbc, ServingBackend::Cpu), None);
    assert!(lbc.uses_quant(QuantScheme::CtInt4G32));

    let (code, stderr) = run_cli(&artifact, &["--simd"]);
    assert_eq!(code, Some(1), "exit code\n{stderr}");
    assert!(
        !stderr.contains("is served on CUDA only"),
        "admission refused a scheme that has kernels:\n{stderr}"
    );
    let expected = if cfg!(feature = "cuda") {
        "this model (CtInt4G32) requires the CUDA backend (--cuda)"
    } else {
        "this model (CtInt4G32) requires a lumen build with the `cuda` feature"
    };
    assert!(
        stderr.contains(expected),
        "the CtInt4G32 check did not fire, or not with its message:\n{stderr}"
    );
}

#[test]
fn a_text_prompt_reaches_the_embedded_tokenizer_past_admission() {
    // A servable artifact with a tokenizer section: admission passes it and
    // the run builds the tokenizer from the section the open parsed. The
    // placeholder vocabulary has no merges, so tokenizing the prompt yields
    // nothing, and that is where the run stops.
    let dir = workdir("prompt");
    let artifact = test_checkpoint::write_q4_0_artifact(dir.path(), Tokenizer::Embedded);
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(unservable_scheme(&lbc, ServingBackend::Cpu), None);
    assert_eq!(lbc.tokenizer.as_ref().map(|t| t.tokens.len()), Some(32));

    let (code, stderr) = run_cli_prompt(&artifact, &["--simd"]);
    assert!(
        !stderr.contains("is served on CUDA only") && !stderr.contains("no embedded tokenizer"),
        "the run did not get past admission and the tokenizer:\n{stderr}"
    );
    assert_eq!(code, Some(1), "exit code\n{stderr}");
    // The whole of what the run printed, apart from the warnings the binary
    // gives about the test harness's own environment variables: nothing
    // before the tokenizer's empty answer, nothing after it.
    let printed: Vec<&str> = stderr
        .lines()
        .filter(|line| !line.starts_with("[lumen] WARNING: "))
        .collect();
    assert_eq!(
        printed,
        ["Error: prompt produced no tokens after tokenization"],
        "the run did not stop where the placeholder tokenizer leaves it:\n{stderr}"
    );
}

#[test]
fn the_benchmark_runner_refuses_an_unservable_artifact() {
    let dir = workdir("bench");
    let artifact = test_checkpoint::write_artifact(dir.path(), Modules::Nvfp4AndFp8)
        .expect("convert the synthetic ModelOpt checkpoint");
    let err = lumen_bench::runner::ensure_model(&lumen_bench::config::ModelSpec::Path(artifact))
        .expect_err("the runner resolved an artifact with no serving kernels")
        .to_string();
    assert!(
        err.contains("Nvfp4") && err.contains("is served on CUDA only"),
        "the refusal does not name the scheme: {err}"
    );

    // The control: an artifact whose scheme serves still resolves.
    let q4 = test_checkpoint::write_q4_0_artifact(dir.path(), Tokenizer::Absent);
    assert_eq!(
        lumen_bench::runner::ensure_model(&lumen_bench::config::ModelSpec::Path(q4.clone()))
            .unwrap(),
        q4
    );
}

#[test]
fn an_existing_scheme_still_passes_admission() {
    // The control for the rule above: a Q4_0 artifact must reach the same
    // code path and NOT be refused by it. What the run does after admission
    // is another test's subject.
    let dir = workdir("q4");
    let artifact = test_checkpoint::write_q4_0_artifact(dir.path(), Tokenizer::Absent);
    let (_, stderr) = run_cli(&artifact, &["--simd"]);
    assert!(
        !stderr.contains("is served on CUDA only"),
        "a Q4_0 artifact was refused by the planar rule:\n{stderr}"
    );
    // And the rule itself says so, on the same artifact the CLI just ran:
    // the scan finds nothing unservable, so admission passes it through.
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(unservable_scheme(&lbc, ServingBackend::Cpu), None);
}
