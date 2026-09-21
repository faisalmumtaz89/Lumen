//! An artifact whose weights are a scheme with no serving kernels is refused
//! at admission, by name, whatever backend was asked for — and an artifact
//! whose schemes do serve is not touched by that rule.
//!
//! The refusal is the point where the CLI stops: it happens before a weight
//! provider opens, so it holds on a machine with no GPU at all. The benchmark
//! runner, the other entry point that opens an artifact from here, refuses on
//! the same rule.

use std::path::{Path, PathBuf};
use std::process::Command;

use lumen_convert::test_checkpoint::{self, Modules, Tokenizer};
use lumen_format::serving_rules::{scheme_has_no_serving_kernels, unservable_scheme};
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

    // The selections that run anywhere: no flag (the mmap provider, and
    // the CLI's own choice of backend), `--streaming` (the same provider
    // with the weights left out of GPU residency), the SIMD CPU backend and
    // the two synchronous/asynchronous providers. `--metal` and `--cuda`
    // are not here because each needs its device present; the refusal
    // happens before any backend is constructed, which is what the no-flag
    // case shows.
    for extra in [
        &[][..],
        &["--streaming"][..],
        &["--simd"][..],
        &["--sync"][..],
        &["--async"][..],
    ] {
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
fn a_planar_layer_slice_under_a_servable_primary_is_refused_by_name() {
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

    let (code, stderr) = run_cli(&artifact, &["--simd"]);
    assert_eq!(code, Some(1), "exit code\n{stderr}");
    assert!(
        stderr.contains("Fp8E4M3") && stderr.contains("no serving kernels for this scheme yet"),
        "the refusal does not name the scheme:\n{stderr}"
    );
}

#[test]
fn an_unservable_artifact_with_no_tokenizer_is_refused_by_scheme_on_a_prompt() {
    // The refusal is decided from what `LbcFile::open` parsed — the header,
    // the index and the tokenizer section — before the tokenizer is built.
    // An artifact with no section parses with none, so a text prompt still
    // sees it refused for its scheme, not for the tokenizer it lacks.
    let dir = workdir("no-tokenizer");
    let artifact =
        test_checkpoint::write_planar_slice_artifact(&dir, Tokenizer::Absent, QuantScheme::Fp8E4M3);
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert!(lbc.tokenizer.is_none());
    assert_eq!(unservable_scheme(&lbc), Some(QuantScheme::Fp8E4M3));

    let (code, stderr) = run_cli_prompt(&artifact, &["--simd"]);
    assert_eq!(code, Some(1), "exit code\n{stderr}");
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
fn a_ct_int4_slice_is_turned_away_by_the_gate_after_admission() {
    // CtInt4G32 has CUDA kernels, so admission passes it and the gate after
    // it decides: on a build without the cuda feature that gate names the
    // feature; with it, the backend the run was not given.
    let dir = workdir("ct-int4");
    let artifact = test_checkpoint::write_planar_slice_artifact(
        &dir,
        Tokenizer::Absent,
        QuantScheme::CtInt4G32,
    );
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(unservable_scheme(&lbc), None);
    assert!(lbc.uses_quant(QuantScheme::CtInt4G32));

    let (code, stderr) = run_cli(&artifact, &["--simd"]);
    assert_eq!(code, Some(1), "exit code\n{stderr}");
    assert!(
        !stderr.contains("no serving kernels for this scheme yet"),
        "admission refused a scheme that has kernels:\n{stderr}"
    );
    let expected = if cfg!(feature = "cuda") {
        "this model (CtInt4G32) requires the CUDA backend (--cuda)"
    } else {
        "this model (CtInt4G32) requires a lumen build with the `cuda` feature"
    };
    assert!(
        stderr.contains(expected),
        "the CtInt4G32 gate did not fire, or not with its message:\n{stderr}"
    );
}

#[test]
fn a_text_prompt_reaches_the_embedded_tokenizer_past_admission() {
    // A servable artifact with a tokenizer section: admission passes it and
    // the run builds the tokenizer from the section the open parsed. The
    // placeholder vocabulary has no merges, so tokenizing the prompt yields
    // nothing, and that is where the run stops.
    let dir = workdir("prompt");
    let artifact = test_checkpoint::write_q4_0_artifact(&dir, Tokenizer::Embedded);
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(unservable_scheme(&lbc), None);
    assert_eq!(lbc.tokenizer.as_ref().map(|t| t.tokens.len()), Some(32));

    let (code, stderr) = run_cli_prompt(&artifact, &["--simd"]);
    assert!(
        !stderr.contains("no serving kernels for this scheme yet")
            && !stderr.contains("no embedded tokenizer"),
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
    let artifact = test_checkpoint::write_artifact(&dir, Modules::Nvfp4AndFp8)
        .expect("convert the synthetic ModelOpt checkpoint");
    let err = lumen_bench::runner::ensure_model(&lumen_bench::config::ModelSpec::Path(artifact))
        .expect_err("the runner resolved an artifact with no serving kernels")
        .to_string();
    assert!(
        err.contains("Nvfp4") && err.contains("no serving kernels for this scheme yet"),
        "the refusal does not name the scheme: {err}"
    );

    // The control: an artifact whose scheme serves still resolves.
    let q4 = test_checkpoint::write_q4_0_artifact(&dir, Tokenizer::Absent);
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
    let artifact = test_checkpoint::write_q4_0_artifact(&dir, Tokenizer::Absent);
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
