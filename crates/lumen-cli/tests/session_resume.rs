//! `lumen run --session-resume` with a prompt of another length than the
//! saved one: the cache is sized from the prompt and the answer, so the
//! resumed run must take the saved session's capacity for the loader to
//! accept it. Runs on the CPU with a synthetic model:
//!
//! ```text
//! cargo test -p lumen-cli --test session_resume
//! ```

use std::path::Path;
use std::process::Command;

use lumen_format::test_model::{generate_test_model, TestModelConfig};

fn run(model: &Path, session: &Path, tokens: &str, flag: &str) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args(["run", "--model", model.to_str().unwrap(), "--simd"])
        .args(["--tokens", tokens, "--max-tokens", "300"])
        .args([flag, session.to_str().unwrap()])
        .output()
        .expect("spawn lumen");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn a_session_resumes_with_a_longer_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.lbc");
    // A window past either run's prompt + answer + 256 (559 and 561, above the
    // 512 floor), so the two runs would size their caches differently.
    std::fs::write(
        &model,
        generate_test_model(&TestModelConfig {
            max_seq_len: 1024,
            ..Default::default()
        }),
    )
    .unwrap();
    let session = dir.path().join("session.kv");

    let (code, err) = run(&model, &session, "1 2 3", "--session-save");
    assert_eq!(code, Some(0), "{err}");
    let (code, err) = run(&model, &session, "1 2 3 4 5", "--session-resume");
    assert_eq!(code, Some(0), "{err}");
}
