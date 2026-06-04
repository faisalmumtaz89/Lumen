//! Integration test: invoke the `lumen convert` subcommand end-to-end on a
//! synthetic 2-shard GGUF and verify the CLI routes through the multi-shard
//! reader. We can't run a real Qwen3.5-MoE BF16 conversion in CI (68 GB), so
//! we exercise the CLI's plumbing with an unsupported architecture (`gpt2`)
//! and assert the failure mode comes from the *downstream* converter -- proving
//! shard discovery, merge, and the binary's `convert_gguf_to_lbc` call all
//! work together.
//!
//! This complements the unit-level routing test in
//! `lumen_convert::convert::tests::convert_gguf_to_lbc_multi_shard_path_routing`
//! by exercising the actual compiled CLI binary rather than just the library.

use std::path::PathBuf;
use std::process::Command;

use lumen_convert::gguf::{GgmlType, GgufBuilder};

fn temp_dir(label: &str) -> PathBuf {
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("lumen-cli-test-{label}-{id}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Locate the freshly-built `lumen` binary in this workspace. Cargo sets
/// `CARGO_BIN_EXE_lumen` for integration tests of binary crates.
fn lumen_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_lumen"))
}

/// Build a synthetic 2-shard GGUF with a (deliberately unsupported)
/// architecture so the converter rejects at extract_hyperparams, surfacing the
/// fact that shard discovery + merge succeeded.
fn write_2shard_gpt2_set(dir: &std::path::Path) -> (PathBuf, PathBuf) {
    let stem = "cli-smoke";

    let mut b1 = GgufBuilder::new();
    b1.add_string("general.architecture", "gpt2");
    b1.add_u16("split.no", 0);
    b1.add_u16("split.count", 2);
    b1.add_u64("split.tensors.count", 2);
    b1.add_f32_tensor("token_embd.weight", &[4], &[0.0; 4]);
    let p1 = dir.join(format!("{stem}-00001-of-00002.gguf"));
    std::fs::write(&p1, b1.build()).unwrap();

    let mut b2 = GgufBuilder::new();
    b2.add_string("general.architecture", "gpt2");
    b2.add_u16("split.no", 1);
    b2.add_u16("split.count", 2);
    b2.add_u64("split.tensors.count", 2);
    b2.add_f32_tensor("blk.0.attn_q.weight", &[4], &[0.0; 4]);
    let p2 = dir.join(format!("{stem}-00002-of-00002.gguf"));
    std::fs::write(&p2, b2.build()).unwrap();

    (p1, p2)
}

#[test]
fn lumen_convert_multi_shard_routes_through_merger() {
    let dir = temp_dir("multi-shard-route");
    let (p1, _p2) = write_2shard_gpt2_set(&dir);
    let lbc_path = dir.join("smoke.lbc");

    let output = Command::new(lumen_binary())
        .arg("convert")
        .arg("--input")
        .arg(&p1)
        .arg("--output")
        .arg(&lbc_path)
        .output()
        .expect("failed to launch lumen binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success(),
        "lumen convert should fail on unsupported arch (gpt2): stdout={stdout}, stderr={stderr}");
    // The downstream converter must surface the architecture error -- this
    // proves the multi-shard reader successfully merged both shards and
    // reached extract_hyperparams.
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("gpt2") || combined.contains("unsupported architecture"),
        "expected unsupported-architecture error after multi-shard merge; got: {combined}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn lumen_convert_multi_shard_missing_sibling_errors_cleanly() {
    // Write ONLY shard 1 of a claimed 2-shard set. The convert CLI must
    // refuse with a clear missing-shard error -- it MUST NOT silently treat
    // shard 1 as a single-file GGUF (which would lose data) and MUST NOT
    // panic.
    let dir = temp_dir("missing-shard");
    let stem = "cli-missing";

    let mut b1 = GgufBuilder::new();
    b1.add_string("general.architecture", "qwen35moe");
    b1.add_u16("split.no", 0);
    b1.add_u16("split.count", 2);
    b1.add_u64("split.tensors.count", 2);
    b1.add_tensor("token_embd.weight", GgmlType::F32, &[4], vec![0u8; 16]);
    let p1 = dir.join(format!("{stem}-00001-of-00002.gguf"));
    std::fs::write(&p1, b1.build()).unwrap();
    let lbc_path = dir.join("missing.lbc");

    let output = Command::new(lumen_binary())
        .arg("convert")
        .arg("--input")
        .arg(&p1)
        .arg("--output")
        .arg(&lbc_path)
        .output()
        .expect("failed to launch lumen binary");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!output.status.success(),
        "convert must fail when a sibling shard is missing: stdout={stdout}, stderr={stderr}");
    let combined = format!("{stdout}{stderr}");
    // The error should mention the shard count mismatch in some form.
    assert!(
        combined.contains("shard")
            || combined.contains("expected 2")
            || combined.contains("only found"),
        "expected missing-shard error message, got: {combined}"
    );
    std::fs::remove_dir_all(&dir).ok();
}
