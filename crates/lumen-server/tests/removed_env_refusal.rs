//! The server binary refuses to start when an environment name an earlier
//! release read, and this one does not, is set: exit code 2 with the name and
//! the remedy, before any model is opened. Host-only. The binary needs the
//! `bin` feature (the CI cpu suite enables it): Cargo names the binary's path
//! whether or not the feature built it, so the test asserts the feature
//! rather than trusting a path — without the feature it FAILS, it never
//! passes against a stale or absent binary, and the coverage cannot be
//! retired by a green run.

use std::process::Command;

#[test]
fn a_removed_env_name_refuses_server_startup_with_its_remedy() {
    assert!(
        cfg!(feature = "bin"),
        "this test spawns the server binary: run with --features lumen-server/bin"
    );
    let bin = env!("CARGO_BIN_EXE_lumen-server");
    for (name, remedy) in lumen_runtime::runtime_defaults::REMOVED_LUMEN_ENV_VARS {
        let out = Command::new(bin)
            .arg("--version")
            .env(name, "1")
            .output()
            .expect("run lumen-server");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{name}: stderr: {stderr}");
        assert!(
            stderr.contains(&format!(
                "{name} is set but this release does not read it: {remedy}"
            )),
            "{name}: the remedy is missing: {stderr}"
        );
    }
}
