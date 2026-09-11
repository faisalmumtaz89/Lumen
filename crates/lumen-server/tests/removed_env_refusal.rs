//! The server binary refuses to start when an environment name an earlier
//! release read, and this one does not, is set: exit code 2 with the name and
//! the remedy, before any model is opened. Host-only. The binary needs the
//! `bin` feature; without it the test has nothing to run and says so.

use std::process::Command;

#[test]
fn a_removed_env_name_refuses_server_startup_with_its_remedy() {
    let Some(bin) = option_env!("CARGO_BIN_EXE_lumen-server") else {
        eprintln!("Skipping: lumen-server binary not built (enable the `bin` feature)");
        return;
    };
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
