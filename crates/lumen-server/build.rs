//! Stamp the git version into the binary so `--version` identifies the exact
//! build. Sets LUMEN_BUILD_VERSION (consumed via option_env! in the version
//! handler, with a CARGO_PKG_VERSION fallback when this script didn't run).
use std::process::Command;

fn main() {
    let git = |args: &[&str]| -> Option<String> {
        let out = Command::new("git").args(args).output().ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    };

    // `git describe` gives "v1.0.0", "v1.0.0-5-gabcdef123456", or (with --always)
    // a bare short SHA, plus "-dirty" for uncommitted trees. Fall back to the crate
    // version when git is unavailable (e.g. a source tarball without .git).
    let version = git(&["describe", "--tags", "--always", "--dirty"])
        .or_else(|| git(&["rev-parse", "--short=12", "HEAD"]))
        .unwrap_or_else(|| std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".into()));

    println!("cargo:rustc-env=LUMEN_BUILD_VERSION={version}");

    // Re-run when HEAD moves so the stamp stays current.
    if let Some(git_dir) = git(&["rev-parse", "--git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/HEAD");
    }
}
