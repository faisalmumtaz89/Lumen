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

    // Re-run when the stamp's inputs move. HEAD is a symbolic ref that a
    // same-branch commit never rewrites, so the whole refs directory of the
    // common dir is watched (loose refs, tags, and refs re-created after a
    // pack), plus packed-refs when it exists — only existing paths are
    // declared, since Cargo treats a missing declared path as always
    // changed. Declaring any path replaces Cargo's default watch on this
    // crate's sources, so they are declared again for the `-dirty` marker,
    // as is the registry TOML the CLI embeds with `include_str!`.
    // Limits: edits elsewhere in the workspace also make `git describe`
    // dirty but do not re-run this script, so the stamp is not a build
    // identity — two byte-different binaries can carry one stamp; harness
    // records identify a binary by its sha256. A fetch that writes a remote
    // ref also re-runs the script (one relink, byte-identical output).
    // The registry TOML is embedded by the CLI regardless of git, so it is
    // watched outside the git-dependent block.
    if std::path::Path::new("../../model_registry.toml").exists() {
        println!("cargo:rerun-if-changed=../../model_registry.toml");
    }
    // Declaring any watch path replaces Cargo's default "re-run on any
    // package change", so the crate's own source and manifest are declared
    // unconditionally: a source edit must re-stamp in a tree without git
    // (a source tarball) exactly as it does in a checkout.
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    if let (Some(git_dir), Some(common_dir)) = (
        git(&["rev-parse", "--git-dir"]),
        git(&["rev-parse", "--git-common-dir"]),
    ) {
        let candidates = [
            format!("{git_dir}/HEAD"),
            format!("{common_dir}/refs"),
            format!("{common_dir}/packed-refs"),
        ];
        for path in candidates
            .iter()
            .filter(|p| std::path::Path::new(p).exists())
        {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}
