//! Pins for the two build scripts' watch declarations. The scripts run at
//! build time and cannot be unit-tested directly; their text can.
#[cfg(test)]
mod tests {
    /// The two scripts, read from the workspace when this crate is built in
    /// it; a packaged crate has only its own, and the sibling is skipped.
    fn scripts() -> Vec<(&'static str, String)> {
        let own = concat!(env!("CARGO_MANIFEST_DIR"), "/build.rs");
        let sibling = concat!(env!("CARGO_MANIFEST_DIR"), "/../lumen-server/build.rs");
        [
            ("lumen-cli/build.rs", own),
            ("lumen-server/build.rs", sibling),
        ]
        .into_iter()
        .filter_map(|(name, path)| std::fs::read_to_string(path).ok().map(|s| (name, s)))
        .collect()
    }

    /// Declaring any watch path replaces Cargo's default source watch, so
    /// `src` and `Cargo.toml` must be declared outside the git guard or a
    /// source edit stops re-stamping in a tree without git.
    #[test]
    fn source_and_manifest_are_watched_before_any_git_guard() {
        let scripts = scripts();
        assert!(
            !scripts.is_empty(),
            "the crate's own build.rs must be readable"
        );
        for (rel, s) in scripts {
            let guard = s
                .find("if let (Some(git_dir), Some(common_dir))")
                .unwrap_or_else(|| panic!("{rel}: no git guard"));
            for decl in [
                "cargo:rerun-if-changed=src",
                "cargo:rerun-if-changed=Cargo.toml",
            ] {
                let at = s
                    .find(decl)
                    .unwrap_or_else(|| panic!("{rel}: {decl} not declared"));
                assert!(at < guard, "{rel}: {decl} sits inside the git guard");
                let line = s[..at].rsplit('\n').next().unwrap_or("");
                assert!(
                    !line.contains("if "),
                    "{rel}: {decl} is conditional: {line}"
                );
            }
        }
    }
}
