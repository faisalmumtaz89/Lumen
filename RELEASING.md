# Releasing Lumen

Lumen ships pre-built binaries for **Linux x86_64 / CUDA** and **macOS arm64 /
Metal**. A release is cut by pushing **one git tag** — everything else is
automated by [`.github/workflows/release.yml`](.github/workflows/release.yml).

## Versioning policy — Semantic Versioning

Releases use [SemVer](https://semver.org/) tags of the form **`vMAJOR.MINOR.PATCH`**:

| Bump | When | Example |
|---|---|---|
| **MAJOR** | breaking change to the CLI flags, server wire API, or LBC format | `v1.0.0 → v2.0.0` |
| **MINOR** | new backward-compatible feature (new model, flag, endpoint) | `v0.1.0 → v0.2.0` |
| **PATCH** | backward-compatible bug fix / perf improvement only | `v0.1.0 → v0.1.1` |

**Pre-releases** append a hyphenated suffix: **`vX.Y.Z-rc.N`** (release candidate),
also `-beta.N` / `-alpha.N`. A hyphenated tag is published as a GitHub
*pre-release* and does **not** move the `latest` channel or the Docker `:latest`
tag — so it is safe for a dry-run.

The binary self-reports its exact version (`lumen --version`): `build.rs` stamps
`git describe --tags`, so a build at the `v0.1.0` commit reports `v0.1.0`, and a
build five commits later reports `v0.1.0-5-g<sha>`. The tag is the source of truth.

## What a tag publishes

```
  build-linux ─┐                          ┌─ publish-docker  → ghcr.io/faisalmumtaz89/lumen
               ├─ validate-cuda (Modal) ───┤
               │                          └─ publish-release ─┐  GitHub Release:
  build-macos ─┴─ validate-macos (Mac) ───────────────────────┘   • tarballs + .sha256
                                                                   • lumen.rb (Homebrew)
```

The binaries are **built once**, then those **exact bytes** are validated on real
hardware (CUDA full arch matrix on Modal + Metal on the self-hosted Mac), and are
published **only if validation is green**. What ships is provably what was
validated — artifacts are promoted (re-used), never independently rebuilt.

Assets attached to each release:

- `lumen-<tag>-macos-arm64-metal.tar.gz` (+ `.sha256`)
- `lumen-<tag>-linux-x86_64-cuda.tar.gz` (+ `.sha256`)
- `lumen-macos-arm64-metal.tar.gz` — tag-less alias so the one-line installer's
  `latest` channel resolves via GitHub's `/releases/latest/download/` redirect
- `lumen.rb` — Homebrew formula with url/version/sha256 filled in
- Docker image `ghcr.io/faisalmumtaz89/lumen:<tag>` (and `:latest` on a final release)

## Cutting a release

### 1. Prep PR (on a branch, merged to `main` first)

- Move the `## [Unreleased]` section of [`CHANGELOG.md`](CHANGELOG.md) to a new
  `## [X.Y.Z] — <date>` heading; add the compare links at the bottom.
- Bump `workspace.package.version` in the root `Cargo.toml` to `X.Y.Z`
  (keeps `Cargo.toml` and crate metadata in sync with the tag).
- Merge the PR. `main` must be green (ci + `cuda-gate` + Metal).

### 2. Tag and push — this triggers the release

```bash
git checkout main && git pull
git tag v0.1.0           # annotated is fine too: git tag -a v0.1.0 -m "Lumen 0.1.0"
git push origin v0.1.0
```

Watch it: `gh run watch` (or the **Actions → release** tab). On success the
GitHub Release, Docker image, and Homebrew formula are published.

### 3. (First release / risky change) dry-run with a pre-release first

The release pipeline's publish glue runs end-to-end only on a real tag. For the
**first release** — or any change to packaging — cut a release candidate first:

```bash
git tag v0.1.0-rc.1 && git push origin v0.1.0-rc.1
```

This runs the full build → validate → publish path but marks the GitHub Release as
a **pre-release** (it will *not* become "Latest", and Docker `:latest` is not
moved). Inspect the published tarballs, the `ghcr` image, and `lumen.rb`. When
satisfied, cut the final `v0.1.0` (step 2).

## After the release (one manual step, until a tap is wired)

`release.yml` generates a correct `dist/lumen.rb` and attaches it to the Release.
To enable `brew install`, copy it into the tap repo once per release:

```bash
# in faisalmumtaz89/homebrew-lumen
cp <downloaded>/lumen.rb Formula/lumen.rb && git commit -am "lumen 0.1.0" && git push
```

(See "Deferred" in [`packaging/ci/SETUP.md`](packaging/ci/SETUP.md) — code signing /
notarization and a tap auto-push are tracked there.)

## Fixing a botched release

A tag is just a pointer — delete and re-cut if validation publishes something
wrong (validation failure means nothing is published, so this is rare):

```bash
gh release delete v0.1.0 --yes          # remove the GitHub Release
git push origin :refs/tags/v0.1.0       # delete the remote tag
git tag -d v0.1.0                        # delete the local tag
# (also delete the bad ghcr image version in the package UI if one was pushed)
```

Prefer rolling **forward** with a `vX.Y.(Z+1)` patch once a release is public.
