# CI setup — Lumen

This PR moves the workflows into `.github/workflows/` (they were staged inert in
`packaging/ci/`), so **merging it activates CI**. The Modal secret and the Mac
runner are one-time account/repo setup; the rest are repo *settings*.

## The design in one screen

Lumen defers **all** GPU kernel compilation to runtime (NVRTC for CUDA, MSL for
Metal), so **building needs no GPU and no vendor SDK** — every artifact is built on
free GitHub-hosted runners. GPUs are needed only to *validate*:

| Workflow | Trigger | Runner | Cost / risk |
|---|---|---|---|
| `ci.yml` | every push + PR | hosted ubuntu + macos-14 | free — build both binaries, CPU tests, lint, link-audit |
| `validate-cuda-modal.yml` | push main / PR (same-repo) / weekly / manual | hosted ubuntu → **Modal** | Modal $ — A100 smoke; skips docs-only PRs; full matrix on manual dispatch |
| `validate-metal.yml` | push main / manual | **self-hosted Mac** | your Mac's GPU; **never** fork PRs |
| `release.yml` | tag `b*`/`v*` | hosted (+ Modal + Mac to validate) | **build → validate the exact bytes → publish only if green** |

CUDA validation runs **on Modal**, driven from a hosted runner that just holds the
token — no always-on exposed GPU box. The only self-hosted runner is the Mac
(hosted macOS runners have no usable Metal GPU). Tags are validated by `release.yml`
on the *promoted artifact*, so the two validate workflows don't trigger on tags.

## Step 1 — Modal secret (CUDA validation) — ✅ done this session

`modal` auth is a token pair (`~/.modal.toml`). Added as encrypted repo secrets
(Settings → Secrets and variables → Actions): `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.
GitHub encrypts them at rest, masks them in logs, withholds them from fork PRs, and
they live only on the hosted runner — never on the Mac.

Seed the model volume the harness reads (one-time, ~9 GB — canonical command):

```bash
modal volume put lumen-models <path-to-9B-q8.lbc> /9bq8.lbc   # dest key MUST be /9bq8.lbc
```

(The volume auto-creates; `modal volume create` is not needed.)

## Step 2 — Mac self-hosted runner (Metal validation) — ✅ done this session

Registered as `<host>-metal` with labels `self-hosted,macos,arm64,metal`, installed
as a launchd service (`./svc.sh install && ./svc.sh start`).

**Strongly recommended:** point the runner at a cached model so the Metal gate
doesn't pull 9 GB (and isn't coupled to registry availability) on every run — set in
the runner's environment:

```bash
export LUMEN_TEST_MODEL=/path/to/qwen3.5-9b-q8_0.lbc
```

## Step 3 — repo settings (do these in the GitHub UI)

### Security (required for a public repo + self-hosted runner)

1. `validate-metal.yml` has **no `pull_request` trigger** — fork code can never run
   on your Mac. Keep it that way.
2. Settings → Actions → General → "Fork pull request workflows from outside
   collaborators" → **Require approval for all outside collaborators**.
3. Run the Mac runner as a **dedicated low-privilege user**, not your daily login;
   no secrets on the runner.
4. *(Optional, stronger)* Put the Metal jobs behind a GitHub **Environment** named
   `metal` with **required reviewers** — then even an owner/collaborator
   `workflow_dispatch` on an arbitrary branch needs a human approval before it runs
   on the Mac. (Add `environment: metal` to the metal jobs to enable.)

### Branch protection (makes "green = mergeable")

Settings → Branches → add a rule for `main` → **Require status checks to pass**, and
select these checks:

- `test-cpu` (from `ci.yml`)
- **`cuda-gate`** — the always-running gate job. **Do NOT pick `modal cuda
  validation`** as the required check: it is skipped on docs-only / fork PRs and a
  skipped check never reports, which would deadlock the PR. `cuda-gate` always
  reports (pass when validation succeeded *or* was legitimately skipped).

Metal is intentionally *not* a required PR check (self-hosted; runs post-merge on
`main`, or on-demand via `workflow_dispatch` against a branch before merging).

## Acceptance (Phase-1 "done")

- `ci.yml` green on push: builds both binaries, CPU suite passes, link-audit clean.
- `validate-cuda-modal.yml`: A100 smoke green on a same-repo PR / push to main;
  `VALIDATION PASSED`, DET-001 1-distinct, PTX cache cold→warm.
- `validate-metal.yml`: DET-001 1-distinct + coherence on the Mac (push to main).
- `release.yml` on a tag: build → CUDA full matrix + macOS DET-001 both green →
  *then* publishes tarballs (macOS + Linux/CUDA) + filled `lumen.rb` + ghcr image.

## Deferred to a later phase (known, documented — not silently relied on)

- **macOS code signing / notarization.** Binaries are unsigned ad-hoc; `install.sh`
  and Homebrew clear the quarantine bit (fine for those paths). A bare download
  needs `xattr -dr com.apple.quarantine` from the terminal. Add Developer-ID
  signing + `notarytool`/`stapler` when an Apple account is available.
- **Homebrew tap.** `release.yml` generates a *correct* `dist/lumen.rb` (filled
  url/version/sha256) and attaches it to the Release. Publishing it is one manual
  step: create `faisalmumtaz89/homebrew-lumen` and copy `dist/lumen.rb` to
  `Formula/lumen.rb` (or wire an auto-push later). Until then, `brew install` from a
  tap is not advertised.
- **Validation breadth.** CI validates 9B-q8 (the startup/determinism/coherence
  class — quant/model-independent for the startup bug). q4/bf16 + MoE/27B
  *correctness* is the manual gold-standard quality suite, not CI.
- **Supply-chain provenance.** Actions are SHA-pinned + Dependabot-tracked; image
  signing / SLSA attestation is a later add.
