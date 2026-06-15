# CI setup — Lumen Phase-1

These workflow files live in `packaging/ci/` (staged, inert) on purpose: they do
**not** run until you (1) add the Modal secret, (2) register the Mac runner, and
(3) move them into `.github/workflows/`. That ordering avoids red first runs.

## The design in one screen

Lumen defers **all** GPU kernel compilation to runtime (NVRTC for CUDA, MSL for
Metal), so **building needs no GPU and no vendor SDK** — every artifact is built on
free GitHub-hosted runners. GPUs are needed only to *validate*:

| Workflow | Trigger | Runner | Cost / risk |
|---|---|---|---|
| `ci.yml` | every push + PR | hosted ubuntu + macos-14 | free — build, CPU tests, lint, link-audit, build-smoke |
| `validate-cuda-modal.yml` | push main / manual / tag | hosted ubuntu → **Modal** | Modal $ (A100 smoke on main; full sm_75–90 matrix on tags) |
| `validate-metal.yml` | manual / tag / push main | **self-hosted Mac** | your Mac's GPU; release-gated, never fork PRs |
| `release.yml` | tag `b*`/`v*` | hosted | tarball + Homebrew + Docker→ghcr |

CUDA validation runs **on Modal**, driven from a free hosted runner that just holds
the token — so there is **no always-on, exposed self-hosted GPU box**. The only
self-hosted runner is the Mac, because hosted macOS runners have no usable Metal GPU.

## Step 1 — Modal secret (for CUDA validation)

`modal` auth is a token pair (see `~/.modal.toml` on your machine). Add them as
**encrypted repo secrets** (Settings → Secrets and variables → Actions → New secret):

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

The CUDA job exports them as env vars; the `modal` CLI reads them automatically.
GitHub encrypts secrets at rest and masks them in logs. They live only on the
hosted runner, never on the Mac.

One-time, on your machine, seed the model volume the harness reads (9B-q8, ~9 GB):

```bash
modal volume put lumen-models 9bq8.lbc /9bq8.lbc   # path to a converted 9B-q8 .lbc
```

## Step 2 — register the Mac as a self-hosted runner (for Metal validation)

On the Mac: Settings → Actions → Runners → New self-hosted runner (macOS/arm64),
then run the shown commands with these **labels**:

```bash
./config.sh --url https://github.com/faisalmumtaz89/Lumen \
            --token <one-time-registration-token> \
            --labels self-hosted,macos,arm64,metal
./svc.sh install && ./svc.sh start     # run as a launchd service
```

Optional: set a cached model so CI doesn't pull 9 GB each run —
`export LUMEN_TEST_MODEL=/path/to/qwen3.5-9b-q8_0.lbc` in the runner's environment.

### ⚠ SECURITY — required for a public repo

A self-hosted runner executes whatever a workflow gives it, so a malicious **fork
PR** could run code on your Mac. Mitigations (all of these):

1. `validate-metal.yml` has **no `pull_request` trigger** — it fires only on tags,
   manual dispatch, and push-to-main (owner-controlled). Keep it that way.
2. Settings → Actions → General → "Fork pull request workflows from outside
   collaborators" → **Require approval for all outside collaborators**.
3. Don't put secrets on the Mac runner; run it as a low-privilege user.
4. Consider a dedicated runner user account, not your daily login.

## Step 3 — activate

```bash
git mv packaging/ci/ci.yml                  .github/workflows/ci.yml
git mv packaging/ci/validate-cuda-modal.yml .github/workflows/validate-cuda-modal.yml
git mv packaging/ci/validate-metal.yml      .github/workflows/validate-metal.yml
git mv packaging/ci/release.yml             .github/workflows/release.yml
# keep metal_validate.sh where it is (the metal workflow calls packaging/ci/metal_validate.sh)
```

`ci.yml` runs on the next push (should be green: hosted build + CPU tests). The GPU
workflows fire per their triggers once the secret + runner are in place.

## Acceptance (Phase-1 "done")

- `ci.yml` green on push: builds both binaries, CPU suite passes, link-audit clean.
- `validate-cuda-modal.yml`: A100 smoke green on main; full sm_75–90 matrix green on
  a tag (0 kernel-compile failures, DET-001 1-distinct per arch).
- `validate-metal.yml`: DET-001 1-distinct + coherence on the self-hosted Mac.
- `release.yml` on a tag: tarball + sha256 attached, Docker image on ghcr.io.

## Notes / open hardening

- `release.yml` Homebrew step is not wired to auto-bump the formula `sha256`/`url`
  (left manual for the first release; `build-tarball.sh` prints the sha256). Add a
  tap-update step when you publish a Homebrew tap.
- `validate_arches.py` should exit non-zero on a hard gate failure; the workflow
  also greps the emitted matrix as a belt-and-suspenders check — verify the exit
  code on the first real run.
