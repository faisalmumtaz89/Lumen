# Security Policy

## Supported versions

Lumen is pre-release (no published version yet). The current `main` branch is
the only supported tree. Once `0.1.0` is published to crates.io, this matrix
will list supported versions.

| Version | Supported |
|---------|-----------|
| `main` (HEAD) | yes |
| (no released versions yet) | — |

## Reporting a vulnerability

Email: **faisalmumtaz89@gmail.com**

Please include:

- A description of the issue
- Step-by-step reproduction (a minimal test case if possible)
- The commit SHA you observed it on
- Your assessment of the impact (model-output corruption, RCE, DoS, info leak)

You should expect an initial response within 7 days. Critical issues
(unauthenticated RCE, data exfiltration from `lumen-server`) are prioritized
above all other work. Lower-severity issues (input validation, error-handling
gaps) are scheduled into the next release cycle.

## In scope

- `lumen` CLI binary
- `lumen-server` HTTP server (axum router, OpenAI + Anthropic wire formats)
- `lumen-convert` GGUF reader (untrusted input parsing)
- `lumen-format` LBC reader (untrusted input parsing)
- All NVRTC/CUDA + Metal shader compilation paths (untrusted-input-derived
  kernel arguments)

## Out of scope

- Vulnerabilities requiring local-root or `LD_PRELOAD`-style attacker control
- Model-output quality issues (these are bugs, not vulnerabilities — open a
  normal issue)
- DoS via legitimate-but-expensive operations (e.g. `--max-tokens 100000`
  decode is slow but not a vulnerability)

## Disclosure

By default Lumen practices coordinated disclosure: report → confirm → fix →
release → disclose, with up to 90 days from report to public disclosure for
fix landing. Earlier disclosure may be coordinated if the user reporting
prefers.
