#!/usr/bin/env bash
#
# install.sh — one-line installer for Lumen on macOS Apple Silicon (Metal).
#
#   curl -fsSL https://<host>/install.sh | bash
#
# Downloads the latest (or $LUMEN_TAG) macOS arm64 tarball, verifies its
# SHA-256, installs the two binaries to a prefix, and clears the Gatekeeper
# quarantine bit (the artifacts are unsigned — Phase 1; notarization is deferred).
#
# This mirrors the README single-command flow: after install, `lumen pull` +
# `lumen "<prompt>"` or `lumen-server --model …` work immediately.
#
set -euo pipefail

# ── Config (override via env) ────────────────────────────────────────────────
RELEASE_BASE="${LUMEN_RELEASE_BASE:-https://github.com/faisalmumtaz89/Lumen/releases}"
LUMEN_TAG="${LUMEN_TAG:-latest}"
PREFIX="${LUMEN_PREFIX:-/usr/local/bin}"

# ── Platform gate ────────────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: Lumen's Metal build is macOS-only." >&2; exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "error: this build targets Apple Silicon (arm64). Intel Macs are unsupported." >&2; exit 1
fi
# macOS 14+ floor (MSL 3.1). sw_vers reports e.g. 14.5 / 26.3.1 .
OSVER="$(sw_vers -productVersion)"; OSMAJ="${OSVER%%.*}"
if (( OSMAJ < 14 )); then
    echo "error: macOS 14 (Sonoma) or newer required; found $OSVER." >&2; exit 1
fi

if [[ "$LUMEN_TAG" == "latest" ]]; then
    # The release publishes a tag-LESS alias asset (lumen-macos-arm64-metal.tar.gz)
    # alongside the tagged one; GitHub's /releases/latest/download/ redirect resolves
    # it to the newest release. (The old hard-coded "lumen-latest-..." name was never
    # produced -> 404.)
    URL="$RELEASE_BASE/latest/download/lumen-macos-arm64-metal.tar.gz"
else
    URL="$RELEASE_BASE/download/$LUMEN_TAG/lumen-${LUMEN_TAG}-macos-arm64-metal.tar.gz"
fi
SHA_URL="$URL.sha256"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
echo "[install] downloading $URL"
curl -fsSL "$URL" -o "$TMP/pkg.tar.gz"

# ── Verify checksum (skip only if no .sha256 published) ──────────────────────
if curl -fsSL "$SHA_URL" -o "$TMP/pkg.sha256" 2>/dev/null; then
    EXPECT="$(awk '{print $1}' "$TMP/pkg.sha256")"
    ACTUAL="$(shasum -a 256 "$TMP/pkg.tar.gz" | awk '{print $1}')"
    if [[ "$EXPECT" != "$ACTUAL" ]]; then
        echo "error: checksum mismatch (expected $EXPECT, got $ACTUAL)." >&2; exit 1
    fi
    echo "[install] checksum OK"
else
    echo "[install] warning: no published .sha256 — skipping integrity check." >&2
fi

tar -C "$TMP" -xzf "$TMP/pkg.tar.gz"
SRC="$(find "$TMP" -maxdepth 1 -type d -name 'lumen-*-macos-arm64-metal')"

echo "[install] installing to $PREFIX (may prompt for sudo)"
for b in lumen lumen-server; do
    if [[ -w "$PREFIX" ]]; then install -m 0755 "$SRC/bin/$b" "$PREFIX/$b";
    else sudo install -m 0755 "$SRC/bin/$b" "$PREFIX/$b"; fi
    # Unsigned artifacts: clear quarantine so they launch without a prompt.
    xattr -dr com.apple.quarantine "$PREFIX/$b" 2>/dev/null || true
done

echo "[install] done."
echo
echo "  lumen pull qwen3.5-9b:q8_0"
echo "  lumen \"Write a haiku about Rust\""
echo
echo "  lumen-server --model qwen3.5-9b --quant q8_0 --port 8000"
