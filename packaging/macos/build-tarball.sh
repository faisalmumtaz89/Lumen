#!/usr/bin/env bash
#
# build-tarball.sh — produce a distributable macOS arm64 (Metal) Lumen tarball.
#
# Output: dist/lumen-<tag>-macos-arm64-metal.tar.gz containing:
#   bin/lumen          (CLI: pull / convert / models / run)
#   bin/lumen-server   (OpenAI/Anthropic-compatible HTTP server)
#   README.txt         (run note + prereqs)
#   LICENSE            (if present at repo root)
#
# The Metal backend compiles its MSL shaders at runtime from source embedded in
# the binary (sub-second). There is NO separate .metallib or shader directory to
# ship — the binaries are self-contained.
#
# Usage (local dev build):
#   packaging/macos/build-tarball.sh            # tag = b<git-rev-count> (local only)
#   LUMEN_TAG=v1.0.0 packaging/macos/build-tarball.sh
# Releases are cut by pushing a v<X.Y.Z> tag; release.yml sets LUMEN_TAG to the tag.
# See RELEASING.md.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Preconditions ────────────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "error: this tarball must be built on macOS Apple Silicon (arm64)." >&2
    exit 1
fi

TAG="${LUMEN_TAG:-b$(git rev-list --count HEAD)}"
COMMIT="$(git rev-parse --short HEAD)"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
OUT_DIR="$REPO_ROOT/dist"
STAGE="$(mktemp -d)/lumen-$TAG-macos-arm64-metal"
trap 'rm -rf "$(dirname "$STAGE")"' EXIT

echo "[build-tarball] tag=$TAG commit=$COMMIT"

# ── Build (Metal is the macOS default; NO --features metal — it does not exist
#    on lumen-server, and `bin,metal` is rejected. Metal compiles automatically) ─
echo "[build-tarball] building lumen (CLI)…"
cargo build --release -p lumen-cli
echo "[build-tarball] building lumen-server…"
cargo build --release -p lumen-server --features bin

# ── Stage ────────────────────────────────────────────────────────────────────
mkdir -p "$STAGE/bin"
cp "$TARGET_DIR/release/lumen"        "$STAGE/bin/"
cp "$TARGET_DIR/release/lumen-server" "$STAGE/bin/"
# Dual-licensed MIT OR Apache-2.0 — ship both license texts.
for lic in LICENSE-APACHE LICENSE-MIT LICENSE; do
    [[ -f "$REPO_ROOT/$lic" ]] && cp "$REPO_ROOT/$lic" "$STAGE/"
done

# Strip is already effectively done by cargo release profile; re-assert arch.
for b in lumen lumen-server; do
    lipo -archs "$STAGE/bin/$b" | grep -qx arm64 \
        || { echo "error: $b is not arm64" >&2; exit 1; }
done

cat > "$STAGE/README.txt" <<EOF
Lumen — LLM inference for Apple Silicon (Metal)
Build: $TAG ($COMMIT)   LBC format: v4

PREREQUISITES
  - Apple Silicon Mac (M-series; M2+ recommended for BF16 models)
  - macOS 14 (Sonoma) or newer
  - No Xcode, no Python, no CUDA, no extra runtime. The binaries link only
    system frameworks (Metal, Foundation, MetalPerformanceShaders[Graph],
    Accelerate, libobjc, libSystem). Shaders compile at first run (~1 s).

INSTALL
  Copy the two binaries onto your PATH:
    sudo cp bin/lumen bin/lumen-server /usr/local/bin/
  (or add this folder's bin/ to your PATH).

  Gatekeeper note: these binaries are unsigned (ad-hoc, not notarized). After
  download macOS quarantines them; clear the bit BEFORE first run (the install.sh
  installer and Homebrew do this for you):
    xattr -dr com.apple.quarantine bin/lumen bin/lumen-server
  (A CLI binary has no Finder "Open" dialog, so clear it from the terminal.)

RUN (chat in one command)
  lumen pull qwen3.5-9b:q8_0          # downloads + converts to ~/.cache/lumen
  lumen run qwen3.5-9b:q8_0 "Write a haiku about Rust"

RUN (HTTP server, OpenAI-compatible)
  lumen pull qwen3.5-moe-35b-a3b:q8_0
  lumen-server --model qwen3.5-moe-35b-a3b --quant q8_0 --port 8000
  curl http://127.0.0.1:8000/v1/models

MODELS
  lumen models                        # list registry cells + cached LBCs
EOF

# ── Pack ─────────────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR"
TARBALL="$OUT_DIR/lumen-$TAG-macos-arm64-metal.tar.gz"
tar -C "$(dirname "$STAGE")" -czf "$TARBALL" "$(basename "$STAGE")"

echo "[build-tarball] wrote $TARBALL ($(du -h "$TARBALL" | cut -f1))"
# Emit a basename-relative checksum so `shasum -c <file>.sha256` works for a
# user who downloads it next to the tarball (an absolute path would only verify
# on this build host). install.sh parses column 1 only, so both forms work there.
( cd "$OUT_DIR" && shasum -a 256 "$(basename "$TARBALL")" | tee "$(basename "$TARBALL").sha256" )

# Also emit a tag-LESS alias so install.sh's "latest" channel can fetch a stable
# asset name via GitHub's /releases/latest/download/ redirect (the old hard-coded
# "lumen-latest-..." name was never produced -> the one-line installer 404'd).
ALIAS="$OUT_DIR/lumen-macos-arm64-metal.tar.gz"
cp "$TARBALL" "$ALIAS"
( cd "$OUT_DIR" && shasum -a 256 "$(basename "$ALIAS")" | tee "$(basename "$ALIAS").sha256" )
echo "[build-tarball] alias: $ALIAS"
