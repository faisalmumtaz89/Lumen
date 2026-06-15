#!/usr/bin/env bash
#
# install.sh — one-command, cross-platform binary onboarder for Lumen.
#
#   curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Lumen/main/packaging/macos/install.sh | bash
#
# Detects your machine (macOS Apple Silicon -> Metal · Linux x86_64 + NVIDIA ->
# CUDA), downloads the matching *validated* prebuilt binaries from the latest
# GitHub release (verifying their SHA-256), installs `lumen` + `lumen-server`,
# lets you pick one of the supported models and a quant, prepares it, and prints
# the exact command to start. No Rust toolchain, no CUDA SDK.
#
# Non-interactive / overrides (flags after `bash -s --`, or env):
#   --model <alias>   LUMEN_MODEL     (qwen3.5-9b | qwen3.5-moe | qwen3.6-27b; accepts name:quant)
#   --quant <tag>     LUMEN_QUANT     (q8_0 | q4_0 | bf16; default q8_0)
#   --yes, -y                          non-interactive (accept defaults, no prompts)
#   --prefix <dir>    LUMEN_PREFIX    (install dir; default /usr/local/bin)
#   LUMEN_TAG         pin a release tag (e.g. v0.1.0, or a v..-rc.N prerelease); default = latest
#   LUMEN_CACHE_DIR   model cache location (passed through to lumen)
#   LUMEN_RELEASE_BASE / LUMEN_INSECURE_SKIP_CHECKSUM   (advanced/testing)
#
# (Path note: this script lives under packaging/macos/ for URL stability but is
#  cross-platform — it serves both the macOS Metal and Linux CUDA binaries.)
set -euo pipefail

REPO="faisalmumtaz89/Lumen"
RELEASE_BASE="${LUMEN_RELEASE_BASE:-https://github.com/$REPO/releases}"
API_LATEST="https://api.github.com/repos/$REPO/releases/latest"
TAG="${LUMEN_TAG:-latest}"
PREFIX="${LUMEN_PREFIX:-/usr/local/bin}"
DEFAULT_MODEL="qwen3.5-9b"
DEFAULT_QUANT="q8_0"

MODEL="${LUMEN_MODEL:-}"
QUANT="${LUMEN_QUANT:-}"
ASSUME_YES=0

say()  { printf '%s\n' "$*"; }
info() { printf '[install] %s\n' "$*"; }
err()  { printf 'error: %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

# ── Arg parse ─────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
  case "$1" in
    --model)   [ $# -ge 2 ] || die "--model needs a value";  MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --quant)   [ $# -ge 2 ] || die "--quant needs a value";  QUANT="$2"; shift 2 ;;
    --quant=*) QUANT="${1#*=}"; shift ;;
    --prefix)  [ $# -ge 2 ] || die "--prefix needs a value"; PREFIX="$2"; shift 2 ;;
    --prefix=*) PREFIX="${1#*=}"; shift ;;
    --yes|-y)  ASSUME_YES=1; shift ;;
    -h|--help) sed -n '2,28p' "$0" 2>/dev/null || say "see header"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

# A name:quant in --model/LUMEN_MODEL sets the quant and skips the quant menu.
case "$MODEL" in
  *:*) [ -n "$QUANT" ] || QUANT="${MODEL##*:}"; MODEL="${MODEL%%:*}" ;;
esac

# ── Step 0 · detect platform ──────────────────────────────────────────────────
have() { command -v "$1" >/dev/null 2>&1; }
has_nvidia() {
  [ -e /dev/nvidia0 ] && return 0
  have nvidia-smi && nvidia-smi >/dev/null 2>&1 && return 0
  return 1
}

OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS" in
  Darwin)
    [ "$ARCH" = "arm64" ] || die "Lumen's prebuilt macOS binary is Apple Silicon (arm64) only; Intel Macs must build from source: cargo install --path crates/lumen-cli"
    OSVER="$(sw_vers -productVersion)"; OSMAJ="${OSVER%%.*}"
    [ "${OSMAJ:-0}" -ge 14 ] || die "macOS 14 (Sonoma) or newer required; found $OSVER."
    PLAT="macos-arm64-metal"; BACKEND="metal"
    ;;
  Linux)
    [ "$ARCH" = "x86_64" ] || die "no prebuilt Linux binary for $ARCH (x86_64 only). Build from source: cargo install --path crates/lumen-cli --features cuda"
    if has_nvidia; then
      PLAT="linux-x86_64-cuda"; BACKEND="cuda"
    else
      err "no NVIDIA GPU detected — the prebuilt Linux binary is CUDA-only."
      err "For a CPU build, build from source: cargo install --path crates/lumen-cli   (or ./scripts/quickstart.sh)"
      exit 1
    fi
    ;;
  *) die "no prebuilt binary for $OS/$ARCH; build from source: https://github.com/$REPO" ;;
esac
info "platform: $OS/$ARCH -> $BACKEND ($PLAT)"

for tool in curl tar; do have "$tool" || die "'$tool' is required but not found"; done
sha256_of() { if have shasum; then shasum -a 256 "$1" | awk '{print $1}'; else sha256sum "$1" | awk '{print $1}'; fi; }
have shasum || have sha256sum || die "need 'shasum' or 'sha256sum' for checksum verification"

# ── Step 1 · resolve the download URL (alias-first, GitHub-API tag fallback) ───
url_ok() { curl -fsIL -o /dev/null "$1" >/dev/null 2>&1; }
resolve_asset_url() {
  local plat="$1"
  if [ "$TAG" != "latest" ]; then
    printf '%s\n' "$RELEASE_BASE/download/$TAG/lumen-$TAG-$plat.tar.gz"; return 0
  fi
  local alias_url="$RELEASE_BASE/latest/download/lumen-$plat.tar.gz"
  if url_ok "$alias_url"; then printf '%s\n' "$alias_url"; return 0; fi
  # Fallback: the latest *stable* (non-prerelease) tag from the API, then the tag-named asset.
  local tag
  tag="$(curl -fsSL "$API_LATEST" 2>/dev/null | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/' || true)"
  if [ -z "$tag" ]; then
    die "could not resolve a stable release (no published release yet, or the GitHub API is rate-limited). Pin one with LUMEN_TAG=<tag>."
  fi
  printf '%s\n' "$RELEASE_BASE/download/$tag/lumen-$tag-$plat.tar.gz"
}
URL="$(resolve_asset_url "$PLAT")"

# ── Step 2 · download + verify checksum (abort installs nothing on mismatch) ──
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
info "downloading $URL"
curl -fSL "$URL" -o "$TMP/pkg.tar.gz" || die "download failed: $URL"
if curl -fsSL "$URL.sha256" -o "$TMP/pkg.sha256" 2>/dev/null; then
  want="$(awk '{print $1}' "$TMP/pkg.sha256")"; got="$(sha256_of "$TMP/pkg.tar.gz")"
  [ "$want" = "$got" ] || die "checksum mismatch (want $want, got $got) — installing nothing."
  info "checksum OK"
elif [ "${LUMEN_INSECURE_SKIP_CHECKSUM:-0}" = "1" ]; then
  info "WARNING: no .sha256 published; skipping verification (LUMEN_INSECURE_SKIP_CHECKSUM=1)."
else
  die "no checksum (.sha256) for $URL; refusing. Set LUMEN_INSECURE_SKIP_CHECKSUM=1 to override."
fi

# ── Step 3 · install both binaries (version-agnostic glob; sudo / ~/.local fb) ─
tar -C "$TMP" -xzf "$TMP/pkg.tar.gz"
SRC="$(find "$TMP" -maxdepth 1 -type d -name "lumen-*-$PLAT" | head -1)"
[ -n "$SRC" ] && [ -d "$SRC/bin" ] || die "unexpected tarball layout under $TMP"

put() { # $1=dir $2=sudo?
  local d="$1" sudo="$2" b
  for b in lumen lumen-server; do
    $sudo install -m 0755 "$SRC/bin/$b" "$d/$b" || return 1
  done
}
DEST="$PREFIX"
if [ -d "$PREFIX" ] && [ -w "$PREFIX" ]; then
  put "$PREFIX" "" || die "install to $PREFIX failed"
elif have sudo; then
  info "installing to $PREFIX (sudo may prompt for your password)"
  put "$PREFIX" "sudo" || die "sudo install to $PREFIX failed"
else
  DEST="$HOME/.local/bin"; mkdir -p "$DEST"
  put "$DEST" "" || die "install to $DEST failed"
  info "installed to $DEST ($PREFIX not writable and no sudo)"
fi
if [ "$BACKEND" = "metal" ]; then
  xattr -dr com.apple.quarantine "$DEST/lumen" "$DEST/lumen-server" 2>/dev/null || true
fi
LUMEN="$DEST/lumen"
info "installed: $DEST/lumen, $DEST/lumen-server ($("$LUMEN" --version 2>/dev/null || echo '?'))"
case ":$PATH:" in *":$DEST:"*) ;; *) info "NOTE: add to PATH:  export PATH=\"$DEST:\$PATH\"" ;; esac
resolved="$(command -v lumen 2>/dev/null || true)"
if [ -n "$resolved" ] && [ "$resolved" != "$DEST/lumen" ]; then
  info "NOTE: 'lumen' on PATH currently resolves to $resolved (not $DEST/lumen)"
fi

# ── Step 4 · pick a model + quant (interactive via /dev/tty; safe under pipe) ──
INTERACTIVE=0
if [ -z "$MODEL" ] && [ "$ASSUME_YES" != "1" ] && [ -r /dev/tty ]; then INTERACTIVE=1; fi
if [ "$INTERACTIVE" = "1" ]; then
  say ""
  say "  Which model?"
  say "    1) Qwen3.5 9B     (dense, ~10 GB at Q8 — fast, recommended)"
  say "    2) Qwen3.5 MoE    (30B-A3B mixture-of-experts)"
  say "    3) Qwen3.6 27B    (dense, largest)"
  printf '  > '
  read -r pick < /dev/tty || pick=""
  case "$pick" in 2) MODEL="qwen3.5-moe" ;; 3) MODEL="qwen3.6-27b" ;; *) MODEL="$DEFAULT_MODEL" ;; esac
  say ""
  say "  Which version (quant)?"
  say "    1) Q8    (recommended — best quality/size balance)   [default]"
  say "    2) Q4    (smaller, faster, slightly lower quality)"
  say "    3) BF16  (full precision, largest — needs lots of memory)"
  printf '  > '
  read -r qpick < /dev/tty || qpick=""
  case "$qpick" in 2) QUANT="q4_0" ;; 3) QUANT="bf16" ;; *) QUANT="$DEFAULT_QUANT" ;; esac
else
  MODEL="${MODEL:-$DEFAULT_MODEL}"; QUANT="${QUANT:-$DEFAULT_QUANT}"
  [ "$ASSUME_YES" = "1" ] || info "no terminal: using default $MODEL:$QUANT (pass --model/--quant to choose)"
fi
info "selected: $MODEL:$QUANT"

# ── Step 5 · prepare the model (installer owns consent -> pull --yes) ─────────
cache="${LUMEN_CACHE_DIR:-}"
if [ -z "$cache" ]; then
  case "$BACKEND" in metal) cache="$HOME/Library/Caches/lumen" ;; *) cache="${XDG_CACHE_HOME:-$HOME/.cache}/lumen" ;; esac
fi
mkdir -p "$cache" 2>/dev/null || true
free_gb="$(df -Pk "$cache" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}')"
if [ -n "${free_gb:-}" ] && [ "$free_gb" -lt 15 ]; then
  info "WARNING: only ~${free_gb} GB free at $cache; models need 5-70 GB (the source GGUF + converted LBC coexist during prepare, and downloads are not resumable)."
  if [ "$INTERACTIVE" = "1" ]; then
    printf '  Continue anyway? [y/N] '
    read -r go < /dev/tty || go=""
    case "$go" in y|Y|yes) ;; *) die "aborted (low disk)." ;; esac
  fi
fi
info "preparing $MODEL:$QUANT (download + convert; one-time)"
LUMEN_CACHE_DIR="$cache" "$LUMEN" pull "$MODEL:$QUANT" --yes

# ── Step 6 · print the exact next steps (correct positional form) ─────────────
say ""
say "  Lumen is ready. The model is cached, so the server starts instantly."
say ""
say "  Chat:"
say "    lumen run $MODEL:$QUANT \"Write a haiku about Rust\""
say ""
say "  OpenAI/Anthropic-compatible server:"
say "    lumen-server --model $MODEL --quant $QUANT --backend $BACKEND --port 8000"
say ""
case ":$PATH:" in *":$DEST:"*) ;; *) say "  (first: export PATH=\"$DEST:\$PATH\")" ;; esac
