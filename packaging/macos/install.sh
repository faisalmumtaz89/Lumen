#!/usr/bin/env bash
#
# install.sh — one-command, cross-platform binary onboarder for Lumen.
#
#   curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Lumen/main/packaging/macos/install.sh | bash
#
# Detects your machine (macOS Apple Silicon -> Metal · Linux x86_64 + NVIDIA ->
# CUDA), downloads the matching prebuilt binaries from the latest GitHub release,
# checks their SHA-256 (integrity of the GitHub-hosted asset; authenticity comes
# from HTTPS to github.com), installs `lumen` + `lumen-server`, lets you pick a
# model + quant, prepares it, and prints the exact command to start. No Rust
# toolchain, no CUDA SDK.
#
# Trust note: this pipes a script from the internet into your shell. To inspect
# first:  curl -fsSL <url> -o install.sh && less install.sh && bash install.sh
#
# Non-interactive / overrides (flags after `bash -s --`, or env):
#   --model <alias>   LUMEN_MODEL   (qwen3.5-9b | qwen3.5-moe | qwen3.6-27b; accepts name:quant)
#   --quant <tag>     LUMEN_QUANT   (q8_0 | q4_0 | bf16; default q8_0)
#   --yes, -y                        non-interactive (accept defaults, no prompts)
#   --prefix <dir>    LUMEN_PREFIX  (install dir; default = auto-selected, see below)
#   LUMEN_TAG         pin a release tag (e.g. v0.1.0, or a v..-rc.N prerelease); default = latest
#   LUMEN_CACHE_DIR   model cache location (passed through to lumen)
#   LUMEN_RELEASE_BASE / LUMEN_ALLOW_INSECURE_BASE / LUMEN_INSECURE_SKIP_CHECKSUM  (advanced/testing)
#   NO_COLOR          set to disable colored output (color is also off when stdout isn't a TTY)
#
# Install location (no --prefix/LUMEN_PREFIX): the first directory that is both
# already on $PATH AND writable without sudo is chosen — typically
# ~/.local/bin (Linux), /opt/homebrew/bin (Apple-Silicon Homebrew), or
# /usr/local/bin. No sudo, no profile edits in that case. Failing that, it uses
# /usr/local/bin via sudo (with a clear heads-up), or finally ~/.local/bin with
# a one-line, idempotent PATH addition to your shell profile.
#
# (Path note: this script lives under packaging/macos/ for URL stability but is
#  cross-platform — it serves both the macOS Metal and Linux CUDA binaries.)
set -euo pipefail

REPO="faisalmumtaz89/Lumen"
RELEASE_BASE="${LUMEN_RELEASE_BASE:-https://github.com/$REPO/releases}"
API_LATEST="https://api.github.com/repos/$REPO/releases/latest"
TAG="${LUMEN_TAG:-latest}"
# Empty unless the user passed --prefix/LUMEN_PREFIX. Auto-selection (below)
# only runs when this is empty, so an explicit prefix always wins.
PREFIX="${LUMEN_PREFIX:-}"
DEFAULT_MODEL="qwen3.5-9b"
DEFAULT_QUANT="q8_0"
MOE_CANONICAL="qwen3.5-moe-35b-a3b"   # the alias that round-trips through pull, run AND lumen-server

MODEL="${LUMEN_MODEL:-}"
QUANT="${LUMEN_QUANT:-}"
ASSUME_YES=0

# ── Presentation ──────────────────────────────────────────────────────────────
# Color ONLY when stdout is a real TTY and NO_COLOR is unset (CI/pipe-safe).
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_CYAN=$'\033[36m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''; C_GREEN=''; C_YELLOW=''; C_CYAN=''
fi
OK_MARK="${C_GREEN}✓${C_RESET}"

say()  { printf '%s\n' "$*"; }
# Aligned "Label   value" line for the main flow (replaces the [install] prefix).
field() { printf '  %s%-10s%s %s\n' "$C_BOLD" "$1" "$C_RESET" "$2"; }
info() { printf '  %sinfo%s  %s\n' "$C_DIM" "$C_RESET" "$*"; }
err()  { printf '  %serror%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
die()  { err "$*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
  cat <<'EOF'
Lumen installer — detects your platform, installs the prebuilt binaries, sets up a model.

  curl -fsSL https://raw.githubusercontent.com/faisalmumtaz89/Lumen/main/packaging/macos/install.sh | bash

Options (after `bash -s --`) / env:
  --model <alias>   LUMEN_MODEL   qwen3.5-9b | qwen3.5-moe | qwen3.6-27b  (accepts name:quant)
  --quant <tag>     LUMEN_QUANT   q8_0 | q4_0 | bf16        (default q8_0)
  --yes, -y                       non-interactive (defaults, no prompts)
  --prefix <dir>    LUMEN_PREFIX  install dir   (default: auto — first writable $PATH dir)
  LUMEN_TAG=<tag>   install a specific release (e.g. v0.1.0, or a v..-rc.N prerelease)
EOF
}

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
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

# A name:quant in --model/LUMEN_MODEL sets the quant and skips the quant menu.
case "$MODEL" in
  *:*) [ -n "$QUANT" ] || QUANT="${MODEL##*:}"; MODEL="${MODEL%%:*}" ;;
esac

# RELEASE_BASE must be an https GitHub origin (the .sha256 shares the asset's
# origin, so a non-github base could serve a matching tarball+hash). Override for
# local testing only.
case "$RELEASE_BASE" in
  https://github.com/*|https://*.githubusercontent.com/*) ;;
  *) [ "${LUMEN_ALLOW_INSECURE_BASE:-0}" = "1" ] \
       || die "LUMEN_RELEASE_BASE must be an https://github.com URL (got '$RELEASE_BASE'); set LUMEN_ALLOW_INSECURE_BASE=1 to override (testing only)." ;;
esac

# ── Banner ────────────────────────────────────────────────────────────────────
say ""
say "  ${C_BOLD}Lumen${C_RESET} ${C_DIM}·${C_RESET} installer"
say ""

# ── Step 0 · detect platform ──────────────────────────────────────────────────
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
    PLAT_LABEL="macOS · Apple Silicon → Metal"
    ;;
  Linux)
    [ "$ARCH" = "x86_64" ] || die "no prebuilt Linux binary for $ARCH (x86_64 only). Build from source: cargo install --path crates/lumen-cli --features cuda"
    if has_nvidia; then
      PLAT="linux-x86_64-cuda"; BACKEND="cuda"
      PLAT_LABEL="Linux · x86_64 + NVIDIA → CUDA"
    else
      err "no NVIDIA GPU detected — the prebuilt Linux binary is CUDA-only."
      err "For a CPU build, build from source: cargo install --path crates/lumen-cli   (or ./scripts/quickstart.sh)"
      exit 1
    fi
    ;;
  *) die "no prebuilt binary for $OS/$ARCH; build from source: https://github.com/$REPO" ;;
esac
field "Platform" "$PLAT_LABEL"

for tool in curl tar; do have "$tool" || die "'$tool' is required but not found"; done
sha256_of() { if have shasum; then shasum -a 256 "$1" | awk '{print $1}'; else sha256sum "$1" | awk '{print $1}'; fi; }
have shasum || have sha256sum || die "need 'shasum' or 'sha256sum' for checksum verification"

# ── Step 1 · resolve the download URL (alias-first, GitHub-API tag fallback) ───
url_ok() { curl -fsIL -o /dev/null "$1" >/dev/null 2>&1; }
api_get() {  # honor GITHUB_TOKEN to dodge the 60-req/hr unauthenticated limit
  if [ -n "${GITHUB_TOKEN:-}" ]; then curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" "$1" 2>/dev/null
  else curl -fsSL "$1" 2>/dev/null; fi
}
# Sets RESOLVED_URL and RESOLVED_TAG (the human-readable release, e.g. v0.2.0 or
# "latest"). Kept as globals so Step 2 can show a short asset name + version
# instead of the long URL.
RESOLVED_TAG="$TAG"
resolve_asset_url() {
  local plat="$1"
  if [ "$TAG" != "latest" ]; then
    RESOLVED_URL="$RELEASE_BASE/download/$TAG/lumen-$TAG-$plat.tar.gz"; RESOLVED_TAG="$TAG"; return 0
  fi
  local alias_url="$RELEASE_BASE/latest/download/lumen-$plat.tar.gz"
  if url_ok "$alias_url"; then RESOLVED_URL="$alias_url"; RESOLVED_TAG="latest"; return 0; fi
  # Fallback: latest *stable* (non-prerelease) tag from the API, then the tag-named asset.
  local tag
  tag="$(api_get "$API_LATEST" | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/' || true)"
  if [ -z "$tag" ]; then
    die "could not resolve a stable release (none published yet, only a pre-release, or the GitHub API is rate-limited). Pin one with LUMEN_TAG=<tag> (e.g. a v..-rc.N pre-release), or set GITHUB_TOKEN to raise the API limit."
  fi
  RESOLVED_URL="$RELEASE_BASE/download/$tag/lumen-$tag-$plat.tar.gz"; RESOLVED_TAG="$tag"
}
resolve_asset_url "$PLAT"
URL="$RESOLVED_URL"
ASSET="lumen-$PLAT"
if [ "$RESOLVED_TAG" = "latest" ]; then
  field "Release" "latest"
else
  field "Release" "$RESOLVED_TAG"
fi

# ── Step 2 · download + verify checksum (abort installs nothing on mismatch) ──
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
curl -fsSL "$URL" -o "$TMP/pkg.tar.gz" || die "download failed: $URL"
# Human-readable size of the downloaded asset (best-effort; blank if stat differs).
DL_SIZE=""
if bytes="$(wc -c < "$TMP/pkg.tar.gz" 2>/dev/null)" && [ -n "$bytes" ]; then
  DL_SIZE="$(awk -v b="$bytes" 'BEGIN{ if (b>=1048576) printf "%.1f MB", b/1048576; else printf "%.0f KB", b/1024 }')"
fi
VERIFIED=""
if curl -fsSL "$URL.sha256" -o "$TMP/pkg.sha256" 2>/dev/null; then
  want="$(awk '{print $1}' "$TMP/pkg.sha256")"; got="$(sha256_of "$TMP/pkg.tar.gz")"
  [ "$want" = "$got" ] || die "checksum mismatch (want $want, got $got) — installing nothing."
  VERIFIED="$OK_MARK verified"
elif [ "${LUMEN_INSECURE_SKIP_CHECKSUM:-0}" = "1" ]; then
  VERIFIED="${C_YELLOW}unverified${C_RESET} (LUMEN_INSECURE_SKIP_CHECKSUM=1)"
  info "WARNING: no .sha256 published; skipping verification."
else
  die "no checksum (.sha256) for $URL; refusing. Set LUMEN_INSECURE_SKIP_CHECKSUM=1 to override."
fi
# Download   lumen-macos-arm64-metal · 6.6 MB · ✓ verified
if [ -n "$DL_SIZE" ]; then
  field "Download" "$ASSET ${C_DIM}·${C_RESET} $DL_SIZE ${C_DIM}·${C_RESET} $VERIFIED"
else
  field "Download" "$ASSET ${C_DIM}·${C_RESET} $VERIFIED"
fi

# ── Step 3 · install both binaries ────────────────────────────────────────────
# Robust, cross-platform location logic (macOS Metal + Linux CUDA). The goal:
# pick a directory that is already on $PATH and writable WITHOUT sudo so the
# common Linux + Homebrew-Mac case needs no password and no profile edit. Only
# when none qualifies do we fall back to sudo /usr/local/bin or to ~/.local/bin
# with a one-line PATH addition.
tar -C "$TMP" -xzf "$TMP/pkg.tar.gz"
SRC="$(find "$TMP" -maxdepth 1 -type d -name "lumen-*-$PLAT" | head -1)"
[ -n "$SRC" ] && [ -d "$SRC/bin" ] || die "unexpected tarball layout under $TMP"

# True if $1 is a literal entry on $PATH.
on_path() { case ":$PATH:" in *":$1:"*) return 0 ;; *) return 1 ;; esac; }
# True if $1 can be written WITHOUT sudo: it exists and is writable, OR it does
# not exist but the nearest existing ancestor is writable (so `mkdir -p` would
# succeed without sudo). Walks up the same way `mkdir -p` does.
writable_nosudo() {
  local d="$1"
  if [ -d "$d" ]; then [ -w "$d" ] && return 0; return 1; fi
  # Climb to the first existing ancestor; if it's writable, mkdir -p succeeds.
  local p="$d" parent
  while :; do
    parent="$(dirname -- "$p")"
    [ "$parent" = "$p" ] && return 1   # reached / without finding one
    if [ -e "$parent" ]; then
      [ -d "$parent" ] && [ -w "$parent" ] && return 0
      return 1
    fi
    p="$parent"
  done
}
# Copy both binaries into $1 using the command prefix $2 ("" or "sudo").
# stderr is suppressed so a no-sudo probe failure is SILENT — the caller decides
# whether the failure is fatal and prints a clean message if so.
install_to() {
  local d="$1" S="$2" b
  $S mkdir -p "$d" 2>/dev/null || return 1
  for b in lumen lumen-server; do
    $S install -m 0755 "$SRC/bin/$b" "$d/$b" 2>/dev/null || return 1
  done
  return 0
}

DEST=""
PROFILE_EDITED=""   # set to the profile path if we appended a PATH line (branch 4)

if [ -n "$PREFIX" ]; then
  # (1) Explicit --prefix / LUMEN_PREFIX → honor it. No auto-selection.
  if writable_nosudo "$PREFIX" && install_to "$PREFIX" ""; then
    DEST="$PREFIX"
  elif have sudo; then
    info "Admin access is needed to write $PREFIX — you'll be asked for your password."
    install_to "$PREFIX" "sudo" && DEST="$PREFIX"
    [ -n "$DEST" ] || die "could not install to $PREFIX (even with sudo). Pick a writable --prefix, e.g. --prefix \"\$HOME/.local/bin\"."
  else
    die "cannot write $PREFIX and 'sudo' is unavailable. Pick a writable --prefix, e.g. --prefix \"\$HOME/.local/bin\"."
  fi
else
  # (2) Auto-select: FIRST dir that is BOTH on $PATH AND writable without sudo.
  # Preference order: ~/.local/bin, /opt/homebrew/bin, /usr/local/bin, then any
  # other writable $PATH entry. (No sudo, no profile edit in this branch.)
  CANDIDATES="$HOME/.local/bin /opt/homebrew/bin /usr/local/bin"
  # Append remaining $PATH entries (split on ':') not already named above.
  _oldifs="$IFS"; IFS=':'
  for p in $PATH; do
    [ -n "$p" ] || continue
    case " $CANDIDATES " in *" $p "*) ;; *) CANDIDATES="$CANDIDATES $p" ;; esac
  done
  IFS="$_oldifs"
  for d in $CANDIDATES; do
    if on_path "$d" && writable_nosudo "$d" && install_to "$d" ""; then
      DEST="$d"; break
    fi
  done

  if [ -z "$DEST" ]; then
    # (3) Nothing on-PATH-and-writable. If /usr/local/bin is on $PATH and sudo is
    # available, install there with sudo (after a clear heads-up).
    if on_path "/usr/local/bin" && have sudo; then
      info "Admin access is needed to write /usr/local/bin — you'll be asked for your password."
      install_to "/usr/local/bin" "sudo" && DEST="/usr/local/bin"
    fi
  fi

  if [ -z "$DEST" ]; then
    # (4) Last resort: install to ~/.local/bin (create it) and put it on PATH by
    # appending a single idempotent line to the user's shell profile.
    DEST="$HOME/.local/bin"
    install_to "$DEST" "" || die "could not install to $DEST."
    if ! on_path "$DEST"; then
      # Pick the profile by login shell; cover bash's two files.
      shell_name="$(basename -- "${SHELL:-}")"
      profiles=""
      case "$shell_name" in
        zsh)  profiles="$HOME/.zshrc" ;;
        bash) profiles="$HOME/.bashrc $HOME/.bash_profile" ;;
        *)    profiles="$HOME/.profile" ;;
      esac
      export_line="export PATH=\"\$HOME/.local/bin:\$PATH\""
      for prof in $profiles; do
        # Idempotent: skip if a line already puts ~/.local/bin on PATH — match
        # BOTH the unexpanded `$HOME/.local/bin` literal we write (so a second
        # run is a no-op) AND an already-expanded absolute path a user may have.
        # shellcheck disable=SC2016  # the '$HOME' literal is intentional: that
        # is the exact text we append, so the single-quoted pattern matches it.
        if [ -f "$prof" ] \
           && { grep -Fq '$HOME/.local/bin' "$prof" 2>/dev/null \
                || grep -Fq "$HOME/.local/bin" "$prof" 2>/dev/null; }; then
          continue
        fi
        printf '\n# Added by Lumen installer\n%s\n' "$export_line" >> "$prof" \
          && PROFILE_EDITED="${PROFILE_EDITED:+$PROFILE_EDITED }$prof"
      done
    fi
  fi
fi

# Strip macOS quarantine so Gatekeeper doesn't block the unsigned binaries.
if [ "$BACKEND" = "metal" ]; then
  xattr -dr com.apple.quarantine "$DEST/lumen" "$DEST/lumen-server" 2>/dev/null || true
fi

LUMEN="$DEST/lumen"
VER="$("$LUMEN" --version 2>/dev/null | awk '{print $NF}' || echo '?')"
field "Installed" "lumen + lumen-server → $DEST ${C_DIM}·${C_RESET} $VER"

if [ -n "$PROFILE_EDITED" ]; then
  info "added $DEST to PATH in: $PROFILE_EDITED"
  info "for this shell, run:  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# Shadow check: if a DIFFERENT `lumen` is found earlier on PATH, the new one is
# masked. Make this an actionable warning, not a passive note.
r="$(command -v lumen 2>/dev/null || true)"
if on_path "$DEST" && [ -n "$r" ] && [ "$r" != "$DEST/lumen" ]; then
  err "another 'lumen' shadows the one just installed:"
  err "    on PATH:   $r"
  err "    installed: $DEST/lumen"
  err "  fix it:  rm \"$r\"   (or put $DEST earlier on PATH)"
fi

# ── Step 4 · pick a model + quant (interactive via /dev/tty; safe under pipe) ──
INTERACTIVE=0
if [ -z "$MODEL" ] && [ "$ASSUME_YES" != "1" ] && [ -r /dev/tty ]; then INTERACTIVE=1; fi
if [ "$INTERACTIVE" = "1" ]; then
  say ""
  say "  ${C_BOLD}Choose a model${C_RESET}"
  say "    1  Qwen3.5 9B    ${C_DIM}dense · ~10 GB @ Q8${C_RESET}        ${C_GREEN}(recommended)${C_RESET}"
  say "    2  Qwen3.5 MoE   ${C_DIM}30B-A3B · mixture-of-experts${C_RESET}"
  say "    3  Qwen3.6 27B   ${C_DIM}dense · largest${C_RESET}"
  printf '  %s›%s ' "$C_CYAN" "$C_RESET"
  read -r pick < /dev/tty || pick=""
  case "$pick" in 2) MODEL="$MOE_CANONICAL" ;; 3) MODEL="qwen3.6-27b" ;; *) MODEL="$DEFAULT_MODEL" ;; esac
  if [ -z "$QUANT" ]; then   # honor an explicit --quant; only prompt if unset
    say ""
    say "  ${C_BOLD}Choose a quant${C_RESET}"
    say "    1  Q8    ${C_DIM}best quality/size${C_RESET}                  ${C_GREEN}(recommended)${C_RESET}"
    say "    2  Q4    ${C_DIM}smaller · faster${C_RESET}"
    say "    3  BF16  ${C_DIM}full precision · large${C_RESET}"
    printf '  %s›%s ' "$C_CYAN" "$C_RESET"
    read -r qpick < /dev/tty || qpick=""
    case "$qpick" in 2) QUANT="q4_0" ;; 3) QUANT="bf16" ;; *) QUANT="$DEFAULT_QUANT" ;; esac
  fi
else
  MODEL="${MODEL:-$DEFAULT_MODEL}"; QUANT="${QUANT:-$DEFAULT_QUANT}"
  [ "$ASSUME_YES" = "1" ] || info "no terminal: using default $MODEL:$QUANT (pass --model/--quant to choose)"
fi
# Canonicalize any MoE alias so pull, run AND lumen-server all agree on the cache stem.
case "$MODEL" in
  qwen3.5-moe|qwen3-5-moe|qwen3.5-moe-35b-a3b|qwen3-5-moe-35b-a3b) MODEL="$MOE_CANONICAL" ;;
esac

# ── Step 5 · prepare the model (installer owns consent -> pull --yes) ─────────
cache="${LUMEN_CACHE_DIR:-}"
if [ -z "$cache" ]; then
  case "$BACKEND" in metal) cache="$HOME/Library/Caches/lumen" ;; *) cache="${XDG_CACHE_HOME:-$HOME/.cache}/lumen" ;; esac
fi
mkdir -p "$cache" 2>/dev/null || true
# Rough peak-disk guard (GGUF + converted LBC coexist during convert; not a catalog).
need=12
case "$QUANT" in
  bf16) case "$MODEL" in *moe*|*27b*) need=150 ;; *) need=40 ;; esac ;;
  q8_0) case "$MODEL" in *moe*) need=85 ;; *27b*) need=70 ;; *) need=24 ;; esac ;;
  *)    case "$MODEL" in *moe*) need=48 ;; *27b*) need=40 ;; *) need=14 ;; esac ;;
esac
free_gb="$(df -Pk "$cache" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}')"
if [ -n "${free_gb:-}" ] && [ "$free_gb" -lt "$need" ]; then
  info "WARNING: ~${free_gb} GB free at $cache; $MODEL:$QUANT needs roughly ${need} GB peak (source GGUF + converted LBC coexist; downloads are NOT resumable)."
  if [ "$INTERACTIVE" = "1" ]; then
    printf '  Continue anyway? [y/N] '
    read -r go < /dev/tty || go=""
    case "$go" in y|Y|yes) ;; *) die "aborted (low disk)." ;; esac
  fi
fi
say ""
info "preparing $MODEL:$QUANT (download + convert; one-time)"
# Strip the installer's own LUMEN_* input vars (model/quant/prefix/tag/release-base/
# insecure flags) from the child env — they configure the installer, not lumen, so
# lumen's env-var typo validator would otherwise warn about each one. LUMEN_CACHE_DIR
# IS a real lumen var and is passed through explicitly.
env -u LUMEN_MODEL -u LUMEN_QUANT -u LUMEN_PREFIX -u LUMEN_TAG -u LUMEN_RELEASE_BASE \
    -u LUMEN_ALLOW_INSECURE_BASE -u LUMEN_INSECURE_SKIP_CHECKSUM \
    LUMEN_CACHE_DIR="$cache" "$LUMEN" pull "$MODEL:$QUANT" --yes \
  || die "model prepare failed for $MODEL:$QUANT. The binaries are installed at $DEST — re-run this installer (it restarts cleanly), or run: $DEST/lumen pull $MODEL:$QUANT --yes"

# ── Step 6 · print the exact next steps (positional model:quant forms) ────────
say ""
say "  $OK_MARK  ${C_BOLD}$MODEL:$QUANT${C_RESET} ready (cached)"
say ""
say "  ${C_BOLD}Chat${C_RESET}     lumen run $MODEL:$QUANT \"Write a haiku about Rust\""
say "  ${C_BOLD}Serve${C_RESET}    lumen-server $MODEL:$QUANT          ${C_DIM}(OpenAI/Anthropic API · :8000)${C_RESET}"
say ""
# PATH reminder only if $DEST isn't already on PATH (e.g. fresh ~/.local/bin).
if ! on_path "$DEST"; then
  say "  ${C_DIM}First, put it on PATH:${C_RESET}  export PATH=\"$DEST:\$PATH\""
  say ""
fi
