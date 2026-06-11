#!/usr/bin/env bash
# =============================================================================
#  Lumen quickstart  —  clone → build → model → serve, in one script.
# =============================================================================
#  Walks a brand-new user from a fresh checkout to a running, OpenAI-compatible
#  inference server:
#
#    1. Detect the GPU backend   (macOS → Metal · Linux+NVIDIA → CUDA · else CPU)
#    2. Preflight prerequisites  (rustc/cargo/curl/git, disk, RAM/VRAM, driver)
#    3. Build the CLI + server   (correct Cargo features for the backend)
#    4. Pick + pull a model      (download GGUF → convert to .lbc → cache; idempotent)
#    5. Start lumen-server       (and print copy-paste curl + `lumen run` examples)
#    6. Clean up on exit         (stop the server, remove this run's partials)
#
#  Runnable from the repo root:  ./scripts/quickstart.sh
#
#  Ground truth: every command/flag/model-id/path here is verified against the
#  Lumen source (crates/lumen-cli, crates/lumen-server, model_registry.toml).
#  Built and tested against macOS bash 3.2 — no bash-4 features are used.
# =============================================================================
#  Non-interactive / CI usage (flags override env override defaults):
#
#    ./scripts/quickstart.sh --yes                       # accept all defaults
#    ./scripts/quickstart.sh --model qwen3.5-9b --quant q8_0 --port 8000 --yes
#    ./scripts/quickstart.sh --build-only --yes          # build, then stop
#    ./scripts/quickstart.sh --dry-run                   # print the plan, do nothing
#    ./scripts/quickstart.sh --backend cpu --yes         # force a backend
#    ./scripts/quickstart.sh --no-serve --yes            # pull + build, don't serve
#
#  Equivalent env vars: LUMEN_QS_MODEL, LUMEN_QS_QUANT, LUMEN_QS_BACKEND,
#  LUMEN_QS_PORT, LUMEN_QS_HOST, LUMEN_QS_YES=1, LUMEN_QS_VERBOSE=1.
#
#  Run  ./scripts/quickstart.sh --help  for the full option list.
# =============================================================================

set -euo pipefail

# Deterministic, IFS-safe word splitting. Newline-only IFS so paths with spaces
# survive; we re-localize IFS in the few spots that need other separators.
IFS=$'\n\t'

# -----------------------------------------------------------------------------
#  Constants  (single source of truth — change here, not inline)
# -----------------------------------------------------------------------------

readonly QS_VERSION="1.0.0"

# Default server endpoint. 127.0.0.1 (loopback only) — a quickstart must never
# bind a model server to a public interface by default.
readonly DEFAULT_HOST="127.0.0.1"
readonly DEFAULT_PORT="8000"

# Default model selection (registry alias + quant tag, lowercase as the CLI
# accepts on the command line).
readonly DEFAULT_MODEL="qwen3.5-9b"
readonly DEFAULT_QUANT="q8_0"

# Readiness probe budget. The server cold-loads weights (GPU-resident upload)
# before /v1/models answers; for a cached Q8 9B this is well under a minute,
# but MoE/BF16 can take longer, so we poll up to this many seconds.
readonly READY_TIMEOUT_SECS="600"
readonly READY_POLL_INTERVAL_SECS="2"

# -----------------------------------------------------------------------------
#  Globals  (assigned during the run; declared up-front for clarity)
# -----------------------------------------------------------------------------

# Resolved from repo layout / detection.
REPO_ROOT=""
BACKEND=""              # metal | cuda | cpu
LUMEN_BIN=""            # path to the built `lumen` binary
SERVER_BIN=""           # path to the built `lumen-server` binary
CACHE_DIR=""            # resolved Lumen cache directory

# Runtime selection (filled by config + interactive menu).
SEL_MODEL=""            # registry alias (e.g. qwen3.5-9b) OR a direct .lbc path
SEL_QUANT=""            # quant tag (e.g. q8_0); empty when SEL_IS_PATH=1
SEL_SPEC=""             # display/spec form: "<model>:<quant>" or the path
SEL_IS_PATH="0"         # "1" when --model is a direct .lbc/.gguf path

# Server bookkeeping (used by the EXIT trap).
SERVER_PID=""
SERVER_LOG=""
SERVER_STARTED="0"

# Per-run scratch (mktemp dir; removed on exit). Also tracks GGUF .part files
# this run created so cleanup never touches a user's pre-existing cache.
SCRATCH_DIR=""
PARTFILES_BEFORE=""     # newline list of *.part present BEFORE we pulled

# CLI flags / config (env defaults applied in configure()).
OPT_MODEL="${LUMEN_QS_MODEL:-}"
OPT_QUANT="${LUMEN_QS_QUANT:-}"
OPT_BACKEND="${LUMEN_QS_BACKEND:-}"
OPT_HOST="${LUMEN_QS_HOST:-$DEFAULT_HOST}"
OPT_PORT="${LUMEN_QS_PORT:-$DEFAULT_PORT}"
OPT_YES="${LUMEN_QS_YES:-0}"
OPT_VERBOSE="${LUMEN_QS_VERBOSE:-0}"
OPT_DRY_RUN="0"
OPT_BUILD_ONLY="0"
OPT_NO_SERVE="0"
OPT_SKIP_BUILD="0"

# -----------------------------------------------------------------------------
#  Colors  (only when stderr is a TTY; NO_COLOR honored)
# -----------------------------------------------------------------------------

C_RESET="" ; C_DIM="" ; C_RED="" ; C_GRN="" ; C_YLW="" ; C_BLU="" ; C_BLD=""
if [ -t 2 ] && [ -z "${NO_COLOR:-}" ]; then
  C_RESET=$'\033[0m' ; C_DIM=$'\033[2m'   ; C_RED=$'\033[31m'
  C_GRN=$'\033[32m'  ; C_YLW=$'\033[33m'  ; C_BLU=$'\033[34m'
  C_BLD=$'\033[1m'
fi

# -----------------------------------------------------------------------------
#  Logging  (leveled, timestamped, to stderr — stdout stays clean)
# -----------------------------------------------------------------------------

_ts() { date "+%H:%M:%S"; }

log_info()  { printf '%s %s[info]%s  %s\n'  "$(_ts)" "$C_BLU" "$C_RESET" "$*" >&2; }
log_warn()  { printf '%s %s[warn]%s  %s\n'  "$(_ts)" "$C_YLW" "$C_RESET" "$*" >&2; }
log_error() { printf '%s %s[error]%s %s\n'  "$(_ts)" "$C_RED" "$C_RESET" "$*" >&2; }
log_ok()    { printf '%s %s[ok]%s    %s\n'  "$(_ts)" "$C_GRN" "$C_RESET" "$*" >&2; }
log_step()  { printf '\n%s%s==>%s %s%s\n'   "$C_BLD" "$C_BLU" "$C_RESET" "$*" "$C_RESET" >&2; }
log_debug() {
  if [ "$OPT_VERBOSE" = "1" ]; then
    printf '%s %s[debug]%s %s\n' "$(_ts)" "$C_DIM" "$C_RESET" "$*" >&2
  fi
}

# Fatal: log + exit non-zero. The EXIT trap handles cleanup.
die() {
  log_error "$*"
  exit 1
}

# -----------------------------------------------------------------------------
#  Command runner  (honors --dry-run and --verbose uniformly)
# -----------------------------------------------------------------------------
#  run_cmd "<human description>" cmd arg arg ...
#  In --dry-run we print the exact argv (safely quoted) and skip execution.
# -----------------------------------------------------------------------------

# Quote a single argv element for safe display/copy-paste.
_shq() {
  # Wrap in single quotes, escaping embedded single quotes.
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

_show_argv() {
  local out="" a
  for a in "$@"; do
    out="$out $(_shq "$a")"
  done
  # Trim the leading space.
  printf '%s' "${out# }"
}

run_cmd() {
  local desc="$1" ; shift
  if [ "$OPT_DRY_RUN" = "1" ]; then
    printf '%s   %s[dry-run]%s %s\n' "$(_ts)" "$C_DIM" "$C_RESET" "$desc" >&2
    printf '             %s$ %s%s\n' "$C_DIM" "$(_show_argv "$@")" "$C_RESET" >&2
    return 0
  fi
  log_debug "exec: $(_show_argv "$@")"
  "$@"
}

# -----------------------------------------------------------------------------
#  Help
# -----------------------------------------------------------------------------

print_help() {
  # Plain text on stdout (so `--help | less` works); no color codes.
  cat <<EOF
lumen quickstart v${QS_VERSION} — from a fresh checkout to a running server.

USAGE:
    ./scripts/quickstart.sh [OPTIONS]

WHAT IT DOES:
    Detects your GPU backend, checks prerequisites, builds the Lumen CLI and
    server, lets you pick + download a model, starts lumen-server, and prints
    copy-paste commands to run inference. Cleans up on Ctrl-C / exit.

OPTIONS:
    --model <name>     Registry model alias (default: ${DEFAULT_MODEL}).
                       e.g. qwen3.5-9b, qwen3.5-moe-35b-a3b
    --quant <q>        Quantization tag (default: ${DEFAULT_QUANT}).
                       e.g. q8_0, q4_0, bf16
    --backend <b>      Force backend: metal | cuda | cpu (default: auto-detect).
    --host <h>         Server bind host (default: ${DEFAULT_HOST}).
    --port <n>         Server port (default: ${DEFAULT_PORT}).
    --yes, -y          Non-interactive: accept defaults, skip all prompts.
    --dry-run          Print the plan and the exact commands; execute nothing.
    --build-only       Detect + preflight + build the binaries, then stop.
    --no-serve         Do everything up to and including the model pull; do not
                       start the server (prints how to start it yourself).
    --skip-build       Use already-built target/release binaries; skip cargo.
    --verbose, -v      Verbose / debug logging.
    --help, -h         Show this help and exit.
    --version, -V      Print the quickstart script version and exit.

ENVIRONMENT (flags take precedence):
    LUMEN_QS_MODEL, LUMEN_QS_QUANT, LUMEN_QS_BACKEND, LUMEN_QS_HOST,
    LUMEN_QS_PORT, LUMEN_QS_YES=1, LUMEN_QS_VERBOSE=1
    LUMEN_CACHE_DIR    Override the Lumen model cache directory.

EXAMPLES:
    ./scripts/quickstart.sh
    ./scripts/quickstart.sh --yes
    ./scripts/quickstart.sh --model qwen3.5-9b --quant q8_0 --port 9000 --yes
    ./scripts/quickstart.sh --dry-run
    ./scripts/quickstart.sh --build-only --yes

MODELS (registry source of truth: model_registry.toml):
    qwen3.5-9b            Q8_0 (~10 GB), Q4_0 (~5.4 GB), BF16 (~18 GB) all
                         downloadable.
    qwen3.5-moe-35b-a3b  Q8_0 (~37 GB), Q4_0 (~19 GB), BF16 (~71 GB 2-shard)
                         all downloadable.
    qwen3.6-27b          Q8_0 (~29 GB), Q4_0 (~15 GB), BF16 (~55 GB 2-shard)
                         all downloadable.

    Quickstart only auto-pulls the combos above; any other (model, quant) is
    refused with guidance (prepare it yourself, then re-run --model <path.lbc>).
    A pull holds the source GGUF and the converted LBC on disk at the same time,
    so the disk check needs roughly (GGUF + LBC), e.g. ~19 GiB for 9B Q8_0 or
    ~71 GiB for MoE Q8_0 — not just the final file size.

This script never commits, never modifies tracked source, and binds the server
to ${DEFAULT_HOST} by default.
EOF
}

# =============================================================================
#  (sections continue below — argument parsing, detection, preflight, build,
#   model selection, serve, cleanup, main)
# =============================================================================

# -----------------------------------------------------------------------------
#  Argument parsing
# -----------------------------------------------------------------------------

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --model)    [ $# -ge 2 ] || die "--model requires a value";   OPT_MODEL="$2";   shift 2 ;;
      --model=*)  OPT_MODEL="${1#*=}";   shift ;;
      --quant)    [ $# -ge 2 ] || die "--quant requires a value";   OPT_QUANT="$2";   shift 2 ;;
      --quant=*)  OPT_QUANT="${1#*=}";   shift ;;
      --backend)  [ $# -ge 2 ] || die "--backend requires a value"; OPT_BACKEND="$2"; shift 2 ;;
      --backend=*) OPT_BACKEND="${1#*=}"; shift ;;
      --host)     [ $# -ge 2 ] || die "--host requires a value";    OPT_HOST="$2";    shift 2 ;;
      --host=*)   OPT_HOST="${1#*=}";    shift ;;
      --port)     [ $# -ge 2 ] || die "--port requires a value";    OPT_PORT="$2";    shift 2 ;;
      --port=*)   OPT_PORT="${1#*=}";    shift ;;
      --yes|-y)        OPT_YES="1";       shift ;;
      --dry-run)       OPT_DRY_RUN="1";   shift ;;
      --build-only)    OPT_BUILD_ONLY="1"; shift ;;
      --no-serve)      OPT_NO_SERVE="1";  shift ;;
      --skip-build)    OPT_SKIP_BUILD="1"; shift ;;
      --verbose|-v)    OPT_VERBOSE="1";   shift ;;
      --help|-h)       print_help; exit 0 ;;
      --version|-V)    printf 'lumen-quickstart %s\n' "$QS_VERSION"; exit 0 ;;
      --) shift; break ;;
      -*) die "unknown option: $1 (try --help)" ;;
      *)  die "unexpected argument: $1 (try --help)" ;;
    esac
  done
}

# Validate parsed options up-front so we fail fast with a clear message.
validate_args() {
  # Port: integer in 1..65535.
  case "$OPT_PORT" in
    ''|*[!0-9]*) die "--port must be a positive integer, got: '$OPT_PORT'" ;;
  esac
  if [ "$OPT_PORT" -lt 1 ] || [ "$OPT_PORT" -gt 65535 ]; then
    die "--port must be in 1..65535, got: $OPT_PORT"
  fi

  # Backend: restricted vocabulary (auto/empty means detect).
  if [ -n "$OPT_BACKEND" ]; then
    case "$(to_lower "$OPT_BACKEND")" in
      metal|cuda|cpu|auto) : ;;
      *) die "--backend must be metal | cuda | cpu (got: '$OPT_BACKEND')" ;;
    esac
  fi

  # Host: reject obvious shell-meta / whitespace injection. A hostname/IP has
  # no business containing spaces, quotes, $, ;, |, &, backticks, etc.
  if printf '%s' "$OPT_HOST" | LC_ALL=C grep -q '[^A-Za-z0-9._:-]'; then
    die "--host contains invalid characters: '$OPT_HOST'"
  fi
}

# -----------------------------------------------------------------------------
#  Small utilities
# -----------------------------------------------------------------------------

# Lowercase without bash-4 ${var,,} (BSD/bash-3.2 safe).
to_lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }
to_upper() { printf '%s' "$1" | tr '[:lower:]' '[:upper:]'; }

# True if a command exists on PATH.
have() { command -v "$1" >/dev/null 2>&1; }

# True (0) if $1 appears as an EXACT line within the newline-separated list $2.
# Used by cleanup to decide whether a partial-download path pre-existed this run
# (exact match, never substring — so one path being a prefix/substring of
# another can never mis-classify).
line_in_list() {
  local needle="$1" haystack="$2" line oldifs="$IFS"
  IFS=$'\n'
  for line in $haystack; do
    if [ "$line" = "$needle" ]; then
      IFS="$oldifs"
      return 0
    fi
  done
  IFS="$oldifs"
  return 1
}

# Prompt yes/no with a default. Honors --yes (always yes) and non-TTY (default).
# Usage: confirm "Question?" "Y"   -> returns 0 for yes, 1 for no.
confirm() {
  local q="$1" def="${2:-Y}" ans=""
  if [ "$OPT_YES" = "1" ]; then
    # --yes means "accept the documented DEFAULT", not "say yes to everything".
    # Default-Y prompts proceed; default-N safety prompts (e.g. "Continue anyway?"
    # on insufficient disk) take the safe abort -- otherwise a non-interactive /CI
    # run would march past a resource gate into a doomed multi-GB download.
    case "$def" in Y|y) return 0 ;; *) return 1 ;; esac
  fi
  if [ ! -t 0 ]; then
    # No interactive stdin: fall back to the default, don't hang.
    case "$def" in Y|y) return 0 ;; *) return 1 ;; esac
  fi
  local hint="[Y/n]"
  case "$def" in N|n) hint="[y/N]" ;; esac
  printf '%s %s ' "$q" "$hint" >&2
  IFS= read -r ans || ans=""
  ans="$(to_lower "$ans")"
  if [ -z "$ans" ]; then
    case "$def" in Y|y) return 0 ;; *) return 1 ;; esac
  fi
  case "$ans" in y|yes) return 0 ;; *) return 1 ;; esac
}

# -----------------------------------------------------------------------------
#  Model catalog  (single source of truth for the menu)
# -----------------------------------------------------------------------------
#  Mirrors model_registry.toml. bash 3.2 has no associative arrays, so we use
#  parallel indexed arrays keyed by the same index. Each entry records whether
#  the (model, quant) pair is DOWNLOADABLE from the registry — pairs that are
#  not downloadable would be flagged here (none currently are — every combo
#  is a direct registry pull). Nested-subdir
#  2-shard BF16 splits are downloadable (download_gguf validates each path
#  segment and caches shards flat + adjacent).
#
#  IMPORTANT — two DIFFERENT physical quantities, never conflate them:
#    * CAT_NEED  = peak RAM/VRAM RESIDENCY when serving (≈ LBC size + overhead).
#                  Compared against system RAM in preflight().
#    * CAT_DISK  = peak TRANSIENT DISK during pull: the source GGUF AND the
#                  converted LBC coexist on disk while `lumen convert` runs
#                  (the GGUF is not deleted until conversion completes). So the
#                  true peak ≈ source_GGUF_size + output_LBC_size — NOT just one
#                  of them, and NOT a "2× the residency" guess. Compared against
#                  free disk in preflight(). Every value here is rounded UP from
#                  the real measured on-disk sizes (an under-estimate is a
#                  resource-safety failure; a slight over-estimate is safe).
#                  Each value is derived from the real measured on-disk sizes
#                  per (model, quant) combination.
#
#  Fields (one entry per array slot):
#    CAT_SPEC   "<model>:<quant>"            user-facing spec
#    CAT_SIZE   on-disk LBC footprint        human string
#    CAT_NEED   peak RAM/VRAM residency      human string
#    CAT_DISK   peak transient disk on pull  human string (GGUF + LBC)
#    CAT_DESC   one-line description
#    CAT_PULL   "1" downloadable | "0" not pullable (local-convert / deferred)
# -----------------------------------------------------------------------------

CAT_SPEC=() ; CAT_SIZE=() ; CAT_NEED=() ; CAT_DISK=() ; CAT_DESC=() ; CAT_PULL=()

_catalog_add() {
  CAT_SPEC+=("$1") ; CAT_SIZE+=("$2") ; CAT_NEED+=("$3")
  CAT_DISK+=("$4") ; CAT_DESC+=("$5") ; CAT_PULL+=("$6")
}

build_catalog() {
  # Measured on-disk sizes (this M3 Ultra, bytes/1024^3 = GiB) drive both the
  # LBC footprint and the transient-disk peak. Peaks rounded UP for safety:
  #   combo               src GGUF   LBC out   transient peak (CAT_DISK)
  #   9B  q8_0             8.89 GiB   9.98 GiB  ~19 GiB  (download + convert)
  #   9B  q4_0             5.35 GiB   5.37 GiB  ~12 GiB  (direct download)
  #   9B  bf16            17.14 GiB  ~17 GiB    ~35 GiB  (single flat GGUF)
  #   MoE q4_0           19.41 GiB  19.31 GiB  ~39 GiB
  #   MoE q8_0           35.22 GiB  35.00 GiB  ~71 GiB
  #   MoE bf16  (2-shard) 66.2 GiB  ~66 GiB   ~140 GiB
  #   27B q8_0            27.1 GiB  ~27 GiB    ~56 GiB
  #   27B q4_0            13.7 GiB  ~14 GiB    ~29 GiB
  #   27B bf16  (2-shard) 50.9 GiB  ~51 GiB   ~105 GiB
  #
  # Dense Qwen3.5-9B. All three quants downloadable (BF16 is the provider's
  # true-bf16 single-file GGUF; Q4_0 is the provider's direct Q4_0 — never
  # derive it via --requant for the generic/CUDA target, the requant LBC is
  # broken there: 2026-06-10 isolation, requant 1/15 vs direct 13/15).
  _catalog_add "qwen3.5-9b:q8_0" "~10 GB" "~11 GB" "~19 GB" "Dense 9B, production default (best quality)" "1"
  _catalog_add "qwen3.5-9b:q4_0" "~5.4 GB" "~6 GB" "~12 GB" "Dense 9B, 4-bit (smaller)" "1"
  _catalog_add "qwen3.5-9b:bf16" "~18 GB" "~19 GB" "~35 GB" "Dense 9B, full precision" "1"
  # MoE 30B-A3B (3B active). All three quants downloadable. CAT_DISK is the
  # true transient peak (source GGUF + converted LBC held simultaneously).
  # BF16 is a 2-shard split GGUF nested in a HF subdirectory; the downloader
  # fetches each shard from its nested URL and caches them flat + adjacent.
  _catalog_add "qwen3.5-moe-35b-a3b:q4_0" "~19 GB" "~21 GB" "~39 GB" "MoE 30B (3B active), 4-bit" "1"
  _catalog_add "qwen3.5-moe-35b-a3b:q8_0" "~37 GB" "~38 GB" "~71 GB" "MoE 30B (3B active), 8-bit" "1"
  _catalog_add "qwen3.5-moe-35b-a3b:bf16" "~71 GB" "~72 GB" "~140 GB" "MoE 30B (3B active), full precision" "1"
  # Dense Qwen3.6-27B (GDN ratio-3). All three quants downloadable; BF16 is a
  # 2-shard split GGUF (same nested HF-subdir layout as the MoE BF16).
  _catalog_add "qwen3.6-27b:q8_0" "~29 GB" "~30 GB" "~56 GB" "Dense 27B, 8-bit (best quality)" "1"
  _catalog_add "qwen3.6-27b:q4_0" "~15 GB" "~16 GB" "~29 GB" "Dense 27B, 4-bit (smaller)" "1"
  _catalog_add "qwen3.6-27b:bf16" "~55 GB" "~56 GB" "~105 GB" "Dense 27B, full precision" "1"
}

# Find the catalog index for a "<model>:<quant>" spec. Echoes index or "-1".
catalog_index_of() {
  local want="$1" i=0
  while [ "$i" -lt "${#CAT_SPEC[@]}" ]; do
    if [ "${CAT_SPEC[$i]}" = "$want" ]; then
      printf '%s' "$i" ; return 0
    fi
    i=$((i + 1))
  done
  printf '%s' "-1"
}

# Print the supported quickstart combos to stderr, one per line, annotated with
# pullable vs local-convert. Used by the refusal messages so a user who picks an
# unsupported (model, quant) is told exactly what IS supported. `build_catalog`
# must have run first.
print_supported_combos() {
  local i=0 tag
  while [ "$i" -lt "${#CAT_SPEC[@]}" ]; do
    if [ "${CAT_PULL[$i]}" = "1" ]; then
      tag="downloadable"
    else
      tag="local convert only"
    fi
    log_error "    ${CAT_SPEC[$i]}   (${CAT_SIZE[$i]} on disk · ${tag})"
    i=$((i + 1))
  done
}

# Known model names: canonical keys + aliases from model_registry.toml
# ([models.*] keys and the [aliases] table). Used to reject a clearly-invalid
# model name up-front — including under --dry-run, where the CLI never runs to
# give the authoritative error. Quant availability is still left to the CLI
# (it owns the registry); this only guards the model name itself.
#
# A value that looks like a file path (contains '/' or '\', or ends in
# .lbc/.gguf) is NOT validated here — the CLI/server accept direct paths, and
# we let them report a missing file.
# Newline-delimited (the global IFS is $'\n\t', so a space-delimited list would
# not word-split correctly — keep this newline-separated and iterate IFS-safely).
KNOWN_MODELS="qwen3-5-9b
qwen3.5-9b
qwen3-5-moe-35b-a3b
qwen3.5-moe-35b-a3b
qwen3.5-moe"

model_looks_like_path() {
  case "$1" in
    */*|*\\*|*.lbc|*.gguf) return 0 ;;
    *) return 1 ;;
  esac
}

# True (0) if the given model name is a known registry alias/key.
model_name_known() {
  local name="$1"
  line_in_list "$name" "$KNOWN_MODELS"
}

# Canonicalize an accepted model name to the spelling the catalog uses (the
# dotted display form), so the catalog lookup matches no matter WHICH valid
# spelling the user typed. The CLI/registry accept several spellings for the
# same model — the dashed canonical TOML keys (qwen3-5-9b), the dotted display
# aliases (qwen3.5-9b), and the short alias (qwen3.5-moe). The catalog is keyed
# on the dotted display form; without this, a perfectly pullable combo typed via
# a different alias (e.g. `--model qwen3.5-moe --quant q8_0`) would be wrongly
# refused as "unsupported". The dotted forms are themselves registry aliases, so
# the canonicalized name remains valid input to `lumen pull` / the server.
#
# Unknown names pass through unchanged (model_name_known guards them separately).
canonical_model_name() {
  case "$1" in
    qwen3-5-9b|qwen3.5-9b)
      printf '%s' "qwen3.5-9b" ;;
    qwen3-5-moe-35b-a3b|qwen3.5-moe-35b-a3b|qwen3.5-moe)
      printf '%s' "qwen3.5-moe-35b-a3b" ;;
    *)
      printf '%s' "$1" ;;
  esac
}

# A copy-pasteable reference to the `lumen` CLI for guidance messages. After the
# build, $LUMEN_BIN is the real absolute path. Before the build (e.g. in
# validate_selection, which runs first), fall back to the conventional
# target/release path under the resolved repo root — still a real, runnable path
# the user can paste — or a bare 'lumen' if even the repo root is unknown.
lumen_cmd_hint() {
  if [ -n "$LUMEN_BIN" ]; then
    printf '%s' "$LUMEN_BIN"
  elif [ -n "$REPO_ROOT" ]; then
    printf '%s/target/release/lumen' "$REPO_ROOT"
  else
    printf '%s' "lumen"
  fi
}


# -----------------------------------------------------------------------------
#  Repo root
# -----------------------------------------------------------------------------

resolve_repo_root() {
  # Prefer git; fall back to the script's own location (scripts/..).
  local root=""
  if have git && git rev-parse --show-toplevel >/dev/null 2>&1; then
    root="$(git rev-parse --show-toplevel 2>/dev/null)"
  fi
  if [ -z "$root" ]; then
    # Script dir without relying on $0 games beyond dirname.
    local sdir
    sdir="$(cd "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
    root="$(cd "$sdir/.." >/dev/null 2>&1 && pwd)"
  fi
  if [ -z "$root" ] || [ ! -f "$root/Cargo.toml" ] || [ ! -f "$root/model_registry.toml" ]; then
    die "could not locate the Lumen repo root (run this from inside the repo: ./scripts/quickstart.sh)"
  fi
  REPO_ROOT="$root"
  log_debug "repo root: $REPO_ROOT"
}

# -----------------------------------------------------------------------------
#  Backend detection
# -----------------------------------------------------------------------------
#  Mirrors the runtime's own auto-detection so the script builds with the
#  features the binaries will actually pick:
#    macOS                         -> metal
#    Linux + usable NVIDIA device  -> cuda
#    otherwise                     -> cpu (correctness reference, not fast)
#
#  The runtime keys CUDA auto-selection off /dev/nvidia0 (see run.rs /
#  lumen-server.rs select_backend). We additionally consult nvidia-smi for a
#  friendlier preflight, but /dev/nvidia0 is the authoritative signal.
# -----------------------------------------------------------------------------

detect_backend() {
  # Explicit override wins (validated already; 'auto' falls through to detect).
  if [ -n "$OPT_BACKEND" ] && [ "$(to_lower "$OPT_BACKEND")" != "auto" ]; then
    BACKEND="$(to_lower "$OPT_BACKEND")"
    log_info "Backend: $BACKEND (forced via --backend)"
    return 0
  fi

  local uname_s
  uname_s="$(uname -s 2>/dev/null || echo unknown)"
  case "$uname_s" in
    Darwin)
      BACKEND="metal"
      log_info "Backend: Metal (Apple Silicon GPU) — detected macOS"
      ;;
    Linux)
      if [ -e /dev/nvidia0 ]; then
        BACKEND="cuda"
        log_info "Backend: CUDA (NVIDIA GPU) — detected /dev/nvidia0"
      elif have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
        # Driver tools present and working but no /dev/nvidia0 yet (e.g. the
        # device node appears on first use). Trust nvidia-smi.
        BACKEND="cuda"
        log_info "Backend: CUDA (NVIDIA GPU) — nvidia-smi reports a device"
      else
        BACKEND="cpu"
        log_warn "Backend: CPU — no NVIDIA GPU detected."
        log_warn "CPU is a correctness reference, NOT optimized for throughput."
      fi
      ;;
    *)
      BACKEND="cpu"
      log_warn "Backend: CPU — unrecognized OS '$uname_s' (no GPU backend)."
      ;;
  esac
}

# Note: cargo feature flags per backend are applied inline in build_binaries()
# (CLI: default features incl. `download`, or `--features cuda`; server:
# `--features bin` or `--features bin,cuda`) so each flag is a discrete,
# exactly-quoted argv element rather than a split string.

# -----------------------------------------------------------------------------
#  Resource probing (portable: macOS sysctl vs Linux /proc, BSD vs GNU)
# -----------------------------------------------------------------------------

# Total physical RAM in GiB (integer). Echoes a number, or empty on failure.
ram_total_gib() {
  local bytes=""
  if [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
    bytes="$(sysctl -n hw.memsize 2>/dev/null || echo '')"
  elif [ -r /proc/meminfo ]; then
    # MemTotal is in kB.
    local kb
    kb="$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null || echo '')"
    if [ -n "$kb" ]; then bytes=$((kb * 1024)); fi
  fi
  if [ -n "$bytes" ] && [ "$bytes" -gt 0 ] 2>/dev/null; then
    printf '%s' $((bytes / 1024 / 1024 / 1024))
  fi
}

# Free disk space in GiB (integer) for a given path's filesystem. Portable via
# POSIX `df -P` (1024-byte blocks via -k). Echoes a number, or empty on failure.
#
# Test/override hook: LUMEN_QS_FAKE_FREE_GIB, when set to a non-negative
# integer, short-circuits the probe and reports that many GiB free. This exists
# solely so the low-disk gate can be exercised deterministically in automated
# tests without actually filling a disk; it is never set in normal use and is
# not a documented user knob.
disk_free_gib() {
  local path="$1" avail_kb=""
  if [ -n "${LUMEN_QS_FAKE_FREE_GIB:-}" ]; then
    case "$LUMEN_QS_FAKE_FREE_GIB" in
      ''|*[!0-9]*) : ;;                                  # ignore garbage, fall through to real probe
      *) printf '%s' "$LUMEN_QS_FAKE_FREE_GIB"; return 0 ;;
    esac
  fi
  # `df -Pk` => POSIX columns; field 4 (Available) is in 1K blocks. The data
  # row may wrap, but -P guarantees a single physical line per filesystem.
  avail_kb="$(df -Pk "$path" 2>/dev/null | awk 'NR==2 {print $4; exit}')"
  if [ -n "$avail_kb" ] && [ "$avail_kb" -gt 0 ] 2>/dev/null; then
    printf '%s' $((avail_kb / 1024 / 1024))
  fi
}

# Parse a "~10 GB" / "~5.4 GB" footprint string into an integer GiB (rounded up).
# Echoes an integer; defaults to a safe upper bound if unparseable.
size_str_to_gib() {
  local s="$1" num=""
  # Strip everything except digits and a single dot.
  num="$(printf '%s' "$s" | LC_ALL=C tr -cd '0-9.' )"
  if [ -z "$num" ]; then printf '%s' "0"; return 0; fi
  # Round up: add 0.999 then truncate via awk (no bash float arithmetic).
  awk -v n="$num" 'BEGIN { printf "%d", (n + 0.999) }'
}

# CPU count (build parallelism). macOS sysctl, Linux nproc, fallback 4.
cpu_count() {
  local n=""
  if have nproc; then
    n="$(nproc 2>/dev/null || echo '')"
  fi
  if [ -z "$n" ] && [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
    n="$(sysctl -n hw.ncpu 2>/dev/null || echo '')"
  fi
  if [ -z "$n" ] || [ "$n" -lt 1 ] 2>/dev/null; then n="4"; fi
  printf '%s' "$n"
}

# -----------------------------------------------------------------------------
#  Preflight: prerequisites + resources
# -----------------------------------------------------------------------------

preflight() {
  log_step "Preflight checks"
  local missing=0

  # Required tools for build + download + run.
  local tool
  for tool in cargo rustc git curl; do
    if have "$tool"; then
      log_debug "found: $tool ($(command -v "$tool"))"
    else
      log_error "missing required tool: $tool"
      missing=1
    fi
  done

  if [ "$missing" = "1" ]; then
    log_error "Install the Rust toolchain (https://rustup.rs) and ensure git + curl are on PATH."
    die "prerequisites missing"
  fi
  log_ok "Toolchain present: rustc + cargo + git + curl"

  # CUDA-specific driver preflight (Linux only). The CUDA build itself needs no
  # CUDA SDK (kernels compile at runtime via NVRTC), but a usable driver is
  # required to actually run. Warn (don't hard-fail) so --build-only still works.
  if [ "$BACKEND" = "cuda" ]; then
    if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
      log_ok "NVIDIA driver responds (nvidia-smi)"
    else
      log_warn "CUDA backend selected but nvidia-smi did not respond."
      log_warn "The build will succeed, but running inference needs a working driver."
    fi
  fi

  # Disk headroom. During a pull the source GGUF AND the converted LBC coexist
  # on disk (the GGUF is not deleted until `lumen convert` finishes), so the
  # peak need is the catalog's TRANSIENT-DISK figure (CAT_DISK = GGUF + LBC),
  # NOT a "2× residency" guess and NOT the residency number. We add a small
  # build/scratch margin on top.
  #
  # If the selection is already cached, or is a direct file path, there is no
  # download/convert — so the transient-disk gate does not apply and we skip it
  # (a cached re-run must never be blocked by a disk number for work it won't do).
  # Uncataloged (model:quant) combos never reach here: validate_selection()
  # already refused them rather than guess a size.
  local cache_probe="$CACHE_DIR"
  if [ ! -d "$cache_probe" ]; then cache_probe="$(dirname "$cache_probe")"; fi
  if [ ! -d "$cache_probe" ]; then cache_probe="$HOME"; fi

  local disk_margin_gib=2
  if [ "$SEL_IS_PATH" = "1" ] || selected_is_cached; then
    log_debug "disk: selection needs no download/convert; skipping transient-disk gate."
  else
    local free_gib peak_gib need_gib
    free_gib="$(disk_free_gib "$cache_probe")"
    peak_gib="$(model_disk_peak_gib)"          # GGUF + LBC, from CAT_DISK
    need_gib=$(( peak_gib + disk_margin_gib ))  # + build/scratch margin

    if [ -z "$free_gib" ]; then
      log_warn "Could not determine free disk space at $cache_probe; skipping disk check."
    elif [ "$peak_gib" -le 0 ]; then
      # Defensive: a cataloged combo with no/garbled CAT_DISK. Do not fabricate a
      # pass — warn that the requirement is unverified rather than print a number.
      log_warn "Disk: transient need for ${SEL_SPEC} is unverified; ~${free_gib} GiB free at $cache_probe."
      log_warn "Downloading + converting holds the source GGUF and the output together; ensure ample free space."
    elif [ "$free_gib" -lt "$need_gib" ]; then
      log_warn "Low disk: ~${free_gib} GiB free at $cache_probe, ~${need_gib} GiB needed to pull ${SEL_SPEC}."
      log_warn "Convert holds the source GGUF (~$(_disk_gguf_note)) and the output LBC at once."
      log_warn "Free space or pick a smaller quant (e.g. q4_0), then re-run."
      if ! confirm "Continue anyway?" "N"; then
        die "aborted: insufficient disk headroom (need ~${need_gib} GiB, have ~${free_gib} GiB)"
      fi
    else
      log_ok "Disk: ~${free_gib} GiB free at $cache_probe (need ~${need_gib} GiB transient to pull + convert)"
    fi
  fi

  # Cache-dir writability. Only matters if we will actually download (not
  # already cached, and not --build-only). Failing here is friendlier than
  # discovering "Permission denied" mid-download. Skipped in --dry-run.
  if [ "$OPT_DRY_RUN" != "1" ] && [ "$OPT_BUILD_ONLY" != "1" ] && ! selected_is_cached; then
    if ! check_cache_writable; then
      die "cache directory is not writable: $CACHE_DIR"
    fi
  fi

  # RAM/VRAM residency check. For Metal/CPU the model loads into unified/system
  # RAM; for CUDA it loads into VRAM (which we cannot size portably here, so we
  # only advise). Compare the residency need against total RAM. This applies even
  # to a cached model (it still has to fit in memory to serve), so it is NOT
  # gated on "will we download" — unlike the transient-disk check above.
  local ram_gib model_gib
  ram_gib="$(ram_total_gib)"
  model_gib="$(model_footprint_gib)"      # peak RAM/VRAM residency, from CAT_NEED
  if [ -n "$ram_gib" ]; then
    case "$BACKEND" in
      cuda)
        log_info "Detected ${ram_gib} GiB system RAM. CUDA loads weights into VRAM;"
        log_info "ensure the GPU has >= ${model_gib} GiB free (check: nvidia-smi)."
        ;;
      *)
        if [ "$ram_gib" -lt "$model_gib" ]; then
          log_warn "Model ${SEL_SPEC} needs ~${model_gib} GiB resident but system has ${ram_gib} GiB RAM."
          log_warn "It may swap heavily or OOM. Consider a smaller quant (e.g. q4_0)."
          if ! confirm "Continue anyway?" "N"; then
            die "aborted: insufficient RAM for ${SEL_SPEC}"
          fi
        else
          log_ok "RAM: ${ram_gib} GiB total (model needs ~${model_gib} GiB resident)"
        fi
        ;;
    esac
  else
    log_warn "Could not determine system RAM; skipping memory check."
  fi
}

# Peak RAM/VRAM residency (GiB) for the SELECTED model, from CAT_NEED.
model_footprint_gib() {
  local idx
  idx="$(catalog_index_of "$SEL_SPEC")"
  if [ "$idx" != "-1" ]; then
    size_str_to_gib "${CAT_NEED[$idx]}"
  else
    # Unknown spec (user passed a custom model/quant). Be conservative.
    printf '%s' "12"
  fi
}

# Peak TRANSIENT DISK (GiB) the pull+convert will need for the SELECTED model,
# from CAT_DISK (source GGUF + converted LBC held simultaneously). Echoes an
# integer; "0" for an uncataloged spec (the caller treats 0 as "unverified" and
# never fabricates a pass — but uncataloged specs are refused before preflight,
# so this is defensive only).
model_disk_peak_gib() {
  local idx
  idx="$(catalog_index_of "$SEL_SPEC")"
  if [ "$idx" != "-1" ]; then
    size_str_to_gib "${CAT_DISK[$idx]}"
  else
    printf '%s' "0"
  fi
}

# Human note: the source-GGUF half of the transient peak, for the low-disk
# warning ("convert holds the source GGUF (~N GB) and the output at once").
# Derived as CAT_DISK − CAT_SIZE (peak minus the LBC output ≈ the GGUF). Falls
# back to a generic phrase when uncataloged.
_disk_gguf_note() {
  local idx peak lbc gguf
  idx="$(catalog_index_of "$SEL_SPEC")"
  if [ "$idx" = "-1" ]; then printf '%s' "the source GGUF"; return 0; fi
  peak="$(size_str_to_gib "${CAT_DISK[$idx]}")"
  lbc="$(size_str_to_gib "${CAT_SIZE[$idx]}")"
  gguf=$(( peak - lbc ))
  if [ "$gguf" -lt 1 ]; then gguf=1; fi
  printf '%s GB' "$gguf"
}

# -----------------------------------------------------------------------------
#  Cache directory resolution (mirrors crates/lumen-cli/src/cache.rs)
# -----------------------------------------------------------------------------

resolve_cache_dir() {
  if [ -n "${LUMEN_CACHE_DIR:-}" ]; then
    CACHE_DIR="$LUMEN_CACHE_DIR"
  elif [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
    CACHE_DIR="$HOME/Library/Caches/lumen"
  else
    CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/lumen"
  fi
  log_debug "cache dir: $CACHE_DIR"
}

# Path where the selected model's LBC would live (mirrors cache.rs lbc_path).
# Echoes the canonical (non-metal) path, or the direct path when SEL_IS_PATH=1.
# The CLI/server additionally accept a "<key>-<QUANT>-metal.lbc" variant on
# macOS; we check both at call sites.
selected_lbc_path() {
  if [ "$SEL_IS_PATH" = "1" ]; then
    printf '%s' "$SEL_MODEL"
    return 0
  fi
  local key quant
  key="$(printf '%s' "$SEL_MODEL" | tr '.' '-')"   # qwen3.5-9b -> qwen3-5-9b
  quant="$(to_upper "$SEL_QUANT")"                 # q8_0 -> Q8_0
  printf '%s/%s-%s.lbc' "$CACHE_DIR" "$key" "$quant"
}

# True (0) if a usable cached/available model file already exists for the
# selection. For a direct path, that means the file exists and is non-empty.
selected_is_cached() {
  if [ "$SEL_IS_PATH" = "1" ]; then
    if [ -s "$SEL_MODEL" ]; then return 0; fi
    return 1
  fi
  local key quant base metal
  key="$(printf '%s' "$SEL_MODEL" | tr '.' '-')"
  quant="$(to_upper "$SEL_QUANT")"
  base="$CACHE_DIR/$key-$quant.lbc"
  metal="$CACHE_DIR/$key-$quant-metal.lbc"
  if [ -s "$metal" ] || [ -s "$base" ]; then
    return 0
  fi
  return 1
}

# True (0) if the cache directory exists-or-can-be-created AND is writable.
# Creates the directory if missing (mkdir -p), then probes with a temp file we
# immediately remove. Emits an actionable error on failure.
check_cache_writable() {
  if [ ! -d "$CACHE_DIR" ]; then
    if ! mkdir -p "$CACHE_DIR" 2>/dev/null; then
      log_error "Cannot create cache directory: $CACHE_DIR"
      log_error "Check permissions on its parent, or set LUMEN_CACHE_DIR to a writable path."
      return 1
    fi
  fi
  local probe="$CACHE_DIR/.lumen-quickstart-write-test.$$"
  if ( : >"$probe" ) 2>/dev/null; then
    rm -f "$probe" 2>/dev/null || true
    log_ok "Cache writable: $CACHE_DIR"
    return 0
  fi
  log_error "Cache directory is not writable: $CACHE_DIR"
  log_error "Fix permissions, or set LUMEN_CACHE_DIR to a writable path."
  return 1
}

# -----------------------------------------------------------------------------
#  Build
# -----------------------------------------------------------------------------

# Echo the expected binary path for a crate's bin name under target/release.
target_bin_path() {
  printf '%s/target/release/%s' "$REPO_ROOT" "$1"
}

# Run a cargo build step with a friendly failure message. Honors --dry-run
# (prints the plan, executes nothing). cargo's own stdout/stderr — including any
# compiler error — streams through untouched; on a non-zero exit we add an
# actionable pointer and abort with a clear message (still non-zero).
#   run_build "<label>" cargo build ...
run_build() {
  local label="$1" ; shift
  if [ "$OPT_DRY_RUN" = "1" ]; then
    printf '%s   %s[dry-run]%s cargo build %s\n' "$(_ts)" "$C_DIM" "$C_RESET" "$label" >&2
    printf '             %s$ %s%s\n' "$C_DIM" "$(_show_argv "$@")" "$C_RESET" >&2
    return 0
  fi
  log_debug "exec: $(_show_argv "$@")"
  # Temporarily relax errexit so a cargo failure becomes a handled error rather
  # than a bare abort, then restore the previous setting.
  local rc=0
  set +e
  "$@"
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    log_error "Build failed for $label (cargo exit $rc). See the cargo output above."
    log_error "Common fixes: update the toolchain (rustup update), then re-run."
    if [ "$BACKEND" = "cuda" ]; then
      log_error "CUDA builds also need the NVIDIA toolchain available at build time."
    fi
    die "compilation failed"
  fi
}

# Build one binary if it isn't already fresh. We let cargo decide freshness
# (it is incremental and fast on a no-op), so "skip already-built" == "cargo
# does nothing". --skip-build bypasses cargo entirely and just checks the path.
build_binaries() {
  log_step "Build the Lumen CLI + server ($BACKEND)"

  LUMEN_BIN="$(target_bin_path lumen)"
  SERVER_BIN="$(target_bin_path lumen-server)"

  if [ "$OPT_SKIP_BUILD" = "1" ]; then
    log_info "--skip-build: using existing binaries."
    if [ ! -x "$LUMEN_BIN" ] || [ ! -x "$SERVER_BIN" ]; then
      die "--skip-build set but binaries are missing. Build first (drop --skip-build) or run cargo build."
    fi
    log_ok "Found: $LUMEN_BIN"
    log_ok "Found: $SERVER_BIN"
    return 0
  fi

  local jobs
  jobs="$(cpu_count)"

  # Build the CLI (with the download feature, on by default) and the server bin.
  # We pass features as discrete argv elements (never a single split string) so
  # quoting is exact. cargo's own output (incl. any compiler error) streams to
  # the terminal; on failure we add a short, actionable pointer and exit
  # non-zero. `run_build` disables errexit only around the cargo call so we can
  # turn cargo's exit code into a friendly message instead of a bare abort.
  log_info "Compiling lumen (CLI)…  this can take a few minutes on a cold build."
  if [ "$BACKEND" = "cuda" ]; then
    run_build "lumen (CLI, cuda)" \
      cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" \
      -p lumen-cli --features cuda --jobs "$jobs"
  else
    run_build "lumen (CLI)" \
      cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" \
      -p lumen-cli --jobs "$jobs"
  fi

  log_info "Compiling lumen-server…"
  if [ "$BACKEND" = "cuda" ]; then
    run_build "lumen-server (bin,cuda)" \
      cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" \
      -p lumen-server --features bin,cuda --jobs "$jobs"
  else
    run_build "lumen-server (bin)" \
      cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" \
      -p lumen-server --features bin --jobs "$jobs"
  fi

  if [ "$OPT_DRY_RUN" = "1" ]; then
    log_info "(dry-run) would verify binaries at target/release/."
    return 0
  fi

  [ -x "$LUMEN_BIN" ]  || die "build reported success but $LUMEN_BIN is missing/not executable"
  [ -x "$SERVER_BIN" ] || die "build reported success but $SERVER_BIN is missing/not executable"
  log_ok "Built: $LUMEN_BIN"
  log_ok "Built: $SERVER_BIN"
}

# -----------------------------------------------------------------------------
#  Model selection
# -----------------------------------------------------------------------------

# Render the interactive catalog menu and set SEL_* from the user's choice.
# In --yes / non-interactive mode this is skipped (configure() set the defaults).
select_model_interactive() {
  printf '\n%sChoose a model%s  (press Enter for the default)\n' "$C_BLD" "$C_RESET" >&2
  printf '%s\n' "  ── # ─ spec ───────────────────────── size ── needs ── notes ──" >&2

  local i=0 default_idx=0 pull_note
  while [ "$i" -lt "${#CAT_SPEC[@]}" ]; do
    if [ "${CAT_SPEC[$i]}" = "$DEFAULT_MODEL:$DEFAULT_QUANT" ]; then
      default_idx="$i"
    fi
    pull_note=""
    if [ "${CAT_PULL[$i]}" != "1" ]; then
      pull_note=" ${C_DIM}(needs local convert)${C_RESET}"
    fi
    printf '   %s%2d%s  %-32s %-8s %-7s %s%s\n' \
      "$C_BLD" "$((i + 1))" "$C_RESET" \
      "${CAT_SPEC[$i]}" "${CAT_SIZE[$i]}" "${CAT_NEED[$i]}" \
      "${CAT_DESC[$i]}" "$pull_note" >&2
    i=$((i + 1))
  done

  local choice="" def_num=$((default_idx + 1))
  printf '\n%sSelection%s [1-%d, default %d]: ' "$C_BLD" "$C_RESET" "${#CAT_SPEC[@]}" "$def_num" >&2
  IFS= read -r choice || choice=""

  if [ -z "$choice" ]; then
    choice="$def_num"
  fi
  # Validate: digits only.
  case "$choice" in
    ''|*[!0-9]*) die "invalid selection: '$choice' (expected a number 1-${#CAT_SPEC[@]})" ;;
  esac
  # Force base-10 for all arithmetic so a leading-zero input (e.g. "08") can
  # never be mis-read as octal and crash under set -e. `[ ]` comparisons already
  # treat the value as decimal; `10#` makes the $(( )) below match.
  local choice_n=$((10#$choice))
  if [ "$choice_n" -lt 1 ] || [ "$choice_n" -gt "${#CAT_SPEC[@]}" ]; then
    die "selection out of range: $choice (expected 1-${#CAT_SPEC[@]})"
  fi

  local idx=$((choice_n - 1))
  set_selection_from_spec "${CAT_SPEC[$idx]}"
}

# Split a "<model>:<quant>" registry spec into SEL_MODEL / SEL_QUANT / SEL_SPEC.
# Resets SEL_IS_PATH (this is the registry-name path, not a file path). The model
# name is canonicalized to the catalog's dotted spelling so any accepted alias
# (dashed key, short alias) maps to the same catalog entry — a pullable combo is
# never mis-refused just because the user spelled the model a different valid way.
set_selection_from_spec() {
  local spec="$1" raw_model raw_quant
  raw_model="${spec%%:*}"
  raw_quant="${spec#*:}"
  if [ "$raw_model" = "$spec" ] || [ -z "$raw_quant" ]; then
    die "internal: malformed model spec '$spec' (expected <model>:<quant>)"
  fi
  SEL_IS_PATH="0"
  SEL_MODEL="$(canonical_model_name "$raw_model")"
  SEL_QUANT="$raw_quant"
  SEL_SPEC="$SEL_MODEL:$SEL_QUANT"
}

# Configure the selection for a DIRECT .lbc/.gguf file path. The path is passed
# to the server verbatim (case-preserved, no quant tag). The server resolves the
# tokenizer + quant from the file's own header.
set_selection_from_path() {
  SEL_IS_PATH="1"
  SEL_MODEL="$1"     # the path, exactly as given
  SEL_QUANT=""
  SEL_SPEC="$1"
}

# Resolve the final selection from flags/env/defaults + (maybe) the menu.
choose_model() {
  log_step "Select a model"
  build_catalog

  local raw_model="${OPT_MODEL:-$DEFAULT_MODEL}"

  # Case 1: --model is a direct file path (.lbc/.gguf or contains a separator).
  # Pass it through verbatim — never lowercase it, never append a quant tag.
  if model_looks_like_path "$raw_model"; then
    set_selection_from_path "$raw_model"
    log_info "Model: $SEL_SPEC (direct file path)"
    validate_selection
    return 0
  fi

  # Case 2: registry name. --model may itself carry a ':quant' tag (e.g.
  # --model qwen3.5-9b:q8_0), the same form the CLI accepts. Split it so we never
  # build a double-tagged spec like 'qwen3.5-9b:q8_0:q8_0'. Quant precedence
  # mirrors the CLI: an explicit ':tag' on --model wins over --quant, which wins
  # over the default.
  local model="$raw_model" tag_quant=""
  case "$raw_model" in
    *:*)
      tag_quant="${raw_model##*:}"
      model="${raw_model%:*}"
      ;;
  esac
  local quant="${tag_quant:-${OPT_QUANT:-$DEFAULT_QUANT}}"

  # Interactive only when: a TTY, not --yes, and the user did NOT pin a model or
  # quant explicitly (via flag or env).
  if [ -t 0 ] && [ "$OPT_YES" != "1" ] && [ -z "$OPT_MODEL" ] && [ -z "$OPT_QUANT" ]; then
    select_model_interactive
  else
    set_selection_from_spec "$(to_lower "$model"):$(to_lower "$quant")"
    log_info "Model: $SEL_SPEC (from flags/defaults)"
  fi

  validate_selection
}

# Guard against (model, quant) pairs that the CLI would reject, with actionable
# guidance — before we spend time building or downloading.
validate_selection() {
  local idx

  # Direct file path: just sanity-check it exists (skip in --dry-run, where we
  # don't touch the filesystem). The server reads quant + tokenizer from the
  # file header; there is nothing to validate against the registry.
  if [ "$SEL_IS_PATH" = "1" ]; then
    if [ "$OPT_DRY_RUN" != "1" ] && [ ! -e "$SEL_MODEL" ]; then
      log_error "Model file not found: $SEL_MODEL"
      die "model path does not exist"
    fi
    case "$SEL_MODEL" in
      *.lbc) : ;;
      *.gguf)
        log_warn "The server loads .lbc files; '$SEL_MODEL' is a GGUF."
        log_warn "Convert first: $LUMEN_BIN convert --input '$SEL_MODEL' --output <out.lbc>"
        ;;
    esac
    log_ok "Using model file: $SEL_MODEL"
    return 0
  fi

  # Reject a model NAME that is not a known registry alias/key. This fires even
  # under --dry-run (where the CLI never runs to surface its own "Unknown model"
  # error), so the printed plan never advertises a model that does not exist.
  if ! model_name_known "$SEL_MODEL"; then
    log_error "Unknown model: '$SEL_MODEL'."
    log_error "Known models (from model_registry.toml):"
    log_error "  qwen3.5-9b"
    log_error "  qwen3.5-moe-35b-a3b   (alias: qwen3.5-moe)"
    log_error "Run 'lumen models' for the live registry + cached LBCs."
    die "model name not recognized"
  fi

  idx="$(catalog_index_of "$SEL_SPEC")"

  if [ "$idx" = "-1" ]; then
    # Known model name, but a (model, quant) pair the quickstart does not cover.
    # We must NOT proceed: the uncataloged path has no verified size (so the disk
    # gate could not protect the user) and the download may be impossible. A
    # quickstart's job is to refuse cleanly with guidance, not to march a new
    # user into a multi-GB download that might fail partway and fill their disk.
    #
    # Exception: if a matching LBC is already cached, the combo is locally usable
    # (the user converted it themselves) — serve it; no download/size question.
    if selected_is_cached; then
      log_ok "$SEL_SPEC found in cache (locally converted) — using it."
      return 0
    fi
    log_error "'$SEL_SPEC' is not a supported quickstart combination."
    log_error "Supported combinations:"
    print_supported_combos
    local lumen_hint ; lumen_hint="$(lumen_cmd_hint)"
    log_error "Either pick one of the above, or prepare '$SEL_SPEC' yourself:"
    log_error "  $lumen_hint pull <model>:<quant> --yes      # if the registry has that GGUF"
    log_error "  $lumen_hint convert --input <src.gguf> --output <out.lbc> [--requant <q>]"
    log_error "  …then re-run:  $0 --model <path-to.lbc>"
    log_error "Run '$lumen_hint models' for the live registry + your cached LBCs."
    die "unsupported quickstart model/quant combination: $SEL_SPEC"
  fi

  # Pair is catalogued but not downloadable. Allow only if a local LBC is already
  # cached; otherwise explain the (combo-specific) manual path and stop — never
  # attempt a pull the CLI cannot satisfy.
  if [ "${CAT_PULL[$idx]}" != "1" ]; then
    if selected_is_cached; then
      log_ok "$SEL_SPEC found in cache (locally converted)."
      return 0
    fi

    # Dense 9B q4_0: derived locally by re-quantizing the Q8_0 source.
    # (BF16 everywhere — incl. the 2-shard MoE/27B splits — is pullable: the
    # downloader fetches nested-subdir shards and caches them flat.)
    local lumen_hint ; lumen_hint="$(lumen_cmd_hint)"
    log_error "$SEL_SPEC is not downloadable from the registry (derive it locally)."
    log_error "Build it from the Q8_0 source, then re-run with --skip-build:"
    log_error "  $lumen_hint pull ${SEL_MODEL}:q8_0 --yes"
    log_error "  $lumen_hint convert --input <q8_0.gguf> --output <out.lbc> --requant ${SEL_QUANT}"
    die "model not available for direct download"
  fi
}

# -----------------------------------------------------------------------------
#  Pull (download + convert + cache) — idempotent
# -----------------------------------------------------------------------------

# Snapshot *.part files in the cache dir BEFORE we pull, so cleanup only removes
# partials that THIS run created (never a concurrent process's in-flight file).
snapshot_partials() {
  PARTFILES_BEFORE=""
  if [ -d "$CACHE_DIR" ]; then
    PARTFILES_BEFORE="$(find "$CACHE_DIR" -maxdepth 1 -type f -name '*.part' 2>/dev/null || true)"
  fi
}

pull_model() {
  log_step "Fetch the model: $SEL_SPEC"

  # Direct file path: nothing to download. The file's existence was already
  # validated in validate_selection (outside --dry-run).
  if [ "$SEL_IS_PATH" = "1" ]; then
    log_ok "Using direct model file (no download needed): $SEL_MODEL"
    return 0
  fi

  if selected_is_cached; then
    log_ok "Already cached — skipping download. ($(selected_lbc_path))"
    return 0
  fi

  log_info "Not cached. Downloading + converting (this can be several GB)."
  log_info "Cache: $CACHE_DIR"
  snapshot_partials

  # `lumen pull` is idempotent and verifies integrity (SHA-256 sidecar). It
  # prints 'Already cached' if a prior run finished. We pass --yes so the
  # confirm prompt never blocks an automated run; interactive users already
  # consented via the menu / their explicit flags.
  if ! run_cmd "lumen pull $SEL_SPEC" "$LUMEN_BIN" pull "$SEL_SPEC" --yes; then
    die "model download/convert failed for $SEL_SPEC (see messages above)"
  fi

  if [ "$OPT_DRY_RUN" != "1" ] && ! selected_is_cached; then
    die "pull reported success but no cached LBC was found for $SEL_SPEC"
  fi
  log_ok "Model ready: $SEL_SPEC"
}

# -----------------------------------------------------------------------------
#  Port availability
# -----------------------------------------------------------------------------

# True (0) if something is already listening on host:port.
port_in_use() {
  local host="$1" port="$2"
  # Prefer lsof (precise, macOS+Linux). Fall back to a bash /dev/tcp probe.
  if have lsof; then
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    return 1
  fi
  # /dev/tcp connect test (loopback). A successful connect => in use.
  if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
    exec 3>&- 3<&- 2>/dev/null || true
    return 0
  fi
  return 1
}

ensure_port_free() {
  if port_in_use "$OPT_HOST" "$OPT_PORT"; then
    log_error "Port $OPT_HOST:$OPT_PORT is already in use."
    log_error "Pick another with --port <n>, or stop the process using it:"
    if have lsof; then
      log_error "  lsof -nP -iTCP:$OPT_PORT -sTCP:LISTEN"
    fi
    die "port unavailable"
  fi
  log_debug "port $OPT_HOST:$OPT_PORT is free"
}

# -----------------------------------------------------------------------------
#  Serve + readiness probe
# -----------------------------------------------------------------------------

base_url() {
  printf 'http://%s:%s' "$OPT_HOST" "$OPT_PORT"
}

start_server() {
  log_step "Start lumen-server ($BACKEND) on $(base_url)"

  ensure_port_free

  SERVER_LOG="$SCRATCH_DIR/lumen-server.log"

  # Build the server argv. Two model forms (both verified against
  # lumen-server.rs::resolve_model_path):
  #   - registry name + --quant: the server resolves it to the cached LBC
  #     (it never auto-downloads).
  #   - direct .lbc path: passed verbatim, with NO --quant (the server reads
  #     quant + tokenizer from the file header; --quant is meaningless here).
  # Backend is explicit to match what we built/detected.
  # NOTE: on Metal the server sets LUMEN_METAL_MMAP_ONLY=1 itself for big
  # models; we don't need to.
  if [ "$SEL_IS_PATH" = "1" ]; then
    set -- "$SERVER_BIN" \
      --model "$SEL_MODEL" \
      --backend "$BACKEND" \
      --host "$OPT_HOST" --port "$OPT_PORT"
  else
    set -- "$SERVER_BIN" \
      --model "$SEL_MODEL" --quant "$SEL_QUANT" \
      --backend "$BACKEND" \
      --host "$OPT_HOST" --port "$OPT_PORT"
  fi

  if [ "$OPT_DRY_RUN" = "1" ]; then
    printf '%s   %s[dry-run]%s start server\n' "$(_ts)" "$C_DIM" "$C_RESET" >&2
    printf '             %s$ %s%s\n' "$C_DIM" "$(_show_argv "$@")" "$C_RESET" >&2
    printf '             %s$ %s > %s 2>&1 &%s\n' "$C_DIM" "(see above)" "$SERVER_LOG" "$C_RESET" >&2
    return 0
  fi

  log_info "Launching: $(_show_argv "$@")"
  log_info "Server log: $SERVER_LOG"

  # Launch detached, capturing all output to the log. We keep the PID for the
  # readiness probe and the cleanup trap.
  "$@" >"$SERVER_LOG" 2>&1 &
  SERVER_PID="$!"
  SERVER_STARTED="1"
  log_info "Server PID: $SERVER_PID"

  wait_for_ready
}

# Poll /v1/models until the server answers or we time out / it dies.
wait_for_ready() {
  log_info "Waiting for the server to load weights and accept requests…"
  local url deadline now
  url="$(base_url)/v1/models"
  now="$(date +%s)"
  deadline=$(( now + READY_TIMEOUT_SECS ))

  while :; do
    # If the server process died, surface its log and fail fast.
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log_error "Server process exited before becoming ready. Last log lines:"
      tail -n 25 "$SERVER_LOG" >&2 2>/dev/null || true
      die "server failed to start"
    fi

    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      log_ok "Server is ready at $(base_url)"
      return 0
    fi

    now="$(date +%s)"
    if [ "$now" -ge "$deadline" ]; then
      log_error "Server did not become ready within ${READY_TIMEOUT_SECS}s. Last log lines:"
      tail -n 25 "$SERVER_LOG" >&2 2>/dev/null || true
      die "readiness timeout"
    fi
    sleep "$READY_POLL_INTERVAL_SECS"
  done
}

# -----------------------------------------------------------------------------
#  Next-steps block (copy-paste curl + lumen run)
# -----------------------------------------------------------------------------

print_next_steps() {
  local url model_id
  url="$(base_url)"
  # The server echoes the --model value as the OpenAI model id.
  model_id="$SEL_MODEL"

  # All to stderr (status/UX), keeping stdout clean.
  {
    printf '\n'
    printf '%s%s━━━ Lumen is running ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%s\n' "$C_BLD" "$C_GRN" "$C_RESET"
    printf '\n'
    printf '  Server:  %s%s%s   (backend: %s, model: %s)\n' "$C_BLD" "$url" "$C_RESET" "$BACKEND" "$SEL_SPEC"
    printf '  Log:     %s\n' "$SERVER_LOG"
    printf '\n'
    printf '  %sOpen a SECOND terminal%s and try one of these:\n' "$C_BLD" "$C_RESET"
    printf '\n'
    printf '  %s# 1) OpenAI-compatible chat completion (curl)%s\n' "$C_DIM" "$C_RESET"
    printf '  curl -fsS %s/v1/chat/completions \\\n' "$url"
    printf '    -H "Content-Type: application/json" \\\n'
    printf '    -d '"'"'{\n'
    printf '      "model": "%s",\n' "$model_id"
    printf '      "messages": [{"role": "user", "content": "Write a haiku about Rust."}],\n'
    printf '      "max_tokens": 128\n'
    printf '    }'"'"'\n'
    printf '\n'
    printf '  %s# 2) List models (readiness probe)%s\n' "$C_DIM" "$C_RESET"
    printf '  curl -fsS %s/v1/models\n' "$url"
    printf '\n'
    printf '  %s# 3) One-shot inference with the CLI (separate process)%s\n' "$C_DIM" "$C_RESET"
    if [ "$SEL_IS_PATH" = "1" ]; then
      printf '  %s run --model %s --prompt "Write a haiku about Rust."\n' \
        "$(_shq "$LUMEN_BIN")" "$(_shq "$SEL_MODEL")"
    else
      printf '  %s run %s "Write a haiku about Rust."\n' "$(_shq "$LUMEN_BIN")" "$SEL_SPEC"
    fi
    printf '\n'
    printf '  %sStop the server:%s press Ctrl-C in THIS terminal.\n' "$C_BLD" "$C_RESET"
    printf '%s%s━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%s\n' "$C_BLD" "$C_GRN" "$C_RESET"
    printf '\n'
  } >&2
}

# Block in the foreground while the server runs, so Ctrl-C reaches our trap and
# the user keeps a live terminal attached to the server.
wait_for_server() {
  if [ "$OPT_DRY_RUN" = "1" ]; then
    return 0
  fi
  log_info "Server is running. Press Ctrl-C to stop."
  # `wait` returns when the server exits or a signal interrupts it.
  wait "$SERVER_PID" 2>/dev/null || true
}

# -----------------------------------------------------------------------------
#  Cleanup  (trap on EXIT / INT / TERM — idempotent, never leaves orphans)
# -----------------------------------------------------------------------------

CLEANUP_DONE="0"

cleanup() {
  # Run once. The EXIT trap fires after INT/TERM handlers, so guard re-entry.
  if [ "$CLEANUP_DONE" = "1" ]; then return 0; fi
  CLEANUP_DONE="1"

  # Stop the server if we started it and it's still alive.
  if [ "$SERVER_STARTED" = "1" ] && [ -n "$SERVER_PID" ]; then
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      log_info "Stopping lumen-server (PID $SERVER_PID)…" 2>/dev/null || true
      # Graceful first (the server traps SIGTERM/SIGINT for clean shutdown).
      kill -TERM "$SERVER_PID" 2>/dev/null || true
      # Give it a moment, then escalate if needed.
      local waited=0
      while [ "$waited" -lt 10 ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
        sleep 1
        waited=$((waited + 1))
      done
      if kill -0 "$SERVER_PID" 2>/dev/null; then
        log_warn "Server did not stop gracefully; sending SIGKILL." 2>/dev/null || true
        kill -KILL "$SERVER_PID" 2>/dev/null || true
      fi
    fi
  fi

  # Remove ONLY the partial downloads this run created (never pre-existing ones,
  # never a concurrent pull's in-flight file). We compare by EXACT line, not
  # substring, so one path being a substring of another can never mis-classify.
  if [ -d "$CACHE_DIR" ]; then
    local now_parts p
    now_parts="$(find "$CACHE_DIR" -maxdepth 1 -type f -name '*.part' 2>/dev/null || true)"
    if [ -n "$now_parts" ]; then
      local oldifs="$IFS"
      IFS=$'\n'
      for p in $now_parts; do
        # Skip if this exact path existed before we started (not ours).
        if line_in_list "$p" "$PARTFILES_BEFORE"; then
          continue
        fi
        log_debug "removing partial download: $p" 2>/dev/null || true
        rm -f "$p" 2>/dev/null || true
      done
      IFS="$oldifs"
    fi
  fi

  # Remove our scratch dir (server log lives here; we already tail'd it on error).
  if [ -n "$SCRATCH_DIR" ] && [ -d "$SCRATCH_DIR" ]; then
    rm -rf "$SCRATCH_DIR" 2>/dev/null || true
  fi
}

on_interrupt() {
  # Print a newline so the ^C doesn't mangle the next line, then exit. The EXIT
  # trap runs cleanup. Exit code 130 = terminated by SIGINT (conventional).
  printf '\n' >&2
  log_warn "Interrupted — shutting down." 2>/dev/null || true
  exit 130
}

install_traps() {
  # SCRATCH_DIR must exist before traps can clean it. The EXIT trap is the real
  # guarantor: it runs on a normal return AND after any of the signal handlers
  # below call `exit`, so the server is stopped and partials are removed on
  # every termination path we can influence.
  #   - INT  (Ctrl-C, foreground): friendly message, exit 130.
  #   - TERM (kill, supervisor):   exit 143.
  #   - HUP  (terminal/SSH disconnect): exit 129 — without this, a server
  #     started by the script could be orphaned when the controlling terminal
  #     goes away.
  # (Async/background launches run with INT=SIG_IGN per POSIX; TERM/HUP/EXIT
  # still fire, so cleanup is preserved there too.)
  trap cleanup EXIT
  trap on_interrupt INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
}

# -----------------------------------------------------------------------------
#  Configuration summary
# -----------------------------------------------------------------------------

print_plan() {
  log_step "Plan"
  log_info "Repo:     $REPO_ROOT"
  log_info "Backend:  $BACKEND"
  log_info "Model:    $SEL_SPEC"
  log_info "Cache:    $CACHE_DIR"
  log_info "Server:   $(base_url)  (after build + pull)"
  if [ "$OPT_DRY_RUN" = "1" ]; then
    log_info "Mode:     DRY RUN (no commands will be executed)"
  fi
  if [ "$OPT_BUILD_ONLY" = "1" ]; then
    log_info "Mode:     BUILD ONLY (stop after building binaries)"
  fi
  if [ "$OPT_NO_SERVE" = "1" ]; then
    log_info "Mode:     NO SERVE (build + pull, then stop)"
  fi
}

# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

main() {
  parse_args "$@"
  validate_args

  # A scratch dir for logs etc. Created early so traps can always clean it.
  SCRATCH_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lumen-quickstart.XXXXXX")" \
    || die "could not create a temporary working directory"
  install_traps

  printf '%s%slumen quickstart v%s%s\n' "$C_BLD" "$C_BLU" "$QS_VERSION" "$C_RESET" >&2

  resolve_repo_root
  detect_backend
  resolve_cache_dir
  choose_model          # sets SEL_MODEL / SEL_QUANT / SEL_SPEC, validates pair
  print_plan

  preflight             # tools + disk + RAM/VRAM for the chosen model

  build_binaries
  if [ "$OPT_BUILD_ONLY" = "1" ]; then
    log_ok "Build complete (--build-only). Binaries in $REPO_ROOT/target/release/."
    log_info "Next: ./scripts/quickstart.sh --skip-build   # to pull + serve"
    return 0
  fi

  pull_model
  if [ "$OPT_NO_SERVE" = "1" ]; then
    log_ok "Model ready (--no-serve). Start the server yourself with:"
    if [ "$SEL_IS_PATH" = "1" ]; then
      log_info "  $(_show_argv "$SERVER_BIN" --model "$SEL_MODEL" --backend "$BACKEND" --port "$OPT_PORT")"
    else
      log_info "  $(_show_argv "$SERVER_BIN" --model "$SEL_MODEL" --quant "$SEL_QUANT" --backend "$BACKEND" --port "$OPT_PORT")"
    fi
    return 0
  fi

  if [ "$OPT_DRY_RUN" = "1" ]; then
    start_server        # prints the planned launch
    print_next_steps
    log_info "(dry-run) complete — nothing was executed."
    return 0
  fi

  start_server          # launches + waits for readiness
  print_next_steps
  wait_for_server       # blocks until Ctrl-C / server exit; trap cleans up
}

main "$@"
