#!/usr/bin/env bash
# Native bench wrapper for CUDA MoE + dense validation.
#
# Thin shell entrypoint over scripts/native_bench.py.
# Sources rustup env, ensures /usr/local/cuda is on PATH, then runs the
# Python harness.  All args are forwarded.
#
# Usage:
#   bash scripts/native_bench.sh
#   bash scripts/native_bench.sh --cells A
#   bash scripts/native_bench.sh --skip-build --cells D1
#
# Outputs:
#   <LUMEN_CACHE_ROOT>/artifacts/native-bench-results.json
#   <LUMEN_CACHE_ROOT>/logs/native-bench.log  (when tee'd in tmux)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source rustup env if available.
if [ -f "${HOME}/.cargo/env" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi

# Ensure CUDA toolchain is reachable for `nvcc` (cudarc invokes it during build).
export PATH="/usr/local/cuda/bin:${HOME}/.cargo/bin:${PATH:-}"

# LUMEN_ROOT defaults to the repo root; can be overridden externally.
export LUMEN_ROOT="${LUMEN_ROOT:-${REPO_ROOT}}"

# Defensive: confirm cargo is reachable so the user sees a clear error early.
if ! command -v cargo >/dev/null 2>&1; then
  echo "ERROR: cargo not on PATH after sourcing rustup env" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "WARN: nvcc not on PATH; CUDA cudarc build may fail" >&2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/native_bench.py" "$@"
