#!/usr/bin/env bash
# Build lumen-server under AddressSanitizer + LeakSanitizer (STAB-003).
#
# The sanitizers need a nightly rustc (`-Zsanitizer` is unstable), an explicit
# target triple, and the system allocator: they track that heap and not
# mimalloc's, so the build enables the `system-allocator` feature. Everything
# goes into target/sanitize/ so the pinned toolchain's artifacts stay
# untouched, and the script checks that the binary it names is instrumented.
# Linux x86_64 only; the features are the production set (bin,cuda,image).
#
# USAGE
#   scripts/sanitize_server.sh
#
# Then run the printed binary under load and stop it with SIGINT: the leak
# report is written when the process exits normally, so a SIGKILL'd server
# reports nothing. Two runtime facts:
#   - CUDA context creation fails with CUDA_ERROR_OUT_OF_MEMORY inside ASan's
#     reserved shadow range; `protect_shadow_gap=0` lets the driver map there.
#   - A run with no report only means something once the report path is known
#     to work: a deliberate leak (allocations kept alive with
#     `std::hint::black_box`, or the optimiser removes them) must produce one
#     under the same ASAN_OPTIONS before a clean run is trusted.
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=x86_64-unknown-linux-gnu
BIN="target/sanitize/$TARGET/release/lumen-server"
[ -z "${CARGO_ENCODED_RUSTFLAGS+x}" ] \
  || { echo "CARGO_ENCODED_RUSTFLAGS is set and would replace RUSTFLAGS; unset it" >&2; exit 1; }
RUSTFLAGS="${RUSTFLAGS:-} -Zsanitizer=address" cargo +nightly build --release \
  --target "$TARGET" --target-dir target/sanitize \
  -p lumen-server --features bin,cuda,image,system-allocator
ASAN_OPTIONS=help=1 "$BIN" --version 2>&1 | grep 'Available flags for AddressSanitizer' > /dev/null \
  || { echo "$BIN is not instrumented" >&2; exit 1; }
echo "binary: $BIN"
echo "run with: ASAN_OPTIONS=detect_leaks=1:protect_shadow_gap=0:log_path=/path/to/asan"
