#!/usr/bin/env bash
#
# Lumen — GPU-resident LLM inference engine
#
# Usage:
#   ./lumen "What is the meaning of life?"
#   ./lumen --model llama-8b "Write a haiku about Rust"
#   ./lumen pull qwen2.5-3b
#   ./lumen models
#
# First run builds from source (requires Rust). Subsequent runs are instant.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/target/release/lumen"

# ── Build if needed ──────────────────────────────────────────────────────────

if [ ! -f "$BINARY" ]; then
    if ! command -v cargo &>/dev/null; then
        echo "Rust is required to build Lumen."
        echo ""
        echo "Install Rust:"
        echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo ""
        echo "Then re-run: ./lumen \"$*\""
        exit 1
    fi

    echo "Building Lumen (first run only, takes ~2 minutes)..."
    echo ""

    # Detect CUDA support
    BUILD_FEATURES=""
    if [ -e /dev/nvidia0 ] || command -v nvidia-smi &>/dev/null; then
        BUILD_FEATURES="--features cuda"
        echo "  NVIDIA GPU detected — building with CUDA support"
    fi

    cargo build --release -p lumen-cli $BUILD_FEATURES 2>&1 | tail -5
    echo ""
    echo "Build complete."
    echo ""
fi

# ── Run ──────────────────────────────────────────────────────────────────────

# If the first argument doesn't start with '-' and isn't a subcommand,
# treat it as a prompt: ./lumen "Hello" → lumen run --prompt "Hello"
FIRST="${1:-}"

case "$FIRST" in
    run|pull|models|convert|bench|purge|generate-test-model|help|--help|-h|--version|-V)
        # Subcommand or flag — pass through directly
        exec "$BINARY" "$@"
        ;;
    "")
        # No arguments — show help
        exec "$BINARY" --help
        ;;
    -*)
        # Flag without subcommand — pass through (might be --help etc)
        exec "$BINARY" "$@"
        ;;
    *)
        # Bare text — treat as prompt
        # ./lumen "Hello world" → lumen run --prompt "Hello world" --max-tokens 200 --temperature 0
        exec "$BINARY" run --prompt "$@" --max-tokens 200 --temperature 0
        ;;
esac
