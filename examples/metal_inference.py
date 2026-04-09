#!/usr/bin/env python3
"""
Lumen Metal Inference Demo
--------------------------
Runs text inference on Apple Silicon using the Metal GPU backend.

Prerequisites:
  Build Lumen:  cargo build --release -p lumen-cli

Usage (model registry -- auto-downloads on first run):
  python examples/metal_inference.py
  python examples/metal_inference.py --model tinyllama:q8_0 --prompt "Write a haiku about Rust"
  python examples/metal_inference.py --model qwen2.5-3b:q8_0 --prompt "Explain gravity" --max-tokens 100

Usage (local .lbc file):
  python examples/metal_inference.py --model /path/to/model.lbc --prompt "The capital of France is"

The script invokes `lumen run` which:
  - Auto-detects Metal backend on macOS (no --metal flag needed)
  - Uses the built-in BPE tokenizer embedded in LBC v3 files
  - Prints generated text to stdout, diagnostics to stderr with --verbose
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def find_lumen_binary():
    """Find the Lumen binary in the repo."""
    repo = Path(__file__).resolve().parent.parent
    for profile in ("release", "debug"):
        binary = repo / "target" / profile / "lumen"
        if binary.exists():
            return str(binary)
    return None


def run_inference(binary, model, prompt, max_tokens, temperature, verbose=False):
    """Run Lumen inference and return (generated_text, metrics_dict).

    The CLI prints generated text to stdout and metrics to stderr (with --verbose).
    """
    cmd = [
        binary, "run",
        "--model", model,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
    ]
    if verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"Lumen error (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr[:2000], file=sys.stderr)
        sys.exit(1)

    generated_text = result.stdout.rstrip("\n")

    # Parse metrics from stderr (only present with --verbose)
    metrics = {}
    for line in result.stderr.split("\n"):
        m = re.search(r"Prefill:\s+([\d.]+)\s+tok/s", line)
        if m:
            metrics["prefill_tps"] = float(m.group(1))
        m = re.search(r"Decode:\s+([\d.]+)\s+tok/s", line)
        if m:
            metrics["decode_tps"] = float(m.group(1))
        m = re.search(r"Tokenized prompt:\s+(\d+)\s+tokens", line)
        if m:
            metrics["prompt_tokens"] = int(m.group(1))

    return generated_text, metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Lumen Metal Inference Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Models from the registry (auto-download on first run):\n"
            "  tinyllama:q8_0, tinyllama:q4_0\n"
            "  qwen2.5-3b:q8_0, qwen2.5-3b:q4_0\n"
            "  llama-8b:q8_0\n"
            "\n"
            "Run 'lumen models' to see all available models."
        ),
    )
    parser.add_argument("--model", default="tinyllama:q8_0",
                        help="Model name from registry (e.g. tinyllama:q8_0) or path to .lbc file")
    parser.add_argument("--prompt", default="The meaning of life is",
                        help="Text prompt (default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max tokens to generate (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature, 0=greedy (default: %(default)s)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show metrics and diagnostics")
    parser.add_argument("--binary", help="Path to lumen binary (auto-detected if not set)")
    args = parser.parse_args()

    # Find binary
    binary = args.binary or find_lumen_binary()
    if not binary or not os.path.exists(binary):
        print("Error: Lumen binary not found.", file=sys.stderr)
        print("Build with: cargo build --release -p lumen-cli", file=sys.stderr)
        sys.exit(1)

    print(f"Model:       {args.model}")
    print(f"Prompt:      {args.prompt}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print()
    print("Running Lumen Metal inference...")
    print("-" * 46)

    generated_text, metrics = run_inference(
        binary, args.model, args.prompt, args.max_tokens, args.temperature,
        verbose=args.verbose,
    )

    print("-" * 46)
    print()

    if metrics.get("prefill_tps"):
        print(f"  Prefill   {metrics['prefill_tps']:.1f} tok/s")
    if metrics.get("decode_tps"):
        print(f"  Decode    {metrics['decode_tps']:.1f} tok/s")
    if metrics:
        print()

    print(f"Prompt: {args.prompt}")
    print()
    print(generated_text)
    print()


if __name__ == "__main__":
    main()
