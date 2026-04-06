#!/usr/bin/env python3
"""
Lumen Metal Inference Demo
--------------------------
Runs real text inference on Apple Silicon using the Metal GPU backend.

Prerequisites:
  1. Build Lumen:  cargo build --release
  2. Convert model: ./target/release/lumen convert --input model.gguf --output model.lbc
  3. Install tokenizer: pip install transformers

Usage:
  python demo/metal_inference.py --model /tmp/lumen-bench/tinyllama-1.1b-q8_0.lbc \
      --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --prompt "The capital of France is" \
      --max-tokens 64
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def find_lumen_binary():
    """Find the Lumen binary in the repo."""
    repo = Path(__file__).resolve().parent.parent
    binary = repo / "target" / "release" / "lumen"
    if binary.exists():
        return str(binary)
    # Try debug build
    binary = repo / "target" / "debug" / "lumen"
    if binary.exists():
        return str(binary)
    return None


def tokenize(text, tokenizer_name):
    """Tokenize text using HuggingFace transformers."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    # Apply chat template for instruct models
    msgs = [{"role": "user", "content": text}]
    try:
        chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(chat_text, add_special_tokens=False)
    except Exception:
        # Fallback for models without chat template
        ids = tok.encode(text, add_special_tokens=False)
    return ids, tok


def detokenize(token_ids, tokenizer):
    """Convert token IDs back to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def run_lumen(binary, model_path, token_ids, max_tokens, temperature, profile=False):
    """Run Lumen inference and return generated token IDs."""
    tokens_str = " ".join(str(t) for t in token_ids)

    cmd = [
        binary, "run",
        "--model", model_path,
        "--tokens", tokens_str,
        "--max-tokens", str(max_tokens),
        "--metal",
        "--temperature", str(temperature),
    ]
    if profile:
        cmd.append("--profile")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"Lumen error (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr[:2000], file=sys.stderr)
        sys.exit(1)

    # Parse generated tokens from output
    # Format: "Generated tokens: [123, 456, 789, ...]"
    output = result.stdout + result.stderr
    match = re.search(r"Generated tokens:\s*\[([^\]]*)\]", output)
    if not match:
        print("Could not parse Lumen output:", file=sys.stderr)
        print(output[:2000], file=sys.stderr)
        sys.exit(1)

    gen_tokens = [int(t.strip()) for t in match.group(1).split(",") if t.strip()]

    # Extract metrics if available
    metrics = {}
    for line in output.split("\n"):
        if "Prefill:" in line and "tok/s" in line:
            m = re.search(r"Prefill:\s+([\d.]+)\s+tok/s", line)
            if m:
                metrics["prefill_tps"] = float(m.group(1))
        if "Decode:" in line and "tok/s" in line:
            m = re.search(r"Decode:\s+([\d.]+)\s+tok/s", line)
            if m:
                metrics["decode_tps"] = float(m.group(1))

    return gen_tokens, metrics, output


def main():
    parser = argparse.ArgumentParser(description="Lumen Metal Inference Demo")
    parser.add_argument("--model", required=True, help="Path to .lbc model file")
    parser.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer name or path")
    parser.add_argument("--prompt", default="The meaning of life is", help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0=greedy)")
    parser.add_argument("--profile", action="store_true", help="Show per-layer timing")
    parser.add_argument("--binary", help="Path to lumen binary (auto-detected if not set)")
    args = parser.parse_args()

    # Find binary
    binary = args.binary or find_lumen_binary()
    if not binary or not os.path.exists(binary):
        print("Error: Lumen binary not found. Build with: cargo build --release", file=sys.stderr)
        sys.exit(1)

    # Verify model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    # Tokenize
    print(f"Model:     {args.model}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Prompt:    {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print()

    prompt_ids, tokenizer = tokenize(args.prompt, args.tokenizer)
    print(f"Prompt tokens ({len(prompt_ids)}): {prompt_ids[:20]}{'...' if len(prompt_ids) > 20 else ''}")
    print()

    # Run inference
    print("Running Lumen Metal inference...")
    print("-" * 60)

    gen_tokens, metrics, _ = run_lumen(
        binary, args.model, prompt_ids, args.max_tokens, args.temperature, args.profile
    )

    # Detokenize
    gen_text = detokenize(gen_tokens, tokenizer)

    print("-" * 46)
    print()
    print(" _   _   _ __  __ ___ _  _ ")
    print("| | | | | |  \\/  | __| \\| |")
    print("| |_| |_| | |\\/| | _|| .` |")
    print("|____\\___/|_|  |_|___|_|\\_|")
    print()
    print(" Rust LLM Inference Engine")
    print()
    print("\u2500" * 46)
    print("  Source    github.com/faisalmumtaz89/Lumen")
    print("  Engine    Lumen v0.1 (Rust + Metal)")
    print("  Backend   Metal (Apple Silicon GPU)")
    print(f"  Model     {args.model}")
    print(f"  Tokens    {len(prompt_ids)} prompt, {len(gen_tokens)} generated")
    if metrics.get("prefill_tps"):
        print(f"  Prefill   {metrics['prefill_tps']:.1f} tok/s")
    if metrics.get("decode_tps"):
        print(f"  Decode    {metrics['decode_tps']:.1f} tok/s")
    print("\u2500" * 46)
    print()
    print(f"Prompt: {args.prompt}")
    print()
    print(gen_text)
    print()


if __name__ == "__main__":
    main()
