#!/usr/bin/env python3
"""
Lumen CUDA Inference Demo
--------------------------
Runs text inference on NVIDIA GPU using the CUDA backend.

Local mode (default, requires NVIDIA GPU + built Lumen):
  Prerequisites:
    cargo build --release --features cuda -p lumen-cli

  Usage (model registry -- auto-downloads on first run):
    python examples/cuda_inference.py
    python examples/cuda_inference.py --model qwen2.5-3b:q8_0 --prompt "Write a haiku about coding"
    python examples/cuda_inference.py --model llama-8b:q8_0 --prompt "Explain gravity" --max-tokens 100

  Usage (local .lbc file):
    python examples/cuda_inference.py --model /path/to/model.lbc --prompt "The capital of France is"

Modal mode (serverless A100, no local GPU needed):
  Prerequisites:
    pip install modal
    modal setup

  Usage:
    modal run examples/cuda_inference.py
    modal run examples/cuda_inference.py --model qwen2.5-3b --prompt "Write a haiku about coding"
    modal run examples/cuda_inference.py --model llama-8b --prompt "Explain gravity" --max-tokens 100

The script invokes `lumen run` which:
  - Auto-detects CUDA backend on Linux with NVIDIA GPU (no --cuda flag needed)
  - Uses the built-in BPE tokenizer embedded in LBC v3 files
  - Prints generated text to stdout, diagnostics to stderr with --verbose
"""

import os
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def find_lumen_binary():
    """Find the Lumen binary in the repo."""
    repo = Path(__file__).resolve().parent.parent
    for profile in ("release", "debug"):
        binary = repo / "target" / profile / "lumen"
        if binary.exists():
            return str(binary)
    return None


def detect_gpu():
    """Detect NVIDIA GPU via nvidia-smi. Returns description or None."""
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
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


def print_banner(*, gpu, model, prefill_tps, decode_tps, prompt, gen_text):
    """Print the Lumen branded output."""
    print()
    print(" _   _   _ __  __ ___ _  _ ")
    print("| | | | | |  \\/  | __| \\| |")
    print("| |_| |_| | |\\/| | _|| .` |")
    print("|____\\___/|_|  |_|___|_|\\_|")
    print()
    print(" Rust LLM Inference Engine")
    print()
    print("\u2500" * 46)
    print("  Engine    Lumen (Rust + CUDA)")
    print("  Backend   CUDA")
    if gpu:
        print(f"  GPU       {gpu}")
    print(f"  Model     {model}")
    if prefill_tps > 0:
        print(f"  Prefill   {prefill_tps:.1f} tok/s")
    if decode_tps > 0:
        print(f"  Decode    {decode_tps:.1f} tok/s")
    print("\u2500" * 46)
    print()
    print(f"Prompt: {prompt}")
    print()
    print(gen_text)
    print()


# ---------------------------------------------------------------------------
# Local mode
# ---------------------------------------------------------------------------

def main_local():
    """Local CUDA inference entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Lumen CUDA Inference Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Models from the registry (auto-download on first run):\n"
            "  tinyllama:q8_0, tinyllama:q4_0\n"
            "  qwen2.5-3b:q8_0, qwen2.5-3b:q4_0\n"
            "  llama-8b:q8_0\n"
            "\n"
            "Run 'lumen models' to see all available models.\n"
            "\n"
            "For Modal serverless GPU: modal run examples/cuda_inference.py"
        ),
    )
    parser.add_argument("--model", default="qwen2.5-3b:q8_0",
                        help="Model name from registry (e.g. qwen2.5-3b:q8_0) or path to .lbc file")
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
        print("Build with: cargo build --release --features cuda -p lumen-cli", file=sys.stderr)
        sys.exit(1)

    # Detect GPU
    gpu = detect_gpu()
    if gpu:
        print(f"GPU:         {gpu}")
    else:
        print("Warning: nvidia-smi not found, cannot detect GPU", file=sys.stderr)

    print(f"Model:       {args.model}")
    print(f"Prompt:      {args.prompt}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print()
    print("Running Lumen CUDA inference...")
    print("-" * 46)

    generated_text, metrics = run_inference(
        binary, args.model, args.prompt, args.max_tokens, args.temperature,
        verbose=args.verbose,
    )

    print_banner(
        gpu=gpu,
        model=args.model,
        prefill_tps=metrics.get("prefill_tps", 0.0),
        decode_tps=metrics.get("decode_tps", 0.0),
        prompt=args.prompt,
        gen_text=generated_text,
    )


# ---------------------------------------------------------------------------
# Modal mode (serverless A100 -- only loaded when modal is installed)
# ---------------------------------------------------------------------------

# Modal model presets: maps short names to registry model + quant pairs.
# The remote build uses `lumen run <model>:<quant> --prompt "..."` which
# auto-downloads and converts the model via the built-in registry.
MODAL_MODELS = {
    "qwen2.5-3b": {
        "display": "Qwen2.5 3B Instruct",
        "registry": "qwen2.5-3b:q8_0",
    },
    "qwen2.5-3b:f16": {
        "display": "Qwen2.5 3B Instruct F16",
        "registry": "qwen2.5-3b:f16",
    },
    "llama-8b": {
        "display": "Llama 3.1 8B Instruct",
        "registry": "llama-8b:q8_0",
    },
    "llama-8b:f16": {
        "display": "Llama 3.1 8B Instruct F16",
        "registry": "llama-8b:f16",
    },
    "tinyllama": {
        "display": "TinyLlama 1.1B Chat",
        "registry": "tinyllama:q8_0",
    },
    "qwen3.5-9b": {
        "display": "Qwen3.5 9B",
        "registry": "qwen3.5-9b:q8_0",
    },
}

REMOTE_REPO = "/root/lumen"
MOUNT_IGNORE = ["target", ".git", ".claude", "*.gguf", "*.lbc", "bench/results",
                ".DS_Store", ".codecompass"]

try:
    import modal

    if modal.is_local():
        # Build a CUDA-capable image. Requires a modal/image.py in the repo
        # that defines `cuda_build_image`. If you do not have one, create a
        # minimal Modal image with CUDA toolkit and Rust toolchain.
        _image_path = Path(__file__).resolve().parent.parent / "modal" / "image.py"
        if _image_path.exists():
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location("lumen_image", str(_image_path))
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            lumen_image = _mod.cuda_build_image
        else:
            # Fallback: user must define their own image or create modal/image.py
            raise FileNotFoundError(
                f"Modal image definition not found at {_image_path}. "
                "Create modal/image.py with a `cuda_build_image` definition, "
                "or see the README for Modal setup instructions."
            )

        _repo_root = str(Path(__file__).resolve().parent.parent)
        lumen_image = (
            lumen_image
            .add_local_dir(_repo_root, remote_path=REMOTE_REPO, ignore=MOUNT_IGNORE)
        )
    else:
        lumen_image = modal.Image.debian_slim()

    app = modal.App("lumen-cuda-inference")

    @app.function(image=lumen_image, gpu="a100-80gb", timeout=600)
    def run_modal_inference(model_key: str, prompt: str, max_tokens: int, temperature: float):
        """Build Lumen on the remote GPU instance and run inference."""
        os.chdir(REMOTE_REPO)

        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True).strip()
        print(f"GPU: {gpu}")

        # Build Lumen with CUDA support
        print("\n=== Building Lumen (CUDA) ===")
        r = subprocess.run(
            "cargo build --release --features cuda -p lumen-cli 2>&1",
            shell=True, capture_output=True, text=True, timeout=600,
        )
        build_out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            print(build_out[-3000:])
            return {"error": f"Build failed (exit {r.returncode})"}
        print("Build OK")
        lumen_bin = "./target/release/lumen"
        if not os.path.exists(lumen_bin):
            return {"error": "Binary not found after build"}

        mdef = MODAL_MODELS[model_key]

        # Run inference via CLI (auto-downloads + converts via registry)
        print(f"\n=== Running Inference: {mdef['display']} ===")
        print(f"Prompt: {prompt}")
        print(f"Max tokens: {max_tokens}")

        cmd = [
            lumen_bin, "run",
            mdef["registry"],
            prompt,
            "--max-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--verbose",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if r.returncode != 0:
            return {"error": f"Inference failed (exit {r.returncode})",
                    "output": (r.stdout + "\n" + r.stderr)[-2000:]}

        generated_text = r.stdout.rstrip("\n")

        # Parse metrics from stderr
        decode_tps = 0.0
        prefill_tps = 0.0
        for line in r.stderr.split("\n"):
            m = re.search(r"Decode:\s+([\d.]+)\s+tok/s", line)
            if m:
                decode_tps = float(m.group(1))
            m = re.search(r"Prefill:\s+([\d.]+)\s+tok/s", line)
            if m:
                prefill_tps = float(m.group(1))

        return {
            "model": mdef["display"],
            "gpu": gpu,
            "prompt": prompt,
            "generated_text": generated_text,
            "decode_tps": decode_tps,
            "prefill_tps": prefill_tps,
        }

    @app.local_entrypoint()
    def modal_main(
        model: str = "qwen2.5-3b",
        prompt: str = "The meaning of life is",
        max_tokens: int = 64,
        temperature: float = 0.0,
    ):
        if model not in MODAL_MODELS:
            print(f"Unknown model: {model}. Available: {list(MODAL_MODELS.keys())}")
            sys.exit(1)

        print(f"Model: {MODAL_MODELS[model]['display']}")
        print(f"Prompt: {prompt}")
        print(f"Max tokens: {max_tokens}")
        print()

        result = run_modal_inference.remote(model, prompt, max_tokens, temperature)

        if "error" in result:
            print(f"ERROR: {result['error']}")
            if "output" in result:
                print(result["output"][-1000:])
            sys.exit(1)

        print_banner(
            gpu=result["gpu"],
            model=result["model"],
            prefill_tps=result["prefill_tps"],
            decode_tps=result["decode_tps"],
            prompt=result["prompt"],
            gen_text=result["generated_text"],
        )

except (ImportError, FileNotFoundError):
    pass


# ---------------------------------------------------------------------------
# Entry point (local mode)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main_local()
