#!/usr/bin/env python3
"""
Lumen CUDA Inference Demo
--------------------------
Runs real text inference on NVIDIA GPU using the CUDA backend.

Local mode (default, requires NVIDIA GPU + built Lumen):
  Prerequisites:
    cargo build --release --features cuda -p lumen-cli
    pip install transformers

  Usage:
    python demo/cuda_inference.py --model /path/to/model.lbc \\
        --tokenizer Qwen/Qwen2.5-3B-Instruct \\
        --prompt "Write a haiku about coding"

Modal mode (serverless A100, no local GPU needed):
  Prerequisites:
    pip install modal transformers
    modal setup

  Usage:
    modal run demo/cuda_inference.py
    modal run demo/cuda_inference.py --model qwen2.5-3b --prompt "Write a haiku about coding"
    modal run demo/cuda_inference.py --model llama-8b --prompt "Explain gravity" --max-tokens 100
"""

import argparse
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
    binary = repo / "target" / "release" / "lumen"
    if binary.exists():
        return str(binary)
    binary = repo / "target" / "debug" / "lumen"
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


def tokenize(text, tokenizer_name):
    """Tokenize text using HuggingFace transformers."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    msgs = [{"role": "user", "content": text}]
    try:
        chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(chat_text, add_special_tokens=False)
    except Exception:
        ids = tok.encode(text, add_special_tokens=False)
    return ids, tok


def detokenize(token_ids, tokenizer):
    """Convert token IDs back to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def print_banner(*, gpu, model, prompt_tokens, gen_tokens, prefill_tps,
                 decode_tps, prompt, gen_text):
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
    print("  Source    github.com/faisalmumtaz89/Lumen")
    print("  Engine    Lumen v0.1 (Rust + CUDA)")
    print("  Backend   CUDA")
    if gpu:
        print(f"  GPU       {gpu}")
    print(f"  Model     {model}")
    print(f"  Tokens    {prompt_tokens} prompt, {gen_tokens} generated")
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

def run_lumen(binary, model_path, token_ids, max_tokens, temperature, profile=False):
    """Run Lumen CUDA inference and return generated token IDs."""
    tokens_str = " ".join(str(t) for t in token_ids)

    cmd = [
        binary, "run",
        "--model", model_path,
        "--tokens", tokens_str,
        "--max-tokens", str(max_tokens),
        "--cuda",
        "--temperature", str(temperature),
    ]
    if profile:
        cmd.append("--profile")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"Lumen error (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr[:2000], file=sys.stderr)
        sys.exit(1)

    output = result.stdout + result.stderr
    match = re.search(r"Generated tokens:\s*\[([^\]]*)\]", output)
    if not match:
        print("Could not parse Lumen output:", file=sys.stderr)
        print(output[:2000], file=sys.stderr)
        sys.exit(1)

    gen_tokens = [int(t.strip()) for t in match.group(1).split(",") if t.strip()]

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

    return gen_tokens, metrics


def main_local():
    """Local CUDA inference entry point."""
    parser = argparse.ArgumentParser(
        description="Lumen CUDA Inference Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="For Modal serverless GPU: modal run demo/cuda_inference.py")
    parser.add_argument("--model", required=True, help="Path to .lbc model file")
    parser.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer name or path")
    parser.add_argument("--prompt", default="The meaning of life is", help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy)")
    parser.add_argument("--profile", action="store_true", help="Show per-layer timing")
    parser.add_argument("--binary", help="Path to lumen binary (auto-detected if not set)")
    args = parser.parse_args()

    # Find binary
    binary = args.binary or find_lumen_binary()
    if not binary or not os.path.exists(binary):
        print("Error: Lumen binary not found. Build with: "
              "cargo build --release --features cuda -p lumen-cli", file=sys.stderr)
        sys.exit(1)

    # Verify model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    # Detect GPU
    gpu = detect_gpu()
    if gpu:
        print(f"GPU:         {gpu}")
    else:
        print("Warning: nvidia-smi not found, cannot detect GPU", file=sys.stderr)

    print(f"Model:       {args.model}")
    print(f"Tokenizer:   {args.tokenizer}")
    print(f"Prompt:      {args.prompt}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print()

    prompt_ids, tokenizer = tokenize(args.prompt, args.tokenizer)
    print(f"Prompt tokens ({len(prompt_ids)}): "
          f"{prompt_ids[:20]}{'...' if len(prompt_ids) > 20 else ''}")
    print()

    # Run inference
    print("Running Lumen CUDA inference...")
    print("-" * 46)

    gen_tokens, metrics = run_lumen(
        binary, args.model, prompt_ids, args.max_tokens, args.temperature, args.profile
    )

    gen_text = detokenize(gen_tokens, tokenizer)

    print_banner(
        gpu=gpu,
        model=args.model,
        prompt_tokens=len(prompt_ids),
        gen_tokens=len(gen_tokens),
        prefill_tps=metrics.get("prefill_tps", 0.0),
        decode_tps=metrics.get("decode_tps", 0.0),
        prompt=args.prompt,
        gen_text=gen_text,
    )


# ---------------------------------------------------------------------------
# Modal mode (only loaded when modal is installed)
# ---------------------------------------------------------------------------

MODAL_MODELS = {
    "qwen2.5-3b": {
        "display": "Qwen2.5 3B Instruct",
        "hf_tokenizer": "Qwen/Qwen2.5-3B-Instruct",
        "gguf_repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "gguf_file": "Qwen2.5-3B-Instruct-Q8_0.gguf",
    },
    "llama-8b": {
        "display": "Llama 3.1 8B Instruct",
        "hf_tokenizer": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "gguf_repo": "mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF",
        "gguf_file": "Meta-Llama-3.1-8B-Instruct.Q8_0.gguf",
    },
}

REMOTE_REPO = "/root/lumen"
MODEL_DIR = "/tmp/models"
MOUNT_IGNORE = ["target", ".git", ".claude", "*.gguf", "*.lbc", "bench/results",
                ".DS_Store", ".codecompass"]

try:
    import modal

    if modal.is_local():
        import importlib.util as _ilu
        _image_path = Path(__file__).resolve().parent.parent / "modal" / "image.py"
        if not _image_path.exists():
            raise FileNotFoundError(
                f"Modal image definition not found at {_image_path}. "
                "Create modal/image.py with cuda_build_image definition."
            )
        _spec = _ilu.spec_from_file_location("lumen_image", str(_image_path))
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _repo_root = str(Path(__file__).resolve().parent.parent)
        lumen_image = (
            _mod.cuda_build_image
            .pip_install("huggingface_hub[cli]", "transformers", "sentencepiece",
                         "protobuf", "jinja2")
            .add_local_dir(_repo_root, remote_path=REMOTE_REPO, ignore=MOUNT_IGNORE)
        )
    else:
        lumen_image = modal.Image.debian_slim()

    app = modal.App("lumen-cuda-inference")

    @app.function(image=lumen_image, gpu="a100-80gb", timeout=600)
    def run_inference(model_key: str, prompt: str, max_tokens: int, temperature: float):
        """Build Lumen, download model, tokenize, run CLI inference, detokenize."""
        os.chdir(REMOTE_REPO)

        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True).strip()
        print(f"GPU: {gpu}")

        # Build
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
            print(f"Binary not found at {lumen_bin}")
            print("Checking target directory:")
            print(subprocess.run(
                "ls -la target/release/lumen* 2>&1 || echo 'No lumen binaries found'",
                shell=True, capture_output=True, text=True).stdout)
            return {"error": "Binary not found after build"}

        mdef = MODAL_MODELS[model_key]

        # Download GGUF
        print(f"\n=== Downloading {mdef['display']} ===")
        from huggingface_hub import hf_hub_download
        os.makedirs(MODEL_DIR, exist_ok=True)
        gguf_path = hf_hub_download(
            repo_id=mdef["gguf_repo"], filename=mdef["gguf_file"], local_dir=MODEL_DIR,
        )
        print(f"Downloaded: {gguf_path}")

        # Convert to LBC
        lbc_path = os.path.join(MODEL_DIR, f"{model_key}-q8_0.lbc")
        if not os.path.exists(lbc_path):
            print("\n=== Converting GGUF -> LBC ===")
            r = subprocess.run(
                [lumen_bin, "convert", "--input", gguf_path, "--output", lbc_path],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode != 0:
                return {"error": f"Convert failed: {(r.stdout + r.stderr)[-1000:]}"}
        print(f"LBC: {lbc_path}")

        # Tokenize
        print("\n=== Tokenizing ===")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(mdef["hf_tokenizer"], trust_remote_code=True)
        msgs = [{"role": "user", "content": prompt}]
        chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok.encode(chat_text, add_special_tokens=False)
        if not prompt_ids:
            return {"error": "Empty prompt after tokenization"}
        tokens_str = " ".join(str(t) for t in prompt_ids)
        print(f"Prompt: {prompt}")
        print(f"Tokens ({len(prompt_ids)}): "
              f"{prompt_ids[:20]}{'...' if len(prompt_ids) > 20 else ''}")

        # Run Lumen CLI with --cuda
        print(f"\n=== Running Inference ({max_tokens} tokens) ===")
        cmd = [
            lumen_bin, "run",
            "--model", lbc_path,
            "--tokens", tokens_str,
            "--max-tokens", str(max_tokens),
            "--cuda",
            "--temperature", str(temperature),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = r.stdout + "\n" + r.stderr

        # Debug output available via --debug flag or LUMEN_DEBUG env var
        if os.environ.get("LUMEN_DEBUG"):
            print("=== Lumen Raw Output (last 3000 chars) ===")
            print(output[-3000:])
            print("=== End Lumen Output ===")

        if r.returncode != 0:
            return {"error": f"Inference failed (exit {r.returncode})",
                    "output": output[-2000:]}

        # Parse generated tokens
        match = re.search(r"Generated tokens:\s*\[([^\]]*)\]", output)
        if not match or not match.group(1).strip():
            return {"error": "No tokens generated", "output": output[-2000:]}

        gen_tokens = [int(t.strip()) for t in match.group(1).split(",") if t.strip()]

        # Parse metrics
        decode_tps = 0.0
        prefill_tps = 0.0
        for line in output.split("\n"):
            m = re.search(r"Decode:\s+([\d.]+)\s+tok/s", line)
            if m:
                decode_tps = float(m.group(1))
            m = re.search(r"Prefill:\s+([\d.]+)\s+tok/s", line)
            if m:
                prefill_tps = float(m.group(1))

        gen_text = tok.decode(gen_tokens, skip_special_tokens=True)

        return {
            "model": mdef["display"],
            "gpu": gpu,
            "prompt": prompt,
            "generated_text": gen_text,
            "tokens_generated": len(gen_tokens),
            "prompt_tokens": len(prompt_ids),
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

        result = run_inference.remote(model, prompt, max_tokens, temperature)

        if "error" in result:
            print(f"ERROR: {result['error']}")
            if "output" in result:
                print(result["output"][-1000:])
            sys.exit(1)

        print_banner(
            gpu=result["gpu"],
            model=result["model"],
            prompt_tokens=result["prompt_tokens"],
            gen_tokens=result["tokens_generated"],
            prefill_tps=result["prefill_tps"],
            decode_tps=result["decode_tps"],
            prompt=result["prompt"],
            gen_text=result["generated_text"],
        )

except ImportError:
    pass


# ---------------------------------------------------------------------------
# Entry point (local mode)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main_local()
