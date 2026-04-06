#!/usr/bin/env python3
"""
Lumen CUDA Inference Demo (Modal)
----------------------------------
Runs real text inference on NVIDIA GPU via Modal serverless.

Prerequisites:
  pip install modal transformers
  modal setup

Usage:
  modal run demo/cuda_inference.py
  modal run demo/cuda_inference.py --model qwen2.5-3b --prompt "Write a haiku about coding"
  modal run demo/cuda_inference.py --model llama-8b --prompt "Explain gravity" --max-tokens 100
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import modal

REMOTE_REPO = "/root/lumen"
MODEL_DIR = "/tmp/models"

MODELS = {
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

MOUNT_IGNORE = ["target", ".git", ".claude", "*.gguf", "*.lbc", "bench/results",
                ".DS_Store", ".codecompass"]

# Build image
if modal.is_local():
    import importlib.util as _ilu
    _image_path = Path(__file__).resolve().parent.parent / "modal" / "image.py"
    _spec = _ilu.spec_from_file_location("lumen_image", str(_image_path))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _repo_root = str(Path(__file__).resolve().parent.parent)
    lumen_image = (
        _mod.cuda_build_image
        .pip_install("huggingface_hub[cli]", "transformers", "sentencepiece", "protobuf", "jinja2")
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
        print(subprocess.run("ls -la target/release/lumen* 2>&1 || echo 'No lumen binaries found'",
            shell=True, capture_output=True, text=True).stdout)
        return {"error": "Binary not found after build"}

    mdef = MODELS[model_key]

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
    print(f"\n=== Tokenizing ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mdef["hf_tokenizer"], trust_remote_code=True)
    # Apply chat template for instruct models
    msgs = [{"role": "user", "content": prompt}]
    chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(chat_text, add_special_tokens=False)
    if not prompt_ids:
        return {"error": "Empty prompt after tokenization"}
    tokens_str = " ".join(str(t) for t in prompt_ids)
    print(f"Prompt: {prompt}")
    print(f"Tokens ({len(prompt_ids)}): {prompt_ids[:20]}{'...' if len(prompt_ids) > 20 else ''}")

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

    # Print full Lumen output for debugging
    print("=== Lumen Raw Output (last 3000 chars) ===")
    print(output[-3000:])
    print("=== End Lumen Output ===")

    if r.returncode != 0:
        return {"error": f"Inference failed (exit {r.returncode})", "output": output[-2000:]}

    # Parse generated tokens
    match = re.search(r"Generated tokens:\s*\[([^\]]*)\]", output)
    if not match or not match.group(1).strip():
        return {"error": "No tokens generated", "output": output[-2000:]}

    gen_tokens = [int(t.strip()) for t in match.group(1).split(",") if t.strip()]
    print(f"  Raw token IDs (first 20): {gen_tokens[:20]}")

    # Parse metrics
    decode_tps = 0.0
    for line in output.split("\n"):
        m = re.search(r"Decode:\s+([\d.]+)\s+tok/s", line)
        if m:
            decode_tps = float(m.group(1))

    # Detokenize
    gen_text = tok.decode(gen_tokens, skip_special_tokens=True)
    full_text = tok.decode(prompt_ids + gen_tokens, skip_special_tokens=True)

    return {
        "model": mdef["display"],
        "gpu": gpu,
        "prompt": prompt,
        "generated_text": gen_text,
        "full_text": full_text,
        "tokens_generated": len(gen_tokens),
        "decode_tps": decode_tps,
    }


@app.local_entrypoint()
def main(
    model: str = "qwen2.5-3b",
    prompt: str = "The meaning of life is",
    max_tokens: int = 64,
    temperature: float = 0.0,
):
    if model not in MODELS:
        print(f"Unknown model: {model}. Available: {list(MODELS.keys())}")
        sys.exit(1)

    print(f"Model: {MODELS[model]['display']}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print()

    result = run_inference.remote(model, prompt, max_tokens, temperature)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        if "output" in result:
            print(result["output"][-1000:])
        sys.exit(1)

    print("=" * 60)
    print(f"GPU: {result['gpu']}")
    print(f"Model: {result['model']}")
    if result['decode_tps'] > 0:
        print(f"Tokens: {result['tokens_generated']} @ {result['decode_tps']:.1f} tok/s")
    else:
        print(f"Tokens: {result['tokens_generated']}")
    print("=" * 60)
    print()
    print(f"PROMPT: {result['prompt']}")
    print(f"OUTPUT: {result['generated_text']}")
    print()
    print("FULL TEXT:")
    print(result["full_text"])
