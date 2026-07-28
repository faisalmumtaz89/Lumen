#!/usr/bin/env python3
"""GQ-005: reference greedy fidelity vs llama.cpp on the SAME GGUF.

The suite calls this FOUNDATIONAL and the runner has been deferring it to a
companion driver that did not exist. It is the gate that most directly tests
this change: activation precision alters the token stream, and llama.cpp
running the identical weights is the only reference outside our own kernels.

Lumen and llama.cpp will not agree token-for-token — different kernels,
different reduction orders — so absolute agreement is NOT the criterion. The
question is comparative:

    does the CANDIDATE diverge from llama.cpp sooner than v0.5.0 does?

Three Lumen arms (prod / all-F32 oracle / candidate) and one llama.cpp
reference, same container, same GPU, same GGUF, greedy, seed 42. For each
prompt we record the index of the first differing token and the agreement rate
over the compared prefix. A candidate that diverges systematically earlier than
prod is a fidelity regression; one that matches or beats prod is not.

The oracle arm separates "this branch's kernels" from "this branch's activation
policy": oracle and candidate share everything except activation precision.
"""
import json
import re
import subprocess
import time

import modal

BASE_SRC = ("/private/tmp/claude-501/-Users-faisalmumtaz-Documents-GitHub-Lumen/"
            "d97d5321-f7a8-4550-a281-8534b9f603a3/scratchpad/lumen-main-ref")
SHIP_SRC = "/Users/faisalmumtaz/Documents/GitHub/lumen-9bq4"
LLAMACPP_PIN = "3b53219361a61b53e7741c479b81b755ec6096b1"  # b10032, the board pin
IGNORE = ["target", "**/target", ".git", "**/*.gguf", "**/*.lbc",
          "**/__pycache__", "**/*.pyc", ".DS_Store", "scratchpad"]

NTOK = 128

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ca-certificates", "curl", "git", "build-essential", "cmake",
                 "pkg-config", "libcurl4-openssl-dev")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                  "| sh -s -- -y --profile minimal --default-toolchain stable")
    .env({"PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
          "CUDA_HOME": "/usr/local/cuda"})
    .run_commands(
        "mkdir -p /opt/lumen/bin",
        "git clone https://github.com/ggerganov/llama.cpp.git /opt/llamacpp-src",
        f"cd /opt/llamacpp-src && git checkout {LLAMACPP_PIN}",
        "cd /opt/llamacpp-src && git rev-parse HEAD > /opt/lumen/llamacpp_sha.txt",
        "cd /opt/llamacpp-src && cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release"
        " -DCMAKE_CUDA_ARCHITECTURES='80-real'"
        " -DGGML_STATIC=ON -DBUILD_SHARED_LIBS=OFF"
        " -DCMAKE_EXE_LINKER_FLAGS=-L/usr/local/cuda/lib64/stubs"
        " -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF"
        " && cmake --build build --target llama-cli -j$(nproc)",
        "cp /opt/llamacpp-src/build/bin/llama-cli /opt/lumen/bin/")
    .add_local_dir(BASE_SRC, "/src/base", copy=True, ignore=IGNORE)
    .add_local_dir(SHIP_SRC, "/src/ship", copy=True, ignore=IGNORE)
    .run_commands(
        # no `| tail`: a pipeline hides a failed build behind tail's exit status
        "cd /src/base && cargo build --release --locked -p lumen-cli --bin lumen --features cuda",
        "cp /src/base/target/release/lumen /opt/lumen/bin/lumen-prod",
        "cd /src/ship && cargo build --release --locked -p lumen-cli --bin lumen --features cuda",
        "cp /src/ship/target/release/lumen /opt/lumen/bin/lumen-cand",
        "cd /src/ship && cargo build --release --locked -p lumen-cli --bin lumen "
        "--features cuda,quality-oracle",
        "cp /src/ship/target/release/lumen /opt/lumen/bin/lumen-oracle",
        "for b in prod cand oracle; do /opt/lumen/bin/lumen-$b --version; done")
    .env({"LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"})
)

app = modal.App("lumen-9bq4-gq005")
vol = modal.Volume.from_name("lumen-prod-gguf")
VOL = "/gguf"
LBC = f"{VOL}/final-bench-staged/9b/9b-q4_0.lbc"
GGUF = f"{VOL}/final-bench-staged/9b/Qwen_Qwen3.5-9B-Q4_0.gguf"

PROMPTS = [
    "Explain the process of photosynthesis in detail.",
    "Describe how a bicycle stays upright while moving.",
    "What causes the seasons on Earth?",
    "Explain the difference between TCP and UDP.",
    "Summarize the water cycle.",
    "Explain why the sky appears blue.",
    "Describe how a refrigerator works.",
    "What is the significance of the Rosetta Stone?",
    "Explain how vaccines train the immune system.",
    "Describe the structure of an atom.",
    "Explain what a hash function is and where it is used.",
    "How does a suspension bridge carry its load?",
]


def norm(text):
    """Whitespace-normalised words. Token IDs would be better, but the two
    engines do not expose comparable ID streams from the CLI, and word-level
    first-divergence is enough to rank arms against the SAME reference."""
    return re.sub(r"\s+", " ", text.strip()).split(" ")


def lumen_gen(binary, prompt):
    p = subprocess.run(
        [binary, "run", LBC, "--cuda", "--prompt", prompt, "--max-tokens", str(NTOK),
         "--temperature", "0", "--seed", "42", "--repetition-penalty", "1.0",
         "--repeat-last-n", "0", "--no-think"],
        capture_output=True, text=True, timeout=1200)
    return (p.stdout or "").strip()


def llama_gen(prompt):
    p = subprocess.run(
        f'/opt/lumen/bin/llama-cli -m "{GGUF}" -ngl 99 -n {NTOK} --temp 0 --seed 42 '
        f'--repeat-penalty 1.0 -no-cnv -p {json.dumps(prompt)} 2>/dev/null',
        shell=True, capture_output=True, text=True, timeout=1800)
    out = (p.stdout or "").strip()
    # llama-cli echoes the prompt; drop it so both streams start at generation
    if out.startswith(prompt):
        out = out[len(prompt):].strip()
    return out


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, n
    return n, n


@app.function(image=image, gpu="A100-80GB", volumes={VOL: vol}, timeout=10800)
def gq005():
    vol.reload()
    sha = open("/opt/lumen/llamacpp_sha.txt").read().strip()
    print(f"=== GQ-005 greedy fidelity vs llama.cpp {sha[:12]} ===", flush=True)

    rows = []
    for prompt in PROMPTS:
        ref = norm(llama_gen(prompt))
        if len(ref) < 10:
            print(f"  SKIP (llama.cpp produced {len(ref)} words): {prompt[:50]}", flush=True)
            continue
        rec = {"prompt": prompt, "ref_words": len(ref)}
        for tag, binary in (("prod", "/opt/lumen/bin/lumen-prod"),
                            ("oracle", "/opt/lumen/bin/lumen-oracle"),
                            ("cand", "/opt/lumen/bin/lumen-cand")):
            got = norm(lumen_gen(binary, prompt))
            idx, n = first_divergence(ref, got)
            rec[tag] = {"first_div": idx, "compared": n,
                        "agree_frac": round(idx / n, 4) if n else 0.0}
        rows.append(rec)
        print(f"  {prompt[:44]:46} " + "  ".join(
            f"{t}={rec[t]['first_div']:3}/{rec[t]['compared']:3}"
            for t in ("prod", "oracle", "cand")), flush=True)

    print("\n=== SUMMARY: mean first-divergence index vs llama.cpp ===", flush=True)
    for t in ("prod", "oracle", "cand"):
        vals = [r[t]["first_div"] for r in rows]
        frac = [r[t]["agree_frac"] for r in rows]
        print(f"  {t:7} mean_first_div={sum(vals)/len(vals):6.2f}  "
              f"mean_agree={sum(frac)/len(frac):.4f}  n={len(vals)}", flush=True)
    worse = [r["prompt"][:40] for r in rows if r["cand"]["first_div"] < r["prod"]["first_div"]]
    better = [r["prompt"][:40] for r in rows if r["cand"]["first_div"] > r["prod"]["first_div"]]
    print(f"\n  candidate diverges EARLIER than v0.5.0 on {len(worse)}/{len(rows)} prompts",
          flush=True)
    print(f"  candidate diverges LATER   than v0.5.0 on {len(better)}/{len(rows)} prompts",
          flush=True)
    for w in worse:
        print(f"    earlier: {w}", flush=True)
    return rows


@app.local_entrypoint()
def main():
    gq005.remote()
