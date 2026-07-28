#!/usr/bin/env python3
"""Per-family int8 tolerance, gated on LONG-FORM quality.

The first bisect gated on three short prompts and cleared every family. The
AH-11 campaign then found four candidate-only failures — all long-form
degeneration (DD-REP / DD-SPAM on 2000+ token generations), reproducible across
three fresh processes, absent from both the v0.5.0 and all-F32 controls.

A three-prompt gate cannot see that failure mode. It proves output is not
garbage; this degradation is fluent text that collapses into repetition after a
thousand tokens. So the same arms are re-run against the gates that actually
detect it: GQ-002 (medium), GQ-004 (verylong, 15 prompts to 3072 tokens) and
GQ-004b (the degeneration traps).

An arm is safe only if it matches the all-F32 control item-for-item. "Fewer
failures than the full policy" is not safe; a family that costs one long-form
generation is a family that ships a regression.
"""
import json
import subprocess
import time

import modal

SHIP_SRC = "/Users/faisalmumtaz/Documents/GitHub/lumen-9bq4"
IGNORE = ["target", "**/target", ".git", "**/*.gguf", "**/*.lbc",
          "**/__pycache__", "**/*.pyc", ".DS_Store", "scratchpad"]

GATES = "GQ-002,GQ-004,GQ-004b"

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ca-certificates", "curl", "git", "build-essential", "pkg-config")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                  "| sh -s -- -y --profile minimal --default-toolchain stable")
    .env({"PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_dir(SHIP_SRC, "/src/ship", copy=True, ignore=IGNORE)
    .run_commands(
        "mkdir -p /opt/lumen/bin",
        "cd /src/ship && cargo build --release --locked -p lumen-server "
        "--features bin,cuda,activation-probe",
        "cp /src/ship/target/release/lumen-server /opt/lumen/bin/server-probe",
        "/opt/lumen/bin/server-probe --version")
    .env({"LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"})
)

app = modal.App("lumen-9bq4-longform-bisect")
vol = modal.Volume.from_name("lumen-prod-gguf")
VOL = "/gguf"
LBC = f"{VOL}/final-bench-staged/9b/9b-q4_0.lbc"
OUT = f"{VOL}/9bq4-longform-bisect"

ARMS = [
    ("none",                   ""),
    ("ffn_down",               "ffn_down"),
    ("ffn_down+ffn_gate_up",   "ffn_down,ffn_gate_up"),
    ("ffn_down+attn_qkv",      "ffn_down,attn_qkv"),
    ("ffn_down+gdn_qkv",       "ffn_down,gdn_qkv"),
    ("ffn_down+gdn_attn_gate", "ffn_down,gdn_attn_gate"),
    ("all_but_wo",             "ffn_down,attn_qkv,ffn_gate_up,gdn_qkv,gdn_attn_gate"),
]


def run_arm(tag, probe, port):
    import os
    env = dict(os.environ, LUMEN_Q4_ACT_PROBE=probe)
    cell = tag.replace("+", "_")
    cmd = ["python3", "/src/ship/quality/run_suite.py",
           "--model", LBC, "--backend", "cuda", "--cell", f"lf-{cell}",
           "--port", str(port), "--gates", GATES,
           "--server-bin", "/opt/lumen/bin/server-probe",
           "--ctx", "16384",
           "--out", f"{OUT}/cell-lf-{cell}.json"]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    print(f"\n===== {tag}  (probe={probe!r}, rc={p.returncode}, "
          f"{time.time()-t0:.0f}s) =====", flush=True)
    for line in p.stdout.splitlines():
        if "| GQ-" in line or "q4_act_plan" in line:
            print("  " + line.rstrip()[:210], flush=True)
    if p.returncode not in (0, 1):
        print("  STDERR:", p.stderr[-2500:], flush=True)
    return p.stdout


@app.function(image=image, gpu="A100-80GB", volumes={VOL: vol}, timeout=32400)
def bisect_lf():
    vol.reload()
    subprocess.run(f"mkdir -p {OUT}", shell=True)
    print("=== per-family int8 tolerance, LONG-FORM gates ===", flush=True)
    print(f"gates: {GATES}", flush=True)
    out = {}
    for i, (tag, probe) in enumerate(ARMS):
        out[tag] = run_arm(tag, probe, 8600 + i)
        vol.commit()
    return out


@app.local_entrypoint()
def main():
    bisect_lf.remote()
