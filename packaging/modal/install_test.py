"""Install-test: run Lumen's REAL one-command installer (``curl … | bash``) in a
clean Linux + NVIDIA container and validate it end-to-end.

This complements ``validate_arches.py`` (which validates the binary). Here the
container has NO baked Lumen binary — the installer must detect CUDA, download the
released Linux x86_64 binary from the GitHub release (alias-first, API-tag
fallback), verify its SHA-256, install it, pull a model, and run it on the GPU.

A second, GPU-less function asserts the installer REFUSES on a no-NVIDIA box.

Run (Modal token from env, e.g. in CI):
    modal run packaging/modal/install_test.py --ref install-onboarder
Gate the CI step on the line ``INSTALL VALIDATION PASSED``.
"""

import os
import subprocess
import sys

import modal

RAW = "https://raw.githubusercontent.com/faisalmumtaz89/Lumen"  # + /<ref>/packaging/macos/install.sh

# Clean CUDA-devel image: libnvrtc + libcublas present (the binary dlopens them),
# plus curl/tar/ca-certs for the installer. NO Lumen binary baked in — that is the
# whole point: the installer fetches it.
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ca-certificates", "curl", "tar")
)
app = modal.App("lumen-install-test")


def _sh(cmd):
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)


@app.function(gpu="A100", image=image, timeout=1800)
def install_e2e(ref: str):
    """Full installer path on a real A100: detect → download → install → pull → run."""
    import json
    import time
    import urllib.request

    out = {"arch": "A100", "steps": [], "errors": []}

    def step(name, cond, detail=""):
        ok = bool(cond)
        out["steps"].append({"name": name, "pass": ok, "detail": str(detail)[:300]})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {str(detail)[:200]}", flush=True)
        if not ok:
            out["errors"].append(name)

    url = f"{RAW}/{ref}/packaging/macos/install.sh"
    print(f"=== running: curl -fsSL {url} | bash -s -- --model qwen3.5-9b --quant q8_0 --yes ===", flush=True)
    r = _sh(f"curl -fsSL {url} | bash -s -- --model qwen3.5-9b --quant q8_0 --yes")
    log = r.stdout + r.stderr
    print(log[-5000:], flush=True)

    step("installer exit 0", r.returncode == 0, f"rc={r.returncode}")
    step("detected CUDA backend", "-> cuda" in log)
    step("checksum verified", "checksum OK" in log)
    step(
        "both binaries installed",
        os.path.exists("/usr/local/bin/lumen") and os.path.exists("/usr/local/bin/lumen-server"),
    )
    v = _sh("/usr/local/bin/lumen --version")
    step("lumen --version", v.returncode == 0, v.stdout.strip())
    step("model prepared (pulled+converted)", ("is ready" in log) or ("Already cached" in log))

    # Load the model ONCE in the server, then DET-001 + coherence over HTTP
    # (fast — avoids cold-loading the 9 GB model per `lumen run`).
    env = dict(os.environ)
    env["LUMEN_CUDA_DECODE_DELAY_US"] = "0"
    logf = open("/tmp/srv.log", "w")
    srv = subprocess.Popen(
        ["/usr/local/bin/lumen-server", "--model", "qwen3.5-9b", "--quant", "q8_0",
         "--backend", "cuda", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"],
        stdout=logf, stderr=subprocess.STDOUT, env=env,
    )
    ready = False
    for _ in range(150):  # up to ~5 min (cold NVRTC compile + 9 GB load)
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=3).read()
            ready = True
            break
        except Exception:
            pass
        if srv.poll() is not None:
            break
        time.sleep(2)
    step("server reached /v1/models", ready, "" if ready else open("/tmp/srv.log").read()[-400:])

    def chat(prompt, n=16):
        body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": n, "temperature": 0}).encode()
        req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]

    if ready:
        outs = set()
        for _ in range(20):
            outs.add(chat("Name one color.").strip())
        step("DET-001 (N=20 greedy -> 1 distinct)", len(outs) == 1, f"distinct={len(outs)}")
        ans = chat("What is the capital of France?", n=24)
        step("coherence (Paris)", "paris" in ans.lower(), ans.strip()[:80])

    srv.kill()
    out["ok"] = len(out["errors"]) == 0
    return out


@app.function(image=image, timeout=600)  # NO gpu= → no NVIDIA in the container
def refuse_no_gpu(ref: str):
    """The installer must refuse (non-zero, no binary) on a Linux box with no NVIDIA GPU."""
    url = f"{RAW}/{ref}/packaging/macos/install.sh"
    r = _sh(f"curl -fsSL {url} | bash -s -- --yes")
    log = r.stdout + r.stderr
    print(log[-2000:], flush=True)
    ok = (r.returncode != 0) and ("no NVIDIA GPU" in log) and (not os.path.exists("/usr/local/bin/lumen"))
    print(f"[{'PASS' if ok else 'FAIL'}] no-GPU refusal (rc={r.returncode})", flush=True)
    return {"name": "no-GPU refusal", "ok": ok, "rc": r.returncode}


@app.local_entrypoint()
def main(ref: str = "install-onboarder"):
    print(f"=== Lumen install-test on Modal (ref={ref}) ===", flush=True)
    gpu = install_e2e.remote(ref)
    nogpu = refuse_no_gpu.remote(ref)

    print("\n=== PER-STEP VERDICT (A100 install_e2e) ===")
    for s in gpu["steps"]:
        print(f"  [{'PASS' if s['pass'] else 'FAIL'}] {s['name']}  {s['detail']}")
    print(f"  [{'PASS' if nogpu['ok'] else 'FAIL'}] no-GPU refusal (rc={nogpu['rc']})")

    all_ok = gpu["ok"] and nogpu["ok"]
    print("\n" + ("INSTALL VALIDATION PASSED" if all_ok else "INSTALL VALIDATION FAILED"))
    if not all_ok:
        sys.exit(1)
