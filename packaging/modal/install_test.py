"""Install-test: run Lumen's REAL one-command installer (``curl … | bash``) in a
clean Linux + NVIDIA container and validate it end-to-end.

Complements ``validate_arches.py`` (which validates the binary). Here the
container has NO baked Lumen binary — the installer must detect CUDA, download the
released Linux x86_64 binary from the GitHub release (alias-first, API-tag
fallback), verify its SHA-256, install it, pull a model, and run it on the GPU.
It also exercises:
  - the true NO-CONTROLLING-TERMINAL headless path (no --yes, no /dev/tty),
  - a Mixture-of-Experts pull + ``lumen-server`` boot (the model whose alias must
    round-trip through pull AND lumen-server — guards the canonical-alias fix),
  - and a GPU-less function that asserts the installer REFUSES on a no-NVIDIA box.

Run (Modal token from env, e.g. in CI):
    modal run packaging/modal/install_test.py --ref install-onboarder
Gate the CI step on the line ``INSTALL VALIDATION PASSED``.
"""

import os
import subprocess
import sys

import modal

RAW = "https://raw.githubusercontent.com/faisalmumtaz89/Lumen"  # + /<ref>/packaging/macos/install.sh
CACHE = "/root/.cache/lumen"  # pin so the installer's pull and the server's read agree

image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ca-certificates", "curl", "tar")
)
app = modal.App("lumen-install-test")


def _sh(cmd, env=None):
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, env=env)


@app.function(gpu="A100", image=image, timeout=2400)
def install_e2e(ref: str):
    """Full installer path on a real A100: detect → download → install → pull → run,
    plus the no-tty headless path and a MoE pull+serve (canonical-alias guard)."""
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

    base_env = dict(os.environ)
    base_env["LUMEN_CACHE_DIR"] = CACHE
    base_env["LUMEN_CUDA_DECODE_DELAY_US"] = "0"
    url = f"{RAW}/{ref}/packaging/macos/install.sh"

    # ── 1. real installer, 9B-Q8, non-interactive ───────────────────────────
    print(f"=== curl -fsSL {url} | bash -s -- --model qwen3.5-9b --quant q8_0 --yes ===", flush=True)
    r = _sh(f"curl -fsSL {url} | bash -s -- --model qwen3.5-9b --quant q8_0 --yes", env=base_env)
    log = r.stdout + r.stderr
    print(log[-5000:], flush=True)
    step("installer exit 0", r.returncode == 0, f"rc={r.returncode}")
    step("detected CUDA backend", "cuda" in log.lower())
    step("checksum verified", "verified" in log)
    step("both binaries installed",
         os.path.exists("/usr/local/bin/lumen") and os.path.exists("/usr/local/bin/lumen-server"))
    v = _sh("/usr/local/bin/lumen --version")
    step("lumen --version", v.returncode == 0, v.stdout.strip())
    # I3/S1: the LBC must exist exactly where lumen-server will look for it.
    lbc_9b = f"{CACHE}/qwen3-5-9b-Q8_0.lbc"
    step("9B LBC at the server read-path", os.path.exists(lbc_9b), lbc_9b)

    # ── server helpers (with error handling — m5) ───────────────────────────
    def launch(model, quant):
        logf = open(f"/tmp/srv-{model}.log", "w")
        p = subprocess.Popen(
            ["/usr/local/bin/lumen-server", "--model", model, "--quant", quant,
             "--backend", "cuda", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"],
            stdout=logf, stderr=subprocess.STDOUT, env=base_env)
        for _ in range(180):  # up to ~6 min (cold NVRTC compile + load)
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=3).read()
                return p, True
            except Exception:
                pass
            if p.poll() is not None:
                return p, False
            time.sleep(2)
        return p, False

    def chat(prompt, n=16):
        try:
            body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}],
                               "max_tokens": n, "temperature": 0}).encode()
            req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                         data=body, headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=180).read())["choices"][0]["message"]["content"], None
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    # ── 2. 9B server: DET-001 + coherence ───────────────────────────────────
    srv, ready = launch("qwen3.5-9b", "q8_0")
    step("9B server reached /v1/models", ready, "" if ready else open("/tmp/srv-qwen3.5-9b.log").read()[-400:])
    if ready:
        outs, errs = set(), 0
        for _ in range(20):
            c, e = chat("Name one color.")
            if e:
                errs += 1
            else:
                outs.add(c.strip())
        step("9B DET-001 (N=20 greedy -> 1 distinct)", errs == 0 and len(outs) == 1, f"distinct={len(outs)} chat_errors={errs}")
        ans, e = chat("What is the capital of France?", n=24)
        step("9B coherence (Paris)", (e is None) and ("paris" in ans.lower()), (ans or e or "")[:80])
    srv.kill()

    # ── 3. headless path (M3): force NO controlling terminal via setsid -w. ──
    # The installer runs under `set -e`, so rc==0 is reached ONLY if it completed
    # every step (detect, download, SHA, install, default-select, pull, print)
    # with no error and — critically — no hang (a blocked read would never return).
    # That rc==0 IS the never-hang-headless guarantee. We print the full output so
    # the default-select path is auditable in the run log.
    r3 = _sh(f"setsid -w bash -c 'curl -fsSL {url} | bash' < /dev/null", env=base_env)
    log3 = r3.stdout + r3.stderr
    print("=== headless (setsid, no controlling tty) installer output ===", flush=True)
    print(log3[-2500:], flush=True)
    step("headless no-tty (setsid): installer completes cleanly, no hang", r3.returncode == 0, f"rc={r3.returncode}")

    # ── 4. MoE pull + serve (B1 canonical-alias guard) ──────────────────────
    rp = _sh("/usr/local/bin/lumen pull qwen3.5-moe-35b-a3b:q4_0 --yes", env=base_env)
    moe_lbc = f"{CACHE}/qwen3-5-moe-35b-a3b-Q4_0.lbc"
    step("MoE pulled to the server read-path", rp.returncode == 0 and os.path.exists(moe_lbc), moe_lbc)
    msrv, mready = launch("qwen3.5-moe-35b-a3b", "q4_0")
    step("MoE server reached /v1/models (canonical alias round-trips)", mready,
         "" if mready else open("/tmp/srv-qwen3.5-moe-35b-a3b.log").read()[-400:])
    if mready:
        ans, e = chat("Say hello in one word.", n=12)
        step("MoE coherence", e is None and bool(ans and ans.strip()), (ans or e or "")[:80])
    msrv.kill()

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
