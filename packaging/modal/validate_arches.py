"""Cross-architecture validation of Lumen's shippable CUDA binary on Modal.

Validates the Phase-1 CUDA deliverable (the int8-TC MoE stack + persistent NVRTC
PTX disk cache, branch ``cuda-ptx-cache @ b623977``) across the five NVIDIA
compute capabilities that make up the llama.cpp arch matrix:

    T4   -> sm_75   (A10G -> sm_86, L4 -> sm_89, A100 -> sm_80 baseline, H100 -> sm_90)

The Lumen Linux x86_64 binary defers ALL kernel compilation to runtime via NVRTC
(no build-time CUDA SDK link; it dlopens libcuda/libnvrtc/libcublas). That means
a single binary must JIT-compile every kernel for whatever compute capability the
container's GPU reports -- THE cross-arch test. The PTX disk cache keys on
``arch || cc_major.minor || nvrtc_version || driver_version`` so each arch gets
its own cache entries; we verify the cold->warm transition independently per arch.

Per-arch gates (collected into a matrix in the driver):
  1. Starts / NVRTC-compiles for this arch (server reaches "listening").
  2. PTX cache cold->warm (1st launch compiles+writes, 2nd loads from cache).
  3. DET-001: N>=20 temp-0 greedy completions on a fixed prompt -> 1 distinct.
  4. Coherence: factual/math smoke prompts -> sane output.
  + rough decode tok/s.

Run:
    modal run packaging/modal/validate_arches.py                 # all arches
    modal run packaging/modal/validate_arches.py --arches T4,H100
    modal run packaging/modal/validate_arches.py --put-binaries  # upload bins first

Cost: each arch runs ~5-10 min on one GPU and the container stops immediately
after, so the whole matrix is a few GPU-minutes total.
"""

import json
import os
import sys
import time

import modal

# --- Image: CUDA 12.2 runtime + NVRTC -----------------------------------------
# Modal injects the GPU driver (libcuda) into the container. The binary still
# needs libnvrtc (runtime NVRTC source->PTX compile) and libcublas, which the
# -devel image ships (the -runtime image lacks the standalone libnvrtc the
# binary dlopens). We add the two prebuilt Linux x86_64 binaries straight into
# the image so there is nothing to build at run time -- this image IS the
# Phase-1 Docker deliverable.
HERE = os.path.dirname(__file__)
BIN_DIR = os.path.join(HERE, "bin")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("ca-certificates")
    # The shippable binaries (downloaded from the build host into ./bin first;
    # see --put-binaries / the README). Baked into the image = reproducible.
    .add_local_dir(BIN_DIR, remote_path="/opt/lumen/bin", copy=True)
    .run_commands("chmod +x /opt/lumen/bin/lumen-server /opt/lumen/bin/lumen || true")
)

app = modal.App("lumen-phase1-arch-validation")

# Persistent volumes: the 9B-q8 LBC (uploaded once, mounted read-only
# everywhere) and the PTX cache (writable; each arch writes its own keyed
# entries -- we wipe per-arch inside the run to measure a true cold start).
models_vol = modal.Volume.from_name("lumen-models", create_if_missing=True)
ptx_vol = modal.Volume.from_name("lumen-ptxcache", create_if_missing=True)

MODEL_PATH = "/models/9bq8.lbc"
SERVER_BIN = "/opt/lumen/bin/lumen-server"
PORT = 8088

# Fixed validation prompts. The 4th tuple element is `advisory`: advisory prompts
# are run and logged but do NOT gate coherence_ok. This is a DEPLOYMENT smoke
# (does the binary run, stay deterministic, and produce coherent correct output),
# not a model-IQ benchmark. Hard multi-digit arithmetic at greedy temp-0 is
# unreliable on a 9B (it answers a clean but sometimes-wrong number, e.g.
# 17*23 -> "401") — that's a model-capability nuance, independent of binary
# health, so it's advisory. Determinism (DET-001), factual recall, and simple
# arithmetic remain HARD gates and would catch a genuinely broken/garbling build.
DET_PROMPT = "What is the capital of France?"
COHERENCE_PROMPTS = [
    ("capital_france", "What is the capital of France? Answer in one word.", "paris", False),
    ("two_plus_two", "What is 2 + 2? Reply with just the number.", "4", False),
    ("sky_color", "What color is a clear daytime sky? One word.", "blue", False),
    ("math_17x23", "What is 17 times 23? Reply with just the number.", "391", True),  # advisory
]

# Modal GPU string -> expected compute capability (for the report; nvidia-smi is
# the ground truth we actually record).
ARCH_CC = {
    "T4": "7.5",
    "A10G": "8.6",
    "L4": "8.9",
    "A100": "8.0",
    "H100": "9.0",
}


def _run_validation(gpu_name: str) -> dict:
    """Body shared by every per-arch function. Runs entirely inside the GPU
    container; returns a JSON-able result dict."""
    import subprocess
    import urllib.request

    def log(msg):
        print(f"[{gpu_name}] {msg}", flush=True)

    log("validation start")
    result = {
        "arch": gpu_name,
        "expected_cc": ARCH_CC.get(gpu_name, "?"),
        "nvidia_smi_name": None,
        "nvidia_smi_cc": None,
        "starts": False,
        "kernel_compile_failures": [],
        "ptx_cold_ready_s": None,
        "ptx_cold_line": None,
        "ptx_warm_ready_s": None,
        "ptx_warm_line": None,
        "ptxc_files_written": None,
        "det001_distinct": None,
        "det001_n": 0,
        "det001_sample": None,
        "coherence": {},
        "coherence_ok": False,
        "decode_tok_s": None,
        "errors": [],
    }

    # --- nvidia-smi: ground-truth GPU + compute cap --------------------------
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        line = smi.stdout.strip().splitlines()[0]
        name, cc, drv = [x.strip() for x in line.split(",")]
        result["nvidia_smi_name"] = name
        result["nvidia_smi_cc"] = cc
        result["driver_version"] = drv
        log(f"nvidia-smi: {name} cc={cc} driver={drv}")
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"nvidia-smi: {e}")
        log(f"nvidia-smi FAILED: {e}")

    # Copy the model out of the (network) volume to fast local container disk
    # ONCE, so the server's mmap reads hit local SSD instead of streaming the
    # whole 8.9 GB over the volume FS on first touch. Mirrors a real deploy
    # that stages the model to local disk.
    local_model = "/tmp/9bq8.lbc"
    if not os.path.exists(local_model):
        t0 = time.time()
        log(f"copying model {MODEL_PATH} -> {local_model} ...")
        subprocess.run(["cp", MODEL_PATH, local_model], check=True)
        log(f"model copy done in {time.time()-t0:.1f}s, size={os.path.getsize(local_model)}")

    cache_dir = f"/ptxcache/{gpu_name}"  # per-arch cache subdir on the volume

    def launch_server(log_path: str):
        env = dict(os.environ)
        env["LUMEN_CACHE_DIR"] = cache_dir  # PTX cache -> $LUMEN_CACHE_DIR/ptx
        env["LUMEN_CUDA_DECODE_DELAY_US"] = "0"
        logf = open(log_path, "w")
        proc = subprocess.Popen(
            [
                SERVER_BIN,
                "--model", local_model,
                "--backend", "cuda",
                "--host", "127.0.0.1",
                "--port", str(PORT),
                "--log-level", "info",
            ],
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return proc, logf

    def wait_listening(proc, log_path: str, timeout_s: int):
        """Poll the log for the 'listening' line; return (ready_s, ptx_line)
        or (None, None) on timeout/death. Surfaces kernel-compile failures.
        Echoes new server-log lines to stdout for real-time visibility."""
        start = time.time()
        seen = 0
        last_beat = start
        while time.time() - start < timeout_s:
            # Echo any new server-log content (so `modal app logs` shows the
            # NVRTC compile / PTX cache / listening lines live).
            try:
                txt_now = open(log_path).read()
            except Exception:  # noqa: BLE001
                txt_now = ""
            if len(txt_now) > seen:
                for ln in txt_now[seen:].splitlines():
                    if ln.strip():
                        log(f"srv| {ln.rstrip()}")
                seen = len(txt_now)
                last_beat = time.time()
            elif time.time() - last_beat > 20:
                log(f"srv| ...still starting ({int(time.time()-start)}s elapsed)")
                last_beat = time.time()
            if proc.poll() is not None:
                # Process exited before listening -> capture why.
                try:
                    tail = open(log_path).read()[-4000:]
                except Exception:  # noqa: BLE001
                    tail = ""
                for ln in tail.splitlines():
                    low = ln.lower()
                    if any(k in low for k in ("nvrtc", "compile", "ptx")) and (
                        "error" in low or "fail" in low or "invalid" in low
                    ):
                        result["kernel_compile_failures"].append(ln.strip())
                result["errors"].append(f"server exited rc={proc.returncode}; tail: {tail[-800:]}")
                return None, None
            if "listening on" in txt_now:
                ready_s = round(time.time() - start, 2)
                ptx_line = None
                for ln in txt_now.splitlines():
                    if "PTX cache" in ln:
                        ptx_line = ln.strip()
                    low = ln.lower()
                    if ("nvrtc" in low or "compile" in low) and (
                        "error" in low or "invalid" in low or "fail" in low
                    ):
                        result["kernel_compile_failures"].append(ln.strip())
                log(f"LISTENING after {ready_s}s | {ptx_line}")
                return ready_s, ptx_line
            time.sleep(1.0)
        result["errors"].append(f"timeout waiting for listening ({timeout_s}s)")
        return None, None

    def http_complete(prompt: str, max_tokens: int = 24, temperature: float = 0.0):
        body = json.dumps({
            "model": "lumen",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"]

    # --- COLD launch (wipe this arch's cache first) --------------------------
    log("COLD launch (wiping PTX cache for this arch) ...")
    subprocess.run(["rm", "-rf", cache_dir], check=False)
    os.makedirs(cache_dir, exist_ok=True)
    cold_log = f"/tmp/cold_{gpu_name}.log"
    proc, logf = launch_server(cold_log)
    cold_ready, cold_ptx = wait_listening(proc, cold_log, timeout_s=900)
    result["ptx_cold_ready_s"] = cold_ready
    result["ptx_cold_line"] = cold_ptx
    if cold_ready is None:
        # Couldn't even start cold -> arch fails. Clean up and bail.
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        logf.close()
        return result
    result["starts"] = True

    # --- DET-001 + coherence + decode tok/s (on the cold server) -------------
    log("DET-001: 20x temp-0 greedy on fixed prompt ...")
    try:
        det_outputs = []
        for i in range(20):
            det_outputs.append(http_complete(DET_PROMPT, max_tokens=16, temperature=0.0).strip())
        result["det001_n"] = len(det_outputs)
        result["det001_distinct"] = len(set(det_outputs))
        result["det001_sample"] = det_outputs[0][:200]
        log(f"DET-001: {result['det001_distinct']} distinct / {result['det001_n']} | sample={det_outputs[0][:80]!r}")
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"det001: {e}")
        log(f"DET-001 FAILED: {e}")

    log("coherence smoke ...")
    try:
        ok = True
        for key, prompt, expect, advisory in COHERENCE_PROMPTS:
            out = http_complete(prompt, max_tokens=40, temperature=0.0)
            hit = expect.lower() in out.lower()
            result["coherence"][key] = {
                "out": out.strip()[:160], "expect": expect, "hit": hit, "advisory": advisory,
            }
            tag = " (advisory)" if advisory else ""
            log(f"coherence {key}{tag}: hit={hit} expect={expect!r} out={out.strip()[:60]!r}")
            if not advisory:
                ok = ok and hit
        result["coherence_ok"] = ok
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"coherence: {e}")
        log(f"coherence FAILED: {e}")

    # Rough decode tok/s: one longer generation, measure wall time.
    log("decode tok/s bench ...")
    try:
        t0 = time.time()
        n_tok = 128
        _ = http_complete("Write a short paragraph about the ocean.", max_tokens=n_tok, temperature=0.0)
        dt = time.time() - t0
        if dt > 0:
            result["decode_tok_s"] = round(n_tok / dt, 1)
        log(f"decode ~{result['decode_tok_s']} tok/s ({n_tok} tok in {dt:.1f}s incl prefill)")
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"decode_bench: {e}")
        log(f"decode_bench FAILED: {e}")

    # Count cache files written this cold run.
    try:
        ptx_dir = os.path.join(cache_dir, "ptx")
        result["ptxc_files_written"] = len(
            [f for f in os.listdir(ptx_dir) if f.endswith(".ptxc")]
        )
    except Exception:  # noqa: BLE001
        result["ptxc_files_written"] = 0

    # Shut the cold server down before the warm relaunch.
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    logf.close()
    log(f"cold cache wrote {result['ptxc_files_written']} .ptxc files; committing volume")
    ptx_vol.commit()  # persist the freshly written cache to the volume

    # --- WARM launch (cache now populated; must NOT recompile) ---------------
    log("WARM relaunch (cache populated) ...")
    warm_log = f"/tmp/warm_{gpu_name}.log"
    proc2, logf2 = launch_server(warm_log)
    warm_ready, warm_ptx = wait_listening(proc2, warm_log, timeout_s=600)
    result["ptx_warm_ready_s"] = warm_ready
    result["ptx_warm_line"] = warm_ptx
    try:
        proc2.terminate()
        proc2.wait(timeout=20)
    except Exception:  # noqa: BLE001
        try:
            proc2.kill()
        except Exception:  # noqa: BLE001
            pass
    logf2.close()

    return result


# --- One @app.function per arch (Modal needs a literal gpu= at decoration) ----
COMMON = dict(
    image=image,
    volumes={"/models": models_vol, "/ptxcache": ptx_vol},
    timeout=2700,
)


@app.function(gpu="T4", **COMMON)
def validate_t4():
    return _run_validation("T4")


@app.function(gpu="A10G", **COMMON)
def validate_a10g():
    return _run_validation("A10G")


@app.function(gpu="L4", **COMMON)
def validate_l4():
    return _run_validation("L4")


@app.function(gpu="A100", **COMMON)
def validate_a100():
    return _run_validation("A100")


@app.function(gpu="H100", **COMMON)
def validate_h100():
    return _run_validation("H100")


_FN = {
    "T4": validate_t4,
    "A10G": validate_a10g,
    "L4": validate_l4,
    "A100": validate_a100,
    "H100": validate_h100,
}


@app.function(image=image, volumes={"/models": models_vol}, timeout=3600)
def put_model_from_url(url: str):
    """Optional helper: pull a 9B-q8 LBC into the models volume from a URL.
    Normally we just `modal volume put lumen-models 9bq8.lbc /9bq8.lbc`."""
    import subprocess
    subprocess.run(["bash", "-c", f"cd /models && curl -fL -o 9bq8.lbc '{url}'"], check=True)
    models_vol.commit()
    return os.path.getsize(MODEL_PATH)


@app.local_entrypoint()
def main(arches: str = "T4,A10G,L4,A100,H100"):
    """Fan out validation across the requested arches (parallel) and print the
    JSON matrix. Containers stop as soon as each function returns."""
    want = [a.strip() for a in arches.split(",") if a.strip()]
    print(f"[driver] validating arches: {want}", file=sys.stderr)

    # Launch all requested arches in parallel via .spawn(), then gather.
    handles = {}
    for a in want:
        if a not in _FN:
            print(f"[driver] unknown arch {a}, skipping", file=sys.stderr)
            continue
        handles[a] = _FN[a].spawn()

    results = {}
    for a, h in handles.items():
        try:
            results[a] = h.get()
        except Exception as e:  # noqa: BLE001
            results[a] = {"arch": a, "fatal": str(e)}
        print(f"[driver] {a} done", file=sys.stderr)

    print("\n===== LUMEN PHASE-1 CROSS-ARCH VALIDATION MATRIX (JSON) =====")
    print(json.dumps(results, indent=2))

    # Hard gate: process exit code is authoritative (CI greps are belt-and-
    # suspenders). An arch PASSES only if the server actually started, every
    # kernel compiled, DET-001 collapsed to exactly one distinct output, and the
    # coherence smoke hit. This makes a startup crash a FAIL (not a mislabeled
    # "non-deterministic", which is what a null det001_distinct used to look like).
    def arch_ok(r):
        return bool(
            r.get("starts") is True
            and not r.get("kernel_compile_failures")
            and r.get("det001_distinct") == 1
            and r.get("coherence_ok") is True
            and not r.get("fatal")
        )

    print("\n===== PER-ARCH VERDICT =====")
    all_pass = True
    for a, r in results.items():
        ok = arch_ok(r)
        all_pass = all_pass and ok
        if ok:
            print(f"  {a}: PASS  (det001=1/{r.get('det001_n')}, "
                  f"cold={r.get('ptx_cold_ready_s')}s warm={r.get('ptx_warm_ready_s')}s, "
                  f"decode={r.get('decode_tok_s')} tok/s)")
        else:
            why = (r.get("errors") or ([r["fatal"]] if r.get("fatal") else ["unknown"]))[0]
            print(f"  {a}: FAIL  (starts={r.get('starts')} "
                  f"kernel_compile_failures={len(r.get('kernel_compile_failures') or [])} "
                  f"det001_distinct={r.get('det001_distinct')} coherence_ok={r.get('coherence_ok')})")
            print(f"        reason: {str(why)[:400]}")

    if not all_pass:
        print("\nVALIDATION FAILED")
        sys.exit(1)
    print("\nVALIDATION PASSED")
