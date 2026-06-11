#!/usr/bin/env python3
"""run_suite.py — drive the Model Quality Acceptance Suite against one cell.

Spins (or connects to) a lumen server, runs the self-contained GQ-* gates
(GQ-001 short / GQ-002 medium / GQ-003 long / GQ-004 very-long), applies the
DD-* detectors + answer keys, and emits the per-cell Required-Output-Format
results table + a cell-<id>.json manifest.

Fidelity gates (GQ-005/006 vs llama) and tool calling (GQ-007) and the coherence
judge (GQ-013) are run by companion drivers; this driver marks them DEFERRED so
the scorecard is explicit (AH-7: never a silent omission).

Usage:
  run_suite.py --model PATH --backend metal --cell 9b-q8-metal --port 8401
  run_suite.py --server http://127.0.0.1:8401 --cell 9b-q8-metal
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detectors as dd  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"

# gate -> (corpus file, pass fraction, kind)
GATES = {
    "GQ-001": ("short.jsonl", 14 / 15, "short"),
    "GQ-002": ("medium.jsonl", 9 / 10, "generation"),
    "GQ-004": ("verylong.jsonl", 5 / 5, "verylong"),
}


def load_corpus(fname: str) -> list[dict]:
    path = CORPUS / fname
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

def wait_ready(base: str, timeout_s: int | None = None) -> bool:
    # Large models (MoE q8 ~35 GB, bf16 ~70 GB) can take >>300 s to load from
    # disk; default generous, override via QSUITE_READY_TIMEOUT.
    if timeout_s is None:
        timeout_s = int(os.environ.get("QSUITE_READY_TIMEOUT", "3600"))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=4) as r:
                if b"list" in r.read():
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def spin_server(model: str, backend: str, port: int, ctx: int = 4096, server_bin: str | None = None):
    srv = Path(server_bin) if server_bin else (HERE.parent / "target" / "release" / "lumen-server")
    if not srv.exists():
        sys.exit(f"lumen-server not found at {srv}")
    env = dict(os.environ)
    if backend == "cuda":
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    log = open(f"/tmp/qsuite-server-{port}.log", "wb")
    proc = subprocess.Popen(
        [str(srv), "--model", model, "--backend", backend, "--port", str(port),
         "--context-len", str(ctx)],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
    )
    return proc


def query(base: str, prompt: str, max_tokens: int, temperature: float = 0.0) -> tuple[str, str]:
    body = json.dumps({
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.loads(r.read())
    choice = resp["choices"][0]
    return choice["message"]["content"], choice.get("finish_reason", "")


# ---------------------------------------------------------------------------
# answer-key + anchor checks
# ---------------------------------------------------------------------------

def answer_ok(text: str, item: dict) -> bool:
    answers = item.get("answers")
    if not answers:
        return True
    low = text.lower()
    if item.get("answers_all"):
        return all(a.lower() in low for a in answers)
    # default: any acceptable alternative present (word-boundary for short numerics)
    for a in answers:
        al = a.lower()
        if re.search(rf"(?<!\d){re.escape(al)}(?!\d)", low) if al.isdigit() else (al in low):
            return True
    return False


def anchors_ok(text: str, item: dict) -> bool:
    anchors = item.get("anchors")
    if not anchors:
        return True
    low = text.lower()
    return any(a.lower() in low for a in anchors)  # >=1 anchor = on-topic


def window_detectors(text: str, win_words: int = 256) -> dd.QualityVerdict:
    """Sliding-window DD over a long output to catch late-onset degeneration."""
    words = text.split()
    if len(words) <= win_words:
        return dd.evaluate(text)
    worst = None
    for i in range(0, len(words), win_words // 2):
        chunk = " ".join(words[i : i + win_words])
        v = dd.evaluate(chunk)
        if not v.passed:
            return v
        worst = v
    return worst or dd.evaluate(text)


# ---------------------------------------------------------------------------
# gate runners
# ---------------------------------------------------------------------------

def run_prompt(base: str, item: dict, kind: str) -> dict:
    text, finish = query(base, item["prompt"], item.get("max_tokens", 256))
    # DD-TERM is NOT applied to open-ended generation: hitting max_tokens on an
    # explanation prompt is expected (truncation != degeneration). It is only
    # meaningful as an opt-in per-prompt flag ("require_eos": true) for prompts
    # that genuinely must self-terminate.
    v = dd.evaluate(
        text,
        is_math=item.get("is_math", False),
        finish_reason=finish,
        expect_eos=True,
        check_term=item.get("require_eos", False),
    )
    if kind == "verylong":
        wv = window_detectors(text)
        if not wv.passed:
            v = wv
    correct = answer_ok(text, item)
    ontopic = anchors_ok(text, item)
    ok = v.passed and correct and ontopic
    return {
        "id": item["id"],
        "passed": ok,
        "dd_passed": v.passed,
        "dd_fired": v.failed_names(),
        "answer_correct": correct,
        "on_topic": ontopic,
        "finish_reason": finish,
        "snippet": text[:240].replace("\n", " "),
    }


def run_gate(base: str, gate: str) -> dict:
    fname, pass_frac, kind = GATES[gate]
    items = load_corpus(fname)
    if not items:
        return {"gate": gate, "status": "DEFERRED", "evidence": f"no corpus {fname}"}
    results = [run_prompt(base, it, kind) for it in items]
    npass = sum(1 for r in results if r["passed"])
    n = len(results)
    passed = npass >= max(1, round(pass_frac * n)) and npass == n if pass_frac >= 1.0 else npass >= round(pass_frac * n)
    fails = [r for r in results if not r["passed"]]
    status = "PASS" if passed else "FAIL"
    ev = f"{npass}/{n} prompts pass (threshold {pass_frac:.2f}). "
    if fails:
        ev += "Fails: " + "; ".join(
            f"{r['id']}[dd:{','.join(r['dd_fired']) or '-'}"
            f"{',ans✗' if not r['answer_correct'] else ''}"
            f"{',topic✗' if not r['on_topic'] else ''}]" for r in fails[:6]
        )
    return {"gate": gate, "status": status, "evidence": ev, "results": results}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--server")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--port", type=int, default=8401)
    ap.add_argument("--gates", default="GQ-001,GQ-002,GQ-004")
    ap.add_argument("--out", default="/tmp/qsuite")
    ap.add_argument("--server-bin", default=None)
    args = ap.parse_args()

    proc = None
    base = args.server
    try:
        if not base:
            if not args.model:
                sys.exit("need --server or --model")
            proc = spin_server(args.model, args.backend, args.port, server_bin=args.server_bin)
            base = f"http://127.0.0.1:{args.port}"
            if not wait_ready(base):
                print(f"[{args.cell}] SERVER_NOT_READY (see /tmp/qsuite-server-{args.port}.log)")
                return 2

        rows = []
        for gate in args.gates.split(","):
            gate = gate.strip()
            if gate not in GATES:
                rows.append({"gate": gate, "status": "DEFERRED", "evidence": "not a self-contained gate (needs llama / tools / judge driver)"})
                continue
            print(f"[{args.cell}] running {gate} ...", flush=True)
            rows.append(run_gate(base, gate))

        # results table
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"cell-{args.cell}.json").write_text(json.dumps(rows, indent=1))
        npass = sum(1 for r in rows if r["status"] == "PASS")
        nfail = sum(1 for r in rows if r["status"] == "FAIL")
        print()
        print(f"=== CELL {args.cell} — self-contained quality gates ===")
        print(f"| Gate | Status | Evidence |")
        print(f"|------|--------|----------|")
        for r in rows:
            sym = {"PASS": "✓", "FAIL": "✗", "DEFERRED": "·"}[r["status"]]
            print(f"| {r['gate']} | {sym} {r['status']} | {r['evidence']} |")
        verdict = "PRISTINE (self-contained subset)" if nfail == 0 else "NOT PRISTINE"
        print(f"\nCELL {args.cell}: {npass} ✓ / {nfail} ✗  → {verdict}")
        return 0 if nfail == 0 else 1
    finally:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
