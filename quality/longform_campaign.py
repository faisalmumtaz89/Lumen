#!/usr/bin/env python3
"""Collect long-form generations per arm and score them continuously.

Runs INSIDE the Modal container. Spins one server per arm (model loads once,
then 45 HTTP requests), stores every generation, scores each, and reports the
paired candidate-minus-reference difference with a bootstrap CI.

Arms share one binary built with `activation-probe`; LUMEN_Q4_ACT_PROBE selects
the activation policy, so the two branch arms differ in nothing else.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_suite as rs                      # noqa: E402
from longform_score import score, paired_delta  # noqa: E402

FIVE = "ffn_down,attn_qkv,ffn_gate_up,gdn_qkv,gdn_attn_gate"


def collect(tag, server_bin, probe, model, port, items):
    os.environ["LUMEN_Q4_ACT_PROBE"] = probe
    proc = rs.spin_server(model, "cuda", port, ctx=8192, server_bin=server_bin)
    base = f"http://127.0.0.1:{port}"
    if not rs.wait_ready(base):
        print(f"[{tag}] server never became ready", flush=True)
        return {}
    out = {}
    t0 = time.time()
    for it in items:
        text, _ = rs.query(base, it["prompt"], it.get("max_tokens", 2048))
        out[it["id"]] = {"text": text, **score(text)}
    print(f"[{tag}] {len(out)} generations in {time.time()-t0:.0f}s", flush=True)
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(5)
    return out


def main(model, outdir):
    items = [json.loads(l) for l in (HERE / "corpus/longform.jsonl").read_text().splitlines() if l.strip()]
    print(f"=== long-form campaign: {len(items)} prompts x 3 arms ===", flush=True)
    arms = {
        "prod":   ("/opt/lumen/bin/server-prod",  "",     8701),
        "banked": ("/opt/lumen/bin/server-probe", "",     8702),
        "fast":   ("/opt/lumen/bin/server-probe", FIVE,   8703),
    }
    res = {t: collect(t, b, p, model, port, items) for t, (b, p, port) in arms.items()}
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "generations.json").write_text(json.dumps(res, indent=1))

    print("\n=== per-arm means ===", flush=True)
    for t, r in res.items():
        if not r:
            continue
        for k in ("distinct3", "worst_window", "tail_ratio", "words"):
            m = sum(v[k] for v in r.values()) / len(r)
            print(f"  {t:7} {k:13} {m:.5f}", flush=True)
    print("\n=== PAIRED vs banked (all-F32 control); negative = worse ===", flush=True)
    for k in ("distinct3", "worst_window", "tail_ratio"):
        for t in ("fast", "prod"):
            d = paired_delta(res[t], res["banked"], k)
            if d:
                verdict = ("WORSE" if d["ci_hi"] < 0 else
                           "BETTER" if d["ci_lo"] > 0 else "no resolvable difference")
                print(f"  {t:6} {k:13} mean={d['mean']:+.5f} "
                      f"CI[{d['ci_lo']:+.5f},{d['ci_hi']:+.5f}] n={d['n']}  {verdict}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
