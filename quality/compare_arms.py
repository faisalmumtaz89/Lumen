#!/usr/bin/env python3
"""Compare quality arms per ITEM, not per gate threshold.

An absolute pass/fail table cannot answer the question this campaign asks. Two
corpus items fail on the SHIPPED baseline (`vlong-howto-01` DD-REP,
`stress-rep-02` DD-SPAM — the latter asks for 1..200 in words, whose correct
answer is inherently repetitive, so the detector is right and the item is
mine to own). A gate that is already red on prod cannot certify a candidate.

The criterion that survives that is codex-sol's: ZERO CANDIDATE-ONLY failures.
An item that fails on prod and on the candidate is a pre-existing property of
the model or a defect in my corpus; an item that passes on prod or oracle and
fails ONLY on the candidate is a regression this change caused.
"""
import json
import pathlib
import sys
from collections import defaultdict

def load(path):
    d = json.loads(pathlib.Path(path).read_text())
    out = {}
    for row in d.get("results", d.get("rows", [])):
        for r in row.get("results", []):
            out[r["id"]] = bool(r.get("passed"))
    return out

def main(paths):
    arms = {}
    for p in paths:
        tag = pathlib.Path(p).stem.replace("cell-9b-q4-cuda-", "")
        arms[tag] = load(p)
    ids = sorted(set().union(*[set(a) for a in arms.values()]))
    cand = [t for t in arms if t.startswith("cand")]
    ctrl = [t for t in arms if t in ("prod", "oracle")]

    only_cand, shared, flaky = [], [], []
    for i in ids:
        c_pass = [arms[t].get(i) for t in cand if i in arms[t]]
        k_pass = [arms[t].get(i) for t in ctrl if i in arms[t]]
        if not c_pass or not k_pass:
            continue
        if len(set(c_pass)) > 1:
            flaky.append(i)          # non-deterministic across candidate reruns
        elif not all(c_pass) and any(k_pass):
            only_cand.append(i)      # a control passed it; the candidate does not
        elif not all(c_pass):
            shared.append(i)

    print(f"arms: {sorted(arms)}  items: {len(ids)}")
    print(f"\nCANDIDATE-ONLY FAILURES ({len(only_cand)}) <- the blocking set")
    for i in only_cand:
        print(f"  {i}: " + " ".join(f"{t}={'P' if arms[t].get(i) else 'F'}" for t in sorted(arms)))
    print(f"\nfails on candidate AND a control ({len(shared)}) — pre-existing or corpus defect")
    for i in shared:
        print(f"  {i}")
    print(f"\nNON-DETERMINISTIC across candidate reruns ({len(flaky)}) — also blocking")
    for i in flaky:
        print(f"  {i}: " + " ".join(f"{t}={'P' if arms[t].get(i) else 'F'}" for t in sorted(arms)))
    return 1 if (only_cand or flaky) else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
