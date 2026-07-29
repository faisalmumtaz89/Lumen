#!/usr/bin/env python3
"""Reduce a lumen_cupti_inject CSV to the numbers a decode campaign needs.

Stdlib only. See README.md for the capture recipe.

The point of this reduction is the busy-vs-span decomposition. Event-bracket
profiling gives region spans, which silently include idle. Here:

    busy = sum(end - start)          true device occupancy
    span = max(end) - min(queued)    the run's full extent
    idle = span - busy               the GPU had nothing to run

A decode path that is launch-bound rather than bandwidth-bound shows small busy
values separated by large gaps.
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                r["queued_ns"] = int(r["queued_ns"])
                r["submitted_ns"] = int(r["submitted_ns"])
                r["start_ns"] = int(r["start_ns"])
                r["end_ns"] = int(r["end_ns"])
            except (KeyError, ValueError, TypeError):
                continue
            if r["end_ns"] <= 0 or r["start_ns"] <= 0:
                continue
            rows.append(r)
    return rows


def us(ns):
    return ns / 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="path written by LUMEN_CUPTI_CSV")
    ap.add_argument("--top", type=int, default=20,
                    help="show this many kernels by total busy time (default 20)")
    ap.add_argument("--gaps", action="store_true",
                    help="also report the largest inter-kernel starvation gaps")
    ap.add_argument("--kind", default="kernel",
                    help="activity kind to summarize (default 'kernel'; 'all' for everything)")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        sys.exit(f"{args.csv}: no usable rows (all timestamps zero or file empty)")

    if args.kind != "all":
        rows = [r for r in rows if r.get("kind") == args.kind]
        if not rows:
            sys.exit(f"{args.csv}: no rows of kind {args.kind!r}")

    rows.sort(key=lambda r: r["start_ns"])

    busy_ns = sum(r["end_ns"] - r["start_ns"] for r in rows)
    have_latency = any(r["queued_ns"] > 0 for r in rows)
    t0 = min((r["queued_ns"] for r in rows if r["queued_ns"] > 0),
             default=rows[0]["start_ns"])
    span_ns = max(r["end_ns"] for r in rows) - t0
    idle_ns = max(span_ns - busy_ns, 0)

    # Per-token slicing: argmax_f32 fires exactly once per decode token, so it
    # delimits tokens. Prefill launches land before the first argmax and are
    # excluded from the per-token figure rather than averaged into it.
    argmax = [i for i, r in enumerate(rows) if "argmax" in r.get("name", "")]
    tokens = len(argmax)

    print(f"file            {args.csv}")
    print(f"records         {len(rows)} (kind={args.kind})")
    print(f"tokens          {tokens} (delimited by argmax kernels)")
    if not have_latency:
        print("WARNING         queued/submitted are all zero -- latency timestamps")
        print("                were NOT enabled; only device busy time is valid.")
    print()
    print(f"device busy     {us(busy_ns):12.1f} us")
    print(f"run span        {us(span_ns):12.1f} us")
    print(f"idle (span-busy){us(idle_ns):12.1f} us   {100.0 * idle_ns / span_ns:5.1f}% of span")
    if tokens:
        print(f"busy/token      {us(busy_ns) / tokens:12.1f} us")
        print(f"span/token      {us(span_ns) / tokens:12.1f} us")
    print()

    agg = defaultdict(list)
    for r in rows:
        agg[r.get("name", "?")].append(r["end_ns"] - r["start_ns"])

    ranked = sorted(agg.items(), key=lambda kv: sum(kv[1]), reverse=True)
    print(f"{'kernel':<52} {'calls':>7} {'tot_us':>10} {'mean_us':>9} "
          f"{'p50_us':>9} {'%busy':>7}")
    print("-" * 100)
    for name, vals in ranked[: args.top]:
        tot = sum(vals)
        print(f"{name[:52]:<52} {len(vals):>7} {us(tot):>10.1f} "
              f"{us(statistics.mean(vals)):>9.2f} "
              f"{us(statistics.median(vals)):>9.2f} "
              f"{100.0 * tot / busy_ns:>6.2f}%")
    if len(ranked) > args.top:
        rest = sum(sum(v) for _, v in ranked[args.top:])
        print(f"{'... ' + str(len(ranked) - args.top) + ' more':<52} "
              f"{'':>7} {us(rest):>10.1f} {'':>9} {'':>9} "
              f"{100.0 * rest / busy_ns:>6.2f}%")

    if args.gaps:
        print()
        print("largest inter-kernel gaps (GPU idle between consecutive kernels)")
        print(f"{'gap_us':>10}  {'after':<44} {'before':<44}")
        print("-" * 100)
        gaps = []
        for prev, cur in zip(rows, rows[1:]):
            g = cur["start_ns"] - prev["end_ns"]
            if g > 0:
                gaps.append((g, prev.get("name", "?"), cur.get("name", "?")))
        gaps.sort(reverse=True)
        for g, a, b in gaps[:20]:
            print(f"{us(g):>10.2f}  {a[:44]:<44} {b[:44]:<44}")
        if gaps:
            total_gap = sum(g for g, _, _ in gaps)
            print()
            print(f"total inter-kernel gap {us(total_gap):.1f} us "
                  f"({100.0 * total_gap / span_ns:.1f}% of span) over {len(gaps)} gaps")


if __name__ == "__main__":
    main()
