"""Summarise `route_tool.mojo eval` JSONL: geometric-mean ratios vs SABRE.

usage: python summarize_eval.py <eval.jsonl> <jobs.json> [--base sabre_o2|sabre_o3] [--md]
Ratios use (cx_units + 1) and (2q depth + 1) so zero-swap jobs count as ties.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict


def main():
    path, jobs_path = sys.argv[1], sys.argv[2]
    base = "sabre_o2"
    if "--base" in sys.argv:
        base = sys.argv[sys.argv.index("--base") + 1]
    md = "--md" in sys.argv
    with open(jobs_path) as f:
        jobs = json.load(f)
    fam = {t["name"]: t.get("family", t["name"]) for t in jobs["targets"]}
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cols = [c for c in ("policy", "portfolio", "heuristic", "scored", "sabre_o3", "sabre_o2") if c != base and any(r.get(c) for r in rows)]
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0, 0, 0]))
    for r in rows:
        b = r[base]
        for c in cols:
            v = r.get(c)
            if not v:
                continue
            for key in (fam.get(r["target"], r["target"]), "ALL"):
                a = agg[key][c]
                a[0] += math.log((v[0] + 1) / (b[0] + 1))
                a[1] += math.log((v[1] + 1) / (b[1] + 1))
                a[2] += math.log(max(v[2], 1e-3) / max(b[2], 1e-3))
                a[3] += 1
                a[4] += v[0] < b[0]
                a[5] += v[0] > b[0]
                if len(v) > 3:
                    a[6] += v[3]
    keys = sorted(k for k in agg if k != "ALL") + ["ALL"]
    if md:
        print(f"| family | config | jobs | 2q cost ratio | 2q depth ratio | time ratio | wins/losses vs {base} | policy won (portfolio) |")
        print("|---|---|---|---|---|---|---|---|")
    for k in keys:
        for c in cols:
            a = agg[k][c]
            if a[3] == 0:
                continue
            n = a[3]
            if md:
                print(f"| {k} | {c} | {n} | {math.exp(a[0]/n):.3f} | {math.exp(a[1]/n):.3f} | {math.exp(a[2]/n):.1f}x | {a[4]}/{a[5]} | {a[6] if c == 'portfolio' else ''} |")
            else:
                print(f"{k:14s} {c:10s} n={n:3d} cx={math.exp(a[0]/n):.3f} d2={math.exp(a[1]/n):.3f} t={math.exp(a[2]/n):6.1f}x W/L={a[4]}/{a[5]}" + (f" pol_won={a[6]}" if c == "portfolio" else ""))


if __name__ == "__main__":
    main()
