"""Compare Benchpress runs (pytest-benchmark JSON) of several SDKs.

    python bench/analysis/benchpress_compare.py qumo=bench/results/benchpress/qumo_full.json \
        qiskit=bench/results/benchpress/qiskit_full.json [tket=published.json ...] > report.md

Per test category: status counts, geometric-mean time ratio and 2Q-count ratio
of each SDK vs the first one (the reference), and win/tie/loss on 2Q count.
Ratios use only tests that both SDKs completed. 2Q counts compare
`output_gate_count_2q` (same target, same validation rule for every gym).
"""
import json
import math
import re
import sys
from collections import defaultdict


def load(path):
    d = json.load(open(path))
    bench = {}
    for b in d.get("benchmarks", []):
        name = b["fullname"].split("::")[-1]
        bench[name] = (b["stats"]["mean"], b["extra_info"].get("output_gate_count_2q"),
                       b["extra_info"].get("output_depth_2q"))
    status = d.get("test_status_counts", {})
    dumps = d.get("test_dumps", {})
    return bench, status, dumps


def category(name):
    base = name.split("[")[0]
    m = re.search(r"-(all-to-all|heavy-hex|square|linear)\]$", name)
    return base + (" · " + m.group(1) if m else "")


def gmean(xs):
    xs = [max(x, 1e-12) for x in xs]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    runs = []
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        runs.append((label, *load(path)))
    ref_label, ref, ref_status, _ = runs[0]
    print("## Status counts\n")
    print("| SDK | passed | failed | skipped | xfailed |")
    print("|---|---:|---:|---:|---:|")
    for label, _, st, _ in runs:
        print(f"| {label} | {st.get('passed', '?')} | {st.get('failed', '?')} | {st.get('skipped', '?')} | {st.get('xfailed', '?')} |")
    for label, other, _, dumps in runs[1:]:
        print(f"\n## {ref_label} vs {label}\n")
        print(f"Ratios are {ref_label}/{label}: **time < 1 ⇒ {ref_label} faster**, **2Q < 1 ⇒ {ref_label} fewer two-qubit gates**.\n")
        print(f"| category | n | time ratio | 2Q ratio | {ref_label} fewer 2Q | tie | {label} fewer 2Q |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        by = defaultdict(list)
        for name, (t, c, _) in ref.items():
            if name in other:
                ot, oc, _ = other[name]
                by[category(name)].append((t / max(ot, 1e-12), None if (c is None or oc is None) else (c + 1) / (oc + 1)))
        allt, allq, w = [], [], [0, 0, 0]
        for cat in sorted(by):
            v = by[cat]
            ts = [a for a, _ in v]
            qs = [b for _, b in v if b is not None]
            win = sum(1 for b in qs if b < 1)
            tie = sum(1 for b in qs if b == 1)
            loss = sum(1 for b in qs if b > 1)
            w[0] += win
            w[1] += tie
            w[2] += loss
            allt += ts
            allq += qs
            q = f"{gmean(qs):.3f}" if qs else "–"
            print(f"| {cat} | {len(v)} | {gmean(ts):.2f} | {q} | {win} | {tie} | {loss} |")
        print(f"| **all** | {len(allt)} | **{gmean(allt):.2f}** | **{gmean(allq):.3f}** | {w[0]} | {w[1]} | {w[2]} |")
        failed = list(dumps.get("failed", {}).keys())
        if failed:
            print(f"\n{label} failures ({len(failed)}): " + ", ".join(f.split('::')[-1] for f in failed[:20]))


if __name__ == "__main__":
    main()
