"""Re-analyse Benchpress published results: TKET vs Qiskit on common tests.

Downloads JSON files from the `previous_results` branch of Qiskit/benchpress
(if not cached) and prints per-category geometric means of the time ratio and
the two-qubit-gate-count ratio, plus win/tie/loss counts.

Usage: python bench/analysis/published_results.py [cache_dir]
"""
import json
import math
import os
import re
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/Qiskit/benchpress/previous_results/"
FILES = {
    "qiskit": ["benchpress_v1/qiskit/qiskit_2.5.0rc1.json"],
    "tket": [
        "benchpress_v1/tket/2.18.0/tket-device.json",
        "benchpress_v1/tket/2.18.0/tket-abstract-all-to-all.json",
        "benchpress_v1/tket/2.18.0/tket-abstract-heavy-hex.json",
    ],
}


def fetch(path, cache):
    local = os.path.join(cache, os.path.basename(path))
    if not os.path.exists(local):
        urllib.request.urlretrieve(BASE + path, local)
    return json.load(open(local))


def load(paths, cache):
    out, failed = {}, []
    for p in paths:
        d = fetch(p, cache)
        failed += list(d.get("test_dumps", {}).get("failed", {}).keys())
        for b in d["benchmarks"]:
            name = b["fullname"].split("::")[-1]
            out[name] = (b["stats"]["mean"], b["extra_info"].get("output_gate_count_2q"))
    return out, failed


def category(name):
    base = name.split("[")[0]
    m = re.search(r"-(all-to-all|heavy-hex|square|linear)\]$", name)
    return base + (":" + m.group(1) if m else "")


def gmean(xs):
    return math.exp(sum(math.log(max(x, 1e-12)) for x in xs) / len(xs))


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache, exist_ok=True)
    q, _ = load(FILES["qiskit"], cache)
    t, t_failed = load(FILES["tket"], cache)
    by = {}
    for k, (tt, tc) in t.items():
        if k in q and tc is not None and q[k][1] is not None:
            qt, qc = q[k]
            by.setdefault(category(k), []).append((tt / qt, (tc + 1) / (qc + 1), k))
    print(f"TKET failures in published run: {len(t_failed)}")
    print(f"{'category':45s} {'n':>4s} {'time tk/qk':>10s} {'2q tk/qk':>9s} {'qk<':>4s} {'tk<':>4s} {'tie':>4s}")
    for c, v in sorted(by.items()):
        print(f"{c:45s} {len(v):4d} {gmean([a for a, _, _ in v]):10.2f} {gmean([b for _, b, _ in v]):9.3f} "
              f"{sum(b > 1 for _, b, _ in v):4d} {sum(b < 1 for _, b, _ in v):4d} {sum(b == 1 for _, b, _ in v):4d}")
    allv = [x for v in by.values() for x in v]
    print(f"ALL n={len(allv)} time={gmean([a for a, _, _ in allv]):.2f} 2q={gmean([b for _, b, _ in allv]):.3f} "
          f"tket_fewer_2q={sum(b < 1 for _, b, _ in allv)} qiskit_fewer_2q={sum(b > 1 for _, b, _ in allv)}")


if __name__ == "__main__":
    main()
