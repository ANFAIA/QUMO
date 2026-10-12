"""Aggregate neutral-benchmark JSONL into a Markdown report.

For every (circuit, target, level) group, each SDK's result is compared with
the best result of the group:
  * q2  = (n2q + 1) / (best n2q + 1)          (1.00 = best, lower is better)
  * esp = esp / best esp                      (1.00 = best, higher is better)
  * t   = wall time / fastest wall time       (1.00 = fastest)
Aggregates are geometric means over groups where the SDK produced a verified
(or, above the verification size, valid) result. Win = strictly best 2q count
in its group; tie = equal to the best. Status counts show unsupported,
incorrect, invalid, error and timeout cases explicitly.

Usage: python bench/neutral/report.py results.jsonl [--level default]
"""
import json
import math
import sys
from collections import defaultdict


def gmean(xs):
    xs = [max(x, 1e-12) for x in xs]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    path = sys.argv[1]
    level_filter = None
    if "--level" in sys.argv:
        level_filter = sys.argv[sys.argv.index("--level") + 1]
    recs = [json.loads(l) for l in open(path)]
    if level_filter:
        recs = [r for r in recs if r["level"] == level_filter]
    sdks = sorted({r["sdk"] for r in recs})
    groups = defaultdict(dict)
    for r in recs:
        groups[(r["circuit"], r["target"], r["level"])][r["sdk"]] = r
    status = defaultdict(lambda: defaultdict(int))
    per = defaultdict(lambda: defaultdict(list))  # (target, sdk) -> metric -> list
    wins = defaultdict(lambda: defaultdict(int))
    for (circ, tgt, lvl), by in groups.items():
        good = {s: r for s, r in by.items() if r["status"] == "ok"}
        for s, r in by.items():
            status[(tgt, s)][r["status"]] += 1
        if len(good) < 2:
            continue
        best2 = min(r["n2q"] for r in good.values())
        bestesp = max(r["esp"] for r in good.values())
        fastest = min(max(r["wall_s"], 1e-6) for r in good.values())
        for s, r in good.items():
            per[(tgt, s)]["q2"].append((r["n2q"] + 1) / (best2 + 1))
            per[(tgt, s)]["esp"].append(r["esp"] / bestesp if bestesp > 0 else 1.0)
            per[(tgt, s)]["t"].append(max(r["wall_s"], 1e-6) / fastest)
            per[(tgt, s)]["wall"].append(max(r["wall_s"], 1e-6))
            if r["n2q"] == best2:
                others = [o["n2q"] for so, o in good.items() if so != s]
                if all(o > best2 for o in others):
                    wins[(tgt, s)]["win"] += 1
                else:
                    wins[(tgt, s)]["tie"] += 1
    targets = sorted({k[0] for k in status})
    print("# Hardware-neutral compiler benchmark — results\n")
    print(f"Records: {len(recs)} · SDKs: {', '.join(sdks)} · groups: {len(groups)}"
          + (f" · level: {level_filter}" if level_filter else " · levels: all") + "\n")
    print("Ratios are geometric means vs. the best SDK in each (circuit, target, level) group: "
          "**2Q** and **time** lower is better (1.00 = best), **ESP** higher is better (1.00 = best).\n")
    print("| target | SDK | ok | 2Q ratio | ESP ratio | time ratio | median wall (ms) | wins | ties | unsupported | incorrect | invalid | error | timeout |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    overall = defaultdict(lambda: defaultdict(list))
    for t in targets:
        for s in sdks:
            st = status[(t, s)]
            m = per[(t, s)]
            for k in ("q2", "esp", "t"):
                overall[s][k] += m[k]
            walls = sorted(m["wall"])
            med = walls[len(walls) // 2] * 1000 if walls else float("nan")
            print(f"| {t} | {s} | {st['ok']} | {gmean(m['q2']):.3f} | {gmean(m['esp']):.3f} | {gmean(m['t']):.1f} | "
                  f"{med:.1f} | {wins[(t, s)]['win']} | {wins[(t, s)]['tie']} | {st['unsupported']} | "
                  f"{st['incorrect']} | {st['invalid']} | {st['error']} | {st['timeout']} |")
    print("\n## Overall (all targets)\n")
    print("| SDK | 2Q ratio | ESP ratio | time ratio | wins | ties |")
    print("|---|---:|---:|---:|---:|---:|")
    for s in sdks:
        w = sum(wins[(t, s)]["win"] for t in targets)
        ti = sum(wins[(t, s)]["tie"] for t in targets)
        o = overall[s]
        print(f"| {s} | {gmean(o['q2']):.3f} | {gmean(o['esp']):.3f} | {gmean(o['t']):.1f} | {w} | {ti} |")


if __name__ == "__main__":
    main()
