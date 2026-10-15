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
       python bench/neutral/report.py results.jsonl --ai [--level max]

`--ai` writes the AI-transpiler report instead: qumo-ai against qumo, Qiskit,
TKET (and the optional qiskit+ibm-ai baseline) per target family and
overall — geometric-mean 2q-count and 2q-depth ratios, wins / ties / losses,
median compile time and its ratio to qumo — plus the per-technique
replacement statistics parsed from the QUMO pass reports.
"""
import json
import math
import sys
from collections import defaultdict


def gmean(xs):
    xs = [max(x, 1e-12) for x in xs]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


FAMILIES = {
    "sc_heavy_hex": "superconducting", "sc_grid": "superconducting", "sc_diagonal_grid": "superconducting",
    "sc_octagonal": "superconducting", "ion_all_to_all": "trapped ion", "ion_zone_ring": "trapped ion",
    "neutral_atom": "neutral atom", "neutral_atom_reconfigurable": "neutral atom",
    "linear_cx": "abstract", "tree": "abstract", "torus": "abstract",
}
TECHNIQUES = ("linear_synth", "perm_synth", "clifford_synth", "pauli_network")


def _ratio(a, b):
    return (a + 1) / (b + 1)


def ai_report(recs, level_filter):
    """Markdown report centred on qumo-ai (see module docstring)."""
    base = "qumo-ai"
    refs = [s for s in ("qumo", "qiskit", "tket", "qiskit+ibm-ai") if any(r["sdk"] == s for r in recs)]
    groups = defaultdict(dict)
    for r in recs:
        groups[(r["circuit"], r["target"], r["level"])][r["sdk"]] = r
    levels = sorted({r["level"] for r in recs})
    status = defaultdict(int)
    for r in recs:
        if r["sdk"] == base:
            status[r["status"]] += 1
    out = []
    p = out.append
    p("# QUMO AI transpiler — hardware-neutral benchmark\n")
    p(f"Records: {len(recs)} · groups: {len(groups)} · levels: {', '.join(levels)} · "
      f"qumo-ai statuses: " + ", ".join(f"{k} {v}" for k, v in sorted(status.items())) + "\n")
    p("qumo-ai = `transpile_ai` (default -> AI level 2, max -> AI level 3, a portfolio with O3). "
      "Ratios are geometric means of (x+1)/(y+1) over (circuit, target, level) groups where both results "
      "are `ok` (statevector-verified up to 14 qubits); < 1 means qumo-ai is better. "
      "W/T/L = qumo-ai has fewer / equal / more 2q gates.\n")
    for lvl in levels:
        p(f"\n## Level `{lvl}`\n")
        p("| family | vs | pairs | 2q ratio | 2q-depth ratio | W/T/L | median ms (qumo-ai) | median ms (ref) | time ratio |")
        p("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        fams = sorted({FAMILIES.get(t, "other") for (_, t, l) in groups if l == lvl}) + ["overall"]
        for fam in fams:
            for ref in refs:
                q2, d2, ta, tr, w, ti, lo = [], [], [], [], 0, 0, 0
                for (c, t, l), by in groups.items():
                    if l != lvl or (fam != "overall" and FAMILIES.get(t, "other") != fam):
                        continue
                    a, b = by.get(base), by.get(ref)
                    if not a or not b or a["status"] != "ok" or b["status"] != "ok":
                        continue
                    q2.append(_ratio(a["n2q"], b["n2q"]))
                    d2.append(_ratio(a["depth2q"], b["depth2q"]))
                    ta.append(max(a["wall_s"], 1e-6))
                    tr.append(max(b["wall_s"], 1e-6))
                    if a["n2q"] < b["n2q"]:
                        w += 1
                    elif a["n2q"] == b["n2q"]:
                        ti += 1
                    else:
                        lo += 1
                if not q2:
                    continue
                ma = sorted(ta)[len(ta) // 2] * 1000
                mr = sorted(tr)[len(tr) // 2] * 1000
                name = f"**{fam}**" if fam == "overall" else fam
                p(f"| {name} | {ref} | {len(q2)} | {gmean(q2):.3f} | {gmean(d2):.3f} | {w}/{ti}/{lo} | "
                  f"{ma:.2f} | {mr:.2f} | {gmean([x / y for x, y in zip(ta, tr)]):.2f} |")
        # per-technique statistics
        p(f"\n### Techniques (level `{lvl}`)\n")
        p("| technique | jobs run | blocks | proposed | verified | replaced | replaced / blocks | saved 2q cost | rejected (not equal) |")
        p("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        jobs = [r for r in recs if r["sdk"] == base and r["level"] == lvl and r["status"] == "ok" and "ai" in r]
        for tech in TECHNIQUES:
            st = [r["ai"][tech] for r in jobs if tech in r["ai"]]
            if not st:
                continue
            tot = {k: sum(x.get(k, 0) for x in st) for k in ("blocks", "proposed", "verified", "replaced", "saved_cost", "rejected")}
            frac = tot["replaced"] / tot["blocks"] if tot["blocks"] else 0.0
            p(f"| {tech} | {len(st)} | {tot['blocks']} | {tot['proposed']} | {tot['verified']} | {tot['replaced']} | "
              f"{frac:.3f} | {tot['saved_cost']} | {tot['rejected']} |")
        tried = sum(r["ai"].get("route_ai_tried", 0) for r in jobs)
        kept = sum(r["ai"].get("route_ai_kept", 0) for r in jobs)
        won = sum(1 for r in jobs if r["ai"].get("winner") == "ai")
        p(f"| learned routing (kept vs SABRE) | {tried} | | | | {kept} | {kept / tried if tried else 0:.3f} | | |")
        p(f"\nPortfolio winner is the AI pipeline in {won} of {len(jobs)} qumo-ai jobs"
          + (" (the rest: the O3/O2 heuristic member won)." if lvl == "max" else "."))
    return "\n".join(out)


def main():
    path = sys.argv[1]
    level_filter = None
    if "--level" in sys.argv:
        level_filter = sys.argv[sys.argv.index("--level") + 1]
    recs = [json.loads(l) for l in open(path)]
    if level_filter:
        recs = [r for r in recs if r["level"] == level_filter]
    if "--ai" in sys.argv:
        print(ai_report(recs, level_filter))
        return
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
