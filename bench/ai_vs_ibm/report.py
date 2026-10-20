"""Aggregate bench/ai_vs_ibm/run.py records into the markdown report.

    python bench/ai_vs_ibm/report.py bench/results/ai_vs_ibm.jsonl > bench/results/ai_vs_ibm.md

Per (device, group, circuit, contender) the repetitions are reduced to medians
(2q count, 2q depth, depth, wall time). A circuit counts as *verified* for a
contender only when every repetition is ISA-valid and passed an equivalence
check (statevector or Clifford); *ISA-only* when every repetition is ISA-valid
but too large / non-Clifford to check. Pairwise statistics use only circuits
that both contenders completed; verified pairs and ISA-only pairs are reported
in separate tables and only verified pairs enter the W/T/L headline.
Ratios are A/B (< 1: A better), geometric means of (a+1)/(b+1) so that
circuits compiled to zero 2q gates stay defined.
"""
import json
import math
import statistics as st
import sys
from collections import defaultdict

HEAD = [("qumo-ai2", "ibm-ai2"), ("qumo-ai3", "ibm-ai3")]
CONTEXT = [("ibm-ai2", "qiskit-o2"), ("ibm-ai3", "qiskit-o3"), ("qumo-ai2", "qiskit-o2"), ("qumo-ai3", "qiskit-o3"),
           ("qumo-o2", "qiskit-o2"), ("qumo-o3", "qiskit-o3"), ("qumo-ai2", "qumo-o2"), ("qumo-ai3", "qumo-o3"),
           ("qumo-o2", "ibm-ai2"), ("qumo-o3", "ibm-ai3")]
ATTRIB = [("ibm-route3", "qiskit-o3"), ("ibm-synth3", "qiskit-o3"), ("ibm-ai3", "qiskit-o3"), ("ibm-full3", "qiskit-o3"),
          ("qumo-route3", "qumo-o3s"), ("qumo-synth3", "qumo-o3s"), ("qumo-ai3s", "qumo-o3s"),
          ("qumo-route3", "ibm-route3"), ("qumo-synth3", "ibm-synth3"), ("qumo-ai3s", "ibm-full3"),
          ("qumo-o3s", "qiskit-o3")]
ORDER = ["qiskit-o2", "qiskit-o3", "ibm-ai2", "ibm-ai3", "ibm-route3", "ibm-synth3", "ibm-full3",
         "qumo-o2", "qumo-o3", "qumo-ai2", "qumo-ai3", "qumo-o3s", "qumo-ai3s", "qumo-route3", "qumo-synth3"]


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def load(path):
    recs, metas = [], []
    for line in open(path):
        r = json.loads(line)
        (metas if r.get("meta") else recs).append(r)
    return recs, metas


def reduce(recs):
    by = defaultdict(list)
    for r in recs:
        by[(r["device"], r["group"], r["circuit"], r["contender"])].append(r)
    cells = {}
    for k, rs in by.items():
        ok = [r for r in rs if r["status"] == "ok"]
        cell = {"reps": len(rs), "qubits": rs[0]["qubits"], "statuses": sorted({r["status"] for r in rs}),
                "load1": st.mean(r.get("load1", 0) for r in rs),
                "misreported": sum(1 for r in rs if r.get("initial_layout_mismatch") or r.get("ibm_initial_layout_corrected")),
                "problems": [p for r in rs for p in (r.get("problems") or [])][:2]}
        if len(ok) == len(rs):
            cell["n2q"] = st.median(r["n2q"] for r in ok)
            cell["depth2q"] = st.median(r["depth2q"] for r in ok)
            cell["depth"] = st.median(r["depth"] for r in ok)
            cell["wall"] = st.median(r["wall_s"] for r in ok)
            cell["n2q_all"] = [r["n2q"] for r in ok]
            cell["verified"] = all(r.get("verified_by") for r in ok)
            cell["how"] = sorted({r.get("verified_by") or "isa-only" for r in ok})
            cell["ai"] = [r.get("ai") for r in ok if r.get("ai")]
        cells[k] = cell
    return cells


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


SWAP_INPUTS = set()  # (group, circuit) whose input contains SWAP gates (see main)


def pair_stats(cells, a, b, device=None, group=None, verified=True, no_swap=False):
    rows = []
    for (dv, g, c, k), ca in cells.items():
        if k != a or (device and dv != device) or (group and g != group):
            continue
        if no_swap and (g, c) in SWAP_INPUTS:
            continue
        cb = cells.get((dv, g, c, b))
        if cb is None or "n2q" not in ca or "n2q" not in cb:
            continue
        both = ca["verified"] and cb["verified"]
        if both != verified:
            continue
        rows.append((dv, g, c, ca, cb))
    if not rows:
        return None
    r2 = [(ca["n2q"] + 1) / (cb["n2q"] + 1) for *_, ca, cb in rows]
    rd2 = [(ca["depth2q"] + 1) / (cb["depth2q"] + 1) for *_, ca, cb in rows]
    rd = [(ca["depth"] + 1) / (cb["depth"] + 1) for *_, ca, cb in rows]
    rt = [max(ca["wall"], 1e-4) / max(cb["wall"], 1e-4) for *_, ca, cb in rows]
    w = sum(1 for *_, ca, cb in rows if ca["n2q"] < cb["n2q"])
    t = sum(1 for *_, ca, cb in rows if ca["n2q"] == cb["n2q"])
    l_ = len(rows) - w - t
    wd = sum(1 for *_, ca, cb in rows if ca["depth2q"] < cb["depth2q"])
    ld = sum(1 for *_, ca, cb in rows if ca["depth2q"] > cb["depth2q"])
    return {"n": len(rows), "r2q": gmean(r2), "rd2q": gmean(rd2), "rdepth": gmean(rd), "rtime_med": st.median(rt),
            "rtime_g": gmean(rt), "W": w, "T": t, "L": l_, "Wd": wd, "Ld": ld, "rows": rows,
            "ta_med": st.median(ca["wall"] for *_, ca, cb in rows), "tb_med": st.median(cb["wall"] for *_, ca, cb in rows)}


def failures(cells, k, device=None, group=None):
    bad = defaultdict(int)
    for (dv, g, c, kk), cell in cells.items():
        if kk != k or (device and dv != device) or (group and g != group):
            continue
        for s in cell["statuses"]:
            if s != "ok":
                bad[s] += 1
    return dict(bad)


def table(cells, pairs, keys, title, verified=True, out=print, no_swap=False):
    out(f"\n{title}\n")
    out("| device | group | A vs B | n | 2q ratio | 2q-depth ratio | depth ratio | W/T/L (2q) | 2q-depth W/L | "
        "median time A / B (s) | median time ratio |")
    out("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dv, g in keys:
        for a, b in pairs:
            s = pair_stats(cells, a, b, dv, g, verified, no_swap)
            if s is None:
                continue
            out(f"| {dv} | {g} | {a} / {b} | {s['n']} | {fmt(s['r2q'])} | {fmt(s['rd2q'])} | {fmt(s['rdepth'])} | "
                f"{s['W']}/{s['T']}/{s['L']} | {s['Wd']}/{s['Ld']} | {fmt(s['ta_med'], 3)} / {fmt(s['tb_med'], 3)} | "
                f"{fmt(s['rtime_med'], 2)} |")


def main():
    path = sys.argv[1]
    recs, metas = load(path)
    cells = reduce(recs)
    import os
    inputs = path[:-len(".jsonl")] + "_inputs.json"
    if os.path.exists(inputs):
        meta_in = json.load(open(inputs))
        SWAP_INPUTS.update(tuple(k.split("/", 1)) for k, v in meta_in.items() if v.get("swaps", 0) > 0)
    out = print
    keys = sorted({(k[0], k[1]) for k in cells}, key=lambda x: (x[0], ["families", "summit", "feynman", "hamlib", "neutral"].index(x[1]) if x[1] in ("families", "summit", "feynman", "hamlib", "neutral") else 9))
    out("# QUMO-AI vs IBM AI transpiler (qiskit-ibm-transpiler local mode): raw tables\n")
    out(f"Generated by `bench/ai_vs_ibm/report.py` from `{path.split('/')[-1]}` ({len(recs)} records). "
        "Interpretation and caveats: docs/AI_TRANSPILER.md, section *Head-to-head vs qiskit-ibm-transpiler*.\n")
    env = {}
    for m in metas:
        if "env" in m:
            for fam, info in m["env"].items():
                env.setdefault(fam, info)
    if env:
        out("## Environment\n")
        for fam, info in env.items():
            out(f"* {fam} worker: " + ", ".join(f"{k} {v}" for k, v in info.items() if k not in ("qumo_py",)))
    drops = [m for m in metas if m.get("dropped_by_deadline")]
    for m in drops:
        out(f"* dropped by the time budget on {m['device']}: {', '.join(m['dropped_by_deadline'])}")
    loads = [r.get("load1", 0) for r in recs]
    if loads:
        out(f"* 1-minute load average during the runs: median {st.median(loads):.1f}, max {max(loads):.1f} "
            "(10-core machine shared with other jobs; times are indicative)")

    out("\n## Status per contender (circuits; a circuit is verified when every repetition passed an equivalence check)\n")
    out("| device | group | contender | circuits | verified | ISA-only | failed (status: count) | runs with misreported initial layout | median time (s) |")
    out("|---|---|---|---:|---:|---:|---|---:|---:|")
    for dv, g in keys:
        for k in ORDER:
            cs = [c for (d2, g2, _, k2), c in cells.items() if d2 == dv and g2 == g and k2 == k]
            if not cs:
                continue
            ver = sum(1 for c in cs if c.get("verified"))
            isa = sum(1 for c in cs if "n2q" in c and not c["verified"])
            times = [c["wall"] for c in cs if "wall" in c]
            mis = sum(c["misreported"] for c in cs)
            runs = sum(c["reps"] for c in cs)
            out(f"| {dv} | {g} | {k} | {len(cs)} | {ver} | {isa} | {failures(cells, k, dv, g) or ''} | {mis}/{runs} | "
                f"{fmt(st.median(times) if times else None, 3)} |")

    bad = [(k, c) for k, c in cells.items() if any(x != "ok" for x in c["statuses"])]
    if bad:
        out("\n## Failed compilations (status != ok in some repetition)\n")
        out("| device | group | circuit | q | contender | statuses | first problems |")
        out("|---|---|---|---:|---|---|---|")
        for (dv, g, c, k), cell in sorted(bad):
            out(f"| {dv} | {g} | {c} | {cell['qubits']} | {k} | {','.join(cell['statuses'])} | {'; '.join(cell['problems'])} |")

    table(cells, HEAD, keys, "## Headline: QUMO-AI vs IBM AI at the same level (verified circuits only)")
    if SWAP_INPUTS:
        table(cells, HEAD, keys, "## Headline without inputs that contain SWAP gates (verified; see the IBM ElidePermutations "
              "workaround: IBM's AIRouting pipelines cannot elide input SWAPs, Qiskit and QUMO do)", no_swap=True)
    table(cells, HEAD, keys, "## Headline, ISA-valid but unverified circuits (not counted as wins)", verified=False)
    table(cells, CONTEXT, keys, "## Context: each AI pipeline against its own heuristic base and across SDKs (verified)")
    table(cells, CONTEXT, keys, "## Context, ISA-only circuits (not counted)", verified=False)
    table(cells, ATTRIB, keys, "## Attribution: routing alone vs synthesis alone (verified)")
    table(cells, ATTRIB, keys, "## Attribution, ISA-only circuits (not counted)", verified=False)

    out("\n## Largest QUMO-AI wins and losses against IBM AI (per circuit, median 2q count; v = verified)\n")
    for a, b in HEAD:
        rows = []
        for ver in (True, False):
            s = pair_stats(cells, a, b, None, None, ver)
            if s:
                rows += [(dv, g, c, ca, cb, ver) for dv, g, c, ca, cb in s["rows"]]
        rows.sort(key=lambda r: (r[3]["n2q"] + 1) / (r[4]["n2q"] + 1))
        out(f"\n### {a} vs {b}\n")
        out("| device | group | circuit | q | A 2q | B 2q | ratio | A 2q-depth | B 2q-depth | A s | B s | check |")
        out("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        pick = rows[:8] + [r for r in rows[-8:] if r not in rows[:8]]
        for dv, g, c, ca, cb, ver in pick:
            out(f"| {dv} | {g} | {c} | {ca['qubits']} | {ca['n2q']:g} | {cb['n2q']:g} | "
                f"{(ca['n2q'] + 1) / (cb['n2q'] + 1):.3f} | {ca['depth2q']:g} | {cb['depth2q']:g} | "
                f"{ca['wall']:.2f} | {cb['wall']:.2f} | {'v' if ver else 'isa'} |")

    out("\n## Per-circuit medians (2q count / 2q depth / wall s)\n")
    cols = [k for k in ORDER if any(kk == k for (_, _, _, kk) in cells)]
    out("| device | group | circuit | q | " + " | ".join(cols) + " |")
    out("|---|---|---|---:|" + "---|" * len(cols))
    circs = sorted({(k[0], k[1], k[2]) for k in cells}, key=lambda x: (x[0], x[1], cells.get((x[0], x[1], x[2], cols[0]), {}).get("qubits", 0), x[2]))
    for dv, g, c in circs:
        vals = []
        q = None
        for k in cols:
            cell = cells.get((dv, g, c, k))
            if cell is None:
                vals.append("")
                continue
            q = cell["qubits"]
            if "n2q" in cell:
                vals.append(f"{cell['n2q']:g}/{cell['depth2q']:g}/{cell['wall']:.2f}{'' if cell['verified'] else '*'}")
            else:
                vals.append(",".join(cell["statuses"]))
        out(f"| {dv} | {g} | {c} | {q} | " + " | ".join(vals) + " |")
    out("\n`*` = ISA-valid, not equivalence-checked (too large and not Clifford).")

    ibm_spread = []
    for (dv, g, c, k), cell in cells.items():
        if k.startswith("ibm") and len(cell.get("n2q_all", [])) > 1:
            ibm_spread.append((max(cell["n2q_all"]) - min(cell["n2q_all"])) / max(1, st.median(cell["n2q_all"])))
    if ibm_spread:
        out(f"\nIBM AI run-to-run spread (unseeded learned routing): median relative 2q range over repetitions "
            f"{st.median(ibm_spread):.3f}, max {max(ibm_spread):.3f} ({len(ibm_spread)} circuit x contender cells). "
            "QUMO and plain Qiskit are deterministic for a fixed seed; their repetitions use different seeds.")


if __name__ == "__main__":
    main()
