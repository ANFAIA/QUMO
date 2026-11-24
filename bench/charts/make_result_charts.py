"""Charts of the Benchpress and state-vector results: QUMO against Qiskit.

    bench/.venv/bin/python bench/charts/make_result_charts.py \
        --qumo bench/results/benchpress/qumo_full_2026-11-23.json.gz \
        --qiskit bench/results/benchpress/qiskit_full.json.gz \
        --sv bench/results/sv_bench_2026-11-23.csv --out bench/results/charts

Writes benchpress_time_ratio, benchpress_2q_ratio and sv_throughput as PNG (2x) and SVG,
with the style of make_charts.py. Ratios are QUMO / Qiskit per test, summarised by the
geometric mean, so a bar left of 1 means QUMO is faster (time) or emits fewer two-qubit
gates (2Q). Only tests both gyms completed are used; the two-qubit ratio compares
output_gate_count_2q as benchpress_compare.py does.

The Qiskit run is the one stored in bench/results/benchpress; the QUMO run and the
state-vector timings were taken on a machine that was not idle, so absolute times are
indicative and the figures say so.
"""
import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
from collections import defaultdict

import altair as alt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_charts as mc  # noqa: E402  (registers the shared Altair theme)

TARGETS = ["heavy-hex", "square", "linear", "all-to-all"]
BLUE, ORANGE = mc.COLORS["QUMO"], mc.COLORS["Qiskit"]


def load_run(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        data = json.load(fh)
    out = {}
    for b in data["benchmarks"]:
        out[b["fullname"].split("::")[-1]] = (b["stats"]["mean"], b["extra_info"].get("output_gate_count_2q"))
    return out


def suite_and_target(name):
    base = name.split("[")[0].replace("test_", "")
    m = re.search(r"-(all-to-all|heavy-hex|square|linear)\]$", name)
    return base, (m.group(1) if m else None)


def gmean(xs):
    return math.exp(sum(math.log(max(x, 1e-12)) for x in xs) / len(xs))


def ratios(qumo, qiskit):
    """(suite, target) -> {time: [..], twoq: [..]} over the tests both runs completed."""
    by = defaultdict(lambda: {"time": [], "twoq": []})
    for name, (t, c) in qumo.items():
        if name not in qiskit:
            continue
        ot, oc = qiskit[name]
        suite, target = suite_and_target(name)
        if target is None:
            continue
        by[(suite, target)]["time"].append(t / max(ot, 1e-12))
        if c is not None and oc is not None:
            by[(suite, target)]["twoq"].append((c + 1) / (oc + 1))
    return by


def ratio_figure(by, key, title, note, xtitle, domain, ticks):
    rows = [{"suite": s, "target": t, "ratio": gmean(v[key]), "n": len(v[key])}
            for (s, t), v in by.items() if v[key]]
    suites = sorted({r["suite"] for r in rows})
    # a bar from 1 to the ratio: bars cannot start at 0 on a log axis
    bars = alt.Chart().mark_bar(size=11).encode(
        y=alt.Y("target:N", sort=TARGETS, title=None),
        x=alt.X("ratio:Q", title=xtitle, scale=alt.Scale(type="log" if domain[1] > 1.5 else "linear", domain=domain),
                axis=alt.Axis(values=ticks, format="~g")),
        x2=alt.datum(1),
        color=alt.condition(alt.datum.ratio < 1, alt.value(BLUE), alt.value(ORANGE)),
        tooltip=[alt.Tooltip("suite:N"), alt.Tooltip("target:N"),
                 alt.Tooltip("ratio:Q", format=".3f"), alt.Tooltip("n:Q", title="tests")])
    rule = alt.Chart().mark_rule(color=mc.INK, strokeDash=[4, 3]).encode(x=alt.datum(1))
    return alt.layer(bars, rule, data=alt.Data(values=rows)).properties(width=520).facet(
        row=alt.Row("suite:N", sort=suites, title=None,
                    header=alt.Header(labelAngle=0, labelAlign="left", labelLimit=220)),
    ).properties(title=mc._title(title, note))


def sv_figure(path, note):
    with open(path) as fh:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    long = []
    for r in rows:
        long.append({"qubits": r["qubits"], "sdk": "QUMO", "seconds": r["qumo_seconds"],
                     "speedup": r["aer_seconds"] / r["qumo_seconds"]})
        long.append({"qubits": r["qubits"], "sdk": "Qiskit Aer", "seconds": r["aer_seconds"],
                     "speedup": r["aer_seconds"] / r["qumo_seconds"]})
    scale = alt.Scale(domain=["QUMO", "Qiskit Aer"], range=[BLUE, ORANGE])
    color = alt.Color("sdk:N", scale=scale, legend=alt.Legend(title=None, orient="top"))
    x = alt.X("qubits:Q", title="qubits", scale=alt.Scale(domain=[15, 27]),
              axis=alt.Axis(values=[16, 18, 20, 22, 24, 26]))
    line = alt.Chart(alt.Data(values=long)).mark_line(point=alt.OverlayMarkDef(filled=True, size=50)).encode(
        x=x, color=color,
        y=alt.Y("seconds:Q", title="seconds, 10 layers (lower is faster)",
                scale=alt.Scale(type="log", domain=[0.005, 8]), axis=alt.Axis(format="~g")),
        tooltip=[alt.Tooltip("sdk:N"), alt.Tooltip("qubits:Q"), alt.Tooltip("seconds:Q", format=".4g")])
    labels = alt.Chart(alt.Data(values=[{"qubits": r["qubits"], "y": max(r["aer_seconds"], r["qumo_seconds"]),
                                         "label": f"{r['aer_seconds'] / r['qumo_seconds']:.1f}x"} for r in rows])
                       ).mark_text(dy=-12, color=mc.MUTED, fontSize=11.5).encode(
        x=alt.X("qubits:Q", scale=alt.Scale(domain=[15, 27])), y=alt.Y("y:Q", scale=alt.Scale(type="log", domain=[0.005, 8])), text="label:N")
    return alt.layer(line, labels).properties(
        width=520, height=300,
        title=mc._title("State-vector time per circuit, QUMO against Qiskit Aer", note))


def cirq_figure(cirq, qumo, qiskit):
    """Time per test for the few Benchpress tests that its Cirq gym implements (circuit build, import, manipulate)."""
    names = [n for n in cirq if n in qumo and n in qiskit]
    rows = []
    for n in names:
        for label, run in (("QUMO", qumo), ("Qiskit", qiskit), ("Cirq", cirq)):
            rows.append({"test": n.replace("test_", ""), "sdk": label, "ms": max(run[n][0] * 1e3, 0.012)})
    order = [n.replace("test_", "") for n in sorted(names, key=lambda n: -cirq[n][0])]
    sdk_order = ["QUMO", "Qiskit", "Cirq"]
    chart = alt.Chart(alt.Data(values=rows)).mark_bar(cornerRadiusEnd=2).encode(
        y=alt.Y("test:N", sort=order, title=None, axis=alt.Axis(labelFontSize=12, ticks=False, domain=False)),
        yOffset=alt.YOffset("sdk:N", sort=sdk_order),
        x=alt.X("ms:Q", scale=alt.Scale(type="log", domain=[0.01, 10000]), title="mean time per call, ms (log scale)",
                axis=alt.Axis(values=[0.01, 0.1, 1, 10, 100, 1000, 10000], format="~g")),
        x2=alt.datum(0.01),
        color=alt.Color("sdk:N", sort=sdk_order, scale=alt.Scale(domain=sdk_order, range=[mc.COLORS[s] for s in sdk_order]),
                        legend=alt.Legend(title=None, orient="top")))
    return chart.properties(width=620, height=40 * len(order),
                            title=mc._title("Benchpress tests its Cirq gym implements", f"{len(order)} of 1066 tests run; 1054 are skipped as Not implemented and 2 fail. Circuit build, import and manipulation, not transpile."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qumo", required=True)
    ap.add_argument("--qiskit", required=True)
    ap.add_argument("--sv", required=True)
    ap.add_argument("--cirq", help="Benchpress Cirq gym run (json or json.gz), optional")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    by = ratios(load_run(args.qumo), load_run(args.qiskit))
    n = sum(len(v["time"]) for v in by.values())
    note = (f"Geometric mean of QUMO / Qiskit over {n} tests both completed. "
            "QUMO run of 2026-11-23 on a loaded machine; Qiskit run as stored in bench/results/benchpress.")
    mc.save(ratio_figure(by, "time", "Benchpress transpile time, QUMO against Qiskit", note,
                         "QUMO time / Qiskit time (left of 1: QUMO faster)",
                         [0.01, 2], [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]), args.out, "benchpress_time_ratio")
    mc.save(ratio_figure(by, "twoq", "Benchpress two-qubit gates, QUMO against Qiskit", note,
                         "QUMO 2Q gates / Qiskit 2Q gates (left of 1: fewer)",
                         [0.5, 1.1], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]), args.out, "benchpress_2q_ratio")
    if args.cirq:
        mc.save(cirq_figure(load_run(args.cirq), load_run(args.qumo), load_run(args.qiskit)), args.out, "benchpress_cirq_tests")
    mc.save(sv_figure(args.sv, "Labels give Aer time / QUMO time. 26 qubits is the largest size; "
                               "run on a loaded machine, so absolute times are indicative."),
            args.out, "sv_throughput")


if __name__ == "__main__":
    main()
