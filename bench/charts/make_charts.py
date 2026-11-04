"""Charts for the hardware-neutral compiler benchmark: compile time and memory per SDK.

    bench/.venv/bin/python bench/charts/make_charts.py \
        --data bench/results/neutral_rss.jsonl --out bench/results/charts

Writes each figure as PNG (2x, for docs) and SVG, one report.html with every figure
interactive, and summary.md with the table and the memory notes. Figures are Altair
(Vega-Lite) specs, so the encodings are declared, not drawn. Each figure's subtitle says
whether the data is a complete run.

Memory caveats (also in summary.md):
  * every worker imports qiskit at module level (run_neutral.py, sdks.py), so each idle
    footprint includes a shared qiskit import. It is measured here in a bare process and
    drawn as its own part of the footprint bars;
  * the statevector verification runs between timed compile calls. Its memory raises the
    resident set before the next job (base RSS), so the drift between jobs is reported
    as harness memory, not as compiler memory.
Memory fields are absent from records made before rss_* was added; those figures are
skipped.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict

import altair as alt

SDK_ORDER = ["qumo", "qiskit", "tket", "cirq"]
SDK_LABEL = {"qumo": "QUMO", "qiskit": "Qiskit", "tket": "TKET", "cirq": "Cirq"}
LABELS = [SDK_LABEL[s] for s in SDK_ORDER]
# Okabe-Ito palette: distinguishable for common colour-vision deficiencies.
COLORS = {"QUMO": "#0072B2", "Qiskit": "#E69F00", "TKET": "#009E73", "Cirq": "#CC79A7"}
SDK_SCALE = alt.Scale(domain=LABELS, range=[COLORS[label] for label in LABELS])
LEVELS = ["default", "max"]
EXPECTED_RECORDS = 5632  # 4 SDKs x 2 levels x 704 (circuit, target) pairs of bench/neutral
SMALL, LARGE = "earlier job <= 8 qubits", "earlier job 9-14 qubits"
SHARED_PART = "shared"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
INK, MUTED, GRID = "#1d1d1f", "#6e6e73", "#ececf0"
QUBIT_TICKS = [1, 2, 4, 6, 8, 10, 12, 14]
TIME_TICKS = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]


@alt.theme.register("qumo", enable=True)
def _theme():
    return {"config": {
        "background": "white",
        "font": FONT,
        "view": {"stroke": None},
        "title": {"font": FONT, "fontSize": 17, "fontWeight": 600, "color": INK, "anchor": "start",
                  "subtitleFont": FONT, "subtitleFontSize": 12, "subtitleColor": MUTED,
                  "subtitlePadding": 6, "offset": 14},
        "axis": {"labelFont": FONT, "titleFont": FONT, "labelFontSize": 11.5, "titleFontSize": 12,
                 "titleFontWeight": 500, "labelColor": MUTED, "titleColor": INK, "gridColor": GRID,
                 "domainColor": "#c7c7cc", "tickColor": "#c7c7cc", "titlePadding": 10},
        "legend": {"labelFont": FONT, "titleFont": FONT, "labelFontSize": 12, "titleFontSize": 12,
                   "labelColor": INK, "titleColor": INK, "symbolSize": 90, "symbolType": "circle",
                   "padding": 4},
        "header": {"labelFont": FONT, "titleFont": FONT, "labelFontSize": 13, "labelFontWeight": 600,
                   "labelColor": INK},
        "line": {"strokeWidth": 2.4, "strokeCap": "round", "strokeJoin": "round"},
    }}


def _title(text, note=None):
    if note:
        return alt.TitleParams(text, subtitle=note, anchor="start")
    return alt.TitleParams(text, anchor="start")


def load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare(rows):
    """Status-ok records as flat dicts with display labels. job is the position of a
    memory record in its worker (file order), None for records without rss_* fields."""
    counts = defaultdict(int)
    out = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        sdk = SDK_LABEL[r["sdk"]]
        peak, base = r.get("rss_peak_mb"), r.get("rss_base_mb")
        job = None
        if peak is not None and base is not None:
            job = counts[sdk]
            counts[sdk] += 1
        out.append({"sdk": sdk, "level": r["level"], "qubits": r["qubits"], "wall_s": r.get("wall_s"),
                    "idle_mb": r.get("rss_idle_mb"), "base_mb": base, "peak_mb": peak,
                    "growth_mb": None if peak is None or base is None else peak - base, "job": job})
    return out


def drift_steps(recs):
    """MB change of base RSS from one job to the next, bucketed by the qubits of the earlier
    job: its statevector verification is what runs between the two timed calls."""
    out = []
    for sdk in LABELS:
        mem = [r for r in recs if r["sdk"] == sdk and r["base_mb"] is not None]
        for prev, cur in zip(mem, mem[1:]):
            earlier = SMALL if prev["qubits"] <= 8 else LARGE
            out.append({"sdk": sdk, "earlier": earlier, "step_mb": cur["base_mb"] - prev["base_mb"]})
    return out


def measure_rss_mb(code):
    """Resident set (MB) of a fresh interpreter after running code, or None if it fails."""
    probe = code + "\nimport psutil\nprint(psutil.Process().memory_info().rss)"
    out = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return int(out.stdout.strip().splitlines()[-1]) / 2**20


def curve_figure(recs, field, ytitle, log, title, note):
    """Median line with an interquartile band, per SDK, one panel per level."""
    data = alt.Data(values=recs)
    scale = alt.Scale(type="log") if log else alt.Scale(zero=True)
    y_axis = alt.Axis(values=TIME_TICKS) if log else alt.Axis()
    x = alt.X("qubits:Q", title="qubits", axis=alt.Axis(values=QUBIT_TICKS))
    color = alt.Color("sdk:N", scale=SDK_SCALE, sort=LABELS, legend=alt.Legend(title=None, orient="top"))
    band = alt.Chart(data).mark_area(opacity=0.14, line=False).encode(
        x=x, color=color,
        y=alt.Y(f"q1({field}):Q", title=ytitle, scale=scale, axis=y_axis),
        y2=alt.Y2(f"q3({field})"))
    line = alt.Chart(data).mark_line(point=alt.OverlayMarkDef(filled=True, size=46)).encode(
        x=x, color=color,
        y=alt.Y(f"median({field}):Q", scale=scale, axis=y_axis),
        tooltip=[alt.Tooltip("sdk:N", title="SDK"), alt.Tooltip("qubits:Q"),
                 alt.Tooltip(f"median({field}):Q", title="median", format=".4g")])
    return alt.layer(band, line).facet(
        column=alt.Column("level:N", sort=LEVELS, title=None,
                          header=alt.Header(labelExpr="'level = ' + datum.label")),
    ).properties(title=_title(title, note))


def box_figure(recs, field, ytitle, log, title, note, by_level=True):
    data = alt.Data(values=recs)
    scale = alt.Scale(type="log") if log else alt.Scale(zero=True)
    chart = alt.Chart(data).mark_boxplot(extent=1.5, size=38).encode(
        x=alt.X("sdk:N", sort=LABELS, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{field}:Q", title=ytitle, scale=scale,
                axis=alt.Axis(values=TIME_TICKS) if log else alt.Axis()),
        color=alt.Color("sdk:N", scale=SDK_SCALE, legend=None)).properties(width=260, height=300)
    if by_level:
        chart = chart.facet(column=alt.Column("level:N", sort=LEVELS, title=None,
                                              header=alt.Header(labelExpr="'level = ' + datum.label")))
    return chart.properties(title=_title(title, note))


def footprint_figure(idle, shared, title, note):
    """Idle footprint per SDK: the qiskit import every worker pays (gray, when measured)
    and the SDK's own part on top, in its colour."""
    bars, totals = [], []
    for sdk in LABELS:
        if sdk not in idle:
            continue
        total = idle[sdk]
        split = min(shared, total) if shared is not None else 0.0
        if split > 0:
            bars.append({"sdk": sdk, "part": SHARED_PART, "lo": 0.0, "hi": split})
        bars.append({"sdk": sdk, "part": "own", "lo": split, "hi": total})
        totals.append({"sdk": sdk, "hi": total, "label": f"{total:.0f} MB"})
    x = alt.X("sdk:N", sort=LABELS, title=None, axis=alt.Axis(labelAngle=0))
    body = alt.Chart(alt.Data(values=bars)).mark_bar(size=56).encode(
        x=x,
        y=alt.Y("lo:Q", title="resident memory after warm-up, MB", scale=alt.Scale(zero=True)),
        y2=alt.Y2("hi:Q"),
        color=alt.condition(alt.datum.part == SHARED_PART, alt.value("#d2d2d7"),
                            alt.Color("sdk:N", scale=SDK_SCALE, legend=None)))
    label = alt.Chart(alt.Data(values=totals)).mark_text(dy=-10, fontSize=12, fontWeight=600, color=INK).encode(
        x=x, y="hi:Q", text="label:N")
    return alt.layer(body, label).properties(
        width=300, height=300,
        title=_title(title, note))


def drift_figure(recs, steps, note):
    mem = [r for r in recs if r["base_mb"] is not None]
    color = alt.Color("sdk:N", scale=SDK_SCALE, sort=LABELS, legend=alt.Legend(title=None, orient="top"))
    line = alt.Chart(alt.Data(values=mem)).mark_line().encode(
        x=alt.X("job:Q", title="job number in its worker"),
        y=alt.Y("base_mb:Q", title="resident memory before the job, MB", scale=alt.Scale(zero=False)),
        color=color,
        tooltip=[alt.Tooltip("sdk:N", title="SDK"), alt.Tooltip("job:Q"), alt.Tooltip("qubits:Q"),
                 alt.Tooltip("base_mb:Q", title="base MB", format=".1f")]).properties(width=380, height=280)
    bars = alt.Chart(alt.Data(values=steps)).mark_bar().encode(
        x=alt.X("sdk:N", sort=LABELS, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("earlier:N", sort=[SMALL, LARGE]),
        y=alt.Y("mean(step_mb):Q", title="mean change of base RSS per job, MB"),
        color=alt.Color("sdk:N", scale=SDK_SCALE, legend=None),
        opacity=alt.Opacity("earlier:N", sort=[SMALL, LARGE], scale=alt.Scale(range=[0.4, 1.0]),
                            legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("sdk:N", title="SDK"), alt.Tooltip("earlier:N", title="earlier job"),
                 alt.Tooltip("mean(step_mb):Q", title="mean MB", format=".2f")])
    zero = alt.Chart(alt.Data(values=[{"y": 0}])).mark_rule(color=INK, opacity=0.35).encode(y="y:Q")
    right = alt.layer(bars, zero).properties(width=300, height=280)
    return alt.hconcat(line, right).properties(
        title=_title("Memory drift between jobs (harness, includes verification)", note))


def save(chart, out_dir, name):
    stem = os.path.join(out_dir, name)
    chart.save(stem + ".png", scale_factor=2)
    chart.save(stem + ".svg")
    print("wrote", stem + ".png")


def summary(rows):
    lines = ["| level | SDK | records | ok | unsupported | failed | median compile s | p90 compile s "
             "| idle MB | median peak MB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for level in LEVELS:
        for sdk in SDK_ORDER:
            recs = [r for r in rows if r.get("sdk") == sdk and r.get("level") == level]
            if not recs:
                continue
            ok = [r for r in recs if r.get("status") == "ok"]
            unsup = sum(1 for r in recs if r.get("status") == "unsupported")
            failed = len(recs) - len(ok) - unsup
            times = [r["wall_s"] for r in ok if r.get("wall_s") is not None]
            p90 = statistics.quantiles(times, n=10)[8] if len(times) >= 2 else (times[0] if times else float("nan"))
            med = statistics.median(times) if times else float("nan")
            idle = [r["rss_idle_mb"] for r in ok if r.get("rss_idle_mb") is not None]
            peak = [r["rss_peak_mb"] for r in ok if r.get("rss_peak_mb") is not None]
            idle_txt = f"{statistics.median(idle):.0f}" if idle else "n/a"
            peak_txt = f"{statistics.median(peak):.0f}" if peak else "n/a"
            lines.append(f"| {level} | {SDK_LABEL[sdk]} | {len(recs)} | {len(ok)} | {unsup} | {failed} "
                         f"| {med:.4f} | {p90:.4f} | {idle_txt} | {peak_txt} |")
    return "\n".join(lines) + "\n"


def memory_notes(recs, shared, bare, records, expected):
    """Markdown section with per-SDK memory figures and their caveats, from the data."""
    mem = [r for r in recs if r["peak_mb"] is not None and r["base_mb"] is not None]
    lines = ["", "## Memory", "",
             f"Records: {records} of {expected} expected. "
             + ("Run complete." if records >= expected else "Run incomplete: the figures below are partial."), ""]
    if shared is not None:
        bare_txt = f"{bare:.0f} MB" if bare is not None else "n/a"
        lines.append(f"Bare python: {bare_txt}. Python with `from qiskit import qasm2`: {shared:.0f} MB. "
                     "Every worker imports it, so this part of each idle footprint is shared, not SDK-specific.")
        lines.append("")
    lines += ["| SDK | records | idle MB | idle minus shared MB | median peak MB "
              "| growth median / p90 / max MB | base RSS, first 10% / last 10% MB |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for sdk in LABELS:
        mine = [r for r in mem if r["sdk"] == sdk]
        if not mine:
            continue
        idle = statistics.median(r["idle_mb"] for r in mine)
        rest = f"{idle - shared:.0f}" if shared is not None else "n/a"
        peak = statistics.median(r["peak_mb"] for r in mine)
        grow = sorted(r["growth_mb"] for r in mine)
        p90 = grow[min(len(grow) - 1, int(0.9 * len(grow)))]
        base = [r["base_mb"] for r in mine]
        k = max(1, len(base) // 10)
        lines.append(f"| {sdk} | {len(mine)} | {idle:.0f} | {rest} | {peak:.0f} "
                     f"| {statistics.median(grow):.1f} / {p90:.1f} / {grow[-1]:.0f} "
                     f"| {statistics.median(base[:k]):.0f} / {statistics.median(base[-k:]):.0f} |")
    steps = drift_steps(recs)
    lines += ["", "Mean change of base RSS from one job to the next, by the qubits of the earlier job (MB):", ""]
    for sdk in LABELS:
        small = [s["step_mb"] for s in steps if s["sdk"] == sdk and s["earlier"] == SMALL]
        large = [s["step_mb"] for s in steps if s["sdk"] == sdk and s["earlier"] == LARGE]
        if small or large:
            lines.append(f"- {sdk}: after <= 8 qubits {statistics.mean(small) if small else float('nan'):+.2f}; "
                         f"after 9-14 qubits {statistics.mean(large) if large else float('nan'):+.2f}.")
    net = {}
    for sdk in LABELS:
        mine = [r["base_mb"] for r in mem if r["sdk"] == sdk]
        if mine:
            k = max(1, len(mine) // 10)
            net[sdk] = statistics.median(mine[-k:]) - statistics.median(mine[:k])
    large_means = [statistics.mean(s["step_mb"] for s in steps if s["sdk"] == sdk and s["earlier"] == LARGE)
                   for sdk in LABELS if any(s["sdk"] == sdk and s["earlier"] == LARGE for s in steps)]
    lines += ["", "Reading:",
              "- Inside the timed compile call the resident set barely grows; medians are in the growth column."]
    if large_means:
        lines.append(f"- Between jobs the base RSS moves in steps. After 9-14 qubit jobs the mean step is "
                     f"{min(large_means):+.2f} to {max(large_means):+.2f} MB across SDKs, the jobs whose statevector "
                     "verification runs between timed calls. The harness, not one compiler, is the likely source.")
    if net:
        drift = ", ".join(f"{sdk} {net[sdk]:+.0f}" for sdk in net)
        lines.append(f"- Net change of base RSS from the first to the last 10% of jobs, MB: {drift}. "
                     "The change is not uniform across SDKs. Not isolated yet: that needs a fresh process per job "
                     "or a run without verification.")
    lines.append("- Idle footprints compare fairly only after subtracting the shared qiskit import.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench/results/neutral_rss.jsonl")
    ap.add_argument("--out", default="bench/results/charts")
    ap.add_argument("--expected", type=int, default=EXPECTED_RECORDS,
                    help="records of the full run, to say whether the data is complete")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.data)
    recs = prepare(rows)
    print(f"{len(rows)} records from {a.data}")
    if len(rows) >= a.expected:
        note = f"complete run: {len(rows)} records"
    else:
        note = f"partial run: {len(rows)} of {a.expected} records"

    figures = {}
    timed = [r for r in recs if r["wall_s"] is not None]
    figures["compile_time_vs_qubits"] = curve_figure(
        timed, "wall_s", "compile time, s (median, IQR band)", True,
        "Compile time vs circuit width, per SDK", note)
    figures["compile_time_distribution"] = box_figure(
        timed, "wall_s", "compile time, s", True, "Compile time over all circuits and targets", note)

    notes = ""
    mem = [r for r in recs if r["peak_mb"] is not None]
    if mem:
        bare = measure_rss_mb("pass")
        shared = measure_rss_mb("from qiskit import qasm2")
        figures["memory_peak_vs_qubits"] = curve_figure(
            mem, "peak_mb", "peak resident memory, MB (median, IQR band)", False,
            "Peak resident memory of the SDK worker during a compile", note)
        idle = {sdk: statistics.median(r["idle_mb"] for r in mem if r["sdk"] == sdk)
                for sdk in LABELS if any(r["sdk"] == sdk for r in mem)}
        footprint = footprint_figure(idle, shared, "Idle footprint per worker", None)
        growth = box_figure([r for r in mem if r["level"] == "default"], "growth_mb",
                            "growth inside the timed call, MB", False,
                            "Growth per compile call (level = default)", None, by_level=False)
        gray = f". Gray: python + qiskit import ({shared:.0f} MB, measured in a bare process)" if shared is not None else ""
        figures["memory_footprint"] = alt.hconcat(footprint, growth).properties(
            title=_title("Memory: idle footprint and growth per compile", note + gray))
        figures["memory_drift_between_jobs"] = drift_figure(mem, drift_steps(recs), note)
        notes = memory_notes(recs, shared, bare, len(rows), a.expected)
    else:
        print("no rss_* fields in this file: memory figures skipped")

    for name, chart in figures.items():
        save(chart, a.out, name)
    alt.vconcat(*figures.values()).configure_concat(spacing=36).save(os.path.join(a.out, "report.html"))
    print("wrote", os.path.join(a.out, "report.html"))

    table = summary(rows)
    path = os.path.join(a.out, "summary.md")
    with open(path, "w") as fh:
        fh.write(table + notes)
    print("wrote", path)
    print(table + notes)


if __name__ == "__main__":
    main()
