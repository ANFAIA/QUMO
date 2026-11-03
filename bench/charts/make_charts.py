"""Charts for the hardware-neutral compiler benchmark: compile time and memory per SDK.

    bench/.venv/bin/python bench/charts/make_charts.py \
        --data bench/results/neutral_rss.jsonl --out bench/results/charts

Reads the records written by bench/neutral/run_neutral.py. Curves and boxes use only
status "ok" records; the summary table counts every status, so failures stay visible.
Time is the compile call (parse excluded). Memory is the resident set of the SDK's
worker process: idle after warm-up, and sampled peak during the compile.
Memory fields are absent from records made before rss_* was added; those charts are
skipped with a message.
"""
import argparse
import json
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SDK_ORDER = ["qumo", "qiskit", "tket", "cirq"]
SDK_LABEL = {"qumo": "QUMO", "qiskit": "Qiskit", "tket": "TKET", "cirq": "Cirq"}
# Okabe-Ito palette: distinguishable for common colour-vision deficiencies.
COLORS = {"qumo": "#0072B2", "qiskit": "#E69F00", "tket": "#009E73", "cirq": "#CC79A7"}
LEVELS = ["default", "max"]


def load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def series(rows, field, level):
    """{sdk: {qubits: [values]}} over the status-ok records of one level."""
    out = {sdk: defaultdict(list) for sdk in SDK_ORDER}
    for r in rows:
        if r.get("status") == "ok" and r.get("level") == level and r.get(field) is not None:
            out[r["sdk"]][r["qubits"]].append(r[field])
    return out


def flat(rows, field, level):
    """{sdk: [values]} over the status-ok records of one level."""
    out = {sdk: [] for sdk in SDK_ORDER}
    for sdk, per_q in series(rows, field, level).items():
        for vals in per_q.values():
            out[sdk].extend(vals)
    return out


def growths(rows, level):
    """{sdk: [peak - base]} in MB over the status-ok records of one level."""
    out = {sdk: [] for sdk in SDK_ORDER}
    for r in rows:
        if r.get("status") == "ok" and r.get("level") == level and r.get("rss_peak_mb") is not None:
            out[r["sdk"]].append(r["rss_peak_mb"] - r["rss_base_mb"])
    return out


def quartiles(vals):
    if len(vals) < 2:
        return vals[0], vals[0], vals[0]
    return tuple(statistics.quantiles(vals, n=4))


def curve_panel(ax, data, ylabel, title, logy):
    for sdk in SDK_ORDER:
        pts = sorted(data[sdk].items())
        if not pts:
            continue
        xs = [q for q, _ in pts]
        q1, med, q3 = zip(*(quartiles(v) for _, v in pts))
        ax.plot(xs, med, marker="o", ms=3, lw=1.4, color=COLORS[sdk], label=SDK_LABEL[sdk])
        ax.fill_between(xs, q1, q3, color=COLORS[sdk], alpha=0.15, linewidth=0)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("qubits")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)


def box_panel(ax, groups, ylabel, title, logy):
    sdks = [s for s in SDK_ORDER if groups.get(s)]
    if not sdks:
        ax.set_title(f"{title} (no data)")
        return
    boxes = ax.boxplot([groups[s] for s in sdks], tick_labels=[SDK_LABEL[s] for s in sdks],
                       patch_artist=True, widths=0.6)
    for patch, sdk in zip(boxes["boxes"], sdks):
        patch.set_facecolor(COLORS[sdk])
        patch.set_alpha(0.35)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", which="both", alpha=0.3)


def bar_panel(ax, values, ylabel, title):
    sdks = [s for s in SDK_ORDER if values.get(s) is not None]
    heights = [values[s] for s in sdks]
    bars = ax.bar([SDK_LABEL[s] for s in sdks], heights, color=[COLORS[s] for s in sdks], alpha=0.8)
    for bar, h in zip(bars, heights):
        ax.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


def make_time_charts(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, level in zip(axes, LEVELS):
        curve_panel(ax, series(rows, "wall_s", level), "compile time, s (median, IQR band)",
                    f"level = {level}", logy=True)
    fig.suptitle("Compile time vs circuit width, per SDK")
    save(fig, out_dir, "compile_time_vs_qubits.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, level in zip(axes, LEVELS):
        box_panel(ax, flat(rows, "wall_s", level), "compile time, s", f"level = {level}", logy=True)
    fig.suptitle("Compile time distribution over all circuits and targets")
    save(fig, out_dir, "compile_time_distribution.png")


def make_memory_charts(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, level in zip(axes, LEVELS):
        curve_panel(ax, series(rows, "rss_peak_mb", level), "peak resident memory, MB (median, IQR band)",
                    f"level = {level}", logy=False)
    fig.suptitle("Peak resident memory of the SDK worker during a compile, vs circuit width")
    save(fig, out_dir, "memory_peak_vs_qubits.png")

    idle = {sdk: statistics.median(vals) for sdk, vals in flat(rows, "rss_idle_mb", "default").items() if vals}
    growth = growths(rows, "default")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bar_panel(axes[0], idle, "resident memory, MB", "footprint after warm-up (imports)")
    box_panel(axes[1], growth, "memory growth during compile, MB", "growth per job (level = default)", logy=False)
    fig.suptitle("Memory: SDK footprint and growth per compile")
    save(fig, out_dir, "memory_footprint.png")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench/results/neutral_rss.jsonl")
    ap.add_argument("--out", default="bench/results/charts")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.data)
    print(f"{len(rows)} records from {a.data}")
    make_time_charts(rows, a.out)
    if any("rss_peak_mb" in r for r in rows):
        make_memory_charts(rows, a.out)
    else:
        print("no rss_* fields in this file: memory charts skipped")
    table = summary(rows)
    path = os.path.join(a.out, "summary.md")
    with open(path, "w") as fh:
        fh.write(table)
    print("wrote", path)
    print(table)


if __name__ == "__main__":
    main()
