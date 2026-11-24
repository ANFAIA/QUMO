"""The two charts of the README results section: compile throughput and memory per SDK (neutral benchmark).

    bench/.venv/bin/python bench/charts/make_readme_charts.py \
        --data bench/results/neutral_mean4_2026-11-23.jsonl.gz --out bench/results/readme

The container chart comes from docker/sdk-bell/make_chart.py. Data is the per-job mean of several runs; only jobs an SDK
compiled enter the figures.
"""
import argparse
import gzip
import json
import os
import statistics

import altair as alt

import make_charts as mc  # registers and enables the shared theme

ORDER = mc.LABELS


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def per_sdk(records, level):
    rows = []
    for key in mc.SDK_ORDER:
        ok = [r for r in records if r["sdk"] == key and r["level"] == level and r["status"] == "ok"]
        wall = sum(r["wall_s"] for r in ok)
        rows.append({"sdk": mc.SDK_LABEL[key], "jobs_per_s": len(ok) / wall,
                     "idle_mb": statistics.median(r["rss_idle_mb"] for r in ok),
                     "peak_mb": statistics.median(r["rss_peak_mb"] for r in ok), "jobs": len(ok)})
    return rows


def throughput(rows, level):
    base = alt.Chart(alt.Data(values=rows))
    scale = alt.Scale(type="log", domain=[0.1, 2000])
    bars = base.mark_bar(cornerRadiusEnd=3, height=28).encode(
        y=alt.Y("sdk:N", sort=ORDER, title=None, axis=alt.Axis(ticks=False, domain=False, labelFontSize=13)),
        x=alt.X("jobs_per_s:Q", scale=scale, title="jobs compiled per second (log scale)", axis=alt.Axis(values=[0.1, 1, 10, 100, 1000], format="~g")),
        x2=alt.datum(0.1), color=alt.Color("sdk:N", scale=mc.SDK_SCALE, legend=None))
    labels = base.mark_text(align="left", dx=6, fontSize=13, color=mc.INK, fontWeight=500).encode(
        y=alt.Y("sdk:N", sort=ORDER), x=alt.X("jobs_per_s:Q", scale=scale), text=alt.Text("jobs_per_s:Q", format=".3~r"))
    return alt.layer(bars, labels).properties(
        width=560, height=190,
        title=mc._title("Compile throughput", f"One job is one circuit on one target, {level} optimization level. Higher is faster."))


def memory(rows):
    long = []
    for r in rows:
        long.append({"sdk": r["sdk"], "part": "idle after warm-up", "mb": r["idle_mb"]})
        long.append({"sdk": r["sdk"], "part": "median peak during a job", "mb": r["peak_mb"]})
    base = alt.Chart(alt.Data(values=long))
    enc = dict(x=alt.X("sdk:N", sort=ORDER, title=None, axis=alt.Axis(labelAngle=0, ticks=False, domain=False, labelFontSize=13)),
               xOffset=alt.XOffset("part:N", sort=["idle after warm-up", "median peak during a job"]),
               y=alt.Y("mb:Q", title="resident memory, MB", scale=alt.Scale(domain=[0, 640])))
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color("sdk:N", scale=mc.SDK_SCALE, legend=None),
        opacity=alt.Opacity("part:N", sort=["idle after warm-up", "median peak during a job"],
                            scale=alt.Scale(domain=["idle after warm-up", "median peak during a job"], range=[0.4, 1.0]),
                            legend=alt.Legend(title=None, orient="top", symbolType="square", symbolFillColor=mc.MUTED)),
        **enc)
    labels = base.mark_text(dy=-7, fontSize=12, color=mc.INK, fontWeight=500).encode(text=alt.Text("mb:Q", format=".0f"), **enc)
    return alt.layer(bars, labels).properties(
        width=560, height=260,
        title=mc._title("Memory per worker", "Light bars: idle after warm-up. Solid bars: median peak. Every worker includes a shared 70 MB python and qiskit import."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench/results/neutral_mean4_2026-11-23.jsonl.gz")
    ap.add_argument("--out", default="bench/results/readme")
    ap.add_argument("--level", default="default")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = per_sdk(load(a.data), a.level)
    mc.save(throughput(rows, a.level), a.out, "throughput")
    mc.save(memory(rows), a.out, "memory")


if __name__ == "__main__":
    main()
