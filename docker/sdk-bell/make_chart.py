"""Chart of image size, time, CPU and peak memory per SDK container (data from run_measure.sh).

    bench/.venv/bin/python docker/sdk-bell/make_chart.py --data docker_metrics.txt --out docker/sdk-bell/charts --name docker_resources \
        --note "preliminary: ..."
"""
import argparse
import os
import statistics
import sys
from collections import defaultdict

import altair as alt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bench", "charts"))
import make_charts as mc  # noqa: E402  (registers and enables the shared theme)

ORDER = ["QUMO", "Qiskit", "TKET", "Cirq"]


def load(path):
    runs = defaultdict(list)
    for line in open(path):
        p = line.split()
        if len(p) < 4:
            continue
        kv = dict(x.split("=") for x in p[3:])
        runs[p[0]].append({"size_mb": int(p[1]) / 1e6, "wall_ms": float(kv["wall_ms"]), "cpu_ms": float(kv["cpu_ms"]),
                           "rss_mib": float(kv["rss_mib"])})
    rows = []
    for key, rs in runs.items():
        name = mc.SDK_LABEL[key]
        rows.append({"sdk": name, "runs": len(rs), "size_mb": rs[0]["size_mb"],
                     "wall_ms": statistics.median(r["wall_ms"] for r in rs),
                     "cpu_ms": statistics.median(r["cpu_ms"] for r in rs),
                     "rss_mib": statistics.median(r["rss_mib"] for r in rs)})
    return sorted(rows, key=lambda r: ORDER.index(r["sdk"]))


def panel(rows, field, title, fmt, log=False, width=250):
    scale = alt.Scale(type="log", domain=[1, max(r[field] for r in rows) * 2.2]) if log else alt.Scale(zero=True, nice=True)
    base = alt.Chart(alt.Data(values=rows))
    bars = base.mark_bar(cornerRadiusEnd=3, height=26).encode(
        y=alt.Y("sdk:N", sort=ORDER, title=None, axis=alt.Axis(labelFontSize=12.5, ticks=False, domain=False)),
        x=alt.X(f"{field}:Q", scale=scale, title=None, axis=alt.Axis(format=fmt if not log else "~g", grid=True, **({"values": [1, 10, 100, 1000, 10000]} if log else {}))),
        x2=alt.datum(1) if log else alt.X2(datum=0),
        color=alt.Color("sdk:N", scale=mc.SDK_SCALE, legend=None))
    labels = base.mark_text(align="left", dx=5, fontSize=12, color=mc.INK, fontWeight=500).encode(
        y=alt.Y("sdk:N", sort=ORDER), x=alt.X(f"{field}:Q", scale=scale), text=alt.Text(f"{field}:Q", format=fmt))
    return alt.layer(bars, labels).properties(width=width, height=130, title=alt.TitleParams(title, fontSize=14, anchor="start"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="docker_resources")
    ap.add_argument("--note", default="")
    ap.add_argument("--grid", action="store_true", help="2 x 2 layout instead of one row")
    a = ap.parse_args()
    rows = load(a.data)
    n = min(r["runs"] for r in rows)
    w = 330 if a.grid else 250
    panels = [panel(rows, "size_mb", "Image size on disk (MB)", ",.0f", width=w),
              panel(rows, "wall_ms", "Run time (ms, log)", ",.0f", log=True, width=w),
              panel(rows, "cpu_ms", "CPU time (ms, log)", ",.0f", log=True, width=w),
              panel(rows, "rss_mib", "Peak memory (MiB)", ",.0f", width=w)]
    body = (alt.vconcat(alt.hconcat(*panels[:2], spacing=40), alt.hconcat(*panels[2:], spacing=40), spacing=28)
            if a.grid else alt.hconcat(*panels, spacing=26))
    chart = body.properties(title=mc._title("Container cost of one SDK run, Bell state", f"Median of {n} runs per image. {a.note}".strip()))
    os.makedirs(a.out, exist_ok=True)
    mc.save(chart, a.out, a.name)


if __name__ == "__main__":
    main()
