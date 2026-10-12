"""Tables and plots for result JSONs of bench/run_benchmarks.py.

Used by the example notebooks and by compare_results.py; stdlib only, plus
matplotlib for the plots (imported lazily).

    import bench_report as br
    docs = br.load_results(["bench/results/cloud/qumo_bench_colab_t4_....json"])
    print(br.environment_markdown(docs))
    print(br.timing_markdown(docs))
    fig = br.plot_time_vs_qubits(docs, out="time_vs_qubits.png")

Encoding of the plots (identical in every run, so pictures from different
machines compare at a glance): colour = engine (QUMO, Aer, CUDA-Q, Lightning),
line style = device (solid GPU, dashed CPU), marker = precision (circle fp64,
square fp32). Seconds on a log axis; the timing table carries the exact values.
"""

import json
import os

# Engine colours: categorical slots 1-4 of the validated reference palette
# (adjacent CVD dE >= 9.1). Aqua/yellow are below 3:1 contrast on white, so the
# notebooks always print the timing table next to the plot.
ENGINE_COLORS = {"qumo": "#2a78d6", "aer": "#eb6834", "cudaq": "#1baf7a", "lightning": "#eda100"}
ENGINE_NAMES = {"qumo": "QUMO", "aer": "Aer", "cudaq": "CUDA-Q", "lightning": "Lightning"}
DEVICE_STYLE = {"gpu": "-", "cpu": "--"}
PRECISION_MARKER = {"fp64": "o", "fp32": "s"}
SV_SUITES = ("sv_cpu", "sv_gpu", "baseline_aer", "baseline_cudaq", "baseline_lightning")


def load_results(paths):
    """Load result JSONs (paths or directories); each doc gets a '_path' key."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".json"))
        else:
            files.append(p)
    docs = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if not str(d.get("schema", "")).startswith("qumo-bench/"):
            continue
        d["_path"] = f
        docs.append(d)
    return _unique_labels(docs)


def gpu_name(doc):
    gpus = doc.get("environment", {}).get("gpus") or []
    return gpus[0].get("name") or "GPU" if gpus else "CPU only"


def run_label(doc):
    """Short label '<label> · <GPU> · <config> · <commit7>' (made unique by load_results)."""
    if doc.get("_label"):
        return doc["_label"]
    env = doc.get("environment", {})
    commit = (env.get("git", {}).get("commit") or "")[:7]
    parts = [doc.get("label") or None, gpu_name(doc), doc.get("config", {}).get("name"), commit or None]
    return " · ".join(str(p).replace("|", "/") for p in parts if p)


def _unique_labels(docs):
    seen = {}
    for d in docs:
        d.pop("_label", None)
        base = run_label(d)
        seen.setdefault(base, []).append(d)
    for base, group in seen.items():
        for i, d in enumerate(group):
            stamp = d.get("environment", {}).get("timestamp_utc", "")[5:19].replace("T", " ")
            d["_label"] = base if len(group) == 1 else f"{base} · {stamp or i + 1}"
    return docs


def statevector_rows(doc, family="layered"):
    rows = []
    for name in SV_SUITES:
        for r in doc.get("suites", {}).get(name, {}).get("rows", []):
            if r.get("kind") == "statevector" and r.get("family", "layered") == family and r.get("seconds"):
                rows.append(r)
    return rows


def series_table(doc, family="layered", metric="seconds"):
    """-> {series: {qubits: value}} for the state-vector rows of one run."""
    out = {}
    for r in statevector_rows(doc, family):
        out.setdefault(r["series"], {})[r["qubits"]] = r.get(metric)
    return out


def _series_key(name):
    order = ["QUMO GPU", "QUMO CPU", "CUDA-Q", "Aer GPU", "Aer CPU", "Lightning"]
    for i, prefix in enumerate(order):
        if name.startswith(prefix):
            return (i, name)
    return (len(order), name)


def _fmt(v, metric="seconds"):
    if v is None:
        return "-"
    if metric == "seconds":
        return f"{v:.3g}"
    return f"{v:,.0f}" if v >= 100 else f"{v:.3g}"


def timing_markdown(docs, family="layered", metric="seconds"):
    """One row per (run, series), one column per qubit count (median seconds)."""
    tabs = [(run_label(d), series_table(d, family, metric)) for d in docs]
    qs = sorted({q for _, t in tabs for s in t.values() for q in s})
    if not qs:
        return "_no state-vector timings in these results_\n"
    multi = len(docs) > 1
    head = (["run"] if multi else []) + ["engine / device / precision"] + [f"{q} q" for q in qs]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, t in tabs:
        for s in sorted(t, key=_series_key):
            cells = ([label] if multi else []) + [s] + [_fmt(t[s].get(q), metric) for q in qs]
            lines.append("| " + " | ".join(cells) + " |")
    unit = "median seconds" if metric == "seconds" else metric
    return f"Circuit family `{family}`, {unit}:\n\n" + "\n".join(lines) + "\n"


def speedup_markdown(docs, reference="Aer CPU fp64", family="layered"):
    """Per run: reference_time / series_time at each width (> 1 = faster than the reference)."""
    out = []
    for d in docs:
        t = series_table(d, family)
        ref = t.get(reference)
        if not ref:
            continue
        qs = sorted(ref)
        lines = [f"**{run_label(d)}** - speed-up vs {reference} (x):", "",
                 "| series | " + " | ".join(f"{q} q" for q in qs) + " |", "|---|" + "---|" * len(qs)]
        for s in sorted(t, key=_series_key):
            if s == reference:
                continue
            lines.append(f"| {s} | " + " | ".join(
                f"{ref[q] / t[s][q]:.2f}" if q in t[s] and t[s][q] else "-" for q in qs) + " |")
        out.append("\n".join(lines))
    return "\n\n".join(out) + "\n" if out else f"_no '{reference}' series to compare against_\n"


def status_markdown(docs):
    names = []
    for d in docs:
        for k in d.get("suites", {}):
            if k not in names:
                names.append(k)
    head = ["suite"] + [run_label(d) for d in docs]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n in names:
        cells = []
        for d in docs:
            s = d.get("suites", {}).get(n)
            if not s:
                cells.append("-")
                continue
            extra = f" ({len(s.get('rows', []))} rows)" if s.get("rows") else ""
            summ = s.get("summary")
            if isinstance(summ, dict) and "pass" in summ:
                extra = f" ({summ['pass']} pass / {summ['fail']} fail)"
            elif isinstance(summ, dict):
                extra = " (" + ", ".join(f"{k}: {v['pass']}/{v['pass'] + v['fail']}" for k, v in summ.items()) + ")"
            cells.append(s.get("status", "?") + extra)
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _gib(b):
    return f"{b / 2 ** 30:.1f} GiB" if b else "?"


def environment_markdown(docs):
    """Hardware / software of each run (what a reader needs to trust a comparison)."""
    head = ["", *[run_label(d) for d in docs]]
    rows = {k: [] for k in ("GPU", "GPU memory", "GPU arch / driver", "CPU", "host RAM", "OS",
                            "Mojo / MAX", "QUMO commit", "date (UTC)", "load avg start / end", "config")}
    for d in docs:
        env = d.get("environment", {})
        g = (env.get("gpus") or [{}])[0]
        tc = env.get("toolchain", {})
        rows["GPU"].append(g.get("name") or "none")
        rows["GPU memory"].append(_gib(g.get("memory_bytes")))
        drv = g.get("driver") or g.get("rocm") or g.get("metal_family") or ""
        rows["GPU arch / driver"].append(" / ".join(x for x in (g.get("arch"), drv) if x) or "-")
        cpu = env.get("cpu", {})
        rows["CPU"].append(f"{cpu.get('model') or cpu.get('machine')} x{cpu.get('usable_cores') or cpu.get('logical_cores')}")
        rows["host RAM"].append(_gib(env.get("ram", {}).get("total_bytes")))
        os_ = env.get("os", {})
        rows["OS"].append(os_.get("distro") or os_.get("platform") or "?")
        rows["Mojo / MAX"].append(f"{tc.get('mojo', '?')} / {tc.get('max-core', '?')}")
        git = env.get("git", {})
        rows["QUMO commit"].append(((git.get("commit") or "?")[:10]) + (" (dirty)" if git.get("dirty_files") else ""))
        rows["date (UTC)"].append(env.get("timestamp_utc", "?"))
        rows["load avg start / end"].append(f"{env.get('loadavg_start')} / {d.get('loadavg_end')}")
        rows["config"].append(d.get("config", {}).get("name", "?") + (" (quick)" if d.get("config", {}).get("quick") else ""))
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k, v in rows.items():
        lines.append(f"| {k} | " + " | ".join(str(x) for x in v) + " |")
    return "\n".join(lines) + "\n"


def problems_markdown(docs):
    """Every non-ok suite with its reason (QM-E96x codes)."""
    out = []
    for d in docs:
        for n, s in d.get("suites", {}).items():
            if s.get("status") != "ok" and s.get("reason") and not s["reason"].startswith("not applicable"):
                out.append(f"- **{run_label(d)} / {n}** ({s['status']}): {s['reason']}")
            for na in s.get("not_applicable", []) or []:
                out.append(f"- {run_label(d)} / {n}: n/a - {na.get('reason')}")
    return "\n".join(out) + "\n" if out else "_all suites ok_\n"


def _style(row):
    eng = row.get("engine", "")
    return dict(color=ENGINE_COLORS.get(eng, "#52514e"), linestyle=DEVICE_STYLE.get(row.get("device"), ":"),
                marker=PRECISION_MARKER.get(row.get("precision"), "^"), linewidth=2, markersize=6)


def _axes_format(ax, qubits):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if qubits:
        ax.set_xticks(qubits)
    ax.set_xlabel("qubits")


def _draw(ax, doc, family, title):
    by = {}
    for r in statevector_rows(doc, family):
        by.setdefault(r["series"], []).append(r)
    for name in sorted(by, key=_series_key):
        pts = sorted(by[name], key=lambda r: r["qubits"])
        ax.plot([r["qubits"] for r in pts], [r["seconds"] for r in pts], label=name, **_style(pts[0]))
    _axes_format(ax, sorted({r["qubits"] for rs in by.values() for r in rs}))
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8)
    ax.grid(True, which="minor", color="#f2f1ec", linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors="#52514e", labelsize=9)
    return len(by)


def plot_time_vs_qubits(docs, family="layered", out=None, title=None):
    """Small multiples: one panel per run, shared log y axis. Returns the figure."""
    import matplotlib
    if out and not os.environ.get("DISPLAY") and "inline" not in matplotlib.get_backend().lower():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    docs = [d for d in docs if statevector_rows(d, family)]
    if not docs:
        return None
    n = len(docs)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n + 2.6, 4.4), sharey=True, squeeze=False)
    handles = {}
    for ax, d in zip(axes[0], docs):
        _draw(ax, d, family, run_label(d))
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    axes[0][0].set_ylabel("median seconds (log)")
    labels = sorted(handles, key=_series_key)
    fig.legend([handles[x] for x in labels], labels, loc="center left", bbox_to_anchor=(0.995, 0.5),
               frameon=False, fontsize=9, title="solid GPU / dashed CPU\ncircle fp64 / square fp32", title_fontsize=8)
    fig.suptitle(title or f"QUMO state-vector benchmark: {family} circuit, time vs qubits", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130, bbox_inches="tight")
    return fig


def plot_series_across_runs(docs, prefix="QUMO GPU", family="layered", out=None):
    """One line per run for every series starting with `prefix` (cross-hardware view)."""
    import matplotlib
    if out and not os.environ.get("DISPLAY") and "inline" not in matplotlib.get_backend().lower():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    k = 0
    for d in docs:
        t = series_table(d, family)
        for s in sorted(t, key=_series_key):
            if not s.startswith(prefix) or k >= len(palette):
                continue
            qs = sorted(t[s])
            prec = "fp32" if "fp32" in s else "fp64"
            ax.plot(qs, [t[s][q] for q in qs], color=palette[k], linewidth=2, markersize=6,
                    marker=PRECISION_MARKER[prec], label=f"{gpu_name(d)}: {s}")
            k += 1
    if k == 0:
        plt.close(fig)
        return None
    _axes_format(ax, sorted({int(x) for line in ax.get_lines() for x in line.get_xdata()}))
    ax.set_ylabel("median seconds (log)")
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(f"{prefix} across machines ({family} circuit)", fontsize=11, loc="left")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130, bbox_inches="tight")
    return fig


def summary_markdown(docs, reference="Aer CPU fp64"):
    """Everything a results page needs, in one markdown string."""
    parts = ["## Machines\n", environment_markdown(docs), "\n## Suites\n", status_markdown(docs),
             "\n## State-vector timings\n", timing_markdown(docs), "\n## Speed-ups\n", speedup_markdown(docs, reference),
             "\n## Problems and skipped parts\n", problems_markdown(docs)]
    return "\n".join(parts)
