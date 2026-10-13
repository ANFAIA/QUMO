"""T4 (Colab) vs Apple M5 comparison figure from QUMO benchmark result JSONs.

usage: plot_t4_vs_m5.py <t4.json> <m5_commit.json> <m5_current.json> <out.png>

CAVEAT (read before using the figure): the QUMO GPU rows of the two machines
come from DIFFERENT GPU engines. The Colab T4 run of 2026-10-12 used commit
1750a81's engine, which launches one kernel per plan op (378 launches for the
26-qubit layered circuit); the M5 "current tree" run used the tile engine (17
launches at 26 qubits). Both result files say "1750a81 (dirty)", because the
older harness recorded only the commit. So panel A and the QUMO GPU bars in
panel D compare engines, not hardware: no T4-vs-M5 conclusion about QUMO GPU
until the T4 is re-run with the current tree (the new harness records a hash
of the GPU engine sources, `environment.sources.gpu_engine_sha256`). The
script prints the engine identity it can establish for each input.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

t4p, m5ap, m5bp, out = sys.argv[1:5]
T4, M5A, M5B = (json.load(open(p)) for p in (t4p, m5ap, m5bp))


def engine(doc):
    """GPU-engine identity of a run: the source hash (new harness) or 'unknown' for a dirty tree."""
    env = doc.get("environment", {})
    h = (env.get("sources") or {}).get("gpu_engine_sha256")
    git = env.get("git", {})
    if h:
        return f"sha256:{h}"
    return f"commit {(git.get('commit') or '?')[:7]}" + (f" + {git['dirty_files']} uncommitted files (engine unknown)"
                                                         if git.get("dirty_files") else "")


for name, doc in (("T4", T4), ("M5 commit run", M5A), ("M5 current-tree run", M5B)):
    print(f"{name}: QUMO GPU engine = {engine(doc)}")
SAME_ENGINE = engine(T4) == engine(M5B) and "unknown" not in engine(T4)
T4_QUMO = ("QUMO T4 (old engine: 1 launch per plan op, 378 at 26 q)" if not SAME_ENGINE
           else "QUMO T4 (same engine as M5)")

COL = {"QUMO": "#2a78d6", "Aer": "#eb6834", "CUDA-Q": "#1baf7a"}
MACH = {"T4": dict(linestyle="-", marker="o"), "M5": dict(linestyle="--", marker="s")}


def rows(doc, suite):
    return [r for r in doc["suites"].get(suite, {}).get("rows", []) if r.get("family", "layered") == "layered"]


def series(doc, suites, engine, device, precision):
    pts = {}
    for s in suites:
        for r in rows(doc, s):
            if r.get("engine", "").lower() == engine and r.get("device") == device and r.get("precision") == precision:
                if r.get("seconds") is not None:
                    pts[r["qubits"]] = r["seconds"]
    q = sorted(pts)
    return q, [pts[k] for k in q]


SV = ("sv_cpu", "sv_gpu", "baseline_aer", "baseline_cudaq")


def fmt(ax, title):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9, colors="#52514e")


def line(ax, doc, engine, device, prec, label, color, mach):
    q, t = series(doc, SV, engine, device, prec)
    if q:
        ax.plot(q, t, label=label, color=color, linewidth=2, markersize=5, **MACH[mach])


fig, axes = plt.subplots(2, 3, figsize=(17, 9.6))
(a, b, c), (d, e, f) = axes

# A: GPU fp32
line(a, T4, "qumo", "gpu", "fp32", T4_QUMO, COL["QUMO"], "T4")
line(a, M5B, "qumo", "gpu", "fp32", "QUMO M5 Metal (tile engine, 17 launches at 26 q)", COL["QUMO"], "M5")
line(a, T4, "cudaq", "gpu", "fp32", "CUDA-Q T4", COL["CUDA-Q"], "T4")
line(a, T4, "aer", "gpu", "fp32", "Aer GPU T4", COL["Aer"], "T4")
fmt(a, "A. GPU state vector, fp32 (layered circuit)" + ("" if SAME_ENGINE else
                                                         "\nQUMO T4 vs M5 = different engines, not a hardware result"))
a.set_ylabel("median seconds (log)")
a.legend(fontsize=8, frameon=False)

# B: GPU fp64 (T4 only; Metal has no fp64)
line(b, T4, "qumo", "gpu", "fp64", T4_QUMO, COL["QUMO"], "T4")
line(b, T4, "cudaq", "gpu", "fp64", "CUDA-Q T4", COL["CUDA-Q"], "T4")
line(b, T4, "aer", "gpu", "fp64", "Aer GPU T4", COL["Aer"], "T4")
fmt(b, "B. GPU state vector, fp64 (T4 only: Metal has no fp64)\nT4 fp64 runs at 1/32 of fp32: limited by fp64 math")
b.legend(fontsize=8, frameon=False)

# C: CPU fp64
line(c, T4, "qumo", "cpu", "fp64", "QUMO CPU, Colab host (2 vCPU Xeon)", COL["QUMO"], "T4")
line(c, T4, "aer", "cpu", "fp64", "Aer CPU, Colab host", COL["Aer"], "T4")
line(c, M5A, "qumo", "cpu", "fp64", "QUMO CPU, M5 (10 cores)", COL["QUMO"], "M5")
line(c, M5A, "aer", "cpu", "fp64", "Aer CPU, M5", COL["Aer"], "M5")
fmt(c, "C. CPU state vector, fp64 (same commit 1750a81)")
c.legend(fontsize=8, frameon=False)
for ax in (a, b, c):
    ax.set_xlabel("qubits")


# D: 26-qubit bars
def at(doc, engine, device, prec, n=26):
    q, t = series(doc, SV, engine, device, prec)
    return dict(zip(q, t)).get(n)


bars = [
    ("Aer CPU fp64\nColab host", at(T4, "aer", "cpu", "fp64"), COL["Aer"], ""),
    ("QUMO CPU fp64\nColab host", at(T4, "qumo", "cpu", "fp64"), COL["QUMO"], ""),
    ("Aer CPU fp64\nM5", at(M5A, "aer", "cpu", "fp64"), COL["Aer"], "//"),
    ("QUMO CPU fp64\nM5", at(M5A, "qumo", "cpu", "fp64"), COL["QUMO"], "//"),
    ("Aer GPU fp32\nT4", at(T4, "aer", "gpu", "fp32"), COL["Aer"], ""),
    ("QUMO GPU fp32\nT4, old engine" if not SAME_ENGINE else "QUMO GPU fp32\nT4", at(T4, "qumo", "gpu", "fp32"),
     COL["QUMO"], ""),
    ("QUMO GPU fp32\nM5, tile engine" if not SAME_ENGINE else "QUMO GPU fp32\nM5", at(M5B, "qumo", "gpu", "fp32"),
     COL["QUMO"], "//"),
    ("CUDA-Q GPU fp32\nT4", at(T4, "cudaq", "gpu", "fp32"), COL["CUDA-Q"], ""),
]
bars = [x for x in bars if x[1]]
ys = range(len(bars))
d.barh(list(ys), [x[1] for x in bars], color=[x[2] for x in bars], hatch=None, edgecolor="white")
for i, x in enumerate(bars):
    if x[3]:
        d.barh(i, x[1], color="none", edgecolor="white", hatch=x[3], linewidth=0)
    d.text(x[1] * 1.08, i, f"{x[1]:.3g} s", va="center", fontsize=8.5, color="#33322e")
d.set_yticks(list(ys))
d.set_yticklabels([x[0] for x in bars], fontsize=8.5)
d.invert_yaxis()
d.set_xscale("log")
d.set_xlim(0.08, 400)
d.set_title("D. 26 qubits, layered circuit (hatched = M5)", fontsize=10.5, loc="left")
d.set_xlabel("median seconds (log)")
for side in ("top", "right"):
    d.spines[side].set_visible(False)
d.tick_params(labelsize=8.5, colors="#52514e")


# E: QEC Pauli frames, 1M shots
def frames(doc):
    acc = {}
    for r in doc["suites"].get("sv_gpu", {}).get("rows", []):
        if r.get("kind") == "qec_frames":
            acc.setdefault(r["d"], []).append((r["cpu_s"], r["gpu_s"]))
    med = lambda v: sorted(v)[len(v) // 2]
    return {k: (med([x[0] for x in v]), med([x[1] for x in v])) for k, v in acc.items()}


ft4, fm5 = frames(T4), frames(M5B)
ds = sorted(set(ft4) | set(fm5))
w = 0.2
for j, (lab, src, idx, col, hatch) in enumerate([
        ("CPU, Colab host", ft4, 0, "#8a8984", ""), ("GPU, T4", ft4, 1, COL["QUMO"], ""),
        ("CPU, M5", fm5, 0, "#8a8984", "//"), ("GPU, M5", fm5, 1, COL["QUMO"], "//")]):
    xs = [k + (j - 1.5) * w for k in range(len(ds))]
    vals = [1e6 / src[dd][idx] if dd in src and src[dd][idx] else 0 for dd in ds]
    e.bar(xs, vals, width=w, color=col, hatch=hatch, edgecolor="white", label=lab)
e.set_xticks(range(len(ds)))
e.set_xticklabels([f"surface d={dd}" for dd in ds], fontsize=9)
e.set_yscale("log")
e.set_ylabel("shots / s (log, higher is better)")
e.set_title("E. QEC Pauli-frame sampling, QUMO (1M shots)", fontsize=10.5, loc="left")
e.legend(fontsize=8, frameon=False)
e.grid(True, axis="y", color="#e6e5e0")
for side in ("top", "right"):
    e.spines[side].set_visible(False)


# F: compiler compile time
def neutral(doc):
    return {r.get("engine"): r for r in doc["suites"].get("neutral", {}).get("rows", []) if r.get("kind") == "compiler"}


nt4, nm5 = neutral(T4), neutral(M5A)
sdks = [s for s in ("qumo", "qiskit", "tket") if s in nt4 or s in nm5]
key = "median_compile_s"
names = {"qumo": "QUMO", "qiskit": "Qiskit", "tket": "TKET"}
cols = {"qumo": COL["QUMO"], "qiskit": "#7b5ea7", "tket": "#c0392b"}
for j, (src, hatch, lab) in enumerate([(nt4, "", "Colab host"), (nm5, "//", "M5")]):
    xs = [k + (j - 0.5) * 0.36 for k in range(len(sdks))]
    vals = [1000 * ((src.get(s) or {}).get(key) or 0) for s in sdks]
    f.bar(xs, vals, width=0.36, color=[cols[s] for s in sdks], hatch=hatch, edgecolor="white")
    for x, v in zip(xs, vals):
        if v:
            f.text(x, v * 1.1, f"{v:.3g}", ha="center", fontsize=8)
f.set_xticks(range(len(sdks)))
f.set_xticklabels([names[s] for s in sdks])
f.set_yscale("log")
f.set_ylabel("median compile time, ms (log)")
f.set_title("F. Compiler, 270 jobs: identical 2q counts on both machines\n(solid = Colab host, hatched = M5)",
            fontsize=10.5, loc="left")
for side in ("top", "right"):
    f.spines[side].set_visible(False)
f.grid(True, axis="y", color="#e6e5e0")

fig.suptitle("QUMO on Google Colab Tesla T4 vs Apple M5 - same layered circuit, medians of 3 runs. "
             "Line style: solid = T4 / Colab host, dashed = M5."
             + ("" if SAME_ENGINE else "\nQUMO GPU: the T4 ran the OLD engine (one launch per plan op), the M5 the "
                                       "tile engine - no T4-vs-M5 hardware conclusion until the T4 re-run."),
             x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out, "neutral key", key)
