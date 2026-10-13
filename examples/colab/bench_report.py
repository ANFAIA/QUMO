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

Comparability rules the reports apply (docs/BENCHMARKING_ON_CLOUD_GPUS.md):
* QUMO GPU rows of runs whose GPU engine differs (or cannot be identified: no
  `sources.gpu_engine_sha256` and a dirty tree) are never put in one table or
  one plot line; the report says so.
* Widths <= 20 qubits are marked: fixed per-call overhead (Python, launches)
  dominates there, so they compare APIs, not simulators.
* fp64 on GPUs with slow fp64 units (T4, L4, RTX, consumer AMD) is limited by
  fp64 multiply-adds for every engine and is reported as such: it does not
  carry over to A100 / H100 / MI300-class GPUs.
"""

import importlib.util
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
OVERHEAD_MAX_QUBITS = 20  # at and below: fixed per-call overhead dominates (marked with a dagger)
# GPUs whose fp64 units run at 1/32 or 1/64 of fp32 (NVIDIA compute capability, AMD consumer RDNA)
WEAK_FP64_ARCH = ("sm_75", "sm_86", "sm_87", "sm_89", "sm_120", "sm_121", "gfx10", "gfx11", "gfx12")


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


def engine_identity(doc):
    """-> (id, reliable) of the QUMO GPU engine a run used.

    Runs written by the fairness-fixed harness carry `environment.sources.gpu_engine_sha256`
    (a hash of qumo/gpu, the plan and bench_gpu). Older runs only have the commit: that is
    reliable for a clean tree, but a dirty tree ("1750a81 (dirty)") can hold any engine."""
    env = doc.get("environment", {})
    h = (env.get("sources") or {}).get("gpu_engine_sha256")
    if h:
        return f"sha256:{h}", True
    git = env.get("git", {})
    commit = (git.get("commit") or "?")[:7]
    if git.get("dirty_files"):
        return f"unknown (commit {commit} + {git['dirty_files']} uncommitted files)", False
    return f"commit {commit}", commit != "?"


def has_fair_harness(doc):
    """True for runs of the harness with per-row timed regions, per-width warm-ups and source hashes."""
    return bool((doc.get("environment", {}).get("sources") or {}).get("harness_sha256"))


def weak_fp64(doc):
    g = (doc.get("environment", {}).get("gpus") or [{}])[0]
    arch = str(g.get("arch") or "")
    return g.get("vendor") in ("nvidia", "amd") and arch.startswith(WEAK_FP64_ARCH)


def _qhead(q):
    return f"{q} q" + ("†" if q <= OVERHEAD_MAX_QUBITS else "")


SMALL_WIDTH_NOTE = (f"† {OVERHEAD_MAX_QUBITS} qubits and fewer: fixed per-call overhead (Python API, launches, "
                    "planning) dominates, so these columns compare APIs, not simulators.")


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
    head = (["run"] if multi else []) + ["engine / device / precision"] + [_qhead(q) for q in qs]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, t in tabs:
        for s in sorted(t, key=_series_key):
            cells = ([label] if multi else []) + [s] + [_fmt(t[s].get(q), metric) for q in qs]
            lines.append("| " + " | ".join(cells) + " |")
    unit = "median seconds" if metric == "seconds" else metric
    foot = "\n" + SMALL_WIDTH_NOTE + "\n" if any(q <= OVERHEAD_MAX_QUBITS for q in qs) else ""
    return (f"Circuit family `{family}`, {unit} (series with a label in parentheses are diagnostic variants; "
            f"each engine's default series is the one without):\n\n" + "\n".join(lines) + "\n" + foot)


DEFAULT_REFERENCE = "Aer CPU {precision}"


def _precision_of(series):
    return "fp32" if "fp32" in series.split() else "fp64" if "fp64" in series.split() else None


def speedup_markdown(docs, reference=DEFAULT_REFERENCE, family="layered"):
    """Per run and precision: reference_time / series_time at each width (> 1 = faster).

    Like for like only: every series is compared with a reference of its OWN
    precision. `reference` may contain `{precision}` (default "Aer CPU
    {precision}": fp64 series vs Aer CPU fp64, fp32 series vs Aer CPU fp32). A
    fixed name (e.g. "Aer CPU fp64") compares only the series of that precision.
    Series without a same-precision reference get a note instead of a ratio:
    an fp32 GPU time divided by an fp64 CPU time is not a speed-up."""
    out = []
    for d in docs:
        t = series_table(d, family)
        by_prec = {}
        for s in t:
            by_prec.setdefault(_precision_of(s), []).append(s)
        for prec in ("fp64", "fp32", None):
            series = by_prec.get(prec)
            if not series or prec is None:
                continue
            ref_name = reference.format(precision=prec) if "{precision}" in reference else reference
            if _precision_of(ref_name) not in (prec, None):
                continue  # a fixed reference of the other precision: not like for like
            ref = t.get(ref_name)
            others = [s for s in sorted(series, key=_series_key) if s != ref_name]
            if not others:
                continue
            if not ref:
                note = (" Add \"fp32\" to `baseline_aer.precisions_cpu` in the preset for an fp32 CPU reference."
                        if ref_name == "Aer CPU fp32" else "")
                out.append(f"**{run_label(d)}** - {prec}: no '{ref_name}' series in this run, so no {prec} speed-ups "
                           f"({', '.join(others)}).{note}")
                continue
            qs = sorted(ref)
            lines = [f"**{run_label(d)}** - {prec} series, speed-up vs {ref_name} (x):", "",
                     "| series | " + " | ".join(_qhead(q) for q in qs) + " |", "|---|" + "---|" * len(qs)]
            for s in others:
                lines.append(f"| {s} | " + " | ".join(
                    f"{ref[q] / t[s][q]:.2f}" if q in t[s] and t[s][q] else "-" for q in qs) + " |")
            if prec == "fp64" and weak_fp64(d):
                g = (d.get("environment", {}).get("gpus") or [{}])[0]
                lines += ["", f"_fp64 on {g.get('name') or 'this GPU'} ({g.get('arch')}): its fp64 units run at 1/32 "
                              "(or less) of the fp32 rate, so every engine's fp64 GPU time here is limited by fp64 "
                              "multiply-adds, not by memory. Report it separately; it does not carry over to "
                              "A100 / H100 / MI300-class GPUs with full-rate fp64._"]
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
            status = s.get("status", "?")
            if status == "skipped" and str(s.get("reason") or "").startswith("not applicable"):
                status = "n/a"  # the device / baseline does not exist on this hardware
            cells.append(status + extra)
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def compiler_markdown(docs):
    """`neutral` suite: one row per run x SDK (same circuits and targets for every SDK)."""
    lines = []
    for d in docs:
        for r in d.get("suites", {}).get("neutral", {}).get("rows", []):
            if r.get("kind") != "compiler":
                continue
            ok = r.get("n_ok", 0)
            bad = sum(r.get(f"n_{k}", 0) for k in ("invalid", "incorrect", "error", "timeout"))
            ms = r.get("median_compile_s")
            esp = r.get("mean_esp")
            lines.append(f"| {run_label(d)} | {r.get('engine')} | {r.get('jobs')} | {ok} | {bad} | "
                         f"{r.get('n_unsupported', 0)} | {_fmt(ms * 1e3, 'n') if ms is not None else '-'} | "
                         f"{_fmt(r.get('median_n2q'), 'n')} | {f'{esp:.4f}' if esp is not None else '-'} |")
    if not lines:
        return "_no compiler-subset results_\n"
    head = ("| run | SDK | jobs | ok | invalid / incorrect / error / timeout | unsupported | median compile ms | "
            "median 2q gates | mean ESP |\n|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(lines) + "\n"


def frames_markdown(docs):
    """QEC Pauli-frame sampling rows of the `sv_gpu` suite (QUMO CPU vs QUMO GPU)."""
    lines = []
    for d in docs:
        for r in d.get("suites", {}).get("sv_gpu", {}).get("rows", []):
            if r.get("kind") != "qec_frames":
                continue
            lines.append(f"| {run_label(d)} | {r.get('code')} | {r.get('d')} | {r.get('shots')} | "
                         f"{_fmt(r.get('cpu_s'))} | {_fmt(r.get('gpu_s'))} | {_fmt(r.get('speedup'))} |")
    if not lines:
        return ""
    return ("| run | code | d | shots | CPU s | GPU s | speed-up |\n|---|---|---|---|---|---|---|\n"
            + "\n".join(lines) + "\n")


def _gib(b):
    return f"{b / 2 ** 30:.1f} GiB" if b else "?"


def environment_markdown(docs):
    """Hardware / software of each run (what a reader needs to trust a comparison)."""
    head = ["", *[run_label(d) for d in docs]]
    rows = {k: [] for k in ("GPU", "GPU memory", "GPU arch / driver", "CPU", "host RAM", "OS",
                            "Mojo / MAX", "QUMO commit", "QUMO GPU engine", "harness", "power / clocks at start",
                            "date (UTC)", "load avg start / end", "config")}
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
        eid, reliable = engine_identity(d)
        rows["QUMO GPU engine"].append(eid + ("" if reliable else " - not identifiable"))
        rows["harness"].append("per-row timed regions, per-width warm-ups" if has_fair_harness(d)
                               else "older (before the timing-fairness fixes)")
        ms = env.get("machine_state") or {}
        pw, gs = ms.get("power") or {}, ms.get("gpu") or {}
        bits = []
        if "lowpowermode" in pw:
            bits.append(f"Low Power Mode {'on' if pw['lowpowermode'] else 'off'}")
        if gs:
            bits.append(f"SM {gs.get('clocks.sm', '?')}/{gs.get('clocks.max.sm', '?')} MHz, {gs.get('temperature.gpu', '?')} C, "
                        f"{gs.get('pstate', '?')}")
        rows["power / clocks at start"].append("; ".join(bits) or "not recorded")
        rows["date (UTC)"].append(env.get("timestamp_utc", "?"))
        rows["load avg start / end"].append(f"{env.get('loadavg_start')} / {d.get('loadavg_end')}")
        rows["config"].append(d.get("config", {}).get("name", "?") + (" (quick)" if d.get("config", {}).get("quick") else ""))
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k, v in rows.items():
        lines.append(f"| {k} | " + " | ".join(str(x) for x in v) + " |")
    return "\n".join(lines) + "\n"


def problems_markdown(docs):
    """Every non-ok suite with its reason (QM-E96x codes), failed diagnostic variants, and
    the runner's cross-engine consistency checks."""
    out = []
    for d in docs:
        for c in d.get("checks", []) or []:
            out.append(f"- **{run_label(d)} / check**: {c}")
        for n, s in d.get("suites", {}).items():
            if s.get("status") != "ok" and s.get("reason") and not s["reason"].startswith("not applicable"):
                out.append(f"- **{run_label(d)} / {n}** ({s['status']}): {s['reason']}")
            for na in s.get("not_applicable", []) or []:
                out.append(f"- {run_label(d)} / {n}: n/a - {na.get('reason')}")
            for v in s.get("variant_problems", []) or []:
                why = v.get("reason") if isinstance(v, dict) else v
                lab = f" `{v.get('variant')}`" if isinstance(v, dict) and v.get("variant") else ""
                out.append(f"- {run_label(d)} / {n}: diagnostic variant{lab} - {str(why)[:300]}")
    return "\n".join(out) + "\n" if out else "_all suites ok_\n"


def _fusion_text(r):
    f = r.get("fusion")
    eng = r.get("engine")
    if eng == "aer":
        if not f and not r.get("aer_fusion"):
            return "not recorded (older harness)"
        rep_ = (f or {}).get("reported") or r.get("aer_fusion") or {}
        conf = (f or {}).get("configured")
        txt = (f"applied={rep_.get('applied')}, max {rep_.get('max_fused_qubits')} qubits, threshold "
               f"{rep_.get('threshold')} qubits" if rep_ else "not reported")
        return txt + (f" (options {conf})" if isinstance(conf, dict) else "")
    if eng == "cudaq" and isinstance(f, dict):
        return (f"{f.get('mode')}: dense <= {f.get('dense_max_qubits')} qubits, diagonal <= "
                f"{f.get('diagonal_max_qubits')} qubits")
    if eng == "qumo" and isinstance(f, dict):
        gpu = "; many plan ops per state sweep (tile runs)" if r.get("device") == "gpu" else ""
        return "1-qubit runs fused into 2x2 (plan)" + gpu
    if isinstance(f, dict):
        return f.get("plan") or "unknown"
    return "not recorded (older harness)"


def fairness_markdown(docs, family="layered"):
    """What each series' timed region contains and what each engine fused, per run, plus the
    comparability warnings (mixed QUMO GPU engines, older harness, slow fp64, small widths)."""
    out = []
    gpu_runs = [d for d in docs if any(r.get("engine") == "qumo" and r.get("device") == "gpu"
                                       for r in statevector_rows(d, family))]
    ids = {}
    for d in gpu_runs:
        ids.setdefault(engine_identity(d), []).append(run_label(d))
    unreliable = [lab for (eid, ok), labs in ids.items() if not ok for lab in labs]
    if len(ids) > 1 or (unreliable and len(gpu_runs) > 1):
        lines = ["**QUMO GPU engines differ, or cannot be identified, across these runs.** Differences between "
                 "their QUMO GPU rows mix engine changes with hardware: draw no hardware conclusion from them. "
                 "Engines:"]
        for (eid, ok), labs in ids.items():
            lines.append(f"- {eid}{'' if ok else ' (not identifiable)'}: {', '.join(labs)}")
        out.append("\n".join(lines))
    old = [run_label(d) for d in docs if not has_fair_harness(d)]
    if old:
        out.append("**Older harness** (" + ", ".join(old) + "): no per-row timed region, no per-repetition times, "
                   "Aer warmed up only at its smallest width, QUMO CPU without warm-up, QUMO GPU widths capped by "
                   "its CPU reference, and no record of GPU clocks or Low Power Mode. Re-run with the current "
                   "harness before comparing these rows with newer runs.")
    for d in docs:
        if weak_fp64(d):
            g = (d.get("environment", {}).get("gpus") or [{}])[0]
            out.append(f"**{run_label(d)}**: fp64 on {g.get('name')} ({g.get('arch')}) runs at 1/32 of fp32 or "
                       "less; its fp64 GPU rows measure fp64 arithmetic throughput, not memory bandwidth, and do "
                       "not carry over to full-rate fp64 GPUs (A100, H100, MI300).")
    regions, rows = {}, []
    for d in docs:
        seen = {}
        for r in statevector_rows(d, family):
            seen.setdefault(r.get("series"), r)
        for name in sorted(seen, key=_series_key):
            r = seen[name]
            key = f"{ENGINE_NAMES.get(r.get('engine'), r.get('engine'))} {str(r.get('device', '')).upper()}"
            regions.setdefault((key, r.get("timed_region") or "not recorded (older harness)"), []).append(run_label(d))
            rows.append(f"| {run_label(d)} | {name} | {_fusion_text(r)} |")
    if regions:
        lines = ["Timed region of each engine (what `seconds` contains):", ""]
        for (key, text), labs in sorted(regions.items()):
            labs = list(dict.fromkeys(labs))
            lines.append(f"- **{key}**: {text}" + (f" _({', '.join(labs)})_" if len(docs) > 1 else ""))
        out.append("\n".join(lines))
    if rows:
        out.append("| run | series | fusion (each engine's own default; a variant label says what it changes) |\n"
                   "|---|---|---|\n" + "\n".join(rows))
    out.append(SMALL_WIDTH_NOTE)
    return "\n\n".join(out) + "\n"


def sweeps_markdown(docs, family="layered"):
    """QUMO GPU: launches (= full state sweeps of the tile engine) per evolve, gates per launch and
    effective bandwidth (2R-1) 2^n B / t, when bench_gpu reports its launch count."""
    lines = []
    for d in docs:
        for r in sorted(statevector_rows(d, family), key=lambda x: (x.get("series", ""), x.get("qubits", 0))):
            if r.get("engine") != "qumo" or r.get("device") != "gpu" or not r.get("launches"):
                continue
            alloc = r.get("seconds_incl_alloc")
            lines.append(f"| {run_label(d)} | {r['series']} | {r['qubits']} | {_fmt(r.get('seconds'))} | "
                         f"{_fmt(alloc) if alloc else '-'} | {r['launches']} | "
                         f"{r.get('gates_per_launch') or '-'} | {_fmt(r.get('effective_gbs'), 'n')} |")
    if not lines:
        return ""
    return ("| run | series | qubits | seconds | seconds incl. allocation | launches (sweeps) | gates / launch | "
            "effective GB/s |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(lines) + "\n\n"
            "Effective GB/s = (2 x launches - 1) x 2^n x (8 B fp32 / 16 B fp64) / seconds: every tile run reads and "
            "writes the state once, the first one only writes (lazy |0>). It is a model from the launch count, not "
            "a hardware counter, and `seconds` includes planning and table uploads: compare it with the GPU's "
            "memory bandwidth (near it = memory bound; far below = work inside the tiles dominates).\n")


def probe_markdown(docs):
    """`sv_gpu_probe` rows (bench_gpu ... probe): device facts on one line, then one table per run with
    a row per metric and a column per (precision, qubits). The GPU plan's decision gate reads
    `<family>_evolve_over_sweep_floor`: evolve time / (tile runs x empty-sweep time)."""
    out = []
    for d in docs:
        rows = [r for r in d.get("suites", {}).get("sv_gpu_probe", {}).get("rows", [])
                if r.get("kind") == "gpu_probe" and not r.get("variant")]
        if not rows:
            continue
        facts = [f"{r['metric']}={r['value']}" for r in rows if r.get("precision") is None and r.get("value") != "n/a"]
        missing = [r["metric"] for r in rows if r.get("precision") is None and r.get("value") == "n/a"]
        if missing:
            facts.append("not offered by this API: " + " ".join(missing))
        cols = sorted({(r["precision"], r.get("qubits")) for r in rows if r.get("precision")},
                      key=lambda c: (c[0], c[1] or 0))
        metrics = list(dict.fromkeys(r["metric"] for r in rows if r.get("precision")))
        val = {(r["precision"], r.get("qubits"), r["metric"]): r["value"] for r in rows if r.get("precision")}
        unit = {}
        for r in rows:
            if r.get("precision") and r.get("unit"):
                unit.setdefault(r["metric"], r["unit"])  # the first one (fp32): units match across precisions

        def cell(v):
            return _fmt(v, "n") if isinstance(v, (int, float)) else (str(v).replace("|", "/") if v is not None else "-")
        lines = [f"**{run_label(d)}**" + (" - device: " + ", ".join(facts) if facts else ""), "",
                 "| metric | " + " | ".join(f"{p} {q} q" if q else p for p, q in cols) + " | note |",
                 "|---|" + "---|" * (len(cols) + 1)]
        for m in metrics:
            lines.append(f"| {m} | " + " | ".join(cell(val.get((p, q, m))) for p, q in cols)
                         + f" | {(unit.get(m) or '').replace('|', '/')} |")
        out.append("\n".join(lines))
    if out:
        out.append("`<family>_evolve_over_sweep_floor` = evolve time / (tile runs x empty-sweep time): near 1 the "
                   "evolve is at the memory floor of its run count (fewer runs would help most); well above 1 the "
                   "work inside the tiles dominates (cheaper ops and transposes would help most). `op_*_us` are "
                   "per-op slopes inside one run, `stream_*B_GBps` the bandwidth with that many contiguous bytes "
                   "per chunk.")
    return "\n\n".join(out) + "\n" if out else ""


def cross_run_markdown(docs, prefix="QUMO GPU", family="layered"):
    """`prefix` series of several runs side by side, ONE TABLE PER QUMO GPU ENGINE: rows of runs with
    different (or unidentifiable) engines are never joined, so a table compares hardware only."""
    groups = {}
    for d in docs:
        t = series_table(d, family)
        names = [s for s in t if s.startswith(prefix)]
        if names:
            groups.setdefault(engine_identity(d), []).append((d, t, names))
    if not groups:
        return ""
    out = []
    for (eid, ok), members in groups.items():
        qs = sorted({q for _, t, names in members for s in names for q in t[s]})
        lines = [f"Engine {eid}" + ("" if ok else " (not identifiable: these rows are not comparable with any other "
                                                 "run)") + ":", "",
                 "| run | series | " + " | ".join(_qhead(q) for q in qs) + " |", "|---|---|" + "---|" * len(qs)]
        for d, t, names in members:
            for s in sorted(names, key=_series_key):
                lines.append(f"| {run_label(d)} | {s} | " + " | ".join(_fmt(t[s].get(q)) for q in qs) + " |")
        out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


def _pyplot(out=None):
    """Import matplotlib.pyplot safely in any host process.

    Jupyter kernels (Colab, JupyterLab) export MPLBACKEND=module://matplotlib_inline...
    to every child process. In a subprocess running another venv (the notebooks run
    compare_results.py with bench/.venv) that module may not exist, and matplotlib
    then fails at import time. An unusable MPLBACKEND is replaced by the file-only
    Agg backend before matplotlib is imported.
    """
    backend = os.environ.get("MPLBACKEND", "")
    if backend.startswith("module://"):
        try:
            usable = importlib.util.find_spec(backend[len("module://"):]) is not None
        except (ImportError, ValueError):
            usable = False
        if not usable:
            os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    if out and not os.environ.get("DISPLAY") and "inline" not in matplotlib.get_backend().lower():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


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
    plt = _pyplot(out)
    docs = [d for d in docs if statevector_rows(d, family)]
    if not docs:
        return None
    n = len(docs)
    ncols = min(n, 3)  # small multiples wrap at three panels per row
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols + 2.6, 4.4 * nrows), sharey=True, squeeze=False)
    flat = [ax for row in axes for ax in row]
    handles = {}
    for ax, d in zip(flat, docs):
        _draw(ax, d, family, run_label(d))
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    for ax in flat[n:]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("median seconds (log)")
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
    plt = _pyplot(out)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    k = 0
    engines = set()
    styles = ["-", "--", ":", "-."]
    for d in docs:
        t = series_table(d, family)
        eid, ok = engine_identity(d)
        for s in sorted(t, key=_series_key):
            if not s.startswith(prefix) or k >= len(palette):
                continue
            engines.add((eid, ok))
            qs = sorted(t[s])
            prec = "fp32" if "fp32" in s else "fp64"
            short = eid.split(" (")[0] if ok else "engine unknown"
            ax.plot(qs, [t[s][q] for q in qs], color=palette[k], linewidth=2, markersize=6,
                    linestyle=styles[(len(engines) - 1) % len(styles)], marker=PRECISION_MARKER[prec],
                    label=f"{run_label(d)}: {s} [{short}]")  # run labels are unique
            k += 1
    if k == 0:
        plt.close(fig)
        return None
    _axes_format(ax, sorted({int(x) for line in ax.get_lines() for x in line.get_xdata()}))
    ax.set_ylabel("median seconds (log)")
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # below the axes: run labels are long, and inside the axes the legend can hide data points
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, -0.16))
    mixed = len(engines) > 1 or any(not ok for _, ok in engines)
    ax.set_title(f"{prefix} across machines ({family} circuit)" + (
        "\nDIFFERENT or unidentified QUMO engines: not a hardware comparison" if mixed and k > 1 else ""),
        fontsize=11, loc="left", color="#b3261e" if mixed and k > 1 else None)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130, bbox_inches="tight")
    return fig


def summary_markdown(docs, reference=DEFAULT_REFERENCE):
    """Everything a results page needs, in one markdown string."""
    parts = ["## Machines\n", environment_markdown(docs), "\n## Suites\n", status_markdown(docs),
             "\n## How each series was timed\n", fairness_markdown(docs),
             "\n## State-vector timings\n", timing_markdown(docs), "\n## Speed-ups\n", speedup_markdown(docs, reference)]
    sweeps = sweeps_markdown(docs)
    if sweeps:
        parts += ["\n## QUMO GPU: state sweeps and effective bandwidth\n", sweeps]
    probe = probe_markdown(docs)
    if probe:
        parts += ["\n## QUMO GPU engine probe\n", probe]
    if len(docs) > 1:
        cross = cross_run_markdown(docs)
        if cross:
            parts += ["\n## QUMO GPU across runs (one table per engine)\n", cross]
    frames = frames_markdown(docs)
    if frames:
        parts += ["\n## QEC Pauli-frame sampling (QUMO GPU vs CPU)\n", frames]
    parts += ["\n## Compiler subset (hardware-neutral benchmark)\n", compiler_markdown(docs),
              "\n## Problems and skipped parts\n", problems_markdown(docs)]
    return "\n".join(parts)
