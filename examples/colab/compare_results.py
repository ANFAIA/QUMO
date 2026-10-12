#!/usr/bin/env python3
"""Merge several QUMO benchmark result JSONs into one markdown report + plots.

    python examples/colab/compare_results.py results/*.json
    python examples/colab/compare_results.py run_t4.json run_mi300x.json run_m5.json \
        --out comparison.md --plot comparison.png --reference "Aer CPU fp64"

Inputs are the JSON files written by bench/run_benchmarks.py (directories are
scanned for *.json). The report lists every machine (GPU, driver, CPU, RAM,
toolchain, commit, load average), the status of every suite, a timing table of
the layered circuit (one row per run and engine/device/precision, one column
per qubit count), speed-ups against a reference series, and every skipped or
failed part with its QM-E96x reason. Plots: one panel per run, plus a
cross-machine view of the QUMO GPU series.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_report as br  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", nargs="+", help="result JSON files or directories")
    ap.add_argument("--out", default="", help="markdown output (default: stdout)")
    ap.add_argument("--plot", default="", help="PNG for the per-run panels (also writes <stem>_qumo_gpu.png)")
    ap.add_argument("--family", default="layered", help="circuit family (layered / qft / random)")
    ap.add_argument("--reference", default="Aer CPU fp64", help="series used for the speed-up table")
    a = ap.parse_args(argv)

    docs = br.load_results(a.results)
    if not docs:
        print("QM-E960: no QUMO result JSON found in the given paths. Hint: pass files written by "
              "bench/run_benchmarks.py (schema qumo-bench/1).", file=sys.stderr)
        return 2
    md = [f"# QUMO benchmark comparison ({len(docs)} run{'s' if len(docs) > 1 else ''})\n",
          "Inputs: " + ", ".join(f"`{os.path.basename(d['_path'])}`" for d in docs) + "\n",
          br.summary_markdown(docs, a.reference)]
    if a.family != "layered":
        md.append(f"\n## {a.family} circuit\n\n" + br.timing_markdown(docs, a.family))
    if a.plot:
        fig = br.plot_time_vs_qubits(docs, a.family, out=a.plot)
        stem = os.path.splitext(a.plot)[0]
        fig2 = br.plot_series_across_runs(docs, "QUMO GPU", a.family, out=stem + "_qumo_gpu.png")
        rel = os.path.dirname(os.path.abspath(a.out)) if a.out else os.getcwd()
        if fig is not None:
            md.append(f"\n![time vs qubits]({os.path.relpath(os.path.abspath(a.plot), rel)})\n")
        if fig2 is not None:
            md.append(f"\n![QUMO GPU across machines]({os.path.relpath(os.path.abspath(stem + '_qumo_gpu.png'), rel)})\n")
    text = "\n".join(md)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
        print(f"wrote {a.out}" + (f" and {a.plot}" if a.plot else ""))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
