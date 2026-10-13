#!/usr/bin/env python3
"""Count the GPU kernels CUDA-Q launches for one layered-circuit state (NVIDIA, Nsight Systems).

    python examples/colab/count_gpu_kernels.py --python $QUMO_CUDAQ_PYTHON --qubits 26 --precision fp32
    python examples/colab/count_gpu_kernels.py ... --env CUDAQ_FUSION_MAX_QUBITS=0      # without fusion

Why: CUDA-Q fuses gates inside the closed cuStateVec Ex library, so its number of passes over
the state vector cannot be read from its source. QUMO's GPU benchmark reports its own launch
count (`launches` in the result JSON); this script measures CUDA-Q's under `nsys`, the same
timed call the benchmark makes (`bench/run_benchmarks.py --worker cudaq`).

Method: two profiles of the benchmark worker at one width, with 1 and with 2 timed
repetitions (each also runs the 8-qubit check and one warm-up call). The difference of the
two kernel counts is the number of kernels ONE `cudaq.get_state` call launches, with every
one-time kernel (initialisation, JIT, the check) cancelled out. Each kernel is at most one
pass over the state; small helper kernels (zeroing, fusion bookkeeping) count too, so the
result is an UPPER bound on state sweeps. The per-call kernel time is the difference of the
two total kernel times.

Needs `nsys` (Nsight Systems CLI: on Colab look in /usr/local/cuda/bin or
/opt/nvidia/nsight-systems/*/bin, or install `nsight-systems-cli` from NVIDIA's apt repository).
Prints one JSON line; `--out` also writes it to a file.
"""

import argparse
import csv
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_nsys(explicit=None):
    if explicit:
        return explicit if os.path.exists(explicit) else shutil.which(explicit)
    for cand in [shutil.which("nsys")] + sorted(glob.glob("/usr/local/cuda*/bin/nsys"), reverse=True) + \
            sorted(glob.glob("/opt/nvidia/nsight-systems*/*/bin/nsys"), reverse=True) + \
            sorted(glob.glob("/opt/nvidia/nsight-systems/*/bin/nsys"), reverse=True):
        if cand and os.path.exists(cand):
            return cand
    return None


def parse_kernel_summary(text):
    """`nsys stats --report cuda_gpu_kern_sum --format csv` output -> (instances, total_ns, rows).

    nsys prints progress lines before the CSV; the table starts at the header that has an
    'Instances' column. Column names differ slightly between nsys versions, so they are
    matched loosely."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "Instances" in ln and "," in ln), None)
    if start is None:
        return None, None, []
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    inst_key = next((k for k in rows[0] if k and "Instances" in k), None) if rows else None
    time_key = next((k for k in rows[0] if k and k.startswith("Total Time")), None) if rows else None
    n = t = 0
    out = []
    for r in rows:
        try:
            i = int(float(str(r.get(inst_key, "0")).replace(",", "")))
        except ValueError:
            continue  # trailing non-table line
        ns = float(str(r.get(time_key, "0")).replace(",", "") or 0) if time_key else 0.0
        n += i
        t += ns
        out.append({"name": (r.get("Name") or "")[:120], "instances": i, "total_ns": ns})
    return n, t, out


def profile(nsys, python, params, env, workdir, tag):
    rep = os.path.join(workdir, f"cudaq_{tag}")
    cmd = [nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--force-overwrite=true",
           "-o", rep, python, os.path.join(ROOT, "bench", "run_benchmarks.py"), "--worker", "cudaq",
           json.dumps(params)]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nsys profile failed ({r.returncode}): {(r.stderr or r.stdout)[-800:]}")
    path = rep + ".nsys-rep" if os.path.exists(rep + ".nsys-rep") else rep + ".qdrep"
    for report in ("cuda_gpu_kern_sum", "gpukernsum"):  # renamed in nsys 2023.x
        s = subprocess.run([nsys, "stats", "--report", report, "--format", "csv", "--force-export=true", path],
                           capture_output=True, text=True)
        n, t, rows = parse_kernel_summary(s.stdout)
        if n is not None:
            return n, t, rows
    raise RuntimeError(f"no kernel summary from nsys stats for {path}: {s.stderr[-400:]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--python", default=os.environ.get("QUMO_CUDAQ_PYTHON", sys.executable),
                    help="interpreter with CUDA-Q (default: $QUMO_CUDAQ_PYTHON)")
    ap.add_argument("--qubits", type=int, default=26)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp64"])
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the CUDA-Q process (repeatable)")
    ap.add_argument("--nsys", default=None, help="path to nsys (default: search PATH and the usual install dirs)")
    ap.add_argument("--out", default="", help="also write the JSON result to this file")
    a = ap.parse_args(argv)
    nsys = find_nsys(a.nsys)
    if not nsys:
        print("QM-E963: nsys (Nsight Systems CLI) not found. Hint: install nsight-systems-cli or pass --nsys.",
              file=sys.stderr)
        return 2
    extra = dict(kv.split("=", 1) for kv in a.env)
    env = dict(os.environ, **extra)
    base = {"qubits": [a.qubits], "depth": 10, "precision": a.precision, "device": "gpu", "verify_qubits": 8,
            "target": "nvidia"}
    with tempfile.TemporaryDirectory() as wd:
        n1, t1, _ = profile(nsys, a.python, dict(base, reps=1), env, wd, "r1")
        n2, t2, rows2 = profile(nsys, a.python, dict(base, reps=2), env, wd, "r2")
    res = {"engine": "cudaq", "qubits": a.qubits, "precision": a.precision, "env": extra,
           "kernels_per_get_state": n2 - n1, "kernel_ms_per_get_state": round((t2 - t1) / 1e6, 3),
           "kernels_profile_reps1": n1, "kernels_profile_reps2": n2,
           "top_kernels_reps2": sorted(rows2, key=lambda r: -r["total_ns"])[:8],
           "note": "kernels of one get_state call (difference of two profiles); an upper bound on state sweeps"}
    text = json.dumps(res)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
