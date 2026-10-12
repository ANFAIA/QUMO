"""Cross-validate QUMO's QEC stack against Stim + PyMatching.

For each (distance, p) the *identical* rotated-surface-code circuit (exported
by QUMO's `to_stim`) is fed to Stim. We compare:
  1. the detector error model (every mechanism and probability),
  2. the logical error rate of QUMO-MWPM vs PyMatching (z-score),
  3. QUMO-UF logical error rate (expected slightly higher),
  4. sampling throughput (QUMO single-thread and multi-thread vs Stim).

    bench/.venv/bin/python bench/bench_qec.py [shots]

The Python venv is only a *benchmark* dependency; QUMO itself is pure Mojo.
"""

import math
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import pymatching
import stim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOJO = os.path.join(ROOT, ".venv", "bin", "mojo")
DISTANCES = [3, 5]
PS = [0.001, 0.002, 0.003, 0.005]


def build_binary(tmp):
    exe = os.path.join(tmp, "bench_qec")
    subprocess.run([MOJO, "build", "-I", ROOT, "-o", exe, os.path.join(ROOT, "bench", "bench_qec.mojo")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return exe


def run(exe, *args):
    out = subprocess.run([exe, *map(str, args)], check=True, capture_output=True, text=True, cwd=ROOT)
    return out.stdout


def parse_dem(text):
    errs = {}
    for line in text.splitlines():
        m = re.match(r"error\(([^)]*)\)(.*)", line.strip())
        if not m:
            continue
        toks = m.group(2).replace("^", " ").split()
        dset, oset = set(), set()
        for t in toks:  # XOR semantics across decomposition pieces
            (dset if t[0] == "D" else oset).symmetric_difference_update({int(t[1:])})
        dets = tuple(sorted(dset))
        obs = tuple(sorted(oset))
        key = (dets, obs)
        p = float(m.group(1))
        q = errs.get(key, 0.0)
        errs[key] = q * (1 - p) + p * (1 - q)
    return errs


def stim_dem_dict(circuit):
    dem = circuit.detector_error_model(decompose_errors=False).flattened()
    out = {}
    for inst in dem:
        if inst.type != "error":
            continue
        t = inst.targets_copy()
        dets = tuple(sorted(x.val for x in t if x.is_relative_detector_id()))
        obs = tuple(sorted(x.val for x in t if x.is_logical_observable_id()))
        p = inst.args_copy()[0]
        q = out.get((dets, obs), 0.0)
        out[(dets, obs)] = q * (1 - p) + p * (1 - q)
    return out


def main():
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    with tempfile.TemporaryDirectory() as tmp:
        print("building QUMO bench binary ...", flush=True)
        exe = build_binary(tmp)
        rows = []
        for d in DISTANCES:
            for p in PS:
                text = run(exe, "--stim", d, p)
                circ = stim.Circuit(text)
                # 1. DEM equality
                ours = parse_dem(run(exe, "--dem", d, p))
                theirs = stim_dem_dict(circ)
                same_keys = set(ours) == set(theirs)
                max_rel = max(abs(ours[k] - theirs[k]) / theirs[k] for k in theirs if k in ours)
                # 2./3. QUMO numbers
                q = run(exe, "--run", d, p, shots).strip().split(",")
                (q_shots, q_mt, q_st, q_uf, q_mw, q_uf_s, q_mw_s) = (int(q[10]), float(q[11]), float(q[12]),
                                                                   int(q[16]), int(q[18]), float(q[17]), float(q[19]))
                # Stim + PyMatching
                dem = circ.detector_error_model(decompose_errors=True)
                m = pymatching.Matching.from_detector_error_model(dem)
                sampler = circ.compile_detector_sampler(seed=17)
                t0 = time.perf_counter()
                dets, obs = sampler.sample(shots, separate_observables=True, bit_packed=True)
                t_stim = time.perf_counter() - t0
                dets_u = np.unpackbits(dets, axis=1, bitorder="little")[:, :circ.num_detectors]
                obs_u = np.unpackbits(obs, axis=1, bitorder="little")[:, :circ.num_observables]
                t0 = time.perf_counter()
                pred = m.decode_batch(dets_u)
                t_pm = time.perf_counter() - t0
                pm_err = int(np.sum(np.any(pred != obs_u, axis=1)))
                p1 = q_mw / shots
                p2 = pm_err / shots
                se = math.sqrt(max(p1 * (1 - p1), 1e-12) / shots + max(p2 * (1 - p2), 1e-12) / shots)
                z = (p1 - p2) / se
                rows.append((d, p, same_keys, max_rel, pm_err / shots, p1, z, q_uf / shots,
                             shots / t_stim, shots / q_st, shots / q_mt, shots / t_pm, shots / q_mw_s, shots / q_uf_s))
                print(f"d={d} p={p}: DEM identical={same_keys} (max rel dp {max_rel:.1e}); "
                      f"PyMatching {pm_err}/{shots}, QUMO-MWPM {q_mw}/{shots} (z={z:+.2f}), QUMO-UF {q_uf}/{shots}",
                      flush=True)
        print()
        print("| d | p | DEM == Stim | p_L PyMatching | p_L QUMO MWPM | z | p_L QUMO UF | Stim shots/s | QUMO shots/s (1 thr) | QUMO shots/s (all thr) | PyMatching dec/s | QUMO MWPM sample+dec/s | QUMO UF sample+dec/s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r[0]} | {r[1]} | {'yes' if r[2] and r[3] < 1e-9 else 'NO'} | {r[4]:.2e} | {r[5]:.2e} | {r[6]:+.2f} | {r[7]:.2e} | "
                  f"{r[8]:.3g} | {r[9]:.3g} | {r[10]:.3g} | {r[11]:.3g} | {r[12]:.3g} | {r[13]:.3g} |")
        worst = max(abs(r[6]) for r in rows)
        print(f"\nmax |z| (QUMO MWPM vs PyMatching) = {worst:.2f}  -> {'agree within statistics' if worst < 3.5 else 'CHECK'}")


if __name__ == "__main__":
    main()
