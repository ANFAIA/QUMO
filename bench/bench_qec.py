"""Cross-validate QUMO's QEC stack against Stim + PyMatching.

Part A - for each (distance, p) the *identical* rotated-surface-code circuit
(exported by QUMO's `to_stim`) is fed to Stim. We compare:
  1. the detector error model (every mechanism and probability),
  2. the logical error rate of QUMO-MWPM (sparse blossom; fused
     sample+decode on QUMO's own samples) vs PyMatching on Stim samples
     (z-score), optionally also the dense MWPM,
  3. QUMO-UF logical error rate (expected slightly higher),
  4. sampling throughput (QUMO single-thread and multi-thread vs Stim).

Part B - *identical syndromes*: Stim samples detection events once (b8 file);
PyMatching 2 and QUMO's sparse blossom (bench/bench_qec_decoders.mojo) decode
the very same shots. We report shot-by-shot prediction agreement, both
logical error rates on the same data, and decode-only throughput (QUMO single
thread and all threads vs PyMatching single thread). PyMatching and QUMO
repetitions are interleaved A/B and medians are reported (shared machine).

    bench/.venv/bin/python bench/bench_qec.py [--part A|B|all] [--shots N]
           [--tp-shots N] [--reps R] [--dense]

The Python venv is only a *benchmark* dependency; QUMO itself is pure Mojo.
"""

import argparse
import math
import os
import re
import subprocess
import tempfile
import time

import numpy as np
import pymatching
import stim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOJO = os.path.join(ROOT, ".venv", "bin", "mojo")
DISTANCES = [3, 5, 7]
PS = [0.001, 0.002, 0.003, 0.005]
# Part B grid: (code, d, p, shots scale); throughput rows are surface_z.
TP_DISTANCES = [5, 7, 9, 11]
TP_PS = [0.001, 0.005]
AGREE_EXTRA = [("surface_x", 5, 0.005), ("surface_x", 7, 0.005), ("repetition", 7, 0.01), ("repetition", 11, 0.005)]


def build(tmp, name):
    exe = os.path.join(tmp, name)
    subprocess.run([MOJO, "build", "-I", ROOT, "-o", exe, os.path.join(ROOT, "bench", name + ".mojo")],
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


def zscore(e1, n1, e2, n2):
    p1, p2 = e1 / n1, e2 / n2
    se = math.sqrt(max(p1 * (1 - p1), 1e-12) / n1 + max(p2 * (1 - p2), 1e-12) / n2)
    return (p1 - p2) / se


def part_a(exe, shots, dense):
    rows = []
    for d in DISTANCES:
        for p in PS:
            text = run(exe, "--stim", d, p)
            circ = stim.Circuit(text)
            ours = parse_dem(run(exe, "--dem", d, p))
            theirs = stim_dem_dict(circ)
            same_keys = set(ours) == set(theirs)
            max_rel = max(abs(ours[k] - theirs[k]) / theirs[k] for k in theirs if k in ours)
            args = ["--run", d, p, shots] + (["dense"] if dense else [])
            q = run(exe, *args).strip().split(",")
            q_mt, q_st = float(q[11]), float(q[12])
            q_uf, q_uf_s, q_mw, q_mw_s = int(q[16]), float(q[17]), int(q[18]), float(q[19])
            q_de = int(q[21])
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
            z = zscore(q_mw, shots, pm_err, shots)
            zd = zscore(q_de, shots, pm_err, shots) if q_de >= 0 else float("nan")
            rows.append((d, p, same_keys, max_rel, pm_err / shots, q_mw / shots, z, q_de / shots, zd, q_uf / shots,
                         shots / t_stim, shots / q_st, shots / q_mt, shots / t_pm, shots / q_mw_s, shots / q_uf_s))
            print(f"d={d} p={p}: DEM identical={same_keys} (max rel dp {max_rel:.1e}); PyMatching {pm_err}/{shots}, "
                  f"QUMO-MWPM(sparse) {q_mw}/{shots} (z={z:+.2f})"
                  + (f", QUMO-MWPM(dense) {q_de}/{shots} (z={zd:+.2f})" if q_de >= 0 else "")
                  + f", QUMO-UF {q_uf}/{shots}", flush=True)
    print()
    print("| d | p | DEM == Stim | p_L PyMatching | p_L QUMO MWPM (sparse) | z | p_L QUMO MWPM (dense) | z | p_L QUMO UF "
          "| Stim shots/s | QUMO sample shots/s (1 thr) | QUMO sample shots/s (all thr) | PyMatching dec/s (1 thr) "
          "| QUMO MWPM sample+dec/s (all thr) | QUMO UF sample+dec/s (all thr) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        dense_cols = f"{r[7]:.2e} | {r[8]:+.2f}" if r[7] >= 0 else "- | -"
        print(f"| {r[0]} | {r[1]} | {'yes' if r[2] and r[3] < 1e-9 else 'NO'} | {r[4]:.2e} | {r[5]:.2e} | {r[6]:+.2f} | "
              f"{dense_cols} | {r[9]:.2e} | {r[10]:.3g} | {r[11]:.3g} | {r[12]:.3g} | {r[13]:.3g} | {r[14]:.3g} | {r[15]:.3g} |")
    worst = max(abs(r[6]) for r in rows)
    print(f"\nmax |z| (QUMO sparse-blossom MWPM vs PyMatching) = {worst:.2f}  -> "
          f"{'agree within statistics' if worst < 3.5 else 'CHECK'}")


def part_b(exe_stim, exe_dec, tp_shots, reps, tmp):
    rows = []
    points = [("surface_z", d, p, True) for d in TP_DISTANCES for p in TP_PS]
    points += [(c, d, p, False) for (c, d, p) in AGREE_EXTRA]
    for code, d, p, timed in points:
        circ = stim.Circuit(run(exe_stim, "--stim", code, d, p))
        # shots: enough to time ~0.2-1 s of PyMatching work per repetition
        shots = tp_shots if (d <= 7 or p < 0.003) else max(2000, tp_shots // 3)
        dets, obs = circ.compile_detector_sampler(seed=1234 + d).sample(shots, separate_observables=True, bit_packed=True)
        path = os.path.join(tmp, f"{code}_{d}_{p}.b8")
        dets.tofile(path)
        obs_u = np.unpackbits(obs, axis=1, bitorder="little")[:, :circ.num_observables]
        actual = np.zeros(shots, dtype=np.uint64)
        for k in range(circ.num_observables):
            actual |= obs_u[:, k].astype(np.uint64) << np.uint64(k)
        m = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
        dets_u = np.unpackbits(dets, axis=1, bitorder="little")[:, :circ.num_detectors]
        t_pm, t_q1, t_qa = [], [], []
        pm_pred = None
        line = None
        for _ in range(reps if timed else 1):
            t0 = time.perf_counter()
            pm_pred = m.decode_batch(dets_u)
            t_pm.append(time.perf_counter() - t0)
            line = run(exe_dec, "--decode", code, d, p, path, shots, "sparse_blossom", 1).strip().split(",")
            t_q1.append(float(line[6]))
            t_qa.append(float(line[7]))
        pmv = np.zeros(shots, dtype=np.uint64)
        for k in range(pm_pred.shape[1]):
            pmv |= pm_pred[:, k].astype(np.uint64) << np.uint64(k)
        qv = np.fromfile(path + ".pred", dtype=np.uint8).astype(np.uint64)
        disagree = int(np.sum(qv != pmv))
        e_pm = int(np.sum(pmv != actual))
        e_q = int(np.sum(qv != actual))
        med = lambda xs: sorted(xs)[len(xs) // 2]
        r = dict(code=code, d=d, p=p, shots=shots, events=float(line[5]), disagree=disagree, e_pm=e_pm, e_q=e_q,
                 anomalies=int(line[9]), failures=int(line[10]), threads=int(line[8]), timed=timed,
                 pm=shots / med(t_pm), q1=shots / med(t_q1), qa=shots / med(t_qa))
        rows.append(r)
        print(f"{code} d={d} p={p}: {shots} identical shots, {r['events']:.1f} events/shot; predictions differ on "
              f"{disagree}; errors PyMatching {e_pm} vs QUMO {e_q}; decode/s PyMatching(1 thr) {r['pm']:.3g}, "
              f"QUMO sparse blossom 1 thr {r['q1']:.3g}, {r['threads']} thr {r['qa']:.3g} "
              f"(ratio 1-thr QUMO/PyMatching {r['q1'] / r['pm']:.2f})", flush=True)
    print()
    print(f"Identical syndromes (Stim samples), decode-only, medians of {reps} interleaved repetitions:")
    print()
    print("| code | d | p | shots | events/shot | predictions differ | errors PyMatching | errors QUMO "
          "| PyMatching dec/s (1 thr) | QUMO sparse blossom dec/s (1 thr) | QUMO / PyMatching (1 thr) | QUMO dec/s (all thr) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        tp = (f"{r['pm']:.3g} | {r['q1']:.3g} | {r['q1'] / r['pm']:.2f} | {r['qa']:.3g}" if r["timed"]
              else "- | - | - | -")
        print(f"| {r['code']} | {r['d']} | {r['p']} | {r['shots']} | {r['events']:.1f} | {r['disagree']} "
              f"({100.0 * r['disagree'] / r['shots']:.3f} %) | {r['e_pm']} | {r['e_q']} | {tp} |")
    bad = [r for r in rows if r["anomalies"] or r["failures"]]
    print(f"\nQUMO anomalies/failures: {'none' if not bad else bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots_pos", nargs="?", type=int, default=None, help="Part A shots (legacy positional)")
    ap.add_argument("--part", default="all", choices=["A", "B", "all"])
    ap.add_argument("--shots", type=int, default=200_000, help="Part A shots per point")
    ap.add_argument("--tp-shots", type=int, default=30_000, help="Part B shots per point (d >= 9 at p >= 0.3%%: /3)")
    ap.add_argument("--reps", type=int, default=3, help="Part B interleaved repetitions (median)")
    ap.add_argument("--dense", action="store_true", help="Part A: also run the dense MWPM decoder")
    a = ap.parse_args()
    shots = a.shots_pos or a.shots
    with tempfile.TemporaryDirectory() as tmp:
        print("building QUMO bench binaries ...", flush=True)
        exe = build(tmp, "bench_qec")
        if a.part in ("A", "all"):
            part_a(exe, shots, a.dense)
        if a.part in ("B", "all"):
            exe_dec = build(tmp, "bench_qec_decoders")
            part_b(exe_dec, exe_dec, a.tp_shots, a.reps, tmp)


if __name__ == "__main__":
    main()
