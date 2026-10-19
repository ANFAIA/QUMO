#!/usr/bin/env python3
"""Sample-based Krylov quantum diagonalisation of a Fermi-Hubbard chain: QPU sampling, CPU configuration recovery, HPC diagonalisation.

Reproduces: Qiskit addon SQD - configuration recovery and subsampled diagonalisation (IBM + RIKEN quantum-centric supercomputing), on a Krylov (SKQD) Hubbard instance
Source: https://github.com/Qiskit/qiskit-addon-sqd
Source: https://arxiv.org/abs/2405.05068
Source: https://arxiv.org/abs/2501.09702
Source: https://quantum.cloud.ibm.com/docs/tutorials/sample-based-krylov-quantum-diagonalization

Pipeline (each device does what the others cannot):
  QPU  (superconducting emulator, 2x8 grid, noisy): sample Krylov states
       U^k |Neel>, U = Trotterised exp(-i H dt) - the only stage that touches
       the quantum state; noise breaks particle-number and spin symmetry.
  CPU  configuration recovery (Robledo-Moreno et al. 2024, qiskit-addon-sqd
       `recover_configurations`): flips bits of wrong-sector samples using
       orbital occupations of the previous ground-state estimate; branchy,
       per-sample, sequential logic.
  HPC  (QUMO Mojo, all CPU cores): product-basis projection of H and its
       lowest eigenpair (qumo_py.subspace_ground_state: parallel ELL matrix,
       restarted Lanczos); returns the occupations that drive the next
       recovery round. On IBM + RIKEN this ran on Fugaku's CPU nodes.
  GPU  (optional, ideal twin): the same Krylov circuits without noise on the
       GPU state vector; verification only (at scale no classical twin
       exists). CPU fallback when no GPU build / device is available.

Problem: 8-site open Hubbard chain, t = 1, U = 4 (particle-hole symmetric
form), half filling, 16 qubits (alpha orbitals 0-7, beta 8-15; the
Jordan-Wigner hopping terms and the on-site interaction form exactly a 2x8
ladder, so the grid target needs no SWAPs). The (4, 4) sector has 4900
determinants; the subspace is capped by samples_per_batch to emulate the
regime where only a sampled fraction of the sector can be diagonalised.

Checks: HPC solver = SciPy on the full sector (1e-8); the QPU samples are
noisy (in-sector fraction <= 0.95); recovery puts 100 % of the shots in the
sector; every SQD energy is variational; the final SQD energy is within the
stated band of the exact energy with a subspace below 60 % of the sector;
recovery is not worse than postselection at the same budget (1e-2
relative slack); twin energy is variational; nothing is submitted remotely.

Run:  bench/.venv/bin/python examples/heterogeneous/sqd/sqd_hubbard.py
      (PYTHONPATH=build:plugins/hardware; build/gpu for the GPU twin)
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 180, "requires": ["qumo_py", "scipy"], "args": ["--seed", "7"]}
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import hetero_common as H  # noqa: E402

DEFAULT_QPU = "local:qumo-sim:grid:2x8/sc_cz+noise:superconducting"


# --------------------------------------------------------------------------- problem
def hubbard_observable(L: int, t: float, U: float) -> str:
    """-t sum (a+_i a_j + h.c.) + U sum (n_up - 1/2)(n_dn - 1/2), Jordan-Wigner, open chain.
    Orbitals: alpha i -> qubit i, beta i -> qubit L + i."""
    terms = []
    for s in range(2):
        for i in range(L - 1):
            a = s * L + i
            terms.append(f"{-t / 2!r}*X{a} X{a + 1}")
            terms.append(f"{-t / 2!r}*Y{a} Y{a + 1}")
    for i in range(L):
        terms.append(f"{U / 4!r}*Z{i} Z{i + L}")
    return " + ".join(terms)


def sector_strings(L: int, N: int) -> list[int]:
    return [sum(1 << i for i in c) for c in itertools.combinations(range(L), N)]


def scipy_sector_energy(L: int, t: float, U: float, Na: int, Nb: int) -> float:
    """Independent reference: the (Na, Nb) sector matrix built in NumPy, lowest eigenvalue by SciPy."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    A = sector_strings(L, Na)
    B = sector_strings(L, Nb)
    ia = {s: k for k, s in enumerate(A)}
    ib = {s: k for k, s in enumerate(B)}
    nA, nB = len(A), len(B)
    rows, cols, vals = [], [], []
    for kb, b in enumerate(B):
        for ka, a in enumerate(A):
            r = kb * nA + ka
            # on-site: U/4 * z_up z_dn, z = 1 - 2 n
            d = sum(U / 4 * (1 - 2 * ((a >> i) & 1)) * (1 - 2 * ((b >> i) & 1)) for i in range(L))
            rows.append(r); cols.append(r); vals.append(d)
            for i in range(L - 1):  # nearest-neighbour hops: no Jordan-Wigner sign within one spin chain
                m = (1 << i) | (1 << (i + 1))
                if bin(a & m).count("1") == 1:
                    rows.append(r); cols.append(kb * nA + ia[a ^ m]); vals.append(-t)
                if bin(b & m).count("1") == 1:
                    rows.append(r); cols.append(ib[b ^ m] * nA + ka); vals.append(-t)
    M = sp.csr_matrix((vals, (rows, cols)), shape=(nA * nB, nA * nB))
    return float(sla.eigsh(M, k=1, which="SA", tol=1e-12)[0][0])


def krylov_qasm(L: int, t: float, U: float, k: int, dt: float, slices: int) -> str:
    """|psi_k> = U(dt)^k |Neel>, Trotter step (even hops, odd hops, on-site), measured."""
    n = 2 * L
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n}] q;", f"bit[{n}] c;"]
    for i in range(L):  # Neel: alpha on even sites, beta on odd sites (Sz = 0, half filling)
        lines.append(f"x q[{i}];" if i % 2 == 0 else f"x q[{i + L}];")
    h = dt / slices
    for _ in range(k * slices):
        for par in (0, 1):
            for s in range(2):
                for i in range(par, L - 1, 2):
                    a = s * L + i
                    lines.append(f"rxx({-t * h!r}) q[{a}], q[{a + 1}];")
                    lines.append(f"ryy({-t * h!r}) q[{a}], q[{a + 1}];")
        for i in range(L):
            lines.append(f"rzz({U * h / 2!r}) q[{i}], q[{i + L}];")
    for i in range(n):
        lines.append(f"c[{i}] = measure q[{i}];")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- configuration recovery (CPU)
def _relu_weight(y: np.ndarray, filling: float, delta: float = 0.01) -> np.ndarray:
    """Modified ReLU of Robledo-Moreno et al. (2024), supplementary S-CR."""
    h0 = filling
    return np.where(y <= h0, delta * y / max(h0, 1e-12), delta + (1 - delta) * (y - h0) / max(1 - h0, 1e-12))


def recover_half(bits: int, n_target: int, occ: np.ndarray, rng: np.random.Generator) -> int:
    """Flip bits of one spin half until it holds n_target electrons (probabilities from occupations)."""
    L = len(occ)
    x = np.array([(bits >> i) & 1 for i in range(L)], dtype=float)
    k = int(x.sum()) - n_target
    if k == 0:
        return bits
    filling = n_target / L
    if k > 0:  # too many electrons: empty occupied orbitals that are likely empty
        cand = np.flatnonzero(x == 1)
        w = _relu_weight(np.abs(1.0 - occ[cand]), filling)
    else:  # too few: fill empty orbitals that are likely occupied
        cand = np.flatnonzero(x == 0)
        w = _relu_weight(np.abs(occ[cand]), filling)
    w = w + 1e-12
    pick = rng.choice(cand, size=abs(k), replace=False, p=w / w.sum())
    for i in pick:
        bits ^= 1 << int(i)
    return bits


def split(x: int, L: int) -> tuple[int, int]:
    return x & ((1 << L) - 1), x >> L


def in_sector(x: int, L: int, Na: int, Nb: int) -> bool:
    a, b = split(x, L)
    return bin(a).count("1") == Na and bin(b).count("1") == Nb


def shots_list(counts: dict[int, int]) -> list[int]:
    out = []
    for x in sorted(counts):
        out += [x] * counts[x]
    return out


def sqd_round(q, obs: str, configs: list[int], L: int, spb: int, batches: int, rng: np.random.Generator) -> dict:
    """Subsample `batches` x `spb` configurations; product basis of the union of alpha and beta
    strings (spin-symmetric, Na = Nb); lowest eigenpair per batch on the HPC stage; keep the best."""
    best = None
    dims, secs = [], 0.0
    configs = np.asarray(configs, dtype=np.int64)
    for _ in range(batches):
        take = configs if len(configs) <= spb else rng.choice(configs, size=spb, replace=False)
        strings = sorted({int(x) & ((1 << L) - 1) for x in take} | {int(x) >> L for x in take})
        basis = [a | (b << L) for b in strings for a in strings]
        r = dict(q.subspace_ground_state(obs, basis, 2 * L))
        secs += r["seconds"]
        dims.append(len(basis))
        if best is None or r["energy"] < best["energy"]:
            best = r
            best["num_strings"] = len(strings)
    best["dims"] = dims
    best["hpc_seconds"] = secs
    return best


# --------------------------------------------------------------------------- pipeline
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", type=int, default=8, help="chain length L (2L qubits); <= 13")
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--U", type=float, default=4.0)
    ap.add_argument("--krylov", type=int, default=3, help="Krylov dimension D (states k = 1..D)")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--slices", type=int, default=1, help="Trotter slices per Krylov step")
    ap.add_argument("--shots", type=int, default=2000, help="QPU shots per Krylov state")
    ap.add_argument("--samples-per-batch", type=int, default=50)
    ap.add_argument("--max-subspace-fraction", type=float, default=0.6,
                    help="check: largest subspace / sector dimension must stay below this")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--iterations", type=int, default=3, help="configuration-recovery rounds")
    ap.add_argument("--band", type=float, default=0.05, help="allowed relative energy error of the final SQD energy")
    ap.add_argument("--qpu", default=DEFAULT_QPU, help=f"QPU target (default {DEFAULT_QPU})")
    ap.add_argument("--gpu-device", choices=["auto", "gpu", "cpu", "none"], default="auto",
                    help="ideal twin: auto (GPU when usable), gpu, cpu, none (skip)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(H.REPO / "build" / "hetero" / "sqd"))
    ap.add_argument("--quiet", action="store_true")
    H.add_safety_args(ap)
    return ap.parse_args(argv)


def run(argv=None, runner=None) -> dict:
    a = parse_args(argv)
    say = (lambda *_: None) if a.quiet else (lambda m: print(m, flush=True))
    from _checks import CheckLog

    t_start = time.perf_counter()
    os.environ.setdefault("QUMO_HW_NOISE_WORKERS", str(max(1, min(8, (os.cpu_count() or 2) - 1))))
    q = H.load_qumo()
    H.require_bindings(q, ["subspace_ground_state", "schedule", "sample_device"])
    H.check_targets([a.qpu], a.allow_remote)
    gpu = H.gpu_status(q)
    L, n = a.sites, 2 * a.sites
    Na = Nb = L // 2
    obs = hubbard_observable(L, a.t, a.U)
    stages = H.StageLog(say)
    rng = np.random.default_rng(a.seed)
    log = CheckLog("heterogeneous/sqd_hubbard")
    say(f"[problem] {L}-site Hubbard chain t={a.t} U={a.U}, {n} qubits, sector ({Na},{Nb}) "
        f"= {math.comb(L, Na) * math.comb(L, Nb)} determinants; QPU {a.qpu}")

    # ---- reference (HPC + SciPy): exact sector energy
    full = [x | (y << L) for y in sector_strings(L, Nb) for x in sector_strings(L, Na)]
    with H.Timer() as tm:
        ref = dict(q.subspace_ground_state(obs, full, n))
    e0 = ref["energy"]
    with H.Timer() as tm2:
        e_scipy = scipy_sector_energy(L, a.t, a.U, Na, Nb)
    stages.add(H.StageRecord("reference", "cpu (QUMO HPC + SciPy)", "exact energy for the checks only "
                             "(possible here because the sector is small)", tm.seconds + tm2.seconds,
                             f"full sector {len(full)} states, {ref['matvecs']} matvecs"))
    log.close("hpc solver equals scipy on the full sector", e0, e_scipy, 1e-8)

    # ---- orchestration: Krylov jobs pinned to the QPU, twin jobs placed by HEFT
    circuits = [krylov_qasm(L, a.t, a.U, k, a.dt, a.slices) for k in range(1, a.krylov + 1)]
    qcircs = [q.load_qasm(c) for c in circuits]
    devices = [{"name": "cpu0", "kind": "cpu", "max_qubits": 30}, {"name": "gpu0", "kind": "gpu", "max_qubits": 30},
               H.qpu_device("sc-qpu", a.qpu, 133)]
    jobs = [{"name": f"krylov_{k + 1}", "circuit": c, "shots": a.shots, "device": "sc-qpu"} for k, c in enumerate(qcircs)]
    if a.gpu_device != "none":
        jobs += [{"name": f"twin_{k + 1}", "circuit": c, "shots": a.shots} for k, c in enumerate(qcircs)]
    sch = H.heft(q, jobs, devices, say)

    # ---- QPU stage
    gate = H.QpuGate(a.allow_remote, a.submit, a.yes, a.max_cost, runner=runner, say=say)
    raw: dict[int, int] = {}
    with H.Timer() as tm:
        for k, c in enumerate(circuits):
            rec = gate.run(c, a.qpu, a.shots, a.seed + k, f"krylov_{k + 1}")
            if rec["counts"] is None:
                continue
            for x, v in H.counts_to_ints(rec["counts"]).items():
                raw[x] = raw.get(x, 0) + v
    cz = [j.get("compiled", {}).get("two_qubit_gates") for j in gate.jobs]
    stages.add(H.StageRecord("qpu-sampling", f"qpu:{a.qpu}", "the only stage that prepares and measures the "
                             "Krylov states", tm.seconds, f"{len(circuits)} circuits x {a.shots} shots, "
                             f"2q gates {cz}", emulated=not H.is_remote(a.qpu),
                             notes=["vendor-neutral depolarizing emulator (modality preset), not a device"]))
    log.check("no remote submission", gate.nothing_remote_submitted(), f"{len(gate.jobs)} QPU job(s)")
    if not raw:
        log.check("qpu returned samples", False, "dry run or refused: nothing to diagonalise")
        log.finish(exit_on_failure=(__name__ == "__main__"))
        return {"checks": log.records}
    total = sum(raw.values())
    frac_in = sum(v for x, v in raw.items() if in_sector(x, L, Na, Nb)) / total
    log.within("qpu samples are noisy (in-sector fraction)", frac_in, 0.0, 0.95)

    # ---- optional GPU twin (ideal samples)
    twin = None
    if a.gpu_device != "none":
        want = "gpu" if a.gpu_device in ("auto", "gpu") and "gpu0" in sch["device_of"] else a.gpu_device
        dev, why = H.place_gpu(gpu, n, "gpu" if want == "gpu" else "cpu")
        ideal: dict[int, int] = {}
        methods = set()
        with H.Timer() as tm:
            for k, c in enumerate(qcircs):
                counts, method, note = q.sample_device(c, a.shots, a.seed + 100 + k, dev)
                methods.add(str(method))
                for x, v in H.counts_to_ints(dict(counts)).items():
                    ideal[x] = ideal.get(x, 0) + v
        twin = {"device": dev, "methods": sorted(methods), "note": why, "counts": ideal}
        stages.add(H.StageRecord("ideal-twin", dev, "noiseless twin of the QPU circuits (verification only)",
                                 tm.seconds, ",".join(sorted(methods)), notes=[why] if why else []))

    # ---- CPU configuration recovery + HPC diagonalisation, iterated
    shots = shots_list(raw)
    post = [x for x in shots if in_sector(x, L, Na, Nb)]
    history = []
    t_cpu = t_hpc = 0.0
    with H.Timer() as tm:
        base = sqd_round(q, obs, post, L, a.samples_per_batch, a.batches, np.random.default_rng(a.seed + 1))
    t_hpc += base["hpc_seconds"]
    history.append({"iteration": 0, "kind": "postselection", "energy": base["energy"], "dims": base["dims"],
                    "usable_shots": len(post)})
    occ = np.array(base["occupations"])
    cur = base
    recovered_all_in = True
    for it in range(1, a.iterations + 1):
        t0 = time.perf_counter()
        r_rng = np.random.default_rng(a.seed * 1000 + it)
        rec_shots = []
        for x in shots:
            al, be = split(x, L)
            al = recover_half(al, Na, occ[:L], r_rng)
            be = recover_half(be, Nb, occ[L:], r_rng)
            rec_shots.append(al | (be << L))
        recovered_all_in &= all(in_sector(x, L, Na, Nb) for x in rec_shots)
        t_cpu += time.perf_counter() - t0
        cur = sqd_round(q, obs, rec_shots, L, a.samples_per_batch, a.batches, np.random.default_rng(a.seed + 1 + it))
        t_hpc += cur["hpc_seconds"]
        occ = np.array(cur["occupations"])
        history.append({"iteration": it, "kind": "recovery", "energy": cur["energy"], "dims": cur["dims"],
                        "usable_shots": len(rec_shots)})
        say(f"[sqd] iteration {it}: E = {cur['energy']:.8f} (rel. err {(cur['energy'] - e0) / abs(e0):.2e}), "
            f"subspace dims {cur['dims']}")
    stages.add(H.StageRecord("config-recovery", "cpu (Python, 1 core)", "per-sample symmetry repair driven by the "
                             "previous eigenvector's occupations", t_cpu,
                             f"{a.iterations} rounds x {len(shots)} shots"))
    stages.add(H.StageRecord("hpc-diagonalisation", f"cpu (QUMO Mojo, {os.cpu_count()} threads)",
                             "projected-H build + restarted Lanczos on up to "
                             f"{max(max(h['dims']) for h in history)} determinants", t_hpc,
                             "qumo_py.subspace_ground_state",
                             notes=["no GPU eigensolver in QUMO yet; IBM + RIKEN ran this stage on Fugaku CPUs"]))
    e_final = min(h["energy"] for h in history[1:]) if len(history) > 1 else base["energy"]
    rel = (e_final - e0) / abs(e0)
    dim_max = max(max(h["dims"]) for h in history)
    log.check("recovery puts every shot in the (Na, Nb) sector", recovered_all_in,
              f"{len(shots)} shots, raw in-sector {frac_in:.3f}")
    log.check("every sqd energy is variational", all(h["energy"] >= e0 - 1e-9 for h in history),
              f"min {min(h['energy'] for h in history)!r} exact {e0!r}")
    log.within(f"final sqd relative energy error (band {a.band:g})", rel, 0.0, a.band)
    log.within(f"subspace below {a.max_subspace_fraction:g} of the sector", dim_max / len(full), 0.0,
               a.max_subspace_fraction)
    log.within("recovery not worse than postselection (1e-2 rel. slack)",
               (e_final - base["energy"]) / abs(e0), -1.0, 1e-2)

    twin_res = None
    if twin is not None:
        ts = shots_list(twin["counts"])
        tr = sqd_round(q, obs, ts, L, a.samples_per_batch, a.batches, np.random.default_rng(a.seed + 99))
        twin_res = {"energy": tr["energy"], "relative_error": (tr["energy"] - e0) / abs(e0), "dims": tr["dims"],
                    "device": twin["device"], "methods": twin["methods"]}
        log.check("ideal twin energy is variational", tr["energy"] >= e0 - 1e-9,
                  f"E_twin={tr['energy']:.8f} on {twin['device']}")
    wall = time.perf_counter() - t_start
    print(f"[time] wall {wall:.1f} s")
    results = {"exact_energy": e0, "scipy_energy": e_scipy, "sector_dimension": len(full),
               "raw_in_sector_fraction": frac_in, "history": history, "final_energy": e_final,
               "final_relative_error": rel, "max_subspace_dimension": dim_max, "twin": twin_res,
               "heft": {"device_of": sch["device_of"], "notes": sch["notes"]}}
    report = {"schema": "qumo.hetero_sqd.v1", "args": vars(a), "environment": H.environment(q), "gpu_info": gpu,
              "results": results, "qpu_jobs": gate.summary(), "wall_seconds": round(wall, 2), "checks": log.records}
    path = H.write_report(Path(a.out) / f"seed{a.seed}", report, stages, "SQD / SKQD of a Hubbard chain")
    say(f"[report] {path}")
    report["_all_passed"] = log.all_passed()
    log.finish(exit_on_failure=(__name__ == "__main__"))
    return report


if __name__ == "__main__":
    run()
