#!/usr/bin/env python3
"""Warm-started QAOA with escalation: CPU warm start, GPU adjoint training, superconducting sampling, trapped-ion verification, CPU polish.

Reproduces: Warm-starting quantum optimization (Egger, Marecek, Woerner, Quantum 5, 479 (2021)) - WS-QAOA for MaxCut from a Goemans-Williamson cut, with a two-QPU escalation step
Source: https://quantum-journal.org/papers/q-2021-06-17-479/
Source: https://arxiv.org/abs/2009.10095
Source: https://dl.acm.org/doi/10.1145/227683.227684

Pipeline (each device does what the others cannot):
  CPU  Goemans-Williamson: the MaxCut SDP solved by a Burer-Monteiro low-rank
       factorisation (projected gradient ascent on unit vectors, restarts),
       random-hyperplane rounding; gives the SDP upper bound and the cut that
       warm-starts QAOA (regularised: c_i in {eps, 1 - eps}).
  GPU  exact training of depth-p WS-QAOA (initial state prod_i RY(theta_i)|0>,
       mixer RY(theta_i) RZ(-2 beta) RY(-theta_i)) with adjoint gradients on
       the GPU state vector (qumo_py.value_and_gradient(device="gpu"): the
       doubled-register adjoint, fp32; CPU fp64 adjoint when no GPU is
       usable or below the threshold with --gpu-device auto).
  QPU  superconducting emulator (grid, noisy): many shots of the trained
       circuit - cheap, fast shots, but SWAPs on the grid add noise.
  QPU  trapped-ion emulator (all-to-all, lower error rates): few shots of the
       same circuit to verify the superconducting top-k candidates and to
       estimate <cut> closer to the ideal value (escalation: expensive,
       slow, high-fidelity shots only where they matter).
  CPU  one-flip local search from the confirmed candidates.

Checks (graph small enough to brute-force): SDP bound >= max cut; GW
rounding >= 0.878 x SDP bound; GPU gradient = CPU fp64 gradient (1e-4);
WS-QAOA ideal ratio >= standard QAOA ratio at the same depth and >= the
warm-start state's ratio; best superconducting sample >= GW cut; the ion
estimate of <cut> is no further from the ideal than the superconducting one
(3 sigma slack); the final cut equals the brute-force optimum; nothing
submitted remotely.

Run:  bench/.venv/bin/python examples/heterogeneous/qaoa_escalation/ws_qaoa_escalation.py
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 180, "requires": ["qumo_py", "scipy"], "args": ["--seed", "7"]}
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import hetero_common as H  # noqa: E402

SC_QPU = "local:qumo-sim:grid:3x4/sc_cz+noise:superconducting"
ION_QPU = "local:qumo-sim:all:16/ion_zz+noise:trapped_ion"


# --------------------------------------------------------------------------- graph + classical (CPU)
def random_graph(n: int, seed: int, density: float = 0.3) -> list[tuple[int, int, float]]:
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                edges.append((i, j, float(np.round(rng.uniform(0.5, 1.5), 3))))
    for i in range(n - 1):  # connected
        if not any((a == i and b == i + 1) or (a == i + 1 and b == i) for a, b, _ in edges):
            edges.append((i, i + 1, 1.0))
    return sorted(edges)


def cut_value(bits: np.ndarray, edges) -> np.ndarray:
    """bits: (..., n) 0/1 -> cut weight."""
    tot = 0.0
    for i, j, w in edges:
        tot = tot + w * (bits[..., i] != bits[..., j])
    return tot


def brute_force(n: int, edges) -> tuple[float, int]:
    xs = np.arange(1 << n, dtype=np.int64)
    bits = ((xs[:, None] >> np.arange(n)) & 1).astype(np.int8)
    cuts = cut_value(bits, edges)
    k = int(np.argmax(cuts))
    return float(cuts[k]), k


def goemans_williamson(n: int, edges, seed: int, restarts: int = 5, iters: int = 4000,
                       hyperplanes: int = 10) -> dict:
    """Burer-Monteiro: max sum w (1 - v_i.v_j)/2 over unit v_i in R^r; then hyperplane rounding."""
    rng = np.random.default_rng(seed)
    r = int(math.ceil(math.sqrt(2 * n))) + 1
    W = np.zeros((n, n))
    for i, j, w in edges:
        W[i, j] = W[j, i] = w
    best = None
    for _ in range(restarts):
        V = rng.normal(size=(n, r))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        step = 1.0 / max(1.0, np.abs(W).sum(axis=1).max())
        for _ in range(iters):
            G = -W @ V / 2  # gradient of the objective w.r.t. V (ascent)
            G -= np.sum(G * V, axis=1, keepdims=True) * V  # tangent projection
            V = V + step * G
            V /= np.linalg.norm(V, axis=1, keepdims=True)
        val = float(np.sum(np.triu(W, 1) * (1 - V @ V.T) / 2))
        if best is None or val > best[0]:
            best = (val, V)
    sdp, V = best
    planes = rng.normal(size=(hyperplanes, V.shape[1]))
    bits = (V @ planes.T > 0).T.astype(np.int8)  # (hyperplanes, n)
    cuts = cut_value(bits, edges)
    k = int(np.argmax(cuts))
    return {"sdp_value": sdp, "rank": r, "cut": float(cuts[k]), "bits": bits[k], "mean_rounded_cut": float(cuts.mean())}


def local_search(bits: np.ndarray, edges) -> tuple[np.ndarray, float]:
    b = bits.copy()
    cur = float(cut_value(b, edges))
    improved = True
    while improved:
        improved = False
        for i in range(len(b)):
            b[i] ^= 1
            v = float(cut_value(b, edges))
            if v > cur + 1e-12:
                cur, improved = v, True
            else:
                b[i] ^= 1
    return b, cur


# --------------------------------------------------------------------------- circuits
def cost_observable(edges) -> str:
    """E = sum w/2 Z_i Z_j = W/2 - <cut>  (minimise E <=> maximise the cut)."""
    return " + ".join(f"{w / 2!r}*Z{i} Z{j}" for i, j, w in edges)


def qaoa_qasm(n: int, edges, p: int, thetas: np.ndarray | None, measure: bool = False) -> str:
    """WS-QAOA (thetas given) or standard QAOA (thetas None); layered gate order so the GPU
    adjoint batches each commuting layer into one reduction."""
    L = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    L += [f"input float gamma_{l};" for l in range(p)] + [f"input float beta_{l};" for l in range(p)]
    L += [f"qubit[{n}] q;"] + ([f"bit[{n}] c;"] if measure else [])
    for i in range(n):
        L.append(f"ry({float(thetas[i])!r}) q[{i}];" if thetas is not None else f"h q[{i}];")
    for l in range(p):
        for i, j, w in edges:
            L.append(f"rzz({w!r}*gamma_{l}) q[{i}], q[{j}];")
        if thetas is None:
            for i in range(n):
                L.append(f"rx(2*beta_{l}) q[{i}];")
        else:
            for i in range(n):
                L.append(f"ry({-float(thetas[i])!r}) q[{i}];")
            for i in range(n):
                L.append(f"rz(-2*beta_{l}) q[{i}];")
            for i in range(n):
                L.append(f"ry({float(thetas[i])!r}) q[{i}];")
    if measure:
        for i in range(n):
            L.append(f"c[{i}] = measure q[{i}];")
    return "\n".join(L) + "\n"


def bind_qasm(text: str, values: list[float], p: int) -> str:
    names = [f"gamma_{l}" for l in range(p)] + [f"beta_{l}" for l in range(p)]
    out = []
    for line in text.splitlines():
        if line.startswith("input float"):
            continue
        for nm, v in sorted(zip(names, values), key=lambda t: -len(t[0])):
            line = line.replace(nm, f"({v!r})")
        out.append(line)
    return "\n".join(out) + "\n"


def train(q, circ, obs: str, p: int, device: str, starts: list[list[float]], maxiter: int) -> dict:
    import scipy.optimize as so

    calls = {"n": 0, "methods": set(), "notes": set()}

    def fg(x):
        v, g, m, note = q.value_and_gradient(circ, obs, [float(t) for t in x], device=device)
        calls["n"] += 1
        calls["methods"].add(str(m))
        if note:
            calls["notes"].add(str(note))
        return float(v), np.array(g, dtype=float)

    # fp32 GPU gradients carry ~1e-6 relative noise: a tighter tolerance only burns line searches
    gtol, ftol = (1e-4, 1e-7) if device == "gpu" else (1e-6, 2.2e-9)
    best = None
    for x0 in starts:
        r = so.minimize(fg, np.array(x0, dtype=float), jac=True, method="L-BFGS-B",
                        options={"maxiter": maxiter, "gtol": gtol, "ftol": ftol})
        if best is None or r.fun < best.fun:
            best = r
    return {"x": [float(t) for t in best.x], "energy": float(best.fun), "evaluations": calls["n"],
            "methods": sorted(calls["methods"]), "notes": sorted(calls["notes"])}


def sample_stats(counts: dict[int, int], n: int, edges) -> dict:
    keys = np.array(list(counts), dtype=np.int64)
    w = np.array([counts[k] for k in counts], dtype=float)
    bits = ((keys[:, None] >> np.arange(n)) & 1).astype(np.int8)
    cuts = cut_value(bits, edges)
    tot = w.sum()
    mean = float(np.dot(w, cuts) / tot)
    var = float(np.dot(w, (cuts - mean) ** 2) / tot)
    return {"mean_cut": mean, "stderr": math.sqrt(var / tot), "best_cut": float(cuts.max()),
            "keys": keys, "cuts": cuts, "weights": w, "bits": bits}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--depth", type=int, default=1, help="QAOA depth p")
    ap.add_argument("--eps", type=float, default=0.25, help="warm-start regularisation")
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--sc-shots", type=int, default=4000)
    ap.add_argument("--ion-shots", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--sc-qpu", default=None, help=f"default {SC_QPU} (grid sized to the graph)")
    ap.add_argument("--ion-qpu", default=ION_QPU)
    ap.add_argument("--gpu-device", choices=["auto", "gpu", "cpu"], default="gpu",
                    help="training device: gpu (default, CPU fallback when unusable), auto (threshold), cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(H.REPO / "build" / "hetero" / "qaoa_escalation"))
    ap.add_argument("--quiet", action="store_true")
    H.add_safety_args(ap)
    return ap.parse_args(argv)


def default_sc(n: int) -> str:
    rows = 3 if n <= 12 else 4
    cols = math.ceil(n / rows)
    return f"local:qumo-sim:grid:{rows}x{cols}/sc_cz+noise:superconducting"


def run(argv=None, runner=None, edges_override=None) -> dict:
    a = parse_args(argv)
    say = (lambda *_: None) if a.quiet else (lambda m: print(m, flush=True))
    from _checks import CheckLog

    t_start = time.perf_counter()
    os.environ.setdefault("QUMO_HW_NOISE_WORKERS", str(max(1, min(8, (os.cpu_count() or 2) - 1))))
    q = H.load_qumo()
    H.require_bindings(q, ["value_and_gradient", "expectation_device", "schedule"])
    sc_qpu = a.sc_qpu or default_sc(a.nodes)
    H.check_targets([sc_qpu, a.ion_qpu], a.allow_remote)
    gpu = H.gpu_status(q)
    stages = H.StageLog(say)
    log = CheckLog("heterogeneous/ws_qaoa_escalation")
    edges = edges_override or random_graph(a.nodes, a.seed)
    n = 1 + max(max(i, j) for i, j, _ in edges)
    W = sum(w for _, _, w in edges)
    opt, opt_key = brute_force(n, edges)
    say(f"[problem] weighted MaxCut, {n} nodes, {len(edges)} edges, total weight {W:.3f}, max cut {opt:.3f} "
        f"(brute force)")

    # ---- CPU: Goemans-Williamson warm start
    with H.Timer() as tm:
        gw = goemans_williamson(n, edges, a.seed)
    stages.add(H.StageRecord("gw-warm-start", "cpu (NumPy)", "SDP relaxation (Burer-Monteiro) + hyperplane "
                             "rounding: a classical bound and a good cut", tm.seconds,
                             f"rank {gw['rank']}, SDP {gw['sdp_value']:.4f}, GW cut {gw['cut']:.4f}"))
    log.within("sdp value is an upper bound on the max cut", gw["sdp_value"], opt - 1e-6, W)
    log.within("gw rounding reaches 0.878 x sdp", gw["cut"], 0.878 * gw["sdp_value"], opt + 1e-9)
    c = np.where(gw["bits"] == 1, 1 - a.eps, a.eps)
    thetas = 2 * np.arcsin(np.sqrt(c))
    obs = cost_observable(edges)

    # ---- GPU: exact training with adjoint gradients
    p = a.depth
    ws = q.load_qasm(qaoa_qasm(n, edges, p, thetas))
    st = q.load_qasm(qaoa_qasm(n, edges, p, None))
    dev, why = H.place_gpu(gpu, n, a.gpu_device)
    x_chk = [0.2] * p + [-0.1] * p
    _, g_cpu, _, _ = q.value_and_gradient(ws, obs, x_chk, device="cpu")
    _, g_dev, m_dev, note_dev = q.value_and_gradient(ws, obs, x_chk, device=dev)
    log.close(f"{dev} adjoint gradient equals cpu fp64 adjoint", float(np.max(np.abs(np.array(g_dev) - g_cpu))), 0.0,
              1e-4 if "gpu" in str(m_dev) else 1e-12)
    starts = [[0.0] * (2 * p), [0.3] * p + [-0.2] * p, [-0.3] * p + [0.2] * p, [0.6] * p + [0.3] * p]
    with H.Timer() as tm:
        tr_ws = train(q, ws, obs, p, dev, starts, a.maxiter)
        tr_st = train(q, st, obs, p, dev, [[0.3] * p + [-0.2] * p, [-0.3] * p + [0.3] * p, [0.6] * p + [0.4] * p,
                                           [0.2] * p + [0.6] * p], a.maxiter)
    stages.add(H.StageRecord("exact-training", dev, "exact energies + adjoint gradients of the full state for an "
                             "optimiser (no shot noise, no QPU time)", tm.seconds, ",".join(tr_ws["methods"]),
                             notes=[x for x in [why, note_dev] if x] +
                                   [f"{tr_ws['evaluations'] + tr_st['evaluations']} value+gradient calls"]))
    e_init = float(q.expectation(ws.bind([0.0] * (2 * p)), obs))  # gamma = beta = 0: the warm-start state itself
    ratio = lambda e: (W / 2 - e) / opt  # noqa: E731
    r_ws, r_st, r_init = ratio(tr_ws["energy"]), ratio(tr_st["energy"]), ratio(e_init)
    say(f"[train] ideal ratios: WS-QAOA {r_ws:.4f}, standard QAOA {r_st:.4f}, warm-start state {r_init:.4f}")
    log.within(f"ws-qaoa ideal ratio >= standard qaoa ratio (p={p})", r_ws, r_st - 1e-9, 1.0 + 1e-9)
    log.within("ws-qaoa ideal ratio >= warm-start state ratio", r_ws, r_init - 1e-9, 1.0 + 1e-9)
    xs = tr_ws["x"]
    payload = bind_qasm(qaoa_qasm(n, edges, p, thetas, measure=True), xs, p)

    # ---- orchestration: QPU jobs pinned, exact verification placed by HEFT
    devices = [{"name": "cpu0", "kind": "cpu", "max_qubits": 30}, {"name": "gpu0", "kind": "gpu", "max_qubits": 30},
               H.qpu_device("sc-qpu", sc_qpu, 133), H.qpu_device("ion-qpu", a.ion_qpu, 56)]
    pc = q.load_qasm(payload)
    sch = H.heft(q, [{"name": "sc_sampling", "circuit": pc, "shots": a.sc_shots, "device": "sc-qpu"},
                     {"name": "ion_verification", "circuit": pc, "shots": a.ion_shots, "device": "ion-qpu"},
                     {"name": "exact_check", "circuit": ws.bind(xs), "kind": "estimate", "observable": obs}],
                 devices, say)

    # ---- QPU stages
    gate = H.QpuGate(a.allow_remote, a.submit, a.yes, a.max_cost, runner=runner, say=say)
    with H.Timer() as tm:
        rec_sc = gate.run(payload, sc_qpu, a.sc_shots, a.seed, "sc_sampling")
    stages.add(H.StageRecord("qpu-sampling", f"qpu:{sc_qpu}", "many cheap shots to find candidate cuts",
                             tm.seconds, f"{a.sc_shots} shots, 2q {rec_sc.get('compiled', {}).get('two_qubit_gates')}",
                             emulated=not H.is_remote(sc_qpu)))
    with H.Timer() as tm:
        rec_ion = gate.run(payload, a.ion_qpu, a.ion_shots, a.seed + 1, "ion_verification")
    stages.add(H.StageRecord("qpu-verification", f"qpu:{a.ion_qpu}", "few high-fidelity shots (no SWAPs) to "
                             "confirm candidates", tm.seconds,
                             f"{a.ion_shots} shots, 2q {rec_ion.get('compiled', {}).get('two_qubit_gates')}",
                             emulated=not H.is_remote(a.ion_qpu)))
    log.check("no remote submission", gate.nothing_remote_submitted(), f"{len(gate.jobs)} QPU job(s)")
    results = {"nodes": n, "edges": len(edges), "max_cut": opt, "gw": {k: v for k, v in gw.items() if k != "bits"},
               "ratios": {"ws_qaoa": r_ws, "standard_qaoa": r_st, "warm_start_state": r_init},
               "params": xs, "training_device": dev, "heft": sch["device_of"]}
    if rec_sc["counts"] is None or rec_ion["counts"] is None:
        log.check("qpu returned samples", False, "dry run or refused")
        log.finish(exit_on_failure=(__name__ == "__main__"))
        return {"checks": log.records, "results": results}
    s_sc = sample_stats(H.counts_to_ints(rec_sc["counts"]), n, edges)
    s_ion = sample_stats(H.counts_to_ints(rec_ion["counts"]), n, edges)
    ideal_mean = W / 2 - tr_ws["energy"]
    log.within("best superconducting sample >= gw cut", s_sc["best_cut"], gw["cut"] - 1e-9, opt + 1e-9)
    err_sc, err_ion = abs(s_sc["mean_cut"] - ideal_mean), abs(s_ion["mean_cut"] - ideal_mean)
    log.within("ion <cut> no further from ideal than superconducting (3 sigma)", err_ion, 0.0,
               err_sc + 3 * (s_sc["stderr"] + s_ion["stderr"]))
    say(f"[qpu] <cut>: ideal {ideal_mean:.4f}, superconducting {s_sc['mean_cut']:.4f} +- {s_sc['stderr']:.4f}, "
        f"ion {s_ion['mean_cut']:.4f} +- {s_ion['stderr']:.4f}")

    # ---- CPU: confirm top-k candidates with the ion samples, then local search
    with H.Timer() as tm:
        order = np.argsort(-s_sc["cuts"], kind="stable")
        seen, top = set(), []
        for k in order:
            key = int(s_sc["keys"][k])
            if key not in seen:
                seen.add(key)
                top.append(k)
            if len(top) >= a.top_k:
                break
        ion_set = {int(x): float(w) for x, w in zip(s_ion["keys"], s_ion["weights"])}
        confirmed = [int(s_sc["keys"][k]) for k in top if int(s_sc["keys"][k]) in ion_set
                     or (int(s_sc["keys"][k]) ^ ((1 << n) - 1)) in ion_set]
        seeds = confirmed or [int(s_sc["keys"][top[0]])]
        best_bits, best_cut = None, -1.0
        for key in seeds:
            b, v = local_search(((key >> np.arange(n)) & 1).astype(np.int8), edges)
            if v > best_cut:
                best_bits, best_cut = b, v
    stages.add(H.StageRecord("local-search", "cpu", "cheap exact polish of a few confirmed candidates", tm.seconds,
                             f"{len(confirmed)} of {len(top)} top-{a.top_k} superconducting candidates confirmed by "
                             "the ion samples"))
    log.close("final cut equals the brute-force optimum", best_cut, opt, 1e-9)
    results.update({"sc": {k: s_sc[k] for k in ("mean_cut", "stderr", "best_cut")},
                    "ion": {k: s_ion[k] for k in ("mean_cut", "stderr", "best_cut")},
                    "ideal_mean_cut": ideal_mean, "top_k": [float(s_sc["cuts"][k]) for k in top],
                    "confirmed": len(confirmed), "final_cut": best_cut, "final_ratio": best_cut / opt})
    wall = time.perf_counter() - t_start
    print(f"[time] wall {wall:.1f} s")
    report = {"schema": "qumo.hetero_ws_qaoa.v1", "args": vars(a), "environment": H.environment(q), "gpu_info": gpu,
              "results": results, "qpu_jobs": gate.summary(), "wall_seconds": round(wall, 2), "checks": log.records}
    path = H.write_report(Path(a.out) / f"seed{a.seed}", report, stages, "Warm-started QAOA with escalation")
    say(f"[report] {path}")
    report["_all_passed"] = log.all_passed()
    log.finish(exit_on_failure=(__name__ == "__main__"))
    return report


if __name__ == "__main__":
    run()
