#!/usr/bin/env python3
"""Circuit cutting across two QPU modalities: a dense fragment on a trapped-ion QPU, a wide shallow fragment on a superconducting QPU.

Reproduces: Qiskit addon cutting / CutQC - wire cutting with quasi-probability reconstruction, here with each fragment compiled for a different QPU modality
Source: https://github.com/Qiskit/qiskit-addon-cutting
Source: https://arxiv.org/abs/2012.02333
Source: https://arxiv.org/abs/1904.00102

The circuit (18 qubits) is a 6-qubit all-to-all block (15 RZZ couplings:
natural on an all-to-all trapped-ion register, many SWAPs on a grid) followed
by a 13-qubit shallow brickwork that shares one qubit with the block (natural
on a 2D grid, too wide for the 8-qubit ion register). No QPU here holds all
18 qubits.

What the cut finder picks (default seed): one wire cut on qubit 7, so the
ion fragment is the dense block plus two brickwork qubits (8 qubits, fills the
ion register; 17 native ZZ gates) and the grid fragment the other 11 qubits
(10 CZ). The cut wire leaves the grid fragment (measured there in the
Z / X / Y bases) and enters the ion fragment (prepared there in 0, 1, +, +i).
The reverse assignment does not fit (11 > 8 ion qubits); putting the dense
block on the grid would cost about 40 CZ with SWAPs (reported as an
alternative).

Pipeline (each device does what the others cannot):
  CPU  cut finder: enumerates wire cuts (<= 2), keeps two-fragment cuts that
       fit the devices, compiles every fragment for every target with QUMO
       and picks the fewest cuts, then the device assignment with the highest
       estimated success (compiled 2q / 1q counts x modality error rates).
  QPU  trapped-ion emulator (all-to-all, 8 qubits, ion noise preset): the
       dense fragment, one circuit per cut-measurement basis.
  QPU  superconducting emulator (4x4 grid, sc noise preset): the wide
       fragment, one circuit per cut-preparation state.
  GPU  full-width (18-qubit) noiseless reference of the uncut circuit and
       exact fragment tensors (fp32; CPU fallback). The 1e-9 identity check
       uses the CPU's fp64 fragment tensors.
  CPU  quasi-probability reconstruction (Peng et al. 2020): the 4^k-term
       tensor contraction (NumPy einsum; k = 1 here, so the contraction is
       tiny - at many cuts this is the GPU-heavy part of CutQC, which QUMO
       does not implement on the GPU).

Checks: the cut finder chooses 1 cut and puts the dense fragment on the ion
QPU; noiseless reconstruction = uncut exact value (1e-9, fp64); GPU uncut
reference and GPU fragment tensors = CPU fp64 (1e-4, fp32); noisy
reconstruction (20 000 shots per fragment circuit) within
5 sigma + 0.03 x sum|c_t| of the exact value (measured errors 0.006-0.04 on
seeds 7-10, value 0.5-1.3); nothing submitted remotely.

Run:  bench/.venv/bin/python examples/heterogeneous/cutting/cut_two_modalities.py
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 180, "requires": ["qumo_py"], "args": ["--seed", "7"]}
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

ION_QPU = "local:qumo-sim:all:8/ion_zz+noise:trapped_ion"
SC_QPU = "local:qumo-sim:grid:4x4/sc_cz+noise:superconducting"

# cut-wire decomposition: identity channel = 1/2 sum_P Tr(P rho) P, P = I, X, Y, Z;
# measurement side: basis of each cut (I and Z both from the Z basis); preparation side:
# eigenstates 0, 1, +, +i and  g(P) = M f  with f = <B> for each prepared state
PAULIS = "IXYZ"
PREPS = ("0", "1", "+", "+i")
M_PREP = np.array([[1, 1, 0, 0],  # I = |0><0| + |1><1|
                   [-1, -1, 2, 0],  # X = 2|+><+| - I
                   [-1, -1, 0, 2],  # Y = 2|+i><+i| - I
                   [1, -1, 0, 0]], dtype=float)  # Z = |0><0| - |1><1|


# --------------------------------------------------------------------------- circuit
def build_circuit(seed: int, dense: int = 6, wide: int = 13) -> tuple[int, list[tuple]]:
    """Gate list (name, qubits, angle). Dense block on 0..dense-1, brickwork on dense-1..dense+wide-2."""
    rng = np.random.default_rng(seed)
    g: list[tuple] = []
    for qb in range(dense):
        g.append(("ry", (qb,), float(rng.uniform(0.2, 0.7))))
    for a, b in itertools.combinations(range(dense), 2):
        g.append(("rzz", (a, b), float(rng.uniform(0.2, 0.9))))
    for qb in range(dense):
        g.append(("rx", (qb,), float(rng.uniform(0.1, 0.4))))
    lo, hi = dense - 1, dense + wide - 1
    for layer in range(2):
        for qb in range(lo, hi):
            g.append(("ry", (qb,), float(rng.uniform(0.2, 0.7))))
        for qb in range(lo + layer, hi - 1, 2):
            g.append(("cz", (qb, qb + 1), 0.0))
    for qb in range(lo, hi):
        g.append(("ry", (qb,), float(rng.uniform(0.1, 0.6))))
    return hi, g


def observable_terms(n: int, dense: int) -> list[tuple[float, dict[int, str]]]:
    """Terms that straddle the two blocks (plus one local term on each side)."""
    return [(1.0, {0: "Z", n - 1: "Z"}), (0.5, {2: "Z", 3: "Z", dense: "Z"}), (0.5, {3: "X", 10: "X"}),
            (0.7, {1: "Z", dense + 2: "Z", dense + 3: "Z"}), (0.3, {4: "Z"}), (0.4, {12: "Z", 13: "Z"})]


def pauli_text(term: dict[int, str], qmap: dict[int, int] | None = None) -> str:
    if not term:
        return "I0"
    return " ".join(f"{p}{qmap[qb] if qmap else qb}" for qb, p in sorted(term.items()))


def qasm_of(n: int, gates: list[tuple], qmap: dict[int, int], preps: dict[int, str] | None = None,
            bases: dict[int, str] | None = None, measure: bool = True) -> str:
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n}] q;"] + ([f"bit[{n}] c;"] if measure else [])
    for qb, s in (preps or {}).items():
        lq = qmap[qb]
        if s == "1":
            lines.append(f"x q[{lq}];")
        elif s == "+":
            lines.append(f"h q[{lq}];")
        elif s == "+i":
            lines.append(f"h q[{lq}];")
            lines.append(f"s q[{lq}];")
    for name, qs, ang in gates:
        args = ", ".join(f"q[{qmap[x]}]" for x in qs)
        lines.append(f"{name}({ang!r}) {args};" if name in ("rx", "ry", "rz", "rzz") else f"{name} {args};")
    for qb, b in (bases or {}).items():  # measure P: rotate its eigenbasis to Z
        lq = qmap[qb]
        if b == "X":
            lines.append(f"h q[{lq}];")
        elif b == "Y":
            lines.append(f"sdg q[{lq}];")
            lines.append(f"h q[{lq}];")
    if measure:
        for i in range(n):
            lines.append(f"c[{i}] = measure q[{i}];")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- cut finder (CPU)
def fragments_for(n: int, gates: list[tuple], cuts: tuple) -> dict | None:
    """Two-fragment decomposition for a set of wire cuts (qubit, gap index), or None."""
    seg = [0] * n
    on_q = [0] * n
    cutset = set(cuts)
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = []  # segment of each gate operand
    for name, qs, _ in gates:
        ids = []
        for qb in qs:
            if (qb, on_q[qb]) in cutset:
                seg[qb] += 1
            on_q[qb] += 1
            ids.append((qb, seg[qb]))
        for x in ids[1:]:
            pa, pb = find(ids[0]), find(x)
            if pa != pb:
                parent[pa] = pb
        for x in ids:
            find(x)
        owner.append(ids)
    roots = sorted({find(x) for x in parent})
    if len(roots) != 2:
        return None
    frag = {r: k for k, r in enumerate(roots)}
    wires: list[set] = [set(), set()]
    for x in parent:
        wires[frag[find(x)]].add(x[0])
    # every cut: segment s of its qubit in fragment A, s + 1 in B; one direction only
    up = None
    for qb, gap in cuts:
        s_before = sum(1 for (cq, cg) in cuts if cq == qb and cg <= gap) - 1
        a, b = (qb, s_before), (qb, s_before + 1)
        if a not in parent or b not in parent:
            return None
        fa, fb = frag[find(a)], frag[find(b)]
        if fa == fb or (up is not None and up != fa):
            return None
        up = fa
    down = 1 - up
    gate_frag = [frag[find(ids[0])] for ids in owner]
    final = {qb: frag[find((qb, seg[qb]))] for qb in range(n) if (qb, seg[qb]) in parent}
    return {"up": up, "down": down, "qubits": [sorted(w) for w in wires], "gate_frag": gate_frag,
            "final_frag": final, "cuts": list(cuts)}


def candidate_cuts(n: int, gates: list[tuple]) -> list[tuple[int, int]]:
    count = [0] * n
    for _, qs, _ in gates:
        for qb in qs:
            count[qb] += 1
    return [(qb, k) for qb in range(n) for k in range(1, count[qb])]


def fragment_gates(gates, fr, which):
    return [g for g, f in zip(gates, fr["gate_frag"]) if f == which]


def estimate_fragment(q, n, gates, fr, which, spec) -> dict:
    """Compile a fragment for a target with QUMO and estimate its success probability."""
    from qumo_hardware.compile import compile_for_target

    qs = fr["qubits"][which]
    if len(qs) > spec.num_qubits:
        return {"fits": False}
    qmap = {qb: k for k, qb in enumerate(qs)}
    text = qasm_of(len(qs), fragment_gates(gates, fr, which), qmap)
    comp = compile_for_target(text, spec, level=2, seed=1)
    return {"fits": True, "two_qubit_gates": comp.two_qubit_gates, "success": float(comp.estimated_success()),
            "depth": comp.depth}


def find_cut(q, n, gates, targets: dict, max_cuts: int, say) -> dict:
    """Fewest cuts, then the best (fragment -> target) assignment by estimated success."""
    from qumo_hardware.registry import get_provider

    specs = {}
    for name, t in targets.items():
        prov, _, be = t.partition(":")
        specs[name] = get_provider(prov).target(be)
    cands = candidate_cuts(n, gates)
    examined = 0
    for k in range(1, max_cuts + 1):
        best = None
        for cuts in itertools.combinations(cands, k):
            fr = fragments_for(n, gates, cuts)
            examined += 1
            if fr is None:
                continue
            for perm in itertools.permutations(specs, 2):
                est = [estimate_fragment(q, n, gates, fr, w, specs[perm[w]]) for w in range(2)]
                if not all(e["fits"] for e in est):
                    continue
                score = est[0]["success"] * est[1]["success"]
                if best is None or score > best["score"]:
                    best = {"fr": fr, "assign": {w: perm[w] for w in range(2)}, "est": est, "score": score}
        if best is not None:
            alts = []
            for perm in itertools.permutations(specs, 2):
                e = [estimate_fragment(q, n, gates, best["fr"], w, specs[perm[w]]) for w in range(2)]
                alts.append({"assignment": {f"fragment{w}": perm[w] for w in range(2)},
                             "fits": all(x["fits"] for x in e),
                             "success": (e[0]["success"] * e[1]["success"]) if all(x["fits"] for x in e) else 0.0,
                             "two_qubit_gates": [x.get("two_qubit_gates") for x in e]})
            best["alternatives"] = alts
            best["examined"] = examined
            best["k"] = k
            best["specs"] = {nme: {"num_qubits": s.num_qubits, "all_to_all": s.all_to_all} for nme, s in specs.items()}
            return best
    raise SystemExit(f"no two-fragment cut with <= {max_cuts} wire cuts fits the devices")


# --------------------------------------------------------------------------- fragments + reconstruction
def split_terms(terms, fr):
    out = []
    for c, t in terms:
        A = {qb: p for qb, p in t.items() if fr["final_frag"][qb] == fr["up"]}
        B = {qb: p for qb, p in t.items() if fr["final_frag"][qb] == fr["down"]}
        out.append((c, A, B))
    return out


def qwc_settings(parts: list[dict]) -> tuple[list[dict], list[int]]:
    """Greedy qubit-wise-commuting grouping -> (settings, setting index of each part)."""
    settings: list[dict] = []
    idx = []
    for p in parts:
        for k, s in enumerate(settings):
            if all(s.get(qb, b) == b for qb, b in p.items()):
                s.update(p)
                idx.append(k)
                break
        else:
            settings.append(dict(p))
            idx.append(len(settings) - 1)
    return settings, idx


def parity_mean(counts: dict[int, int], local_qubits: list[int]) -> tuple[float, float]:
    tot = sum(counts.values())
    s = 0.0
    for x, v in counts.items():
        par = sum((x >> lq) & 1 for lq in local_qubits) & 1
        s += (-1.0 if par else 1.0) * v
    m = s / tot
    return m, math.sqrt(max(1e-12, 1 - m * m) / tot)


def contract(E1: np.ndarray, F2: np.ndarray, coeffs: np.ndarray, k: int) -> float:
    """sum_t c_t (1/2)^k sum_P E1[t, P] (M^{(x)k} f)[t, P]  (P over {I,X,Y,Z}^k)."""
    G2 = F2.reshape(len(coeffs), *([4] * k))
    for ax in range(k):
        G2 = np.moveaxis(np.tensordot(G2, M_PREP, axes=([1 + ax], [1])), -1, 1 + ax)
    E = E1.reshape(len(coeffs), *([4] * k))
    letters = "abcdefgh"[:k]
    val = np.einsum(f"t{letters},t{letters}->t", E, G2) * (0.5 ** k)
    return float(np.dot(coeffs, val))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ion-qpu", default=ION_QPU)
    ap.add_argument("--sc-qpu", default=SC_QPU)
    ap.add_argument("--max-cuts", type=int, default=2)
    ap.add_argument("--ion-shots", type=int, default=20000)
    ap.add_argument("--sc-shots", type=int, default=20000)
    ap.add_argument("--band", type=float, default=0.03, help="noise bias allowance, times sum |c_t|")
    ap.add_argument("--gpu-device", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(H.REPO / "build" / "hetero" / "cutting"))
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
    H.require_bindings(q, ["expectations_device", "schedule"])
    H.ensure_hardware_plugin()
    H.check_targets([a.ion_qpu, a.sc_qpu], a.allow_remote)
    gpu = H.gpu_status(q)
    stages = H.StageLog(say)
    log = CheckLog("heterogeneous/cut_two_modalities")
    n, gates = build_circuit(a.seed)
    terms = observable_terms(n, 6)
    coeffs = np.array([c for c, _ in terms])
    obs_full = " + ".join(f"{c!r}*{pauli_text(t)}" for c, t in terms)
    say(f"[problem] {n}-qubit circuit, {len(gates)} gates; O = {obs_full}")

    # ---- CPU: cut finder
    targets = {"ion": a.ion_qpu, "sc": a.sc_qpu}
    with H.Timer() as tm:
        best = find_cut(q, n, gates, targets, a.max_cuts, say)
    fr, k = best["fr"], best["k"]
    up, down = fr["up"], fr["down"]
    dev_up, dev_down = best["assign"][up], best["assign"][down]
    stages.add(H.StageRecord("cut-finding", "cpu", "combinatorial search + per-target compilation (QUMO)",
                             tm.seconds, f"{best['examined']} cut sets, k={k}, cuts {fr['cuts']}",
                             notes=[f"fragment {w}: {len(fr['qubits'][w])} qubits -> {best['assign'][w]} "
                                    f"(2q {best['est'][w]['two_qubit_gates']}, est. success "
                                    f"{best['est'][w]['success']:.3f})" for w in (0, 1)]))
    for alt in best["alternatives"]:
        say(f"[cut] alternative {alt['assignment']}: fits={alt['fits']} success={alt['success']:.3f} "
            f"2q={alt['two_qubit_gates']}")
    dense_frag = max((0, 1), key=lambda w: sum(1 for g, f in zip(gates, fr["gate_frag"]) if f == w and g[0] == "rzz"))
    log.check("cut finder uses one wire cut", k == 1, f"cuts={fr['cuts']}")
    log.check("cut finder puts the dense fragment on the ion qpu", best["assign"][dense_frag] == "ion",
              f"assignment {best['assign']}")

    cut_qubits = [qb for qb, _ in fr["cuts"]]
    qs_up, qs_down = fr["qubits"][up], fr["qubits"][down]
    map_up = {qb: i for i, qb in enumerate(qs_up)}
    map_down = {qb: i for i, qb in enumerate(qs_down)}
    g_up, g_down = fragment_gates(gates, fr, up), fragment_gates(gates, fr, down)
    parts = split_terms(terms, fr)
    ptuples = list(itertools.product(range(4), repeat=k))
    prep_tuples = list(itertools.product(range(4), repeat=k))

    # ---- exact reference: full circuit (GPU at full width) and fp64 fragment tensors (CPU)
    full_circ = q.load_qasm(qasm_of(n, gates, {i: i for i in range(n)}, measure=False))
    with H.Timer() as tm:
        (v_cpu,), _, _ = q.expectations_device(full_circ, [obs_full], "cpu")
    t_cpu_full = tm.seconds
    dev, why = H.place_gpu(gpu, n, a.gpu_device)
    with H.Timer() as tm:
        (v_dev,), method, note = q.expectations_device(full_circ, [obs_full], dev)
    stages.add(H.StageRecord("uncut-reference", dev, f"full-width {n}-qubit noiseless state (no QPU here holds "
                             "the uncut circuit)", tm.seconds, str(method), notes=[x for x in (why, note) if x]))
    log.close(f"uncut reference on {dev} equals cpu fp64", v_dev, v_cpu, 1e-4 if dev == "gpu" else 1e-12)

    def exact_tensors(device: str):
        E1 = np.zeros((len(terms), 4 ** k))
        F2 = np.zeros((len(terms), 4 ** k))
        cu = q.load_qasm(qasm_of(len(qs_up), g_up, map_up, measure=False))
        obs_list = []
        for _, A, _ in parts:
            for pt in ptuples:
                t = dict(A)
                for cq, pi in zip(cut_qubits, pt):
                    if pi:
                        t[cq] = PAULIS[pi]
                obs_list.append(pauli_text(t, map_up))
        vals, m1, _ = q.expectations_device(cu, obs_list, device)
        E1[:] = np.array(vals).reshape(len(terms), 4 ** k)
        meth = {str(m1)}
        for j, pr in enumerate(prep_tuples):
            preps = {cq: PREPS[pi] for cq, pi in zip(cut_qubits, pr)}
            cd = q.load_qasm(qasm_of(len(qs_down), g_down, map_down, preps=preps, measure=False))
            vals, m2, _ = q.expectations_device(cd, [pauli_text(B, map_down) for _, _, B in parts], device)
            F2[:, j] = vals
            meth.add(str(m2))
        return E1, F2, meth

    with H.Timer() as tm:
        E1x, F2x, meth = exact_tensors("cpu")
        v_rec_exact = contract(E1x, F2x, coeffs, k)
    stages.add(H.StageRecord("exact-fragments", "cpu", "fp64 fragment tensors for the 1e-9 identity check",
                             tm.seconds, ",".join(sorted(meth))))
    log.close("noiseless reconstruction equals the uncut exact value", v_rec_exact, v_cpu, 1e-9)
    if dev == "gpu":
        with H.Timer() as tm:
            E1g, F2g, methg = exact_tensors("gpu")
            v_rec_gpu = contract(E1g, F2g, coeffs, k)
        stages.add(H.StageRecord("exact-fragments-gpu", "gpu", "batched exact fragment tensors (all observables "
                                 "of a fragment in one GPU reduction batch)", tm.seconds, ",".join(sorted(methg))))
        log.close("gpu fragment tensors reconstruct the exact value (fp32)", v_rec_gpu, v_cpu, 1e-4)

    # ---- orchestration: fragment variants pinned to their QPUs, exact jobs placed by HEFT
    set_up, idx_up = qwc_settings([A for _, A, _ in parts])
    set_down, idx_down = qwc_settings([B for _, _, B in parts])
    up_jobs, down_jobs = [], []
    for s, st in enumerate(set_up):
        for bases in itertools.product("ZXY", repeat=k):
            bb = dict(st)
            bb.update({cq: b for cq, b in zip(cut_qubits, bases)})
            up_jobs.append((s, bases, qasm_of(len(qs_up), g_up, map_up, bases=bb)))
    for s, st in enumerate(set_down):
        for pr in prep_tuples:
            preps = {cq: PREPS[pi] for cq, pi in zip(cut_qubits, pr)}
            down_jobs.append((s, pr, qasm_of(len(qs_down), g_down, map_down, preps=preps, bases=st)))
    devices = [{"name": "cpu0", "kind": "cpu", "max_qubits": 30}, {"name": "gpu0", "kind": "gpu", "max_qubits": 30},
               H.qpu_device("ion-qpu", targets["ion"], best["specs"]["ion"]["num_qubits"]),
               H.qpu_device("sc-qpu", targets["sc"], best["specs"]["sc"]["num_qubits"])]
    pin = {"ion": "ion-qpu", "sc": "sc-qpu"}
    hj = [{"name": f"up_{i}", "circuit": q.load_qasm(t), "shots": a.ion_shots if dev_up == "ion" else a.sc_shots,
           "device": pin[dev_up]} for i, (_, _, t) in enumerate(up_jobs)]
    hj += [{"name": f"down_{i}", "circuit": q.load_qasm(t), "shots": a.sc_shots if dev_down == "sc" else a.ion_shots,
            "device": pin[dev_down]} for i, (_, _, t) in enumerate(down_jobs)]
    hj.append({"name": "uncut_exact", "circuit": full_circ, "kind": "estimate", "observable": obs_full})
    sch = H.heft(q, hj, devices, say)

    # ---- QPU stages
    gate = H.QpuGate(a.allow_remote, a.submit, a.yes, a.max_cost, runner=runner, say=say)
    E1 = np.zeros((len(terms), 4 ** k))
    V1 = np.zeros((len(terms), 4 ** k))
    filled = np.zeros((len(terms), 4 ** k), dtype=bool)
    shots_up = a.ion_shots if dev_up == "ion" else a.sc_shots
    shots_down = a.sc_shots if dev_down == "sc" else a.ion_shots
    with H.Timer() as tm:
        for i, (s, bases, text) in enumerate(up_jobs):
            rec = gate.run(text, targets[dev_up], shots_up, a.seed + i, f"up_{i}")
            if rec["counts"] is None:
                continue
            cnt = H.counts_to_ints(rec["counts"])
            for ti, (_, A, _) in enumerate(parts):
                if idx_up[ti] != s:
                    continue
                for j, pt in enumerate(ptuples):
                    if not all(pi == 0 or PAULIS[pi] == b for pi, b in zip(pt, bases)):
                        continue
                    if filled[ti, j] and any(pi == 0 for pi in pt):
                        continue  # identity entries: first basis that provides them
                    lqs = [map_up[qb] for qb in A] + [map_up[cq] for cq, pi in zip(cut_qubits, pt) if pi]
                    E1[ti, j], V1[ti, j] = parity_mean(cnt, lqs)
                    filled[ti, j] = True
    role_up = ("all-to-all couplings: the dense block without SWAPs" if dev_up == "ion" else
               f"{len(qs_up)} qubits: wider than the ion register, nearest-neighbour on the grid")
    role_down = ("all-to-all couplings: the dense block without SWAPs" if dev_down == "ion" else
                 f"{len(qs_down)} qubits: wider than the ion register, nearest-neighbour on the grid")
    stages.add(H.StageRecord(f"qpu-fragment-{dev_up}", f"qpu:{targets[dev_up]}", role_up + " (cut measured here)", tm.seconds, f"{len(up_jobs)} circuits x {shots_up} shots",
                             emulated=not H.is_remote(targets[dev_up]),
                             notes=[f"2q gates {[j.get('compiled', {}).get('two_qubit_gates') for j in gate.jobs]}"]))
    n_up_jobs = len(gate.jobs)
    F2 = np.zeros((len(terms), 4 ** k))
    V2 = np.zeros((len(terms), 4 ** k))
    with H.Timer() as tm:
        for i, (s, pr, text) in enumerate(down_jobs):
            rec = gate.run(text, targets[dev_down], shots_down, a.seed + 1000 + i, f"down_{i}")
            if rec["counts"] is None:
                continue
            cnt = H.counts_to_ints(rec["counts"])
            j = prep_tuples.index(pr)
            for ti, (_, _, B) in enumerate(parts):
                if idx_down[ti] == s:
                    F2[ti, j], V2[ti, j] = parity_mean(cnt, [map_down[qb] for qb in B])
    stages.add(H.StageRecord(f"qpu-fragment-{dev_down}", f"qpu:{targets[dev_down]}", role_down + " (cut prepared "
                             "here)", tm.seconds,
                             f"{len(down_jobs)} circuits x {shots_down} shots", emulated=not H.is_remote(targets[dev_down]),
                             notes=[f"2q gates {[j.get('compiled', {}).get('two_qubit_gates') for j in gate.jobs[n_up_jobs:]]}"]))
    log.check("no remote submission", gate.nothing_remote_submitted(), f"{len(gate.jobs)} QPU job(s)")
    ran = all(j["counts"] is not None for j in gate.jobs)

    # ---- reconstruction (CPU / NumPy)
    results = {"num_qubits": n, "cuts": fr["cuts"], "fragments": {"up": qs_up, "down": qs_down},
               "assignment": {"up": targets[dev_up], "down": targets[dev_down]}, "alternatives": best["alternatives"],
               "exact": v_cpu, "reference_device": dev, "reference_value": v_dev,
               "noiseless_reconstruction": v_rec_exact, "heft": {"device_of": sch["device_of"]}}
    if ran:
        with H.Timer() as tm:
            v_noisy = contract(E1, F2, coeffs, k)
            # first-order error propagation of independent fragment estimates
            G2 = (F2 @ np.kron(*([M_PREP] * k)).T if k > 1 else F2 @ M_PREP.T)
            Mk = np.kron(*([M_PREP] * k)) if k > 1 else M_PREP
            var = 0.0
            for ti in range(len(terms)):
                c = coeffs[ti] * (0.5 ** k)
                var += c * c * (np.sum((G2[ti] * V1[ti]) ** 2) + np.sum(((E1[ti] @ Mk) * V2[ti]) ** 2))
            sigma = math.sqrt(var)
        stages.add(H.StageRecord("reconstruction", "cpu (NumPy)", "quasi-probability tensor contraction "
                                 f"(4^{k} terms per observable term)", tm.seconds, "einsum",
                                 notes=["QUMO has no GPU tensor-contraction kernel; trivial at k=1"]))
        allow = 5 * sigma + a.band * float(np.sum(np.abs(coeffs)))
        log.close(f"noisy reconstruction within 5 sigma + {a.band:g} x sum|c| of exact", v_noisy, v_cpu, allow)
        results.update({"noisy_reconstruction": v_noisy, "sigma": sigma, "error": v_noisy - v_cpu,
                        "sampling_overhead_terms": 4 ** k})
        say(f"[result] exact {v_cpu:.6f}  noiseless cut {v_rec_exact:.6f}  noisy cut {v_noisy:.6f} +- {sigma:.4f}")
    wall = time.perf_counter() - t_start
    print(f"[time] wall {wall:.1f} s")
    report = {"schema": "qumo.hetero_cutting.v1", "args": vars(a), "environment": H.environment(q), "gpu_info": gpu,
              "results": results, "qpu_jobs": gate.summary(), "wall_seconds": round(wall, 2), "checks": log.records}
    path = H.write_report(Path(a.out) / f"seed{a.seed}", report, stages, "Circuit cutting across two QPU modalities")
    say(f"[report] {path}")
    report["_all_passed"] = log.all_passed()
    log.finish(exit_on_failure=(__name__ == "__main__"))
    return report


if __name__ == "__main__":
    run()
