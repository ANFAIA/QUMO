"""Evaluate a trained run and publish it into models/ai_transpiler/synth when it
beats QUMO's greedy synthesizer (the drop rule).

    python eval.py RUN_DIR --id lf_line6 [--instances 100] [--models ../../../models/ai_transpiler/synth]
                   [--staging DIR] [--eval-bin build/eval_blocks] [--force-keep]

Steps:
 1. export the run (f16) into a staging model root (export_model.py);
 2. run eval_blocks.mojo on seeded random blocks of the model's own coupling
    graph: policy (greedy ensemble; beam 8 + 4 samples) vs block_synth greedy /
    auto / exact, every output verified exactly;
 3. keep the model when the best policy setting is at least as cheap as greedy
    on >= 60% of the instances AND cheaper on average (cost ratio < 1); record
    the eval (and the decision) in the manifest's training.eval;
 4. when kept, re-export into the models root and rebuild INDEX.json.
Exit status 0 in both cases; the decision is printed as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = sys.executable


def qiskit_baseline(root, mid, ninst):
    """Qiskit 2.5.2 transpile(optimization_level=3) of uniformly random blocks
    (LinearFunction / PermutationGate / random Clifford) onto the model graph
    with a fixed trivial layout; the routing permutation left in the final
    layout is charged with its optimal SWAP count (BFS, 3 CX per SWAP) so the
    result is exact. Unpaired with the Mojo instances (same distribution only)."""
    import numpy as np
    from collections import deque
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import LinearFunction, PermutationGate
    from qiskit.quantum_info import random_clifford
    from qiskit.synthesis.linear import random_invertible_binary_matrix
    from qiskit.transpiler import CouplingMap
    m = json.load(open(os.path.join(root, mid, "manifest.json")))
    n, kind = m["num_qubits"], m["kind"]
    if kind == "pauli_network" or n > 9:
        return {"skipped": "kind/size"}
    edges = [tuple(e) for e in m["edges"]]
    cm = CouplingMap([list(e) for e in edges] + [[b, a] for a, b in edges])
    start = tuple(range(n)); dist = {start: 0}; q = deque([start])
    while q:
        s = q.popleft()
        for a, b in edges:
            t = list(s); t[a], t[b] = t[b], t[a]; t = tuple(t)
            if t not in dist:
                dist[t] = dist[s] + 1; q.append(t)
    rng = np.random.default_rng(7)
    costs = []
    for i in range(min(ninst, 50)):
        qc = QuantumCircuit(n)
        if kind == "permutation":
            qc.append(PermutationGate(list(rng.permutation(n))), range(n))
        elif kind == "linear_function":
            qc.append(LinearFunction(random_invertible_binary_matrix(n, seed=int(rng.integers(1 << 30)))), range(n))
        else:
            qc.compose(random_clifford(n, seed=int(rng.integers(1 << 30))).to_circuit(), inplace=True)
        tq = transpile(qc, coupling_map=cm, basis_gates=["cx", "u"], optimization_level=3,
                       seed_transpiler=7, initial_layout=list(range(n)))
        fl = tq.layout.final_index_layout() if tq.layout is not None else list(range(n))
        fix = dist[tuple(int(x) for x in np.argsort(fl))]
        costs.append(tq.count_ops().get("cx", 0) + 3 * fix)
    return {"instances": len(costs), "mean_cost": float(np.mean(costs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--id", required=True)
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--models", default=os.path.join(REPO, "models", "ai_transpiler", "synth"))
    ap.add_argument("--staging", default=os.path.join(REPO, "build", "ai_staging", "synth"))
    ap.add_argument("--eval-bin", default=os.path.join(REPO, "build", "eval_blocks"))
    ap.add_argument("--force-keep", action="store_true")
    ap.add_argument("--dtype", default="f16")
    a = ap.parse_args()

    os.makedirs(a.staging, exist_ok=True)
    exp = os.path.join(HERE, "export_model.py")
    subprocess.run([PY, exp, a.run, "--id", a.id, "--out", a.staging, "--dtype", a.dtype], check=True)
    if not os.path.exists(a.eval_bin):
        os.makedirs(os.path.dirname(a.eval_bin), exist_ok=True)
        subprocess.run([os.path.join(REPO, ".venv", "bin", "mojo"), "build", "-I", REPO, "-o", a.eval_bin,
                        os.path.join(HERE, "eval_blocks.mojo")], check=True, cwd=REPO)
    ej = os.path.join(a.run, "eval_blocks.json")
    subprocess.run([a.eval_bin, a.staging, a.id, str(a.instances), ej], check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL)
    ev = json.load(open(ej))
    pv = ev.get("policy_vs_greedy", {})
    meth = ev["methods"]
    fails = sum(m.get("verify_failures", 0) for m in meth.values())
    keep = (pv.get("win_or_tie", 0) >= 0.6 and pv.get("cost_ratio", 9) < 1.0 and fails == 0)
    try:
        ev["qiskit_o3_unpaired"] = qiskit_baseline(a.staging, a.id, a.instances)
    except Exception as e:  # optional baseline
        ev["qiskit_o3_unpaired"] = {"error": str(e)[:200]}
    ev["rule"] = "keep iff policy (beam8+4 samples) <= greedy cost on >= 60% of instances, mean cost ratio < 1, 0 verify failures"
    ev["kept"] = bool(keep or a.force_keep)
    with open(os.path.join(a.run, "eval_final.json"), "w") as f:
        json.dump(ev, f, indent=1)
    if ev["kept"]:
        subprocess.run([PY, exp, a.run, "--id", a.id, "--out", a.models, "--dtype", a.dtype,
                        "--eval-json", os.path.join(a.run, "eval_final.json")], check=True)
    print(json.dumps({"id": a.id, "kept": ev["kept"], "policy_vs_greedy": pv,
                      "policy_vs_auto": ev.get("policy_vs_auto"), "policy_vs_exact": ev.get("policy_vs_exact"),
                      "costs": {k: round(v["mean_cost"], 3) for k, v in meth.items()},
                      "ms": {k: round(v["ms_per_block"], 3) for k, v in meth.items()},
                      "verify_failures": fails, "qiskit_o3_unpaired": ev.get("qiskit_o3_unpaired")}))


if __name__ == "__main__":
    main()
