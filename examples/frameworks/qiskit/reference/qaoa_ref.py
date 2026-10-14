"""Qiskit reference for examples/frameworks/qiskit/qaoa_maxcut.mojo (optional).

Rebuilds the tutorial's p=2 QAOA circuit with qiskit's QAOAAnsatz on the
5-node graph (0,1)(0,2)(0,4)(1,2)(2,3)(3,4) and evaluates the tutorial's
objective <sum_(u,v) Z_u Z_v> with an exact Statevector at

  * the two fixed points whose values are committed in qaoa_maxcut.mojo
    (the tutorial's initial point and an arbitrary point), and
  * optionally QUMO's optimum: --qumo-angles g0 g1 b0 b1 (QUMO convention).

Angle conventions: QUMO's qaoa_maxcut_ansatz applies RZZ(gamma) per edge,
i.e. exp(-i gamma/2 ZZ), and RX(2 beta); qiskit's QAOAAnsatz applies
exp(-i gamma ZZ) for the cost operator sum ZZ and RX(2 beta). Hence
gamma_qiskit = gamma_qumo / 2 and beta_qiskit = beta_qumo. QAOAAnsatz orders
its parameters (beta[0], beta[1], gamma[0], gamma[1]).

Run (QUMO repo root):
    bench/.venv/bin/python examples/frameworks/qiskit/reference/qaoa_ref.py
    bench/.venv/bin/python examples/frameworks/qiskit/reference/qaoa_ref.py --qumo-angles <g0> <g1> <b0> <b1> --qumo-value <sumZZ>
Needs qiskit (2.5.2 was used for the committed numbers).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from _checks import CheckLog  # noqa: E402

import qiskit  # noqa: E402
from qiskit.circuit.library import QAOAAnsatz  # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402

EDGES = [(0, 1), (0, 2), (0, 4), (1, 2), (2, 3), (3, 4)]
N = 5

# Values committed in qaoa_maxcut.mojo (QUMO convention: g0, g1, b0, b1).
FIXED_POINTS = {
    "tutorial initial point": [2 * math.pi, 2 * math.pi, math.pi / 2, math.pi / 2],
    "arbitrary point": [0.7, 1.3, 0.4, 0.25],
}


def cost_operator() -> SparsePauliOp:
    terms = []
    for u, v in EDGES:
        label = ["I"] * N
        label[N - 1 - u] = "Z"
        label[N - 1 - v] = "Z"
        terms.append(("".join(label), 1.0))
    return SparsePauliOp.from_list(terms)


def sum_zz(qumo_angles: list[float]) -> float:
    g0, g1, b0, b1 = qumo_angles
    h = cost_operator()
    ansatz = QAOAAnsatz(h, reps=2)
    names = [p.name for p in ansatz.parameters]
    assert names == ["β[0]", "β[1]", "γ[0]", "γ[1]"], names
    bound = ansatz.assign_parameters([b0, b1, g0 / 2, g1 / 2])
    sv = Statevector(bound.decompose(reps=3))
    return float(sv.expectation_value(h).real)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qumo-angles", nargs=4, type=float)
    ap.add_argument("--qumo-value", type=float, help="QUMO's <sum ZZ> at --qumo-angles")
    args = ap.parse_args()
    log = CheckLog("qiskit/qaoa_ref")
    print("qiskit", qiskit.__version__)
    for name, angles in FIXED_POINTS.items():
        v = sum_zz(angles)
        print(f"{name}: QUMO angles {angles} -> qiskit <sum ZZ> = {v!r}")
        log.check(f"{name} evaluated", math.isfinite(v), f"value={v!r}")
    if args.qumo_angles:
        v = sum_zz(args.qumo_angles)
        print(f"QUMO optimum {args.qumo_angles} -> qiskit <sum ZZ> = {v!r}, expected cut = {(len(EDGES) - v) / 2!r}")
        if args.qumo_value is not None:
            log.close("qiskit vs QUMO at the QUMO optimum", v, args.qumo_value, 1e-10)
    log.finish()


if __name__ == "__main__":
    main()
