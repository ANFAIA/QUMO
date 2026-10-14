"""pytket reference for examples/frameworks/tket/measurement_reduction.mojo (optional).

Uses pytket's measurement_reduction on the page's operator
O = 0.1 Z0 + 0.4 Y0 Z1 X2 X3 Y4 + 0.2 X0 X1 and prints the number of
measurement circuits for NonConflictingSets (qubit-wise commuting) and
CommutingSets, the CX count of each CommutingSets circuit, and the exact
expectation on the page state V(q2) H(q4)|00000> from pytket's state vector.
The QUMO example asserts 3 / 2 circuits, 1 CX in the second commuting set,
and the value 0.1.

Run (QUMO repo root): bench/.venv/bin/python examples/frameworks/tket/reference/measurement_ref.py
Needs pytket (2.18.5 was used) and numpy.
"""

from __future__ import annotations

import json

import numpy as np
from pytket.circuit import Circuit, OpType, Qubit
from pytket.partition import PauliPartitionStrat, measurement_reduction
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils.operators import QubitPauliOperator

TERMS = [
    (0.1, {0: Pauli.Z}),
    (0.4, {0: Pauli.Y, 1: Pauli.Z, 2: Pauli.X, 3: Pauli.X, 4: Pauli.Y}),
    (0.2, {0: Pauli.X, 1: Pauli.X}),
]


def main() -> None:
    strings = {QubitPauliString({Qubit(q): p for q, p in t.items()}): c for c, t in TERMS}
    op = QubitPauliOperator(strings)
    state = Circuit(5)
    state.V(2)
    state.H(4)
    psi = state.get_statevector()
    value = float(np.real(op.state_expectation(psi)))
    out = {"exact_value": value}
    for name, strat in (("NonConflictingSets", PauliPartitionStrat.NonConflictingSets),
                        ("CommutingSets", PauliPartitionStrat.CommutingSets)):
        setup = measurement_reduction(list(strings), strat)
        out[name] = {
            "circuits": len(setup.measurement_circs),
            "cx_per_circuit": [c.n_gates_of_type(OpType.CX) for c in setup.measurement_circs],
        }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
