"""cirq reference for examples/frameworks/cirq/variational_sweep.mojo (optional).

Rebuilds the fixed 3x3 instance (random.seed(7) draw of the page's
rand2d/random_instance) and the page's ansatz with cirq gates (H,
ZPowGate on h = +1, CZPowGate with X conjugation for J = -1, XPowGate),
then prints the exact energy <E> from cirq's state vector at 5 parameter
points. The QUMO example hardcodes these values in ref_values().

Run (QUMO repo root): bench/.venv/bin/python examples/frameworks/cirq/reference/variational_ref.py
Needs cirq (1.7.0 was used for the quoted numbers) and numpy.
"""

from __future__ import annotations

import json
import random

import cirq
import numpy as np

L = 3


def rand2d(rows: int, cols: int) -> list[list[int]]:
    return [[random.choice([+1, -1]) for _ in range(cols)] for _ in range(rows)]


def instance(seed: int = 7):
    random.seed(seed)
    return rand2d(L, L), rand2d(L - 1, L), rand2d(L, L - 1)


def ansatz(h, jr, jc, alpha: float, beta: float, gamma: float):
    qs = [[cirq.GridQubit(i, j) for j in range(L)] for i in range(L)]
    c = cirq.Circuit()
    c.append(cirq.H(q) for row in qs for q in row)
    c.append(cirq.ZPowGate(exponent=beta)(qs[i][j]) for i in range(L) for j in range(L) if h[i][j] == 1)

    def link(a, b, j):
        ops = []
        if j == -1:
            ops += [cirq.X(a), cirq.X(b)]
        ops.append(cirq.CZPowGate(exponent=gamma)(a, b))
        if j == -1:
            ops += [cirq.X(a), cirq.X(b)]
        return ops

    for i in range(L - 1):
        for j in range(L):
            c.append(link(qs[i][j], qs[i + 1][j], jr[i][j]))
    for i in range(L):
        for j in range(L - 1):
            c.append(link(qs[i][j], qs[i][j + 1], jc[i][j]))
    c.append(cirq.XPowGate(exponent=alpha)(q) for row in qs for q in row)
    return c, [q for row in qs for q in row]


def energy_diag(h, jr, jc) -> np.ndarray:
    """E(bitstring) with cirq's big-endian state index over qubits ordered (0,0), (0,1), ..."""
    n = L * L
    out = np.zeros(1 << n)
    for idx in range(1 << n):
        bits = [(idx >> (n - 1 - k)) & 1 for k in range(n)]
        s = [[1 - 2 * bits[L * i + j] for j in range(L)] for i in range(L)]
        e = sum(h[i][j] * s[i][j] for i in range(L) for j in range(L))
        e += sum(jr[i][j] * s[i][j] * s[i + 1][j] for i in range(L - 1) for j in range(L))
        e += sum(jc[i][j] * s[i][j] * s[i][j + 1] for i in range(L) for j in range(L - 1))
        out[idx] = e
    return out


def main() -> None:
    h, jr, jc = instance()
    diag = energy_diag(h, jr, jc)
    points = [(0.1, 0.2, 0.3), (0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (7 / 9, 2 / 9, 4 / 9)]
    sim = cirq.Simulator(dtype=np.complex128)
    values = []
    for a, b, g in points:
        c, order = ansatz(h, jr, jc, a, b, g)
        psi = sim.simulate(c, qubit_order=order).final_state_vector
        values.append(float(np.real(np.vdot(psi, diag * psi))))
    print(json.dumps({"cirq": cirq.__version__, "h": h, "jr": jr, "jc": jc,
                      "points": points, "energies": values}, indent=None))


if __name__ == "__main__":
    main()
