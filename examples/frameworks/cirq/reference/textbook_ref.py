"""cirq reference for examples/frameworks/cirq/textbook_algorithms.mojo (optional).

Reruns the page instances with cirq and prints the numbers the QUMO example
checks:
- teleportation of the X**0.25 state: Bob's Bloch vector (deferred-measurement
  form, density-matrix simulator), expected [0, -0.7071, 0.7071];
- QFT on 4 qubits: max |cirq.qft(4) - DFT| (cirq's qft without reversal is
  compared with the bit-reversed DFT);
- QPE of Z**(2 theta): readout distribution for theta = 0.234 (3 bits, |1>)
  and theta = 0.123456 (4 bits, |+>), against the Fejer kernel the QUMO example
  uses;
- Grover on 2 qubits + ancilla, one iteration: P(marked) for all 4 strings.

Run (QUMO repo root): bench/.venv/bin/python examples/frameworks/cirq/reference/textbook_ref.py
Needs cirq (1.7.0 was used) and numpy.
"""

from __future__ import annotations

import json
import math

import cirq
import numpy as np


def teleport_bloch() -> list[float]:
    msg, alice, bob = cirq.LineQubit.range(3)
    c = cirq.Circuit(
        cirq.X(msg) ** 0.25,
        cirq.H(alice), cirq.CNOT(alice, bob),
        cirq.CNOT(msg, alice), cirq.H(msg),
        cirq.CNOT(alice, bob), cirq.CZ(msg, bob),  # deferred-measurement corrections
    )
    rho = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(c).final_density_matrix
    rb = cirq.partial_trace(rho.reshape([2] * 6), [2])
    paulis = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    return [float(np.real(np.trace(rb @ p))) for p in paulis]


def qft_error(n: int = 4) -> float:
    dim = 1 << n
    dft = np.array([[np.exp(2j * np.pi * j * k / dim) for j in range(dim)] for k in range(dim)]) / math.sqrt(dim)
    u = cirq.unitary(cirq.qft(*cirq.LineQubit.range(n)))  # cirq: big-endian, includes the reversal
    return float(np.max(np.abs(u - dft)))


def fejer(theta: float, n: int, m: int) -> float:
    d = theta - m / 2**n
    s = math.sin(math.pi * d)
    if abs(s) < 1e-15:
        return 1.0
    return math.sin(math.pi * 2**n * d) ** 2 / (4**n * s * s)


def qpe_probs(theta: float, n: int, plus: bool) -> list[float]:
    reg = cirq.LineQubit.range(n)
    tgt = cirq.LineQubit(n)
    u = cirq.Z ** (2 * theta)
    c = cirq.Circuit(cirq.H(tgt) if plus else cirq.X(tgt))
    c.append(cirq.H.on_each(*reg))
    for i, q in enumerate(reg):
        c.append(cirq.ControlledGate(u ** (2 ** (n - 1 - i)))(q, tgt))
    c.append(cirq.qft(*reg, without_reverse=False, inverse=True))
    psi = cirq.Simulator(dtype=np.complex128).simulate(c).final_state_vector
    p = np.abs(psi.reshape([2] * (n + 1))) ** 2
    marg = p.sum(axis=n).reshape(-1)  # big-endian register value
    return [float(x) for x in marg]


def grover_success() -> list[float]:
    out = []
    for marked in range(4):
        q0, q1, anc = cirq.LineQubit.range(3)
        bits = [(marked >> 1) & 1, marked & 1]
        flips = [cirq.X(q) for q, b in zip((q0, q1), bits) if b == 0]
        c = cirq.Circuit(cirq.X(anc), cirq.H(anc), cirq.H(q0), cirq.H(q1))
        c.append(flips + [cirq.TOFFOLI(q0, q1, anc)] + flips)
        c.append([cirq.H(q0), cirq.H(q1), cirq.X(q0), cirq.X(q1), cirq.H(q1), cirq.CNOT(q0, q1),
                  cirq.H(q1), cirq.X(q0), cirq.X(q1), cirq.H(q0), cirq.H(q1)])
        psi = cirq.Simulator(dtype=np.complex128).simulate(c).final_state_vector
        p = np.abs(psi.reshape(2, 2, 2)) ** 2
        out.append(float(p[bits[0], bits[1], :].sum()))
    return out


def main() -> None:
    p1 = qpe_probs(0.234, 3, plus=False)
    p2 = qpe_probs(0.123456, 4, plus=True)
    fej1 = [fejer(0.234, 3, m) for m in range(8)]
    # |+> = (|0> + |1>)/sqrt(2): eigenphases 0 and theta with weight 1/2 each
    fej2 = [0.5 * (1.0 if m == 0 else 0.0) + 0.5 * fejer(0.123456, 4, m) for m in range(16)]
    print(json.dumps({
        "cirq": cirq.__version__,
        "teleport_bloch_bob": teleport_bloch(),
        "qft4_max_abs_diff_vs_dft": qft_error(),
        "qpe_theta_0.234_n3_probs": p1,
        "qpe_theta_0.234_n3_max_diff_vs_fejer": max(abs(a - b) for a, b in zip(p1, fej1)),
        "qpe_theta_0.234_n3_argmax_estimate": p1.index(max(p1)) / 8,
        "qpe_theta_0.123456_n4_plus_max_diff_vs_fejer": max(abs(a - b) for a, b in zip(p2, fej2)),
        "grover_success_per_marked": grover_success(),
    }))


if __name__ == "__main__":
    main()
