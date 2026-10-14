"""Qiskit reference for examples/frameworks/qiskit/shor_order_finding.mojo (optional).

Rebuilds the tutorial's N = 15, a = 2 order-finding circuit in qiskit
(8 control + 4 target qubits; M2 = SWAP(2,3) SWAP(1,2) SWAP(0,1),
M4 = SWAP(1,3) SWAP(0,2), made controlled with `.control()`; inverse QFT on
the control register) and compares it with the distribution QUMO computes
exactly in the Mojo example: 1/4 on each of 0, 64, 128 and 192.

Checks:
- qiskit's exact Statevector marginal equals {0, 64, 128, 192} -> 0.25 (1e-12);
- 10,000 qiskit-aer shots: total variation distance to that distribution < 0.02.

Run (QUMO repo root): bench/.venv/bin/python examples/frameworks/qiskit/reference/shor15_ref.py
Needs qiskit (2.5.2) and qiskit-aer (0.17.2).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from _checks import CheckLog  # noqa: E402

import qiskit  # noqa: E402
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile  # noqa: E402
from qiskit.circuit.library import QFTGate  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

N_CTRL = 8
SHOTS = 10_000
EXACT = {0: 0.25, 64: 0.25, 128: 0.25, 192: 0.25}


def m2() -> QuantumCircuit:
    qc = QuantumCircuit(4, name="M2")
    qc.swap(2, 3)
    qc.swap(1, 2)
    qc.swap(0, 1)
    return qc


def m4() -> QuantumCircuit:
    qc = QuantumCircuit(4, name="M4")
    qc.swap(1, 3)
    qc.swap(0, 2)
    return qc


def circuit(measure: bool) -> QuantumCircuit:
    ctrl = QuantumRegister(N_CTRL, "control")
    tgt = QuantumRegister(4, "target")
    qc = QuantumCircuit(ctrl, tgt)
    if measure:
        qc.add_register(ClassicalRegister(N_CTRL, "out"))
    qc.x(tgt[0])
    qc.h(ctrl)
    qc.compose(m2().control(), qubits=[ctrl[0]] + list(tgt), inplace=True)
    qc.compose(m4().control(), qubits=[ctrl[1]] + list(tgt), inplace=True)
    qc.append(QFTGate(N_CTRL).inverse(), ctrl)
    if measure:
        qc.measure(ctrl, range(N_CTRL))
    return qc


def main() -> None:
    log = CheckLog("qiskit/shor15_ref")
    print("qiskit", qiskit.__version__)
    probs = Statevector(circuit(False).decompose(reps=4)).probabilities(list(range(N_CTRL)))
    err = max(abs(probs[y] - EXACT.get(y, 0.0)) for y in range(1 << N_CTRL))
    log.close("qiskit Statevector marginal equals QUMO's exact peaks", err, 0.0, 1e-12)
    from qiskit_aer import AerSimulator

    sim = AerSimulator(seed_simulator=15)
    counts = sim.run(transpile(circuit(True), sim), shots=SHOTS).result().get_counts()
    print("aer counts:", dict(sorted(counts.items())))
    tvd = 0.5 * sum(abs(counts.get(format(y, "08b"), 0) / SHOTS - EXACT.get(y, 0.0)) for y in range(1 << N_CTRL))
    log.within("TVD(aer 10k shots, exact)", tvd, 0.0, 0.02)
    log.finish()


if __name__ == "__main__":
    main()
