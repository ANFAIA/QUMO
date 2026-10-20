"""Circuit suite of the QUMO-AI vs IBM AI transpiler head-to-head.

Every circuit is built once, deterministically (fixed seeds), normalised to the
same OpenQASM 2 text (no measurements / barriers, the common qelib1 gate set of
bench/neutral/prepare_circuits.py) and handed unchanged to every contender, so
all compilers see identical inputs.

Groups
  summit    Benchpress device suite "100Q" workouts (QFT, QV, circular
            EfficientSU2, BV, square Heisenberg, QAOA, BV-like, Clifford)
  feynman   Benchpress' Feynman circuits that fit the device
  hamlib    Benchpress' 100 representative HamLib Hamiltonians that fit the
            device, one Lie-Trotter step (Qiskit's default PauliEvolutionGate
            synthesis, flattened once)
  families  out-of-sample families highlighted by IBM's paper / README:
            EfficientSU2 (circular, full), QFT, random Cliffords and linear
            functions of 5-9 qubits, random permutations
  neutral   bench/neutral/circuits (QASMBench small/medium + Feynman, <= 14 q)

Deviations from Benchpress (documented in the report): EfficientSU2 parameters
are bound to seeded random angles (Benchpress transpiles them unbound) and
HamLib evolutions are flattened before compilation (Benchpress hands Qiskit a
PauliEvolutionGate and QUMO its own builder); both keep the inputs identical
for every contender.
"""
import json
import math
import os
import sys

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile

COMMON_BASIS = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz",
                "cx", "cz", "swap", "ccx", "cswap", "cp", "crz"]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def normalise(qc):
    out = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name in ("measure", "barrier", "reset", "delay"):
            continue
        out.append(inst.operation, [out.qubits[qc.find_bit(q).index] for q in inst.qubits])
    out = transpile(out, basis_gates=COMMON_BASIS, optimization_level=0)
    return qasm2.dumps(out)


def _bind(qc, seed):
    rng = np.random.default_rng(seed)
    return qc.assign_parameters(rng.uniform(0, 2 * math.pi, qc.num_parameters))


def efficient_su2(n, reps, entanglement, seed=12345):
    from qiskit.circuit.library import efficient_su2 as esu2
    return _bind(esu2(n, reps=reps, entanglement=entanglement), seed)


def qft(n):
    from qiskit.circuit.library import QFTGate
    qc = QuantumCircuit(n)
    qc.append(QFTGate(n), range(n))
    return qc


def bv_all_ones(n):  # Benchpress qiskit_gym.circuits.bv_all_ones (measurements dropped)
    qc = QuantumCircuit(n)
    qc.x(n - 1)
    qc.h(range(n))
    qc.cx(range(n - 1), n - 1)
    qc.h(range(n - 1))
    return qc


def trivial_bvlike(n):  # Benchpress qiskit_gym.circuits.trivial_bvlike_circuit
    qc = QuantumCircuit(n)
    for k in range(n - 1):
        qc.cx(k, n - 1)
    qc.x(n - 1)
    qc.z(n - 2)
    for k in range(n - 2, -1, -1):
        qc.cx(k, n - 1)
    return qc


def random_clifford(n, seed):
    from qiskit.quantum_info import random_clifford as rc
    return rc(n, seed=seed).to_circuit()


def random_linear(n, seed):
    from qiskit.circuit.library import LinearFunction
    from qiskit.synthesis.linear import random_invertible_binary_matrix
    return LinearFunction(random_invertible_binary_matrix(n, seed=seed)).definition


def random_permutation(n, seed):
    from qiskit.circuit.library import PermutationGate
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    qc.append(PermutationGate(list(rng.permutation(n))), range(n))
    return qc


def hamiltonian(terms, coeffs, evo_time=1.0):
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    op = SparsePauliOp(terms, coeffs)
    qc = QuantumCircuit(op.num_qubits)
    qc.append(PauliEvolutionGate(op, time=evo_time), range(op.num_qubits))
    return qc


def _qasm_file(path):
    return qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def build(groups, max_qubits, benchpress=None, neutral_dir=None, neutral_max=14):
    """[(group, name, qasm_text)] for the requested groups (circuits wider than
    `max_qubits` are skipped)."""
    bp = benchpress or os.path.join(REPO, ".ref", "benchpress")
    qdir = os.path.join(bp, "benchpress", "qasm")
    out = []

    def add(group, name, qc):
        if qc.num_qubits <= max_qubits:
            try:
                out.append((group, name, normalise(qc)))
            except Exception as e:  # e.g. Feynman files with classical instructions: skipped, listed
                print(f"skip {group}/{name}: {type(e).__name__}: {e}", file=sys.stderr)

    if "summit" in groups:
        add("summit", "QFT_100", _qasm_file(os.path.join(qdir, "qft", "qft_N100.qasm")))
        add("summit", "QV_100", _qasm_file(os.path.join(qdir, "qv", "qv_N100_12345.qasm")))
        add("summit", "circSU2_89", efficient_su2(89, 3, "circular"))
        add("summit", "circSU2_100", efficient_su2(100, 3, "circular"))
        add("summit", "BV_100", bv_all_ones(100))
        add("summit", "square_heisenberg_100",
            _qasm_file(os.path.join(qdir, "square-heisenberg", "square_heisenberg_N100.qasm")))
        add("summit", "QAOA_100", _qasm_file(os.path.join(qdir, "qaoa", "qaoa_barabasi_albert_N100_3reps.qasm")))
        add("summit", "BVlike_100", trivial_bvlike(100))
        add("summit", "clifford_100", _qasm_file(os.path.join(qdir, "clifford", "clifford_100_12345.qasm")))
    if "feynman" in groups:
        fdir = os.path.join(qdir, "feynman")
        for f in sorted(os.listdir(fdir)):
            if f.endswith(".qasm"):
                try:
                    qc = _qasm_file(os.path.join(fdir, f))
                except Exception:  # unparsable source: reported by count only
                    continue
                add("feynman", f[:-5], qc)
    if "hamlib" in groups:
        recs = json.load(open(os.path.join(bp, "benchpress", "hamiltonian", "hamlib", "100_representative.json")))
        for h in recs:
            if h["ham_qubits"] > max_qubits:
                continue
            add("hamlib", "ham_" + h["ham_instance"][1:-1],
                hamiltonian(h["ham_hamlib_hamiltonian_terms"], h["ham_hamlib_hamiltonian_coefficients"]))
    if "families" in groups:
        for n in (20, 50, 100):
            add("families", f"esu2_circular_r3_n{n}", efficient_su2(n, 3, "circular"))
            add("families", f"esu2_full_r1_n{n}", efficient_su2(n, 1, "full"))
            add("families", f"qft_n{n}", qft(n))
        for n in (5, 6, 7, 8, 9):
            for s in (1, 2):
                add("families", f"clifford_n{n}_s{s}", random_clifford(n, 100 * n + s))
                add("families", f"linear_n{n}_s{s}", random_linear(n, 200 * n + s))
        for n in (20, 40):
            add("families", f"clifford_n{n}_s1", random_clifford(n, 100 * n + 1))
            add("families", f"linear_n{n}_s1", random_linear(n, 200 * n + 1))
        for n in (8, 12, 27, 65):
            add("families", f"perm_n{n}", random_permutation(n, 300 + n))
    if "neutral" in groups:
        nd = neutral_dir or os.path.join(REPO, "bench", "neutral", "circuits")
        for f in sorted(os.listdir(nd)):
            if not f.endswith(".qasm"):
                continue
            text = open(os.path.join(nd, f)).read()
            qc = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            if qc.num_qubits <= min(max_qubits, neutral_max):
                out.append(("neutral", f[:-5], text))  # already normalised by prepare_circuits.py
    return out
