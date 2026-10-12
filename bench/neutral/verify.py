"""SDK-independent validation, correctness check and metrics.

The compiled circuit of every SDK arrives as a Qiskit QuantumCircuit used only
as a neutral container. Checks:
  * native: every 2q gate is the target's 2q gate and sits on a coupling edge;
    1q gates may be any single-qubit unitary (spelling of 1q gates differs per
    SDK and is normalised by counting 1q *runs*, see below);
  * correct: for <= MAX_VERIFY_QUBITS logical qubits, a random product input
    state is pushed through the original and the compiled circuit (using the
    SDK's reported initial/final layouts, ancillas start in |0>) and the
    output states must have fidelity >= 1 - 1e-6.
Metrics (identical formulas for all SDKs):
  * n2q, depth2q, depth (all unitary gates), runs1q = number of maximal runs of
    1q gates per qubit between 2q gates (= number of arbitrary 1q unitaries
    needed, independent of rz/sx/u spelling);
  * ESP = (1-e1)^runs1q * (1-e2)^n2q * exp(-T/T1) * exp(-T/T2) per qubit-time,
    with T from the scheduled critical path (1q run = t_1q unless it is a pure
    virtual-Z run on modalities with virtual Z).
"""
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

MAX_VERIFY_QUBITS = 14
NON_UNITARY = {"measure", "barrier", "reset", "delay"}


def check_native(qc: QuantumCircuit, target):
    bad = []
    for inst in qc.data:
        name = inst.operation.name
        if name in NON_UNITARY:
            continue
        nq = len(inst.qubits)
        if nq == 1:
            continue
        if nq != 2 or name != target.gate_2q:
            bad.append(f"non-native {name} on {nq} qubits")
            continue
        a, b = (qc.find_bit(q).index for q in inst.qubits)
        if not target.all_to_all and (a, b) not in target.edge_set:
            bad.append(f"{name} on uncoupled pair ({a},{b})")
    return bad[:5]


def metrics(qc: QuantumCircuit, target):
    n = qc.num_qubits
    level = [0] * n  # 2q depth
    full = [0] * n  # depth
    run_open = [False] * n
    run_virtual = [True] * n
    clock = [0.0] * n
    n2q = 0
    runs = 0
    noise = target.noise

    def close_run(q):
        nonlocal runs
        if run_open[q]:
            runs += 1
            if not (noise.virtual_z and run_virtual[q]):
                clock[q] += noise.t_1q_ns
            run_open[q] = False
            run_virtual[q] = True

    for inst in qc.data:
        name = inst.operation.name
        if name in NON_UNITARY:
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if len(qs) == 1:
            q = qs[0]
            run_open[q] = True
            if name not in ("rz", "z", "s", "sdg", "t", "tdg", "p", "u1"):
                run_virtual[q] = False
            full[q] += 1
            continue
        for q in qs:
            close_run(q)
        n2q += 1
        l2 = max(level[q] for q in qs) + 1
        lf = max(full[q] for q in qs) + 1
        t0 = max(clock[q] for q in qs) + noise.t_2q_ns
        for q in qs:
            level[q] = l2
            full[q] = lf
            clock[q] = t0
    for q in range(n):
        close_run(q)
    duration_ns = max(clock) if clock else 0.0
    used = sum(1 for q in range(n) if full[q] > 0) or 1
    t_us = duration_ns / 1000.0
    idle = math.exp(-t_us / noise.t1_us) * math.exp(-t_us / noise.t2_us)
    esp = (1 - noise.err_1q) ** runs * (1 - noise.err_2q) ** n2q * idle ** used
    return {
        "n2q": n2q,
        "depth2q": max(level) if level else 0,
        "depth": max(full) if full else 0,
        "runs1q": runs,
        "duration_us": t_us,
        "esp": esp,
    }


def _prep(nlog, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, np.pi, nlog), rng.uniform(0, 2 * np.pi, nlog)


def _compress(out):
    """Restrict the compiled circuit to physical qubits that matter (touched by
    a gate or holding a logical qubit); returns (circuit, initial, final)."""
    qc = out.circuit
    active = set(out.initial) | set(out.final)
    for inst in qc.data:
        if inst.operation.name in NON_UNITARY:
            continue
        for q in inst.qubits:
            active.add(qc.find_bit(q).index)
    order = sorted(active)
    pos = {p: i for i, p in enumerate(order)}
    small = QuantumCircuit(len(order))
    for inst in qc.data:
        if inst.operation.name in NON_UNITARY:
            continue
        small.append(inst.operation, [pos[qc.find_bit(q).index] for q in inst.qubits])
    small.global_phase = qc.global_phase
    return small, [pos[p] for p in out.initial], [pos[p] for p in out.final]


def verify(original: QuantumCircuit, out, seed=7):
    """Fidelity between expected and compiled output state (None if too large)."""
    nlog = original.num_qubits
    if nlog > MAX_VERIFY_QUBITS:
        return None
    body, initial, final = _compress(out)
    m = body.num_qubits
    if m > MAX_VERIFY_QUBITS + 6:
        return None
    a, b = _prep(nlog, seed)
    ref = QuantumCircuit(nlog)
    for i in range(nlog):
        ref.ry(a[i], i)
        ref.rz(b[i], i)
    ref.compose(original.remove_final_measurements(inplace=False), inplace=True)
    psi = Statevector(ref).data
    phys = QuantumCircuit(m)
    for i in range(nlog):
        phys.ry(a[i], initial[i])
        phys.rz(b[i], initial[i])
    phys.compose(body, qubits=list(range(m)), inplace=True)
    phi = Statevector(phys).data
    exp = np.zeros(2 ** m, dtype=complex)
    for idx in range(2 ** nlog):
        amp = psi[idx]
        if amp == 0:
            continue
        p = 0
        for i in range(nlog):
            if (idx >> i) & 1:
                p |= 1 << final[i]
        exp[p] = amp
    return float(abs(np.vdot(exp, phi)) ** 2)
