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
  * Clifford circuits above MAX_VERIFY_QUBITS (60+ qubits) are checked exactly
    by Heisenberg propagation of Pauli sums (`verify_clifford`): every logical
    X / Z generator, pushed through the original circuit and relabelled by the
    final layout, must equal the same generator placed by the initial layout
    and pushed through the compiled circuit, signs included (equality up to
    global phase).
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


# --------------------------------------------------------------------------- Clifford (Pauli propagation)
# Paulis are (x, z) bit masks over the qubits; per qubit (x, z) = (0,0) I,
# (1,0) X, (0,1) Z, (1,1) Y (all Hermitian), so conjugation by a unitary has
# a real Pauli transfer matrix (PTM) and Pauli sums keep real coefficients.
_P1 = [np.eye(2, dtype=complex), np.array([[0, 1], [1, 0]], dtype=complex),
       np.array([[1, 0], [0, -1]], dtype=complex), np.array([[0, -1j], [1j, 0]], dtype=complex)]
_IDX_BITS = [(0, 0), (1, 0), (0, 1), (1, 1)]  # pauli index -> (x, z)
_PTM_CACHE = {}
MAX_PAULI_TERMS = 4096
# Pauli-coefficient tolerance, matched to the statevector check (fidelity >=
# 1 - 1e-6, i.e. amplitude errors up to ~1e-3): compilers emit floating-point
# angles and their residues accumulate (Qiskit O2 on a 60-qubit linear function
# on heavy-hex leaves coefficient errors of 1.2e-4). A wrong Clifford changes
# some coefficient by O(1), far above this tolerance.
CLIFFORD_TOL = 1e-3
# drop Pauli terms below this weight during propagation
PRUNE = 1e-8


def _ptm(u):
    """Real PTM R[j][i] = tr(P_j U P_i U^dag) / d of a 1q / 2q unitary."""
    k = u.shape[0]
    basis = _P1 if k == 2 else [np.kron(_P1[b], _P1[a]) for b in range(4) for a in range(4)]
    d = float(k)
    r = np.zeros((len(basis), len(basis)))
    ud = u.conj().T
    for i, pi_ in enumerate(basis):
        img = u @ pi_ @ ud
        for j, pj in enumerate(basis):
            r[j, i] = np.real(np.trace(pj @ img)) / d
    return r


def _gate_ptm(op):
    key = (op.name, tuple(round(float(p), 12) for p in op.params))
    r = _PTM_CACHE.get(key)
    if r is None:
        # qiskit matrices are little-endian: qubit 0 of the gate is the low bit
        r = _ptm(np.asarray(op.to_matrix(), dtype=complex))
        _PTM_CACHE[key] = r
    return r


_COLS_CACHE = {}


def _gate_cols(op):
    """Sparse PTM columns: cols[i] = [(j, R[j][i]) for nonzero entries]."""
    key = (op.name, tuple(round(float(p), 12) for p in op.params))
    cols = _COLS_CACHE.get(key)
    if cols is None:
        r = _gate_ptm(op)
        cols = [[(int(j), float(r[j, i])) for j in range(r.shape[0]) if abs(r[j, i]) > 1e-12]
                for i in range(r.shape[1])]
        _COLS_CACHE[key] = cols
    return cols


def _pauli_index(x, z, q):
    return ((x >> q) & 1) | (((z >> q) & 1) << 1)


def _set_pauli(x, z, q, idx):
    bx, bz = _IDX_BITS[idx]
    x = (x & ~(1 << q)) | (bx << q)
    z = (z & ~(1 << q)) | (bz << q)
    return x, z


def _ops(qc):
    """[(operation, qubit indices, qubit bit mask)] of the unitary gates."""
    out = []
    for inst in qc.data:
        if inst.operation.name in NON_UNITARY:
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        mask = 0
        for q in qs:
            mask |= 1 << q
        out.append((inst.operation, qs, mask))
    return out


def _propagate(ops, terms):
    """Heisenberg-propagate a Pauli sum {(x, z): coeff} through the gates
    (P -> U P U^dag, in circuit order). None if it grows past MAX_PAULI_TERMS
    or meets a gate on more than 2 qubits."""
    supp = 0
    for x, z in terms:
        supp |= x | z
    for op, qs, mask in ops:
        if not supp & mask:
            continue
        if len(qs) > 2:
            return None
        touched = [(k, c) for k, c in terms.items() if (k[0] | k[1]) & mask]
        cols = _gate_cols(op)
        for k, _ in touched:
            del terms[k]
        for (x, z), c in touched:
            if len(qs) == 1:
                col = cols[_pauli_index(x, z, qs[0])]
            else:
                col = cols[_pauli_index(x, z, qs[0]) + 4 * _pauli_index(x, z, qs[1])]
            for j, v in col:
                if len(qs) == 1:
                    nk = _set_pauli(x, z, qs[0], j)
                else:
                    nx, nz = _set_pauli(x, z, qs[0], j & 3)
                    nk = _set_pauli(nx, nz, qs[1], j >> 2)
                terms[nk] = terms.get(nk, 0.0) + c * v
        for k in [k for k, c in terms.items() if abs(c) < PRUNE]:
            del terms[k]
        if len(terms) > MAX_PAULI_TERMS:
            return None
        supp = 0
        for x, z in terms:
            supp |= x | z
    return terms


def _relabel(terms, mapping):
    out = {}
    for (x, z), c in terms.items():
        nx = nz = 0
        q = 0
        while (x | z) >> q:
            if (x >> q) & 1:
                nx |= 1 << mapping[q]
            if (z >> q) & 1:
                nz |= 1 << mapping[q]
            q += 1
        out[(nx, nz)] = c
    return out


def is_clifford_circuit(qc: QuantumCircuit):
    """True when every gate is a Clifford (1q gates by their matrix)."""
    for op, qs, _ in _ops(qc):
        if len(qs) > 2:
            return False
        r = _gate_ptm(op)
        if not np.all((np.abs(r) < 1e-9) | (np.abs(np.abs(r) - 1) < 1e-9)):
            return False
    return True


def verify_clifford(original: QuantumCircuit, out, seed=7):
    """Exact equivalence check for Clifford originals of any width (1.0 / 0.0;
    None when the original is not Clifford or a Pauli sum explodes).

    Every logical generator X_l / Z_l is pushed through the original circuit
    and relabelled by the final layout, and must equal the same generator
    placed by the initial layout and pushed through the compiled circuit (so
    the compiled circuit = original on the logical qubits (x) anything on the
    ancillas). Non-Clifford-looking 1q gates in compiled output (Euler forms,
    rotations commuted through CZ, KAK local gates) are handled exactly by
    Pauli-sum propagation. Global phase is not checked.
    """
    orig = original.remove_final_measurements(inplace=False)
    if not is_clifford_circuit(orig):
        return None
    oo = _ops(orig)
    oc = _ops(out.circuit)
    n = orig.num_qubits
    for l in range(n):
        for bit in ("x", "z"):
            src = {((1 << l) if bit == "x" else 0, (1 << l) if bit == "z" else 0): 1.0}
            want = _propagate(oo, src)
            if want is None:
                return None
            want = _relabel(want, out.final)
            p = out.initial[l]
            got = _propagate(oc, {((1 << p) if bit == "x" else 0, (1 << p) if bit == "z" else 0): 1.0})
            if got is None:
                return None
            keys = set(got) | set(want)
            if any(abs(got.get(k, 0.0) - want.get(k, 0.0)) > CLIFFORD_TOL for k in keys):
                return 0.0
    return 1.0
