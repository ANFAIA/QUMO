"""Benchmark circuit generators (Benchpress / QASMBench / QV families).

Structural equivalents of the generators used by Qiskit Benchpress so QUMO
can be measured on the same workloads. Randomness uses QUMO's deterministic
RNG, so instances are reproducible but not bit-identical to numpy-seeded ones
(whenever a bit-identical input matters, Benchpress' QASM files are used).
"""

from std.math import pi, sqrt, acos, cos, sin
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_arity
from qumo.ir.param import Param
from qumo.ir.pauli import PauliSum, PauliTerm
from qumo.sim.rng import Rng


def _random_su2(mut c: Circuit, q: Int, mut rng: Rng):
    """Haar-random 1q unitary as U(theta, phi, lam) (theta ~ arccos(1-2u))."""
    var theta = 2.0 * acos(sqrt(rng.uniform()))
    var phi = 2.0 * pi * rng.uniform()
    var lam = 2.0 * pi * rng.uniform()
    c.u(theta, phi, lam, q)


def random_su4_block(mut c: Circuit, a: Int, b: Int, mut rng: Rng):
    """Random two-qubit unitary in KAK form: (A1xA2) N(x,y,z) (B1xB2), 3 CX.

    N(x,y,z) = exp(i(x XX + y YY + z ZZ)) via the Vatan-Williams 3-CNOT circuit.
    """
    _random_su2(c, a, rng)
    _random_su2(c, b, rng)
    var x = pi / 4 * rng.uniform()
    var y = pi / 4 * rng.uniform()
    var z = pi / 4 * rng.uniform()
    c.rz(-pi / 2, b)
    c.cx(b, a)
    c.rz(pi / 2 - 2 * z, a)
    c.ry(2 * x - pi / 2, b)
    c.cx(a, b)
    c.ry(pi / 2 - 2 * y, b)
    c.cx(b, a)
    c.rz(pi / 2, a)
    _random_su2(c, a, rng)
    _random_su2(c, b, rng)


def quantum_volume(n: Int, depth: Int, seed: UInt64 = 12345) -> Circuit:
    """QV model circuit: `depth` layers of random permutation + random SU(4) on pairs."""
    var c = Circuit(n, 0, "qv_" + String(n))
    var rng = Rng(seed)
    var perm = List[Int](length=n, fill=0)
    for _ in range(depth):
        for i in range(n):
            perm[i] = i
        var k = n - 1
        while k > 0:
            var j = rng.below(k + 1)
            var t = perm[k]
            perm[k] = perm[j]
            perm[j] = t
            k -= 1
        var p = 0
        while p + 1 < n:
            random_su4_block(c, perm[p], perm[p + 1], rng)
            p += 2
    return c^


def efficient_su2(n: Int, reps: Int, entanglement: String = "circular") -> Circuit:
    """EfficientSU2: (RY, RZ) rotation layers + CX entanglers, (reps+1)*2*n params."""
    var c = Circuit(n, 0, "efficient_su2")
    var count = 0
    for r in range(reps + 1):
        for q in range(n):
            c.ry(c.parameter("θ[" + String(count) + "]"), q)
            count += 1
        for q in range(n):
            c.rz(c.parameter("θ[" + String(count) + "]"), q)
            count += 1
        if r == reps:
            break
        if entanglement == "circular" and n > 2:
            c.cx(n - 1, 0)
        if entanglement == "full":
            for i in range(n):
                for j in range(i + 1, n):
                    c.cx(i, j)
        else:
            for q in range(n - 1):
                c.cx(q, q + 1)
    return c^


def dtc_unitary(n: Int, g: Float64 = 0.95, seed: UInt64 = 12345) -> Circuit:
    """Floquet unitary of a discrete time crystal (RX kicks, random RZZ, random RZ)."""
    var c = Circuit(n, 0, "dtc")
    var rng = Rng(seed)
    for i in range(n):
        c.rx(g * pi, i)
    var i = 0
    while i < n - 1:
        var phi = pi / 16 + rng.uniform() * (2 * pi / 16)
        c.rzz(2 * phi, i, i + 1)
        i += 2
    i = 1
    while i < n - 1:
        var phi = pi / 16 + rng.uniform() * (2 * pi / 16)
        c.rzz(2 * phi, i, i + 1)
        i += 2
    for q in range(n):
        var h = -pi + 2 * pi * rng.uniform()
        c.rz(h * pi, q)
    return c^


def random_clifford_circuit(n: Int, num_gates: Int, seed: UInt64 = 12345) -> Circuit:
    """Random Clifford circuit over {cx, cz, cy, swap, x, y, z, s, sdg, h}."""
    var c = Circuit(n, 0, "random_clifford")
    var rng = Rng(seed)
    var kinds: List[Int] = [Op.CX, Op.CZ, Op.CY, Op.SWAP, Op.X, Op.Y, Op.Z, Op.S, Op.SDG, Op.H]
    for _ in range(num_gates):
        var k = kinds[rng.below(len(kinds))]
        if op_arity(k) == 2 and n > 1:
            var a = rng.below(n)
            var b = rng.below(n - 1)
            if b >= a:
                b += 1
            c.add(k, a, b)
        elif op_arity(k) == 1:
            c.add(k, rng.below(n))
    return c^


def bv_all_ones(n: Int) -> Circuit:
    """Bernstein-Vazirani with all-ones secret: n-1 data qubits + 1 ancilla."""
    var c = Circuit(n, n - 1, "bv_all_ones")
    c.x(n - 1)
    for q in range(n):
        c.h(q)
    for q in range(n - 1):
        c.cx(q, n - 1)
    for q in range(n - 1):
        c.h(q)
    for q in range(n - 1):
        c.measure(q, q)
    return c^


def trivial_bvlike_circuit(n: Int) -> Circuit:
    """CX fan-in, X, Z, CX fan-out: optimizes to a single X and Z."""
    var c = Circuit(n, 0, "bvlike")
    for k in range(n - 1):
        c.cx(k, n - 1)
    c.x(n - 1)
    c.z(n - 2)
    var k = n - 2
    while k >= 0:
        c.cx(k, n - 1)
        k -= 1
    return c^


def multi_control_x(mut c: Circuit, controls: List[Int], target: Int, ancillas: List[Int]) raises:
    """MCX via a V-chain of Toffolis with clean ancillas (len(ancillas) >= k-2)."""
    var k = len(controls)
    if k == 0:
        c.x(target)
        return
    if k == 1:
        c.cx(controls[0], target)
        return
    if k == 2:
        c.ccx(controls[0], controls[1], target)
        return
    if len(ancillas) < k - 2:
        raise Error("QM-E820: multi_control_x with " + String(k) + " controls needs " + String(k - 2) + " clean ancillas.")
    c.ccx(controls[0], controls[1], ancillas[0])
    for i in range(2, k - 1):
        c.ccx(controls[i], ancillas[i - 2], ancillas[i - 1])
    c.ccx(controls[k - 1], ancillas[k - 3], target)
    var i = k - 2
    while i >= 2:
        c.ccx(controls[i], ancillas[i - 2], ancillas[i - 1])
        i -= 1
    c.ccx(controls[0], controls[1], ancillas[0])


def multi_control_circuit(n: Int) raises -> Circuit:
    """X, CX, CCX, ..., C^{n-1}X on qubits 0..n-1 (Benchpress multi_control_circuit).

    Gates with k > 2 controls are synthesised eagerly with a Toffoli V-chain on
    n extra clean ancillas (QUMO has no lazy multi-controlled opcode).
    """
    var c = Circuit(2 * n, 0, "multi_control")
    var anc = List[Int]()
    for a in range(n, 2 * n):
        anc.append(a)
    c.x(0)
    for k in range(1, n):
        var ctrls = List[Int]()
        for q in range(k):
            ctrls.append(q)
        multi_control_x(c, ctrls, k, anc)
    return c^


# ---------------------------------------------------------------------------
# Pauli twirling of 2q Clifford gates
# ---------------------------------------------------------------------------
def _pauli_kind(x: Int, z: Int) -> Int:
    if x == 1 and z == 1: return Op.Y
    if x == 1: return Op.X
    if z == 1: return Op.Z
    return Op.I


def pauli_twirl_2q(c: Circuit, seed: UInt64 = 12345) -> Circuit:
    """Twirl every CX/CZ with random Paulis P before and C P C^dagger after."""
    var out = c.empty_like()
    var rng = Rng(seed)
    for g in c.ops:
        if g.kind != Op.CX and g.kind != Op.CZ:
            out.append(g)
            continue
        var r = rng.below(16)
        var xa = r & 1
        var za = (r >> 1) & 1
        var xb = (r >> 2) & 1
        var zb = (r >> 3) & 1
        var pa = _pauli_kind(xa, za)
        var pb = _pauli_kind(xb, zb)
        if pa != Op.I: out.add(pa, g.q0)
        if pb != Op.I: out.add(pb, g.q1)
        out.append(g)
        # conjugate through the gate (signs only change global phase)
        var is_cx = g.kind == Op.CX
        var nxa = xa
        var nza = (za ^ zb) if is_cx else (za ^ xb)
        var nxb = (xb ^ xa) if is_cx else xb
        var nzb = zb if is_cx else (zb ^ xa)
        var qa = _pauli_kind(nxa, nza)
        var qb = _pauli_kind(nxb, nzb)
        if qa != Op.I: out.add(qa, g.q0)
        if qb != Op.I: out.add(qb, g.q1)
    return out^


# ---------------------------------------------------------------------------
# Hamiltonian simulation circuits (first-order Trotter, one step)
# ---------------------------------------------------------------------------
def pauli_evolution(mut c: Circuit, t: PauliTerm, angle: Float64):
    """exp(-i angle/2 P) via basis change + CX ladder + RZ."""
    var qs = List[Int]()
    for q in range(64):
        var b = UInt64(1) << UInt64(q)
        if ((t.x | t.z) & b) != 0:
            qs.append(q)
    if len(qs) == 0:
        c.global_phase -= angle / 2
        return
    for q in qs:
        var b = UInt64(1) << UInt64(q)
        var hx = (t.x & b) != 0
        var hz = (t.z & b) != 0
        if hx and hz:
            c.sdg(q)
            c.h(q)
        elif hx:
            c.h(q)
    for i in range(len(qs) - 1):
        c.cx(qs[i], qs[i + 1])
    c.rz(angle, qs[len(qs) - 1])
    var i = len(qs) - 2
    while i >= 0:
        c.cx(qs[i], qs[i + 1])
        i -= 1
    for q in qs:
        var b = UInt64(1) << UInt64(q)
        var hx = (t.x & b) != 0
        var hz = (t.z & b) != 0
        if hx and hz:
            c.h(q)
            c.s(q)
        elif hx:
            c.h(q)


def hamiltonian_circuit(h: PauliSum, num_qubits: Int, time: Float64 = 1.0) -> Circuit:
    """exp(-i t H) with one Lie-Trotter step (term order as given)."""
    var c = Circuit(num_qubits, 0, "hamiltonian")
    for t in h.terms:
        pauli_evolution(c, t, 2.0 * t.coeff * time)
    return c^


def hamiltonian_from_labels(labels: List[String], coeffs: List[Float64]) raises -> PauliSum:
    var h = PauliSum()
    for i in range(len(labels)):
        h.add_label(coeffs[i], labels[i])
    return h^


def hamiltonian_circuit_from_labels(labels: List[String], coeffs: List[Float64],
                                    time: Float64 = 1.0) raises -> Circuit:
    """exp(-i t H), H = sum_k c_k P_k given dense labels of any width
    (leftmost char = highest qubit, Qiskit SparsePauliOp convention)."""
    var n = 0
    if len(labels) > 0:
        n = labels[0].byte_length()
    var c = Circuit(n, 0, "hamiltonian")
    var qs = List[Int]()
    var kinds = List[Int]()
    for k in range(len(labels)):
        var bs = labels[k].as_bytes()
        if labels[k].byte_length() != n:
            raise Error("QM-E303: Pauli labels have different lengths.")
        qs.clear()
        kinds.clear()
        for i in range(n):
            var ch = Int(bs[i])
            if ch != ord("I"):
                qs.append(n - 1 - i)
                kinds.append(ch)
        var angle = 2.0 * coeffs[k] * time
        if len(qs) == 0:
            c.global_phase -= angle / 2
            continue
        # ascending qubit order for the ladder
        qs.reverse()
        kinds.reverse()
        for j in range(len(qs)):
            if kinds[j] == ord("Y"):
                c.sdg(qs[j])
                c.h(qs[j])
            elif kinds[j] == ord("X"):
                c.h(qs[j])
        for j in range(len(qs) - 1):
            c.cx(qs[j], qs[j + 1])
        c.rz(angle, qs[len(qs) - 1])
        var j = len(qs) - 2
        while j >= 0:
            c.cx(qs[j], qs[j + 1])
            j -= 1
        for j2 in range(len(qs)):
            if kinds[j2] == ord("Y"):
                c.h(qs[j2])
                c.s(qs[j2])
            elif kinds[j2] == ord("X"):
                c.h(qs[j2])
    return c^
