"""Pauli propagation: expectation values in the Heisenberg picture, up to 64 qubits.

`propagate_pauli(circuit, observable, options)` walks the circuit from the last
instruction to the first and replaces the observable O by G^dag O G at each gate,
so at the end it holds U^dag O U as a sum of Pauli strings, and
<0^n| U^dag O U |0^n> is the sum of the coefficients of its Z/I-only strings.
The method is the one of Rudolph, Jones, Teng, Angrisani and Holmes, "Pauli
Propagation: A Computational Framework for Simulating Quantum Systems"
(arXiv:2505.21606, Sec. II.2 and III; the PauliPropagation.jl package).

What each instruction does to a Pauli string P (coefficient c):

* Clifford opcode (I, X, Y, Z, H, S, SDG, SX, SXDG, CX, CY, CZ, SWAP, ISWAP,
  ISWAPDG, ECR): one string to one string with a sign, from the shared table
  `qumo.ir.clifford_table.clifford_heisenberg`.
* Pauli rotation exp(-i theta G / 2): P is kept if it commutes with G; otherwise
  c P -> c cos(theta) P + c sin(theta) (-i P G), two strings.
* Other gates are written exactly as Clifford gates and Pauli rotations (global
  phase dropped, it cancels in U^dag O U): T, TDG, RX, RY, RZ, P, U (RZ RY RZ),
  RXX, RYY, RZZ, CP, CRX, CRY, CRZ (rotations about Z_a, Z_b, Z_a Z_b, X_b,
  Z_a X_b, ...), CH (RY(-pi/4), CZ, RY(pi/4) on the target), CCZ (seven
  commuting Z-string rotations), CCX (H CCZ H), CSWAP (CX CCX CX).
* Pauli noise, exactly, as the adjoint channel: a string is multiplied by
  1 - 2 (total probability of the Paulis that anticommute with it). Taken:
  X_ERROR, Y_ERROR, Z_ERROR, PAULI_CHANNEL_1, DEPOLARIZE1 (p/3 each),
  DEPOLARIZE2 (p/15 each). With noise the result is Tr(O E(|0><0|)).
* Barriers, ticks, detectors, observables: skipped. Terminal measurements are
  ignored, as `expectation()` does: the value is that of the state before them.

Refused with QM-E430: more than 64 qubits, classically conditioned gates,
resets, mid-circuit measurements (a gate or noise on a measured qubit),
Kraus channels, PAULI_CHANNEL_2 and correlated errors (E / ELSE).

Truncation (`PropagationOptions`), applied after every instruction:
`min_abs_coeff` drops strings whose |c| is below it, `max_weight` drops strings
acting on more qubits than it. Both are off by default; with both off nothing is
dropped except strings whose coefficient is exactly 0.0, and the result is exact
up to floating-point rounding. `max_terms` is a hard cap, not a truncation: when
more strings than that survive an instruction, the run stops with QM-E431. A
rotation can double the list before it is truncated: with n strings held, the
peak is the 2n strings (24 bytes each), a buffer of the n new ones and a hash
index of 4n to 8n slots (8 bytes each), at most about 140 bytes per string of
max_terms (some 600 MB at the default 2^22).

What the reported error means. `discarded_abs_sum` is the sum of |c| over every
string dropped, at every step. Each dropped piece D_k would have contributed
Tr(rho_k D_k) to the answer, where rho_k is the (possibly noisy) state at that
point of the circuit, and |Tr(rho D)| <= ||D|| <= sum |c|; unitaries and Pauli
channels do not increase the operator norm. So, in exact arithmetic,

    |exact expectation - reported expectation| <= discarded_abs_sum

for this circuit (this is the worst-case bound of arXiv:2505.21606, Theory
Box 3). The argument does not use the initial state, so the same number bounds
|Tr(rho U^dag O U) - Tr(rho O')| for the returned operator O' in any state rho
(and, with noise, Tr(E^dag(O) rho)). It is an upper bound, not an estimate: the
actual error is often far smaller (dropped strings with X or Y factors have no
overlap with |0^n>, and signs cancel); tests/test_pauli_prop.mojo measures it at
about 1.6% of the bound on average on its random circuits. It does not cover
floating-point rounding (of order 1e-16 times the l1 norm of the kept
coefficients), and it is not a bound on the operator error in other norms.
`discarded_terms` counts the dropped strings; a run with none is exact.
"""

from std.bit import pop_count
from std.math import cos, sin, pi
from qumo.ir.circuit import Circuit, Instruction, check_finite_values
from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary, op_is_noise
from qumo.ir.pauli import PauliSum, PauliTerm
from qumo.ir.clifford_table import clifford_heisenberg

comptime PP_MAX_QUBITS = 64
comptime PP_DEFAULT_MAX_TERMS = 1 << 22


struct PropagationOptions(ImplicitlyCopyable):
    """Truncation settings. Defaults: no truncation, a cap of 2^22 strings."""

    var min_abs_coeff: Float64  # drop |c| < this; 0 = off
    var max_weight: Int  # drop weight > this; -1 = off
    var max_terms: Int  # hard cap: QM-E431 above it

    def __init__(out self, min_abs_coeff: Float64 = 0.0, max_weight: Int = -1, max_terms: Int = PP_DEFAULT_MAX_TERMS):
        self.min_abs_coeff = min_abs_coeff
        self.max_weight = max_weight
        self.max_terms = max_terms

    def truncates(self) -> Bool:
        return self.min_abs_coeff > 0.0 or self.max_weight >= 0


struct PropagationResult(Copyable, Writable):
    var expectation: Float64  # <0^n| (propagated observable) |0^n>
    var observable: PauliSum  # U^dag O U after truncation
    var discarded_abs_sum: Float64  # sum |c| of every dropped string: bounds the error
    var discarded_terms: Int  # number of strings dropped (exact zeros not counted)
    var peak_terms: Int  # most strings held after any instruction

    def __init__(out self):
        self.expectation = 0.0
        self.observable = PauliSum()
        self.discarded_abs_sum = 0.0
        self.discarded_terms = 0
        self.peak_terms = 0

    def is_exact(self) -> Bool:
        """True when no string was dropped (exact up to rounding)."""
        return self.discarded_terms == 0

    def write_to(self, mut w: Some[Writer]):
        w.write("PropagationResult(expectation=", self.expectation, ", terms=", len(self.observable))
        w.write(", peak_terms=", self.peak_terms, ", discarded_terms=", self.discarded_terms)
        w.write(", error_bound=", self.discarded_abs_sum, ")")


# ----------------------------------------------------------------------------
# Steps: a gate as Clifford opcodes and Pauli rotations, in circuit order
# ----------------------------------------------------------------------------
@fieldwise_init
struct _Step(ImplicitlyCopyable):
    var rot: Bool  # True: exp(-i theta G / 2) with G = (gx, gz); False: Clifford opcode
    var kind: Int
    var a: Int
    var b: Int
    var gx: UInt64
    var gz: UInt64
    var theta: Float64


def _bit(q: Int) -> UInt64:
    return UInt64(1) << UInt64(q)


def _cliff(mut s: List[_Step], kind: Int, a: Int, b: Int = -1):
    s.append(_Step(False, kind, a, b, 0, 0, 0.0))


def _rot(mut s: List[_Step], gx: UInt64, gz: UInt64, theta: Float64):
    s.append(_Step(True, 0, -1, -1, gx, gz, theta))


def _ccz_steps(mut s: List[_Step], a: Int, b: Int, c: Int):
    """exp(i pi |111><111|) = exp(i pi/8 (I - sum Z + sum ZZ - ZZZ))."""
    var za = _bit(a)
    var zb = _bit(b)
    var zc = _bit(c)
    _rot(s, 0, za, pi / 4)
    _rot(s, 0, zb, pi / 4)
    _rot(s, 0, zc, pi / 4)
    _rot(s, 0, za | zb, -pi / 4)
    _rot(s, 0, za | zc, -pi / 4)
    _rot(s, 0, zb | zc, -pi / 4)
    _rot(s, 0, za | zb | zc, pi / 4)


def _steps(g: Instruction, params: List[Float64], index: Int) raises -> List[_Step]:
    var s = List[_Step]()
    var k = g.kind
    var a = g.q0
    var b = g.q1
    var t0 = g.p0.resolve(params)
    var ba = _bit(a)
    if k == Op.I or k == Op.X or k == Op.Y or k == Op.Z or k == Op.H or k == Op.S or k == Op.SDG or k == Op.SX or k == Op.SXDG:
        _cliff(s, k, a)
    elif k == Op.CX or k == Op.CY or k == Op.CZ or k == Op.SWAP or k == Op.ISWAP or k == Op.ISWAPDG or k == Op.ECR:
        _cliff(s, k, a, b)
    elif k == Op.T:
        _rot(s, 0, ba, pi / 4)
    elif k == Op.TDG:
        _rot(s, 0, ba, -pi / 4)
    elif k == Op.RX:
        _rot(s, ba, 0, t0)
    elif k == Op.RY:
        _rot(s, ba, ba, t0)
    elif k == Op.RZ or k == Op.P:
        _rot(s, 0, ba, t0)
    elif k == Op.U:
        # U3(theta, phi, lam) = e^{i(phi+lam)/2} RZ(phi) RY(theta) RZ(lam)
        _rot(s, 0, ba, g.p2.resolve(params))
        _rot(s, ba, ba, t0)
        _rot(s, 0, ba, g.p1.resolve(params))
    else:
        var bb = _bit(b)
        if k == Op.RXX:
            _rot(s, ba | bb, 0, t0)
        elif k == Op.RYY:
            _rot(s, ba | bb, ba | bb, t0)
        elif k == Op.RZZ:
            _rot(s, 0, ba | bb, t0)
        elif k == Op.CP:
            _rot(s, 0, ba, t0 / 2)
            _rot(s, 0, bb, t0 / 2)
            _rot(s, 0, ba | bb, -t0 / 2)
        elif k == Op.CRZ:
            _rot(s, 0, bb, t0 / 2)
            _rot(s, 0, ba | bb, -t0 / 2)
        elif k == Op.CRX:
            _rot(s, bb, 0, t0 / 2)
            _rot(s, bb, ba, -t0 / 2)
        elif k == Op.CRY:
            _rot(s, bb, bb, t0 / 2)
            _rot(s, bb, ba | bb, -t0 / 2)
        elif k == Op.CH:
            _rot(s, bb, bb, -pi / 4)
            _cliff(s, Op.CZ, a, b)
            _rot(s, bb, bb, pi / 4)
        elif k == Op.CCZ:
            _ccz_steps(s, a, b, g.q2)
        elif k == Op.CCX:
            _cliff(s, Op.H, g.q2)
            _ccz_steps(s, a, b, g.q2)
            _cliff(s, Op.H, g.q2)
        elif k == Op.CSWAP:
            # cswap c, a, b = cx b, a; ccx c, a, b; cx b, a  (c = q0)
            var c = g.q0
            var x = g.q1
            var y = g.q2
            _cliff(s, Op.CX, y, x)
            _cliff(s, Op.H, y)
            _ccz_steps(s, c, x, y)
            _cliff(s, Op.H, y)
            _cliff(s, Op.CX, y, x)
        else:
            raise _unsupported(g, index, "no decomposition into Clifford gates and Pauli rotations")
    return s^


def _unsupported(g: Instruction, index: Int, why: String) -> Error:
    return Error("QM-E430: Pauli propagation cannot take instruction #" + String(index) + " '" + op_name(g.kind) + "': " + why + ". Hint: it takes unitary gates, terminal measurements and X/Y/Z_ERROR, PAULI_CHANNEL_1, DEPOLARIZE1/2 noise on up to 64 qubits; use density_evolve() or sample() for the rest.")


# ----------------------------------------------------------------------------
# The term list
# ----------------------------------------------------------------------------
struct _Terms(Movable, Sized):
    var xs: List[UInt64]
    var zs: List[UInt64]
    var cs: List[Float64]
    var slots: List[Int]  # open-addressing index, -1 = empty

    def __init__(out self):
        self.xs = List[UInt64]()
        self.zs = List[UInt64]()
        self.cs = List[Float64]()
        self.slots = List[Int]()

    def __len__(self) -> Int:
        return len(self.cs)


def _hash(x: UInt64, z: UInt64) -> UInt64:
    var h = x * UInt64(0x9E3779B97F4A7C15) ^ (z + UInt64(0x632BE59BD9B4E019))
    h ^= h >> 31
    h *= UInt64(0xBF58476D1CE4E5B9)
    h ^= h >> 29
    return h


def _rebuild_index(mut t: _Terms, need: Int):
    var cap = 16
    while cap < 2 * need:
        cap *= 2
    t.slots = List[Int](length=cap, fill=-1)
    var mask = UInt64(cap - 1)
    for i in range(len(t.cs)):
        var h = _hash(t.xs[i], t.zs[i]) & mask
        while t.slots[Int(h)] >= 0:
            h = (h + 1) & mask
        t.slots[Int(h)] = i


def _add_merge(mut t: _Terms, x: UInt64, z: UInt64, c: Float64):
    """Add c to string (x, z), appending it if absent (index has room)."""
    var mask = UInt64(len(t.slots) - 1)
    var h = _hash(x, z) & mask
    while True:
        var j = t.slots[Int(h)]
        if j < 0:
            t.slots[Int(h)] = len(t.cs)
            t.xs.append(x)
            t.zs.append(z)
            t.cs.append(c)
            return
        if t.xs[j] == x and t.zs[j] == z:
            t.cs[j] += c
            return
        h = (h + 1) & mask


def _pc(v: UInt64) -> Int:
    return Int(pop_count(v))


def _cos_sin(theta: Float64) -> Tuple[Float64, Float64]:
    """cos and sin, exactly 0 / +-1 when theta is a multiple of pi/2 to rounding."""
    var r = theta / (pi / 2)
    var n = round(r)
    if abs(n) < 4194304.0 and abs(theta - n * (pi / 2)) <= 4e-16 * max(1.0, abs(theta)):
        var m = Int(n) & 3
        if m == 0: return (1.0, 0.0)
        if m == 1: return (0.0, 1.0)
        if m == 2: return (-1.0, 0.0)
        return (0.0, -1.0)
    return (cos(theta), sin(theta))


def _apply_rotation(mut t: _Terms, gx: UInt64, gz: UInt64, theta: Float64):
    """Heisenberg image of exp(-i theta G/2): P -> cos P + sin (-i P G) on anticommuting P."""
    var cssn = _cos_sin(theta)
    var co = cssn[0]
    var si = cssn[1]
    var n = len(t.cs)
    var nx = List[UInt64]()
    var nz = List[UInt64]()
    var nc = List[Float64]()
    var ag = _pc(gx & gz)
    for i in range(n):
        var x = t.xs[i]
        var z = t.zs[i]
        if ((_pc(x & gz) + _pc(z & gx)) & 1) == 0:
            continue
        var c = t.cs[i]
        t.cs[i] = c * co
        if si == 0.0:
            continue
        var x3 = x ^ gx
        var z3 = z ^ gz
        # P G = i^m P3; -i P G = i^(m-1) P3, m odd
        var m = _pc(x & z) + ag + 2 * _pc(z & gx) - _pc(x3 & z3)
        var sgn = 1.0 if ((m - 1) & 3) == 0 else -1.0
        nx.append(x3)
        nz.append(z3)
        nc.append(c * si * sgn)
    if len(nc) == 0:
        return
    _rebuild_index(t, n + len(nc))
    for j in range(len(nc)):
        _add_merge(t, nx[j], nz[j], nc[j])


def _pc2(v: Int) -> Int:
    return (v & 1) + ((v >> 1) & 1)


def _apply_clifford(mut t: _Terms, kind: Int, a: Int, b: Int) raises:
    """Heisenberg image G^dag P G through a 16-entry table built from clifford_heisenberg."""
    var two = b >= 0
    # images of X_a, Z_a, X_b, Z_b as (u, v, k): local bits (bit 0 = a, bit 1 = b)
    var iu = List[Int](length=4, fill=0)
    var iv = List[Int](length=4, fill=0)
    var ik = List[Int](length=4, fill=0)
    var nimg = 4 if two else 2
    for idx in range(nimg):
        var xa = 0
        var za = 0
        var xb = 0
        var zb = 0
        var k = 0
        if not clifford_heisenberg(kind, idx, xa, za, xb, zb, k):
            raise Error("QM-E430: Pauli propagation has no Clifford table entry for '" + op_name(kind) + "'. Hint: report it; the decomposition named a gate the table does not hold.")
        iu[idx] = xa | (xb << 1)
        iv[idx] = za | (zb << 1)
        ik[idx] = k
    var out_u = List[Int](length=16, fill=0)
    var out_v = List[Int](length=16, fill=0)
    var out_s = List[Float64](length=16, fill=1.0)
    for p in range(16):
        if not two and p >= 4:
            break
        var kk = (p & 1) & ((p >> 1) & 1)
        kk += ((p >> 2) & 1) & ((p >> 3) & 1)
        var u = 0
        var v = 0
        # X_a^xa Z_a^za X_b^xb Z_b^zb, images multiplied in this order
        for idx in range(nimg):
            if ((p >> idx) & 1) == 0:
                continue
            kk += ik[idx] + 2 * _pc2(v & iu[idx])
            u ^= iu[idx]
            v ^= iv[idx]
        var e = kk - _pc2(u & v)
        out_u[p] = u
        out_v[p] = v
        out_s[p] = -1.0 if (e & 2) != 0 else 1.0
    var ba = UInt64(a)
    var bb = UInt64(b) if two else UInt64(0)
    var clear = ~(_bit(a) | (_bit(b) if two else UInt64(0)))
    for i in range(len(t.cs)):
        var x = t.xs[i]
        var z = t.zs[i]
        var p = Int((x >> ba) & 1) | (Int((z >> ba) & 1) << 1)
        if two:
            p |= (Int((x >> bb) & 1) << 2) | (Int((z >> bb) & 1) << 3)
        if p == 0:
            continue
        var u = out_u[p]
        var v = out_v[p]
        var nxv = (x & clear) | (UInt64(u & 1) << ba)
        var nzv = (z & clear) | (UInt64(v & 1) << ba)
        if two:
            nxv |= UInt64((u >> 1) & 1) << bb
            nzv |= UInt64((v >> 1) & 1) << bb
        t.xs[i] = nxv
        t.zs[i] = nzv
        t.cs[i] *= out_s[p]


def _check_prob(p: Float64, g: Instruction, index: Int) raises:
    if not (p >= 0.0 and p <= 1.0):
        raise Error("QM-E450: instruction #" + String(index) + " '" + op_name(g.kind) + "' has probability " + String(p) + ". Hint: probabilities must lie in [0, 1].")


def _apply_noise(mut t: _Terms, g: Instruction, params: List[Float64], index: Int) raises:
    """Adjoint of a Pauli channel: each string times 1 - 2 * P(anticommuting Pauli)."""
    var k = g.kind
    var a = g.q0
    var ba = UInt64(a)
    var p = g.p0.resolve(params)
    _check_prob(p, g, index)
    var px = 0.0
    var py = 0.0
    var pz = 0.0
    if k == Op.X_ERROR: px = p
    elif k == Op.Y_ERROR: py = p
    elif k == Op.Z_ERROR: pz = p
    elif k == Op.DEPOLARIZE1:
        px = p / 3.0; py = p / 3.0; pz = p / 3.0
    elif k == Op.PAULI_CHANNEL_1:
        px = p
        py = g.p1.resolve(params)
        pz = g.p2.resolve(params)
        _check_prob(py, g, index)
        _check_prob(pz, g, index)
        _check_prob(px + py + pz, g, index)
    if k == Op.DEPOLARIZE2:
        var mask = _bit(a) | _bit(g.q1)
        var f2 = 1.0 - 16.0 * p / 15.0
        for i in range(len(t.cs)):
            if ((t.xs[i] | t.zs[i]) & mask) != 0:
                t.cs[i] *= f2
        return
    # factor for local X, Y, Z on qubit a
    var fx = 1.0 - 2.0 * (py + pz)
    var fy = 1.0 - 2.0 * (px + pz)
    var fz = 1.0 - 2.0 * (px + py)
    for i in range(len(t.cs)):
        var hx = (t.xs[i] >> ba) & 1
        var hz = (t.zs[i] >> ba) & 1
        if hx == 1 and hz == 1: t.cs[i] *= fy
        elif hx == 1: t.cs[i] *= fx
        elif hz == 1: t.cs[i] *= fz


def _compact(mut t: _Terms, opts: PropagationOptions, mut res: PropagationResult):
    var w = 0
    for i in range(len(t.cs)):
        var c = t.cs[i]
        if c == 0.0:
            continue
        var drop = abs(c) < opts.min_abs_coeff
        if not drop and opts.max_weight >= 0:
            drop = _pc(t.xs[i] | t.zs[i]) > opts.max_weight
        if drop:
            res.discarded_abs_sum += abs(c)
            res.discarded_terms += 1
            continue
        t.xs[w] = t.xs[i]
        t.zs[w] = t.zs[i]
        t.cs[w] = c
        w += 1
    t.xs.resize(w, 0)
    t.zs.resize(w, 0)
    t.cs.resize(w, 0.0)


def _check_circuit(c: Circuit) raises:
    """Refuse what Pauli propagation does not take, before any work."""
    if c.num_qubits > PP_MAX_QUBITS:
        raise Error("QM-E430: Pauli propagation takes at most " + String(PP_MAX_QUBITS) + " qubits; the circuit has " + String(c.num_qubits) + ". Hint: Pauli strings are bit-packed into 64 qubits; use the stabilizer simulator for wider Clifford circuits.")
    var measured = List[Bool](length=c.num_qubits, fill=False)
    for i in range(len(c.ops)):
        ref g = c.ops[i]
        var k = g.kind
        if k == Op.BARRIER or k == Op.TICK or k == Op.DETECTOR or k == Op.OBSERVABLE:
            continue
        if g.cond >= 0:
            raise _unsupported(g, i, "it is classically conditioned")
        if k == Op.MEASURE:
            measured[g.q0] = True
            continue
        if k == Op.RESET:
            raise _unsupported(g, i, "resets are not taken")
        var ok = op_is_unitary(k) or k == Op.X_ERROR or k == Op.Y_ERROR or k == Op.Z_ERROR or k == Op.DEPOLARIZE1 or k == Op.DEPOLARIZE2 or k == Op.PAULI_CHANNEL_1
        if not ok:
            if op_is_noise(k):
                raise _unsupported(g, i, "only X/Y/Z_ERROR, PAULI_CHANNEL_1 and DEPOLARIZE1/2 noise is taken")
            raise _unsupported(g, i, "not a unitary gate or a Pauli channel")
        for j in range(op_arity(k)):
            if measured[g.qubit(j)]:
                raise _unsupported(g, i, "it acts on qubit " + String(g.qubit(j)) + " after that qubit was measured (mid-circuit measurement)")


def propagate_pauli(
    circuit: Circuit,
    observable: PauliSum,
    options: PropagationOptions = PropagationOptions(),
    params: List[Float64] = List[Float64](),
) raises -> PropagationResult:
    """U^dag O U by Pauli propagation and its expectation in |0^n>, with the error bound.

    The result holds the expectation, the propagated observable, and
    `discarded_abs_sum`, an upper bound on |exact - expectation| (module doc).
    """
    circuit.validate()
    if len(params) != circuit.num_params():
        raise Error("QM-E101: circuit has " + String(circuit.num_params()) + " parameters " + circuit.param_names_str() + " but " + String(len(params)) + " values were given.")
    check_finite_values(params)
    _check_circuit(circuit)
    if options.max_terms < 1:
        raise Error("QM-E431: max_terms = " + String(options.max_terms) + " leaves no room for the observable. Hint: pass max_terms >= 1 (the default is 2^22).")
    if observable.num_qubits() > circuit.num_qubits:
        raise Error("QM-E405: the observable acts on qubit " + String(observable.num_qubits() - 1) + " but the circuit has " + String(circuit.num_qubits) + " qubits. Hint: Pauli indices are 0-based; widen the circuit or fix the observable.")
    var t = _Terms()
    for ref term in observable.terms:
        t.xs.append(term.x)
        t.zs.append(term.z)
        t.cs.append(term.coeff)
    # merge repeated strings of the input (PauliSum.add merges, but the list is public)
    var xs0 = t.xs.copy()
    var zs0 = t.zs.copy()
    var cs0 = t.cs.copy()
    t.xs.clear(); t.zs.clear(); t.cs.clear()
    _rebuild_index(t, len(cs0))
    for i in range(len(cs0)):
        _add_merge(t, xs0[i], zs0[i], cs0[i])
    var res = PropagationResult()
    _compact(t, options, res)
    res.peak_terms = len(t)
    var i = len(circuit.ops) - 1
    while i >= 0:
        ref g = circuit.ops[i]
        var k = g.kind
        if op_is_unitary(k):
            var steps = _steps(g, params, i)
            var j = len(steps) - 1
            while j >= 0:
                var s = steps[j]
                if s.rot:
                    _apply_rotation(t, s.gx, s.gz, s.theta)
                else:
                    _apply_clifford(t, s.kind, s.a, s.b)
                _compact(t, options, res)
                if len(t) > options.max_terms:
                    raise Error("QM-E431: Pauli propagation holds " + String(len(t)) + " Pauli strings after instruction #" + String(i) + " '" + op_name(k) + "', above max_terms = " + String(options.max_terms) + ". Hint: truncate (min_abs_coeff, max_weight) and read the reported error bound, or raise max_terms if memory allows (up to about 140 bytes per string of max_terms at the peak).")
                j -= 1
        elif op_is_noise(k):
            _apply_noise(t, g, params, i)
            _compact(t, options, res)
        res.peak_terms = max(res.peak_terms, len(t))
        i -= 1
    var e = 0.0
    for j in range(len(t.cs)):
        if t.xs[j] == 0:
            e += t.cs[j]
        res.observable.terms.append(PauliTerm(t.cs[j], t.xs[j], t.zs[j]))
    res.expectation = e
    return res^
