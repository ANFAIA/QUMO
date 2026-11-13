"""Stabilizer branching: near-Clifford circuits past the state-vector limit.

The idea of QDK 1.32's `BranchingState`: the state is C sum_x a_x |x>, one shared
Clifford frame C (an inverse tableau) and a sparse list of amplitudes. Clifford
gates change only the frame, whatever the number of branches. A non-Clifford
rotation exp(-i t/2 P) first maps P to P' = C^dag P C and then acts on the sparse
part, at most doubling it. So the cost is 2^(non-Clifford rotations that do not
commute), not 2^n: hundreds of qubits with a few T gates are cheap, and a generic
circuit with many T gates hits the limit of 2^20 branches (QM-E412).

Measurements project the sparse part. QDK's version keeps the projected pairs, so
a measurement can double the list; here the frame takes the projection instead:
with P' = C^dag Z_q C non-diagonal, the same column operations the tableau
simulator uses for a random outcome turn P' into a diagonal Pauli, each projected
pair collapses to one string, and measuring never grows the list.

Pauli noise and classical feed-forward run shot by shot as on the tableau.
"""

from std.bit import pop_count, count_trailing_zeros
from std.collections import Dict
from std.math import pi, cos, sin, sqrt
from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary, op_is_noise, op_is_annotation
from qumo.ir.circuit import Circuit, Instruction, check_shots
from qumo.sim.rng import Rng
from qumo.sim.stabilizer import TableauSimulator, bound_instruction
from qumo.sim.runner import SampleResult
from qumo.ir.gateset import GateSet
from qumo.ir.gateset import expand_rule
from max.algorithm import parallelize

comptime MAX_BRANCHES = 1 << 20
comptime AMPLITUDE_EPSILON = 1e-14
comptime KEY_WORDS = 8  # up to 512 qubits


@fieldwise_init
struct _Key(Hashable, Equatable, ImplicitlyCopyable):
    """A computational basis state, or an X / Z mask."""

    var w: SIMD[DType.uint64, KEY_WORDS]

    @staticmethod
    def zero() -> _Key:
        return _Key(SIMD[DType.uint64, KEY_WORDS](0))

    def bit(self, q: Int) -> Int:
        return Int((self.w[q >> 6] >> UInt64(q & 63)) & 1)

    def flip(mut self, q: Int):
        self.w[q >> 6] = self.w[q >> 6] ^ (UInt64(1) << UInt64(q & 63))

    def __eq__(self, o: _Key) -> Bool:
        return all(self.w.eq(o.w))

    def is_zero(self) -> Bool:
        return all(self.w.eq(0))

    def parity_with(self, o: _Key) -> Int:
        var c = 0
        var m = self.w & o.w
        for i in range(KEY_WORDS):
            c += Int(pop_count(m[i]))
        return c & 1


@fieldwise_init
struct _Pauli(ImplicitlyCopyable):
    """i^k X^x Z^z."""

    var k: Int
    var x: _Key
    var z: _Key

    def times(self, o: _Pauli) -> _Pauli:
        # X^a Z^b X^c Z^d = (-1)^(b.c) X^(a+c) Z^(b+d)
        var sign = self.z.parity_with(o.x)
        return _Pauli((self.k + o.k + 2 * sign) & 3, _Key(self.x.w ^ o.x.w), _Key(self.z.w ^ o.z.w))


struct BranchingSimulator(Movable):
    var frame: TableauSimulator
    var keys: List[_Key]
    var re: List[Float64]
    var im: List[Float64]
    var peak: Int

    def __init__(out self, num_qubits: Int, seed: UInt64 = 0) raises:
        if num_qubits > KEY_WORDS * 64:
            raise Error("QM-E413: stabilizer branching handles up to " + String(KEY_WORDS * 64) + " qubits, got " + String(num_qubits) + ". Hint: run Clifford circuits on the tableau simulator.")
        self.frame = TableauSimulator(num_qubits, seed)
        self.keys = List[_Key]()
        self.keys.append(_Key.zero())
        self.re = [1.0]
        self.im = [0.0]
        self.peak = 1

    def num_branches(self) -> Int:
        return len(self.keys)

    def _row(self, row: Int) -> _Pauli:
        """Tableau row as i^k X^x Z^z."""
        var nw = self.frame.nw
        var x = _Key.zero()
        var z = _Key.zero()
        for w in range(nw):
            x.w[w] = self.frame.xs[row * nw + w]
            z.w[w] = self.frame.zs[row * nw + w]
        var ny = 0
        var m = x.w & z.w
        for w in range(nw):
            ny += Int(pop_count(m[w]))
        # letters to XZ form: Y = i X Z
        return _Pauli((2 * Int(self.frame.signs[row]) + ny) & 3, x, z)

    def preimage(self, xmask: _Key, zmask: _Key) -> _Pauli:
        """C^dag P C for the Hermitian letter product with these masks."""
        var n = self.frame.n
        var out = _Pauli(0, _Key.zero(), _Key.zero())
        for q in range(n):
            var xb = xmask.bit(q)
            var zb = zmask.bit(q)
            if xb == 1:
                out = out.times(self._row(q))
            if zb == 1:
                out = out.times(self._row(n + q))
            if xb == 1 and zb == 1:
                out.k = (out.k + 1) & 3  # Y = i X Z
        return out

    def _apply_linear(mut self, cr: Float64, ci: Float64, pr: Float64, pi_: Float64, p: _Pauli) raises:
        """|phi> <- (c + d P)|phi>, c = cr + i ci, d = pr + i pi_."""
        var index = Dict[_Key, Int]()
        var nk = List[_Key]()
        var nr = List[Float64]()
        var ni = List[Float64]()

        @always_inline
        def put(key: _Key, ar: Float64, ai: Float64) {mut index, mut nk, mut nr, mut ni}:
            var at = index.get(key, -1)
            if at < 0:
                index[key] = len(nk)
                nk.append(key)
                nr.append(ar)
                ni.append(ai)
            else:
                nr[at] += ar
                ni[at] += ai

        # i^k
        var ir = 1.0 if p.k == 0 else (-1.0 if p.k == 2 else 0.0)
        var ii = 1.0 if p.k == 1 else (-1.0 if p.k == 3 else 0.0)
        var dr = pr * ir - pi_ * ii
        var di = pr * ii + pi_ * ir
        for j in range(len(self.keys)):
            var key = self.keys[j]
            var ar = self.re[j]
            var ai = self.im[j]
            put(key, cr * ar - ci * ai, cr * ai + ci * ar)
            var sign = -1.0 if p.z.parity_with(key) == 1 else 1.0
            var br = sign * (dr * ar - di * ai)
            var bi = sign * (dr * ai + di * ar)
            put(_Key(key.w ^ p.x.w), br, bi)
        self.keys = List[_Key]()
        self.re = List[Float64]()
        self.im = List[Float64]()
        for j in range(len(nk)):
            if nr[j] * nr[j] + ni[j] * ni[j] > AMPLITUDE_EPSILON * AMPLITUDE_EPSILON:
                self.keys.append(nk[j])
                self.re.append(nr[j])
                self.im.append(ni[j])
        if len(self.keys) > MAX_BRANCHES:
            raise Error("QM-E412: stabilizer branching passed " + String(MAX_BRANCHES) + " branches. Hint: the circuit has too many non-commuting non-Clifford gates; use the state vector, or fewer rotations.")
        self.peak = max(self.peak, len(self.keys))

    def _normalize(mut self):
        var s = 0.0
        for j in range(len(self.keys)):
            s += self.re[j] * self.re[j] + self.im[j] * self.im[j]
        var f = 1.0 / sqrt(s)
        for j in range(len(self.keys)):
            self.re[j] *= f
            self.im[j] *= f

    def rotate(mut self, theta: Float64, xmask: _Key, zmask: _Key) raises:
        """exp(-i theta/2 P) on the physical state."""
        var p = self.preimage(xmask, zmask)
        self._apply_linear(cos(theta / 2), 0.0, 0.0, -sin(theta / 2), p)
        self._normalize()

    def rotate_axis(mut self, axis: Int, q: Int, theta: Float64) raises:
        """1 X, 2 Y, 3 Z."""
        var x = _Key.zero()
        var z = _Key.zero()
        if axis == 1 or axis == 2:
            x.flip(q)
        if axis == 3 or axis == 2:
            z.flip(q)
        self.rotate(theta, x, z)

    def expectation_pauli(self, xmask: _Key, zmask: _Key) -> Float64:
        """<psi|P|psi> for the Hermitian letter product with these masks."""
        var p = self.preimage(xmask, zmask)
        return self._expect(p)

    def _expect(self, p: _Pauli) -> Float64:
        var index = Dict[_Key, Int]()
        for j in range(len(self.keys)):
            index[self.keys[j]] = j
        var ir = 1.0 if p.k == 0 else (-1.0 if p.k == 2 else 0.0)
        var ii = 1.0 if p.k == 1 else (-1.0 if p.k == 3 else 0.0)
        var total = 0.0
        for j in range(len(self.keys)):
            var at = index.get(_Key(self.keys[j].w ^ p.x.w), -1)
            if at < 0:
                continue
            var sign = -1.0 if p.z.parity_with(self.keys[j]) == 1 else 1.0
            # conj(a_at) i^k sign a_j
            var br = sign * (ir * self.re[j] - ii * self.im[j])
            var bi = sign * (ir * self.im[j] + ii * self.re[j])
            total += self.re[at] * br + self.im[at] * bi
        return total

    def measure(mut self, q: Int, mut rng: Rng) raises -> Int:
        """Z measurement; the list of branches never grows."""
        var n = self.frame.n
        var row = n + q
        var p = self._row(row)
        var e = self._expect(p)
        var p0 = max(0.0, min(1.0, (1.0 + e) / 2))
        var outcome = 0 if rng.uniform() < p0 else 1
        var prob = p0 if outcome == 0 else 1.0 - p0
        if prob < 1e-300:
            outcome = 1 - outcome
        # project: (1 +- P)/2 on the sparse part
        self._apply_linear(0.5, 0.0, 0.5 if outcome == 0 else -0.5, 0.0, p)
        self._normalize()
        if p.x.is_zero():
            return outcome
        # make P diagonal: the frame absorbs the projection, as in TableauSimulator.measure
        var pivot = -1
        for w in range(self.frame.nw):
            if p.x.w[w] != 0:
                pivot = w * 64 + Int(count_trailing_zeros(p.x.w[w]))
                break
        var fan = p.x
        fan.flip(pivot)
        for t in range(n):
            if fan.bit(t) == 1:
                self.frame._col_cx(pivot, t)
        var yy = p.z.bit(pivot) == 1
        if yy:
            self.frame._col_hyz(pivot)
        else:
            self.frame._col_h(pivot)
        # the same gates on the sparse part: |phi'> = G^dag |phi>
        var index = Dict[_Key, Int]()
        var nk = List[_Key]()
        var nr = List[Float64]()
        var ni = List[Float64]()
        comptime SQRT1_2 = 0.70710678118654752440
        for j in range(len(self.keys)):
            var key = self.keys[j]
            if key.bit(pivot) == 1:
                key = _Key(key.w ^ fan.w)
            var b = key.bit(pivot)
            var k0 = key
            var k1 = key
            if b == 0:
                k1.flip(pivot)
            else:
                k0.flip(pivot)
            var ar = self.re[j]
            var ai = self.im[j]
            # column of H or H_YZ = (Y + Z)/sqrt2 for input bit b
            var c0r: Float64
            var c0i: Float64
            var c1r: Float64
            var c1i: Float64
            if not yy:
                c0r = SQRT1_2
                c0i = 0.0
                c1r = SQRT1_2 if b == 0 else -SQRT1_2
                c1i = 0.0
            else:
                c0r = SQRT1_2 if b == 0 else 0.0
                c0i = 0.0 if b == 0 else -SQRT1_2
                c1r = 0.0 if b == 0 else -SQRT1_2
                c1i = SQRT1_2 if b == 0 else 0.0
            for s in range(2):
                var kk = k0 if s == 0 else k1
                var cr = c0r if s == 0 else c1r
                var ci = c0i if s == 0 else c1i
                var vr = cr * ar - ci * ai
                var vi = cr * ai + ci * ar
                var at = index.get(kk, -1)
                if at < 0:
                    index[kk] = len(nk)
                    nk.append(kk)
                    nr.append(vr)
                    ni.append(vi)
                else:
                    nr[at] += vr
                    ni[at] += vi
        self.keys = List[_Key]()
        self.re = List[Float64]()
        self.im = List[Float64]()
        for j in range(len(nk)):
            if nr[j] * nr[j] + ni[j] * ni[j] > AMPLITUDE_EPSILON * AMPLITUDE_EPSILON:
                self.keys.append(nk[j])
                self.re.append(nr[j])
                self.im.append(ni[j])
        self._normalize()
        return outcome

    def apply(mut self, g: Instruction, depth: Int = 0) raises:
        """One unitary gate: Clifford on the frame, the rest as Pauli rotations."""
        if g.is_clifford():
            self.frame.apply_gate(g)
            return
        var k = g.kind
        var v = g.p0.value()
        var a = g.q0
        var b = g.q1
        if k == Op.T:
            self.rotate_axis(3, a, pi / 4)
        elif k == Op.TDG:
            self.rotate_axis(3, a, -pi / 4)
        elif k == Op.RZ or k == Op.P:
            self.rotate_axis(3, a, v)
        elif k == Op.RX:
            self.rotate_axis(1, a, v)
        elif k == Op.RY:
            self.rotate_axis(2, a, v)
        elif k == Op.U:
            self.rotate_axis(3, a, g.p2.value())
            self.rotate_axis(2, a, v)
            self.rotate_axis(3, a, g.p1.value())
        elif k == Op.RZZ or k == Op.RXX or k == Op.RYY:
            var x = _Key.zero()
            var z = _Key.zero()
            if k != Op.RZZ:
                x.flip(a)
                x.flip(b)
            if k != Op.RXX:
                z.flip(a)
                z.flip(b)
            self.rotate(v, x, z)
        elif k == Op.CP or k == Op.CRZ:
            var x = _Key.zero()
            var z = _Key.zero()
            z.flip(a)
            z.flip(b)
            if k == Op.CP:
                self.rotate_axis(3, a, v / 2)
            self.rotate_axis(3, b, v / 2)
            self.rotate(-v / 2, x, z)
        elif depth < 4:
            var seq = List[Instruction]()
            var dropped = False
            var anyset = GateSet()
            _ = expand_rule(g, anyset, seq, dropped)
            if len(seq) == 0:
                raise Error("QM-E413: stabilizer branching has no decomposition for '" + op_name(k) + "'. Hint: translate to Clifford gates and rotations first.")
            for s in seq:
                self.apply(s, depth + 1)
        else:
            raise Error("QM-E413: stabilizer branching has no decomposition for '" + op_name(k) + "'. Hint: translate to Clifford gates and rotations first.")

    def run(mut self, c: Circuit, mut rng: Rng, noisy: Bool = True) raises:
        var n_clbits = c.num_clbits
        while len(self.frame.creg) < n_clbits:
            self.frame.creg.append(0)
        self.run_range(c, 0, len(c.ops), rng, noisy, List[Float64]())

    def run_range(mut self, c: Circuit, lo: Int, hi: Int, mut rng: Rng, noisy: Bool, params: List[Float64]) raises:
        """Execute ops [lo, hi); a non-empty `params` binds symbolic gates as they run."""
        var bind = len(params) > 0
        for i in range(lo, hi):
            var g = c.ops[i]
            if g.cond >= 0 and not self.frame._condition_holds(c, g):
                if g.kind == Op.MEASURE:
                    self.frame.records.append(0)  # records are indexed by measurement site
                continue
            if bind:
                g = bound_instruction(g, params)
            var k = g.kind
            if op_is_unitary(k):
                if g.is_parametric():
                    raise Error("QM-E413: stabilizer branching needs bound parameters. Hint: call bind() first.")
                self.apply(g)
            elif k == Op.MEASURE:
                var m = self.measure(g.q0, rng)
                self.frame.records.append(UInt8(m))
                if g.clbit >= 0:
                    self.frame.creg[g.clbit] = UInt8(m)
            elif k == Op.RESET:
                if self.measure(g.q0, rng) == 1:
                    self.frame.x(g.q0)
            elif op_is_noise(k):
                if noisy:
                    self.frame.apply_noise_in(c, g)


def sample_branching(c: Circuit, shots: Int, seed: UInt64 = 1234) raises -> SampleResult:
    """Shot-by-shot sampling on the branching simulator (any qubit count up to 512)."""
    check_shots(shots)
    c.validate()
    if c.num_measurements == 0:
        raise Error("QM-E402: circuit has no measurements. Hint: call measure_all() or use evolve()/expectation().")
    if c.num_clbits > 62:
        raise Error("QM-E406: the sampler packs classical registers into 62 bits; circuit has " + String(c.num_clbits) + " clbits. Hint: measure fewer bits or split the register.")
    if c.has_non_pauli_noise():
        raise Error("QM-E414: stabilizer branching takes Pauli noise only. Hint: use density_evolve() or sample().")
    var res = SampleResult(c.num_clbits, shots, "stabilizer:branching")
    var cregs = List[Int](length=shots, fill=0)
    var errors = List[String](length=shots, fill=String(""))
    var cp = cregs.unsafe_ptr().as_unsafe_any_origin()
    var ep = errors.unsafe_ptr().as_unsafe_any_origin()
    var base = Rng(seed)

    def shot(s: Int) {imm c, imm cp, imm ep, imm base}:
        try:
            var rng = base.split(s)
            var sim = BranchingSimulator(c.num_qubits, rng.next_u64())
            sim.run(c, rng)
            var creg = 0
            for b in range(c.num_clbits):
                if sim.frame.creg[b] != 0:
                    creg |= 1 << b
            cp[unsafe_offset=s] = creg
        except e:
            ep[unsafe_offset=s] = String(e)

    parallelize(shot, shots)
    _ = len(cregs)
    _ = len(errors)
    for s in range(shots):
        if errors[s].byte_length() > 0:
            raise Error(errors[s])
        res.record(cregs[s])
    return res^
