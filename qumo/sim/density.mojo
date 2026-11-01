"""Density-matrix simulator: exact noise, any Kraus channel.

rho is stored as a state vector on 2n qubits: bit q of the index is the row bit
of qubit q and bit q + n its column bit, so entry i + 2^n j is rho_ij. Then
U rho U^dag is U on qubit q and conj(U) on qubit q + n, which the state-vector
kernels do directly. A channel sum_k K_k rho K_k^dag is one dense matrix
S = sum_k K_k (x) conj(K_k) on those bits, so its cost does not grow with the
number of Kraus operators.

Exact where trajectories are statistical, at 4^n cost: 16 bytes x 4^n, and at
most 15 qubits (16 GiB; QM-E410). Mid-circuit measurements are non-selective
(the qubit dephases). A single classical branch has no meaning in rho, so
classical conditions raise QM-E411; `sample()` runs those.
"""

from std.math import sqrt
from qumo.ir.gates import Op, M2, M4, op_arity, op_is_unitary, op_is_noise, op_is_annotation, matrix_1q, matrix_2q
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.kraus import KrausChannel
from qumo.ir.pauli import PauliSum
from qumo.sim.statevector import StateVector
from qumo.sim.rng import Rng
from qumo.sim.runner import apply_instruction, gate_matrix_1q, SampleResult, terminal_measure_map

comptime MAX_DM_QUBITS = 15


def check_density_size(n: Int) raises:
    if n > MAX_DM_QUBITS:
        raise Error("QM-E410: a density matrix on " + String(n) + " qubits needs 4^" + String(n) + " x 16 bytes; the limit is " + String(MAX_DM_QUBITS) + " qubits (16 GiB). Hint: sample_ptsbe() or sample() take Pauli noise on more qubits.")


def _pauli_1q(which: Int) -> M2:
    if which == 1:
        return matrix_1q(Op.X)
    if which == 2:
        return matrix_1q(Op.Y)
    if which == 3:
        return matrix_1q(Op.Z)
    return matrix_1q(Op.I)


def _kron_channel_1q(mut ch: KrausChannel, weight: Float64, m: M2):
    """Add sqrt(weight) m as a Kraus operator."""
    var w = sqrt(weight)
    ch.add([w * m.re[0], w * m.re[1], w * m.re[2], w * m.re[3]],
           [w * m.im[0], w * m.im[1], w * m.im[2], w * m.im[3]])


def _pauli_channel_1q(px: Float64, py: Float64, pz: Float64) -> KrausChannel:
    var ch = KrausChannel("pauli", 1)
    _kron_channel_1q(ch, 1.0 - px - py - pz, _pauli_1q(0))
    if px > 0.0:
        _kron_channel_1q(ch, px, _pauli_1q(1))
    if py > 0.0:
        _kron_channel_1q(ch, py, _pauli_1q(2))
    if pz > 0.0:
        _kron_channel_1q(ch, pz, _pauli_1q(3))
    return ch^


def _depolarize_2q(p: Float64) -> KrausChannel:
    """(1 - p) rho + p/15 sum_{P != II} P rho P; first letter on q0 (low bit)."""
    var ch = KrausChannel("depolarize2", 2)
    for r in range(16):
        var w = sqrt(1.0 - p if r == 0 else p / 15.0)
        var a = _pauli_1q(r & 3)
        var b = _pauli_1q(r >> 2)
        var re = List[Float64](length=16, fill=0.0)
        var im = List[Float64](length=16, fill=0.0)
        for i in range(4):
            for j in range(4):
                # (b (x) a): index = b(q0) + 2 b(q1)
                var ar = a.re[2 * (i & 1) + (j & 1)]
                var ai = a.im[2 * (i & 1) + (j & 1)]
                var br = b.re[2 * (i >> 1) + (j >> 1)]
                var bi = b.im[2 * (i >> 1) + (j >> 1)]
                re[4 * i + j] = w * (ar * br - ai * bi)
                im[4 * i + j] = w * (ar * bi + ai * br)
        ch.add(re^, im^)
    return ch^


def _measure_channel() -> KrausChannel:
    var ch = KrausChannel("dephase", 1)
    ch.add_real([1.0, 0.0, 0.0, 0.0])
    ch.add_real([0.0, 0.0, 0.0, 1.0])
    return ch^


def _reset_channel() -> KrausChannel:
    var ch = KrausChannel("reset", 1)
    ch.add_real([1.0, 0.0, 0.0, 0.0])
    ch.add_real([0.0, 1.0, 0.0, 0.0])
    return ch^


def _conj(m: M2) -> M2:
    return M2(m.re, -m.im)


def _conj4(m: M4) -> M4:
    var o = m
    o.im = -m.im
    return o


struct DensityMatrix(Copyable, Writable):
    var num_qubits: Int
    var state: StateVector

    def __init__(out self, num_qubits: Int) raises:
        check_density_size(num_qubits)
        self.num_qubits = num_qubits
        self.state = StateVector(2 * num_qubits)

    def element(self, i: Int, j: Int) -> Tuple[Float64, Float64]:
        """rho_ij as (re, im)."""
        var k = i + (j << self.num_qubits)
        return (self.state.re[k], self.state.im[k])

    def trace(self) -> Float64:
        var t = 0.0
        for i in range(1 << self.num_qubits):
            t += self.state.re[i + (i << self.num_qubits)]
        return t

    def purity(self) -> Float64:
        """Tr rho^2 = sum |rho_ij|^2: 1 for a pure state."""
        return self.state.norm_sq()

    def probabilities(self) -> List[Float64]:
        var out = List[Float64](capacity=1 << self.num_qubits)
        for i in range(1 << self.num_qubits):
            out.append(self.state.re[i + (i << self.num_qubits)])
        return out^

    def expectation(self, h: PauliSum) raises -> Float64:
        """Tr(rho H)."""
        var total = 0.0
        var dim = 1 << self.num_qubits
        for t in h.terms:
            var ny = Int(pop_count_u(t.x & t.z))
            # P = i^ny X^x Z^z on letters; Tr(rho P) = sum_j <j|rho P|j> over j
            var acc_re = 0.0
            var acc_im = 0.0
            for j in range(dim):
                var i = j ^ Int(t.x)
                var sign = -1.0 if pop_count_u(UInt64(j) & t.z) % 2 == 1 else 1.0
                var e = self.element(j, i)  # <j|rho|i>, with P|j> = phase |i>
                acc_re += sign * e[0]
                acc_im += sign * e[1]
            # i^ny
            var pr = 1.0 if ny % 4 == 0 else (-1.0 if ny % 4 == 2 else 0.0)
            var pi_ = 1.0 if ny % 4 == 1 else (-1.0 if ny % 4 == 3 else 0.0)
            total += t.coeff * (pr * acc_re - pi_ * acc_im)
        return total

    def fidelity_with(self, sv: StateVector) -> Float64:
        """<psi|rho|psi>."""
        var dim = 1 << self.num_qubits
        var total = 0.0
        for i in range(dim):
            for j in range(dim):
                var e = self.element(i, j)
                # conj(psi_i) rho_ij psi_j, real part
                var ar = sv.re[i] * sv.re[j] + sv.im[i] * sv.im[j]
                var ai = sv.re[i] * sv.im[j] - sv.im[i] * sv.re[j]
                total += ar * e[0] - ai * e[1]
        return total

    def write_to(self, mut w: Some[Writer]):
        w.write("DensityMatrix(qubits=", self.num_qubits, ", trace=", self.trace(), ", purity=", self.purity(), ")")


def pop_count_u(v: UInt64) -> Int:
    var c = 0
    var x = v
    while x != 0:
        x &= x - 1
        c += 1
    return c


def _apply_channel(mut rho: DensityMatrix, ch: KrausChannel, q0: Int, q1: Int) raises:
    var n = rho.num_qubits
    if ch.num_qubits == 1:
        # S = sum K (x) conj(K) on (q, q + n), index = row bit + 2 col bit
        var s = M4()
        for k in range(ch.count()):
            for r in range(4):
                for c in range(4):
                    var a = (r & 1) * 2 + (c & 1)
                    var b = (r >> 1) * 2 + (c >> 1)
                    var ar = ch.re[k][a]
                    var ai = ch.im[k][a]
                    var br = ch.re[k][b]
                    var bi = -ch.im[k][b]
                    s.re[4 * r + c] += ar * br - ai * bi
                    s.im[4 * r + c] += ar * bi + ai * br
        rho.state.apply_2q(s, q0, q0 + n)
        return
    # two qubits: one Kraus operator at a time (M4 on the rows, conj on the columns)
    var out_re = List[Float64](length=rho.state.size(), fill=0.0)
    var out_im = List[Float64](length=rho.state.size(), fill=0.0)
    for k in range(ch.count()):
        var m = M4()
        for r in range(4):
            for c in range(4):
                m.re[4 * r + c] = ch.re[k][4 * r + c]
                m.im[4 * r + c] = ch.im[k][4 * r + c]
        var t = rho.state.copy()
        t.apply_2q(m, q0, q1)
        t.apply_2q(_conj4(m), q0 + n, q1 + n)
        for i in range(t.size()):
            out_re[i] += t.re[i]
            out_im[i] += t.im[i]
    rho.state.re = out_re^
    rho.state.im = out_im^


def _noise_channel(c: Circuit, g: Instruction, params: List[Float64]) raises -> KrausChannel:
    var k = g.kind
    if k == Op.KRAUS1 or k == Op.KRAUS2:
        return c.kraus[Int(g.p0.value())].copy()
    var p = g.p0.resolve(params)
    if k == Op.X_ERROR:
        return _pauli_channel_1q(p, 0.0, 0.0)
    if k == Op.Y_ERROR:
        return _pauli_channel_1q(0.0, p, 0.0)
    if k == Op.Z_ERROR:
        return _pauli_channel_1q(0.0, 0.0, p)
    if k == Op.DEPOLARIZE1:
        return _pauli_channel_1q(p / 3, p / 3, p / 3)
    if k == Op.PAULI_CHANNEL_1:
        return _pauli_channel_1q(p, g.p1.resolve(params), g.p2.resolve(params))
    if k == Op.DEPOLARIZE2:
        return _depolarize_2q(p)
    raise Error("QM-E414: the density-matrix simulator does not take this noise instruction. Hint: use Kraus channels or Pauli noise.")


def density_evolve(c: Circuit, params: List[Float64] = List[Float64]()) raises -> DensityMatrix:
    """Final density matrix of `c` from |0...0>."""
    c.validate()
    check_density_size(c.num_qubits)
    if len(params) != c.num_params():
        raise Error("QM-E101: circuit has " + String(c.num_params()) + " parameters " + c.param_names_str() + " but " + String(len(params)) + " values were given.")
    var n = c.num_qubits
    var rho = DensityMatrix(n)
    var measure = _measure_channel()
    var reset = _reset_channel()
    for g in c.ops:
        var k = g.kind
        if op_is_annotation(k):
            continue
        if g.cond >= 0:
            raise Error("QM-E411: instruction '" + String(g) + "' depends on a classical bit, which a density matrix does not track. Hint: use sample(), which runs per-shot trajectories.")
        if op_is_unitary(k):
            var a = op_arity(k)
            apply_instruction(rho.state, g, params)
            if a == 1:
                if k != Op.I:
                    rho.state.apply_1q(_conj(gate_matrix_1q(g, params)), g.q0 + n)
            elif a == 2:
                rho.state.apply_2q(_conj4(matrix_2q(k, g.p0.resolve(params))), g.q0 + n, g.q1 + n)
            else:
                var h = g.copy()  # 3q gates have real matrices
                h.q0 += n
                h.q1 += n
                h.q2 += n
                apply_instruction(rho.state, h, params)
        elif k == Op.MEASURE:
            _apply_channel(rho, measure, g.q0, -1)
        elif k == Op.RESET:
            _apply_channel(rho, reset, g.q0, -1)
        elif op_is_noise(k):
            _apply_channel(rho, _noise_channel(c, g, params), g.q0, g.q1)
    return rho^


def sample_density(c: Circuit, shots: Int, seed: UInt64 = 1234,
                   params: List[Float64] = List[Float64]()) raises -> SampleResult:
    """Shots from the exact outcome distribution of a circuit with terminal measurements."""
    if c.num_measurements == 0:
        raise Error("QM-E402: circuit has no measurements. Hint: call measure_all() or use density_evolve().")
    if c.num_clbits > 62:
        raise Error("QM-E406: the sampler packs classical registers into 62 bits; circuit has " + String(c.num_clbits) + " clbits. Hint: measure fewer bits or split the register.")
    # is_dynamic also flags noise right after a measurement, which is fine for rho,
    # but then the record would differ from the measured bit
    if c.is_dynamic():
        raise Error("QM-E411: sampling needs terminal measurements; a density matrix does not keep the bit of a mid-circuit measurement. Hint: use sample(), which runs per-shot trajectories.")
    var rho = density_evolve(c, params)
    var probs = rho.probabilities()
    var mm = terminal_measure_map(c)
    var res = SampleResult(c.num_clbits, shots, "density_matrix")
    var rng = Rng(seed)
    # sorted uniforms over the CDF
    var us = List[Float64](capacity=shots)
    for _ in range(shots):
        us.append(rng.uniform())
    sort(us)
    var i = 0
    var cum = probs[0]
    var tr = rho.trace()
    for u in us:
        while u * tr >= cum and i + 1 < len(probs):
            i += 1
            cum += probs[i]
        res.record(mm.pack(i))
    return res^
