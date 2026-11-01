"""IQP circuits and their classical expectation values (PennyLane `IQP`, `iqp_expval`;
Recio-Armengol et al., arXiv:2503.02934).

An IQP circuit is prod_g exp(i theta_g X_g), each generator g a product of X on
a set of qubits. Sampling it is believed hard, but <Z_a> is not:

    <Z_a> = E_z cos( sum_g 2 theta_g (a.g mod 2) (-1)^(z.g) ),   z uniform on {0,1}^n

so a Monte Carlo average needs no state vector and thousands of qubits are within
reach, which is the point of the method: train an IQP model classically, then
sample it on a device. The gradient comes from the same samples.
"""

from std.bit import pop_count
from std.math import cos, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.param import Param
from qumo.sim.rng import Rng
from max.algorithm import parallelize


struct IqpModel(Copyable, Writable):
    """Bit-packed generators; generators in one group share a weight."""

    var num_qubits: Int
    var words: Int
    var generators: List[UInt64]  # generator g at [g * words ...]
    var group_of: List[Int]
    var num_groups: Int

    def __init__(out self, num_qubits: Int) raises:
        if num_qubits < 1:
            raise Error("QM-E818: an IQP model needs at least one qubit. Hint: IqpModel(n).")
        self.num_qubits = num_qubits
        self.words = (num_qubits + 63) // 64
        self.generators = List[UInt64]()
        self.group_of = List[Int]()
        self.num_groups = 0

    def add_generator(mut self, qubits: List[Int], group: Int = -1) raises:
        """exp(i theta X_qubits); `group` shares a weight with earlier generators."""
        if len(qubits) == 0:
            raise Error("QM-E818: an IQP generator needs at least one qubit. Hint: e.g. [0, 2].")
        var row = List[UInt64](length=self.words, fill=0)
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                raise Error("QM-E818: generator qubit " + String(q) + " is outside 0.." + String(self.num_qubits - 1) + ". Hint: enlarge the model.")
            row[q >> 6] ^= UInt64(1) << UInt64(q & 63)
        for w in row:
            self.generators.append(w)
        var g = group if group >= 0 else self.num_groups
        self.group_of.append(g)
        self.num_groups = max(self.num_groups, g + 1)

    def num_generators(self) -> Int:
        return len(self.group_of)

    def write_to(self, mut w: Some[Writer]):
        w.write("IqpModel(qubits=", self.num_qubits, ", generators=", self.num_generators(), ", groups=", self.num_groups, ")")


def _check_weights(model: IqpModel, weights: List[Float64]) raises:
    if len(weights) != model.num_groups:
        raise Error("QM-E818: " + String(len(weights)) + " weights for " + String(model.num_groups) + " parameter groups. Hint: one weight per group.")


def append_iqp(mut c: Circuit, model: IqpModel, weights: List[Float64]) raises:
    """The gates: H on all, exp(i theta Z_g) as a CX ladder, H on all."""
    _check_weights(model, weights)
    var n = model.num_qubits
    for q in range(n):
        c.h(q)
    for g in range(model.num_generators()):
        var support = List[Int]()
        for q in range(n):
            if ((model.generators[g * model.words + (q >> 6)] >> UInt64(q & 63)) & 1) != 0:
                support.append(q)
        var last = support[len(support) - 1]
        for i in range(len(support) - 1):
            c.cx(support[i], last)
        c.rz(-2.0 * weights[model.group_of[g]], last)
        var i = len(support) - 2
        while i >= 0:
            c.cx(support[i], last)
            i -= 1
    for q in range(n):
        c.h(q)


@fieldwise_init
struct IqpExpval(Copyable, Writable):
    var means: List[Float64]
    var stds: List[Float64]  # standard errors of the means
    var gradients: List[List[Float64]]  # [op][group], empty unless requested

    def write_to(self, mut w: Some[Writer]):
        w.write("IqpExpval(")
        for i in range(len(self.means)):
            w.write(" ", self.means[i], "+-", self.stds[i])
        w.write(" )")


def _parity_word(a: UInt64, b: UInt64) -> Int:
    return Int(pop_count(a & b)) & 1


def iqp_expval(model: IqpModel, weights: List[Float64], ops: List[List[Int]], samples: Int, seed: UInt64 = 1,
               gradient: Bool = False) raises -> IqpExpval:
    """<Z_a> for each qubit list in `ops`, from `samples` Monte Carlo draws of z.

    Gates-free: O(samples x generators x words) per op, parallel over samples, any
    width. Deterministic for a seed. Standard errors are the sample error of the
    cosine; `gradient` adds d<Z_a>/d theta from the same draws."""
    _check_weights(model, weights)
    if samples < 1:
        raise Error("QM-E818: iqp_expval needs samples >= 1. Hint: e.g. 10000.")
    var words = model.words
    var ng = model.num_generators()
    # for each op: generators that anticommute with Z_a, each with its angle and group
    var out_means = List[Float64]()
    var out_stds = List[Float64]()
    var grads = List[List[Float64]]()
    var base = Rng(seed)
    var zs = List[UInt64](length=samples * words, fill=0)
    for s in range(samples):
        var rng = base.split(s)
        for w in range(words):
            zs[s * words + w] = rng.next_u64()
    for op in ops:
        var a = List[UInt64](length=words, fill=0)
        for q in op:
            if q < 0 or q >= model.num_qubits:
                raise Error("QM-E818: operator qubit " + String(q) + " is outside the model. Hint: qubits run from 0 to " + String(model.num_qubits - 1) + ".")
            a[q >> 6] ^= UInt64(1) << UInt64(q & 63)
        var active = List[Int]()
        var angles = List[Float64]()
        for g in range(ng):
            var odd = 0
            for w in range(words):
                odd ^= _parity_word(a[w], model.generators[g * words + w])
            if odd == 1:
                active.append(g)
                angles.append(2.0 * weights[model.group_of[g]])
        var nchunks = 64
        var sums = List[Float64](length=nchunks, fill=0.0)
        var squares = List[Float64](length=nchunks, fill=0.0)
        var ngroups = model.num_groups
        var gsum = List[Float64](length=nchunks * ngroups, fill=0.0)
        var sp = sums.unsafe_ptr().as_unsafe_any_origin()
        var qp = squares.unsafe_ptr().as_unsafe_any_origin()
        var gp = gsum.unsafe_ptr().as_unsafe_any_origin()
        var zp = zs.unsafe_ptr().as_unsafe_any_origin()
        var mp = model.generators.unsafe_ptr().as_unsafe_any_origin()
        var ap = active.unsafe_ptr().as_unsafe_any_origin()
        var tp = angles.unsafe_ptr().as_unsafe_any_origin()
        var cp = model.group_of.unsafe_ptr().as_unsafe_any_origin()
        var na = len(active)

        def body(slot: Int) {imm sp, imm qp, imm gp, imm zp, imm mp, imm ap, imm tp, imm cp, imm na, imm words, imm gradient, imm ngroups, imm nchunks, imm samples}:
            var lo = slot * samples // nchunks
            var hi = (slot + 1) * samples // nchunks
            var signs = List[Float64](length=na, fill=0.0)
            for s in range(lo, hi):
                var phase = 0.0
                for i in range(na):
                    var g = ap[unsafe_offset=i]
                    var odd = 0
                    for w in range(words):
                        odd ^= _parity_word(zp[unsafe_offset=s * words + w], mp[unsafe_offset=g * words + w])
                    var sg = -1.0 if odd == 1 else 1.0
                    signs[i] = sg
                    phase += sg * tp[unsafe_offset=i]
                var c = cos(phase)
                sp[unsafe_offset=slot] += c
                qp[unsafe_offset=slot] += c * c
                if gradient:
                    var d = -sin(phase)
                    for i in range(na):
                        gp[unsafe_offset=slot * ngroups + cp[unsafe_offset=ap[unsafe_offset=i]]] += d * 2.0 * signs[i]

        parallelize(body, nchunks)
        _ = len(active)
        _ = len(angles)
        _ = len(zs)
        _ = len(model.generators)
        var sum = 0.0
        var sq = 0.0
        var g_total = List[Float64](length=ngroups, fill=0.0)
        for c in range(nchunks):
            sum += sums[c]
            sq += squares[c]
            for gi in range(ngroups):
                g_total[gi] += gsum[c * ngroups + gi]
        var mean = sum / Float64(samples)
        out_means.append(mean)
        var variance = max(0.0, sq / Float64(samples) - mean * mean)
        out_stds.append(sqrt(variance / Float64(samples)))
        if gradient:
            for gi in range(ngroups):
                g_total[gi] /= Float64(samples)
            grads.append(g_total^)
    return IqpExpval(out_means^, out_stds^, grads^)
