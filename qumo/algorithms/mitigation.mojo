"""Zero-noise extrapolation (ZNE): unitary gate folding plus extrapolation.

The idea (Temme, Bravyi & Gambetta 2017; Giurgica-Tiron et al. 2020; Mitiq):
run the same logical circuit at several *amplified* noise levels, then
extrapolate the measured expectation value back to zero noise.

Noise is amplified by **unitary folding**, which never changes the ideal
result:

  * `fold_gates_global(c, scale)` - Mitiq `fold_global`:
        U (U^dag U)^k  [then L^dag L for the last few gates L of U]
    k = floor((scale - 1) / 2) full folds; the fractional part folds the last
    round(frac * d / 2) gates (d = number of unitary gates), so the folded
    circuit has scale * d gates (+-1).
  * `fold_gates_from_left(c, scale)` - local folding G -> G (G^dag G)^m, with
    F = round(d (scale - 1) / 2) single folds spread over the gates (every
    gate gets floor(F / d), the first F mod d gates one more).
  * `fold_gates_at_random(c, scale, seed)` - as `from_left`, but the extra
    folds go to F mod d distinct gates picked by a seeded shuffle.

Terminal measurements are removed before folding and re-appended after it.
Circuits with noise channels, resets, classical control or mid-circuit
measurements are rejected (QM-E801): fold first, then add noise
(`qumo.qec.characterization.apply_noise_model`).

Extrapolation to scale 0 (Mitiq factories):

  * `richardson_extrapolate(scales, values)` - Lagrange interpolation through
    every point (exact for polynomials of degree len-1).
  * `linear_extrapolate(scales, values)`, `poly_extrapolate(scales, values,
    order)` - least squares (QR, stable for small orders).
  * `exp_extrapolate(scales, values, asymptote)` - y = a + b exp(-c s) with a
    known asymptote a (log-linear least squares).

`zne_expectation(circuit, h, noise, scales, shots, seed, method)` runs the
whole pipeline: fold -> append measurement-basis rotations -> apply the
`NoiseModel` (so the basis changes and readout are noisy too, as on hardware)
-> trajectory sampling, one measurement setting per qubit-wise commuting
group of `h` -> extrapolation. The exact Estimator primitive
(`qumo.exec.primitives.run_estimator`) evolves only the unitary part of a
circuit and therefore cannot see noise channels; this module samples instead.

Error codes: QM-E801 (invalid option: scale < 1, unfoldable circuit, unknown
method, data that cannot be fitted), QM-E802 (invalid size: mismatched or too
few points, duplicate scales, shots < 1).
"""

from std.math import exp, log, sqrt, floor
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_is_unitary, op_num_params, op_inverse_kind
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import sample
from qumo.sim.rng import Rng
from qumo.qec.characterization import NoiseModel, apply_noise_model


# ---------------------------------------------------------------------------
# Folding helpers
# ---------------------------------------------------------------------------
def _round_half_up(x: Float64) -> Int:
    return Int(floor(x + 0.5))


def _check_scale(scale: Float64, what: String) raises:
    if not (scale >= 1.0):  # also rejects NaN
        raise Error("QM-E801: " + what + " needs scale >= 1, got " + String(scale) + ". Hint: scale 1 is the unfolded circuit; odd integers (1, 3, 5) give exact global folds.")


def _split_for_folding(circuit: Circuit, what: String, mut gates: List[Instruction], mut meas: List[Instruction]) raises:
    """Unitary gates and terminal measurements of a foldable circuit."""
    if circuit.has_noise():
        raise Error("QM-E801: " + what + " got a circuit with noise channels. Hint: fold the ideal circuit first, then add noise with apply_noise_model().")
    if circuit.is_dynamic():
        raise Error("QM-E801: " + what + " needs a circuit whose measurements are all terminal (no mid-circuit measurement, reset or classical control). Hint: fold only the unitary part.")
    for g in circuit.ops:
        if g.kind == Op.MEASURE:
            meas.append(g)
        elif op_is_unitary(g.kind):
            gates.append(g)
        # barriers / ticks / QEC annotations carry no quantum effect: dropped


def _inverse_inst(g: Instruction) -> Instruction:
    var n = g.copy()
    n.kind = op_inverse_kind(g.kind)
    if g.kind == Op.U:
        n.p0 = -g.p0
        n.p1 = -g.p2
        n.p2 = -g.p1
    elif op_num_params(g.kind) == 1:
        n.p0 = -g.p0
    return n


def _finish_fold(circuit: Circuit, mut out: Circuit, meas: List[Instruction]):
    for g in meas:
        out.measure(g.q0, g.clbit)
    out.num_clbits = max(out.num_clbits, circuit.num_clbits)
    out.name = circuit.name + "_folded"


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------
def fold_gates_global(circuit: Circuit, scale: Float64) raises -> Circuit:
    """Global folding U (U^dag U)^k plus partial folding of the final gates.

    Gate count: d (1 + 2k) + 2 round(frac d / 2) = scale * d (+-1) where
    k = floor((scale - 1) / 2), frac = scale - 1 - 2k. Exactly unitary-
    equivalent to `circuit` (same global phase). Params are preserved.
    """
    _check_scale(scale, "fold_gates_global()")
    var gates = List[Instruction]()
    var meas = List[Instruction]()
    _split_for_folding(circuit, "fold_gates_global()", gates, meas)
    var out = circuit.empty_like()
    out.num_clbits = 0
    var d = len(gates)
    for g in gates:
        out.append(g)
    var k = Int(floor((scale - 1.0) / 2.0))
    var frac = (scale - 1.0) - 2.0 * Float64(k)
    for _ in range(k):
        var i = d - 1
        while i >= 0:
            out.append(_inverse_inst(gates[i]))
            i -= 1
        for g in gates:
            out.append(g)
    var num_to_fold = min(_round_half_up(frac * Float64(d) / 2.0), d)
    if num_to_fold > 0:
        var i = d - 1
        while i >= d - num_to_fold:
            out.append(_inverse_inst(gates[i]))
            i -= 1
        for j in range(d - num_to_fold, d):
            out.append(gates[j])
    _finish_fold(circuit, out, meas)
    return out^


def _fold_local(circuit: Circuit, scale: Float64, what: String, randomize: Bool, seed: UInt64) raises -> Circuit:
    _check_scale(scale, what)
    var gates = List[Instruction]()
    var meas = List[Instruction]()
    _split_for_folding(circuit, what, gates, meas)
    var out = circuit.empty_like()
    out.num_clbits = 0
    var d = len(gates)
    var folds = List[Int](length=d, fill=0)
    if d > 0:
        var total = _round_half_up(Float64(d) * (scale - 1.0) / 2.0)
        var base = total // d
        var extra = total % d
        for i in range(d):
            folds[i] = base
        if randomize:
            # seeded partial Fisher-Yates: the first `extra` entries are the chosen gates
            var order = List[Int](length=d, fill=0)
            for i in range(d):
                order[i] = i
            var rng = Rng(seed)
            for i in range(extra):
                var j = i + rng.below(d - i)
                var tmp = order[i]
                order[i] = order[j]
                order[j] = tmp
                folds[order[i]] += 1
        else:
            for i in range(extra):
                folds[i] += 1
    for i in range(d):
        out.append(gates[i])
        var inv = _inverse_inst(gates[i])
        for _ in range(folds[i]):
            out.append(inv)
            out.append(gates[i])
    _finish_fold(circuit, out, meas)
    return out^


def fold_gates_from_left(circuit: Circuit, scale: Float64) raises -> Circuit:
    """Local folding G -> G (G^dag G)^m, extra folds on the leftmost gates.

    round(d (scale - 1) / 2) single folds in total (scale * d gates, +-1).
    """
    return _fold_local(circuit, scale, "fold_gates_from_left()", False, 0)


def fold_gates_at_random(circuit: Circuit, scale: Float64, seed: UInt64 = 1234) raises -> Circuit:
    """Local folding G -> G (G^dag G)^m, extra folds on seeded random distinct gates."""
    return _fold_local(circuit, scale, "fold_gates_at_random()", True, seed)


def fold_gates(circuit: Circuit, scale: Float64, folding: String = "global", seed: UInt64 = 1234) raises -> Circuit:
    """Dispatch by name: 'global', 'left' (from_left) or 'random' (at_random)."""
    if folding == "global":
        return fold_gates_global(circuit, scale)
    if folding == "left" or folding == "from_left":
        return fold_gates_from_left(circuit, scale)
    if folding == "random" or folding == "at_random":
        return fold_gates_at_random(circuit, scale, seed)
    raise Error("QM-E801: unknown folding '" + folding + "'. Hint: use 'global', 'left' or 'random'.")


def folded_scale(original: Circuit, folded: Circuit) -> Float64:
    """Achieved noise scale = unitary gate count ratio (I gates excluded)."""
    var d = original.num_unitary_gates()
    if d == 0:
        return 1.0
    return Float64(folded.num_unitary_gates()) / Float64(d)


# ---------------------------------------------------------------------------
# Extrapolation to zero noise
# ---------------------------------------------------------------------------
def _check_points(scales: List[Float64], values: List[Float64], need: Int, what: String) raises:
    if len(scales) != len(values):
        raise Error("QM-E802: " + what + " got " + String(len(scales)) + " scales but " + String(len(values)) + " values.")
    if len(scales) < need:
        raise Error("QM-E802: " + what + " needs at least " + String(need) + " points, got " + String(len(scales)) + ". Hint: run more noise scales, e.g. [1, 3, 5].")
    for i in range(len(scales)):
        for j in range(i):
            if abs(scales[i] - scales[j]) < 1e-12:
                raise Error("QM-E802: " + what + " got the noise scale " + String(scales[i]) + " twice.")


def richardson_extrapolate(scales: List[Float64], values: List[Float64]) raises -> Float64:
    """Value at scale 0 of the interpolating polynomial through all points.

    = sum_i v_i prod_{j != i} s_j / (s_j - s_i). Exact for polynomial data of
    degree len-1 (amplifies shot noise: keep 2-4 points).
    """
    _check_points(scales, values, 2, "richardson_extrapolate()")
    var m = len(scales)
    var est = 0.0
    for i in range(m):
        var w = 1.0
        for j in range(m):
            if j != i:
                w *= scales[j] / (scales[j] - scales[i])
        est += w * values[i]
    return est


def poly_extrapolate(scales: List[Float64], values: List[Float64], order: Int) raises -> Float64:
    """Least-squares polynomial of degree `order`, evaluated at scale 0.

    Solved by modified Gram-Schmidt QR of the Vandermonde matrix (stable for
    the small orders used in ZNE). Needs len(scales) >= order + 1.
    """
    if order < 0:
        raise Error("QM-E801: poly_extrapolate() needs order >= 0, got " + String(order) + ".")
    _check_points(scales, values, order + 1, "poly_extrapolate()")
    var m = len(scales)
    var p = order + 1
    # columns of A (Vandermonde), column-major: a[k * m + i] = s_i^k
    var a = List[Float64](length=p * m, fill=0.0)
    for i in range(m):
        var v = 1.0
        for k in range(p):
            a[k * m + i] = v
            v *= scales[i]
    var r = List[Float64](length=p * p, fill=0.0)
    for k in range(p):
        for j in range(k):
            var dot = 0.0
            for i in range(m):
                dot += a[j * m + i] * a[k * m + i]
            r[j * p + k] = dot
            for i in range(m):
                a[k * m + i] -= dot * a[j * m + i]
        var nrm = 0.0
        for i in range(m):
            nrm += a[k * m + i] * a[k * m + i]
        nrm = sqrt(nrm)
        if nrm < 1e-14:
            raise Error("QM-E801: poly_extrapolate() cannot fit order " + String(order) + " (degenerate scales). Hint: use distinct scales or a lower order.")
        r[k * p + k] = nrm
        for i in range(m):
            a[k * m + i] /= nrm
    # c = R^-1 Q^T y
    var qty = List[Float64](length=p, fill=0.0)
    for k in range(p):
        var dot = 0.0
        for i in range(m):
            dot += a[k * m + i] * values[i]
        qty[k] = dot
    var c = List[Float64](length=p, fill=0.0)
    var k = p - 1
    while k >= 0:
        var s = qty[k]
        for j in range(k + 1, p):
            s -= r[k * p + j] * c[j]
        c[k] = s / r[k * p + k]
        k -= 1
    return c[0]


def linear_extrapolate(scales: List[Float64], values: List[Float64]) raises -> Float64:
    """Least-squares straight line, evaluated at scale 0."""
    return poly_extrapolate(scales, values, 1)


def exp_extrapolate(scales: List[Float64], values: List[Float64], asymptote: Float64 = 0.0) raises -> Float64:
    """Fit y = a + b exp(-c s) with known asymptote a; return a + b (scale 0).

    Log-linear least squares on ln|y - a|; every value must lie strictly on
    the same side of the asymptote (QM-E801 otherwise).
    """
    _check_points(scales, values, 2, "exp_extrapolate()")
    var sign = 1.0 if values[0] - asymptote > 0.0 else -1.0
    var logs = List[Float64]()
    for v in values:
        var d = (v - asymptote) * sign
        if not (d > 0.0):
            raise Error("QM-E801: exp_extrapolate() needs every value on the same side of the asymptote " + String(asymptote) + ". Hint: the data is too noisy for an exponential fit; use richardson or linear.")
        logs.append(log(d))
    var lb = poly_extrapolate(scales, logs, 1)
    return asymptote + sign * exp(lb)


def extrapolate(scales: List[Float64], values: List[Float64], method: String = "richardson", asymptote: Float64 = 0.0) raises -> Float64:
    """Dispatch by name: 'richardson', 'linear', 'poly2', 'poly3', 'exp'."""
    if method == "richardson":
        return richardson_extrapolate(scales, values)
    if method == "linear":
        return linear_extrapolate(scales, values)
    if method == "poly2":
        return poly_extrapolate(scales, values, 2)
    if method == "poly3":
        return poly_extrapolate(scales, values, 3)
    if method == "exp":
        return exp_extrapolate(scales, values, asymptote)
    raise Error("QM-E801: unknown extrapolation method '" + method + "'. Hint: use 'richardson', 'linear', 'poly2', 'poly3' or 'exp'.")


# ---------------------------------------------------------------------------
# Shot-based expectation values (noise-aware)
# ---------------------------------------------------------------------------
def _parity(x: Int) -> Int:
    var v = x
    var p = 0
    while v != 0:
        v &= v - 1
        p ^= 1
    return p


def _estimate(circuit: Circuit, h: PauliSum, shots: Int, seed: UInt64, use_model: Bool, noise: NoiseModel) raises -> Tuple[Float64, Float64]:
    if shots < 1:
        raise Error("QM-E802: shot-based estimation needs shots >= 1, got " + String(shots) + ".")
    if circuit.is_dynamic():
        raise Error("QM-E801: shot-based estimation needs measurements to be terminal. Hint: remove mid-circuit measurements / resets / classical control.")
    var n = circuit.num_qubits
    if h.num_qubits() > n:
        raise Error("QM-E802: the observable acts on " + String(h.num_qubits()) + " qubits but the circuit has " + String(n) + ".")
    # unitary + noise part (terminal measurements are replaced by our own)
    var body = circuit.empty_like()
    for g in circuit.ops:
        if g.kind != Op.MEASURE:
            body.append(g)
    var groups = h.group_qwc()
    var ev = 0.0
    var var_total = 0.0
    for gi in range(len(groups)):
        ref grp = groups[gi]
        var gx: UInt64 = 0
        var gy: UInt64 = 0
        var has_support = False
        for ti in grp:
            var t = h.terms[ti]
            if (t.x | t.z) == 0:
                ev += t.coeff
            else:
                has_support = True
            gx |= t.x
            gy |= t.x & t.z
        if not has_support:
            continue
        var m = body.copy()
        m.num_clbits = 0
        for q in range(n):
            var b = UInt64(1) << UInt64(q)
            if (gx & b) != 0:
                if (gy & b) != 0:
                    m.sdg(q)
                m.h(q)
        for q in range(n):
            m.measure(q, q)
        if use_model:
            m = apply_noise_model(m, noise)
        var res = sample(m, shots, seed + UInt64(gi) * 1000003)
        var s1 = 0.0
        var s2 = 0.0
        for o in res.memory:
            var v = 0.0
            for ti in grp:
                var t = h.terms[ti]
                var support = Int(t.x | t.z)
                if support == 0:
                    continue
                v += -t.coeff if _parity(o & support) == 1 else t.coeff
            s1 += v
            s2 += v * v
        var mean = s1 / Float64(shots)
        ev += mean
        var var1 = max(0.0, s2 / Float64(shots) - mean * mean)
        var_total += var1 / Float64(shots)
    return (ev, sqrt(var_total))


def sampled_expectation(circuit: Circuit, h: PauliSum, shots: Int, seed: UInt64 = 1234) raises -> Tuple[Float64, Float64]:
    """(<h>, standard error) from `shots` samples per qubit-wise commuting group.

    Noise channels already in `circuit` are honoured (trajectory sampling);
    terminal measurements are replaced by basis rotations + measure-all.
    """
    return _estimate(circuit, h, shots, seed, False, NoiseModel(circuit.num_qubits))


def noisy_expectation(circuit: Circuit, h: PauliSum, noise: NoiseModel, shots: Int, seed: UInt64 = 1234) raises -> Tuple[Float64, Float64]:
    """As `sampled_expectation`, with `noise` applied to the circuit *including*
    the measurement-basis rotations and readout (hardware-faithful)."""
    return _estimate(circuit, h, shots, seed, True, noise)


# ---------------------------------------------------------------------------
# ZNE driver
# ---------------------------------------------------------------------------
struct ZNEResult(Copyable, Writable):
    var scales: List[Float64]  # requested noise scales
    var achieved_scales: List[Float64]  # gate-count ratios actually used for the fit
    var values: List[Float64]  # noisy expectation value per scale
    var stds: List[Float64]  # standard error per scale
    var estimate: Float64  # extrapolated zero-noise value
    var method: String
    var folding: String
    var shots: Int

    def __init__(out self, method: String, folding: String, shots: Int):
        self.scales = List[Float64]()
        self.achieved_scales = List[Float64]()
        self.values = List[Float64]()
        self.stds = List[Float64]()
        self.estimate = 0.0
        self.method = method
        self.folding = folding
        self.shots = shots

    def raw(self) -> Float64:
        """Unmitigated value (the first scale, normally 1)."""
        return self.values[0] if len(self.values) > 0 else 0.0

    def write_to(self, mut w: Some[Writer]):
        w.write("ZNEResult(method=", self.method, ", folding=", self.folding, ", shots=", self.shots, ", points=[")
        for i in range(len(self.values)):
            if i > 0:
                w.write(", ")
            w.write("(", self.achieved_scales[i], ": ", self.values[i], " +- ", self.stds[i], ")")
        w.write("], estimate=", self.estimate, ")")


def zne_expectation(circuit: Circuit, h: PauliSum, noise: NoiseModel, scales: List[Float64],
                    shots: Int, seed: UInt64, method: String = "richardson",
                    folding: String = "global", asymptote: Float64 = 0.0) raises -> ZNEResult:
    """Zero-noise-extrapolated <h> of an ideal `circuit` under `noise`.

    For each scale: fold (`folding` = global | left | random), then estimate
    <h> with `noisy_expectation` (`shots` per QWC group, seeds derived from
    `seed`), then extrapolate with `method` (richardson | linear | poly2 |
    poly3 | exp) over the achieved scales.
    """
    _check_points(scales, scales, 1, "zne_expectation()")
    var res = ZNEResult(method, folding, shots)
    for si in range(len(scales)):
        var folded = fold_gates(circuit, scales[si], folding, seed + UInt64(si) * 7919)
        var est = noisy_expectation(folded, h, noise, shots, seed + UInt64(si) * 104729)
        res.scales.append(scales[si])
        res.achieved_scales.append(folded_scale(circuit, folded))
        res.values.append(est[0])
        res.stds.append(est[1])
    if len(scales) == 1:
        res.estimate = res.values[0]
    else:
        res.estimate = extrapolate(res.achieved_scales, res.values, method, asymptote)
    return res^
