"""Shot-based expectation values of a PauliSum: the ONE implementation.

<H> from shots = for each qubit-wise-commuting (QWC) group of H (`PauliSum.group_qwc`):
rotate every qubit into the group's basis (X -> H, Y -> Sdg H), measure all qubits,
and read every term of the group from the SAME shots. The value of a group is the
mean of the per-shot weighted sum of its terms and its variance is the variance of
that sum, so the covariance between terms is counted (Z0 - Z1 on a Bell pair is 0
on every shot: standard error 0, not sqrt(2 / shots)).

Layers, lowest first:
  * `qwc_setting`        the measured circuit of one group (optional X twirl mask);
  * `qwc_moments`        sum and sum of squares of a per-shot weighted parity sum;
  * `qwc_group_stats`    (mean, variance of the mean) of one group;
  * `qwc_estimate`       all groups through a `run(circuit, shots, seed)` closure;
  * `shots_for_precision` shots per setting for a target standard error.

Callers: `qumo.exec.primitives.run_estimator` (precision > 0),
`qumo.algorithms.mitigation` (`sampled_expectation`, `noisy_expectation`, so ZNE, and
`trex_expectation`). The module imports only `qumo.ir` and `SampleResult`: the
sampler is the caller's closure, so noise models, GPUs and devices plug in above.
"""

from std.math import sqrt, ceil

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import SampleResult

comptime MAX_ESTIMATOR_SHOTS = 10_000_000
comptime QWC_GROUP_SEED_STRIDE = 1000003  # group gi is sampled with seed + gi * stride


@always_inline
def pop_parity(x: Int) -> Bool:
    var v = x
    var p = 0
    while v != 0:
        v &= v - 1
        p ^= 1
    return p == 1


def qwc_setting(circuit: Circuit, h: PauliSum, grp: List[Int], mut base: Int, flip: UInt64 = 0) -> Circuit:
    """The circuit that measures one QWC group of `h`: the basis rotation, then every
    qubit q into clbit `base + q`. A circuit whose measurements are all terminal
    loses them (base = 0). A dynamic one (reset, mid-circuit measurement, classical
    control) keeps every operation and gets the new bits after its own.
    `flip` puts an X on its qubits between the rotation and the measurement (the
    readout twirl of T-REX); read the shots back with the same `flip`."""
    var n = circuit.num_qubits
    var gx: UInt64 = 0
    var gy: UInt64 = 0
    for ti in grp:
        gx |= h.terms[ti].x
        gy |= h.terms[ti].x & h.terms[ti].z  # Y marker
    var m: Circuit
    if circuit.is_dynamic():
        m = circuit.copy()
        base = circuit.num_clbits
    else:
        m = circuit.empty_like()
        for g in circuit.ops:
            if g.kind != Op.MEASURE:
                m.append(g)
        m.num_clbits = 0
        base = 0
    for q in range(n):
        var b = UInt64(1) << UInt64(q)
        if (gx & b) != 0:
            if (gy & b) != 0:
                m.sdg(q)
            m.h(q)
    if flip != 0:
        for q in range(n):
            if (flip & (UInt64(1) << UInt64(q))) != 0:
                m.x(q)
    for q in range(n):
        m.measure(q, base + q)
    return m^


def qwc_moments(res: SampleResult, supports: List[Int], weights: List[Float64], base: Int, flip: Int,
                mut s1: Float64, mut s2: Float64):
    """Add to (s1, s2) the sum and the sum of squares, over the shots of `res`, of
    v = sum_k weights[k] * (-1)^parity(bits & supports[k]), where bits are the
    setting's clbits (from `base`), flipped back through `flip`."""
    for o in res.memory:
        var bits = (o >> base) ^ flip
        var v = 0.0
        for k in range(len(supports)):
            v += -weights[k] if pop_parity(bits & supports[k]) else weights[k]
        s1 += v
        s2 += v * v


def moments_variance_of_mean(s1: Float64, s2: Float64, shots: Int) -> Float64:
    """Variance of the mean of `shots` values with sum s1 and sum of squares s2."""
    var mean = s1 / Float64(shots)
    var var1 = max(0.0, s2 / Float64(shots) - mean * mean)
    return var1 / Float64(shots)


def qwc_group_stats(res: SampleResult, h: PauliSum, grp: List[Int], base: Int) -> Tuple[Float64, Float64]:
    """(mean, variance of the mean) of sum_t coeff_t <P_t> over the group's non-identity
    terms. All terms are read from the SAME shots, so the variance is that of the
    per-shot weighted sum, not the sum of the per-term variances: they are correlated
    (Z0 - Z1 on a Bell pair is 0 on every shot)."""
    var shots = len(res.memory)
    var supports = List[Int]()
    var weights = List[Float64]()
    for ti in grp:
        var t = h.terms[ti]
        var support = Int(t.x | t.z)
        if support == 0:
            continue
        supports.append(support)
        weights.append(t.coeff)
    var s1 = 0.0
    var s2 = 0.0
    qwc_moments(res, supports, weights, base, 0, s1, s2)
    return (s1 / Float64(shots), moments_variance_of_mean(s1, s2, shots))


def qwc_estimate[S: def(Circuit, Int, UInt64) raises -> SampleResult](
    run: S, circuit: Circuit, h: PauliSum, shots: Int, seed: UInt64
) raises -> Tuple[Float64, Float64]:
    """(<h>, standard error): one measurement setting per QWC group of `h`, each
    sampled by `run(setting, shots, seed + gi * QWC_GROUP_SEED_STRIDE)`. `run` is the
    real sampler of the caller (parameters bound, noise model applied, a device...),
    so reset, noise and mid-circuit measurement count. Identity terms add their
    coefficient and cost no shots."""
    var groups = h.group_qwc()
    var ev = 0.0
    var var_total = 0.0
    for gi in range(len(groups)):
        ref g = groups[gi]
        var has_support = False
        for ti in g:
            var t = h.terms[ti]
            if (t.x | t.z) == 0:
                ev += t.coeff
            else:
                has_support = True
        if not has_support:
            continue
        var base = 0
        var m = qwc_setting(circuit, h, g, base)
        var res = run(m, shots, seed + UInt64(gi) * QWC_GROUP_SEED_STRIDE)
        var st = qwc_group_stats(res, h, g, base)
        ev += st[0]
        var_total += st[1]
    return (ev, sqrt(var_total))


def shots_for_precision(observables: List[PauliSum], precision: Float64) raises -> Int:
    """Shots per measurement setting so that every observable's standard error is at
    most `precision`. A group's per-shot value is bounded by L = sum |coeff| of its
    non-identity terms, so its variance is at most L^2 / shots, and the observable's
    standard error is at most sqrt(sum_g L_g^2 / shots). Hence
    shots = ceil(max over observables of sum_g L_g^2 / precision^2) (at least 1).
    It is a bound, the reported std is usually smaller. QM-E913 past
    MAX_ESTIMATOR_SHOTS."""
    if not (precision > 0.0 and precision < Float64.MAX):
        raise Error("QM-E913: argument 'precision' = " + String(precision) + " is out of range. Hint: pass a finite standard error > 0, or 0 for the exact value.")
    var worst = 0.0
    for o in range(len(observables)):
        var groups = observables[o].group_qwc()
        var total = 0.0
        for gi in range(len(groups)):
            var l1 = 0.0
            for ti in groups[gi]:
                var t = observables[o].terms[ti]
                if (t.x | t.z) != 0:
                    l1 += abs(t.coeff)
            total += l1 * l1
        worst = max(worst, total)
    var need = ceil(worst / (precision * precision))
    if need > Float64(MAX_ESTIMATOR_SHOTS):
        raise Error("QM-E913: precision " + String(precision) + " needs " + String(need) + " shots per setting for these observables (limit " + String(MAX_ESTIMATOR_SHOTS) + "). Hint: raise precision, scale the observable down, or use precision=0 for the exact value.")
    return max(1, Int(need))
