"""Tests for qumo.algorithms.mitigation: unitary folding (global / from left /
at random), zero-noise extrapolation factories and the ZNE driver.
Prints `test_mitigation OK`."""

from std.math import pi, exp
from std.testing import assert_true, assert_almost_equal, assert_equal

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.pauli import PauliSum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve
from qumo.qec.characterization import NoiseModel
from qumo.algorithms.library import ghz
from qumo.algorithms.mitigation import (
    fold_gates_global, fold_gates_from_left, fold_gates_at_random, fold_gates, folded_scale,
    richardson_extrapolate, linear_extrapolate, poly_extrapolate, exp_extrapolate, extrapolate,
    sampled_expectation, noisy_expectation, zne_expectation, ZNEResult,
)


def _err_code(msg: String) -> String:
    return String(msg[byte=0:7]) if msg.byte_length() >= 7 else msg


def _max_diff(a: StateVector, b: StateVector) -> Float64:
    var d = 0.0
    for i in range(a.size()):
        d = max(d, abs(a.re[i] - b.re[i]))
        d = max(d, abs(a.im[i] - b.im[i]))
    return d


def _mixed_circuit() -> Circuit:
    """Every gate family the folding code must invert (incl. U and a global phase)."""
    var c = Circuit(4, name="mixed")
    c.h(0)
    c.sx(1)
    c.t(2)
    c.s(3)
    c.u(0.3, -0.7, 1.1, 0)
    c.cx(0, 1)
    c.cp(0.4, 1, 2)
    c.crx(-0.9, 2, 3)
    c.rzz(0.25, 0, 3)
    c.iswap(1, 2)
    c.ecr(2, 3)
    c.ry(0.6, 1)
    c.ccx(0, 1, 3)
    c.tdg(0)
    c.sdg(1)
    c.rx(1.3, 2)
    c.global_phase = 0.37
    return c^


def test_folding_equivalence() raises:
    var c = _mixed_circuit()
    var ref_sv = evolve(c)
    var d = c.num_unitary_gates()
    var scales: List[Float64] = [1.0, 3.0, 5.0, 2.5, 1.7, 4.2]
    var methods: List[String] = ["global", "left", "random"]
    for m in methods:
        for s in scales:
            var f = fold_gates(c, s, m, 11)
            var sv = evolve(f)
            assert_true(_max_diff(sv, ref_sv) < 1e-12, "folded statevector differs: " + m + " scale " + String(s))
            var target = s * Float64(d)
            var got = Float64(f.num_unitary_gates())
            assert_true(abs(got - target) <= 1.0 + 1e-9, "gate count " + String(got) + " vs " + String(target) + " (" + m + ")")
            assert_true(abs(folded_scale(c, f) - got / Float64(d)) < 1e-15)
    # odd integer global folds are exact multiples
    assert_equal(fold_gates_global(c, 3.0).num_unitary_gates(), 3 * d)
    assert_equal(fold_gates_global(c, 5.0).num_unitary_gates(), 5 * d)
    assert_equal(fold_gates_global(c, 1.0).num_unitary_gates(), d)
    # random folding is seeded (reproducible) and seed-dependent
    var r1 = fold_gates_at_random(c, 2.0, 5)
    var r2 = fold_gates_at_random(c, 2.0, 5)
    assert_equal(len(r1), len(r2))
    var same = True
    for i in range(len(r1)):
        if r1.ops[i].kind != r2.ops[i].kind:
            same = False
    assert_true(same)
    print("  folding: unitary-equivalent at 1e-12 for global/left/random, gates =", d, "->",
          fold_gates_global(c, 5.0).num_unitary_gates(), "at scale 5")


def test_folding_symbolic_and_measurements() raises:
    var c = Circuit(2, name="sym")
    var th = c.parameter("theta")
    c.h(0)
    c.ry(th, 1)
    c.cx(0, 1)
    c.rz(th * 2.0, 0)
    c.measure(0, 0)
    c.measure(1, 1)
    var f = fold_gates_global(c, 3.0)
    assert_equal(f.num_params(), 1)
    assert_equal(f.num_measurements, 2)
    assert_equal(f.count(Op.MEASURE), 2)
    # measurements are the last two instructions
    assert_equal(f.ops[len(f.ops) - 1].kind, Op.MEASURE)
    assert_equal(f.ops[len(f.ops) - 2].kind, Op.MEASURE)
    var p: List[Float64] = [0.83]
    assert_true(_max_diff(evolve(f, p), evolve(c, p)) < 1e-12)
    var fl = fold_gates_from_left(c, 3.0)
    assert_true(_max_diff(evolve(fl, p), evolve(c, p)) < 1e-12)


def test_extrapolation() raises:
    # Richardson is exact on a polynomial of degree len-1
    var s3: List[Float64] = [1.0, 3.0, 5.0]
    var v3 = List[Float64]()
    for s in s3:
        v3.append(0.9 - 0.04 * s + 0.003 * s * s)
    assert_almost_equal(richardson_extrapolate(s3, v3), 0.9, atol=1e-12)
    var s4: List[Float64] = [1.0, 1.5, 2.0, 3.0]
    var v4 = List[Float64]()
    for s in s4:
        v4.append(0.5 + 0.1 * s - 0.2 * s * s + 0.01 * s * s * s)
    assert_almost_equal(richardson_extrapolate(s4, v4), 0.5, atol=1e-12)
    # least squares: exact on matching models
    var s5: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var lin = List[Float64]()
    var quad = List[Float64]()
    var ex = List[Float64]()
    for s in s5:
        lin.append(0.8 - 0.07 * s)
        quad.append(0.8 - 0.07 * s + 0.002 * s * s)
        ex.append(0.1 + 0.85 * exp(-0.3 * s))
    assert_almost_equal(linear_extrapolate(s5, lin), 0.8, atol=1e-12)
    assert_almost_equal(poly_extrapolate(s5, quad, 2), 0.8, atol=1e-12)
    assert_almost_equal(exp_extrapolate(s5, ex, 0.1), 0.95, atol=1e-12)
    assert_almost_equal(extrapolate(s5, quad, "poly2"), 0.8, atol=1e-12)
    # negative side of the asymptote
    var exn = List[Float64]()
    for s in s5:
        exn.append(-0.6 * exp(-0.2 * s))
    assert_almost_equal(exp_extrapolate(s5, exn, 0.0), -0.6, atol=1e-12)


def test_zne_ghz() raises:
    """GHZ-6 <Z^6>, depolarizing gate errors e1 = 1e-3, e2 = 1e-2: ZNE at least halves the bias."""
    var c = ghz(6)
    var h = PauliSum()
    h.add_label(1.0, "ZZZZZZ")
    var nm = NoiseModel(6)
    for q in range(6):
        nm.gate_error_1q[q] = 1e-3
    nm.gate_error_2q = 1e-2
    var scales: List[Float64] = [1.0, 3.0, 5.0]
    var r = zne_expectation(c, h, nm, scales, 20000, 7)
    var ideal = evolve(c).expectation(h)
    assert_almost_equal(ideal, 1.0, atol=1e-12)
    var raw_bias = abs(r.raw() - ideal)
    var zne_bias = abs(r.estimate - ideal)
    print("  zne ghz6: raw", r.raw(), "zne", r.estimate, "bias", raw_bias, "->", zne_bias)
    assert_true(raw_bias > 0.02, "noise should bias the raw value")
    assert_true(zne_bias < 0.5 * raw_bias, "ZNE must at least halve the bias")
    # values decrease with the noise scale; achieved scales equal requested (odd global folds)
    assert_true(r.values[0] > r.values[1] and r.values[1] > r.values[2])
    for i in range(3):
        assert_almost_equal(r.achieved_scales[i], scales[i], atol=1e-12)
    # deterministic: same seed, same numbers
    var r2 = zne_expectation(c, h, nm, scales, 20000, 7)
    assert_equal(r2.estimate, r.estimate)
    # linear and exponential factories on the same data are also better than raw
    var lin = linear_extrapolate(r.achieved_scales, r.values)
    var ex = exp_extrapolate(r.achieved_scales, r.values, 0.0)
    assert_true(abs(lin - ideal) < raw_bias and abs(ex - ideal) < raw_bias)
    # noiseless sampling agrees with the exact value within 4 sigma
    var est = sampled_expectation(c, h, 4000, 3)
    assert_true(abs(est[0] - ideal) <= 4.0 * est[1] + 1e-12)
    # a multi-group observable (X and Z groups) through noisy_expectation
    var h2 = PauliSum()
    h2.add_label(1.0, "XXXXXX")
    h2.add_label(0.5, "IIIIZZ")
    h2.add_label(0.25, "IIIIII")
    var exact2 = evolve(c).expectation(h2)
    var noiseless = NoiseModel(6)
    var est2 = noisy_expectation(c, h2, noiseless, 8000, 5)
    assert_true(abs(est2[0] - exact2) <= 4.0 * est2[1] + 1e-12, "noisy_expectation (zero noise) within 4 sigma")


def test_errors() raises:
    var codes = List[String]()
    var c = ghz(3)
    try:
        _ = fold_gates_global(c, 0.5)
    except e:
        codes.append(_err_code(String(e)))
    var noisy = ghz(3)
    noisy.depolarize1(0.01, 0)
    try:
        _ = fold_gates_global(noisy, 3.0)
    except e:
        codes.append(_err_code(String(e)))
    var mid = Circuit(2)
    mid.h(0)
    mid.measure(0, 0)
    mid.cx(0, 1)
    try:
        _ = fold_gates_from_left(mid, 3.0)
    except e:
        codes.append(_err_code(String(e)))
    try:
        _ = fold_gates(c, 3.0, "sideways")
    except e:
        codes.append(_err_code(String(e)))
    var s2: List[Float64] = [1.0, 3.0]
    var v1: List[Float64] = [0.9]
    try:
        _ = richardson_extrapolate(s2, v1)
    except e:
        codes.append(_err_code(String(e)))
    var dup: List[Float64] = [1.0, 1.0]
    var v2: List[Float64] = [0.9, 0.8]
    try:
        _ = richardson_extrapolate(dup, v2)
    except e:
        codes.append(_err_code(String(e)))
    var mixed: List[Float64] = [0.5, -0.2]
    try:
        _ = exp_extrapolate(s2, mixed, 0.0)
    except e:
        codes.append(_err_code(String(e)))
    try:
        _ = extrapolate(s2, v2, "cubic-spline")
    except e:
        codes.append(_err_code(String(e)))
    var expected: List[String] = ["QM-E801", "QM-E801", "QM-E801", "QM-E801", "QM-E802", "QM-E802", "QM-E801", "QM-E801"]
    assert_equal(len(codes), len(expected))
    for i in range(len(expected)):
        assert_equal(codes[i], expected[i])


def main() raises:
    test_folding_equivalence()
    test_folding_symbolic_and_measurements()
    test_extrapolation()
    test_zne_ghz()
    test_errors()
    print("test_mitigation OK")
