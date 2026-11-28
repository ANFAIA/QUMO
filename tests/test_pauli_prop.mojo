"""Tests for qumo.sim.pauli_prop.

References: the state vector (`qumo.sim.runner.expectation`) for unitary circuits and
the density matrix (`qumo.sim.density.density_evolve`) for Pauli noise; both are
separate code paths from the propagation. Wide cases use closed forms.
"""

from std.math import pi, cos, sin
from std.testing import assert_true, assert_equal
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.pauli import PauliSum
from qumo.core.rng import Rng
from qumo.sim.runner import expectation
from qumo.sim.density import density_evolve
from qumo.sim.pauli_prop import propagate_pauli, PropagationOptions


def _angle(mut rng: Rng) -> Float64:
    return (rng.uniform() * 2.0 - 1.0) * pi


def add_random_gate(mut c: Circuit, mut rng: Rng, n: Int):
    var q = rng.below(n)
    var u = (q + 1 + rng.below(n - 1)) % n
    var w = (u + 1 + rng.below(n - 2)) % n if n >= 3 else u
    while n >= 3 and (w == q or w == u):
        w = (w + 1) % n
    var r = rng.below(34)
    if r == 0: c.h(q)
    elif r == 1: c.s(q)
    elif r == 2: c.sdg(q)
    elif r == 3: c.sx(q)
    elif r == 4: c.sxdg(q)
    elif r == 5: c.x(q)
    elif r == 6: c.y(q)
    elif r == 7: c.z(q)
    elif r == 8: c.t(q)
    elif r == 9: c.tdg(q)
    elif r == 10: c.rx(_angle(rng), q)
    elif r == 11: c.ry(_angle(rng), q)
    elif r == 12: c.rz(_angle(rng), q)
    elif r == 13: c.p(_angle(rng), q)
    elif r == 14: c.u(_angle(rng), _angle(rng), _angle(rng), q)
    elif r == 15: c.cx(q, u)
    elif r == 16: c.cy(q, u)
    elif r == 17: c.cz(q, u)
    elif r == 18: c.ch(q, u)
    elif r == 19: c.swap(q, u)
    elif r == 20: c.iswap(q, u)
    elif r == 21: c.add(Op.ISWAPDG, q, u)
    elif r == 22: c.ecr(q, u)
    elif r == 23: c.cp(_angle(rng), q, u)
    elif r == 24: c.crx(_angle(rng), q, u)
    elif r == 25: c.cry(_angle(rng), q, u)
    elif r == 26: c.crz(_angle(rng), q, u)
    elif r == 27: c.rxx(_angle(rng), q, u)
    elif r == 28: c.ryy(_angle(rng), q, u)
    elif r == 29: c.rzz(_angle(rng), q, u)
    elif n >= 3 and r == 30: c.ccx(q, u, w)
    elif n >= 3 and r == 31: c.cswap(q, u, w)
    elif n >= 3 and r == 32: c.ccz(q, u, w)
    else: c.id(q)


def random_observable(n: Int, terms: Int, mut rng: Rng) -> PauliSum:
    var o = PauliSum()
    for _ in range(terms):
        var x: UInt64 = 0
        var z: UInt64 = 0
        for q in range(n):
            var r = rng.below(4)
            if r == 1 or r == 2: x |= UInt64(1) << UInt64(q)
            if r == 2 or r == 3: z |= UInt64(1) << UInt64(q)
        o.add(rng.uniform() * 2.0 - 1.0, x, z)
    return o^


def test_every_gate_on_all_paulis() raises:
    """Each opcode after a random 3-qubit state: all 63 Pauli strings."""
    var rng = Rng(11)
    var worst = 0.0
    for trial in range(200):
        var c = Circuit(3)
        for q in range(3):
            c.u(_angle(rng), _angle(rng), _angle(rng), q)
        c.cx(0, 1)
        c.cx(1, 2)
        for q in range(3):
            c.u(_angle(rng), _angle(rng), _angle(rng), q)
        add_random_gate(c, rng, 3)
        for code in range(1, 64):
            var o = PauliSum()
            var x: UInt64 = 0
            var z: UInt64 = 0
            for q in range(3):
                var d = (code >> (2 * q)) & 3
                if d == 1 or d == 2: x |= UInt64(1) << UInt64(q)
                if d == 2 or d == 3: z |= UInt64(1) << UInt64(q)
            o.add(1.0, x, z)
            var ref_v = expectation(c, o)
            var r = propagate_pauli(c, o)
            worst = max(worst, abs(r.expectation - ref_v))
            assert_true(r.is_exact())
        _ = trial
    print("  per-gate worst |error| =", worst)
    assert_true(worst < 1e-12)


def test_random_circuits_exact() raises:
    """Seeded random circuits, 2..10 qubits, no truncation, against the state vector."""
    var worst = 0.0
    var peak = 0
    for seed in range(40):
        var rng = Rng(UInt64(1000 + seed))
        var n = 2 + seed % 9
        var c = Circuit(n)
        var gates = 12 + 3 * n
        for _ in range(gates):
            add_random_gate(c, rng, n)
        var o = random_observable(n, 4, rng)
        var ref_v = expectation(c, o)
        var r = propagate_pauli(c, o)
        assert_true(r.is_exact())
        assert_equal(r.discarded_abs_sum, 0.0)
        worst = max(worst, abs(r.expectation - ref_v))
        peak = max(peak, r.peak_terms)
    print("  random circuits worst |error| =", worst, " peak strings =", peak)
    assert_true(worst < 1e-10)


def test_returned_operator_is_heisenberg() raises:
    """The returned U^dag O U gives the right value in other states too: prepend a layer."""
    var rng = Rng(5)
    var n = 5
    var c = Circuit(n)
    for _ in range(30):
        add_random_gate(c, rng, n)
    var o = random_observable(n, 3, rng)
    var r = propagate_pauli(c, o)
    var prep = Circuit(n)
    for q in range(n):
        prep.u(_angle(rng), _angle(rng), _angle(rng), q)
    var full = prep.copy()
    full.compose(c)
    var a = expectation(full, o)
    var b = expectation(prep, r.observable)
    assert_true(abs(a - b) < 1e-10)


def test_noise_matches_density_matrix() raises:
    var worst = 0.0
    for seed in range(12):
        var rng = Rng(UInt64(77 + seed))
        var n = 3 + seed % 3
        var c = Circuit(n)
        for _ in range(25):
            add_random_gate(c, rng, n)
            var q = rng.below(n)
            var p = rng.uniform() * 0.2
            var r = rng.below(6)
            if r == 0: c.x_error(p, q)
            elif r == 1: c.y_error(p, q)
            elif r == 2: c.z_error(p, q)
            elif r == 3: c.depolarize1(p, q)
            elif r == 4: c.pauli_channel_1(p * 0.5, p * 0.3, p * 0.2, q)
            else: c.depolarize2(p, q, (q + 1) % n)
        var o = random_observable(n, 4, rng)
        var rho = density_evolve(c)
        var ref_v = rho.expectation(o)
        var res = propagate_pauli(c, o)
        worst = max(worst, abs(res.expectation - ref_v))
    print("  noisy worst |error| vs density matrix =", worst)
    assert_true(worst < 1e-10)


def test_truncation_bound_holds() raises:
    """|exact - truncated| <= discarded_abs_sum, and the bound is not vacuous."""
    var violations = 0
    var truncated_runs = 0
    var ratio_sum = 0.0
    for seed in range(30):
        var rng = Rng(UInt64(500 + seed))
        var n = 6 + seed % 5
        var c = Circuit(n)
        for _ in range(8 * n):
            add_random_gate(c, rng, n)
        var o = random_observable(n, 3, rng)
        var exact = expectation(c, o)
        var opts = PropagationOptions(min_abs_coeff=1e-2) if seed % 2 == 0 else PropagationOptions(max_weight=3)
        var r = propagate_pauli(c, o, opts)
        var err = abs(r.expectation - exact)
        if err > r.discarded_abs_sum + 1e-12:
            violations += 1
            print("  bound violated: seed", seed, "err", err, "bound", r.discarded_abs_sum)
        if r.discarded_terms > 0:
            truncated_runs += 1
            assert_true(r.discarded_abs_sum > 0.0)
            assert_true(not r.is_exact())
            ratio_sum += err / r.discarded_abs_sum
        else:
            assert_true(err < 1e-10)
    print("  truncated runs:", truncated_runs, "of 30; mean error / bound =", ratio_sum / Float64(max(1, truncated_runs)))
    assert_equal(violations, 0)
    assert_true(truncated_runs >= 20)


def test_truncation_with_noise_bound_holds() raises:
    var rng = Rng(9)
    var n = 6
    var c = Circuit(n)
    for _ in range(60):
        add_random_gate(c, rng, n)
        c.depolarize1(0.02, rng.below(n))
    var o = random_observable(n, 2, rng)
    var exact = density_evolve(c).expectation(o)
    var r = propagate_pauli(c, o, PropagationOptions(min_abs_coeff=5e-3))
    assert_true(r.discarded_terms > 0)
    assert_true(abs(r.expectation - exact) <= r.discarded_abs_sum + 1e-12)


def test_rotation_branching_formula() raises:
    """RX(theta)^dag Z RX(theta) = cos(theta) Z + sin(theta) Y, by hand."""
    var c = Circuit(1)
    var th = 0.7
    c.rx(th, 0)
    var o = PauliSum()
    o.add_label(1.0, "Z")
    var r = propagate_pauli(c, o)
    # RX^dag Z RX = cos(th) Z + sin(th) Y  (Y = -i Z X)
    assert_equal(len(r.observable), 2)
    for t in r.observable.terms:
        if t.x == 0:
            assert_true(abs(t.coeff - cos(th)) < 1e-15)
        else:
            assert_true(t.x == 1 and t.z == 1)
            assert_true(abs(t.coeff - sin(th)) < 1e-15)


def test_quarter_turns_do_not_branch() raises:
    var c = Circuit(2)
    c.rx(pi / 2, 0)
    c.rzz(pi, 0, 1)
    c.ry(-pi / 2, 1)
    var o = PauliSum()
    o.add_label(1.0, "ZZ")
    var r = propagate_pauli(c, o)
    assert_equal(len(r.observable), 1)
    assert_true(abs(r.expectation - expectation(c, o)) < 1e-14)


def test_64_qubits() raises:
    """GHZ on 64 qubits with T layers that cancel; closed forms."""
    var n = 64
    var c = Circuit(n)
    c.h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    var zz = PauliSum()
    zz.add(1.0, 0, (UInt64(1) << 63) | 1)
    var xs = PauliSum()
    xs.add(1.0, ~UInt64(0), 0)
    assert_true(abs(propagate_pauli(c, zz).expectation - 1.0) < 1e-14)
    assert_true(abs(propagate_pauli(c, xs).expectation - 1.0) < 1e-14)
    # RZ(theta) on qubit 63 of the GHZ: <X...X> = cos(theta)
    var d = c.copy()
    d.rz(0.3, 63)
    d.t(10)
    d.tdg(10)
    var r = propagate_pauli(d, xs)
    assert_true(abs(r.expectation - cos(0.3)) < 1e-14)
    assert_true(r.is_exact())


def _err(c: Circuit, o: PauliSum, opts: PropagationOptions = PropagationOptions()) -> String:
    try:
        _ = propagate_pauli(c, o, opts)
    except e:
        return String(e)
    return ""


def _z(label: String) raises -> PauliSum:
    var o = PauliSum()
    o.add_label(1.0, label)
    return o^


def test_refusals() raises:
    assert_true(_err(Circuit(65), _z("Z")).startswith("QM-E430"))

    var rs = Circuit(2)
    rs.reset(0)
    assert_true(_err(rs, _z("Z")).startswith("QM-E430"))

    var mid = Circuit(2, 1)
    mid.measure(0, 0)
    mid.h(0)
    assert_true(_err(mid, _z("Z")).startswith("QM-E430"))

    var cond = Circuit(2, 1)
    cond.x(0)
    cond.c_if(0, 1)
    assert_true(_err(cond, _z("Z")).startswith("QM-E430"))

    var pc2 = Circuit(2)
    pc2.pauli_channel_2(List[Float64](length=15, fill=0.01), 0, 1)
    assert_true(_err(pc2, _z("ZZ")).startswith("QM-E430"))

    var cap = Circuit(8)
    for q in range(8):
        cap.rx(0.3, q)
    assert_true(_err(cap, _z("ZZZZZZZZ"), PropagationOptions(max_terms=50)).startswith("QM-E431"))

    assert_true(_err(Circuit(2), _z("ZII")).startswith("QM-E405"))

    # terminal measurements are ignored, as expectation() does
    var c = Circuit(2, 2)
    c.h(0)
    c.cx(0, 1)
    c.measure_all()
    assert_true(abs(propagate_pauli(c, _z("XX")).expectation - 1.0) < 1e-14)


def test_symbolic_parameters() raises:
    var c = Circuit(2)
    var th = c.parameter("theta")
    c.h(0)
    c.cx(0, 1)
    c.ry(th, 1)
    var o = PauliSum()
    o.add_label(1.0, "ZZ")
    var vals: List[Float64] = [0.4]
    var r = propagate_pauli(c, o, PropagationOptions(), vals)
    assert_true(abs(r.expectation - expectation(c, o, vals)) < 1e-14)


def main() raises:
    test_rotation_branching_formula()
    test_quarter_turns_do_not_branch()
    test_every_gate_on_all_paulis()
    test_random_circuits_exact()
    test_returned_operator_is_heisenberg()
    test_noise_matches_density_matrix()
    test_truncation_bound_holds()
    test_truncation_with_noise_bound_holds()
    test_64_qubits()
    test_refusals()
    test_symbolic_parameters()
    print("test_pauli_prop OK")
