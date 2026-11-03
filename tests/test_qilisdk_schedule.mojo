"""Time-dependent analog schedules (qumo.algorithms.schedule).

Checks: a constant schedule equals exact_evolve; the sign convention exp(-iHt);
a commuting linear ramp against its closed form; second-order convergence for a
non-commuting schedule; interpolation rules; and the QM-E813 / QM-E814 errors.
"""

from std.math import cos, sin, sqrt
from std.testing import assert_equal, assert_true

from qumo.ir.pauli import PauliSum
from qumo.sim.statevector import StateVector
from qumo.algorithms.hamiltonians import exact_evolve
from qumo.algorithms.schedule import AnalogSchedule, INTERP_LINEAR, INTERP_STEP, evolve_schedule


def _plus(num_qubits: Int) -> StateVector:
    var psi = StateVector(num_qubits)
    var amp = 1.0 / sqrt(Float64(psi.size()))
    for i in range(psi.size()):
        psi.re[i] = amp
    return psi^


def _max_diff(a: StateVector, b: StateVector) -> Float64:
    var m = 0.0
    for i in range(a.size()):
        m = max(m, max(abs(a.re[i] - b.re[i]), abs(a.im[i] - b.im[i])))
    return m


def _phase_state(theta: Float64) -> StateVector:
    """(e^{-i theta} |0> + e^{+i theta} |1>) / sqrt(2): the state of |+> under exp(-i theta Z)."""
    var want = StateVector(1)
    var c = cos(theta) / sqrt(2.0)
    var s = sin(theta) / sqrt(2.0)
    want.re[0] = c
    want.im[0] = -s
    want.re[1] = c
    want.im[1] = s
    return want^


def _ctor_error(total: Float64) -> String:
    try:
        _ = AnalogSchedule(total, 1)
    except e:
        return String(e)
    return ""


def _add_error(mut s: AnalogSchedule, label: String, h: PauliSum, times: List[Float64], values: List[Float64], interpolation: Int) -> String:
    try:
        s.add_term(label, h, times, values, interpolation)
    except e:
        return String(e)
    return ""


def _evolve_error(s: AnalogSchedule, psi: StateVector, steps: Int) -> String:
    try:
        _ = evolve_schedule(s, psi, steps)
    except e:
        return String(e)
    return ""


def _expect_code(msg: String, code: String, what: String) raises:
    assert_true(msg.startswith(code), what + " should raise " + code + ", got: '" + msg + "'")
    assert_true(msg.find("Hint: ") > 0, what + " error has no Hint: " + msg)


def test_constant_schedule_matches_exact_evolve() raises:
    # Ising-like 2-qubit H = -Z0 Z1 - 0.7 (X0 + X1), constant in time.
    var h = PauliSum()
    h.add_label(-1.0, "ZZ")
    h.add_label(-0.7, "XI")
    h.add_label(-0.7, "IX")
    var times: List[Float64] = [0.0]
    var vals: List[Float64] = [1.0]
    var s = AnalogSchedule(2.0, 2)
    s.add_term("H", h, times, vals, INTERP_STEP)
    var psi = _plus(2)
    var want = exact_evolve(h, psi, 2.0)
    var counts: List[Int] = [1, 8]
    for k in range(len(counts)):
        var got = evolve_schedule(s, psi, counts[k])
        var d = _max_diff(got, want)
        assert_true(d < 1e-10, "constant schedule, steps=" + String(counts[k]) + ": max diff " + String(d))


def test_exact_evolve_sign_convention() raises:
    # exact_evolve applies exp(-i H t): for H = c Z, |0> gets e^{-i c t}, |1> gets e^{+i c t}.
    var h = PauliSum()
    h.add_label(1.0, "Z")
    var t = 0.5
    var got = exact_evolve(h, _plus(1), t)
    var d = _max_diff(got, _phase_state(t))
    assert_true(d < 1e-10, "exact_evolve sign convention: max diff " + String(d))


def test_commuting_linear_closed_form() raises:
    # H(t) = t Z0 on [0, 1]. The phase is exp(-i theta Z0) with theta = int_0^1 t dt = 1/2,
    # so the midpoint rule is exact for any step count.
    var h = PauliSum()
    h.add_label(1.0, "Z")
    var times: List[Float64] = [0.0, 1.0]
    var vals: List[Float64] = [0.0, 1.0]
    var s = AnalogSchedule(1.0, 1)
    s.add_term("ramp", h, times, vals, INTERP_LINEAR)
    var want = _phase_state(0.5)
    var counts: List[Int] = [1, 4, 16]
    for k in range(len(counts)):
        var got = evolve_schedule(s, _plus(1), counts[k])
        var d = _max_diff(got, want)
        assert_true(d < 1e-10, "commuting ramp, steps=" + String(counts[k]) + ": max diff " + String(d))


def test_convergence_non_commuting() raises:
    # H(t) = 2 cos(3t) X + 1.5 sin(2t) Z on [0, 1], coefficients by 26 knots (linear).
    # Knots every 0.04 are step boundaries of the 25-, 100- and 400-step grids, so each
    # step sees one linear piece. Finer or misaligned knots add kink errors that hide the
    # second-order rate at these step counts.
    var x = PauliSum()
    x.add_label(1.0, "X")
    var z = PauliSum()
    z.add_label(1.0, "Z")
    var times = List[Float64]()
    var ax = List[Float64]()
    var bz = List[Float64]()
    for i in range(26):
        var t = Float64(i) / 25.0
        times.append(t)
        ax.append(2.0 * cos(3.0 * t))
        bz.append(1.5 * sin(2.0 * t))
    var s = AnalogSchedule(1.0, 1)
    s.add_term("drive_x", x, times, ax, INTERP_LINEAR)
    s.add_term("field_z", z, times, bz, INTERP_LINEAR)
    var psi = _plus(1)
    var finest = evolve_schedule(s, psi, 400)
    var e25 = _max_diff(evolve_schedule(s, psi, 25), finest)
    var e100 = _max_diff(evolve_schedule(s, psi, 100), finest)
    assert_true(e100 > 0.0 and e25 > e100, "errors must shrink: e25=" + String(e25) + " e100=" + String(e100))
    var ratio = e25 / e100
    assert_true(ratio > 8.0, "x4 steps should cut the error about 16x, got ratio " + String(ratio))


def test_interpolation_rules() raises:
    var h = PauliSum()
    h.add_label(1.0, "Z")
    var times: List[Float64] = [0.0, 0.5, 1.0]
    var vals: List[Float64] = [2.0, 4.0, 1.0]
    var step = AnalogSchedule(1.0, 1)
    step.add_term("s", h, times, vals, INTERP_STEP)
    assert_true(abs(step.coefficient(0, 0.25) - 2.0) < 1e-12, "step: earlier knot holds")
    assert_true(abs(step.coefficient(0, 0.75) - 4.0) < 1e-12, "step: earlier knot holds")
    assert_true(abs(step.coefficient(0, 0.5) - 4.0) < 1e-12, "step: knot value at its own time")
    assert_true(abs(step.coefficient(0, 1.0) - 1.0) < 1e-12, "step: last knot at T")

    var lin = AnalogSchedule(1.0, 1)
    lin.add_term("l", h, times, vals, INTERP_LINEAR)
    assert_true(abs(lin.coefficient(0, 0.25) - 3.0) < 1e-12, "linear: midpoint is the mean of 2 and 4")
    assert_true(abs(lin.coefficient(0, 0.75) - 2.5) < 1e-12, "linear: midpoint is the mean of 4 and 1")
    assert_true(abs(lin.coefficient(0, 0.5) - 4.0) < 1e-12, "linear: knot value at its own time")

    # outside the knot range the nearest end value is held
    var inner_t: List[Float64] = [0.25, 0.75]
    var inner_v: List[Float64] = [1.0, 5.0]
    var held = AnalogSchedule(1.0, 1)
    held.add_term("h", h, inner_t, inner_v, INTERP_LINEAR)
    assert_true(abs(held.coefficient(0, 0.0) - 1.0) < 1e-12, "before the first knot: first value")
    assert_true(abs(held.coefficient(0, 0.9) - 5.0) < 1e-12, "after the last knot: last value")


def test_hamiltonian_at_merges_terms() raises:
    var z = PauliSum()
    z.add_label(1.0, "Z")
    var one: List[Float64] = [0.0]
    var c_one: List[Float64] = [1.0]
    var ramp_t: List[Float64] = [0.0, 1.0]
    var ramp_v: List[Float64] = [0.0, 2.0]
    var s = AnalogSchedule(1.0, 1)
    s.add_term("const", z, one, c_one, INTERP_STEP)
    s.add_term("ramp", z, ramp_t, ramp_v, INTERP_LINEAR)
    assert_equal(s.num_qubits(), 1)
    var hh = s.hamiltonian_at(0.5)
    assert_equal(len(hh), 1)
    assert_true(abs(hh.terms[0].coeff - 2.0) < 1e-12, "1 + ramp(0.5) = 1 + 1 = 2 for the merged Z term, got " + String(hh.terms[0].coeff))


def test_errors() raises:
    # total_time must be > 0 and finite
    _expect_code(_ctor_error(0.0), "QM-E813", "total_time 0")
    _expect_code(_ctor_error(-1.0), "QM-E813", "total_time < 0")
    var zero = 0.0
    _expect_code(_ctor_error(1.0 / zero), "QM-E813", "total_time inf")

    var h = PauliSum()
    h.add_label(1.0, "Z")
    var s = AnalogSchedule(2.0, 1)

    # knot times not strictly increasing
    var t_unsorted: List[Float64] = [0.0, 0.6, 0.3]
    var v3: List[Float64] = [1.0, 1.0, 1.0]
    _expect_code(_add_error(s, "a", h, t_unsorted, v3, INTERP_LINEAR), "QM-E813", "unsorted knots")
    var t_repeat: List[Float64] = [0.0, 0.5, 0.5]
    _expect_code(_add_error(s, "a", h, t_repeat, v3, INTERP_LINEAR), "QM-E813", "repeated knot time")

    # knot time outside [0, total_time]
    var t_late: List[Float64] = [0.0, 3.0]
    var v2: List[Float64] = [1.0, 1.0]
    _expect_code(_add_error(s, "a", h, t_late, v2, INTERP_LINEAR), "QM-E813", "knot after total_time")
    var t_early: List[Float64] = [-0.5, 1.0]
    _expect_code(_add_error(s, "a", h, t_early, v2, INTERP_LINEAR), "QM-E813", "knot before 0")

    # knot counts
    var t_one: List[Float64] = [0.0, 1.0]
    var v_short: List[Float64] = [1.0]
    _expect_code(_add_error(s, "a", h, t_one, v_short, INTERP_LINEAR), "QM-E813", "times and values differ in length")
    var t_none = List[Float64]()
    var v_none = List[Float64]()
    _expect_code(_add_error(s, "a", h, t_none, v_none, INTERP_LINEAR), "QM-E813", "no knots")

    # unknown interpolation code
    _expect_code(_add_error(s, "a", h, t_one, v2, 7), "QM-E813", "unknown interpolation code")

    # duplicate label: the first add succeeds, the second is refused
    var ok = _add_error(s, "dup", h, t_one, v2, INTERP_LINEAR)
    assert_equal(ok, "", "first add of 'dup' should succeed")
    _expect_code(_add_error(s, "dup", h, t_one, v2, INTERP_LINEAR), "QM-E813", "duplicate label")

    # every term acts on the same number of qubits
    var zz = PauliSum()
    zz.add_label(1.0, "ZZ")
    _expect_code(_add_error(s, "two_qubit", zz, t_one, v2, INTERP_LINEAR), "QM-E814", "term on another qubit count")

    # steps must be >= 1
    var psi1 = _plus(1)
    _expect_code(_evolve_error(s, psi1, 0), "QM-E814", "steps = 0")
    _expect_code(_evolve_error(s, psi1, -3), "QM-E814", "steps < 0")

    # the state must have the schedule's qubit count
    _expect_code(_evolve_error(s, _plus(2), 4), "QM-E814", "state with 2 qubits on a 1-qubit schedule")


def test_mixed_width_terms() raises:
    """A one-qubit drive and a two-qubit coupling share a two-qubit schedule. The
    width used to be inferred from the terms, which refused this pair."""
    var drive = PauliSum()
    drive.add_label(1.0, "IX")  # X on qubit 0: the rightmost letter is qubit 0
    var coupling = PauliSum()
    coupling.add_label(1.0, "ZZ")
    var times = List[Float64]()
    times.append(0.0)
    times.append(1.0)
    var vals = List[Float64]()
    vals.append(0.0)
    vals.append(1.0)
    var s = AnalogSchedule(1.0, 2)
    s.add_term("drive", drive, times, vals, INTERP_LINEAR)
    s.add_term("coupling", coupling, times, vals, INTERP_LINEAR)
    assert_equal(s.num_qubits(), 2)
    var out = evolve_schedule(s, _plus(2), 8)
    var norm = 0.0
    for i in range(len(out.re)):
        norm += out.re[i] * out.re[i] + out.im[i] * out.im[i]
    assert_true(abs(norm - 1.0) < 1e-10, "norm after the mixed schedule: " + String(norm))


def main() raises:
    test_constant_schedule_matches_exact_evolve()
    test_exact_evolve_sign_convention()
    test_commuting_linear_closed_form()
    test_convergence_non_commuting()
    test_interpolation_rules()
    test_hamiltonian_at_merges_terms()
    test_errors()
    test_mixed_width_terms()
    print("test_qilisdk_schedule OK")
