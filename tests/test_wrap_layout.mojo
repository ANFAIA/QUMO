"""Tests for target angle bounds, wrap_angles, calibration and post_layout."""

from std.math import pi, log
from std.testing import assert_equal, assert_true
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.runner import evolve
from qumo.sim.rng import Rng
from qumo.compiler.target import Target, parse_target
from qumo.compiler.wrap import wrap_angles
from qumo.compiler.layout_fidelity import post_layout
from qumo.compiler.pass_manager import CompileResult, transpile, transpile_ai, compile_spec, preset_pipeline, preset_pipeline_ai, conformance_error
from qumo.bench.circuits import quantum_volume


def prep(n: Int, seed: UInt64) -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n)
    for q in range(n):
        c.ry(rng.uniform() * 3.0, q)
        c.rz(rng.uniform() * 6.0, q)
    for q in range(n - 1):
        c.cx(q, q + 1)
    return c^


def exact_diff(a: Circuit, b: Circuit, n: Int) raises -> Float64:
    """Largest amplitude difference on a generic input, global phase included."""
    var pa = prep(n, 9)
    var pb = pa.copy()
    pa.compose(a)
    pb.compose(b)
    var sa = evolve(pa)
    var sb = evolve(pb)
    var worst = 0.0
    for i in range(sa.size()):
        worst = max(worst, abs(sa.re[i] - sb.re[i]), abs(sa.im[i] - sb.im[i]))
    return worst


def test_wrap_exact() raises:
    var kinds2 = [Op.RZZ, Op.RXX, Op.RYY]
    var kinds1 = [Op.RZ, Op.RX, Op.RY, Op.P]
    var rng = Rng(3)
    var t = Target.all_to_all(2)
    t.set_angle_bounds(Op.RZZ, 0.0, pi / 2)
    t.set_angle_bounds(Op.RXX, 0.0, pi / 2)
    t.set_angle_bounds(Op.RYY, 0.0, pi / 2)
    t.set_angle_bounds(Op.RZ, 0.0, pi / 2)
    t.set_angle_bounds(Op.RX, 0.0, pi / 2)
    t.set_angle_bounds(Op.RY, 0.0, pi / 2)
    t.set_angle_bounds(Op.P, 0.0, pi / 2)
    for it in range(120):
        var theta = (rng.uniform() - 0.5) * 24.0
        if it < 8:
            theta = Float64(it - 4) * pi / 2  # exact multiples
        for k in kinds2:
            var c = Circuit(2)
            c.add(k, 0, 1, -1, theta)
            var w = c.copy()
            _ = wrap_angles(w, t)
            assert_true(exact_diff(c, w, 2) < 1e-10, "wrap exact, theta " + String(theta))
            assert_equal(conformance_error(w, t), "", "all angles in range")
        for k in kinds1:
            var c = Circuit(2)
            c.add(k, 0, -1, -1, theta)
            var w = c.copy()
            _ = wrap_angles(w, t)
            assert_true(exact_diff(c, w, 2) < 1e-10, "wrap exact 1q, theta " + String(theta))
            assert_equal(conformance_error(w, t), "", "all angles in range")
    print("  wrap_angles exact with global phase for rzz, rxx, ryy, rz, rx, ry, p")


def test_periodic_gates() raises:
    var t = Target.all_to_all(2)
    t.set_angle_bounds(Op.CP, -pi, pi)
    t.set_angle_bounds(Op.CRZ, -2 * pi, 2 * pi)
    var c = Circuit(2)
    c.cp(9.0, 0, 1)
    c.crz(15.0, 0, 1)
    var w = c.copy()
    assert_equal(wrap_angles(w, t), 2, "both change")
    assert_true(exact_diff(c, w, 2) < 1e-10, "cp period 2 pi, crz period 4 pi")
    t.set_angle_bounds(Op.CP, 0.0, 0.5)
    var e = Circuit(2)
    e.cp(2.0, 0, 1)
    var raised = False
    try:
        _ = wrap_angles(e, t)
    except err:
        raised = String(err).startswith("QM-E622")
    assert_true(raised, "QM-E622 when no period reaches the interval")
    print("  periodic gates and QM-E622")


def test_spec_and_presets() raises:
    var t = parse_target("line:3/rz,sx,x,cz,rzz[0:pi/2]")
    assert_true(t.has_angle_bounds() and t.angle_bound[Op.RZZ], "spec sets the bound")
    assert_true(abs(t.angle_hi[Op.RZZ] - pi / 2) < 1e-15, "pi/2 parsed")
    var c = Circuit(3)
    c.rzz(2.5, 0, 1)
    c.rzz(-0.7, 1, 2)
    var plain = parse_target("line:3/rz,sx,x,cz,rzz")
    assert_true(preset_pipeline(2, plain).find("wrap_angles") < 0, "no bounds: pipeline unchanged")
    for lvl in range(4):
        var r = transpile(c, t, lvl)
        assert_equal(conformance_error(r.circuit, t), "", "preset respects bounds, level " + String(lvl))
    # check reports an out-of-range angle without wrap_angles
    var raised = False
    try:
        _ = compile_spec(c, t, "check")
    except e:
        raised = String(e).startswith("QM-E615") and String(e).find("outside") >= 0
    assert_true(raised, "check sees the angle")
    var bad = False
    try:
        _ = parse_target("line:3/rz,rzz[1:0]")
    except e:
        bad = String(e).startswith("QM-E621")
    assert_true(bad, "QM-E621 empty interval")
    print("  spec syntax, presets, check")


def test_ai_presets_share_the_tail() raises:
    var t = parse_target("line:3/rz,sx,x,cz,rzz[0:pi/2]")
    var c = Circuit(3)
    c.rzz(2.5, 0, 1)
    c.rzz(-0.7, 1, 2)
    for lvl in range(1, 4):
        assert_true(preset_pipeline_ai(lvl, t).find("wrap_angles") >= 0, "ai preset wraps angles")
        var r = transpile_ai(c, t, lvl)
        assert_equal(conformance_error(r.circuit, t), "", "ai preset respects bounds, level " + String(lvl))
    print("  AI presets take the device tail")


def test_compiler_refuses_correlated_error() raises:
    var c = Circuit(2)
    c.correlated_error(0.1, [0, 1], [1, 3])
    var raised = False
    try:
        _ = transpile(c, Target.line(2), 1)
    except e:
        raised = String(e).startswith("QM-E624")
    assert_true(raised, "E needs a compiled circuit")
    var far = Circuit(2)
    far.correlated_error(0.1, [5], [1])
    var outside = False
    try:
        far.validate()
    except e:
        outside = String(e).startswith("QM-E001")
    assert_true(outside, "validate reads the Pauli table")
    print("  compiler refuses E, validate reads its table")


def calibrated_line() raises -> Target:
    """A 6-qubit line whose edges 3-4 and 4-5 are ten times better than the rest."""
    var t = Target.line(6, "any")
    t.set_edge_error(0, 1, 0.05)
    t.set_edge_error(1, 2, 0.05)
    t.set_edge_error(2, 3, 0.05)
    t.set_edge_error(3, 4, 0.005)
    t.set_edge_error(4, 5, 0.005)
    return t^


def test_post_layout_known_embedding() raises:
    var t = calibrated_line()
    assert_true(t.is_calibrated(), "non-uniform errors")
    var c = Circuit(6, 3)
    c.cx(0, 1)
    c.cx(1, 2)
    c.measure(0, 0)
    c.measure(1, 1)
    c.measure(2, 2)
    var before = t.estimate_fidelity(c)
    var wm = List[Int]()
    var gain = post_layout(c, t, wm)
    var after = t.estimate_fidelity(c)
    assert_true(abs(log(after / before) - gain) < 1e-12, "reported gain is the fidelity change")
    # the chain must land on the good edges: physical 3, 4, 5 (either direction)
    assert_true(c.ops[0].q0 >= 3 and c.ops[0].q1 >= 3 and c.ops[1].q1 >= 3, "good edges chosen")
    assert_true(abs(after - 0.995 * 0.995) < 1e-12, "both gates on 0.5% edges")
    print("  post_layout finds the known best embedding")


def test_post_layout_properties() raises:
    var t = Target.grid(3, 3, "sc_cz")
    var rng = Rng(11)
    var edge_e = List[Float64]()
    for _ in range(t.num_edges()):
        edge_e.append(0.002 + 0.03 * rng.uniform())
    var e1 = List[Float64]()
    var ro = List[Float64]()
    for _ in range(9):
        e1.append(1e-4 + 1e-3 * rng.uniform())
        ro.append(0.005 + 0.05 * rng.uniform())
    t.set_calibration(e1, edge_e, ro)
    t.duration_ns[Op.SX] = 30.0
    t.duration_ns[Op.X] = 30.0
    var uniform = Target.grid(3, 3, "sc_cz")
    uniform.duration_ns[Op.SX] = 30.0
    uniform.duration_ns[Op.X] = 30.0
    for seed in range(6):
        var c = quantum_volume(5, 3, UInt64(100 + seed))
        c.measure_all()
        var plain = transpile(c, uniform, 2, 7)
        var r = transpile(c, t, 2, 7)
        # never worse than the same output without the pass
        assert_true(t.estimate_fidelity(r.circuit) >= t.estimate_fidelity(plain.circuit) - 1e-12, "post_layout never lowers the estimate")
        assert_equal(conformance_error(r.circuit, t), "", "stays on the coupling map")
        assert_equal(r.circuit.num_2q_gates(), plain.circuit.num_2q_gates(), "gates untouched")
        # same outcome distribution, read through the layouts
        var a = sample_dist(r, c)
        assert_true(a > 1 - 1e-9, "same logical state after relabelling")
    print("  post_layout keeps the circuit, never lowers the estimate")


def sample_dist(r: CompileResult, c: Circuit) raises -> Float64:
    """Fidelity between the logical state of `c` and the compiled one, through layouts."""
    var bare = Circuit(c.num_qubits)
    for g in c.ops:
        if g.kind != Op.MEASURE:
            bare.append(g)
    var sa = evolve(bare)
    var comp = Circuit(r.circuit.num_qubits)
    for g in r.circuit.ops:
        if g.kind != Op.MEASURE:
            comp.append(g)
    var sb = evolve(comp)
    var re = 0.0
    var im = 0.0
    for i in range(1 << c.num_qubits):
        var j = 0
        for l in range(c.num_qubits):
            if (i >> l) & 1 == 1:
                j |= 1 << r.final_layout[l]
        re += sa.re[i] * sb.re[j] + sa.im[i] * sb.im[j]
        im += sa.re[i] * sb.im[j] - sa.im[i] * sb.re[j]
    return re * re + im * im


def test_uniform_is_untouched() raises:
    var u = parse_target("grid:3x3/sc_cz")
    assert_true(not u.is_calibrated(), "presets are uniform")
    assert_true(preset_pipeline(3, u).find("post_layout") < 0, "no noise pass on a uniform target")
    var c = quantum_volume(4, 3, 5)
    var wm = List[Int]()
    var r = transpile(c, u, 1, 7)
    var out = r.circuit.copy()
    assert_equal(post_layout(out, u, wm), 0.0, "nothing to gain")
    assert_equal(String(out), String(r.circuit), "uniform: identity stays")
    print("  uniform targets stay byte-identical")


def test_calibration_errors() raises:
    var t = Target.line(3)
    var raised = 0
    try:
        t.set_calibration([0.1], List[Float64](), List[Float64]())
    except e:
        if String(e).startswith("QM-E621"):
            raised += 1
    try:
        t.set_edge_error(0, 2, 0.1)
    except e:
        if String(e).startswith("QM-E621"):
            raised += 1
    assert_equal(raised, 2, "calibration size and edge errors")
    print("  calibration errors")


def main() raises:
    test_wrap_exact()
    test_periodic_gates()
    test_spec_and_presets()
    test_post_layout_known_embedding()
    test_post_layout_properties()
    test_uniform_is_untouched()
    test_calibration_errors()
    test_ai_presets_share_the_tail()
    test_compiler_refuses_correlated_error()
    print("test_wrap_layout OK")
