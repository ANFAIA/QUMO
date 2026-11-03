"""Regressions from the independent review of the Program stack.

Each test names its finding. Every one of them failed before its fix: run against
commit a55628b they raise, or pass a wrong value.
"""

from std.math import pi
from std.testing import assert_equal, assert_true, assert_raises

from qumo.exec.dispatch import choose_method_cpu
from qumo.ir.classical import CExpr, CN, CStore, ce_binary, ce_cast, ce_lnot, int_type
from qumo.ir.gates import Op
from qumo.ir.program import ProgramBuilder
from qumo.ir.qasm import load_any, parse_qasm, parse_qasm_program, to_qasm
from qumo.sim.program_runner import ProgramResult, sample_interpreted, sample_program
from qumo.sim.runner import SampleResult
from qumo.sim.stabilizer import quarter_turns, sample_stabilizer

comptime H = "OPENQASM 3.0; qubit[2] q; bit[8] c; "


def _run(body: String, shots: Int = 3) raises -> ProgramResult:
    return sample_program(parse_qasm_program(H + body), shots, 5)


def _int(body: String, name: String = "r") raises -> Int:
    return _run(body).int_output(name)[2]


def _real(body: String, name: String = "r") raises -> Float64:
    return _run(body).real_output(name)[2]


def _error_of(src: String) -> String:
    try:
        _ = sample_program(parse_qasm_program(src), 3, 5)
    except e:
        return String(e)
    return ""


def _err(body: String) -> String:
    return _error_of(H + body)


def _is(msg: String, code: String) raises:
    assert_true(msg.startswith(code), "expected " + code + ", got: " + msg)


def _reparsed(body: String, name: String = "r") raises -> Int:
    """The same output after a pass through to_qasm."""
    var text = to_qasm(parse_qasm_program(H + body))
    return sample_program(parse_qasm_program(text), 3, 5).int_output(name)[2]


def _same(a: ProgramResult, b: ProgramResult) raises:
    assert_equal(len(a.sample.memory), len(b.sample.memory))
    for s in range(len(a.sample.memory)):
        assert_equal(a.sample.memory[s], b.sample.memory[s])


def _tv(a: SampleResult, b: SampleResult) -> Float64:
    var total = 0.0
    for k in a.sorted_keys():
        total += abs(a.probability(k) - b.probability(k))
    for k in b.sorted_keys():
        if a.counts.get(k, 0) == 0:
            total += b.probability(k)
    return total / 2.0


def test_a1_gate_reads_count_as_reads() raises:
    var branch = String("qubit[2] q; bit[2] c; float x; h q[0]; c[0] = measure q[0]; if (c[0]) { x = pi; } rx(x) q[1]; c[1] = measure q[1];")
    _is(_error_of(branch), "QM-E581")
    var msg = _error_of("qubit[1] q; bit[1] c; float x; rz(x) q[0]; c[0] = measure q[0];")
    _is(msg, "QM-E581")
    assert_true(msg.find("__qumo") < 0, msg)
    assert_true(msg.find("'x'") > 0, msg)
    var viaany = String()
    try:
        _ = sample_program(load_any("qubit[1] q; bit[1] c; float x; rz(x) q[0]; c[0] = measure q[0];"), 2, 1)
    except e:
        viaany = String(e)
    _is(viaany, "QM-E581")


def test_a1_for_variable_is_set_on_the_true_edge() raises:
    # a loop that may run zero times leaves its variable unassigned
    var b = ProgramBuilder(1, 2)
    b.body.h(0)
    b.body.measure(0, 0)
    var i = b.declare("i", int_type(64))
    var r = b.declare("r", int_type(64), True)
    var stop = b.declare("n", int_type(64))
    b.assign(stop, ce_binary(CN.SUB, ce_cast(b.bit(0), int_type(64)), CExpr.literal_int(1)))
    b.begin_for(i, CExpr.literal_int(5), CExpr.literal_int(1), ce_binary(CN.ADD, b.read(stop), CExpr.literal_int(5)))
    b.end()
    b.assign(r, b.read(i))
    with assert_raises(contains="QM-E581"):
        _ = b.finish()


def test_a1_workers_with_angles_set_under_branches() raises:
    # x is unassigned on one path: it used to hold the last shot's value, so the
    # counts changed with the number of workers. Now it does not run at all.
    var stale = String("qubit[2] q; bit[2] c; float x; h q[0]; c[0] = measure q[0]; if (c[0]) { x = pi; } rx(x) q[1]; c[1] = measure q[1];")
    var refused = False
    try:
        var q = parse_qasm_program(stale)
        _same(sample_interpreted(q, 2000, 5, workers=1), sample_interpreted(q, 2000, 5, workers=8))
    except e:
        refused = String(e).startswith("QM-E581")
    assert_true(refused)
    var src = String("qubit[2] q; bit[2] c; float x; x = 0.0; h q[0]; c[0] = measure q[0]; if (c[0]) { x = pi; } rx(x) q[1]; x = x / 2; c[1] = measure q[1]; if (c[1]) { rz(x) q[1]; }")
    var p = parse_qasm_program(src)
    var a = sample_interpreted(p, 3000, 11, workers=1)
    _same(a, sample_interpreted(p, 3000, 11, workers=3))
    _same(a, sample_interpreted(p, 3000, 11, workers=8))
    # c[1] follows c[0] exactly: x = pi flips the qubit, x = 0 does not
    assert_equal(a.sample.counts.get("01", 0) + a.sample.counts.get("10", 0), 0)


def test_a1_nothing_survives_a_shot() raises:
    var base: List[Float64] = [0.5, 0.0]
    var st = CStore(2, 1, base.copy())
    st.ints[1] = 7
    st.reals[0] = 1.5
    st.params[1] = 3.0
    st.begin_shot(base)
    assert_equal(st.ints[1], 0)
    assert_true(st.reals[0] == 0.0 and st.params[1] == 0.0 and st.params[0] == 0.5)


def test_a2_literals_at_or_above_2_to_63() raises:
    assert_equal(_int("output bool r; uint[8] a = 255; a = a; r = a == 18446744073709551615;"), 0)
    assert_equal(_int("output bool r; uint[8] a = 1; a = a; r = a < 9223372036854775808;"), 1)
    assert_equal(_int("output bool r; uint[8] a = 1; a = a; r = a > 9223372036854775808;"), 0)
    assert_equal(_int("output bool r; uint[63] a = 0; a = a; r = a == 9223372036854775808;"), 0)
    assert_equal(_int("output bool r; bit[8] b = \"11111111\"; b = b; r = b == 18446744073709551615;"), 0)
    assert_equal(_int("output bool r; uint a = 18446744073709551615; a = a; r = a == 18446744073709551615;"), 1)


def test_a3_literal_division_truncates() raises:
    assert_true(_real("output float r; r = 1/2;") == 0.0)
    assert_true(_real("output float r; float x = 1/2; x = x + 0.0; r = x;") == 0.0)
    assert_true(_real("output float r; r = 7/2;") == 3.0)
    assert_equal(_int("output int r; r = (7 / 2) * 2;"), 6)
    assert_equal(_int("output int r; r = 1 / 2 + 1 / 2;"), 0)
    # constants are exact, so here folding and the run differ: rule R1 of the second review
    assert_true(_real("output float r; r = 2 ** -1;") == 0.5)
    _is(_err("output int r; int a = 2; a = a; int b = -1; b = b; r = a ** b;"), "QM-E481")


def test_a4_unrolled_and_region_loops_agree() raises:
    # assigning the loop variable changes the rest of the pass, not the iteration
    assert_equal(_int("output int r = 0; for int i in [0:1] { i = 5; r += i; }"), 10)
    assert_equal(_int("output int r = 0; int n = 1; n = n; for int i in [0:n] { i = 5; r += i; }"), 10)
    var both = _run("output int r = 0; output int o = 0; int i = 3; i = 3; for int i in [0:1] { i = 5; r += i; } o = i;")
    assert_equal(both.int_output("r")[0], 10)
    assert_equal(both.int_output("o")[0], 3)
    # a range the type cannot hold wraps the variable, in both forms
    assert_equal(_int("output int r = 0; for uint[2] i in [0:5] { r += int(i); }"), 7)
    assert_equal(_int("output int r = 0; for uint[2] i in [0:5] { if (r == 99) { break; } r += int(i); }"), 7)
    # the variable keeps its type when it unrolls
    assert_equal(_int("output int r = 0; for int i in [0:3] { r = r + i / 2; }"), 2)
    assert_equal(_int("output int r = 0; for int i in [0:3] { if (i == 99) { break; } r = r + i / 2; }"), 2)
    var c = parse_qasm("qubit[1] q; for int i in [1:3] { rx(i / 2) q[0]; }")
    assert_true(c.ops[0].p0.value() == 0.0 and c.ops[2].p0.value() == 1.0)
    # the body shares the variable's scope
    _is(_err("output int r = 0; for int i in [0:1] { int i = 7; r += i; }"), "QM-E522")
    _is(_err("output int r = 0; int n = 1; n = n; for int i in [0:n] { int i = 7; r += i; }"), "QM-E522")
    # a nested block may shadow it
    assert_equal(_int("output int r = 0; for int i in [0:1] { { int i = 7; i = i + 1; r += i; } }"), 16)
    assert_equal(_int("output int r = 0; int n = 1; n = n; for int i in [0:n] { { int i = 7; i = i + 1; r += i; } }"), 16)


def test_a5_negative_constant_under_power() raises:
    assert_equal(_reparsed("output int r; uint e = 3; e = e; r = (-2) ** e;"), -8)
    assert_equal(_reparsed("output int r; uint e = 2; e = e; r = (-2) ** e;"), 4)
    assert_equal(_reparsed("output int r; int x = 2; x = x; uint e = 2; e = e; r = (-x) ** e;"), 4)
    assert_equal(_reparsed("output int r; int x = 2; x = x; uint e = 2; e = e; r = -x ** e;"), -4)
    assert_equal(_reparsed("output int r; int x = 5; x = x; r = x - -3;"), 8)
    assert_equal(_reparsed("output int r; int x = 5; x = x; r = - -x;"), 5)
    var f = parse_qasm_program(H + "output float r; float e = 2.0; e = e; r = (-2.0) ** e;")
    var again = parse_qasm_program(to_qasm(f))
    assert_true(sample_program(again, 1, 1).real_output("r")[0] == 4.0)


def test_a6_bool_and_bit_are_int_in_arithmetic() raises:
    assert_equal(_int("output int r; r = true + 1;"), 2)
    assert_equal(_int("output int r; bool a = true; a = a; r = a + 1;"), 2)
    assert_equal(_int("output int r; bool a = true; a = a; bool b = true; b = b; r = a + b;"), 2)
    assert_equal(_int("output int r; c[0] = 1; c[1] = 1; r = c[0] + c[1];"), 2)
    assert_equal(_int("output int r; c[0] = 1; r = c[0] + 1;"), 2)
    assert_equal(_int("output int r; r = -true;"), -1)
    assert_equal(_int("output int r; bool a = true; a = a; r = -a;"), -1)
    assert_equal(_int("output int r; bit[4] a = \"1111\"; a = a; r = a + 1;"), 16)
    assert_equal(_int("output int r; bit[4] a = \"0000\"; a = a; r = a - 1;"), -1)
    assert_equal(_int("output int r; int[8] i = 127; i = i; bool a = true; a = a; r = i + a;"), 128)
    assert_equal(_int("output int r; r = true * 3 + false;"), 3)


def test_a7_exact_integers() raises:
    assert_equal(_int("output int r; int u = 9007199254740993; u = u; switch (u) { case 9007199254740993 { r = 1; } default { r = 0; } }"), 1)
    assert_equal(_int("output int r; int u = 9007199254740992; u = u; switch (u) { case 9007199254740993 { r = 1; } default { r = 0; } }"), 0)
    assert_equal(_int("output int r; uint u = 18446744073709551615; u = u; switch (u) { case 18446744073709551615 { r = 1; } default { r = 0; } }"), 1)
    _is(_err("output int r; uint u = 18446744073709551615; u = u; switch (u) { case -1 { r = 1; } default { r = 0; } }"), "QM-E580")
    assert_equal(_reparsed("output int r; uint u = 18446744073709551615; u = u; switch (u) { case 18446744073709551615 { r = 1; } default { r = 0; } }"), 1)
    assert_equal(_int("output int r = 0; for int i in {9007199254740993} { if (i == 9007199254740993) { r = 1; } }"), 1)
    assert_equal(_int("output int r = 0; for int i in {9007199254740993, 2} { if (i == 2) { break; } if (i == 9007199254740993) { r = 1; } }"), 1)
    var big = 18446744073709551616.0
    assert_true(parse_qasm("qubit[1] q; rz(18446744073709551616) q[0];").ops[0].p0.value() == big)
    assert_true(parse_qasm_program("qubit[1] q; rz(18446744073709551616) q[0];").body.ops[0].p0.value() == big)
    assert_true(parse_qasm("qubit[1] q; rz(100000000000000000000000) q[0];").ops[0].p0.value() == 1e23)


def test_a7_huge_angles_are_not_clifford() raises:
    # past 2^53 every float looks like a multiple of pi/2; auto must not pick the tableau
    var p = parse_qasm_program("qubit[1] q; bit[1] c; int k = 0; k = 1; rz(18446744073709551616) q[0]; c[0] = measure q[0];")
    assert_equal(sample_program(p, 5, 1).sample.method, "statevector:trajectories")


def test_flat_huge_angles_are_not_clifford() raises:
    # the ratio to pi/2 comes out whole for both, and neither is near a multiple of
    # pi/2: 2^60 is 0.78 rad away and the other, with one fractional bit left in the
    # ratio, 0.57 rad. On a dynamic circuit auto used to pick the tableau.
    var angles: List[Float64] = [1152921504606846976.0, 4493496820199122.0]
    for a in angles:
        var text = "qubit[1] q; bit[2] c; h q[0]; rz(" + String(Int(a)) + ") q[0]; h q[0]; reset q[0]; x q[0]; h q[0]; rz(" + String(Int(a)) + ") q[0]; h q[0]; c[0] = measure q[0]; c[1] = measure q[0];"
        var f = parse_qasm(text)
        assert_true(not f.is_clifford())
        assert_equal(choose_method_cpu(f, 100), "statevector")
        with assert_raises(contains="QM-E701"):
            _ = quarter_turns(a)
        with assert_raises(contains="QM-E701"):
            _ = sample_stabilizer(f, 10, 1)
    # the limit is on the ratio: a whole number of quarter turns below it stays Clifford
    assert_equal(quarter_turns(1000001.0 * pi / 2), 1)
    assert_true(parse_qasm("qubit[1] q; rz(1500.0 * pi) q[0];").is_clifford())
    assert_true(not parse_qasm("qubit[1] q; rz(1000000000.0 * pi) q[0];").is_clifford())


def test_a7_typed_constants_in_integer_positions() raises:
    # found by the oracle: i % 3 computed in floats gave 2 for 2^63 - 1, not 1
    assert_equal(_int("output int r = 0; for int i in [9223372036854775807:9223372036854775807] { for int j in [0:(i % 3)] { r += 1; } }"), 2)
    assert_equal(_int("const int n = 9223372036854775807; output int r = 0; for int j in [1:n % 3] { r += 1; }"), 1)
    assert_equal(parse_qasm("const int n = 7; qubit[4] q; h q[n / 2];").ops[0].q0, 3)
    assert_equal(parse_qasm("qubit[4] q; for int i in [3:3] { h q[i / 2]; }").ops[0].q0, 1)
    assert_equal(parse_qasm("qubit[4] q; h q[3/2];").ops[0].q0, 1)  # the classical table, since the third review
    _is(_err("h q[1.5];"), "QM-E530")


def test_a8_for_stepping_at_the_int64_ends() raises:
    assert_equal(_int("output int r = 0; int n = 9223372036854775807; n = n; for int i in [9223372036854775806:n] { r += 1; if (r > 5) { break; } }"), 2)
    assert_equal(_int("output int r = 0; int n = -9223372036854775808; n = n; for int i in [-9223372036854775807:-1:n] { r += 1; if (r > 5) { break; } }"), 2)
    assert_equal(_int("output int r = 0; int n = 9223372036854775807; n = n; for int i in [9223372036854775800:100:n] { r += 1; if (r > 5) { break; } }"), 1)
    assert_equal(_int("output int r = 0; for int i in [9223372036854775806:9223372036854775807] { r += 1; }"), 2)
    assert_equal(_int("output int r = 0; for int i in [-9223372036854775807:-1:-9223372036854775808] { r += 1; }"), 2)


def test_a9_typed_constants_in_gate_arguments() raises:
    var flat = parse_qasm("const int a = 7; qubit[1] q; rz(a / 2) q[0];")
    var prog = parse_qasm_program("const int a = 7; qubit[1] q; bit[1] c; rz(a / 2) q[0]; while (false) { }")
    assert_true(flat.ops[0].p0.value() == 3.0)
    assert_true(prog.body.ops[0].p0.value() == 3.0)
    # a literal-only argument keeps the constant path's meaning in both parsers
    var l1 = parse_qasm("qubit[1] q; rz(1/2) q[0]; rz(-7 % 2) q[0];")
    var l2 = parse_qasm_program("qubit[1] q; rz(1/2) q[0]; rz(-7 % 2) q[0]; while (false) { }")
    for i in range(2):
        assert_true(l1.ops[i].p0.value() == l2.body.ops[i].p0.value())
    assert_true(l1.ops[0].p0.value() == 0.5 and l1.ops[1].p0.value() == 1.0)
    # an angle constant wraps as a variable does: the controlled phase shows it
    var head = String("OPENQASM 3.0; qubit[2] q; bit[1] c; h q[0]; h q[1]; ")
    var tail = String(" x q[1]; ctrl @ rz(-a) q[0], q[1]; h q[0]; c[0] = measure q[0];")
    var const_form = sample_program(parse_qasm_program(head + "const angle[8] a = pi/2;" + tail), 8000, 3)
    var run_form = sample_program(parse_qasm_program(head + "angle[8] a = pi/2; a = a;" + tail), 8000, 3)
    assert_true(_tv(const_form.sample, run_form.sample) < 0.05)
    # inputs stay symbolic next to a typed constant
    var s = parse_qasm("input float t; const int n = 3; qubit[1] q; rz(2 * t + n / 2) q[0];")
    assert_true(s.ops[0].p0.is_symbolic() and s.ops[0].p0.offset == 1.0 and s.ops[0].p0.coeff == 2.0)


def test_c2_float_of_bits_reparses() raises:
    for body in ["output float r; c[0] = 1; c[1] = 1; r = c[0:1] * 1.5;", "output float r; c[0] = 1; r = c[0] * 1.5;", "output float r; c[0] = 1; float y = 0.5; y = y; r = y + c[0];"]:
        var a = parse_qasm_program(H + String(body))
        var b = parse_qasm_program(to_qasm(a))
        assert_true(sample_program(a, 1, 1).real_output("r")[0] == sample_program(b, 1, 1).real_output("r")[0])


def test_c3_non_finite_constants_have_no_literal() raises:
    var inf_ = parse_qasm_program(H + "output float r; float x = 1e308 * 10.0; x = x; r = x;")
    with assert_raises(contains="QM-E552"):
        _ = to_qasm(inf_)
    var nan_ = parse_qasm_program(H + "output float r; float x = sqrt(-1.0); x = x; r = x;")
    with assert_raises(contains="QM-E552"):
        _ = to_qasm(nan_)


def test_d1_nan_is_checked_at_the_gate() raises:
    # stored but never used by a gate that runs: no error
    var r = _run("float x = -1.0; x = x; float n; n = sqrt(x); if (c[0]) { rx(n) q[0]; } c[1] = measure q[0];")
    assert_equal(r.sample.memory[0], 0)
    # read by a gate that runs: QM-E482 with the shot
    _is(_err("float x = -1.0; x = x; float n; n = sqrt(x); x q[0]; c[0] = measure q[0]; if (c[0]) { rx(n) q[0]; }"), "QM-E482")
    _is(_err("float n; n = sqrt(-1.0 + 0.0); n = n + 0.0; rx(n) q[0];"), "QM-E482")


def test_d1_angles_that_overflow_at_the_gate() raises:
    # found by the oracle: coeff * x overflowed with a finite x, and the state went NaN
    _is(_err("float t = -1.0; t = t * 10.0; rz(t * 1e308 * 1e308) q[0];"), "QM-E482")
    _is(_err("float t = 2.0; t = t * 1e308; int k = 0; k = 1; rz(t * 1e300) q[0];"), "QM-E482")
    _is(_err("int k = 0; k = 1; rz(1e308 * 10) q[0];"), "QM-E482")


def test_d2_valid_shadowing() raises:
    assert_true(_err("if (c[0]) { bit b; } else { bit b; }") == "")
    assert_equal(_int("output int r = 0; int x = 1; if (!c[0]) { int x = 2; r = x; }"), 2)
    assert_true(_real("output float r = 1.0; if (!c[0]) { float c0 = 2.0; r = c0; }") == 2.0)
    assert_true(sample_program(parse_qasm_program("input float t; qubit[1] q; bit[1] c; output float r = 0.0; if (!c[0]) { float t = 2.0; r = t; }"), 2, 1, [0.5]).real_output("r")[0] == 2.0)
    assert_equal(_int("output int r = 0; const int x = 1; if (!c[0]) { int x = 2; x = x + 1; r = x; } r = r * 10 + x;"), 31)
    assert_equal(_int("output int r = 0; bit[2] d; d[1] = 1; if (!c[0]) { int d = 7; d = d + 1; r = d; } r = r * 10 + int(d);"), 82)
    # a name still means one thing in its own scope
    _is(_err("int x = 1; x = 2; int x = 3;"), "QM-E522")
    _is(_err("if (true) { int x = 1; x = 2; int x = 3; }"), "QM-E522")


def test_d3_prescan_sees_an_assignment_after_a_range() raises:
    assert_equal(_int("int x = 0; for int i in [0:2] x += 1; output int r = x;"), 3)
    assert_equal(_int("int x = 0; for int i in {0, 1} x += 1; output int r = x;"), 2)
    assert_equal(_int("int x = 0; int[8] y = 1; output int r = y + x;"), 1)


def test_d4_conservative_error_stays() raises:
    _is(_err("output int r; int x; while (true) { x = 1; break; } r = x;"), "QM-E581")


def test_d5_range_values_must_be_integers() raises:
    var step = _err("output int r = 0; for int i in [0:0.5:3] { r += 1; }")
    _is(step, "QM-E570")
    assert_true(step.find("step") > 0 and step.find("integer") > 0, step)
    var region = _err("output int r = 0; for int i in [0:0.5:3] { if (i == 9) { break; } r += 1; }")
    _is(region, "QM-E570")
    assert_true(region.find("integer") > 0, region)
    var bound = _err("output int r = 0; for int i in [0:2.5] { r += 1; }")
    _is(bound, "QM-E570")
    assert_true(bound.find("integer") > 0, bound)


def test_d6_constant_zero_denominators() raises:
    _is(_err("output int r; int x = 1; x = x; r = x / 0;"), "QM-E533")
    _is(_err("output int r; int x = 1; x = x; r = x % 0;"), "QM-E533")
    _is(_err("output float r; float x = 1.0; x = x; r = x / 0.0;"), "QM-E533")
    _is(_err("output float r; float x = 1.0; x = x; r = mod(x, 0.0);"), "QM-E481")
    _is(_err("output float r; r = mod(1.0, 0.0);"), "QM-E533")
    _is(_err("output bool r; r = false && (1 / 0 > 1);"), "QM-E533")
    _is(_err("output float r; float x = 1.0; x = x; float y = 0.0; y = y; r = mod(x, y);"), "QM-E481")


def test_e1_bit_in_a_loop_body() raises:
    assert_equal(_int("output int r = 0; int k = 0; while (k < 3) { k += 1; bit b; if (k == 1) { b = 1; } if (b == 1) { r += 1; } }"), 1)
    assert_equal(_int("output int r = 0; for int k in [0:2] { bit b; if (k == 0) { b = 1; } if (b == 1) { r += 1; } }"), 1)
    assert_equal(_int("output int r = 0; for int k in [0:2] { if (r > 99) { break; } bit b; if (k == 0) { b = 1; } if (b == 1) { r += 1; } }"), 1)
    assert_true(_err("int k = 0; k = 0; while (k < 1) { k += 1; bit b; b = 1; } b = 1;").byte_length() > 0)
    assert_true(_err("if (c[0]) { bit b; } else { bit[2] b; b[1] = 1; }") == "")
    # unrolled and region forms keep one classical bit per declaration, even when
    # the loop never runs (the oracle found that one)
    assert_equal(_run("for int k in [0:2] { bit b; b = 1; }").sample.num_clbits, 9)
    assert_equal(_run("for int k in [1:0] { bit b; b = 1; }").sample.num_clbits, 9)
    assert_equal(_run("for int k in [1:0] { if (c[0]) { break; } bit b; b = 1; }").sample.num_clbits, 9)
    assert_equal(sample_program(load_any(H + "for int k in [1:0] { bit b; b = 1; }"), 2, 1).sample.num_clbits, 9)


def test_e2_literal_wider_than_its_operand() raises:
    _is(_err("output int r; int[8] a = 100; a = a; r = a + 1000;"), "QM-E580")
    _is(_err("output uint r; uint[8] a = 255; a = a; r = a + 300;"), "QM-E580")
    _is(_err("output uint r; uint[8] a = 255; a = a; r = a & 300;"), "QM-E580")
    assert_true(_err("output uint r; uint[8] a = 255; a = a; r = a + 300;").find("cast") > 0)
    # fitting literals keep the operand's type, and wrap with it
    assert_equal(_int("output int r; uint[8] a = 255; a = a; r = int(a + 1);"), 0)
    # shift counts and comparisons take any literal
    assert_equal(_int("output int r; uint[8] a = 1; a = a; r = int(a << 300);"), 0)
    assert_equal(_int("output bool r; int[8] a = 100; a = a; r = a < 1000;"), 1)
    assert_equal(_int("output bool r; uint[8] a = 100; a = a; r = a != 300;"), 1)


def test_e3_flat_and_program_declarations_agree() raises:
    var rows: List[String] = [
        "int[8] x = 300; rx(x) q[0];",
        "uint[8] x = -1; rx(x) q[0];",
        "angle[8] x = 0.1; rx(x) q[0];",
        "const int x = -7 % 3; rx(x) q[0];",
        "const int a = 7; const float x = a / 2; rx(x) q[0];",
        "int x = 7 / 2; rx(x) q[0];",
        "float x = 7 / 2; rx(x) q[0];",
        "const int x = 1.9; rx(x) q[0];",
        "const int x = -1.9; rx(x) q[0];",
        "const bool x = 2; rx(x) q[0];",
        "float x = pi / 3; rx(x) q[0]; float y = x * 2; rx(y) q[0];",
    ]
    var want: List[Float64] = [44.0, 255.0, 0.0981747704246810, -1.0, 3.0, 3.0, 3.0, 1.0, -1.0, 1.0, pi / 3]
    for i in range(len(rows)):
        var src = H + rows[i]
        var a = parse_qasm(src).ops[0].p0.value()
        var b = parse_qasm_program(src + " while (false) { }").body.ops[0].p0.value()
        assert_true(abs(a - want[i]) < 1e-12, rows[i] + " flat " + String(a))
        assert_true(a == b, rows[i] + " program " + String(b))
    # the type decides the qubit, in both parsers
    var f = parse_qasm(H + "uint[1] i = 3; h q[i];")
    var p = parse_qasm_program(H + "uint[1] i = 3; h q[i]; while (false) { }")
    assert_equal(f.ops[0].q0, 1)
    assert_equal(p.body.ops[0].q0, 1)


def test_e4_float_to_angle_ties_go_to_even() raises:
    assert_equal(_int("output angle[2] r; float f = pi/4; f = f; r = angle[2](f);"), 0)
    assert_equal(_int("output angle[2] r; float f = 3*pi/4; f = f; r = angle[2](f);"), 2)
    assert_equal(_int("output angle[3] r; float f = 3*pi/8; f = f; r = angle[3](f);"), 2)
    assert_equal(_int("output angle[3] r; float f = 5*pi/8; f = f; r = angle[3](f);"), 2)


def test_e5_refusals() raises:
    _is(_err("output uint[16] r; int[8] x = -1; x = x; r = uint[16](bit[16](x));"), "QM-E580")
    _is(_err("bit[4] a = \"1111\"; a = a; bit[8] b; b = bit[8](a);"), "QM-E580")
    assert_equal(_int("output uint[16] r; int[8] x = -1; x = x; r = uint[16](bit[8](x));"), 255)
    _is(_err("output int r = 0; float f = 1.9; f = f; switch (f) { case 1 { r = 1; } default { r = 2; } }"), "QM-E580")
    _is(_err("int i = 0; i = 0; while (i < 1) { qubit r; i += 1; }"), "QM-E504")


def test_open_box_as_a_loop_body() raises:
    assert_equal(_int("output int r = 0; for int i in [0:2] box { x q[0]; } r = 1;"), 1)
    assert_equal(_run("for int i in [0:2] box { x q[0]; } c[0] = measure q[0];", 5).sample.memory[0], 1)
    assert_equal(_run("for int i in [3:2] box { x q[0]; } c[0] = measure q[0];", 5).sample.memory[0], 0)
    var c = parse_qasm("qubit[1] q; for int i in [3:2] box { x q[0]; } h q[0];")
    assert_equal(c.count(Op.H), 1)
    assert_equal(c.count(Op.X), 0)


def test_open_input_angle_is_radians_in_both_parsers() raises:
    var src = String("input angle[8] t; qubit[1] q; rz(t) q[0];")
    var a = parse_qasm(src)
    var b = parse_qasm_program(src + " while (false) { }")
    assert_true(a.ops[0].p0.is_symbolic() and b.body.ops[0].p0.is_symbolic())
    assert_equal(b.num_params(), 1)


def main() raises:
    test_a1_gate_reads_count_as_reads()
    test_a1_for_variable_is_set_on_the_true_edge()
    test_a1_workers_with_angles_set_under_branches()
    test_a1_nothing_survives_a_shot()
    test_a2_literals_at_or_above_2_to_63()
    test_a3_literal_division_truncates()
    test_a4_unrolled_and_region_loops_agree()
    test_a5_negative_constant_under_power()
    test_a6_bool_and_bit_are_int_in_arithmetic()
    test_a7_exact_integers()
    test_a7_huge_angles_are_not_clifford()
    test_flat_huge_angles_are_not_clifford()
    test_a7_typed_constants_in_integer_positions()
    test_a8_for_stepping_at_the_int64_ends()
    test_a9_typed_constants_in_gate_arguments()
    test_c2_float_of_bits_reparses()
    test_c3_non_finite_constants_have_no_literal()
    test_d1_nan_is_checked_at_the_gate()
    test_d1_angles_that_overflow_at_the_gate()
    test_d2_valid_shadowing()
    test_d3_prescan_sees_an_assignment_after_a_range()
    test_d4_conservative_error_stays()
    test_d5_range_values_must_be_integers()
    test_d6_constant_zero_denominators()
    test_e1_bit_in_a_loop_body()
    test_e2_literal_wider_than_its_operand()
    test_e3_flat_and_program_declarations_agree()
    test_e4_float_to_angle_ties_go_to_even()
    test_e5_refusals()
    test_open_box_as_a_loop_body()
    test_open_input_angle_is_radians_in_both_parsers()
    print("test_program_review OK")
