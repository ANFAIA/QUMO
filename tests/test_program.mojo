"""Program IR, classical expressions and the program interpreter."""

from std.math import pi, sqrt, sin, cos, log
from std.testing import assert_equal, assert_true, assert_raises, assert_almost_equal

from qumo.ir.circuit import Circuit
from qumo.ir.qasm import parse_qasm
from qumo.ir.classical import (
    CExpr, CN, CF, CT, CStore, CEval, NoBits, eval_nodes, int_type, uint_type, bit_type, bool_type, float_type,
    angle_type, ce_binary, ce_cast, ce_bool, ce_lnot, ce_bnot, ce_neg, ce_logical, ce_function, wrap_int,
    angle_radians, can_cast, type_name,
)
from qumo.ir.program import Program, ProgramBuilder, SK, TERM, cfg_of, check_program
from qumo.sim.runner import sample
from qumo.sim.program_runner import sample_program, sample_interpreted


def _lit(v: Int) -> CExpr:
    return CExpr.literal_int(v)


def _ti(e: CExpr) raises -> Int:
    """Value of a constant-folded integer expression."""
    assert_true(e.is_const(), "expected a folded constant")
    return e.const_int()


def _run_one(mut b: ProgramBuilder, name: String) raises -> Int:
    var p = b.finish()
    var r = sample_program(p, 1, 1)
    return r.int_output(name)[0]


def _run_real(mut b: ProgramBuilder, name: String) raises -> Float64:
    var p = b.finish()
    var r = sample_program(p, 1, 1)
    return r.real_output(name)[0]


def test_wrapping() raises:
    assert_equal(wrap_int(16, uint_type(4)), 0)
    assert_equal(wrap_int(-1, uint_type(4)), 15)
    assert_equal(wrap_int(7, int_type(3)), -1)
    assert_equal(wrap_int(4, int_type(3)), -4)
    assert_equal(wrap_int(-5, int_type(3)), 3)
    assert_equal(wrap_int(-1, int_type(64)), -1)
    # runtime wrap: a uint[4] variable counts modulo 16
    var b = ProgramBuilder(0)
    var k = b.declare("k", uint_type(4), True)
    b.assign(k, _lit(15))
    b.assign(k, ce_binary(CN.ADD, b.read(k), _lit(1)))
    assert_equal(_run_one(b, "k"), 0)
    # int[3]: 3 + 1 wraps to -4
    var b2 = ProgramBuilder(0)
    var i = b2.declare("i", int_type(3), True)
    b2.assign(i, _lit(3))
    b2.assign(i, ce_binary(CN.ADD, b2.read(i), _lit(1)))
    assert_equal(_run_one(b2, "i"), -4)
    # int[64] and uint[64] wrap at 2^64
    var b3 = ProgramBuilder(0)
    var u = b3.declare("u", uint_type(64), True)
    b3.assign(u, _lit(0))
    b3.assign(u, ce_binary(CN.SUB, b3.read(u), _lit(1)))
    assert_equal(_run_one(b3, "u"), -1)  # the bits of 2^64 - 1


def test_division() raises:
    var i32 = int_type(32)
    var seven = CExpr.typed_const(i32, 7)
    var minus = CExpr.typed_const(i32, -7)
    assert_equal(_ti(ce_binary(CN.DIV, seven, _lit(2))), 3)
    assert_equal(_ti(ce_binary(CN.DIV, minus, _lit(2))), -3)
    assert_equal(_ti(ce_binary(CN.DIV, seven, _lit(-2))), -3)
    assert_equal(_ti(ce_binary(CN.MOD, minus, _lit(3))), -1)
    assert_equal(_ti(ce_binary(CN.MOD, seven, _lit(-3))), 1)
    # two literals follow the same rule: folding equals what the run would compute
    assert_equal(_ti(ce_binary(CN.DIV, _lit(1), _lit(2))), 0)
    assert_equal(_ti(ce_binary(CN.DIV, _lit(6), _lit(2))), 3)
    assert_equal(_ti(ce_binary(CN.MOD, _lit(-7), _lit(3))), -1)
    with assert_raises(contains="QM-E533"):
        _ = ce_binary(CN.DIV, _lit(1), _lit(0))
    with assert_raises(contains="QM-E533"):
        _ = ce_binary(CN.MOD, _lit(1), _lit(0))
    # INT64_MIN / -1 wraps instead of trapping
    var minv = -9223372036854775807 - 1
    assert_equal(_ti(ce_binary(CN.DIV, CExpr.typed_const(int_type(64), minv), CExpr.typed_const(int_type(64), -1))), minv)
    # uint[64] above 2^63 divides as unsigned
    assert_equal(_ti(ce_binary(CN.DIV, CExpr.typed_const(uint_type(64), -2), CExpr.typed_const(uint_type(64), 2))), (1 << 63) - 1)
    # a run-time zero reports QM-E481 with the shot
    var b = ProgramBuilder(0)
    var x = b.declare("x", int_type(32), True)
    var y = b.declare("y", int_type(32))
    b.assign(y, _lit(0))
    b.assign(x, ce_binary(CN.DIV, _lit(1), b.read(y)))
    var p = b.finish()
    with assert_raises(contains="QM-E481"):
        _ = sample_program(p, 1, 1)


def test_shifts_and_bits() raises:
    var w = uint_type(8)
    assert_equal(_ti(ce_binary(CN.SHL, CExpr.typed_const(w, 0x81), _lit(1))), 2)
    assert_equal(_ti(ce_binary(CN.SHR, CExpr.typed_const(w, 0x81), _lit(1))), 0x40)
    assert_equal(_ti(ce_binary(CN.SHL, CExpr.typed_const(w, 1), _lit(8))), 0)
    assert_equal(_ti(ce_binary(CN.SHR, CExpr.typed_const(w, 0xFF), _lit(9))), 0)
    assert_equal(_ti(ce_binary(CN.SHL, CExpr.typed_const(uint_type(64), 1), _lit(64))), 0)
    assert_equal(_ti(ce_binary(CN.SHR, CExpr.typed_const(uint_type(64), -1), _lit(63))), 1)
    with assert_raises(contains="QM-E580"):
        _ = ce_binary(CN.SHL, CExpr.typed_const(int_type(8), 1), _lit(1))
    assert_equal(_ti(ce_bnot(CExpr.typed_const(w, 0x0F))), 0xF0)
    assert_equal(_ti(ce_binary(CN.XOR, CExpr.typed_const(w, 0xF0), CExpr.typed_const(w, 0xFF))), 0x0F)
    with assert_raises(contains="QM-E580"):
        _ = ce_binary(CN.AND, CExpr.typed_const(int_type(8), 1), CExpr.typed_const(int_type(8), 1))


def test_mixing_rules() raises:
    var a = CExpr.variable(0, int_type(8))
    var u = CExpr.variable(1, uint_type(8))
    with assert_raises(contains="QM-E580"):
        _ = ce_binary(CN.ADD, a, u)
    with assert_raises(contains="QM-E580"):
        _ = ce_binary(CN.LT, a, u)
    # literals adapt to the other side
    assert_equal(ce_binary(CN.ADD, u, _lit(3)).ty().tag, CT.UINT)
    assert_equal(ce_binary(CN.ADD, a, _lit(-3)).ty().tag, CT.INT)
    # a negative literal never adapts to uint
    with assert_raises(contains="QM-E580"):
        _ = ce_binary(CN.ADD, u, _lit(-3))
    # a literal too wide for the operand widens it: uint[4] k < 100 stays exact
    var k = CExpr.variable(0, uint_type(4))
    var cmp = ce_binary(CN.LT, k, _lit(100))
    assert_equal(cmp.nodes[0].ty.width, 4)
    assert_equal(cmp.nodes[1].ty.width, 7)
    # bit registers read as unsigned
    assert_equal(ce_binary(CN.EQ, CExpr.bits(0, 3), _lit(5)).ty().tag, CT.BOOL)


def test_casts() raises:
    assert_true(can_cast(float_type(), angle_type(8)))
    assert_true(not can_cast(angle_type(8), float_type()))
    assert_true(not can_cast(angle_type(8), int_type(8)))
    assert_true(not can_cast(angle_type(8), uint_type(8)))
    assert_true(can_cast(angle_type(8), bool_type()))
    assert_true(can_cast(angle_type(8), bit_type(8)))
    assert_true(not can_cast(angle_type(8), bit_type(4)))
    assert_true(not can_cast(int_type(8), angle_type(8)))
    assert_true(not can_cast(float_type(), bit_type(1)))
    assert_true(not can_cast(bit_type(1), float_type()))
    with assert_raises(contains="QM-E580"):
        _ = ce_cast(CExpr.variable(0, angle_type(8)), float_type())
    # float -> int truncates toward zero
    assert_equal(_ti(ce_cast(CExpr.literal_float(2.9), int_type(8))), 2)
    assert_equal(_ti(ce_cast(CExpr.literal_float(-2.9), int_type(8))), -2)
    with assert_raises(contains="QM-E482"):
        _ = ce_cast(CExpr.literal_float(128.0), int_type(8))
    with assert_raises(contains="QM-E482"):
        _ = ce_cast(CExpr.literal_float(1e300), int_type(64))
    with assert_raises(contains="QM-E482"):
        _ = ce_cast(CExpr.literal_float(-1.0), uint_type(8))
    assert_equal(_ti(ce_cast(CExpr.literal_float(255.5), uint_type(8))), 255)
    # at run time as well
    var b = ProgramBuilder(0)
    var f = b.declare("f", float_type())
    var x = b.declare("x", int_type(8), True)
    b.assign(f, CExpr.literal_float(1000.0))
    b.assign(x, b.read(f))
    var p = b.finish()
    with assert_raises(contains="QM-E482"):
        _ = sample_program(p, 1, 1)
    # float -> angle[n]: nearest step
    assert_equal(_ti(ce_cast(CExpr.literal_float(pi), angle_type(8))), 128)
    assert_equal(_ti(ce_cast(CExpr.literal_float(-pi / 2), angle_type(8))), 192)
    assert_almost_equal(angle_radians(128, 8), pi)
    assert_equal(type_name(angle_type(8)), "angle[8]")
    assert_equal(type_name(int_type(64, True)), "int")
    assert_equal(type_name(bit_type(3)), "bit[3]")


def test_short_circuit() raises:
    # (y != 0) && (1 / y > 0): the division must not run
    var b = ProgramBuilder(0)
    var y = b.declare("y", int_type(32))
    var r = b.declare("r", bool_type(), True)
    b.assign(y, _lit(0))
    var e = ce_logical(True, ce_binary(CN.NE, b.read(y), _lit(0)), ce_binary(CN.GT, ce_binary(CN.DIV, _lit(1), b.read(y)), _lit(0)))
    b.assign(r, e)
    assert_equal(_run_one(b, "r"), 0)
    var c = ProgramBuilder(0)
    var y2 = c.declare("y", int_type(32))
    var r2 = c.declare("r", bool_type(), True)
    c.assign(y2, _lit(0))
    var e2 = ce_logical(False, ce_binary(CN.EQ, c.read(y2), _lit(0)), ce_binary(CN.GT, ce_binary(CN.DIV, _lit(1), c.read(y2)), _lit(0)))
    c.assign(r2, e2)
    assert_equal(_run_one(c, "r"), 1)


def test_functions_and_power() raises:
    assert_equal(_ti(ce_binary(CN.POW, _lit(3), _lit(4))), 81)
    assert_almost_equal(ce_function(CF.SQRT, _lit(2)).const_float(), sqrt(2.0))
    assert_equal(_ti(ce_function(CF.POPCOUNT, CExpr.typed_const(uint_type(8), 0x13))), 3)
    assert_equal(_ti(ce_function(CF.ABS, CExpr.typed_const(int_type(8), -3))), 3)
    with assert_raises(contains="QM-E533"):
        _ = ce_binary(CN.POW, _lit(2), _lit(-1))
    with assert_raises(contains="QM-E533"):
        _ = ce_binary(CN.POW, CExpr.typed_const(int_type(32), 2), _lit(-1))


def test_float_and_angle_vars() raises:
    var b = ProgramBuilder(0)
    var f = b.declare("f", float_type(), True)
    b.assign(f, CExpr.literal_float(0.25))
    b.assign(f, ce_binary(CN.MUL, b.read(f), _lit(3)))
    assert_almost_equal(_run_real(b, "f"), 0.75)
    var c = ProgramBuilder(0)
    var a = c.declare("a", angle_type(4), True)
    c.assign(a, CExpr.literal_float(3.0 * pi / 2))  # 12 of 16 steps
    c.assign(a, ce_binary(CN.ADD, c.read(a), c.read(a)))  # 24 mod 16
    assert_equal(_run_one(c, "a"), 8)
    with assert_raises(contains="QM-E580"):
        _ = ce_binary(CN.ADD, CExpr.variable(0, angle_type(4)), CExpr.variable(1, angle_type(8)))
    with assert_raises(contains="QM-E580"):
        _ = ce_binary(CN.ADD, CExpr.variable(0, angle_type(4)), CExpr.variable(1, float_type()))


def test_definite_assignment() raises:
    var b = ProgramBuilder(1, 1)
    var x = b.declare("x", int_type(8))
    var y = b.declare("y", int_type(8), True)
    b.begin_if(b.bit(0))
    b.assign(x, _lit(1))
    b.end()
    b.assign(y, b.read(x))
    with assert_raises(contains="QM-E581"):
        _ = b.finish()
    # both branches assign: fine
    var c = ProgramBuilder(1, 1)
    var x2 = c.declare("x", int_type(8))
    var y2 = c.declare("y", int_type(8), True)
    c.begin_if(c.bit(0))
    c.assign(x2, _lit(1))
    c.else_()
    c.assign(x2, _lit(2))
    c.end()
    c.assign(y2, c.read(x2))
    assert_equal(_run_one(c, "y"), 2)
    # assigned only in a while body: the loop may not run
    var d = ProgramBuilder(1, 1)
    var x3 = d.declare("x", int_type(8))
    var y3 = d.declare("y", int_type(8), True)
    d.begin_while(d.bit(0))
    d.assign(x3, _lit(1))
    d.end()
    d.assign(y3, d.read(x3))
    with assert_raises(contains="QM-E581"):
        _ = d.finish()
    # an output left unassigned on some path
    var e = ProgramBuilder(1, 1)
    var y4 = e.declare("y", int_type(8), True)
    e.begin_if(e.bit(0))
    e.assign(y4, _lit(1))
    e.end()
    with assert_raises(contains="QM-E581"):
        _ = e.finish()


def test_structure_errors() raises:
    var b = ProgramBuilder(1, 1)
    with assert_raises(contains="QM-E583"):
        b.break_()
    with assert_raises(contains="QM-E583"):
        b.else_()
    with assert_raises(contains="QM-E583"):
        b.end()
    var c = ProgramBuilder(1, 1)
    c.begin_while(c.bit(0))
    with assert_raises(contains="QM-E583"):
        _ = c.finish()
    # a detector in a program with a runtime region
    var d = ProgramBuilder(1, 1)
    d.begin_if(d.bit(0))
    d.end()
    d.body.measure(0, 0)
    var recs: List[Int] = [0]
    d.body.detector(recs)
    with assert_raises(contains="QM-E586"):
        _ = d.finish()
    # an opcode a block does not know
    var cc = Circuit(1, 1)
    cc.h(0)
    cc.measure(0, 0)
    cc.ops[0].kind = 200
    var p = Program.from_circuit(cc)
    with assert_raises(contains="QM-E484"):
        _ = check_program(p)


def test_cfg_shape() raises:
    var b = ProgramBuilder(1, 1)
    b.body.h(0)
    b.begin_while(ce_lnot(b.bit(0)))
    b.body.measure(0, 0)
    b.begin_if(b.bit(0))
    b.break_()
    b.end()
    b.continue_()
    b.end()
    var p = b.finish()
    var cfg = cfg_of(p)
    var br = 0
    var back = 0
    var halts = 0
    for bb in cfg.blocks:
        if bb.term == TERM.BR:
            br += 1
        if bb.back:
            back += 1
        if bb.term == TERM.HALT:
            halts += 1
    assert_equal(br, 2)
    assert_equal(back, 2)  # continue + the end of the body
    assert_equal(halts, 1)


def _circuits() raises -> List[String]:
    var s = List[String]()
    s.append("qubit[3] q; bit[3] c; h q[0]; cx q[0], q[1]; c[0] = measure q[0]; if (c[0] == 1) x q[1]; h q[2]; t q[2]; cx q[2], q[1]; c[1] = measure q[1]; reset q[1]; rx(0.7) q[1]; c[2] = measure q[1];")
    s.append("qubit[2] q; bit[2] c; ry(0.4) q[0]; #pragma qumo depolarize1(0.05) q[0];\n#pragma qumo x_error(0.1) q[1];\n#pragma qumo depolarize2(0.03) q[0], q[1];\nc[0] = measure q[0]; ry(1.1) q[1]; c[1] = measure q[1];")
    s.append("qubit[2] q; bit[2] c; h q[0]; #pragma qumo pauli_channel_2(0.1, 0.02, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1) q[0], q[1];\n#pragma qumo correlated_error(0.2) x q[0], z q[1];\n#pragma qumo else_correlated_error(0.3) y q[0];\n#pragma qumo kraus1 \"ad\" (1, 0, 0, 0, 0, 0, 0.9, 0, 0, 0, 0.4358898943540674, 0, 0, 0, 0, 0) q[1];\nt q[0]; c = measure q;")
    s.append("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; creg c[2]; h q[0]; h q[1]; measure q[0] -> c[0]; measure q[1] -> c[1]; if(c==3) x q[2]; measure q[2] -> c[0];")
    s.append("qubit[3] q; bit[3] c; h q; cx q[0], q[1]; c = measure q;")  # static
    return s^


def test_from_circuit_matches_sample() raises:
    """Exit test: identical counts, through `sample_program` and through the
    interpreter itself (same RNG draws in the same order)."""
    var src = _circuits()
    for i in range(len(src)):
        var c = parse_qasm(src[i])
        var p = Program.from_circuit(c)
        var a = sample(c, 3000, 77)
        var b = sample_program(p, 3000, 77)
        assert_equal(String(a), String(b.sample))
        assert_equal(a.method, b.sample.method)
        assert_true(p.try_flatten())
        if c.is_dynamic() or c.has_noise():
            var d = sample_interpreted(p, 3000, 77)
            assert_equal(String(a), String(d.sample))
            for s in range(len(a.memory)):
                assert_equal(a.memory[s], d.sample.memory[s])


def _geometric_tv(iters: List[Int], p: Float64) -> Float64:
    """TV distance of `iters` (extra rounds of a success-probability p loop) from
    Geom(p) on 0, 1, 2, ..."""
    var n = len(iters)
    var top = 0
    for v in iters:
        top = max(top, v)
    var hist = List[Int](length=top + 2, fill=0)
    for v in iters:
        hist[v] += 1
    var tv = 0.0
    var tail = 1.0
    for k in range(top + 1):
        var q = p * ((1.0 - p) ** k)
        tv += abs(Float64(hist[k]) / Float64(n) - q)
        tail -= q
    return 0.5 * (tv + tail)


def test_repeat_until_success_is_geometric() raises:
    """Exit test. Thresholds fixed before the first run: for 20000 shots the mean
    stays within 5 standard errors of (1-p)/p and TV < 0.02."""
    var shots = 20000
    for theta in [1.0, 1.9, 2.6]:
        var p = sin(theta / 2.0) ** 2
        var b = ProgramBuilder(1, 1)
        b.body.ry(theta, 0)
        b.body.measure(0, 0)
        b.begin_while(ce_lnot(b.bit(0)))
        b.body.reset(0)
        b.body.ry(theta, 0)
        b.body.measure(0, 0)
        b.end()
        var prog = b.finish()
        var r = sample_program(prog, shots, 12345)
        assert_equal(r.sample.counts.get("1", 0), shots)
        var mean = r.mean_iterations()
        var want = (1.0 - p) / p
        var se = sqrt((1.0 - p) / (p * p) / Float64(shots))
        assert_true(abs(mean - want) < 5.0 * se, "mean " + String(mean) + " want " + String(want))
        assert_true(_geometric_tv(r.iterations, p) < 0.02)


def test_infinite_loop_is_stopped() raises:
    """Exit test: a purely classical infinite loop ends in QM-E480."""
    var b = ProgramBuilder(1, 1)
    b.begin_while(CExpr.literal_bool(True))
    b.end()
    var p = b.finish()
    with assert_raises(contains="QM-E480"):
        _ = sample_program(p, 1, 1, max_iterations=100)
    # a counted loop uses its budget and no more
    var c = ProgramBuilder(1, 1)
    var k = c.declare("k", uint_type(16), True)
    c.assign(k, _lit(0))
    c.begin_while(ce_binary(CN.LT, c.read(k), _lit(100)))
    c.assign(k, ce_binary(CN.ADD, c.read(k), _lit(1)))
    c.end()
    var q = c.finish()
    assert_equal(sample_program(q, 2, 1, max_iterations=100).iterations[1], 100)
    with assert_raises(contains="QM-E480"):
        _ = sample_program(q, 2, 1, max_iterations=99)


def test_control_flow_paths() raises:
    # for with continue and break: i = 0..6, skip 2, stop at 5, sum = 0+1+3+4
    var b = ProgramBuilder(1, 1)
    var total = b.declare("total", int_type(16), True)
    var i = b.declare("i", int_type(16))
    b.assign(total, _lit(0))
    b.begin_for(i, _lit(0), _lit(1), _lit(6))
    b.begin_if(ce_binary(CN.EQ, b.read(i), _lit(2)))
    b.continue_()
    b.end()
    b.begin_if(ce_binary(CN.EQ, b.read(i), _lit(5)))
    b.break_()
    b.end()
    b.assign(total, ce_binary(CN.ADD, b.read(total), b.read(i)))
    b.end()
    assert_equal(_run_one(b, "total"), 8)
    # negative step, inclusive; empty range
    var c = ProgramBuilder(1, 1)
    var t2 = c.declare("t", int_type(16), True)
    var j = c.declare("j", int_type(16))
    c.assign(t2, _lit(0))
    c.begin_for(j, _lit(6), _lit(-2), _lit(0))
    c.assign(t2, ce_binary(CN.ADD, ce_binary(CN.MUL, c.read(t2), _lit(10)), c.read(j)))
    c.end()
    c.begin_for(j, _lit(3), _lit(1), _lit(2))
    c.assign(t2, _lit(-1))
    c.end()
    assert_equal(_run_one(c, "t"), 6420)
    # switch: 3 chooses the second case, 9 the default; no fall through
    for pair in [(3, 20), (9, 30), (1, 10)]:
        var d = ProgramBuilder(1, 1)
        var t3 = d.declare("t", int_type(16), True)
        var s = d.declare("s", int_type(16))
        d.assign(s, _lit(pair[0]))
        d.begin_switch(d.read(s))
        var v1: List[Int] = [1, 2]
        d.case_(v1)
        d.assign(t3, _lit(10))
        var v2: List[Int] = [3]
        d.case_(v2)
        d.assign(t3, _lit(20))
        d.default_()
        d.assign(t3, _lit(30))
        d.end()
        assert_equal(_run_one(d, "t"), pair[1])
    # nested loops: break leaves only the inner one
    var e = ProgramBuilder(1, 1)
    var t4 = e.declare("t", int_type(16), True)
    var a = e.declare("a", int_type(16))
    var o = e.declare("o", int_type(16))
    e.assign(t4, _lit(0))
    e.begin_for(o, _lit(0), _lit(1), _lit(2))
    e.begin_for(a, _lit(0), _lit(1), _lit(9))
    e.begin_if(ce_binary(CN.EQ, e.read(a), _lit(1)))
    e.break_()
    e.end()
    e.assign(t4, ce_binary(CN.ADD, e.read(t4), _lit(1)))
    e.end()
    e.end()
    assert_equal(_run_one(e, "t"), 3)


def test_runtime_angle() raises:
    """A gate reads a variable: three additions of pi/3 make ry(pi)."""
    var b = ProgramBuilder(1, 1)
    var a = b.declare("a", float_type())
    var i = b.declare("i", int_type(8))
    b.assign(a, CExpr.literal_float(0.0))
    b.begin_for(i, _lit(1), _lit(1), _lit(3))
    b.assign(a, ce_binary(CN.ADD, b.read(a), CExpr.literal_float(pi / 3.0)))
    b.end()
    var ang = b.angle(a)
    b.body.ry(ang, 0)
    b.body.measure(0, 0)
    var p = b.finish()
    assert_equal(sample_program(p, 50, 3).sample.counts.get("1", 0), 50)
    # the value is the one at the gate, across a block boundary with pending gates
    var c = ProgramBuilder(1, 1)
    var x = c.declare("x", float_type())
    c.assign(x, CExpr.literal_float(pi))
    c.body.ry(c.angle(x), 0)  # pending in the fuser when the assignment runs
    c.assign(x, CExpr.literal_float(0.0))
    c.body.ry(c.angle(x), 0)
    c.body.measure(0, 0)
    var q = c.finish()
    assert_equal(sample_program(q, 50, 3).sample.counts.get("1", 0), 50)
    # NaN reaches a gate: refused, not simulated
    var d = ProgramBuilder(1, 1)
    var y = d.declare("y", float_type())
    d.assign(y, CExpr.literal_float(-1.0))
    d.assign(y, ce_function(CF.SQRT, d.read(y)))
    d.body.ry(d.angle(y), 0)
    d.body.measure(0, 0)
    var r = d.finish()
    with assert_raises(contains="QM-E482"):
        _ = sample_program(r, 1, 3)


def test_inputs() raises:
    var b = ProgramBuilder(1, 1)
    var t = b.input("theta")
    b.body.ry(t, 0)
    b.body.measure(0, 0)
    var v = b.declare("v", int_type(8), True)
    b.assign(v, _lit(1))
    var p = b.finish()
    var vals: List[Float64] = [pi]
    assert_equal(sample_program(p, 20, 3, vals).sample.counts.get("1", 0), 20)
    with assert_raises(contains="QM-E101"):
        _ = sample_program(p, 20, 3)


def test_bit_assignment() raises:
    var b = ProgramBuilder(1, 3)
    b.assign_bits(0, 2, _lit(2))  # c[1:0] = 10
    b.begin_if(ce_binary(CN.EQ, CExpr.bits(0, 2), _lit(2)))
    b.assign_bits(2, 1, _lit(1))
    b.end()
    var p = b.finish()
    assert_equal(sample_program(p, 5, 3).sample.counts.get("110", 0), 5)


def main() raises:
    test_wrapping()
    test_division()
    test_shifts_and_bits()
    test_mixing_rules()
    test_casts()
    test_short_circuit()
    test_functions_and_power()
    test_float_and_angle_vars()
    test_definite_assignment()
    test_structure_errors()
    test_cfg_shape()
    test_from_circuit_matches_sample()
    test_repeat_until_success_is_geometric()
    test_infinite_loop_is_stopped()
    test_control_flow_paths()
    test_runtime_angle()
    test_inputs()
    test_bit_assignment()
    print("test_program OK")
