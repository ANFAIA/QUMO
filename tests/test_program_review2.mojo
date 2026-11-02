"""Regressions from the second independent review of the Program stack.

Each test names its finding by number, or R1 and QM-W055 for the two rules that
came with them. The fixes are on top of commit 5e1944f, and each of these failed
there; the exceptions are named in the report.
"""

from std.math import pi
from std.testing import assert_equal, assert_true, assert_raises

from qumo.ai.diagnostics import error_catalog
from qumo.ir.classical import CN, CExpr, ce_binary, ce_neg
from qumo.ir.gates import Op
from qumo.ir.qasm import load_any, parse_qasm, parse_qasm_program, parse_qasm_with_warnings, to_qasm
from qumo.sim.program_runner import ProgramResult, sample_program

comptime H = "OPENQASM 3.0; qubit[4] q; bit[4] c; "
comptime EPS = 1e-12


def _flat(body: String) raises -> Float64:
    """The first gate angle of a flat parse."""
    return parse_qasm(H + body).ops[0].p0.value()


def _prog(body: String) raises -> Float64:
    """The same in the program parser, which must agree."""
    return parse_qasm_program(H + body + " while (false) { }").body.ops[0].p0.value()


def _angle(body: String, want: Float64) raises:
    var a = _flat(body)
    var b = _prog(body)
    assert_true(abs(a - want) <= EPS * max(1.0, abs(want)), body + " flat " + String(a))
    assert_true(abs(b - want) <= EPS * max(1.0, abs(want)), body + " program " + String(b))


def _err_flat(body: String) -> String:
    try:
        _ = parse_qasm(H + body)
    except e:
        return String(e)
    return ""


def _err_prog(body: String) -> String:
    try:
        _ = sample_program(parse_qasm_program(H + body), 3, 5)
    except e:
        return String(e)
    return ""


def _is(msg: String, code: String) raises:
    assert_true(msg.startswith(code), "expected " + code + ", got: " + msg)


def _int(body: String, name: String = "r") raises -> Int:
    return sample_program(parse_qasm_program(H + body), 3, 5).int_output(name)[2]


def _warnings(src: String) raises -> List[String]:
    var w = List[String]()
    _ = parse_qasm_with_warnings(src, w)
    return w^


def _count(w: List[String], code: String) -> Int:
    var n = 0
    for s in w:
        if s.startswith(code):
            n += 1
    return n


def test_r1_constants_do_not_wrap() raises:
    _angle("float t = 2 ** 70; rz(t) q[0];", 1180591620717411303424.0)
    _angle("const int n = 63; rz(pi / 2 ** n) q[0];", pi / 9223372036854775808.0)
    _angle("const int n = 64; rz(pi / 2 ** n) q[0];", pi / 18446744073709551616.0)
    _angle("const int a = 10000000000; rz(a * a) q[0];", 1e20)
    _angle("for uint k in [1:1] { rz(-k) q[0]; }", -1.0)
    _angle("for uint k in [3:3] { rz(1 - k) q[0]; }", -2.0)
    _angle("for uint k in [1:1] { rz(pi * 2.0 ** (-k)) q[0]; }", pi / 2.0)
    _angle("for uint k in [2:2] { rz(2 ** (-k)) q[0]; }", 0.25)
    assert_true(_flat("const int n = 10; rz(n ** -3) q[0];") == 0.001)


def test_r1_a_negative_exponent_gives_a_float() raises:
    _angle("const float x = 2 ** -1; rz(x) q[0];", 0.5)
    _angle("const int e = -3; rz(10 ** e) q[0];", 0.001)
    assert_true(ce_binary(CN.POW, CExpr.literal_int(2), CExpr.literal_int(-1)).const_float() == 0.5)
    # run time did not change
    _is(_err_prog("output int r; int a = 2; a = a; int b = -1; b = b; r = a ** b;"), "QM-E481")


def test_r1_unsized_int_and_uint_mix() raises:
    _angle("const int a = 3; const uint b = 5; rz(a + b) q[0];", 8.0)
    _angle("const int a = 3; const uint b = 5; rz(a - b) q[0];", -2.0)
    assert_equal(parse_qasm(H + "const int n = 3; const uint k = 1; h q[n - k];").ops[0].q0, 2)
    assert_equal(parse_qasm_program(H + "const int n = 3; const uint k = 1; h q[n - k]; while (false) { }").body.ops[0].q0, 2)
    _is(_err_prog("output int r; int a = 1; uint[8] b = 1; a = a; b = b; r = a + b;"), "QM-E580")


def test_r1_division_still_truncates() raises:
    _angle("const int a = 7; const float x = a / 2; rz(x) q[0];", 3.0)
    _angle("const int a = -7; rz(a / 2) q[0];", -3.0)
    _angle("const int a = -7; rz(a % 3) q[0];", -1.0)


def test_r1_sized_types_wrap_and_out_of_range_is_an_error() raises:
    _angle("int[8] x = 300; rz(x) q[0];", 44.0)
    _angle("const int k = 300; rz(int[8](k)) q[0];", 44.0)
    _is(_err_flat("const int x = 2 ** 64; rz(x) q[0];"), "QM-E482")
    _is(_err_flat("int x = 18446744073709551615; rz(x) q[0];"), "QM-E482")
    _is(_err_flat("uint x = -1; rz(x) q[0];"), "QM-E482")
    _is(_err_prog("int x = 18446744073709551615; h q[0];"), "QM-E482")
    _is(_err_prog("output uint x; x = -1;"), "QM-E482")
    _is(_err_flat("uint[8] x = 2 ** 70; rz(x) q[0];"), "QM-E482")


def test_r1_negation_at_the_edges() raises:
    # D6: this reads 2^63 - 1 and negates it
    assert_equal(_int("output int r; r = -(9223372036854775808 - 1);"), -9223372036854775807)
    assert_equal(_int("output int r; r = -9223372036854775808;"), -9223372036854775807 - 1)
    assert_true(ce_neg(CExpr.literal_uint(-1)).const_float() == -18446744073709551615.0)
    _angle("rz(18446744073709551616 + 0) q[0];", 18446744073709551616.0)


def test_w055_truncated_constant_used_as_a_float() raises:
    var w = _warnings(H + "float t = 3 / 4 * pi; rz(t) q[0];")
    assert_equal(_count(w, "QM-W055"), 1)
    assert_true(w[0].find("3.0 / 4") > 0, w[0])
    assert_equal(_count(_warnings(H + "for int i in [1:3] { rz(i / 4 * pi) q[0]; }"), "QM-W055"), 1)
    assert_equal(_count(_warnings(H + "const int k = 3; rz(k / 4) q[0];"), "QM-W055"), 1)
    # an integer stays an integer, and nothing was lost here
    assert_equal(_count(_warnings(H + "int n = 7 / 2; h q[n];"), "QM-W055"), 0)
    assert_equal(_count(_warnings(H + "float t = 4 / 2 * pi; rz(t) q[0];"), "QM-W055"), 0)
    assert_equal(_count(_warnings(H + "float t = 3.0 / 4 * pi; rz(t) q[0];"), "QM-W055"), 0)


def test_w055_real_division_in_a_literal_argument() raises:
    var w = _warnings(H + "rz(3/4*pi) q[0];")
    assert_equal(_count(w, "QM-W055"), 1)
    assert_true(w[0].find("Qiskit") > 0, w[0])
    assert_true(abs(parse_qasm(H + "rz(3/4*pi) q[0];").ops[0].p0.value() - 0.75 * pi) < 1e-15)
    assert_equal(_count(_warnings(H + "rz(pi/2) q[0]; rz(4/2) q[0]; rz(3.0/4) q[0];"), "QM-W055"), 0)
    assert_equal(_count(_warnings(H + "gate g a { rz(1/2) a; } g q[0]; g q[1];"), "QM-W055"), 1)


def test_w055_needs_an_openqasm_3_header() raises:
    assert_equal(_count(_warnings("OPENQASM 2.0; qreg q[1]; rz(3/4*pi) q[0];"), "QM-W055"), 0)
    assert_equal(_count(_warnings("qubit[1] q; rz(3/4*pi) q[0];"), "QM-W055"), 0)
    var found = False
    for e in error_catalog():
        found = found or e.code == "QM-W055"
    assert_true(found)


def test_1_non_finite_constants() raises:
    _is(_err_flat("float t = sqrt(-1.0); rz(t) q[0];"), "QM-E533")
    _is(_err_flat("const float t = ln(0.0); rz(t) q[0];"), "QM-E533")
    _is(_err_flat("const float a = -1.0; rz(sqrt(a)) q[0];"), "QM-E533")
    _is(_err_flat("const float a = 0.0; rz(ln(a)) q[0];"), "QM-E533")
    _is(_err_flat("const float a = 2.0; rz(arcsin(a)) q[0];"), "QM-E533")
    _is(_err_flat("float t = 1e308 * 10; rz(t) q[0];"), "QM-E533")
    _is(_err_flat("rz(arcsin(2.0)) q[0];"), "QM-E533")
    _is(_err_flat("const float a = -1.0; gate g b { rz(sqrt(a)) b; } g q[0];"), "QM-E533")
    _is(_err_flat("gate g b { rz(1e308 * 10) b; } g q[0];"), "QM-E533")


def test_2_typed_constants_in_gate_bodies() raises:
    _angle("const int k = 3; gate g r { rx(k / 2) r; } g q[0];", 1.0)
    _angle("const int m = -7; gate g r { rx(m % 3) r; } g q[0];", -1.0)
    _angle("const int[8] a = 100; gate g r { rx(a * 2) r; } g q[0];", -56.0)
    # gate parameters stay affine
    _angle("const int k = 3; gate g(t) r { rx(t / 4 + k / 2) r; } g(2) q[0];", 1.5)
    var c = parse_qasm("OPENQASM 3.0; input float u; qubit[1] q; const int k = 3; gate g(t) r { rx(2 * t + k / 2) r; } g(u) q[0];")
    assert_true(c.ops[0].p0.coeff == 2.0 and c.ops[0].p0.offset == 1.0)
    _angle("gate g(t) r { for int i in [1:1] { rz(i / 2 + t) r; } } g(0.5) q[0];", 0.5)


def test_16_modifiers_read_typed_constants() raises:
    var c = parse_qasm(H + "const int k = 5; pow(k / 2) @ x q[0];")
    assert_equal(c.count(Op.X), 2)
    assert_equal(parse_qasm(H + "const int k = 3; ctrl(k / 2) @ x q[0], q[1];").count(Op.CX), 1)


def test_4_uint_range_across_2_to_63() raises:
    assert_equal(_int("output int r = 0; uint n = 9223372036854775808; n = n; for uint i in [9223372036854775807:n] { r += 1; }"), 2)
    assert_equal(_int("output int r = 0; for uint i in [9223372036854775807:9223372036854775808] { r += 1; }"), 2)
    assert_equal(_int("output int r = 0; uint n = 18446744073709551615; n = n; for uint i in [18446744073709551614:n] { r += 1; }"), 2)
    assert_equal(_int("output int r = 0; uint a = 9223372036854775809; a = a; for uint i in [a:-1:9223372036854775807] { r += 1; }"), 3)
    assert_equal(_int("output int r = 0; uint n = 18446744073709551615; n = n; for uint i in [18446744073709551613:2:n] { r += 1; if (r > 9) { break; } }"), 2)
    # a signed bound keeps a signed comparison
    assert_equal(_int("output int r = 0; int n = -1; n = n; for int i in [-2:n] { r += 1; }"), 2)
    assert_equal(_int("output int r = 0; int n = -1; n = n; for uint i in [0:n] { r += 1; }"), 0)


def test_5_angle_is_not_an_integer() raises:
    _is(_err_flat("angle[2] a = pi/2; h q[a];"), "QM-E530")
    _is(_err_flat("angle[3] a = pi/2; for int i in [0:a] { x q[0]; }"), "QM-E530")
    _is(_err_flat("angle[3] a = pi/2; qubit[a] r;"), "QM-E530")
    # what may pass
    assert_equal(parse_qasm(H + "const bool b = true; h q[b];").ops[0].q0, 1)
    assert_equal(parse_qasm(H + "const float n = 2.0; h q[n];").ops[0].q0, 2)
    assert_equal(parse_qasm(H + "const uint[2] b = 3; h q[b];").ops[0].q0, 3)


def test_7_block_bit_keeps_its_size() raises:
    var p = String("OPENQASM 3.0; qubit[1] q; for int i in [0:1] { bit[i + 1] b; b = 3; }")
    var message = String()
    try:
        _ = sample_program(parse_qasm_program(p), 1, 1)
    except e:
        message = String(e)
    _is(message, "QM-E504")
    assert_equal(sample_program(parse_qasm_program("OPENQASM 3.0; qubit[1] q; for int i in [0:1] { bit[2] b; b = 3; }"), 1, 1).sample.num_clbits, 2)


def test_12_symbolic_constants_read_as_affine() raises:
    var c = parse_qasm("OPENQASM 3.0; input float t; qubit[1] q; float u = t * 2; float w = 1.0; rz(u + w) q[0];")
    assert_true(c.ops[0].p0.is_symbolic() and c.ops[0].p0.coeff == 2.0 and c.ops[0].p0.offset == 1.0)
    var d = parse_qasm("OPENQASM 3.0; input float t; qubit[1] q; const int k = 2; float u = t * k; const float s = 0.5; rz(u * s) q[0]; rz(u) q[0];")
    assert_true(d.ops[0].p0.coeff == 1.0 and d.ops[1].p0.coeff == 2.0)
    var e = parse_qasm("OPENQASM 3.0; input float t; qubit[1] q; float u = t; float v = 1.0; rz(u + v) q[0];")
    assert_true(e.ops[0].p0.coeff == 1.0 and e.ops[0].p0.offset == 1.0)
    # what is not affine keeps its error
    var message = String()
    try:
        _ = parse_qasm("OPENQASM 3.0; input float t; qubit[1] q; float u = t; float v = 1.0; rz(u * u + v) q[0];")
    except err:
        message = String(err)
    _is(message, "QM-E531")


def test_13_conditions_with_typed_constants() raises:
    var c = parse_qasm(H + "const int v = 1; if (c[0] == v && c[1] == 0) x q[0];")
    assert_equal(c.ops[0].cond_reg >= 0, True)
    var d = parse_qasm(H + "const int v = 0; if (c[0] == 1 && c[1] == v) x q[0];")
    assert_equal(d.ops[0].cond_reg >= 0, True)
    var e = parse_qasm(H + "const int v = 1; if (c[0] == v) x q[0];")
    assert_equal(e.ops[0].cond_val, 1)


def test_15_sized_casts_in_a_flat_gate_argument() raises:
    _angle("const int k = 300; rz(int[8](k)) q[0];", 44.0)
    _angle("const int k = 300; rz(k + int[8](k)) q[0];", 344.0)
    _angle("for int i in [1:1] { rz(int[8](i) + 1) q[0]; }", 2.0)
    _angle("rz(int[8](300)) q[0];", 44.0)


def test_17_bool_and_bit_next_to_a_narrow_type() raises:
    assert_equal(_int("output int r; uint[8] u = 255; u = u; bool b = true; b = b; r = int(u + b);"), 256)
    assert_equal(_int("output int r; int[8] a = 127; a = a; r = a + true;"), 128)
    assert_equal(_int("output int r; uint[8] u = 255; u = u; c[0] = 1; r = int(u + c[0]);"), 256)
    assert_equal(_int("output int r; uint[8] u = 255; u = u; r = int(true + u);"), 256)
    assert_equal(_int("output uint r; uint u = 18446744073709551615; u = u; c[0] = 1; r = u - c[0];") + 2, 0)
    assert_equal(_int("output int r; int[8] a = -128; a = a; r = a - true;"), -129)


def test_18_float_declarations_and_range_bounds() raises:
    _angle("float t = 1 / 2; rz(t) q[0];", 0.0)
    assert_true(_flat("float[32] t = 1 / 2; rz(t) q[0];") == 0.0)
    assert_true(_flat("float[16] t = 0.25; rz(t * 2) q[0];") == 0.5)
    assert_equal(_int("output int r = 0; for int i in [0.0:1] { r += 1; }"), 2)
    assert_equal(_int("output int r = 0; for int i in [0.0:1] { if (r > 9) { break; } r += 1; }"), 2)
    assert_equal(_int("output int r = 0; for int i in [0:1.0:1] { if (r > 9) { break; } r += 1; }"), 2)
    _is(_err_prog("output int r = 0; for int i in [0.5:1] { if (r > 9) { break; } r += 1; }"), "QM-E570")


def main() raises:
    test_r1_constants_do_not_wrap()
    test_r1_a_negative_exponent_gives_a_float()
    test_r1_unsized_int_and_uint_mix()
    test_r1_division_still_truncates()
    test_r1_sized_types_wrap_and_out_of_range_is_an_error()
    test_r1_negation_at_the_edges()
    test_w055_truncated_constant_used_as_a_float()
    test_w055_real_division_in_a_literal_argument()
    test_w055_needs_an_openqasm_3_header()
    test_1_non_finite_constants()
    test_2_typed_constants_in_gate_bodies()
    test_16_modifiers_read_typed_constants()
    test_4_uint_range_across_2_to_63()
    test_5_angle_is_not_an_integer()
    test_7_block_bit_keeps_its_size()
    test_12_symbolic_constants_read_as_affine()
    test_13_conditions_with_typed_constants()
    test_15_sized_casts_in_a_flat_gate_argument()
    test_17_bool_and_bit_next_to_a_narrow_type()
    test_18_float_declarations_and_range_bounds()
    print("test_program_review2 OK")
