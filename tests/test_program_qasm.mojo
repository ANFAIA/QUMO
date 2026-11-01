"""OpenQASM 3 for Program: parsing, the emitter's fixed point, routing in load_any."""

from std.math import pi
from std.os import listdir
from std.testing import assert_equal, assert_true, assert_raises

from qumo.ir.qasm import parse_qasm, parse_qasm_program, load_any, to_qasm
from qumo.ir.program import Program, SK
from qumo.sim.runner import sample
from qumo.sim.program_runner import sample_program, ProgramResult


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _run(src: String, shots: Int = 200, seed: UInt64 = 5) raises -> ProgramResult:
    return sample_program(parse_qasm_program(src), shots, seed)


def _int(src: String, name: String) raises -> Int:
    """A deterministic integer output."""
    var r = _run(src, 3)
    return r.int_output(name)[2]


def _real(src: String, name: String) raises -> Float64:
    return _run(src, 2).real_output(name)[1]


def _error_of(src: String) -> String:
    try:
        _ = parse_qasm_program(src)
    except e:
        return String(e)
    return ""


def _bin(s: String) -> Int:
    var v = 0
    for c in s.as_bytes():
        v = 2 * v + Int(c) - 0x30
    return v


def _fixed_point(src: String) raises -> String:
    """to_qasm stabilises after one pass and keeps the meaning."""
    var t1 = to_qasm(parse_qasm_program(src))
    var t2 = to_qasm(parse_qasm_program(t1))
    assert_equal(t1, t2)
    var p = parse_qasm_program(src)
    var vals = List[Float64](length=p.num_params(), fill=0.3)
    var a = sample_program(p, 400, 9, vals)
    var b = sample_program(parse_qasm_program(t1), 400, 9, vals)
    assert_equal(String(a.sample), String(b.sample))
    return t1^


def test_qiskit_fixtures() raises:
    """Real qiskit.qasm3.dumps output (tests/fixtures/control_flow)."""
    var names = listdir("tests/fixtures/control_flow")
    assert_true(len(names) >= 4)
    for nm in names:
        if not nm.endswith(".qasm"):
            continue
        var src = _read("tests/fixtures/control_flow/" + nm)
        var p = load_any(src)
        assert_true(p.has_runtime_region())
        _ = _fixed_point(src)
    # the counter of repeat-until-success stops at 7, and m holds it
    var r = sample_program(load_any(_read("tests/fixtures/control_flow/qiskit_rus_counter.qasm")), 2000, 3)
    assert_true(r.sample.probability("00011") > 0.45)  # one round: c = 1, m = 1
    assert_true(r.sample.probability("00101") > 0.2)  # two rounds
    # b = a ^ 5, then b[0] takes !(a == b), which is true: a ^ 5 never equals a
    var e = sample_program(load_any(_read("tests/fixtures/control_flow/qiskit_store_expr.qasm")), 500, 3)
    for kv in e.sample.counts.items():
        var a = _bin(String(kv.key[byte=3:6]))
        var b = _bin(String(kv.key[byte=0:3]))
        assert_equal(b, (a ^ 5) | 1, kv.key)


def test_load_any_routing() raises:
    # what a Circuit holds stays flat
    var flat = load_any("qubit[2] q; bit[2] c; h q[0]; c[0] = measure q[0]; if (c[0]) x q[1]; c[1] = measure q[1];")
    assert_true(not flat.has_runtime_region())
    assert_true(flat.try_flatten())
    var flat2 = load_any("int i = 1; qubit[2] q; h q[i];")
    assert_true(not flat2.has_runtime_region())
    # these need a program
    var src: List[String] = [
        "qubit[1] q; bit[1] c; while (!c[0]) { h q[0]; c[0] = measure q[0]; }",
        "qubit[1] q; bit[2] c; x q[0]; c[0] = measure q[0]; if (c[0] == 1) { c[0] = measure q[0]; c[1] = measure q[0]; }",
        "uint[4] k = 0; k = k + 1; qubit[1] q;",
        "qubit[1] q; bit[1] c; bit b = c[0]; if (b) x q[0];",
        "qubit[2] q; bit[2] c; int k = 1; if (k == 1) x q[0];",
        "qubit[1] q; bit[1] c; int k = 1; c[0] = 1;",
        "qubit[2] q; bit[2] c; int d; d = c; rz(d) q[0];",
        "qubit[2] q; bit[2] c; if (c[0] || c[1]) x q[0];",
        "qubit[2] q; bit[2] c; if (c[0]) { if (c[1]) x q[0]; }",
        "input int n; qubit[1] q; for int i in [1:n] { h q[0]; }",
    ]
    for s in src:
        assert_true(load_any(s).has_runtime_region(), s)
    # a plain mistake reports the flat error
    with assert_raises(contains="QM-E510"):
        _ = load_any("qubit[1] q; hh q[0];")
    with assert_raises(contains="QM-E520"):
        _ = load_any("qubit[1] q; h r[0];")
    # parse_qasm keeps refusing them
    with assert_raises(contains="QM-E504"):
        _ = parse_qasm("qubit[1] q; while (true) { h q[0]; }")
    with assert_raises(contains="QM-E504"):
        _ = parse_qasm("qubit[2] q; bit[2] c; rz(c) q[0];")


def test_values_and_precedence() raises:
    assert_equal(_int("output int r; r = 2 + 3 * 4;", "r"), 14)
    assert_equal(_int("output int r; int a = 1; a = 2; r = (a + 3) * 4;", "r"), 20)
    assert_equal(_int("output int r; r = 7 - 2 - 1;", "r"), 4)
    assert_equal(_int("output uint r; uint a; a = 6; r = a & 3 | 8;", "r"), 10)
    assert_equal(_int("output uint r; uint a; a = 1; r = a << 2 + 1;", "r"), 8)
    assert_equal(_int("output int r; int a; a = 5; if (a > 3 && a < 6 || a == 0) { r = 1; } else { r = 0; }", "r"), 1)
    assert_equal(_int("output int r; int a; a = 2; r = a ** 3 ** 2;", "r"), 512)
    assert_equal(_int("output int r; int a; a = -7; r = a / 2;", "r"), -3)
    assert_equal(_int("output int r; int a; a = -7; r = a % 3;", "r"), -1)
    assert_equal(_int("output int r; int a; a = 5; a += 2; a *= 3; a -= 1; a /= 4; r = a;", "r"), 5)
    assert_equal(_int("output uint r; uint[8] a; a = 3; a <<= 2; a |= 1; a ^= 5; a &= 15; a >>= 1; a %= 3; r = a;", "r"), 1)
    assert_equal(_int("output int r; bool b = true; b = !b; if (b) r = 1; else r = 2;", "r"), 2)
    assert_equal(_int("output uint[4] r; uint[4] a; a = 15; r = a + 1;", "r"), 0)
    assert_equal(_int("output int r; int[8] a; a = 127; a = a + 1; r = a;", "r"), -128)
    assert_equal(_int("output uint r; bit[4] c; c = \"1010\"; r = c;", "r"), 10)
    assert_equal(_int("output uint r; bit[4] c; c[1] = 1; c[3] = true; r = c;", "r"), 10)
    assert_equal(_int("output uint r; bit[4] c; c[0:1] = 3; r = c >> 1;", "r"), 1)
    assert_equal(_int("output int r; bit[3] c = 5; r = int[8](c) - 7;", "r"), -2)
    assert_equal(_int("output int r; float f; f = 2.9; r = int(f);", "r"), 2)
    assert_equal(_int("output uint r; r = popcount(bit[8](255));", "r"), 8)
    assert_true(abs(_real("output float r; float x; x = 2.0; r = sqrt(x) * 3 / 2 + 1;", "r") - 3.121320343559643) < 1e-12)
    # literals divide as integers, as a variable would
    assert_true(abs(_real("output float r; r = 1/2;", "r")) < 1e-15)
    assert_equal(_int("output int r; const int a = 7; r = a / 2;", "r"), 3)
    # a long literal stays exact
    assert_equal(_int("output int r; r = 9007199254740993;", "r"), 9007199254740993)
    assert_equal(_int("output uint r; r = 18446744073709551615;", "r"), -1)


def test_folding_and_scope() raises:
    # never reassigned: folds, so it can index qubits
    var p = parse_qasm_program("int i = 1; qubit[2] q; bit[1] c; h q[i]; c[0] = measure q[i]; while (false) { }")
    assert_equal(len(p.vars), 0)
    assert_equal(_int("const uint[4] a = 20; output int r; r = 1; if (a == 4) r = 2;", "r"), 2)
    # reassigned: runtime, and indexing with it is refused
    assert_true(_error_of("int i = 1; i = 0; qubit[2] q; h q[i];").startswith("QM-E584"))
    assert_true(_error_of("qubit[2] q; bit[2] c; x q[c];").startswith("QM-E5"))
    assert_true(_error_of("qubit[2] q; bit[1] c; for int i in [0:1] { h q[i]; if (c[0]) break; }").startswith("QM-E584"))
    # names resolve to the innermost declaration, on the constant path too
    assert_true(_error_of("const int i = 1; qubit[2] q; bit[1] c; for int i in [0:1] { if (c[0]) break; h q[i]; }").startswith("QM-E584"))
    assert_equal(_int("output int r; int x; x = 1; if (x == 1) { const int x = 5; r = x; } else { r = 0; }", "r"), 5)
    assert_true(_error_of("bit[1] c; for float f in [0:1] { if (c[0]) break; }").startswith("QM-E580"))
    assert_true(_error_of("qubit[1] q; int x; x = 1; x = measure q[0];").startswith("QM-E504"))
    # a loop that cannot run ahead of time keeps its index at run time, but a
    # loop without a jump still unrolls and indexes qubits
    var u = _run("qubit[3] q; bit[3] c; for int i in [0:2] { x q[i]; c[i] = measure q[i]; }", 5)
    assert_equal(u.sample.counts.get("111", 0), 5)
    # inner scope shadows; the outer value comes back
    assert_equal(_int("output int r; int x; x = 1; { int x; x = 5; } r = x;", "r"), 1)
    assert_equal(_int("output int r; r = 0; int x; x = 1; if (true) { int x = 7; x = x + 1; r = x; }", "r"), 8)
    assert_true(_error_of("int x = 1; int x = 2;").startswith("QM-E522"))
    # a declaration in a loop body runs each time
    assert_equal(_int("output int r; r = 0; int i; i = 0; while (i < 3) { int y = 2; y = y + 1; r = r + y; i = i + 1; }", "r"), 9)
    # a constant for with a declaration in its body
    assert_equal(_int("output int r; r = 0; for int i in [1:3] { int y = i; y = y * 2; r = r + y; }", "r"), 12)


def test_errors() raises:
    assert_true(_error_of("float[32] x;").startswith("QM-E582"))
    assert_true(_error_of("angle a; a = 1.0;").startswith("QM-E582"))
    assert_true(_error_of("int[65] x;").startswith("QM-E582"))
    assert_true(_error_of("uint[0] x;").startswith("QM-E582"))
    assert_true(_error_of("break;").startswith("QM-E583"))
    assert_true(_error_of("continue;").startswith("QM-E583"))
    assert_true(_error_of("int x; int y; y = x + 1;").startswith("QM-E581"))
    assert_true(_error_of("int x; x = 1; uint y; y = 2; output int r; r = x + y;").startswith("QM-E580"))
    assert_true(_error_of("angle[8] a; a = 1.0; float f; f = a;").startswith("QM-E580"))
    assert_true(_error_of("int x; x = 1 / 0;").startswith("QM-E533"))
    assert_true(_error_of("int x = 1; x = x; const int y = x;").startswith("QM-E584"))
    assert_true(_error_of("const int x;").startswith("QM-E584"))
    assert_true(_error_of("complex[float] z;").startswith("QM-E5"))
    assert_true(_error_of("int x; x = 1; x = rotl(x, 1);").startswith("QM-E534"))
    # positions survive
    var msg = _error_of("int x;\nx = 1;\nuint y;\ny = x + 1;\nx = x + y;")
    assert_true(msg.find("line 5") > 0, msg)
    # a run-time failure names its line
    var run = String()
    try:
        _ = _run("int x;\nx = 0;\nint y;\ny = 1 / x;", 1)
    except e:
        run = String(e)
    assert_true(run.startswith("QM-E481") and run.find("line 4") > 0, run)


def test_gate_arguments() raises:
    # constant arguments mean what parse_qasm means, 1/2 included
    var a = parse_qasm_program("qubit[1] q; bit[1] c; while (false) { } rz(1/2) q[0]; rx(-7 % 3) q[0]; ry(2 ** -1) q[0];")
    var b = parse_qasm("qubit[1] q; bit[1] c; rz(1/2) q[0]; rx(-7 % 3) q[0]; ry(2 ** -1) q[0];")
    for i in range(3):
        assert_equal(a.body.ops[i].p0.value(), b.ops[i].p0.value())
    # a variable, affine: no temporary
    var t = to_qasm(parse_qasm_program("qubit[1] q; float v; v = 0.5; rz(2*v + 0.25) q[0]; v = 1.0; ry(-v) q[0];"))
    assert_true(t.find("rz(2*v + 0.25)") > 0, t)
    assert_true(t.find("__t") < 0, t)
    # products of variables, and integer arithmetic, go through a temporary
    var u = _run("qubit[1] q; bit[1] c; float a; float b; a = pi/2; b = 2.0; rx(a * b) q[0]; c[0] = measure q[0];", 30)
    assert_equal(u.sample.counts.get("1", 0), 30)
    var v = _run("qubit[1] q; bit[1] c; uint[2] k; k = 3; k = k + 2; rx(k * pi) q[0]; c[0] = measure q[0];", 30)
    assert_equal(v.sample.counts.get("1", 0), 30)
    # an angle[n] argument is its fraction of a turn
    var w = _run("qubit[1] q; bit[1] c; angle[3] a; a = pi / 2; a = a + a; rx(a) q[0]; c[0] = measure q[0];", 30)
    assert_equal(w.sample.counts.get("1", 0), 30)
    # operators the constant path stops at
    var sh = _run("qubit[1] q; bit[1] c; uint k; k = 1; rx(float(k << 1) * pi / 2) q[0]; rx((1 << 0) * pi) q[0]; c[0] = measure q[0];", 30)
    assert_equal(sh.sample.counts.get("0", 0), 30)
    assert_equal(_int("output int r; r = 0; for int i in [0:(1 << 2)] { r = r + 1; if (r == 9) break; }", "r"), 5)
    # the temporary is assigned where the gate is: a later change does not count
    var x = _run("qubit[1] q; bit[1] c; int k; k = 2; for int i in [1:k] { rx(pi / k) q[0]; } k = 0; c[0] = measure q[0];", 30)
    assert_equal(x.sample.counts.get("1", 0), 30)
    # inside a modifier and through a user gate
    var y = _run("gate g(t) a { rx(t) a; } qubit[2] q; bit[2] c; float t; t = pi; x q[0]; ctrl @ rx(t) q[0], q[1]; c[0] = measure q[1]; g(t) q[1]; c[1] = measure q[1];", 30)
    assert_equal(y.sample.counts.get("01", 0), 30)
    var inp = parse_qasm_program("input float theta; qubit[1] q; bit[1] c; float s; s = 2.0; rx(theta * s) q[0]; c[0] = measure q[0];")
    var vals: List[Float64] = [pi / 2]
    assert_equal(sample_program(inp, 30, 2, vals).sample.counts.get("1", 0), 30)
    assert_equal(inp.num_params(), 1)


def test_flow() raises:
    # if with else without braces; nested; a body writing what it tests
    var r = _run("qubit[3] q; bit[3] c; x q[0]; c[0] = measure q[0]; if (c[0]==1) { c[0] = measure q[1]; x q[2]; } c[2] = measure q[2];", 20)
    assert_equal(r.sample.counts.get("100", 0), 20)  # the flat parser refuses this
    assert_equal(_int("output int r; bit[2] c; c[1] = 1; if (c[1]) { if (c[0]) r = 1; else r = 2; } else r = 3;", "r"), 2)
    # int copy is a copy
    var d = _run("qubit[2] q; bit[2] c; int d; bit[2] r; qubit[1] t; x q[0]; x q[1]; c[0] = measure q[0]; d = c; c[1] = measure q[1]; switch (d) { case 1 { x t[0]; } default { } } r[0] = measure t[0];", 20)
    assert_equal(d.sample.counts.get("0111", 0), 20)
    # for with runtime bound, step, negative direction, set with break
    assert_equal(_int("output int r; int n; n = 3; r = 0; for int i in [1:n] { r = r + i; }", "r"), 6)
    assert_equal(_int("output int r; int n; n = 2; r = 0; for int i in [10:-n:5] { r = r + i; }", "r"), 24)
    assert_equal(_int("output int r; r = 0; for int i in {3, 5, 9} { if (i == 9) break; r = r + i; }", "r"), 8)
    assert_equal(_int("output int r; r = 0; for uint[2] i in [0:5] { if (i == 0) continue; r = r + 1; }", "r"), 4)
    # break in a switch case leaves the loop; Qiskit's old form keeps its meaning
    assert_equal(_int("output int r; r = 0; int i; i = 0; while (true) { switch (i) { case 2 { break; } default { r = r + 1; } } i = i + 1; }", "r"), 2)
    assert_equal(_int("output int r; int k; k = 1; r = 0; switch (k) { case 0: { r = 1; } break; case 1: case 2: { r = 2; } break; default: { r = 3; } break; }", "r"), 2)
    assert_equal(_int("output int r; int k; k = 9; r = 0; switch (k) { case 1: default: { r = 3; } }", "r"), 3)
    # box, pragmas and delay inside a region
    var b = _run("qubit[1] q; bit[1] c; int i; i = 0; while (i < 2) { box { x q[0]; } #pragma qumo tick\n i = i + 1; } c[0] = measure q[0];", 5)
    assert_equal(b.sample.counts.get("0", 0), 5)
    # noise in a loop
    var n = _run("qubit[1] q; bit[1] c; int i; i = 0; while (i < 1) { #pragma qumo x_error(1) q[0];\n i = i + 1; } c[0] = measure q[0];", 5)
    assert_equal(n.sample.counts.get("1", 0), 5)


def test_fixed_points() raises:
    var src: List[String] = [
        "qubit[2] q; bit[2] c; int k; k = 0; while (k < 3 && !c[0]) { h q[0]; c[0] = measure q[0]; k = k + 1; } c[1] = measure q[1];",
        "qubit[1] q; bit[3] c; uint[8] x; x = 200; x = x + 100; c = x; if (x > 40 || c[0]) { h q[0]; } else { x = x / 3; } c[0] = measure q[0];",
        "input float t; qubit[2] q; bit[2] c; angle[12] a; a = pi / 3; a = a * 2; float f; f = 0.1; for int i in [0:3] { if (i == 2) continue; rz(f + t) q[0]; rx(a) q[1]; if (i == 3) break; } c = measure q;",
        "qubit[1] q; bit[2] c; int d; d = c; switch (d) { case 0 { x q[0]; } case 1, 2 { z q[0]; } default { } } c[1] = measure q[0];",
        "output int r; output float s; int a; a = 7; bool b; b = a > 3; s = sin(a * 0.5) + 2 ** 0.5; if (b && a != 2) r = 1; else r = 0;",
        "output uint r; bit[4] c; c = \"1001\"; uint[4] u; u = ~(c ^ 3); r = u >> 1 | 8;",
        "qubit[1] q; bit[1] c; int i; for int j in [0:2] { for int k in [0:2:6] { if (k == 4) break; x q[0]; } } i = 2; while (i > 0) { i -= 1; } c[0] = measure q[0];",
    ]
    for s in src:
        _ = _fixed_point(s)
    var flat = String("qubit[2] q; bit[2] c; h q[0]; c[0] = measure q[0]; if (c[0]) x q[1]; c[1] = measure q[1];")
    assert_equal(to_qasm(load_any(flat)), to_qasm(parse_qasm(flat)))


def test_outputs_in_counts() raises:
    # int outputs are per shot and independent of counts keys
    var r = _run("qubit[1] q; bit[1] c; output int rounds; rounds = 0; while (!c[0]) { h q[0]; c[0] = measure q[0]; rounds = rounds + 1; reset q[0]; }", 2000)
    var mean = 0.0
    for v in r.int_output("rounds"):
        mean += Float64(v)
    mean /= 2000.0
    assert_true(abs(mean - 2.0) < 0.15, String(mean))
    assert_equal(r.sample.counts.get("1", 0), 2000)
    assert_equal(len(r.iterations), 2000)
    assert_equal(r.out_names[0], "rounds")


def main() raises:
    test_load_any_routing()
    test_values_and_precedence()
    test_folding_and_scope()
    test_errors()
    test_gate_arguments()
    test_flow()
    test_fixed_points()
    test_outputs_in_counts()
    test_qiskit_fixtures()
    print("test_program_qasm OK")
