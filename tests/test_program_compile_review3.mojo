"""Regressions from the third review of the compiler for programs: angles that passes
used to round off under the verifier's eyes (finding 1)."""

from std.math import pi
from std.testing import assert_equal, assert_true

from qumo.compiler.pass_manager import transpile
from qumo.compiler.program_compile import compact_qubits, compile_program
from qumo.compiler.target import Target, parse_target
from qumo.ir.gates import Op
from qumo.ir.qasm import parse_qasm, parse_qasm_program
from qumo.sim.program_runner import sample_program

comptime HEAD = "OPENQASM 3.0; qubit[2] q; bit[2] c; int k = 0; while (k < 2) { k += 1; "


def _compile_all(body: String) raises:
    """Every level on targets that keep, translate and resynthesise the gate."""
    var src = HEAD + body + " }"
    var p = parse_qasm_program(src)
    var specs: List[String] = ["line:2", "line:3/rz,sx,x,cz", "line:3/rz,sx,x,cx", "line:3/u,cx", "line:3/rz,rx,ry,rxx", "line:3/rz,ry,rzz"]
    for spec in specs:
        var t = parse_target(spec)
        for level in range(1, 4):
            try:
                _ = compile_program(p, t, level, 7)
            except e:
                raise Error(body + " on " + spec + " level " + String(level) + ": " + String(e))


def test_1_small_angles_sweep() raises:
    var gates: List[String] = ["rx(%) q[0];", "rz(%) q[0];", "p(%) q[0];", "crz(%) q[0], q[1];", "cp(%) q[0], q[1];", "rzz(%) q[0], q[1];"]
    var exponents: List[Int] = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    var mantissas: List[String] = ["1", "2.5", "5"]
    for g in gates:
        for e in exponents:
            for m in mantissas:
                var theta = m + "e-" + String(e)
                _compile_all("h q[0]; " + g.replace("%", theta))
                _compile_all(g.replace("%", "-" + theta) + " cx q[0], q[1];")


def test_1_powers_of_two_of_pi() raises:
    var gates: List[String] = ["rx(%) q[0];", "rz(%) q[0];", "p(%) q[0];", "crz(%) q[0], q[1];", "cp(%) q[0], q[1];", "rzz(%) q[0], q[1];"]
    for g in gates:
        for k in range(1, 44, 3):
            _compile_all("h q[0]; " + g.replace("%", "pi / 2 ** " + String(k)))


def test_1_near_named_gates() raises:
    var rows: List[String] = [
        "rz(pi / 2 + 1e-11) q[0];", "rz(pi / 2 - 3e-12) q[0];", "rx(pi + 1e-12) q[0];", "p(pi / 4 - 3e-11) q[0];",
        "rx(2 * pi + 1e-11) q[0];", "rz(4 * pi - 1e-11) q[0];", "cp(pi + 2e-11) q[0], q[1];", "crz(4 * pi + 1e-12) q[0], q[1];",
        "rzz(2 * pi - 1e-11) q[0], q[1];", "ry(pi / 2 + 1e-10) q[0];", "U(pi / 2 + 1e-11, 1e-11, pi - 1e-11) q[0];",
    ]
    for r in rows:
        _compile_all("h q[1]; " + r + " cx q[0], q[1];")


def _chain(n: Int, one: String) raises:
    var body = String()
    for _ in range(n):
        body += one
    var p = parse_qasm_program(HEAD + body + " }")
    for level in range(1, 4):
        _ = compile_program(p, parse_target("line:3/rz,sx,x,cz"), level, 7)
        _ = compile_program(p, Target.line(2), level, 7)


def test_1_small_errors_add_up() raises:
    _chain(20000, "rx(1e-10) q[0]; cx q[0], q[1]; ")
    _chain(20000, "rz(1e-11) q[1]; cx q[0], q[1]; ")
    _chain(20000, "rz(0.3) q[0]; rz(-0.3 + 1e-11) q[0]; cx q[0], q[1]; ")
    _chain(20000, "rz(pi / 2 + 1e-11) q[0]; cx q[0], q[1]; rz(-pi / 2) q[0]; cx q[0], q[1]; ")
    _chain(5000, "rzz(1e-10) q[0], q[1]; h q[0]; cx q[0], q[1]; rx(1e-10) q[1]; cx q[0], q[1]; h q[0]; ")


def test_1_small_angles_still_act() raises:
    # the angle reaches the simulator: 2000 passes of 1e-3 turn the qubit by 2 rad
    var src = String("OPENQASM 3.0; qubit[1] q; bit[1] c; int k = 0; while (k < 2000) { k += 1; rx(1e-3) q[0]; } c[0] = measure q[0];")
    var p = parse_qasm_program(src)
    var r = compile_program(p, Target.line(2), 2, 7)
    var a = sample_program(p, 4000, 3).sample.probability("1")
    var b = sample_program(compact_qubits(r.program), 4000, 3).sample.probability("1")
    assert_true(abs(a - b) < 0.06)
    assert_true(abs(a - 0.7081) < 0.04)


def test_1_a_flat_compile_still_drops_what_it_did() raises:
    # documented: outside a program, a rotation below about 2e-6 goes
    var c = parse_qasm("OPENQASM 3.0; qubit[1] q; rx(1e-7) q[0]; h q[0];")
    assert_equal(transpile(c, Target.all_to_all(1), 2, 7).circuit.count(Op.RX), 0)
    # and it stays in a block, where the check would see it go
    var p = parse_qasm_program(HEAD + "rx(1e-7) q[0]; }")
    assert_equal(compile_program(p, Target.line(2), 2, 7).program.body.count(Op.RX) + compile_program(p, Target.line(2), 2, 7).program.body.count(Op.U), 1)


def main() raises:
    test_1_small_angles_sweep()
    test_1_powers_of_two_of_pi()
    test_1_near_named_gates()
    test_1_small_errors_add_up()
    test_1_small_angles_still_act()
    test_1_a_flat_compile_still_drops_what_it_did()
    print("test_program_compile_review3 OK")
