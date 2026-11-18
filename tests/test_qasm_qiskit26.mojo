"""Regression tests from the Qiskit 2.6.x OpenQASM bugs (agents/reports watch-ir proposal 1):
pow on ctrl/negctrl gates, domain errors in qasm2 expressions, qasm3 export with
array-valued parameters, and identifiers that shadow the constants pi, tau, euler."""

from std.testing import assert_equal, assert_true, assert_almost_equal

from qumo.ir.circuit import Circuit
from qumo.ir.param import Param
from qumo.ir.qasm import parse_qasm, to_qasm
from qumo.sim.runner import evolve


def head(n: Int) -> String:
    var s = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[" + String(n) + "] q;\n"
    for q in range(n):
        s += "ry(" + String(0.3 + 0.41 * Float64(q)) + ") q[" + String(q) + "];\n"
        s += "rz(" + String(0.7 - 0.23 * Float64(q)) + ") q[" + String(q) + "];\n"
    for q in range(n - 1):
        s += "cx q[" + String(q) + "], q[" + String(q + 1) + "];\n"
    return s^


def same_state(a: String, b: String, n: Int) raises:
    """Exact statevector equality (global phase included)."""
    var sa = evolve(parse_qasm(head(n) + a))
    var sb = evolve(parse_qasm(head(n) + b))
    for i in range(sa.size()):
        if abs(sa.re[i] - sb.re[i]) > 1e-10 or abs(sa.im[i] - sb.im[i]) > 1e-10:
            raise Error("states differ for:\n" + a + "\nvs\n" + b)


def expect_error(src: String, code: String) raises:
    try:
        _ = parse_qasm(src)
    except e:
        if not String(e).startswith(code):
            raise Error("expected " + code + ", got:\n" + String(e))
        return
    raise Error("expected " + code + " for:\n" + src)


def test_pow_on_controlled() raises:
    # pow(k) @ ctrl @ g == ctrl @ pow(k) @ g == ctrl(g^k), phase included
    same_state("pow(2) @ ctrl @ s q[0], q[1];", "cz q[0], q[1];", 2)
    same_state("ctrl @ pow(2) @ s q[0], q[1];", "cz q[0], q[1];", 2)
    same_state("pow(-1) @ ctrl @ s q[0], q[1];", "ctrl @ sdg q[0], q[1];", 2)
    same_state("pow(3) @ ctrl @ t q[0], q[1];", "ctrl @ pow(3) @ t q[0], q[1];", 2)
    same_state("pow(0) @ ctrl @ x q[0], q[1];", "", 2)
    same_state("pow(2) @ ctrl @ x q[0], q[1];", "", 2)
    same_state("pow(3) @ ctrl @ x q[0], q[1];", "cx q[0], q[1];", 2)
    # negctrl: the control fires on |0>
    same_state("pow(2) @ negctrl @ s q[0], q[1];", "x q[0]; cz q[0], q[1]; x q[0];", 2)
    same_state("pow(-1) @ negctrl @ s q[0], q[1];", "negctrl @ sdg q[0], q[1];", 2)
    same_state("pow(2) @ negctrl @ x q[0], q[1];", "", 2)
    # two controls, mixed
    same_state("pow(2) @ ctrl @ negctrl @ s q[0], q[1], q[2];", "x q[1]; ccz q[0], q[1], q[2]; x q[1];", 3)
    same_state("pow(2) @ ctrl(2) @ s q[0], q[1], q[2];", "ccz q[0], q[1], q[2];", 3)
    # a gate with a phase: p(a) has no global phase, rz does
    same_state("pow(2) @ ctrl @ rz(0.7) q[0], q[1];", "ctrl @ rz(1.4) q[0], q[1];", 2)
    same_state("pow(2) @ negctrl @ rz(0.7) q[0], q[1];", "negctrl @ rz(1.4) q[0], q[1];", 2)
    # inv and pow compose
    same_state("inv @ pow(2) @ ctrl @ s q[0], q[1];", "ctrl @ sdg q[0], q[1]; ctrl @ sdg q[0], q[1];", 2)
    # a user gate under pow + ctrl
    same_state("gate g a { s a; } pow(2) @ ctrl @ g q[0], q[1];", "cz q[0], q[1];", 2)
    print("  pow on ctrl / negctrl")


def test_pow_errors_on_controlled() raises:
    expect_error("include \"stdgates.inc\"; qubit[2] q; pow(0.5) @ ctrl @ x q[0], q[1];", "QM-E550")
    expect_error("include \"stdgates.inc\"; qubit[2] q; pow(2) @ ctrl @ x q[0];", "QM-E512")
    print("  pow(non-integer) @ ctrl rejected, operand count")


def test_qasm2_domain_errors() raises:
    var pre = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\n"
    expect_error(pre + "rz(sqrt(-1)) q[0];", "QM-E533")
    expect_error(pre + "rz(ln(0)) q[0];", "QM-E533")
    expect_error(pre + "rz(ln(-2)) q[0];", "QM-E533")
    expect_error(pre + "rz(1/0) q[0];", "QM-E533")
    # Qiskit 2.6 turned these into a QASM2ParseError instead of a nan / crash
    expect_error(pre + "rz(asin(2)) q[0];", "QM-E533")
    expect_error(pre + "rz(acos(-3)) q[0];", "QM-E533")
    expect_error(pre + "rz((-1)^0.5) q[0];", "QM-E533")
    expect_error(pre + "rz(0^(-1)) q[0];", "QM-E533")
    expect_error(pre + "rz(tan(pi/2) * 1e400) q[0];", "QM-E533")
    # in-domain values still parse
    var c = parse_qasm(pre + "rz(asin(1)) q[0]; rz(acos(-1)) q[0]; rz(sqrt(4)) q[0]; rz(ln(1)) q[0];")
    assert_equal(len(c.ops), 4, "in-domain calls parse")
    # `^` is the OpenQASM 2 power operator (right associative); QUMO used to reject it
    var cp = parse_qasm(pre + "rz(2^3) q[0]; rz(2^3^2) q[0]; rz(-2^2) q[0];")
    assert_almost_equal(cp.ops[0].p0.offset, 8.0)
    assert_almost_equal(cp.ops[1].p0.offset, 512.0)
    assert_almost_equal(cp.ops[2].p0.offset, -4.0)
    # also where the expression takes the typed path (a `const` in an OpenQASM 2 file)
    var ct = parse_qasm(pre + "const int n = 3;\nrz(2^n) q[0];")
    assert_almost_equal(ct.ops[0].p0.offset, 8.0)
    # in OpenQASM 3 `^` stays XOR in typed expressions and is not a power in gate arguments
    expect_error("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[1] q;\nrz(2^3) q[0];", "QM-E500")
    print("  qasm2 expression domain errors")


def test_array_parameters() raises:
    # array-valued parameters are not gate angles; QUMO must refuse or round-trip them,
    # never emit text it cannot read back
    var src = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\ninput array[float[64], 2] ang;\nqubit[1] q;\nrx(ang[0]) q[0];\nrz(ang[1]) q[0];\n"
    var parsed = True
    var c = Circuit(1)
    try:
        c = parse_qasm(src)
    except e:
        parsed = False
        assert_true(String(e).startswith("QM-E"), "array input rejected with a catalogue code: " + String(e))
    if parsed:
        var out = to_qasm(c)
        var again = parse_qasm(out)  # the export must read back
        assert_equal(len(again.ops), len(c.ops), "array parameters round trip")
        assert_equal(again.num_params(), c.num_params(), "same parameter count")
    print("  array-valued parameters: parse or refuse, export reads back")


def test_constant_names() raises:
    # Qiskit 2.6: pi, tau and euler are reserved. Shadowing must be an error or
    # keep the constant; never silently give the program another value.
    for nm in ["pi", "tau", "euler"]:
        var decls = List[String]()
        decls.append("float " + nm + " = 1.0;")
        decls.append("const float " + nm + " = 1.0;")
        decls.append("input float " + nm + ";")
        decls.append("qubit[1] " + nm + ";")
        decls.append("gate " + nm + " a { x a; }")
        for d in decls:
            var src = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n" + d + "\nqubit[1] q;\nrx(" + nm + ") q[0];\n"
            var ok = True
            try:
                _ = parse_qasm(src)
            except e:
                ok = False
                assert_true(String(e).startswith("QM-E522"), "QM-E522 for '" + d + "': " + String(e))
            # a name equal to a constant used to be accepted and then ignored
            assert_true(not ok, "'" + d + "' must be rejected")
        # the same holds where the name is a gate parameter or a loop variable
        expect_error("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[1] q;\ngate g(" + nm + ") a { rz(" + nm + ") a; }\ng(0.5) q[0];\n", "QM-E522")
        expect_error("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ngate g(" + nm + ") a { rz(" + nm + ") a; }\ng(0.5) q[0];\n", "QM-E522")
        expect_error("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[1] q;\nfor int " + nm + " in [0:1] { rz(" + nm + ") q[0]; }\n", "QM-E522")
    # a qubit argument may carry such a name: it is never read as a value
    var cq = parse_qasm("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[1] q;\ngate g pi { rz(0.25) pi; }\ng q[0];\n")
    assert_equal(len(cq.ops), 1, "qubit argument named pi")
    # a register named like a constant must survive to_qasm export of its own circuit
    var c2 = Circuit(2)
    c2.rx(Param(3.141592653589793), 0)
    var rt = parse_qasm(to_qasm(c2))
    assert_equal(len(rt.ops), 1, "pi angle round trip")
    assert_almost_equal(rt.ops[0].p0.offset, 3.141592653589793, atol=1e-15)
    print("  identifiers named pi / tau / euler")


def main() raises:
    test_pow_on_controlled()
    test_pow_errors_on_controlled()
    test_qasm2_domain_errors()
    test_array_parameters()
    test_constant_names()
    print("test_qasm_qiskit26 OK")
