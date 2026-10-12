"""Regression tests for the OpenQASM 2 exporter (qumo.ir.qasm2).

Contract under test (see the qumo.ir.qasm2 docstring):
  * gates used without a definition belong to the original qelib1.inc;
  * every other gate is defined before its first use, once, and its body only
    calls qelib1 gates or earlier definitions (an earlier exporter defined
    `ecr` through an undeclared `rzx`, which Qiskit, pytket and Cirq reject);
  * definition bodies are exact unitaries, including global phase. This is
    checked by renaming each defined gate (so QUMO's parser inlines the body
    instead of using its native gate) and comparing amplitudes.
The cross-check against Qiskit (strict and legacy qasm2), pytket and Cirq runs
outside Mojo; see docs/AI_NATIVE.md.
"""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.math import pi

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_name, op_arity
from qumo.ir.qasm2 import to_qasm2, qasm2_real, qelib1_gates, qasm2_gate_definition, qasm2_gate_name
from qumo.ir.qasm import parse_qasm
from qumo.sim.runner import evolve
from qumo.sim.statevector import StateVector


# ---------------------------------------------------------------------------
# A tiny OpenQASM 2 scanner (enough for the exporter's own output)
# ---------------------------------------------------------------------------
def _contains(xs: List[String], s: String) -> Bool:
    for x in xs:
        if x == s:
            return True
    return False


def _is_ident(c: Int) -> Bool:
    return (c >= ord("a") and c <= ord("z")) or (c >= ord("A") and c <= ord("Z")) or (c >= ord("0") and c <= ord("9")) or c == ord("_")


def _lead_ident(stmt: String) -> String:
    """Leading identifier of a statement, after an optional `if(...)` prefix."""
    var s = String(stmt.strip())
    if s.startswith("if("):
        var close = s.find(")")
        s = String(String(s[byte=close + 1:]).strip())
    var bs = s.as_bytes()
    var e = 0
    while e < len(bs) and _is_ident(Int(bs[e])):
        e += 1
    return String(s[byte=0:e])


def _body_statements(line: String) -> List[String]:
    """Statements inside `gate ... { a; b; }`."""
    var lb = line.find("{")
    var rb = line.find("}")
    var out = List[String]()
    for part in String(line[byte=lb + 1:rb]).split(";"):
        var t = String(part.strip())
        if t.byte_length() > 0:
            out.append(t^)
    return out^


def check_contract(text: String) raises -> List[String]:
    """Assert the export contract; return the names defined in `text`."""
    var qelib = qelib1_gates()
    var defined = List[String]()
    var used_defs = List[String]()
    var lines = text.split("\n")
    assert_equal(String(lines[0]), "OPENQASM 2.0;")
    assert_equal(String(lines[1]), "include \"qelib1.inc\";")
    for i in range(2, len(lines)):
        var line = String(lines[i].strip())
        if line.byte_length() == 0 or line.startswith("//"):
            continue
        if line.startswith("gate "):
            var name = _lead_ident(String(line[byte=5:]))
            assert_true(not _contains(qelib, name), "qelib1 gate '" + name + "' redefined")
            assert_true(not _contains(defined, name), "gate '" + name + "' defined twice")
            for st in _body_statements(line):
                var g = _lead_ident(st)
                assert_true(_contains(qelib, g) or _contains(defined, g),
                            "definition of '" + name + "' calls undeclared gate '" + g + "'")
            defined.append(name)
            continue
        var head = _lead_ident(line)
        if head == "qreg" or head == "creg" or head == "measure" or head == "reset" or head == "barrier":
            continue
        assert_true(line.endswith(";"), "statement without ';': " + line)
        assert_true(_contains(qelib, head) or _contains(defined, head), "gate '" + head + "' used without a definition: " + line)
        if _contains(defined, head) and not _contains(used_defs, head):
            used_defs.append(head)
    for d in defined:
        assert_true(_contains(used_defs, d), "gate '" + d + "' is defined but never used")
    return defined^


def rename_defined(text: String, defined: List[String]) -> String:
    """Rename every defined gate NAME to NAME_def (header and uses), so QUMO's
    parser inlines the exported body instead of using its native gate."""
    var out = String()
    for raw in text.split("\n"):
        var line = String(raw)
        if line.startswith("gate "):
            var name = _lead_ident(String(line[byte=5:]))
            out += "gate " + name + "_def" + String(line[byte=5 + name.byte_length():]) + "\n"
            continue
        var head = _lead_ident(line)
        if head.byte_length() > 0 and _contains(defined, head):
            var at = line.find(head)
            out += String(line[byte=0:at]) + head + "_def" + String(line[byte=at + head.byte_length():]) + "\n"
            continue
        out += line + "\n"
    return out^


def assert_same_state(a: StateVector, b: StateVector, what: String) raises:
    """Amplitude-wise equality (global phase included)."""
    assert_equal(len(a.re), len(b.re))
    for i in range(len(a.re)):
        assert_true(abs(a.re[i] - b.re[i]) < 1e-12 and abs(a.im[i] - b.im[i]) < 1e-12,
                    what + ": amplitude " + String(i) + " differs: " + String(a.re[i]) + "+" + String(a.im[i]) + "i vs " + String(b.re[i]) + "+" + String(b.im[i]) + "i")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_every_gate_contract_and_round_trip() raises:
    var c = Circuit(3)
    for k in range(Op.CCZ + 1):
        c.add(k, 0, 1, 2, 0.3, -0.7, 1.1)
        c.add(k, 2, 0, 1, -0.2, 0.4, 0.9)
    var text = to_qasm2(c)
    var defined = check_contract(text)
    # every gate outside qelib1 got a definition
    for k in range(Op.CCZ + 1):
        if qasm2_gate_definition(k).byte_length() > 0:
            assert_true(_contains(defined, op_name(k)), "missing definition for " + op_name(k))
        else:
            assert_true(_contains(qelib1_gates(), qasm2_gate_name(k)), op_name(k) + " is neither qelib1 nor defined")
    # QUMO reads its own output back (native gates)
    assert_same_state(evolve(c), evolve(parse_qasm(text)), "native round trip")
    # ... and the exported definitions are exact (bodies inlined)
    assert_same_state(evolve(c), evolve(parse_qasm(rename_defined(text, defined))), "inlined definitions")


def test_definitions_exact_per_gate() raises:
    """Each definition equals QUMO's matrix on every basis input (exact phase)."""
    var angles: List[Float64] = [0.37, -2.1, 1.9]
    for k in range(Op.CCZ + 1):
        if qasm2_gate_definition(k).byte_length() == 0:
            continue
        var a = op_arity(k)
        var n = a
        for b in range(1 << n):
            for rev in range(2):
                var c = Circuit(n)
                for q in range(n):
                    if (b >> q) & 1 == 1:
                        c.x(q)
                c.h(0)
                c.rz(0.25, n - 1)  # non-trivial phases on the input
                var q0 = n - 1 if rev == 1 else 0
                var q1 = (0 if rev == 1 else 1) if a > 1 else -1
                var q2 = 1 if a > 2 else -1
                if a > 2 and rev == 1:
                    q2 = n - 2
                c.add(k, q0, q1, q2, angles[0], angles[1], angles[2])
                var text = to_qasm2(c)
                var defined = check_contract(text)
                var inl = parse_qasm(rename_defined(text, defined))
                assert_same_state(evolve(c), evolve(inl), op_name(k) + " on basis input " + String(b))


def test_angles() raises:
    assert_equal(qasm2_real(pi / 2), "pi/2")
    assert_equal(qasm2_real(-3 * pi / 4), "-3*pi/4")
    assert_equal(qasm2_real(0.3), "0.3")
    assert_equal(qasm2_real(2.0), "2")
    assert_equal(qasm2_real(0.0), "0")
    var vals: List[Float64] = [1e-7, -2.5e-12, 1e20, 123456789.0]
    for v in vals:
        var s = qasm2_real(v)
        # OpenQASM 2 `real` needs a '.' before an exponent
        if s.find("e") >= 0:
            assert_true(s.find(".") >= 0 and s.find(".") < s.find("e"), s)
        assert_almost_equal(Float64(s), v)
    var c = Circuit(1)
    c.rz(1e-7, 0)
    var back = parse_qasm(to_qasm2(c))
    assert_equal(back.ops[0].p0.value(), 1e-7)


def test_non_unitary_and_errors() raises:
    var c = Circuit(2, 1)
    c.h(0)
    c.add(Op.DEPOLARIZE1, 0, p0=0.01)
    c.barrier()
    c.measure(0, 0)
    c.x(1)
    c.c_if(0, 1)
    c.reset(0)
    c.global_phase = 0.5
    var text = to_qasm2(c)
    _ = check_contract(text)
    assert_true(text.find("// noise channel not exported: depolarize1(0.01) q[0]\n") >= 0, text)
    assert_true(text.find("barrier q;\nmeasure q[0] -> c[0];\nif(c==1) x q[1];\nreset q[0];\n") >= 0, text)
    assert_true(text.find("// global phase 0.5") >= 0, text)
    # unbound parameters -> QM-E102
    var p = Circuit(1)
    var t = p.parameter("t")
    p.rx(t, 0)
    var code = String()
    try:
        _ = to_qasm2(p)
    except e:
        code = String(e)
    assert_true(code.startswith("QM-E102:") and code.find("Hint:") > 0, code)
    # single-bit condition on a multi-bit register -> QM-E552
    var d = Circuit(2, 2)
    d.measure(0, 1)
    d.x(1)
    d.c_if(1, 1)
    code = String()
    try:
        _ = to_qasm2(d)
    except e:
        code = String(e)
    assert_true(code.startswith("QM-E552:") and code.find("Hint:") > 0, code)


def main() raises:
    test_every_gate_contract_and_round_trip()
    test_definitions_exact_per_gate()
    test_angles()
    test_non_unitary_and_errors()
    print("test_qasm2_export OK")
