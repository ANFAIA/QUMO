"""Tests for qumo.ir.qasm: OpenQASM 2/3 parsing, emission, round trips, errors."""

from std.math import pi, sqrt, cos, sin
from std.time import perf_counter_ns
from std.testing import assert_equal, assert_true, assert_false, assert_almost_equal

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_name, op_num_params, op_arity
from qumo.ir.param import Param
from qumo.ir.qasm import parse_qasm, parse_qasm_with_warnings, to_qasm, format_angle, tokenize_qasm
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, sample


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sv_max_diff(a: StateVector, b: StateVector) -> Float64:
    if a.num_qubits != b.num_qubits:
        return 1e9
    var m = 0.0
    for i in range(a.size()):
        m = max(m, abs(a.re[i] - b.re[i]))
        m = max(m, abs(a.im[i] - b.im[i]))
    return m


def assert_same_state(a: Circuit, b: Circuit, tol: Float64 = 1e-10, params: List[Float64] = List[Float64]()) raises:
    var sa = evolve(a, params)
    var sb = evolve(b, params)
    var d = sv_max_diff(sa, sb)
    if d > tol:
        print("--- A ---")
        print(a)
        print("--- B ---")
        print(b)
        raise Error("statevectors differ by " + String(d))


def assert_same_up_to_phase(a: Circuit, b: Circuit) raises:
    var sa = evolve(a)
    var sb = evolve(b)
    assert_almost_equal(sa.fidelity(sb), 1.0, atol=1e-10)


def prep(n: Int) -> String:
    """QASM snippet preparing a generic entangled state on q[0..n-1]."""
    var s = String()
    for q in range(n):
        s += "ry(" + String(0.3 + 0.41 * Float64(q)) + ") q[" + String(q) + "];\n"
        s += "rz(" + String(0.7 - 0.23 * Float64(q)) + ") q[" + String(q) + "];\n"
    for q in range(n - 1):
        s += "cx q[" + String(q) + "], q[" + String(q + 1) + "];\n"
    for q in range(n):
        s += "rx(" + String(0.11 * Float64(q + 1)) + ") q[" + String(q) + "];\n"
    return s^


def head(n: Int) -> String:
    return "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[" + String(n) + "] q;\n" + prep(n)


def expect_error(src: String, code: String, loc: String = "") raises:
    try:
        _ = parse_qasm(src)
    except e:
        var s = String(e)
        if not s.startswith(code):
            raise Error("expected " + code + " got:\n" + s)
        if loc.byte_length() > 0 and s.find(loc) < 0:
            raise Error("expected location '" + loc + "' in:\n" + s)
        if s.find("Hint: ") < 0:
            raise Error("missing hint in:\n" + s)
        return
    raise Error("expected error " + code + " for:\n" + src)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_bell_qasm2_qasm3() raises:
    var q3 = parse_qasm("""
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
""")
    var q2 = parse_qasm("""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
CX q[0],q[1];
measure q -> c;
""")
    assert_equal(q3.num_qubits, 2)
    assert_equal(q3.num_clbits, 2)
    assert_equal(q3.num_measurements, 2)
    assert_equal(q2.num_measurements, 2)
    assert_same_state(q3, q2)
    var sv = evolve(q3)
    assert_almost_equal(sv.re[0], 1 / sqrt(2.0))
    assert_almost_equal(sv.re[3], 1 / sqrt(2.0))
    var r = sample(q2, 1000, seed=3)
    assert_equal(r.counts.get("01", 0) + r.counts.get("10", 0), 0)
    # c[i] = measure q[j] form and measure -> form agree on clbits
    var m = parse_qasm("qubit[2] q; bit[2] c; x q[1]; c[0] = measure q[1]; measure q[0] -> c[1];")
    var rm = sample(m, 10, seed=1)
    assert_equal(rm.counts.get("01", 0), 10)


def test_registers_and_broadcast() raises:
    var c = parse_qasm("""
OPENQASM 2.0;
include "qelib1.inc";
qreg a[2];
qreg b[3];
creg ca[2];
creg cb[3];
h a;            // broadcast
cx a[1], b[0];  /* flattened: 1 -> 2 */
cx a[0], b;     // single control, register target
x b[0:1];
measure a -> ca;
measure b -> cb;
""")
    assert_equal(c.num_qubits, 5)
    assert_equal(c.num_clbits, 5)
    assert_equal(c.count(Op.H), 2)
    assert_equal(c.count(Op.CX), 4)
    assert_equal(c.ops[2].q0, 1)
    assert_equal(c.ops[2].q1, 2)
    assert_equal(c.ops[3].q0, 0)
    assert_equal(c.ops[5].q1, 4)
    assert_equal(c.count(Op.X), 2)
    assert_equal(c.ops[len(c.ops) - 1].clbit, 4)
    # negative index and index sets
    var d = parse_qasm("qubit[4] q; x q[-1]; h q[{0, 2}];")
    assert_equal(d.ops[0].q0, 3)
    assert_equal(d.ops[2].q0, 2)
    # hardware qubits
    var hw = parse_qasm("OPENQASM 3; h $0; cx $0, $2;")
    assert_equal(hw.num_qubits, 3)


def test_aliases_and_params() raises:
    var src = head(3) + """
cnot q[0], q[1];
toffoli q[0], q[1], q[2];
phase(pi/4) q[0];
cphase(-pi/8) q[0], q[2];
cu1(0.25) q[1], q[2];
u1(τ/8) q[1];
u3(0.1, 0.2, 0.3) q[0];
u2(0.4, 0.5) q[1];
U(1, 2, 3) q[2];
id q[0];
i q[1];
sx q[2];
sxdg q[2];
rzz(2*pi/3) q[0], q[2];
fredkin q[2], q[0], q[1];
cu3(0.3, 0.2, 0.1) q[0], q[1];
csx q[1], q[2];
cu(0.3, 0.2, 0.1, 0.4) q[2], q[0];
rx(π) q[0];
ry(euler) q[1];
rz(sin(0.5) + cos(pi) * sqrt(2) - 3 ** 2 / 4 + (1 - 2)) q[2];
"""
    var c = parse_qasm(src)
    var want = parse_qasm(head(3))
    want.cx(0, 1)
    want.ccx(0, 1, 2)
    want.p(pi / 4, 0)
    want.cp(-pi / 8, 0, 2)
    want.cp(0.25, 1, 2)
    want.p(2 * pi / 8, 1)
    want.u(0.1, 0.2, 0.3, 0)
    want.u(pi / 2, 0.4, 0.5, 1)
    want.u(1.0, 2.0, 3.0, 2)
    want.id(0)
    want.id(1)
    want.sx(2)
    want.sxdg(2)
    want.rzz(2 * pi / 3, 0, 2)
    want.cswap(2, 0, 1)
    var cu3 = Circuit(3)
    cu3.u(0.3, 0.2, 0.1, 1)
    want.compose(cu3.controlled(0))
    var csx = Circuit(3)
    csx.sx(2)
    want.compose(csx.controlled(1))
    want.p(0.4, 2)
    var cu = Circuit(3)
    cu.u(0.3, 0.2, 0.1, 0)
    want.compose(cu.controlled(2))
    want.rx(pi, 0)
    want.ry(2.718281828459045, 1)
    want.rz(sin(0.5) + cos(pi) * sqrt(2.0) - 9.0 / 4.0 - 1.0, 2)
    assert_same_state(c, want)


def test_input_params() raises:
    var src = """
OPENQASM 3.0;
include "stdgates.inc";
input float theta;
input angle[32] phi;
const float half = 0.5;
qubit[2] q;
rx(theta) q[0];
ry(2*theta + pi/2) q[1];
rz(-theta/2 - half) q[0];
cp(phi * 3) q[0], q[1];
rzz(-(phi - 1)) q[0], q[1];
"""
    var c = parse_qasm(src)
    assert_equal(c.num_params(), 2)
    assert_equal(c.param_names[0], "theta")
    assert_equal(c.param_names[1], "phi")
    assert_true(c.ops[1].p0 == Param(pi / 2, 0, 2.0))
    assert_true(c.ops[2].p0 == Param(-0.5, 0, -0.5))
    assert_true(c.ops[3].p0 == Param(0.0, 1, 3.0))
    assert_true(c.ops[4].p0 == Param(1.0, 1, -1.0))
    var vals: List[Float64] = [0.37, -1.21]
    var want = Circuit(2)
    want.rx(0.37, 0)
    want.ry(2 * 0.37 + pi / 2, 1)
    want.rz(-0.37 / 2 - 0.5, 0)
    want.cp(-1.21 * 3, 0, 1)
    want.rzz(-(-1.21 - 1), 0, 1)
    var bound = c.bind(vals)
    assert_same_state(bound, want, 1e-12)
    # symbolic params survive emission
    var text = to_qasm(c)
    assert_true(text.find("input float[64] theta;") >= 0)
    assert_true(text.find("ry(2*theta + pi/2) q[1];") >= 0, text)
    var back = parse_qasm(text)
    assert_same_state(back, c, 0.0, vals)


def test_gate_defs_and_modifiers() raises:
    var src = head(3) + """
gate rot(a, b) t { rz(a) t; ry(b/2) t; }
gate ent(th) x, y { h x; cx x, y; rot(th, 2*th) y; }
ent(0.3) q[0], q[2];
inv @ ent(0.3) q[1], q[0];
ctrl @ x q[0], q[1];
ctrl @ ctrl @ x q[0], q[1], q[2];
ctrl(2) @ x q[2], q[1], q[0];
inv @ s q[1];
pow(2) @ s q[2];
pow(-1) @ t q[2];
negctrl @ x q[0], q[1];
ctrl @ rz(0.4) q[2], q[0];
ctrl @ rot(0.2, 0.6) q[1], q[2];
inv @ ctrl @ U(0.5, 0.1, -0.2) q[0], q[2];
ctrl @ gphase(0.7) q[1];
"""
    var c = parse_qasm(src)
    var want = parse_qasm(head(3))
    # ent(0.3) q0,q2
    var ent = Circuit(3)
    ent.h(0)
    ent.cx(0, 2)
    ent.rz(0.3, 2)
    ent.ry(0.3, 2)
    want.compose(ent)
    var ent2 = Circuit(3)
    ent2.h(1)
    ent2.cx(1, 0)
    ent2.rz(0.3, 0)
    ent2.ry(0.3, 0)
    want.compose(ent2.inverse())
    want.cx(0, 1)
    want.ccx(0, 1, 2)
    want.ccx(2, 1, 0)
    want.sdg(1)
    want.z(2)
    want.tdg(2)
    want.x(0)
    want.cx(0, 1)
    want.x(0)
    want.crz(0.4, 2, 0)
    var rot = Circuit(3)
    rot.rz(0.2, 2)
    rot.ry(0.3, 2)
    want.compose(rot.controlled(1))
    var cu = Circuit(3)
    cu.u(0.5, 0.1, -0.2, 2)
    want.compose(cu.controlled(0).inverse())
    want.p(0.7, 1)
    assert_same_state(c, want, 1e-10)
    # definition of a native gate is a declaration (exact native semantics)
    var d = parse_qasm("qubit[2] q; gate rzz(t) a, b { cx a, b; rz(t) b; cx a, b; } rzz(0.3) q[0], q[1];")
    assert_equal(len(d.ops), 1)
    assert_equal(d.ops[0].kind, Op.RZZ)
    # user gate with symbolic parameter
    var e = parse_qasm("input float t; qubit[1] q; gate g(a) x { rx(a/2) x; rz(-a) x; } g(2*t + 1) q[0];")
    assert_true(e.ops[0].p0 == Param(0.5, 0, 1.0))
    assert_true(e.ops[1].p0 == Param(-1.0, 0, -2.0))


def test_for_loops_and_consts() raises:
    var c = parse_qasm("""
OPENQASM 3.0;
include "stdgates.inc";
const int n = 5;
qubit[n] q;
bit[n] c;
h q[0];
for uint i in [0:n-2] {
    cx q[i], q[i+1];
}
for i in {4, 2} x q[i];
for int k in [3:-1:1] { rz(k * pi / 8) q[k]; }
for uint j in [1:0] { h q[0]; }  // empty
c = measure q;
""")
    assert_equal(c.num_qubits, 5)
    assert_equal(c.count(Op.CX), 4)
    assert_equal(c.ops[4].q0, 3)
    assert_equal(c.ops[4].q1, 4)
    assert_equal(c.count(Op.X), 2)
    assert_equal(c.ops[5].q0, 4)
    assert_equal(c.count(Op.RZ), 3)
    assert_equal(c.ops[7].q0, 3)
    assert_equal(c.ops[7].p0.value(), 3 * pi / 8)
    assert_equal(c.count(Op.H), 1)


def test_feed_forward() raises:
    var tele3 = parse_qasm("""
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
ry(1.1) q[0];
h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];
c[0] = measure q[0];
c[1] = measure q[1];
if (c[1] == 1) x q[2];
if (c[0]) { z q[2]; }
c[2] = measure q[2];
""")
    var tele2 = parse_qasm("""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c0[1];
creg c1[1];
creg c2[1];
ry(1.1) q[0];
h q[1];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
measure q[0] -> c0[0];
measure q[1] -> c1[0];
if(c1==1) x q[2];
if(c0==1) z q[2];
measure q[2] -> c2[0];
""")
    assert_true(tele3.is_dynamic())
    assert_equal(tele3.ops[7].cond, 1)
    assert_equal(tele3.ops[8].cond, 0)
    assert_equal(tele2.ops[7].cond, 1)
    var shots = 4000
    var r3 = sample(tele3, shots, seed=11)
    var r2 = sample(tele2, shots, seed=11)
    # P(q2 = 1) = sin^2(1.1/2)
    var p1 = sin(0.55) * sin(0.55)
    var n3 = 0
    var n2 = 0
    for kv in r3.counts.items():
        if Int(kv.key.as_bytes()[0]) == ord("1"):
            n3 += kv.value
    for kv in r2.counts.items():
        if Int(kv.key.as_bytes()[0]) == ord("1"):
            n2 += kv.value
    assert_true(abs(Float64(n3) / Float64(shots) - p1) < 0.03)
    assert_equal(n2, n3)  # same seed, same trajectories
    # else branch and negation
    var e = parse_qasm("qubit[2] q; bit[1] c; c[0] = measure q[0]; if (!c[0]) { x q[1]; } else { h q[1]; }")
    assert_equal(e.ops[1].cond_val, 0)
    assert_equal(e.ops[2].cond_val, 1)
    var t = to_qasm(tele3)
    assert_true(t.find("if (c[1] == 1) x q[2];") >= 0)
    assert_equal(len(parse_qasm(t).ops), len(tele3.ops))


def test_pragmas_roundtrip() raises:
    var c = Circuit(3)
    c.h(0)
    c.depolarize1(0.01, 0)
    c.cx(0, 1)
    c.depolarize2(0.02, 0, 1)
    c.x_error(0.1, 2)
    c.pauli_channel_1(0.01, 0.02, 0.03, 1)
    c.tick()
    c.measure(0, 0)
    c.measure(1, 1)
    c.measure(2, 2)
    var recs: List[Int] = [0, 1]
    c.detector(recs)
    var o: List[Int] = [2]
    c.observable(0, o)
    c.reset(2)
    c.barrier()
    var text = to_qasm(c)
    assert_true(text.find("#pragma qumo depolarize1(0.01) q[0];") >= 0, text)
    assert_true(text.find("#pragma qumo detector 0, 1") >= 0, text)
    assert_true(text.find("#pragma qumo observable(0) 2") >= 0, text)
    var back = parse_qasm(text)
    assert_equal(len(back.ops), len(c.ops))
    for i in range(len(c.ops)):
        assert_equal(back.ops[i].kind, c.ops[i].kind)
    assert_equal(back.num_detectors, 1)
    assert_equal(back.num_observables, 1)
    assert_equal(back.records[0], 0)
    assert_equal(back.records[2], 2)
    assert_equal(to_qasm(back), text)
    # `pragma` without '#', non-qumo pragmas and annotations ignored
    var p = parse_qasm("qubit[1] q;\npragma qumo x_error(0.5) q[0];\n#pragma braket noise\n@reversible\nh q[0];\n")
    assert_equal(len(p.ops), 2)
    assert_equal(p.ops[0].kind, Op.X_ERROR)


def random_circuit(n: Int, depth: Int, seed: Int) -> Circuit:
    var c = Circuit(n)
    var th = c.parameter("theta[0]")
    var ph = c.parameter("phi")
    var s = seed
    for _ in range(depth):
        for k in range(Op.CCZ + 1):
            s = (s * 1103515245 + 12345) % 2147483648
            var a = s % n
            var b = (a + 1 + (s // 7) % (n - 1)) % n
            var d = 0
            for cand in range(n):
                if cand != a and cand != b:
                    d = cand
                    break
            var ang = Float64(s % 1000) / 137.0 - 3.0
            var p0 = Param(ang)
            if k % 5 == 0:
                p0 = th * 0.5 + ang
            elif k % 7 == 0:
                p0 = -ph
            c.add(k, a, b if op_arity(k) >= 2 else -1, d if op_arity(k) >= 3 else -1, p0, Param(ang * 0.3), Param(-ang))
    c.global_phase = 0.125
    return c^


def test_roundtrip_all_gates() raises:
    var c = random_circuit(4, 3, 42)
    var vals: List[Float64] = [0.77, -0.31]
    var text = to_qasm(c)
    var back = parse_qasm(text)
    assert_equal(back.param_names[0], "theta[0]")  # restored via #pragma qumo param
    assert_equal(len(back.ops), len(c.ops))
    assert_same_state(back, c, 0.0, vals)  # bit-exact
    assert_equal(to_qasm(back), text)  # canonical form is a fixed point
    # parameter values identical
    for i in range(len(c.ops)):
        assert_true(back.ops[i].p0 == c.ops[i].p0 or op_num_params(c.ops[i].kind) == 0)
    # pretty angles
    assert_equal(format_angle(pi / 2), "pi/2")
    assert_equal(format_angle(-3 * pi / 4), "-3*pi/4")
    assert_equal(format_angle(2 * pi), "2*pi")
    assert_equal(format_angle(0.1), "0.1")
    assert_equal(format_angle(1e300), "1e+300")
    assert_equal(format_angle(-7.0), "-7")
    assert_equal(format_angle(Float64(0.0) / Float64(0.0)), "nan")


def test_emitted_definitions_are_exact() raises:
    """The `gate xyz {...}` blocks written for non-stdgates natives must equal
    the native unitaries exactly (other toolchains will use them)."""
    var natives: List[Int] = [Op.SXDG, Op.ISWAP, Op.ISWAPDG, Op.ECR, Op.RXX, Op.RYY, Op.RZZ, Op.CCZ]
    for k in natives:
        var c = Circuit(3)
        var nq = op_arity(k)
        if op_num_params(k) == 1:
            c.add(k, 0, 1 if nq > 1 else -1, 2 if nq > 2 else -1, Param(0.731))
        else:
            c.add(k, 0, 1 if nq > 1 else -1, 2 if nq > 2 else -1)
        var text = to_qasm(c)
        var name = op_name(k)
        # rename the definition and the call so the body is really used
        var renamed = text.replace("gate " + name + " ", "gate my_" + name + " ").replace("gate " + name + "(", "gate my_" + name + "(")
        renamed = renamed.replace("\n" + name + " ", "\nmy_" + name + " ").replace("\n" + name + "(", "\nmy_" + name + "(")
        if k == Op.ISWAPDG:
            renamed = renamed.replace("gate iswap ", "gate my_iswap ")
        var prep_c = parse_qasm(head(3))
        var lhs = prep_c.copy()
        lhs.compose(parse_qasm(renamed))
        var rhs = prep_c.copy()
        rhs.compose(c)
        var used_body = False
        var pc = parse_qasm(renamed)
        for g in pc.ops:
            if g.kind != k:
                used_body = True
        assert_true(used_body or k == Op.SXDG, "definition body not used for " + name)
        assert_same_state(lhs, rhs, 1e-12)


def test_unicode_and_comments() raises:
    var c = parse_qasm("""
// leading comment
OPENQASM 3.0; /* block
comment */ include "stdgates.inc";
input float θ;
qubit[1] q;
gate rr(φ, λ) a { rz(φ) a; rx(λ) a; }
rr(θ/2, π/4) q[0];   // trailing
""")
    assert_equal(c.param_names[0], "θ")
    assert_true(c.ops[0].p0 == Param(0.0, 0, 0.5))
    assert_equal(c.ops[1].p0.value(), pi / 4)
    var t = to_qasm(c)
    assert_true(t.find("input float[64] p;") >= 0 or t.find("#pragma qumo param") >= 0, t)
    var back = parse_qasm(t)
    assert_equal(back.param_names[0], "θ")


def test_real_world_dialects() raises:
    """Shapes emitted by Qiskit (qasm2/qasm3 exporters), Braket and LLMs."""
    var qk3 = parse_qasm("""
OPENQASM 3.0;
include "stdgates.inc";
input float[64] theta;
gate rzx_140(_gate_p_0) _gate_q_0, _gate_q_1 {
  h _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rz(_gate_p_0) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_1;
}
bit[2] c;
qubit[2] q;
h q[0];
rzx_140(theta) q[0], q[1];
barrier q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
""")
    assert_equal(qk3.num_params(), 1)
    assert_equal(qk3.count(Op.CX), 2)
    assert_true(qk3.ops[3].p0 == Param.symbol(0))
    var qk2 = parse_qasm("""
OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[2];
creg meas[2];
u(pi/2,0,pi) q[0];
ecr q[0],q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
""")
    assert_equal(qk2.count(Op.ECR), 1)  # native definition -> exact native gate
    var hw = parse_qasm("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nbit[2] c;\nrz(pi/2) $0;\nsx $0;\ncx $0, $1;\nc[0] = measure $0;\nc[1] = measure $1;\n")
    assert_equal(hw.num_qubits, 2)
    var bk = parse_qasm("OPENQASM 3.0;\n#pragma braket verbatim\nqubit[2] q;\nbit[2] b;\nh q[0];\ncnot q[0], q[1];\nb = measure q;\n")
    assert_equal(bk.num_measurements, 2)
    var llm = parse_qasm("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[3] q;\nh q[0];\ncx q[0], q[1];\ncx q[1], q[2];\nbit[3] c = measure q;\n")
    assert_equal(llm.num_clbits, 3)
    assert_equal(llm.num_measurements, 3)
    assert_equal(llm.ops[5].clbit, 2)
    var singles = parse_qasm("qubit a; qubit b; bit m; h a; cx a, b; m = measure b;")
    assert_equal(singles.num_qubits, 2)
    assert_equal(singles.ops[2].q0, 1)


def test_errors() raises:
    expect_error("OPENQASM 3.0;\nqubit[2] q;\nhx q[0];\n", "QM-E510", "(line 3, col 1)")
    expect_error("qubit[2] q;\nrz q[0];", "QM-E511", "(line 2, col 1)")
    expect_error("qubit[2] q;\ncx q[0];", "QM-E512")
    expect_error("qubit[2] q;\n  cx q[0], q[0];", "QM-E513", "(line 2, col 3)")
    expect_error("qubit[2] a; qubit[3] b;\ncx a, b;", "QM-E514")
    expect_error("qubit[2] q;\nh r[0];", "QM-E520", "(line 2, col 3)")
    expect_error("qubit[2] q;\nh q[5];", "QM-E521")
    expect_error("qubit[2] q;\nqubit[3] q;", "QM-E522")
    expect_error("qubit[0] q;", "QM-E523")
    expect_error("input float t; qubit[2] q; h q[t];", "QM-E530")
    expect_error("input float a; input float b; qubit[1] q; rz(a*b) q[0];", "QM-E531")
    expect_error("input float a; input float b; qubit[1] q; rz(a+b) q[0];", "QM-E532")
    expect_error("qubit[1] q; rz(1/0) q[0];", "QM-E533")
    expect_error("qubit[1] q; rz(sinh(1)) q[0];", "QM-E534")
    expect_error("qubit[1] q; while (true) { h q[0]; }", "QM-E504")
    expect_error("def f() { }", "QM-E504")
    expect_error("qubit[2] q; bit[1] c; measure q -> c[0];", "QM-E540")
    expect_error("qubit[1] q; pow(0.5) @ x q[0];", "QM-E550")
    expect_error("qubit[1] q; gate g a { g a; } g q[0];", "QM-E551")
    expect_error("qubit[1] q;\n#pragma qumo detector 5\n", "QM-E560", "(line 2")
    expect_error("qubit[1] q; for uint i in [0:0:3] { h q[0]; }", "QM-E570")
    expect_error("qubit[1] q; h q[0]; ?", "QM-E501")
    expect_error("qubit[1] q; /* never closed", "QM-E502")
    expect_error("OPENQASM 4.0;", "QM-E503")
    expect_error("qubit[2] q;\nh q[0]\nx q[1];", "QM-E500", "(line 2, col 7)")
    expect_error("input float t; qubit[2] q;\nctrl @ U(t, 0, 0) q[0], q[1];", "QM-E202", "(line 2")
    expect_error("qubit[1] q; gate g a { h a; }\ng(0.1) q[0];", "QM-E511")
    # caret snippet rendering
    try:
        _ = parse_qasm("qubit[2] q;\ncx q[0], q[7];")
    except e:
        var s = String(e)
        assert_true(s.find("(line 2, col 12)") >= 0 and s.find("\n 2 | cx q[0], q[7];\n   |            ^") >= 0, s)
    # suggestions
    try:
        _ = parse_qasm("qubit[2] q; cnto q[0], q[1];")
    except e:
        assert_true(String(e).find("did you mean 'cnot'") >= 0, String(e))
    try:
        _ = parse_qasm("input float theta; qubit[1] q; rx(thetta) q[0];")
    except e:
        assert_true(String(e).find("did you mean 'theta'") >= 0, String(e))
    # warnings
    var w = List[String]()
    _ = parse_qasm_with_warnings("qubit[1] q; delay[100ns] q[0]; h q[0];", w)
    assert_equal(len(w), 1)
    assert_true(w[0].startswith("QM-W051"))


def test_speed() raises:
    var n = 20
    var src = String("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[20] q;\nbit[20] c;\n")
    var lines = 0
    for i in range(20000):
        var a = i % n
        var b = (i * 7 + 3) % n
        if a == b:
            b = (b + 1) % n
        src += "h q[" + String(a) + "];\n"
        src += "rz(0.123456789) q[" + String(b) + "];\n"
        src += "cx q[" + String(a) + "], q[" + String(b) + "];\n"
        src += "u3(pi/2, -pi/4, 0.5) q[" + String(a) + "];\n"
        src += "cp(-0.75) q[" + String(b) + "], q[" + String(a) + "];\n"
        lines += 5
    src += "c = measure q;\n"
    var t0 = perf_counter_ns()
    var c = parse_qasm(src)
    var t1 = perf_counter_ns()
    var text = to_qasm(c)
    var t2 = perf_counter_ns()
    assert_equal(len(c.ops), lines + 20)
    var mb = Float64(src.byte_length()) / 1e6
    var ps = Float64(t1 - t0) / 1e9
    print("  parse:", lines, "gates,", mb, "MB in", ps * 1e3, "ms ->", Float64(lines) / ps / 1e6, "Mgates/s;  emit:", Float64(t2 - t1) / 1e6, "ms,", text.byte_length(), "bytes")


def main() raises:
    test_bell_qasm2_qasm3()
    test_registers_and_broadcast()
    test_aliases_and_params()
    test_input_params()
    test_gate_defs_and_modifiers()
    test_for_loops_and_consts()
    test_feed_forward()
    test_pragmas_roundtrip()
    test_roundtrip_all_gates()
    test_emitted_definitions_are_exact()
    test_unicode_and_comments()
    test_real_world_dialects()
    test_errors()
    test_speed()
    print("test_qasm OK")
