"""Tests for qumo.ft.encode / implement / emulate / tools: compile, implement_ops, encode.

Exactness first: every compiled logical program is compared with its source
(state vectors, no encoding), and every encoded program's logical outputs are
compared with the unencoded program's distribution (tableau for Clifford
programs, state vector for T / rotations), for both feed-forward modes, QEC
policies, state factories and both architectures.
"""

from std.math import pi, sqrt
from std.testing import assert_equal, assert_true

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.runner import evolve
from qumo.ai.tools import ToolRegistry
from qumo.ft.tools import register_ft_tools
from qumo.ai.json import parse_json
from qumo.ft.classical import CExpr
from qumo.ft.logical import LogicalProgram, LOp
from qumo.ft.arch import SteaneBuilder, IcebergBuilder, QECPolicy, FactoryConf, architecture_by_name
from qumo.ft.encode import compile, encode, check_may_encode
from qumo.ft.implement import implement_ops
from qumo.ft.emulate import (
    run_encoded, reference_counts, total_variation, logical_to_circuit, LogicalCounts,
)
from qumo.qec.codes import NoiseParams


def _fidelity(a: Circuit, b: Circuit) raises -> Float64:
    var sa = evolve(a)
    var sb = evolve(b)
    var re = 0.0
    var im = 0.0
    for i in range(sa.size()):
        re += sa.re[i] * sb.re[i] + sa.im[i] * sb.im[i]
        im += sa.re[i] * sb.im[i] - sa.im[i] * sb.re[i]
    return sqrt(re * re + im * im)


def _gate_zoo(clifford_t_only: Bool) -> List[Circuit]:
    """3-qubit circuits, each touching qubits 0, 1, 2 first (allocation order)."""
    var out = List[Circuit]()
    var ang = pi / 4 if clifford_t_only else 0.37
    for kind in range(19):
        var c = Circuit(3)
        c.h(0)
        c.h(1)
        c.h(2)
        c.t(0)
        c.s(1)
        if kind == 0:
            c.x(0); c.y(1); c.z(2); c.sx(0); c.sxdg(1); c.sdg(2); c.tdg(0)
        elif kind == 1:
            c.rz(3 * ang, 0); c.p(2 * ang, 1); c.rx(ang, 2)
        elif kind == 2:
            c.ry(3 * ang, 0); c.u(2 * ang, ang, 4 * ang, 1)
        elif kind == 3:
            c.cx(0, 1); c.cy(1, 2); c.cz(2, 0)
        elif kind == 4:
            c.ch(0, 2)
        elif kind == 5:
            c.swap(0, 2); c.t(2); c.cx(2, 1)
        elif kind == 6:
            c.iswap(0, 1)
        elif kind == 7:
            c.add(Op.ISWAPDG, 1, 2)
        elif kind == 8:
            c.crz(2 * ang, 0, 1); c.cp(2 * ang, 1, 2)
        elif kind == 9:
            c.crx(4 * ang, 2, 0); c.cry(2 * ang, 0, 1)
        elif kind == 10:
            c.rzz(ang, 0, 1); c.rxx(2 * ang, 1, 2); c.ryy(ang, 2, 0)
        elif kind == 11:
            c.ccx(0, 1, 2)
        elif kind == 12:
            c.ccz(2, 0, 1)
        elif kind == 13:
            c.cswap(1, 0, 2)
        elif kind == 14:
            c.h(0); c.cx(0, 1); c.cx(1, 2); c.t(2); c.h(1)
        elif kind == 15:
            c.sx(2); c.rz(ang, 2); c.cz(1, 2)
        elif kind == 16:
            c.y(0); c.cx(2, 0); c.tdg(1); c.ch(1, 2)
        elif kind == 17:
            c.rx(5 * ang, 1); c.ry(ang, 0); c.rz(7 * ang, 2)
        else:
            c.swap(1, 2); c.swap(0, 1); c.cz(0, 2)
        out.append(c^)
    return out^


def test_logical_semantics() raises:
    """Compile preserves the unitary (up to global phase) on both architectures."""
    var steane = SteaneBuilder().build(4)
    var ice = IcebergBuilder(2).build(2)
    var n = 0
    for c in _gate_zoo(True):
        var lp = compile(steane, c)
        var lc = logical_to_circuit(lp)
        assert_true(abs(_fidelity(c, lc) - 1.0) < 1e-9, "steane compile of zoo #" + String(n))
        n += 1
    n = 0
    for c in _gate_zoo(False):
        var lp = compile(ice, c)
        var lc = logical_to_circuit(lp)
        assert_true(abs(_fidelity(c, lc) - 1.0) < 1e-9, "iceberg compile of zoo #" + String(n))
        n += 1
    # static Clifford+T synthesis of a non-pi/4 angle: within the reported error
    var c = Circuit(1)
    c.h(0)
    c.rz(0.3, 0)
    var lp = compile(steane, c)
    assert_equal(len(lp.notes), 1)
    var f = _fidelity(c, logical_to_circuit(lp))
    assert_true(f > 1.0 - 2.5e-4, "static rz fidelity " + String(f))
    assert_true(lp.t_count() > 4)


def teleport() -> Circuit:
    var c = Circuit(3, 3, "teleport")
    c.x(0)
    c.h(1)
    c.cx(1, 2)
    c.cx(0, 1)
    c.h(0)
    c.measure(0, 0)
    c.measure(1, 1)
    c.z(2)
    c.c_if(0, 1)
    c.x(2)
    c.c_if(1, 1)
    c.measure(2, 2)
    return c^


def dynamic_program() -> Circuit:
    """Mid-circuit measurement, register condition, gates after measurement,
    reset, a conditioned H and a classical-only lane qubit (OR of two bits)."""
    var c = Circuit(5, 6, "dynamic")
    c.h(0)
    c.cx(0, 1)
    c.h(2)
    c.measure(0, 0)
    c.measure(2, 1)
    var bits: List[Int] = [0, 1]
    var vals: List[Int] = [1, 0]
    c.x(1)
    try:
        c.c_if_register(bits, vals)
    except:
        pass
    c.h(1)
    c.c_if(1, 1)
    c.x(0)  # gate after measurement: qubit 0 is |c0>, now |not c0>
    c.measure(0, 2)
    c.reset(2)
    c.x(2)
    c.measure(2, 3)
    c.measure(1, 4)
    # classical lane: q4 := c0 OR c1
    c.x(4)
    c.x(4)
    try:
        var zb: List[Int] = [0, 1]
        var zv: List[Int] = [0, 0]
        c.c_if_register(zb, zv)
    except:
        pass
    c.measure(4, 5)
    return c^


def _close(a: LogicalCounts, b: LogicalCounts, tol: Float64, what: String) raises:
    var tv = total_variation(a, b)
    assert_true(tv <= tol, what + ": TVD " + String(tv) + "\n  encoded   " + String(a) + "\n  reference " + String(b))


def test_encoded_clifford_programs() raises:
    var tp = teleport()
    var dyn = dynamic_program()
    var ref_tp = reference_counts(tp, 800, 5)
    var ref_dyn = reference_counts(dyn, 800, 6)
    # teleportation always delivers |1>
    for k in ref_tp.keys():
        assert_true(k.startswith("1"))
    var archs = List[String]()
    archs.append("steane frame")
    archs.append("steane eager")
    archs.append("steane knill")
    archs.append("steane steane-ec factories")
    archs.append("iceberg_2")
    archs.append("iceberg_4 syndrome")
    for name in archs:
        var a = SteaneBuilder().build(5)
        if name == "steane eager":
            a = SteaneBuilder().with_feed_forward("eager").build(5)
        elif name == "steane knill":
            a = SteaneBuilder().with_qec_policy(QECPolicy("knill", 2.0).with_cost("h", 1.0).with_cost("cx", 1.0)).build(5)
        elif name == "steane steane-ec factories":
            a = SteaneBuilder().with_qec_policy(QECPolicy("steane", 1.0).with_cost("cx", 1.0)).with_zero_factory_conf(FactoryConf(2, 3)).build(5)
        elif name == "iceberg_2":
            a = IcebergBuilder(2).build(3)
        elif name == "iceberg_4 syndrome":
            a = IcebergBuilder(4).with_qec_policy(QECPolicy("syndrome", 1.0).with_cost("rzz", 1.0)).build(2)
        var p1 = encode(a, tp)
        var e1 = run_encoded(p1, 800, 15)
        assert_equal(e1.accepted, 800)
        _close(e1, ref_tp, 0.08, name + " teleport")
        for k in e1.keys():
            assert_true(k.startswith("1"), name + " teleport delivered 0")
        var p2 = encode(a, dyn)
        var e2 = run_encoded(p2, 800, 16)
        _close(e2, ref_dyn, 0.08, name + " dynamic")
    # QEC policies actually inserted cycles; frame mode avoided physical corrections
    var knill = SteaneBuilder().with_qec_policy(QECPolicy("knill", 2.0).with_cost("h", 1.0).with_cost("cx", 1.0)).build(5)
    var pk = encode(knill, tp)
    assert_equal(pk.stats.qec_cycles, 2)
    var plain = encode(SteaneBuilder().build(3), tp)
    assert_equal(plain.stats.frame_flushes, 0)
    assert_equal(plain.stats.frame_corrections, 2)
    assert_true(plain.circuit.depth() < pk.circuit.depth())


def test_logical_api_and_factories() raises:
    """A logical program written directly (Guppy FT 'logical_program' notebook)."""
    var lp = LogicalProgram("bell_logical")
    var a = lp.alloc()
    var b = lp.alloc()
    lp.h(a)
    lp.cx(a, b)
    lp.qec_cycle(a)
    lp.qec_cycle(b)
    lp.measure(a, 0)
    lp.measure(b, 1)
    var arch = SteaneBuilder().with_zero_factory_conf(FactoryConf(2, 3)).build(2)
    var prog = implement_ops(arch, lp)
    assert_equal(prog.stats.qec_cycles, 2)
    var r = run_encoded(prog, 400, 3)
    assert_equal(r.counts.get("01", 0) + r.counts.get("10", 0), 0)
    assert_true(r.counts.get("00", 0) > 150 and r.counts.get("11", 0) > 150)
    # factory with 3 attempts: 2 extra attempts per prepared |0>_L
    assert_true(prog.stats.rus_attempts >= 2 * prog.stats.zero_states)
    # Iceberg: four logical qubits in one [[6,4,2]] block, rotations + block readout
    var c = Circuit(4, 0, "ice4")
    c.h(0)
    c.rzz(0.9, 0, 1)
    c.rx(0.4, 2)
    c.cx(2, 3)
    c.rz(1.1, 3)
    c.h(1)
    c.rxx(0.7, 1, 3)
    c.measure_all()
    var ice = IcebergBuilder(4).build(1)
    var pi4 = encode(ice, c)
    assert_true(pi4.circuit.num_qubits <= 9)
    var e = run_encoded(pi4, 600, 9)
    var rf = reference_counts(c, 600, 10)
    _close(e, rf, 0.09, "iceberg 4-qubit rotations")
    # same program over two [[4,2,2]] blocks (cross-block 4-body rotations)
    var ice2 = IcebergBuilder(2).build(2)
    var e2 = run_encoded(encode(ice2, c), 600, 11)
    _close(e2, rf, 0.09, "iceberg 2x[[4,2,2]] rotations")


def test_t_gates_statevector() raises:
    var steane = SteaneBuilder().build(2)
    # H T^4 H |0> = |1>
    var c = Circuit(1, 1, "t4")
    c.h(0)
    for _ in range(4):
        c.t(0)
    c.h(0)
    c.measure(0, 0)
    var prog = encode(steane, c)
    assert_equal(prog.stats.injections, 4)
    assert_equal(prog.stats.frame_flushes, 0)  # S corrections stay in the frame
    var r = run_encoded(prog, 12, 3)
    assert_equal(r.counts.get("1", 0), 12)
    # T^dag after T and S, S^dag: identity -> 0
    var c2 = Circuit(1, 1, "tt")
    c2.h(0)
    c2.t(0)
    c2.s(0)
    c2.tdg(0)
    c2.sdg(0)
    c2.h(0)
    c2.measure(0, 0)
    var r2 = run_encoded(encode(SteaneBuilder().with_feed_forward("eager").build(2), c2), 12, 4)
    assert_equal(r2.counts.get("0", 0), 12)


def test_errors() raises:
    var hit = 0
    var c = Circuit(3, 3)
    c.h(0)
    c.h(1)
    c.h(2)
    try:
        _ = encode(SteaneBuilder().build(2), c)
    except e:
        assert_true(String(e).startswith("QM-E725"), String(e))
        hit += 1
    var c2 = Circuit(2)
    c2.ecr(0, 1)
    try:
        _ = compile(SteaneBuilder().build(2), c2)
    except e:
        assert_true(String(e).startswith("QM-E724"), String(e))
        hit += 1
    var c3 = Circuit(2, 1)
    c3.h(0)
    c3.measure(0, 0)
    c3.t(1)
    c3.c_if(0, 1)
    try:
        _ = encode(SteaneBuilder().build(3), c3)
    except e:
        assert_true(String(e).startswith("QM-E726"), String(e))
        hit += 1
    var c4 = Circuit(1)
    c4.depolarize1(0.1, 0)
    try:
        check_may_encode(SteaneBuilder().build(1), c4)
    except e:
        assert_true(String(e).startswith("QM-E730"), String(e))
        hit += 1
    var c5 = Circuit(1)
    var th = c5.parameter("theta")
    c5.rz(th, 0)
    try:
        _ = compile(SteaneBuilder().build(1), c5)
    except e:
        assert_true(String(e).startswith("QM-E730"), String(e))
        hit += 1
    try:
        _ = QECPolicy("knill", 1.0).with_cost("teleport", 1.0)
    except e:
        assert_true(String(e).startswith("QM-E727"), String(e))
        hit += 1
    try:
        _ = SteaneBuilder().with_feed_forward("lazy").build(1)
    except e:
        assert_true(String(e).startswith("QM-E727"), String(e))
        hit += 1
    try:
        _ = architecture_by_name("toric", 1)
    except e:
        assert_true(String(e).startswith("QM-E727"), String(e))
        hit += 1
    var lp = LogicalProgram()
    var q = lp.alloc()
    lp.measure(q, 0)
    lp.h(q)
    try:
        lp.validate()
    except e:
        assert_true(String(e).startswith("QM-E727"), String(e))
        hit += 1
    assert_equal(hit, 9)


def test_noise_and_tools() raises:
    # circuit-level noise: the encoded teleportation stays mostly correct
    var prog = encode(SteaneBuilder().build(3), teleport())
    var noisy = run_encoded(prog, 400, 21, NoiseParams(0.002, 0.002, 0.002, 0.0))
    var ok = 0
    for k in noisy.keys():
        if k.startswith("1"):
            ok += noisy.counts.get(k, 0)
    assert_true(Float64(ok) / Float64(noisy.accepted) > 0.97)
    assert_true(noisy.accepted <= 400)
    # MCP tools are registered and answer
    var reg = ToolRegistry()  # full_registry() adds the same call (qumo/ai/registry.mojo)
    register_ft_tools(reg)
    var tnames: List[String] = ["qumo_ft_codes", "qumo_ft_encode", "qumo_ft_verify"]
    for name in tnames:
        assert_true(reg.find(name) >= 0, name)
    var r = reg.call("qumo_ft_codes", parse_json("{\"code\":\"iceberg_4\",\"architecture\":\"iceberg_4\"}"))
    assert_true(not r.is_error)
    assert_true(r.structured.find("\"verified_distance\":2") >= 0)
    assert_true(r.structured.find("\"valid\":false") < 0)
    var q = "OPENQASM 3.0; include \\\"stdgates.inc\\\"; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c[0] = measure q[0]; c[1] = measure q[1];"
    var r2 = reg.call("qumo_ft_encode", parse_json("{\"qasm\":\"" + q + "\",\"architecture\":\"iceberg_2\",\"shots\":200}"))
    assert_true(not r2.is_error, r2.structured)
    assert_true(r2.structured.find("\"total_variation_distance\"") >= 0)
    var r3 = reg.call("qumo_ft_encode", parse_json("{\"qasm\":\"" + q + "\",\"architecture\":\"steane\",\"n_blocks\":1}"))
    assert_true(r3.is_error)
    assert_true(r3.structured.find("QM-E725") >= 0)
    var sem = "OPENQASM 3.0; include \\\"stdgates.inc\\\"; qubit[1] q; h q[0];"
    var imp = "OPENQASM 3.0; include \\\"stdgates.inc\\\"; qubit[7] q; h q;"
    var r4 = reg.call("qumo_ft_verify", parse_json("{\"code\":\"steane\",\"semantic_qasm\":\"" + sem + "\",\"impl_qasm\":\"" + imp + "\"}"))
    assert_true(r4.structured.find("\"valid\":true") >= 0, r4.structured)


def main() raises:
    test_logical_semantics()
    test_encoded_clifford_programs()
    test_logical_api_and_factories()
    test_t_gates_statevector()
    test_errors()
    test_noise_and_tools()
    print("test_ft_encode OK")
