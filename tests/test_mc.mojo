"""Tests for qumo.ir.mc: multi-controlled gates against exact amplitudes."""

from std.math import pi, sqrt, cos, sin
from std.testing import assert_equal, assert_true
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.mc import (
    append_mcx, append_mcp, append_mcrz, controlled_by, _plan_ands, _AndStep, _two_qubit_cost,
)
from qumo.ir.qasm import parse_qasm, to_qasm
from qumo.sim.runner import evolve
from qumo.sim.rng import Rng
from qumo.sim.statevector import StateVector


def random_prep(n: Int, seed: UInt64, skip_from: Int = 1000) -> Circuit:
    """Generic entangled state on the first min(n, skip_from) qubits."""
    var rng = Rng(seed)
    var c = Circuit(n)
    var m = min(n, skip_from)
    for _ in range(2):
        for q in range(m):
            c.ry(rng.uniform() * 3.0, q)
            c.rz(rng.uniform() * 6.0, q)
        for q in range(m - 1):
            c.cx(q, q + 1)
    return c^


def mask_of(qs: List[Int]) -> Int:
    var m = 0
    for q in qs:
        m |= 1 << q
    return m


def assert_close(a: StateVector, re: List[Float64], im: List[Float64], label: String) raises:
    var worst = 0.0
    for i in range(a.size()):
        worst = max(worst, abs(a.re[i] - re[i]), abs(a.im[i] - im[i]))
    if worst > 1e-10:
        raise Error(label + ": differs by " + String(worst))


def check_mcx(n: Int, controls: List[Int], target: Int, ancillas: List[Int], mode: String, clean: Bool, label: String) raises:
    var prep = random_prep(n, 5, n - len(ancillas) if clean else 1000)
    var s0 = evolve(prep)
    var re = List[Float64](capacity=s0.size())
    var im = List[Float64](capacity=s0.size())
    var cm = mask_of(controls)
    for i in range(s0.size()):
        var j = i ^ (1 << target) if (i & cm) == cm else i
        re.append(s0.re[j])
        im.append(s0.im[j])
    var c = prep.copy()
    append_mcx(c, controls, target, ancillas, mode, clean)
    assert_close(evolve(c), re, im, label)


def check_diag(n: Int, controls: List[Int], target: Int, angle: Float64, c: Circuit, is_rz: Bool, label: String) raises:
    var prep = random_prep(n, 7)
    var s0 = evolve(prep)
    var re = List[Float64](capacity=s0.size())
    var im = List[Float64](capacity=s0.size())
    var cm = mask_of(controls)
    for i in range(s0.size()):
        var ph = 0.0
        if (i & cm) == cm:
            var bit = (i >> target) & 1
            if is_rz:
                ph = -angle / 2 if bit == 0 else angle / 2
            elif bit == 1:
                ph = angle
        re.append(s0.re[i] * cos(ph) - s0.im[i] * sin(ph))
        im.append(s0.re[i] * sin(ph) + s0.im[i] * cos(ph))
    var full = prep.copy()
    full.compose(c)
    assert_close(evolve(full), re, im, label)


def range_list(a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(a, b):
        out.append(i)
    return out^


def test_mcx_noaux() raises:
    for k in range(1, 10):
        check_mcx(k + 1, range_list(0, k), k, List[Int](), "noaux", False, "noaux k=" + String(k))
    # gray-code and recursive branch with shuffled operand order
    var cs: List[Int] = [5, 0, 3, 1, 6, 2]
    check_mcx(7, cs, 4, List[Int](), "auto", False, "noaux shuffled")
    print("  mcx noaux exact up to 9 controls")


def test_mcx_ancillas() raises:
    for k in range(3, 10):
        var cs = range_list(0, k)
        check_mcx(k + 1 + (k - 2), cs, k, range_list(k + 1, 2 * k - 1), "dirty", False, "dirty k=" + String(k))
        var count = Circuit(2 * k)
        append_mcx(count, cs, k, range_list(k + 1, 2 * k - 1), "dirty", False)
        assert_equal(_two_qubit_cost(count), 8 * k - 6, "dirty count")
    for k in range(3, 11):
        var cs = range_list(0, k)
        check_mcx(k + 2, cs, k, range_list(k + 1, k + 2), "clean", True, "clean k=" + String(k))
        var count = Circuit(k + 2)
        append_mcx(count, cs, k, range_list(k + 1, k + 2), "clean", True)
        assert_equal(_two_qubit_cost(count), 6 * (k - 1), "clean count (Khattar-Gidney)")
    for k in range(3, 10):
        check_mcx(k + 2, range_list(0, k), k, range_list(k + 1, k + 2), "split", False, "split k=" + String(k))
    print("  mcx dirty 8k-6, 1 clean 6(k-1), 1 dirty split: exact")


def test_plan_reaches_two_controls() raises:
    for k in range(3, 65):
        var steps = List[_AndStep]()
        var active = _plan_ands(range_list(0, k), range_list(k, k + 1), steps)
        assert_equal(len(active), 2, "conditionally clean plan stops at 2 controls, k=" + String(k))
        assert_equal(len(steps), k - 2, "one Toffoli per removed control")
    print("  conditionally clean plan: k - 2 Toffolis for k = 3..64")


def test_mcp_mcrz() raises:
    for k in range(1, 9):
        var cs = range_list(0, k)
        var c = Circuit(k + 1)
        append_mcp(c, 0.7, cs, k)
        check_diag(k + 1, cs, k, 0.7, c, False, "mcp k=" + String(k))
        var r = Circuit(k + 1)
        append_mcrz(r, 0.9, cs, k)
        check_diag(k + 1, cs, k, 0.9, r, True, "mcrz k=" + String(k))
    # one clean ancilla (prepared |0>: only qubits 0..5 are entangled)
    var cs = range_list(0, 5)
    var c = Circuit(7)
    append_mcp(c, 0.7, cs, 5, range_list(6, 7), True)
    var ref_c = Circuit(7)
    append_mcp(ref_c, 0.7, cs, 5)
    var pa = random_prep(7, 3, 6)
    var pb = pa.copy()
    pa.compose(c)
    pb.compose(ref_c)
    assert_true(evolve(pa).fidelity(evolve(pb)) > 1 - 1e-12, "mcp with clean ancilla")
    print("  mcp, mcrz exact up to 8 controls")


def test_symbolic_angle() raises:
    var c = Circuit(4)
    var th = c.parameter("theta")
    append_mcp(c, th * 2.0, range_list(0, 3), 3)
    var bound = c.bind([0.3])
    var d = Circuit(4)
    append_mcp(d, 0.6, range_list(0, 3), 3)
    var p1 = random_prep(4, 2)
    var p2 = p1.copy()
    p1.compose(bound)
    p2.compose(d)
    assert_true(evolve(p1).fidelity(evolve(p2)) > 1 - 1e-12, "symbolic scales through")
    print("  symbolic angles")


def controlled_reference(g: Circuit, n: Int, controls: List[Int]) raises:
    """Check controlled_by(g) against its definition: g acts where all controls are 1."""
    var prep = random_prep(n, 11)
    var s0 = evolve(prep)
    var c = prep.copy()
    c.compose(controlled_by(g, controls))
    var with_g = prep.copy()
    with_g.compose(g)
    var s1 = evolve(with_g)
    var cm = mask_of(controls)
    var re = List[Float64](capacity=s0.size())
    var im = List[Float64](capacity=s0.size())
    # the gates of g leave the controls alone, so the sectors evolve independently
    for i in range(s0.size()):
        var on = (i & cm) == cm
        re.append(s1.re[i] if on else s0.re[i])
        im.append(s1.im[i] if on else s0.im[i])
    assert_close(evolve(c), re, im, "controlled_by " + String(g.ops[0]))


def one_gate(n: Int, kind: Int, a: Int, b: Int = -1, d: Int = -1, p0: Float64 = 0.0, p1: Float64 = 0.0, p2: Float64 = 0.0) -> Circuit:
    var c = Circuit(n)
    c.add(kind, a, b, d, p0, p1, p2)
    return c^


def test_controlled_by_gate_set() raises:
    # controls 0..2, gate on 3..5. The prep entangles the sectors, so a gate
    # that leaks outside the all-ones sector or misses it fails.
    var n = 6
    var cs = range_list(0, 3)
    for k in [Op.X, Op.Y, Op.Z, Op.H, Op.S, Op.SDG, Op.T, Op.TDG, Op.SX, Op.SXDG]:
        controlled_reference(one_gate(n, k, 3), n, cs)
    for k in [Op.RX, Op.RY, Op.RZ, Op.P]:
        controlled_reference(one_gate(n, k, 3, p0=0.77), n, cs)
    controlled_reference(one_gate(n, Op.U, 3, p0=0.4, p1=1.1, p2=-0.6), n, cs)
    for k in [Op.CX, Op.CY, Op.CZ, Op.CH, Op.SWAP, Op.ISWAP, Op.ISWAPDG, Op.ECR]:
        controlled_reference(one_gate(n, k, 3, 4), n, cs)
    for k in [Op.CP, Op.CRX, Op.CRY, Op.CRZ, Op.RXX, Op.RYY, Op.RZZ]:
        controlled_reference(one_gate(n, k, 3, 4, p0=0.53), n, cs)
    for k in [Op.CCX, Op.CCZ, Op.CSWAP]:
        controlled_reference(one_gate(n, k, 3, 4, 5), n, cs)
    # a global phase becomes a controlled phase
    var g = Circuit(n)
    g.x(3)
    g.global_phase = 0.4
    controlled_reference(g, n, cs)
    print("  controlled_by covers every unitary opcode")


def test_qasm_ctrl() raises:
    # the case that failed with QM-E202
    var c = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[5] q; ctrl(4) @ x q[0], q[1], q[2], q[3], q[4];")
    var ref_c = Circuit(5)
    append_mcx(ref_c, range_list(0, 4), 4)
    var p1 = random_prep(5, 4)
    var p2 = p1.copy()
    p1.compose(c)
    p2.compose(ref_c)
    assert_true(evolve(p1).fidelity(evolve(p2)) > 1 - 1e-12, "ctrl(4) @ x")
    # idle qubits are borrowed: 8 controls with 6 spare qubits is the dirty V-chain
    var spare = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[15] q; ctrl(8) @ x q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];")
    assert_equal(_two_qubit_cost(spare), 8 * 8 - 6, "dirty V-chain chosen from idle qubits")
    # negctrl: fires on 000 only
    var neg = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[4] q; negctrl(3) @ x q[0], q[1], q[2], q[3];")
    var s = evolve(neg)
    assert_true(abs(s.re[8] - 1) < 1e-10, "negctrl(3) @ x |0000> -> |1000>")
    # symbolic angle under two controls, user gate under three
    var sym = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; input float theta; qubit[3] q; ctrl(2) @ rz(theta) q[0], q[1], q[2];")
    assert_equal(sym.num_params(), 1, "symbolic ctrl keeps the parameter")
    var user = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; gate g a, b { h a; cx a, b; } qubit[5] q; ctrl(3) @ g q[0], q[1], q[2], q[3], q[4];")
    assert_true(len(user) > 0, "ctrl on a user gate")
    # mixed modifiers and the round trip through OpenQASM
    var mix = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[5] q; ctrl(2) @ negctrl @ ctrl @ x q[0], q[1], q[2], q[3], q[4];")
    var back = parse_qasm(to_qasm(mix))
    assert_true(evolve(mix).fidelity(evolve(back)) > 1 - 1e-12, "round trip")
    # unchanged forms keep the exact old expansion
    var old = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[3] q; ctrl(2) @ x q[0], q[1], q[2];")
    assert_equal(len(old), 1, "ctrl(2) @ x stays one ccx")
    assert_equal(old.ops[0].kind, Op.CCX, "ctrl(2) @ x stays one ccx")
    print("  QASM ctrl(k) @, negctrl(k) @")


def test_errors() raises:
    var c = Circuit(4)
    var raised = False
    try:
        append_mcx(c, range_list(0, 3), 2)
    except e:
        raised = String(e).startswith("QM-E204")
    assert_true(raised, "QM-E204 repeated qubit")
    raised = False
    try:
        append_mcx(c, range_list(0, 3), 3, List[Int](), "bogus")
    except e:
        raised = String(e).startswith("QM-E205")
    assert_true(raised, "QM-E205 unknown mode")
    raised = False
    try:
        append_mcx(c, range_list(0, 3), 3, List[Int](), "dirty")
    except e:
        raised = String(e).startswith("QM-E206")
    assert_true(raised, "QM-E206 missing ancillas")
    print("  error codes")


def main() raises:
    test_mcx_noaux()
    test_mcx_ancillas()
    test_plan_reaches_two_controls()
    test_mcp_mcrz()
    test_symbolic_angle()
    test_controlled_by_gate_set()
    test_qasm_ctrl()
    test_errors()
    print("test_mc OK")
