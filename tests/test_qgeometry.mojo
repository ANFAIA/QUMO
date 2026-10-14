"""Tests for qumo.algorithms.qgeometry: parametrized layers, Fubini-Study metric
(block-diag / diag / full), quantum natural gradient. Reference: PennyLane 0.45.1
tutorial_quantum_natural_gradient (3 qubits, params [0.432, -0.123, 0.543, 0.233],
observable Y0). Prints `test_qgeometry OK`."""

from std.math import pi
from std.testing import assert_true, assert_equal

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve
from qumo.algorithms.gradients import ExpectationEngine
from qumo.algorithms.library import qaoa_maxcut_ansatz
from qumo.algorithms.hamiltonians import Graph
from qumo.algorithms.qml_train import StepOptimizer
from qumo.algorithms.qgeometry import parameter_layers, metric_tensor, qng_direction, QNGOptimizer


def _code(msg: String) -> String:
    return String(msg[byte=0:7]) if msg.byte_length() >= 7 else msg


def _close(a: Float64, b: Float64, tol: Float64, what: String) raises:
    assert_true(abs(a - b) <= tol, what + ": " + String(a) + " vs " + String(b) + " (|diff| " + String(abs(a - b)) + " > " + String(tol) + ")")


def _qng_circuit() raises -> Circuit:
    var q = Circuit(3, 0, "qng_demo")
    var th = q.parameters("p", 4)
    q.ry(pi / 4, 0)
    q.ry(pi / 3, 1)
    q.ry(pi / 7, 2)
    q.rz(th[0], 0)
    q.rz(th[1], 1)
    q.cx(0, 1)
    q.cx(1, 2)
    q.ry(th[2], 1)
    q.rx(th[3], 2)
    q.cx(0, 1)
    q.cx(1, 2)
    return q^


def test_metric() raises:
    var q = _qng_circuit()
    var p: List[Float64] = [0.432, -0.123, 0.543, 0.233]
    var lay = parameter_layers(q)
    assert_equal(len(lay), 2)
    assert_equal(lay[0][0], 0)
    assert_equal(lay[0][1], 1)
    assert_equal(lay[1][0], 2)
    assert_equal(lay[1][1], 3)
    var G = metric_tensor(q, p)
    _close(G[0], 0.125, 1e-14, "g00")
    _close(G[5], 0.1875, 1e-14, "g11")
    _close(G[10], 0.24973433337719794, 1e-14, "g22")
    _close(G[15], 0.20293622523234162, 1e-14, "g33")
    _close(G[11], -0.015247010412867025, 1e-14, "g23")
    _close(G[14], G[11], 0.0, "symmetry")
    _close(G[1], 0.0, 1e-14, "g01")
    _close(G[2], 0.0, 0.0, "between layers")
    var F = metric_tensor(q, p, "full")
    var blocks: List[Int] = [0, 1, 4, 5, 10, 11, 14, 15]
    for i in blocks:
        _close(F[i], G[i], 1e-14, "full == block-diag on the blocks")
    assert_true(abs(F[2]) > 1e-3, "full metric couples the layers")
    var D = metric_tensor(q, p, "diag")
    for i in range(4):
        for j in range(4):
            _close(D[i * 4 + j], G[i * 4 + j] if i == j else 0.0, 0.0, "diag of block-diag")
    var W = metric_tensor(q, p, "block-diag", [3, 2])
    _close(W[0], G[15], 0.0, "wrt order")
    _close(W[1], G[11], 0.0, "wrt order")


def test_full_metric_vs_finite_differences() raises:
    """'full' == Re<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi> from finite-difference states,
    on a non-layered circuit with every supported generator and affine coefficients."""
    var c = Circuit(3, 0, "mixed")
    var t = c.parameters("t", 11)
    c.h(0)
    c.h(1)
    c.rx(t[0] * 2.0, 0)
    c.ry(t[1], 1)
    c.rzz(t[2], 0, 1)
    c.rz(t[3] * -0.5 + 0.1, 2)
    c.crx(t[4], 1, 2)
    c.p(t[5], 0)
    c.cry(t[6], 0, 2)
    c.rxx(t[7], 1, 2)
    c.crz(t[8], 2, 0)
    c.ryy(t[9], 0, 2)
    c.cp(t[10], 1, 0)
    var p: List[Float64] = [0.3, -0.7, 1.1, 0.4, -1.3, 0.9, 0.25, -0.6, 1.7, 0.05, -0.45]
    var F = metric_tensor(c, p, "full")
    var n = 11
    var h = 1e-5
    var psi = evolve(c, p)
    var ds = List[StateVector]()
    for i in range(n):
        var pp = p.copy()
        pp[i] += h
        var a = evolve(c, pp)
        pp[i] -= 2.0 * h
        var b = evolve(c, pp)
        var d = a.copy()
        for k in range(d.size()):
            d.re[k] = (a.re[k] - b.re[k]) / (2.0 * h)
            d.im[k] = (a.im[k] - b.im[k]) / (2.0 * h)
        ds.append(d^)
    for i in range(n):
        for j in range(n):
            var dd = ds[i].inner(ds[j])
            var a = ds[i].inner(psi)
            var b = psi.inner(ds[j])
            var ref_ = dd[0] - (a[0] * b[0] - a[1] * b[1])
            _close(F[i * n + j], ref_, 1e-8, "full metric [" + String(i) + "," + String(j) + "]")


def test_qng() raises:
    var q = _qng_circuit()
    var h = parse_pauli_sum("Y0")
    var p: List[Float64] = [0.432, -0.123, 0.543, 0.233]
    var opt = QNGOptimizer(0.01)
    var cost = List[Float64]()
    for _ in range(201):
        cost.append(opt.step(q, h, p))  # cost before each step: cost[k] = after k steps
    _close(cost[0], 0.0747230475052496, 1e-12, "QNG initial cost")
    _close(cost[1], 0.06448978651301013, 1e-10, "QNG step 1")
    _close(cost[10], -0.03996196845262606, 1e-10, "QNG step 10")
    _close(cost[50], -0.4575443493032344, 1e-10, "QNG step 50")
    _close(cost[100], -0.5947941532960949, 1e-10, "QNG step 100")
    _close(cost[200], -0.6122039920741909, 1e-10, "QNG step 200")
    # plain gradient descent reference (PennyLane GradientDescentOptimizer(0.01), 200 steps)
    var pg: List[Float64] = [0.432, -0.123, 0.543, 0.233]
    var gd = StepOptimizer.gd(0.01)
    var eng = ExpectationEngine(q, h)
    for _ in range(200):
        var g = List[Float64]()
        _ = eng.value_and_gradient(pg, g)
        gd.step(pg, g)
    _close(eng.value(pg), -0.3737189660016403, 1e-10, "GD step 200")
    # pinv direction on a singular metric
    var d = qng_direction([1.0, 0.0, 0.0, 0.0], [2.0, 3.0])
    _close(d[0], 2.0, 1e-15, "pinv")
    _close(d[1], 0.0, 1e-15, "pinv null space")
    var dl = qng_direction([1.0, 0.0, 0.0, 0.0], [2.0, 3.0], 0.5)
    _close(dl[1], 6.0, 1e-12, "lam regularisation")


def test_errors() raises:
    var qa = qaoa_maxcut_ansatz(Graph.ring(4), 1)
    var code = String()
    try:
        _ = metric_tensor(qa, [0.1, 0.2])
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E832")
    var cu = Circuit(1)
    var a = cu.parameters("a", 3)
    cu.u(a[0], a[1], a[2], 0)
    code = String()
    try:
        _ = metric_tensor(cu, [0.1, 0.2, 0.3])
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E832")
    code = String()
    try:
        _ = metric_tensor(_qng_circuit(), [0.1, 0.2, 0.3, 0.4], "exact")
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E832")


def main() raises:
    test_metric()
    test_full_metric_vs_finite_differences()
    test_qng()
    test_errors()
    print("test_qgeometry OK")
