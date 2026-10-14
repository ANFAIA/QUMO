"""Quantum natural gradient: the Fubini-Study metric versus plain gradient descent.

Reproduces: PennyLane - Quantum natural gradient
Source: https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient
Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_quantum_natural_gradient/demo.py (Apache-2.0)

Three qubits, fixed RY(pi/4), RY(pi/3), RY(pi/7), then two parametrized
layers (RZ RZ | CNOT CNOT | RY RX | CNOT CNOT), cost <Y0>, start
[0.432, -0.123, 0.543, 0.233]. QUMO computes the block-diagonal metric tensor
(PennyLane's default approximation), the diagonal approximation and the exact
full metric, then runs 200 steps of QNGOptimizer(0.01) and 200 steps of
gradient descent(0.01). The natural gradient converges much faster.

Checks, against examples/qml/reference/pennylane_reference.json (PennyLane 0.45.1):
  * metric diagonal 0.125, 0.1875, 0.24973433337719794, 0.20293622523234162
    and g23 = -0.015247010412867025 (1e-14); diag approximation (1e-14);
  * full metric (PennyLane approx=None with an auxiliary wire) (1e-12);
  * QNG cost after steps 1/10/50/100/200 and the whole trajectory (1e-10);
  * GD cost after 200 steps -0.3737189660016403 and the trajectory (1e-10);
  * QNG ends lower than GD.

Run: .venv/bin/mojo run -I . examples/qml/quantum_natural_gradient.mojo
example-meta: {"framework": "pennylane", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import pi
from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.algorithms.gradients import ExpectationEngine, Grad
from qumo.algorithms.qml_train import StepOptimizer
from qumo.algorithms.qgeometry import metric_tensor, parameter_layers, QNGOptimizer
from qumo.ai.checks import CheckLog

from _common import Fixture, vmax_abs_diff, seconds_since, fmt_s


def qng_circuit() raises -> Circuit:
    var c = Circuit(3, 0, "qng_demo")
    var p = c.parameters("p", 4)
    c.ry(pi / 4, 0)
    c.ry(pi / 3, 1)
    c.ry(pi / 7, 2)
    c.rz(p[0], 0)
    c.rz(p[1], 1)
    c.cx(0, 1)
    c.cx(1, 2)
    c.ry(p[2], 1)
    c.rx(p[3], 2)
    c.cx(0, 1)
    c.cx(1, 2)
    return c^


def main() raises:
    var log = CheckLog("quantum_natural_gradient")
    var fx = Fixture("quantum_natural_gradient")
    var c = qng_circuit()
    var h = parse_pauli_sum("Y0")
    var p0 = fx.vec("init_params")

    var eng = ExpectationEngine(c, h)
    var g = List[Float64]()
    var e0 = eng.value_and_gradient(p0, g, Grad.ADJOINT)
    _ = log.close("initial cost <Y0>", e0, fx.num("init_cost"), 1e-12)
    _ = log.close("initial gradient max abs diff", vmax_abs_diff(g, fx.vec("init_grad")), 0.0, 1e-12)

    # metric tensors
    var layers = parameter_layers(c)
    _ = log.check("two parametrized layers", len(layers) == 2, "layers=" + String(len(layers)))
    var G = metric_tensor(c, p0, "block-diag")
    print("block-diagonal metric tensor:")
    for i in range(4):
        print("  ", G[4 * i], G[4 * i + 1], G[4 * i + 2], G[4 * i + 3])
    var diag_ref: List[Float64] = [0.125, 0.1875, 0.24973433337719794, 0.20293622523234162]
    for i in range(4):
        _ = log.close("metric g" + String(i) + String(i), G[5 * i], diag_ref[i], 1e-14)
    _ = log.close("metric g23", G[11], -0.015247010412867025, 1e-14)
    _ = log.close("block-diag metric vs fixture max abs diff", vmax_abs_diff(G, fx.vec("metric_block_diag")), 0.0, 1e-14)
    var D = metric_tensor(c, p0, "diag")
    _ = log.close("diag metric vs fixture max abs diff", vmax_abs_diff(D, fx.vec("metric_diag")), 0.0, 1e-14)
    var F = metric_tensor(c, p0, "full")
    _ = log.close("full metric vs fixture (aux-wire) max abs diff", vmax_abs_diff(F, fx.vec("metric_full")), 0.0, 1e-12)

    # 200 QNG steps (PennyLane QNGOptimizer(0.01), block-diag metric)
    var t0 = perf_counter_ns()
    var p = p0.copy()
    var qng = QNGOptimizer(0.01)
    var qng_cost = List[Float64]()
    for _ in range(200):
        _ = qng.step(c, h, p)
        qng_cost.append(eng.value(p))
    var t_qng = seconds_since(t0)
    var ref_qng = fx.vec("qng_cost_after_step")
    var steps: List[Int] = [1, 10, 50, 100, 200]
    var page: List[Float64] = [0.06448978651301013, -0.03996196845262606, -0.4575443493032344, -0.5947941532960949, -0.6122039920741909]
    for k in range(len(steps)):
        var s = steps[k]
        print("  QNG step", s, " cost", qng_cost[s - 1])
        _ = log.close("QNG cost after step " + String(s), qng_cost[s - 1], page[k], 1e-10)
    _ = log.close("QNG trajectory (200 steps) max abs diff", vmax_abs_diff(qng_cost, ref_qng), 0.0, 1e-10)
    _ = log.close("QNG final params max abs diff", vmax_abs_diff(p, fx.vec("qng_final_params")), 0.0, 1e-10)

    # 200 GD steps
    t0 = perf_counter_ns()
    var pg = p0.copy()
    var gd = StepOptimizer.gd(0.01)
    var gd_cost = List[Float64]()
    for _ in range(200):
        _ = eng.value_and_gradient(pg, g, Grad.ADJOINT)
        gd.step(pg, g)
        gd_cost.append(eng.value(pg))
    var t_gd = seconds_since(t0)
    print("  GD  step 200  cost", gd_cost[199])
    _ = log.close("GD cost after step 200", gd_cost[199], -0.3737189660016403, 1e-10)
    _ = log.close("GD trajectory (200 steps) max abs diff", vmax_abs_diff(gd_cost, fx.vec("gd_cost_after_step")), 0.0, 1e-10)
    _ = log.check("QNG reaches a lower cost than GD", qng_cost[199] < gd_cost[199] - 0.1,
                  "qng=" + String(qng_cost[199]) + " gd=" + String(gd_cost[199]))

    print("timing (indicative, different engines): QUMO QNG", fmt_s(t_qng), "s, GD", fmt_s(t_gd),
          "s; PennyLane default.qubit QNG", fmt_s(fx.num("qng_seconds")), "s, GD", fmt_s(fx.num("gd_seconds")), "s (fixture)")
    log.finish()
