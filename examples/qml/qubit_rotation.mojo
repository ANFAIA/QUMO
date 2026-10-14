"""Qubit rotation: gradient descent from |0> to cost -1.

Reproduces: PennyLane - Basic tutorial: qubit rotation
Source: https://pennylane.ai/qml/demos/tutorial_qubit_rotation
Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_qubit_rotation/demo.py (Apache-2.0)

One qubit, RX(phi1) then RY(phi2), cost <Z> = cos(phi1) cos(phi2). Plain
gradient descent (stepsize 0.4, 100 steps from [0.011, 0.012]) rotates |0>
to |1>. Gradients come from QUMO's adjoint method, the update from
StepOptimizer.gd (PennyLane GradientDescentOptimizer).

Checks, against examples/qml/reference/pennylane_reference.json (PennyLane 0.45.1):
  * initial cost and gradient (1e-12);
  * the whole 100-step trajectory of costs and parameters (1e-12);
  * cost < -0.999999 first reached at the fixture's step (37);
  * final parameters about [0, pi] (1e-9).

Run: .venv/bin/mojo run -I . examples/qml/qubit_rotation.mojo
example-meta: {"framework": "pennylane", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import pi
from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.algorithms.gradients import ExpectationEngine, Grad
from qumo.algorithms.qml_train import StepOptimizer
from qumo.ai.checks import CheckLog

from _common import Fixture, vmax_abs_diff, seconds_since, fmt_s


def main() raises:
    var log = CheckLog("qubit_rotation")
    var fx = Fixture("qubit_rotation")

    var c = Circuit(1, 0, "qubit_rotation")
    var phi = c.parameters("phi", 2)
    c.rx(phi[0], 0)
    c.ry(phi[1], 0)
    var eng = ExpectationEngine(c, parse_pauli_sum("Z0"))

    var x: List[Float64] = [0.011, 0.012]
    var g = List[Float64]()
    var c0 = eng.value_and_gradient(x, g, Grad.ADJOINT)
    print("initial cost <Z> =", c0, " gradient =", g[0], g[1])
    _ = log.close("initial cost", c0, fx.num("init_cost"), 1e-12)
    _ = log.close("initial gradient max abs diff", vmax_abs_diff(g, fx.vec("init_grad")), 0.0, 1e-12)

    var opt = StepOptimizer.gd(0.4)
    var costs = List[Float64]()
    var params = List[Float64]()
    var first = -1
    var t0 = perf_counter_ns()
    for step in range(100):
        _ = eng.value_and_gradient(x, g, Grad.ADJOINT)
        opt.step(x, g)
        var cost = eng.value(x)
        costs.append(cost)
        params.append(x[0])
        params.append(x[1])
        if first < 0 and cost < -0.999999:
            first = step + 1
        if (step + 1) % 20 == 0:
            print("  step", step + 1, " cost", cost, " params", x[0], x[1])
    var secs = seconds_since(t0)

    _ = log.close("cost trajectory max abs diff (100 steps)", vmax_abs_diff(costs, fx.vec("cost_after_step")), 0.0, 1e-12)
    _ = log.close("parameter trajectory max abs diff", vmax_abs_diff(params, fx.vec("params_after_step")), 0.0, 1e-12)
    var ref_first = fx.int("first_step_below_m0_999999")
    _ = log.check("first step with cost < -0.999999", first == ref_first,
                  "qumo=" + String(first) + " pennylane=" + String(ref_first))
    _ = log.close("final phi1 about 0", x[0], 0.0, 1e-9)
    _ = log.close("final phi2 about pi", x[1], pi, 1e-9)
    _ = log.close("final cost", costs[99], -1.0, 1e-12)

    print("timing (indicative, different engines): QUMO", fmt_s(secs), "s for 100 steps;",
          "PennyLane default.qubit", fmt_s(fx.num("seconds")), "s (fixture)")
    log.finish()
