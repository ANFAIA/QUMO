"""Quantum models as Fourier series: serial and parallel data encoding.

Reproduces: PennyLane - Quantum models as Fourier series
Source: https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series
Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_expressivity_fourier_series/demo.py (Apache-2.0)

A model f(x) = <Z0> whose data enters through RX(x) can only express a
truncated Fourier series: one encoding gate gives frequencies {-1, 0, 1}, r
encodings give {-r, ..., r}. The demo fits the degree-1 target
g(x) = 0.1 + 2 Re[(0.15 + 0.15i) e^{ix}] on 70 points of [-6, 6]:
  * serial model (1 qubit): W(theta_0) S(x) W(theta_1), W = Rot, S = RX(x);
  * parallel model (3 qubits): SEL(3 layers) S(x) on every wire SEL(3 layers).
Both train with PennyLane-form Adam(0.3) on random batches of 25 points (the
fixture's batch indices are replayed) and the demo's 0.5 * MSE loss.

Checks, against examples/qml/reference/pennylane_reference.json (PennyLane 0.45.1):
  * serial f(0) = 0.7068878706839132 at the fixture's initial weights (1e-12);
  * serial cost after steps 10/20/30/40/50 and the whole trajectory (1e-8); the
    research-rerun plain-MSE values [0.22856, 0.025282, 0.0042809, 0.0038642, 0.00026741]
    equal twice these to 1e-3 relative;
  * Fourier coefficients of the trained serial model match the fixture (1e-8) and
    vanish above degree 1 (< 1e-12);
  * parallel model: f(0), 70-step trajectory (1e-8), coefficients vanish above degree 3.

Run: .venv/bin/mojo run -I . examples/qml/fourier_expressivity.mojo
example-meta: {"framework": "pennylane", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import pi, cos, sin
from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.algorithms.gradients import batch_value_and_gradient, batch_expectation
from qumo.algorithms.qml import append_rot, append_strongly_entangling_layers
from qumo.algorithms.qml_train import StepOptimizer
from qumo.testing.checks import CheckLog

from _common import Fixture, vmax_abs_diff, seconds_since, fmt_s


def serial_model() raises -> Circuit:
    var c = Circuit(1, 0, "serial_fourier")
    var w = c.parameters("w", 6)
    var x = c.parameters("x", 1)
    append_rot(c, w[0], w[1], w[2], 0)
    c.rx(x[0], 0)
    append_rot(c, w[3], w[4], w[5], 0)
    return c^


def parallel_model() raises -> Circuit:
    var c = Circuit(3, 0, "parallel_fourier")
    var w = c.parameters("w", 54)  # [block][layer][wire][k]
    var x = c.parameters("x", 1)
    var wires: List[Int] = [0, 1, 2]
    var b0 = List[type_of(w[0])]()
    var b1 = List[type_of(w[0])]()
    for i in range(27):
        b0.append(w[i])
        b1.append(w[27 + i])
    append_strongly_entangling_layers(c, b0, wires)
    for q in range(3):
        c.rx(x[0], q)
    append_strongly_entangling_layers(c, b1, wires)
    return c^


def sets_for(w: List[Float64], xs: List[Float64]) -> List[List[Float64]]:
    var out = List[List[Float64]]()
    for x in xs:
        var p = w.copy()
        p.append(x)
        out.append(p^)
    return out^


def model_values(c: Circuit, h: PauliSum, w: List[Float64], xs: List[Float64]) raises -> List[Float64]:
    return batch_expectation(c, h, sets_for(w, xs))


def half_mse(c: Circuit, h: PauliSum, w: List[Float64], xs: List[Float64], ys: List[Float64]) raises -> Float64:
    var p = model_values(c, h, w, xs)
    var s = 0.0
    for i in range(len(xs)):
        s += (ys[i] - p[i]) ** 2
    return 0.5 * s / Float64(len(xs))


def train(c: Circuit, h: PauliSum, mut w: List[Float64], xs: List[Float64], ys: List[Float64],
          batches: List[List[Int]]) raises -> List[Float64]:
    """Adam(0.3) on 0.5 * MSE over the replayed batches; returns the full-data cost
    before training and after every step."""
    var opt = StepOptimizer.adam(0.3)
    var nw = len(w)
    var cost = List[Float64]()
    cost.append(half_mse(c, h, w, xs, ys))
    for b in batches:
        var bx = List[Float64]()
        for i in b:
            bx.append(xs[i])
        var r = batch_value_and_gradient(c, h, sets_for(w, bx))
        var g = List[Float64](length=nw, fill=0.0)
        var n = Float64(len(b))
        for k in range(len(b)):
            var dl = -(ys[b[k]] - r[0][k]) / n  # d/dp of 0.5 * mean((t - p)^2)
            for j in range(nw):
                g[j] += dl * r[1][k][j]
        opt.step(w, g)
        cost.append(half_mse(c, h, w, xs, ys))
    return cost^


def fourier_coefficients(c: Circuit, h: PauliSum, w: List[Float64], K: Int) raises -> List[Float64]:
    """Coefficients numpy.fft.rfft(f(t)) / N on N = 2K + 1 points of [0, 2 pi): [re, im] per degree 0..K."""
    var N = 2 * K + 1
    var ts = List[Float64]()
    for k in range(N):
        ts.append(2.0 * pi * Float64(k) / Float64(N))
    var f = model_values(c, h, w, ts)
    var out = List[Float64]()
    for m in range(K + 1):
        var re = 0.0
        var im = 0.0
        for k in range(N):
            var a = 2.0 * pi * Float64(m * k) / Float64(N)
            re += f[k] * cos(a)
            im -= f[k] * sin(a)
        out.append(re / Float64(N))
        out.append(im / Float64(N))
    return out^


def max_above(coeffs: List[Float64], degree: Int) -> Float64:
    var m = 0.0
    for i in range(2 * (degree + 1), len(coeffs)):
        m = max(m, abs(coeffs[i]))
    return m


def main() raises:
    var log = CheckLog("fourier_expressivity")
    var fx = Fixture("fourier_expressivity")
    var xs = fx.vec("x")
    var ys = fx.vec("target_y")
    var h = parse_pauli_sum("Z0")

    # target check: g(x) = 0.1 + 2 (0.15 cos x - 0.15 sin x)
    var md = 0.0
    for i in range(len(xs)):
        md = max(md, abs(ys[i] - (0.1 + 0.3 * cos(xs[i]) - 0.3 * sin(xs[i]))))
    _ = log.close("target series max abs diff", md, 0.0, 1e-14)

    print("== serial model (1 qubit, r = 1)")
    var cs = serial_model()
    var w = fx.vec("serial_init_weights")
    var zero: List[Float64] = [0.0]
    _ = log.close("serial f(0) at the initial weights", model_values(cs, h, w, zero)[0], 0.7068878706839132, 1e-12)
    var t0 = perf_counter_ns()
    var cost = train(cs, h, w, xs, ys, fx.imat("serial_batches"))
    var t_s = seconds_since(t0)
    var every = List[Float64]()
    var page_ref = fx.vec("page_values/serial_cost_every10_mse")
    var rel = 0.0
    for k in range(5):
        var v = cost[10 * (k + 1)]
        every.append(v)
        print("  step", 10 * (k + 1), " cost (0.5*MSE)", v, " MSE", 2.0 * v, " page MSE", page_ref[k])
        rel = max(rel, abs(2.0 * v - page_ref[k]) / page_ref[k])
    _ = log.close("serial cost every 10 steps max abs diff", vmax_abs_diff(every, fx.vec("serial_cost_every10")), 0.0, 1e-8)
    _ = log.close("serial 50-step trajectory max abs diff", vmax_abs_diff(cost, fx.vec("serial_cost")), 0.0, 1e-8)
    _ = log.close("research-rerun MSE values, max relative diff", rel, 0.0, 1e-3)
    _ = log.close("serial final weights max abs diff", vmax_abs_diff(w, fx.vec("serial_final_weights")), 0.0, 1e-8)
    var co = fourier_coefficients(cs, h, w, 5)
    print("  trained coefficients c0 =", co[0], " c1 =", co[2], "+", co[3], "i  (target 0.1, 0.15 + 0.15i)")
    _ = log.close("serial Fourier coefficients vs fixture", vmax_abs_diff(co, fx.vec("serial_trained_coeffs_K5")), 0.0, 1e-8)
    _ = log.close("serial coefficients above degree 1", max_above(co, 1), 0.0, 1e-12)

    print("== parallel model (3 qubits, r = 3)")
    var cp = parallel_model()
    var pw = fx.vec("parallel_init_weights")
    _ = log.close("parallel f(0) at the initial weights", model_values(cp, h, pw, zero)[0], fx.num("parallel_f0_init"), 1e-12)
    t0 = perf_counter_ns()
    var pcost = train(cp, h, pw, xs, ys, fx.imat("parallel_batches"))
    var t_p = seconds_since(t0)
    print("  cost after 70 steps", pcost[70])
    _ = log.close("parallel 70-step trajectory max abs diff", vmax_abs_diff(pcost, fx.vec("parallel_cost")), 0.0, 1e-8)
    _ = log.close("parallel final weights max abs diff", vmax_abs_diff(pw, fx.vec("parallel_final_weights")), 0.0, 1e-8)
    var cpo = fourier_coefficients(cp, h, pw, 6)
    _ = log.close("parallel coefficients above degree 3", max_above(cpo, 3), 0.0, 1e-12)
    _ = log.check("parallel model fits the target", pcost[70] < 0.01, "cost=" + String(pcost[70]))
    print("timing (indicative, different engines): QUMO serial", fmt_s(t_s), "s, parallel", fmt_s(t_p),
          "s; PennyLane default.qubit", fmt_s(fx.num("serial_seconds")), "s and", fmt_s(fx.num("parallel_seconds")), "s (fixture)")
    log.finish()
