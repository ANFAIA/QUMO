"""Barren plateaus: gradient variance of random circuits, global versus local cost.

Reproduces: PennyLane - Barren plateaus in quantum neural networks
Source: https://pennylane.ai/qml/demos/tutorial_barren_plateaus
Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_barren_plateaus/demo.py (Apache-2.0)
Reproduces: PennyLane - Alleviating barren plateaus with local cost functions
Source: https://pennylane.ai/qml/demos/tutorial_local_cost_functions
Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_local_cost_functions/demo.py (Apache-2.0)

The random circuit: RY(pi/4) on every qubit, then a random gate in {RX, RY, RZ}
with a random angle on each qubit, then a CZ ladder. The global cost is the
|0...0> projector, the local cost (1/n) sum_j |0><0|_j. QUMO samples 2000 random
circuits per width (its own RNG, seed 7) and computes the exact adjoint
gradient with respect to the last angle; the variance shrinks exponentially in
the number of qubits for the global cost and much more slowly for the local one.

Checks:
  * exact values and gradients (global and local) for 10 gate sequences exported
    from PennyLane 0.45.1 (examples/qml/reference/pennylane_reference.json) (1e-12);
  * the global gradient variance falls with every added qubit;
  * slope of ln Var versus n in [-1.3, -0.5] (PennyLane reruns: -0.77 to -0.96;
    fixture -0.955); the 200-sample estimate (the demo's size) is printed too;
  * the local-cost variance decays more slowly than the global one.

Run: .venv/bin/mojo run -I . examples/qml/barren_plateaus.mojo [--max-qubits 12]
example-meta: {"framework": "pennylane", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import pi, log as ln
from std.sys import argv
from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.rng import Rng
from qumo.algorithms.gradients import ExpectationEngine, Grad
from qumo.algorithms.qml import projector_zero, local_projector_zero
from qumo.testing.checks import CheckLog

from _common import Fixture, vmax_abs_diff, seconds_since, fmt_s


def max_qubits() raises -> Int:
    var raw = argv()
    var m = 6
    for i in range(1, len(raw) - 1):
        if String(raw[i]) == "--max-qubits":
            m = Int(String(raw[i + 1]))
    return m


def rand_circuit(gates: List[Int]) raises -> Circuit:
    """Gate codes: gates[i] in {0: RX, 1: RY, 2: RZ} acts on qubit i with parameter i."""
    var n = len(gates)
    var c = Circuit(n, 0, "random_circuit")
    var p = c.parameters("p", n)
    for i in range(n):
        c.ry(pi / 4, i)
    for i in range(n):
        if gates[i] == 0:
            c.rx(p[i], i)
        elif gates[i] == 1:
            c.ry(p[i], i)
        else:
            c.rz(p[i], i)
    for i in range(n - 1):
        c.cz(i, i + 1)
    return c^


def variance(v: List[Float64], count: Int) -> Float64:
    var m = 0.0
    for i in range(count):
        m += v[i]
    m /= Float64(count)
    var s = 0.0
    for i in range(count):
        s += (v[i] - m) ** 2
    return s / Float64(count)


def slope(xs: List[Float64], ys: List[Float64]) -> Float64:
    """Least-squares slope of ys against xs (numpy.polyfit degree 1)."""
    var n = Float64(len(xs))
    var mx = 0.0
    var my = 0.0
    for i in range(len(xs)):
        mx += xs[i]
        my += ys[i]
    mx /= n
    my /= n
    var sxy = 0.0
    var sxx = 0.0
    for i in range(len(xs)):
        sxy += (xs[i] - mx) * (ys[i] - my)
        sxx += (xs[i] - mx) ** 2
    return sxy / sxx


def main() raises:
    var log = CheckLog("barren_plateaus")
    var fx = Fixture("barren_plateaus")

    # 1. exact values and gradients for PennyLane's exported sequences
    var md = 0.0
    var nseq = len(fx.node("sequences"))
    for s in range(nseq):
        var key = "sequences/" + String(s) + "/"
        var names = fx.node(key + "gates")
        var gates = List[Int]()
        for i in range(len(names)):
            var nm = names.at(i).as_string()
            gates.append(0 if nm == "RX" else (1 if nm == "RY" else 2))
        var c = rand_circuit(gates)
        var n = len(gates)
        var params = fx.vec(key + "params")
        var eg = ExpectationEngine(c, projector_zero(n))
        var el = ExpectationEngine(c, local_projector_zero(n))
        var g = List[Float64]()
        md = max(md, abs(eg.value_and_gradient(params, g, Grad.ADJOINT) - fx.num(key + "global_value")))
        md = max(md, vmax_abs_diff(g, fx.vec(key + "global_grad")))
        md = max(md, abs(el.value_and_gradient(params, g, Grad.ADJOINT) - fx.num(key + "local_value")))
        md = max(md, vmax_abs_diff(g, fx.vec(key + "local_grad")))
    _ = log.close("10 PennyLane sequences: values and gradients max abs diff", md, 0.0, 1e-12)

    # 2. gradient-variance statistics with QUMO's RNG
    var nmax = max_qubits()
    var samples = 2000
    var rng = Rng(7)
    var qs = List[Float64]()
    var lg = List[Float64]()
    var ll = List[Float64]()
    var lg200 = List[Float64]()
    var t0 = perf_counter_ns()
    print("qubits  Var(global dC/dtheta_last)  Var(local)   [2000 samples; 200-sample global]")
    for n in range(2, nmax + 1):
        var gg = List[Float64]()
        var gl = List[Float64]()
        for _ in range(samples):
            var gates = List[Int]()
            for _ in range(n):
                gates.append(rng.below(3))
            var params = List[Float64]()
            for _ in range(n):
                params.append(pi * rng.uniform())
            var c = rand_circuit(gates)
            var eg = ExpectationEngine(c, projector_zero(n))
            var el = ExpectationEngine(c, local_projector_zero(n))
            gg.append(eg.gradient(params, Grad.ADJOINT)[n - 1])
            gl.append(el.gradient(params, Grad.ADJOINT)[n - 1])
        var vg = variance(gg, samples)
        var vl = variance(gl, samples)
        var v200 = variance(gg, 200)
        print("  ", n, "     ", vg, "   ", vl, "   [", v200, "]")
        qs.append(Float64(n))
        lg.append(ln(vg))
        ll.append(ln(vl))
        lg200.append(ln(v200))
    var secs = seconds_since(t0)

    var falls = True
    for i in range(1, len(lg)):
        if lg[i] >= lg[i - 1]:
            falls = False
    _ = log.check("global variance falls with every added qubit", falls, "n = 2.." + String(nmax))
    var sg = slope(qs, lg)
    var sl = slope(qs, ll)
    print("ln-variance slope: global", sg, " local", sl, " (200-sample global estimate", slope(qs, lg200), ")")
    print("PennyLane fixture (200 samples, n = 2..6): global", fx.num("global_slope"), " local", fx.num("local_slope"))
    _ = log.within("global ln-variance slope", sg, -1.3, -0.5)
    _ = log.check("local cost decays more slowly than global", sl > sg + 0.1,
                  "local=" + String(sl) + " global=" + String(sg))
    _ = log.check("local variance exceeds global at the largest width", ll[len(ll) - 1] > lg[len(lg) - 1], "")
    print("timing (indicative, different engines): QUMO", fmt_s(secs), "s for", samples, "x 2 gradients per width (n = 2..",
          nmax, "); PennyLane default.qubit", fmt_s(fx.num("stats_seconds")), "s for 200 per width, n = 2..6 (fixture)")
    log.finish()
