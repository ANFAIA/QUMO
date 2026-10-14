"""Quantum circuit Born machine: learning the 3x3 bars-and-stripes distribution with an MMD loss.

Reproduces: PennyLane - Quantum Circuit Born Machines
Source: https://pennylane.ai/qml/demos/tutorial_qcbm
Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_qcbm/demo.py (Apache-2.0)

Nine qubits (one per pixel), six StronglyEntanglingLayers (ranges 1, 162
angles, the demo's seed-42 initial weights from the fixture). The model
distribution p = |<b|U(theta)|0>|^2 is pulled toward the uniform distribution
over the 14 valid bars-and-stripes patterns by the maximum mean discrepancy
MMD(p, q) = (p - q)^T K (p - q), K a mixture of Gaussian kernels on the basis
index with bandwidths [0.25, 0.5, 1]. Gradient: since dMMD/dp = 2 K (p - q), the
gradient is ONE adjoint pass on the diagonal observable diag(2 K (p - q))
(QUMO diagonal_observable); the update is optax-form Adam(0.1), 100 iterations.
PennyLane wire w is QUMO qubit 8 - w, so basis indices coincide.

Checks, against examples/qml/reference/pennylane_reference.json (PennyLane 0.45.1 + JAX/optax, float64):
  * initial MMD loss and the first 8 probabilities (1e-12); initial gradient (1e-10);
  * the adjoint gradient matches central finite differences (1e-6);
  * MMD loss and KL trajectories (100 iterations) equal the fixture (1e-8);
  * final KL < 0.2 (fixture 0.0750) and valid-pattern probability > 0.85 (fixture 0.929).

Run: .venv/bin/mojo run -I . examples/qml/qcbm.mojo
example-meta: {"framework": "pennylane", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import exp, log
from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve
from qumo.algorithms.gradients import ExpectationEngine, Grad
from qumo.algorithms.datasets import bars_and_stripes
from qumo.algorithms.qml import append_strongly_entangling_layers, diagonal_observable
from qumo.algorithms.qml_train import StepOptimizer
from qumo.ai.checks import CheckLog

from _common import Fixture, vmax_abs_diff, seconds_since, fmt_s


comptime NQ = 9
comptime DIM = 512


def qcbm_circuit() raises -> Circuit:
    var c = Circuit(NQ, 0, "qcbm")
    var w = c.parameters("w", 6 * NQ * 3)
    var wires = List[Int]()
    for q in range(NQ):
        wires.append(NQ - 1 - q)  # PennyLane wire w -> QUMO qubit 8 - w
    append_strongly_entangling_layers(c, w, wires, [1, 1, 1, 1, 1, 1])
    return c^


def mmd_kernel() -> List[Float64]:
    var K = List[Float64](length=DIM * DIM, fill=0.0)
    var bw: List[Float64] = [0.25, 0.5, 1.0]
    for i in range(DIM):
        for j in range(DIM):
            var d2 = Float64((i - j) * (i - j))
            var s = 0.0
            for b in bw:
                s += exp(-d2 / (2.0 * b * b))
            K[i * DIM + j] = s / 3.0
    return K^


def kdiff(K: List[Float64], p: List[Float64], q: List[Float64]) -> List[Float64]:
    """K (p - q)."""
    var d = List[Float64](length=DIM, fill=0.0)
    for i in range(DIM):
        d[i] = p[i] - q[i]
    var out = List[Float64](length=DIM, fill=0.0)
    for i in range(DIM):
        var s = 0.0
        for j in range(DIM):
            s += K[i * DIM + j] * d[j]
        out[i] = s
    return out^


def mmd(K: List[Float64], p: List[Float64], q: List[Float64]) -> Float64:
    var kd = kdiff(K, p, q)
    var s = 0.0
    for i in range(DIM):
        s += (p[i] - q[i]) * kd[i]
    return s


def kl(p: List[Float64], q: List[Float64]) -> Float64:
    var s = 0.0
    for i in range(DIM):
        if q[i] > 0.0:
            s -= q[i] * log(p[i] / q[i])
    return s


def loss_and_grad(c: Circuit, K: List[Float64], q: List[Float64], w: List[Float64],
                  mut probs: List[Float64]) raises -> Tuple[Float64, List[Float64]]:
    """MMD loss and its gradient: one adjoint pass on diag(2 K (p - q))."""
    probs = evolve(c, w).probabilities()
    var kd = kdiff(K, probs, q)
    var L = 0.0
    var v = List[Float64](length=DIM, fill=0.0)
    for i in range(DIM):
        L += (probs[i] - q[i]) * kd[i]
        v[i] = 2.0 * kd[i]
    var eng = ExpectationEngine(c, diagonal_observable(v))
    return (L, eng.gradient(w, Grad.ADJOINT))


def main() raises:
    var log_ = CheckLog("qcbm")
    var fx = Fixture("qcbm")
    var valid = bars_and_stripes(3, 3)
    _ = log_.check("bars-and-stripes patterns equal the demo's", vmax_abs_diff(_f(valid), fx.vec("valid_patterns")) == 0.0,
                   String(len(valid)) + " patterns")
    var q = List[Float64](length=DIM, fill=0.0)
    for v in valid:
        q[v] = 1.0 / Float64(len(valid))
    var K = mmd_kernel()
    var c = qcbm_circuit()
    var w = fx.vec("init_weights")

    var probs = List[Float64]()
    var r = loss_and_grad(c, K, q, w, probs)
    var first8 = List[Float64]()
    for i in range(8):
        first8.append(probs[i])
    _ = log_.close("initial MMD loss", r[0], fx.num("init_loss"), 1e-12)
    _ = log_.close("initial probabilities [0..7] max abs diff", vmax_abs_diff(first8, fx.vec("init_probs_first8")), 0.0, 1e-12)
    _ = log_.close("initial gradient max abs diff (adjoint vs JAX)", vmax_abs_diff(r[1], fx.vec("init_grad")), 0.0, 1e-10)
    var h = 1e-5
    var fd = 0.0
    for k in [0, 40, 101, 161]:
        var wp = w.copy()
        wp[k] += h
        var lp = mmd(K, evolve(c, wp).probabilities(), q)
        wp[k] -= 2.0 * h
        var lm = mmd(K, evolve(c, wp).probabilities(), q)
        fd = max(fd, abs((lp - lm) / (2.0 * h) - r[1][k]))
    _ = log_.close("adjoint gradient vs central finite differences", fd, 0.0, 1e-6)

    var opt = StepOptimizer.adam(0.1, 0.9, 0.999, 1e-8, "optax")
    var losses = List[Float64]()
    var kls = List[Float64]()
    var t0 = perf_counter_ns()
    for it in range(100):
        var rr = loss_and_grad(c, K, q, w, probs)
        losses.append(rr[0])  # loss and KL at the parameters before the update (as the demo logs them)
        kls.append(kl(probs, q))
        opt.step(w, rr[1])
        if (it + 1) % 20 == 0:
            print("  iteration", it + 1, " MMD", rr[0], " KL", kls[it])
    var secs = seconds_since(t0)
    _ = log_.close("MMD loss trajectory max abs diff", vmax_abs_diff(losses, fx.vec("loss")), 0.0, 1e-8)
    _ = log_.close("KL trajectory max abs diff", vmax_abs_diff(kls, fx.vec("kl")), 0.0, 1e-8)
    _ = log_.close("final weights max abs diff", vmax_abs_diff(w, fx.vec("final_weights")), 0.0, 1e-6)

    var pf = evolve(c, w).probabilities()
    var fkl = kl(pf, q)
    var vf = 0.0
    for v in valid:
        vf += pf[v]
    print("final: MMD", mmd(K, pf, q), " KL", fkl, " probability of a valid pattern", vf)
    _ = log_.close("final KL vs fixture", fkl, fx.num("final_kl"), 1e-8)
    _ = log_.within("final KL", fkl, 0.0, 0.2)
    _ = log_.within("valid-pattern probability", vf, 0.85, 1.0)
    print("timing (indicative, different engines): QUMO", fmt_s(secs), "s for 100 iterations; PennyLane default.qubit + JAX jit",
          fmt_s(fx.num("train_seconds")), "s (fixture)")
    log_.finish()


def _f(v: List[Int]) -> List[Float64]:
    var out = List[Float64]()
    for x in v:
        out.append(Float64(x))
    return out^
