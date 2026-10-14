"""Quantum convolutional neural network: generalisation from few training digits.

Reproduces: PennyLane - Generalization in QML from few training data
Source: https://pennylane.ai/qml/demos/tutorial_learning_few_data
Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_learning_few_data/demo.py (Apache-2.0)

Digits 0 and 1 of the UCI/sklearn 8x8 digits (360 images, examples/qml/data/digits01.csv),
each image L2-normalised and amplitude-encoded on 6 qubits. QUMO loads the
amplitudes directly as the engine's initial state (no state-preparation gates);
the Mottonen RY tree is run once as a cross-check. The QCNN: two blocks of
convolution (U3, IsingXX/YY/ZZ, U3 on neighbouring pairs) and pooling
(mid-circuit measurement + conditioned U3, implemented by deferred measurement
as a controlled U3), then ArbitraryUnitary on the two remaining wires (15
weights); the output is the probability of the label on PennyLane wire 0.
Cost 1 - mean p(label); optax Adam with cosine_decay_schedule(0.1, 100,
alpha=0.95); training sizes 5, 10, 20, 40 with 10 repetitions each, 100 epochs,
100 test images. Initial weights and the train/test draws are the fixture's
(the demo draws weights from an unseeded RNG; the generator seeds and records them).

Checks, against examples/qml/reference/pennylane_reference.json (PennyLane 0.45.1 + JAX/optax, float64):
  * forward pass p = [0.4247292672981652, 0.5752707327018325] with fixed weights (1e-11);
  * dp(label)/dW and dp(label)/dlast (1e-10); Mottonen-prepared circuit agrees (1e-12);
  * all 40 training-cost trajectories (100 epochs each) equal the fixture (1e-8);
  * all 40 final train/test accuracies equal the fixture;
  * mean test accuracy at 40 training images >= 0.9 (fixture 0.946).

Run: .venv/bin/mojo run -I . examples/qml/qcnn.mojo
example-meta: {"framework": "pennylane", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.ir.param import Param
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.statevector import StateVector
from qumo.algorithms.gradients import ExpectationEngine, Grad, batch_value_and_gradient
from qumo.algorithms.datasets import load_csv, l2_normalize_rows
from qumo.algorithms.qml import (
    append_qcnn_conv, append_qcnn_pool, append_arbitrary_unitary, append_amplitude_embedding_real, amplitude_state,
)
from qumo.algorithms.qml_train import StepOptimizer, cosine_lr
from qumo.testing.checks import CheckLog

from _common import Fixture, vmax_abs_diff, seconds_since, fmt_s, DATA_DIR


comptime NP = 51  # 36 block weights W[k][j] at 2k + j, then 15 dense weights


def qcnn() raises -> Circuit:
    var c = Circuit(6, 0, "qcnn")
    var W = c.parameters("W", 36)
    var L = c.parameters("L", 15)
    var wires: List[Int] = [5, 4, 3, 2, 1, 0]  # PennyLane wire w = QUMO qubit 5 - w
    for j in range(2):
        var kw = List[Param]()
        for k in range(15):
            kw.append(W[2 * k + j])
        append_qcnn_conv(c, kw, wires, j != 0)
        var pw: List[Param] = [W[30 + j], W[32 + j], W[34 + j]]
        var sinks = List[Int]()
        var sources = List[Int]()
        for i in range(1, len(wires), 2):
            sinks.append(wires[i - 1])
            sources.append(wires[i])
        append_qcnn_pool(c, pw, sinks, sources)
        var nw = List[Int]()
        for i in range(0, len(wires), 2):
            nw.append(wires[i])
        wires = nw^
    append_arbitrary_unitary(c, L, wires)
    return c^


def fixed_params() -> List[Float64]:
    from std.math import sin, cos
    var p = List[Float64]()
    for k in range(18):
        for j in range(2):
            p.append(sin(0.37 * Float64(k) + 1.3 * Float64(j) + 0.1))
    for i in range(15):
        p.append(cos(0.21 * Float64(i) + 0.4))
    return p^


def outputs(c: Circuit, w: List[Float64], states: List[StateVector], labels: List[Float64],
            idx: List[Int], with_grad: Bool) raises -> Tuple[List[Float64], List[Float64]]:
    """(p(label) per row, gradient of the cost 1 - mean p(label)). p(label) = 1/2 +- <Z5>/2."""
    var sets = List[List[Float64]]()
    var st = List[StateVector]()
    for i in idx:
        sets.append(w.copy())
        st.append(states[i].copy())
    var h = parse_pauli_sum("Z5")
    var r = batch_value_and_gradient(c, h, sets, Grad.ADJOINT, st)
    var p = List[Float64]()
    var g = List[Float64](length=NP, fill=0.0)
    var n = Float64(len(idx))
    for b in range(len(idx)):
        var s = 1.0 if labels[idx[b]] == 0.0 else -1.0
        p.append(0.5 + 0.5 * s * r[0][b])
        if with_grad:
            for k in range(NP):
                g[k] -= 0.5 * s * r[1][b][k] / n
    return (p^, g^)


def acc(p: List[Float64]) -> Float64:
    var ok = 0
    for v in p:
        if v > 0.5:
            ok += 1
    return Float64(ok) / Float64(len(p))


def main() raises:
    var log = CheckLog("qcnn")
    var fx = Fixture("qcnn")
    var d = load_csv(DATA_DIR + "digits01.csv")
    var feats = l2_normalize_rows(d[0])
    var labels = d[1].copy()
    var states = List[StateVector]()
    for f in feats:
        states.append(amplitude_state(f, 6, True, 0.5))
    var c = qcnn()

    # fixed-weight forward pass and gradient on image 0 (label 0)
    var p = fixed_params()
    var eng = ExpectationEngine(c, parse_pauli_sum("0.5 + 0.5 * Z5"))  # P(PennyLane wire 0 = 0)
    eng.set_initial_state(states[0])
    var g = List[Float64]()
    var p0 = eng.value_and_gradient(p, g, Grad.ADJOINT)
    var probs: List[Float64] = [p0, 1.0 - p0]
    print("forward pass, image 0: p =", probs[0], probs[1])
    _ = log.close("forward pass probabilities max abs diff", vmax_abs_diff(probs, fx.vec("fixed_probs_wire0")), 0.0, 1e-11)
    _ = log.check("fixed sample label is 0", fx.int("fixed_label") == Int(labels[0]), "")
    var gW = List[Float64]()
    var gL = List[Float64]()
    for k in range(36):
        gW.append(g[k])
    for k in range(15):
        gL.append(g[36 + k])
    _ = log.close("dp/dW max abs diff", vmax_abs_diff(gW, fx.vec("fixed_grad_p_label_W")), 0.0, 1e-10)
    _ = log.close("dp/dlast max abs diff", vmax_abs_diff(gL, fx.vec("fixed_grad_p_label_last")), 0.0, 1e-10)
    var full = Circuit(6, 0, "qcnn_mottonen")
    append_amplitude_embedding_real(full, feats[0], [5, 4, 3, 2, 1, 0])
    full.compose(c)
    var e2 = ExpectationEngine(full, parse_pauli_sum("0.5 + 0.5 * Z5"))
    var g2 = List[Float64]()
    var pm = e2.value_and_gradient(p, g2, Grad.ADJOINT)
    _ = log.close("Mottonen-prepared circuit: value and gradient max abs diff",
                  max(abs(pm - p0), vmax_abs_diff(g2, g)), 0.0, 1e-12)
    print("  state preparation: Mottonen tree", full.num_2q_gates() - c.num_2q_gates(), "CNOTs, or 0 gates as an initial state")

    # training: 4 sizes x 10 repetitions x 100 epochs
    var sizes = fx.ivec("train_sizes")
    var nrep = fx.int("n_reps")
    var nep = fx.int("n_epochs")
    var max_cost = 0.0
    var acc_mismatch = 0
    var t0 = perf_counter_ns()
    var mean40 = 0.0
    var run = 0
    for s in sizes:
        var mean_te = 0.0
        for _ in range(nrep):
            var key = "runs/" + String(run) + "/"
            var tr = fx.ivec(key + "train_idx")
            var te = fx.ivec(key + "test_idx")
            var w = fx.vec(key + "init_W")
            for v in fx.vec(key + "init_last"):
                w.append(v)
            var ref_cost = fx.vec(key + "train_cost")
            var opt = StepOptimizer.adam(0.1, 0.9, 0.999, 1e-8, "optax")
            var costs = List[Float64]()
            for ep in range(nep):
                var r = outputs(c, w, states, labels, tr, True)
                var m = 0.0
                for v in r[0]:
                    m += v
                costs.append(1.0 - m / Float64(len(tr)))
                opt.set_lr(cosine_lr(ep, nep, 0.1, 0.095))
                opt.step(w, r[1])
            max_cost = max(max_cost, vmax_abs_diff(costs, ref_cost))
            var a_tr = acc(outputs(c, w, states, labels, tr, False)[0])
            var a_te = acc(outputs(c, w, states, labels, te, False)[0])
            if a_tr != fx.num(key + "final_train_acc") or a_te != fx.num(key + "final_test_acc"):
                acc_mismatch += 1
            mean_te += a_te
            run += 1
        mean_te /= Float64(nrep)
        print("  n_train", s, " mean test accuracy", mean_te, " (fixture", fx.num("mean_test_acc/" + String(s)), ")")
        if s == 40:
            mean40 = mean_te
    var secs = seconds_since(t0)
    _ = log.close("40 training-cost trajectories max abs diff", max_cost, 0.0, 1e-8)
    _ = log.check("40 final train/test accuracies equal the fixture", acc_mismatch == 0, String(acc_mismatch) + " mismatches")
    _ = log.within("mean test accuracy with 40 training images", mean40, 0.9, 1.0)
    print("timing (indicative, different engines): QUMO", fmt_s(secs), "s for 40 training runs; PennyLane default.qubit + JAX jit",
          fmt_s(fx.num("train_seconds")), "s (fixture)")
    log.finish()
