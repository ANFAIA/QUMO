"""Data re-uploading: a single-qubit universal classifier for points inside a circle.

Reproduces: PennyLane - Data-reuploading classifier
Source: https://pennylane.ai/qml/demos/tutorial_data_reuploading_classifier
Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_data_reuploading_classifier/demo.py (Apache-2.0)
Reproduces: Qibo - Data reuploading for a universal quantum classifier
Source: https://qibo.science/qibo/stable/code-examples/tutorials/reuploading_classifier/README.html

One qubit, three layers of [Rot(x1, x2, 0), Rot(p_l)]: the data point is
"re-uploaded" before every trainable rotation. The label states are |0>
(outside the circle) and |1> (inside); the loss is the mean of (1 - F)^2 with
F = <y|rho|y> (QUMO: the projectors (1 +/- Z)/2). PennyLane-form Adam(0.6,
beta2 = 0.999), minibatches of 32 in order (iterate_minibatches), 10 epochs,
on the fixture's circle data (200 train / 2000 test points drawn by NumPy).

Checks, against examples/qml/reference/pennylane_reference.json (PennyLane 0.45.1):
  * initial fidelity of training point 0 with its label (1e-12);
  * epoch-0 accuracies equal the fixture;
  * the 60 minibatch losses (first 20 at 1e-10, all at 1e-8) and the final parameters (1e-8);
  * per-epoch train/test accuracies and the 2000 final test predictions equal the fixture.
The page's 0.870 / 0.779 accuracies come from another NumPy draw order and are
quoted, not asserted (fixture rerun: 0.89 / 0.847).

Run: .venv/bin/mojo run -I . examples/qml/data_reuploading.mojo
example-meta: {"framework": "pennylane", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.algorithms.gradients import ExpectationEngine, Grad, batch_value_and_gradient, batch_expectation
from qumo.algorithms.qml import append_rot
from qumo.algorithms.qml_train import StepOptimizer, fidelity_loss
from qumo.ai.checks import CheckLog

from _common import Fixture, vmax_abs_diff, seconds_since, fmt_s


comptime NW = 9  # 3 layers x Rot(3)


def reuploading_circuit() raises -> Circuit:
    var c = Circuit(1, 0, "reuploading_classifier")
    var p = c.parameters("p", NW)  # weights first
    var x = c.parameters("x", 3)   # then the point (x1, x2, 0), re-uploaded in every layer
    for l in range(3):
        append_rot(c, x[0], x[1], x[2], 0)
        append_rot(c, p[3 * l], p[3 * l + 1], p[3 * l + 2], 0)
    return c^


def pset(w: List[Float64], x: List[Float64]) -> List[Float64]:
    var p = w.copy()
    p.append(x[0])
    p.append(x[1])
    p.append(0.0)
    return p^


def predict(c: Circuit, h0: PauliSum, w: List[Float64], X: List[List[Float64]]) raises -> List[Float64]:
    """Argmax over the two label fidelities: F(|1>) = 1 - F(|0>), ties go to label 0."""
    var sets = List[List[Float64]]()
    for x in X:
        sets.append(pset(w, x))
    var f0 = batch_expectation(c, h0, sets)
    var out = List[Float64]()
    for f in f0:
        out.append(1.0 if (1.0 - f) > f else 0.0)
    return out^


def acc(pred: List[Float64], y: List[Float64]) -> Float64:
    var ok = 0
    for i in range(len(y)):
        if pred[i] == y[i]:
            ok += 1
    return Float64(ok) / Float64(len(y))


def main() raises:
    var log = CheckLog("data_reuploading")
    var fx = Fixture("data_reuploading")
    var Xtr = fx.mat("X_train")
    var ytr = fx.vec("y_train")
    var Xte = fx.mat("X_test")
    var yte = fx.vec("y_test")
    var w = fx.vec("init_params")
    var c = reuploading_circuit()
    var h0 = parse_pauli_sum("0.5 + 0.5 * Z0")  # |0><0|
    var h1 = parse_pauli_sum("0.5 - 0.5 * Z0")  # |1><1|
    var hs: List[PauliSum] = [h0.copy(), h1.copy()]

    var e = ExpectationEngine(c, hs[Int(ytr[0])])
    _ = log.close("initial fidelity, train point 0", e.value(pset(w, Xtr[0])), fx.num("init_fidelity_x0_label_y0"), 1e-12)
    _ = log.close("epoch-0 train accuracy", acc(predict(c, h0, w, Xtr), ytr), fx.num("epoch0_train_acc"), 0.0)
    _ = log.close("epoch-0 test accuracy", acc(predict(c, h0, w, Xte), yte), fx.num("epoch0_test_acc"), 0.0)

    var opt = StepOptimizer.adam(0.6, 0.9, 0.999, 1e-8, "pennylane")
    var bs = 32
    var losses = List[Float64]()
    var ep_tr = List[Float64]()
    var ep_te = List[Float64]()
    var pte = List[Float64]()
    var t0 = perf_counter_ns()
    for epoch in range(10):
        var start = 0
        while start + bs <= len(Xtr):  # iterate_minibatches: in order, last partial batch dropped
            # group the batch by label so each group is one batched adjoint call
            var grad = List[Float64](length=NW, fill=0.0)
            var fids = List[Float64](length=bs, fill=0.0)
            var grads = List[List[Float64]](length=bs, fill=List[Float64]())
            for lab in range(2):
                var sets = List[List[Float64]]()
                var pos = List[Int]()
                for i in range(start, start + bs):
                    if Int(ytr[i]) == lab:
                        sets.append(pset(w, Xtr[i]))
                        pos.append(i - start)
                if len(sets) == 0:
                    continue
                var r = batch_value_and_gradient(c, hs[lab], sets)
                for k in range(len(pos)):
                    fids[pos[k]] = r[0][k]
                    grads[pos[k]] = r[1][k].copy()
            var L = fidelity_loss(fids)
            var dl = L[1].copy()
            for b in range(bs):
                for k in range(NW):
                    grad[k] += dl[b] * grads[b][k]
            losses.append(L[0])  # step_and_cost: the cost before the step
            opt.step(w, grad)
            start += bs
        var ptr = predict(c, h0, w, Xtr)
        pte = predict(c, h0, w, Xte)
        ep_tr.append(acc(ptr, ytr))
        ep_te.append(acc(pte, yte))
        print("  epoch", epoch + 1, " last batch loss", losses[len(losses) - 1], " train acc", ep_tr[epoch], " test acc", ep_te[epoch])
    var secs = seconds_since(t0)

    var ref_losses = fx.vec("step_losses")
    var first20 = List[Float64]()
    var ref20 = List[Float64]()
    for i in range(20):
        first20.append(losses[i])
        ref20.append(ref_losses[i])
    _ = log.close("first 20 minibatch losses max abs diff", vmax_abs_diff(first20, ref20), 0.0, 1e-10)
    _ = log.close("all 60 minibatch losses max abs diff", vmax_abs_diff(losses, ref_losses), 0.0, 1e-8)
    _ = log.close("final parameters max abs diff", vmax_abs_diff(w, fx.vec("final_params")), 0.0, 1e-8)
    _ = log.check("per-epoch train accuracies equal fixture", vmax_abs_diff(ep_tr, fx.vec("epoch_train_acc")) == 0.0, "")
    _ = log.check("per-epoch test accuracies equal fixture", vmax_abs_diff(ep_te, fx.vec("epoch_test_acc")) == 0.0, "")
    _ = log.check("2000 final test predictions equal fixture", vmax_abs_diff(pte, fx.vec("final_test_predictions")) == 0.0, "")
    _ = log.close("final train accuracy", ep_tr[9], fx.num("final_train_acc"), 0.0)
    _ = log.close("final test accuracy", ep_te[9], fx.num("final_test_acc"), 0.0)
    print("page values (quoted, not asserted): train 0.870, test 0.779; fixture rerun:",
          fx.num("final_train_acc"), "/", fx.num("final_test_acc"))

    # what the same workflow costs on a 1-qubit QPU (no device is contacted)
    var shots = 1000
    var train_runs = len(losses) * bs * (1 + 2 * NW)
    print("QPU cost model (1 qubit,", shots, "shots per circuit):")
    print("  training by parameter shift:", len(losses), "steps x", bs, "points x (1 + 2 x", NW, ") =", train_runs, "circuit runs")
    print("  inference on the test set:", len(Xte), "circuit runs (F(|1>) = 1 - F(|0>) from the same counts),",
          len(Xte) * shots, "shots")
    print("timing (indicative, different engines): QUMO", fmt_s(secs), "s for 10 epochs; PennyLane default.qubit",
          fmt_s(fx.num("train_seconds")), "s (fixture)")
    log.finish()
