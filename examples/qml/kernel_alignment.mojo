"""Training a quantum embedding kernel by kernel-target alignment (DoubleCake data).

Reproduces: PennyLane - Training and evaluating quantum kernels
Source: https://pennylane.ai/qml/demos/tutorial_kernels_module
Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_kernels_module/demo.py (Apache-2.0)

DoubleCake (6 points, two rings of 3 sectors). The embedding is 6 layers on 5
qubits of [H, RZ(x_(i mod 2)), RY(theta)] on every wire plus a ring of CRZ(theta)
(60 trainable angles, the demo's seed-1359 uniform initialisation from the
fixture). The kernel is the |00000> probability of U(x1) U(x2)^dagger (QUMO
kernel_circuit), and training maximises the kernel-target alignment
KTA = <K, y y^T> / (||K|| ||y y^T||) (labels rescaled by class sizes) with
500 gradient-descent(0.2) steps, each on 4 points drawn with replacement
(the fixture's subsets are replayed). Gradients: one adjoint pass per kernel
entry (kernel_matrix_symmetric_and_grad) chained through target_alignment_grad.

Checks, against examples/qml/reference/pennylane_reference.json (PennyLane 0.45.1 + sklearn 1.9.1):
  * K_init (1e-12), k01 = 0.029983453404801606, KTA_init = 0.07433831454242083,
    dk01/dtheta and dKTA/dtheta (1e-12);
  * alignment every 50 steps, from 0.0904 to 0.2661 (1e-6), final parameters (1e-8);
  * trained Gram matrix (1e-8);
  * SVM (QUMO C-SVC, C = 1) training accuracy 0.833 before and 1.000 after training.

Run: .venv/bin/mojo run -I . examples/qml/kernel_alignment.mojo
example-meta: {"framework": "pennylane", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.algorithms.datasets import double_cake, take_rows, take_values
from qumo.algorithms.qkernels import (
    kernel_matrix_symmetric, kernel_circuit, kernel_value_and_grad, kernel_matrix_symmetric_and_grad,
    target_alignment, target_alignment_grad, svc_fit, svc_decision, svc_predict,
)
from qumo.testing.checks import CheckLog

from _common import Fixture, vmax_abs_diff, seconds_since, fmt_s


def doublecake_embedding() raises -> Circuit:
    """Params: theta (60, layout l*10 + r*5 + j: r = 0 RY, r = 1 CRZ ring), then x (2)."""
    var c = Circuit(5, 0, "doublecake_embedding")
    var th = c.parameters("th", 60)
    var x = c.parameters("x", 2)
    for l in range(6):
        var i = l * 5
        for j in range(5):
            c.h(j)
            c.rz(x[i % 2], j)
            i += 1
            c.ry(th[l * 10 + j], j)
        for j in range(5):
            c.crz(th[l * 10 + 5 + j], j, (j + 1) % 5)
    return c^


def svm_train_accuracy(K: List[Float64], Y: List[Float64]) raises -> Tuple[Float64, List[Float64]]:
    var m = svc_fit(K, Y, 1.0)
    var pred = svc_predict(m, K)
    var ok = 0
    for i in range(len(Y)):
        if pred[i] == Y[i]:
            ok += 1
    return (Float64(ok) / Float64(len(Y)), svc_decision(m, K))


def main() raises:
    var log = CheckLog("kernel_alignment")
    var fx = Fixture("kernel_alignment")
    var dc = double_cake(3)
    var X = dc[0].copy()
    var Y = dc[1].copy()
    _ = log.close("DoubleCake points max abs diff", vmax_abs_diff(fx.vec("X"), _flat(X)), 0.0, 1e-15)
    var embed = doublecake_embedding()
    var kc = kernel_circuit(embed, 2)
    var theta = fx.vec("init_params")

    # initial kernel, alignment and gradients
    var r = kernel_matrix_symmetric_and_grad(kc, theta, X)
    var K = r[0].copy()
    _ = log.close("K_init max abs diff", vmax_abs_diff(K, fx.vec("K_init")), 0.0, 1e-12)
    _ = log.close("K_init state route == circuit route", vmax_abs_diff(K, kernel_matrix_symmetric(embed, X, theta)), 0.0, 1e-13)
    var kv = kernel_value_and_grad(kc, theta, X[0], X[1])
    _ = log.close("k01_init", kv[0], 0.029983453404801606, 1e-12)
    _ = log.close("dk01/dtheta max abs diff", vmax_abs_diff(kv[1], fx.vec("dk01_dtheta_init")), 0.0, 1e-12)
    var kta0 = target_alignment(K, Y, True)
    print("initial kernel-target alignment", kta0, "(page 0.074)")
    _ = log.close("KTA_init", kta0, 0.07433831454242083, 1e-12)
    _ = log.close("dKTA/dtheta max abs diff", vmax_abs_diff(target_alignment_grad(K, r[1], Y, True), fx.vec("dkta_dtheta_init")), 0.0, 1e-12)
    var s0 = svm_train_accuracy(K, Y)
    print("SVM training accuracy with the initial kernel", s0[0])
    _ = log.close("SVM accuracy, initial kernel", s0[0], fx.num("svm_acc_init"), 1e-12)
    _ = log.close("SVM decision values, initial kernel, vs sklearn", vmax_abs_diff(s0[1], fx.vec("svm_decision_init")), 0.0, 5e-3)

    # 500 GD(0.2) steps on -KTA of 4-point subsets (the fixture's draws)
    var subsets = fx.imat("subsets")
    var align = List[Float64]()
    var t0 = perf_counter_ns()
    for i in range(500):
        var Xs = take_rows(X, subsets[i])
        var Ys = take_values(Y, subsets[i])
        var rs = kernel_matrix_symmetric_and_grad(kc, theta, Xs)
        var g = target_alignment_grad(rs[0], rs[1], Ys, True)
        for k in range(len(theta)):
            theta[k] += 0.2 * g[k]  # gradient descent on -KTA
        if (i + 1) % 50 == 0:
            var a = target_alignment(kernel_matrix_symmetric(embed, X, theta), Y, True)
            align.append(a)
            print("  step", i + 1, " alignment", a)
    var secs = seconds_since(t0)
    _ = log.close("alignment after 50 steps (page 0.0904)", align[0], fx.vec("alignment_every50")[0], 1e-6)
    _ = log.close("alignment after 500 steps (page 0.2661)", align[9], fx.vec("alignment_every50")[9], 1e-6)
    _ = log.close("alignment every 50 steps max abs diff", vmax_abs_diff(align, fx.vec("alignment_every50")), 0.0, 1e-6)
    _ = log.close("final parameters max abs diff", vmax_abs_diff(theta, fx.vec("final_params")), 0.0, 1e-8)
    var Kt = kernel_matrix_symmetric(embed, X, theta)
    _ = log.close("trained Gram matrix max abs diff", vmax_abs_diff(Kt, fx.vec("K_trained")), 0.0, 1e-8)
    var s1 = svm_train_accuracy(Kt, Y)
    print("SVM training accuracy with the trained kernel", s1[0])
    _ = log.close("SVM accuracy, trained kernel", s1[0], fx.num("svm_acc_trained"), 1e-12)
    _ = log.check("training improved the SVM", s1[0] > s0[0], String(s0[0]) + " -> " + String(s1[0]))
    print("timing (indicative, different engines): QUMO", fmt_s(secs), "s for 500 alignment steps; PennyLane default.qubit",
          fmt_s(fx.num("train_seconds")), "s (fixture; the page quotes about 317 s)")
    log.finish()


def _flat(X: List[List[Float64]]) -> List[Float64]:
    var out = List[Float64]()
    for row in X:
        for v in row:
            out.append(v)
    return out^
