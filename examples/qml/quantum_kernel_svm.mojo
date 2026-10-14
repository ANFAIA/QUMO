"""Kernel-based training: a quantum-kernel SVM on Iris, and what it costs on a QPU.

Reproduces: PennyLane - Kernel-based training of quantum models with scikit-learn
Source: https://pennylane.ai/qml/demos/tutorial_kernel_based_training
Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_kernel_based_training/demo.py (Apache-2.0)

Iris classes 0 and 1 (100 points), standard-scaled, split 75/25 with the
fixture's sklearn train_test_split(random_state=42) indices. The kernel is the
fidelity k(a, b) = |<0| U(b)^dagger U(a) |0>|^2 of a 4-qubit AngleEmbedding,
computed two ways: QUMO's state route (kernel_matrix: one embedding per point,
then overlaps) and the hardware-style compute-uncompute circuit
(kernel_circuit + the |0000> projector), plus shot-sampled entries (the QPU
route). An IQPEmbedding(n_repeats=2) kernel is the second model. QUMO's own
C-SVC (libsvm SMO, C = 1) replaces sklearn's SVC(kernel="precomputed").

Checks, against examples/qml/reference/pennylane_reference.json (PennyLane 0.45.1 + sklearn 1.9.1):
  * scaler mean; K_train sum = 1849.5885793209322 (1e-9), first rows (1e-12);
  * compute-uncompute circuit == state route (1e-12);
  * IQP kernel entry 0.280132222844837 (1e-12) and Gram-matrix sum (1e-9);
  * SVC against sklearn: same number of support vectors, dual objective (relative 1e-4),
    intercept (2e-3) and test decision values (5e-3): both are SMO solvers stopped at
    libsvm's KKT tolerance 1e-3, so coefficients agree only to that order;
    test accuracy 1.0 for both kernels;
  * sampled entries within 4 standard errors of the exact kernel;
  * the demo's cost model: 7500 kernel circuit runs against 96025 for the
    variational model (24 x 100 x 20 x 2 + 25; the page text says 74025).

Run: .venv/bin/mojo run -I . examples/qml/quantum_kernel_svm.mojo
example-meta: {"framework": "pennylane", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import sqrt, ceil
from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.algorithms.datasets import load_csv, standard_scale, take_rows, take_values
from qumo.algorithms.qml import append_angle_embedding, iqp_embedding, iqp_feature_values
from qumo.algorithms.qkernels import (
    kernel_matrix, kernel_matrix_symmetric, kernel_circuit, kernel_circuit_params, kernel_value_and_grad,
    sampled_kernel_entry, svc_fit, svc_decision, svc_predict, SVCModel,
)
from qumo.testing.checks import CheckLog

from _common import Fixture, vmax_abs_diff, seconds_since, fmt_s, DATA_DIR


def total(v: List[Float64]) -> Float64:
    var s = 0.0
    for x in v:
        s += x
    return s


def row(K: List[Float64], i: Int, n: Int) -> List[Float64]:
    var r = List[Float64]()
    for j in range(n):
        r.append(K[i * n + j])
    return r^


def dual_objective(dc: List[Float64], K: List[Float64], n: Int) -> Float64:
    """C-SVC dual objective sum_i alpha_i - 1/2 sum_ij (alpha y)_i (alpha y)_j K_ij."""
    var w = 0.0
    for i in range(n):
        w += abs(dc[i])
        for j in range(n):
            w -= 0.5 * dc[i] * dc[j] * K[i * n + j]
    return w


def check_svm(mut log: CheckLog, name: String, Ktr: List[Float64], Kte: List[Float64],
              ytr: List[Float64], yte: List[Float64], fx: Fixture, key: String) raises:
    var m = svc_fit(Ktr, ytr, 1.0)
    var dec = svc_decision(m, Kte)
    var pred = svc_predict(m, Kte)
    var ok = 0
    for i in range(len(yte)):
        if pred[i] == yte[i]:
            ok += 1
    var acc = Float64(ok) / Float64(len(yte))
    print("  ", name, "SVC:", m.num_support(), "support vectors, test accuracy", acc)
    var sup = fx.ivec(key + "/support")
    var dc = fx.vec(key + "/dual_coef")
    var n = len(ytr)
    var sk = List[Float64](length=n, fill=0.0)  # sklearn's dual_coef_ scattered over the training points
    for k in range(len(sup)):
        sk[sup[k]] = dc[k]
    var md = vmax_abs_diff(m.dual_coef, sk)
    var w_q = dual_objective(m.dual_coef, Ktr, n)
    var w_s = dual_objective(sk, Ktr, n)
    var tight = svc_fit(Ktr, ytr, 1.0, 1e-10)
    var w_t = dual_objective(tight.dual_coef, Ktr, n)
    print("   ", name, "dual objective: QUMO", w_q, " sklearn", w_s, " QUMO tol 1e-10", w_t,
          "; max |dual coef diff| vs sklearn", md)
    # Both solvers are libsvm-style SMO stopped at the KKT tolerance 1e-3, so their
    # dual coefficients agree only to O(1e-2); the objective and decisions agree closely.
    _ = log.check(name + " support vectors equal sklearn's", m.num_support() == len(sup),
                  "qumo=" + String(m.num_support()) + " sklearn=" + String(len(sup)))
    _ = log.close(name + " dual objective vs sklearn (relative)", (w_q - w_s) / abs(w_s), 0.0, 1e-4)
    _ = log.within(name + " tighter solve does not lose objective", w_t - max(w_q, w_s), -1e-9, 1e-3)
    _ = log.close(name + " intercept vs sklearn", m.intercept, fx.num(key + "/intercept"), 2e-3)
    _ = log.close(name + " test decision values max abs diff vs sklearn", vmax_abs_diff(dec, fx.vec(key + "/test_decision")), 0.0, 5e-3)
    _ = log.close(name + " test accuracy", acc, fx.num(key + "/test_acc"), 0.0)


def main() raises:
    var log = CheckLog("quantum_kernel_svm")
    var fx = Fixture("quantum_kernel_svm")

    var d = load_csv(DATA_DIR + "iris.csv")
    var Xall = d[0].copy()
    var yall = d[1].copy()
    var X100 = List[List[Float64]]()
    var y100 = List[Float64]()
    for i in range(100):
        X100.append(Xall[i].copy())
        y100.append(2.0 * (yall[i] - 0.5))
    var ss = standard_scale(X100)
    var Xs = ss[0].copy()
    _ = log.close("scaler mean max abs diff", vmax_abs_diff(ss[1], fx.vec("scaler_mean")), 0.0, 1e-12)
    var itr = fx.ivec("train_idx")
    var ite = fx.ivec("test_idx")
    var Xtr = take_rows(Xs, itr)
    var Xte = take_rows(Xs, ite)
    var ytr = take_values(y100, itr)
    var yte = take_values(y100, ite)
    var M = len(Xtr)

    # AngleEmbedding kernel, state route
    var embed = Circuit(4, 0, "angle")
    var xf = embed.parameters("x", 4)
    append_angle_embedding(embed, xf, [0, 1, 2, 3])
    var t0 = perf_counter_ns()
    var Ktr = kernel_matrix_symmetric(embed, Xtr)
    var t_k = seconds_since(t0)
    var Kte = kernel_matrix(embed, Xte, Xtr)
    print("angle kernel: K_train", M, "x", M, "in", fmt_s(t_k), "s;  k(tr0, tr1) =", Ktr[1])
    _ = log.close("angle k(tr0, tr1)", Ktr[1], fx.num("angle_k_tr0_tr1"), 1e-12)
    _ = log.close("angle K_train sum", total(Ktr), 1849.5885793209322, 1e-9)
    _ = log.close("angle K_train row 0 max abs diff", vmax_abs_diff(row(Ktr, 0, M), fx.vec("angle_Ktrain_row0")), 0.0, 1e-12)
    _ = log.close("angle K_test row 0 max abs diff", vmax_abs_diff(row(Kte, 0, M), fx.vec("angle_Ktest_row0")), 0.0, 1e-12)

    # same kernel, compute-uncompute circuit (what a QPU runs)
    var kc = kernel_circuit(embed, 4)
    var none = List[Float64]()
    var md = 0.0
    for i in range(0, M, 7):
        for j in range(0, M, 11):
            var kv = kernel_value_and_grad(kc, none, Xtr[i], Xtr[j])
            md = max(md, abs(kv[0] - Ktr[i * M + j]))
    _ = log.close("compute-uncompute circuit vs state route max abs diff", md, 0.0, 1e-12)
    var shots = 20000
    var worst = 0.0
    var pairs: List[Int] = [0, 1, 3, 10, 20, 40, 5, 70]
    for k in range(0, len(pairs), 2):
        var a = pairs[k]
        var b = pairs[k + 1]
        var exact = Ktr[a * M + b]
        var est = sampled_kernel_entry(kc, kernel_circuit_params(none, Xtr[a], Xtr[b]), shots, UInt64(100 + k))
        var se = sqrt(max(exact * (1.0 - exact), 1e-12) / Float64(shots))
        worst = max(worst, abs(est - exact) / se)
        print("  sampled k(tr", a, ", tr", b, ") =", est, " exact", exact, " (", shots, "shots )")
    _ = log.within("sampled entries: worst deviation in standard errors", worst, 0.0, 4.0)

    check_svm(log, "angle", Ktr, Kte, ytr, yte, fx, "angle_svm")

    # IQP kernel (n_repeats = 2): products x_i x_j become extra feature parameters
    var iqp = iqp_embedding(4, 2)
    var Itr = List[List[Float64]]()
    var Ite = List[List[Float64]]()
    for x in Xtr:
        Itr.append(iqp_feature_values(x))
    for x in Xte:
        Ite.append(iqp_feature_values(x))
    t0 = perf_counter_ns()
    var K2tr = kernel_matrix_symmetric(iqp, Itr)
    var t_iqp = seconds_since(t0)
    var K2te = kernel_matrix(iqp, Ite, Itr)
    print("IQP kernel: K_train in", fmt_s(t_iqp), "s;  k(tr0, tr1) =", K2tr[1])
    _ = log.close("IQP k(tr0, tr1)", K2tr[1], 0.280132222844837, 1e-12)
    _ = log.close("IQP K_train sum", total(K2tr), fx.num("iqp_Ktrain_sum"), 1e-9)
    _ = log.close("IQP K_train row 0 max abs diff", vmax_abs_diff(row(K2tr, 0, M), fx.vec("iqp_Ktrain_row0")), 0.0, 1e-12)
    check_svm(log, "IQP", K2tr, K2te, ytr, yte, fx, "iqp_svm")

    # the demo's resource argument: kernel methods need far fewer circuit runs
    var Mtr = Int(ceil(0.75 * 100.0))
    var Mpred = 100 - Mtr
    var kernel_runs = Mtr * Mtr + Mtr * Mpred
    var var_runs = 24 * 100 * 20 * 2 + Mpred
    print("circuit-run cost model: kernel", kernel_runs, "(M^2 + M Mpred);  variational", var_runs,
          "(24 params x 100 steps x batch 20 x 2 shifts + 25)")
    _ = log.close("kernel cost model (circuit runs)", Float64(kernel_runs), Float64(fx.int("cost_model/kernel_runs")), 0.0)
    _ = log.close("variational cost model (circuit runs)", Float64(var_runs), Float64(fx.int("cost_model/variational_runs")), 0.0)
    print("  (the page text quotes 74025 for the variational model; its own formula gives 96025)")
    print("timing (indicative, different engines): QUMO K_train angle", fmt_s(t_k), "s, IQP", fmt_s(t_iqp),
          "s; PennyLane default.qubit", fmt_s(fx.num("angle_Ktrain_seconds")), "s and",
          fmt_s(fx.num("iqp_Ktrain_seconds")), "s (fixture)")
    log.finish()
