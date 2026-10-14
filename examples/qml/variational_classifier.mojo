"""Variational classifier: parity of bitstrings and Iris classes, plus a Qibo-style 3-class Iris model.

Reproduces: PennyLane - Variational classifier
Source: https://pennylane.ai/qml/demos/tutorial_variational_classifier
Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_variational_classifier/demo.py (Apache-2.0)
Reproduces: Qibo - Variational quantum classifier (3-class Iris, 4 qubits, 5 layers)
Source: https://qibo.science/qibo/stable/code-examples/tutorials/variational_classifier/README.html

Part 1, parity (4 qubits): BasisState(x) (QUMO: RX(pi x)), two layers of Rot on
every wire plus a CNOT ring, prediction <Z0> + bias, square loss, Nesterov
momentum. Part 2, Iris classes 1 and 2 (2 qubits): the demo's get_angles
amplitude-encoding circuit (equal to QUMO's mottonen_angles), six Rot layers.
Part 3, Qibo-style: all three Iris classes on 4 qubits, 5 layers of RY plus a
CZ chain, classes mapped to the basis states |00>, |01>, |10> of qubits 0 and 1,
quadratic loss on those three probabilities.

Checks, against examples/qml/reference/pennylane_reference.json (PennyLane 0.45.1):
  * parity forward 0.9065742515591111 and its 24 gradients (1e-12);
  * 30-step full-batch Nesterov(0.5) cost trajectory from 1.9992624749350514 (1e-10);
  * stochastic run (batch 5, 100 iterations, the fixture's batch indices replayed):
    training cost trajectory (1e-8), final cost, accuracy 1.0 on train and test
    (page cost 1.74e-5 is quoted, the fixture is asserted);
  * Iris: get_angles == mottonen_angles (1e-11), state-preparation amplitudes,
    <Z0> = 0.45177693213489395 and gradients with fixed weights (1e-12),
    60 Nesterov(0.01) iterations replayed (1e-8), final validation accuracy 1.0;
  * Qibo-style 3-class Iris: test accuracy >= 0.9 (Qibo's page reports about 0.73
    with Powell and sampling; QUMO uses exact adjoint gradients and Adam).

Run: .venv/bin/mojo run -I . examples/qml/variational_classifier.mojo
example-meta: {"framework": "pennylane", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import pi, sqrt, asin
from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.ir.param import Param
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.sim.runner import evolve
from qumo.algorithms.gradients import ExpectationEngine, Grad, batch_value_and_gradient, batch_expectation
from qumo.algorithms.datasets import parity_dataset, load_csv, train_test_split, take_rows, take_values
from qumo.algorithms.qml import append_rot, append_basis_embedding, append_angle_embedding, mottonen_angles
from qumo.algorithms.qml_train import StepOptimizer, square_loss, expectations, vjp
from qumo.ai.checks import CheckLog

from _common import Fixture, load_table, vmax_abs_diff, seconds_since, fmt_s, DATA_DIR


# ---------------------------------------------------------------------------
# Shared training helpers: prediction = <h>(weights, x) + bias
# ---------------------------------------------------------------------------
def make_sets(theta: List[Float64], nw: Int, X: List[List[Float64]], idx: List[Int]) -> List[List[Float64]]:
    """Parameter vectors weights ++ features for the rows `idx` (theta = weights ++ [bias])."""
    var sets = List[List[Float64]]()
    for i in idx:
        var p = List[Float64](capacity=nw + len(X[i]))
        for k in range(nw):
            p.append(theta[k])
        for v in X[i]:
            p.append(v)
        sets.append(p^)
    return sets^


def all_rows(n: Int) -> List[Int]:
    var r = List[Int]()
    for i in range(n):
        r.append(i)
    return r^


def cost_and_grad(c: Circuit, h: PauliSum, theta: List[Float64], nw: Int,
                  X: List[List[Float64]], Y: List[Float64], idx: List[Int]) raises -> Tuple[Float64, List[Float64]]:
    """Mean square loss over rows `idx` and its gradient w.r.t. weights ++ [bias]."""
    var r = batch_value_and_gradient(c, h, make_sets(theta, nw, X, idx))
    var preds = List[Float64]()
    var ys = List[Float64]()
    for b in range(len(idx)):
        preds.append(r[0][b] + theta[nw])
        ys.append(Y[idx[b]])
    var L = square_loss(preds, ys)
    var dl = L[1].copy()
    var g = List[Float64](length=nw + 1, fill=0.0)
    for b in range(len(idx)):
        for k in range(nw):
            g[k] += dl[b] * r[1][b][k]
        g[nw] += dl[b]
    return (L[0], g^)


def outputs(c: Circuit, h: PauliSum, theta: List[Float64], nw: Int, X: List[List[Float64]]) raises -> List[Float64]:
    var v = batch_expectation(c, h, make_sets(theta, nw, X, all_rows(len(X))))
    for ref x in v:
        x += theta[nw]
    return v^


def mse_and_acc(vals: List[Float64], Y: List[Float64], idx: List[Int]) -> Tuple[Float64, Float64]:
    """Mean square error and sign accuracy (|label - sign(out)| < 1e-5) over rows `idx`."""
    var s = 0.0
    var ok = 0
    for i in idx:
        s += (Y[i] - vals[i]) ** 2
        var sg = 1.0 if vals[i] > 0.0 else (-1.0 if vals[i] < 0.0 else 0.0)
        if abs(Y[i] - sg) < 1e-5:
            ok += 1
    return (s / Float64(len(idx)), Float64(ok) / Float64(len(idx)))


# ---------------------------------------------------------------------------
# Part 1: parity
# ---------------------------------------------------------------------------
def parity_circuit() raises -> Circuit:
    var c = Circuit(4, 0, "parity_classifier")
    var w = c.parameters("w", 24)  # weights first: w[l][q][k] at 12 l + 3 q + k
    var x = c.parameters("x", 4)   # then the bit features
    var wires: List[Int] = [0, 1, 2, 3]
    append_basis_embedding(c, x, wires)
    for l in range(2):
        for q in range(4):
            var o = l * 12 + q * 3
            append_rot(c, w[o], w[o + 1], w[o + 2], q)
        c.cx(0, 1)
        c.cx(1, 2)
        c.cx(2, 3)
        c.cx(3, 0)
    return c^


def part_parity(mut log: CheckLog, fx: Fixture) raises:
    print("== parity (4 qubits, 2 layers)")
    var c = parity_circuit()
    var h = parse_pauli_sum("Z0")
    var p = List[Float64]()
    for l in range(2):
        for q in range(4):
            for k in range(3):
                p.append(0.1 * Float64(l + 1) + 0.01 * Float64(q) + 0.001 * Float64(k))
    for b in fx.vec("parity_fixed_x"):
        p.append(b)
    var eng = ExpectationEngine(c, h)
    var g = List[Float64]()
    var e = eng.value_and_gradient(p, g, Grad.ADJOINT)
    _ = log.close("parity forward <Z0> (x=0101)", e, 0.9065742515591111, 1e-12)
    var g24 = List[Float64]()
    for k in range(24):
        g24.append(g[k])
    _ = log.close("parity weight gradients max abs diff", vmax_abs_diff(g24, fx.vec("parity_fixed_grad")), 0.0, 1e-12)

    # 30 full-batch Nesterov(0.5) steps on all 16 strings; cost at the look-ahead point
    var ds = parity_dataset(4)
    var X16 = ds[0].copy()
    var Y16 = ds[1].copy()
    var theta = fx.vec("parity_init_weights")
    theta.append(0.0)
    var init = theta.copy()
    var opt = StepOptimizer.nesterov(0.5)
    var traj = List[Float64]()
    var t0 = perf_counter_ns()
    for _ in range(30):
        var la = opt.lookahead(theta)
        var r = cost_and_grad(c, h, la, 24, X16, Y16, all_rows(16))
        traj.append(r[0])
        opt.step(theta, r[1])
    var t_full = seconds_since(t0)
    _ = log.close("full-batch Nesterov first cost", traj[0], 1.9992624749350514, 1e-12)
    _ = log.close("full-batch Nesterov 30-step trajectory max abs diff", vmax_abs_diff(traj, fx.vec("parity_fullbatch_cost_traj")), 0.0, 1e-10)
    var w_ref = fx.vec("parity_fullbatch_final_weights")
    w_ref.append(fx.num("parity_fullbatch_final_bias"))
    _ = log.close("full-batch final weights max abs diff", vmax_abs_diff(theta, w_ref), 0.0, 1e-10)

    # the demo's stochastic run on parity_train.txt with the fixture's batches
    var tr = load_table(DATA_DIR + "parity_train.txt")
    var te = load_table(DATA_DIR + "parity_test.txt")
    var Xtr = List[List[Float64]]()
    var Ytr = List[Float64]()
    for row in tr:
        Xtr.append([row[0], row[1], row[2], row[3]])
        Ytr.append(2.0 * row[4] - 1.0)
    var Xte = List[List[Float64]]()
    var Yte = List[Float64]()
    for row in te:
        Xte.append([row[0], row[1], row[2], row[3]])
        Yte.append(2.0 * row[4] - 1.0)
    var batches = fx.imat("parity_batches")
    theta = init.copy()
    opt = StepOptimizer.nesterov(0.5)
    var costs = List[Float64]()
    var accs = List[Float64]()
    t0 = perf_counter_ns()
    for it in range(100):
        var la = opt.lookahead(theta)
        var r = cost_and_grad(c, h, la, 24, Xtr, Ytr, batches[it])
        opt.step(theta, r[1])
        var ma = mse_and_acc(outputs(c, h, theta, 24, Xtr), Ytr, all_rows(len(Xtr)))
        costs.append(ma[0])
        accs.append(ma[1])
        if (it + 1) % 25 == 0:
            print("  iter", it + 1, " cost", ma[0], " train acc", ma[1])
    var t_sto = seconds_since(t0)
    _ = log.close("stochastic cost trajectory max abs diff (100 iters)", vmax_abs_diff(costs, fx.vec("parity_train_cost")), 0.0, 1e-8)
    _ = log.check("stochastic accuracy trajectory equals fixture", vmax_abs_diff(accs, fx.vec("parity_train_acc")) == 0.0, "")
    _ = log.close("final training cost", costs[99], fx.vec("parity_train_cost")[99], 1e-8)
    _ = log.close("final training accuracy", accs[99], 1.0, 0.0)
    var to = outputs(c, h, theta, 24, Xte)
    _ = log.close("test outputs max abs diff", vmax_abs_diff(to, fx.vec("parity_test_outputs")), 0.0, 1e-8)
    var tacc = mse_and_acc(to, Yte, all_rows(len(Xte)))[1]
    _ = log.close("test accuracy", tacc, fx.num("parity_test_acc"), 0.0)
    print("  page: final cost 1.74e-5 (other RNG draws); fixture final cost", fx.vec("parity_train_cost")[99])
    print("  timing (indicative): QUMO full-batch", fmt_s(t_full), "s, stochastic", fmt_s(t_sto),
          "s; PennyLane", fmt_s(fx.num("parity_fullbatch_seconds")), "s and", fmt_s(fx.num("parity_seconds")), "s (fixture)")


# ---------------------------------------------------------------------------
# Part 2: Iris classes 1 and 2 with the demo's state preparation
# ---------------------------------------------------------------------------
def get_angles(x: List[Float64]) -> List[Float64]:
    """The demo's get_angles, including its 1e-12 regularisers (bit-compatible)."""
    var beta0 = 2.0 * asin(sqrt(x[1] ** 2) / sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    var beta1 = 2.0 * asin(sqrt(x[3] ** 2) / sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    var n_hi = sqrt(x[2] ** 2 + x[3] ** 2)
    var n_all = sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    var beta2 = 2.0 * asin(n_hi / n_all)
    return [beta2, -beta1 / 2.0, beta1 / 2.0, -beta0 / 2.0, beta0 / 2.0]


def append_state_preparation(mut c: Circuit, a: List[Param]):
    c.ry(a[0], 0)
    c.cx(0, 1)
    c.ry(a[1], 1)
    c.cx(0, 1)
    c.ry(a[2], 1)
    c.x(0)
    c.cx(0, 1)
    c.ry(a[3], 1)
    c.cx(0, 1)
    c.ry(a[4], 1)
    c.x(0)


def iris_circuit() raises -> Circuit:
    var c = Circuit(2, 0, "iris_classifier")
    var w = c.parameters("w", 36)  # w[l][q][k] at 6 l + 3 q + k
    var a = c.parameters("a", 5)
    append_state_preparation(c, a)
    for l in range(6):
        append_rot(c, w[6 * l], w[6 * l + 1], w[6 * l + 2], 0)
        append_rot(c, w[6 * l + 3], w[6 * l + 4], w[6 * l + 5], 1)
        c.cx(0, 1)
    return c^


def part_iris(mut log: CheckLog, fx: Fixture) raises:
    print("== Iris classes 1 and 2 (2 qubits, 6 layers)")
    var xv = fx.vec("iris_doc_vector")
    var ang = get_angles(xv)
    _ = log.close("get_angles(doc vector) max abs diff", vmax_abs_diff(ang, fx.vec("iris_doc_angles")), 0.0, 1e-15)
    _ = log.close("mottonen_angles == get_angles", vmax_abs_diff(mottonen_angles(xv), ang), 0.0, 1e-11)
    var sp = Circuit(2, 0, "state_preparation")
    var spa = sp.parameters("a", 5)
    append_state_preparation(sp, spa)
    var sv = evolve(sp, ang)
    var amps = List[Float64]()
    for i in range(4):  # PennyLane index i = 2 b0 + b1 -> QUMO index b0 + 2 b1
        amps.append(sv.re[((i >> 1) & 1) | ((i & 1) << 1)])
    _ = log.close("state preparation amplitudes max abs diff", vmax_abs_diff(amps, fx.vec("iris_doc_state")), 0.0, 1e-12)

    var c = iris_circuit()
    var h = parse_pauli_sum("Z0")
    var p = List[Float64]()
    for l in range(6):
        for q in range(2):
            for k in range(3):
                p.append(0.05 * Float64(l + 1) - 0.02 * Float64(q) + 0.003 * Float64(k))
    for v in ang:
        p.append(v)
    var eng = ExpectationEngine(c, h)
    var g = List[Float64]()
    var e = eng.value_and_gradient(p, g, Grad.ADJOINT)
    _ = log.close("iris forward <Z0> with fixed weights", e, 0.45177693213489395, 1e-12)
    var g36 = List[Float64]()
    for k in range(36):
        g36.append(g[k])
    _ = log.close("iris weight gradients max abs diff", vmax_abs_diff(g36, fx.vec("iris_fixed_grad")), 0.0, 1e-12)

    # data: first two features padded with 0.1, 0.1, normalised, encoded by get_angles
    var data = load_table(DATA_DIR + "iris_classes1and2_scaled.txt")
    var feats = List[List[Float64]]()
    var Y = List[Float64]()
    for row in data:
        var v: List[Float64] = [row[0], row[1], 0.1, 0.1]
        var nrm = sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2 + v[3] ** 2)
        for ref t in v:
            t /= nrm
        feats.append(get_angles(v))
        Y.append(row[4])
    var f3 = List[Float64]()
    for i in range(3):
        for v in feats[i]:
            f3.append(v)
    _ = log.close("first three feature vectors max abs diff", vmax_abs_diff(f3, fx.vec("iris_features_first3")), 0.0, 1e-14)

    var tr_i = fx.ivec("iris_train_idx")
    var va_i = fx.ivec("iris_val_idx")
    var ftr = take_rows(feats, tr_i)
    var ytr = take_values(Y, tr_i)
    var theta = fx.vec("iris_init_weights")
    theta.append(0.0)
    var opt = StepOptimizer.nesterov(0.01)
    var batches = fx.imat("iris_batches")
    var costs = List[Float64]()
    var acc_tr = List[Float64]()
    var acc_va = List[Float64]()
    var t0 = perf_counter_ns()
    for it in range(60):
        var la = opt.lookahead(theta)
        var r = cost_and_grad(c, h, la, 36, ftr, ytr, batches[it])
        opt.step(theta, r[1])
        var vals = outputs(c, h, theta, 36, feats)
        costs.append(mse_and_acc(vals, Y, all_rows(len(Y)))[0])
        acc_tr.append(mse_and_acc(vals, Y, tr_i)[1])
        acc_va.append(mse_and_acc(vals, Y, va_i)[1])
        if (it + 1) % 15 == 0:
            print("  iter", it + 1, " cost", costs[it], " acc train", acc_tr[it], " acc validation", acc_va[it])
    var secs = seconds_since(t0)
    _ = log.close("iris cost trajectory max abs diff (60 iters)", vmax_abs_diff(costs, fx.vec("iris_cost")), 0.0, 1e-8)
    _ = log.check("iris accuracy trajectories equal fixture",
                  vmax_abs_diff(acc_tr, fx.vec("iris_acc_train")) == 0.0 and vmax_abs_diff(acc_va, fx.vec("iris_acc_val")) == 0.0, "")
    var w_ref = fx.vec("iris_final_weights")
    w_ref.append(fx.num("iris_final_bias"))
    _ = log.close("iris final weights max abs diff", vmax_abs_diff(theta, w_ref), 0.0, 1e-8)
    _ = log.close("iris final validation accuracy", acc_va[59], 1.0, 0.0)
    print("  timing (indicative): QUMO", fmt_s(secs), "s; PennyLane", fmt_s(fx.num("iris_seconds")), "s (fixture)")


# ---------------------------------------------------------------------------
# Part 3: Qibo-style 3-class Iris classifier
# ---------------------------------------------------------------------------
def qibo_circuit(nq: Int, nl: Int) raises -> Circuit:
    var c = Circuit(nq, 0, "qibo_classifier")
    var w = c.parameters("w", nl * nq)
    var x = c.parameters("x", nq)
    var wires = List[Int]()
    for q in range(nq):
        wires.append(q)
    append_angle_embedding(c, x, wires, "y")
    for l in range(nl):
        for q in range(nq):
            c.ry(w[l * nq + q], q)
        for q in range(nq - 1):
            c.cz(q, q + 1)
    return c^


def class_accuracy(rows: List[Int], c: Circuit, obs: List[PauliSum], w: List[Float64], X: List[List[Float64]], y: List[Float64]) raises -> Float64:
    var ok = 0
    for i in rows:
        var p = w.copy()
        for v in X[i]:
            p.append(v)
        var probs = expectations(c, obs, p)
        var best = 0
        for k in range(1, 3):
            if probs[k] > probs[best]:
                best = k
        if best == Int(y[i]):
            ok += 1
    return Float64(ok) / Float64(len(rows))


def part_qibo(mut log: CheckLog) raises:
    print("== Qibo-style 3-class Iris (4 qubits, 5 layers)")
    var d = load_csv(DATA_DIR + "iris.csv")
    var X = d[0].copy()
    var y = d[1].copy()
    # min-max scale every feature to [0, pi] (RY angles)
    for j in range(4):
        var lo = X[0][j]
        var hi = X[0][j]
        for row in X:
            lo = min(lo, row[j])
            hi = max(hi, row[j])
        for ref row in X:
            row[j] = pi * (row[j] - lo) / (hi - lo)
    var split = train_test_split(len(X), 0.25, 7)
    var tr = split[0].copy()
    var te = split[1].copy()
    var nq = 4
    var nl = 5
    var nw = nl * nq
    var c = qibo_circuit(nq, nl)
    # probabilities of |q1 q0> = |00>, |01>, |10> (class k = basis state k of qubits 0, 1)
    var obs: List[PauliSum] = [
        parse_pauli_sum("0.25 + 0.25 * Z0 + 0.25 * Z1 + 0.25 * Z0 Z1"),
        parse_pauli_sum("0.25 - 0.25 * Z0 + 0.25 * Z1 - 0.25 * Z0 Z1"),
        parse_pauli_sum("0.25 + 0.25 * Z0 - 0.25 * Z1 - 0.25 * Z0 Z1"),
    ]
    var w = List[Float64]()
    for i in range(nw):
        w.append(0.1 * Float64((i * 7) % 11) - 0.5)  # fixed deterministic start
    var opt = StepOptimizer.adam(0.05)
    var t0 = perf_counter_ns()
    var last = 0.0
    for it in range(120):
        var g = List[Float64](length=nw, fill=0.0)
        var loss = 0.0
        for i in tr:
            var p = w.copy()
            for v in X[i]:
                p.append(v)
            var probs = expectations(c, obs, p)
            var u = List[Float64]()
            for k in range(3):
                var t = 1.0 if Int(y[i]) == k else 0.0
                loss += (probs[k] - t) ** 2
                u.append(2.0 * (probs[k] - t) / Float64(len(tr)))
            var gv = vjp(c, obs, p, u)
            for k in range(nw):
                g[k] += gv[k]
        last = loss / Float64(len(tr))
        opt.step(w, g)
        if (it + 1) % 30 == 0:
            print("  iter", it + 1, " loss", last)
    var secs = seconds_since(t0)

    var a_tr = class_accuracy(tr, c, obs, w, X, y)
    var a_te = class_accuracy(te, c, obs, w, X, y)
    print("  final loss", last, " train accuracy", a_tr, " test accuracy", a_te, " (", fmt_s(secs), "s )")
    _ = log.within("Qibo-style 3-class test accuracy", a_te, 0.9, 1.0)
    _ = log.within("Qibo-style 3-class train accuracy", a_tr, 0.85, 1.0)


def main() raises:
    var log = CheckLog("variational_classifier")
    var fx = Fixture("variational_classifier")
    part_parity(log, fx)
    part_iris(log, fx)
    part_qibo(log)
    log.finish()
