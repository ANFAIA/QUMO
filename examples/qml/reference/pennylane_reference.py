#!/usr/bin/env python3
"""PennyLane reference fixtures for QUMO's QML examples (examples/qml/*.mojo).

The QUMO examples re-implement PennyLane demos and compare their numbers with
this file's output, examples/qml/reference/pennylane_reference.json: data,
initial parameters, batch indices, forward values, gradients, per-step
trajectories, final metrics and PennyLane wall times (indicative only).

PennyLane, scikit-learn, JAX, optax and torch are NOT QUMO dependencies. Run
this script in its own virtual environment, never in bench/.venv:

    uv venv /tmp/qumo-qml-ref --python 3.12
    uv pip install --python /tmp/qumo-qml-ref/bin/python -r examples/qml/reference/requirements.txt
    /tmp/qumo-qml-ref/bin/python examples/qml/reference/pennylane_reference.py --write-data
    /tmp/qumo-qml-ref/bin/python examples/qml/reference/pennylane_reference.py            # every section
    /tmp/qumo-qml-ref/bin/python examples/qml/reference/pennylane_reference.py --only qcbm qcnn

--write-data (re)writes examples/qml/data/ from scikit-learn's bundled copies
of the UCI data sets plus deterministic synthetic sets; nothing is
downloaded. Sections read the data back from examples/qml/data/, so the
fixture always describes the committed files.

Sections can run in parallel processes (`--only a` and `--only b`); each one
updates its own key in the JSON under a file lock.

Each section follows the demo named in its "source" field. Where the demo
draws random numbers without a seed (QCNN weights) this script seeds NumPy and
records the draws. JAX runs with jax_enable_x64, so JAX references are float64
(the demos run float32). The Torch section keeps torch's float32 training as
in the demo and adds a float64 forward/Jacobian reference.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
QML_DIR = HERE.parent
DATA = QML_DIR / "data"
OUT = HERE / "pennylane_reference.json"
LOCK = Path(tempfile.gettempdir()) / "qumo_pennylane_reference.lock"

DEMO = "https://pennylane.ai/qml/demos/"
SRC = "https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/"


def fl(v):
    """Nested numpy/python numbers -> plain JSON-ready lists of float."""
    import numpy as onp

    a = onp.asarray(v)
    if a.ndim == 0:
        return float(a)
    return [fl(x) for x in a]


def il(v):
    import numpy as onp

    a = onp.asarray(v)
    if a.ndim == 0:
        return int(a)
    return [il(x) for x in a]


def flat(v):
    import numpy as onp

    return [float(x) for x in onp.ravel(onp.asarray(v))]


def source(slug):
    return {"demo": DEMO + slug, "code": SRC + slug + "/demo.py"}


# --------------------------------------------------------------------------
# Data (written once with --write-data, read back by every section)
# --------------------------------------------------------------------------

PARITY_TRAIN = ["0001", "0010", "0100", "0101", "0110", "0111", "1000", "1001", "1011", "1111"]


def write_data():
    import numpy as onp
    from sklearn.datasets import load_digits, load_iris, make_moons

    DATA.mkdir(parents=True, exist_ok=True)
    X, y = load_iris(return_X_y=True)
    with open(DATA / "iris.csv", "w") as f:
        f.write("sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,label\n")
        for row, lab in zip(X, y):
            f.write(",".join(repr(float(v)) for v in row) + f",{int(lab)}\n")

    # Classes 1 and 2 in 1-based numbering (setosa, versicolor), min-max scaled
    # to [0, 1] per feature, labels -1 / +1: the layout of PennyLane's
    # iris_classes1and2_scaled.txt. The features stay non-negative because the
    # demo's get_angles encodes |x| only.
    X100 = X[:100]
    Xs = (X100 - X100.min(axis=0)) / (X100.max(axis=0) - X100.min(axis=0))
    with open(DATA / "iris_classes1and2_scaled.txt", "w") as f:
        for row, lab in zip(Xs, y[:100]):
            f.write(" ".join(repr(float(v)) for v in row) + (" -1\n" if lab == 0 else " 1\n"))

    strings = [format(i, "04b") for i in range(16)]
    with open(DATA / "parity_train.txt", "w") as f:
        for s in PARITY_TRAIN:
            f.write(" ".join(s) + f" {s.count('1') % 2}\n")
    with open(DATA / "parity_test.txt", "w") as f:
        for s in strings:
            if s not in PARITY_TRAIN:
                f.write(" ".join(s) + f" {s.count('1') % 2}\n")

    d = load_digits()
    mask = (d.target == 0) | (d.target == 1)
    with open(DATA / "digits01.csv", "w") as f:
        f.write(",".join(f"p{i}" for i in range(64)) + ",label\n")
        for row, lab in zip(d.data[mask], d.target[mask]):
            f.write(",".join(str(int(v)) for v in row) + f",{int(lab)}\n")

    Xm, ym = make_moons(n_samples=200, noise=0.1, random_state=42)
    with open(DATA / "moons_200.csv", "w") as f:
        f.write("x0,x1,label\n")
        for row, lab in zip(Xm, ym):
            f.write(f"{float(row[0])!r},{float(row[1])!r},{int(lab)}\n")
    print("wrote", sorted(p.name for p in DATA.iterdir()))


def read_csv(name):
    import numpy as onp

    a = onp.loadtxt(DATA / name, delimiter=",", skiprows=1)
    return a[:, :-1], a[:, -1].astype(int)


# --------------------------------------------------------------------------
# 1. tutorial_qubit_rotation
# --------------------------------------------------------------------------

def sec_qubit_rotation():
    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(p):
        qml.RX(p[0], wires=0)
        qml.RY(p[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    t0 = time.perf_counter()
    p = pnp.array([0.011, 0.012], requires_grad=True)
    init_cost = float(circuit(p))
    init_grad = flat(qml.grad(circuit)(p))
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    costs, params = [], []
    for _ in range(100):
        p = opt.step(circuit, p)
        costs.append(float(circuit(p)))
        params.append(flat(p))
    first = next(i + 1 for i, c in enumerate(costs) if c < -0.999999)
    return {
        "source": source("tutorial_qubit_rotation"),
        "init_params": [0.011, 0.012], "stepsize": 0.4, "steps": 100,
        "init_cost": init_cost, "init_grad": init_grad,
        "cost_after_step": costs, "params_after_step": params,
        "first_step_below_m0_999999": first,
        "final_params": params[-1], "final_cost": costs[-1],
        "seconds": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------
# 2. tutorial_variational_classifier (parity + iris)
# --------------------------------------------------------------------------

def sec_variational_classifier():
    import numpy as onp
    import pennylane as qml
    from pennylane import numpy as np

    res = {"source": source("tutorial_variational_classifier"),
           "qibo_source": "https://qibo.science/qibo/stable/code-examples/tutorials/variational_classifier/README.html"}

    # ---- parity ----
    dev = qml.device("default.qubit", wires=4)

    def layer(W):
        for w in range(4):
            qml.Rot(*W[w], wires=w)
        for a, b in ([0, 1], [1, 2], [2, 3], [3, 0]):
            qml.CNOT(wires=[a, b])

    @qml.qnode(dev)
    def circuit(weights, x):
        qml.BasisState(x, wires=[0, 1, 2, 3])
        for W in weights:
            layer(W)
        return qml.expval(qml.PauliZ(0))

    def classifier(weights, bias, x):
        return circuit(weights, x) + bias

    def square_loss(labels, predictions):
        return np.mean((labels - qml.math.stack(predictions)) ** 2)

    def accuracy(labels, predictions):
        acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
        return acc / len(labels)

    def cost(weights, bias, X, Y):
        predictions = [classifier(weights, bias, x) for x in X]
        return square_loss(Y, predictions)

    wfix = np.array([[[0.1 * (l + 1) + 0.01 * q + 0.001 * k for k in range(3)] for q in range(4)]
                     for l in range(2)], requires_grad=True)
    x0101 = np.array([0, 1, 0, 1], requires_grad=False)
    res["parity_fixed_weights_formula"] = "w[l][q][k] = 0.1*(l+1) + 0.01*q + 0.001*k"
    res["parity_fixed_x"] = [0, 1, 0, 1]
    res["parity_fixed_expval"] = float(circuit(wfix, x0101))
    res["parity_fixed_grad"] = flat(qml.grad(circuit, argnums=0)(wfix, x0101))

    tr = onp.loadtxt(DATA / "parity_train.txt", dtype=int)
    te = onp.loadtxt(DATA / "parity_test.txt", dtype=int)
    Xtr, Ytr = np.array(tr[:, :-1], requires_grad=False), np.array(tr[:, -1] * 2 - 1, requires_grad=False)
    Xte, Yte = np.array(te[:, :-1], requires_grad=False), np.array(te[:, -1] * 2 - 1, requires_grad=False)
    X16 = np.array([[int(b) for b in format(i, "04b")] for i in range(16)], requires_grad=False)
    Y16 = np.array([2 * (sum(r) % 2) - 1 for r in X16], requires_grad=False)

    np.random.seed(0)
    w0 = 0.01 * np.random.randn(2, 4, 3, requires_grad=True)
    b0 = np.array(0.0, requires_grad=True)
    res["parity_init_weights"] = flat(w0)

    # deterministic full-batch run on all 16 strings (cost is evaluated at the
    # Nesterov look-ahead point, as step_and_cost returns it)
    t0 = time.perf_counter()
    opt = qml.NesterovMomentumOptimizer(0.5)
    w, b = np.array(w0, requires_grad=True), np.array(b0, requires_grad=True)
    traj = []
    for _ in range(30):
        (w, b, _, _), c = opt.step_and_cost(cost, w, b, X16, Y16)
        traj.append(float(c))
    res["parity_fullbatch_cost_traj"] = traj
    res["parity_fullbatch_final_cost"] = float(cost(w, b, X16, Y16))
    res["parity_fullbatch_final_weights"] = flat(w)
    res["parity_fullbatch_final_bias"] = float(b)
    res["parity_fullbatch_seconds"] = time.perf_counter() - t0

    # demo run: Nesterov(0.5), batch 5, 100 iterations on parity_train.txt
    t0 = time.perf_counter()
    opt = qml.NesterovMomentumOptimizer(0.5)
    w, b = np.array(w0, requires_grad=True), np.array(b0, requires_grad=True)
    batches, costs, accs = [], [], []
    for _ in range(100):
        idx = np.random.randint(0, len(Xtr), (5,))
        batches.append(il(idx))
        w, b = opt.step(cost, w, b, X=Xtr[idx], Y=Ytr[idx])
        outs = onp.array([float(classifier(w, b, x)) for x in Xtr])
        costs.append(float(onp.mean((onp.asarray(Ytr) - outs) ** 2)))
        accs.append(float(accuracy(Ytr, onp.sign(outs))))
    res["parity_batches"] = batches
    res["parity_train_cost"] = costs
    res["parity_train_acc"] = accs
    res["parity_final_weights"] = flat(w)
    res["parity_final_bias"] = float(b)
    test_out = [float(classifier(w, b, x)) for x in Xte]
    res["parity_test_outputs"] = test_out
    res["parity_test_acc"] = float(accuracy(Yte, [onp.sign(v) for v in test_out]))
    res["parity_seconds"] = time.perf_counter() - t0

    # ---- iris ----
    def get_angles(x):
        beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
        beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
        beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / np.linalg.norm(x))
        return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

    def state_preparation(a):
        qml.RY(a[0], wires=0)
        qml.CNOT(wires=[0, 1]); qml.RY(a[1], wires=1); qml.CNOT(wires=[0, 1]); qml.RY(a[2], wires=1)
        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1]); qml.RY(a[3], wires=1); qml.CNOT(wires=[0, 1]); qml.RY(a[4], wires=1)
        qml.PauliX(wires=0)

    dev2 = qml.device("default.qubit", wires=2)

    @qml.qnode(dev2)
    def sp_state(a):
        state_preparation(a)
        return qml.state()

    def ilayer(W):
        qml.Rot(*W[0], wires=0)
        qml.Rot(*W[1], wires=1)
        qml.CNOT(wires=[0, 1])

    @qml.qnode(dev2)
    def icircuit(weights, a):
        state_preparation(a)
        for W in weights:
            ilayer(W)
        return qml.expval(qml.PauliZ(0))

    def iclassifier(weights, bias, a):
        return icircuit(weights, a) + bias

    def icost(weights, bias, A, Y):
        return square_loss(Y, [iclassifier(weights, bias, a) for a in A])

    xv = np.array([0.53896774, 0.79503606, 0.27826503, 0.0], requires_grad=False)
    ang = get_angles(xv)
    res["iris_doc_vector"] = flat(xv)
    res["iris_doc_angles"] = flat(ang)
    res["iris_doc_state"] = flat(onp.real(sp_state(ang)))
    wi = np.array([[[0.05 * (l + 1) - 0.02 * q + 0.003 * k for k in range(3)] for q in range(2)]
                   for l in range(6)], requires_grad=True)
    res["iris_fixed_weights_formula"] = "w[l][q][k] = 0.05*(l+1) - 0.02*q + 0.003*k"
    res["iris_fixed_expval"] = float(icircuit(wi, ang))
    res["iris_fixed_grad"] = flat(qml.grad(icircuit, argnums=0)(wi, ang))

    data = onp.loadtxt(DATA / "iris_classes1and2_scaled.txt")
    X = data[:, 0:2]
    padding = onp.ones((len(X), 2)) * 0.1
    X_pad = onp.c_[X, padding]
    normalization = onp.sqrt(onp.sum(X_pad ** 2, -1))
    X_norm = (X_pad.T / normalization).T
    features = np.array([get_angles(x) for x in X_norm], requires_grad=False)
    Y = np.array(data[:, -1], requires_grad=False)
    res["iris_features_first3"] = fl(features[:3])

    t0 = time.perf_counter()
    np.random.seed(0)
    num_data = len(Y)
    num_train = int(0.75 * num_data)
    index = np.random.permutation(range(num_data))
    tr_i, va_i = index[:num_train], index[num_train:]
    res["iris_train_idx"] = il(tr_i)
    res["iris_val_idx"] = il(va_i)
    ftr, Ytr_i = features[tr_i], Y[tr_i]
    fva, Yva = features[va_i], Y[va_i]
    wi0 = 0.01 * np.random.randn(6, 2, 3, requires_grad=True)
    bi0 = np.array(0.0, requires_grad=True)
    res["iris_init_weights"] = flat(wi0)
    opt = qml.NesterovMomentumOptimizer(0.01)
    w, b = wi0, bi0
    batches, costs, acc_tr, acc_va = [], [], [], []
    for _ in range(60):
        bi = np.random.randint(0, num_train, (5,))
        batches.append(il(bi))
        w, b, _, _ = opt.step(icost, w, b, ftr[bi], Ytr_i[bi])
        outs = onp.array([float(iclassifier(w, b, f)) for f in features])
        acc_tr.append(float(accuracy(Ytr_i, onp.sign(outs[tr_i]))))
        acc_va.append(float(accuracy(Yva, onp.sign(outs[va_i]))))
        costs.append(float(onp.mean((onp.asarray(Y) - outs) ** 2)))
    res["iris_batches"] = batches
    res["iris_cost"] = costs
    res["iris_acc_train"] = acc_tr
    res["iris_acc_val"] = acc_va
    res["iris_final_weights"] = flat(w)
    res["iris_final_bias"] = float(b)
    res["iris_seconds"] = time.perf_counter() - t0
    return res


# --------------------------------------------------------------------------
# 3. tutorial_data_reuploading_classifier
# --------------------------------------------------------------------------

def sec_data_reuploading():
    import numpy as onp
    import pennylane as qml
    from pennylane import numpy as np

    def density_matrix(state):
        return state * np.conj(state).T

    label_0 = [[1], [0]]
    label_1 = [[0], [1]]
    state_labels = np.array([label_0, label_1], requires_grad=False)

    def circle(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
        Xvals, yvals = [], []
        for _ in range(samples):
            x = 2 * (np.random.rand(2)) - 1
            y = 0
            if np.linalg.norm(x - center) < radius:
                y = 1
            Xvals.append(x)
            yvals.append(y)
        return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def qcircuit(params, x, y):
        for p in params:
            qml.Rot(*x, wires=0)
            qml.Rot(*p, wires=0)
        return qml.expval(qml.Hermitian(y, wires=[0]))

    dm_labels = [density_matrix(s) for s in state_labels]

    def cost(params, x, y, state_labels=None):
        loss = 0.0
        for i in range(len(x)):
            f = qcircuit(params, x[i], dm_labels[y[i]])
            loss = loss + (1 - f) ** 2
        return loss / len(x)

    def test(params, x, y, state_labels=None):
        predicted = []
        fidelities = []
        for i in range(len(x)):
            fids = [qcircuit(params, x[i], dm) for dm in dm_labels]
            predicted.append(int(np.argmax(fids)))
            fidelities.append(fids)
        return np.array(predicted), fidelities

    def accuracy_score(y_true, y_pred):
        return float(np.sum(y_true == y_pred) / len(y_true))

    def iterate_minibatches(inputs, targets, batch_size):
        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield inputs[idxs], targets[idxs]

    np.random.seed(42)
    _plot_X, _plot_y = circle(500)  # the demo draws a plot set first; it advances the RNG
    Xdata, y_train = circle(200)
    X_train = np.hstack((Xdata, np.zeros((Xdata.shape[0], 1), requires_grad=False)))
    Xtest, y_test = circle(2000)
    X_test = np.hstack((Xtest, np.zeros((Xtest.shape[0], 1), requires_grad=False)))
    params = np.random.uniform(size=(3, 3), requires_grad=True)

    res = {"source": source("tutorial_data_reuploading_classifier"),
           "qibo_source": "https://qibo.science/qibo/stable/code-examples/tutorials/reuploading_classifier/README.html",
           "rng_note": "np.random.seed(42); circle(500) plot draw; circle(200) train; circle(2000) test; uniform(3,3) init",
           "X_train": fl(Xdata), "y_train": il(y_train), "X_test": fl(Xtest), "y_test": il(y_test),
           "init_params": fl(params)}
    res["init_fidelity_x0_label_y0"] = float(qcircuit(params, X_train[0], dm_labels[y_train[0]]))
    p0, _ = test(params, X_train, y_train)
    p0t, _ = test(params, X_test, y_test)
    res["epoch0_train_acc"] = accuracy_score(y_train, p0)
    res["epoch0_test_acc"] = accuracy_score(y_test, p0t)
    res["epoch0_test_loss"] = float(cost(params, X_test, y_test))

    t0 = time.perf_counter()
    opt = qml.AdamOptimizer(0.6, beta1=0.9, beta2=0.999)
    step_losses, ep_loss, ep_tr, ep_te = [], [], [], []
    for _ in range(10):
        for Xb, yb in iterate_minibatches(X_train, y_train, batch_size=32):
            (params, _, _, _), c = opt.step_and_cost(cost, params, Xb, yb, state_labels)
            step_losses.append(float(c))
        ptr, _ = test(params, X_train, y_train)
        ep_tr.append(accuracy_score(y_train, ptr))
        ep_loss.append(float(cost(params, X_train, y_train)))
        pte, _ = test(params, X_test, y_test)
        ep_te.append(accuracy_score(y_test, pte))
    res["train_seconds"] = time.perf_counter() - t0
    res["adam"] = {"lr": 0.6, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "form": "pennylane"}
    res["batch_size"] = 32
    res["epochs"] = 10
    res["step_losses"] = step_losses
    res["epoch_train_loss"] = ep_loss
    res["epoch_train_acc"] = ep_tr
    res["epoch_test_acc"] = ep_te
    res["final_params"] = fl(params)
    res["final_test_predictions"] = il(pte)
    res["final_train_acc"] = ep_tr[-1]
    res["final_test_acc"] = ep_te[-1]
    res["page_values"] = {"train_acc": 0.87, "test_acc": 0.779,
                          "note": "not reproducible with PennyLane 0.45.1; this fixture is the rerun"}
    return res


# --------------------------------------------------------------------------
# 4. tutorial_kernel_based_training
# --------------------------------------------------------------------------

def sec_quantum_kernel_svm():
    import numpy as onp
    import pennylane as qml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y = read_csv("iris.csv")
    X, y = X[:100], y[:100]
    Xs = StandardScaler().fit(X).transform(X)
    ys = 2 * (y - 0.5)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, random_state=42)
    itr, ite = train_test_split(onp.arange(100), random_state=42)
    assert onp.allclose(Xs[itr], Xtr) and onp.allclose(Xs[ite], Xte)

    n = 4
    dev = qml.device("default.qubit", wires=n)
    projector = onp.zeros((2 ** n, 2 ** n))
    projector[0, 0] = 1

    @qml.qnode(dev)
    def kernel(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n))
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n))
        return qml.expval(qml.Hermitian(projector, wires=range(n)))

    @qml.qnode(dev)
    def iqp_kernel(x1, x2):
        qml.IQPEmbedding(x1, wires=range(n), n_repeats=2)
        qml.adjoint(qml.IQPEmbedding)(x2, wires=range(n), n_repeats=2)
        return qml.probs(wires=range(n))

    def svm_block(Ktr, Kte):
        svm = SVC(kernel="precomputed").fit(Ktr, ytr)
        dec = svm.decision_function(Kte)
        return {"test_acc": float(onp.mean(svm.predict(Kte) == yte)),
                "support": il(svm.support_), "dual_coef": flat(svm.dual_coef_),
                "intercept": float(svm.intercept_[0]), "test_decision": flat(dec)}

    res = {"source": source("tutorial_kernel_based_training"),
           "split_note": "StandardScaler on iris[:100]; train_test_split(random_state=42), 75/25",
           "train_idx": il(itr), "test_idx": il(ite),
           "scaler_mean": flat(StandardScaler().fit(X).mean_), "scaler_scale": flat(StandardScaler().fit(X).scale_)}
    t0 = time.perf_counter()
    Ktr = onp.array([[float(kernel(a, b)) for b in Xtr] for a in Xtr])
    res["angle_Ktrain_seconds"] = time.perf_counter() - t0
    Kte = onp.array([[float(kernel(a, b)) for b in Xtr] for a in Xte])
    res["angle_k_tr0_tr1"] = float(Ktr[0, 1])
    res["angle_Ktrain_sum"] = float(onp.sum(Ktr))
    res["angle_Ktrain_trace"] = float(onp.trace(Ktr))
    res["angle_Ktrain_row0"] = flat(Ktr[0])
    res["angle_Ktest_row0"] = flat(Kte[0])
    res["angle_svm"] = svm_block(Ktr, Kte)

    t0 = time.perf_counter()
    Ktr2 = onp.array([[float(iqp_kernel(a, b)[0]) for b in Xtr] for a in Xtr])
    res["iqp_Ktrain_seconds"] = time.perf_counter() - t0
    Kte2 = onp.array([[float(iqp_kernel(a, b)[0]) for b in Xtr] for a in Xte])
    res["iqp_k_tr0_tr1"] = float(Ktr2[0, 1])
    res["iqp_Ktrain_sum"] = float(onp.sum(Ktr2))
    res["iqp_Ktrain_row0"] = flat(Ktr2[0])
    res["iqp_svm"] = svm_block(Ktr2, Kte2)

    M = int(onp.ceil(0.75 * 100))
    Mpred = 100 - M
    res["cost_model"] = {
        "kernel_formula": "M*M + M*Mpred", "kernel_runs": M * M + M * Mpred,
        "variational_formula": "n_params*n_steps*batch_size*shift_terms + Mpred",
        "variational_inputs": {"n_params": 24, "n_steps": 100, "batch_size": 20, "shift_terms": 2},
        "variational_runs": 24 * 100 * 20 * 2 + Mpred,
    }
    return res


# --------------------------------------------------------------------------
# 5. tutorial_kernels_module (DoubleCake, kernel-target alignment)
# --------------------------------------------------------------------------

def sec_kernel_alignment():
    import numpy as onp
    import pennylane as qml
    from pennylane import numpy as np
    from sklearn.svm import SVC

    np.random.seed(1359)

    def _make_circular_data(num_sectors):
        center_indices = np.array(range(0, num_sectors))
        sector_angle = 2 * np.pi / num_sectors
        angles = (center_indices + 0.5) * sector_angle
        x = 0.7 * np.cos(angles)
        y = 0.7 * np.sin(angles)
        labels = 2 * np.remainder(np.floor_divide(angles, sector_angle), 2) - 1
        return x, y, labels

    def make_double_cake_data(num_sectors):
        x1, y1, labels1 = _make_circular_data(num_sectors)
        x2, y2, labels2 = _make_circular_data(num_sectors)
        x = np.hstack([x1, 0.5 * x2])
        y = np.hstack([y1, 0.5 * y2])
        X = np.vstack([x, y]).T
        labels = np.hstack([labels1, -1 * labels2])
        return X, labels.astype(int)

    X, Y = make_double_cake_data(3)

    def layer(x, params, wires, i0=0, inc=1):
        i = i0
        for j, wire in enumerate(wires):
            qml.Hadamard(wires=[wire])
            qml.RZ(x[i % len(x)], wires=[wire])
            i += inc
            qml.RY(params[0, j], wires=[wire])
        n = len(wires)
        for j in range(n):
            qml.CRZ(params[1][j], wires=[wires[j], wires[(j + 1) % n]])

    def ansatz(x, params, wires):
        for j, layer_params in enumerate(params):
            layer(x, layer_params, wires, i0=j * len(wires))

    adjoint_ansatz = qml.adjoint(ansatz)
    dev = qml.device("default.qubit", wires=5)
    wires = list(range(5))

    @qml.qnode(dev)
    def kernel_circuit(x1, x2, params):
        ansatz(x1, params, wires=wires)
        adjoint_ansatz(x2, params, wires=wires)
        return qml.probs(wires=wires)

    def kernel(x1, x2, params):
        return kernel_circuit(x1, x2, params)[0]

    init_params = np.random.uniform(0, 2 * np.pi, (6, 2, 5), requires_grad=True)

    def target_alignment(X, Y, kern, assume_normalized_kernel=False, rescale_class_labels=True):
        K = qml.kernels.square_kernel_matrix(X, kern, assume_normalized_kernel=assume_normalized_kernel)
        if rescale_class_labels:
            nplus = np.count_nonzero(np.array(Y) == 1)
            nminus = len(Y) - nplus
            _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
        else:
            _Y = np.array(Y)
        T = np.outer(_Y, _Y)
        inner_product = np.sum(K * T)
        norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
        return inner_product / norm

    res = {"source": source("tutorial_kernels_module"),
           "params_layout": "params[l][r][j] flattened as l*10 + r*5 + j (r=0: RY angles, r=1: CRZ ring angles)",
           "X": fl(X), "Y": il(Y), "init_params": flat(init_params)}
    init_kernel = lambda a, b: kernel(a, b, init_params)
    K = qml.kernels.square_kernel_matrix(X, init_kernel, assume_normalized_kernel=True)
    res["K_init"] = fl(K)
    res["k01_init"] = float(kernel(X[0], X[1], init_params))
    res["kta_init"] = float(target_alignment(X, Y, init_kernel, assume_normalized_kernel=True))
    res["kta_init_qml"] = float(qml.kernels.target_alignment(X, Y, init_kernel, assume_normalized_kernel=True))
    res["dk01_dtheta_init"] = flat(qml.grad(lambda p: kernel(X[0], X[1], p))(init_params))
    res["dkta_dtheta_init"] = flat(qml.grad(
        lambda p: target_alignment(X, Y, lambda a, b: kernel(a, b, p), assume_normalized_kernel=True))(init_params))

    def svm_acc(params):
        km = lambda A, B: qml.kernels.kernel_matrix(A, B, lambda a, b: kernel(a, b, params))
        svm = SVC(kernel=km).fit(X, Y)
        return float(1 - onp.count_nonzero(svm.predict(X) - Y) / len(Y)), flat(svm.decision_function(X))

    res["svm_acc_init"], res["svm_decision_init"] = svm_acc(init_params)

    t0 = time.perf_counter()
    params = init_params
    opt = qml.GradientDescentOptimizer(0.2)
    subsets, align = [], []
    for i in range(500):
        subset = np.random.choice(list(range(len(X))), 4)
        subsets.append(il(subset))
        cost = lambda _params: -target_alignment(
            X[subset], Y[subset], lambda x1, x2: kernel(x1, x2, _params), assume_normalized_kernel=True)
        params = opt.step(cost, params)
        if (i + 1) % 50 == 0:
            align.append(float(target_alignment(X, Y, lambda x1, x2: kernel(x1, x2, params),
                                                assume_normalized_kernel=True)))
    res["train_seconds"] = time.perf_counter() - t0
    res["subsets"] = subsets
    res["alignment_every50"] = align
    res["final_params"] = flat(params)
    trained_kernel = lambda a, b: kernel(a, b, params)
    res["K_trained"] = fl(qml.kernels.square_kernel_matrix(X, trained_kernel, assume_normalized_kernel=True))
    res["svm_acc_trained"], res["svm_decision_trained"] = svm_acc(params)
    res["page_values"] = {"kta_init": 0.074, "alignment_first": 0.0904, "alignment_last": 0.2661,
                          "svm_acc_init": 0.833, "svm_acc_trained": 1.0}
    return res


# --------------------------------------------------------------------------
# 6. tutorial_quantum_natural_gradient
# --------------------------------------------------------------------------

def sec_quantum_natural_gradient():
    import numpy as onp
    import pennylane as qml
    from pennylane import numpy as np

    dev = qml.device("default.qubit", wires=3)

    def body(params):
        qml.RY(np.pi / 4, wires=0)
        qml.RY(np.pi / 3, wires=1)
        qml.RY(np.pi / 7, wires=2)
        qml.RZ(params[0], wires=0)
        qml.RZ(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.RY(params[2], wires=1)
        qml.RX(params[3], wires=2)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.expval(qml.PauliY(0))

    circuit = qml.QNode(body, dev, interface="autograd")
    # the full metric tensor needs one auxiliary wire
    circuit_aux = qml.QNode(body, qml.device("default.qubit", wires=4), interface="autograd")
    p0 = np.array([0.432, -0.123, 0.543, 0.233], requires_grad=True)
    res = {"source": source("tutorial_quantum_natural_gradient"), "init_params": flat(p0)}
    res["init_cost"] = float(circuit(p0))
    res["init_grad"] = flat(qml.grad(circuit)(p0))
    res["metric_block_diag"] = fl(qml.metric_tensor(circuit, approx="block-diag")(p0))
    res["metric_diag"] = fl(qml.metric_tensor(circuit, approx="diag")(p0))
    try:
        res["metric_full"] = fl(qml.metric_tensor(circuit_aux, approx=None, aux_wire=3)(p0))
    except Exception as e:  # pragma: no cover - depends on PennyLane internals
        res["metric_full_error"] = str(e)[:200]
    for name, opt in (("gd", qml.GradientDescentOptimizer(0.01)), ("qng", qml.QNGOptimizer(0.01))):
        t0 = time.perf_counter()
        p = p0.copy()
        hist = []
        for _ in range(200):
            p = opt.step(circuit, p)
            hist.append(float(circuit(p)))
        res[f"{name}_cost_after_step"] = hist
        res[f"{name}_final_params"] = flat(p)
        res[f"{name}_seconds"] = time.perf_counter() - t0
    return res


# --------------------------------------------------------------------------
# 7. tutorial_barren_plateaus (+ tutorial_local_cost_functions)
# --------------------------------------------------------------------------

def sec_barren_plateaus():
    import numpy as onp
    import pennylane as qml
    from pennylane import numpy as np

    gate_set = [qml.RX, qml.RY, qml.RZ]
    names = {qml.RX: "RX", qml.RY: "RY", qml.RZ: "RZ"}

    def rand_circuit(params, random_gate_sequence=None, num_qubits=None, local=False):
        for i in range(num_qubits):
            qml.RY(np.pi / 4, wires=i)
        for i in range(num_qubits):
            random_gate_sequence[i](params[i], wires=i)
        for i in range(num_qubits - 1):
            qml.CZ(wires=[i, i + 1])
        if local:
            return qml.expval(qml.sum(*[qml.s_prod(1.0 / num_qubits, qml.Projector([0], wires=[j]))
                                        for j in range(num_qubits)]))
        H = np.zeros((2 ** num_qubits, 2 ** num_qubits))
        H[0, 0] = 1
        return qml.expval(qml.Hermitian(H, wires=list(range(num_qubits))))

    res = {"source": source("tutorial_barren_plateaus"),
           "local_source": source("tutorial_local_cost_functions"),
           "circuit": "RY(pi/4) on every wire, then gate[i](params[i]) on wire i, then CZ(i, i+1); "
                      "global cost <|0..0><0..0|>, local cost (1/n) sum_j <|0><0|_j>"}

    # exact gradients for fixed exported sequences
    rs = onp.random.RandomState(2024)
    seqs = []
    for n in [2, 3, 4, 5, 6, 3, 4, 5, 6, 7]:
        g = [gate_set[k] for k in rs.randint(0, 3, size=n)]
        params = np.array(rs.uniform(0, 2 * onp.pi, size=n), requires_grad=True)
        dev = qml.device("default.qubit", wires=n)
        qg = qml.QNode(rand_circuit, dev)
        gsq = {i: g[i] for i in range(n)}
        seqs.append({
            "n": n, "gates": [names[x] for x in g], "params": flat(params),
            "global_value": float(qg(params, random_gate_sequence=gsq, num_qubits=n)),
            "global_grad": flat(qml.grad(qg, argnums=0)(params, random_gate_sequence=gsq, num_qubits=n)),
            "local_value": float(qg(params, random_gate_sequence=gsq, num_qubits=n, local=True)),
            "local_grad": flat(qml.grad(qg, argnums=0)(params, random_gate_sequence=gsq, num_qubits=n, local=True)),
        })
    res["sequences"] = seqs

    # the demo's statistics (indicative; QUMO uses its own RNG stream)
    t0 = time.perf_counter()
    np.random.seed(42)
    num_samples = 200
    gv = []
    for _ in range(num_samples):
        dev = qml.device("default.qubit", wires=4)
        qc = qml.QNode(rand_circuit, dev, interface="autograd")
        gs = {i: np.random.choice(gate_set) for i in range(4)}
        params = np.random.uniform(0, 2 * np.pi, size=4)
        gv.append(qml.grad(qc, argnums=0)(params, random_gate_sequence=gs, num_qubits=4)[-1])
    res["var_4q"] = float(np.var(gv))
    res["mean_4q"] = float(np.mean(gv))
    qubits = [2, 3, 4, 5, 6]
    variances, lvariances = [], []
    for nq in qubits:
        g1, g2 = [], []
        for _ in range(num_samples):
            dev = qml.device("default.qubit", wires=nq)
            qc = qml.QNode(rand_circuit, dev, interface="autograd")
            gs = {i: np.random.choice(gate_set) for i in range(nq)}
            params = np.random.uniform(0, np.pi, size=nq)
            g1.append(qml.grad(qc, argnums=0)(params, random_gate_sequence=gs, num_qubits=nq)[-1])
            g2.append(qml.grad(qc, argnums=0)(params, random_gate_sequence=gs, num_qubits=nq, local=True)[-1])
        variances.append(float(np.var(g1)))
        lvariances.append(float(np.var(g2)))
    res["qubits"] = qubits
    res["global_variances"] = variances
    res["local_variances"] = lvariances
    res["global_slope"] = float(onp.polyfit(qubits, onp.log(onp.array(variances)), 1)[0])
    res["local_slope"] = float(onp.polyfit(qubits, onp.log(onp.array(lvariances)), 1)[0])
    res["stats_seconds"] = time.perf_counter() - t0
    res["page_values"] = {"var_4q": 0.00602, "note": "rerun slopes vary between about -0.77 and -0.96"}
    return res


# --------------------------------------------------------------------------
# 8. tutorial_expressivity_fourier_series
# --------------------------------------------------------------------------

def sec_fourier_expressivity():
    import numpy as onp
    import pennylane as qml
    from pennylane import numpy as np

    np.random.seed(42)
    degree = 1
    scaling = 1
    coeffs = [0.15 + 0.15j] * degree
    coeff0 = 0.1

    def target_function(x):
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = complex(0, scaling * (idx + 1) * x)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70, requires_grad=False)
    target_y = np.array([target_function(x_) for x_ in x], requires_grad=False)

    def square_loss(targets, predictions):
        loss = 0
        for t, p in zip(targets, predictions):
            loss += (t - p) ** 2
        loss = loss / len(targets)
        return 0.5 * loss

    # serial model, r = 1
    dev = qml.device("default.qubit", wires=1)

    def S(x):
        qml.RX(scaling * x, wires=0)

    def W(theta):
        qml.Rot(theta[0], theta[1], theta[2], wires=0)

    @qml.qnode(dev)
    def serial_quantum_model(weights, x):
        for theta in weights[:-1]:
            W(theta)
            S(x)
        W(weights[-1])
        return qml.expval(qml.PauliZ(wires=0))

    r = 1
    weights = 2 * np.pi * np.random.random(size=(r + 1, 3), requires_grad=True)
    res = {"source": source("tutorial_expressivity_fourier_series"),
           "x": flat(x), "target_y": flat(target_y), "loss": "0.5 * mean((t - p)^2)",
           "serial_init_weights": flat(weights),
           "serial_f0_init": float(serial_quantum_model(weights, 0.0))}

    def cost(weights, x, y):
        predictions = [serial_quantum_model(weights, x_) for x_ in x]
        return square_loss(y, predictions)

    t0 = time.perf_counter()
    opt = qml.AdamOptimizer(0.3)
    batches, cst = [], [float(cost(weights, x, target_y))]
    for step in range(50):
        batch_index = np.random.randint(0, len(x), (25,))
        batches.append(il(batch_index))
        weights, _, _ = opt.step(cost, weights, x[batch_index], target_y[batch_index])
        cst.append(float(cost(weights, x, target_y)))
    res["serial_seconds"] = time.perf_counter() - t0
    res["serial_batches"] = batches
    res["serial_cost"] = cst
    res["cost_note"] = "serial_cost[0] / parallel_cost[0] is the initial cost; entry k is the cost after step k"
    res["serial_cost_every10"] = [cst[k] for k in (10, 20, 30, 40, 50)]
    res["serial_final_weights"] = flat(weights)

    def fourier_coefficients(f, K):
        n_coeffs = 2 * K + 1
        t = onp.linspace(0, 2 * onp.pi, n_coeffs, endpoint=False)
        y = onp.fft.rfft(f(t)) / t.size
        return y

    coeffs_trained = fourier_coefficients(
        lambda t: onp.array([float(serial_quantum_model(weights, tt)) for tt in t]), 5)
    res["serial_trained_coeffs_K5"] = [[float(c.real), float(c.imag)] for c in coeffs_trained]

    # dummy ansatz draw from the demo (advances the RNG before the parallel weights)
    _dummy = 2 * np.pi * np.random.random(size=(2, 3, 3))

    rr = 3
    devp = qml.device("default.qubit", wires=rr)

    def Sp(x):
        for w in range(rr):
            qml.RX(scaling * x, wires=w)

    def Wp(theta):
        qml.StronglyEntanglingLayers(theta, wires=range(rr))

    @qml.qnode(devp)
    def parallel_quantum_model(weights, x):
        Wp(weights[0])
        Sp(x)
        Wp(weights[1])
        return qml.expval(qml.PauliZ(wires=0))

    trainable_block_layers = 3
    pweights = 2 * np.pi * np.random.random(size=(2, trainable_block_layers, rr, 3), requires_grad=True)
    res["parallel_init_weights"] = flat(pweights)
    res["parallel_layout"] = "weights[block][layer][wire][k], block 0 before S(x), block 1 after"
    res["parallel_f0_init"] = float(parallel_quantum_model(pweights, 0.0))

    def pcost(weights, x, y):
        return square_loss(y, [parallel_quantum_model(weights, x_) for x_ in x])

    t0 = time.perf_counter()
    opt = qml.AdamOptimizer(0.3)
    pb, pc = [], [float(pcost(pweights, x, target_y))]
    for step in range(70):
        batch_index = np.random.randint(0, len(x), (25,))
        pb.append(il(batch_index))
        pweights, _, _ = opt.step(pcost, pweights, x[batch_index], target_y[batch_index])
        pc.append(float(pcost(pweights, x, target_y)))
    res["parallel_seconds"] = time.perf_counter() - t0
    res["parallel_batches"] = pb
    res["parallel_cost"] = pc
    res["parallel_final_weights"] = flat(pweights)
    res["page_values"] = {"serial_cost_every10_mse": [0.22856, 0.025282, 0.0042809, 0.0038642, 0.00026741],
                          "note": "research rerun with plain MSE; the demo's 0.5*MSE halves these"}
    return res


# --------------------------------------------------------------------------
# 9. tutorial_qcbm (bars and stripes 3x3)
# --------------------------------------------------------------------------

def sec_qcbm():
    import numpy as onp
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax
    import pennylane as qml

    def get_bars_and_stripes(n):
        bitstrings = [list(onp.binary_repr(i, n))[::-1] for i in range(2 ** n)]
        bitstrings = onp.array(bitstrings, dtype=int)
        stripes = bitstrings.copy()
        stripes = onp.repeat(stripes, n, 0)
        stripes = stripes.reshape(2 ** n, n * n)
        bars = bitstrings.copy()
        bars = bars.reshape(2 ** n * n, 1)
        bars = onp.repeat(bars, n, 1)
        bars = bars.reshape(2 ** n, n * n)
        return onp.vstack((stripes[0: stripes.shape[0] - 1], bars[1: bars.shape[0]]))

    n = 3
    size = n ** 2
    data = get_bars_and_stripes(n)
    bitstrings, nums = [], []
    for d in data:
        bitstrings += ["".join(str(int(i)) for i in d)]
        nums += [int(bitstrings[-1], 2)]
    probs = onp.zeros(2 ** size)
    probs[nums] = 1 / len(data)

    class MMD:
        def __init__(self, scales, space):
            gammas = 1 / (2 * (scales ** 2))
            sq_dists = jnp.abs(space[:, None] - space[None, :]) ** 2
            self.K = sum(jnp.exp(-gamma * sq_dists) for gamma in gammas) / len(scales)
            self.scales = scales

        def k_expval(self, px, py):
            return px @ self.K @ py

        def __call__(self, px, py):
            pxy = px - py
            return self.k_expval(pxy, pxy)

    onp.random.seed(42)
    n_qubits = size
    dev = qml.device("default.qubit", wires=n_qubits)
    n_layers = 6
    wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    weights = onp.random.random(size=wshape)

    @qml.qnode(dev)
    def circuit(weights):
        qml.StronglyEntanglingLayers(weights=weights, ranges=[1] * n_layers, wires=range(n_qubits))
        return qml.probs()

    jit_circuit = jax.jit(circuit)
    bandwidth = jnp.array([0.25, 0.5, 1])
    space = jnp.arange(2 ** n_qubits)
    mmd = MMD(bandwidth, space)
    py = jnp.array(probs)

    def mmd_loss(params):
        px = jit_circuit(params)
        return mmd(px, py), px

    opt = optax.adam(learning_rate=0.1)

    @jax.jit
    def update_step(params, opt_state):
        (loss_val, qcbm_probs), grads = jax.value_and_grad(mmd_loss, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        log_ratio = jnp.where(py == 0, 0, jnp.log(qcbm_probs / py))
        kl_div = -jnp.sum(py * log_ratio)
        return params, opt_state, loss_val, kl_div

    w = jnp.array(weights)
    (l0, p0), g0 = jax.value_and_grad(mmd_loss, has_aux=True)(w)
    res = {"source": source("tutorial_qcbm"),
           "valid_patterns": [int(v) for v in nums], "bitstrings": bitstrings,
           "bandwidth": [0.25, 0.5, 1.0], "n_layers": n_layers, "ranges": [1] * n_layers,
           "init_weights": flat(weights), "weights_layout": "weights[layer][wire][k] (Rot phi, theta, omega)",
           "init_loss": float(l0), "init_grad": flat(g0),
           "init_probs_first8": flat(p0[:8])}
    opt_state = opt.init(w)
    t0 = time.perf_counter()
    history, divs = [], []
    for i in range(100):
        w, opt_state, loss_val, kl_div = update_step(w, opt_state)
        history.append(float(loss_val))
        divs.append(float(kl_div))
    res["train_seconds"] = time.perf_counter() - t0
    res["loss"] = history
    res["kl"] = divs
    pf = onp.asarray(jit_circuit(w))
    res["final_weights"] = flat(w)
    res["final_loss"] = float(mmd(jnp.array(pf), py))
    mask = probs > 0
    res["final_kl"] = float(-onp.sum(probs[mask] * onp.log(pf[mask] / probs[mask])))
    res["final_valid_fraction_exact"] = float(onp.sum(pf[nums]))
    res["optax"] = {"lr": 0.1, "b1": 0.9, "b2": 0.999, "eps": 1e-8}
    return res


# --------------------------------------------------------------------------
# 10. tutorial_learning_few_data (QCNN on digits 0/1)
# --------------------------------------------------------------------------

def sec_qcnn():
    import numpy as onp
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax
    import pennylane as qml
    from pennylane import numpy as pnp

    def convolutional_layer(weights, wires, skip_first_layer=True):
        n_wires = len(wires)
        assert n_wires >= 3
        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    if indx % 2 == 0 and not skip_first_layer:
                        qml.U3(*weights[:3], wires=[w])
                        qml.U3(*weights[3:6], wires=[wires[indx + 1]])
                    qml.IsingXX(weights[6], wires=[w, wires[indx + 1]])
                    qml.IsingYY(weights[7], wires=[w, wires[indx + 1]])
                    qml.IsingZZ(weights[8], wires=[w, wires[indx + 1]])
                    qml.U3(*weights[9:12], wires=[w])
                    qml.U3(*weights[12:], wires=[wires[indx + 1]])

    def pooling_layer(weights, wires):
        n_wires = len(wires)
        assert len(wires) >= 2
        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                m_outcome = qml.measure(w)
                qml.cond(m_outcome, qml.U3)(*weights, wires=wires[indx - 1])

    def conv_and_pooling(kernel_weights, n_wires, skip_first_layer=True):
        convolutional_layer(kernel_weights[:15], n_wires, skip_first_layer=skip_first_layer)
        pooling_layer(kernel_weights[15:], n_wires)

    def dense_layer(weights, wires):
        qml.ArbitraryUnitary(weights, wires)

    num_wires = 6
    device = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(device)
    def conv_net(weights, last_layer_weights, features):
        layers = weights.shape[1]
        wires = list(range(num_wires))
        qml.AmplitudeEmbedding(features=features, wires=wires, pad_with=0.5)
        for j in range(layers):
            conv_and_pooling(weights[:, j], wires, skip_first_layer=(not j == 0))
            wires = wires[::2]
        dense_layer(last_layer_weights, wires)
        return qml.probs(wires=(0))

    feats, labels = read_csv("digits01.csv")
    feats = feats / onp.linalg.norm(feats, axis=1).reshape((-1, 1))

    # fixed-weight forward and gradient (autograd)
    x0 = feats[0]
    y0 = int(labels[0])
    W = pnp.array([[onp.sin(0.37 * k + 1.3 * j + 0.1) for j in range(2)] for k in range(18)], requires_grad=True)
    L = pnp.array([onp.cos(0.21 * i + 0.4) for i in range(15)], requires_grad=True)
    pr = conv_net(W, L, x0)
    g = qml.grad(lambda W, L: conv_net(W, L, x0)[y0], argnums=[0, 1])(W, L)
    ops = qml.ArbitraryUnitary.compute_decomposition(onp.arange(15) * 1.0, wires=[0, 1])
    res = {"source": source("tutorial_learning_few_data"),
           "weights_formula": "W[k][j] = sin(0.37k + 1.3j + 0.1) (18x2), last[i] = cos(0.21 i + 0.4) (15)",
           "weights_layout": "W[k][j]: k = 0..17 (15 conv + 3 pool), j = block",
           "fixed_sample": 0, "fixed_label": y0,
           "fixed_probs_wire0": flat(pr),
           "fixed_grad_p_label_W": flat(g[0]), "fixed_grad_p_label_last": flat(g[1]),
           "arbitrary_unitary_words": [op.hyperparameters["pauli_word"] for op in ops]}

    # training (JAX, float64)
    @jax.jit
    def compute_out(weights, weights_last, features, labels):
        cost = lambda weights, weights_last, feature, label: conv_net(weights, weights_last, feature)[label]
        return jax.vmap(cost, in_axes=(None, None, 0, 0), out_axes=0)(weights, weights_last, features, labels)

    def compute_accuracy(weights, weights_last, features, labels):
        out = compute_out(weights, weights_last, features, labels)
        return jnp.sum(out > 0.5) / len(out)

    def compute_cost(weights, weights_last, features, labels):
        out = compute_out(weights, weights_last, features, labels)
        return 1.0 - jnp.sum(out) / len(labels)

    value_and_grad = jax.jit(jax.value_and_grad(compute_cost, argnums=[0, 1]))
    rng = onp.random.default_rng(seed=0)
    onp.random.seed(0)  # the demo draws the weights from the unseeded global RNG
    n_test, n_epochs, n_reps = 100, 100, 10
    train_sizes = [5, 10, 20, 40]
    runs = []
    t0 = time.perf_counter()
    for n_train in train_sizes:
        for rep in range(n_reps):
            train_idx = rng.choice(len(labels), n_train, replace=False)
            test_idx = rng.choice(onp.setdiff1d(range(len(labels)), train_idx), n_test, replace=False)
            w0 = onp.random.normal(loc=0, scale=1, size=(18, 2))
            l0 = onp.random.normal(loc=0, scale=1, size=4 ** 2 - 1)
            weights, weights_last = jnp.array(w0), jnp.array(l0)
            sched = optax.cosine_decay_schedule(0.1, decay_steps=n_epochs, alpha=0.95)
            optimizer = optax.adam(learning_rate=sched)
            opt_state = optimizer.init((weights, weights_last))
            xtr, ytr = jnp.array(feats[train_idx]), jnp.array(labels[train_idx])
            xte, yte = jnp.array(feats[test_idx]), jnp.array(labels[test_idx])
            costs = []
            for step in range(n_epochs):
                c, grad_circuit = value_and_grad(weights, weights_last, xtr, ytr)
                updates, opt_state = optimizer.update(grad_circuit, opt_state)
                weights, weights_last = optax.apply_updates((weights, weights_last), updates)
                costs.append(float(c))
            runs.append({
                "n_train": n_train, "rep": rep, "train_idx": il(train_idx), "test_idx": il(test_idx),
                "init_W": flat(w0), "init_last": flat(l0), "train_cost": costs,
                "final_train_acc": float(compute_accuracy(weights, weights_last, xtr, ytr)),
                "final_test_acc": float(compute_accuracy(weights, weights_last, xte, yte)),
                "final_test_cost": float(compute_cost(weights, weights_last, xte, yte)),
            })
        print(f"  qcnn n_train={n_train} done ({time.perf_counter() - t0:.1f} s)", flush=True)
    res["train_seconds"] = time.perf_counter() - t0
    res["optax"] = {"schedule": "cosine_decay_schedule(0.1, decay_steps=100, alpha=0.95)",
                    "b1": 0.9, "b2": 0.999, "eps": 1e-8}
    res["train_sizes"] = train_sizes
    res["n_test"] = n_test
    res["n_epochs"] = n_epochs
    res["n_reps"] = n_reps
    res["runs"] = runs
    res["mean_test_acc"] = {str(s): float(onp.mean([r["final_test_acc"] for r in runs if r["n_train"] == s]))
                            for s in train_sizes}
    return res


# --------------------------------------------------------------------------
# 11. tutorial_qnn_module_torch (phase 2 consumer: hybrid_torch_layer.py)
# --------------------------------------------------------------------------

def sec_hybrid_torch_layer():
    import numpy as onp
    import pennylane as qml
    import torch

    torch.manual_seed(42)
    onp.random.seed(42)
    Xm, ym = read_csv("moons_200.csv")
    y_ = torch.unsqueeze(torch.tensor(ym), 1)
    y_hot = torch.scatter(torch.zeros((200, 2)), 1, y_, 1)

    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    n_layers = 6
    weight_shapes = {"weights": (n_layers, n_qubits)}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    clayer_1 = torch.nn.Linear(2, 2)
    clayer_2 = torch.nn.Linear(2, 2)
    softmax = torch.nn.Softmax(dim=1)
    model = torch.nn.Sequential(clayer_1, qlayer, clayer_2, softmax)
    res = {"source": source("tutorial_qnn_module_torch"),
           "init": {"clayer_1.weight": fl(clayer_1.weight.detach().numpy().astype(float)),
                    "clayer_1.bias": flat(clayer_1.bias.detach().numpy().astype(float)),
                    "qlayer.weights": fl(qlayer.weights.detach().numpy().astype(float)),
                    "clayer_2.weight": fl(clayer_2.weight.detach().numpy().astype(float)),
                    "clayer_2.bias": flat(clayer_2.bias.detach().numpy().astype(float))},
           "data_note": "moons_200.csv = make_moons(200, noise=0.1, random_state=42)"}

    # float64 forward and Jacobian of the quantum layer at fixed values
    xin = torch.tensor([0.3, -0.7], dtype=torch.float64)
    wq = torch.tensor([[0.1 * (l + 1) + 0.05 * q for q in range(2)] for l in range(6)], dtype=torch.float64)
    f = lambda a, b: torch.stack(qnode(a, b))
    res["layer_fixed_inputs"] = flat(xin.numpy())
    res["layer_fixed_weights"] = fl(wq.numpy())
    res["layer_fixed_weights_formula"] = "w[l][q] = 0.1*(l+1) + 0.05*q"
    res["layer_fixed_out"] = flat(f(xin, wq).detach().numpy())
    jx, jw = torch.autograd.functional.jacobian(f, (xin, wq))
    res["layer_fixed_jac_inputs"] = fl(jx.numpy())
    res["layer_fixed_jac_weights"] = fl(jw.reshape(2, -1).numpy())

    opt = torch.optim.SGD(model.parameters(), lr=0.2)
    loss = torch.nn.L1Loss()
    X = torch.tensor(Xm, requires_grad=True).float()
    y_hot = y_hot.float()
    batch_size = 5
    batches = 200 // batch_size
    data_loader = torch.utils.data.DataLoader(
        list(zip(X, y_hot, torch.arange(200))), batch_size=5, shuffle=True, drop_last=True)
    epochs = 6
    order, avg = [], []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        running_loss = 0
        ep = []
        for xs, ys, ids in data_loader:
            ep.append(il(ids.numpy()))
            opt.zero_grad()
            loss_evaluated = loss(model(xs), ys)
            loss_evaluated.backward()
            opt.step()
            running_loss += loss_evaluated
        order.append(ep)
        avg.append(float(running_loss / batches))
    res["train_seconds"] = time.perf_counter() - t0
    res["batch_order"] = order
    res["avg_loss_per_epoch"] = avg
    y_pred = model(X)
    predictions = torch.argmax(y_pred, axis=1).detach().numpy()
    res["accuracy"] = float(onp.mean(predictions == ym))
    res["page_values"] = {"avg_loss_first": 0.4943, "avg_loss_last": 0.1666, "accuracy": 0.855}
    return res


SECTIONS = {
    "qubit_rotation": sec_qubit_rotation,
    "variational_classifier": sec_variational_classifier,
    "data_reuploading": sec_data_reuploading,
    "quantum_kernel_svm": sec_quantum_kernel_svm,
    "kernel_alignment": sec_kernel_alignment,
    "quantum_natural_gradient": sec_quantum_natural_gradient,
    "barren_plateaus": sec_barren_plateaus,
    "fourier_expressivity": sec_fourier_expressivity,
    "qcbm": sec_qcbm,
    "qcnn": sec_qcnn,
    "hybrid_torch_layer": sec_hybrid_torch_layer,
}


@contextlib.contextmanager
def locked():
    with open(LOCK, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("pennylane", "numpy", "sklearn", "scipy", "jax", "optax", "torch"):
        try:
            out[m] = __import__(m).__version__
        except Exception:  # pragma: no cover - optional packages
            out[m] = None
    return out


def update_json(name, value):
    with locked():
        doc = json.loads(OUT.read_text()) if OUT.exists() else {}
        doc.setdefault("meta", {})
        doc["meta"].update({
            "generator": "examples/qml/reference/pennylane_reference.py",
            "versions": versions(),
            "note": "Reference values for QUMO's examples/qml. Wall times are indicative "
                    "(PennyLane default.qubit on the generating machine), not a benchmark.",
        })
        doc["meta"].setdefault("generated", {})[name] = time.strftime("%Y-%m-%d")
        doc[name] = value
        ordered = {"meta": doc["meta"]}
        for k in SECTIONS:
            if k in doc:
                ordered[k] = doc[k]
        tmp = OUT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ordered, indent=1) + "\n")
        os.replace(tmp, OUT)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", choices=sorted(SECTIONS), help="sections to (re)generate")
    ap.add_argument("--write-data", action="store_true", help="(re)write examples/qml/data/ and exit")
    args = ap.parse_args(argv)
    if args.write_data:
        write_data()
        return 0
    names = args.only or list(SECTIONS)
    for name in names:
        t0 = time.perf_counter()
        print(f"[{name}] ...", flush=True)
        value = SECTIONS[name]()
        value.setdefault("generator_seconds", time.perf_counter() - t0)
        update_json(name, value)
        print(f"[{name}] done in {time.perf_counter() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
