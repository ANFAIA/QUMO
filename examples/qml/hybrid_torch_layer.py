#!/usr/bin/env python3
"""Hybrid classical-quantum model: a Torch layer backed by QUMO adjoint gradients.

Reproduces: PennyLane - Turning quantum nodes into Torch Layers
Source: https://pennylane.ai/qml/demos/tutorial_qnn_module_torch
Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_qnn_module_torch/demo.py (Apache-2.0)

Moons data (examples/qml/data/moons_200.csv = make_moons(200, noise=0.1,
random_state=42)); model Linear(2, 2) -> QUMO layer -> Linear(2, 2) -> softmax.
The QUMO layer is AngleEmbedding(x) + BasicEntanglerLayers(6 x 2) on two qubits
with outputs (<Z0>, <Z1>). It is a torch.autograd.Function: the forward pass
calls qumo_py.value_and_gradient per sample, the backward pass calls
qumo_py.vjp (one adjoint pass on sum_j u_j Z_j per sample), so Torch's
autograd sees exact gradients with respect to both the weights and the inputs.
SGD(0.2), L1 loss on one-hot labels, batch 5, 6 epochs. Initial weights and the
DataLoader order are the fixture's (torch.manual_seed(42) in the generator).

Checks, against examples/qml/reference/pennylane_reference.json (PennyLane 0.45.1 + torch 2.14.1):
  * layer forward pass and Jacobian (inputs and weights) at fixed float64 values (1e-10);
  * the per-epoch average loss equals PennyLane's (1e-6; the page shows 0.4943 -> 0.1666);
  * final accuracy equals the fixture (page: about 0.855).

PHASE 2: needs the qumo_py bindings value_and_gradient and vjp (plan item L16)
and torch; until then examples/run_examples.py reports SKIP.

Run: bench/.venv/bin/python examples/qml/hybrid_torch_layer.py   (after `make py`, with torch installed)
example-meta: {"framework": "pennylane", "tier": "fast", "timeout": 120, "requires": ["torch", "qumo_py", "phase2:qumo_py.vjp"], "args": []}
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
from _checks import CheckLog  # noqa: E402

FIXTURE = HERE / "reference" / "pennylane_reference.json"
N_LAYERS = 6
N_QUBITS = 2
N_WEIGHTS = N_LAYERS * N_QUBITS


def load_qumo_py():
    """Import qumo_py from PYTHONPATH, else build/ (QUMO_PY_PATH names the .so when set)."""
    try:
        import qumo_py  # noqa: F401
        return sys.modules["qumo_py"]
    except ImportError:
        pass
    so = os.environ.get("QUMO_PY_PATH")
    cand = Path(so).parent if so else REPO / "build"
    sys.path.insert(0, str(cand))
    import qumo_py  # noqa: F401
    return sys.modules["qumo_py"]


def layer_qasm() -> str:
    """OpenQASM 3 of the quantum layer; parameters are the inputs in declaration
    order: weights w_l_q (l-major), then features x_0, x_1 (weights first, as in QUMO's QML layer)."""
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            lines.append(f"input float[64] w_{l}_{q};")
    for q in range(N_QUBITS):
        lines.append(f"input float[64] x_{q};")
    lines.append(f"qubit[{N_QUBITS}] q;")
    for q in range(N_QUBITS):                      # AngleEmbedding (rotation X)
        lines.append(f"rx(x_{q}) q[{q}];")
    for l in range(N_LAYERS):                      # BasicEntanglerLayers: RX layer + one CNOT for 2 wires
        for q in range(N_QUBITS):
            lines.append(f"rx(w_{l}_{q}) q[{q}];")
        lines.append("cx q[0], q[1];")
    return "\n".join(lines) + "\n"


OBSERVABLES = ["Z0", "Z1"]


class QumoLayer:
    """Values and vector-Jacobian products of (<Z0>, <Z1>) through qumo_py (float64)."""

    def __init__(self, qp):
        missing = [f for f in ("value_and_gradient", "vjp", "load_qasm") if not hasattr(qp, f)]
        if missing:
            raise SystemExit(f"qumo_py lacks {missing}: these bindings land in phase 2 (plan item L16)")
        self.qp = qp
        self.circ = qp.load_qasm(layer_qasm())

    def values(self, params: list[float]) -> list[float]:
        return [float(self.qp.value_and_gradient(self.circ, o, params)[0]) for o in OBSERVABLES]

    def vjp(self, params: list[float], cot: list[float]) -> list[float]:
        return [float(v) for v in self.qp.vjp(self.circ, OBSERVABLES, params, cot)]


def make_function(layer: QumoLayer):
    import torch

    class QumoFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs, weights):
            w = weights.detach().double().reshape(-1).tolist()
            xs = inputs.detach().double().tolist()
            out = [layer.values(w + x) for x in xs]
            ctx.save_for_backward(inputs, weights)
            return torch.tensor(out, dtype=inputs.dtype)

        @staticmethod
        def backward(ctx, grad_out):
            inputs, weights = ctx.saved_tensors
            w = weights.detach().double().reshape(-1).tolist()
            xs = inputs.detach().double().tolist()
            gw = torch.zeros(N_WEIGHTS, dtype=torch.float64)
            gx = torch.zeros((len(xs), N_QUBITS), dtype=torch.float64)
            for b, x in enumerate(xs):
                g = layer.vjp(w + x, grad_out[b].detach().double().tolist())
                gw += torch.tensor(g[:N_WEIGHTS], dtype=torch.float64)
                gx[b] = torch.tensor(g[N_WEIGHTS:], dtype=torch.float64)
            return gx.to(inputs.dtype), gw.reshape(weights.shape).to(weights.dtype)

    return QumoFn


def main() -> None:
    import torch

    log = CheckLog("hybrid_torch_layer")
    fx = json.loads(FIXTURE.read_text())["hybrid_torch_layer"]
    layer = QumoLayer(load_qumo_py())
    Fn = make_function(layer)

    # 1. layer forward pass and Jacobian at fixed float64 values
    xin = fx["layer_fixed_inputs"]
    wq = [v for row in fx["layer_fixed_weights"] for v in row]
    out = layer.values(wq + xin)
    log.close("layer forward max abs diff", max(abs(a - b) for a, b in zip(out, fx["layer_fixed_out"])), 0.0, 1e-10)
    md = 0.0
    for j in range(2):  # row j of the Jacobian = vjp with the unit cotangent e_j
        g = layer.vjp(wq + xin, [1.0 if k == j else 0.0 for k in range(2)])
        md = max(md, max(abs(a - b) for a, b in zip(g[N_WEIGHTS:], fx["layer_fixed_jac_inputs"][j])))
        md = max(md, max(abs(a - b) for a, b in zip(g[:N_WEIGHTS], fx["layer_fixed_jac_weights"][j])))
    log.close("layer Jacobian (inputs and weights) max abs diff", md, 0.0, 1e-10)

    # 2. the demo's training run with the fixture's initial weights and batch order
    rows = list(csv.reader(open(HERE / "data" / "moons_200.csv")))[1:]
    X = torch.tensor([[float(r[0]), float(r[1])] for r in rows]).float()
    y = torch.tensor([int(r[2]) for r in rows])
    y_hot = torch.zeros((len(y), 2)).scatter(1, y.unsqueeze(1), 1).float()

    init = fx["init"]
    c1, c2 = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    with torch.no_grad():
        c1.weight.copy_(torch.tensor(init["clayer_1.weight"]))
        c1.bias.copy_(torch.tensor(init["clayer_1.bias"]))
        c2.weight.copy_(torch.tensor(init["clayer_2.weight"]))
        c2.bias.copy_(torch.tensor(init["clayer_2.bias"]))
    qw = torch.nn.Parameter(torch.tensor(init["qlayer.weights"]).float())

    def model(xs):
        return torch.softmax(c2(Fn.apply(c1(xs), qw)), dim=1)

    opt = torch.optim.SGD(list(c1.parameters()) + [qw] + list(c2.parameters()), lr=0.2)
    loss_fn = torch.nn.L1Loss()
    avg = []
    t0 = time.perf_counter()
    for epoch, order in enumerate(fx["batch_order"]):
        running = 0.0
        for ids in order:
            idx = torch.tensor(ids)
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y_hot[idx])
            loss.backward()
            opt.step()
            running += float(loss.detach())
        avg.append(running / len(order))
        print(f"  epoch {epoch + 1}: average loss {avg[-1]:.4f}  (PennyLane {fx['avg_loss_per_epoch'][epoch]:.4f})")
    secs = time.perf_counter() - t0
    log.close("average loss per epoch max abs diff", max(abs(a - b) for a, b in zip(avg, fx["avg_loss_per_epoch"])), 0.0, 1e-6)
    with torch.no_grad():
        acc = int((torch.argmax(model(X), dim=1) == y).sum()) / len(y)
    print(f"accuracy {acc:.3f} (page {fx['page_values']['accuracy']})")
    log.close("final accuracy", acc, fx["accuracy"], 1e-12)
    print(f"timing (indicative, different engines): QUMO layer {secs:.2f} s; PennyLane TorchLayer {fx['train_seconds']:.2f} s (fixture)")
    log.finish()


if __name__ == "__main__":
    main()
