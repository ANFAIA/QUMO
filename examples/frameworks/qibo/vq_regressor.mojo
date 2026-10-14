"""One-qubit variational regressor for sin(2x): parameter shift and Adam.

Reproduces: Qibo - Variational Quantum Regressor (re-uploading, parameter-shift rule)
Source: https://qibo.science/qibo/stable/code-examples/tutorials/vqregressor/README.html (Qibo docs, Apache-2.0)

The Qibo tutorial fits y = sin(2x) on x in [-1, 1] with a single-qubit
re-uploading circuit. It uses 30 training points, MSE loss, gradients from the
parameter-shift rule (shift pi/2, r = 1/2), and Adam with learning rate 0.05
for 200 epochs. Its command line uses 1 layer; the gate layout appears only as
an image, and the page prints no numbers. The prediction is <Z> = P(0) - P(1).

QUMO version. Each layer is RY(w_l x) RY(b_l) RZ(c_l), applied to |0>. A
`Param` is affine (coeff * theta + offset), so the data point x is folded into
the coefficient of w_l when the circuit is built for that point. The two RY
rotations compose to RY(w_l x + b_l), the usual re-uploading encoding. This
example uses 2 layers, i.e. 6 parameters. The loss is
  L = (1/M) sum_i (<Z>(x_i) - y_i)^2,
and its gradient is assembled by the chain rule from the per-point
`value_and_gradient(..., Grad.PARAM_SHIFT)`. The existing `Adam` optimiser
(qumo/algorithms/optimizers.mojo) minimises L through the `Objective` trait.

Checks:
- the parameter-shift loss gradient equals the adjoint one (1e-12);
- the training MSE decreases;
- the test MSE on a 100-point grid of [-1, 1] is < 0.01.

Run: .venv/bin/mojo run -I . examples/frameworks/qibo/vq_regressor.mojo
example-meta: {"framework": "qibo", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import sin
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.sim.rng import Rng
from qumo.algorithms.gradients import Grad, value_and_gradient, expectation_value
from qumo.algorithms.optimizers import Objective, Adam
from qumo.testing.checks import CheckLog

comptime LAYERS = 2
comptime NPARAMS = 3 * LAYERS
comptime NDATA = 30
comptime NTEST = 100


def model(x: Float64) -> Circuit:
    """Re-uploading circuit for one data point; params [w_l, b_l, c_l] per layer."""
    var c = Circuit(1, name="vqregressor")
    var th = c.parameters("theta", NPARAMS)
    for l in range(LAYERS):
        c.ry(th[3 * l] * x, 0)  # RY(w_l x)
        c.ry(th[3 * l + 1], 0)  # RY(b_l)
        c.rz(th[3 * l + 2], 0)  # RZ(c_l)
    return c^


struct RegressionLoss(Objective, Movable):
    """MSE over the training set; gradient method selectable."""

    var xs: List[Float64]
    var ys: List[Float64]
    var z: PauliSum
    var method: Int

    def __init__(out self, xs: List[Float64], ys: List[Float64], method: Int) raises:
        self.xs = xs.copy()
        self.ys = ys.copy()
        self.z = parse_pauli_sum("Z0")
        self.method = method

    def num_params(self) -> Int:
        return NPARAMS

    def value(mut self, x: List[Float64]) raises -> Float64:
        var s = 0.0
        for i in range(len(self.xs)):
            var r = expectation_value(model(self.xs[i]), self.z, x) - self.ys[i]
            s += r * r
        return s / Float64(len(self.xs))

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        var m = Float64(len(self.xs))
        var total = List[Float64](length=NPARAMS, fill=0.0)
        var s = 0.0
        for i in range(len(self.xs)):
            var g = List[Float64]()
            var f = value_and_gradient(model(self.xs[i]), self.z, x, g, self.method)
            var r = f - self.ys[i]
            s += r * r
            for k in range(NPARAMS):
                total[k] += 2.0 * r * g[k] / m
        grad = total^
        return s / m


def main() raises:
    var log = CheckLog("qibo/vq_regressor")
    var rng = Rng(2024)
    var xs = List[Float64]()
    var ys = List[Float64]()
    for _ in range(NDATA):
        var x = 2.0 * rng.uniform() - 1.0
        xs.append(x)
        ys.append(sin(2.0 * x))
    var x0 = rng.normal_list(NPARAMS, 0.5)

    # Parameter shift versus adjoint at the initial point.
    var loss_ps = RegressionLoss(xs, ys, Grad.PARAM_SHIFT)
    var loss_adj = RegressionLoss(xs, ys, Grad.ADJOINT)
    var g_ps = List[Float64]()
    var g_adj = List[Float64]()
    var l0 = loss_ps.value_and_gradient(x0, g_ps)
    _ = loss_adj.value_and_gradient(x0, g_adj)
    var dg = 0.0
    for k in range(NPARAMS):
        dg = max(dg, abs(g_ps[k] - g_adj[k]))
    print("initial training MSE", l0, "; max |grad_PS - grad_adjoint| =", dg)
    _ = log.close("parameter-shift gradient equals adjoint", dg, 0.0, 1e-12)

    # Adam, lr 0.05, 200 epochs (the page's settings), parameter-shift gradients.
    var adam = Adam(lr=0.05, maxiter=200, gtol=1e-10)
    var res = adam.minimize(loss_ps, x0)
    print("Adam:", res)
    print("trained parameters [w, b, c] per layer:")
    for l in range(LAYERS):
        print("  layer", l, ":", res.x[3 * l], res.x[3 * l + 1], res.x[3 * l + 2])
    _ = log.check("training MSE decreases", res.fun < l0, "initial=" + String(l0) + " final=" + String(res.fun))

    # Test grid.
    var z = parse_pauli_sum("Z0")
    var mse = 0.0
    print("   x       sin(2x)     model")
    for i in range(NTEST):
        var x = -1.0 + 2.0 * Float64(i) / Float64(NTEST - 1)
        var p = expectation_value(model(x), z, res.x)
        var r = p - sin(2.0 * x)
        mse += r * r
        if i % 11 == 0:
            print("  ", x, "  ", sin(2.0 * x), "  ", p)
    mse /= Float64(NTEST)
    print("test MSE on", NTEST, "grid points:", mse)
    _ = log.within("test MSE < 0.01", mse, 0.0, 0.01)
    log.finish()
