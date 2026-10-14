"""Training utilities for quantum machine learning (PennyLane-form).

QML demos train with explicit minibatch loops and PennyLane's step
optimizers. `qumo.algorithms.optimizers.Optimizer.minimize` is a full-batch
driver that returns the best iterate; this module provides the step-by-step
pieces instead, with formulas identical to PennyLane 0.45 (and optax for
Adam with form='optax'), so trajectories agree to machine precision:

  * `StepOptimizer`: gd, momentum, nesterov, adam, adagrad, rmsprop
    (`lookahead` gives Nesterov's evaluation point, `step` updates in place);
  * learning-rate schedules `cosine_lr` (optax cosine_decay_schedule) and
    `step_lr`;
  * `Batcher`: seeded minibatch index streams, or a replay of recorded batches;
  * losses returning (value, d value / d prediction): square, hinge, L1,
    cross-entropy, fidelity; `accuracy`;
  * `vjp`: the vector-Jacobian product of several expectation values in one
    adjoint pass (the backward of a Torch/JAX layer), and `expectations`.

Error codes: QM-E833 (optimizer state / settings), QM-E831 (shape mismatch).
"""

from std.math import sqrt, cos, log, pi, pow

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.rng import Rng
from qumo.sim.runner import evolve
from qumo.algorithms.gradients import ExpectationEngine, Grad


# ---------------------------------------------------------------------------
# Step optimizers (PennyLane formulas)
# ---------------------------------------------------------------------------
struct StepOptimizer(Copyable, Writable):
    """Stateful first-order optimizer applying one update per `step` call.

    Formulas (g = gradient at the evaluation point, t = step count from 1):
      gd:        x <- x - lr g
      momentum:  a <- m a + lr g;  x <- x - a
      nesterov:  as momentum, but g is evaluated at lookahead(x) = x - m a
      adam 'pennylane': m1 <- b1 m1 + (1-b1) g;  m2 <- b2 m2 + (1-b2) g^2;
                 lr_t = lr sqrt(1-b2^t) / (1-b1^t);  x <- x - lr_t m1 / (sqrt(m2) + eps)
      adam 'optax':  x <- x - lr (m1/(1-b1^t)) / (sqrt(m2/(1-b2^t)) + eps)
      adagrad:   s <- s + g^2;  x <- x - lr / sqrt(s + eps) g
      rmsprop:   s <- d s + (1-d) g^2;  x <- x - lr / sqrt(s + eps) g
    Defaults follow PennyLane (adam beta2 = 0.99; pass beta2=0.999 for optax).
    """

    comptime GD = 0
    comptime MOMENTUM = 1
    comptime NESTEROV = 2
    comptime ADAM = 3
    comptime ADAGRAD = 4
    comptime RMSPROP = 5

    var kind: Int
    var lr: Float64
    var momentum_coeff: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var decay: Float64
    var optax_form: Bool
    var t: Int
    var acc: List[Float64]  # momentum a / adam first moment / adagrad, rmsprop accumulator
    var acc2: List[Float64]  # adam second moment

    def __init__(out self, kind: Int, lr: Float64, momentum_coeff: Float64 = 0.0, beta1: Float64 = 0.9,
                 beta2: Float64 = 0.99, eps: Float64 = 1e-8, decay: Float64 = 0.9, optax_form: Bool = False):
        self.kind = kind
        self.lr = lr
        self.momentum_coeff = momentum_coeff
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.decay = decay
        self.optax_form = optax_form
        self.t = 0
        self.acc = List[Float64]()
        self.acc2 = List[Float64]()

    @staticmethod
    def gd(lr: Float64) raises -> StepOptimizer:
        _check_lr(lr)
        return StepOptimizer(StepOptimizer.GD, lr)

    @staticmethod
    def momentum(lr: Float64, m: Float64 = 0.9) raises -> StepOptimizer:
        _check_lr(lr)
        _check_unit("momentum", m)
        return StepOptimizer(StepOptimizer.MOMENTUM, lr, momentum_coeff=m)

    @staticmethod
    def nesterov(lr: Float64, m: Float64 = 0.9) raises -> StepOptimizer:
        _check_lr(lr)
        _check_unit("momentum", m)
        return StepOptimizer(StepOptimizer.NESTEROV, lr, momentum_coeff=m)

    @staticmethod
    def adam(lr: Float64, beta1: Float64 = 0.9, beta2: Float64 = 0.99, eps: Float64 = 1e-8,
             form: String = "pennylane") raises -> StepOptimizer:
        _check_lr(lr)
        _check_unit("beta1", beta1)
        _check_unit("beta2", beta2)
        var f = form.lower()
        if f != "pennylane" and f != "optax":
            raise Error("QM-E833: StepOptimizer.adam(): unknown form '" + form + "'. Hint: use 'pennylane' (eps outside the bias-corrected step) or 'optax'.")
        return StepOptimizer(StepOptimizer.ADAM, lr, beta1=beta1, beta2=beta2, eps=eps, optax_form=(f == "optax"))

    @staticmethod
    def adagrad(lr: Float64, eps: Float64 = 1e-8) raises -> StepOptimizer:
        _check_lr(lr)
        return StepOptimizer(StepOptimizer.ADAGRAD, lr, eps=eps)

    @staticmethod
    def rmsprop(lr: Float64, decay: Float64 = 0.9, eps: Float64 = 1e-8) raises -> StepOptimizer:
        _check_lr(lr)
        _check_unit("decay", decay)
        return StepOptimizer(StepOptimizer.RMSPROP, lr, decay=decay, eps=eps)

    def name(self) -> String:
        if self.kind == StepOptimizer.GD:
            return "gd"
        if self.kind == StepOptimizer.MOMENTUM:
            return "momentum"
        if self.kind == StepOptimizer.NESTEROV:
            return "nesterov"
        if self.kind == StepOptimizer.ADAM:
            return "adam-optax" if self.optax_form else "adam"
        if self.kind == StepOptimizer.ADAGRAD:
            return "adagrad"
        return "rmsprop"

    def lookahead(self, x: List[Float64]) -> List[Float64]:
        """Where to evaluate the gradient for the next step: x - m a for
        Nesterov after the first step, x itself otherwise."""
        var out = x.copy()
        if self.kind == StepOptimizer.NESTEROV and len(self.acc) == len(x):
            for i in range(len(x)):
                out[i] = x[i] - self.momentum_coeff * self.acc[i]
        return out^

    def step(mut self, mut x: List[Float64], grad: List[Float64]) raises:
        """One update of `x` in place with the gradient `grad` (evaluated at
        lookahead(x) for Nesterov). QM-E833 if a length changes."""
        var n = len(x)
        if len(grad) != n:
            raise Error("QM-E833: StepOptimizer.step(): gradient has " + String(len(grad)) + " entries for " + String(n) + " parameters.")
        if self.t > 0 and len(self.acc) != n:
            raise Error("QM-E833: StepOptimizer.step(): the parameter vector changed length (" + String(len(self.acc)) + " -> " + String(n) + "). Hint: call reset() before optimising a different vector.")
        if self.t == 0:
            self.acc = List[Float64](length=n, fill=0.0)
            self.acc2 = List[Float64](length=n, fill=0.0)
        self.t += 1
        var k = self.kind
        if k == StepOptimizer.GD:
            for i in range(n):
                x[i] = x[i] - self.lr * grad[i]
        elif k == StepOptimizer.MOMENTUM or k == StepOptimizer.NESTEROV:
            for i in range(n):
                self.acc[i] = self.momentum_coeff * self.acc[i] + self.lr * grad[i]
                x[i] = x[i] - self.acc[i]
        elif k == StepOptimizer.ADAM:
            var b1t = pow(self.beta1, Float64(self.t))
            var b2t = pow(self.beta2, Float64(self.t))
            var lr_t = self.lr * sqrt(1.0 - b2t) / (1.0 - b1t)
            for i in range(n):
                self.acc[i] = self.beta1 * self.acc[i] + (1.0 - self.beta1) * grad[i]
                self.acc2[i] = self.beta2 * self.acc2[i] + (1.0 - self.beta2) * (grad[i] * grad[i])
                if self.optax_form:
                    var mh = self.acc[i] / (1.0 - b1t)
                    var vh = self.acc2[i] / (1.0 - b2t)
                    x[i] = x[i] - self.lr * (mh / (sqrt(vh) + self.eps))
                else:
                    x[i] = x[i] - lr_t * self.acc[i] / (sqrt(self.acc2[i]) + self.eps)
        elif k == StepOptimizer.ADAGRAD:
            for i in range(n):
                self.acc[i] = self.acc[i] + grad[i] * grad[i]
                x[i] = x[i] - self.lr / sqrt(self.acc[i] + self.eps) * grad[i]
        else:
            for i in range(n):
                self.acc[i] = self.decay * self.acc[i] + (1.0 - self.decay) * (grad[i] * grad[i])
                x[i] = x[i] - self.lr / sqrt(self.acc[i] + self.eps) * grad[i]

    def set_lr(mut self, lr: Float64) raises:
        """Change the step size (schedules); the moments are kept."""
        _check_lr(lr)
        self.lr = lr

    def reset(mut self):
        """Forget the accumulated state (PennyLane `reset()`)."""
        self.t = 0
        self.acc = List[Float64]()
        self.acc2 = List[Float64]()

    def write_to(self, mut w: Some[Writer]):
        w.write("StepOptimizer(", self.name(), ", lr=", self.lr, ", steps=", self.t, ")")


def _check_lr(lr: Float64) raises:
    if not (lr >= 0.0):
        raise Error("QM-E833: learning rate must be >= 0, got " + String(lr) + ".")


def _check_unit(name: String, v: Float64) raises:
    if not (v >= 0.0 and v < 1.0):
        raise Error("QM-E833: " + name + " must lie in [0, 1), got " + String(v) + ".")


# ---------------------------------------------------------------------------
# Learning-rate schedules
# ---------------------------------------------------------------------------
def cosine_lr(step: Int, total: Int, lr0: Float64, lr_min: Float64 = 0.0) -> Float64:
    """optax.cosine_decay_schedule(lr0, total, alpha=lr_min/lr0) at `step`:
    lr0 ((1 - alpha) 0.5 (1 + cos(pi min(step, total) / total)) + alpha)."""
    if total <= 0:
        return lr0
    var alpha = lr_min / lr0 if lr0 != 0.0 else 0.0
    var count = Float64(min(max(step, 0), total))
    var cd = 0.5 * (1.0 + cos(pi * count / Float64(total)))
    return lr0 * ((1.0 - alpha) * cd + alpha)


def step_lr(step: Int, every: Int, gamma: Float64, lr0: Float64) -> Float64:
    """Step decay: lr0 * gamma^(step // every) (torch StepLR)."""
    if every <= 0:
        return lr0
    return lr0 * pow(gamma, Float64(max(step, 0) // every))


# ---------------------------------------------------------------------------
# Minibatches
# ---------------------------------------------------------------------------
struct Batcher(Copyable):
    """Seeded stream of minibatch index lists over 0..n-1.

    Default: epochs of a seeded permutation (`shuffle`) cut into consecutive
    batches (torch DataLoader; the last partial batch is kept unless
    `drop_last`, which mirrors PennyLane's `iterate_minibatches`).
    `replacement=True`: every batch draws `batch` indices uniformly with
    replacement (PennyLane's `np.random.randint(0, n, (batch,))`).
    `Batcher.replay(batches)` replays recorded batches (fixtures) cyclically.
    """

    var n: Int
    var batch: Int
    var shuffle: Bool
    var drop_last: Bool
    var replacement: Bool
    var rng: Rng
    var order: List[Int]
    var pos: Int
    var epoch: Int
    var replay_batches: List[List[Int]]
    var replay_pos: Int

    def __init__(out self, n: Int, batch: Int, seed: UInt64, shuffle: Bool = True,
                 drop_last: Bool = False, replacement: Bool = False) raises:
        if n < 1 or batch < 1 or (drop_last and batch > n):
            raise Error("QM-E831: Batcher needs n >= 1 and 1 <= batch (<= n with drop_last), got n=" + String(n) + ", batch=" + String(batch) + ".")
        self.n = n
        self.batch = batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.replacement = replacement
        self.rng = Rng(seed)
        self.order = List[Int]()
        self.pos = n  # forces a new epoch on the first next()
        self.epoch = 0
        self.replay_batches = List[List[Int]]()
        self.replay_pos = 0

    @staticmethod
    def replay(indices: List[List[Int]]) raises -> Batcher:
        """A Batcher that returns `indices[0]`, `indices[1]`, ... (cycling)."""
        if len(indices) == 0:
            raise Error("QM-E831: Batcher.replay() needs at least one batch.")
        var mx = 0
        for b in indices:
            if len(b) == 0:
                raise Error("QM-E831: Batcher.replay(): empty batch.")
            for i in b:
                if i < 0:
                    raise Error("QM-E831: Batcher.replay(): negative index " + String(i) + ".")
                mx = max(mx, i)
        var out = Batcher(mx + 1, len(indices[0]), 0, False)
        out.replay_batches = indices.copy()
        return out^

    def batches_per_epoch(self) -> Int:
        if len(self.replay_batches) > 0:
            return len(self.replay_batches)
        if self.drop_last:
            return self.n // self.batch
        return (self.n + self.batch - 1) // self.batch

    def _new_epoch(mut self):
        self.order = List[Int](capacity=self.n)
        for i in range(self.n):
            self.order.append(i)
        if self.shuffle:
            var k = self.n - 1
            while k > 0:
                var j = self.rng.below(k + 1)
                var t = self.order[k]
                self.order[k] = self.order[j]
                self.order[j] = t
                k -= 1
        self.pos = 0
        self.epoch += 1

    def next(mut self) -> List[Int]:
        """Indices of the next minibatch."""
        if len(self.replay_batches) > 0:
            var b = self.replay_batches[self.replay_pos].copy()
            self.replay_pos = (self.replay_pos + 1) % len(self.replay_batches)
            return b^
        if self.replacement:
            var out = List[Int](capacity=self.batch)
            for _ in range(self.batch):
                out.append(self.rng.below(self.n))
            return out^
        var remaining = self.n - self.pos
        if remaining <= 0 or (self.drop_last and remaining < self.batch):
            self._new_epoch()
        var hi = min(self.pos + self.batch, self.n)
        var out = List[Int](capacity=hi - self.pos)
        for i in range(self.pos, hi):
            out.append(self.order[i])
        self.pos = hi
        return out^


# ---------------------------------------------------------------------------
# Losses: (value, d value / d prediction)
# ---------------------------------------------------------------------------
def _same_len(a: Int, b: Int, who: String) raises:
    if a != b or a == 0:
        raise Error("QM-E831: " + who + "(): " + String(a) + " predictions for " + String(b) + " labels.")


def square_loss(pred: List[Float64], y: List[Float64]) raises -> Tuple[Float64, List[Float64]]:
    """mean_i (y_i - p_i)^2 (PennyLane tutorial `square_loss`); d/dp_i = -2 (y_i - p_i) / N."""
    _same_len(len(pred), len(y), "square_loss")
    var n = Float64(len(pred))
    var total = 0.0
    var g = List[Float64](capacity=len(pred))
    for i in range(len(pred)):
        var r = y[i] - pred[i]
        total += r * r
        g.append(-2.0 * r / n)
    return (total / n, g^)


def hinge_loss(pred: List[Float64], y: List[Float64]) raises -> Tuple[Float64, List[Float64]]:
    """mean_i max(0, 1 - y_i p_i) for labels +/-1; d/dp_i = -y_i / N where the margin is violated."""
    _same_len(len(pred), len(y), "hinge_loss")
    var n = Float64(len(pred))
    var total = 0.0
    var g = List[Float64](capacity=len(pred))
    for i in range(len(pred)):
        var m = 1.0 - y[i] * pred[i]
        if m > 0.0:
            total += m
            g.append(-y[i] / n)
        else:
            g.append(0.0)
    return (total / n, g^)


def l1_loss(pred: List[Float64], y: List[Float64]) raises -> Tuple[Float64, List[Float64]]:
    """mean_i |p_i - y_i| (torch L1Loss); d/dp_i = sign(p_i - y_i) / N (0 at a tie)."""
    _same_len(len(pred), len(y), "l1_loss")
    var n = Float64(len(pred))
    var total = 0.0
    var g = List[Float64](capacity=len(pred))
    for i in range(len(pred)):
        var d = pred[i] - y[i]
        total += abs(d)
        g.append((1.0 if d > 0.0 else (-1.0 if d < 0.0 else 0.0)) / n)
    return (total / n, g^)


def cross_entropy(probs: List[List[Float64]], labels: List[Int]) raises -> Tuple[Float64, List[Float64]]:
    """-mean_i log probs[i][labels[i]] (probabilities, not logits).

    The gradient is w.r.t. the probabilities, flattened row-major
    (entry i * C + c): -1 / (N p_i) at the label, 0 elsewhere.
    """
    _same_len(len(probs), len(labels), "cross_entropy")
    var n = len(probs)
    var nc = len(probs[0])
    var total = 0.0
    var g = List[Float64](length=n * nc, fill=0.0)
    for i in range(n):
        if len(probs[i]) != nc:
            raise Error("QM-E831: cross_entropy(): row " + String(i) + " has " + String(len(probs[i])) + " classes, row 0 has " + String(nc) + ".")
        var c = labels[i]
        if c < 0 or c >= nc:
            raise Error("QM-E831: cross_entropy(): label " + String(c) + " outside 0.." + String(nc - 1) + ".")
        var p = max(probs[i][c], 1e-300)
        total -= log(p)
        g[i * nc + c] = -1.0 / (Float64(n) * p)
    return (total / Float64(n), g^)


def fidelity_loss(fid: List[Float64]) raises -> Tuple[Float64, List[Float64]]:
    """mean_i (1 - F_i)^2 (data re-uploading cost); d/dF_i = -2 (1 - F_i) / N."""
    if len(fid) == 0:
        raise Error("QM-E831: fidelity_loss(): no fidelities.")
    var n = Float64(len(fid))
    var total = 0.0
    var g = List[Float64](capacity=len(fid))
    for f in fid:
        var r = 1.0 - f
        total += r * r
        g.append(-2.0 * r / n)
    return (total / n, g^)


def _sign(v: Float64) -> Float64:
    if v > 0.0:
        return 1.0
    if v < 0.0:
        return -1.0
    return 0.0


def accuracy(pred: List[Float64], y: List[Float64]) raises -> Float64:
    """Fraction of sign(pred_i) == sign(y_i) (labels +/-1; sign(0) = 0 as numpy)."""
    _same_len(len(pred), len(y), "accuracy")
    var ok = 0
    for i in range(len(pred)):
        if _sign(pred[i]) == _sign(y[i]):
            ok += 1
    return Float64(ok) / Float64(len(pred))


# ---------------------------------------------------------------------------
# Multi-output expectation values and their vector-Jacobian product
# ---------------------------------------------------------------------------
def expectations(circuit: Circuit, observables: List[PauliSum], params: List[Float64]) raises -> List[Float64]:
    """<O_j> for every observable from one state evolution (a layer's forward)."""
    var sv = evolve(circuit, params)
    var out = List[Float64](capacity=len(observables))
    for o in observables:
        out.append(sv.expectation(o))
    return out^


def vjp(circuit: Circuit, observables: List[PauliSum], params: List[Float64], cotangent: List[Float64]) raises -> List[Float64]:
    """sum_j u_j d<O_j>/dtheta in ONE adjoint pass on the observable sum_j u_j O_j
    (the backward of a quantum layer with outputs <O_j> and upstream gradient u)."""
    if len(observables) != len(cotangent) or len(observables) == 0:
        raise Error("QM-E831: vjp(): " + String(len(cotangent)) + " cotangent entries for " + String(len(observables)) + " observables.")
    var h = PauliSum()
    for j in range(len(observables)):
        for t in observables[j].terms:
            h.add(cotangent[j] * t.coeff, t.x, t.z)
    var eng = ExpectationEngine(circuit, h)
    return eng.gradient(params, Grad.ADJOINT)
