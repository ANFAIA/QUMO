"""Classical optimisers for variational algorithms.

Interface (SciPy `minimize` semantics, typed for Mojo):

    trait Objective:
        num_params(self) -> Int
        value(mut self, x) raises -> Float64
        value_and_gradient(mut self, x, mut grad) raises -> Float64

    trait Optimizer:
        name(self) -> String
        minimize[O: Objective](self, mut obj: O, x0) raises -> OptimizeResult

Mojo closures are statically typed, so instead of passing callables the
objective is a struct (e.g. `qumo.algorithms.variational.VQEObjective`, which
owns its state-vector buffers and gradient method) - the optimiser is generic
over it and fully monomorphised: no dynamic dispatch in the loop.

Optimisers:
  * GradientDescent (with momentum)        - baseline
  * Adam (Kingma & Ba)                      - robust default for noisy/shot-based losses
  * BFGS (dense inverse Hessian + Armijo)   - fastest for exact simulator gradients
  * SPSA (Spall; Qiskit-style calibration)  - gradient-free, 2 evals/iter, seeded
  * NelderMead (adaptive, Gao & Han 2012)   - gradient-free simplex

`minimize(obj, x0, method="bfgs", ...)` dispatches by name (AI-friendly).
Every run returns an `OptimizeResult` (x, fun, nfev, ngrad, nit, history,
converged, message, method). Error code: QM-E806 (unknown optimiser / bad
configuration).
"""

from std.math import sqrt
from qumo.sim.rng import Rng


trait Objective:
    def num_params(self) -> Int:
        ...

    def value(mut self, x: List[Float64]) raises -> Float64:
        ...

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        ...


struct OptimizeResult(Copyable, Writable):
    var x: List[Float64]
    var fun: Float64
    var nfev: Int
    var ngrad: Int
    var nit: Int
    var history: List[Float64]
    var converged: Bool
    var message: String
    var method: String

    def __init__(out self, method: String):
        self.x = List[Float64]()
        self.fun = 0.0
        self.nfev = 0
        self.ngrad = 0
        self.nit = 0
        self.history = List[Float64]()
        self.converged = False
        self.message = String()
        self.method = method

    def write_to(self, mut w: Some[Writer]):
        w.write("OptimizeResult(method=", self.method, ", fun=", self.fun, ", nit=", self.nit,
                ", nfev=", self.nfev, ", ngrad=", self.ngrad, ", converged=", self.converged,
                ", message='", self.message, "')")


trait Optimizer(Copyable):
    def name(self) -> String:
        ...

    def minimize[O: Objective](self, mut obj: O, x0: List[Float64]) raises -> OptimizeResult:
        ...


def _inf_norm(v: List[Float64]) -> Float64:
    var m = 0.0
    for a in v:
        m = max(m, abs(a))
    return m


def _check_x0[O: Objective](obj: O, x0: List[Float64]) raises:
    if len(x0) != obj.num_params():
        raise Error("QM-E804: x0 has " + String(len(x0)) + " entries but the objective has " + String(obj.num_params()) + " parameters.")


# ---------------------------------------------------------------------------
# Gradient descent (+ momentum)
# ---------------------------------------------------------------------------
struct GradientDescent(Optimizer, Copyable):
    var lr: Float64
    var momentum: Float64
    var maxiter: Int
    var gtol: Float64

    def __init__(out self, lr: Float64 = 0.1, momentum: Float64 = 0.0, maxiter: Int = 500, gtol: Float64 = 1e-6):
        self.lr = lr
        self.momentum = momentum
        self.maxiter = maxiter
        self.gtol = gtol

    def name(self) -> String:
        return "gradient-descent"

    def minimize[O: Objective](self, mut obj: O, x0: List[Float64]) raises -> OptimizeResult:
        _check_x0(obj, x0)
        var res = OptimizeResult(self.name())
        var n = len(x0)
        var x = x0.copy()
        var v = List[Float64](length=n, fill=0.0)
        var g = List[Float64](length=n, fill=0.0)
        var best_f = 1e300
        var best_x = x.copy()
        res.message = "maximum iterations reached"
        for it in range(self.maxiter):
            var f = obj.value_and_gradient(x, g)
            res.nfev += 1
            res.ngrad += 1
            res.history.append(f)
            res.nit = it + 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            if _inf_norm(g) < self.gtol:
                res.converged = True
                res.message = "gradient norm below gtol"
                break
            for i in range(n):
                v[i] = self.momentum * v[i] - self.lr * g[i]
                x[i] += v[i]
        res.x = best_x^
        res.fun = best_f
        return res^


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------
struct Adam(Optimizer, Copyable):
    var lr: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var maxiter: Int
    var gtol: Float64

    def __init__(out self, lr: Float64 = 0.05, beta1: Float64 = 0.9, beta2: Float64 = 0.999, eps: Float64 = 1e-8, maxiter: Int = 500, gtol: Float64 = 1e-6):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.maxiter = maxiter
        self.gtol = gtol

    def name(self) -> String:
        return "adam"

    def minimize[O: Objective](self, mut obj: O, x0: List[Float64]) raises -> OptimizeResult:
        _check_x0(obj, x0)
        var res = OptimizeResult(self.name())
        var n = len(x0)
        var x = x0.copy()
        var m = List[Float64](length=n, fill=0.0)
        var v = List[Float64](length=n, fill=0.0)
        var g = List[Float64](length=n, fill=0.0)
        var best_f = 1e300
        var best_x = x.copy()
        var b1t = 1.0
        var b2t = 1.0
        res.message = "maximum iterations reached"
        for it in range(self.maxiter):
            var f = obj.value_and_gradient(x, g)
            res.nfev += 1
            res.ngrad += 1
            res.history.append(f)
            res.nit = it + 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            if _inf_norm(g) < self.gtol:
                res.converged = True
                res.message = "gradient norm below gtol"
                break
            b1t *= self.beta1
            b2t *= self.beta2
            for i in range(n):
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g[i]
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g[i] * g[i]
                var mh = m[i] / (1.0 - b1t)
                var vh = v[i] / (1.0 - b2t)
                x[i] -= self.lr * mh / (sqrt(vh) + self.eps)
        res.x = best_x^
        res.fun = best_f
        return res^


# ---------------------------------------------------------------------------
# BFGS
# ---------------------------------------------------------------------------
struct BFGS(Optimizer, Copyable):
    """Quasi-Newton with dense inverse-Hessian update and Armijo backtracking.

    Trial points of the line search cost one `value` each; accepted points one
    `value_and_gradient`. Converges when ||grad||_inf < gtol.
    """

    var maxiter: Int
    var gtol: Float64
    var c1: Float64
    var max_step: Float64

    def __init__(out self, maxiter: Int = 200, gtol: Float64 = 1e-7, c1: Float64 = 1e-4, max_step: Float64 = 1.0):
        self.maxiter = maxiter
        self.gtol = gtol
        self.c1 = c1
        self.max_step = max_step

    def name(self) -> String:
        return "bfgs"

    def minimize[O: Objective](self, mut obj: O, x0: List[Float64]) raises -> OptimizeResult:
        _check_x0(obj, x0)
        var res = OptimizeResult(self.name())
        var n = len(x0)
        var x = x0.copy()
        var g = List[Float64](length=n, fill=0.0)
        var f = obj.value_and_gradient(x, g)
        res.nfev += 1
        res.ngrad += 1
        var hinv = List[Float64](length=n * n, fill=0.0)
        for i in range(n):
            hinv[i * n + i] = 1.0
        var p = List[Float64](length=n, fill=0.0)
        var xn = List[Float64](length=n, fill=0.0)
        var gn = List[Float64](length=n, fill=0.0)
        var s = List[Float64](length=n, fill=0.0)
        var y = List[Float64](length=n, fill=0.0)
        var hy = List[Float64](length=n, fill=0.0)
        var first = True
        res.message = "maximum iterations reached"
        for it in range(self.maxiter):
            res.history.append(f)
            res.nit = it + 1
            if _inf_norm(g) < self.gtol:
                res.converged = True
                res.message = "gradient norm below gtol"
                break
            # search direction p = -Hinv g
            var gp = 0.0
            for i in range(n):
                var acc = 0.0
                for j in range(n):
                    acc -= hinv[i * n + j] * g[j]
                p[i] = acc
                gp += acc * g[i]
            if gp >= 0.0:
                # not a descent direction: reset curvature
                for i in range(n):
                    for j in range(n):
                        hinv[i * n + j] = 1.0 if i == j else 0.0
                    p[i] = -g[i]
                gp = 0.0
                for i in range(n):
                    gp -= g[i] * g[i]
            var pn = _inf_norm(p)
            var alpha = 1.0
            if pn > self.max_step:
                alpha = self.max_step / pn
            var accepted = False
            for _ in range(40):
                for i in range(n):
                    xn[i] = x[i] + alpha * p[i]
                var f_trial = obj.value(xn)
                res.nfev += 1
                if f_trial <= f + self.c1 * alpha * gp:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                res.message = "line search failed (flat or noisy objective)"
                res.converged = _inf_norm(g) < 1e3 * self.gtol
                break
            var f_new = obj.value_and_gradient(xn, gn)
            res.nfev += 1
            res.ngrad += 1
            var sy = 0.0
            var yy = 0.0
            for i in range(n):
                s[i] = xn[i] - x[i]
                y[i] = gn[i] - g[i]
                sy += s[i] * y[i]
                yy += y[i] * y[i]
            if sy > 1e-14:
                if first:
                    var scale = sy / yy
                    for i in range(n):
                        for j in range(n):
                            hinv[i * n + j] = scale if i == j else 0.0
                    first = False
                var yhy = 0.0
                for i in range(n):
                    var acc = 0.0
                    for j in range(n):
                        acc += hinv[i * n + j] * y[j]
                    hy[i] = acc
                    yhy += y[i] * acc
                var a = (sy + yhy) / (sy * sy)
                for i in range(n):
                    for j in range(n):
                        hinv[i * n + j] += a * s[i] * s[j] - (hy[i] * s[j] + s[i] * hy[j]) / sy
            for i in range(n):
                x[i] = xn[i]
                g[i] = gn[i]
            f = f_new
        res.x = x^
        res.fun = f
        return res^


# ---------------------------------------------------------------------------
# SPSA
# ---------------------------------------------------------------------------
struct SPSA(Optimizer, Copyable):
    """Simultaneous-perturbation stochastic approximation (Spall 1992/1998).

    a_k = a / (k + 1 + A)^alpha, c_k = c / (k + 1)^gamma, Rademacher
    perturbations from a seeded `Rng` (bit-reproducible). a <= 0 triggers
    Qiskit-style calibration: a is chosen so the first step has magnitude
    `target_step`. Costs 2 objective evaluations per iteration.
    """

    var maxiter: Int
    var a: Float64
    var c: Float64
    var alpha: Float64
    var gamma: Float64
    var stability: Float64
    var target_step: Float64
    var seed: UInt64

    def __init__(out self, maxiter: Int = 300, a: Float64 = -1.0, c: Float64 = 0.1, alpha: Float64 = 0.602, gamma: Float64 = 0.101, stability: Float64 = 0.0, target_step: Float64 = 0.2, seed: UInt64 = 1234):
        self.maxiter = maxiter
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.stability = stability
        self.target_step = target_step
        self.seed = seed

    def name(self) -> String:
        return "spsa"

    def minimize[O: Objective](self, mut obj: O, x0: List[Float64]) raises -> OptimizeResult:
        _check_x0(obj, x0)
        var res = OptimizeResult(self.name())
        var n = len(x0)
        var rng = Rng(self.seed)
        var x = x0.copy()
        var xp = List[Float64](length=n, fill=0.0)
        var xm = List[Float64](length=n, fill=0.0)
        var delta = List[Float64](length=n, fill=0.0)
        var a = self.a
        if a <= 0.0:
            var avg = 0.0
            var samples = 10
            for _ in range(samples):
                for i in range(n):
                    delta[i] = 1.0 if rng.below(2) == 1 else -1.0
                    xp[i] = x[i] + self.c * delta[i]
                    xm[i] = x[i] - self.c * delta[i]
                var fp = obj.value(xp)
                var fm = obj.value(xm)
                res.nfev += 2
                avg += abs(fp - fm) / (2.0 * self.c)
            avg /= Float64(samples)
            a = self.target_step * ((1.0 + self.stability) ** self.alpha) / max(avg, 1e-12)
        for k in range(self.maxiter):
            var ak = a / ((Float64(k) + 1.0 + self.stability) ** self.alpha)
            var ck = self.c / ((Float64(k) + 1.0) ** self.gamma)
            for i in range(n):
                delta[i] = 1.0 if rng.below(2) == 1 else -1.0
                xp[i] = x[i] + ck * delta[i]
                xm[i] = x[i] - ck * delta[i]
            var fp = obj.value(xp)
            var fm = obj.value(xm)
            res.nfev += 2
            var gscale = (fp - fm) / (2.0 * ck)
            for i in range(n):
                x[i] -= ak * gscale / delta[i]
            res.history.append(0.5 * (fp + fm))
            res.nit = k + 1
        res.fun = obj.value(x)
        res.nfev += 1
        res.x = x^
        res.converged = True
        res.message = "completed " + String(self.maxiter) + " SPSA iterations (no gradient-based stopping test)"
        return res^


# ---------------------------------------------------------------------------
# Nelder-Mead
# ---------------------------------------------------------------------------
struct NelderMead(Optimizer, Copyable):
    """Downhill simplex with adaptive coefficients (Gao & Han 2012, SciPy default)."""

    var maxiter: Int
    var initial_step: Float64
    var xatol: Float64
    var fatol: Float64
    var adaptive: Bool

    def __init__(out self, maxiter: Int = 2000, initial_step: Float64 = 0.25, xatol: Float64 = 1e-8, fatol: Float64 = 1e-10, adaptive: Bool = True):
        self.maxiter = maxiter
        self.initial_step = initial_step
        self.xatol = xatol
        self.fatol = fatol
        self.adaptive = adaptive

    def name(self) -> String:
        return "nelder-mead"

    def minimize[O: Objective](self, mut obj: O, x0: List[Float64]) raises -> OptimizeResult:
        _check_x0(obj, x0)
        var res = OptimizeResult(self.name())
        var n = len(x0)
        var nd = Float64(max(n, 1))
        var rho = 1.0
        var chi = 2.0
        var psi = 0.5
        var sigma = 0.5
        if self.adaptive and n > 1:
            chi = 1.0 + 2.0 / nd
            psi = 0.75 - 1.0 / (2.0 * nd)
            sigma = 1.0 - 1.0 / nd
        # simplex stored row-major: (n+1) x n
        var sx = List[Float64](length=(n + 1) * n, fill=0.0)
        var sf = List[Float64](length=n + 1, fill=0.0)
        for v in range(n + 1):
            for i in range(n):
                sx[v * n + i] = x0[i]
            if v > 0:
                sx[v * n + v - 1] += self.initial_step
        var xt = List[Float64](length=n, fill=0.0)
        for v in range(n + 1):
            for i in range(n):
                xt[i] = sx[v * n + i]
            sf[v] = obj.value(xt)
            res.nfev += 1
        var xbar = List[Float64](length=n, fill=0.0)
        var xr = List[Float64](length=n, fill=0.0)
        var xe = List[Float64](length=n, fill=0.0)
        var xc = List[Float64](length=n, fill=0.0)
        res.message = "maximum iterations reached"
        for it in range(self.maxiter):
            # insertion sort by f
            for i in range(1, n + 1):
                var j = i
                while j > 0 and sf[j] < sf[j - 1]:
                    var tf = sf[j]
                    sf[j] = sf[j - 1]
                    sf[j - 1] = tf
                    for k in range(n):
                        var tx = sx[j * n + k]
                        sx[j * n + k] = sx[(j - 1) * n + k]
                        sx[(j - 1) * n + k] = tx
                    j -= 1
            res.history.append(sf[0])
            res.nit = it + 1
            var fspread = 0.0
            var xspread = 0.0
            for v in range(1, n + 1):
                fspread = max(fspread, abs(sf[v] - sf[0]))
                for k in range(n):
                    xspread = max(xspread, abs(sx[v * n + k] - sx[k]))
            if fspread <= self.fatol and xspread <= self.xatol:
                res.converged = True
                res.message = "simplex converged (fatol/xatol)"
                break
            for k in range(n):
                var s = 0.0
                for v in range(n):
                    s += sx[v * n + k]
                xbar[k] = s / nd
            var w = n  # worst vertex
            for k in range(n):
                xr[k] = xbar[k] + rho * (xbar[k] - sx[w * n + k])
            var fr = obj.value(xr)
            res.nfev += 1
            var do_shrink = False
            if fr < sf[0]:
                for k in range(n):
                    xe[k] = xbar[k] + rho * chi * (xbar[k] - sx[w * n + k])
                var fe = obj.value(xe)
                res.nfev += 1
                if fe < fr:
                    for k in range(n):
                        sx[w * n + k] = xe[k]
                    sf[w] = fe
                else:
                    for k in range(n):
                        sx[w * n + k] = xr[k]
                    sf[w] = fr
            elif fr < sf[n - 1]:
                for k in range(n):
                    sx[w * n + k] = xr[k]
                sf[w] = fr
            else:
                if fr < sf[w]:
                    for k in range(n):
                        xc[k] = xbar[k] + psi * rho * (xbar[k] - sx[w * n + k])
                    var fc = obj.value(xc)
                    res.nfev += 1
                    if fc <= fr:
                        for k in range(n):
                            sx[w * n + k] = xc[k]
                        sf[w] = fc
                    else:
                        do_shrink = True
                else:
                    for k in range(n):
                        xc[k] = xbar[k] - psi * (xbar[k] - sx[w * n + k])
                    var fcc = obj.value(xc)
                    res.nfev += 1
                    if fcc < sf[w]:
                        for k in range(n):
                            sx[w * n + k] = xc[k]
                        sf[w] = fcc
                    else:
                        do_shrink = True
            if do_shrink:
                for v in range(1, n + 1):
                    for k in range(n):
                        sx[v * n + k] = sx[k] + sigma * (sx[v * n + k] - sx[k])
                        xt[k] = sx[v * n + k]
                    sf[v] = obj.value(xt)
                    res.nfev += 1
        var best = 0
        for v in range(1, n + 1):
            if sf[v] < sf[best]:
                best = v
        var xb = List[Float64](length=n, fill=0.0)
        for k in range(n):
            xb[k] = sx[best * n + k]
        res.x = xb^
        res.fun = sf[best]
        return res^


# ---------------------------------------------------------------------------
# Name-based dispatch
# ---------------------------------------------------------------------------
def optimizer_names() -> List[String]:
    var out: List[String] = ["bfgs", "adam", "gradient-descent", "spsa", "nelder-mead"]
    return out^


def minimize[O: Objective](mut obj: O, x0: List[Float64], method: String = "bfgs", maxiter: Int = 200, tol: Float64 = 1e-7, lr: Float64 = 0.05, seed: UInt64 = 1234) raises -> OptimizeResult:
    """Minimise `obj` from x0 with an optimiser chosen by name.

    method: "bfgs" | "adam" | "gradient-descent" ("gd") | "spsa" | "nelder-mead" ("nm").
    tol is the gradient tolerance for gradient methods and the f-tolerance for
    Nelder-Mead; lr is the learning rate for adam / gradient descent; seed
    drives SPSA perturbations.
    """
    var m = method.lower()
    if m == "bfgs":
        return BFGS(maxiter=maxiter, gtol=tol).minimize(obj, x0)
    if m == "adam":
        return Adam(lr=lr, maxiter=maxiter, gtol=tol).minimize(obj, x0)
    if m == "gradient-descent" or m == "gd" or m == "sgd":
        return GradientDescent(lr=lr, maxiter=maxiter, gtol=tol).minimize(obj, x0)
    if m == "spsa":
        return SPSA(maxiter=maxiter, seed=seed).minimize(obj, x0)
    if m == "nelder-mead" or m == "nm":
        return NelderMead(maxiter=maxiter, fatol=tol, xatol=sqrt(tol)).minimize(obj, x0)
    raise Error("QM-E806: unknown optimizer '" + method + "'. Hint: use one of bfgs, adam, gradient-descent, spsa, nelder-mead.")


# ---------------------------------------------------------------------------
# Reference objective (optimiser tests / agent sanity checks)
# ---------------------------------------------------------------------------
struct Rosenbrock(Objective, Movable):
    """f(x) = sum_i 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2, minimum 0 at (1, ..., 1)."""

    var n: Int
    var calls: Int

    def __init__(out self, n: Int = 2):
        self.n = n
        self.calls = 0

    def num_params(self) -> Int:
        return self.n

    def value(mut self, x: List[Float64]) raises -> Float64:
        self.calls += 1
        var s = 0.0
        for i in range(self.n - 1):
            var a = x[i + 1] - x[i] * x[i]
            var b = 1.0 - x[i]
            s += 100.0 * a * a + b * b
        return s

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        if len(grad) != self.n:
            grad = List[Float64](length=self.n, fill=0.0)
        for i in range(self.n):
            grad[i] = 0.0
        for i in range(self.n - 1):
            var a = x[i + 1] - x[i] * x[i]
            grad[i] += -400.0 * x[i] * a - 2.0 * (1.0 - x[i])
            grad[i + 1] += 200.0 * a
        return self.value(x)
