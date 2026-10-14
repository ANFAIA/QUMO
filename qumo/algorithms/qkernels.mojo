"""Quantum kernels, kernel-target alignment and kernel classifiers.

A feature embedding is any unitary `Circuit` whose parameters are
`weights ++ features` (the QML convention of qumo.algorithms.qml: declare the
trainable weights first, then the features). The fidelity kernel is
k(a, b) = |<psi(b)|psi(a)>|^2 with |psi(x)> = U(w, x)|0>.

Three routes, all giving the same numbers (tests/test_qkernels.mojo):
  * state route (`kernel_matrix`, `kernel_matrix_symmetric`): one simulation
    per data point plus inner products - exact and fastest on a simulator;
  * compute-uncompute circuit (`kernel_circuit`): U(w, a) U(w, b)^dagger with
    k = <0|...|0> probability, differentiable by the adjoint method
    (`kernel_value_and_grad`, `kernel_matrix_symmetric_and_grad`);
  * sampled (`sampled_kernel_entry`): the frequency of 0...0 over `shots`
    runs of the kernel circuit - what a QPU measures.

Kernel matrices are row-major `List[Float64]` (entry i * M + j).
Classifiers: `svc_fit` (C-SVC dual by SMO with libsvm's second-order
working-set selection) and `kernel_ridge_fit`.

Error code: QM-E831 (shape mismatch); kernel circuits with more than 12
qubits for gradients raise QM-E830 (projector observable size).
"""

from std.math import sqrt
from std.runtime import parallelism_level
from max.algorithm import parallelize

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve_into, sample, bitstring
from qumo.algorithms.gradients import ExpectationEngine, Grad, batch_value_and_gradient
from qumo.algorithms.qml import projector_zero
from qumo.algorithms.linalg import jacobi_eigh


comptime KERNEL_PAR_MAX_QUBITS = 12


# ---------------------------------------------------------------------------
# State route
# ---------------------------------------------------------------------------
def _point_params(weights: List[Float64], x: List[Float64]) -> List[Float64]:
    var p = weights.copy()
    for v in x:
        p.append(v)
    return p^


def embedding_states(embed: Circuit, X: List[List[Float64]], weights: List[Float64] = List[Float64]()) raises -> List[StateVector]:
    """|psi(x_i)> = U(weights, x_i)|0> for every row of X (parallel for <= 12 qubits)."""
    embed.validate()
    var nq = embed.num_qubits
    var np_ = embed.num_params()
    var nx = len(X)
    var sets = List[List[Float64]](capacity=nx)
    for i in range(nx):
        if len(weights) + len(X[i]) != np_:
            raise Error("QM-E831: embedding has " + String(np_) + " parameters " + embed.param_names_str() + " but weights (" + String(len(weights)) + ") + features of row " + String(i) + " (" + String(len(X[i])) + ") give " + String(len(weights) + len(X[i])) + ". Hint: declare weights first, then features.")
        sets.append(_point_params(weights, X[i]))
    var states = List[StateVector](capacity=nx)
    for _ in range(nx):
        states.append(StateVector(nq))
    if nq > KERNEL_PAR_MAX_QUBITS or nx < 2:
        for i in range(nx):
            evolve_into(states[i], embed, sets[i])
        return states^
    var sp = states.unsafe_ptr().as_unsafe_any_origin()
    var failed = List[Int](length=nx, fill=0)
    var fp = failed.unsafe_ptr().as_unsafe_any_origin()

    def task(i: Int) {imm embed, imm sets, imm sp, imm fp}:
        try:
            evolve_into(sp[unsafe_offset=i], embed, sets[i])
        except:
            fp[unsafe_offset=i] = 1

    parallelize(task, nx)
    _ = len(states)
    for i in range(nx):
        if failed[i] != 0:
            raise Error("QM-E831: embedding_states() failed on row " + String(i) + ".")
    return states^


def _overlaps(A: List[StateVector], B: List[StateVector], symmetric: Bool) -> List[Float64]:
    """|<B_j|A_i>|^2 row-major (len(A) x len(B)); symmetric fills j >= i and mirrors, diag = 1."""
    var na = len(A)
    var nb = len(B)
    var out = List[Float64](length=na * nb, fill=0.0)
    var op = out.unsafe_ptr().as_unsafe_any_origin()

    def row(i: Int) {imm A, imm B, imm op, imm nb, imm symmetric}:
        var j0 = i + 1 if symmetric else 0
        if symmetric:
            op[unsafe_offset=i * nb + i] = 1.0
        for j in range(j0, nb):
            var ip = B[j].inner(A[i])
            op[unsafe_offset=i * nb + j] = ip[0] * ip[0] + ip[1] * ip[1]

    if na >= 2 and A[0].num_qubits <= KERNEL_PAR_MAX_QUBITS:
        parallelize(row, na)
    else:
        for i in range(na):
            row(i)
    _ = len(out)
    if symmetric:
        for i in range(na):
            for j in range(i):
                out[i * nb + j] = out[j * nb + i]
    return out^


def kernel_matrix(embed: Circuit, X1: List[List[Float64]], X2: List[List[Float64]], weights: List[Float64] = List[Float64]()) raises -> List[Float64]:
    """K[i, j] = |<psi(X2_j)|psi(X1_i)>|^2, row-major len(X1) x len(X2)
    (PennyLane qml.kernels.kernel_matrix(X1, X2, kernel))."""
    var A = embedding_states(embed, X1, weights)
    var B = embedding_states(embed, X2, weights)
    return _overlaps(A, B, False)


def kernel_matrix_symmetric(embed: Circuit, X: List[List[Float64]], weights: List[Float64] = List[Float64]()) raises -> List[Float64]:
    """Square Gram matrix of X: N(N-1)/2 overlaps, unit diagonal (PennyLane
    square_kernel_matrix with assume_normalized_kernel=True)."""
    var A = embedding_states(embed, X, weights)
    return _overlaps(A, A, True)


# ---------------------------------------------------------------------------
# Compute-uncompute route (differentiable, hardware-style)
# ---------------------------------------------------------------------------
def kernel_circuit(embed: Circuit, num_features: Int) raises -> Circuit:
    """U(w, a) followed by U(w, b)^dagger: <0|.|0> probability = k(a, b).

    The last `num_features` parameters of `embed` are the features; they are
    renamed xa[i] (first copy) and xb[i] (inverted copy) while the weights
    are shared by name (compose_mapped). Parameter vector: weights ++ xa ++ xb
    (`kernel_circuit_params`).
    """
    embed.validate()
    var np_ = embed.num_params()
    if num_features < 0 or num_features > np_:
        raise Error("QM-E831: kernel_circuit(): num_features " + String(num_features) + " outside 0.." + String(np_) + ".")
    var nw = np_ - num_features
    for k in range(nw):
        var nm = embed.param_names[k]
        if nm.startswith("xa[") or nm.startswith("xb["):
            raise Error("QM-E831: kernel_circuit(): weight name '" + nm + "' clashes with the feature names xa[i]/xb[i]. Hint: rename the weights.")
    var A = embed.copy()
    var B = embed.copy()
    for i in range(num_features):
        A.param_names[nw + i] = "xa[" + String(i) + "]"
        B.param_names[nw + i] = "xb[" + String(i) + "]"
    var kc = Circuit(embed.num_qubits, 0, "kernel_" + embed.name)
    kc.compose(A)
    kc.compose(B.inverse())
    return kc^


def kernel_circuit_params(weights: List[Float64], xa: List[Float64], xb: List[Float64]) -> List[Float64]:
    """weights ++ xa ++ xb: the parameter vector of a kernel_circuit."""
    var p = weights.copy()
    for v in xa:
        p.append(v)
    for v in xb:
        p.append(v)
    return p^


def _check_kc(kc: Circuit, weights: List[Float64], xa: List[Float64], xb: List[Float64]) raises:
    if len(xa) != len(xb) or len(weights) + len(xa) + len(xb) != kc.num_params():
        raise Error("QM-E831: kernel circuit has " + String(kc.num_params()) + " parameters, got " + String(len(weights)) + " weights + " + String(len(xa)) + " + " + String(len(xb)) + " features.")


def kernel_value_and_grad(kc: Circuit, weights: List[Float64], xa: List[Float64], xb: List[Float64]) raises -> Tuple[Float64, List[Float64]]:
    """(k(xa, xb), dk/dweights) by one adjoint pass on |0...0><0...0| (n <= 12).
    The gradient has len(weights) entries (features are not differentiated)."""
    _check_kc(kc, weights, xa, xb)
    var eng = ExpectationEngine(kc, projector_zero(kc.num_qubits))
    var g = List[Float64]()
    var k = eng.value_and_gradient(kernel_circuit_params(weights, xa, xb), g, Grad.ADJOINT)
    var gw = List[Float64](capacity=len(weights))
    for i in range(len(weights)):
        gw.append(g[i])
    return (k, gw^)


def kernel_matrix_symmetric_and_grad(kc: Circuit, weights: List[Float64], X: List[List[Float64]]) raises -> Tuple[List[Float64], List[List[Float64]]]:
    """(K, dK): the unit-diagonal Gram matrix of X from the kernel circuit and
    dK[p] = dK/dweights[p] (row-major N x N each, zero diagonal). The
    N(N-1)/2 adjoint passes run in parallel (batch_value_and_gradient)."""
    var n = len(X)
    var nw = len(weights)
    var sets = List[List[Float64]]()
    for i in range(n):
        for j in range(i + 1, n):
            _check_kc(kc, weights, X[i], X[j])
            sets.append(kernel_circuit_params(weights, X[i], X[j]))
    var K = List[Float64](length=n * n, fill=0.0)
    var dK = List[List[Float64]](capacity=nw)
    for _ in range(nw):
        dK.append(List[Float64](length=n * n, fill=0.0))
    for i in range(n):
        K[i * n + i] = 1.0
    if len(sets) == 0:
        return (K^, dK^)
    var r = batch_value_and_gradient(kc, projector_zero(kc.num_qubits), sets)
    var b = 0
    for i in range(n):
        for j in range(i + 1, n):
            K[i * n + j] = r[0][b]
            K[j * n + i] = r[0][b]
            for p in range(nw):
                var d = r[1][b][p]
                dK[p][i * n + j] = d
                dK[p][j * n + i] = d
            b += 1
    return (K^, dK^)


def sampled_kernel_entry(kc: Circuit, params: List[Float64], shots: Int, seed: UInt64) raises -> Float64:
    """Shot estimate of k = P(0...0) from `shots` runs of the kernel circuit
    (the QPU route; standard error sqrt(k (1-k) / shots))."""
    if shots < 1:
        raise Error("QM-E831: sampled_kernel_entry() needs shots >= 1.")
    var c = kc.copy()
    c.measure_all()
    var res = sample(c, shots, seed, params)
    var zero = bitstring(0, c.num_clbits)
    return Float64(res.counts.get(zero, 0)) / Float64(shots)


# ---------------------------------------------------------------------------
# Kernel-target alignment (PennyLane qml.kernels.target_alignment)
# ---------------------------------------------------------------------------
def _square_n(K: List[Float64], who: String) raises -> Int:
    var n = 0
    while n * n < len(K):
        n += 1
    if n * n != len(K) or n == 0:
        raise Error("QM-E831: " + who + "(): kernel matrix has " + String(len(K)) + " entries, not N x N.")
    return n


def _target(y: List[Float64], rescale: Bool) -> List[Float64]:
    var t = y.copy()
    if rescale:
        var nplus = 0
        for v in y:
            if v == 1.0:
                nplus += 1
        var nminus = len(y) - nplus
        for i in range(len(y)):
            t[i] = y[i] / Float64(nplus) if y[i] == 1.0 else y[i] / Float64(nminus)
    return t^


def target_alignment(K: List[Float64], y: List[Float64], assume_normalized_kernel: Bool = False,
                     rescale_class_labels: Bool = True) raises -> Float64:
    """<K, T>_F / sqrt(<K, K>_F <T, T>_F) with T = y y^T; labels +/-1 are
    rescaled by the class sizes when `rescale_class_labels`;
    `assume_normalized_kernel` treats the diagonal of K as 1."""
    var n = _square_n(K, "target_alignment")
    if len(y) != n:
        raise Error("QM-E831: target_alignment(): " + String(len(y)) + " labels for a " + String(n) + " x " + String(n) + " kernel.")
    var t = _target(y, rescale_class_labels)
    var a = 0.0
    var b = 0.0
    var c = 0.0
    for i in range(n):
        for j in range(n):
            var k = 1.0 if (assume_normalized_kernel and i == j) else K[i * n + j]
            var tij = t[i] * t[j]
            a += k * tij
            b += k * k
            c += tij * tij
    return a / sqrt(b * c)


def target_alignment_grad(K: List[Float64], dK: List[List[Float64]], y: List[Float64],
                          assume_normalized_kernel: Bool = False, rescale_class_labels: Bool = True) raises -> List[Float64]:
    """d target_alignment / d theta_p for every derivative matrix dK[p] = dK/dtheta_p:
    <dK, T>/sqrt(B C) - A <K, dK> / (sqrt(C) B^(3/2))."""
    var n = _square_n(K, "target_alignment_grad")
    if len(y) != n:
        raise Error("QM-E831: target_alignment_grad(): " + String(len(y)) + " labels for a " + String(n) + " x " + String(n) + " kernel.")
    var t = _target(y, rescale_class_labels)
    var a = 0.0
    var b = 0.0
    var c = 0.0
    for i in range(n):
        for j in range(n):
            var k = 1.0 if (assume_normalized_kernel and i == j) else K[i * n + j]
            var tij = t[i] * t[j]
            a += k * tij
            b += k * k
            c += tij * tij
    var out = List[Float64](capacity=len(dK))
    for p in range(len(dK)):
        if len(dK[p]) != n * n:
            raise Error("QM-E831: target_alignment_grad(): dK[" + String(p) + "] has " + String(len(dK[p])) + " entries, expected " + String(n * n) + ".")
        var da = 0.0
        var db = 0.0
        for i in range(n):
            for j in range(n):
                if assume_normalized_kernel and i == j:
                    continue
                var d = dK[p][i * n + j]
                da += d * t[i] * t[j]
                db += 2.0 * K[i * n + j] * d
        out.append(da / sqrt(b * c) - 0.5 * a * db / (sqrt(c) * b * sqrt(b)))
    return out^


# ---------------------------------------------------------------------------
# Support vector classifier (C-SVC, precomputed kernel)
# ---------------------------------------------------------------------------
struct SVCModel(Copyable, Writable):
    """Dual solution of a C-SVC: decision(x) = sum_i alpha_i y_i K(x_i, x) - rho.

    `dual_coef[i]` = alpha_i y_i (sklearn `dual_coef_` scattered over all
    training points, zeros for non-support vectors), `intercept` = -rho.
    """

    var alpha: List[Float64]
    var y: List[Float64]
    var dual_coef: List[Float64]
    var rho: Float64
    var intercept: Float64
    var C: Float64
    var iterations: Int
    var converged: Bool

    def __init__(out self, var alpha: List[Float64], var y: List[Float64], rho: Float64, C: Float64, iterations: Int, converged: Bool):
        self.dual_coef = List[Float64](capacity=len(alpha))
        for i in range(len(alpha)):
            self.dual_coef.append(alpha[i] * y[i])
        self.alpha = alpha^
        self.y = y^
        self.rho = rho
        self.intercept = -rho
        self.C = C
        self.iterations = iterations
        self.converged = converged

    def num_support(self) -> Int:
        var k = 0
        for a in self.alpha:
            if a > 0.0:
                k += 1
        return k

    def support(self) -> List[Int]:
        var s = List[Int]()
        for i in range(len(self.alpha)):
            if self.alpha[i] > 0.0:
                s.append(i)
        return s^

    def write_to(self, mut w: Some[Writer]):
        w.write("SVCModel(n=", len(self.alpha), ", support=", self.num_support(), ", intercept=", self.intercept, ", iterations=", self.iterations, ")")


def svc_fit(K: List[Float64], y: List[Float64], C: Float64 = 1.0, tol: Float64 = 1e-3, max_iter: Int = 100000) raises -> SVCModel:
    """C-SVC on a precomputed N x N kernel, labels +/-1.

    SMO as libsvm's Solver (Chang & Lin; sklearn SVC): maximal-violating i,
    second-order choice of j (Fan, Chen, Lin 2005), analytic two-variable
    update clipped to the box [0, C], gradient update, stop when the KKT gap
    m(alpha) - M(alpha) < tol; rho from the free support vectors (or the
    midpoint of the bounds). No shrinking (same optimum).
    """
    var n = _square_n(K, "svc_fit")
    if len(y) != n:
        raise Error("QM-E831: svc_fit(): " + String(len(y)) + " labels for a " + String(n) + " x " + String(n) + " kernel.")
    for v in y:
        if v != 1.0 and v != -1.0:
            raise Error("QM-E831: svc_fit() needs labels +1/-1, got " + String(v) + ".")
    if not (C > 0.0):
        raise Error("QM-E831: svc_fit() needs C > 0.")
    comptime TAU = 1e-12
    var alpha = List[Float64](length=n, fill=0.0)
    var G = List[Float64](length=n, fill=-1.0)
    var it = 0
    var converged = False
    while it < max_iter:
        # select i: maximal violation
        var gmax = -1e300
        var i = -1
        for t in range(n):
            if y[t] == 1.0:
                if alpha[t] < C and -G[t] >= gmax:
                    gmax = -G[t]
                    i = t
            else:
                if alpha[t] > 0.0 and G[t] >= gmax:
                    gmax = G[t]
                    i = t
        var gmax2 = -1e300
        var j = -1
        var obj_min = 1e300
        for t in range(n):
            var kit = K[i * n + t] if i >= 0 else 0.0
            var quad = (K[i * n + i] + K[t * n + t] - 2.0 * kit) if i >= 0 else 0.0
            if y[t] == 1.0:
                if alpha[t] > 0.0:
                    var gd = gmax + G[t]
                    if G[t] >= gmax2:
                        gmax2 = G[t]
                    if gd > 0.0 and i >= 0:
                        var od = -(gd * gd) / (quad if quad > 0.0 else TAU)
                        if od <= obj_min:
                            j = t
                            obj_min = od
            else:
                if alpha[t] < C:
                    var gd = gmax - G[t]
                    if -G[t] >= gmax2:
                        gmax2 = -G[t]
                    if gd > 0.0 and i >= 0:
                        var od = -(gd * gd) / (quad if quad > 0.0 else TAU)
                        if od <= obj_min:
                            j = t
                            obj_min = od
        if gmax + gmax2 < tol or i < 0 or j < 0:
            converged = True
            break
        it += 1
        var yi = y[i]
        var yj = y[j]
        var qij = yi * yj * K[i * n + j]
        var oai = alpha[i]
        var oaj = alpha[j]
        if yi != yj:
            var quad = K[i * n + i] + K[j * n + j] + 2.0 * qij
            if quad <= 0.0:
                quad = TAU
            var delta = (-G[i] - G[j]) / quad
            var diff = alpha[i] - alpha[j]
            alpha[i] += delta
            alpha[j] += delta
            if diff > 0.0:
                if alpha[j] < 0.0:
                    alpha[j] = 0.0
                    alpha[i] = diff
            else:
                if alpha[i] < 0.0:
                    alpha[i] = 0.0
                    alpha[j] = -diff
            if diff > 0.0:  # C_i - C_j = 0
                if alpha[i] > C:
                    alpha[i] = C
                    alpha[j] = C - diff
            else:
                if alpha[j] > C:
                    alpha[j] = C
                    alpha[i] = C + diff
        else:
            var quad = K[i * n + i] + K[j * n + j] - 2.0 * qij
            if quad <= 0.0:
                quad = TAU
            var delta = (G[i] - G[j]) / quad
            var s = alpha[i] + alpha[j]
            alpha[i] -= delta
            alpha[j] += delta
            if s > C:
                if alpha[i] > C:
                    alpha[i] = C
                    alpha[j] = s - C
            else:
                if alpha[j] < 0.0:
                    alpha[j] = 0.0
                    alpha[i] = s
            if s > C:
                if alpha[j] > C:
                    alpha[j] = C
                    alpha[i] = s - C
            else:
                if alpha[i] < 0.0:
                    alpha[i] = 0.0
                    alpha[j] = s
        var dai = alpha[i] - oai
        var daj = alpha[j] - oaj
        for k in range(n):
            G[k] += y[k] * (yi * K[i * n + k] * dai + yj * K[j * n + k] * daj)
    # rho
    var nfree = 0
    var sfree = 0.0
    var ub = 1e300
    var lb = -1e300
    for t in range(n):
        var yg = y[t] * G[t]
        if alpha[t] >= C:
            if y[t] == -1.0:
                ub = min(ub, yg)
            else:
                lb = max(lb, yg)
        elif alpha[t] <= 0.0:
            if y[t] == 1.0:
                ub = min(ub, yg)
            else:
                lb = max(lb, yg)
        else:
            nfree += 1
            sfree += yg
    var rho = sfree / Float64(nfree) if nfree > 0 else 0.5 * (ub + lb)
    return SVCModel(alpha^, y.copy(), rho, C, it, converged)


def svc_decision(model: SVCModel, K_rows: List[Float64]) raises -> List[Float64]:
    """Decision values for M points from their kernel rows against the N
    training points (row-major M x N, e.g. kernel_matrix(embed, X_test, X_train))."""
    var n = len(model.alpha)
    if n == 0 or len(K_rows) % n != 0:
        raise Error("QM-E831: svc_decision(): " + String(len(K_rows)) + " kernel entries is not a multiple of the " + String(n) + " training points.")
    var m = len(K_rows) // n
    var out = List[Float64](capacity=m)
    for r in range(m):
        var s = 0.0
        for i in range(n):
            s += model.dual_coef[i] * K_rows[r * n + i]
        out.append(s - model.rho)
    return out^


def svc_predict(model: SVCModel, K_rows: List[Float64]) raises -> List[Float64]:
    """Labels +1 (decision > 0) / -1 for M points (kernel rows as in svc_decision)."""
    var d = svc_decision(model, K_rows)
    for ref v in d:
        v = 1.0 if v > 0.0 else -1.0
    return d^


# ---------------------------------------------------------------------------
# Kernel ridge regression
# ---------------------------------------------------------------------------
def kernel_ridge_fit(K: List[Float64], y: List[Float64], lam: Float64) raises -> List[Float64]:
    """alpha = (K + lam I)^-1 y via the symmetric eigendecomposition
    (jacobi_eigh); predictions are K_rows @ alpha (kernel_ridge_predict).
    Eigenvalues of K + lam I below 1e-12 * max are dropped (pseudo-inverse)."""
    var n = _square_n(K, "kernel_ridge_fit")
    if len(y) != n:
        raise Error("QM-E831: kernel_ridge_fit(): " + String(len(y)) + " targets for a " + String(n) + " x " + String(n) + " kernel.")
    var A = K.copy()
    for i in range(n):
        A[i * n + i] += lam
    var e = jacobi_eigh(A, n)
    var emax = 0.0
    for v in e.values:
        emax = max(emax, abs(v))
    var alpha = List[Float64](length=n, fill=0.0)
    for k in range(n):
        var lk = e.values[k]
        if abs(lk) <= 1e-12 * emax:
            continue
        var proj = 0.0
        for i in range(n):
            proj += e.vectors[i * n + k] * y[i]
        proj /= lk
        for i in range(n):
            alpha[i] += proj * e.vectors[i * n + k]
    return alpha^


def kernel_ridge_predict(alpha: List[Float64], K_rows: List[Float64]) raises -> List[Float64]:
    """K_rows (M x N row-major) @ alpha."""
    var n = len(alpha)
    if n == 0 or len(K_rows) % n != 0:
        raise Error("QM-E831: kernel_ridge_predict(): " + String(len(K_rows)) + " kernel entries is not a multiple of " + String(n) + ".")
    var m = len(K_rows) // n
    var out = List[Float64](capacity=m)
    for r in range(m):
        var s = 0.0
        for i in range(n):
            s += K_rows[r * n + i] * alpha[i]
        out.append(s)
    return out^
