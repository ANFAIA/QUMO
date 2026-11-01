"""Multi-product formulas (Carrera Vazquez et al., Quantum 7, 1067; qiskit-addon-mpf).

A weighted sum of product formulas with different step counts cancels their
leading errors: mu(t) = sum_j x_j rho_kj(t), which classical post-processing can do
with expectation values, at no extra circuit depth beyond the largest k.

`mpf_coefficients` solves the linear system of the addon's `setup_static_lse`:

    sum_j x_j = 1,    sum_j x_j / k_j^(order + i * step) = 0,  i = 0, 1, ...

with step 2 for symmetric formulas, whose error expansion has even powers only.
The exact solution can have a large L1 norm, which amplifies shot noise, so
`method="approx"` (Zhuk et al., PRR 6, 033309) minimises the residual of the same
system with sum x = 1 and ||x||_1 <= max_l1, by trying each sign pattern of x.
"""

from qumo.ir.pauli import PauliSum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve_into
from qumo.algorithms.library import trotter


def _solve(var a: List[List[Float64]], var b: List[Float64]) raises -> List[Float64]:
    """Gaussian elimination with partial pivoting."""
    var n = len(b)
    for col in range(n):
        var piv = col
        for r in range(col + 1, n):
            if abs(a[r][col]) > abs(a[piv][col]):
                piv = r
        if abs(a[piv][col]) < 1e-14:
            raise Error("QM-E817: the multi-product system is singular. Hint: use distinct step counts.")
        if piv != col:
            a.swap_elements(piv, col)
            b.swap_elements(piv, col)
        for r in range(col + 1, n):
            var f = a[r][col] / a[col][col]
            if f != 0.0:
                for c in range(col, n):
                    a[r][c] -= f * a[col][c]
                b[r] -= f * b[col]
    var x = List[Float64](length=n, fill=0.0)
    var i = n - 1
    while i >= 0:
        var s = b[i]
        for c in range(i + 1, n):
            s -= a[i][c] * x[c]
        x[i] = s / a[i][i]
        i -= 1
    return x^


def _system(steps: List[Int], order: Int, symmetric: Bool) -> List[List[Float64]]:
    var n = len(steps)
    var inc = 2 if symmetric else 1
    var a = List[List[Float64]]()
    a.append(List[Float64](length=n, fill=1.0))
    for i in range(n - 1):
        var row = List[Float64](capacity=n)
        for k in steps:
            row.append(pow(Float64(k), -Float64(order + i * inc)))
        a.append(row^)
    return a^


def _residual(a: List[List[Float64]], x: List[Float64]) -> Float64:
    var s = 0.0
    for i in range(len(a)):
        var r = -1.0 if i == 0 else 0.0
        for j in range(len(x)):
            r += a[i][j] * x[j]
        s += r * r
    return s


def _approx(a: List[List[Float64]], max_l1: Float64) raises -> List[Float64]:
    """min ||A x - b||^2 with sum x = 1, ||x||_1 <= max_l1: a quadratic over a polytope,
    solved on each face x_j s_j >= 0 by the equality-constrained KKT system."""
    var n = len(a[0])
    if n > 12:
        raise Error("QM-E817: method 'approx' takes at most 12 formulas, got " + String(n) + ". Hint: use 'exact'.")
    var best = List[Float64]()
    var best_cost = 1e300
    # A^T A and A^T b
    var ata = List[List[Float64]]()
    for i in range(n):
        var row = List[Float64](length=n, fill=0.0)
        for j in range(n):
            for k in range(len(a)):
                row[j] += a[k][i] * a[k][j]
        ata.append(row^)
    for mask in range(1 << n):
        for use_l1 in range(2):
            var m = n + 1 + use_l1
            var kkt = List[List[Float64]]()
            for _ in range(m):
                kkt.append(List[Float64](length=m, fill=0.0))
            var rhs = List[Float64](length=m, fill=0.0)
            for i in range(n):
                var s = 1.0 if (mask >> i) & 1 == 1 else -1.0
                for j in range(n):
                    kkt[i][j] = ata[i][j]
                kkt[i][n] = 1.0
                kkt[n][i] = 1.0
                if use_l1 == 1:
                    kkt[i][n + 1] = s
                    kkt[n + 1][i] = s
                rhs[i] = a[0][i]  # b = e_0
            rhs[n] = 1.0
            if use_l1 == 1:
                rhs[n + 1] = max_l1
            var sol: List[Float64]
            try:
                sol = _solve(kkt^, rhs^)
            except:
                continue
            var x = List[Float64](capacity=n)
            var l1 = 0.0
            var feasible = True
            for i in range(n):
                var s = 1.0 if (mask >> i) & 1 == 1 else -1.0
                x.append(sol[i])
                l1 += abs(sol[i])
                if sol[i] * s < -1e-10:
                    feasible = False
            if not feasible or l1 > max_l1 + 1e-9:
                continue
            var cost = _residual(a, x)
            if cost < best_cost:
                best_cost = cost
                best = x^
    if len(best) == 0:
        raise Error("QM-E817: no coefficients with sum 1 and L1 norm <= " + String(max_l1) + ". Hint: max_l1 must be at least 1.")
    return best^


def mpf_coefficients(steps: List[Int], order: Int = 2, symmetric: Bool = True, method: String = "exact",
                     max_l1: Float64 = 3.0) raises -> List[Float64]:
    """Coefficients x_j for the product formulas with `steps[j]` Trotter steps.

    `order`: order of the product formula; `symmetric`: only even powers appear in
    its error (Strang and Suzuki). `method`: "exact" or "approx" (bounded L1 norm)."""
    if len(steps) == 0:
        raise Error("QM-E817: mpf_coefficients needs at least one step count. Hint: e.g. [1, 2, 3].")
    for i in range(len(steps)):
        if steps[i] < 1:
            raise Error("QM-E817: step counts must be positive, got " + String(steps[i]) + ". Hint: e.g. [1, 2, 3].")
        for j in range(i + 1, len(steps)):
            if steps[i] == steps[j]:
                raise Error("QM-E817: step count " + String(steps[i]) + " appears twice. Hint: use distinct step counts.")
    var a = _system(steps, order, symmetric)
    if method == "exact":
        var b = List[Float64](length=len(steps), fill=0.0)
        b[0] = 1.0
        return _solve(a^, b^)
    if method == "approx":
        return _approx(a, max_l1)
    raise Error("QM-E817: unknown method '" + method + "'. Hint: use 'exact' or 'approx'.")


def mpf_expectation(h: PauliSum, observable: PauliSum, time: Float64, steps: List[Int], coefficients: List[Float64],
                    prep: StateVector, order: Int = 2) raises -> Float64:
    """sum_j x_j <prep| U_kj^dag observable U_kj |prep>: the post-processed estimate."""
    var total = 0.0
    var n = prep.num_qubits
    for j in range(len(steps)):
        var sv = prep.copy()
        evolve_into(sv, trotter(h, time, steps[j], order, n), List[Float64]())
        total += coefficients[j] * sv.expectation(observable)
    return total
