"""Jordan's gradient estimation: the notebook's linear, quadratic and 2-D examples on QUMO.

Reproduces: Classiq library - Fast Quantum Algorithm for Numerical Gradient Estimation
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/quantum_primitives/gradient_estimation/gradient_estimation.ipynb (Classiq Library, MIT)

The notebook estimates the gradient at the origin with one query: Hadamards on a signed n-bit register of N = 2^n grid points, the phase 2 pi f_norm(delta) with f_norm(delta) = N f(l delta / N) / (l m), then an inverse QFT whose outcome k gives the gradient m k / N. The file runs the notebook's defaults (n = 3, N = 8, l = 0.5, m = 2) on its examples: the phase table of f = 0.5 x + 0.25, the linear slopes -0.5 and 0.55, the quadratic 0.6 x^2 + 0.25 x + 0.1 at l = 0.1 and l = 1, the 2-D quadratic, and the m sweep of the performance section. Checks: the phase state equals the notebook's printed phase table, every output law equals a brute-force Fourier sum of the notebook's own f_norm, the majority gradients and the notebook's asserts hold, and the success rates match the notebook's printed sampling frequencies (2048 shots).

Deviations from the notebook:
- The phase oracle phase(f_norm(x), 2 pi) is compiled by hand instead of by Classiq synthesis. f_norm is a polynomial of degree 2 in the signed index delta, so its phase is exactly a product of P gates on single bits and CP gates on bit pairs, with angles from the polynomial coefficients. Bit p of a signed n-bit register has weight 2^p, and -2^(n-1) for the sign bit.
- Probabilities are exact state-vector values, not Classiq sampling. A printed notebook frequency is compared with the exact value within 4 binomial standard errors at 2048 shots; a printed 0.00% is compared with the rule-of-three bound 3/2048.
- The notebook's majority outcome at l = 1 is not checked: the exact probabilities of k = 0 and k = 2 are slightly above that of k = 1, so the notebook's "correct" there depends on sampling.
- The helper module's params defaults (l = 0.5, m = 2, n = 3) and success rule (|estimate - gradient| < 0.5 m / N, or the notebook's epsilon in the sweep) are copied here.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/gradient_estimation.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import atan2, cos, floor, pi, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve
from qumo.algorithms.library import iqft
from qumo.testing.checks import CheckLog


def signed_value(j: Int, n: Int) -> Int:
    """Value of the n-bit two's-complement register that holds the unsigned index j."""
    var N = 1 << n
    if j >= N // 2:
        return j - N
    return j


def weight(p: Int, n: Int) -> Float64:
    """Two's-complement weight of bit p: 2^p, and -2^(n-1) for the sign bit."""
    var w = Float64(1 << p)
    if p == n - 1:
        return -w
    return w


def phase_circuit(
    n: Int,
    dims: Int,
    q: List[Float64],
    lin: List[Float64],
    cross: Float64,
    l: Float64,
    m: Float64,
    with_iqft: Bool,
) raises -> Circuit:
    """Hadamards on the coordinate registers, the phase 2 pi f_norm(delta), then the inverse QFT on each register.

    Register i is qubits i*n .. i*n+n-1 (little-endian). The function is
    f(x) = sum_i q[i] x_i^2 + sum_i lin[i] x_i + cross x_0 x_1 (its constant only sets a global phase).
    With delta_i = sum_p weight(p) b_{i,p}, the normalized phase is
    f_norm = sum_i (q[i] l / (m N)) delta_i^2 + sum_i (lin[i] / m) delta_i + (cross l / (m N)) delta_0 delta_1,
    and delta^2 = sum_p w_p^2 b_p + 2 sum_{p<r} w_p w_r b_p b_r for bits b_p in {0, 1}.
    """
    var N = Float64(1 << n)
    var c = Circuit(dims * n, name="gradient_estimation")
    for k in range(dims * n):
        c.h(k)
    for i in range(dims):
        var qa = q[i] * l / (m * N)
        var la = lin[i] / m
        for p in range(n):
            var wp = weight(p, n)
            c.p(2.0 * pi * (qa * wp * wp + la * wp), i * n + p)
        for p in range(n):
            for r in range(p + 1, n):
                c.cp(2.0 * pi * 2.0 * qa * weight(p, n) * weight(r, n), i * n + p, i * n + r)
    if dims == 2:
        var cc = cross * l / (m * N)
        for p in range(n):
            for r in range(n):
                c.cp(2.0 * pi * cc * weight(p, n) * weight(r, n), p, n + r)
    if with_iqft:
        for i in range(dims):
            c.compose(iqft(n), i * n)
    return c^


# ---------------------------------------------------------------------------
# Brute-force references: the notebook's f_norm evaluated directly on the grid,
# then the discrete Fourier sum (independent of the gates above).
# ---------------------------------------------------------------------------
def reference_1d(a: Float64, b: Float64, c: Float64, l: Float64, m: Float64, n: Int) -> List[Float64]:
    """P(k) = |N^-1 sum_delta exp(2 pi i (f_norm(delta) - k delta / N))|^2 for f = a x^2 + b x + c."""
    var N = 1 << n
    var Nf = Float64(N)
    var ph = List[Float64](length=N, fill=0.0)
    for j in range(N):
        var x = l / Nf * Float64(signed_value(j, n))
        ph[j] = (a * x * x + b * x + c) * Nf / (l * m)
    var out = List[Float64](length=N, fill=0.0)
    for k in range(N):
        var re = 0.0
        var im = 0.0
        for j in range(N):
            var ang = 2.0 * pi * (ph[j] - Float64(k * signed_value(j, n)) / Nf)
            re += cos(ang)
            im += sin(ang)
        out[k] = (re * re + im * im) / (Nf * Nf)
    return out^


def reference_2d(
    a: Float64, b: Float64, c: Float64, d: Float64, e: Float64, f0: Float64,
    l: Float64, m: Float64, n: Int,
) -> List[Float64]:
    """Same as reference_1d for f = a x^2 + b y^2 + c x y + d x + e y + f0; index k = kx + N ky."""
    var N = 1 << n
    var Nf = Float64(N)
    var ph = List[Float64](length=N * N, fill=0.0)
    for jy in range(N):
        for jx in range(N):
            var x = l / Nf * Float64(signed_value(jx, n))
            var y = l / Nf * Float64(signed_value(jy, n))
            ph[jx + N * jy] = (a * x * x + b * y * y + c * x * y + d * x + e * y + f0) * Nf / (l * m)
    var out = List[Float64](length=N * N, fill=0.0)
    for ky in range(N):
        for kx in range(N):
            var re = 0.0
            var im = 0.0
            for jy in range(N):
                for jx in range(N):
                    var kd = kx * signed_value(jx, n) + ky * signed_value(jy, n)
                    var ang = 2.0 * pi * (ph[jx + N * jy] - Float64(kd) / Nf)
                    re += cos(ang)
                    im += sin(ang)
            out[kx + N * ky] = (re * re + im * im) / (Nf * Nf * Nf * Nf)
    return out^


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def circular_gap(x: Float64, y: Float64) -> Float64:
    """Distance between two phases given in cycles (mod 1)."""
    var d = x - y
    d = d - floor(d)
    return min(d, 1.0 - d)


def max_abs_diff(a: List[Float64], b: List[Float64]) -> Float64:
    var worst = 0.0
    for i in range(len(a)):
        worst = max(worst, abs(a[i] - b[i]))
    return worst


def argmax(p: List[Float64]) -> Int:
    var best = 0
    for i in range(len(p)):
        if p[i] > p[best]:
            best = i
    return best


def gradient_of(k: Int, m: Float64, n: Int) -> Float64:
    """Gradient estimate of the unsigned outcome k: m/N times its signed value."""
    return m / Float64(1 << n) * Float64(signed_value(k, n))


def tolerance(m: Float64, n: Int) -> Float64:
    """The notebook's default success tolerance, 0.5 m / N (resolution / 2)."""
    return 0.5 * m / Float64(1 << n)


def success_1d(probs: List[Float64], n: Int, m: Float64, g: Float64, tol: Float64) -> Float64:
    """Probability that the estimate lies strictly within tol of the analytic gradient g."""
    var total = 0.0
    for k in range(1 << n):
        if abs(gradient_of(k, m, n) - g) < tol:
            total += probs[k]
    return total


def success_2d(probs: List[Float64], n: Int, m: Float64, gx: Float64, gy: Float64, tol: Float64) -> Float64:
    var N = 1 << n
    var total = 0.0
    for ky in range(N):
        for kx in range(N):
            if abs(gradient_of(kx, m, n) - gx) < tol and abs(gradient_of(ky, m, n) - gy) < tol:
                total += probs[kx + N * ky]
    return total


def match_notebook(mut log: CheckLog, label: String, value: Float64, printed: Float64) -> Bool:
    """Compare an exact probability with a frequency the notebook printed from 2048 shots.

    PASS iff the exact value lies within 4 binomial standard errors of the printed frequency.
    A printed 0.00% means no hit in 2048 shots: PASS iff the exact value is below the
    rule-of-three bound 3/2048.
    """
    if printed == 0.0:
        return log.within(label, value, 0.0, 3.0 / 2048.0)
    var band = 4.0 * sqrt(printed * (1.0 - printed) / 2048.0)
    return log.within(label, value, printed - band, printed + band)


def main() raises:
    var log = CheckLog("classiq/gradient_estimation")
    var n = 3
    var N = 1 << n
    var q_zero: List[Float64] = [0.0]

    # ---- phase state of f = 0.5 x + 0.25 (notebook cells 13-15): phase/2pi is 0.25 delta ----
    var lin_phase: List[Float64] = [0.5]
    var ps = evolve(phase_circuit(n, 1, q_zero, lin_phase, 0.0, 0.5, 2.0, False))
    var phase0 = atan2(ps.im[0], ps.re[0])
    var table: List[Float64] = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    var worst_phase = 0.0
    for j in range(N):
        var rel = (atan2(ps.im[j], ps.re[j]) - phase0) / (2.0 * pi)
        worst_phase = max(worst_phase, circular_gap(rel, table[signed_value(j, n) + N // 2]))
    _ = log.close("phase state equals the notebook phase table", worst_phase, 0.0, 1e-9)

    # ---- linear f = -0.5 x + 0.25 (cell 33): the inverse QFT outcome is k = -2 ----
    var lin_neg: List[Float64] = [-0.5]
    var probs_a = evolve(phase_circuit(n, 1, q_zero, lin_neg, 0.0, 0.5, 2.0, True)).probabilities()
    var ref_a = reference_1d(0.0, -0.5, 0.25, 0.5, 2.0, n)
    _ = log.close("linear -0.5 law equals the brute-force Fourier sum", max_abs_diff(probs_a, ref_a), 0.0, 1e-9)
    var g_a = gradient_of(argmax(probs_a), 2.0, n)
    _ = log.close("linear -0.5 majority gradient", g_a, -0.5, 1e-12)
    var succ_a = success_1d(probs_a, n, 2.0, -0.5, tolerance(2.0, n))
    _ = log.close("linear -0.5 success rate (notebook 100%)", succ_a, 1.0, 1e-9)
    print("linear -0.5: majority gradient", g_a, "success", succ_a)

    # ---- linear f = 0.55 x + 0.25 (cell 35): 0.55 is not a multiple of m/N = 0.25 ----
    var lin_055: List[Float64] = [0.55]
    var probs_b = evolve(phase_circuit(n, 1, q_zero, lin_055, 0.0, 0.5, 2.0, True)).probabilities()
    var ref_b = reference_1d(0.0, 0.55, 0.25, 0.5, 2.0, n)
    _ = log.close("0.55 law equals the brute-force Fourier sum", max_abs_diff(probs_b, ref_b), 0.0, 1e-9)
    var g_b = gradient_of(argmax(probs_b), 2.0, n)
    _ = log.check(
        "0.55 majority gradient is within the resolution",
        abs(g_b - 0.55) < tolerance(2.0, n),
        "gradient=" + String(g_b),
    )
    var succ_b = success_1d(probs_b, n, 2.0, 0.55, tolerance(2.0, n))
    _ = match_notebook(log, "0.55 success matches the notebook sampling", succ_b, 0.880859375)
    print("0.55: majority gradient", g_b, "success", succ_b)

    # ---- quadratic f = 0.6 x^2 + 0.25 x + 0.1 at l = 0.1 (cell 39): f'(0) = 0.25 ----
    var q_quad: List[Float64] = [0.6]
    var lin_quad: List[Float64] = [0.25]
    var probs_c = evolve(phase_circuit(n, 1, q_quad, lin_quad, 0.0, 0.1, 2.0, True)).probabilities()
    var ref_c = reference_1d(0.6, 0.25, 0.1, 0.1, 2.0, n)
    _ = log.close("quadratic l=0.1 law equals the brute-force Fourier sum", max_abs_diff(probs_c, ref_c), 0.0, 1e-9)
    var g_c = gradient_of(argmax(probs_c), 2.0, n)
    _ = log.close("quadratic l=0.1 majority gradient", g_c, 0.25, 1e-12)
    var succ_c = success_1d(probs_c, n, 2.0, 0.25, tolerance(2.0, n))
    _ = match_notebook(log, "quadratic l=0.1 success matches the notebook sampling", succ_c, 0.98681640625)
    _ = log.check("quadratic l=0.1 success above 0.9 (notebook assert)", succ_c > 0.9, "success=" + String(succ_c))
    print("quadratic l=0.1: majority gradient", g_c, "success", succ_c)

    # ---- the same quadratic at l = 1 (cell 41): outside the linear regime ----
    var probs_d = evolve(phase_circuit(n, 1, q_quad, lin_quad, 0.0, 1.0, 2.0, True)).probabilities()
    var ref_d = reference_1d(0.6, 0.25, 0.1, 1.0, 2.0, n)
    _ = log.close("quadratic l=1 law equals the brute-force Fourier sum", max_abs_diff(probs_d, ref_d), 0.0, 1e-9)
    var succ_d = success_1d(probs_d, n, 2.0, 0.25, tolerance(2.0, n))
    _ = match_notebook(log, "quadratic l=1 success matches the notebook sampling", succ_d, 0.248046875)
    _ = log.check("quadratic l=1 success below 0.5 (notebook assert)", succ_d < 0.5, "success=" + String(succ_d))
    print("quadratic l=1: success", succ_d)

    # ---- 2-D quadratic (cell 49): gradient (0.25, -0.5) ----
    var q2: List[Float64] = [0.6, -0.4]
    var lin2: List[Float64] = [0.25, -0.5]
    var probs_e = evolve(phase_circuit(n, 2, q2, lin2, 0.25, 0.1, 2.0, True)).probabilities()
    var ref_e = reference_2d(0.6, -0.4, 0.25, 0.25, -0.5, 0.1, 0.1, 2.0, n)
    _ = log.close("2-D law equals the brute-force Fourier sum", max_abs_diff(probs_e, ref_e), 0.0, 1e-9)
    var ke = argmax(probs_e)
    var gx = gradient_of(ke % N, 2.0, n)
    var gy = gradient_of(ke // N, 2.0, n)
    _ = log.check(
        "2-D majority gradient is (0.25, -0.5)",
        abs(gx - 0.25) < 1e-12 and abs(gy + 0.5) < 1e-12,
        "gradient=(" + String(gx) + ", " + String(gy) + ")",
    )
    var succ_e = success_2d(probs_e, n, 2.0, 0.25, -0.5, tolerance(2.0, n))
    _ = match_notebook(log, "2-D success matches the notebook sampling", succ_e, 0.974609375)
    _ = log.check("2-D success above 0.9 (notebook assert)", succ_e > 0.9, "success=" + String(succ_e))
    print("2-D: majority gradient", gx, gy, "success", succ_e)

    # ---- m sweep (cell 44): quadratic at l = 0.1, success within epsilon = 0.2 ----
    var sweep_m: List[Float64] = [0.5, 1.0, 2.0, 3.0, 3.67, 10.0]
    var sweep_nb: List[Float64] = [0.1094, 0.9941, 0.9878, 0.6704, 0.0, 0.0]
    for i in range(len(sweep_m)):
        var mm = sweep_m[i]
        var pm = evolve(phase_circuit(n, 1, q_quad, lin_quad, 0.0, 0.1, mm, True)).probabilities()
        var sm = success_1d(pm, n, mm, 0.25, 0.2)
        _ = match_notebook(log, "m sweep m=" + String(mm) + " success matches the notebook", sm, sweep_nb[i])

    log.finish()
