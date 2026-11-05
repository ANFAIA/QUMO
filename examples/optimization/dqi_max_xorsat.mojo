"""Decoded quantum interferometry (DQI) for max-XORSAT: the notebook's Max-Cut instance on QUMO.

Reproduces: Classiq library - Decoded Quantum Interferometry Algorithm
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/search_and_optimization/dqi/dqi_max_xorsat.ipynb (Classiq Library, MIT)
Framework: Classiq - https://github.com/Classiq/classiq-library

The notebook runs DQI on Max-Cut for a 6-cycle: m = 6 constraints (edges) on n = 6 variables (nodes), MAX_ERRORS = 2, the optimal w_k of the tridiagonal matrix of the paper, and a lookup-table decoder for the 22 errors of weight at most 2. QUMO builds the same 12-qubit circuit (y on qubits 0..5, the solution on qubits 6..11): the Dicke-weighted state, the phase (-1)^(v.y), the map y -> B^T y, the lookup decoder that returns y to |0>, and the Hadamard transform. The checks compare the optimal w_k and the 22 error syndromes with the notebook's printed values, the state before decoding and the output amplitudes with the closed form of the algorithm, the cleanup of y, the expected objective (exactly 4 in closed form; the notebook prints 4.0012 from 10000 shots), the probability 75/256 of each of the two optimal cuts, and a 10000-shot sample (fixed seed) with the notebook's printed counts (2925 and 2916 of 10000) within four standard errors.

Deviations from the notebook:
- Stages 1 to 3 (prepare_amplitudes on the k register, binary_to_unary, prepare_dicke_state_unary_input) are one real amplitude embedding (Mottonen tree) of the same state sum_y w_|y| / sqrt(C(6, |y|)) |y> on the y register. The Dicke circuit is not ported. The prepared state is checked against the closed form before decoding.
- vector_product_phase with v = all ones is a Z gate on each y qubit, which is exactly (-1)^|y|. matrix_vector_product is the CX network of the printed B matrix. hadamard_transform is H on each solution qubit.
- syndrome_decode_lookuptable: each control(syndrome == s, inplace_xor(e, error)) is one 6-control X per bit of e (append_mcx, no ancilla), with X conjugation on the zero bits of s.
- The w_k come from power iteration on the notebook's tridiagonal A, not numpy eigh. B is the printed incidence matrix of the notebook's graph, hard-coded.
- The notebook samples on Classiq. Here QUMO's sampler runs 10000 shots with seed 1234, and the notebook's own printed numbers are compared within statistical bands.

Run: .venv/bin/mojo run -I . examples/optimization/dqi_max_xorsat.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.bit import pop_count
from std.math import sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.mc import append_mcx
from qumo.algorithms.qml import append_amplitude_embedding_real
from qumo.sim.runner import evolve, sample
from qumo.testing.checks import CheckLog

comptime NY = 6  # constraints: the y register on qubits 0..5
comptime NX = 6  # variables: the solution register on qubits 6..11
comptime L = 2  # MAX_ERRORS of the notebook
comptime SHOTS = 10000


def bit(x: Int, i: Int) -> Int:
    return (x >> i) & 1


def parity(x: Int) -> Int:
    return Int(pop_count(x)) & 1


def b_matrix() -> List[Int]:
    """The notebook's printed B (row j = edge j, column i = node i), row-major: a 6-cycle."""
    var b: List[Int] = [
        1, 1, 0, 0, 0, 0,
        1, 0, 0, 0, 1, 0,
        0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 0, 1,
        0, 0, 0, 0, 1, 1,
    ]
    return b^


def b_times_x(b: List[Int], x: Int) -> Int:
    """Bx over F_2, packed with bit j = row j."""
    var out = 0
    for j in range(NY):
        var s = 0
        for i in range(NX):
            s += b[j * NX + i] * bit(x, i)
        out |= (s & 1) << j
    return out


def bt_times_y(b: List[Int], y: Int) -> Int:
    """B^T y over F_2, packed with bit i = column i (the syndrome of the error y)."""
    var out = 0
    for i in range(NX):
        var s = 0
        for j in range(NY):
            s += b[j * NX + i] * bit(y, j)
        out |= (s & 1) << i
    return out


def objective(b: List[Int], x: Int) -> Int:
    """The objective f(x) = sum_j (-1)^(v_j + (Bx)_j) with v = all ones: satisfied minus unsatisfied constraints."""
    var bx = b_times_x(b, x)
    var f = 0
    for j in range(NY):
        var e = (1 + bit(bx, j)) & 1
        f += 1 - 2 * e
    return f


def error_patterns() -> List[Int]:
    """All y in F_2^6 of weight at most L (22 of them)."""
    var out = List[Int]()
    for y in range(1 << NY):
        if Int(pop_count(y)) <= L:
            out.append(y)
    return out^


def binomial(n: Int, k: Int) -> Int:
    var c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


def count_distinct(xs: List[Int]) -> Int:
    var n = 0
    for i in range(len(xs)):
        var seen = False
        for j in range(i):
            if xs[j] == xs[i]:
                seen = True
        if not seen:
            n += 1
    return n


def tridiagonal_a(m: Int, l: Int) -> List[Float64]:
    """The notebook's A for max-XORSAT (p = 2, r = 1, so d = 0), row-major, size (l+1)^2."""
    var p = 2
    var r = 1
    var d = Float64(p - 2 * r) / sqrt(Float64(r * (p - r)))
    var n = l + 1
    var a = List[Float64](length=n * n, fill=0.0)
    for i in range(n):
        a[i * n + i] = Float64(i) * d
    for k in range(l):
        var off = sqrt(Float64((k + 1) * (m - k)))
        a[k * n + k + 1] = off
        a[(k + 1) * n + k] = off
    return a^


def principal_eigenvector(a: List[Float64], n: Int) -> List[Float64]:
    """Eigenvector of the largest eigenvalue of a symmetric matrix with non-negative off-diagonal
    entries, by power iteration on A + s I with s = 1 + the largest absolute row sum. A + s I is
    non-negative, so the iterate stays positive and the Perron vector is the limit."""
    var s = 0.0
    for i in range(n):
        var row = 0.0
        for j in range(n):
            row += abs(a[i * n + j])
        s = max(s, row)
    s += 1.0
    var u = List[Float64](length=n, fill=1.0)
    for _ in range(400):
        var v = List[Float64](length=n, fill=0.0)
        var nrm = 0.0
        for i in range(n):
            var acc = s * u[i]
            for j in range(n):
                acc += a[i * n + j] * u[j]
            v[i] = acc
            nrm += acc * acc
        nrm = sqrt(nrm)
        for i in range(n):
            u[i] = v[i] / nrm
    return u^


def rayleigh(a: List[Float64], u: List[Float64], n: Int) -> Float64:
    var num = 0.0
    var den = 0.0
    for i in range(n):
        var acc = 0.0
        for j in range(n):
            acc += a[i * n + j] * u[j]
        num += u[i] * acc
        den += u[i] * u[i]
    return num / den


def dicke_amplitudes(w: List[Float64]) -> List[Float64]:
    """The amplitudes a_y = w_|y| / sqrt(C(6, |y|)) for |y| < len(w), else 0. This is the notebook's
    state after Stage 3, sum_k w_k / sqrt(C(m, k)) sum_{|y| = k} |y>, with w_k = 0 for k >= len(w)."""
    var a = List[Float64](length=1 << NY, fill=0.0)
    for y in range(1 << NY):
        var k = Int(pop_count(y))
        if k < len(w):
            a[y] = w[k] / sqrt(Float64(binomial(NY, k)))
    return a^


def amplitude_closed_form(a: List[Float64], b: List[Int], x: Int) -> Float64:
    """<x| psi> = 2^-3 sum_y a_y (-1)^(y . (v + Bx)). The state after decoding is
    sum_y a_y (-1)^(v.y) |0>|B^T y>; the Hadamard transform gives (-1)^(x . B^T y) = (-1)^((Bx) . y)."""
    var c = ((1 << NY) - 1) ^ b_times_x(b, x)
    var s = 0.0
    for y in range(1 << NY):
        if parity(y & c) == 0:
            s += a[y]
        else:
            s -= a[y]
    return s / Float64(1 << (NX // 2))


def dqi_circuit(a: List[Float64], b: List[Int], decode: Bool, hadamard: Bool, measure: Bool) raises -> Circuit:
    """Stages 1-3 (amplitude embedding of a on y), phase, B^T y, optional decoder and Hadamard.
    Qubits: y = 0..5 (bit j of y = qubit j), solution = 6..11 (bit i of x = qubit 6 + i)."""
    var c = Circuit(NY + NX, NX if measure else 0, name="dqi_max_xorsat")
    var wires = List[Int]()
    for i in range(NY):
        wires.append(NY - 1 - i)
    append_amplitude_embedding_real(c, a, wires)
    for j in range(NY):
        c.z(j)
    for j in range(NY):
        for i in range(NX):
            if b[j * NX + i] == 1:
                c.cx(j, NY + i)
    if decode:
        var ctrls = List[Int]()
        for i in range(NX):
            ctrls.append(NY + i)
        for e in error_patterns():
            var syn = bt_times_y(b, e)
            for i in range(NX):
                if bit(syn, i) == 0:
                    c.x(NY + i)
            for j in range(NY):
                if bit(e, j) == 1:
                    append_mcx(c, ctrls, j)
            for i in range(NX):
                if bit(syn, i) == 0:
                    c.x(NY + i)
    if hadamard:
        for i in range(NX):
            c.h(NY + i)
    if measure:
        for i in range(NX):
            c.measure(NY + i, i)
    return c^


def prob_of_x(sv_ar: List[Float64], sv_ai: List[Float64], x: Int) -> Float64:
    """Probability of solution value x with the y register clean (index x << 6)."""
    var idx = x << NY
    return sv_ar[idx] * sv_ar[idx] + sv_ai[idx] * sv_ai[idx]


def main() raises:
    var log = CheckLog("classiq/dqi_max_xorsat")
    var b = b_matrix()

    # ---- Optimal w_k: the notebook's get_optimal_w(m = 6, n = 6, l = 2) ----
    var a_mat = tridiagonal_a(NY, L)
    var u = principal_eigenvector(a_mat, L + 1)
    var lam = rayleigh(a_mat, u, L + 1)
    var printed_w: List[Float64] = [0.4330127, 0.70710678, 0.55901699]
    var worst_w = 0.0
    for k in range(L + 1):
        worst_w = max(worst_w, abs(u[k] - printed_w[k]))
    _ = log.close("optimal w_k matches the notebook's printed vector (max deviation)", worst_w, 0.0, 1e-8)
    _ = log.close("principal eigenvalue of A is 4 (lambda (lambda^2 - 16) = 0)", lam, 4.0, 1e-9)
    var worst_res = 0.0
    for i in range(L + 1):
        var acc = 0.0
        for j in range(L + 1):
            acc += a_mat[i * (L + 1) + j] * u[j]
        worst_res = max(worst_res, abs(acc - 4.0 * u[i]))
    _ = log.close("w_k is an eigenvector of A with eigenvalue 4 (max residual)", worst_res, 0.0, 1e-9)

    # ---- Decodability: the notebook prints 22 errors and 22 syndromes ----
    var errs = error_patterns()
    var syn = List[Int]()
    for e in errs:
        syn.append(bt_times_y(b, e))
    _ = log.check("22 errors of weight at most 2 (notebook prints 22)", len(errs) == 22, "errors=" + String(len(errs)))
    var nd = count_distinct(syn)
    _ = log.check("their 22 syndromes are distinct (notebook prints 22)", nd == 22, "distinct=" + String(nd))

    # ---- The objective by brute force ----
    var f_max = -100
    var f_sum = 0
    var fvals = List[Int]()
    for x in range(1 << NX):
        var fx = objective(b, x)
        fvals.append(fx)
        f_sum += fx
        f_max = max(f_max, fx)
    _ = log.check("MaxCut optimum of the 6-cycle is f = 6 (brute force)", f_max == 6, "max_f=" + String(f_max))
    _ = log.close("uniform sampling gives <f_uniform> = 0 (notebook prints 0.0)", Float64(f_sum) / 64.0, 0.0, 1e-12)

    # ---- Closed form of the output state ----
    var a = dicke_amplitudes(u)
    var psi = List[Float64]()
    var norm_an = 0.0
    var e_an = 0.0
    var e2_an = 0.0
    for x in range(1 << NX):
        var p = amplitude_closed_form(a, b, x)
        psi.append(p)
        norm_an += p * p
        e_an += p * p * Float64(fvals[x])
        e2_an += p * p * Float64(fvals[x] * fvals[x])
    _ = log.close("closed-form output amplitudes are normalised", norm_an, 1.0, 1e-12)
    _ = log.close("closed-form expected objective is 4", e_an, 4.0, 1e-12)
    var var_an = e2_an - e_an * e_an

    # ---- Before decoding: sum_y (-1)^|y| a_y |y>|B^T y> ----
    var pre = evolve(dqi_circuit(a, b, False, False, False))
    var worst_pre = 0.0
    var mass_pre = 0.0
    for y in range(1 << NY):
        var idx = y + (bt_times_y(b, y) << NY)
        var want = a[y]
        if parity(y) == 1:
            want = -want
        worst_pre = max(worst_pre, abs(pre.re[idx] - want) + abs(pre.im[idx]))
        mass_pre += pre.re[idx] * pre.re[idx] + pre.im[idx] * pre.im[idx]
    _ = log.close("state before decoding has the Dicke amplitudes with phase (-1)^|y| (max deviation)", worst_pre, 0.0, 1e-10)
    _ = log.close("state before decoding lies on |y>|B^T y> (probability)", mass_pre, 1.0, 1e-10)

    # ---- Full circuit: decoder, then Hadamard ----
    var sv = evolve(dqi_circuit(a, b, True, True, False))
    var worst_out = 0.0
    var p_clean = 0.0
    var e_qumo = 0.0
    for x in range(1 << NX):
        var idx = x << NY
        var ar = sv.re[idx]
        var ai = sv.im[idx]
        worst_out = max(worst_out, abs(ar - psi[x]) + abs(ai))
        p_clean += ar * ar + ai * ai
        e_qumo += (ar * ar + ai * ai) * Float64(fvals[x])
    _ = log.close("decoder returns y to |0> (probability)", p_clean, 1.0, 1e-10)
    _ = log.close("output amplitudes equal the closed form for all 64 x (max deviation)", worst_out, 0.0, 1e-10)
    _ = log.close("expected objective from the QUMO state is 4", e_qumo, 4.0, 1e-9)
    _ = log.close("probability of the optimal cut x = 37 is 75/256", prob_of_x(sv.re, sv.im, 37), 75.0 / 256.0, 1e-10)
    _ = log.close("probability of the optimal cut x = 26 is 75/256", prob_of_x(sv.re, sv.im, 26), 75.0 / 256.0, 1e-10)

    # ---- Sampling: 10000 shots, as in the notebook ----
    var res = sample(dqi_circuit(a, b, True, True, True), SHOTS, seed=UInt64(1234))
    var shots_f = Float64(SHOTS)
    var hits37 = 0
    var hits26 = 0
    var f_hits = 0
    for x in res.memory:
        if x == 37:
            hits37 += 1
        if x == 26:
            hits26 += 1
        f_hits += objective(b, x)
    _ = log.check("sampler returns one outcome per shot", len(res.memory) == SHOTS, "shots=" + String(len(res.memory)))
    var se = sqrt(var_an / shots_f)
    var f_mean = Float64(f_hits) / shots_f
    _ = log.within("sampled <f> lies within 4 standard errors of the closed form", f_mean, e_an - 4.0 * se, e_an + 4.0 * se)
    _ = log.within("notebook's printed <f_DQI> 4.0012 lies within 4 standard errors of the closed form", 4.0012, e_an - 4.0 * se, e_an + 4.0 * se)

    var p37 = psi[37] * psi[37]
    var s37 = sqrt(p37 * (1.0 - p37) / shots_f)
    _ = log.within("sampled frequency of x = 37 lies within 4 standard errors of 75/256", Float64(hits37) / shots_f, p37 - 4.0 * s37, p37 + 4.0 * s37)
    _ = log.within("notebook's printed count 2925 of 10000 for x = 37 lies within 4 standard errors", 0.2925, p37 - 4.0 * s37, p37 + 4.0 * s37)
    var p26 = psi[26] * psi[26]
    var s26 = sqrt(p26 * (1.0 - p26) / shots_f)
    _ = log.within("sampled frequency of x = 26 lies within 4 standard errors of 75/256", Float64(hits26) / shots_f, p26 - 4.0 * s26, p26 + 4.0 * s26)
    _ = log.within("notebook's printed count 2916 of 10000 for x = 26 lies within 4 standard errors", 0.2916, p26 - 4.0 * s26, p26 + 4.0 * s26)

    log.finish()
