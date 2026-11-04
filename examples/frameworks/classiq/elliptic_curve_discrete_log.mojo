"""Elliptic curve discrete logarithm: the notebook's Shor ECDLP circuit on E(F_7) on QUMO.

Reproduces: Classiq library - Solving Elliptic Curve Discrete Logarithm Problem with Shor's Algorithm
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/number_theory_and_cryptography/elliptic_curves/elliptic_curve_discrete_log.ipynb (Classiq Library, MIT)

The curve is y^2 = x^3 + 5x + 4 over F_7, G = (0, 5) has order 5, the notebook's start point is P0 = (4, 2) and its target is P = (0, 2) = 4G, so -P = G. Two 3-qubit registers x1 and x2 are put in uniform superposition, the point register starts at P0 and receives x1 G and x2 (-P) = x2 G by controlled point additions (bit i adds 2^i G), then inverse QFTs act on x1 and x2 and the registers are measured. The point register holds P0 + (x1 + x2) G, so the outcomes concentrate on y1 = y2 (mod 5), and the logarithm -y1 y2^-1 (mod 5) is 4.
The checks compare QUMO with the notebook's worked example [4,2] + [0,5] = [4,5], with the multiples 2G = [2,1] and 4G = [0,2] it prints, with its frequencies (2048 shots, its top ten rows), and with its assert that the top outcomes decode to 4. The exact law is checked against a brute-force Fourier sum over the oracle values computed from the definition, and the ancillas, the coset invariant and seeded samples are checked too.

Deviations from the notebook:
* ec_point_add is not a reversible modular-arithmetic circuit. QUMO has no quantum modular multiply, square or inverse (the notebook's modular_multiply, modular_square and Kaliski's modular_inverse_inplace). Each controlled addition of 2^i G is a controlled permutation table on the 6-qubit point register (append_controlled_permutation, as QUMO's order finding does for modular multiplication). The table is the notebook's generic-case point addition (the case its own text says it implements) on the coset P0 + <G>, the only set of points the register ever reaches from P0, and the identity elsewhere. The notebook's circuit was not run here; its printed rows and worked example are the check. The ancillas return to |0>.
* The circuit has 16 qubits: 12 data qubits and 4 clean ancillas. Classiq's synthesized 23-qubit circuit (54,420 CX gates, depth 69,538), its cloud synthesis, its job links and the full Kaliski inverse circuit are not reproduced.
* The exact law comes from QUMO's statevector. Sampling uses QUMO's seeded sampler with the notebook's 2048 shots, and the notebook's printed frequencies are compared with the exact probabilities within 4 sigma.
* As in the notebook's code (not its markdown), each 3-bit register takes a full Hadamard over the values 0..7; the trimmed uniform state is not used.
* The notebook's closest_fraction rounds with ties to even, as Python's round does; decode() does the same.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/elliptic_curve_discrete_log.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, pi, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import append_controlled_permutation, iqft, num_mcz_ancillas
from qumo.testing.checks import CheckLog


comptime P_MOD = 7  # the field F_7; the curve is y^2 = x^3 + 5x + 4
comptime CURVE_A = 5
comptime EVAL = 3  # bits per scalar register: x1 and x2 range over 0..7
comptime NDATA = 12  # x1 = qubits 0..2, x2 = 3..5, ecp.x = 6..8, ecp.y = 9..11
comptime ECP0 = 6  # first qubit of the point register; its value is x + 8 y
comptime SHOTS = 2048  # the notebook's shot count (its printed counts sum to 2048)


# ---------------------------------------------------------------------------
# Classical arithmetic (the notebook's curve, computed from the definition)
# ---------------------------------------------------------------------------
def mod_n(v: Int, n: Int) -> Int:
    var r = v % n
    if r < 0:
        r += n
    return r


def inv_mod_n(b: Int, n: Int) -> Int:
    """The inverse of b mod n by search (gcd(b, n) = 1), or -1."""
    for k in range(n):
        if mod_n(b * k, n) == 1:
            return k
    return -1


def gcd(a: Int, b: Int) -> Int:
    var x = abs(a)
    var y = abs(b)
    while y != 0:
        var t = x % y
        x = y
        y = t
    return x


def round_half_even(num: Int, den: Int) -> Int:
    """The integer nearest to num / den (num >= 0, den > 0), ties to even as Python's round."""
    var q = num // den
    var rem = num - q * den
    if 2 * rem > den or (2 * rem == den and q % 2 == 1):
        q += 1
    return q


def ec_add(P: Tuple[Int, Int], Q: Tuple[Int, Int]) raises -> Tuple[Int, Int]:
    """Affine P + Q on y^2 = x^3 + 5x + 4 over F_7 (the generic case, P.x != Q.x, as in the notebook)."""
    var dx = mod_n(P[0] - Q[0], P_MOD)
    if dx == 0:
        raise Error("QM-E802: ec_add() needs P.x != Q.x (the generic case). Hint: the register stays in the coset P0 + <G>, where this never happens.")
    var lam = mod_n((P[1] - Q[1]) * inv_mod_n(dx, P_MOD), P_MOD)
    var x3 = mod_n(lam * lam - P[0] - Q[0], P_MOD)
    var y3 = mod_n(lam * (P[0] - x3) - P[1], P_MOD)
    return (x3, y3)


def chain(P0: Tuple[Int, Int], G: Tuple[Int, Int], n: Int) raises -> List[Tuple[Int, Int]]:
    """P0 + m G for m = 0 .. n-1, by repeated affine addition."""
    var out = List[Tuple[Int, Int]]()
    out.append(P0)
    for m in range(1, n):
        var nxt = ec_add(out[m - 1], G)
        out.append(nxt)
    return out^


def coset_of(C: List[Tuple[Int, Int]], x: Int, y: Int) -> Int:
    """The index j with C[j] = (x, y), or -1."""
    for j in range(len(C)):
        if C[j][0] == x and C[j][1] == y:
            return j
    return -1


def pidx(x: Int, y: Int) -> Int:
    """The value of the point register for (x, y): ecp.x + 8 ecp.y."""
    return x + 8 * y


# ---------------------------------------------------------------------------
# Oracle: controlled point additions as permutation tables
# ---------------------------------------------------------------------------
def addend_perm(Q: Tuple[Int, Int], C: List[Tuple[Int, Int]]) raises -> List[Int]:
    """Permutation of the 6-bit point register for P -> P + Q, controlled addition of a classical point.
    On the coset C = P0 + <G> it is the notebook's point addition (Q is in <G>, so C + Q = C; the
    generic case holds since no point of C has x equal to Q.x); elsewhere it is the identity."""
    var perm = List[Int](length=64, fill=0)
    for v in range(64):
        perm[v] = v
        var x = v % 8
        var y = v // 8
        if x < P_MOD and y < P_MOD and coset_of(C, x, y) >= 0:
            var r = ec_add((x, y), Q)
            if coset_of(C, r[0], r[1]) < 0:
                raise Error("QM-E802: point addition left the coset. Hint: the addend must lie in <G>.")
            perm[v] = pidx(r[0], r[1])
    return perm^


def load_point(mut c: Circuit, x: Int, y: Int):
    """X gates that write the classical point (x, y) into the point register."""
    for j in range(3):
        if ((x >> j) & 1) == 1:
            c.x(ECP0 + j)
        if ((y >> j) & 1) == 1:
            c.x(ECP0 + 3 + j)


def ancilla_list(nanc: Int) -> List[Int]:
    var anc = List[Int]()
    for k in range(nanc):
        anc.append(NDATA + k)
    return anc^


def point_add_gate(mut c: Circuit, control: Int, perm: List[Int], anc: List[Int]) raises:
    """When `control` is 1, the point register (qubits 6..11) takes the table permutation."""
    var ctrls: List[Int] = [control]
    var targets: List[Int] = [6, 7, 8, 9, 10, 11]
    append_controlled_permutation(c, ctrls, targets, perm, anc)


def scalar_mult_add(mut c: Circuit, reg: List[Int], p1: List[Int], p2: List[Int], p4: List[Int], anc: List[Int]) raises:
    """The notebook's ec_scalar_mult_add: scalar bit i (LSB first) controls the addition of 2^i G = G, 2G, 4G."""
    point_add_gate(c, reg[0], p1, anc)
    point_add_gate(c, reg[1], p2, anc)
    point_add_gate(c, reg[2], p4, anc)


def ecdlp_circuit(p1: List[Int], p2: List[Int], p4: List[Int], P0: Tuple[Int, Int], nanc: Int, measure: Bool) raises -> Circuit:
    """The notebook's shor_ecdlp: H on x1 and x2, the point register loaded with P0, the two scalar
    multiplications, inverse QFTs on x1 and x2, and optionally a measurement of all 12 data qubits.
    Layout: x1 = qubits 0..2, x2 = 3..5, ecp.x = 6..8, ecp.y = 9..11, clean ancillas 12..12+nanc-1.
    Clbit q = qubit q, so a shot is y1 + 8 y2 + 64 (ecp.x + 8 ecp.y)."""
    var nclb = NDATA if measure else 0
    var c = Circuit(NDATA + nanc, nclb, name="shor_ecdlp")
    var X1: List[Int] = [0, 1, 2]
    var X2: List[Int] = [3, 4, 5]
    var anc = ancilla_list(nanc)
    for q in range(2 * EVAL):
        c.h(q)
    load_point(c, P0[0], P0[1])
    scalar_mult_add(c, X1, p1, p2, p4, anc)
    scalar_mult_add(c, X2, p1, p2, p4, anc)
    c.compose(iqft(EVAL), 0)
    c.compose(iqft(EVAL), EVAL)
    if measure:
        for q in range(NDATA):
            c.measure(q, q)
    return c^


# ---------------------------------------------------------------------------
# References computed from the definitions, never from the circuit
# ---------------------------------------------------------------------------
def reference_law(M: List[Tuple[Int, Int]], C: List[Tuple[Int, Int]], size: Int) -> List[Float64]:
    """Exact law of (y1, y2, ecp) from the definition: ecp = M[x1 + x2] = P0 + (x1 + x2) G, and the
    amplitude of |y1, y2, z> after the inverse QFTs is (1/8)(1/8) sum over the (x1, x2) with ecp = z of
    exp(-2 pi i (x1 y1 + x2 y2) / 8). Index y1 + 8 y2 + 64 pidx(z), zero when an ancilla is set."""
    var re = List[Float64](length=5 * 64, fill=0.0)
    var im = List[Float64](length=5 * 64, fill=0.0)
    for x1 in range(8):
        for x2 in range(8):
            var z = M[x1 + x2]
            var j = coset_of(C, z[0], z[1])
            for y1 in range(8):
                for y2 in range(8):
                    var ang = -2.0 * pi * Float64((x1 * y1 + x2 * y2) % 8) / 8.0
                    re[j * 64 + y1 + 8 * y2] += cos(ang)
                    im[j * 64 + y1 + 8 * y2] += sin(ang)
    var law = List[Float64](length=size, fill=0.0)
    for j in range(5):
        var base = pidx(C[j][0], C[j][1]) * 64
        for k in range(64):
            var a = re[j * 64 + k]
            var b = im[j * 64 + k]
            law[base + k] = (a * a + b * b) / 4096.0
    return law^


def pair_ref(law: List[Float64], C: List[Tuple[Int, Int]], k: Int) -> Float64:
    """Exact P(y1, y2) for k = y1 + 8 y2, summed over the coset points (brute-force law)."""
    var s = 0.0
    for j in range(5):
        s += law[k + 64 * pidx(C[j][0], C[j][1])]
    return s


def ecp_mass(probs: List[Float64], x: Int, y: Int) -> Float64:
    """P(ecp = (x, y)) with the x registers summed and the ancillas at |0>."""
    var s = 0.0
    var base = pidx(x, y) * 64
    for k in range(64):
        s += probs[base + k]
    return s


def decode(y1: Int, y2: Int) -> Int:
    """The notebook's post-processing of one outcome: round 5 y / 8 to the nearest integer (ties to even),
    keep the pair when the second value is coprime to 5, and return -a1 a2^-1 mod 5. -1 when discarded."""
    var a1 = mod_n(round_half_even(5 * y1, 8), 5)
    var a2 = mod_n(round_half_even(5 * y2, 8), 5)
    if gcd(a2, 5) != 1:
        return -1
    return mod_n(-a1 * inv_mod_n(a2, 5), 5)


def notebook_row(mut log: CheckLog, probs: List[Float64], k: Int, y1: Int, y2: Int, zx: Int, zy: Int, freq: Float64):
    """One printed row of the notebook's table: its frequency (2048 shots) against the exact probability."""
    var p = probs[y1 + 8 * y2 + 64 * pidx(zx, zy)]
    var sigma = sqrt(p * (1.0 - p) / Float64(SHOTS))
    var ok = abs(freq - p) <= 4.0 * sigma
    var label = "B notebook row " + String(k) + " frequency within 4 sigma of the exact probability"
    _ = log.check(label, ok, "notebook=" + String(freq) + " exact=" + String(p) + " 4sigma=" + String(4.0 * sigma))


def basis_run(perm: List[Int], steps: Int, control_on: Bool, P0x: Int, P0y: Int, nanc: Int) raises -> List[Float64]:
    """Start with the point at P0 (and the control at |1> when control_on), apply `steps` controlled
    table additions, and return the exact probabilities."""
    var c = Circuit(NDATA + nanc, 0, name="point_add")
    var anc = ancilla_list(nanc)
    if control_on:
        c.x(0)
    load_point(c, P0x, P0y)
    for _ in range(steps):
        point_add_gate(c, 0, perm, anc)
    return evolve(c).probabilities()


def main() raises:
    var log = CheckLog("classiq/elliptic_curve_discrete_log")
    var nanc = num_mcz_ancillas(1 + 6)  # one control and the six point qubits
    var P0 = (4, 2)  # the notebook's INITIAL_POINT
    var G = (0, 5)  # GENERATOR_G, of order 5
    var G2 = (2, 1)  # 2 G as printed in the notebook
    var G4 = (0, 2)  # 4 G as printed in the notebook (the target P = [0, 2])

    var M = chain(P0, G, 15)  # M[m] = P0 + m G
    var C = chain(P0, G, 5)  # the coset P0 + <G> that the register stays in
    var perm_g = addend_perm(G, C)
    var perm_2g = addend_perm(G2, C)
    var perm_4g = addend_perm(G4, C)

    # ---------------- A: the controlled point addition on basis states ----------------
    var pA = basis_run(perm_g, 1, True, P0[0], P0[1], nanc)
    _ = log.close("A notebook example [4,2] + [0,5] gives [4,5] (probability)", pA[1 + 64 * pidx(4, 5)], 1.0, 1e-10)

    var pOff = basis_run(perm_g, 1, False, P0[0], P0[1], nanc)
    _ = log.close("A control 0 leaves the point at [4,2] (probability)", pOff[64 * pidx(4, 2)], 1.0, 1e-10)

    var r2 = ec_add(P0, G2)
    var p2 = basis_run(perm_2g, 1, True, P0[0], P0[1], nanc)
    _ = log.close("A [4,2] + 2G from the 2G table (probability)", p2[1 + 64 * pidx(r2[0], r2[1])], 1.0, 1e-10)

    var r4 = ec_add(P0, G4)
    var p4 = basis_run(perm_4g, 1, True, P0[0], P0[1], nanc)
    _ = log.close("A [4,2] + 4G from the 4G table (probability)", p4[1 + 64 * pidx(r4[0], r4[1])], 1.0, 1e-10)

    # the notebook's order of G is 5: five controlled additions of G return the point to P0
    for k in range(1, 6):
        var pk = basis_run(perm_g, k, True, P0[0], P0[1], nanc)
        var want = M[k]
        _ = log.close(
            "A " + String(k) + " controlled additions of G reach P0 + " + String(k) + "G (probability)",
            pk[1 + 64 * pidx(want[0], want[1])], 1.0, 1e-10,
        )

    # ---------------- B: the full ECDLP circuit ----------------
    var circ = ecdlp_circuit(perm_g, perm_2g, perm_4g, P0, nanc, False)
    var probs = evolve(circ).probabilities()
    var law = reference_law(M, C, len(probs))

    var dev = 0.0
    for k in range(len(probs)):
        dev = max(dev, abs(probs[k] - law[k]))
    _ = log.close("B exact law of (y1, y2, ecp) equals the brute-force law (max deviation)", dev, 0.0, 1e-10)

    var clean = 0.0
    for k in range(1 << NDATA):
        clean += probs[k]
    _ = log.close("B ancillas return to |0> (probability)", clean, 1.0, 1e-10)

    var on_coset = 0.0
    for j in range(5):
        on_coset += ecp_mass(probs, C[j][0], C[j][1])
    _ = log.close("B ecp register stays on the coset P0 + <G> (probability)", on_coset, 1.0, 1e-10)

    var count = List[Float64](length=5, fill=0.0)
    for x1 in range(8):
        for x2 in range(8):
            var z = M[x1 + x2]
            count[coset_of(C, z[0], z[1])] += 1.0 / 64.0
    var dev_e = 0.0
    for j in range(5):
        dev_e = max(dev_e, abs(ecp_mass(probs, C[j][0], C[j][1]) - count[j]))
    _ = log.close("B ecp marginal matches the brute-force count of (x1, x2) per point (max deviation)", dev_e, 0.0, 1e-10)

    # the notebook's printed rows: (y1, y2, ecp.x, ecp.y, frequency), y = 0.375 -> 3 and 0.625 -> 5
    notebook_row(log, probs, 1, 0, 0, 3, 2, 0.045410)
    notebook_row(log, probs, 2, 0, 0, 4, 5, 0.042480)
    notebook_row(log, probs, 3, 3, 3, 3, 2, 0.035645)
    notebook_row(log, probs, 4, 5, 5, 3, 2, 0.035645)
    notebook_row(log, probs, 5, 0, 0, 3, 5, 0.039551)
    notebook_row(log, probs, 6, 0, 0, 5, 0, 0.033203)
    notebook_row(log, probs, 7, 5, 5, 3, 5, 0.032715)
    notebook_row(log, probs, 8, 3, 3, 4, 5, 0.031738)
    notebook_row(log, probs, 9, 5, 5, 4, 5, 0.031738)
    notebook_row(log, probs, 10, 0, 0, 4, 2, 0.028809)

    # the notebook asserts that its valid outcomes decode to logarithm 4: the ten most probable valid pairs do
    var pair = List[Float64](length=64, fill=0.0)
    for k in range(64):
        var s = 0.0
        for j in range(5):
            s += probs[k + 64 * pidx(C[j][0], C[j][1])]
        pair[k] = s
    var used = List[Bool](length=64, fill=False)
    var top_all_4 = True
    var top_n = 0
    for _ in range(10):
        var best = -1
        var bestp = -1.0
        for k in range(64):
            if used[k] or decode(k % 8, k // 8) < 0:
                continue
            if pair[k] > bestp:
                bestp = pair[k]
                best = k
        used[best] = True
        top_n += 1
        if decode(best % 8, best // 8) != 4:
            top_all_4 = False
    _ = log.check("B the ten most probable valid outcomes decode to the notebook's logarithm 4", top_all_4 and top_n == 10, "count=" + String(top_n))

    # ---------------- seeded samples of the measured registers ----------------
    var mcirc = ecdlp_circuit(perm_g, perm_2g, perm_4g, P0, nanc, True)
    var res = sample(mcirc, SHOTS, seed=UInt64(1234))
    var forbidden = 0
    var valid_n = 0
    var log4_n = 0
    for m in res.memory:
        if law[m] <= 1e-12:
            forbidden += 1
        var d = decode(m & 7, (m >> 3) & 7)
        if d >= 0:
            valid_n += 1
            if d == 4:
                log4_n += 1
    var p_valid = 0.0
    var p_log4 = 0.0
    for k in range(64):
        var d = decode(k % 8, k // 8)
        if d >= 0:
            p_valid += pair_ref(law, C, k)
            if d == 4:
                p_log4 += pair_ref(law, C, k)
    var p_cond = p_log4 / p_valid
    var f_valid = Float64(valid_n) / Float64(SHOTS)
    var f_cond = Float64(log4_n) / Float64(valid_n)
    var z_valid = abs(f_valid - p_valid) / sqrt(p_valid * (1.0 - p_valid) / Float64(SHOTS))
    var z_cond = abs(f_cond - p_cond) / sqrt(p_cond * (1.0 - p_cond) / Float64(valid_n))
    print("exact P(beta coprime to 5) =", p_valid, " P(log 4 | coprime) =", p_cond)
    print("sampled", SHOTS, "shots: coprime", valid_n, " log 4:", log4_n)
    _ = log.check("B sampled shots only land on outcomes the exact law allows", forbidden == 0, "forbidden=" + String(forbidden))
    _ = log.within("B sampled coprime frequency, |z| against the exact value", z_valid, 0.0, 4.0)
    _ = log.within("B sampled conditional frequency of log 4, |z| against the exact value", z_cond, 0.0, 4.0)

    log.finish()


