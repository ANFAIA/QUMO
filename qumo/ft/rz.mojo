"""Rz synthesis for Clifford+T architectures: static and run-time angles.

Two paths, both used by `qumo.ft.encode.compile` on the Steane architecture:

1. Static angles (known when the circuit is built):
   * exact: Rz(m pi/4) is T^m up to a global phase (T, S, Z, S T, ...);
   * `synthesize_rz(theta, epsilon, max_t)`: the Clifford+T word with the fewest
     T gates whose distance to Rz(theta) (global-phase invariant,
     d(U, V) = sqrt(1 - |tr(U^dag V)| / 2)) is <= epsilon, found by exhaustive
     enumeration of Matsumoto-Amano normal forms (T|e)(HT|SHT)^m C with C one
     of the 24 single-qubit Cliffords, up to `max_t` T gates. Optimal T-count
     for the given budget, but exponential in max_t (2^max_t words), so it is
     meant for moderate precision; `rz_precision_table()` reports the
     T-count / error trade-off (docs/FT.md). Number-theoretic synthesis
     (gridsynth, T ~ 3 log2(1/eps)) is the scalable alternative and is not
     implemented here.

2. Run-time angles: `append_comparator_rz` builds the repeat-until-success
   circuit of arXiv:2404.05618 (Algorithm 1, as in Guppy FT's
   decompose/_comparator_based_rz.py): n = 1 + ceil(log2(1/eps)) qubits in
   |+>^n, a comparator a >= k flipping the target, S, the inverse comparator,
   H^n and an all-zero test (success probability > 1/2; on failure the target
   received exactly a Z, which is undone before retrying). The angle enters
   only through classical bits (k' = 2^n - k and the quadrant), so with
   `angle_bits` the *same* circuit serves every angle; the constant-angle
   variant emits the k-dependent X gates statically.

   The comparator is derived here as the carry of a + (2^n - k): with
   K' = 2^n - k, c_{i+1} = (a_i OR c_i) if K'_i else (a_i AND c_i); ANDs are
   Gidney temporary logical-ANDs (4 T, measurement-based uncompute, 0 T), ORs
   are De Morgan conjugations by X, so every k-dependent gate is an X (a Pauli
   conditioned on a classical bit) - which a Pauli frame absorbs for free once
   the circuit is encoded. Workspace: n - 1 ancillas (Guppy FT's cascade uses
   n - 2; we spend one qubit on c_1 to keep it angle-independent).
   T-count per attempt: 4 (n - 2) + 2 * 7 (two Toffolis onto the target).
   Repeat-until-success is unrolled `max_attempts` times; attempts after the
   first act on a scratch qubit that a classically-conditioned SWAP exchanges
   with the target only when the previous attempts failed, so no T gate is
   ever classically conditioned. The classical retry logic is computed on a
   classical lane qubit, which `compile` turns into classical expressions.

Idea from Guppy FT (Apache-2.0), files src/guppyft/decompose/_comparator_based_rz.py
and _passes.py (ComparatorRzDecomposer, ToffoliDecomposer); algorithm from
arXiv:2404.05618, temporary AND from arXiv:1709.06648; re-derived.

Error codes: QM-E728 (invalid synthesis options).
"""

from std.math import pi, cos, sin, sqrt, log2, ceil, floor, tan, atan2

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, M2, matrix_1q, op_name


# ---------------------------------------------------------------------------
# Static synthesis
# ---------------------------------------------------------------------------
def rz_matrix(theta: Float64) -> M2:
    return matrix_1q(Op.RZ, theta)


def phase_distance(u: M2, v: M2) -> Float64:
    """sqrt(1 - |tr(U^dag V)| / 2): 0 iff U = e^{i a} V."""
    var p = u.dagger() @ v
    var tr_re = p.re[0] + p.re[3]
    var tr_im = p.im[0] + p.im[3]
    var a = sqrt(tr_re * tr_re + tr_im * tr_im) / 2.0
    if a > 1.0:
        a = 1.0
    return sqrt(1.0 - a)


def sequence_matrix(seq: List[Int]) -> M2:
    """Unitary of a gate sequence in circuit order (first gate applied first)."""
    var m = M2.identity()
    for k in seq:
        m = matrix_1q(k) @ m
    return m


def exact_rz_sequence(theta: Float64, mut out: List[Int]) -> Bool:
    """Rz(m pi/4) as T/S/Z gates (up to global phase); False if theta is not
    a multiple of pi/4."""
    out.clear()
    var r = theta / (pi / 4)
    var m = Int(round(r))
    if abs(r - Float64(m)) > 1e-12:
        return False
    m = ((m % 8) + 8) % 8
    if m == 1:
        out.append(Op.T)
    elif m == 2:
        out.append(Op.S)
    elif m == 3:
        out.append(Op.S)
        out.append(Op.T)
    elif m == 4:
        out.append(Op.Z)
    elif m == 5:
        out.append(Op.Z)
        out.append(Op.T)
    elif m == 6:
        out.append(Op.SDG)
    elif m == 7:
        out.append(Op.TDG)
    return True


struct CliffordTable(Movable):
    """The 24 single-qubit Cliffords (mod phase) as H/S words."""

    var mats: List[M2]
    var words: List[List[Int]]

    def __init__(out self):
        self.mats = List[M2]()
        self.words = List[List[Int]]()
        self.mats.append(M2.identity())
        self.words.append(List[Int]())
        var head = 0
        var gens: List[Int] = [Op.H, Op.S]
        while head < len(self.mats) and len(self.mats) < 24:
            var base = self.mats[head]
            var bw = self.words[head].copy()
            head += 1
            for gk in gens:
                var m = matrix_1q(gk) @ base
                var seen = False
                for o in self.mats:
                    if phase_distance(o, m) < 1e-9:
                        seen = True
                        break
                if not seen:
                    var w = bw.copy()
                    w.append(gk)
                    self.mats.append(m)
                    self.words.append(w^)


struct RzSynthesis(Copyable, Writable):
    var theta: Float64
    var gates: List[Int]  # circuit order
    var t_count: Int
    var error: Float64
    var method: String  # "exact" | "search"

    def __init__(out self, theta: Float64, var gates: List[Int], t_count: Int, error: Float64, method: String):
        self.theta = theta
        self.gates = gates^
        self.t_count = t_count
        self.error = error
        self.method = method

    def write_to(self, mut w: Some[Writer]):
        w.write("Rz(", self.theta, ") ~ ", len(self.gates), " gates, T-count ", self.t_count, ", error ", self.error, " (", self.method, ")")


def _search(target: M2, t_left: Int, mut cur: M2, mut word: List[Int], tbl: CliffordTable,
            mut best_err: Float64, mut best_word: List[Int], mut best_c: Int, eps: Float64, depth_t: Int) -> Bool:
    """DFS over syllables HT / SHT with exactly depth_t remaining T gates; the
    trailing Clifford is optimised in place. Returns True when within eps."""
    if t_left == 0:
        for ci in range(len(tbl.mats)):
            var v = cur @ tbl.mats[ci]
            var e = phase_distance(target, v)
            if e < best_err - 1e-15:
                best_err = e
                best_word = word.copy()
                best_c = ci
        return best_err <= eps
    var hm = matrix_1q(Op.H)
    var sm = matrix_1q(Op.S)
    var tm = matrix_1q(Op.T)
    # syllable HT
    var c1 = cur @ hm @ tm
    word.append(0)
    if _search(target, t_left - 1, c1, word, tbl, best_err, best_word, best_c, eps, depth_t):
        _ = word.pop()
        return True
    _ = word.pop()
    var c2 = cur @ sm @ hm @ tm
    word.append(1)
    if _search(target, t_left - 1, c2, word, tbl, best_err, best_word, best_c, eps, depth_t):
        _ = word.pop()
        return True
    _ = word.pop()
    return False


def synthesize_rz(theta: Float64, epsilon: Float64 = 1e-2, max_t: Int = 16) raises -> RzSynthesis:
    """Fewest-T Clifford+T approximation of Rz(theta) within epsilon (see module doc)."""
    if epsilon <= 0.0 or epsilon >= 1.0:
        raise Error("QM-E728: epsilon must lie in (0, 1) (got " + String(epsilon) + "). Hint: 1e-2 needs ~15 T gates.")
    if max_t < 0 or max_t > 24:
        raise Error("QM-E728: max_t must lie in [0, 24] (got " + String(max_t) + "). Hint: the search enumerates ~2^max_t words.")
    var ex = List[Int]()
    if exact_rz_sequence(theta, ex):
        var tc = 0
        for g in ex:
            if g == Op.T or g == Op.TDG:
                tc += 1
        return RzSynthesis(theta, ex^, tc, 0.0, "exact")
    var target = rz_matrix(theta)
    var tbl = CliffordTable()
    var best_err = 2.0
    var best_word = List[Int]()
    var best_c = 0
    var best_lead = False
    var found = False
    for t in range(max_t + 1):
        # (T | e) prefix: e with t syllables, or T with t-1 syllables
        for lead in range(2):
            var syl = t - lead
            if syl < 0:
                continue
            var cur = M2.identity()
            if lead == 1:
                cur = matrix_1q(Op.T)
            var word = List[Int]()
            var e_before = best_err
            var w_save = best_word.copy()
            var c_save = best_c
            if _search(target, syl, cur, word, tbl, best_err, best_word, best_c, epsilon, syl):
                best_lead = lead == 1
                found = True
                break
            if best_err < e_before:
                best_lead = lead == 1
            else:
                best_word = w_save^
                best_c = c_save
        if found:
            break
    # operator V = [T] * prod(syllables) * C ; circuit order = reverse
    var ops = List[Int]()
    for g in tbl.words[best_c]:
        ops.append(g)
    var i = len(best_word) - 1
    while i >= 0:
        # syllable operator H T (or S H T): applied T first, then H (then S)
        ops.append(Op.T)
        ops.append(Op.H)
        if best_word[i] == 1:
            ops.append(Op.S)
        i -= 1
    if best_lead:
        ops.append(Op.T)
    var tc = 0
    for g in ops:
        if g == Op.T or g == Op.TDG:
            tc += 1
    var err = phase_distance(target, sequence_matrix(ops))
    return RzSynthesis(theta, ops^, tc, err, "search")


def rz_precision_table(theta: Float64, max_t: Int) raises -> List[RzSynthesis]:
    """Best error reachable with T-count budgets 0..max_t (epsilon -> tiny)."""
    var out = List[RzSynthesis]()
    for t in range(max_t + 1):
        out.append(synthesize_rz(theta, 1e-15, t))
    return out^


# ---------------------------------------------------------------------------
# Comparator-based repeat-until-success Rz
# ---------------------------------------------------------------------------
def comparator_register_size(epsilon: Float64) raises -> Int:
    if epsilon <= 0.0 or epsilon >= 1.0:
        raise Error("QM-E728: epsilon must lie in (0, 1) (got " + String(epsilon) + ").")
    return 1 + Int(ceil(log2(1.0 / epsilon)))


@fieldwise_init
struct ComparatorAngle(ImplicitlyCopyable, Writable):
    """Classical data that encodes an angle for the comparator circuit."""

    var kp: Int  # K' = 2^n - k (n bits)
    var quadrant: Int  # 0..3: Rz(quadrant * pi/2) applied exactly (S / Z)
    var n: Int
    var effective_theta: Float64  # the angle the circuit implements on success

    def write_to(self, mut w: Some[Writer]):
        w.write("ComparatorAngle(kp=", self.kp, ", quadrant=", self.quadrant, ", n=", self.n, ", theta_eff=", self.effective_theta, ")")


def comparator_angle(theta: Float64, n: Int) -> ComparatorAngle:
    """Host-side classical preprocessing of theta (Guppy FT does it at run time
    with its own tan/floor): quadrant, k = 2^{n-1} + floor(2^{n-1} tan(t/2) + 1/2),
    K' = 2^n - k."""
    var hp = pi / 2
    var q = Int(floor(theta / hp))
    var tr = theta - Float64(q) * hp
    var u = 1 << (n - 1)
    var k = u + Int(floor(Float64(u) * tan(tr / 2.0) + 0.5))
    if k >= 2 * u:
        q += 1
        k = u
    var quad = ((q % 4) + 4) % 4
    var x = Float64(k - u)
    var uf = Float64(u)
    var eff = Float64(quad) * hp + 2.0 * atan2(uf + x, uf - x) - hp
    return ComparatorAngle((1 << n) - k, quad, n, eff)


struct ComparatorRzInfo(Copyable, Writable):
    var n: Int
    var ancillas: Int  # workspace + scratch + lane qubits added
    var attempts: Int
    var fail_clbit: Int  # 1 = every attempt failed (post-select on 0)
    var t_count: Int  # T gates emitted (all attempts, Toffolis as 7 T)
    var angle_clbits: List[Int]

    def __init__(out self):
        self.n = 0
        self.ancillas = 0
        self.attempts = 0
        self.fail_clbit = -1
        self.t_count = 0
        self.angle_clbits = List[Int]()

    def write_to(self, mut w: Some[Writer]):
        w.write("ComparatorRz(n=", self.n, ", ancillas=", self.ancillas, ", attempts=", self.attempts, ", T=", self.t_count, ", fail_clbit=", self.fail_clbit, ")")


struct _CB:
    """Circuit builder helpers with an optional classical condition bit."""

    @staticmethod
    def x_if(mut c: Circuit, q: Int, bit: Int, static_bit: Int):
        """X on q when the K' bit is 1: static (static_bit 0/1) or classical (bit >= 0)."""
        if bit >= 0:
            c.x(q)
            c.c_if(bit, 1)
        elif static_bit == 1:
            c.x(q)


def _toffoli(mut c: Circuit, a: Int, b: Int, t: Int, mut tcount: Int):
    """Clifford+T Toffoli (7 T), Nielsen & Chuang Fig. 4.9 ordering."""
    c.h(t)
    c.cx(b, t)
    c.tdg(t)
    c.cx(a, t)
    c.t(t)
    c.cx(b, t)
    c.tdg(t)
    c.cx(a, t)
    c.t(b)
    c.t(t)
    c.h(t)
    c.cx(a, b)
    c.t(a)
    c.tdg(b)
    c.cx(a, b)
    tcount += 7


def _tand_compute(mut c: Circuit, a: Int, b: Int, t: Int, mut tcount: Int):
    """t (in |0>) <- a AND b with 4 T gates (Gidney temporary logical-AND)."""
    c.h(t)
    c.t(t)
    c.cx(b, t)
    c.tdg(t)
    c.cx(a, t)
    c.t(t)
    c.cx(b, t)
    c.tdg(t)
    c.h(t)
    c.sdg(t)
    tcount += 4


def _tand_uncompute(mut c: Circuit, a: Int, b: Int, t: Int):
    """Measurement-based uncompute: X-measure t, CZ(a, b) if 1, reset t."""
    c.h(t)
    var m = c.num_clbits
    c.measure(t, m)
    c.cz(a, b)
    c.c_if(m, 1)
    c.reset(t)


def _comparator(mut c: Circuit, a: List[Int], w: List[Int], target: Int, kbits: List[Int], kp: Int,
                inverse: Bool, mut tcount: Int):
    """target ^= [a >= k] via the carry of a + K' (see module doc)."""
    var n = len(a)
    var kbv = List[Int]()  # static K' bit (or -1 when read at run time)
    var kcv = List[Int]()  # classical bit holding K'_i (or -1 when static)
    for i in range(n):
        if len(kbits) > 0:
            kbv.append(-1)
            kcv.append(kbits[i])
        else:
            kbv.append((kp >> i) & 1)
            kcv.append(-1)

    if not inverse:
        # c_1 = K'_0 AND a_0 into w[0]
        if len(kbits) > 0:
            c.cx(a[0], w[0])
            c.c_if(kbits[0], 1)
        elif (kp & 1) == 1:
            c.cx(a[0], w[0])
        for i in range(1, n - 1):
            _CB.x_if(c, a[i], kcv[i], kbv[i])
            _CB.x_if(c, w[i - 1], kcv[i], kbv[i])
            _tand_compute(c, a[i], w[i - 1], w[i], tcount)
            _CB.x_if(c, w[i], kcv[i], kbv[i])
            _CB.x_if(c, a[i], kcv[i], kbv[i])
            _CB.x_if(c, w[i - 1], kcv[i], kbv[i])
    var j = n - 1
    _CB.x_if(c, a[j], kcv[j], kbv[j])
    _CB.x_if(c, w[j - 1], kcv[j], kbv[j])
    _toffoli(c, a[j], w[j - 1], target, tcount)
    _CB.x_if(c, a[j], kcv[j], kbv[j])
    _CB.x_if(c, w[j - 1], kcv[j], kbv[j])
    _CB.x_if(c, target, kcv[j], kbv[j])
    if inverse:
        var i = n - 2
        while i >= 1:
            _CB.x_if(c, a[i], kcv[i], kbv[i])
            _CB.x_if(c, w[i - 1], kcv[i], kbv[i])
            _CB.x_if(c, w[i], kcv[i], kbv[i])
            _tand_uncompute(c, a[i], w[i - 1], w[i])
            _CB.x_if(c, a[i], kcv[i], kbv[i])
            _CB.x_if(c, w[i - 1], kcv[i], kbv[i])
            i -= 1
        if len(kbits) > 0:
            c.cx(a[0], w[0])
            c.c_if(kbits[0], 1)
        elif (kp & 1) == 1:
            c.cx(a[0], w[0])


def _new_qubits(mut c: Circuit, count: Int) -> List[Int]:
    var out = List[Int]()
    for _ in range(count):
        out.append(c.num_qubits)
        c.num_qubits += 1
    return out^


def _new_clbit(mut c: Circuit) -> Int:
    var b = c.num_clbits
    c.num_clbits += 1
    return b


def append_comparator_rz(mut c: Circuit, target: Int, theta: Float64, epsilon: Float64,
                         max_attempts: Int, angle_bits: List[Int] = List[Int]()) raises -> ComparatorRzInfo:
    """Append the comparator RUS circuit for Rz(theta) on `target` (new qubits
    and clbits are added to `c`). With `angle_bits` = [K'_0..K'_{n-1}, q0, q1]
    (clbits holding comparator_angle(theta, n)) the circuit ignores `theta`
    and reads the angle at run time."""
    var n = comparator_register_size(epsilon)
    if n < 3:
        n = 3
    if max_attempts < 1:
        raise Error("QM-E728: max_attempts must be >= 1. Hint: success probability per attempt is > 1/2; 4-8 attempts are typical.")
    var info = ComparatorRzInfo()
    info.n = n
    info.attempts = max_attempts
    var runtime = len(angle_bits) > 0
    if runtime and len(angle_bits) != n + 2:
        raise Error("QM-E728: angle_bits must hold n + 2 = " + String(n + 2) + " clbits (K' bits little-endian, then 2 quadrant bits). Hint: use comparator_register_size(epsilon) and comparator_angle(theta, n).")
    var ang = comparator_angle(theta, n)
    var kbits = List[Int]()
    if runtime:
        for i in range(n):
            kbits.append(angle_bits[i])
        info.angle_clbits = angle_bits.copy()
        # quadrant: Rz(q pi/2) = S^q up to phase
        c.s(target)
        c.c_if(angle_bits[n], 1)
        c.z(target)
        c.c_if(angle_bits[n + 1], 1)
    else:
        if ang.quadrant >= 2:
            c.z(target)
        if (ang.quadrant & 1) == 1:
            c.s(target)
    var a = _new_qubits(c, n)
    var w = _new_qubits(c, n - 1)
    var lane = _new_qubits(c, 1)[0]
    var work = -1
    if max_attempts > 1:
        work = _new_qubits(c, 1)[0]
    info.ancillas = 2 * n + (1 if max_attempts > 1 else 0)
    var tcount = 0
    var retry = -1  # clbit: all previous attempts failed
    for att in range(max_attempts):
        var tq = target
        if att > 0:
            tq = work
            c.swap(target, work)
            c.c_if(retry, 1)
        for q in a:
            c.reset(q)
        for q in w:
            c.reset(q)
        for q in a:
            c.h(q)
        _comparator(c, a, w, tq, kbits, ang.kp, False, tcount)
        c.s(tq)
        _comparator(c, a, w, tq, kbits, ang.kp, True, tcount)
        for q in a:
            c.h(q)
        var ms = List[Int]()
        for q in a:
            var b = _new_clbit(c)
            c.measure(q, b)
            ms.append(b)
        # fail = NOT (all zero), computed on the classical lane
        var zeros = List[Int](length=n, fill=0)
        c.reset(lane)
        c.x(lane)
        c.x(lane)
        c.c_if_register(ms, zeros)
        var f = _new_clbit(c)
        c.measure(lane, f)
        c.z(tq)
        c.c_if(f, 1)
        if att > 0:
            c.swap(target, work)
            c.c_if(retry, 1)
            c.reset(work)
            # retry := retry AND fail
            c.reset(lane)
            c.x(lane)
            var bits: List[Int] = [retry, f]
            var ones: List[Int] = [1, 1]
            c.c_if_register(bits, ones)
            var r = _new_clbit(c)
            c.measure(lane, r)
            retry = r
        else:
            retry = f
    info.fail_clbit = retry
    info.t_count = tcount
    return info^


def comparator_t_count_per_attempt(n: Int) -> Int:
    return 4 * (n - 2) + 14


def angle_register(mut c: Circuit, ang: ComparatorAngle) -> List[Int]:
    """Demo helper: n + 2 fresh qubits set to the bits of (K', quadrant) and
    measured into fresh clbits, i.e. the angle arrives as measurement results
    at run time (the rest of the circuit does not depend on it)."""
    var bits = List[Int]()
    var vals = List[Int]()
    for i in range(ang.n):
        vals.append((ang.kp >> i) & 1)
    vals.append(ang.quadrant & 1)
    vals.append((ang.quadrant >> 1) & 1)
    for v in vals:
        var q = _new_qubits(c, 1)[0]
        if v == 1:
            c.x(q)
        var b = _new_clbit(c)
        c.measure(q, b)
        bits.append(b)
    return bits^
