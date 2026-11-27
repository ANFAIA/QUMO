"""Exact output error and acceptance of small magic-state factories, by enumerating every input fault.

Reproduces: Balada Gaggioli, Marecek - The limits of magic-state distillation (3 T to CS, 8-input CCZ cube factory, 9-input T factory of Prop. 7)
Source: https://arxiv.org/abs/2610.31115
Framework: QUMO - https://github.com/ANFAIA/QUMO

Noise model (the paper's): each of the n inputs is the magic state |T> = (|0> + e^{i pi/4}|1>)/sqrt(2),
replaced by Z|T> with probability p, independently (the twirled input (1-p)|T><T| + p Z|T><T|Z).
Clifford operations, Pauli measurements and decoding are ideal. A fault pattern e has weight
p^|e| (1-p)^(n-|e|). The output error is conditional on acceptance:

    eps(p) = sum_e w_e h_e / sum_e w_e a_e,

with a_e the probability that pattern e is accepted and h_e = a_e (1 - F_e) the accepted weight
that is not the target state (F_e the fidelity of the accepted output with the target). Because
the noise is a mixture of Pauli patterns, summing over all 2^n patterns of state vectors is the
exact density-matrix answer, not a sample.

Every a_e and h_e below comes from QUMO's state-vector simulator: the noisy inputs are prepared
as circuits, projected onto the code by applying (I + g)/2 for each stabilizer g, and compared
with a target state built by a separate circuit. The paper's polynomials and counts are the
claims under test; no formula from the paper is used to compute the reference side.

Checks:
- 3 T -> CS|++>: acceptance 1/2 for every pattern; eps = 3p(1-p) (paper, Sec. S1) at four p;
  all six weight-1 and weight-2 patterns harmful, weight 3 harmless;
- 9-input T factory (the paper's eight signed stabilizers and logical operators): the
  generators commute, the logicals commute with them; acceptance equals the paper's
  D(p)/256 and the harmful weight equals N(p)/256 at four p (so eps = N/D), the fault-free
  acceptance is 1/128, the harmful-pattern counts by weight are 36, 45, 90, 75, 76, 15, 6, 1,
  and the p^2 coefficient of eps is 27/2;
- 8-input CCZ factory, two independent constructions:
  (a) circuit form: data a, b, c and one check qubit in |+>; a T or T-dagger on each of the
      eight parities u.(a,b,c) xor check, which multiplies to CCZ on the data and the identity
      on the check; the check is measured in X and must read +. Fault-free acceptance is 1.
  (b) the paper's single-code projection: inputs on cube vertices, stabilizers X on even and
      Y on odd vertices (sign +, which the rendered text does not print), Z on the faces
      v1=0, v2=0, v3=0, v1=1; fault-free acceptance 1/16 (paper's G_00).
  In both, every odd-weight pattern is rejected, all 28 weight-2 patterns are harmful, and the
  p^2 coefficient of eps is 28 (paper: eps = 28 p^2 + O(p^3)); both give the same harmful
  patterns, hence the same eps(p).

Not checked here: the paper's optimality claims (minimum input counts from code searches),
the larger factories (36-input cubic and 48-input quartic CCZ) and the 15-to-1 35 p^3 figure.
The p^2 coefficients are leading-order results; the exact eps(p) is checked only for the
3 T -> CS and 9-input T factories, the only ones whose exact polynomial the paper prints.

Run: .venv/bin/mojo run -I . examples/error_correction/magic_state_distillation_exact.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 180, "requires": [], "args": []}
"""

from std.math import pi, sqrt
from qumo.ir.circuit import Circuit
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, evolve_into
from qumo.testing.checks import CheckLog


comptime TOL = 1e-12


def _ps() -> List[Float64]:
    """Fault rates at which whole polynomials are compared."""
    return [0.001, 0.01, 0.05, 0.1]


def _popcount(x: Int) -> Int:
    var c = 0
    var y = x
    while y != 0:
        c += y & 1
        y >>= 1
    return c


def _weight(p: Float64, k: Int, n: Int) -> Float64:
    """Probability of one given fault pattern with k faults out of n inputs."""
    var w = 1.0
    for _ in range(k):
        w *= p
    for _ in range(n - k):
        w *= 1.0 - p
    return w


def _noisy_inputs(n: Int, mask: Int) raises -> StateVector:
    """|T>^n with Z on every input whose bit is set in mask."""
    var c = Circuit(n)
    for q in range(n):
        c.h(q)
        c.t(q)
        if ((mask >> q) & 1) == 1:
            c.z(q)
    return evolve(c)


def _apply_pauli(mut sv: StateVector, s: String) raises:
    """Apply a signed Pauli string ("+XZI..." or "-..."); character k acts on qubit k."""
    var bs = s.as_bytes()
    var neg = Int(bs[0]) == ord("-")
    var off = 1 if (Int(bs[0]) == ord("-") or Int(bs[0]) == ord("+")) else 0
    var c = Circuit(sv.num_qubits)
    for k in range(sv.num_qubits):
        var ch = Int(bs[k + off])
        if ch == ord("X"):
            c.x(k)
        elif ch == ord("Y"):
            c.y(k)
        elif ch == ord("Z"):
            c.z(k)
    evolve_into(sv, c, List[Float64]())
    if neg:
        for i in range(sv.size()):
            sv.re[i] = -sv.re[i]
            sv.im[i] = -sv.im[i]


def _project(mut sv: StateVector, gens: List[String]) raises:
    """sv <- prod_g (I + g)/2 sv (the generators commute, so the order does not matter)."""
    for g in gens:
        var t = sv.copy()
        _apply_pauli(t, g)
        for i in range(sv.size()):
            sv.re[i] = 0.5 * (sv.re[i] + t.re[i])
            sv.im[i] = 0.5 * (sv.im[i] + t.im[i])


def _expect(sv: StateVector, s: String) raises -> Float64:
    """Re <sv| P |sv> for a signed Pauli string."""
    var t = sv.copy()
    _apply_pauli(t, s)
    return sv.inner(t)[0]


def _commute(a: String, b: String) -> Bool:
    """Two Pauli strings commute iff they differ (both non-identity) on an even number of sites."""
    var x = a.as_bytes()
    var y = b.as_bytes()
    var oa = 1 if (Int(x[0]) == ord("-") or Int(x[0]) == ord("+")) else 0
    var ob = 1 if (Int(y[0]) == ord("-") or Int(y[0]) == ord("+")) else 0
    var n = a.byte_length() - oa
    var anti = 0
    for k in range(n):
        var p = Int(x[k + oa])
        var q = Int(y[k + ob])
        if p != ord("I") and q != ord("I") and p != q:
            anti += 1
    return anti % 2 == 0


struct Tally(Movable):
    """Accepted weight a_e and harmful weight h_e of every fault pattern."""

    var n: Int
    var acc: List[Float64]
    var harm: List[Float64]

    def __init__(out self, n: Int):
        self.n = n
        self.acc = List[Float64](length=1 << n, fill=0.0)
        self.harm = List[Float64](length=1 << n, fill=0.0)

    def accepted(self, p: Float64) -> Float64:
        var s = 0.0
        for e in range(1 << self.n):
            s += _weight(p, _popcount(e), self.n) * self.acc[e]
        return s

    def harmful(self, p: Float64) -> Float64:
        var s = 0.0
        for e in range(1 << self.n):
            s += _weight(p, _popcount(e), self.n) * self.harm[e]
        return s

    def eps(self, p: Float64) -> Float64:
        return self.harmful(p) / self.accepted(p)

    def harmful_count(self, k: Int) -> Int:
        var c = 0
        for e in range(1 << self.n):
            if _popcount(e) == k and self.harm[e] > TOL:
                c += 1
        return c

    def rejected_count(self, k: Int) -> Int:
        var c = 0
        for e in range(1 << self.n):
            if _popcount(e) == k and self.acc[e] < TOL:
                c += 1
        return c

    def harm_at_weight(self, k: Int) -> Float64:
        """Sum of h_e over |e| = k: the p^k coefficient of the harmful weight when lower weights are harmless."""
        var s = 0.0
        for e in range(1 << self.n):
            if _popcount(e) == k:
                s += self.harm[e]
        return s


def _three_t_to_cs(mut log: CheckLog) raises:
    var gens: List[String] = ["+ZZZ"]
    # Target CS|++> up to the Clifford S (x) S applied after a CS-dagger: amplitudes i^(y1 + y2 - y1 y2) / 2.
    var tc = Circuit(2)
    tc.h(0)
    tc.h(1)
    tc.cp(-pi / 2.0, 0, 1)
    tc.s(0)
    tc.s(1)
    var target = evolve(tc)
    var tal = Tally(3)
    for e in range(8):
        var sv = _noisy_inputs(3, e)
        _project(sv, gens)
        # Decode U|y1 y2> = |y1, y2, y1^y2>: two CNOTs leave qubit 2 in |0>.
        var dc = Circuit(3)
        dc.cx(0, 2)
        dc.cx(1, 2)
        evolve_into(sv, dc, List[Float64]())
        var out = StateVector(2)
        for i in range(4):
            out.re[i] = sv.re[i]
            out.im[i] = sv.im[i]
        var a = sv.norm_sq()
        tal.acc[e] = a
        tal.harm[e] = a - target.fidelity(out)
    var flat = True
    for e in range(8):
        if abs(tal.acc[e] - 0.5) > TOL:
            flat = False
    _ = log.check("3T-CS acceptance is 1/2 for every pattern", flat)
    for p in _ps():
        _ = log.close("3T-CS eps(" + String(p) + ") = 3p(1-p)", tal.eps(p), 3.0 * p * (1.0 - p), 1e-12)
    _ = log.check("3T-CS harmful patterns by weight 0,1,2,3 = 0,3,3,0",
                  tal.harmful_count(0) == 0 and tal.harmful_count(1) == 3 and tal.harmful_count(2) == 3 and tal.harmful_count(3) == 0)


def _nine_t(mut log: CheckLog) raises:
    var gens: List[String] = ["-XZIIZIIIX", "+XIZIZIIXI", "+ZXZZIIXZI", "+ZZXIZXIIZ",
                              "-YXYIXIIZZ", "-YYXXIIIII", "+IZIZZIZII", "+IIZZZZIII"]
    var lx = String("-IIIIZIZXI")
    var ly = String("-IIIXIXYII")
    var lz = String("-IIIXZXXXI")
    var ok = True
    for i in range(len(gens)):
        for j in range(len(gens)):
            if not _commute(gens[i], gens[j]):
                ok = False
        if not (_commute(gens[i], lx) and _commute(gens[i], ly) and _commute(gens[i], lz)):
            ok = False
    _ = log.check("9T generators commute and logicals commute with them", ok)
    _ = log.check("9T logical X and Y anticommute", not _commute(lx, ly))

    var tal = Tally(9)
    var r2 = 1.0 / sqrt(2.0)
    var max_purity_dev = 0.0
    for e in range(1 << 9):
        var sv = _noisy_inputs(9, e)
        _project(sv, gens)
        var a = sv.norm_sq()
        tal.acc[e] = a
        if a < TOL:
            continue
        var x = _expect(sv, lx) / a
        var y = _expect(sv, ly) / a
        var z = _expect(sv, lz) / a
        max_purity_dev = max(max_purity_dev, abs(x * x + y * y + z * z - 1.0))
        # Fidelity of the logical qubit with |T>, whose Bloch vector is (1/sqrt2, 1/sqrt2, 0).
        var f = 0.5 * (1.0 + (x + y) * r2)
        tal.harm[e] = a * (1.0 - f)
    _ = log.close("9T accepted logical state is pure for every accepted pattern", max_purity_dev, 0.0, 1e-9)
    _ = log.close("9T fault-free acceptance = 1/128", tal.acc[0], 1.0 / 128.0, 1e-14)
    for p in _ps():
        var d = 2.0 - 15.0 * p + 87.0 * p**2 - 256.0 * p**3 + 408.0 * p**4 - 336.0 * p**5 + 112.0 * p**6
        var nn = 27.0 * p**2 - 141.0 * p**3 + 324.0 * p**4 - 384.0 * p**5 + 224.0 * p**6 - 48.0 * p**7
        _ = log.close("9T acceptance(" + String(p) + ") = D(p)/256", tal.accepted(p), d / 256.0, 1e-14)
        _ = log.close("9T harmful weight(" + String(p) + ") = N(p)/256", tal.harmful(p), nn / 256.0, 1e-14)
        _ = log.close("9T eps(" + String(p) + ") = N(p)/D(p)", tal.eps(p), nn / d, 1e-12)
    var want: List[Int] = [0, 0, 36, 45, 90, 75, 76, 15, 6, 1]
    var got = String("")
    var same = True
    for k in range(10):
        var c = tal.harmful_count(k)
        got += String(c) + (" " if k < 9 else "")
        if c != want[k]:
            same = False
    _ = log.check("9T harmful patterns by weight 0..9 = 0 0 36 45 90 75 76 15 6 1", same, "got " + got)
    _ = log.close("9T p^2 coefficient of eps = 27/2", tal.harm_at_weight(2) / tal.acc[0], 13.5, 1e-9)


def _ccz_circuit(mut log: CheckLog) raises -> Tally:
    """Circuit form: T^(s_u) on parity u.(a,b,c) xor check, for all u in F2^3; Z after a faulty T."""
    var tc = Circuit(3)
    for q in range(3):
        tc.h(q)
    tc.ccz(0, 1, 2)
    var target = evolve(tc)
    var tal = Tally(8)
    for e in range(1 << 8):
        var c = Circuit(4)
        for q in range(4):
            c.h(q)
        for u in range(8):
            var wt = _popcount(u)
            for i in range(3):
                if ((u >> i) & 1) == 1:
                    c.cx(i, 3)
            # Phase exponents (units of pi/4) sum to 4abc: +1 on weight-1 and weight-3 parities,
            # -1 on weight-2 parities and on the bare check (u = 0), so the check sees the identity.
            if wt == 1 or wt == 3:
                c.t(3)
            else:
                c.tdg(3)
            if ((e >> u) & 1) == 1:
                c.z(3)
            for i in range(3):
                if ((u >> i) & 1) == 1:
                    c.cx(i, 3)
        c.h(3)
        var sv = evolve(c)
        var out = StateVector(3)
        for i in range(8):  # check qubit 3 reads 0 after H, i.e. X = +1
            out.re[i] = sv.re[i]
            out.im[i] = sv.im[i]
        var a = out.norm_sq()
        tal.acc[e] = a
        tal.harm[e] = a - target.fidelity(out) if a > TOL else 0.0
    _ = log.close("CCZ circuit form fault-free acceptance = 1", tal.acc[0], 1.0, 1e-12)
    _ = log.close("CCZ circuit form fault-free output is CCZ|+++>", tal.harm[0], 0.0, 1e-12)
    return tal^


def _ccz_projection(mut log: CheckLog) raises -> Tally:
    """The paper's single-code form; vertex v = (v1, v2, v3) is qubit 4 v1 + 2 v2 + v3."""
    var g1 = String("+")
    var f1 = String("+")
    var f2 = String("+")
    var f3 = String("+")
    var f4 = String("+")
    for i in range(8):
        var v1 = (i >> 2) & 1
        var v2 = (i >> 1) & 1
        var v3 = i & 1
        g1 += "X" if (v1 + v2 + v3) % 2 == 0 else "Y"
        f1 += "Z" if v1 == 0 else "I"
        f2 += "Z" if v2 == 0 else "I"
        f3 += "Z" if v3 == 0 else "I"
        f4 += "Z" if v1 == 1 else "I"
    var gens: List[String] = [g1, f1, f2, f3, f4]
    var ok = True
    for i in range(len(gens)):
        for j in range(len(gens)):
            if not _commute(gens[i], gens[j]):
                ok = False
    _ = log.check("CCZ projection generators commute", ok)
    var ideal = _noisy_inputs(8, 0)
    _project(ideal, gens)
    var g00 = ideal.norm_sq()
    _ = log.close("CCZ projection fault-free acceptance G00 = 1/16", g00, 1.0 / 16.0, 1e-12)
    var tal = Tally(8)
    for e in range(1 << 8):
        var sv = _noisy_inputs(8, e)
        _project(sv, gens)
        var a = sv.norm_sq()
        tal.acc[e] = a
        if a < TOL:
            continue
        # Harmless iff the accepted state is proportional to the fault-free one (the paper's
        # Gram condition G00 Gee = |G0e|^2); otherwise the accepted state differs from the target.
        var ov = ideal.fidelity(sv)
        tal.harm[e] = a - ov / g00
    return tal^


def _ccz_form(mut log: CheckLog, name: String, t: Tally):
    var odd_rejected = True
    for k in [1, 3, 5, 7]:
        if t.rejected_count(k) != _binom(8, k):
            odd_rejected = False
    _ = log.check("CCZ " + name + " form rejects every odd-weight pattern", odd_rejected)
    _ = log.check("CCZ " + name + " form all 28 weight-2 patterns harmful", t.harmful_count(2) == 28,
                  "got " + String(t.harmful_count(2)))
    _ = log.close("CCZ " + name + " form p^2 coefficient of eps = 28", t.harm_at_weight(2) / t.acc[0], 28.0, 1e-9)


def _ccz(mut log: CheckLog) raises:
    var c = _ccz_circuit(log)
    var pr = _ccz_projection(log)
    _ccz_form(log, "circuit", c)
    _ccz_form(log, "projection", pr)
    var same = True
    for e in range(1 << 8):
        if (c.harm[e] > TOL) != (pr.harm[e] > TOL) or (c.acc[e] > TOL) != (pr.acc[e] > TOL):
            same = False
    _ = log.check("CCZ both forms accept and harm the same patterns", same)
    for p in _ps():
        _ = log.close("CCZ eps(" + String(p) + ") equal in both forms", c.eps(p), pr.eps(p), 1e-12)
    var p = 1e-4
    _ = log.within("CCZ eps(1e-4) / p^2 near 28", c.eps(p) / (p * p), 27.99, 28.02)
    print("CCZ harmful patterns by weight 2, 4, 6:", c.harmful_count(2), c.harmful_count(4), c.harmful_count(6))
    print("CCZ eps at p = 0.001, 0.01:", c.eps(0.001), c.eps(0.01))


def _binom(n: Int, k: Int) -> Int:
    var r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def main() raises:
    var log = CheckLog("error_correction/magic_state_distillation_exact")
    _three_t_to_cs(log)
    _nine_t(log)
    _ccz(log)
    log.finish()
