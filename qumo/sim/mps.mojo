"""Matrix product state (MPS) simulator for noiseless circuits.

Scope: pure states of up to MPS_MAX_QUBITS qubits, 1-qubit and 2-qubit unitary gates on any
pair of qubits, and the 3-qubit gates (CCX, CCZ, CSWAP) through their exact decomposition into
1- and 2-qubit gates (`qumo.ir.gateset.expand_rule`, global phase kept). Results: amplitudes,
Pauli expectation values and seeded samples of all qubits in the computational basis.
Not supported (QM-E432): measurements followed by more gates on the measured qubit, resets,
noise channels and classically conditioned gates. Terminal measurements, barriers, ticks,
detectors and observables are ignored, as in `evolve`.

Representation. Site k holds qubit k (QUMO's order: qubit k is bit k of a basis index) as a
tensor A_k[l, s, r] of shape (D_{k-1}, 2, D_k), D_{-1} = D_{n-1} = 1, stored row-major
((l * 2 + s) * D_k + r) with split real and imaginary parts. The amplitude of the basis state
with qubit values b_0 ... b_{n-1} is the 1 x 1 product A_0[:, b_0, :] A_1[:, b_1, :] ...
The state is kept in mixed canonical form: sites left of the orthogonality centre are left
isometries, sites right of it right isometries.

Gates. A 1-qubit gate is contracted into its site. A 2-qubit gate on neighbours k, k + 1
moves the centre to k (exact SVDs, `qumo.core.svd`), contracts the two sites, applies the
4 x 4 matrix, and splits the (2 D_{k-1}) x (2 D_{k+1}) result by SVD. A gate on qubits further
apart is applied by swapping the farther qubit next to the nearer one with adjacent SWAP
gates, applying the gate and swapping back: the qubit order of the MPS never changes.

Truncation. At each split the singular values s_0 >= s_1 >= ... are kept while
i < max_bond, s_i > 0 and s_i > cutoff * s_0 (at least one is always kept). Exactly zero
singular values are dropped silently, since that changes nothing. The state is NOT
renormalized after a truncation, and two numbers are accumulated:

* `discarded_weight` = sum over truncations of w_i, the sum of the squared singular values
  dropped at that truncation. Because the environment of the split is orthonormal, each
  truncation is an orthogonal projection of the current state removing exactly w_i of its
  squared norm, so ||phi||^2 = 1 - discarded_weight (up to rounding), phi the final MPS.
* `error_bound` = sum over truncations of sqrt(w_i). With psi the exact final state
  (U_N ... U_1 |0...0>) and phi the final MPS, ||psi - phi||_2 <= error_bound: the error
  after a step is the earlier error, mapped by a unitary or a projection (neither increases
  norms), plus the new discarded component of norm sqrt(w_i) (triangle inequality).

Consequences, with delta = error_bound: an amplitude <x|phi> from `amplitude` differs from
<x|psi> by at most delta; for the normalized state phi^ = phi / ||phi|| (what `expectation`
and `sample` use), ||psi - phi^|| <= 2 delta, a Pauli expectation value differs from the exact
one by at most 4 delta per unit of |coefficient|, and the distribution of `sample` is within
total variation distance 2 delta of the exact one. The bounds do not include floating-point
rounding (about 1e-15 per gate on small states). With cutoff = 0 and a max_bond no smaller than
2^floor(n / 2), nothing is truncated and discarded_weight and error_bound are exactly 0.

Gate angles (after binding `params`) and the global phase must be finite (QM-E482).

Bounds (QM-E433): `sample` takes 1 to MPS_MAX_SHOTS shots and at most MPS_MAX_SAMPLE_BITS
(2^27) sampled bits per call (shots * num_qubits: every distinct outcome keeps an n-character
key); 1 <= num_qubits <= MPS_MAX_QUBITS, 1 <= max_bond <= MPS_MAX_BOND (512, so
that every SVD stays within the cap of `qumo.core.svd`), 0 <= cutoff < 1, and the largest
memory the tensors can reach, sum over sites of 32 D_{k-1} D_k bytes with
D_k = min(2^(k+1), 2^(n-k-1), max_bond), at most MPS_MAX_BYTES (4 GiB). A 2-qubit gate costs
one SVD of a (2 D) x (2 D) matrix, O(D^3) (on one core of an Apple M5: about 0.2 s at
D = 128, 1.6 s at D = 256), plus up to
2 |q0 - q1| - 2 more for the swaps of a non-adjacent gate.
"""

from std.math import sqrt, cos, sin, isnan, isinf
from std.collections import Dict
from qumo.core.rng import Rng
from qumo.core.svd import CMatrix, svd
from qumo.ir.circuit import Circuit, Instruction, check_finite_values
from qumo.ir.gates import (
    Op, op_arity, op_num_params, op_is_unitary, op_is_noise, op_is_annotation, op_name, M2, M4,
    matrix_1q, matrix_2q,
)
from qumo.ir.gateset import GateSet, expand_rule
from qumo.ir.pauli import PauliSum

comptime MPS_MAX_QUBITS = 4096
comptime MPS_MAX_BOND = 512
comptime MPS_MAX_BYTES = 1 << 32
comptime MPS_MAX_SHOTS = 10_000_000
comptime MPS_MAX_SAMPLE_BITS = 1 << 27  # shots * num_qubits: bounds the memory of the count keys


struct MPSSite(Copyable):
    """Tensor A[l, s, r] of one qubit, shape (dl, 2, dr), row-major, split re/im."""

    var dl: Int
    var dr: Int
    var re: List[Float64]
    var im: List[Float64]

    def __init__(out self, dl: Int, dr: Int):
        self.dl = dl
        self.dr = dr
        self.re = List[Float64](length=dl * 2 * dr, fill=0.0)
        self.im = List[Float64](length=dl * 2 * dr, fill=0.0)

    @always_inline
    def at(self, l: Int, s: Int, r: Int) -> Int:
        return (l * 2 + s) * self.dr + r


def _bond_cap(n: Int, k: Int, max_bond: Int) -> Int:
    """Largest dimension bond k (between sites k and k + 1) can reach."""
    var e = min(k + 1, n - k - 1)
    if e >= 20:
        return max_bond
    return min(1 << e, max_bond)


def mps_memory_bound(num_qubits: Int, max_bond: Int) -> Int:
    """Bytes the tensors can reach at most: sum over sites of 32 D_left D_right."""
    var total = 0
    var left = 1
    for k in range(num_qubits):
        var right = 1 if k == num_qubits - 1 else _bond_cap(num_qubits, k, max_bond)
        total += 32 * left * right
        if total > MPS_MAX_BYTES:
            return total
        left = right
    return total


struct MPS(Copyable):
    """Matrix product state of `num_qubits` qubits, initially |0...0> (see the module doc)."""

    var num_qubits: Int
    var max_bond: Int
    var cutoff: Float64
    var sites: List[MPSSite]
    var center: Int
    var discarded_weight: Float64  # sum of the squared singular values dropped
    var error_bound: Float64  # sum of sqrt(weight dropped per truncation) >= ||psi - phi||
    var truncations: Int  # splits that dropped a non-zero singular value
    var max_bond_seen: Int

    def __init__(out self, num_qubits: Int, max_bond: Int = 64, cutoff: Float64 = 1e-14) raises:
        if num_qubits < 1 or num_qubits > MPS_MAX_QUBITS:
            raise Error("QM-E433: an MPS needs 1 to " + String(MPS_MAX_QUBITS) + " qubits, got " + String(num_qubits) + ". Hint: split the register or reduce the circuit.")
        if max_bond < 1 or max_bond > MPS_MAX_BOND:
            raise Error("QM-E433: max_bond must be in [1, " + String(MPS_MAX_BOND) + "], got " + String(max_bond) + ". Hint: the SVD of a split is (2 max_bond) x (2 max_bond); larger bonds are not supported.")
        if not (cutoff >= 0.0 and cutoff < 1.0):
            raise Error("QM-E433: cutoff must be in [0, 1), got " + String(cutoff) + ". Hint: cutoff is relative to the largest singular value of a split; 0 keeps every non-zero one.")
        var mem = mps_memory_bound(num_qubits, max_bond)
        if mem > MPS_MAX_BYTES:
            raise Error("QM-E433: an MPS of " + String(num_qubits) + " qubits with max_bond " + String(max_bond) + " can need more than " + String(MPS_MAX_BYTES) + " bytes. Hint: lower max_bond or the number of qubits.")
        self.num_qubits = num_qubits
        self.max_bond = max_bond
        self.cutoff = cutoff
        self.sites = List[MPSSite](capacity=num_qubits)
        for _ in range(num_qubits):
            var s = MPSSite(1, 1)
            s.re[0] = 1.0
            self.sites.append(s^)
        self.center = 0
        self.discarded_weight = 0.0
        self.error_bound = 0.0
        self.truncations = 0
        self.max_bond_seen = 1

    # ------------------------------------------------------------------ checks
    def _check_qubit(self, q: Int) raises:
        if q < 0 or q >= self.num_qubits:
            raise Error("QM-E434: qubit " + String(q) + " is outside this " + String(self.num_qubits) + "-qubit MPS. Hint: qubits are 0 .. num_qubits - 1.")

    def bond_dims(self) -> List[Int]:
        """D_0 .. D_{n-2}: the dimension of each bond."""
        var out = List[Int]()
        for k in range(self.num_qubits - 1):
            out.append(self.sites[k].dr)
        return out^

    # ------------------------------------------------------------------ gauge
    def _shift_right(mut self) raises:
        """Move the centre from k to k + 1: A_k = U, A_{k+1} = S V^H A_{k+1}."""
        var k = self.center
        var dl = self.sites[k].dl
        var dr = self.sites[k].dr
        var m = CMatrix(dl * 2, dr)
        for i in range(dl * 2 * dr):
            m.re[i] = self.sites[k].re[i]
            m.im[i] = self.sites[k].im[i]
        var f = svd(m)
        var keep = 0
        for i in range(len(f.s)):
            if f.s[i] > 0.0:
                keep += 1
        keep = max(keep, 1)
        var a = MPSSite(dl, keep)
        for row in range(dl * 2):
            for c in range(keep):
                a.re[row * keep + c] = f.u.get_re(row, c)
                a.im[row * keep + c] = f.u.get_im(row, c)
        ref nxt = self.sites[k + 1]
        var b = MPSSite(keep, nxt.dr)
        for x in range(keep):  # (S V^H)[x, y] = s_x conj(V[y, x])
            for y in range(dr):
                var vr = f.s[x] * f.v.get_re(y, x)
                var vi = -f.s[x] * f.v.get_im(y, x)
                if vr == 0.0 and vi == 0.0:
                    continue
                for j in range(2 * nxt.dr):
                    var o = y * 2 * nxt.dr + j
                    var nr = nxt.re[o]
                    var ni = nxt.im[o]
                    b.re[x * 2 * nxt.dr + j] += vr * nr - vi * ni
                    b.im[x * 2 * nxt.dr + j] += vr * ni + vi * nr
        self.sites[k] = a^
        self.sites[k + 1] = b^
        self.center = k + 1

    def _shift_left(mut self) raises:
        """Move the centre from k to k - 1: A_k = V^H, A_{k-1} = A_{k-1} U S."""
        var k = self.center
        var dl = self.sites[k].dl
        var dr = self.sites[k].dr
        var m = CMatrix(dl, 2 * dr)
        for i in range(dl * 2 * dr):
            m.re[i] = self.sites[k].re[i]
            m.im[i] = self.sites[k].im[i]
        var f = svd(m)
        var keep = 0
        for i in range(len(f.s)):
            if f.s[i] > 0.0:
                keep += 1
        keep = max(keep, 1)
        var a = MPSSite(keep, dr)
        for x in range(keep):
            for j in range(2 * dr):
                a.re[x * 2 * dr + j] = f.v.get_re(j, x)
                a.im[x * 2 * dr + j] = -f.v.get_im(j, x)
        ref prv = self.sites[k - 1]
        var b = MPSSite(prv.dl, keep)
        for row in range(prv.dl * 2):
            for y in range(dl):
                var pr = prv.re[row * dl + y]
                var pim = prv.im[row * dl + y]
                if pr == 0.0 and pim == 0.0:
                    continue
                for x in range(keep):  # (U S)[y, x] = U[y, x] s_x
                    var ur = f.u.get_re(y, x) * f.s[x]
                    var ui = f.u.get_im(y, x) * f.s[x]
                    b.re[row * keep + x] += pr * ur - pim * ui
                    b.im[row * keep + x] += pr * ui + pim * ur
        self.sites[k] = a^
        self.sites[k - 1] = b^
        self.center = k - 1

    def move_center(mut self, k: Int) raises:
        """Bring the orthogonality centre to site k (exact; changes the gauge, not the state)."""
        self._check_qubit(k)
        while self.center < k:
            self._shift_right()
        while self.center > k:
            self._shift_left()

    # ------------------------------------------------------------------ gates
    def apply_1q(mut self, m: M2, q: Int) raises:
        self._check_qubit(q)
        ref st = self.sites[q]
        for l in range(st.dl):
            for r in range(st.dr):
                var i0 = st.at(l, 0, r)
                var i1 = st.at(l, 1, r)
                var ar = st.re[i0]
                var ai = st.im[i0]
                var br = st.re[i1]
                var bi = st.im[i1]
                st.re[i0] = m.re[0] * ar - m.im[0] * ai + m.re[1] * br - m.im[1] * bi
                st.im[i0] = m.re[0] * ai + m.im[0] * ar + m.re[1] * bi + m.im[1] * br
                st.re[i1] = m.re[2] * ar - m.im[2] * ai + m.re[3] * br - m.im[3] * bi
                st.im[i1] = m.re[2] * ai + m.im[2] * ar + m.re[3] * bi + m.im[3] * br

    def apply_2q(mut self, m: M4, q0: Int, q1: Int) raises:
        """Apply a 4 x 4 unitary (index b(q0) + 2 b(q1), as `qumo.ir.gates.M4`) to q0, q1."""
        self._check_qubit(q0)
        self._check_qubit(q1)
        if q0 == q1:
            raise Error("QM-E434: a 2-qubit gate got qubit " + String(q0) + " twice. Hint: pass two distinct qubits.")
        var lo = min(q0, q1)
        var hi = max(q0, q1)
        if hi == lo + 1:
            self._apply_adjacent(m, lo, q0 == lo)
            return
        var swap = matrix_2q(Op.SWAP)
        var j = hi
        while j > lo + 1:  # bring qubit hi to site lo + 1
            self._apply_adjacent(swap, j - 1, True)
            j -= 1
        self._apply_adjacent(m, lo, q0 == lo)
        j = lo + 2
        while j <= hi:  # and back: every other qubit returns to its site
            self._apply_adjacent(swap, j - 1, True)
            j += 1

    def _apply_adjacent(mut self, m: M4, k: Int, q0_left: Bool) raises:
        """Gate on sites k, k + 1; q0 (bit 0 of the 4 x 4 index) is site k iff q0_left."""
        self.move_center(k)
        var dl = self.sites[k].dl
        var dm = self.sites[k].dr
        var dr = self.sites[k + 1].dr
        # theta[l, s1, s2, r] = sum_m A_k[l, s1, m] A_{k+1}[m, s2, r]
        var tr = List[Float64](length=dl * 4 * dr, fill=0.0)
        var ti = List[Float64](length=dl * 4 * dr, fill=0.0)
        ref a = self.sites[k]
        ref b = self.sites[k + 1]
        for l in range(dl):
            for s1 in range(2):
                for x in range(dm):
                    var ar = a.re[(l * 2 + s1) * dm + x]
                    var ai = a.im[(l * 2 + s1) * dm + x]
                    if ar == 0.0 and ai == 0.0:
                        continue
                    for j in range(2 * dr):  # j = s2 * dr + r
                        var br = b.re[x * 2 * dr + j]
                        var bi = b.im[x * 2 * dr + j]
                        var o = (l * 2 + s1) * 2 * dr + j
                        tr[o] += ar * br - ai * bi
                        ti[o] += ar * bi + ai * br
        # Gate: local index of (s1, s2) in the 4 x 4 matrix.
        var idx = List[Int](length=4, fill=0)
        for s1 in range(2):
            for s2 in range(2):
                idx[s1 * 2 + s2] = (s1 + 2 * s2) if q0_left else (s2 + 2 * s1)
        var th = CMatrix(dl * 2, 2 * dr)
        for l in range(dl):
            for r in range(dr):
                var vr = List[Float64](length=4, fill=0.0)
                var vi = List[Float64](length=4, fill=0.0)
                for s1 in range(2):
                    for s2 in range(2):
                        var o = ((l * 2 + s1) * 2 + s2) * dr + r
                        vr[s1 * 2 + s2] = tr[o]
                        vi[s1 * 2 + s2] = ti[o]
                for t in range(4):
                    var row = idx[t]
                    var sr = 0.0
                    var si = 0.0
                    for u in range(4):
                        var col = idx[u]
                        var gr = m.re[row * 4 + col]
                        var gi = m.im[row * 4 + col]
                        sr += gr * vr[u] - gi * vi[u]
                        si += gr * vi[u] + gi * vr[u]
                    var t1 = t >> 1
                    var t2 = t & 1
                    th.set(l * 2 + t1, t2 * dr + r, sr, si)
        var f = svd(th)
        var smax = f.s[0] if len(f.s) > 0 else 0.0
        var keep = 0
        for i in range(len(f.s)):
            if i < self.max_bond and f.s[i] > 0.0 and f.s[i] > self.cutoff * smax:
                keep += 1
            else:
                break
        keep = max(keep, 1)
        var w = 0.0
        for i in range(keep, len(f.s)):
            w += f.s[i] * f.s[i]
        if w > 0.0:
            self.discarded_weight += w
            self.error_bound += sqrt(w)
            self.truncations += 1
        var na = MPSSite(dl, keep)
        for row in range(dl * 2):
            for c in range(keep):
                na.re[row * keep + c] = f.u.get_re(row, c)
                na.im[row * keep + c] = f.u.get_im(row, c)
        var nb = MPSSite(keep, dr)
        for x in range(keep):
            for j in range(2 * dr):
                nb.re[x * 2 * dr + j] = f.s[x] * f.v.get_re(j, x)
                nb.im[x * 2 * dr + j] = -f.s[x] * f.v.get_im(j, x)
        self.sites[k] = na^
        self.sites[k + 1] = nb^
        self.center = k + 1
        self.max_bond_seen = max(self.max_bond_seen, keep)

    def apply_global_phase(mut self, phase: Float64):
        if phase == 0.0:
            return
        var c = cos(phase)
        var s = sin(phase)
        ref st = self.sites[self.center]
        for i in range(len(st.re)):
            var r = st.re[i]
            var im = st.im[i]
            st.re[i] = c * r - s * im
            st.im[i] = c * im + s * r

    # ------------------------------------------------------------------ queries
    def norm_sq(self) -> Float64:
        """||phi||^2 = 1 - discarded_weight (up to rounding): the centre tensor's norm."""
        ref st = self.sites[self.center]
        var t = 0.0
        for i in range(len(st.re)):
            t += st.re[i] * st.re[i] + st.im[i] * st.im[i]
        return t

    def amplitude(self, bits: List[Int]) raises -> Tuple[Float64, Float64]:
        """<b|phi> for qubit values bits[q] (q = 0 .. n-1), not renormalized."""
        if len(bits) != self.num_qubits:
            raise Error("QM-E434: amplitude needs one bit per qubit (" + String(self.num_qubits) + "), got " + String(len(bits)) + ". Hint: bits[q] is the value of qubit q.")
        var vr = List[Float64](length=1, fill=1.0)
        var vi = List[Float64](length=1, fill=0.0)
        for k in range(self.num_qubits):
            var b = bits[k]
            if b != 0 and b != 1:
                raise Error("QM-E434: bit " + String(k) + " is " + String(b) + ". Hint: bits are 0 or 1.")
            ref st = self.sites[k]
            var nr = List[Float64](length=st.dr, fill=0.0)
            var ni = List[Float64](length=st.dr, fill=0.0)
            for l in range(st.dl):
                var xr = vr[l]
                var xi = vi[l]
                for r in range(st.dr):
                    var o = st.at(l, b, r)
                    nr[r] += xr * st.re[o] - xi * st.im[o]
                    ni[r] += xr * st.im[o] + xi * st.re[o]
            vr = nr^
            vi = ni^
        return (vr[0], vi[0])

    def amplitude_index(self, index: Int) raises -> Tuple[Float64, Float64]:
        """<index|phi>, little-endian (bit k of index = qubit k); needs n <= 62."""
        if self.num_qubits > 62 or index < 0 or index >= (1 << self.num_qubits):
            raise Error("QM-E434: basis index " + String(index) + " is not valid for " + String(self.num_qubits) + " qubits (0 <= index < 2^n, n <= 62). Hint: use amplitude(bits) for wider states.")
        var bits = List[Int](capacity=self.num_qubits)
        for k in range(self.num_qubits):
            bits.append((index >> k) & 1)
        return self.amplitude(bits)

    def _pauli_value(self, ops: List[Int]) -> Tuple[Float64, Float64]:
        """<phi|P|phi> for P = tensor of ops[q] (0 I, 1 X, 2 Y, 3 Z); not normalized."""
        var lr = List[Float64](length=1, fill=1.0)  # L[l, l'] row-major, dl x dl
        var li = List[Float64](length=1, fill=0.0)
        for k in range(self.num_qubits):
            ref st = self.sites[k]
            var dl = st.dl
            var dr = st.dr
            var p = ops[k]
            # B[l', s, r'] = sum_s' P[s, s'] A[l', s', r']
            var br = List[Float64](length=dl * 2 * dr, fill=0.0)
            var bi = List[Float64](length=dl * 2 * dr, fill=0.0)
            for l in range(dl):
                for r in range(dr):
                    var a0r = st.re[st.at(l, 0, r)]
                    var a0i = st.im[st.at(l, 0, r)]
                    var a1r = st.re[st.at(l, 1, r)]
                    var a1i = st.im[st.at(l, 1, r)]
                    var o0 = st.at(l, 0, r)
                    var o1 = st.at(l, 1, r)
                    if p == 0:
                        br[o0] = a0r; bi[o0] = a0i; br[o1] = a1r; bi[o1] = a1i
                    elif p == 1:
                        br[o0] = a1r; bi[o0] = a1i; br[o1] = a0r; bi[o1] = a0i
                    elif p == 2:  # Y = [[0, -i], [i, 0]]
                        br[o0] = a1i; bi[o0] = -a1r; br[o1] = -a0i; bi[o1] = a0r
                    else:
                        br[o0] = a0r; bi[o0] = a0i; br[o1] = -a1r; bi[o1] = -a1i
            # T[l, s, r'] = sum_l' L[l, l'] B[l', s, r']
            var tr = List[Float64](length=dl * 2 * dr, fill=0.0)
            var ti = List[Float64](length=dl * 2 * dr, fill=0.0)
            for l in range(dl):
                for lp in range(dl):
                    var xr = lr[l * dl + lp]
                    var xi = li[l * dl + lp]
                    if xr == 0.0 and xi == 0.0:
                        continue
                    for j in range(2 * dr):
                        var yr = br[lp * 2 * dr + j]
                        var yi = bi[lp * 2 * dr + j]
                        tr[l * 2 * dr + j] += xr * yr - xi * yi
                        ti[l * 2 * dr + j] += xr * yi + xi * yr
            # L'[r, r'] = sum_{l, s} conj(A[l, s, r]) T[l, s, r']
            var nr = List[Float64](length=dr * dr, fill=0.0)
            var ni = List[Float64](length=dr * dr, fill=0.0)
            for ls in range(dl * 2):
                for r in range(dr):
                    var ar = st.re[ls * dr + r]
                    var ai = -st.im[ls * dr + r]
                    if ar == 0.0 and ai == 0.0:
                        continue
                    for rp in range(dr):
                        var yr = tr[ls * dr + rp]
                        var yi = ti[ls * dr + rp]
                        nr[r * dr + rp] += ar * yr - ai * yi
                        ni[r * dr + rp] += ar * yi + ai * yr
            lr = nr^
            li = ni^
        return (lr[0], li[0])

    def expectation_ops(self, ops: List[Int]) raises -> Float64:
        """<P> in the normalized state, P = tensor of ops[q] (0 I, 1 X, 2 Y, 3 Z), any width."""
        if len(ops) != self.num_qubits:
            raise Error("QM-E434: expectation_ops needs one Pauli per qubit (" + String(self.num_qubits) + "), got " + String(len(ops)) + ". Hint: ops[q] is 0 (I), 1 (X), 2 (Y) or 3 (Z) for qubit q.")
        for o in ops:
            if o < 0 or o > 3:
                raise Error("QM-E434: Pauli code " + String(o) + " is not 0 (I), 1 (X), 2 (Y) or 3 (Z). Hint: ops[q] is one of 0, 1, 2, 3.")
        var nn = self.norm_sq()
        if nn <= 0.0:
            raise Error("QM-E434: the MPS has zero norm (everything was truncated). Hint: raise max_bond or lower cutoff.")
        var v = self._pauli_value(ops)
        return v[0] / nn

    def expectation_label(self, label: String) raises -> Float64:
        """<P> for a Pauli label of length n, Qiskit order (leftmost character = qubit n - 1)."""
        var n = label.byte_length()
        if n != self.num_qubits:
            raise Error("QM-E434: Pauli label has " + String(n) + " characters for a " + String(self.num_qubits) + "-qubit MPS. Hint: one character (I, X, Y, Z) per qubit, the leftmost for qubit n - 1.")
        var bs = label.as_bytes()
        var ops = List[Int](length=n, fill=0)
        for i in range(n):
            var ch = Int(bs[i])
            var q = n - 1 - i
            if ch == ord("I"):
                ops[q] = 0
            elif ch == ord("X"):
                ops[q] = 1
            elif ch == ord("Y"):
                ops[q] = 2
            elif ch == ord("Z"):
                ops[q] = 3
            else:
                raise Error("QM-E434: Pauli label character '" + String(label[byte=i:i + 1]) + "' is not I, X, Y or Z. Hint: use upper-case I, X, Y, Z.")
        return self.expectation_ops(ops)

    def expectation(self, h: PauliSum) raises -> Float64:
        """<H> in the normalized state for a Pauli sum (its terms address qubits < 64)."""
        if h.num_qubits() > self.num_qubits:
            raise Error("QM-E405: observable acts on qubit " + String(h.num_qubits() - 1) + " but the MPS has " + String(self.num_qubits) + " qubits. Hint: check Pauli indices (0-based, qubit 0 = least significant).")
        var total = 0.0
        for t in h.terms:
            var ops = List[Int](length=self.num_qubits, fill=0)
            for q in range(min(self.num_qubits, 64)):
                var b = UInt64(1) << UInt64(q)
                var x = (t.x & b) != 0
                var z = (t.z & b) != 0
                ops[q] = 2 if (x and z) else (1 if x else (3 if z else 0))
            total += t.coeff * self.expectation_ops(ops)
        return total

    def sample(mut self, shots: Int, seed: UInt64) raises -> Dict[String, Int]:
        """Counts of `shots` samples of every qubit, from the normalized state. Keys are
        Qiskit-style (leftmost character = qubit n - 1). Moves the centre to site 0."""
        if shots < 1 or shots > MPS_MAX_SHOTS:
            raise Error("QM-E433: shots must be in [1, " + String(MPS_MAX_SHOTS) + "], got " + String(shots) + ". Hint: split the run into several calls with different seeds.")
        if shots > MPS_MAX_SAMPLE_BITS // self.num_qubits:
            raise Error("QM-E433: " + String(shots) + " shots of " + String(self.num_qubits) + " qubits is over the limit of 2^27 sampled bits (shots * num_qubits): each distinct outcome keeps an n-character key. Hint: at most " + String(MPS_MAX_SAMPLE_BITS // self.num_qubits) + " shots per call; split the run into calls with different seeds.")
        self.move_center(0)
        if self.norm_sq() <= 0.0:
            raise Error("QM-E434: the MPS has zero norm (everything was truncated). Hint: raise max_bond or lower cutoff.")
        var rng = Rng(seed)
        var counts = Dict[String, Int]()
        var n = self.num_qubits
        var chars = List[UInt8](length=n, fill=UInt8(ord("0")))
        for _ in range(shots):
            var vr = List[Float64](length=1, fill=1.0)
            var vi = List[Float64](length=1, fill=0.0)
            for k in range(n):
                ref st = self.sites[k]
                var w0r = List[Float64](length=st.dr, fill=0.0)
                var w0i = List[Float64](length=st.dr, fill=0.0)
                var w1r = List[Float64](length=st.dr, fill=0.0)
                var w1i = List[Float64](length=st.dr, fill=0.0)
                for l in range(st.dl):
                    var xr = vr[l]
                    var xi = vi[l]
                    if xr == 0.0 and xi == 0.0:
                        continue
                    for r in range(st.dr):
                        var o0 = st.at(l, 0, r)
                        var o1 = st.at(l, 1, r)
                        w0r[r] += xr * st.re[o0] - xi * st.im[o0]
                        w0i[r] += xr * st.im[o0] + xi * st.re[o0]
                        w1r[r] += xr * st.re[o1] - xi * st.im[o1]
                        w1i[r] += xr * st.im[o1] + xi * st.re[o1]
                var p0 = 0.0
                var p1 = 0.0
                for r in range(st.dr):
                    p0 += w0r[r] * w0r[r] + w0i[r] * w0i[r]
                    p1 += w1r[r] * w1r[r] + w1i[r] * w1i[r]
                var bit = 0 if rng.uniform() * (p0 + p1) < p0 else 1
                var inv = 1.0 / sqrt(p1 if bit == 1 else p0)
                if bit == 0:
                    for r in range(st.dr):
                        w0r[r] *= inv
                        w0i[r] *= inv
                    vr = w0r^
                    vi = w0i^
                else:
                    for r in range(st.dr):
                        w1r[r] *= inv
                        w1i[r] *= inv
                    vr = w1r^
                    vi = w1i^
                chars[n - 1 - k] = UInt8(ord("1")) if bit == 1 else UInt8(ord("0"))
            var key = String(unsafe_from_utf8=Span(chars))
            counts[key] = counts.get(key, 0) + 1
        return counts^


def check_mps_circuit(circuit: Circuit) raises:
    """Raise QM-E432 for what the MPS simulator does not run (see the module doc)."""
    var measured = List[Bool](length=circuit.num_qubits, fill=False)
    for i in range(len(circuit.ops)):
        ref g = circuit.ops[i]
        var k = g.kind
        var what = String("")
        if g.cond >= 0 or g.cond_reg >= 0:
            what = "a classically conditioned '" + op_name(k) + "'"
        elif k == Op.RESET:
            what = "a reset"
        elif op_is_noise(k):
            what = "the noise channel '" + op_name(k) + "'"
        elif k == Op.MEASURE:
            if g.q0 >= 0 and g.q0 < circuit.num_qubits:
                measured[g.q0] = True
            continue
        elif op_is_unitary(k):
            for j in range(op_arity(k)):
                var q = g.qubit(j)
                if q >= 0 and q < circuit.num_qubits and measured[q]:
                    what = "a gate on qubit " + String(q) + " after it was measured (mid-circuit measurement)"
                    break
        elif not op_is_annotation(k):
            what = "the operation '" + op_name(k) + "'"
        if what.byte_length() > 0:
            raise Error("QM-E432: the MPS simulator cannot take a circuit with " + what + " (instruction #" + String(i) + "): it simulates one noiseless pure state. Hint: use sample() (trajectories) or the density-matrix simulator; terminal measurements are fine.")


def _apply_instruction(mut st: MPS, g: Instruction, params: List[Float64], depth: Int) raises:
    var k = g.kind
    var a = op_arity(k)
    if a == 1:
        if k != Op.I:
            st.apply_1q(matrix_1q(k, g.p0.resolve(params), g.p1.resolve(params), g.p2.resolve(params)), g.q0)
        return
    if a == 2:
        st.apply_2q(matrix_2q(k, g.p0.resolve(params)), g.q0, g.q1)
        return
    # 3-qubit gates: the exact rewrite of the IR (CSWAP -> CX, CCX, CX -> ...).
    var seq = List[Instruction]()
    var dropped = False
    var phase = expand_rule(g, GateSet(), seq, dropped)
    if len(seq) == 0 or depth > 3:
        raise Error("QM-E432: the MPS simulator has no decomposition for '" + op_name(k) + "'. Hint: rewrite it into 1- and 2-qubit gates.")
    for s in seq:
        _apply_instruction(st, s, params, depth + 1)
    st.apply_global_phase(phase)


def mps_evolve(circuit: Circuit, max_bond: Int = 64, cutoff: Float64 = 1e-14,
               params: List[Float64] = List[Float64]()) raises -> MPS:
    """Run a noiseless circuit on an MPS from |0...0> (see the module doc for the options,
    the truncation rule and what `error_bound` bounds)."""
    circuit.validate()
    check_mps_circuit(circuit)
    if len(params) != circuit.num_params():
        raise Error("QM-E101: circuit has " + String(circuit.num_params()) + " parameters " + circuit.param_names_str() + " but " + String(len(params)) + " values were given.")
    check_finite_values(params)
    if isnan(circuit.global_phase) or isinf(circuit.global_phase):
        raise Error("QM-E482: the circuit's global phase is " + String(circuit.global_phase) + ". Hint: a global phase must be a finite number.")
    for i in range(len(circuit.ops)):
        ref g = circuit.ops[i]
        if not op_is_unitary(g.kind):
            continue
        for j in range(op_num_params(g.kind)):
            var v = g.p0.resolve(params) if j == 0 else (g.p1.resolve(params) if j == 1 else g.p2.resolve(params))
            if isnan(v) or isinf(v):
                raise Error("QM-E482: gate '" + op_name(g.kind) + "' (instruction #" + String(i) + ") has angle " + String(v) + ". Hint: gate angles must be finite numbers; a NaN would run on as a NaN state.")
    var st = MPS(circuit.num_qubits, max_bond, cutoff)
    for g in circuit.ops:
        if op_is_unitary(g.kind):
            _apply_instruction(st, g, params, 0)
    st.apply_global_phase(circuit.global_phase)
    return st^
