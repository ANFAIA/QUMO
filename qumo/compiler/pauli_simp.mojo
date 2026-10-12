"""Pauli-gadget resynthesis (TKET `PauliSimp`, Qiskit Pauli-network synthesis).

Every unitary circuit over Clifford gates and rotations can be written as

    U = C_total . R_m ... R_1,        R_k = exp(-i theta_k / 2 P_k)

where the Pauli strings P_k are the rotations' Paulis conjugated back to the
*input* frame through the Clifford prefix before them (`C^dag P C`), and
C_total is the product of all Clifford gates - i.e. the original circuit with
its rotations deleted. The pass

  1. sweeps the circuit with a stabiliser tableau of C_prefix^-1 (bit-packed
     rows, O(n / 64) per Clifford gate) to extract the gadgets (rz, rx, ry, p,
     t, u, rzz/rxx/ryy, cp, crz/crx/cry, Clifford-angle rotations become
     Cliffords; ch / 3q gates are expanded first),
  2. merges gadgets with identical Pauli strings through commuting gadgets,
  3. re-synthesises each gadget as basis change + CX chain + rz + inverse,
     ordering every chain so that its prefix repeats the previous gadget's
     chain on qubits with the same Pauli letter: consecutive gadgets then share
     (and cancel) their ladders - the dominant saving on Trotterised
     Hamiltonians whose terms come sorted (HamLib, SparsePauliOp),
  4. appends the Clifford remainder, cleans both with cancel/commute and keeps
     the result only when it has fewer 2q gates than the cleaned original
     (otherwise the cleaned original is returned). The original's cleanup
     and the candidate's synthesis run on two threads; the gadget-weight
     reject (hopeless budget) is checked while extracting, so random
     circuits (QV) bail out after a few hundred gates.

Exact (global phase tracked). Skipped for symbolic circuits, conditioned
gates and circuits with rotations after a measurement / reset / barrier.
"""

from std.math import pi
from std.bit import pop_count
from qumo.ir.gates import Op, op_arity, op_is_unitary, op_num_params
from qumo.ir.param import Param
from qumo.ir.circuit import Circuit, Instruction
from qumo.compiler.common import (
    GateSet, HALF_PI, QUARTER_PI, TWO_PI, ANG_TOL, wrap_angle, shell_like,
)
from qumo.compiler.basis import expand_rule
from qumo.compiler.optimize import cancel_inverses, commute_rotations
from qumo.compiler.kak import consolidate_2q
from max.algorithm import parallelize


struct Tableau(Movable):
    """Rows M(X_i) (row 2i) and M(Z_i) (row 2i+1) of the map A -> C^dag A C,
    each a Pauli i^k prod X^x Z^z stored as bit-packed x / z words + k mod 4."""

    var n: Int
    var w: Int
    var x: List[UInt64]
    var z: List[UInt64]
    var k: List[Int]

    def __init__(out self, n: Int):
        self.n = n
        self.w = (n + 63) // 64
        self.x = List[UInt64](length=2 * n * self.w, fill=0)
        self.z = List[UInt64](length=2 * n * self.w, fill=0)
        self.k = List[Int](length=2 * n, fill=0)
        for i in range(n):
            var word = i // 64
            var bit = UInt64(1) << UInt64(i % 64)
            self.x[(2 * i) * self.w + word] = bit
            self.z[(2 * i + 1) * self.w + word] = bit

    def mul_row_into(self, row: Int, mut ax: List[UInt64], mut az: List[UInt64], mut ak: Int):
        """acc <- acc . row (XZ-form product, phase tracked)."""
        var base = row * self.w
        var c = 0
        for j in range(self.w):
            c += Int(pop_count(az[j] & self.x[base + j]))
        ak = (ak + self.k[row] + 2 * c) & 3
        for j in range(self.w):
            ax[j] ^= self.x[base + j]
            az[j] ^= self.z[base + j]

    def image(self, qa: Int, xa: Int, za: Int, qb: Int, xb: Int, zb: Int, k0: Int,
              mut ox: List[UInt64], mut oz: List[UInt64], mut ok: Int):
        """M(i^k0 X_a^xa Z_a^za X_b^xb Z_b^zb)."""
        for j in range(self.w):
            ox[j] = 0
            oz[j] = 0
        ok = k0 & 3
        if xa != 0:
            self.mul_row_into(2 * qa, ox, oz, ok)
        if za != 0:
            self.mul_row_into(2 * qa + 1, ox, oz, ok)
        if qb >= 0:
            if xb != 0:
                self.mul_row_into(2 * qb, ox, oz, ok)
            if zb != 0:
                self.mul_row_into(2 * qb + 1, ox, oz, ok)

    def set_row(mut self, row: Int, rx: List[UInt64], rz: List[UInt64], rk: Int):
        var base = row * self.w
        for j in range(self.w):
            self.x[base + j] = rx[j]
            self.z[base + j] = rz[j]
        self.k[row] = rk & 3


# G^dag A G tables (XZ form): per gate, for A in (X_a, Z_a, X_b, Z_b):
# (x_a, z_a, x_b, z_b, k). Verified numerically (see tests).
def _table(kind: Int, idx: Int, mut xa: Int, mut za: Int, mut xb: Int, mut zb: Int, mut k: Int) -> Bool:
    xa = 0; za = 0; xb = 0; zb = 0; k = 0
    if kind == Op.H:
        if idx == 0: za = 1
        else: xa = 1
        return True
    if kind == Op.S:
        if idx == 0: xa = 1; za = 1; k = 3
        else: za = 1
        return True
    if kind == Op.SDG:
        if idx == 0: xa = 1; za = 1; k = 1
        else: za = 1
        return True
    if kind == Op.X:
        if idx == 0: xa = 1
        else: za = 1; k = 2
        return True
    if kind == Op.Y:
        if idx == 0: xa = 1; k = 2
        else: za = 1; k = 2
        return True
    if kind == Op.Z:
        if idx == 0: xa = 1; k = 2
        else: za = 1
        return True
    if kind == Op.SX:
        if idx == 0: xa = 1
        else: xa = 1; za = 1; k = 1
        return True
    if kind == Op.SXDG:
        if idx == 0: xa = 1
        else: xa = 1; za = 1; k = 3
        return True
    if kind == Op.I:
        if idx == 0: xa = 1
        else: za = 1
        return True
    if kind == Op.CX:
        if idx == 0: xa = 1; xb = 1
        elif idx == 1: za = 1
        elif idx == 2: xb = 1
        else: za = 1; zb = 1
        return True
    if kind == Op.CZ:
        if idx == 0: xa = 1; zb = 1
        elif idx == 1: za = 1
        elif idx == 2: za = 1; xb = 1
        else: zb = 1
        return True
    if kind == Op.CY:
        if idx == 0: xa = 1; xb = 1; zb = 1; k = 1
        elif idx == 1: za = 1
        elif idx == 2: za = 1; xb = 1
        else: za = 1; zb = 1
        return True
    if kind == Op.SWAP:
        if idx == 0: xb = 1
        elif idx == 1: zb = 1
        elif idx == 2: xa = 1
        else: za = 1
        return True
    if kind == Op.ISWAP or kind == Op.ISWAPDG:
        var kk = 3 if kind == Op.ISWAP else 1
        if idx == 0: za = 1; xb = 1; zb = 1; k = kk
        elif idx == 1: zb = 1
        elif idx == 2: xa = 1; za = 1; zb = 1; k = kk
        else: za = 1
        return True
    if kind == Op.ECR:
        if idx == 0: xa = 1; za = 1; xb = 1; k = 3
        elif idx == 1: za = 1; k = 2
        elif idx == 2: xb = 1
        else: za = 1; xb = 1; zb = 1; k = 1
        return True
    return False


def _is_clifford_kind(k: Int) -> Bool:
    return (k == Op.H or k == Op.S or k == Op.SDG or k == Op.X or k == Op.Y or k == Op.Z
            or k == Op.SX or k == Op.SXDG or k == Op.I or k == Op.CX or k == Op.CZ
            or k == Op.CY or k == Op.SWAP or k == Op.ISWAP or k == Op.ISWAPDG or k == Op.ECR)


struct GadgetList(Movable):
    var n: Int
    var w: Int
    var x: List[UInt64]
    var z: List[UInt64]
    var angle: List[Float64]
    var alive: List[Bool]

    def __init__(out self, n: Int):
        self.n = n
        self.w = (n + 63) // 64
        self.x = List[UInt64]()
        self.z = List[UInt64]()
        self.angle = List[Float64]()
        self.alive = List[Bool]()

    def count(self) -> Int:
        return len(self.angle)

    def commute(self, a: Int, b: Int) -> Bool:
        var c = 0
        for j in range(self.w):
            c += Int(pop_count((self.x[a * self.w + j] & self.z[b * self.w + j]) ^ (self.z[a * self.w + j] & self.x[b * self.w + j])))
        return c % 2 == 0

    def same(self, a: Int, b: Int) -> Bool:
        for j in range(self.w):
            if self.x[a * self.w + j] != self.x[b * self.w + j] or self.z[a * self.w + j] != self.z[b * self.w + j]:
                return False
        return True

    def letter(self, g: Int, q: Int) -> Int:
        """0 none, 1 X, 2 Y, 3 Z."""
        var word = q // 64
        var bit = UInt64(1) << UInt64(q % 64)
        var xb = (self.x[g * self.w + word] & bit) != 0
        var zb = (self.z[g * self.w + word] & bit) != 0
        if xb and zb:
            return 2
        if xb:
            return 1
        if zb:
            return 3
        return 0


struct _Extract(Movable):
    var tab: Tableau
    var gad: GadgetList
    var rem: List[Instruction]
    var phase: Float64
    var ox: List[UInt64]
    var oz: List[UInt64]
    var tx: List[UInt64]
    var tz: List[UInt64]

    def __init__(out self, n: Int):
        self.tab = Tableau(n)
        self.gad = GadgetList(n)
        self.rem = List[Instruction]()
        self.phase = 0.0
        var w = (n + 63) // 64
        self.ox = List[UInt64](length=w, fill=0)
        self.oz = List[UInt64](length=w, fill=0)
        self.tx = List[UInt64](length=4 * w, fill=0)
        self.tz = List[UInt64](length=4 * w, fill=0)

    def clifford(mut self, kind: Int, qa: Int, qb: Int):
        """Append a Clifford gate: M <- M o (G^dag . G); also kept in the remainder."""
        var w = self.tab.w
        var nrows = 2 if qb < 0 else 4
        var ks = List[Int](length=4, fill=0)
        for idx in range(nrows):
            var xa = 0
            var za = 0
            var xb = 0
            var zb = 0
            var k = 0
            _ = _table(kind, idx, xa, za, xb, zb, k)
            var ok = 0
            self.tab.image(qa, xa, za, qb, xb, zb, k, self.ox, self.oz, ok)
            for j in range(w):
                self.tx[idx * w + j] = self.ox[j]
                self.tz[idx * w + j] = self.oz[j]
            ks[idx] = ok
        for idx in range(nrows):
            var q = qa if idx < 2 else qb
            var row = 2 * q + (idx % 2)
            var base = row * w
            for j in range(w):
                self.tab.x[base + j] = self.tx[idx * w + j]
                self.tab.z[base + j] = self.tz[idx * w + j]
            self.tab.k[row] = ks[idx]
        self.rem.append(Instruction.make(kind, qa, qb))

    def rotation(mut self, qa: Int, la: Int, qb: Int, lb: Int, theta: Float64):
        """exp(-i theta/2 P_a P_b) with letters 1=X, 2=Y, 3=Z (qb < 0: 1q)."""
        var k0 = 0
        var xa = 1 if la == 1 or la == 2 else 0
        var za = 1 if la == 3 or la == 2 else 0
        if la == 2:
            k0 += 1  # Y = i X Z
        var xb = 0
        var zb = 0
        if qb >= 0:
            xb = 1 if lb == 1 or lb == 2 else 0
            zb = 1 if lb == 3 or lb == 2 else 0
            if lb == 2:
                k0 += 1
        var ok = 0
        self.tab.image(qa, xa, za, qb, xb, zb, k0, self.ox, self.oz, ok)
        # Hermitian letter form: XZ = -iY, so sign = i^(k - #Y)
        var ny = 0
        for j in range(self.tab.w):
            ny += Int(pop_count(self.ox[j] & self.oz[j]))
        var s = (ok - ny) & 3
        var th = theta if s == 0 else -theta
        for j in range(self.tab.w):
            self.gad.x.append(self.ox[j])
            self.gad.z.append(self.oz[j])
        self.gad.angle.append(th)
        self.gad.alive.append(True)

    def rot_or_clifford(mut self, axis: Int, q: Int, theta: Float64):
        """Single-qubit rotation exp(-i theta/2 P_axis); Clifford angles go to the tableau."""
        var r = theta / HALF_PI
        var m = Int(round(r))
        if abs(r - Float64(m)) < 1e-12:
            var t = ((m % 4) + 4) % 4
            # R_axis(m pi/2) = e^{-i m pi/4} * (named Clifford)^m
            self.phase -= Float64(m) * QUARTER_PI
            for _ in range(t):
                if axis == 3:
                    self.clifford(Op.S, q, -1)
                elif axis == 1:
                    self.clifford(Op.SX, q, -1)
                else:
                    # S . SX . S^dag = e^{i pi/4} RY(pi/2)  (S X S^dag = Y)
                    self.clifford(Op.SDG, q, -1)
                    self.clifford(Op.SX, q, -1)
                    self.clifford(Op.S, q, -1)
            return
        self.rotation(q, axis, -1, 0, theta)


def _process(mut ex: _Extract, g: Instruction, depth: Int) -> Bool:
    """Feed one gate; False when unsupported."""
    var k = g.kind
    var a = g.q0
    var b = g.q1
    if _is_clifford_kind(k):
        if k != Op.I:
            ex.clifford(k, a, b if op_arity(k) == 2 else -1)
        return True
    if g.is_parametric():
        return False
    var v = g.p0.value()
    if k == Op.RZ:
        ex.rot_or_clifford(3, a, v)
        return True
    if k == Op.RX:
        ex.rot_or_clifford(1, a, v)
        return True
    if k == Op.RY:
        ex.rot_or_clifford(2, a, v)
        return True
    if k == Op.P or k == Op.T or k == Op.TDG:
        var lam = v
        if k == Op.T:
            lam = QUARTER_PI
        elif k == Op.TDG:
            lam = -QUARTER_PI
        ex.phase += lam / 2  # P(l) = e^{i l/2} RZ(l)
        ex.rot_or_clifford(3, a, lam)
        return True
    if k == Op.U:
        var th = g.p0.value()
        var ph = g.p1.value()
        var la = g.p2.value()
        ex.phase += (ph + la) / 2
        ex.rot_or_clifford(3, a, la)
        ex.rot_or_clifford(2, a, th)
        ex.rot_or_clifford(3, a, ph)
        return True
    if k == Op.RZZ or k == Op.RXX or k == Op.RYY:
        var l = 3 if k == Op.RZZ else (1 if k == Op.RXX else 2)
        ex.rotation(a, l, b, l, v)
        return True
    if k == Op.CP:
        ex.phase += v / 4
        ex.rot_or_clifford(3, a, v / 2)
        ex.rot_or_clifford(3, b, v / 2)
        ex.rotation(a, 3, b, 3, -v / 2)
        return True
    if k == Op.CRZ or k == Op.CRX or k == Op.CRY:
        var l = 3 if k == Op.CRZ else (1 if k == Op.CRX else 2)
        ex.rot_or_clifford(l, b, v / 2)
        ex.rotation(a, 3, b, l, -v / 2)
        return True
    if depth < 4 and op_arity(k) >= 2:
        var seq = List[Instruction]()
        var dropped = False
        var anyset = GateSet()
        var ph = expand_rule(g, anyset, seq, dropped)
        if len(seq) == 0:
            return False
        ex.phase += ph
        for s in seq:
            if not _process(ex, s, depth + 1):
                return False
        return True
    return False


def _chain_order(gl: GadgetList, gi: Int, prev: List[Int], nxt: Int) -> List[Int]:
    """Chain order of gadget gi: longest prefix of the previous chain whose qubits
    carry the same letter, then qubits shared with the next gadget, then the rest
    (high qubit first)."""
    var n = gl.n
    var order = List[Int]()
    var used = List[Bool](length=n, fill=False)
    for q in prev:
        var lp = gl.letter(gi, q)
        if lp == 0:
            break
        order.append(q)
        used[q] = True
    var q = n - 1
    while q >= 0:
        if not used[q] and gl.letter(gi, q) != 0 and nxt >= 0 and gl.letter(nxt, q) == gl.letter(gi, q):
            order.append(q)
            used[q] = True
        q -= 1
    q = n - 1
    while q >= 0:
        if not used[q] and gl.letter(gi, q) != 0:
            order.append(q)
            used[q] = True
        q -= 1
    return order^


def _letter_prefix_ok(gl: GadgetList, gi: Int, pj: Int, prev: List[Int]) -> List[Int]:
    """Prefix of the previous chain whose qubits have identical letters in gi and pj."""
    var out = List[Int]()
    for q in prev:
        var l = gl.letter(gi, q)
        if l == 0 or l != gl.letter(pj, q):
            break
        out.append(q)
    return out^


def _emit_gadget(gl: GadgetList, gi: Int, chain: List[Int], mut out: List[Instruction]):
    var kq = len(chain)
    for q in chain:
        var l = gl.letter(gi, q)
        if l == 1:
            out.append(Instruction.make(Op.H, q))
        elif l == 2:
            out.append(Instruction.make(Op.SDG, q))
            out.append(Instruction.make(Op.H, q))
    for i in range(kq - 1):
        out.append(Instruction.make(Op.CX, chain[i], chain[i + 1]))
    out.append(Instruction.make(Op.RZ, chain[kq - 1], p0=gl.angle[gi]))
    var i = kq - 2
    while i >= 0:
        out.append(Instruction.make(Op.CX, chain[i], chain[i + 1]))
        i -= 1
    for q in chain:
        var l = gl.letter(gi, q)
        if l == 1:
            out.append(Instruction.make(Op.H, q))
        elif l == 2:
            out.append(Instruction.make(Op.H, q))
            out.append(Instruction.make(Op.S, q))


def _synth_candidate(mut ex: _Extract, c: Circuit, window: Int, tail: List[Instruction],
                     mut cand: Circuit):
    """Steps 2-4: merge identical Paulis, chained gadget synthesis, cleanup."""
    var n = c.num_qubits
    var m = ex.gad.count()
    # 2. merge identical Paulis through commuting gadgets
    for i in range(m):
        var j = i - 1
        var steps = 0
        while j >= 0 and steps < window:
            if ex.gad.alive[j]:
                if ex.gad.same(j, i):
                    ex.gad.angle[j] += ex.gad.angle[i]
                    ex.gad.alive[i] = False
                    break
                if not ex.gad.commute(j, i):
                    break
                steps += 1
            j -= 1
    # 3. chained synthesis
    var ops = List[Instruction](capacity=len(c.ops))
    var prev = List[Int]()
    var prev_idx = -1
    var live = List[Int]()
    for i in range(m):
        if ex.gad.alive[i] and abs(wrap_angle(ex.gad.angle[i], 4 * pi)) > 1e-12:
            live.append(i)
    # greedy reordering of commuting gadgets: from the next `window` pending
    # gadgets pick one that commutes with everything it overtakes and shares
    # the longest same-letter prefix with the previous chain
    var taken = List[Bool](length=len(live), fill=False)
    var first = 0
    var nlive = len(live)
    var cand_pos = List[Int]()
    for _ in range(nlive):
        while first < nlive and taken[first]:
            first += 1
        cand_pos.clear()
        var p = first
        while p < nlive and len(cand_pos) < 12:
            if not taken[p]:
                cand_pos.append(p)
            p += 1
        var best_p = cand_pos[0]
        if prev_idx >= 0:
            var best_s = len(_letter_prefix_ok(ex.gad, live[best_p], prev_idx, prev))
            for j in range(1, len(cand_pos)):
                var gj = live[cand_pos[j]]
                var sc = len(_letter_prefix_ok(ex.gad, gj, prev_idx, prev))
                if sc <= best_s:
                    continue
                var movable = True
                for t2 in range(j):
                    if not ex.gad.commute(live[cand_pos[t2]], gj):
                        movable = False
                        break
                if movable:
                    best_s = sc
                    best_p = cand_pos[j]
        taken[best_p] = True
        var gi = live[best_p]
        var nxt = -1
        for q2 in range(first, nlive):
            if not taken[q2]:
                nxt = live[q2]
                break
        var pfx = List[Int]()
        if prev_idx >= 0:
            pfx = _letter_prefix_ok(ex.gad, gi, prev_idx, prev)
        var chain = _chain_order(ex.gad, gi, pfx, nxt)
        _emit_gadget(ex.gad, gi, chain, ops)
        prev = chain^
        prev_idx = gi
    for g in ex.rem:
        ops.append(g)
    for g in tail:
        ops.append(g)
    cand = shell_like(c, n)
    cand.ops = ops^
    cand.global_phase += ex.phase
    var anyset = GateSet()
    _ = cancel_inverses(cand, anyset, 64)
    _ = commute_rotations(cand, anyset, 64)
    _ = cancel_inverses(cand, anyset, 64)


def pauli_resynthesize(mut c: Circuit, window: Int = 64, kak_gs: GateSet = GateSet()) -> Int:
    """Gadget extraction + chained resynthesis; returns 2q gates saved.

    0 = the original structure is kept; when the extraction got as far as
    the comparison, the kept circuit is returned cleaned (cancel / commute,
    window 64), which the following fixpoint would compute anyway.
    `kak_gs` (target basis) prices both versions after KAK consolidation."""
    if c.is_parametric():
        return 0
    var n = c.num_qubits
    var ex = _Extract(n)
    var seen_nonunitary = False
    var tail = List[Instruction]()
    var rotations = 0
    # cheap reject, checked while extracting (the estimate only grows, so
    # this is the same decision as checking after the sweep): synthesising
    # every gadget costs 2(|support|-1) CX before any ladder sharing; give up
    # when that is hopeless (e.g. random circuits)
    var orig2 = c.num_2q_gates()
    var budget = 3 * orig2 + 16
    var est = 0
    for g in c.ops:
        var k = g.kind
        if not op_is_unitary(k) or g.cond >= 0:
            if g.cond >= 0:
                return 0
            seen_nonunitary = True
            tail.append(g)
            continue
        if seen_nonunitary:
            return 0  # gates after a measurement / reset / barrier: keep the circuit
        var before = ex.gad.count()
        if not _process(ex, g, 0):
            return 0
        var after = ex.gad.count()
        rotations += after - before
        for i in range(before, after):
            var wgt = 0
            for j in range(ex.gad.w):
                wgt += Int(pop_count(ex.gad.x[i * ex.gad.w + j] | ex.gad.z[i * ex.gad.w + j]))
            est += 2 * max(wgt - 1, 0)
        if est > budget:
            return 0
    if rotations == 0:
        return 0
    # base (original, cleaned) and candidate are independent: build them on
    # two threads; the comparison prices both after KAK consolidation (gadget
    # chains must beat what KAK already gets from the original structure,
    # e.g. rxx.ryy.rzz on a pair)
    var base = c.copy()
    var cand = Circuit(1)
    var nb = 0
    var nc = 0

    def job(t: Int) {mut base, mut cand, mut ex, mut nb, mut nc, imm c, imm window, imm tail, imm kak_gs}:
        var anyset = GateSet()
        var bb = 0
        if t == 0:
            _ = cancel_inverses(base, anyset, 64)
            _ = commute_rotations(base, anyset, 64)
            var bk = base.copy()
            nb = bk.num_2q_gates() - consolidate_2q(bk, kak_gs, bb, False)
        else:
            _synth_candidate(ex, c, window, tail, cand)
            nc = cand.num_2q_gates() - consolidate_2q(cand, kak_gs, bb, False)

    parallelize(job, 2)
    if nc < nb:
        c = cand^
        return nb - nc
    # keep the original structure, cleaned (cancel / commute, exact)
    c = base^
    return 0
