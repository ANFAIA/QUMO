"""Peephole and commutation-aware optimisation passes (all exact, phase tracked).

Passes (each returns the number of rewrites, mutates the circuit in place):
  * `remove_identities`  - Qiskit `RemoveIdentityEquivalent`: any bound 1q/2q
    gate within 1e-12 process infidelity of e^{i a} I (id, rz(4pi), rx(1e-9),
    cp(2pi), ...) is dropped and its trace phase moved into `global_phase`.
  * `cancel_inverses`    - TKET `RemoveRedundancies` / Qiskit
    `InverseCancellation` + `CommutativeInverseCancellation`: walks back along
    a wire through *commuting* gates (per-wire Pauli-axis rule) looking for the
    inverse (h h, cx cx, s sdg, swap swap, cz in either orientation, rx(a)
    rx(-a), u u^-1, ...). O(n * window) with the `WireList` frontier.
  * `merge_rotations`    - adjacent same-axis rotations (rz rz, rx rx, p p,
    t t -> s, rzz rzz, cp cp, crz crz, ...), symbolic params merge through
    `Param.try_add` when they share the symbol.
  * `commute_rotations`  - Qiskit `CommutativeCancellation`: like merge, but the
    partner may sit behind commuting gates (rz through cx controls / cz /
    diagonal gates, rx through cx targets, rzz through cz, ...).
  * `fuse_1q`            - Qiskit `Optimize1qGatesDecomposition`: fuse every
    maximal 1q run into one 2x2 matrix, resynthesise with `synth_1q` over the
    target basis, keep the original run unless the new one is shorter or the
    run contains non-basis gates.
  * `eject_z`            - Cirq `eject_z`: virtual-Z pushing. Z rotations are
    absorbed into a per-wire frame, commuted through diagonal / Z-type wires,
    swapped through SWAP, Pauli-Z frames propagated through CX targets, folded
    into U gates (U(t,f,l) rz(a) = e^{-ia/2} U(t,f,l+a)) and only materialised
    in front of the first non-commuting gate or at the end.
  * `elide_swaps`        - Qiskit `ElidePermutations`: SWAPs are removed by
    relabelling later operations; the permutation is returned for the final
    layout (only trailing swaps once a circuit is routed).
"""

from std.collections import Dict
from std.math import pi, atan2
from qumo.ir.gates import (
    Op, M2, matrix_2q, op_arity, op_num_params, op_is_unitary, op_inverse_kind, op_is_symmetric,
)
from qumo.ir.param import Param
from qumo.ir.circuit import Circuit, Instruction
from qumo.compiler.common import (
    GateSet, WireList, TWO_PI, FOUR_PI, HALF_PI, ANG_TOL, AX_Z, AX_ANY, AX_OTHER,
    is_plain_gate, is_bound, is_all_wire_op, commutes, same_qubit_set, compact,
    wrap_angle, near_mod, m2_identity_phase, inst_matrix_1q, rot_axis, rot_is_r,
    rot_angle, combine_rotations, emit_rotation, wire_axis,
)
from qumo.compiler.synth import synth_1q

comptime COMB_NONE = 0
comptime COMB_MERGED = 1
comptime COMB_IDENTITY = 2

comptime MODE_CANCEL = 0
comptime MODE_MERGE = 1
comptime MODE_COMMUTE = 2


# ----------------------------------------------------------------------------
# identities
# ----------------------------------------------------------------------------
def _period_rule(k: Int) -> Int:
    """0: none, 1: identity iff a = 0 mod 2pi (p/cp), 2: R-type (-I at 2pi), 3: identity iff 0 mod 4pi."""
    if k == Op.P or k == Op.CP:
        return 1
    if k == Op.RX or k == Op.RY or k == Op.RZ or k == Op.RXX or k == Op.RYY or k == Op.RZZ:
        return 2
    if k == Op.CRX or k == Op.CRY or k == Op.CRZ:
        return 3
    return 0


def gate_identity_phase(g: Instruction, mut phase: Float64) -> Bool:
    """True when the (bound) gate equals e^{i phase} I."""
    var k = g.kind
    if k == Op.I:
        phase = 0.0
        return True
    if not op_is_unitary(k) or g.is_parametric():
        return False
    var a = op_arity(k)
    if a == 1:
        return m2_identity_phase(inst_matrix_1q(g), phase)
    var rule = _period_rule(k)
    if rule == 0:
        return False
    var v = g.p0.value()
    if rule == 1:
        if near_mod(v, 0.0, TWO_PI):
            phase = 0.0
            return True
        return False
    if near_mod(v, 0.0, FOUR_PI):
        phase = 0.0
        return True
    if rule == 2 and near_mod(v, TWO_PI, FOUR_PI):
        phase = pi
        return True
    return False


comptime IDENTITY_INFIDELITY = 1e-12


def gate_near_identity(g: Instruction, mut phase: Float64) -> Bool:
    """True when 1 - |Tr U / d|^2 < 1e-12 (Qiskit RemoveIdentityEquivalent's
    default): numerically invisible rotations (|angle| < ~2e-6) are dropped and
    the trace phase goes to `global_phase`."""
    var k = g.kind
    if k == Op.I:
        phase = 0.0
        return True
    if not op_is_unitary(k) or g.is_parametric():
        return False
    var a = op_arity(k)
    var tr_r = 0.0
    var tr_i = 0.0
    var d = 2.0
    if a == 1:
        var m = inst_matrix_1q(g)
        tr_r = m.re[0] + m.re[3]
        tr_i = m.im[0] + m.im[3]
    elif a == 2:
        var m = matrix_2q(k, g.p0.value())
        for j in range(4):
            tr_r += m.re[j * 5]
            tr_i += m.im[j * 5]
        d = 4.0
    else:
        return False
    var f = (tr_r * tr_r + tr_i * tr_i) / (d * d)
    if 1.0 - f < IDENTITY_INFIDELITY:
        phase = atan2(tr_i, tr_r)
        return True
    return False


def remove_identities(mut c: Circuit) -> Int:
    """Drop gates equal (to 1e-12 process infidelity) to a global phase."""
    var n = len(c.ops)
    var alive = List[Bool](length=n, fill=True)
    var removed = 0
    for i in range(n):
        var g = c.ops[i]
        if not op_is_unitary(g.kind):
            continue
        var ph = 0.0
        if gate_near_identity(g, ph):
            alive[i] = False
            removed += 1
            if g.cond < 0:
                c.global_phase += ph
    if removed > 0:
        compact(c, alive)
    return removed


# ----------------------------------------------------------------------------
# pairwise combination
# ----------------------------------------------------------------------------
def is_inverse_pair(h: Instruction, g: Instruction, mut phase: Float64) -> Bool:
    """h followed by g equals e^{i phase} I (same operands up to gate symmetry)."""
    var a = op_arity(h.kind)
    if a != op_arity(g.kind) or not same_qubit_set(h, g):
        return False
    var np = op_num_params(h.kind)
    if np == 0 and op_num_params(g.kind) == 0:
        if g.kind == op_inverse_kind(h.kind):
            phase = 0.0
            return True
        if a == 1 and rot_axis(h.kind) == AX_OTHER and rot_axis(g.kind) == AX_OTHER:
            var m = inst_matrix_1q(g) @ inst_matrix_1q(h)
            return m2_identity_phase(m, phase)
        return False
    if a == 1:
        # rotations of the same axis are summed by combine_rotations; rotations
        # of different axes never multiply to the identity (fast reject)
        if rot_axis(h.kind) != AX_OTHER or rot_axis(g.kind) != AX_OTHER:
            if h.kind == g.kind and np == 1:
                var ok = False
                var s = h.p0.try_add(g.p0, ok)
                if ok and not s.is_symbolic() and abs(s.value()) < ANG_TOL:
                    phase = 0.0
                    return True
            return False
        if is_bound(h) and is_bound(g):
            var m = inst_matrix_1q(g) @ inst_matrix_1q(h)
            return m2_identity_phase(m, phase)
        if h.kind == g.kind and np == 1:
            var ok = False
            var s = h.p0.try_add(g.p0, ok)
            if ok and not s.is_symbolic() and abs(s.value()) < ANG_TOL:
                phase = 0.0
                return True
        return False
    if h.kind != g.kind or np != 1:
        return False
    var ok = False
    var s = h.p0.try_add(g.p0, ok)
    if not ok or s.is_symbolic():
        return False
    var tmp = Instruction.make(h.kind, h.q0, h.q1, h.q2, s)
    return gate_identity_phase(tmp, phase)


def combine(h: Instruction, g: Instruction, gs: GateSet, mode: Int,
            mut newh: Instruction, mut phase: Float64) -> Int:
    """Try to replace (h then g) by a single gate (COMB_MERGED) or nothing."""
    var ph = 0.0
    if is_inverse_pair(h, g, ph):
        phase += ph
        return COMB_IDENTITY
    if mode == MODE_CANCEL:
        return COMB_NONE
    var a = op_arity(h.kind)
    if a != op_arity(g.kind):
        return COMB_NONE
    if a == 1:
        if h.q0 != g.q0:
            return COMB_NONE
        var tmp = List[Instruction]()
        var p2 = 0.0
        if not combine_rotations(h, g, gs, tmp, p2):
            return COMB_NONE
        if len(tmp) == 0:
            phase += p2
            return COMB_IDENTITY
        if len(tmp) == 1:
            newh = tmp[0]
            phase += p2
            return COMB_MERGED
        return COMB_NONE
    if a != 2 or not same_qubit_set(h, g):
        return COMB_NONE
    var hk = h.kind
    var gk = g.kind
    if hk == gk and op_num_params(hk) == 1:
        var ok = False
        var s = h.p0.try_add(g.p0, ok)
        if not ok:
            return COMB_NONE
        var cand = Instruction.make(hk, h.q0, h.q1, -1, s)
        var p3 = 0.0
        if not s.is_symbolic() and gate_identity_phase(cand, p3):
            phase += p3
            return COMB_IDENTITY
        newh = cand
        return COMB_MERGED
    # cz == cp(pi)
    if (hk == Op.CZ or hk == Op.CP) and (gk == Op.CZ or gk == Op.CP):
        var ha = Param(pi) if hk == Op.CZ else h.p0
        var ga = Param(pi) if gk == Op.CZ else g.p0
        var ok = False
        var s = ha.try_add(ga, ok)
        if not ok:
            return COMB_NONE
        if not s.is_symbolic():
            var v = s.value()
            if near_mod(v, 0.0, TWO_PI):
                return COMB_IDENTITY
            if near_mod(v, pi, TWO_PI) and gs.has(Op.CZ):
                newh = Instruction.make(Op.CZ, h.q0, h.q1)
                return COMB_MERGED
        if gs.has(Op.CP):
            newh = Instruction.make(Op.CP, h.q0, h.q1, -1, s)
            return COMB_MERGED
    return COMB_NONE


def _clear_path(ops: List[Instruction], wl: WireList, j: Int, g: Instruction,
                allow_commute: Bool, window: Int) -> Bool:
    """Gate g can be moved back to be adjacent to op j on all of its wires."""
    var a = op_arity(g.kind)
    for s in range(1, a):
        var w = g.qubit(s)
        var k = wl.top[w]
        var steps = 0
        while k != j:
            if k < 0 or steps > window or not allow_commute:
                return False
            if not commutes(ops[k], g):
                return False
            k = wl.before(ops, k, w)
            steps += 1
    return True


def rewrite_scan(mut c: Circuit, gs: GateSet, mode: Int, window: Int = 32) -> Int:
    """Shared engine of cancel / merge / commute passes."""
    var n = len(c.ops)
    var alive = List[Bool](length=n, fill=True)
    var wl = WireList(n, c.num_qubits)
    var changes = 0
    var phase = 0.0
    var allow_commute = mode != MODE_MERGE
    for i in range(n):
        var g = c.ops[i]
        if is_all_wire_op(g):
            wl.cut_all()
            continue
        var a = op_arity(g.kind)
        if a == 0:
            continue
        if not is_plain_gate(g):
            wl.cut_op(g)
            continue
        var w0 = g.q0
        var j = wl.top[w0]
        var steps = 0
        var done = False
        while j >= 0 and steps <= window:
            var h = c.ops[j]
            if op_arity(h.kind) == a:
                var newh = h
                var ph = 0.0
                var r = combine(h, g, gs, mode, newh, ph)
                if r != COMB_NONE and _clear_path(c.ops, wl, j, g, allow_commute, window):
                    if r == COMB_IDENTITY:
                        wl.remove(c.ops, j)
                        alive[j] = False
                    else:
                        c.ops[j] = newh
                    alive[i] = False
                    phase += ph
                    changes += 1
                    done = True
                    break
            if not allow_commute:
                break
            if not commutes(h, g):
                break
            j = wl.before(c.ops, j, w0)
            steps += 1
        if not done:
            wl.push(c.ops, i)
    if changes > 0:
        compact(c, alive)
        c.global_phase += phase
    return changes


def cancel_inverses(mut c: Circuit, gs: GateSet, window: Int = 32) -> Int:
    return rewrite_scan(c, gs, MODE_CANCEL, window)


def merge_rotations(mut c: Circuit, gs: GateSet) -> Int:
    return rewrite_scan(c, gs, MODE_MERGE, 0)


def commute_rotations(mut c: Circuit, gs: GateSet, window: Int = 32) -> Int:
    return rewrite_scan(c, gs, MODE_COMMUTE, window)


# ----------------------------------------------------------------------------
# 1q run fusion
# ----------------------------------------------------------------------------
def _mat_key(m: M2) -> UInt64:
    var h: UInt64 = 0xCBF29CE484222325
    comptime for k in range(4):
        var a = Int(round(m.re[k] * 1e9))
        var b = Int(round(m.im[k] * 1e9))
        h = (h ^ UInt64(a)) * 0x100000001B3
        h = (h ^ UInt64(b)) * 0x100000001B3
    return h


def fuse_1q(mut c: Circuit, gs: GateSet) -> Int:
    """Resynthesise maximal runs of bound 1q gates; returns gates saved."""
    var n = len(c.ops)
    var nq = c.num_qubits
    var cur = List[Int](length=nq, fill=-1)
    var run_of = List[Int](length=n, fill=-1)
    var run_last = List[Int]()
    var run_count = List[Int]()
    var run_bad = List[Bool]()
    var run_mat = List[M2]()
    var run_q = List[Int]()
    for i in range(n):
        var g = c.ops[i]
        var k = g.kind
        if is_all_wire_op(g):
            for q in range(nq):
                cur[q] = -1
            continue
        var a = op_arity(k)
        if a == 0:
            continue
        if a == 1 and is_plain_gate(g) and is_bound(g):
            var q = g.q0
            var r = cur[q]
            if r < 0:
                r = len(run_last)
                run_last.append(i)
                run_count.append(0)
                run_bad.append(False)
                run_mat.append(M2.identity())
                run_q.append(q)
                cur[q] = r
            run_mat[r] = inst_matrix_1q(g) @ run_mat[r]
            run_count[r] += 1
            run_last[r] = i
            if not gs.has(k) or k == Op.I:
                run_bad[r] = True
            run_of[i] = r
            continue
        for s in range(a):
            var q = g.qubit(s)
            if q >= 0 and q < nq:
                cur[q] = -1
    var nr = len(run_last)
    var cache_head = Dict[UInt64, Int]()
    var cache_next = List[Int]()
    var cache_mat = List[M2]()
    var cache_ok = List[Bool]()
    var cache_start = List[Int]()
    var cache_len = List[Int]()
    var cache_phase = List[Float64]()
    var cache_ops = List[Instruction]()
    var rep_start = List[Int](length=nr, fill=-1)
    var rep_len = List[Int](length=nr, fill=0)
    var rep = List[Instruction]()
    var saved = 0
    var phase = 0.0
    for r in range(nr):
        var tmp = List[Instruction]()
        var ph = 0.0
        if run_count[r] == 1 and not run_bad[r]:
            # a single basis gate only improves when it is the identity
            if m2_identity_phase(run_mat[r], ph):
                rep_start[r] = len(rep)
                rep_len[r] = 0
                phase += ph
                saved += 1
            continue
        # memoised synthesis: structured circuits (Clifford, QFT, ...) repeat a
        # handful of distinct 1q runs; key = rounded matrix, verified exactly
        var m = run_mat[r]
        var key = _mat_key(m)
        var hit = -1
        var e = cache_head.get(key, -1)
        while e >= 0:
            var cm = cache_mat[e]
            var same = True
            comptime for k in range(4):
                if abs(cm.re[k] - m.re[k]) > 1e-12 or abs(cm.im[k] - m.im[k]) > 1e-12:
                    same = False
            if same:
                hit = e
                break
            e = cache_next[e]
        if hit >= 0:
            if not cache_ok[hit]:
                continue
            for k in range(cache_len[hit]):
                var t = cache_ops[cache_start[hit] + k]
                t.q0 = run_q[r]
                tmp.append(t)
            ph = cache_phase[hit]
        else:
            var ok = synth_1q(m, 0, gs, tmp, ph)
            var ce = len(cache_mat)
            cache_mat.append(m)
            cache_next.append(cache_head.get(key, -1))
            cache_head[key] = ce
            cache_ok.append(ok)
            cache_start.append(len(cache_ops))
            cache_len.append(len(tmp))
            cache_phase.append(ph)
            for t in tmp:
                cache_ops.append(t)
            if not ok:
                continue
            for ref t in tmp:
                t.q0 = run_q[r]
        if len(tmp) < run_count[r] or run_bad[r]:
            rep_start[r] = len(rep)
            rep_len[r] = len(tmp)
            for t in tmp:
                rep.append(t)
            phase += ph
            saved += run_count[r] - len(tmp)
    var out = List[Instruction](capacity=n)
    var changed = False
    for i in range(n):
        var r = run_of[i]
        if r >= 0 and rep_start[r] >= 0:
            changed = True
            if i == run_last[r]:
                for k in range(rep_len[r]):
                    out.append(rep[rep_start[r] + k])
            continue
        out.append(c.ops[i])
    if changed:
        c.ops = out^
        c.global_phase += phase
    return saved


# ----------------------------------------------------------------------------
# eject_z: virtual-Z frame pushing
# ----------------------------------------------------------------------------
def _flush_z(mut out: List[Instruction], mut pend: List[Float64], q: Int, gs: GateSet,
             mut phase: Float64):
    var a = pend[q]
    pend[q] = 0.0
    if abs(wrap_angle(a, FOUR_PI)) < ANG_TOL:
        return
    var tmp = List[Instruction]()
    if emit_rotation(AX_Z, True, Param(a), q, gs, tmp, phase):
        for t in tmp:
            out.append(t)
        return
    # basis without a usable Z rotation: fall back to rz (cannot happen when eject_z is gated)
    out.append(Instruction.make(Op.RZ, q, p0=a))


def eject_z(mut c: Circuit, gs: GateSet) -> Int:
    """Push bound Z rotations rightwards; returns the number of Z gates absorbed."""
    if not (gs.any or gs.has_z_rot() or gs.has(Op.U)):
        return 0
    var n = len(c.ops)
    var nq = c.num_qubits
    var pend = List[Float64](length=nq, fill=0.0)  # R-convention angle per wire
    var out = List[Instruction](capacity=n)
    var phase = 0.0
    for i in range(n):
        var g = c.ops[i]
        var k = g.kind
        if is_all_wire_op(g):
            for q in range(nq):
                _flush_z(out, pend, q, gs, phase)
            out.append(g)
            continue
        var a = op_arity(k)
        if a == 0:
            out.append(g)
            continue
        if not is_plain_gate(g):
            for s in range(a):
                var q = g.qubit(s)
                if q >= 0 and q < nq:
                    _flush_z(out, pend, q, gs, phase)
            out.append(g)
            continue
        if a == 1 and rot_axis(k) == AX_Z and is_bound(g):
            var ang = rot_angle(g).value()
            if not rot_is_r(k):
                phase += ang / 2  # P(a) = e^{i a/2} RZ(a)
            pend[g.q0] += ang
            continue
        if k == Op.I:
            continue
        if k == Op.U and is_bound(g) and abs(pend[g.q0]) > 0.0:
            # U(t, f, l) RZ(a) = e^{-i a/2} U(t, f, l + a)
            var pa = pend[g.q0]
            pend[g.q0] = 0.0
            phase -= pa / 2
            var ng = g
            ng.p2 = Param(wrap_angle(g.p2.value() + pa, TWO_PI))
            out.append(ng)
            continue
        if k == Op.SWAP:
            var t = pend[g.q0]
            pend[g.q0] = pend[g.q1]
            pend[g.q1] = t
            out.append(g)
            continue
        if k == Op.CX and near_mod(pend[g.q1], pi, TWO_PI):
            # (-iZ_t) then CX == CX then (-i Z_c Z_t) = CX then i RZ(pi)_c RZ(pi)_t:
            # the target frame rides through, the control frame gains pi.
            pend[g.q0] += pi
            phase += HALF_PI
            out.append(g)
            continue
        for s in range(a):
            var q = g.qubit(s)
            var ax = wire_axis(k, s)
            if ax == AX_Z or ax == AX_ANY:
                continue
            if abs(pend[q]) > 0.0:
                _flush_z(out, pend, q, gs, phase)
        out.append(g)
    for q in range(nq):
        if abs(pend[q]) > 0.0:
            _flush_z(out, pend, q, gs, phase)
    var saved = n - len(out)
    c.ops = out^
    c.global_phase += phase
    return saved


# ----------------------------------------------------------------------------
# elide_swaps: permutation elision (Qiskit ElidePermutations)
# ----------------------------------------------------------------------------
def elide_swaps(mut c: Circuit, trailing_only: Bool, mut wire_map: List[Int]) -> Int:
    """Delete SWAP gates by relabelling every later operation.

    `wire_map[w]` = wire that holds, at the end, what the original circuit
    would hold on wire w (the caller folds it into the final layout).
    `trailing_only`: only elide swaps not followed by a kept multi-qubit gate
    on their wires (keeps a routed circuit on coupling edges).
    """
    var n = len(c.ops)
    var nq = c.num_qubits
    var elide = List[Bool](length=n, fill=False)
    var count = 0
    if trailing_only:
        var touched = List[Bool](length=nq, fill=False)
        var i = n - 1
        while i >= 0:
            var g = c.ops[i]
            var a = op_arity(g.kind)
            if is_all_wire_op(g) or a == 0:
                i -= 1
                continue
            if g.kind == Op.SWAP and is_plain_gate(g) and not touched[g.q0] and not touched[g.q1]:
                elide[i] = True
                count += 1
            elif a >= 2:
                for s in range(a):
                    touched[g.qubit(s)] = True
            i -= 1
    else:
        for i in range(n):
            var g = c.ops[i]
            if g.kind == Op.SWAP and is_plain_gate(g):
                elide[i] = True
                count += 1
    wire_map = List[Int](length=nq, fill=0)
    for w in range(nq):
        wire_map[w] = w
    if count == 0:
        return 0
    var out = List[Instruction](capacity=n)
    for i in range(n):
        var g = c.ops[i]
        if elide[i]:
            var t = wire_map[g.q0]
            wire_map[g.q0] = wire_map[g.q1]
            wire_map[g.q1] = t
            continue
        var x = g
        var a = op_arity(g.kind)
        if a >= 1 and x.q0 >= 0:
            x.q0 = wire_map[x.q0]
        if a >= 2:
            x.q1 = wire_map[x.q1]
        if a >= 3:
            x.q2 = wire_map[x.q2]
        out.append(x)
    c.ops = out^
    return count
