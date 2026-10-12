"""Basis-aware single-qubit synthesis (exact, global phase tracked).

Best-of synthesis:
  * Qiskit `Optimize1qGatesDecomposition` / `OneQubitEulerDecomposer`: one
    Euler decomposition per basis family - U, ZYZ (rz+ry), ZXZ (rz+rx), XYX
    (rx+ry), ZSX (rz+sx[+x], 1 or 2 sx) - and the shortest valid candidate wins.
  * Exactness first: every candidate is checked numerically against the target
    2x2 matrix and its global phase is *measured* (`m2_phase_to`) instead of
    hand-derived, so no decomposition can silently be off by a phase.
  * Named gates (x, sx, s, t, h, ...) are tried first so Clifford+T structure is
    preserved when the basis allows it (TKET-style "prefer named gates").
  * Symbolic (unbound) rotations use affine templates (`rx(t) = rz sx rz(t+pi)
    sx rz`), which the affine `Param` represents exactly; the constant part of
    the global phase is tracked, a parameter-dependent part (rz <-> p) is
    reported through `dropped` because `Circuit.global_phase` is numeric.
"""

from std.math import pi, cos, sin, atan2, sqrt
from qumo.ir.gates import Op, M2, matrix_1q, zyz_decompose, op_num_params
from qumo.ir.param import Param
from qumo.ir.circuit import Instruction
from qumo.compiler.common import (
    GateSet, AX_X, AX_Y, AX_Z, HALF_PI, TWO_PI, FOUR_PI, ANG_TOL,
    m2_phase_to, m2_identity_phase, seq_matrix_1q, wrap_angle, emit_rotation,
    rot_axis, rot_is_r,
)


def can_synthesize_1q(gs: GateSet) -> Bool:
    """True when every 1-qubit unitary is expressible in `gs`."""
    if gs.any or gs.has(Op.U):
        return True
    var z = gs.has_z_rot()
    if z and (gs.has(Op.SX) or gs.has(Op.SXDG) or gs.has(Op.RY) or gs.has(Op.RX) or gs.has(Op.H)):
        return True
    if gs.has(Op.RX) and gs.has(Op.RY):
        return True
    return False


def _zrot(a: Param, q: Int, gs: GateSet) -> Instruction:
    if gs.has(Op.RZ):
        return Instruction.make(Op.RZ, q, p0=a)
    if gs.has(Op.P):
        return Instruction.make(Op.P, q, p0=a)
    return Instruction.make(Op.U, q, p0=0.0, p1=0.0, p2=a)


def _push_z(mut seq: List[Instruction], a: Float64, q: Int, gs: GateSet):
    """Append a Z rotation unless it is the identity up to phase."""
    if abs(wrap_angle(a, TWO_PI)) < ANG_TOL:
        return
    seq.append(_zrot(wrap_angle(a, TWO_PI), q, gs))


def _consider(m: M2, seq: List[Instruction], mut best: List[Instruction],
              mut best_phase: Float64, mut found: Bool):
    if found and len(seq) >= len(best):
        return
    var ph = 0.0
    if m2_phase_to(m, seq_matrix_1q(seq), ph):
        best = seq.copy()
        best_phase = ph
        found = True


def synth_1q(m: M2, q: Int, gs: GateSet, mut out: List[Instruction], mut phase: Float64) -> Bool:
    """Append the shortest gate sequence over `gs` equal to `m` up to a phase.

    `phase` is incremented so that m = e^{i phase_added} * sequence exactly.
    Returns False (nothing appended) when `gs` cannot express `m`.
    """
    var ph = 0.0
    if m2_identity_phase(m, ph):
        phase += ph
        return True
    var best = List[Instruction]()
    var best_phase = 0.0
    var found = False
    # 1. single named gate
    var named: List[Int] = [Op.X, Op.Y, Op.Z, Op.H, Op.S, Op.SDG, Op.T, Op.TDG, Op.SX, Op.SXDG]
    for k in named:
        if gs.has(k):
            var c = List[Instruction]()
            c.append(Instruction.make(k, q))
            _consider(m, c, best, best_phase, found)
            if found:
                break
    var diag = abs(m.re[1]) + abs(m.im[1]) + abs(m.re[2]) + abs(m.im[2]) < 1e-12
    if not found:
        if diag:
            var lam = atan2(m.im[3], m.re[3]) - atan2(m.im[0], m.re[0])
            var c = List[Instruction]()
            var dummy = 0.0
            if emit_rotation(AX_Z, False, Param(lam), q, gs, c, dummy):
                _consider(m, c, best, best_phase, found)
        var z = zyz_decompose(m)
        var theta = z.theta
        # pure X / Y rotations
        if not diag:
            for axis in [AX_X, AX_Y]:
                for sgn in [1.0, -1.0]:
                    var c = List[Instruction]()
                    var dummy = 0.0
                    if emit_rotation(axis, True, Param(sgn * theta), q, gs, c, dummy):
                        _consider(m, c, best, best_phase, found)
        if gs.has(Op.U):
            var c = List[Instruction]()
            c.append(Instruction.make(Op.U, q, p0=z.theta, p1=z.phi, p2=z.lam))
            _consider(m, c, best, best_phase, found)
        var zr = gs.has_z_rot() and not (found and len(best) == 1)
        if zr:
            var phi = z.phi
            var lam = z.lam
            # anti-diagonal: X then Z rotation
            if gs.has(Op.X) and abs(m.re[0]) + abs(m.im[0]) < 1e-12:
                var c = List[Instruction]()
                c.append(Instruction.make(Op.X, q))
                var a = atan2(m.im[2], m.re[2]) - atan2(m.im[1], m.re[1])
                _push_z(c, a, q, gs)
                _consider(m, c, best, best_phase, found)
            if gs.has(Op.RY):
                var c = List[Instruction]()
                _push_z(c, lam, q, gs)
                c.append(Instruction.make(Op.RY, q, p0=theta))
                _push_z(c, phi, q, gs)
                _consider(m, c, best, best_phase, found)
            if gs.has(Op.RX):
                var c = List[Instruction]()
                _push_z(c, lam - HALF_PI, q, gs)
                c.append(Instruction.make(Op.RX, q, p0=theta))
                _push_z(c, phi + HALF_PI, q, gs)
                _consider(m, c, best, best_phase, found)
            if gs.has(Op.SX):
                if abs(theta - HALF_PI) < 1e-9:
                    var c = List[Instruction]()
                    _push_z(c, lam - HALF_PI, q, gs)
                    c.append(Instruction.make(Op.SX, q))
                    _push_z(c, phi + HALF_PI, q, gs)
                    _consider(m, c, best, best_phase, found)
                var c = List[Instruction]()
                _push_z(c, lam, q, gs)
                c.append(Instruction.make(Op.SX, q))
                _push_z(c, theta + pi, q, gs)
                c.append(Instruction.make(Op.SX, q))
                _push_z(c, phi + pi, q, gs)
                _consider(m, c, best, best_phase, found)
            elif gs.has(Op.SXDG):
                # sxdg = rz(pi) sx rz(pi) up to phase -> mirrored ZSX
                var c = List[Instruction]()
                _push_z(c, lam - pi, q, gs)
                c.append(Instruction.make(Op.SXDG, q))
                _push_z(c, theta + pi, q, gs)
                c.append(Instruction.make(Op.SXDG, q))
                _push_z(c, phi, q, gs)
                _consider(m, c, best, best_phase, found)
            if not found and gs.has(Op.H):
                # rz . h . rz . h . rz  (ZXZ with rx(t) = h rz(t) h)
                var c = List[Instruction]()
                _push_z(c, lam - HALF_PI, q, gs)
                c.append(Instruction.make(Op.H, q))
                _push_z(c, theta, q, gs)
                c.append(Instruction.make(Op.H, q))
                _push_z(c, phi + HALF_PI, q, gs)
                _consider(m, c, best, best_phase, found)
        if gs.has(Op.RX) and gs.has(Op.RY) and not (found and len(best) == 1):
            # XYX: conjugate by V = RY(pi/2) (V Z V^ = X, V Y V^ = Y)
            var v = matrix_1q(Op.RY, HALF_PI)
            var w = v.dagger() @ m @ v
            var zw = zyz_decompose(w)
            var c = List[Instruction]()
            if abs(wrap_angle(zw.lam, TWO_PI)) > ANG_TOL:
                c.append(Instruction.make(Op.RX, q, p0=wrap_angle(zw.lam, TWO_PI)))
            c.append(Instruction.make(Op.RY, q, p0=zw.theta))
            if abs(wrap_angle(zw.phi, TWO_PI)) > ANG_TOL:
                c.append(Instruction.make(Op.RX, q, p0=wrap_angle(zw.phi, TWO_PI)))
            _consider(m, c, best, best_phase, found)
    if not found:
        return False
    for g in best:
        out.append(g)
    phase += best_phase
    return True


def _eval_seq(seq: List[Instruction], x: Float64) -> M2:
    """Matrix of a template where every symbol takes the value x."""
    var maxi = 0
    for g in seq:
        maxi = max(maxi, max(g.p0.index, max(g.p1.index, g.p2.index)))
    var vals = List[Float64](length=maxi + 1, fill=x)
    var m = M2.identity()
    for g in seq:
        m = matrix_1q(g.kind, g.p0.resolve(vals), g.p1.resolve(vals), g.p2.resolve(vals)) @ m
    return m


def _template_phase(g: Instruction, seq: List[Instruction], mut phase: Float64,
                    mut dropped: Bool) -> Bool:
    """Phase with g = e^{i phase(x)} seq for every symbol value x.

    The phase of an affine template is affine in x: phase(x) = c + s x. `c` is
    returned; a non-zero slope `s` (parameter-dependent global phase, e.g.
    p <-> rz) sets `dropped`. Verified at a third point.
    """
    var one = List[Instruction]()
    one.append(g)
    var x1 = 0.37
    var x2 = 0.38
    var x3 = -1.21
    var p1 = 0.0
    var p2 = 0.0
    var p3 = 0.0
    if not m2_phase_to(_eval_seq(one, x1), _eval_seq(seq, x1), p1):
        return False
    if not m2_phase_to(_eval_seq(one, x2), _eval_seq(seq, x2), p2):
        return False
    if not m2_phase_to(_eval_seq(one, x3), _eval_seq(seq, x3), p3):
        return False
    var slope = wrap_angle(p2 - p1, TWO_PI) / (x2 - x1)
    var c0 = p1 - slope * x1
    if abs(wrap_angle(c0 + slope * x3 - p3, TWO_PI)) > 1e-7:
        return False
    if abs(slope) > 1e-9:
        dropped = True
    phase = c0
    return True


def synth_1q_symbolic(g: Instruction, gs: GateSet, mut out: List[Instruction],
                      mut phase: Float64, mut dropped: Bool) -> Bool:
    """Translate a symbolic 1q rotation into `gs` with affine templates."""
    var k = g.kind
    var q = g.q0
    if gs.has(k):
        out.append(g)
        return True
    if k == Op.U:
        # U(t, f, l) = e^{i (f + l)/2} RZ(f) RY(t) RZ(l)
        var parts = List[Instruction]()
        parts.append(Instruction.make(Op.RZ, q, p0=g.p2))
        parts.append(Instruction.make(Op.RY, q, p0=g.p0))
        parts.append(Instruction.make(Op.RZ, q, p0=g.p1))
        var tmp = List[Instruction]()
        var ph = (g.p1.offset + g.p2.offset) / 2
        if g.p1.is_symbolic() or g.p2.is_symbolic():
            dropped = True
        for p in parts:
            if p.is_parametric():
                if not synth_1q_symbolic(p, gs, tmp, ph, dropped):
                    return False
            else:
                if not synth_1q(matrix_1q(p.kind, p.p0.value()), q, gs, tmp, ph):
                    return False
        for t in tmp:
            out.append(t)
        phase += ph
        return True
    var a = g.p0
    var cands = List[List[Instruction]]()
    if k == Op.RZ or k == Op.P:
        # same matrix up to the parameter-dependent phase a/2
        var conv_ph = a.offset / 2 if k == Op.P else -a.offset / 2
        var sym_phase = True
        var tmp = List[Instruction]()
        if k == Op.RZ and gs.has(Op.P):
            tmp.append(Instruction.make(Op.P, q, p0=a))
        elif k == Op.P and gs.has(Op.RZ):
            tmp.append(Instruction.make(Op.RZ, q, p0=a))
        elif gs.has(Op.U):
            tmp.append(Instruction.make(Op.U, q, p0=0.0, p1=0.0, p2=a))
            if k == Op.P:
                conv_ph = 0.0
                sym_phase = False
        elif gs.has(Op.RX) and gs.has(Op.RY):
            # RZ(a) = RX(+-pi/2) RY(a) RX(-+pi/2): pick the sign that verifies
            var gz = Instruction.make(Op.RZ, q, p0=a)
            var tp = 0.0
            var ok = False
            for sgn in [1.0, -1.0]:
                if ok:
                    break
                var c = List[Instruction]()
                c.append(Instruction.make(Op.RX, q, p0=sgn * HALF_PI))
                c.append(Instruction.make(Op.RY, q, p0=a))
                c.append(Instruction.make(Op.RX, q, p0=-sgn * HALF_PI))
                var dz = False
                if _template_phase(gz, c, tp, dz):
                    tmp = c^
                    ok = True
            if not ok:
                return False
            conv_ph = tp + (a.offset / 2 if k == Op.P else 0.0)
            if k == Op.RZ:
                sym_phase = False
        else:
            return False
        if a.is_symbolic() and sym_phase:
            dropped = True
        for t in tmp:
            out.append(t)
        phase += conv_ph
        return True
    if k == Op.RX:
        if gs.has(Op.U):
            var c = List[Instruction]()
            c.append(Instruction.make(Op.U, q, p0=a, p1=-HALF_PI, p2=HALF_PI))
            cands.append(c^)
        if gs.has_z_rot() and gs.has(Op.RY):
            var c = List[Instruction]()
            c.append(_zrot(HALF_PI, q, gs))
            c.append(Instruction.make(Op.RY, q, p0=a))
            c.append(_zrot(-HALF_PI, q, gs))
            cands.append(c^)
        if gs.has_z_rot() and gs.has(Op.SX):
            var c = List[Instruction]()
            c.append(_zrot(HALF_PI, q, gs))
            c.append(Instruction.make(Op.SX, q))
            c.append(_zrot(a + pi, q, gs))
            c.append(Instruction.make(Op.SX, q))
            c.append(_zrot(HALF_PI, q, gs))
            cands.append(c^)
        if gs.has_z_rot() and gs.has(Op.SXDG):
            var c = List[Instruction]()
            c.append(_zrot(-HALF_PI, q, gs))
            c.append(Instruction.make(Op.SXDG, q))
            c.append(_zrot(a + pi, q, gs))
            c.append(Instruction.make(Op.SXDG, q))
            c.append(_zrot(-HALF_PI, q, gs))
            cands.append(c^)
        if gs.has_z_rot() and gs.has(Op.H):
            var c = List[Instruction]()
            c.append(Instruction.make(Op.H, q))
            c.append(_zrot(a, q, gs))
            c.append(Instruction.make(Op.H, q))
            cands.append(c^)
    elif k == Op.RY:
        if gs.has(Op.U):
            var c = List[Instruction]()
            c.append(Instruction.make(Op.U, q, p0=a, p1=0.0, p2=0.0))
            cands.append(c^)
        if gs.has_z_rot() and gs.has(Op.RX):
            var c = List[Instruction]()
            c.append(_zrot(-HALF_PI, q, gs))
            c.append(Instruction.make(Op.RX, q, p0=a))
            c.append(_zrot(HALF_PI, q, gs))
            cands.append(c^)
        if gs.has_z_rot() and gs.has(Op.SX):
            var c = List[Instruction]()
            c.append(Instruction.make(Op.SX, q))
            c.append(_zrot(a + pi, q, gs))
            c.append(Instruction.make(Op.SX, q))
            c.append(_zrot(pi, q, gs))
            cands.append(c^)
        if gs.has_z_rot() and gs.has(Op.SXDG):
            var c = List[Instruction]()
            c.append(_zrot(pi, q, gs))
            c.append(Instruction.make(Op.SXDG, q))
            c.append(_zrot(a + pi, q, gs))
            c.append(Instruction.make(Op.SXDG, q))
            cands.append(c^)
        if gs.has_z_rot() and gs.has(Op.H):
            var c = List[Instruction]()
            c.append(_zrot(-HALF_PI, q, gs))
            c.append(Instruction.make(Op.H, q))
            c.append(_zrot(a, q, gs))
            c.append(Instruction.make(Op.H, q))
            c.append(_zrot(HALF_PI, q, gs))
            cands.append(c^)
    for c in cands:
        var tp = 0.0
        var dz = False
        if _template_phase(g, c, tp, dz):
            for t in c:
                out.append(t)
            phase += tp
            if dz:
                dropped = True
            return True
    return False
