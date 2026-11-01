"""`wrap_angles`: bring rotation angles into the range a target accepts.

Some devices take a parameterised gate only inside an interval (fractional RZZ on
IBM hardware: 0 <= theta <= pi/2). `Target.angle_bounds`, set by a spec like
`rz,sx,x,cz,rzz[0:pi/2]`, records it, and this pass rewrites what falls outside
with Pauli corrections, exact up to the tracked `global_phase`:

    R(t)  = R(k pi) R(t - k pi),  R(pi) = -i P    (P = Z Z for RZZ, X for RX, ...)
    R(-t) = Q R(t) Q                              (Q anticommutes with P on one qubit)

P(lam) follows the same two rules with Z and X (X P(t) X = e^{it} P(-t)). CP only
has its period 2 pi, CRX / CRY / CRZ their period 4 pi. The smallest shift wins,
without reflection when that fits. Symbolic angles stay, they are bound later.
A corrected conditional gate gets conditional Paulis; its phase is dropped, as a
classically conditioned phase is unobservable.
"""

from std.math import pi, floor
from qumo.ir.gates import Op, op_name
from qumo.ir.param import Param
from qumo.ir.circuit import Circuit, Instruction
from qumo.compiler.target import Target


def _in_range(v: Float64, lo: Float64, hi: Float64) -> Bool:
    return v >= lo - 1e-12 and v <= hi + 1e-12


def _clamp(v: Float64, lo: Float64, hi: Float64) -> Float64:
    return min(max(v, lo), hi)


def _put(mut out: Circuit, like: Instruction, kind: Int, q: Int):
    """A correction gate carrying the condition of the gate it corrects."""
    var inst = like.copy()
    inst.kind = kind
    inst.q0 = q
    inst.q1 = -1
    inst.q2 = -1
    inst.p0 = 0.0
    inst.p1 = 0.0
    inst.p2 = 0.0
    out.append(inst)


def _rotation_axes(kind: Int, mut pauli: Int, mut flip: Int) -> Bool:
    """Pauli of a rotation gate and a Pauli anticommuting with it. False otherwise."""
    if kind == Op.RX or kind == Op.RXX:
        pauli = Op.X
        flip = Op.Z
    elif kind == Op.RY or kind == Op.RYY:
        pauli = Op.Y
        flip = Op.Z
    elif kind == Op.RZ or kind == Op.RZZ:
        pauli = Op.Z
        flip = Op.X
    else:
        return False
    return True


def _wrap_one(mut out: Circuit, g: Instruction, lo: Float64, hi: Float64) raises:
    var theta = g.p0.value()
    var pauli = 0
    var flip = 0
    var two_qubit = g.kind == Op.RXX or g.kind == Op.RYY or g.kind == Op.RZZ
    if _rotation_axes(g.kind, pauli, flip) or g.kind == Op.P:
        if g.kind == Op.P:
            pauli = Op.Z
            flip = Op.X
        var center = Int(floor(theta / pi + 0.5))
        # nearest multiple of pi first, then outwards on both sides
        for step in range(0, 16):
            var k = center + (step + 1) // 2 if step % 2 == 1 else center - step // 2
            for s in range(2):
                var phi = theta - Float64(k) * pi
                if s == 1:
                    phi = -phi
                if not _in_range(phi, lo, hi):
                    continue
                phi = _clamp(phi, lo, hi)
                if s == 1:
                    _put(out, g, flip, g.q0)
                var inst = g.copy()
                inst.p0 = phi
                out.append(inst)
                if s == 1:
                    _put(out, g, flip, g.q0)
                if k % 2 != 0:
                    _put(out, g, pauli, g.q0)
                    if two_qubit:
                        _put(out, g, pauli, g.q1)
                if g.cond < 0:
                    if g.kind == Op.P:
                        if s == 1:
                            out.global_phase -= phi
                    else:
                        out.global_phase -= Float64(k) * pi / 2
                return
    else:
        var period = 2 * pi if g.kind == Op.CP else 4 * pi
        var first = Int(floor((lo - theta) / period)) - 1
        for m in range(first, first + 4):
            var phi = theta + Float64(m) * period
            if _in_range(phi, lo, hi):
                var inst = g.copy()
                inst.p0 = _clamp(phi, lo, hi)
                out.append(inst)
                return
    raise Error("QM-E622: angle " + String(theta) + " of '" + op_name(g.kind) + "' cannot be brought into [" + String(lo) + ", " + String(hi) + "]. Hint: a Pauli rotation needs an interval that covers [0, pi/2] up to a period; " + op_name(g.kind) + " may need a wider one.")


def wrap_angles(mut c: Circuit, target: Target) raises -> Int:
    """Rewrite out-of-bounds constant angles. Returns how many gates changed."""
    if not target.has_angle_bounds():
        return 0
    var out = c.empty_like()
    out.num_measurements = c.num_measurements
    out.num_detectors = c.num_detectors
    out.num_observables = c.num_observables
    out.records = c.records.copy()
    var changed = 0
    for g in c.ops:
        if not target.angle_bound[g.kind] or g.p0.is_symbolic():
            out.append(g)
            continue
        var lo = target.angle_lo[g.kind]
        var hi = target.angle_hi[g.kind]
        if _in_range(g.p0.value(), lo, hi):
            out.append(g)
            continue
        _wrap_one(out, g, lo, hi)
        changed += 1
    if changed > 0:
        c = out^
    return changed
