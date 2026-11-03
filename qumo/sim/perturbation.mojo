"""Gate-angle parameter perturbation (qilisdk GaussianPerturbation and OffsetPerturbation).

Perturbed instructions: the single-angle rotations RX, RY, RZ, P, CP, CRX, CRY, CRZ,
RXX, RYY and RZZ, whose one angle is p0 (`is_angle_rotation`). U, GPHASE, noise
probabilities and every other gate are left unchanged.

  * Angles are perturbed per gate instance: two rx gates on one parameter get
    independent samples.
  * The perturbation is additive, in radians.
  * A symbolic angle keeps its coefficient: coeff * theta + offset becomes
    coeff * theta + (offset + delta).
  * Only gate angles are perturbed. qilisdk also perturbs schedule parameters,
    and its per-parameter-name scoping is not implemented: every rotation is
    perturbed.
"""

from std.math import isinf, isnan

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.param import Param
from qumo.sim.rng import Rng


def is_angle_rotation(kind: Int) -> Bool:
    """True for the single-angle rotations, whose angle is p0."""
    return (
        kind == Op.RX or kind == Op.RY or kind == Op.RZ or kind == Op.P
        or kind == Op.CP or kind == Op.CRX or kind == Op.CRY or kind == Op.CRZ
        or kind == Op.RXX or kind == Op.RYY or kind == Op.RZZ
    )


def _moved(p: Param, delta: Float64) -> Param:
    """p shifted by delta radians: a constant stays constant, a symbolic angle keeps its coefficient."""
    if p.is_symbolic():
        return Param(p.offset + delta, p.index, p.coeff)
    return Param(p.offset + delta)


def gaussian_angle_perturbation(circuit: Circuit, std: Float64, seed: UInt64) raises -> Circuit:
    """Copy of `circuit` with every rotation angle moved by its own N(0, std^2) sample, in radians.

    Samples are drawn from Rng(seed) in op order, one per rotation. std == 0 returns
    an identical copy. The input is not modified.
    """
    if isnan(std) or isinf(std) or std < 0.0:
        raise Error("QM-E415: gaussian_angle_perturbation: std " + String(std) + " is negative or not finite. Hint: pass a standard deviation >= 0, e.g. gaussian_angle_perturbation(circuit, 0.01, seed).")
    var out = circuit.copy()
    if std == 0.0:
        return out^
    var rng = Rng(seed)
    for ref g in out.ops:
        if is_angle_rotation(g.kind):
            var delta = std * rng.normal()
            g.p0 = _moved(g.p0, delta)
    return out^


def offset_angle_perturbation(circuit: Circuit, offset: Float64) raises -> Circuit:
    """Copy of `circuit` with `offset` radians added to every rotation angle.

    Constant and symbolic angles both move; a symbolic angle keeps its coefficient.
    offset == 0 returns an identical copy. The input is not modified.
    """
    if isnan(offset) or isinf(offset):
        raise Error("QM-E415: offset_angle_perturbation: offset " + String(offset) + " is not finite. Hint: pass a finite offset, e.g. offset_angle_perturbation(circuit, 0.01).")
    var out = circuit.copy()
    if offset == 0.0:
        return out^
    for ref g in out.ops:
        if is_angle_rotation(g.kind):
            g.p0 = _moved(g.p0, offset)
    return out^
