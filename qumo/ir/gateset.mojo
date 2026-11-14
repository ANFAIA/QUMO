"""Gate sets and exact rewrite rules for multi-qubit gates (the IR level of the compiler).

`GateSet` names the unitary opcodes a target may emit; `expand_rule` rewrites one multi-qubit
gate into simpler gates over a given set. Both are used by compiler passes and by the branching
simulator (qumo.sim.branching), so they live here, below both, and import nothing from
qumo.compiler, qumo.sim or qumo.ai.
"""

from std.math import pi
from qumo.ir.gates import Op, op_name, op_is_unitary
from qumo.ir.param import Param
from qumo.ir.circuit import Instruction

comptime HALF_PI = 1.5707963267948966192313
comptime QUARTER_PI = 0.7853981633974483096157
comptime ANG_TOL = 1e-10
# What a pass may round off per gate in a block of a Program, in place of ANG_TOL and of
# 1e-9 on a matrix. verify_program sees a distance of 1e-7, and small errors add up over
# the gates of a block, so this leaves room for the longest block its budget takes.
comptime EXACT_TOL = 1e-14


# ----------------------------------------------------------------------------
# Gate sets
# ----------------------------------------------------------------------------
struct GateSet(Copyable, Writable):
    """Set of allowed unitary opcodes. `any` = every gate is allowed."""

    var any: Bool
    var allowed: List[Bool]
    var exact: Bool  # round off EXACT_TOL at most: see `angle_tol`

    def __init__(out self):
        self.any = True
        self.allowed = List[Bool](length=Op.NUM_OPS, fill=True)
        self.exact = False

    def __init__(out self, kinds: List[Int]):
        self.exact = False
        if len(kinds) == 0:
            self.any = True
            self.allowed = List[Bool](length=Op.NUM_OPS, fill=True)
            return
        self.any = False
        self.allowed = List[Bool](length=Op.NUM_OPS, fill=False)
        for k in kinds:
            if k >= 0 and k < Op.NUM_OPS:
                self.allowed[k] = True

    def angle_tol(self) -> Float64:
        """An angle this close to zero, or to a named gate's, counts as equal."""
        return EXACT_TOL if self.exact else ANG_TOL

    def matrix_tol(self) -> Float64:
        """An entry of a resynthesised 1-qubit matrix may be this far off."""
        return EXACT_TOL if self.exact else 1e-9

    @staticmethod
    def all_but_3q() -> GateSet:
        var g = GateSet()
        g.any = False
        g.allowed[Op.CCX] = False
        g.allowed[Op.CCZ] = False
        g.allowed[Op.CSWAP] = False
        return g^

    def has(self, k: Int) -> Bool:
        if k < 0 or k >= Op.NUM_OPS:
            return False
        if not op_is_unitary(k):
            return True
        return self.allowed[k]

    def has_z_rot(self) -> Bool:
        return self.has(Op.RZ) or self.has(Op.P)

    def kinds(self) -> List[Int]:
        var out = List[Int]()
        if self.any:
            return out^
        for k in range(Op.CCZ + 1):
            if self.allowed[k]:
                out.append(k)
        return out^

    def write_to(self, mut w: Some[Writer]):
        if self.any:
            w.write("{any}")
            return
        w.write("{")
        var first = True
        for k in range(Op.CCZ + 1):
            if self.allowed[k]:
                if not first:
                    w.write(",")
                first = False
                w.write(op_name(k))
        w.write("}")



@always_inline
def rule_gate(k: Int, q0: Int, q1: Int = -1, q2: Int = -1, p0: Param = 0.0) -> Instruction:
    return Instruction.make(k, q0, q1, q2, p0)




def expand_rule(g: Instruction, gs: GateSet, mut seq: List[Instruction], mut dropped: Bool) -> Float64:
    """One rewrite step for a multi-qubit gate: g = e^{i phase} * seq."""
    var k = g.kind
    var a = g.q0
    var b = g.q1
    var c = g.q2
    var th = g.p0
    var direct_zz = gs.has(Op.RZZ) and not gs.has(Op.CX)
    if k == Op.CZ:
        if gs.has(Op.CP):
            seq.append(rule_gate(Op.CP, a, b, -1, pi))
            return 0.0
        if direct_zz:
            seq.append(rule_gate(Op.RZ, a, p0=HALF_PI))
            seq.append(rule_gate(Op.RZ, b, p0=HALF_PI))
            seq.append(rule_gate(Op.RZZ, a, b, -1, -HALF_PI))
            return QUARTER_PI
        seq.append(rule_gate(Op.H, b))
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.H, b))
        return 0.0
    if k == Op.CY:
        seq.append(rule_gate(Op.SDG, b))
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.S, b))
        return 0.0
    if k == Op.CH:
        seq.append(rule_gate(Op.S, b))
        seq.append(rule_gate(Op.H, b))
        seq.append(rule_gate(Op.T, b))
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.TDG, b))
        seq.append(rule_gate(Op.H, b))
        seq.append(rule_gate(Op.SDG, b))
        return 0.0
    if k == Op.SWAP:
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.CX, b, a))
        seq.append(rule_gate(Op.CX, a, b))
        return 0.0
    if k == Op.ISWAP and gs.has(Op.ISWAPDG):
        # Z_a . iSWAPdg . Z_a = iSWAP
        seq.append(rule_gate(Op.Z, a))
        seq.append(rule_gate(Op.ISWAPDG, a, b))
        seq.append(rule_gate(Op.Z, a))
        return 0.0
    if k == Op.ISWAPDG and gs.has(Op.ISWAP):
        seq.append(rule_gate(Op.Z, a))
        seq.append(rule_gate(Op.ISWAP, a, b))
        seq.append(rule_gate(Op.Z, a))
        return 0.0
    if k == Op.ISWAP:
        seq.append(rule_gate(Op.S, a))
        seq.append(rule_gate(Op.S, b))
        seq.append(rule_gate(Op.H, a))
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.CX, b, a))
        seq.append(rule_gate(Op.H, b))
        return 0.0
    if k == Op.ISWAPDG:
        seq.append(rule_gate(Op.H, b))
        seq.append(rule_gate(Op.CX, b, a))
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.H, a))
        seq.append(rule_gate(Op.SDG, b))
        seq.append(rule_gate(Op.SDG, a))
        return 0.0
    if k == Op.ECR:
        seq.append(rule_gate(Op.X, a))
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.SDG, a))
        seq.append(rule_gate(Op.SXDG, b))
        return QUARTER_PI
    if k == Op.CP:
        if direct_zz or (gs.has(Op.RXX) and not gs.has(Op.CX)):
            # CP(l) = e^{i l/4} RZ(l/2) x RZ(l/2) . RZZ(-l/2)
            seq.append(rule_gate(Op.RZ, a, p0=th * 0.5))
            seq.append(rule_gate(Op.RZ, b, p0=th * 0.5))
            seq.append(rule_gate(Op.RZZ, a, b, -1, th * -0.5))
            if th.is_symbolic():
                dropped = True
            return th.offset / 4
        seq.append(rule_gate(Op.P, a, p0=th * 0.5))
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.P, b, p0=th * -0.5))
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.P, b, p0=th * 0.5))
        return 0.0
    if k == Op.CRZ:
        if direct_zz or (gs.has(Op.RXX) and not gs.has(Op.CX)):
            # CRZ(t) = RZ(t/2)_target . RZZ(-t/2)
            seq.append(rule_gate(Op.RZ, b, p0=th * 0.5))
            seq.append(rule_gate(Op.RZZ, a, b, -1, th * -0.5))
            return 0.0
        seq.append(rule_gate(Op.RZ, b, p0=th * 0.5))
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.RZ, b, p0=th * -0.5))
        seq.append(rule_gate(Op.CX, a, b))
        return 0.0
    if k == Op.CRY:
        seq.append(rule_gate(Op.RY, b, p0=th * 0.5))
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.RY, b, p0=th * -0.5))
        seq.append(rule_gate(Op.CX, a, b))
        return 0.0
    if k == Op.CRX:
        seq.append(rule_gate(Op.H, b))
        seq.append(rule_gate(Op.CRZ, a, b, -1, th))
        seq.append(rule_gate(Op.H, b))
        return 0.0
    if k == Op.RZZ:
        if gs.has(Op.RXX):
            seq.append(rule_gate(Op.H, a))
            seq.append(rule_gate(Op.H, b))
            seq.append(rule_gate(Op.RXX, a, b, -1, th))
            seq.append(rule_gate(Op.H, a))
            seq.append(rule_gate(Op.H, b))
            return 0.0
        if gs.has(Op.RYY):
            seq.append(rule_gate(Op.RX, a, p0=-HALF_PI))
            seq.append(rule_gate(Op.RX, b, p0=-HALF_PI))
            seq.append(rule_gate(Op.RYY, a, b, -1, th))
            seq.append(rule_gate(Op.RX, a, p0=HALF_PI))
            seq.append(rule_gate(Op.RX, b, p0=HALF_PI))
            return 0.0
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.RZ, b, p0=th))
        seq.append(rule_gate(Op.CX, a, b))
        return 0.0
    if k == Op.RXX:
        if gs.has(Op.RYY) and not gs.has(Op.RZZ):
            seq.append(rule_gate(Op.RZ, a, p0=HALF_PI))
            seq.append(rule_gate(Op.RZ, b, p0=HALF_PI))
            seq.append(rule_gate(Op.RYY, a, b, -1, th))
            seq.append(rule_gate(Op.RZ, a, p0=-HALF_PI))
            seq.append(rule_gate(Op.RZ, b, p0=-HALF_PI))
            return 0.0
        seq.append(rule_gate(Op.H, a))
        seq.append(rule_gate(Op.H, b))
        seq.append(rule_gate(Op.RZZ, a, b, -1, th))
        seq.append(rule_gate(Op.H, a))
        seq.append(rule_gate(Op.H, b))
        return 0.0
    if k == Op.RYY:
        if gs.has(Op.RXX) and not gs.has(Op.RZZ):
            seq.append(rule_gate(Op.RZ, a, p0=-HALF_PI))
            seq.append(rule_gate(Op.RZ, b, p0=-HALF_PI))
            seq.append(rule_gate(Op.RXX, a, b, -1, th))
            seq.append(rule_gate(Op.RZ, a, p0=HALF_PI))
            seq.append(rule_gate(Op.RZ, b, p0=HALF_PI))
            return 0.0
        seq.append(rule_gate(Op.RX, a, p0=HALF_PI))
        seq.append(rule_gate(Op.RX, b, p0=HALF_PI))
        seq.append(rule_gate(Op.RZZ, a, b, -1, th))
        seq.append(rule_gate(Op.RX, a, p0=-HALF_PI))
        seq.append(rule_gate(Op.RX, b, p0=-HALF_PI))
        return 0.0
    if k == Op.CCX or k == Op.CCZ:
        if k == Op.CCX:
            seq.append(rule_gate(Op.H, c))
        seq.append(rule_gate(Op.CX, b, c))
        seq.append(rule_gate(Op.TDG, c))
        seq.append(rule_gate(Op.CX, a, c))
        seq.append(rule_gate(Op.T, c))
        seq.append(rule_gate(Op.CX, b, c))
        seq.append(rule_gate(Op.TDG, c))
        seq.append(rule_gate(Op.CX, a, c))
        seq.append(rule_gate(Op.T, b))
        seq.append(rule_gate(Op.T, c))
        if k == Op.CCX:
            seq.append(rule_gate(Op.H, c))
        seq.append(rule_gate(Op.CX, a, b))
        seq.append(rule_gate(Op.T, a))
        seq.append(rule_gate(Op.TDG, b))
        seq.append(rule_gate(Op.CX, a, b))
        return 0.0
    if k == Op.CSWAP:
        seq.append(rule_gate(Op.CX, c, b))
        seq.append(rule_gate(Op.CCX, a, b, c))
        seq.append(rule_gate(Op.CX, c, b))
        return 0.0
    return 0.0
