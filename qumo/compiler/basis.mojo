"""Basis translation: rewrite every gate into an arbitrary target basis.

Design (Qiskit `BasisTranslator` equivalence library + TKET `rebase`):
  * Exact rules, written once per gate into {1q, CX} (or directly into a native
    entangler when that is cheaper: cp/crz -> 1 rzz, rxx <-> rzz <-> ryy by
    local conjugation, cz -> cp(pi)), applied recursively until every gate is
    native. Rule phases are hard-coded *and* verified by the test-suite.
  * CX itself is mapped onto whatever entangler family the basis offers:
    cx, cz, ecr, iswap, iswapdg, rzz, rxx (Molmer-Sorensen), ryy or cp. So any
    modality (superconducting cz/ecr/iswap, trapped-ion rxx/rzz, neutral-atom
    cz) is reachable from any input gate.
  * Single-qubit gates go through `synth_1q` (shortest Euler form over the
    basis, numerically verified, exact phase) or affine templates for
    symbolic parameters, so parametric circuits stay parametric.
  * Conditioned gates are translated gate-by-gate keeping their condition (a
    classically controlled sequence); their phases are branch-global and are
    not added to `global_phase`.
Errors: QM-E603 (no usable 2q entangler), QM-E604 (1q gate not expressible).
"""

from std.math import pi
from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary
from qumo.ir.param import Param
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gateset import rule_gate as _g, GateSet, HALF_PI, QUARTER_PI, expand_rule
from qumo.compiler.common import inst_matrix_1q
from qumo.compiler.synth import synth_1q, synth_1q_symbolic, can_synthesize_1q
from qumo.compiler.optimize import fuse_1q


def choose_entangler(gs: GateSet) -> Int:
    """Opcode CX is rewritten into (-1 when the basis has no entangler)."""
    var order: List[Int] = [Op.CX, Op.CZ, Op.ECR, Op.ISWAP, Op.ISWAPDG, Op.RZZ, Op.RXX, Op.RYY, Op.CP]
    for k in order:
        if gs.has(k):
            return k
    return -1


def cx_rule(ent: Int, c: Int, t: Int, mut seq: List[Instruction]) -> Float64:
    """CX(c, t) = e^{i phase} * seq (time order); returns phase. ent = target entangler."""
    if ent == Op.CZ:
        seq.append(_g(Op.H, t))
        seq.append(_g(Op.CZ, c, t))
        seq.append(_g(Op.H, t))
        return 0.0
    if ent == Op.CP:
        seq.append(_g(Op.H, t))
        seq.append(_g(Op.CP, c, t, -1, pi))
        seq.append(_g(Op.H, t))
        return 0.0
    if ent == Op.ECR:
        seq.append(_g(Op.X, c))
        seq.append(_g(Op.ECR, c, t))
        seq.append(_g(Op.S, c))
        seq.append(_g(Op.SX, t))
        return -QUARTER_PI
    if ent == Op.ISWAP:
        seq.append(_g(Op.RZ, c, p0=-HALF_PI))
        seq.append(_g(Op.RX, t, p0=HALF_PI))
        seq.append(_g(Op.RZ, t, p0=HALF_PI))
        seq.append(_g(Op.ISWAP, c, t))
        seq.append(_g(Op.RX, c, p0=HALF_PI))
        seq.append(_g(Op.ISWAP, c, t))
        seq.append(_g(Op.RZ, t, p0=HALF_PI))
        return QUARTER_PI
    if ent == Op.ISWAPDG:
        seq.append(_g(Op.RZ, t, p0=-HALF_PI))
        seq.append(_g(Op.ISWAPDG, c, t))
        seq.append(_g(Op.RX, c, p0=-HALF_PI))
        seq.append(_g(Op.ISWAPDG, c, t))
        seq.append(_g(Op.RZ, t, p0=-HALF_PI))
        seq.append(_g(Op.RX, t, p0=-HALF_PI))
        seq.append(_g(Op.RZ, c, p0=HALF_PI))
        return -QUARTER_PI
    if ent == Op.RZZ:
        seq.append(_g(Op.H, t))
        seq.append(_g(Op.RZ, c, p0=HALF_PI))
        seq.append(_g(Op.RZ, t, p0=HALF_PI))
        seq.append(_g(Op.RZZ, c, t, -1, -HALF_PI))
        seq.append(_g(Op.H, t))
        return QUARTER_PI
    if ent == Op.RXX or ent == Op.RYY:
        seq.append(_g(Op.RY, c, p0=HALF_PI))
        if ent == Op.RXX:
            seq.append(_g(Op.RXX, c, t, -1, HALF_PI))
        else:
            seq.append(_g(Op.RZ, c, p0=HALF_PI))
            seq.append(_g(Op.RZ, t, p0=HALF_PI))
            seq.append(_g(Op.RYY, c, t, -1, HALF_PI))
            seq.append(_g(Op.RZ, c, p0=-HALF_PI))
            seq.append(_g(Op.RZ, t, p0=-HALF_PI))
        seq.append(_g(Op.RX, c, p0=-HALF_PI))
        seq.append(_g(Op.RX, t, p0=-HALF_PI))
        seq.append(_g(Op.RY, c, p0=-HALF_PI))
        return -QUARTER_PI
    return 0.0


struct BasisCtx(Movable):
    var gs: GateSet
    var ent: Int
    var dropped: Bool
    var basis_desc: String

    def __init__(out self, gs: GateSet):
        self.gs = gs.copy()
        self.ent = choose_entangler(gs)
        self.dropped = False
        self.basis_desc = String(gs)


def first_inexpressible_1q(c: Circuit, gs: GateSet) raises -> Int:
    """Index of the first 1q gate with no exact form in `gs`, or -1."""
    for i in range(len(c.ops)):
        var g = c.ops[i]
        if not op_is_unitary(g.kind) or op_arity(g.kind) != 1 or gs.has(g.kind) or g.kind == Op.I or g.is_parametric():
            continue
        var tmp = List[Instruction]()
        var ph = 0.0
        if not synth_1q(inst_matrix_1q(g), g.q0, gs, tmp, ph):
            return i
    return -1


def _translate(g: Instruction, mut ctx: BasisCtx, mut out: List[Instruction],
               mut phase: Float64, depth: Int, keep_1q: Bool) raises:
    """Rewrite g into the basis. keep_1q: leave 1q gates for a later fusion step."""
    var k = g.kind
    if not op_is_unitary(k) or ctx.gs.has(k):
        out.append(g)
        return
    if k == Op.I:
        return
    if depth > 16:
        raise Error("QM-E603: basis translation of '" + op_name(k) + "' does not terminate for basis " + ctx.basis_desc + ". Hint: add cx or cz to the basis.")
    var a = op_arity(k)
    if a == 1:
        if keep_1q:
            out.append(g)
            return
        var tmp = List[Instruction]()
        var ph = 0.0
        var ok: Bool
        if g.is_parametric():
            ok = synth_1q_symbolic(g, ctx.gs, tmp, ph, ctx.dropped)
        else:
            ok = synth_1q(inst_matrix_1q(g), g.q0, ctx.gs, tmp, ph)
        if not ok:
            var hint = String("include rz+sx, rz+ry, rz+rx, rx+ry or u in the basis")
            if not can_synthesize_1q(ctx.gs):
                hint = "the basis cannot express arbitrary 1q unitaries; " + hint
            raise Error("QM-E604: cannot express 1-qubit gate '" + String(g) + "' in basis " + ctx.basis_desc + ". Hint: " + hint + ".")
        for t in tmp:
            var x = t
            x.cond = g.cond
            x.cond_val = g.cond_val
            x.cond_reg = g.cond_reg
            out.append(x)
        phase += ph
        return
    var seq = List[Instruction]()
    var ph: Float64
    if k == Op.CX:
        if ctx.ent < 0:
            raise Error("QM-E603: basis " + ctx.basis_desc + " has no two-qubit gate to express '" + op_name(k) + "'. Hint: add one of cx, cz, ecr, iswap, rzz, rxx, ryy, cp to the basis.")
        ph = cx_rule(ctx.ent, g.q0, g.q1, seq)
    else:
        ph = expand_rule(g, ctx.gs, seq, ctx.dropped)
        if len(seq) == 0:
            raise Error("QM-E603: no decomposition rule for '" + op_name(k) + "' into basis " + ctx.basis_desc + ". Hint: add it to the basis.")
    for s in seq:
        var x = s
        x.cond = g.cond
        x.cond_val = g.cond_val
        x.cond_reg = g.cond_reg
        _translate(x, ctx, out, phase, depth + 1, keep_1q)
    phase += ph


def translate_to_basis(mut c: Circuit, gs: GateSet, mut dropped: Bool) raises -> Int:
    """Rewrite all gates into `gs`; returns the number of gates rewritten.

    Three steps keep the output small: (1) multi-qubit gates are expanded by
    the rules with their 1q gates left abstract, (2) every maximal 1q run that
    contains a non-basis gate is fused and resynthesised once (`fuse_1q`),
    (3) leftovers (symbolic or classically conditioned 1q gates) are
    translated gate by gate.
    """
    if gs.any:
        return 0
    var ctx = BasisCtx(gs)
    var out = List[Instruction](capacity=len(c.ops) * 2)
    var rewritten = 0
    var need_1q = False
    for g in c.ops:
        if not op_is_unitary(g.kind) or gs.has(g.kind):
            out.append(g)
            continue
        rewritten += 1
        var ph = 0.0
        _translate(g, ctx, out, ph, 0, True)
        if g.cond < 0:
            c.global_phase += ph
    c.ops = out^
    for g in c.ops:
        if op_is_unitary(g.kind) and not gs.has(g.kind):
            need_1q = True
            break
    if need_1q:
        _ = fuse_1q(c, gs)
        var out2 = List[Instruction](capacity=len(c.ops))
        for g in c.ops:
            if not op_is_unitary(g.kind) or gs.has(g.kind):
                out2.append(g)
                continue
            var ph = 0.0
            _translate(g, ctx, out2, ph, 0, False)
            if g.cond < 0:
                c.global_phase += ph
        c.ops = out2^
    if ctx.dropped:
        dropped = True
    return rewritten


def unroll_3q(mut c: Circuit) raises -> Int:
    """Decompose ccx / ccz / cswap into 1q + cx (needed before routing)."""
    var has3 = False
    for g in c.ops:
        if op_is_unitary(g.kind) and op_arity(g.kind) == 3:
            has3 = True
            break
    if not has3:
        return 0
    var dropped = False
    return translate_to_basis(c, GateSet.all_but_3q(), dropped)
