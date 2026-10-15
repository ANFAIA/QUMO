"""`compile`, `implement_ops`, `encode`: the three Guppy FT passes for QUMO circuits.

    computational Circuit --compile--> LogicalProgram --implement_ops--> EncodedProgram
                          \\_____________________ encode ____________________/

`compile(arch, circuit)` lowers a computational `qumo.ir.circuit.Circuit`
(noiseless, any gate set, mid-circuit measurement, reset, `c_if` and
`c_if_register` feed-forward) into the architecture's logical gate set:

* qubits are allocated lazily (first use) and re-allocated after a reset or
  after a measurement followed by more gates (a measured qubit is re-prepared
  in |m> by a conditioned logical X, so QUMO's "gates after measurement"
  semantics is kept);
* classical-only qubits (touched only by reset, X - possibly conditioned - and
  measure, e.g. a classical lane computing an OR of measured bits) are not
  encoded at all: their value is propagated as a classical expression and the
  measurement becomes a classical assignment (LOp.CSET);
* gates are decomposed into the logical gate set: Steane = Clifford + T
  (SX = H S H, CY = Sdg CX S, CH, ISWAP, CCX (7 T), CCZ, CSWAP, RX/RY/U/CRZ/CP/
  RZZ/RXX/RYY through RZ); Iceberg = native rotations (H = RZ RX RZ,
  CZ = RZ RZ RZZ(-pi/2), ...);
* RZ on Steane: exact for multiples of pi/4, otherwise `rz_mode` "static"
  (fewest-T word within rz_epsilon, `qumo.ft.rz.synthesize_rz`; the achieved
  error is recorded in the program notes), "comparator" (repeat-until-success
  circuit of arXiv:2404.05618 expanded in place before compilation) or
  "exact" (reject non-pi/4 angles).

Conditions are copied onto every logical op of a decomposition (the same
classical condition on each gate of a sequence conditions the sequence).

`implement_ops` (qumo.ft.implement) links the logical ops to gadgets;
`encode` = compile + implement_ops. `check_may_encode` reports the first
problem without encoding (Guppy FT `check_may_encode`).

Idea from Guppy FT (Apache-2.0), files src/guppyft/encode/__init__.py
(`encode`, `EncodeSpec`), _compile.py (`ReplacementCompiler`) and
src/guppyft/code/steane/encode.py (`SteaneInstance.compile/encode`); re-derived.

Error codes: QM-E724 (gate not supported by the architecture),
QM-E730 (symbolic parameters / noise / QEC annotations in a computational circuit).
"""

from std.math import pi

from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary, op_is_noise, op_is_annotation
from qumo.ft.classical import CTerm, CExpr
from qumo.ft.logical import LogicalProgram, LogicalInstr, LOp
from qumo.ft.arch import Architecture
from qumo.ft.rz import synthesize_rz, exact_rz_sequence, append_comparator_rz
from qumo.ft.implement import implement_ops, EncodedProgram


struct _Compiler(Movable):
    var arch: Architecture
    var lp: LogicalProgram
    var handle: List[Int]  # computational qubit -> live logical handle (-1)
    var measured: List[Int]  # qubit -> clbit of its last measurement when dead (-1)
    var classical: List[Bool]
    var cval: List[CExpr]  # value of classical-only qubits
    var src: Circuit
    var cur_cond: Int
    var cur_val: Int
    var cur_bits: List[Int]
    var cur_vals: List[Int]
    var cur_kind: Int  # 0 none, 1 bit, 2 register
    var rz_notes: Int
    var max_rz_err: Float64

    def __init__(out self, arch: Architecture, src: Circuit):
        self.arch = arch.copy()
        self.lp = LogicalProgram(src.name)
        self.handle = List[Int](length=src.num_qubits, fill=-1)
        self.measured = List[Int](length=src.num_qubits, fill=-1)
        self.classical = List[Bool](length=src.num_qubits, fill=False)
        self.cval = List[CExpr]()
        for _ in range(src.num_qubits):
            self.cval.append(CExpr())
        self.src = src.copy()
        self.cur_cond = -1
        self.cur_val = 0
        self.cur_bits = List[Int]()
        self.cur_vals = List[Int]()
        self.cur_kind = 0
        self.rz_notes = 0
        self.max_rz_err = 0.0

    def _condition(mut self) raises:
        if self.cur_kind == 1:
            self.lp.c_if(self.cur_cond, self.cur_val)
        elif self.cur_kind == 2:
            self.lp.c_if_register(self.cur_bits, self.cur_vals)

    def q(mut self, cq: Int) raises -> Int:
        """Live logical handle of computational qubit cq (allocated on demand)."""
        if self.handle[cq] < 0:
            var h = self.lp.alloc()
            self.handle[cq] = h
            if self.measured[cq] >= 0:
                # gates after a measurement: the qubit is |m>, re-create it
                self.lp.x(h)
                self.lp.c_if(self.measured[cq], 1)
                self.measured[cq] = -1
        return self.handle[cq]

    def g1(mut self, kind: Int, cq: Int, angle: Float64 = 0.0) raises:
        var h = self.q(cq)
        self.lp.gate(kind, h, -1, angle)
        self._condition()

    def g2(mut self, kind: Int, ca: Int, cb: Int, angle: Float64 = 0.0) raises:
        var a = self.q(ca)
        var b = self.q(cb)
        self.lp.gate(kind, a, b, angle)
        self._condition()

    def is_steane(self) -> Bool:
        return self.arch.kind == "steane"

    # ------------------------------------------------------------- rotations
    def rz(mut self, theta: Float64, cq: Int) raises:
        if not self.is_steane():
            self.g1(LOp.RZ, cq, theta)
            return
        var seq = List[Int]()
        if exact_rz_sequence(theta, seq):
            for k in seq:
                self.gate1(k, cq)
            return
        if self.arch.rz_mode == "exact":
            raise Error("QM-E724: RZ(" + String(theta) + ") is not a multiple of pi/4 and rz_mode='exact'. Hint: use rz_mode 'static' (Clifford+T synthesis) or 'comparator'.")
        # static synthesis ("comparator" angles are expanded before compile)
        var syn = synthesize_rz(theta, self.arch.rz_epsilon, self.arch.rz_max_t)
        if syn.error > self.max_rz_err:
            self.max_rz_err = syn.error
        self.rz_notes += 1
        for k in syn.gates:
            self.gate1(k, cq)

    def gate1(mut self, k: Int, cq: Int) raises:
        """A fixed 1-qubit gate (Op kind) in the logical gate set."""
        if k == Op.I:
            return
        if self.is_steane():
            if k == Op.X: self.g1(LOp.X, cq)
            elif k == Op.Y: self.g1(LOp.Y, cq)
            elif k == Op.Z: self.g1(LOp.Z, cq)
            elif k == Op.H: self.g1(LOp.H, cq)
            elif k == Op.S: self.g1(LOp.S, cq)
            elif k == Op.SDG: self.g1(LOp.SDG, cq)
            elif k == Op.T: self.g1(LOp.T, cq)
            elif k == Op.TDG: self.g1(LOp.TDG, cq)
            elif k == Op.SX:
                self.gate1(Op.H, cq)
                self.gate1(Op.S, cq)
                self.gate1(Op.H, cq)
            elif k == Op.SXDG:
                self.gate1(Op.H, cq)
                self.gate1(Op.SDG, cq)
                self.gate1(Op.H, cq)
            else:
                raise Error("QM-E724: internal: gate1(" + op_name(k) + ").")
            return
        # Iceberg: Paulis native, everything else through rotations
        var hp = pi / 2
        if k == Op.X: self.g1(LOp.X, cq)
        elif k == Op.Y: self.g1(LOp.Y, cq)
        elif k == Op.Z: self.g1(LOp.Z, cq)
        elif k == Op.S: self.g1(LOp.RZ, cq, hp)
        elif k == Op.SDG: self.g1(LOp.RZ, cq, -hp)
        elif k == Op.T: self.g1(LOp.RZ, cq, pi / 4)
        elif k == Op.TDG: self.g1(LOp.RZ, cq, -pi / 4)
        elif k == Op.SX: self.g1(LOp.RX, cq, hp)
        elif k == Op.SXDG: self.g1(LOp.RX, cq, -hp)
        elif k == Op.H:
            self.g1(LOp.RZ, cq, hp)
            self.g1(LOp.RX, cq, hp)
            self.g1(LOp.RZ, cq, hp)
        else:
            raise Error("QM-E724: internal: gate1(" + op_name(k) + ").")

    def rx(mut self, theta: Float64, cq: Int) raises:
        if not self.is_steane():
            self.g1(LOp.RX, cq, theta)
            return
        self.gate1(Op.H, cq)
        self.rz(theta, cq)
        self.gate1(Op.H, cq)

    def ry(mut self, theta: Float64, cq: Int) raises:
        # RY = S RX S^dag (circuit order: Sdg, RX, S)
        self.gate1(Op.SDG, cq)
        self.rx(theta, cq)
        self.gate1(Op.S, cq)

    def cx(mut self, c: Int, t: Int) raises:
        if self.is_steane():
            self.g2(LOp.CX, c, t)
            return
        self.gate1(Op.H, t)
        self.cz(c, t)
        self.gate1(Op.H, t)

    def cz(mut self, a: Int, b: Int) raises:
        if self.is_steane():
            self.g2(LOp.CZ, a, b)
            return
        var hp = pi / 2
        self.g1(LOp.RZ, a, hp)
        self.g1(LOp.RZ, b, hp)
        self.g2(LOp.RZZ, a, b, -hp)

    def rzz(mut self, theta: Float64, a: Int, b: Int) raises:
        if not self.is_steane():
            self.g2(LOp.RZZ, a, b, theta)
            return
        self.cx(a, b)
        self.rz(theta, b)
        self.cx(a, b)

    def rxx(mut self, theta: Float64, a: Int, b: Int) raises:
        if not self.is_steane():
            self.g2(LOp.RXX, a, b, theta)
            return
        self.gate1(Op.H, a)
        self.gate1(Op.H, b)
        self.rzz(theta, a, b)
        self.gate1(Op.H, a)
        self.gate1(Op.H, b)

    def ccx(mut self, a: Int, b: Int, t: Int) raises:
        # 7-T Toffoli (same network as qumo.ft.rz._toffoli)
        self.gate1(Op.H, t)
        self.cx(b, t)
        self.gate1(Op.TDG, t)
        self.cx(a, t)
        self.gate1(Op.T, t)
        self.cx(b, t)
        self.gate1(Op.TDG, t)
        self.cx(a, t)
        self.gate1(Op.T, b)
        self.gate1(Op.T, t)
        self.gate1(Op.H, t)
        self.cx(a, b)
        self.gate1(Op.T, a)
        self.gate1(Op.TDG, b)
        self.cx(a, b)

    # ------------------------------------------------------------- dispatch
    def instruction(mut self, g: Instruction) raises:
        var k = g.kind
        if k == Op.I:
            return
        if k == Op.X or k == Op.Y or k == Op.Z or k == Op.H or k == Op.S or k == Op.SDG or k == Op.T or k == Op.TDG or k == Op.SX or k == Op.SXDG:
            self.gate1(k, g.q0)
        elif k == Op.RZ or k == Op.P:
            self.rz(g.p0.value(), g.q0)
        elif k == Op.RX:
            self.rx(g.p0.value(), g.q0)
        elif k == Op.RY:
            self.ry(g.p0.value(), g.q0)
        elif k == Op.U:
            # U(theta, phi, lam) = RZ(phi) RY(theta) RZ(lam) up to phase
            self.rz(g.p2.value(), g.q0)
            self.ry(g.p0.value(), g.q0)
            self.rz(g.p1.value(), g.q0)
        elif k == Op.CX:
            self.cx(g.q0, g.q1)
        elif k == Op.CZ:
            self.cz(g.q0, g.q1)
        elif k == Op.CY:
            self.gate1(Op.SDG, g.q1)
            self.cx(g.q0, g.q1)
            self.gate1(Op.S, g.q1)
        elif k == Op.CH:
            # CH = (S H T on t) CX (T^dag H S^dag on t), circuit order below
            self.gate1(Op.SDG, g.q1)
            self.gate1(Op.H, g.q1)
            self.gate1(Op.TDG, g.q1)
            self.cx(g.q0, g.q1)
            self.gate1(Op.T, g.q1)
            self.gate1(Op.H, g.q1)
            self.gate1(Op.S, g.q1)
        elif k == Op.SWAP:
            self.g2(LOp.SWAP, g.q0, g.q1)
        elif k == Op.ISWAP:
            self.gate1(Op.S, g.q0)
            self.gate1(Op.S, g.q1)
            self.gate1(Op.H, g.q0)
            self.cx(g.q0, g.q1)
            self.cx(g.q1, g.q0)
            self.gate1(Op.H, g.q1)
        elif k == Op.ISWAPDG:
            self.gate1(Op.H, g.q1)
            self.cx(g.q1, g.q0)
            self.cx(g.q0, g.q1)
            self.gate1(Op.H, g.q0)
            self.gate1(Op.SDG, g.q0)
            self.gate1(Op.SDG, g.q1)
        elif k == Op.CRZ or k == Op.CP:
            var l = g.p0.value()
            if k == Op.CP:
                self.rz(l / 2, g.q0)
            self.rz(l / 2, g.q1)
            self.cx(g.q0, g.q1)
            self.rz(-l / 2, g.q1)
            self.cx(g.q0, g.q1)
        elif k == Op.CRX or k == Op.CRY:
            var l = g.p0.value()
            # CRX = H_t CRZ H_t ; CRY = S_t CRX S_t^dag
            if k == Op.CRY:
                self.gate1(Op.SDG, g.q1)
            self.gate1(Op.H, g.q1)
            self.rz(l / 2, g.q1)
            self.cx(g.q0, g.q1)
            self.rz(-l / 2, g.q1)
            self.cx(g.q0, g.q1)
            self.gate1(Op.H, g.q1)
            if k == Op.CRY:
                self.gate1(Op.S, g.q1)
        elif k == Op.RZZ:
            self.rzz(g.p0.value(), g.q0, g.q1)
        elif k == Op.RXX:
            self.rxx(g.p0.value(), g.q0, g.q1)
        elif k == Op.RYY:
            self.gate1(Op.SX, g.q0)
            self.gate1(Op.SX, g.q1)
            self.rzz(g.p0.value(), g.q0, g.q1)
            self.gate1(Op.SXDG, g.q0)
            self.gate1(Op.SXDG, g.q1)
        elif k == Op.CCX:
            self.ccx(g.q0, g.q1, g.q2)
        elif k == Op.CCZ:
            self.gate1(Op.H, g.q2)
            self.ccx(g.q0, g.q1, g.q2)
            self.gate1(Op.H, g.q2)
        elif k == Op.CSWAP:
            self.cx(g.q2, g.q1)
            self.ccx(g.q0, g.q1, g.q2)
            self.cx(g.q2, g.q1)
        else:
            raise Error("QM-E724: gate '" + op_name(k) + "' has no decomposition into the " + self.arch.kind + " logical gate set. Hint: translate it to {h, s, t, cx, rz} first (qumo.compiler basis translation).")


def _classical_qubits(c: Circuit) -> List[Bool]:
    """Qubits that only ever see reset / X (any condition) / measure stay in a
    computational basis state: they are classical registers in disguise."""
    var cl = List[Bool](length=c.num_qubits, fill=True)
    var used = List[Bool](length=c.num_qubits, fill=False)
    for g in c.ops:
        var a = op_arity(g.kind)
        if op_is_annotation(g.kind) or g.kind == Op.BARRIER or g.kind == Op.TICK:
            continue
        for j in range(a):
            var q = g.qubit(j)
            used[q] = True
            if not (g.kind == Op.RESET or g.kind == Op.X or g.kind == Op.MEASURE or g.kind == Op.I):
                cl[q] = False
    for q in range(c.num_qubits):
        if not used[q]:
            cl[q] = False
    return cl^


def check_may_encode(arch: Architecture, c: Circuit) raises:
    """Raise the first problem `compile` would hit (QM-E724 / QM-E730)."""
    c.validate()
    for i in range(len(c.ops)):
        var g = c.ops[i]
        if op_is_noise(g.kind):
            raise Error("QM-E730: instruction #" + String(i) + " '" + op_name(g.kind) + "' is a noise channel: encode() takes noiseless computational circuits. Hint: add noise to the encoded program with qumo.ft.emulate.add_noise.")
        if g.kind == Op.DETECTOR or g.kind == Op.OBSERVABLE:
            raise Error("QM-E730: instruction #" + String(i) + " is a QEC annotation (detector/observable); computational circuits are QEC-agnostic. Hint: remove it; the architecture inserts its own checks.")
        if g.is_parametric():
            raise Error("QM-E730: instruction #" + String(i) + " '" + op_name(g.kind) + "' has an unbound symbolic parameter. Hint: bind() it, or for angles only known at run time build the comparator circuit (qumo.ft.rz.append_comparator_rz with angle_bits).")


def _expand_comparator_rotations(arch: Architecture, c: Circuit) raises -> Circuit:
    """rz_mode 'comparator': replace non-pi/4 RZ/P on Steane by the RUS circuit."""
    var out = c.empty_like()
    out.num_qubits = c.num_qubits
    var seq = List[Int]()
    for g in c.ops:
        if (g.kind == Op.RZ or g.kind == Op.P) and g.cond < 0 and not exact_rz_sequence(g.p0.value(), seq):
            _ = append_comparator_rz(out, g.q0, g.p0.value(), arch.rz_epsilon, arch.comparator_attempts)
            continue
        if g.kind == Op.MEASURE:
            out.measure(g.q0, g.clbit)
            out.ops[len(out.ops) - 1].cond = g.cond
            out.ops[len(out.ops) - 1].cond_val = g.cond_val
            out.ops[len(out.ops) - 1].cond_reg = g.cond_reg
            continue
        if (g.kind == Op.RZ or g.kind == Op.P or g.kind == Op.RX or g.kind == Op.RY or g.kind == Op.U) and g.cond >= 0:
            raise Error("QM-E726: a classically-conditioned non-Clifford rotation cannot use the comparator circuit. Hint: use rz_mode 'static' or make the rotation unconditional.")
        if g.kind == Op.RX or g.kind == Op.RY or g.kind == Op.U:
            raise Error("QM-E724: rz_mode 'comparator' expands RZ / P only. Hint: rewrite RX / RY / U as H / S conjugated RZ first.")
        out.append(g)
    return out^


def compile(arch: Architecture, circuit: Circuit) raises -> LogicalProgram:
    """Computational circuit -> logical program in the architecture's gate set."""
    arch.validate()
    check_may_encode(arch, circuit)
    var src = circuit.copy()
    if arch.kind == "steane" and arch.rz_mode == "comparator":
        src = _expand_comparator_rotations(arch, circuit)
    var cmp = _Compiler(arch, src)
    cmp.classical = _classical_qubits(src)
    cmp.lp.num_clbits = src.num_clbits
    for g in src.ops:
        var k = g.kind
        if k == Op.BARRIER or k == Op.TICK:
            cmp.lp.barrier()
            continue
        # current condition (copied to every logical op of the decomposition)
        cmp.cur_kind = 0
        if g.cond >= 0:
            if g.cond_reg >= 0:
                cmp.cur_kind = 2
                cmp.cur_bits = List[Int]()
                cmp.cur_vals = List[Int]()
                var cnt = src.cond_table[g.cond_reg]
                for j in range(cnt):
                    cmp.cur_bits.append(src.cond_table[g.cond_reg + 1 + 2 * j])
                    cmp.cur_vals.append(src.cond_table[g.cond_reg + 2 + 2 * j])
            else:
                cmp.cur_kind = 1
                cmp.cur_cond = g.cond
                cmp.cur_val = g.cond_val
        # classical-only qubits: value propagation
        if op_arity(k) >= 1 and g.q0 >= 0 and cmp.classical[g.q0]:
            var q = g.q0
            if k == Op.RESET:
                if g.cond >= 0:
                    raise Error("QM-E726: conditioned reset of a classical qubit. Hint: reset unconditionally.")
                cmp.cval[q] = CExpr()
            elif k == Op.X:
                var t: CTerm
                if cmp.cur_kind == 0:
                    t = CTerm.one()
                    cmp.cval[q].xor_term(t)
                elif cmp.cur_kind == 1:
                    cmp.cval[q].xor_term(CTerm.clbit(cmp.cur_cond))
                    if cmp.cur_val == 0:
                        cmp.cval[q].xor_term(CTerm.one())
                else:
                    cmp.cval[q].xor_term(CTerm.minterm(cmp.cur_bits, cmp.cur_vals))
            elif k == Op.MEASURE:
                if g.cond >= 0:
                    raise Error("QM-E726: conditioned measurement of a classical qubit. Hint: measure unconditionally.")
                cmp.lp.set_bit(g.clbit, cmp.cval[q])
            continue
        if k == Op.MEASURE:
            if g.cond >= 0:
                raise Error("QM-E726: classically-conditioned measurements are not supported by the encoder. Hint: measure unconditionally and condition the following gates.")
            var h = cmp.q(g.q0)
            cmp.lp.measure(h, g.clbit)
            cmp.handle[g.q0] = -1
            cmp.measured[g.q0] = g.clbit
            continue
        if k == Op.RESET:
            if g.cond >= 0:
                raise Error("QM-E726: classically-conditioned reset is not supported by the encoder. Hint: reset unconditionally.")
            if cmp.handle[g.q0] >= 0:
                cmp.lp.free(cmp.handle[g.q0])
                cmp.handle[g.q0] = -1
            cmp.measured[g.q0] = -1
            continue
        if not op_is_unitary(k):
            raise Error("QM-E730: '" + op_name(k) + "' cannot be encoded.")
        cmp.instruction(g)
    if cmp.rz_notes > 0:
        cmp.lp.notes.append("static Clifford+T synthesis of " + String(cmp.rz_notes) + " RZ gate(s); max approximation error " + String(cmp.max_rz_err) + " (epsilon " + String(arch.rz_epsilon) + ", T budget " + String(arch.rz_max_t) + ")")
    cmp.lp.validate()
    return cmp.lp.copy()


def encode(arch: Architecture, circuit: Circuit) raises -> EncodedProgram:
    """compile + implement_ops (Guppy FT `SteaneInstance.encode`)."""
    var lp = compile(arch, circuit)
    return implement_ops(arch, lp)
