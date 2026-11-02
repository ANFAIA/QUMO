"""QUMO circuit IR.

Design (best-of synthesis):
  * Flat, cache-friendly instruction list (struct-of-POD), like Qiskit's Rust
    `CircuitData` - no Python objects per gate and no pointer-chasing DAG;
    compiler passes work on the list directly.
  * Symbolic affine parameters (`Param`) bound without allocation.
  * Stim-style measurement records + DETECTOR / OBSERVABLE annotations and
    noise channels live in the same IR, so the QEC stack and the algorithm
    stack share one representation (no Cirq<->Stim conversion step).
  * Q#-style functors: `inverse()` (Adjoint) and `controlled()` are derived
    automatically.
  * Explicit `global_phase` so compiler rewrites stay exactly unitary-equivalent.
"""

from std.math import isinf, isnan, pi
from qumo.ir.gates import (
    Op,
    op_name,
    op_arity,
    op_num_params,
    op_is_unitary,
    op_is_noise,
    op_is_pauli_noise,
    op_is_annotation,
    op_is_clifford,
    op_inverse_kind,
    op_is_controlled_1q,
)
from qumo.ir.param import Param
from qumo.ir.draw import draw_circuit
from qumo.ir.kraus import KrausChannel

# Largest |theta / (pi/2)| read as a count of quarter turns. Past it the rounding of
# the ratio alone is more than the 1e-9 tolerance of `quarter_turns`, so a ratio that
# comes out whole no longer says the angle is near a multiple of pi/2: at 2^51, with
# one fractional bit left, some such angles are 0.57 rad away from every one.
comptime QUARTER_TURN_LIMIT = 4194304.0  # 2^22


def check_finite_values(values: List[Float64]) raises:
    """Input values on entry: a NaN would otherwise run on as a NaN state."""
    for i in range(len(values)):
        if isnan(values[i]) or isinf(values[i]):
            raise Error("QM-E482: input value " + String(i) + " is " + String(values[i]) + ". Hint: pass a finite number.")


def check_shots(shots: Int) raises:
    if shots < 0:
        raise Error("QM-E913: argument 'shots' = " + String(shots) + " is out of range. Hint: ask for at least one shot.")


@fieldwise_init
struct Instruction(ImplicitlyCopyable, Writable):
    var kind: Int
    var q0: Int
    var q1: Int
    var q2: Int
    var p0: Param
    var p1: Param
    var p2: Param
    var clbit: Int  # MEASURE: classical target bit
    var cond: Int  # classical condition bit (-1 = unconditional); for register
    #                conditions this is the first bit of the register, so
    #                `cond >= 0` always means "classically conditioned"
    var cond_val: Int
    var cond_reg: Int  # -1, or offset into Circuit.cond_table (register condition)

    @staticmethod
    def make(kind: Int, q0: Int, q1: Int = -1, q2: Int = -1,
             p0: Param = 0.0, p1: Param = 0.0, p2: Param = 0.0) -> Instruction:
        return Instruction(kind, q0, q1, q2, p0, p1, p2, -1, -1, 0, -1)

    def arity(self) -> Int:
        return op_arity(self.kind)

    def qubit(self, i: Int) -> Int:
        if i == 0: return self.q0
        if i == 1: return self.q1
        return self.q2

    def acts_on(self, q: Int) -> Bool:
        var a = op_arity(self.kind)
        if a >= 1 and self.q0 == q: return True
        if a >= 2 and self.q1 == q: return True
        if a >= 3 and self.q2 == q: return True
        return False

    def param(self, i: Int) -> Param:
        if i == 0: return self.p0
        if i == 1: return self.p1
        return self.p2

    def is_parametric(self) -> Bool:
        var n = op_num_params(self.kind)
        return (n >= 1 and self.p0.is_symbolic()) or (n >= 2 and self.p1.is_symbolic()) or (n >= 3 and self.p2.is_symbolic())

    def is_clifford(self) -> Bool:
        """Clifford check including rotations by multiples of pi/2."""
        if op_is_clifford(self.kind):
            return True
        var k = self.kind
        if k == Op.RX or k == Op.RY or k == Op.RZ or k == Op.P or k == Op.RZZ or k == Op.RXX or k == Op.RYY:
            if self.p0.is_symbolic():
                return False
            var r = self.p0.value() / (pi / 2)
            return abs(r) < QUARTER_TURN_LIMIT and abs(r - round(r)) < 1e-12
        return False

    def write_to(self, mut w: Some[Writer]):
        if self.cond_reg >= 0:
            w.write("if(creg-condition#", self.cond_reg, ") ")
        elif self.cond >= 0:
            w.write("if(c", self.cond, "==", self.cond_val, ") ")
        w.write(op_name(self.kind))
        var np = op_num_params(self.kind)
        if np > 0:
            w.write("(", self.p0)
            if np > 1: w.write(", ", self.p1)
            if np > 2: w.write(", ", self.p2)
            w.write(")")
        var a = op_arity(self.kind)
        if a >= 1: w.write(" q", self.q0)
        if a >= 2: w.write(", q", self.q1)
        if a >= 3: w.write(", q", self.q2)
        if self.kind == Op.MEASURE:
            w.write(" -> c", self.clbit)


struct Circuit(Copyable, Writable, Sized):
    var name: String
    var num_qubits: Int
    var num_clbits: Int
    var ops: List[Instruction]
    var param_names: List[String]
    var records: List[Int]  # flat measurement-record targets of detectors/observables
    var cond_table: List[Int]  # register conditions: [count, bit0, val0, bit1, val1, ...]
    var kraus: List[KrausChannel]  # side table of Op.KRAUS1 / Op.KRAUS2
    var noise_table: List[Float64]  # 15 probabilities per Op.PAULI_CHANNEL_2
    var pauli_table: List[Int]  # qubit, letter (1 X, 2 Y, 3 Z) pairs per Op.CORRELATED_ERROR
    var num_measurements: Int
    var num_detectors: Int
    var num_observables: Int
    var global_phase: Float64

    def __init__(out self, num_qubits: Int, num_clbits: Int = 0, name: String = "circuit"):
        self.name = name
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.ops = List[Instruction]()
        self.param_names = List[String]()
        self.records = List[Int]()
        self.cond_table = List[Int]()
        self.kraus = List[KrausChannel]()
        self.noise_table = List[Float64]()
        self.pauli_table = List[Int]()
        self.num_measurements = 0
        self.num_detectors = 0
        self.num_observables = 0
        self.global_phase = 0.0

    def __len__(self) -> Int:
        return len(self.ops)

    def empty_like(self) -> Circuit:
        """Same registers and parameter table, no instructions."""
        var c = Circuit(self.num_qubits, self.num_clbits, self.name)
        c.param_names = self.param_names.copy()
        c.cond_table = self.cond_table.copy()
        c.kraus = self.kraus.copy()
        c.noise_table = self.noise_table.copy()
        c.pauli_table = self.pauli_table.copy()
        c.global_phase = self.global_phase
        return c^

    # ------------------------------------------------------------------ params
    def parameter(mut self, name: String) -> Param:
        """Declare (or fetch) a named symbolic parameter."""
        for i in range(len(self.param_names)):
            if self.param_names[i] == name:
                return Param.symbol(i)
        self.param_names.append(name)
        return Param.symbol(len(self.param_names) - 1)

    def parameters(mut self, prefix: String, count: Int) -> List[Param]:
        var out = List[Param]()
        for i in range(count):
            out.append(self.parameter(prefix + "[" + String(i) + "]"))
        return out^

    def num_params(self) -> Int:
        return len(self.param_names)

    # ------------------------------------------------------------------ append
    def append(mut self, inst: Instruction):
        self.ops.append(inst)

    def add(mut self, kind: Int, q0: Int, q1: Int = -1, q2: Int = -1,
            p0: Param = 0.0, p1: Param = 0.0, p2: Param = 0.0):
        self.ops.append(Instruction.make(kind, q0, q1, q2, p0, p1, p2))

    # 1q fixed
    def id(mut self, q: Int): self.add(Op.I, q)
    def x(mut self, q: Int): self.add(Op.X, q)
    def y(mut self, q: Int): self.add(Op.Y, q)
    def z(mut self, q: Int): self.add(Op.Z, q)
    def h(mut self, q: Int): self.add(Op.H, q)
    def s(mut self, q: Int): self.add(Op.S, q)
    def sdg(mut self, q: Int): self.add(Op.SDG, q)
    def t(mut self, q: Int): self.add(Op.T, q)
    def tdg(mut self, q: Int): self.add(Op.TDG, q)
    def sx(mut self, q: Int): self.add(Op.SX, q)
    def sxdg(mut self, q: Int): self.add(Op.SXDG, q)
    # 1q parametric
    def rx(mut self, theta: Param, q: Int): self.add(Op.RX, q, p0=theta)
    def ry(mut self, theta: Param, q: Int): self.add(Op.RY, q, p0=theta)
    def rz(mut self, theta: Param, q: Int): self.add(Op.RZ, q, p0=theta)
    def p(mut self, lam: Param, q: Int): self.add(Op.P, q, p0=lam)
    def u(mut self, theta: Param, phi: Param, lam: Param, q: Int):
        self.add(Op.U, q, p0=theta, p1=phi, p2=lam)
    # 2q
    def cx(mut self, c: Int, t: Int): self.add(Op.CX, c, t)
    def cy(mut self, c: Int, t: Int): self.add(Op.CY, c, t)
    def cz(mut self, a: Int, b: Int): self.add(Op.CZ, a, b)
    def ch(mut self, c: Int, t: Int): self.add(Op.CH, c, t)
    def swap(mut self, a: Int, b: Int): self.add(Op.SWAP, a, b)
    def iswap(mut self, a: Int, b: Int): self.add(Op.ISWAP, a, b)
    def ecr(mut self, a: Int, b: Int): self.add(Op.ECR, a, b)
    def cp(mut self, lam: Param, c: Int, t: Int): self.add(Op.CP, c, t, p0=lam)
    def crx(mut self, theta: Param, c: Int, t: Int): self.add(Op.CRX, c, t, p0=theta)
    def cry(mut self, theta: Param, c: Int, t: Int): self.add(Op.CRY, c, t, p0=theta)
    def crz(mut self, theta: Param, c: Int, t: Int): self.add(Op.CRZ, c, t, p0=theta)
    def rxx(mut self, theta: Param, a: Int, b: Int): self.add(Op.RXX, a, b, p0=theta)
    def ryy(mut self, theta: Param, a: Int, b: Int): self.add(Op.RYY, a, b, p0=theta)
    def rzz(mut self, theta: Param, a: Int, b: Int): self.add(Op.RZZ, a, b, p0=theta)
    # 3q
    def ccx(mut self, c0: Int, c1: Int, t: Int): self.add(Op.CCX, c0, c1, t)
    def cswap(mut self, c: Int, a: Int, b: Int): self.add(Op.CSWAP, c, a, b)
    def ccz(mut self, a: Int, b: Int, c: Int): self.add(Op.CCZ, a, b, c)

    # non-unitary
    def measure(mut self, q: Int, c: Int):
        """Measure qubit q into classical bit c (auto-grows the creg)."""
        if c >= self.num_clbits:
            self.num_clbits = c + 1
        var inst = Instruction.make(Op.MEASURE, q, self.num_measurements)
        inst.clbit = c
        self.ops.append(inst)
        self.num_measurements += 1

    def measure_all(mut self):
        var base = self.num_clbits
        for q in range(self.num_qubits):
            self.measure(q, base + q)

    def reset(mut self, q: Int): self.add(Op.RESET, q)
    def barrier(mut self): self.add(Op.BARRIER, -1)
    def tick(mut self): self.add(Op.TICK, -1)

    def c_if(mut self, clbit: Int, value: Int):
        """Condition the most recently appended instruction on a classical bit."""
        self.ops[len(self.ops) - 1].cond = clbit
        self.ops[len(self.ops) - 1].cond_val = value

    def c_if_register(mut self, bits: List[Int], values: List[Int]) raises:
        """Condition the most recent instruction on several classical bits
        (OpenQASM 2 `if(c==N)`): runs only if every bits[i] == values[i]."""
        if len(bits) == 0 or len(bits) != len(values):
            raise Error("QM-E004: register condition needs matching non-empty bit/value lists.")
        var off = len(self.cond_table)
        self.cond_table.append(len(bits))
        for i in range(len(bits)):
            if bits[i] >= self.num_clbits:
                self.num_clbits = bits[i] + 1
            self.cond_table.append(bits[i])
            self.cond_table.append(values[i])
        ref g = self.ops[len(self.ops) - 1]
        g.cond = bits[0]
        g.cond_val = values[0]
        g.cond_reg = off

    def condition_holds(self, g: Instruction, creg: Int) -> Bool:
        """Evaluate g's classical condition against a register value (bit c = clbit c)."""
        if g.cond < 0:
            return True
        if g.cond_reg < 0:
            return ((creg >> g.cond) & 1) == g.cond_val
        var n = self.cond_table[g.cond_reg]
        for i in range(n):
            var b = self.cond_table[g.cond_reg + 1 + 2 * i]
            var v = self.cond_table[g.cond_reg + 2 + 2 * i]
            if ((creg >> b) & 1) != v:
                return False
        return True

    # noise
    def depolarize1(mut self, p: Float64, q: Int): self.add(Op.DEPOLARIZE1, q, p0=p)
    def depolarize2(mut self, p: Float64, a: Int, b: Int): self.add(Op.DEPOLARIZE2, a, b, p0=p)
    def x_error(mut self, p: Float64, q: Int): self.add(Op.X_ERROR, q, p0=p)
    def y_error(mut self, p: Float64, q: Int): self.add(Op.Y_ERROR, q, p0=p)
    def z_error(mut self, p: Float64, q: Int): self.add(Op.Z_ERROR, q, p0=p)
    def pauli_channel_1(mut self, px: Float64, py: Float64, pz: Float64, q: Int):
        self.add(Op.PAULI_CHANNEL_1, q, p0=px, p1=py, p2=pz)

    def pauli_channel_2(mut self, probs: List[Float64], a: Int, b: Int) raises:
        """Disjoint two-qubit Pauli channel; `probs` in Stim's order IX, IY, IZ, XI,
        XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ (first letter on qubit a)."""
        if len(probs) != 15:
            raise Error("QM-E732: pauli_channel_2 needs 15 probabilities, got " + String(len(probs)) + ". Hint: Stim's order: IX, IY, IZ, XI, XX, ..., ZZ.")
        var offset = len(self.noise_table)
        for p in probs:
            self.noise_table.append(p)
        self.add(Op.PAULI_CHANNEL_2, a, b, p0=Float64(offset))

    def correlated_error(mut self, p: Float64, qubits: List[Int], letters: List[Int], else_branch: Bool = False) raises:
        """Stim E / ELSE_CORRELATED_ERROR: with probability p, a Pauli product on
        any number of qubits. `letters`: 1 X, 2 Y, 3 Z. An `else_branch` only fires
        when no error of its chain fired."""
        if len(qubits) != len(letters) or len(qubits) == 0:
            raise Error("QM-E732: correlated_error needs one letter per qubit. Hint: e.g. qubits [0, 3], letters [1, 3] for X0 Z3.")
        var offset = len(self.pauli_table)
        for i in range(len(qubits)):
            self.pauli_table.append(qubits[i])
            self.pauli_table.append(letters[i])
        self.add(Op.CORRELATED_ERROR, offset, len(qubits), 1 if else_branch else 0, p0=p)

    def apply_channel(mut self, channel: KrausChannel, q0: Int, q1: Int = -1) raises:
        """General noise channel on 1 or 2 qubits (density matrix, trajectories)."""
        channel.validate()
        if channel.num_qubits != (1 if q1 < 0 else 2):
            raise Error("QM-E414: channel '" + channel.name + "' acts on " + String(channel.num_qubits) + " qubit(s). Hint: pass that many qubits.")
        var index = len(self.kraus)
        self.kraus.append(channel.copy())
        self.add(Op.KRAUS1 if q1 < 0 else Op.KRAUS2, q0, q1, p0=Float64(index))

    # QEC annotations
    def detector(mut self, recs: List[Int]):
        """Detector over absolute measurement-record indices."""
        var off = len(self.records)
        for r in recs:
            self.records.append(r)
        self.add(Op.DETECTOR, off, len(recs), self.num_detectors)
        self.num_detectors += 1

    def observable(mut self, index: Int, recs: List[Int]):
        var off = len(self.records)
        for r in recs:
            self.records.append(r)
        self.add(Op.OBSERVABLE, off, len(recs), index)
        if index >= self.num_observables:
            self.num_observables = index + 1

    def last_measurement(self, back: Int = 1) -> Int:
        """Absolute record index of the `back`-th most recent measurement (Stim rec[-k])."""
        return self.num_measurements - back

    # ------------------------------------------------------------- composition
    def compose(mut self, other: Circuit, qubit_offset: Int = 0, clbit_offset: Int = 0):
        """Append `other` with its qubit i mapped to i + qubit_offset."""
        var qmap = List[Int]()
        for i in range(other.num_qubits):
            qmap.append(i + qubit_offset)
        self.compose_mapped(other, qmap, clbit_offset)

    def compose_mapped(mut self, other: Circuit, qmap: List[Int], clbit_offset: Int = 0):
        """Append `other`, remapping qubits by `qmap`; parameters merged by name."""
        var pmap = List[Int]()
        for name in other.param_names:
            pmap.append(self.parameter(name).index)
        var rec_base = self.num_measurements
        var kraus_base = len(self.kraus)
        for c in other.kraus:
            self.kraus.append(c.copy())
        var noise_base = len(self.noise_table)
        for p in other.noise_table:
            self.noise_table.append(p)
        var pauli_base = len(self.pauli_table)
        for i in range(0, len(other.pauli_table), 2):
            self.pauli_table.append(qmap[other.pauli_table[i]])
            self.pauli_table.append(other.pauli_table[i + 1])
        for inst in other.ops:
            var n = inst.copy()
            var a = op_arity(n.kind)
            if n.kind == Op.CORRELATED_ERROR:
                n.q0 += pauli_base
                self.ops.append(n)
                continue
            if n.kind == Op.PAULI_CHANNEL_2:
                n.p0 = n.p0.value() + Float64(noise_base)
            if n.kind == Op.DETECTOR or n.kind == Op.OBSERVABLE:
                var off = len(self.records)
                for j in range(inst.q1):
                    self.records.append(other.records[inst.q0 + j] + rec_base)
                n.q0 = off
                if n.kind == Op.DETECTOR:
                    n.q2 = self.num_detectors
                    self.num_detectors += 1
                elif n.q2 >= self.num_observables:
                    self.num_observables = n.q2 + 1
                self.ops.append(n)
                continue
            if a >= 1 and n.q0 >= 0: n.q0 = qmap[n.q0]
            if a >= 2: n.q1 = qmap[n.q1]
            if a >= 3: n.q2 = qmap[n.q2]
            if n.kind == Op.MEASURE:
                n.q1 = self.num_measurements
                self.num_measurements += 1
                n.clbit += clbit_offset
                if n.clbit >= self.num_clbits:
                    self.num_clbits = n.clbit + 1
            if n.cond >= 0:
                n.cond += clbit_offset
            if n.cond_reg >= 0:
                var off = len(self.cond_table)
                var cnt = other.cond_table[inst.cond_reg]
                self.cond_table.append(cnt)
                for j in range(cnt):
                    self.cond_table.append(other.cond_table[inst.cond_reg + 1 + 2 * j] + clbit_offset)
                    self.cond_table.append(other.cond_table[inst.cond_reg + 2 + 2 * j])
                n.cond_reg = off
            if n.kind == Op.KRAUS1 or n.kind == Op.KRAUS2:
                n.p0 = n.p0.value() + Float64(kraus_base)
            if n.p0.index >= 0: n.p0.index = pmap[n.p0.index]
            if n.p1.index >= 0: n.p1.index = pmap[n.p1.index]
            if n.p2.index >= 0: n.p2.index = pmap[n.p2.index]
            self.ops.append(n)
        self.global_phase += other.global_phase

    def inverse(self) raises -> Circuit:
        """Adjoint functor (Q# `Adjoint`): reversed order, each gate inverted."""
        var out = self.empty_like()
        out.global_phase = -self.global_phase
        var i = len(self.ops) - 1
        while i >= 0:
            var g = self.ops[i]
            i -= 1
            if op_is_annotation(g.kind):
                continue
            if not op_is_unitary(g.kind):
                raise Error("QM-E201: inverse() requires a unitary circuit; found '" + op_name(g.kind) + "'. Hint: remove measurements/noise or split the circuit.")
            var n = g.copy()
            n.kind = op_inverse_kind(g.kind)
            if g.kind == Op.U:
                n.p0 = -g.p0
                n.p1 = -g.p2
                n.p2 = -g.p1
            elif op_num_params(g.kind) == 1:
                n.p0 = -g.p0
            out.ops.append(n)
        return out^

    def power(self, k: Int) raises -> Circuit:
        """U^k for integer k (negative k uses the inverse)."""
        var base = self.copy() if k >= 0 else self.inverse()
        var out = self.empty_like()
        out.global_phase = 0.0
        for _ in range(abs(k)):
            out.compose(base)
        return out^

    def controlled(self, control: Int) raises -> Circuit:
        """Controlled functor (Q# `Controlled`): adds `control` as a new control qubit.

        Supported gate set covers everything needed by QPE / amplitude
        estimation / Hadamard tests; unsupported gates raise QM-E202 with a hint.
        """
        var nq = self.num_qubits
        if control >= nq:
            nq = control + 1
        var out = Circuit(nq, self.num_clbits, self.name + "_ctrl")
        out.param_names = self.param_names.copy()
        if self.global_phase != 0.0:
            out.p(self.global_phase, control)
        for g in self.ops:
            if op_is_annotation(g.kind):
                continue
            if g.cond >= 0:
                raise Error("QM-E202: controlled() cannot control a classically-conditioned '" + op_name(g.kind) + "'. Hint: control the gates first, then add the condition.")
            if g.acts_on(control):
                raise Error("QM-E203: control qubit " + String(control) + " is already used by the circuit.")
            var k = g.kind
            if k == Op.I:
                continue
            elif k == Op.X: out.cx(control, g.q0)
            elif k == Op.Y: out.cy(control, g.q0)
            elif k == Op.Z: out.cz(control, g.q0)
            elif k == Op.H: out.ch(control, g.q0)
            elif k == Op.S: out.cp(pi / 2, control, g.q0)
            elif k == Op.SDG: out.cp(-pi / 2, control, g.q0)
            elif k == Op.T: out.cp(pi / 4, control, g.q0)
            elif k == Op.TDG: out.cp(-pi / 4, control, g.q0)
            elif k == Op.P: out.cp(g.p0, control, g.q0)
            elif k == Op.RX: out.crx(g.p0, control, g.q0)
            elif k == Op.RY: out.cry(g.p0, control, g.q0)
            elif k == Op.RZ: out.crz(g.p0, control, g.q0)
            elif k == Op.SX:
                # SX = e^{i pi/4} RX(pi/2)
                out.crx(pi / 2, control, g.q0)
                out.p(pi / 4, control)
            elif k == Op.SXDG:
                out.crx(-pi / 2, control, g.q0)
                out.p(-pi / 4, control)
            elif k == Op.U:
                # qelib1 cu3 decomposition
                var th = g.p0
                var ph = g.p1
                var la = g.p2
                if th.is_symbolic() or ph.is_symbolic() or la.is_symbolic():
                    raise Error("QM-E202: controlled(U) needs bound parameters. Hint: call bind() first.")
                var tv = th.value()
                var pv = ph.value()
                var lv = la.value()
                out.p((lv + pv) / 2, control)
                out.p((lv - pv) / 2, g.q0)
                out.cx(control, g.q0)
                out.u(-tv / 2, 0.0, -(pv + lv) / 2, g.q0)
                out.cx(control, g.q0)
                out.u(tv / 2, pv, 0.0, g.q0)
            elif k == Op.CX: out.ccx(control, g.q0, g.q1)
            elif k == Op.CZ: out.ccz(control, g.q0, g.q1)
            elif k == Op.CY:
                out.sdg(g.q1)
                out.ccx(control, g.q0, g.q1)
                out.s(g.q1)
            elif k == Op.SWAP: out.cswap(control, g.q0, g.q1)
            elif k == Op.CP or k == Op.CRZ:
                if g.p0.is_symbolic():
                    raise Error("QM-E202: controlled(" + op_name(k) + ") needs bound parameters.")
                var l = g.p0.value()
                var a = g.q0
                var b = g.q1
                # doubly-controlled diagonal via angle identity a + c - (a xor c) = 2ac
                if k == Op.CRZ:
                    out.crz(l / 2, a, b)
                    out.cx(control, a)
                    out.crz(-l / 2, a, b)
                    out.cx(control, a)
                    out.crz(l / 2, control, b)
                else:
                    out.cp(l / 2, a, b)
                    out.cx(control, a)
                    out.cp(-l / 2, a, b)
                    out.cx(control, a)
                    out.cp(l / 2, control, b)
            else:
                raise Error("QM-E202: controlled() does not support gate '" + op_name(k) + "'. Hint: use qumo.ir.mc.controlled_by, which takes any unitary gate and any number of controls.")
        return out^

    # ---------------------------------------------------------------- binding
    def bind(self, values: List[Float64]) raises -> Circuit:
        """Replace every symbolic parameter by its numeric value."""
        if len(values) != len(self.param_names):
            raise Error("QM-E101: bind() got " + String(len(values)) + " values, circuit has " + String(len(self.param_names)) + " parameters " + self.param_names_str() + ".")
        check_finite_values(values)
        var out = self.copy()
        out.param_names = List[String]()
        for ref g in out.ops:
            if g.p0.is_symbolic(): g.p0 = Param(g.p0.resolve(values))
            if g.p1.is_symbolic(): g.p1 = Param(g.p1.resolve(values))
            if g.p2.is_symbolic(): g.p2 = Param(g.p2.resolve(values))
        return out^

    def param_names_str(self) -> String:
        var s = String("[")
        for i in range(len(self.param_names)):
            if i > 0: s += ", "
            s += self.param_names[i]
        return s + "]"

    # ---------------------------------------------------------------- analysis
    def count(self, kind: Int) -> Int:
        var n = 0
        for g in self.ops:
            if g.kind == kind:
                n += 1
        return n

    def num_unitary_gates(self) -> Int:
        var n = 0
        for g in self.ops:
            if op_is_unitary(g.kind) and g.kind != Op.I:
                n += 1
        return n

    def num_2q_gates(self) -> Int:
        var n = 0
        for g in self.ops:
            if op_is_unitary(g.kind) and op_arity(g.kind) == 2:
                n += 1
        return n

    def t_count(self) -> Int:
        var n = 0
        for g in self.ops:
            if g.kind == Op.T or g.kind == Op.TDG:
                n += 1
            elif (g.kind == Op.RZ or g.kind == Op.P) and not g.is_clifford():
                n += 1
        return n

    def depth(self, unitary_only: Bool = True) -> Int:
        """Circuit depth (longest path), barriers synchronise all qubits."""
        var level = List[Int](length=self.num_qubits, fill=0)
        var d = 0
        for g in self.ops:
            if g.kind == Op.BARRIER:
                var m = 0
                for q in range(self.num_qubits):
                    m = max(m, level[q])
                for q in range(self.num_qubits):
                    level[q] = m
                continue
            if op_is_annotation(g.kind) or op_is_noise(g.kind):
                continue
            if unitary_only and not op_is_unitary(g.kind):
                continue
            var a = op_arity(g.kind)
            var l = 0
            for j in range(a):
                l = max(l, level[g.qubit(j)])
            l += 1
            for j in range(a):
                level[g.qubit(j)] = l
            d = max(d, l)
        return d

    def is_clifford(self) -> Bool:
        for g in self.ops:
            if op_is_unitary(g.kind) and not g.is_clifford():
                return False
        return True

    def has_noise(self) -> Bool:
        for g in self.ops:
            if op_is_noise(g.kind):
                return True
        return False

    def has_table_noise(self) -> Bool:
        """PAULI_CHANNEL_2 or E: Pauli noise that only the CPU stabilizer stack takes."""
        for g in self.ops:
            if g.kind == Op.PAULI_CHANNEL_2 or g.kind == Op.CORRELATED_ERROR:
                return True
        return False

    def has_non_pauli_noise(self) -> Bool:
        for g in self.ops:
            if op_is_noise(g.kind) and not op_is_pauli_noise(g.kind):
                return True
        return False

    def is_dynamic(self) -> Bool:
        """True when measurement/reset/conditionals occur before the final layer."""
        var measured = List[Bool](length=self.num_qubits, fill=False)
        for g in self.ops:
            if g.cond >= 0 or g.kind == Op.RESET:
                return True
            if g.kind == Op.MEASURE:
                measured[g.q0] = True
                continue
            if op_is_unitary(g.kind) or op_is_noise(g.kind):
                for j in range(op_arity(g.kind)):
                    if measured[g.qubit(j)]:
                        return True
        return False

    def is_parametric(self) -> Bool:
        return len(self.param_names) > 0

    def validate(self) raises:
        """Structural validation with machine-readable error codes."""
        for i in range(len(self.ops)):
            var g = self.ops[i]
            var a = op_arity(g.kind)
            for j in range(a):
                var q = g.qubit(j)
                if q < 0 or q >= self.num_qubits:
                    raise Error("QM-E001: instruction #" + String(i) + " '" + op_name(g.kind) + "' uses qubit " + String(q) + " but circuit has " + String(self.num_qubits) + " qubits (valid 0.." + String(self.num_qubits - 1) + ").")
            if a >= 2 and g.q0 == g.q1:
                raise Error("QM-E002: instruction #" + String(i) + " '" + op_name(g.kind) + "' repeats qubit " + String(g.q0) + ".")
            if a >= 3 and (g.q2 == g.q0 or g.q2 == g.q1):
                raise Error("QM-E002: instruction #" + String(i) + " '" + op_name(g.kind) + "' repeats a qubit.")
            if g.kind == Op.CORRELATED_ERROR:
                for j in range(g.q1):
                    var q = self.pauli_table[g.q0 + 2 * j]
                    if q < 0 or q >= self.num_qubits:
                        raise Error("QM-E001: instruction #" + String(i) + " '" + op_name(g.kind) + "' uses qubit " + String(q) + " but circuit has " + String(self.num_qubits) + " qubits (valid 0.." + String(self.num_qubits - 1) + ").")
            if g.cond >= self.num_clbits:
                raise Error("QM-E003: instruction #" + String(i) + " conditions on clbit " + String(g.cond) + " beyond creg size " + String(self.num_clbits) + ".")

    def draw(self, width: Int = 80, style: String = "ascii") raises -> String:
        """Text drawing of the circuit (see qumo.ir.draw)."""
        return draw_circuit(self, width, style)

    def write_to(self, mut w: Some[Writer]):
        w.write("circuit ", self.name, " qubits=", self.num_qubits, " clbits=", self.num_clbits)
        if len(self.param_names) > 0:
            w.write(" params=", self.param_names_str())
        if self.global_phase != 0.0:
            w.write(" global_phase=", self.global_phase)
        w.write("\n")
        for g in self.ops:
            if g.kind == Op.DETECTOR or g.kind == Op.OBSERVABLE:
                w.write("  ", op_name(g.kind))
                if g.kind == Op.OBSERVABLE:
                    w.write("(", g.q2, ")")
                w.write(" rec[")
                for j in range(g.q1):
                    if j > 0: w.write(",")
                    w.write(self.records[g.q0 + j])
                w.write("]\n")
                continue
            if g.kind == Op.BARRIER or g.kind == Op.TICK:
                w.write("  ", op_name(g.kind), "\n")
                continue
            w.write("  ", g, "\n")
