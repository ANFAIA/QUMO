"""Pauli-based computation: `to_pbc` (Litinski, arXiv:1808.02892).

Commuting every Clifford gate to the end turns a circuit into

    U = C . M_m ... R_2 . M_1 . R_1

a time-ordered list of Pauli-product rotations `R = exp(-i angle/2 P)` and
Pauli-product measurements `M`, in the *input* frame, plus one final Clifford C.
QUMO's rotation convention gives Litinski's pi/8 rotation (T) as `angle = pi/4`.

A Pauli product can act on any number of qubits and `Instruction` holds three,
so the result is a side structure, `PauliProgram`, not new opcodes: nothing else
in the stack needs to know about it. `to_circuit` turns it back into gates (basis
change, CX ladder, rz or measure, undo) for simulation and equivalence checks.
"""

from std.math import pi
from std.bit import pop_count
from qumo.ir.gates import Op, op_name, op_is_unitary, op_is_annotation
from qumo.ir.circuit import Circuit, Instruction, QUARTER_TURN_LIMIT
from qumo.compiler.pauli_simp import _Extract, _process
from qumo.core.json import JsonWriter


struct PauliOp(Copyable):
    """A rotation (`clbit` < 0) or a measurement of `sign * P` into `clbit`."""

    var x: List[UInt64]
    var z: List[UInt64]
    var angle: Float64  # rotation angle, 0 for a measurement
    var sign: Int  # measurement sign, +1 or -1
    var clbit: Int

    def __init__(out self, var x: List[UInt64], var z: List[UInt64], angle: Float64, sign: Int, clbit: Int):
        self.x = x^
        self.z = z^
        self.angle = angle
        self.sign = sign
        self.clbit = clbit

    def is_measurement(self) -> Bool:
        return self.clbit >= 0

    def weight(self) -> Int:
        var w = 0
        for i in range(len(self.x)):
            w += Int(pop_count(self.x[i] | self.z[i]))
        return w

    def letter(self, q: Int) -> Int:
        """0 I, 1 X, 2 Y, 3 Z."""
        var bit = UInt64(1) << UInt64(q % 64)
        var xb = (self.x[q // 64] & bit) != 0
        var zb = (self.z[q // 64] & bit) != 0
        return 2 if xb and zb else (1 if xb else (3 if zb else 0))

    def label(self, n: Int) -> String:
        """Pauli string, highest qubit leftmost (Qiskit order)."""
        var s = String("")
        var q = n - 1
        while q >= 0:
            var l = self.letter(q)
            s += "IXYZ"[byte=l:l + 1]
            q -= 1
        return s^

    def commutes(self, o: PauliOp) -> Bool:
        var c = 0
        for i in range(len(self.x)):
            c += Int(pop_count((self.x[i] & o.z[i]) ^ (self.z[i] & o.x[i])))
        return c % 2 == 0


struct PauliProgram(Copyable):
    var num_qubits: Int
    var num_clbits: Int
    var ops: List[PauliOp]
    var clifford: Circuit  # the final Clifford, empty after fix_clifford=False
    var global_phase: Float64

    def __init__(out self, num_qubits: Int, num_clbits: Int):
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.ops = List[PauliOp]()
        self.clifford = Circuit(num_qubits)
        self.global_phase = 0.0

    def counts(self) -> Dict[String, Int]:
        """rotations, pi8 (odd multiples of pi/4), arbitrary, measurements, max_weight, layers."""
        var d = Dict[String, Int]()
        var rot = 0
        var pi8 = 0
        var ppm = 0
        var w = 0
        for op in self.ops:
            w = max(w, op.weight())
            if op.is_measurement():
                ppm += 1
                continue
            rot += 1
            var r = op.angle / (pi / 4)
            if not (abs(r) < 2.0 * QUARTER_TURN_LIMIT):
                continue  # not a multiple of pi/8 that anyone can tell
            var m = Int(round(r))
            if abs(r - Float64(m)) < 1e-9 and m % 2 != 0:
                pi8 += 1
        d["rotations"] = rot
        d["pi8"] = pi8
        d["arbitrary"] = rot - pi8
        d["measurements"] = ppm
        d["max_weight"] = w
        d["layers"] = self.layers()
        return d^

    def layers(self) -> Int:
        """Parallel layers of mutually commuting operations, ASAP."""
        var level = List[Int](capacity=len(self.ops))
        var depth = 0
        for i in range(len(self.ops)):
            var l = 1
            var j = i - 1
            while j >= 0:
                if not self.ops[i].commutes(self.ops[j]):
                    l = max(l, level[j] + 1)
                j -= 1
            level.append(l)
            depth = max(depth, l)
        return depth

    def to_circuit(self) -> Circuit:
        """Equivalent gate-level circuit."""
        var c = Circuit(self.num_qubits, self.num_clbits, "pbc")
        c.global_phase = self.global_phase
        for op in self.ops:
            _emit(c, op, self.num_qubits)
        c.compose(self.clifford)
        c.global_phase = self.global_phase
        return c^

    def write_json(self, mut w: JsonWriter):
        w.begin_object()
        w.kv_int("num_qubits", self.num_qubits)
        w.kv_int("num_clbits", self.num_clbits)
        w.key("ops")
        w.begin_array()
        for op in self.ops:
            w.begin_object()
            w.kv_string("pauli", op.label(self.num_qubits))
            if op.is_measurement():
                w.kv_string("kind", "measure")
                w.kv_int("sign", op.sign)
                w.kv_int("clbit", op.clbit)
            else:
                w.kv_string("kind", "rotate")
                w.kv_float("angle", op.angle)
            w.end_object()
        w.end_array()
        w.key("counts")
        w.begin_object()
        var d = self.counts()
        for kv in d.items():
            w.kv_int(kv.key, kv.value)
        w.end_object()
        w.kv_int("final_clifford_gates", len(self.clifford))
        w.end_object()


def _emit(mut c: Circuit, op: PauliOp, n: Int):
    var qs = List[Int]()
    for q in range(n):
        if op.letter(q) != 0:
            qs.append(q)
    if len(qs) == 0:
        return
    for q in qs:
        var l = op.letter(q)
        if l == 1:
            c.h(q)
        elif l == 2:
            c.rx(pi / 2, q)
    var last = qs[len(qs) - 1]
    for i in range(len(qs) - 1):
        c.cx(qs[i], last)
    if op.is_measurement():
        if op.sign < 0:
            c.x(last)
        c.measure(last, op.clbit)
        if op.sign < 0:
            c.x(last)
    else:
        c.rz(op.angle, last)
    var i = len(qs) - 2
    while i >= 0:
        c.cx(qs[i], last)
        i -= 1
    for q in qs:
        var l = op.letter(q)
        if l == 1:
            c.h(q)
        elif l == 2:
            c.rx(-pi / 2, q)


def to_pbc(c: Circuit, fix_clifford: Bool = True) raises -> PauliProgram:
    """Pauli-based form of a bound circuit without resets or classical conditions.

    `fix_clifford=False` drops the final Clifford: right when everything of
    interest is measured before it, as in Litinski's scheme."""
    if c.is_parametric():
        raise Error("QM-E623: to_pbc needs bound parameters. Hint: call bind() first.")
    var n = c.num_qubits
    var out = PauliProgram(n, c.num_clbits)
    var ex = _Extract(n)
    # gadget index -> measured clbit (rotations and measurements share the sweep)
    var clbit_of = List[Int]()
    for g in c.ops:
        if op_is_annotation(g.kind):
            if g.kind == Op.DETECTOR or g.kind == Op.OBSERVABLE:
                raise Error("QM-E623: to_pbc does not take detectors or observables. Hint: remove the QEC annotations first.")
            continue
        if g.cond >= 0:
            raise Error("QM-E623: to_pbc does not take classically conditioned gates. Hint: Pauli-based form is defined here for feed-forward-free circuits.")
        if g.kind == Op.MEASURE:
            var before = ex.gad.count()
            ex.rotation(g.q0, 3, -1, 0, 1.0)
            while len(clbit_of) < before:
                clbit_of.append(-1)
            clbit_of.append(g.clbit)
            continue
        if not op_is_unitary(g.kind):
            raise Error("QM-E623: to_pbc does not take '" + op_name(g.kind) + "'. Hint: use unitary gates and measurements only.")
        if not _process(ex, g, 0):
            raise Error("QM-E623: to_pbc cannot extract '" + op_name(g.kind) + "'. Hint: translate to Clifford + rotations first.")
    while len(clbit_of) < ex.gad.count():
        clbit_of.append(-1)
    var w = ex.gad.w
    for i in range(ex.gad.count()):
        var x = List[UInt64](capacity=w)
        var z = List[UInt64](capacity=w)
        for j in range(w):
            x.append(ex.gad.x[i * w + j])
            z.append(ex.gad.z[i * w + j])
        var angle = ex.gad.angle[i]
        if clbit_of[i] >= 0:
            out.ops.append(PauliOp(x^, z^, 0.0, 1 if angle > 0 else -1, clbit_of[i]))
        else:
            out.ops.append(PauliOp(x^, z^, angle, 1, -1))
    out.global_phase = c.global_phase + ex.phase
    if fix_clifford:
        for g in ex.rem:
            out.clifford.append(g)
    return out^
