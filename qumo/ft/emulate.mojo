"""Emulation of encoded programs, logical reference semantics and error-rate sweeps.

* `add_noise(circuit, noise, quiet)`: Stim-style circuit-level noise on the
  physical circuit (`qumo.qec.codes.NoiseParams`: DEPOLARIZE1/2 after every
  1-/2-qubit gate, X_ERROR before measurements and after resets). Noise ops
  inherit the classical condition of the gate they follow (a skipped gate
  causes no error). Classical-lane qubits (`EncodedProgram.lane_qubits`) are
  never noisy: they stand for classical registers.
* `run_encoded(prog, shots, seed, noise)`: per-shot trajectories on QUMO's
  tableau simulator (Clifford programs, any width) or on the state vector
  (non-Clifford programs such as T injection, <= 26 qubits), returning
  `LogicalCounts` over the logical output bits after post-selection on
  `prog.postselect` (detection flags, failed state factories).
* `logical_to_circuit(lp)` gives the *semantics* of a logical program as a
  plain circuit on logical qubits (handles re-use freed qubits), so a
  compiled program can be compared with its source without encoding;
  `reference_counts(circuit)` samples a computational circuit with QUMO's own
  samplers.
* `error_rate_point` / `error_rate_sweep`: logical error rate of an encoded
  deterministic program vs the error rate of the same unencoded program under
  the same circuit-level noise (the "logical vs physical" demo).

Idea from Guppy FT (Apache-2.0): `SteaneInstance.emulator` (Selene + Stim /
state vector) in src/guppyft/code/steane/encode.py and the examples
notebooks; here on QUMO's own simulators.

Error codes: QM-E729 (emulation request not supported: too wide for the state
vector, invalid shots).
"""

from std.math import sqrt

from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_arity, op_is_unitary, op_is_noise, op_is_annotation, matrix_1q
from qumo.sim.stabilizer import TableauSimulator
from qumo.sim.statevector import StateVector
from qumo.sim.runner import apply_instruction, sample
from qumo.sim.rng import Rng
from qumo.qec.codes import NoiseParams
from qumo.ft.classical import CExpr
from qumo.ft.logical import LogicalProgram, LOp
from qumo.ft.implement import EncodedProgram
from qumo.ft.arch import Architecture
from qumo.ft.encode import encode
from qumo.ai.json import JsonWriter


struct LogicalCounts(Copyable, Writable):
    var counts: Dict[String, Int]
    var shots: Int
    var accepted: Int
    var num_bits: Int
    var method: String

    def __init__(out self, num_bits: Int, method: String):
        self.counts = Dict[String, Int]()
        self.shots = 0
        self.accepted = 0
        self.num_bits = num_bits
        self.method = method

    def record(mut self, key: String):
        self.counts[key] = self.counts.get(key, 0) + 1
        self.accepted += 1

    def probability(self, key: String) -> Float64:
        return Float64(self.counts.get(key, 0)) / Float64(max(self.accepted, 1))

    def acceptance(self) -> Float64:
        return Float64(self.accepted) / Float64(max(self.shots, 1))

    def most_frequent(self) -> String:
        var best = String()
        var bc = -1
        for kv in self.counts.items():
            if kv.value > bc or (kv.value == bc and kv.key < best):
                bc = kv.value
                best = kv.key
        return best

    def keys(self) -> List[String]:
        var ks = List[String]()
        for k in self.counts.keys():
            ks.append(k)
        sort(ks)
        return ks^

    def write_json(self, mut w: JsonWriter):
        w.begin_object()
        w.key("counts")
        w.begin_object()
        for k in self.keys():
            w.kv_int(k, self.counts.get(k, 0))
        w.end_object()
        w.kv_int("shots", self.shots)
        w.kv_int("accepted", self.accepted)
        w.kv_float("acceptance", self.acceptance())
        w.kv_string("method", self.method)
        w.end_object()

    def write_to(self, mut w: Some[Writer]):
        w.write("{")
        var first = True
        for k in self.keys():
            if not first:
                w.write(", ")
            first = False
            w.write("\"", k, "\": ", self.counts.get(k, 0))
        w.write("} accepted ", self.accepted, "/", self.shots, " (", self.method, ")")


def key_of(creg: List[UInt8], bits: List[Int]) -> String:
    """Qiskit-style key: leftmost char = highest logical clbit; unwritten bits read 0."""
    var s = String()
    var i = len(bits) - 1
    while i >= 0:
        var b = bits[i]
        s += String("1") if (b >= 0 and creg[b] != 0) else String("0")
        i -= 1
    return s


def total_variation(a: LogicalCounts, b: LogicalCounts) -> Float64:
    var keys = a.keys()
    for k in b.keys():
        var seen = False
        for x in keys:
            if x == k:
                seen = True
        if not seen:
            keys.append(k)
    var tv = 0.0
    for k in keys:
        tv += abs(a.probability(k) - b.probability(k))
    return tv / 2.0


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
def _copy_cond(mut c: Circuit, g: Instruction):
    ref h = c.ops[len(c.ops) - 1]
    h.cond = g.cond
    h.cond_val = g.cond_val
    h.cond_reg = g.cond_reg


def add_noise(circuit: Circuit, noise: NoiseParams, quiet: List[Int] = List[Int]()) raises -> Circuit:
    noise.check()
    var q_ok = List[Bool](length=circuit.num_qubits, fill=True)
    for q in quiet:
        if q >= 0 and q < circuit.num_qubits:
            q_ok[q] = False
    var out = circuit.empty_like()
    var p1 = noise.after_clifford_depolarization
    var pm = noise.before_measure_flip_probability
    var pr = noise.after_reset_flip_probability
    for g in circuit.ops:
        var k = g.kind
        if k == Op.MEASURE:
            if pm > 0 and q_ok[g.q0]:
                out.x_error(pm, g.q0)
                _copy_cond(out, g)
            out.measure(g.q0, g.clbit)
            _copy_cond(out, g)
            continue
        if k == Op.DETECTOR or k == Op.OBSERVABLE:
            continue
        out.append(g)
        if k == Op.RESET:
            if pr > 0 and q_ok[g.q0]:
                out.x_error(pr, g.q0)
                _copy_cond(out, g)
        elif op_is_unitary(k) and k != Op.I and p1 > 0:
            var a = op_arity(k)
            if a == 1 and q_ok[g.q0]:
                out.depolarize1(p1, g.q0)
                _copy_cond(out, g)
            elif a == 2:
                if q_ok[g.q0] and q_ok[g.q1]:
                    out.depolarize2(p1, g.q0, g.q1)
                    _copy_cond(out, g)
            elif a == 3:
                for j in range(3):
                    if q_ok[g.qubit(j)]:
                        out.depolarize1(p1, g.qubit(j))
                        _copy_cond(out, g)
    out.num_clbits = circuit.num_clbits
    return out^


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------
def condition_holds(c: Circuit, g: Instruction, creg: List[UInt8]) -> Bool:
    if g.cond < 0:
        return True
    if g.cond_reg < 0:
        return Int(creg[g.cond]) == g.cond_val
    var n = c.cond_table[g.cond_reg]
    for i in range(n):
        var b = c.cond_table[g.cond_reg + 1 + 2 * i]
        var v = c.cond_table[g.cond_reg + 2 + 2 * i]
        if Int(creg[b]) != v:
            return False
    return True


def _pauli_sv(mut sv: StateVector, which: Int, q: Int):
    if which == 1:
        sv.apply_x(q)
    elif which == 2:
        sv.apply_1q(matrix_1q(Op.Y), q)
    elif which == 3:
        sv.apply_diag_1q(1.0, 0.0, -1.0, 0.0, q)


def _noise_sv(mut sv: StateVector, g: Instruction, mut rng: Rng):
    var k = g.kind
    var p = g.p0.value()
    if k == Op.X_ERROR:
        if rng.uniform() < p: _pauli_sv(sv, 1, g.q0)
    elif k == Op.Y_ERROR:
        if rng.uniform() < p: _pauli_sv(sv, 2, g.q0)
    elif k == Op.Z_ERROR:
        if rng.uniform() < p: _pauli_sv(sv, 3, g.q0)
    elif k == Op.DEPOLARIZE1:
        if rng.uniform() < p: _pauli_sv(sv, 1 + rng.below(3), g.q0)
    elif k == Op.DEPOLARIZE2:
        if rng.uniform() < p:
            var r = 1 + rng.below(15)
            _pauli_sv(sv, r & 3, g.q0)
            _pauli_sv(sv, r >> 2, g.q1)
    elif k == Op.PAULI_CHANNEL_1:
        var u = rng.uniform()
        var px = g.p0.value()
        var py = g.p1.value()
        var pz = g.p2.value()
        if u < px: _pauli_sv(sv, 1, g.q0)
        elif u < px + py: _pauli_sv(sv, 2, g.q0)
        elif u < px + py + pz: _pauli_sv(sv, 3, g.q0)


def sv_trajectory(c: Circuit, mut sv: StateVector, mut rng: Rng, mut creg: List[UInt8]) raises:
    """One shot on a state vector with List-based classical register (no 62-bit limit)."""
    var empty = List[Float64]()
    for g in c.ops:
        if g.cond >= 0 and not condition_holds(c, g, creg):
            continue
        var k = g.kind
        if op_is_unitary(k):
            apply_instruction(sv, g, empty)
        elif k == Op.MEASURE:
            var m = sv.measure(g.q0, rng)
            if g.clbit >= 0:
                creg[g.clbit] = UInt8(m)
        elif k == Op.RESET:
            sv.reset_qubit(g.q0, rng)
        elif op_is_noise(k):
            _noise_sv(sv, g, rng)


def _accepted(creg: List[UInt8], post: List[Int]) -> Bool:
    for b in post:
        if creg[b] != 0:
            return False
    return True


def run_physical(c: Circuit, outputs: List[Int], postselect: List[Int], shots: Int, seed: UInt64,
                 method: String = "auto") raises -> LogicalCounts:
    """Shots of a physical circuit; keys from `outputs`, post-selected on `postselect`."""
    if shots < 1:
        raise Error("QM-E729: shots must be >= 1. Hint: e.g. 1000.")
    var m = method
    if m == "auto":
        m = String("tableau") if c.is_clifford() else String("statevector")
    var res = LogicalCounts(len(outputs), m)
    var root = Rng(seed)
    if m == "tableau":
        if not c.is_clifford():
            raise Error("QM-E729: the circuit contains non-Clifford gates; the tableau method cannot run it. Hint: method='statevector' (<= 26 qubits) or method='auto'.")
        var sim = TableauSimulator(c.num_qubits, 0)
        for s in range(shots):
            sim.reset_state()
            sim.rng = root.split(s)
            sim.run(c)
            res.shots += 1
            if _accepted(sim.creg, postselect):
                res.record(key_of(sim.creg, outputs))
        return res^
    if c.num_qubits > 26:
        raise Error("QM-E729: " + String(c.num_qubits) + " physical qubits is too wide for state-vector trajectories (limit 26). Hint: use magic_prep='nonft' / fewer logical qubits, or emulate the Clifford part with method='tableau'.")
    var sv = StateVector(c.num_qubits)
    var creg = List[UInt8](length=max(c.num_clbits, 1), fill=0)
    for s in range(shots):
        sv.reset_to_zero()
        for i in range(len(creg)):
            creg[i] = 0
        var rng = root.split(s)
        sv_trajectory(c, sv, rng, creg)
        res.shots += 1
        if _accepted(creg, postselect):
            res.record(key_of(creg, outputs))
    return res^


def run_encoded(prog: EncodedProgram, shots: Int, seed: UInt64 = 1234,
                noise: NoiseParams = NoiseParams.noiseless(), method: String = "auto") raises -> LogicalCounts:
    """Emulate an encoded program (optionally with circuit-level noise)."""
    var c = prog.circuit.copy()
    var p = noise.after_clifford_depolarization + noise.before_measure_flip_probability + noise.after_reset_flip_probability
    if p > 0.0:
        c = add_noise(prog.circuit, noise, prog.lane_qubits)
    return run_physical(c, prog.outputs, prog.postselect, shots, seed, method)


def reference_counts(circuit: Circuit, shots: Int, seed: UInt64 = 1234,
                     noise: NoiseParams = NoiseParams.noiseless(), method: String = "auto") raises -> LogicalCounts:
    """Unencoded reference: the computational circuit itself (all clbits are outputs)."""
    var c = circuit.copy()
    var p = noise.after_clifford_depolarization + noise.before_measure_flip_probability + noise.after_reset_flip_probability
    if p > 0.0:
        c = add_noise(circuit, noise)
    var outs = List[Int]()
    for b in range(circuit.num_clbits):
        outs.append(b)
    return run_physical(c, outs, List[Int](), shots, seed, method)


# ---------------------------------------------------------------------------
# Logical semantics
# ---------------------------------------------------------------------------
def logical_to_circuit(lp: LogicalProgram) raises -> Circuit:
    """The logical program as a plain circuit on logical qubits (unencoded):
    T via a T gate, PREP_T via H T, INJECT_T(q, m) via T on q, CSET via a
    classical lane qubit, QEC cycles as identities."""
    lp.validate()
    var c = Circuit(1, lp.num_clbits, lp.name + "_logical")
    var qmap = List[Int](length=lp.num_logical, fill=-1)
    var free = List[Int]()
    var nq = 0
    var lane = -1
    for g in lp.ops:
        var k = g.kind
        if k == LOp.ALLOC or k == LOp.PREP_T:
            var q: Int
            if len(free) > 0:
                q = free.pop()
                c.reset(q)
            else:
                q = nq
                nq += 1
                c.num_qubits = max(c.num_qubits, nq)
            qmap[g.q0] = q
            if k == LOp.PREP_T:
                c.h(q)
                c.t(q)
            continue
        if k == LOp.FREE:
            free.append(qmap[g.q0])
            continue
        if k == LOp.QEC or k == LOp.BARRIER:
            continue
        if k == LOp.CSET:
            if lane < 0:
                lane = nq
                nq += 1
                c.num_qubits = max(c.num_qubits, nq)
            c.reset(lane)
            for t in lp.cexprs[g.q0].terms:
                c.x(lane)
                if t.kind == 0:
                    c.c_if(t.lits[0], 1)
                elif t.kind == 1:
                    c.c_if_register(t.bits(), t.vals())
            c.measure(lane, g.clbit)
            continue
        if k == LOp.MEASURE:
            c.measure(qmap[g.q0], g.clbit)
            free.append(qmap[g.q0])
            continue
        var a = qmap[g.q0]
        var b = qmap[g.q1] if g.q1 >= 0 else -1
        if k == LOp.X: c.x(a)
        elif k == LOp.Y: c.y(a)
        elif k == LOp.Z: c.z(a)
        elif k == LOp.H: c.h(a)
        elif k == LOp.S: c.s(a)
        elif k == LOp.SDG: c.sdg(a)
        elif k == LOp.T: c.t(a)
        elif k == LOp.TDG: c.tdg(a)
        elif k == LOp.CX: c.cx(a, b)
        elif k == LOp.CZ: c.cz(a, b)
        elif k == LOp.SWAP: c.swap(a, b)
        elif k == LOp.RZ: c.rz(g.angle, a)
        elif k == LOp.RX: c.rx(g.angle, a)
        elif k == LOp.RZZ: c.rzz(g.angle, a, b)
        elif k == LOp.RXX: c.rxx(g.angle, a, b)
        elif k == LOp.INJECT_T or k == LOp.INJECT_TDG:
            if k == LOp.INJECT_T:
                c.t(a)
            else:
                c.tdg(a)
            free.append(qmap[g.q1])
            continue
        if g.cond >= 0:
            if g.cond_reg >= 0:
                c.c_if_register(lp.condition_bits(g), lp.condition_vals(g))
            else:
                c.c_if(g.cond, g.cond_val)
    c.num_clbits = max(c.num_clbits, lp.num_clbits)
    return c^


# ---------------------------------------------------------------------------
# Logical vs physical error rate
# ---------------------------------------------------------------------------
@fieldwise_init
struct ErrorRatePoint(ImplicitlyCopyable, Writable):
    var p: Float64
    var physical_error: Float64  # unencoded program, same noise
    var logical_error: Float64  # encoded program, after post-selection
    var acceptance: Float64
    var shots: Int

    def write_to(self, mut w: Some[Writer]):
        w.write("p=", self.p, " physical=", self.physical_error, " logical=", self.logical_error, " acceptance=", self.acceptance)


def error_rate_point(arch: Architecture, circuit: Circuit, p: Float64, shots: Int, seed: UInt64 = 7) raises -> ErrorRatePoint:
    """Deterministic program: error = P(outcome != noiseless outcome)."""
    var prog = encode(arch, circuit)
    var ideal = reference_counts(circuit, 64, seed)
    if len(ideal.counts) != 1:
        raise Error("QM-E729: error_rate_point needs a deterministic program (noiseless outcome distribution has " + String(len(ideal.counts)) + " outcomes). Hint: measure in an eigenbasis of the final state.")
    var expected = ideal.most_frequent()
    var noise = NoiseParams(p, p, p, 0.0)
    var phys = reference_counts(circuit, shots, seed + 1, noise)
    var logi = run_encoded(prog, shots, seed + 2, noise)
    return ErrorRatePoint(p, 1.0 - phys.probability(expected), 1.0 - logi.probability(expected), logi.acceptance(), shots)


def error_rate_sweep(arch: Architecture, circuit: Circuit, ps: List[Float64], shots: Int, seed: UInt64 = 7) raises -> List[ErrorRatePoint]:
    var out = List[ErrorRatePoint]()
    for i in range(len(ps)):
        out.append(error_rate_point(arch, circuit, ps[i], shots, seed + UInt64(17 * i)))
    return out^


# ---------------------------------------------------------------------------
# Comparator-Rz check (unencoded, state vector)
# ---------------------------------------------------------------------------
@fieldwise_init
struct RzCheck(ImplicitlyCopyable, Writable):
    var theta: Float64
    var max_angle_error: Float64  # worst |phase - theta| over successful shots
    var max_eff_error: Float64  # worst |phase - effective_theta| (exactness of the circuit)
    var success_rate: Float64
    var shots: Int

    def write_to(self, mut w: Some[Writer]):
        w.write("RzCheck(theta=", self.theta, ", max_err=", self.max_angle_error, ", vs_effective=", self.max_eff_error,
                ", success=", self.success_rate, ")")


def _wrap(x: Float64) -> Float64:
    from std.math import pi, floor
    var y = x - 2.0 * pi * floor((x + pi) / (2.0 * pi))
    return y


def target_relative_phase(sv: StateVector, target: Int) raises -> Float64:
    """Relative phase arg(a1 / a0) of `target` when every other qubit is in a
    basis state (product state)."""
    from std.math import atan2
    var best = 0
    var bp = -1.0
    for i in range(sv.size()):
        var p = sv.re[i] * sv.re[i] + sv.im[i] * sv.im[i]
        if p > bp:
            bp = p
            best = i
    var i0 = best & ~(1 << target)
    var i1 = best | (1 << target)
    var r0 = sv.re[i0]
    var m0 = sv.im[i0]
    var r1 = sv.re[i1]
    var m1 = sv.im[i1]
    # a1 / a0 = a1 * conj(a0) / |a0|^2
    var re = r1 * r0 + m1 * m0
    var im = m1 * r0 - r1 * m0
    return atan2(im, re)


def comparator_rz_check(theta: Float64, epsilon: Float64, attempts: Int, shots: Int, seed: UInt64 = 11,
                        runtime_angle: Bool = False) raises -> RzCheck:
    """|+> -> comparator Rz(theta) on a state vector; checks the target phase
    on every successful shot."""
    from qumo.ft.rz import append_comparator_rz, comparator_angle, comparator_register_size, angle_register
    var c = Circuit(1, 0, "comparator_rz")
    c.h(0)
    var n = comparator_register_size(epsilon)
    if n < 3:
        n = 3
    var ang = comparator_angle(theta, n)
    var bits = List[Int]()
    if runtime_angle:
        bits = angle_register(c, ang)
    var info = append_comparator_rz(c, 0, theta, epsilon, attempts, bits)
    if c.num_qubits > 24:
        raise Error("QM-E729: comparator check needs " + String(c.num_qubits) + " qubits (> 24). Hint: use a larger epsilon.")
    var sv = StateVector(c.num_qubits)
    var creg = List[UInt8](length=c.num_clbits, fill=0)
    var root = Rng(seed)
    var ok = 0
    var worst = 0.0
    var worst_eff = 0.0
    for s in range(shots):
        sv.reset_to_zero()
        for i in range(len(creg)):
            creg[i] = 0
        var rng = root.split(s)
        sv_trajectory(c, sv, rng, creg)
        if creg[info.fail_clbit] != 0:
            continue
        ok += 1
        var ph = target_relative_phase(sv, 0)
        var e = abs(_wrap(ph - theta))
        var e2 = abs(_wrap(ph - ang.effective_theta))
        if e > worst:
            worst = e
        if e2 > worst_eff:
            worst_eff = e2
    return RzCheck(theta, worst, worst_eff, Float64(ok) / Float64(max(shots, 1)), shots)
