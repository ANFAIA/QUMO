"""QEC memory-experiment circuit generators (Stim `Circuit.generated` style).

Codes
  * `repetition_code_memory(d, rounds, noise)` - bit-flip repetition code,
    same layout as Stim's "repetition_code:memory" (data = even qubits).
  * `surface_code_memory(d, rounds, noise, basis)` - rotated surface code
    (d^2 data + d^2-1 measure qubits) with the standard hook-safe
    syndrome-extraction schedule: X checks use the "N" order
    (NE, SE, NW, SW) and Z checks the "Z" order (NE, NW, SE, SW), the same
    choice as Stim's generator, so a hook error on an X (Z) check is
    perpendicular to the X (Z) logical operator and the circuit distance is d.

Noise model parameters use Stim's names (Fowler-style circuit noise):
  after_clifford_depolarization    DEPOLARIZE1/2 after every H / CX
  before_measure_flip_probability  X_ERROR before every measurement
  after_reset_flip_probability     X_ERROR after every reset
  before_round_data_depolarization DEPOLARIZE1 on data at the start of a round
`NoiseParams.uniform(p)` sets all four to p (the convention of Stim/Sinter
benchmarks, "uniform circuit-level depolarizing noise").

Detectors compare each stabilizer with its previous round (first round: only
the stabilizers that are deterministic for the prepared basis), plus final
detectors built from the data-qubit measurements. Observable 0 is the
logical operator of the memory basis. `CodeCircuit` keeps the qubit and
detector coordinates so `to_stim()` emits QUBIT_COORDS / DETECTOR(x, y, t),
letting users paste QUMO circuits into Stim/Crumble/Sinter for cross-checks.

Error codes: QM-E706 (bad code parameters), QM-E711 (bad probabilities).
"""

from std.math import pi
from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary, op_is_noise
from qumo.ir.circuit import Circuit, Instruction
from qumo.sim.stabilizer import quarter_turns


@fieldwise_init
struct NoiseParams(ImplicitlyCopyable, Writable):
    var after_clifford_depolarization: Float64
    var before_measure_flip_probability: Float64
    var after_reset_flip_probability: Float64
    var before_round_data_depolarization: Float64

    @staticmethod
    def uniform(p: Float64) -> NoiseParams:
        return NoiseParams(p, p, p, p)

    @staticmethod
    def noiseless() -> NoiseParams:
        return NoiseParams(0.0, 0.0, 0.0, 0.0)

    def check(self) raises:
        var vals: List[Float64] = [
            self.after_clifford_depolarization, self.before_measure_flip_probability,
            self.after_reset_flip_probability, self.before_round_data_depolarization,
        ]
        for v in vals:
            if v < 0.0 or v > 1.0:
                raise Error("QM-E711: noise probability " + String(v) + " outside [0, 1]. Hint: NoiseParams.uniform(0.001) is a typical circuit-level noise strength.")

    def write_to(self, mut w: Some[Writer]):
        w.write("NoiseParams(after_clifford_depolarization=", self.after_clifford_depolarization,
                ", before_measure_flip_probability=", self.before_measure_flip_probability,
                ", after_reset_flip_probability=", self.after_reset_flip_probability,
                ", before_round_data_depolarization=", self.before_round_data_depolarization, ")")


struct CodeCircuit(Copyable, Writable):
    """A generated QEC circuit plus the metadata decoders/agents need."""

    var circuit: Circuit
    var code: String
    var basis: String
    var distance: Int
    var rounds: Int
    var noise: NoiseParams
    var qubit_coords: List[Float64]  # flat (x, y) per qubit
    var detector_coords: List[Float64]  # flat (x, y, t) per detector
    var data_qubits: List[Int]
    var x_checks: List[Int]
    var z_checks: List[Int]

    def __init__(out self, var circuit: Circuit, code: String, basis: String, distance: Int, rounds: Int, noise: NoiseParams):
        self.circuit = circuit^
        self.code = code
        self.basis = basis
        self.distance = distance
        self.rounds = rounds
        self.noise = noise
        self.qubit_coords = List[Float64]()
        self.detector_coords = List[Float64]()
        self.data_qubits = List[Int]()
        self.x_checks = List[Int]()
        self.z_checks = List[Int]()

    def to_stim(self) raises -> String:
        return to_stim(self.circuit, self.qubit_coords, self.detector_coords)

    def write_to(self, mut w: Some[Writer]):
        w.write("CodeCircuit(code=", self.code, ", basis=", self.basis, ", d=", self.distance,
                ", rounds=", self.rounds, ", qubits=", self.circuit.num_qubits,
                ", detectors=", self.circuit.num_detectors, ", ops=", len(self.circuit.ops), ")")


def _check_code_args(distance: Int, rounds: Int, min_d: Int) raises:
    if distance < min_d:
        raise Error("QM-E706: distance " + String(distance) + " too small (minimum " + String(min_d) + "). Hint: use d = 3, 5, 7, ...")
    if rounds < 1:
        raise Error("QM-E706: rounds must be >= 1 (got " + String(rounds) + "). Hint: rounds = distance is the usual choice.")


def _x_error_all(mut c: Circuit, p: Float64, qs: List[Int]):
    if p > 0:
        for q in qs:
            c.x_error(p, q)


def _dep1_all(mut c: Circuit, p: Float64, qs: List[Int]):
    if p > 0:
        for q in qs:
            c.depolarize1(p, q)


# ---------------------------------------------------------------------------
# Repetition code
# ---------------------------------------------------------------------------
def repetition_code_memory(distance: Int, rounds: Int, noise: NoiseParams) raises -> CodeCircuit:
    """Bit-flip repetition code memory (Stim "repetition_code:memory" layout)."""
    _check_code_args(distance, rounds, 2)
    noise.check()
    var d = distance
    var n = 2 * d - 1
    var c = Circuit(n, 0, "repetition_code_memory_d" + String(d) + "_r" + String(rounds))
    var data = List[Int]()
    var meas = List[Int]()
    var allq = List[Int]()
    for i in range(d):
        data.append(2 * i)
    for i in range(d - 1):
        meas.append(2 * i + 1)
    for q in range(n):
        allq.append(q)
    var out = CodeCircuit(Circuit(1), "repetition", "Z", d, rounds, noise)
    for q in range(n):
        out.qubit_coords.append(Float64(q))
        out.qubit_coords.append(0.0)
    for q in allq:
        c.reset(q)
    _x_error_all(c, noise.after_reset_flip_probability, allq)
    c.tick()
    var p2 = noise.after_clifford_depolarization
    var prev = List[Int]()
    for r in range(rounds):
        _dep1_all(c, noise.before_round_data_depolarization, data)
        for i in range(d - 1):
            c.cx(2 * i, 2 * i + 1)
        if p2 > 0:
            for i in range(d - 1):
                c.depolarize2(p2, 2 * i, 2 * i + 1)
        c.tick()
        for i in range(d - 1):
            c.cx(2 * i + 2, 2 * i + 1)
        if p2 > 0:
            for i in range(d - 1):
                c.depolarize2(p2, 2 * i + 2, 2 * i + 1)
        c.tick()
        _x_error_all(c, noise.before_measure_flip_probability, meas)
        var cur = List[Int]()
        for q in meas:
            cur.append(c.num_measurements)
            c.measure(q, 0)
        for q in meas:
            c.reset(q)
        _x_error_all(c, noise.after_reset_flip_probability, meas)
        for i in range(d - 1):
            var recs = List[Int]()
            recs.append(cur[i])
            if r > 0:
                recs.append(prev[i])
            c.detector(recs)
            out.detector_coords.append(Float64(2 * i + 1))
            out.detector_coords.append(0.0)
            out.detector_coords.append(Float64(r))
        prev = cur^
        c.tick()
    _x_error_all(c, noise.before_measure_flip_probability, data)
    var drec = List[Int]()
    for q in data:
        drec.append(c.num_measurements)
        c.measure(q, 0)
    for i in range(d - 1):
        var recs: List[Int] = [drec[i], drec[i + 1], prev[i]]
        c.detector(recs)
        out.detector_coords.append(Float64(2 * i + 1))
        out.detector_coords.append(0.0)
        out.detector_coords.append(Float64(rounds))
    var lo: List[Int] = [drec[d - 1]]
    c.observable(0, lo)
    c.num_clbits = 1
    out.circuit = c^
    out.data_qubits = data^
    out.z_checks = meas^
    return out^


# ---------------------------------------------------------------------------
# Rotated surface code
# ---------------------------------------------------------------------------
@fieldwise_init
struct _Check(ImplicitlyCopyable):
    var a: Int  # plaquette centre is (2a, 2b)
    var b: Int
    var is_x: Bool
    var qubit: Int
    var n0: Int  # data qubit touched in CX layer k (-1 = none)
    var n1: Int
    var n2: Int
    var n3: Int

    def nb(self, k: Int) -> Int:
        if k == 0: return self.n0
        if k == 1: return self.n1
        if k == 2: return self.n2
        return self.n3


def surface_code_memory(distance: Int, rounds: Int, noise: NoiseParams, basis: String = "Z") raises -> CodeCircuit:
    """Rotated surface-code memory experiment in the Z or X basis.

    Layout: data (i, j), 0 <= i, j < d, at coordinates (2i+1, 2j+1); a check
    at plaquette centre (2a, 2b) is X-type when a+b is even. Weight-2 X checks
    live on the left/right boundaries, weight-2 Z checks on top/bottom, so
    Z_L = Z on column i = 0 and X_L = X on row j = 0.
    """
    _check_code_args(distance, rounds, 2)
    noise.check()
    var bu = basis.upper()
    if bu != "Z" and bu != "X":
        raise Error("QM-E706: basis must be 'Z' or 'X' (got '" + basis + "').")
    var is_xmem = bu == "X"
    var d = distance
    var nd = d * d
    var out = CodeCircuit(Circuit(1), "surface", bu, d, rounds, noise)
    # data qubits
    for q in range(nd):
        var i = q % d
        var j = q // d
        out.qubit_coords.append(Float64(2 * i + 1))
        out.qubit_coords.append(Float64(2 * j + 1))
        out.data_qubits.append(q)
    # checks
    var x_dirs: List[Int] = [1, 1, 1, -1, -1, 1, -1, -1]  # N order: (dx, dy) pairs
    var z_dirs: List[Int] = [1, 1, -1, 1, 1, -1, -1, -1]  # Z order
    var checks = List[_Check]()
    var nq = nd
    for b in range(d + 1):
        for a in range(d + 1):
            var is_x = ((a + b) % 2) == 0
            var ba = a == 0 or a == d
            var bb = b == 0 or b == d
            if ba and bb:
                continue
            if ba and not is_x:
                continue
            if bb and is_x:
                continue
            var nbs = List[Int]()
            for k in range(4):
                var dx = x_dirs[2 * k] if is_x else z_dirs[2 * k]
                var dy = x_dirs[2 * k + 1] if is_x else z_dirs[2 * k + 1]
                var cx_ = 2 * a + dx
                var cy_ = 2 * b + dy
                var i = (cx_ - 1) // 2
                var j = (cy_ - 1) // 2
                if cx_ > 0 and cy_ > 0 and i < d and j < d:
                    nbs.append(j * d + i)
                else:
                    nbs.append(-1)
            checks.append(_Check(a, b, is_x, nq, nbs[0], nbs[1], nbs[2], nbs[3]))
            out.qubit_coords.append(Float64(2 * a))
            out.qubit_coords.append(Float64(2 * b))
            if is_x:
                out.x_checks.append(nq)
            else:
                out.z_checks.append(nq)
            nq += 1
    var c = Circuit(nq, 0, "surface_code_memory_" + bu + "_d" + String(d) + "_r" + String(rounds))
    var data = out.data_qubits.copy()
    var mq = List[Int]()
    for ch in checks:
        mq.append(ch.qubit)
    var xq = out.x_checks.copy()
    var allq = List[Int]()
    for q in range(nq):
        allq.append(q)
    var p1 = noise.after_clifford_depolarization
    # --- initialisation
    for q in allq:
        c.reset(q)
    _x_error_all(c, noise.after_reset_flip_probability, allq)
    if is_xmem:
        for q in data:
            c.h(q)  # RX = R; H (noise-free basis change, Stim's RX)
    c.tick()
    var prev = List[Int](length=len(checks), fill=-1)
    for r in range(rounds):
        _dep1_all(c, noise.before_round_data_depolarization, data)
        for q in xq:
            c.h(q)
        _dep1_all(c, p1, xq)
        c.tick()
        for k in range(4):
            for ch in checks:
                var dq = ch.nb(k)
                if dq < 0:
                    continue
                if ch.is_x:
                    c.cx(ch.qubit, dq)
                else:
                    c.cx(dq, ch.qubit)
            if p1 > 0:
                for ch in checks:
                    var dq = ch.nb(k)
                    if dq < 0:
                        continue
                    if ch.is_x:
                        c.depolarize2(p1, ch.qubit, dq)
                    else:
                        c.depolarize2(p1, dq, ch.qubit)
            c.tick()
        for q in xq:
            c.h(q)
        _dep1_all(c, p1, xq)
        c.tick()
        _x_error_all(c, noise.before_measure_flip_probability, mq)
        var cur = List[Int]()
        for q in mq:
            cur.append(c.num_measurements)
            c.measure(q, 0)
        for q in mq:
            c.reset(q)
        _x_error_all(c, noise.after_reset_flip_probability, mq)
        for ci in range(len(checks)):
            var ch = checks[ci]
            if r == 0 and ch.is_x != is_xmem:
                continue  # random in the first round
            var recs = List[Int]()
            recs.append(cur[ci])
            if r > 0:
                recs.append(prev[ci])
            c.detector(recs)
            out.detector_coords.append(Float64(2 * ch.a))
            out.detector_coords.append(Float64(2 * ch.b))
            out.detector_coords.append(Float64(r))
        prev = cur^
        c.tick()
    # --- final data measurement in the memory basis
    if is_xmem:
        for q in data:
            c.h(q)
    _x_error_all(c, noise.before_measure_flip_probability, data)
    var drec = List[Int](length=nd, fill=-1)
    for q in data:
        drec[q] = c.num_measurements
        c.measure(q, 0)
    for ci in range(len(checks)):
        var ch = checks[ci]
        if ch.is_x != is_xmem:
            continue
        var recs = List[Int]()
        for k in range(4):
            var dq = ch.nb(k)
            if dq >= 0:
                recs.append(drec[dq])
        recs.append(prev[ci])
        c.detector(recs)
        out.detector_coords.append(Float64(2 * ch.a))
        out.detector_coords.append(Float64(2 * ch.b))
        out.detector_coords.append(Float64(rounds))
    var lrec = List[Int]()
    for t in range(d):
        if is_xmem:
            lrec.append(drec[t])  # row j = 0: X_L
        else:
            lrec.append(drec[t * d])  # column i = 0: Z_L
    c.observable(0, lrec)
    c.num_clbits = 1
    out.circuit = c^
    return out^


def generate_code(code: String, distance: Int, rounds: Int, noise: NoiseParams) raises -> CodeCircuit:
    """Factory by name: "repetition", "surface_z" (alias "surface"), "surface_x"."""
    var k = code.lower()
    if k == "repetition" or k == "repetition_code" or k == "rep":
        return repetition_code_memory(distance, rounds, noise)
    if k == "surface" or k == "surface_z" or k == "rotated_surface_z" or k == "surface_code":
        return surface_code_memory(distance, rounds, noise, "Z")
    if k == "surface_x" or k == "rotated_surface_x":
        return surface_code_memory(distance, rounds, noise, "X")
    raise Error("QM-E706: unknown code '" + code + "'. Hint: use 'repetition', 'surface_z' or 'surface_x'.")


# ---------------------------------------------------------------------------
# Stim export
# ---------------------------------------------------------------------------
def _stim_name(k: Int) -> String:
    if k == Op.I: return "I"
    if k == Op.X: return "X"
    if k == Op.Y: return "Y"
    if k == Op.Z: return "Z"
    if k == Op.H: return "H"
    if k == Op.S: return "S"
    if k == Op.SDG: return "S_DAG"
    if k == Op.SX: return "SQRT_X"
    if k == Op.SXDG: return "SQRT_X_DAG"
    if k == Op.CX: return "CX"
    if k == Op.CY: return "CY"
    if k == Op.CZ: return "CZ"
    if k == Op.SWAP: return "SWAP"
    if k == Op.ISWAP: return "ISWAP"
    if k == Op.ISWAPDG: return "ISWAP_DAG"
    if k == Op.MEASURE: return "M"
    if k == Op.RESET: return "R"
    if k == Op.DEPOLARIZE1: return "DEPOLARIZE1"
    if k == Op.DEPOLARIZE2: return "DEPOLARIZE2"
    if k == Op.X_ERROR: return "X_ERROR"
    if k == Op.Y_ERROR: return "Y_ERROR"
    if k == Op.Z_ERROR: return "Z_ERROR"
    if k == Op.PAULI_CHANNEL_1: return "PAULI_CHANNEL_1"
    return ""


def _fmt(x: Float64) -> String:
    return String(x)


struct _StimWriter:
    var out: String
    var cur: String  # pending merged line head, e.g. "CX" or "DEPOLARIZE2(0.001)"
    var targets: String

    def __init__(out self):
        self.out = String()
        self.cur = String()
        self.targets = String()

    def flush(mut self):
        if self.cur.byte_length() > 0:
            self.out += self.cur + self.targets + "\n"
            self.cur = String()
            self.targets = String()

    def emit(mut self, head: String, targets: String):
        """Append targets to the current line when the gate head matches."""
        if head != self.cur:
            self.flush()
            self.cur = head
        self.targets += targets

    def line(mut self, text: String):
        self.flush()
        self.out += text + "\n"


def _quarter_names(k: Int, t: Int) -> String:
    """Stim gate for RX/RY/RZ/RZZ/RXX/RYY by t quarter turns ("" = identity)."""
    if t == 0:
        return ""
    if k == Op.RX:
        return "SQRT_X" if t == 1 else ("X" if t == 2 else "SQRT_X_DAG")
    if k == Op.RY:
        return "SQRT_Y" if t == 1 else ("Y" if t == 2 else "SQRT_Y_DAG")
    if k == Op.RZ or k == Op.P:
        return "S" if t == 1 else ("Z" if t == 2 else "S_DAG")
    if k == Op.RZZ:
        return "SQRT_ZZ" if t == 1 else ("ZZ" if t == 2 else "SQRT_ZZ_DAG")
    if k == Op.RXX:
        return "SQRT_XX" if t == 1 else ("XX" if t == 2 else "SQRT_XX_DAG")
    return "SQRT_YY" if t == 1 else ("YY" if t == 2 else "SQRT_YY_DAG")


def to_stim(circuit: Circuit, qubit_coords: List[Float64] = List[Float64](),
            detector_coords: List[Float64] = List[Float64]()) raises -> String:
    """Export a Clifford+noise QUMO circuit as Stim circuit text.

    Consecutive instructions of the same gate are merged into one line
    (Stim semantics are sequential within a line, so this is exact).
    Measurement-record references are emitted Stim-style as rec[-k].
    """
    var w = _StimWriter()
    if len(qubit_coords) >= 2 * circuit.num_qubits:
        for q in range(circuit.num_qubits):
            w.line("QUBIT_COORDS(" + _fmt(qubit_coords[2 * q]) + ", " + _fmt(qubit_coords[2 * q + 1]) + ") " + String(q))
    var nmeas = 0
    var clbit_rec = List[Int](length=max(1, circuit.num_clbits), fill=-1)
    for g in circuit.ops:
        var k = g.kind
        if k == Op.TICK:
            w.line("TICK")
            continue
        if k == Op.BARRIER:
            continue
        if k == Op.DETECTOR or k == Op.OBSERVABLE:
            var s: String
            if k == Op.DETECTOR:
                s = "DETECTOR"
                var di = g.q2
                if len(detector_coords) >= 3 * (di + 1):
                    s += "(" + _fmt(detector_coords[3 * di]) + ", " + _fmt(detector_coords[3 * di + 1]) + ", " + _fmt(detector_coords[3 * di + 2]) + ")"
            else:
                s = "OBSERVABLE_INCLUDE(" + String(g.q2) + ")"
            for j in range(g.q1):
                s += " rec[-" + String(nmeas - circuit.records[g.q0 + j]) + "]"
            w.line(s)
            continue
        if g.cond >= 0:
            if g.cond_reg >= 0:
                raise Error("QM-E704: to_stim() cannot express register conditions (c_if_register). Hint: condition on single measured clbits.")
            var r = clbit_rec[g.cond]
            var ctl = ""
            if k == Op.X: ctl = "CX"
            elif k == Op.Y: ctl = "CY"
            elif k == Op.Z: ctl = "CZ"
            if ctl == "" or r < 0:
                raise Error("QM-E704: to_stim() supports only X/Y/Z feedback conditioned on a measured clbit (instruction '" + op_name(k) + "').")
            if g.cond_val == 0:
                w.emit(_stim_name(k), " " + String(g.q0))
            w.line(ctl + " rec[-" + String(nmeas - r) + "] " + String(g.q0))
            continue
        if k == Op.MEASURE:
            w.emit("M", " " + String(g.q0))
            if g.clbit >= 0:
                clbit_rec[g.clbit] = nmeas
            nmeas += 1
            continue
        if k == Op.ECR:
            w.emit("S", " " + String(g.q0))
            w.emit("CX", " " + String(g.q0) + " " + String(g.q1))
            w.emit("X", " " + String(g.q0))
            w.emit("SQRT_X", " " + String(g.q1))
            continue
        if k == Op.RX or k == Op.RY or k == Op.RZ or k == Op.P or k == Op.RZZ or k == Op.RXX or k == Op.RYY:
            if g.p0.is_symbolic():
                raise Error("QM-E702: to_stim() needs bound parameters. Hint: circuit.bind(values).")
            var nm = _quarter_names(k, quarter_turns(g.p0.value()))
            if nm == "":
                continue
            var tg = " " + String(g.q0)
            if op_arity(k) == 2:
                tg += " " + String(g.q1)
            if nm == "ZZ" or nm == "XX" or nm == "YY":
                var single = String(unsafe_from_utf8=nm.as_bytes()[0:1])
                w.emit(single, " " + String(g.q0) + " " + String(g.q1))
            else:
                w.emit(nm, tg)
            continue
        var name = _stim_name(k)
        if name == "":
            raise Error("QM-E701: '" + op_name(k) + "' has no Stim equivalent (non-Clifford). Hint: to_stim() exports Clifford+Pauli-noise circuits only.")
        var head = name
        if k == Op.PAULI_CHANNEL_1:
            head += "(" + _fmt(g.p0.value()) + ", " + _fmt(g.p1.value()) + ", " + _fmt(g.p2.value()) + ")"
        elif op_is_noise(k):
            head += "(" + _fmt(g.p0.value()) + ")"
        var tg = " " + String(g.q0)
        if op_arity(k) >= 2:
            tg += " " + String(g.q1)
        w.emit(head, tg)
    w.flush()
    return w.out
