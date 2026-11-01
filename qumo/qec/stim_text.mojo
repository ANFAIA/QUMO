"""Stim text import: circuits (`from_stim`) and detector error models (`parse_dem`).

`from_stim` makes QUMO's QEC stack usable on any Stim circuit, not only on its two
built-in code generators. `REPEAT` is unrolled and rec[-k] becomes an absolute
record. A Pauli-product measurement is lowered to a basis change, a CX ladder and
a single-qubit M, which is what `stim.Circuit.decomposed()` does, so MPP needs no
opcode and every simulator takes it. The same lowering covers MX, MY, MR, RX, RY,
MXX, MZZ. `M(p)` flips the record and not the qubit, which X_ERROR cannot express,
so it measures a copy on one spare qubit appended to the circuit.

Not imported (QM-E734): rec-controlled gates, HERALDED_*, sweep bits, MPAD, SPP.
`to_stim` exports the result; compare with `.flattened()` or by detector error
model, since MPP comes back lowered.
"""

from std.collections import Dict
from std.math import log
from qumo.ir.gates import Op
from qumo.ir.circuit import Circuit
from qumo.qec.codes import CodeCircuit, NoiseParams
from qumo.qec.dem import DetectorErrorModel, DemError, _xor_sorted, _split
from qumo.sim.frame import DetectorSamples, BATCH, FW
from qumo.sim.rng import Rng

comptime _MAX_INSTRUCTIONS = 50_000_000


def _stim_error(line: Int, msg: String) -> Error:
    return Error("QM-E734: Stim circuit line " + String(line) + ": " + msg + ".")


struct _Parser:
    var lines: List[String]
    var numbers: List[Int]  # source line of each entry of `lines`
    var q_max: Int
    var clifford_ancilla: Bool

    def __init__(out self, text: String):
        self.lines = List[String]()
        self.numbers = List[Int]()
        self.q_max = -1
        self.clifford_ancilla = False
        var n = 0
        for raw in text.split("\n"):
            n += 1
            var s = String(raw)
            var hash = s.find("#")
            if hash >= 0:
                var head = String(s[byte=0:hash])
                s = head^
            var line = String(s.strip())
            if line.byte_length() > 0:
                self.lines.append(line^)
                self.numbers.append(n)


def _split_instruction(line: String, mut name: String, mut args: List[Float64], mut targets: List[String], where_: Int) raises:
    """`NAME(a, b) t0 t1`: fields, stripped."""
    var bs = line.as_bytes()
    var i = 0
    while i < len(bs) and Int(bs[i]) != ord(" ") and Int(bs[i]) != ord("("):
        i += 1
    name = String(line[byte=0:i]).upper()
    var rest = String(line[byte=i:].strip())
    args.clear()
    if rest.startswith("("):
        var close = rest.find(")")
        if close < 0:
            raise _stim_error(where_, "missing ')' in '" + line + "'. Hint: arguments look like NAME(0.1, 0.2)")
        var inner = String(rest[byte=1:close])
        for a in inner.split(","):
            var t = String(a.strip())
            if t.byte_length() > 0:
                args.append(Float64(t))
        var tail = String(rest[byte=close + 1:].strip())
        rest = tail^
    targets.clear()
    for t in rest.split(" "):
        var x = String(t.strip())
        if x.byte_length() > 0:
            targets.append(x^)


def _qubit(t: String, where_: Int) raises -> Int:
    var s = t
    if s.startswith("!"):
        var tail = String(s[byte=1:])
        s = tail^
    for c in s.as_bytes():
        if Int(c) < ord("0") or Int(c) > ord("9"):
            raise _stim_error(where_, "'" + t + "' is not a qubit target. Hint: rec-controlled gates and sweep bits are not imported")
    return Int(s)


struct _Builder:
    var c: Circuit
    var coords: List[Float64]  # per detector
    var shift: List[Float64]
    var qubit_coords: Dict[Int, List[Float64]]
    var spare: Int
    var count: Int

    def __init__(out self, n: Int):
        self.c = Circuit(n, 0, "stim")
        self.coords = List[Float64]()
        self.shift = List[Float64]()
        self.qubit_coords = Dict[Int, List[Float64]]()
        self.spare = n - 1
        self.count = 0

    def measure(mut self, q: Int, invert: Bool, p: Float64):
        """Z measurement with a record-only flip probability and an inverted record."""
        if p > 0.0:
            # M(p) flips the record, not the qubit: copy to the spare qubit, measure there
            self.c.reset(self.spare)
            self.c.cx(q, self.spare)
            self.c.x_error(p, self.spare)
            self.c.measure(self.spare, self.c.num_measurements)
            self.c.cx(q, self.spare)  # uncompute: the spare stays free for the next copy
            self.c.reset(self.spare)
            return
        if invert:
            self.c.x(q)
        self.c.measure(q, self.c.num_measurements)
        if invert:
            self.c.x(q)

    def basis_in(mut self, q: Int, letter: Int):
        if letter == 1:
            self.c.h(q)
        elif letter == 2:
            self.c.rx(1.5707963267948966, q)

    def basis_out(mut self, q: Int, letter: Int):
        if letter == 1:
            self.c.h(q)
        elif letter == 2:
            self.c.rx(-1.5707963267948966, q)

    def product_measure(mut self, qs: List[Int], letters: List[Int], invert: Bool, p: Float64) raises:
        for i in range(len(qs)):
            self.basis_in(qs[i], letters[i])
        var last = qs[len(qs) - 1]
        for i in range(len(qs) - 1):
            self.c.cx(qs[i], last)
        self.measure(last, invert, p)
        var i = len(qs) - 2
        while i >= 0:
            self.c.cx(qs[i], last)
            i -= 1
        for j in range(len(qs)):
            self.basis_out(qs[j], letters[j])


def _letter(s: String, where_: Int) raises -> Int:
    if s == "X":
        return 1
    if s == "Y":
        return 2
    if s == "Z":
        return 3
    raise _stim_error(where_, "'" + s + "' is not a Pauli target")


def _pauli_target(t: String, mut q: Int, mut l: Int, where_: Int) raises:
    """`X5` -> qubit 5, letter 1."""
    if t.byte_length() < 2:
        raise _stim_error(where_, "bad Pauli target '" + t + "'")
    l = _letter(String(t[byte=0:1]).upper(), where_)
    q = _qubit(String(t[byte=1:]), where_)


def _gate_1q(mut c: Circuit, name: String, q: Int) -> Bool:
    if name == "I": c.id(q)
    elif name == "X": c.x(q)
    elif name == "Y": c.y(q)
    elif name == "Z": c.z(q)
    elif name == "H" or name == "H_XZ": c.h(q)
    elif name == "S" or name == "SQRT_Z": c.s(q)
    elif name == "S_DAG" or name == "SQRT_Z_DAG": c.sdg(q)
    elif name == "SQRT_X": c.sx(q)
    elif name == "SQRT_X_DAG": c.sxdg(q)
    elif name == "SQRT_Y":
        c.ry(1.5707963267948966, q)
    elif name == "SQRT_Y_DAG":
        c.ry(-1.5707963267948966, q)
    elif name == "H_YZ":
        c.h(q)
        c.s(q)
        c.h(q)
        c.z(q)
    elif name == "H_XY":
        c.x(q)
        c.s(q)
    elif name == "C_XYZ":
        c.sdg(q)
        c.h(q)
    elif name == "C_ZYX":
        c.h(q)
        c.s(q)
    else:
        return False
    return True


def _gate_2q(mut c: Circuit, name: String, a: Int, b: Int) -> Bool:
    if name == "CX" or name == "CNOT" or name == "ZCX": c.cx(a, b)
    elif name == "CY" or name == "ZCY": c.cy(a, b)
    elif name == "CZ" or name == "ZCZ": c.cz(a, b)
    elif name == "SWAP": c.swap(a, b)
    elif name == "ISWAP": c.iswap(a, b)
    elif name == "ISWAP_DAG": c.add(Op.ISWAPDG, a, b)
    elif name == "XCX":
        c.h(a)
        c.cx(a, b)
        c.h(a)
    elif name == "XCZ": c.cx(b, a)
    elif name == "XCY":
        c.h(a)
        c.cy(a, b)
        c.h(a)
    elif name == "YCX" or name == "YCY" or name == "YCZ":
        c.rx(1.5707963267948966, a)
        if name == "YCX": c.cx(a, b)
        elif name == "YCY": c.cy(a, b)
        else: c.cz(a, b)
        c.rx(-1.5707963267948966, a)
    elif name == "SQRT_ZZ": c.rzz(1.5707963267948966, a, b)
    elif name == "SQRT_ZZ_DAG": c.rzz(-1.5707963267948966, a, b)
    elif name == "SQRT_XX": c.rxx(1.5707963267948966, a, b)
    elif name == "SQRT_XX_DAG": c.rxx(-1.5707963267948966, a, b)
    elif name == "SQRT_YY": c.ryy(1.5707963267948966, a, b)
    elif name == "SQRT_YY_DAG": c.ryy(-1.5707963267948966, a, b)
    else:
        return False
    return True


def _records(c: Circuit, targets: List[String], where_: Int) raises -> List[Int]:
    var out = List[Int]()
    for t in targets:
        if not t.startswith("rec[-") or not t.endswith("]"):
            raise _stim_error(where_, "'" + t + "' is not a measurement record. Hint: write rec[-1]")
        var back = Int(String(t[byte=5:t.byte_length() - 1]))
        var r = c.num_measurements - back
        if r < 0:
            raise _stim_error(where_, t + " reaches before the first measurement")
        out.append(r)
    return out^


def _run_block(p: _Parser, start: Int, stop: Int, mut b: _Builder) raises:
    """Execute lines[start:stop] (REPEAT recursion, targets resolved as we go)."""
    var i = start
    while i < stop:
        ref line = p.lines[i]
        var where_ = p.numbers[i]
        i += 1
        b.count += 1
        if b.count > _MAX_INSTRUCTIONS:
            raise _stim_error(where_, "more than " + String(_MAX_INSTRUCTIONS) + " instructions after unrolling. Hint: lower the REPEAT counts")
        var name = String("")
        var args = List[Float64]()
        var targets = List[String]()
        _split_instruction(line, name, args, targets, where_)
        if name == "REPEAT":
            var count = Int(targets[0])
            var depth = 1
            var j = i
            while j < stop and depth > 0:
                if p.lines[j].endswith("{"):
                    depth += 1
                elif p.lines[j] == "}":
                    depth -= 1
                j += 1
            if depth != 0:
                raise _stim_error(where_, "REPEAT block is not closed")
            for _ in range(count):
                _run_block(p, i, j - 1, b)
            i = j
            continue
        if name == "}":
            continue
        if name == "TICK":
            b.c.tick()
            continue
        if name == "QUBIT_COORDS":
            for t in targets:
                b.qubit_coords[_qubit(t, where_)] = args.copy()
            continue
        if name == "SHIFT_COORDS":
            while len(b.shift) < len(args):
                b.shift.append(0.0)
            for k in range(len(args)):
                b.shift[k] += args[k]
            continue
        if name == "DETECTOR":
            for k in range(3):
                var v = args[k] if k < len(args) else 0.0
                b.coords.append(v + (b.shift[k] if k < len(b.shift) else 0.0))
            b.c.detector(_records(b.c, targets, where_))
            continue
        if name == "OBSERVABLE_INCLUDE":
            if len(args) != 1:
                raise _stim_error(where_, "OBSERVABLE_INCLUDE takes one index")
            b.c.observable(Int(args[0]), _records(b.c, targets, where_))
            continue
        # noise
        var p0 = args[0] if len(args) > 0 else 0.0
        if name == "X_ERROR" or name == "Y_ERROR" or name == "Z_ERROR" or name == "DEPOLARIZE1":
            for t in targets:
                var q = _qubit(t, where_)
                if name == "X_ERROR": b.c.x_error(p0, q)
                elif name == "Y_ERROR": b.c.y_error(p0, q)
                elif name == "Z_ERROR": b.c.z_error(p0, q)
                else: b.c.depolarize1(p0, q)
            continue
        if name == "DEPOLARIZE2":
            for k in range(0, len(targets), 2):
                b.c.depolarize2(p0, _qubit(targets[k], where_), _qubit(targets[k + 1], where_))
            continue
        if name == "PAULI_CHANNEL_1":
            if len(args) != 3:
                raise _stim_error(where_, "PAULI_CHANNEL_1 takes 3 probabilities")
            for t in targets:
                b.c.pauli_channel_1(args[0], args[1], args[2], _qubit(t, where_))
            continue
        if name == "PAULI_CHANNEL_2":
            for k in range(0, len(targets), 2):
                b.c.pauli_channel_2(args, _qubit(targets[k], where_), _qubit(targets[k + 1], where_))
            continue
        if name == "E" or name == "CORRELATED_ERROR" or name == "ELSE_CORRELATED_ERROR":
            var qs = List[Int]()
            var ls = List[Int]()
            for t in targets:
                var q = 0
                var l = 0
                _pauli_target(t, q, l, where_)
                qs.append(q)
                ls.append(l)
            b.c.correlated_error(p0, qs, ls, name == "ELSE_CORRELATED_ERROR")
            continue
        # measurements and resets
        if name == "M" or name == "MZ" or name == "MR" or name == "MRZ" or name == "MX" or name == "MY" or name == "MRX" or name == "MRY":
            for t in targets:
                var q = _qubit(t, where_)
                var inv = t.startswith("!")
                var letter = 1 if name == "MX" or name == "MRX" else (2 if name == "MY" or name == "MRY" else 3)
                b.basis_in(q, letter)
                b.measure(q, inv, p0)
                b.basis_out(q, letter)
                if name.startswith("MR"):
                    b.c.reset(q)
                    if letter == 1:
                        b.c.h(q)
                    elif letter == 2:
                        b.c.rx(-1.5707963267948966, q)
            continue
        if name == "R" or name == "RZ" or name == "RX" or name == "RY":
            for t in targets:
                var q = _qubit(t, where_)
                b.c.reset(q)
                if name == "RX":
                    b.c.h(q)
                elif name == "RY":
                    b.c.rx(-1.5707963267948966, q)
            continue
        if name == "MPP":
            for t in targets:
                var inv = t.startswith("!")
                var body = String(t[byte=1:]) if inv else t
                var qs = List[Int]()
                var ls = List[Int]()
                for part in body.split("*"):
                    var q = 0
                    var l = 0
                    _pauli_target(String(part), q, l, where_)
                    qs.append(q)
                    ls.append(l)
                b.product_measure(qs, ls, inv, p0)
            continue
        if name == "MXX" or name == "MYY" or name == "MZZ":
            var l = 1 if name == "MXX" else (2 if name == "MYY" else 3)
            for k in range(0, len(targets), 2):
                var qs: List[Int] = [_qubit(targets[k], where_), _qubit(targets[k + 1], where_)]
                var ls: List[Int] = [l, l]
                b.product_measure(qs, ls, targets[k].startswith("!") != targets[k + 1].startswith("!"), p0)
            continue
        # gates
        var ok = False
        if len(targets) > 0 and _gate_1q(b.c, name, _qubit(targets[0], where_)):
            ok = True
            for k in range(1, len(targets)):
                _ = _gate_1q(b.c, name, _qubit(targets[k], where_))
        elif len(targets) >= 2 and _gate_2q(b.c, name, _qubit(targets[0], where_), _qubit(targets[1], where_)):
            ok = True
            if len(targets) % 2 != 0:
                raise _stim_error(where_, name + " takes pairs of targets")
            for k in range(2, len(targets), 2):
                _ = _gate_2q(b.c, name, _qubit(targets[k], where_), _qubit(targets[k + 1], where_))
        if not ok:
            raise _stim_error(where_, "'" + name + "' is not imported. Hint: docs/QEC.md lists the supported instructions")


def _max_qubit(p: _Parser, mut needs_spare: Bool) raises -> Int:
    var q_max = -1
    var name = String("")
    var args = List[Float64]()
    var targets = List[String]()
    for i in range(len(p.lines)):
        if p.lines[i] == "}":
            continue
        _split_instruction(p.lines[i], name, args, targets, p.numbers[i])
        if name.startswith("REPEAT") or name == "DETECTOR" or name == "OBSERVABLE_INCLUDE" or name == "TICK" or name == "SHIFT_COORDS":
            continue
        if name.startswith("M") and len(args) > 0 and args[0] > 0.0:
            needs_spare = True
        for t in targets:
            # digits of each operand: X1*Y2, !3, 7
            var v = 0
            var has = False
            for c in t.as_bytes():
                var d = Int(c)
                if d >= ord("0") and d <= ord("9"):
                    v = 10 * v + (d - ord("0"))
                    has = True
                else:
                    if has:
                        q_max = max(q_max, v)
                    v = 0
                    has = False
            if has:
                q_max = max(q_max, v)
    return q_max


def from_stim(text: String) raises -> CodeCircuit:
    """Parse Stim circuit text. QM-E734 on anything it cannot represent exactly."""
    var p = _Parser(text)
    var spare = False
    var n = _max_qubit(p, spare) + 1
    if spare:
        n += 1
    var b = _Builder(max(n, 1))
    _run_block(p, 0, len(p.lines), b)
    b.c.num_clbits = max(b.c.num_clbits, 0)
    var cc = CodeCircuit(b.c.copy(), "stim", "", 0, 0, NoiseParams.noiseless())
    cc.detector_coords = b.coords.copy()
    var flat = List[Float64](length=2 * b.c.num_qubits, fill=0.0)
    for kv in b.qubit_coords.items():
        for k in range(min(2, len(kv.value))):
            flat[2 * kv.key + k] = kv.value[k]
    cc.qubit_coords = flat^
    return cc^


# ---------------------------------------------------------------------------
# Detector error models
# ---------------------------------------------------------------------------
def _dem_error(line: Int, msg: String) -> Error:
    return Error("QM-E735: detector error model line " + String(line) + ": " + msg + ".")


struct _DemBuilder:
    var dem: DetectorErrorModel
    var offset: Int
    var max_det: Int
    var max_obs: Int

    def __init__(out self):
        self.dem = DetectorErrorModel(0, 0)
        self.offset = 0
        self.max_det = -1
        self.max_obs = -1


def _run_dem(p: _Parser, start: Int, stop: Int, mut b: _DemBuilder) raises:
    var i = start
    while i < stop:
        ref line = p.lines[i]
        var where_ = p.numbers[i]
        i += 1
        var name = String("")
        var args = List[Float64]()
        var targets = List[String]()
        _split_instruction(line, name, args, targets, where_)
        name = name.lower()
        if name == "repeat":
            var count = Int(targets[0])
            var depth = 1
            var j = i
            while j < stop and depth > 0:
                if p.lines[j].endswith("{"):
                    depth += 1
                elif p.lines[j] == "}":
                    depth -= 1
                j += 1
            if depth != 0:
                raise _dem_error(where_, "repeat block is not closed")
            for _ in range(count):
                _run_dem(p, i, j - 1, b)
            i = j
        elif name == "}":
            continue
        elif name == "shift_detectors":
            if len(targets) != 1:
                raise _dem_error(where_, "shift_detectors takes one count")
            b.offset += Int(targets[0])
        elif name == "detector":
            for t in targets:
                b.max_det = max(b.max_det, b.offset + Int(String(t[byte=1:])))
        elif name == "logical_observable":
            for t in targets:
                b.max_obs = max(b.max_obs, Int(String(t[byte=1:])))
        elif name == "error":
            if len(args) != 1:
                raise _dem_error(where_, "error takes one probability")
            var comps = List[List[Int]]()
            var comp_obs = List[UInt64]()
            comps.append(List[Int]())
            comp_obs.append(0)
            for t in targets:
                if t == "^":
                    comps.append(List[Int]())
                    comp_obs.append(0)
                elif t.startswith("D"):
                    var d = b.offset + Int(String(t[byte=1:]))
                    b.max_det = max(b.max_det, d)
                    comps[len(comps) - 1].append(d)
                elif t.startswith("L"):
                    var k = Int(String(t[byte=1:]))
                    if k >= 64:
                        raise _dem_error(where_, "at most 64 logical observables")
                    b.max_obs = max(b.max_obs, k)
                    comp_obs[len(comp_obs) - 1] ^= UInt64(1) << UInt64(k)
                else:
                    raise _dem_error(where_, "bad target '" + t + "'. Hint: targets look like D3, L0 or ^")
            var full = List[Int]()
            var obs = UInt64(0)
            for c in range(len(comps)):
                sort(comps[c])
                full = _xor_sorted(full, comps[c])
                obs ^= comp_obs[c]
            var e = DemError(args[0], full^, obs)
            if len(comps) > 1:
                e.comp_dets = comps.copy()
                e.comp_obs = comp_obs.copy()
            b.dem.errors.append(e^)
        else:
            raise _dem_error(where_, "unknown instruction '" + name + "'")


def parse_dem(text: String) raises -> DetectorErrorModel:
    """Stim detector error model text; `repeat` blocks are expanded."""
    var p = _Parser(text)
    var b = _DemBuilder()
    _run_dem(p, 0, len(p.lines), b)
    b.dem.num_detectors = b.max_det + 1
    b.dem.num_observables = b.max_obs + 1
    return b.dem.copy()


def sample_dem(dem: DetectorErrorModel, shots: Int, seed: UInt64 = 1234) raises -> DetectorSamples:
    """Detection events drawn from the model directly, no circuit: each error fires
    in each shot with its probability, geometric skips over the shots."""
    var out = DetectorSamples(shots, dem.num_detectors, dem.num_observables)
    var rng = Rng(seed)
    var D = dem.num_detectors
    var K = dem.num_observables
    for e in dem.errors:
        if e.p <= 0.0:
            continue
        if e.p > 1.0:
            raise Error("QM-E735: error probability " + String(e.p) + " is above 1. Hint: probabilities are in [0, 1].")
        var lp = log(1.0 - min(e.p, 1.0 - 1e-16))
        var shot = Int(log(1.0 - rng.uniform()) / lp)
        while shot < shots:
            var b = shot // BATCH
            var s = shot % BATCH
            var word = s >> 6
            var bit = UInt64(1) << UInt64(s & 63)
            for d in e.dets:
                var at = (b * D + d) * FW + word
                out.dets[at] = out.dets[at] ^ bit
            if e.obs != 0:
                for k in range(K):
                    if (e.obs >> UInt64(k)) & 1 != 0:
                        var at = (b * K + k) * FW + word
                        out.obs[at] = out.obs[at] ^ bit
            shot += 1 + Int(log(1.0 - rng.uniform()) / lp)
    return out^
