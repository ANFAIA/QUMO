"""Bit-packed Pauli-frame sampler (Stim's key idea, in Mojo).

How it works
  1. One noiseless *reference sample* of the circuit is taken with the
     tableau simulator (`qumo.sim.stabilizer.reference_sample`), with every
     random measurement forced to 0.
  2. Each shot is then described only by its *Pauli frame*: the Pauli error
     by which it differs from the reference. Frames are propagated through the
     Clifford circuit with pure bit operations (H swaps x/z, CX xors columns,
     ...). We store one bit per shot: a batch is `FW` UInt64 words = 512 shots,
     and every gate is one SIMD[uint64, 8] load/xor/store per operand.
  3. Noise channels flip frame bits. Bernoulli lanes are drawn by geometric
     skipping over the whole 512-shot batch, so the cost is O(512 p) per
     channel instead of O(512).
  4. A measurement's flip relative to the reference is the frame's X bit.
     Like Stim, the Z part of the frame is re-randomised after every
     measurement/reset and at initialisation (a Z on a Z-eigenstate is a
     gauge), which makes non-deterministic measurements correctly random.
  5. Detectors / observables are XORs of measurement flips: they are the
     detection events Stim reports (parity relative to the noiseless value).
  6. Batches are independent (seed split per batch index), so sampling is
     parallel *and* bit-reproducible regardless of thread count.

Classical feed-forward of Pauli gates (`x(q); c_if(clbit, v)`) is supported
(the frame picks up the Pauli when the controlling record flips); any other
conditioned gate raises QM-E704.

Error codes: QM-E701 (non-Clifford), QM-E704 (unsupported feed-forward).
"""

from std.math import log, log1p
from std.time import perf_counter_ns
from std.bit import pop_count, count_trailing_zeros
from std.runtime import parallelism_level
from max.algorithm import parallelize

from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary, op_is_noise
from qumo.ir.circuit import Circuit, Instruction
from qumo.sim.rng import Rng
from qumo.sim.stabilizer import reference_sample, quarter_turns

comptime FW = 8  # UInt64 words per batch
comptime BATCH = FW * 64  # shots per batch
comptime WPtr = MutPointer[UInt64, MutAnyOrigin]
comptime Words = SIMD[DType.uint64, FW]


struct FK:
    """Frame-op kinds (signs are irrelevant for frames, so S == SDG etc.)."""

    comptime H = 0
    comptime S = 1  # z ^= x
    comptime SX = 2  # x ^= z
    comptime CX = 3
    comptime CZ = 4
    comptime SWAP = 5
    comptime M = 6  # a = qubit, b = record index
    comptime R = 7
    comptime XE = 8
    comptime YE = 9
    comptime ZE = 10
    comptime DEP1 = 11
    comptime DEP2 = 12
    comptime PC1 = 13
    comptime DET = 14  # a = record offset, b = count, c = detector id
    comptime OBS = 15  # a = record offset, b = count, c = observable index
    comptime FEED = 16  # a = record, b = qubit, c = pauli (1=X, 2=Y, 3=Z)


@fieldwise_init
struct FrameOp(ImplicitlyCopyable):
    var kind: Int
    var a: Int
    var b: Int
    var c: Int
    var p: Float64  # total error probability
    var px: Float64
    var py: Float64
    var lp: Float64  # log(1 - p) for geometric skipping

    @staticmethod
    def gate(kind: Int, a: Int, b: Int = -1) -> FrameOp:
        return FrameOp(kind, a, b, -1, 0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def noise(kind: Int, a: Int, b: Int, p: Float64, px: Float64 = 0.0, py: Float64 = 0.0) -> FrameOp:
        var lp = log1p(-p) if p < 1.0 else -1e300
        return FrameOp(kind, a, b, -1, p, px, py, lp)


def _lower_gate(mut ops: List[FrameOp], g: Instruction) raises:
    """Append the frame primitives of one Clifford instruction."""
    var k = g.kind
    var a = g.q0
    var b = g.q1
    if k == Op.I or k == Op.X or k == Op.Y or k == Op.Z:
        return
    if k == Op.H:
        ops.append(FrameOp.gate(FK.H, a))
    elif k == Op.S or k == Op.SDG:
        ops.append(FrameOp.gate(FK.S, a))
    elif k == Op.SX or k == Op.SXDG:
        ops.append(FrameOp.gate(FK.SX, a))
    elif k == Op.CX:
        ops.append(FrameOp.gate(FK.CX, a, b))
    elif k == Op.CZ:
        ops.append(FrameOp.gate(FK.CZ, a, b))
    elif k == Op.CY:
        ops.append(FrameOp.gate(FK.S, b))
        ops.append(FrameOp.gate(FK.CX, a, b))
        ops.append(FrameOp.gate(FK.S, b))
    elif k == Op.SWAP:
        ops.append(FrameOp.gate(FK.SWAP, a, b))
    elif k == Op.ISWAP:
        ops.append(FrameOp.gate(FK.CZ, a, b))
        ops.append(FrameOp.gate(FK.SWAP, a, b))
        ops.append(FrameOp.gate(FK.S, a))
        ops.append(FrameOp.gate(FK.S, b))
    elif k == Op.ISWAPDG:
        ops.append(FrameOp.gate(FK.S, a))
        ops.append(FrameOp.gate(FK.S, b))
        ops.append(FrameOp.gate(FK.SWAP, a, b))
        ops.append(FrameOp.gate(FK.CZ, a, b))
    elif k == Op.ECR:
        ops.append(FrameOp.gate(FK.S, a))
        ops.append(FrameOp.gate(FK.CX, a, b))
        ops.append(FrameOp.gate(FK.SX, b))
    elif k == Op.RX or k == Op.RY or k == Op.RZ or k == Op.P or k == Op.RZZ or k == Op.RXX or k == Op.RYY:
        if g.p0.is_symbolic():
            raise Error("QM-E702: '" + op_name(k) + "' has an unbound parameter. Hint: call circuit.bind(values) first.")
        var t = quarter_turns(g.p0.value())
        if (t & 1) == 0:
            return  # identity or a Pauli: no frame action
        if k == Op.RX:
            ops.append(FrameOp.gate(FK.SX, a))
        elif k == Op.RY:
            ops.append(FrameOp.gate(FK.H, a))
        elif k == Op.RZ or k == Op.P:
            ops.append(FrameOp.gate(FK.S, a))
        else:
            var pre = -1
            if k == Op.RXX:
                pre = FK.H
            elif k == Op.RYY:
                pre = FK.SX
            if pre >= 0:
                ops.append(FrameOp.gate(pre, a))
                ops.append(FrameOp.gate(pre, b))
            ops.append(FrameOp.gate(FK.CZ, a, b))
            ops.append(FrameOp.gate(FK.S, a))
            ops.append(FrameOp.gate(FK.S, b))
            if pre >= 0:
                ops.append(FrameOp.gate(pre, a))
                ops.append(FrameOp.gate(pre, b))
    else:
        raise Error(
            "QM-E701: gate '" + op_name(k) + "' is not Clifford; Pauli-frame sampling needs a"
            " Clifford circuit. Hint: use qumo.sim.runner.sample() (state vector) instead."
        )


struct FrameProgram(Copyable):
    """A circuit lowered to frame primitives + its noiseless reference sample."""

    var ops: List[FrameOp]
    var records: List[Int]  # detector / observable record lists
    var num_qubits: Int
    var num_measurements: Int
    var num_detectors: Int
    var num_observables: Int
    var num_circuit_ops: Int  # instructions in the source circuit (throughput unit)
    var reference: List[UInt8]

    def __init__(out self, circuit: Circuit, with_reference: Bool = True) raises:
        circuit.validate()
        self.ops = List[FrameOp]()
        self.records = circuit.records.copy()
        self.num_qubits = circuit.num_qubits
        self.num_measurements = circuit.num_measurements
        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables
        self.num_circuit_ops = len(circuit.ops)
        var clbit_rec = List[Int](length=max(circuit.num_clbits, 1), fill=-1)
        for g in circuit.ops:
            var k = g.kind
            if g.cond >= 0:
                if (k == Op.X or k == Op.Y or k == Op.Z) and g.cond_reg < 0:
                    var r = clbit_rec[g.cond]
                    if r >= 0:
                        var pc = 1 if k == Op.X else (2 if k == Op.Y else 3)
                        self.ops.append(FrameOp(FK.FEED, r, g.q0, pc, 0.0, 0.0, 0.0, 0.0))
                    continue
                if k == Op.I or k == Op.BARRIER or k == Op.TICK:
                    continue
                raise Error(
                    "QM-E704: classically-conditioned '" + op_name(k) + "' cannot be Pauli-frame"
                    " simulated (only X/Y/Z feedback conditioned on a single clbit is supported). Hint: use"
                    " qumo.sim.stabilizer.sample_stabilizer() for general feed-forward."
                )
            if op_is_unitary(k):
                _lower_gate(self.ops, g)
            elif k == Op.MEASURE:
                self.ops.append(FrameOp.gate(FK.M, g.q0, g.q1))
                if g.clbit >= 0:
                    clbit_rec[g.clbit] = g.q1
            elif k == Op.RESET:
                self.ops.append(FrameOp.gate(FK.R, g.q0))
            elif op_is_noise(k):
                var p = g.p0.value()
                if k == Op.X_ERROR:
                    if p > 0: self.ops.append(FrameOp.noise(FK.XE, g.q0, -1, p))
                elif k == Op.Y_ERROR:
                    if p > 0: self.ops.append(FrameOp.noise(FK.YE, g.q0, -1, p))
                elif k == Op.Z_ERROR:
                    if p > 0: self.ops.append(FrameOp.noise(FK.ZE, g.q0, -1, p))
                elif k == Op.DEPOLARIZE1:
                    if p > 0: self.ops.append(FrameOp.noise(FK.DEP1, g.q0, -1, p))
                elif k == Op.DEPOLARIZE2:
                    if p > 0: self.ops.append(FrameOp.noise(FK.DEP2, g.q0, g.q1, p))
                elif k == Op.PAULI_CHANNEL_1:
                    var px = g.p0.value()
                    var py = g.p1.value()
                    var pz = g.p2.value()
                    var tot = px + py + pz
                    if tot > 0: self.ops.append(FrameOp.noise(FK.PC1, g.q0, -1, tot, px, py))
            elif k == Op.DETECTOR:
                self.ops.append(FrameOp(FK.DET, g.q0, g.q1, g.q2, 0.0, 0.0, 0.0, 0.0))
            elif k == Op.OBSERVABLE:
                self.ops.append(FrameOp(FK.OBS, g.q0, g.q1, g.q2, 0.0, 0.0, 0.0, 0.0))
        if with_reference:
            self.reference = reference_sample(circuit)
        else:
            self.reference = List[UInt8](length=circuit.num_measurements, fill=0)


@always_inline
def _rand_words(mut rng: Rng) -> Words:
    var v = Words(0)
    comptime for i in range(FW):
        v[i] = rng.next_u64()
    return v


@always_inline
def _skip(mut rng: Rng, lp: Float64) -> Int:
    """Number of Bernoulli failures before the next success (geometric)."""
    return Int(log(1.0 - rng.uniform()) / lp)


@always_inline
def _flip(p: WPtr, base: Int, pos: Int):
    var w = base + (pos >> 6)
    p[unsafe_offset=w] = p[unsafe_offset=w] ^ (UInt64(1) << UInt64(pos & 63))


struct FrameBuffers(Movable):
    """Per-worker scratch for one 512-shot batch (reused across batches).

    Layouts (word index): xs/zs [qubit*FW + w], recs [record*FW + w],
    dets [detector*FW + w], obs [observable*FW + w]. Bit b of word w is shot
    64*w + b of the batch.
    """

    var xs: List[UInt64]
    var zs: List[UInt64]
    var recs: List[UInt64]
    var dets: List[UInt64]
    var obs: List[UInt64]

    def __init__(out self, prog: FrameProgram):
        self.xs = List[UInt64](length=max(1, prog.num_qubits) * FW, fill=0)
        self.zs = List[UInt64](length=max(1, prog.num_qubits) * FW, fill=0)
        self.recs = List[UInt64](length=max(1, prog.num_measurements) * FW, fill=0)
        self.dets = List[UInt64](length=max(1, prog.num_detectors) * FW, fill=0)
        self.obs = List[UInt64](length=max(1, prog.num_observables) * FW, fill=0)

    def run(mut self, prog: FrameProgram, mut rng: Rng):
        """Propagate one batch of 512 Pauli frames through the program."""
        var xp = self.xs.unsafe_ptr().as_unsafe_any_origin()
        var zp = self.zs.unsafe_ptr().as_unsafe_any_origin()
        var rp = self.recs.unsafe_ptr().as_unsafe_any_origin()
        var dp = self.dets.unsafe_ptr().as_unsafe_any_origin()
        var op_ = self.obs.unsafe_ptr().as_unsafe_any_origin()
        var rec_list = prog.records.unsafe_ptr()
        for q in range(prog.num_qubits):
            xp.unsafe_store(q * FW, Words(0))
            zp.unsafe_store(q * FW, _rand_words(rng))
        for i in range(len(self.obs)):
            op_[unsafe_offset=i] = 0
        for ref fo in prog.ops:
            var k = fo.kind
            if k == FK.CX:
                var c = fo.a * FW
                var t = fo.b * FW
                xp.unsafe_store(t, xp.unsafe_load[width=FW](t) ^ xp.unsafe_load[width=FW](c))
                zp.unsafe_store(c, zp.unsafe_load[width=FW](c) ^ zp.unsafe_load[width=FW](t))
            elif k == FK.H:
                var o = fo.a * FW
                var vx = xp.unsafe_load[width=FW](o)
                xp.unsafe_store(o, zp.unsafe_load[width=FW](o))
                zp.unsafe_store(o, vx)
            elif k == FK.DEP2:
                var a = fo.a * FW
                var b = fo.b * FW
                var pos = _skip(rng, fo.lp)
                while pos < BATCH:
                    var r = 1 + rng.below(15)
                    var pa = r & 3
                    var pb = r >> 2
                    if pa == 1 or pa == 2: _flip(xp, a, pos)
                    if pa >= 2: _flip(zp, a, pos)
                    if pb == 1 or pb == 2: _flip(xp, b, pos)
                    if pb >= 2: _flip(zp, b, pos)
                    pos += 1 + _skip(rng, fo.lp)
            elif k == FK.DEP1:
                var a = fo.a * FW
                var pos = _skip(rng, fo.lp)
                while pos < BATCH:
                    var r = rng.below(3)  # 0=X 1=Y 2=Z
                    if r <= 1: _flip(xp, a, pos)
                    if r >= 1: _flip(zp, a, pos)
                    pos += 1 + _skip(rng, fo.lp)
            elif k == FK.M:
                var o = fo.a * FW
                rp.unsafe_store(fo.b * FW, xp.unsafe_load[width=FW](o))
                zp.unsafe_store(o, zp.unsafe_load[width=FW](o) ^ _rand_words(rng))
            elif k == FK.R:
                var o = fo.a * FW
                xp.unsafe_store(o, Words(0))
                zp.unsafe_store(o, _rand_words(rng))
            elif k == FK.XE or k == FK.YE or k == FK.ZE:
                var a = fo.a * FW
                var pos = _skip(rng, fo.lp)
                while pos < BATCH:
                    if k != FK.ZE: _flip(xp, a, pos)
                    if k != FK.XE: _flip(zp, a, pos)
                    pos += 1 + _skip(rng, fo.lp)
            elif k == FK.PC1:
                var a = fo.a * FW
                var pos = _skip(rng, fo.lp)
                while pos < BATCH:
                    var u = rng.uniform() * fo.p
                    if u < fo.px:
                        _flip(xp, a, pos)
                    elif u < fo.px + fo.py:
                        _flip(xp, a, pos)
                        _flip(zp, a, pos)
                    else:
                        _flip(zp, a, pos)
                    pos += 1 + _skip(rng, fo.lp)
            elif k == FK.DET:
                var acc = Words(0)
                for j in range(fo.b):
                    acc ^= rp.unsafe_load[width=FW](rec_list[unsafe_offset=fo.a + j] * FW)
                dp.unsafe_store(fo.c * FW, acc)
            elif k == FK.OBS:
                var acc = op_.unsafe_load[width=FW](fo.c * FW)
                for j in range(fo.b):
                    acc ^= rp.unsafe_load[width=FW](rec_list[unsafe_offset=fo.a + j] * FW)
                op_.unsafe_store(fo.c * FW, acc)
            elif k == FK.S:
                var o = fo.a * FW
                zp.unsafe_store(o, zp.unsafe_load[width=FW](o) ^ xp.unsafe_load[width=FW](o))
            elif k == FK.SX:
                var o = fo.a * FW
                xp.unsafe_store(o, xp.unsafe_load[width=FW](o) ^ zp.unsafe_load[width=FW](o))
            elif k == FK.CZ:
                var a = fo.a * FW
                var b = fo.b * FW
                var xa = xp.unsafe_load[width=FW](a)
                var xb = xp.unsafe_load[width=FW](b)
                zp.unsafe_store(a, zp.unsafe_load[width=FW](a) ^ xb)
                zp.unsafe_store(b, zp.unsafe_load[width=FW](b) ^ xa)
            elif k == FK.SWAP:
                var a = fo.a * FW
                var b = fo.b * FW
                var xa = xp.unsafe_load[width=FW](a)
                var za = zp.unsafe_load[width=FW](a)
                xp.unsafe_store(a, xp.unsafe_load[width=FW](b))
                zp.unsafe_store(a, zp.unsafe_load[width=FW](b))
                xp.unsafe_store(b, xa)
                zp.unsafe_store(b, za)
            elif k == FK.FEED:
                var flips = rp.unsafe_load[width=FW](fo.a * FW)
                var o = fo.b * FW
                if fo.c == 1 or fo.c == 2:
                    xp.unsafe_store(o, xp.unsafe_load[width=FW](o) ^ flips)
                if fo.c >= 2:
                    zp.unsafe_store(o, zp.unsafe_load[width=FW](o) ^ flips)


@always_inline
def batch_rng(seed: UInt64, batch: Int) -> Rng:
    """Deterministic per-batch stream (independent of thread scheduling)."""
    return Rng(seed).split(batch)


def num_batches_for(shots: Int) -> Int:
    return (shots + BATCH - 1) // BATCH


struct DetectorSamples(Copyable, Writable):
    """Bit-packed detection events + observable flips.

    Layout: dets[(batch * D + d) * FW + w], obs[(batch * K + k) * FW + w];
    shot s lives in batch s // 512, word (s % 512) // 64, bit s % 64.
    """

    var shots: Int
    var num_detectors: Int
    var num_observables: Int
    var num_batches: Int
    var dets: List[UInt64]
    var obs: List[UInt64]
    var seconds: Float64

    def __init__(out self, shots: Int, num_detectors: Int, num_observables: Int):
        self.shots = shots
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.num_batches = num_batches_for(shots)
        self.dets = List[UInt64](length=max(1, self.num_batches * num_detectors * FW), fill=0)
        self.obs = List[UInt64](length=max(1, self.num_batches * num_observables * FW), fill=0)
        self.seconds = 0.0

    @always_inline
    def _bit(self, buf: List[UInt64], row: Int, rows: Int, shot: Int) -> Bool:
        var b = shot // BATCH
        var s = shot % BATCH
        var w = buf[(b * rows + row) * FW + (s >> 6)]
        return ((w >> UInt64(s & 63)) & 1) != 0

    def get(self, shot: Int, detector: Int) -> Bool:
        return self._bit(self.dets, detector, self.num_detectors, shot)

    def observable(self, shot: Int, k: Int) -> Bool:
        return self._bit(self.obs, k, self.num_observables, shot)

    def observables_mask(self, shot: Int) -> UInt64:
        var m: UInt64 = 0
        for k in range(min(64, self.num_observables)):
            if self.observable(shot, k):
                m |= UInt64(1) << UInt64(k)
        return m

    def fired(self, shot: Int) -> List[Int]:
        var out = List[Int]()
        for d in range(self.num_detectors):
            if self.get(shot, d):
                out.append(d)
        return out^

    def _valid_mask(self, b: Int, w: Int) -> UInt64:
        var base = b * BATCH + w * 64
        var left = self.shots - base
        if left >= 64:
            return ~UInt64(0)
        if left <= 0:
            return 0
        return (UInt64(1) << UInt64(left)) - 1

    def num_detection_events(self) -> Int:
        var total = 0
        for b in range(self.num_batches):
            for d in range(self.num_detectors):
                for w in range(FW):
                    total += Int(pop_count(self.dets[(b * self.num_detectors + d) * FW + w] & self._valid_mask(b, w)))
        return total

    def detector_fraction(self, d: Int) -> Float64:
        var total = 0
        for b in range(self.num_batches):
            for w in range(FW):
                total += Int(pop_count(self.dets[(b * self.num_detectors + d) * FW + w] & self._valid_mask(b, w)))
        return Float64(total) / Float64(max(1, self.shots))

    def observable_flip_count(self, k: Int = 0) -> Int:
        var total = 0
        for b in range(self.num_batches):
            for w in range(FW):
                total += Int(pop_count(self.obs[(b * self.num_observables + k) * FW + w] & self._valid_mask(b, w)))
        return total

    def write_to(self, mut w: Some[Writer]):
        w.write("DetectorSamples(shots=", self.shots, ", detectors=", self.num_detectors,
                ", observables=", self.num_observables, ", events=", self.num_detection_events(), ")")


struct MeasurementSamples(Copyable, Writable):
    """Bit-packed raw measurement records: bits[(batch * M + m) * FW + w]."""

    var shots: Int
    var num_measurements: Int
    var num_batches: Int
    var bits: List[UInt64]
    var seconds: Float64

    def __init__(out self, shots: Int, num_measurements: Int):
        self.shots = shots
        self.num_measurements = num_measurements
        self.num_batches = num_batches_for(shots)
        self.bits = List[UInt64](length=max(1, self.num_batches * num_measurements * FW), fill=0)
        self.seconds = 0.0

    def get(self, shot: Int, m: Int) -> Bool:
        var b = shot // BATCH
        var s = shot % BATCH
        var w = self.bits[(b * self.num_measurements + m) * FW + (s >> 6)]
        return ((w >> UInt64(s & 63)) & 1) != 0

    def shot_bits(self, shot: Int) -> List[UInt8]:
        var out = List[UInt8](capacity=self.num_measurements)
        for m in range(self.num_measurements):
            out.append(1 if self.get(shot, m) else 0)
        return out^

    def fraction_one(self, m: Int) -> Float64:
        var c = 0
        for s in range(self.shots):
            if self.get(s, m):
                c += 1
        return Float64(c) / Float64(max(1, self.shots))

    def write_to(self, mut w: Some[Writer]):
        w.write("MeasurementSamples(shots=", self.shots, ", measurements=", self.num_measurements, ")")


def _num_chunks(nb: Int) -> Int:
    return max(1, min(nb, 4 * max(1, parallelism_level())))


struct FrameSimulator(Copyable):
    """High-throughput sampler for noisy Clifford circuits (QEC workhorse).

        var fs = FrameSimulator(circuit)
        var ds = fs.sample_detectors(1_000_000, seed=7)
        print(ds.num_detection_events(), fs.throughput(ds))
    """

    var prog: FrameProgram

    def __init__(out self, circuit: Circuit) raises:
        self.prog = FrameProgram(circuit)

    def sample_detectors(self, shots: Int, seed: UInt64 = 1234, threads: Int = 0) -> DetectorSamples:
        """Detection events + observable flips; `threads=1` forces a single worker."""
        var out = DetectorSamples(shots, self.prog.num_detectors, self.prog.num_observables)
        var nb = out.num_batches
        if nb == 0:
            return out^
        var D = self.prog.num_detectors
        var K = self.prog.num_observables
        var dptr = out.dets.unsafe_ptr().as_unsafe_any_origin()
        var optr = out.obs.unsafe_ptr().as_unsafe_any_origin()
        var nchunks = _num_chunks(nb) if threads <= 0 else min(nb, threads)
        var per = (nb + nchunks - 1) // nchunks
        var t0 = perf_counter_ns()

        def task(c: Int) {imm self, imm dptr, imm optr, imm per, imm nb, imm D, imm K, imm seed}:
            var buf = FrameBuffers(self.prog)
            var lo = c * per
            var hi = min(nb, lo + per)
            for b in range(lo, hi):
                var rng = batch_rng(seed, b)
                buf.run(self.prog, rng)
                for i in range(D * FW):
                    dptr[unsafe_offset=b * D * FW + i] = buf.dets[i]
                for i in range(K * FW):
                    optr[unsafe_offset=b * K * FW + i] = buf.obs[i]

        parallelize(task, nchunks)
        out.seconds = Float64(perf_counter_ns() - t0) / 1e9
        return out^

    def sample_measurements(self, shots: Int, seed: UInt64 = 1234, threads: Int = 0) -> MeasurementSamples:
        """Raw measurement records (reference XOR frame flips) per shot."""
        var out = MeasurementSamples(shots, self.prog.num_measurements)
        var nb = out.num_batches
        if nb == 0:
            return out^
        var M = self.prog.num_measurements
        var bptr = out.bits.unsafe_ptr().as_unsafe_any_origin()
        var nchunks = _num_chunks(nb) if threads <= 0 else min(nb, threads)
        var per = (nb + nchunks - 1) // nchunks
        var t0 = perf_counter_ns()

        def task(c: Int) {imm self, imm bptr, imm per, imm nb, imm M, imm seed}:
            var buf = FrameBuffers(self.prog)
            var lo = c * per
            var hi = min(nb, lo + per)
            for b in range(lo, hi):
                var rng = batch_rng(seed, b)
                buf.run(self.prog, rng)
                for m in range(M):
                    var refw: UInt64 = ~UInt64(0) if self.prog.reference[m] != 0 else 0
                    for w in range(FW):
                        bptr[unsafe_offset=(b * M + m) * FW + w] = buf.recs[m * FW + w] ^ refw

        parallelize(task, nchunks)
        out.seconds = Float64(perf_counter_ns() - t0) / 1e9
        return out^

    def ops_per_shot(self) -> Int:
        return self.prog.num_circuit_ops

    def throughput(self, samples: DetectorSamples) -> Float64:
        """Circuit instructions x shots per second for a finished sampling run."""
        return Float64(self.prog.num_circuit_ops) * Float64(samples.shots) / max(samples.seconds, 1e-12)
