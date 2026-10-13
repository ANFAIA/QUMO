"""Tile scheduler: packs a fused gate plan into register-blocked GPU gate runs.

Input: the CPU runner's fused plan (`qumo.sim.runner.build_plan`: runs of
1-qubit gates already merged per wire). Output: a `TileProgram` -- device
tables plus a list of `TileRun`s, each one `gk_tile` launch (see
`qumo.gpu.kernels`):

  * A run fixes a set S of k "tile" qubits: the `low` = min(low_bits, n)
    lowest qubits (coalescing; low_bits = GpuProfile.low, LOW_MIN..LOW_BITS)
    plus k - low "high" ones. Each thread block
    loads one tile (2^k amplitudes that differ only in S), applies every gate
    of the run and writes it back: one read and one write of the state per
    run instead of per gate (qsim / cuStateVec gate blocking).
  * Only the *targets* of non-diagonal gates must lie in S (the X target of
    CX / CCX, the pair of SWAP / CSWAP, both qubits of a dense 4x4, the qubit
    of a dense 1q gate). Controls and diagonal gates (Z, S, T, RZ, P, CZ, CP,
    CRZ, RZZ, CCZ, the global phase) test bits of the global amplitude index,
    so they can act on any qubit and never consume tile capacity. A QFT is
    then a handful of runs: only its Hadamards need tile slots.
  * Greedy packing with commutation (like qsim's gate fuser and the CPU
    runner's reordering window): scan the remaining gates in order; take a
    gate if it touches no *blocked* qubit and its new target qubits still fit
    in S; otherwise block all its qubits (a later gate on a blocked qubit must
    stay after it). Gates on disjoint qubits commute, so the order inside a
    run, and across runs, preserves the circuit's semantics.
  * Unused slots of S are filled with the lowest free qubits.
  * k = the device's tile size (GpuProfile.tile_bits: 11 qubits in fp32 and
    10 in fp64 by default, up to 2 more on GPUs with large shared memory),
    lowered for small states so that a launch keeps >= 2^MIN_TILES_LOG2
    blocks; states with n <= k are a single tile.
  * The global phase is one more diagonal op in the last run; an initial
    basis state |init> can be written by the first run's load (no extra pass).
  * Inside a run, threads keep 2^nr amplitudes in registers. The run is cut
    into *segments*, each with a register set R of nr tile qubits that must
    contain the targets of its gates. Greedy: keep R while gates fit, else
    open a new R holding the next gate's targets plus those of the following
    gates (lookahead). Each change of R is one shared-memory transpose,
    except at the run's ends when R has no low qubit (direct load / store).
  * finalize() then builds the device op records (kernels.DOP): register
    positions, matrix offsets, and control / parity masks split into the
    part outside the register set and per-register bits.
  * A run holds at most RUN_MAX_OPS ops / RUN_MAX_MATS matrix values (they
    are cached in shared memory by the kernel).
  * Consecutive diagonal gates of a run on at most diag_width qubits become
    one phase-table op (TK.DGT): a QFT's controlled phases cost one complex
    multiply per amplitude per group instead of one per gate.

Everything here is host-only and deterministic (unit-tested in emulation and
by `tests/test_gpu.mojo` against the CPU state vector).
"""

from std.bit import pop_count
from std.math import cos, sin, atan2

from qumo.ir.gates import Op, M2, M4
from qumo.sim.runner import KOp, K
from qumo.gpu.kernels import (
    TK, TOP, DOP, CODE_CTRL, DIAG_MAX_WIDTH, LOW_BITS, LOW_MIN, REG_BITS, SEG_W, SEG_FLAG_SHIFT, SEG_TRANSPOSE, SEG_DIRECT, FIN_NATURAL,
    FIN_TRANSPOSE, FIN_DIRECT, RUN_MAX_OPS, RUN_MAX_MATS,
)

comptime TILE_WINDOW = 4096  # pending ops examined per run
comptime MIN_TILES_LOG2 = 6  # keep >= 64 blocks per launch when the state allows


@fieldwise_init
struct TileRun(ImplicitlyCopyable, Writable):
    """One gk_tile launch: ops [o0, o1) and matrix values [m0, m1) of the
    tables, high tile qubits hp[hoff : hoff + nh], segments [s0, s1) (set by
    finalize), fin = how the run ends (kernels FIN_NATURAL / FIN_TRANSPOSE /
    FIN_DIRECT)."""

    var o0: Int
    var o1: Int
    var hoff: Int
    var nh: Int
    var s0: Int
    var s1: Int
    var fin: Int
    var m0: Int
    var m1: Int

    def write_to(self, mut w: Some[Writer]):
        w.write("TileRun(ops ", self.o0, "..", self.o1, ", nh=", self.nh, ", segments ", self.s0, "..", self.s1, ")")


@always_inline
def is_x_matrix(m: M2) -> Bool:
    return (abs(m.re[0]) + abs(m.im[0]) + abs(m.re[3]) + abs(m.im[3])) < 1e-15 and m.re[1] == 1.0 and m.re[2] == 1.0 and m.im[1] == 0.0 and m.im[2] == 0.0


@always_inline
def _i32(mask: Int) -> Int32:
    """Bit pattern of a mask < 2^32 as Int32 (qubit 31 sets the sign bit)."""
    if mask >= (1 << 31):
        return Int32(mask - (1 << 32))
    return Int32(mask)


def tile_geometry(n: Int, tb: Int, low_bits: Int = LOW_BITS) -> Tuple[Int, Int]:
    """(k, low) for an n-qubit state, a maximum of tb tile qubits and
    `low_bits` always-resident low qubits (GpuProfile.low)."""
    var low = min(max(low_bits, LOW_MIN), n)
    if n <= tb:
        return (n, low)
    var k = max(low + 3, min(tb, n - MIN_TILES_LOG2))
    if k >= n:
        k = n
    return (k, low)


def _is_real(m: M2) -> Bool:
    return m.im[0] == 0.0 and m.im[1] == 0.0 and m.im[2] == 0.0 and m.im[3] == 0.0


def _row_phase(m: M2, i: Int) -> Tuple[Bool, Float64, Float64, Float64]:
    """Row i of m as e^{i t} (x, y) with x, y real: (ok, t, x, y); ok when
    the row's entries share a phase up to sign (residual <= 1e-15)."""
    var ar = m.re[2 * i]
    var ai = m.im[2 * i]
    var br = m.re[2 * i + 1]
    var bi = m.im[2 * i + 1]
    var t = atan2(ai, ar) if ar * ar + ai * ai >= br * br + bi * bi else atan2(bi, br)
    var c = cos(t)
    var sn = sin(t)
    var ok = abs(ai * c - ar * sn) <= 1e-15 and abs(bi * c - br * sn) <= 1e-15
    return (ok, t, ar * c + ai * sn, br * c + bi * sn)


def split_1q_ok(m: M2) -> Bool:
    """A complex 1q gate is split (TileProgram.split_1q) only when it is
    U = diag(e^{i t0}, e^{i t1}) . R with R real (each row's entries share
    a phase up to sign: RZ . RY, RZ . H, phase . RX-free real products --
    the RY / RZ ansatz form). Then the split costs G1R (4 multiply-adds per
    amplitude) + one diagonal that joins a phase table, against 8 for a
    dense G1. A generic U3 would need a second diagonal before R (two per
    gate: +17% fp64 multiply-adds on random circuits), so it stays dense."""
    return _row_phase(m, 0)[0] and _row_phase(m, 1)[0]


def kop_mats(op: KOp, split_1q: Bool = False) -> Int:
    """Matrix values the op takes in the tables (before diagonal fusion,
    which only shrinks a run's tables): an upper bound (8 for any 1q op), or
    with split_1q the exact count for uncontrolled 1q ops (what _emit /
    _one_q produce), so splitting never cuts runs early."""
    if op.kind == K.TWO:
        return 32
    if split_1q and op.kind == K.ONE:
        if is_x_matrix(op.m):
            return 0
        if op.m.is_diagonal() or _is_real(op.m):
            return 4
        return 8  # dense G1, or split: real 2x2 (4) + trailing diagonal (4)
    return 8


def kop_ops(op: KOp, split_1q: Bool = False) -> Int:
    """Tile ops the plan op emits (before diagonal fusion; exact)."""
    if split_1q and op.kind == K.ONE:
        var m = op.m
        if not is_x_matrix(m) and not m.is_diagonal() and not _is_real(m) and split_1q_ok(m):
            return 2  # G1R + trailing diagonal
    return 1


def kop_qubits(op: KOp) raises -> Int:
    """Mask of every qubit the op touches (QM-E475 for ops without a kernel)."""
    var k = op.kind
    if k == K.ONE:
        return 1 << op.a
    if k == K.OTHER:
        var g = op.inst
        if g.kind != Op.CCX and g.kind != Op.CCZ and g.kind != Op.CSWAP:
            raise Error("QM-E475: the GPU executor has no kernel for opcode " + String(g.kind) + ". Hint: run this circuit on the CPU state vector (qumo.sim.runner).")
        return (1 << g.q0) | (1 << g.q1) | (1 << g.q2)
    return (1 << op.a) | (1 << op.b)


def kop_targets(op: KOp) -> Int:
    """Mask of the qubits the op needs inside the tile (non-diagonal targets)."""
    var k = op.kind
    if k == K.ONE:
        return 0 if (op.m.is_diagonal() and not is_x_matrix(op.m)) else 1 << op.a
    if k == K.CX:
        return 1 << op.b
    if k == K.C1:
        return 0 if op.m.is_diagonal() else 1 << op.b
    if k == K.SWAP or k == K.TWO:
        return (1 << op.a) | (1 << op.b)
    if k == K.OTHER:
        var g = op.inst
        if g.kind == Op.CCX:
            return 1 << g.q2
        if g.kind == Op.CSWAP:
            return (1 << g.q1) | (1 << g.q2)
    return 0  # CPHASE, RZZ, CCZ: diagonal


struct TileProgram[dt: DType](Movable):
    """Device tables + launch list for one flush of gates (see module doc)."""

    var n: Int
    var k: Int
    var low: Int
    var nr: Int  # register qubits per thread (2^nr amplitudes)
    var ops: List[Int32]  # host records, TOP Int32 per op: kind, a, b, 0, moff, pa, pb, cm
    var dops: List[Int32]  # device records, DOP Int32 per op (kernels.DOP; built by finalize)
    var mats: List[Scalar[Self.dt]]
    var hp: List[Int32]  # per run: ascending high tile qubits
    var diag_width: Int  # widest fused diagonal table (TK.DGT); <= 1: no fusion
    var split_1q: Bool  # complex uncontrolled 1q gates of the form diagonal . real 2x2 as G1R + diagonal
    var dg_gates: List[Float64]  # fused diagonal gates: pm, am, c0r, c0i, c1r, c1i each
    var dg_groups: List[Int]  # per DGT op: first gate, gate count (pairs)
    var segs: List[Int32]  # SEG_W Int32 per segment
    var runs: List[TileRun]

    def __init__(out self, n: Int, tb: Int, low_bits: Int = LOW_BITS, diag_width: Int = DIAG_MAX_WIDTH,
                 split_1q: Bool = False):
        comptime assert REG_BITS >= 1 and REG_BITS <= 4, "QUMO_GPU_REG_BITS must be 1..4 (4-bit register fields)"
        var g = tile_geometry(n, tb, low_bits)
        self.n = n
        self.k = g[0]
        self.low = g[1]
        self.nr = min(REG_BITS, self.k)
        self.ops = List[Int32]()
        self.dops = List[Int32]()
        self.mats = List[Scalar[Self.dt]]()
        self.hp = List[Int32]()
        self.segs = List[Int32]()
        self.runs = List[TileRun]()
        self.diag_width = min(diag_width, DIAG_MAX_WIDTH)
        self.split_1q = split_1q and self.diag_width >= 2
        self.dg_gates = List[Float64]()
        self.dg_groups = List[Int]()

    def num_ops(self) -> Int:
        return len(self.ops) // TOP

    def num_blocks(self) -> Int:
        """Blocks (tiles) per launch."""
        return 1 << (self.n - self.k)

    def threads_per_block(self) -> Int:
        return 1 << (self.k - self.nr)

    def full_registers(self) -> Bool:
        """nr == REG_BITS: the kernel's FULL specialisation applies."""
        return self.nr == REG_BITS

    def heavy(self, r: Int) -> Bool:
        """Run r has SWAP / dense 4x4 ops (needs the HEAVY kernel)."""
        for i in range(self.runs[r].o0, self.runs[r].o1):
            var kind = Int(self.ops[TOP * i])
            if kind == TK.SW or kind == TK.G2:
                return True
        return False

    def num_segments(self) -> Int:
        return len(self.segs) // SEG_W

    def seg_flag(self, i: Int) -> Int:
        """Remap flag of segment i (kernels.SEG_KEEP / SEG_TRANSPOSE)."""
        return (Int(self.segs[SEG_W * i]) >> SEG_FLAG_SHIFT) & 3

    def seg_ops(self, i: Int) -> Tuple[Int, Int]:
        """Absolute op range [begin, end) of segment i (records store it
        relative to the run's first op)."""
        var w1 = Int(self.segs[SEG_W * i + 1]) & 0xFFFFFFFF
        var o0 = 0
        for r in self.runs:
            if i >= r.s0 and i < r.s1:
                o0 = r.o0
        return (o0 + (w1 & 0xFFFF), o0 + (w1 >> 16))

    def seg_qubits(self, i: Int) -> List[Int]:
        """Register set of segment i (ascending local qubits)."""
        var out = List[Int]()
        var w0 = Int(self.segs[SEG_W * i])
        for j in range(self.nr):
            out.append((w0 >> (4 * j)) & 15)
        return out^

    def num_transposes(self) -> Int:
        """Shared-memory transposes over all runs (register-set changes)."""
        var t = 0
        for i in range(self.num_segments()):
            if self.seg_flag(i) == SEG_TRANSPOSE:
                t += 1
        for r in self.runs:
            t += 1 if r.fin == FIN_TRANSPOSE else 0
        return t

    def _rec(mut self, kind: Int, a: Int32, b: Int32, cm: Int32 = 0):
        self.ops.append(Int32(kind))
        self.ops.append(a)
        self.ops.append(b)
        self.ops.append(0)
        self.ops.append(Int32(len(self.mats)))
        self.ops.append(-1)  # pa, pb: set by finalize
        self.ops.append(-1)
        self.ops.append(cm)

    def _m(mut self, v: Float64):
        self.mats.append(Scalar[Self.dt](v))

    def _mat2(mut self, m: M2):
        for j in range(4):
            self._m(m.re[j])
        for j in range(4):
            self._m(m.im[j])

    def _one_q(mut self, a: Int32, m: M2, cm: Int32 = 0, q: Int = -1):
        """Dense 1q op: G1R when every imaginary part is exactly 0 (H, RY, real
        ansatz products: exact, half the flops), else G1 -- or, with
        split_1q, an uncontrolled complex U (global qubit q) of the form
        U = D . R (split_1q_ok: RZ . RY products) as G1R(R) then diag(D):
        4 instead of 8 multiply-adds per amplitude for the 2x2, and the
        diagonals of many qubits join one phase table (TK.DGT, up to 6
        qubits per complex multiply). For GPUs whose fp64 (or fp32)
        arithmetic, not memory, bounds the run (GpuProfile.split_1q)."""
        var real = _is_real(m)
        if not real and self.split_1q and cm == 0 and q >= 0 and split_1q_ok(m):
            var r0 = _row_phase(m, 0)
            var r1 = _row_phase(m, 1)
            self._rec(TK.G1R, a, 0, cm)
            self._m(r0[2])
            self._m(r0[3])
            self._m(r1[2])
            self._m(r1[3])
            self._diag(1 << q, 0, cos(r0[1]), sin(r0[1]), cos(r1[1]), sin(r1[1]))
            return
        if real:
            self._rec(TK.G1R, a, 0, cm)
            for j in range(4):
                self._m(m.re[j])
        else:
            self._rec(TK.G1, a, 0, cm)
            self._mat2(m)

    def _diag(mut self, pm: Int, am: Int, c0r: Float64, c0i: Float64, c1r: Float64, c1i: Float64):
        """Diagonal op: where all of `am` is set multiply by c[parity(g & pm)]."""
        self._rec(TK.DG, _i32(pm), _i32(am))
        self._m(c0r)
        self._m(c0i)
        self._m(c1r)
        self._m(c1i)

    def _begin_run(mut self, smask: Int, nhigh: Int) -> Tuple[Int, List[Int]]:
        """Fix S = low qubits + smask's high qubits (+ lowest free padding);
        returns (hoff, local index of each qubit, -1 outside S)."""
        var hoff = len(self.hp)
        var loc = List[Int](length=max(self.n, 1), fill=-1)
        for q in range(self.low):
            loc[q] = q
        var pad = self.k - self.low - nhigh
        var r = self.low
        for q in range(self.low, self.n):
            var take = (smask >> q) & 1 == 1
            if not take and pad > 0:
                take = True
                pad -= 1
            if take:
                self.hp.append(Int32(q))
                loc[q] = r
                r += 1
        return (hoff, loc^)

    def _emit(mut self, op: KOp, loc: List[Int]) raises:
        """One plan op -> one tile op (targets local, controls/diagonals global)."""
        var k = op.kind
        if k == K.ONE:
            var m = op.m
            if is_x_matrix(m):
                self._rec(TK.XC, Int32(loc[op.a]), 0)
            elif m.is_diagonal():
                self._diag(1 << op.a, 0, m.re[0], m.im[0], m.re[3], m.im[3])
            else:
                self._one_q(Int32(loc[op.a]), m, 0, op.a)
        elif k == K.CX:
            self._rec(TK.XC, Int32(loc[op.b]), 0, _i32(1 << op.a))
        elif k == K.CPHASE:
            self._diag(0, (1 << op.a) | (1 << op.b), op.f0, op.f1, op.f0, op.f1)
        elif k == K.C1:
            if op.m.is_diagonal():
                self._diag(1 << op.b, 1 << op.a, op.m.re[0], op.m.im[0], op.m.re[3], op.m.im[3])
            else:
                self._one_q(Int32(loc[op.b]), op.m, _i32(1 << op.a))
        elif k == K.RZZ:
            # exp(-i t/2 Z Z): e^{-i t/2} on even parity, e^{+i t/2} on odd
            self._diag((1 << op.a) | (1 << op.b), 0, op.f0, -op.f1, op.f0, op.f1)
        elif k == K.SWAP:
            self._rec(TK.SW, Int32(loc[op.a]), Int32(loc[op.b]))
        elif k == K.TWO:
            self._rec(TK.G2, Int32(loc[op.a]), Int32(loc[op.b]))
            for j in range(16):
                self._m(op.m4.re[j])
            for j in range(16):
                self._m(op.m4.im[j])
        else:
            var g = op.inst
            if g.kind == Op.CCX:
                self._rec(TK.XC, Int32(loc[g.q2]), 0, _i32((1 << g.q0) | (1 << g.q1)))
            elif g.kind == Op.CCZ:
                self._diag(0, (1 << g.q0) | (1 << g.q1) | (1 << g.q2), -1.0, 0.0, -1.0, 0.0)
            else:
                self._rec(TK.SW, Int32(loc[g.q1]), Int32(loc[g.q2]), _i32(1 << g.q0))

    def add_run(mut self, plan: List[KOp], sel: List[Int], smask: Int, nhigh: Int) raises:
        """Append one run made of plan[sel[0]], plan[sel[1]], ... (in order)."""
        var o0 = self.num_ops()
        var m0 = len(self.mats)
        var r = self._begin_run(smask, nhigh)
        for i in sel:
            self._emit(plan[i], r[1])
        if self.diag_width >= 2:
            self._fuse_diagonals(o0, m0, r[0])
        if self.num_ops() - o0 > RUN_MAX_OPS or len(self.mats) - m0 > RUN_MAX_MATS:
            raise Error("QM-E475: internal: a GPU tile run exceeds its shared-memory cache (" + String(self.num_ops() - o0)
                        + " ops, " + String(len(self.mats) - m0) + " matrix values). Hint: report this circuit; run it on the CPU state vector.")
        self.runs.append(TileRun(o0, self.num_ops(), r[0], len(self.hp) - r[0], 0, 0, 0, m0, len(self.mats)))

    def _fuse_diagonals(mut self, o0: Int, m0: Int, hoff: Int):
        """Replace sequences of consecutive diagonal ops of the run being
        built (ops [o0, end), matrices [m0, end)) whose qubits span at most w
        qubits by one TK.DGT op: a 2^w-entry phase table, products of the
        gates' coefficients as stored in the run (rounded to dt by _diag)
        computed in fp64 and rounded once (filled by finalize, when the table's bit
        order -- qubits outside the register set first -- is known). One
        complex multiply per amplitude replaces one per gate (the diagonal
        fusion of cuStateVec / CUDA-Q and qsim). w is the widest <=
        diag_width whose tables all fit the run's matrix cache (RUN_MAX_MATS);
        if none does, width-3 tables are taken in order while they fit."""
        var ops = self._sink_diagonals(o0, hoff)
        var vals = List[Scalar[Self.dt]]()
        for i in range(m0, len(self.mats)):
            vals.append(self.mats[i])
        var nops = len(ops) // TOP
        var budget = RUN_MAX_MATS - 4 - len(vals)  # (4: the global phase may join the run)
        var groups = List[Int]()  # start, end, qubit mask per fused group
        var w = self.diag_width
        while w >= 3:
            groups = _diag_groups(ops, nops, w)
            var extra = 0
            for gi in range(len(groups) // 3):
                extra += _table_size(groups[3 * gi + 2]) - 4 * (groups[3 * gi + 1] - groups[3 * gi])
            if extra <= budget:
                break
            w -= 1
        if w < 3:  # no width fits whole: width-3 tables in order while they fit
            var all = _diag_groups(ops, nops, min(3, self.diag_width))
            groups = List[Int]()
            for gi in range(len(all) // 3):
                var cost = _table_size(all[3 * gi + 2]) - 4 * (all[3 * gi + 1] - all[3 * gi])
                if cost <= budget:
                    budget -= cost
                    for f in range(3):
                        groups.append(all[3 * gi + f])
        self.ops.shrink(TOP * o0)
        self.mats.shrink(m0)
        var i = 0
        var g = 0
        while i < nops:
            if g < len(groups) // 3 and groups[3 * g] == i:
                var j = groups[3 * g + 1]
                var gmask = groups[3 * g + 2]
                var gfirst = len(self.dg_gates) // 6
                for o in range(i, j):
                    var mo = Int(ops[TOP * o + 4]) - m0
                    self.dg_gates.append(Float64(Int(ops[TOP * o + 1]) & 0xFFFFFFFF))
                    self.dg_gates.append(Float64(Int(ops[TOP * o + 2]) & 0xFFFFFFFF))
                    for v in range(4):
                        self.dg_gates.append(Float64(vals[mo + v]))
                self.dg_groups.append(gfirst)
                self.dg_groups.append(j - i)
                self._rec(TK.DGT, _i32(gmask), Int32(len(self.dg_groups) // 2 - 1))
                for _ in range(_table_size(gmask)):
                    self._m(0.0)  # filled by _build_device_records
                i = j
                g += 1
            else:
                self._copy_op(ops, vals, i, m0)
                i += 1

    def _sink_diagonals(self, o0: Int, hoff: Int) -> List[Int32]:
        """Host records of the run being built (ops [o0, end)), with each
        diagonal op moved later past the following non-diagonal ops whose
        targets it does not touch (they commute; controls commute with
        diagonals), so that diagonals gather into long consecutive sequences
        for _fuse_diagonals. When a non-diagonal op targets a pending
        diagonal's qubit, every pending diagonal is emitted before it."""
        var out = List[Int32]()
        var pend = List[Int]()
        var pmask = 0
        var nops = (len(self.ops) - TOP * o0) // TOP
        for i in range(nops):
            var b = TOP * (o0 + i)
            var kind = Int(self.ops[b])
            if kind == TK.DG:
                pend.append(i)
                pmask |= (Int(self.ops[b + 1]) & 0xFFFFFFFF) | (Int(self.ops[b + 2]) & 0xFFFFFFFF)
                continue
            var tmask = 0
            if kind == TK.G1 or kind == TK.G1R or kind == TK.XC or kind == TK.SW or kind == TK.G2:
                tmask = 1 << self._global_of_local(hoff, Int(self.ops[b + 1]))
                if kind == TK.SW or kind == TK.G2:
                    tmask |= 1 << self._global_of_local(hoff, Int(self.ops[b + 2]))
            if tmask & pmask != 0:
                for p in pend:
                    for f in range(TOP):
                        out.append(self.ops[TOP * (o0 + p) + f])
                pend = List[Int]()
                pmask = 0
            for f in range(TOP):
                out.append(self.ops[b + f])
        for p in pend:
            for f in range(TOP):
                out.append(self.ops[TOP * (o0 + p) + f])
        return out^

    def _global_of_local(self, hoff: Int, q: Int) -> Int:
        """Global qubit of local tile qubit q of the run whose high qubits start at hoff."""
        if q < self.low:
            return q
        return Int(self.hp[hoff + q - self.low])

    def _copy_op(mut self, ops: List[Int32], vals: List[Scalar[Self.dt]], i: Int, m0: Int):
        """Re-append host record i of a run being rebuilt, with its matrix
        values (vals = the run's matrices from m0 on)."""
        var kind = Int(ops[TOP * i])
        var nm = 0
        if kind == TK.G1:
            nm = 8
        elif kind == TK.G1R or kind == TK.DG:
            nm = 4
        elif kind == TK.G2:
            nm = 32
        var mo = Int(ops[TOP * i + 4]) - m0
        for f in range(TOP):
            self.ops.append(ops[TOP * i + f])
        self.ops[len(self.ops) - TOP + 4] = Int32(len(self.mats))
        for v in range(nm):
            self.mats.append(vals[mo + v])

    def add_phase(mut self, phase: Float64):
        """exp(i phase) on the whole state, as the last op of the last run."""
        if len(self.runs) == 0:
            self.add_empty_run()
        self._diag(0, 0, cos(phase), sin(phase), cos(phase), sin(phase))
        self.runs[len(self.runs) - 1].o1 = self.num_ops()
        self.runs[len(self.runs) - 1].m1 = len(self.mats)

    def add_empty_run(mut self):
        """A run without gates (used to write an initial basis state)."""
        var r = self._begin_run(0, 0)
        self.runs.append(TileRun(self.num_ops(), self.num_ops(), r[0], len(self.hp) - r[0], 0, 0, 0, len(self.mats), len(self.mats)))

    def _targets(self, i: Int) -> List[Int]:
        """Local tile qubits op i needs in registers (targets of G1 / SW / G2)."""
        var t = List[Int]()
        var kind = Int(self.ops[TOP * i])
        if kind == TK.G1 or kind == TK.G1R or kind == TK.XC:
            t.append(Int(self.ops[TOP * i + 1]))
        elif kind == TK.SW or kind == TK.G2:
            t.append(Int(self.ops[TOP * i + 1]))
            t.append(Int(self.ops[TOP * i + 2]))
        return t^

    def finalize(mut self):
        """Cut every run into register segments (module doc) and fill the
        ops' register positions pa / pb. Called once, after the last op."""
        var nr = self.nr
        var nat = List[Int]()
        for i in range(nr):
            nat.append(self.k - nr + i)
        for ri in range(len(self.runs)):
            var run = self.runs[ri]
            run.s0 = self.num_segments()
            var last = nat.copy()  # register set of the last emitted segment
            var cur = nat.copy()
            var first = True  # next segment is the run's first (may load directly)
            var start = run.o0
            var i = run.o0
            while i <= run.o1:
                var fits = i < run.o1
                if fits:
                    var t = self._targets(i)
                    for q in t:
                        if not _has(cur, q):
                            fits = False
                if fits:
                    i += 1
                    continue
                if i > start:
                    self._emit_segment(cur, last, start, i, run.o0, first)
                    first = False
                    last = cur.copy()
                if i == run.o1:
                    break
                # new register set: targets of op i, then of the next ops while they fit
                var nxt = self._targets(i)
                var j = i + 1
                while j < run.o1:
                    var tj = self._targets(j)
                    var add = List[Int]()
                    for q in tj:
                        if not _has(nxt, q):
                            add.append(q)
                    if len(nxt) + len(add) > nr:
                        break
                    for q in add:
                        nxt.append(q)
                    j += 1
                for q in cur:  # pad: keep current qubits first, then the highest free ones
                    if len(nxt) < nr and not _has(nxt, q):
                        nxt.append(q)
                var q = self.k - 1
                while len(nxt) < nr:
                    if not _has(nxt, q):
                        nxt.append(q)
                    q -= 1
                sort(nxt)
                cur = nxt^
                start = i
            run.s1 = self.num_segments()
            if _same(last, nat):
                run.fin = FIN_NATURAL
            elif self._coalesced(last):
                run.fin = FIN_DIRECT
            else:
                run.fin = FIN_TRANSPOSE
            self.runs[ri] = run
        self._build_device_records()

    def _global_qubit(self, run: TileRun, q: Int) -> Int:
        """Global qubit of local tile qubit q in a run."""
        if q < self.low:
            return q
        return Int(self.hp[run.hoff + q - self.low])

    def _build_device_records(mut self):
        """Device op records (kernels.DOP), from the host records and each
        op's register set: register positions, matrix offset relative to the
        run, and the control / parity / all-set masks split into the part
        outside the register set (tested once on g0) and per-register bits."""
        self.dops = List[Int32](capacity=DOP * self.num_ops())
        var nrg = 1 << self.nr
        for ri in range(len(self.runs)):
            var run = self.runs[ri]
            for si in range(run.s0, run.s1):
                var regs = self.seg_qubits(si)
                var gmask = List[Int]()  # global bit of each register position
                var rmask = 0
                for q in regs:
                    var b = 1 << self._global_qubit(run, q)
                    gmask.append(b)
                    rmask |= b
                var rg = self.seg_ops(si)
                for o in range(rg[0], rg[1]):
                    var kind = Int(self.ops[TOP * o])
                    var mo = Int(self.ops[TOP * o + 4]) - run.m0
                    var code = kind
                    var d1: Int
                    var d2 = 0
                    var d3 = 0
                    if kind == TK.DGT:
                        var t = self._diag_table(run, o, regs, gmask, rmask)
                        d1 = t[0]
                        d2 = t[1]
                    elif kind == TK.DG:
                        var pm = Int(self.ops[TOP * o + 1]) & 0xFFFFFFFF
                        var am = Int(self.ops[TOP * o + 2]) & 0xFFFFFFFF
                        d1 = pm & ~rmask
                        d2 = am & ~rmask
                        for j in range(nrg):
                            var par = 0
                            var has = True
                            for i in range(self.nr):
                                if (j >> i) & 1 == 1 and gmask[i] & pm != 0:
                                    par ^= 1
                                if (j >> i) & 1 == 0 and gmask[i] & am != 0:
                                    has = False
                            d3 |= par << j
                            if has:
                                d3 |= 1 << (16 + j)
                    else:
                        var pa = Int(self.ops[TOP * o + 5])
                        var pb = Int(self.ops[TOP * o + 6])
                        code |= pa << 3
                        if pb >= 0:
                            code |= pb << 5
                        var cm = Int(self.ops[TOP * o + 7]) & 0xFFFFFFFF
                        if cm != 0:
                            code |= CODE_CTRL
                        d1 = cm & ~rmask
                        for j in range(nrg):
                            var ok = True
                            for i in range(self.nr):
                                if (j >> i) & 1 == 0 and gmask[i] & cm != 0:
                                    ok = False
                            if ok:
                                d3 |= 1 << j
                    self.dops.append(_i32(code | (mo << 8)))
                    self.dops.append(_i32(d1))
                    self.dops.append(_i32(d2))
                    self.dops.append(_i32(d3))

    def _diag_table(mut self, run: TileRun, o: Int, regs: List[Int], gmask: List[Int], rmask: Int) -> Tuple[Int, Int]:
        """Fill the phase table of DGT op o for its segment's register set and
        return its device words (d1, d2). Table bit order: the qubits outside
        the register set (ascending; the kernel gathers them from g0), then
        the register qubits."""
        var qm = Int(self.ops[TOP * o + 1]) & 0xFFFFFFFF
        var grp = Int(self.ops[TOP * o + 2])
        var mo = Int(self.ops[TOP * o + 4])
        var qubit_of_bit = List[Int]()  # global qubit of each table bit
        var d1 = 0
        for q in range(32):
            if (qm >> q) & 1 == 1 and (rmask >> q) & 1 == 0:
                d1 |= q << (5 * len(qubit_of_bit))
                qubit_of_bit.append(q)
        var nout = len(qubit_of_bit)
        var d2 = nout << 16
        for i in range(REG_BITS):
            var tb = 7
            if i < len(regs) and gmask[i] & qm != 0:
                tb = len(qubit_of_bit)
                var q = 0
                while (gmask[i] >> q) & 1 == 0:
                    q += 1
                qubit_of_bit.append(q)
            d2 |= tb << (3 * i)
        var w = len(qubit_of_bit)
        var g0 = self.dg_groups[2 * grp]
        var ng = self.dg_groups[2 * grp + 1]
        for idx in range(1 << w):
            var bits = 0  # the global-mask pattern of this table entry
            for t in range(w):
                if (idx >> t) & 1 == 1:
                    bits |= 1 << qubit_of_bit[t]
            var fr = 1.0
            var fi = 0.0
            for g in range(g0, g0 + ng):
                var pm = Int(self.dg_gates[6 * g])
                var am = Int(self.dg_gates[6 * g + 1])
                if bits & am != am:
                    continue
                var odd = Int(pop_count(bits & pm)) & 1 == 1
                var cr = self.dg_gates[6 * g + 4] if odd else self.dg_gates[6 * g + 2]
                var ci = self.dg_gates[6 * g + 5] if odd else self.dg_gates[6 * g + 3]
                var nr_ = fr * cr - fi * ci
                fi = fr * ci + fi * cr
                fr = nr_
            self.mats[mo + 2 * idx] = Scalar[Self.dt](fr)
            self.mats[mo + 2 * idx + 1] = Scalar[Self.dt](fi)
        return (d1, d2)

    def _coalesced(self, regs: List[Int]) -> Bool:
        """A register set whose local qubits are all >= low keeps thread bits
        0..low-1 on local bits 0..low-1: global loads / stores in that map
        stay coalesced (kernels SEG_DIRECT / FIN_DIRECT)."""
        for q in regs:
            if q < self.low:
                return False
        return True

    def _emit_segment(mut self, cur: List[Int], last: List[Int], o0: Int, o1: Int, run_o0: Int, first: Bool):
        """Packed record (kernels.SEG_W): register set + remap flag, op range
        relative to the run's first op."""
        var w0 = 0
        for i in range(len(cur)):
            w0 |= cur[i] << (4 * i)
        if not _same(cur, last):
            var flag = SEG_DIRECT if (first and self._coalesced(cur)) else SEG_TRANSPOSE
            w0 |= flag << SEG_FLAG_SHIFT
        self.segs.append(Int32(w0))
        self.segs.append(_i32((o0 - run_o0) | ((o1 - run_o0) << 16)))
        for i in range(o0, o1):
            var kind = Int(self.ops[TOP * i])
            if kind == TK.G1 or kind == TK.G1R or kind == TK.XC or kind == TK.SW or kind == TK.G2:
                var a = Int(self.ops[TOP * i + 1])
                var pa = _pos(cur, a)
                var pb = -1
                if kind == TK.SW or kind == TK.G2:
                    var b = Int(self.ops[TOP * i + 2])
                    pb = _pos(cur, b)
                    if kind == TK.SW and pa > pb:  # symmetric: store pa < pb
                        self.ops[TOP * i + 1] = Int32(b)
                        self.ops[TOP * i + 2] = Int32(a)
                        var t = pa
                        pa = pb
                        pb = t
                self.ops[TOP * i + 5] = Int32(pa)
                self.ops[TOP * i + 6] = Int32(pb)

    def pad_tables(mut self):
        """Never-empty tables (zero-size device buffers are not portable)."""
        if len(self.ops) == 0:
            for _ in range(TOP):
                self.ops.append(0)
        if len(self.dops) == 0:
            for _ in range(DOP):
                self.dops.append(0)
        if len(self.mats) == 0:
            self.mats.append(0)
        if len(self.hp) == 0:
            self.hp.append(0)
        if len(self.segs) == 0:
            for _ in range(SEG_W):
                self.segs.append(0)


def build_tile_program[dt: DType](plan: List[KOp], n: Int, phase: Float64, tb: Int,
                                  low_bits: Int = LOW_BITS, diag_width: Int = DIAG_MAX_WIDTH,
                                  split_1q: Bool = False) raises -> TileProgram[dt]:
    """Greedy commutation-aware packing of `plan` into tile runs (module doc)."""
    var prog = TileProgram[dt](n, tb, low_bits, diag_width, split_1q)
    var split = prog.split_1q
    var np_ = len(plan)
    var qmasks = List[Int](capacity=np_)
    var tmasks = List[Int](capacity=np_)
    for op in plan:
        var m = kop_qubits(op)
        if m >= (1 << n):
            raise Error("QM-E474: plan op acts on a qubit >= " + String(n) + ". Hint: allocate GpuStateVector(circuit.num_qubits).")
        qmasks.append(m)
        tmasks.append(kop_targets(op))
    var done = List[Bool](length=np_, fill=False)
    var cap = prog.k - prog.low
    var lowmask = (1 << prog.low) - 1
    var allmask = (1 << n) - 1
    # leave room for the global-phase op that may join the last run
    var max_ops = RUN_MAX_OPS - 1
    var max_mats = RUN_MAX_MATS - 4
    var first = 0
    while first < np_:
        var smask = lowmask
        var nhigh = 0
        var blocked = 0
        var sel = List[Int]()
        var nmats = 0
        var nemit = 0  # tile ops the selection emits
        var examined = 0
        var i = first
        while i < np_ and examined < TILE_WINDOW and nemit < max_ops:
            if not done[i]:
                examined += 1
                var qm = qmasks[i]
                if qm & blocked == 0:
                    var newh = tmasks[i] & ~smask
                    var cnt = Int(pop_count(newh))
                    var mt = kop_mats(plan[i], split)
                    var ne = kop_ops(plan[i], split)
                    if nmats + mt > max_mats or nemit + ne > max_ops:
                        break
                    if nhigh + cnt <= cap:
                        nmats += mt
                        nemit += ne
                        smask |= newh
                        nhigh += cnt
                        sel.append(i)
                        done[i] = True
                    else:
                        blocked |= qm
                else:
                    blocked |= qm
                if blocked == allmask:
                    break  # every qubit blocked: nothing else can join
            i += 1
        while first < np_ and done[first]:
            first += 1
        prog.add_run(plan, sel, smask & ~lowmask, nhigh)
    if phase != 0.0:
        prog.add_phase(phase)
    prog.finalize()
    return prog^


def _diag_groups(ops: List[Int32], nops: Int, width: Int) -> List[Int]:
    """Maximal sequences (>= 2 ops) of consecutive DG host records spanning
    at most `width` qubits: (start, end, qubit mask) triples."""
    var out = List[Int]()
    var i = 0
    while i < nops:
        if Int(ops[TOP * i]) != TK.DG:
            i += 1
            continue
        var gmask = 0
        var j = i
        while j < nops and Int(ops[TOP * j]) == TK.DG:
            var q = (Int(ops[TOP * j + 1]) & 0xFFFFFFFF) | (Int(ops[TOP * j + 2]) & 0xFFFFFFFF)
            if Int(pop_count(gmask | q)) > width:
                break
            gmask |= q
            j += 1
        if j - i >= 2:
            out.append(i)
            out.append(j)
            out.append(gmask)
            i = j
        else:
            i += 1
    return out^


@always_inline
def _table_size(gmask: Int) -> Int:
    """Matrix values of a phase table over the qubits of gmask (2^w complex
    values; at least 4 so every matrix stays 16-byte aligned)."""
    return max(4, 2 << Int(pop_count(gmask)))


@always_inline
def _has(xs: List[Int], v: Int) -> Bool:
    for x in xs:
        if x == v:
            return True
    return False


@always_inline
def _pos(xs: List[Int], v: Int) -> Int:
    for i in range(len(xs)):
        if xs[i] == v:
            return i
    return -1


def _same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True
