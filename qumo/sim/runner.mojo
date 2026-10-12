"""Circuit -> state-vector execution: gate fusion, sampling, trajectories.

Execution strategy is chosen automatically (and reported, for agents):
  * static circuits (terminal measurements only, no noise): evolve once, then
    sample all shots from the final distribution - O(gates * 2^n + shots).
  * dynamic circuits (mid-circuit measurement, reset, feed-forward) or noisy
    circuits: per-shot quantum trajectories with independent RNG streams.
"""

from std.math import cos, sin, atan2
from std.collections import Dict
from std.bit import pop_count

from qumo.ir.gates import (
    Op, M2, M4, matrix_1q, matrix_2q, op_arity, op_is_unitary, op_is_noise,
    op_is_controlled_1q, op_base_1q,
)
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.pauli import PauliSum
from qumo.sim.statevector import (
    StateVector, FPtr, k1_range, kdiag_range, kx_range, kc1_range, kcx_range,
    kcphase_range, krzz_range, kswap_range, k2_range, kscale_range,
)
from qumo.sim.rng import Rng
from max.algorithm import parallelize


struct SampleResult(Copyable, Writable):
    var counts: Dict[String, Int]
    var memory: List[Int]  # classical register per shot (bit c = clbit c)
    var num_clbits: Int
    var shots: Int
    var method: String

    def __init__(out self, num_clbits: Int, shots: Int, method: String):
        self.counts = Dict[String, Int]()
        self.memory = List[Int]()
        self.num_clbits = num_clbits
        self.shots = shots
        self.method = method

    def record(mut self, creg: Int):
        self.memory.append(creg)
        var key = bitstring(creg, self.num_clbits)
        self.counts[key] = self.counts.get(key, 0) + 1

    def probability(self, key: String) -> Float64:
        return Float64(self.counts.get(key, 0)) / Float64(max(self.shots, 1))

    def most_frequent(self) -> String:
        var best = String()
        var bc = -1
        for kv in self.counts.items():
            if kv.value > bc or (kv.value == bc and kv.key < best):
                bc = kv.value
                best = kv.key
        return best

    def sorted_keys(self) -> List[String]:
        var keys = List[String]()
        for k in self.counts.keys():
            keys.append(k)
        sort(keys)
        return keys^

    def write_to(self, mut w: Some[Writer]):
        w.write("{")
        var first = True
        for k in self.sorted_keys():
            if not first:
                w.write(", ")
            first = False
            w.write("\"", k, "\": ", self.counts.get(k, 0))
        w.write("}")


def bitstring(value: Int, width: Int) -> String:
    """Qiskit-style big-endian key: leftmost char = highest clbit."""
    var s = String()
    var b = width - 1
    while b >= 0:
        s += "1" if (value >> b) & 1 == 1 else "0"
        b -= 1
    return s


# ---------------------------------------------------------------------------
# Gate dispatch
# ---------------------------------------------------------------------------
def gate_matrix_1q(g: Instruction, params: List[Float64]) -> M2:
    return matrix_1q(g.kind, g.p0.resolve(params), g.p1.resolve(params), g.p2.resolve(params))


def apply_instruction(mut sv: StateVector, g: Instruction, params: List[Float64]):
    """Apply one unitary instruction (no fusion)."""
    var k = g.kind
    if op_arity(k) == 1 and op_is_unitary(k):
        if k == Op.X:
            sv.apply_x(g.q0)
        elif k != Op.I:
            sv.apply_1q(gate_matrix_1q(g, params), g.q0)
        return
    if k == Op.CX:
        sv.apply_cx(g.q0, g.q1)
    elif k == Op.CZ:
        sv.apply_cphase(-1.0, 0.0, g.q0, g.q1)
    elif k == Op.CP:
        var l = g.p0.resolve(params)
        sv.apply_cphase(cos(l), sin(l), g.q0, g.q1)
    elif op_is_controlled_1q(k):
        sv.apply_controlled_1q(matrix_1q(op_base_1q(k), g.p0.resolve(params)), g.q0, g.q1)
    elif k == Op.SWAP:
        sv.apply_swap(g.q0, g.q1)
    elif k == Op.RZZ:
        sv.apply_rzz(g.p0.resolve(params), g.q0, g.q1)
    elif op_arity(k) == 2:
        sv.apply_2q(matrix_2q(k, g.p0.resolve(params)), g.q0, g.q1)
    elif k == Op.CCX:
        sv.apply_ccx(g.q0, g.q1, g.q2)
    elif k == Op.CCZ:
        sv.apply_ccz(g.q0, g.q1, g.q2)
    elif k == Op.CSWAP:
        sv.apply_cswap(g.q0, g.q1, g.q2)


struct Fuser(Movable):
    """Accumulates runs of 1-qubit gates per wire into a single 2x2 matrix."""

    var pending: List[M2]
    var has: List[Bool]

    def __init__(out self, n: Int):
        self.pending = List[M2](length=n, fill=M2.identity())
        self.has = List[Bool](length=n, fill=False)

    def push(mut self, m: M2, q: Int):
        if self.has[q]:
            self.pending[q] = m @ self.pending[q]
        else:
            self.pending[q] = m
            self.has[q] = True

    def flush(mut self, mut sv: StateVector, q: Int):
        if self.has[q]:
            var m = self.pending[q]
            # pure X permutation fast-path
            if (abs(m.re[0]) + abs(m.im[0]) + abs(m.re[3]) + abs(m.im[3])) < 1e-15 and m.re[1] == 1.0 and m.re[2] == 1.0 and m.im[1] == 0.0 and m.im[2] == 0.0:
                sv.apply_x(q)
            else:
                sv.apply_1q(m, q)
            self.has[q] = False

    def flush_all(mut self, mut sv: StateVector):
        for q in range(len(self.has)):
            self.flush(sv, q)


# ---------------------------------------------------------------------------
# Execution plan + cache blocking
# ---------------------------------------------------------------------------
struct K:
    comptime ONE = 0  # fused 1q matrix
    comptime CX = 1
    comptime CPHASE = 2  # f0 + i f1 on |11>
    comptime C1 = 3  # controlled 1q matrix (a = control, b = target)
    comptime RZZ = 4  # f0 = cos(t/2), f1 = sin(t/2)
    comptime SWAP = 5
    comptime TWO = 6  # dense 4x4
    comptime OTHER = 7  # 3q gates: applied globally


@fieldwise_init
struct KOp(ImplicitlyCopyable):
    var kind: Int
    var a: Int
    var b: Int
    var m: M2
    var m4: M4
    var f0: Float64
    var f1: Float64
    var inst: Instruction

    @staticmethod
    def make(kind: Int, a: Int, b: Int, inst: Instruction) -> KOp:
        return KOp(kind, a, b, M2.identity(), M4.identity(), 0.0, 0.0, inst)

    def max_qubit(self) -> Int:
        return max(self.a, self.b)


comptime BLOCK_BITS = 15  # 2^15 amplitudes x 16 B = 512 KiB per block (fits L2)
comptime BLOCK_MIN_QUBITS = 19
comptime MIN_BLOCKS_LOG2 = 5  # keep >= 32 blocks so every core gets work
comptime REORDER_WINDOW = 4096  # pending ops examined per scheduling decision
comptime SWAP_GAIN_FIRST = 4  # first remap must also pay for the final restore
comptime SWAP_GAIN = 2  # later remaps: >= 2 global passes saved per swap pass


def _to_kop(g: Instruction, params: List[Float64]) -> KOp:
    var k = g.kind
    if k == Op.CX:
        return KOp.make(K.CX, g.q0, g.q1, g)
    if k == Op.CZ:
        var op = KOp.make(K.CPHASE, g.q0, g.q1, g)
        op.f0 = -1.0
        return op
    if k == Op.CP:
        var l = g.p0.resolve(params)
        var op = KOp.make(K.CPHASE, g.q0, g.q1, g)
        op.f0 = cos(l)
        op.f1 = sin(l)
        return op
    if op_is_controlled_1q(k):
        var op = KOp.make(K.C1, g.q0, g.q1, g)
        op.m = matrix_1q(op_base_1q(k), g.p0.resolve(params))
        return op
    if k == Op.SWAP:
        return KOp.make(K.SWAP, g.q0, g.q1, g)
    if k == Op.RZZ:
        var t = g.p0.resolve(params)
        var op = KOp.make(K.RZZ, g.q0, g.q1, g)
        op.f0 = cos(t / 2)
        op.f1 = sin(t / 2)
        return op
    if op_arity(k) == 2:
        var op = KOp.make(K.TWO, g.q0, g.q1, g)
        op.m4 = matrix_2q(k, g.p0.resolve(params))
        return op
    return KOp.make(K.OTHER, max(g.q0, max(g.q1, g.q2)), -1, g)


def build_plan(circuit: Circuit, params: List[Float64]) raises -> List[KOp]:
    """Unitary part of `circuit` as kernel ops, runs of 1q gates fused per wire."""
    var plan = List[KOp]()
    var n = circuit.num_qubits
    var pending = List[M2](length=n, fill=M2.identity())
    var has = List[Bool](length=n, fill=False)
    var dummy = Instruction.make(Op.U, 0)

    @always_inline
    def flush(q: Int, mut plan: List[KOp], mut has: List[Bool], pending: List[M2]) {imm dummy}:
        if has[q]:
            var op = KOp.make(K.ONE, q, -1, dummy)
            op.m = pending[q]
            plan.append(op)
            has[q] = False

    for g in circuit.ops:
        var k = g.kind
        if not op_is_unitary(k):
            continue
        if g.cond >= 0:
            raise Error("QM-E401: evolve() cannot apply classically-conditioned gates. Hint: use sample(), which runs trajectories for dynamic circuits.")
        var a = op_arity(k)
        if a == 1:
            if k != Op.I:
                var m = gate_matrix_1q(g, params)
                if has[g.q0]:
                    pending[g.q0] = m @ pending[g.q0]
                else:
                    pending[g.q0] = m
                    has[g.q0] = True
            continue
        for j in range(a):
            flush(g.qubit(j), plan, has, pending)
        plan.append(_to_kop(g, params))
    for q in range(n):
        flush(q, plan, has, pending)
    return plan^


def apply_kop(mut sv: StateVector, op: KOp):
    """Apply one kernel op to the whole state (parallel kernels)."""
    var k = op.kind
    if k == K.ONE:
        var m = op.m
        if (abs(m.re[0]) + abs(m.im[0]) + abs(m.re[3]) + abs(m.im[3])) < 1e-15 and m.re[1] == 1.0 and m.re[2] == 1.0 and m.im[1] == 0.0 and m.im[2] == 0.0:
            sv.apply_x(op.a)
        else:
            sv.apply_1q(m, op.a)
    elif k == K.CX:
        sv.apply_cx(op.a, op.b)
    elif k == K.CPHASE:
        sv.apply_cphase(op.f0, op.f1, op.a, op.b)
    elif k == K.C1:
        sv.apply_controlled_1q(op.m, op.a, op.b)
    elif k == K.RZZ:
        var theta = 2.0 * atan2(op.f1, op.f0)
        sv.apply_rzz(theta, op.a, op.b)
    elif k == K.SWAP:
        sv.apply_swap(op.a, op.b)
    elif k == K.TWO:
        sv.apply_2q(op.m4, op.a, op.b)
    else:
        apply_instruction(sv, op.inst, List[Float64]())


@always_inline
def apply_kop_local(rp: FPtr, ip: FPtr, nb: Int, op: KOp):
    """Apply one kernel op sequentially to a block of 2^nb amplitudes."""
    var k = op.kind
    var half = 1 << (nb - 1)
    var quarter = 1 << (nb - 2)
    if k == K.ONE:
        var m = op.m
        if m.is_diagonal():
            kdiag_range(rp, ip, op.a, m.re[0], m.im[0], m.re[3], m.im[3], 0, half)
        else:
            k1_range(rp, ip, op.a, m, 0, half)
    elif k == K.CX:
        kcx_range(rp, ip, op.a, op.b, 0, quarter)
    elif k == K.CPHASE:
        kcphase_range(rp, ip, op.a, op.b, op.f0, op.f1, 0, quarter)
    elif k == K.C1:
        kc1_range(rp, ip, op.a, op.b, op.m, 0, quarter)
    elif k == K.RZZ:
        krzz_range(rp, ip, op.a, op.b, op.f0, op.f1, 0, 1 << nb)
    elif k == K.SWAP:
        kswap_range(rp, ip, op.a, op.b, 0, quarter)
    elif k == K.TWO:
        k2_range(rp, ip, op.a, op.b, op.m4, 0, quarter)


def _apply_block(mut sv: StateVector, plan: List[KOp], lo: Int, hi: Int, nb: Int):
    """Apply plan[lo:hi] (all on qubits < nb) block by block: each 2^nb
    amplitude block is loaded once into cache and receives every op."""
    var rp = sv.re.unsafe_ptr().as_unsafe_any_origin()
    var ip = sv.im.unsafe_ptr().as_unsafe_any_origin()
    var pp = plan.unsafe_ptr().as_unsafe_any_origin()
    var nblocks = 1 << (sv.num_qubits - nb)

    def task(blk: Int) {imm rp, imm ip, imm pp, imm lo, imm hi, imm nb}:
        var base = blk << nb
        var brp = rp.unsafe_offset(base)
        var bip = ip.unsafe_offset(base)
        for i in range(lo, hi):
            apply_kop_local(brp, bip, nb, pp[unsafe_offset=i])

    parallelize(task, nblocks)
    _ = len(plan)


def execute_plan(mut sv: StateVector, plan: List[KOp]):
    """Run a plan. Below BLOCK_MIN_QUBITS every op is one parallel pass (the
    state lives in cache anyway); above, the qubit-reordering cache-blocked
    executor runs (`execute_plan_reordered`)."""
    var n = sv.num_qubits
    if n < BLOCK_MIN_QUBITS:
        for op in plan:
            apply_kop(sv, op)
        return
    execute_plan_reordered(sv, plan, min(BLOCK_BITS, n - MIN_BLOCKS_LOG2))


# ---------------------------------------------------------------------------
# Qubit reordering (index-bit swaps) for cache blocking
#
# Design (cuStateVec swapIndexBits + Qiskit-Aer chunk swaps + qsim blocking):
#   * The state is stored under a logical->physical qubit permutation
#     (`QubitMap`): logical qubit q is amplitude-index bit l2p[q]. Physical
#     bits [0, nb) are "local" (inside one 2^nb-amplitude cache block), the
#     others are "global".
#   * An op can run inside a block pass when every qubit on which it acts
#     NON-diagonally is local (`_op_need`): a diagonal factor or a control on a
#     global bit is a per-block constant (read from the block's base index),
#     so CP/CZ/RZZ/diagonal 1q gates and the controls of CX/CRy/... never
#     force a swap. SWAP gates are free relabels of the map.
#   * Scheduling is greedy with lookahead and commutation: pending ops are
#     scanned in order; an op that cannot run blocks its wires, later ops on
#     other wires may still run (gates on disjoint wires commute).
#   * When the head op needs a global bit, `_select_local` scans ahead and
#     builds the set S (<= nb physical bits) that lets the most upcoming ops
#     run; if that saves >= SWAP_GAIN global passes, ONE `apply_swap_layer`
#     pass exchanges the global bits of S with local bits that are not needed
#     soon (Belady: farthest next use). Otherwise the op runs as one global
#     pass.
#   * At the end the permutation is undone with <= 2 swap-layer passes (any
#     permutation is a product of two involutions), so the StateVector seen
#     by every caller is in canonical little-endian order (qubit k = bit k):
#     amplitudes, probabilities, sampling, expectation, measurement and reset
#     are unaffected.
# ---------------------------------------------------------------------------
struct QubitMap(Copyable, Movable):
    """Logical <-> physical qubit permutation of the amplitude index bits."""

    var l2p: List[Int]
    var p2l: List[Int]

    def __init__(out self, n: Int):
        self.l2p = List[Int](capacity=n)
        self.p2l = List[Int](capacity=n)
        for q in range(n):
            self.l2p.append(q)
            self.p2l.append(q)

    def swap_phys(mut self, x: Int, y: Int):
        """Data at physical bits x and y were exchanged (index-bit swap)."""
        var lx = self.p2l[x]
        var ly = self.p2l[y]
        self.p2l[x] = ly
        self.p2l[y] = lx
        self.l2p[lx] = y
        self.l2p[ly] = x

    def relabel(mut self, a: Int, b: Int):
        """Logical SWAP(a, b): wires a and b exchange their data, no data moves."""
        var pa = self.l2p[a]
        var pb = self.l2p[b]
        self.l2p[a] = pb
        self.l2p[b] = pa
        self.p2l[pb] = a
        self.p2l[pa] = b

    def phys_mask(self, logical_mask: Int) -> Int:
        var out = 0
        var m = logical_mask
        var q = 0
        while m != 0:
            if (m & 1) != 0:
                out |= 1 << self.l2p[q]
            m >>= 1
            q += 1
        return out

    def is_identity(self) -> Bool:
        for q in range(len(self.l2p)):
            if self.l2p[q] != q:
                return False
        return True


@always_inline
def _bit(q: Int) -> Int:
    return 0 if q < 0 else 1 << q


def _op_wires(op: KOp) -> Int:
    """Logical wires the op touches (its dependency footprint)."""
    if op.kind == K.OTHER:
        return _bit(op.inst.q0) | _bit(op.inst.q1) | _bit(op.inst.q2)
    return _bit(op.a) | _bit(op.b)


def _op_need(op: KOp) -> Int:
    """Logical wires that must be local (non-diagonal action) for the op to
    run inside a cache block; controls and diagonal factors may be global."""
    var k = op.kind
    if k == K.ONE:
        return 0 if op.m.is_diagonal() else _bit(op.a)
    if k == K.CX:
        return _bit(op.b)
    if k == K.CPHASE or k == K.RZZ or k == K.SWAP:
        return 0
    if k == K.C1:
        return 0 if op.m.is_diagonal() else _bit(op.b)
    if k == K.TWO:
        return _bit(op.a) | _bit(op.b)
    return _op_wires(op)


def _phys(op: KOp, qm: QubitMap) -> KOp:
    """The op with logical qubits replaced by their physical bits."""
    var p = op
    if op.kind == K.OTHER:
        p.inst.q0 = qm.l2p[op.inst.q0]
        if op.inst.q1 >= 0:
            p.inst.q1 = qm.l2p[op.inst.q1]
        if op.inst.q2 >= 0:
            p.inst.q2 = qm.l2p[op.inst.q2]
        p.a = max(p.inst.q0, max(p.inst.q1, p.inst.q2))
        return p
    p.a = qm.l2p[op.a]
    if op.b >= 0:
        p.b = qm.l2p[op.b]
    return p


@always_inline
def _scale_block(rp: FPtr, ip: FPtr, cr: Float64, ci: Float64, size: Int):
    if cr == 1.0 and ci == 0.0:
        return
    kscale_range(rp, ip, cr, ci, 0, size)


@always_inline
def apply_kop_block(rp: FPtr, ip: FPtr, nb: Int, base: Int, op: KOp):
    """Apply one PHYSICAL op to the block of 2^nb amplitudes starting at index
    `base`. Bits >= nb are global: constant inside the block, read from
    `base`. Requires `_op_need(op)` bits to be < nb (checked by the planner)."""
    var k = op.kind
    var a = op.a
    var b = op.b
    var full = 1 << nb
    var half = 1 << (nb - 1)
    var quarter = 1 << (nb - 2)
    if k == K.ONE:
        var m = op.m
        if a >= nb:  # diagonal on a global bit: one scalar per block
            if (base >> a) & 1 == 1:
                _scale_block(rp, ip, m.re[3], m.im[3], full)
            else:
                _scale_block(rp, ip, m.re[0], m.im[0], full)
        elif m.is_diagonal():
            kdiag_range(rp, ip, a, m.re[0], m.im[0], m.re[3], m.im[3], 0, half)
        else:
            k1_range(rp, ip, a, m, 0, half)
    elif k == K.CX:
        if a >= nb:  # global control: X on the target or nothing
            if (base >> a) & 1 == 1:
                kx_range(rp, ip, b, 0, half)
        else:
            kcx_range(rp, ip, a, b, 0, quarter)
    elif k == K.CPHASE:
        var ga = a >= nb
        var gb = b >= nb
        if ga and gb:
            if (base >> a) & 1 == 1 and (base >> b) & 1 == 1:
                _scale_block(rp, ip, op.f0, op.f1, full)
        elif gb:
            if (base >> b) & 1 == 1:
                kdiag_range(rp, ip, a, 1.0, 0.0, op.f0, op.f1, 0, half)
        elif ga:
            if (base >> a) & 1 == 1:
                kdiag_range(rp, ip, b, 1.0, 0.0, op.f0, op.f1, 0, half)
        else:
            kcphase_range(rp, ip, a, b, op.f0, op.f1, 0, quarter)
    elif k == K.C1:
        var m = op.m
        if a >= nb:  # global control
            if (base >> a) & 1 == 1:
                if b >= nb:  # diagonal on a global target
                    if (base >> b) & 1 == 1:
                        _scale_block(rp, ip, m.re[3], m.im[3], full)
                    else:
                        _scale_block(rp, ip, m.re[0], m.im[0], full)
                elif m.is_diagonal():
                    kdiag_range(rp, ip, b, m.re[0], m.im[0], m.re[3], m.im[3], 0, half)
                else:
                    k1_range(rp, ip, b, m, 0, half)
        elif b >= nb:  # diagonal, global target: phase on control=1 amplitudes
            if (base >> b) & 1 == 1:
                kdiag_range(rp, ip, a, 1.0, 0.0, m.re[3], m.im[3], 0, half)
            else:
                kdiag_range(rp, ip, a, 1.0, 0.0, m.re[0], m.im[0], 0, half)
        else:
            kc1_range(rp, ip, a, b, m, 0, quarter)
    elif k == K.RZZ:
        var cr = op.f0
        var si = op.f1
        var ga = a >= nb
        var gb = b >= nb
        if ga and gb:
            var odd = ((base >> a) ^ (base >> b)) & 1 == 1
            _scale_block(rp, ip, cr, si if odd else -si, full)
        elif ga or gb:
            var l = b if ga else a
            var g = a if ga else b
            var s0 = si if (base >> g) & 1 == 1 else -si
            kdiag_range(rp, ip, l, cr, s0, cr, -s0, 0, half)
        else:
            krzz_range(rp, ip, a, b, cr, si, 0, full)
    elif k == K.SWAP:
        kswap_range(rp, ip, a, b, 0, quarter)
    elif k == K.TWO:
        k2_range(rp, ip, a, b, op.m4, 0, quarter)


def _apply_run(mut sv: StateVector, run: List[KOp], nb: Int):
    """One cache-blocked pass: every 2^nb block receives all ops of `run`
    (physical qubits) while it is resident in L2."""
    var rp = sv.re.unsafe_ptr().as_unsafe_any_origin()
    var ip = sv.im.unsafe_ptr().as_unsafe_any_origin()
    var pp = run.unsafe_ptr().as_unsafe_any_origin()
    var cnt = len(run)
    var nblocks = 1 << (sv.num_qubits - nb)

    def task(blk: Int) {imm rp, imm ip, imm pp, imm cnt, imm nb}:
        var base = blk << nb
        var brp = rp.unsafe_offset(base)
        var bip = ip.unsafe_offset(base)
        for i in range(cnt):
            apply_kop_block(brp, bip, nb, base, pp[unsafe_offset=i])

    parallelize(task, nblocks)
    _ = len(run)


@fieldwise_init
struct _Selection(ImplicitlyCopyable):
    var mask: Int  # physical bits that should be local
    var gain: Int  # upcoming ops that need a currently-global bit and fit


def _select_local(plan: List[KOp], wires: List[Int], need: List[Int], done: List[Bool],
                  head: Int, qm: QubitMap, nb: Int, allw: Int) -> _Selection:
    """Greedy lookahead: the set of <= nb physical bits that lets the longest
    dependency-respecting stream of upcoming ops run inside blocks."""
    var local = (1 << nb) - 1
    var smask = 0
    var blocked = 0
    var gain = 0
    var map = qm.l2p.copy()  # scan-local copy: SWAP gates relabel wires
    var seen = 0
    for j in range(head, len(plan)):
        if done[j]:
            continue
        seen += 1
        if seen > REORDER_WINDOW or blocked == allw:
            break
        var w = wires[j]
        if (w & blocked) != 0:
            blocked |= w
            continue
        var k = plan[j].kind
        if k == K.OTHER:
            blocked |= w
            continue
        if k == K.SWAP:
            var t = map[plan[j].a]
            map[plan[j].a] = map[plan[j].b]
            map[plan[j].b] = t
            continue
        var np = 0
        var nm = need[j]
        var q = 0
        while nm != 0:
            if (nm & 1) != 0:
                np |= 1 << map[q]
            nm >>= 1
            q += 1
        if np == 0:
            continue
        var u = smask | np
        if Int(pop_count(u)) <= nb:
            smask = u
            if (np & ~local) != 0:
                gain += 1
        else:
            blocked |= w
    return _Selection(smask, gain)


def _swap_in(mut sv: StateVector, mut qm: QubitMap, smask: Int, plan: List[KOp],
             need: List[Int], done: List[Bool], head: Int, nb: Int):
    """One index-bit swap pass making every bit of `smask` local; evicts the
    local bits outside `smask` whose next use is farthest away."""
    var local = (1 << nb) - 1
    var incoming = smask & ~local
    var cand = local & ~smask
    var big = len(plan) + REORDER_WINDOW + 1
    var dist = List[Int](length=nb, fill=big)
    var seen_bits = 0
    var seen = 0
    for j in range(head, len(plan)):
        if done[j]:
            continue
        seen += 1
        if seen > REORDER_WINDOW or seen_bits == cand:
            break
        var np = qm.phys_mask(need[j]) & cand & ~seen_bits
        var p = 0
        while np != 0:
            if (np & 1) != 0:
                dist[p] = seen
                seen_bits |= 1 << p
            np >>= 1
            p += 1
    var ins = List[Int]()
    for p in range(nb, sv.num_qubits):
        if (incoming >> p) & 1 == 1:
            ins.append(p)
    # victims: farthest next use first, ties -> higher bit (longer contiguous runs)
    var outs = List[Int]()
    var taken = 0
    for _ in range(len(ins)):
        var best = -1
        for p in range(nb):
            if (cand >> p) & 1 == 0 or (taken >> p) & 1 == 1:
                continue
            if best < 0 or dist[p] > dist[best] or (dist[p] == dist[best] and p > best):
                best = p
        taken |= 1 << best
        outs.append(best)
    sort(outs)
    sv.apply_swap_layer(ins, outs)
    for i in range(len(ins)):
        qm.swap_phys(ins[i], outs[i])


def _collect_run(plan: List[KOp], wires: List[Int], need: List[Int], mut done: List[Bool],
                 head: Int, mut qm: QubitMap, nb: Int, allw: Int, mut run: List[KOp]):
    """Pull every pending op that can run in a block now (in dependency
    order, skipping over blocked wires) into `run` as physical ops."""
    var local = (1 << nb) - 1
    var blocked = 0
    var seen = 0
    for j in range(head, len(plan)):
        if done[j]:
            continue
        seen += 1
        if seen > REORDER_WINDOW or blocked == allw:
            break
        var w = wires[j]
        if (w & blocked) != 0:
            blocked |= w
            continue
        var op = plan[j]
        if op.kind == K.OTHER:
            blocked |= w
            continue
        if op.kind == K.SWAP:
            qm.relabel(op.a, op.b)
            done[j] = True
            continue
        if (qm.phys_mask(need[j]) & ~local) != 0:
            blocked |= w
            continue
        run.append(_phys(op, qm))
        done[j] = True


def _restore_identity(mut sv: StateVector, mut qm: QubitMap):
    """Undo the permutation with <= 2 swap-layer passes: the bit permutation
    (data at physical p must move to bit p2l[p]) is written as tau2 o tau1 of
    two involutions, cycle by cycle (c_i -> c_{-i}, then c_i -> c_{1-i})."""
    if qm.is_identity():
        return
    var n = sv.num_qubits
    var visited = List[Bool](length=n, fill=False)
    var a1 = List[Int]()
    var b1 = List[Int]()
    var a2 = List[Int]()
    var b2 = List[Int]()
    for p in range(n):
        if visited[p] or qm.p2l[p] == p:
            continue
        var cyc = List[Int]()
        var x = p
        while not visited[x]:
            visited[x] = True
            cyc.append(x)
            x = qm.p2l[x]
        var m = len(cyc)
        for i in range(m):
            var j = (m - i) % m
            if i < j:
                a1.append(cyc[i])
                b1.append(cyc[j])
            var j2 = (m + 1 - i) % m
            if i < j2:
                a2.append(cyc[i])
                b2.append(cyc[j2])
    sv.apply_swap_layer(a1, b1)
    for i in range(len(a1)):
        qm.swap_phys(a1[i], b1[i])
    sv.apply_swap_layer(a2, b2)
    for i in range(len(a2)):
        qm.swap_phys(a2[i], b2[i])


def execute_plan_reordered(mut sv: StateVector, plan: List[KOp], nb: Int):
    """Cache-blocked execution with qubit reordering (see the design notes
    above). `nb` = log2 of the block size in amplitudes (BLOCK_BITS by
    default; exposed for tuning). The state is in canonical order on return."""
    var n = sv.num_qubits
    var m = len(plan)
    var wires = List[Int](capacity=m)
    var need = List[Int](capacity=m)
    for op in plan:
        wires.append(_op_wires(op))
        need.append(_op_need(op))
    var done = List[Bool](length=m, fill=False)
    var qm = QubitMap(n)
    var local = (1 << nb) - 1
    var allw = (1 << n) - 1
    var swaps = 0
    var run = List[KOp]()
    var head = 0
    while True:
        while head < m and done[head]:
            head += 1
        if head >= m:
            break
        var op = plan[head]
        if op.kind == K.OTHER:
            apply_kop(sv, _phys(op, qm))
            done[head] = True
            continue
        if op.kind == K.SWAP:
            qm.relabel(op.a, op.b)
            done[head] = True
            continue
        var head_local = (qm.phys_mask(need[head]) & ~local) == 0
        var sel = _select_local(plan, wires, need, done, head, qm, nb, allw)
        var thr = SWAP_GAIN if swaps > 0 else SWAP_GAIN_FIRST
        if sel.gain >= thr:
            _swap_in(sv, qm, sel.mask, plan, need, done, head, nb)
            swaps += 1
        elif not head_local:
            apply_kop(sv, _phys(op, qm))
            done[head] = True
            continue
        _collect_run(plan, wires, need, done, head, qm, nb, allw, run)
        if len(run) == 1:
            apply_kop(sv, run[0])
        elif len(run) > 1:
            _apply_run(sv, run, nb)
        run.clear()
    _restore_identity(sv, qm)


def evolve_into(mut sv: StateVector, circuit: Circuit, params: List[Float64]) raises:
    """Apply all unitary instructions (fused, cache-blocked). Non-unitaries are
    skipped (callers decide measurement semantics)."""
    var plan = build_plan(circuit, params)
    execute_plan(sv, plan)
    sv.apply_global_phase(circuit.global_phase)


comptime MAX_SV_QUBITS = 32


def check_statevector_size(n: Int) raises:
    """Refuse allocations that cannot fit instead of being OOM-killed."""
    if n > MAX_SV_QUBITS:
        raise Error("QM-E404: state vector for " + String(n) + " qubits needs 2^" + String(n) + " x 16 bytes; limit is " + String(MAX_SV_QUBITS) + " qubits. Hint: Clifford circuits run on the stabilizer simulator; otherwise reduce qubits or use hardware.")


def evolve(circuit: Circuit, params: List[Float64] = List[Float64]()) raises -> StateVector:
    circuit.validate()
    check_statevector_size(circuit.num_qubits)
    if len(params) != circuit.num_params():
        raise Error("QM-E101: circuit has " + String(circuit.num_params()) + " parameters " + circuit.param_names_str() + " but " + String(len(params)) + " values were given.")
    var sv = StateVector(circuit.num_qubits)
    evolve_into(sv, circuit, params)
    return sv^


def expectation(circuit: Circuit, observable: PauliSum, params: List[Float64] = List[Float64]()) raises -> Float64:
    var sv = evolve(circuit, params)
    return sv.expectation(observable)


def _apply_pauli(mut sv: StateVector, which: Int, q: Int):
    """which: 1=X, 2=Y, 3=Z."""
    if which == 1:
        sv.apply_x(q)
    elif which == 2:
        sv.apply_1q(matrix_1q(Op.Y), q)
    elif which == 3:
        sv.apply_diag_1q(1.0, 0.0, -1.0, 0.0, q)


def _apply_noise(mut sv: StateVector, g: Instruction, mut rng: Rng, params: List[Float64]):
    var k = g.kind
    var p = g.p0.resolve(params)
    if k == Op.X_ERROR:
        if rng.uniform() < p: _apply_pauli(sv, 1, g.q0)
    elif k == Op.Y_ERROR:
        if rng.uniform() < p: _apply_pauli(sv, 2, g.q0)
    elif k == Op.Z_ERROR:
        if rng.uniform() < p: _apply_pauli(sv, 3, g.q0)
    elif k == Op.DEPOLARIZE1:
        if rng.uniform() < p: _apply_pauli(sv, 1 + rng.below(3), g.q0)
    elif k == Op.PAULI_CHANNEL_1:
        var u = rng.uniform()
        var px = g.p0.resolve(params)
        var py = g.p1.resolve(params)
        var pz = g.p2.resolve(params)
        if u < px: _apply_pauli(sv, 1, g.q0)
        elif u < px + py: _apply_pauli(sv, 2, g.q0)
        elif u < px + py + pz: _apply_pauli(sv, 3, g.q0)
    elif k == Op.DEPOLARIZE2:
        if rng.uniform() < p:
            var r = 1 + rng.below(15)
            _apply_pauli(sv, r & 3, g.q0)
            _apply_pauli(sv, r >> 2, g.q1)


def sample(circuit: Circuit, shots: Int, seed: UInt64 = 1234,
           params: List[Float64] = List[Float64]()) raises -> SampleResult:
    """Sample measurement outcomes. Picks static sampling or trajectories."""
    circuit.validate()
    check_statevector_size(circuit.num_qubits)
    if len(params) != circuit.num_params():
        raise Error("QM-E101: circuit has " + String(circuit.num_params()) + " parameters " + circuit.param_names_str() + " but " + String(len(params)) + " values were given.")
    var rng = Rng(seed)
    if circuit.num_clbits > 62:
        raise Error("QM-E406: the sampler packs classical registers into 62 bits; circuit has " + String(circuit.num_clbits) + " clbits. Hint: measure fewer bits or split the register.")
    if circuit.num_measurements == 0:
        raise Error("QM-E402: circuit has no measurements. Hint: call measure_all() or use evolve()/expectation().")
    if not circuit.is_dynamic() and not circuit.has_noise():
        var res = SampleResult(circuit.num_clbits, shots, "statevector:static")
        var sv = StateVector(circuit.num_qubits)
        evolve_into(sv, circuit, params)
        # terminal measurement map qubit -> clbit (last write wins)
        var qs = List[Int]()
        var cs = List[Int]()
        for g in circuit.ops:
            if g.kind == Op.MEASURE:
                qs.append(g.q0)
                cs.append(g.clbit)
        var outs = sv.sample(shots, rng)
        for o in outs:
            var creg = 0
            for j in range(len(qs)):
                if (o >> qs[j]) & 1 == 1:
                    creg |= 1 << cs[j]
            res.record(creg)
        return res^
    # trajectories
    var res = SampleResult(circuit.num_clbits, shots, "statevector:trajectories")
    var sv = StateVector(circuit.num_qubits)
    for shot in range(shots):
        var srng = rng.split(shot)
        sv.reset_to_zero()
        var creg = 0
        var fz = Fuser(circuit.num_qubits)
        for g in circuit.ops:
            var k = g.kind
            if g.cond >= 0 and not circuit.condition_holds(g, creg):
                continue
            if op_is_unitary(k):
                var a = op_arity(k)
                if a == 1 and g.cond < 0:
                    if k != Op.I:
                        fz.push(gate_matrix_1q(g, params), g.q0)
                    continue
                for j in range(a):
                    fz.flush(sv, g.qubit(j))
                apply_instruction(sv, g, params)
                continue
            if k == Op.MEASURE:
                fz.flush(sv, g.q0)
                var b = sv.measure(g.q0, srng)
                if b == 1:
                    creg |= 1 << g.clbit
                else:
                    creg &= ~(1 << g.clbit)
            elif k == Op.RESET:
                fz.flush(sv, g.q0)
                sv.reset_qubit(g.q0, srng)
            elif op_is_noise(k):
                fz.flush(sv, g.q0)
                if op_arity(k) == 2:
                    fz.flush(sv, g.q1)
                _apply_noise(sv, g, srng, params)
        res.record(creg)
    return res^
