"""Circuit -> state-vector execution: gate fusion, sampling, trajectories.

Execution strategy is chosen automatically (and reported, for agents):
  * static circuits (terminal measurements only, no noise): evolve once, then
    sample all shots from the final distribution - O(gates * 2^n + shots).
  * dynamic circuits (mid-circuit measurement, reset, feed-forward) or noisy
    circuits: per-shot quantum trajectories with independent RNG streams.
Both report a clbit written by several MEASUREs as the LAST one in program
order (Qiskit/Aer semantics; static path: `terminal_measure_map`).
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
    StateVector, FPtr, LOGW, k1_range, kdiag_range, kx_range, kc1_range, kcx_range,
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


def execute_plan(mut sv: StateVector, plan: List[KOp]) raises:
    """Run a plan. Below BLOCK_MIN_QUBITS every op is one parallel pass (the
    state lives in cache anyway); above, the qubit-reordering cache-blocked
    executor runs (`execute_plan_reordered`). Either way the state is left in
    canonical little-endian order (qubit k = amplitude-index bit k)."""
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
#   * In-vector bits 0..2 (LANE_MASK) are expensive to move: every pair that
#     involves one costs an extra vectorized full-state SWAP pass in
#     `apply_swap_layer`, now and usually again in the final restore. Victims
#     are therefore taken among bits >= 3 first. When S fills the block (no
#     spare local bit >= 3 is left) a lane bit would have to go; the
#     selection is then redone with bits 0..2 pinned local and that one is
#     used if it loses at most as many ops as lane victims it avoids (and
#     still meets the gain threshold). This is a cost trade-off, not a
#     guarantee: lane pairs remain possible (ReorderStats.vector_pairs).
#   * At the end the permutation is undone with <= 2 swap layers (any
#     permutation is a product of two involutions), so the StateVector seen
#     by every caller is in canonical little-endian order (qubit k = bit k):
#     amplitudes, probabilities, sampling, expectation, measurement and reset
#     are unaffected. A layer is one in-place pass, plus one vectorized SWAP
#     pass per pair that involves an in-vector bit 0..2 (from SWAP-gate
#     relabels of qubits on bits 0..2, or from lane victims of swap-ins).
#   * Memory: no scratch buffers; the plan bookkeeping is O(len(plan)).
#   * `execute_plan_reordered_stats` reports what was done (`ReorderStats`).
#     See docs/SIMULATION.md for the design, numbers and limitations.
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
    var half = full >> 1
    var quarter = full >> 2  # 0 when nb < 2: no 2q op is then fully local
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
                  head: Int, qm: QubitMap, nb: Int, allw: Int, pinned: Int) -> _Selection:
    """Greedy lookahead: the set of <= nb physical bits that lets the longest
    dependency-respecting stream of upcoming ops run inside blocks. Bits of
    `pinned` (local bits that must stay local, e.g. LANE_MASK) count against
    the nb slots whether or not an op needs them."""
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
        if Int(pop_count(u | pinned)) <= nb:
            smask = u
            if (np & ~local) != 0:
                gain += 1
        else:
            blocked |= w
    return _Selection(smask, gain)


comptime LANE_MASK = (1 << LOGW) - 1  # physical in-vector bits 0..2


def _lane_victims(smask: Int, nb: Int) -> Int:
    """In-vector bits a swap-in to `smask` must evict: the incoming bits
    beyond the free local bits >= LOGW."""
    var local = (1 << nb) - 1
    var ins = Int(pop_count(smask & ~local))
    var free = Int(pop_count(local & ~smask & ~LANE_MASK))
    return max(0, ins - free)


def _swap_in(mut sv: StateVector, mut qm: QubitMap, smask: Int, plan: List[KOp],
             need: List[Int], done: List[Bool], head: Int, nb: Int,
             mut stats: ReorderStats) raises:
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
    # victims: bits >= LOGW first (an in-vector bit costs an extra vectorized
    # pass in apply_swap_layer), then farthest next use, ties -> higher bit
    # (longer contiguous runs in the swap pass)
    var outs = List[Int]()
    var taken = 0
    for _ in range(len(ins)):
        var best = -1
        for p in range(nb):
            if (cand >> p) & 1 == 0 or (taken >> p) & 1 == 1:
                continue
            if best < 0:
                best = p
                continue
            var lane_p = p < LOGW
            var lane_b = best < LOGW
            if lane_p != lane_b:
                if lane_b:
                    best = p
                continue
            if dist[p] > dist[best] or (dist[p] == dist[best] and p > best):
                best = p
        taken |= 1 << best
        outs.append(best)
        if best < LOGW:
            stats.vector_pairs += 1
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


struct ReorderStats(Copyable, Writable):
    """What the cache-blocked executor did (for tests, tuning and agents)."""

    var block_bits: Int  # nb: log2 of the cache block size in amplitudes
    var block_passes: Int  # cache-blocked passes (each = one sweep over the state)
    var block_ops: Int  # ops applied inside those passes
    var global_ops: Int  # ops applied as their own full-state pass
    var swap_layers: Int  # index-bit swap passes that moved qubits into blocks
    var swapped_bits: Int  # qubits moved in by those passes
    var relabels: Int  # SWAP gates absorbed into the qubit map (no data moved)
    var restore_layers: Int  # swap-layer passes of the final restore (0..2)
    var restore_pairs: Int  # bit pairs exchanged by the final restore
    var restore_min_bit: Int  # lowest physical bit moved by the restore (-1: none)
    var vector_pairs: Int  # swapped bit pairs with an in-vector bit 0..2 (swap-in + restore)

    def __init__(out self, nb: Int):
        self.block_bits = nb
        self.block_passes = 0
        self.block_ops = 0
        self.global_ops = 0
        self.swap_layers = 0
        self.swapped_bits = 0
        self.relabels = 0
        self.restore_layers = 0
        self.restore_pairs = 0
        self.restore_min_bit = -1
        self.vector_pairs = 0

    def write_to(self, mut w: Some[Writer]):
        w.write("ReorderStats(nb=", self.block_bits, ", block_passes=", self.block_passes,
                ", block_ops=", self.block_ops, ", global_ops=", self.global_ops,
                ", swap_layers=", self.swap_layers, ", swapped_bits=", self.swapped_bits,
                ", relabels=", self.relabels, ", restore_layers=", self.restore_layers,
                ", restore_pairs=", self.restore_pairs, ", restore_min_bit=", self.restore_min_bit,
                ", vector_pairs=", self.vector_pairs, ")")


def _restore_identity(mut sv: StateVector, mut qm: QubitMap, mut stats: ReorderStats) raises:
    """Undo the permutation with <= 2 swap-layer passes: the bit permutation
    (data at physical p must move to bit p2l[p]) is written as tau2 o tau1 of
    two involutions, cycle by cycle (c_i -> c_{-i}, then c_i -> c_{1-i}).
    A 2-cycle needs only tau2; the identity needs nothing."""
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
        if stats.restore_min_bit < 0:
            stats.restore_min_bit = p  # p ascends: first moved bit is the lowest
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
                if min(cyc[i], cyc[j]) < LOGW:
                    stats.vector_pairs += 1
            var j2 = (m + 1 - i) % m
            if i < j2:
                a2.append(cyc[i])
                b2.append(cyc[j2])
                if min(cyc[i], cyc[j2]) < LOGW:
                    stats.vector_pairs += 1
    if len(a1) > 0:
        sv.apply_swap_layer(a1, b1)
        for i in range(len(a1)):
            qm.swap_phys(a1[i], b1[i])
        stats.restore_layers += 1
        stats.restore_pairs += len(a1)
    if len(a2) > 0:
        sv.apply_swap_layer(a2, b2)
        for i in range(len(a2)):
            qm.swap_phys(a2[i], b2[i])
        stats.restore_layers += 1
        stats.restore_pairs += len(a2)


def execute_plan_reordered(mut sv: StateVector, plan: List[KOp], nb: Int) raises:
    """Cache-blocked execution with qubit reordering (see the design notes
    above). `nb` = log2 of the block size in amplitudes, 1 <= nb <= n
    (`execute_plan` uses min(BLOCK_BITS, n - MIN_BLOCKS_LOG2); other values
    are for tuning and tests; QM-E408 otherwise). The state is in canonical
    order on return."""
    _ = execute_plan_reordered_stats(sv, plan, nb)


def execute_plan_reordered_stats(mut sv: StateVector, plan: List[KOp], nb: Int) raises -> ReorderStats:
    """`execute_plan_reordered` that also reports what it did."""
    var n = sv.num_qubits
    if nb < 1 or nb > n:
        raise Error("QM-E408: cache block of 2^" + String(nb) + " amplitudes is invalid for a " + String(n) + "-qubit state (need 1 <= nb <= " + String(n) + "). Hint: use execute_plan(), which picks nb = min(" + String(BLOCK_BITS) + ", n - " + String(MIN_BLOCKS_LOG2) + "), or pass nb in [1, num_qubits].")
    var stats = ReorderStats(nb)
    var m = len(plan)
    var wires = List[Int](capacity=m)
    var need = List[Int](capacity=m)
    for op in plan:
        wires.append(_op_wires(op))
        need.append(_op_need(op))
        if op.kind == K.SWAP:
            stats.relabels += 1  # every SWAP gate becomes a relabel
    var done = List[Bool](length=m, fill=False)
    var qm = QubitMap(n)
    var local = (1 << nb) - 1
    var allw = (1 << n) - 1
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
            stats.global_ops += 1
            continue
        if op.kind == K.SWAP:
            qm.relabel(op.a, op.b)
            done[head] = True
            continue
        var head_local = (qm.phys_mask(need[head]) & ~local) == 0
        var sel = _select_local(plan, wires, need, done, head, qm, nb, allw, 0)
        var thr = SWAP_GAIN if stats.swap_layers > 0 else SWAP_GAIN_FIRST
        if nb >= 2 * LOGW:
            # a full S would evict in-vector bits (an extra full-state pass
            # each): retry with bits 0..2 pinned, keep it if it loses <= as
            # many ops as lane victims it avoids
            var lv = _lane_victims(sel.mask, nb)
            if lv > 0:
                var alt = _select_local(plan, wires, need, done, head, qm, nb, allw, LANE_MASK)
                if alt.gain + lv >= sel.gain and (alt.gain >= thr or sel.gain < thr):
                    sel = alt
        if sel.gain >= thr:
            stats.swapped_bits += Int(pop_count(sel.mask & ~local))
            _swap_in(sv, qm, sel.mask, plan, need, done, head, nb, stats)
            stats.swap_layers += 1
        elif not head_local:
            apply_kop(sv, _phys(op, qm))
            done[head] = True
            stats.global_ops += 1
            continue
        _collect_run(plan, wires, need, done, head, qm, nb, allw, run)
        if len(run) == 1:
            apply_kop(sv, run[0])
            stats.global_ops += 1
        elif len(run) > 1:
            _apply_run(sv, run, nb)
            stats.block_passes += 1
            stats.block_ops += len(run)
        run.clear()
    _restore_identity(sv, qm, stats)
    return stats^


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


struct TerminalMeasureMap(Copyable, Movable):
    """Packs one sampled basis index into a classical register, for circuits
    whose measurements are all terminal (static sampling, CPU and GPU).

    Semantics are last-write-wins, identical to the per-shot trajectory path,
    the stabilizer/Pauli-frame samplers and Qiskit-Aer: when several MEASUREs
    write the same clbit, only the LAST one in program order decides its
    value (an earlier 1 must not survive a later 0). The map therefore keeps
    exactly one source qubit per clbit, and packing is a plain OR of disjoint
    bits. Build it with `terminal_measure_map(circuit)`."""

    var qubits: List[Int]  # source amplitude-index bit, one entry per clbit
    var clbits: List[Int]  # destination clbit (ascending, each at most once)
    var mask: Int  # >= 0: every entry is qubit k -> clbit k, creg = outcome & mask

    def __init__(out self, var qubits: List[Int], var clbits: List[Int]):
        self.qubits = qubits^
        self.clbits = clbits^
        var m = 0
        for j in range(len(self.qubits)):
            if self.qubits[j] != self.clbits[j]:
                m = -1
                break
            m |= 1 << self.clbits[j]
        self.mask = m

    def pack(self, outcome: Int) -> Int:
        """Classical register (bit c = clbit c) for basis index `outcome`."""
        if self.mask >= 0:
            return outcome & self.mask
        var creg = 0
        for j in range(len(self.qubits)):
            creg |= ((outcome >> self.qubits[j]) & 1) << self.clbits[j]
        return creg


def terminal_measure_map(circuit: Circuit) -> TerminalMeasureMap:
    """clbit <- qubit map of a circuit's terminal measurements, LAST MEASURE
    per clbit wins. Only meaningful when `not circuit.is_dynamic()` (every
    MEASURE is terminal, so all of them read the same final state)."""
    var width = circuit.num_clbits
    for g in circuit.ops:
        if g.kind == Op.MEASURE and g.clbit >= width:
            width = g.clbit + 1
    var last_q = List[Int](length=max(width, 1), fill=-1)
    for g in circuit.ops:
        if g.kind == Op.MEASURE and g.clbit >= 0:
            last_q[g.clbit] = g.q0
    var qs = List[Int]()
    var cs = List[Int]()
    for c in range(width):
        if last_q[c] >= 0:
            qs.append(last_q[c])
            cs.append(c)
    return TerminalMeasureMap(qs^, cs^)


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
        # clbit <- qubit, last MEASURE into a clbit wins (as in the loop below)
        var mm = terminal_measure_map(circuit)
        var outs = sv.sample(shots, rng)
        for o in outs:
            res.record(mm.pack(o))
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
