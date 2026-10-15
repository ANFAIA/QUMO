"""`implement_ops`: logical program -> physical circuit (Guppy FT's last pass).

The `Engine` replays a `LogicalProgram` and links every logical op to its
physical gadget (`qumo.ft.gadgets`), while managing the resources that Guppy
FT keeps in its global runtime `STATE` struct:

* logical-qubit addresses: a LIFO stack of (block, slot) addresses (works for
  k > 1: Iceberg slots); allocation pops, measurement/free pushes back;
  QM-E725 when the architecture runs out of blocks (Guppy FT:
  "allocate_next_addr: No more logical qubits to allocate");
* physical blocks: blocks are re-labelled instead of moved (Guppy FT's
  `mem_swap`): QEC-by-teleportation, magic-state injection and logical SWAP
  only exchange which physical block a handle points to;
* state factories (|0>_L and T|+>_L): `size` states are kept prepared ahead of
  use and refilled after each consumption (bounded by the remaining demand
  counted from the program), each with up to `max_attempts` flag-verified
  repeat-until-success attempts unrolled under classical conditions;
* QEC-cycle insertion from per-block cost counters (QECPolicy);
* a compile-time Pauli frame per logical qubit whose entries are classical
  expressions (`qumo.ft.classical.CExpr`) of measured bits: Pauli corrections
  (teleportation, EC, injection, user `if m: x(q)`) cost no gates; a frame is
  flushed into classically-conditioned physical Paulis only before an
  operation that does not commute with it (rotations, conditioned non-Pauli
  gates). A classically-conditioned S keeps the frame linear when its
  condition equals (or is the complement of) the X frame, which is exactly the
  T-injection correction, so T gates never force a flush.
* decoding: a transversal Z measurement of a block is decoded with the code's
  Z-type checks (lookup table up to (d-1)//2 X errors; error-detecting codes
  turn a non-zero syndrome into a post-selection flag). The decoded value is a
  CExpr: raw parity XOR syndrome minterms, materialised by the classical lane
  only when a quantum operation depends on it.

Static vs runtime: Guppy FT runs the allocator, factories and QEC counters in
the device's classical runtime because Guppy programs have loops and runtime
control flow. QUMO circuits are static, so every decision that Guppy takes at
run time from *program structure* (addresses, counters, factory refills) is
taken here at encode time with identical results; every decision that depends
on *measurement outcomes* (decoded corrections, RUS retries, conditioned
gates) stays a run-time classical condition in the circuit.

Idea from Guppy FT (Apache-2.0), files src/guppyft/code/steane/encode.py
(STATE, allocate_next_addr, qec_policy, op replacements),
src/guppyft/std/state_factory.py and src/guppyft/encode/_implement_ops.py;
re-derived for a static IR.

Error codes: QM-E724 (op unsupported by the architecture), QM-E725 (out of
logical blocks), QM-E726 (unsupported classically-conditioned logical op),
QM-E727 (malformed logical program).
"""

from std.math import pi

from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_arity
from qumo.ft.code import StabilizerCode
from qumo.ft.pauli import SymPauli
from qumo.ft.classical import CTerm, CExpr
from qumo.ft.logical import LogicalProgram, LogicalInstr, LOp, lop_name, lop_arity, lop_is_pauli
from qumo.ft.arch import Architecture
from qumo.ft.gadgets import (
    steane_encoder_zero, steane_prep_zero_ft, steane_prep_t_nonft, steane_prep_t_ft,
    transversal_1q, transversal_2q, steane_physical_kind, iceberg_prep_zero, iceberg_pauli,
    iceberg_rotation, iceberg_syndrome_cycle,
)
from qumo.ai.json import JsonWriter


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
struct EncodeStats(Copyable, Writable):
    var logical_ops: Int
    var physical_qubits: Int
    var lane_qubits: Int
    var blocks_prepared: Int
    var zero_states: Int
    var magic_states: Int
    var injections: Int
    var qec_cycles: Int
    var decoded_measurements: Int
    var lane_runs: Int
    var frame_corrections: Int  # Pauli corrections absorbed by the frame
    var frame_flushes: Int  # conditioned physical Paulis emitted from the frame
    var rus_attempts: Int  # extra unrolled RUS attempts emitted
    var peak_logical: Int

    def __init__(out self):
        self.logical_ops = 0
        self.physical_qubits = 0
        self.lane_qubits = 0
        self.blocks_prepared = 0
        self.zero_states = 0
        self.magic_states = 0
        self.injections = 0
        self.qec_cycles = 0
        self.decoded_measurements = 0
        self.lane_runs = 0
        self.frame_corrections = 0
        self.frame_flushes = 0
        self.rus_attempts = 0
        self.peak_logical = 0

    def write_json(self, mut w: JsonWriter):
        w.begin_object()
        w.kv_int("logical_ops", self.logical_ops)
        w.kv_int("physical_qubits", self.physical_qubits)
        w.kv_int("lane_qubits", self.lane_qubits)
        w.kv_int("blocks_prepared", self.blocks_prepared)
        w.kv_int("zero_states", self.zero_states)
        w.kv_int("magic_states", self.magic_states)
        w.kv_int("injections", self.injections)
        w.kv_int("qec_cycles", self.qec_cycles)
        w.kv_int("decoded_measurements", self.decoded_measurements)
        w.kv_int("lane_runs", self.lane_runs)
        w.kv_int("frame_corrections", self.frame_corrections)
        w.kv_int("frame_flushes", self.frame_flushes)
        w.kv_int("rus_extra_attempts", self.rus_attempts)
        w.kv_int("peak_live_logical_qubits", self.peak_logical)
        w.end_object()

    def write_to(self, mut w: Some[Writer]):
        w.write("qubits=", self.physical_qubits, " blocks=", self.blocks_prepared, " zero=", self.zero_states,
                " magic=", self.magic_states, " qec=", self.qec_cycles, " lane_runs=", self.lane_runs,
                " frame_corr=", self.frame_corrections, " flushes=", self.frame_flushes)


struct EncodedProgram(Copyable, Writable):
    """A physical circuit plus how to read the logical results from it."""

    var arch_name: String
    var code_name: String
    var circuit: Circuit
    var outputs: List[Int]  # logical clbit c -> physical clbit holding it (-1 = never written)
    var postselect: List[Int]  # physical clbits that must all read 0 (detections, factory failures)
    var lane_qubits: List[Int]  # classical-lane qubits (noise-free by construction)
    var stats: EncodeStats
    var logical: LogicalProgram

    def __init__(out self, arch_name: String, code_name: String, var circuit: Circuit, var logical: LogicalProgram):
        self.arch_name = arch_name
        self.code_name = code_name
        self.circuit = circuit^
        self.outputs = List[Int]()
        self.postselect = List[Int]()
        self.lane_qubits = List[Int]()
        self.stats = EncodeStats()
        self.logical = logical^

    def num_outputs(self) -> Int:
        return len(self.outputs)

    def write_json(self, mut w: JsonWriter):
        w.begin_object()
        w.kv_string("architecture", self.arch_name)
        w.kv_string("code", self.code_name)
        w.kv_int("physical_qubits", self.circuit.num_qubits)
        w.kv_int("physical_clbits", self.circuit.num_clbits)
        w.kv_int("physical_ops", len(self.circuit.ops))
        w.kv_int_list("outputs", self.outputs)
        w.kv_int_list("postselect_clbits", self.postselect)
        w.kv_int_list("lane_qubits", self.lane_qubits)
        w.key("stats")
        self.stats.write_json(w)
        w.key("logical")
        self.logical.summary_json(w)
        w.end_object()

    def to_json(self) -> String:
        var w = JsonWriter()
        self.write_json(w)
        return w.finish()

    def write_to(self, mut w: Some[Writer]):
        w.write("EncodedProgram(", self.arch_name, ", qubits=", self.circuit.num_qubits, ", ops=", len(self.circuit.ops),
                ", outputs=", len(self.outputs), ", postselect=", len(self.postselect), ", ", self.stats, ")")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
@fieldwise_init
struct Cond(Copyable):
    """kind 0: unconditional; 1: clbit == val; 2: AND of (bits[i] == vals[i]); 3: never."""

    var kind: Int
    var bits: List[Int]
    var vals: List[Int]

    @staticmethod
    def none() -> Cond:
        return Cond(0, List[Int](), List[Int]())

    @staticmethod
    def never() -> Cond:
        return Cond(3, List[Int](), List[Int]())

    @staticmethod
    def bit(b: Int, v: Int) -> Cond:
        var bs = List[Int]()
        bs.append(b)
        var vs = List[Int]()
        vs.append(v)
        return Cond(1, bs^, vs^)


struct Block(Copyable):
    var qubits: List[Int]
    var slots: List[Int]  # handle per slot, -1 = free
    var residual: List[CExpr]  # X frame a freed slot leaves behind (Iceberg)
    var live: Bool
    var counter: Float64
    var logical_block: Int  # Iceberg address block, -1 otherwise

    def __init__(out self, var qubits: List[Int], k: Int):
        self.qubits = qubits^
        self.slots = List[Int](length=k, fill=-1)
        self.residual = List[CExpr]()
        for _ in range(k):
            self.residual.append(CExpr())
        self.live = True
        self.counter = 0.0
        self.logical_block = -1


struct Engine(Movable):
    var arch: Architecture
    var code: StabilizerCode
    var n: Int
    var k: Int
    var circ: Circuit
    var nq: Int
    var dirty: List[Int]
    var lane: Int
    var lanes: List[Int]
    var nclb: Int
    var zero_bit: Int
    var blocks: List[Block]
    # handles
    var hblock: List[Int]
    var hslot: List[Int]
    var haddr: List[Bool]  # counts against the n_blocks capacity
    var hlive: List[Bool]
    var fx: List[CExpr]
    var fz: List[CExpr]
    var live_addressed: Int
    # Iceberg addresses
    var addr_stack: List[Int]  # b * k + s
    var ice_pblock: List[Int]  # logical block -> physical block (-1)
    # classical results
    var out_expr: List[CExpr]
    var out_written: List[Bool]
    var post: List[CExpr]
    # factories
    var zero_pool: List[Int]
    var magic_pool: List[Int]
    var remaining_zero: Int
    var remaining_magic: Int
    # decoding table (CSS, Z basis): flips per logical slot per syndrome pattern
    var zchecks: List[List[Int]]
    var dec_flip: List[List[Int]]  # [slot][sigma] in {0, 1}; -1 = uncorrectable (flag)
    var stats: EncodeStats

    def __init__(out self, arch: Architecture) raises:
        self.arch = arch.copy()
        self.code = arch.code.copy()
        self.n = arch.code.n
        self.k = arch.code.k
        self.circ = Circuit(1, 0, "encoded")
        self.nq = 0
        self.dirty = List[Int]()
        self.lane = -1
        self.lanes = List[Int]()
        self.nclb = 0
        self.zero_bit = -1
        self.blocks = List[Block]()
        self.hblock = List[Int]()
        self.hslot = List[Int]()
        self.haddr = List[Bool]()
        self.hlive = List[Bool]()
        self.fx = List[CExpr]()
        self.fz = List[CExpr]()
        self.live_addressed = 0
        self.addr_stack = List[Int]()
        self.ice_pblock = List[Int](length=arch.n_blocks, fill=-1)
        var total = arch.n_blocks * arch.code.k
        # LIFO: pop gives (0, 0), (0, 1), ... first
        var a = total - 1
        while a >= 0:
            self.addr_stack.append(a)
            a -= 1
        self.out_expr = List[CExpr]()
        self.out_written = List[Bool]()
        self.post = List[CExpr]()
        self.zero_pool = List[Int]()
        self.magic_pool = List[Int]()
        self.remaining_zero = 0
        self.remaining_magic = 0
        self.zchecks = List[List[Int]]()
        self.dec_flip = List[List[Int]]()
        self.stats = EncodeStats()
        self._build_decoder()

    # ------------------------------------------------------------ decoder
    def _build_decoder(mut self) raises:
        for gi in self.code.z_type_generators():
            self.zchecks.append(self.code.generators[gi].support())
        var m = len(self.zchecks)
        var t = (self.code.d - 1) // 2
        var npat = 1 << m
        for j in range(self.k):
            var zl = self.code.z_logicals[j].copy()
            var flips = List[Int](length=npat, fill=-1)
            flips[0] = 0
            if not zl.is_z_type():
                self.dec_flip.append(flips^)
                continue
            # brute force X errors of weight <= t, lowest weight first
            for w in range(1, t + 1):
                var idx = List[Int]()
                for i in range(w):
                    idx.append(i)
                while True:
                    var sig = 0
                    for ci in range(m):
                        var par = 0
                        for q in idx:
                            for s in self.zchecks[ci]:
                                if s == q:
                                    par ^= 1
                        sig |= par << ci
                    if flips[sig] < 0:
                        var f = 0
                        for q in idx:
                            f ^= zl.z(q)
                        flips[sig] = f
                    var i = w - 1
                    while i >= 0 and idx[i] == self.n - w + i:
                        i -= 1
                    if i < 0:
                        break
                    idx[i] += 1
                    for jj in range(i + 1, w):
                        idx[jj] = idx[jj - 1] + 1
            self.dec_flip.append(flips^)

    # ------------------------------------------------------------ resources
    def new_clbit(mut self) -> Int:
        var c = self.nclb
        self.nclb += 1
        return c

    def phys(mut self) -> Int:
        if len(self.dirty) > 0:
            var q = self.dirty.pop()
            self.circ.reset(q)
            return q
        var q = self.nq
        self.nq += 1
        return q

    def phys_fresh_list(mut self, count: Int) -> List[Int]:
        var out = List[Int]()
        for _ in range(count):
            out.append(self.phys())
        return out^

    def release_qubits(mut self, qs: List[Int]):
        for q in qs:
            self.dirty.append(q)

    def _sync_size(mut self):
        if self.nq > self.circ.num_qubits:
            self.circ.num_qubits = self.nq

    def new_handle(mut self) -> Int:
        var h = len(self.hblock)
        self.hblock.append(-1)
        self.hslot.append(-1)
        self.haddr.append(False)
        self.hlive.append(False)
        self.fx.append(CExpr())
        self.fz.append(CExpr())
        return h

    def ensure_handles(mut self, count: Int):
        while len(self.hblock) < count:
            _ = self.new_handle()

    # ------------------------------------------------------------ emission
    def apply_cond(mut self, c: Cond) raises:
        if c.kind == 1:
            self.circ.c_if(c.bits[0], c.vals[0])
        elif c.kind == 2:
            self.circ.c_if_register(c.bits, c.vals)

    def emit(mut self, kind: Int, q0: Int, q1: Int, angle: Float64, c: Cond) raises:
        if c.kind == 3:
            return
        self._sync_size()
        if kind == Op.MEASURE or kind == Op.RESET:
            raise Error("QM-E727: internal: emit() is for gates.")
        self.circ.add(kind, q0, q1, p0=angle)
        self.apply_cond(c)

    def emit_template(mut self, t: Circuit, qmap: List[Int], c: Cond, mut flags: List[Int]) raises:
        """Append template `t` (local qubits -> qmap), every op conditioned on c;
        local clbit j -> fresh physical clbit, appended to `flags`."""
        if c.kind == 3:
            return
        self._sync_size()
        var clmap = List[Int]()
        for _ in range(t.num_clbits):
            clmap.append(self.new_clbit())
        for b in clmap:
            flags.append(b)
        for g in t.ops:
            var a = op_arity(g.kind)
            if g.kind == Op.MEASURE:
                self.circ.measure(qmap[g.q0], clmap[g.clbit])
            elif g.kind == Op.RESET:
                self.circ.reset(qmap[g.q0])
            elif g.kind == Op.BARRIER or g.kind == Op.TICK:
                continue
            else:
                var q1 = qmap[g.q1] if a >= 2 else -1
                var q2 = qmap[g.q2] if a >= 3 else -1
                self.circ.add(g.kind, qmap[g.q0], q1, q2, g.p0, g.p1, g.p2)
            self.apply_cond(c)
        if self.circ.num_clbits < self.nclb:
            self.circ.num_clbits = self.nclb

    def get_lane(mut self) -> Int:
        if self.lane < 0:
            self.lane = self.nq
            self.nq += 1
            self.lanes.append(self.lane)
            self._sync_size()
        else:
            self.circ.reset(self.lane)
        return self.lane

    def const_zero_bit(mut self) -> Int:
        if self.zero_bit < 0:
            self.zero_bit = self.new_clbit()
            if self.circ.num_clbits < self.nclb:
                self.circ.num_clbits = self.nclb
        return self.zero_bit

    def materialize(mut self, e: CExpr) raises -> Int:
        """Physical clbit holding the value of e (classical lane when needed)."""
        if len(e.terms) == 1 and e.terms[0].kind == 0:
            return e.terms[0].lits[0]
        if e.is_zero():
            return self.const_zero_bit()
        var ln = self.get_lane()
        for t in e.terms:
            self.circ.add(Op.X, ln)
            if t.kind == 0:
                self.circ.c_if(t.lits[0], 1)
            elif t.kind == 1:
                self.circ.c_if_register(t.bits(), t.vals())
        var c = self.new_clbit()
        self.circ.measure(ln, c)
        self.stats.lane_runs += 1
        return c

    def cond_of(mut self, e: CExpr) raises -> Cond:
        if e.is_zero():
            return Cond.never()
        if len(e.terms) == 1:
            var t = e.terms[0].copy()
            if t.kind == 2:
                return Cond.none()
            if t.kind == 0:
                return Cond.bit(t.lits[0], 1)
            return Cond(2, t.bits(), t.vals())
        return Cond.bit(self.materialize(e), 1)

    # ------------------------------------------------------------ blocks
    def block_qubits(self, h: Int) -> List[Int]:
        return self.blocks[self.hblock[h]].qubits.copy()

    def _new_block(mut self, var qubits: List[Int]) -> Int:
        var b = Block(qubits^, self.k)
        self.blocks.append(b^)
        self.stats.blocks_prepared += 1
        return len(self.blocks) - 1

    def _prepare_rus(mut self, template: Circuit, qubits: List[Int], n_anc: Int, max_attempts: Int) raises:
        """Emit `template` with up to max_attempts unrolled verified attempts.
        Data qubits `qubits` must be in |0>; ancillas are taken from the pool."""
        var anc = self.phys_fresh_list(n_anc)
        var qmap = qubits.copy()
        for a in anc:
            qmap.append(a)
        var retry = CExpr()
        for att in range(max_attempts):
            var c = Cond.none()
            if att > 0:
                c = self.cond_of(retry)
                self.stats.rus_attempts += 1
                for q in qmap:
                    self.circ.reset(q)
                    self.apply_cond(c)
            var flags = List[Int]()
            self.emit_template(template, qmap, c, flags)
            if len(flags) == 0:
                retry = CExpr()
                break
            var zeros = List[Int](length=len(flags), fill=0)
            var ok = CTerm.minterm(flags, zeros)
            if att == 0:
                retry = CExpr.one()
                retry.xor_term(ok)
            else:
                # retry := retry AND NOT ok = retry XOR (retry AND ok)
                var r = self.materialize(retry)
                var bits = flags.copy()
                bits.append(r)
                var vals = zeros.copy()
                vals.append(1)
                retry = CExpr.bit(r)
                retry.xor_term(CTerm.minterm(bits, vals))
        if not retry.is_zero():
            self.post.append(retry^)
        self.release_qubits(anc)

    def _prepare_zero_block(mut self) raises -> Int:
        var qs = self.phys_fresh_list(self.n)
        var b = self._new_block(qs.copy())
        if self.arch.kind == "steane":
            if self.arch.zero_prep == "ft":
                self._prepare_rus(steane_prep_zero_ft(), qs, 1, self.arch.zero_factory.max_attempts)
            else:
                self._prepare_rus(steane_encoder_zero(), qs, 0, 1)
        else:
            self._prepare_rus(iceberg_prep_zero(self.n), qs, 0, 1)
        self.stats.zero_states += 1
        return b

    def _prepare_magic_block(mut self) raises -> Int:
        var qs = self.phys_fresh_list(self.n)
        var b = self._new_block(qs.copy())
        if self.arch.magic_prep == "ft":
            self._prepare_rus(steane_prep_t_ft(), qs, 3, self.arch.magic_factory.max_attempts)
        else:
            self._prepare_rus(steane_prep_t_nonft(), qs, 0, 1)
        self.stats.magic_states += 1
        return b

    def _refill(mut self) raises:
        var zs = self.arch.zero_factory.size
        while len(self.zero_pool) < zs and len(self.zero_pool) < self.remaining_zero:
            self.zero_pool.append(self._prepare_zero_block())
        var ms = self.arch.magic_factory.size
        while len(self.magic_pool) < ms and len(self.magic_pool) < self.remaining_magic:
            self.magic_pool.append(self._prepare_magic_block())

    def take_zero_block(mut self, demand: Bool = True) raises -> Int:
        var b: Int
        if len(self.zero_pool) > 0:
            b = self.zero_pool.pop(0)
        else:
            b = self._prepare_zero_block()
        if demand and self.remaining_zero > 0:
            self.remaining_zero -= 1
        self._refill()
        return b

    def take_magic_block(mut self) raises -> Int:
        var b: Int
        if len(self.magic_pool) > 0:
            b = self.magic_pool.pop(0)
        else:
            b = self._prepare_magic_block()
        if self.remaining_magic > 0:
            self.remaining_magic -= 1
        self._refill()
        return b

    def release_block(mut self, b: Int):
        self.blocks[b].live = False
        self.release_qubits(self.blocks[b].qubits.copy())
        if self.blocks[b].logical_block >= 0:
            self.ice_pblock[self.blocks[b].logical_block] = -1

    # ------------------------------------------------------------ allocation
    def _check_capacity(mut self) raises:
        if self.live_addressed >= self.arch.n_blocks * self.k:
            raise Error("QM-E725: no more logical qubits to allocate: the architecture has n_blocks=" + String(self.arch.n_blocks) + " (" + String(self.arch.n_blocks * self.k) + " logical qubits) and all are live. Hint: build the architecture with a larger n_blocks (Guppy FT: SteaneBuilder().build(n_blocks=...)).")

    def alloc(mut self, h: Int, magic: Bool, addressed: Bool) raises:
        self.ensure_handles(h + 1)
        if addressed:
            self._check_capacity()
            self.live_addressed += 1
            if self.live_addressed > self.stats.peak_logical:
                self.stats.peak_logical = self.live_addressed
        self.haddr[h] = addressed
        self.hlive[h] = True
        self.fx[h] = CExpr()
        self.fz[h] = CExpr()
        if self.arch.kind == "steane":
            var b = self.take_magic_block() if magic else self.take_zero_block(addressed)
            self.hblock[h] = b
            self.hslot[h] = 0
            self.blocks[b].slots[0] = h
            return
        # Iceberg: (block, slot) address from the LIFO stack
        if magic:
            raise Error("QM-E724: the Iceberg architecture has no magic-state factory (rotations are native). Hint: use rz(pi/4) / t on Iceberg qubits directly.")
        var addr = self.addr_stack.pop()
        var lb = addr // self.k
        var s = addr % self.k
        if self.ice_pblock[lb] < 0:
            var pb = self._prepare_zero_block()
            self.blocks[pb].logical_block = lb
            self.ice_pblock[lb] = pb
        var pb = self.ice_pblock[lb]
        self.hblock[h] = pb
        self.hslot[h] = s
        self.blocks[pb].slots[s] = h
        self.fx[h] = self.blocks[pb].residual[s].copy()
        self.blocks[pb].residual[s] = CExpr()

    def _retire(mut self, h: Int):
        if self.haddr[h]:
            self.live_addressed -= 1
        self.hlive[h] = False
        var b = self.hblock[h]
        var s = self.hslot[h]
        if b >= 0 and self.blocks[b].slots[s] == h:
            self.blocks[b].slots[s] = -1
        if self.arch.kind == "iceberg" and b >= 0 and self.blocks[b].logical_block >= 0:
            self.addr_stack.append(self.blocks[b].logical_block * self.k + s)
        self.fx[h] = CExpr()
        self.fz[h] = CExpr()

    def _other_live_slots(self, h: Int) -> Int:
        var b = self.hblock[h]
        var c = 0
        for s in range(self.k):
            var o = self.blocks[b].slots[s]
            if o >= 0 and o != h:
                c += 1
        return c

    # ------------------------------------------------------------ frames
    def _flush_x(mut self, h: Int) raises:
        if self.fx[h].is_zero():
            return
        var c = self.cond_of(self.fx[h].copy())
        self._phys_pauli(h, "X", c)
        self.fx[h] = CExpr()
        self.stats.frame_flushes += 1

    def _flush_z(mut self, h: Int) raises:
        if self.fz[h].is_zero():
            return
        var c = self.cond_of(self.fz[h].copy())
        self._phys_pauli(h, "Z", c)
        self.fz[h] = CExpr()
        self.stats.frame_flushes += 1

    def flush(mut self, h: Int) raises:
        self._flush_x(h)
        self._flush_z(h)

    def _phys_pauli(mut self, h: Int, p: String, c: Cond) raises:
        var qs = self.block_qubits(h)
        var dummy = List[Int]()
        if self.arch.kind == "steane":
            var kind = Op.X if p == "X" else (Op.Z if p == "Z" else Op.Y)
            self.emit_template(transversal_1q(kind, 7), qs, c, dummy)
        else:
            self.emit_template(iceberg_pauli(self.n, self.hslot[h], p), qs, c, dummy)

    def apply_pauli(mut self, h: Int, p: String, e: CExpr) raises:
        """Apply Pauli p to handle h when e evaluates to 1."""
        if e.is_zero():
            return
        if len(e.terms) == 1 and e.terms[0].kind == 2:
            self._phys_pauli(h, p, Cond.none())
            return
        if self.arch.feed_forward == "eager":
            self._phys_pauli(h, p, self.cond_of(e))
            return
        if p == "X" or p == "Y":
            self.fx[h].xor(e)
        if p == "Z" or p == "Y":
            self.fz[h].xor(e)
        self.stats.frame_corrections += 1

    # ------------------------------------------------------------ gates
    def _steane_gate(mut self, kind: Int, h0: Int, h1: Int, c: Cond) raises:
        var dummy = List[Int]()
        var pk = steane_physical_kind(lop_name(kind))
        if lop_arity(kind) == 1:
            self.emit_template(transversal_1q(pk, 7), self.block_qubits(h0), c, dummy)
        else:
            var qs = self.block_qubits(h0)
            for q in self.block_qubits(h1):
                qs.append(q)
            self.emit_template(transversal_2q(pk, 7), qs, c, dummy)

    def _propagate(mut self, kind: Int, h0: Int, h1: Int):
        if kind == LOp.H:
            var t = self.fx[h0].copy()
            self.fx[h0] = self.fz[h0].copy()
            self.fz[h0] = t^
        elif kind == LOp.S or kind == LOp.SDG:
            var x0 = self.fx[h0].copy()
            self.fz[h0].xor(x0)
        elif kind == LOp.CX:
            var x0 = self.fx[h0].copy()
            var z1 = self.fz[h1].copy()
            self.fx[h1].xor(x0)
            self.fz[h0].xor(z1)
        elif kind == LOp.CZ:
            var a = self.fx[h0].copy()
            var b = self.fx[h1].copy()
            self.fz[h0].xor(b)
            self.fz[h1].xor(a)

    def swap_handles(mut self, a: Int, b: Int):
        """Exchange the physical locations (and frames) of two handles: free."""
        var ba = self.hblock[a]
        var sa = self.hslot[a]
        var bb = self.hblock[b]
        var sb = self.hslot[b]
        self.hblock[a] = bb
        self.hslot[a] = sb
        self.hblock[b] = ba
        self.hslot[b] = sa
        if ba >= 0:
            self.blocks[ba].slots[sa] = b
        if bb >= 0:
            self.blocks[bb].slots[sb] = a
        var tx = self.fx[a].copy()
        self.fx[a] = self.fx[b].copy()
        self.fx[b] = tx^
        var tz = self.fz[a].copy()
        self.fz[a] = self.fz[b].copy()
        self.fz[b] = tz^

    def gate(mut self, kind: Int, h0: Int, h1: Int, angle: Float64, e: CExpr) raises:
        """Apply a logical gate when the classical expression e is 1 (CExpr.one() = always)."""
        if e.is_zero():
            return
        var unconditional = len(e.terms) == 1 and e.terms[0].kind == 2
        if lop_is_pauli(kind):
            var p = String("X") if kind == LOp.X else (String("Y") if kind == LOp.Y else String("Z"))
            self.apply_pauli(h0, p, e)
            return
        if kind == LOp.SWAP:
            if unconditional:
                self.swap_handles(h0, h1)
                return
            if self.arch.kind == "iceberg" and self.hblock[h0] != self.hblock[h1]:
                raise Error("QM-E726: a classically-conditioned SWAP between different Iceberg blocks is not supported. Hint: swap unconditionally (free relabel) or keep both qubits in one block.")
            self.flush(h0)
            self.flush(h1)
            var c = self.cond_of(e)
            if self.arch.kind == "steane":
                var qa = self.block_qubits(h0)
                var qb = self.block_qubits(h1)
                for i in range(self.n):
                    self.emit(Op.SWAP, qa[i], qb[i], 0.0, c)
            else:
                var qs = self.block_qubits(h0)
                self.emit(Op.SWAP, qs[2 + self.hslot[h0]], qs[2 + self.hslot[h1]], 0.0, c)
            return
        if self.arch.kind == "iceberg":
            self._iceberg_gate(kind, h0, h1, angle, e)
            return
        # ---- Steane
        if kind == LOp.T or kind == LOp.TDG:
            if not unconditional:
                raise Error("QM-E726: a classically-conditioned T gate needs a conditioned magic-state injection, which the Steane architecture does not provide. Hint: restructure the program so that the T gate is unconditional (e.g. conditioned SWAP into a scratch qubit), or condition Clifford gates only.")
            if self.arch.qec.before_non_clifford and self.arch.qec.style != "none":
                self.qec_cycle(h0)
            var m = self.new_handle()
            self.alloc(m, True, True)
            self.inject(h0, m, kind == LOp.TDG)
            return
        if kind == LOp.RZ or kind == LOp.RX or kind == LOp.RZZ or kind == LOp.RXX:
            raise Error("QM-E724: logical '" + lop_name(kind) + "' with an arbitrary angle is not in the Steane gate set. Hint: run compile() (static Clifford+T synthesis or the comparator circuit) first.")
        if kind == LOp.H or kind == LOp.S or kind == LOp.SDG or kind == LOp.CX or kind == LOp.CZ:
            if unconditional:
                self._steane_gate(kind, h0, h1, Cond.none())
                self._propagate(kind, h0, h1)
                return
            if (kind == LOp.S or kind == LOp.SDG) and self.arch.feed_forward == "frame":
                var fxe = self.fx[h0].copy()
                if fxe.is_zero():
                    pass
                elif _same_expr(fxe, e):
                    self.fz[h0].xor(fxe)
                elif _same_expr(fxe.xored(CExpr.one()), e):
                    pass
                else:
                    self._flush_x(h0)
                self._steane_gate(kind, h0, h1, self.cond_of(e))
                return
            self.flush(h0)
            if lop_arity(kind) == 2:
                self.flush(h1)
            self._steane_gate(kind, h0, h1, self.cond_of(e))
            return
        raise Error("QM-E724: logical op '" + lop_name(kind) + "' is not supported by the Steane architecture.")

    def inject(mut self, h: Int, m: Int, dagger: Bool) raises:
        """T (T^dag) on h by consuming the magic qubit m (Guppy FT inject_t):
        CX(m -> h); relabel; measure the old data block; X^r then S^r
        (T^dag: X^r then S^dag^(1-r))."""
        var one = CExpr.one()
        self.gate(LOp.CX, m, h, 0.0, one)
        self.swap_handles(m, h)
        var r = self.measure(m)
        self.apply_pauli(h, "X", r)
        if dagger:
            self.gate(LOp.SDG, h, -1, 0.0, r.xored(CExpr.one()))
        else:
            self.gate(LOp.S, h, -1, 0.0, r)
        self.stats.injections += 1

    def _iceberg_rot(mut self, kind: String, h0: Int, h1: Int, angle: Float64, e: CExpr) raises:
        var c = self.cond_of(e)
        var dummy = List[Int]()
        if kind == "rz" or kind == "rx":
            self.emit_template(iceberg_rotation(self.n, kind, self.hslot[h0], -1, angle), self.block_qubits(h0), c, dummy)
            return
        if self.hblock[h0] == self.hblock[h1]:
            self.emit_template(iceberg_rotation(self.n, kind, self.hslot[h0], self.hslot[h1], angle), self.block_qubits(h0), c, dummy)
            return
        # two blocks: 4-body Pauli rotation by a CX parity ladder
        var qa = self.block_qubits(h0)
        var qb = self.block_qubits(h1)
        var anchor = 1 if kind == "rzz" else 0  # Z_L uses b, X_L uses t
        var qs = List[Int]()
        qs.append(qa[anchor])
        qs.append(qa[2 + self.hslot[h0]])
        qs.append(qb[anchor])
        qs.append(qb[2 + self.hslot[h1]])
        if kind == "rxx":
            for q in qs:
                self.emit(Op.H, q, -1, 0.0, c)
        for i in range(3):
            self.emit(Op.CX, qs[i], qs[3], 0.0, c)
        self.emit(Op.RZ, qs[3], -1, angle, c)
        var i = 2
        while i >= 0:
            self.emit(Op.CX, qs[i], qs[3], 0.0, c)
            i -= 1
        if kind == "rxx":
            for q in qs:
                self.emit(Op.H, q, -1, 0.0, c)

    def _iceberg_gate(mut self, kind: Int, h0: Int, h1: Int, angle: Float64, e: CExpr) raises:
        var hp = pi / 2
        if kind == LOp.RZ:
            self._flush_x(h0)
            self._iceberg_rot("rz", h0, -1, angle, e)
        elif kind == LOp.RX:
            self._flush_z(h0)
            self._iceberg_rot("rx", h0, -1, angle, e)
        elif kind == LOp.RZZ:
            self._flush_x(h0)
            self._flush_x(h1)
            self._iceberg_rot("rzz", h0, h1, angle, e)
        elif kind == LOp.RXX:
            self._flush_z(h0)
            self._flush_z(h1)
            self._iceberg_rot("rxx", h0, h1, angle, e)
        elif kind == LOp.S:
            self.gate(LOp.RZ, h0, -1, hp, e)
        elif kind == LOp.SDG:
            self.gate(LOp.RZ, h0, -1, -hp, e)
        elif kind == LOp.T:
            self.gate(LOp.RZ, h0, -1, pi / 4, e)
        elif kind == LOp.TDG:
            self.gate(LOp.RZ, h0, -1, -pi / 4, e)
        elif kind == LOp.H:
            self.gate(LOp.RZ, h0, -1, hp, e)
            self.gate(LOp.RX, h0, -1, hp, e)
            self.gate(LOp.RZ, h0, -1, hp, e)
        elif kind == LOp.CZ:
            self.gate(LOp.RZ, h0, -1, hp, e)
            self.gate(LOp.RZ, h1, -1, hp, e)
            self.gate(LOp.RZZ, h0, h1, -hp, e)
        elif kind == LOp.CX:
            self.gate(LOp.H, h1, -1, 0.0, e)
            self.gate(LOp.CZ, h0, h1, 0.0, e)
            self.gate(LOp.H, h1, -1, 0.0, e)
        else:
            raise Error("QM-E724: logical op '" + lop_name(kind) + "' is not supported by the Iceberg architecture.")

    # ------------------------------------------------------------ measurement
    def _decode(mut self, raw: List[Int], slot: Int) raises -> CExpr:
        """Decoded Z_L(slot) of a destructively measured block; uncorrectable
        syndromes become post-selection flags."""
        var zl = self.code.z_logicals[slot].copy()
        if not zl.is_z_type():
            raise Error("QM-E724: decoding needs a Z-type logical Z operator (code '" + self.code.name + "'). Hint: transversal Z measurement decodes CSS codes only.")
        var val = CExpr()
        for q in range(self.n):
            if zl.z(q) == 1:
                val.xor_term(CTerm.clbit(raw[q]))
        var m = len(self.zchecks)
        if m == 0:
            return val^
        var flips = self.dec_flip[slot].copy()
        var any_correction = False
        var any_flag = False
        for sig in range(1, 1 << m):
            if flips[sig] == 1:
                any_correction = True
            if flips[sig] < 0:
                any_flag = True
        if m == 1 and not any_correction:
            # single parity check: the flag is its XOR, no lane needed
            var f = CExpr()
            for q in self.zchecks[0]:
                f.xor_term(CTerm.clbit(raw[q]))
            self.post.append(f^)
            return val^
        var sbits = List[Int]()
        for ci in range(m):
            var s = CExpr()
            for q in self.zchecks[ci]:
                s.xor_term(CTerm.clbit(raw[q]))
            sbits.append(self.materialize(s))
        var flag = CExpr()
        for sig in range(1, 1 << m):
            var vals = List[Int]()
            for ci in range(m):
                vals.append((sig >> ci) & 1)
            if flips[sig] == 1:
                val.xor_term(CTerm.minterm(sbits, vals))
            elif flips[sig] < 0:
                flag.xor_term(CTerm.minterm(sbits, vals))
        if any_flag:
            self.post.append(flag^)
        return val^

    def measure(mut self, h: Int) raises -> CExpr:
        """Destructive logical Z measurement of handle h; returns its value."""
        var b = self.hblock[h]
        var s = self.hslot[h]
        var qs = self.blocks[b].qubits.copy()
        if self.arch.kind == "iceberg" and self._other_live_slots(h) > 0:
            # non-destructive Z_b Z_s parity with one ancilla (other slots live on)
            var anc = self.phys()
            self._sync_size()
            self.circ.cx(qs[1], anc)
            self.circ.cx(qs[2 + s], anc)
            var mb = self.new_clbit()
            self.circ.measure(anc, mb)
            var al = List[Int]()
            al.append(anc)
            self.release_qubits(al)
            var v = CExpr.bit(mb)
            v.xor(self.fx[h])
            self._retire(h)
            self.blocks[b].residual[s] = CExpr.bit(mb)
            self.stats.decoded_measurements += 1
            return v^
        self._sync_size()
        var raw = List[Int]()
        for q in qs:
            var c = self.new_clbit()
            self.circ.measure(q, c)
            raw.append(c)
        var v = self._decode(raw, s)
        v.xor(self.fx[h])
        self.stats.decoded_measurements += 1
        self._retire(h)
        self.release_block(b)
        return v^

    def measure_block(mut self, hs: List[Int]) raises -> List[CExpr]:
        """Destructive measurement of a whole block holding exactly the live
        handles `hs`: one transversal readout, every slot decoded, one shared
        error-detection flag (instead of k-1 non-destructive parities)."""
        var b = self.hblock[hs[0]]
        var qs = self.blocks[b].qubits.copy()
        self._sync_size()
        var raw = List[Int]()
        for q in qs:
            var c = self.new_clbit()
            self.circ.measure(q, c)
            raw.append(c)
        var out = List[CExpr]()
        var first = True
        for h in hs:
            var v: CExpr
            if first:
                v = self._decode(raw, self.hslot[h])
                first = False
            else:
                v = self._decode_value_only(raw, self.hslot[h])
            v.xor(self.fx[h])
            out.append(v^)
            self.stats.decoded_measurements += 1
        for h in hs:
            self._retire(h)
        self.release_block(b)
        return out^

    def _decode_value_only(mut self, raw: List[Int], slot: Int) raises -> CExpr:
        """Decoded value without registering the (shared) detection flag again."""
        var saved = len(self.post)
        var v = self._decode(raw, slot)
        while len(self.post) > saved:
            _ = self.post.pop()
        return v^

    def free(mut self, h: Int) raises:
        if self.arch.kind == "iceberg" and self._other_live_slots(h) > 0:
            _ = self.measure(h)
            return
        var b = self.hblock[h]
        self._retire(h)
        self.release_block(b)

    # ------------------------------------------------------------ QEC
    def qec_cycle(mut self, h: Int) raises:
        var style = self.arch.qec.style
        if style == "none":
            return
        self.stats.qec_cycles += 1
        if self.arch.kind == "iceberg":
            var qs = self.block_qubits(h)
            var anc = self.phys_fresh_list(2)
            for a in anc:
                qs.append(a)
            var flags = List[Int]()
            self.emit_template(iceberg_syndrome_cycle(self.n), qs, Cond.none(), flags)
            for f in flags:
                self.post.append(CExpr.bit(f))
            self.release_qubits(anc)
            self.blocks[self.hblock[h]].counter = 0.0
            return
        var one = CExpr.one()
        if style == "steane":
            # X-type syndrome by one-bit teleportation into a fresh |0>_L
            var a = self.new_handle()
            self.alloc(a, False, False)
            self.swap_handles(h, a)
            self.gate(LOp.CX, a, h, 0.0, one)
            self.gate(LOp.H, a, -1, 0.0, one)
            var r = self.measure(a)
            self.apply_pauli(h, "Z", r)
            # Z-type syndrome by one-bit teleportation into a fresh |+>_L
            var a2 = self.new_handle()
            self.alloc(a2, False, False)
            self.gate(LOp.H, a2, -1, 0.0, one)
            self.swap_handles(h, a2)
            self.gate(LOp.CX, h, a2, 0.0, one)
            var r2 = self.measure(a2)
            self.apply_pauli(h, "X", r2)
        else:  # knill: Bell-pair teleportation
            var a0 = self.new_handle()
            self.alloc(a0, False, False)
            var a1 = self.new_handle()
            self.alloc(a1, False, False)
            self.gate(LOp.H, a0, -1, 0.0, one)
            self.gate(LOp.CX, a0, a1, 0.0, one)
            self.swap_handles(h, a1)
            self.gate(LOp.CX, a1, a0, 0.0, one)
            self.gate(LOp.H, a1, -1, 0.0, one)
            var r0 = self.measure(a0)
            self.apply_pauli(h, "X", r0)
            var r1 = self.measure(a1)
            self.apply_pauli(h, "Z", r1)
        self.blocks[self.hblock[h]].counter = 0.0

    def after_op(mut self, kind: Int, h0: Int, h1: Int) raises:
        """QECPolicy: add the op cost to every touched block; cycle at threshold."""
        var cost = self.arch.qec.costs[kind]
        if cost <= 0.0 or self.arch.qec.style == "none":
            return
        var hs = List[Int]()
        if h0 >= 0 and self.hlive[h0]:
            hs.append(h0)
        if h1 >= 0 and self.hlive[h1] and self.hblock[h1] != self.hblock[h0]:
            hs.append(h1)
        for h in hs:
            var b = self.hblock[h]
            self.blocks[b].counter += cost
            if self.blocks[b].counter >= self.arch.qec.threshold:
                self.qec_cycle(h)

    # ------------------------------------------------------------ classical
    def logical_cond(mut self, lp: LogicalProgram, g: LogicalInstr) raises -> CExpr:
        if g.cond < 0:
            return CExpr.one()
        var bits = lp.condition_bits(g)
        var vals = lp.condition_vals(g)
        if len(bits) == 1:
            var e = self.out_value(bits[0])
            if vals[0] == 0:
                e.xor_term(CTerm.one())
            return e^
        var pb = List[Int]()
        for b in bits:
            pb.append(self.materialize(self.out_value(b)))
        return CExpr.of(CTerm.minterm(pb, vals))

    def out_value(self, c: Int) -> CExpr:
        if c < len(self.out_expr):
            return self.out_expr[c].copy()
        return CExpr()

    def set_out(mut self, c: Int, var e: CExpr):
        while len(self.out_expr) <= c:
            self.out_expr.append(CExpr())
            self.out_written.append(False)
        self.out_expr[c] = e^
        self.out_written[c] = True

    def translate(mut self, e: CExpr) raises -> CExpr:
        """CExpr over logical clbits -> CExpr over physical clbits."""
        var out = CExpr()
        for t in e.terms:
            if t.kind == 2:
                out.xor_term(CTerm.one())
            elif t.kind == 0:
                out.xor(self.out_value(t.lits[0]))
            else:
                var pb = List[Int]()
                for b in t.bits():
                    pb.append(self.materialize(self.out_value(b)))
                out.xor_term(CTerm.minterm(pb, t.vals()))
        return out^


def _same_expr(a: CExpr, b: CExpr) -> Bool:
    var x = a.xored(b)
    return x.is_zero()


def implement_ops(arch: Architecture, lp: LogicalProgram) raises -> EncodedProgram:
    """Link every logical op of `lp` to its physical gadget (Guppy FT
    `SteaneInstance.implement_ops`)."""
    arch.validate()
    lp.validate()
    var eng = Engine(arch)
    eng.ensure_handles(lp.num_logical)
    eng.remaining_magic = lp.count(LOp.PREP_T) + lp.count(LOp.T) + lp.count(LOp.TDG)
    eng.remaining_zero = lp.count(LOp.ALLOC)
    eng._refill()
    var skip_measured = List[Int]()
    for i in range(len(lp.ops)):
        var g = lp.ops[i]
        eng.stats.logical_ops += 1
        var k = g.kind
        if k == LOp.ALLOC:
            eng.alloc(g.q0, False, True)
            eng.after_op(k, g.q0, -1)
        elif k == LOp.PREP_T:
            if arch.kind != "steane":
                raise Error("QM-E724: prep_t (magic state) is a Steane-architecture op. Hint: Iceberg applies rz(pi/4) natively.")
            eng.alloc(g.q0, True, True)
            eng.after_op(k, g.q0, -1)
        elif k == LOp.FREE:
            eng.free(g.q0)
        elif k == LOp.MEASURE:
            if arch.kind == "iceberg" and eng._other_live_slots(g.q0) > 0:
                # look ahead: consecutive measurements covering the whole block
                var b = eng.hblock[g.q0]
                var hs = List[Int]()
                var cs = List[Int]()
                var j = i
                while j < len(lp.ops) and lp.ops[j].kind == LOp.MEASURE and lp.ops[j].cond < 0:
                    if eng.hlive[lp.ops[j].q0] and eng.hblock[lp.ops[j].q0] == b:
                        hs.append(lp.ops[j].q0)
                        cs.append(lp.ops[j].clbit)
                    j += 1
                if len(hs) == eng._other_live_slots(g.q0) + 1:
                    var vals = eng.measure_block(hs)
                    for t in range(len(hs)):
                        eng.set_out(cs[t], vals[t].copy())
                    skip_measured.extend(hs.copy())
                    continue
            var already = False
            for x in skip_measured:
                if x == g.q0:
                    already = True
            if already:
                continue
            var v = eng.measure(g.q0)
            eng.set_out(g.clbit, v^)
        elif k == LOp.QEC:
            eng.qec_cycle(g.q0)
        elif k == LOp.INJECT_T or k == LOp.INJECT_TDG:
            if arch.kind != "steane":
                raise Error("QM-E724: inject_t is a Steane-architecture op.")
            if g.cond >= 0:
                raise Error("QM-E726: conditioned magic-state injection is not supported. Hint: inject unconditionally.")
            eng.inject(g.q0, g.q1, k == LOp.INJECT_TDG)
            eng.after_op(k, g.q0, -1)
        elif k == LOp.BARRIER:
            eng._sync_size()
            eng.circ.barrier()
        elif k == LOp.CSET:
            if g.cond >= 0:
                raise Error("QM-E727: CSET cannot be conditioned. Hint: fold the condition into the expression.")
            var e = eng.translate(lp.cexprs[g.q0])
            eng.set_out(g.clbit, e^)
        else:
            var e = eng.logical_cond(lp, g)
            eng.gate(k, g.q0, g.q1, g.angle, e)
            eng.after_op(k, g.q0, g.q1)
    # outputs and post-selection flags are read only now (deferred reads)
    var outs = List[Int]()
    for c in range(lp.num_clbits):
        if c < len(eng.out_written) and eng.out_written[c]:
            var oe = eng.out_expr[c].copy()
            outs.append(eng.materialize(oe))
        else:
            outs.append(-1)
    var posts = List[Int]()
    var flags = eng.post.copy()
    for f in flags:
        if not f.is_zero():
            posts.append(eng.materialize(f))
    eng._sync_size()
    if eng.circ.num_clbits < eng.nclb:
        eng.circ.num_clbits = eng.nclb
    eng.circ.name = lp.name + "_encoded_" + arch.kind
    eng.stats.physical_qubits = eng.nq
    eng.stats.lane_qubits = len(eng.lanes)
    var prog = EncodedProgram(arch.name(), arch.code.name, eng.circ.copy(), lp.copy())
    prog.outputs = outs^
    prog.postselect = posts^
    prog.lane_qubits = eng.lanes.copy()
    prog.stats = eng.stats.copy()
    prog.circuit.validate()
    return prog^
