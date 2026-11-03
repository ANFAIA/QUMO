"""Shot-by-shot interpreter of a `Program`.

The interpreter walks `cfg_of(p)`: blocks of quantum ops go to a backend, ASSIGN
and branch conditions run on a `CStore` that is allocated once per worker.
Backends are static generics over `ShotBackend`:
  * `SvBackend`, the state vector. Blocks run through `run_trajectory_ops`, the
    loop of `sample`, in the same RNG order, and pending 1q gates survive block
    boundaries. Long stretches of static unitary gates run from a `BlockPlan`,
    built once by `build_plan`.
  * `TableauBackend` and `BranchingBackend`, through `run_range`. They refuse
    symbolic gates, so each runtime angle is bound as its gate runs; a value that
    is not a Clifford angle raises QM-E701 on the tableau.

`sample_program` picks the method. A program without runtime regions is a plain
circuit and goes to `sample`, which is what makes `from_circuit` give identical
counts. Otherwise a program of Clifford gates and Pauli noise goes to the tableau,
and the rest to the state vector. 'gpu' runs on the CPU with QM-W020, because each
shot branches on its own. 'density', 'ptsbe' and 'frame' raise QM-E483: each of them
needs one fixed sequence of operations for all shots.

Shots run in parallel up to PAR_SHOTS_MAX_QUBITS qubits, each on `rng.split(shot)`,
so the counts do not depend on the number of workers; above that, shots run in
series and the kernels in parallel.

A shot counts the back-edges it takes; past `max_iterations` it raises QM-E480,
which is how a classical infinite loop ends. Parameters are `input` values first
in `Program.user_params` order; slots of runtime angles fill in as variables are
assigned.
"""

from std.math import isnan, isinf
from std.runtime import parallelism_level

from max.algorithm import parallelize

from qumo.ir.circuit import Circuit, Instruction, check_finite_values, check_shots
from qumo.ir.classical import BitReader, CT, CEval, CStore, angle_radians, eval_nodes, wrap_int
from qumo.ir.gates import Op, op_is_noise, op_is_pauli_noise, op_is_unitary
from qumo.ir.program import Cfg, FORR, Program, SK, TERM, check_program, slot_prefix
from qumo.sim.branching import BranchingSimulator
from qumo.sim.rng import Rng
from qumo.sim.runner import (
    Fuser, KOp, PAR_SHOTS_MAX_QUBITS, SampleResult, build_plan, check_statevector_size, execute_plan,
    run_trajectory_ops, sample,
)
from qumo.sim.stabilizer import TableauSimulator
from qumo.sim.statevector import StateVector

comptime DEFAULT_MAX_ITERATIONS = 65536
comptime PLAN_MIN_GATES = 32  # a stretch of static gates this long runs from a plan


struct ProgramResult(Copyable, Writable):
    """Counts plus what the classical side computed in each shot."""

    var sample: SampleResult
    var out_names: List[String]  # declared outputs other than bit registers
    var out_is_real: List[Bool]
    var out_is_unsigned: List[Bool]  # uint and angle: `out_int` holds their 64 bits
    var out_int: List[Int]  # shots x outputs: value of output j in shot s at [s * n + j]
    var out_real: List[Float64]  # same shape; holds the float outputs
    var iterations: List[Int]  # back-edges per shot
    var warnings: List[String]  # QM-Wxxx notes on how the run was done

    def __init__(out self, var sample: SampleResult):
        self.sample = sample^
        self.out_names = List[String]()
        self.out_is_real = List[Bool]()
        self.out_is_unsigned = List[Bool]()
        self.out_int = List[Int]()
        self.out_real = List[Float64]()
        self.iterations = List[Int]()
        self.warnings = List[String]()

    def _output(self, name: String) raises -> Int:
        for j in range(len(self.out_names)):
            if self.out_names[j] == name:
                return j
        raise Error("QM-E913: no output named '" + name + "'. Hint: declare it 'output int " + name + ";' or read one of the declared outputs.")

    def int_output(self, name: String) raises -> List[Int]:
        """Value of an integer, bool or angle output per shot (angle: its bits). A uint
        at or above 2^63 comes back as its 64 bits, negative: `uint_output` reads it."""
        var j = self._output(name)
        var n = len(self.out_names)
        var out = List[Int](capacity=len(self.iterations))
        for s in range(len(self.iterations)):
            out.append(self.out_int[s * n + j])
        return out^

    def uint_output(self, name: String) raises -> List[UInt64]:
        """Value of a uint or angle output per shot; QM-E913 for a signed one."""
        var j = self._output(name)
        if not self.out_is_unsigned[j]:
            raise Error("QM-E913: output '" + name + "' is not unsigned. Hint: read it with int_output or real_output.")
        var n = len(self.out_names)
        var out = List[UInt64](capacity=len(self.iterations))
        for s in range(len(self.iterations)):
            out.append(UInt64(self.out_int[s * n + j]))
        return out^

    def real_output(self, name: String) raises -> List[Float64]:
        var j = self._output(name)
        var n = len(self.out_names)
        var out = List[Float64](capacity=len(self.iterations))
        for s in range(len(self.iterations)):
            out.append(self.out_real[s * n + j])
        return out^

    def mean_iterations(self) -> Float64:
        var t = 0
        for v in self.iterations:
            t += v
        return Float64(t) / Float64(max(len(self.iterations), 1))

    def write_to(self, mut w: Some[Writer]):
        w.write(self.sample, " mean_iterations=", self.mean_iterations())


@fieldwise_init
struct _Span(ImplicitlyCopyable):
    var lo: Int
    var hi: Int
    var plan: Int  # index in BlockPlan.plans, or -1: run op by op


struct BlockPlan(Copyable):
    """Each OPS block as spans: stretches of at least PLAN_MIN_GATES gates that
    are unitary, unconditioned and free of runtime angles become one fused,
    cache-blocked plan, built once per run; each worker holds a copy."""

    var spans: List[_Span]
    var first: List[Int]  # by op index: the first span of the block that starts there
    var plans: List[List[KOp]]

    def __init__(out self):
        self.spans = List[_Span]()
        self.first = List[Int]()
        self.plans = List[List[KOp]]()

    def __init__(out self, p: Program, params: List[Float64]) raises:
        self.spans = List[_Span]()
        self.plans = List[List[KOp]]()
        ref c = p.body
        self.first = List[Int](length=len(c.ops) + 1, fill=-1)
        var runtime = List[Bool](length=c.num_params(), fill=False)
        for i in range(c.num_params()):
            runtime[i] = c.param_names[i].startswith(slot_prefix())
        for s in p.tape:
            if s.kind != SK.OPS:
                continue
            self.first[s.a] = len(self.spans)
            var i = s.a
            while i < s.b:
                var j = i
                var gates = 0
                while j < s.b and _plannable(c.ops[j], runtime):
                    if op_is_unitary(c.ops[j].kind):
                        gates += 1
                    j += 1
                if gates >= PLAN_MIN_GATES:
                    var sub = c.empty_like()
                    for k in range(i, j):
                        sub.ops.append(c.ops[k])
                    self.spans.append(_Span(i, j, len(self.plans)))
                    self.plans.append(build_plan(sub, params))
                    i = j
                    continue
                var e = max(j, i + 1)
                var at = len(self.spans) - 1
                if at >= self.first[s.a] and self.spans[at].plan < 0:
                    self.spans[at].hi = e
                else:
                    self.spans.append(_Span(i, e, -1))
                i = e


def _plannable(g: Instruction, runtime: List[Bool]) -> Bool:
    if g.kind == Op.BARRIER or g.kind == Op.TICK:
        return True
    if g.cond >= 0 or not op_is_unitary(g.kind):
        return False
    return not ((g.p0.index >= 0 and runtime[g.p0.index]) or (g.p1.index >= 0 and runtime[g.p1.index]) or (g.p2.index >= 0 and runtime[g.p2.index]))


trait ShotBackend(BitReader, Movable, Deinitable):
    """One simulator as the interpreter sees it."""

    @staticmethod
    def make(p: Program, root: Rng, plan: BlockPlan) raises -> Self:
        ...

    @staticmethod
    def method() -> String:
        ...

    def begin_shot(mut self, shot: Int) raises:
        ...

    def run_block(mut self, p: Program, lo: Int, hi: Int, params: List[Float64]) raises:
        ...

    def set_bit(mut self, c: Int, v: Int):
        ...

    def packed_bits(self, n: Int) -> Int:
        """The first n clbits, bit c = clbit c."""
        ...


struct SvBackend(ShotBackend):
    """State vector, with the trajectory loop of `sample`."""

    var sv: StateVector
    var fz: Fuser
    var root: Rng
    var rng: Rng
    var creg: Int
    var chain_fired: Bool
    var plan: BlockPlan

    def __init__(out self, n: Int, root: Rng, plan: BlockPlan):
        self.sv = StateVector(n)
        self.fz = Fuser(n)
        self.root = root.copy()
        self.rng = root.copy()
        self.creg = 0
        self.chain_fired = False
        self.plan = plan.copy()

    @staticmethod
    def make(p: Program, root: Rng, plan: BlockPlan) raises -> Self:
        return SvBackend(p.num_qubits(), root, plan)

    @staticmethod
    def method() -> String:
        return "statevector:trajectories"

    def begin_shot(mut self, shot: Int) raises:
        self.rng = self.root.split(shot)
        self.sv.reset_to_zero()
        self.fz.clear()
        self.creg = 0
        self.chain_fired = False

    def run_block(mut self, p: Program, lo: Int, hi: Int, params: List[Float64]) raises:
        var at = self.plan.first[lo]
        var pos = lo
        while pos < hi:
            var sp = self.plan.spans[at]
            at += 1
            if sp.plan >= 0:
                self.fz.flush_all(self.sv)
                execute_plan(self.sv, self.plan.plans[sp.plan])
            else:
                run_trajectory_ops(p.body, sp.lo, sp.hi, params, self.sv, self.fz, self.rng, self.creg, self.chain_fired)
            pos = sp.hi

    def get_bit(self, c: Int) -> Int:
        return (self.creg >> c) & 1

    def set_bit(mut self, c: Int, v: Int):
        if v != 0:
            self.creg |= 1 << c
        else:
            self.creg &= ~(1 << c)

    def packed_bits(self, n: Int) -> Int:
        return self.creg


struct TableauBackend(ShotBackend):
    """Clifford programs of any width."""

    var sim: TableauSimulator
    var root: Rng
    var nc: Int

    def __init__(out self, n: Int, nc: Int, root: Rng) raises:
        self.sim = TableauSimulator(n, 0)
        self.root = root.copy()
        self.nc = nc
        self.sim.creg = List[UInt8](length=nc, fill=0)

    @staticmethod
    def make(p: Program, root: Rng, plan: BlockPlan) raises -> Self:
        return TableauBackend(p.num_qubits(), p.num_clbits(), root)

    @staticmethod
    def method() -> String:
        return "stabilizer:tableau"

    def begin_shot(mut self, shot: Int) raises:
        self.sim.reset_state()
        self.sim.rng = self.root.split(shot)
        self.sim.chain_fired = False
        for _ in range(self.nc):
            self.sim.creg.append(0)

    def run_block(mut self, p: Program, lo: Int, hi: Int, params: List[Float64]) raises:
        self.sim.run_range(p.body, lo, hi, True, False, params)

    def get_bit(self, c: Int) -> Int:
        return Int(self.sim.creg[c])

    def set_bit(mut self, c: Int, v: Int):
        self.sim.creg[c] = UInt8(1 if v != 0 else 0)

    def packed_bits(self, n: Int) -> Int:
        var out = 0
        for b in range(n):
            if self.sim.creg[b] != 0:
                out |= 1 << b
        return out


struct BranchingBackend(ShotBackend):
    """Near-Clifford programs past the state-vector limit."""

    var sim: BranchingSimulator
    var root: Rng
    var rng: Rng
    var n: Int
    var nc: Int

    def __init__(out self, n: Int, nc: Int, root: Rng) raises:
        self.sim = BranchingSimulator(n, 0)
        self.root = root.copy()
        self.rng = root.copy()
        self.n = n
        self.nc = nc

    @staticmethod
    def make(p: Program, root: Rng, plan: BlockPlan) raises -> Self:
        return BranchingBackend(p.num_qubits(), p.num_clbits(), root)

    @staticmethod
    def method() -> String:
        return "stabilizer:branching"

    def begin_shot(mut self, shot: Int) raises:
        self.rng = self.root.split(shot)
        var fresh = BranchingSimulator(self.n, self.rng.next_u64())
        self.sim = fresh^
        self.sim.frame.creg = List[UInt8](length=self.nc, fill=0)

    def run_block(mut self, p: Program, lo: Int, hi: Int, params: List[Float64]) raises:
        self.sim.run_range(p.body, lo, hi, self.rng, True, params)

    def get_bit(self, c: Int) -> Int:
        return Int(self.sim.frame.creg[c])

    def set_bit(mut self, c: Int, v: Int):
        self.sim.frame.creg[c] = UInt8(1 if v != 0 else 0)

    def packed_bits(self, n: Int) -> Int:
        var out = 0
        for b in range(n):
            if self.sim.frame.creg[b] != 0:
                out |= 1 << b
        return out


def _eval_error(status: Int, shot: Int, line: Int) -> Error:
    var at = "in shot " + String(shot) + " (line " + String(line) + ")"
    if status == CEval.DIV_ZERO:
        return Error("QM-E481: division by zero " + at + ". Hint: guard the divisor with an if.")
    if status == CEval.NEG_POWER:
        return Error("QM-E481: integer power with a negative exponent " + at + ". Hint: use a float base or a non-negative exponent.")
    return Error("QM-E482: float value is NaN, infinite or out of range for its target " + at + ". Hint: guard the value, or use a wider integer.")


@always_inline
def _eval[B: ShotBackend](p: Program, e: Int, mut st: CStore, b: B, shot: Int, line: Int) raises -> Int:
    """Evaluate expression `e`; returns its root node."""
    var root = p.expr_root[e]
    var status = eval_nodes(p.nodes, p.expr_lo[e], root, st, b)
    if status != CEval.OK:
        raise _eval_error(status, shot, line)
    return root


def _store[B: ShotBackend](p: Program, var_id: Int, root: Int, mut st: CStore, mut b: B) raises:
    """var = value of node `root`, keeping the angle slot in step. A slot may hold
    NaN: that is an error only for a gate that reads it, see `run_shot`."""
    ref v = p.vars[var_id]
    var tag = v.ty.tag
    if tag == CT.BIT:
        var x = st.ti[root]
        for j in range(v.ty.width):
            b.set_bit(v.store + j, (x >> j) & 1)
        return
    if tag == CT.FLOAT:
        st.reals[v.store] = st.tf[root]
    else:
        st.ints[v.store] = st.ti[root]
    if v.slot >= 0:
        var a: Float64
        if tag == CT.FLOAT:
            a = st.tf[root]
        elif tag == CT.ANGLE:
            a = angle_radians(st.ti[root], v.ty.width)
        elif tag == CT.INT:
            a = Float64(st.ti[root])
        else:
            a = Float64(UInt64(st.ti[root]))
        st.params[v.slot] = a


def _for_init[B: ShotBackend](p: Program, at: Int, mut st: CStore, b: B, shot: Int) raises:
    ref s = p.tape[at]
    var off = s.b
    var store = p.for_ranges[off + FORR.STORE]
    if p.for_ranges[off + FORR.SET] >= 0:
        st.ints[store] = 0
        return
    var r0 = _eval(p, p.for_ranges[off + FORR.START], st, b, shot, s.line)
    var cur = st.ti[r0]
    var r1 = _eval(p, p.for_ranges[off + FORR.STEP], st, b, shot, s.line)
    var step = st.ti[r1]
    var r2 = _eval(p, p.for_ranges[off + FORR.STOP], st, b, shot, s.line)
    if step == 0:
        raise Error("QM-E570: for-loop step is 0 in shot " + String(shot) + " (line " + String(s.line) + "). Hint: use a non-zero step.")
    var stop = st.ti[r2]
    var flags = p.for_ranges[off + FORR.UNSIGNED]
    if flags > FORR.BOUNDS_UNSIGNED:  # a uint in a 64-bit signed slot: see FORR
        var signed_ = (flags & FORR.BOUNDS_UNSIGNED) == 0
        if ((flags & FORR.STEP_UINT) != 0 and step < 0) or (signed_ and (((flags & FORR.START_UINT) != 0 and cur < 0) or ((flags & FORR.STOP_UINT) != 0 and stop < 0))):
            raise Error("QM-E482: a for-loop range holds a uint at or above 2^63, which it cannot read, in shot " + String(shot) + " (line " + String(s.line) + "). Hint: keep the step below 2^63, and both bounds uint if one is above it.")
    st.ints[store] = cur
    st.ints[store + 1] = step
    st.ints[store + 2] = stop


def _for_test[B: ShotBackend](p: Program, at: Int, mut st: CStore, mut b: B) raises -> Bool:
    """True while the loop has a next value, which goes to the loop variable."""
    ref s = p.tape[at]
    var off = s.b
    var store = p.for_ranges[off + FORR.STORE]
    var cur = st.ints[store]
    var value: Int
    var set_off = p.for_ranges[off + FORR.SET]
    if set_off >= 0:
        if cur >= p.case_vals[set_off]:
            return False
        value = p.case_vals[set_off + 1 + cur]
    else:
        var step = st.ints[store + 1]
        var stop = st.ints[store + 2]
        var unsigned_ = (p.for_ranges[off + FORR.UNSIGNED] & FORR.BOUNDS_UNSIGNED) != 0
        var past: Bool
        if unsigned_:
            past = UInt64(cur) > UInt64(stop) if step > 0 else UInt64(cur) < UInt64(stop)
        else:
            past = cur > stop if step > 0 else cur < stop
        if step == 0 or past:
            return False
        value = cur
        if unsigned_ and value < 0 and p.vars[s.a].ty.tag == CT.INT and p.vars[s.a].ty.width >= 64:
            raise Error("QM-E482: an int loop variable reached 2^63 (line " + String(s.line) + "). Hint: count in a uint.")
    var ty = p.vars[s.a].ty
    # the loop variable goes through a node slot, like any assignment
    var tmp = len(st.ti) - 1
    st.ti[tmp] = wrap_int(value, ty)
    _store(p, s.a, tmp, st, b)
    return True


def _for_step(p: Program, at: Int, mut st: CStore):
    """Advance a FOR. The next value may not exist in 64 bits, so the distance to
    the bound is compared first; step 0 then marks a finished loop (a zero step in
    the source is refused at the start of the loop)."""
    var off = p.tape[at].b
    var store = p.for_ranges[off + FORR.STORE]
    if p.for_ranges[off + FORR.SET] >= 0:
        st.ints[store] += 1
        return
    var cur = st.ints[store]
    var step = st.ints[store + 1]
    var stop = st.ints[store + 2]
    var room = UInt64(stop) - UInt64(cur) if step > 0 else UInt64(cur) - UInt64(stop)
    var stride = UInt64(step) if step > 0 else UInt64(0) - UInt64(step)
    if stride > room:
        st.ints[store + 1] = 0
    else:
        st.ints[store] = cur + step


def _check_angles(p: Program, ops: List[Int], st: CStore, shot: Int, line: Int) raises:
    """These gates are about to run: an angle that came out NaN or infinite stops
    here. A variable may hold one until a gate reads it."""
    for i in ops:
        ref g = p.body.ops[i]
        for f in range(3):
            var q = g.param(f)
            if q.index >= 0:
                var v = q.resolve(st.params)
                if isnan(v) or isinf(v):
                    raise _eval_error(CEval.FLOAT_RANGE, shot, line)


def run_shot[B: ShotBackend](p: Program, cfg: Cfg, mut b: B, mut st: CStore, shot: Int, max_iterations: Int) raises -> Int:
    """One shot through the graph; returns the back-edges taken."""
    var iters = 0
    var at = 0
    while True:
        ref bb = cfg.blocks[at]
        for i in range(bb.lo, bb.hi):
            ref s = p.tape[i]
            if s.kind == SK.OPS:
                _check_angles(p, cfg.slot_ops[i], st, shot, s.line)
                b.run_block(p, s.a, s.b, st.params)
            else:
                var root = _eval(p, s.b, st, b, shot, s.line)
                _store(p, s.a, root, st, b)
        if bb.init >= 0:
            _for_init(p, bb.init, st, b, shot)
        if bb.step >= 0:
            _for_step(p, bb.step, st)
        var term = bb.term
        if term == TERM.JMP:
            if bb.back:
                iters += 1
                if iters > max_iterations:
                    raise Error("QM-E480: shot " + String(shot) + " passed " + String(max_iterations) + " loop iterations (line " + String(bb.line) + "). Hint: the loop may never exit; raise max_iterations if it is only long.")
            at = bb.t0
        elif term == TERM.BR:
            var take: Bool
            if bb.cond >= 0:
                var root = _eval(p, bb.cond, st, b, shot, bb.line)
                take = st.ti[root] != 0
            else:
                take = _for_test(p, -1 - bb.cond, st, b)
            at = bb.t0 if take else bb.t1
        elif term == TERM.SWITCH:
            var root = _eval(p, bb.cond, st, b, shot, bb.line)
            var v = st.ti[root]
            var next = cfg.sw_bb[bb.t1]
            for j in range(bb.t0, bb.t1):
                if cfg.sw_keys[j] == v:
                    next = cfg.sw_bb[j]
                    break
            at = next
        else:
            return iters


def shot_params(p: Program, params: List[Float64]) raises -> List[Float64]:
    """The full parameter vector of a shot: `input` values, slots at 0."""
    check_finite_values(params)
    if len(params) != p.num_params():
        var names = String()
        for nm in p.user_param_names():
            names += " " + nm
        raise Error("QM-E101: program has " + String(p.num_params()) + " parameters [" + names + " ] but " + String(len(params)) + " values were given.")
    var out = List[Float64](length=p.body.num_params(), fill=0.0)
    for i in range(len(params)):
        out[p.user_params[i]] = params[i]
    return out^


def _output_vars(p: Program) -> List[Int]:
    var out = List[Int]()
    for i in range(len(p.vars)):
        if p.vars[i].is_output and p.vars[i].ty.tag != CT.BIT:
            out.append(i)
    return out^


def _run_shots[B: ShotBackend](p: Program, cfg: Cfg, shots: Int, seed: UInt64, full: List[Float64],
                               plan: BlockPlan, max_iterations: Int, workers: Int) raises -> ProgramResult:
    """All shots, in chunks of consecutive shots. Each chunk owns a backend and a
    store and writes to its own slots of the shared arrays, so the outcome is the
    same for any number of workers; the first failing shot is the one reported."""
    var outs = _output_vars(p)
    var nout = len(outs)
    var nc = p.num_clbits()
    var cregs = List[Int](length=shots, fill=0)
    var iters = List[Int](length=shots, fill=0)
    var ints = List[Int](length=shots * nout, fill=0)
    var reals = List[Float64](length=shots * nout, fill=0.0)
    var chunks = 1 if workers <= 1 else min(shots, workers * 4)
    chunks = max(chunks, 1)
    var errors = List[String](length=chunks, fill=String(""))
    var cp = cregs.unsafe_ptr().as_unsafe_any_origin()
    var ip = iters.unsafe_ptr().as_unsafe_any_origin()
    var vi = ints.unsafe_ptr().as_unsafe_any_origin()
    var vr = reals.unsafe_ptr().as_unsafe_any_origin()
    var ep = errors.unsafe_ptr().as_unsafe_any_origin()
    var prog = Pointer(to=p).as_unsafe_any_origin()
    var graph = Pointer(to=cfg).as_unsafe_any_origin()
    var fullp = Pointer(to=full).as_unsafe_any_origin()
    var planp = Pointer(to=plan).as_unsafe_any_origin()
    var outp = Pointer(to=outs).as_unsafe_any_origin()
    var root = Rng(seed)

    def chunk(c: Int) {imm cp, imm ip, imm vi, imm vr, imm ep, imm prog, imm graph, imm fullp, imm planp, imm outp, imm root, imm shots, imm chunks, imm nout, imm nc, imm max_iterations}:
        try:
            var b = B.make(prog[], root, planp[])
            var st = CStore(prog[].num_store, len(prog[].nodes) + 1, fullp[].copy())
            for shot in range(c * shots // chunks, (c + 1) * shots // chunks):
                b.begin_shot(shot)
                st.begin_shot(fullp[])
                ip[unsafe_offset=shot] = run_shot(prog[], graph[], b, st, shot, max_iterations)
                cp[unsafe_offset=shot] = b.packed_bits(nc)
                for j in range(nout):
                    ref cv = prog[].vars[outp[][j]]
                    if cv.ty.tag == CT.FLOAT:
                        vr[unsafe_offset=shot * nout + j] = st.reals[cv.store]
                    else:
                        vi[unsafe_offset=shot * nout + j] = st.ints[cv.store]
        except e:
            ep[unsafe_offset=c] = String(e)

    if chunks > 1:
        parallelize(chunk, chunks)
    else:
        chunk(0)
    _ = len(cregs)
    _ = len(iters)
    _ = len(ints)
    _ = len(reals)
    _ = len(outs)
    _ = root.s0
    for c in range(chunks):
        if errors[c].byte_length() > 0:
            raise Error(errors[c])
    var res = ProgramResult(SampleResult(nc, shots, B.method()))
    for v in outs:
        res.out_names.append(p.vars[v].name)
        res.out_is_real.append(p.vars[v].ty.tag == CT.FLOAT)
        res.out_is_unsigned.append(p.vars[v].ty.tag == CT.UINT or p.vars[v].ty.tag == CT.ANGLE)
    for s in range(shots):
        res.sample.record(cregs[s])
    res.iterations = iters^
    res.out_int = ints^
    res.out_real = reals^
    return res^


def is_clifford_program(p: Program) -> Bool:
    """Every block holds Clifford gates with constant angles and Pauli noise."""
    for s in p.tape:
        if s.kind != SK.OPS:
            continue
        for i in range(s.a, s.b):
            ref g = p.body.ops[i]
            if op_is_unitary(g.kind) and (g.is_parametric() or not g.is_clifford()):
                return False
            if op_is_noise(g.kind) and not op_is_pauli_noise(g.kind):
                return False
    return True


def _method_for(p: Program, method: String) raises -> String:
    """Which backend runs `p`: 'statevector', 'tableau' or 'branching'."""
    if method == "density" or method == "ptsbe" or method == "frame":
        raise Error("QM-E483: method '" + method + "' cannot run a program: it needs one fixed sequence of operations for all shots. Hint: use 'auto', 'statevector', 'tableau' or 'branching'.")
    if method == "auto" or method == "gpu":
        return "tableau" if is_clifford_program(p) else "statevector"
    if method == "stabilizer":
        return "tableau"
    if method == "statevector" or method == "tableau" or method == "branching":
        return method
    raise Error("QM-E483: unknown method '" + method + "'. Hint: use 'auto', 'statevector', 'tableau' or 'branching'.")


def sample_interpreted(p: Program, shots: Int, seed: UInt64 = 1234,
                       params: List[Float64] = List[Float64](),
                       max_iterations: Int = DEFAULT_MAX_ITERATIONS,
                       method: String = "statevector", workers: Int = 0) raises -> ProgramResult:
    """Run every shot through the interpreter, even for a plain circuit. `workers`
    0 picks parallel shots up to PAR_SHOTS_MAX_QUBITS qubits; 1 runs in series."""
    var cfg = check_program(p)
    if p.num_clbits() > 62:
        raise Error("QM-E406: the sampler packs classical registers into 62 bits; program has " + String(p.num_clbits()) + " clbits. Hint: measure fewer bits or split the register.")
    var full = shot_params(p, params)
    var m = _method_for(p, method)
    var w = workers
    if w <= 0:
        w = parallelism_level() if m != "statevector" or p.num_qubits() <= PAR_SHOTS_MAX_QUBITS else 1
    if m == "tableau":
        return _run_shots[TableauBackend](p, cfg, shots, seed, full, BlockPlan(), max_iterations, w)
    if m == "branching":
        return _run_shots[BranchingBackend](p, cfg, shots, seed, full, BlockPlan(), max_iterations, w)
    check_statevector_size(p.num_qubits())
    return _run_shots[SvBackend](p, cfg, shots, seed, full, BlockPlan(p, full), max_iterations, w)


def sample_program(p: Program, shots: Int, seed: UInt64 = 1234,
                   params: List[Float64] = List[Float64](),
                   max_iterations: Int = DEFAULT_MAX_ITERATIONS,
                   method: String = "auto") raises -> ProgramResult:
    """Sample a program with the method named in the module docstring. Any method
    other than 'auto' is taken at its word, and `result.warnings` tells what ran."""
    check_shots(shots)
    var chosen = _method_for(p, method)
    if (method == "auto" or method == "statevector") and not p.has_runtime_region():
        var c = p.try_flatten()
        if c:
            var res = ProgramResult(sample(c.value(), shots, seed, params))
            res.iterations = List[Int](length=shots, fill=0)
            return res^
    var res = sample_interpreted(p, shots, seed, params, max_iterations, chosen)
    if method == "gpu":
        res.warnings.append("QM-W020: device 'gpu' requested but a program branches shot by shot, which the GPU samplers do not run: ran on the CPU (" + res.sample.method + ").")
    return res^
