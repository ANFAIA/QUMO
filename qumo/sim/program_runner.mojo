"""Shot-by-shot interpreter of a `Program`.

The interpreter walks `cfg_of(p)`: blocks of quantum ops go to a backend, ASSIGN
and branch conditions run on a `CStore` that is allocated once, so a shot
allocates nothing. Backends are static generics over `ShotBackend`.

A program without runtime regions is a plain circuit: `sample_program` hands it to
`sample`, which is what makes `from_circuit` give identical counts. The state-vector
backend runs blocks through the same `run_trajectory_ops` as `sample`, in the same
RNG order, and keeps pending 1q gates across blocks (qumo.sim.runner.Fuser).

A shot counts the back-edges it takes; past `max_iterations` it raises QM-E480,
which is how a classical infinite loop ends. Parameters are `input` values first
in `Program.user_params` order; slots of runtime angles fill in as variables are
assigned.
"""

from std.math import isnan, isinf

from qumo.ir.circuit import Circuit
from qumo.ir.classical import BitReader, CN, CT, CEval, CStore, angle_radians, eval_nodes, wrap_int
from qumo.ir.program import BB, Cfg, FORR, Program, SK, TERM, check_program
from qumo.sim.rng import Rng
from qumo.sim.runner import (
    Fuser, SampleResult, check_statevector_size, run_trajectory_ops, sample,
)
from qumo.sim.statevector import StateVector

comptime DEFAULT_MAX_ITERATIONS = 65536


struct ProgramResult(Copyable, Writable):
    """Counts plus what the classical side computed in each shot."""

    var sample: SampleResult
    var out_names: List[String]  # declared outputs other than bit registers
    var out_is_real: List[Bool]
    var out_int: List[Int]  # shots x outputs: value of output j in shot s at [s * n + j]
    var out_real: List[Float64]  # same shape; holds the float outputs
    var iterations: List[Int]  # back-edges per shot

    def __init__(out self, var sample: SampleResult):
        self.sample = sample^
        self.out_names = List[String]()
        self.out_is_real = List[Bool]()
        self.out_int = List[Int]()
        self.out_real = List[Float64]()
        self.iterations = List[Int]()

    def _output(self, name: String) raises -> Int:
        for j in range(len(self.out_names)):
            if self.out_names[j] == name:
                return j
        raise Error("QM-E913: no output named '" + name + "'. Hint: declare it 'output int " + name + ";' or read one of the declared outputs.")

    def int_output(self, name: String) raises -> List[Int]:
        """Value of an integer, bool or angle output per shot (angle: its bits)."""
        var j = self._output(name)
        var n = len(self.out_names)
        var out = List[Int](capacity=len(self.iterations))
        for s in range(len(self.iterations)):
            out.append(self.out_int[s * n + j])
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


trait ShotBackend(BitReader):
    """One simulator as the interpreter sees it."""

    def begin_shot(mut self, shot: Int):
        ...

    def run_block(mut self, p: Program, lo: Int, hi: Int, params: List[Float64]) raises:
        ...

    def set_bit(mut self, c: Int, v: Int):
        ...

    def packed_bits(self, n: Int) -> Int:
        """The first n clbits, bit c = clbit c."""
        ...


struct SvBackend(ShotBackend, Movable):
    """State vector, with the trajectory loop of `sample`."""

    var sv: StateVector
    var fz: Fuser
    var root: Rng
    var rng: Rng
    var creg: Int
    var chain_fired: Bool

    def __init__(out self, n: Int, root: Rng):
        self.sv = StateVector(n)
        self.fz = Fuser(n)
        self.root = root.copy()
        self.rng = root.copy()
        self.creg = 0
        self.chain_fired = False

    def begin_shot(mut self, shot: Int):
        self.rng = self.root.split(shot)
        self.sv.reset_to_zero()
        self.fz.clear()
        self.creg = 0
        self.chain_fired = False

    def run_block(mut self, p: Program, lo: Int, hi: Int, params: List[Float64]) raises:
        run_trajectory_ops(p.body, lo, hi, params, self.sv, self.fz, self.rng, self.creg, self.chain_fired)

    def get_bit(self, c: Int) -> Int:
        return (self.creg >> c) & 1

    def set_bit(mut self, c: Int, v: Int):
        if v != 0:
            self.creg |= 1 << c
        else:
            self.creg &= ~(1 << c)

    def packed_bits(self, n: Int) -> Int:
        return self.creg


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


def _store[B: ShotBackend](p: Program, var_id: Int, root: Int, mut st: CStore, mut b: B, shot: Int, line: Int) raises:
    """var = value of node `root`, keeping the angle slot in step."""
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
            if isnan(a) or isinf(a):
                raise _eval_error(CEval.FLOAT_RANGE, shot, line)
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
    st.ints[store] = cur
    st.ints[store + 1] = step
    st.ints[store + 2] = st.ti[r2]


def _for_test[B: ShotBackend](p: Program, at: Int, mut st: CStore, mut b: B, shot: Int) raises -> Bool:
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
        if (step > 0 and cur > stop) or (step < 0 and cur < stop):
            return False
        value = cur
    var ty = p.vars[s.a].ty
    # the loop variable goes through a node slot, like any assignment
    var tmp = len(st.ti) - 1
    st.ti[tmp] = wrap_int(value, ty)
    _store(p, s.a, tmp, st, b, shot, s.line)
    return True


def _for_step(p: Program, at: Int, mut st: CStore):
    var off = p.tape[at].b
    var store = p.for_ranges[off + FORR.STORE]
    if p.for_ranges[off + FORR.SET] >= 0:
        st.ints[store] += 1
    else:
        st.ints[store] += st.ints[store + 1]


def run_shot[B: ShotBackend](p: Program, cfg: Cfg, mut b: B, mut st: CStore, shot: Int, max_iterations: Int) raises -> Int:
    """One shot through the graph; returns the back-edges taken."""
    var iters = 0
    var at = 0
    while True:
        ref bb = cfg.blocks[at]
        for i in range(bb.lo, bb.hi):
            ref s = p.tape[i]
            if s.kind == SK.OPS:
                b.run_block(p, s.a, s.b, st.params)
            else:
                var root = _eval(p, s.b, st, b, shot, s.line)
                _store(p, s.a, root, st, b, shot, s.line)
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
                take = _for_test(p, -1 - bb.cond, st, b, shot)
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


def _record_outputs(p: Program, outs: List[Int], st: CStore, mut res: ProgramResult):
    for v in outs:
        ref cv = p.vars[v]
        if cv.ty.tag == CT.FLOAT:
            res.out_int.append(0)
            res.out_real.append(st.reals[cv.store])
        else:
            res.out_int.append(st.ints[cv.store])
            res.out_real.append(0.0)


def _new_result(p: Program, shots: Int, method: String) -> ProgramResult:
    var res = ProgramResult(SampleResult(p.num_clbits(), shots, method))
    for v in _output_vars(p):
        res.out_names.append(p.vars[v].name)
        res.out_is_real.append(p.vars[v].ty.tag == CT.FLOAT)
    return res^


def sample_interpreted(p: Program, shots: Int, seed: UInt64 = 1234,
                       params: List[Float64] = List[Float64](),
                       max_iterations: Int = DEFAULT_MAX_ITERATIONS) raises -> ProgramResult:
    """Run every shot through the interpreter, even for a plain circuit."""
    var cfg = check_program(p)
    check_statevector_size(p.num_qubits())
    if p.num_clbits() > 62:
        raise Error("QM-E406: the sampler packs classical registers into 62 bits; program has " + String(p.num_clbits()) + " clbits. Hint: measure fewer bits or split the register.")
    var full = shot_params(p, params)
    var res = _new_result(p, shots, "statevector:trajectories")
    var outs = _output_vars(p)
    var st = CStore(p.num_store, len(p.nodes) + 1, full^)
    var b = SvBackend(p.num_qubits(), Rng(seed))
    for shot in range(shots):
        b.begin_shot(shot)
        res.iterations.append(run_shot(p, cfg, b, st, shot, max_iterations))
        res.sample.record(b.packed_bits(p.num_clbits()))
        _record_outputs(p, outs, st, res)
    return res^


def sample_program(p: Program, shots: Int, seed: UInt64 = 1234,
                   params: List[Float64] = List[Float64](),
                   max_iterations: Int = DEFAULT_MAX_ITERATIONS) raises -> ProgramResult:
    """Sample a program. A plain circuit takes the path of `sample`."""
    var c = p.try_flatten()
    if c:
        var res = ProgramResult(sample(c.value(), shots, seed, params))
        res.iterations = List[Int](length=shots, fill=0)
        return res^
    return sample_interpreted(p, shots, seed, params, max_iterations)
