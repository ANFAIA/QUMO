"""Collect-and-resynthesize driver with verified smart replacement.

Native, hardware-neutral version of qiskit-ibm-transpiler 0.18.0's
`AICliffordSynthesis` / `AILinearFunctionSynthesis` / `AIPermutationSynthesis`
/ `AIPauliNetworkSynthesis` replace-only-if-better logic (ai/synthesis.py,
Apache-2.0; re-implemented, no code copied), with two differences:
every proposal is verified exactly before use (IBM does not check), and the
cost is counted in CX units of the *target's* entangler (vendor neutral).

Flow of `resynthesize_blocks` (one call = one pass):
  for rep in 0..reps-1, for each direction (forward / backward / both):
    1. `collect.collect_blocks_ordered` collects blocks of `opts.kind`.
    2. Per block (in parallel; the seed is derived from `seed` and the
       block's content hash, so the result depends neither on the thread
       count nor on the block's position): the block's ops are
       relabelled to local qubits 0..n-1 (sorted wire order) and candidate
       circuits are requested from the synthesizers:
         exact | astar | greedy -> block_synth with that method
         policy                -> policy_synth (learned proposer)
         auto                  -> block_synth "auto" (greedy portfolio plus a
                                  budgeted exact search for small blocks) and
                                  the policy when a model matches the block.
       Blocks with fewer than two 2q gates are skipped (already optimal), and
       identical blocks (same local ops and coupling) are searched once.
       Blocks whose (2q cost, 2q depth) already equals a sound lower bound
       (`_lower_bound`: SWAP-distance bound for permutations, light-cone
       bound of the GF(2) matrix / tableau supports for linear, Clifford and
       all-Clifford Pauli blocks) are not searched: no candidate could be
       strictly better (reported as at_lower_bound=k in the note).
    3. Every candidate that would replace the block (strictly better than
       the best so far, see 4) is VERIFIED independently of the synthesizers:
         - only 1q / 2q unconditioned bound gates on local qubits, every 2q
           gate on a coupling edge of the block's induced subgraph
           (physical=True) or anywhere (physical=False);
         - PERM: SWAPs only, same permutation; LINEAR: CX / SWAP only, same
           GF(2) matrix (both are exact unitary equality, phase 0);
         - CLIFFORD: same stabilizer tableau including signs
           (`TableauSimulator`), so equal up to a global phase, which is then
           measured on one statevector column (n <= 12);
         - PAULI: dense unitary equality up to a global phase (n <= 8):
           phase = arg tr(U^dag V), then every entry of V - e^{i phase} U
           must have modulus <= 1e-9 (UNITARY_TOL).
       The measured phase is added to `c.global_phase`, so the output is
       statevector-equivalent including the phase. Rejected candidates are
       counted in `rejected_not_equal`; `strict=True` raises QM-E619.
    4. SMART REPLACEMENT: score = (2q cost in target CX units, 2q depth) with
       `kak.cx_cost` (native entangler = 1, iSWAP = 2 per CX, SWAP = 3 unless
       native), compared lexicographically (`metric=cost`: cost only). The
       best verified candidate replaces the block only when strictly better;
       a tie keeps the original (IBM rule).
    5. Replacements are spliced into the collection order (a topological
       order in which every block is contiguous, see collect.mojo).
  After all sweeps: `cancel_inverses`, then, if the circuit was already
  expressed in the target basis before the pass, `translate_to_basis` (the
  KAK pattern), so a pass never leaves the basis.

Error codes: QM-E616 (unknown method / kind / metric / direction, or an
explicitly requested model that cannot be used), QM-E618 (max_qubits above
the limit of the verifier for that kind; with strict=1 also an explicit
exact / astar method on a block too wide for it, which is otherwise skipped
and reported as skipped_too_wide=k), QM-E619 (strict mode: a proposal was not
equivalent).
"""

from std.math import atan2, sqrt, cos, sin, pi
from std.memory import bitcast
from std.bit import pop_count
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_arity, matrix_2q
from qumo.compiler.target import Target
from qumo.compiler.common import GateSet, is_plain_gate, is_bound, inst_matrix_1q
from qumo.compiler.kak import cx_cost
from qumo.compiler.optimize import cancel_inverses
from qumo.compiler.basis import translate_to_basis
from qumo.compiler.collect import (
    Block, collect_blocks_ordered, block_gate_ok,
    KIND_CLIFFORD, KIND_LINEAR, KIND_PERM, KIND_PAULI,
)
from qumo.compiler.blockalg import SubGraph, Perm, GF2Mat, SympTab, PauliNet
from qumo.compiler.block_synth import synth_perm, synth_linear, synth_clifford, synth_pauli_network, exact_supported
from qumo.compiler.policy import model_search_dirs
from qumo.compiler.policy_synth import PolicySession
from qumo.sim.stabilizer import TableauSimulator
from qumo.sim.rng import Rng
from max.algorithm import parallelize

# Verifier width limits per kind (QM-E618 above them).
comptime MAX_W_CLIFFORD = 12  # phase measured on a 2^n statevector column
comptime MAX_W_PAULI = 8  # dense 2^n x 2^n unitary check
comptime MAX_W_LINEAR = 64  # GF(2) rows are 64-bit masks
comptime MAX_W_PERM = 64

# method ids (winner bookkeeping)
comptime _M_EXACT = 0
comptime _M_ASTAR = 1
comptime _M_GREEDY = 2
comptime _M_POLICY = 3
comptime _M_IDENTITY = 4
comptime _M_AUTO = 5  # block_synth "auto": greedy portfolio + budgeted exact search
comptime _M_BOUND = 6  # not a method: blocks skipped because already at the lower bound
comptime _NUM_M = 7


struct SynthOptions(Copyable):
    """Options of one collect-and-resynthesize pass.

    kind:       KIND_CLIFFORD (0) / KIND_LINEAR (1) / KIND_PERM (2) / KIND_PAULI (3).
    method:     auto | exact | astar | greedy | policy.
    max_qubits: block width cap (qubits); min_block: minimum gates per block.
    direction:  forward | backward | both (collection sweep direction).
    commute:    commutation-aware collection.
    reps:       collect-and-resynthesize repetitions (stop early when nothing changes).
    model:      auto | off | <model id> (learned proposer).
    beams / searches: policy search width / number of seeded searches.
    seed:       base seed (per block: Rng(seed).split(sweep).split(block)).
    metric:     cost_depth (2q cost, then 2q depth) | cost.
    strict:     raise QM-E619 when a proposal fails verification.
    budget:     node-expansion cap of exact / A* searches (0 = default).
    model_dir:  explicit model directory (searched first).
    """

    var kind: Int
    var method: String
    var max_qubits: Int
    var min_block: Int
    var direction: String
    var commute: Bool
    var reps: Int
    var model: String
    var beams: Int
    var searches: Int
    var seed: Int
    var metric: String
    var strict: Bool
    var budget: Int
    var model_dir: String

    def __init__(out self, kind: Int):
        self.kind = kind
        self.method = "auto"
        self.max_qubits = 9
        self.min_block = 2
        self.direction = "both"
        self.commute = True
        self.reps = 10
        self.model = "auto"
        self.beams = 8
        self.searches = 4
        self.seed = 7
        self.metric = "cost_depth"
        self.strict = False
        self.budget = 0
        self.model_dir = ""

    @staticmethod
    def default(kind: Int) -> SynthOptions:
        """IBM collector defaults: Clifford 9 qubits / 2 gates, linear function
        9 / 4, permutation 12 / 4, Pauli network 6 / 4; reps 10, both
        directions, commutation-aware, method auto, model auto."""
        var o = SynthOptions(kind)
        if kind == KIND_LINEAR:
            o.max_qubits = 9
            o.min_block = 4
        elif kind == KIND_PERM:
            o.max_qubits = 12
            o.min_block = 4
        elif kind == KIND_PAULI:
            o.max_qubits = 6
            o.min_block = 4
        return o^


struct SynthStats(Copyable):
    """Counters of one pass (accumulated over sweeps)."""

    var blocks: Int  # blocks considered (>= 1 two-qubit gate, connected)
    var proposed: Int  # non-empty candidates returned by synthesizers
    var verified: Int  # strictly better candidates that passed verification
    #                    (candidates that would not be used are not verified)
    var replaced: Int  # blocks replaced (strictly better)
    var saved_cost: Int  # 2q cost saved (target CX units)
    var saved_depth: Int  # sum of local 2q-depth reductions of replaced blocks
    var rejected_not_equal: Int  # strictly better candidates that failed verification
    var note: String  # winners per method, sweeps

    def __init__(out self):
        self.blocks = 0
        self.proposed = 0
        self.verified = 0
        self.replaced = 0
        self.saved_cost = 0
        self.saved_depth = 0
        self.rejected_not_equal = 0
        self.note = ""


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def score_cost(ops: List[Instruction], gs: GateSet) -> Int:
    """2q cost in target CX units (kak.cx_cost)."""
    var s = 0
    for g in ops:
        if op_arity(g.kind) == 2:
            s += cx_cost(g.kind, gs)
    return s


def score_depth2q(ops: List[Instruction], n: Int) -> Int:
    """Depth counting only 2q gates (1q gates are free)."""
    var lvl = List[Int](length=max(n, 1), fill=0)
    var d = 0
    for g in ops:
        if op_arity(g.kind) == 2:
            var l = max(lvl[g.q0], lvl[g.q1]) + 1
            lvl[g.q0] = l
            lvl[g.q1] = l
            d = max(d, l)
    return d


def _better(c1: Int, d1: Int, c0: Int, d0: Int, cost_only: Bool) -> Bool:
    """(c1, d1) strictly better than (c0, d0)."""
    if c1 != c0:
        return c1 < c0
    if cost_only:
        return False
    return d1 < d0


# ----------------------------------------------------------------------------
# verification (independent of the synthesizers)
# ----------------------------------------------------------------------------
def _gates_ok(kind: Int, cand: List[Instruction], n: Int, adj: List[Bool]) -> Bool:
    for g in cand:
        if not is_plain_gate(g) or not is_bound(g):
            return False
        var a = op_arity(g.kind)
        if a < 1 or a > 2:
            return False
        if g.q0 < 0 or g.q0 >= n:
            return False
        if a == 2:
            if g.q1 < 0 or g.q1 >= n or g.q1 == g.q0:
                return False
            if not adj[g.q0 * n + g.q1]:
                return False
        if kind == KIND_PERM and g.kind != Op.SWAP:
            return False
        if kind == KIND_LINEAR and g.kind != Op.SWAP and g.kind != Op.CX:
            return False
    return True


def _perm_of(ops: List[Instruction], n: Int) -> List[Int]:
    """where[i] = wire holding the state that started on wire i."""
    var at = List[Int](length=n, fill=0)  # at[w] = original wire whose state is on w
    for i in range(n):
        at[i] = i
    for g in ops:
        var t = at[g.q0]
        at[g.q0] = at[g.q1]
        at[g.q1] = t
    var where = List[Int](length=n, fill=0)
    for w in range(n):
        where[at[w]] = w
    return where^


def _gf2_of(ops: List[Instruction], n: Int) -> List[UInt64]:
    """Rows of the GF(2) matrix: x -> M x (CX(c,t): row t ^= row c)."""
    var rows = List[UInt64](length=n, fill=0)
    for i in range(n):
        rows[i] = UInt64(1) << UInt64(i)
    for g in ops:
        if g.kind == Op.CX:
            rows[g.q1] ^= rows[g.q0]
        else:
            var t = rows[g.q0]
            rows[g.q0] = rows[g.q1]
            rows[g.q1] = t
    return rows^


def _tableau_equal(a: List[Instruction], b: List[Instruction], n: Int) raises -> Bool:
    var ta = TableauSimulator(n)
    var tb = TableauSimulator(n)
    for g in a:
        ta.apply_gate(g)
    for g in b:
        tb.apply_gate(g)
    for i in range(len(ta.xs)):
        if ta.xs[i] != tb.xs[i] or ta.zs[i] != tb.zs[i]:
            return False
    for i in range(2 * n):
        if ta.signs[i] != tb.signs[i]:
            return False
    return True


def _unitary(ops: List[Instruction], n: Int, cols: Int, mut re: List[Float64], mut im: List[Float64]):
    """First `cols` columns of the dense unitary of bound 1q / 2q ops on n
    local qubits (little-endian), column-major: column c (= U|c>) is
    re[c*dim : (c+1)*dim]. Each gate's matrix is computed once and applied to
    every column."""
    var dim = 1 << n
    re = List[Float64](length=cols * dim, fill=0.0)
    im = List[Float64](length=cols * dim, fill=0.0)
    for c in range(cols):
        re[c * dim + c] = 1.0
    var pr = re.unsafe_ptr().as_unsafe_any_origin()
    var pi_ = im.unsafe_ptr().as_unsafe_any_origin()
    for g in ops:
        if op_arity(g.kind) == 1:
            var m = inst_matrix_1q(g)
            var bit = 1 << g.q0
            var m0r = m.re[0]
            var m0i = m.im[0]
            var m1r = m.re[1]
            var m1i = m.im[1]
            var m2r = m.re[2]
            var m2i = m.im[2]
            var m3r = m.re[3]
            var m3i = m.im[3]
            for base in range(0, cols * dim, dim):
                for i in range(dim):
                    if i & bit != 0:
                        continue
                    var x = base + i
                    var y = x + bit
                    var ar = pr[unsafe_offset=x]
                    var ai = pi_[unsafe_offset=x]
                    var br = pr[unsafe_offset=y]
                    var bi = pi_[unsafe_offset=y]
                    pr[unsafe_offset=x] = m0r * ar - m0i * ai + m1r * br - m1i * bi
                    pi_[unsafe_offset=x] = m0r * ai + m0i * ar + m1r * bi + m1i * br
                    pr[unsafe_offset=y] = m2r * ar - m2i * ai + m3r * br - m3i * bi
                    pi_[unsafe_offset=y] = m2r * ai + m2i * ar + m3r * bi + m3i * br
            continue
        var m4 = matrix_2q(g.kind, g.p0.value())
        var b0 = 1 << g.q0
        var b1 = 1 << g.q1
        for base in range(0, cols * dim, dim):
            for i in range(dim):
                if i & b0 != 0 or i & b1 != 0:
                    continue
                var i0 = base + i
                var i1 = i0 + b0
                var i2 = i0 + b1
                var i3 = i0 + b0 + b1
                var v = SIMD[DType.float64, 4](pr[unsafe_offset=i0], pr[unsafe_offset=i1], pr[unsafe_offset=i2], pr[unsafe_offset=i3])
                var w = SIMD[DType.float64, 4](pi_[unsafe_offset=i0], pi_[unsafe_offset=i1], pi_[unsafe_offset=i2], pi_[unsafe_offset=i3])
                var outr = SIMD[DType.float64, 4](0)
                var outi = SIMD[DType.float64, 4](0)
                comptime for r in range(4):
                    var sr = 0.0
                    var si = 0.0
                    comptime for k in range(4):
                        var mr = m4.re[r * 4 + k]
                        var mi = m4.im[r * 4 + k]
                        sr += mr * v[k] - mi * w[k]
                        si += mr * w[k] + mi * v[k]
                    outr[r] = sr
                    outi[r] = si
                pr[unsafe_offset=i0] = outr[0]
                pr[unsafe_offset=i1] = outr[1]
                pr[unsafe_offset=i2] = outr[2]
                pr[unsafe_offset=i3] = outr[3]
                pi_[unsafe_offset=i0] = outi[0]
                pi_[unsafe_offset=i1] = outi[1]
                pi_[unsafe_offset=i2] = outi[2]
                pi_[unsafe_offset=i3] = outi[3]
    _ = len(re)
    _ = len(im)


comptime UNITARY_TOL = 1e-9  # max entrywise |U_b - e^{i phi} U_a|


def _unitary_phase(a: List[Instruction], b: List[Instruction], n: Int, cols: Int, mut phase: Float64) -> Bool:
    """True when U_b = e^{i phase} U_a on the first `cols` columns (all of
    them when cols = 2^n): phase = arg tr(U_a^dag U_b) over those columns,
    then every entry must satisfy |U_b - e^{i phase} U_a| <= UNITARY_TOL
    (an entrywise bound, so e.g. a rotation angle off by 1e-8 is rejected)."""
    var ar = List[Float64]()
    var ai = List[Float64]()
    var br = List[Float64]()
    var bi = List[Float64]()
    _unitary(a, n, cols, ar, ai)
    _unitary(b, n, cols, br, bi)
    var total = cols * (1 << n)
    var sr = 0.0
    var si = 0.0
    for i in range(total):
        sr += ar[i] * br[i] + ai[i] * bi[i]
        si += ar[i] * bi[i] - ai[i] * br[i]
    if sqrt(sr * sr + si * si) < 0.5:
        return False  # far from proportional (also avoids an ill-defined arg)
    var ph = atan2(si, sr)
    var er = cos(ph)
    var ei = sin(ph)
    var tol2 = UNITARY_TOL * UNITARY_TOL
    for i in range(total):
        var dr = br[i] - (er * ar[i] - ei * ai[i])
        var di = bi[i] - (er * ai[i] + ei * ar[i])
        if dr * dr + di * di > tol2:
            return False
    phase = ph
    return True


def verify_candidate(kind: Int, orig: List[Instruction], cand: List[Instruction], n: Int,
                     adj: List[Bool], mut phase: Float64) -> Bool:
    """True when `cand` implements the same unitary as `orig` (local qubits
    0..n-1) using only allowed gates on `adj` edges; `phase` is set to phi with
    U_cand = e^{i phi} U_orig."""
    phase = 0.0
    if not _gates_ok(kind, cand, n, adj):
        return False
    if kind == KIND_PERM:
        var a = _perm_of(orig, n)
        var b = _perm_of(cand, n)
        for i in range(n):
            if a[i] != b[i]:
                return False
        return True
    if kind == KIND_LINEAR:
        var a = _gf2_of(orig, n)
        var b = _gf2_of(cand, n)
        for i in range(n):
            if a[i] != b[i]:
                return False
        return True
    if kind == KIND_CLIFFORD:
        if n > MAX_W_CLIFFORD:
            return False
        try:
            if not _tableau_equal(orig, cand, n):
                return False
        except:
            return False  # a non-Clifford gate in either circuit
        # equal tableaux: U_cand = e^{i phi} U_orig; measure phi on one column
        var ph = 0.0
        if not _unitary_phase(orig, cand, n, 1, ph):
            return False  # cannot happen for equal tableaux; be safe
        phase = ph
        return True
    if kind == KIND_PAULI:
        if n > MAX_W_PAULI:
            return False
        var ph = 0.0
        if not _unitary_phase(orig, cand, n, 1 << n, ph):
            return False
        phase = ph
        return True
    return False


# ----------------------------------------------------------------------------
# proposals (synthesizers are plugged in here)
# ----------------------------------------------------------------------------
def _method_name(m: Int) -> String:
    if m == _M_EXACT:
        return "exact"
    if m == _M_ASTAR:
        return "astar"
    if m == _M_GREEDY:
        return "greedy"
    if m == _M_POLICY:
        return "policy"
    if m == _M_AUTO:
        return "auto"
    return "identity"


def _propose(kind: Int, local: List[Instruction], g: SubGraph, method: Int, opts: SynthOptions,
             seed: Int, sess: PolicySession) raises -> List[Instruction]:
    """Candidate circuit (local qubits) from one method; empty = no proposal.
    The policy branch only uses the read-only `propose_*` methods of a
    preloaded session, so it is safe from parallel workers."""
    var n = g.n
    if method == _M_POLICY:
        if kind == KIND_PERM:
            return sess.propose_perm(Perm.from_ops(local, n), g, opts.model, opts.beams, opts.searches, seed)
        if kind == KIND_LINEAR:
            return sess.propose_linear(GF2Mat.from_ops(local, n), g, opts.model, opts.beams, opts.searches, seed)
        if kind == KIND_CLIFFORD:
            return sess.propose_clifford(SympTab.from_ops(local, n), g, opts.model, opts.beams, opts.searches, seed)
        return sess.propose_pauli_network(PauliNet.from_ops(local, n), g, opts.model, opts.beams, opts.searches, seed, local)
    var m = _method_name(method)
    if kind == KIND_PERM:
        return synth_perm(Perm.from_ops(local, n), g, m, opts.budget, seed)
    if kind == KIND_LINEAR:
        return synth_linear(GF2Mat.from_ops(local, n), g, m, opts.budget, seed)
    if kind == KIND_CLIFFORD:
        return synth_clifford(SympTab.from_ops(local, n), g, m, opts.budget, seed)
    return synth_pauli_network(PauliNet.from_ops(local, n), g, m, opts.budget, seed, local)


# ----------------------------------------------------------------------------
# per-block search
# ----------------------------------------------------------------------------
@fieldwise_init
struct _BlockResult(ImplicitlyCopyable):
    var ok: Bool  # replacement found (strictly better, verified)
    var cost: Int
    var depth: Int
    var phase: Float64
    var length: Int
    var winner: Int
    var proposed: Int
    var verified: Int
    var rejected: Int
    var error: Bool
    var at_bound: Bool  # incumbent provably optimal (lower bound), not searched


# ----------------------------------------------------------------------------
# lower bounds (sound; used to skip searches that cannot win)
# ----------------------------------------------------------------------------
def _ceil_log2(k: Int) -> Int:
    var d = 0
    while (1 << d) < k:
        d += 1
    return d


def _all_clifford(local: List[Instruction]) -> Bool:
    for g in local:
        if not g.is_clifford():
            return False
    return True


def _min_unit(gs: GateSet) -> Int:
    """Cheapest 2q gate of any kind in target CX units (>= 1)."""
    var m = 1 << 30
    for k in range(Op.CX, Op.RZZ + 1):
        m = min(m, cx_cost(k, gs))
    return max(m, 1)


def _cone_bound(dep: List[UInt64], n: Int, unit: Int, mut lbc: Int, mut lbd: Int):
    """Light-cone bounds from a dependency relation: dep[i] = mask of inputs
    that output i depends on (Clifford: support of the images; linear: matrix
    rows). Output i depending on input j needs a chain of 2q gates linking
    them, so every connected component of size k of the relation needs at least
    k - 1 two-qubit gates (components of the gate graph are unions of them),
    and a cone of size s (forward or backward) needs 2q depth >= ceil(log2 s),
    since one 2q layer at most doubles a cone."""
    var parent = List[Int](length=n, fill=0)
    for i in range(n):
        parent[i] = i
    var fwd = List[UInt64](length=n, fill=0)  # fwd[j] = outputs depending on input j
    var maxcone = 1
    for i in range(n):
        var m = dep[i]
        maxcone = max(maxcone, Int(pop_count(m)))
        for j in range(n):
            if (m >> UInt64(j)) & 1 != 0:
                fwd[j] |= UInt64(1) << UInt64(i)
                var a = _uf_find(parent, i)
                var b = _uf_find(parent, j)
                if a != b:
                    parent[a] = b
    for j in range(n):
        maxcone = max(maxcone, Int(pop_count(fwd[j])))
    var comps = 0
    for i in range(n):
        if _uf_find(parent, i) == i:
            comps += 1
    lbc = (n - comps) * unit
    lbd = _ceil_log2(maxcone)


def _uf_find(mut parent: List[Int], x: Int) -> Int:
    var r = x
    while parent[r] != r:
        r = parent[r]
    var y = x
    while parent[y] != r:
        var nx = parent[y]
        parent[y] = r
        y = nx
    return r


def _lower_bound(kind: Int, local: List[Instruction], g: SubGraph, gs: GateSet, mut lbc: Int, mut lbd: Int):
    """Sound lower bounds (2q cost in target CX units, 2q depth) of ANY
    circuit equal to the block (0, 0 when unknown):
      PERM:     each SWAP moves two states by one hop: ceil(sum dist / 2)
                SWAPs, depth >= max dist;
      LINEAR:   light-cone bound of the GF(2) matrix (see _cone_bound);
      CLIFFORD: light-cone bound of the tableau image supports;
      PAULI:    the Clifford bound when every gate is Clifford, else 0."""
    lbc = 0
    lbd = 0
    var n = g.n
    if n < 2 or n > 64:
        return
    if kind == KIND_PERM:
        var p = _perm_of(local, n)
        var tot = 0
        var md = 0
        for i in range(n):
            var d = g.dist[i * n + p[i]]
            if d < 0:
                return
            tot += d
            md = max(md, d)
        lbc = ((tot + 1) // 2) * cx_cost(Op.SWAP, gs)
        lbd = md
        return
    if kind == KIND_LINEAR:
        _cone_bound(_gf2_of(local, n), n, _min_unit(gs), lbc, lbd)
        return
    if kind == KIND_PAULI and not _all_clifford(local):
        return
    var rows = List[UInt64](length=n, fill=0)
    if _clifford_dep(local, n, rows):
        _cone_bound(rows, n, _min_unit(gs), lbc, lbd)


def _quarter_turns(g: Instruction) -> Int:
    """k mod 4 for a rotation by k*pi/2, -1 when not such a rotation."""
    if g.p0.is_symbolic():
        return -1
    var r = g.p0.value() / (pi / 2)
    var k = round(r)
    if abs(r - k) > 1e-12:
        return -1
    var ki = Int(k) % 4
    return ki + 4 if ki < 0 else ki


@always_inline
def _rot_xor(mut X: List[UInt64], mut Z: List[UInt64], a: Int, xa: Bool, za: Bool, b: Int, xb: Bool, zb: Bool):
    """Support update of conjugation by exp(-i pi/4 P), P = P_a (x) P_b
    (b < 0: 1q): rows anticommuting with P get P's bits xor-ed in (signs are
    ignored). X[q] / Z[q] are masks over tableau rows."""
    var anti: UInt64 = 0
    if xa:
        anti ^= Z[a]
    if za:
        anti ^= X[a]
    if b >= 0:
        if xb:
            anti ^= Z[b]
        if zb:
            anti ^= X[b]
    if xa:
        X[a] ^= anti
    if za:
        Z[a] ^= anti
    if b >= 0:
        if xb:
            X[b] ^= anti
        if zb:
            Z[b] ^= anti


def _clifford_dep(local: List[Instruction], n: Int, mut rows: List[UInt64]) -> Bool:
    """Supports of the forward tableau of a Clifford block, without signs:
    rows[j] = mask of inputs q such that C X_q C^dag or C Z_q C^dag acts on
    output qubit j. Column storage (X[j] / Z[j] = masks over the 2n tableau
    rows) makes every gate O(1). False for n > 32 or a gate it does not know."""
    if n > 32:
        return False
    var X = List[UInt64](length=n, fill=0)
    var Z = List[UInt64](length=n, fill=0)
    for q in range(n):
        X[q] = UInt64(1) << UInt64(q)
        Z[q] = UInt64(1) << UInt64(n + q)
    for g in local:
        var k = g.kind
        var a = g.q0
        var b = g.q1
        if k == Op.I or k == Op.X or k == Op.Y or k == Op.Z or k == Op.BARRIER:
            continue
        if k == Op.H:
            var t = X[a]
            X[a] = Z[a]
            Z[a] = t
        elif k == Op.S or k == Op.SDG:
            Z[a] ^= X[a]
        elif k == Op.SX or k == Op.SXDG:
            X[a] ^= Z[a]
        elif k == Op.CX:
            X[b] ^= X[a]
            Z[a] ^= Z[b]
        elif k == Op.CZ:
            Z[b] ^= X[a]
            Z[a] ^= X[b]
        elif k == Op.CY:
            var xa = X[a]
            var xb = X[b]
            var zb = Z[b]
            Z[a] ^= xb ^ zb
            X[b] = xb ^ xa
            Z[b] = zb ^ xa
        elif k == Op.SWAP:
            var tx = X[a]
            X[a] = X[b]
            X[b] = tx
            var tz = Z[a]
            Z[a] = Z[b]
            Z[b] = tz
        elif k == Op.ISWAP or k == Op.ISWAPDG:
            # exp(+-i pi/4 (XX + YY)), the two terms commute
            _rot_xor(X, Z, a, True, False, b, True, False)
            _rot_xor(X, Z, a, True, True, b, True, True)
        elif k == Op.ECR:
            # RZX(pi/2) then X on the first qubit (support-wise RZX only)
            _rot_xor(X, Z, a, False, True, b, True, False)
        elif k == Op.RX or k == Op.RY or k == Op.RZ or k == Op.P or k == Op.RXX or k == Op.RYY or k == Op.RZZ:
            var qt = _quarter_turns(g)
            if qt < 0:
                return False
            if qt % 2 == 0:
                continue  # identity or a Pauli: supports unchanged
            var px = k == Op.RX or k == Op.RY or k == Op.RXX or k == Op.RYY
            var pz = k == Op.RZ or k == Op.P or k == Op.RY or k == Op.RZZ or k == Op.RYY
            if op_arity(k) == 2:
                _rot_xor(X, Z, a, px, pz, b, px, pz)
            else:
                _rot_xor(X, Z, a, px, pz, -1, False, False)
        else:
            return False
    var low: UInt64 = (UInt64(1) << UInt64(n)) - 1
    for j in range(n):
        var m = X[j] | Z[j]
        rows[j] = (m | (m >> UInt64(n))) & low
    return True


def _consider(kind: Int, local: List[Instruction], cand: List[Instruction], n: Int, adj: List[Bool],
              gs: GateSet, cost_only: Bool, method: Int, mut res: _BlockResult, mut best: List[Instruction]):
    if len(cand) == 0:
        return
    res.proposed += 1
    var c1 = score_cost(cand, gs)
    var d1 = score_depth2q(cand, n)
    if not _better(c1, d1, res.cost, res.depth, cost_only):
        return  # would not be used, so it is not verified (only used proposals are)
    var ph = 0.0
    if not verify_candidate(kind, local, cand, n, adj, ph):
        res.rejected += 1
        return
    res.verified += 1
    res.ok = True
    res.cost = c1
    res.depth = d1
    res.phase = ph
    res.winner = method
    best = cand.copy()


def _method_list(opts: SynthOptions, kind: Int, g: SubGraph, sess: PolicySession, use_policy: Bool) -> List[Int]:
    var ms = List[Int]()
    if opts.method == "exact":
        ms.append(_M_EXACT)
    elif opts.method == "astar":
        ms.append(_M_ASTAR)
    elif opts.method == "greedy":
        ms.append(_M_GREEDY)
    elif opts.method == "policy":
        ms.append(_M_POLICY)
    else:
        # block_synth's own auto mode already combines its greedy portfolio
        # with a budgeted exact search for small blocks (exact_supported)
        ms.append(_M_AUTO)
        if use_policy and sess.available(kind, g, opts.model):
            ms.append(_M_POLICY)
    return ms^


def _at_bound(kind: Int, local: List[Instruction], g: SubGraph, gs: GateSet, cost_only: Bool, mut lbc: Int) -> Bool:
    """True when the incumbent block is provably optimal: no candidate can
    be strictly better under the replacement metric. `lbc` = cost bound."""
    var lbd = 0
    _lower_bound(kind, local, g, gs, lbc, lbd)
    return score_cost(local, gs) <= lbc and (cost_only or score_depth2q(local, g.n) <= lbd)


def _search_block(kind: Int, local: List[Instruction], g: SubGraph, adj: List[Bool], gs: GateSet,
                  opts: SynthOptions, seed: Int, sess: PolicySession, use_policy: Bool,
                  mut best: List[Instruction], check_bound: Bool = True, known_lbc: Int = -1) -> _BlockResult:
    var n = g.n
    var cost_only = opts.metric == "cost"
    var res = _BlockResult(False, score_cost(local, gs), score_depth2q(local, n), 0.0, 0, -1, 0, 0, 0, False, False)
    # provably optimal incumbent: no candidate can be strictly better
    var lbc = known_lbc
    if check_bound:
        if _at_bound(kind, local, g, gs, cost_only, lbc):
            res.at_bound = True
            return res
    # an identity block is replaced by nothing (an empty list is "no proposal"
    # for synthesizers, so this case is checked here); only possible when the
    # lower bound is 0 (Pauli: only for all-Clifford blocks, where the bound
    # is the Clifford one)
    if lbc <= 0 and (kind != KIND_PAULI or _all_clifford(local)):
        var ph = 0.0
        var empty = List[Instruction]()
        if verify_candidate(kind, local, empty, n, adj, ph):
            res.ok = True
            res.cost = 0
            res.depth = 0
            res.phase = ph
            res.winner = _M_IDENTITY
            best.clear()
            res.length = 0
            return res
    var ms = _method_list(opts, kind, g, sess, use_policy)
    for k in range(len(ms)):
        try:
            var cand = _propose(kind, local, g, ms[k], opts, seed + 7919 * k, sess)
            _consider(kind, local, cand, n, adj, gs, cost_only, ms[k], res, best)
        except:
            res.error = True
    res.length = len(best)
    return res


# ----------------------------------------------------------------------------
# one sweep
# ----------------------------------------------------------------------------
def _block_local(c: Circuit, b: Block, mut local: List[Instruction]):
    """Ops of `b` relabelled to local qubits (index in the sorted b.qubits)."""
    local.clear()
    for i in b.ops:
        var g = c.ops[i]
        var l = g
        l.q0 = _index_of(b.qubits, g.q0)
        if op_arity(g.kind) == 2:
            l.q1 = _index_of(b.qubits, g.q1)
        local.append(l)


def _index_of(qs: List[Int], q: Int) -> Int:
    var lo = 0
    var hi = len(qs) - 1
    while lo <= hi:
        var mid = (lo + hi) // 2
        if qs[mid] == q:
            return mid
        if qs[mid] < q:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def _block_adj(target: Target, b: Block, physical: Bool) -> List[Bool]:
    """Local adjacency (n*n) of the block's induced coupling subgraph."""
    var n = len(b.qubits)
    var adj = List[Bool](length=n * n, fill=False)
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i * n + j] = (not physical) or target.are_connected(b.qubits[i], b.qubits[j])
    return adj^


def _connected(adj: List[Bool], n: Int) -> Bool:
    var seen = List[Bool](length=n, fill=False)
    var stack = List[Int]()
    stack.append(0)
    seen[0] = True
    var cnt = 1
    while len(stack) > 0:
        var v = stack.pop()
        for u in range(n):
            if adj[v * n + u] and not seen[u]:
                seen[u] = True
                cnt += 1
                stack.append(u)
    return cnt == n


def _block_key(local: List[Instruction], adj: List[Bool], n: Int) -> UInt64:
    """FNV-1a hash of a localized block (ops, parameters, coupling)."""
    var h: UInt64 = 0xCBF29CE484222325
    var pr: UInt64 = 0x100000001B3
    h = (h ^ UInt64(n)) * pr
    for g in local:
        h = (h ^ UInt64(g.kind)) * pr
        h = (h ^ UInt64(g.q0 + 1)) * pr
        h = (h ^ UInt64(g.q1 + 1)) * pr
        h = (h ^ bitcast[DType.uint64](g.p0.value())) * pr
    for i in range(len(adj)):
        if adj[i]:
            h = (h ^ UInt64(i + 7)) * pr
    return h


def _same_block(a: List[Instruction], b: List[Instruction], adja: List[Bool], adjb: List[Bool]) -> Bool:
    if len(a) != len(b) or len(adja) != len(adjb):
        return False
    for i in range(len(a)):
        var x = a[i]
        var y = b[i]
        if x.kind != y.kind or x.q0 != y.q0 or x.q1 != y.q1 or not (x.p0 == y.p0):
            return False
    for i in range(len(adja)):
        if adja[i] != adjb[i]:
            return False
    return True


struct _Memo(Movable):
    """Search results by block content (local ops + coupling), shared by all
    sweeps of one pass: a block that survives a sweep unchanged is not
    searched again. Seeds derive from the content hash, so a block's result
    does not depend on its position, the sweep or the thread count."""

    var keys: Dict[UInt64, List[Int]]
    var locals: List[List[Instruction]]
    var adjs: List[List[Bool]]
    var res: List[_BlockResult]
    var ops: List[List[Instruction]]

    def __init__(out self):
        self.keys = Dict[UInt64, List[Int]]()
        self.locals = List[List[Instruction]]()
        self.adjs = List[List[Bool]]()
        self.res = List[_BlockResult]()
        self.ops = List[List[Instruction]]()

    def find(self, key: UInt64, local: List[Instruction], adj: List[Bool]) -> Int:
        var hit = self.keys.get(key)
        if not hit:
            return -1
        for e in hit.value():
            if _same_block(self.locals[e], local, self.adjs[e], adj):
                return e
        return -1

    def add(mut self, key: UInt64, local: List[Instruction], adj: List[Bool], r: _BlockResult,
            ops: List[Instruction]) -> Int:
        var e = len(self.res)
        self.locals.append(local.copy())
        self.adjs.append(adj.copy())
        self.res.append(r)
        self.ops.append(ops.copy())
        if key in self.keys:
            try:
                self.keys[key].append(e)
            except:
                pass
        else:
            var l = List[Int]()
            l.append(e)
            self.keys[key] = l^
        return e


def _sweep(mut c: Circuit, target: Target, physical: Bool, opts: SynthOptions, from_back: Bool,
           gs: GateSet, mut memo: _Memo, mut sess: PolicySession, mut use_policy: Bool, mut sess_ready: Bool,
           mut stats: SynthStats, mut wins: List[Int], mut skipped_wide: Int, mut errors: Int,
           mut policy_note: String) raises -> Int:
    var order = List[Int]()
    var all_blocks = collect_blocks_ordered(c, opts.kind, opts.min_block, opts.max_qubits, from_back, opts.commute, order)
    # keep blocks with >= 2 two-qubit gates whose induced coupling subgraph is
    # connected (a block with a single entangling gate is already optimal in
    # 2q cost, and cannot be an identity)
    var blocks = List[Block]()
    var adjs = List[List[Bool]]()
    var locals = List[List[Instruction]]()
    var keys = List[UInt64]()
    for b in all_blocks:
        var n2 = 0
        for i in b.ops:
            if op_arity(c.ops[i].kind) == 2:
                n2 += 1
        if n2 < 2 or len(b.qubits) < 2:
            continue
        var adj = _block_adj(target, b, physical)
        if not _connected(adj, len(b.qubits)):
            continue
        var loc = List[Instruction]()
        _block_local(c, b, loc)
        var n = len(b.qubits)
        keys.append(_block_key(loc, adj, n))
        blocks.append(b.copy())
        adjs.append(adj^)
        locals.append(loc^)
    var nb = len(blocks)
    stats.blocks += nb
    if nb == 0:
        return 0
    # memo hits, and the distinct new blocks to search (`todo`)
    var ent = List[Int](length=nb, fill=-1)  # memo entry, once known
    var tix = List[Int](length=nb, fill=-1)  # index into todo
    var todo = List[Int]()
    var tkeys = Dict[UInt64, List[Int]]()
    for j in range(nb):
        ent[j] = memo.find(keys[j], locals[j], adjs[j])
        if ent[j] >= 0:
            continue
        var found = -1
        var hit = tkeys.get(keys[j])
        if hit:
            for t in hit.value():
                if _same_block(locals[todo[t]], locals[j], adjs[todo[t]], adjs[j]):
                    found = t
                    break
        if found >= 0:
            tix[j] = found
            continue
        tix[j] = len(todo)
        if keys[j] in tkeys:
            tkeys[keys[j]].append(len(todo))
        else:
            var l = List[Int]()
            l.append(len(todo))
            tkeys[keys[j]] = l^
        todo.append(j)
    var nt = len(todo)
    # coupling subgraphs of the blocks to search (todo order)
    var subs = List[SubGraph]()
    for j in todo:
        if physical:
            subs.append(SubGraph.induced(target, blocks[j].qubits))
        else:
            subs.append(SubGraph.complete(len(blocks[j].qubits)))
    # phase 1 (parallel, cheap): blocks already at their lower bound need no
    # search, so they also need no model matching
    var bound = List[Bool](length=max(nt, 1), fill=False)
    var bp = bound.unsafe_ptr().as_unsafe_any_origin()
    var lbcs = List[Int](length=max(nt, 1), fill=0)
    var lp = lbcs.unsafe_ptr().as_unsafe_any_origin()
    var cost_only = opts.metric == "cost"
    var nch1 = 1 if nt < 64 else min(nt // 16, 64)

    def bwork(w: Int) {imm subs, imm locals, imm gs, imm opts, imm todo, imm nt, imm nch1, imm bp, imm lp, imm cost_only}:
        var lo = w * nt // nch1
        var hi = (w + 1) * nt // nch1
        for t in range(lo, hi):
            var lbc = 0
            bp[unsafe_offset=t] = _at_bound(opts.kind, locals[todo[t]], subs[t], gs, cost_only, lbc)
            lp[unsafe_offset=t] = lbc

    if nt > 0:
        if nch1 == 1:
            bwork(0)
        else:
            parallelize(bwork, nch1)
    _ = len(bound)
    _ = len(lbcs)
    var nsearch = 0
    for t in range(nt):
        if not bound[t]:
            nsearch += 1
    # learned proposer: the session (model index) is opened on first use and
    # the models matching these blocks are loaded serially, so the parallel
    # workers only use read-only methods
    if nsearch > 0 and use_policy and not sess_ready:
        sess = PolicySession(model_search_dirs(opts.model_dir))
        sess_ready = True
    if nsearch > 0 and use_policy:
        try:
            for t in range(nt):
                if bound[t]:
                    continue
                var mm = sess.match(opts.kind, subs[t], opts.model)
                if mm.ok:
                    _ = sess.load(String(mm.dir))
        except e:
            var explicit_model = opts.method == "policy" or (opts.model != "auto" and opts.model != "off")
            if explicit_model:
                raise Error(String(e))
            policy_note = " policy_disabled(" + String(e).split(".")[0] + ")"
            use_policy = False
    # search the new blocks in parallel (fixed output slots per block)
    var slot_start = List[Int](length=nt + 1, fill=0)
    var seeds = List[Int](length=nt, fill=0)
    for t in range(nt):
        var j = todo[t]
        var cap = 8 * len(blocks[j].ops) + 16 * len(blocks[j].qubits) + 64
        slot_start[t + 1] = slot_start[t] + cap
        var r = Rng(UInt64(opts.seed) ^ keys[j])
        seeds[t] = Int(r.next_u64() >> 34)
    var slots = List[Instruction](length=max(slot_start[nt], 1), fill=Instruction.make(Op.I, 0))
    var results = List[_BlockResult](length=max(nt, 1), fill=_BlockResult(False, 0, 0, 0.0, 0, -1, 0, 0, 0, False, False))
    var sp = slots.unsafe_ptr().as_unsafe_any_origin()
    var rp = results.unsafe_ptr().as_unsafe_any_origin()
    var nchunks = 1 if nt < 4 else min(nt, 64)
    var up = use_policy

    def work(w: Int) {imm adjs, imm subs, imm locals, imm gs, imm opts, imm seeds, imm sess, imm up, imm todo, imm slot_start, imm nt, imm nchunks, imm sp, imm rp, imm bp, imm lp}:
        var lo = w * nt // nchunks
        var hi = (w + 1) * nt // nchunks
        for t in range(lo, hi):
            var j = todo[t]
            if bp[unsafe_offset=t]:
                rp[unsafe_offset=t] = _BlockResult(False, 0, 0, 0.0, 0, -1, 0, 0, 0, False, True)
                continue
            var best = List[Instruction]()
            var res = _search_block(opts.kind, locals[j], subs[t], adjs[j], gs, opts, seeds[t], sess, up, best,
                                    False, lp[unsafe_offset=t])
            var cap = slot_start[t + 1] - slot_start[t]
            if res.ok and len(best) > cap:
                res.ok = False  # cannot store (never expected for a cheaper block)
            if res.ok:
                for e in range(len(best)):
                    sp[unsafe_offset=slot_start[t] + e] = best[e]
            rp[unsafe_offset=t] = res

    if nt > 0:
        if nchunks == 1:
            work(0)
        else:
            parallelize(work, nchunks)
    _ = len(slots)
    _ = len(results)
    _ = len(bound)
    _ = len(lbcs)
    _ = len(adjs)
    _ = len(subs)
    _ = len(locals)
    _ = len(todo)
    _ = len(seeds)
    _ = len(slot_start)
    # new results: errors, strict mode, then into the memo (block order)
    var explicit = opts.method != "auto" or (opts.model != "auto" and opts.model != "off")
    var tent = List[Int](length=nt, fill=-1)
    for t in range(nt):
        var j = todo[t]
        var r = results[t]
        stats.proposed += r.proposed
        stats.verified += r.verified
        stats.rejected_not_equal += r.rejected
        if r.at_bound:
            wins[_M_BOUND] += 1
        if r.error:
            errors += 1
            if explicit:
                # re-run serially to surface the synthesizer's error message;
                # a block too wide for an explicit exact / astar method is
                # skipped (reported as skipped_too_wide) unless strict=1
                var ms = _method_list(opts, opts.kind, subs[t], sess, use_policy)
                for k in range(len(ms)):
                    try:
                        _ = _propose(opts.kind, locals[j], subs[t], ms[k], opts, seeds[t] + 7919 * k, sess)
                    except e:
                        var msg = String(e)
                        if msg.startswith("QM-E618") and not opts.strict:
                            skipped_wide += 1
                            errors -= 1
                        else:
                            raise Error(msg)
        if r.rejected > 0 and opts.strict:
            raise Error("QM-E619: a proposed " + _kind_name(opts.kind) + " synthesis for qubits " + String(blocks[j].qubits)
                        + " is not equivalent to the original block (rejected by the exact check). Hint: report the"
                        " circuit; the pass is safe without strict=1 (bad proposals are dropped).")
        var ops = List[Instruction]()
        if r.ok:
            for e in range(r.length):
                ops.append(slots[slot_start[t] + e])
        tent[t] = memo.add(keys[j], locals[j], adjs[j], r, ops)
    # replacements
    var replaced = 0
    var saved = 0
    var blk_of = List[Int](length=len(c.ops), fill=-1)
    for j in range(nb):
        if ent[j] < 0:
            ent[j] = tent[tix[j]]
        var r = memo.res[ent[j]]
        if not r.ok:
            continue
        var c0 = score_cost(locals[j], gs)
        var d0 = score_depth2q(locals[j], len(blocks[j].qubits))
        saved += c0 - r.cost
        stats.saved_depth += d0 - r.depth
        replaced += 1
        if r.winner >= 0:
            wins[r.winner] += 1
        for i in blocks[j].ops:
            blk_of[i] = j
    if replaced == 0:
        return 0
    # splice in the collection order
    var out = List[Instruction](capacity=len(c.ops))
    var phase = 0.0
    for i in order:
        var j = blk_of[i]
        if j < 0:
            out.append(c.ops[i])
            continue
        if i != blocks[j].ops[0]:
            continue
        var e = ent[j]
        var qs = blocks[j].qubits.copy()
        var rops = memo.ops[e].copy()
        for g0 in rops:
            var g = g0
            g.q0 = qs[g.q0]
            if op_arity(g.kind) == 2:
                g.q1 = qs[g.q1]
            out.append(g)
        phase -= memo.res[e].phase
    c.ops = out^
    c.global_phase += phase
    stats.replaced += replaced
    stats.saved_cost += saved
    return saved


def _kind_name(kind: Int) -> String:
    if kind == KIND_CLIFFORD:
        return "Clifford"
    if kind == KIND_LINEAR:
        return "linear-function"
    if kind == KIND_PERM:
        return "permutation"
    return "Pauli-network"


def _validate(c: Circuit, target: Target, physical: Bool, opts: SynthOptions) raises:
    if opts.kind < KIND_CLIFFORD or opts.kind > KIND_PAULI:
        raise Error("QM-E616: unknown block kind " + String(opts.kind) + ". Hint: 0 = Clifford, 1 = linear function, 2 = permutation, 3 = Pauli network.")
    var m = opts.method
    if m != "auto" and m != "exact" and m != "astar" and m != "greedy" and m != "policy":
        raise Error("QM-E616: unknown synthesis method '" + m + "'. Hint: method=auto|exact|astar|greedy|policy.")
    if opts.metric != "cost_depth" and opts.metric != "cost":
        raise Error("QM-E616: unknown replacement metric '" + opts.metric + "'. Hint: metric=cost_depth|cost.")
    if opts.direction != "forward" and opts.direction != "backward" and opts.direction != "both":
        raise Error("QM-E616: unknown collection direction '" + opts.direction + "'. Hint: direction=forward|backward|both.")
    if opts.model.byte_length() == 0:
        raise Error("QM-E616: empty model name. Hint: model=auto|off|<model id>.")
    var lim = MAX_W_PERM
    if opts.kind == KIND_CLIFFORD:
        lim = MAX_W_CLIFFORD
    elif opts.kind == KIND_PAULI:
        lim = MAX_W_PAULI
    elif opts.kind == KIND_LINEAR:
        lim = MAX_W_LINEAR
    if opts.max_qubits > lim or opts.max_qubits < 1:
        raise Error("QM-E618: max_qubits=" + String(opts.max_qubits) + " is outside 1.." + String(lim) + " for "
                    + _kind_name(opts.kind) + " blocks (the exact verifier's limit). Hint: lower max_qubits (default "
                    + String(SynthOptions.default(opts.kind).max_qubits) + ").")
    if physical and c.num_qubits > target.num_qubits:
        raise Error("QM-E605: circuit has " + String(c.num_qubits) + " qubits but target '" + target.name + "' has "
                    + String(target.num_qubits) + ". Hint: route the circuit first or pass physical=False.")


def _in_basis(c: Circuit, gs: GateSet) -> Bool:
    if gs.any:
        return False
    for g in c.ops:
        if is_plain_gate(g) and op_arity(g.kind) >= 1 and not gs.has(g.kind):
            return False
    return True


def resynthesize_blocks(mut c: Circuit, target: Target, physical: Bool, opts: SynthOptions, mut stats: SynthStats) raises -> Int:
    """Collect blocks of `opts.kind` and replace each one by a verified,
    strictly cheaper synthesis (see the module docstring). Returns the 2q cost
    saved in target CX units.

    physical=True: wires are physical qubits of `target` (routed circuit) and
    candidates must use the block's induced coupling subgraph (blocks whose
    subgraph is disconnected are skipped); physical=False: all-to-all."""
    _validate(c, target, physical, opts)
    var gs = GateSet(target.basis)
    var was_in_basis = _in_basis(c, gs)
    # learned proposer: one model session per pass, preloaded so that workers
    # only use its read-only methods
    var use_policy = opts.method == "policy" or (opts.method == "auto" and opts.model != "off")
    var dirs = List[String]()
    var memo = _Memo()
    var sess = PolicySession(dirs)  # empty until a block needs it
    var sess_ready = False
    var policy_note = String()
    var wins = List[Int](length=_NUM_M, fill=0)
    var skipped_wide = 0
    var errors = 0
    var total = 0
    var sweeps = 0
    var reps = max(opts.reps, 1)
    for _ in range(reps):
        var gained = 0
        var before = stats.replaced
        if opts.direction == "forward" or opts.direction == "both":
            gained += _sweep(c, target, physical, opts, False, gs, memo, sess, use_policy, sess_ready, stats, wins, skipped_wide, errors, policy_note)
            sweeps += 1
        if opts.direction == "backward" or opts.direction == "both":
            gained += _sweep(c, target, physical, opts, True, gs, memo, sess, use_policy, sess_ready, stats, wins, skipped_wide, errors, policy_note)
            sweeps += 1
        total += gained
        if stats.replaced == before:
            break
    if stats.replaced > 0:
        _ = cancel_inverses(c, GateSet(), 32)
        if was_in_basis:
            var dropped = False
            _ = translate_to_basis(c, gs, dropped)
    var note = String("sweeps=") + String(sweeps)
    note += " wins(exact=" + String(wins[_M_EXACT]) + ",astar=" + String(wins[_M_ASTAR])
    note += ",greedy=" + String(wins[_M_GREEDY]) + ",auto=" + String(wins[_M_AUTO]) + ",policy=" + String(wins[_M_POLICY])
    note += ",identity=" + String(wins[_M_IDENTITY]) + ")"
    if wins[_M_BOUND] > 0:
        note += " at_lower_bound=" + String(wins[_M_BOUND])
    if skipped_wide > 0:
        note += " skipped_too_wide=" + String(skipped_wide)
    if errors > 0:
        note += " synth_errors=" + String(errors)
    note += policy_note
    stats.note = note
    _ = len(dirs)
    return total
