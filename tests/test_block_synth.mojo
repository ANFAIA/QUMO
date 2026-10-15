"""Tests for qumo/compiler/blockalg.mojo and qumo/compiler/block_synth.mojo.

Every synthesized block is checked exactly (permutation / GF(2) matrix /
signed tableau equality, dense unitary up to global phase for Clifford and
Pauli networks on n <= 6) and every 2q gate must lie on an edge of the
subgraph. Exact methods are cross-checked against brute-force BFS."""

from std.math import pi
from std.time import perf_counter_ns
from std.testing import assert_equal, assert_true

from qumo.ir.circuit import Instruction
from qumo.ir.gates import Op, op_arity
from qumo.sim.rng import Rng
from qumo.compiler.target import Target
from qumo.compiler.common import GateSet
from qumo.compiler.blockalg import (
    BlockKind, SubGraph, Perm, GF2Mat, SympTab, PauliRot, PauliNet, emit_rotation,
    ops_cost, ops_cx_count, ops_depth2q, ops_on_edges, local_ops, global_ops, inverse_ops,
    apply_ops_dense, dense_equiv, phase_offset, g1, g2,
)
from qumo.compiler.block_synth import (
    synth_perm, synth_linear, synth_clifford, synth_pauli_network, exact_supported, synth_local,
)


# ----------------------------------------------------------------------------
# graphs
# ----------------------------------------------------------------------------
def line(n: Int) -> SubGraph:
    var e = List[Int]()
    for i in range(n - 1):
        e.append(i); e.append(i + 1)
    return SubGraph(n, e)


def ring(n: Int) -> SubGraph:
    var e = List[Int]()
    for i in range(n):
        e.append(i); e.append((i + 1) % n)
    return SubGraph(n, e)


def grid(r: Int, c: Int, king: Bool = False) -> SubGraph:
    var e = List[Int]()
    for i in range(r):
        for j in range(c):
            var v = i * c + j
            if j + 1 < c:
                e.append(v); e.append(v + 1)
            if i + 1 < r:
                e.append(v); e.append(v + c)
            if king and i + 1 < r and j + 1 < c:
                e.append(v); e.append(v + c + 1)
            if king and i + 1 < r and j > 0:
                e.append(v); e.append(v + c - 1)
    return SubGraph(r * c, e)


def hh7() -> SubGraph:
    """7-qubit heavy-hex 'H' fragment (Falcon r5.11H coupling)."""
    var e: List[Int] = [0, 1, 1, 2, 1, 3, 3, 5, 4, 5, 5, 6]
    return SubGraph(7, e)


def hh12() -> SubGraph:
    """10-ring with two spokes (12 qubits, heavy-hex cell plus pendants)."""
    var e = List[Int]()
    for i in range(10):
        e.append(i); e.append((i + 1) % 10)
    e.append(0); e.append(10)
    e.append(5); e.append(11)
    return SubGraph(12, e)


def graphs() -> List[SubGraph]:
    var gs = List[SubGraph]()
    for n in range(4, 9):
        gs.append(line(n))
    gs.append(ring(6))
    gs.append(grid(2, 3))
    gs.append(grid(3, 3))
    gs.append(hh7())
    gs.append(hh12())
    gs.append(grid(2, 3, True))
    for n in range(4, 7):
        gs.append(SubGraph.complete(n))
    return gs^


# ----------------------------------------------------------------------------
# random problems
# ----------------------------------------------------------------------------
def rand_perm(n: Int, mut rng: Rng) -> Perm:
    var p = List[Int]()
    for i in range(n):
        p.append(i)
    var i = n - 1
    while i > 0:
        var j = rng.below(i + 1)
        var t = p[i]; p[i] = p[j]; p[j] = t
        i -= 1
    return Perm(p)


def rand_lf_ops(n: Int, depth: Int, mut rng: Rng) -> List[Instruction]:
    var ops = List[Instruction]()
    for _ in range(depth):
        var a = rng.below(n)
        var b = rng.below(n - 1)
        if b >= a:
            b += 1
        ops.append(g2(Op.CX, a, b))
    return ops^


def rand_clifford_ops(n: Int, depth: Int, mut rng: Rng, exotic: Bool = False) -> List[Instruction]:
    var ops = List[Instruction]()
    var one: List[Int] = [Op.H, Op.S, Op.SDG, Op.SX, Op.SXDG, Op.X, Op.Y, Op.Z]
    var two: List[Int] = [Op.CX, Op.CZ, Op.SWAP]
    if exotic:
        one.append(Op.RX); one.append(Op.RY); one.append(Op.RZ)
        two.append(Op.CY); two.append(Op.ISWAP); two.append(Op.ISWAPDG); two.append(Op.ECR)
        two.append(Op.RZZ); two.append(Op.RXX); two.append(Op.RYY)
    for _ in range(depth):
        var th = Float64(rng.below(7) - 3) * (pi / 2)
        if n >= 2 and rng.uniform() < 0.45:
            var a = rng.below(n)
            var b = rng.below(n - 1)
            if b >= a:
                b += 1
            ops.append(Instruction.make(two[rng.below(len(two))], a, b, p0=th))
        else:
            ops.append(Instruction.make(one[rng.below(len(one))], rng.below(n), p0=th))
    return ops^


def rand_pauli_ops(n: Int, depth: Int, mut rng: Rng) -> List[Instruction]:
    var ops = List[Instruction]()
    var one: List[Int] = [Op.H, Op.S, Op.SDG, Op.SX, Op.X, Op.Z]
    var rot1: List[Int] = [Op.RX, Op.RY, Op.RZ]
    var rot2: List[Int] = [Op.RZZ, Op.RXX, Op.RYY]
    for _ in range(depth):
        var u = rng.uniform()
        var th = (rng.uniform() - 0.5) * 3.0
        var a = rng.below(n)
        var b = rng.below(n - 1)
        if b >= a:
            b += 1
        if u < 0.3:
            ops.append(g2(Op.CX, a, b))
        elif u < 0.45:
            ops.append(g1(one[rng.below(len(one))], a))
        elif u < 0.75:
            ops.append(Instruction.make(rot1[rng.below(3)], a, p0=th))
        else:
            ops.append(Instruction.make(rot2[rng.below(3)], a, b, p0=th))
    return ops^


# ----------------------------------------------------------------------------
# brute force references
# ----------------------------------------------------------------------------
def perm_key(arr: List[Int]) -> Int:
    var k = 0
    for i in range(len(arr)):
        k = k * 16 + arr[i]
    return k


def perm_bfs_dist(g: SubGraph) raises -> Dict[Int, Int]:
    """SWAP distance from the identity arrangement to every arrangement."""
    var n = g.n
    var d = Dict[Int, Int]()
    var q = List[List[Int]]()
    var start = List[Int]()
    for i in range(n):
        start.append(i)
    d[perm_key(start)] = 0
    q.append(start^)
    var head = 0
    while head < len(q):
        var cur = q[head].copy()
        head += 1
        var dc = d[perm_key(cur)]
        for e in range(g.num_edges()):
            var a = g.edges[2 * e]
            var b = g.edges[2 * e + 1]
            var nx = cur.copy()
            var t = nx[a]; nx[a] = nx[b]; nx[b] = t
            var k = perm_key(nx)
            if k not in d:
                d[k] = dc + 1
                q.append(nx^)
    return d^


def arrangement_of(p: Perm) -> List[Int]:
    """Token placement after the permutation: vertex p[i] holds token i."""
    var arr = List[Int](length=p.size(), fill=0)
    for i in range(p.size()):
        arr[p.p[i]] = i
    return arr^


def lf_key(m: GF2Mat) -> Int:
    var k = 0
    for i in range(m.n):
        k |= Int(m.rows[i]) << (m.n * i)
    return k


def lf_bfs_dist(g: SubGraph) raises -> Dict[Int, Int]:
    var n = g.n
    var d = Dict[Int, Int]()
    var q = List[Int]()
    var s = lf_key(GF2Mat.identity(n))
    d[s] = 0
    q.append(s)
    var head = 0
    var mask = (1 << n) - 1
    while head < len(q):
        var k = q[head]
        head += 1
        var dc = d[k]
        for e in range(g.num_edges()):
            for dirn in range(2):
                var c = g.edges[2 * e + dirn]
                var t = g.edges[2 * e + 1 - dirn]
                var rc = (k >> (n * c)) & mask
                var nk = k ^ (rc << (n * t))
                if nk not in d:
                    d[nk] = dc + 1
                    q.append(nk)
    return d^


def cliff_key(t: SympTab) -> Int:
    var k = 0
    for i in range(2 * t.n):
        k = (k << (2 * t.n)) | Int(t.x[i]) | (Int(t.z[i]) << t.n)
    return k


def clifford_bfs_dist(g: SubGraph) raises -> Dict[Int, Int]:
    """0-1 BFS over phase-free Cliffords: local H / S free, CX on edges costs 1."""
    var n = g.n
    var d = Dict[Int, Int]()
    var frontier = List[SympTab]()
    var id = SympTab(n)
    d[cliff_key(id)] = 0
    frontier.append(id^)
    var level = 0
    while len(frontier) > 0:
        # close the level under free local gates
        var head = 0
        while head < len(frontier):
            var t = frontier[head].copy()
            head += 1
            for q in range(n):
                for k in [Op.H, Op.S]:
                    var u = t.copy()
                    u.apply(g1(k, q))
                    var key = cliff_key(u)
                    if key not in d:
                        d[key] = level
                        frontier.append(u^)
        var nxt = List[SympTab]()
        for t in frontier:
            for e in range(g.num_edges()):
                for dirn in range(2):
                    var u = t.copy()
                    u.apply(g2(Op.CX, g.edges[2 * e + dirn], g.edges[2 * e + 1 - dirn]))
                    var key = cliff_key(u)
                    if key not in d:
                        d[key] = level + 1
                        nxt.append(u^)
        frontier = nxt^
        level += 1
    return d^


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
def check_edges(ops: List[Instruction], g: SubGraph) raises:
    assert_true(ops_on_edges(ops, g), "2q gate off the coupling graph")
    for o in ops:
        var k = o.kind
        assert_true(k == Op.H or k == Op.S or k == Op.SDG or k == Op.SX or k == Op.SXDG or k == Op.X or k == Op.Y or k == Op.Z or k == Op.CX or k == Op.CZ or k == Op.SWAP or k == Op.RX or k == Op.RY or k == Op.RZ, "gate outside the output set")


def pauli_vec_ops(x: UInt64, z: UInt64, n: Int) -> List[Instruction]:
    var ops = List[Instruction]()
    for j in range(n):
        var xb = ((x >> UInt64(j)) & 1) != 0
        var zb = ((z >> UInt64(j)) & 1) != 0
        if xb and zb:
            ops.append(g1(Op.Y, j))
        elif xb:
            ops.append(g1(Op.X, j))
        elif zb:
            ops.append(g1(Op.Z, j))
    return ops^


def check_tableau_dense(ops: List[Instruction], n: Int) raises:
    """Every row of SympTab.from_ops(ops) satisfies U P = R U numerically."""
    var t = SympTab.from_ops(ops, n)
    var dim = 1 << n
    for row in range(2 * n):
        var q = row % n
        var p_ops = List[Instruction]()
        p_ops.append(g1(Op.X if row < n else Op.Z, q))
        var r_ops = pauli_vec_ops(t.x[row], t.z[row], n)
        var sgn = -1.0 if t.r[row] != 0 else 1.0
        for col in [0, dim - 1, dim // 2]:
            var ar = List[Float64](length=dim, fill=0.0)
            var ai = List[Float64](length=dim, fill=0.0)
            ar[col] = 1.0
            var br = ar.copy()
            var bi = ai.copy()
            apply_ops_dense(p_ops, n, ar, ai)
            apply_ops_dense(ops, n, ar, ai)
            apply_ops_dense(ops, n, br, bi)
            apply_ops_dense(r_ops, n, br, bi)
            for i in range(dim):
                assert_true(abs(ar[i] - sgn * br[i]) < 1e-9 and abs(ai[i] - sgn * bi[i]) < 1e-9, "tableau row disagrees with the dense unitary")


def naive_pauli_net_ops(pn: PauliNet, src: List[Instruction]) raises -> List[Instruction]:
    """Reference circuit for a PauliNet: one CX-ladder gadget per rotation on
    the complete graph, then the final Clifford (all-to-all greedy)."""
    var n = pn.n
    var ops = List[Instruction]()
    for rot in pn.rots:
        var qs = List[Int]()
        var pre = List[Instruction]()
        for j in range(n):
            var xb = ((rot.x >> UInt64(j)) & 1) != 0
            var zb = ((rot.z >> UInt64(j)) & 1) != 0
            if not xb and not zb:
                continue
            qs.append(j)
            if xb and zb:
                pre.append(g1(Op.SX, j))  # Y -> Z
            elif xb:
                pre.append(g1(Op.H, j))
        for o in pre:
            ops.append(o)
        for i in range(len(qs) - 1):
            ops.append(g2(Op.CX, qs[i], qs[i + 1]))
        ops.append(emit_rotation(rot, src, qs[len(qs) - 1], 2, 1))
        var i = len(qs) - 2
        while i >= 0:
            ops.append(g2(Op.CX, qs[i], qs[i + 1]))
            i -= 1
        for o in inverse_ops(pre):
            ops.append(o)
    var fin = synth_clifford(pn.final, SubGraph.complete(n), "greedy", 0, 1)
    for o in fin:
        ops.append(o)
    return ops^


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def test_algebra() raises:
    var rng = Rng(11)
    # tableau vs dense, all supported Clifford gates
    for trial in range(40):
        var n = 1 + trial % 3
        var ops = rand_clifford_ops(n, 12, rng, True)
        check_tableau_dense(ops, n)
        var t = SympTab.from_ops(ops, n)
        # inverse / compose / prepend consistency
        assert_true(t.compose(t.inverse()).is_identity(), "C^-1 C != I")
        assert_true(t.inverse().compose(t).is_identity(), "C C^-1 != I")
        var both = ops.copy()
        for o in inverse_ops(ops):
            both.append(o)
        assert_true(SympTab.from_ops(both, n).is_identity(), "ops . ops^dag != I")
        var tp = SympTab(n)
        var i = len(ops) - 1
        while i >= 0:
            tp.prepend(ops[i])
            i -= 1
        assert_true(tp == t, "prepend disagrees with apply")
        assert_true(SympTab.from_ops(inverse_ops(ops), n) == t.inverse(), "inverse tableau")
    # GF2 / Perm / linear tableau
    for trial in range(20):
        var n = 3 + trial % 4
        var ops = rand_lf_ops(n, 15, rng)
        var m = GF2Mat.from_ops(ops, n)
        assert_true(m.compose(m.inverse()).is_identity(), "GF2 inverse")
        assert_true(SympTab.from_linear(m) == SympTab.from_ops(ops, n), "from_linear")
        assert_true(SympTab.from_ops(ops, n).linear_part() == m, "linear_part")
        var p = rand_perm(n, rng)
        var sw = synth_perm(p, SubGraph.complete(n), "greedy", 0, 1)
        assert_true(Perm.from_ops(sw, n) == p, "perm greedy on K_n")
        assert_true(GF2Mat.from_ops(sw, n) == GF2Mat.from_perm(p), "from_perm")
        assert_true(GF2Mat.from_perm(p).to_perm() == p, "to_perm")
        assert_true(p.compose(p.inverse()).is_identity(), "perm inverse")
    # Perm convention: SWAP(0,1) then SWAP(1,2): state of 0 ends on 2
    var sw2 = List[Instruction]()
    sw2.append(g2(Op.SWAP, 0, 1)); sw2.append(g2(Op.SWAP, 1, 2))
    var pp = Perm.from_ops(sw2, 3)
    assert_equal(pp.p[0], 2)
    assert_equal(pp.p[1], 0)
    assert_equal(pp.p[2], 1)
    # QM-E701 / QM-E618
    var bad = List[Instruction]()
    bad.append(g1(Op.T, 0))
    var raised = False
    try:
        _ = SympTab.from_ops(bad, 1)
    except e:
        raised = String(e).startswith("QM-E701")
    assert_true(raised, "T in a tableau must raise QM-E701")
    raised = False
    try:
        _ = Perm.from_ops(bad, 1)
    except e:
        raised = String(e).startswith("QM-E618")
    assert_true(raised, "T in a permutation must raise QM-E618")
    raised = False
    try:
        _ = synth_perm(Perm.identity(3), line(3), "magic", 0, 1)
    except e:
        raised = String(e).startswith("QM-E616")
    assert_true(raised, "unknown method must raise QM-E616")
    raised = False
    try:
        _ = synth_linear(GF2Mat.identity(9), grid(3, 3), "exact", 0, 1)
    except e:
        raised = String(e).startswith("QM-E618")
    assert_true(raised, "exact LF on 9 qubits must raise QM-E618")
    # SubGraph.induced / local_ops / costs
    var tg = Target.line(6)
    var phys: List[Int] = [5, 3, 4]
    var sg = SubGraph.induced(tg, phys)
    assert_true(sg.has_edge(0, 2) and sg.has_edge(1, 2) and not sg.has_edge(0, 1), "induced edges")
    assert_equal(sg.distance(0, 1), 2)
    var wops = List[Instruction]()
    wops.append(g2(Op.CX, 5, 4)); wops.append(g2(Op.SWAP, 3, 4))
    var lops = local_ops(wops, phys)
    assert_equal(lops[0].q0, 0)
    assert_equal(lops[0].q1, 2)
    var back = global_ops(lops, phys)
    assert_equal(back[1].q0, 3)
    assert_equal(ops_cost(wops, GateSet()), 4)
    var swap_native: List[Int] = [Op.SWAP, Op.CX]
    assert_equal(ops_cost(wops, GateSet(swap_native)), 2)
    assert_equal(ops_depth2q(lops, 3), 2)
    print("  algebra ok")


def test_pauli_net_algebra() raises:
    var rng = Rng(5)
    for trial in range(12):
        var n = 2 + trial % 3
        var ops = rand_pauli_ops(n, 14, rng)
        var pn = PauliNet.from_ops(ops, n)
        var refc = naive_pauli_net_ops(pn, ops)
        var ph = 0.0
        assert_true(dense_equiv(ops, refc, n, ph), "PauliNet.from_ops semantics")
    print("  pauli-net algebra ok")


def test_perm() raises:
    var gs = graphs()
    var rng = Rng(21)
    var methods: List[String] = ["greedy", "auto", "astar", "exact"]
    for gi in range(len(gs)):
        var g = gs[gi].copy()
        for _ in range(4):
            var p = rand_perm(g.n, rng)
            for m in methods:
                if (m == "exact" and not exact_supported(BlockKind.PERM, g.n)) or (m == "astar" and g.n > 9):
                    continue
                var ops = synth_perm(p, g, m, 0, 3)
                if len(ops) == 0:
                    assert_true(p.is_identity() or m == "exact" or m == "astar", "no proposal from " + m)
                    continue
                check_edges(ops, g)
                assert_true(Perm.from_ops(ops, g.n) == p, "perm not exact (" + m + ")")
    # optimality vs brute force BFS (n <= 6) and greedy / exact ratio (n <= 8)
    var small = List[SubGraph]()
    small.append(line(5)); small.append(line(6)); small.append(ring(6)); small.append(grid(2, 3)); small.append(SubGraph.complete(5))
    for g in small:
        var d = perm_bfs_dist(g)
        for _ in range(15):
            var p = rand_perm(g.n, rng)
            var ex = synth_perm(p, g, "exact", 0, 1)
            var opt = d[perm_key(arrangement_of(p))]
            assert_equal(len(ex), opt)
    var tot_g = 0
    var tot_e = 0
    var cases = List[SubGraph]()
    cases.append(line(8)); cases.append(grid(2, 4)); cases.append(hh7()); cases.append(ring(8)); cases.append(grid(2, 3, True))
    for g in cases:
        for _ in range(10):
            var p = rand_perm(g.n, rng)
            var gr = synth_perm(p, g, "greedy", 0, 1)
            var ex = synth_perm(p, g, "exact", 0, 1)
            assert_true(len(ex) > 0 or p.is_identity(), "exact perm over budget on n <= 8")
            tot_g += len(gr)
            tot_e += len(ex)
    var ratio = Float64(tot_g) / Float64(max(tot_e, 1))
    print("  perm: greedy/exact SWAP ratio (n <= 8, 50 instances) =", ratio)
    assert_true(ratio <= 1.5, "greedy permutation synthesis too far from optimal")
    print("  perm ok")


def route_baseline(m: GF2Mat, g: SubGraph) raises -> Int:
    """Topology-unaware synthesis (greedy on K_n) + naive SWAP routing + final
    layout restoration; returns the CX-unit cost (verified)."""
    var n = g.n
    var lin = synth_linear(m, SubGraph.complete(n), "greedy", 0, 1)
    var pos = List[Int]()  # logical -> physical
    var at = List[Int]()  # physical -> logical
    for i in range(n):
        pos.append(i)
        at.append(i)
    var out = List[Instruction]()
    for o in lin:
        var pc = pos[o.q0]
        var pt = pos[o.q1]
        while not g.has_edge(pc, pt):
            # step the control one hop towards the target
            var nxt = -1
            for u in g.neighbors(pc):
                if g.distance(u, pt) < g.distance(pc, pt):
                    nxt = u
                    break
            out.append(g2(Op.SWAP, pc, nxt))
            var la = at[pc]
            var lb = at[nxt]
            at[pc] = lb; at[nxt] = la
            pos[la] = nxt; pos[lb] = pc
            pc = nxt
        out.append(g2(Op.CX, pc, pt))
    # restore the layout: logical i sits on pos[i] and must return to i
    var back = List[Int](length=n, fill=0)
    for i in range(n):
        back[pos[i]] = i
    var sw = synth_perm(Perm(back), g, "auto", 0, 1)
    for o in sw:
        out.append(o)
    assert_true(GF2Mat.from_ops(out, n) == m, "baseline routing not exact")
    return ops_cx_count(out)


def test_linear() raises:
    var gs = graphs()
    var rng = Rng(31)
    var methods: List[String] = ["greedy", "auto", "astar", "exact"]
    for gi in range(len(gs)):
        var g = gs[gi].copy()
        for _ in range(3):
            var m = GF2Mat.from_ops(rand_lf_ops(g.n, 3 * g.n, rng), g.n)
            for meth in methods:
                if (meth == "exact" and not exact_supported(BlockKind.LINEAR, g.n)) or (meth == "astar" and g.n > 4):
                    continue
                var ops = synth_linear(m, g, meth, 50000, 3)
                if len(ops) == 0:
                    assert_true(m.is_identity() or meth == "exact" or meth == "astar", "no LF proposal from " + meth)
                    continue
                check_edges(ops, g)
                assert_true(GF2Mat.from_ops(ops, g.n) == m, "LF not exact (" + meth + ")")
    # optimality vs brute force (n <= 4)
    var small = List[SubGraph]()
    small.append(line(4)); small.append(ring(4)); small.append(SubGraph.complete(4)); small.append(line(3))
    for g in small:
        var d = lf_bfs_dist(g)
        for _ in range(12):
            var m = GF2Mat.from_ops(rand_lf_ops(g.n, 12, rng), g.n)
            var ex = synth_linear(m, g, "exact", 0, 1)
            assert_equal(len(ex), d[lf_key(m)])
            var a = synth_linear(m, g, "astar", 0, 1)
            assert_equal(len(a), d[lf_key(m)])
            var au = synth_linear(m, g, "auto", 0, 1)
            assert_equal(len(au), d[lf_key(m)])
    # greedy vs topology-unaware synthesis + SWAP routing, 6-8 qubits
    var cases = List[SubGraph]()
    cases.append(line(6)); cases.append(line(8)); cases.append(grid(2, 3)); cases.append(grid(2, 4)); cases.append(hh7())
    var tot_ai = 0
    var tot_base = 0
    for g in cases:
        for _ in range(8):
            var m = GF2Mat.from_ops(rand_lf_ops(g.n, 40, rng), g.n)
            var ops = synth_linear(m, g, "auto", 0, 1)
            assert_true(GF2Mat.from_ops(ops, g.n) == m, "LF auto not exact")
            tot_ai += ops_cx_count(ops)
            tot_base += route_baseline(m, g)
    var red = 1.0 - Float64(tot_ai) / Float64(tot_base)
    print("  linear: topology-aware CX", tot_ai, "vs synth+route", tot_base, "-> reduction", red)
    assert_true(red >= 0.30, "topology-aware LF synthesis must save >= 30% CX")
    print("  linear ok")


def test_clifford() raises:
    var gs = graphs()
    var rng = Rng(41)
    for gi in range(len(gs)):
        var g = gs[gi].copy()
        for trial in range(3):
            var ops = rand_clifford_ops(g.n, 8 * g.n, rng, trial == 0)
            var t = SympTab.from_ops(ops, g.n)
            for meth in ["greedy", "auto"]:
                var out = synth_clifford(t, g, meth, 0, 5)
                check_edges(out, g)
                assert_true(SympTab.from_ops(out, g.n) == t, "Clifford not exact (signs included)")
                if g.n <= 5:
                    var ph = 0.0
                    assert_true(dense_equiv(ops, out, g.n, ph), "Clifford unitary mismatch")
    # exact: optimal CX count vs brute-force 0-1 BFS
    var small = List[SubGraph]()
    small.append(line(2)); small.append(line(3)); small.append(SubGraph.complete(3))
    for g in small:
        var d = clifford_bfs_dist(g)
        for _ in range(10):
            var ops = rand_clifford_ops(g.n, 20, rng, True)
            var t = SympTab.from_ops(ops, g.n)
            var ex = synth_clifford(t, g, "exact", 0, 1)
            if t.is_identity():
                continue
            check_edges(ex, g)
            assert_true(SympTab.from_ops(ex, g.n) == t, "exact Clifford not exact")
            var ph = 0.0
            assert_true(dense_equiv(ops, ex, g.n, ph), "exact Clifford unitary mismatch")
            var tf = t.copy()
            for i in range(2 * t.n):
                tf.r[i] = 0
            assert_equal(ops_cx_count(ex), d[cliff_key(tf)])
    # 4 qubits, budgeted
    for _ in range(3):
        var g = line(4)
        var ops = rand_clifford_ops(4, 10, rng)
        var t = SympTab.from_ops(ops, 4)
        var ex = synth_clifford(t, g, "exact", 200000, 1)
        var gr = synth_clifford(t, g, "greedy", 0, 1)
        if len(ex) > 0:
            assert_true(SympTab.from_ops(ex, 4) == t, "exact 4q Clifford not exact")
            assert_true(ops_cx_count(ex) <= ops_cx_count(gr), "exact worse than greedy")
    # linear Clifford goes through the LF synthesizer (CX only + Paulis)
    var lops = rand_lf_ops(5, 20, rng)
    lops.append(g1(Op.X, 2))
    lops.append(g1(Op.Z, 4))
    var tl = SympTab.from_ops(lops, 5)
    var lo = synth_clifford(tl, line(5), "auto", 0, 1)
    assert_true(SympTab.from_ops(lo, 5) == tl, "linear Clifford not exact")
    print("  clifford ok")


def test_pauli() raises:
    var gs = graphs()
    var rng = Rng(51)
    for gi in range(len(gs)):
        var g = gs[gi].copy()
        if g.n > 6:
            continue
        for _ in range(3):
            var ops = rand_pauli_ops(g.n, 6 * g.n, rng)
            var pn = PauliNet.from_ops(ops, g.n)
            for meth in ["greedy", "auto"]:
                var out = synth_pauli_network(pn, g, meth, 0, 9, ops)
                assert_true(len(out) > 0, "no Pauli-network proposal")
                check_edges(out, g)
                var ph = 0.0
                assert_true(dense_equiv(ops, out, g.n, ph), "Pauli network not unitary-equivalent")
                var ph2 = phase_offset(ops, out, g.n)
                assert_true(abs(ph2 - ph) < 1e-6 or abs(abs(ph2 - ph) - 2 * pi) < 1e-6, "phase_offset disagrees")
    # wider networks: algebraic check of the Clifford part via the residual
    for _ in range(3):
        var g = grid(3, 3)
        var ops = rand_pauli_ops(9, 40, rng)
        var pn = PauliNet.from_ops(ops, 9)
        var out = synth_pauli_network(pn, g, "auto", 0, 9, ops)
        check_edges(out, g)
        var pn2 = PauliNet.from_ops(out, 9)
        assert_true(len(pn2.rots) == len(pn.rots), "rotation count changed")
    print("  pauli ok")


def test_timing_and_determinism() raises:
    var rng = Rng(61)
    var g = grid(3, 3)
    var reps = 20
    var tp = 0
    var tl = 0
    var tc = 0
    var cp = 0
    var cl = 0
    var cc = 0
    for _ in range(reps):
        var p = rand_perm(9, rng)
        var m = GF2Mat.from_ops(rand_lf_ops(9, 40, rng), 9)
        var t = SympTab.from_ops(rand_clifford_ops(9, 60, rng), 9)
        var t0 = Int(perf_counter_ns())
        var a = synth_perm(p, g, "auto", 0, 7)
        var t1 = Int(perf_counter_ns())
        var b = synth_linear(m, g, "auto", 0, 7)
        var t2 = Int(perf_counter_ns())
        var c = synth_clifford(t, g, "auto", 0, 7)
        var t3 = Int(perf_counter_ns())
        tp += t1 - t0
        tl += t2 - t1
        tc += t3 - t2
        cp += ops_cx_count(a)
        cl += ops_cx_count(b)
        cc += ops_cx_count(c)
        # determinism
        var a2 = synth_perm(p, g, "auto", 0, 7)
        assert_equal(len(a2), len(a))
        var c2 = synth_clifford(t, g, "auto", 0, 7)
        assert_equal(len(c2), len(c))
        for i in range(len(c)):
            assert_true(c[i].kind == c2[i].kind and c[i].q0 == c2[i].q0 and c[i].q1 == c2[i].q1, "non-deterministic")
    var r = Float64(reps) * 1e6
    print("  auto 9q (grid 3x3) ms/block: perm", Float64(tp) / r, "LF", Float64(tl) / r, "Clifford", Float64(tc) / r)
    print("  auto 9q mean CX: perm", Float64(cp) / Float64(reps), "LF", Float64(cl) / Float64(reps), "Clifford", Float64(cc) / Float64(reps))


def main() raises:
    test_algebra()
    test_pauli_net_algebra()
    test_perm()
    test_linear()
    test_clifford()
    test_pauli()
    test_timing_and_determinism()
    print("test_block_synth OK")
