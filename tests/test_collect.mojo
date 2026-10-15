"""Block collectors + collect-and-resynthesize driver (qumo/compiler/collect.mojo,
qumo/compiler/resynth.mojo).

Collector unit tests (sizes, width cap, min_block, LIFO determinism,
commutation, split components, barriers / conditioned gates, from_back), the
exact verifier (random equal and mutated candidates of every kind, checked
against a dense statevector reference), and the driver: statevector
equivalence including the global phase, 2q cost never increases, topology
respected, deterministic output. Asserts that need real synthesizers are
guarded by `stats.proposed > 0`, so the suite passes with stub synthesizers.
"""

from std.math import pi, sqrt, atan2
from std.testing import assert_equal, assert_true, assert_false
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_arity
from qumo.ir.qasm import to_qasm
from qumo.sim.runner import evolve
from qumo.sim.statevector import StateVector
from qumo.sim.rng import Rng
from qumo.compiler.common import GateSet
from qumo.compiler.target import Target
from qumo.compiler.kak import cx_cost
from qumo.compiler.basis import translate_to_basis
from qumo.compiler.collect import (
    Block, collect_blocks, collect_blocks_ordered, block_gate_ok,
    KIND_CLIFFORD, KIND_LINEAR, KIND_PERM, KIND_PAULI,
)
from qumo.compiler.resynth import (
    SynthOptions, SynthStats, resynthesize_blocks, verify_candidate, score_cost, score_depth2q,
    _lower_bound, _clifford_dep,
)
from qumo.compiler.blockalg import SubGraph, Perm, GF2Mat, SympTab
from qumo.compiler.block_synth import synth_perm, synth_linear, synth_clifford


# ----------------------------------------------------------------------------
# helpers (copied from tests/test_compiler.mojo; tests do not import each other)
# ----------------------------------------------------------------------------
def prep_circuit(n: Int, seed: UInt64) -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n, name="prep")
    for q in range(n):
        c.ry(rng.uniform() * 3.0, q)
        c.rz(rng.uniform() * 6.0, q)
    for q in range(n - 1):
        c.cx(q, q + 1)
    for q in range(n):
        c.rx(rng.uniform() * 3.0, q)
    return c^


def assert_equiv(a: Circuit, b: Circuit, label: String) raises:
    """Check b == a on two random input states, including the global phase."""
    assert_equal(a.num_qubits, b.num_qubits, label + ": width")
    for s in range(2):
        var p = prep_circuit(a.num_qubits, UInt64(1000 + s))
        var pa = p.copy()
        pa.compose(a)
        var pb = p.copy()
        pb.compose(b)
        var sa = evolve(pa)
        var sb = evolve(pb)
        var ip = sa.inner(sb)
        var re = ip[0]
        var im = ip[1]
        if abs(re - 1) > 1e-7 or abs(im) > 1e-7:
            print("NOT EQUIVALENT:", label, re, im)
        assert_true(abs(re - 1) < 1e-7 and abs(im) < 1e-7, label + ": equivalent with phase")


def expect_error(code: String, msg: String) raises:
    if not msg.startswith(code):
        print("expected", code, "got:", msg)
    assert_true(msg.startswith(code), "error code " + code)
    assert_true(msg.find("Hint:") >= 0, "error has hint: " + msg)


def total_cost(c: Circuit, gs: GateSet) -> Int:
    var s = 0
    for g in c.ops:
        if op_arity(g.kind) == 2:
            s += cx_cost(g.kind, gs)
    return s


def on_edges(c: Circuit, t: Target) -> Bool:
    for g in c.ops:
        if op_arity(g.kind) == 2 and not t.are_connected(g.q0, g.q1):
            return False
    return True


def ints(*xs: Int) -> List[Int]:
    var out = List[Int]()
    for x in xs:
        out.append(x)
    return out^


def same_list(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


# ----------------------------------------------------------------------------
# random circuits on a coupling map
# ----------------------------------------------------------------------------
def pick_edge(t: Target, mut rng: Rng, mut a: Int, mut b: Int):
    var m = len(t.edges) // 2
    var e = rng.below(m)
    a = t.edges[2 * e]
    b = t.edges[2 * e + 1]
    if rng.below(2) == 1:
        var x = a
        a = b
        b = x


def random_kind_circuit(kind: Int, t: Target, ngates: Int, seed: UInt64) -> Circuit:
    """Random circuit mostly made of `kind` gates on coupling edges, with a few
    foreign gates (t, ry, measure-free) that split blocks."""
    var rng = Rng(seed)
    var n = t.num_qubits
    var c = Circuit(n, name="rand")
    for _ in range(ngates):
        var a = 0
        var b = 0
        pick_edge(t, rng, a, b)
        var r = rng.below(20)
        if r == 0:
            c.t(a)  # foreign gate for every kind
            continue
        if kind == KIND_PERM:
            if r < 3:
                c.rz(0.37 * Float64(r), a)
            else:
                c.swap(a, b)
        elif kind == KIND_LINEAR:
            if r < 3:
                c.h(a)
            elif r < 6:
                c.swap(a, b)
            else:
                c.cx(a, b)
        elif kind == KIND_CLIFFORD:
            if r < 3: c.h(a)
            elif r < 5: c.s(a)
            elif r < 6: c.sdg(b)
            elif r < 7: c.sx(a)
            elif r < 8: c.x(b)
            elif r < 9: c.z(a)
            elif r < 10: c.rz(pi / 2, a)
            elif r < 14: c.cx(a, b)
            elif r < 17: c.cz(a, b)
            else: c.swap(a, b)
        else:
            if r < 3: c.h(a)
            elif r < 4: c.s(a)
            elif r < 6: c.rz(rng.uniform() * 3.0, a)
            elif r < 7: c.rx(rng.uniform() * 3.0, b)
            elif r < 8: c.ry(rng.uniform() * 3.0, a)
            elif r < 9: c.rzz(rng.uniform() * 3.0, a, b)
            elif r < 15: c.cx(a, b)
            elif r < 17: c.cz(a, b)
            else: c.swap(a, b)
    return c^


# ----------------------------------------------------------------------------
# collector tests
# ----------------------------------------------------------------------------
def test_gate_sets() raises:
    var c = Circuit(2)
    c.cx(0, 1)
    c.swap(0, 1)
    c.h(0)
    c.rz(pi / 2, 0)
    c.rz(0.3, 0)
    c.t(0)
    c.rzz(0.2, 0, 1)
    var p = c.parameter("th")
    c.rz(p, 0)
    var cx = c.ops[0]
    var sw = c.ops[1]
    var h = c.ops[2]
    var rzc = c.ops[3]
    var rzg = c.ops[4]
    var tg = c.ops[5]
    var rzz = c.ops[6]
    var sym = c.ops[7]
    assert_true(block_gate_ok(sw, KIND_PERM) and not block_gate_ok(cx, KIND_PERM), "perm set")
    assert_true(block_gate_ok(cx, KIND_LINEAR) and block_gate_ok(sw, KIND_LINEAR) and not block_gate_ok(h, KIND_LINEAR), "linear set")
    assert_true(block_gate_ok(h, KIND_CLIFFORD) and block_gate_ok(rzc, KIND_CLIFFORD), "clifford set")
    assert_false(block_gate_ok(rzg, KIND_CLIFFORD) or block_gate_ok(tg, KIND_CLIFFORD), "non-clifford excluded")
    assert_true(block_gate_ok(rzg, KIND_PAULI) and block_gate_ok(rzz, KIND_PAULI), "pauli set")
    assert_false(block_gate_ok(tg, KIND_PAULI), "t not in pauli set")
    assert_false(block_gate_ok(sym, KIND_PAULI), "symbolic excluded")
    var cond = cx
    cond.cond = 0
    cond.cond_val = 1
    assert_false(block_gate_ok(cond, KIND_LINEAR), "conditioned excluded")


def test_basic_and_width() raises:
    # 0:cx01 1:h2 2:cx12 3:t0 4:swap23 5:cx01 6:cx23 7:cx03
    var c = Circuit(4)
    c.cx(0, 1); c.h(2); c.cx(1, 2); c.t(0); c.swap(2, 3); c.cx(0, 1); c.cx(2, 3); c.cx(0, 3)
    var b = collect_blocks(c, KIND_LINEAR, 1, 9, False, False)
    assert_equal(len(b), 2, "per-wire: two blocks")
    assert_true(same_list(b[0].ops, ints(0, 2, 4, 6)), "first block ops")
    assert_true(same_list(b[0].qubits, ints(0, 1, 2, 3)), "first block qubits")
    assert_true(same_list(b[1].ops, ints(5, 7)), "second block ops (t closes wire 0)")
    assert_equal(b[0].first, 0, "first")
    assert_equal(b[0].last, 6, "last")
    # width cap
    var bw = collect_blocks(c, KIND_LINEAR, 1, 2, False, True)
    for x in bw:
        assert_true(len(x.qubits) <= 2, "width cap")
    var total = 0
    for x in bw:
        total += len(x.ops)
    assert_equal(total, 6, "every linear gate collected once")
    # min_block
    var bm = collect_blocks(c, KIND_LINEAR, 3, 9, False, False)
    assert_equal(len(bm), 1, "min_block drops the 2-gate block")
    # determinism of the LIFO order
    var o1 = List[Int]()
    var o2 = List[Int]()
    var b1 = collect_blocks_ordered(c, KIND_CLIFFORD, 2, 9, False, True, o1)
    var b2 = collect_blocks_ordered(c, KIND_CLIFFORD, 2, 9, False, True, o2)
    assert_true(same_list(o1, o2), "order deterministic")
    assert_equal(len(b1), len(b2), "blocks deterministic")
    for i in range(len(b1)):
        assert_true(same_list(b1[i].ops, b2[i].ops), "block ops deterministic")
    assert_equal(len(o1), len(c.ops), "order covers every op")


def test_commutation() raises:
    # cz(0,1); t(0); cz(0,1): t commutes with cz on wire 0 -> one Clifford block
    var c = Circuit(2)
    c.cz(0, 1); c.t(0); c.cz(0, 1); c.h(1)
    var nc = collect_blocks(c, KIND_CLIFFORD, 2, 9, False, False)
    var wc = collect_blocks(c, KIND_CLIFFORD, 2, 9, False, True)
    assert_equal(len(nc), 1, "per-wire: cz alone is below min, [cz,h] block")
    assert_equal(len(nc[0].ops), 2, "per-wire block size")
    assert_equal(len(wc), 1, "commutation: one block")
    assert_true(same_list(wc[0].ops, ints(0, 2, 3)), "cz pulled through t")
    # rz on a CX control commutes; on the target it does not
    var d = Circuit(3)
    d.cx(0, 1); d.rz(0.3, 0); d.cx(0, 2); d.rz(0.4, 2); d.cx(0, 2)
    var bl = collect_blocks(d, KIND_LINEAR, 1, 9, False, True)
    assert_true(same_list(bl[0].ops, ints(0, 2)), "rz on control commutes")
    assert_equal(len(bl), 2, "rz on target splits")


def test_split_barrier_cond_back() raises:
    # disjoint pairs inside one sweep -> split into components
    var c = Circuit(4)
    c.cx(0, 1); c.cx(2, 3); c.cx(0, 1); c.cx(2, 3)
    var b = collect_blocks(c, KIND_LINEAR, 1, 9, False, False)
    assert_equal(len(b), 2, "split components")
    assert_true(same_list(b[0].qubits, ints(0, 1)) and same_list(b[1].qubits, ints(2, 3)), "component qubits")
    # barrier closes blocks
    var d = Circuit(2)
    d.cx(0, 1); d.cx(1, 0); d.barrier(); d.cx(0, 1); d.cx(1, 0)
    var bd = collect_blocks(d, KIND_LINEAR, 1, 9, False, True)
    assert_equal(len(bd), 2, "barrier splits")
    # conditioned gates and measurements are never in a block and keep order
    var e = Circuit(2, 1)
    e.cx(0, 1); e.measure(1, 0); e.cx(0, 1)
    var cg = Instruction.make(Op.CX, 0, 1)
    cg.cond = 0
    cg.cond_val = 1
    e.append(cg)
    e.cx(0, 1)
    var be = collect_blocks(e, KIND_LINEAR, 1, 9, False, True)
    for x in be:
        for i in x.ops:
            assert_true(i != 1 and i != 3, "measure / conditioned excluded")
    # from_back with a width cap groups from the end
    var f = Circuit(4)
    f.cx(0, 1); f.cx(1, 2); f.cx(2, 3)
    var fw = collect_blocks(f, KIND_LINEAR, 1, 3, False, False)
    var bw = collect_blocks(f, KIND_LINEAR, 1, 3, True, False)
    assert_true(same_list(fw[0].ops, ints(0, 1)) and same_list(fw[1].ops, ints(2)), "forward grouping")
    assert_true(same_list(bw[0].ops, ints(0)) and same_list(bw[1].ops, ints(1, 2)), "backward grouping")


def test_order_is_topological() raises:
    """Emitting the ops in the collection order must keep the unitary."""
    var t = Target.grid(3, 3)
    for kind in range(4):
        for fb in range(2):
            var c = random_kind_circuit(kind, t, 120, UInt64(50 + kind))
            var order = List[Int]()
            var bl = collect_blocks_ordered(c, kind, 2, 4, fb == 1, True, order)
            var d = c.empty_like()
            for i in order:
                d.append(c.ops[i])
            assert_equiv(c, d, "collection order kind " + String(kind))
            # every block is contiguous in the order (up to other components)
            var pos = List[Int](length=len(c.ops), fill=0)
            for k in range(len(order)):
                pos[order[k]] = k
            for x in bl:
                for k in range(1, len(x.ops)):
                    assert_true(pos[x.ops[k]] > pos[x.ops[k - 1]], "block ops in order")


# ----------------------------------------------------------------------------
# verifier fuzz: random blocks, equal and mutated candidates
# ----------------------------------------------------------------------------
def dense_equal(a: List[Instruction], b: List[Instruction], n: Int, mut phase: Float64) raises -> Bool:
    """Reference check with the statevector simulator on all basis columns."""
    var dim = 1 << n
    var sr = 0.0
    var si = 0.0
    for col in range(dim):
        var ca = Circuit(n)
        var cb = Circuit(n)
        for q in range(n):
            if (col >> q) & 1 == 1:
                ca.x(q)
                cb.x(q)
        for g in a:
            ca.append(g)
        for g in b:
            cb.append(g)
        var ip = evolve(ca).inner(evolve(cb))
        if abs(sqrt(ip[0] * ip[0] + ip[1] * ip[1]) - 1.0) > 1e-9:
            return False
        sr += ip[0]
        si += ip[1]
    phase = atan2(si, sr)
    return sqrt(sr * sr + si * si) / Float64(dim) > 1.0 - 1e-9


def test_verifier_fuzz() raises:
    var lt = Target.line(4)
    var full = List[Bool](length=16, fill=True)
    var line_adj = List[Bool](length=16, fill=False)
    for i in range(4):
        for j in range(4):
            line_adj[i * 4 + j] = abs(i - j) == 1
    var accepted_bad = 0
    var checked = 0
    for kind in range(4):
        for it in range(220):
            var rng = Rng(UInt64(9000 + 1000 * kind + it))
            var c = random_kind_circuit(kind, lt, 6 + rng.below(10), UInt64(77 + 1000 * kind + it))
            var orig = List[Instruction]()
            for g in c.ops:
                if block_gate_ok(g, kind):
                    orig.append(g)
            if len(orig) == 0:
                continue
            # an equal candidate: insert an inverse pair (and for Clifford, XZXZ = -I)
            var cand = orig.copy()
            var pos = rng.below(len(cand) + 1)
            var a = 0
            var b = 0
            pick_edge(lt, rng, a, b)
            var pair = Instruction.make(Op.SWAP, a, b)
            if kind == KIND_LINEAR:
                pair = Instruction.make(Op.CX, a, b)
            elif kind == KIND_CLIFFORD or kind == KIND_PAULI:
                pair = Instruction.make(Op.CZ, a, b)
            cand.insert(pos, pair)
            cand.insert(pos, pair)
            if kind == KIND_CLIFFORD:
                cand.append(Instruction.make(Op.X, a))
                cand.append(Instruction.make(Op.Z, a))
                cand.append(Instruction.make(Op.X, a))
                cand.append(Instruction.make(Op.Z, a))
            var ph = 0.0
            var ref_ph = 0.0
            assert_true(verify_candidate(kind, orig, cand, 4, line_adj, ph), "equal candidate verifies")
            assert_true(dense_equal(orig, cand, 4, ref_ph), "reference agrees")
            var dph = ph - ref_ph
            while dph > pi:
                dph -= 2 * pi
            while dph < -pi:
                dph += 2 * pi
            assert_true(abs(dph) < 1e-9, "phase matches the reference")
            # a mutated candidate: drop one gate (may still be equal by chance)
            var bad = orig.copy()
            _ = bad.pop(rng.below(len(bad)))
            var ok = verify_candidate(kind, orig, bad, 4, full, ph)
            var truth = dense_equal(orig, bad, 4, ref_ph)
            if ok and not truth:
                accepted_bad += 1
            assert_equal(ok, truth, "mutated verdict matches the reference")
            checked += 1
            # off-topology candidate is rejected
            var off = orig.copy()
            var g2 = Instruction.make(Op.SWAP, 0, 3)
            if kind == KIND_LINEAR:
                g2 = Instruction.make(Op.CX, 0, 3)
            elif kind != KIND_PERM:
                g2 = Instruction.make(Op.CZ, 0, 3)
            off.append(g2)
            off.append(g2)
            assert_false(verify_candidate(kind, orig, off, 4, line_adj, ph), "off-edge rejected")
    assert_equal(accepted_bad, 0, "no accepted non-equivalent proposal")
    assert_true(checked >= 800, "200+ blocks per kind")


# ----------------------------------------------------------------------------
# driver tests
# ----------------------------------------------------------------------------
def test_validation() raises:
    var c = Circuit(3)
    c.cx(0, 1)
    var t = Target.line(3)
    var st = SynthStats()
    var o = SynthOptions.default(KIND_LINEAR)
    o.method = "magic"
    try:
        _ = resynthesize_blocks(c, t, True, o, st)
        assert_true(False, "unknown method must raise")
    except e:
        expect_error("QM-E616", String(e))
    var o2 = SynthOptions.default(KIND_PAULI)
    o2.max_qubits = 20
    try:
        _ = resynthesize_blocks(c, t, True, o2, st)
        assert_true(False, "too wide must raise")
    except e:
        expect_error("QM-E618", String(e))
    var d = SynthOptions.default(KIND_PERM)
    assert_equal(d.max_qubits, 12, "perm default width")
    assert_equal(d.min_block, 4, "perm default min")
    assert_equal(SynthOptions.default(KIND_CLIFFORD).min_block, 2, "clifford default min")
    assert_equal(SynthOptions.default(KIND_PAULI).max_qubits, 6, "pauli default width")
    assert_equal(d.reps, 10, "reps default")


def test_identity_blocks() raises:
    """Identity blocks are removed (no synthesizer needed); phases kept."""
    var c = Circuit(3)
    c.cx(0, 1); c.rz(0.3, 0); c.rx(0.2, 1); c.cx(0, 1); c.h(2); c.cx(1, 2)
    var t = Target.line(3)
    var o = SynthOptions.default(KIND_LINEAR)
    o.min_block = 2
    o.max_qubits = 2
    var st = SynthStats()
    var d = c.copy()
    _ = resynthesize_blocks(d, t, True, o, st)
    assert_equiv(c, d, "linear identity block")
    assert_equal(total_cost(d, GateSet()), 1, "cx pair removed through commuting rotations")
    # Clifford block equal to -I up to the 2q part: XZXZ = -I, phase pi tracked
    var e = Circuit(2)
    e.x(0); e.z(0); e.cz(0, 1); e.cz(0, 1); e.x(0); e.z(0)
    var oc = SynthOptions.default(KIND_CLIFFORD)
    var st2 = SynthStats()
    var f = e.copy()
    _ = resynthesize_blocks(f, Target.line(2), True, oc, st2)
    assert_equiv(e, f, "clifford identity with phase")
    assert_equal(len(f.ops), 0, "clifford identity removed")


def check_driver(kind: Int, t: Target, physical: Bool, ngates: Int, seed: UInt64, label: String) raises -> Int:
    var c = random_kind_circuit(kind, t, ngates, seed)
    var o = SynthOptions.default(kind)
    o.seed = Int(seed)
    var gs = GateSet(t.basis)
    var st = SynthStats()
    var d = c.copy()
    var before = total_cost(c, gs)
    var saved = resynthesize_blocks(d, t, physical, o, st)
    assert_equiv(c, d, label)
    var after = total_cost(d, gs)
    assert_true(after <= before, label + ": 2q cost never increases")
    if physical:
        assert_true(on_edges(d, t), label + ": coupling respected")
    assert_true(saved >= 0, label + ": saved >= 0")
    assert_true(st.rejected_not_equal + st.verified <= st.proposed, label + ": verified + rejected <= proposed")
    if st.rejected_not_equal > 0:
        print("  note:", label, "-", st.rejected_not_equal, "synthesizer proposals rejected by the verifier (dropped safely)")
    if st.proposed == 0:
        # stub synthesizers: only identity blocks may change the circuit
        assert_true(after <= before, label + ": stub run")
    # determinism: same seed, byte-identical QASM
    var d2 = c.copy()
    var st3 = SynthStats()
    _ = resynthesize_blocks(d2, t, physical, o, st3)
    assert_equal(to_qasm(d), to_qasm(d2), label + ": deterministic")
    return st.replaced


def test_driver_random() raises:
    var targets = List[Target]()
    targets.append(Target.grid(3, 3))
    targets.append(Target.heavy_hex(3))
    targets.append(Target.line(7))
    var names: List[String] = ["grid3x3", "heavy_hex3", "line7"]
    var replaced = 0
    for ti in range(len(targets)):
        for kind in range(4):
            for s in range(2):
                var lbl = names[ti] + " kind " + String(kind) + " seed " + String(s)
                replaced += check_driver(kind, targets[ti], True, 90, UInt64(300 + 17 * ti + 5 * kind + s), lbl + " physical")
                replaced += check_driver(kind, targets[ti], False, 90, UInt64(700 + 17 * ti + 5 * kind + s), lbl + " all-to-all")
    print("  driver: blocks replaced over the random set:", replaced)


def test_basis_kept() raises:
    """A circuit already in the target basis stays in it."""
    var t = Target.grid(3, 3, "sc_cz")
    var c = random_kind_circuit(KIND_CLIFFORD, Target.grid(3, 3), 80, 4242)
    var dropped = False
    _ = translate_to_basis(c, GateSet(t.basis), dropped)
    var o = SynthOptions.default(KIND_CLIFFORD)
    var st = SynthStats()
    var d = c.copy()
    _ = resynthesize_blocks(d, t, True, o, st)
    assert_equiv(c, d, "sc_cz clifford")
    var gs = GateSet(t.basis)
    for g in d.ops:
        assert_true(gs.has(g.kind), "stays in basis")
    assert_true(total_cost(d, gs) <= total_cost(c, gs), "sc_cz cost never increases")


def test_no_change_without_proposals() raises:
    """With nothing to propose and no identity block, the circuit is untouched."""
    var c = Circuit(3)
    c.cx(0, 1); c.cx(1, 2); c.cx(0, 1); c.cx(1, 2)
    var t = Target.line(3)
    var o = SynthOptions.default(KIND_LINEAR)
    var st = SynthStats()
    var d = c.copy()
    _ = resynthesize_blocks(d, t, True, o, st)
    if st.proposed == 0:
        assert_equal(to_qasm(c), to_qasm(d), "untouched")
        assert_equal(st.replaced, 0, "nothing replaced")
    assert_equiv(c, d, "no-change case")



# ----------------------------------------------------------------------------
# lower bounds (used to skip searches that cannot win) must be sound
# ----------------------------------------------------------------------------
def random_clifford_op(n: Int, mut rng: Rng) -> Instruction:
    """Any Clifford gate kind the support tracker knows, incl. k*pi/2 rotations."""
    var a = rng.below(n)
    var b = (a + 1 + rng.below(n - 1)) % n
    var r = rng.below(21)
    var ang = Float64(rng.below(4) - 1) * pi / 2
    if r == 0: return Instruction.make(Op.H, a)
    if r == 1: return Instruction.make(Op.S, a)
    if r == 2: return Instruction.make(Op.SDG, a)
    if r == 3: return Instruction.make(Op.SX, a)
    if r == 4: return Instruction.make(Op.SXDG, a)
    if r == 5: return Instruction.make(Op.Y, a)
    if r == 6: return Instruction.make(Op.RZ, a, p0=ang)
    if r == 7: return Instruction.make(Op.RX, a, p0=ang)
    if r == 8: return Instruction.make(Op.RY, a, p0=ang)
    if r == 9: return Instruction.make(Op.P, a, p0=ang)
    if r == 10: return Instruction.make(Op.CX, a, b)
    if r == 11: return Instruction.make(Op.CY, a, b)
    if r == 12: return Instruction.make(Op.CZ, a, b)
    if r == 13: return Instruction.make(Op.SWAP, a, b)
    if r == 14: return Instruction.make(Op.ISWAP, a, b)
    if r == 15: return Instruction.make(Op.ISWAPDG, a, b)
    if r == 16: return Instruction.make(Op.ECR, a, b)
    if r == 17: return Instruction.make(Op.RZZ, a, b, p0=ang)
    if r == 18: return Instruction.make(Op.RXX, a, b, p0=ang)
    if r == 19: return Instruction.make(Op.RYY, a, b, p0=ang)
    return Instruction.make(Op.X, b)


def test_clifford_support_tracker() raises:
    """_clifford_dep (column masks, no signs) equals the supports of the
    signed tableau (blockalg.SympTab) on random Clifford blocks."""
    var rng = Rng(9191)
    var checked = 0
    for it in range(300):
        var n = 2 + it % 5
        var ops = List[Instruction]()
        for _ in range(1 + rng.below(25)):
            ops.append(random_clifford_op(n, rng))
        var rows = List[UInt64](length=n, fill=0)
        assert_true(_clifford_dep(ops, n, rows), "known gates")
        var t = SympTab.from_ops(ops, n)
        for j in range(n):
            var want: UInt64 = 0
            for q in range(n):
                var sup = t.x[q] | t.z[q] | t.x[n + q] | t.z[n + q]
                if (sup >> UInt64(j)) & 1 != 0:
                    want |= UInt64(1) << UInt64(q)
            assert_equal(rows[j], want, "support row " + String(j) + " at iteration " + String(it))
        checked += 1
    var unk = List[Instruction]()
    unk.append(Instruction.make(Op.T, 0))
    var rows = List[UInt64](length=2, fill=0)
    assert_false(_clifford_dep(unk, 2, rows), "non-Clifford gate: unknown")
    print("  Clifford support tracker matches the tableau on", checked, "blocks")


def test_lower_bounds_sound() raises:
    """No verified circuit (incumbent, auto or exact synthesis) is ever below
    the bound, on random blocks of every kind (200+ per kind)."""
    var targets = List[Target]()
    targets.append(Target.line(3))
    targets.append(Target.line(4))
    targets.append(Target.grid(2, 2))
    targets.append(Target.grid(2, 3))
    var gss = List[GateSet]()
    gss.append(GateSet())
    gss.append(GateSet(Target.grid(2, 2, "sc_cz").basis))
    var tight = 0
    var witnessed = 0
    for kind in range(4):
        var checked = 0
        for it in range(220):
            var t = targets[it % len(targets)].copy()
            var gs = gss[(it // 4) % 2].copy()
            var n = t.num_qubits
            var rng = Rng(UInt64(5000 + 1000 * kind + it))
            var ops = List[Instruction]()
            var ng = 2 + rng.below(14)
            for _ in range(ng):
                var a = 0
                var b = 0
                pick_edge(t, rng, a, b)
                if kind == KIND_PERM:
                    ops.append(Instruction.make(Op.SWAP, a, b))
                elif kind == KIND_LINEAR:
                    ops.append(Instruction.make(Op.CX if rng.below(4) != 0 else Op.SWAP, a, b))
                else:
                    var g = random_clifford_op(n, rng)
                    if op_arity(g.kind) == 2:
                        g.q0 = a
                        g.q1 = b
                    ops.append(g)
            var qs = List[Int]()
            for q in range(n):
                qs.append(q)
            var sg = SubGraph.induced(t, qs)
            var lbc = 0
            var lbd = 0
            _lower_bound(kind, ops, sg, gs, lbc, lbd)
            var c0 = score_cost(ops, gs)
            var d0 = score_depth2q(ops, n)
            assert_true(lbc <= c0 and lbd <= d0, "bound <= incumbent")
            if lbc == c0 and lbd == d0:
                tight += 1
            # every verified synthesized circuit respects the bound
            var adj = List[Bool](length=n * n, fill=False)
            for i in range(n):
                for j in range(n):
                    adj[i * n + j] = i != j and t.are_connected(i, j)
            var methods: List[String] = ["auto"]
            if n <= 4 and kind != KIND_PAULI and (kind != KIND_CLIFFORD or (n <= 3 and it % 20 == 0)):
                methods.append("exact")
            for m in methods:
                var cand: List[Instruction]
                if kind == KIND_PERM:
                    cand = synth_perm(Perm.from_ops(ops, n), sg, m, 0, it)
                elif kind == KIND_LINEAR:
                    cand = synth_linear(GF2Mat.from_ops(ops, n), sg, m, 0, it)
                else:
                    cand = synth_clifford(SympTab.from_ops(ops, n), sg, m, 0, it)
                if len(cand) == 0:
                    continue
                var vk = kind if kind != KIND_PAULI else KIND_CLIFFORD
                var ph = 0.0
                if not verify_candidate(vk, ops, cand, n, adj, ph):
                    continue
                witnessed += 1
                assert_true(score_cost(cand, gs) >= lbc, "verified " + m + " candidate below the cost bound (kind " + String(kind) + ")")
                assert_true(score_depth2q(cand, n) >= lbd, "verified " + m + " candidate below the depth bound (kind " + String(kind) + ")")
            checked += 1
        assert_true(checked >= 200, "200+ blocks per kind")
    print("  lower bounds sound on 880 blocks;", tight, "incumbents provably optimal;", witnessed, "verified syntheses checked")


def test_unitary_tolerance() raises:
    """The dense verifier is entrywise 1e-9: a rotation angle off by 1e-8 is
    rejected, the same angle is accepted."""
    var rng = Rng(77)
    var full = List[Bool](length=16, fill=True)
    var rejected = 0
    var accepted = 0
    for it in range(100):
        var orig = List[Instruction]()
        for _ in range(6):
            var a = rng.below(4)
            var b = (a + 1 + rng.below(3)) % 4
            orig.append(Instruction.make(Op.CX, a, b))
            orig.append(Instruction.make(Op.RZ, b, p0=rng.uniform() * 3.0))
        var same = orig.copy()
        var bad = orig.copy()
        var eps = 1e-8 if it % 2 == 0 else 1e-6
        orig.append(Instruction.make(Op.RZ, 0, p0=0.7))
        same.append(Instruction.make(Op.RZ, 0, p0=0.7))
        bad.append(Instruction.make(Op.RZ, 0, p0=0.7 + eps))
        var ph = 0.0
        if verify_candidate(KIND_PAULI, orig, same, 4, full, ph):
            accepted += 1
        if not verify_candidate(KIND_PAULI, orig, bad, 4, full, ph):
            rejected += 1
    assert_equal(accepted, 100, "equal Pauli blocks accepted")
    assert_equal(rejected, 100, "angle perturbations of 1e-8 / 1e-6 rejected")


def test_pauli_identity_and_strict_width() raises:
    """An all-Clifford identity Pauli block is removed; strict=1 turns an
    explicit exact method on a too-wide block into QM-E618."""
    var c = Circuit(3)
    c.cx(0, 1); c.cz(1, 2); c.s(0); c.sdg(0); c.cz(1, 2); c.cx(0, 1)
    var o = SynthOptions.default(KIND_PAULI)
    var st = SynthStats()
    var d = c.copy()
    _ = resynthesize_blocks(d, Target.line(3), True, o, st)
    assert_equiv(c, d, "pauli identity")
    assert_equal(len(d.ops), 0, "pauli identity block removed")
    # a 7-qubit CX chain: too wide for linear exact (<= 5 qubits)
    var w = Circuit(7)
    for _ in range(2):
        for q in range(6):
            w.cx(q, q + 1)
    var ol = SynthOptions.default(KIND_LINEAR)
    ol.method = "exact"
    var st2 = SynthStats()
    var w2 = w.copy()
    _ = resynthesize_blocks(w2, Target.line(7), True, ol, st2)
    assert_true(st2.note.find("skipped_too_wide=") >= 0, "non-strict: skipped and reported: " + st2.note)
    assert_equiv(w, w2, "skipped block unchanged")
    ol.strict = True
    var w3 = w.copy()
    var st3 = SynthStats()
    try:
        _ = resynthesize_blocks(w3, Target.line(7), True, ol, st3)
        assert_true(False, "strict exact on a too-wide block must raise")
    except e:
        expect_error("QM-E618", String(e))

def main() raises:
    test_clifford_support_tracker()
    test_lower_bounds_sound()
    test_unitary_tolerance()
    test_pauli_identity_and_strict_width()
    test_gate_sets()
    test_basic_and_width()
    test_commutation()
    test_split_barrier_cond_back()
    test_order_is_topological()
    test_verifier_fuzz()
    test_validation()
    test_identity_blocks()
    test_no_change_without_proposals()
    test_basis_kept()
    test_driver_random()
    print("test_collect OK")
