"""Compiler test-suite: every pass and preset is statevector-verified.

Equivalence is checked on random input states (prep circuit) with the *exact*
global phase (|<a|b> - 1| < 1e-7), which also validates `global_phase`
bookkeeping; routed circuits are compared through the initial / final layout
permutation, and must use coupling edges + basis gates only.
"""

from std.math import pi, sqrt
from std.testing import assert_equal, assert_true, assert_false
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary, op_num_params
from qumo.ir.param import Param
from qumo.sim.runner import evolve, sample
from qumo.sim.statevector import StateVector
from qumo.sim.rng import Rng
from qumo.compiler.common import GateSet
from qumo.compiler.target import Target, parse_target, parse_basis
from qumo.compiler.optimize import (
    remove_identities, cancel_inverses, merge_rotations, commute_rotations, fuse_1q, eject_z,
)
from qumo.compiler.phase_fold import phase_fold
from qumo.compiler.basis import translate_to_basis, unroll_3q
from qumo.compiler.routing import RouteOptions, route_circuit, star_preroute, slide_swaps
from qumo.compiler.kak import consolidate_2q, kak_synthesize, gate_m4
from qumo.compiler.pauli_simp import pauli_resynthesize
from qumo.bench.circuits import hamiltonian_circuit_from_labels
from qumo.compiler.pass_manager import (
    PassManager, CompileResult, parse_pipeline, pipeline_to_string, preset_pipeline,
    transpile, compile_spec, conformance_error, describe_passes,
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def rand_angle(mut rng: Rng) -> Float64:
    var u = rng.uniform()
    if u < 0.25:
        # Clifford+T-friendly angles exercise named-gate merges
        return Float64(rng.below(8) - 4) * pi / 4
    return (rng.uniform() * 2 - 1) * pi


def random_circuit(n: Int, ngates: Int, seed: UInt64, with_3q: Bool = True) -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n, name="rand" + String(seed))
    var one: List[Int] = [Op.I, Op.X, Op.Y, Op.Z, Op.H, Op.S, Op.SDG, Op.T, Op.TDG, Op.SX, Op.SXDG,
                          Op.RX, Op.RY, Op.RZ, Op.P, Op.U]
    var two: List[Int] = [Op.CX, Op.CY, Op.CZ, Op.CH, Op.SWAP, Op.ISWAP, Op.ISWAPDG, Op.ECR, Op.CP,
                          Op.CRX, Op.CRY, Op.CRZ, Op.RXX, Op.RYY, Op.RZZ]
    var three: List[Int] = [Op.CCX, Op.CSWAP, Op.CCZ]
    for _ in range(ngates):
        var u = rng.uniform()
        var nops = len(c.ops)
        if u < 0.15 and nops > 0:
            # redundancy patterns: inverse / repeated / same-axis partner of the last gate
            var last = c.ops[nops - 1]
            var g = last
            if op_num_params(last.kind) == 1:
                if rng.uniform() < 0.5:
                    g.p0 = -last.p0
                else:
                    g.p0 = Param(rand_angle(rng))
            elif op_num_params(last.kind) == 0:
                if last.kind == Op.S: g.kind = Op.SDG
                elif last.kind == Op.T: g.kind = Op.TDG
                elif last.kind == Op.SX: g.kind = Op.SXDG
            c.append(g)
            continue
        var k: Int
        var r = rng.uniform()
        if r < 0.5:
            k = one[rng.below(len(one))]
        elif r < 0.93 or not with_3q or n < 3:
            k = two[rng.below(len(two))]
        else:
            k = three[rng.below(len(three))]
        var a = op_arity(k)
        var q0 = rng.below(n)
        var q1 = (q0 + 1 + rng.below(n - 1)) % n
        var q2 = -1
        if a == 3:
            q2 = (q0 + 1 + rng.below(n - 1)) % n
            while q2 == q0 or q2 == q1:
                q2 = (q2 + 1) % n
        if a == 1:
            q1 = -1
        c.add(k, q0, q1, q2, rand_angle(rng), rand_angle(rng), rand_angle(rng))
    return c^


def clifford_t_circuit(n: Int, ngates: Int, seed: UInt64) -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n, name="cliffT")
    for _ in range(ngates):
        var r = rng.below(8)
        var q = rng.below(n)
        var t = (q + 1 + rng.below(n - 1)) % n
        if r == 0: c.h(q)
        elif r == 1: c.s(q)
        elif r == 2: c.t(q)
        elif r == 3: c.tdg(q)
        elif r == 4: c.x(q)
        else: c.cx(q, t)
    return c^


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


def overlap(a: StateVector, b: StateVector, mut re: Float64, mut im: Float64):
    var ip = a.inner(b)
    re = ip[0]
    im = ip[1]


def assert_equiv(a: Circuit, b: Circuit, label: String, exact_phase: Bool = True) raises:
    """Check b == a on two random input states (exact global phase by default)."""
    assert_equal(a.num_qubits, b.num_qubits, label + ": width")
    for s in range(2):
        var p = prep_circuit(a.num_qubits, UInt64(1000 + s))
        var pa = p.copy()
        pa.compose(a)
        var pb = p.copy()
        pb.compose(b)
        var sa = evolve(pa)
        var sb = evolve(pb)
        var re = 0.0
        var im = 0.0
        overlap(sa, sb, re, im)
        var fid = re * re + im * im
        if fid < 1 - 1e-9:
            print("NOT EQUIVALENT:", label, "fidelity", fid)
            print(a)
            print(b)
        assert_true(fid > 1 - 1e-9, label + ": fidelity")
        if exact_phase:
            if abs(re - 1) > 1e-7 or abs(im) > 1e-7:
                print("PHASE MISMATCH:", label, re, im)
            assert_true(abs(re - 1) < 1e-7 and abs(im) < 1e-7, label + ": global phase")


def assert_routed_equiv(orig: Circuit, res: CompileResult, label: String) raises:
    """Compiled circuit on N physical qubits == orig, through the layouts."""
    var n = orig.num_qubits
    var cc = res.circuit.copy()
    var N = cc.num_qubits
    for s in range(2):
        var p = prep_circuit(n, UInt64(2000 + s))
        var pa = p.copy()
        pa.compose(orig)
        var sa = evolve(pa)
        var qmap = List[Int]()
        for l in range(n):
            qmap.append(res.initial_layout[l])
        var pb = Circuit(N)
        pb.compose_mapped(p, qmap)
        pb.compose(cc)
        var sb = evolve(pb)
        var re = 0.0
        var im = 0.0
        for i in range(1 << n):
            var j = 0
            for l in range(n):
                if (i >> l) & 1 == 1:
                    j |= 1 << res.final_layout[l]
            re += sa.re[i] * sb.re[j] + sa.im[i] * sb.im[j]
            im += sa.re[i] * sb.im[j] - sa.im[i] * sb.re[j]
        var fid = re * re + im * im
        if fid < 1 - 1e-9 or abs(re - 1) > 1e-7:
            print("ROUTED MISMATCH:", label, fid, re, im)
            print(res.table())
        assert_true(fid > 1 - 1e-9, label + ": routed fidelity")
        assert_true(abs(re - 1) < 1e-7 and abs(im) < 1e-7, label + ": routed global phase")


def assert_conforms(c: Circuit, t: Target, label: String) raises:
    var err = conformance_error(c, t)
    if err.byte_length() > 0:
        print("NONCONFORMING:", label, err)
    assert_equal(err, String(""), label + ": conformance")


def expect_error(code: String, msg: String) raises:
    if not msg.startswith(code):
        print("expected", code, "got:", msg)
    assert_true(msg.startswith(code), "error code " + code)
    assert_true(msg.find("Hint:") >= 0, "error has hint: " + msg)


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def test_targets() raises:
    var h3 = Target.heavy_hex(3)
    assert_equal(h3.num_qubits, 19)
    assert_equal(h3.num_edges(), 20)
    var h5 = Target.heavy_hex(5)
    assert_equal(h5.num_qubits, 57)
    for d in [3, 5, 7]:
        var h = Target.heavy_hex(d)
        assert_equal(h.num_qubits, (5 * d * d - 2 * d - 1) // 2)
        assert_true(h.is_connected_graph())
        for q in range(h.num_qubits):
            assert_true(h.degree(q) <= 3, "heavy hex degree <= 3")
    var g = Target.grid(3, 4)
    assert_equal(g.num_edges(), 17)
    assert_equal(g.distance(0, 11), 5)
    assert_true(g.are_connected(0, 4))
    assert_false(g.are_connected(0, 5))
    var tor = Target.torus(3, 3)
    assert_equal(tor.num_edges(), 18)
    assert_equal(Target.ring(5).num_edges(), 5)
    assert_equal(Target.tree(7).num_edges(), 6)
    var na = Target.neutral_atom(3, 3)
    assert_equal(na.num_edges(), 20)  # king graph 3x3
    var ion = Target.trapped_ion(5)
    assert_true(ion.full and ion.are_connected(0, 4))
    assert_true(ion.supports(Op.RXX) and not ion.supports(Op.CX))
    var a2a = Target.all_to_all(4)
    assert_true(a2a.supports(Op.CCX))
    var t = parse_target("grid:2x3/sc_cz")
    assert_equal(t.num_qubits, 6)
    assert_true(t.supports(Op.CZ) and t.supports(Op.SX) and not t.supports(Op.CX))
    var d = t.describe()
    assert_true(d.find("\"num_qubits\":6") >= 0 and d.find("\"basis\":[\"rz\",\"sx\",\"x\",\"cz\"]") >= 0)
    var sc = Target.superconducting_heavy_hex(3)
    assert_true(sc.estimate_fidelity(random_circuit(5, 20, 3)) < 1.0)
    var e = Target.from_edges(3, [0, 1, 1, 2], "rz,ry,rzz")
    assert_true(e.supports(Op.RZZ) and e.are_connected(1, 2))
    try:
        _ = Target.heavy_hex(4)
        assert_true(False, "even d must fail")
    except err:
        expect_error("QM-E612", String(err))
    try:
        _ = parse_basis("rz,foo")
        assert_true(False)
    except err:
        expect_error("QM-E610", String(err))
    try:
        _ = parse_target("blob:3")
        assert_true(False)
    except err:
        expect_error("QM-E611", String(err))
    print("  targets ok")


def basis_list() -> List[String]:
    var b: List[String] = [
        "generic", "sc_cx", "sc_cz", "sc_ecr", "sc_iswap", "ion_ms", "ion_zz", "atom_cz",
        "rx,ry,rz,cz", "rz,rx,rxx", "rz,ry,rzz", "u,cx", "rz,sx,x,cz", "p,sx,cp",
        "u,iswapdg", "rz,sx,ryy", "rx,ry,cz", "rz,sxdg,ecr",
    ]
    return b^


def test_basis_rules() raises:
    """Every multi-qubit gate x every basis: exact single-gate translation."""
    var rng = Rng(11)
    for b in basis_list():
        var gs = GateSet(parse_basis(b))
        for k in range(Op.CCZ + 1):
            var c = Circuit(3)
            var a = op_arity(k)
            if a == 1:
                c.add(k, 1, -1, -1, rand_angle(rng), rand_angle(rng), rand_angle(rng))
            elif a == 2:
                c.add(k, 2, 0, -1, rand_angle(rng))
            else:
                c.add(k, 2, 0, 1)
            var t = c.copy()
            var dropped = False
            _ = translate_to_basis(t, gs, dropped)
            for g in t.ops:
                assert_true(gs.has(g.kind), "basis " + b + " contains " + op_name(g.kind))
            assert_equiv(c, t, "rule " + op_name(k) + " -> " + b)
    print("  basis rules ok (", len(basis_list()), "bases x", Op.CCZ + 1, "gates )")


def test_passes_random() raises:
    var anyset = GateSet()
    for seed in range(10):
        var n = 5 + seed % 4
        var c = random_circuit(n, 60, UInt64(seed + 1))
        # remove_identities
        var x = c.copy()
        _ = remove_identities(x)
        assert_equiv(c, x, "remove_identities seed " + String(seed))
        # cancel
        x = c.copy()
        var r = cancel_inverses(x, anyset)
        assert_true(len(x.ops) <= len(c.ops))
        assert_equiv(c, x, "cancel seed " + String(seed))
        _ = r
        # merge
        x = c.copy()
        _ = merge_rotations(x, anyset)
        assert_equiv(c, x, "merge seed " + String(seed))
        # commute
        x = c.copy()
        _ = commute_rotations(x, anyset)
        assert_equiv(c, x, "commute seed " + String(seed))
        # fuse1q (any + bases)
        x = c.copy()
        _ = fuse_1q(x, anyset)
        assert_equiv(c, x, "fuse1q any seed " + String(seed))
        # eject_z
        x = c.copy()
        _ = eject_z(x, anyset)
        assert_equiv(c, x, "eject_z seed " + String(seed))
        # phase_fold
        x = c.copy()
        _ = phase_fold(x, anyset)
        assert_equiv(c, x, "phase_fold seed " + String(seed))
        # kak block consolidation
        x = c.copy()
        var nblk = 0
        _ = consolidate_2q(x, anyset, nblk)
        assert_equiv(c, x, "kak seed " + String(seed))
        # elide_swaps (permutation folded into the final layout)
        var er = compile_spec(c, Target.all_to_all(n), "elide_swaps")
        assert_routed_equiv(c, er, "elide_swaps seed " + String(seed))
        # unroll3q
        x = c.copy()
        _ = unroll_3q(x)
        for g in x.ops:
            assert_true(op_arity(g.kind) <= 2)
        assert_equiv(c, x, "unroll3q seed " + String(seed))
        # basis + basis-aware fuse1q / eject_z / commute keep the basis
        for b in basis_list():
            var gs = GateSet(parse_basis(b))
            var y = c.copy()
            var dropped = False
            _ = translate_to_basis(y, gs, dropped)
            assert_equiv(c, y, "basis " + b + " seed " + String(seed))
            _ = fuse_1q(y, gs)
            _ = eject_z(y, gs)
            _ = commute_rotations(y, gs)
            _ = cancel_inverses(y, gs)
            _ = phase_fold(y, gs)
            _ = fuse_1q(y, gs)
            for g in y.ops:
                if not gs.has(g.kind):
                    print("non-basis gate after optimisation:", b, g)
                assert_true(gs.has(g.kind), "optimised output stays in basis " + b)
            assert_equiv(c, y, "basis+opt " + b + " seed " + String(seed))
    print("  passes on random circuits ok")


def test_cancellation_examples() raises:
    var anyset = GateSet()
    var c = Circuit(3)
    c.h(0)
    c.h(0)
    c.cx(0, 1)
    c.rz(0.3, 0)  # commutes with cx control
    c.cx(0, 1)
    c.cz(1, 2)
    c.cz(2, 1)
    c.s(2)
    c.sdg(2)
    c.swap(0, 2)
    c.swap(2, 0)
    var x = c.copy()
    _ = cancel_inverses(x, anyset)
    assert_equal(len(x.ops), 1)
    assert_equal(x.ops[0].kind, Op.RZ)
    assert_equiv(c, x, "cancel example")
    # commutation-aware rotation merge: rz . cx(ctrl) . rz -> 1 rz ; rx on target
    var m = Circuit(2)
    m.rz(0.2, 0)
    m.rx(0.4, 1)
    m.cx(0, 1)
    m.rz(0.3, 0)
    m.rx(-0.4, 1)
    m.t(0)
    var y = m.copy()
    _ = commute_rotations(y, anyset)
    assert_equal(len(y.ops), 2)
    assert_equiv(m, y, "commute example")
    # t t -> s, symbolic merge
    var s = Circuit(1)
    var th = s.parameter("th")
    s.t(0)
    s.t(0)
    s.rz(th, 0)
    s.rz(th * 2.0, 0)
    var z = s.copy()
    _ = merge_rotations(z, anyset)
    assert_equal(len(z.ops), 1)  # t t rz(th) rz(2 th) -> rz(3 th + pi/2), phase pi/4
    assert_equal(z.ops[0].kind, Op.RZ)
    assert_equal(z.ops[0].p0.coeff, 3.0)
    assert_equiv(s.bind([0.7]), z.bind([0.7]), "symbolic merge")
    # remove identities with phase
    var r = Circuit(2)
    r.rz(2 * pi, 0)
    r.rx(4 * pi, 1)
    r.p(2 * pi, 1)
    r.rzz(2 * pi, 0, 1)
    r.id(0)
    r.h(1)
    var w = r.copy()
    _ = remove_identities(w)
    assert_equal(len(w.ops), 1)
    assert_equiv(r, w, "remove identities")
    print("  cancellation examples ok")


def test_phase_fold() raises:
    var anyset = GateSet()
    # CCX . CCX = I decomposed into Clifford+T: 14 T gates fold to none
    var c = Circuit(3, name="toffoli2")
    c.ccx(0, 1, 2)
    c.ccx(0, 1, 2)
    _ = unroll_3q(c)
    var t0 = c.t_count()
    assert_equal(t0, 14)
    var x = c.copy()
    _ = cancel_inverses(x, anyset)
    _ = phase_fold(x, anyset)
    _ = cancel_inverses(x, anyset)
    assert_equiv(c, x, "phase_fold toffoli^2")
    assert_true(x.t_count() < t0, "phase folding reduces T-count")
    assert_equal(x.t_count(), 0)
    # textbook: parity x0^x1 rotated twice
    var p = Circuit(2)
    p.cx(0, 1)
    p.t(1)
    p.cx(0, 1)
    p.h(0)
    p.h(0)
    p.cx(0, 1)
    p.t(1)
    p.cx(0, 1)
    var y = p.copy()
    _ = phase_fold(y, anyset)
    assert_equal(y.t_count(), 2)  # different H-separated variables on wire 0 -> not merged
    var q = Circuit(3)
    q.cx(0, 1)
    q.t(1)
    q.h(2)
    q.cx(0, 1)
    q.x(1)
    q.cx(0, 1)
    q.tdg(1)  # parity x0^x1 ^1 -> folds with the first t
    q.cx(0, 1)
    q.rzz(0.3, 0, 1)  # parity x0^x1 again -> folds into the anchor, one 2q gate less
    q.rz(0.5, 1)
    var z = q.copy()
    var removed = phase_fold(z, anyset)
    assert_true(removed >= 2)
    assert_equal(z.num_2q_gates(), q.num_2q_gates() - 1)  # rzz folded away
    assert_equiv(q, z, "phase_fold parity example")
    # random Clifford+T: T-count never increases, equivalence holds
    var total_before = 0
    var total_after = 0
    for seed in range(6):
        var r = clifford_t_circuit(6, 120, UInt64(seed + 40))
        var f = r.copy()
        _ = phase_fold(f, anyset)
        assert_equiv(r, f, "phase_fold clifford+t seed " + String(seed))
        assert_true(f.t_count() <= r.t_count())
        total_before += r.t_count()
        total_after += f.t_count()
    print("  phase_fold ok: toffoli^2 T 14 ->", x.t_count(), "; random Clifford+T T", total_before, "->", total_after)


def test_kak() raises:
    """Weyl/KAK synthesis: minimal CX counts and exactness on special + random blocks."""
    var rng = Rng(21)
    var fails = 0
    var counts = List[Int](length=4, fill=0)
    for trial in range(400):
        var c = Circuit(2)
        var kind = trial % 8
        if kind == 0:
            c.cx(0, 1)
            c.cx(0, 1)  # identity -> 0
        elif kind == 1:
            c.swap(0, 1)
            c.cx(1, 0)  # swap.cx -> 2
        elif kind == 2:
            c.h(0)
            c.cz(0, 1)
            c.rx(rand_angle(rng), 1)  # one cx
        elif kind == 3:
            c.rzz(rand_angle(rng), 0, 1)
            c.rxx(rand_angle(rng), 0, 1)  # c = 0 -> 2
        elif kind == 4:
            c.iswap(0, 1)
            c.u(rand_angle(rng), rand_angle(rng), rand_angle(rng), 0)
        else:
            for _ in range(6):
                c.u(rand_angle(rng), rand_angle(rng), rand_angle(rng), rng.below(2))
                if rng.uniform() < 0.5:
                    c.cx(0, 1)
                else:
                    c.add(Op.CRY + rng.below(5), 1, 0, -1, rand_angle(rng))
        var u = gate_m4(Instruction.make(Op.I, 0), 0, 1)
        for g in c.ops:
            u = gate_m4(g, 0, 1) @ u
        # continuous-entangler mode: one rxx/ryy/rzz per non-zero Weyl coordinate
        var outn = List[Instruction]()
        var phn = 0.0
        var kn = kak_synthesize(u, 0, 1, outn, phn, True)
        if kn < 0:
            fails += 1
        else:
            var rn = Circuit(2)
            for g in outn:
                rn.append(g)
            rn.global_phase = phn
            assert_equiv(c, rn, "kak native trial " + String(trial))
            assert_true(kn <= 3)
        # native iSWAP mode: 0/1/2 iSWAPs when a Weyl coordinate vanishes, else 4
        var outi = List[Instruction]()
        var phi = 0.0
        var ki = kak_synthesize(u, 0, 1, outi, phi, False, True)
        if ki < 0:
            fails += 1
        else:
            var ri = Circuit(2)
            for g in outi:
                ri.append(g)
            ri.global_phase = phi
            assert_equiv(c, ri, "kak iswap trial " + String(trial))
            if kind == 1:
                assert_true(ki <= 1)  # swap.cx is iSWAP-class
            if kind == 4:
                assert_equal(ki, 1)
        var out = List[Instruction]()
        var ph = 0.0
        var k = kak_synthesize(u, 0, 1, out, ph)
        if k < 0:
            fails += 1
            continue
        counts[k] += 1
        var r = Circuit(2)
        for g in out:
            r.append(g)
        r.global_phase = ph
        assert_equiv(c, r, "kak trial " + String(trial))
        if kind == 0: assert_equal(k, 0)
        if kind == 1: assert_equal(k, 2)
        if kind == 2: assert_true(k <= 1)
        if kind == 3: assert_true(k <= 2)
        assert_equal(r.num_2q_gates(), k)
    assert_equal(fails, 0)
    print("  kak ok: cx-count histogram", counts[0], counts[1], counts[2], counts[3])


def random_pauli_labels(n: Int, m: Int, seed: UInt64) -> List[String]:
    var rng = Rng(seed)
    var out = List[String]()
    var letters: List[String] = ["I", "X", "Y", "Z"]
    for _ in range(m):
        var s = String()
        for _ in range(n):
            s += letters[rng.below(4)]
        out.append(s)
    sort(out)
    return out^


def test_pauli_simp() raises:
    """Gadget resynthesis: exact (with phase) and fewer 2q gates on Trotter circuits."""
    var saved_total = 0
    for seed in range(8):
        var n = 3 + seed % 4
        var labels = random_pauli_labels(n, 6 + 3 * seed, UInt64(seed + 300))
        var rng = Rng(UInt64(seed + 900))
        var coeffs = List[Float64]()
        for _ in range(len(labels)):
            coeffs.append(rng.uniform() - 0.5)
        var c = hamiltonian_circuit_from_labels(labels, coeffs, 1.0)
        # Clifford-angle / named rotations and 2q rotations mixed in
        c.t(0)
        c.cp(0.3, 0, n - 1)
        c.rzz(0.7, 1, 2)
        c.crx(0.4, 2, 0)
        c.u(0.3, 0.2, 0.1, 1)
        c.rz(pi / 2, 1)
        c.ry(-pi, 0)
        var x = c.copy()
        var saved = pauli_resynthesize(x)
        saved_total += saved
        assert_equiv(c, x, "pauli_simp seed " + String(seed))
    # random circuits: accepted or rejected, always exact
    for seed in range(6):
        var c = random_circuit(4 + seed % 3, 40, UInt64(seed + 4242))
        var x = c.copy()
        _ = pauli_resynthesize(x)
        assert_equiv(c, x, "pauli_simp random seed " + String(seed))
    assert_true(saved_total > 0)
    print("  pauli_simp ok (2q saved on Trotter circuits:", saved_total, ")")


def test_routing() raises:
    var targets = List[Target]()
    targets.append(Target.line(8))
    targets.append(Target.ring(7))
    targets.append(Target.grid(3, 3))
    targets.append(Target.heavy_hex(3))
    targets.append(Target.tree(9))
    var layouts: List[String] = ["trivial", "dense", "sabre", "auto"]
    var total_swaps = 0
    for ti in range(len(targets)):
        var t = targets[ti].copy()
        for li in range(len(layouts)):
            var seed = ti * 10 + li
            var n = min(5 + seed % 3, t.num_qubits)
            var c = random_circuit(n, 40, UInt64(seed + 77))
            var spec = "route(sabre,layout=" + layouts[li] + ",trials=3,seed=" + String(seed) + ")"
            var res = compile_spec(c, t, spec)
            assert_conforms(res.circuit, t, spec + " on " + t.name)
            assert_routed_equiv(c, res, spec + " on " + t.name)
            total_swaps += res.num_swaps()
    # VF2: a line-shaped interaction graph embeds into a grid with zero swaps
    var line = Circuit(6, name="chain")
    for q in range(5):
        line.cx(q, q + 1)
        line.rz(0.1 * Float64(q), q + 1)
    line.cz(4, 5)
    var g = Target.grid(3, 3)
    var res = compile_spec(line, g, "route(sabre,layout=vf2)")
    assert_equal(res.num_swaps(), 0)
    assert_routed_equiv(line, res, "vf2 chain on grid")
    # star pre-routing: exact up to the returned wire permutation, star -> path
    var bv = Circuit(7, name="star")
    for q in range(7):
        bv.h(q)
    for q in range(6):
        bv.cx(q, 6)
        bv.rz(0.1 * Float64(q + 1), 6)
    bv.cz(2, 4)
    var sc = Circuit(1)
    var wm = List[Int]()
    var ins = star_preroute(bv, 3, sc, wm)
    assert_equal(ins, 5)
    var sres = CompileResult(sc)
    for q in range(7):
        sres.initial_layout.append(q)
    sres.final_layout = wm.copy()
    assert_routed_equiv(bv, sres, "star_preroute")
    var bvr = compile_spec(bv, Target.line(7, "sc_cx"), "route(sabre,trials=4),kak,basis,check")
    assert_routed_equiv(bv, bvr, "star routed on line")
    # swap sliding: SWAP.G == G'.SWAP, exact, stays on the coupling map
    var ln = Target.line(4)
    for seed in range(5):
        var rc = Circuit(4, name="slide")
        var rng = Rng(UInt64(seed + 31))
        for _ in range(30):
            var r = rng.below(5)
            var q = rng.below(3)
            if r == 0: rc.swap(q, q + 1)
            elif r == 1: rc.cx(q, q + 1)
            elif r == 2: rc.cx(q + 1, q)
            elif r == 3: rc.rz(rng.uniform(), q)
            else: rc.ry(rng.uniform(), q + 1)
        var sl = rc.copy()
        _ = slide_swaps(sl.ops, ln)
        assert_conforms(sl, ln, "slide_swaps")
        assert_equiv(rc, sl, "slide_swaps seed " + String(seed))
    # small routed circuit: the routing SWAP merges into a CX block (Qiskit: 7)
    var tof = Circuit(3, name="toffoli")
    tof.x(0)
    tof.x(1)
    tof.ccx(0, 1, 2)
    var tr = transpile(tof, Target.line(3, "rz,sx,x,cx"), 2, seed=12345)
    assert_routed_equiv(tof, tr, "toffoli line")
    assert_true(tr.circuit.num_2q_gates() <= 7)
    # measurements follow the routed qubits: deterministic outcome preserved
    var m = Circuit(5, name="det")
    m.x(0)
    m.x(3)
    m.cx(0, 4)
    m.cx(3, 1)
    m.swap(1, 2)
    m.cx(4, 2)
    m.measure_all()
    var mr = transpile(m, Target.line(5, "sc_cz"), 2)
    var s1 = sample(m, 50, seed=3)
    var s2 = sample(mr.circuit, 50, seed=3)
    assert_equal(s1.most_frequent(), s2.most_frequent())
    assert_equal(s2.counts.get(s2.most_frequent(), 0), 50)
    # determinism
    var c = random_circuit(7, 80, 99)
    var r1 = transpile(c, Target.heavy_hex(3, "sc_ecr"), 3, seed=5)
    var r2 = transpile(c, Target.heavy_hex(3, "sc_ecr"), 3, seed=5)
    assert_equal(String(r1.circuit), String(r2.circuit))
    print("  routing ok (total swaps", total_swaps, ")")


def test_presets() raises:
    var targets = List[Target]()
    targets.append(Target.all_to_all(7))
    targets.append(Target.all_to_all(7, "generic"))
    targets.append(Target.line(7, "sc_cx"))
    targets.append(Target.superconducting_grid(3, 3))
    targets.append(Target.superconducting_heavy_hex(3, "sc_ecr"))
    targets.append(Target.superconducting_grid_iswap(2, 4))
    targets.append(Target.trapped_ion(7))
    targets.append(Target.trapped_ion(7, "ion_zz"))
    targets.append(Target.neutral_atom(3, 3))
    targets.append(Target.ring(8, "rz,ry,rzz"))
    for ti in range(len(targets)):
        var t = targets[ti].copy()
        var c = random_circuit(5 + ti % 3, 50, UInt64(500 + ti))
        for level in range(4):
            var res = transpile(c, t, level, seed=ti)
            assert_conforms(res.circuit, t, "O" + String(level) + " " + t.name)
            assert_routed_equiv(c, res, "O" + String(level) + " " + t.name)
        var r3 = transpile(c, t, 3, seed=ti)
        print("   ", t.name, "basis", r3.basis, ": 2q", c.num_2q_gates(), "->", r3.circuit.num_2q_gates(),
              "| O1", transpile(c, t, 1).circuit.num_2q_gates(), "O2", transpile(c, t, 2).circuit.num_2q_gates())
    print("  presets O0-O3 ok on", len(targets), "targets")


def qft_circuit(n: Int) -> Circuit:
    var c = Circuit(n, name="qft" + String(n))
    for j in range(n):
        c.h(j)
        for k in range(j + 1, n):
            c.cp(pi / Float64(1 << (k - j)), k, j)
    for j in range(n // 2):
        c.swap(j, n - 1 - j)
    return c^


def test_structured() raises:
    """QFT and QAOA-style blocks on every modality (exact, through layouts)."""
    var q = qft_circuit(9)
    var qa = Circuit(8, name="qaoa")
    for i in range(8):
        qa.h(i)
    for layer in range(2):
        for i in range(8):
            qa.cx(i, (i + 3) % 8)
            qa.rz(0.4 + 0.1 * Float64(layer), (i + 3) % 8)
            qa.cx(i, (i + 3) % 8)
        for i in range(8):
            qa.rx(0.7, i)
    var specs: List[String] = [
        "all_to_all:9/generic", "line:9/sc_cx", "grid:3x3/sc_cz", "heavy_hex:3/sc_ecr",
        "superconducting_grid_iswap:3x3", "trapped_ion:9", "trapped_ion:9/ion_zz", "neutral_atom:3x3",
    ]
    for s in specs:
        var t = parse_target(s)
        for lvl in [1, 2, 3]:
            var r = transpile(q, t, lvl)
            assert_conforms(r.circuit, t, "qft O" + String(lvl) + " " + s)
            assert_routed_equiv(q, r, "qft O" + String(lvl) + " " + s)
            var r2 = transpile(qa, t, lvl)
            assert_conforms(r2.circuit, t, "qaoa O" + String(lvl) + " " + s)
            assert_routed_equiv(qa, r2, "qaoa O" + String(lvl) + " " + s)
        var rq = transpile(qa, t, 2)
        print("   ", s, ": qft9 2q", transpile(q, t, 2).circuit.num_2q_gates(), "| qaoa8 2q", qa.num_2q_gates(), "->", rq.circuit.num_2q_gates())
    print("  structured circuits ok")


def test_dynamic() raises:
    """Mid-circuit measurement, feed-forward, reset and barriers survive every level."""
    var c = Circuit(5, name="dyn")
    c.x(0)
    c.h(3)
    c.h(3)
    c.measure(0, 0)
    c.x(1)
    c.c_if(0, 1)  # fires: q1 -> 1
    c.cx(1, 4)
    c.barrier()
    c.t(2)
    c.tdg(2)
    c.reset(0)
    c.cx(4, 2)
    c.swap(2, 3)
    c.measure(1, 1)
    c.measure(2, 2)
    c.measure(3, 3)
    c.measure(4, 4)
    c.measure(0, 5)
    var ref_ = sample(c, 40, seed=5)
    assert_equal(ref_.counts.get(ref_.most_frequent(), 0), 40)
    for s in ["line:5/sc_cz", "heavy_hex:3/sc_ecr", "trapped_ion:5", "neutral_atom:2x3"]:
        var t = parse_target(s)
        for lvl in range(4):
            var r = transpile(c, t, lvl)
            assert_conforms(r.circuit, t, "dynamic O" + String(lvl) + " " + s)
            assert_equal(r.circuit.count(Op.MEASURE), 6)
            assert_equal(r.circuit.count(Op.RESET), 1)
            var out = sample(r.circuit, 40, seed=5)
            assert_equal(out.most_frequent(), ref_.most_frequent())
            assert_equal(out.counts.get(out.most_frequent(), 0), 40)
    print("  dynamic circuits ok (", ref_.most_frequent(), ")")


def test_conditioned() raises:
    """Classically conditioned cx / cz / ccx (bit and register conditions) are
    routed like any gate and keep their condition through basis translation."""
    var c = Circuit(6, name="cond")
    c.x(0)
    c.x(2)
    c.measure(0, 0)
    c.measure(2, 1)
    c.cx(0, 5)
    c.c_if(0, 1)  # fires: q5 = 1
    c.cz(1, 4)
    c.c_if(0, 1)  # fires (phase only)
    c.ccx(0, 2, 3)
    c.c_if_register([0, 1], [1, 1])  # fires: q3 = 1
    c.cx(2, 1)
    c.c_if(0, 0)  # does not fire
    c.cx(5, 1)
    c.c_if_register([0, 1], [1, 0])  # does not fire
    c.x(4)
    c.cx(4, 0)
    c.c_if(1, 1)  # fires: q0 back to 0
    for q in range(6):
        c.measure(q, 2 + q)
    var ref_ = sample(c, 30, seed=9)
    var key = ref_.most_frequent()
    assert_equal(ref_.counts.get(key, 0), 30)
    for s in ["line:6/sc_cz", "line:6/generic", "heavy_hex:3/sc_ecr", "ring:6/ion_zz", "grid:2x3/atom_cz"]:
        var t = parse_target(s)
        for lvl in range(4):
            var r = transpile(c, t, lvl)
            assert_conforms(r.circuit, t, "conditioned O" + String(lvl) + " " + s)
            var ncond = 0
            for g in r.circuit.ops:
                if g.cond >= 0:
                    ncond += 1
                    if g.cond_reg >= 0:
                        assert_true(g.cond_reg < len(r.circuit.cond_table))
            assert_true(ncond >= 5)
            var out = sample(r.circuit, 30, seed=9)
            assert_equal(out.most_frequent(), key)
            assert_equal(out.counts.get(key, 0), 30)
    print("  conditioned gates ok (", key, ")")


def test_symbolic() raises:
    var c = Circuit(3, name="ansatz")
    var th = c.parameters("t", 3)
    c.ry(th[0], 0)
    c.rz(th[1], 1)
    c.cx(0, 1)
    c.rz(th[1] * -1.0, 1)
    c.rz(th[2], 1)
    c.crz(th[2], 1, 2)
    c.cp(th[0], 0, 2)
    c.rxx(th[1], 0, 2)
    c.rx(th[2] * 0.5, 2)
    c.p(th[0], 2)
    var vals: List[Float64] = [0.3, -1.1, 0.8]
    for b in basis_list():
        var t = Target.all_to_all(3, b)
        var res = compile_spec(c, t, "remove_identities,cancel,commute,basis,cancel,commute,check")
        assert_true(res.circuit.is_parametric())
        assert_equiv(c.bind(vals), res.circuit.bind(vals), "symbolic " + b, exact_phase=not res.symbolic_phase_dropped)
    print("  symbolic parameters ok")


def test_pipeline_parsing() raises:
    var spec = " remove_identities, cancel(64) ; merge,commute,fuse1q(rz,sx,x),phase_fold,route(sabre,layout=dense,seed=3),basis(rz,sx,x,cx),fixpoint(cancel,commute,max=3),check"
    var ps = parse_pipeline(spec)
    assert_equal(len(ps), 10)
    assert_equal(pipeline_to_string(ps), "remove_identities,cancel(64),merge,commute,fuse1q(rz,sx,x),phase_fold,route(sabre,layout=dense,seed=3),basis(rz,sx,x,cx),fixpoint(cancel,commute,max=3),check")
    assert_true(describe_passes().find("phase_fold") >= 0)
    for lvl in range(4):
        _ = parse_pipeline(preset_pipeline(lvl, Target.heavy_hex(3, "sc_cz")))
    try:
        _ = parse_pipeline("remove_identities,cancle")
        assert_true(False)
    except e:
        var m = String(e)
        expect_error("QM-E600", m)
        assert_true(m.find("Did you mean 'cancel'") >= 0 and m.find("valid passes are") >= 0)
    try:
        _ = parse_pipeline("cancel,route(sabre")
        assert_true(False)
    except e:
        expect_error("QM-E601", String(e))
    try:
        _ = parse_pipeline("route(sabre,layout=magic)")
        assert_true(False)
    except e:
        expect_error("QM-E602", String(e))
    try:
        _ = parse_pipeline("route(sabre,speed=9)")
        assert_true(False)
    except e:
        expect_error("QM-E602", String(e))
    try:
        _ = parse_pipeline("merge(3)")
        assert_true(False)
    except e:
        expect_error("QM-E602", String(e))
    try:
        _ = parse_pipeline("basis(rz,zz_top)")
        assert_true(False)
    except e:
        expect_error("QM-E610", String(e))
    try:
        _ = preset_pipeline(5, Target.line(3))
        assert_true(False)
    except e:
        expect_error("QM-E609", String(e))
    var wide = random_circuit(6, 10, 1)
    try:
        _ = transpile(wide, Target.line(4), 1)
        assert_true(False)
    except e:
        expect_error("QM-E605", String(e))
    try:
        _ = compile_spec(wide, Target.all_to_all(6, "rz,sx,x"), "basis")
        assert_true(False)
    except e:
        expect_error("QM-E603", String(e))
    var oneq = Circuit(1)
    oneq.h(0)
    try:
        _ = compile_spec(oneq, Target.all_to_all(1, "rz,cx"), "basis")
        assert_true(False)
    except e:
        expect_error("QM-E604", String(e))
    var bad = Circuit(3)
    bad.cx(0, 2)
    try:
        _ = compile_spec(bad, Target.line(3), "check")
        assert_true(False)
    except e:
        expect_error("QM-E615", String(e))
    print("  pipeline parsing + error codes ok")


def main() raises:
    test_targets()
    test_basis_rules()
    test_cancellation_examples()
    test_passes_random()
    test_phase_fold()
    test_kak()
    test_pauli_simp()
    test_routing()
    test_presets()
    test_structured()
    test_dynamic()
    test_conditioned()
    test_symbolic()
    test_pipeline_parsing()
    print("test_compiler OK")
