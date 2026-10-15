"""Learned-policy routing (qumo/compiler/ai_route.mojo).

Fixture-free mechanics with an inline random-weight model; parity against the
committed router model's embedded test vectors (obs -> logits and full routed
episodes) when models/ai_transpiler/route holds one; statevector equivalence
of routed circuits through the layouts on grid / heavy-hex / line / ring;
fallback termination on adversarial policies; portfolio never worse than
SABRE; determinism; layout modes; QM-E617 / QM-E620 diagnostics.
"""

from std.testing import assert_equal, assert_true, assert_false
from std.os.path import exists
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_arity, op_is_unitary
from qumo.sim.runner import evolve
from qumo.sim.rng import Rng
from qumo.ai.json import parse_json
from qumo.compiler.target import Target
from qumo.compiler.routing import RouteOptions, route_circuit
from qumo.compiler.ai_route import (
    AIRouteOptions, ai_route, ai_route_with_model, routed_cost, RouteModel, load_route_model,
    find_route_model, RProblem, Episode, episode_max_steps, verify_episode, MODE_POLICY, MODE_HEURISTIC,
    MODE_SCORED, OBS_SIZE, NUM_ACTIVE_SWAPS,
)
from qumo.compiler.policy import model_search_dirs
from qumo.compiler.basis import unroll_3q


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


def rand_circ(n: Int, ng: Int, seed: UInt64) -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n)
    var two: List[Int] = [Op.CX, Op.CZ, Op.SWAP, Op.CP, Op.RZZ, Op.CX, Op.CX, Op.ISWAP]
    for _ in range(ng):
        var u = rng.uniform()
        if u < 0.35:
            var q = rng.below(n)
            if rng.uniform() < 0.5:
                c.h(q)
            else:
                c.rz(rng.uniform() * 3.0, q)
        elif u < 0.38 and n >= 3:
            var a = rng.below(n)
            var b = (a + 1 + rng.below(n - 1)) % n
            var d = (b + 1 + rng.below(n - 1)) % n
            while d == a or d == b:
                d = (d + 1) % n
            c.ccx(a, b, d)
        else:
            var a = rng.below(n)
            var b = (a + 1 + rng.below(n - 1)) % n
            c.add(two[rng.below(len(two))], a, b, -1, rng.uniform() * 2.0, 0.0, 0.0)
    return c^


def routed_equiv(orig: Circuit, cc: Circuit, il: List[Int], fl: List[Int], label: String) raises:
    """Routed `cc` on N physical qubits == orig through the layouts (exact global phase);
    same logic as tests/test_compiler.mojo assert_routed_equiv."""
    var n = orig.num_qubits
    var N = cc.num_qubits
    for s in range(2):
        var p = prep_circuit(n, UInt64(2000 + s))
        var pa = p.copy()
        pa.compose(orig)
        var sa = evolve(pa)
        var qmap = List[Int]()
        for l in range(n):
            qmap.append(il[l])
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
                    j |= 1 << fl[l]
            re += sa.re[i] * sb.re[j] + sa.im[i] * sb.im[j]
            im += sa.re[i] * sb.im[j] - sa.im[i] * sb.re[j]
        if abs(re - 1) > 1e-7 or abs(im) > 1e-7:
            print("ROUTED MISMATCH:", label, re, im)
        assert_true(abs(re - 1) < 1e-7 and abs(im) < 1e-7, label + ": routed equivalence")


def conforms(c: Circuit, t: Target) -> Bool:
    for g in c.ops:
        if op_is_unitary(g.kind) and op_arity(g.kind) == 2 and not t.are_connected(g.q0, g.q1):
            return False
        if op_is_unitary(g.kind) and op_arity(g.kind) == 3:
            return False
    return True


def expect_error(code: String, msg: String) raises:
    if not msg.startswith(code):
        print("expected", code, "got:", msg)
    assert_true(msg.startswith(code), "error code " + code)
    assert_true(msg.find("Hint:") >= 0, "error has hint: " + msg)


def route(c: Circuit, t: Target, ao: AIRouteOptions, model: RouteModel, mut out: Circuit,
          mut il: List[Int], mut fl: List[Int], mut note: String) raises -> Bool:
    var ro = RouteOptions()
    ro.seed = ao.seed
    return ai_route_with_model(c, t, ro, ao, model, out, il, fl, note)


def test_equivalence(model: RouteModel) raises:
    var targets = List[Target]()
    targets.append(Target.grid(3, 3))
    targets.append(Target.heavy_hex(3))
    targets.append(Target.line(7))
    targets.append(Target.ring(8))
    var names: List[String] = ["grid3x3", "heavy_hex3", "line7", "ring8"]
    var pols: List[String] = ["policy", "heuristic", "scored"]
    var count = 0
    for ti in range(len(targets)):
        var t = targets[ti].copy()
        for ci in range(2):
            var n = min(7, t.num_qubits) - ci
            var c = rand_circ(n, 36 + 14 * ci, UInt64(100 * ti + ci))
            for pi_ in range(3):
                var ao = AIRouteOptions()
                ao.level = 1
                ao.seed = 11 + ci
                ao.policy = "x" if pols[pi_] == "policy" else pols[pi_]
                ao.portfolio = False
                var out = Circuit(1)
                var il = List[Int]()
                var fl = List[Int]()
                var note = String()
                var ok = route(c, t, ao, model, out, il, fl, note)
                var lab = names[ti] + "/c" + String(ci) + "/" + pols[pi_]
                assert_true(ok, lab + ": ai_route returned a result")
                assert_true(note.find("ai_route: policy=") >= 0, lab + ": policy result (" + note + ")")
                assert_true(conforms(out, t), lab + ": conforms")
                assert_equal(out.num_qubits, t.num_qubits, lab + ": physical width")
                routed_equiv(c, out, il, fl, lab)
                count += 1
    print("  statevector-equivalent routes:", count, "(grid3x3, heavy_hex3, line7, ring8; policy / heuristic / scored)")


def test_layout_modes(model: RouteModel) raises:
    var t = Target.grid(3, 3)
    var c = rand_circ(7, 40, 7)
    var modes: List[String] = ["optimize", "improve", "keep"]
    for mi in range(3):
        var ao = AIRouteOptions()
        ao.level = 1
        ao.policy = "x"
        ao.portfolio = False
        ao.layout_mode = modes[mi]
        if mi >= 1:
            ao.seed_layout = [8, 7, 6, 5, 4, 3, 2]
        var out = Circuit(1)
        var il = List[Int]()
        var fl = List[Int]()
        var note = String()
        assert_true(route(c, t, ao, model, out, il, fl, note), modes[mi])
        routed_equiv(c, out, il, fl, "layout_mode=" + modes[mi])
        if mi == 2:
            for w in range(7):
                assert_equal(il[w], 8 - w, "keep: initial layout is the given one")
    # keep without a layout: trivial initial layout
    var ao = AIRouteOptions()
    ao.level = 1
    ao.policy = "heuristic"
    ao.portfolio = False
    ao.layout_mode = "keep"
    var out = Circuit(1)
    var il = List[Int]()
    var fl = List[Int]()
    var note = String()
    assert_true(route(c, t, ao, model, out, il, fl, note), "keep trivial")
    for w in range(7):
        assert_equal(il[w], w, "keep: trivial layout")
    routed_equiv(c, out, il, fl, "keep trivial")
    print("  layout_mode optimize / improve / keep OK")


def test_portfolio_and_determinism(model: RouteModel) raises:
    var targets = List[Target]()
    targets.append(Target.grid(3, 3))
    targets.append(Target.heavy_hex(3))
    targets.append(Target.line(9))
    var wins = 0
    for ti in range(3):
        var t = targets[ti].copy()
        for k in range(3):
            var c = rand_circ(8, 50 + 10 * k, UInt64(500 + 10 * ti + k))
            var ro = RouteOptions()
            ro.seed = 3
            var sc = c.copy()
            _ = unroll_3q(sc)
            var sil = List[Int]()
            var sfl = List[Int]()
            var snote = String()
            _ = route_circuit(sc, t, ro, sil, sfl, snote)
            var cs = routed_cost(sc, t)
            var ao = AIRouteOptions()
            ao.level = 1
            ao.seed = 3
            ao.policy = "x"
            ao.portfolio = True
            var out = Circuit(1)
            var il = List[Int]()
            var fl = List[Int]()
            var note = String()
            assert_true(ai_route_with_model(c, t, ro, ao, model, out, il, fl, note), "portfolio")
            var ca = routed_cost(out, t)
            assert_true(ca[0] < cs[0] or (ca[0] == cs[0] and ca[1] <= cs[1]), "portfolio never worse than SABRE: " + note)
            if note.find("policy won") >= 0:
                wins += 1
            routed_equiv(c, out, il, fl, "portfolio")
            # determinism: same seed -> identical circuit and layouts
            var out2 = Circuit(1)
            var il2 = List[Int]()
            var fl2 = List[Int]()
            var note2 = String()
            assert_true(ai_route_with_model(c, t, ro, ao, model, out2, il2, fl2, note2), "rerun")
            assert_equal(len(out.ops), len(out2.ops), "deterministic length")
            for i in range(len(out.ops)):
                assert_true(out.ops[i].kind == out2.ops[i].kind and out.ops[i].q0 == out2.ops[i].q0 and out.ops[i].q1 == out2.ops[i].q1, "deterministic op")
            for w in range(len(il)):
                assert_equal(il[w], il2[w], "deterministic init layout")
                assert_equal(fl[w], fl2[w], "deterministic final layout")
            assert_equal(note, note2, "deterministic note")
    print("  portfolio never worse than SABRE on 9 jobs (policy won", wins, "); deterministic reruns")


def test_fallback_and_fences() raises:
    # adversarial policy: constant logits favour slot 0 (a random active swap)
    var m = RouteModel.random(16, 1)
    for i in range(len(m.emb)):
        m.emb[i] = 0.0
    for i in range(len(m.w1)):
        m.w1[i] = 0.0
    for k in range(NUM_ACTIVE_SWAPS):
        m.b1[k] = 1.0 if k == 0 else 0.0
    var t = Target.line(10)
    var c = Circuit(10, 4)
    for i in range(10):
        for j in range(i + 1, 10):
            c.cx(i, j)
            c.cx(j, i)
    c.measure(0, 0)
    c.barrier()
    c.measure(9, 1)
    c.cz(0, 9)
    c.c_if(0, 1)
    c.measure(4, 2)
    var prob = RProblem(c.ops, 10)
    var l = List[Int]()
    for i in range(10):
        l.append(i)
    for s in range(3):
        var ep = Episode(prob, t, l, UInt64(s), True)
        ep.run(prob, t, m, MODE_POLICY, 1.0, episode_max_steps(prob, t))
        assert_false(ep.failed, "adversarial policy episode terminates")
        assert_true(ep.finished(prob), "all gates executed")
        assert_true(ep.n_fallback > 0, "fallback used")
        var why = String()
        assert_true(verify_episode(prob, ep, t, why), "verified: " + why)
    var ao = AIRouteOptions()
    ao.level = 1
    ao.policy = "x"
    ao.portfolio = False
    var out = Circuit(1)
    var il = List[Int]()
    var fl = List[Int]()
    var note = String()
    assert_true(ai_route_with_model(c, t, RouteOptions(), ao, m, out, il, fl, note), "adversarial ai_route")
    assert_true(conforms(out, t), "adversarial conforms")
    assert_equal(out.count(Op.MEASURE), 3, "measurements kept")
    assert_equal(out.count(Op.BARRIER), 1, "barrier kept")
    print("  adversarial policy terminates via fallback; measure / barrier / c_if fences verified")


def test_no_model_and_errors() raises:
    var t = Target.grid(3, 3)
    var c = rand_circ(6, 30, 77)
    var ao = AIRouteOptions()
    ao.policy = "auto"
    var out = Circuit(1)
    var il = List[Int]()
    var fl = List[Int]()
    var note = String()
    # policy=auto without a model: SABRE result, said in the note
    assert_true(ai_route_with_model(c, t, RouteOptions(), ao, RouteModel(), out, il, fl, note), "no model")
    assert_true(note.find("no router model found") >= 0 and note.find("SABRE") >= 0, note)
    routed_equiv(c, out, il, fl, "no model -> SABRE")
    ao.portfolio = False
    assert_false(ai_route_with_model(c, t, RouteOptions(), ao, RouteModel(), out, il, fl, note), "no model, no portfolio: no proposal")
    ao.policy = "off"
    ao.portfolio = True
    assert_true(ai_route_with_model(c, t, RouteOptions(), ao, RouteModel(), out, il, fl, note), "policy=off")
    assert_true(note.find("policy=off") >= 0, note)
    # explicit missing model -> QM-E617
    var ao2 = AIRouteOptions()
    ao2.policy = "no_such_router_model"
    try:
        _ = ai_route(c, t, RouteOptions(), ao2, out, il, fl, note)
        assert_true(False, "missing model must raise")
    except e:
        expect_error("QM-E617", String(e))
    if exists("tests/fixtures/ai/policy/bad_checksum/manifest.json"):
        try:
            _ = load_route_model("tests/fixtures/ai/policy/bad_checksum")
            assert_true(False, "bad checksum must raise")
        except e:
            expect_error("QM-E617", String(e))
    if exists("tests/fixtures/ai/policy/synth/fx_perm_line3/manifest.json"):
        try:
            _ = load_route_model("tests/fixtures/ai/policy/synth/fx_perm_line3")
            assert_true(False, "synthesis model must be rejected")
        except e:
            expect_error("QM-E620", String(e))
    # all-to-all target: nothing to route
    var ta = Target.all_to_all(6)
    assert_true(ai_route(c, ta, RouteOptions(), AIRouteOptions(), out, il, fl, note), "all-to-all")
    assert_equal(len(out.ops), len(c.ops), "all-to-all unchanged")
    print("  no-model / off / QM-E617 / QM-E620 / all-to-all OK")


def test_committed_model() raises:
    """Parity with the committed router model's embedded test vectors."""
    var d = find_route_model("auto", model_search_dirs(""))
    if d.byte_length() == 0:
        print("  no committed router model found: parity test skipped (policy=auto runs SABRE)")
        return
    var m = load_route_model(d)
    var man = parse_json(m.manifest)
    var tv = man.get("test_vectors")
    var ol = tv.get("obs_logits")
    var hid = List[Float64](length=m.hidden, fill=0.0)
    var lg = List[Float64](length=NUM_ACTIVE_SWAPS, fill=0.0)
    var worst = 0.0
    for i in range(len(ol)):
        var v = ol.at(i)
        var obs = List[Int]()
        var oj = v.get("obs")
        for k in range(OBS_SIZE):
            var o = oj.at(k).as_int()
            obs.append(1 if o > 0 else (-1 if o < 0 else 0))
        m.forward(obs, hid, lg)
        var want = v.get("logits")
        for k in range(NUM_ACTIVE_SWAPS):
            worst = max(worst, abs(lg[k] - want.at(k).as_float()))
    assert_true(worst < 1e-6, "obs -> logits parity (max err " + String(worst) + ")")
    var eps = tv.get("episodes")
    for i in range(len(eps)):
        var e = eps.at(i)
        var cj = e.get("circuit")
        var n = cj.get("n").as_int()
        var c = Circuit(n, n)
        var ops = cj.get("ops")
        var mi = 0
        for k in range(len(ops)):
            var o = ops.at(k)
            var kind = o.at(0).as_string()
            var a = o.at(1).as_int()
            var b = o.at(2).as_int()
            if kind == "cx": c.cx(a, b)
            elif kind == "cz": c.cz(a, b)
            elif kind == "swap": c.swap(a, b)
            elif kind == "u2": c.rzz(0.3, a, b)
            elif kind == "u1": c.h(a)
            elif kind == "measure":
                c.measure(a, mi)
                mi += 1
            elif kind == "barrier": c.barrier()
        var tj = e.get("target")
        var ed = List[Int]()
        var ej = tj.get("edges")
        for k in range(len(ej)):
            ed.append(ej.at(k).at(0).as_int())
            ed.append(ej.at(k).at(1).as_int())
        var t = Target.from_edges(tj.get("n").as_int(), ed)
        var lay = List[Int]()
        var lj = e.get("layout")
        for k in range(len(lj)):
            lay.append(lj.at(k).as_int())
        var prob = RProblem(c.ops, n)
        var ep = Episode(prob, t, lay, UInt64(e.get("seed").as_int()), True)
        ep.run(prob, t, m, MODE_POLICY, 1.0, episode_max_steps(prob, t))
        var acts = e.get("actions")
        assert_equal(len(ep.actions), len(acts), "episode " + String(i) + " length")
        for k in range(len(acts)):
            assert_equal(ep.actions[k], acts.at(k).as_int(), "episode action")
        assert_equal(ep.overhead(), e.get("overhead").as_int(), "episode overhead")
        var fj = e.get("final")
        for k in range(len(fj)):
            assert_equal(ep.l2p[k], fj.at(k).as_int(), "episode final layout")
        var why = String()
        assert_true(verify_episode(prob, ep, t, why), why)
    print("  committed model", m.id, ": obs->logits parity (max err", worst, "),", len(eps), "episodes bit-exact")
    # the committed model through the public entry point, verified by statevector
    var t = Target.heavy_hex(3)
    var c = rand_circ(8, 60, 4242)
    var ao = AIRouteOptions()
    ao.level = 1
    var out = Circuit(1)
    var il = List[Int]()
    var fl = List[Int]()
    var note = String()
    assert_true(ai_route(c, t, RouteOptions(), ao, out, il, fl, note), "ai_route with committed model")
    assert_true(note.find(m.id) >= 0, "note names the model: " + note)
    routed_equiv(c, out, il, fl, "committed model on heavy_hex3")


def main() raises:
    var model = RouteModel.random(32, 5)
    test_equivalence(model)
    test_layout_modes(model)
    test_portfolio_and_determinism(model)
    test_fallback_and_fences()
    test_no_model_and_errors()
    test_committed_model()
    print("test_ai_route OK")
