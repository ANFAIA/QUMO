"""Tests for the hybrid heuristic + AI transpiler integration (front 'integrate').

Covers the pass-spec surface of the AI passes (clifford_synth, linear_synth,
perm_synth, pauli_network and the route keys policy / ai_level / layout_mode /
model_dir), their error codes (QM-E602, QM-E609, QM-E616..E620), the pass
catalog, `preset_pipeline_ai` / `transpile_ai` levels 1-3 verified by
statevector through the layouts on grid, heavy-hex, line, ion and
neutral-atom targets, the level-3 portfolio guarantee (never worse than O3 in
2q count), determinism, and a snapshot proving the O0-O3 presets are
byte-identical to the pre-AI pass manager.
"""

from std.math import pi
from std.testing import assert_equal, assert_true, assert_false
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_arity, op_num_params
from qumo.ir.param import Param
from qumo.sim.runner import evolve
from qumo.sim.statevector import StateVector
from qumo.sim.rng import Rng
from qumo.compiler.target import Target, parse_target
from qumo.compiler.pass_manager import (
    CompileResult, PassId, parse_pipeline, pipeline_to_string, describe_passes, pass_name,
    preset_pipeline, preset_pipeline_ai, transpile, transpile_ai, compile_spec,
    conformance_error, synth_options_from_args,
)

# Snapshot of transpile() O0-O3 on 5 circuits x 5 targets, computed with the
# pass manager before the AI passes were added (FNV-1a of the printed circuit,
# pipeline and layouts). Any change means O0-O3 output changed.
comptime SNAPSHOT_TOTAL: UInt64 = 1465841153960109920


def snap_random(n: Int, ngates: Int, seed: UInt64) -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n, name="snap" + String(seed))
    for _ in range(ngates):
        var r = rng.below(10)
        var q = rng.below(n)
        var t = (q + 1 + rng.below(n - 1)) % n
        var a = (rng.uniform() * 2 - 1) * pi
        if r == 0: c.h(q)
        elif r == 1: c.rz(a, q)
        elif r == 2: c.rx(a, q)
        elif r == 3: c.s(q)
        elif r == 4: c.t(q)
        elif r == 5: c.cz(q, t)
        elif r == 6: c.swap(q, t)
        elif r == 7: c.rzz(a, q, t)
        else: c.cx(q, t)
    return c^


def snap_qft(n: Int) -> Circuit:
    var c = Circuit(n, name="qft" + String(n))
    for j in range(n):
        c.h(j)
        for k in range(j + 1, n):
            c.cp(pi / Float64(1 << (k - j)), k, j)
    for j in range(n // 2):
        c.swap(j, n - 1 - j)
    return c^


def snap_clifford(n: Int, ngates: Int, seed: UInt64) -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n, name="cliff")
    for _ in range(ngates):
        var r = rng.below(6)
        var q = rng.below(n)
        var t = (q + 1 + rng.below(n - 1)) % n
        if r == 0: c.h(q)
        elif r == 1: c.s(q)
        elif r == 2: c.sx(q)
        else: c.cx(q, t)
    return c^


def snap_linear(n: Int, ngates: Int, seed: UInt64) -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n, name="lin")
    for _ in range(ngates):
        var q = rng.below(n)
        var t = (q + 1 + rng.below(n - 1)) % n
        if rng.below(5) == 0: c.swap(q, t)
        else: c.cx(q, t)
    return c^


def snap_circuits() -> List[Circuit]:
    var cs = List[Circuit]()
    cs.append(snap_random(6, 60, 11))
    cs.append(snap_qft(6))
    cs.append(snap_clifford(7, 70, 23))
    cs.append(snap_linear(7, 40, 31))
    cs.append(snap_random(5, 45, 47))
    return cs^


def snap_targets() raises -> List[Target]:
    var ts = List[Target]()
    ts.append(parse_target("grid:3x3/sc_cz"))
    ts.append(parse_target("heavy_hex:3/sc_cz"))
    ts.append(parse_target("line:7/sc_cx"))
    ts.append(parse_target("trapped_ion:7"))
    ts.append(parse_target("neutral_atom:3x3"))
    return ts^


def fnv64(s: String) -> UInt64:
    var h: UInt64 = 0xCBF29CE484222325
    var bs = s.as_bytes()
    for i in range(len(bs)):
        h ^= UInt64(Int(bs[i]))
        h *= 0x100000001B3
    return h


def result_text(r: CompileResult) -> String:
    var s = String(r.circuit) + "|" + r.pipeline + "|"
    for x in r.initial_layout:
        s += String(x) + ","
    s += "|"
    for x in r.final_layout:
        s += String(x) + ","
    return s


def snapshot_hash() raises -> UInt64:
    var h: UInt64 = 0
    var cs = snap_circuits()
    var ts = snap_targets()
    for ci in range(len(cs)):
        for ti in range(len(ts)):
            for lvl in range(4):
                var r = transpile(cs[ci], ts[ti], lvl, 7)
                h = h * 1099511628211 + fnv64(result_text(r))
    return h



# ----------------------------------------------------------------------------
# helpers (assert_routed_equiv as in tests/test_compiler.mojo)
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


def assert_routed_equiv(orig: Circuit, res: CompileResult, label: String) raises:
    """Compiled circuit on N physical qubits == orig, through the layouts (exact phase)."""
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


def expect_parse_error(spec: String, code: String) raises:
    try:
        _ = parse_pipeline(spec)
        print("expected", code, "for", spec)
        assert_true(False, "no error for " + spec)
    except e:
        expect_error(code, String(e))


def pauli_rot_circuit(n: Int, terms: Int, seed: UInt64) -> Circuit:
    """Trotter-like Pauli-rotation network (basis change + CX ladder + rz)."""
    var rng = Rng(seed)
    var c = Circuit(n, name="pauli")
    for _ in range(terms):
        var a = rng.below(n)
        var b = (a + 1 + rng.below(n - 1)) % n
        var basis = rng.below(3)
        if basis == 1:
            c.h(a)
            c.h(b)
        elif basis == 2:
            c.sx(a)
        c.cx(a, b)
        c.rz((rng.uniform() * 2 - 1) * pi, b)
        c.cx(a, b)
        if basis == 1:
            c.h(a)
            c.h(b)
        elif basis == 2:
            c.sxdg(a)
        c.rx((rng.uniform() * 2 - 1) * pi, a)
    return c^


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def test_spec_surface() raises:
    assert_equal(PassId.NUM, 19)
    assert_equal(pass_name(PassId.CLIFFORD_SYNTH), "clifford_synth")
    assert_equal(pass_name(PassId.LINEAR_SYNTH), "linear_synth")
    assert_equal(pass_name(PassId.PERM_SYNTH), "perm_synth")
    assert_equal(pass_name(PassId.PAULI_NETWORK), "pauli_network")
    var spec = "ai_clifford(method=exact,max_qubits=4),ai_linear(greedy,direction=backward,commute=0),ai_permutation(min_block=2,budget=5000),ai_pauli_network(model=off,beams=4,searches=2,metric=cost,strict=1,reps=3,model_dir=/tmp/models,seed=3),route(sabre,policy=auto,ai_level=3,layout_mode=improve,model_dir=models/ai_transpiler)"
    var ps = parse_pipeline(spec)
    assert_equal(len(ps), 5)
    assert_equal(pipeline_to_string(ps), "clifford_synth(method=exact,max_qubits=4),linear_synth(greedy,direction=backward,commute=0),perm_synth(min_block=2,budget=5000),pauli_network(model=off,beams=4,searches=2,metric=cost,strict=1,reps=3,model_dir=/tmp/models,seed=3),route(sabre,policy=auto,ai_level=3,layout_mode=improve,model_dir=models/ai_transpiler)")
    # options: IBM collector defaults, overridden by arguments
    var o = synth_options_from_args(PassId.LINEAR_SYNTH, ps[1].args, 11)
    assert_equal(o.kind, 1)
    assert_equal(o.method, "greedy")
    assert_equal(o.direction, "backward")
    assert_false(o.commute)
    assert_equal(o.seed, 11)
    assert_equal(o.max_qubits, 9)
    assert_equal(o.min_block, 4)
    var o2 = synth_options_from_args(PassId.PAULI_NETWORK, ps[3].args, 11)
    assert_equal(o2.kind, 3)
    assert_equal(o2.max_qubits, 6)
    assert_equal(o2.model, "off")
    assert_true(o2.strict)
    assert_equal(o2.seed, 3)
    assert_equal(o2.model_dir, "/tmp/models")
    var o3 = synth_options_from_args(PassId.PERM_SYNTH, List[String](), 7)
    assert_equal(o3.kind, 2)
    assert_equal(o3.max_qubits, 12)
    var o4 = synth_options_from_args(PassId.CLIFFORD_SYNTH, List[String](), 7)
    assert_equal(o4.kind, 0)
    assert_equal(o4.max_qubits, 9)
    assert_equal(o4.min_block, 2)
    # errors
    expect_parse_error("linear_synth(method=magic)", "QM-E616")
    expect_parse_error("clifford_synth(quantum)", "QM-E616")
    expect_parse_error("perm_synth(model=)", "QM-E616")
    expect_parse_error("perm_synth(max_qubits=x)", "QM-E602")
    expect_parse_error("perm_synth(max_qubits=0)", "QM-E602")
    expect_parse_error("perm_synth(max_qubits=99)", "QM-E618")
    expect_parse_error("pauli_network(direction=sideways)", "QM-E602")
    expect_parse_error("pauli_network(commute=maybe)", "QM-E602")
    expect_parse_error("pauli_network(metric=fidelity)", "QM-E602")
    expect_parse_error("linear_synth(speed=3)", "QM-E602")
    expect_parse_error("route(sabre,ai_level=4)", "QM-E602")
    expect_parse_error("route(sabre,layout_mode=random)", "QM-E602")
    expect_parse_error("route(sabre,policy=)", "QM-E616")
    expect_parse_error("route(sabre,speed=9)", "QM-E602")
    try:
        _ = parse_pipeline("linear_synt")
        assert_true(False)
    except e:
        var m = String(e)
        expect_error("QM-E600", m)
        assert_true(m.find("linear_synth") >= 0)
    try:
        _ = preset_pipeline_ai(0, Target.line(3))
        assert_true(False)
    except e:
        expect_error("QM-E609", String(e))
    try:
        _ = preset_pipeline_ai(4, Target.line(3))
        assert_true(False)
    except e:
        expect_error("QM-E609", String(e))
    # presets parse on every target family
    var specs: List[String] = ["grid:3x3/sc_cz", "heavy_hex:3/sc_cz", "line:7/sc_cx", "trapped_ion:7", "neutral_atom:3x3", "all_to_all:5"]
    for s in specs:
        var t = parse_target(s)
        for lvl in [1, 2, 3]:
            var p = preset_pipeline_ai(lvl, t)
            _ = parse_pipeline(p)
            if not t.full:
                assert_true(p.find("policy=auto") >= 0, "policy routing in " + p)
            if lvl >= 2:
                assert_true(p.find("linear_synth") >= 0 and p.find("perm_synth") >= 0)
            if lvl == 3:
                assert_true(p.find("clifford_synth") >= 0 and p.find("pauli_network") >= 0)
    # describe catalog lists the new passes
    var d = describe_passes()
    for nm in ["clifford_synth", "linear_synth", "perm_synth", "pauli_network"]:
        assert_true(d.find("\"" + nm + "\"") >= 0, "describe lists " + nm)
    print("  AI pass specs, options and error codes ok")


def test_snapshot_unchanged() raises:
    var h = snapshot_hash()
    if h != SNAPSHOT_TOTAL:
        print("SNAPSHOT CHANGED:", h, "expected", SNAPSHOT_TOTAL)
    assert_true(h == SNAPSHOT_TOTAL, "transpile O0-O3 outputs unchanged")
    print("  O0-O3 outputs byte-identical to the pre-AI snapshot")


def test_transpile_ai_equivalence() raises:
    var specs: List[String] = ["grid:3x3/sc_cz", "heavy_hex:3/sc_cz", "line:7/sc_cx", "trapped_ion:7", "neutral_atom:3x3"]
    var circs = List[Circuit]()
    circs.append(snap_random(6, 50, 101))
    circs.append(snap_clifford(6, 60, 103))
    circs.append(snap_linear(7, 36, 107))
    circs.append(pauli_rot_circuit(5, 8, 109))
    var better = 0
    for s in specs:
        var t = parse_target(s)
        var line = String("    ") + s + ":"
        for ci in range(len(circs)):
            for lvl in [1, 2, 3]:
                var r = transpile_ai(circs[ci], t, lvl, 7)
                var label = "ai" + String(lvl) + " c" + String(ci) + " " + s
                assert_conforms(r.circuit, t, label)
                assert_routed_equiv(circs[ci], r, label)
                if lvl == 3:
                    var o3 = transpile(circs[ci], t, 3, 7)
                    var a2 = r.circuit.num_2q_gates()
                    var b2 = o3.circuit.num_2q_gates()
                    if a2 > b2:
                        print("AI3 WORSE THAN O3:", label, a2, b2)
                    assert_true(a2 <= b2, label + ": ai3 never worse than O3 (2q)")
                    if a2 < b2: better += 1
                    line += " " + String(b2) + "->" + String(a2)
        print(line)
    print("  transpile_ai levels 1-3 statevector-equivalent on", len(specs), "targets; ai3 < O3 on", better, "of", len(specs) * len(circs))


def test_all_to_all_and_synth_passes() raises:
    # all-to-all overload and degraded mode (no routing, synthesis still runs)
    var c = snap_clifford(5, 50, 211)
    var r = transpile_ai(c, 2, 5)
    assert_equal(r.circuit.num_qubits, 5)
    assert_routed_equiv(c, r, "ai2 all-to-all")
    var r3 = transpile_ai(c, 3, 5)
    assert_routed_equiv(c, r3, "ai3 all-to-all")
    # each synthesis pass alone, before and after routing, with every method
    var t = parse_target("grid:3x3/sc_cz")
    var lin = snap_linear(6, 30, 223)
    for m in ["auto", "exact", "greedy", "astar", "policy"]:
        for pn in ["linear_synth", "perm_synth", "clifford_synth", "pauli_network"]:
            var spec = String(pn) + "(method=" + m + ")"
            var pre = compile_spec(lin, Target.all_to_all(6), spec, 3)
            assert_routed_equiv(lin, pre, "pre-route " + spec)
            var post = compile_spec(lin, t, "route(sabre)," + spec + ",basis,check", 3)
            assert_conforms(post.circuit, t, "post-route " + spec)
            assert_routed_equiv(lin, post, "post-route " + spec)
    # an explicitly requested routing model must exist (auto silently falls back)
    try:
        _ = compile_spec(lin, t, "route(sabre,policy=no_such_model_xyz),basis", 3)
        assert_true(False, "unknown routing model accepted")
    except e:
        var m = String(e)
        assert_true(m.startswith("QM-E616") or m.startswith("QM-E617") or m.startswith("QM-E620"), m)
        assert_true(m.find("Hint:") >= 0)
    print("  all-to-all degraded mode + every synthesis pass/method exact")


def test_determinism() raises:
    var t = parse_target("heavy_hex:3/sc_cz")
    var c = snap_random(7, 60, 307)
    for lvl in [1, 2, 3]:
        var a = transpile_ai(c, t, lvl, 13)
        var b = transpile_ai(c, t, lvl, 13)
        assert_true(result_text(a) == result_text(b), "ai" + String(lvl) + " deterministic")
    print("  transpile_ai deterministic for a seed")


def test_reports() raises:
    var t = parse_target("line:6/sc_cx")
    var c = snap_linear(6, 40, 401)
    var r = transpile_ai(c, t, 2, 7)
    var js = r.to_json()
    assert_true(js.find("linear_synth(direction=forward,commute=0)") >= 0)
    assert_true(js.find("blocks=") >= 0, "synthesis notes in the report")
    assert_true(js.find("policy=auto") >= 0)
    print("  reports carry the synthesis statistics")


def main() raises:
    test_spec_surface()
    test_snapshot_unchanged()
    test_reports()
    test_all_to_all_and_synth_passes()
    test_determinism()
    test_transpile_ai_equivalence()
    print("test_ai_transpiler OK")
