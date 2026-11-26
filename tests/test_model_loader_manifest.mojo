# Tests for the manifest checks of the AI-transpiler model loader
# (qumo.compiler.policy.load_model_file, shared by ai_route.load_route_model and
# the synthesis policy) and for the visibility of an unreadable INDEX.json.
#
# The FNV-1a 64 checksum covers weights.bin only, so a manifest with a valid
# checksum can still carry hostile numbers. Every crafted manifest below keeps
# the fixture's weights.bin (checksum valid) and must be refused with QM-E617,
# never abort the process or load.
from std.ffi import external_call, c_int
from std.os import makedirs
from std.os.path import exists
from std.tempfile import mkdtemp
from std.testing import assert_true, assert_equal, assert_false
from qumo.ir.circuit import Circuit
from qumo.core.json import parse_json
from qumo.compiler.target import Target, parse_target
from qumo.compiler.pass_manager import compile_spec
from qumo.compiler.routing import RouteOptions
from qumo.compiler.blockalg import SubGraph, BlockKind
from qumo.compiler.policy import (
    load_model_file, read_model_index, read_model_index_checked, model_index_problems, MAX_TENSORS,
)
from qumo.compiler.policy_synth import PolicySession
from qumo.compiler.ai_route import (
    AIRouteOptions, ai_route, load_route_model, find_route_model, route_index_warning,
)
from qumo.compiler.resynth import SynthOptions, SynthStats, resynthesize_blocks

comptime FX = "tests/fixtures/ai/policy/synth/fx_perm_line3"
comptime SHIPPED = "models/ai_transpiler"
# the first tensor of the fixture manifest, as written
comptime T0 = "{\"name\":\"embeddings.weight\",\"shape\":[9,16],\"dtype\":\"f32\",\"offset\":0}"


def _read(path: String) raises -> String:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return text


def _write(path: String, text: String) raises:
    with open(path, "w") as f:
        f.write(text)


def _copy_weights(dst_dir: String) raises:
    var data: List[UInt8]
    with open(String(FX) + "/weights.bin", "r") as f:
        data = f.read_bytes()
    with open(dst_dir + "/weights.bin", "w") as f:
        f.write_bytes(data)


def _symlink(target: String, link: String) -> Int:
    var t = target.copy()
    var l = link.copy()
    var rc = external_call["symlink", c_int](t.as_c_string_span().ptr(), l.as_c_string_span().ptr())
    return Int(rc)


def _model(root: String, name: String, old: String, new: String) raises -> String:
    """A copy of the fixture model in <root>/<name> with `old` replaced by
    `new` in its manifest (weights.bin unchanged: the checksum stays valid)."""
    var d = root + "/" + name
    makedirs(d)
    var man = _read(String(FX) + "/manifest.json")
    if old.byte_length() > 0:
        assert_true(man.find(old) >= 0, "fixture manifest has: " + old)
        man = man.replace(old, new)
    _write(d + "/manifest.json", man)
    _copy_weights(d)
    return d


def _expect_e617(dir: String, what: String, needle: String) raises:
    var msg: String
    try:
        var m = load_model_file(dir)
        msg = "loaded with " + String(len(m.names)) + " tensors"
    except e:
        msg = String(e)
    if not msg.startswith("QM-E617"):
        print("  ", what, "->", msg)
    assert_true(msg.startswith("QM-E617"), what + ": expected QM-E617, got: " + msg)
    assert_true(msg.find("Hint:") >= 0, what + ": error has a hint")
    assert_true(msg.find(needle) >= 0, what + ": error names the cause (" + needle + "): " + msg)


def test_crafted_manifests() raises:
    var root = mkdtemp()
    # the unmodified copy loads: the crafted ones differ only in the manifest
    var good = load_model_file(_model(root, "good", "", ""))
    assert_equal(len(good.names), 8)
    assert_equal(len(good.tensors[0]), 9 * 16)

    # size wraps to 0 in 64 bits: 2^31 * 2^31 * 4 bytes
    _expect_e617(_model(root, "shape_wrap", "\"shape\":[9,16]", "\"shape\":[2147483648,2147483648]"),
                 "shape [2^31, 2^31]", "larger than weights.bin")
    # product overflows on the third factor
    _expect_e617(_model(root, "shape_wrap3", "\"shape\":[9,16]", "\"shape\":[3037000500,3037000500,2]"),
                 "shape [3037000500, 3037000500, 2]", "larger than weights.bin")
    # offset + bytes wraps negative
    _expect_e617(_model(root, "offset_wrap", String(T0), String(T0).replace("\"offset\":0", "\"offset\":9223372036854775800")),
                 "offset 2^63 - 8", "lies outside weights.bin")
    _expect_e617(_model(root, "offset_neg", String(T0), String(T0).replace("\"offset\":0", "\"offset\":-64")),
                 "offset -64", "lies outside weights.bin")
    # negative dimensions whose product is positive
    _expect_e617(_model(root, "shape_neg", "\"shape\":[9,16]", "\"shape\":[-4,-2]"),
                 "shape [-4, -2]", "non-positive dimension")
    _expect_e617(_model(root, "shape_zero", "\"shape\":[9,16]", "\"shape\":[0,16]"),
                 "shape [0, 16]", "non-positive dimension")
    _expect_e617(_model(root, "rank", "\"shape\":[9,16]", "\"shape\":[1,1,1,1,1,1,1,1,1]"),
                 "9 dimensions", "dimensions")
    # one element past the end / exactly at the end (2048-byte weights.bin)
    _expect_e617(_model(root, "past_end", "\"shape\":[1],\"dtype\":\"f32\",\"offset\":1984", "\"shape\":[1],\"dtype\":\"f32\",\"offset\":2045"),
                 "last tensor 1 byte past the end", "lies outside weights.bin")
    var at_end = load_model_file(_model(root, "at_end", "\"shape\":[1],\"dtype\":\"f32\",\"offset\":1984", "\"shape\":[1],\"dtype\":\"f32\",\"offset\":2044"))
    assert_equal(len(at_end.tensors[7]), 1)
    # too many tensors
    var many = String()
    for _ in range(MAX_TENSORS):
        many += String(T0) + ","
    _expect_e617(_model(root, "many", String(T0), many + String(T0)), "MAX_TENSORS + 8 tensors", "tensors (at most")

    # weights_file must be a plain file name inside the model directory
    _ = _model(root, "models_neg", "", "")
    _expect_e617(_model(root, "outside", "\"weights_file\":\"weights.bin\"", "\"weights_file\":\"../models_neg/weights.bin\""),
                 "weights_file ../models_neg/weights.bin", "not a plain file name")
    _expect_e617(_model(root, "absolute", "\"weights_file\":\"weights.bin\"", "\"weights_file\":\"" + root + "/models_neg/weights.bin\""),
                 "absolute weights_file", "not a plain file name")
    _expect_e617(_model(root, "dotdot", "\"weights_file\":\"weights.bin\"", "\"weights_file\":\"..\""),
                 "weights_file ..", "not a plain file name")
    _expect_e617(_model(root, "empty_name", "\"weights_file\":\"weights.bin\"", "\"weights_file\":\"\""),
                 "empty weights_file", "not a plain file name")
    makedirs(root + "/nested/sub")
    _expect_e617(_model(root, "nested_ref", "\"weights_file\":\"weights.bin\"", "\"weights_file\":\"sub/weights.bin\""),
                 "weights_file sub/weights.bin", "not a plain file name")
    # a symbolic link that leaves the directory (same bytes: checksum valid)
    var ld = _model(root, "linked", "\"weights_file\":\"weights.bin\"", "\"weights_file\":\"w.bin\"")
    if _symlink(root + "/models_neg/weights.bin", ld + "/w.bin") == 0:
        _expect_e617(ld, "weights_file symlink out of the directory", "symbolic link")
        # a link that stays inside the directory is a plain name: accepted
        var li = _model(root, "linked_in", "\"weights_file\":\"weights.bin\"", "\"weights_file\":\"w.bin\"")
        if _symlink("weights.bin", li + "/w.bin") == 0:
            assert_equal(len(load_model_file(li).names), 8)
    else:
        print("  symlink() unavailable here: symbolic-link case skipped")
    # load_route_model goes through the same loader
    var msg = String()
    try:
        _ = load_route_model(root + "/shape_wrap")
    except e:
        msg = String(e)
    assert_true(msg.startswith("QM-E617"), "load_route_model: " + msg)
    print("  crafted manifests (shape / offset overflow, negative shape, tensor count, weights_file) -> QM-E617 OK")


def _line_circuit() -> Circuit:
    var c = Circuit(5)
    for r in range(3):
        c.h(r)
        c.cx(0, 4)
        c.cx(1, 3)
        c.cx(0, 2)
        c.cx(2, 4)
        c.cx(1, 4)
    return c^


def _ops_text(c: Circuit) -> String:
    var s = String()
    for g in c.ops:
        s += String(Int(g.kind)) + ":" + String(g.q0) + "," + String(g.q1) + ";"
    return s


def test_shipped_models_still_load() raises:
    """Every committed model passes the stricter loader, and routing with the
    committed router model still works (the routed circuit is printed; it was
    compared with the pre-fix output when the checks were added)."""
    if not exists(String(SHIPPED) + "/route/INDEX.json"):
        print("  no committed models: skipped")
        return
    var problem = String()
    var rows = read_model_index_checked(SHIPPED, "synth", problem)
    assert_equal(problem, "")
    assert_equal(len(rows), len(read_model_index(SHIPPED, "synth")))
    for r in rows:
        var m = load_model_file(r.path)
        for i in range(len(m.names)):
            var cnt = 1
            for s in m.shapes[i]:
                assert_true(s > 0, r.id + ": positive dimension")
                cnt *= s
            assert_equal(len(m.tensors[i]), cnt)
    var rrows = read_model_index_checked(SHIPPED, "route", problem)
    assert_equal(problem, "")
    assert_true(len(rrows) >= 1, "committed router model listed")
    for r in rrows:
        assert_true(load_route_model(r.path).ok, r.id)
    var dirs = List[String]()
    dirs.append(SHIPPED)
    assert_equal(model_index_problems(dirs, "route"), "")
    assert_equal(model_index_problems(dirs, "synth"), "")
    var ao = AIRouteOptions()
    ao.policy = "auto"
    ao.model_dir = SHIPPED
    ao.portfolio = False  # the policy's own routing, not the SABRE candidate
    assert_equal(route_index_warning(ao), "")
    var out = Circuit(1)
    var il = List[Int]()
    var fl = List[Int]()
    var note = String()
    assert_true(ai_route(_line_circuit(), Target.line(5), RouteOptions(), ao, out, il, fl, note), "routes")
    assert_true(note.find("warning") < 0, "valid index: no warning in the note: " + note)
    print("  routed digest:", len(out.ops), "ops;", note)
    print("  ROUTED " + _ops_text(out))
    print("  shipped models load (", len(rows), "synth,", len(rrows), "route ); valid index gives no warning OK")


def test_unreadable_index_is_visible() raises:
    var root = mkdtemp()
    makedirs(root + "/route")
    makedirs(root + "/synth")
    # row 1 has no "dir": before the fix the whole index silently became empty
    _write(root + "/route/INDEX.json",
           "[{\"id\":\"r0\",\"kind\":\"routing\",\"num_qubits\":0,\"edges\":[],\"wl_hash\":\"\",\"dir\":\"r0\"},"
           + "{\"id\":\"r1\",\"kind\":\"routing\",\"num_qubits\":0,\"edges\":[],\"wl_hash\":\"\"}]")
    _write(root + "/synth/INDEX.json", "[{\"id\":\"s0\",\"kind\":\"permutation\",\"num_qubits\":\"three\",\"edges\":[],\"dir\":\"s0\"}]")
    var problem = String()
    assert_equal(len(read_model_index_checked(root, "route", problem)), 0)
    assert_true(problem.find(root + "/route/INDEX.json") >= 0 and problem.find("row 1") >= 0, problem)
    assert_true(problem.find("\"") < 0 and problem.find("\n") < 0, "problem text is safe inside a JSON note: " + problem)
    assert_equal(len(read_model_index(root, "route")), 0)  # unchanged result
    assert_equal(len(read_model_index_checked(root, "synth", problem)), 0)
    assert_true(problem.find("row 0") >= 0, problem)
    _write(root + "/synth/INDEX.json", "{\"not\": \"an array\"}")
    assert_equal(len(read_model_index_checked(root, "synth", problem)), 0)
    assert_true(problem.find("not a JSON array") >= 0, problem)
    _write(root + "/synth/INDEX.json", "[{\"id\": ")
    assert_equal(len(read_model_index_checked(root, "synth", problem)), 0)
    assert_true(problem.find("is unreadable") >= 0, problem)
    # absent index: no rows and no problem
    assert_equal(len(read_model_index_checked(root, "nothing", problem)), 0)
    assert_equal(problem, "")

    var dirs = List[String]()
    dirs.append(root)
    # router, policy=auto: still SABRE (no model), and the note says why
    assert_equal(find_route_model("auto", dirs), "")
    var ao = AIRouteOptions()
    ao.policy = "auto"
    ao.model_dir = root
    var out = Circuit(1)
    var il = List[Int]()
    var fl = List[Int]()
    var note = String()
    # the committed models root is searched after `root`, so auto may still find
    # the committed router model; the warning must be in the note either way
    assert_true(ai_route(_line_circuit(), Target.line(5), RouteOptions(), ao, out, il, fl, note), "auto still routes")
    assert_true(note.find("warning: model index " + root + "/route/INDEX.json is unreadable (row 1)") >= 0, note)
    # router, explicit id: QM-E617 with the real cause
    var msg = String()
    try:
        _ = find_route_model("r1", dirs)
    except e:
        msg = String(e)
    assert_true(msg.startswith("QM-E617") and msg.find("Hint:") >= 0, msg)
    assert_true(msg.find("route/INDEX.json is unreadable (row 1)") >= 0, "explicit id names the index: " + msg)
    # heuristic / off need no model: no warning
    ao.policy = "heuristic"
    assert_equal(route_index_warning(ao), "")

    # synthesis, explicit id: QM-E616 with the real cause
    var sess = PolicySession(dirs)
    assert_true(sess.index_problem.find("synth/INDEX.json is unreadable") >= 0, sess.index_problem)
    var edges: List[Int] = [0, 1, 1, 2]
    var g = SubGraph(3, edges)
    msg = String()
    try:
        _ = sess.match(BlockKind.PERM, g, "fx_perm_line3")
    except e:
        msg = String(e)
    assert_true(msg.startswith("QM-E616") and msg.find("Hint:") >= 0, msg)
    assert_true(msg.find("synth/INDEX.json is unreadable") >= 0, "explicit synthesis id names the index: " + msg)
    assert_false(sess.match(BlockKind.PERM, g, "auto").ok, "auto: no model, no raise")
    # synthesis, model=auto through the pass: the note carries the warning
    var c = Circuit(3)
    for _ in range(3):
        c.swap(0, 1)
        c.swap(1, 2)
        c.swap(0, 1)
        c.swap(1, 2)
    var o = SynthOptions.default(BlockKind.PERM)
    o.method = "policy"
    o.model = "auto"
    o.min_block = 2
    o.model_dir = root
    var st = SynthStats()
    _ = resynthesize_blocks(c, Target.line(3), True, o, st)
    assert_true(st.note.find("model_index_warning(") >= 0 and st.note.find("synth/INDEX.json is unreadable") >= 0, st.note)
    # a valid index: no warning in the pass note
    var o2 = o.copy()
    o2.model_dir = "tests/fixtures/ai/policy"
    var st2 = SynthStats()
    var c2 = c.copy()
    _ = resynthesize_blocks(c2, Target.line(3), True, o2, st2)
    assert_true(st2.note.find("model_index_warning") < 0, st2.note)
    # the same through the pass manager: the report of a text pipeline says it,
    # whichever of SABRE / the policy is kept; an explicit id fails with the cause
    var t5 = parse_target("line:5/sc_cx")
    var rep = compile_spec(_line_circuit(), t5, "route(sabre,policy=auto,model_dir=" + root + "),basis,check", 3).to_json()
    assert_true(rep.find("route/INDEX.json is unreadable (row 1)") >= 0, "pipeline report carries the warning: " + rep)
    _ = parse_json(rep)  # the note keeps the report valid JSON
    var ok_rep = compile_spec(_line_circuit(), t5, "route(sabre,policy=auto,model_dir=models/ai_transpiler),basis,check", 3).to_json()
    assert_true(ok_rep.find("unreadable") < 0, "valid index: no warning in the report")
    msg = String()
    try:
        _ = compile_spec(_line_circuit(), t5, "route(sabre,policy=r1,model_dir=" + root + "),basis,check", 3)
    except e:
        msg = String(e)
    assert_true(msg.startswith("QM-E617") and msg.find("is unreadable (row 1)") >= 0, "pipeline, explicit id: " + msg)
    print("  unreadable INDEX.json: warning in the notes for auto, cause in the error for an explicit id OK")


def main() raises:
    test_crafted_manifests()
    test_shipped_models_still_load()
    test_unreadable_index_is_visible()
    print("test_model_loader_manifest OK")
