"""Tests for qrl.container (MOJO_PORT.md 13.2 `test_rl_container`, acceptance A2).

* the 10 shipped models re-encode from their decoded tensors to byte-identical weights.bin, manifest.json, fnv and sha
* INDEX regeneration reproduces the shipped INDEX.json files (byte for byte, router default preserved or moved)
* `verify_models` on models/ai_transpiler: 0 problems; a flipped byte, a missing file, a stray .bin and a broken
  INDEX are reported with the legacy FAIL lines; all five INDEX layouts of the old checker are understood
* atomic directory swap: a simulated stop between the rename steps leaves a complete old or new directory and
  `recover_dir` restores it; a failed write leaves the previous directory untouched
* two common layers (consecutive numbering) load in `MLPPolicy` and agree with a naive Float64 forward
"""

from std.collections import List
from std.memory import bitcast
from std.os import makedirs, listdir
from std.os.path import exists, isdir, isfile
from qrl.json import parse_json
from qrl.model import load_model_file, MLPPolicy
from qrl.container import (
    TensorOut,
    ModelDirInfo,
    VerifyReport,
    write_model_dir,
    atomic_replace_dir,
    atomic_replace_dir_stop,
    recover_dir,
    rebuild_index,
    verify_models,
    dump_indent1,
    index_row_from_dir,
    split_manifest,
    remove_tree,
    write_text_atomic,
)
from qrl.digest import sha256_hex, sha256_file_hex
from qrl.testutil import rand_fill, read_bytes, read_text, model_root, has_model_root, skip_without_models

comptime WORK = "build/test_rl_container"


def check(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAIL: " + what)


def expect_error(msg: String, code: String, what: String) raises:
    if not msg.startswith(code):
        raise Error("FAIL: " + what + ": expected " + code + ", got '" + msg + "'")


def write_bytes_file(path: String, data: List[UInt8]) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(data))


def copy_dir(src: String, dst: String) raises:
    makedirs(dst, exist_ok=True)
    for n in listdir(src):
        write_bytes_file(dst + "/" + n, read_bytes(src + "/" + n))


def model_ids(sub: String) raises -> List[String]:
    var out = List[String]()
    for n in listdir(model_root() + "/" + sub):
        if isdir(model_root() + "/" + sub + "/" + n):
            out.append(n)
    sort(out)
    return out^


def reencode(src: String, dst: String) raises -> ModelDirInfo:
    var lm = load_model_file(src)
    var tensors = List[TensorOut]()
    for i in range(len(lm.names)):
        tensors.append(TensorOut(lm.names[i], lm.shapes[i].copy(), lm.tensors[i].copy(), "f16"))
    var parts = split_manifest(lm.manifest)
    return write_model_dir(dst, parts.head, tensors, "f16", List[String](), parts.tail)


# ------------------------------------------------------------------------------------------ tiny test models


def tiny_head(id: String, common: String) -> String:
    return (
        '"format":"qumo.rlsynth.v0","id":"'
        + id
        + '","kind":"permutation","num_qubits":3,"edges":[[0,1],[1,2]],'
        + '"actions":[["SWAP",0,1],["SWAP",1,2]],'
        + '"obs":{"encoding":"sparse_index","size":9,"layout":"3x3","max_rotations":0},"max_depth":128,'
        + '"arch":{"type":"basic_mlp","emb":8,"common":'
        + common
        + ',"act":"relu","first_layer_transposed":true},'
        + '"symmetries":{"obs_perms":[],"act_perms":[]},"license":"Apache-2.0"'
    )


def rnd(n: Int, seed: UInt64) -> List[Float32]:
    var v = List[Float32](length=n, fill=0.0)
    rand_fill(v, seed, -1.0, 1.0)
    return v^


def tiny_tensors(seed: UInt64, two_common: Bool) -> List[TensorOut]:
    var t = List[TensorOut]()
    var c1 = 6
    t.append(TensorOut("embeddings.weight", [9, 8], rnd(72, seed + 1), "f16"))
    t.append(TensorOut("embeddings.bias", [8], rnd(8, seed + 2), "f16"))
    t.append(TensorOut("common.0.weight", [c1, 8], rnd(c1 * 8, seed + 3), "f16"))
    t.append(TensorOut("common.0.bias", [c1], rnd(c1, seed + 4), "f16"))
    var last = c1
    if two_common:
        t.append(TensorOut("common.1.weight", [5, c1], rnd(5 * c1, seed + 5), "f16"))
        t.append(TensorOut("common.1.bias", [5], rnd(5, seed + 6), "f16"))
        last = 5
    t.append(TensorOut("action.0.weight", [2, last], rnd(2 * last, seed + 7), "f16"))
    t.append(TensorOut("action.0.bias", [2], rnd(2, seed + 8), "f16"))
    t.append(TensorOut("value.0.weight", [1, last], rnd(last, seed + 9), "f16"))
    t.append(TensorOut("value.0.bias", [1], rnd(1, seed + 10), "f16"))
    return t^


def write_tiny(dir: String, id: String, seed: UInt64) raises -> ModelDirInfo:
    return write_model_dir(
        dir, tiny_head(id, "[6]"), tiny_tensors(seed, False), "f16", List[String](), ""
    )


def weights_sha(dir: String) raises -> String:
    return sha256_file_hex(dir + "/weights.bin")


# ------------------------------------------------------------------------------------------------ tests


def test_reencode_shipped() raises:
    if skip_without_models("test_reencode_shipped"):
        return
    var n_models = 0
    var subs: List[String] = ["synth", "route"]
    for s in subs:
        for id in model_ids(s):
            var src = model_root() + "/" + s + "/" + id
            var dst = WORK + "/models/" + s + "/" + id
            var info = reencode(src, dst)
            var w0 = read_bytes(src + "/weights.bin")
            var w1 = read_bytes(dst + "/weights.bin")
            check(w0 == w1, id + ": weights.bin differs after re-encoding")
            var m0 = read_bytes(src + "/manifest.json")
            var m1 = read_bytes(dst + "/manifest.json")
            check(m0 == m1, id + ": manifest.json differs after re-encoding")
            var man = parse_json(read_text(src + "/manifest.json"))
            check(info.fnv1a64 == man.get("fnv1a64").as_string(), id + ": fnv1a64")
            check(info.sha256 == man.get("sha256").as_string(), id + ": sha256")
            check(info.weights_bytes == len(w0), id + ": weights_bytes")
            check(info.manifest_bytes == len(m0), id + ": manifest_bytes")
            check(info.weights_bytes % 64 == 0, id + ": not padded to 64")
            check(not exists(dst + ".tmp") and not exists(dst + ".old"), id + ": leftover staging directory")
            n_models += 1
    check(n_models == 10, "expected the 10 shipped models, found " + String(n_models))


def test_index_regeneration() raises:
    if skip_without_models("test_index_regeneration"):
        return
    var root = WORK + "/models"
    rebuild_index(root, "synth", "")
    var got = read_text(root + "/synth/INDEX.json")
    var want = read_text(model_root() + "/synth/INDEX.json")
    var gj = parse_json(got)
    var wj = parse_json(want)
    check(len(gj) == len(wj) and len(gj) == 9, "synth INDEX row count")
    for i in range(len(wj)):
        var keys = wj.at(i).keys()
        check(len(keys) == len(gj.at(i).keys()), "synth INDEX row " + String(i) + ": member count")
        for k in keys:
            check(gj.at(i).get(k).to_json() == wj.at(i).get(k).to_json(), "synth INDEX row " + String(i) + " member " + k)
    check(got == want, "synth INDEX.json is not byte identical to the shipped file")

    rebuild_index(root, "route", "route_mlp_h64_v1")
    check(read_text(root + "/route/INDEX.json") == read_text(model_root() + "/route/INDEX.json"), "route INDEX (default set)")
    rebuild_index(root, "route", "")
    check(read_text(root + "/route/INDEX.json") == read_text(model_root() + "/route/INDEX.json"), "route INDEX (default preserved)")

    # a second router model: the previous default survives an export without --default, and can be moved
    var src = root + "/route/route_mlp_h64_v1"
    var lm = load_model_file(src)
    var tensors = List[TensorOut]()
    for i in range(len(lm.names)):
        tensors.append(TensorOut(lm.names[i], lm.shapes[i].copy(), lm.tensors[i].copy(), "f16"))
    var man = split_manifest(lm.manifest)
    var head = String(man.head.replace('"id":"route_mlp_h64_v1"', '"id":"route_b"'))
    check(head != man.head, "id replacement in the router manifest")
    _ = write_model_dir(root + "/route/route_b", head, tensors, "f16", List[String](), man.tail)
    rebuild_index(root, "route", "")
    var j = parse_json(read_text(root + "/route/INDEX.json"))
    check(len(j) == 2, "route INDEX has two rows")
    check(j.at(0).get("id").as_string() == "route_b" and not j.at(0).get("default").as_bool(), "route_b is not default")
    check(j.at(1).get("id").as_string() == "route_mlp_h64_v1" and j.at(1).get("default").as_bool(), "previous default preserved")
    rebuild_index(root, "route", "route_b")
    j = parse_json(read_text(root + "/route/INDEX.json"))
    check(j.at(0).get("default").as_bool() and not j.at(1).get("default").as_bool(), "default moved to route_b")
    var failed = False
    try:
        rebuild_index(root, "route", "no_such_model")
    except e:
        failed = True
        expect_error(String(e), "QM-E640", "unknown default id")
    check(failed, "unknown default id must raise")
    check(parse_json(read_text(root + "/route/INDEX.json")).at(0).get("default").as_bool(), "failed rebuild left INDEX untouched")
    remove_tree(root + "/route/route_b")
    rebuild_index(root, "route", "route_mlp_h64_v1")

    # one row of a directory
    var row = parse_json(index_row_from_dir(root + "/synth/lf_line6"))
    var shipped_idx = parse_json(read_text(model_root() + "/synth/INDEX.json"))
    var found = False
    for i in range(len(shipped_idx)):
        if shipped_idx.at(i).get("id").as_string() == "lf_line6":
            found = True
            for k in shipped_idx.at(i).keys():
                check(row.get(k).to_json() == shipped_idx.at(i).get(k).to_json(), "index_row_from_dir member " + k)
    check(found, "lf_line6 row in the shipped INDEX")


def test_dump_indent1() raises:
    var src = '{"a":[1,2],"b":{},"c":[],"d":"café\\n\\"q\\"","e":null,"f":true,"g":[[1],[]],"h":{"x":-3,"y":[{"z":0}]}}'
    var want = (
        '{\n "a": [\n  1,\n  2\n ],\n "b": {},\n "c": [],\n "d": "caf\\u00e9\\n\\"q\\"",\n "e": null,\n "f": true,\n'
        + ' "g": [\n  [\n   1\n  ],\n  []\n ],\n "h": {\n  "x": -3,\n  "y": [\n   {\n    "z": 0\n   }\n  ]\n }\n}'
    )
    var got = dump_indent1(parse_json(src))
    check(got == want, "dump_indent1 layout:\n" + got)
    check(dump_indent1(parse_json("[]")) == "[]", "empty array")
    check(dump_indent1(parse_json("[1]")) == "[\n 1\n]", "one element array")


def test_split_manifest() raises:
    var text = '{"a":{"k":"}{,"},"b":[1,{"t":"\\""}],"tensors":[{"name":"x"}],"weights_file":"weights.bin","weights_bytes":64,"fnv1a64":"0","sha256":"1","tv":[{"o":[1,2]}],"z":true}'
    var p = split_manifest(text)
    check(p.head == '"a":{"k":"}{,"},"b":[1,{"t":"\\""}]', "head: " + p.head)
    check(p.tail == '"tv":[{"o":[1,2]}],"z":true', "tail: " + p.tail)
    var q = split_manifest('{"a":1, "b" : 2}')
    check(q.head == '"a":1,"b" : 2' and q.tail == "", "no tensors member: " + q.head)
    var bad = False
    try:
        _ = split_manifest('{"a":[1,2}')
    except e:
        bad = True
        expect_error(String(e), "QM-E644", "malformed manifest")
    check(bad, "malformed manifest must raise")


def fixed2_expected(total: Int) -> String:
    var c = Int(Float64(total) / 1.0e6 * 100.0 + 0.5)
    var frac = c % 100
    return String(c // 100) + "." + ("0" if frac < 10 else "") + String(frac)


def test_verify_shipped() raises:
    if skip_without_models("test_verify_shipped"):
        return
    var rep = verify_models(model_root())
    check(rep.n_problems == 0, "shipped tree has problems: " + rep.lines[0])
    check(rep.n_index == 2 and rep.n_files == 10, "counts " + String(rep.n_index) + "/" + String(rep.n_files))
    check(len(rep.lines) == 1, "only the summary line expected")
    var total = 0
    var subs: List[String] = ["synth", "route"]
    for s in subs:
        total += len(read_bytes(model_root() + "/" + s + "/INDEX.json"))
        for id in model_ids(s):
            total += len(read_bytes(model_root() + "/" + s + "/" + id + "/weights.bin"))
            total += len(read_bytes(model_root() + "/" + s + "/" + id + "/manifest.json"))
    check(rep.total_bytes == total, "total bytes " + String(rep.total_bytes) + " vs " + String(total))
    var want = "models-verify: 2 INDEX.json, 10 files ok, 0 problems, " + fixed2_expected(total) + " MB total"
    check(rep.lines[0] == want, "summary line: " + rep.lines[0] + " vs " + want)
    var rep2 = verify_models(WORK + "/models")
    check(rep2.n_problems == 0 and rep2.n_files == 10, "re-encoded staging tree verifies")
    check(rep2.lines[0] == want, "staging tree summary identical (byte-identical files)")
    var rep3 = verify_models(WORK + "/no_such_root")
    check(rep3.n_problems == 0 and len(rep3.lines) == 1, "missing root is not a problem")
    check(rep3.lines[0] == "models-verify: no model directory '" + WORK + "/no_such_root' (nothing to verify)", "missing root line")


def test_verify_failures() raises:
    var root = WORK + "/v1"
    remove_tree(root)
    _ = write_tiny(root + "/synth/tiny", "tiny", 7)
    rebuild_index(root, "synth", "")
    var rep = verify_models(root)
    check(rep.n_problems == 0 and rep.n_files == 1 and rep.n_index == 1, "clean tiny root")

    # flipped byte
    var w = read_bytes(root + "/synth/tiny/weights.bin")
    var good = w.copy()
    w[5] = w[5] ^ UInt8(1)
    write_bytes_file(root + "/synth/tiny/weights.bin", w)
    rep = verify_models(root)
    check(rep.n_problems == 1 and rep.n_files == 0, "flipped byte: counts")
    check(rep.lines[0] == "FAIL " + root + "/synth/tiny/weights.bin: sha256 mismatch (QM-E617)", "flipped byte line: " + rep.lines[0])
    check(rep.lines[1].startswith("models-verify: 1 INDEX.json, 0 files ok, 1 problems, "), "flipped byte summary")
    write_bytes_file(root + "/synth/tiny/weights.bin", good)

    # stray .bin
    write_bytes_file(root + "/synth/tiny/extra.bin", good)
    rep = verify_models(root)
    check(rep.n_problems == 1, "stray bin: counts")
    check(rep.lines[0] == "FAIL " + root + "/synth/tiny/extra.bin: weights not listed in any INDEX.json", "stray bin line: " + rep.lines[0])
    remove_tree(root + "/synth/tiny/extra.bin")

    # missing file
    remove_tree(root + "/synth/tiny/weights.bin")
    rep = verify_models(root)
    check(rep.n_problems == 1, "missing: counts")
    check(
        rep.lines[0] == "FAIL " + root + "/synth/tiny/weights.bin: missing (listed in " + root + "/synth/INDEX.json)",
        "missing line: " + rep.lines[0],
    )
    write_bytes_file(root + "/synth/tiny/weights.bin", good)

    # broken INDEX: unreadable, and its weights are then not listed anywhere
    var index_text = read_text(root + "/synth/INDEX.json")
    with open(root + "/synth/INDEX.json", "w") as f:
        f.write("not json")
    rep = verify_models(root)
    check(rep.n_problems == 2 and rep.n_index == 1, "broken INDEX: counts")
    check(rep.lines[0].startswith("FAIL " + root + "/synth/INDEX.json: unreadable ("), "broken INDEX line: " + rep.lines[0])
    check(rep.lines[1] == "FAIL " + root + "/synth/tiny/weights.bin: weights not listed in any INDEX.json", "broken INDEX second line")
    with open(root + "/synth/INDEX.json", "w") as f:
        f.write(index_text)
    check(verify_models(root).n_problems == 0, "restored root verifies")


def test_verify_layouts() raises:
    var root = WORK + "/lay"
    remove_tree(root)
    makedirs(root + "/d1", exist_ok=True)
    makedirs(root + "/sub", exist_ok=True)
    makedirs(root + "/d2", exist_ok=True)
    makedirs(root + "/d3", exist_ok=True)
    makedirs(root + "/d4", exist_ok=True)
    var hashes = List[String]()
    var paths: List[String] = [
        "d1/weights.bin", "p1.bin", "sub/f1.bin", "d2/a.bin", "d2/b.bin", "q1.bin", "q2x.bin", "d3/r1.bin", "d3/r2.bin", "d4/weights.bin",
    ]
    for i in range(len(paths)):
        var data = List[UInt8]()
        for k in range(10 + i):
            data.append(UInt8((k * 7 + i) & 0xFF))
        write_bytes_file(root + "/" + paths[i], data)
        hashes.append(sha256_hex(data))
    var h = hashes.copy()
    var index = (
        '[{"id":"x","dir":"d1","sha256":"' + h[0] + '"},'
        + '{"path":"p1.bin","sha256":"' + h[1] + '"},'
        + '{"file":"sub/f1.bin","sha256":"' + h[2].upper() + '"},'
        + '{"dir":"d2","sha256":{"a.bin":"' + h[3] + '","b.bin":"' + h[4] + '"}},'
        + '{"files":{"q1.bin":"' + h[5] + '","q2":{"path":"q2x.bin","sha256":"' + h[6] + '"}}},'
        + '{"dir":"d3","files":[{"path":"r1.bin","sha256":"' + h[7] + '"},{"file":"r2.bin","sha256":"' + h[8] + '"}]},'
        + '{"nested":{"deep":[{"dir":"d4","sha256":"' + h[9] + '"}]}}]'
    )
    with open(root + "/INDEX.json", "w") as f:
        f.write(index)
    var rep = verify_models(root)
    check(rep.n_problems == 0, "layouts: " + rep.lines[0])
    check(rep.n_files == 10 and rep.n_index == 1, "layouts: counts " + String(rep.n_files))
    # break one entry of each family and expect exactly the matching FAIL lines
    var bad = index.replace(h[1], h[0]).replace(h[7], h[0])
    with open(root + "/INDEX.json", "w") as f:
        f.write(bad)
    rep = verify_models(root)
    check(rep.n_problems == 2 and rep.n_files == 8, "layouts: two mismatches, got " + String(rep.n_problems))
    check(rep.lines[0] == "FAIL " + root + "/p1.bin: sha256 mismatch (QM-E617)", "mismatch line 0: " + rep.lines[0])
    check(rep.lines[1] == "FAIL " + root + "/d3/r1.bin: sha256 mismatch (QM-E617)", "mismatch line 1: " + rep.lines[1])


def test_atomic() raises:
    var root = WORK + "/at"
    remove_tree(root)
    var m = root + "/m"
    var v1 = write_tiny(m, "m", 1)
    check(not exists(m + ".tmp") and not exists(m + ".old"), "no staging leftovers after the first write")
    check(weights_sha(m) == v1.sha256, "v1 on disk")
    var v2 = write_tiny(m, "m", 2)
    check(v1.sha256 != v2.sha256, "versions differ")
    check(weights_sha(m) == v2.sha256, "v2 replaced v1")
    check(not exists(m + ".tmp") and not exists(m + ".old"), "no leftovers after replacing")

    var v3 = write_tiny(root + "/n", "n", 3)

    # stop after the first rename: final missing, old complete, tmp complete
    copy_dir(root + "/n", m + ".tmp")
    atomic_replace_dir_stop(m + ".tmp", m, 1)
    check(not exists(m) and exists(m + ".old") and exists(m + ".tmp"), "stop 1: state")
    check(weights_sha(m + ".old") == v2.sha256, "stop 1: .old is the complete old model")
    recover_dir(m)
    check(exists(m) and not exists(m + ".old") and not exists(m + ".tmp"), "stop 1: recovered state")
    check(weights_sha(m) == v2.sha256, "stop 1: the old model is back")
    _ = load_model_file(m)

    # stop after the second rename: final is the new complete model, old still present
    copy_dir(root + "/n", m + ".tmp")
    atomic_replace_dir_stop(m + ".tmp", m, 2)
    check(exists(m) and exists(m + ".old") and not exists(m + ".tmp"), "stop 2: state")
    check(weights_sha(m) == v3.sha256, "stop 2: the new model is in place")
    _ = load_model_file(m)
    recover_dir(m)
    check(exists(m) and not exists(m + ".old"), "stop 2: recovered state")
    check(weights_sha(m) == v3.sha256, "stop 2: new model kept")

    # a half-written staging directory next to a good final is dropped
    makedirs(m + ".tmp", exist_ok=True)
    with open(m + ".tmp/weights.bin", "w") as f:
        f.write("partial")
    recover_dir(m)
    check(not exists(m + ".tmp") and weights_sha(m) == v3.sha256, "stale .tmp dropped, final untouched")

    # full swap, and a swap onto a path that does not exist yet
    var v4 = write_tiny(root + "/o", "o", 4)
    copy_dir(root + "/o", m + ".tmp")
    atomic_replace_dir(m + ".tmp", m)
    check(weights_sha(m) == v4.sha256 and not exists(m + ".tmp") and not exists(m + ".old"), "atomic_replace_dir")
    copy_dir(root + "/o", root + "/fresh.tmp")
    atomic_replace_dir(root + "/fresh.tmp", root + "/fresh")
    check(weights_sha(root + "/fresh") == v4.sha256 and not exists(root + "/fresh.tmp"), "swap onto a new path")
    recover_dir(root + "/never_written")  # nothing to do, must not raise

    # a failed write leaves the previous directory untouched and cleans its staging directory
    var over = tiny_tensors(9, False)
    over[0].data[3] = 1.0e6  # beyond the f16 range
    var raised = False
    try:
        _ = write_model_dir(m, tiny_head("m", "[6]"), over, "f16", List[String](), "")
    except e:
        raised = True
        expect_error(String(e), "QM-E645", "f16 overflow")
        check("embeddings.weight" in String(e), "overflow message names the tensor")
    check(raised, "f16 overflow must raise")
    check(weights_sha(m) == v4.sha256 and not exists(m + ".tmp") and not exists(m + ".old"), "overflow left the old model")
    # the same tensor is fine as f32
    var f32_info = write_model_dir(root + "/big", tiny_head("big", "[6]"), over, "f32", List[String](), "")
    var lm = load_model_file(root + "/big")
    check(lm.tensors[0][3] == 1.0e6, "f32 tensor keeps 1e6")
    check(f32_info.weights_bytes % 64 == 0, "f32 padding")

    # a manifest the loader rejects: read-back fails, old model untouched, staging removed
    raised = False
    try:
        _ = write_model_dir(m, '"id":"m"', tiny_tensors(9, False), "f16", List[String](), "")
    except e:
        raised = True
        expect_error(String(e), "QM-E644", "bad manifest")
    check(raised, "a manifest without format must fail the read-back")
    check(weights_sha(m) == v4.sha256 and not exists(m + ".tmp") and not exists(m + ".old"), "bad manifest left the old model")

    # usage errors
    raised = False
    try:
        _ = write_model_dir(m, tiny_head("m", "[6]") + ",", tiny_tensors(9, False), "f16", List[String](), "")
    except e:
        raised = True
        expect_error(String(e), "QM-E640", "trailing comma")
    check(raised, "head with a trailing comma must raise")
    raised = False
    try:
        _ = write_model_dir(m, tiny_head("m", "[6]"), tiny_tensors(9, False), "f64", List[String](), "")
    except e:
        raised = True
        expect_error(String(e), "QM-E640", "dtype")
    check(raised, "dtype f64 must raise")
    var wrong = tiny_tensors(9, False)
    wrong[0].shape = [9, 9]
    raised = False
    try:
        _ = write_model_dir(m, tiny_head("m", "[6]"), wrong, "f16", List[String](), "")
    except e:
        raised = True
        expect_error(String(e), "QM-E640", "shape")
    check(raised, "shape/data mismatch must raise")
    check(weights_sha(m) == v4.sha256, "usage errors left the old model")


def test_dtype_rules_and_layout() raises:
    var root = WORK + "/dt"
    remove_tree(root)
    var tensors = tiny_tensors(21, False)
    tensors[2].dtype = String("f32")  # common.0.weight forced to f32 by the tensor itself
    var names: List[String] = ["action.0.weight"]
    var info = write_model_dir(root + "/m", tiny_head("m", "[6]"), tensors, "f16", names, '"extra":{"a":1}')
    var man = parse_json(read_text(root + "/m/manifest.json"))
    var tj = man.get("tensors")
    var dts = List[String]()
    for i in range(len(tj)):
        dts.append(tj.at(i).get("dtype").as_string())
        check(tj.at(i).get("offset").as_int() % 64 == 0, "offset aligned: " + tj.at(i).get("name").as_string())
    var want: List[String] = ["f16", "f32", "f32", "f32", "f32", "f32", "f16", "f32"]
    check(dts == want, "dtype per tensor")
    check(man.get("extra").get("a").as_int() == 1, "manifest tail member written")
    check(man.get("weights_bytes").as_int() == info.weights_bytes, "weights_bytes member")
    check(man.get("sha256").as_string() == info.sha256 and man.get("fnv1a64").as_string() == info.fnv1a64, "digest members")
    # trailing slash and a nested missing parent directory
    var info2 = write_tiny(root + "/a/b/c/", "c", 5)
    check(weights_sha(root + "/a/b/c") == info2.sha256, "nested directory with a trailing slash")
    # write_text_atomic replaces a file in one step
    write_text_atomic(root + "/note.txt", "one")
    write_text_atomic(root + "/note.txt", "two")
    check(read_text(root + "/note.txt") == "two" and not exists(root + "/note.txt.tmp"), "write_text_atomic")


def naive_forward(
    lm_tensors: List[List[Float32]], idx: List[Int], emb: Int, mut logits: List[Float64]
) -> Float64:
    """Plain Float64 forward over decoded tensors, two common layers, one action and one value layer."""
    var h = List[Float64](length=emb, fill=0.0)
    for e in range(emb):
        h[e] = Float64(lm_tensors[1][e])
    for i in idx:
        for e in range(emb):
            h[e] += Float64(lm_tensors[0][i * emb + e])
    for e in range(emb):
        if h[e] < 0.0:
            h[e] = 0.0
    var cur = h.copy()
    # common.0 [6, emb], common.1 [5, 6]
    var sizes: List[Int] = [6, 5]
    var in_n = emb
    var ti = 2
    for l in range(2):
        var out_n = sizes[l]
        var nxt = List[Float64](length=out_n, fill=0.0)
        for o in range(out_n):
            var s = Float64(lm_tensors[ti + 1][o])
            for k in range(in_n):
                s += Float64(lm_tensors[ti][o * in_n + k]) * cur[k]
            nxt[o] = s if s > 0.0 else 0.0
        cur = nxt^
        in_n = out_n
        ti += 2
    logits = List[Float64](length=2, fill=0.0)
    for o in range(2):
        var s = Float64(lm_tensors[ti + 1][o])
        for k in range(in_n):
            s += Float64(lm_tensors[ti][o * in_n + k]) * cur[k]
        logits[o] = s
    var v = Float64(lm_tensors[ti + 3][0])
    for k in range(in_n):
        v += Float64(lm_tensors[ti + 2][k]) * cur[k]
    return v


def test_two_common_layers() raises:
    var root = WORK + "/two"
    remove_tree(root)
    var head = tiny_head("two", "[6,5]")
    _ = write_model_dir(root + "/two", head, tiny_tensors(33, True), "f16", List[String](), "")
    var lm = load_model_file(root + "/two")
    var want: List[String] = [
        "embeddings.weight", "embeddings.bias", "common.0.weight", "common.0.bias", "common.1.weight", "common.1.bias",
        "action.0.weight", "action.0.bias", "value.0.weight", "value.0.bias",
    ]
    check(lm.names == want, "consecutive tensor names")
    var pol = MLPPolicy.from_model(lm)
    check(len(pol.common) == 2 and pol.num_actions == 2 and pol.emb == 8 and pol.obs_size == 9, "two-layer policy shape")
    var obs: List[Int] = [0, 4, 8]
    var logits = List[Float32]()
    var v = pol.forward(obs, logits)
    var nl = List[Float64]()
    var nv = naive_forward(lm.tensors, obs, 8, nl)
    check(abs(Float64(v) - nv) < 1e-5, "value vs Float64 forward: " + String(v) + " vs " + String(nv))
    for a in range(2):
        check(abs(Float64(logits[a]) - nl[a]) < 1e-5, "logit " + String(a))
    # a model whose second common layer is numbered 2 (a gap) must not load both layers
    var gap = tiny_tensors(33, True)
    gap[4].name = String("common.2.weight")
    gap[5].name = String("common.2.bias")
    _ = write_model_dir(root + "/gap", head, gap, "f16", List[String](), "")
    var gm = load_model_file(root + "/gap")
    var raised = False
    try:
        _ = MLPPolicy.from_model(gm)
    except e:
        raised = True
    check(raised, "a numbering gap must not load (action head no longer chains)")


def main() raises:
    makedirs(WORK, exist_ok=True)
    remove_tree(WORK + "/models")
    test_reencode_shipped()
    test_index_regeneration()
    test_dump_indent1()
    test_split_manifest()
    test_verify_shipped()
    test_verify_failures()
    test_verify_layouts()
    test_atomic()
    test_dtype_rules_and_layout()
    test_two_common_layers()
    print("test_rl_container OK")
