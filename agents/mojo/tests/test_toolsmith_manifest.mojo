"""Manifest, names, versions, entry shapes, hash verification and build records
(DESIGN.md 2.9, 6.4, 6.5). Offline; works in a temporary store."""

from std.os import getenv, setenv, unsetenv
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import parse_json

from qumo_agent.native.fs import (
    exists, join_path, make_dirs, make_temp_dir, read_text_lossy, realpath_strict, remove_tree,
    symlink, write_text,
)
from qumo_agent.util.sha256 import sha256_file_hex, sha256_hex_text
from qumo_agent.toolsmith.manifest import (
    BuildRecord, Manifest, ToolEntry, MANIFEST_SCHEMA, build_record_from_json, check_entry_shape,
    current_platform, entry_dir, load_manifest, sdk_files, sdk_sha256, tool_store_dir,
    valid_tool_name, valid_tool_version, verify_entry,
)


def make_entry(name: String, version: String, sha: String) -> ToolEntry:
    return ToolEntry(name, version, "desc of " + name, entry_dir(name, version, sha),
                     String("a") * 64, String("b") * 64, sha, String("c") * 64, "Mojo 1.2.0 (2016d0cd)",
                     current_platform(), 30, 1048576, "2026-11-06T12:00:00.123456+00:00",
                     "mojo-toolsmith", "agents/x-20261106-120000", 2, 2, False, "", "")


def test_names_and_versions() raises:
    assert_true(valid_tool_name("gen_ab"))
    assert_true(valid_tool_name("gen_bitstring_stats"))
    assert_true(valid_tool_name("gen_" + String("a") * 40))
    assert_false(valid_tool_name("gen_" + String("a") * 41))
    assert_false(valid_tool_name("gen_a"))
    assert_false(valid_tool_name("gen_Ab"))
    assert_false(valid_tool_name("gen_a-b"))
    assert_false(valid_tool_name("tool_ab"))
    assert_false(valid_tool_name("gen_../x"))
    assert_false(valid_tool_name(""))

    assert_true(valid_tool_version("0.1.0"))
    assert_true(valid_tool_version("1.2.3-rc1"))
    assert_true(valid_tool_version("1.2.3.post1"))
    assert_true(valid_tool_version("9999.9999.9999-" + String("a") * 16))
    assert_false(valid_tool_version("9999.9999.9999-" + String("a") * 17))
    assert_false(valid_tool_version("10000.0.0"))
    assert_false(valid_tool_version("1.2"))
    assert_false(valid_tool_version("../../x"))
    assert_false(valid_tool_version("1.2.3-"))
    assert_false(valid_tool_version("1.2.3_rc1"))
    assert_false(valid_tool_version("1.2.3-rc/1"))
    assert_false(valid_tool_version("1..3"))
    assert_false(valid_tool_version(""))
    assert_equal(entry_dir("gen_x1", "0.1.0", "3f2a9c0d1e2b" + String("0") * 52), "gen_x1/0.1.0-3f2a9c0d1e2b")
    var p = current_platform()
    assert_true(p == "darwin-arm64" or p == "darwin-x86_64" or p == "linux-x86_64" or p == "linux-aarch64")


def test_shapes(store: String) raises:
    var sha = String("0123456789ab") + String("f") * 52
    var good = make_entry("gen_good", "1.2.3-rc1", sha)
    assert_true(check_entry_shape(store, good).ok)

    var e = make_entry("gen_bad", "../../x", sha)
    var c = check_entry_shape(store, e)
    assert_false(c.ok)
    assert_equal(c.problems[0], "QA-E150: gen_bad: malformed manifest entry (version must look like 1.2.3 or 1.2.3-rc1)")

    e = make_entry("gen_bad", "1.2", sha)
    assert_false(check_entry_shape(store, e).ok)

    e = make_entry("gen_ok", "1.0.0", sha)
    e.dir = "gen_ok/../../outside"
    c = check_entry_shape(store, e)
    assert_false(c.ok)
    assert_equal(c.problems[0], "QA-E150: gen_ok: malformed manifest entry (dir must be a relative path without '..')")

    e.dir = "/etc"
    assert_false(check_entry_shape(store, e).ok)

    e.dir = "gen_ok/1.0.0-000000000000"
    c = check_entry_shape(store, e)
    assert_false(c.ok)
    assert_equal(c.problems[0], "QA-E150: gen_ok: malformed manifest entry (dir must equal <name>/<version>-<binary_sha256[:12]>)")

    e = make_entry("gen_ok", "1.0.0", "ABC")
    e.dir = "gen_ok/1.0.0-ABC"
    assert_false(check_entry_shape(store, e).ok)

    # A name with a newline is shown as a Python repr, never raw.
    e = make_entry("gen_x\ny", "1.0.0", sha)
    c = check_entry_shape(store, e)
    assert_false(c.ok)
    assert_true(c.problems[0].startswith("QA-E150: 'gen_x\\ny': malformed manifest entry (name"))

    # A directory that resolves outside the store (a planted symlink) is refused.
    var outside = make_temp_dir("qumo-ts-outside-")
    e = make_entry("gen_link", "1.0.0", sha)
    make_dirs(join_path(store, "gen_link"))
    symlink(outside, join_path(store, e.dir))
    c = check_entry_shape(store, e)
    assert_false(c.ok)
    assert_equal(c.problems[0], "QA-E150: gen_link: malformed manifest entry (dir resolves outside the store)")
    remove_tree(join_path(store, "gen_link"))
    assert_true(exists(outside))   # remove_tree never follows the link
    remove_tree(outside)


def publish_files(store: String, mut e: ToolEntry) raises:
    """Write tool.mojo, cases.json and a fake binary for `e` into the store and set
    its hashes and dir."""
    var tmp = make_temp_dir("qumo-ts-bin-")
    var bin = join_path(tmp, e.name)
    write_text(bin, "binary of " + e.name)
    e.binary_sha256 = sha256_file_hex(bin)
    e.dir = entry_dir(e.name, e.version, e.binary_sha256)
    var d = join_path(store, e.dir)
    make_dirs(d)
    write_text(join_path(d, "tool.mojo"), "source of " + e.name)
    write_text(join_path(d, "cases.json"), "cases of " + e.name)
    write_text(join_path(d, e.name), "binary of " + e.name)
    e.source_sha256 = sha256_file_hex(join_path(d, "tool.mojo"))
    e.cases_sha256 = sha256_file_hex(join_path(d, "cases.json"))
    remove_tree(tmp)


def test_round_trip_and_verify(store: String) raises:
    var empty = load_manifest(store)
    assert_equal(len(empty.entries), 0)
    assert_equal(empty.path, join_path(store, "manifest.json"))

    var m = empty.copy()
    var b = make_entry("gen_zeta", "0.1.0", String("0") * 64)
    publish_files(store, b)
    var a = make_entry("gen_alpha", "2.0.1-rc2", String("0") * 64)
    publish_files(store, a)
    a.reviewed = True
    a.reviewed_by = "ana"
    a.reviewed_at = "2026-11-07T10:00:00+00:00"
    m.put(b)
    m.put(a)
    m.save()

    var text = read_text_lossy(m.path)
    assert_true(text.startswith("{\n \"schema\": \"qumo.agent_tools.v1\",\n \"tools\": {\n  \"gen_alpha\": {\n   \"name\": \"gen_alpha\",\n"))
    assert_true(text.find("\"tests\": {\n    \"passed\": 2,\n    \"total\": 2\n   },") > 0)
    assert_true(text.find("gen_alpha") < text.find("gen_zeta"))   # sorted by name
    assert_true(text.endswith("}\n"))

    var back = load_manifest(store)
    assert_equal(len(back.entries), 2)
    var got = back.get("gen_alpha").value().copy()
    assert_equal(got.to_json(), a.to_json())
    assert_equal(back.get("gen_zeta").value().to_json(), b.to_json())
    assert_false(back.get("gen_none"))

    assert_true(verify_entry(store, got).ok)
    # Tampering with any stored file is caught.
    write_text(join_path(join_path(store, got.dir), "gen_alpha"), "evil binary")
    var v = verify_entry(store, got)
    assert_false(v.ok)
    assert_equal(v.problems[0], "QA-E150: gen_alpha: the binary does not match its manifest hash")
    write_text(join_path(join_path(store, got.dir), "gen_alpha"), "binary of gen_alpha")
    assert_true(verify_entry(store, got).ok)
    write_text(join_path(join_path(store, got.dir), "cases.json"), "{}")
    assert_equal(verify_entry(store, got).problems[0], "QA-E150: gen_alpha: cases.json does not match its manifest hash")
    write_text(join_path(join_path(store, got.dir), "cases.json"), "cases of gen_alpha")
    remove_tree(join_path(join_path(store, got.dir), "tool.mojo"))
    assert_equal(verify_entry(store, got).problems[0], "QA-E150: gen_alpha: tool.mojo is missing from the store")
    write_text(join_path(join_path(store, got.dir), "tool.mojo"), "source of gen_alpha")
    got.platform = "plan9-mips"
    assert_equal(verify_entry(store, got).problems[0], "QA-E150: gen_alpha: built for 'plan9-mips', this machine is " + current_platform())

    # remove/put
    assert_true(back.remove("gen_zeta"))
    assert_false(back.remove("gen_zeta"))
    assert_equal(len(back.entries), 1)

    # A malformed entry is kept on load and refused by verify_entry.
    var raw = read_text_lossy(m.path).replace("\"version\": \"0.1.0\"", "\"version\": \"../../x\"")
    write_text(m.path, raw)
    var flagged = load_manifest(store)
    assert_equal(len(flagged.entries), 2)
    var z = flagged.get("gen_zeta").value().copy()
    assert_equal(z.version, "../../x")
    var vz = verify_entry(store, z)
    assert_false(vz.ok)
    assert_true(vz.problems[0].startswith("QA-E150: gen_zeta: malformed manifest entry (version"))

    # Missing fields become empty values (and fail the shape check), never an error.
    write_text(m.path, "{\"schema\": \"qumo.agent_tools.v1\", \"tools\": {\"gen_q1\": {\"reviewed\": \"yes\"}}}")
    var sparse = load_manifest(store)
    var q = sparse.entries[0].copy()
    assert_equal(q.name, "gen_q1")
    assert_false(q.reviewed)
    assert_equal(q.timeout_s, 30)
    assert_false(check_entry_shape(store, q).ok)

    # A file that is not a manifest raises QA-E150.
    write_text(m.path, "{not json")
    var raised = False
    try:
        _ = load_manifest(store)
    except e:
        raised = True
        assert_true(String(e).startswith("QA-E150: " + m.path + ": not valid JSON"))
    assert_true(raised)
    write_text(m.path, "{\"schema\": \"other\", \"tools\": {}}")
    raised = False
    try:
        _ = load_manifest(store)
    except e:
        raised = True
        assert_equal(String(e), "QA-E150: " + m.path + ": schema must be qumo.agent_tools.v1")
    assert_true(raised)


def test_build_record_and_sdk() raises:
    var rec = BuildRecord("s", "k", "b", "Mojo 1.2.0 (2016d0cd)", "2026-11-07T00:00:00+00:00")
    var text = rec.to_json()
    assert_equal(text, "{\"source_sha256\":\"s\",\"sdk_sha256\":\"k\",\"binary_sha256\":\"b\",\"mojo_version\":\"Mojo 1.2.0 (2016d0cd)\",\"built_at\":\"2026-11-07T00:00:00+00:00\"}")
    assert_equal(build_record_from_json(text).to_json(), text)
    var raised = False
    try:
        _ = build_record_from_json("[")
    except:
        raised = True
    assert_true(raised)

    var root = realpath_strict(".")
    var files = sdk_files()
    assert_equal(len(files), 3)
    assert_equal(files[0], "agents/mojo/qumo_agent/tool_sdk.mojo")
    var hashes = List[String]()
    for f in files:
        hashes.append(sha256_file_hex(join_path(root, f)))
    assert_equal(sdk_sha256(root), sha256_hex_text(hashes[0] + "\n" + hashes[1] + "\n" + hashes[2]))

    var saved = getenv("QUMO_AGENT_TOOLS_DIR", "")
    _ = unsetenv("QUMO_AGENT_TOOLS_DIR")
    assert_equal(tool_store_dir("/r"), "/r/build/agent-tools")
    _ = setenv("QUMO_AGENT_TOOLS_DIR", "/elsewhere/tools", True)
    assert_equal(tool_store_dir("/r"), "/elsewhere/tools")
    if saved.byte_length() > 0:
        _ = setenv("QUMO_AGENT_TOOLS_DIR", saved, True)
    else:
        _ = unsetenv("QUMO_AGENT_TOOLS_DIR")


def main() raises:
    test_names_and_versions()
    var store = realpath_strict(make_temp_dir("qumo-ts-store-"))
    test_shapes(store)
    test_round_trip_and_verify(store)
    test_build_record_and_sdk()
    remove_tree(store)
    print("test_toolsmith_manifest OK")
