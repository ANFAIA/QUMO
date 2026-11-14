"""Tests `qumo-agent tools list | show | review | revoke | remove | verify` on a temporary
store (DESIGN.md 6.6, 7.1): states, hash verification, the name prompt (the name is piped on
stdin), `--yes` refused without a terminal (usage error), revoke, remove (never deletes
outside the store), and malformed entries. The rebuild of `review` is covered by
test_app_toolsmith_e2e."""

from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import parse_json

from qumo_agent.app.testing import app_env, run_app
from qumo_agent.native.fs import exists, make_dirs, make_temp_dir, read_text_lossy, remove_tree, write_text
from qumo_agent.toolsmith.manifest import (
    ToolEntry, Manifest, load_manifest, entry_dir, current_platform, empty_manifest,
)
from qumo_agent.util.sha256 import sha256_hex_text


def make_entry(store: String, name: String, version: String, reviewed: Bool) raises -> ToolEntry:
    """Writes tool.mojo, cases.json and a stand-in binary into the store and returns a
    matching manifest entry."""
    var source = "def main(): pass  # " + name + "\n"
    var cases = "{\"schema\": \"qumo.tool_cases.v1\", \"tool\": \"" + name + "\", \"cases\": []}\n"
    var binary = "not a real binary " + name + "\n"
    var sha_bin = sha256_hex_text(binary)
    var dir = entry_dir(name, version, sha_bin)
    make_dirs(store + "/" + dir)
    write_text(store + "/" + dir + "/tool.mojo", source)
    write_text(store + "/" + dir + "/cases.json", cases)
    write_text(store + "/" + dir + "/" + name, binary)
    return ToolEntry(
        name, version, "Demo tool " + name, dir, sha256_hex_text(source), sha256_hex_text(cases), sha_bin, "sdk",
        "Mojo 1.2.0", current_platform(), 30, 1048576, "2026-11-06T12:00:00+00:00", "mojo-toolsmith",
        "agents/demo-branch", 2, 2, reviewed, "alice" if reviewed else String(), "2026-11-07T08:00:00+00:00" if reviewed else String(),
    )


def tools(store: String, args: List[String], stdin_data: String) raises -> Tuple[Int, String, String]:
    var full = List[String]()
    full.append("tools")
    for a in args:
        full.append(a)
    full.append("--tools-dir")
    full.append(store)
    var r = run_app(full, app_env(List[String]()), stdin_data, 60000)
    return (r.exit_code, r.stdout, r.stderr)


def test_list_show_verify() raises:
    var dir = make_temp_dir("qumo-app-tools-")
    var store = dir + "/store"
    var r = tools(store, ["list"], "")
    assert_equal(r[0], 0, r[2])
    assert_true(r[1].find("no generated tools in " + store) >= 0, r[1])

    var m = empty_manifest(store)
    m.put(make_entry(store, "gen_demo", "0.1.0", False))
    m.put(make_entry(store, "gen_done", "1.2.3", True))
    m.save()
    r = tools(store, ["list"], "")
    assert_equal(r[0], 0, r[2])
    assert_true(r[1].find("gen_demo") >= 0 and r[1].find("UNREVIEWED") >= 0, r[1])
    assert_true(r[1].find("reviewed by alice on 2026-11-07") >= 0, r[1])
    assert_true(r[1].find("tests 2/2") >= 0, r[1])
    assert_true(r[1].find("Demo tool gen_demo") >= 0)

    r = tools(store, ["show", "gen_demo"], "")
    assert_equal(r[0], 0, r[1] + r[2])
    assert_true(r[1].find("\"name\": \"gen_demo\"") >= 0, r[1])
    assert_true(r[1].find("verification: the stored source, cases and binary match their hashes") >= 0, r[1])

    r = tools(store, ["verify"], "")
    assert_equal(r[0], 0, r[1] + r[2])
    assert_true(r[1].find("[ok] gen_demo 0.1.0") >= 0 and r[1].find("[ok] gen_done 1.2.3") >= 0, r[1])
    r = tools(store, ["verify", "gen_demo"], "")
    assert_equal(r[0], 0)
    r = tools(store, ["verify", "gen_nope"], "")
    assert_equal(r[0], 1)
    assert_true(r[2].find("QA-E150: no generated tool gen_nope") >= 0, r[2])

    # A tampered binary fails verification, show and review (exit 1) and changes nothing.
    var e = m.get("gen_demo").value().copy()
    write_text(store + "/" + e.dir + "/gen_demo", "tampered\n")
    r = tools(store, ["verify"], "")
    assert_equal(r[0], 1)
    assert_true(r[1].find("[--] gen_demo") >= 0 and r[1].find("does not match its manifest hash") >= 0, r[1])
    assert_true(r[1].find("[ok] gen_done") >= 0, r[1])
    r = tools(store, ["show", "gen_demo"], "")
    assert_equal(r[0], 1)
    assert_true(r[1].find("verification: FAILED") >= 0, r[1])
    r = tools(store, ["review", "gen_demo"], "gen_demo\n")
    assert_equal(r[0], 1)
    assert_true(r[1].find("nothing changed") >= 0, r[1])
    assert_false(load_manifest(store).get("gen_demo").value().reviewed)
    remove_tree(dir)


def test_review_prompt_and_usage() raises:
    var dir = make_temp_dir("qumo-app-tools-review-")
    var store = dir + "/store"
    var m = empty_manifest(store)
    m.put(make_entry(store, "gen_demo", "0.1.0", False))
    m.save()
    # Anything but the tool's name aborts with exit 1; the provenance and source were shown.
    var r = tools(store, ["review", "gen_demo"], "yes\n")
    assert_equal(r[0], 1)
    assert_true(r[1].find("Type the tool name to mark it reviewed: ") >= 0, r[1])
    assert_true(r[1].find("--- tool.mojo ---") >= 0 and r[1].find("def main(): pass  # gen_demo") >= 0, r[1])
    assert_true(r[1].find("--- cases.json ---") >= 0, r[1])
    assert_true(r[1].find("published   2026-11-06T12:00:00+00:00 by mojo-toolsmith (branch agents/demo-branch)") >= 0, r[1])
    assert_true(r[1].find("aborted: the name did not match; nothing changed") >= 0, r[1])
    r = tools(store, ["review", "gen_demo"], "")
    assert_equal(r[0], 1)
    assert_false(load_manifest(store).get("gen_demo").value().reviewed)

    # --yes needs a terminal on stdin: with a pipe it is a usage error (exit 2) before anything is shown.
    r = tools(store, ["review", "gen_demo", "--yes"], "gen_demo\n")
    assert_equal(r[0], 2)
    assert_true(r[2].find("--yes needs a terminal") >= 0, r[2])
    assert_false(r[1].find("--- tool.mojo ---") >= 0)

    # Usage errors.
    r = tools(store, List[String](), "")
    assert_equal(r[0], 2)
    r = tools(store, ["review"], "")
    assert_equal(r[0], 2)
    r = tools(store, ["frobnicate"], "")
    assert_equal(r[0], 2)
    r = tools(store, ["list", "extra"], "")
    assert_equal(r[0], 2)
    r = tools(store, ["revoke", "gen_demo", "--yes"], "")
    assert_equal(r[0], 2)
    r = tools(store, ["review", "gen_nope"], "gen_nope\n")
    assert_equal(r[0], 1)
    assert_true(r[2].find("QA-E150: no generated tool gen_nope") >= 0, r[2])
    remove_tree(dir)


def test_revoke_and_remove() raises:
    var dir = make_temp_dir("qumo-app-tools-rm-")
    var store = dir + "/store"
    var m = empty_manifest(store)
    m.put(make_entry(store, "gen_done", "1.2.3", True))
    m.put(make_entry(store, "gen_keep", "0.2.0", True))
    m.save()
    var r = tools(store, ["revoke", "gen_done"], "")
    assert_equal(r[0], 0, r[2])
    var e = load_manifest(store).get("gen_done").value().copy()
    assert_false(e.reviewed)
    assert_equal(e.reviewed_by, "")
    assert_true(load_manifest(store).get("gen_keep").value().reviewed)
    r = tools(store, ["revoke", "gen_nope"], "")
    assert_equal(r[0], 1)

    var dir_done = store + "/" + e.dir
    assert_true(exists(dir_done))
    r = tools(store, ["remove", "gen_done"], "")
    assert_equal(r[0], 0, r[2])
    assert_false(exists(dir_done))
    assert_false(exists(store + "/gen_done"))                       # the empty name directory goes too
    assert_true(exists(store + "/manifest.json"))
    assert_false(load_manifest(store).get("gen_done").__bool__())
    assert_true(load_manifest(store).get("gen_keep").__bool__())
    r = tools(store, ["remove", "gen_done"], "")
    assert_equal(r[0], 1)

    # A malformed entry (its dir points outside the store): `list` flags it, `remove` drops
    # the manifest entry and deletes no file.
    make_dirs(dir + "/outside")
    write_text(dir + "/outside/precious.txt", "keep me\n")
    var bad = make_entry(store, "gen_bad", "0.1.0", True)
    bad.dir = "../outside"
    var m2 = load_manifest(store)
    m2.put(bad)
    m2.save()
    r = tools(store, ["list"], "")
    assert_true(r[1].find("MALFORMED") >= 0, r[1])
    r = tools(store, ["verify", "gen_bad"], "")
    assert_equal(r[0], 1)
    assert_true(r[1].find("malformed manifest entry") >= 0, r[1])
    r = tools(store, ["remove", "gen_bad"], "")
    assert_equal(r[0], 0, r[2])
    assert_true(exists(dir + "/outside/precious.txt"))
    assert_true(r[1].find("no files were deleted") >= 0, r[1])
    assert_false(load_manifest(store).get("gen_bad").__bool__())
    remove_tree(dir)


def main() raises:
    test_list_show_verify()
    test_review_prompt_and_usage()
    test_revoke_and_remove()
    print("test_app_tools_cmd OK")
