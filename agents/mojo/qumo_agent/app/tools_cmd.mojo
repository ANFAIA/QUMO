"""`qumo-agent tools list | show | review | revoke | remove | verify` (DESIGN.md 6.6, 7.1).

These commands are the human side of the generated-tool flow: nothing here runs on behalf
of a model. `review` is the only way a tool becomes `reviewed`: it checks the stored hashes,
prints the provenance and the stored source, asks the human to type the tool name (or takes
`--yes` from a terminal only), then rebuilds the binary from exactly the source that was
shown and re-runs the stored cases (`review_rebuild`) before it marks the entry reviewed.
"""

from std.os import getenv
from std.sys import stderr

from qumo.ai.json import parse_json
from qumo.ai.mcp import LineReader

from qumo_agent.native.clock import utc_iso_now
from qumo_agent.native.fs import exists, is_dir, join_path, list_dir, read_text_lossy, realpath_strict, remove_tree
from qumo_agent.native.libc import c_isatty
from qumo_agent.native.sandbox import sandbox_kind
from qumo_agent.toolsmith.manifest import (
    Manifest, ToolEntry, check_entry_shape, load_manifest, sdk_sha256, tool_store_dir, verify_entry,
)
from qumo_agent.toolsmith.tools import review_rebuild, SANDBOX_KIND_UNSANDBOXED
from qumo_agent.util.pyfmt import cp_head, py_json_dumps

from qumo_agent.app.options import AppOptions

comptime TOOLS_USAGE = "usage: qumo-agent tools list | show NAME | review NAME [--yes] | revoke NAME | remove NAME | verify [NAME]"


def _error(msg: String) -> Int:
    print("qumo-agent: error: " + msg, file=stderr)
    return 1


def _usage_error(msg: String) -> Int:
    print("qumo-agent: " + msg, file=stderr)
    print(TOOLS_USAGE, file=stderr)
    return 2


def _pad(s: String, width: Int) -> String:
    var out = s
    while out.byte_length() < width:
        out += " "
    return out^


def _state(e: ToolEntry) -> String:
    if e.reviewed:
        var day = cp_head(e.reviewed_at, 10)
        return "reviewed by " + e.reviewed_by + " on " + day
    return "UNREVIEWED"


def _store(opts: AppOptions) -> String:
    if opts.tools_store.byte_length() > 0:
        return opts.tools_store
    return tool_store_dir(opts.root)


def _list(store: String) raises -> Int:
    var m = load_manifest(store)
    if len(m.entries) == 0:
        print("no generated tools in " + store)
        return 0
    for e in m.entries:
        var shape = check_entry_shape(store, e)
        var state = _state(e) if shape.ok else String("MALFORMED")
        print(_pad(e.name, 30) + " " + _pad(e.version, 12) + " " + _pad(state, 36) + " tests " + String(e.tests_passed)
              + "/" + String(e.tests_total) + "  " + cp_head(e.description.replace("\n", " "), 60))
    return 0


def _find(m: Manifest, name: String, store: String) raises -> ToolEntry:
    var e = m.get(name)
    if not e:
        raise Error("QA-E150: no generated tool " + name + " in " + store)
    return e.value().copy()


def _show(store: String, name: String) raises -> Int:
    var m = load_manifest(store)
    var e = _find(m, name, store)
    print(py_json_dumps(parse_json(e.to_json()), False, 1))
    var check = verify_entry(store, e)
    if check.ok:
        print("verification: the stored source, cases and binary match their hashes")
    else:
        print("verification: FAILED")
        for p in check.problems:
            print("  " + p)
    return 0 if check.ok else 1


def _verify(store: String, name: String) raises -> Int:
    var m = load_manifest(store)
    var names = List[String]()
    if name.byte_length() > 0:
        _ = _find(m, name, store)
        names.append(name)
    else:
        for e in m.entries:
            names.append(e.name)
    var bad = 0
    for n in names:
        var e = _find(m, n, store)
        var check = verify_entry(store, e)
        if check.ok:
            print("[ok] " + n + " " + e.version)
        else:
            bad += 1
            print("[--] " + n)
            for p in check.problems:
                print("     " + p)
    if len(names) == 0:
        print("no generated tools in " + store)
    return 1 if bad > 0 else 0


def _revoke(store: String, name: String) raises -> Int:
    var m = load_manifest(store)
    var e = _find(m, name, store)
    e.reviewed = False
    e.reviewed_by = String()
    e.reviewed_at = String()
    m.put(e)
    m.save()
    print("revoked " + name + ": agents no longer load it until it is reviewed again")
    return 0


def _remove(store: String, name: String) raises -> Int:
    var m = load_manifest(store)
    var e = _find(m, name, store)
    var shape = check_entry_shape(store, e)
    if shape.ok:
        var dir = join_path(realpath_strict(store), e.dir)
        if exists(dir):
            remove_tree(dir)
        var parent = join_path(realpath_strict(store), e.name)
        if is_dir(parent) and len(list_dir(parent)) == 0:
            remove_tree(parent)
    else:
        print("note: the manifest entry is malformed, so no files were deleted")
    _ = m.remove(name)
    m.save()
    print("removed " + name)
    return 0


def _reviewer() -> String:
    var u = getenv("USER", "")
    if u.byte_length() > 0:
        return u
    var l = getenv("LOGNAME", "")
    if l.byte_length() > 0:
        return l
    return String("unknown")


def _review(opts: AppOptions, store: String, name: String, yes: Bool) raises -> Int:
    var m = load_manifest(store)
    var e = _find(m, name, store)
    var shape = check_entry_shape(store, e)
    if not shape.ok:
        for p in shape.problems:
            print(p)
        return 1
    var check = verify_entry(store, e)
    if not check.ok:
        for p in check.problems:
            print(p)
        print("nothing changed")
        return 1
    var dir = join_path(realpath_strict(store), e.dir)
    var repo_root = realpath_strict(opts.root)
    print("tool        " + e.name + " " + e.version)
    print("description " + e.description)
    print("published   " + e.published_at + " by " + e.published_by + " (branch " + e.sandbox_branch + ")")
    print("built with  " + e.mojo_version + " on " + e.platform + "; tests " + String(e.tests_passed) + "/" + String(e.tests_total))
    print("sha256      source " + e.source_sha256)
    print("            cases  " + e.cases_sha256)
    print("            binary " + e.binary_sha256)
    if e.sdk_sha256 != sdk_sha256(repo_root):
        print("note: the tool SDK changed since this binary was built; the rebuild below uses the current SDK")
    print("")
    print("--- tool.mojo ---")
    print(read_text_lossy(join_path(dir, "tool.mojo")))
    print("--- cases.json ---")
    print(read_text_lossy(join_path(dir, "cases.json")))
    print("---")
    if not yes:
        print("Type the tool name to mark it reviewed: ", end="", flush=True)
        var reader = LineReader(0)
        var line = String()
        var got = reader.read_line(line)
        if not got or String(line.strip()) != name:
            print("aborted: the name did not match; nothing changed")
            return 1
    var kind = sandbox_kind(opts.self_binary)
    if kind == "none":
        if not opts.allow_unsandboxed_tool_tests:
            print("QA-E163: no OS sandbox is available on this machine, so the stored cases cannot be run safely. "
                  "Hint: a human can pass --allow-unsandboxed-tool-tests")
            return 1
        kind = String(SANDBOX_KIND_UNSANDBOXED)
    print("rebuilding from the stored source and re-running the stored cases ...", flush=True)
    var updated = review_rebuild(repo_root, store, e, opts.self_binary, kind)
    updated.reviewed = True
    updated.reviewed_by = _reviewer()
    updated.reviewed_at = utc_iso_now()
    m.put(updated)
    m.save()
    print("reviewed " + name + " " + updated.version + " (binary " + cp_head(updated.binary_sha256, 12) + ", tests "
          + String(updated.tests_passed) + "/" + String(updated.tests_total) + ")")
    return 0


def run_tools_command(opts: AppOptions, args: List[String], yes: Bool) -> Int:
    """`args` are the words after `tools` (subcommand and name). Returns the exit code."""
    if len(args) == 0:
        return _usage_error("tools needs a subcommand")
    var sub = args[0]
    var store = _store(opts)
    try:
        if sub == "list":
            if len(args) != 1:
                return _usage_error("tools list takes no argument")
            return _list(store)
        if sub == "verify":
            if len(args) > 2:
                return _usage_error("tools verify takes at most one name")
            return _verify(store, args[1] if len(args) == 2 else String())
        if sub == "show" or sub == "review" or sub == "revoke" or sub == "remove":
            if len(args) != 2:
                return _usage_error("tools " + sub + " needs a tool name")
            if yes and sub != "review":
                return _usage_error("--yes only applies to tools review")
            if sub == "show":
                return _show(store, args[1])
            if sub == "revoke":
                return _revoke(store, args[1])
            if sub == "remove":
                return _remove(store, args[1])
            if yes and c_isatty(0) != 1:
                return _usage_error("--yes needs a terminal on stdin; pipe the tool name instead")
            return _review(opts, store, args[1], yes)
        return _usage_error("unknown tools subcommand " + sub)
    except e:
        return _error(String(e))
