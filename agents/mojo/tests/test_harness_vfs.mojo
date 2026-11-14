"""Tests harness/vfs.mojo: Python dict order, the reducer, VfsTx views, JSON round trip and
validate_vfs_path (Appendix C.1 vectors)."""

from std.testing import assert_equal, assert_true, assert_false

from qumo_agent.harness.vfs import (
    Vfs, VfsTx, FileData, apply_tx, vfs_from_json, new_file_data, updated_file_data,
    validate_vfs_path,
)


def _keys(v: Vfs) -> String:
    return String(",").join(v.paths())


def test_order_and_reducer() raises:
    var base = Vfs()
    base.set("/a", new_file_data("A", "t0"))
    base.set("/b", new_file_data("B", "t0"))
    base.set("/c", new_file_data("C", "t0"))
    base.set("/a", updated_file_data(base.get("/a").value(), "A2", "t1"))   # keeps position
    assert_equal(_keys(base), "/a,/b,/c")
    assert_equal(base.get("/a").value().created_at, "t0")
    assert_equal(base.get("/a").value().modified_at, "t1")
    var tx1 = VfsTx()
    tx1.set("/d", new_file_data("D", "t2"))
    tx1.delete("/b")
    tx1.set("/a", new_file_data("A3", "t2"))
    assert_true(Bool(tx1.get(base, "/d")))
    assert_false(Bool(tx1.get(base, "/b")))
    assert_equal(tx1.get(base, "/c").value().content, "C")
    assert_equal(_keys(tx1.view(base)), "/a,/c,/d")
    var tx2 = VfsTx()
    tx2.set("/b", new_file_data("B2", "t3"))   # a later call re-creates /b
    var state = base.copy()
    apply_tx(state, tx1)
    apply_tx(state, tx2)
    assert_equal(_keys(state), "/a,/c,/d,/b")
    assert_equal(state.get("/a").value().content, "A3")


def test_json_and_equals() raises:
    var v = Vfs()
    v.set("/n é.md", FileData("x\n\"y\"", "utf-8", "2026-11-06T12:00:00+00:00", "2026-11-06T12:00:01+00:00"))
    v.set("/b", new_file_data("", "t"))
    var back = vfs_from_json(v.to_json())
    assert_true(back.equals(v))
    assert_equal(_keys(back), "/n é.md,/b")
    var other = Vfs()
    other.set("/b", new_file_data("", "t"))
    other.set("/n é.md", FileData("x\n\"y\"", "utf-8", "2026-11-06T12:00:00+00:00", "2026-11-06T12:00:01+00:00"))
    assert_true(other.equals(v))     # order-insensitive
    other.set("/b", new_file_data("", "t2"))
    assert_false(other.equals(v))    # timestamps count
    assert_equal(vfs_from_json("").to_json(), "{}")


def test_validate_path() raises:
    assert_equal(validate_vfs_path("foo/bar"), "/foo/bar")
    assert_equal(validate_vfs_path("/./x//y/"), "/x/y")
    assert_equal(validate_vfs_path(""), "/.")
    assert_equal(validate_vfs_path("."), "/.")
    assert_equal(validate_vfs_path("/large_tool_results/call_1"), "/large_tool_results/call_1")
    assert_equal(validate_vfs_path("a\\b"), "/a/b")
    assert_equal(validate_vfs_path("//x"), "//x")
    assert_equal(validate_vfs_path("foo..bar.txt"), "/foo..bar.txt")
    var cases: List[String] = ["../etc", "a/../b", "~/x", "a\\..\\b", "C:/x"]
    var msgs: List[String] = [
        "Path traversal not allowed: ../etc",
        "Path traversal not allowed: a/../b",
        "Path traversal not allowed: ~/x",
        "Path traversal not allowed: a\\..\\b",
        "Windows absolute paths are not supported: C:/x. Please use virtual paths starting with / (e.g., /workspace/file.txt)",
    ]
    for i in range(len(cases)):
        var got = String()
        try:
            _ = validate_vfs_path(cases[i])
        except e:
            got = String(e)
        assert_equal(got, msgs[i])


def main() raises:
    test_order_and_reducer()
    test_json_and_equals()
    test_validate_path()
    print("test_harness_vfs OK")
