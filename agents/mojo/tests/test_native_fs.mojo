"""Tests for qumo_agent.native.fs (DESIGN.md 2.1.4, 8.2).

Path vectors come from Python 3.12 (`os.path.normpath`, `pathlib.PurePosixPath(a) / b`,
`pathlib.Path.resolve()` on the same directory layout).
"""

from std.testing import assert_equal, assert_true, assert_false
from std.os import symlink as std_symlink, mkdir

from qumo_agent.native.fs import *


def _expect_raises(msg_prefix: String, out_text: String) raises:
    assert_true(out_text.startswith(msg_prefix), "got: " + out_text)


def test_normpath() raises:
    var cases: List[String] = [
        "", ".", ".", ".", "/", "/", "//", "//", "///", "/", "a/b/../c", "a/c", "/a/./b/", "/a/b",
        "../a", "../a", "a/../..", "..", "//a//b", "//a/b", "///a", "/a", "a//b/./c/..", "a/b",
        "/..", "/", "/../a", "/a", "./", ".", "a/b/c/../../..", ".", "../../x/..", "../..",
        "foo/bar/", "foo/bar",
    ]
    var i = 0
    while i < len(cases):
        assert_equal(normpath(cases[i]), cases[i + 1], "normpath " + cases[i])
        i += 2


def test_join_path() raises:
    var cases: List[String] = [
        "a", "b", "a/b", "a/", "b", "a/b", "a", "/b", "/b", "", "b", "b", ".", "x", "x",
        "a/./b", "c/", "a/b/c", "//a", "b", "//a/b", "///a", "b", "/a/b", "a", ".", "a",
        "a", "", "a", "/", "x", "/x", "a/b/..", "c", "a/b/../c",
    ]
    var i = 0
    while i < len(cases):
        assert_equal(join_path(cases[i], cases[i + 1]), cases[i + 2], "join " + cases[i] + " / " + cases[i + 1])
        i += 3


def test_files_and_dirs() raises:
    var t = make_temp_dir("qumo-fs-test-")
    assert_equal(file_mode(t), 0o700)
    var r = realpath_strict(t)
    assert_true(r.startswith("/"))
    # write / read / atomic / copy
    write_text(t + "/a.txt", "héllo\n")
    assert_equal(read_text_lossy(t + "/a.txt"), "héllo\n")
    assert_equal(len(read_bytes(t + "/a.txt")), 7)
    write_text_atomic(t + "/a.txt", "second")
    assert_equal(read_text_lossy(t + "/a.txt"), "second")
    write_text_atomic(t + "/new.txt", "x")
    assert_equal(read_text_lossy(t + "/new.txt"), "x")
    # No temp file left behind.
    var names = list_dir(t)
    assert_equal(len(names), 2)
    copy_file(t + "/a.txt", t + "/b.bin", 0o755)
    assert_equal(file_mode(t + "/b.bin"), 0o755)
    assert_equal(read_text_lossy(t + "/b.bin"), "second")
    chmod(t + "/b.bin", 0o600)
    assert_equal(file_mode(t + "/b.bin"), 0o600)
    # invalid UTF-8 is replaced
    var bad = List[UInt8]()
    bad.append(0x61)
    bad.append(0xFF)
    bad.append(0x62)
    with open(t + "/bad.txt", "w") as f:
        f.write_bytes(Span(bad))
    assert_equal(read_text_lossy(t + "/bad.txt"), "a�b")
    # errors carry Python's text
    try:
        _ = read_bytes(t + "/missing.txt")
        assert_true(False, "read of a missing file must raise")
    except e:
        assert_equal(String(e), "QA-E160: [Errno 2] No such file or directory: '" + t + "/missing.txt'")
    try:
        _ = read_bytes(t)
        assert_true(False, "read of a directory must raise")
    except e:
        assert_equal(String(e), "QA-E160: [Errno 21] Is a directory: '" + t + "'")
    try:
        _ = realpath_strict(t + "/nope/x")
        assert_true(False, "realpath of a missing path must raise")
    except e:
        _expect_raises("QA-E160: [Errno 2] No such file or directory: ", String(e))
    assert_equal(py_oserror(13, "it's"), "[Errno 13] Permission denied: \"it's\"")
    # make_dirs, exists, is_*
    make_dirs(t + "/d1/d2/d3")
    make_dirs(t + "/d1/d2/d3")
    assert_true(is_dir(t + "/d1/d2/d3") and exists(t + "/d1/d2"))
    assert_true(is_file(t + "/a.txt") and not is_dir(t + "/a.txt"))
    assert_false(exists(t + "/zzz"))
    # symlinks
    symlink("a.txt", t + "/link.txt")
    assert_true(is_symlink(t + "/link.txt") and is_file(t + "/link.txt"))
    try:
        symlink("a.txt", t + "/link.txt")
        assert_true(False, "symlink over an existing name must raise")
    except e:
        assert_equal(String(e), "QA-E160: [Errno 17] File exists: 'a.txt' -> '" + t + "/link.txt'")
    symlink("nowhere", t + "/dangling")
    assert_true(is_symlink(t + "/dangling") and not exists(t + "/dangling"))
    remove_tree(t)
    assert_false(exists(t))


def test_walk_and_remove_tree() raises:
    var t = make_temp_dir("qumo-walk-")
    var outside = make_temp_dir("qumo-walk-outside-")
    write_text(outside + "/keep.txt", "keep")
    make_dirs(t + "/a/b")
    write_text(t + "/a/b/f.txt", "1")
    write_text(t + "/top.txt", "2")
    std_symlink(outside, t + "/a/ext")
    var found = walk(t, False)
    var rel = List[String]()
    for p in found:
        assert_true(p.startswith(t + "/"))
        rel.append(String(p[byte = t.byte_length() + 1 :]))
    # every entry once, symlinked dir listed but not entered
    assert_equal(len(rel), 5)
    var pos_a = -1
    var pos_ab = -1
    var pos_abf = -1
    for i in range(len(rel)):
        if rel[i] == "a":
            pos_a = i
        if rel[i] == "a/b":
            pos_ab = i
        if rel[i] == "a/b/f.txt":
            pos_abf = i
        assert_true(rel[i] != "a/ext/keep.txt")
    assert_true(pos_a >= 0 and pos_a < pos_ab and pos_ab < pos_abf)
    var followed = walk(t, True)
    var saw_keep = False
    for p in followed:
        if p.endswith("/a/ext/keep.txt"):
            saw_keep = True
    assert_true(saw_keep)
    assert_equal(len(followed), 6)
    # A loop through a followed symlink terminates.
    std_symlink(t, t + "/a/b/up")
    var looped = walk(t, True)
    assert_true(len(looped) >= 7 and len(looped) < 100)
    # remove_tree removes the link, not the target
    remove_tree(t)
    assert_false(exists(t))
    assert_equal(read_text_lossy(outside + "/keep.txt"), "keep")
    try:
        remove_tree(t)
        assert_true(False, "remove_tree of a missing path must raise")
    except e:
        _expect_raises("QA-E160: [Errno 2]", String(e))
    remove_tree(outside)


def test_resolve_nonstrict() raises:
    var t = make_temp_dir("qumo-resolve-")
    var r = realpath_strict(t)
    mkdir(t + "/real")
    std_symlink("real", t + "/l")
    std_symlink("missing/x", t + "/d")
    std_symlink(t + "/l", t + "/abs")
    assert_equal(resolve_nonstrict(t + "/l/f"), r + "/real/f")
    assert_equal(resolve_nonstrict(t + "/l/../x"), r + "/x")
    assert_equal(resolve_nonstrict(t + "/d/y"), r + "/missing/x/y")
    assert_equal(resolve_nonstrict(t + "/nonexist/../real"), r + "/real")
    assert_equal(resolve_nonstrict(t + "/abs/g"), r + "/real/g")
    assert_equal(resolve_nonstrict(t + "/l"), r + "/real")
    assert_equal(resolve_nonstrict(t + "/.//l/./f/"), r + "/real/f")
    # a symlink loop does not hang
    std_symlink("loop2", t + "/loop1")
    std_symlink("loop1", t + "/loop2")
    assert_equal(resolve_nonstrict(t + "/loop1/z"), r + "/loop1/z")
    # relative paths resolve against the current directory
    assert_true(resolve_nonstrict("some/rel").endswith("/some/rel"))
    remove_tree(t)


def main() raises:
    test_normpath()
    test_join_path()
    test_files_and_dirs()
    test_walk_and_remove_tree()
    test_resolve_nonstrict()
    print("test_native_fs OK")
