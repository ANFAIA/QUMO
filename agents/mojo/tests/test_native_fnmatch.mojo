"""Tests for qumo_agent.native.fnmatch (DESIGN.md 2.1.6, 8.2).

`fn_cases` come from Python 3.12 `fnmatch.fnmatchcase`, `wc_cases` from deepagents
0.7.32 `compile_grep_include_glob` (wcmatch 12.0), the rglob table from
`pathlib.Path.rglob` over the same file tree. libc fnmatch is exercised on ASCII
cases with this platform's FNM_* constants.
"""

from std.testing import assert_equal, assert_true, assert_false

from qumo_agent.native.fnmatch import *
from qumo_agent.native.libc import CStr, c_fnmatch, FNM_NOESCAPE, FNM_PATHNAME, FNM_PERIOD, FNM_NOMATCH, is_macos


def test_fnmatch_py() raises:
    var fn_cases: List[String] = [
        "a.py", "*.py", "1",
        "a/b.py", "*.py", "1",
        ".env", "*", "1",
        ".env", "*env", "1",
        "x", "[^a]", "0",
        "^", "[^a]", "1",
        "a", "[^a]", "1",
        "b", "[!a]", "1",
        "a", "[!a]", "0",
        "a\\b", "a\\b", "1",
        "a\\b", "a\\\\b", "0",
        "ab", "a\\b", "0",
        "é", "?", "1",
        "éé", "??", "1",
        "[", "[", "1",
        "[a", "[a", "1",
        "]", "[]]", "1",
        "a", "[]a]", "1",
        "-", "[a-]", "1",
        "b", "[a-c]", "1",
        "-", "[!-a]", "0",
        "x", "[z-a]", "0",
        "z", "[z-a]", "0",
        "x", "[!]", "0",
        "!", "[!]", "0",
        "a", "[!]", "0",
        "a-b", "a[!x]b", "1",
        "a/b", "a?b", "1",
        "a/b", "a*", "1",
        "AB", "ab", "0",
        "d", "[a-c-e]", "0",
        "-", "[a-c-e]", "1",
        "e", "[a-c-e]", "1",
        "&", "[&&]", "1",
        "a.b.py", "*.py", "1",
        "abc", "a*c*", "1",
        "abcbd", "a*b?", "1",
        "x\ny", "x?y", "1",
        "", "*", "1",
        "", "?", "0",
        "^", "[\\^]", "1",
        "\\", "[\\^]", "1",
        "b", "[a-b-c]", "1",
        "c", "[a--c]", "1",
        "-", "[--a]", "1",
        "[", "[[]", "1",
        "x", "[x-]", "1",
        "-", "[x-]", "1",
    ]
    var i = 0
    while i < len(fn_cases):
        var got = "1" if fnmatch_py(fn_cases[i], fn_cases[i + 1]) else "0"
        assert_equal(got, fn_cases[i + 2], "fnmatch(" + fn_cases[i] + ", " + fn_cases[i + 1] + ")")
        i += 3


def test_wc_glob() raises:
    var wc_cases: List[String] = [
        "*.py", "src/app/main.py", "1",
        "src/**/*.py", "src/app/main.py", "1",
        "src/**/*.py", "src/main.py", "1",
        "/*.py", "top.py", "1",
        "/*.py", "src/app/main.py", "0",
        "**/*.yml", ".github/workflows/ci.yml", "0",
        "*.yml", ".github/workflows/ci.yml", "1",
        ".github/**/*.yml", ".github/workflows/ci.yml", "1",
        "*.{yml,yaml}", "a/b.yaml", "1",
        "*.{yml,yaml}", "a/b.yml", "1",
        "*.{yml,yaml}", "a/b.json", "0",
        "*.{py", "a.{py", "1",
        "*.{py", "a.py", "0",
        "[a-", "[a-", "1",
        "[a-", "a", "0",
        "*", "a/.env", "0",
        "*", ".env", "0",
        ".*", ".env", "1",
        "[.]env", ".env", "0",
        "?env", ".env", "0",
        "a/*/c", "a/.b/c", "0",
        "a/.*/c", "a/.b/c", "1",
        "**", "a/b/c", "1",
        "**", "a/.b/c", "1",
        "a/**", "a/b/c", "1",
        "a/**", "a", "0",
        "a/**/c", "a/c", "1",
        "[!a]*", "b.py", "1",
        "[^a]*", "b.py", "1",
        "[^a]*", "a.py", "0",
        "\\*.py", "*.py", "1",
        "\\*.py", "a.py", "0",
        "a/b", "a/b", "1",
        "a//b", "a/b", "1",
        "*.PY", "a.py", "0",
        "{a,b}/{c,{d,e}}", "b/e", "1",
        "{a,b}/{c,{d,e}}", "b/f", "0",
        "x{1..3}", "x2", "1",
        "a/**/b/**/c", "a/x/b/y/z/c", "1",
        "[[:alpha:]]*", "abc", "1",
        "[[:digit:]]*", "abc", "0",
        "é?", "éa", "1",
        "?", "é", "1",
        "src/*", "src/a/b", "0",
        "src/**", "src/a/b", "1",
        "**/x", "x", "1",
        "**/x", "a/b/x", "1",
        "a/**/", "a/b/", "1",
        "*/", "a/", "1",
        "a{,b}", "a", "1",
        "a{b}", "a{b}", "1",
        "a{b}", "ab", "0",
        "[]a]", "]", "1",
        "[a-c]x", "bx", "1",
        "[!.]*", ".x", "0",
        "*~", "a~", "1",
        "a/./b", "a/b", "0",
        "./a", "a", "0",
        "**", ".a", "0",
        "**", "a/.b", "0",
        "**", ".a/b", "1",
        "a/**", "a/.b", "0",
        "a/**", "a/.b/c", "0",
        "a/**", "a/b/.c", "0",
        "**/c", "a/.b/c", "0",
        "a/**/c", "a/.b/c", "0",
        "**/", "a/", "1",
        "**/*", "a/.b/c", "0",
        "a/**/**/c", "a/b/c", "1",
        "a/**b/c", "a/xb/c", "1",
        "a/**b/c", "a/x/b/c", "0",
        "**.py", "a/b.py", "1",
        "x/**.py", "x/a/b.py", "0",
        "a/*", "a/", "0",
        "a/*", "a", "0",
        "*/b", "/b", "0",
        "a/b/", "a/b", "0",
        "a/b", "a/b/", "1",
        "[\\]]", "]", "1",
        "[a\\-z]", "-", "1",
        "[z-a]", "b", "0",
        "[a-]", "-", "1",
        "\\", "\\", "0",
        "a\\", "a\\", "0",
        "[!]", "!", "0",
        "[]", "]", "0",
        "[]]", "]", "1",
        "[*]", "*", "1",
        "[*]", "a", "0",
        "x{a,b", "x{a,b", "1",
        "x{a,b}}", "xa}", "1",
        "x{a,{b,c}", "x{a,b", "1",
        "{1..10..3}", "7", "1",
        "{01..3}", "02", "1",
        "{a..c}", "b", "1",
        "{c..a}", "b", "1",
        "{1..3}x", "1x", "1",
        "{a,b,}c", "c", "1",
        "[[:upper:]]", "A", "1",
        "[[:upper:]]", "a", "0",
        "[[:alnum:]_]*", "a_b", "1",
        "[[:foo:]]", "f", "0",
        "[a-c-e]", "d", "0",
        "[a-c-e]", "-", "1",
        "?", "/", "0",
        "*", "a/b", "1",
        "a?b", "a/b", "0",
        "[/]", "/", "0",
        "**/.git/**", ".git/x", "1",
        ".git/**", ".git/x", "1",
        "*.py", "dir/.hidden.py", "0",
        ".*.py", "dir/.hidden.py", "1",
        "{.*,*}.py", ".h.py", "1",
        "\\{a,b}", "{a,b}", "1",
        "\\{a,b}", "a", "0",
        "a{b,c}d{e,f}", "acdf", "1",
        "[\\x41]", "A", "0",
        "[\\x41]", "x", "1",
        "\\x41", "A", "0",
        "\\a", "a", "1",
        "é*", "étoile", "1",
    ]
    var i = 0
    while i < len(wc_cases):
        var got: String
        try:
            got = "1" if wc_glob_match(wc_cases[i + 1], wc_cases[i]) else "0"
        except:
            got = "E"
        assert_equal(got, wc_cases[i + 2], "glob " + wc_cases[i] + " on " + wc_cases[i + 1])
        i += 3
    # error texts
    try:
        _ = wc_glob_match("x", "../*.py")
        assert_true(False)
    except e:
        assert_equal(String(e), "Path traversal not allowed in glob pattern '../*.py'")
    try:
        _ = wc_glob_match("x", "a\\..\\b")
        assert_true(False)
    except e:
        assert_equal(String(e), "Path traversal not allowed in glob pattern 'a\\\\..\\\\b'")
    var big = String("{")
    for k in range(1001):
        if k > 0:
            big += ","
        big += String(k)
    big += "}"
    try:
        _ = wc_glob_match("x", big)
        assert_true(False)
    except e:
        assert_true(String(e).startswith("Invalid glob pattern '{0,1,2,"))
        assert_true(String(e).endswith("}': Pattern limit exceeded the limit of 1000"))
    assert_true(wc_glob_match("999", "{1..1000}"))
    try:
        _ = wc_glob_match("x", "{1..1001}")
        assert_true(False)
    except e:
        assert_true(String(e).endswith("Pattern limit exceeded the limit of 1000"))


def _expect(got: List[String], want: List[String]) raises:
    assert_equal(len(got), len(want), String(",").join(got))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_expand_braces() raises:
    _expect(expand_braces("a{b,c}d"), ["abd", "acd"])
    _expect(expand_braces("{a,b}{c,d}"), ["ac", "ad", "bc", "bd"])
    _expect(expand_braces("x{1..3}y"), ["x1y", "x2y", "x3y"])
    _expect(expand_braces("{01..10..3}"), ["01", "04", "07", "10"])
    _expect(expand_braces("{-2..2}"), ["-2", "-1", "0", "1", "2"])
    _expect(expand_braces("{-01..1}"), ["-01", "000", "001"])
    _expect(expand_braces("{a..e..2}"), ["a", "c", "e"])
    _expect(expand_braces("{z..x}"), ["z", "y", "x"])
    _expect(expand_braces("a{,b}"), ["a", "ab"])
    _expect(expand_braces("{a}"), ["{a}"])
    _expect(expand_braces("a{}b"), ["a{}b"])
    _expect(expand_braces("{a,{b,{c,d}}}e"), ["ae", "be", "ce", "de"])
    _expect(expand_braces("{1..2}{x,y}"), ["1x", "1y", "2x", "2y"])
    _expect(expand_braces("{a..}"), ["{a..}"])
    _expect(expand_braces("{aa..b}"), ["{aa..b}"])
    _expect(expand_braces("x{a,b"), ["x{a,b"])
    _expect(expand_braces("a\\{b,c}"), ["a\\{b,c}"])   # escapes are kept for the glob stage


def test_pathlib_rglob() raises:
    var files: List[String] = ["a.py", "src/b.py", "src/sub/c.py", ".hidden/d.py", "src/.e.py", "README.md", "src/x.txt", "docs/a/b/c.md", "tests/test_a.py", "tests/data/t.json", "é/ü.py"]
    var pats: List[String] = ["*.py", "src/*.py", "src/**/*.py", "**/*.md", "test_*.py", "tests/*", "*", "*.{py,md}", "[st]*/*.py", "a/b/*.md", "sub/*.py", "d*/**/*.md", "?.py", "**"]
    var want: List[String] = [
        ".hidden/d.py,a.py,src/.e.py,src/b.py,src/sub/c.py,tests/test_a.py,é/ü.py",
        "src/.e.py,src/b.py",
        "src/.e.py,src/b.py,src/sub/c.py",
        "README.md,docs/a/b/c.md",
        "tests/test_a.py",
        "tests/test_a.py",
        ".hidden/d.py,README.md,a.py,docs/a/b/c.md,src/.e.py,src/b.py,src/sub/c.py,src/x.txt,tests/data/t.json,tests/test_a.py,é/ü.py",
        "",
        "src/.e.py,src/b.py,src/sub/c.py,tests/test_a.py",
        "docs/a/b/c.md",
        "src/sub/c.py",
        "docs/a/b/c.md",
        ".hidden/d.py,a.py,src/b.py,src/sub/c.py,é/ü.py",
        "",
    ]
    for k in range(len(pats)):
        var hits = List[String]()
        for f in files:
            var parts = List[String]()
            for s in f.split("/"):
                parts.append(String(s))
            if pathlib_glob_match(parts, pats[k]):
                hits.append(f)
        sort(hits)
        assert_equal(String(",").join(hits), want[k], "rglob " + pats[k])
    assert_false(pathlib_glob_match(["a.py"], "/a.py"))


def _libc_fnmatch(name: String, pattern: String, flags: Int) -> Bool:
    var n = CStr(name)
    var p = CStr(pattern)
    var r = c_fnmatch(p.addr(), n.addr(), flags)
    n.keep()
    p.keep()
    return r == 0


def test_libc_constants() raises:
    if is_macos():
        assert_equal(FNM_NOESCAPE, 1)
        assert_equal(FNM_PATHNAME, 2)
    else:
        assert_equal(FNM_NOESCAPE, 2)
        assert_equal(FNM_PATHNAME, 1)
    assert_equal(FNM_PERIOD, 4)
    # FNM_NOESCAPE: a backslash is literal (Python fnmatch agrees on these).
    assert_true(_libc_fnmatch("a\\b", "a\\b", FNM_NOESCAPE))
    assert_false(_libc_fnmatch("ab", "a\\b", FNM_NOESCAPE))
    assert_true(_libc_fnmatch("ab", "a\\b", 0))
    assert_equal(fnmatch_py("a\\b", "a\\b"), _libc_fnmatch("a\\b", "a\\b", FNM_NOESCAPE))
    # FNM_PERIOD / FNM_PATHNAME
    assert_false(_libc_fnmatch(".env", "*", FNM_PERIOD))
    assert_true(_libc_fnmatch(".env", "*", 0))
    assert_false(_libc_fnmatch("a/b", "a*", FNM_PATHNAME))
    assert_true(_libc_fnmatch("a/b", "a*", 0))
    var ascii: List[String] = ["a.py", "*.py", "src/a.py", "*.py", "b", "[!a]", "abc", "a*c", "x", "?"]
    var i = 0
    while i < len(ascii):
        assert_equal(fnmatch_py(ascii[i], ascii[i + 1]), _libc_fnmatch(ascii[i], ascii[i + 1], FNM_NOESCAPE))
        i += 2


def main() raises:
    test_fnmatch_py()
    test_wc_glob()
    test_expand_braces()
    test_pathlib_rglob()
    test_libc_constants()
    print("test_native_fnmatch OK")
