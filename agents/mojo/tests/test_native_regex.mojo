"""Tests for qumo_agent.native.regex (DESIGN.md 2.1.5, 8.2).

The match table was computed with Python 3.12 `re.search` over the same lines; each
pattern goes through `py_regex_to_ere` and libc regcomp/regexec with this platform's
REG_* constants.
"""

from std.testing import assert_equal, assert_true, assert_false

from qumo_agent.native.regex import *
from qumo_agent.native.libc import REG_NOSUB, REG_NEWLINE, is_macos


def test_python_table() raises:
    var lines: List[String] = ["foo bar", "def run(x):", "call 555-1234 now", "import os", "import sys2", "An ERROR here", "color", "colour", "abc", "a.c", "the word is", "swordfish", "y", "xxxy", "MY_VAR=1", "abc", "trailing  ", "me@x.org", "a.b", "axb", "$HOME/x", "aa", "a", "b", "aab", "{x}", "x{", "3.14", "my-file.py", "]", "-", "^", "\\", "ababc", "\tx", "é", "QM-E123", "[ok]", "()", "aab", ".", "AB", "/path", "ab", "", "bx", "dx", "zzz"]
    var pats: List[String] = [
        "foo",
        "def \\w+\\(",
        "\\d{3}-\\d{4}",
        "^import (os|sys)$",
        "(?i)error",
        "colou?r",
        "a.c",
        "\\bword\\b",
        "x*?y",
        "[A-Z_]+=",
        "[^a-z]",
        "\\s+$",
        "\\S+@\\S+",
        "a\\.b",
        "\\$HOME",
        "a{2,}",
        "a{,2}b",
        "{x}",
        "x{",
        "[\\d.]+",
        "[\\w-]+\\.py",
        "[]a]",
        "[\\]]",
        "[a\\-z]",
        "[\\^x]",
        "[\\\\]",
        "(?:ab)+c",
        "\\tx",
        "é+",
        "QM-E\\d+",
        "\\[ok\\]",
        "a|b|c",
        "\\(\\)",
        "a+?b",
        "[.]",
        "\\x41B",
        "\\/path",
        "(a)(b)?",
        "^$",
        "\\.",
        "[a-c]x",
    ]
    var want: List[String] = [
        "100000000000000000000000000000000000000000000000",
        "010000000000000000000000000000000000000000000000",
        "001000000000000000000000000000000000000000000000",
        "000100000000000000000000000000000000000000000000",
        "000001000000000000000000000000000000000000000000",
        "000000110000000000000000000000000000000000000000",
        "000000001100000100000000000000000100000000000000",
        "000000000010000000000000000000000000000000000000",
        "000010000000110000000000000010000000000000000000",
        "000000000000001000000000000000000000000000000000",
        "111111000110001011101000011111111011111011100000",
        "000000000000000010000000000000000000000000000000",
        "000000000000000001000000000000000000000000000000",
        "000000000000000000100000000000000000000000000000",
        "000000000000000000001000000000000000000000000000",
        "000000000000000000000100100000000000000100000000",
        "100000001000000100110001100000000100000100010100",
        "000000000000000000000000010000000000000000000000",
        "000000000000000000000000001000000000000000000000",
        "001010000100001001100000000110000000100010000000",
        "000000000000000000000000000010000000000000000000",
        "101000001100000110110110100001000100010100110000",
        "000000000000000000000000000001000000010000000000",
        "101000001100000110110110100010100100100100110001",
        "010000000000010001011000011000010010000000000110",
        "000000000000000000000000000000001000000000000000",
        "000000001000000100000000000000000100000000000000",
        "000000000000000000000000000000000010000000000000",
        "000000000000000000000000000000000001000000000000",
        "000000000000000000000000000000000000100000000000",
        "000000000000000000000000000000000000010000000000",
        "101000111100000110110111100000000100000100110100",
        "000000000000000000000000000000000000001000000000",
        "000000001000000100000000100000000100000100010000",
        "000000000100000001100000000110000000000010000000",
        "000000000000000000000000000000000000000001000000",
        "000000000000000000000000000000000000000000100000",
        "101000001100000110110110100000000100000100110000",
        "000000000000000000000000000000000000000000001000",
        "000000000100000001100000000110000000000010000000",
        "000000000000000000010000000000000000000000000100",
    ]
    assert_equal(len(pats), len(want))
    for k in range(len(pats)):
        var t = py_regex_to_ere(pats[k])
        var rx = PosixRegex(t[0], t[1])
        var got = String()
        for line in lines:
            got += "1" if rx.search(line) else "0"
        assert_equal(got, want[k], "pattern " + pats[k] + " -> " + t[0])


def _ere(p: String) raises -> String:
    return py_regex_to_ere(p)[0]


def test_translation() raises:
    assert_equal(_ere("\\d\\D\\w\\W\\s\\S"), "[0-9][^0-9][[:alnum:]_][^[:alnum:]_][[:space:]][^[:space:]]")
    assert_equal(_ere("[\\d\\w\\s]"), "[0-9[:alnum:]_[:space:]]")
    assert_equal(_ere("a\\.b\\*c\\\\d\\^e\\$"), "a[.]b[*]c\\\\d\\^e[$]")
    assert_equal(_ere("(?:ab)+"), "(ab)+")
    assert_equal(_ere("a*?b+?c??d{1,2}?"), "a*b+c?d{1,2}")
    assert_equal(_ere("a{,3}"), "a{0,3}")
    assert_equal(_ere("x{"), "x[{]")
    assert_equal(_ere("\\bfoo\\b"), "(^|[^[:alnum:]_])foo([^[:alnum:]_]|$)")
    assert_equal(_ere("\\t\\n"), "\t\n")
    assert_equal(_ere("[]a]"), "[]a]")
    assert_equal(_ere("[a\\]\\-^]"), "[]a^-]")
    assert_equal(_ere("[\\^]"), "\\^")
    assert_equal(_ere("[\\[x]"), "[x[]")
    var t = py_regex_to_ere("(?i)Hello")
    assert_equal(t[0], "Hello")
    assert_true(t[1])
    assert_false(py_regex_to_ere("Hello")[1])


def _refused(p: String, want: String) raises:
    try:
        _ = py_regex_to_ere(p)
        assert_true(False, "expected a refusal for " + p)
    except e:
        assert_equal(String(e), want, "pattern " + p)


def test_refusals() raises:
    _refused("a(?=b)", "unsupported in POSIX ERE: (?=")
    _refused("a(?!b)", "unsupported in POSIX ERE: (?!")
    _refused("(?<=a)b", "unsupported in POSIX ERE: (?<=")
    _refused("(?<!a)b", "unsupported in POSIX ERE: (?<!")
    _refused("(?P<n>a)", "unsupported in POSIX ERE: (?P<")
    _refused("(a)\\1", "unsupported in POSIX ERE: \\1")
    _refused("\\Aabc", "unsupported in POSIX ERE: \\A")
    _refused("abc\\Z", "unsupported in POSIX ERE: \\Z")
    _refused("a\\Bb", "unsupported in POSIX ERE: \\B")
    _refused("a\\bb", "unsupported in POSIX ERE: \\b")
    _refused("a++", "unsupported in POSIX ERE: possessive quantifier")
    _refused("(?>a)", "unsupported in POSIX ERE: (?>")
    _refused("x(?s)y", "unsupported in POSIX ERE: (?s")
    _refused("(?m)^a", "unsupported in POSIX ERE: (?m")
    _refused("[\\D]", "unsupported in POSIX ERE: \\D inside [...]")
    _refused("[é]", "unsupported in POSIX ERE: non-ASCII character inside [...]")
    _refused("\\p{L}", "unsupported in POSIX ERE: \\p")
    _refused("[abc", "unterminated character set at position 0")
    _refused("*a", "nothing to repeat at position 0")
    _refused("a**", "multiple repeat at position 2")


def test_platform_constants() raises:
    if is_macos():
        assert_equal(REG_NOSUB, 4)
        assert_equal(REG_NEWLINE, 8)
    else:
        assert_equal(REG_NOSUB, 8)
        assert_equal(REG_NEWLINE, 4)
    # REG_NOSUB with the right value: regexec without a match array works.
    var rx = PosixRegex("b+", False)
    assert_true(rx.search("abbbc"))
    assert_false(rx.search("ac"))
    var ic = PosixRegex("hello", True)
    assert_true(ic.search("say HeLLo"))
    try:
        _ = PosixRegex("a(b", False)
        assert_true(False, "regcomp must fail on an unbalanced group")
    except e:
        assert_true(String(e).byte_length() > 0)


def main() raises:
    test_python_table()
    test_translation()
    test_refusals()
    test_platform_constants()
    print("test_native_regex OK")
