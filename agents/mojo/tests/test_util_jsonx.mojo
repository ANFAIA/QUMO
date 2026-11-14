"""Tests for util/jsonx.mojo: lenient parsing, the B2 argument rules, JSON edits."""

from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, parse_json
from qumo_agent.util.jsonx import *
from qumo_agent.util.pyfmt import py_json_dumps


def _args(raw_json: String) raises -> ParsedArgs:
    """parse_tool_arguments of the `arguments` value written as JSON text."""
    var doc = parse_json('{"arguments": ' + raw_json + '}')
    return parse_tool_arguments(doc.get("arguments"))


def _ok_empty(raw_json: String) raises:
    var p = _args(raw_json)
    assert_true(p.ok, raw_json)
    assert_true(p.args.is_object(), raw_json)
    assert_equal(len(p.args), 0, raw_json)


def _invalid(raw_json: String, error: String) raises:
    var p = _args(raw_json)
    assert_false(p.ok, raw_json)
    assert_equal(p.error, error, raw_json)


def test_lenient() raises:
    # Raw newline and tab inside a string (json.loads(strict=False)).
    var text = String('{"code": "line1\nline2\tx", "n": 1}')
    var v = parse_json_lenient(text)
    assert_equal(v.get("code").as_string(), "line1\nline2\tx")
    assert_equal(escape_raw_controls(text), '{"code": "line1\\u000aline2\\u0009x", "n": 1}')
    # Escaped quotes and backslashes do not confuse the string tracking.
    var t2 = String('{"a": "q\\"\n\\\\", "b": "\x01"}')
    var v2 = parse_json_lenient(t2)
    assert_equal(v2.get("a").as_string(), "q\"\n\\")
    assert_equal(v2.get("b").as_string(), "\x01")
    # Whitespace between tokens is untouched; text without controls is returned as is.
    assert_equal(escape_raw_controls('{\n "a": 1\n}'), '{\n "a": 1\n}')
    var raised = False
    try:
        _ = parse_json_lenient('{"a": }')
    except:
        raised = True
    assert_true(raised)


def test_b2_table() raises:
    # Absent, null, "": {}.
    var p0 = parse_tool_arguments(parse_json("{}").get("arguments"))
    assert_true(p0.ok)
    assert_equal(len(p0.args), 0)
    _ok_empty("null")
    _ok_empty('""')
    # Strings holding falsy non-objects: {}.
    _ok_empty('"null"')
    _ok_empty('"[]"')
    _ok_empty('"0"')
    _ok_empty('"false"')
    _ok_empty('"\\"\\""')
    _ok_empty('"{}"')
    _ok_empty('"  {}  "')
    # Objects.
    var p = _args('"{\\"path\\": \\"/é\\", \\"n\\": 2}"')
    assert_true(p.ok)
    assert_equal(p.args.get("path").as_string(), "/é")
    assert_equal(py_json_dumps(p.args, False, -1), '{"path": "/é", "n": 2}')
    # An object given directly (not as a string) is accepted.
    var d = _args('{"x": [1, 2]}')
    assert_true(d.ok)
    assert_equal(len(d.args.get("x")), 2)
    # A raw control character inside the argument string is accepted.
    var lc = _args('"{\\"code\\": \\"a\\nb\\"}"')
    assert_true(lc.ok)
    # Truthy non-objects (X13).
    _invalid('"[1]"', "arguments must be a JSON object")
    _invalid('"5"', "arguments must be a JSON object")
    _invalid('"\\"x\\""', "arguments must be a JSON object")
    _invalid('"true"', "arguments must be a JSON object")
    _invalid('[1]', "arguments must be a JSON object")
    _invalid('5', "arguments must be a JSON object")
    # Direct falsy non-objects.
    _ok_empty('[]')
    _ok_empty('0')
    _ok_empty('false')
    # Parse failures carry the parser's text.
    var bad = _args('"{\\"a\\": "')
    assert_false(bad.ok)
    assert_true(bad.error.startswith("QM-E900: invalid JSON"), bad.error)
    assert_true(bad.args.is_object())


def test_edits() raises:
    var e = empty_object()
    assert_true(e.is_object())
    assert_equal(len(e), 0)
    var o = parse_json('{"a": 1, "b": {"c": [true, null]}, "d": "x"}')
    var w = json_with(o, "b", '"new"')
    assert_equal(py_json_dumps(w, False, -1), '{"a": 1, "b": "new", "d": "x"}')
    var w2 = json_with(o, "z", '[1.5, 2]')
    assert_equal(py_json_dumps(w2, False, -1), '{"a": 1, "b": {"c": [true, null]}, "d": "x", "z": [1.5, 2]}')
    var w3 = json_with(e, "k", '{}')
    assert_equal(py_json_dumps(w3, False, -1), '{"k": {}}')
    var wo = json_without(o, "b")
    assert_equal(py_json_dumps(wo, False, -1), '{"a": 1, "d": "x"}')
    assert_equal(py_json_dumps(json_without(o, "missing"), False, -1), py_json_dumps(o, False, -1))
    # Floats keep their float form; duplicate keys collapse like a Python dict.
    var f = parse_json('{"x": 1.0, "y": 2, "x": 3.5}')
    assert_equal(py_json_dumps(json_with(f, "y", "null"), False, -1), '{"x": 3.5, "y": null}')
    var raised = False
    try:
        _ = json_with(o, "a", "{bad")
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        _ = json_without(parse_json("[1]"), "a")
    except:
        raised = True
    assert_true(raised)


def main() raises:
    test_lenient()
    test_b2_table()
    test_edits()
    print("test_util_jsonx OK")
