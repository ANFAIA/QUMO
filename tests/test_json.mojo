"""Tests for qumo.ai.json: writer escaping/precision and arena parser."""

from qumo.ai.json import (
    JsonWriter, JsonKind, JsonValue, parse_json, pretty_json, json_quote, json_number,
)
from std.testing import assert_equal, assert_true, assert_false, assert_almost_equal
from std.math import pi


def expect_json_error(text: String, frag: String) raises:
    try:
        _ = parse_json(text)
    except e:
        var s = String(e)
        assert_true(s.startswith("QM-E900"), "code: " + s)
        assert_true(s.find(frag) >= 0, "expected '" + frag + "' in: " + s)
        return
    raise Error("expected JSON error for: " + text)


def main() raises:
    # ---------------------------------------------------------------- writer
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("name", "bell \"pair\"\n\ttab\\")
    w.kv_int("shots", 1024)
    w.kv_float("pi", pi)
    w.kv_float("tiny", 1e-300)
    w.kv_float("nan", Float64(0.0) / Float64(0.0))
    w.kv_bool("ok", True)
    w.kv_null("none")
    w.key("list")
    w.begin_array()
    w.integer(1)
    w.number(0.5)
    w.string("x")
    w.begin_object()
    w.end_object()
    w.begin_array()
    w.end_array()
    w.end_array()
    w.kv_raw("raw", "{\"a\":[1,2]}")
    w.kv_string("ctl", "\x01")
    w.end_object()
    var text = w.finish()
    print(text)
    assert_equal(
        text,
        "{\"name\":\"bell \\\"pair\\\"\\n\\ttab\\\\\",\"shots\":1024,\"pi\":3.141592653589793,"
        + "\"tiny\":1e-300,\"nan\":null,\"ok\":true,\"none\":null,\"list\":[1,0.5,\"x\",{},[]],"
        + "\"raw\":{\"a\":[1,2]},\"ctl\":\"\\u0001\"}",
    )
    assert_equal(json_quote("π"), "\"π\"")
    assert_equal(json_number(0.1), "0.1")

    # ---------------------------------------------------------------- parser
    var v = parse_json(text)
    assert_true(v.is_object())
    assert_equal(len(v), 10)
    assert_equal(v.get("name").as_string(), "bell \"pair\"\n\ttab\\")
    assert_equal(v.get("shots").as_int(), 1024)
    assert_equal(v.get("pi").as_float(), pi)  # exact round-trip
    assert_equal(v.get("tiny").as_float(), 1e-300)
    assert_true(v.get("nan").is_null())
    assert_true(v.get("ok").as_bool())
    assert_true(v.has("none"))
    assert_false(v.has("missing"))
    assert_equal(v.get("missing").kind(), JsonKind.UNDEFINED)
    assert_true(v.get("missing").get("deeper").at(3).kind() == JsonKind.UNDEFINED)
    var lst = v.get("list")
    assert_equal(len(lst), 5)
    assert_equal(lst.at(0).as_int(), 1)
    assert_equal(lst.at(1).as_float(), 0.5)
    assert_equal(lst.at(2).as_string(), "x")
    assert_true(lst.at(3).is_object() and len(lst.at(3)) == 0)
    assert_true(lst.at(4).is_array() and len(lst.at(4)) == 0)
    assert_equal(v.get("raw").get("a").at(1).as_int(), 2)
    assert_equal(v.get("ctl").as_string(), "\x01")
    assert_equal(v.key_at(1), "shots")
    # round trip
    assert_equal(v.to_json(), text)
    assert_equal(parse_json(v.to_pretty_json()).to_json(), text)

    # escapes and unicode
    var u = parse_json("{\"a\":\"\\u00e9\\u4e2d\\ud83d\\ude00\\/\",\"b\":-0.0,\"c\":-12,\"d\":2.5E+3,\"e\":12345678901234567890}")
    assert_equal(u.get("a").as_string(), "é中😀/")
    assert_equal(u.get("c").as_int(), -12)
    assert_equal(u.get("d").as_float(), 2500.0)
    assert_equal(u.get("d").as_int(), 2500)
    assert_almost_equal(u.get("e").as_float(), 1.2345678901234567e19)
    # whitespace / scalars at top level
    assert_equal(parse_json("  42 ").as_int(), 42)
    assert_true(parse_json("null").is_null())
    assert_equal(parse_json("[true,false]").at(1).as_bool(), False)
    # defaults
    var o = parse_json("{\"shots\": 10, \"seed\": null}")
    assert_equal(o.get_int("shots", 1), 10)
    assert_equal(o.get_int("seed", 7), 7)
    assert_equal(o.get_string("qasm", "dflt"), "dflt")
    try:
        _ = o.get_string("shots", "")
        raise Error("expected type error")
    except e:
        assert_true(String(e).startswith("QM-E911: field 'shots'"), String(e))
    # duplicate keys: last wins
    assert_equal(parse_json("{\"a\":1,\"a\":2}").get("a").as_int(), 2)

    # errors with positions
    expect_json_error("{\"a\":1,}", "expected a quoted object key")
    expect_json_error("{\"a\" 1}", "line 1, col 6")
    expect_json_error("[1,2", "expected ',' or ']'")
    expect_json_error("{\"a\":tru}", "invalid literal")
    expect_json_error("\"abc", "unterminated string")
    expect_json_error("{} x", "trailing characters")
    expect_json_error("", "unexpected end of input")
    expect_json_error("{\n  \"a\": 01\n}", "line 2")
    expect_json_error("[1.]", "digit expected")
    expect_json_error("{'a':1}", "quoted object key")

    # pretty printer
    var p = pretty_json("{\"a\":[1,2],\"b\":{\"c\":null}}")
    assert_equal(p, "{\n  \"a\": [1,2],\n  \"b\": {\n    \"c\": null\n  }\n}")

    # speed sanity: 20k-element array
    var big = JsonWriter()
    big.begin_array()
    for i in range(20000):
        big.begin_object()
        big.kv_int("i", i)
        big.kv_float("x", Float64(i) * 0.25)
        big.end_object()
    big.end_array()
    var bt = big.finish()
    var bv = parse_json(bt)
    assert_equal(len(bv), 20000)
    assert_equal(bv.at(19999).get("x").as_float(), 4999.75)
    print("test_json OK")
