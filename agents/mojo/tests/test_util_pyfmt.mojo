"""Tests for util/pyfmt.mojo against tests/golden/pyfmt.json (written by gen_pyfmt.py)."""

from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, parse_json
from qumo_agent.util.pyfmt import *

comptime GOLDEN = "agents/mojo/tests/golden/pyfmt.json"


def _strings(v: JsonValue) raises -> List[String]:
    var out = List[String]()
    for i in range(len(v)):
        out.append(v.at(i).as_string())
    return out^


def _check_list(got: List[String], want: JsonValue, what: String) raises:
    assert_equal(len(got), len(want), what)
    for i in range(len(got)):
        assert_equal(got[i], want.at(i).as_string(), what)


def test_repr(g: JsonValue) raises:
    var rs = g.get("repr")
    for i in range(len(rs)):
        var s = rs.at(i).at(0).as_string()
        assert_equal(py_str_repr(s), rs.at(i).at(1).as_string(), "repr #" + String(i))
    var lr = g.get("list_repr")
    for i in range(len(lr)):
        assert_equal(py_list_repr(_strings(lr.at(i).at(0))), lr.at(i).at(1).as_string())


def test_json(g: JsonValue) raises:
    var docs = g.get("json_docs")
    for i in range(len(docs)):
        var d = docs.at(i)
        var what = "doc " + d.get("doc").as_string()
        var v = parse_json(d.get("doc").as_string())
        assert_equal(py_json_dumps(v, False, -1), d.get("dumps_false").as_string(), what)
        assert_equal(py_json_dumps(v, True, -1), d.get("dumps_true").as_string(), what)
        assert_equal(py_json_dumps(v, False, 0), d.get("indent0").as_string(), what)
        assert_equal(py_json_dumps(v, False, 2), d.get("indent2").as_string(), what)
        assert_equal(py_json_dumps(v, True, 1), d.get("indent1_ascii").as_string(), what)
        assert_equal(py_repr_json(v), d.get("repr").as_string(), what)
    var ds = g.get("dumps_string")
    for i in range(len(ds)):
        var s = ds.at(i).at(0).as_string()
        assert_equal(py_json_dumps_string(s, False), ds.at(i).at(1).as_string())
        assert_equal(py_json_dumps_string(s, True), ds.at(i).at(2).as_string())
    # Duplicate keys: first position, last value (Python dict semantics).
    assert_equal(py_json_dumps(parse_json('{"a": 1, "b": 2, "a": 3}'), False, -1), '{"a": 3, "b": 2}')


def test_floats(g: JsonValue) raises:
    var fr = g.get("float_repr")
    for i in range(len(fr)):
        var x = fr.at(i).at(0).as_float()
        assert_equal(py_float_repr(x), fr.at(i).at(1).as_string())
    var nan = Float64(0.0) / Float64(0.0)
    var inf = Float64(1.0) / Float64(0.0)
    var sp = g.get("float_special")
    assert_equal(py_float_repr(nan), sp.at(0).at(1).as_string())
    assert_equal(py_float_repr(inf), sp.at(1).at(1).as_string())
    assert_equal(py_float_repr(-inf), sp.at(2).at(1).as_string())


def test_str_methods(g: JsonValue) raises:
    var sl = g.get("splitlines")
    for i in range(len(sl)):
        _check_list(py_splitlines(sl.at(i).at(0).as_string()), sl.at(i).at(1), "splitlines #" + String(i))
    var sw = g.get("split_ws")
    for i in range(len(sw)):
        _check_list(py_split_ws(sw.at(i).at(0).as_string()), sw.at(i).at(1), "split #" + String(i))
    var st = g.get("strip")
    for i in range(len(st)):
        var s = st.at(i).at(0).as_string()
        assert_equal(py_strip(s), st.at(i).at(1).as_string(), "strip #" + String(i))
        assert_equal(py_rstrip(s), st.at(i).at(2).as_string(), "rstrip #" + String(i))
    var sp = g.get("isspace")
    for i in range(len(sp)):
        assert_equal(py_is_space(sp.at(i).at(0).as_int()), sp.at(i).at(1).as_bool(), "isspace #" + String(i))
    var un = g.get("universal_newlines")
    for i in range(len(un)):
        assert_equal(universal_newlines(un.at(i).at(0).as_string()), un.at(i).at(1).as_string())
    var xm = g.get("xml")
    for i in range(len(xm)):
        var s = xm.at(i).at(0).as_string()
        assert_equal(xml_escape(s), xm.at(i).at(1).as_string())
        assert_equal(xml_quoteattr(s), xm.at(i).at(2).as_string())


def test_code_points(g: JsonValue) raises:
    var cs = g.get("cp_slice")
    for i in range(len(cs)):
        var c = cs.at(i)
        var s = c.at(0).as_string()
        assert_equal(cp_slice(s, c.at(1).as_int(), c.at(2).as_int()), c.at(3).as_string(), "cp_slice #" + String(i))
        assert_equal(cp_len(s), c.at(4).as_int())
    var ht = g.get("cp_head_tail")
    for i in range(len(ht)):
        var c = ht.at(i)
        var s = c.at(0).as_string()
        var n = c.at(1).as_int()
        assert_equal(cp_head(s, n), c.at(2).as_string(), "cp_head #" + String(i))
        assert_equal(cp_tail(s, n), c.at(3).as_string(), "cp_tail #" + String(i))
    # 80,000 three-byte characters: code points, not bytes.
    var big = String()
    for _ in range(1000):
        big += "€€€€€€€€€€"
    assert_equal(cp_len(big), 10000)
    assert_equal(cp_len(cp_head(big, 9999)), 9999)
    assert_equal(cp_tail(big, 2), "€€")


def test_c1(g: JsonValue) raises:
    """The vectors of DESIGN.md Appendix C.1 that belong to this module."""
    var c = g.get("c1")
    var rs = c.get("repr")
    for i in range(len(rs)):
        assert_equal(py_str_repr(rs.at(i).at(0).as_string()), rs.at(i).at(1).as_string())
    var d1 = parse_json('{"path": "/é", "n": [1, 2.5, null, true], "s": "a\\n \\u007f\\u0001\\""}')
    assert_equal(py_json_dumps(d1, False, -1), c.get("json_dumps_ensure_ascii_false").as_string())
    var d2 = parse_json('{"d": "é—\\ud83d\\ude00"}')
    assert_equal(py_json_dumps(d2, True, -1), c.get("json_dumps_ensure_ascii_true").as_string())
    var d3 = parse_json('{"a": [], "b": {}, "c": [1, {"x": "é"}]}')
    assert_equal(py_json_dumps(d3, False, 1), c.get("json_dumps_indent1").as_string())
    var fr = c.get("float_repr")
    for i in range(len(fr)):
        assert_equal(py_float_repr(fr.at(i).at(0).as_float()), fr.at(i).at(1).as_string())
    _check_list(py_splitlines("a\r\nb\rc\x0bd\x0ce\x1cf\u0085g h\n"), c.get("splitlines").at(0), "c1 splitlines")
    _check_list(py_split_ws(" a b　c\t"), c.get("split_ws"), "c1 split")
    var tc = parse_json('[{"name": "ls", "args": {"path": "/é", "n": 1.5, "ok": true, "z": null}, "id": "call_1", "type": "tool_call"}]')
    assert_equal(py_repr_json(tc), c.get("repr_tool_calls").as_string())


def test_printable_table() raises:
    # Spot checks of the generated table (Python 3.12).
    assert_equal(py_str_repr("͏"), "'͏'")        # combining grapheme joiner: printable
    assert_equal(py_str_repr("️"), "'️'")        # variation selector: printable
    assert_equal(py_str_repr("᠎"), "'\\u180e'")
    assert_equal(py_str_repr("\U000e0001"), "'\\U000e0001'")
    assert_equal(py_str_repr("\U0010ffff"), "'\\U0010ffff'")
    assert_equal(py_str_repr("\U0001f600"), "'\U0001f600'")
    assert_equal(py_str_repr("͸"), "'\\u0378'")       # unassigned


def main() raises:
    var g = parse_json(open(GOLDEN, "r").read())
    test_repr(g)
    test_json(g)
    test_floats(g)
    test_str_methods(g)
    test_code_points(g)
    test_c1(g)
    test_printable_table()
    print("test_util_pyfmt OK")
