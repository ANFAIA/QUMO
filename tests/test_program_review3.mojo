"""Regressions from the third independent review of the Program stack.

Each test names its finding by number, or `depth` for the parser nesting limit that
came with them. The fixes are on top of commit fb44177, and each of these failed
there, most of the JSON ones by ending the process; the exceptions are named in the
report. Finding 1 lives in tests/test_program_compile_review3.mojo.
"""

from std.math import inf, nan, pi
from std.testing import assert_equal, assert_true

from qumo.ai.describe import circuit_from_json, program_from_json, program_to_json
from qumo.ai.diagnostics import error_catalog
from qumo.ai.json import JsonWriter, parse_json
from qumo.ai.mcp import McpServer
from qumo.ai.tools import default_registry, write_program_outputs
from qumo.compiler.pbc import to_pbc
from qumo.exec.dispatch import sample_auto, sample_on
from qumo.exec.resources import estimate_resources
from qumo.ft.rz import exact_rz_sequence
from qumo.ir.circuit import Circuit
from qumo.ir.classical import float_mod, float_rem
from qumo.ir.qasm import draw_program, load_any, load_any_with_warnings, parse_qasm, parse_qasm_program, parse_qasm_with_warnings, to_qasm
from qumo.sim.density import density_evolve, sample_density
from qumo.sim.program_runner import sample_program
from qumo.sim.runner import evolve, sample

comptime H = "OPENQASM 3.0; qubit[4] q; bit[4] c; "
comptime LOOPS = "OPENQASM 3.0; input int n; qubit[2] q; bit[2] c; bit[2] d; int r = 0; for int j in {1, 3} { r += j; if (c[0]) break; } for int k in [0:n] { r += k; } c[0] = measure q[0]; if (d == 1) { x q[1]; }"
comptime INT_MAX = "9223372036854775807"


def _code(text: String) -> String:
    """What loading a document says: the code, or nothing when it loads."""
    try:
        var p = program_from_json(parse_json(text))
        _ = sample_program(p, 2, 1, [Float64(2.0)])
    except e:
        return String(String(e)[byte=:7])
    return ""


def _swap(text: String, old: String, new: String) raises -> String:
    assert_true(text.find(old) >= 0, "the document has no " + old)
    return text.replace(old, new)


def _between(text: String, a: String, b: String) -> String:
    var lo = text.find(a) + a.byte_length()
    return String(text[byte=lo : text.find(b, lo)])


def _loops() raises -> String:
    return program_to_json(parse_qasm_program(LOOPS))


def _refused(text: String, what: String) raises:
    assert_equal(_code(text), "QM-E587", what)


def test_2_a_huge_index_does_not_overflow_a_bounds_check() raises:
    var doc = _loops()
    assert_equal(_code(doc), "", "the unchanged document runs")
    # a FOR's offset into for_ranges, and what that table holds
    var table = _between(doc, "\"for_ranges\":", ",\"case_values\"")
    var ranges = String(table[byte=1 : table.byte_length() - 1]).split(",")
    _refused(_swap(doc, "\"kind\":\"for\",\"a\":1,\"b\":0", "\"kind\":\"for\",\"a\":1,\"b\":" + INT_MAX), "for b")
    for field in range(len(ranges)):
        for huge in [String(INT_MAX), String("-9223372036854775808")]:
            var edited = List[String]()
            for i in range(len(ranges)):
                edited.append(huge if i == field else String(ranges[i]))
            var code = _code(_swap(doc, table, "[" + ",".join(edited) + "]"))
            assert_true(code == "QM-E587", "for_ranges[" + String(field) + "] " + huge + ": " + code)
    # the bit fields of a register, a variable and an expression
    _refused(_swap(doc, "\"first\":2,\"width\":2", "\"first\":" + INT_MAX + ",\"width\":2"), "register first")
    _refused(_swap(doc, "\"first\":2,\"width\":2", "\"first\":2,\"width\":" + INT_MAX), "register width")
    _refused(_swap(doc, "\"type\":\"bit[2]\",\"x\":2,\"y\":2", "\"type\":\"bit[2]\",\"x\":" + INT_MAX + ",\"y\":2"), "bits x")
    _refused(_swap(doc, "{\"name\":\"r\",\"type\":\"int\",\"store\":0", "{\"name\":\"r\",\"type\":\"bit\",\"store\":" + INT_MAX), "bit store")


def test_2_a_set_of_a_huge_length() raises:
    var doc = _loops()
    var values = _between(doc, "\"case_values\":", ",\"statements\"")
    assert_equal(values, "[2,1,3]")
    _refused(_swap(doc, values, "[" + INT_MAX + ",1,3]"), "set count")


def test_2_every_document_error_is_e587() raises:
    _refused("{\"format\":\"qumo.program.v1\"}", "no circuit")
    _refused("{\"format\":7,\"circuit\":{\"num_qubits\":1}}", "a format that is not a string")
    _refused("[]", "not an object")
    var doc = _loops()
    _refused(_swap(doc, "\"output\":false", "\"output\":7"), "an output flag that is not a bool")
    # the circuit part reports through the program's own code
    _refused(_swap(doc, "\"num_qubits\":2", "\"num_qubits\":-1"), "negative qubits")
    _refused(_swap(doc, "\"num_qubits\":2", "\"num_qubits\":9223372036854775808"), "2^63 qubits")
    _refused(_swap(doc, "\"num_qubits\":2", "\"num_qubits\":1"), "a qubit past the end")


def test_8_a_measure_out_of_range() raises:
    var doc = _loops()
    _refused(_swap(doc, "\"clbit\":0", "\"clbit\":-1"), "clbit -1")
    _refused(_swap(doc, "\"clbit\":0", "\"clbit\":4"), "the first clbit past the end")
    _refused(_swap(doc, "\"clbit\":0", "\"clbit\":" + INT_MAX), "clbit INT_MAX")


def test_8_a_parameter_nothing_gives_a_value() raises:
    var src = "OPENQASM 3.0; input float t; qubit[1] q; rz(t) q[0]; while (false) { }"
    var doc = program_to_json(parse_qasm_program(src))
    assert_true(_code(doc) != "QM-E587", "the unchanged document loads")
    _refused(_swap(doc, "\"user_params\":[0]", "\"user_params\":[]"), "neither an input nor a slot")
    # a program file that names its parameter loads and reads back
    var p = program_from_json(parse_json(doc))
    assert_true(to_qasm(p).find("input") >= 0)


def test_9_a_small_document_cannot_ask_for_much() raises:
    var doc = _loops()
    assert_true(doc.find("\"num_store\":9") >= 0, doc)
    _refused(_swap(doc, "\"num_store\":9", "\"num_store\":10"), "one store past what it justifies")
    _refused(_swap(doc, "\"num_store\":9", "\"num_store\":1048576"), "the review's store")
    _refused(_swap(doc, "\"num_store\":9", "\"num_store\":" + INT_MAX), "INT_MAX stores")
    _refused(_swap(doc, "\"num_clbits\":4", "\"num_clbits\":5000"), "clbits nothing uses")
    _refused(_swap(doc, "\"num_clbits\":4", "\"num_clbits\":" + INT_MAX), "INT_MAX clbits")
    _refused(_swap(doc, "\"num_qubits\":2", "\"num_qubits\":" + INT_MAX), "INT_MAX qubits")
    _refused(_swap(doc, "\"num_qubits\":2", "\"num_qubits\":1000000"), "a million qubits, for a tableau")
    # a width that a real program can have still loads, and instructions justify more
    var wide = program_to_json(parse_qasm_program("OPENQASM 3.0; qubit[4000] q; bit[4000] c; int i = 0; while (i < 1) { i += 1; h q[3999]; } c[3999] = measure q[3999];"))
    assert_equal(_load_said(wide), "")
    var used = String("OPENQASM 3.0; qubit[5000] q; int i = 0; while (i < 1) { i += 1; h q; }")
    assert_equal(_load_said(program_to_json(parse_qasm_program(used))), "")


def test_9_the_flat_loader_caps_what_a_measure_grows() raises:
    var flat = "{\"num_qubits\":1,\"num_clbits\":1,\"instructions\":[{\"op\":\"measure\",\"qubits\":[0],\"clbit\":" + String(INT_MAX) + "}]}"
    var said = String("")
    try:
        _ = circuit_from_json(parse_json(flat))
    except e:
        said = String(e)
    assert_true(said.startswith("QM-E911"), said)
    # an ordinary measure past the end still grows the register, as it always did
    var c = circuit_from_json(parse_json("{\"num_qubits\":1,\"instructions\":[{\"op\":\"measure\",\"qubits\":[0],\"clbit\":3}]}"))
    assert_equal(c.num_clbits, 4)


def _depth_err(src: String, program: Bool) -> String:
    try:
        if program:
            _ = parse_qasm_program(src)
        else:
            _ = parse_qasm(src)
    except e:
        return String(e)
    return ""


def _load_err(src: String) -> String:
    try:
        _ = load_any(src)
    except e:
        return String(e)
    return ""


def _repeat(s: String, n: Int) -> String:
    var out = String()
    out.reserve(s.byte_length() * n)
    for _ in range(n):
        out += s
    return out^


def _too_deep(src: String, what: String) raises:
    """Refused the same way through each door, with the process still alive."""
    var e = _depth_err(src, False)
    assert_true(e.startswith("QM-E505"), what + " parse_qasm: " + String(e[byte=:min(e.byte_length(), 90)]))
    e = _depth_err(src, True)
    assert_true(e.startswith("QM-E505"), what + " parse_qasm_program: " + String(e[byte=:min(e.byte_length(), 90)]))
    e = _load_err(src)
    assert_true(e.startswith("QM-E505"), what + " load_any: " + String(e[byte=:min(e.byte_length(), 90)]))


comptime FLAT_HEAD = "OPENQASM 3.0; qubit[2] q; bit[2] c; "
comptime DEEP = 100000


def test_depth_100000_levels_are_refused_and_nothing_crashes() raises:
    var n = DEEP
    _too_deep(FLAT_HEAD + "rz(" + _repeat("(", n) + "1" + _repeat(")", n) + ") q[0];", "parens")
    _too_deep(FLAT_HEAD + "rz(" + _repeat("- ", n) + "1) q[0];", "minus")
    _too_deep(FLAT_HEAD + "rz(" + _repeat("+ ", n) + "1) q[0];", "plus")
    _too_deep(FLAT_HEAD + "rz(" + _repeat("1 ** ", n) + "1) q[0];", "powers")
    _too_deep(FLAT_HEAD + "rz(" + _repeat("sin(", n) + "1" + _repeat(")", n) + ") q[0];", "calls")
    _too_deep(FLAT_HEAD + "h q[" + _repeat("(", n) + "0" + _repeat(")", n) + "];", "index")
    _too_deep(FLAT_HEAD + _repeat("{ ", n) + "h q[0];" + _repeat(" }", n), "braces")
    _too_deep(FLAT_HEAD + _repeat("for int i in [0:0] { ", n) + "h q[0];" + _repeat(" }", n), "for blocks")
    _too_deep(FLAT_HEAD + _repeat("for int i in [0:0] ", n) + "h q[0];", "for statements")
    _too_deep(FLAT_HEAD + _repeat("box { ", n) + "h q[0];" + _repeat(" }", n), "boxes")


def test_depth_the_program_parser_on_what_only_it_reads() raises:
    var n = DEEP
    var head = String(FLAT_HEAD) + "int x = 0; x = x; bool b = false; b = b; "
    for src in [head + "x = " + _repeat("~", n) + "1;",
                head + "b = " + _repeat("!", n) + "true;",
                head + "x = " + _repeat("int(", n) + "1" + _repeat(")", n) + ";",
                head + "x = " + _repeat("int[8](", n) + "1" + _repeat(")", n) + ";",
                head + "x = " + _repeat("x ** ", n) + "x;",
                head + "bit y = " + _repeat("c[", n) + "0" + _repeat("]", n) + ";",
                head + _repeat("if (x > 0) { ", n) + "h q[0];" + _repeat(" }", n),
                head + _repeat("if (x > 0) ", n) + "h q[0];",
                head + _repeat("while (x > 0) { ", n) + "h q[0];" + _repeat(" }", n),
                head + _repeat("switch (x) { case 0 { ", n) + "h q[0];" + _repeat(" } }", n),
                head + _repeat("if (x > 0) { } else ", n) + "{ }"]:
        var e = _depth_err(src, True)
        assert_true(e.startswith("QM-E505"), String(src[byte=head.byte_length() : head.byte_length() + 24]) + ": " + String(e[byte=:min(e.byte_length(), 90)]))


def test_depth_256_levels_are_read_and_257_are_not() raises:
    var ok = FLAT_HEAD + "rz(" + _repeat("(", 256) + "1" + _repeat(")", 256) + ") q[0];"
    assert_equal(parse_qasm(ok).ops[0].p0.value(), 1.0)
    assert_true(_depth_err(FLAT_HEAD + "rz(" + _repeat("(", 257) + "1" + _repeat(")", 257) + ") q[0];", False).startswith("QM-E505"))
    var head = String(FLAT_HEAD) + "int x = 0; x = x; "
    var blocks = head + _repeat("if (x == 0) { ", 256) + "x = x + 1;" + _repeat(" }", 256) + " output int r = x;"
    assert_equal(_depth_err(blocks, True), "")
    assert_equal(sample_program(parse_qasm_program(blocks), 1, 1).int_output("r")[0], 1)
    assert_true(_depth_err(head + _repeat("if (x == 0) { ", 257) + _repeat(" }", 257), True).startswith("QM-E505"))
    assert_equal(_depth_err(head + _repeat("if (x == 0) ", 256) + "x = 1;", True), "")
    assert_true(_depth_err(head + _repeat("if (x == 0) ", 257) + "x = 1;", True).startswith("QM-E505"))
    assert_equal(_depth_err(FLAT_HEAD + _repeat("{ ", 256) + "h q[0];" + _repeat(" }", 256), False), "")
    assert_true(_depth_err(FLAT_HEAD + _repeat("{ ", 257) + "h q[0];" + _repeat(" }", 257), False).startswith("QM-E505"))
    # what 256 levels hold still goes through the file format
    assert_equal(_load_said(program_to_json(parse_qasm_program(blocks))), "")


def test_depth_a_failed_try_gives_its_levels_back() raises:
    # each argument backtracks through the parser that does not read it
    var src = String("OPENQASM 3.0; qubit[1] q; int x = 1; ")
    for _ in range(300):
        src += "rz(float(int[8](((" + "1" + ")))) + x) q[0]; "
    assert_equal(_depth_err(src + " while (false) { }", True), "")


def test_depth_json_recursion() raises:
    var n = DEEP
    for text in [_repeat("[", n), _repeat("{\"a\":", n), _repeat("[", n) + _repeat("]", n)]:
        var said = String("")
        try:
            _ = parse_json(text)
        except e:
            said = String(e)
        assert_true(said.startswith("QM-E900"), String(said[byte=:min(said.byte_length(), 90)]))
    var doc = _swap(_loops(), "\"circuit\":{", "\"circuit\":" + _repeat("[", n))
    var told = _load_said(doc)
    assert_true(told.startswith("QM-E900"), String(told[byte=:min(told.byte_length(), 90)]))
    # nesting in the statements, where the JSON itself stays flat
    var one = program_to_json(parse_qasm_program("OPENQASM 3.0; qubit[1] q; bool b = true; b = b; if (b) { }"))
    var head = "{\"kind\":\"if\"" + _between(one, "{\"kind\":\"if\"", "},{\"kind\":\"end\"") + "},"
    var close = "{\"kind\":\"end\",\"line\":1}"
    var at = head + close
    assert_equal(_load_said(one), "")
    assert_equal(_load_said(_swap(one, at, _repeat(head, 256) + _repeat(close + ",", 255) + close)), "")
    var said = _load_said(_swap(one, at, _repeat(head, 257) + _repeat(close + ",", 256) + close))
    assert_true(said.startswith("QM-E587") and said.find("nests past") >= 0, said)
    said = _load_said(_swap(one, at, _repeat(head, n) + _repeat(close + ",", n - 1) + close))
    assert_true(said.startswith("QM-E587") and said.find("nests past") >= 0, said)


def _load_said(text: String) -> String:
    try:
        _ = program_from_json(parse_json(text))
    except e:
        return String(e)
    return ""


def test_depth_has_a_code_in_the_catalogue() raises:
    var found = False
    for d in error_catalog():
        if d.code == "QM-E505":
            found = True
    assert_true(found)


def _said(ok: Bool, what: String) raises:
    assert_true(ok, what)


def _nan_refused(mut p_source: String, value: Float64, what: String) raises:
    var p = parse_qasm_program(p_source)
    var said = String("")
    try:
        _ = sample_program(p, 2, 1, [value])
    except e:
        said = String(e)
    assert_true(said.startswith("QM-E482"), what + ": " + said)


def test_3_a_value_that_is_not_finite_is_refused_on_entry() raises:
    var nan_ = nan[DType.float64]()
    var inf_ = inf[DType.float64]()
    var runtime = String("OPENQASM 3.0; input float t; qubit[1] q; bit[1] c; c[0] = measure q[0]; if (c[0]) { rz(t) q[0]; } c[0] = measure q[0];")
    var flat = String("OPENQASM 3.0; input float t; qubit[1] q; bit[1] c; rz(t) q[0]; c[0] = measure q[0];")
    for v in [nan_, inf_, -inf_]:
        _nan_refused(runtime, v, "a program that never reads it in shot 0")
        _nan_refused(flat, v, "a program that is one circuit")
    var c = parse_qasm(flat)
    for v in [nan_, inf_]:
        for k in range(5):
            var said = String("")
            try:
                if k == 0:
                    _ = sample(c, 2, 1, [v])
                elif k == 1:
                    _ = evolve(c, [v])
                elif k == 2:
                    _ = c.bind([v])
                elif k == 3:
                    _ = density_evolve(c, [v])
                else:
                    _ = sample_auto(c, 2, 1, [v])
            except e:
                said = String(e)
            assert_true(said.startswith("QM-E482"), String(k) + ": " + said)
    var note = String("")
    var said = String("")
    try:
        _ = sample_on(c, 2, 1, [nan_], "cpu", note)
    except e:
        said = String(e)
    assert_true(said.startswith("QM-E482"), said)


def _tool(name: String, args: String) raises -> String:
    var s = McpServer(default_registry())
    return s.handle("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"" + name + "\",\"arguments\":" + args + "}}")


def test_3_through_the_mcp_server() raises:
    var src = "\"OPENQASM 3.0; input float t; qubit[1] q; bit[1] c; rz(t) q[0]; c[0] = measure q[0];\""
    var r = _tool("qumo_run_qasm", "{\"qasm\":" + src + ",\"params\":[1e400]}")
    assert_true(r.find("QM-E482") >= 0, r)


comptime HUGE = "OPENQASM 3.0; output uint r; output uint s; output int n; output angle[64] a; bit[1] c; qubit[1] q; uint u = 18446744073709551615; u = u; r = u; s = 7; n = -1; a = 3 * pi / 2; c[0] = measure q[0];"


def test_4_a_uint_output_reads_through_its_type() raises:
    var r = sample_program(parse_qasm_program(HUGE), 2, 1)
    assert_equal(r.uint_output("r")[0], UInt64(18446744073709551615))
    assert_equal(r.uint_output("s")[1], UInt64(7))
    assert_equal(r.int_output("n")[0], -1)
    assert_true(r.uint_output("a")[0] > UInt64(9223372036854775807))
    var said = String("")
    try:
        _ = r.uint_output("n")
    except e:
        said = String(e)
    assert_true(said.startswith("QM-E913"), said)


def test_4_in_every_summary() raises:
    var r = sample_program(parse_qasm_program(HUGE), 2, 1)
    var w = JsonWriter()
    w.begin_object()
    write_program_outputs(w, r)
    w.end_object()
    var text = w.finish()
    assert_true(text.find("\"min\":18446744073709551615,\"max\":18446744073709551615") >= 0, text)
    assert_true(text.find("\"first_shots\":[18446744073709551615,18446744073709551615]") >= 0, text)
    assert_true(text.find("\"min\":-1") >= 0, text)
    # the reader of that text keeps every digit, and only an integer is one
    var v = parse_json(text).get("outputs").get("r")
    assert_equal(v.get("max").number_text(), "18446744073709551615")
    assert_equal(parse_json("18446744073709551616").number_text(), String(18446744073709551616.0))
    assert_equal(parse_json("9223372036854775808").number_text(), "9223372036854775808")
    assert_equal(parse_json("-9223372036854775808").as_int(), -9223372036854775807 - 1)
    assert_equal(parse_json("[9223372036854775808]").to_json(), "[9223372036854775808]")


def test_5_literal_positions_follow_the_classical_table() raises:
    var c = parse_qasm("OPENQASM 3.0; qubit[8] q; h q[7/2*2]; h q[9007199254740993 - 9007199254740992]; h q[-7 % 3 + 2];")
    assert_equal(c.ops[0].q0, 6)
    assert_equal(c.ops[1].q0, 1)
    assert_equal(c.ops[2].q0, 1)
    assert_equal(parse_qasm("OPENQASM 3.0; qubit[7/2*2] r;").num_qubits, 6)
    var d = parse_qasm("OPENQASM 3.0; qubit[8] q; for int i in [0:7/2*2] { h q[i]; }")
    assert_equal(len(d.ops), 7)
    d = parse_qasm("OPENQASM 3.0; qubit[8] q; for int i in {7/2*2} { h q[i]; }")
    assert_equal(d.ops[0].q0, 6)
    d = parse_qasm("OPENQASM 3.0; qubit[8] q; h q[0:7/2];")
    assert_equal(len(d.ops), 4)
    d = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[1] q; pow(7/2*2) @ rz(0.1) q[0];")
    assert_equal(len(d.ops), 6)  # rz: a gate with no period, so the exponent shows
    d = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[3] q; ctrl(3/2) @ x q[0], q[1];")
    assert_equal(d.ops[0].q0, 0)
    d = parse_qasm("OPENQASM 3.0; qubit[2] q; int[2*4] x = 300; rz(x) q[0];")
    assert_equal(d.ops[0].p0.value(), 44.0)
    d = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; switch (7/2*2) { case 6 { x q[0]; } default { y q[1]; } }")
    assert_equal(d.ops[0].q0, 0)
    # the table gives pow(1/2) no gate at all, and says what it dropped
    var w = List[String]()
    d = parse_qasm_with_warnings("OPENQASM 3.0; include \"stdgates.inc\"; qubit[1] q; pow(1/2) @ x q[0]; h q[0];", w)
    assert_equal(len(d.ops), 1)
    assert_equal(len(w), 1)
    assert_true(w[0].startswith("QM-W055"), w[0])
    # a program reads them the same
    assert_equal(parse_qasm_program("OPENQASM 3.0; qubit[8] q; h q[7/2*2]; while (false) { }").body.ops[0].q0, 6)
    assert_equal(load_any("OPENQASM 3.0; qubit[8] q; h q[7/2*2];").body.ops[0].q0, 6)


def test_5_what_is_not_an_integer_position_stays_as_it_was() raises:
    # a gate argument keeps real arithmetic, with QM-W055
    var w = List[String]()
    var c = parse_qasm_with_warnings("OPENQASM 3.0; qubit[1] q; rz(1/2) q[0];", w)
    assert_equal(c.ops[0].p0.value(), 0.5)
    assert_equal(len(w), 1)
    # refused, as it always was
    for src in ["h q[1.5];", "h q[2**-1];", "h q[1e20];", "h q[18446744073709551615 + 1];"]:
        var e = _depth_err("OPENQASM 3.0; qubit[8] q; " + String(src), False)
        assert_true(e.startswith("QM-E530") or e.startswith("QM-E580"), String(src) + ": " + e)
    var e = _depth_err("OPENQASM 3.0; input float t; qubit[8] q; h q[t];", False)
    assert_true(e.startswith("QM-E530"), e)
    e = _depth_err("OPENQASM 3.0; qubit[8] q; h q[y];", False)
    assert_true(e.startswith("QM-E520"), e)


def test_6_for_values_past_2_to_63_are_right_or_refused() raises:
    var head = String("OPENQASM 3.0; qubit[1] q; bit[1] c; output int r = 0; ")
    for src in ["for uint i in [0:9223372036854775808:18446744073709551615] { r += 1; c[0] = 0; }",
                "int n = 0; n = n; for int i in [n:18446744073709551615] { r += 1; }",
                "int n = 0; n = n; for int i in [18446744073709551615:n] { r += 1; }",
                "int n = 0; n = n; for int i in [n:9223372036854775808:5] { r += 1; }",
                "for int i in {18446744073709551615, 0} { r += 1; if (c[0]) break; }"]:
        var e = _err_run(head + String(src))
        assert_true(e.startswith("QM-E580"), String(src) + ": " + e)
    # the same with values only a run can know
    var ru = head + "uint s = 9223372036854775808; s = s; int n = 0; n = n; "
    var e = _err_run(ru + "for int i in [n:s:5] { r += 1; }")
    assert_true(e.startswith("QM-E482"), e)
    e = _err_run(ru + "for int i in [n:s] { r += 1; if (r > 2) break; }")
    assert_true(e.startswith("QM-E482"), e)
    e = _err_run(ru + "for int i in [s:n] { r += 1; }")
    assert_true(e.startswith("QM-E482"), e)
    # an int that starts or climbs past 2^63 stops the shot instead of going negative
    e = _err_run(head + "uint s = 9223372036854775807; s = s; for int i in [s:18446744073709551615] { r += 1; }")
    assert_true(e.startswith("QM-E482"), e)
    e = _err_run(head + "uint n = 18446744073709551615; n = n; for int i in [18446744073709551615:n] { r += 1; }")
    assert_true(e.startswith("QM-E482"), e)
    # correct as they stand: a uint counter crosses 2^63 and stops on time
    var ok = head + "output uint u = 0; uint s = 9223372036854775807; s = s; for uint i in [s:2:9223372036854775810] { r += 1; u = i; }"
    var got = sample_program(parse_qasm_program(ok), 1, 1)
    assert_equal(got.int_output("r")[0], 2)
    assert_equal(got.uint_output("u")[0], UInt64(9223372036854775809))


def _err_run(src: String) -> String:
    try:
        _ = sample_program(parse_qasm_program(src + " while (false) { }"), 1, 1)
    except e:
        return String(e)
    return ""


def test_7_a_huge_angle_is_no_multiple_of_a_quarter_turn() raises:
    var theta = 2251799813685509.0
    var seq = List[Int]()
    assert_true(not exact_rz_sequence(theta, seq))
    assert_true(exact_rz_sequence(pi / 2, seq))
    var c = Circuit(1)
    c.rz(theta, 0)
    var stats = to_pbc(c).counts()
    assert_equal(stats["rotations"], 1)
    assert_equal(stats["pi8"], 0)
    assert_equal(stats["arbitrary"], 1)
    var res = estimate_resources(c)
    assert_equal(res.rotations, 1)
    assert_true(res.t_count > 0)  # what a rotation costs, which a Clifford would not
    assert_true(not res.is_clifford)


def test_10_negative_shots() raises:
    var c = parse_qasm("OPENQASM 3.0; qubit[1] q; bit[1] c; c[0] = measure q[0];")
    var p = parse_qasm_program("OPENQASM 3.0; qubit[1] q; bit[1] c; c[0] = measure q[0]; while (false) { }")
    var note = String("")
    for k in range(5):
        var said = String("")
        try:
            if k == 0:
                _ = sample(c, -1, 1)
            elif k == 1:
                _ = sample_program(p, -1, 1)
            elif k == 2:
                _ = sample_auto(c, -1, 1)
            elif k == 3:
                _ = sample_on(c, -1, 1, List[Float64](), "cpu", note)
            else:
                _ = sample_density(c, -1, 1)
        except e:
            said = String(e)
        assert_true(said.startswith("QM-E913"), String(k) + ": " + said)


def test_11_float_remainder_is_exact() raises:
    assert_equal(float_rem(1e20, 7.0), 2.0)
    assert_equal(float_rem(-7.5, 2.0), -1.5)
    assert_equal(float_mod(-7.5, 2.0), 0.5)
    assert_equal(float_mod(1e20, 7.0), 2.0)
    assert_equal(float_mod(1e20, -7.0), -5.0)
    assert_equal(float_mod(6.0, 3.0), 0.0)
    # at run time, folded, and in the constant path
    assert_equal(sample_program(parse_qasm_program(H + "output float r; float a = 1e20; a = a; r = a % 7.0;"), 1, 1).real_output("r")[0], 2.0)
    assert_equal(sample_program(parse_qasm_program(H + "output float r; r = 1e20 % 7.0;"), 1, 1).real_output("r")[0], 2.0)
    assert_equal(sample_program(parse_qasm_program(H + "output float r; float a = 1e20; a = a; r = mod(a, 7.0);"), 1, 1).real_output("r")[0], 2.0)
    assert_equal(parse_qasm(H + "rz(1e20 % 7.0) q[0];").ops[0].p0.value(), 2.0)
    assert_equal(parse_qasm(H + "rz(mod(1e20, 7.0)) q[0];").ops[0].p0.value(), 2.0)


def test_12_draw_shows_a_constant_with_no_literal() raises:
    var p = parse_qasm_program("OPENQASM 3.0; qubit[1] q; float u = 1.0; while (u < 1e400) { u = 2e400; }")
    var text = draw_program(p, 80, "ascii")
    assert_true(text.find("inf") >= 0, text)
    assert_true(text.find("inf.0") < 0, text)
    var said = String("")
    try:
        _ = to_qasm(p)  # OpenQASM has no such literal, and that stays an error
    except e:
        said = String(e)
    assert_true(said.startswith("QM-E552"), said)


def test_13_a_remainder_that_differs_from_the_table_warns() raises:
    var w = List[String]()
    var c = parse_qasm_with_warnings("OPENQASM 3.0; qubit[1] q; rz(-7 % 2) q[0];", w)
    assert_equal(c.ops[0].p0.value(), 1.0)
    assert_equal(len(w), 1)
    assert_true(w[0].startswith("QM-W055"), w[0])
    # no warning where the two agree
    w.clear()
    _ = parse_qasm_with_warnings("OPENQASM 3.0; qubit[1] q; rz(7 % 2) q[0];", w)
    assert_equal(len(w), 0)
    w.clear()
    _ = parse_qasm_with_warnings("OPENQASM 3.0; qubit[1] q; rz(7.5 % 2.0) q[0];", w)
    assert_equal(len(w), 0)
    w.clear()
    var p = load_any_with_warnings("OPENQASM 3.0; qubit[1] q; rz(-7 % 2) q[0]; while (false) { }", w)
    assert_equal(p.body.ops[0].p0.value(), 1.0)
    assert_equal(len(w), 1)


def test_13_only_integer_operands_go_through_the_table() raises:
    # parse_term skips the second reading when an operand has a fraction, because that
    # is most gate arguments. Every integer division has to reach it all the same.
    for arg in ["3/4*pi", "pi*(3/4)", "(7/2)/pi", "2*pi*(1/2)", "-(3/4)", "pi/(7/2)", "sqrt(7/2)", "sin(3/4)", "2**(1/2)", "(1/3)**2", "1/2/pi", "pi+1/2"]:
        var w = List[String]()
        _ = parse_qasm_with_warnings("OPENQASM 3.0; qubit[1] q; rz(" + String(arg) + ") q[0];", w)
        assert_equal(len(w), 1, String(arg))
    for arg in ["pi/2", "2*pi/3", "-pi/2", "pi/2/2", "3*pi/4", "1.5/2", "3/pi", "3/2.5", "3.0/4", "6/3", "pi/2**3", "sqrt(2)/2"]:
        var w = List[String]()
        _ = parse_qasm_with_warnings("OPENQASM 3.0; qubit[1] q; rz(" + String(arg) + ") q[0];", w)
        assert_equal(len(w), 0, String(arg))


def test_14_a_shift_of_zero_is_zero() raises:
    assert_equal(parse_qasm("OPENQASM 3.0; qubit[1] q; float t = 0 << 1100; rz(t) q[0];").ops[0].p0.value(), 0.0)
    assert_equal(sample_program(parse_qasm_program(H + "output float t = 0 << 1100;"), 1, 1).real_output("t")[0], 0.0)
    assert_equal(sample_program(parse_qasm_program(H + "output float t = 0 >> 1100;"), 1, 1).real_output("t")[0], 0.0)


def main() raises:
    test_2_a_huge_index_does_not_overflow_a_bounds_check()
    test_2_a_set_of_a_huge_length()
    test_2_every_document_error_is_e587()
    test_8_a_measure_out_of_range()
    test_8_a_parameter_nothing_gives_a_value()
    test_9_a_small_document_cannot_ask_for_much()
    test_9_the_flat_loader_caps_what_a_measure_grows()
    test_depth_100000_levels_are_refused_and_nothing_crashes()
    test_depth_the_program_parser_on_what_only_it_reads()
    test_depth_256_levels_are_read_and_257_are_not()
    test_depth_a_failed_try_gives_its_levels_back()
    test_depth_json_recursion()
    test_depth_has_a_code_in_the_catalogue()
    test_3_a_value_that_is_not_finite_is_refused_on_entry()
    test_3_through_the_mcp_server()
    test_4_a_uint_output_reads_through_its_type()
    test_4_in_every_summary()
    test_5_literal_positions_follow_the_classical_table()
    test_5_what_is_not_an_integer_position_stays_as_it_was()
    test_6_for_values_past_2_to_63_are_right_or_refused()
    test_7_a_huge_angle_is_no_multiple_of_a_quarter_turn()
    test_10_negative_shots()
    test_11_float_remainder_is_exact()
    test_12_draw_shows_a_constant_with_no_literal()
    test_13_a_remainder_that_differs_from_the_table_warns()
    test_13_only_integer_operands_go_through_the_table()
    test_14_a_shift_of_zero_is_zero()
    print("test_program_review3 OK")
