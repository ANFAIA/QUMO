"""Regressions from the third independent review of the Program stack.

Each test names its finding by number, or `depth` for the parser nesting limit that
came with them. The fixes are on top of commit fb44177, and each of these failed
there, most of the JSON ones by ending the process; the exceptions are named in the
report. Finding 1 lives in tests/test_program_compile_review3.mojo.
"""

from std.testing import assert_equal, assert_true

from qumo.ai.describe import circuit_from_json, program_from_json, program_to_json
from qumo.ai.json import parse_json
from qumo.ir.qasm import parse_qasm_program, to_qasm
from qumo.sim.program_runner import sample_program

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
    return String(text[byte=text.find(a) + a.byte_length() : text.find(b)])


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
    _refused(_swap(doc, "\"num_clbits\":4", "\"num_clbits\":2097152"), "clbits nothing uses")
    _refused(_swap(doc, "\"num_clbits\":4", "\"num_clbits\":" + INT_MAX), "INT_MAX clbits")
    _refused(_swap(doc, "\"num_qubits\":2", "\"num_qubits\":" + INT_MAX), "INT_MAX qubits")
    # a width that a real program can have still loads
    var wide = program_to_json(parse_qasm_program("OPENQASM 3.0; qubit[5000] q; bit[5000] c; int i = 0; while (i < 1) { i += 1; h q[4999]; } c[4999] = measure q[4999];"))
    assert_true(_code(wide) == "" or _code(wide) != "QM-E587", _code(wide))


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


def main() raises:
    test_2_a_huge_index_does_not_overflow_a_bounds_check()
    test_2_a_set_of_a_huge_length()
    test_2_every_document_error_is_e587()
    test_8_a_measure_out_of_range()
    test_8_a_parameter_nothing_gives_a_value()
    test_9_a_small_document_cannot_ask_for_much()
    test_9_the_flat_loader_caps_what_a_measure_grows()
    print("test_program_review3 OK")
