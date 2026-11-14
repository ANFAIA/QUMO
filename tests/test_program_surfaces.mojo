"""Programs on every surface: JSON, tools, lint and drawing."""

from std.os import listdir
from std.testing import assert_equal, assert_false, assert_true

from qumo.ai.describe import program_from_json, program_to_json
from qumo.ai.diagnostics import error_catalog, lint_program
from qumo.core.json import JsonValue, parse_json, json_quote
from qumo.ai.registry import full_registry
from qumo.ai.tools import ToolRegistry
from qumo.compiler.program_compile import compile_program
from qumo.compiler.target import parse_target
from qumo.ir.classical import CExpr, float_type
from qumo.ir.circuit import Circuit
from qumo.ir.program import Program, ProgramBuilder, SK, stuck_loops
from qumo.ir.qasm import draw_program, load_any, load_any_with_warnings, parse_qasm, parse_qasm_program, to_qasm
from qumo.sim.program_runner import sample_program

comptime H = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; bit[3] c; "
comptime RUS = H + "uint[4] k = 0; output uint[4] tries; float a = 0.25; while (!c[0] && k < 7) { reset q[0]; h q[0]; rz(a) q[0]; c[0] = measure q[0]; k = k + 1; a = a * 2; } tries = k;"


def _programs() -> List[String]:
    """One of each thing the tables hold."""
    return [
        RUS,
        H + "output float r; float x = 1.5; int k = 0; while (k < 4) { x = x * 1.5 - 0.25; k += 1; if (k == 2) { continue; } rz(x) q[0]; } r = x;",
        H + "output int r = 0; for int i in [0:2:8] { if (i == 6) { break; } r += i; }",
        H + "output int r = 0; int n = 3; n = n + 1; for int i in {1, 5, 9} { r += i * n; }",
        H + "output uint[8] r = 0; h q[0]; c[0] = measure q[0]; uint[2] s = 0; s = 2; switch (s) { case 1, 2 { r = 7; } case 3 { r = 8; } default { r = 9; } }",
        H + "output bool r = false; float x = 0.5; x = x; r = c[0] == 0 && x > 0.25 || sin(x) < 0.1; h q[0];",
        H + "input float t; output float r; float x = 0.0; x = t * 2 + 1; ry(x) q[1]; r = mod(x, 2.0) + abs(x) + floor(x) ** 2.0;",
        H + "output int r; int[8] a = 127; a = a + 1; uint[8] b = 200; b = b << 1; r = int(a) + int(b) + int(popcount(b));",
        H + "output int r; angle[8] a = 0.1; a = a + a; bit[2] d = 1; d[1] = 1; r = int(uint[2](d)); rz(a) q[0];",
        H + "output uint[64] r = 18446744073709551615; r = r - 1; output float z = 1e300; z = z * 2.5;",
        H + "#pragma qumo depolarize1(0.01) q[0];\nbit b; int k = 0; while (k < 3) { k += 1; bit e; e = 1; if (e) { x q[1]; } c[1] = measure q[1]; }",
    ]


def _same_run(a: Program, b: Program, shots: Int = 40) raises:
    var ra = sample_program(a, shots, 11, _inputs(a))
    var rb = sample_program(b, shots, 11, _inputs(b))
    assert_equal(ra.sample.method, rb.sample.method)
    assert_equal(len(ra.sample.memory), len(rb.sample.memory))
    for s in range(len(ra.sample.memory)):
        assert_equal(ra.sample.memory[s], rb.sample.memory[s])
    assert_equal(len(ra.out_int), len(rb.out_int))
    for i in range(len(ra.out_int)):
        assert_equal(ra.out_int[i], rb.out_int[i])
    for i in range(len(ra.out_real)):
        assert_true(ra.out_real[i] == rb.out_real[i] or (ra.out_real[i] != ra.out_real[i] and rb.out_real[i] != rb.out_real[i]))
    for i in range(len(ra.iterations)):
        assert_equal(ra.iterations[i], rb.iterations[i])


def _inputs(p: Program) -> List[Float64]:
    return List[Float64](length=p.num_params(), fill=0.7)


def _through_json(p: Program) raises -> Program:
    return program_from_json(parse_json(program_to_json(p)))


def test_json_round_trip() raises:
    for src in _programs():
        var p = parse_qasm_program(src)
        var back = _through_json(p)
        assert_equal(to_qasm(back), to_qasm(p))
        assert_equal(program_to_json(back), program_to_json(p))
        _same_run(p, back)


def test_json_round_trip_of_the_control_flow_fixtures() raises:
    var seen = 0
    for name in listdir("tests/fixtures/control_flow"):
        if not String(name).endswith(".qasm"):
            continue
        var text: String
        with open("tests/fixtures/control_flow/" + name, "r") as f:
            text = f.read()
        var p = parse_qasm_program(text)
        var back = _through_json(p)
        assert_equal(to_qasm(back), to_qasm(p))
        seen += 1
    assert_true(seen >= 4)


def test_json_round_trip_of_a_compiled_program() raises:
    var p = parse_qasm_program(RUS)
    var r = compile_program(p, parse_target("line:4/rz,sx,x,cz"), 2, 7)
    var back = _through_json(r.program)
    assert_equal(to_qasm(back), to_qasm(r.program))
    _same_run(r.program, back)


def test_json_wraps_a_circuit() raises:
    var v = parse_json(program_to_json(parse_qasm_program(RUS)))
    assert_equal(v.get("format").as_string(), "qumo.program.v1")
    assert_equal(v.get("circuit").get("format").as_string(), "qumo.circuit.v1")
    var flat = Program.from_circuit(parse_qasm("qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;"))
    assert_equal(to_qasm(_through_json(flat)), to_qasm(flat))


def test_json_keeps_constants_json_cannot_write() raises:
    for v in [Float64("nan"), Float64("inf"), Float64("-inf"), -0.0, 1e-308, 0.1]:
        var b = ProgramBuilder(1, 1)
        var x = b.declare("x", float_type(), True)
        b.assign(x, CExpr.literal_float(v))
        var p = b.finish()
        var q = _through_json(p)
        assert_equal(String(q.nodes[p.expr_root[0]].f), String(v))
        assert_equal(program_to_json(q), program_to_json(p))


def test_json_keeps_integers_to_64_bits() raises:
    # the oracle's corpus found this: JSON numbers past 18 digits came back rounded
    var src = String(H + "output int r = 9223372036854775806; r = r - 1; output uint[64] u = 18446744073709551614; u = u - 1;")
    var p = parse_qasm_program(src)
    var q = _through_json(p)
    assert_equal(to_qasm(q), to_qasm(p))
    assert_equal(sample_program(q, 1, 1).int_output("r")[0], 9223372036854775805)
    assert_equal(parse_json("[9223372036854775807]").at(0).as_int(), 9223372036854775807)
    assert_equal(parse_json("[-9223372036854775808]").at(0).as_int(), -9223372036854775807 - 1)
    # past 64 bits a number stays a float
    assert_equal(parse_json("[9223372036854775808]").at(0).as_float(), 9223372036854775808.0)


def test_json_keeps_a_coefficient_that_overflowed() raises:
    # `null` used to come back as coeff = 1, which ran the gate; the source raises
    var src = String(H + "float x = 0.0; x = 1.0; rz(x * 1e308 * 10.0) q[0];")
    var p = parse_qasm_program(src)
    var q = _through_json(p)
    var a = String()
    var b = String()
    try:
        _ = sample_program(p, 1, 1)
    except e:
        a = String(e)
    try:
        _ = sample_program(q, 1, 1)
    except e:
        b = String(e)
    assert_true(a.startswith("QM-E482"), a)
    assert_true(b.startswith("QM-E482"), b)


def _float_expression(text: String) raises -> Int:
    """Index of an expression of float type."""
    var v = parse_json(text)
    var nodes = v.get("nodes")
    var exprs = v.get("expressions")
    for e in range(len(exprs)):
        var root = exprs.at(e).at(1).as_int()
        if nodes.at(root).get("type").as_string().startswith("float"):
            return e
    raise Error("no float expression")


def _reject(text: String, code: String) raises:
    var message = String()
    try:
        _ = program_from_json(parse_json(text))
    except e:
        message = String(e)
    assert_true(message.startswith(code), "expected " + code + " for " + text + ", got: " + message)


def test_json_refuses_what_a_shot_would_trust() raises:
    """Every index in the file is checked: a bad one must not reach a shot."""
    var good = program_to_json(parse_qasm_program(RUS))
    # the good one loads, so a refusal below comes from the one change
    _ = program_from_json(parse_json(good))
    _reject("[]", "QM-E587")
    _reject("{\"format\":\"qumo.circuit.v1\"}", "QM-E587")
    _reject(good.replace("qumo.program.v1", "qumo.program.v2"), "QM-E587")
    _reject(good.replace("\"num_store\":", "\"num_store\":99999999,\"unused\":"), "QM-E587")
    _reject(good.replace("\"nodes\":[", "\"nodes\":[{\"op\":\"goto\",\"type\":\"int\"},"), "QM-E587")
    _reject(good.replace("\"nodes\":[", "\"nodes\":[{\"op\":\"const\",\"type\":\"int[99]\"},"), "QM-E587")
    _reject(good.replace("\"expressions\":[[", "\"expressions\":[[0,999999],["), "QM-E587")
    _reject(good.replace("\"statements\":[", "\"statements\":[{\"kind\":\"jump\"},"), "QM-E587")
    _reject(good.replace("\"statements\":[", "\"statements\":[{\"kind\":\"ops\",\"a\":0,\"b\":999},"), "QM-E587")
    _reject(good.replace("\"statements\":[", "\"statements\":[{\"kind\":\"assign\",\"a\":99,\"b\":0},"), "QM-E587")
    _reject(good.replace("\"statements\":[", "\"statements\":[{\"kind\":\"while\",\"a\":99},"), "QM-E587")
    _reject(good.replace("\"statements\":[", "\"statements\":[{\"kind\":\"for\",\"a\":0,\"b\":9999},"), "QM-E587")
    _reject(good.replace("\"variables\":[", "\"variables\":[{\"name\":\"z\",\"type\":\"int\",\"store\":9999,\"slot\":-1},"), "QM-E587")
    _reject(good.replace("\"variables\":[", "\"variables\":[{\"name\":\"z\",\"type\":\"bit[9]\",\"store\":0,\"slot\":-1},"), "QM-E587")
    _reject(good.replace("\"variables\":[", "\"variables\":[{\"name\":\"z\",\"type\":\"int\",\"store\":0,\"slot\":99},"), "QM-E587")
    _reject(good.replace("\"registers\":[", "\"registers\":[{\"name\":\"z\",\"first\":0,\"width\":99},"), "QM-E587")
    _reject(good.replace("\"user_params\":[", "\"user_params\":[99"), "QM-E587")
    # blocks cover the arena in order: the compiler and its verifier rely on it
    var two = program_to_json(parse_qasm_program(H + "h q[0]; c[0] = measure q[0]; if (c[0]) { x q[0]; } h q[1];"))
    assert_true(two.find("{\"kind\":\"ops\",\"a\":0,\"b\":2") > 0, two)
    _reject(two.replace("{\"kind\":\"ops\",\"a\":0,\"b\":2", "{\"kind\":\"ops\",\"a\":0,\"b\":1,\"line\":1},{\"kind\":\"ops\",\"a\":1,\"b\":2"), "QM-E587")
    _reject(two.replace("{\"kind\":\"ops\",\"a\":0,\"b\":2", "{\"kind\":\"ops\",\"a\":1,\"b\":2"), "QM-E587")
    _reject(two.replace("\"statements\":[{\"kind\":\"ops\",\"a\":0,\"b\":2,\"line\":1},", "\"statements\":["), "QM-E587")
    # a statement must also hold what the interpreter reads from it
    var both = program_to_json(parse_qasm_program(H + "output int r = 0; float x = 0.5; x = x; for int i in [0:1] { r += i; }"))
    _ = program_from_json(parse_json(both))
    _reject(both.replace("\"statements\":[", "\"statements\":[{\"kind\":\"assign\",\"a\":0,\"b\":" + String(_float_expression(both)) + "},"), "QM-E587")
    _reject(both.replace("\"statements\":[", "\"statements\":[{\"kind\":\"if\",\"a\":" + String(_float_expression(both)) + "},{\"kind\":\"end\"},"), "QM-E587")
    _reject(both.replace("\"type\":\"int\",\"store\":", "\"type\":\"float\",\"store\":"), "QM-E587")
    _reject(good.replace("\"variables\":[", "\"variables\":[{\"name\":\"z\",\"type\":\"bit\",\"store\":0,\"slot\":1},"), "QM-E587")
    # the links are rebuilt, so a nesting that is wrong cannot be smuggled in
    _reject(good.replace("\"statements\":[", "\"statements\":[{\"kind\":\"end\"},"), "QM-E583")
    _reject(good.replace("\"statements\":[", "\"statements\":[{\"kind\":\"else\"},"), "QM-E583")
    _reject(good.replace("\"statements\":[", "\"statements\":[{\"kind\":\"break\"},"), "QM-E583")
    _reject(good.replace("\"statements\":[", "\"statements\":[{\"kind\":\"if\",\"a\":0},"), "QM-E583")
    var sw = program_to_json(parse_qasm_program(H + "output int r = 0; int s = 1; s = s; switch (s) { case 1 { r = 1; } default { r = 2; } }"))
    _reject(sw.replace("{\"kind\":\"case\"", "{\"kind\":\"ops\",\"a\":0,\"b\":0},{\"kind\":\"case\""), "QM-E583")


def test_json_checks_nodes_inside_expressions() raises:
    # an operand from another expression, or after its node, reads a stale result
    var good = program_to_json(parse_qasm_program(H + "output int r; int a = 1; a = a + 1; r = a;"))
    assert_true(good.find("\"op\":\"add\"") > 0)
    var head = good.find("\"op\":\"add\"")
    var x_at = good.find("\"x\":", head)
    var comma = good.find(",", x_at)
    var end_at = good.find("}", x_at)
    var cut = comma if comma >= 0 and comma < end_at else end_at
    var forged = String(good[byte=0:x_at]) + "\"x\":999999" + String(good[byte=cut:])
    _reject(forged, "QM-E587")


def test_json_keeps_short_circuits_whole() raises:
    # the fuzzer found these: a stray SC_END walked off the start of the nodes, a jump
    # read as a value made a cycle that overflowed the stack in to_qasm, and a set
    # loop kept unchecked bounds
    var good = program_to_json(parse_qasm_program(H + "output bool r; int a = 1; a = a; r = a > 0 || a < 9;"))
    var v = parse_json(good)
    var nodes = v.get("nodes")
    var end_at = -1
    var jump_at = -1
    for i in range(len(nodes)):
        if nodes.at(i).get("op").as_string() == "sc_end":
            end_at = i
        if nodes.at(i).get("op").as_string() == "sc_or":
            jump_at = i
    assert_true(end_at > jump_at and jump_at >= 0)
    _reject(good.replace("\"sc_or\"", "\"neg\""), "QM-E587")
    _reject(good.replace("\"sc_end\"", "\"neg\""), "QM-E587")
    # the right operand reads the jump
    var cycle = String()
    for i in range(len(nodes)):
        var text = nodes.at(i).to_json()
        if i == end_at:
            text = "{\"op\":\"sc_end\",\"type\":\"bool\",\"x\":" + String(jump_at) + "}"
        cycle += ("," if i > 0 else "") + text
    var start = good.find("\"nodes\":[")
    var stop = good.find("],\"expressions\"")
    _reject(String(good[byte=0:start]) + "\"nodes\":[" + cycle + String(good[byte=stop:]), "QM-E587")
    var loop = program_to_json(parse_qasm_program(H + "output int r = 0; int n = 2; n = n; for int i in {1, 2} { if (r > n) { break; } r += i; }"))
    assert_true(loop.find("\"for_ranges\":[-1,-1,-1,") > 0, loop)
    _reject(loop.replace("\"for_ranges\":[-1,-1,-1,", "\"for_ranges\":[4611686018427387904,-1,-1,"), "QM-E587")


def test_an_expression_too_deep_to_write_is_refused() raises:
    var src = String(H + "output int r; int x = 1; x = x; r = x")
    for _ in range(17000):
        src += " + x"
    var message = String()
    try:
        _ = to_qasm(parse_qasm_program(src + ";"))
    except e:
        message = String(e)
    assert_true(message.startswith("QM-E552"), message)


def test_unassigned_read_in_json_is_refused() raises:
    var good = program_to_json(parse_qasm_program(H + "output int r; int a = 1; a = a + 1; r = a;"))
    var v = parse_json(good)
    var stmts = v.get("statements")
    var kept = String()
    var dropped = False
    for i in range(len(stmts)):
        var s = stmts.at(i)
        if not dropped and s.get_string("kind", "") == "assign":
            dropped = True
            continue
        kept += ("," if kept.byte_length() > 0 else "") + s.to_json()
    var at = good.find("\"statements\":[")
    var message = String()
    try:
        _ = program_from_json(parse_json(String(good[byte=0:at]) + "\"statements\":[" + kept + "]}"))
    except e:
        message = String(e)
    assert_true(message.startswith("QM-E581"), message)


def _call(reg: ToolRegistry, name: String, args: String) raises -> JsonValue:
    var r = reg.call(name, parse_json(args))
    assert_false(r.is_error, name + ": " + r.structured)
    return parse_json(r.structured)


def _fail(reg: ToolRegistry, name: String, args: String) raises -> String:
    var r = reg.call(name, parse_json(args))
    assert_true(r.is_error, name + " did not fail")
    return parse_json(r.structured).get("error").get("code").as_string()


def _qasm_args(src: String, extra: String = "") -> String:
    return "{\"qasm\":" + json_quote(src) + extra + "}"


def test_run_tool() raises:
    var reg = full_registry()
    var v = _call(reg, "qumo_run_qasm", _qasm_args(RUS, ",\"shots\":500,\"seed\":3"))
    assert_true(v.get("iterations").get("mean").as_float() >= 1.0)
    assert_true(v.get("iterations").get("max").as_int() <= 7)
    var tries = v.get("outputs").get("tries")
    assert_true(tries.get("min").as_int() >= 1 and tries.get("max").as_int() <= 7)
    assert_true(len(tries.get("first_shots")) == 16)
    assert_false(v.get("auto_measured").as_bool())
    # the same bytes from the JSON form
    var j = program_to_json(parse_qasm_program(RUS))
    var w = _call(reg, "qumo_run_qasm", "{\"program\":" + j + ",\"shots\":500,\"seed\":3}")
    assert_equal(w.get("counts").to_json(), v.get("counts").to_json())
    var u = _call(reg, "qumo_run_qasm", "{\"circuit\":" + j + ",\"shots\":500,\"seed\":3}")
    assert_equal(u.get("counts").to_json(), v.get("counts").to_json())


def test_run_tool_takes_input_values() raises:
    var reg = full_registry()
    var src = String(H + "input float t; output float r; float a = 0.0; a = t; r = a * 2;")
    assert_equal(_call(reg, "qumo_run_qasm", _qasm_args(src, ",\"params\":[1.5],\"shots\":4")).get("outputs").get("r").get("max").as_float(), 3.0)
    assert_equal(_call(reg, "qumo_run_qasm", _qasm_args(src, ",\"params\":{\"t\":0.5},\"shots\":4")).get("outputs").get("r").get("max").as_float(), 1.0)
    assert_equal(_fail(reg, "qumo_run_qasm", _qasm_args(src)), "QM-E101")
    assert_equal(_fail(reg, "qumo_run_qasm", _qasm_args(src, ",\"params\":{\"z\":1.0,\"t\":1.0}")), "QM-E913")


def test_run_tool_max_iterations() raises:
    var reg = full_registry()
    var src = String(H + "int k = 0; while (k < 100) { k += 1; }")
    assert_equal(_fail(reg, "qumo_run_qasm", _qasm_args(src, ",\"max_iterations\":10,\"shots\":2")), "QM-E480")
    assert_equal(_call(reg, "qumo_run_qasm", _qasm_args(src, ",\"max_iterations\":100,\"shots\":2")).get("iterations").get("max").as_int(), 100)
    assert_equal(_fail(reg, "qumo_run_qasm", _qasm_args(src, ",\"max_iterations\":0")), "QM-E913")


def test_circuit_only_tools_refuse_a_program() raises:
    var reg = full_registry()
    assert_equal(_fail(reg, "qumo_statevector", _qasm_args(RUS)), "QM-E585")
    assert_equal(_fail(reg, "qumo_expectation", _qasm_args(RUS, ",\"observable\":\"Z0\"")), "QM-E585")
    assert_equal(_fail(reg, "qumo_density", _qasm_args(RUS)), "QM-E585")
    assert_equal(_fail(reg, "qumo_estimate_resources", _qasm_args(RUS)), "QM-E585")
    assert_equal(_fail(reg, "qumo_emulate", _qasm_args(RUS, ",\"target\":\"line:4\"")), "QM-E585")
    assert_equal(_fail(reg, "qumo_convert", _qasm_args(RUS, ",\"to\":\"qasm2\"")), "QM-E585")
    assert_equal(_fail(reg, "qumo_convert", _qasm_args(RUS, ",\"params\":[]")), "QM-E585")
    # a program that is one circuit is that circuit
    var lit = String(H + "rz(int[8](3)) q[0];")
    assert_equal(_call(reg, "qumo_statevector", _qasm_args(lit)).get("num_qubits").as_int(), 2)


def test_errors_keep_their_codes() raises:
    var reg = full_registry()
    assert_equal(_fail(reg, "qumo_run_qasm", _qasm_args("qubit[1] q; hh q[0];")), "QM-E510")
    assert_equal(_fail(reg, "qumo_statevector", _qasm_args("qubit[1] q; hh q[0];")), "QM-E510")
    assert_equal(_fail(reg, "qumo_run_qasm", "{}"), "QM-E910")
    assert_equal(_fail(reg, "qumo_run_qasm", "{\"program\":{\"format\":\"nope\"}}"), "QM-E587")


def test_convert_tool() raises:
    var reg = full_registry()
    var p = parse_qasm_program(RUS)
    assert_equal(_call(reg, "qumo_convert", _qasm_args(RUS)).get("qasm").as_string(), to_qasm(p))
    var j = _call(reg, "qumo_convert", _qasm_args(RUS, ",\"to\":\"json\""))
    assert_equal(j.get("format").as_string(), "qumo.program.v1")
    assert_equal(to_qasm(program_from_json(j.get("program"))), to_qasm(p))
    # flat stays a circuit
    var c = _call(reg, "qumo_convert", _qasm_args("qubit[1] q; h q[0];", ",\"to\":\"json\""))
    assert_equal(c.get("format").as_string(), "qumo.circuit.v1")


def test_validate_and_info_tools() raises:
    var reg = full_registry()
    var v = _call(reg, "qumo_validate_qasm", _qasm_args(RUS))
    assert_true(v.get("valid").as_bool())
    assert_equal(v.get("summary").get("num_loops").as_int(), 1)
    assert_true(v.get("summary").get("has_runtime_regions").as_bool())
    var bad = _call(reg, "qumo_validate_qasm", _qasm_args(H + "while (true) { h q[0]; "))
    assert_false(bad.get("valid").as_bool())
    var stuck = _call(reg, "qumo_validate_qasm", _qasm_args(H + "int k = 0; while (k < 3) { h q[0]; }"))
    assert_true(stuck.get("valid").as_bool())
    assert_equal(stuck.get("warnings").at(0).get("code").as_string(), "QM-W054")
    var info = _call(reg, "qumo_circuit_info", _qasm_args(RUS, ",\"include_qasm\":true,\"include_instructions\":true"))
    assert_equal(info.get("summary").get("num_blocks").as_int(), 1)
    assert_equal(info.get("qasm").as_string(), to_qasm(parse_qasm_program(RUS)))
    assert_equal(info.get("program").get("format").as_string(), "qumo.program.v1")


def test_transpile_tool() raises:
    var reg = full_registry()
    var v = _call(reg, "qumo_transpile", _qasm_args(RUS, ",\"target\":\"line:4/rz,sx,x,cz\",\"include_program\":true"))
    assert_true(v.get("blocks_checked").as_int() + v.get("blocks_unchecked").as_int() >= 1)
    var again = parse_qasm_program(v.get("qasm").as_string())
    assert_equal(again.num_qubits(), 4)
    assert_equal(v.get("program").get("format").as_string(), "qumo.program.v1")
    assert_equal(_fail(reg, "qumo_transpile", _qasm_args(RUS, ",\"target\":\"line:4\",\"ai\":true")), "QM-E630")
    var spec = _call(reg, "qumo_transpile", _qasm_args(RUS, ",\"target\":\"line:4\",\"pipeline\":\"cancel,route(sabre),basis\""))
    assert_true(spec.get("pipeline").as_string().find("route") >= 0)


def test_run_device_tool() raises:
    var reg = full_registry()
    var v = _call(reg, "qumo_run_device", _qasm_args(RUS, ",\"device\":\"gpu\",\"shots\":50"))
    assert_equal(v.get("device_used").as_string(), "cpu")
    assert_equal(v.get("warnings").at(0).get("code").as_string(), "QM-W020")
    var c = _call(reg, "qumo_run_device", _qasm_args(RUS, ",\"device\":\"cpu\",\"shots\":50,\"precision\":\"fp32\""))
    assert_equal(c.get("warnings").at(0).get("code").as_string(), "QM-W021")
    assert_equal(_fail(reg, "qumo_run_device", _qasm_args(RUS, ",\"device\":\"tpu\"")), "QM-E940")
    assert_equal(_fail(reg, "qumo_run_device", _qasm_args(H + "int k = 0; while (k < 9) { k += 1; }", ",\"max_iterations\":3")), "QM-E480")


def test_draw_tool_and_blocks() raises:
    var reg = full_registry()
    var src = String(H + "output int r = 0; float a = 0.5; a = a * 2; for int i in [0:1] { rz(a) q[0]; switch (i) { case 0 { x q[0]; } default { r += 1; break; } } if (c[0]) { h q[1]; } else { continue; } }")
    var v = _call(reg, "qumo_draw", _qasm_args(src, ",\"width\":40"))
    var text = v.get("drawing").as_string()
    assert_true(text.find("for int i in [0:1] {\n") >= 0, text)
    assert_true(text.find("\n  q0: --RZ(a)--") >= 0, text)
    assert_true(text.find("__qumo") < 0, text)
    assert_true(text.find("\n  switch (i) {\n    case 0:\n      q0: --X--") >= 0, text)
    assert_true(text.find("\n    default:\n      r = r + 1;\n      break;\n  }\n") >= 0, text)
    assert_true(text.find("\n  if (c[0]) {\n    q0: -----\n    q1: --H--\n  } else {\n    continue;\n  }\n}\n") >= 0, text)
    # a program that is one circuit draws as one
    var c = parse_qasm("qubit[2] q; h q[0]; cx q[0], q[1];")
    assert_equal(draw_program(Program.from_circuit(c), 80, "unicode"), c.draw(80, "unicode"))


def test_lint_w054() raises:
    def stuck(body: String) raises -> Int:
        return len(lint_program(parse_qasm_program(H + body)))

    # nothing in the body writes what the condition reads
    assert_equal(stuck("int k = 0; while (k < 3) { h q[0]; }"), 1)
    assert_equal(stuck("while (true) { h q[0]; }"), 1)
    assert_equal(stuck("while (false) { h q[0]; }"), 1)
    assert_equal(stuck("int k = 0; int j = 0; while (k < 3) { j += 1; }"), 1)
    assert_equal(stuck("while (c[0] == 0) { c[1] = measure q[0]; }"), 1)
    assert_equal(stuck("input float t; float x = 0.0; x = t; while (x > 0.5) { h q[0]; }"), 1)
    # an inner break leaves the inner loop only
    assert_equal(stuck("int k = 0; while (k < 3) { while (true) { break; } }"), 1)
    # any of these can end it
    assert_equal(stuck("int k = 0; while (k < 3) { k += 1; }"), 0)
    assert_equal(stuck("while (true) { h q[0]; c[0] = measure q[0]; if (c[0]) { break; } }"), 0)
    assert_equal(stuck("while (c[0] == 0) { h q[0]; c[0] = measure q[0]; }"), 0)
    assert_equal(stuck("bit b; while (b == 0) { b = 1; }"), 0)
    assert_equal(stuck("while (c[0:1] == 0) { h q[0]; c[1] = measure q[0]; }"), 0)
    assert_equal(stuck("int k = 0; while (k < 3) { if (k == 1) { for int i in [0:1] { k += 1; } } else { k = 1; } }"), 0)
    assert_equal(stuck("int i = 0; i = 0; while (i < 3) { for int j in [0:1] { i += j; } }"), 0)
    # one finding per loop, with its line
    var p = parse_qasm_program("OPENQASM 3.0;\nqubit[1] q;\nint k = 0;\nwhile (k < 3) {\n  while (true) { h q[0]; }\n  k += 1;\n}\n")
    var d = lint_program(p)
    assert_equal(len(d), 1)
    assert_equal(d[0].code, "QM-W054")
    assert_equal(d[0].line, 5)
    assert_equal(len(stuck_loops(p)), 1)


def test_warnings_come_from_the_parser_that_stood() raises:
    var w = List[String]()
    _ = load_any_with_warnings("qubit[1] q; delay[10ns] q[0]; h q[0];", w)
    assert_equal(len(w), 1)
    var v = List[String]()
    _ = load_any_with_warnings("qubit[1] q; delay[10ns] q[0]; int k = 0; k = 1;", v)
    assert_equal(len(v), 1)
    assert_true(v[0].startswith("QM-W051"), v[0])


def test_every_code_of_the_stack_is_in_the_catalogue() raises:
    var cat = error_catalog()
    var codes = List[String]()
    for c in [480, 481, 482, 483, 484, 580, 581, 582, 583, 584, 585, 586, 587, 630, 631, 632, 633]:
        codes.append("QM-E" + String(c))
    codes.append("QM-W054")
    for want in codes:
        var found = False
        for e in cat:
            if e.code == want:
                found = True
        assert_true(found, want + " is missing from error_catalog()")


def main() raises:
    test_json_round_trip()
    test_json_round_trip_of_the_control_flow_fixtures()
    test_json_round_trip_of_a_compiled_program()
    test_json_wraps_a_circuit()
    test_json_keeps_constants_json_cannot_write()
    test_json_keeps_integers_to_64_bits()
    test_json_keeps_a_coefficient_that_overflowed()
    test_json_refuses_what_a_shot_would_trust()
    test_json_checks_nodes_inside_expressions()
    test_json_keeps_short_circuits_whole()
    test_an_expression_too_deep_to_write_is_refused()
    test_unassigned_read_in_json_is_refused()
    test_run_tool()
    test_run_tool_takes_input_values()
    test_run_tool_max_iterations()
    test_circuit_only_tools_refuse_a_program()
    test_errors_keep_their_codes()
    test_convert_tool()
    test_validate_and_info_tools()
    test_transpile_tool()
    test_run_device_tool()
    test_draw_tool_and_blocks()
    test_lint_w054()
    test_warnings_come_from_the_parser_that_stood()
    test_every_code_of_the_stack_is_in_the_catalogue()
    print("test_program_surfaces OK")
