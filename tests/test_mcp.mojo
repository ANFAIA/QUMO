"""Tests for the MCP server (qumo.ai.mcp), tools, describe and diagnostics."""

from std.testing import assert_equal, assert_true, assert_false, assert_almost_equal
from std.math import sqrt
from std.os import getenv, listdir
from std.os.path import isdir, exists
from std.sys.defines import is_defined

from qumo.ai.json import JsonWriter, JsonValue, parse_json, json_quote
from qumo.ai.mcp import McpServer, handle_request
from qumo.ai.tools import default_registry
from qumo.ai.describe import (
    describe_gates_json, describe_errors_json, describe_errors_markdown, describe_error_json,
    describe_capabilities_json, circuit_to_json, circuit_from_json, circuit_summary_json,
)
from qumo.ai.diagnostics import (
    diagnostic_from_error, parse_error_code, lint, error_catalog, error_ranges, range_index_of,
    default_hint,
)
from qumo.ir.qasm import parse_qasm, to_qasm
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve

comptime BELL = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\nc = measure q;\n"


def call_line(id: Int, tool: String, args_json: String) -> String:
    return "{\"jsonrpc\":\"2.0\",\"id\":" + String(id) + ",\"method\":\"tools/call\",\"params\":{\"name\":" + json_quote(tool) + ",\"arguments\":" + args_json + "}}"


def rpc(mut s: McpServer, line: String) raises -> JsonValue:
    var out = s.handle(line)
    assert_true(out.byte_length() > 0, "no reply for " + line)
    assert_true(out.find("\n") < 0, "reply must be a single line")
    var v = parse_json(out)
    assert_equal(v.get("jsonrpc").as_string(), "2.0")
    return v^


def tool(mut s: McpServer, id: Int, name: String, args_json: String) raises -> JsonValue:
    var v = rpc(s, call_line(id, name, args_json))
    assert_equal(v.get("id").as_int(), id)
    var r = v.get("result")
    assert_true(r.is_object(), v.to_json())
    assert_equal(r.get("content").at(0).get("type").as_string(), "text")
    assert_true(r.get("structuredContent").is_object())
    return r^


def test_protocol() raises:
    var s = McpServer()
    var init = rpc(s, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}")
    var res = init.get("result")
    assert_equal(res.get("protocolVersion").as_string(), "2025-06-18")
    assert_equal(res.get("serverInfo").get("name").as_string(), "qumo")
    assert_true(res.get("capabilities").get("tools").is_object())
    assert_true(res.get("instructions").as_string().byte_length() > 50)
    # older client version is echoed, unknown version -> ours
    var old = rpc(s, "{\"jsonrpc\":\"2.0\",\"id\":\"a\",\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-03-26\"}}")
    assert_equal(old.get("result").get("protocolVersion").as_string(), "2025-03-26")
    assert_equal(old.get("id").as_string(), "a")
    var fut = rpc(s, "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2099-01-01\"}}")
    assert_equal(fut.get("result").get("protocolVersion").as_string(), "2025-06-18")
    # notifications: no reply
    assert_equal(s.handle("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}"), "")
    assert_true(s.initialized)
    assert_equal(s.handle("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/cancelled\",\"params\":{\"requestId\":3}}"), "")
    assert_equal(s.handle("   "), "")
    # ping
    var p = rpc(s, "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}")
    assert_true(p.get("result").is_object() and len(p.get("result")) == 0)
    # tools/list
    var tl = rpc(s, "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/list\"}")
    var tools = tl.get("result").get("tools")
    assert_true(len(tools) >= 7)
    var names = List[String]()
    for i in range(len(tools)):
        var t = tools.at(i)
        names.append(t.get("name").as_string())
        assert_true(t.get("description").as_string().byte_length() > 20)
        var sch = t.get("inputSchema")
        assert_equal(sch.get("type").as_string(), "object")
        assert_true(sch.get("properties").is_object())
        assert_true(t.get("annotations").get("readOnlyHint").as_bool())
    var want: List[String] = [
        "qumo_run_qasm", "qumo_statevector", "qumo_expectation", "qumo_validate_qasm", "qumo_describe",
        "qumo_circuit_info", "qumo_convert", "qumo_transpile", "qumo_estimate_resources", "qumo_vqe",
        "qumo_qaoa_maxcut", "qumo_qec_memory", "qumo_qec_dem", "qumo_gpu_info", "qumo_run_device",
    ]
    for w in want:
        var found = False
        for n in names:
            if n == w:
                found = True
        assert_true(found, "missing tool " + w)
    # schema details agents rely on
    for i in range(len(tools)):
        var t = tools.at(i)
        var props = t.get("inputSchema").get("properties")
        if t.get("name").as_string() == "qumo_qec_memory":
            var en = props.get("decoder").get("enum").to_json()
            for d in ["mwpm", "sparse_blossom", "mwpm_dense", "uf", "lookup"]:
                assert_true(en.find("\"" + String(d) + "\"") >= 0, "decoder enum lacks " + String(d) + ": " + en)
            assert_equal(props.get("decoder").get("default").as_string(), "mwpm")
        if t.get("name").as_string() == "qumo_run_device":
            assert_equal(props.get("device").get("enum").to_json(), "[\"auto\",\"cpu\",\"gpu\"]")
            assert_equal(props.get("precision").get("enum").to_json(), "[\"auto\",\"fp32\",\"fp64\"]")
            assert_equal(props.get("precision").get("default").as_string(), "auto")
            var desc = t.get("description").as_string()
            assert_true(desc.find("QM-W020") >= 0 and desc.find("pauli_frame") >= 0, desc)
        if t.get("name").as_string() == "qumo_convert":
            assert_true(props.get("to").get("enum").to_json().find("\"qasm2\"") >= 0)
        if t.get("name").as_string() == "qumo_describe":
            assert_equal(props.get("format").get("enum").to_json(), "[\"json\",\"markdown\"]")
    # errors
    var pe = rpc(s, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"")
    assert_equal(pe.get("error").get("code").as_int(), -32700)
    assert_true(pe.get("id").is_null())
    assert_equal(pe.get("error").get("data").get("code").as_string(), "QM-E900")
    var mnf = rpc(s, "{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"resources/frobnicate\"}")
    assert_equal(mnf.get("error").get("code").as_int(), -32601)
    assert_equal(mnf.get("id").as_int(), 9)
    var ut = rpc(s, call_line(10, "qumo_nope", "{}"))
    assert_equal(ut.get("error").get("code").as_int(), -32602)
    assert_equal(ut.get("error").get("data").get("code").as_string(), "QM-E912")
    var batch = rpc(s, "[{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}]")
    assert_equal(batch.get("error").get("code").as_int(), -32600)
    var badargs = rpc(s, "{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/call\",\"params\":{\"name\":\"qumo_describe\",\"arguments\":[1]}}")
    assert_equal(badargs.get("error").get("code").as_int(), -32602)
    # stateless helper
    var h = parse_json(handle_request("{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"ping\"}"))
    assert_equal(h.get("id").as_int(), 5)


def test_run_tool() raises:
    var s = McpServer()
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("qasm", BELL)
    w.kv_int("shots", 1000)
    w.kv_int("seed", 42)
    w.end_object()
    var args = w.finish()
    var r = tool(s, 20, "qumo_run_qasm", args)
    assert_false(r.get("isError").as_bool(), r.to_json())
    var sc = r.get("structuredContent")
    var counts = sc.get("counts")
    assert_equal(len(counts), 2)
    var c00 = counts.get("00").as_int()
    var c11 = counts.get("11").as_int()
    assert_equal(c00 + c11, 1000)
    assert_true(abs(c00 - 500) < 80)
    assert_equal(sc.get("method").as_string(), "statevector:static")
    assert_equal(sc.get("shots").as_int(), 1000)
    assert_true(sc.get("timing_ms").get("simulate").as_float() >= 0.0)
    # text content mirrors the structured JSON
    var txt = parse_json(r.get("content").at(0).get("text").as_string())
    assert_equal(txt.get("counts").to_json(), counts.to_json())
    # determinism: same seed -> identical counts
    var r2 = tool(s, 21, "qumo_run_qasm", args)
    assert_equal(r2.get("structuredContent").get("counts").to_json(), counts.to_json())
    # no measurements -> auto measure_all + warning
    var r3 = tool(s, 22, "qumo_run_qasm", "{\"qasm\":\"qubit[1] q; x q[0];\",\"shots\":10}")
    assert_true(r3.get("structuredContent").get("auto_measured").as_bool())
    assert_equal(r3.get("structuredContent").get("counts").get("1").as_int(), 10)
    assert_equal(r3.get("structuredContent").get("warnings").at(0).get("code").as_string(), "QM-W004")
    # params (array and object forms)
    var pq = "OPENQASM 3.0; input float t; qubit[1] q; bit[1] c; rx(t) q[0]; c[0] = measure q[0];"
    var r4 = tool(s, 23, "qumo_run_qasm", "{\"qasm\":" + json_quote(pq) + ",\"params\":[3.141592653589793],\"shots\":50}")
    assert_equal(r4.get("structuredContent").get("counts").get("1").as_int(), 50)
    var r5 = tool(s, 24, "qumo_run_qasm", "{\"qasm\":" + json_quote(pq) + ",\"params\":{\"t\":0},\"shots\":50}")
    assert_equal(r5.get("structuredContent").get("counts").get("0").as_int(), 50)
    # missing params -> structured error
    var r6 = tool(s, 25, "qumo_run_qasm", "{\"qasm\":" + json_quote(pq) + "}")
    assert_true(r6.get("isError").as_bool())
    assert_equal(r6.get("structuredContent").get("error").get("code").as_string(), "QM-E101")
    # QASM error -> isError with location + hint
    var r7 = tool(s, 26, "qumo_run_qasm", "{\"qasm\":\"OPENQASM 3.0;\\nqubit[2] q;\\nhh q[0];\"}")
    assert_true(r7.get("isError").as_bool())
    var err = r7.get("structuredContent").get("error")
    assert_equal(err.get("code").as_string(), "QM-E510")
    assert_equal(err.get("line").as_int(), 3)
    assert_equal(err.get("col").as_int(), 1)
    assert_true(err.get("hint").as_string().find("'h'") >= 0)
    assert_true(err.get("snippet").as_string().find("hh q[0];") >= 0)
    assert_true(r7.get("content").at(0).get("text").as_string().startswith("error[QM-E510]"))
    # bad argument type
    var r8 = tool(s, 27, "qumo_run_qasm", "{\"qasm\":" + json_quote(BELL) + ",\"shots\":\"many\"}")
    assert_equal(r8.get("structuredContent").get("error").get("code").as_string(), "QM-E911")
    var r9 = tool(s, 28, "qumo_run_qasm", "{\"shots\":5}")
    assert_equal(r9.get("structuredContent").get("error").get("code").as_string(), "QM-E910")
    # feed-forward teleportation of a Clifford state is dispatched to the tableau simulator
    var tele = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[3] q; bit[3] c; x q[0]; h q[1]; cx q[1], q[2]; cx q[0], q[1]; h q[0]; c[0] = measure q[0]; c[1] = measure q[1]; if (c[1]) x q[2]; if (c[0]) z q[2]; c[2] = measure q[2];"
    var r10 = tool(s, 29, "qumo_run_qasm", "{\"qasm\":" + json_quote(tele) + ",\"shots\":200}")
    var sc10 = r10.get("structuredContent")
    assert_equal(sc10.get("method").as_string(), "stabilizer:tableau")
    var ones = 0
    for i in range(len(sc10.get("counts"))):
        var k = sc10.get("counts").key_at(i)
        if Int(k.as_bytes()[0]) == ord("1"):
            ones += sc10.get("counts").at(i).as_int()
    assert_equal(ones, 200)


def test_state_and_expectation_tools() raises:
    var s = McpServer()
    var r = tool(s, 30, "qumo_statevector", "{\"qasm\":" + json_quote(BELL) + "}")
    var sc = r.get("structuredContent")
    assert_equal(sc.get("num_qubits").as_int(), 2)
    var amps = sc.get("amplitudes")
    assert_equal(len(amps), 2)
    assert_equal(amps.at(0).get("bitstring").as_string(), "00")
    assert_equal(amps.at(1).get("index").as_int(), 3)
    assert_almost_equal(amps.at(1).get("re").as_float(), 1 / sqrt(2.0))
    assert_true(sc.get("measurements_ignored").as_bool())
    assert_almost_equal(sc.get("norm").as_float(), 1.0)
    var r1 = tool(s, 31, "qumo_statevector", "{\"qasm\":\"qubit[3] q; h q;\",\"max_amplitudes\":3}")
    assert_true(r1.get("structuredContent").get("truncated").as_bool())
    assert_equal(r1.get("structuredContent").get("num_nonzero").as_int(), 8)
    var e = tool(s, 32, "qumo_expectation", "{\"qasm\":" + json_quote(BELL) + ",\"observable\":\"Z0 Z1\"}")
    assert_almost_equal(e.get("structuredContent").get("expectation").as_float(), 1.0)
    var e2 = tool(s, 33, "qumo_expectation", "{\"qasm\":" + json_quote(BELL) + ",\"observable\":[\"0.5*XX\",\"Y0 Y1\",\"Z0\"]}")
    var res = e2.get("structuredContent").get("results")
    assert_almost_equal(res.at(0).get("expectation").as_float(), 0.5)
    assert_almost_equal(res.at(1).get("expectation").as_float(), -1.0)
    assert_almost_equal(res.at(2).get("expectation").as_float(), 0.0)
    var e3 = tool(s, 34, "qumo_expectation", "{\"qasm\":" + json_quote(BELL) + ",\"observable\":\"Z5\"}")
    assert_equal(e3.get("structuredContent").get("error").get("code").as_string(), "QM-E914")
    var e4 = tool(s, 35, "qumo_statevector", "{\"qasm\":\"qubit[30] q; h q[0];\"}")
    assert_equal(e4.get("structuredContent").get("error").get("code").as_string(), "QM-E930")


def test_validate_info_describe_convert() raises:
    var s = McpServer()
    var v = tool(s, 40, "qumo_validate_qasm", "{\"qasm\":" + json_quote(BELL) + "}")
    var sc = v.get("structuredContent")
    assert_true(sc.get("valid").as_bool())
    assert_equal(sc.get("summary").get("num_qubits").as_int(), 2)
    var bad = tool(s, 41, "qumo_validate_qasm", "{\"qasm\":\"OPENQASM 3.0;\\nqubit[2] q;\\ncx q[0], q[0];\"}")
    assert_false(bad.get("isError").as_bool())
    var bsc = bad.get("structuredContent")
    assert_false(bsc.get("valid").as_bool())
    assert_equal(bsc.get("errors").at(0).get("code").as_string(), "QM-E513")
    assert_equal(bsc.get("errors").at(0).get("line").as_int(), 3)
    var lintq = "OPENQASM 3.0; qubit[3] q; bit[2] c; h q[0]; h q[0]; c[0] = measure q[0]; x q[0]; rz(0) q[1];"
    var l = tool(s, 42, "qumo_validate_qasm", "{\"qasm\":" + json_quote(lintq) + "}")
    var ws = l.get("structuredContent").get("warnings")
    var codes = String()
    for i in range(len(ws)):
        codes += ws.at(i).get("code").as_string() + " "
    assert_true(codes.find("QM-W010") >= 0, codes)
    assert_true(codes.find("QM-W002") >= 0, codes)
    assert_true(codes.find("QM-W011") >= 0, codes)
    assert_true(codes.find("QM-W001") >= 0, codes)
    assert_true(codes.find("QM-W009") >= 0, codes)
    var info = tool(s, 43, "qumo_circuit_info", "{\"qasm\":" + json_quote(BELL) + ",\"include_qasm\":true,\"include_instructions\":true}")
    var isc = info.get("structuredContent")
    var summ = isc.get("summary")
    assert_equal(summ.get("depth").as_int(), 2)
    assert_equal(summ.get("two_qubit_gates").as_int(), 1)
    assert_true(summ.get("is_clifford").as_bool())
    assert_equal(summ.get("ops").get("measure").as_int(), 2)
    assert_equal(summ.get("ops").key_at(0), "cx")  # sorted keys
    assert_true(isc.get("qasm").as_string().startswith("OPENQASM 3.0;"))
    assert_equal(len(isc.get("circuit").get("instructions")), 4)
    for topic in ["gates", "errors", "capabilities"]:
        var d = tool(s, 44, "qumo_describe", "{\"topic\":\"" + String(topic) + "\"}")
        assert_false(d.get("isError").as_bool(), d.to_json())
    var g = tool(s, 45, "qumo_describe", "{\"topic\":\"gate\",\"name\":\"cnot\"}")
    assert_equal(g.get("structuredContent").get("name").as_string(), "cx")
    var ge = tool(s, 46, "qumo_describe", "{\"topic\":\"error\",\"name\":\"qm-e510\"}")
    assert_equal(ge.get("structuredContent").get("code").as_string(), "QM-E510")
    var cv = tool(s, 47, "qumo_convert", "{\"qasm\":\"OPENQASM 2.0; include \\\"qelib1.inc\\\"; qreg q[1]; u1(pi/2) q[0];\"}")
    assert_equal(cv.get("structuredContent").get("qasm").as_string(), "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[1] q;\np(pi/2) q[0];\n")
    assert_true(cv.get("content").at(0).get("text").as_string().startswith("OPENQASM 3.0;"))
    var cj = tool(s, 48, "qumo_convert", "{\"qasm\":" + json_quote(BELL) + ",\"to\":\"json\"}")
    var circ = cj.get("structuredContent").get("circuit")
    # JSON circuit accepted as input of other tools
    var r = tool(s, 49, "qumo_run_qasm", "{\"circuit\":" + circ.to_json() + ",\"shots\":64}")
    assert_equal(len(r.get("structuredContent").get("counts")), 2)


def test_describe_and_diagnostics() raises:
    var g = parse_json(describe_gates_json())
    var gates = g.get("gates")
    var found_h = False
    for i in range(len(gates)):
        var x = gates.at(i)
        if x.get("name").as_string() == "h":
            found_h = True
            assert_true(x.get("clifford").as_bool())
            assert_true(x.get("self_inverse").as_bool())
            assert_almost_equal(x.get("matrix").at(1).at(1).at(0).as_float(), -1 / sqrt(2.0))
        if x.get("name").as_string() == "cx":
            var al = x.get("aliases").to_json()
            assert_true(al.find("cnot") >= 0 and al.find("CX") >= 0, al)
            assert_equal(len(x.get("matrix")), 4)
    assert_true(found_h)
    var e = parse_json(describe_errors_json())
    assert_true(len(e.get("codes")) > 40)
    assert_equal(e.get("ranges").at(0).get("title").as_string(), "Circuit structure")
    var e707 = parse_json(describe_error_json("QM-E707"))
    for d in ["'mwpm'", "'sparse_blossom'", "'mwpm_dense'", "'uf'", "'lookup'", "pymatching"]:
        assert_true(e707.get("hint").as_string().find(String(d)) >= 0, e707.to_json())
    assert_true(default_hint("QM-W020").find("QUMO_GPU") >= 0)
    assert_true(default_hint("QM-E407").byte_length() > 0 and default_hint("QM-E712").byte_length() > 0)
    var rs = error_ranges()
    assert_equal(rs[range_index_of("QM-E472", rs)].code, "QM-E47x")
    assert_equal(rs[range_index_of("QM-E404", rs)].code, "QM-E4xx")
    assert_equal(rs[range_index_of("QM-E955", rs)].code, "QM-E95x")
    assert_equal(rs[range_index_of("QM-W020", rs)].code, "QM-W0xx")
    assert_equal(range_index_of("QM-X001", rs), -1)
    var cap = parse_json(describe_capabilities_json(default_registry().names()))
    assert_equal(cap.get("name").as_string(), "qumo")
    assert_true(len(cap.get("mcp_tools")) >= 7)
    # every catalog code is unique
    var cat = error_catalog()
    for i in range(len(cat)):
        for j in range(i + 1, len(cat)):
            assert_true(cat[i].code != cat[j].code, "duplicate " + cat[i].code)
    # diagnostics parsing of core errors (no location) and qasm errors
    var d = diagnostic_from_error("QM-E401: evolve() cannot apply classically-conditioned gates. Hint: use sample(), which runs trajectories for dynamic circuits.")
    assert_equal(d.code, "QM-E401")
    assert_equal(d.message, "evolve() cannot apply classically-conditioned gates")
    assert_equal(d.hint, "use sample(), which runs trajectories for dynamic circuits")
    assert_equal(d.line, 0)
    var d2 = diagnostic_from_error("QM-E101: bind() got 2 values, circuit has 3 parameters [a, b, c].")
    assert_equal(d2.message, "bind() got 2 values, circuit has 3 parameters [a, b, c]")
    assert_true(d2.hint.byte_length() > 0)  # filled from the catalog
    var d3 = diagnostic_from_error("EOF")
    assert_equal(d3.code, "QM-E999")
    assert_equal(parse_error_code("wrapped: QM-E702: x"), "QM-E702")
    try:
        _ = parse_qasm("qubit[2] q;\n  rz(1, 2) q[0];")
    except err:
        var dq = diagnostic_from_error(String(err))
        assert_equal(dq.code, "QM-E511")
        assert_equal(dq.line, 2)
        assert_equal(dq.col, 3)
        assert_true(dq.snippet.find("rz(1, 2) q[0];") >= 0)
        assert_true(dq.hint.find("usage: rz(p0) q[0];") >= 0, dq.hint)
        var j = parse_json(dq.to_json())
        assert_equal(j.get("docs").as_string(), "docs/ERRORS.md#qm-e511")
    # circuit JSON round trip with params, conditions, detectors
    var c = parse_qasm("OPENQASM 3.0; input float t; qubit[2] q; bit[2] c; rx(2*t + 0.5) q[0]; c[0] = measure q[0]; if (c[0]) x q[1]; c[1] = measure q[1];\n#pragma qumo detector 0, 1\n")
    var js = circuit_to_json(c)
    var back = circuit_from_json(parse_json(js))
    assert_equal(circuit_to_json(back), js)
    assert_equal(to_qasm(back), to_qasm(c))
    var sm = parse_json(circuit_summary_json(c))
    assert_true(sm.get("is_dynamic").as_bool())
    assert_equal(sm.get("params").at(0).as_string(), "t")


def test_convert_qasm2_and_describe_markdown() raises:
    var s = McpServer()
    var r = tool(s, 60, "qumo_convert", "{\"qasm\":" + json_quote(BELL) + ",\"to\":\"qasm2\"}")
    assert_false(r.get("isError").as_bool(), r.to_json())
    var sc = r.get("structuredContent")
    assert_equal(sc.get("format").as_string(), "openqasm2")
    assert_equal(sc.get("qasm").as_string(), "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n")
    assert_true(r.get("content").at(0).get("text").as_string().startswith("OPENQASM 2.0;"))
    # gates outside qelib1 come with exact definitions (no undeclared rzx)
    var r2 = tool(s, 61, "qumo_convert", "{\"qasm\":\"qubit[2] q; ecr q[0], q[1]; rzz(0.5) q[1], q[0];\",\"to\":\"qasm2\"}")
    var q2 = r2.get("structuredContent").get("qasm").as_string()
    assert_true(q2.find("gate ecr a,b {") >= 0 and q2.find("gate rzz(theta) a,b {") >= 0, q2)
    assert_true(q2.find("rzx") < 0, q2)
    # parameters: bound by 'params', otherwise QM-E102
    var pq = "OPENQASM 3.0; input float t; qubit[1] q; rx(t) q[0];"
    var r3 = tool(s, 62, "qumo_convert", "{\"qasm\":" + json_quote(pq) + ",\"to\":\"qasm2\",\"params\":[0.5]}")
    assert_true(r3.get("structuredContent").get("qasm").as_string().find("rx(0.5) q[0];") >= 0, r3.to_json())
    var r4 = tool(s, 63, "qumo_convert", "{\"qasm\":" + json_quote(pq) + ",\"to\":\"qasm2\"}")
    assert_equal(r4.get("structuredContent").get("error").get("code").as_string(), "QM-E102")
    var r5 = tool(s, 64, "qumo_convert", "{\"qasm\":" + json_quote(pq) + ",\"params\":{\"t\":0.25}}")
    assert_true(r5.get("structuredContent").get("qasm").as_string().find("rx(0.25) q[0];") >= 0, r5.to_json())
    # describe errors as markdown = docs/ERRORS.md text
    var m = tool(s, 65, "qumo_describe", "{\"topic\":\"errors\",\"format\":\"markdown\"}")
    assert_false(m.get("isError").as_bool(), m.to_json())
    assert_equal(m.get("structuredContent").get("markdown").as_string(), describe_errors_markdown())
    assert_true(m.get("content").at(0).get("text").as_string().startswith("# QUMO error and warning codes"))
    var m2 = tool(s, 66, "qumo_describe", "{\"topic\":\"gates\",\"format\":\"markdown\"}")
    assert_equal(m2.get("structuredContent").get("error").get("code").as_string(), "QM-E913")


def test_exec_tools() raises:
    """QEC decoders and device-explicit runs through the MCP server."""
    var s = McpServer()
    for d in ["mwpm", "sparse_blossom", "mwpm_dense", "uf", "lookup", "pymatching"]:
        var r = tool(s, 70, "qumo_qec_memory", "{\"code\":\"repetition\",\"distance\":3,\"p\":0.01,\"shots\":2000,\"seed\":5,\"decoder\":\"" + String(d) + "\"}")
        assert_false(r.get("isError").as_bool(), String(d) + ": " + r.to_json())
        var sc = r.get("structuredContent")
        assert_true(sc.get("shots").as_int() >= 2000, r.to_json())
        assert_true(sc.get("logical_error_rate").as_float() < 0.05, r.to_json())
    var bad = tool(s, 71, "qumo_qec_memory", "{\"distance\":3,\"shots\":10,\"decoder\":\"blossom5\"}")
    var err = bad.get("structuredContent").get("error")
    assert_equal(err.get("code").as_string(), "QM-E707")
    assert_true(err.get("hint").as_string().find("sparse_blossom") >= 0 and err.get("hint").as_string().find("mwpm_dense") >= 0, err.to_json())
    # explicit devices
    var ghz = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[3] q; bit[3] c; h q[0]; cx q[0], q[1]; cx q[1], q[2]; t q[2]; c = measure q;"
    var rc = tool(s, 72, "qumo_run_device", "{\"qasm\":" + json_quote(ghz) + ",\"shots\":400,\"seed\":9,\"device\":\"cpu\"}")
    var sc = rc.get("structuredContent")
    assert_equal(sc.get("device_used").as_string(), "cpu")
    assert_equal(len(sc.get("warnings")), 0)
    assert_equal(sc.get("counts").get("000").as_int() + sc.get("counts").get("111").as_int(), 400)
    comptime if not is_defined["QUMO_GPU"]():
        var rg = tool(s, 73, "qumo_run_device", "{\"qasm\":" + json_quote(ghz) + ",\"shots\":400,\"seed\":9,\"device\":\"gpu\"}")
        var sg = rg.get("structuredContent")
        assert_false(rg.get("isError").as_bool(), rg.to_json())
        assert_equal(sg.get("device_used").as_string(), "cpu")
        var w = sg.get("warnings").at(0)
        assert_equal(w.get("code").as_string(), "QM-W020")
        assert_true(w.get("message").as_string().find("not compiled") >= 0, w.to_json())
        assert_true(w.get("hint").as_string().byte_length() > 0)
        # same seed and same CPU simulator -> identical counts
        assert_equal(sg.get("counts").to_json(), sc.get("counts").to_json())
    var rb = tool(s, 74, "qumo_run_device", "{\"qasm\":" + json_quote(ghz) + ",\"device\":\"tpu\"}")
    assert_equal(rb.get("structuredContent").get("error").get("code").as_string(), "QM-E940")
    var gi = tool(s, 75, "qumo_gpu_info", "{}")
    assert_false(gi.get("isError").as_bool(), gi.to_json())
    var gsc = gi.get("structuredContent")
    for k in ["precision", "precision_build_default", "precision_env", "precisions", "precision_note", "fp64_supported"]:
        assert_true(gsc.get(String(k)).is_defined(), "qumo_gpu_info lacks " + String(k) + ": " + gi.to_json())
    # GPU precision: explicit selector on the CPU -> fp64 + QM-W021; unknown selector -> QM-E941
    var rp = tool(s, 76, "qumo_run_device", "{\"qasm\":" + json_quote(ghz) + ",\"shots\":400,\"seed\":9,\"device\":\"cpu\",\"precision\":\"fp32\"}")
    var sp = rp.get("structuredContent")
    assert_false(rp.get("isError").as_bool(), rp.to_json())
    assert_equal(sp.get("precision").as_string(), "fp32")
    assert_equal(sp.get("precision_used").as_string(), "fp64")
    assert_equal(sp.get("warnings").at(0).get("code").as_string(), "QM-W021")
    assert_equal(sp.get("counts").to_json(), sc.get("counts").to_json())  # same CPU simulator, same seed
    assert_equal(sc.get("precision").as_string(), "auto")
    assert_equal(sc.get("precision_used").as_string(), "fp64")
    var rq = tool(s, 77, "qumo_run_device", "{\"qasm\":" + json_quote(ghz) + ",\"device\":\"gpu\",\"precision\":\"fp16\"}")
    assert_equal(rq.get("structuredContent").get("error").get("code").as_string(), "QM-E941")


# ---------------------------------------------------------------------------
# docs/ERRORS.md and the error catalogue
# ---------------------------------------------------------------------------
def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _walk(dir: String, ext: String, mut files: List[String]) raises:
    if not exists(dir):
        return
    for name in listdir(dir):
        var path = dir + "/" + name
        if isdir(path):
            _walk(path, ext, files)
        elif path.endswith(ext):
            files.append(path)


def _codes_in(text: String) -> List[String]:
    """Every `QM-[EW]ddd` occurring in `text`."""
    var out = List[String]()
    var bs = text.as_bytes()
    var i = 0
    while i + 7 <= len(bs):
        if Int(bs[i]) == ord("Q") and Int(bs[i + 1]) == ord("M") and Int(bs[i + 2]) == ord("-"):
            var k = Int(bs[i + 3])
            var digits = True
            for j in range(4, 7):
                var c = Int(bs[i + j])
                if c < ord("0") or c > ord("9"):
                    digits = False
            var next_ok = i + 7 == len(bs) or Int(bs[i + 7]) < ord("0") or Int(bs[i + 7]) > ord("9")
            if (k == ord("E") or k == ord("W")) and digits and next_ok:
                out.append(String(text[byte=i:i + 7]))
        i += 1
    return out^


def test_errors_doc_in_sync() raises:
    """The checked-in docs/ERRORS.md equals the generated text (QUMO_UPDATE_DOCS=1 rewrites it)."""
    var md = describe_errors_markdown()
    if getenv("QUMO_UPDATE_DOCS") == "1":
        with open("docs/ERRORS.md", "w") as f:
            f.write(md)
        print("docs/ERRORS.md regenerated from error_catalog()")
    var cur = _read("docs/ERRORS.md")
    assert_true(cur == md, "docs/ERRORS.md is stale: regenerate it with QUMO_UPDATE_DOCS=1 .venv/bin/mojo run -I . tests/test_mcp.mojo")
    var ranges = error_ranges()
    for e in error_catalog():
        assert_true(range_index_of(e.code, ranges) >= 0, e.code + " is outside every range")
        assert_true(md.find("\n### " + e.code + "\n") >= 0, e.code + " has no heading (docs anchor)")
        assert_true(e.meaning.byte_length() > 0 and e.hint.byte_length() > 0, e.code)
    assert_true(md.find("rzx") < 0)


def test_every_raised_code_is_catalogued() raises:
    """A code used in qumo/, cli/, python/, the hardware plugins or the
    benchmark harness must be in error_catalog() (and thus in docs/ERRORS.md)."""
    var files = List[String]()
    _walk("qumo", ".mojo", files)
    _walk("cli", ".mojo", files)
    _walk("python", ".mojo", files)
    _walk("plugins/hardware/qumo_hardware", ".py", files)
    if exists("bench/run_benchmarks.py"):
        files.append("bench/run_benchmarks.py")
    assert_true(len(files) > 30, "run the tests from the repository root")
    var known = List[String]()
    for e in error_catalog():
        known.append(e.code)
    var missing = String()
    var seen = 0
    for f in files:
        for code in _codes_in(_read(f)):
            seen += 1
            var ok = False
            for k in known:
                if k == code:
                    ok = True
                    break
            if not ok and missing.find(code + " ") < 0:
                missing += code + " (" + f + ") "
    assert_true(seen > 100)
    assert_true(missing.byte_length() == 0, "codes used but missing from error_catalog() in qumo/ai/diagnostics.mojo: " + missing)


def main() raises:
    test_protocol()
    test_run_tool()
    test_state_and_expectation_tools()
    test_validate_info_describe_convert()
    test_describe_and_diagnostics()
    test_convert_qasm2_and_describe_markdown()
    test_exec_tools()
    test_errors_doc_in_sync()
    test_every_raised_code_is_catalogued()
    print("test_mcp OK")
