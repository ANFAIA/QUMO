"""Tests for tools/qumo_mcp.mojo (DESIGN.md 2.8.5, 8.2) against `build/qumo mcp`
(`make cli`, or QUMO_BIN).

- the tool specs of every agent equal what Python sends (tests/fixtures/tools/qumo_mcp_golden.json,
  written by gen_qumo_mcp_golden.py: guided descriptions, `convert_to_openai_tool` parameters),
- a Bell circuit through qumo_run_qasm,
- one persistent session gives the same results as a fresh session per call (X3),
- the session restarts once after the server is killed, and a JSON-RPC error does not restart it,
- a missing binary, shutdown.
"""

from std.os import getenv
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import parse_json, json_quote
from qumo_agent.core.toolspec import ToolOutcome, TOOL_MCP
from qumo_agent.mcp.protocol import MCP_CLIENT_NAME, MCP_CLIENT_VERSION
from qumo_agent.mcp.results import call_result_to_outcome
from qumo_agent.mcp.stdio_client import StdioServerParams, McpStdioSession, DEFAULT_MAX_MESSAGE_BYTES
from qumo_agent.native.libc import c_kill, SIGKILL
from qumo_agent.native.sandbox import Confinement
from qumo_agent.native.spawn import STDIO_INHERIT, mcp_server_env
from qumo_agent.tools.config import Registry
from qumo_agent.tools.qumo_mcp import QumoMcp, ToolGuide, default_qumo_bin

comptime FX = "agents/mojo/tests/fixtures/tools"
comptime BELL = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\nc = measure q;\n"


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def qumo_bin() -> String:
    return default_qumo_bin(".")


def load_guide() raises -> ToolGuide:
    var g = ToolGuide()
    g.load(read_text("agents/prompts/tool_guide.json"))
    return g^


def alive(pid: Int) -> Bool:
    return c_kill(pid, 0) == 0


def json_str(s: String) -> String:
    return json_quote(s)


def run_args(qasm: String, shots: Int, seed: Int) -> String:
    return "{\"qasm\": " + json_str(qasm) + ", \"shots\": " + String(shots) + ", \"seed\": " + String(seed) + "}"


def strip_timing(s: String) -> String:
    """Drops the `"timing_ms":{...}` member, the only part of a result that varies per run."""
    var i = s.find("\"timing_ms\":{")
    if i < 0:
        return s
    var j = s.find("}", i)
    var h_end = i
    var t_start = j + 1
    if i > 0 and s.as_bytes()[i - 1] == 0x2C and s.as_bytes()[t_start] == 0x7D:
        h_end = i - 1
    elif s.as_bytes()[t_start] == 0x2C:
        t_start += 1
    return String(s[byte=0:h_end]) + String(s[byte=t_start:s.byte_length()])


def same(a: ToolOutcome, b: ToolOutcome) -> Bool:
    if a.is_error != b.is_error or a.is_parts != b.is_parts or a.text != b.text or len(a.parts) != len(b.parts):
        return False
    for i in range(len(a.parts)):
        if strip_timing(a.parts[i]) != strip_timing(b.parts[i]):
            return False
    return True


def fresh_call(name: String, args_json: String) raises -> ToolOutcome:
    """The Python way: a new server process and session for one call."""
    var args: List[String] = ["mcp"]
    var params = StdioServerParams(qumo_bin(), args^, mcp_server_env(), "", 0, DEFAULT_MAX_MESSAGE_BYTES,
                                   STDIO_INHERIT, Optional[Confinement](None), "")
    var s = McpStdioSession(params)
    s.initialize(MCP_CLIENT_NAME, MCP_CLIENT_VERSION)
    var out = call_result_to_outcome(s.call_tool(name, args_json))
    _ = s.close()
    return out^


def test_specs() raises:
    var guide = load_guide()
    var mcp = QumoMcp(qumo_bin(), guide)
    var golden = parse_json(read_text(FX + "/qumo_mcp_golden.json"))
    var gt = golden.get("tools")
    assert_equal(len(gt), 24)
    assert_equal(len(mcp.tools), 24)
    var reg = Registry(".")
    var all_specs = mcp.specs_for(True, List[String]())
    assert_equal(len(all_specs), 24)
    for i in range(24):
        var s = all_specs[i].copy()
        assert_equal(s.kind, TOOL_MCP)
        assert_false(s.validate_args)
        assert_equal(s.name, gt.at(i).get_string("name", ""))
        var f = parse_json(s.wire_json()).get("function")
        assert_equal(f.to_json(), gt.at(i).to_json(), "wire function of " + s.name)
    # the nine agents: their subset, in server order
    var agents = golden.get("agents")
    assert_equal(len(agents), 9)
    var empty = 0
    for i in range(len(agents)):
        var name = agents.key_at(i)
        var spec = reg.get(name)
        var got = mcp.specs_for(spec.mcp_all, spec.mcp_tools)
        var want = agents.at(i)
        assert_equal(len(got), len(want), name)
        for k in range(len(want)):
            assert_equal(got[k].name, want.at(k).as_string(), name)
        if len(want) == 0:
            empty += 1
    assert_equal(empty, 2)       # research-watch and code-review have no QUMO tools
    # a name the server does not have is simply absent
    var some: List[String] = ["qumo_describe", "not_a_tool"]
    var sub = mcp.specs_for(False, some)
    assert_equal(len(sub), 1)
    assert_equal(sub[0].name, "qumo_describe")
    assert_true(sub[0].description.startswith("Use when you need facts about QUMO itself"))
    assert_equal(len(mcp.specs_for(False, List[String]())), 0)
    mcp.close()


def test_calls() raises:
    var mcp = QumoMcp(qumo_bin(), load_guide())
    var pid = mcp.pid()
    assert_true(pid > 0)
    # Bell circuit
    var bell = mcp.call("qumo_run_qasm", run_args(BELL, 1000, 7))
    assert_false(bell.is_error)
    assert_true(bell.is_parts and len(bell.parts) == 1)
    var doc = parse_json(bell.parts[0])
    var counts = doc.get("counts")
    assert_true(counts.is_object(), bell.parts[0])
    var total = 0
    for i in range(len(counts)):
        var key = counts.key_at(i)
        assert_true(key == "00" or key == "11", "Bell state gave " + key)
        total += counts.at(i).as_int()
    assert_equal(total, 1000)
    # a tool failure is an outcome, the session keeps working
    var bad = mcp.call("qumo_run_qasm", "{\"qasm\": \"OPENQASM 3.0; qubit[1] q; nope q[0];\"}")
    assert_true(bad.is_error and bad.is_parts and len(bad.parts) == 1)
    # a JSON-RPC error from a live server is raised and does not restart it
    var raised = String()
    try:
        _ = mcp.call("no_such_tool", "{}")
    except e:
        raised = String(e)
    assert_true(raised.startswith("QA-E130: MCP error -32602 on tools/call: Unknown tool: no_such_tool"), raised)
    assert_equal(mcp.restarts, 0)
    assert_equal(mcp.pid(), pid)
    assert_false(mcp.call("qumo_describe", "{}").is_error)
    mcp.close()
    assert_false(alive(pid))
    assert_equal(mcp.pid(), -1)
    mcp.close()    # a second close does nothing


def test_persistent_equals_fresh() raises:
    var mcp = QumoMcp(qumo_bin(), load_guide())
    var names: List[String] = ["qumo_describe", "qumo_run_qasm", "qumo_validate_qasm", "qumo_statevector", "qumo_run_qasm", "qumo_circuit_info"]
    var qasm_bad = "{\"qasm\": \"OPENQASM 3.0; qubit[1] q; nope q[0];\"}"
    var argss: List[String] = [
        "{\"topic\": \"gates\"}",
        run_args(BELL, 200, 3),
        "{\"qasm\": " + json_str(BELL) + "}",
        "{\"qasm\": " + json_str("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nh q[0];\ncx q[0], q[1];\n") + "}",
        qasm_bad,
        "{\"qasm\": " + json_str(BELL) + "}",
    ]
    for i in range(len(names)):
        var persistent = mcp.call(names[i], argss[i])
        var fresh = fresh_call(names[i], argss[i])
        assert_true(same(persistent, fresh), "call " + String(i) + " " + names[i] + " differs between persistent and fresh sessions")
    assert_equal(mcp.restarts, 0)
    mcp.close()


def test_restart() raises:
    var mcp = QumoMcp(qumo_bin(), load_guide())
    assert_false(mcp.call("qumo_describe", "{}").is_error)
    var pid = mcp.pid()
    assert_equal(c_kill(pid, SIGKILL), 0)
    # the next call finds the server dead, starts a new one and succeeds
    var r = mcp.call("qumo_run_qasm", run_args(BELL, 100, 1))
    assert_false(r.is_error)
    assert_equal(mcp.restarts, 1)
    assert_true(mcp.pid() > 0 and mcp.pid() != pid)
    # killed again, and again
    assert_equal(c_kill(mcp.pid(), SIGKILL), 0)
    assert_false(mcp.call("qumo_describe", "{}").is_error)
    assert_equal(mcp.restarts, 2)
    var last = mcp.pid()
    mcp.close()
    assert_false(alive(last))


def test_missing_binary() raises:
    var msg = String()
    try:
        _ = QumoMcp("build/no-such-qumo", load_guide())
    except e:
        msg = String(e)
    assert_equal(msg, "QA-E140: build/no-such-qumo not found. Hint: build it with make cli, or set QUMO_BIN")
    # a binary that is not an MCP server fails at start with QA-E130
    var guide = load_guide()
    var bad = String()
    try:
        _ = QumoMcp("/usr/bin/true", guide)
    except e:
        bad = String(e)
    assert_true(bad.startswith("QA-E130"), bad)


def main() raises:
    test_specs()
    test_calls()
    test_persistent_equals_fresh()
    test_restart()
    test_missing_binary()
    print("test_tools_qumo_mcp OK")
