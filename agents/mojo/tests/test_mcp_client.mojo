"""McpClient against QUMO's own MCP server (build/qumo mcp): initialize, list the
tools, run a seeded Bell circuit, and get an isError result for broken QASM."""

from std.os import getenv
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import parse_json, JsonWriter
from qumo_agent.mcp_client import McpClient


def main() raises:
    var qumo = getenv("QUMO_BIN", "build/qumo")
    assert_true(exists(qumo), "build/qumo is missing: run `make cli`")
    var c = McpClient(qumo, ["mcp"])
    c.initialize()
    assert_equal(c.server_name, "qumo")
    var tools = c.list_tools()
    var has_run = False
    var has_describe = False
    for t in tools:
        if t.name == "qumo_run_qasm":
            has_run = True
            assert_true(parse_json(t.input_schema).is_object())
        if t.name == "qumo_describe":
            has_describe = True
    assert_true(has_run and has_describe)
    assert_true(len(tools) >= 24, String("expected at least 24 tools, got ") + String(len(tools)))

    var w = JsonWriter()
    w.begin_object()
    w.kv_string("qasm", "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;")
    w.kv_int("shots", 1000)
    w.kv_int("seed", 7)
    w.end_object()
    var r = c.call_tool("qumo_run_qasm", w.finish())
    assert_false(r.is_error, r.text)
    var counts = parse_json(r.structured).get("counts")
    assert_true(counts.is_object(), r.structured)
    var total = 0
    for k in counts.keys():
        assert_true(k == "00" or k == "11", "unexpected key " + k)
        total += counts.get(k).as_int()
    assert_equal(total, 1000)

    var bad = c.call_tool("qumo_validate_qasm", "{\"qasm\":\"OPENQASM 2.0; qreg q[2]; cx q[0],q[2];\"}")
    assert_true(bad.is_error or bad.text.find("QM-E") >= 0, bad.text)
    assert_equal(c.close(), 0)
    print("test_mcp_client: " + String(len(tools)) + " tools")
    print("test_mcp_client OK")
