"""Tests for mcp/server.mojo: `handle_line` envelope errors, notifications, version
negotiation, tools/list and tools/call through a fake handler."""

from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, JsonWriter, parse_json
from qumo_agent.mcp.server import McpToolHandler, handle_line, handler_rpc_error


struct EchoHandler(McpToolHandler):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def server_info_json(self) -> String:
        return "{\"name\":\"qumo-agent-test\",\"title\":\"Test agent\",\"version\":\"0.2.0\"}"

    def instructions(self) -> String:
        return "Call run."

    def tools_list_json(self) -> String:
        return "{\"tools\":[{\"name\":\"run\",\"description\":\"d\",\"inputSchema\":{\"type\":\"object\"}}]}"

    def call(mut self, name: String, args: JsonValue) -> String:
        self.calls += 1
        var task = args.get("task")
        if not task.is_string():
            return handler_rpc_error(-32602, "Invalid params: run needs {\"task\": string}")
        var w = JsonWriter()
        w.begin_object()
        w.key("content")
        w.begin_array()
        w.begin_object()
        w.kv_string("type", "text")
        try:
            w.kv_string("text", "ran: " + task.as_string())
        except:
            w.kv_string("text", "")
        w.end_object()
        w.end_array()
        w.kv_bool("isError", False)
        w.end_object()
        return w.finish()


def _reply(mut h: EchoHandler, line: String) raises -> JsonValue:
    var out = handle_line(h, line)
    assert_true(out.byte_length() > 0, "expected a reply to " + line)
    assert_true(out.find("\n") < 0)
    return parse_json(out)


def _error_code(mut h: EchoHandler, line: String) raises -> Int:
    var r = _reply(h, line)
    return r.get("error").get_int("code", 0)


def test_envelope() raises:
    var h = EchoHandler()
    assert_equal(handle_line(h, ""), "")
    assert_equal(handle_line(h, "  \t\r"), "")
    var pe = _reply(h, "{not json")
    assert_equal(pe.get("error").get_int("code", 0), -32700)
    assert_true(pe.get("id").is_null())
    assert_true(pe.get("error").get_string("message", "").startswith("Parse error: "))
    assert_equal(_error_code(h, "[1,2]"), -32600)
    assert_equal(_error_code(h, "5"), -32600)
    assert_equal(_error_code(h, "{\"jsonrpc\":\"1.0\",\"id\":1,\"method\":\"ping\"}"), -32600)
    assert_equal(_error_code(h, "{\"jsonrpc\":\"2.0\",\"id\":1}"), -32600)
    assert_equal(_error_code(h, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"resources/list\"}"), -32601)
    # Notifications and stray responses get no reply.
    assert_equal(handle_line(h, "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}"), "")
    assert_equal(handle_line(h, "{\"jsonrpc\":\"2.0\",\"method\":\"tools/call\",\"params\":{\"name\":\"run\"}}"), "")
    assert_equal(handle_line(h, "{\"jsonrpc\":\"2.0\",\"result\":{}}"), "")
    assert_equal(h.calls, 0)
    # ping keeps a string id.
    var ping = _reply(h, "{\"jsonrpc\":\"2.0\",\"id\":\"p-1\",\"method\":\"ping\"}")
    assert_equal(ping.get("id").as_string(), "p-1")
    assert_equal(ping.get("result").to_json(), "{}")


def test_initialize() raises:
    var h = EchoHandler()
    for v in ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]:
        var r = _reply(h, "{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"" + v + "\",\"capabilities\":{},\"clientInfo\":{\"name\":\"mcp\",\"version\":\"0.1.0\"}}}")
        assert_equal(r.get("result").get_string("protocolVersion", ""), v)
    var other = _reply(h, "{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2030-01-01\"}}")
    var res = other.get("result")
    assert_equal(res.get_string("protocolVersion", ""), "2025-11-25")
    assert_equal(res.get("capabilities").to_json(), "{\"tools\":{\"listChanged\":false}}")
    assert_equal(res.get("serverInfo").get_string("name", ""), "qumo-agent-test")
    assert_equal(res.get_string("instructions", ""), "Call run.")
    var none = _reply(h, "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"initialize\"}")
    assert_equal(none.get("result").get_string("protocolVersion", ""), "2025-11-25")
    assert_equal(none.get("id").as_int(), 3)


def test_tools() raises:
    var h = EchoHandler()
    var lst = _reply(h, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}")
    assert_equal(lst.get("result").get("tools").at(0).get_string("name", ""), "run")
    var call = _reply(h, "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"run\",\"arguments\":{\"task\":\"bell\"}}}")
    assert_equal(call.get("id").as_int(), 2)
    assert_equal(call.get("result").get("content").at(0).get_string("text", ""), "ran: bell")
    assert_equal(h.calls, 1)
    # Unknown tool names the only tool, as the earlier server did.
    var unknown = _reply(h, "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"nope\",\"arguments\":{}}}")
    assert_equal(unknown.get("error").get_int("code", 0), -32602)
    assert_equal(unknown.get("error").get_string("message", ""), "Unknown tool: nope (this server has one tool, run)")
    assert_equal(_error_code(h, "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"params\":[]}"), -32602)
    assert_equal(_error_code(h, "{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"tools/call\",\"params\":{\"arguments\":{}}}"), -32602)
    assert_equal(_error_code(h, "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/call\",\"params\":{\"name\":\"run\",\"arguments\":[1]}}"), -32602)
    assert_equal(h.calls, 1)
    # A handler-chosen JSON-RPC error (absent arguments reach the handler as {}).
    var bad = _reply(h, "{\"jsonrpc\":\"2.0\",\"id\":8,\"method\":\"tools/call\",\"params\":{\"name\":\"run\"}}")
    assert_equal(bad.get("error").get_int("code", 0), -32602)
    assert_equal(bad.get("error").get_string("message", ""), "Invalid params: run needs {\"task\": string}")
    assert_equal(bad.get("id").as_int(), 8)
    assert_equal(h.calls, 2)


def main() raises:
    test_envelope()
    test_initialize()
    test_tools()
    print("test_mcp_server OK")
