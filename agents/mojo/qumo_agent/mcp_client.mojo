"""A minimal MCP client (JSON-RPC 2.0 over stdio, protocol 2025-06-18).

QUMO ships an MCP server (qumo/ai/mcp.mojo) but no client; this is the client side:
`initialize`, `tools/list` and `tools/call` against any stdio server, such as
`build/qumo mcp` or `build/qumo-agent serve <agent>`. Messages without an `id`
(notifications) and replies to other ids are skipped while waiting for a reply.
"""

from qumo.ai.json import JsonWriter, JsonValue, JsonKind, parse_json, json_quote
from qumo.ai.mcp import MCP_PROTOCOL_VERSION

from qumo_agent.proc import ChildProcess


@fieldwise_init
struct McpTool(Copyable):
    var name: String
    var description: String
    var input_schema: String  # JSON text


@fieldwise_init
struct ToolCallResult(Copyable):
    var is_error: Bool
    var text: String  # the text content blocks joined with newlines
    var structured: String  # structuredContent as JSON text ("" when absent)


struct McpClient(Movable):
    var child: ChildProcess
    var next_id: Int
    var server_name: String

    def __init__(out self, program: String, args: List[String]) raises:
        self.child = ChildProcess(program, args)
        self.next_id = 1
        self.server_name = String()

    def request(mut self, method: String, params_json: String) raises -> JsonValue:
        """Send one request and return its `result`; a JSON-RPC error raises."""
        var id = self.next_id
        self.next_id += 1
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("jsonrpc", "2.0")
        w.kv_int("id", id)
        w.kv_string("method", method)
        w.kv_raw("params", params_json)
        w.end_object()
        self.child.send_line(w.finish())
        var line = String()
        while True:
            if not self.child.read_line(line):
                raise Error("MCP server closed the connection while waiting for " + method)
            if line.byte_length() == 0:
                continue
            var msg = parse_json(line)
            if not msg.is_object():
                continue
            var rid = msg.get("id")
            if not rid.is_number() or rid.as_int() != id:
                continue  # a notification or a reply to something else
            var err = msg.get("error")
            if err.is_defined():
                raise Error("MCP error " + err.get("code").to_json() + " on " + method + ": " + err.get_string("message", ""))
            return msg.get("result")

    def notify(mut self, method: String) raises:
        self.child.send_line("{\"jsonrpc\":\"2.0\",\"method\":" + json_quote(method) + "}")

    def initialize(mut self, client_name: String = "qumo-agent") raises:
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("protocolVersion", MCP_PROTOCOL_VERSION)
        w.key("capabilities")
        w.begin_object()
        w.end_object()
        w.key("clientInfo")
        w.begin_object()
        w.kv_string("name", client_name)
        w.kv_string("version", "0.1.0")
        w.end_object()
        w.end_object()
        var res = self.request("initialize", w.finish())
        var info = res.get("serverInfo")
        if info.is_object():
            self.server_name = info.get_string("name", "")
        self.notify("notifications/initialized")

    def list_tools(mut self) raises -> List[McpTool]:
        var res = self.request("tools/list", "{}")
        var tools = res.get("tools")
        var out = List[McpTool]()
        for i in range(len(tools)):
            var t = tools.at(i)
            var schema = t.get("inputSchema")
            out.append(McpTool(t.get_string("name", ""), t.get_string("description", ""),
                               schema.to_json() if schema.is_object() else String("{\"type\":\"object\"}")))
        return out^

    def call_tool(mut self, name: String, args_json: String) raises -> ToolCallResult:
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("name", name)
        w.kv_raw("arguments", args_json)
        w.end_object()
        var res = self.request("tools/call", w.finish())
        var text = String()
        var content = res.get("content")
        for i in range(len(content)):
            var c = content.at(i)
            if c.get_string("type", "") == "text":
                if text.byte_length() > 0:
                    text += "\n"
                text += c.get_string("text", "")
        var sc = res.get("structuredContent")
        var structured = sc.to_json() if sc.is_defined() and not sc.is_null() else String()
        if text.byte_length() == 0 and structured.byte_length() > 0:
            text = structured
        var is_err = False
        var ev = res.get("isError")
        if ev.is_bool():
            is_err = ev.as_bool()
        return ToolCallResult(is_err, text^, structured^)

    def close(mut self) raises -> Int:
        """Close the server's stdin and wait for it; returns its exit code."""
        return self.child.finish()
