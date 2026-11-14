"""The MCP stdio server loop used by `qumo-agent serve` (JSON-RPC 2.0, one message per line).

The tools themselves come from a `McpToolHandler`; this module only does the envelope,
with the behaviour of QUMO's own server (`qumo/ai/mcp.mojo`) and the earlier
`qumo_agent/server.mojo`:

- a blank line is ignored; unparsable JSON gives `-32700`; a non-object (a batch
  included) or a request without a string `method` gives `-32600`;
- every message without an `id` is a notification and gets no reply;
- `initialize` echoes the requested protocol version when it is one of the four
  supported versions, otherwise answers the latest (`2025-11-25`, as the Python SDK's
  server does);
- `ping` answers `{}`, `tools/list` the handler's list, `tools/call` the handler's
  `CallToolResult`; an unknown tool or malformed params give `-32602`, any other
  method `-32601`, and an unexpected failure `-32603`;
- stdout carries only protocol lines (logs belong on stderr).
"""

from qumo.ai.json import JsonValue, JsonWriter, parse_json
from qumo.ai.mcp import rpc_result, rpc_error, LineReader

from qumo_agent.mcp.protocol import MCP_CLIENT_PROTOCOL, mcp_supported_version

# A handler's `call` may return `handler_rpc_error(code, message)` instead of a
# CallToolResult to answer with a JSON-RPC error (e.g. -32602 for invalid arguments).
comptime _RPC_ERROR_KEY = "{\"__qumo_rpc_error__\":"


trait McpToolHandler(Movable):
    def server_info_json(self) -> String:
        """`{"name","title","version"}`."""
        ...

    def instructions(self) -> String:
        """The `instructions` of the initialize result ("" = left out)."""
        ...

    def tools_list_json(self) -> String:
        """`{"tools":[...]}`."""
        ...

    def call(mut self, name: String, args: JsonValue) -> String:
        """A CallToolResult JSON (or `handler_rpc_error(...)`); never raises."""
        ...


def handler_rpc_error(code: Int, message: String) -> String:
    """What `McpToolHandler.call` returns to make `handle_line` answer with the
    JSON-RPC error `{code, message}` instead of a result."""
    var w = JsonWriter()
    w.begin_object()
    w.key("__qumo_rpc_error__")
    w.begin_object()
    w.kv_int("code", code)
    w.kv_string("message", message)
    w.end_object()
    w.end_object()
    return w.finish()


def _null_id() raises -> JsonValue:
    return parse_json("null")


def _is_blank(line: String) -> Bool:
    var bs = line.as_bytes()
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c != 0x20 and c != 0x09 and c != 0x0D and c != 0x0A:
            return False
    return True


def _initialize_result[H: McpToolHandler](handler: H, params: JsonValue) raises -> String:
    var requested = String()
    if params.is_object():
        var pv = params.get("protocolVersion")
        if pv.is_string():
            requested = pv.as_string()
    var version = requested if mcp_supported_version(requested) else String(MCP_CLIENT_PROTOCOL)
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("protocolVersion", version)
    w.key("capabilities")
    w.begin_object()
    w.key("tools")
    w.begin_object()
    w.kv_bool("listChanged", False)
    w.end_object()
    w.end_object()
    w.kv_raw("serverInfo", handler.server_info_json())
    var instructions = handler.instructions()
    if instructions.byte_length() > 0:
        w.kv_string("instructions", instructions)
    w.end_object()
    return w.finish()


def _tool_names(tools_list_json: String) -> List[String]:
    var out = List[String]()
    try:
        var tools = parse_json(tools_list_json).get("tools")
        for i in range(len(tools)):
            var nv = tools.at(i).get("name")
            if nv.is_string():
                out.append(nv.as_string())
    except:
        pass
    return out^


def _call_reply[H: McpToolHandler](mut handler: H, id: JsonValue, params: JsonValue) raises -> String:
    if not params.is_object():
        return rpc_error(id, -32602, "Invalid params: tools/call needs {\"name\": ..., \"arguments\": {...}}")
    var nv = params.get("name")
    if not nv.is_string():
        return rpc_error(id, -32602, "Invalid params: missing tool name")
    var name = nv.as_string()
    var names = _tool_names(handler.tools_list_json())
    var known = False
    for n in names:
        if n == name:
            known = True
    if not known:
        var msg = "Unknown tool: " + name
        if len(names) == 1:
            msg += " (this server has one tool, " + names[0] + ")"
        return rpc_error(id, -32602, msg)
    var args = params.get("arguments")
    if not args.is_defined() or args.is_null():
        args = parse_json("{}")
    elif not args.is_object():
        return rpc_error(id, -32602, "Invalid params: arguments must be an object")
    var result = handler.call(name, args)
    if result.startswith(_RPC_ERROR_KEY):
        var err = parse_json(result).get("__qumo_rpc_error__")
        return rpc_error(id, err.get_int("code", -32603), err.get_string("message", ""))
    return rpc_result(id, result)


def _handle[H: McpToolHandler](mut handler: H, line: String) raises -> String:
    var msg: JsonValue
    try:
        msg = parse_json(line)
    except e:
        return rpc_error(_null_id(), -32700, "Parse error: " + String(e))
    if not msg.is_object():
        var why = String("batch requests are not supported; send one message per line") if msg.is_array() else String("a JSON-RPC message must be an object")
        return rpc_error(_null_id(), -32600, "Invalid Request: " + why)
    var id = msg.get("id")
    var is_notification = not id.is_defined()
    var jv = msg.get("jsonrpc")
    if jv.is_defined() and (not jv.is_string() or jv.as_string() != "2.0"):
        if is_notification:
            return String()
        return rpc_error(id, -32600, "Invalid Request: jsonrpc must be \"2.0\"")
    var mv = msg.get("method")
    if is_notification:
        return String()     # notifications (and stray responses) get no reply
    if not mv.is_string():
        return rpc_error(id, -32600, "Invalid Request: missing method")
    var method = mv.as_string()
    var params = msg.get("params")
    if method == "initialize":
        return rpc_result(id, _initialize_result(handler, params))
    if method == "ping":
        return rpc_result(id, "{}")
    if method == "tools/list":
        return rpc_result(id, handler.tools_list_json())
    if method == "tools/call":
        return _call_reply(handler, id, params)
    return rpc_error(id, -32601, "Method not found: " + method)


def handle_line[H: McpToolHandler](mut handler: H, line: String) -> String:
    """One JSON-RPC line in, the reply out ("" for notifications and blank lines)."""
    if _is_blank(line):
        return String()
    try:
        return _handle(handler, line)
    except e:
        try:
            return rpc_error(_null_id(), -32603, "Internal error: " + String(e))
        except:
            return String()


def serve_stdio[H: McpToolHandler](mut handler: H) raises:
    """Answer requests from stdin on stdout until stdin ends."""
    var reader = LineReader(0)
    var line = String()
    while reader.read_line(line):
        var reply = handle_line(handler, line)
        if reply.byte_length() > 0:
            print(reply, flush=True)
