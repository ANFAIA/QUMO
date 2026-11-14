"""MCP JSON-RPC messages and result parsing shared by the stdio and HTTP clients.

The client side follows the Python SDK (mcp 1.30.0) as QUMO's agents use it:
`initialize` requests protocol `2025-11-25` with `capabilities: {}` and the given
`clientInfo`, ids are integers starting at 0, the server's answer must be one of the
four supported versions, and `tools/list` is paged with `nextCursor`.
"""

from qumo.ai.json import JsonValue, JsonWriter, json_quote

from qumo_agent.core.errors import E_MCP, qa_error

comptime MCP_CLIENT_PROTOCOL = "2025-11-25"     # SP/mcp/types.py:27

# Client info Python's ClientSession sends (SP/mcp/client/session.py:19).
comptime MCP_CLIENT_NAME = "mcp"
comptime MCP_CLIENT_VERSION = "0.1.0"

comptime _EMPTY_SCHEMA = "{\"type\":\"object\"}"


def mcp_supported_version(v: String) -> Bool:
    """SP/mcp/shared/version.py: 2024-11-05, 2025-03-26, 2025-06-18, 2025-11-25."""
    return v == "2024-11-05" or v == "2025-03-26" or v == "2025-06-18" or v == "2025-11-25"


def rpc_request_json(id: Int, method: String, params_json: String) -> String:
    """`{"jsonrpc":"2.0","id":<id>,"method":<method>,"params":<params>}`; `params` is
    left out when `params_json` is empty (as the SDK does for `tools/list` without a
    cursor)."""
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("jsonrpc", "2.0")
    w.kv_int("id", id)
    w.kv_string("method", method)
    if params_json.byte_length() > 0:
        w.kv_raw("params", params_json)
    w.end_object()
    return w.finish()


def rpc_notification_json(method: String) -> String:
    """`{"jsonrpc":"2.0","method":<method>}`."""
    return "{\"jsonrpc\":\"2.0\",\"method\":" + json_quote(method) + "}"


def initialize_params_json(client_name: String, client_version: String) -> String:
    """`{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{...}}`."""
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("protocolVersion", MCP_CLIENT_PROTOCOL)
    w.key("capabilities")
    w.begin_object()
    w.end_object()
    w.key("clientInfo")
    w.begin_object()
    w.kv_string("name", client_name)
    w.kv_string("version", client_version)
    w.end_object()
    w.end_object()
    return w.finish()


def tools_list_params_json(cursor: String) -> String:
    """Params of `tools/list`: none for the first page, `{"cursor": ...}` after."""
    if cursor.byte_length() == 0:
        return String()
    return "{\"cursor\":" + json_quote(cursor) + "}"


def call_params_json(name: String, args_json: String) -> String:
    """Params of `tools/call`: `{"name": ..., "arguments": ...}` (`{}` when empty)."""
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("name", name)
    if args_json.byte_length() > 0:
        w.kv_raw("arguments", args_json)
    else:
        w.kv_raw("arguments", "{}")
    w.end_object()
    return w.finish()


@fieldwise_init
struct McpTool(Copyable):
    var name: String
    var title: String
    var description: String
    var input_schema_json: String   # raw inputSchema ("{\"type\":\"object\"}" when absent)


@fieldwise_init
struct McpCallResult(Copyable):
    var is_error: Bool
    var parts: List[String]         # one string per content block (section 2.6.4)
    var structured_json: String     # "" when absent (never shown to the model)


def _opt_string(obj: JsonValue, key: String, what: String) raises -> String:
    """A string member, "" when absent or null; anything else is a protocol error."""
    var v = obj.get(key)
    if not v.is_defined() or v.is_null():
        return String()
    if not v.is_string():
        raise qa_error(E_MCP, what + " has a non-string '" + key + "' (" + v.kind_name() + ")", "")
    return v.as_string()


def block_text(block: JsonValue) raises -> String:
    """The text the model sees for one content block (results.content_block_text):
    a `text` block's text; a `resource` block's `resource.text`; for any other block
    a one-line placeholder (X9). Python's adapter would pass images and files as
    non-text blocks or crash on audio; this port is text-only."""
    if not block.is_object():
        raise qa_error(E_MCP, "content block is not an object (" + block.kind_name() + ")", "")
    var tv = block.get("type")
    var kind = tv.as_string() if tv.is_string() else String("unknown")
    if kind == "text":
        var t = block.get("text")
        if not t.is_string():
            raise qa_error(E_MCP, "text content block has no string 'text'", "")
        return t.as_string()
    if kind == "resource":
        var r = block.get("resource")
        if r.is_object():
            var rt = r.get("text")
            if rt.is_string():
                return rt.as_string()
    return "[" + kind + " content omitted: qumo-agent passes text only]"


def parse_tools_page(result: JsonValue) raises -> Tuple[List[McpTool], String]:
    """One `tools/list` result: (tools in server order, nextCursor or "")."""
    if not result.is_object():
        raise qa_error(E_MCP, "tools/list result is not an object (" + result.kind_name() + ")", "")
    var out = List[McpTool]()
    var tools = result.get("tools")
    if tools.is_defined() and not tools.is_null():
        if not tools.is_array():
            raise qa_error(E_MCP, "tools/list result has a non-array 'tools' (" + tools.kind_name() + ")", "")
        for i in range(len(tools)):
            var t = tools.at(i)
            if not t.is_object():
                raise qa_error(E_MCP, "tools/list entry " + String(i) + " is not an object", "")
            var nv = t.get("name")
            if not nv.is_string():
                raise qa_error(E_MCP, "tools/list entry " + String(i) + " has no string 'name'", "")
            var name = nv.as_string()
            var what = "tool '" + name + "'"
            var schema = t.get("inputSchema")
            var schema_json: String
            if not schema.is_defined() or schema.is_null():
                schema_json = String(_EMPTY_SCHEMA)
            elif schema.is_object():
                schema_json = schema.to_json()
            else:
                raise qa_error(E_MCP, what + " has a non-object 'inputSchema' (" + schema.kind_name() + ")", "")
            out.append(McpTool(name, _opt_string(t, "title", what), _opt_string(t, "description", what), schema_json^))
    var cursor = _opt_string(result, "nextCursor", "tools/list result")
    return (out^, cursor^)


def parse_call_result(result: JsonValue) raises -> McpCallResult:
    """A `tools/call` result: `isError`, one text per content block, raw structuredContent."""
    if not result.is_object():
        raise qa_error(E_MCP, "tools/call result is not an object (" + result.kind_name() + ")", "")
    var parts = List[String]()
    var content = result.get("content")
    if content.is_defined() and not content.is_null():
        if not content.is_array():
            raise qa_error(E_MCP, "tools/call result has a non-array 'content' (" + content.kind_name() + ")", "")
        for i in range(len(content)):
            parts.append(block_text(content.at(i)))
    var is_error = False
    var ev = result.get("isError")
    if ev.is_bool():
        is_error = ev.as_bool()
    var structured = String()
    var sc = result.get("structuredContent")
    if sc.is_defined() and not sc.is_null():
        structured = sc.to_json()
    return McpCallResult(is_error, parts^, structured^)
