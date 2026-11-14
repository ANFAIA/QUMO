"""Persistent MCP client over stdio (JSON-RPC 2.0, one message per line).

One session is one server process, started with `posix_spawn` in its own process
group with exactly the environment the caller gives (critic C2: QUMO's server gets
`mcp_server_env()`, the six names Python passes), and kept for many calls (X3).

Per request: the request line is written, then lines are read until the response
with the request's id arrives. Notifications and responses to other ids are skipped;
requests from the server are answered with `-32601`; lines that are not JSON are
skipped. A JSON-RPC `error` raises `QA-E130: MCP error <code> on <method>: <message>`.
With `call_timeout_ms > 0`, a request not answered within that time raises
`QA-E130: MCP server did not answer <method> within <n> ms` and the server's process
group is killed; EOF raises `QA-E130: MCP server closed the connection while waiting
for <method>`. After any transport failure the session is unusable (callers start a
new one). Error texts built from server data go through `redact_all`.

With `STDIO_PIPE`, the server's stderr is drained by `Child.read_line` while waiting,
so a server that writes a lot to stderr never blocks; the last 64 KiB stay available
through `stderr_tail()`.

`close()` follows the Python SDK (critic C3): close stdin, wait 2 s, SIGTERM to the
group, wait 2 s, SIGKILL to the group.
"""

from std.time import perf_counter_ns

from qumo.ai.json import JsonValue, parse_json
from qumo.ai.mcp import rpc_error

from qumo_agent.core.errors import E_MCP, qa_error
from qumo_agent.native.libc import SIGKILL
from qumo_agent.native.spawn import (
    SpawnSpec, Child, spawn, STDIO_PIPE, READ_LINE, READ_EOF, READ_TIMEOUT,
    environ_list, is_secret_env_name,
)
from qumo_agent.native.sandbox import Confinement, confine
from qumo_agent.util.redact import redact_all, secret_env_names
from qumo_agent.mcp.protocol import (
    McpTool, McpCallResult, mcp_supported_version, rpc_request_json, rpc_notification_json,
    initialize_params_json, tools_list_params_json, call_params_json,
    parse_tools_page, parse_call_result,
)

comptime DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
comptime CLOSE_GRACE_MS = 2000          # SP/mcp/client/stdio/__init__.py:48 PROCESS_TERMINATION_TIMEOUT
comptime MAX_TOOL_PAGES = 1000          # SP/langchain_mcp_adapters/tools.py:67 MAX_ITERATIONS


@fieldwise_init
struct StdioServerParams(Copyable):
    var program: String
    var args: List[String]
    var env: List[String]            # exact child environment
    var cwd: String                  # "" = inherit
    var call_timeout_ms: Int         # 0 = no timeout (QUMO MCP, as Python); generated tools: from manifest
    var max_message_bytes: Int       # one JSON-RPC line, e.g. 16 MiB
    var stderr_mode: Int             # STDIO_INHERIT (QUMO), STDIO_NULL (generated-tool calls) or STDIO_PIPE (toolsmith_test, review re-test)
    var confinement: Optional[Confinement]   # None for QUMO; CONF_TOOL for generated tools (6.8)
    var self_binary: String          # trampoline for `confine`; "" when confinement is None


def redact_mcp_text(text: String) -> String:
    """`redact_all` over the endpoint-independent secret names plus every variable of
    this process whose name looks secret (`*_API_KEY`, `*_TOKEN`, ...): an MCP session
    does not know which model key is in use, so it hides all of them."""
    var names = List[String]()
    for n in secret_env_names(""):
        if n.byte_length() > 0:
            names.append(n)
    for entry in environ_list():
        var eq = entry.find("=")
        if eq <= 0:
            continue
        var name = String(entry[byte=0:eq])
        if not is_secret_env_name(name):
            continue
        var seen = False
        for n in names:
            if n == name:
                seen = True
        if not seen:
            names.append(name^)
    return redact_all(text, names)


def _now_ms() -> Int:
    return Int(perf_counter_ns() // 1_000_000)


def _is_blank(line: String) -> Bool:
    var bs = line.as_bytes()
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c != 0x20 and c != 0x09 and c != 0x0D and c != 0x0A:
            return False
    return True


def _closed_error(method: String) -> Error:
    return qa_error(E_MCP, "MCP server closed the connection while waiting for " + method, "")


struct McpStdioSession(Movable):
    var server_name: String
    var protocol_version: String
    var _child: Child
    var _next_id: Int
    var _timeout_ms: Int
    var _max_bytes: Int
    var _broken: Bool       # a transport failure happened; no further requests
    var _closed: Bool
    var _reaped: Bool       # the server was reaped (after a kill); `_exit_code` holds its status
    var _exit_code: Int

    def __init__(out self, params: StdioServerParams) raises:
        """Spawn the server (its own process group); `initialize` is a separate step."""
        var argv = List[String]()
        argv.append(params.program)
        for a in params.args:
            argv.append(a)
        var spec = SpawnSpec(params.program, argv^, params.env.copy(), params.cwd,
                             STDIO_PIPE, STDIO_PIPE, params.stderr_mode, True)
        if params.confinement:
            spec = confine(spec, params.confinement.value(), params.self_binary)
        self._child = spawn(spec)
        self.server_name = String()
        self.protocol_version = String()
        self._next_id = 0
        self._timeout_ms = params.call_timeout_ms if params.call_timeout_ms > 0 else 0
        self._max_bytes = params.max_message_bytes if params.max_message_bytes > 0 else DEFAULT_MAX_MESSAGE_BYTES
        self._broken = False
        self._closed = False
        self._reaped = False
        self._exit_code = 0

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    def _send_line(mut self, text: String, method: String) raises:
        if self._broken or self._closed:
            raise _closed_error(method)
        var data = text + "\n"
        try:
            self._child.write_all(data.as_bytes())
        except:
            self._broken = True
            raise _closed_error(method)

    def _kill(mut self):
        """Kill the server's process group and reap it (after a timeout)."""
        self._broken = True
        self._child.signal_group(SIGKILL)
        try:
            self._exit_code = self._child.wait()
            self._reaped = True
        except:
            pass

    def _request(mut self, method: String, params_json: String) raises -> JsonValue:
        """Send one request and return its `result`."""
        var id = self._next_id
        self._next_id += 1
        self._send_line(rpc_request_json(id, method, params_json), method)
        var deadline = _now_ms() + self._timeout_ms if self._timeout_ms > 0 else 0
        var line = String()
        while True:
            var wait_ms = -1
            if self._timeout_ms > 0:
                wait_ms = deadline - _now_ms()
                if wait_ms <= 0:
                    self._kill()
                    raise qa_error(E_MCP, "MCP server did not answer " + method + " within " + String(self._timeout_ms) + " ms", "")
            var status: Int
            try:
                status = self._child.read_line(line, wait_ms, self._max_bytes)
            except e:
                # QA-E161 (line too long) or a read error: the stream is out of step.
                self._kill()
                raise e^
            if status == READ_TIMEOUT:
                continue        # the deadline check above reports it
            if status == READ_EOF:
                self._broken = True
                raise _closed_error(method)
            if _is_blank(line):
                continue
            var msg: JsonValue
            try:
                msg = parse_json(line)
            except:
                continue        # not JSON-RPC (a stray print): skip it
            if not msg.is_object():
                continue
            var rid = msg.get("id")
            var mv = msg.get("method")
            if mv.is_string():
                if rid.is_defined() and not rid.is_null():
                    # A request from the server (sampling, roots, ...): not supported.
                    try:
                        self._send_line(rpc_error(rid, -32601, "Method not found"), method)
                    except:
                        pass
                continue        # notifications are skipped
            if not rid.is_number():
                continue
            var got: Int
            try:
                got = rid.as_int()
            except:
                continue
            if got != id:
                continue
            var err = msg.get("error")
            if err.is_defined() and not err.is_null():
                var code = String("?")
                var cv = err.get("code")
                if cv.is_number():
                    code = cv.number_text()
                elif cv.is_defined():
                    code = cv.to_json()
                var message = String()
                var mt = err.get("message")
                if mt.is_string():
                    message = mt.as_string()
                elif mt.is_defined():
                    message = mt.to_json()
                raise qa_error(E_MCP, redact_mcp_text("MCP error " + code + " on " + method + ": " + message), "")
            return msg.get("result")

    # ------------------------------------------------------------------
    # MCP methods
    # ------------------------------------------------------------------
    def initialize(mut self, client_name: String, client_version: String) raises:
        """`initialize` (protocol 2025-11-25), check the answered version, then
        `notifications/initialized`."""
        var result = self._request("initialize", initialize_params_json(client_name, client_version))
        if not result.is_object():
            raise qa_error(E_MCP, "initialize result is not an object (" + result.kind_name() + ")", "")
        var pv = result.get("protocolVersion")
        var version = pv.as_string() if pv.is_string() else pv.to_json()
        if not mcp_supported_version(version):
            raise qa_error(E_MCP, redact_mcp_text("Unsupported protocol version from the server: " + version), "")
        self.protocol_version = version
        var info = result.get("serverInfo")
        if info.is_object():
            var nv = info.get("name")
            if nv.is_string():
                self.server_name = nv.as_string()
        self._send_line(rpc_notification_json("notifications/initialized"), "notifications/initialized")

    def list_tools(mut self) raises -> List[McpTool]:
        """All tools in server order, following `nextCursor` up to 1000 pages."""
        var out = List[McpTool]()
        var cursor = String()
        var pages = 0
        while True:
            pages += 1
            if pages > MAX_TOOL_PAGES:
                raise qa_error(E_MCP, "Reached max of 1000 iterations while listing tools.", "")
            var page = parse_tools_page(self._request("tools/list", tools_list_params_json(cursor)))
            for t in page[0]:
                out.append(t.copy())
            if page[1].byte_length() == 0:
                break
            cursor = page[1]
        return out^

    def call_tool(mut self, name: String, args_json: String) raises -> McpCallResult:
        """`tools/call {name, arguments}`; a tool failure is `is_error`, not a raise."""
        return parse_call_result(self._request("tools/call", call_params_json(name, args_json)))

    def close(mut self) raises -> Int:
        """Close stdin, wait 2 s, SIGTERM the group, wait 2 s, SIGKILL the group;
        returns the exit code (-N for signal N). A second call returns the same code."""
        if self._closed:
            return self._exit_code
        self._closed = True
        if self._reaped:
            return self._exit_code
        self._exit_code = self._child.terminate(CLOSE_GRACE_MS)
        return self._exit_code

    def pid(self) -> Int:
        return self._child.pid

    def stderr_tail(self) -> String:
        """The last 64 KiB of the server's stderr ("" unless stderr_mode == STDIO_PIPE)."""
        return self._child.stderr_tail()
