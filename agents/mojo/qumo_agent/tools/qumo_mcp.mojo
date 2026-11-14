"""QUMO's own MCP server (`build/qumo mcp`) as agent tools (DESIGN.md 2.8.5).

`ToolGuide` puts a short guide (when to use the tool, what to call first, pitfalls) in front
of the server's description, byte-identical to `qumo_agents.tools.qumo_mcp.guided_description`.
The guides come from agents/prompts/tool_guide.json (exported from Python).

`QumoMcp` keeps one persistent stdio session with `build/qumo mcp` for the whole process (X3):
`tools/list` once at the start, `tools/call` for every call, no `tools/list` after a call and
no `outputSchema` validation. A transport failure (server killed, closed pipe, line too long)
drops the session and the call is retried once on a fresh one; QUMO's tools are stateless per
call, so a repeat is safe. A JSON-RPC error answered by a live server (`QA-E130: MCP error
...`, e.g. an unknown tool) is not a transport failure: it is raised as it is.
"""

from std.memory import ArcPointer
from std.os import getenv
from std.os.path import exists

from qumo.ai.json import parse_json

from qumo_agent.core.errors import E_CONFIG, E_MCP, qa_error
from qumo_agent.core.toolspec import ToolSpec, ToolOutcome, TOOL_MCP
from qumo_agent.mcp.protocol import McpTool, MCP_CLIENT_NAME, MCP_CLIENT_VERSION
from qumo_agent.mcp.results import call_result_to_outcome
from qumo_agent.mcp.schema import transform_input_schema
from qumo_agent.mcp.stdio_client import StdioServerParams, McpStdioSession, DEFAULT_MAX_MESSAGE_BYTES
from qumo_agent.native.sandbox import Confinement
from qumo_agent.native.spawn import STDIO_INHERIT, mcp_server_env

comptime TOOL_GUIDE_SCHEMA = "qumo.tool_guide.v1"
comptime _MCP_ERROR_PREFIX = "QA-E130: MCP error "


struct ToolGuide(Copyable):
    var names: List[String]
    var use_when: List[String]
    var instead_or_first: List[String]
    var watch_out: List[String]

    def __init__(out self):
        self.names = List[String]()
        self.use_when = List[String]()
        self.instead_or_first = List[String]()
        self.watch_out = List[String]()

    def load(mut self, text: String) raises:
        """tool_guide.json text (qumo.tool_guide.v1)."""
        var doc = parse_json(text)
        if doc.get_string("schema", "") != TOOL_GUIDE_SCHEMA:
            raise Error("QA-E140: agents/prompts/tool_guide.json is stale. Hint: run agents/.venv/bin/python -m qumo_agents export-prompts")
        var tools = doc.get("tools")
        for i in range(len(tools)):
            var g = tools.at(i)
            self.names.append(tools.key_at(i))
            self.use_when.append(g.get_string("use_when", ""))
            self.instead_or_first.append(g.get_string("instead_or_first", ""))
            self.watch_out.append(g.get_string("watch_out", ""))

    def guided_description(self, name: String, server_text: String) -> String:
        """Byte-identical to qumo_agents.tools.qumo_mcp.guided_description."""
        var k = -1
        for i in range(len(self.names)):
            if self.names[i] == name:
                k = i
        if k < 0:
            return server_text
        var out = "Use when " + self.use_when[k]
        if self.instead_or_first[k].byte_length() > 0:
            out += "\nInstead or first: " + self.instead_or_first[k]
        if self.watch_out[k].byte_length() > 0:
            out += "\nWatch out: " + self.watch_out[k]
        return out + "\n\n" + server_text


def default_qumo_bin(root: String) -> String:
    """`$QUMO_BIN`, else `<root>/build/qumo` (Python: `QUMO_BIN`, REPO_ROOT / build / qumo)."""
    var b = getenv("QUMO_BIN", "")
    if b.byte_length() > 0:
        return b
    return root + "/build/qumo"


def _contains(items: List[String], x: String) -> Bool:
    for s in items:
        if s == x:
            return True
    return False


def _start_session(qumo_bin: String) raises -> McpStdioSession:
    """Spawn `qumo mcp` with the six-name environment of Python's MCP client and initialize it."""
    var args: List[String] = ["mcp"]
    var params = StdioServerParams(qumo_bin, args^, mcp_server_env(), "", 0, DEFAULT_MAX_MESSAGE_BYTES,
                                   STDIO_INHERIT, Optional[Confinement](None), "")
    var session = McpStdioSession(params)
    try:
        session.initialize(MCP_CLIENT_NAME, MCP_CLIENT_VERSION)
    except e:
        try:
            _ = session.close()
        except:
            pass
        raise e^
    return session^


def _is_protocol_error(e: Error) -> Bool:
    """A JSON-RPC error answered by a working server (not a transport failure)."""
    return String(e).startswith(_MCP_ERROR_PREFIX)


struct QumoMcp(Movable):
    var tools: List[McpTool]                 # server order
    var restarts: Int                        # sessions started after the first one
    var _bin: String
    var _guide: ToolGuide
    var _session: Optional[ArcPointer[McpStdioSession]]

    def __init__(out self, qumo_bin: String, guide: ToolGuide) raises:
        if not exists(qumo_bin):
            raise qa_error(E_CONFIG, qumo_bin + " not found", "build it with make cli, or set QUMO_BIN")
        var session = _start_session(qumo_bin)
        var tools: List[McpTool]
        try:
            tools = session.list_tools()
        except e:
            try:
                _ = session.close()
            except:
                pass
            raise e^
        self.tools = tools^
        self.restarts = 0
        self._bin = qumo_bin
        self._guide = guide.copy()
        self._session = Optional[ArcPointer[McpStdioSession]](ArcPointer(session^))

    def specs_for(self, mcp_all: Bool, names: List[String]) raises -> List[ToolSpec]:
        """The agent's QUMO tools in server order: all of them (`mcp_all`) or those named.
        The description is the guided server text; the parameters are the `inputSchema`
        through `transform_input_schema` (a schema `description` serves only when the
        description is empty)."""
        var out = List[ToolSpec]()
        for t in self.tools:
            if not mcp_all and not _contains(names, t.name):
                continue
            var description = self._guide.guided_description(t.name, t.description)
            var transformed = transform_input_schema(t.input_schema_json, description)
            out.append(ToolSpec(t.name, transformed[1], transformed[0], TOOL_MCP, False, String()))
        return out^

    def pid(self) -> Int:
        """The server's process id, -1 when no session is running (tests, diagnostics)."""
        if self._session:
            return self._session.value()[].pid()
        return -1

    def _drop(mut self):
        """Forget the session (its process is killed when the last reference goes)."""
        if self._session:
            try:
                _ = self._session.value()[].close()
            except:
                pass
        self._session = Optional[ArcPointer[McpStdioSession]](None)

    def _call_once(mut self, name: String, args_json: String) raises -> ToolOutcome:
        if not self._session:
            self._session = Optional[ArcPointer[McpStdioSession]](ArcPointer(_start_session(self._bin)))
            self.restarts += 1
        try:
            return call_result_to_outcome(self._session.value()[].call_tool(name, args_json))
        except e:
            if not _is_protocol_error(e):
                self._drop()
            raise e^

    def call(mut self, name: String, args_json: String) raises -> ToolOutcome:
        """`tools/call` on the persistent session. After a transport failure the session
        is restarted once; a second failure raises (QA-E130, or the spawn error)."""
        try:
            return self._call_once(name, args_json)
        except e:
            if _is_protocol_error(e):
                raise e^
        try:
            return self._call_once(name, args_json)
        except e2:
            var text = String(e2)
            if text.startswith("QA-E130"):
                raise e2^
            raise qa_error(E_MCP, "QUMO MCP server is unavailable: " + text, "check that " + self._bin + " runs (make cli)")

    def close(mut self) raises:
        """Shut the server down (stdin closed, then SIGTERM, then SIGKILL to its group)."""
        if self._session:
            var s = self._session.take()
            _ = s[].close()
