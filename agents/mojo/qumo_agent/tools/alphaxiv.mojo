"""alphaXiv (https://www.alphaxiv.org) as the paper source, through its MCP server
(DESIGN.md 2.8.6; Python: `qumo_agents/tools/alphaxiv.py`).

Streamable HTTP, bearer key read from `$ALPHAXIV_API_KEY` (the name comes from the exported
config) on every request and never stored or returned; `mcp/http_client.mojo` redacts it from
every error. Only the five read tools are loaded; the library, folder and follow tools change
the user's alphaXiv account, so agents never get them. Descriptions are
`GUIDE[name] + "\\n\\n" + server_description + suffix` and the parameters go through
`transform_input_schema`, as for QUMO's own tools.

The tool list is loaded once (one session: initialize, tools/list, close). Each call runs in
its own session (initialize, tools/call, close), as the Python adapter does.

`config_json` is the "alphaxiv" object of agents/prompts/tools.json
(`LocalToolsCatalog.alphaxiv_json()`): url_env, url_default, key_env, read_tools, guide,
description_suffix.
"""

from std.memory import ArcPointer
from std.os import getenv

from qumo.ai.json import parse_json

from qumo_agent.core.errors import E_CONFIG, qa_error
from qumo_agent.core.toolspec import ToolSpec, ToolOutcome, TOOL_ALPHAXIV
from qumo_agent.mcp.http_client import McpHttpSession
from qumo_agent.mcp.protocol import McpTool, MCP_CLIENT_NAME, MCP_CLIENT_VERSION
from qumo_agent.mcp.results import call_result_to_outcome
from qumo_agent.mcp.schema import transform_input_schema
from qumo_agent.net.http import HttpClient
from qumo_agent.net.tls import TlsContext


def alphaxiv_available(key_env: String) -> Bool:
    """True when the key variable is set and not empty (Python: `bool(os.environ.get(KEY))`)."""
    return key_env.byte_length() > 0 and getenv(key_env, "").byte_length() > 0


def _contains(items: List[String], x: String) -> Bool:
    for s in items:
        if s == x:
            return True
    return False


struct AlphaXiv(Movable):
    var tls: ArcPointer[TlsContext]
    var url: String
    var key_env: String
    var read_tools: List[String]
    var guide_names: List[String]
    var guide_texts: List[String]
    var description_suffix: String
    var sessions: Int           # MCP sessions opened so far (tests, stats)

    def __init__(out self, tls: ArcPointer[TlsContext], config_json: String) raises:
        var cfg = parse_json(config_json)
        if not cfg.is_object():
            raise qa_error(E_CONFIG, "agents/prompts/tools.json has no 'alphaxiv' object",
                           "run agents/.venv/bin/python -m qumo_agents export-prompts")
        var url_env = cfg.get_string("url_env", "")
        var url = String()
        if url_env.byte_length() > 0:
            url = getenv(url_env, "")
        if url.byte_length() == 0:
            url = cfg.get_string("url_default", "")
        if url.byte_length() == 0:
            raise qa_error(E_CONFIG, "agents/prompts/tools.json is stale: alphaxiv.url_default is missing",
                           "run agents/.venv/bin/python -m qumo_agents export-prompts")
        var key_env = cfg.get_string("key_env", "")
        if not alphaxiv_available(key_env):
            raise qa_error(E_CONFIG, "set " + key_env + " to use alphaXiv", "Settings > API Keys on alphaxiv.org")
        self.tls = tls
        self.url = url^
        self.key_env = key_env^
        self.read_tools = List[String]()
        var rt = cfg.get("read_tools")
        for i in range(len(rt)):
            self.read_tools.append(rt.at(i).as_string())
        self.guide_names = List[String]()
        self.guide_texts = List[String]()
        var g = cfg.get("guide")
        for i in range(len(g)):
            self.guide_names.append(g.key_at(i))
            self.guide_texts.append(g.at(i).as_string())
        self.description_suffix = cfg.get_string("description_suffix", "")
        self.sessions = 0

    def _guide(self, name: String) -> String:
        for i in range(len(self.guide_names)):
            if self.guide_names[i] == name:
                return self.guide_texts[i]
        return String()

    def _open(mut self) raises -> McpHttpSession:
        """A new initialized session. Raises QA-E140 (key not set), QA-E104 or QA-E130."""
        var session = McpHttpSession(HttpClient(Optional[ArcPointer[TlsContext]](self.tls)), self.url, self.key_env)
        self.sessions += 1
        session.initialize()
        return session^

    def load_specs(mut self) raises -> List[ToolSpec]:
        """The read tools in the server's order: one session (initialize, tools/list, close)."""
        var session = self._open()
        var tools: List[McpTool]
        try:
            tools = session.list_tools()
        except e:
            session.close()
            raise e^
        session.close()
        var out = List[ToolSpec]()
        for t in tools:
            if not _contains(self.read_tools, t.name):
                continue
            var description = self._guide(t.name) + "\n\n" + t.description + self.description_suffix
            var transformed = transform_input_schema(t.input_schema_json, description)
            out.append(ToolSpec(t.name, transformed[1], transformed[0], TOOL_ALPHAXIV, False, String()))
        return out^

    def call(mut self, name: String, args_json: String) raises -> ToolOutcome:
        """One session per call: initialize, tools/call, close. Raises QA-E130 for MCP and
        HTTP failures (a tool failure is an `is_error` outcome, not a raise), QA-E104 for
        transport failures."""
        var session = self._open()
        var outcome: ToolOutcome
        try:
            outcome = call_result_to_outcome(session.call_tool(name, args_json))
        except e:
            session.close()
            raise e^
        session.close()
        return outcome^
