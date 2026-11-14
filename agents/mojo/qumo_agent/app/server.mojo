"""One agent as an MCP stdio server (`qumo-agent serve AGENT`), so any MCP client (Claude
Code, the Python agents, another Mojo agent through `qumo-agent call`) can call it like a
library (DESIGN.md 2.10, 7.2).

It exposes a single tool, `run`, with input {task: string}. The result's text is the
agent's answer; structuredContent carries {answer, agent, model, stats}. A failing run is a
result with isError: true, never a crash of the server loop. Stdout carries only protocol
messages; the agent's logs go to stderr. Each call is a fresh run (new messages and file
system) that shares the process-wide MCP session, TLS context and model client.

The JSON-RPC envelope is `qumo_agent.mcp.server`; the texts are those of the previous
`server.mojo`.
"""

from std.memory import ArcPointer
from std.sys import stderr

from qumo.ai.json import JsonValue, JsonWriter

from qumo_agent.app.env import AppShared, AppOptions, make_shared, run_agent_once, shutdown_shared
from qumo_agent.mcp.server import McpToolHandler, handler_rpc_error
from qumo_agent.util.redact import redact_all

comptime SERVER_VERSION = "0.1.0"


struct AgentMcpHandler(McpToolHandler):
    var shared: ArcPointer[AppShared]
    var agent: String
    var model: String
    var description: String

    def __init__(out self, var opts: AppOptions, agent: String) raises:
        """Loads the configuration (an unknown agent or a missing key raises QA-E140)."""
        var override = opts.model_override
        self.shared = make_shared(opts^, agent)
        self.agent = agent
        self.model = self.shared[].registry.model_for(agent, override)
        self.description = self.shared[].registry.get(agent).description

    def server_info_json(self) -> String:
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("name", "qumo-agent-" + self.agent)
        w.kv_string("title", "QUMO " + self.agent + " agent (Mojo)")
        w.kv_string("version", SERVER_VERSION)
        w.end_object()
        return w.finish()

    def instructions(self) -> String:
        return "Call the tool 'run' with {\"task\": ...}. One call runs the whole agent loop and can take minutes."

    def tools_list_json(self) -> String:
        var w = JsonWriter()
        w.begin_object()
        w.key("tools")
        w.begin_array()
        w.begin_object()
        w.kv_string("name", "run")
        w.kv_string("title", "Run the QUMO " + self.agent + " agent")
        w.kv_string("description", "QUMO agent '" + self.agent + "' (Mojo, model " + self.model + "): " + self.description +
                    " Give it a complete task in plain language; it answers with the result and the tool calls it made.")
        w.kv_raw("inputSchema", "{\"type\":\"object\",\"properties\":{\"task\":{\"type\":\"string\",\"description\":\"The task, in plain language, with every circuit or number it needs.\"}},\"required\":[\"task\"],\"additionalProperties\":false}")
        w.end_object()
        w.end_array()
        w.end_object()
        return w.finish()

    def _call_result(self, is_error: Bool, text: String, structured: String) -> String:
        var w = JsonWriter()
        w.begin_object()
        w.key("content")
        w.begin_array()
        w.begin_object()
        w.kv_string("type", "text")
        w.kv_string("text", text)
        w.end_object()
        w.end_array()
        if structured.byte_length() > 0:
            w.kv_raw("structuredContent", structured)
        w.kv_bool("isError", is_error)
        w.end_object()
        return w.finish()

    def _run(mut self, task: String) -> String:
        try:
            var r = run_agent_once(self.shared, self.agent, task)
            if self.shared[].opts.verbose:
                print("[qumo-agent serve] " + self.agent + ": " + String(r.stats), file=stderr, flush=True)
            var w = JsonWriter()
            w.begin_object()
            w.kv_string("answer", r.answer)
            w.kv_string("agent", self.agent)
            w.kv_string("model", self.model)
            w.kv_raw("stats", r.stats.to_json())
            w.end_object()
            return self._call_result(False, r.answer, w.finish())
        except e:
            var msg = redact_all(String(e), self.shared[].secrets)
            return self._call_result(True, "qumo-agent " + self.agent + " failed: " + msg, String())

    def call(mut self, name: String, args: JsonValue) -> String:
        try:
            if name != "run":
                return handler_rpc_error(-32602, "Unknown tool: " + name + " (this server has one tool, run)")
            var task = args.get("task")
            if not task.is_string():
                return handler_rpc_error(-32602, "Invalid params: run needs {\"task\": string}")
            return self._run(task.as_string())
        except e:
            return self._call_result(True, "qumo-agent " + self.agent + " failed: " + String(e), String())

    def close(mut self):
        """Ends the QUMO MCP session (call it when stdin ended)."""
        shutdown_shared(self.shared)
