"""One agent as an MCP stdio server, so any MCP client (Claude Code, the Python agents,
another Mojo agent through `qumo-agent call`) can call it like a library.

It exposes a single tool, `run`, with input {task: string}. The result's text is the
agent's answer; structuredContent carries {answer, agent, model, stats}. A failing run
is a result with isError: true, never a crash of the server loop. Stdout carries only
protocol messages; the agent's logs go to stderr.

The JSON-RPC envelope (rpc_result / rpc_error) and the line reader come from QUMO's
own server, qumo/ai/mcp.mojo.
"""

from std.sys import stderr

from qumo.ai.json import JsonWriter, JsonValue, parse_json
from qumo.ai.mcp import rpc_result, rpc_error, LineReader, MCP_PROTOCOL_VERSION

from qumo_agent.config import AgentConfig
from qumo_agent.llm import LlmClient
from qumo_agent.agent import Agent, AgentResult, DEFAULT_MAX_TURNS, DEFAULT_MAX_TOKENS


def _supported(v: String) -> Bool:
    return v == "2025-06-18" or v == "2025-03-26" or v == "2024-11-05"


struct AgentServer(Movable):
    var root: String
    var agent_name: String
    var model: String
    var replay: String
    var max_turns: Int
    var max_tokens: Int
    var description: String
    var agent: Optional[Agent]  # started at the first call, then reused

    def __init__(out self, root: String, agent_name: String, model: String, replay: String,
                 max_turns: Int = DEFAULT_MAX_TURNS, max_tokens: Int = DEFAULT_MAX_TOKENS) raises:
        var cfg = AgentConfig(root, agent_name, model)  # refuses Python-only agents up front
        self.root = root
        self.agent_name = agent_name
        self.model = cfg.model
        self.description = cfg.description
        self.replay = replay
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.agent = None

    def _tools_json(self) -> String:
        var w = JsonWriter()
        w.begin_object()
        w.key("tools")
        w.begin_array()
        w.begin_object()
        w.kv_string("name", "run")
        w.kv_string("title", "Run the QUMO " + self.agent_name + " agent")
        w.kv_string("description", "QUMO agent '" + self.agent_name + "' (Mojo, model " + self.model + "): " + self.description +
                    " Give it a complete task in plain language; it answers with the result and the tool calls it made.")
        w.kv_raw("inputSchema", "{\"type\":\"object\",\"properties\":{\"task\":{\"type\":\"string\",\"description\":\"The task, in plain language, with every circuit or number it needs.\"}},\"required\":[\"task\"],\"additionalProperties\":false}")
        w.end_object()
        w.end_array()
        w.end_object()
        return w.finish()

    def _initialize(self, params: JsonValue) raises -> String:
        var req = params.get_string("protocolVersion", "") if params.is_object() else String()
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("protocolVersion", req if _supported(req) else String(MCP_PROTOCOL_VERSION))
        w.key("capabilities")
        w.begin_object()
        w.key("tools")
        w.begin_object()
        w.kv_bool("listChanged", False)
        w.end_object()
        w.end_object()
        w.key("serverInfo")
        w.begin_object()
        w.kv_string("name", "qumo-agent-" + self.agent_name)
        w.kv_string("title", "QUMO " + self.agent_name + " agent (Mojo)")
        w.kv_string("version", "0.1.0")
        w.end_object()
        w.kv_string("instructions", "Call the tool 'run' with {\"task\": ...}. One call runs the whole agent loop and can take minutes.")
        w.end_object()
        return w.finish()

    def _ensure_agent(mut self) raises:
        if self.agent:
            return
        var cfg = AgentConfig(self.root, self.agent_name, self.model)
        var llm: LlmClient
        if self.replay.byte_length() > 0:
            llm = LlmClient(replay_file=self.replay)
        else:
            llm = LlmClient(cfg.base_url, cfg.api_key_env, cfg.header_names, cfg.header_values,
                            cfg.timeout_s, cfg.max_retries)
        self.agent = Agent(cfg^, llm^, self.max_turns, self.max_tokens)

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
            self._ensure_agent()
            var r = self.agent.value().run(task)
            print("[qumo-agent serve] " + self.agent_name + ": " + String(r.stats), file=stderr, flush=True)
            var w = JsonWriter()
            w.begin_object()
            w.kv_string("answer", r.answer)
            w.kv_string("agent", self.agent_name)
            w.kv_string("model", self.model)
            w.kv_raw("stats", r.stats.to_json())
            w.end_object()
            return self._call_result(False, r.answer, w.finish())
        except e:
            return self._call_result(True, "qumo-agent " + self.agent_name + " failed: " + String(e), "")

    def handle(mut self, line: String) -> String:
        """One JSON-RPC message in, the reply line out ("" for notifications)."""
        try:
            var msg: JsonValue
            try:
                msg = parse_json(line)
            except e:
                return rpc_error(parse_json("null"), -32700, "Parse error: " + String(e))
            if not msg.is_object():
                return rpc_error(parse_json("null"), -32600, "Invalid Request: a JSON-RPC message must be an object")
            var id = msg.get("id")
            var mv = msg.get("method")
            if not id.is_defined():
                return String()  # notification
            if not mv.is_string():
                return rpc_error(id, -32600, "Invalid Request: missing method")
            var method = mv.as_string()
            var params = msg.get("params")
            if method == "initialize":
                return rpc_result(id, self._initialize(params))
            if method == "ping":
                return rpc_result(id, "{}")
            if method == "tools/list":
                return rpc_result(id, self._tools_json())
            if method == "tools/call":
                var name = params.get_string("name", "") if params.is_object() else String()
                if name != "run":
                    return rpc_error(id, -32602, "Unknown tool: " + name + " (this server has one tool, run)")
                var args = params.get("arguments")
                if not args.is_object() or not args.get("task").is_string():
                    return rpc_error(id, -32602, "Invalid params: run needs {\"task\": string}")
                return rpc_result(id, self._run(args.get("task").as_string()))
            return rpc_error(id, -32601, "Method not found: " + method)
        except e:
            try:
                return rpc_error(parse_json("null"), -32603, "Internal error: " + String(e))
            except:
                return String()

    def serve_stdio(mut self) raises:
        var reader = LineReader(0)
        var line = String()
        while reader.read_line(line):
            if String(line.strip()).byte_length() == 0:
                continue
            var reply = self.handle(line)
            if reply.byte_length() > 0:
                print(reply, flush=True)
        if self.agent:
            self.agent.value().close()
