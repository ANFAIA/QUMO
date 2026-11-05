"""The agent loop: system prompt + task, the agent's QUMO MCP tools (with guided
descriptions) plus a local write_report, then call the model, run every tool call,
append the results, and stop when the model answers without tool calls or after
`max_turns` model calls.

Logs go to stderr only, so `serve` can keep stdout for protocol messages.
"""

from std.os import makedirs
from std.os.path import exists
from std.subprocess import run
from std.sys import stderr
from std.time import perf_counter_ns

from qumo.ai.json import JsonWriter, parse_json

from qumo_agent.config import AgentConfig
from qumo_agent.llm import (LlmClient, ChatResponse, build_request, message_json, tool_message_json,
                            function_tool_json, truncate_utf8)
from qumo_agent.mcp_client import McpClient, McpTool

comptime MAX_TOOL_OUTPUT = 12000
comptime DEFAULT_MAX_TURNS = 25
comptime DEFAULT_MAX_TOKENS = 8192

comptime WRITE_REPORT_DESC = "Save a markdown report under agents/reports/ as <date>-<name>.md and return its path. Use it for every finding list, research digest or plan."
comptime WRITE_REPORT_SCHEMA = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"markdown\":{\"type\":\"string\"}},\"required\":[\"name\",\"markdown\"]}"


@fieldwise_init
struct RunStats(Copyable, Writable):
    var turns: Int
    var tool_calls: Int
    var mcp_calls: Int
    var prompt_tokens: Int
    var completion_tokens: Int
    var wall_s: Float64
    var stopped_at_limit: Bool

    def write_to(self, mut w: Some[Writer]):
        w.write("turns=", self.turns, " tool_calls=", self.tool_calls, " mcp_calls=", self.mcp_calls,
                " prompt_tokens=", self.prompt_tokens, " completion_tokens=", self.completion_tokens,
                " wall_s=", self.wall_s)
        if self.stopped_at_limit:
            w.write(" (stopped at max_turns)")

    def to_json(self) -> String:
        var w = JsonWriter()
        w.begin_object()
        w.kv_int("turns", self.turns)
        w.kv_int("tool_calls", self.tool_calls)
        w.kv_int("mcp_calls", self.mcp_calls)
        w.kv_int("prompt_tokens", self.prompt_tokens)
        w.kv_int("completion_tokens", self.completion_tokens)
        w.kv_float("wall_s", self.wall_s)
        w.kv_bool("stopped_at_limit", self.stopped_at_limit)
        w.end_object()
        return w.finish()


@fieldwise_init
struct AgentResult(Copyable):
    var answer: String
    var stats: RunStats


def report_slug(name: String) -> String:
    """Python: re.sub(r"[^\\w-]+", "-", name).strip("-")[:80] or "report". Word
    characters are ASCII letters, digits and '_' here; Python's \\w also keeps non-ASCII
    letters, which this port replaces with '-' (file names stay ASCII)."""
    var bs = name.as_bytes()
    var out = String()
    var in_run = False
    for i in range(len(bs)):
        var c = Int(bs[i])
        var word = (c >= 0x30 and c <= 0x39) or (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c == 0x5F or c == 0x2D
        if word:
            out += chr(c)
            in_run = False
        elif not in_run:
            out += "-"
            in_run = True
    var s = String(out.strip("-"))
    if s.byte_length() > 80:
        var head = String(s[byte=0:80])
        s = head^
    if s.byte_length() == 0:
        return "report"
    return s^


def write_report(root: String, name: String, markdown: String) raises -> String:
    var dir = root + "/agents/reports"
    if not exists(dir):
        makedirs(dir, exist_ok=True)
    var date = String(run("date +%F").strip())
    var rel = "agents/reports/" + date + "-" + report_slug(name) + ".md"
    with open(root + "/" + rel, "w") as f:
        f.write(markdown)
    return rel^


def _log(msg: String):
    print("[qumo-agent] " + msg, file=stderr, flush=True)


struct Agent(Movable):
    var cfg: AgentConfig
    var llm: LlmClient
    var mcp: McpClient
    var tools_json: List[String]
    var mcp_names: List[String]
    var max_turns: Int
    var max_tokens: Int
    var verbose: Bool

    def __init__(out self, var cfg: AgentConfig, var llm: LlmClient, max_turns: Int = DEFAULT_MAX_TURNS,
                 max_tokens: Int = DEFAULT_MAX_TOKENS, verbose: Bool = True) raises:
        var mcp = McpClient(cfg.qumo_bin, ["mcp"])
        mcp.initialize()
        var tools = List[String]()
        var names = List[String]()
        for t in mcp.list_tools():
            if cfg.wants_tool(t.name):
                tools.append(function_tool_json(t.name, cfg.guide.guided_description(t.name, t.description), t.input_schema))
                names.append(t.name)
        if not cfg.mcp_all:
            for want in cfg.mcp_tools:
                var found = False
                for n in names:
                    if n == want:
                        found = True
                if not found:
                    _log("warning: the QUMO MCP server has no tool " + want)
        tools.append(function_tool_json("write_report", WRITE_REPORT_DESC, WRITE_REPORT_SCHEMA))
        self.cfg = cfg^
        self.llm = llm^
        self.mcp = mcp^
        self.tools_json = tools^
        self.mcp_names = names^
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.verbose = verbose

    def _is_mcp(self, name: String) -> Bool:
        for n in self.mcp_names:
            if n == name:
                return True
        return False

    def _run_tool(mut self, name: String, args_text: String, mut stats: RunStats) -> String:
        var args_json = String("{}")
        if String(args_text.strip()).byte_length() > 0:
            try:
                var a = parse_json(args_text)
                if not a.is_object():
                    return "error: tool arguments must be a JSON object"
                args_json = a.to_json()
            except e:
                return "error: tool arguments are not valid JSON: " + String(e)
        if name == "write_report":
            try:
                var a = parse_json(args_json)
                return write_report(self.cfg.root, a.get_string("name", "report"), a.get_string("markdown", ""))
            except e:
                return "error: " + String(e)
        if not self._is_mcp(name):
            var avail = String()
            for n in self.mcp_names:
                avail += n + ", "
            return "error: unknown tool " + name + "; available: " + avail + "write_report"
        try:
            stats.mcp_calls += 1
            var r = self.mcp.call_tool(name, args_json)
            if r.is_error:
                return "error: " + r.text
            return r.text
        except e:
            return "error: " + String(e)

    def run(mut self, task: String) raises -> AgentResult:
        var t0 = perf_counter_ns()
        var stats = RunStats(0, 0, 0, 0, 0, 0.0, False)
        var messages = List[String]()
        messages.append(message_json("system", self.cfg.prompt))
        messages.append(message_json("user", task))
        var answer = String()
        var done = False
        while stats.turns < self.max_turns:
            var resp = self.llm.complete(build_request(self.cfg.model, messages, self.tools_json, self.max_tokens))
            stats.turns += 1
            stats.prompt_tokens += resp.prompt_tokens
            stats.completion_tokens += resp.completion_tokens
            messages.append(resp.assistant_message_json())
            answer = resp.content
            if len(resp.tool_calls) == 0:
                done = True
                break
            for tc in resp.tool_calls:
                stats.tool_calls += 1
                if self.verbose:
                    _log("turn " + String(stats.turns) + ": " + tc.name + " " + truncate_utf8(tc.arguments, 160))
                var out = truncate_utf8(self._run_tool(tc.name, tc.arguments, stats), MAX_TOOL_OUTPUT)
                messages.append(tool_message_json(tc.id, out))
        if not done:
            stats.stopped_at_limit = True
            answer += "\n\n[qumo-agent: stopped after " + String(self.max_turns) + " model turns without a final answer]"
        stats.wall_s = Float64(perf_counter_ns() - t0) / 1e9
        return AgentResult(answer^, stats^)

    def close(mut self) raises:
        _ = self.mcp.close()
