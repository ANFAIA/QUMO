"""`FakeEnv`: an offline `AgentEnv` for the harness tests and the application's tests
(DESIGN.md 2.7.12).

- the model is a `ReplayModel` over a JSONL file (routed by `replay_at`);
- `agents_json` is a JSON array of `{"name", "description", "prompt", "model",
  "subagents": [...], "tools": [...]?, "recursion_limit": N?}` (the first entry is the
  agent the env starts as; `tools` defaults to echo, big, parts, fail;
  `recursion_limit` defaults to 200 at the top and 9999 for subagents);
- fake user tools, validated like local tools: `echo{text}` -> text; `big{n}` -> n lines
  `line<i>` (1-based); `parts{items}` -> text parts; `fail{}` -> error outcome `boom`;
  `crash{}` raises `QA-E130: crash`. Their descriptions and schemas are what Python sends
  for the same tools in tests/golden/gen_golden.py;
- clock `2026-01-01T00:00:<ss>+00:00`, one second per call; uuid4 hex: a counter as 32
  hex digits;
- relative file operations act on an in-memory map and are recorded;
- `run_subagents` runs `run_subagent_inline` sequentially and records the batch sizes.
"""

from std.collections import Dict
from std.memory import ArcPointer

from qumo.ai.json import JsonValue, parse_json

from qumo_agent.core.messages import ChatRequest, ChatResponse
from qumo_agent.core.replay import ReplayModel
from qumo_agent.core.stats import RunStats, zero_stats
from qumo_agent.core.toolspec import (
    ToolSpec, ToolOutcome, TOOL_LOCAL, text_outcome, error_outcome, parts_outcome,
)
from qumo_agent.harness.env import (
    AgentEnv, AgentDef, FileRoutes, RelativeFileOp, ToolInvocation, SubagentRequest,
    SubagentResult, FILE_READ, FILE_WRITE, FILE_EDIT,
)
from qumo_agent.harness.loop import run_subagent_inline
from qumo_agent.harness.subagents import count_json_for

comptime _HEX = "0123456789abcdef"


@fieldwise_init
struct FakeAgent(Copyable):
    var name: String
    var description: String
    var prompt: String
    var model: String
    var subagents: List[String]
    var tools: List[String]
    var recursion_limit: Int        # -1 = default


struct FakeShared(Movable):
    var replay: ReplayModel
    var agents: List[FakeAgent]
    var routes: FileRoutes
    var clock: Int
    var uuid: Int
    var rel_files: Dict[String, String]
    var rel_ops: List[String]       # "read <path> <start> <max>", "write <path>", "edit <path>"
    var batches: List[Int]          # sizes of run_subagents batches
    var parallel: Int               # parallel_limit()
    var sandbox_path: String
    var logs: List[String]
    var tool_calls: List[String]    # names passed to call_tool, in order

    def __init__(out self, var replay: ReplayModel, var agents: List[FakeAgent], routes: FileRoutes):
        self.replay = replay^
        self.agents = agents^
        self.routes = routes.copy()
        self.clock = 0
        self.uuid = 0
        self.rel_files = Dict[String, String]()
        self.rel_ops = List[String]()
        self.batches = List[Int]()
        self.parallel = 1
        self.sandbox_path = String()
        self.logs = List[String]()
        self.tool_calls = List[String]()


def _strings(v: JsonValue) raises -> List[String]:
    var out = List[String]()
    if v.is_array():
        for i in range(len(v)):
            out.append(v.at(i).as_string())
    return out^


def parse_fake_agents(agents_json: String) raises -> List[FakeAgent]:
    var doc = parse_json(agents_json)
    if not doc.is_array() or len(doc) == 0:
        raise Error("QA-E140: agents_json must be a non-empty JSON array")
    var out = List[FakeAgent]()
    for i in range(len(doc)):
        var a = doc.at(i)
        var tools = _strings(a.get("tools"))
        if not a.has("tools"):
            tools = ["echo", "big", "parts", "fail"]
        out.append(FakeAgent(a.get_string("name", ""), a.get_string("description", ""),
                             a.get_string("prompt", ""), a.get_string("model", "m"),
                             _strings(a.get("subagents")), tools^, a.get_int("recursion_limit", -1)))
    return out^


def fake_tool_spec(name: String) -> ToolSpec:
    """The wire spec of a fake tool (Python's convert_to_openai_tool of the same tool)."""
    var desc: String
    var params: String
    var count_params: String
    if name == "echo":
        desc = "Echo the text back."
        params = "{\"properties\":{\"text\":{\"type\":\"string\"}},\"required\":[\"text\"],\"type\":\"object\"}"
        count_params = "{\"properties\":{\"text\":{\"title\":\"Text\",\"type\":\"string\"}},\"required\":[\"text\"],\"type\":\"object\"}"
    elif name == "big":
        desc = "Return n numbered lines."
        params = "{\"properties\":{\"n\":{\"type\":\"integer\"}},\"required\":[\"n\"],\"type\":\"object\"}"
        count_params = "{\"properties\":{\"n\":{\"title\":\"N\",\"type\":\"integer\"}},\"required\":[\"n\"],\"type\":\"object\"}"
    elif name == "parts":
        desc = "Return each item as a text part."
        params = "{\"properties\":{\"items\":{\"items\":{\"type\":\"string\"},\"type\":\"array\"}},\"required\":[\"items\"],\"type\":\"object\"}"
        count_params = "{\"properties\":{\"items\":{\"items\":{\"type\":\"string\"},\"title\":\"Items\",\"type\":\"array\"}},\"required\":[\"items\"],\"type\":\"object\"}"
    elif name == "fail":
        desc = "Always fails."
        params = "{\"properties\":{},\"type\":\"object\"}"
        count_params = params
    else:
        desc = "Always crashes."
        params = "{\"properties\":{},\"type\":\"object\"}"
        count_params = params
    return ToolSpec(name, desc, params, TOOL_LOCAL, True, count_json_for(name, desc, count_params))


def _two(n: Int) -> String:
    return ("0" if n < 10 else "") + String(n)


struct FakeEnv(AgentEnv):
    var shared: ArcPointer[FakeShared]
    var agent: String

    def __init__(out self, replay_path: String, agents_json: String, routes: FileRoutes) raises:
        var agents = parse_fake_agents(agents_json)
        var first = agents[0].name
        self.shared = ArcPointer(FakeShared(ReplayModel(replay_path), agents^, routes))
        self.agent = first

    def __init__(out self, *, shared: ArcPointer[FakeShared], agent: String):
        self.shared = shared
        self.agent = agent

    def _find(self, name: String) raises -> FakeAgent:
        for a in self.shared[].agents:
            if a.name == name:
                return a.copy()
        raise Error("QA-E140: unknown agent '" + name + "'")

    def agent_def(self, name: String, ancestors: List[String]) raises -> AgentDef:
        var a = self._find(name)
        var subs = List[String]()
        for s in a.subagents:
            if s == name:
                continue
            var is_ancestor = False
            for x in ancestors:
                if x == s:
                    is_ancestor = True
            if not is_ancestor:
                subs.append(s)
        var limit = a.recursion_limit
        if limit < 0:
            limit = 200 if len(ancestors) == 0 else 9999
        return AgentDef(a.name, a.description, a.prompt, a.model, subs^, limit)

    def for_agent(self, name: String) raises -> Self:
        _ = self._find(name)
        return FakeEnv(shared=self.shared, agent=name)

    def user_tools(self) -> List[ToolSpec]:
        var out = List[ToolSpec]()
        try:
            for t in self._find(self.agent).tools:
                out.append(fake_tool_spec(t))
        except:
            pass
        return out^

    def file_routes(self) -> FileRoutes:
        return self.shared[].routes.copy()

    def call_tool(mut self, inv: ToolInvocation) raises -> ToolOutcome:
        self.shared[].tool_calls.append(inv.name)
        if inv.name == "echo":
            return text_outcome(inv.args.get("text").as_string())
        if inv.name == "big":
            var n = inv.args.get("n").as_int()
            var out = String()
            for i in range(1, n + 1):
                if i > 1:
                    out += "\n"
                out += "line" + String(i)
            return text_outcome(out)
        if inv.name == "parts":
            var items = _strings(inv.args.get("items"))
            return parts_outcome(items^, False)
        if inv.name == "fail":
            return error_outcome("boom")
        if inv.name == "crash":
            raise Error("QA-E130: crash")
        return error_outcome("unknown fake tool " + inv.name)

    def relative_file_op(mut self, op: RelativeFileOp) raises -> ToolOutcome:
        if op.op == FILE_READ:
            self.shared[].rel_ops.append("read " + op.path + " " + String(op.start_line) + " " + String(op.max_lines))
            var v = self.shared[].rel_files.get(op.path)
            if v:
                return text_outcome("rel:" + v.value())
            return error_outcome("error: no such file: " + op.path)
        if op.op == FILE_WRITE:
            self.shared[].rel_ops.append("write " + op.path)
            self.shared[].rel_files[op.path] = op.content
            return text_outcome("wrote " + op.path)
        self.shared[].rel_ops.append("edit " + op.path)
        var cur = self.shared[].rel_files.get(op.path)
        if not cur:
            return error_outcome("error: no such file: " + op.path)
        self.shared[].rel_files[op.path] = cur.value().replace(op.old, op.new)
        return text_outcome("edited " + op.path)

    def complete(mut self, req: ChatRequest) raises -> ChatResponse:
        return self.shared[].replay.complete(req)

    def parallel_limit(self) -> Int:
        return self.shared[].parallel

    def run_subagents(mut self, var reqs: List[SubagentRequest]) raises -> List[SubagentResult]:
        self.shared[].batches.append(len(reqs))
        var out = List[SubagentResult]()
        for r in reqs:
            out.append(run_subagent_inline(self, r))
        return out^

    def sandbox(self) -> String:
        return self.shared[].sandbox_path

    def adopt_sandbox(mut self, path: String):
        self.shared[].sandbox_path = path

    def now_utc_iso(mut self) -> String:
        var s = self.shared[].clock
        self.shared[].clock += 1
        return "2026-01-01T00:" + _two((s // 60) % 60) + ":" + _two(s % 60) + "+00:00"

    def new_uuid4_hex(mut self) raises -> String:
        self.shared[].uuid += 1
        var v = self.shared[].uuid
        var out = String()
        var hx = _HEX.as_bytes()
        for i in range(31, -1, -1):
            var nib = (v >> (4 * i)) & 15 if i < 16 else 0
            out += String(unsafe_from_utf8=hx[nib:nib + 1])
        return out^

    def log(self, msg: String):
        self.shared[].logs.append(msg)

    def take_transport_stats(mut self) -> RunStats:
        return zero_stats()
