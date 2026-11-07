"""The harness's view of the outside world (DESIGN.md 2.7.1).

The harness depends only on `native/`, `util/` and `core/`. Everything else (QUMO
tools, the model client, MCP sessions, child processes, the clock) reaches it through
the `AgentEnv` trait, which the application implements (`app/env.mojo`) and the tests
fake (`harness/testing.mojo`).

`SubagentRequest` and `SubagentResult` are also the wire format of the
`qumo-agent subagent` child protocol (Appendix B.7): one JSON line each way.
"""

from qumo.ai.json import JsonValue, JsonWriter, parse_json

from qumo_agent.core.messages import ChatRequest, ChatResponse
from qumo_agent.core.stats import RunStats, zero_stats
from qumo_agent.core.toolspec import ToolSpec, ToolOutcome

comptime FILE_READ = 0
comptime FILE_WRITE = 1
comptime FILE_EDIT = 2

comptime SUBAGENT_REQUEST_SCHEMA = "qumo.subagent_request.v1"
comptime SUBAGENT_RESULT_SCHEMA = "qumo.subagent_result.v1"


@fieldwise_init
struct AgentDef(Copyable):
    var name: String
    var description: String
    var prompt: String
    var model: String
    var subagents: List[String]      # named subagents, already without self and ancestors
    var recursion_limit: Int


@fieldwise_init
struct FileRoutes(Copyable):
    var repo_read: Bool              # relative read_file reads the repository (or the open sandbox): agent has the `read` group
    var sandbox_write: Bool          # relative write_file/edit_file write the open sandbox: `sandbox` group or mojo-toolsmith


@fieldwise_init
struct RelativeFileOp(Copyable):
    var op: Int
    var path: String                 # relative path as the model wrote it
    var start_line: Int              # FILE_READ, 1-based
    var max_lines: Int               # FILE_READ
    var content: String              # FILE_WRITE
    var old: String                  # FILE_EDIT
    var new: String                  # FILE_EDIT
    var replace_all: Bool            # FILE_EDIT


@fieldwise_init
struct ToolInvocation(Copyable):
    var name: String
    var args: JsonValue              # object, validated and coerced when the spec says so
    var call_id: String
    var agent: String


def _string_field(doc: JsonValue, key: String) raises -> String:
    """A string member; absent or null gives ""; another type raises."""
    var v = doc.get(key)
    if not v.is_defined() or v.is_null():
        return String()
    if not v.is_string():
        raise Error("QA-E120: field '" + key + "' must be a string, got " + v.kind_name())
    return v.as_string()


def _files_field(doc: JsonValue) raises -> String:
    """The `files` member as JSON text; absent or null gives ""."""
    var v = doc.get("files")
    if not v.is_defined() or v.is_null():
        return String()
    if not v.is_object():
        raise Error("QA-E120: field 'files' must be an object, got " + v.kind_name())
    return v.to_json()


@fieldwise_init
struct SubagentRequest(Copyable):
    var parent_agent: String
    var parent_model: String
    var subagent_type: String        # "general-purpose" or a named agent
    var description: String
    var tool_call_id: String
    var ancestors: List[String]      # delegation path, ending with the parent
    var files_json: String           # Vfs.to_json() of the calling tool's view
    var sandbox: String
    var replay_at: String            # the child's replay path

    def to_json(self) -> String:
        """`{"schema":"qumo.subagent_request.v1", ...}` on one line (B.7)."""
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("schema", SUBAGENT_REQUEST_SCHEMA)
        w.kv_string("parent_agent", self.parent_agent)
        w.kv_string("parent_model", self.parent_model)
        w.kv_string("subagent_type", self.subagent_type)
        w.kv_string("description", self.description)
        w.kv_string("tool_call_id", self.tool_call_id)
        w.kv_string_list("ancestors", self.ancestors)
        w.key("files")
        if self.files_json.byte_length() > 0:
            w.raw(self.files_json)
        else:
            w.raw("{}")
        w.kv_string("sandbox", self.sandbox)
        w.kv_string("replay_at", self.replay_at)
        w.end_object()
        return w.finish()


@fieldwise_init
struct SubagentResult(Copyable):
    var ok: Bool
    var content: String
    var files_changed: Bool
    var files_json: String           # the child's whole file dict when changed, else ""
    var sandbox: String
    var stats: RunStats
    var error: String

    def to_json(self) -> String:
        """`{"schema":"qumo.subagent_result.v1", ...}` on one line (B.7); `files` is
        null unless `files_changed`."""
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("schema", SUBAGENT_RESULT_SCHEMA)
        w.kv_bool("ok", self.ok)
        w.kv_string("content", self.content)
        w.kv_bool("files_changed", self.files_changed)
        w.key("files")
        if self.files_changed and self.files_json.byte_length() > 0:
            w.raw(self.files_json)
        else:
            w.null()
        w.kv_string("sandbox", self.sandbox)
        w.key("stats")
        w.raw(self.stats.to_json())
        w.kv_string("error", self.error)
        w.end_object()
        return w.finish()


def subagent_request_from_json(text: String) raises -> SubagentRequest:
    """Decodes one request line (B.7). Raises QA-E120 for a malformed request."""
    var doc: JsonValue
    try:
        doc = parse_json(text)
    except e:
        raise Error("QA-E120: malformed subagent request: " + String(e))
    if not doc.is_object():
        raise Error("QA-E120: malformed subagent request: expected a JSON object, got " + doc.kind_name())
    var schema = _string_field(doc, "schema")
    if schema != SUBAGENT_REQUEST_SCHEMA:
        raise Error("QA-E120: malformed subagent request: schema is '" + schema
                    + "', expected '" + SUBAGENT_REQUEST_SCHEMA + "'")
    var ancestors = List[String]()
    var anc = doc.get("ancestors")
    if anc.is_array():
        for i in range(len(anc)):
            if not anc.at(i).is_string():
                raise Error("QA-E120: malformed subagent request: ancestors must be strings")
            ancestors.append(anc.at(i).as_string())
    elif anc.is_defined() and not anc.is_null():
        raise Error("QA-E120: malformed subagent request: ancestors must be an array")
    var subagent_type = _string_field(doc, "subagent_type")
    if subagent_type.byte_length() == 0:
        raise Error("QA-E120: malformed subagent request: subagent_type is missing")
    return SubagentRequest(
        _string_field(doc, "parent_agent"),
        _string_field(doc, "parent_model"),
        subagent_type,
        _string_field(doc, "description"),
        _string_field(doc, "tool_call_id"),
        ancestors^,
        _files_field(doc),
        _string_field(doc, "sandbox"),
        _string_field(doc, "replay_at"),
    )


def subagent_result_from_json(text: String) raises -> SubagentResult:
    """Decodes one result line (B.7). A child that could not run writes only
    `{"ok": false, "error": "..."}`; missing fields take empty values."""
    var doc: JsonValue
    try:
        doc = parse_json(text)
    except e:
        raise Error("QA-E120: malformed subagent result: " + String(e))
    if not doc.is_object():
        raise Error("QA-E120: malformed subagent result: expected a JSON object, got " + doc.kind_name())
    var ok_v = doc.get("ok")
    if not ok_v.is_bool():
        raise Error("QA-E120: malformed subagent result: 'ok' must be a boolean")
    var changed = False
    var ch = doc.get("files_changed")
    if ch.is_bool():
        changed = ch.as_bool()
    var stats = zero_stats()
    var st = doc.get("stats")
    if st.is_object():
        stats.from_json(st)
    var files_json = _files_field(doc)
    if not changed:
        files_json = String()
    return SubagentResult(
        ok_v.as_bool(),
        _string_field(doc, "content"),
        changed,
        files_json,
        _string_field(doc, "sandbox"),
        stats^,
        _string_field(doc, "error"),
    )


trait AgentEnv(Movable, Deinitable):
    """Everything the harness needs from the application.

    Contract: `call_tool` returns `error_outcome(...)` for tool-level failures and raises
    only for failures Python also lets crash the run (QUMO MCP or alphaXiv transport
    failure after one restart). `relative_file_op` implements QUMO's repo/sandbox
    semantics and output texts. `run_subagents` runs the requests concurrently (child
    processes) and returns results in request order.
    """

    def agent_def(self, name: String, ancestors: List[String]) raises -> AgentDef:
        ...

    def for_agent(self, name: String) raises -> Self:
        ...

    def user_tools(self) -> List[ToolSpec]:
        """QUMO tools in wire order, without read_file/write_file/edit_file."""
        ...

    def file_routes(self) -> FileRoutes:
        ...

    def call_tool(mut self, inv: ToolInvocation) raises -> ToolOutcome:
        ...

    def relative_file_op(mut self, op: RelativeFileOp) raises -> ToolOutcome:
        ...

    def complete(mut self, req: ChatRequest) raises -> ChatResponse:
        ...

    def parallel_limit(self) -> Int:
        """1 = inline subagents only."""
        ...

    def run_subagents(mut self, var reqs: List[SubagentRequest]) raises -> List[SubagentResult]:
        ...

    def sandbox(self) -> String:
        ...

    def adopt_sandbox(mut self, path: String):
        ...

    def now_utc_iso(mut self) -> String:
        ...

    def new_uuid4_hex(mut self) raises -> String:
        ...

    def log(self, msg: String):
        ...

    def take_transport_stats(mut self) -> RunStats:
        """Counters since the last call (http, mcp)."""
        ...
