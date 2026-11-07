"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).

The agent loop (DESIGN.md 3.2 and 3.3): one deepagents agent, with the middleware stack
QUMO gets for an OpenAI-compatible model [defaults §1], [ctx §0]:

R0/R1  history = [user(task)], dangling tool calls patched (one super-step)
M0-M4  per model call: recursion check, invalid-call patch (request only), human-message
       eviction and stubs, summarization / overflow recovery, the assistant message
T0-T6  per tool turn: recursion check, malformed-call errors, every valid call against
       the turn-start file system plus its own writes (X1), mutation guard, validation,
       dispatch, eviction, `task` calls (inline or child processes), writes merged in
       call order
R3     the after-agent super-step (not for the general-purpose subagent)
"""

from std.math import ceil
from std.random import random_float64
from std.time import perf_counter_ns, sleep

from qumo.ai.json import JsonValue, parse_json

from qumo_agent.core.errors import error_code, error_http_status
from qumo_agent.core.messages import (
    Message, ToolCall, ChatRequest, ChatResponse, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL,
    user_message, tool_message,
)
from qumo_agent.core.stats import RunStats, zero_stats
from qumo_agent.core.toolspec import (
    ToolSpec, ToolOutcome, TOOL_HARNESS, TOOL_MCP, TOOL_ALPHAXIV, TOOL_GENERATED,
    text_outcome, error_outcome,
)
from qumo_agent.util.pyfmt import cp_len, py_strip
from qumo_agent.harness._fmt import fmt1, fmt2
from qumo_agent.harness.env import (
    AgentEnv, AgentDef, FileRoutes, ToolInvocation, SubagentRequest, SubagentResult,
)
from qumo_agent.harness.eviction import (
    evict_tool_outcome, is_eviction_exempt, human_message_stub, TOOL_EVICT_CHARS, HUMAN_EVICT_CHARS,
)
from qumo_agent.harness.files import merged_file_tools, mutation_key, run_file_tool
from qumo_agent.harness.fs_tools import (
    vfs_ls, vfs_glob, vfs_grep, vfs_delete, execute_unavailable,
)
from qumo_agent.harness.patch import (
    patch_dangling_tool_calls, patch_invalid_tool_calls, malformed_call_message,
)
from qumo_agent.harness.subagents import (
    GENERAL_PURPOSE, task_tool_spec, general_purpose_def, unknown_subagent_text,
    child_replay_at, subagent_final_text, merge_subagent_files, count_json_for,
)
from qumo_agent.harness.summarize import (
    SummaryEvent, SUMMARY_TRIGGER_TOKENS, SUMMARY_KEEP_MESSAGES, apply_event,
    truncate_tool_args, reported_tokens_trigger, safe_cutoff, state_cutoff, xml_transcript,
    summary_prompt_text, overflow_retry_prompt, summary_message, history_section,
    clip_tail_for_overflow,
)
from qumo_agent.harness.texts import (
    LS_DESCRIPTION, LS_PARAMETERS, DELETE_DESCRIPTION, DELETE_PARAMETERS, GLOB_DESCRIPTION,
    GLOB_PARAMETERS, GREP_DESCRIPTION, GREP_PARAMETERS, TASK_PARAMETERS,
    GENERAL_PURPOSE_DESCRIPTION, INVALID_TOOL_NAME, ERR_PARALLEL_MUTATION, RECURSION_LIMIT,
    OVERFLOW_RECOVERY_EXHAUSTED, SUMMARY_NO_HISTORY, COUNT_PARAMETERS_LS,
    COUNT_PARAMETERS_DELETE, COUNT_PARAMETERS_GLOB, COUNT_PARAMETERS_GREP,
)
from qumo_agent.harness.tokens import count_tokens_approx
from qumo_agent.harness.validate import (
    validate_args, task_extra_key_errors, invocation_error_text,
)
from qumo_agent.harness.vfs import (
    Vfs, VfsTx, apply_tx, vfs_from_json, new_file_data, updated_file_data, FileData,
)

comptime _SUMMARY_ATTEMPTS = 3


@fieldwise_init
struct AgentResult(Copyable):
    var answer: String              # content of the last message (QUMO graph.py:91-92)
    var final_text: String          # subagent_final_text(messages)
    var files: Vfs
    var messages: List[Message]
    var stats: RunStats


struct _RunState(Movable):
    var messages: List[Message]
    var files: Vfs
    var summary: Optional[SummaryEvent]
    var session_id: String
    var steps: Int
    var stats: RunStats

    def __init__(out self, task: String, var files: Vfs):
        self.messages = List[Message]()
        self.messages.append(user_message(task))
        self.files = files^
        self.summary = None
        self.session_id = String()
        self.steps = 0
        self.stats = zero_stats()


def _harness_spec(name: String, description: String, parameters: String, count_parameters: String) -> ToolSpec:
    return ToolSpec(name, description, parameters, TOOL_HARNESS, False,
                    count_json_for(name, description, count_parameters))


def _uuid_str(hex: String) -> String:
    """str(uuid) from 32 hex digits: 8-4-4-4-12."""
    if hex.byte_length() != 32:
        return hex
    return (String(hex[byte=0:8]) + "-" + String(hex[byte=8:12]) + "-" + String(hex[byte=12:16]) + "-"
            + String(hex[byte=16:20]) + "-" + String(hex[byte=20:32]))


def _contains(items: List[String], x: String) -> Bool:
    for i in items:
        if i == x:
            return True
    return False


def _str_arg(args: JsonValue, key: String) -> Optional[String]:
    var v = args.get(key)
    if v.is_string():
        try:
            return v.as_string()
        except:
            pass
    return None


def _int_arg(args: JsonValue, key: String) -> Optional[Int]:
    var v = args.get(key)
    if v.is_number():
        try:
            return v.as_int()
        except:
            pass
    return None


def _is_retryable_summary_error(e: Error) -> Bool:
    """LangChain default_retry_on: not retried are context overflows and HTTP 400/401/403/404."""
    var code = error_code(e)
    if code == "QA-E102":
        return False
    if code == "QA-E101":
        var status = error_http_status(e)
        if status == 400 or status == 401 or status == 403 or status == 404:
            return False
    return True


struct Agent[E: AgentEnv](Movable):
    var env: Self.E
    var spec: AgentDef
    var ancestors: List[String]
    var replay_at: String
    var general_purpose: Bool

    def __init__(out self, var env: Self.E, spec: AgentDef, ancestors: List[String], replay_at: String, general_purpose: Bool):
        self.env = env^
        self.spec = spec.copy()
        self.ancestors = ancestors.copy()
        self.replay_at = replay_at
        self.general_purpose = general_purpose

    # -----------------------------------------------------------------------
    # tools
    # -----------------------------------------------------------------------
    def _subagent_names(self) -> List[String]:
        var names = List[String]()
        names.append(GENERAL_PURPOSE)
        for s in self.spec.subagents:
            names.append(s)
        return names^

    def _child_ancestors(self) -> List[String]:
        var a = self.ancestors.copy()
        a.append(self.spec.name)
        return a^

    def wire_tools(self) -> List[ToolSpec]:
        """Section 5.1: ls, read_file, write_file, edit_file, delete, glob, grep, task, then
        the user tools. The general-purpose subagent has no `task`."""
        var routes = self.env.file_routes()
        var merged = merged_file_tools(routes)
        var out = List[ToolSpec]()
        out.append(_harness_spec("ls", LS_DESCRIPTION, LS_PARAMETERS, COUNT_PARAMETERS_LS))
        for t in merged:
            out.append(t.copy())
        out.append(_harness_spec("delete", DELETE_DESCRIPTION, DELETE_PARAMETERS, COUNT_PARAMETERS_DELETE))
        out.append(_harness_spec("glob", GLOB_DESCRIPTION, GLOB_PARAMETERS, COUNT_PARAMETERS_GLOB))
        out.append(_harness_spec("grep", GREP_DESCRIPTION, GREP_PARAMETERS, COUNT_PARAMETERS_GREP))
        if not self.general_purpose:
            var names = self._subagent_names()
            var descriptions = List[String]()
            descriptions.append(GENERAL_PURPOSE_DESCRIPTION)
            var anc = self._child_ancestors()
            for i in range(1, len(names)):
                var d = String()
                try:
                    d = self.env.agent_def(names[i], anc).description
                except e:
                    self.env.log("could not describe subagent " + names[i] + ": " + String(e))
                descriptions.append(d)
            out.append(task_tool_spec(names, descriptions))
        for t in self.env.user_tools():
            out.append(t.copy())
        return out^

    def tool_node_names(self) -> List[String]:
        """Wire names with the hidden `execute` inserted right after `grep`."""
        var out = List[String]()
        for t in self.wire_tools():
            out.append(t.name)
            if t.name == "grep":
                out.append("execute")
        return out^

    # -----------------------------------------------------------------------
    # recursion
    # -----------------------------------------------------------------------
    def _step(self, mut st: _RunState) raises:
        st.steps += 1
        if st.steps > self.spec.recursion_limit:
            var text = "QA-E110: " + fmt1(RECURSION_LIMIT, "limit", String(self.spec.recursion_limit))
            if len(self.ancestors) == 0:
                text += ". Hint: raise --recursion-limit"
            raise Error(text)

    # -----------------------------------------------------------------------
    # model calls (section 3.3)
    # -----------------------------------------------------------------------
    def _call_model(mut self, mut st: _RunState, msgs: List[Message], tools: List[ToolSpec]) raises -> ChatResponse:
        var req = ChatRequest(self.spec.model, self.spec.prompt, msgs.copy(), tools.copy(), self.spec.name, self.replay_at)
        var resp = self.env.complete(req)
        st.stats.model_calls += 1
        st.stats.prompt_tokens += resp.prompt_tokens
        st.stats.completion_tokens += resp.completion_tokens
        return resp^

    def _count(self, msgs: List[Message], tools: List[ToolSpec]) -> Int:
        return count_tokens_approx(self.spec.prompt, msgs, tools, True)

    def _state_index(self, st: _RunState, effective_index: Int) -> Int:
        if st.summary:
            if effective_index == 0:
                return -1
            return st.summary.value().cutoff_index + effective_index - 1
        return effective_index

    def _call_with_budget(mut self, mut st: _RunState, var msgs: List[Message], tools: List[ToolSpec], error: String, rejected: List[Message]) raises -> ChatResponse:
        var replaced_any = False
        if error.byte_length() > 0:
            var tx = VfsTx()
            var clipped = clip_tail_for_overflow(msgs, st.files, tx, self.env.now_utc_iso())
            apply_tx(st.files, tx)
            for i in clipped[1]:
                var si = self._state_index(st, i)
                if si >= 0 and si < len(st.messages):
                    st.messages[si] = clipped[0][i].copy()
            replaced_any = len(clipped[1]) > 0
            if self._count(clipped[0], tools) >= self._count(rejected, tools):
                raise Error("QA-E102: " + OVERFLOW_RECOVERY_EXHAUSTED)
            msgs = clipped[0].copy()
        try:
            return self._call_model(st, msgs, tools)
        except e:
            if error_code(e) != "QA-E102":
                raise e^
            if error.byte_length() > 0 or replaced_any:
                raise Error("QA-E102: " + OVERFLOW_RECOVERY_EXHAUSTED)
            return self._call_with_budget(st, msgs.copy(), tools, String(e), msgs)

    def _invoke_summary(mut self, mut st: _RunState, prompt: String) raises -> String:
        """One summary request (one user message, no tools), up to 3 attempts."""
        var attempt = 0
        while True:
            attempt += 1
            try:
                var msgs = List[Message]()
                msgs.append(user_message(prompt))
                var req = ChatRequest(self.spec.model, "", msgs^, List[ToolSpec](), self.spec.name, self.replay_at)
                var resp = self.env.complete(req)
                st.stats.model_calls += 1
                st.stats.prompt_tokens += resp.prompt_tokens
                st.stats.completion_tokens += resp.completion_tokens
                return py_strip(resp.message.text())
            except e:
                st.stats.model_calls += 1
                if attempt >= _SUMMARY_ATTEMPTS or not _is_retryable_summary_error(e):
                    raise e^
                # tenacity wait_exponential_jitter defaults: 1 s, then 2 s, plus U(0, 1).
                var wait = Float64(1 << (attempt - 1)) + random_float64(0.0, 1.0)
                self.env.log("summary request failed (" + String(e) + "); retrying in " + String(wait) + " s")
                sleep(wait)

    def _create_summary(mut self, mut st: _RunState, to_sum: List[Message]) raises -> String:
        if len(to_sum) == 0:
            return SUMMARY_NO_HISTORY
        try:
            return self._invoke_summary(st, summary_prompt_text(xml_transcript(to_sum)))
        except e:
            if error_code(e) != "QA-E102":
                raise e^
            var retry = overflow_retry_prompt(to_sum)
            if not retry:
                raise e^
            return self._invoke_summary(st, retry.value())

    def _model_step(mut self, mut st: _RunState, req_msgs: List[Message], tools: List[ToolSpec]) raises -> ChatResponse:
        var eff = apply_event(req_msgs, st.summary)
        var total = self._count(eff, tools)
        var tr = truncate_tool_args(eff)
        var msgs = tr[0].copy()
        if tr[1]:
            total = self._count(msgs, tools)
        var overflow = String()
        if not (total >= SUMMARY_TRIGGER_TOKENS or reported_tokens_trigger(msgs)):
            try:
                return self._call_model(st, msgs, tools)
            except e:
                if error_code(e) != "QA-E102":
                    raise e^
                overflow = String(e)
        var cutoff = safe_cutoff(msgs, SUMMARY_KEEP_MESSAGES)
        if cutoff <= 0:
            return self._call_with_budget(st, msgs.copy(), tools, overflow, msgs)
        var to_sum = List[Message]()
        var keep = List[Message]()
        for i in range(len(msgs)):
            if i < cutoff:
                to_sum.append(msgs[i].copy())
            else:
                keep.append(msgs[i].copy())
        if st.session_id.byte_length() == 0:
            st.session_id = "session_" + self.env.new_uuid4_hex()
        var path = "/conversation_history/" + st.session_id + ".md"
        var now = self.env.now_utc_iso()
        var existing = st.files.get(path)
        var section = history_section(now, to_sum)
        if existing:
            st.files.set(path, updated_file_data(existing.value(), existing.value().content + section, now))
        else:
            st.files.set(path, new_file_data(section, now))
        var summary = self._create_summary(st, to_sum)
        var smsg = summary_message(summary, path)
        st.summary = SummaryEvent(state_cutoff(st.summary, cutoff), smsg.copy(), path, len(req_msgs))
        st.stats.summarizations += 1
        var next = List[Message]()
        next.append(smsg^)
        for m in keep:
            next.append(m.copy())
        return self._call_with_budget(st, next^, tools, overflow, msgs)

    # -----------------------------------------------------------------------
    # tools of one turn
    # -----------------------------------------------------------------------
    def _find_user_tool(self, tools: List[ToolSpec], name: String) -> Int:
        for i in range(len(tools)):
            if tools[i].name == name and tools[i].kind != TOOL_HARNESS:
                return i
        return -1

    def _find_tool(self, tools: List[ToolSpec], name: String) -> Int:
        for i in range(len(tools)):
            if tools[i].name == name:
                return i
        return -1

    def _dispatch(mut self, mut st: _RunState, call: ToolCall, tools: List[ToolSpec], node_names: List[String], routes: FileRoutes, base: Vfs, mut tx: VfsTx) raises -> ToolOutcome:
        """Step T3.b for every tool but a valid `task` call (handled by the caller)."""
        var name = call.name
        if not _contains(node_names, name):
            return error_outcome(fmt2(INVALID_TOOL_NAME, "requested_tool", name, "available_tools", String(", ").join(node_names)))
        if name == "execute":
            return execute_unavailable()
        var ti = self._find_tool(tools, name)
        if name == "ls" or name == "glob" or name == "grep" or name == "delete":
            var chk = validate_args(tools[ti].parameters_json, call.args)
            if not chk.ok:
                return error_outcome(invocation_error_text(name, call.args, chk.errors))
            var a = chk.args.copy()
            if name == "ls":
                return vfs_ls(base, tx, _str_arg(a, "path").value())
            if name == "delete":
                return vfs_delete(base, tx, _str_arg(a, "file_path").value())
            if name == "glob":
                return vfs_glob(base, tx, _str_arg(a, "pattern").value(), _str_arg(a, "path"))
            var mode = _str_arg(a, "output_mode")
            return vfs_grep(base, tx, _str_arg(a, "pattern").value(), _str_arg(a, "path"), _str_arg(a, "glob"),
                            mode.value() if mode else String("files_with_matches"), _int_arg(a, "max_count"))
        if name == "read_file" or name == "write_file" or name == "edit_file":
            var chk = validate_args(tools[ti].parameters_json, call.args)
            if not chk.ok:
                return error_outcome(invocation_error_text(name, call.args, chk.errors))
            return run_file_tool(self.env, name, chk.args, call.args, base, tx, routes)
        # user tools
        var ui = self._find_user_tool(tools, name)
        if ui < 0:
            return error_outcome(fmt2(INVALID_TOOL_NAME, "requested_tool", name, "available_tools", String(", ").join(node_names)))
        var args = call.args.copy()
        if tools[ui].validate_args:
            var chk = validate_args(tools[ui].parameters_json, call.args)
            if not chk.ok:
                return error_outcome(invocation_error_text(name, call.args, chk.errors))
            args = chk.args.copy()
        var kind = tools[ui].kind
        if kind == TOOL_MCP or kind == TOOL_ALPHAXIV or kind == TOOL_GENERATED:
            st.stats.mcp_calls += 1
        return self.env.call_tool(ToolInvocation(name, args^, call.id, self.spec.name))

    def _evict(mut self, name: String, call_id: String, outcome: ToolOutcome, base: Vfs, mut tx: VfsTx, mut st: _RunState) raises -> ToolOutcome:
        """Step T3.c; the clock and the uuid are only read when a result is evicted."""
        if is_eviction_exempt(name) or cp_len(outcome.joined_text()) <= TOOL_EVICT_CHARS:
            return outcome.copy()
        var fallback = String()
        if call_id.byte_length() == 0:
            fallback = String(self.env.new_uuid4_hex()[byte=0:8])
        var r = evict_tool_outcome(name, call_id, outcome, base, tx, self.env.now_utc_iso(), fallback)
        if r[1]:
            st.stats.evictions += 1
        return r[0].copy()

    def _evict_human(mut self, mut st: _RunState) raises:
        """FilesystemMiddleware human-message eviction (step M2)."""
        var last = len(st.messages) - 1
        if last < 0:
            return
        var m = st.messages[last].copy()
        if m.role != ROLE_USER or m.evicted_to.byte_length() > 0 or m.lc_source.byte_length() > 0:
            return
        if cp_len(m.content) <= HUMAN_EVICT_CHARS:
            return
        var path = "/conversation_history/" + _uuid_str(self.env.new_uuid4_hex()) + ".md"
        st.files.set(path, new_file_data(m.content, self.env.now_utc_iso()))
        st.messages[last].evicted_to = path
        st.stats.evictions += 1

    # -----------------------------------------------------------------------
    # run
    # -----------------------------------------------------------------------
    def run(mut self, task: String, var files: Vfs) raises -> AgentResult:
        var t0 = perf_counter_ns()
        var st = _RunState(task, files^)
        # R1: before_agent
        st.messages = patch_dangling_tool_calls(st.messages)
        self._step(st)
        var tools = self.wire_tools()
        var node_names = self.tool_node_names()
        var routes = self.env.file_routes()
        while True:
            # M0
            self._step(st)
            # M2 (state part): evict an oversized last user message.
            self._evict_human(st)
            # M1
            var req_msgs = patch_invalid_tool_calls(st.messages)
            # M2 (request part): evicted user messages become stubs.
            for i in range(len(req_msgs)):
                if req_msgs[i].role == ROLE_USER and req_msgs[i].evicted_to.byte_length() > 0:
                    req_msgs[i].content = human_message_stub(req_msgs[i].evicted_to, req_msgs[i].content)
            # M3
            var resp = self._model_step(st, req_msgs, tools)
            # M4
            var msg = resp.message.copy()
            msg.name = self.spec.name
            st.messages.append(msg.copy())
            # R2
            if len(msg.tool_calls) == 0 and len(msg.invalid_tool_calls) == 0:
                break
            # T0
            self._step(st)
            # T1
            for c in msg.invalid_tool_calls:
                st.messages.append(malformed_call_message(c.name, c.id))
                st.stats.tool_calls += 1
                st.stats.tool_errors += 1
            # T2
            var base = st.files.copy()
            var n = len(msg.tool_calls)
            var txs = List[VfsTx]()
            var outcomes = List[ToolOutcome]()
            var keys = List[String]()
            var tasks = List[Int]()
            var ran_subagent = List[Bool]()
            for _ in range(n):
                ran_subagent.append(False)
                txs.append(VfsTx())
                outcomes.append(text_outcome(""))
            # T3
            for k in range(n):
                var call = msg.tool_calls[k].copy()
                var key = mutation_key(call.name, call.args, routes)
                if key.byte_length() > 0 and _contains(keys, key):
                    outcomes[k] = error_outcome(ERR_PARALLEL_MUTATION)
                    keys.append(key)
                    continue
                keys.append(key)
                if call.name == "task" and _contains(node_names, "task"):
                    var extra = task_extra_key_errors(call.args)
                    if len(extra) > 0:
                        outcomes[k] = error_outcome(invocation_error_text("task", call.args, extra))
                        continue
                    var chk = validate_args(TASK_PARAMETERS, call.args)
                    if not chk.ok:
                        outcomes[k] = error_outcome(invocation_error_text("task", call.args, chk.errors))
                        continue
                    var st_type = _str_arg(chk.args, "subagent_type").value()
                    var allowed = self._subagent_names()
                    if not _contains(allowed, st_type):
                        outcomes[k] = text_outcome(unknown_subagent_text(st_type, allowed))
                        continue
                    tasks.append(k)
                    continue
                outcomes[k] = self._dispatch(st, call, tools, node_names, routes, base, txs[k])
            # T4
            if len(tasks) > 0:
                var reqs = List[SubagentRequest]()
                for k in tasks:
                    var call = msg.tool_calls[k].copy()
                    reqs.append(SubagentRequest(
                        self.spec.name, self.spec.model, _str_arg(call.args, "subagent_type").value(),
                        _str_arg(call.args, "description").value(), call.id, self._child_ancestors(),
                        base.to_json(), self.env.sandbox(), child_replay_at(self.replay_at, call.id)))
                var results = List[SubagentResult]()
                if len(reqs) >= 2 and self.env.parallel_limit() > 1:
                    results = self.env.run_subagents(reqs.copy())
                    if len(results) != len(reqs):
                        raise Error("QA-E120: run_subagents returned " + String(len(results)) + " results for "
                                    + String(len(reqs)) + " requests")
                else:
                    for r in reqs:
                        results.append(run_subagent_inline(self.env, r))
                for i in range(len(tasks)):
                    var k = tasks[i]
                    if not results[i].ok:
                        raise Error("QA-E120: subagent " + reqs[i].subagent_type + " failed: " + results[i].error)
                    merge_subagent_files(results[i], txs[k])
                    if results[i].sandbox.byte_length() > 0 and self.env.sandbox().byte_length() == 0:
                        self.env.adopt_sandbox(results[i].sandbox)
                    outcomes[k] = text_outcome(results[i].content)
                    ran_subagent[k] = True
                    st.stats.add(results[i].stats)
                    st.stats.subagent_runs += 1
            # T3.c eviction, then T5/T6
            for k in range(n):
                var call = msg.tool_calls[k].copy()
                var outcome = self._evict(call.name, call.id, outcomes[k], base, txs[k], st)
                st.stats.tool_calls += 1
                if outcome.is_error:
                    st.stats.tool_errors += 1
                outcomes[k] = outcome^
            for k in range(n):
                apply_tx(st.files, txs[k])
            for k in range(n):
                var call = msg.tool_calls[k].copy()
                # deepagents builds a subagent's result as ToolMessage(content, tool_call_id) without a name.
                var tool_name = String() if ran_subagent[k] else call.name
                st.messages.append(tool_message(call.id, tool_name, outcomes[k]))
        # R3: after_agent
        if not self.general_purpose:
            self._step(st)
        var answer = st.messages[len(st.messages) - 1].text()
        var final_text = subagent_final_text(st.messages)
        st.stats.add(self.env.take_transport_stats())
        var wall = Int((perf_counter_ns() - t0) // 1_000_000)
        if wall > st.stats.wall_ms:
            st.stats.wall_ms = wall
        return AgentResult(answer^, final_text^, st.files.copy(), st.messages.copy(), st.stats.copy())


def run_subagent_inline[E: AgentEnv](mut env: E, req: SubagentRequest) raises -> SubagentResult:
    """The single code path of a subagent run, inline or in a `qumo-agent subagent` child:
    the child sees only the description and a copy of the caller's files; it returns its
    final text and, when its file dict changed, the whole dict. Exceptions become
    `ok = False` with the error text (the caller raises QA-E120, X11)."""
    try:
        var gp = req.subagent_type == GENERAL_PURPOSE
        var spec: AgentDef
        var child_env: E
        if gp:
            var parent = AgentDef(req.parent_agent, String(), String(), req.parent_model, List[String](), 0)
            spec = general_purpose_def(parent)
            child_env = env.for_agent(req.parent_agent)
        else:
            spec = env.agent_def(req.subagent_type, req.ancestors)
            child_env = env.for_agent(req.subagent_type)
        if req.sandbox.byte_length() > 0 and child_env.sandbox().byte_length() == 0:
            child_env.adopt_sandbox(req.sandbox)
        var input = vfs_from_json(req.files_json)
        var agent = Agent[E](child_env^, spec, req.ancestors, req.replay_at, gp)
        var result = agent.run(req.description, input.copy())
        var changed = not result.files.equals(input)
        return SubagentResult(True, result.final_text, changed, result.files.to_json() if changed else String(),
                              agent.env.sandbox(), result.stats.copy(), String())
    except e:
        return SubagentResult(False, String(), False, String(), String(), zero_stats(), String(e))
