"""`AppEnv`: the application's implementation of the harness' `AgentEnv` (DESIGN.md 2.10).

One `AppShared` per process (held behind an `ArcPointer`) carries what every agent of the
delegation tree shares: the registry, the QUMO tools configuration, the repository state
(including the open sandbox), the model client or the replayed model, the QUMO MCP session,
the alphaXiv tool list, the generated tools and the transport statistics. An `AppEnv` is one
agent's view of it: its specification and its wire tool list.

Services start lazily: the QUMO MCP session when the first agent that needs MCP tools is
built, the research HTTP client at the first research call. `--replay` makes the whole run
offline: the model is the replay file and alphaXiv is not contacted.
"""

from std.memory import ArcPointer
from std.os import getenv
from std.sys import stderr

from qumo.ai.json import JsonValue

from qumo_agent.core.endpoint import ModelEndpoint
from qumo_agent.core.errors import E_CONFIG, qa_error
from qumo_agent.core.messages import ChatRequest, ChatResponse
from qumo_agent.core.replay import ReplayModel
from qumo_agent.core.stats import RunStats, zero_stats
from qumo_agent.core.toolspec import (
    ToolSpec, ToolOutcome, TOOL_LOCAL, TOOL_MCP, TOOL_ALPHAXIV, TOOL_GENERATED, TOOL_TOOLSMITH,
    text_outcome, error_outcome,
)
from qumo_agent.harness.env import (
    AgentEnv, AgentDef, FileRoutes, RelativeFileOp, ToolInvocation, SubagentRequest,
    SubagentResult, FILE_READ, FILE_WRITE,
)
from qumo_agent.harness.loop import Agent, AgentResult
from qumo_agent.harness.vfs import Vfs
from qumo_agent.llm.client import LlmClient
from qumo_agent.native.clock import unix_now, utc_iso_now
from qumo_agent.native.fs import read_text_lossy, realpath_strict, join_path
from qumo_agent.native.rand import uuid4_hex
from qumo_agent.native.sandbox import sandbox_kind
from qumo_agent.net.http import HttpClient
from qumo_agent.net.tls import TlsContext
from qumo_agent.tools.alphaxiv import AlphaXiv, alphaxiv_available
from qumo_agent.tools.catalog import file_routes_for, user_tool_specs
from qumo_agent.tools.config import (
    AgentSpecData, Registry, Modules, LocalToolsCatalog, EXPORT_HINT,
)
from qumo_agent.tools.qumo_mcp import QumoMcp, ToolGuide, default_qumo_bin
from qumo_agent.tools.repo import (
    RepoState, tool_read_file, tool_list_files, tool_search_code, tool_run_check,
    tool_write_report, tool_open_sandbox, tool_sandbox_write, tool_sandbox_edit,
)
from qumo_agent.tools.research import ResearchTools
from qumo_agent.toolsmith.generated import GeneratedTools
from qumo_agent.toolsmith.manifest import tool_store_dir
from qumo_agent.toolsmith.tools import Toolsmith, toolsmith_specs
from qumo_agent.util.pyfmt import cp_head
from qumo_agent.util.redact import redact_all, secret_env_names

from qumo_agent.app.options import AppOptions, default_options, DEFAULT_RECURSION_LIMIT, DEFAULT_PARALLEL
from qumo_agent.app.subproc import run_subagent_children

comptime LOG_ARGS_CODE_POINTS = 160
comptime LOG_PREFIX = "[qumo-agent] "
comptime ALPHAXIV_KEY_ENV = "ALPHAXIV_API_KEY"


struct AppShared(Movable):
    var opts: AppOptions
    var registry: Registry
    var modules: Modules
    var catalog: LocalToolsCatalog
    var guide: ToolGuide
    var repo: RepoState
    var tls: Optional[ArcPointer[TlsContext]]
    var llm: Optional[ArcPointer[LlmClient]]
    var replay: Optional[ArcPointer[ReplayModel]]
    var qumo_mcp: Optional[ArcPointer[QumoMcp]]
    var alphaxiv: Optional[ArcPointer[AlphaXiv]]
    var alphaxiv_specs: List[ToolSpec]
    var generated: Optional[ArcPointer[GeneratedTools]]   # one per process with every loadable tool; agents filter with specs_for
    var research: Optional[ArcPointer[ResearchTools]]
    var sandbox_kind: String                               # sandbox_kind(opts.self_binary), probed once
    var toolsmith: Toolsmith
    var transport: RunStats        # transport counters already reported by take_transport_stats
    var clock_tick: Int
    var id_counter: Int
    var tree: List[String]         # the top agent and every agent reachable through subagents
    var secrets: List[String]      # variable names whose values are redacted from logs

    def __init__(
        out self,
        var opts: AppOptions,
        var registry: Registry,
        var modules: Modules,
        var catalog: LocalToolsCatalog,
        var guide: ToolGuide,
        var repo: RepoState,
        kind: String,
        var toolsmith: Toolsmith,
        var tree: List[String],
    ):
        self.secrets = secret_env_names(opts.api_key_env)
        self.opts = opts^
        self.registry = registry^
        self.modules = modules^
        self.catalog = catalog^
        self.guide = guide^
        self.repo = repo^
        self.tls = None
        self.llm = None
        self.replay = None
        self.qumo_mcp = None
        self.alphaxiv = None
        self.alphaxiv_specs = List[ToolSpec]()
        self.generated = None
        self.research = None
        self.sandbox_kind = kind
        self.toolsmith = toolsmith^
        self.transport = zero_stats()
        self.clock_tick = 0
        self.id_counter = 0
        self.tree = tree^


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _contains(items: List[String], x: String) -> Bool:
    for i in items:
        if i == x:
            return True
    return False


def log_line(shared: ArcPointer[AppShared], msg: String):
    """One log line on stderr (redacted). Warnings and per-turn logs use it; `--quiet`
    silences it."""
    if not shared[].opts.verbose:
        return
    print(LOG_PREFIX + redact_all(msg, shared[].secrets), file=stderr, flush=True)


def warn_line(shared: ArcPointer[AppShared], msg: String):
    """A warning that `--quiet` does not silence."""
    print(LOG_PREFIX + "warning: " + redact_all(msg, shared[].secrets), file=stderr, flush=True)


def tree_of(registry: Registry, top: String) raises -> List[String]:
    """`top` and every agent reachable through subagents, in discovery order."""
    var out = List[String]()
    var queue = List[String]()
    queue.append(top)
    var i = 0
    while i < len(queue):
        var name = queue[i]
        i += 1
        if _contains(out, name):
            continue
        out.append(name)
        for s in registry.get(name).subagents:
            if not _contains(out, s):
                queue.append(s)
    return out^


def _tree_needs_sandbox_kind(registry: Registry, tree: List[String]) raises -> Bool:
    for n in tree:
        var spec = registry.get(n)
        if spec.generated_all or len(spec.generated_tools) > 0:
            return True
        for g in spec.tool_groups:
            if g == "sandbox" or g == "check" or g == "toolsmith":
                return True
    return False


def _tree_wants_generated(registry: Registry, tree: List[String]) raises -> Bool:
    for n in tree:
        var spec = registry.get(n)
        if spec.generated_all or len(spec.generated_tools) > 0:
            return True
    return False


def _tree_has_research(registry: Registry, tree: List[String]) raises -> Bool:
    for n in tree:
        if _contains(registry.get(n).tool_groups, "research"):
            return True
    return False


def ensure_tls(shared: ArcPointer[AppShared]) raises -> ArcPointer[TlsContext]:
    """The process-wide TLS context, created on first use (QA-E105 when OpenSSL or the
    trust store is missing)."""
    if shared[].tls:
        return shared[].tls.value().copy()
    var t = ArcPointer(TlsContext())
    shared[].tls = t.copy()
    return t^


def ensure_qumo_mcp(shared: ArcPointer[AppShared]) raises -> ArcPointer[QumoMcp]:
    """The QUMO MCP session of this process (X3), started on first use."""
    if shared[].qumo_mcp:
        return shared[].qumo_mcp.value().copy()
    var m = ArcPointer(QumoMcp(default_qumo_bin(shared[].opts.root), shared[].guide.copy()))
    shared[].qumo_mcp = m.copy()
    return m^


def _ensure_research(shared: ArcPointer[AppShared]) raises -> ArcPointer[ResearchTools]:
    if shared[].research:
        return shared[].research.value().copy()
    var tls: Optional[ArcPointer[TlsContext]] = None
    if shared[].tls:
        tls = shared[].tls.value().copy()
    var r = ArcPointer(ResearchTools(HttpClient(tls), shared[].modules.copy(), shared[].catalog.copy()))
    shared[].research = r.copy()
    return r^


def shutdown_shared(shared: ArcPointer[AppShared]):
    """Closes the QUMO MCP session (close stdin, SIGTERM, SIGKILL); never raises."""
    if shared[].qumo_mcp:
        try:
            var m = shared[].qumo_mcp.value().copy()
            _ = m[].close()
        except:
            pass
        shared[].qumo_mcp = None


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def _read_export(root: String, rel: String) raises -> String:
    try:
        return read_text_lossy(join_path(root, rel))
    except:
        raise qa_error(E_CONFIG, rel + " is missing or unreadable", EXPORT_HINT)


def _build_shared(var opts: AppOptions, top_agent: String, services: Bool) raises -> ArcPointer[AppShared]:
    if opts.root.byte_length() == 0:
        raise qa_error(E_CONFIG, "the repository root is unknown", "run from the repository or set QUMO_ROOT")
    var registry = Registry(opts.root)
    _ = registry.get(top_agent)                     # unknown agent: QA-E140 before anything starts
    var endpoint = registry.endpoint()
    opts.api_key_env = endpoint.api_key_env
    var modules = Modules(opts.root)
    var catalog = LocalToolsCatalog(opts.root)
    var guide = ToolGuide()
    guide.load(_read_export(opts.root, "agents/prompts/tool_guide.json"))
    var tree = tree_of(registry, top_agent)
    var kind = String("none")
    if _tree_needs_sandbox_kind(registry, tree):
        kind = sandbox_kind(opts.self_binary)
    var repo_root = realpath_strict(opts.root)
    var store = opts.tools_store if opts.tools_store.byte_length() > 0 else tool_store_dir(opts.root)
    opts.tools_store = store
    var repo = RepoState(
        repo_root, String(), catalog.skip_dirs(), catalog.max_read(), catalog.max_output(),
        catalog.run_check_allowed(), opts.self_binary, kind, opts.allow_unsandboxed_tool_tests,
    )
    var smith = Toolsmith(repo_root, store, opts.self_binary, kind, opts.allow_unsandboxed_tool_tests)
    var wants_generated = _tree_wants_generated(registry, tree)
    var has_research = _tree_has_research(registry, tree)
    var replay_path = opts.replay_file
    var allow_unreviewed = opts.allow_unreviewed_tools
    var allow_unsandboxed = opts.allow_unsandboxed_tool_tests
    var self_binary = opts.self_binary
    var shared = ArcPointer(AppShared(opts^, registry^, modules^, catalog^, guide^, repo^, kind, smith^, tree^))
    if services:
        if replay_path.byte_length() > 0:
            shared[].replay = ArcPointer(ReplayModel(replay_path))
        else:
            if endpoint.base_url.lower().startswith("https://"):
                shared[].tls = ensure_tls(shared)
            shared[].llm = ArcPointer(LlmClient(endpoint, shared[].tls))
            if has_research and alphaxiv_available(ALPHAXIV_KEY_ENV):
                try:
                    var ax = ArcPointer(AlphaXiv(ensure_tls(shared), shared[].catalog.alphaxiv_json()))
                    shared[].alphaxiv_specs = ax[].load_specs()
                    shared[].alphaxiv = ax^
                except e:
                    warn_line(shared, "alphaXiv tools are not available: " + String(e))
    if wants_generated:
        try:
            shared[].generated = ArcPointer(GeneratedTools(store, allow_unreviewed, self_binary, kind, allow_unsandboxed))
            for r in shared[].generated.value()[].refused:
                warn_line(shared, "generated tool refused: " + r)
        except e:
            warn_line(shared, "generated tools are not available: " + String(e))
    return shared^


def make_shared(var opts: AppOptions, top_agent: String) raises -> ArcPointer[AppShared]:
    """Loads the configuration and starts what the agent tree needs: the model client (or
    the replay file), alphaXiv, the generated tools. The QUMO MCP session starts with the
    first agent that has MCP tools."""
    return _build_shared(opts^, top_agent, True)


def make_shared_offline(var opts: AppOptions, top_agent: String) raises -> ArcPointer[AppShared]:
    """The same without a model: for `--dry-run`, `list` and `doctor`."""
    return _build_shared(opts^, top_agent, False)


# ---------------------------------------------------------------------------
# AppEnv
# ---------------------------------------------------------------------------
def _two(n: Int) -> String:
    return ("0" if n < 10 else "") + String(n)


comptime _HEX = "0123456789abcdef"


struct AppEnv(AgentEnv):
    var shared: ArcPointer[AppShared]
    var agent: String
    var spec: AgentSpecData
    var tools: List[ToolSpec]
    var turn: Int

    def __init__(out self, shared: ArcPointer[AppShared], agent: String, mcp: Bool = True) raises:
        """The environment of `agent`. `mcp=False` leaves the QUMO MCP tools out (so no
        server starts): for listings that must not need `build/qumo`."""
        self.shared = shared
        self.agent = agent
        self.spec = shared[].registry.get(agent)
        self.turn = 0
        var mcp_specs = List[ToolSpec]()
        if mcp and (self.spec.mcp_all or len(self.spec.mcp_tools) > 0):
            var m = ensure_qumo_mcp(shared)
            mcp_specs = m[].specs_for(self.spec.mcp_all, self.spec.mcp_tools)
        var smith = List[ToolSpec]()
        if _contains(self.spec.tool_groups, "toolsmith"):
            smith = toolsmith_specs()
        var generated = List[ToolSpec]()
        if shared[].generated:
            generated = shared[].generated.value()[].specs_for(self.spec.generated_all, self.spec.generated_tools)
        self.tools = user_tool_specs(self.spec, shared[].catalog, mcp_specs, shared[].alphaxiv_specs, smith, generated)

    def agent_def(self, name: String, ancestors: List[String]) raises -> AgentDef:
        """A subagent's definition: its own model (the `--model` flag applies to the top
        agent only) and the limit every subagent gets."""
        return self.shared[].registry.agent_def(name, ancestors, String(), 9999)

    def for_agent(self, name: String) raises -> Self:
        return AppEnv(self.shared, name)

    def user_tools(self) -> List[ToolSpec]:
        return self.tools.copy()

    def file_routes(self) -> FileRoutes:
        return file_routes_for(self.spec)

    # -- tool dispatch ------------------------------------------------------
    def _tool_kind(self, name: String) -> Int:
        for t in self.tools:
            if t.name == name:
                return t.kind
        return -1

    def _log_call(self, inv: ToolInvocation):
        log_line(self.shared, self.agent + " turn " + String(self.turn) + ": " + inv.name + " "
                 + cp_head(inv.args.to_json(), LOG_ARGS_CODE_POINTS))

    def _local(mut self, inv: ToolInvocation) raises -> ToolOutcome:
        var a = inv.args.copy()
        var name = inv.name
        if name == "search_arxiv" or name == "sdk_releases" or name == "github_release_notes" \
                or name == "module_watchlist" or name == "fetch_page":
            var r = _ensure_research(self.shared)
            if name == "search_arxiv":
                return text_outcome(r[].search_arxiv(a.get_string("query", ""), a.get_string("module", ""),
                                                     a.get_int("max_results", 10)))
            if name == "sdk_releases":
                return text_outcome(r[].sdk_releases(a.get_string("package", "")))
            if name == "github_release_notes":
                return text_outcome(r[].github_release_notes(a.get_string("repo", ""), a.get_int("count", 3)))
            if name == "module_watchlist":
                return text_outcome(r[].module_watchlist(a.get_string("module", "")))
            return text_outcome(r[].fetch_page(a.get_string("url", "")))
        if name == "list_files":
            return text_outcome(tool_list_files(self.shared[].repo, a.get_string("path", "."), a.get_string("pattern", "*")))
        if name == "search_code":
            return text_outcome(tool_search_code(self.shared[].repo, a.get_string("regex", ""),
                                                 a.get_string("path", "."), a.get_string("glob", "*.mojo")))
        if name == "run_check":
            var argv = List[String]()
            var av = a.get("argv")
            for i in range(len(av)):
                argv.append(av.at(i).as_string())
            return text_outcome(tool_run_check(self.shared[].repo, argv, a.get_int("timeout_s", 600)))
        if name == "write_report":
            return text_outcome(tool_write_report(self.shared[].repo, a.get_string("name", ""), a.get_string("markdown", "")))
        if name == "open_sandbox":
            return text_outcome(tool_open_sandbox(self.shared[].repo, a.get_string("topic", "")))
        return error_outcome("error: unknown tool " + name)

    def _toolsmith(mut self, inv: ToolInvocation) raises -> ToolOutcome:
        var a = inv.args.copy()
        var sandbox = self.shared[].repo.sandbox
        var name = a.get_string("name", "")
        if inv.name == "toolsmith_new":
            return text_outcome(self.shared[].toolsmith.new_tool(sandbox, name, a.get_string("description", ""), a.get_string("source", ""),
                                               a.get_string("cases", ""), a.get_bool("overwrite", False)))
        if inv.name == "toolsmith_build":
            return text_outcome(self.shared[].toolsmith.build(sandbox, name, a.get_int("timeout_s", 600)))
        if inv.name == "toolsmith_test":
            return text_outcome(self.shared[].toolsmith.test(sandbox, name, a.get_int("timeout_s", 30)))
        if inv.name == "toolsmith_publish":
            return text_outcome(self.shared[].toolsmith.publish(sandbox, name, a.get_string("version", "0.1.0")))
        return error_outcome("error: unknown tool " + inv.name)

    def call_tool(mut self, inv: ToolInvocation) raises -> ToolOutcome:
        self._log_call(inv)
        var kind = self._tool_kind(inv.name)
        if kind == TOOL_LOCAL:
            return self._local(inv)
        if kind == TOOL_TOOLSMITH:
            return self._toolsmith(inv)
        if kind == TOOL_MCP:
            var m = ensure_qumo_mcp(self.shared)
            return m[].call(inv.name, inv.args.to_json())
        if kind == TOOL_ALPHAXIV:
            if not self.shared[].alphaxiv:
                return error_outcome("error: alphaXiv is not available in this process")
            var ax = self.shared[].alphaxiv.value().copy()
            return ax[].call(inv.name, inv.args.to_json())
        if kind == TOOL_GENERATED:
            if not self.shared[].generated:
                return error_outcome("error: generated tools are not loaded")
            try:
                var g = self.shared[].generated.value().copy()
                return g[].call(inv.name, inv.args.to_json())
            except e:
                var t = redact_all(String(e), self.shared[].secrets)
                if t.startswith("error: ") or t.startswith("refused: "):
                    return error_outcome(t)
                return error_outcome("error: " + t)
        return error_outcome("error: unknown tool " + inv.name)

    def relative_file_op(mut self, op: RelativeFileOp) raises -> ToolOutcome:
        if op.op == FILE_READ:
            return text_outcome(tool_read_file(self.shared[].repo, op.path, op.start_line, op.max_lines))
        if op.op == FILE_WRITE:
            return text_outcome(tool_sandbox_write(self.shared[].repo, op.path, op.content))
        return text_outcome(tool_sandbox_edit(self.shared[].repo, op.path, op.old, op.new, op.replace_all))

    # -- model ----------------------------------------------------------------
    def complete(mut self, req: ChatRequest) raises -> ChatResponse:
        if len(req.tools) > 0:
            self.turn += 1
        if self.shared[].replay:
            var rp = self.shared[].replay.value().copy()
            return rp[].complete(req)
        if self.shared[].llm:
            var c = self.shared[].llm.value().copy()
            return c[].complete(req)
        raise qa_error(E_CONFIG, "no model is configured in this process", "pass --replay FILE or set the model key")

    # -- subagents ------------------------------------------------------------
    def parallel_limit(self) -> Int:
        return self.shared[].opts.parallel

    def run_subagents(mut self, var reqs: List[SubagentRequest]) raises -> List[SubagentResult]:
        return run_subagent_children(self.shared[].opts, reqs^, self.shared[].opts.parallel)

    def sandbox(self) -> String:
        return self.shared[].repo.sandbox

    def adopt_sandbox(mut self, path: String):
        self.shared[].repo.sandbox = path

    # -- clock and ids ----------------------------------------------------------
    def now_utc_iso(mut self) -> String:
        if self.shared[].opts.deterministic:
            var s = self.shared[].clock_tick
            self.shared[].clock_tick += 1
            return "2026-01-01T00:" + _two((s // 60) % 60) + ":" + _two(s % 60) + "+00:00"
        return utc_iso_now()

    def new_uuid4_hex(mut self) raises -> String:
        if self.shared[].opts.deterministic:
            self.shared[].id_counter += 1
            var v = self.shared[].id_counter
            var out = String()
            var hx = _HEX.as_bytes()
            for i in range(31, -1, -1):
                var nib = (v >> (4 * i)) & 15 if i < 16 else 0
                out += String(unsafe_from_utf8=hx[nib:nib + 1])
            return out^
        return uuid4_hex()

    def log(self, msg: String):
        log_line(self.shared, self.agent + ": " + msg)

    def take_transport_stats(mut self) -> RunStats:
        """HTTP counters since the previous call (the model client's own)."""
        var delta = zero_stats()
        if self.shared[].llm:
            var c = self.shared[].llm.value().copy()
            var now = c[].stats.copy()
            delta.http_requests = now.http_requests - self.shared[].transport.http_requests
            delta.http_retries = now.http_retries - self.shared[].transport.http_retries
            self.shared[].transport.http_requests = now.http_requests
            self.shared[].transport.http_retries = now.http_retries
        return delta^


# ---------------------------------------------------------------------------
# one top-level run
# ---------------------------------------------------------------------------
def run_agent_once(shared: ArcPointer[AppShared], agent: String, task: String) raises -> AgentResult:
    """A fresh run of `agent` (new messages and file system) in this process: what `run`,
    each `serve` call and the tests use."""
    var env = AppEnv(shared, agent)
    var opts = shared[].opts.copy()
    var spec = shared[].registry.agent_def(agent, List[String](), opts.model_override, opts.recursion_limit)
    var a = Agent[AppEnv](env^, spec, List[String](), "", False)
    return a.run(task, Vfs())
