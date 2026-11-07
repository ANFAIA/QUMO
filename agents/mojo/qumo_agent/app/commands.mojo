"""The `qumo-agent` command line (DESIGN.md 7): parsing, usage, exit codes and the commands.

`main_entry(argv) -> Int` is the only entry point `main.mojo` needs:

    def main():
        exit(main_entry(args))

It dispatches the two trampolines first (`__confine`, `__exec-in`: they must run before any
other work), then ignores SIGPIPE, installs the shutdown handlers (SIGINT, SIGTERM and SIGHUP end
every child process group, native/shutdown.mojo) and runs the command. Exit codes (7.3): 0 success, 1 the
run failed, 2 usage error, 3 configuration error (QA-E140, QA-E105).

Commands: list, run, serve, call, tools, doctor, reports, help, and the internal
`subagent` (the child side of parallel `task` calls, B.7).
"""

from std.os import getenv, stat
from std.sys import stderr

from qumo.ai.json import JsonWriter, parse_json
from qumo.ai.mcp import LineReader

from qumo_agent.app.doctor import run_doctor
from qumo_agent.app.env import (
    AppEnv, AppOptions, make_shared, make_shared_offline, run_agent_once, shutdown_shared,
)
from qumo_agent.app.options import DEFAULT_PARALLEL, DEFAULT_RECURSION_LIMIT
from qumo_agent.app.server import AgentMcpHandler
from qumo_agent.app.tools_cmd import run_tools_command
from qumo_agent.core.errors import error_code
from qumo_agent.core.stats import zero_stats
from qumo_agent.harness.env import SubagentResult, subagent_request_from_json
from qumo_agent.harness.loop import Agent, run_subagent_inline
from qumo_agent.harness.subagents import GENERAL_PURPOSE
from qumo_agent.mcp.server import serve_stdio
from qumo_agent.mcp.stdio_client import McpStdioSession, StdioServerParams, redact_mcp_text
from qumo_agent.native.clock import monotonic_ms, unix_now
from qumo_agent.native.fs import (
    is_dir, join_path, list_dir, make_dirs, read_text_lossy, realpath_strict, resolve_nonstrict, write_text_atomic,
)
from qumo_agent.native.libc import is_macos
from qumo_agent.native.sandbox import Confinement, confine_main
from qumo_agent.native.shutdown import install_shutdown_handlers
from qumo_agent.native.spawn import (
    STDIO_INHERIT, exec_in_main, harness_child_env, ignore_sigpipe, set_chdir_trampoline, which,
)
from qumo_agent.tools.alphaxiv import alphaxiv_available
from qumo_agent.tools.catalog import toolsmith_tool_names
from qumo_agent.tools.config import AgentSpecData, LocalToolsCatalog, Modules, Registry, find_root
from qumo_agent.toolsmith.manifest import load_manifest, tool_store_dir
from qumo_agent.util.pyfmt import py_json_dumps, py_str_repr


comptime USAGE = """usage:
  qumo-agent list [--json]
  qumo-agent run   AGENT TASK [--module M] [--model ID] [--replay FILE] [--recursion-limit N]
                              [--parallel N] [--allow-unreviewed-tools] [--tools-dir DIR]
                              [--allow-unsandboxed-tool-tests]
                              [--quiet] [--stats-json FILE] [--dry-run]
  qumo-agent serve AGENT [--model ID] [--replay FILE] [--recursion-limit N] [--parallel N]
                         [--allow-unreviewed-tools] [--allow-unsandboxed-tool-tests] [--tools-dir DIR]
  qumo-agent call  AGENT TASK [same options as run, except --dry-run and --stats-json]
  qumo-agent tools list | show NAME | review NAME [--yes] | revoke NAME | remove NAME | verify [NAME]
                   [--tools-dir DIR] [--allow-unsandboxed-tool-tests]
  qumo-agent doctor [--offline] [--replay FILE]
  qumo-agent reports [--show TEXT] [--limit N]
  qumo-agent help
TASK `-` reads the task from stdin. The answer goes to stdout, statistics and logs to stderr.
Exit codes: 0 ok, 1 the run failed, 2 usage error, 3 configuration error."""

def _value_options() -> List[String]:
    return ["--module", "--model", "--replay", "--recursion-limit", "--parallel", "--tools-dir",
            "--stats-json", "--show", "--limit"]


def _flag_options() -> List[String]:
    return ["--allow-unreviewed-tools", "--allow-unsandboxed-tool-tests", "--quiet", "-q", "--dry-run",
            "--json", "--yes", "--offline"]


def _run_options() -> List[String]:
    return ["--module", "--model", "--replay", "--recursion-limit", "--parallel", "--allow-unreviewed-tools",
            "--tools-dir", "--allow-unsandboxed-tool-tests", "--quiet", "-q"]


struct Cli(Movable):
    var command: String
    var positional: List[String]
    var module: String
    var model: String
    var replay: String
    var recursion_limit: Int
    var parallel: Int
    var tools_dir: String
    var stats_json: String
    var show: String
    var limit: Int
    var allow_unreviewed: Bool
    var allow_unsandboxed: Bool
    var quiet: Bool
    var dry_run: Bool
    var json: Bool
    var yes: Bool
    var offline: Bool

    def __init__(out self):
        self.command = String()
        self.positional = List[String]()
        self.module = String()
        self.model = String()
        self.replay = String()
        self.recursion_limit = DEFAULT_RECURSION_LIMIT
        self.parallel = DEFAULT_PARALLEL
        self.tools_dir = String()
        self.stats_json = String()
        self.show = String()
        self.limit = 20
        self.allow_unreviewed = False
        self.allow_unsandboxed = False
        self.quiet = False
        self.dry_run = False
        self.json = False
        self.yes = False
        self.offline = False


def _contains(items: List[String], x: String) -> Bool:
    for i in items:
        if i == x:
            return True
    return False


def _allowed_options(command: String) -> List[String]:
    """The options a command accepts (everything else is a usage error)."""
    if command == "list":
        return ["--json"]
    if command == "run":
        var o = _run_options()
        o.append("--stats-json")
        o.append("--dry-run")
        return o^
    if command == "call":
        return _run_options()
    if command == "serve":
        var o = _run_options()
        var out = List[String]()
        for x in o:
            if x != "--module":
                out.append(x)
        return out^
    if command == "subagent":
        return ["--replay", "--tools-dir", "--allow-unreviewed-tools", "--allow-unsandboxed-tool-tests",
                "--parallel", "--quiet", "-q"]
    if command == "tools":
        return ["--tools-dir", "--allow-unsandboxed-tool-tests", "--yes"]
    if command == "doctor":
        return ["--offline", "--replay", "--tools-dir"]
    if command == "reports":
        return ["--show", "--limit"]
    return List[String]()


def _positive(value: String, name: String, mut out: Int) -> String:
    try:
        var v = Int(value)
        if v < 1:
            return name + " must be at least 1"
        out = v
        return String()
    except:
        return name + " needs a whole number, got " + py_str_repr(value)


def parse_cli(args: List[String], mut cli: Cli) -> String:
    """Fills `cli` from `args` (args[0] is the program); returns an error message
    ("" when fine). `--name=value` and `--name value` are both accepted."""
    if len(args) < 2:
        return "missing command"
    cli.command = args[1]
    var allowed = _allowed_options(cli.command)
    var i = 2
    var options_done = False
    while i < len(args):
        var a = args[i]
        i += 1
        if options_done or not a.startswith("-") or a == "-":
            cli.positional.append(a)
            continue
        if a == "--":
            options_done = True
            continue
        var name = a
        var value = String()
        var has_value = False
        var eq = a.find("=")
        if a.startswith("--") and eq > 0:
            name = String(a[byte=0:eq])
            value = String(a[byte=eq + 1:])
            has_value = True
        if not _contains(_value_options(), name) and not _contains(_flag_options(), name):
            return "unknown option " + a
        if not _contains(allowed, name):
            return "option " + name + " does not apply to " + cli.command
        if _contains(_value_options(), name):
            if not has_value:
                if i >= len(args):
                    return name + " needs a value"
                value = args[i]
                i += 1
            if name == "--module":
                cli.module = value
            elif name == "--model":
                cli.model = value
            elif name == "--replay":
                cli.replay = value
            elif name == "--tools-dir":
                cli.tools_dir = value
            elif name == "--stats-json":
                cli.stats_json = value
            elif name == "--show":
                cli.show = value
            elif name == "--recursion-limit":
                var err = _positive(value, name, cli.recursion_limit)
                if err.byte_length() > 0:
                    return err
            elif name == "--parallel":
                var err = _positive(value, name, cli.parallel)
                if err.byte_length() > 0:
                    return err
            elif name == "--limit":
                try:
                    cli.limit = Int(value)
                except:
                    return "--limit needs a whole number, got " + py_str_repr(value)
        else:
            if has_value:
                return name + " takes no value"
            if name == "--allow-unreviewed-tools":
                cli.allow_unreviewed = True
            elif name == "--allow-unsandboxed-tool-tests":
                cli.allow_unsandboxed = True
            elif name == "--quiet" or name == "-q":
                cli.quiet = True
            elif name == "--dry-run":
                cli.dry_run = True
            elif name == "--json":
                cli.json = True
            elif name == "--yes":
                cli.yes = True
            elif name == "--offline":
                cli.offline = True
    return String()


def usage_error(msg: String) -> Int:
    if msg.byte_length() > 0:
        print("qumo-agent: " + msg, file=stderr)
    print(USAGE, file=stderr)
    return 2


def report_error(e: Error) -> Int:
    """Prints `qumo-agent: error: <message>` (secrets redacted) and maps the error to an
    exit code: 3 for configuration (QA-E140, QA-E105), 1 for everything else."""
    var text = redact_mcp_text(String(e))
    print("qumo-agent: error: " + text, file=stderr)
    var code = error_code(e)
    if code == "QA-E140" or code == "QA-E105":
        return 3
    return 1


# ---------------------------------------------------------------------------
# common setup
# ---------------------------------------------------------------------------
def resolve_self_binary(argv0: String, root: String) -> String:
    """This executable, as an absolute path: $QUMO_AGENT_BIN, else argv[0] when it is a
    path (or found in PATH), else <root>/build/qumo-agent."""
    var env = getenv("QUMO_AGENT_BIN", "")
    if env.byte_length() > 0:
        return env
    if argv0.find("/") >= 0:
        return resolve_nonstrict(argv0)
    var found = which(argv0, getenv("PATH", ""))
    if found.byte_length() > 0:
        return resolve_nonstrict(found)
    return root + "/build/qumo-agent"


def make_options(cli: Cli, root: String, self_binary: String) -> AppOptions:
    return AppOptions(
        root, cli.model, cli.replay, cli.recursion_limit, cli.parallel, cli.allow_unreviewed,
        cli.allow_unsandboxed, cli.tools_dir, not cli.quiet, self_binary,
        getenv("QUMO_AGENT_DETERMINISTIC", "") == "1", String(),
    )


def read_task(cli: Cli, root: String) raises -> String:
    """TASK (`-` reads stdin), prefixed by `--module`. Raises a message starting with
    "usage:" for an empty task or an unknown module (exit code 2)."""
    var t = cli.positional[1]
    if t == "-":
        var r = LineReader(0)
        t = r.read_all()
    if String(t.strip()).byte_length() == 0:
        raise Error("usage: the task is empty")
    if cli.module.byte_length() > 0:
        try:
            t = Modules(root).task_prefix(cli.module) + t
        except e:
            if error_code(e) == "QA-E140":
                raise e^
            raise Error("usage: " + String(e))
    return t^


def _ljust(s: String, width: Int) -> String:
    var out = s
    while out.byte_length() < width:
        out += " "
    return out^


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
def cmd_list(cli: Cli, opts: AppOptions) raises -> Int:
    var registry = Registry(opts.root)
    var modules = Modules(opts.root)
    var store = opts.tools_store if opts.tools_store.byte_length() > 0 else tool_store_dir(opts.root)
    var reviewed = 0
    var total = 0
    try:
        var m = load_manifest(store)
        total = len(m.entries)
        for e in m.entries:
            if e.reviewed:
                reviewed += 1
    except:
        pass
    var ax = alphaxiv_available("ALPHAXIV_API_KEY")
    if cli.json:
        var w = JsonWriter()
        w.begin_object()
        w.key("agents")
        w.begin_array()
        for name in registry.names():
            var spec = registry.get(name)
            w.begin_object()
            w.kv_string("name", name)
            w.kv_string("model", registry.model_for(name, String()))
            w.kv_string("description", spec.description)
            w.kv_string_list("subagents", spec.subagents)
            w.kv_bool("mojo_only", spec.mojo_only)
            w.end_object()
        w.end_array()
        w.kv_string_list("modules", modules.names())
        w.kv_bool("alphaxiv", ax)
        w.key("generated_tools")
        w.begin_object()
        w.kv_int("reviewed", reviewed)
        w.kv_int("total", total)
        w.end_object()
        w.end_object()
        print(py_json_dumps(parse_json(w.finish()), True, 1))
        return 0
    for name in registry.names():
        var spec = registry.get(name)
        var subs = String()
        if len(spec.subagents) > 0:
            subs = " -> " + String(", ").join(spec.subagents)
        var tag = String(" [mojo-only]") if spec.mojo_only else String()
        print(_ljust(name, 20) + " " + _ljust(registry.model_for(name, String()), 32) + " " + spec.description + subs + tag)
    print("")
    print("modules: " + String(", ").join(modules.names()))
    print("alphaXiv: " + (String("on") if ax else String("off (set ALPHAXIV_API_KEY; arXiv fallback)")))
    print("generated tools: " + String(reviewed) + "/" + String(total) + " reviewed")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def _local_tool_names(catalog: LocalToolsCatalog, spec: AgentSpecData) raises -> List[String]:
    """Python's `[t.name for t in registry.local_tools(spec)]`: groups in spec order, each
    group's tools in tools.json order, first occurrence kept (read_file, write_file and
    edit_file included: the harness merges them later)."""
    var out = List[String]()
    for g in spec.tool_groups:
        var names: List[String]
        if g == "toolsmith":
            names = toolsmith_tool_names()
        else:
            names = catalog.group(g)
        for n in names:
            if not _contains(out, n):
                out.append(n)
    return out^


def cmd_dry_run(cli: Cli, opts: AppOptions, agent: String) raises -> Int:
    var shared = make_shared_offline(opts.copy(), agent)
    var registry = shared[].registry.copy()
    var spec = registry.get(agent)
    var env: AppEnv
    try:
        env = AppEnv(shared, agent)
    except e:
        if error_code(e) != "QA-E140":
            raise e^
        print("[qumo-agent] warning: " + String(e) + "; wire_tools leaves the QUMO MCP tools out", file=stderr)
        env = AppEnv(shared, agent, False)
    var def_ = registry.agent_def(agent, List[String](), cli.model, cli.recursion_limit)
    var generated = List[String]()
    if shared[].generated:
        for t in shared[].generated.value()[].specs_for(spec.generated_all, spec.generated_tools):
            generated.append(t.name)
    var a = Agent[AppEnv](env^, def_, List[String](), "", False)
    var wire = List[String]()
    for t in a.wire_tools():
        wire.append(t.name)
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("agent", agent)
    w.kv_string("model", def_.model)
    w.kv_string_list("local_tools", _local_tool_names(shared[].catalog, spec))
    if spec.mcp_all:
        w.kv_string("qumo_mcp", "all")
    else:
        w.kv_string_list("qumo_mcp", spec.mcp_tools)
    w.kv_bool("alphaxiv", _contains(spec.tool_groups, "research") and alphaxiv_available("ALPHAXIV_API_KEY"))
    w.kv_string_list("subagents", spec.subagents)
    w.kv_string_list("wire_tools", wire)
    w.kv_string_list("generated_tools", generated)
    w.end_object()
    print(py_json_dumps(parse_json(w.finish()), True, 1))
    shutdown_shared(shared)
    return 0


def cmd_run(cli: Cli, opts: AppOptions) raises -> Int:
    var agent = cli.positional[0]
    if cli.dry_run:
        _ = read_task(cli, opts.root)
        return cmd_dry_run(cli, opts, agent)
    var task = read_task(cli, opts.root)
    var shared = make_shared(opts.copy(), agent)
    var model = shared[].registry.model_for(agent, cli.model)
    try:
        var r = run_agent_once(shared, agent, task)
        print(r.answer)
        print("[qumo-agent] " + agent + " model=" + model + " " + String(r.stats), file=stderr)
        if cli.stats_json.byte_length() > 0:
            var w = JsonWriter()
            w.begin_object()
            w.kv_string("agent", agent)
            w.kv_string("model", model)
            w.kv_raw("stats", r.stats.to_json())
            w.end_object()
            write_text_atomic(cli.stats_json, w.finish() + "\n")
    except e:
        shutdown_shared(shared)
        raise e^
    shutdown_shared(shared)
    return 0


# ---------------------------------------------------------------------------
# serve and call
# ---------------------------------------------------------------------------
def cmd_serve(cli: Cli, opts: AppOptions) raises -> Int:
    var handler = AgentMcpHandler(opts.copy(), cli.positional[0])
    try:
        serve_stdio(handler)
    except e:
        handler.close()
        raise e^
    handler.close()
    return 0


def cmd_call(cli: Cli, opts: AppOptions) raises -> Int:
    var agent = cli.positional[0]
    var registry = Registry(opts.root)
    _ = registry.get(agent)                      # refuse an unknown agent before spawning
    var api_key_env = registry.endpoint().api_key_env
    var task = read_task(cli, opts.root)
    var t0 = monotonic_ms()
    var args: List[String] = ["serve", agent, "--recursion-limit", String(cli.recursion_limit)]
    if cli.model.byte_length() > 0:
        args.append("--model")
        args.append(cli.model)
    for f in opts.child_flags():
        args.append(f)
    var params = StdioServerParams(opts.self_binary, args^, harness_child_env(api_key_env), String(), 0,
                                   268435456, STDIO_INHERIT, Optional[Confinement](None), String())
    var session = McpStdioSession(params)
    var text: String
    var is_error: Bool
    var stats = String()
    var server: String
    try:
        session.initialize("qumo-agent call", "0.2")
        var tools = session.list_tools()
        if len(tools) != 1 or tools[0].name != "run":
            raise Error("QA-E130: the child server does not expose the run tool")
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("task", task)
        w.end_object()
        var r = session.call_tool("run", w.finish())
        text = String("\n").join(r.parts)
        is_error = r.is_error
        if r.structured_json.byte_length() > 0:
            stats = parse_json(r.structured_json).get("stats").to_json()
        server = session.server_name
    except e:
        try:
            _ = session.close()
        except:
            pass
        raise e^
    var code = session.close()
    var wall = Float64(monotonic_ms() - t0) / 1000.0
    print(text)
    print("[qumo-agent call] " + server + " exit=" + String(code) + " wall_s=" + String(wall) + " child_stats=" + stats, file=stderr)
    return 1 if is_error else 0


# ---------------------------------------------------------------------------
# subagent (child side of parallel task calls)
# ---------------------------------------------------------------------------
def _window_stamp() -> String:
    var t = unix_now()
    return String(t[0]) + " " + String(t[1])


def _log_window(replay_at: String, started: String):
    """Test aid: with $QUMO_AGENT_REPLAY_LOG set, records the child's wall-clock window
    (`<sec> <usec>` at start and at end) in `<dir>/child-<replay_at>.window`, so a test
    can see that two children ran at the same time."""
    var dir = getenv("QUMO_AGENT_REPLAY_LOG", "")
    if dir.byte_length() == 0:
        return
    try:
        make_dirs(dir)
        write_text_atomic(join_path(dir, "child-" + replay_at.replace("/", "__") + ".window"),
                          started + "\n" + _window_stamp() + "\n")
    except:
        pass


def cmd_subagent(cli: Cli, opts: AppOptions) raises -> Int:
    """One request line on stdin, one result line on stdout (B.7). The whole stdin is
    read before anything is written."""
    var started = _window_stamp()
    var replay_at = String()
    var result: SubagentResult
    try:
        var reader = LineReader(0)
        var line = reader.read_all()
        var req = subagent_request_from_json(line)
        replay_at = req.replay_at
        var base = req.parent_agent if req.subagent_type == GENERAL_PURPOSE else req.subagent_type
        var shared = make_shared(opts.copy(), base)
        try:
            var env = AppEnv(shared, base)
            result = run_subagent_inline(env, req)
        except e:
            shutdown_shared(shared)
            raise e^
        shutdown_shared(shared)
    except e:
        result = SubagentResult(False, String(), False, String(), String(), zero_stats(), redact_mcp_text(String(e)))
    _log_window(replay_at, started)
    print(result.to_json(), flush=True)
    return 0 if result.ok else 1


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def _sorted_desc(mut names: List[String]):
    """Insertion sort, descending (names start with the date, so newest first)."""
    for i in range(1, len(names)):
        var j = i
        while j > 0 and names[j - 1] < names[j]:
            var tmp = names[j - 1]
            names[j - 1] = names[j]
            names[j] = tmp^
            j -= 1


def cmd_reports(cli: Cli, opts: AppOptions) raises -> Int:
    var dir = join_path(opts.root, "agents/reports")
    var files = List[String]()
    if is_dir(dir):
        for n in list_dir(dir):
            if n.endswith(".md"):
                files.append(n)
    _sorted_desc(files)
    if cli.show.byte_length() > 0:
        for n in files:
            if n.find(cli.show) >= 0:
                print(read_text_lossy(join_path(dir, n)))
                return 0
        print("no report matches " + py_str_repr(cli.show), file=stderr)
        return 1
    var shown = 0
    for n in files:
        if shown >= cli.limit:
            break
        shown += 1
        var size = stat(join_path(dir, n)).st_size
        var s = String(size)
        while s.byte_length() < 7:
            s = " " + s
        print(s + "  agents/reports/" + n)
    return 0


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------
def _dispatch(args: List[String]) raises -> Int:
    var cli = Cli()
    var err = parse_cli(args, cli)
    if cli.command == "help" or cli.command == "--help" or cli.command == "-h":
        print(USAGE)
        return 0
    if err.byte_length() > 0:
        return usage_error(err)
    var command = cli.command
    if command != "list" and command != "run" and command != "serve" and command != "call" and command != "tools" \
            and command != "doctor" and command != "reports" and command != "subagent":
        return usage_error("unknown command " + command)
    # Arity checks before any work.
    if command == "run" or command == "call":
        if len(cli.positional) != 2:
            return usage_error(command + " needs AGENT and TASK")
    elif command == "serve" or command == "subagent":
        if command == "serve" and len(cli.positional) != 1:
            return usage_error("serve needs AGENT")
        if command == "subagent" and len(cli.positional) != 0:
            return usage_error("subagent takes no argument")
    elif command == "list" or command == "doctor" or command == "reports":
        if len(cli.positional) != 0:
            return usage_error(command + " takes no argument")
    var root: String
    try:
        root = realpath_strict(find_root(args[0]))
    except e:
        if command != "doctor":
            return report_error(e)
        root = String()
    var self_binary = resolve_self_binary(args[0], root)
    set_chdir_trampoline(self_binary)
    var opts = make_options(cli, root, self_binary)
    try:
        if command == "list":
            return cmd_list(cli, opts)
        if command == "run":
            return cmd_run(cli, opts)
        if command == "serve":
            return cmd_serve(cli, opts)
        if command == "call":
            return cmd_call(cli, opts)
        if command == "tools":
            return run_tools_command(opts, cli.positional, cli.yes)
        if command == "doctor":
            return run_doctor(opts, cli.offline)
        if command == "reports":
            return cmd_reports(cli, opts)
        return cmd_subagent(cli, opts)
    except e:
        var text = String(e)
        if text.startswith("usage: "):
            return usage_error(String(text[byte=7:]))
        return report_error(e)


def main_entry(args: List[String]) -> Int:
    """Runs the command in `args` (args[0] is the program) and returns the exit code."""
    if len(args) >= 2 and args[1] == "__confine":
        return confine_main(args)
    if len(args) >= 2 and args[1] == "__exec-in":
        return exec_in_main(args)
    ignore_sigpipe()
    install_shutdown_handlers()
    try:
        return _dispatch(args)
    except e:
        return report_error(e)


def run_cli(args: List[String]) -> Int:
    """Alias of `main_entry`."""
    return main_entry(args)
