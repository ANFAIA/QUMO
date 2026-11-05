"""The qumo-agent launcher: QUMO agents as native executables (no Python at run time).

Build: `make agents-mojo` (= mojo build -I . -I agents/mojo -o build/qumo-agent agents/mojo/main.mojo).

    qumo-agent list
    qumo-agent run   AGENT "TASK" [--model ID] [--replay FILE] [--max-turns N] [--max-tokens N] [--quiet]
    qumo-agent serve AGENT        [--model ID] [--replay FILE] [--max-turns N] [--max-tokens N]
    qumo-agent call  AGENT "TASK" [--model ID] [--replay FILE] [--max-turns N] [--max-tokens N]

`run` runs the agent in this process. `serve` makes it an MCP stdio server with one
tool, `run` {task}. `call` starts `qumo-agent serve AGENT` as a child process over
pipes and calls its `run` tool through an MCP client: executable to executable, like
a library call. TASK `-` reads the task from stdin.

Agents: qumo-operator and execution. The others need tools that exist only in the
Python layer; run them with bin/qumo-agents. The answer goes to stdout, statistics
and logs to stderr. Exit code 0 ok, 1 the run failed, 2 usage error.
"""

from std.sys import argv, exit, stderr
from std.os import getenv
from std.time import perf_counter_ns

from qumo.ai.json import JsonWriter, parse_json
from qumo.ai.mcp import LineReader

from qumo_agent.config import AgentConfig, find_root, describe_agents
from qumo_agent.llm import LlmClient
from qumo_agent.agent import Agent, DEFAULT_MAX_TURNS, DEFAULT_MAX_TOKENS
from qumo_agent.server import AgentServer
from qumo_agent.mcp_client import McpClient

comptime USAGE = """usage:
  qumo-agent list
  qumo-agent run   AGENT "TASK" [--model ID] [--replay FILE] [--max-turns N] [--max-tokens N] [--quiet]
  qumo-agent serve AGENT        [--model ID] [--replay FILE] [--max-turns N] [--max-tokens N]
  qumo-agent call  AGENT "TASK" [--model ID] [--replay FILE] [--max-turns N] [--max-tokens N]
Mojo agents: qumo-operator, execution (the others: bin/qumo-agents)."""


struct Opts(Movable):
    var command: String
    var positional: List[String]
    var model: String
    var replay: String
    var max_turns: Int
    var max_tokens: Int
    var quiet: Bool

    def __init__(out self):
        self.command = String()
        self.positional = List[String]()
        self.model = String()
        self.replay = String()
        self.max_turns = DEFAULT_MAX_TURNS
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.quiet = False


def _usage(msg: String) -> Int:
    if msg.byte_length() > 0:
        print("qumo-agent: " + msg, file=stderr)
    print(USAGE, file=stderr)
    return 2


def parse_opts(args: List[String], mut o: Opts) raises -> String:
    """Fills `o`; returns an error message ("" when fine)."""
    if len(args) < 2:
        return "missing command"
    o.command = args[1]
    var i = 2
    while i < len(args):
        var a = args[i]
        if a == "--quiet" or a == "-q":
            o.quiet = True
        elif a == "--model" or a == "--replay" or a == "--max-turns" or a == "--max-tokens":
            if i + 1 >= len(args):
                return a + " needs a value"
            var v = args[i + 1]
            i += 1
            if a == "--model":
                o.model = v
            elif a == "--replay":
                o.replay = v
            elif a == "--max-turns":
                o.max_turns = Int(v)
                if o.max_turns < 1:
                    return "--max-turns must be at least 1"
            else:
                o.max_tokens = Int(v)
        elif a.startswith("--"):
            return "unknown option " + a
        else:
            o.positional.append(a)
        i += 1
    return String()


def _task(o: Opts) raises -> String:
    var t = o.positional[1]
    if t == "-":
        var r = LineReader(0)
        t = r.read_all()
    if String(t.strip()).byte_length() == 0:
        raise Error("the task is empty")
    return t^


def cmd_run(root: String, o: Opts) raises -> Int:
    var cfg = AgentConfig(root, o.positional[0], o.model)
    var model = cfg.model
    var llm: LlmClient
    if o.replay.byte_length() > 0:
        llm = LlmClient(replay_file=o.replay)
    else:
        llm = LlmClient(cfg.base_url, cfg.api_key_env, cfg.header_names, cfg.header_values, cfg.timeout_s, cfg.max_retries)
    var agent = Agent(cfg^, llm^, o.max_turns, o.max_tokens, not o.quiet)
    var r = agent.run(_task(o))
    agent.close()
    print(r.answer)
    print("[qumo-agent] " + o.positional[0] + " model=" + model + " " + String(r.stats), file=stderr)
    return 1 if r.stats.stopped_at_limit else 0


def cmd_serve(root: String, o: Opts) raises -> Int:
    var s = AgentServer(root, o.positional[0], o.model, o.replay, o.max_turns, o.max_tokens)
    s.serve_stdio()
    return 0


def self_path(root: String) -> String:
    """This executable, to start it again as `serve`: QUMO_AGENT_BIN, else argv[0]
    when it is a path, else <root>/build/qumo-agent."""
    var env = getenv("QUMO_AGENT_BIN", "")
    if env.byte_length() > 0:
        return env
    var a0 = String(argv()[0])
    if a0.find("/") >= 0:
        return a0
    return root + "/build/qumo-agent"


def cmd_call(root: String, o: Opts) raises -> Int:
    var agent = o.positional[0]
    _ = AgentConfig(root, agent, o.model)  # refuse Python-only agents before spawning
    var task = _task(o)
    var t0 = perf_counter_ns()
    var args: List[String] = ["serve", agent, "--max-turns", String(o.max_turns), "--max-tokens", String(o.max_tokens)]
    if o.model.byte_length() > 0:
        args.append("--model")
        args.append(o.model)
    if o.replay.byte_length() > 0:
        args.append("--replay")
        args.append(o.replay)
    var c = McpClient(self_path(root), args)
    c.initialize("qumo-agent call")
    var tools = c.list_tools()
    if len(tools) != 1 or tools[0].name != "run":
        raise Error("the child server does not expose the run tool")
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("task", task)
    w.end_object()
    var r = c.call_tool("run", w.finish())
    var code = c.close()
    var wall = Float64(perf_counter_ns() - t0) / 1e9
    print(r.text)
    var stats = String()
    if r.structured.byte_length() > 0:
        stats = parse_json(r.structured).get("stats").to_json()
    print("[qumo-agent call] " + c.server_name + " exit=" + String(code) + " wall_s=" + String(wall) + " child_stats=" + stats, file=stderr)
    return 1 if r.is_error else 0


def main():
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var code: Int
    try:
        var o = Opts()
        var err = parse_opts(args, o)
        if err.byte_length() > 0:
            exit(_usage(err))
        if o.command == "help" or o.command == "--help" or o.command == "-h":
            print(USAGE)
            exit(0)
        var root = find_root(args[0])
        if o.command == "list":
            print(describe_agents(root), end="")
            code = 0
        elif o.command == "run" or o.command == "call":
            if len(o.positional) != 2:
                exit(_usage(o.command + " needs AGENT and TASK"))
            code = cmd_run(root, o) if o.command == "run" else cmd_call(root, o)
        elif o.command == "serve":
            if len(o.positional) != 1:
                exit(_usage("serve needs AGENT"))
            code = cmd_serve(root, o)
        else:
            exit(_usage("unknown command " + o.command))
            code = 2
    except e:
        print("qumo-agent: error: " + String(e), file=stderr)
        code = 1
    exit(code)
