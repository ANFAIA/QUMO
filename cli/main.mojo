"""The QUMO command-line interface (`qumo`).

Build: `mojo build -I . -o build/qumo cli/main.mojo`. (The entry file must
not be called `qumo.mojo`: a main module named `qumo` shadows the `qumo`
package and every `from qumo...` import fails.)

    qumo <command> [args] [--json] [--pretty]

Every command is a thin adapter over a tool of `qumo.ai.tools`, so
`qumo run x.qasm --json` prints exactly the `structuredContent` the MCP tool
`qumo_run_qasm` returns. Human-readable output by default, JSON with
`--json`; exit code 0 = success, 1 = the operation failed (diagnostic
printed: human on stderr, JSON on stdout), 2 = usage error.

Extending: append a `Command(...)` to `command_table()` (name, summary,
usage, handler `def(CliArgs) raises thin -> Int`), or just register an MCP
tool in `qumo.ai.tools.default_registry()` - it is immediately callable as
`qumo tool <name> '<json-args>'`.
"""

from std.sys import argv, exit, stderr

from qumo.ai.json import JsonWriter, JsonValue, parse_json, pretty_json
from qumo.ai.tools import ToolRegistry, ToolResult, tool_error
from qumo.ai.registry import full_registry
from qumo.ai.mcp import serve_stdio, LineReader, MCP_PROTOCOL_VERSION
from qumo.ai.diagnostics import diagnostic_from_error, Diagnostic
from qumo.ai.describe import QUMO_VERSION


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
struct CliArgs(Copyable):
    var command: String
    var positional: List[String]
    var opt_names: List[String]
    var opt_values: List[String]
    var json: Bool
    var pretty: Bool

    def __init__(out self):
        self.command = String()
        self.positional = List[String]()
        self.opt_names = List[String]()
        self.opt_values = List[String]()
        self.json = False
        self.pretty = False

    def has(self, name: String) -> Bool:
        for n in self.opt_names:
            if n == name:
                return True
        return False

    def get(self, name: String, default: String) -> String:
        var out = default
        for i in range(len(self.opt_names)):
            if self.opt_names[i] == name:
                out = self.opt_values[i]
        return out^

    def get_int(self, name: String, default: Int) raises -> Int:
        if not self.has(name):
            return default
        var v = self.get(name, "")
        try:
            return Int(v)
        except:
            raise Error("QM-E921: option --" + name + " expects an integer, got '" + v + "'. Hint: e.g. --" + name + " 1000.")

    def pos(self, i: Int, what: String) raises -> String:
        if i >= len(self.positional):
            raise Error("QM-E921: missing argument <" + what + "> for '" + self.command + "'. Hint: run 'qumo help " + self.command + "'.")
        return self.positional[i]


def _is_flag(name: String) -> Bool:
    return name == "json" or name == "pretty" or name == "help" or name == "quiet"


def parse_cli(args: List[String]) -> CliArgs:
    var a = CliArgs()
    var i = 0
    while i < len(args):
        var s = args[i]
        i += 1
        if s.startswith("--") and s.byte_length() > 2:
            var body = String(s[byte=2:])
            var eq = body.find("=")
            if eq >= 0:
                a.opt_names.append(String(body[byte=0:eq]))
                a.opt_values.append(String(body[byte=eq + 1:]))
            elif _is_flag(body):
                a.opt_names.append(body)
                a.opt_values.append("true")
            elif i < len(args):
                a.opt_names.append(body)
                a.opt_values.append(args[i])
                i += 1
            else:
                a.opt_names.append(body)
                a.opt_values.append("")
        elif s == "-o" and i < len(args):
            a.opt_names.append("output")
            a.opt_values.append(args[i])
            i += 1
        elif s == "-h":
            a.opt_names.append("help")
            a.opt_values.append("true")
        elif a.command.byte_length() == 0:
            a.command = s
        else:
            a.positional.append(s)
    a.json = a.has("json")
    a.pretty = a.has("pretty")
    return a^


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def fmt_fixed(v: Float64, decimals: Int) -> String:
    """Fixed-point decimal rendering (Mojo has no format specifiers)."""
    var scale = 1
    for _ in range(decimals):
        scale *= 10
    var x = Int(round(abs(v) * Float64(scale)))
    var s = String("-") if v < 0 and x != 0 else String()
    s += String(x // scale)
    if decimals > 0:
        var f = String(x % scale)
        s += "."
        for _ in range(decimals - f.byte_length()):
            s += "0"
        s += f
    return s^


def rpad(s: String, w: Int) -> String:
    var out = s
    for _ in range(w - s.byte_length()):
        out += " "
    return out^


def lpad(s: String, w: Int) -> String:
    var out = String()
    for _ in range(w - s.byte_length()):
        out += " "
    return out + s


def read_source(path: String) raises -> String:
    if path == "-":
        var r = LineReader(0)
        return r.read_all()
    try:
        with open(path, "r") as f:
            return f.read()
    except e:
        raise Error("QM-E920: cannot read file '" + path + "': " + String(e) + ". Hint: check the path, or pass '-' to read from stdin.")


def print_error(a: CliArgs, err: String):
    """Diagnostic for a failure: JSON on stdout with --json, else human on stderr."""
    var r = tool_error(err)
    if a.json:
        print(_maybe_pretty(a, r.structured))
    else:
        print(r.text, file=stderr)


def _maybe_pretty(a: CliArgs, s: String) -> String:
    if a.pretty:
        try:
            return pretty_json(s)
        except:
            return s
    return s


def call_tool(name: String, var w: JsonWriter) raises -> ToolResult:
    var reg = full_registry()
    return reg.call(name, parse_json(w.finish()))


def finish(a: CliArgs, r: ToolResult, human: String) -> Int:
    if r.is_error:
        if a.json:
            print(_maybe_pretty(a, r.structured))
        else:
            print(r.text, file=stderr)
        return 1
    if a.json:
        print(_maybe_pretty(a, r.structured))
    else:
        print(human)
    return 0


def _source_args(a: CliArgs, mut w: JsonWriter) raises:
    w.kv_string("qasm", read_source(a.pos(0, "file.qasm")))
    if a.has("params"):
        var p = a.get("params", "")
        _ = parse_json(p)  # validate early (QM-E900 with position)
        w.kv_raw("params", p)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_run(a: CliArgs) raises -> Int:
    # --device / --precision select the simulator explicitly (MCP tool
    # qumo_run_device); without them `qumo run` is qumo_run_qasm (automatic).
    var explicit = a.has("device") or a.has("precision")
    var w = JsonWriter()
    w.begin_object()
    _source_args(a, w)
    w.kv_int("shots", a.get_int("shots", 1024))
    w.kv_int("seed", a.get_int("seed", 1234))
    if a.has("max-qubits"):
        w.kv_int("max_qubits", a.get_int("max-qubits", 28))
    if explicit:
        w.kv_string("device", a.get("device", "auto"))
        w.kv_string("precision", a.get("precision", "auto"))
    w.end_object()
    var r = call_tool("qumo_run_device" if explicit else "qumo_run_qasm", w^)
    var human = String()
    if not r.is_error:
        var v = parse_json(r.structured)
        var shots = v.get("shots").as_int()
        var ms = v.get("total_ms").as_float() if explicit else v.get("timing_ms").get("simulate").as_float()
        human = a.positional[0] + ": " + String(v.get("num_qubits").as_int()) + " qubits, " + String(shots) + " shots, seed " + String(v.get("seed").as_int()) + ", " + v.get("method").as_string() + ", " + fmt_fixed(ms, 3) + " ms"
        var counts = v.get("counts")
        var width = 4
        for i in range(len(counts)):
            width = max(width, counts.key_at(i).byte_length())
        var shown = 0
        for i in range(len(counts)):
            if shown >= 64:
                human += "\n  ... " + String(len(counts) - shown) + " more outcome(s) (use --json)"
                break
            shown += 1
            var c = counts.at(i).as_int()
            var p = Float64(c) / Float64(shots)
            var bar = String()
            for _ in range(Int(round(p * 40.0))):
                bar += "#"
            human += "\n  " + rpad(counts.key_at(i), width) + "  " + lpad(String(c), 8) + "  " + fmt_fixed(p, 4) + "  " + bar
        var ws = v.get("warnings")
        for i in range(len(ws)):
            human += "\n  note[" + ws.at(i).get("code").as_string() + "]: " + ws.at(i).get("message").as_string()
    return finish(a, r, human)


def cmd_statevector(a: CliArgs) raises -> Int:
    var w = JsonWriter()
    w.begin_object()
    _source_args(a, w)
    w.kv_int("max_amplitudes", a.get_int("max-amplitudes", 64))
    if a.has("max-qubits"):
        w.kv_int("max_qubits", a.get_int("max-qubits", 24))
    w.end_object()
    var r = call_tool("qumo_statevector", w^)
    var human = String()
    if not r.is_error:
        var v = parse_json(r.structured)
        var amps = v.get("amplitudes")
        human = String(v.get("num_qubits").as_int()) + " qubit(s), " + String(v.get("num_nonzero").as_int()) + " nonzero amplitude(s), norm " + fmt_fixed(v.get("norm").as_float(), 12)
        if v.get("measurements_ignored").as_bool():
            human += " (measurements/resets ignored)"
        for i in range(len(amps)):
            var x = amps.at(i)
            var re = x.get("re").as_float()
            var im = x.get("im").as_float()
            human += "\n  |" + x.get("bitstring").as_string() + ">  " + ("+" if re >= 0 else "") + fmt_fixed(re, 10) + " " + ("+" if im >= 0 else "") + fmt_fixed(im, 10) + "i   p=" + fmt_fixed(x.get("probability").as_float(), 6)
        if v.get("truncated").as_bool():
            human += "\n  ... truncated (use --max-amplitudes)"
    return finish(a, r, human)


def cmd_expect(a: CliArgs) raises -> Int:
    var w = JsonWriter()
    w.begin_object()
    _source_args(a, w)
    if len(a.positional) > 2:
        w.key("observable")
        w.begin_array()
        for i in range(1, len(a.positional)):
            w.string(a.positional[i])
        w.end_array()
    else:
        w.kv_string("observable", a.pos(1, "observable"))
    w.end_object()
    var r = call_tool("qumo_expectation", w^)
    var human = String()
    if not r.is_error:
        var res = parse_json(r.structured).get("results")
        for i in range(len(res)):
            if i > 0:
                human += "\n"
            human += "<" + res.at(i).get("observable").as_string() + "> = " + String(res.at(i).get("expectation").as_float())
    return finish(a, r, human)


def cmd_validate(a: CliArgs) raises -> Int:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("qasm", read_source(a.pos(0, "file.qasm")))
    w.end_object()
    var r = call_tool("qumo_validate_qasm", w^)
    if r.is_error:
        return finish(a, r, "")
    var v = parse_json(r.structured)
    var valid = v.get("valid").as_bool()
    if a.json:
        print(_maybe_pretty(a, r.structured))
        return 0 if valid else 1
    var errs = v.get("errors")
    var ws = v.get("warnings")
    var out = String()
    for i in range(len(errs)):
        out += _render_diag(errs.at(i)) + "\n"
    for i in range(len(ws)):
        out += _render_diag(ws.at(i)) + "\n"
    if valid:
        var s = v.get("summary")
        out += "OK: " + a.positional[0] + " is valid (" + String(s.get("num_qubits").as_int()) + " qubits, " + String(s.get("num_instructions").as_int()) + " instructions, " + String(len(ws)) + " note(s))"
        print(out)
        return 0
    out += "INVALID: " + String(len(errs)) + " error(s)"
    print(out, file=stderr)
    return 1


def _render_diag(d: JsonValue) raises -> String:
    var s = d.get_string("severity", "error") + "[" + d.get_string("code", "") + "]: " + d.get_string("message", "")
    if d.has("line"):
        s += "\n  --> line " + String(d.get("line").as_int()) + ", col " + String(d.get("col").as_int())
    if d.has("instruction"):
        s += "\n  --> instruction #" + String(d.get("instruction").as_int())
    if d.has("snippet"):
        s += "\n" + d.get("snippet").as_string()
    var h = d.get_string("hint", "")
    if h.byte_length() > 0:
        s += "\n  hint: " + h
    return s^


def cmd_info(a: CliArgs) raises -> Int:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("qasm", read_source(a.pos(0, "file.qasm")))
    w.kv_bool("include_qasm", a.has("qasm"))
    w.kv_bool("include_instructions", a.has("instructions"))
    w.end_object()
    var r = call_tool("qumo_circuit_info", w^)
    var human = String()
    if not r.is_error:
        var v = parse_json(r.structured)
        var s = v.get("summary")
        var keys = s.keys()
        for i in range(len(keys)):
            var k = keys[i]
            var x = s.at(i)
            if i > 0:
                human += "\n"
            human += rpad(k, 22) + " " + x.to_json()
        var l = v.get("lint")
        for i in range(len(l)):
            human += "\n" + _render_diag(l.at(i))
        if v.has("qasm"):
            human += "\n\n" + v.get("qasm").as_string()
    return finish(a, r, human)


def cmd_describe(a: CliArgs) raises -> Int:
    var topic = a.positional[0] if len(a.positional) > 0 else String("capabilities")
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("topic", topic)
    if len(a.positional) > 1:
        w.kv_string("name", a.positional[1])
    w.end_object()
    var r = call_tool("qumo_describe", w^)
    var human = String()
    if not r.is_error:
        var v = parse_json(r.structured)
        if topic == "gates" and len(a.positional) == 1:
            human = rpad("name", 16) + rpad("qubits", 7) + rpad("params", 7) + rpad("flags", 32) + "description"
            var gs = v.get("gates")
            for i in range(len(gs)):
                var g = gs.at(i)
                var flags = String()
                if g.get_bool("clifford", False): flags += "clifford "
                if g.get_bool("diagonal", False): flags += "diagonal "
                if g.get_bool("self_inverse", False): flags += "self-inverse"
                var nm = g.get("name").as_string()
                var al = g.get("aliases")
                if len(al) > 0:
                    nm += " (" + al.at(0).as_string() + ")"
                human += "\n" + rpad(nm, 16) + rpad(String(g.get("num_qubits").as_int()), 7) + rpad(String(g.get("num_params").as_int()), 7) + rpad(flags, 32) + g.get("description").as_string()
        elif topic == "errors" and len(a.positional) == 1:
            var cs = v.get("codes")
            for i in range(len(cs)):
                var e = cs.at(i)
                if i > 0:
                    human += "\n"
                human += e.get("code").as_string() + "  " + e.get("meaning").as_string() + "\n          hint: " + e.get("hint").as_string()
        else:
            human = pretty_json(r.structured)
    return finish(a, r, human)


def cmd_convert(a: CliArgs) raises -> Int:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("qasm", read_source(a.pos(0, "file.qasm")))
    w.kv_string("to", a.get("to", "qasm3"))
    w.end_object()
    var r = call_tool("qumo_convert", w^)
    if r.is_error:
        return finish(a, r, "")
    var v = parse_json(r.structured)
    var text: String
    if v.has("qasm"):
        text = v.get("qasm").as_string()
    else:
        text = v.get("circuit").to_pretty_json() + "\n"
    if a.has("output"):
        var path = a.get("output", "")
        try:
            with open(path, "w") as f:
                f.write(text)
        except e:
            print_error(a, "QM-E920: cannot write file '" + path + "': " + String(e) + ". Hint: check the directory exists and is writable.")
            return 1
        if a.json:
            print(_maybe_pretty(a, r.structured))
        return 0
    if a.json:
        print(_maybe_pretty(a, r.structured))
    else:
        print(text, end="")
    return 0


def cmd_tool(a: CliArgs) raises -> Int:
    var reg = full_registry()
    var name = a.pos(0, "tool-name|list")
    if name == "list":
        if a.json:
            var w = JsonWriter()
            w.begin_object()
            w.key("tools")
            reg.write_list_json(w)
            w.end_object()
            print(_maybe_pretty(a, w.finish()))
        else:
            for t in reg.tools:
                print(rpad(t.name, 22), t.title)
        return 0
    var args_text = a.positional[1] if len(a.positional) > 1 else String("{}")
    if args_text == "-":
        args_text = read_source("-")
    var r = reg.call(name, parse_json(args_text))
    if r.is_error:
        print(_maybe_pretty(a, r.structured))  # `tool` always speaks JSON
        return 2 if r.structured.find("QM-E912") >= 0 else 1
    print(_maybe_pretty(a, r.structured))
    return 0


def cmd_mcp(a: CliArgs) raises -> Int:
    serve_stdio()
    return 0


def cmd_version(a: CliArgs) raises -> Int:
    if a.json:
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("name", "qumo")
        w.kv_string("version", QUMO_VERSION)
        w.kv_string("mcp_protocol", MCP_PROTOCOL_VERSION)
        w.kv_string("language", "Mojo 1.1")
        w.end_object()
        print(w.finish())
    else:
        print("qumo", QUMO_VERSION, "(Mojo 1.1, MCP " + String(MCP_PROTOCOL_VERSION) + ")")
    return 0


def cmd_noop(a: CliArgs) raises -> Int:
    return 0


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------
comptime CmdHandler = def(CliArgs) raises thin -> Int


struct Command(Copyable):
    var name: String
    var summary: String
    var usage: String
    var handler: CmdHandler

    def __init__(out self, name: String, summary: String, usage: String, handler: CmdHandler):
        self.name = name
        self.summary = summary
        self.usage = usage
        self.handler = handler


def command_table() -> List[Command]:
    var t = List[Command]()
    t.append(Command("run", "Sample measurement counts (seeded, deterministic)", "qumo run <file.qasm|-> [--shots N] [--seed S] [--params '[0.1]'|'{\"theta\":0.1}'] [--max-qubits N] [--device auto|cpu|gpu] [--precision auto|fp32|fp64] [--json]\n\n--device / --precision run through the MCP tool qumo_run_device: 'gpu' needs a build made with -D QUMO_GPU (make cli-gpu); --precision applies to GPU state vectors (auto = QUMO_GPU_PRECISION or the build default, fp32; fp64 on NVIDIA / AMD only).", cmd_run))
    t.append(Command("statevector", "Print the exact final state vector", "qumo statevector <file.qasm|-> [--max-amplitudes N] [--params ...] [--json]", cmd_statevector))
    t.append(Command("expect", "Exact Pauli expectation value(s)", "qumo expect <file.qasm|-> \"0.5*Z0 Z1 - X0\" [more observables...] [--params ...] [--json]", cmd_expect))
    t.append(Command("validate", "Parse + lint OpenQASM with structured diagnostics", "qumo validate <file.qasm|-> [--json]", cmd_validate))
    t.append(Command("info", "Circuit summary: depth, gate counts, T-count, flags, lint", "qumo info <file.qasm|-> [--qasm] [--instructions] [--json]", cmd_info))
    t.append(Command("describe", "Self-description: gates | gate <name> | errors | error <code> | capabilities", "qumo describe [gates|gate <name>|errors|error <code>|capabilities] [--json]", cmd_describe))
    t.append(Command("convert", "Re-emit canonical OpenQASM 3 (or circuit JSON)", "qumo convert <file.qasm|-> [--to qasm3|json] [-o out]", cmd_convert))
    t.append(Command("tool", "Call any registered MCP tool with JSON arguments", "qumo tool list | qumo tool <name> '<json-args>'|- [--pretty]", cmd_tool))
    t.append(Command("mcp", "Run the MCP server on stdio (JSON-RPC 2.0, MCP 2025-06-18)", "qumo mcp", cmd_mcp))
    t.append(Command("version", "Print version", "qumo version [--json]", cmd_version))
    t.append(Command("help", "Show help for all commands or one command", "qumo help [command]", cmd_noop))
    return t^


def print_help(table: List[Command], topic: String):
    if topic.byte_length() > 0:
        for c in table:
            if c.name == topic:
                print("qumo " + c.name + " - " + c.summary + "\n\nusage: " + c.usage)
                return
    var s = String("qumo " + String(QUMO_VERSION) + " - AI-native quantum toolkit (Mojo)\n\nusage: qumo <command> [args] [--json] [--pretty]\n\ncommands:")
    for c in table:
        s += "\n  " + rpad(c.name, 12) + " " + c.summary
    s += "\n\nInput files may be OpenQASM 3.0 or 2.0; use '-' for stdin. Exit codes: 0 ok, 1 failed, 2 usage."
    s += "\nErrors carry stable codes (QM-Exxx) with line:col and a hint: 'qumo describe errors'."
    print(s)


def main():
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var a = parse_cli(args)
    var table = command_table()
    if a.command.byte_length() == 0 or a.command == "help" or a.command == "--help":
        print_help(table, a.positional[0] if len(a.positional) > 0 else String())
        exit(0)
    if a.has("help"):
        print_help(table, a.command)
        exit(0)
    var code = 2
    var found = False
    for c in table:
        if c.name == a.command:
            found = True
            var h = c.handler
            try:
                code = h(a)
            except e:
                var msg = String(e)
                print_error(a, msg)
                code = 2 if msg.startswith("QM-E921") else 1
            break
    if not found:
        var names = List[String]()
        for c in table:
            names.append(c.name)
        print_error(a, "QM-E912: unknown command '" + a.command + "'. Hint: commands are " + String(", ").join(names) + " (see 'qumo help').")
        code = 2
    exit(code)
