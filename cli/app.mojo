"""The QUMO command-line interface as a library: `run_cli(args) -> CliOut`.

`cli/main.mojo` is the executable (it prints a `CliOut` and exits with its
code); tests call `run_cli` in process (tests/test_cli_frontdoor.mojo).

    qumo <command> [args] [options] [--json] [--pretty]

Every command is a thin adapter over a tool of the registry
(`qumo.ai.registry.full_registry`), so `qumo run x.qasm --json` prints exactly
the `structuredContent` the MCP tool `qumo_run_qasm` returns, and no command
adds simulation or compilation logic of its own. Human-readable output by
default, JSON with `--json`; exit code 0 = success, 1 = the operation failed
(diagnostic printed: human on stderr, JSON on stdout), 2 = usage error.

Options are strict: a command declares its options in `command_table()`, the
help text is generated from the same declaration, and an option the command
does not declare is a usage error (QM-E921, exit code 2) with a "did you
mean" hint. Nothing is ignored silently.

Extending: add a `Command` to `command_table()` (name, summary, synopsis,
options, handler `def(CliArgs) raises thin -> CliOut`), or just register an
MCP tool in a `register_*_tools` function - it is immediately callable as
`qumo tool <name> '<json-args>'` and documented by `qumo tool <name> --help`.
"""

from qumo.core.json import JsonWriter, JsonValue, parse_json, pretty_json
from qumo.ai.tools import ToolRegistry, ToolResult, tool_error
from qumo.ai.registry import full_registry
from qumo.ai.mcp import serve_stdio, LineReader, MCP_PROTOCOL_VERSION
from qumo.ai.describe import QUMO_VERSION
from qumo.ai.cli_help import did_you_mean, tool_help_text, tool_takes, with_file_as_qasm, wrap_text


# ---------------------------------------------------------------------------
# Result of one invocation
# ---------------------------------------------------------------------------
struct CliOut(Movable):
    """What one invocation prints and returns: `out` goes to stdout and `err`
    to stderr, both verbatim (they carry their own final newline)."""

    var code: Int
    var out: String
    var err: String

    def __init__(out self, code: Int, var out: String, var err: String):
        self.code = code
        self.out = out^
        self.err = err^


def _ok(text: String) -> CliOut:
    return CliOut(0, text + "\n", String())


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
        """Value of an option; the last one wins when it is repeated."""
        var out = default
        for i in range(len(self.opt_names)):
            if self.opt_names[i] == name:
                out = self.opt_values[i]
        return out^

    def get_all(self, name: String) -> List[String]:
        """Every value of a repeatable option, in command-line order."""
        var out = List[String]()
        for i in range(len(self.opt_names)):
            if self.opt_names[i] == name:
                out.append(self.opt_values[i])
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
            raise Error("QM-E921: missing argument <" + what + "> for 'qumo " + self.command + "'. Hint: run 'qumo " + self.command + " --help'.")
        return self.positional[i]


struct Opt(Copyable):
    var name: String  # without the leading "--"
    var metavar: String  # "" = a flag (takes no value)
    var help: String

    def __init__(out self, name: String, metavar: String, help: String):
        self.name = name
        self.metavar = metavar
        self.help = help


comptime CmdHandler = def(CliArgs) raises thin -> CliOut


struct Command(Copyable):
    var name: String
    var summary: String
    var synopsis: String  # positional part of the usage line
    var max_positional: Int  # -1 = any number
    var options: List[Opt]
    var examples: List[String]
    var notes: String
    var handler: CmdHandler

    def __init__(out self, name: String, summary: String, synopsis: String, max_positional: Int, handler: CmdHandler):
        self.name = name
        self.summary = summary
        self.synopsis = synopsis
        self.max_positional = max_positional
        self.options = List[Opt]()
        self.examples = List[String]()
        self.notes = String()
        self.handler = handler

    def opt(mut self, name: String, metavar: String, help: String):
        self.options.append(Opt(name, metavar, help))

    def example(mut self, text: String):
        self.examples.append(text)

    def option_kind(self, name: String) -> Int:
        """0 = not an option of this command, 1 = flag, 2 = takes a value."""
        if name == "json" or name == "pretty" or name == "help":
            return 1
        for o in self.options:
            if o.name == name:
                return 2 if o.metavar.byte_length() > 0 else 1
        return 0

    def option_names(self) -> List[String]:
        var out = List[String]()
        for o in self.options:
            out.append(o.name)
        out.append("json")
        out.append("pretty")
        out.append("help")
        return out^


def _unknown_option(c: Command, shown: String, name: String) -> Error:
    var names = c.option_names()
    var near = did_you_mean(name, names)
    var hint = String()
    if near.byte_length() > 0:
        hint = "did you mean '--" + near + "'? "
    var dashed = List[String]()
    for n in names:
        dashed.append("--" + n)
    return Error("QM-E921: unknown option '" + shown + "' for 'qumo " + c.name + "'. Hint: " + hint + "Options of 'qumo " + c.name + "': " + String(", ").join(dashed) + " (see 'qumo " + c.name + " --help').")


def _is_short_option(s: String) -> Bool:
    """`-x`: one dash and one ASCII letter ('-' alone is stdin, '-X0' and '-0.5*Z0' are observables)."""
    var bs = s.as_bytes()
    if len(bs) != 2 or Int(bs[0]) != ord("-"):
        return False
    var ch = Int(bs[1])
    return (ch >= ord("a") and ch <= ord("z")) or (ch >= ord("A") and ch <= ord("Z"))


def parse_args(c: Command, tokens: List[String]) raises -> CliArgs:
    """Strict parse of the tokens after the command name against the options
    `c` declares. `--name value`, `--name=value`, flags, `-o` for `--output`,
    and `--` to end the options. QM-E921 for an unknown option, a missing
    value or a positional argument too many."""
    var a = CliArgs()
    a.command = c.name
    var i = 0
    var only_positional = False
    while i < len(tokens):
        var s = tokens[i]
        i += 1
        if only_positional:
            a.positional.append(s)
        elif s == "--":
            only_positional = True
        elif s.startswith("--"):
            var body = String(s[byte=2:])
            var eq = body.find("=")
            var name = body if eq < 0 else String(body[byte=0:eq])
            var kind = c.option_kind(name)
            if kind == 0:
                raise _unknown_option(c, "--" + name, name)
            if kind == 1:
                var val = String("true") if eq < 0 else String(body[byte=eq + 1:])
                if val != "true" and val != "false":
                    raise Error("QM-E921: option --" + name + " is a flag and takes no value, got '" + val + "'. Hint: write just --" + name + ".")
                if val == "true":
                    a.opt_names.append(name)
                    a.opt_values.append("true")
            elif eq >= 0:
                a.opt_names.append(name)
                a.opt_values.append(String(body[byte=eq + 1:]))
            elif i < len(tokens):
                a.opt_names.append(name)
                a.opt_values.append(tokens[i])
                i += 1
            else:
                raise Error("QM-E921: option --" + name + " needs a value. Hint: see 'qumo " + c.name + " --help'.")
        elif s == "-o":
            if c.option_kind("output") != 2:
                raise _unknown_option(c, "-o", "output")
            if i >= len(tokens):
                raise Error("QM-E921: option -o needs a file name. Hint: e.g. -o out.qasm.")
            a.opt_names.append("output")
            a.opt_values.append(tokens[i])
            i += 1
        elif s == "-h":
            a.opt_names.append("help")
            a.opt_values.append("true")
        elif _is_short_option(s):
            raise _unknown_option(c, s, String(s[byte=1:]))
        else:
            a.positional.append(s)
    if c.max_positional >= 0 and len(a.positional) > c.max_positional:
        var extra = a.positional[c.max_positional]
        raise Error("QM-E921: unexpected argument '" + extra + "' for 'qumo " + c.name + "' (it takes at most " + String(c.max_positional) + " positional argument(s)). Hint: usage: qumo " + c.name + " " + c.synopsis + "; quote an argument that contains spaces.")
    a.json = a.has("json")
    a.pretty = a.has("pretty")
    return a^


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def fmt_fixed(v: Float64, decimals: Int) -> String:
    """Fixed-point decimal rendering (Mojo has no format specifiers)."""
    if abs(v) >= 1e12:  # past what an Int holds once it is scaled
        return String(v)
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


def _write_file(path: String, text: String) raises:
    try:
        with open(path, "w") as f:
            f.write(text)
    except e:
        raise Error("QM-E920: cannot write file '" + path + "': " + String(e) + ". Hint: check the directory exists and is writable.")


def _maybe_pretty(pretty: Bool, s: String) -> String:
    if pretty:
        try:
            return pretty_json(s)
        except:
            return s
    return s


def _usage_code(err: String) -> Int:
    """Exit code of an error text: 2 for usage errors (QM-E921 bad command line, QM-E912 unknown command), else 1."""
    return 2 if err.startswith("QM-E921") or err.startswith("QM-E912") else 1


def fail(json: Bool, pretty: Bool, err: String) -> CliOut:
    """Diagnostic for a failure: JSON on stdout with --json, else human on stderr."""
    var r = tool_error(err)
    if json:
        return CliOut(_usage_code(err), _maybe_pretty(pretty, r.structured) + "\n", String())
    return CliOut(_usage_code(err), String(), r.text + "\n")


def call_tool(name: String, var w: JsonWriter) raises -> ToolResult:
    var reg = full_registry()
    return reg.call(name, parse_json(w.finish()))


def finish(a: CliArgs, r: ToolResult, human: String) -> CliOut:
    if r.is_error:
        if a.json:
            return CliOut(1, _maybe_pretty(a.pretty, r.structured) + "\n", String())
        return CliOut(1, String(), r.text + "\n")
    if a.json:
        return _ok(_maybe_pretty(a.pretty, r.structured))
    return _ok(human)


def _source_args(a: CliArgs, mut w: JsonWriter) raises:
    w.kv_string("qasm", read_source(a.pos(0, "file.qasm")))
    if a.has("params"):
        var p = a.get("params", "")
        _ = parse_json(p)  # validate early (QM-E900 with position)
        w.kv_raw("params", p)


def _counts_table(counts: JsonValue, shots: Int) raises -> String:
    """One line per outcome: bitstring, count, probability, bar (at most 64 outcomes)."""
    var out = String()
    var width = 4
    for i in range(len(counts)):
        width = max(width, counts.key_at(i).byte_length())
    var shown = 0
    for i in range(len(counts)):
        if shown >= 64:
            out += "\n  ... " + String(len(counts) - shown) + " more outcome(s) (use --json)"
            break
        shown += 1
        var c = counts.at(i).as_int()
        var p = Float64(c) / Float64(shots)
        var bar = String()
        for _ in range(Int(round(p * 40.0))):
            bar += "#"
        out += "\n  " + rpad(counts.key_at(i), width) + "  " + lpad(String(c), 8) + "  " + fmt_fixed(p, 4) + "  " + bar
    return out^


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_run(a: CliArgs) raises -> CliOut:
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
    if a.has("max-iterations"):
        w.kv_int("max_iterations", a.get_int("max-iterations", 65536))
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
        human += _counts_table(v.get("counts"), shots)
        if v.has("iterations"):
            var it = v.get("iterations")
            human += "\n  loop iterations per shot: mean " + fmt_fixed(it.get("mean").as_float(), 3) + ", max " + String(it.get("max").as_int())
            var outs = v.get("outputs")
            var names = outs.keys()
            for i in range(len(names)):
                var o = outs.at(i)
                human += "\n  output " + names[i] + ": mean " + fmt_fixed(o.get("mean").as_float(), 6) + ", range " + o.get("min").number_text() + " to " + o.get("max").number_text()
        var ws = v.get("warnings")
        for i in range(len(ws)):
            human += "\n  note[" + ws.at(i).get("code").as_string() + "]: " + ws.at(i).get("message").as_string()
    return finish(a, r, human)


def cmd_statevector(a: CliArgs) raises -> CliOut:
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


def cmd_expect(a: CliArgs) raises -> CliOut:
    # Observables: positional (after the file) and / or --observable, repeatable.
    var obs = List[String]()
    for i in range(1, len(a.positional)):
        obs.append(a.positional[i])
    for o in a.get_all("observable"):
        obs.append(o)
    if len(obs) == 0:
        raise Error("QM-E921: missing <observable> for 'qumo expect'. Hint: give a Pauli sum after the file or with --observable, e.g. qumo expect bell.qasm \"Z0 Z1\" or qumo expect bell.qasm --observable \"0.5*Z0 Z1 - X0\".")
    var w = JsonWriter()
    w.begin_object()
    _source_args(a, w)
    if len(obs) > 1:
        w.kv_string_list("observable", obs)
    else:
        w.kv_string("observable", obs[0])
    if a.has("max-qubits"):
        w.kv_int("max_qubits", a.get_int("max-qubits", 28))
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


def cmd_validate(a: CliArgs) raises -> CliOut:
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
        return CliOut(0 if valid else 1, _maybe_pretty(a.pretty, r.structured) + "\n", String())
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
        return _ok(out)
    out += "INVALID: " + String(len(errs)) + " error(s)"
    if not v.get_bool("errors_complete", True):
        out += "; checking stopped at the last one (it cannot be skipped safely): fix it and validate again"
    return CliOut(1, String(), out + "\n")


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


def cmd_info(a: CliArgs) raises -> CliOut:
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


def cmd_draw(a: CliArgs) raises -> CliOut:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("qasm", read_source(a.pos(0, "file.qasm")))
    w.kv_int("width", a.get_int("width", 80))
    w.kv_string("style", "unicode" if a.has("unicode") else "ascii")
    w.end_object()
    var r = call_tool("qumo_draw", w^)
    var human = String()
    if not r.is_error:
        human = parse_json(r.structured).get("drawing").as_string()
    return finish(a, r, human)


def cmd_describe(a: CliArgs) raises -> CliOut:
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


def cmd_convert(a: CliArgs) raises -> CliOut:
    var w = JsonWriter()
    w.begin_object()
    _source_args(a, w)
    w.kv_string("to", a.get("to", "qasm3"))
    w.end_object()
    var r = call_tool("qumo_convert", w^)
    if r.is_error:
        return finish(a, r, "")
    var v = parse_json(r.structured)
    var text: String
    if v.has("qasm"):
        text = v.get("qasm").as_string()
    elif v.has("program"):
        text = v.get("program").to_pretty_json() + "\n"
    else:
        text = v.get("circuit").to_pretty_json() + "\n"
    if a.has("output"):
        _write_file(a.get("output", ""), text)
        if a.json:
            return _ok(_maybe_pretty(a.pretty, r.structured))
        return CliOut(0, String(), String())
    if a.json:
        return _ok(_maybe_pretty(a.pretty, r.structured))
    return CliOut(0, text^, String())


def _int_list_text(v: JsonValue) raises -> String:
    var out = String("[")
    for i in range(len(v)):
        if i > 0:
            out += ", "
        out += String(v.at(i).as_int())
    return out + "]"


def _metric_row(name: String, before: String, after: String) -> String:
    return "\n  " + rpad(name, 20) + lpad(before, 9) + lpad(after, 9)


def cmd_transpile(a: CliArgs) raises -> CliOut:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("qasm", read_source(a.pos(0, "file.qasm")))
    if a.has("target"):
        w.kv_string("target", a.get("target", ""))
    if a.has("basis"):
        w.kv_string("basis", a.get("basis", "any"))
    if a.has("level"):
        w.kv_int("level", a.get_int("level", 2))
    if a.has("seed"):
        w.kv_int("seed", a.get_int("seed", 7))
    if a.has("pipeline"):
        w.kv_string("pipeline", a.get("pipeline", ""))
    if a.has("ai"):
        w.kv_bool("ai", True)
    if a.has("ai-level"):
        w.kv_int("ai_level", a.get_int("ai-level", 2))
    w.end_object()
    var r = call_tool("qumo_transpile", w^)
    if r.is_error:
        return finish(a, r, "")
    var v = parse_json(r.structured)
    var qasm = v.get("qasm").as_string()
    if a.has("output"):
        _write_file(a.get("output", ""), qasm)
    if a.json:
        return _ok(_maybe_pretty(a.pretty, r.structured))
    if a.has("qasm"):
        return CliOut(0, qasm^, String())
    var t = v.get("target")
    var how = String("pipeline") if a.has("pipeline") else ("AI level " + String(a.get_int("ai-level", 2)) if a.has("ai") else "level " + String(a.get_int("level", 2)))
    var human = a.positional[0] + " -> " + t.get("name").as_string() + " (" + String(t.get("num_qubits").as_int()) + " qubits, " + t.get("connectivity").as_string() + "): " + how + ", seed " + String(a.get_int("seed", 7)) + ", " + fmt_fixed(v.get("compile_ms").as_float(), 3) + " ms"
    human += "\n  pipeline: " + v.get("pipeline").as_string()
    if v.has("gates_before"):
        human += _metric_row("", "before", "after")
        human += _metric_row("gates", String(v.get("gates_before").as_int()), String(v.get("gates_after").as_int()))
        human += _metric_row("two-qubit gates", String(v.get("two_qubit_before").as_int()), String(v.get("two_qubit_after").as_int()))
        human += _metric_row("depth", String(v.get("depth_before").as_int()), String(v.get("depth_after").as_int()))
        human += _metric_row("swaps inserted", "", String(v.get("swaps").as_int()))
        human += _metric_row("estimated fidelity", "", fmt_fixed(v.get("estimated_fidelity").as_float(), 6))
    else:  # a program with runtime control flow: compiled block by block
        human += "\n  " + v.get("metrics").as_string()
        human += "\n  blocks verified: " + String(v.get("blocks_checked").as_int()) + ", not verified: " + String(v.get("blocks_unchecked").as_int()) + ", restored swaps: " + String(v.get("restored_swaps").as_int())
    human += "\n  layout: initial " + _int_list_text(v.get("initial_layout")) + ", final " + _int_list_text(v.get("final_layout"))
    if a.has("output"):
        human += "\n  compiled OpenQASM 3 written to " + a.get("output", "")
    else:
        human += "\n  (-o out.qasm saves the compiled circuit, --qasm prints it, --json gives the per-pass report)"
    return _ok(human)


def _emulate_call(a: CliArgs, source: String, compile: Bool) raises -> ToolResult:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("qasm", source)
    w.kv_string("target", a.get("target", ""))
    w.kv_int("shots", a.get_int("shots", 1024))
    w.kv_int("seed", a.get_int("seed", 1234))
    if compile:
        w.kv_bool("compile", True)
        w.kv_int("level", a.get_int("level", 2))
    w.end_object()
    return call_tool("qumo_emulate", w^)


def cmd_emulate(a: CliArgs) raises -> CliOut:
    var file = a.pos(0, "file.qasm")
    if not a.has("target"):
        raise Error("QM-E921: 'qumo emulate' needs --target (the device model whose noise is applied). Hint: e.g. qumo emulate " + file + " --target grid:2x2/sc_cz; other specs: line:5/u,cx, heavy_hex:3/rz,sx,x,ecr, trapped_ion:5, neutral_atom:3x3.")
    if a.has("compile") and a.has("no-compile"):
        raise Error("QM-E921: --compile and --no-compile are mutually exclusive. Hint: drop one of them; without either, the circuit is compiled only when it does not fit the target.")
    var source = read_source(file)
    var target = a.get("target", "")
    # The circuit runs as written when it already fits the target; otherwise it
    # is compiled first (the tool's own compile=true), and the output says so.
    var r = _emulate_call(a, source, a.has("compile"))
    var auto_compiled = False
    if r.is_error and r.structured.find("QM-E615") >= 0 and not a.has("compile"):
        if a.has("no-compile"):
            var msg = parse_json(r.structured).get("error").get("message").as_string()
            return fail(a.json, a.pretty, "QM-E615: " + msg + ". Hint: compile it first: qumo transpile " + file + " --target " + target + " -o compiled.qasm && qumo emulate compiled.qasm --target " + target + " --no-compile; or drop --no-compile and 'qumo emulate' compiles it for you.")
        r = _emulate_call(a, source, True)
        auto_compiled = True
    var human = String()
    if not r.is_error:
        var v = parse_json(r.structured)
        var shots = v.get("shots").as_int()
        human = file + " on " + v.get("target").as_string() + ": " + String(v.get("num_qubits").as_int()) + " qubits, " + String(shots) + " shots, seed " + String(v.get("seed").as_int()) + ", " + v.get("method").as_string() + ", estimated fidelity " + fmt_fixed(v.get("estimated_fidelity").as_float(), 6)
        human += _counts_table(v.get("counts"), shots)
        if auto_compiled:
            human += "\n  note: the circuit did not fit the target as written, so it was compiled first (level " + String(a.get_int("level", 2)) + "). --no-compile forbids that; 'qumo transpile' shows what the compiler did."
    return finish(a, r, human)


def cmd_vqe(a: CliArgs) raises -> CliOut:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("hamiltonian", a.pos(0, "hamiltonian"))
    if a.has("qubits"):
        w.kv_int("num_qubits", a.get_int("qubits", 0))
    if a.has("layers"):
        w.kv_int("layers", a.get_int("layers", 2))
    if a.has("entanglement"):
        w.kv_string("entanglement", a.get("entanglement", "linear"))
    if a.has("optimizer"):
        w.kv_string("optimizer", a.get("optimizer", "bfgs"))
    if a.has("gradient"):
        w.kv_string("gradient", a.get("gradient", "adjoint"))
    if a.has("maxiter"):
        w.kv_int("maxiter", a.get_int("maxiter", 200))
    if a.has("seed"):
        w.kv_int("seed", a.get_int("seed", 7))
    w.end_object()
    var r = call_tool("qumo_vqe", w^)
    var human = String()
    if not r.is_error:
        var v = parse_json(r.structured)
        human = "energy " + String(v.get("energy").as_float()) + " (" + ("converged" if v.get("converged").as_bool() else "NOT converged") + ")"
        human += "\n  ansatz     " + v.get("ansatz").as_string() + ", seed " + String(a.get_int("seed", 7))
        human += "\n  optimizer  " + v.get("optimizer").as_string() + ", gradient " + v.get("gradient").as_string()
        human += "\n  cost       " + String(v.get("iterations").as_int()) + " iterations, " + String(v.get("circuit_evaluations").as_int()) + " circuit evaluations, " + fmt_fixed(v.get("wall_time_s").as_float(), 4) + " s"
        human += "\n  params     " + v.get("params").to_json()
    return finish(a, r, human)


def _find_tool(reg: ToolRegistry, name: String) -> Int:
    """Index of a tool by its name, or by its name without the 'qumo_' prefix."""
    var i = reg.find(name)
    if i < 0:
        i = reg.find("qumo_" + name)
    return i


def _unknown_tool(reg: ToolRegistry, name: String) -> String:
    var near = did_you_mean(name, reg.names())
    if near.byte_length() == 0:
        near = did_you_mean("qumo_" + name, reg.names())
    var hint = String("did you mean '" + near + "'? ") if near.byte_length() > 0 else String()
    return "QM-E912: unknown tool '" + name + "'. Hint: " + hint + "'qumo tool list' shows every tool, 'qumo tool <name> --help' its arguments."


def cmd_tool(a: CliArgs) raises -> CliOut:
    var reg = full_registry()
    var name = a.pos(0, "tool-name|list")
    if name == "list":
        if a.json:
            var w = JsonWriter()
            w.begin_object()
            w.key("tools")
            reg.write_list_json(w)
            w.end_object()
            return _ok(_maybe_pretty(a.pretty, w.finish()))
        var s = String()
        for t in reg.tools:
            s += rpad(t.name, 24) + t.title + "\n"
        s += "\nqumo tool <name> --help shows a tool's arguments and an example."
        return _ok(s)
    # `tool` always speaks JSON: results and errors alike go to stdout.
    var idx = _find_tool(reg, name)
    if idx < 0:
        return fail(True, a.pretty, _unknown_tool(reg, name))
    var args_text = a.positional[1] if len(a.positional) > 1 else String("{}")
    if args_text == "-":
        args_text = read_source("-")
    var args: JsonValue
    try:
        args = parse_json(args_text)
        if args.is_object() and args.has("file") and tool_takes(reg.tools[idx], "qasm") and not tool_takes(reg.tools[idx], "file"):
            # CLI-only convenience: {"file": "path"} stands for {"qasm": "<its contents>"}.
            if args.has("qasm"):
                raise Error("QM-E913: give either 'file' or 'qasm', not both. Hint: 'file' is read into 'qasm'.")
            args = parse_json(with_file_as_qasm(args, read_source(args.get("file").as_string())))
    except e:
        return fail(True, a.pretty, String(e))
    var r = reg.call(reg.tools[idx].name, args)
    return CliOut(1 if r.is_error else 0, _maybe_pretty(a.pretty, r.structured) + "\n", String())


def cmd_mcp(a: CliArgs) raises -> CliOut:
    serve_stdio()
    return CliOut(0, String(), String())


def cmd_version(a: CliArgs) raises -> CliOut:
    if a.json:
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("name", "qumo")
        w.kv_string("version", QUMO_VERSION)
        w.kv_string("mcp_protocol", MCP_PROTOCOL_VERSION)
        w.kv_string("language", "Mojo 1.2")
        w.end_object()
        return _ok(w.finish())
    return _ok("qumo " + String(QUMO_VERSION) + " (Mojo 1.2, MCP " + String(MCP_PROTOCOL_VERSION) + ")")


def cmd_noop(a: CliArgs) raises -> CliOut:
    return CliOut(0, String(), String())


# ---------------------------------------------------------------------------
# Command table
# ---------------------------------------------------------------------------
comptime _BELL = "examples/foundations/bell.qasm"
comptime _PARAMS_HELP = "values of 'input float' parameters: '[0.1, 0.2]' in declaration order or '{\"theta\": 0.1}'"
comptime _TARGET_HELP = "device spec: grid:2x2/sc_cz, line:5/u,cx, heavy_hex:3/rz,sx,x,ecr, trapped_ion:5, neutral_atom:3x3 (topology:size/basis)"


def command_table() -> List[Command]:
    var t = List[Command]()

    var run = Command("run", "Sample measurement counts (seeded, deterministic)", "<file.qasm|->", 1, cmd_run)
    run.opt("shots", "N", "number of shots (default 1024)")
    run.opt("seed", "S", "random seed (default 1234); the same seed gives the same counts")
    run.opt("params", "JSON", _PARAMS_HELP)
    run.opt("max-qubits", "N", "refuse wider dense simulations (default 28)")
    run.opt("max-iterations", "N", "loop back-edges allowed per shot (default 65536); past it the run stops with QM-E480")
    run.opt("device", "auto|cpu|gpu", "pick the simulator device (MCP tool qumo_run_device); 'gpu' needs a build made with -D QUMO_GPU (make cli-gpu)")
    run.opt("precision", "auto|fp32|fp64", "GPU state-vector precision (auto = QUMO_GPU_PRECISION or the build default, fp32; fp64 on NVIDIA / AMD only)")
    run.example("qumo run " + _BELL + " --shots 1000 --seed 7")
    run.example("qumo run circuit.qasm --params '{\"theta\": 0.3}' --json")
    t.append(run^)

    var sv = Command("statevector", "Print the exact final state vector", "<file.qasm|->", 1, cmd_statevector)
    sv.opt("max-amplitudes", "N", "print at most N amplitudes (default 64)")
    sv.opt("params", "JSON", _PARAMS_HELP)
    sv.opt("max-qubits", "N", "refuse wider circuits (default 24)")
    sv.example("qumo statevector " + _BELL)
    t.append(sv^)

    var ex = Command("expect", "Exact Pauli expectation value(s)", "<file.qasm|-> \"<observable>\" [more observables...]", -1, cmd_expect)
    ex.opt("observable", "PAULI_SUM", "an observable, as an option instead of a positional argument; repeatable")
    ex.opt("params", "JSON", _PARAMS_HELP)
    ex.opt("max-qubits", "N", "refuse wider circuits (default 28)")
    ex.example("qumo expect " + _BELL + " \"Z0 Z1\"")
    ex.example("qumo expect " + _BELL + " --observable \"0.5*Z0 Z1 - X0\" --observable \"X0 X1\"")
    ex.notes = "An observable is a Pauli sum: '0.5*Z0 Z1 - 1.2*X0 + 0.3', or a dense string such as 'ZZI'. Quote it: it contains spaces. One that starts with '--' goes after a lone '--'."
    t.append(ex^)

    var tr = Command("transpile", "Compile for a target: optimize, map, route; prints a metrics summary", "<file.qasm|->", 1, cmd_transpile)
    tr.opt("target", "SPEC", _TARGET_HELP + "; without it: all-to-all, any gate (optimization only)")
    tr.opt("level", "0-3", "optimization preset (default 2)")
    tr.opt("seed", "S", "seed of the randomized passes (default 7, the default of qumo.compiler.transpile)")
    tr.opt("basis", "GATES", "basis gates when no --target names them, e.g. rz,sx,x,cz")
    tr.opt("pipeline", "SPEC", "explicit pass list instead of a level, e.g. 'cancel,merge,route(sabre),basis' (qumo describe passes)")
    tr.opt("ai", "", "hybrid heuristic + learned passes (needs the models of models/ai_transpiler)")
    tr.opt("ai-level", "1-3", "AI preset (default 2), with --ai")
    tr.opt("output", "FILE", "write the compiled OpenQASM 3 to FILE (short form: -o FILE)")
    tr.opt("qasm", "", "print the compiled OpenQASM 3 instead of the summary")
    tr.example("qumo transpile " + _BELL + " --target grid:2x2/sc_cz")
    tr.example("qumo transpile " + _BELL + " --target line:5/rz,sx,x,cz --level 3 -o bell_line.qasm")
    t.append(tr^)

    var em = Command("emulate", "Noisy run on a device model (the target's T1/T2, gate and readout errors)", "<file.qasm|-> --target SPEC", 1, cmd_emulate)
    em.opt("target", "SPEC", "required. " + _TARGET_HELP)
    em.opt("shots", "N", "number of shots (default 1024)")
    em.opt("seed", "S", "random seed (default 1234)")
    em.opt("compile", "", "always compile for the target first")
    em.opt("no-compile", "", "never compile: a circuit that does not fit the target is an error (QM-E615)")
    em.opt("level", "0-3", "optimization preset when compiling (default 2)")
    em.example("qumo emulate " + _BELL + " --target grid:2x2/sc_cz --shots 2000")
    em.notes = "A circuit that already fits the target runs as written. One that does not is compiled first (the output says so) unless --no-compile is given. Custom calibration data: 'qumo tool qumo_emulate --help'."
    t.append(em^)

    var va = Command("validate", "Parse + lint OpenQASM, programs included; reports every error it can find", "<file.qasm|->", 1, cmd_validate)
    va.example("qumo validate " + _BELL)
    va.notes = "Exit code 1 when the file is invalid. After a failing gate call, measurement or assignment the check goes on to the next statement; it stops at an error in a declaration or a block header, since later lines depend on those."
    t.append(va^)

    var info = Command("info", "Circuit or program summary: depth, gate counts, T-count, flags, lint", "<file.qasm|->", 1, cmd_info)
    info.opt("qasm", "", "also print canonical OpenQASM 3")
    info.opt("instructions", "", "include the instruction list (with --json)")
    info.example("qumo info " + _BELL)
    t.append(info^)

    var draw = Command("draw", "Text drawing of a circuit, or of a program by blocks", "<file.qasm|->", 1, cmd_draw)
    draw.opt("width", "N", "fold the drawing at N columns (default 80; 0 = never)")
    draw.opt("unicode", "", "box-drawing characters instead of ASCII")
    draw.example("qumo draw " + _BELL + " --unicode")
    t.append(draw^)

    var conv = Command("convert", "Re-emit canonical OpenQASM 3, portable OpenQASM 2, or circuit / program JSON", "<file.qasm|->", 1, cmd_convert)
    conv.opt("to", "qasm3|qasm2|json", "output format (default qasm3)")
    conv.opt("params", "JSON", "bind 'input float' parameters before converting (qasm2 needs them bound)")
    conv.opt("output", "FILE", "write to FILE instead of stdout (short form: -o FILE)")
    conv.example("qumo convert " + _BELL + " --to qasm2 -o bell2.qasm")
    t.append(conv^)

    var vq = Command("vqe", "Ground-state energy of a Pauli-sum Hamiltonian (hardware-efficient ansatz)", "\"<hamiltonian>\"", 1, cmd_vqe)
    vq.opt("layers", "N", "ansatz layers (default 2)")
    vq.opt("entanglement", "linear|circular|full|none", "entangling pattern (default linear)")
    vq.opt("optimizer", "bfgs|adam|gd|spsa|nelder-mead", "optimizer (default bfgs)")
    vq.opt("gradient", "adjoint|parameter-shift|finite-diff", "gradient method (default adjoint)")
    vq.opt("maxiter", "N", "iteration limit (default 200)")
    vq.opt("qubits", "N", "ansatz width, when wider than the Hamiltonian")
    vq.opt("seed", "S", "seed of the initial point (default 7)")
    vq.example("qumo vqe \"-1.05*II + 0.39*ZI - 0.39*IZ - 0.01*ZZ + 0.18*XX\" --layers 1")
    vq.notes = "Other algorithms are tools: qumo tool qumo_qaoa_maxcut --help, qumo tool qumo_build_circuit --help (grover, qft, ghz, ...: pipe the circuit into 'qumo run -')."
    t.append(vq^)

    var desc = Command("describe", "Self-description: gates | gate <name> | errors | error <code> | capabilities | passes | models", "[gates|gate <name>|errors|error <code>|capabilities|passes|models [model_dir]]", 2, cmd_describe)
    desc.example("qumo describe gate cx")
    desc.example("qumo describe error QM-E615")
    t.append(desc^)

    var tool = Command("tool", "Call any registered MCP tool with JSON arguments", "list | <name> ['<json-args>'|-]", 2, cmd_tool)
    tool.example("qumo tool list")
    tool.example("qumo tool qumo_transpile --help")
    tool.example("qumo tool qumo_run_qasm '{\"file\": \"" + _BELL + "\", \"shots\": 100}'")
    tool.notes = "The output is always JSON. A tool that takes 'qasm' also accepts \"file\": \"path\" here (the CLI reads the file; the MCP server does not). The 'qumo_' prefix may be left out: qumo tool draw '{...}'."
    t.append(tool^)

    t.append(Command("mcp", "Run the MCP server on stdio (JSON-RPC 2.0, MCP 2025-06-18)", "", 0, cmd_mcp))
    t.append(Command("version", "Print version", "", 0, cmd_version))
    t.append(Command("help", "Show help for all commands, one command or one tool", "[command | tool <name>]", 2, cmd_noop))
    return t^


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
def _opt_row(left: String, help: String) -> String:
    var w = wrap_text(help, 100, 26)
    if left.byte_length() > 22:
        return "  " + left + "\n" + w + "\n"
    return "  " + rpad(left, 24) + String(w[byte=26:]) + "\n"


def command_help(c: Command) -> String:
    """Help of one command, generated from its declaration in `command_table()`."""
    var has_opts = len(c.options) > 0
    var s = "qumo " + c.name + " - " + c.summary + "\n\nusage: qumo " + c.name
    if c.synopsis.byte_length() > 0:
        s += " " + c.synopsis
    s += " [options]\n\noptions:\n"
    for o in c.options:
        s += _opt_row("--" + o.name + (" " + o.metavar if o.metavar.byte_length() > 0 else String()), o.help)
    if c.name != "mcp" and c.name != "help":
        s += _opt_row("--json", "print the result as JSON" + (" (always on for 'tool')" if c.name == "tool" else String(", exactly the MCP tool's structuredContent")))
        s += _opt_row("--pretty", "indent the JSON")
    s += _opt_row("-h, --help", "show this help")
    if has_opts:
        s += "\nOptions are written --name value or --name=value. An option that is not listed is an error.\n"
    if len(c.examples) > 0:
        s += "\nexamples:\n"
        for e in c.examples:
            s += "  " + e + "\n"
    if c.notes.byte_length() > 0:
        s += "\n" + wrap_text(c.notes, 100, 0) + "\n"
    if c.synopsis.find("file.qasm") >= 0:
        s += "\nThe file is OpenQASM 3.0 or 2.0; '-' reads it from stdin."
    return String(s.strip())


def global_help(table: List[Command]) -> String:
    var s = String("qumo " + String(QUMO_VERSION) + " - quantum toolkit in Mojo: simulate, compile, analyze\n\nusage: qumo <command> [args] [options] [--json] [--pretty]\n")
    s += "\nfirst steps:"
    var steps: List[String] = ["qumo run " + _BELL + " --shots 1000", "qumo draw " + _BELL, "qumo transpile " + _BELL + " --target grid:2x2/sc_cz"]
    var what: List[String] = ["sample a Bell pair", "see the circuit", "compile it for a device"]
    for i in range(len(steps)):
        s += "\n  " + rpad(steps[i], 72) + what[i]
    s += "\n\ncommands:"
    for c in table:
        s += "\n  " + rpad(c.name, 12) + " " + c.summary
    s += "\n\n'qumo <command> --help' lists a command's options; 'qumo tool <name> --help' documents a tool."
    s += "\nInput files may be OpenQASM 3.0 or 2.0; use '-' for stdin. Exit codes: 0 ok, 1 failed, 2 usage."
    s += "\nErrors carry stable codes (QM-Exxx) with line:col and a hint: 'qumo describe errors'."
    return s^


def _tool_help(name: String) raises -> String:
    var reg = full_registry()
    var i = _find_tool(reg, name)
    if i < 0:
        raise Error(_unknown_tool(reg, name))
    return tool_help_text(reg.tools[i])


def _find_command(table: List[Command], name: String) -> Int:
    for i in range(len(table)):
        if table[i].name == name:
            return i
    return -1


def _unknown_command(table: List[Command], name: String) -> Error:
    var names = List[String]()
    for c in table:
        names.append(c.name)
    var near = did_you_mean(name, names)
    var hint = String("did you mean '" + near + "'? ") if near.byte_length() > 0 else String()
    return Error("QM-E912: unknown command '" + name + "'. Hint: " + hint + "Commands are " + String(", ").join(names) + " (see 'qumo help').")


def _help_for(table: List[Command], topics: List[String]) raises -> String:
    """`qumo help`, `qumo help <command>`, `qumo help tool <name>`, `qumo help <tool>`."""
    if len(topics) == 0:
        return global_help(table)
    if topics[0] == "tool" and len(topics) > 1 and topics[1] != "list":
        return _tool_help(topics[1])
    var i = _find_command(table, topics[0])
    if i >= 0:
        return command_help(table[i])
    var reg = full_registry()
    if _find_tool(reg, topics[0]) >= 0:
        return tool_help_text(reg.tools[_find_tool(reg, topics[0])])
    raise _unknown_command(table, topics[0])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _takes_value_somewhere(table: List[Command], name: String) -> Bool:
    for c in table:
        if c.option_kind(name) == 2:
            return True
    return False


def _command_index(table: List[Command], args: List[String]) -> Int:
    """Index in `args` of the command name: the first token that is neither an
    option nor the value of a preceding `--option value`; -1 when there is none."""
    var i = 0
    while i < len(args):
        var s = args[i]
        if s == "--":
            return -1
        if s.startswith("--"):
            var body = String(s[byte=2:])
            if body.find("=") < 0 and _takes_value_somewhere(table, body):
                i += 1
        elif s == "-o":
            i += 1
        elif not s.startswith("-"):
            return i
        i += 1
    return -1


def _has_flag(args: List[String], long: String, short: String) -> Bool:
    for s in args:
        if s == "--":
            return False
        if s == long or (short.byte_length() > 0 and s == short):
            return True
    return False


def _dispatch(table: List[Command], args: List[String]) raises -> CliOut:
    var ci = _command_index(table, args)
    var wants_help = _has_flag(args, "--help", "-h")
    if ci < 0:
        if _has_flag(args, "--version", "") and not wants_help:
            return _ok("qumo " + String(QUMO_VERSION) + " (Mojo 1.2, MCP " + String(MCP_PROTOCOL_VERSION) + ")")
        for s in args:
            if s.startswith("-") and s != "--help" and s != "-h" and s != "--json" and s != "--pretty":
                raise Error("QM-E921: no command given before option '" + s + "'. Hint: usage: qumo <command> [args] [options]; see 'qumo help'.")
        return _ok(global_help(table))
    var name = args[ci]
    var rest = List[String]()
    for i in range(len(args)):
        if i != ci:
            rest.append(args[i])
    if name == "help" or wants_help:
        # Help never fails on the other options of the line: it explains them.
        var topics = List[String]()
        if name != "help" or wants_help:
            topics.append(name)
        for s in rest:
            if not s.startswith("-"):
                topics.append(s)
        return _ok(_help_for(table, topics))
    var k = _find_command(table, name)
    if k < 0:
        raise _unknown_command(table, name)
    var a = parse_args(table[k], rest)
    var h = table[k].handler
    try:
        return h(a)
    except e:
        return fail(a.json, a.pretty, String(e))


def run_cli(args: List[String]) -> CliOut:
    """Run one `qumo` invocation (`args` without the program name). Never
    raises and never exits: the caller prints `out` / `err` and uses `code`."""
    var table = command_table()
    try:
        return _dispatch(table, args)
    except e:
        return fail(_has_flag(args, "--json", ""), _has_flag(args, "--pretty", ""), String(e))
