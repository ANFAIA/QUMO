"""Tests for the CLI front door (cli/app.mojo, qumo/ai/cli_help.mojo): strict
options, first-class transpile / emulate / vqe, generated help, tool help and
examples, the CLI-only "file" argument, and validate's multi-error report.

Everything runs in process through `run_cli` (no build/qumo needed), from the
repository root, like every other test.
"""

from std.testing import assert_equal, assert_true, assert_false
from std.os import remove
from std.tempfile import gettempdir
from std.time import perf_counter_ns

from cli.app import run_cli, command_table, command_help, CliOut
from qumo.ai.cli_help import did_you_mean, edit_distance, tool_example, tool_help_text, tool_takes, with_file_as_qasm
from qumo.core.json import parse_json, JsonValue
from qumo.ai.registry import full_registry

comptime BELL = "examples/foundations/bell.qasm"


def cli(line: List[String]) -> CliOut:
    return run_cli(line)


def has(text: String, part: String) -> Bool:
    return text.find(part) >= 0


def tmp_path(name: String) raises -> String:
    return String(gettempdir().value()) + "/qumo_test_frontdoor_" + String(perf_counter_ns()) + "_" + name


def write_file(path: String, text: String) raises:
    with open(path, "w") as f:
        f.write(text)


def read_file(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def test_suggestions() raises:
    assert_equal(edit_distance("shot", "shots"), 1)
    assert_equal(edit_distance("rnu", "run"), 1)  # a swap is one edit
    assert_equal(edit_distance("abc", "abc"), 0)
    var opts: List[String] = ["shots", "seed", "params", "max-qubits"]
    assert_equal(did_you_mean("shot", opts), "shots")
    assert_equal(did_you_mean("sead", opts), "seed")
    assert_equal(did_you_mean("max", opts), "max-qubits")  # prefix
    assert_equal(did_you_mean("zzzzzz", opts), "")
    assert_equal(did_you_mean("x", opts), "")


def test_strict_options() raises:
    # the bug this replaces: --shot was ignored and 1024 shots ran with rc 0
    var r = cli(["run", BELL, "--shot", "10"])
    assert_equal(r.code, 2)
    assert_equal(r.out, "")
    assert_true(has(r.err, "QM-E921"), r.err)
    assert_true(has(r.err, "unknown option '--shot'"), r.err)
    assert_true(has(r.err, "did you mean '--shots'?"), r.err)
    # with --json the diagnostic is JSON on stdout
    var j = cli(["run", BELL, "--shot", "10", "--json"])
    assert_equal(j.code, 2)
    assert_equal(j.err, "")
    var e = parse_json(j.out).get("error")
    assert_equal(e.get("code").as_string(), "QM-E921")
    assert_true(has(e.get("hint").as_string(), "--shots"))
    # the right spelling, both syntaxes, and the options before the command
    var ok = cli(["run", BELL, "--shots", "10", "--seed", "7"])
    assert_equal(ok.code, 0)
    assert_true(has(ok.out, "10 shots, seed 7"), ok.out)
    var eq = cli(["run", BELL, "--shots=10", "--json"])
    assert_equal(eq.code, 0)
    assert_equal(parse_json(eq.out).get("shots").as_int(), 10)
    var front = cli(["--shots", "10", "--json", "run", BELL])
    assert_equal(front.code, 0)
    assert_equal(parse_json(front.out).get("shots").as_int(), 10)
    # an option of another command is unknown here
    var other = cli(["draw", BELL, "--shots", "10"])
    assert_equal(other.code, 2)
    assert_true(has(other.err, "unknown option '--shots' for 'qumo draw'"), other.err)
    # value / flag shape
    var missing = cli(["run", BELL, "--shots"])
    assert_equal(missing.code, 2)
    assert_true(has(missing.err, "needs a value"), missing.err)
    var notint = cli(["run", BELL, "--shots", "many"])
    assert_equal(notint.code, 2)
    assert_true(has(notint.err, "expects an integer"), notint.err)
    var flagval = cli(["draw", BELL, "--unicode=yes"])
    assert_equal(flagval.code, 2)
    assert_true(has(flagval.err, "is a flag"), flagval.err)
    var short = cli(["run", BELL, "-s", "10"])
    assert_equal(short.code, 2)
    assert_true(has(short.err, "unknown option '-s'"), short.err)
    var dash_o = cli(["run", BELL, "-o", "x.json"])
    assert_equal(dash_o.code, 2)
    # a positional argument too many (an unquoted value, a typo)
    var extra = cli(["run", BELL, "extra"])
    assert_equal(extra.code, 2)
    assert_true(has(extra.err, "unexpected argument 'extra'"), extra.err)
    # commands
    var cmd = cli(["rnu", BELL])
    assert_equal(cmd.code, 2)
    assert_true(has(cmd.err, "QM-E912") and has(cmd.err, "did you mean 'run'?"), cmd.err)
    var nocmd = cli(["--shots", "10"])
    assert_equal(nocmd.code, 2)
    # failures that are not usage errors keep exit code 1
    var nofile = cli(["run", "no/such/file.qasm"])
    assert_equal(nofile.code, 1)
    assert_true(has(nofile.err, "QM-E920"), nofile.err)


def test_expect() raises:
    var pos = cli(["expect", BELL, "Z0 Z1"])
    assert_equal(pos.code, 0)
    assert_true(pos.out.startswith("<Z0 Z1> = 1"), pos.out)
    var opt = cli(["expect", BELL, "--observable", "Z0 Z1"])
    assert_equal(opt.code, 0)
    assert_equal(opt.out, pos.out)
    var eq = cli(["expect", BELL, "--observable=Z0 Z1"])
    assert_equal(eq.out, pos.out)
    var many = cli(["expect", BELL, "X0 X1", "--observable", "Z0 Z1", "--observable", "-1*Z0", "--json"])
    assert_equal(many.code, 0)
    var res = parse_json(many.out).get("results")
    assert_equal(len(res), 3)
    assert_equal(res.at(0).get("observable").as_string(), "X0 X1")
    assert_equal(res.at(2).get("observable").as_string(), "-1*Z0")
    var neg = cli(["expect", BELL, "-Z0 Z1"])  # a leading minus is an observable, not an option
    assert_equal(neg.code, 0)
    assert_true(neg.out.startswith("<-Z0 Z1> = -1"), neg.out)
    var none = cli(["expect", BELL])
    assert_equal(none.code, 2)
    assert_true(has(none.err, "--observable"), none.err)


def test_transpile() raises:
    var r = cli(["transpile", BELL, "--target", "grid:2x2/sc_cz"])
    assert_equal(r.code, 0)
    assert_true(has(r.out, "grid_2x2 (4 qubits"), r.out)
    assert_true(has(r.out, "level 2, seed 7"), r.out)
    for part in ["pipeline:", "before", "after", "two-qubit gates", "depth", "swaps inserted", "estimated fidelity", "layout: initial"]:
        assert_true(has(r.out, part), "missing '" + String(part) + "' in:\n" + r.out)
    # --json is the tool's structuredContent
    var j = cli(["transpile", BELL, "--target", "grid:2x2/sc_cz", "--level", "3", "--seed", "11", "--json"])
    assert_equal(j.code, 0)
    var v = parse_json(j.out)
    assert_equal(v.get("target").get("name").as_string(), "grid_2x2")
    assert_true(v.has("gates_after") and v.has("report") and v.has("qasm"))
    # the same call through the generic tool gives the same circuit
    var t = cli(["tool", "qumo_transpile", "{\"file\": \"" + String(BELL) + "\", \"target\": \"grid:2x2/sc_cz\", \"level\": 3, \"seed\": 11}"])
    assert_equal(parse_json(t.out).get("qasm").as_string(), v.get("qasm").as_string())
    # --qasm prints the circuit, -o writes it
    var q = cli(["transpile", BELL, "--target", "grid:2x2/sc_cz", "--qasm"])
    assert_true(q.out.startswith("OPENQASM 3.0;"), q.out)
    var path = tmp_path("bell_grid.qasm")
    var o = cli(["transpile", BELL, "--target", "grid:2x2/sc_cz", "-o", path])
    assert_equal(o.code, 0)
    assert_true(has(o.out, "written to " + path), o.out)
    assert_equal(read_file(path), q.out)
    # the compiled file fits the target: emulate runs it as written
    var em = cli(["emulate", path, "--target", "grid:2x2/sc_cz", "--no-compile", "--shots", "200"])
    assert_equal(em.code, 0)
    assert_false(has(em.out, "compiled first"), em.out)
    remove(path)
    # errors: bad target spec is the tool's error (exit 1), bad option a usage error (exit 2)
    var bad = cli(["transpile", BELL, "--target", "nonsense:9"])
    assert_equal(bad.code, 1)
    assert_true(has(bad.err, "QM-E"), bad.err)
    var typo = cli(["transpile", BELL, "--targt", "grid:2x2/sc_cz"])
    assert_equal(typo.code, 2)
    assert_true(has(typo.err, "did you mean '--target'?"), typo.err)
    # a program with runtime control flow compiles block by block
    var prog = cli(["transpile", "tests/fixtures/control_flow/qiskit_while_break.qasm", "--target", "line:4/rz,sx,x,cz"])
    assert_equal(prog.code, 0)
    assert_true(has(prog.out, "blocks verified"), prog.out)


def test_emulate() raises:
    # not compiled for the target: compiled first, and the output says so
    var r = cli(["emulate", BELL, "--target", "grid:2x2/sc_cz", "--shots", "300", "--seed", "5"])
    assert_equal(r.code, 0)
    assert_true(has(r.out, "on grid_2x2"), r.out)
    assert_true(has(r.out, "300 shots, seed 5"), r.out)
    assert_true(has(r.out, "compiled first"), r.out)
    var j = cli(["emulate", BELL, "--target", "grid:2x2/sc_cz", "--shots", "300", "--seed", "5", "--json"])
    assert_equal(j.code, 0)
    var v = parse_json(j.out)
    assert_true(v.get("compiled").as_bool())
    assert_equal(v.get("shots").as_int(), 300)
    var total = 0
    var counts = v.get("counts")
    for i in range(len(counts)):
        total += counts.at(i).as_int()
    assert_equal(total, 300)
    # --compile gives the same result as the automatic compile
    var c = cli(["emulate", BELL, "--target", "grid:2x2/sc_cz", "--shots", "300", "--seed", "5", "--compile", "--json"])
    assert_equal(c.out, j.out)
    # --no-compile: QM-E615 with the exact commands that fix it
    var n = cli(["emulate", BELL, "--target", "grid:2x2/sc_cz", "--no-compile"])
    assert_equal(n.code, 1)
    assert_true(has(n.err, "QM-E615"), n.err)
    assert_true(has(n.err, "qumo transpile " + String(BELL) + " --target grid:2x2/sc_cz -o compiled.qasm"), n.err)
    # the tool itself (MCP) names compile=true in its hint
    var t = cli(["tool", "qumo_emulate", "{\"file\": \"" + String(BELL) + "\", \"target\": \"grid:2x2/sc_cz\"}"])
    assert_equal(t.code, 1)
    var e = parse_json(t.out).get("error")
    assert_equal(e.get("code").as_string(), "QM-E615")
    assert_true(has(e.get("hint").as_string(), "\"compile\": true"), e.get("hint").as_string())
    # usage
    var nt = cli(["emulate", BELL])
    assert_equal(nt.code, 2)
    assert_true(has(nt.err, "--target"), nt.err)
    var both = cli(["emulate", BELL, "--target", "line:2", "--compile", "--no-compile"])
    assert_equal(both.code, 2)


def test_vqe() raises:
    var r = cli(["vqe", "-1.05*II + 0.39*ZI - 0.39*IZ - 0.01*ZZ + 0.18*XX", "--layers", "1"])
    assert_equal(r.code, 0)
    assert_true(r.out.startswith("energy -1.8"), r.out)
    assert_true(has(r.out, "converged") and has(r.out, "hardware_efficient(n=2, layers=1, linear)"), r.out)
    var j = cli(["vqe", "Z0 + Z1", "--layers", "1", "--optimizer", "adam", "--json"])
    assert_equal(j.code, 0)
    assert_equal(parse_json(j.out).get("optimizer").as_string(), "adam")
    assert_equal(cli(["vqe"]).code, 2)


def test_help() raises:
    var g = cli(["--help"])
    assert_equal(g.code, 0)
    assert_true(has(g.out, "first steps:"), g.out)
    assert_true(has(g.out, "qumo run " + String(BELL)), g.out)
    assert_true(has(g.out, "qumo transpile " + String(BELL) + " --target grid:2x2/sc_cz"), g.out)
    assert_equal(cli([]).out, g.out)
    assert_equal(cli(["help"]).out, g.out)
    # every command: listed in the global help, has --help, and its help names every option it accepts
    var table = command_table()
    for c in table:
        assert_true(has(g.out, "\n  " + c.name + " "), "command " + c.name + " not in global help")
        var h = cli([c.name, "--help"])
        assert_equal(h.code, 0)
        assert_equal(h.err, "")
        assert_true(h.out.startswith("qumo " + c.name + " - "), h.out)
        assert_true(has(h.out, "usage: qumo " + c.name), h.out)
        assert_equal(h.out, command_help(c) + "\n")
        assert_equal(cli(["help", c.name]).out, h.out)
        for o in c.options:
            assert_true(has(h.out, "--" + o.name), "option --" + o.name + " missing in help of " + c.name)
    # --help wins over anything else on the line, typos included
    var hh = cli(["run", BELL, "--shot", "10", "-h"])
    assert_equal(hh.code, 0)
    assert_true(hh.out.startswith("qumo run - "), hh.out)
    # the verbs asked for exist
    for name in ["run", "transpile", "emulate", "expect", "statevector", "draw", "validate"]:
        var found = False
        for c in table:
            if c.name == String(name):
                found = True
        assert_true(found, "missing command " + String(name))
    var unk = cli(["help", "transpille"])
    assert_equal(unk.code, 2)
    assert_true(has(unk.err, "did you mean 'transpile'?"), unk.err)
    assert_true(cli(["--version"]).out.startswith("qumo "))
    assert_equal(cli(["version"]).out, cli(["--version"]).out)


def test_tool_help_and_examples() raises:
    var reg = full_registry()
    var l = cli(["tool", "list"])
    assert_equal(l.code, 0)
    for t in reg.tools:
        assert_true(has(l.out, t.name), t.name)
        # readable help, three ways in
        var h = cli(["tool", t.name, "--help"])
        assert_equal(h.code, 0)
        assert_true(h.out.startswith(t.name + " - " + t.title), h.out)
        assert_equal(h.out, tool_help_text(t) + "\n")
        assert_equal(cli(["help", "tool", t.name]).out, h.out)
        assert_equal(cli(["help", t.name]).out, h.out)
        # every argument of the schema is listed
        var props = parse_json(t.input_schema).get("properties")
        for k in props.keys():
            assert_true(has(h.out, "\n  " + k + " "), "argument " + k + " missing in help of " + t.name)
        if tool_takes(t, "qasm"):
            assert_true(has(h.out, "\n  file "), "file missing in help of " + t.name)
        # one example per tool, and it runs as written
        var ex = tool_example(t.name)
        assert_true(ex.byte_length() > 0, "no example for " + t.name + " (qumo/ai/cli_help.mojo tool_example)")
        assert_true(has(h.out, "example:\n  qumo tool " + t.name + " '" + ex + "'"), h.out)
        var r = cli(["tool", t.name, ex])
        assert_equal(r.code, 0, "example of " + t.name + " failed: " + r.out)
        assert_true(parse_json(r.out).is_object())
    var rq = cli(["tool", "qumo_run_qasm", "--help"])
    for part in ["shots", "integer", "default 1024", "range 1 to 10000000", "array | object"]:
        assert_true(has(rq.out, part), "missing '" + String(part) + "' in:\n" + rq.out)
    var ex = cli(["tool", "qumo_expectation", "--help"])
    assert_true(has(ex.out, "required"), ex.out)
    var dr = cli(["tool", "qumo_draw", "--help"])
    assert_true(has(dr.out, "one of 'ascii', 'unicode'"), dr.out)
    assert_true(has(cli(["tool", "qumo_gpu_info", "--help"]).out, "arguments: none"))


def test_tool_file_argument() raises:
    var src = read_file(BELL)
    var by_file = cli(["tool", "qumo_circuit_info", "{\"file\": \"" + String(BELL) + "\"}"])
    assert_equal(by_file.code, 0)
    var w = parse_json("{\"file\": \"x\", \"include_lint\": false}")
    var by_qasm = cli(["tool", "qumo_circuit_info", with_file_as_qasm(parse_json("{\"file\": \"x\"}"), src)])
    assert_equal(by_file.out, by_qasm.out)
    var merged = parse_json(with_file_as_qasm(w, "h q[0];"))
    assert_false(merged.has("file"))
    assert_equal(merged.get("qasm").as_string(), "h q[0];")
    assert_false(merged.get("include_lint").as_bool())
    # the 'qumo_' prefix is optional
    var short = cli(["tool", "circuit_info", "{\"file\": \"" + String(BELL) + "\"}"])
    assert_equal(short.out, by_file.out)
    # errors are JSON on stdout
    var both = cli(["tool", "qumo_draw", "{\"file\": \"" + String(BELL) + "\", \"qasm\": \"x\"}"])
    assert_equal(both.code, 1)
    assert_equal(parse_json(both.out).get("error").get("code").as_string(), "QM-E913")
    var nofile = cli(["tool", "qumo_draw", "{\"file\": \"no/such.qasm\"}"])
    assert_equal(nofile.code, 1)
    assert_equal(parse_json(nofile.out).get("error").get("code").as_string(), "QM-E920")
    var badjson = cli(["tool", "qumo_draw", "{file: 1}"])
    assert_equal(badjson.code, 1)
    assert_equal(parse_json(badjson.out).get("error").get("code").as_string(), "QM-E900")
    var unk = cli(["tool", "qumo_run_qsm", "{}"])
    assert_equal(unk.code, 2)
    var ue = parse_json(unk.out).get("error")
    assert_equal(ue.get("code").as_string(), "QM-E912")
    assert_true(has(ue.get("hint").as_string(), "did you mean 'qumo_run_qasm'?"))
    # a tool without 'qasm' does not get the file treatment: 'file' is its own unknown argument or ignored
    assert_false(tool_takes(full_registry().tools[full_registry().find("qumo_vqe")], "qasm"))


comptime TWO_ERRORS = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nbit[2] c;\nh q[0];\nfoo q[0];\ncx q[0], q[1];\ncx q[0], q[5];\nc = measure q;\n"
comptime DECL_ERROR = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2 q;\nh q[0];\nfoo q[1];\n"


def test_validate_all_errors() raises:
    var path = tmp_path("two_errors.qasm")
    write_file(path, TWO_ERRORS)
    var j = cli(["validate", path, "--json"])
    assert_equal(j.code, 1)
    var v = parse_json(j.out)
    assert_false(v.get("valid").as_bool())
    var errs = v.get("errors")
    assert_equal(len(errs), 2, j.out)
    assert_equal(errs.at(0).get("line").as_int(), 6)
    assert_equal(errs.at(1).get("line").as_int(), 8)
    assert_true(errs.at(0).get("code").as_string() != errs.at(1).get("code").as_string(), j.out)
    assert_true(v.get("errors_complete").as_bool())
    var h = cli(["validate", path])
    assert_equal(h.code, 1)
    assert_equal(h.out, "")
    assert_true(has(h.err, "line 6") and has(h.err, "line 8"), h.err)
    assert_true(h.err.endswith("INVALID: 2 error(s)\n"), h.err)
    # an error in a declaration: later lines depend on it, so the report stops there and says so
    write_file(path, DECL_ERROR)
    var d = cli(["validate", path, "--json"])
    var dv = parse_json(d.out)
    assert_equal(len(dv.get("errors")), 1, d.out)
    assert_equal(dv.get("errors").at(0).get("line").as_int(), 3)
    assert_false(dv.get("errors_complete").as_bool())
    var dh = cli(["validate", path])
    assert_true(has(dh.err, "INVALID: 1 error(s); checking stopped"), dh.err)
    remove(path)
    # a valid file is unchanged: exit 0, no errors_complete key
    var ok = cli(["validate", BELL, "--json"])
    assert_equal(ok.code, 0)
    assert_true(parse_json(ok.out).get("valid").as_bool())
    assert_false(parse_json(ok.out).has("errors_complete"))
    assert_true(cli(["validate", BELL]).out.startswith("OK: "))


def test_existing_commands_unchanged() raises:
    # run --json is still exactly the tool's structuredContent
    var a = parse_json(cli(["run", BELL, "--shots", "50", "--seed", "3", "--json"]).out)
    var b = parse_json(cli(["tool", "qumo_run_qasm", "{\"file\": \"" + String(BELL) + "\", \"shots\": 50, \"seed\": 3}"]).out)
    assert_equal(a.get("counts").to_json(), b.get("counts").to_json())
    assert_equal(a.get("seed").as_int(), 3)
    # default seed of run is 1234
    assert_equal(parse_json(cli(["run", BELL, "--json"]).out).get("seed").as_int(), 1234)
    assert_equal(parse_json(cli(["run", BELL, "--json"]).out).get("shots").as_int(), 1024)
    var sv = cli(["statevector", BELL])
    assert_true(has(sv.out, "|00>") and has(sv.out, "|11>"), sv.out)
    assert_true(has(cli(["draw", BELL]).out, "q0"), cli(["draw", BELL]).out)
    assert_true(cli(["convert", BELL, "--to", "qasm2"]).out.startswith("OPENQASM 2.0;"))
    assert_true(has(cli(["info", BELL]).out, "num_qubits"))
    assert_equal(cli(["describe", "gate", "cx", "--json"]).code, 0)
    assert_equal(cli(["describe", "nonsense"]).code, 1)
    var dev = cli(["run", BELL, "--device", "cpu", "--shots", "20", "--json"])
    assert_equal(dev.code, 0)
    assert_equal(parse_json(dev.out).get("device_used").as_string(), "cpu")


def main() raises:
    test_suggestions()
    test_strict_options()
    test_expect()
    test_transpile()
    test_emulate()
    test_vqe()
    test_help()
    test_tool_help_and_examples()
    test_tool_file_argument()
    test_validate_all_errors()
    test_existing_commands_unchanged()
    print("test_cli_frontdoor OK")
