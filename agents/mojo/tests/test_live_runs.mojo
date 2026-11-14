"""Live checks 8.3.3 to 8.3.6: real model runs of qumo-operator, execution, research-watch and
code-review. Needs QUMO_LIVE=1, OPENROUTER_API_KEY and network access; SKIPs otherwise.
research-watch also uses ALPHAXIV_API_KEY and GITHUB_TOKEN when they are set; run it once with
and once without ALPHAXIV_API_KEY (8.3.5) by changing the shell's environment.

Each run prints its wall time and the statistics line (turns, tokens, tool calls) so the
numbers can be pasted into the PR description (8.3.11).

Run: make agents-mojo-live
"""

from std.testing import assert_true

from live import skip_reason, report_skip, run_live, fail_text

comptime NAME = "test_live_runs"
comptime RUN_MS = 900000


def _last_line(text: String) -> String:
    var lines = String(text.strip()).split("\n")
    return String(lines[len(lines) - 1])


def _run(label: String, args: List[String]) raises -> String:
    var r = run_live(args, "", RUN_MS)
    assert_true(r.exit_code == 0, label + " failed, " + fail_text(r))
    assert_true(r.stdout.byte_length() > 0, label + " printed no answer")
    assert_true(r.stderr.find("sk-or-") < 0, label + " leaked a key")
    print("  " + label + ": " + _last_line(r.stderr))
    return r.stdout


def test_operator_ghz() raises:
    var out = _run("operator ghz", [
        "run", "qumo-operator", "-q",
        "Build a 3-qubit GHZ, simulate 1000 shots with seed 7, check the counts"])
    assert_true(out.find("000") >= 0 and out.find("111") >= 0, out)


def test_operator_qm_e521() raises:
    var out = _run("operator QM-E521", [
        "run", "qumo-operator", "-q",
        "This OpenQASM fails with QM-E521: OPENQASM 3.0; include \"stdgates.inc\"; qubit[3] q; bit[3] c; "
        "h q[0]; cx q[0], q[1]; cx q[1], q[2]; c = measure q; h q[7];\n"
        "Find and fix the error, simulate with 2000 shots and seed 11, and check the counts."])
    assert_true(out.find("000") >= 0 and out.find("111") >= 0, out)


def test_execution_20_qubits() raises:
    var out = _run("execution 20q", [
        "run", "execution", "-q",
        "Run a 20-qubit GHZ circuit with 100 shots: estimate the resources, choose a simulator and report the counts"])
    assert_true(out.find("20") >= 0, out)


def test_research_watch() raises:
    _ = _run("research-watch compiler", ["run", "research-watch", "-q", "--module", "compiler",
                                         "What changed for the compiler in the last 30 days? Include sdk_releases for qiskit."])


def test_code_review() raises:
    _ = _run("code-review", ["run", "code-review", "-q",
                             "Review qumo/ai/json.mojo for performance and security. Use run_check and write a report."])


def main() raises:
    var why = skip_reason(NAME, ["OPENROUTER_API_KEY"])
    if why.byte_length() > 0:
        report_skip(NAME, why)
        return
    test_operator_ghz()
    test_operator_qm_e521()
    test_execution_20_qubits()
    test_research_watch()
    test_code_review()
    print(NAME + " OK")
