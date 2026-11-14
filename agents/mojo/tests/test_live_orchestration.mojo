"""Live checks 8.3.7 to 8.3.10: the orchestrator with two concurrent subagents, autoresearch in
a sandbox worktree, the mojo-toolsmith flow with review and use, and `call` (a child over MCP).
Needs QUMO_LIVE=1, OPENROUTER_API_KEY and network access; SKIPs otherwise.

autoresearch creates a git worktree and branch; it runs only when QUMO_LIVE_AUTORESEARCH=1.
The toolsmith test builds gen_bitstring_entropy into agents/mojo/tools/ of a sandbox worktree
and reviews it into the tool store (build/agent-tools or $QUMO_AGENT_TOOLS_DIR): use
QUMO_LIVE_TOOLSMITH=1, and remove the tool afterwards with `tools remove gen_bitstring_entropy`.

Run: make agents-mojo-live
"""

from std.os import getenv
from std.testing import assert_true

from live import skip_reason, report_skip, run_live, fail_text

comptime NAME = "test_live_orchestration"
comptime RUN_MS = 1200000


def _opt_in(name: String) -> Bool:
    return getenv(name, "") == "1"


def test_orchestrator_two_subagents() raises:
    var r = run_live(["run", "orchestrator", "-q",
                      "In one step, ask the research-watch subagent for the newest Qiskit SDK release and "
                      "the compiler-research subagent for one recent routing paper, at the same time; cite both."], "", RUN_MS)
    assert_true(r.exit_code == 0, "orchestrator failed, " + fail_text(r))
    assert_true(r.stdout.byte_length() > 0, "no answer")
    print("  orchestrator: " + String(r.stderr.byte_length()) + " bytes of logs; check the two subagent children overlap")


def test_autoresearch() raises:
    var r = run_live(["run", "autoresearch", "-q", "--module", "ir",
                      "Make one small, safe, measured improvement to a comment or docstring in qumo/ir and stop."], "", RUN_MS)
    assert_true(r.exit_code == 0, "autoresearch failed, " + fail_text(r))
    print("  autoresearch ok; clean up with `git worktree list`, `git worktree remove` and `git branch -D`")


def test_toolsmith_flow() raises:
    var r = run_live(["run", "mojo-toolsmith", "-q",
                      "Build gen_bitstring_entropy: Shannon entropy in bits of a counts JSON"], "", RUN_MS)
    assert_true(r.exit_code == 0, "toolsmith failed, " + fail_text(r))
    # `tools review` needs a terminal for the confirmation; the name on stdin is the other path.
    var rv = run_live(["tools", "review", "gen_bitstring_entropy"], "gen_bitstring_entropy\n", 600000)
    assert_true(rv.exit_code == 0, "review failed, " + fail_text(rv))
    var use = run_live(["run", "qumo-operator", "-q",
                        "Use gen_bitstring_entropy on the counts {\"00\": 500, \"11\": 500} and report the entropy"], "", RUN_MS)
    assert_true(use.exit_code == 0, "using the tool failed, " + fail_text(use))
    assert_true(use.stdout.find("1") >= 0, use.stdout)


def test_call_child() raises:
    var r = run_live(["call", "qumo-operator",
                      "Build a Bell pair, simulate 200 shots with seed 3 and report the counts"], "", RUN_MS)
    assert_true(r.exit_code == 0, "call failed, " + fail_text(r))
    assert_true(r.stderr.find("[qumo-agent call]") >= 0, r.stderr)


def main() raises:
    var why = skip_reason(NAME, ["OPENROUTER_API_KEY"])
    if why.byte_length() > 0:
        report_skip(NAME, why)
        return
    test_orchestrator_two_subagents()
    test_call_child()
    if _opt_in("QUMO_LIVE_AUTORESEARCH"):
        test_autoresearch()
    else:
        print("  autoresearch skipped: set QUMO_LIVE_AUTORESEARCH=1 (creates a worktree and a branch)")
    if _opt_in("QUMO_LIVE_TOOLSMITH"):
        test_toolsmith_flow()
    else:
        print("  toolsmith skipped: set QUMO_LIVE_TOOLSMITH=1 (writes into the tool store)")
    print(NAME + " OK")
