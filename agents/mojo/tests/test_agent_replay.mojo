"""The agent loop offline: a replayed model asks for qumo_validate_qasm, then
qumo_run_qasm, then answers. Both calls reach QUMO's real MCP server."""

from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import parse_json
from qumo_agent.config import AgentConfig
from qumo_agent.llm import LlmClient
from qumo_agent.agent import Agent, report_slug


def main() raises:
    var cfg = AgentConfig(".", "qumo-operator", "")
    var llm = LlmClient(replay_file="agents/mojo/tests/fixtures/replay_bell.jsonl")
    var agent = Agent(cfg^, llm^, verbose=False)
    assert_true(len(agent.mcp_names) >= 24)  # qumo-operator gets every MCP tool
    var r = agent.run("Run a Bell state with seed 7")
    assert_equal(r.answer, "Bell state: only 00 and 11 were measured, as theory predicts.")
    assert_equal(r.stats.turns, 3)
    assert_equal(r.stats.tool_calls, 2)
    assert_equal(r.stats.mcp_calls, 2)
    assert_equal(r.stats.prompt_tokens, 600)
    assert_equal(r.stats.completion_tokens, 60)
    assert_false(r.stats.stopped_at_limit)

    # What the model saw: the system prompt from agents/prompts, guided tools, real results.
    assert_equal(len(agent.llm.requests), 3)
    var first = parse_json(agent.llm.requests[0])
    assert_true(first.get("messages").at(0).get_string("content", "").find("Role: QUMO operator.") >= 0)
    var tools = first.get("tools")
    var saw_guided = False
    var saw_report = False
    for i in range(len(tools)):
        var f = tools.at(i).get("function")
        if f.get_string("name", "") == "qumo_validate_qasm":
            saw_guided = f.get_string("description", "").startswith("Use when you wrote or received OpenQASM")
        if f.get_string("name", "") == "write_report":
            saw_report = True
    assert_true(saw_guided and saw_report)
    var last = parse_json(agent.llm.requests[2]).get("messages")
    var tool_msg = last.at(len(last) - 1)
    assert_equal(tool_msg.get_string("role", ""), "tool")
    assert_equal(tool_msg.get_string("tool_call_id", ""), "call_2")
    var counts = parse_json(tool_msg.get_string("content", "")).get("counts")
    for k in counts.keys():
        assert_true(k == "00" or k == "11", "unexpected key " + k)
    agent.close()

    # execution gets only its subset.
    var cfg2 = AgentConfig(".", "execution", "")
    var agent2 = Agent(cfg2^, LlmClient(replay_file="agents/mojo/tests/fixtures/replay_bell.jsonl"), verbose=False)
    assert_equal(len(agent2.mcp_names), 11)
    agent2.close()

    assert_equal(report_slug("GHZ check: 3 qubits / seed=11!"), "GHZ-check-3-qubits-seed-11")
    assert_equal(report_slug("***"), "report")
    var refused = False
    try:
        _ = AgentConfig(".", "research-watch", "")
    except e:
        refused = String(e).find("Python-only") >= 0
    assert_true(refused)
    print("test_agent_replay OK")
