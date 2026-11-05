"""deepagents graphs with a scripted fake model: every agent compiles, and the
qumo-operator really calls QUMO's MCP server. No network, no model API."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from qumo_agents import graph, registry
from qumo_agents.config import Settings
from qumo_agents.tools import qumo_mcp

pytestmark = pytest.mark.skipif(not qumo_mcp.QUMO_BIN.exists(), reason="build/qumo not built")

BELL = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n'


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _shared():
    return asyncio.run(graph.load_shared_tools(need_alphaxiv=False))


def test_every_agent_compiles_with_its_subagents():
    shared = _shared()
    s = Settings.load()
    for spec in registry.AGENTS.values():
        agent = graph.build_agent(spec, s, shared, lambda m: ScriptedModel(messages=iter([AIMessage("ok")])))
        assert agent is not None, spec.name
    names = {t.name for t in graph.tools_for(registry.get("qumo-operator"), shared)}
    assert {"qumo_run_qasm", "qumo_describe", "write_report"} <= names
    assert not any(n.startswith("qumo_") for n in (t.name for t in graph.tools_for(registry.get("code-review"), shared)))
    assert [a.name for a in graph.tree(registry.get("orchestrator"))][0] == "orchestrator"
    assert len(graph.tree(registry.get("orchestrator"))) == 9


def test_operator_runs_a_circuit_through_qumo_mcp():
    script = iter([
        AIMessage("", tool_calls=[{"name": "qumo_validate_qasm", "args": {"qasm": BELL}, "id": "c1"}]),
        AIMessage("", tool_calls=[{"name": "qumo_run_qasm", "args": {"qasm": BELL, "shots": 1000, "seed": 7}, "id": "c2"}]),
        AIMessage("Bell state: only 00 and 11 appear."),
    ])
    out = asyncio.run(graph.run_agent(registry.get("qumo-operator"), "Run a Bell circuit", Settings.load(),
                                      model_factory=lambda m: ScriptedModel(messages=script)))
    assert "00 and 11" in out


def test_mcp_result_reaches_the_model():
    shared = _shared()
    run = next(t for t in shared["qumo"] if t.name == "qumo_run_qasm")
    res = asyncio.run(run.ainvoke({"qasm": BELL, "shots": 400, "seed": 3}))
    text = str(res)
    assert "00" in text and "11" in text and "01\"" not in text
