"""Write wire_tool_names.json: the tool names each Python agent sends in its first model request.

Dev-time only (owner T, DESIGN.md 8.2). Builds every agent of the registry with deepagents and
a fake chat model that records the tools bound to its first call (the r9-native-harness
capture.py technique), with the real QUMO MCP server (build/qumo mcp) and without alphaXiv.
Also records the QUMO MCP server's tool order, which test_tools_catalog uses as its fake list.

    agents/.venv/bin/python -I agents/mojo/tests/fixtures/tools/gen_wire_names.py
"""
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "agents"))

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from qumo_agents import graph, registry  # noqa: E402
from qumo_agents.config import Settings  # noqa: E402

CAPTURE: dict[str, list[str]] = {}


class Capturing(GenericFakeChatModel):
    agent: str = ""

    def bind_tools(self, tools, **kwargs):
        CAPTURE.setdefault(self.agent, [convert_to_openai_tool(t)["function"]["name"] for t in tools])
        return self


async def main() -> None:
    shared = await graph.load_shared_tools(need_alphaxiv=False)
    settings = Settings.load()
    for name in registry.AGENTS:
        agent = graph.build_agent(
            registry.get(name), settings, shared,
            lambda m, n=name: Capturing(agent=n, messages=iter([AIMessage("ok")] * 3)))
        await agent.ainvoke({"messages": [{"role": "user", "content": "TASK"}]})
    out = {"schema": "qumo.test_wire_tool_names.v1",
           "mcp_server_order": [t.name for t in shared["qumo"]],
           "agents": {n: CAPTURE[n] for n in registry.AGENTS}}
    (HERE / "wire_tool_names.json").write_text(json.dumps(out, indent=1) + "\n")


asyncio.run(main())
