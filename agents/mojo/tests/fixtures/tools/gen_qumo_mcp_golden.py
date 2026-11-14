"""Write qumo_mcp_golden.json: the QUMO MCP tools exactly as Python sends them to the model.

Dev-time only (owner T, DESIGN.md 8.2). Loads build/qumo mcp through langchain-mcp-adapters
with the guided descriptions (qumo_agents.tools.qumo_mcp.load_qumo_tools_async), converts each
tool with langchain-core's convert_to_openai_tool and records the "function" object. Also
records each agent's QUMO tool names (first-request order). test_tools_qumo_mcp compares
QumoMcp.specs_for with it. Regenerate when build/qumo's tool descriptions change.

    agents/.venv/bin/python -I agents/mojo/tests/fixtures/tools/gen_qumo_mcp_golden.py
"""
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "agents"))

from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from qumo_agents import registry  # noqa: E402
from qumo_agents.tools import qumo_mcp  # noqa: E402


async def main() -> None:
    tools = await qumo_mcp.load_qumo_tools_async()
    funcs = [convert_to_openai_tool(t)["function"] for t in tools]
    per_agent = {}
    for name in registry.AGENTS:
        spec = registry.get(name)
        if spec.mcp_tools is None:
            per_agent[name] = [f["name"] for f in funcs]
        else:
            per_agent[name] = [f["name"] for f in funcs if f["name"] in spec.mcp_tools]
    out = {"schema": "qumo.test_qumo_mcp_golden.v1", "tools": funcs, "agents": per_agent}
    (HERE / "qumo_mcp_golden.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


asyncio.run(main())
