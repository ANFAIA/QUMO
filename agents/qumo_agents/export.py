"""Export the agent prompts, specs and MCP tool guides to agents/prompts/.

The Python agents keep their prompts in prompts.py and registry.py. The Mojo agents
(agents/mojo/) cannot import Python, so they read the same text from these files.
`python -m qumo_agents export-prompts` writes them; tests/test_agents.py fails when
they are stale, so the two implementations never drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

from qumo_agents import registry
from qumo_agents.config import AGENTS_DIR
from qumo_agents.tools import qumo_mcp

PROMPTS_DIR = AGENTS_DIR / "prompts"


def rendered() -> dict[Path, str]:
    out: dict[Path, str] = {}
    specs = {}
    for spec in registry.AGENTS.values():
        out[PROMPTS_DIR / f"{spec.name}.md"] = spec.prompt
        specs[spec.name] = {
            "description": spec.description,
            "tool_groups": list(spec.tool_groups),
            "mcp_tools": "all" if spec.mcp_tools is None else list(spec.mcp_tools),
            "subagents": list(spec.subagents),
            "writes_code": spec.writes_code,
        }
    out[PROMPTS_DIR / "specs.json"] = json.dumps({"schema": "qumo.agent_specs.v1", "agents": specs}, indent=1) + "\n"
    guide = {name: {"use_when": u, "instead_or_first": p, "watch_out": w}
             for name, (u, p, w) in qumo_mcp.TOOL_GUIDE.items()}
    out[PROMPTS_DIR / "tool_guide.json"] = json.dumps({"schema": "qumo.tool_guide.v1", "tools": guide},
                                                     indent=1) + "\n"
    return out


def stale() -> list[Path]:
    return [p for p, text in rendered().items() if not p.is_file() or p.read_text() != text]


def write() -> list[Path]:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    changed = stale()
    for p, text in rendered().items():
        if p in changed:
            p.write_text(text)
    return changed
