"""Export the agent prompts, specs, tools and MCP tool guides to agents/prompts/.

The Python agents keep their prompts in prompts.py and registry.py, their modules in
modules.py and their local tools in tools/. The Mojo agents (agents/mojo/) cannot import
Python, so they read the same text from these files. `python -m qumo_agents export-prompts`
writes them; tests/test_agents.py fails when they are stale, so the two implementations
never drift apart.

Files:
* <agent>.md         system prompt of each agent
* specs.json         qumo.agent_specs.v1: tool groups, MCP tools, subagents, generated tools
* tool_guide.json    qumo.tool_guide.v1: the QUMO MCP tool guides
* preamble.md        the preamble shared by every prompt (Mojo-only agents start with it)
* modules.json       qumo.modules.v1: the QUMO modules table
* tools.json         qumo.local_tools.v1: every local tool as LangChain sends it to the model,
                     the form the token counter measures, the tool groups, and the limits of
                     the repository, research and alphaXiv tools
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qumo_agents import modules, prompts, registry
from qumo_agents.config import AGENTS_DIR
from qumo_agents.tools import alphaxiv, qumo_mcp, repo, research

PROMPTS_DIR = AGENTS_DIR / "prompts"


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=1) + "\n"


def specs_json() -> dict:
    specs = {}
    for spec in registry.AGENTS.values():
        specs[spec.name] = {
            "description": spec.description,
            "tool_groups": list(spec.tool_groups),
            "mcp_tools": "all" if spec.mcp_tools is None else list(spec.mcp_tools),
            "subagents": list(spec.subagents),
            "writes_code": spec.writes_code,
            "generated_tools": "all" if spec.generated_tools is None else list(spec.generated_tools),
        }
    return {"schema": "qumo.agent_specs.v1", "agents": specs}


def modules_json() -> dict:
    return {"schema": "qumo.modules.v1", "modules": {
        m.name: {"summary": m.summary, "paths": list(m.paths), "docs": list(m.docs), "tests": list(m.tests),
                 "sdks": list(m.sdks), "arxiv_categories": list(m.arxiv_categories),
                 "arxiv_queries": list(m.arxiv_queries), "mcp_tools": list(m.mcp_tools)}
        for m in modules.MODULES.values()}}


def count_parameters(t) -> dict:
    """The parameters langchain-core's count_tokens_approximately measures for a tool:
    the JSON schema of tool_call_schema without its top-level title and description
    (langchain_core/messages/utils.py). It keeps every property title, so it is longer
    than the wire form."""
    from langchain_core.utils.pydantic import model_json_schema

    schema = t.tool_call_schema
    params = dict(schema) if isinstance(schema, dict) else dict(model_json_schema(schema))
    params.pop("title", None)
    params.pop("description", None)
    return params


def local_tool_entry(t) -> dict:
    """convert_to_openai_tool(t)["function"] without "name", plus "count_parameters"."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    fn = dict(convert_to_openai_tool(t)["function"])
    fn.pop("name")
    fn["count_parameters"] = count_parameters(t)
    return fn


def tools_json() -> dict:
    tools: dict[str, dict] = {}
    for group in registry.TOOL_GROUPS.values():
        for t in group:
            if t.name not in tools:
                tools[t.name] = local_tool_entry(t)
    return {
        "schema": "qumo.local_tools.v1",
        "tools": tools,
        "groups": {g: [t.name for t in ts] for g, ts in registry.TOOL_GROUPS.items()},
        "repo": {
            "skip_dirs": sorted(repo.SKIP_DIRS),
            "secret_names": repo.SECRET_NAMES.pattern,
            "max_read": repo.MAX_READ,
            "max_output": repo.MAX_OUTPUT,
            "run_check_allowed": [[desc, list(argv)] for desc, argv in repo.ALLOWED],
            "arg_ok": repo.ARG_OK.pattern,
        },
        "research": {"user_agent": research.USER_AGENT, "max_bytes": research.MAX_BYTES,
                     "timeout": research.TIMEOUT},
        "alphaxiv": {
            "url_env": alphaxiv.URL_ENV,
            "url_default": alphaxiv.DEFAULT_URL,
            "key_env": alphaxiv.KEY_ENV,
            "read_tools": list(alphaxiv.READ_TOOLS),
            "guide": dict(alphaxiv.GUIDE),
            "description_suffix": alphaxiv.DESCRIPTION_SUFFIX,
        },
    }


def rendered() -> dict[Path, str]:
    out: dict[Path, str] = {}
    for spec in registry.AGENTS.values():
        out[PROMPTS_DIR / f"{spec.name}.md"] = spec.prompt
    out[PROMPTS_DIR / "specs.json"] = _dumps(specs_json())
    guide = {name: {"use_when": u, "instead_or_first": p, "watch_out": w}
             for name, (u, p, w) in qumo_mcp.TOOL_GUIDE.items()}
    out[PROMPTS_DIR / "tool_guide.json"] = _dumps({"schema": "qumo.tool_guide.v1", "tools": guide})
    out[PROMPTS_DIR / "preamble.md"] = prompts.PREAMBLE
    out[PROMPTS_DIR / "modules.json"] = _dumps(modules_json())
    out[PROMPTS_DIR / "tools.json"] = _dumps(tools_json())
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
