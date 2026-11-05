"""deepagents graphs for the QUMO agents (the openrouter back end).

Each agent is a deep agent with its prompt, its local tools, its subset of QUMO MCP tools
and, for research agents, alphaXiv's read tools. Agents that delegate get their
subagents as compiled deep agents (CompiledSubAgent), so delegation nests: the
orchestrator can call autoresearch, which can call research-watch.

deepagents' own file tools use its default state backend: a virtual file system that
lives in the run's state, never on disk. Real reads and writes go through the guarded
tools in tools/repo.py.
"""

from __future__ import annotations

from typing import Any, Callable

from qumo_agents import registry
from qumo_agents.config import Settings, openrouter_chat_model
from qumo_agents.registry import AgentSpec
from qumo_agents.tools import alphaxiv, qumo_mcp


async def load_shared_tools(need_alphaxiv: bool) -> dict[str, list]:
    """MCP tools loaded once per run and shared by every agent in the tree."""
    shared = {"qumo": await qumo_mcp.load_qumo_tools_async(), "alphaxiv": []}
    if need_alphaxiv and alphaxiv.available():
        shared["alphaxiv"] = await alphaxiv.load_alphaxiv_tools_async()
    return shared


def tools_for(spec: AgentSpec, shared: dict[str, list]) -> list:
    tools = registry.local_tools(spec)
    if spec.mcp_tools is None:
        tools += shared["qumo"]
    elif spec.mcp_tools:
        tools += [t for t in shared["qumo"] if t.name in spec.mcp_tools]
    if "research" in spec.tool_groups:
        tools += shared["alphaxiv"]
    return tools


def tree(spec: AgentSpec) -> list[AgentSpec]:
    """spec and every agent it can reach through delegation, each once."""
    seen, order, stack = set(), [], [spec.name]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        order.append(registry.get(name))
        stack.extend(registry.get(name).subagents)
    return order


def build_agent(spec: AgentSpec, settings: Settings, shared: dict[str, list],
                model_factory: Callable[[str], Any], model_override: str | None = None,
                _cache: dict[str, Any] | None = None, _path: tuple[str, ...] = ()):
    """Compile spec as a deep agent. model_factory maps a model id to a chat model,
    so tests can pass a fake model."""
    from deepagents import CompiledSubAgent, create_deep_agent

    cache = {} if _cache is None else _cache
    if spec.name in cache:
        return cache[spec.name]
    subagents = []
    for sub_name in spec.subagents:
        if sub_name in _path or sub_name == spec.name:
            continue  # never delegate back up the chain
        sub = registry.get(sub_name)
        runnable = build_agent(sub, settings, shared, model_factory, None, cache, _path + (spec.name,))
        subagents.append(CompiledSubAgent(name=sub.name, description=sub.description, runnable=runnable))
    agent = create_deep_agent(
        model=model_factory(settings.model_for(spec.name, model_override)),
        tools=tools_for(spec, shared),
        system_prompt=spec.prompt,
        subagents=subagents or None,
        name=spec.name,
    )
    cache[spec.name] = agent
    return agent


async def run_agent(spec: AgentSpec, task: str, settings: Settings, model_override: str | None = None,
                    model_factory: Callable[[str], Any] | None = None, recursion_limit: int = 200) -> str:
    factory = model_factory or (lambda model: openrouter_chat_model(settings, model))
    need_ax = any("research" in s.tool_groups for s in tree(spec))
    shared = await load_shared_tools(need_ax)
    agent = build_agent(spec, settings, shared, factory, model_override)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]},
                                 config={"recursion_limit": recursion_limit})
    last = result["messages"][-1]
    return last.content if isinstance(last.content, str) else str(last.content)
