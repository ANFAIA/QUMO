"""The nine QUMO agents: prompt, tools, MCP tools and the agents each one may delegate to."""

from __future__ import annotations

from dataclasses import dataclass

from qumo_agents import prompts
from qumo_agents.tools import repo, research

ALL_MCP = None  # every QUMO MCP tool
ALL_GENERATED = None  # every reviewed generated tool (Mojo harness only)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str  # what the orchestrating agent reads when it picks a subagent
    prompt: str
    tool_groups: tuple[str, ...]  # keys of TOOL_GROUPS
    mcp_tools: tuple[str, ...] | None = ()  # () none, None all, or a subset
    subagents: tuple[str, ...] = ()
    writes_code: bool = False
    # Reviewed Mojo-generated tools (gen_*) this agent may call in the native Mojo harness:
    # () none, None every reviewed tool, or a list of names. The Python agents ignore it.
    generated_tools: tuple[str, ...] | None = ()
    # Subagents whose answers may carry web-derived text into this agent. Required for every
    # such edge of an agent that can reach both web fetch and code execution (see
    # check_web_exec); graph.build_agent then returns their answers wrapped (untrusted.wrap).
    untrusted_subagents: tuple[str, ...] = ()

    @property
    def uses_mcp(self) -> bool:
        return self.mcp_tools is None or len(self.mcp_tools) > 0


TOOL_GROUPS = {
    "research": research.RESEARCH_TOOLS,
    "read": repo.READ_TOOLS,
    "check": repo.CHECK_TOOLS,
    "report": repo.REPORT_TOOLS,
    "sandbox": repo.SANDBOX_TOOLS,
}

QUMO_SIM = ("qumo_describe", "qumo_build_circuit", "qumo_validate_qasm", "qumo_circuit_info",
            "qumo_estimate_resources", "qumo_run_qasm", "qumo_statevector", "qumo_expectation")

AGENTS: dict[str, AgentSpec] = {a.name: a for a in [
    AgentSpec(
        "research-watch",
        "Finds new papers and new releases of alternative SDKs for one QUMO module, and ranks what QUMO should adopt.",
        prompts.RESEARCH_WATCH, ("research", "read", "report")),
    AgentSpec(
        "code-review",
        "Reviews QUMO code for performance and security on an independent model; reports file:line findings with failure scenarios.",
        prompts.CODE_REVIEW, ("read", "check", "report")),
    AgentSpec(
        "autoresearch",
        "Improves one QUMO module step by step in a sandbox branch: hypothesis, baseline, change, measurement, review.",
        prompts.AUTORESEARCH, ("read", "check", "report", "sandbox"), QUMO_SIM,
        subagents=("research-watch", "code-review", "algorithm-research", "qec-research", "compiler-research"),
        untrusted_subagents=("research-watch", "algorithm-research", "qec-research", "compiler-research"),
        writes_code=True, generated_tools=ALL_GENERATED),
    AgentSpec(
        "qumo-operator",
        "Uses QUMO through MCP: builds algorithms, simulates and compiles circuits, runs QEC experiments, explains error codes.",
        prompts.QUMO_OPERATOR, ("report",), ALL_MCP, generated_tools=ALL_GENERATED),
    AgentSpec(
        "algorithm-research",
        "Explores new quantum algorithms, checks them against QUMO's library and prototypes toy instances.",
        prompts.ALGORITHM_RESEARCH, ("research", "read", "report"), QUMO_SIM),
    AgentSpec(
        "qec-research",
        "Explores new error-correcting codes and decoders and compares them with QUMO's QEC baseline.",
        prompts.QEC_RESEARCH, ("research", "read", "report"),
        ("qumo_describe", "qumo_qec_memory", "qumo_qec_dem", "qumo_stim_import", "qumo_ft_codes")),
    AgentSpec(
        "compiler-research",
        "Explores new circuit optimisation and compilation methods and measures QUMO's current compiler as the baseline.",
        prompts.COMPILER_RESEARCH, ("research", "read", "report"),
        ("qumo_describe", "qumo_transpile", "qumo_circuit_info", "qumo_build_circuit", "qumo_to_pbc")),
    AgentSpec(
        "execution",
        "Runs circuits the best available way: validates, sizes, picks simulator or GPU, compiles and emulates targets.",
        prompts.EXECUTION, ("report",),
        ("qumo_validate_qasm", "qumo_circuit_info", "qumo_estimate_resources", "qumo_gpu_info", "qumo_run_qasm",
         "qumo_run_device", "qumo_statevector", "qumo_density", "qumo_emulate", "qumo_transpile", "qumo_describe")),
    AgentSpec(
        "orchestrator",
        "Splits a problem between classical, AI and quantum stages and delegates to every research and execution agent.",
        prompts.ORCHESTRATOR, ("read", "report"), ("qumo_estimate_resources", "qumo_describe"),
        subagents=("qumo-operator", "execution", "algorithm-research", "qec-research", "compiler-research",
                   "research-watch", "code-review", "autoresearch"),
        untrusted_subagents=("algorithm-research", "qec-research", "compiler-research", "research-watch",
                             "autoresearch")),
]}


WEB_GROUPS = frozenset({"research"})  # fetch_page reaches any public https host
EXEC_GROUPS = frozenset({"check"})  # run_check runs code, including code the agent wrote


def closure(name: str, agents: dict[str, AgentSpec]) -> set[str]:
    """name and every agent reachable from it through delegation."""
    seen, stack = set(), [name]
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n)
            stack.extend(agents[n].subagents)
    return seen


def _reach(name: str, agents: dict[str, AgentSpec]) -> tuple[bool, bool]:
    groups = {g for n in closure(name, agents) for g in agents[n].tool_groups}
    return bool(groups & WEB_GROUPS), bool(groups & EXEC_GROUPS)


def reaches_web_and_exec(agents: dict[str, AgentSpec] | None = None) -> set[str]:
    """Agents whose delegation closure holds both a web-fetch and a code-execution tool."""
    agents = AGENTS if agents is None else agents
    return {n for n in agents if all(_reach(n, agents))}


def check_web_exec(agents: dict[str, AgentSpec] | None = None) -> None:
    """The invariant, enforced at import:
    1. no agent has web fetch and code execution among its own tool groups;
    2. every agent that reaches both through delegation declares, for each subagent that
       reaches web fetch, that its answer is web-derived (untrusted_subagents), and
       declares nothing else (the declaration is exact);
    graph.build_agent turns each declared edge into a wrapped, size-bounded answer."""
    agents = AGENTS if agents is None else agents
    for spec in agents.values():
        if spec.tool_groups and WEB_GROUPS & set(spec.tool_groups) and EXEC_GROUPS & set(spec.tool_groups):
            raise RuntimeError(f"{spec.name}: research and check tools may not be combined")
        for s in spec.untrusted_subagents:
            if s not in spec.subagents:
                raise RuntimeError(f"{spec.name}: untrusted_subagents names {s!r}, which is not a subagent")
    for name in reaches_web_and_exec(agents):
        spec = agents[name]
        need = {s for s in spec.subagents if _reach(s, agents)[0]}
        missing, extra = need - set(spec.untrusted_subagents), set(spec.untrusted_subagents) - need
        if missing:
            raise RuntimeError(f"{name}: reaches web fetch and code execution; its answers from "
                               f"{', '.join(sorted(missing))} carry web text and must be in untrusted_subagents")
        if extra:
            raise RuntimeError(f"{name}: untrusted_subagents lists {', '.join(sorted(extra))}, "
                               "which do not reach web fetch")
    for name, spec in agents.items():
        if name not in reaches_web_and_exec(agents) and spec.untrusted_subagents:
            raise RuntimeError(f"{name}: declares untrusted_subagents but does not reach both web and exec")


check_web_exec()


def get(name: str) -> AgentSpec:
    try:
        return AGENTS[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; choose one of {', '.join(AGENTS)}") from None


def local_tools(spec: AgentSpec) -> list:
    seen, out = set(), []
    for g in spec.tool_groups:
        for t in TOOL_GROUPS[g]:
            if t.name not in seen:
                seen.add(t.name)
                out.append(t)
    return out
