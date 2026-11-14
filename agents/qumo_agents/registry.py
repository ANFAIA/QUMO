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
        prompts.AUTORESEARCH, ("research", "read", "check", "report", "sandbox"), QUMO_SIM,
        subagents=("research-watch", "code-review", "algorithm-research", "qec-research", "compiler-research"),
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
                   "research-watch", "code-review", "autoresearch")),
]}


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
