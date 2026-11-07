"""The QUMO tools each agent sees after deepagents' own slots, in wire order (DESIGN.md 2.8.2,
5.1), and how its relative file paths are routed (5.3)."""

from qumo_agent.core.errors import E_CONFIG, qa_error
from qumo_agent.core.toolspec import ToolSpec
from qumo_agent.harness.env import FileRoutes
from qumo_agent.tools.config import AgentSpecData, LocalToolsCatalog
from qumo_agent.util.pyfmt import py_str_repr

# The toolsmith tool group (Mojo-only, component S); its specs come from toolsmith/tools.mojo.
comptime TOOLSMITH_GROUP = "toolsmith"


def toolsmith_tool_names() -> List[String]:
    return ["toolsmith_new", "toolsmith_build", "toolsmith_test", "toolsmith_publish"]


def _contains(items: List[String], x: String) -> Bool:
    for s in items:
        if s == x:
            return True
    return False


def file_routes_for(spec: AgentSpecData) -> FileRoutes:
    """repo_read: the agent has the `read` group; sandbox_write: it has the `sandbox` group
    (autoresearch, mojo-toolsmith)."""
    return FileRoutes(_contains(spec.tool_groups, "read"), _contains(spec.tool_groups, "sandbox"))


def local_tool_names(spec: AgentSpecData, catalog: LocalToolsCatalog) raises -> List[String]:
    """registry.local_tools: groups in spec order, each group's tools in tools.json order,
    first occurrence kept, without read_file/write_file/edit_file (they live in deepagents'
    slots, merged by the harness)."""
    var out = List[String]()
    for g in spec.tool_groups:
        var names: List[String]
        if g == TOOLSMITH_GROUP:
            names = toolsmith_tool_names()
        else:
            names = catalog.group(g)
        for n in names:
            if n == "read_file" or n == "write_file" or n == "edit_file":
                continue
            if not _contains(out, n):
                out.append(n)
    return out^


def _sorted_by_name(specs: List[ToolSpec]) -> List[ToolSpec]:
    """Stable insertion sort by name (code-point order equals UTF-8 byte order)."""
    var out = List[ToolSpec]()
    for s in specs:
        var i = len(out)
        out.append(s.copy())
        while i > 0 and out[i - 1].name > out[i].name:
            var tmp = out[i - 1].copy()
            out[i - 1] = out[i].copy()
            out[i] = tmp^
            i -= 1
    return out^


def user_tool_specs(spec: AgentSpecData, catalog: LocalToolsCatalog, mcp: List[ToolSpec], alphaxiv: List[ToolSpec], toolsmith: List[ToolSpec], generated: List[ToolSpec]) raises -> List[ToolSpec]:
    """Local tools, then QUMO MCP tools (server order, filtered by the spec), then alphaXiv
    (agents with `research`; the caller passes an empty list when ALPHAXIV_API_KEY is not
    set), then generated tools sorted by name (`generated` is already filtered for this
    agent by the caller)."""
    var out = List[ToolSpec]()
    var smith = toolsmith_tool_names()
    for n in local_tool_names(spec, catalog):
        if _contains(smith, n):
            var found = False
            for t in toolsmith:
                if t.name == n:
                    out.append(t.copy())
                    found = True
                    break
            if not found:
                raise qa_error(E_CONFIG, "agent " + py_str_repr(spec.name) + " needs the toolsmith tool " + py_str_repr(n) + ", which was not loaded", "")
        else:
            out.append(catalog.spec(n))
    for t in mcp:
        if spec.mcp_all or _contains(spec.mcp_tools, t.name):
            out.append(t.copy())
    if _contains(spec.tool_groups, "research"):
        for t in alphaxiv:
            out.append(t.copy())
    for t in _sorted_by_name(generated):
        out.append(t.copy())
    return out^
