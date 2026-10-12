"""Full tool registry: core interface tools + tools contributed by every module.

`qumo.ai.tools.default_registry()` only holds the core (circuit I/O,
simulation, description) tools so that low-level modules can import the
registry types without import cycles. Front ends (MCP server, CLI) use
`full_registry()`, which adds one line per module.
"""

from qumo.ai.tools import ToolRegistry, default_registry
from qumo.exec.tools import register_exec_tools


def full_registry() -> ToolRegistry:
    var reg = default_registry()
    register_exec_tools(reg)  # compile, resources, VQE, QAOA
    return reg^
