"""QUMO's own MCP server (`build/qumo mcp`) as agent tools (DESIGN.md 2.8.5).

`ToolGuide` puts a short guide (when to use the tool, what to call first, pitfalls) in front
of the server's description, byte-identical to `qumo_agents.tools.qumo_mcp.guided_description`.
The guides come from agents/prompts/tool_guide.json (exported from Python).

The persistent stdio session (`QumoMcp`) is added by component T-2, on top of the MCP client.
"""

from qumo.ai.json import parse_json

comptime TOOL_GUIDE_SCHEMA = "qumo.tool_guide.v1"


struct ToolGuide(Copyable):
    var names: List[String]
    var use_when: List[String]
    var instead_or_first: List[String]
    var watch_out: List[String]

    def __init__(out self):
        self.names = List[String]()
        self.use_when = List[String]()
        self.instead_or_first = List[String]()
        self.watch_out = List[String]()

    def load(mut self, text: String) raises:
        """tool_guide.json text (qumo.tool_guide.v1)."""
        var doc = parse_json(text)
        if doc.get_string("schema", "") != TOOL_GUIDE_SCHEMA:
            raise Error("QA-E140: agents/prompts/tool_guide.json is stale. Hint: run agents/.venv/bin/python -m qumo_agents export-prompts")
        var tools = doc.get("tools")
        for i in range(len(tools)):
            var g = tools.at(i)
            self.names.append(tools.key_at(i))
            self.use_when.append(g.get_string("use_when", ""))
            self.instead_or_first.append(g.get_string("instead_or_first", ""))
            self.watch_out.append(g.get_string("watch_out", ""))

    def guided_description(self, name: String, server_text: String) -> String:
        """Byte-identical to qumo_agents.tools.qumo_mcp.guided_description."""
        var k = -1
        for i in range(len(self.names)):
            if self.names[i] == name:
                k = i
        if k < 0:
            return server_text
        var out = "Use when " + self.use_when[k]
        if self.instead_or_first[k].byte_length() > 0:
            out += "\nInstead or first: " + self.instead_or_first[k]
        if self.watch_out[k].byte_length() > 0:
            out += "\nWatch out: " + self.watch_out[k]
        return out + "\n\n" + server_text
