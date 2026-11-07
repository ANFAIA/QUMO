"""Tool descriptions as the model sees them, and tool results as the harness gets them."""

from qumo.ai.json import JsonWriter

comptime TOOL_HARNESS = 0     # ls, read_file, write_file, edit_file, delete, glob, grep, task, execute
comptime TOOL_LOCAL = 1       # QUMO Python-defined tools (validated like pydantic)
comptime TOOL_MCP = 2         # QUMO MCP server
comptime TOOL_ALPHAXIV = 3
comptime TOOL_GENERATED = 4
comptime TOOL_TOOLSMITH = 5


@fieldwise_init
struct ToolSpec(Copyable):
    var name: String
    var description: String
    var parameters_json: String   # exact wire "parameters" JSON text
    var kind: Int
    var validate_args: Bool       # True for TOOL_LOCAL and TOOL_TOOLSMITH
    var count_json: String        # {"name","description","parameters"} as Python's token counter sees the tool (2.7.6); "" = derive from the wire form

    def wire_json(self) -> String:
        """`{"type": "function", "function": {"name", "description", "parameters"}}`."""
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("type", "function")
        w.key("function")
        w.begin_object()
        w.kv_string("name", self.name)
        w.kv_string("description", self.description)
        w.key("parameters")
        if self.parameters_json.byte_length() > 0:
            w.raw(self.parameters_json)
        else:
            w.raw("{}")
        w.end_object()
        w.end_object()
        return w.finish()


@fieldwise_init
struct ToolOutcome(Copyable):
    var text: String
    var parts: List[String]
    var is_parts: Bool
    var is_error: Bool

    def joined_text(self) -> String:
        """Parts joined with "\\n" when is_parts (deepagents _extract_text_from_message), else text."""
        if self.is_parts:
            return String("\n").join(self.parts)
        return self.text


def text_outcome(text: String) -> ToolOutcome:
    return ToolOutcome(text, List[String](), False, False)


def error_outcome(text: String) -> ToolOutcome:
    return ToolOutcome(text, List[String](), False, True)


def parts_outcome(var parts: List[String], is_error: Bool) -> ToolOutcome:
    return ToolOutcome(String(), parts^, True, is_error)
