"""Templates written by toolsmith_new when the model gives no source or cases
(DESIGN.md Appendix B.5).

The placeholders are literal markers, not format fields (the source is full of
braces): `@@NAME@@`, `@@SUMMARY@@` (the first line of the description, without any
triple double quote) and `@@DESCRIPTION@@` (the description escaped for a Mojo string
literal). Markers are replaced in one pass, so a description that itself contains a
marker is copied as written.
"""


def _tool_template_text() -> String:
    var t = String()
    t += "\"\"\"@@NAME@@: @@SUMMARY@@\n"
    t += "\n"
    t += "Generated QUMO tool built on qumo_agent.tool_sdk. It runs with an empty environment in a\n"
    t += "temporary directory, and a human reviews it before any agent can call it.\n"
    t += "\"\"\"\n"
    t += "\n"
    t += "from qumo.ai.json import JsonValue\n"
    t += "from qumo_agent.tool_sdk import ToolInfo, ToolOutput, ok, fail, ok_json, arg_string, serve_tool\n"
    t += "\n"
    t += "comptime NAME = \"@@NAME@@\"\n"
    t += "comptime VERSION = \"0.1.0\"\n"
    t += "comptime SCHEMA = \"{\\\"type\\\":\\\"object\\\",\\\"properties\\\":{\\\"text\\\":{\\\"type\\\":\\\"string\\\",\\\"description\\\":\\\"Input text.\\\"}},\\\"required\\\":[\\\"text\\\"],\\\"additionalProperties\\\":false}\"\n"
    t += "\n"
    t += "\n"
    t += "def handle(args: JsonValue) raises -> ToolOutput:\n"
    t += "    var text = arg_string(args, \"text\", \"\")\n"
    t += "    if text.byte_length() == 0:\n"
    t += "        return fail(\"text is required\")\n"
    t += "    return ok(text)\n"
    t += "\n"
    t += "\n"
    t += "def main() raises:\n"
    t += "    serve_tool(ToolInfo(NAME, NAME, \"@@DESCRIPTION@@\", SCHEMA, VERSION), handle)\n"
    return t^


def _cases_template_text() -> String:
    var t = String()
    t += "{\"schema\": \"qumo.tool_cases.v1\", \"tool\": \"@@NAME@@\", \"cases\": [\n"
    t += " {\"name\": \"echo\", \"arguments\": {\"text\": \"hello\"}, \"expect\": {\"is_error\": false, \"text_equals\": \"hello\"}},\n"
    t += " {\"name\": \"missing text\", \"arguments\": {}, \"expect\": {\"is_error\": true, \"text_contains\": [\"text is required\"]}}\n"
    t += "]}\n"
    return t^


def _summary(description: String) -> String:
    """First line of the description without triple double quotes; backslashes are
    doubled so the docstring has no invalid escape."""
    var first = description
    var nl = description.find("\n")
    if nl >= 0:
        first = String(description[byte=0:nl])
    var cr = first.find("\r")
    if cr >= 0:
        var head = String(first[byte=0:cr])
        first = head^
    var no_quotes = first.replace("\"\"\"", "")
    return no_quotes.replace("\\", "\\\\")


def mojo_string_escape(s: String) -> String:
    """`s` escaped for the inside of a Mojo string literal: backslash, double quote,
    newline, carriage return and tab."""
    var res = String()
    var bs = s.as_bytes()
    var run = 0
    for i in range(len(bs)):
        var c = Int(bs[i])
        var esc: String
        if c == 0x5C:
            esc = "\\\\"
        elif c == 0x22:
            esc = "\\\""
        elif c == 0x0A:
            esc = "\\n"
        elif c == 0x0D:
            esc = "\\r"
        elif c == 0x09:
            esc = "\\t"
        else:
            continue
        res += String(unsafe_from_utf8=bs[run:i])
        res += esc
        run = i + 1
    res += String(unsafe_from_utf8=bs[run:len(bs)])
    return res^


def _fill(template: String, name: String, summary: String, description: String) -> String:
    """Replace the three markers in one left-to-right pass."""
    var res = String()
    var bs = template.as_bytes()
    var i = 0
    var run = 0
    var n = len(bs)
    while i < n:
        if Int(bs[i]) == 0x40 and i + 1 < n and Int(bs[i + 1]) == 0x40:
            var rest = String(unsafe_from_utf8=bs[i:n])
            var value = String()
            var marker_len = 0
            if rest.startswith("@@NAME@@"):
                value = name
                marker_len = 8
            elif rest.startswith("@@SUMMARY@@"):
                value = summary
                marker_len = 11
            elif rest.startswith("@@DESCRIPTION@@"):
                value = description
                marker_len = 15
            if marker_len > 0:
                res += String(unsafe_from_utf8=bs[run:i])
                res += value
                i += marker_len
                run = i
                continue
        i += 1
    res += String(unsafe_from_utf8=bs[run:n])
    return res^


def tool_template(name: String, description: String) -> String:
    """The template tool.mojo for `name` (a valid tool name) and its description."""
    return _fill(_tool_template_text(), name, _summary(description), mojo_string_escape(description))


def cases_template(name: String) -> String:
    """The template cases.json for `name`."""
    return _fill(_cases_template_text(), name, String(), String())
