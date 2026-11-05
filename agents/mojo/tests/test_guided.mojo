"""Guided tool descriptions must be byte-identical to Python's guided_description().

The fixture was produced by qumo_agents.tools.qumo_mcp.guided_description for every
tool in TOOL_GUIDE plus one tool without a guide:

    cd agents && .venv/bin/python -c '
    import json
    from qumo_agents.tools.qumo_mcp import TOOL_GUIDE, guided_description
    t = "Server text of {name}: \\"quoted\\", \u00fcn\u00efcode.\\nSecond line."
    names = list(TOOL_GUIDE) + ["qumo_not_in_the_guide"]
    d = {n: guided_description(n, t.replace("{name}", n)) for n in names}
    print(json.dumps({"schema": "qumo.guided_fixture.v1", "server_text": t, "descriptions": d}, indent=1))
    ' > mojo/tests/fixtures/guided_descriptions.json

The test also fails when tool_guide.json gains or loses a tool the fixture does not
cover, so a stale fixture cannot pass silently."""

from std.testing import assert_equal, assert_true

from qumo.ai.json import parse_json
from qumo_agent.config import ToolGuide, read_text


def main() raises:
    var g = ToolGuide()
    g.load(read_text("agents/prompts/tool_guide.json"))
    var fx = parse_json(read_text("agents/mojo/tests/fixtures/guided_descriptions.json"))
    var tmpl = fx.get_string("server_text", "")
    var d = fx.get("descriptions")
    assert_equal(len(d), len(g.names) + 1)
    for name in g.names:
        assert_true(d.get(name).is_string(), "fixture lacks " + name)
    var checked = 0
    for i in range(len(d)):
        var name = d.key_at(i)
        var server = tmpl.replace("{name}", name)
        assert_equal(g.guided_description(name, server), d.at(i).as_string(), "tool " + name)
        checked += 1
    # A tool the guide does not know keeps the server text unchanged.
    assert_equal(g.guided_description("qumo_not_in_the_guide", "plain"), "plain")
    print("test_guided: " + String(checked) + " descriptions identical")
    print("test_guided OK")
