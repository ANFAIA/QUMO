"""Tests for tools/catalog.mojo (DESIGN.md 2.8.2, 5.1, 5.3, 8.2).

The wire tool names of the nine Python agents equal the first request each one sends in
Python (tests/fixtures/tools/wire_tool_names.json, written by gen_wire_names.py with deepagents
and the real QUMO MCP server), with a fake MCP list in that server's order. deepagents' eight
slots come first; the harness adds them, so the test prepends them."""

from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import parse_json
from qumo_agent.core.toolspec import ToolSpec, TOOL_MCP, TOOL_ALPHAXIV, TOOL_GENERATED, TOOL_TOOLSMITH, TOOL_LOCAL
from qumo_agent.tools.catalog import file_routes_for, local_tool_names, user_tool_specs
from qumo_agent.tools.config import Registry, LocalToolsCatalog

comptime FX = "agents/mojo/tests/fixtures/tools"


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def fake(name: String, kind: Int) -> ToolSpec:
    return ToolSpec(name, "d " + name, "{\"type\":\"object\",\"properties\":{}}", kind, False, "")


def names_of(specs: List[ToolSpec]) -> List[String]:
    var out: List[String] = ["ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "task"]
    for s in specs:
        out.append(s.name)
    return out^


def joined(items: List[String]) -> String:
    var out = String()
    for i in range(len(items)):
        if i > 0:
            out += ","
        out += items[i]
    return out^


def test_python_wire_lists() raises:
    var fx = parse_json(read_text(FX + "/wire_tool_names.json"))
    var mcp = List[ToolSpec]()
    var order = fx.get("mcp_server_order")
    for i in range(len(order)):
        mcp.append(fake(order.at(i).as_string(), TOOL_MCP))
    assert_equal(len(mcp), 24)
    var reg = Registry(".")
    var cat = LocalToolsCatalog(".")
    var agents = fx.get("agents")
    assert_equal(len(agents), 9)
    for i in range(len(agents)):
        var name = agents.key_at(i)
        var want = List[String]()
        for j in range(len(agents.at(i))):
            want.append(agents.at(i).at(j).as_string())
        var spec = reg.get(name)
        var got = user_tool_specs(spec, cat, mcp, List[ToolSpec](), List[ToolSpec](), List[ToolSpec]())
        assert_equal(joined(names_of(got)), joined(want), name)
        for t in got:
            if t.kind == TOOL_LOCAL:
                assert_true(t.validate_args and t.count_json.byte_length() > 0, t.name)


def test_alphaxiv_and_generated() raises:
    var reg = Registry(".")
    var cat = LocalToolsCatalog(".")
    var mcp: List[ToolSpec] = [fake("qumo_run_qasm", TOOL_MCP), fake("qumo_describe", TOOL_MCP)]
    var ax: List[ToolSpec] = [fake("discover_papers", TOOL_ALPHAXIV), fake("get_paper_content", TOOL_ALPHAXIV)]
    var gen: List[ToolSpec] = [fake("gen_zeta", TOOL_GENERATED), fake("gen_alpha", TOOL_GENERATED), fake("gen_mid", TOOL_GENERATED)]
    # research agents get alphaXiv after the MCP tools; generated tools come last, by name.
    var rw = names_of(user_tool_specs(reg.get("research-watch"), cat, mcp, ax, List[ToolSpec](), List[ToolSpec]()))
    assert_equal(joined(rw), "ls,read_file,write_file,edit_file,delete,glob,grep,task,search_arxiv,sdk_releases,github_release_notes,module_watchlist,fetch_page,list_files,search_code,write_report,discover_papers,get_paper_content")
    var auto = names_of(user_tool_specs(reg.get("autoresearch"), cat, mcp, ax, List[ToolSpec](), gen))
    assert_equal(joined(auto), "ls,read_file,write_file,edit_file,delete,glob,grep,task,search_arxiv,sdk_releases,github_release_notes,module_watchlist,fetch_page,list_files,search_code,run_check,write_report,open_sandbox,qumo_run_qasm,qumo_describe,discover_papers,get_paper_content,gen_alpha,gen_mid,gen_zeta")
    # no research group: no alphaXiv even when loaded
    var op = names_of(user_tool_specs(reg.get("qumo-operator"), cat, mcp, ax, List[ToolSpec](), gen))
    assert_equal(joined(op), "ls,read_file,write_file,edit_file,delete,glob,grep,task,write_report,qumo_run_qasm,qumo_describe,gen_alpha,gen_mid,gen_zeta")


def test_toolsmith_agent() raises:
    """A Mojo-only agent with the toolsmith group (fixture `smith`, the same groups and MCP
    tools as agents/mojo/specs/mojo-toolsmith.json): no run_check (K5)."""
    var reg = Registry(FX + "/root")
    var cat = LocalToolsCatalog(".")
    var spec = reg.get("smith")
    var local = local_tool_names(spec, cat)
    assert_equal(joined(local), "list_files,search_code,write_report,open_sandbox,toolsmith_new,toolsmith_build,toolsmith_test,toolsmith_publish")
    var smith: List[ToolSpec] = [fake("toolsmith_publish", TOOL_TOOLSMITH), fake("toolsmith_new", TOOL_TOOLSMITH),
                                 fake("toolsmith_test", TOOL_TOOLSMITH), fake("toolsmith_build", TOOL_TOOLSMITH)]
    var mcp: List[ToolSpec] = [fake("qumo_run_qasm", TOOL_MCP), fake("qumo_describe", TOOL_MCP)]
    var gen: List[ToolSpec] = [fake("gen_b", TOOL_GENERATED), fake("gen_a", TOOL_GENERATED)]
    var got = user_tool_specs(spec, cat, mcp, List[ToolSpec](), smith, gen)
    assert_equal(joined(names_of(got)), "ls,read_file,write_file,edit_file,delete,glob,grep,task,list_files,search_code,write_report,open_sandbox,toolsmith_new,toolsmith_build,toolsmith_test,toolsmith_publish,qumo_describe,gen_a,gen_b")
    assert_equal(got[4].name, "toolsmith_new")
    assert_equal(got[4].kind, TOOL_TOOLSMITH)
    # the toolsmith specs must be loaded when the agent has the group
    try:
        _ = user_tool_specs(spec, cat, mcp, List[ToolSpec](), List[ToolSpec](), gen)
        assert_true(False, "missing toolsmith specs accepted")
    except e:
        assert_true(String(e).startswith("QA-E140: agent 'smith' needs the toolsmith tool 'toolsmith_new'"))


def test_file_routes() raises:
    var reg = Registry(".")
    var v0 = file_routes_for(reg.get("qumo-operator"))
    assert_false(v0.repo_read or v0.sandbox_write)
    var v1 = file_routes_for(reg.get("research-watch"))
    assert_true(v1.repo_read)
    assert_false(v1.sandbox_write)
    var v2 = file_routes_for(reg.get("autoresearch"))
    assert_true(v2.repo_read and v2.sandbox_write)
    var smith = file_routes_for(Registry(FX + "/root").get("smith"))
    assert_true(smith.repo_read and smith.sandbox_write)


def main() raises:
    test_python_wire_lists()
    test_alphaxiv_and_generated()
    test_toolsmith_agent()
    test_file_routes()
    print("test_tools_catalog OK")
