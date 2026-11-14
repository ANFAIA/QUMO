"""Tests for tools/atom.mojo and tools/html.mojo (DESIGN.md 2.8.4, 8.2).

tests/fixtures/tools/research_golden.json holds, for each Atom feed, the JSON text Python's
search_arxiv_raw gives (through research._json) or its "ParseError: ..." text, and for each
HTML page the text fetch_page returns. gen_research_golden.py writes it from agents/.venv."""

from std.testing import assert_equal, assert_true

from qumo.ai.json import parse_json
from qumo_agent.tools.atom import parse_arxiv_atom, arxiv_entries_json, ArxivEntry
from qumo_agent.tools.html import strip_html

comptime FX = "agents/mojo/tests/fixtures/tools"


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def atom_result(xml: String) -> String:
    try:
        return arxiv_entries_json(parse_arxiv_atom(xml))
    except e:
        return String(e)


def main() raises:
    var golden = parse_json(read_text(FX + "/research_golden.json"))
    var atom = golden.get("atom")
    assert_true(len(atom) >= 15)
    for i in range(len(atom)):
        var c = atom.at(i)
        var name = c.get_string("name", "")
        assert_equal(atom_result(c.get_string("xml", "")), c.get_string("expected", ""), "atom case " + name)
    var html = golden.get("html")
    assert_true(len(html) >= 8)
    for i in range(len(html)):
        var c = html.at(i)
        var name = c.get_string("name", "")
        assert_equal(strip_html(c.get_string("html", "")), c.get_string("expected", ""), "html case " + name)

    # The entry fields as the research tool builds them.
    var entries = parse_arxiv_atom(atom.at(0).get_string("xml", ""))
    assert_equal(len(entries), 1)
    assert_equal(entries[0].id, "2610.00001v1")
    assert_equal(entries[0].title, "A routing method")
    assert_equal(entries[0].authors[0], "A. B.")
    assert_true(not entries[0].url_null and not entries[0].authors_null[0])
    assert_equal(arxiv_entries_json(List[ArxivEntry]()), "[]")
    print("test_tools_atom_html OK")
