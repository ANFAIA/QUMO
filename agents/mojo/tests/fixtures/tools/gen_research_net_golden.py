"""Write research_net_golden.json: what Python's PyPI, GitHub and arXiv-URL code gives for fixed
inputs, and the module_watchlist outputs.

Dev-time only (owner T, DESIGN.md 8.2): test_tools_research compares tools/research.mojo with
these outputs. Network is never used: research._get is replaced by a function that records
the URL and returns the fixture bytes.

    agents/.venv/bin/python -I agents/mojo/tests/fixtures/tools/gen_research_net_golden.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "agents"))

from qumo_agents import modules  # noqa: E402
from qumo_agents.tools import research  # noqa: E402


def release(upload_time, n=1):
    return [{"upload_time": upload_time, "filename": f"f{i}.whl"} for i in range(n)]


PYPI_CASES = {
    # more than eight releases, date ties (stable order), empty releases, three kinds of links
    "qiskit_like": ("qiskit", {
        "info": {
            "version": "2.1.0", "summary": "Software for developing quantum computing programs — café",
            "requires_python": ">=3.9",
            "project_urls": {
                "Documentation": "https://quantum.cloud.ibm.com/docs",
                "Release Notes": "https://quantum.cloud.ibm.com/docs/api/qiskit/release-notes",
                "Source Code": "https://github.com/Qiskit/qiskit",
                "Changelog": "https://example.org/changelog",
            },
            "description": "long text that is never returned",
        },
        "releases": {
            "0.1.0": release("2019-03-05T10:00:00"),
            "0.2.0": [],
            "1.0.0": release("2024-02-15T08:00:01", 3),
            "1.0.1": release("2024-02-15T09:00:00"),
            "1.0.2": release("2024-03-01T00:00:00"),
            "1.1.0": release("2024-05-02T12:00:00"),
            "1.1.1": release("2024-06-03T12:00:00"),
            "1.2.0": release("2024-07-01T12:00:00"),
            "1.2.1": release("2024-07-01T13:00:00"),
            "1.3.0": release("2024-11-04T12:00:00"),
            "2.0.0": release("2025-03-31T12:00:00"),
            "2.1.0": release("2025-06-12T12:00:00"),
        },
    }),
    "minimal": ("tiny", {"info": {"version": "0.0.1"}}),
    "null_urls": ("nourls", {"info": {"version": "1.0", "summary": None, "requires_python": None, "project_urls": None},
                             "releases": {"1.0": release("2020-01-01T00:00:00")}}),
    "null_url_value": ("nullval", {"info": {"version": "1.0", "project_urls": {"News": None, "Home": None, "Repo": "http://github.com/a/b"}},
                                   "releases": {}}),
    "changelog_case": ("casey", {"info": {"version": "3", "project_urls": {"CHANGELOG": "https://c.example/CHANGELOG.md",
                                                                          "github": "https://github.com/x/y"}},
                                 "releases": {"3": release("2021-05-05T05:05:05")}}),
    "unicode_package": ("café/中", {"info": {"version": "9"}, "releases": {"9": release("2022-02-02T00:00:00")}}),
    "missing_info": ("noinfo", {"releases": {}}),
    "missing_upload_time": ("noupload", {"info": {"version": "1"}, "releases": {"1": [{"filename": "x"}]}}),
}

GITHUB_CASES = {
    "two_releases": ("Qiskit/qiskit", [
        {"tag_name": "2.1.0", "published_at": "2025-06-12T12:00:00Z", "name": "Qiskit 2.1.0",
         "body": "## Notes\n\nCafé — new things\n" + "x" * 5000, "html_url": "https://github.com/Qiskit/qiskit/releases/tag/2.1.0"},
        {"tag_name": "2.0.2", "published_at": None, "name": None, "body": None,
         "html_url": "https://github.com/Qiskit/qiskit/releases/tag/2.0.2"},
    ]),
    "sparse": ("a/b", [{"tag_name": "v1"}, {}]),
    "empty": ("a_b/c.d-e", []),
    "exact_4000": ("o/n", [{"tag_name": "t", "published_at": "2024-01-02", "name": "n", "body": "é" * 4000, "html_url": "u"},
                           {"tag_name": "t2", "published_at": "2024-01-02T", "name": "n", "body": "é" * 4001, "html_url": "u"}]),
}

ATOM_MIN = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1v1</id>'
            "<title>t</title></entry></feed>")
ARXIV_URL_CASES = [
    ("routing", ["quant-ph"], 10),
    ("qubit routing", ["quant-ph", "cs.DC"], 100),
    ("a&b=c é\"x\" +/?", ["quant-ph"], 0),
    ("x", [], 5),
    ("two  spaces", ["quant-ph"], -3),
    ("y", ["quant-ph", "cs.ET", "cs.AR"], 50),
]


def run(fn):
    seen = []
    research._get = lambda url, headers=None: (seen.append([url, headers]), fn())[1]
    return seen


def pypi_expected(package, doc):
    seen = run(lambda: json.dumps(doc).encode())
    out = research.sdk_releases.invoke({"package": package})
    return {"url": seen[0][0], "expected": out}


def github_expected(repo, doc, count):
    seen = run(lambda: json.dumps(doc).encode())
    out = research.github_release_notes.invoke({"repo": repo, "count": count})
    return {"url": seen[0][0], "headers": seen[0][1], "expected": out}


def arxiv_url(query, cats, n):
    seen = run(lambda: ATOM_MIN.encode())
    research.search_arxiv_raw(query, tuple(cats), n)
    return seen[0][0]


def watch(name):
    try:
        return research.module_watchlist.invoke({"module": name})
    except ValueError as e:
        return research._json({"error": str(e)})


out = {
    "schema": "qumo.test_research_net_golden.v1",
    "pypi": [dict(name=n, package=p, doc=d, **pypi_expected(p, d)) for n, (p, d) in PYPI_CASES.items()],
    "github": [dict(name=n, repo=r, doc=d, count=c, **github_expected(r, d, c))
               for n, (r, d) in GITHUB_CASES.items() for c in (3, 0, 99)],
    "arxiv_urls": [{"query": q, "categories": c, "max_results": n, "url": arxiv_url(q, c, n)} for q, c, n in ARXIV_URL_CASES],
    "watchlist": [{"module": m, "expected": watch(m)} for m in list(modules.MODULES) + ["nope", ""]],
}
(HERE / "research_net_golden.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
