"""Offline tests for the QUMO agent layer: no network, no model calls."""

from __future__ import annotations

import glob
import json

import pytest

from qumo_agents import modules, registry
from qumo_agents.config import REPO_ROOT, Settings
from qumo_agents.tools import alphaxiv, qumo_mcp, repo, research


def test_module_paths_docs_and_tests_exist():
    for m in modules.MODULES.values():
        for p in m.paths + m.docs:
            assert (REPO_ROOT / p).exists(), f"{m.name}: {p} missing"
        for g in m.tests:
            assert glob.glob(str(REPO_ROOT / g)), f"{m.name}: no file matches {g}"


def test_registry_is_consistent():
    assert len(registry.AGENTS) == 9
    for spec in registry.AGENTS.values():
        for s in spec.subagents:
            assert s in registry.AGENTS and s != spec.name
        for name in spec.mcp_tools or ():
            assert name in qumo_mcp.TOOL_GUIDE, f"{spec.name}: unknown MCP tool {name}"
        assert registry.local_tools(spec) or spec.uses_mcp
    assert set(registry.get("orchestrator").subagents) >= {"execution", "qumo-operator", "algorithm-research",
                                                            "qec-research", "compiler-research"}
    assert registry.get("autoresearch").writes_code
    assert not any(s.writes_code for s in registry.AGENTS.values() if s.name != "autoresearch")


def test_settings_models_cover_every_agent():
    s = Settings.load()
    assert s.default_backend in ("openrouter", "claude-code", "codex")
    for name in registry.AGENTS:
        assert "/" in s.model_for(name), name  # OpenRouter ids are vendor/model
    assert s.model_for("code-review", "x/y") == "x/y"


def test_read_tools_stay_inside_the_repository():
    assert "error" in repo.read_file.invoke({"path": "../../etc/passwd"})
    assert "not readable" in repo.read_file.invoke({"path": ".venv/bin/python"})
    assert "not readable" in repo.read_file.invoke({"path": ".env"})
    assert "1\t" in repo.read_file.invoke({"path": "README.md", "max_lines": 3})


def test_writes_need_a_sandbox_and_commands_need_the_allow_list():
    assert repo.write_file.invoke({"path": "x.txt", "content": "x"}).startswith("refused")
    assert repo.edit_file.invoke({"path": "README.md", "old": "a", "new": "b"}).startswith("refused")
    assert repo.run_check.invoke({"argv": ["rm", "-rf", "/"]}).startswith("refused")
    assert repo.run_check.invoke({"argv": ["git", "diff", "$(id)"]}).startswith("refused")
    assert repo.run_check.invoke({"argv": [".venv/bin/mojo", "run", "-I", ".", "../x.mojo"]}).startswith("refused")
    assert repo.run_check.invoke({"argv": ["git", "status", "--short"]}).startswith("exit 0")


def test_claude_code_command_keeps_the_safety_rules(monkeypatch):
    from qumo_agents import backends
    monkeypatch.setenv(alphaxiv.KEY_ENV, "secret-value-123456")
    s = Settings.load()
    watch = backends.claude_code_command(registry.get("research-watch"), "t", s, None)
    allowed = watch[watch.index("--allowedTools") + 1]
    assert "Edit" not in allowed.split(",") and "WebSearch" in allowed
    assert "mcp__alphaxiv__discover_papers" in allowed and "save_papers_to_folder" not in allowed
    assert "git commit" in watch[watch.index("--disallowedTools") + 1]
    auto = backends.claude_code_command(registry.get("autoresearch"), "t", s, None)
    assert "Edit" in auto[auto.index("--allowedTools") + 1].split(",")
    cfg = backends._mcp_config_file(registry.get("research-watch"))
    try:
        text = cfg.read_text()
        assert "${ALPHAXIV_API_KEY}" in text and "secret-value" not in text
    finally:
        cfg.unlink()
    for cmd in (watch, auto, backends.codex_command(registry.get("research-watch"), "t", s, REPO_ROOT)):
        assert not any("secret-value" in c for c in cmd)


def test_codex_command_sandbox_modes():
    from qumo_agents import backends
    s = Settings.load()
    ro = backends.codex_command(registry.get("code-review"), "t", s, REPO_ROOT)
    rw = backends.codex_command(registry.get("autoresearch"), "t", s, REPO_ROOT)
    assert ro[ro.index("--sandbox") + 1] == "read-only"
    assert rw[rw.index("--sandbox") + 1] == "workspace-write"
    assert any("mcp_servers.qumo.command" in c for c in rw)


def test_guided_descriptions_keep_the_server_text():
    d = qumo_mcp.guided_description("qumo_run_qasm", "SERVER TEXT")
    assert d.startswith("Use when") and d.endswith("SERVER TEXT")
    assert qumo_mcp.guided_description("unknown_tool", "S") == "S"


ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2610.00001v1</id><title>A  routing
method</title><published>2026-10-01T00:00:00Z</published>
<summary>We route.</summary><author><name>A. B.</name></author></entry></feed>"""


def test_arxiv_parser(monkeypatch):
    monkeypatch.setattr(research, "_get", lambda url, headers=None: ATOM)
    out = research.search_arxiv_raw("qubit routing", ("quant-ph",), 5)
    assert out == [{"id": "2610.00001v1", "title": "A routing method", "published": "2026-10-01",
                    "authors": ["A. B."], "summary": "We route.", "url": "http://arxiv.org/abs/2610.00001v1"}]


def test_fetch_refuses_plain_http():
    assert "error" in json.loads(research.fetch_page.invoke({"url": "http://example.com"}))


@pytest.mark.skipif(not qumo_mcp.QUMO_BIN.exists(), reason="build/qumo not built")
def test_qumo_mcp_tools_load_with_guides():
    tools = qumo_mcp.load_qumo_tools()
    names = {t.name for t in tools}
    assert set(qumo_mcp.TOOL_GUIDE) <= names, set(qumo_mcp.TOOL_GUIDE) - names
    run = next(t for t in tools if t.name == "qumo_run_qasm")
    assert run.description.startswith("Use when")


def test_network_failures_become_tool_errors(monkeypatch):
    import http.client

    def boom(url, headers=None):
        raise http.client.RemoteDisconnected("Remote end closed connection without response")

    monkeypatch.setattr(research, "_get", boom)
    calls = [(research.search_arxiv, {"query": "routing"}), (research.sdk_releases, {"package": "qiskit"}),
             (research.github_release_notes, {"repo": "Qiskit/qiskit"}), (research.fetch_page, {"url": "https://x.org"})]
    for t, args in calls:
        assert "error" in json.loads(t.invoke(args)), t.name
