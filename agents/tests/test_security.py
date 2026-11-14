"""Offline tests for the agent layer's outbound and command limits. No real network traffic:
addresses are numeric or loopback names, which resolve without DNS."""

from __future__ import annotations

import urllib.request

import pytest

from qumo_agents import registry
from qumo_agents.tools import repo, research


@pytest.mark.parametrize("url", [
    "http://export.arxiv.org/api/query?x=1",          # plain http
    "https://user:pw@pypi.org/pypi/qiskit/json",       # credentials in URL
    "https://pypi.org:8443/pypi/qiskit/json",          # non-default port
    "https://127.0.0.1:8443/",                         # loopback literal
    "https://[::1]/",                                  # IPv6 loopback
    "https://169.254.169.254/latest/meta-data/",       # cloud metadata
    "https://10.0.0.5/",                               # private range
    "https://localhost/",                              # resolves to loopback
])
def test_fetch_refuses_unsafe_urls(url):
    with pytest.raises(ValueError):
        research._check_url(url)


def test_api_tools_limited_to_api_hosts():
    with pytest.raises(ValueError, match="allow-list"):
        research._check_url("https://example.com/", research.API_HOSTS)


def test_public_https_accepted():
    research._check_url("https://8.8.8.8/")  # numeric public address, no DNS needed


def test_redirect_to_private_address_refused():
    handler = research._CheckedRedirects(None)
    req = urllib.request.Request("https://8.8.8.8/start")
    with pytest.raises(ValueError):
        handler.redirect_request(req, None, 302, "Found", {}, "https://127.0.0.1/admin")


def test_redirect_downgrade_to_http_refused():
    handler = research._CheckedRedirects(research.API_HOSTS)
    req = urllib.request.Request("https://api.github.com/x")
    with pytest.raises(ValueError):
        handler.redirect_request(req, None, 302, "Found", {}, "http://api.github.com/x")


def test_redirect_drops_authorization():
    handler = research._CheckedRedirects(research.API_HOSTS)
    req = urllib.request.Request("https://api.github.com/x", headers={"Authorization": "Bearer t"})
    new = handler.redirect_request(req, None, 302, "Found", {}, "https://api.github.com/y")
    assert new is not None
    assert "Authorization" not in new.headers


def test_redirect_limit():
    handler = research._CheckedRedirects(None)
    req = urllib.request.Request("https://8.8.8.8/")
    for _ in range(research.MAX_REDIRECTS):
        handler.redirect_request(req, None, 302, "Found", {}, "https://8.8.8.8/")
    with pytest.raises(ValueError, match="too many"):
        handler.redirect_request(req, None, 302, "Found", {}, "https://8.8.8.8/")


def test_checks_get_no_secrets(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = repo._check_env()
    assert "GITHUB_TOKEN" not in env and "OPENROUTER_API_KEY" not in env
    assert env["PATH"] == "/usr/bin"


@pytest.mark.parametrize("argv", [
    ["git", "diff", "--no-index", "/etc/hosts", "/dev/null"],      # read any file
    ["git", "diff", "/etc/hosts", "/dev/null"],                    # implicit --no-index
    ["git", "diff", "--output=/tmp/x", "HEAD"],                    # write any file
    ["git", "diff", "--output=x", "HEAD"],                         # write inside the tree
    ["git", "diff", "--ext-diff"],
    ["git", "status", "--short", "--output=/tmp/x"],
    [".venv/bin/mojo", "run", "-I", ".", "/tmp/x.mojo"],           # code outside the tree
    ["build/qumo", "run", "/etc/hosts"],
])
def test_run_check_stays_in_the_tree(argv):
    assert repo.run_check.invoke({"argv": argv}).startswith("refused")


def test_run_check_plain_git_still_works():
    assert repo.run_check.invoke({"argv": ["git", "diff", "--stat"]}).startswith("exit 0")
    assert repo.run_check.invoke({"argv": ["git", "diff", "--name-only", "HEAD", "--", "agents"]}).startswith("exit 0")


def test_autoresearch_cannot_fetch_and_run():
    assert "research" not in registry.get("autoresearch").tool_groups
    for spec in registry.AGENTS.values():
        assert not ("research" in spec.tool_groups and "check" in spec.tool_groups)
