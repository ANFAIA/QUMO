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


# ----------------------------------------------------------------------------- build/qumo
@pytest.fixture
def fake_tree(tmp_path, monkeypatch):
    """A fake repository root with a bell file and a secret-looking file, and subprocess stubbed."""
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "bell.qasm").write_text("OPENQASM 3.0;\n")
    (tmp_path / ".env").write_text("TOPSECRET=1\n")
    (tmp_path / "README.md").write_text("readme\n")
    monkeypatch.setattr(repo, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(repo._state, "sandbox", None)
    calls = []

    class Done:
        returncode, stdout, stderr = 0, "ok", ""

    monkeypatch.setattr(repo.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)) or Done())
    return tmp_path, calls


@pytest.mark.parametrize("argv", [
    ["build/qumo", "convert", "examples/bell.qasm", "-o", "README.md"],         # reported: overwrite a file
    ["build/qumo", "convert", "examples/bell.qasm", "--output", "README.md"],
    ["build/qumo", "convert", "examples/bell.qasm", "--output=README.md"],
    ["build/qumo", "transpile", "examples/bell.qasm", "-o", "x.qasm"],
    ["build/qumo", "validate", ".env"],                                         # echoes the source line
    ["build/qumo", "validate", "examples/.env.local"],
    ["build/qumo", "run", "/etc/hosts"],                                        # absolute path
    ["build/qumo", "run", "--output=/tmp/x"],
    ["build/qumo", "tool", "list"],                                             # not an allowed subcommand
    ["build/qumo", "mcp"],
    ["build/qumo"],
    ["build/qumo", "run", "examples/bell.qasm", "--unknown"],
    ["build/qumo", "convert", "examples/bell.qasm", "-oREADME.md"],
    ["build/qumo", "validate", "build/qumo"],                                   # skipped directory
])
def test_build_qumo_refused_without_a_sandbox(fake_tree, argv):
    _, calls = fake_tree
    assert repo.run_check.invoke({"argv": argv}).startswith("refused")
    assert calls == []


def test_build_qumo_output_only_inside_the_sandbox(fake_tree, tmp_path_factory, monkeypatch):
    _, calls = fake_tree
    sandbox = tmp_path_factory.mktemp("sandbox")
    (sandbox / "examples").mkdir()
    monkeypatch.setitem(repo._state, "sandbox", sandbox)
    ok = repo.run_check.invoke({"argv": ["build/qumo", "convert", "examples/bell.qasm", "-o", "out.qasm"]})
    assert ok.startswith("exit 0") and calls[-1][1]["cwd"] == sandbox
    for out in ("../escape.qasm", ".env", ".git/hooks/x", "/tmp/x"):
        argv = ["build/qumo", "convert", "examples/bell.qasm", "-o", out]
        assert repo.run_check.invoke({"argv": argv}).startswith("refused"), out
    (sandbox / "link").symlink_to(fake_tree[0])
    argv = ["build/qumo", "convert", "examples/bell.qasm", "-o", "link/README.md"]
    assert repo.run_check.invoke({"argv": argv}).startswith("refused")
    assert len(calls) == 1


@pytest.mark.parametrize("argv", [
    ["build/qumo", "validate", "examples/bell.qasm"],
    ["build/qumo", "run", "examples/bell.qasm", "--shots", "100", "--seed", "3", "--json"],
    ["build/qumo", "transpile", "examples/bell.qasm", "--target", "grid:2x2/sc_cz", "--level", "2"],
    ["build/qumo", "convert", "examples/bell.qasm", "--to", "qasm2"],
    ["build/qumo", "expect", "examples/bell.qasm", "ZZ"],
    ["build/qumo", "info", "-", "--qasm"],
    ["build/qumo", "describe", "gates"],
    ["build/qumo", "version"],
])
def test_build_qumo_read_only_uses_still_run(fake_tree, argv):
    _, calls = fake_tree
    assert repo.run_check.invoke({"argv": argv}).startswith("exit 0")
    assert calls[-1][0] == argv


def test_cli_back_end_has_no_bare_build_qumo_rule():
    from qumo_agents import backends
    from qumo_agents.config import Settings
    for name in registry.AGENTS:
        cmd = backends.claude_code_command(registry.get(name), "t", Settings.load(), None)
        rules = cmd[cmd.index("--allowedTools") + 1].split(",")
        assert "Bash(build/qumo:*)" not in rules, name
        assert not any(r.startswith(("Bash(build/qumo convert", "Bash(build/qumo transpile",
                                     "Bash(build/qumo tool", "Bash(build/qumo mcp")) for r in rules), name


# ----------------------------------------------------------------------------- CLI environment
FAKE_ENV = {"PATH": "/bin", "HOME": "/h", "OPENROUTER_API_KEY": "or", "GITHUB_TOKEN": "gh",
            "AWS_SECRET_ACCESS_KEY": "aws", "IBM_QUANTUM_TOKEN": "ibm", "SSH_AUTH_SOCK": "/s",
            "ANTHROPIC_API_KEY": "ant", "CLAUDE_CODE_OAUTH_TOKEN": "oauth", "OPENAI_API_KEY": "oai",
            "CODEX_HOME": "/c", "ALPHAXIV_API_KEY": "ax"}


def test_cli_env_is_unchanged_by_default_and_scrubbed_on_request():
    from qumo_agents import backends
    from qumo_agents.config import Settings
    raw = Settings.load().raw
    assert backends.claude_code_env(Settings(raw), FAKE_ENV) == FAKE_ENV
    assert backends.codex_env(Settings(raw), FAKE_ENV) == FAKE_ENV
    claude = backends.claude_code_env(Settings({**raw, "claude_code": {"env": "scrub"}}), FAKE_ENV)
    assert claude == {"PATH": "/bin", "HOME": "/h", "SSH_AUTH_SOCK": "/s", "ANTHROPIC_API_KEY": "ant",
                      "CLAUDE_CODE_OAUTH_TOKEN": "oauth", "CODEX_HOME": "/c", "ALPHAXIV_API_KEY": "ax"}
    both = backends.claude_code_env(Settings({**raw, "claude_code": {"env": "scrub", "auth": "subscription"}}),
                                    FAKE_ENV)
    assert "ANTHROPIC_API_KEY" not in both and both["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth"
    codex = backends.codex_env(Settings({**raw, "codex": {"env": "scrub"}}), FAKE_ENV)
    assert codex["OPENAI_API_KEY"] == "oai" and codex["CODEX_HOME"] == "/c"
    assert not {"OPENROUTER_API_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "IBM_QUANTUM_TOKEN",
                "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"} & set(codex)
    with pytest.raises(RuntimeError, match="scrub"):
        backends.claude_code_env(Settings({**raw, "claude_code": {"env": "none"}}), FAKE_ENV)
def _secret_tree(tmp_path):
    root = tmp_path.resolve() / "repo"
    (root / "sub").mkdir(parents=True)
    (root / "secrets_dir").mkdir()
    (root / ".env").write_text("TOKEN=fake-env-value\n")
    (root / "sub" / "secrets.yaml").write_text("key: fake-yaml-value\n")
    (root / "x.pem").write_text("fake-pem-value\n")
    (root / "secrets_dir" / "plain.txt").write_text("fake-in-secret-dir\n")
    (root / "normal.txt").write_text("normal line\n")
    outside = tmp_path.resolve() / "outside.txt"
    outside.write_text("fake-outside-value\n")
    (root / "link.txt").symlink_to(outside)
    (root / "link_to_env.txt").symlink_to(root / ".env")
    return root


def test_search_and_list_skip_secret_files(tmp_path, monkeypatch):
    root = _secret_tree(tmp_path)
    monkeypatch.setitem(repo._state, "sandbox", root)
    found = repo.search_code.invoke({"regex": ".", "glob": "*"})
    listed = repo.list_files.invoke({"path": ".", "pattern": "*"})
    assert "normal.txt" in found and "normal.txt" in listed
    for bad in (".env", "secrets.yaml", "x.pem", "secrets_dir", "plain.txt", "link.txt", "link_to_env"):
        assert bad not in found and bad not in listed
    assert "fake-" not in found
    for p in (".env", "sub/secrets.yaml", "x.pem", "secrets_dir/plain.txt", "link_to_env.txt"):
        assert repo.read_file.invoke({"path": p}).startswith("error:")
