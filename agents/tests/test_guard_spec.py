"""Every vector of agents/spec/guard_rules.json through the real functions of the Python
harness. agents/mojo/tests/test_guard_spec.mojo runs the same vectors through the Mojo
harness: the two tests reading one file is what keeps the two sets of rules from drifting.

Nothing here opens a socket: name resolution is replaced by the addresses a vector lists, and
an IP literal goes through getaddrinfo with AI_NUMERICHOST (a parse, never a lookup).
"""

from __future__ import annotations

import importlib.util
import os
import re
import socket
import urllib.request
from pathlib import Path

import pytest

from qumo_agents import backends, config, guard_spec, registry, untrusted
from qumo_agents.tools import repo, research

SPEC = guard_spec.SPEC
VECTORS = SPEC["vectors"]
HARNESS = "python"
RULES = {"path", "command", "url", "check_env", "secret_env_name", "cli_scrub_env", "model_key", "delegation",
         "delegation_shipped", "untrusted_wrap"}


def _expected(vec: dict) -> tuple[str, str | None]:
    over = vec.get("by_harness", {}).get(HARNESS, {})
    if "expect" in over:  # another decision in this harness: the reason is its own, if any
        return over["expect"], over.get("reason")
    return vec["expect"], over.get("reason", vec.get("reason"))


def _reason(rule: str, text: str) -> str:
    """The reason class of a refusal text: the first class of the rule with a marker in it."""
    for name, markers in SPEC["reasons"][rule].items():
        if any(m in text for m in markers):
            return name
    return f"unclassified: {text}"


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> Path:
    """The file tree of `path_fixture`; returns its root."""
    base = tmp_path_factory.mktemp("guard-spec").resolve()
    fx = SPEC["path_fixture"]
    for rel in fx["files"]:
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    for link, target in fx["symlinks"]:
        (base / link).symlink_to(target)
    return base / "root"


# ----------------------------------------------------------------------------- the rules
def run_path(inp: dict, tree: Path, monkeypatch) -> str:
    try:
        repo._resolve(inp["path"], tree)
    except ValueError as e:
        return _reason("path", str(e))
    return ""


def run_command(inp: dict, tree: Path, monkeypatch) -> str:
    monkeypatch.setattr(repo, "REPO_ROOT", tree)
    monkeypatch.setitem(repo._state, "sandbox", tree if inp["sandbox_open"] else None)
    why = repo._run_check_refusal(list(inp["argv"]))
    return _reason("command", why) if why else ""


def _fake_resolver(table: dict[str, list[str]]):
    real = socket.getaddrinfo

    def getaddrinfo(host, port, *args, **kwargs):
        if host in table:
            return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port))
                    for a in table[host]]
        try:  # an IP literal in any spelling the platform reads; never a DNS query
            return real(host, port, type=socket.SOCK_STREAM, flags=socket.AI_NUMERICHOST)
        except socket.gaierror:
            raise AssertionError(f"the vector must list the addresses of {host!r} in `resolve`") from None

    return getaddrinfo


def run_url(inp: dict, tree: Path, monkeypatch) -> str:
    """The first URL through _check_url, every further hop through the redirect handler."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_resolver(inp.get("resolve", {})))
    hosts = research.API_HOSTS if inp["hosts"] == "api" else None
    hops = inp["hops"]
    handler = research._CheckedRedirects(hosts)
    try:
        research._check_url(hops[0], hosts)
        req = urllib.request.Request(hops[0])
        for target in hops[1:]:
            req = handler.redirect_request(req, None, 302, "Found", {}, target)
    except ValueError as e:
        return _reason("url", str(e))
    return ""


def run_check_env(inp: dict, tree: Path, monkeypatch) -> str:
    monkeypatch.setenv(inp["name"], "value")
    return "" if inp["name"] in repo._check_env() else "dropped"


def run_secret_env_name(inp: dict, tree: Path, monkeypatch) -> str:
    return "secret" if backends.CLI_SECRET_ENV.search(inp["name"]) else ""


def run_cli_scrub_env(inp: dict, tree: Path, monkeypatch) -> str:
    return "" if inp["name"] in backends.scrub_env({inp["name"]: "value"}, inp["section"]) else "dropped"


def run_model_key(inp: dict, tree: Path, monkeypatch) -> str:
    try:
        config.check_key_url(inp["base_url"], "TEST_KEY_ENV", "value" if inp["key"] else "")
    except config.ConfigError as e:
        return _reason("model_key", str(e))
    return ""


def _specs(agents: dict) -> dict[str, registry.AgentSpec]:
    return {name: registry.AgentSpec(name, "", "", tuple(a["tool_groups"]), (), tuple(a["subagents"]),
                                     untrusted_subagents=tuple(a["untrusted_subagents"]))
            for name, a in agents.items()}


def run_delegation(inp: dict, tree: Path, monkeypatch) -> str:
    try:
        registry.check_web_exec(_specs(inp["agents"]))
    except RuntimeError as e:
        return _reason("delegation", str(e))
    return ""


def run_delegation_shipped(inp: dict, tree: Path, monkeypatch) -> str:
    registry.check_web_exec()
    return ""


RUNNERS = {"path": run_path, "command": run_command, "url": run_url, "check_env": run_check_env,
           "secret_env_name": run_secret_env_name, "cli_scrub_env": run_cli_scrub_env, "model_key": run_model_key,
           "delegation": run_delegation, "delegation_shipped": run_delegation_shipped}


@pytest.mark.parametrize("vec", [v for v in VECTORS if v["rule"] != "untrusted_wrap"], ids=lambda v: v["id"])
def test_vector(vec, tree, monkeypatch):
    if HARNESS not in vec.get("harnesses", [HARNESS]):
        pytest.skip(vec.get("difference", "not for this harness"))
    expect, reason = _expected(vec)
    got = RUNNERS[vec["rule"]](vec["input"], tree, monkeypatch)
    if expect == "allowed":
        assert got == "", f"{vec['id']}: expected allowed, refused with {got}"
    else:
        assert got != "", f"{vec['id']}: expected a refusal, was allowed"
        if reason:
            assert got == reason, vec["id"]


@pytest.mark.parametrize("vec", [v for v in VECTORS if v["rule"] == "untrusted_wrap"], ids=lambda v: v["id"])
def test_untrusted_wrap_vector(vec):
    inp = vec["input"]
    out = untrusted.wrap(inp["text"], inp["source"], inp["limit"])
    token = re.match(rf"<<<{untrusted.MARKER} ([0-9a-f]{{8}}) ", out).group(1)
    assert out.replace(token, "TOKEN") == vec["output"]


# ----------------------------------------------------------------------------- the file itself
def test_spec_is_well_formed():
    ids = [v["id"] for v in VECTORS]
    assert len(ids) == len(set(ids)), "vector ids are unique"
    for v in VECTORS:
        assert v["rule"] in RULES, v["id"]
        assert v["expect"] in ("allowed", "refused", "wrapped"), v["id"]
        if "reason" in v:
            assert v["reason"] in SPEC["reasons"][v["rule"]], v["id"]
        # a difference between the harnesses is never silent: it carries its reason
        if "by_harness" in v or "harnesses" in v:
            assert v.get("difference"), v["id"]
        assert set(v.get("harnesses", [])) <= {"python", "mojo"}, v["id"]
    for rule, classes in SPEC["reasons"].items():
        used = {v.get("reason") for v in VECTORS if v["rule"] == rule}
        assert set(classes) <= used, f"{rule}: every reason class has a vector ({set(classes) - used})"


def test_lists_are_the_ones_in_use():
    """The tables of the harness are the lists of the file, not copies."""
    r, e = guard_spec.REPO, guard_spec.ENV
    assert repo.SKIP_DIRS == set(r["skip_dirs"]) and sorted(r["skip_dirs"]) == r["skip_dirs"]
    assert [list(p) for _, p in repo.ALLOWED] == [p for _, p in r["run_check_allowed"]]
    assert repo.GIT_FLAGS == set(r["git_flags"]) and repo.QUMO_SUBCOMMANDS == set(r["qumo_subcommands"])
    assert repo.QUMO_OPTIONS == set(r["qumo_options"]) and repo.QUMO_OUTPUT_OPTIONS == set(r["qumo_output_options"])
    assert research.API_HOSTS == set(guard_spec.RESEARCH["api_hosts"])
    assert research.MAX_REDIRECTS == guard_spec.RESEARCH["max_redirects"]
    assert registry.WEB_GROUPS == set(guard_spec.DELEGATION["web_groups"])
    assert registry.EXEC_GROUPS == set(guard_spec.DELEGATION["exec_groups"])
    assert set(backends.CLI_OWN_ENV) == set(e["cli_own_prefixes"])
    for word in e["secret_name_substrings"]:
        assert backends.CLI_SECRET_ENV.search("X_" + word.lower()) and repo.SECRET_ENV.search(word)
    for word in e["check_env_extra_substrings"]:
        assert repo.SECRET_ENV.search(word) and not backends.CLI_SECRET_ENV.search(word)
    assert untrusted.MARKER == guard_spec.UNTRUSTED["marker"]


def test_generated_mojo_rules_are_current():
    """agents/mojo/qumo_agent/core/guard_rules.mojo is generated from the file; a stale copy
    fails here (and in test_guard_spec.mojo, which compares it with the file value by value)."""
    gen_path = guard_spec.SPEC_PATH.parent / "gen_mojo_rules.py"
    mod_spec = importlib.util.spec_from_file_location("gen_mojo_rules", gen_path)
    gen = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(gen)
    assert gen.TARGET.read_text() == gen.render(SPEC), (
        f"{gen.TARGET} is stale: run `python3 agents/spec/gen_mojo_rules.py`")
    assert os.path.exists(gen.TARGET)
