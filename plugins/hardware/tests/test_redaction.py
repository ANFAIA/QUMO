"""Credential redaction in error texts (errors.redact_text / safe_exc_text, CLI QM-E999).

Every key below is made up for the test. Both directions are checked: what must be
redacted, and what must be kept (job ids, hashes, ordinary sentences).
"""

from __future__ import annotations

import json
import urllib.parse

import pytest

from qumo_hardware import cli
from qumo_hardware.errors import redact_text, safe_exc_text

KEY32 = "Zq3vT8mK1xR7pL2wN9cB4dF6hJ0sYaUe"          # 32 characters, made up
BASIC = "bWFkZS11cDpub3QtYS1yZWFsLXBhc3M="           # base64 of a made-up user:password
ENV_VARS = ("QISKIT_IBM_TOKEN", "IONQ_API_KEY", "QBRAID_API_KEY", "QUMO_SPEQTRUM_APIKEY")


@pytest.fixture(autouse=True)
def no_credentials_in_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- must redact
@pytest.mark.parametrize("text, expected", [
    (f"x-api-key: {KEY32}", "x-api-key: <redacted>"),
    (f"Invalid credentials key {KEY32}", "Invalid credentials key <redacted>"),
    (f"Authorization: Basic {BASIC}", "Authorization: Basic <redacted>"),
    (f"authorization: bearer {KEY32}", "authorization: bearer <redacted>"),
    (f"Authorization: {KEY32}", "Authorization: <redacted>"),
    (f"sent header basic {BASIC} to the server", "sent header basic <redacted> to the server"),
    ("password=hunter2!x", "password=<redacted>"),
    ("secret=abc123", "secret=<redacted>"),
    (f"access_token={KEY32}", "access_token=<redacted>"),
    (f"GET https://api.example.test/v1/jobs?api_key={KEY32}&limit=5",
     "GET https://api.example.test/v1/jobs?api_key=<redacted>&limit=5"),
    ('{"password": "two words", "user": "ana"}', '{"password": <redacted>, "user": "ana"}'),
    (f"token {KEY32} was rejected", "token <redacted> was rejected"),
    (f"apikey:{KEY32}", "apikey:<redacted>"),
    ("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJtYWRlLXVwIn0.c2lnbmF0dXJl end", "jwt <redacted> end"),
    ("opaque " + "Ab1_" * 12, "opaque <redacted>"),
])
def test_patterns_are_redacted(text, expected):
    assert redact_text(text) == expected
    assert safe_exc_text(RuntimeError(text)) == f"RuntimeError: {expected}"


@pytest.mark.parametrize("name", ENV_VARS)
def test_current_value_of_each_credential_variable_is_redacted(monkeypatch, name):
    value = 'k9/Qw"7+z e\\Xy'                       # made up; needs escaping in a URL and in JSON
    monkeypatch.setenv(name, value)
    forms = [value, urllib.parse.quote(value, safe=""), urllib.parse.quote_plus(value), json.dumps(value)[1:-1]]
    assert len(set(forms)) == 4
    for form in forms:
        out = redact_text(f"the server said: bad value [{form}] here")
        assert out == "the server said: bad value [<redacted>] here", form


def test_value_is_read_at_each_call_and_other_secret_named_variables_count(monkeypatch):
    monkeypatch.setenv("IONQ_API_KEY", "first-made-up-1")
    assert redact_text("echo first-made-up-1") == "echo <redacted>"
    monkeypatch.setenv("IONQ_API_KEY", "second-made-up-2")
    assert redact_text("echo first-made-up-1 second-made-up-2") == "echo first-made-up-1 <redacted>"
    monkeypatch.setenv("SOME_VENDOR_TOKEN", "plainlookingvalue")
    assert redact_text("got plainlookingvalue") == "got <redacted>"


def test_short_values_rule(monkeypatch):
    # 4 to 7 characters: only as a whole word; under 4: never (documented trade-off)
    monkeypatch.setenv("IONQ_API_KEY", "ab12z")
    assert redact_text("got ab12z, cab12zd and (ab12z)") == "got <redacted>, cab12zd and (<redacted>)"
    monkeypatch.setenv("IONQ_API_KEY", "abc")
    assert redact_text("abc and cabcd") == "abc and cabcd"


# --------------------------------------------------------------------------- must keep
@pytest.mark.parametrize("text", [
    "job d1f2a3b4c5d6e7f8a9b0 is QUEUED",                                       # IBM-style job id
    "Nexus job '0b6f1c1e-7d2a-4c53-9a41-2f8e6d3b9c10' not found",               # UUID
    "job job-0b6f1c1e-7d2a-4c53-9a41-2f8e6d3b9c10 not found",                   # prefixed UUID (40 characters)
    "payload sha256 9f86d081884c7d659a2fbd3ca0c0b1f1e7b6a9c2d4e5f60718293a4b5c6d7e8f",
    "commit da39a3ee5e6b4b0d3255bfef95601890afd80709 is missing",
    "md5 d41d8cd98f00b204e9800998ecf8427e",
    "the token expired; log in again",
    "IonQ API key not found: the IONQ_API_KEY environment variable is not set",
    "use basic authentication for this endpoint",
    "KeyError: 'shots'",
    "backend ibm_torino has 133 qubits; the circuit needs 140",
    "max_tokens exceeded for monkey business",
])
def test_ordinary_text_is_kept(text):
    assert redact_text(text) == text


def test_safe_exc_text_keeps_type_first_line_and_limit():
    assert safe_exc_text(ValueError("line one\nline two")) == "ValueError: line one"
    assert safe_exc_text(ValueError("")) == "ValueError"
    out = safe_exc_text(ValueError("y" * 30 + " " + "z " * 400), limit=50)
    assert len(out) == len("ValueError: ") + 50 and out.endswith("...")


# --------------------------------------------------------------------------- CLI
@pytest.mark.parametrize("as_json", [False, True])
def test_cli_unexpected_error_is_redacted(monkeypatch, capsys, as_json):
    monkeypatch.setenv("IONQ_API_KEY", "made-up/env+value 77")

    def boom(args):
        raise RuntimeError(f"401 from server, x-api-key: {KEY32}, echo made-up/env+value 77, job d1f2a3b4c5d6e7f8a9b0")

    monkeypatch.setattr(cli, "cmd_jobs", boom)
    assert cli.main(["jobs"] + (["--json"] if as_json else [])) == cli.EXIT_ERR
    cap = capsys.readouterr()
    shown = cap.out + cap.err
    assert KEY32 not in shown and "made-up/env+value 77" not in shown
    assert "QM-E999" in shown and "unexpected RuntimeError: 401 from server" in shown
    assert "x-api-key: <redacted>" in shown and "job d1f2a3b4c5d6e7f8a9b0" in shown


def test_cli_debug_traceback_is_redacted(monkeypatch, capsys):
    monkeypatch.setenv("QUMO_HW_DEBUG", "1")

    def boom(args):
        raise RuntimeError(f"access_token={KEY32}")

    monkeypatch.setattr(cli, "cmd_jobs", boom)
    assert cli.main(["jobs"]) == cli.EXIT_ERR
    err = capsys.readouterr().err
    assert KEY32 not in err and "Traceback" in err and "access_token=<redacted>" in err
