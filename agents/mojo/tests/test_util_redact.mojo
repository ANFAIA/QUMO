"""Tests for qumo_agent.util.redact (DESIGN.md 2.2.4, 8.2)."""

from std.testing import assert_equal, assert_true
from std.os import setenv, unsetenv

from qumo_agent.util.redact import *


def main() raises:
    # redact (moved verbatim from the old llm.mojo)
    assert_equal(redact("key sk-or-v1-abcdef123456 end", ""), "key sk-[REDACTED] end")
    assert_equal(redact("Authorization: Bearer abcdefgh12345", ""), "Authorization: Bearer [REDACTED]")
    assert_equal(redact("risk-assessment ask-me", ""), "risk-assessment ask-me")
    assert_equal(redact("short sk-abc", ""), "short sk-abc")
    assert_equal(redact("value=supersecretvalue!", "supersecretvalue"), "value=[REDACTED]!")
    assert_equal(redact("tiny=abc", "abc"), "tiny=abc")
    assert_equal(redact("é sk-ñandú_12345678", ""), "é sk-ñandú_12345678")

    assert_equal(len(secret_env_names("OPENROUTER_API_KEY")), 3)
    var dedup = secret_env_names("GITHUB_TOKEN")
    assert_equal(len(dedup), 2)
    assert_equal(dedup[0], "GITHUB_TOKEN")
    assert_equal(dedup[1], "ALPHAXIV_API_KEY")
    assert_equal(len(secret_env_names("")), 2)

    _ = setenv("QUMO_TEST_KEY_A", "alpha-value-0001")
    _ = setenv("QUMO_TEST_KEY_B", "beta/value+0002")
    _ = setenv("QUMO_TEST_KEY_C", "gamma value 0003")
    _ = unsetenv("QUMO_TEST_KEY_UNSET")
    _ = setenv("QUMO_TEST_KEY_EMPTY", "")
    var names: List[String] = ["QUMO_TEST_KEY_A", "QUMO_TEST_KEY_B", "QUMO_TEST_KEY_C", "QUMO_TEST_KEY_UNSET", "QUMO_TEST_KEY_EMPTY"]
    var body = String('{"error": {"message": "bad key alpha-value-0001", "echo": "beta/value+0002"}, "x": "gamma value 0003"}')
    assert_equal(
        redact_all(body, names),
        '{"error": {"message": "bad key [REDACTED]", "echo": "[REDACTED]"}, "x": "[REDACTED]"}',
    )
    assert_equal(redact_all("plain alpha-value-0001 and Bearer abcdefgh12345", names), "plain [REDACTED] and Bearer [REDACTED]")
    assert_equal(redact_all("nothing here", names), "nothing here")
    # read at each call, never cached
    _ = setenv("QUMO_TEST_KEY_A", "changed-value-0009")
    assert_equal(redact_all("alpha-value-0001 changed-value-0009", names), "alpha-value-0001 [REDACTED]")
    for n in names:
        _ = unsetenv(n)
    print("test_util_redact OK")
