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

    # Bearer / Basic in any letter case; the whole value goes, dots and base64 signs included
    assert_equal(redact("authorization: bearer abcdefgh12345", ""), "authorization: bearer [REDACTED]")
    assert_equal(redact("AUTHORIZATION: BEARER abcdefgh12345", ""), "AUTHORIZATION: BEARER [REDACTED]")
    assert_equal(redact("Bearer eyJhbGciOiJ9.eyJzdWIiOiJ4In0.c2lnbmF0dXJl end", ""), "Bearer [REDACTED] end")
    assert_equal(redact("Authorization: Basic bWFkZS11cDpub3QtcmVhbA== end", ""), "Authorization: Basic [REDACTED] end")
    assert_equal(redact("authorization: basic dXNlcjpwYXNz", ""), "authorization: basic [REDACTED]")
    # must keep: prose after the scheme words, short values, words that only contain them
    assert_equal(redact("use basic authentication here", ""), "use basic authentication here")
    assert_equal(redact("Basic Authentication is off", ""), "Basic Authentication is off")
    assert_equal(redact("bearer abc", ""), "bearer abc")
    assert_equal(redact("forbearer abcdefgh12345", ""), "forbearer abcdefgh12345")
    assert_equal(redact("job d1f2a3b4c5d6e7f8a9b0 sha 9f86d081884c7d659a2fbd3ca0c0b1f1e7b6a9c2", ""),
                 "job d1f2a3b4c5d6e7f8a9b0 sha 9f86d081884c7d659a2fbd3ca0c0b1f1e7b6a9c2")

    # JSON-escaped form of a known value (a server echoing the key inside a JSON string)
    var odd = String('ab"cd\\ef-0001')
    assert_equal(redact('{"echo": "ab\\"cd\\\\ef-0001"} raw ab"cd\\ef-0001', odd), '{"echo": "[REDACTED]"} raw [REDACTED]')

    # short values: 4 to 7 bytes only as a whole word, under 4 never
    assert_equal(redact("got ab12z, cab12zd and (ab12z)", "ab12z"), "got [REDACTED], cab12zd and ([REDACTED])")
    assert_equal(redact("ab12z", "ab12z"), "[REDACTED]")
    assert_equal(redact("abc and cabcd", "abc"), "abc and cabcd")

    # every variable of the process whose name looks secret, not only the listed ones;
    # a value that contains another one is replaced whole
    _ = setenv("QUMO_TEST_OTHER_TOKEN", "omega-value-0004")
    _ = setenv("QUMO_TEST_LONGER_API_KEY", "omega-value-0004-and-more")
    var none = List[String]()
    assert_equal(redact_all("a omega-value-0004 b omega-value-0004-and-more c", none), "a [REDACTED] b [REDACTED] c")
    _ = setenv("QUMO_TEST_OTHER_TOKEN", 'om"ega-0005')
    assert_equal(redact_all('{"m": "bad om\\"ega-0005"}', none), '{"m": "bad [REDACTED]"}')
    _ = unsetenv("QUMO_TEST_OTHER_TOKEN")
    _ = unsetenv("QUMO_TEST_LONGER_API_KEY")
    print("test_util_redact OK")
