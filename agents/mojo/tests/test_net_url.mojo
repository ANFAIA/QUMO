"""Tests for net/url.mojo: parsing, RFC 3986 resolution and urllib quoting
(expected values from Python 3.12's urllib.parse)."""

from std.testing import assert_equal, assert_true, assert_false

from qumo_agent.net.url import Url, parse_url, resolve_reference, quote, quote_plus, urlencode


def _refused(url: String, part: String) raises:
    var got = String()
    try:
        _ = parse_url(url)
    except e:
        got = String(e)
    assert_true(got.startswith("QA-E104: invalid URL") and got.find(part) >= 0,
                "expected refusal '" + part + "' for " + url + ", got '" + got + "'")


def test_parse() raises:
    var u = parse_url("HTTPS://OpenRouter.AI/api/v1/chat/completions")
    assert_equal(u.scheme, "https")
    assert_equal(u.host, "openrouter.ai")
    assert_equal(u.port, 443)
    assert_equal(u.path_query, "/api/v1/chat/completions")
    assert_equal(u.host_header(), "openrouter.ai")
    assert_equal(u.origin(), "https://openrouter.ai")
    assert_equal(u.to_string(), "https://openrouter.ai/api/v1/chat/completions")

    var p = parse_url("http://127.0.0.1:8123?x=1#frag")
    assert_equal(p.scheme, "http")
    assert_equal(p.port, 8123)
    assert_equal(p.path_query, "/?x=1")
    assert_equal(p.host_header(), "127.0.0.1:8123")

    var v6 = parse_url("http://[::1]:9000/a")
    assert_equal(v6.host, "::1")
    assert_equal(v6.host_header(), "[::1]:9000")
    assert_equal(v6.to_string(), "http://[::1]:9000/a")

    var d = parse_url("https://example.com:443")
    assert_equal(d.path_query, "/")
    assert_equal(d.host_header(), "example.com")
    var e = parse_url("http://example.com:/x")
    assert_equal(e.port, 80)

    # Non-ASCII path bytes are percent-encoded.
    assert_equal(parse_url("https://h/é?q=ü").path_query, "/%C3%A9?q=%C3%BC")

    _refused("https://user:pw@host/", "user information")
    _refused("ftp://host/", "only http and https")
    _refused("https://host/a b", "control character")
    _refused("https://host/a\r\nX: y", "control character")
    _refused("https://host/\n", "control character")
    _refused("https://host:0/", "invalid port")
    _refused("https://host:99999/", "invalid port")
    _refused("https://host:12a/", "invalid port")
    _refused("https:///path", "invalid host")
    _refused("https://hö.st/", "invalid host")
    _refused("https://[::1/", "unterminated")
    _refused("host/path", "no scheme")
    _refused("http:host/path", "no host")


def test_resolve() raises:
    var base = parse_url("http://a/b/c/d;p?q")
    var refs: List[String] = ["g", "./g", "g/", "/g", "//g", "?y", "g?y", "", "..", "../", "../g",
        "../..", "../../", "../../../g", "/./g", "/../g", "g.", ".g", "g..", "..g", "./../g", "./g/.",
        "g/./h", "g/../h", "g;x=1/./y", "g;x=1/../y", "#s", "g#s"]
    var want: List[String] = ["http://a/b/c/g", "http://a/b/c/g", "http://a/b/c/g/", "http://a/g",
        "http://g/", "http://a/b/c/d;p?y", "http://a/b/c/g?y", "http://a/b/c/d;p?q", "http://a/b/",
        "http://a/b/", "http://a/b/g", "http://a/", "http://a/", "http://a/g", "http://a/g", "http://a/g",
        "http://a/b/c/g.", "http://a/b/c/.g", "http://a/b/c/g..", "http://a/b/c/..g", "http://a/b/g",
        "http://a/b/c/g/", "http://a/b/c/g/h", "http://a/b/c/h", "http://a/b/c/g;x=1/y", "http://a/b/c/y",
        "http://a/b/c/d;p?q", "http://a/b/c/g"]
    for i in range(len(refs)):
        assert_equal(resolve_reference(base, refs[i]).to_string(), want[i], "ref " + refs[i])
    # Absolute and scheme-relative references keep their own host and port.
    var s = parse_url("https://api.github.com/repos/x/y")
    assert_equal(resolve_reference(s, "https://objects.example.org:8443/p/../q?a=1").to_string(),
                 "https://objects.example.org:8443/q?a=1")
    assert_equal(resolve_reference(s, "//cdn.example.org/z").to_string(), "https://cdn.example.org/z")
    assert_equal(resolve_reference(s, "http://down.example/").scheme, "http")
    var bad = String()
    try:
        _ = resolve_reference(s, "/x\r\nSet-Cookie: a")
    except e:
        bad = String(e)
    assert_true(bad.startswith("QA-E104"))


def test_quote() raises:
    assert_equal(quote("qiskit-ibm runtime/é~*", ""), "qiskit-ibm%20runtime%2F%C3%A9~%2A")
    assert_equal(quote("a b/c?d=é", "/"), "a%20b/c%3Fd%3D%C3%A9")
    assert_equal(quote_plus("a b+c&d=é/"), "a+b%2Bc%26d%3D%C3%A9%2F")
    assert_equal(quote_plus("plain"), "plain")
    # The arXiv query of mojo_gaps section 2, byte-exact.
    var names: List[String] = ["search_query", "sortBy", "sortOrder", "max_results"]
    var values: List[String] = ['(all:"qubit routing") AND (cat:quant-ph OR cat:cs.ET)', "submittedDate",
                                "descending", "10"]
    assert_equal(urlencode(names, values),
        "search_query=%28all%3A%22qubit+routing%22%29+AND+%28cat%3Aquant-ph+OR+cat%3Acs.ET%29"
        + "&sortBy=submittedDate&sortOrder=descending&max_results=10")
    assert_equal(urlencode(List[String](), List[String]()), "")


def main() raises:
    test_parse()
    test_resolve()
    test_quote()
    print("test_net_url OK")
