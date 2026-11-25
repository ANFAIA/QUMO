"""Tests for net/guard.mojo: which addresses are globally routable, the IP literal spellings,
and the URL rules of the research GETs. Pure: nothing here resolves a name or opens a socket.

The address lists were written by agents/.venv's Python 3.12.13 with
`ipaddress.ip_address(x).is_global` (an IPv4-mapped IPv6 address judged as its IPv4 address,
as `_public_host` in qumo_agents/tools/research.py does).
"""

from std.testing import assert_equal, assert_true, assert_false

from qumo_agent.net.url import parse_url
from qumo_agent.net.guard import (
    UrlGuard, no_guard, ipv4_is_global, ipv6_is_global, parse_ipv4_literal, parse_ipv6_literal,
    literal_is_global,
)


def test_addresses() raises:
    var global4: List[String] = [
        "1.1.1.1", "8.8.8.8", "9.255.255.255", "11.0.0.0", "100.63.255.255", "100.128.0.0", "126.255.255.255",
        "128.0.0.0", "169.253.255.255", "169.255.0.0", "172.15.255.255", "172.32.0.0", "192.0.0.9",
        "192.0.0.10", "192.0.1.0", "192.0.3.0", "192.167.255.255", "192.169.0.0", "198.17.255.255",
        "198.20.0.0", "198.51.101.0", "203.0.114.0", "223.255.255.255", "224.0.0.1", "239.255.255.255",
        "140.82.112.3", "151.101.0.223",
    ]
    var other4: List[String] = [
        "0.0.0.0", "0.1.2.3", "10.0.0.0", "10.0.0.5", "10.255.255.255", "100.64.0.0", "100.64.0.1",
        "100.127.255.255", "127.0.0.1", "127.255.255.255", "169.254.0.1", "169.254.169.254", "172.16.0.1",
        "172.31.255.255", "192.0.0.1", "192.0.0.8", "192.0.0.11", "192.0.0.170", "192.0.2.1", "192.168.0.1",
        "192.168.255.255", "198.18.0.1", "198.19.255.255", "198.51.100.7", "203.0.113.9", "240.0.0.1",
        "255.255.255.255",
    ]
    var global6: List[String] = [
        "::2", "::ffff:8.8.8.8", "::127.0.0.1", "64:ff9b::7f00:1", "64:ff9b:2::1", "100:0:0:1::1", "2001:1::1",
        "2001:1::2", "2001:3::1", "2001:4:112::1", "2001:20::1", "2001:2f::1", "2001:30::1", "2001:3f::1",
        "2001:200::1", "2001:db9::1", "2003::1", "3fff:1000::1", "4000::1", "fe00::1", "fec0::1", "ff02::1",
        "2606:4700:4700::1111", "2a04:4e42::223", "1:2:3:4:5:6:7:8", "1:2:3:4:5:6:1.2.3.4",
    ]
    var other6: List[String] = [
        "::", "::1", "::ffff:127.0.0.1", "::ffff:10.0.0.5", "::ffff:169.254.169.254", "::ffff:100.64.0.1",
        "::ffff:7f00:1", "64:ff9b:1::1", "100::1", "2001::1", "2001:1::3", "2001:4:113::1", "2001:40::1",
        "2001:1ff:ffff::1", "2001:db8::1", "2002::1", "2002:7f00:1::1", "3fff::1", "3fff:fff::1", "fc00::1",
        "fd00::1", "fdff:ffff::1", "fe80::1", "febf::1", "fe80::1%en0",
    ]
    for a in global4:
        assert_true(ipv4_is_global(parse_ipv4_literal(a)), a)
        assert_equal(literal_is_global(a), 1, a)
    for a in other4:
        assert_true(parse_ipv4_literal(a) >= 0, a)
        assert_false(ipv4_is_global(parse_ipv4_literal(a)), a)
        assert_equal(literal_is_global(a), 0, a)
    for a in global6:
        assert_true(ipv6_is_global(parse_ipv6_literal(a)), a)
        assert_equal(literal_is_global(a), 1, a)
    for a in other6:
        assert_equal(len(parse_ipv6_literal(a)), 16, a)
        assert_false(ipv6_is_global(parse_ipv6_literal(a)), a)
        assert_equal(literal_is_global(a), 0, a)
    assert_equal(len(global4) + len(other4), 54)
    assert_equal(len(global6) + len(other6), 51)
    assert_false(ipv4_is_global(-1))
    assert_false(ipv4_is_global(1 << 32))
    assert_false(ipv6_is_global(List[UInt8]()))


def test_literals() raises:
    # every spelling inet_aton (and so getaddrinfo) reads as 127.0.0.1
    for a in ["127.0.0.1", "127.1", "127.0.1", "2130706433", "0x7f000001", "0X7F000001", "0x7f.1", "0x7f.0.0.1",
              "0177.0.0.1", "017700000001", "0177.0.0x1", "127.0.0.01"]:
        assert_equal(parse_ipv4_literal(a), 0x7F000001, a)
        assert_equal(literal_is_global(a), 0, a)
    assert_equal(parse_ipv4_literal("0"), 0)
    assert_equal(parse_ipv4_literal("10.5"), 0x0A000005)
    assert_equal(parse_ipv4_literal("169.254.43518"), 0xA9FEA9FE)      # 169.254.169.254
    assert_equal(parse_ipv4_literal("4294967295"), 0xFFFFFFFF)
    assert_equal(parse_ipv4_literal("134744072"), 0x08080808)
    assert_equal(literal_is_global("134744072"), 1)
    # names and malformed numbers are not literals: they are judged after resolution
    for a in ["", "example.org", "localhost", "pypi.org.", "4294967296", "256.0.0.1", "1.2.3.4.5", "1..2", "08", "0x",
              "0xg", "1.2.3.256", "1.2.65536", "127.0.0.1.", "a.1", "1e3", "-1"]:
        assert_equal(parse_ipv4_literal(a), -1, a)
        assert_equal(literal_is_global(a), -1, a)
    assert_equal(len(parse_ipv6_literal("::1")), 16)
    assert_equal(Int(parse_ipv6_literal("::1")[15]), 1)
    var m = parse_ipv6_literal("::ffff:10.0.0.5")
    assert_equal(Int(m[10]), 0xFF)
    assert_equal(Int(m[12]), 10)
    assert_equal(Int(m[15]), 5)
    var f = parse_ipv6_literal("fe80::1%en0")
    assert_equal(Int(f[0]), 0xFE)
    assert_equal(Int(f[1]), 0x80)
    for a in ["", ":", ":::", "1::2::3", "1:2:3:4:5:6:7", "1:2:3:4:5:6:7:8:9", "1:2:3:4:5:6:7::8", "12345::1", "g::1",
              "1.2.3.4::1", "::1.2.3", "::1.2.3.4:5", "::1.2.3.256", "1:2:3:4:5:6:7:1.2.3.4"]:
        assert_equal(len(parse_ipv6_literal(a)), 0, a)
        assert_equal(literal_is_global(a), -1, a)


def _check(guard: UrlGuard, url: String) -> String:
    """"" when the guard lets `url` through, else the refusal."""
    try:
        _ = guard.check(parse_url(url))
        return String()
    except e:
        return String(e)


def test_url_rules() raises:
    var page = UrlGuard(True, List[String](), String())      # fetch_page: any public host
    var api = UrlGuard(True, ["export.arxiv.org", "pypi.org", "api.github.com"], String())
    assert_equal(_check(page, "https://example.org/x"), "")
    assert_equal(_check(page, "https://example.org:443/x"), "")
    assert_equal(_check(page, "https://8.8.8.8/x"), "")
    assert_equal(_check(page, "https://[2606:4700:4700::1111]/x"), "")
    assert_true(page.check(parse_url("https://example.org/x")))       # the resolved addresses are still to be checked
    assert_equal(_check(page, "http://example.org/x"), "QA-E104: refused URL: only https URLs are fetched")
    assert_equal(_check(page, "http://export.arxiv.org/api/query"), "QA-E104: refused URL: only https URLs are fetched")
    assert_equal(_check(page, "https://example.org:8443/x"), "QA-E104: refused URL: only the default https port is fetched")
    assert_equal(_check(page, "https://example.org:80/x"), "QA-E104: refused URL: only the default https port is fetched")
    assert_equal(_check(page, "https://127.0.0.1:8080/x"), "QA-E104: refused URL: only the default https port is fetched")
    for h in ["127.0.0.1", "10.0.0.5", "169.254.169.254", "169.254.0.1", "192.168.1.1", "172.16.0.1", "100.64.0.1",
              "0.0.0.0", "2130706433", "0x7f000001", "0x7f.1", "127.1", "0177.0.0.1", "[::1]", "[::]", "[::ffff:127.0.0.1]",
              "[::ffff:10.0.0.5]", "[::ffff:a9fe:a9fe]", "[fe80::1]", "[fd00::1]", "[2002:7f00:1::1]"]:
        var bare = String(h).replace("[", "").replace("]", "")
        assert_equal(_check(page, "https://" + String(h) + "/x"), "QA-E104: refused URL: " + bare + " is not a public address", h)
        assert_equal(_check(page, "https://" + String(h) + ":443/latest/meta-data/"),
                     "QA-E104: refused URL: " + bare + " is not a public address", h)
    # the allow-list of the API tools
    assert_equal(_check(api, "https://pypi.org/pypi/qiskit/json"), "")
    assert_equal(_check(api, "https://API.GitHub.com/repos/a/b/releases"), "")
    assert_true(api.check(parse_url("https://export.arxiv.org/api/query")))
    for h in ["example.org", "pypi.org.evil.example", "evilpypi.org", "pypi.org.", "github.com", "8.8.8.8", "127.0.0.1"]:
        assert_equal(_check(api, "https://" + String(h) + "/x"),
                     "QA-E104: refused URL: " + String(h) + " is not on the allow-list for this tool", h)
    assert_equal(_check(api, "https://pypi.org:8443/x"), "QA-E104: refused URL: only the default https port is fetched")
    assert_equal(_check(api, "http://pypi.org/x"), "QA-E104: refused URL: only https URLs are fetched")
    # the test seam: only URLs under the prefix skip the rules
    var seam = UrlGuard(True, ["pypi.org"], "http://127.0.0.1:8123/")
    assert_equal(_check(seam, "http://127.0.0.1:8123/pypi/x/json"), "")
    assert_false(seam.check(parse_url("http://127.0.0.1:8123/pypi/x/json")))
    assert_equal(_check(seam, "http://127.0.0.1.evil.example:8123/x"), "QA-E104: refused URL: only https URLs are fetched")
    assert_equal(_check(seam, "http://127.0.0.1:8124/x"), "QA-E104: refused URL: only https URLs are fetched")
    assert_equal(_check(seam, "https://127.0.0.1/x"), "QA-E104: refused URL: 127.0.0.1 is not on the allow-list for this tool")
    # off: the model client and MCP HTTP keep their own rules
    assert_equal(_check(no_guard(), "http://127.0.0.1:8080/x"), "")
    assert_false(no_guard().check(parse_url("https://example.org/x")))


def main() raises:
    test_addresses()
    test_literals()
    test_url_rules()
    print("test_net_guard OK")
