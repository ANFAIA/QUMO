"""Which URLs and addresses a research GET may reach (the rules of `_check_url` and
`_public_host` in `qumo_agents/tools/research.py`).

`UrlGuard.check` is applied to the first URL and to every redirect target before a
connection is opened: https only, port 443 only, the host on the tool's allow-list when it
has one, and no IP literal that is not globally routable. For a host name the addresses are
checked after resolution, in `tcp_connect(..., public_only=True)`: the connection is refused
when any address the name resolves to is not global, and the socket then connects to those
same addresses (one `getaddrinfo` call, so there is no second lookup to rebind).

"Global" is `ipaddress.ip_address(x).is_global` of CPython 3.12.13, with an IPv4-mapped IPv6
address judged as its IPv4 address. Everything here is pure (no libc, no network).

Refusals are raised as `QM-E959: refused URL: <text>`, where `<text>` is the message of the
`ValueError` Python raises for the same URL.
"""

from qumo.net.errors import E_NETWORK
from qumo.net.url import Url

comptime REFUSED = "refused URL: "


def refusal(text: String) -> Error:
    return Error(E_NETWORK + ": " + REFUSED + text)


# ---------------------------------------------------------------------------
# Address classes
# ---------------------------------------------------------------------------
def _v4(a: Int, b: Int, c: Int, d: Int) -> Int:
    return (a << 24) | (b << 16) | (c << 8) | d


def _in4(v: Int, net: Int, bits: Int) -> Bool:
    return (v >> (32 - bits)) == (net >> (32 - bits))


def ipv4_is_global(v: Int) -> Bool:
    """`IPv4Address(v).is_global` for the 32-bit value `v`."""
    if v < 0 or v > 0xFFFFFFFF:
        return False
    if v == _v4(192, 0, 0, 9) or v == _v4(192, 0, 0, 10):
        return True
    if _in4(v, _v4(100, 64, 0, 0), 10):          # shared address space (CGNAT)
        return False
    return not (
        _in4(v, _v4(0, 0, 0, 0), 8)              # "this network"
        or _in4(v, _v4(10, 0, 0, 0), 8)          # private
        or _in4(v, _v4(127, 0, 0, 0), 8)         # loopback
        or _in4(v, _v4(169, 254, 0, 0), 16)      # link-local (cloud metadata)
        or _in4(v, _v4(172, 16, 0, 0), 12)       # private
        or _in4(v, _v4(192, 0, 0, 0), 24)        # IETF protocol assignments
        or _in4(v, _v4(192, 0, 2, 0), 24)        # documentation
        or _in4(v, _v4(192, 168, 0, 0), 16)      # private
        or _in4(v, _v4(198, 18, 0, 0), 15)       # benchmarking
        or _in4(v, _v4(198, 51, 100, 0), 24)     # documentation
        or _in4(v, _v4(203, 0, 113, 0), 24)      # documentation
        or _in4(v, _v4(240, 0, 0, 0), 4)         # reserved, broadcast
    )


def _in6(g: List[Int], net: List[Int], bits: Int) -> Bool:
    """Whether the 8 groups `g` start with the first `bits` bits of `net`."""
    var full = bits // 16
    for i in range(full):
        var want = net[i] if i < len(net) else 0
        if g[i] != want:
            return False
    var rest = bits % 16
    if rest > 0:
        var want = net[full] if full < len(net) else 0
        if (g[full] >> (16 - rest)) != (want >> (16 - rest)):
            return False
    return True


def ipv6_is_global(b: List[UInt8]) -> Bool:
    """`IPv6Address(b).is_global` for the 16 bytes `b`; an IPv4-mapped address
    (`::ffff:a.b.c.d`) is judged as `a.b.c.d`."""
    if len(b) != 16:
        return False
    var g = List[Int](capacity=8)
    for i in range(8):
        g.append((Int(b[2 * i]) << 8) | Int(b[2 * i + 1]))
    if g[0] == 0 and g[1] == 0 and g[2] == 0 and g[3] == 0 and g[4] == 0 and g[5] == 0xFFFF:
        return ipv4_is_global((g[6] << 16) | g[7])
    # The exceptions inside 2001::/23.
    if g[0] == 0x2001 and g[1] == 1 and g[2] == 0 and g[3] == 0 and g[4] == 0 and g[5] == 0 and g[6] == 0 and (g[7] == 1 or g[7] == 2):
        return True
    if _in6(g, [0x2001, 3], 32) or _in6(g, [0x2001, 4, 0x112], 48) or _in6(g, [0x2001, 0x20], 28) or _in6(g, [0x2001, 0x30], 28):
        return True
    return not (
        _in6(g, [0, 0, 0, 0, 0, 0, 0, 1], 128)   # loopback
        or _in6(g, [0], 128)                     # unspecified
        or _in6(g, [0x64, 0xFF9B, 1], 48)        # local-use NAT64
        or _in6(g, [0x100], 64)                  # discard
        or _in6(g, [0x2001], 23)                 # IETF protocol assignments
        or _in6(g, [0x2001, 0xDB8], 32)          # documentation
        or _in6(g, [0x2002], 16)                 # 6to4
        or _in6(g, [0x3FFF], 20)                 # documentation
        or _in6(g, [0xFC00], 7)                  # unique local
        or _in6(g, [0xFE80], 10)                 # link-local
    )


# ---------------------------------------------------------------------------
# IP literals
# ---------------------------------------------------------------------------
def _digit_value(c: Int) -> Int:
    if c >= 0x30 and c <= 0x39:
        return c - 0x30
    if c >= 0x61 and c <= 0x66:
        return c - 0x61 + 10
    if c >= 0x41 and c <= 0x46:
        return c - 0x41 + 10
    return -1


def _aton_part(p: String) -> Int:
    """One number of an `inet_aton` address: decimal, `0x` hex or `0` octal; -1 when it is not one."""
    var bs = p.as_bytes()
    var n = len(bs)
    if n == 0 or n > 12:
        return -1
    var base = 10
    var start = 0
    if n >= 2 and Int(bs[0]) == 0x30 and (Int(bs[1]) == 0x78 or Int(bs[1]) == 0x58):
        base = 16
        start = 2
        if n == 2:
            return -1
    elif n >= 2 and Int(bs[0]) == 0x30:
        base = 8
        start = 1
    var v = 0
    for i in range(start, n):
        var d = _digit_value(Int(bs[i]))
        if d < 0 or d >= base:
            return -1
        v = v * base + d
    return v


def parse_ipv4_literal(host: String) -> Int:
    """The 32-bit value of an IPv4 literal in any spelling `inet_aton` (and so `getaddrinfo`)
    reads: `127.0.0.1`, `127.1`, `2130706433`, `0x7f000001`, `0177.0.0.1`. -1 when `host` is
    not one."""
    var parts = host.split(".")
    var n = len(parts)
    if n < 1 or n > 4:
        return -1
    var v = 0
    for i in range(n):
        var x = _aton_part(String(parts[i]))
        if x < 0:
            return -1
        if i < n - 1:
            if x > 255:
                return -1
            v = (v << 8) | x
        else:
            var room = 8 * (4 - i)          # the last number fills the remaining bytes
            if x >= (1 << room):
                return -1
            v = (v << room) | x
    return v


def _dotted_quad(p: String) -> Int:
    var parts = p.split(".")
    if len(parts) != 4:
        return -1
    var v = 0
    for i in range(4):
        var bs = String(parts[i]).as_bytes()
        if len(bs) == 0 or len(bs) > 3:
            return -1
        var x = 0
        for k in range(len(bs)):
            var c = Int(bs[k])
            if c < 0x30 or c > 0x39:
                return -1
            x = x * 10 + (c - 0x30)
        if x > 255:
            return -1
        v = (v << 8) | x
    return v


def _groups(part: String, v4_tail: Bool, mut out: List[Int]) -> Bool:
    """Appends the 16-bit groups of one side of an IPv6 literal; False when it is malformed."""
    if part.byte_length() == 0:
        return True
    var pieces = part.split(":")
    for i in range(len(pieces)):
        var p = String(pieces[i])
        if p.find(".") >= 0:
            if not v4_tail or i != len(pieces) - 1:
                return False
            var q = _dotted_quad(p)
            if q < 0:
                return False
            out.append(q >> 16)
            out.append(q & 0xFFFF)
            continue
        var bs = p.as_bytes()
        if len(bs) == 0 or len(bs) > 4:
            return False
        var x = 0
        for k in range(len(bs)):
            var d = _digit_value(Int(bs[k]))
            if d < 0:
                return False
            x = x * 16 + d
        out.append(x)
    return True


def parse_ipv6_literal(host: String) -> List[UInt8]:
    """The 16 bytes of an IPv6 literal (no brackets; a `%zone` is ignored), or an empty list
    when `host` is not one."""
    var none = List[UInt8]()
    var text = host.copy()
    var pct = text.find("%")
    if pct >= 0:
        var cut = String(text[byte=0:pct])
        text = cut^
    var g = List[Int]()
    var gap = text.find("::")
    if gap < 0:
        if not _groups(text, True, g) or len(g) != 8:
            return none^
    else:
        var left = String(text[byte=0:gap])
        var right = String(text[byte=gap + 2:])
        if right.find("::") >= 0:
            return none^
        var tail = List[Int]()
        if not _groups(left, False, g) or not _groups(right, True, tail) or len(g) + len(tail) > 7:
            return none^
        while len(g) + len(tail) < 8:
            g.append(0)
        for x in tail:
            g.append(x)
    var out = List[UInt8](capacity=16)
    for x in g:
        out.append(UInt8(x >> 8))
        out.append(UInt8(x & 0xFF))
    return out^


def literal_is_global(host: String) -> Int:
    """1 when `host` is an IP literal with a global address, 0 when it is an IP literal with
    any other address, -1 when it is not an IP literal (a name: judged after resolution)."""
    if host.find(":") >= 0:
        var b = parse_ipv6_literal(host)
        if len(b) != 16:
            return -1
        return 1 if ipv6_is_global(b) else 0
    var v = parse_ipv4_literal(host)
    if v < 0:
        return -1
    return 1 if ipv4_is_global(v) else 0


# ---------------------------------------------------------------------------
# The URL rules
# ---------------------------------------------------------------------------
@fieldwise_init
struct UrlGuard(Copyable):
    var enabled: Bool
    var allowed_hosts: List[String]   # empty: any public host
    var exempt_prefix: String         # tests only: URLs starting with it (a loopback fixture server) are not checked

    def check(self, url: Url) raises -> Bool:
        """Raises a refusal unless `url` may be fetched. Returns whether the addresses its
        host resolves to must still be checked (False only when the guard is off or the URL
        is exempt)."""
        if not self.enabled:
            return False
        if self.exempt_prefix.byte_length() > 0 and url.to_string().startswith(self.exempt_prefix):
            return False
        if url.scheme != "https":
            raise refusal("only https URLs are fetched")
        if url.port != 443:
            raise refusal("only the default https port is fetched")
        if len(self.allowed_hosts) > 0:
            var listed = False
            for h in self.allowed_hosts:
                if h == url.host:
                    listed = True
                    break
            if not listed:
                raise refusal(url.host + " is not on the allow-list for this tool")
        if literal_is_global(url.host) == 0:
            raise refusal(url.host + " is not a public address")
        return True


def no_guard() -> UrlGuard:
    return UrlGuard(False, List[String](), String())
