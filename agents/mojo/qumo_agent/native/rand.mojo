"""Randomness from the operating system (DESIGN.md 2.1.7): /dev/urandom, uuid4.

These are for identifiers (uuid4 session ids, temp names) and retry jitter, not for
reproducible simulations (QUMO's seeded `qumo.sim.rng` is for those).
"""

comptime _HEX = "0123456789abcdef"


def random_bytes(n: Int) raises -> List[UInt8]:
    """`n` bytes read from /dev/urandom."""
    if n <= 0:
        return List[UInt8]()
    var out = List[UInt8](capacity=n)
    with open("/dev/urandom", "r") as f:
        while len(out) < n:
            var chunk = f.read_bytes(n - len(out))
            if len(chunk) == 0:
                raise Error("QA-E160: /dev/urandom returned no data")
            for i in range(len(chunk)):
                out.append(chunk[i])
    return out^


def _hex(bs: List[UInt8]) -> String:
    var hexd = String(_HEX).as_bytes()
    var out = List[UInt8](capacity=len(bs) * 2)
    for i in range(len(bs)):
        var b = Int(bs[i])
        out.append(hexd[b >> 4])
        out.append(hexd[b & 0xF])
    return String(unsafe_from_utf8=Span(out))


def _uuid4_bytes() raises -> List[UInt8]:
    var b = random_bytes(16)
    b[6] = (b[6] & 0x0F) | 0x40   # version 4
    b[8] = (b[8] & 0x3F) | 0x80   # RFC 4122 variant
    return b^


def uuid4_hex() raises -> String:
    """`uuid.uuid4().hex`: 32 lowercase hex digits."""
    return _hex(_uuid4_bytes())


def uuid4_str() raises -> String:
    """`str(uuid.uuid4())`: 8-4-4-4-12."""
    var h = uuid4_hex()
    return (
        String(h[byte=0:8]) + "-" + String(h[byte=8:12]) + "-" + String(h[byte=12:16]) + "-"
        + String(h[byte=16:20]) + "-" + String(h[byte=20:32])
    )


def random_unit() raises -> Float64:
    """Uniform in [0, 1) from 53 random bits."""
    var b = random_bytes(8)
    var v: UInt64 = 0
    for i in range(8):
        v = (v << 8) | UInt64(b[i])
    return Float64(v >> 11) * (1.0 / 9007199254740992.0)
