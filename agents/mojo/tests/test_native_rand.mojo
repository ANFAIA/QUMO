"""Tests for qumo_agent.native.rand (DESIGN.md 2.1.7, 8.2)."""

from std.testing import assert_equal, assert_true, assert_false

from qumo_agent.native.rand import *


def _is_hex(s: String) -> Bool:
    for b in s.as_bytes():
        var c = Int(b)
        if not ((c >= 0x30 and c <= 0x39) or (c >= 0x61 and c <= 0x66)):
            return False
    return True


def main() raises:
    assert_equal(len(random_bytes(0)), 0)
    var a = random_bytes(64)
    var b = random_bytes(64)
    assert_equal(len(a), 64)
    var same = True
    for i in range(64):
        if a[i] != b[i]:
            same = False
    assert_false(same)
    assert_equal(len(random_bytes(100000)), 100000)

    for _ in range(50):
        var h = uuid4_hex()
        assert_equal(h.byte_length(), 32)
        assert_true(_is_hex(h))
        assert_equal(String(h[byte=12:13]), "4")                     # version
        var v = String(h[byte=16:17])
        assert_true(v == "8" or v == "9" or v == "a" or v == "b")     # RFC 4122 variant
        var s = uuid4_str()
        assert_equal(s.byte_length(), 36)
        var parts = s.split("-")
        assert_equal(len(parts), 5)
        assert_equal(parts[0].byte_length(), 8)
        assert_equal(parts[1].byte_length(), 4)
        assert_equal(parts[2].byte_length(), 4)
        assert_equal(parts[3].byte_length(), 4)
        assert_equal(parts[4].byte_length(), 12)
        assert_true(String(parts[2]).startswith("4"))
    assert_true(uuid4_hex() != uuid4_hex())

    var lo = 1.0
    var hi = 0.0
    for _ in range(2000):
        var u = random_unit()
        assert_true(u >= 0.0 and u < 1.0)
        lo = min(lo, u)
        hi = max(hi, u)
    assert_true(lo < 0.05 and hi > 0.95)
    print("test_native_rand OK")
