"""Tests for qumo_agent.util.sha256 (DESIGN.md 2.2.3, 8.2): NIST FIPS 180-4 vectors."""

from std.testing import assert_equal, assert_true

from qumo_agent.util.sha256 import *
from qumo_agent.native.fs import make_temp_dir, write_text, remove_tree


def main() raises:
    assert_equal(sha256_hex_text(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    assert_equal(sha256_hex_text("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert_equal(
        sha256_hex_text("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    )
    var million = List[UInt8](length=1000000, fill=0x61)
    assert_equal(sha256_hex(Span(million)), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0")
    # lengths around the padding boundaries, checked against Python's hashlib
    var lens: List[Int] = [55, 56, 63, 64, 65, 119, 120]
    var want: List[String] = [
        "d5e285683cd4efc02d021a5c62014694958901005d6f71e89e0989fac77e4072",
        "04c26261370ee7541549d16dee320c723e3fd14671e66a099afe0a377c16888e",
        "75220b47218278e656f2013bb8f0c455a25eaf01e86c64924e9d48d89776d6f2",
        "7ce100971f64e7001e8fe5a51973ecdfe1ced42befe7ee8d5fd6219506b5393c",
        "9537c5fdf120482f7d58d25e9ed583f52c02b4e304ea814db1633ad565aed7e9",
        "000b48d4edf0fa7bee3c6236ecd2785baa5db4eeb8bb54341b029e0d9fa5fb0c",
        "13f05a0b594787f5ecd315edc96141bd3243203d1b7d4f0836f37308b276ba98",
    ]
    for i in range(len(lens)):
        var b = List[UInt8](length=lens[i], fill=0x78)
        assert_equal(sha256_hex(Span(b)), want[i], "length " + String(lens[i]))
    assert_equal(sha256_hex_text("é"), sha256_hex("é".as_bytes()))
    var t = make_temp_dir("qumo-sha-")
    write_text(t + "/f", "abc")
    assert_equal(sha256_file_hex(t + "/f"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    try:
        _ = sha256_file_hex(t + "/missing")
        assert_true(False, "a missing file must raise")
    except e:
        assert_true(String(e).startswith("QA-E160: [Errno 2]"))
    remove_tree(t)
    print("test_util_sha256 OK")
