"""Tests for qumo.rl.digest (SHA-256, CRC-32) and qumo.rl.f16 (f32 -> f16 encoder).

Covers MOJO_PORT.md 13.2 `test_rl_digest`: FIPS 180-4 vectors, streaming splits, every shipped model file hashed in
1 MiB chunks against its INDEX sha256, CRC-32 check values (shell `gzip` trailers as the zlib oracle), all 65536 f16
patterns round trip, the 326-value `f16_golden.json`, overflow errors, and the G7 throughput gate.
"""

from std.collections import List
from std.memory import bitcast
from std.time import perf_counter_ns
from std.os import makedirs
from qumo.compiler.policy import f16_to_f32, fnv1a64_bytes, hex16
from qumo.rl.digest import (
    Sha256,
    sha256_hex,
    sha256_span_hex,
    sha256_file_hex,
    crc32,
    crc32_update,
    crc32_string,
    hex_string,
)
from qumo.rl.f16 import (
    f32_to_f16_bits,
    f16_bits_finite,
    encode_f16,
    encode_f16_span,
    f16_overflows,
    max_abs_f32,
)
from qumo.rl.testutil import load_fixture, ints, read_bytes
from qumo.ai.json import parse_json


def check(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAIL: " + what)


def bytes_of(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var b = s.as_bytes()
    for i in range(len(b)):
        out.append(b[i])
    return out^


def check_hex(got: String, want: String, what: String) raises:
    if got != want:
        raise Error("FAIL: " + what + ": got " + got + " want " + want)


# ------------------------------------------------------------------------------------------------ SHA-256


def test_fips_vectors() raises:
    check_hex(
        sha256_hex(List[UInt8]()),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "sha256 empty",
    )
    check_hex(
        sha256_hex(bytes_of("abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "sha256 abc",
    )
    check_hex(
        sha256_hex(bytes_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        "sha256 448-bit message",
    )
    check_hex(
        sha256_hex(
            bytes_of(
                "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            )
        ),
        "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
        "sha256 896-bit message",
    )
    var million = List[UInt8](length=1000000, fill=UInt8(97))
    check_hex(
        sha256_hex(million),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        "sha256 1,000,000 x a",
    )
    # the same million through the streaming interface in 1000-byte pieces, via a span of the buffer
    var h = Sha256()
    for i in range(1000):
        h.update(Span(million)[i * 1000 : (i + 1) * 1000])
    check_hex(
        h.finalize(),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        "sha256 streamed 1000 x 1000",
    )
    print("  fips vectors ok")


def test_streaming_splits() raises:
    """Every message length 0..200 hashed in one piece equals the hash over pieces of varied sizes, including
    the padding boundaries 55, 56, 63, 64, 65."""
    var data = List[UInt8](capacity=200)
    for i in range(200):
        data.append(UInt8((i * 37 + 11) % 256))
    var sizes = List[Int]()
    sizes.append(1)
    sizes.append(3)
    sizes.append(17)
    sizes.append(63)
    sizes.append(64)
    sizes.append(65)
    sizes.append(100)
    for n in range(0, 201):
        var whole = sha256_span_hex(Span(data)[0:n])
        for si in range(len(sizes)):
            var step = sizes[si]
            var h = Sha256()
            var pos = 0
            while pos < n:
                var end = pos + step
                if end > n:
                    end = n
                h.update(Span(data)[pos:end])
                pos = end
            check_hex(h.finalize(), whole, "split n=" + String(n) + " step=" + String(step))
    # a few known lengths at the padding boundaries (shell `shasum -a 256` of printf 'a' x n)
    var a55 = List[UInt8](length=55, fill=UInt8(97))
    var a56 = List[UInt8](length=56, fill=UInt8(97))
    var a64 = List[UInt8](length=64, fill=UInt8(97))
    check_hex(
        sha256_hex(a55),
        "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
        "sha256 55 x a",
    )
    check_hex(
        sha256_hex(a56),
        "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
        "sha256 56 x a",
    )
    check_hex(
        sha256_hex(a64),
        "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
        "sha256 64 x a",
    )
    print("  streaming splits ok")


def index_rows(index_path: String) raises -> List[String]:
    """'dir|sha256' for every row of an INDEX.json."""
    var text: String
    with open(index_path, "r") as f:
        text = f.read()
    var v = parse_json(text)
    var out = List[String]()
    for i in range(len(v)):
        var row = v.at(i)
        out.append(row.get("dir").as_string() + "|" + row.get("sha256").as_string())
    return out^


def test_shipped_models() raises:
    var total = 0
    var count = 0
    var roots = List[String]()
    roots.append("models/ai_transpiler/synth")
    roots.append("models/ai_transpiler/route")
    for ri in range(len(roots)):
        var rows = index_rows(roots[ri] + "/INDEX.json")
        for k in range(len(rows)):
            var parts = rows[k].split("|")
            var wpath = roots[ri] + "/" + String(parts[0]) + "/weights.bin"
            var want = String(parts[1])
            var got = sha256_file_hex(wpath)
            check_hex(got, want, "file " + wpath)
            # the whole-file path must agree with the chunked path
            var raw = read_bytes(wpath)
            check_hex(sha256_hex(raw), want, "whole " + wpath)
            total += len(raw)
            count += 1
    check(count == 10, "10 shipped models, saw " + String(count))
    print("  shipped models ok:", count, "files,", total, "bytes")


def test_file_chunking() raises:
    """A file larger than one 1 MiB chunk is hashed in several chunks (and equals the in-memory hash)."""
    makedirs("build", exist_ok=True)
    var path = "build/test_rl_digest_big.bin"
    var big = List[UInt8](capacity=2621441)
    for i in range(2621441):
        big.append(UInt8((i * 131 + (i >> 8)) % 256))
    with open(path, "w") as f:
        f.write_bytes(big)
    check_hex(sha256_file_hex(path), sha256_hex(big), "2.5 MiB file in 1 MiB chunks")
    print("  file chunking ok")


# ------------------------------------------------------------------------------------------------ CRC-32


def test_crc32() raises:
    check(crc32(bytes_of("123456789")) == UInt32(0xCBF43926), "crc32 check value")
    check(crc32(List[UInt8]()) == UInt32(0), "crc32 empty")
    # zlib.crc32 of the file names the eval-set placement uses; oracle = gzip trailer from the shell
    check(crc32_string("feynman__barenco_tof_3.qasm") == UInt32(0x07B6122A), "crc32 barenco_tof_3")
    check(crc32_string("feynman__grover_5.qasm") == UInt32(0xC5B5FE41), "crc32 grover_5")
    check(crc32_string("feynman__adder.qasm") == UInt32(0xEAEFBDC0), "crc32 adder")
    # running form: crc(a + b) == update(crc(a), b)
    var whole = bytes_of("The quick brown fox jumps over the lazy dog")
    check(crc32(whole) == UInt32(0x414FA339), "crc32 fox")
    var first = crc32_update(UInt32(0), Span(whole)[0:10])
    var both = crc32_update(first, Span(whole)[10:len(whole)])
    check(both == UInt32(0x414FA339), "crc32 running")
    print("  crc32 ok")


# ------------------------------------------------------------------------------------------------ f16


def f32_of(h: UInt16) -> Float32:
    """Independent decode through the hardware conversion (not policy.f16_to_f32)."""
    return Float32(bitcast[DType.float16](h))


def test_f16_roundtrip_all() raises:
    var bad = 0
    var nan_count = 0
    for hi in range(65536):
        var h = UInt16(hi)
        var x = f16_to_f32(h)
        var y = f32_of(h)
        var is_nan = (hi & 0x7C00) == 0x7C00 and (hi & 0x3FF) != 0
        if is_nan:
            nan_count += 1
            # NaN payload is ignored; the encoded value must still be a NaN pattern
            var e = f32_to_f16_bits(x)
            if (e & UInt16(0x7C00)) != UInt16(0x7C00) or (e & UInt16(0x3FF)) == 0:
                bad += 1
            continue
        # decode: the library decoder and the hardware conversion agree bit for bit
        if bitcast[DType.uint32](x) != bitcast[DType.uint32](y):
            bad += 1
        # encode(decode(h)) == h
        if f32_to_f16_bits(x) != h:
            bad += 1
    check(bad == 0, "f16 round trip: " + String(bad) + " bad patterns")
    check(nan_count == 2046, "NaN pattern count " + String(nan_count))
    print("  f16 round trip over 65536 patterns ok")


def test_f16_golden() raises:
    var g = load_fixture("torch/f16_golden.json")
    var f32b = ints(g.get("f32_bits"))
    var f16b = ints(g.get("f16_bits"))
    var dec = ints(g.get("f16_decoded_f32_bits"))
    var overflow = g.get("overflow_count").as_int()
    check(len(f32b) == 326 and len(f16b) == 326 and len(dec) == 326, "golden sizes")
    var non_finite = 0
    for i in range(len(f32b)):
        var x = bitcast[DType.float32](UInt32(f32b[i]))
        var h = f32_to_f16_bits(x)
        if Int(h) != f16b[i]:
            raise Error(
                "FAIL: f16_golden[" + String(i) + "] x=" + String(x) + " got " + String(Int(h)) + " want " + String(f16b[i])
            )
        var back = bitcast[DType.uint32](f16_to_f32(h))
        if Int(back) != dec[i]:
            raise Error("FAIL: f16_golden decoded bits at " + String(i))
        if not f16_bits_finite(h):
            non_finite += 1
    check(non_finite == overflow, "overflow count " + String(non_finite) + " vs " + String(overflow))
    print("  f16_golden ok (", len(f32b), "values,", non_finite, "overflow )")


def test_f16_edge_values() raises:
    # probe values of MOJO_PORT.md 1.4 (numpy astype('<f2') results)
    check(Int(f32_to_f16_bits(Float32(65519.99))) == 31743, "65519.99")
    check(Int(f32_to_f16_bits(Float32(65520.0))) == 0x7C00, "65520 -> inf")
    check(Int(f32_to_f16_bits(Float32(-70000.0))) == 0xFC00, "-70000 -> -inf")
    check(Int(f32_to_f16_bits(Float32(1.0 / 33554432.0))) == 0, "2^-25 ties to 0")
    check(Int(f32_to_f16_bits(Float32(1.0 / 16777216.0))) == 1, "2^-24 -> 1")
    check(Int(f32_to_f16_bits(Float32(0.1))) == 11878, "0.1")
    check(Int(f32_to_f16_bits(Float32(1.0) + Float32(1.0 / 2048.0))) == 15360, "1 + 2^-11 tie to even")
    check(Int(f32_to_f16_bits(Float32(0.0))) == 0, "+0")
    check(Int(f32_to_f16_bits(Float32(-0.0))) == 0x8000, "-0")
    check(Int(f32_to_f16_bits(Float32(65504.0))) == 0x7BFF, "max finite")
    print("  f16 edge values ok")


def test_f16_encode() raises:
    var src = List[Float32]()
    src.append(1.0)
    src.append(-2.5)
    src.append(0.1)
    src.append(65504.0)
    var enc = encode_f16(src, "w")
    check(len(enc) == 8, "encode length")
    # little endian: 1.0 = 0x3C00, -2.5 = 0xC100, 0.1 = 0x2E66 (11878), 65504 = 0x7BFF
    check(Int(enc[0]) == 0x00 and Int(enc[1]) == 0x3C, "1.0 bytes")
    check(Int(enc[2]) == 0x00 and Int(enc[3]) == 0xC1, "-2.5 bytes")
    check(Int(enc[4]) == 0x66 and Int(enc[5]) == 0x2E, "0.1 bytes")
    check(Int(enc[6]) == 0xFF and Int(enc[7]) == 0x7B, "65504 bytes")
    check(not f16_overflows(Span(src)), "no overflow in range")
    check(max_abs_f32(Span(src)) == 65504.0, "max_abs")
    # empty input encodes to empty bytes
    check(len(encode_f16(List[Float32](), "empty")) == 0, "empty encode")

    var big = List[Float32]()
    big.append(1.0)
    big.append(65520.0)
    check(f16_overflows(Span(big)), "65520 overflows")
    var raised = False
    try:
        _ = encode_f16(big, "embeddings.weight")
    except e:
        raised = True
        var msg = String(e)
        check("QM-E645" in msg, "message has QM-E645: " + msg)
        check("tensor 'embeddings.weight' overflows f16" in msg, "message text: " + msg)
        check("element 1" in msg, "message names the element: " + msg)
    check(raised, "overflow must raise")

    # NaN and inf inputs are non-finite after rounding as well
    var nan_src = List[Float32]()
    nan_src.append(bitcast[DType.float32](UInt32(0x7FC00000)))
    raised = False
    try:
        _ = encode_f16(nan_src, "n")
    except e:
        raised = True
    check(raised, "NaN must raise")
    var inf_src = List[Float32]()
    inf_src.append(bitcast[DType.float32](UInt32(0x7F800000)))
    raised = False
    try:
        _ = encode_f16_span(Span(inf_src), "i")
    except e:
        raised = True
    check(raised, "inf must raise")
    print("  f16 encode and overflow ok")


def test_f16_bulk_matches_scalar() raises:
    """The bulk encoder on 20000 pseudo-random f32 values (several decades of magnitude) equals the scalar encoder."""
    var src = List[Float32](capacity=20000)
    var s = UInt64(88172645463325252)
    for _ in range(20000):
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        var mant = Float32(Int(s >> 40)) / Float32(16777216.0) - 0.5
        var ex = Int((s >> 8) & 0x1F) - 20  # 2^-20 .. 2^11, all finite after f16 rounding
        var scale = Float32(1.0)
        if ex >= 0:
            for _k in range(ex):
                scale *= 2.0
        else:
            for _k in range(-ex):
                scale *= 0.5
        src.append(mant * scale)
    var enc = encode_f16(src, "bulk")
    for i in range(len(src)):
        var h = f32_to_f16_bits(src[i])
        if Int(enc[2 * i]) != Int(h & 0xFF) or Int(enc[2 * i + 1]) != Int(h >> 8):
            raise Error("FAIL: bulk encode differs at " + String(i))
        # and the result agrees with the hardware conversion
        if h != bitcast[DType.uint16](Float16(src[i])):
            raise Error("FAIL: scalar encode differs from Float16 at " + String(i))
    print("  f16 bulk encode ok")


# ------------------------------------------------------------------------------------------------ throughput


def test_throughput() raises:
    """G7: SHA-256 >= 100 MB/s. 32 MiB hashed; best of 3 (the machine may be shared)."""
    var n = 32 * 1048576
    var buf = List[UInt8](capacity=n)
    for i in range(n):
        buf.append(UInt8((i * 7 + 3) % 256))
    var best = 0.0

    for _ in range(3):
        var t0 = perf_counter_ns()
        _ = sha256_hex(buf)
        var t1 = perf_counter_ns()
        var sec = Float64(t1 - t0) * 1e-9
        var mbs = Float64(n) / 1048576.0 / sec
        if mbs > best:
            best = mbs
    print("  sha256 throughput:", Int(best), "MB/s (gate >= 100)")
    check(best >= 100.0, "G7: sha256 throughput " + String(best) + " MB/s < 100")


def main() raises:
    test_fips_vectors()
    test_streaming_splits()
    test_shipped_models()
    test_file_chunking()
    test_crc32()
    test_f16_roundtrip_all()
    test_f16_golden()
    test_f16_edge_values()
    test_f16_encode()
    test_f16_bulk_matches_scalar()
    test_throughput()
    print("test_rl_digest OK")
