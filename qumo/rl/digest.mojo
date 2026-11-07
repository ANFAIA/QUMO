"""SHA-256 (FIPS 180-4) and CRC-32 (IEEE 802.3, the zlib / gzip checksum) for model verification and eval-set naming.

* `Sha256`: streaming hash over 64-byte blocks. `update` may be called with any split of the input; `finalize`
  returns the 64 lowercase hex characters. `sha256_hex(data)` is the one-shot form and `sha256_file_hex(path)`
  streams a file in 1 MiB chunks (the `models_verify` / container path), so a model file is never held twice.
* `crc32(data)`: reflected polynomial 0xEDB88320, initial value and final xor 0xFFFFFFFF, table driven. Equals
  `zlib.crc32(data)`; used for the eval-set placement `(crc32(file name) + 3 k) % n` (MOJO_PORT.md 4.7).

Portable scalar code, no vendor intrinsics. `std.hashlib` has no SHA-256 in Mojo 1.2, hence this file.
"""

from std.collections import List
from std.bit import rotate_bits_right
from std.memory import memcpy


comptime _HEX = "0123456789abcdef"
comptime FILE_CHUNK = 1048576


def _sha_k() -> List[UInt32]:
    var k = List[UInt32](capacity=64)
    k.append(0x428A2F98)
    k.append(0x71374491)
    k.append(0xB5C0FBCF)
    k.append(0xE9B5DBA5)
    k.append(0x3956C25B)
    k.append(0x59F111F1)
    k.append(0x923F82A4)
    k.append(0xAB1C5ED5)
    k.append(0xD807AA98)
    k.append(0x12835B01)
    k.append(0x243185BE)
    k.append(0x550C7DC3)
    k.append(0x72BE5D74)
    k.append(0x80DEB1FE)
    k.append(0x9BDC06A7)
    k.append(0xC19BF174)
    k.append(0xE49B69C1)
    k.append(0xEFBE4786)
    k.append(0x0FC19DC6)
    k.append(0x240CA1CC)
    k.append(0x2DE92C6F)
    k.append(0x4A7484AA)
    k.append(0x5CB0A9DC)
    k.append(0x76F988DA)
    k.append(0x983E5152)
    k.append(0xA831C66D)
    k.append(0xB00327C8)
    k.append(0xBF597FC7)
    k.append(0xC6E00BF3)
    k.append(0xD5A79147)
    k.append(0x06CA6351)
    k.append(0x14292967)
    k.append(0x27B70A85)
    k.append(0x2E1B2138)
    k.append(0x4D2C6DFC)
    k.append(0x53380D13)
    k.append(0x650A7354)
    k.append(0x766A0ABB)
    k.append(0x81C2C92E)
    k.append(0x92722C85)
    k.append(0xA2BFE8A1)
    k.append(0xA81A664B)
    k.append(0xC24B8B70)
    k.append(0xC76C51A3)
    k.append(0xD192E819)
    k.append(0xD6990624)
    k.append(0xF40E3585)
    k.append(0x106AA070)
    k.append(0x19A4C116)
    k.append(0x1E376C08)
    k.append(0x2748774C)
    k.append(0x34B0BCB5)
    k.append(0x391C0CB3)
    k.append(0x4ED8AA4A)
    k.append(0x5B9CCA4F)
    k.append(0x682E6FF3)
    k.append(0x748F82EE)
    k.append(0x78A5636F)
    k.append(0x84C87814)
    k.append(0x8CC70208)
    k.append(0x90BEFFFA)
    k.append(0xA4506CEB)
    k.append(0xBEF9A3F7)
    k.append(0xC67178F2)
    return k^


def _compress(
    st: MutPointer[UInt32, MutAnyOrigin],
    k: MutPointer[UInt32, MutAnyOrigin],
    w: MutPointer[UInt32, MutAnyOrigin],
    p: Pointer[UInt8, _],
):
    """One 64-byte block at `p` into the chaining value `st` (`w` is 64 words of scratch)."""
    for t in range(16):
        var o = 4 * t
        w[unsafe_offset=t] = (
            (UInt32(p[unsafe_offset=o]) << 24)
            | (UInt32(p[unsafe_offset=o + 1]) << 16)
            | (UInt32(p[unsafe_offset=o + 2]) << 8)
            | UInt32(p[unsafe_offset=o + 3])
        )
    for t in range(16, 64):
        var x = w[unsafe_offset=t - 15]
        var y = w[unsafe_offset=t - 2]
        var s0 = rotate_bits_right[7](x) ^ rotate_bits_right[18](x) ^ (x >> 3)
        var s1 = rotate_bits_right[17](y) ^ rotate_bits_right[19](y) ^ (y >> 10)
        w[unsafe_offset=t] = w[unsafe_offset=t - 16] + s0 + w[unsafe_offset=t - 7] + s1
    var a = st[unsafe_offset=0]
    var b = st[unsafe_offset=1]
    var c = st[unsafe_offset=2]
    var d = st[unsafe_offset=3]
    var e = st[unsafe_offset=4]
    var f = st[unsafe_offset=5]
    var g = st[unsafe_offset=6]
    var h = st[unsafe_offset=7]
    for t in range(64):
        var big1 = rotate_bits_right[6](e) ^ rotate_bits_right[11](e) ^ rotate_bits_right[25](e)
        var ch = (e & f) ^ (~e & g)
        var t1 = h + big1 + ch + k[unsafe_offset=t] + w[unsafe_offset=t]
        var big0 = rotate_bits_right[2](a) ^ rotate_bits_right[13](a) ^ rotate_bits_right[22](a)
        var maj = (a & b) ^ (a & c) ^ (b & c)
        var t2 = big0 + maj
        h = g
        g = f
        f = e
        e = d + t1
        d = c
        c = b
        b = a
        a = t1 + t2
    st[unsafe_offset=0] += a
    st[unsafe_offset=1] += b
    st[unsafe_offset=2] += c
    st[unsafe_offset=3] += d
    st[unsafe_offset=4] += e
    st[unsafe_offset=5] += f
    st[unsafe_offset=6] += g
    st[unsafe_offset=7] += h


struct Sha256(Movable):
    """Streaming SHA-256. `update` accepts any byte span; `finalize` consumes the state (do not reuse afterwards)."""

    var total: Int  # bytes absorbed so far
    var st: List[UInt32]  # chaining value h0..h7
    var k: List[UInt32]  # round constants
    var w: List[UInt32]  # message schedule scratch (64 words)
    var buf: List[UInt8]  # pending partial block (64 bytes)
    var fill: Int  # bytes pending in `buf`

    def __init__(out self):
        self.total = 0
        self.st = List[UInt32](capacity=8)
        self.st.append(0x6A09E667)
        self.st.append(0xBB67AE85)
        self.st.append(0x3C6EF372)
        self.st.append(0xA54FF53A)
        self.st.append(0x510E527F)
        self.st.append(0x9B05688C)
        self.st.append(0x1F83D9AB)
        self.st.append(0x5BE0CD19)
        self.k = _sha_k()
        self.w = List[UInt32](length=64, fill=UInt32(0))
        self.buf = List[UInt8](length=64, fill=UInt8(0))
        self.fill = 0

    def _block(mut self, p: Pointer[UInt8, _]):
        """Absorb one 64-byte block at `p`."""
        _compress(self.st.unsafe_ptr().as_unsafe_any_origin(), self.k.unsafe_ptr().as_unsafe_any_origin(), self.w.unsafe_ptr().as_unsafe_any_origin(), p)
        _ = len(self.st)
        _ = len(self.k)
        _ = len(self.w)

    def update(mut self, data: Span[UInt8, _]):
        var n = len(data)
        if n == 0:
            return
        self.total += n
        var src = data.unsafe_ptr()
        var bufp = self.buf.unsafe_ptr().as_unsafe_any_origin()
        var pos = 0
        if self.fill > 0:
            var need = 64 - self.fill
            var take = need if n >= need else n
            for i in range(take):
                bufp[unsafe_offset=self.fill + i] = src[unsafe_offset=i]
            self.fill += take
            pos = take
            if self.fill < 64:
                return
            self._block(bufp)
            self.fill = 0
        while pos + 64 <= n:
            self._block(src.unsafe_offset(pos))
            pos += 64
        var rest = n - pos
        for i in range(rest):
            bufp[unsafe_offset=i] = src[unsafe_offset=pos + i]
        if rest > 0:
            self.fill = rest
        _ = len(self.buf)

    def finalize_bytes(mut self) -> List[UInt8]:
        """The 32 digest bytes. Consumes the state."""
        var bit_len = UInt64(self.total) * 8
        var pad = List[UInt8]()
        pad.append(UInt8(0x80))
        # pad so that (total + len(pad)) % 64 == 56, then the 8-byte big-endian bit length
        var zeros = (55 - self.total % 64 + 64) % 64
        for _ in range(zeros):
            pad.append(UInt8(0))
        for i in range(8):
            pad.append(UInt8((bit_len >> UInt64(8 * (7 - i))) & 0xFF))
        self.update(Span(pad))
        var out = List[UInt8](capacity=32)
        for i in range(8):
            var v = self.st[i]
            out.append(UInt8((v >> 24) & 0xFF))
            out.append(UInt8((v >> 16) & 0xFF))
            out.append(UInt8((v >> 8) & 0xFF))
            out.append(UInt8(v & 0xFF))
        return out^

    def finalize(mut self) -> String:
        """64 lowercase hex characters. Consumes the state."""
        var d = self.finalize_bytes()
        return hex_string(d)


def hex_string(data: List[UInt8]) -> String:
    """Lowercase hex of a byte list."""
    var digits = _HEX.as_bytes()
    var bytes_out = List[UInt8](capacity=2 * len(data))
    for i in range(len(data)):
        var v = Int(data[i])
        bytes_out.append(digits[v >> 4])
        bytes_out.append(digits[v & 15])
    return String(unsafe_from_utf8=Span(bytes_out))


def sha256_hex(data: List[UInt8]) -> String:
    """SHA-256 of `data` as 64 lowercase hex characters."""
    var h = Sha256()
    h.update(Span(data))
    return h.finalize()


def sha256_span_hex(data: Span[UInt8, _]) -> String:
    """Same as `sha256_hex` for any byte span (no copy)."""
    var h = Sha256()
    h.update(data)
    return h.finalize()


def sha256_file_hex(path: String) raises -> String:
    """SHA-256 of a whole file, streamed in 1 MiB chunks."""
    var h = Sha256()
    with open(path, "r") as f:
        while True:
            var chunk = f.read_bytes(FILE_CHUNK)
            if len(chunk) == 0:
                break
            h.update(Span(chunk))
    return h.finalize()


# ----------------------------------------------------------------------------------------------------- CRC-32


def _crc_table() -> List[UInt32]:
    var t = List[UInt32](capacity=256)
    for i in range(256):
        var c = UInt32(i)
        for _ in range(8):
            if (c & 1) != 0:
                c = (c >> 1) ^ UInt32(0xEDB88320)
            else:
                c = c >> 1
        t.append(c)
    return t^


def crc32_update(crc: UInt32, data: Span[UInt8, _]) -> UInt32:
    """Running CRC-32: pass 0 for the first chunk, then each result for the next (same convention as
    `zlib.crc32(data, value)`)."""
    var table = _crc_table()
    var tp = table.unsafe_ptr()
    var p = data.unsafe_ptr()
    var c = crc ^ UInt32(0xFFFFFFFF)
    for i in range(len(data)):
        c = tp[unsafe_offset=Int((c ^ UInt32(p[unsafe_offset=i])) & 0xFF)] ^ (c >> 8)
    _ = len(table)
    return c ^ UInt32(0xFFFFFFFF)


def crc32(data: List[UInt8]) -> UInt32:
    """IEEE, reflected 0xEDB88320, table-driven; equals `zlib.crc32(data)`."""
    return crc32_update(UInt32(0), Span(data))


def crc32_string(s: String) -> UInt32:
    """CRC-32 of the UTF-8 bytes of `s` (`zlib.crc32(s.encode())`)."""
    return crc32_update(UInt32(0), s.as_bytes())
