"""f32 -> f16 encoding with overflow checks for the model container (MOJO_PORT.md 4.7, 9.1).

`Float16(Float32(x))` rounds to nearest even in Mojo 1.2 and equals the usual array library's f16 cast bit for bit (probed on
65519.99, 65520, 2^-25, 2^-24, 0.1 and a tie; the exhaustive golden in `tests/test_rl_digest.mojo` gates it).
Decoding f16 -> f32 is exact and lives in `qumo.compiler.policy.f16_to_f32`.

`encode_f16` writes little-endian bytes, as the container does, and raises QM-E645 when any converted value is not
finite (an f32 above 65519.99... rounds to infinity, and NaN / inf inputs stay non-finite), like the old exporter's
`tensor 'X' overflows f16` check.
"""

from std.collections import List
from std.memory import bitcast


def f32_to_f16_bits(x: Float32) -> UInt16:
    """Round to nearest even (ties to even), via Float16. Overflow gives the infinity pattern 0x7C00 / 0xFC00."""
    return bitcast[DType.uint16](Float16(x))


def f16_bits_finite(h: UInt16) -> Bool:
    """True when the binary16 pattern is neither infinity nor NaN (exponent field below 31)."""
    return (h & UInt16(0x7C00)) != UInt16(0x7C00)


def max_abs_f32(src: Span[Float32, _]) -> Float32:
    """Largest |x| over finite elements (0 for an empty span; NaN and inf are ignored). Used for the
    `max|w| > 60000` warning of the trainer (R10)."""
    var m = Float32(0.0)
    for i in range(len(src)):
        var v = src[i]
        var a = v if v >= 0 else -v
        if a > m and a < Float32.MAX:
            m = a
    return m


def f16_overflows(src: Span[Float32, _]) -> Bool:
    """True when at least one element does not survive the conversion to a finite f16."""
    for i in range(len(src)):
        if not f16_bits_finite(f32_to_f16_bits(src[i])):
            return True
    return False


def encode_f16_span(src: Span[Float32, _], tensor_name: String) raises -> List[UInt8]:
    """Little-endian f16 bytes of `src`; raises QM-E645 when any result is non-finite."""
    var out = List[UInt8](length=2 * len(src), fill=UInt8(0))
    var p = out.unsafe_ptr()
    var bad = -1
    for i in range(len(src)):
        var h = f32_to_f16_bits(src[i])
        if bad < 0 and not f16_bits_finite(h):
            bad = i
        p[unsafe_offset=2 * i] = UInt8(h & 0xFF)
        p[unsafe_offset=2 * i + 1] = UInt8(h >> 8)
    if bad >= 0:
        raise Error(
            "QM-E645: tensor '"
            + tensor_name
            + "' overflows f16 (element "
            + String(bad)
            + " = "
            + String(src[bad])
            + "; the largest finite f16 is 65504). Hint: export with dtype f32, or reduce the weight scale (check max|w| in the training log)."
        )
    return out^


def encode_f16(src: List[Float32], tensor_name: String) raises -> List[UInt8]:
    """Little-endian f16 bytes; raises QM-E645 `tensor 'X' overflows f16` when any result is non-finite."""
    return encode_f16_span(Span(src), tensor_name)
