"""f32 -> f16 encoding with overflow checks for the model container.

STUB (W-1): the signatures of MOJO_PORT.md section 5 (and `f32_to_f16_bits` of 4.7). Component C1 replaces
this file.
"""

from std.os import abort


def f32_to_f16_bits(x: Float32) -> UInt16:
    """Round to nearest even, via Float16."""
    abort("qumo.rl.f16.f32_to_f16_bits: STUB (component C1)")


def encode_f16(src: List[Float32], tensor_name: String) raises -> List[UInt8]:
    """Little-endian f16 bytes; raises QM-E645 `tensor 'X' overflows f16` when any result is non-finite."""
    raise Error("QM-E640: qumo.rl.f16.encode_f16 is a STUB (component C1)")
