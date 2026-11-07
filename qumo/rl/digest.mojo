"""SHA-256 (FIPS 180-4) and CRC-32 (IEEE) for model verification and eval-set naming.

STUB (W-1): the signatures of MOJO_PORT.md section 5 (plus the streaming `Sha256` of 4.7). Component C1
replaces this file; every behavioural body aborts with a "STUB" message.
"""

from std.os import abort


struct Sha256(Movable):
    """Streaming SHA-256 over 64-byte blocks (PROVISIONAL shape, MOJO_PORT.md 4.7)."""

    var total: Int

    def __init__(out self):
        self.total = 0

    def update(mut self, data: Span[UInt8, _]):
        abort("qumo.rl.digest.Sha256.update: STUB (component C1)")

    def finalize(mut self) -> String:
        """64 lowercase hex characters."""
        abort("qumo.rl.digest.Sha256.finalize: STUB (component C1)")


def sha256_hex(data: List[UInt8]) -> String:
    abort("qumo.rl.digest.sha256_hex: STUB (component C1)")


def crc32(data: List[UInt8]) -> UInt32:
    """IEEE, reflected 0xEDB88320, table-driven."""
    abort("qumo.rl.digest.crc32: STUB (component C1)")
