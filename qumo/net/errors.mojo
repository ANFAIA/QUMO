"""Error codes of the native network stack (qumo/net), in the hardware plugin range.

QM-E959: network failure (DNS, connect, TLS handshake, timeout, HTTP framing).
QM-E951: TLS setup failure (OpenSSL 3 not found or unusable, no trust store).
"""

comptime E_NETWORK = "QM-E959"
comptime E_TLS_SETUP = "QM-E951"
