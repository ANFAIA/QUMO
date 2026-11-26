"""Helpers shared by the vendor clients of this package."""


def loopback_host(host: String) -> Bool:
    """True for the hosts a client may reach over plain http (local fake servers in tests):
    127.0.0.1, localhost and ::1. Every other host needs https."""
    return host == "127.0.0.1" or host == "localhost" or host == "::1"
