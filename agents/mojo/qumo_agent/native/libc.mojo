"""libc bindings of qumo_agent: a re-export of `qumo.net.libc`, the only module with
`external_call` in the binary (one wrapper and one Mojo signature per C symbol for the whole binary;
conventions, lifetimes and platform constants are documented there and in DESIGN.md 4.2).

`fa_addchdir` is defined here because its error carries an agent code (QA-E162).
"""

from std.ffi import c_int, OwnedDLHandle, RTLD
from std.memory import OpaquePointer

from qumo.net.libc import *

comptime _VP = OpaquePointer[MutAnyOrigin]


def _p(addr: Int) -> _VP:
    return _VP(unsafe_from_address=addr)


def fa_addchdir(fa: Int, path: Int) raises -> Int:
    """posix_spawn_file_actions_addchdir_np through dlsym (macOS 10.15+, glibc 2.29+).

    Raises QA-E162 when this libc does not have it."""
    var h = OwnedDLHandle(RTLD.NOW)
    if not h.check_symbol("posix_spawn_file_actions_addchdir_np"):
        raise Error("QA-E162: this libc has no posix_spawn chdir action. Hint: use glibc 2.29 or newer")
    return Int(h.get_function[c_int]("posix_spawn_file_actions_addchdir_np")(_p(fa), _p(path)))
