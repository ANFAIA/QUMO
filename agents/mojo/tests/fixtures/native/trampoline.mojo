"""Test-only program for test_native_sandbox (built on demand into
build/test-fixtures/native_trampoline).

It stands in for `qumo-agent` where the tests need a real executable:
- `__confine CONF_JSON PATH ARG...` and `__exec-in DIR PATH ARG...` dispatch to the
  trampolines exactly as `main.mojo` does;
- `hello` prints a line (a "reference Mojo binary" run under CONF_TOOL);
- `connect PORT` tries a TCP connection to 127.0.0.1:PORT and prints the outcome;
- `read PATH` tries to read a file and prints the outcome;
- `spin` burns CPU forever (RLIMIT_CPU);
- `fork-mirror SH_COMMAND` forks a child that runs `/bin/sh -c SH_COMMAND` and exits like the
  Linux `__confine` path does (`wait_and_mirror`).
"""

from std.sys import argv, exit

from qumo_agent.native.sandbox import confine_main, wait_and_mirror
from qumo_agent.native.spawn import exec_in_main
from qumo_agent.native.libc import (
    AF_INET,
    SOCK_STREAM,
    buf_addr,
    store_u16,
    c_socket,
    c_connect,
    c_close,
    c_execve,
    c_fork,
    errno,
    CStr,
    CStrArray,
    is_macos,
)


def _args() -> List[String]:
    var out = List[String]()
    for a in argv():
        out.append(String(a))
    return out^


def _connect(port: Int) -> String:
    var fd = c_socket(AF_INET, SOCK_STREAM, 0)
    if fd < 0:
        return "socket failed: errno " + String(errno())
    var sa = List[UInt8](length=16, fill=0)
    if is_macos():
        sa[0] = 16
        sa[1] = UInt8(AF_INET)
    else:
        store_u16(buf_addr(sa), AF_INET)
    sa[2] = UInt8((port >> 8) & 0xFF)
    sa[3] = UInt8(port & 0xFF)
    sa[4] = 127
    sa[7] = 1
    var rc = c_connect(fd, buf_addr(sa), 16)
    var e = errno()
    _ = len(sa)
    _ = c_close(fd)
    if rc == 0:
        return "connected"
    return "connect failed: errno " + String(e)


def main() raises:
    var a = _args()
    if len(a) < 2:
        print("usage: trampoline MODE ...")
        exit(2)
    var mode = a[1]
    if mode == "__confine":
        exit(confine_main(a))
    if mode == "__exec-in":
        exit(exec_in_main(a))
    if mode == "hello":
        print("hello from fixture")
        return
    if mode == "connect":
        print(_connect(Int(a[2])))
        return
    if mode == "read":
        try:
            with open(a[2], "r") as f:
                var n = len(f.read_bytes())
                print("read ok " + String(n))
        except:
            print("read failed")
        return
    if mode == "fork-mirror":
        var pid = c_fork()
        if pid == 0:
            var argv: List[String] = ["sh", "-c", a[2]]
            var p = CStr("/bin/sh")
            var av = CStrArray(argv)
            var ev = CStrArray(List[String]())
            _ = c_execve(p.addr(), av.addr(), ev.addr())
            p.keep()
            av.keep()
            ev.keep()
            exit(127)
        exit(wait_and_mirror(pid))
    if mode == "spin":
        var x: UInt64 = 1
        while True:
            x = x * 6364136223846793005 + 1442695040888963407
            if x == 0:
                break
        print(x)
        return
    print("unknown mode " + mode)
    exit(2)
