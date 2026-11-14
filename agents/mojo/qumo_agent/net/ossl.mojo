"""OpenSSL 3, loaded at run time with `dlopen` (DESIGN.md 4.3 and 4.4).

OpenSSL is never linked: a direct libc-style FFI call to its symbols fails under `mojo run` and
under `mojo build` without `-lcrypto`, so every function is called through
`OwnedDLHandle.get_function`. Pointers are passed and returned as `Int` (a C string
returned by OpenSSL is copied with `c_string_at`; an `Optional[CStringSpan]` return
crashed in the probes). Every bound function is checked with `check_symbol` when
`OpenSsl` is created, so a missing symbol fails early with one message.

Lookup order (`find_openssl`): `$QUMO_LIBSSL` (and `$QUMO_LIBCRYPTO`, default: the
same path with `libssl` replaced by `libcrypto` in the file name); then on macOS the
Homebrew `openssl@3` prefixes; on Linux the bare sonames `libssl.so.3` and
`libcrypto.so.3`. Unversioned names are never tried: on macOS `/usr/lib/libssl.dylib`
aborts the process and `/usr/lib/libssl.48.dylib` is LibreSSL.
"""

from std.ffi import OwnedDLHandle, RTLD, c_int
from std.os import getenv
from std.os.path import exists
from std.sys import CompilationTarget

from qumo_agent.core.errors import E_TLS_SETUP
from qumo_agent.native.libc import c_string_at, buf_addr

comptime _MAC = CompilationTarget.is_macos()

comptime _CRYPTO_SYMBOLS: List[String] = [
    "OpenSSL_version_num", "OpenSSL_version", "ERR_get_error", "ERR_clear_error",
    "ERR_error_string_n", "X509_verify_cert_error_string", "X509_free",
    "X509_VERIFY_PARAM_set1_ip_asc", "X509_get_default_cert_file", "X509_get_default_cert_dir",
    "X509_get_default_cert_file_env", "X509_get_default_cert_dir_env",
]

comptime _SSL_SYMBOLS: List[String] = [
    "TLS_client_method", "SSL_CTX_new", "SSL_CTX_free", "SSL_CTX_ctrl", "SSL_CTX_set_options",
    "SSL_CTX_set_verify", "SSL_CTX_set_default_verify_paths", "SSL_CTX_load_verify_locations",
    "SSL_new", "SSL_free", "SSL_set_fd", "SSL_ctrl", "SSL_set_hostflags", "SSL_set1_host",
    "SSL_get0_param", "SSL_connect", "SSL_read_ex", "SSL_write_ex", "SSL_get_error",
    "SSL_get_verify_result", "SSL_get1_peer_certificate", "SSL_get_version", "SSL_shutdown",
]

# SSL_get_error codes (ssl.h).
comptime SSL_ERROR_NONE = 0
comptime SSL_ERROR_SSL = 1
comptime SSL_ERROR_WANT_READ = 2
comptime SSL_ERROR_WANT_WRITE = 3
comptime SSL_ERROR_SYSCALL = 5
comptime SSL_ERROR_ZERO_RETURN = 6

# Reason codes (low 23 bits of an ERR_get_error value).
comptime SSL_R_CERTIFICATE_VERIFY_FAILED = 134
comptime SSL_R_UNEXPECTED_EOF_WHILE_READING = 294

# Macros expressed as ctrl calls, options and flags.
comptime SSL_CTRL_SET_TLSEXT_HOSTNAME = 55
comptime TLSEXT_NAMETYPE_host_name = 0
comptime SSL_CTRL_SET_MIN_PROTO_VERSION = 123
comptime TLS1_2_VERSION = 0x0303
comptime SSL_OP_NO_COMPRESSION = UInt64(1) << 17
comptime SSL_OP_NO_RENEGOTIATION = UInt64(1) << 30
comptime SSL_VERIFY_PEER = 1
comptime X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS = 0x4
comptime X509_V_OK = 0


def _not_found_hint() -> String:
    return "brew install openssl@3, apt install libssl3, or set QUMO_LIBSSL and QUMO_LIBCRYPTO"


def _crypto_for(libssl: String) -> String:
    """The libcrypto path next to `libssl`: `libssl` replaced by `libcrypto` in the file name."""
    var slash = libssl.rfind("/")
    var dir = String(libssl[byte=0:slash + 1]) if slash >= 0 else String()
    var name = String(libssl[byte=slash + 1:]) if slash >= 0 else libssl.copy()
    return dir + name.replace("libssl", "libcrypto")


def _dlopen_ok(path: String) -> Bool:
    try:
        var h = OwnedDLHandle(path, RTLD.NOW | RTLD.LOCAL)
        _ = h.check_symbol("OpenSSL_version_num")
        return True
    except:
        return False


def find_openssl() raises -> Tuple[String, String]:
    """(libssl, libcrypto) per DESIGN.md 4.4. `$QUMO_LIBSSL`, when set, is the only
    candidate (so setting it to a missing path simulates a machine without OpenSSL)."""
    var tried = List[String]()
    var env_ssl = getenv("QUMO_LIBSSL", "")
    if env_ssl.byte_length() > 0:
        var env_crypto = getenv("QUMO_LIBCRYPTO", "")
        if env_crypto.byte_length() == 0:
            env_crypto = _crypto_for(env_ssl)
        if exists(env_ssl) and exists(env_crypto):
            return (env_ssl, env_crypto)
        if env_ssl.find("/") < 0 and _dlopen_ok(env_crypto) and _dlopen_ok(env_ssl):
            return (env_ssl, env_crypto)
        tried.append(env_ssl + " (QUMO_LIBSSL)")
    else:
        comptime if _MAC:
            var dirs: List[String] = ["/opt/homebrew/opt/openssl@3/lib", "/usr/local/opt/openssl@3/lib"]
            for d in dirs:
                var s = d + "/libssl.3.dylib"
                var c = d + "/libcrypto.3.dylib"
                if exists(s) and exists(c):
                    return (s, c)
                tried.append(s)
        else:
            if _dlopen_ok("libcrypto.so.3") and _dlopen_ok("libssl.so.3"):
                return (String("libssl.so.3"), String("libcrypto.so.3"))
            tried.append("libssl.so.3")
    var listed = String()
    for i in range(len(tried)):
        if i > 0:
            listed += ", "
        listed += tried[i]
    raise Error(E_TLS_SETUP + ": OpenSSL 3 not found (tried: " + listed + "). Hint: " + _not_found_hint())


struct OpenSsl(Movable):
    var crypto: OwnedDLHandle
    var ssl: OwnedDLHandle
    var version_text: String        # OpenSSL_version(0), e.g. "OpenSSL 3.6.4 25 Aug 2026"
    var version_num: UInt
    var libssl_path: String
    var libcrypto_path: String

    def __init__(out self) raises:
        """Finds, loads (libcrypto first) and checks OpenSSL 3; raises QA-E105."""
        var paths = find_openssl()
        self.libssl_path = paths[0]
        self.libcrypto_path = paths[1]
        var flags = RTLD.NOW | RTLD.LOCAL | RTLD.NODELETE
        try:
            self.crypto = OwnedDLHandle(self.libcrypto_path, flags)
        except e:
            raise Error(E_TLS_SETUP + ": cannot load " + self.libcrypto_path + ": " + String(e)
                        + ". Hint: " + _not_found_hint())
        try:
            self.ssl = OwnedDLHandle(self.libssl_path, flags)
        except e:
            raise Error(E_TLS_SETUP + ": cannot load " + self.libssl_path + ": " + String(e)
                        + ". Hint: " + _not_found_hint())
        if not self.crypto.check_symbol("OpenSSL_version_num") or not self.crypto.check_symbol("OpenSSL_version"):
            raise Error(E_TLS_SETUP + ": " + self.libcrypto_path + " is not OpenSSL 3. Hint: " + _not_found_hint())
        self.version_num = self.crypto.get_function[UInt]("OpenSSL_version_num")()
        self.version_text = c_string_at(self.crypto.get_function[Int]("OpenSSL_version")(c_int(0)))
        if self.version_num < 0x30000000:
            raise Error(E_TLS_SETUP + ": " + self.libcrypto_path + " (" + self.version_text
                        + ") is older than OpenSSL 3.0. Hint: " + _not_found_hint())
        for name in materialize[_CRYPTO_SYMBOLS]():
            if not self.crypto.check_symbol(name):
                raise Error(E_TLS_SETUP + ": " + self.libcrypto_path + " (" + self.version_text + ") lacks " + name)
        for name in materialize[_SSL_SYMBOLS]():
            if not self.ssl.check_symbol(name):
                raise Error(E_TLS_SETUP + ": " + self.libssl_path + " (" + self.version_text + ") lacks " + name)

    def errors(self) -> String:
        """Drains the thread's OpenSSL error queue into "error:...; error:..."."""
        var reasons = List[Int]()
        return self.drain_errors(reasons)

    def drain_errors(self, mut reasons: List[Int]) -> String:
        """Like `errors`, also appending each entry's reason code (low 23 bits) to `reasons`."""
        var out = String()
        try:
            var buf = List[UInt8](length=256, fill=0)
            for _ in range(64):
                var e = self.ERR_get_error()
                if e == 0:
                    break
                reasons.append(Int(e & 0x7FFFFF))
                self.ERR_error_string_n(e, buf_addr(buf), 256)
                if out.byte_length() > 0:
                    out += "; "
                out += c_string_at(buf_addr(buf))
            _ = len(buf)
        except:
            pass
        return out^

    # ---- libcrypto ----
    def OpenSSL_version_num(self) raises -> UInt:
        return self.crypto.get_function[UInt]("OpenSSL_version_num")()

    def OpenSSL_version(self, t: Int) raises -> Int:
        return self.crypto.get_function[Int]("OpenSSL_version")(c_int(t))

    def ERR_get_error(self) raises -> UInt:
        return self.crypto.get_function[UInt]("ERR_get_error")()

    def ERR_clear_error(self) raises:
        self.crypto.get_function[NoneType]("ERR_clear_error")()

    def ERR_error_string_n(self, e: UInt, buf: Int, n: Int) raises:
        self.crypto.get_function[NoneType]("ERR_error_string_n")(e, buf, UInt(n))

    def X509_verify_cert_error_string(self, n: Int) raises -> Int:
        return self.crypto.get_function[Int]("X509_verify_cert_error_string")(n)

    def X509_free(self, x: Int) raises:
        self.crypto.get_function[NoneType]("X509_free")(x)

    def X509_VERIFY_PARAM_set1_ip_asc(self, param: Int, ip: Int) raises -> Int:
        return Int(self.crypto.get_function[c_int]("X509_VERIFY_PARAM_set1_ip_asc")(param, ip))

    def X509_get_default_cert_file(self) raises -> Int:
        return self.crypto.get_function[Int]("X509_get_default_cert_file")()

    def X509_get_default_cert_dir(self) raises -> Int:
        return self.crypto.get_function[Int]("X509_get_default_cert_dir")()

    def X509_get_default_cert_file_env(self) raises -> Int:
        return self.crypto.get_function[Int]("X509_get_default_cert_file_env")()

    def X509_get_default_cert_dir_env(self) raises -> Int:
        return self.crypto.get_function[Int]("X509_get_default_cert_dir_env")()

    # ---- libssl ----
    def TLS_client_method(self) raises -> Int:
        return self.ssl.get_function[Int]("TLS_client_method")()

    def SSL_CTX_new(self, method: Int) raises -> Int:
        return self.ssl.get_function[Int]("SSL_CTX_new")(method)

    def SSL_CTX_free(self, ctx: Int) raises:
        self.ssl.get_function[NoneType]("SSL_CTX_free")(ctx)

    def SSL_CTX_ctrl(self, ctx: Int, cmd: Int, larg: Int, parg: Int) raises -> Int:
        return self.ssl.get_function[Int]("SSL_CTX_ctrl")(ctx, c_int(cmd), larg, parg)

    def SSL_CTX_set_options(self, ctx: Int, op: UInt64) raises -> UInt64:
        return self.ssl.get_function[UInt64]("SSL_CTX_set_options")(ctx, op)

    def SSL_CTX_set_verify(self, ctx: Int, mode: Int, callback: Int) raises:
        self.ssl.get_function[NoneType]("SSL_CTX_set_verify")(ctx, c_int(mode), callback)

    def SSL_CTX_set_default_verify_paths(self, ctx: Int) raises -> Int:
        return Int(self.ssl.get_function[c_int]("SSL_CTX_set_default_verify_paths")(ctx))

    def SSL_CTX_load_verify_locations(self, ctx: Int, ca_file: Int, ca_path: Int) raises -> Int:
        return Int(self.ssl.get_function[c_int]("SSL_CTX_load_verify_locations")(ctx, ca_file, ca_path))

    def SSL_new(self, ctx: Int) raises -> Int:
        return self.ssl.get_function[Int]("SSL_new")(ctx)

    def SSL_free(self, ssl: Int) raises:
        self.ssl.get_function[NoneType]("SSL_free")(ssl)

    def SSL_set_fd(self, ssl: Int, fd: Int) raises -> Int:
        return Int(self.ssl.get_function[c_int]("SSL_set_fd")(ssl, c_int(fd)))

    def SSL_ctrl(self, ssl: Int, cmd: Int, larg: Int, parg: Int) raises -> Int:
        return self.ssl.get_function[Int]("SSL_ctrl")(ssl, c_int(cmd), larg, parg)

    def SSL_set_hostflags(self, ssl: Int, flags: Int) raises:
        self.ssl.get_function[NoneType]("SSL_set_hostflags")(ssl, UInt32(flags))

    def SSL_set1_host(self, ssl: Int, host: Int) raises -> Int:
        return Int(self.ssl.get_function[c_int]("SSL_set1_host")(ssl, host))

    def SSL_get0_param(self, ssl: Int) raises -> Int:
        return self.ssl.get_function[Int]("SSL_get0_param")(ssl)

    def SSL_connect(self, ssl: Int) raises -> Int:
        return Int(self.ssl.get_function[c_int]("SSL_connect")(ssl))

    def SSL_read_ex(self, ssl: Int, buf: Int, num: Int, readbytes_addr: Int) raises -> Int:
        return Int(self.ssl.get_function[c_int]("SSL_read_ex")(ssl, buf, UInt(num), readbytes_addr))

    def SSL_write_ex(self, ssl: Int, buf: Int, num: Int, written_addr: Int) raises -> Int:
        return Int(self.ssl.get_function[c_int]("SSL_write_ex")(ssl, buf, UInt(num), written_addr))

    def SSL_get_error(self, ssl: Int, ret: Int) raises -> Int:
        return Int(self.ssl.get_function[c_int]("SSL_get_error")(ssl, c_int(ret)))

    def SSL_get_verify_result(self, ssl: Int) raises -> Int:
        return self.ssl.get_function[Int]("SSL_get_verify_result")(ssl)

    def SSL_get1_peer_certificate(self, ssl: Int) raises -> Int:
        return self.ssl.get_function[Int]("SSL_get1_peer_certificate")(ssl)

    def SSL_get_version(self, ssl: Int) raises -> Int:
        return self.ssl.get_function[Int]("SSL_get_version")(ssl)

    def SSL_shutdown(self, ssl: Int) raises -> Int:
        return Int(self.ssl.get_function[c_int]("SSL_shutdown")(ssl))
