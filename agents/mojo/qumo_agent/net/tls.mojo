"""TLS client context and handshake with strict verification (DESIGN.md 4.5).

One `TlsContext` per process: TLS 1.2 minimum, no compression, no renegotiation, peer
verification on, and a trust store chosen in this order: `$SSL_CERT_FILE` /
`$SSL_CERT_DIR`; OpenSSL's compiled-in default file or directory; `/etc/ssl/cert.pem`.
The macOS Keychain is never consulted. There is no switch that disables verification;
tests build a context that trusts only their own CA file.

Each handshake sets SNI and RFC 6125 host-name checking (no partial wildcards) for a
host name, or the iPAddress check (and no SNI) for an IP literal, then requires
`SSL_connect == 1`, a peer certificate and `SSL_get_verify_result == X509_V_OK`.
"""

from std.memory import ArcPointer
from std.os import getenv, listdir
from std.os.path import exists, isdir

from qumo_agent.core.errors import E_NETWORK, E_TLS_SETUP
from qumo_agent.native.libc import CStr, c_string_at, buf_addr, errno, strerror_text, EAGAIN
from qumo_agent.net.ossl import (
    OpenSsl, SSL_ERROR_WANT_READ, SSL_ERROR_WANT_WRITE, SSL_ERROR_SYSCALL, SSL_ERROR_SSL,
    SSL_CTRL_SET_TLSEXT_HOSTNAME, TLSEXT_NAMETYPE_host_name, SSL_CTRL_SET_MIN_PROTO_VERSION,
    TLS1_2_VERSION, SSL_OP_NO_COMPRESSION, SSL_OP_NO_RENEGOTIATION, SSL_VERIFY_PEER,
    X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS, X509_V_OK,
)
from qumo_agent.net.fdutil import set_io_timeouts, ip_literal_family


def _base_context(lib: OpenSsl) raises -> Int:
    """Steps 1-4 of DESIGN.md 4.5: a client context with TLS >= 1.2 and peer verification."""
    lib.ERR_clear_error()
    var ctx = lib.SSL_CTX_new(lib.TLS_client_method())
    if ctx == 0:
        raise Error(E_TLS_SETUP + ": SSL_CTX_new failed: " + lib.errors())
    if lib.SSL_CTX_ctrl(ctx, SSL_CTRL_SET_MIN_PROTO_VERSION, TLS1_2_VERSION, 0) != 1:
        var q = lib.errors()
        lib.SSL_CTX_free(ctx)
        raise Error(E_TLS_SETUP + ": cannot require TLS 1.2: " + q)
    _ = lib.SSL_CTX_set_options(ctx, SSL_OP_NO_COMPRESSION | SSL_OP_NO_RENEGOTIATION)
    lib.SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, 0)
    return ctx


def _load_locations(lib: OpenSsl, ctx: Int, file: String, dir: String) raises -> Bool:
    var f = CStr(file)
    var d = CStr(dir)
    var fa = f.addr() if file.byte_length() > 0 else 0
    var da = d.addr() if dir.byte_length() > 0 else 0
    var ok = lib.SSL_CTX_load_verify_locations(ctx, fa, da) == 1
    f.keep()
    d.keep()
    return ok


def _dir_has_entries(path: String) -> Bool:
    try:
        if not isdir(path):
            return False
        return len(listdir(path)) > 0
    except:
        return False


struct TlsContext(Movable):
    var lib: OpenSsl
    var ctx: Int
    var trust_source: String        # "SSL_CERT_FILE=/x", "SSL_CERT_DIR=/y", "openssl-default:<file>", "/etc/ssl/cert.pem", "ca-file:<f>"

    def __init__(out self) raises:
        """Production trust store (DESIGN.md 4.5 step 5)."""
        var lib = OpenSsl()
        var ctx = _base_context(lib)
        var source = String()
        try:
            var env_file = getenv("SSL_CERT_FILE", "")
            var env_dir = getenv("SSL_CERT_DIR", "")
            if env_file.byte_length() > 0 or env_dir.byte_length() > 0:
                if env_file.byte_length() > 0:
                    source = "SSL_CERT_FILE=" + env_file
                if env_dir.byte_length() > 0:
                    if source.byte_length() > 0:
                        source += ", "
                    source += "SSL_CERT_DIR=" + env_dir
                if not _load_locations(lib, ctx, env_file, env_dir):
                    raise Error(E_TLS_SETUP + ": cannot load the CA certificates of " + source + ": "
                                + lib.errors() + ". Hint: point SSL_CERT_FILE at a PEM bundle")
            else:
                var dfile = c_string_at(lib.X509_get_default_cert_file())
                var ddir = c_string_at(lib.X509_get_default_cert_dir())
                var file_ok = dfile.byte_length() > 0 and exists(dfile)
                if file_ok or _dir_has_entries(ddir):
                    if lib.SSL_CTX_set_default_verify_paths(ctx) != 1:
                        raise Error(E_TLS_SETUP + ": cannot load OpenSSL's default CA certificates: "
                                    + lib.errors() + ". Hint: set SSL_CERT_FILE")
                    source = "openssl-default:" + (dfile if file_ok else ddir)
                elif exists("/etc/ssl/cert.pem"):
                    if not _load_locations(lib, ctx, "/etc/ssl/cert.pem", ""):
                        raise Error(E_TLS_SETUP + ": cannot load /etc/ssl/cert.pem: " + lib.errors()
                                    + ". Hint: set SSL_CERT_FILE")
                    source = "/etc/ssl/cert.pem"
                else:
                    raise Error(E_TLS_SETUP + ": no CA certificates found. Hint: set SSL_CERT_FILE")
        except e:
            lib.SSL_CTX_free(ctx)
            raise e^
        self.lib = lib^
        self.ctx = ctx
        self.trust_source = source^

    def __init__(out self, ca_file: String) raises:
        """Tests: trust only the certificates in `ca_file`."""
        var lib = OpenSsl()
        var ctx = _base_context(lib)
        if not _load_locations(lib, ctx, ca_file, ""):
            var q = lib.errors()
            lib.SSL_CTX_free(ctx)
            raise Error(E_TLS_SETUP + ": cannot load CA file " + ca_file + ": " + q)
        self.lib = lib^
        self.ctx = ctx
        self.trust_source = "ca-file:" + ca_file

    def __deinit__(deinit self):
        if self.ctx != 0:
            try:
                self.lib.SSL_CTX_free(self.ctx)
            except:
                pass


def _handshake_error(host: String, reason: String) -> Error:
    return Error(E_NETWORK + ": TLS handshake with " + host + " failed: " + reason)


def tls_handshake(ctx: ArcPointer[TlsContext], fd: Int, host: String, timeout_ms: Int) raises -> Int:
    """Runs the client handshake on the connected socket `fd` and returns the SSL*
    (DESIGN.md 4.5 steps 7-9). The caller owns `fd` (closed by the caller on failure);
    the SSL object is freed here when the handshake fails."""
    ref lib = ctx[].lib
    _ = set_io_timeouts(fd, timeout_ms)
    lib.ERR_clear_error()
    var ssl = lib.SSL_new(ctx[].ctx)
    if ssl == 0:
        raise _handshake_error(host, "SSL_new: " + lib.errors())
    var hc = CStr(host)
    try:
        if lib.SSL_set_fd(ssl, fd) != 1:
            raise _handshake_error(host, "SSL_set_fd: " + lib.errors())
        if ip_literal_family(host) != 0:
            # RFC 6066: no SNI for an IP literal; check the iPAddress SAN instead.
            if lib.X509_VERIFY_PARAM_set1_ip_asc(lib.SSL_get0_param(ssl), hc.addr()) != 1:
                raise _handshake_error(host, "cannot set the expected IP address: " + lib.errors())
        else:
            if lib.SSL_ctrl(ssl, SSL_CTRL_SET_TLSEXT_HOSTNAME, TLSEXT_NAMETYPE_host_name, hc.addr()) != 1:
                raise _handshake_error(host, "cannot set SNI: " + lib.errors())
            lib.SSL_set_hostflags(ssl, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS)
            if lib.SSL_set1_host(ssl, hc.addr()) != 1:
                raise _handshake_error(host, "cannot set the expected host name: " + lib.errors())
        lib.ERR_clear_error()
        var r = lib.SSL_connect(ssl)
        if r != 1:
            var se = lib.SSL_get_error(ssl, r)
            var err_no = errno()
            var vr = lib.SSL_get_verify_result(ssl)
            var reasons = List[Int]()
            var queue = lib.drain_errors(reasons)
            var reason: String
            if se == SSL_ERROR_WANT_READ or se == SSL_ERROR_WANT_WRITE:
                reason = "timed out after " + String(timeout_ms) + " ms"
            elif vr != X509_V_OK:
                reason = c_string_at(lib.X509_verify_cert_error_string(vr)) + " (" + String(vr) + ")"
            elif se == SSL_ERROR_SYSCALL and queue.byte_length() == 0:
                if err_no == 0:
                    reason = "connection closed by the server"
                else:
                    reason = strerror_text(err_no)
            else:
                reason = "SSL_get_error=" + String(se)
            if queue.byte_length() > 0:
                reason += "; " + queue
            raise _handshake_error(host, reason)
        var cert = lib.SSL_get1_peer_certificate(ssl)
        if cert == 0:
            raise _handshake_error(host, "the server sent no certificate")
        lib.X509_free(cert)
        var vr2 = lib.SSL_get_verify_result(ssl)
        if vr2 != X509_V_OK:
            raise _handshake_error(host, c_string_at(lib.X509_verify_cert_error_string(vr2)) + " (" + String(vr2) + ")")
    except e:
        hc.keep()
        lib.SSL_free(ssl)
        raise e^
    hc.keep()
    return ssl
