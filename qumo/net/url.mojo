"""HTTP(S) URLs: parsing, reference resolution (redirect `Location`) and urllib quoting.

Only `http` and `https` are accepted. A URL with user information (`user@host`) is
refused, the fragment is dropped, and CR, LF, space and other control characters are
refused anywhere in the text (they would let a caller inject request lines). Bytes
above 0x7F in the path or query are percent-encoded; a non-ASCII host is refused.
"""

from qumo.net.errors import E_NETWORK


comptime _HEX_UPPER = "0123456789ABCDEF"


@fieldwise_init
struct Url(Copyable):
    var scheme: String       # "http" | "https" (lower-case)
    var host: String         # lower-case; IPv6 without brackets
    var port: Int
    var path_query: String   # starts with "/", "/" when empty

    def default_port(self) -> Int:
        return 443 if self.scheme == "https" else 80

    def host_part(self) -> String:
        """The host as written in a URL: IPv6 literals in brackets."""
        if self.host.find(":") >= 0:
            return "[" + self.host + "]"
        return self.host

    def host_header(self) -> String:
        """host or [v6], with ":port" when it is not the scheme's default port."""
        if self.port == self.default_port():
            return self.host_part()
        return self.host_part() + ":" + String(self.port)

    def origin(self) -> String:
        return self.scheme + "://" + self.host_header()

    def to_string(self) -> String:
        return self.origin() + self.path_query


def _url_error(url: String, why: String) -> Error:
    return Error(E_NETWORK + ": invalid URL '" + url + "': " + why)


def _is_alpha(c: Int) -> Bool:
    return (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A)


def _is_digit(c: Int) -> Bool:
    return c >= 0x30 and c <= 0x39


def _lower_ascii(s: String) -> String:
    var bs = s.as_bytes()
    var out = List[UInt8](capacity=len(bs))
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c >= 0x41 and c <= 0x5A:
            c += 0x20
        out.append(UInt8(c))
    return String(from_utf8_lossy=Span(out))


def _encode_high_bytes(s: String) -> String:
    """Percent-encodes every byte above 0x7F (a URL path or query typed with non-ASCII)."""
    var bs = s.as_bytes()
    var plain = True
    for i in range(len(bs)):
        if Int(bs[i]) >= 0x80:
            plain = False
            break
    if plain:
        return s
    var hexd = _HEX_UPPER.as_bytes()
    var out = List[UInt8](capacity=len(bs) * 3)
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c >= 0x80:
            out.append(0x25)
            out.append(hexd[c >> 4])
            out.append(hexd[c & 0xF])
        else:
            out.append(UInt8(c))
    return String(unsafe_from_utf8=Span(out))


def _scheme_end(s: String) -> Int:
    """Index of the ':' ending a URI scheme (RFC 3986: ALPHA *( ALPHA / DIGIT / + - . )), or -1."""
    var bs = s.as_bytes()
    if len(bs) == 0 or not _is_alpha(Int(bs[0])):
        return -1
    for i in range(1, len(bs)):
        var c = Int(bs[i])
        if c == 0x3A:
            return i
        if not (_is_alpha(c) or _is_digit(c) or c == 0x2B or c == 0x2D or c == 0x2E):
            return -1
    return -1


def _check_chars(url: String) raises:
    var bs = url.as_bytes()
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c <= 0x20 or c == 0x7F:
            raise _url_error(url, "control character or space")


def _valid_host(host: String, bracketed: Bool) -> Bool:
    var bs = host.as_bytes()
    if len(bs) == 0:
        return False
    for i in range(len(bs)):
        var c = Int(bs[i])
        if bracketed:
            if not (_is_digit(c) or _is_alpha(c) or c == 0x3A or c == 0x2E or c == 0x25):
                return False
        elif not (_is_digit(c) or _is_alpha(c) or c == 0x2D or c == 0x2E or c == 0x5F):
            return False
    return True


def parse_url(url: String) raises -> Url:
    """Parses an absolute http(s) URL. Rejects user information, drops the fragment,
    refuses CR, LF, space and other control characters."""
    _check_chars(url)
    var colon = _scheme_end(url)
    if colon < 0:
        raise _url_error(url, "no scheme")
    var scheme = _lower_ascii(String(url[byte=0:colon]))
    if scheme != "http" and scheme != "https":
        raise _url_error(url, "only http and https URLs are supported")
    var rest = String(url[byte=colon + 1:])
    if not rest.startswith("//"):
        raise _url_error(url, "no host")
    var tail = String(rest[byte=2:])
    rest = tail^
    var hash = rest.find("#")
    if hash >= 0:
        var cut = String(rest[byte=0:hash])
        rest = cut^
    # The authority ends at the first '/' or '?'.
    var rb = rest.as_bytes()
    var auth_end = len(rb)
    for i in range(len(rb)):
        var c = Int(rb[i])
        if c == 0x2F or c == 0x3F:
            auth_end = i
            break
    var authority = String(rest[byte=0:auth_end])
    var path_query = String(rest[byte=auth_end:])
    if authority.find("@") >= 0:
        raise _url_error(url, "user information in URLs is not supported")
    var host: String
    var port_text = String()
    var bracketed = False
    if authority.startswith("["):
        var close = authority.find("]")
        if close < 0:
            raise _url_error(url, "unterminated IPv6 literal")
        host = String(authority[byte=1:close])
        bracketed = True
        var after = String(authority[byte=close + 1:])
        if after.byte_length() > 0:
            if not after.startswith(":"):
                raise _url_error(url, "invalid authority")
            port_text = String(after[byte=1:])
    else:
        var c2 = authority.find(":")
        if c2 >= 0:
            host = String(authority[byte=0:c2])
            port_text = String(authority[byte=c2 + 1:])
        else:
            host = authority.copy()
    host = _lower_ascii(host)
    if not _valid_host(host, bracketed):
        raise _url_error(url, "invalid host")
    if bracketed and host.find(":") < 0:
        raise _url_error(url, "invalid IPv6 literal")
    var port = 443 if scheme == "https" else 80
    if port_text.byte_length() > 0:
        var pb = port_text.as_bytes()
        if len(pb) > 5:
            raise _url_error(url, "invalid port")
        var v = 0
        for i in range(len(pb)):
            var c = Int(pb[i])
            if not _is_digit(c):
                raise _url_error(url, "invalid port")
            v = v * 10 + (c - 0x30)
        if v < 1 or v > 65535:
            raise _url_error(url, "invalid port")
        port = v
    if path_query.byte_length() == 0:
        path_query = "/"
    elif path_query.startswith("?"):
        path_query = "/" + path_query
    return Url(scheme, host, port, _encode_high_bytes(path_query))


def _remove_dot_segments(path: String) -> String:
    """RFC 3986 section 5.2.4."""
    var input = path.copy()
    var out = List[String]()   # output segments, each starting with "/" (or the first one without)
    while input.byte_length() > 0:
        if input.startswith("../"):
            var t = String(input[byte=3:])
            input = t^
        elif input.startswith("./"):
            var t = String(input[byte=2:])
            input = t^
        elif input.startswith("/./"):
            var t = String(input[byte=2:])
            input = t^
        elif input == "/.":
            input = "/"
        elif input.startswith("/../"):
            var t = String(input[byte=3:])
            input = t^
            if len(out) > 0:
                _ = out.pop()
        elif input == "/..":
            input = "/"
            if len(out) > 0:
                _ = out.pop()
        elif input == "." or input == "..":
            input = ""
        else:
            # Move the first segment (with its leading "/" if any) to the output.
            var bs = input.as_bytes()
            var start = 1 if Int(bs[0]) == 0x2F else 0
            var end = len(bs)
            for i in range(start, len(bs)):
                if Int(bs[i]) == 0x2F:
                    end = i
                    break
            out.append(String(input[byte=0:end]))
            var t = String(input[byte=end:])
            input = t^
    var result = String()
    for s in out:
        result += s
    return result^


def resolve_reference(base: Url, reference: String) raises -> Url:
    """RFC 3986 section 5.2: the target of `reference` (a redirect `Location`) relative to `base`."""
    _check_chars(reference)
    var r = reference.copy()
    var hash = r.find("#")
    if hash >= 0:
        var cut = String(r[byte=0:hash])
        r = cut^
    if _scheme_end(r) >= 0:
        var u = parse_url(r)
        return _with_normalized_path(u^)
    if r.startswith("//"):
        var u = parse_url(base.scheme + ":" + r)
        return _with_normalized_path(u^)
    var path = r.copy()
    var query = String()
    var has_query = False
    var q = r.find("?")
    if q >= 0:
        path = String(r[byte=0:q])
        query = String(r[byte=q + 1:])
        has_query = True
    var base_path = base.path_query.copy()
    var base_query = String()
    var base_has_query = False
    var bq = base_path.find("?")
    if bq >= 0:
        base_query = String(base.path_query[byte=bq + 1:])
        base_path = String(base.path_query[byte=0:bq])
        base_has_query = True
    var target_path: String
    if path.byte_length() == 0:
        target_path = base_path.copy()
        if not has_query:
            query = base_query.copy()
            has_query = base_has_query
    elif path.startswith("/"):
        target_path = _remove_dot_segments(path)
    else:
        var slash = base_path.rfind("/")
        var merged = String(base_path[byte=0:slash + 1]) + path if slash >= 0 else "/" + path
        target_path = _remove_dot_segments(merged)
    if target_path.byte_length() == 0 or not target_path.startswith("/"):
        target_path = "/" + target_path
    var pq = target_path
    if has_query:
        pq += "?" + query
    return Url(base.scheme, base.host, base.port, _encode_high_bytes(pq))


def _with_normalized_path(var u: Url) -> Url:
    var pq = u.path_query.copy()
    var q = pq.find("?")
    var path = String(pq[byte=0:q]) if q >= 0 else pq.copy()
    var query = String(pq[byte=q:]) if q >= 0 else String()
    var norm = _remove_dot_segments(path)
    if not norm.startswith("/"):
        norm = "/" + norm
    u.path_query = norm + query
    return u^


def quote(s: String, safe: String) -> String:
    """urllib.parse.quote: UTF-8 bytes, `A-Za-z0-9_.-~` and the bytes of `safe` kept,
    everything else as %XX (upper-case hex)."""
    var sb = safe.as_bytes()
    var bs = s.as_bytes()
    var hexd = _HEX_UPPER.as_bytes()
    var out = List[UInt8](capacity=len(bs) * 3)
    for i in range(len(bs)):
        var c = Int(bs[i])
        var keep = _is_alpha(c) or _is_digit(c) or c == 0x5F or c == 0x2E or c == 0x2D or c == 0x7E
        if not keep and c < 0x80:
            for j in range(len(sb)):
                if Int(sb[j]) == c:
                    keep = True
                    break
        if keep:
            out.append(UInt8(c))
        else:
            out.append(0x25)
            out.append(hexd[c >> 4])
            out.append(hexd[c & 0xF])
    return String(unsafe_from_utf8=Span(out))


def quote_plus(s: String) -> String:
    """urllib.parse.quote_plus(s): like quote with safe "", but a space becomes "+"."""
    if s.find(" ") < 0:
        return quote(s, "")
    return quote(s, " ").replace(" ", "+")


def urlencode(names: List[String], values: List[String]) -> String:
    """urllib.parse.urlencode of ordered pairs (quote_via=quote_plus)."""
    var out = String()
    var n = min(len(names), len(values))
    for i in range(n):
        if i > 0:
            out += "&"
        out += quote_plus(names[i]) + "=" + quote_plus(values[i])
    return out^
