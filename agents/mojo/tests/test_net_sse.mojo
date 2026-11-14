"""Tests for net/sse.mojo: every split of the stream, CR/LF/CRLF, comments, multi-line
data, id/retry, BOM, skipped empty events."""

from std.testing import assert_equal, assert_true, assert_false

from qumo_agent.net.sse import SseEvent, SseParser


def _bytes(s: String) -> List[UInt8]:
    var bs = s.as_bytes()
    var out = List[UInt8](capacity=len(bs))
    for i in range(len(bs)):
        out.append(bs[i])
    return out^


def _render(mut p: SseParser) -> String:
    """Pending events as "event|data|id|retry" lines."""
    var out = String()
    var ev = SseEvent(String(), String(), String(), -1)
    while p.next_event(ev):
        out += ev.event + "|" + ev.data + "|" + ev.id + "|" + String(ev.retry_ms) + "\n"
    return out^


def _parse_whole(stream: List[UInt8]) raises -> String:
    var p = SseParser()
    p.feed(stream)
    return _render(p)


def _check_all_splits(stream: List[UInt8], want: String) raises:
    assert_equal(_parse_whole(stream), want)
    # Two pieces, split at every offset.
    for k in range(len(stream) + 1):
        var a = List[UInt8]()
        var b = List[UInt8]()
        for i in range(len(stream)):
            if i < k:
                a.append(stream[i])
            else:
                b.append(stream[i])
        var p = SseParser()
        p.feed(a)
        var got = _render(p)
        p.feed(b)
        got += _render(p)
        assert_equal(got, want, "split at " + String(k))
    # One byte at a time.
    var p = SseParser()
    var got = String()
    for i in range(len(stream)):
        var one = List[UInt8]()
        one.append(stream[i])
        p.feed(one)
        got += _render(p)
    assert_equal(got, want, "byte by byte")


def test_basic() raises:
    var s = (": priming comment\n"
             + "id: 7\nretry: 1500\ndata:\n\n"                      # priming event: empty data with an id
             + "event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/x\"}\n\n"
             + "data: line one\ndata:line two\ndata:  indented\n\n"
             + "event: endpoint\ndata: /messages?s=1\n\n"
             + "data\n\n"                                            # field without colon, empty data, skipped
             + "event: ping\n\n"                                     # no data, no id: skipped
             + "id: 8\ndata: é ü\n\n"
             + "data: unterminated")
    var want = ("message||7|1500\n"
                + "message|{\"jsonrpc\":\"2.0\",\"method\":\"notifications/x\"}|7|-1\n"
                + "message|line one\nline two\n indented|7|-1\n"
                + "endpoint|/messages?s=1|7|-1\n"
                + "message|é ü|8|-1\n")
    _check_all_splits(_bytes(s), want)


def test_line_endings() raises:
    var lf = "data: a\ndata: b\n\nevent: e\ndata: c\n\n"
    var crlf = "data: a\r\ndata: b\r\n\r\nevent: e\r\ndata: c\r\n\r\n"
    var cr = "data: a\rdata: b\r\revent: e\rdata: c\r\r"
    var mixed = "data: a\r\ndata: b\r\n\nevent: e\rdata: c\n\r\n"
    var want = "message|a\nb||-1\ne|c||-1\n"
    _check_all_splits(_bytes(lf), want)
    _check_all_splits(_bytes(crlf), want)
    _check_all_splits(_bytes(cr), want)
    _check_all_splits(_bytes(mixed), want)


def test_bom_and_fields() raises:
    var s = List[UInt8]()
    s.append(0xEF)
    s.append(0xBB)
    s.append(0xBF)
    for b in _bytes("data: x\n\n"):
        s.append(b)
    _check_all_splits(s, "message|x||-1\n")
    # A BOM later in the stream is data, not ignored; bad retry values are ignored;
    # an id containing NUL is ignored.
    var t = _bytes("retry: 12x\ndata: y\n\nretry: 250\nid: a\0b\ndata: z\n\n")
    _check_all_splits(t, "message|y||-1\nmessage|z||250\n")
    # A stream starting with 0xEF that is not a BOM keeps the byte: the first line's
    # field is then not "data", so only the second event is dispatched.
    var u = List[UInt8]()
    u.append(0xEF)
    for b in _bytes("data: q\n\ndata: r\n\n"):
        u.append(b)
    _check_all_splits(u, "message|r||-1\n")


def main() raises:
    test_basic()
    test_line_endings()
    test_bom_and_fields()
    print("test_net_sse OK")
