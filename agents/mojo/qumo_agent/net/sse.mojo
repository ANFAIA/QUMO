"""Server-sent events reader (WHATWG event-stream format), for MCP streamable HTTP.

`SseParser.feed` accepts any split of the byte stream; lines end with CR, LF or CRLF
(a CR at the end of one chunk and an LF at the start of the next are one ending). A
leading UTF-8 BOM is ignored. Comment lines (starting with ':') are ignored. An event
is dispatched at an empty line; an event whose data is empty and whose block had no
`id` field is skipped. `id` is the last event id seen on the stream (it persists across
events, as in the specification); `retry_ms` is the `retry` field of the event's own
block, -1 when absent.
"""

from qumo_agent.core.errors import E_NETWORK

comptime MAX_SSE_LINE_BYTES = 32 * 1024 * 1024


@fieldwise_init
struct SseEvent(Copyable):
    var event: String          # "message" when no event field
    var data: String           # data lines joined with "\n"
    var id: String
    var retry_ms: Int          # -1 when absent


def _bom_byte(i: Int) -> Int:
    if i == 0:
        return 0xEF
    if i == 1:
        return 0xBB
    return 0xBF


struct SseParser(Movable):
    var _line: List[UInt8]
    var _bom_state: Int        # bytes of the BOM matched so far; -1 once the stream start is past
    var _skip_lf: Bool         # the previous byte was a CR ending a line
    var _event: String
    var _data: String
    var _data_lines: Int
    var _block_has_id: Bool
    var _last_id: String
    var _retry: Int
    var _queue: List[SseEvent]
    var _head: Int

    def __init__(out self):
        self._line = List[UInt8]()
        self._bom_state = 0
        self._skip_lf = False
        self._event = String()
        self._data = String()
        self._data_lines = 0
        self._block_has_id = False
        self._last_id = String()
        self._retry = -1
        self._queue = List[SseEvent]()
        self._head = 0

    def feed(mut self, chunk: List[UInt8]) raises:
        """Consumes a piece of the byte stream; complete events become available
        through `next_event`."""
        for i in range(len(chunk)):
            var b = Int(chunk[i])
            if self._bom_state >= 0:
                if b == _bom_byte(self._bom_state):
                    self._bom_state += 1
                    if self._bom_state == 3:
                        self._bom_state = -1
                    continue
                # Not a BOM after all: replay the bytes matched so far.
                var matched = self._bom_state
                self._bom_state = -1
                for k in range(matched):
                    self._byte(_bom_byte(k))
            self._byte(b)

    def _byte(mut self, b: Int) raises:
        if self._skip_lf:
            self._skip_lf = False
            if b == 0x0A:
                return
        if b == 0x0D:
            self._skip_lf = True
            self._end_line()
            return
        if b == 0x0A:
            self._end_line()
            return
        if len(self._line) >= MAX_SSE_LINE_BYTES:
            raise Error(E_NETWORK + ": SSE line longer than " + String(MAX_SSE_LINE_BYTES) + " bytes")
        self._line.append(UInt8(b))

    def _end_line(mut self):
        var line = String(from_utf8_lossy=Span(self._line))
        self._line.clear()
        if line.byte_length() == 0:
            self._dispatch()
            return
        if line.startswith(":"):
            return
        var field: String
        var value: String
        var colon = line.find(":")
        if colon < 0:
            field = line.copy()
            value = String()
        else:
            field = String(line[byte=0:colon])
            value = String(line[byte=colon + 1:])
            if value.startswith(" "):
                var t = String(value[byte=1:])
                value = t^
        if field == "event":
            self._event = value^
        elif field == "data":
            if self._data_lines > 0:
                self._data += "\n"
            self._data += value
            self._data_lines += 1
        elif field == "id":
            if value.find("\0") < 0:
                self._last_id = value^
                self._block_has_id = True
        elif field == "retry":
            var bs = value.as_bytes()
            if len(bs) > 0 and len(bs) <= 12:
                var v = 0
                var ok = True
                for i in range(len(bs)):
                    var c = Int(bs[i])
                    if c < 0x30 or c > 0x39:
                        ok = False
                        break
                    v = v * 10 + (c - 0x30)
                if ok:
                    self._retry = v

    def _dispatch(mut self):
        if self._data.byte_length() > 0 or self._block_has_id:
            var name = self._event.copy() if self._event.byte_length() > 0 else String("message")
            self._queue.append(SseEvent(name^, self._data.copy(), self._last_id.copy(), self._retry))
        self._event = String()
        self._data = String()
        self._data_lines = 0
        self._block_has_id = False
        self._retry = -1

    def next_event(mut self, mut ev: SseEvent) -> Bool:
        """Pops the oldest dispatched event into `ev`; False when none is pending."""
        if self._head >= len(self._queue):
            if len(self._queue) > 0:
                self._queue.clear()
                self._head = 0
            return False
        ev = self._queue[self._head].copy()
        self._head += 1
        return True
