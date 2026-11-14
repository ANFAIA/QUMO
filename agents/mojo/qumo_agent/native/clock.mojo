"""Wall clock, local dates, monotonic time and HTTP dates (DESIGN.md 2.1.3).

UTC timestamps use `clock_gettime(CLOCK_REALTIME)` + `gmtime_r` + `strftime`; local
dates use `time` + `localtime_r` (the process time zone, as Python's
`date.today()`). No `date` process is ever started.
"""

from std.time import perf_counter_ns, sleep

from qumo_agent.native.libc import (
    CLOCK_REALTIME,
    SIZEOF_TM,
    TM_SEC_OFFSET,
    TM_MIN_OFFSET,
    TM_HOUR_OFFSET,
    TM_MDAY_OFFSET,
    TM_MON_OFFSET,
    TM_YEAR_OFFSET,
    CStr,
    buf_addr,
    store_u32,
    c_clock_gettime,
    c_time,
    c_gmtime_r,
    c_localtime_r,
    c_strftime,
    c_timegm,
)


def unix_now() -> Tuple[Int, Int]:
    """(seconds, microseconds) since the epoch, from clock_gettime(CLOCK_REALTIME)."""
    var ts = List[Int64](length=2, fill=0)
    _ = c_clock_gettime(CLOCK_REALTIME, Int(ts.unsafe_ptr()))
    var sec = Int(ts[0])
    var usec = Int(ts[1]) // 1000
    return (sec, usec)


def _two(v: Int) -> String:
    if v < 10:
        return "0" + String(v)
    return String(v)


def _six(v: Int) -> String:
    var s = String(v)
    while s.byte_length() < 6:
        s = "0" + s
    return s


def _strftime(fmt: String, sec: Int, local: Bool) raises -> String:
    if fmt.byte_length() == 0:
        return String()
    var t = List[Int64](length=1, fill=0)
    t[0] = Int64(sec)
    var tm = List[UInt8](length=SIZEOF_TM, fill=0)
    var r: Int
    if local:
        r = c_localtime_r(Int(t.unsafe_ptr()), buf_addr(tm))
    else:
        r = c_gmtime_r(Int(t.unsafe_ptr()), buf_addr(tm))
    _ = len(t)
    if r == 0:
        raise Error("QA-E160: time " + String(sec) + " cannot be converted")
    var cfmt = CStr(fmt)
    var size = 256
    while size <= 65536:
        var buf = List[UInt8](length=size, fill=0)
        var n = c_strftime(buf_addr(buf), size, cfmt.addr(), buf_addr(tm))
        if n > 0:
            var out = String(from_utf8_lossy=Span(buf)[0:n])
            _ = len(buf)
            cfmt.keep()
            _ = len(tm)
            return out^
        _ = len(buf)
        size *= 4
    cfmt.keep()
    _ = len(tm)
    # strftime returns 0 both for "buffer too small" and for an empty result
    # (e.g. a format of only "%p" in some locales); after 64 KiB it is empty.
    return String()


def strftime_utc(fmt: String, sec: Int) raises -> String:
    """`time.strftime(fmt, time.gmtime(sec))`."""
    return _strftime(fmt, sec, False)


def strftime_local(fmt: String, sec: Int) raises -> String:
    """`time.strftime(fmt, time.localtime(sec))`."""
    return _strftime(fmt, sec, True)


def utc_iso(sec: Int, usec: Int) -> String:
    """Python `datetime.fromtimestamp(sec + usec/1e6, UTC).isoformat()`:
    `YYYY-MM-DDTHH:MM:SS+00:00`, with `.ffffff` before the offset when usec != 0."""
    var base: String
    try:
        base = strftime_utc("%Y-%m-%dT%H:%M:%S", sec)
    except:
        base = String("1970-01-01T00:00:00")
    if usec != 0:
        base += "." + _six(usec)
    return base + "+00:00"


def utc_iso_now() -> String:
    var now = unix_now()
    return utc_iso(now[0], now[1])


def local_date() -> String:
    """`date.today().isoformat()` ("%Y-%m-%d" in the local time zone)."""
    try:
        return strftime_local("%Y-%m-%d", c_time())
    except:
        return String()


def local_stamp() -> String:
    """"%Y%m%d-%H%M%S" in the local time zone (open_sandbox branch names)."""
    try:
        return strftime_local("%Y%m%d-%H%M%S", c_time())
    except:
        return String()


def monotonic_ms() -> Int:
    """`time.perf_counter_ns() // 1_000_000`."""
    return Int(perf_counter_ns()) // 1_000_000


def sleep_ms(ms: Int):
    if ms <= 0:
        return
    sleep(Float64(ms) / 1000.0)


# ---------------------------------------------------------------------------
# HTTP dates (RFC 7231 section 7.1.1.1)
# ---------------------------------------------------------------------------
comptime _MONTHS = "janfebmaraprmayjunjulaugsepoctnovdec"
comptime _DAYS = "monday tuesday wednesday thursday friday saturday sunday"


def _month_index(tok: String) -> Int:
    """0-based month for a 3-letter English month name (any case), else -1."""
    if tok.byte_length() != 3:
        return -1
    var low = tok.lower()
    var months = String(_MONTHS)
    for i in range(12):
        if String(months[byte=i * 3 : i * 3 + 3]) == low:
            return i
    return -1


def _is_day_name(tok: String) -> Bool:
    var low = tok.lower()
    if low.byte_length() < 3:
        return False
    for d in String(_DAYS).split(" "):
        var day = String(d)
        if low == day or low == String(day[byte=0:3]):
            return True
    return False


def _parse_uint(tok: String) -> Int:
    """Decimal digits only; -1 otherwise."""
    var bs = tok.as_bytes()
    if len(bs) == 0 or len(bs) > 9:
        return -1
    var v = 0
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c < 0x30 or c > 0x39:
            return -1
        v = v * 10 + (c - 0x30)
    return v


def _parse_hms(tok: String) -> Optional[Tuple[Int, Int, Int]]:
    var parts = tok.split(":")
    if len(parts) != 3:
        return None
    var h = _parse_uint(String(parts[0]))
    var m = _parse_uint(String(parts[1]))
    var s = _parse_uint(String(parts[2]))
    if h < 0 or m < 0 or s < 0 or h > 23 or m > 59 or s > 60:
        return None
    return (h, m, s)


def _to_unix(year: Int, mon: Int, day: Int, hms: Tuple[Int, Int, Int]) -> Optional[Int]:
    if year < 1900 or mon < 0 or mon > 11 or day < 1 or day > 31:
        return None
    var tm = List[UInt8](length=SIZEOF_TM, fill=0)
    var a = buf_addr(tm)
    store_u32(a + TM_SEC_OFFSET, hms[2])
    store_u32(a + TM_MIN_OFFSET, hms[1])
    store_u32(a + TM_HOUR_OFFSET, hms[0])
    store_u32(a + TM_MDAY_OFFSET, day)
    store_u32(a + TM_MON_OFFSET, mon)
    store_u32(a + TM_YEAR_OFFSET, year - 1900)
    var t = c_timegm(a)
    _ = len(tm)
    return t


def http_date_to_unix(value: String) -> Optional[Int]:
    """Seconds since the epoch for an HTTP date in one of the three RFC 7231 forms:
    IMF-fixdate `Sun, 06 Nov 1994 08:49:37 GMT`, RFC 850 `Sunday, 06-Nov-94 08:49:37 GMT`
    (two-digit years: > 68 is 19xx, else 20xx, as Python's email.utils), and asctime
    `Sun Nov  6 08:49:37 1994`. None when the value is none of them."""
    var toks = List[String]()
    for t in value.replace(",", " ").split(" "):
        var tok = String(t).strip()
        if tok.byte_length() > 0:
            toks.append(String(tok))
    if len(toks) == 0:
        return None
    var i = 0
    if _is_day_name(toks[0]):
        i = 1
    var rest = len(toks) - i
    # RFC 850: 06-Nov-94 08:49:37 GMT
    if rest == 3 and toks[i].find("-") >= 0:
        var dparts = toks[i].split("-")
        if len(dparts) != 3:
            return None
        var day = _parse_uint(String(dparts[0]))
        var mon = _month_index(String(dparts[1]))
        var year = _parse_uint(String(dparts[2]))
        var hms = _parse_hms(toks[i + 1])
        var zone = toks[i + 2].upper()
        if day < 0 or mon < 0 or year < 0 or not hms or (zone != "GMT" and zone != "UTC"):
            return None
        if year < 100:
            year += 1900 if year > 68 else 2000
        return _to_unix(year, mon, day, hms.value())
    # IMF-fixdate: 06 Nov 1994 08:49:37 GMT
    if rest == 5 and _parse_uint(toks[i]) >= 0:
        var day = _parse_uint(toks[i])
        var mon = _month_index(toks[i + 1])
        var year = _parse_uint(toks[i + 2])
        var hms = _parse_hms(toks[i + 3])
        var zone = toks[i + 4].upper()
        if mon < 0 or year < 0 or not hms or (zone != "GMT" and zone != "UTC" and zone != "+0000" and zone != "-0000"):
            return None
        if year < 100:
            year += 1900 if year > 68 else 2000
        return _to_unix(year, mon, day, hms.value())
    # asctime: Nov  6 08:49:37 1994
    if rest == 4 and _month_index(toks[i]) >= 0:
        var mon = _month_index(toks[i])
        var day = _parse_uint(toks[i + 1])
        var hms = _parse_hms(toks[i + 2])
        var year = _parse_uint(toks[i + 3])
        if day < 0 or year < 0 or not hms:
            return None
        return _to_unix(year, mon, day, hms.value())
    return None
