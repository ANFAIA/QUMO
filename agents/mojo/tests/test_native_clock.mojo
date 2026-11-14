"""Tests for qumo_agent.native.clock (DESIGN.md 2.1.3, 8.2).

Expected values come from Python 3.12 (`datetime.isoformat`, `email.utils`).
"""

from std.testing import assert_equal, assert_true, assert_false

from qumo_agent.native.clock import *


def test_utc_iso() raises:
    assert_equal(utc_iso(0, 0), "1970-01-01T00:00:00+00:00")
    assert_equal(utc_iso(784111777, 0), "1994-11-06T08:49:37+00:00")
    assert_equal(utc_iso(1700000000, 123456), "2023-11-14T22:13:20.123456+00:00")
    assert_equal(utc_iso(1700000000, 1), "2023-11-14T22:13:20.000001+00:00")
    var now = unix_now()
    assert_true(now[0] > 1700000000)
    assert_true(now[1] >= 0 and now[1] < 1000000)
    var s = utc_iso_now()
    assert_true(s.endswith("+00:00"))
    assert_true(s.byte_length() == 25 or s.byte_length() == 32)


def _all_digits(s: String) -> Bool:
    for b in s.as_bytes():
        if Int(b) < 0x30 or Int(b) > 0x39:
            return False
    return True


def test_local_formats() raises:
    var d = local_date()
    assert_equal(d.byte_length(), 10)
    assert_equal(String(d[byte=4:5]), "-")
    assert_equal(String(d[byte=7:8]), "-")
    assert_true(_all_digits(String(d[byte=0:4])) and _all_digits(String(d[byte=5:7])) and _all_digits(String(d[byte=8:10])))
    var st = local_stamp()
    assert_equal(st.byte_length(), 15)
    assert_equal(String(st[byte=8:9]), "-")
    assert_true(_all_digits(String(st[byte=0:8])) and _all_digits(String(st[byte=9:15])))
    assert_equal(strftime_utc("%Y-%m-%d %H:%M:%S", 1445412480), "2015-10-21 07:28:00")
    assert_equal(strftime_utc("", 0), "")
    assert_equal(strftime_local("%Y", 1445412480).byte_length(), 4)
    # A long expansion grows the buffer.
    var fmt = String()
    for _ in range(100):
        fmt += "%Y-%m-%d "
    assert_equal(strftime_utc(fmt, 0).byte_length(), 1100)


def test_http_dates() raises:
    assert_equal(http_date_to_unix("Sun, 06 Nov 1994 08:49:37 GMT").value(), 784111777)
    assert_equal(http_date_to_unix("Sunday, 06-Nov-94 08:49:37 GMT").value(), 784111777)
    assert_equal(http_date_to_unix("Sun Nov  6 08:49:37 1994").value(), 784111777)
    assert_equal(http_date_to_unix("Wed, 21 Oct 2015 07:28:00 GMT").value(), 1445412480)
    assert_equal(http_date_to_unix("Thursday, 01-Jan-70 00:00:01 GMT").value(), 1)
    assert_equal(http_date_to_unix("Mon, 01-Jan-01 00:00:00 GMT").value(), 978307200)
    assert_false(Bool(http_date_to_unix("")))
    assert_false(Bool(http_date_to_unix("120")))
    assert_false(Bool(http_date_to_unix("Sun, 06 Foo 1994 08:49:37 GMT")))
    assert_false(Bool(http_date_to_unix("Sun, 06 Nov 1994 25:49:37 GMT")))


def test_monotonic() raises:
    var a = monotonic_ms()
    sleep_ms(30)
    var b = monotonic_ms()
    assert_true(b - a >= 25 and b - a < 2000)
    sleep_ms(0)
    sleep_ms(-5)


def main() raises:
    test_utc_iso()
    test_local_formats()
    test_http_dates()
    test_monotonic()
    print("test_native_clock OK")
