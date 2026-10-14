"""Tests for qumo.ai.checks.CheckLog (the CHECK line format parsed by
examples/run_examples.py and by agents)."""

from std.testing import assert_equal, assert_true, assert_false
from std.math import nan

from qumo.ai.checks import CheckLog


def _starts(s: String, prefix: String) -> Bool:
    return s.startswith(prefix)


def test_close_format() raises:
    var log = CheckLog("demo", quiet=True)
    assert_true(log.close("energy", 0.5, 0.25, 0.5))
    assert_equal(log.last_line, "CHECK energy: PASS value=0.5 ref=0.25 diff=0.25 tol=0.5")
    assert_false(log.close("energy tight", 1.0, 0.0, 1e-12))
    assert_equal(log.last_line, "CHECK energy tight: FAIL value=1.0 ref=0.0 diff=1.0 tol=1e-12")
    assert_equal(log.passed, 1)
    assert_equal(log.failed, 1)


def test_within_format() raises:
    var log = CheckLog("demo", quiet=True)
    assert_true(log.within("ratio", 0.8, 0.7559, 1.0))
    assert_equal(log.last_line, "CHECK ratio: PASS value=0.8 range=[0.7559, 1.0]")
    assert_false(log.within("ratio low", 0.5, 0.7559, 1.0))
    assert_true(_starts(log.last_line, "CHECK ratio low: FAIL value=0.5 range=["))
    # bounds are inclusive
    assert_true(log.within("edge", 1.0, 1.0, 1.0))


def test_check_format() raises:
    var log = CheckLog("demo", quiet=True)
    assert_true(log.check("optimal", True, "best=5 opt=5"))
    assert_equal(log.last_line, "CHECK optimal: PASS best=5 opt=5")
    assert_false(log.check("flag", False))
    assert_equal(log.last_line, "CHECK flag: FAIL")


def test_nan_fails() raises:
    var log = CheckLog("demo", quiet=True)
    var x = nan[DType.float64]()
    assert_false(log.close("nan close", x, 0.0, 1e9))
    assert_false(log.within("nan within", x, -1e9, 1e9))
    assert_equal(log.failed, 2)


def test_finish() raises:
    var ok = CheckLog("good", quiet=True)
    _ = ok.close("a", 1.0, 1.0, 0.0)
    _ = ok.check("b", True)
    ok.finish()  # must not raise
    assert_equal(ok.summary(), "ALL CHECKS PASSED (2)")

    var bad = CheckLog("bad", quiet=True)
    _ = bad.check("a", True)
    _ = bad.check("b", False)
    _ = bad.close("c", 2.0, 1.0, 0.1)
    var raised = False
    try:
        bad.finish()
    except e:
        raised = True
        var msg = String(e)
        assert_equal(msg, "bad: 2 of 3 checks failed (b, c)")
        assert_true(msg.find("QM-") < 0)
    assert_true(raised)


def test_empty_log_passes() raises:
    var log = CheckLog("empty", quiet=True)
    log.finish()
    assert_equal(log.summary(), "ALL CHECKS PASSED (0)")


def main() raises:
    test_close_format()
    test_within_format()
    test_check_format()
    test_nan_fails()
    test_finish()
    test_empty_log_passes()
    print("test_checks OK")
