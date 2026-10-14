"""Machine-readable verification lines for examples (and agents that run them).

Every QUMO example reports what it verified through one `CheckLog`, so the
example runner (`examples/run_examples.py`), CI and AI agents parse a fixed
format instead of free text. The Python twin is `examples/_checks.py`; both
print exactly the same line shapes:

```
CHECK <label>: PASS|FAIL value=<v> ref=<r> diff=<d> tol=<t>     # close()
CHECK <label>: PASS|FAIL value=<v> range=[<lo>, <hi>]          # within()
CHECK <label>: PASS|FAIL <detail>                              # check()
ALL CHECKS PASSED (<n>)                                        # finish()
```

`finish()` raises `Error("<example>: k of n checks failed")` (no QM code: a
failed check is a result, not a QUMO error), so `mojo run` exits non-zero.

```mojo
from qumo.testing.checks import CheckLog

def main() raises:
    var log = CheckLog("qaoa_maxcut")
    _ = log.close("energy", e, -4.4959, 1e-3)
    _ = log.within("ratio", r, 0.7559, 1.0)
    _ = log.check("best cut is optimal", best == opt, "best=" + String(best))
    log.finish()
```

Labels must not contain ':' (the runner splits on the first ': ').
"""

from std.math import isnan


def _fmt(x: Float64) -> String:
    return String(x)


struct CheckLog(Movable):
    """Counts and prints CHECK lines; `finish()` summarises."""

    var example: String
    var passed: Int
    var failed: Int
    var failed_labels: List[String]
    var last_line: String  # the most recent CHECK line (for tests and wrappers)
    var quiet: Bool  # True: record but do not print (tests)

    def __init__(out self, example: String, quiet: Bool = False):
        self.example = example
        self.passed = 0
        self.failed = 0
        self.failed_labels = List[String]()
        self.last_line = String()
        self.quiet = quiet

    def total(self) -> Int:
        return self.passed + self.failed

    def _record(mut self, label: String, ok: Bool, detail: String) -> Bool:
        var status = String("PASS") if ok else String("FAIL")
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failed_labels.append(label)
        if detail.byte_length() > 0:
            self.last_line = "CHECK " + label + ": " + status + " " + detail
        else:
            self.last_line = "CHECK " + label + ": " + status
        if not self.quiet:
            print(self.last_line)
        return ok

    def close(mut self, label: String, value: Float64, expected: Float64, tol: Float64) -> Bool:
        """PASS iff |value - expected| <= tol (NaN always fails). Printed as `ref=`.
        (`ref` is a Mojo keyword, hence the argument name.)"""
        var diff = abs(value - expected)
        var ok = (not isnan(value)) and (not isnan(expected)) and diff <= tol
        return self._record(
            label, ok,
            "value=" + _fmt(value) + " ref=" + _fmt(expected) + " diff=" + _fmt(diff) + " tol=" + _fmt(tol),
        )

    def within(mut self, label: String, value: Float64, lo: Float64, hi: Float64) -> Bool:
        """PASS iff lo <= value <= hi (NaN always fails)."""
        var ok = (not isnan(value)) and value >= lo and value <= hi
        return self._record(label, ok, "value=" + _fmt(value) + " range=[" + _fmt(lo) + ", " + _fmt(hi) + "]")

    def check(mut self, label: String, cond: Bool, detail: String = "") -> Bool:
        """PASS iff cond; `detail` is printed after the status."""
        return self._record(label, cond, detail)

    def all_passed(self) -> Bool:
        return self.failed == 0

    def summary(self) -> String:
        if self.failed == 0:
            return "ALL CHECKS PASSED (" + String(self.passed) + ")"
        return (
            self.example + ": " + String(self.failed) + " of " + String(self.total())
            + " checks failed (" + String(", ").join(self.failed_labels) + ")"
        )

    def finish(self) raises:
        """Print 'ALL CHECKS PASSED (n)' or raise '<example>: k of n checks failed'."""
        if self.failed == 0:
            if not self.quiet:
                print(self.summary())
            return
        raise Error(self.summary())
