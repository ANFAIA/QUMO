"""Machine-readable verification lines for Python examples.

Python twin of ``qumo/ai/checks.mojo`` (same line format, same semantics)::

    CHECK <label>: PASS|FAIL value=<v> ref=<r> diff=<d> tol=<t>     # close()
    CHECK <label>: PASS|FAIL value=<v> range=[<lo>, <hi>]          # within()
    CHECK <label>: PASS|FAIL <detail>                              # check()
    ALL CHECKS PASSED (<n>)                                        # finish()

``finish()`` exits with status 1 after printing
``<example>: k of n checks failed`` to stderr when any check failed.

Usage::

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from _checks import CheckLog          # (from examples/<dir>/: add the parent)

    log = CheckLog("my_example")
    log.close("energy", e, -1.137, 1e-6)
    log.within("ratio", r, 0.7559, 1.0)
    log.check("optimal sampled", best == opt, f"best={best}")
    log.finish()

Labels must not contain ':' (the runner splits on the first ': ').
"""

from __future__ import annotations

import math
import sys

__all__ = ["CheckLog", "fmt"]


def fmt(x: float) -> str:
    """Shortest round-trip repr (Mojo prints Float64 the same way)."""
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return repr(x)


class CheckLog:
    """Counts and prints CHECK lines; ``finish()`` summarises."""

    def __init__(self, example: str, stream=None):
        self.example = example
        self.passed = 0
        self.failed = 0
        self.failed_labels: list[str] = []
        self.records: list[dict] = []
        self._stream = stream

    def _print(self, line: str) -> None:
        print(line, file=self._stream or sys.stdout, flush=True)

    @property
    def total(self) -> int:
        return self.passed + self.failed

    def _record(self, label: str, ok: bool, detail: str, **data) -> bool:
        ok = bool(ok)
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failed_labels.append(label)
        status = "PASS" if ok else "FAIL"
        self._print(f"CHECK {label}: {status} {detail}" if detail else f"CHECK {label}: {status}")
        self.records.append({"label": label, "status": status, "detail": detail, **data})
        return ok

    def close(self, label: str, value: float, ref: float, tol: float) -> bool:
        """PASS iff |value - ref| <= tol (NaN always fails)."""
        value, ref, tol = float(value), float(ref), float(tol)
        diff = abs(value - ref)
        ok = not (math.isnan(value) or math.isnan(ref)) and diff <= tol
        return self._record(
            label, ok,
            f"value={fmt(value)} ref={fmt(ref)} diff={fmt(diff)} tol={fmt(tol)}",
            value=value, ref=ref, diff=diff, tol=tol,
        )

    def within(self, label: str, value: float, lo: float, hi: float) -> bool:
        """PASS iff lo <= value <= hi (NaN always fails)."""
        value, lo, hi = float(value), float(lo), float(hi)
        ok = not math.isnan(value) and lo <= value <= hi
        return self._record(label, ok, f"value={fmt(value)} range=[{fmt(lo)}, {fmt(hi)}]",
                            value=value, lo=lo, hi=hi)

    def check(self, label: str, cond: bool, detail: str = "") -> bool:
        """PASS iff cond; ``detail`` is printed after the status."""
        return self._record(label, bool(cond), detail)

    def all_passed(self) -> bool:
        return self.failed == 0

    def summary(self) -> str:
        if self.failed == 0:
            return f"ALL CHECKS PASSED ({self.passed})"
        return (f"{self.example}: {self.failed} of {self.total} checks failed "
                f"({', '.join(self.failed_labels)})")

    def finish(self, exit_on_failure: bool = True) -> bool:
        """Print 'ALL CHECKS PASSED (n)', or report failures and exit(1)."""
        if self.failed == 0:
            self._print(self.summary())
            return True
        print(self.summary(), file=sys.stderr, flush=True)
        if exit_on_failure:
            sys.exit(1)
        return False
