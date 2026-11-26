"""--max-cost on qBraid, offline: the native client calls are replaced by a stub, nothing is sent.

qBraid quotes most devices in credits (GET /jobs/cost-estimate). When it gives no quote the
estimate has unit "unknown" and amount 0, which is not a price: a cap cannot be compared with it.
"""

from __future__ import annotations

import pytest
from conftest import BELL

from qumo_hardware import runner
from qumo_hardware.errors import E_LIMIT, HardwareError
from qumo_hardware.qbraid import QbraidProvider

DEVICE = "qbraid:test:sim:stub"


class StubQbraid(QbraidProvider):
    """QbraidProvider whose native calls answer from memory; `quote` is (available, credits, reason)."""

    def __init__(self, quote):
        super().__init__()
        self.quote = quote
        self.calls: list[str] = []

    def _call(self, fn: str, *args):
        self.calls.append(fn)
        if fn == "qbraid_device":
            return {"qrn": DEVICE, "status": "ONLINE", "qubits": 30, "vendor": "test", "device_type": "SIMULATOR",
                    "paradigm": "gate_model", "run_input_types": ["qasm2"], "per_task": 0.0, "per_shot": 0.0,
                    "per_minute": 0.0, "pricing_model": "fixed"}
        if fn == "qbraid_cost_estimate":
            return self.quote
        if fn == "qbraid_submit":
            return "qbraid:job:stub-1"
        raise AssertionError(f"unexpected native call {fn}")


def test_max_cost_without_a_quote_is_refused():
    prov = StubQbraid((False, 0.0, "no pricing for this device"))
    with pytest.raises(HardwareError) as exc:
        runner.run(BELL, prov, DEVICE, shots=100, confirm=True, max_cost=5.0, wait=False)
    assert exc.value.code == E_LIMIT and "--max-cost" in exc.value.message
    assert "drop --max-cost" in exc.value.hint
    assert "qbraid_submit" not in prov.calls


def test_provider_submit_refuses_a_cap_without_a_quote():
    prov = StubQbraid((False, 0.0, "no pricing for this device"))
    compiled = runner.plan(BELL, prov, DEVICE, shots=100).compiled
    with pytest.raises(HardwareError) as exc:
        prov.submit(compiled, 100, DEVICE, max_cost=5.0, confirm=True)
    assert exc.value.code == E_LIMIT
    assert "qbraid_submit" not in prov.calls


def test_no_cap_accepts_the_unquoted_job():
    prov = StubQbraid((False, 0.0, "no pricing for this device"))
    out = runner.run(BELL, prov, DEVICE, shots=100, confirm=True, wait=False)
    assert out["submitted"] is True and prov.calls[-1] == "qbraid_submit"


def test_cap_is_compared_with_a_quote_in_credits():
    prov = StubQbraid((True, 3.0, ""))
    out = runner.run(BELL, prov, DEVICE, shots=100, confirm=True, max_cost=5.0, wait=False)
    assert out["submitted"] is True
    over = StubQbraid((True, 8.0, ""))
    with pytest.raises(HardwareError) as exc:
        runner.run(BELL, over, DEVICE, shots=100, confirm=True, max_cost=5.0, wait=False)
    assert exc.value.code == E_LIMIT and "qbraid_submit" not in over.calls
