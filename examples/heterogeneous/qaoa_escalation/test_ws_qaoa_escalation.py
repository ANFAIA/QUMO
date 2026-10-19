"""Tests for qaoa_escalation/ws_qaoa_escalation.py (offline; local emulators only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "plugins" / "hardware"))

import ws_qaoa_escalation as W  # noqa: E402

HAVE = (REPO / "build" / "qumo_py.so").is_file() or (REPO / "build" / "gpu" / "qumo_py.so").is_file()
needs_qumo = pytest.mark.skipif(not HAVE, reason="build/qumo_py.so missing (make py)")


@pytest.fixture(autouse=True)
def hw_home(tmp_path, monkeypatch):
    monkeypatch.setenv("QUMO_HARDWARE_HOME", str(tmp_path / "hw"))


def test_gw_bound_and_local_search():
    for seed in (1, 2, 3):
        e = W.random_graph(10, seed)
        opt, _ = W.brute_force(10, e)
        gw = W.goemans_williamson(10, e, seed)
        assert gw["sdp_value"] >= opt - 1e-6 and gw["cut"] <= opt + 1e-9
        b, v = W.local_search(gw["bits"], e)
        assert v >= gw["cut"] - 1e-12
        for i in range(10):  # 1-flip local optimum
            b2 = b.copy()
            b2[i] ^= 1
            assert W.cut_value(b2, e) <= v + 1e-12


@needs_qumo
def test_gradients_match_finite_differences():
    import hetero_common as H

    q = H.load_qumo()
    e = W.random_graph(6, 4)
    th = 2 * np.arcsin(np.sqrt(np.array([0.25, 0.75] * 3)))
    c = q.load_qasm(W.qaoa_qasm(6, e, 2, th))
    obs = W.cost_observable(e)
    x = [0.3, -0.2, 0.4, 0.1]
    v, g, method, _ = q.value_and_gradient(c, obs, x, device="cpu")
    assert method == "cpu:adjoint:fp64"
    for k in range(4):
        xp, xm = list(x), list(x)
        xp[k] += 1e-5
        xm[k] -= 1e-5
        fd = (q.expectation(c.bind(xp), obs) - q.expectation(c.bind(xm), obs)) / 2e-5
        assert abs(fd - g[k]) < 1e-6
    vg, gg, mg, note = q.value_and_gradient(c, obs, x, device="gpu")
    assert abs(vg - v) < 1e-4 and max(abs(a - b) for a, b in zip(gg, g)) < 1e-4
    assert ("gpu" in mg) or note  # GPU ran, or the CPU fallback says why


@needs_qumo
def test_run_passes_and_escalates(tmp_path):
    rep = W.run(["--quiet", "--out", str(tmp_path)])
    assert rep["_all_passed"], rep["checks"]
    r = rep["results"]
    assert r["final_cut"] == pytest.approx(r["max_cut"]) and r["gw"]["cut"] < r["max_cut"]
    assert [j["modality"] for j in rep["qpu_jobs"]] == ["superconducting", "trapped_ion"]
    assert rep["qpu_jobs"][1]["shots"] < rep["qpu_jobs"][0]["shots"]
