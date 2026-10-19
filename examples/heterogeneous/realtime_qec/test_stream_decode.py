"""Tests for realtime_qec/stream_decode.py and the QecMemory binding."""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

HAVE = (REPO / "build" / "qumo_py.so").is_file() or (REPO / "build" / "gpu" / "qumo_py.so").is_file()
needs_qumo = pytest.mark.skipif(not HAVE, reason="build/qumo_py.so missing (make py)")
pytest.importorskip("pymatching")
pytest.importorskip("stim")


@needs_qumo
def test_qec_memory_binding_roundtrip():
    import hetero_common as H

    q = H.load_qumo()
    m = q.qec_memory("surface_z", 3, 3, 0.003)
    info = m.info()
    assert info["num_detectors"] == 24 and info["num_observables"] == 1
    d1, o1, method, _, _ = m.sample(3000, 5, "cpu", 1)
    d2, o2, _, _, _ = m.sample(3000, 5, "cpu", 0)
    assert (d1, o1) == (d2, o2) and method == "cpu:pauli_frame"  # thread count does not change samples
    p1, _, ev1 = m.decode(d1, 3000, 1)
    p0, _, ev0 = m.decode(d1, 3000, 0)
    assert p1 == p0 and ev1 == ev0
    assert len(d1) == 2 * 3 * 3000 and len(p1) == 2 * 3000
    with pytest.raises(Exception) as exc:
        m.decode(d1[:-2], 3000, 1)
    assert "QM-E913" in str(exc.value)
    _, _, _, note, _ = m.sample(10, 1, "gpu", 1)
    g = q.gpu_info()
    assert bool(note) == (not (g["compiled"] and g["available"]))


@needs_qumo
def test_stream_run_passes_and_cleans_up(tmp_path):
    import stream_decode as SD

    rep = SD.run(["--shots", "30000", "--batch", "1000", "--workers", "2", "--quiet", "--out", str(tmp_path)])
    assert rep["_all_passed"], rep["checks"]
    r = rep["results"]
    assert r["agreement"] >= 0.999 and r["batches"] == 30
    assert r["pool_capacity_rounds_per_s"] > r["stream_rounds_per_s"]
    assert not mp.active_children()
    budget = [s for s in rep["stages"] if s["stage"] == "realtime-budget"][0]
    assert budget["emulated"] and "fpga" in budget["device"]
