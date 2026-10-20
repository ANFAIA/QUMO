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


@needs_qumo
def test_dem_text_matches_qumo_decoder():
    """Regression: dem_text() used to export undecomposed hyperedges, so
    PyMatching built from it saw a different graph than QUMO's decoders and
    disagreed on ~0.2% of shots. It now exports the decomposition QUMO uses."""
    import pymatching
    import stim

    import hetero_common as H

    q = H.load_qumo()
    m = q.qec_memory("surface_z", 5, 5, 0.004)
    dem = stim.DetectorErrorModel(m.dem_text())
    for inst in dem.flattened():  # every error is graph-like (pieces of <= 2 detectors)
        if inst.type == "error":
            piece = 0
            for t in inst.targets_copy():
                if t.is_separator():
                    piece = 0
                elif t.is_relative_detector_id():
                    piece += 1
                    assert piece <= 2
    shots = 20000
    dh, _, _, _, _ = m.sample(shots, 11, "cpu", 0)
    nd, bd = dem.num_detectors, (dem.num_detectors + 7) // 8
    raw = np.frombuffer(bytes.fromhex(dh), dtype=np.uint8).reshape(shots, bd)
    dets = np.unpackbits(raw, axis=1, bitorder="little")[:, :nd]
    ph, _, _ = m.decode(dh, shots, 0)
    ours = np.unpackbits(np.frombuffer(bytes.fromhex(ph), dtype=np.uint8).reshape(shots, -1), axis=1,
                         bitorder="little")[:, :1]
    theirs = pymatching.Matching.from_detector_error_model(dem).decode_batch(dets)[:, :1]
    assert int((ours != theirs).any(axis=1).sum()) <= 3  # equal-weight ties only


def test_import_keeps_host_python_environment(tmp_path):
    """Regression: importing qumo_py let the Mojo runtime export
    PYTHONEXECUTABLE / PYTHONPATH / MOJO_PYTHON_LIBRARY for another Python at
    the C level, so child processes started with sys.executable ran it."""
    import os
    import subprocess

    if not HAVE:
        pytest.skip("build/qumo_py.so missing (make py)")
    build = REPO / ("build" if (REPO / "build" / "qumo_py.so").is_file() else "build/gpu")
    code = ("import ctypes, sys, subprocess; sys.path.insert(0, %r); import qumo_py; "
            "libc = ctypes.CDLL(None); libc.getenv.restype = ctypes.c_char_p; "
            "print([libc.getenv(k) for k in (b'PYTHONEXECUTABLE', b'MOJO_PYTHON_LIBRARY')]); "
            "print(libc.getenv(b'PYTHONPATH')); "
            "print(subprocess.run([sys.executable, '-c', 'import sys; print(sys.executable)'], "
            "capture_output=True, text=True).stdout.strip())") % str(build)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONEXECUTABLE", "MOJO_PYTHON_LIBRARY")}
    env["PYTHONPATH"] = str(tmp_path)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).stdout.splitlines()
    assert out[0] == "[None, None]"
    assert out[1] == repr(str(tmp_path).encode())  # the user's PYTHONPATH is kept as it was
    assert out[2] == sys.executable
