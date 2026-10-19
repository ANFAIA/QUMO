"""Tests for sqd/sqd_hubbard.py (offline; local emulators only)."""

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

import sqd_hubbard as S  # noqa: E402

HAVE = (REPO / "build" / "qumo_py.so").is_file() or (REPO / "build" / "gpu" / "qumo_py.so").is_file()
needs_qumo = pytest.mark.skipif(not HAVE, reason="build/qumo_py.so missing (make py)")


@pytest.fixture(autouse=True)
def hw_home(tmp_path, monkeypatch):
    monkeypatch.setenv("QUMO_HARDWARE_HOME", str(tmp_path / "hw"))


def test_recover_half_reaches_target_and_flips_one_direction():
    rng = np.random.default_rng(1)
    occ = np.array([0.9, 0.1, 0.8, 0.2, 0.5, 0.5])
    for bits in range(1 << 6):
        out = S.recover_half(bits, 3, occ, rng)
        assert bin(out).count("1") == 3
        if bin(bits).count("1") > 3:
            assert out & ~bits == 0  # only 1 -> 0 flips
        elif bin(bits).count("1") < 3:
            assert bits & ~out == 0  # only 0 -> 1 flips
        else:
            assert out == bits


def test_recovery_prefers_likely_orbitals():
    occ = np.array([0.99, 0.01, 0.99, 0.01])
    rng = np.random.default_rng(3)
    outs = [S.recover_half(0b0000, 2, occ, rng) for _ in range(200)]
    assert np.mean([o == 0b0101 for o in outs]) > 0.9


@needs_qumo
def test_scipy_reference_matches_qumo_subspace_solver():
    import hetero_common as H

    q = H.load_qumo()
    L = 4
    full = [x | (y << L) for y in S.sector_strings(L, 2) for x in S.sector_strings(L, 2)]
    r = q.subspace_ground_state(S.hubbard_observable(L, 1.0, 4.0), full, 2 * L)
    assert abs(r["energy"] - S.scipy_sector_energy(L, 1.0, 4.0, 2, 2)) < 1e-9
    assert abs(sum(r["occupations"]) - 4.0) < 1e-9 and abs(sum(r["probabilities"]) - 1.0) < 1e-9


@needs_qumo
def test_small_run_passes_all_checks(tmp_path):
    rep = S.run(["--sites", "6", "--samples-per-batch", "10", "--batches", "4", "--shots", "1000",
                 "--seed", "3", "--band", "0.1", "--quiet", "--out", str(tmp_path)])
    assert rep["_all_passed"], rep["checks"]
    stages = {s["stage"]: s for s in rep["stages"]}
    assert stages["qpu-sampling"]["emulated"] and stages["config-recovery"]["device"].startswith("cpu")
    assert all(j["local"] and j["emulated"] for j in rep["qpu_jobs"])
    assert rep["results"]["heft"]["device_of"][:3] == ["sc-qpu"] * 3  # Krylov jobs pinned to the QPU


@needs_qumo
def test_remote_target_needs_allow_remote(tmp_path):
    with pytest.raises(SystemExit) as exc:
        S.run(["--qpu", "ibm:ibm_fez", "--quiet", "--out", str(tmp_path)])
    assert "--allow-remote" in str(exc.value)
