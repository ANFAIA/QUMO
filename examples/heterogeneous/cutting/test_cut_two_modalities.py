"""Tests for cutting/cut_two_modalities.py (offline; local emulators only)."""

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

import cut_two_modalities as C  # noqa: E402

HAVE = (REPO / "build" / "qumo_py.so").is_file() or (REPO / "build" / "gpu" / "qumo_py.so").is_file()
needs_qumo = pytest.mark.skipif(not HAVE, reason="build/qumo_py.so missing (make py)")


@pytest.fixture(autouse=True)
def hw_home(tmp_path, monkeypatch):
    monkeypatch.setenv("QUMO_HARDWARE_HOME", str(tmp_path / "hw"))


def test_wire_cut_identity_on_random_qubit_states():
    """1/2 sum_P Tr(P rho) Tr(O M(P)) with M from the eigenstate preparations equals Tr(O rho)."""
    rng = np.random.default_rng(0)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1.0, -1.0]).astype(complex)
    paulis = [np.eye(2), X, Y, Z]
    kets = {"0": np.array([1, 0]), "1": np.array([0, 1]), "+": np.array([1, 1]) / 2 ** 0.5,
            "+i": np.array([1, 1j]) / 2 ** 0.5}
    for _ in range(20):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v /= np.linalg.norm(v)
        rho = np.outer(v, v.conj())
        O = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        O = O + O.conj().T
        E1 = np.array([[np.trace(P @ rho).real for P in paulis]])
        F2 = np.array([[np.real(kets[s].conj() @ O @ kets[s]) for s in C.PREPS]])
        assert abs(C.contract(E1, F2, np.array([1.0]), 1) - np.trace(O @ rho).real) < 1e-12


def test_fragments_and_directions():
    n, gates = C.build_circuit(7)
    assert n == 18
    assert C.fragments_for(n, gates, ()) is None  # one connected circuit
    cands = C.candidate_cuts(n, gates)
    good = [c for c in cands if C.fragments_for(n, gates, (c,)) is not None]
    assert good, "some single wire cut splits the circuit"
    fr = C.fragments_for(n, gates, (good[0],))
    assert sorted(fr["qubits"][0] + fr["qubits"][1]).count(good[0][0]) == 2  # the cut wire is in both


@needs_qumo
def test_default_run_passes_and_uses_both_modalities(tmp_path):
    rep = C.run(["--seed", "8", "--quiet", "--out", str(tmp_path), "--ion-shots", "8000", "--sc-shots", "8000"])
    assert rep["_all_passed"], rep["checks"]
    mods = {j["modality"] for j in rep["qpu_jobs"]}
    assert mods == {"trapped_ion", "superconducting"}
    assert all(j["local"] and not j["submitted"] or j["local"] for j in rep["qpu_jobs"])
    assert abs(rep["results"]["noiseless_reconstruction"] - rep["results"]["exact"]) < 1e-9
