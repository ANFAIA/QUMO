"""Results come back in the user's logical clbit order, whatever layout/routing did."""

from __future__ import annotations

import pytest
from conftest import (
    PERMUTED_EXPECTED,
    PERMUTED_MEASURE,
    ROUTED,
    ROUTED_EXPECTED,
    ghz,
    needs_guppy,
    needs_ibm,
    needs_pecos,
    x_on,
)

from qumo_hardware import runner
from qumo_hardware.compile import key_from_bits, physical_counts_to_clbits, reorder_counts
from qumo_hardware.local import LocalEmulatorProvider


def test_key_from_bits():
    assert key_from_bits([1, 0, 0]) == "001"
    assert key_from_bits([0, 1, 1, 0]) == "0110"


def test_reorder_counts_variants():
    # Qiskit-style keys (rightmost = order[0]).
    assert reorder_counts({"01": 5}, [0, 1], 2) == {"01": 5}
    # pytket tuples / Selene strings: leftmost = first listed clbit.
    assert reorder_counts({(1, 0, 0): 3}, [0, 1, 2], 3, leftmost_is_first=True) == {"001": 3}
    assert reorder_counts({"100": 3}, [0, 1, 2], 3, leftmost_is_first=True) == {"001": 3}
    # Source position i holds clbit order[i]; merging duplicates.
    assert reorder_counts({"10": 2, "01": 1}, [1, 0], 2) == {"01": 2, "10": 1}


def test_physical_counts_to_clbits():
    meta = {"measure_map": {"0": 7, "1": 3, "2": 5}, "num_clbits": 3}
    # Braket style: measured_qubits = [3, 5, 7], key position i (from the left) = qubit measured_qubits[i]
    counts = {"101": 10, "010": 4}
    out = physical_counts_to_clbits(counts, [3, 5, 7], meta)
    # "101": q3=1 -> c1, q5=0 -> c2, q7=1 -> c0  => c = [1,1,0] -> "011"
    assert out == {"011": 10, "100": 4}


LOCAL = ["qumo-sim:line:5/sc_cz", "qumo-sim:ring:5/sc_ecr", "qumo-sim:grid:2x3/rz,ry,cz", "qumo-sim:all:5/ion_zz"]


@pytest.mark.parametrize("backend", LOCAL)
@pytest.mark.parametrize("k", range(4))
def test_single_x_lands_on_its_clbit(backend, k):
    out = runner.run(x_on(4, k), "local", backend, shots=64, seed=k)
    assert out["counts"] == {"0" * (3 - k) + "1" + "0" * k: 64}


@pytest.mark.parametrize("backend", LOCAL)
def test_routed_circuit_and_permuted_measurements(backend):
    out = runner.run(ROUTED, "local", backend, shots=32)
    assert out["counts"] == {ROUTED_EXPECTED: 32}
    out = runner.run(PERMUTED_MEASURE, "local", backend, shots=32)
    assert out["counts"] == {PERMUTED_EXPECTED: 32}


def test_auto_measure_logical_order():
    out = runner.run(x_on(3, 1, measure=False), "local", "qumo-sim:line:3/sc_cz", shots=16)
    assert out["counts"] == {"010": 16}
    assert out["compiled"]["auto_measured"]


@pytest.mark.parametrize("backend", ["qumo-sim:line:6/sc_cz", "qumo-sim:grid:2x3/sc_cz"])
def test_ghz_two_outcomes(backend):
    out = runner.run(ghz(6), "local", backend, shots=200, seed=5)
    assert set(out["counts"]) == {"000000", "111111"}


@needs_ibm
def test_ideal_simulation_of_ibm_isa_payload():
    """Exact IBM TargetSpec (133 qubits, routed, region-selected), simulated ideally by QUMO."""
    out = runner.run(ROUTED, "local", "qumo-sim:ibm:fake_torino", shots=50)
    assert out["counts"] == {ROUTED_EXPECTED: 50}
    assert out["compiled"]["num_physical"] == 133


@needs_ibm
def test_ibm_fake_backend_noisy_bit_order():
    """Same code path as real IBM hardware (client-side Sampler), Aer noise model."""
    out = runner.run(x_on(3, 2), "ibm", "fake_torino", shots=400, seed=11)
    counts = out["counts"]
    assert sum(counts.values()) == 400
    assert max(counts, key=counts.get) == "100"
    assert counts["100"] / 400 > 0.8
    assert out["submitted"] and out["job_id"].startswith("ibm-local-")
    # Persisted: result can be read back from a new "process".
    again = runner.job_result(out["job_id"])
    assert again["counts"] == counts


@needs_pecos
@pytest.mark.slow
def test_quantinuum_local_emulator_bit_order():
    out = runner.run(PERMUTED_MEASURE, "quantinuum", "H2-1LE", shots=20, seed=3)
    assert out["counts"] == {PERMUTED_EXPECTED: 20}


@needs_guppy
@pytest.mark.slow
def test_selene_local_emulator_bit_order():
    out = runner.run(ROUTED, "local", "selene", shots=10, seed=3)
    assert out["counts"] == {ROUTED_EXPECTED: 10}


def test_local_provider_lists_backends():
    names = [b.name for b in LocalEmulatorProvider().list_backends()]
    assert "qumo-sim" in names and "H2-1LE" in names and "selene" in names
