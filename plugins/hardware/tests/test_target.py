"""Neutral TargetSpec extraction (IBM fake backends, Quantinuum devices, local topologies)."""

from __future__ import annotations

import pytest
from conftest import needs_ibm

from qumo_hardware.base import TargetSpec
from qumo_hardware.errors import E_UNKNOWN_BACKEND, HardwareError
from qumo_hardware.local import parse_sim_backend
from qumo_hardware.quantinuum import QuantinuumProvider


@needs_ibm
@pytest.mark.parametrize("name,nq,twoq", [
    ("fake_torino", 133, "cz"),
    ("fake_fez", 156, "cz"),
    ("fake_kingston", 156, "cz"),
    ("fake_sherbrooke", 127, "ecr"),
    ("fake_brisbane", 127, "ecr"),
])
def test_ibm_fake_targets(name, nq, twoq):
    from qumo_hardware.ibm import IBMProvider

    spec = IBMProvider().target(name)
    assert spec.num_qubits == nq
    assert spec.basis[:3] == ["rz", "sx", "x"] and twoq in spec.basis
    assert not spec.all_to_all and spec.local
    assert all(a < b for a, b in spec.edges) and len(set(spec.edges)) == len(spec.edges)
    assert all(0 <= a < nq and 0 <= b < nq for a, b in spec.edges)
    for key in ("1q", "2q", "readout"):
        assert 0 < spec.errors[key] < 0.5
        assert spec.durations[key] > 0
    assert spec.max_shots >= 10_000
    assert spec.extras["default_rep_delay"] > 0
    assert len(spec.readout_errors) == nq
    # Every coupling of the Qiskit target is present (as an undirected pair).
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    t = FakeProviderForBackendV2().backend(name).target
    pairs = {tuple(sorted(q)) for q in t[twoq].keys()}
    assert pairs == set(spec.edges)
    # Round trip through JSON-able dict.
    again = TargetSpec.from_dict(spec.to_dict())
    assert again.edges == spec.edges and again.basis == spec.basis


@needs_ibm
def test_ibm_directionality():
    from qumo_hardware.ibm import IBMProvider

    heron = IBMProvider().target("fake_torino")
    eagle = IBMProvider().target("fake_sherbrooke")
    assert heron.directed_edges is None  # CZ is symmetric
    assert eagle.directed_edges is not None  # ECR has a native direction
    assert len(eagle.directed_edges) == len(eagle.edges)


@needs_ibm
def test_ibm_fractional_gates_opt_in():
    from qiskit_ibm_runtime.fake_provider import FakeFractionalBackend

    from qumo_hardware.ibm import target_from_qiskit

    b = FakeFractionalBackend()
    plain = target_from_qiskit(b)
    frac = target_from_qiskit(b, use_fractional_gates=True)
    assert "rzz" not in plain.basis
    assert "rzz" in frac.basis and "rx" in frac.basis
    assert frac.extras["fractional_gates_available"]


@needs_ibm
def test_ibm_unknown_fake_backend():
    from qumo_hardware.ibm import IBMProvider

    with pytest.raises(HardwareError) as ei:
        IBMProvider().target("fake_does_not_exist")
    assert ei.value.code == E_UNKNOWN_BACKEND


@pytest.mark.parametrize("name,nq,family", [
    ("H2-1", 56, "H2"), ("H2-1E", 56, "H2"), ("H2-1SC", 56, "H2"),
    ("Helios-1", 98, "Helios"), ("Helios-1E", 98, "Helios"), ("selene", 98, "Helios"),
])
def test_quantinuum_targets(name, nq, family):
    spec = QuantinuumProvider().target(name)
    assert spec.all_to_all and spec.edges == [] and spec.qumo_edges() == []
    assert spec.basis == ["rz", "rx", "ry", "rzz"]
    assert spec.num_qubits == nq
    assert spec.extras["family"] == family
    assert spec.max_shots == 10_000


def test_quantinuum_names():
    p = QuantinuumProvider()
    assert p.is_local("H2-1LE") and p.is_local("selene")
    assert not p.is_local("H2-1") and not p.is_local("Helios-1E")
    assert p.target("H2-3E").extras["kind"] == "cloud-emulator"  # inferred from the naming scheme
    with pytest.raises(HardwareError) as ei:
        p.target("ionq_aria")
    assert ei.value.code == E_UNKNOWN_BACKEND


def test_local_topologies():
    s = parse_sim_backend("qumo-sim:line:5/sc_cz")
    assert s.num_qubits == 5 and s.edges == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert s.basis == ["rz", "sx", "x", "cz"]
    g = parse_sim_backend("qumo-sim:grid:2x3/rz,ry,cz")
    assert g.num_qubits == 6 and len(g.edges) == 7 and g.basis == ["rz", "ry", "cz"]
    r = parse_sim_backend("qumo-sim:ring:4")
    assert (0, 3) in r.edges and r.basis == []
    a = parse_sim_backend("qumo-sim")
    assert a.all_to_all
    with pytest.raises(HardwareError):
        parse_sim_backend("qumo-sim:moebius:7")
