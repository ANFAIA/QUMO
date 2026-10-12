"""QUMO compilation for provider targets: ISA validity and semantic equivalence."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import BELL, EXAMPLES as EXAMPLES_DIR, RANDOMISH, ROUTED, ghz, needs_ibm, needs_pytket, x_on

from qumo_hardware.compile import compile_for_target, select_region
from qumo_hardware.errors import E_COMPILE, HardwareError
from qumo_hardware.local import parse_sim_backend

pytest.importorskip("qiskit")

CIRCUITS = {"bell": BELL, "ghz4": ghz(4), "randomish": RANDOMISH, "routed": ROUTED}
NON_UNITARY = {"measure", "barrier", "reset", "delay"}


def fidelity_up_to_layout(compiled, physical=None, seed: int = 7) -> float:
    """|<expected|actual>|^2 for a random product input, through initial/final layouts."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    from qumo_hardware.qasm_bridge import to_qiskit

    logical = to_qiskit(compiled.source_qasm).remove_final_measurements(inplace=False)
    phys = physical if physical is not None else to_qiskit(compiled.qasm)
    n = compiled.num_logical
    used = set(compiled.initial_layout) | set(compiled.final_layout)
    for inst in phys.data:
        if inst.operation.name not in NON_UNITARY:
            used.update(phys.find_bit(q).index for q in inst.qubits)
    order = sorted(used)
    pos = {p: i for i, p in enumerate(order)}
    m = len(order)
    assert m <= 14, "test circuit touches too many physical qubits"
    body = QuantumCircuit(m)
    for inst in phys.data:
        if inst.operation.name in NON_UNITARY:
            continue
        body.append(inst.operation, [pos[phys.find_bit(q).index] for q in inst.qubits])
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(0, np.pi, n), rng.uniform(0, 2 * np.pi, n)
    ref = QuantumCircuit(n)
    for i in range(n):
        ref.ry(a[i], i)
        ref.rz(b[i], i)
    ref.compose(logical, inplace=True)
    psi = Statevector(ref).data
    prep = QuantumCircuit(m)
    for i in range(n):
        prep.ry(a[i], pos[compiled.initial_layout[i]])
        prep.rz(b[i], pos[compiled.initial_layout[i]])
    prep.compose(body, inplace=True)
    phi = Statevector(prep).data
    exp = np.zeros(2**m, dtype=complex)
    for idx in range(2**n):
        p = 0
        for i in range(n):
            if (idx >> i) & 1:
                p |= 1 << pos[compiled.final_layout[i]]
        exp[p] = psi[idx]
    return float(abs(np.vdot(exp, phi)) ** 2)


LOCAL_TARGETS = ["qumo-sim:line:5/sc_cz", "qumo-sim:grid:2x3/sc_iswap", "qumo-sim:ring:5/ion_ms",
                 "qumo-sim:line:5/sc_ecr", "qumo-sim:all:5/ion_zz"]


@pytest.mark.parametrize("target", LOCAL_TARGETS)
@pytest.mark.parametrize("circuit", sorted(CIRCUITS))
def test_equivalence_local_topologies(target, circuit):
    spec = parse_sim_backend(target)
    if spec.num_qubits < 5 and circuit == "routed":
        pytest.skip("too small")
    c = compile_for_target(CIRCUITS[circuit], spec, level=2, seed=3)
    assert set(c.ops) <= set(spec.basis) | NON_UNITARY
    assert fidelity_up_to_layout(c) > 1 - 1e-6


@needs_ibm
@pytest.mark.parametrize("backend", ["fake_torino", "fake_sherbrooke", "fake_fez"])
@pytest.mark.parametrize("circuit", sorted(CIRCUITS))
def test_ibm_isa_and_equivalence(backend, circuit):
    from qumo_hardware.ibm import IBMProvider

    prov = IBMProvider()
    spec = prov.target(backend)
    c = compile_for_target(CIRCUITS[circuit], spec, level=2, seed=1)
    assert c.num_qubits == spec.num_qubits  # ISA circuits span the whole device
    isa, report = prov.prepare(c, backend)  # runs is_isa_circuit + GatesInBasis, raises if invalid
    assert prov.validate(isa, backend)["isa"]
    # Qiskit's own check, independently of the plugin:
    from qiskit_ibm_runtime.utils.utils import is_isa_circuit

    assert is_isa_circuit(isa, prov.backend(backend).target) == ""
    # Semantics are preserved by QUMO and by the (direction) finishing passes.
    assert fidelity_up_to_layout(c) > 1 - 1e-6
    assert fidelity_up_to_layout(c, physical=isa) > 1 - 1e-6
    if backend == "fake_sherbrooke" and c.two_qubit_gates:
        assert spec.directed_edges is not None


@needs_ibm
def test_ibm_fractional_rzz_angles_folded():
    from qumo_hardware.ibm import IBMProvider

    prov = IBMProvider(use_fractional_gates=True)
    spec = prov.target("fake_fractional")
    assert "rzz" in spec.basis
    qaoa = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0]; h q[1]; h q[2];
rzz(2.9) q[0],q[1];
rzz(-0.7) q[1],q[2];
rzz(1.9) q[0],q[2];
rx(0.8) q[0]; rx(0.8) q[1]; rx(0.8) q[2];
measure q -> c;
"""
    c = compile_for_target(qaoa, spec, level=2, seed=1)
    isa, report = prov.prepare(c, "fake_fractional")
    angles = [float(i.operation.params[0]) for i in isa.data if i.operation.name == "rzz"]
    assert all(0 <= a <= np.pi / 2 + 1e-9 for a in angles)
    assert fidelity_up_to_layout(c, physical=isa) > 1 - 1e-6


@needs_pytket
@pytest.mark.parametrize("circuit", sorted(CIRCUITS))
def test_quantinuum_equivalence(circuit):
    from qumo_hardware.quantinuum import QuantinuumProvider

    spec = QuantinuumProvider().target("H2-1")
    c = compile_for_target(CIRCUITS[circuit], spec, level=2, seed=1)
    assert set(c.ops) <= {"rz", "rx", "ry", "rzz"} | NON_UNITARY
    assert c.num_qubits == c.num_logical  # all-to-all: no ancillas
    assert fidelity_up_to_layout(c) > 1 - 1e-6


def test_measure_all_added():
    spec = parse_sim_backend("qumo-sim:line:3/sc_cz")
    c = compile_for_target(x_on(3, 1, measure=False), spec)
    assert c.auto_measured and c.num_clbits == 3
    assert c.logical_measure_map == {0: 0, 1: 1, 2: 2}


def test_too_many_qubits():
    spec = parse_sim_backend("qumo-sim:line:3")
    with pytest.raises(HardwareError) as ei:
        compile_for_target(ghz(4), spec)
    assert ei.value.code == E_COMPILE


def test_parse_error_is_coded():
    with pytest.raises(HardwareError) as ei:
        compile_for_target("OPENQASM 2.0; qreg q[2]; cnto q[0],q[1];", parse_sim_backend("qumo-sim"))
    assert ei.value.code == E_COMPILE


@needs_ibm
def test_region_selection_is_connected_and_used():
    from qumo_hardware.ibm import IBMProvider

    spec = IBMProvider().target("fake_torino")
    region = select_region(spec, 7)
    assert region is not None and len(region) == 7
    adj = {q: set() for q in region}
    for a, b in spec.edges:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    seen, todo = {region[0]}, [region[0]]
    while todo:
        for nb in adj[todo.pop()]:
            if nb not in seen:
                seen.add(nb)
                todo.append(nb)
    assert seen == set(region)  # connected
    c = compile_for_target(ghz(4), spec)
    assert c.region is not None and set(c.active_qubits) <= set(c.region)
    whole = compile_for_target(ghz(4), spec, region="none")
    assert whole.region is None
    # The selected region avoids the worst readout qubits.
    ro = spec.readout_errors
    assert max(ro[q] for q in c.region) <= sorted(x for x in ro if x is not None)[int(0.9 * len(ro))]


VQE = """OPENQASM 3.0;
include "stdgates.inc";
input float a;
input float b;
qubit[2] q;
bit[2] c;
ry(a) q[0];
ry(b) q[1];
cx q[0], q[1];
c = measure q;
"""


def test_parameter_binding_list_and_dict():
    from qumo_hardware import runner

    spec = parse_sim_backend("qumo-sim:line:2/sc_cz")
    with pytest.raises(HardwareError) as ei:
        compile_for_target(VQE, spec)
    assert ei.value.code == E_COMPILE and "--params" in ei.value.hint
    by_list = compile_for_target(VQE, spec, params=[3.141592653589793, 0.0])
    by_name = compile_for_target(VQE, spec, params={"b": 0.0, "a": 3.141592653589793})
    assert by_list.qasm == by_name.qasm
    out = runner.run(VQE, "local", "qumo-sim:line:2/sc_cz", shots=20, params=[3.141592653589793, 0.0])
    assert out["counts"] == {"11": 20}  # ry(pi) on q0 then cx -> |11>
    with pytest.raises(HardwareError):
        compile_for_target(VQE, spec, params=[0.1])


def test_simulation_only_instructions_refused():
    noisy = (EXAMPLES_DIR / "noisy_bell.qasm").read_text()
    with pytest.raises(HardwareError) as ei:
        compile_for_target(noisy, parse_sim_backend("qumo-sim"))
    assert ei.value.code == E_COMPILE and "depolarize" in ei.value.message


def test_register_conditioned_dynamic_circuit_runs_locally():
    from qumo_hardware import runner

    dyn = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
x q[0];
measure q[0] -> c[0];
if(c==1) x q[1];
measure q[1] -> c[1];
"""
    out = runner.run(dyn, "local", "qumo-sim:line:2/sc_cz", shots=10)
    assert out["counts"] == {"11": 10}


def test_single_bit_condition_is_a_coded_limitation():
    tele = (EXAMPLES_DIR / "teleport.qasm").read_text()
    with pytest.raises(HardwareError) as ei:
        compile_for_target(tele, parse_sim_backend("qumo-sim"))
    assert ei.value.code == E_COMPILE and "single-bit conditions" in ei.value.hint
