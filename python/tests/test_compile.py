"""qumo.transpile: target strings, explicit coupling map + basis, result object."""

import numpy as np
import pytest

import qumo

LINE4 = [(0, 1), (1, 2), (2, 3)]
SC_CZ = {"rz", "sx", "x", "cz"}


def star() -> qumo.Circuit:
    return qumo.Circuit(4).h(0).cx(0, 1).cx(0, 2).cx(0, 3)


def equivalent(a: qumo.Circuit, result: qumo.TranspileResult) -> bool:
    """Same state up to a global phase, after undoing the layout."""
    psi = qumo.statevector(a)
    out = qumo.statevector(result.circuit)
    n_phys = result.circuit.num_qubits
    n = a.num_qubits
    mapped = np.zeros(2 ** n_phys, dtype=complex)
    for i, amp in enumerate(psi):
        j = 0
        for logical in range(n):
            if (i >> logical) & 1:
                j |= 1 << result.final_layout[logical]
        mapped[j] = amp
    k = int(np.argmax(np.abs(mapped)))
    phase = out[k] / mapped[k]
    return bool(np.allclose(out, phase * mapped, atol=1e-9))


def test_target_string_like_the_cli():
    c = star()
    r = qumo.transpile(c, "grid:2x2/sc_cz")
    assert isinstance(r, qumo.TranspileResult) and isinstance(r.circuit, qumo.Circuit)
    assert set(r.circuit.count_ops()) <= SC_CZ
    edges = set(qumo.target_info("grid:2x2/sc_cz")["edges"])
    assert all((min(a, b), max(a, b)) in edges for a, b in r.circuit.two_qubit_edges())
    assert sorted(r.initial_layout) == [0, 1, 2, 3] and sorted(r.final_layout) == [0, 1, 2, 3]
    assert equivalent(c, r)
    assert r.report["target"] == "grid_2x2" and r.report["basis"] == "rz,sx,x,cz"
    assert len(c) == 4  # the input is not modified


@pytest.mark.parametrize("spec", ["line:4", "ring:5/sc_cx", "heavy_hex:3/rz,sx,x,ecr", "trapped_ion:4",
                                  "all_to_all:4/generic", "grid:2x3"])
def test_more_target_strings(spec):
    r = qumo.transpile(star(), spec)
    info = qumo.target_info(spec)
    assert r.circuit.num_qubits == info["num_qubits"]
    if info["basis"]:
        assert set(r.circuit.count_ops()) <= set(info["basis"])
    assert equivalent(star(), r)


def test_explicit_coupling_map_and_basis():
    c = star()
    by_name = qumo.transpile(c, coupling_map=LINE4, basis="sc_cz")
    by_list = qumo.transpile(c, coupling_map=LINE4, basis=["rz", "sx", "x", "cz"])
    assert by_name.circuit == by_list.circuit
    assert set(by_name.circuit.count_ops()) <= SC_CZ
    assert all(abs(a - b) == 1 for a, b in by_name.circuit.two_qubit_edges())
    assert equivalent(c, by_name)
    wide = qumo.transpile(c, coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], basis="sc_cx")
    assert wide.circuit.num_qubits == 6
    assert qumo.transpile(c, coupling_map=LINE4, basis="sc_cz", num_qubits=4).circuit.num_qubits == 4
    only_basis = qumo.transpile(c, basis="clifford_t")
    assert set(only_basis.circuit.count_ops()) <= {"h", "s", "sdg", "t", "tdg", "x", "y", "z", "cx"}
    plain = qumo.transpile(c)  # all-to-all, any gate
    assert plain.circuit.num_qubits == 4 and equivalent(c, plain)


def test_basis_set_name_is_not_iterated_per_character(q):
    """Regression: basis='sc_cz' was read as the gates 's', 'c', '_', 'c', 'z'."""
    raw = star().native
    by_name = q.transpile(raw, 4, LINE4, "sc_cz", 2, 7)
    by_list = q.transpile(raw, 4, LINE4, ["rz", "sx", "x", "cz"], 2, 7)
    assert by_name[0].to_qasm3() == by_list[0].to_qasm3()
    assert set(by_name[0].count_ops()) <= SC_CZ
    assert set(q.translate(raw, "sc_cz").count_ops()) <= SC_CZ
    assert q.transpile(raw, 4, LINE4, "rz,sx,x,cz", 2, 7)[0].to_qasm3() == by_list[0].to_qasm3()
    with pytest.raises(Exception, match="QM-E6"):
        q.transpile(raw, 4, LINE4, "not_a_basis", 2, 7)
    with pytest.raises(qumo.QumoError) as e:
        qumo.transpile(star(), coupling_map=LINE4, basis="not_a_basis")
    assert e.value.code.startswith("QM-E6")


def test_result_object():
    r = qumo.transpile(star(), "line:4/sc_cz", level=3, seed=11)
    m = r.metrics
    assert m["gates"][0] == 4 and m["gates"][1] == r.circuit.size()
    assert m["two_qubit"][0] == 3 and m["two_qubit"][1] == r.circuit.num_2q_gates()
    assert m["depth"][1] == r.circuit.depth()
    assert 0.0 < m["est_fidelity"] <= 1.0 and m["compile_ms"] > 0
    assert isinstance(r.report["passes"], list) and r.report["pipeline"]
    circuit, initial, final = r  # unpacks like the low-level tuple
    assert circuit is r.circuit and initial == r.initial_layout and final == r.final_layout
    assert "TranspileResult" in repr(r) and "2q 3->" in repr(r)
    again = qumo.transpile(star(), "line:4/sc_cz", level=3, seed=11)
    assert again.circuit == r.circuit  # deterministic for a seed


def test_levels_and_measurements_survive(bell):
    for level in range(4):
        r = qumo.transpile(bell, "line:2/sc_cz", level=level)
        assert r.circuit.count_ops()["measure"] == 2
        assert set(qumo.sample(r.circuit, 300)) == {"00", "11"}


def test_parametric_circuits_compile():
    c = qumo.Circuit(2).ry("theta", 0).cx(0, 1)
    r = qumo.transpile(c, "line:2/sc_cz")
    assert r.circuit.parameter_names == ["theta"]
    assert equivalent(c.bind(theta=0.4), qumo.TranspileResult(r.circuit.bind(theta=0.4), r.initial_layout,
                                                              r.final_layout, "{}"))


def test_errors():
    with pytest.raises(qumo.QumoError) as e:
        qumo.transpile(star(), "hexagon:4")
    assert e.value.code == "QM-E611" and "heavy_hex" in e.value.hint
    with pytest.raises(qumo.QumoError):
        qumo.transpile(star(), "line:4", basis="sc_cz")  # two ways at once
    with pytest.raises(TypeError):
        qumo.transpile(star(), 4)
    with pytest.raises(TypeError):
        qumo.transpile(star(), coupling_map=[(0, 1, 2)])
    with pytest.raises(TypeError):
        qumo.transpile(star(), basis=["rz", 3])
    with pytest.raises(qumo.QumoError):
        qumo.transpile(star(), "line:2")  # too few qubits
    with pytest.raises(TypeError):
        qumo.transpile(star(), "line:4", level="2")
