"""Research drivers exported from Mojo: VQE, QAOA, Hamiltonians, circuit library."""

import math

import numpy as np
import pytest

import qumo


def test_h2_hamiltonian_and_exact_energy():
    h = qumo.h2_sto3g()
    assert len(h) == 15 and all(isinstance(label, str) and len(label) == 4 for label, _ in h)
    assert all(isinstance(coeff, float) for _, coeff in h)
    e0 = qumo.ground_energy(h)
    assert math.isclose(e0, -1.137306, abs_tol=1e-5)
    assert math.isclose(qumo.ground_energy(qumo.h2_sto3g(2)), e0, abs_tol=1e-6)
    electronic = qumo.ground_energy(qumo.h2_sto3g(include_nuclear_repulsion=False))
    assert electronic < e0 - 0.5


def test_vqe_h2_reaches_the_ground_state():
    h = qumo.h2_sto3g()
    r = qumo.vqe(qumo.h2_ansatz(), h)
    assert math.isclose(r["energy"], qumo.ground_energy(h), abs_tol=1e-6)
    assert r["converged"] and r["optimizer"] == "bfgs" and r["grad_method"] == "adjoint"
    assert isinstance(r["params"], np.ndarray) and isinstance(r["history"], np.ndarray)
    assert list(r["parameters"]) == r["param_names"] == qumo.h2_ansatz().parameter_names
    assert r["history"][-1] <= r["history"][0] + 1e-12 and r["nfev"] >= 1
    # the optimum plugs back into expectation()
    assert math.isclose(qumo.expectation(qumo.h2_ansatz(), h, r["parameters"]), r["energy"], abs_tol=1e-9)


@pytest.mark.parametrize("method", ["adam", "nelder-mead"])
def test_vqe_options(method):
    c = qumo.Circuit(1).ry("theta", 0)
    r = qumo.vqe(c, "Z0", method=method, x0={"theta": 0.4}, maxiter=400)
    assert r["energy"] < -0.99 and r["optimizer"] == method
    assert set(qumo.OPTIMIZERS) >= {method}


def test_vqe_with_own_circuit_and_string_hamiltonian():
    ansatz = qumo.hardware_efficient_ansatz(2, layers=2)
    h = qumo.transverse_field_ising(2, j=1.0, h=0.5)
    r = qumo.vqe(ansatz, h, seed=3)
    assert math.isclose(r["energy"], qumo.ground_energy(h), abs_tol=1e-5)
    r2 = qumo.vqe(qumo.Circuit(1).ry("t", 0), "X0", grad="parameter-shift")
    assert math.isclose(r2["energy"], -1.0, abs_tol=1e-6)


def test_vqe_errors():
    with pytest.raises(qumo.QumoError) as e:
        qumo.vqe(qumo.Circuit(1).h(0), "Z0")
    assert e.value.code == "QM-E101"
    c = qumo.Circuit(1).ry("t", 0)
    with pytest.raises(qumo.QumoError):
        qumo.vqe(c, "Z0", method="simulated-annealing")
    with pytest.raises(qumo.QumoError):
        qumo.vqe(c, "Z0", grad="magic")
    with pytest.raises(qumo.QumoError):
        qumo.vqe(c, "Z0", x0=[0.1, 0.2])


def test_qaoa_maxcut_ring():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    r = qumo.qaoa_maxcut(edges, p=1)
    assert r["max_cut"] == 4.0 and r["best_cut"] == 4.0 and r["best_bitstring"] in ("0101", "1010")
    assert math.isclose(r["approx_ratio"], 0.75, abs_tol=1e-6)
    assert isinstance(r["gammas"], np.ndarray) and r["gammas"].shape == (1,) and r["betas"].shape == (1,)
    assert r["shots"] == 1024 and 0.0 < r["prob_optimal"] <= 1.0
    deeper = qumo.qaoa_maxcut(edges, p=2, restarts=2)
    assert deeper["approx_ratio"] >= r["approx_ratio"] - 1e-6
    weighted = qumo.qaoa_maxcut([(0, 1, 2.0), (1, 2, 1.0)], num_nodes=3)
    assert weighted["max_cut"] == 3.0
    with pytest.raises(TypeError):
        qumo.qaoa_maxcut([(0,)])
    with pytest.raises(qumo.QumoError):
        qumo.qaoa_maxcut([])


def test_hamiltonian_builders():
    assert qumo.transverse_field_ising(2, h=0.5) == [("ZZ", -1.0), ("IX", -0.5), ("XI", -0.5)]
    xxz = dict(qumo.heisenberg_xxz(3, delta=0.5))
    assert xxz["IXX"] == 1.0 and xxz["ZZI"] == 0.5
    assert len(qumo.heisenberg_xxz(3, periodic=True)) == 9
    mc = dict(qumo.maxcut_hamiltonian([(0, 1), (1, 2)]))
    assert mc == {"IZZ": 0.5, "ZZI": 0.5, "III": -1.0}
    assert "III" not in dict(qumo.maxcut_hamiltonian([(0, 1), (1, 2)], include_offset=False))
    hub = qumo.fermi_hubbard([(0, 1)], t=1.0, u=4.0)
    assert all(len(label) == 4 for label, _ in hub)
    # the lists work everywhere an observable does
    assert math.isclose(qumo.expectation(qumo.Circuit(3).x(0), list(mc.items())), -1.0, abs_tol=1e-12)
    assert qumo.ground_energy(qumo.maxcut_hamiltonian([(0, 1), (1, 2), (2, 0)])) == pytest.approx(-2.0)


def test_circuit_library():
    assert np.allclose(np.abs(qumo.statevector(qumo.ghz(3))) ** 2, [0.5, 0, 0, 0, 0, 0, 0, 0.5])
    assert set(qumo.sample(qumo.ghz(4, measure=True), 100)) == {"0000", "1111"}
    assert np.allclose(np.abs(qumo.statevector(qumo.qft(3))), 8 ** -0.5)
    ansatz = qumo.hardware_efficient_ansatz(3, layers=2, entanglement="circular")
    assert ansatz.num_parameters == 18 and ansatz.num_qubits == 3
    assert qumo.qaoa_maxcut_ansatz([(0, 1), (1, 2)], p=2).parameter_names == ["gamma[0]", "gamma[1]", "beta[0]",
                                                                               "beta[1]"]
    assert qumo.uccsd_ansatz(4, 2).num_parameters == 3
    step = qumo.trotter("Z0 Z1 + 0.5 * X0", time=0.3, steps=4, order=2)
    exact = qumo.statevector(qumo.trotter("Z0 Z1 + 0.5 * X0", time=0.3, steps=200, order=2))
    assert abs(np.vdot(exact, qumo.statevector(step))) > 0.9999
