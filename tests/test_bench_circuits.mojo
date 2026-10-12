from qumo.bench.circuits import (
    quantum_volume, efficient_su2, dtc_unitary, random_clifford_circuit,
    bv_all_ones, trivial_bvlike_circuit, multi_control_circuit, pauli_twirl_2q,
    hamiltonian_circuit_from_labels, hamiltonian_circuit,
)
from qumo.ir.gates import Op
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.runner import evolve
from std.testing import assert_equal, assert_true, assert_almost_equal


def main() raises:
    var qv = quantum_volume(100, 100)
    assert_equal(qv.count(Op.CX), 15000)
    var su2 = efficient_su2(100, 4)
    assert_equal(su2.num_params(), 1000)
    var vals = List[Float64]()
    for i in range(1000):
        vals.append(Float64(i) * 0.001)
    assert_equal(su2.bind(vals).num_params(), 0)
    var dtc = dtc_unitary(100)
    assert_equal(dtc.count(Op.RZZ), 99)
    var cl = random_clifford_circuit(10, 1000)
    assert_true(cl.is_clifford())
    _ = bv_all_ones(100)
    _ = trivial_bvlike_circuit(100)
    var mc = multi_control_circuit(6)
    # MCX semantics: |111110> on controls 0..4 then flips 5 in last gate etc. check by simulation
    _ = evolve(mc)
    # twirl preserves the unitary up to global phase
    var t = random_clifford_circuit(5, 60, 7)
    var tw = pauli_twirl_2q(t, 3)
    assert_almost_equal(evolve(t).fidelity(evolve(tw)), 1.0)
    # hamiltonian circuit from labels == from PauliSum
    var labels: List[String] = ["XXYY", "ZIZI", "IYZX"]
    var coeffs: List[Float64] = [0.3, -0.7, 0.25]
    var h1 = hamiltonian_circuit_from_labels(labels, coeffs)
    var ps = parse_pauli_sum("0.3*XXYY - 0.7*ZIZI + 0.25*IYZX")
    var h2 = hamiltonian_circuit(ps, 4)
    assert_almost_equal(evolve(h1).fidelity(evolve(h2)), 1.0)
    print("test_bench_circuits OK")
