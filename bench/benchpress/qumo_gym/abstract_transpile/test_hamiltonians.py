"""HamLib Hamiltonians on Benchpress' abstract topologies (QUMO)."""
import pytest

from benchpress.config import Configuration
from benchpress.workouts.validation import benchpress_test_validation
from benchpress.workouts.abstract_transpile.hamlib_hamiltonians import (
    HAM_TOPO, HAM_TOPO_NAMES, WorkoutAbstractHamiltonians,
)
from benchpress.qumo_gym.utils.qumo_utils import (
    flexible_target, transpile, input_properties, output_properties, validate, hamiltonian_circuit,
)

LEVEL = Configuration.options.get("qumo", {}).get("optimization_level", 2)


@benchpress_test_validation
class TestWorkoutAbstractHamiltonians(WorkoutAbstractHamiltonians):
    @pytest.mark.parametrize("circ_and_topo", HAM_TOPO, ids=HAM_TOPO_NAMES)
    def test_hamiltonians(self, benchmark, circ_and_topo):
        circuit = hamiltonian_circuit(circ_and_topo[0].pop("ham_hamlib_hamiltonian"))
        input_properties(circuit, benchmark)
        target = flexible_target(circuit.num_qubits(), circ_and_topo[1])

        @benchmark
        def result():
            return transpile(circuit, target, LEVEL)

        benchmark.extra_info.update(circ_and_topo[0])
        output_properties(result, target.two_q_gate_type, benchmark)
        assert validate(result, target)
