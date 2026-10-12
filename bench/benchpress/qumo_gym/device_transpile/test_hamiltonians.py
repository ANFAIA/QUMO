"""HamLib Hamiltonians transpiled to the configured device (QUMO)."""
import json

import pytest
from qiskit.quantum_info import SparsePauliOp  # only to read the shared HamLib records

from benchpress.config import Configuration
from benchpress.workouts.validation import benchpress_test_validation
from benchpress.workouts.device_transpile import WorkoutDeviceHamlibHamiltonians
from benchpress.qumo_gym.utils.qumo_utils import (
    device_target, transpile, input_properties, output_properties, validate, hamiltonian_circuit,
)

TARGET = device_target()
TWO_Q_GATE = TARGET.two_q_gate_type
LEVEL = Configuration.options.get("qumo", {}).get("optimization_level", 2)


def pytest_generate_tests(metafunc):
    directory = Configuration.get_hamiltonian_dir("hamlib")
    ham_records = json.load(open(directory + "100_representative.json", "r"))
    for h in ham_records:
        terms = h.pop("ham_hamlib_hamiltonian_terms")
        coefficients = h.pop("ham_hamlib_hamiltonian_coefficients")
        h["ham_hamlib_hamiltonian"] = SparsePauliOp(terms, coefficients)
    metafunc.parametrize("hamiltonian_info", ham_records, ids=lambda x: "ham_" + x["ham_instance"][1:-1])


@benchpress_test_validation
class TestWorkoutDeviceHamlibHamiltonians(WorkoutDeviceHamlibHamiltonians):
    def test_hamlib_hamiltonians_transpile(self, benchmark, hamiltonian_info):
        if hamiltonian_info["ham_qubits"] > TARGET.num_qubits:
            pytest.skip("Circuit too large for given backend.")
        circuit = hamiltonian_circuit(hamiltonian_info.pop("ham_hamlib_hamiltonian"))
        input_properties(circuit, benchmark)

        @benchmark
        def result():
            return transpile(circuit, TARGET, LEVEL)

        benchmark.extra_info.update(hamiltonian_info)
        output_properties(result, TWO_Q_GATE, benchmark)
        assert validate(result, TARGET)
