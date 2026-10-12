"""Feynman circuits transpiled to the configured device (QUMO)."""
import os

import pytest

from benchpress.config import Configuration
from benchpress.workouts.validation import benchpress_test_validation
from benchpress.workouts.device_transpile import WorkoutDeviceFeynman
from benchpress.qumo_gym.utils.qumo_utils import (
    device_target, transpile, qasm_loader, output_properties, validate,
)

TARGET = device_target()
TWO_Q_GATE = TARGET.two_q_gate_type
LEVEL = Configuration.options.get("qumo", {}).get("optimization_level", 2)


def pytest_generate_tests(metafunc):
    directory = Configuration.get_qasm_dir("feynman")
    metafunc.parametrize("filename", [x for x in os.listdir(directory) if x.endswith(".qasm")])


@benchpress_test_validation
class TestWorkoutDeviceFeynman(WorkoutDeviceFeynman):
    def test_feynman_transpile(self, benchmark, filename):
        circuit = qasm_loader(f"{Configuration.get_qasm_dir('feynman')}{filename}", benchmark)
        if circuit.num_qubits() > TARGET.num_qubits:
            pytest.skip("Circuit too large for given backend.")

        @benchmark
        def result():
            return transpile(circuit, TARGET, LEVEL)

        output_properties(result, TWO_Q_GATE, benchmark)
        assert validate(result, TARGET)
