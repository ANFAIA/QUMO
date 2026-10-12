"""QASMBench circuits on Benchpress' abstract topologies (QUMO)."""
import pytest

from benchpress.config import Configuration
from benchpress.workouts.validation import benchpress_test_validation
from benchpress.workouts.abstract_transpile import (
    WorkoutAbstractQasmBenchSmall, WorkoutAbstractQasmBenchMedium, WorkoutAbstractQasmBenchLarge,
)
from benchpress.workouts.abstract_transpile.qasmbench import (
    SMALL_CIRC_TOPO, SMALL_NAMES, MEDIUM_CIRC_TOPO, MEDIUM_NAMES, LARGE_CIRC_TOPO, LARGE_NAMES,
)
from benchpress.qumo_gym.utils.qumo_utils import (
    flexible_target, transpile, qasm_loader, output_properties, validate,
)

LEVEL = Configuration.options.get("qumo", {}).get("optimization_level", 2)


def _run(benchmark, circ_and_topo):
    circuit = qasm_loader(circ_and_topo[0], benchmark)
    target = flexible_target(circuit.num_qubits(), circ_and_topo[1])

    @benchmark
    def result():
        return transpile(circuit, target, LEVEL)

    output_properties(result, target.two_q_gate_type, benchmark)
    assert validate(result, target)


@benchpress_test_validation
class TestWorkoutAbstractQasmBenchSmall(WorkoutAbstractQasmBenchSmall):
    @pytest.mark.parametrize("circ_and_topo", SMALL_CIRC_TOPO, ids=SMALL_NAMES)
    def test_QASMBench_small(self, benchmark, circ_and_topo):
        _run(benchmark, circ_and_topo)


@benchpress_test_validation
class TestWorkoutAbstractQasmBenchMedium(WorkoutAbstractQasmBenchMedium):
    @pytest.mark.parametrize("circ_and_topo", MEDIUM_CIRC_TOPO, ids=MEDIUM_NAMES)
    def test_QASMBench_medium(self, benchmark, circ_and_topo):
        _run(benchmark, circ_and_topo)


@benchpress_test_validation
class TestWorkoutAbstractQasmBenchLarge(WorkoutAbstractQasmBenchLarge):
    @pytest.mark.parametrize("circ_and_topo", LARGE_CIRC_TOPO, ids=LARGE_NAMES)
    def test_QASMBench_large(self, benchmark, circ_and_topo):
        _run(benchmark, circ_and_topo)
