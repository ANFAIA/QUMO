"""Device transpilation workouts (QUMO) against the configured Benchpress device."""
import qumo_py
from benchpress.config import Configuration
from benchpress.workouts.validation import benchpress_test_validation
from benchpress.workouts.device_transpile import WorkoutDeviceTranspile100Q
from benchpress.qumo_gym.utils.qumo_utils import (
    device_target, transpile, qasm_loader, input_properties, output_properties, validate,
)

TARGET = device_target()
TWO_Q_GATE = TARGET.two_q_gate_type
LEVEL = Configuration.options.get("qumo", {}).get("optimization_level", 2)


def _run(benchmark, circuit):
    @benchmark
    def result():
        return transpile(circuit, TARGET, LEVEL)

    output_properties(result, TWO_Q_GATE, benchmark)
    assert validate(result, TARGET)


@benchpress_test_validation
class TestWorkoutDeviceTranspile100Q(WorkoutDeviceTranspile100Q):
    def test_QFT_100_transpile(self, benchmark):
        _run(benchmark, qasm_loader(Configuration.get_qasm_dir("qft") + "qft_N100.qasm", benchmark))

    def test_QV_100_transpile(self, benchmark):
        # same QV-100 instance as the QASM import test (pre-synthesised by Benchpress)
        _run(benchmark, qasm_loader(Configuration.get_qasm_dir("qv") + "qv_N100_12345.qasm", benchmark))

    def test_circSU2_89_transpile(self, benchmark):
        c = qumo_py.efficient_su2(89, 3, "circular")
        input_properties(c, benchmark)
        _run(benchmark, c)

    def test_circSU2_100_transpile(self, benchmark):
        c = qumo_py.efficient_su2(100, 3, "circular")
        input_properties(c, benchmark)
        _run(benchmark, c)

    def test_BV_100_transpile(self, benchmark):
        c = qumo_py.bv_all_ones(100)
        input_properties(c, benchmark)
        _run(benchmark, c)

    def test_square_heisenberg_100_transpile(self, benchmark):
        _run(benchmark, qasm_loader(Configuration.get_qasm_dir("square-heisenberg") + "square_heisenberg_N100.qasm", benchmark))

    def test_QAOA_100_transpile(self, benchmark):
        _run(benchmark, qasm_loader(Configuration.get_qasm_dir("qaoa") + "qaoa_barabasi_albert_N100_3reps.qasm", benchmark))

    def test_BVlike_simplification_transpile(self, benchmark):
        c = qumo_py.trivial_bvlike_circuit(100)
        input_properties(c, benchmark)
        _run(benchmark, c)

    def test_clifford_100_transpile(self, benchmark):
        _run(benchmark, qasm_loader(Configuration.get_qasm_dir("clifford") + "clifford_100_12345.qasm", benchmark))
