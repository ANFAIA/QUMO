"""Circuit manipulation workouts (QUMO)."""
import qumo_py
from benchpress.config import Configuration
from benchpress.workouts.validation import benchpress_test_validation
from benchpress.workouts.manipulate import WorkoutCircuitManipulate


@benchpress_test_validation
class TestWorkoutCircuitManipulate(WorkoutCircuitManipulate):
    def test_DTC100_twirling(self, benchmark):
        circuit = qumo_py.load_qasm_file(Configuration.get_qasm_dir("dtc") + "dtc_100_cx_12345.qasm")
        assert benchmark(qumo_py.pauli_twirl_2q, circuit, 12345)

    def test_multi_control_decompose(self, benchmark):
        circ = qumo_py.multi_control_circuit(16)

        @benchmark
        def result():
            return qumo_py.translate(circ, ["rx", "ry", "rz", "cz"])

        benchmark.extra_info["gate_count_2q"] = result.count_gate("cz")
        assert result

    def test_QV100_basis_change(self, benchmark):
        circ = qumo_py.load_qasm_file(Configuration.get_qasm_dir("qv") + "qv_N100_12345.qasm")

        @benchmark
        def result():
            return qumo_py.translate(circ, ["sx", "x", "rz", "cz"])

        benchmark.extra_info["gate_count_2q"] = result.count_gate("cz")
        assert result

    def test_random_clifford_decompose(self, benchmark):
        circ = qumo_py.load_qasm_file(Configuration.get_qasm_dir("clifford") + "clifford_20_12345.qasm")

        @benchmark
        def result():
            return qumo_py.translate(circ, ["rz", "sx", "x", "cz"])

        benchmark.extra_info["gate_count_2q"] = result.count_gate("cz")
        benchmark.extra_info["depth_2q"] = result.depth_of("cz")
        assert result
