"""Circuit construction workouts (QUMO).

Note: QUMO has no lazy gate objects; QV and multi-controlled gates are
synthesised eagerly (more work than Qiskit's lazy UnitaryGate / .control()).
"""
import numpy as np

import qumo_py
from benchpress.config import Configuration
from benchpress.workouts.validation import benchpress_test_validation
from benchpress.workouts.build import WorkoutCircuitConstruction
from benchpress.qumo_gym.utils.qumo_utils import output_properties

SEED = 12345


@benchpress_test_validation
class TestWorkoutCircuitConstruction(WorkoutCircuitConstruction):
    def test_QV100_build(self, benchmark):
        @benchmark
        def result():
            return qumo_py.quantum_volume(100, 100, SEED)

        assert result.count_gate("cx") == 15000

    def test_DTC100_set_build(self, benchmark):
        @benchmark
        def result():
            initial = qumo_py.circuit(100)
            dtc = qumo_py.dtc_unitary(100, 0.95, SEED)
            circs = [initial]
            for tt in range(100):
                circs.append(circs[tt].compose(dtc))
            return circs[-1]

        output_properties(result, "rzz", benchmark)
        assert result.count_gate("rzz") == 9900

    def test_multi_control_circuit(self, benchmark):
        @benchmark
        def result():
            return qumo_py.multi_control_circuit(16)

        assert result

    def test_clifford_build(self, benchmark):
        @benchmark
        def result():
            qumo_py.random_clifford_circuit(100, 10 * 100 * 100, SEED)
            return True

        assert result

    def test_param_circSU2_100_build(self, benchmark):
        @benchmark
        def result():
            return qumo_py.efficient_su2(100, 4, "circular")

        assert result.num_parameters() == 1000

    def test_param_circSU2_100_bind(self, benchmark):
        qc = qumo_py.efficient_su2(100, 4, "circular")
        params = list(np.linspace(0, 2 * np.pi, qc.num_parameters()))

        @benchmark
        def result():
            return qc.bind(params)

        assert result.num_parameters() == 0

    def test_QV100_qasm2_import(self, benchmark):
        @benchmark
        def result():
            return qumo_py.load_qasm_file(Configuration.get_qasm_dir("qv") + "qv_N100_12345.qasm")

        output_properties(result, "cx", benchmark)
        ops = result.count_ops()
        assert ops.get("rz", 0) == 120000
        assert ops.get("rx", 0) == 80000
        assert ops.get("cx", 0) == 15000

    def test_bigint_qasm2_import(self, benchmark):
        @benchmark
        def result():
            return qumo_py.load_qasm_file(Configuration.get_qasm_dir("bigint") + "bigint.qasm")

        output_properties(result, "cx", benchmark)
        assert result
