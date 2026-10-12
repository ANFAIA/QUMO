"""Experimentalist toolkit: calibration -> noise model -> RB, T1/T2, readout
mitigation, and a QEC memory experiment under the calibrated noise.

    .venv/bin/mojo run -I . examples/qec_characterization.mojo
"""

from qumo.ir.circuit import Circuit
from qumo.sim.stabilizer import sample_stabilizer
from qumo.qec.codes import surface_code_memory, NoiseParams
from qumo.qec.experiments import run_circuit_experiment
from qumo.qec.characterization import (
    NoiseModel, apply_noise_model, randomized_benchmarking_1q, t1_experiment,
    ramsey_experiment, ReadoutCalibration, mitigate_counts, expectation_z,
)


def main() raises:
    # Calibration data (microseconds): T1, T2, 1q/2q gate errors, readout, durations.
    var model = NoiseModel.uniform(17, 120.0, 90.0, 3e-4, 4e-3, 1.5e-2, 0.035, 0.25, 0.8)
    print(model)

    # Randomized benchmarking with the gate-level calibrated model.
    var lengths: List[Int] = [1, 10, 25, 50, 100, 200, 400]
    var rb = randomized_benchmarking_1q(lengths, 8, 4000, seed=5, model=model.copy())
    print(rb)

    # T1 / Ramsey on qubit 0.
    var delays: List[Float64] = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0]
    print("T1:", t1_experiment(model, delays, 20000, 1))
    print("T2:", ramsey_experiment(model, delays, 20000, 2))

    # Readout mitigation on a noisy Bell measurement.
    var bell = Circuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure(0, 0)
    bell.measure(1, 1)
    var bm = NoiseModel.uniform(2, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0)
    var counts = sample_stabilizer(apply_noise_model(bell, bm), 100000, 3).counts.copy()
    var cal = ReadoutCalibration.from_model(bm)
    var mit = mitigate_counts(counts, cal)
    var zz: List[Int] = [0, 1]
    print("raw P(01)=", Float64(counts.get("01", 0)) / 1e5, " mitigated P(01)=", mit.get("01", 0.0),
          " <ZZ> mitigated =", expectation_z(mit, zz))

    # Surface code under the calibrated (non-uniform, idle-aware) noise.
    var cc = surface_code_memory(3, 3, NoiseParams.noiseless(), "Z")
    cc.circuit = apply_noise_model(cc.circuit, model)
    print(run_circuit_experiment(cc, 500000, "mwpm", 9, 500))
