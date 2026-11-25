"""Compatibility path: the noise model and characterization now live in `qumo/noise/characterization.mojo`.

`from qumo.qec.characterization import NoiseModel, apply_noise_model` keeps working and gives the same
objects. This path is kept for one release; new code imports `qumo.noise.characterization`.
"""

from qumo.noise.characterization import (
    PauliProbs, thermal_pauli_channel, NoiseModel, apply_noise_model, DecayFit, fit_decay,
    fit_exponential_decay, Clifford1Q, clifford_group_1q, RBResult, rb_sequence_circuit,
    survival_probability, randomized_benchmarking_1q, ReadoutCalibration, readout_calibration_circuits,
    mitigate_counts, expectation_z, t1_circuit, ramsey_circuit, CoherenceResult, t1_experiment,
    ramsey_experiment,
)
