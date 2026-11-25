"""Noise models and their calibration, below QEC and above the simulators.

Modules:
  characterization  NoiseModel from calibration data (T1, T2, gate times, gate and readout errors),
                    `apply_noise_model`, randomized benchmarking, T1/Ramsey experiments and fits,
                    readout calibration and `mitigate_counts`
  mthree            M3 readout mitigation in the subspace of observed bitstrings

`qumo.noise` imports `qumo.ir`, `qumo.core`, `qumo.device` (for `Target`) and the simulators (`qumo.sim.frame`,
`qumo.sim.statevector`); the algorithms and the execution layer use it, and it imports neither them nor
`qumo.qec` nor the compiler. `qumo.qec.characterization` and `qumo.qec.mthree` re-export it for one release.
Error codes QM-E709 and QM-E710 belong here. See docs/QEC.md (Characterization).
"""
