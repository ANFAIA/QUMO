"""Compatibility path: `emulate` now lives in `qumo/noise/emulate.mojo` (it is the target's noise model
plus the cheapest CPU sampler).

`from qumo.exec.emulate import emulate` keeps working and gives the same functions. This path is kept
for one release; new code imports `qumo.noise.emulate`.
"""

from qumo.noise.emulate import active_qubits, compact, noisy_circuit, emulate
