"""Compatibility path: simulator selection now lives in `qumo/sim/dispatch.mojo` (CPU methods) and
`qumo/gpu/dispatch.mojo` (the selection that also knows the GPU: `sample_auto`, `sample_on`).

`from qumo.exec.dispatch import sample_auto, sample_on, choose_method` keeps working and gives the
same functions. This path is kept for one release; new code imports `qumo.gpu.dispatch` or
`qumo.sim.dispatch`.
"""

from qumo.sim.dispatch import (
    STABILIZER_MIN_QUBITS, FRAME_MIN_SHOTS, choose_method_cpu, sample_frame, counts_from_measurements,
)
from qumo.gpu.dispatch import (
    gpu_eligible, choose_method, frame_eligible_gpu, gpu_sample_route, sample_frame_gpu, sample_auto, sample_on,
)
