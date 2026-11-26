"""Compatibility path: resource estimation now lives in `qumo/compiler/resources.mojo` (static analysis
of a circuit; it imports only `qumo.ir`).

`from qumo.exec.resources import estimate_resources` keeps working and gives the same objects. This
path is kept for one release; new code imports `qumo.compiler.resources`.
"""

from qumo.compiler.resources import (
    SynthesisModel, t_cost, ResourceEstimate, format_bytes, estimate_resources, HardwareProfile,
    HardwareEstimate, estimate_hardware, FTEstimate, estimate_fault_tolerant,
)
