"""qumo_hardware: run QUMO-compiled circuits on quantum hardware.

Providers implement one neutral interface (`qumo_hardware.base.Provider`):

* `ibm`        IBM Quantum (qiskit-ibm-runtime client-side Sampler)
* `quantinuum` Quantinuum Nexus (qnexus; H2 pytket circuits, Helios HUGR)
* `local`      offline emulation (QUMO simulator, IBM fake backends + Aer,
               Quantinuum local emulators)

Quick start (dry-run by default, nothing is submitted without confirm=True):

    from qumo_hardware import runner
    plan = runner.run(open("bell.qasm").read(), "ibm", "ibm_torino", shots=1000)  # raises QM-E957 with the plan
    out = runner.run(qasm, "local", "qumo-sim:ibm:fake_torino", shots=1000)          # runs locally

See docs/HARDWARE_PLUGINS.md.
"""

from .base import BackendInfo, CostEstimate, JobHandle, Provider, TargetSpec
from .compile import CompiledCircuit, compile_for_target
from .errors import HardwareError
from .registry import get_provider, provider_names

__version__ = "0.1.0"

__all__ = [
    "BackendInfo",
    "CompiledCircuit",
    "CostEstimate",
    "HardwareError",
    "JobHandle",
    "Provider",
    "TargetSpec",
    "compile_for_target",
    "get_provider",
    "provider_names",
    "__version__",
]
