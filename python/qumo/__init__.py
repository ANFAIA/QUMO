"""QUMO for Python: build, simulate and compile quantum circuits; run research drivers.

The package is a thin layer over the compiled extension `qumo_py` (written in
Mojo, built with `make py`); every heavy step runs natively.

    import qumo

    bell = qumo.Circuit(2).h(0).cx(0, 1).measure_all()
    print(bell.draw())
    counts = qumo.sample(bell, shots=1000, seed=7)        # {'00': ..., '11': ...}
    psi = qumo.statevector(qumo.Circuit(2).h(0).cx(0, 1)) # NumPy complex128
    e = qumo.expectation(qumo.ghz(3), "Z0 Z1")
    r = qumo.transpile(bell, "grid:2x2/sc_cz")            # r.circuit, r.initial_layout, r.metrics

Reference: docs/PYTHON.md. `qumo.native` is the low-level module, for what
this package does not wrap (hardware clients, programs with control flow).
"""

from ._native import native
from .errors import QumoError, QumoNativeMissing, QumoTypeError
from .circuit import Circuit, Parameter, from_stim, load, load_qasm
from .sim import (
    Counts, batch_value_and_gradient, expectation, probabilities, run, sample, statevector, value_and_gradient,
)
from .compiler import BASIS_SETS, TOPOLOGIES, TranspileResult, target_info, transpile
from .algorithms import (
    OPTIMIZERS, fermi_hubbard, ghz, ground_energy, h2_ansatz, h2_sto3g, hardware_efficient_ansatz, heisenberg_xxz,
    maxcut_hamiltonian, qaoa_maxcut, qaoa_maxcut_ansatz, qft, transverse_field_ising, trotter, uccsd_ansatz, vqe,
)
from .qec import CODES, DECODERS, QecMemory, memory_experiment, threshold_sweep

__version__ = str(native.version())


def bind(circuit, values=None, **named):
    """A new circuit with its parameters replaced: `values` is ``{name: value}`` or a sequence.

    Example:
        >>> import qumo
        >>> c = qumo.Circuit(1).rx("theta", 0).measure_all()
        >>> qumo.bind(c, {"theta": 3.141592653589793}).num_parameters
        0
    """
    if not isinstance(circuit, Circuit):
        circuit = Circuit.from_native(circuit)
    return circuit.bind(values, **named)


def gpu_info() -> dict:
    """GPU status of this build: compiled, available, name, precision, ..."""
    return dict(native.gpu_info())


__all__ = [
    "Circuit", "Parameter", "load", "load_qasm", "from_stim", "bind",
    "sample", "run", "statevector", "probabilities", "expectation", "Counts",
    "value_and_gradient", "batch_value_and_gradient",
    "transpile", "TranspileResult", "target_info", "BASIS_SETS", "TOPOLOGIES",
    "vqe", "qaoa_maxcut", "ground_energy", "OPTIMIZERS",
    "h2_sto3g", "transverse_field_ising", "heisenberg_xxz", "fermi_hubbard", "maxcut_hamiltonian",
    "ghz", "qft", "hardware_efficient_ansatz", "qaoa_maxcut_ansatz", "uccsd_ansatz", "h2_ansatz", "trotter",
    "memory_experiment", "threshold_sweep", "QecMemory", "CODES", "DECODERS",
    "QumoError", "QumoTypeError", "QumoNativeMissing", "gpu_info", "native", "__version__",
]
