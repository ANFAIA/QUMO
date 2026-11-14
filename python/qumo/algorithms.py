"""Research drivers: VQE, QAOA, Hamiltonian builders, ready-made circuits.

The optimiser loops, gradients and simulations run natively; Python passes the
problem once and receives the result. Hamiltonians are plain lists of
``(label, coeff)`` pairs (dense Pauli labels, leftmost = highest qubit), which
`expectation`, `vqe` and `ground_energy` accept, and which
``qiskit.quantum_info.SparsePauliOp.from_list`` reads as they are.
"""

from __future__ import annotations

from ._native import native as _q
from .circuit import Circuit, _call, _native_of
from .errors import QumoError, QumoTypeError
from .sim import _observable

__all__ = [
    "vqe", "qaoa_maxcut", "ground_energy",
    "h2_sto3g", "transverse_field_ising", "heisenberg_xxz", "fermi_hubbard", "maxcut_hamiltonian",
    "ghz", "qft", "hardware_efficient_ansatz", "qaoa_maxcut_ansatz", "uccsd_ansatz", "h2_ansatz", "trotter",
    "OPTIMIZERS",
]

OPTIMIZERS = ("bfgs", "adam", "gradient-descent", "spsa", "nelder-mead")


def _edges(edges, num_nodes):
    out = []
    try:
        for e in edges:
            e = tuple(e)
            if len(e) == 2:
                out.append((e[0].__index__(), e[1].__index__()))
            elif len(e) == 3:
                out.append((e[0].__index__(), e[1].__index__(), float(e[2])))
            else:
                raise ValueError
    except (TypeError, ValueError, AttributeError):
        raise QumoTypeError("edges is a list of (u, v) or (u, v, weight)", "QM-E911",
                            "e.g. edges=[(0, 1), (1, 2), (2, 0)].") from None
    if not out:
        raise QumoError("the graph has no edges", "QM-E913", "e.g. edges=[(0, 1), (1, 2), (2, 0)].")
    return out, (0 if num_nodes is None else int(num_nodes))


def vqe(ansatz, hamiltonian, *, method: str = "bfgs", x0=None, maxiter: int = 200, tol: float = 1e-8,
        grad: str = "adjoint", seed: int = 7) -> dict:
    """Variational quantum eigensolver: minimise ``<H>`` over the ansatz parameters.

    The whole optimisation (state vectors, adjoint gradients, optimiser) runs
    natively in one call.

    Args:
        ansatz: a parametric `qumo.Circuit`; every symbolic parameter is optimised.
        hamiltonian: a Pauli-sum str or a list of ``(label, coeff)`` pairs.
        method: ``'bfgs'`` (default) | ``'adam'`` | ``'gradient-descent'`` | ``'spsa'`` | ``'nelder-mead'``.
        x0: start point, ``{name: value}`` or a sequence; default: small seeded random values.
        maxiter, tol: optimiser limits (defaults 200, 1e-8).
        grad: ``'adjoint'`` (default, exact) | ``'parameter-shift'`` | ``'finite-diff'``.
        seed: seed of the default start point and of SPSA (default 7).

    Returns:
        dict with ``energy`` (float), ``params`` (NumPy array), ``parameters``
        (``{name: value}``), ``history`` (NumPy array of energies), ``nit``,
        ``nfev``, ``ngrad``, ``circuit_evals``, ``wall_time_s``, ``converged``,
        ``message``, ``optimizer``, ``grad_method``.

    Example:
        >>> import qumo
        >>> r = qumo.vqe(qumo.h2_ansatz(), qumo.h2_sto3g())
        >>> round(r["energy"], 5), r["converged"]
        (-1.13731, True)
    """
    import numpy as np

    from .circuit import param_values

    nat = _native_of(ansatz)
    if nat.num_parameters() == 0:
        raise QumoError("the ansatz has no symbolic parameters to optimise", "QM-E101",
                        "use parameter names as angles, e.g. c.ry('theta', 0), or qumo.hardware_efficient_ansatz(n, layers).")
    start = [] if x0 is None else param_values(ansatz, x0)
    r = dict(_call(_q.vqe, nat, _observable(hamiltonian), method, start, maxiter, float(tol), grad, seed))
    r["params"] = np.asarray(r["params"], dtype=np.float64)
    r["history"] = np.asarray(r["history"], dtype=np.float64)
    r["param_names"] = list(r["param_names"])
    r["parameters"] = dict(zip(r["param_names"], r["params"].tolist()))
    return r


def qaoa_maxcut(edges, *, num_nodes: int | None = None, p: int = 1, method: str = "bfgs", restarts: int = 4,
                shots: int = 1024, seed: int = 7, maxiter: int = 200) -> dict:
    """Solve MaxCut on a graph with depth-`p` QAOA (ansatz, optimisation and sampling run natively).

    Args:
        edges: ``[(u, v), ...]`` or ``[(u, v, weight), ...]``.
        num_nodes: number of nodes (default: largest node + 1).
        p: QAOA depth. restarts: optimisation starts (best kept).
        shots, seed: sampling of the optimised circuit (defaults 1024, 7).

    Returns:
        dict with ``expected_cut``, ``max_cut`` (brute force), ``approx_ratio``,
        ``gammas`` and ``betas`` (NumPy arrays), ``best_bitstring``,
        ``best_cut``, ``prob_optimal``, ``most_frequent``, ``history`` (NumPy
        array), ``nfev``, ``circuit_evals``, ``wall_time_s``, ``optimizer``.

    Example:
        >>> import qumo
        >>> r = qumo.qaoa_maxcut([(0, 1), (1, 2), (2, 3), (3, 0)], p=1)
        >>> r["max_cut"], r["best_cut"], round(r["approx_ratio"], 2)
        (4.0, 4.0, 0.75)
    """
    import numpy as np

    es, n = _edges(edges, num_nodes)
    r = dict(_call(_q.qaoa_maxcut, n, es, p, method, restarts, shots, seed, maxiter))
    for key in ("gammas", "betas", "history"):
        r[key] = np.asarray(r[key], dtype=np.float64)
    return r


def ground_energy(hamiltonian, num_qubits: int | None = None) -> float:
    """Exact ground-state energy of a Pauli-sum Hamiltonian (Lanczos), to check a VQE result.

    Example:
        >>> import qumo
        >>> round(qumo.ground_energy(qumo.h2_sto3g()), 5)
        -1.13731
    """
    return float(_call(_q.ground_energy, _observable(hamiltonian), 0 if num_qubits is None else num_qubits))


# ----------------------------------------------------------------------------- Hamiltonians
def h2_sto3g(num_qubits: int = 4, include_nuclear_repulsion: bool = True) -> list[tuple[str, float]]:
    """Molecular hydrogen in the STO-3G basis at 0.735 angstrom, as ``[(label, coeff), ...]``.

    `num_qubits` 4 (Jordan-Wigner) or 2 (tapered).

    Example:
        >>> import qumo
        >>> len(qumo.h2_sto3g()), qumo.h2_sto3g()[1][0]
        (15, 'IIIZ')
    """
    return list(_call(_q.h2_sto3g, num_qubits, bool(include_nuclear_repulsion)))


def transverse_field_ising(n: int, j: float = 1.0, h: float = 1.0, periodic: bool = False) -> list[tuple[str, float]]:
    """``-j sum Z_i Z_{i+1} - h sum X_i`` on a chain of `n` qubits.

    Example:
        >>> import qumo
        >>> qumo.transverse_field_ising(2, h=0.5)
        [('ZZ', -1.0), ('IX', -0.5), ('XI', -0.5)]
    """
    return list(_call(_q.transverse_field_ising, n, float(j), float(h), bool(periodic)))


def heisenberg_xxz(n: int, j: float = 1.0, delta: float = 1.0, periodic: bool = False,
                   field: float = 0.0) -> list[tuple[str, float]]:
    """XXZ chain ``j sum (XX + YY + delta ZZ) + field sum Z`` on `n` qubits."""
    return list(_call(_q.heisenberg_xxz, n, float(j), float(delta), bool(periodic), float(field)))


def fermi_hubbard(edges, *, num_sites: int | None = None, t: float = 1.0, u: float = 4.0,
                  mu: float = 0.0) -> list[tuple[str, float]]:
    """Fermi-Hubbard model on a lattice given by `edges` (Jordan-Wigner, two qubits per site)."""
    es, n = _edges(edges, num_sites)
    return list(_call(_q.fermi_hubbard, n, es, float(t), float(u), float(mu)))


def maxcut_hamiltonian(edges, *, num_nodes: int | None = None, include_offset: bool = True) -> list[tuple[str, float]]:
    """MaxCut cost Hamiltonian of a graph: ``<H> = -<cut>``."""
    es, n = _edges(edges, num_nodes)
    return list(_call(_q.maxcut_hamiltonian, n, es, bool(include_offset)))


# ----------------------------------------------------------------------------- circuits
def ghz(n: int, measure: bool = False) -> Circuit:
    """GHZ state on `n` qubits (``measure=True`` adds `measure_all`).

    Example:
        >>> import qumo
        >>> sorted(qumo.sample(qumo.ghz(3, measure=True), shots=100))
        ['000', '111']
    """
    c = Circuit.from_native(_call(_q.ghz, n))
    return c.measure_all() if measure else c


def qft(n: int, do_swaps: bool = True) -> Circuit:
    """Quantum Fourier transform on `n` qubits."""
    return Circuit.from_native(_call(_q.qft, n, bool(do_swaps)))


def hardware_efficient_ansatz(n: int, layers: int = 1, entanglement: str = "linear",
                              rotations: str = "ry,rz") -> Circuit:
    """Parametric ansatz: rotation layers and entanglers (``'linear'``, ``'circular'``, ``'full'``).

    Example:
        >>> import qumo
        >>> qumo.hardware_efficient_ansatz(2, layers=1).num_parameters
        8
    """
    return Circuit.from_native(_call(_q.hardware_efficient_ansatz, n, layers, entanglement, rotations))


def qaoa_maxcut_ansatz(edges, p: int = 1, *, num_nodes: int | None = None) -> Circuit:
    """QAOA ansatz for MaxCut; parameters ``gamma[0..p-1]``, ``beta[0..p-1]``."""
    es, n = _edges(edges, num_nodes)
    return Circuit.from_native(_call(_q.qaoa_maxcut_ansatz, n, es, p))


def uccsd_ansatz(num_qubits: int, num_electrons: int) -> Circuit:
    """UCCSD ansatz (Jordan-Wigner) with one parameter per excitation."""
    return Circuit.from_native(_call(_q.uccsd_ansatz, num_qubits, num_electrons))


def h2_ansatz() -> Circuit:
    """UCCSD ansatz for the 4-qubit H2 Hamiltonian of `h2_sto3g()`."""
    return Circuit.from_native(_call(_q.h2_ansatz))


def trotter(hamiltonian, time: float, steps: int = 1, order: int = 1, num_qubits: int | None = None) -> Circuit:
    """Trotter-Suzuki circuit for ``exp(-i H time)`` (`order` 1, 2 or 4)."""
    return Circuit.from_native(_call(_q.trotter, _observable(hamiltonian), float(time), steps, order,
                                     0 if num_qubits is None else num_qubits))
