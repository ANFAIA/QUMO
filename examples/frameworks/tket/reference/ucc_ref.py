"""pytket reference for examples/frameworks/tket/ucc_vqe.mojo (optional).

Builds the same H2 UCC gadget table as the QUMO example (Jordan-Wigner,
interleaved spin orbitals, Hartree-Fock |0011> = X on q0 and q1) with
PauliExpBox, then reports CX counts after DecomposeBoxes and after PauliSimp,
the exact energy at the QUMO VQE optimum (pytket's own statevector), and the
optimum itself found with scipy (Nelder-Mead, as on the TKET page).

Run (QUMO repo root): bench/.venv/bin/python examples/frameworks/tket/reference/ucc_ref.py
Needs pytket (2.18.5 was used for the numbers quoted in the example), numpy, scipy.
"""

from __future__ import annotations

import json
import math

import numpy as np
from pytket.circuit import Circuit, OpType, PauliExpBox
from pytket.pauli import Pauli
from pytket.passes import DecomposeBoxes, FullPeepholeOptimise, PauliSimp

P = {"X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z, "I": Pauli.I}

# (parameter index, coefficient, {qubit: letter}); same table as gadget_table() in ucc_vqe.mojo
GADGETS = [
    (0, 1.0, {0: "X", 1: "Z", 2: "Y"}), (0, -1.0, {0: "Y", 1: "Z", 2: "X"}),
    (1, 1.0, {1: "X", 2: "Z", 3: "Y"}), (1, -1.0, {1: "Y", 2: "Z", 3: "X"}),
    (2, -0.25, {0: "X", 1: "X", 2: "X", 3: "Y"}), (2, -0.25, {0: "X", 1: "X", 2: "Y", 3: "X"}),
    (2, 0.25, {0: "X", 1: "Y", 2: "X", 3: "X"}), (2, -0.25, {0: "X", 1: "Y", 2: "Y", 3: "Y"}),
    (2, 0.25, {0: "Y", 1: "X", 2: "X", 3: "X"}), (2, -0.25, {0: "Y", 1: "X", 2: "Y", 3: "Y"}),
    (2, 0.25, {0: "Y", 1: "Y", 2: "X", 3: "Y"}), (2, 0.25, {0: "Y", 1: "Y", 2: "Y", 3: "X"}),
]

# QUMO h2_sto3g(4) integrals (qumo/algorithms/hamiltonians.mojo), rebuilt here.
HG, HU = -1.2563390730032498, -0.47189600728114184
JG, JU, JGU, K = 0.6757101548035161, 0.6985737227320210, 0.6645817302552968, 0.18093119978423156
ENUC = 0.7199689944489797


def hamiltonian() -> np.ndarray:
    """Dense 16x16 H2 Hamiltonian, index = sum_q bit_q 2^q (QUMO little-endian)."""
    I2 = np.eye(2)
    mats = {"I": I2, "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}

    def op(letters: dict[int, str]) -> np.ndarray:
        m = np.array([[1.0]])
        for q in reversed(range(4)):  # kron with qubit 3 leftmost -> little-endian index
            m = np.kron(m, mats[letters.get(q, "I")])
        return m

    zg = -HG / 2 - JG / 4 - JGU / 2 + K / 4
    zu = -HU / 2 - JU / 4 - JGU / 2 + K / 4
    h = (HG + HU + JG / 4 + JU / 4 + JGU - K / 2 + ENUC) * op({})
    h = h + zg * (op({0: "Z"}) + op({1: "Z"})) + zu * (op({2: "Z"}) + op({3: "Z"}))
    h = h + JG / 4 * op({0: "Z", 1: "Z"}) + JU / 4 * op({2: "Z", 3: "Z"})
    h = h + (JGU - K) / 4 * (op({0: "Z", 2: "Z"}) + op({1: "Z", 3: "Z"}))
    h = h + JGU / 4 * (op({0: "Z", 3: "Z"}) + op({1: "Z", 2: "Z"}))
    c = K / 4
    h = h - c * op({0: "X", 1: "X", 2: "Y", 3: "Y"}) + c * op({0: "X", 1: "Y", 2: "Y", 3: "X"})
    h = h + c * op({0: "Y", 1: "X", 2: "X", 3: "Y"}) - c * op({0: "Y", 1: "Y", 2: "X", 3: "X"})
    return h


def ucc(params: list[float]) -> Circuit:
    c = Circuit(4)
    c.X(0)
    c.X(1)
    for k, coeff, letters in GADGETS:
        qubits = sorted(letters)
        # QUMO gadget: exp(-i (coeff p)/2 P); PauliExpBox(t): exp(-i pi t/2 P) -> t = coeff p / pi
        box = PauliExpBox([P[letters[q]] for q in qubits], coeff * params[k] / math.pi)
        c.add_pauliexpbox(box, qubits)
    return c


def statevector_le(c: Circuit) -> np.ndarray:
    """pytket statevector (ILO-BE: qubit 0 most significant) -> QUMO little-endian order."""
    sv = c.get_statevector()
    out = np.zeros(16, complex)
    for i in range(16):
        le = sum(((i >> (3 - q)) & 1) << q for q in range(4))
        out[le] = sv[i]
    return out


def energy(params: list[float], h: np.ndarray) -> float:
    c = ucc(params)
    DecomposeBoxes().apply(c)
    psi = statevector_le(c)
    return float(np.real(np.conj(psi) @ h @ psi))


def main() -> None:
    from scipy.optimize import minimize

    h = hamiltonian()
    exact = float(np.linalg.eigvalsh(h)[0])
    res = minimize(energy, [1e-4, 1e-4, 4e-1], args=(h,), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
    out: dict = {"pytket": __import__("pytket").__version__, "exact_energy": exact,
                 "nelder_mead_energy": float(res.fun), "nelder_mead_params": [float(v) for v in res.x]}
    for label, params in (("generic", [0.3, -0.2, 0.11]), ("optimum", [0.0, 0.0, float(res.x[2])])):
        c = ucc(params)
        DecomposeBoxes().apply(c)
        cx_dec = c.n_gates_of_type(OpType.CX)
        c2 = c.copy()
        PauliSimp().apply(c2)
        cx_ps = c2.n_gates_of_type(OpType.CX)
        c3 = c2.copy()
        FullPeepholeOptimise().apply(c3)
        out[label] = {"params": params, "cx_decomposed": cx_dec, "cx_paulisimp": cx_ps,
                      "cx_paulisimp_peephole": c3.n_gates_of_type(OpType.CX),
                      "energy": energy(params, h)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
