"""VQE for molecular hydrogen (STO-3G, R = 0.735 A).

Compares optimisers and gradient methods on the 4-qubit Jordan-Wigner
Hamiltonian with the UCCSD ansatz, and a hardware-efficient ansatz on the
2-qubit parity-reduced Hamiltonian. Reference: exact diagonalisation (Lanczos).

Run: .venv/bin/mojo run -I . examples/vqe_h2.mojo
"""

from qumo.algorithms.library import h2_ansatz, hardware_efficient_ansatz
from qumo.algorithms.hamiltonians import h2_sto3g, ground_state, H2_HF_ENERGY
from qumo.algorithms.gradients import Grad
from qumo.algorithms.optimizers import SPSA
from qumo.algorithms.variational import vqe, vqe_with, VQEResult

comptime CHEMICAL_ACCURACY = 1.6e-3


def report(label: String, r: VQEResult, exact: Float64):
    var err = abs(r.energy - exact)
    var ok = "yes" if err < CHEMICAL_ACCURACY else "NO"
    print("  ", label, "E =", r.energy, " |err| =", err, " chem.acc:", ok,
          " iters:", r.nit, " circuit evals:", r.circuit_evals, " time:", r.wall_time_s * 1e3, "ms")


def main() raises:
    var h4 = h2_sto3g(4)
    var gs = ground_state(h4, 4)
    var exact = gs.energy
    print("H2 / STO-3G @ 0.735 A:", len(h4), "Pauli terms on 4 qubits (Jordan-Wigner)")
    print("  exact ground energy  :", exact, "Ha  (Lanczos,", gs.iterations, "iterations)")
    print("  Hartree-Fock energy  :", H2_HF_ENERGY, "Ha  (correlation", exact - H2_HF_ENERGY, "Ha)")

    var ansatz = h2_ansatz()
    print("UCCSD ansatz:", ansatz.num_qubits, "qubits,", len(ansatz), "gates, params", ansatz.param_names_str())
    var r1 = vqe(ansatz, h4, "bfgs", grad_method=Grad.ADJOINT)
    report("bfgs        + adjoint     ", r1, exact)
    var r2 = vqe(ansatz, h4, "adam", maxiter=300, grad_method=Grad.PARAM_SHIFT)
    report("adam        + param-shift ", r2, exact)
    var r3 = vqe(ansatz, h4, "nelder-mead", maxiter=2000)
    report("nelder-mead (grad-free)   ", r3, exact)
    var r4 = vqe_with(ansatz, h4, SPSA(maxiter=300, seed=11))
    report("spsa        (grad-free)   ", r4, exact)
    print("  optimal params:", r1.param_names[0], "=", r1.params[0], ",", r1.param_names[1], "=", r1.params[1], ",", r1.param_names[2], "=", r1.params[2])

    var h2 = h2_sto3g(2)
    var hea = hardware_efficient_ansatz(2, 2, "linear", "ry,rz")
    print("Hardware-efficient ansatz on 2-qubit parity-reduced H2:", hea.num_params(), "params")
    var r5 = vqe(hea, h2, "bfgs")
    report("bfgs        + adjoint     ", r5, exact)
