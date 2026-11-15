"""VQE for molecular hydrogen (H2, STO-3G basis, bond length 0.735 angstrom).

The variational quantum eigensolver (VQE) finds the lowest energy of a molecule: a small
circuit with adjustable angles prepares a trial state, a classical optimiser changes the angles
to lower the measured energy, and it stops when the energy no longer drops. Here the answer is
known exactly, so every optimiser can be checked against it:

  exact ground energy   from exact diagonalisation (Lanczos), about -1.137 Hartree
  Hartree-Fock energy   the cheap approximation VQE improves on
  chem.acc: yes         the error is below chemical accuracy, 1.6 milli-Hartree

It runs four optimisers (BFGS, Adam, Nelder-Mead and SPSA; the last two use no gradients) on
the 4-qubit Jordan-Wigner Hamiltonian with the UCCSD ansatz, then a hardware-efficient ansatz on
the 2-qubit parity-reduced Hamiltonian. All of them should print "chem.acc: yes".

Run: .venv/bin/mojo run -I . examples/chemistry_and_physics/vqe_h2.mojo

Framework: QUMO - https://github.com/ANFAIA/QUMO
"""

# Chemistry and variational algorithms are one explicit import away: they live in
# qumo.algorithms, not in the root.
from qumo.algorithms.library import h2_ansatz, hardware_efficient_ansatz
from qumo.algorithms.hamiltonians import h2_sto3g, ground_state, H2_HF_ENERGY
from qumo.algorithms.gradients import Grad
from qumo.algorithms.optimizers import SPSA
from qumo.algorithms.variational import vqe, vqe_with, VQEResult

comptime CHEMICAL_ACCURACY = 1.6e-3  # Hartree: the usual target for a chemistry calculation


def report(label: String, r: VQEResult, exact: Float64):
    """One line per run: the energy, its distance from the exact value and the cost."""
    var err = abs(r.energy - exact)
    var ok = "yes" if err < CHEMICAL_ACCURACY else "NO"
    print("  ", label, "E =", r.energy, " |err| =", err, " chem.acc:", ok,
          " iters:", r.nit, " circuit evals:", r.circuit_evals, " time:", r.wall_time_s * 1e3, "ms")


def main() raises:
    # 1. The problem: the molecule's Hamiltonian as a sum of Pauli terms, and the exact answer.
    var h4 = h2_sto3g(4)
    var gs = ground_state(h4, 4)
    var exact = gs.energy
    print("H2 / STO-3G @ 0.735 A:", len(h4), "Pauli terms on 4 qubits (Jordan-Wigner)")
    print("  exact ground energy  :", exact, "Ha  (Lanczos,", gs.iterations, "iterations)")
    print("  Hartree-Fock energy  :", H2_HF_ENERGY, "Ha  (correlation", exact - H2_HF_ENERGY, "Ha)")

    # 2. The trial circuit (ansatz): chemistry-inspired UCCSD with three adjustable angles.
    var ansatz = h2_ansatz()
    print("UCCSD ansatz:", ansatz.num_qubits, "qubits,", len(ansatz), "gates, params", ansatz.param_names_str())

    # 3. Four ways to tune the angles. Gradient-based ones converge in fewer evaluations.
    var r1 = vqe(ansatz, h4, "bfgs", grad_method=Grad.ADJOINT)
    report("bfgs        + adjoint     ", r1, exact)
    var r2 = vqe(ansatz, h4, "adam", maxiter=300, grad_method=Grad.PARAM_SHIFT)
    report("adam        + param-shift ", r2, exact)
    var r3 = vqe(ansatz, h4, "nelder-mead", maxiter=2000)
    report("nelder-mead (grad-free)   ", r3, exact)
    var r4 = vqe_with(ansatz, h4, SPSA(maxiter=300, seed=11))
    report("spsa        (grad-free)   ", r4, exact)
    print("  optimal params:", r1.param_names[0], "=", r1.params[0], ",", r1.param_names[1], "=", r1.params[1], ",", r1.param_names[2], "=", r1.params[2])

    # 4. The same molecule on 2 qubits (symmetry-reduced), with a generic hardware-friendly ansatz.
    var h2 = h2_sto3g(2)
    var hea = hardware_efficient_ansatz(2, 2, "linear", "ry,rz")
    print("Hardware-efficient ansatz on 2-qubit parity-reduced H2:", hea.num_params(), "params")
    var r5 = vqe(hea, h2, "bfgs")
    report("bfgs        + adjoint     ", r5, exact)
