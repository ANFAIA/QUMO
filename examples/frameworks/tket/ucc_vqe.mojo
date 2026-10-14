"""UCC VQE for H2 with Pauli gadgets, PauliSimp and measurement grouping.

Reproduces: TKET - VQE with UCC ansatz
Source: https://docs.quantinuum.com/tket/user-guide/examples/algorithms_and_protocols/ucc_vqe.html (pytket user guide, Apache-2.0)

The TKET page writes the H2 / STO-3G UCC ansatz as a list of Pauli
exponentials (PauliExpBox) with three parameters (two singles, one double),
simplifies the circuit with PauliSimp, groups the Hamiltonian terms so fewer
measurement circuits are needed, estimates the energy with 4000 shots per
circuit and minimises it. The same steps in QUMO:

- the ansatz is a gadget table (Pauli string, coefficient, parameter) turned
  into gates by `append_pauli_rotation` (basis change + CX ladder + RZ);
- `compile_spec(..., "remove_identities,pauli_simp,kak,cancel,commute")`:
  QUMO's PauliSimp (Clifford tableau extraction, merging, ladder sharing)
  followed by KAK block resynthesis, reported as 2-qubit counts;
- `PauliSum.group_qwc` + `EstimatorPub` with 4000 shots per group;
- `vqe(..., "bfgs", Grad.ADJOINT)` minimises the exact energy; the page's
  Nelder-Mead start [1e-4, 1e-4, 0.4] is also run.

Differences from the page, on purpose: QUMO's built-in H2 operator
(`h2_sto3g(4)`, R = 0.735 A, interleaved spin orbitals, Hartree-Fock |0011>)
replaces the page's qiskit-aqua operator at 0.75 A, so the gadget strings are
the Jordan-Wigner excitations in QUMO's orbital order. The page's
CommutingSets grouping (general commutation + Clifford diagonalisation) is not
in QUMO, which offers qubit-wise-commuting groups (TKET's NonConflictingSets).
The page prints no energies or CX counts; the CX counts quoted here come from
reference/ucc_ref.py (rerun pytket 2.18.5 on the same gadget table).

Checks:
- the gadget-table ansatz equals the library `h2_ansatz()` state (1e-12);
- VQE (BFGS + adjoint) reaches the exact ground energy -1.1373 Ha (1e-6);
- the compiled circuits (a generic point and the optimum) are equivalent on
  all 16 basis inputs (1e-10), never have more 2-qubit gates than the input,
  and the compiled optimum has the exact energy (1e-9);
- the 4000-shot QWC estimate is within 4 standard errors of the exact energy.

Run: .venv/bin/mojo run -I . examples/frameworks/tket/ucc_vqe.mojo
example-meta: {"framework": "tket", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, evolve_into
from qumo.algorithms.library import append_pauli_rotation, h2_ansatz
from qumo.algorithms.hamiltonians import h2_sto3g, ground_energy, H2_EXACT_ENERGY, H2_HF_ENERGY
from qumo.algorithms.gradients import Grad
from qumo.algorithms.variational import vqe
from qumo.compiler.target import Target
from qumo.compiler.pass_manager import compile_spec
from qumo.exec.primitives import EstimatorPub, run_estimator
from qumo.sim.rng import Rng
from qumo.ai.checks import CheckLog

comptime SHOTS = 4000
comptime PIPELINE = "remove_identities,pauli_simp,kak,cancel,commute"


def gadget_table() -> List[String]:
    """One Pauli sum per parameter: exp(-i p/2 * sum_k c_k P_k) as commuting gadgets.

    Qubits: q0 = g-up, q1 = g-down, q2 = u-up, q3 = u-down (Jordan-Wigner,
    Z strings inside each excitation). p0: single 0->2, p1: single 1->3,
    p2: double (0,1 -> 2,3). Same role as the page's PauliExpBox list.
    """
    var t: List[String] = [
        "X0 Z1 Y2 - Y0 Z1 X2",
        "X1 Z2 Y3 - Y1 Z2 X3",
        "-0.25*X0 X1 X2 Y3 - 0.25*X0 X1 Y2 X3 + 0.25*X0 Y1 X2 X3 - 0.25*X0 Y1 Y2 Y3"
        + " + 0.25*Y0 X1 X2 X3 - 0.25*Y0 X1 Y2 Y3 + 0.25*Y0 Y1 X2 Y3 + 0.25*Y0 Y1 Y2 X3",
    ]
    return t^


def ucc_from_gadgets() raises -> Circuit:
    var c = Circuit(4, name="ucc_gadgets")
    c.x(0)  # Hartree-Fock |0011>
    c.x(1)
    var table = gadget_table()
    for k in range(len(table)):
        var p = c.parameter("p" + String(k))
        var terms = parse_pauli_sum(table[k])
        for t in terms.terms:
            append_pauli_rotation(c, t.x, t.z, p * t.coeff)
    return c^


def max_unitary_diff(a: Circuit, b: Circuit) raises -> Float64:
    """Largest |<i|A - B|j>| over all basis inputs j (global phase tracked, not removed)."""
    var n = a.num_qubits
    var empty = List[Float64]()
    var worst = 0.0
    for j in range(1 << n):
        var sa = StateVector.basis(n, j)
        var sb = StateVector.basis(n, j)
        evolve_into(sa, a, empty)
        evolve_into(sb, b, empty)
        for i in range(1 << n):
            worst = max(worst, abs(sa.re[i] - sb.re[i]) + abs(sa.im[i] - sb.im[i]))
    return worst


def pass_note(notes: List[String], names: List[String], name: String) -> String:
    for i in range(len(names)):
        if names[i] == name:
            return notes[i]
    return "not run"


def main() raises:
    var log = CheckLog("tket/ucc_vqe")
    var h = h2_sto3g(4)
    var e_exact = ground_energy(h, 4)
    print("H2 / STO-3G (R = 0.735 A):", len(h), "Pauli terms on 4 qubits")
    print("  exact ground energy (Lanczos):", e_exact, "Ha; FCI constant", H2_EXACT_ENERGY, "; HF", H2_HF_ENERGY)

    # 1. Gadget-table ansatz == library UCCSD ansatz.
    var ucc = ucc_from_gadgets()
    var lib = h2_ansatz()
    print("UCC from", len(gadget_table()), "gadget groups:", len(ucc), "gates,", ucc.num_2q_gates(),
          "two-qubit gates, params", ucc.param_names_str())
    var rng = Rng(7)
    var worst = 0.0
    for _ in range(5):
        var x: List[Float64] = [rng.uniform() - 0.5, rng.uniform() - 0.5, rng.uniform() - 0.5]
        var sa = evolve(ucc, x)
        var sb = evolve(lib, x)
        for i in range(16):
            worst = max(worst, abs(sa.re[i] - sb.re[i]) + abs(sa.im[i] - sb.im[i]))
    _ = log.close("gadget ansatz equals h2_ansatz()", worst, 0.0, 1e-12)

    # 2. VQE: BFGS + adjoint gradients (exact energy), and the page's Nelder-Mead start.
    var r = vqe(ucc, h, "bfgs", grad_method=Grad.ADJOINT)
    print("VQE bfgs+adjoint: E =", r.energy, " iterations", r.nit, " circuit evals", r.circuit_evals,
          " params", r.params[0], r.params[1], r.params[2])
    _ = log.close("VQE energy equals exact ground energy", r.energy, e_exact, 1e-6)
    var x0: List[Float64] = [1e-4, 1e-4, 4e-1]  # the page's initial_params
    var rn = vqe(ucc, h, "nelder-mead", x0=x0, maxiter=2000)
    print("VQE nelder-mead from the page's [1e-4, 1e-4, 0.4]: E =", rn.energy, " nfev", rn.nfev,
          " (rerun scipy Nelder-Mead in reference/ucc_ref.py: -1.1373060093711371)")

    # 3. PauliSimp on bound circuits (QUMO's pauli_simp skips symbolic circuits).
    #    Singles vanish at the optimum by symmetry, so it is compiled at [0, 0, p2*].
    var labels: List[String] = ["generic [0.3, -0.2, 0.11]", "optimum [0, 0, p2*]"]
    var points: List[List[Float64]] = [[0.3, -0.2, 0.11], [0.0, 0.0, r.params[2]]]
    var pytket_ps: List[Int] = [30, 14]  # rerun pytket 2.18.5: DecomposeBoxes 64 CX -> PauliSimp
    var pytket_fpo: List[Int] = [23, 13]  # ... -> PauliSimp + FullPeepholeOptimise
    var compiled_opt = Circuit(4)
    for k in range(2):
        var bound = ucc.bind(points[k])
        var res = compile_spec(bound, Target.all_to_all(4), PIPELINE)
        var out = res.circuit.copy()
        var names = List[String]()
        var notes = List[String]()
        for rep in res.reports:
            names.append(rep.name)
            notes.append(rep.note)
        print("compile", labels[k], "with '" + String(PIPELINE) + "':")
        print("  2-qubit gates", res.twoq_before, "->", out.num_2q_gates(), "| depth", res.depth_before, "->", out.depth(),
              "| pauli_simp:", pass_note(notes, names, "pauli_simp"), "| kak:", pass_note(notes, names, "kak"))
        print("  rerun pytket 2.18.5 on the same table: CX 64 -> PauliSimp", pytket_ps[k],
              "-> + FullPeepholeOptimise", pytket_fpo[k])
        var diff = max_unitary_diff(bound, out)
        _ = log.close("compiled " + labels[k] + " equivalent", diff, 0.0, 1e-10)
        _ = log.check("compiled " + labels[k] + " has no extra 2-qubit gates", out.num_2q_gates() <= res.twoq_before,
                      "2q " + String(res.twoq_before) + " -> " + String(out.num_2q_gates()))
        if k == 1:
            compiled_opt = out^
    var e_comp = evolve(compiled_opt).expectation(h)
    _ = log.close("compiled optimum energy", e_comp, e_exact, 1e-9)

    # 4. Measurement reduction + shots: one setting per qubit-wise-commuting group.
    var groups = h.group_qwc()
    var naive = 0
    for t in h.terms:
        if t.x != 0 or t.z != 0:
            naive += 1
    print("measurement circuits: naive", naive, "(one per non-identity term) -> QWC groups", len(groups))
    var obs = List[PauliSum]()
    obs.append(h.copy())
    var precision = 1.0 / sqrt(Float64(SHOTS) - 0.5)  # run_estimator: shots = int(1/precision^2) + 1
    var est = run_estimator(EstimatorPub(compiled_opt.copy(), obs, List[List[Float64]](), precision, 11))
    var e_shots = est.evs[0][0]
    var sd = est.stds[0][0]
    print("shot estimate (", est.shots_per_set, "shots per group,", est.method, "): E =", e_shots, "+/-", sd)
    _ = log.check("4000 shots per measurement circuit", est.shots_per_set == SHOTS, "shots=" + String(est.shots_per_set))
    _ = log.close("shot estimate within 4 sigma", e_shots, e_exact, 4.0 * sd)
    log.finish()
