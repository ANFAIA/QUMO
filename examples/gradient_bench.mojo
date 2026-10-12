"""Adjoint differentiation vs parameter-shift: wall time for one full gradient.

16-qubit hardware-efficient ansatz (RY+RZ, 2 layers, linear CX) = 96 parameters,
TFIM Hamiltonian. Parameter-shift needs 2 circuit executions per parameter
occurrence (192 here); adjoint needs one forward pass and one reverse sweep
regardless of the parameter count.

Run: .venv/bin/mojo run -I . examples/gradient_bench.mojo
"""

from std.time import perf_counter_ns
from qumo.sim.rng import Rng
from qumo.algorithms.library import hardware_efficient_ansatz
from qumo.algorithms.hamiltonians import transverse_field_ising
from qumo.algorithms.gradients import ExpectationEngine, Grad, shift_evaluations


def main() raises:
    var n = 16
    var ansatz = hardware_efficient_ansatz(n, 2, "linear", "ry,rz")
    var h = transverse_field_ising(n, 1.0, 0.8)
    var rng = Rng(42)
    var x = List[Float64]()
    for _ in range(ansatz.num_params()):
        x.append(rng.uniform() * 2.0 - 1.0)
    var eng = ExpectationEngine(ansatz, h)
    _ = eng.value(x)  # warm-up

    var t0 = perf_counter_ns()
    var reps = 5
    var ga = List[Float64]()
    for _ in range(reps):
        ga = eng.gradient(x, Grad.ADJOINT)
    var t_adj = Float64(perf_counter_ns() - t0) * 1e-6 / Float64(reps)

    var evals0 = eng.nfev
    t0 = perf_counter_ns()
    var gp = eng.gradient(x, Grad.PARAM_SHIFT)
    var t_ps = Float64(perf_counter_ns() - t0) * 1e-6
    var ps_evals = eng.nfev - evals0

    t0 = perf_counter_ns()
    _ = eng.value(x)
    var t_val = Float64(perf_counter_ns() - t0) * 1e-6

    var maxdiff = 0.0
    for i in range(len(ga)):
        maxdiff = max(maxdiff, abs(ga[i] - gp[i]))
    print("gradient benchmark:", n, "qubits,", ansatz.num_params(), "params,", len(ansatz), "gates,", len(h), "Pauli terms")
    print("  one energy evaluation      :", t_val, "ms")
    print("  adjoint gradient           :", t_adj, "ms  (", t_adj / t_val, "x one evaluation)")
    print("  parameter-shift gradient   :", t_ps, "ms  (", ps_evals, "circuit executions = 1 energy +", shift_evaluations(ansatz), "shifts )")
    print("  speed-up adjoint vs shift  :", t_ps / t_adj, "x")
    print("  max |adjoint - shift|      :", maxdiff)
