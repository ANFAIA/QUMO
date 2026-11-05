"""Multi-product formulas and Suzuki product formulas.

Reproduces: qiskit-addon-mpf - setup_static_lse and exact coefficients (Quantum 7, 1067)
Source: https://qiskit.github.io/qiskit-addon-mpf/ (Apache-2.0)
Framework: Qiskit (IBM) - https://github.com/Qiskit/qiskit

A weighted sum of Trotter expectations with different step counts cancels their
leading errors in classical post-processing. The addon's documented example has
exact coefficients 0.0416667, -1.0666667, 2.025 for k = 1, 2, 3, order 2. Suzuki's
recursion gives higher orders of one formula instead.

Checks:
- the documented coefficients, and the L1-bounded solution -1, 1.9648...;
- the MPF beats its deepest formula on a 4-qubit Ising chain;
- doubling the steps divides the state error by 2^order for orders 2, 4, 6.

Run: .venv/bin/mojo run -I . examples/chemistry_and_physics/multi_product_formula.mojo
example-meta: {"framework": "qiskit", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import sqrt, log
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.runner import evolve
from qumo.sim.statevector import StateVector
from qumo.algorithms.hamiltonians import transverse_field_ising, exact_evolve
from qumo.algorithms.library import trotter
from qumo.algorithms.mpf import mpf_coefficients, mpf_expectation
from qumo.testing.checks import CheckLog


def distance(a: StateVector, b: StateVector) -> Float64:
    var s = 0.0
    for i in range(a.size()):
        var dr = a.re[i] - b.re[i]
        var di = a.im[i] - b.im[i]
        s += dr * dr + di * di
    return sqrt(s)


def main() raises:
    var log_ = CheckLog("multi_product_formula")
    var x = mpf_coefficients([1, 2, 3])
    _ = log_.close("coefficient k=1", x[0], 0.04166667, 1e-7)
    _ = log_.close("coefficient k=2", x[1], -1.06666667, 1e-7)
    _ = log_.close("coefficient k=3", x[2], 2.025, 1e-7)
    var bounded = mpf_coefficients([1, 2, 3], 2, True, "approx", 3.0)
    _ = log_.close("L1-bounded coefficient k=3", bounded[2], 1.96486533, 1e-6)
    var n = 4
    var h = transverse_field_ising(n, 1.0, 0.8)
    var prep = StateVector(n)
    var obs = parse_pauli_sum("Z0 Z1")
    var exact_state = exact_evolve(h, prep, 1.0)
    var exact = exact_state.expectation(obs)
    var steps: List[Int] = [2, 3, 4]
    var deep = abs(mpf_expectation(h, obs, 1.0, [4], [1.0], prep) - exact)
    var multi = abs(mpf_expectation(h, obs, 1.0, steps, mpf_coefficients(steps), prep) - exact)
    print("error of 4 steps:", deep, "MPF of 2, 3, 4 steps:", multi)
    _ = log_.check("MPF beats its deepest formula", multi < deep / 100, String(deep) + " -> " + String(multi))
    for order in [2, 4, 6]:
        var a = distance(evolve(trotter(h, 1.0, 2, order, n)), exact_state)
        var b = distance(evolve(trotter(h, 1.0, 4, order, n)), exact_state)
        _ = log_.close("order " + String(order) + " slope", log(a / b) / log(2.0), Float64(order), 0.6)
    log_.finish()
