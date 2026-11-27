"""An optimised 8th-order product formula: rotations needed for a target error, against Suzuki.

Reproduces: Maležič, Ostmeyer - Optimised 8th-Order Trotter-Suzuki Schemes for Quantum Simulation (the q = 21 scheme of Table 3 on the Heisenberg XZ chain of Section III B, local ordering)
Source: https://arxiv.org/abs/2611.27896
Framework: QUMO - https://github.com/ANFAIA/QUMO

`trotter(h, t, steps, 8, scheme="mo8_q21")` composes 21 Strang steps per step with the
weights of Table 3 (w_j = 2 c_j), where Suzuki's order-8 recursion needs 125. The model is
the paper's open XZ chain, L = 6, H = sum_i (X_i X_{i+1} + Z_i Z_{i+1}) + sum_i h_i Z_i,
one Pauli exponential per term in bond-by-bond order (16 terms). The fields h_i are fixed
values in [-0.1, 0.1] here, in place of the paper's random draws. t = 10, initial state
|000000>. For each scheme and target error the smallest step count is found (doubling, then
bisection), and its cost is the number of Pauli rotations trotter() emits:
steps x Strang steps per step x (2 x 16 - 1).

Reference: exact_evolve (a Taylor series of exp(-iHt), not a product formula).

Checks:
- the weights sum to 1 (to 8e-16) and the circuit has the stated number of rotations;
- at loose targets (1e-1, 1e-2) Suzuki order 4 needs fewer rotations than mo8_q21;
- at 1e-4, 1e-6 and 1e-8 mo8_q21 needs the fewest rotations of Strang and Suzuki 4, 6, 8,
  at 1e-6 at least 1.5 times fewer than the best of them and 3 times fewer than Suzuki 8.
Noiseless only: the paper finds Strang the best choice at today's two-qubit error rates
(its Table 5), and nothing here measures noise.

Run: .venv/bin/mojo run -I . examples/chemistry_and_physics/trotter_order8_cost.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import evolve
from qumo.sim.statevector import StateVector
from qumo.algorithms.hamiltonians import exact_evolve
from qumo.algorithms.library import trotter, mo8_q21_weights
from qumo.testing.checks import CheckLog


def distance(a: StateVector, b: StateVector) -> Float64:
    var s = 0.0
    for i in range(a.size()):
        var dr = a.re[i] - b.re[i]
        var di = a.im[i] - b.im[i]
        s += dr * dr + di * di
    return sqrt(s)


def xz_chain(fields: List[Float64]) -> PauliSum:
    """Open XZ chain, terms bond by bond: X_i X_{i+1}, Z_i Z_{i+1}, h_i Z_i, then h_{L-1} Z_{L-1}."""
    var L = len(fields)
    var h = PauliSum()
    for i in range(L - 1):
        var m = (UInt64(1) << UInt64(i)) | (UInt64(1) << UInt64(i + 1))
        h.add(1.0, m, 0)
        h.add(1.0, 0, m)
        h.add(fields[i], 0, UInt64(1) << UInt64(i))
    h.add(fields[L - 1], 0, UInt64(1) << UInt64(L - 1))
    return h^


def build(h: PauliSum, t: Float64, steps: Int, order: Int, scheme: String) raises -> Circuit:
    return trotter(h, t, steps, order, h.num_qubits(), "", scheme)


def rotations(c: Circuit) -> Int:
    var k = 0
    for g in c.ops:
        if g.kind == Op.RZ or g.kind == Op.RZZ:
            k += 1
    return k


def min_steps(h: PauliSum, t: Float64, order: Int, scheme: String, exact: StateVector, eps: Float64) raises -> Int:
    """Smallest step count found with error < eps (doubling, then bisection); -1 above 4096 steps."""
    var hi = 1
    while distance(evolve(build(h, t, hi, order, scheme)), exact) >= eps:
        hi *= 2
        if hi > 4096:
            return -1
    var lo = hi // 2
    if lo == 0:
        return 1
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if distance(evolve(build(h, t, mid, order, scheme)), exact) < eps:
            hi = mid
        else:
            lo = mid
    return hi


def main() raises:
    var log_ = CheckLog("chemistry_and_physics/trotter_order8_cost")
    var w = mo8_q21_weights()
    var total = 0.0
    for x in w:
        total += x
    _ = log_.close("mo8_q21 weights sum to 1", total, 1.0, 8e-16)

    var fields: List[Float64] = [0.05, -0.08, 0.03, 0.09, -0.02, -0.06]
    var h = xz_chain(fields)
    var m = len(h.terms)
    var t = 10.0
    var exact = exact_evolve(h, StateVector(6), t)
    _ = log_.check("rotations per step", rotations(build(h, t, 1, 8, "mo8_q21")) == 21 * (2 * m - 1), String(m) + " terms")

    var names: List[String] = ["Strang", "Suzuki-4", "Suzuki-6", "Suzuki-8", "mo8_q21"]
    var orders: List[Int] = [2, 4, 6, 8, 8]
    var schemes: List[String] = ["suzuki", "suzuki", "suzuki", "suzuki", "mo8_q21"]
    var per_step: List[Int] = [1, 5, 25, 125, 21]
    var targets: List[Float64] = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8]
    print("XZ chain L = 6, 16 terms, t = 10: rotations (steps) for state error < target")
    for e in targets:
        var cost = List[Int]()
        var line = "  " + String(e) + ":"
        for i in range(5):
            var r = min_steps(h, t, orders[i], schemes[i], exact, e)
            var c = r * per_step[i] * (2 * m - 1) if r > 0 else -1
            cost.append(c)
            line += "  " + names[i] + " " + (String(c) + " (" + String(r) + ")" if r > 0 else "not reached")
        print(line)
        var best_other = -1
        for i in range(4):
            if cost[i] > 0 and (best_other < 0 or cost[i] < best_other):
                best_other = cost[i]
        if e >= 1e-2:
            _ = log_.check("Suzuki-4 is cheaper than mo8_q21 at " + String(e), cost[1] < cost[4], String(cost[1]) + " vs " + String(cost[4]))
        else:
            _ = log_.check("mo8_q21 is the cheapest at " + String(e), cost[4] > 0 and cost[4] < best_other, String(cost[4]) + " vs " + String(best_other))
        if e == 1e-6:
            _ = log_.check("mo8_q21 at 1e-6 is 1.5x cheaper than the best Suzuki or Strang", Float64(best_other) >= 1.5 * Float64(cost[4]), String(best_other) + " / " + String(cost[4]))
            _ = log_.check("mo8_q21 at 1e-6 is 3x cheaper than Suzuki-8", Float64(cost[3]) >= 3.0 * Float64(cost[4]), String(cost[3]) + " / " + String(cost[4]))
    log_.finish()
