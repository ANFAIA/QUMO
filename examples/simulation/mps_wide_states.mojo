"""Matrix product states: circuits wider than the state vector, and the truncation error.

Reproduces: Vidal - Efficient classical simulation of slightly entangled quantum computations (local updates of a matrix product state; swaps for non-adjacent gates)
Source: https://arxiv.org/abs/quant-ph/0301063
Framework: QUMO - https://github.com/ANFAIA/QUMO

`qumo.sim.mps.mps_evolve` keeps one tensor per qubit and splits every two-qubit gate
with an SVD (`qumo.core.svd`), so memory and time grow linearly in the number of qubits
and polynomially in the bond dimension. The state vector stops at 32 qubits (QM-E404).

1. An 80-qubit linear cluster state (H on every qubit, CZ on neighbours). Its stabilizers
   K_q = Z_{q-1} X_q Z_{q+1} have expectation 1 and every X_q in the bulk has 0; every
   bond has dimension 2.
2. A 60-qubit GHZ state made with CX(0, q) for every q, so every gate but the first is
   non-adjacent and runs through swaps. Samples are only 0...0 and 1...1, half each.
3. A 12-qubit random brickwork circuit at bond dimensions 2 to 64, against the dense
   state vector: the distance ||psi - phi|| never exceeds the reported `error_bound`,
   ||phi||^2 = 1 - discarded_weight, and at 64 = 2^6 (the largest Schmidt rank of 12
   qubits) nothing is truncated and the states agree to rounding.

Checks:
- cluster stabilizers = 1, bulk <X_q> = 0, bond dimensions 2, no truncation;
- GHZ samples only 0...0 and 1...1, each within 5 sigma of half;
- distance <= error_bound and the norm identity at every bond dimension; exact at 64.

Run: .venv/bin/mojo run -I . examples/simulation/mps_wide_states.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 180, "requires": [], "args": []}
"""

from std.math import sqrt, pi
from qumo.core.rng import Rng
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve
from qumo.sim.mps import mps_evolve
from qumo.testing.checks import CheckLog


def main() raises:
    var log = CheckLog("simulation/mps_wide_states")

    # 1. Cluster state on 80 qubits.
    var n = 80
    var c = Circuit(n)
    for q in range(n):
        c.h(q)
    for q in range(n - 1):
        c.cz(q, q + 1)
    var st = mps_evolve(c, max_bond=8)
    var worst_k = 0.0
    var worst_x = 0.0
    for q in range(1, n - 1, 7):
        var ops = List[Int](length=n, fill=0)
        ops[q - 1] = 3
        ops[q] = 1
        ops[q + 1] = 3
        worst_k = max(worst_k, abs(st.expectation_ops(ops) - 1.0))
        var xo = List[Int](length=n, fill=0)
        xo[q] = 1
        worst_x = max(worst_x, abs(st.expectation_ops(xo)))
    _ = log.close("80-qubit cluster stabilizers Z X Z equal 1 (max deviation)", worst_k, 0.0, 1e-12)
    _ = log.close("80-qubit cluster bulk X_q equals 0 (max deviation)", worst_x, 0.0, 1e-12)
    var all2 = True
    for d in st.bond_dims():
        all2 = all2 and d == 2
    _ = log.check("every cluster bond has dimension 2", all2)
    _ = log.close("cluster state needs no truncation (error bound)", st.error_bound, 0.0, 0.0)

    # 2. GHZ on 60 qubits through long-range CX gates.
    var m = 60
    var g = Circuit(m)
    g.h(0)
    for q in range(1, m):
        g.cx(0, q)
    var gs = mps_evolve(g, max_bond=8)
    var shots = 4000
    var counts = gs.sample(shots, 2026)
    var zeros = String("")
    var ones = String("")
    for _ in range(m):
        zeros += "0"
        ones += "1"
    var n0 = counts.get(zeros, 0)
    var n1 = counts.get(ones, 0)
    _ = log.check("60-qubit GHZ samples are only 0...0 and 1...1", n0 + n1 == shots, String(n0 + n1) + " of " + String(shots))
    var half = Float64(shots) / 2
    var sigma = sqrt(Float64(shots) * 0.25)
    _ = log.within("60-qubit GHZ count of 0...0", Float64(n0), half - 5 * sigma, half + 5 * sigma)
    print("  GHZ(60): largest bond", gs.max_bond_seen, " counts", n0, n1)

    # 3. Truncation on 12 qubits against the state vector.
    var k = 12
    var b = Circuit(k)
    var rng = Rng(7)
    for layer in range(10):
        for q in range(k):
            b.u(2 * pi * rng.uniform(), 2 * pi * rng.uniform(), 2 * pi * rng.uniform(), q)
        var q0 = layer % 2
        while q0 + 1 < k:
            b.cx(q0, q0 + 1)
            b.rzz(2 * pi * rng.uniform(), q0, q0 + 1)
            q0 += 2
    var sv = evolve(b)
    var all_ok = True
    var exact_d = 1.0
    var exact_bound = 1.0
    print("  chi   ||psi - phi||   error_bound   1 - ||phi||^2   discarded_weight")
    for chi in [2, 4, 8, 16, 32, 64]:
        var t = mps_evolve(b, max_bond=chi, cutoff=0.0)
        var d2 = 0.0
        for i in range(sv.size()):
            var a = t.amplitude_index(i)
            d2 += (a[0] - sv.re[i]) ** 2 + (a[1] - sv.im[i]) ** 2
        var d = sqrt(d2)
        var lost = 1.0 - t.norm_sq()
        print("  ", chi, "  ", d, "  ", t.error_bound, "  ", lost, "  ", t.discarded_weight)
        all_ok = all_ok and d <= t.error_bound + 1e-10 and abs(lost - t.discarded_weight) <= 1e-10
        if chi == 64:
            exact_d = d
            exact_bound = t.error_bound
    _ = log.check("distance <= error_bound and 1 - norm^2 = discarded_weight at every bond dimension", all_ok)
    _ = log.close("bond dimension 64 is exact on 12 qubits (distance)", exact_d, 0.0, 1e-10)
    _ = log.close("bond dimension 64 reports no truncation", exact_bound, 0.0, 0.0)
    log.finish()
