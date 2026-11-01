"""Trim SQD on a Fermi-Hubbard chain, with configuration recovery in Mojo.

Reproduces: qiskit-addon-sqd 0.14 - TrimPolicy, partition_subsample, configuration recovery
Source: https://github.com/Qiskit/qiskit-addon-sqd/blob/main/qiskit_addon_sqd/trim.py (Apache-2.0)

SQD diagonalises the Hamiltonian in the span of sampled configurations. Trim SQD
uses the batches only to screen: each is diagonalised, its heaviest strings go
into one merged subspace, and that diagonalisation is the answer. Here noisy
bitstrings first go through configuration recovery, which restores the electron
number; the samples are a stand-in for a QPU.

Checks:
- recovery puts every string in the (2, 2) sector of 4 sites;
- the 2-site Hamiltonian matches (U - sqrt(U^2 + 16 t^2)) / 2;
- a trim iteration costs num_batches + 1 diagonalisations;
- Trim SQD reaches the exact ground energy of the sector.

Run: .venv/bin/mojo run -I . examples/frameworks/qiskit/sqd_trim_hubbard.mojo
example-meta: {"framework": "qiskit", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.bit import pop_count
from std.math import sqrt
from qumo.algorithms.hamiltonians import Graph, fermi_hubbard
from qumo.algorithms.sqd import recover_configurations, sqd_iteration, product_basis, subspace_ground_state
from qumo.sim.rng import Rng
from qumo.testing.checks import CheckLog


def sector(L: Int, n: Int) -> List[Int]:
    var out = List[Int]()
    for s in range(1 << L):
        if Int(pop_count(s)) == n:
            out.append(s)
    return out^


def main() raises:
    var log = CheckLog("sqd_trim_hubbard")
    var two = fermi_hubbard(Graph.path(2), 1.0, 4.0)
    var s2 = sector(2, 1)
    var e2 = subspace_ground_state(two, product_basis(s2, s2, 2), 4).energy
    _ = log.close("2-site Hubbard", e2, (4.0 - sqrt(16.0 + 16.0)) / 2, 1e-9)
    var L = 4
    var h = fermi_hubbard(Graph.path(L), 1.0, 4.0)
    var strings = sector(L, 2)
    var exact = subspace_ground_state(h, product_basis(strings, strings, L), 2 * L).energy
    var rng = Rng(11)
    var noisy = List[Int]()
    for _ in range(800):
        noisy.append(rng.below(1 << (2 * L)))
    var occ = List[Float64](length=2 * L, fill=0.5)
    var fixed = recover_configurations(noisy, L, 2, 2, occ, rng)
    var wrong = 0
    for x in fixed:
        if Int(pop_count(x & 15)) != 2 or Int(pop_count(x >> L)) != 2:
            wrong += 1
    _ = log.close("recovery reaches the sector", Float64(wrong), 0.0, 0.0)
    var r = sqd_iteration(h, fixed, L, 30, 4, 3, "trim", 0.9)
    print(r)
    _ = log.close("num_batches + 1 diagonalisations", Float64(r.diagonalizations), 5.0, 0.0)
    _ = log.within("variational", r.energy, exact - 1e-9, 0.0)
    var next = sqd_iteration(h, fixed, L, 30, 4, 4, "trim", 0.9, carryover_alpha=r.alpha, carryover_beta=r.beta)
    print(next)
    _ = log.close("Trim SQD energy after the carryover", next.energy, exact, 1e-6)
    log.finish()
