"""Dynamical Lie algebras and lattice models.

Reproduces: Qibo 0.3.6 - lie_closure, FermiHubbard models
Source: https://qibo.science/qibo/stable/api-reference/qibo.html (Qibo docs, Apache-2.0)

The dimension of the algebra a circuit's generators span limits what it can reach
and how its gradients vanish. The open transverse-field Ising chain has dimension
n (2n - 1), far below 4^n, so it does not have a barren plateau.

Checks:
- dimension n (2n - 1) for n = 2 to 8;
- adding one non-local term makes the algebra much larger;
- 2-site Hubbard energy (U - sqrt(U^2 + 16 t^2)) / 2;
- bond counts of the square and triangular lattices.

Run: .venv/bin/mojo run -I . examples/frameworks/qibo/lie_algebra_and_lattices.mojo
example-meta: {"framework": "qibo", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import sqrt
from qumo.ir.pauli import PauliTerm
from qumo.algorithms.qgeometry import lie_closure
from qumo.algorithms.hamiltonians import Graph, fermi_hubbard
from qumo.algorithms.sqd import product_basis, subspace_ground_state
from qumo.testing.checks import CheckLog


def tfim(n: Int) -> List[PauliTerm]:
    var g = List[PauliTerm]()
    for q in range(n):
        g.append(PauliTerm(1.0, UInt64(1) << UInt64(q), 0))
    for q in range(n - 1):
        g.append(PauliTerm(1.0, 0, (UInt64(1) << UInt64(q)) | (UInt64(1) << UInt64(q + 1))))
    return g^


def main() raises:
    var log = CheckLog("lie_algebra_and_lattices")
    for n in range(2, 9):
        _ = log.close("TFIM chain dimension, n=" + String(n), Float64(len(lie_closure(tfim(n)))), Float64(n * (2 * n - 1)), 0.0)
    var more = tfim(5)
    more.append(PauliTerm(1.0, 0, 1 | (1 << 3)))
    var dim = len(lie_closure(more, 2000))
    print("one Z0 Z3 term: dimension", dim)
    _ = log.check("a non-local term enlarges the algebra", dim > 3 * 45)
    var one: List[Int] = [1, 2]
    for u in [2.0, 4.0, 8.0]:
        var e = subspace_ground_state(fermi_hubbard(Graph.path(2), 1.0, u), product_basis(one, one, 2), 4).energy
        _ = log.close("2-site Hubbard, U=" + String(u), e, (u - sqrt(u * u + 16.0)) / 2, 1e-9)
    _ = log.close("square 3x3 bonds", Float64(len(Graph.square_lattice(3, 3).edges)), 12.0, 0.0)
    _ = log.close("triangular 3x3 bonds", Float64(len(Graph.triangular_lattice(3, 3).edges)), 16.0, 0.0)
    log.finish()
