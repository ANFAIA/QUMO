"""Pauli propagation: a 32-qubit Trotter expectation value with its truncation error bound.

Reproduces: PauliPropagation.jl - README quick start (transverse-field Ising Trotter circuit, <Z_16>)
Source: https://github.com/MSRudolph/PauliPropagation.jl/blob/26d7f2d33c0ee986c876727901f7e196b08c55e8/README.md
Framework: QUMO - https://github.com/ANFAIA/QUMO

`propagate_pauli` moves the observable backwards through the circuit (Heisenberg
picture): Clifford gates map one Pauli string to one, a rotation exp(-i t P / 2)
splits an anticommuting string in two. Truncating small coefficients and heavy
strings keeps the list short, and `discarded_abs_sum` is a worst-case bound on the
error this makes (Rudolph et al., arXiv:2505.21606, Theory Box 3).

The circuit is the package's `tfitrottercircuit` with `bricklayertopology(n;
periodic=true)` (src/Circuits/builders.jl and topologies.jl at the commit above):
a layer of RZZ on the brickwork bonds, then (RX on every qubit, RZZ layer) repeated,
ending with an RX layer; every angle dt = 0.1. QUMO's RZZ(t) = exp(-i t ZZ/2) and
RX(t) = exp(-i t X/2) are the package's PauliRotation convention (paulirotations.jl).

Checks:
- 10 qubits, 8 layers, no truncation: equal to the state vector to 1e-10;
- the same circuit truncated (max_weight 6, min_abs_coeff 1e-4): the error is within
  the reported bound, and the bound is not zero;
- the README case (32 qubits, 32 layers, max_weight 6, min_abs_coeff 1e-4) gives the
  README's 0.154596728241 (the README prints 12 digits; tolerance 1e-9).
  There is no exact reference at 32 qubits. The reported bound is printed, not checked:
  here it is about 3.3, more than the whole range of <Z_16> in [-1, 1], so it says
  nothing; the worst-case bound is loose on deep circuits, and the README value is
  the evidence for this case.

Run: .venv/bin/mojo run -I . examples/simulation/pauli_propagation.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 300, "requires": [], "args": []}
"""

from std.time import perf_counter_ns
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import expectation
from qumo.sim.pauli_prop import propagate_pauli, PropagationOptions
from qumo.testing.checks import CheckLog


def brick_bonds(n: Int) -> List[Tuple[Int, Int]]:
    """Bonds of bricklayertopology(1:n; periodic=true), 0-based: odd bonds, then even, ring bond where Julia puts it."""
    var b = List[Tuple[Int, Int]]()
    for i in range(0, n - 1, 2):
        b.append((i, i + 1))
    if n % 2 == 1:
        b.append((n - 1, 0))
    for i in range(1, n - 1, 2):
        b.append((i, i + 1))
    if n % 2 == 0:
        b.append((n - 1, 0))
    return b^


def tfi_trotter(n: Int, layers: Int, dt: Float64) -> Circuit:
    var c = Circuit(n)
    var bonds = brick_bonds(n)
    for b in bonds:
        c.rzz(dt, b[0], b[1])
    for _ in range(layers - 1):
        for q in range(n):
            c.rx(dt, q)
        for b in bonds:
            c.rzz(dt, b[0], b[1])
    for q in range(n):
        c.rx(dt, q)
    return c^


def z_on(q: Int) -> PauliSum:
    var o = PauliSum()
    o.add(1.0, 0, UInt64(1) << UInt64(q))
    return o^


def main() raises:
    var log = CheckLog("simulation/pauli_propagation")
    var trunc = PropagationOptions(min_abs_coeff=1e-4, max_weight=6)

    var small = tfi_trotter(10, 8, 0.1)
    var zs = z_on(5)
    var exact = expectation(small, zs)
    var full = propagate_pauli(small, zs)
    print("10 qubits, 8 layers: state vector", exact, " propagation", full.expectation, " strings", len(full.observable), " peak", full.peak_terms)
    _ = log.close("10 qubits untruncated equals the state vector", full.expectation, exact, 1e-10)
    _ = log.check("nothing was dropped", full.is_exact())
    var cut = propagate_pauli(small, zs, trunc)
    var err = abs(cut.expectation - exact)
    print("  truncated: value", cut.expectation, " |error|", err, " bound", cut.discarded_abs_sum, " dropped", cut.discarded_terms)
    _ = log.check("truncation error is within the reported bound", err <= cut.discarded_abs_sum, "error " + String(err) + ", bound " + String(cut.discarded_abs_sum))
    _ = log.check("the bound is reported", cut.discarded_abs_sum > 0.0 and cut.discarded_terms > 0)

    var big = tfi_trotter(32, 32, 0.1)
    var t0 = perf_counter_ns()
    var r = propagate_pauli(big, z_on(15), trunc)
    var secs = Float64(perf_counter_ns() - t0) * 1e-9
    print("32 qubits, 32 layers:", r.expectation, " strings", len(r.observable), " peak", r.peak_terms, " dropped", r.discarded_terms, " error bound", r.discarded_abs_sum, " time", secs, "s")
    _ = log.close("README value 0.154596728241", r.expectation, 0.154596728241, 1e-9)
    log.finish()
