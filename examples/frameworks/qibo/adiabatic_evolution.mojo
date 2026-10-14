"""Adiabatic evolution H(s) = (1 - s) H0 + s H1 towards the TFIM ground state.

Reproduces: Qibo - Adiabatic evolution (linear schedule, Trotter error)
Source: https://qibo.science/qibo/stable/code-examples/tutorials/adiabatic/README.html (Qibo docs, Apache-2.0)

The Qibo tutorial evolves from the ground state of H0 = -sum_i X_i (Qibo's
`hamiltonians.X`) to the ground state of H1 = TFIM, -sum_i (Z_i Z_{i+1} + h X_i)
with periodic boundaries, under the linear schedule s(t) = t / T, on N = 4 qubits.
The page reports, as plots only, that the best overlap among the tested T is at
T = 4. It also reports that its Trotter error (1 - overlap, N = 6) scales as dt^4.
QUMO builds the same pieces from its own API:
- H(s) = H0.scaled(1 - s) + H1.scaled(s), from `PauliSum.__add__` and `scaled`;
- one second-order `trotter` step per time slice dt, with s at the slice midpoint;
- `exact_evolve` slice by slice as the reference;
- `ground_state` (Lanczos) for the target.
Here h = 1 (the tutorial's `--hfield` default), and N = 4.

Checks:
- the final ground-state fidelity rises with T over T = 0.5, 1, 2, 4, 8, and is
  >= 0.9 at T = 8;
- the final energy at T = 8 is within 1% of ground_energy;
- Trotter versus exact slice-by-slice evolution: final-state difference < 1e-3
  at dt = 0.01 (T = 4);
- (as on the page) Trotter infidelity 1 - |<exact|trotter>|^2 for N = 6, T = 1
  scales as dt^4 (log-log slope 4 +/- 0.4).

Run: .venv/bin/mojo run -I . examples/frameworks/qibo/adiabatic_evolution.mojo
example-meta: {"framework": "qibo", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import sqrt, log
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, evolve_into
from qumo.algorithms.hamiltonians import transverse_field_ising, ground_state, exact_evolve
from qumo.algorithms.library import trotter
from qumo.algorithms.linalg import sv_dot
from qumo.ai.checks import CheckLog


def x_field(n: Int) -> PauliSum:
    """Qibo's hamiltonians.X: H0 = -sum_i X_i (ground state |+...+>)."""
    var h = PauliSum()
    for q in range(n):
        h.add(-1.0, UInt64(1) << UInt64(q), 0)
    return h^


def plus_state(n: Int) raises -> StateVector:
    var c = Circuit(n)
    for q in range(n):
        c.h(q)
    return evolve(c)


def fidelity(a: StateVector, b: StateVector) -> Float64:
    var d = sv_dot(a, b)
    return d[0] * d[0] + d[1] * d[1]


def state_diff(a: StateVector, b: StateVector) -> Float64:
    var s = 0.0
    for i in range(a.size()):
        var dr = a.re[i] - b.re[i]
        var di = a.im[i] - b.im[i]
        s += dr * dr + di * di
    return sqrt(s)


def anneal(h0: PauliSum, h1: PauliSum, n: Int, total: Float64, dt: Float64, use_trotter: Bool) raises -> StateVector:
    """Piecewise-constant H(s) at the slice midpoints, from |+...+>."""
    var slices = Int(total / dt + 0.5)
    var step = total / Float64(slices)
    var psi = plus_state(n)
    for k in range(slices):
        var s = (Float64(k) + 0.5) / Float64(slices)
        var hs = h0.scaled(1.0 - s) + h1.scaled(s)
        if use_trotter:
            evolve_into(psi, trotter(hs, step, 1, order=2, num_qubits=n), List[Float64]())
        else:
            psi = exact_evolve(hs, psi, step)
    return psi^


def slope(xs: List[Float64], ys: List[Float64]) -> Float64:
    """Least-squares slope of log(y) against log(x)."""
    var m = Float64(len(xs))
    var sx = 0.0
    var sy = 0.0
    var sxx = 0.0
    var sxy = 0.0
    for i in range(len(xs)):
        var lx = log(xs[i])
        var ly = log(ys[i])
        sx += lx
        sy += ly
        sxx += lx * lx
        sxy += lx * ly
    return (m * sxy - sx * sy) / (m * sxx - sx * sx)


def main() raises:
    var log = CheckLog("qibo/adiabatic_evolution")
    var n = 4
    var h0 = x_field(n)
    var h1 = transverse_field_ising(n, 1.0, 1.0, periodic=True)
    var gs = ground_state(h1, n)
    print("N = 4, H1 = TFIM(h = 1, periodic): ground energy", gs.energy)

    # ---- fidelity versus total time T (Trotter, dt = 0.01) ----
    var times: List[Float64] = [0.5, 1.0, 2.0, 4.0, 8.0]
    var dt = 0.01
    var prev = -1.0
    var rising = True
    var f_last = 0.0
    var e_last = 0.0
    print("  T     fidelity with ground state   final <H1>")
    for t in times:
        var psi = anneal(h0, h1, n, t, dt, True)
        var f = fidelity(gs.state, psi)
        var e = psi.expectation(h1)
        print("  ", t, "  ", f, "  ", e)
        if f <= prev:
            rising = False
        prev = f
        f_last = f
        e_last = e
    _ = log.check("ground-state fidelity rises with T", rising, "fidelity(T=8)=" + String(f_last))
    _ = log.within("ground-state fidelity at T = 8", f_last, 0.9, 1.0 + 1e-12)
    _ = log.close("final energy at T = 8 within 1% of the ground energy", e_last, gs.energy, 0.01 * abs(gs.energy))

    # ---- Trotter versus exact slice evolution ----
    var p_tr = anneal(h0, h1, n, 4.0, dt, True)
    var p_ex = anneal(h0, h1, n, 4.0, dt, False)
    var d = state_diff(p_tr, p_ex)
    print("T = 4, dt = 0.01: ||psi_trotter - psi_exact_slices|| =", d)
    _ = log.within("Trotter vs exact slice evolution at dt = 0.01", d, 0.0, 1e-3)

    # ---- Trotter error scaling, N = 6, T = 1 (page: 1 - overlap ~ dt^4) ----
    var n6 = 6
    var h0b = x_field(n6)
    var h1b = transverse_field_ising(n6, 1.0, 1.0, periodic=True)
    var ref_state = anneal(h0b, h1b, n6, 1.0, 1e-3, False)  # fine exact slices: continuous-time reference
    var dts: List[Float64] = [0.2, 0.1, 0.05, 0.025]
    var infid = List[Float64]()
    print("N = 6, T = 1: dt, 1 - |<exact|trotter>|^2")
    for step in dts:
        var p = anneal(h0b, h1b, n6, 1.0, step, True)
        infid.append(1.0 - fidelity(ref_state, p))
        print("  ", step, "  ", infid[len(infid) - 1])
    var sl = slope(dts, infid)
    print("  log-log slope", sl, "(page: dt^4)")
    _ = log.close("Trotter infidelity scales as dt^4", sl, 4.0, 0.4)
    log.finish()
