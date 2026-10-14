"""Heisenberg Trotter simulation vs exact evolution, plus the Hadamard test.

Reproduces: CUDA-Q - Hamiltonian simulation (Heisenberg chain) and the Hadamard test
Source: https://nvidia.github.io/cuda-quantum/0.14.0/applications/python/hamiltonian_simulation.html (CUDA-Q docs, Apache-2.0)
Source: https://nvidia.github.io/cuda-quantum/0.14.0/applications/python/hadamard_test.html (CUDA-Q docs, Apache-2.0)

Part A, the page instance: Heisenberg chain Jx = Jy = Jz = 1 on 4 spins (open
chain), initial state |1010> (the page's code flips the even qubits), total
time t = pi in K = 100 first-order Trotter steps (`exp_pauli` per term), with
the survival probability |<psi(0)|psi(t)>|^2 and the energy <H> recorded over
time. The page shows only plots. QUMO: `heisenberg_xxz` + `trotter` (order 1)
against `exact_evolve` (scaled Taylor series) at every 10th step.

Part B, convergence on a larger chain (10 spins, t = 1): the state error
||psi_trotter - psi_exact|| for order 1 (Lie-Trotter) and order 2 (Strang)
over 10..160 steps; the log-log slopes are the formula orders.

Part C, the Hadamard test: Re <psi|O|phi> with an ancilla and controlled
blocks built by the IR functor `controlled()`. Page instance: psi = H(q1)|00>,
phi = X(q0)|00>, O = X0 X1, 100,000 shots: exact 0.7071067690849304, sampled
{0: 85509, 1: 14491} -> 0.71018 +- 0.00157 (page). Its multi-QPU part uses
psi = RY(angle)(q1), phi = RX(theta)(q0) with angle = [0, 1.5, 3.14, 0.7] and
theta = [0.6, 1.2, 2.2, 3.0]; the page's real parts are all about 0
(analytic: Re = 0, Im = -sin(angle/2) sin(theta/2)); QUMO also measures the
imaginary part (S^dagger on the ancilla).

Checks:
- exact evolution conserves <H> (1e-10) and the order-1 Trotter survival
  probability tracks the exact one (max difference < 0.02 at dt = pi/100);
- Part B slopes: about -1 for order 1 and about -2 for order 2 (+/- 0.2);
- Hadamard-test estimates are within 4 sigma of the exact values, and the
  page's sampled 0.71018 is within 4 sigma of QUMO's exact 1/sqrt(2).

Run: .venv/bin/mojo run -I . examples/frameworks/cudaq/hamiltonian_simulation.mojo
example-meta: {"framework": "cudaq", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import pi, sqrt, log, sin
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve_into, sample
from qumo.algorithms.hamiltonians import heisenberg_xxz, exact_evolve
from qumo.algorithms.library import trotter
from qumo.algorithms.linalg import sv_dot
from qumo.testing.checks import CheckLog

comptime HT_SHOTS = 100000


def neel(n: Int) -> StateVector:
    """|1010...>: X on the even qubits (the page's kernel)."""
    var idx = 0
    for q in range(0, n, 2):
        idx |= 1 << q
    return StateVector.basis(n, idx)


def survival(a: StateVector, b: StateVector) -> Float64:
    var d = sv_dot(a, b)
    return d[0] * d[0] + d[1] * d[1]


def state_error(a: StateVector, b: StateVector) -> Float64:
    var s = 0.0
    for i in range(a.size()):
        var dr = a.re[i] - b.re[i]
        var di = a.im[i] - b.im[i]
        s += dr * dr + di * di
    return sqrt(s)


def slope(xs: List[Float64], ys: List[Float64]) -> Float64:
    """Least-squares slope of log(y) against log(x)."""
    var n = Float64(len(xs))
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
    return (n * sxy - sx * sy) / (n * sxx - sx * sx)


def hadamard_test(u: Circuit, imaginary: Bool, shots: Int, seed: UInt64) raises -> Tuple[Float64, Float64]:
    """Estimate Re (or Im) <0|U|0> with one ancilla (qubit n): (value, std error)."""
    var n = u.num_qubits
    var c = Circuit(n + 1, 1)
    c.h(n)
    if imaginary:
        c.sdg(n)
    var cu = u.controlled(n)
    c.compose(cu)
    c.h(n)
    c.measure(n, 0)
    var r = sample(c, shots, seed=seed)
    var p0 = r.probability("0")
    var v = 2.0 * p0 - 1.0
    return (v, sqrt(max(1.0 - v * v, 1e-12) / Float64(shots)))


def main() raises:
    var log = CheckLog("cudaq/hamiltonian_simulation")

    # ---- Part A: the page instance (4 spins, t = pi, K = 100, first order) ----
    var n = 4
    var h = heisenberg_xxz(n, 1.0, 1.0)
    var psi0 = neel(n)
    var t_total = pi
    var k_steps = 100
    var dt = t_total / Float64(k_steps)
    var step = trotter(h, dt, 1, order=1, num_qubits=n)
    var psi_t = psi0.copy()
    var psi_e = psi0.copy()
    var e0 = psi0.expectation(h)
    var max_drift = 0.0
    var max_surv_diff = 0.0
    print("Part A: Heisenberg chain n = 4, |1010>, t = pi, K = 100 first-order steps; <H>(0) =", e0)
    print("  step    t        survival(trotter)  survival(exact)   <H>(trotter)    <H>(exact)")
    for k in range(1, k_steps + 1):
        evolve_into(psi_t, step, List[Float64]())
        psi_e = exact_evolve(h, psi_e, dt)
        if k % 10 == 0:
            var st = survival(psi0, psi_t)
            var se = survival(psi0, psi_e)
            var et = psi_t.expectation(h)
            var ee = psi_e.expectation(h)
            max_drift = max(max_drift, abs(ee - e0))
            max_surv_diff = max(max_surv_diff, abs(st - se))
            print("  ", k, "  ", Float64(k) * dt, "  ", st, "  ", se, "  ", et, "  ", ee)
    _ = log.close("exact evolution conserves <H>", max_drift, 0.0, 1e-10)
    _ = log.within("order-1 survival probability vs exact (max diff)", max_surv_diff, 0.0, 0.02)

    # ---- Part B: error scaling on 10 spins ----
    var nb = 10
    var hb = heisenberg_xxz(nb, 1.0, 1.0)
    var tb = 1.0
    var ref_state = exact_evolve(hb, neel(nb), tb)
    var steps: List[Int] = [10, 20, 40, 80, 160]
    var xs = List[Float64]()
    var e1 = List[Float64]()
    var e2 = List[Float64]()
    print("Part B: n = 10, t = 1: steps, ||psi - psi_exact|| order 1, order 2")
    for s in steps:
        var st1 = neel(nb)
        evolve_into(st1, trotter(hb, tb, s, order=1, num_qubits=nb), List[Float64]())
        var st2 = neel(nb)
        evolve_into(st2, trotter(hb, tb, s, order=2, num_qubits=nb), List[Float64]())
        xs.append(Float64(s))
        e1.append(state_error(st1, ref_state))
        e2.append(state_error(st2, ref_state))
        print("  ", s, "  ", e1[len(e1) - 1], "  ", e2[len(e2) - 1])
    var s1 = slope(xs, e1)
    var s2 = slope(xs, e2)
    print("  log-log slopes: order 1", s1, " order 2", s2)
    _ = log.close("order-1 error slope", s1, -1.0, 0.2)
    _ = log.close("order-2 error slope", s2, -2.0, 0.2)

    # ---- Part C: Hadamard test ----
    # U = U_psi^dagger . O . U_phi, so <00|U|00> = <psi|O|phi>.
    var u = Circuit(2)
    u.x(0)  # U_phi = X(q0)
    u.x(0)  # O = X0 X1
    u.x(1)
    u.h(1)  # U_psi^dagger = H(q1)
    var re = hadamard_test(u, False, HT_SHOTS, 21)
    var exact = 1.0 / sqrt(2.0)
    print("Part C: <psi|X0 X1|phi> estimate", re[0], "+/-", re[1], " exact", exact,
          " (page: exact 0.7071067690849304, sampled 0.71018 +- 0.00157)")
    _ = log.close("Hadamard test Re<psi|O|phi> within 4 sigma", re[0], exact, 4.0 * re[1])
    _ = log.close("page sample 0.71018 within 4 sigma of exact", 0.71018, exact, 4.0 * 0.0015742369065677502)
    var angles: List[Float64] = [0.0, 1.5, 3.14, 0.7]
    var thetas: List[Float64] = [0.6, 1.2, 2.2, 3.0]
    for i in range(4):
        var ui = Circuit(2)
        ui.rx(thetas[i], 0)  # U_phi
        ui.x(0)  # O
        ui.x(1)
        ui.ry(-angles[i], 1)  # U_psi^dagger
        var r_re = hadamard_test(ui, False, HT_SHOTS, UInt64(31 + i))
        var r_im = hadamard_test(ui, True, HT_SHOTS, UInt64(41 + i))
        var im_exact = -sin(angles[i] / 2.0) * sin(thetas[i] / 2.0)
        print("  element", i, ": angle", angles[i], "theta", thetas[i], " Re", r_re[0], "+/-", r_re[1],
              " Im", r_im[0], "+/-", r_im[1], " exact Im", im_exact)
        _ = log.close("element " + String(i) + " Re within 4 sigma of 0", r_re[0], 0.0, 4.0 * r_re[1])
        _ = log.close("element " + String(i) + " Im within 4 sigma", r_im[0], im_exact, 4.0 * r_im[1])
    log.finish()
