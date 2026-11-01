"""Pre-trajectory sampling and exact noise with the density matrix.

Reproduces: CUDA-Q - PTSBE example, density-matrix-cpu target
Source: https://nvidia.github.io/cuda-quantum/0.14.0/using/examples/ptsbe.html (CUDA-Q docs, Apache-2.0)

At low noise most shots share an error pattern, so drawing the patterns first and
evolving each distinct one once costs far less than one state per shot. For Pauli
noise a pattern's chance does not depend on the state, so the distribution is the
one of per-shot trajectories. The density matrix gives the exact answer to compare
with, and takes amplitude damping, which no Pauli simulator can.

Checks:
- far fewer distinct patterns than shots at p = 1e-3;
- pattern-grouped sampling matches the exact density-matrix distribution;
- amplitude damping: rho_11 = exp(-t/T1), rho_01 = exp(-t/T2) / 2.

Run: .venv/bin/mojo run -I . examples/frameworks/cudaq/trajectory_grouping.mojo
example-meta: {"framework": "cudaq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import exp
from qumo.ir.circuit import Circuit
from qumo.ir.kraus import thermal_relaxation
from qumo.sim.density import density_evolve
from qumo.sim.ptsbe import sample_ptsbe, plan_trajectories
from qumo.testing.checks import CheckLog


def noisy_circuit(n: Int, p: Float64) -> Circuit:
    var c = Circuit(n)
    c.h(0)
    c.t(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
        c.depolarize2(p, q, q + 1)
    for q in range(n):
        c.x_error(p, q)
    c.measure_all()
    return c^


def main() raises:
    var log = CheckLog("trajectory_grouping")
    var c = noisy_circuit(6, 0.001)
    var shots = 100000
    var plan = plan_trajectories(c, shots, 5)
    print(plan)
    _ = log.check("few distinct error patterns", plan.unique < shots // 20, String(plan.unique) + " of " + String(shots))
    var exact = density_evolve(c).probabilities()
    var r = sample_ptsbe(c, shots, 5)
    var tv = 0.0
    for i in range(1 << 6):
        var key = String("")
        var q = 5
        while q >= 0:
            key += "1" if (i >> q) & 1 else "0"
            q -= 1
        tv += abs(r.probability(key) - exact[i])
    _ = log.close("distance from the exact distribution", tv / 2, 0.0, 0.01)

    var t1 = 50.0
    var t2 = 70.0
    var d = Circuit(1)
    d.h(0)
    d.apply_channel(thermal_relaxation(20.0, t1, t2), 0)
    var rho = density_evolve(d)
    _ = log.close("rho_11 = exp(-t/T1) / 2", rho.element(1, 1)[0], 0.5 * exp(-20.0 / t1), 1e-12)
    _ = log.close("rho_01 = exp(-t/T2) / 2", rho.element(0, 1)[0], 0.5 * exp(-20.0 / t2), 1e-12)
    print("purity", rho.purity())
    log.finish()
