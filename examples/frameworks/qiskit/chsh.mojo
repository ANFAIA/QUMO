"""CHSH inequality with Estimator parameter broadcasting (Qiskit tutorial instance).

Reproduces: Qiskit - CHSH inequality
Source: https://quantum.cloud.ibm.com/docs/tutorials/chsh-inequality (IBM Quantum documentation; tutorial code Apache-2.0, text CC BY-SA 4.0)

The tutorial prepares the Bell state |Phi+> = (|00> + |11>)/sqrt(2) with
H(0), CX(0, 1), rotates qubit 0 by RY(theta) and evaluates two CHSH witnesses
for 21 values theta = linspace(0, 2 pi, 21) in ONE Estimator PUB
(1 circuit x 21 parameter sets x 2 observables):

    S1 = ZZ - ZX + XZ + XX        S2 = ZZ + ZX - XZ + XX

Labels are Qiskit-style (rightmost letter = qubit 0); QUMO's `add_label`
uses the same convention, so the strings are copied verbatim. QUMO's
`EstimatorPub` is the same declarative unit: precision 0 runs the exact
state vector, precision 0.01 runs hardware-faithful shot estimation
(one measurement setting per qubit-wise commuting group, standard errors).

Closed form (analytic): with |psi> = RY(theta)_0 |Phi+>, <ZZ> = <XX> = cos theta,
<ZX> = sin theta and <XZ> = -sin theta, so S1(theta) = 2 (cos theta - sin theta)
and S2(theta) = 2 (cos theta + sin theta); the grid maximum is
2 sqrt(2) cos(pi/20) = 2.7936 at theta = 0.2 pi / 0.3 pi (and +pi). The page's
hardware run (64 Bell pairs) reports a mean peak witness of about 2.75; that
number is quoted, not checked.

Checks:
- exact S1 and S2 equal the closed form at all 21 points (1e-12);
- max |S| over the grid = 2 sqrt(2) cos(pi/20) = 2.7936 (analytic, 1e-12);
- every shot estimate (precision 0.01) is within 4 standard errors of exact;
- the shot-based maximum violates the classical bound: 2 < max |S| <= 2 sqrt(2) + 4 sigma.

Run: .venv/bin/mojo run -I . examples/frameworks/qiskit/chsh.mojo
example-meta: {"framework": "qiskit", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import pi, sqrt, cos, sin
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.exec.primitives import EstimatorPub, run_estimator
from qumo.testing.checks import CheckLog

comptime NUM_THETA = 21
comptime PRECISION = 0.01


def chsh_observables() raises -> List[PauliSum]:
    var s1 = PauliSum()
    s1.add_label(1.0, "ZZ")
    s1.add_label(-1.0, "ZX")
    s1.add_label(1.0, "XZ")
    s1.add_label(1.0, "XX")
    var s2 = PauliSum()
    s2.add_label(1.0, "ZZ")
    s2.add_label(1.0, "ZX")
    s2.add_label(-1.0, "XZ")
    s2.add_label(1.0, "XX")
    var out = List[PauliSum]()
    out.append(s1^)
    out.append(s2^)
    return out^


def main() raises:
    var log = CheckLog("qiskit/chsh")
    var c = Circuit(2, name="chsh")
    var theta = c.parameter("theta")
    c.h(0)
    c.cx(0, 1)
    c.ry(theta, 0)
    var thetas = List[List[Float64]]()
    for k in range(NUM_THETA):
        var t: List[Float64] = [2.0 * pi * Float64(k) / Float64(NUM_THETA - 1)]
        thetas.append(t^)
    var obs = chsh_observables()

    var exact = run_estimator(EstimatorPub(c, obs, thetas, 0.0))
    var shots = run_estimator(EstimatorPub(c, obs, thetas, PRECISION, seed=2024))
    print("one PUB: 1 circuit x", NUM_THETA, "parameter sets x 2 observables")
    print("  exact:", exact.method, " shots:", shots.method, "(", shots.shots_per_set, "shots per setting )")
    print("  theta/pi    S1 exact   S2 exact   S1 shots          S2 shots")
    var err_closed = 0.0
    var max_abs = 0.0
    var max_shot = 0.0
    var max_shot_sigma = 0.0
    var max_z = 0.0
    for k in range(NUM_THETA):
        var t = thetas[k][0]
        var e1 = exact.evs[k][0]
        var e2 = exact.evs[k][1]
        err_closed = max(err_closed, abs(e1 - 2.0 * (cos(t) - sin(t))))
        err_closed = max(err_closed, abs(e2 - 2.0 * (cos(t) + sin(t))))
        max_abs = max(max_abs, max(abs(e1), abs(e2)))
        for o in range(2):
            var d = abs(shots.evs[k][o] - exact.evs[k][o])
            var sd = shots.stds[k][o]
            if sd > 0.0:
                max_z = max(max_z, d / sd)
            elif d > 1e-12:
                max_z = 1e300
            if abs(shots.evs[k][o]) > max_shot:
                max_shot = abs(shots.evs[k][o])
                max_shot_sigma = sd
        print("  ", Float64(k) / 10.0, "   ", e1, " ", e2, " ", shots.evs[k][0], "+-", shots.stds[k][0], " ",
              shots.evs[k][1], "+-", shots.stds[k][1])
    var tsirelson = 2.0 * sqrt(2.0)
    var grid_max = tsirelson * cos(pi / 20.0)
    print("max |S| exact =", max_abs, " analytic 2 sqrt(2) cos(pi/20) =", grid_max, " Tsirelson bound =", tsirelson)
    print("max |S| with shots =", max_shot, "+-", max_shot_sigma, " (page hardware mean peak ~2.75, quoted)")
    _ = log.close("S1 and S2 match the closed form at 21 points", err_closed, 0.0, 1e-12)
    _ = log.close("grid max |S| = 2 sqrt(2) cos(pi/20) (analytic)", max_abs, grid_max, 1e-12)
    _ = log.within("max shot z-score vs exact", max_z, 0.0, 4.0)
    _ = log.within("shot max |S| violates CHSH", max_shot, 2.0 + 1e-9, tsirelson + 4.0 * max_shot_sigma)
    log.finish()
