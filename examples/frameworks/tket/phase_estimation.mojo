"""QPE with controlled unitary boxes (TKET example instance).

Reproduces: TKET - Phase estimation (trivial eigenstate example)
Source: https://docs.quantinuum.com/tket/user-guide/examples/algorithms_and_protocols/phase_estimation.html (pytket user guide, Apache-2.0)

The TKET page builds U = diag(1, 1, e^{i pi/4}, e^{i pi/8}) as a DiagonalBox
on 2 qubits, prepares the eigenstate |11> and runs phase estimation with 4
evaluation qubits: every one of the 1000 shots reads |0001>, so
theta = 1/2^4 = 0.0625 (page). QUMO builds the same estimator with `qpe`
(qumo/algorithms/library.mojo), whose controlled-U^(2^j) blocks come from the
generic IR functors `power()` and `controlled()` (TKET's QControlBox).

TKET labels basis states big-endian (|q0 q1>); QUMO is little-endian, so the
diagonal is written as gates: U = CP(-pi/8)(q0, q1) . P(pi/4)(q0), which gives
phase 0 on |00> and |01>, pi/4 on |10> and pi/8 on |11>.

Checks:
- U equals diag(1, 1, e^{i pi/4}, e^{i pi/8}) on all four basis states (1e-12);
- P(readout 0001 | |11>) = 1 exactly (1e-12) and all 1000 shots read 0001 (page);
- the estimate theta = 1/16 = 0.0625 (page);
- the other three eigenstates give theta = 0, 0 and 1/8 with probability 1 (analytic).

Run: .venv/bin/mojo run -I . examples/frameworks/tket/phase_estimation.mojo
example-meta: {"framework": "tket", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import pi, cos, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, evolve_into, sample
from qumo.algorithms.library import qpe, qpe_phase, bitstring_to_int
from qumo.ai.checks import CheckLog

comptime N_EVAL = 4
comptime SHOTS = 1000


def diagonal_u() -> Circuit:
    """U = diag(1, 1, e^{i pi/4}, e^{i pi/8}) in TKET's |q0 q1> order."""
    var u = Circuit(2, name="diag_u")
    u.p(pi / 4, 0)
    u.cp(-pi / 8, 0, 1)
    return u^


def eval_marginal(c: Circuit, n_eval: Int) raises -> List[Float64]:
    """Exact probability of each evaluation-register value (qubits 0..n_eval-1)."""
    var probs = evolve(c).probabilities()
    var out = List[Float64](length=1 << n_eval, fill=0.0)
    var mask = (1 << n_eval) - 1
    for i in range(len(probs)):
        out[i & mask] += probs[i]
    return out^


def main() raises:
    var log = CheckLog("tket/phase_estimation")
    var u = diagonal_u()

    # 1. The circuit is the page's diagonal matrix (phases in TKET order |q0 q1>).
    var expected_phase: List[Float64] = [0.0, 0.0, pi / 4, pi / 8]  # |00>, |01>, |10>, |11>
    var max_err = 0.0
    for q0 in range(2):
        for q1 in range(2):
            var idx = q0 + 2 * q1  # QUMO little-endian index
            var sv = StateVector.basis(2, idx)
            evolve_into(sv, u, List[Float64]())
            var ph = expected_phase[2 * q0 + q1]  # TKET big-endian index
            max_err = max(max_err, abs(sv.re[idx] - cos(ph)) + abs(sv.im[idx] - sin(ph)))
            for j in range(4):
                if j != idx:
                    max_err = max(max_err, abs(sv.re[j]) + abs(sv.im[j]))
    print("U = CP(-pi/8) . P(pi/4): max deviation from diag(1, 1, e^{i pi/4}, e^{i pi/8}) =", max_err)
    _ = log.close("U equals the page diagonal", max_err, 0.0, 1e-12)

    # 2. The page instance: eigenstate |11>, 4 evaluation qubits.
    var prep = Circuit(2)
    prep.x(0)
    prep.x(1)
    var c = qpe(u, N_EVAL, prep.copy())
    print("QPE circuit:", c.num_qubits, "qubits,", len(c), "gates,", c.num_2q_gates(), "two-qubit gates")
    var marg = eval_marginal(c, N_EVAL)
    _ = log.close("P(0001) for eigenstate |11>", marg[1], 1.0, 1e-12)
    var cm = qpe(u, N_EVAL, prep.copy(), measure=True)
    var r = sample(cm, SHOTS, seed=2024)
    print("counts over", SHOTS, "shots:", r)
    _ = log.check("all shots read 0001 (page)", r.counts.get("0001", 0) == SHOTS,
                  "count(0001)=" + String(r.counts.get("0001", 0)) + " of " + String(SHOTS))
    var theta = qpe_phase(bitstring_to_int(r.most_frequent()), N_EVAL)
    print("theta estimate =", theta, "(page: 0.0625 = 1/16)")
    _ = log.close("theta estimate (page 1/16)", theta, 0.0625, 1e-15)

    # 3. The other eigenstates of the same U (analytic: theta = phase / 2pi).
    var names: List[String] = ["|00>", "|01>", "|10>"]
    var expect_value: List[Int] = [0, 0, 2]  # 0, 0, (pi/4)/(2 pi) * 16 = 2
    for k in range(3):
        var q0 = k // 2
        var q1 = k % 2
        var p = Circuit(2)
        if q0 == 1:
            p.x(0)
        if q1 == 1:
            p.x(1)
        var ck = qpe(u, N_EVAL, p.copy())
        var mk = eval_marginal(ck, N_EVAL)
        print("eigenstate", names[k], ": P(readout", expect_value[k], ") =", mk[expect_value[k]],
              " theta =", qpe_phase(expect_value[k], N_EVAL))
        _ = log.close("P(readout " + String(expect_value[k]) + ") for " + names[k], mk[expect_value[k]], 1.0, 1e-12)
    log.finish()
