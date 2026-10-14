"""Phase estimation: textbook QPE and iterative QPE (Q# samples).

Reproduces: Q# (Microsoft QDK) - Quantum phase estimation sample, and the iterative phase estimation notebook
Source: https://github.com/microsoft/qdk/blob/main/samples/algorithms/PhaseEstimation.qs (MIT)
Source: https://github.com/microsoft/qdk/tree/main/samples/notebooks (iterative_phase_estimation.ipynb; MIT)

The Q# sample estimates the eigenvalue of the 3-qubit operator
    U = Rzz(pi/3) on (q0, q1), then Rzz(pi/3) on (q1, q2)
on its eigenvector |010> (X on q1) with 6 readout qubits, read as a
little-endian binary fraction times 2 pi. Each Rzz(pi/3) = exp(-i pi/6 ZZ)
contributes e^{+i pi/6} on |010>, so the eigenphase is pi/3 = 1.0472 (page:
"(1.0, pi/3)"); with 6 bits the readout grid is 2 pi/64 = 0.098, so the
estimate is the nearest grid point 11 * 2 pi/64 = 1.0799.

QUMO pieces:
- textbook QPE with `qpe(U, 6, prep)` (controlled-U^(2^j) from the IR functors
  `power()` and `controlled()`, then `iqft`). Rzz is written as CX . RZ . CX
  (Q#'s own decomposition), which the controlled() functor supports;
- iterative QPE (semiclassical Fourier transform, the notebook's method):
  one ancilla, rounds r = 0..5 apply controlled-U^(2^(5-r)) and a phase
  correction P(omega) with omega = -2 pi sum_(l<r) b_l 2^(l-r-1). The
  correction is ONE symbolic parameter "omega", re-bound every round; the
  exact outcome distribution is obtained by branching on every measured bit;
- the same algorithm as a dynamic circuit (mid-circuit measurement, reset
  and c_if-conditioned corrections), sampled with 4,000 trajectory shots.

Checks:
- U|010> = e^{i pi/3}|010> (1e-12);
- the most likely textbook readout is 11, the nearest 6-bit value to (pi/3)/(2 pi) * 64 = 10.67;
- |estimate - pi/3| <= 2 pi/2^6 (page value pi/3);
- the iterative (semiclassical) distribution equals the textbook distribution (1e-12);
- the dynamic circuit's 4,000 shots: most frequent outcome 11 and TVD to exact < 0.05.

Run: .venv/bin/mojo run -I . examples/frameworks/qsharp/phase_estimation.mojo
example-meta: {"framework": "qsharp", "tier": "fast", "timeout": 90, "requires": [], "args": []}
"""

from std.math import pi, cos, sin, floor
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import qpe
from qumo.testing.checks import CheckLog

comptime BITS = 6
comptime ANC = 3
comptime SHOTS = 4000


def rzz(mut c: Circuit, theta: Float64, a: Int, b: Int):
    c.cx(a, b)
    c.rz(theta, b)
    c.cx(a, b)


def sample_operator() -> Circuit:
    var u = Circuit(3, name="U")
    rzz(u, pi / 3.0, 0, 1)
    rzz(u, pi / 3.0, 1, 2)
    return u^


def round_circuit(u: Circuit, power: Int) raises -> Circuit:
    """One IQPE round on system 0..2 + ancilla 3, with symbolic feedback angle omega."""
    var c = Circuit(4, name="iqpe_round")
    var omega = c.parameter("omega")
    c.x(1)
    c.h(ANC)
    var qmap: List[Int] = [0, 1, 2, ANC]
    c.compose_mapped(u.power(power).controlled(3), qmap)
    c.p(omega, ANC)
    c.h(ANC)
    return c^


def feedback(bits: Int, r: Int) -> Float64:
    var w = 0.0
    for l in range(r):
        if ((bits >> l) & 1) == 1:
            w -= 2.0 * pi * (2.0 ** Float64(l - r - 1))
    return w


def main() raises:
    var log = CheckLog("qsharp/phase_estimation")
    var u = sample_operator()
    var prep = Circuit(3)
    prep.x(1)

    # 1. eigenvector check
    var t = prep.copy()
    t.compose(u)
    var sv = evolve(t)
    var dev = abs(sv.re[2] - cos(pi / 3.0)) + abs(sv.im[2] - sin(pi / 3.0))
    print("U|010> amplitude:", sv.re[2], "+", sv.im[2], "i  (e^{i pi/3} =", cos(pi / 3.0), "+", sin(pi / 3.0), "i)")
    _ = log.close("U|010> = e^{i pi/3}|010>", dev, 0.0, 1e-12)

    # 2. textbook QPE
    var c = qpe(u, BITS, prep.copy())
    var probs = evolve(c).probabilities()
    var tb = List[Float64](length=1 << BITS, fill=0.0)
    for i in range(len(probs)):
        tb[i & ((1 << BITS) - 1)] += probs[i]
    var mode = 0
    for k in range(1 << BITS):
        if tb[k] > tb[mode]:
            mode = k
    var estimate = 2.0 * pi * Float64(mode) / Float64(1 << BITS)
    print("textbook QPE:", c.num_qubits, "qubits,", len(c), "gates; P(10) =", tb[10], " P(11) =", tb[11], " P(12) =", tb[12])
    print("  most likely readout", mode, "-> estimate", estimate, " (true pi/3 =", pi / 3.0, ")")
    var grid_pos: Float64 = (pi / 3.0) / (2.0 * pi) * 64.0
    var nearest = Int(floor(grid_pos + 0.5))
    _ = log.check("most likely readout is the nearest 6-bit value", mode == nearest and mode == 11,
                  "mode=" + String(mode) + " nearest=" + String(nearest))
    _ = log.within("|estimate - pi/3| <= 2 pi / 2^6 (page pi/3)", abs(estimate - pi / 3.0), 0.0, 2.0 * pi / 64.0)

    # 3. iterative QPE, exact by branching; one symbolic feedback angle re-bound per round
    var rounds = List[Circuit]()
    for r in range(BITS):
        rounds.append(round_circuit(u, 1 << (BITS - 1 - r)))
    var pref = List[Int]()
    var pw = List[Float64]()
    pref.append(0)
    pw.append(1.0)
    for r in range(BITS):
        var np = List[Int]()
        var nw = List[Float64]()
        for i in range(len(pref)):
            var w: List[Float64] = [feedback(pref[i], r)]
            var ps = evolve(rounds[r].bind(w)).probabilities()
            var p1 = 0.0
            for j in range(len(ps)):
                if ((j >> ANC) & 1) == 1:
                    p1 += ps[j]
            np.append(pref[i])
            nw.append(pw[i] * (1.0 - p1))
            np.append(pref[i] | (1 << r))
            nw.append(pw[i] * p1)
        pref = np^
        pw = nw^
    var it = List[Float64](length=1 << BITS, fill=0.0)
    for i in range(len(pref)):
        it[pref[i]] += pw[i]
    var it_err = 0.0
    for k in range(1 << BITS):
        it_err = max(it_err, abs(it[k] - tb[k]))
    print("iterative QPE (4 qubits, 63 bound round circuits): P(11) =", it[11], " max |P_iter - P_textbook| =", it_err)
    _ = log.close("iterative distribution equals textbook QPE", it_err, 0.0, 1e-12)

    # 4. the same algorithm as one dynamic circuit, sampled
    var d = Circuit(4, BITS, name="iqpe_dynamic")
    d.x(1)
    var qmap: List[Int] = [0, 1, 2, ANC]
    for r in range(BITS):
        if r > 0:
            d.reset(ANC)
        d.h(ANC)
        d.compose_mapped(u.power(1 << (BITS - 1 - r)).controlled(3), qmap)
        for l in range(r):
            d.p(-2.0 * pi * (2.0 ** Float64(l - r - 1)), ANC)
            d.c_if(l, 1)
        d.h(ANC)
        d.measure(ANC, r)
    var res = sample(d, SHOTS, seed=6)
    var tvd = 0.0
    for k in range(1 << BITS):
        var key = String("")
        var b = BITS - 1
        while b >= 0:
            key += "1" if ((k >> b) & 1) == 1 else "0"
            b -= 1
        tvd += abs(res.probability(key) - tb[k])
    tvd *= 0.5
    print("dynamic IQPE (", res.method, "):", SHOTS, "shots, most frequent", res.most_frequent(), " TVD to exact =", tvd)
    _ = log.check("dynamic IQPE most frequent outcome is 11", res.most_frequent() == "001011", res.most_frequent())
    _ = log.within("dynamic IQPE TVD to exact", tvd, 0.0, 0.05)
    log.finish()
