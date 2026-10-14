"""Cirq textbook algorithms: teleportation, QFT, phase estimation, Grover.

Reproduces: Cirq - Textbook algorithms in Cirq
Source: https://quantumai.google/cirq/experiments/textbook_algorithms (Cirq docs, CC BY 4.0 text, Apache-2.0 code)

1. Teleportation. The message qubit is prepared with X**0.25 (= RX(pi/4) up to
   a global phase); Alice and Bob share a Bell pair; Alice applies CNOT and H,
   measures, and Bob corrects with CNOT(alice, bob) and CZ(msg, bob). The page
   prints the message and Bob's Bloch vectors, both [0, -0.707, 0.707].
   QUMO runs it twice: with deferred measurement (corrections as quantum
   gates, exact state vector) and with real mid-circuit measurements and
   classically conditioned corrections (`c_if`, trajectory sampling).
2. QFT on 4 qubits. The page compares its hand-made QFT (H + CZ**(1/2^k),
   no final reversal) with `cirq.qft`. QUMO compares `qft(4)` with the DFT
   matrix, `qft(4, do_swaps=False)` with the bit-reversed DFT (the page's
   `without_reverse=True` form) and `iqft(4)` with the inverse DFT.
3. Phase estimation of U = Z**(2 theta) = diag(1, e^{2 pi i theta}).
   Page: theta = 0.234 with 3 bits reads 0.25 on all 10 repetitions;
   theta = 0.123456 with 4 bits on the non-eigenstate |+> reads 0 or 0.125
   (6 / 4 of 10 on the page). QUMO's `qpe` probabilities are compared with the
   analytic Fejer kernel P(m) = sin^2(pi 2^n d) / (4^n sin^2(pi d)),
   d = theta - m / 2^n, and the convergence sweep over n is reproduced.
4. Grover on 2 qubits + 1 ancilla in |->: the page's oracle (X on the zero
   bits, Toffoli onto the ancilla, undo) and one iteration find the marked
   bitstring with certainty (page: Counter({'00': 10})).

Checks: Bloch vectors [0, -0.7071, 0.7071] (1e-12 deferred, 4 sigma sampled);
QFT = DFT (1e-12, three forms); QPE readout 0.25 is the most likely and the
probabilities equal the Fejer kernel (1e-12) for both page instances; the
argmax estimate is within half a bin of theta for n = 1..10; Grover finds each
of the 4 marked strings with probability 1 (1e-12) and on all 10 shots.
Optional: reference/textbook_ref.py reruns the instances with cirq 1.7.0.

Run: .venv/bin/mojo run -I . examples/frameworks/cirq/textbook_algorithms.mojo
example-meta: {"framework": "cirq", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import pi, sqrt, cos, sin
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, evolve_into, sample
from qumo.algorithms.library import qft, iqft, qpe, qpe_phase, grover
from qumo.testing.checks import CheckLog

comptime SHOTS = 20000


# --------------------------------------------------------------------------- teleportation
def teleport(deferred: Bool, basis: String) raises -> Circuit:
    """Teleport circuit: q0 = Message, q1 = Alice, q2 = Bob; basis: '' (no readout), 'X', 'Y' or 'Z' readout of Bob."""
    var c = Circuit(3, name="teleport")
    c.h(1)
    c.cx(1, 2)  # Bell pair Alice-Bob
    c.rx(pi / 4, 0)  # X**0.25 up to a global phase
    c.cx(0, 1)
    c.h(0)
    if deferred:
        c.cx(1, 2)
        c.cz(0, 2)
    else:
        c.measure(0, 0)
        c.measure(1, 1)
        c.x(2)
        c.c_if(1, 1)
        c.z(2)
        c.c_if(0, 1)
    if basis == "X":
        c.h(2)
    elif basis == "Y":
        c.sdg(2)
        c.h(2)
    if basis.byte_length() > 0:
        c.measure(2, 2)
    return c^


def bloch_exact(c: Circuit, q: Int) raises -> List[Float64]:
    var sv = evolve(c)
    var out = List[Float64]()
    for p in ["X", "Y", "Z"]:
        out.append(sv.expectation(parse_pauli_sum(p + String(q))))
    return out^


# --------------------------------------------------------------------------- QFT
def dft_error(c: Circuit, n: Int, sign: Float64, bit_reversed: Bool) raises -> Float64:
    """Largest |<k|C|j> - exp(sign 2 pi i j k / N) / sqrt(N)| over all j, k."""
    var nn = 1 << n
    var worst = 0.0
    for j in range(nn):
        var sv = StateVector.basis(n, j)
        evolve_into(sv, c, List[Float64]())
        for k in range(nn):
            var kk = k
            if bit_reversed:
                kk = 0
                for b in range(n):
                    if ((k >> b) & 1) == 1:
                        kk |= 1 << (n - 1 - b)
            var ang = sign * 2.0 * pi * Float64(j * k) / Float64(nn)
            var er = sv.re[kk] - cos(ang) / sqrt(Float64(nn))
            var ei = sv.im[kk] - sin(ang) / sqrt(Float64(nn))
            worst = max(worst, abs(er) + abs(ei))
    return worst


# --------------------------------------------------------------------------- QPE
def fejer(theta: Float64, m: Int, n: Int) -> Float64:
    var d = theta - Float64(m) / Float64(1 << n)
    var s = sin(pi * d)
    if abs(s) < 1e-15:
        return 1.0
    var num = sin(pi * Float64(1 << n) * d)
    return num * num / (Float64(1 << (2 * n)) * s * s)


def qpe_marginal(theta: Float64, n: Int, plus_state: Bool) raises -> List[Float64]:
    var u = Circuit(1)
    u.p(2.0 * pi * theta, 0)  # cirq.Z**(2 theta)
    var prep = Circuit(1)
    if plus_state:
        prep.h(0)
    else:
        prep.x(0)
    var probs = evolve(qpe(u, n, prep.copy())).probabilities()
    var out = List[Float64](length=1 << n, fill=0.0)
    for i in range(len(probs)):
        out[i & ((1 << n) - 1)] += probs[i]
    return out^


def argmax(xs: List[Float64]) -> Int:
    var best = 0
    for i in range(len(xs)):
        if xs[i] > xs[best]:
            best = i
    return best


# --------------------------------------------------------------------------- Grover
def grover_page(x0: Int, x1: Int) -> Circuit:
    """Cirq page circuit: q0, q1 search register, q2 ancilla, one iteration."""
    var c = Circuit(3, name="grover2")
    c.x(2)
    c.h(2)  # ancilla in |->
    c.h(0)
    c.h(1)
    # oracle: X on the zero bits of x', Toffoli onto the ancilla, undo
    if x0 == 0:
        c.x(0)
    if x1 == 0:
        c.x(1)
    c.ccx(0, 1, 2)
    if x0 == 0:
        c.x(0)
    if x1 == 0:
        c.x(1)
    # diffusion (as on the page): H, X, H(q1) CNOT H(q1), X, H
    c.h(0)
    c.h(1)
    c.x(0)
    c.x(1)
    c.h(1)
    c.cx(0, 1)
    c.h(1)
    c.x(0)
    c.x(1)
    c.h(0)
    c.h(1)
    c.measure(0, 0)
    c.measure(1, 1)
    return c^


def main() raises:
    var log = CheckLog("cirq/textbook_algorithms")

    # 1. Teleportation
    var msg = Circuit(1)
    msg.rx(pi / 4, 0)
    var bm = bloch_exact(msg, 0)
    var bb = bloch_exact(teleport(True, ""), 2)
    print("message Bloch vector:", bm[0], bm[1], bm[2], "| Bob (deferred measurement):", bb[0], bb[1], bb[2],
          "| page: [0, -0.707, 0.707]")
    var bloch_ref: List[Float64] = [0.0, -sqrt(0.5), sqrt(0.5)]
    var d = 0.0
    for k in range(3):
        d = max(d, max(abs(bm[k] - bloch_ref[k]), abs(bb[k] - bloch_ref[k])))
    _ = log.close("teleported Bloch vector (deferred measurement)", d, 0.0, 1e-12)
    var names: List[String] = ["X", "Y", "Z"]
    for k in range(3):
        var r = sample(teleport(False, names[k]), SHOTS, seed=UInt64(10 + k))
        var s = 0.0
        for m in r.memory:
            s += -1.0 if ((m >> 2) & 1) == 1 else 1.0
        var v = s / Float64(SHOTS)
        var sd = sqrt(max(1.0 - bloch_ref[k] * bloch_ref[k], 1e-12) / Float64(SHOTS))
        print("  measured + c_if (", r.method, "): <", names[k], "_bob> =", v)
        _ = log.close("teleported <" + names[k] + "> with mid-circuit measurement", v, bloch_ref[k], 4.0 * sd)

    # 2. QFT on 4 qubits
    var e1 = dft_error(qft(4), 4, 1.0, False)
    var e2 = dft_error(qft(4, do_swaps=False), 4, 1.0, True)
    var e3 = dft_error(iqft(4), 4, -1.0, False)
    print("QFT(4) vs DFT:", e1, "| without reversal vs bit-reversed DFT:", e2, "| IQFT vs inverse DFT:", e3)
    _ = log.close("qft(4) equals the DFT", e1, 0.0, 1e-12)
    _ = log.close("qft(4, do_swaps=False) equals the bit-reversed DFT", e2, 0.0, 1e-12)
    _ = log.close("iqft(4) equals the inverse DFT", e3, 0.0, 1e-12)

    # 3. Phase estimation
    var m1 = qpe_marginal(0.234, 3, False)
    var best = argmax(m1)
    var dk = 0.0
    for m in range(8):
        dk = max(dk, abs(m1[m] - fejer(0.234, m, 3)))
    print("theta = 0.234, n = 3: P(readout) =", m1[0], m1[1], m1[2], m1[3], "... -> most likely",
          qpe_phase(best, 3), "(page: 0.25 on 10 of 10)")
    _ = log.close("most likely QPE estimate (page 0.25)", qpe_phase(best, 3), 0.25, 1e-15)
    _ = log.close("QPE probabilities equal the Fejer kernel (theta 0.234)", dk, 0.0, 1e-12)
    var u = Circuit(1)
    u.p(2.0 * pi * 0.234, 0)
    var prep = Circuit(1)
    prep.x(0)
    var cm = qpe(u, 3, prep.copy(), measure=True)
    var r10 = sample(cm, 10, seed=7)
    print("  10 repetitions:", r10, " (P(all 10 read 0.25) =", m1[2] ** 10, ")")
    var m2 = qpe_marginal(0.123456, 4, True)
    var dk2 = 0.0
    for m in range(16):
        var expect = 0.5 * fejer(0.123456, m, 4) + (0.5 if m == 0 else 0.0)
        dk2 = max(dk2, abs(m2[m] - expect))
    print("theta = 0.123456, n = 4, input |+>: P(0) =", m2[0], " P(0.125) =", m2[2], "(page: 6 and 4 of 10)")
    _ = log.close("non-eigenstate QPE equals 0.5 delta_0 + 0.5 Fejer kernel", dk2, 0.0, 1e-12)
    var worst_bin = 0.0
    for n in range(1, 11):
        var mn = qpe_marginal(0.123456, n, False)
        var est = qpe_phase(argmax(mn), n)
        worst_bin = max(worst_bin, abs(est - 0.123456) * Float64(1 << n))
        print("  n =", n, " estimate", est, " |error| =", abs(est - 0.123456))
    _ = log.within("argmax estimate within half a bin (|error| * 2^n)", worst_bin, 0.0, 0.5)

    # 4. Grover, 2 qubits + ancilla
    var worst_g = 0.0
    var all_shots = True
    for target in range(4):
        var x0 = target & 1
        var x1 = (target >> 1) & 1
        var c = grover_page(x0, x1)
        var probs = evolve(c).probabilities()
        var p = 0.0
        for i in range(len(probs)):
            if (i & 3) == target:
                p += probs[i]
        worst_g = max(worst_g, abs(p - 1.0))
        var r = sample(c, 10, seed=UInt64(50 + target))
        all_shots = all_shots and r.counts.get(r.most_frequent(), 0) == 10 and r.most_frequent() == String(x1) + String(x0)
        var lib = evolve(grover(2, [target])).probabilities()
        var pl = 0.0
        for i in range(len(lib)):
            if (i & 3) == target:
                pl += lib[i]
        worst_g = max(worst_g, abs(pl - 1.0))
        print("Grover marked [", x0, ",", x1, "] (Cirq order): P =", p, " library grover():", pl, " 10 shots:", r)
    _ = log.close("Grover success probability (page circuit and library)", worst_g, 0.0, 1e-12)
    _ = log.check("Grover: all 10 shots read the marked bitstring", all_shots, "4 marked bitstrings")
    log.finish()
