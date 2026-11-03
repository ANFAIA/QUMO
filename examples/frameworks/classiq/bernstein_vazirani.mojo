"""Bernstein-Vazirani: one query recovers the hidden string 01101 (13).

Reproduces: Classiq library - Bernstein-Vazirani Algorithm
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/foundational/bernstein_vazirani/bernstein_vazirani.ipynb (Classiq Library, MIT)

The notebook builds H^n, then the predicate f(x) = a.x mod 2 as CX(x_i, f) into an ancilla
in |->, then H^n again, for n = 5 and a = 13, and samples it with 1000 shots (P = 1.0).
This file rebuilds that circuit on QUMO and checks the 1000-shot result, the exact final
state for all 32 five-bit secrets, and the predicate against a classical a.x mod 2.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/bernstein_vazirani.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.time import perf_counter_ns
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample, bitstring
from qumo.testing.checks import CheckLog

comptime N = 5  # STRING_LENGTH in the notebook; qubits x_0..x_4 (x_i = qubit i)
comptime SECRET = 13  # SECRET_INT = 01101 in binary, bit i of the integer is a_i
comptime SHOTS = 1000  # NUM_SHOTS in the notebook


def bv_circuit(n: Int, secret: Int, measure: Bool) raises -> Circuit:
    """The notebook's bv_function: the input register on qubits 0..n-1, the oracle
    ancilla f on qubit n. Structure: H^n on x; within it, X and H on f; the predicate
    CX(x_i, f) for every set bit a_i; then the undo of X and H on f; then H^n on x."""
    var c = Circuit(n + 1, name="bernstein_vazirani")
    for q in range(n):
        c.h(q)
    c.x(n)
    c.h(n)
    for i in range(n):
        if ((secret >> i) & 1) == 1:
            c.cx(i, n)
    c.h(n)
    c.x(n)
    for q in range(n):
        c.h(q)
    if measure:
        for q in range(n):
            c.measure(q, q)
    return c^


def oracle_on_basis(n: Int, secret: Int, x: Int) raises -> Float64:
    """P(f = 1) of the predicate alone, run on the basis state |x> with f = |0>."""
    var c = Circuit(n + 1, name="predicate")
    for q in range(n):
        if ((x >> q) & 1) == 1:
            c.x(q)
    for i in range(n):
        if ((secret >> i) & 1) == 1:
            c.cx(i, n)
    var probs = evolve(c).probabilities()
    var p1 = 0.0
    for j in range(len(probs)):
        if ((j >> n) & 1) == 1:
            p1 += probs[j]
    return p1


def dot_mod2(a: Int, x: Int) -> Int:
    """Classical a.x mod 2: the parity of the bits that are set in both a and x."""
    var v = a & x
    var parity = 0
    while v > 0:
        parity = (parity + (v & 1)) % 2
        v = v >> 1
    return parity


def main() raises:
    var log = CheckLog("classiq/bernstein_vazirani")
    var n = N
    var key = bitstring(SECRET, n)
    print("Bernstein-Vazirani: n =", n, ", secret integer =", SECRET, ", bit string =", key)

    # ---- 1: the notebook result, 1000 shots of the measured circuit ----
    var t0 = perf_counter_ns()
    var r = sample(bv_circuit(n, SECRET, True), SHOTS, seed=UInt64(2024))
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    var p = r.probability(key)
    print("1: sampled", r.shots, "shots in", ms, "ms; most frequent =", r.most_frequent(), ", P =", p)
    _ = log.close("P(01101) = 1.0 over 1000 shots (notebook result)", p, 1.0, 1e-12)
    _ = log.check("the recovered secret is 13 (01101)", r.most_frequent() == key, "got=" + r.most_frequent())

    # ---- 2: exact final state of the unitary circuit, for every 5-bit secret ----
    var worst_p = 0.0
    var worst_amp = 0.0
    for a in range(1 << n):
        var sv = evolve(bv_circuit(n, a, False))
        var probs = sv.probabilities()
        worst_p = max(worst_p, abs(probs[a] - 1.0))
        worst_amp = max(worst_amp, max(abs(sv.re[a] - 1.0), abs(sv.im[a])))
    print("2: all", 1 << n, "secrets: max |P(a) - 1| =", worst_p, ", max |amp(a) - 1| =", worst_amp)
    _ = log.close("exact success probability is 1 for all 32 secrets (max deviation)", worst_p, 0.0, 1e-12)
    _ = log.close("final amplitude of |a> is exactly 1 for all 32 secrets (max deviation)", worst_amp, 0.0, 1e-12)

    # ---- 3: the predicate against a classical a.x mod 2, on all 32 inputs ----
    var worst_f = 0.0
    var ones = 0
    for x in range(1 << n):
        var p1 = oracle_on_basis(n, SECRET, x)
        var f = dot_mod2(SECRET, x)
        ones += f
        worst_f = max(worst_f, abs(p1 - Float64(f)))
    print("3: predicate on 32 inputs, f = 1 on", ones, "of them; max |P(f=1) - a.x mod 2| =", worst_f)
    _ = log.close("CX predicate equals a.x mod 2 on all 32 inputs (max deviation)", worst_f, 0.0, 1e-12)

    log.finish()
