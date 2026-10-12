"""State-vector throughput benchmark (CSV on stdout).

Usage:  bench_sv [nmax=26] [family=layered] [nmin=16] [reps=1]

  family = layered : 10 layers of ry+rz on every qubit + CX brick (default,
                     identical to the original benchmark; matches
                     bench/bench_sv_aer.py)
           qft     : textbook QFT (h, cp, final swaps); most cp gates touch
                     high qubits
           random  : 10 layers of u3 on every qubit + CX on a random perfect
                     matching of the qubits (gates on random, mostly high,
                     qubit pairs). The pseudo-random stream is a 64-bit LCG
                     so the identical circuit can be rebuilt in Python
                     (see docs/SIMULATION.md).

Output columns are unchanged: qubits,depth,gates,seconds,gates_per_sec.
With reps > 1, `seconds` is the median over the repetitions.
"""

from std.time import perf_counter_ns
from std.sys import argv
from std.math import pi
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve


def layered(n: Int, depth: Int) -> Circuit:
    var c = Circuit(n)
    for d in range(depth):
        for q in range(n):
            c.ry(0.1 + 0.01 * Float64(q + d), q)
            c.rz(0.2 + 0.03 * Float64(q * d), q)
        for q in range(d % 2, n - 1, 2):
            c.cx(q, q + 1)
    return c^


def qft(n: Int) -> Circuit:
    var c = Circuit(n)
    var j = n - 1
    while j >= 0:
        c.h(j)
        var k = j - 1
        while k >= 0:
            c.cp(pi / Float64(1 << (j - k)), k, j)
            k -= 1
        j -= 1
    for q in range(n // 2):
        c.swap(q, n - 1 - q)
    return c^


struct Lcg:
    """64-bit LCG (Knuth MMIX constants); output = state >> 33."""

    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Int:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Int(self.s >> 33)

    def angle(mut self) -> Float64:
        return Float64(self.next() % 1000) / 1000.0 * pi


def random_pairs(n: Int, depth: Int, seed: UInt64 = 7) -> Circuit:
    var rng = Lcg(seed)
    var c = Circuit(n)
    for _ in range(depth):
        for q in range(n):
            var th = rng.angle()
            var ph = rng.angle()
            var la = rng.angle()
            c.u(th, ph, la, q)
        var perm = List[Int]()
        for q in range(n):
            perm.append(q)
        var i = n - 1
        while i > 0:
            var j = rng.next() % (i + 1)
            var t = perm[i]
            perm[i] = perm[j]
            perm[j] = t
            i -= 1
        var k = 0
        while k + 1 < n:
            c.cx(perm[k], perm[k + 1])
            k += 2
    return c^


def make(family: String, n: Int) -> Circuit:
    if family == "qft":
        return qft(n)
    if family == "random":
        return random_pairs(n, 10)
    return layered(n, 10)


def main() raises:
    var args = argv()
    var nmin = 16
    var nmax = 26
    var family = String("layered")
    var reps = 1
    if len(args) > 1:
        nmax = Int(String(args[1]))
    if len(args) > 2:
        family = String(args[2])
    if len(args) > 3:
        nmin = Int(String(args[3]))
    if len(args) > 4:
        reps = Int(String(args[4]))
    print("qubits,depth,gates,seconds,gates_per_sec")
    for n in range(nmin, nmax + 1, 2):
        var c = make(family, n)
        var depth = n if family == "qft" else 10
        var times = List[Float64]()
        for _ in range(reps):
            var t0 = perf_counter_ns()
            var sv = evolve(c)
            times.append(Float64(perf_counter_ns() - t0) / 1e9)
            _ = sv.norm_sq()
        sort(times)
        var dt = times[len(times) // 2]
        print(n, ",", depth, ",", len(c), ",", dt, ",", Float64(len(c)) / dt)
