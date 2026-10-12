"""Baseline: the circuits of bench/bench_sv.mojo on Qiskit Aer (statevector, CPU, fp64).

Usage:  bench_sv_aer.py [nmax=26] [family=layered] [nmin=16] [reps=1] [--aer-internal]

  family = layered | qft | random  -- gate-for-gate the same circuits as
           bench/bench_sv.mojo (same angles, same qubit order, same 64-bit LCG
           stream for `random`), so QUMO and Aer can be compared per family.

Output (CSV, unchanged): qubits,seconds(aer)
  * seconds = wall time of `AerSimulator.run(c).result()` (as before); with
    reps > 1 it is the median over the repetitions.
  * --aer-internal reports Aer's own experiment time (`result.results[0].
    time_taken`) instead, i.e. without Python and result-conversion overhead
    (the 2^n statevector copy); use it to bound how much of the wall time is
    harness overhead.
"""
import math
import sys
import time

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

MASK64 = (1 << 64) - 1


def layered(n, depth):
    c = QuantumCircuit(n)
    for d in range(depth):
        for q in range(n):
            c.ry(0.1 + 0.01 * (q + d), q)
            c.rz(0.2 + 0.03 * (q * d), q)
        for q in range(d % 2, n - 1, 2):
            c.cx(q, q + 1)
    return c


def qft(n):
    """Textbook QFT exactly as bench_sv.mojo: h, cp(pi / 2^(j-k), k, j), swaps."""
    c = QuantumCircuit(n)
    for j in range(n - 1, -1, -1):
        c.h(j)
        for k in range(j - 1, -1, -1):
            c.cp(math.pi / float(1 << (j - k)), k, j)
    for q in range(n // 2):
        c.swap(q, n - 1 - q)
    return c


class Lcg:
    """64-bit LCG (Knuth MMIX constants); output = state >> 33 (bench_sv.mojo)."""

    def __init__(self, seed):
        self.s = seed & MASK64

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & MASK64
        return self.s >> 33

    def angle(self):
        return float(self.next() % 1000) / 1000.0 * math.pi


def random_pairs(n, depth, seed=7):
    """`depth` layers of u3 on every qubit + CX on a random perfect matching."""
    rng = Lcg(seed)
    c = QuantumCircuit(n)
    for _ in range(depth):
        for q in range(n):
            th = rng.angle()
            ph = rng.angle()
            la = rng.angle()
            c.u(th, ph, la, q)
        perm = list(range(n))
        for i in range(n - 1, 0, -1):
            j = rng.next() % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        for k in range(0, n - 1, 2):
            c.cx(perm[k], perm[k + 1])
    return c


def make(family, n):
    if family == "qft":
        return qft(n)
    if family == "random":
        return random_pairs(n, 10)
    if family == "layered":
        return layered(n, 10)
    raise SystemExit(f"unknown family {family!r}: use layered | qft | random")


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    internal = "--aer-internal" in flags
    nmax = int(args[0]) if len(args) > 0 else 26
    family = args[1] if len(args) > 1 else "layered"
    nmin = int(args[2]) if len(args) > 2 else 16
    reps = int(args[3]) if len(args) > 3 else 1
    sim = AerSimulator(method="statevector", precision="double")
    print("qubits,seconds(aer)")
    for n in range(nmin, nmax + 1, 2):
        c = make(family, n)
        c.save_statevector()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            res = sim.run(c).result()
            wall = time.perf_counter() - t0
            times.append(res.results[0].time_taken if internal else wall)
        times.sort()
        print(n, ",", times[len(times) // 2])  # upper median, as bench_sv.mojo


if __name__ == "__main__":
    main()
