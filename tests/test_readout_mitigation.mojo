"""Tests for T-REX (qumo.algorithms.mitigation) and M3 (qumo.qec.mthree)."""

from std.math import pi, sqrt
from std.collections import Dict
from std.testing import assert_equal, assert_true, assert_almost_equal
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.runner import evolve, sample, SampleResult
from qumo.sim.rng import Rng
from qumo.algorithms.mitigation import trex_expectation
from qumo.qec.characterization import ReadoutCalibration, mitigate_counts
from qumo.qec.mthree import mitigate_counts_m3


struct Readout(Copyable):
    """Asymmetric classical readout channel, which Pauli noise cannot express."""

    var p01: Float64
    var p10: Float64

    def __init__(out self, p01: Float64, p10: Float64):
        self.p01 = p01
        self.p10 = p10

    def run(self, c: Circuit, shots: Int, seed: UInt64) raises -> SampleResult:
        var r = sample(c, shots, seed)
        var rng = Rng(seed ^ 0xABCDEF)
        var out = SampleResult(c.num_clbits, shots, "readout")
        for creg in r.memory:
            var o = 0
            for b in range(c.num_clbits):
                var bit = (creg >> b) & 1
                var flip = rng.uniform() < (self.p01 if bit == 0 else self.p10)
                if (bit == 1) != flip:
                    o |= 1 << b
            out.record(o)
        return out^


def test_trex_symmetric() raises:
    # GHZ: <Z0 Z1 Z2 Z3> = 1, with readout flip p the raw value is (1 - 2p)^4
    var c = Circuit(4)
    c.h(0)
    for q in range(3):
        c.cx(q, q + 1)
    var p = 0.05
    var dev = Readout(p, p)

    def run(circ: Circuit, shots: Int, seed: UInt64) raises {imm dev} -> SampleResult:
        return dev.run(circ, shots, seed)

    var h = parse_pauli_sum("Z0 Z1 Z2 Z3")
    var r = trex_expectation(run, c, h, 40000, 3)
    var expected_raw = (1 - 2 * p) * (1 - 2 * p) * (1 - 2 * p) * (1 - 2 * p)
    assert_true(abs(r.raw - expected_raw) < 0.03, "raw scales as (1-2p)^w: " + String(r))
    assert_true(abs(r.value - 1.0) < 4 * r.std + 0.01, "T-REX recovers 1: " + String(r))
    print("  symmetric readout:", r)


def test_trex_asymmetric() raises:
    # asymmetric readout: <Z> of |+> rotated; the twirl symmetrises it
    var c = Circuit(3)
    c.ry(0.7, 0)
    c.cx(0, 1)
    c.ry(1.1, 2)
    var dev = Readout(0.02, 0.12)

    def run(circ: Circuit, shots: Int, seed: UInt64) raises {imm dev} -> SampleResult:
        return dev.run(circ, shots, seed)

    var h = parse_pauli_sum("0.5*Z0 Z1 + Z2 - 0.25*X0 X1")
    var r = trex_expectation(run, c, h, 40000, 9)
    var exact = evolve(c).expectation(h)
    assert_true(abs(r.value - exact) < 5 * r.std + 0.01, "exact " + String(exact) + " vs " + String(r))
    print("  asymmetric readout: exact", exact, "mitigated", r.value, "+-", r.std, "raw", r.raw)


def counts_of(bits: List[String], weights: List[Int]) -> Dict[String, Int]:
    var d = Dict[String, Int]()
    for i in range(len(bits)):
        d[bits[i]] = weights[i]
    return d^


def test_m3_equals_dense() raises:
    # every string observed + full Hamming range: the same linear system as mitigate_counts
    var n = 6
    var cal = ReadoutCalibration(n)
    for q in range(n):
        cal.p01[q] = 0.01 + 0.005 * Float64(q)
        cal.p10[q] = 0.03 + 0.004 * Float64(q)
    var rng = Rng(4)
    var counts = Dict[String, Int]()
    for i in range(1 << n):
        var key = String("")
        var q = n - 1
        while q >= 0:
            key += "1" if (i >> q) & 1 else "0"
            q -= 1
        counts[key] = 1 + rng.below(50) * (1 if i % 7 else 30)
    var dense = mitigate_counts(counts, cal)
    var m3 = mitigate_counts_m3(counts, cal, n)
    assert_equal(len(m3), 64, "all strings")
    for kv in dense.items():
        assert_true(abs(m3[kv.key] - kv.value) < 1e-9, "equal to the dense inverse at " + kv.key)
    print("  M3 equals the dense inverse at full Hamming range")


def test_m3_wide() raises:
    # 40 qubits: mitigate_counts refuses, M3 recovers a GHZ-like distribution
    var n = 40
    var cal = ReadoutCalibration(n)
    var rng = Rng(8)
    for q in range(n):
        cal.p01[q] = 0.005
        cal.p10[q] = 0.01
    var counts = Dict[String, Int]()
    var shots = 40000
    for _ in range(shots):
        var truth = 0 if rng.uniform() < 0.5 else 1
        var key = String("")
        for _ in range(n):
            var u = rng.uniform()
            var flip = u < (0.005 if truth == 0 else 0.01)
            key += "1" if (truth == 1) != flip else "0"
        counts[key] = counts.get(key, 0) + 1
    var refused = False
    try:
        _ = mitigate_counts(counts, cal)
    except e:
        refused = String(e).startswith("QM-E705")
    assert_true(refused, "dense path stops at 24 qubits")
    var m3 = mitigate_counts_m3(counts, cal, 3)
    var zeros = String("0") * n
    var ones = String("1") * n
    var raw0 = Float64(counts.get(zeros, 0)) / Float64(shots)
    var raw1 = Float64(counts.get(ones, 0)) / Float64(shots)
    var v0 = m3.get(zeros, 0.0)
    var v1 = m3.get(ones, 0.0)
    assert_true(abs(v0 + v1 - 1.0) < raw0 + raw1 and v0 + v1 > 0.8, "weight returns: raw " + String(raw0 + raw1) + " -> " + String(v0 + v1))
    print("  40 qubits:", len(counts), "strings, P(0..0) + P(1..1): raw", raw0 + raw1, "M3", v0 + v1)


def main() raises:
    test_trex_symmetric()
    test_trex_asymmetric()
    test_m3_equals_dense()
    test_m3_wide()
    print("test_readout_mitigation OK")
