"""Tests for qumo.sim.branching against the state vector, and past its limit."""

from std.math import pi
from std.testing import assert_equal, assert_true, assert_almost_equal
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import evolve, sample
from qumo.sim.rng import Rng
from qumo.sim.branching import BranchingSimulator, sample_branching, _Key


def random_circuit(n: Int, gates: Int, seed: UInt64, max_t: Int) -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n)
    var t_left = max_t
    for _ in range(gates):
        var r = rng.below(11)
        var q = rng.below(n)
        var u = (q + 1 + rng.below(n - 1)) % n
        if r == 0: c.h(q)
        elif r == 1: c.s(q)
        elif r == 2: c.sx(q)
        elif r == 3: c.cx(q, u)
        elif r == 4: c.cz(q, u)
        elif r == 5: c.swap(q, u)
        elif r == 6: c.y(q)
        elif t_left > 0:
            t_left -= 1
            if r == 7: c.t(q)
            elif r == 8: c.rz(rng.uniform() * 3.0, q)
            elif r == 9: c.rx(rng.uniform() * 3.0, q)
            else: c.rzz(rng.uniform() * 3.0, q, u)
    return c^


def test_expectations_match_statevector() raises:
    for seed in range(8):
        var n = 6 + seed % 5
        var c = random_circuit(n, 60, UInt64(seed + 1), 8)
        var sim = BranchingSimulator(n, 1)
        var rng = Rng(7)
        sim.run(c, rng)
        var sv = evolve(c)
        var pick = Rng(UInt64(100 + seed))
        for _ in range(30):
            var x = _Key.zero()
            var z = _Key.zero()
            var xw = UInt64(0)
            var zw = UInt64(0)
            for q in range(n):
                var l = pick.below(4)
                if l == 1 or l == 2:
                    x.flip(q)
                    xw |= UInt64(1) << UInt64(q)
                if l == 3 or l == 2:
                    z.flip(q)
                    zw |= UInt64(1) << UInt64(q)
            var a = sim.expectation_pauli(x, z)
            var b = sv.expectation_term(xw, zw)
            assert_true(abs(a - b) < 1e-9, "Pauli expectation " + String(a) + " vs " + String(b) + " seed " + String(seed))
    print("  random Clifford + rotations: every Pauli expectation matches the state vector")


def test_distribution_matches_statevector() raises:
    for seed in range(3):
        var c = random_circuit(8, 50, UInt64(seed + 20), 6)
        c.measure_all()
        var a = sample(c, 8000, 1)
        var b = sample_branching(c, 8000, 2)
        var tv = 0.0
        for kv in a.counts.items():
            tv += abs(a.probability(kv.key) - b.probability(kv.key))
        for kv in b.counts.items():
            if kv.key not in a.counts:
                tv += b.probability(kv.key)
        assert_true(tv / 2 < 0.08, "TV " + String(tv / 2))
    print("  sampled distribution matches")


def test_measurement_does_not_grow_branches() raises:
    # GHZ with a T: two branches stay two while every qubit is measured
    var n = 30
    var c = Circuit(n)
    c.h(0)
    c.t(0)
    c.h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    for q in range(n):
        c.h(q)
    var sim = BranchingSimulator(n, 3)
    var rng = Rng(3)
    sim.run(c, rng)
    assert_equal(sim.num_branches(), 2, "one T gives two branches")
    for q in range(n):
        _ = sim.measure(q, rng)
        assert_true(sim.num_branches() <= 2, "measurement keeps the list small")
    print("  measuring 30 qubits keeps 2 branches")


def test_t_echo_past_the_limit() raises:
    # QDK's notebook program: T / Tdg echo on wide GHZ states; data qubits return to 0
    var n = 111
    var c = Circuit(n)
    c.h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    for q in range(0, n, 11):
        c.h(q)
        c.t(q)
        c.h(q)
    for q in range(0, n, 11):
        c.h(q)
        c.tdg(q)
        c.h(q)
    for q in range(n - 2, -1, -1):
        c.cx(q, q + 1)
    c.h(0)
    # 111 qubits, 11 T gates: 62 classical bits is the sampler's limit
    for q in range(62):
        c.measure(q, q)
    var r = sample_branching(c, 64, 5)
    assert_equal(r.method, "stabilizer:branching", "method")
    assert_equal(len(r.counts), 1, "the echo is deterministic")
    assert_equal(r.counts.get(r.most_frequent(), 0), 64, "all zero")
    assert_true(r.most_frequent().find("1") < 0, "data qubits back to 0")
    print("  111 qubits, T / Tdg echo returns to 0")


def test_feed_forward_t_injection() raises:
    # teleport a T: |T> on q1 from an ancilla, measured with S correction
    var c = Circuit(2, 2)
    c.h(1)
    c.t(1)
    c.h(0)
    c.cx(0, 1)
    c.measure(1, 0)
    c.s(0)
    c.c_if(0, 1)
    c.h(0)
    c.measure(0, 1)
    var a = sample(c, 20000, 1)
    var b = sample_branching(c, 20000, 2)
    var pa = 0.0
    var pb = 0.0
    for kv in a.counts.items():
        if kv.key.startswith("1"):
            pa += a.probability(kv.key)
    for kv in b.counts.items():
        if kv.key.startswith("1"):
            pb += b.probability(kv.key)
    assert_true(abs(pa - pb) < 0.015, "feed-forward: " + String(pa) + " vs " + String(pb))
    print("  feed-forward agrees with the state vector")


def test_branch_limit() raises:
    var c = Circuit(24)
    for q in range(24):
        c.h(q)
        c.t(q)
        c.h(q)
    var sim = BranchingSimulator(24, 1)
    var rng = Rng(1)
    var hit = False
    try:
        sim.run(c, rng)
    except e:
        hit = String(e).startswith("QM-E412")
    assert_true(hit, "QM-E412 at 2^20 branches")
    print("  branch limit")


def main() raises:
    test_expectations_match_statevector()
    test_distribution_matches_statevector()
    test_measurement_does_not_grow_branches()
    test_t_echo_past_the_limit()
    test_feed_forward_t_injection()
    test_branch_limit()
    print("test_branching OK")
