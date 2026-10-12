from std.math import pi
from std.time import perf_counter_ns
from std.testing import assert_equal, assert_true

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.pauli import PauliTerm
from qumo.sim.rng import Rng
from qumo.sim.runner import evolve, sample
from qumo.sim.stabilizer import TableauSimulator, sample_stabilizer, reference_sample


def random_clifford(n: Int, depth: Int, mut rng: Rng) -> Circuit:
    var c = Circuit(n)
    var one_q: List[Int] = [Op.H, Op.S, Op.SDG, Op.SX, Op.SXDG, Op.X, Op.Y, Op.Z, Op.RX, Op.RY, Op.RZ, Op.P]
    var two_q: List[Int] = [Op.CX, Op.CY, Op.CZ, Op.SWAP, Op.ISWAP, Op.ISWAPDG, Op.ECR, Op.RZZ, Op.RXX, Op.RYY]
    for _ in range(depth):
        if n >= 2 and rng.uniform() < 0.4:
            var k = two_q[rng.below(len(two_q))]
            var a = rng.below(n)
            var b = rng.below(n - 1)
            if b >= a:
                b += 1
            var th = Float64(rng.below(7) - 3) * (pi / 2)
            c.add(k, a, b, p0=th)
        else:
            var k = one_q[rng.below(len(one_q))]
            var th = Float64(rng.below(7) - 3) * (pi / 2)
            c.add(k, rng.below(n), p0=th)
    return c^


def random_pauli(n: Int, mut rng: Rng) -> PauliTerm:
    var x: UInt64 = 0
    var z: UInt64 = 0
    for q in range(n):
        var p = rng.below(4)
        var b = UInt64(1) << UInt64(q)
        if p == 1 or p == 2:
            x |= b
        if p == 2 or p == 3:
            z |= b
    return PauliTerm(1.0, x, z)


def check_against_statevector(c: Circuit, mut sim: TableauSimulator, mut rng: Rng, tag: String) raises:
    var sv = evolve(c)
    for _ in range(40):
        var t = random_pauli(c.num_qubits, rng)
        var e_sv = sv.expectation_term(t.x, t.z)
        var e_tab = sim.expectation_term(t)
        if abs(e_sv - Float64(e_tab)) > 1e-9:
            print(c)
            raise Error(tag + ": mismatch for Pauli " + t.label(c.num_qubits) + " sv=" + String(e_sv) + " tableau=" + String(e_tab))


def main() raises:
    var rng = Rng(2024)
    # 1. Random Clifford circuits: every Pauli expectation must match the state vector.
    for trial in range(300):
        var n = 1 + rng.below(6)
        var c = random_clifford(n, 5 + rng.below(40), rng)
        var sim = TableauSimulator(n, UInt64(trial))
        sim.run(c)
        check_against_statevector(c, sim, rng, "trial " + String(trial))

    # 2. Collapse: measuring a random qubit must leave the projected state.
    for trial in range(200):
        var n = 1 + rng.below(5)
        var c = random_clifford(n, 10 + rng.below(30), rng)
        var sim = TableauSimulator(n, UInt64(trial))
        sim.run(c)
        var q = rng.below(n)
        var sv = evolve(c)
        var p1 = sv.prob_one(q)
        var peek = sim.peek_z(q)
        if peek == 0:
            assert_true(abs(p1 - 0.5) < 1e-9, "random outcome must have p=1/2")
        else:
            assert_true(abs(p1 - (0.0 if peek == 1 else 1.0)) < 1e-9, "deterministic outcome mismatch")
        var srng = Rng(UInt64(trial) + 99)
        var m_sv = sv.measure(q, srng)
        var m_tab = sim.measure(q, m_sv)
        assert_equal(m_tab, m_sv)
        for _ in range(30):
            var t = random_pauli(n, rng)
            var e_sv = sv.expectation_term(t.x, t.z)
            var e_tab = sim.expectation_term(t)
            if abs(e_sv - Float64(e_tab)) > 1e-9:
                raise Error("post-measurement mismatch " + t.label(n))

    # 3. Teleportation with feed-forward is deterministic.
    var tp = Circuit(3)
    tp.h(0)
    tp.s(0)  # |psi> = |+i>
    tp.h(1)
    tp.cx(1, 2)
    tp.cx(0, 1)
    tp.h(0)
    tp.measure(0, 0)
    tp.measure(1, 1)
    tp.x(2)
    tp.c_if(1, 1)
    tp.z(2)
    tp.c_if(0, 1)
    tp.sdg(2)
    tp.h(2)
    tp.measure(2, 2)
    var tr = sample_stabilizer(tp, 400, seed=5)
    for k in tr.counts.keys():
        assert_true(k.startswith("0"), "teleported qubit must measure 0")
    assert_true(len(tr.counts) == 4, "the two Bell-measurement bits are uniformly random")

    # 3b. Register condition (c_if_register): X on q2 iff c0 == 1 and c1 == 1.
    var rcnd = Circuit(3)
    rcnd.x(0)
    rcnd.x(1)
    rcnd.measure(0, 0)
    rcnd.measure(1, 1)
    rcnd.x(2)
    var bits: List[Int] = [0, 1]
    var vals: List[Int] = [1, 1]
    rcnd.c_if_register(bits, vals)
    rcnd.measure(2, 2)
    var rres = sample_stabilizer(rcnd, 50, seed=1)
    assert_equal(rres.counts.get("111", 0), 50)

    # 4. Noisy distribution vs state-vector trajectories.
    var nc = Circuit(3)
    nc.h(0)
    nc.cx(0, 1)
    nc.depolarize1(0.2, 1)
    nc.cx(1, 2)
    nc.x_error(0.1, 2)
    nc.measure_all()
    var a = sample_stabilizer(nc, 20000, seed=1)
    var b = sample(nc, 20000, seed=2)
    for k in b.counts.keys():
        assert_true(abs(a.probability(k) - b.probability(k)) < 0.02, "noisy distribution mismatch for " + k)

    # 5. Reference sample: random outcomes forced to 0, deterministic ones kept.
    var rc = Circuit(2)
    rc.h(0)
    rc.measure(0, 0)
    rc.x(1)
    rc.measure(1, 1)
    var ref_ = reference_sample(rc)
    assert_equal(Int(ref_[0]), 0)
    assert_equal(Int(ref_[1]), 1)

    # 6. 1000+ qubits: GHZ with mid-circuit measurements.
    var n = 1200
    var g = Circuit(n)
    g.h(0)
    for q in range(n - 1):
        g.cx(q, q + 1)
    for q in range(n):
        g.measure(q, q)
    var t0 = perf_counter_ns()
    var sim = TableauSimulator(n, 7)
    sim.run(g)
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    for q in range(n):
        assert_equal(Int(sim.records[q]), Int(sim.records[0]))
    print("GHZ-" + String(n) + " tableau run:", dt, "s")
    print("test_stabilizer OK")
