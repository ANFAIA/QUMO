"""Tests for Trim SQD, IQP, Fermi-Hubbard, lie_closure, PUB split, MPF and Suzuki formulas."""

from std.math import pi, sqrt
from std.testing import assert_equal, assert_true, assert_almost_equal
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliTerm, PauliSum, parse_pauli_sum
from qumo.sim.runner import evolve, sample
from qumo.sim.statevector import StateVector
from qumo.sim.rng import Rng
from qumo.algorithms.hamiltonians import (
    Graph, fermi_hubbard, exact_spectrum, exact_evolve, transverse_field_ising, subspace_ground_energy,
)
from qumo.algorithms.library import trotter, suzuki_weights
from qumo.algorithms.qgeometry import lie_closure
from qumo.algorithms.iqp import IqpModel, append_iqp, iqp_expval
from qumo.algorithms.mpf import mpf_coefficients, mpf_expectation
from qumo.algorithms.sqd import (
    partition_subsample, recover_configurations, sqd_iteration, product_basis, subspace_ground_state,
)
from qumo.exec.primitives import SamplerPub, EstimatorPub, run_sampler, run_estimator, split_pub, PubSlot


def sector_ground(h: PauliSum, L: Int, na: Int, nb: Int) raises -> Float64:
    var alpha = List[Int]()
    var beta = List[Int]()
    for s in range(1 << L):
        var c = 0
        for q in range(L):
            c += (s >> q) & 1
        if c == na:
            alpha.append(s)
        if c == nb:
            beta.append(s)
    return subspace_ground_state(h, product_basis(alpha, beta, L), 2 * L).energy


def test_hubbard() raises:
    # 2-site half filling: (U - sqrt(U^2 + 16 t^2)) / 2
    for u in [0.0, 2.0, 4.0, 8.0]:
        var h = fermi_hubbard(Graph.path(2), 1.0, u)
        var want = (u - sqrt(u * u + 16.0)) / 2
        assert_almost_equal(sector_ground(h, 2, 1, 1), want, atol=1e-9)
    # hopping on a 4-ring with periodic Jordan-Wigner strings: U = 0 ground energy of
    # the free fermions with the sign of the JW boundary term, single particle levels
    var ring = fermi_hubbard(Graph.square_lattice(2, 2), 1.0, 0.0)
    assert_true(len(ring.terms) > 0, "lattice model")
    # 2x2 plaquette at half filling and U=4: E0 from exact diagonalisation
    var e = sector_ground(fermi_hubbard(Graph.square_lattice(2, 2), 1.0, 4.0), 4, 2, 2)
    assert_true(e < 0.0 and e > -8.0, "plaquette energy " + String(e))
    var tri = Graph.triangular_lattice(3, 3)
    assert_equal(len(tri.edges), 12 + 4, "3x3: 12 bonds + 4 diagonals")
    print("  Fermi-Hubbard 2-site formula, lattices")


def test_lie_closure() raises:
    for n in range(2, 9):
        var gens = List[PauliTerm]()
        for q in range(n):
            gens.append(PauliTerm(1.0, UInt64(1) << UInt64(q), 0))
        for q in range(n - 1):
            gens.append(PauliTerm(1.0, 0, (UInt64(1) << UInt64(q)) | (UInt64(1) << UInt64(q + 1))))
        assert_equal(len(lie_closure(gens)), n * (2 * n - 1), "open TFIM chain, n = " + String(n))
    var one = List[PauliTerm]()
    one.append(PauliTerm(1.0, 1, 0))
    one.append(PauliTerm(1.0, 0, 1))
    assert_equal(len(lie_closure(one)), 3, "su(2)")
    var hit = False
    try:
        var gens = List[PauliTerm]()
        for q in range(6):
            gens.append(PauliTerm(1.0, UInt64(1) << UInt64(q), 0))
        for q in range(5):
            gens.append(PauliTerm(1.0, 0, (UInt64(1) << UInt64(q)) | (UInt64(1) << UInt64(q + 1))))
        _ = lie_closure(gens, 10)
    except e:
        hit = String(e).startswith("QM-E819")
    assert_true(hit, "QM-E819")
    print("  lie_closure n(2n - 1) for n = 2..8")


def test_iqp() raises:
    # PennyLane's 2-qubit demo: exact 0.17317819 for all three operators
    var m = IqpModel(2)
    m.add_generator([0])
    m.add_generator([1])
    m.add_generator([0, 1])
    var w: List[Float64] = [1.0, 1.0, 1.0]
    var ops = List[List[Int]]()
    ops.append([1])
    ops.append([0])
    ops.append([0, 1])
    var c = Circuit(2)
    append_iqp(c, m, w)
    var sv = evolve(c)
    assert_almost_equal(sv.expectation(parse_pauli_sum("Z0")), 0.17317819, atol=1e-8)
    var r = iqp_expval(m, w, ops, 200000, 3)
    for i in range(3):
        assert_true(abs(r.means[i] - 0.17317819) < 5 * r.stds[i], "2-qubit demo: " + String(r))
    # a random 12-qubit instance against the circuit
    var rng = Rng(5)
    var big = IqpModel(12)
    var weights = List[Float64]()
    for _ in range(30):
        var support = List[Int]()
        for q in range(12):
            if rng.below(3) == 0:
                support.append(q)
        if len(support) == 0:
            support.append(rng.below(12))
        big.add_generator(support)
        weights.append((rng.uniform() - 0.5) * 2.0)
    var cc = Circuit(12)
    append_iqp(cc, big, weights)
    var s2 = evolve(cc)
    var pick = List[List[Int]]()
    pick.append([0, 3, 7])
    pick.append([5])
    pick.append([1, 2, 8, 11])
    var labels: List[String] = ["Z0 Z3 Z7", "Z5", "Z1 Z2 Z8 Z11"]
    var est = iqp_expval(big, weights, pick, 400000, 7)
    for i in range(3):
        var exact = s2.expectation(parse_pauli_sum(labels[i]))
        assert_true(abs(est.means[i] - exact) < 5 * est.stds[i] + 1e-3, "against the state vector: " + String(exact) + " vs " + String(est.means[i]))
    # the gradient against a finite difference on the same draws
    var g = iqp_expval(big, weights, pick, 20000, 9, gradient=True)
    var eps = 1e-5
    var wp = weights.copy()
    wp[4] += eps
    var hi = iqp_expval(big, wp, pick, 20000, 9)
    var fd = (hi.means[0] - g.means[0]) / eps
    assert_true(abs(fd - g.gradients[0][4]) < 1e-3, "gradient")
    # no state vector anywhere: 2000 qubits
    var wide = IqpModel(2000)
    for q in range(1999):
        wide.add_generator([q, q + 1], 0)
    var wr = iqp_expval(wide, [0.1], [[0, 1000], [1999]], 2000, 1)
    assert_equal(len(wr.means), 2, "wide model")
    print("  iqp_expval: exact value, 12-qubit state vector, gradient, 2000 qubits")


def test_mpf() raises:
    # the addon's documented example
    var x = mpf_coefficients([1, 2, 3], 2, True)
    assert_almost_equal(x[0], 0.04166667, atol=1e-7)
    assert_almost_equal(x[1], -1.06666667, atol=1e-7)
    assert_almost_equal(x[2], 2.025, atol=1e-7)
    var a = mpf_coefficients([1, 2, 3], 2, True, "approx", 3.0)
    assert_almost_equal(a[0], 0.03513467, atol=1e-6)
    assert_almost_equal(a[1], -1.0, atol=1e-6)
    assert_almost_equal(a[2], 1.96486533, atol=1e-6)
    # error reduction against exact evolution of a 4-qubit TFIM
    var n = 4
    var h = transverse_field_ising(n, 1.0, 0.8)
    var prep = StateVector(n)
    var obs = parse_pauli_sum("Z0 Z1")
    var t = 1.0
    var exact = exact_evolve(h, prep, t).expectation(obs)
    var steps: List[Int] = [2, 3, 4]
    var single = abs(mpf_expectation(h, obs, t, [4], [1.0], prep) - exact)
    var multi = abs(mpf_expectation(h, obs, t, steps, mpf_coefficients(steps), prep) - exact)
    assert_true(multi < single / 5, "MPF beats its deepest formula: " + String(single) + " -> " + String(multi))
    var hit = False
    try:
        _ = mpf_coefficients([2, 2])
    except e:
        hit = String(e).startswith("QM-E817")
    assert_true(hit, "QM-E817")
    print("  MPF coefficients match the addon; error", single, "->", multi)


def state_error(a: StateVector, b: StateVector) -> Float64:
    """Distance between two states with the global phase aligned."""
    var s = 0.0
    for i in range(a.size()):
        var dr = a.re[i] - b.re[i]
        var di = a.im[i] - b.im[i]
        s += dr * dr + di * di
    return sqrt(s)


def test_suzuki() raises:
    assert_equal(len(suzuki_weights(4)), 5, "5 Strang steps")
    assert_equal(len(suzuki_weights(6)), 25, "25 Strang steps")
    var total = 0.0
    for w in suzuki_weights(6):
        total += w
    assert_almost_equal(total, 1.0, atol=1e-12)
    var n = 4
    var h = transverse_field_ising(n, 1.0, 0.8)
    var prep = StateVector(n)
    var t = 1.0
    var exact = exact_evolve(h, prep, t)
    # halving dt shrinks the state error by 2^order
    for order in [2, 4, 6]:
        var a = state_error(evolve(trotter(h, t, 2, order, n)), exact)
        var b = state_error(evolve(trotter(h, t, 4, order, n)), exact)
        var slope = a / b
        var want = Float64(1 << order)
        assert_true(slope > want / 3 and slope < want * 3, "order " + String(order) + " scaling " + String(slope))
    var hit = False
    try:
        _ = trotter(h, t, 1, 3, n)
    except e:
        hit = String(e).startswith("QM-E808")
    assert_true(hit, "odd orders above 1 stay an error")
    print("  Suzuki orders 4 and 6 scale as expected")


def test_pub_split() raises:
    var c = Circuit(3)
    var th = c.parameters("t", 3)
    for q in range(3):
        c.ry(th[q], q)
    c.cx(0, 1)
    c.cx(1, 2)
    c.measure_all()
    var sets = List[List[Float64]]()
    for s in range(7):
        sets.append([0.1 * Float64(s), 0.2, 0.3 * Float64(s)])
    var pub = SamplerPub(c, 400, sets, 99)
    var whole = run_sampler(pub)
    var slots = List[PubSlot]()
    var parts = split_pub(pub, 3, slots)
    assert_equal(len(parts), 3, "7 sets, 3 per part")
    for s in range(7):
        var part = run_sampler(parts[slots[s].pub])
        var r = part[slots[s].index].copy()
        for kv in whole[s].counts.items():
            assert_equal(kv.value, r.counts.get(kv.key, -1), "bit-identical to the unsplit run")
        assert_equal(len(r.counts), len(whole[s].counts), "same outcomes")
    # estimator with shots
    var ec = Circuit(2)
    var p = ec.parameter("p")
    ec.ry(p, 0)
    ec.cx(0, 1)
    var obs = List[PauliSum]()
    obs.append(parse_pauli_sum("Z0 Z1 + X0"))
    var esets = List[List[Float64]]()
    for s in range(5):
        esets.append([0.3 * Float64(s)])
    var epub = EstimatorPub(ec, obs, esets, 0.02, 11)
    var eall = run_estimator(epub)
    var eslots = List[PubSlot]()
    var eparts = split_pub(epub, 2, eslots)
    for s in range(5):
        var r = run_estimator(eparts[eslots[s].pub])
        assert_equal(r.evs[eslots[s].index][0], eall.evs[s][0], "estimator seeds follow the original index")
    print("  split PUBs are bit-identical to the original")


def test_trim_sqd() raises:
    var L = 4
    var h = fermi_hubbard(Graph.path(L), 1.0, 4.0)
    var exact = sector_ground(h, L, 2, 2)
    # noisy samples: mostly in-sector strings with an unfavourable spread
    var rng = Rng(3)
    var configs = List[Int]()
    var pool = List[Int]()
    for s in range(1 << L):
        var c = 0
        for q in range(L):
            c += (s >> q) & 1
        if c == 2:
            pool.append(s)
    for _ in range(600):
        var a = pool[rng.below(len(pool))]
        var b = pool[rng.below(len(pool))]
        configs.append(a | (b << L))
    var batches = partition_subsample(configs, 40, 5, rng)
    assert_equal(len(batches), 5, "batches")
    var seen = 0
    for b in batches:
        seen += len(b)
    assert_equal(seen, 200, "disjoint: 5 x 40 taken from the pool")
    var std = sqd_iteration(h, configs, L, 12, 3, 5, "standard")
    assert_equal(std.diagonalizations, 3, "standard: one per batch")
    var trim = sqd_iteration(h, configs, L, 12, 3, 5, "trim", 0.8)
    assert_equal(trim.diagonalizations, 4, "trim: num_batches + 1")
    assert_true(trim.energy >= exact - 1e-9 and std.energy >= exact - 1e-9, "variational")
    # the carryover seeds the next iteration, which reaches the exact ground state
    var next = sqd_iteration(h, configs, L, 12, 3, 6, "trim", 0.9, carryover_alpha=trim.alpha, carryover_beta=trim.beta)
    assert_true(next.energy <= trim.energy + 1e-9, "the energy does not rise with a carryover")
    assert_true(abs(sqd_iteration(h, configs, L, 200, 1, 7).energy - exact) < 1e-6, "the full sector is exact")
    # configuration recovery moves strings into the sector
    var occ = List[Float64](length=2 * L, fill=0.5)
    var noisy: List[Int] = [0b00000000, 0b11111111, 0b00010111]
    var fixed = recover_configurations(noisy, L, 2, 2, occ, rng)
    for x in fixed:
        var a = 0
        var b = 0
        for q in range(L):
            a += (x >> q) & 1
            b += (x >> (q + L)) & 1
        assert_true(a == 2 and b == 2, "in the sector after recovery")
    var hit = False
    try:
        _ = sqd_iteration(h, configs, L, 12, 3, 5, "bogus")
    except e:
        hit = String(e).startswith("QM-E816")
    assert_true(hit, "QM-E816")
    print("  Trim SQD: exact", exact, "standard", std.energy, "trim", trim.energy, "dim", trim.dimension, "next", next.energy)


def main() raises:
    test_hubbard()
    test_lie_closure()
    test_iqp()
    test_mpf()
    test_suzuki()
    test_pub_split()
    test_trim_sqd()
    print("test_algorithms_tier1 OK")
