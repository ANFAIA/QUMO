"""QUBO model, Ising conversion, brute force and simulated annealing (qumo.algorithms.qubo)."""

from std.testing import assert_equal, assert_true

from qumo.algorithms.qubo import (
    Qubo,
    QuboSolution,
    ising_to_qubo,
    qubo_brute_force,
    qubo_energy,
    qubo_energy_of_ising,
    qubo_simulated_annealing,
    qubo_to_ising,
)
from qumo.ir.pauli import PauliSum
from qumo.sim.rng import Rng


def _random_qubo(n: Int, seed: UInt64) raises -> Qubo:
    var rng = Rng(seed)
    var q = Qubo(n)
    for i in range(n):
        q.add_linear(i, 2.0 * rng.uniform() - 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            q.add_quadratic(i, j, 2.0 * rng.uniform() - 1.0)
    return q^


def test_ising_matches_energy() raises:
    var q = _random_qubo(6, 11)
    var h = qubo_to_ising(q)
    for m in range(1 << 6):
        var mask = UInt64(m)
        var a = qubo_energy(q, mask)
        var b = qubo_energy_of_ising(h, mask)
        assert_true(abs(a - b) < 1e-10, "mask " + String(m) + ": " + String(a) + " vs " + String(b))


def test_ising_round_trip() raises:
    var q = _random_qubo(6, 11)
    var back = ising_to_qubo(qubo_to_ising(q))
    assert_equal(back.num_vars, 6)
    for m in range(1 << 6):
        var mask = UInt64(m)
        var a = qubo_energy(q, mask)
        var b = qubo_energy(back, mask)
        assert_true(abs(a - b) < 1e-10, "mask " + String(m) + ": " + String(a) + " vs " + String(b))


def test_brute_force_is_the_minimum() raises:
    var q = _random_qubo(10, 5)
    var sol = qubo_brute_force(q)
    var best_mask: UInt64 = 0
    var best = qubo_energy(q, 0)
    for m in range(1 << 10):
        var e = qubo_energy(q, UInt64(m))
        if e < best:
            best = e
            best_mask = UInt64(m)
    assert_equal(sol.num_vars, 10)
    assert_equal(sol.mask, best_mask)
    assert_true(abs(sol.energy - best) < 1e-10, String(sol.energy) + " vs " + String(best))
    for i in range(10):
        assert_equal(sol.bit(i), Int((best_mask >> UInt64(i)) & 1))


def test_brute_force_tie_takes_smallest_mask() raises:
    # Energy is -1 whenever x_0 = 1, so masks 1, 3, 5 and 7 tie: the answer is 1.
    var q = Qubo(3)
    q.add_linear(0, -1.0)
    var sol = qubo_brute_force(q)
    assert_equal(sol.mask, UInt64(1))
    assert_true(abs(sol.energy + 1.0) < 1e-12, String(sol.energy))
    # Every mask ties at 0 for an empty QUBO.
    assert_equal(qubo_brute_force(Qubo(4)).mask, UInt64(0))


def test_annealing_reaches_optimum() raises:
    var q = _random_qubo(12, 21)
    var exact = qubo_brute_force(q)
    var sa = qubo_simulated_annealing(q, 20, 200, 0.1, 10.0, 7)
    assert_true(
        abs(sa.energy - exact.energy) < 1e-9,
        "annealing " + String(sa.energy) + " vs optimum " + String(exact.energy),
    )
    assert_true(abs(sa.energy - qubo_energy(q, sa.mask)) < 1e-12, "energy is not that of the mask")
    var again = qubo_simulated_annealing(q, 20, 200, 0.1, 10.0, 7)
    assert_equal(again.mask, sa.mask)
    assert_true(again.energy == sa.energy, "same seed, different energy")


def test_ising_rejects_non_qubo_terms() raises:
    var h = PauliSum()
    h.add(1.0, 1, 0)  # X_0
    try:
        _ = ising_to_qubo(h)
        raise Error("ising_to_qubo accepted X_0")
    except e:
        assert_true(String(e).startswith("QM-E812: "), String(e))
    var w3 = PauliSum()
    w3.add(1.0, 0, 7)  # Z_0 Z_1 Z_2
    try:
        _ = ising_to_qubo(w3)
        raise Error("ising_to_qubo accepted a weight-3 Z product")
    except e:
        assert_true(String(e).startswith("QM-E812: "), String(e))
    try:
        _ = qubo_energy_of_ising(h, 0)
        raise Error("qubo_energy_of_ising accepted X_0")
    except e:
        assert_true(String(e).startswith("QM-E812: "), String(e))


def test_error_codes() raises:
    try:
        _ = qubo_brute_force(Qubo(25))
        raise Error("brute force accepted 25 variables")
    except e:
        assert_true(String(e).startswith("QM-E811: "), String(e))
    try:
        _ = qubo_simulated_annealing(_random_qubo(4, 1), 0, 10, 0.1, 1.0, 1)
        raise Error("annealing accepted num_reads = 0")
    except e:
        assert_true(String(e).startswith("QM-E822: "), String(e))
    try:
        _ = qubo_simulated_annealing(_random_qubo(4, 1), 5, 10, 2.0, 1.0, 1)
        raise Error("annealing accepted beta_min > beta_max")
    except e:
        assert_true(String(e).startswith("QM-E822: "), String(e))
    try:
        var q = Qubo(3)
        q.add_quadratic(1, 1, 1.0)
        raise Error("add_quadratic accepted i == j")
    except e:
        assert_true(String(e).startswith("QM-E802: "), String(e))
    try:
        var q = Qubo(3)
        q.add_linear(3, 1.0)
        raise Error("add_linear accepted index 3 on 3 variables")
    except e:
        assert_true(String(e).startswith("QM-E802: "), String(e))
    try:
        _ = Qubo(65)
        raise Error("Qubo accepted 65 variables")
    except e:
        assert_true(String(e).startswith("QM-E802: "), String(e))


def test_repeated_pairs_are_merged() raises:
    var q = Qubo(3)
    q.add_quadratic(2, 0, 1.5)
    q.add_quadratic(0, 2, 0.5)
    assert_equal(len(q.quadratic), 1)
    assert_equal(q.quadratic[0][0], 0)
    assert_equal(q.quadratic[0][1], 2)
    assert_true(abs(q.quadratic[0][2] - 2.0) < 1e-12, String(q.quadratic[0][2]))


def main() raises:
    test_ising_matches_energy()
    test_ising_round_trip()
    test_brute_force_is_the_minimum()
    test_brute_force_tie_takes_smallest_mask()
    test_annealing_reaches_optimum()
    test_ising_rejects_non_qubo_terms()
    test_error_codes()
    test_repeated_pairs_are_merged()
    print("test_qilisdk_qubo OK")
