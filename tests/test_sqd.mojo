"""Tests for qumo.algorithms.sqd: subspace ground state (SQD / SKQD classical half).

Checks: agreement with `subspace_ground_energy` (dense and sparse paths) on
real (Hubbard) and complex (Heisenberg + Y field) Hamiltonians and random
subspaces; the full basis and the full symmetry sector reproduce the exact
ground energy; the Ritz vector is an eigenvector (residual) and its
occupations sum to the particle number in a fixed-N sector; product_basis;
error codes.
"""

from std.testing import assert_true, assert_equal

from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.sim.rng import Rng
from qumo.algorithms.hamiltonians import heisenberg_xxz, ground_energy, subspace_ground_energy
from qumo.algorithms.sqd import subspace_ground_state, product_basis


def hubbard_1d(L: Int, t: Float64, u: Float64) raises -> PauliSum:
    """Jordan-Wigner Fermi-Hubbard chain: alpha orbitals 0..L-1, beta L..2L-1 (open)."""
    var s = String()
    for spin in range(2):
        for i in range(L - 1):
            var a = spin * L + i
            var b = a + 1
            s += String(-t / 2.0) + "*X" + String(a) + " X" + String(b) + " + "
            s += String(-t / 2.0) + "*Y" + String(a) + " Y" + String(b) + " + "
    for i in range(L):
        # particle-hole symmetric U (n_up - 1/2)(n_dn - 1/2) = U/4 Z_a Z_b: ground state at half filling
        s += String(u / 4.0) + "*Z" + String(i) + " Z" + String(i + L)
        if i + 1 < L:
            s += " + "
    return parse_pauli_sum(s)


def popcount(x: Int) -> Int:
    var c = 0
    var v = x
    while v > 0:
        c += v & 1
        v >>= 1
    return c


def sector(L: Int, na: Int, nb: Int) -> List[Int]:
    var a = List[Int]()
    var b = List[Int]()
    for s in range(1 << L):
        if popcount(s) == na:
            a.append(s)
        if popcount(s) == nb:
            b.append(s)
    var out = List[Int]()
    for y in b:
        for x in a:
            out.append(x | (y << L))
    return out^


def test_hubbard() raises:
    var L = 4
    var h = hubbard_1d(L, 1.0, 4.0)
    var e0 = ground_energy(h, 2 * L)
    var sec = sector(L, 2, 2)  # 36 states, contains the ground state (half filling, Sz = 0)
    var r = subspace_ground_state(h, sec, 2 * L)
    assert_true(abs(r.energy - e0) < 1e-9, String("sector energy ") + String(r.energy) + " vs " + String(e0))
    assert_true(r.converged, "converged")
    var ntot = 0.0
    for v in r.occupations:
        ntot += v
    assert_true(abs(ntot - 4.0) < 1e-9, "occupations sum to N")
    # up/down symmetric ground state: alpha and beta occupations match
    for i in range(L):
        assert_true(abs(r.occupations[i] - r.occupations[i + L]) < 1e-6, "spin symmetry")
    # random sub-subspaces: agree with the dense projected solver, variational
    var rng = Rng(3)
    for trial in range(5):
        var sub = List[Int]()
        for s in sec:
            if rng.uniform() < 0.6:
                sub.append(s)
        if len(sub) == 0:
            continue
        var ref_ = subspace_ground_energy(h, sub, 2 * L)
        var got = subspace_ground_state(h, sub, 2 * L)
        assert_true(abs(got.energy - ref_) < 1e-9, String("subset ") + String(trial))
        assert_true(got.energy >= e0 - 1e-9, "variational")


def test_complex_and_sparse() raises:
    var n = 10
    var h = heisenberg_xxz(n, 1.0, 1.0, False, 1.0)
    for q in range(n):
        var b = UInt64(1) << UInt64(q)
        h.add(0.7, b, b)  # Y field: complex projected matrix
        h.add(0.3, b, 0)
    var rng = Rng(9)
    var sub = List[Int]()
    for s in range(1 << n):
        if rng.uniform() < 0.3:
            sub.append(s)
    var ref_ = subspace_ground_energy(h, sub, n)  # > 192 states: sparse Lanczos path
    var got = subspace_ground_state(h, sub, n, krylov=30)
    assert_true(abs(got.energy - ref_) < 1e-8, String("complex sparse: ") + String(got.energy) + " vs " + String(ref_))
    assert_true(got.residual < 1e-6, "residual")
    var full = List[Int]()
    for s in range(1 << n):
        full.append(s)
    var e_full = subspace_ground_state(h, full, n)
    assert_true(abs(e_full.energy - ground_energy(h, n)) < 1e-8, "full basis = exact")
    var norm = 0.0
    for k in range(len(full)):
        norm += e_full.re[k] * e_full.re[k] + e_full.im[k] * e_full.im[k]
    assert_true(abs(norm - 1.0) < 1e-12, "normalised Ritz vector")


def test_product_basis_and_errors() raises:
    var a: List[Int] = [3, 5]
    var b: List[Int] = [6]
    var pb = product_basis(a, b, 3)
    assert_equal(len(pb), 2)
    assert_equal(pb[0], 3 | (6 << 3))
    assert_equal(pb[1], 5 | (6 << 3))
    var h = parse_pauli_sum("Z0 + X1")
    var raised = False
    try:
        var dup: List[Int] = [1, 1]
        _ = subspace_ground_state(h, dup, 2)
    except e:
        raised = String(e).startswith("QM-E802")
    assert_true(raised, "duplicates")
    raised = False
    try:
        _ = subspace_ground_state(h, List[Int](), 2)
    except e:
        raised = String(e).startswith("QM-E802")
    assert_true(raised, "empty")
    raised = False
    try:
        var bad: List[Int] = [0, 9]
        _ = subspace_ground_state(h, bad, 2)
    except e:
        raised = String(e).startswith("QM-E802")
    assert_true(raised, "out of range")
    var one: List[Int] = [2]
    var r = subspace_ground_state(h, one, 2)
    assert_true(abs(r.energy - 1.0) < 1e-12, "1x1 subspace")


def main() raises:
    test_hubbard()
    test_complex_and_sparse()
    test_product_basis_and_errors()
    print("test_sqd OK")
