"""Tests for the algorithm-library extensions: controlled permutation oracles,
modular multiplication and order finding (library.mojo), and subspace
projection / sample-based diagonalisation (hamiltonians.mojo).
Prints `test_algorithms_ext OK`."""

from std.math import pi
from std.testing import assert_true, assert_almost_equal, assert_equal

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import evolve
from qumo.sim.rng import Rng
from qumo.algorithms.library import (
    append_controlled_permutation, modmul_permutation, append_controlled_modmul, order_finding,
    continued_fraction_convergents, order_from_phase, num_mcz_ancillas,
)
from qumo.algorithms.hamiltonians import (
    heisenberg_xxz, transverse_field_ising, dense_matrix, ground_energy,
    projected_hamiltonian, subspace_ground_energy,
)


def _err_code(msg: String) -> String:
    return String(msg[byte=0:7]) if msg.byte_length() >= 7 else msg


def _random_perm(size: Int, mut rng: Rng) -> List[Int]:
    var p = List[Int](length=size, fill=0)
    for i in range(size):
        p[i] = i
    var i = size - 1
    while i > 0:
        var j = rng.below(i + 1)
        var t = p[i]
        p[i] = p[j]
        p[j] = t
        i -= 1
    return p^


def _check_perm_circuit(k: Int, nc: Int, perm: List[Int]) raises -> Int:
    """Exhaustive basis-state check; returns the oracle gate count."""
    var nanc = num_mcz_ancillas(nc + k)
    var n = nc + k + nanc
    var controls = List[Int]()
    for i in range(nc):
        controls.append(i)
    var targets = List[Int]()
    for i in range(k):
        targets.append(nc + i)
    var anc = List[Int]()
    for i in range(nanc):
        anc.append(nc + k + i)
    var oracle = Circuit(n)
    append_controlled_permutation(oracle, controls, targets, perm, anc)
    var cmask_all = (1 << nc) - 1
    for x in range(1 << k):
        # controls all on, and (when there are controls) one pattern with a control off
        var patterns = List[Int]()
        patterns.append(cmask_all)
        if nc > 0:
            patterns.append(cmask_all ^ (1 << (x % nc)))
        for cm in patterns:
            var c = Circuit(n)
            for i in range(nc):
                if ((cm >> i) & 1) == 1:
                    c.x(i)
            for j in range(k):
                if ((x >> j) & 1) == 1:
                    c.x(nc + j)
            c.compose(oracle)
            var sv = evolve(c)
            var y = perm[x] if cm == cmask_all else x
            var idx = cm | (y << nc)
            assert_true(abs(sv.re[idx] - 1.0) < 1e-12 and abs(sv.im[idx]) < 1e-12,
                        "perm oracle k=" + String(k) + " nc=" + String(nc) + " x=" + String(x))
    return oracle.num_unitary_gates()


def test_controlled_permutation() raises:
    var rng = Rng(42)
    for k in range(3, 6):
        for nc in range(0, 3):
            var perm = _random_perm(1 << k, rng)
            var g = _check_perm_circuit(k, nc, perm)
            print("  random permutation k=", k, " controls=", nc, " gates=", g)
    # identity and a single transposition
    var ident = List[Int]()
    for i in range(8):
        ident.append(i)
    assert_equal(_check_perm_circuit(3, 1, ident), 0)
    var swap07: List[Int] = [7, 1, 2, 3, 4, 5, 6, 0]
    _ = _check_perm_circuit(3, 0, swap07)


def test_modmul_and_order() raises:
    var p15 = modmul_permutation(2, 15, 4)
    assert_equal(len(p15), 16)
    assert_equal(p15[1], 2)
    assert_equal(p15[8], 1)
    assert_equal(p15[7], 14)
    assert_equal(p15[15], 15)
    var p21 = modmul_permutation(2, 21, 5)
    assert_equal(len(p21), 32)
    assert_equal(p21[11], 1)
    assert_equal(p21[20], 19)
    assert_equal(p21[25], 25)
    var p21b = modmul_permutation(-19, 21, 5)  # -19 = 2 mod 21
    for i in range(32):
        assert_equal(p21b[i], p21[i])
    # controlled modmul on a 5-bit register is exact for every x < 21
    var gates = _check_perm_circuit(5, 1, modmul_permutation(4, 21, 5))
    print("  controlled x4 mod 21 oracle gates:", gates)
    # continued fractions
    var cf = continued_fraction_convergents(85, 256)
    assert_equal(cf[0][0], 0)
    assert_equal(cf[0][1], 1)
    assert_equal(cf[1][0], 1)
    assert_equal(cf[1][1], 3)
    var last = cf[len(cf) - 1]
    assert_equal(last[0], 85)
    assert_equal(last[1], 256)
    # order from phase: N = 15, a = 2 (r = 4) from 64, 128, 192 out of 256
    assert_equal(order_from_phase(64, 8, 2, 15), 4)
    assert_equal(order_from_phase(128, 8, 2, 15), 4)
    assert_equal(order_from_phase(192, 8, 2, 15), 4)
    assert_equal(order_from_phase(0, 8, 2, 15), 0)
    # N = 21, a = 2 (r = 6): every non-zero peak s/6 * 256 (rounded)
    var peaks: List[Int] = [43, 85, 128, 171, 213]
    for y in peaks:
        assert_equal(order_from_phase(y, 8, 2, 21), 6)
    # order-finding circuit with generic oracles: N = 15 peaks {0, 64, 128, 192}, 1/4 each
    var c15 = order_finding(2, 15, 8)
    var probs = evolve(c15).probabilities()
    var pe = List[Float64](length=256, fill=0.0)
    for i in range(len(probs)):
        pe[i & 255] += probs[i]
    var targets: List[Int] = [0, 64, 128, 192]
    for t in targets:
        assert_almost_equal(pe[t], 0.25, atol=1e-12)
    print("  order_finding(2, 15, 8): qubits", c15.num_qubits, " gates", len(c15), " peaks 0.25 each")


def _random_hamiltonian(n: Int, terms: Int, seed: UInt64) -> PauliSum:
    var rng = Rng(seed)
    var h = PauliSum()
    var mask = (1 << n) - 1
    for _ in range(terms):
        var x = UInt64(Int(rng.next_u64() & UInt64(mask)))
        var z = UInt64(Int(rng.next_u64() & UInt64(mask)))
        h.add(rng.uniform() * 2.0 - 1.0, x, z)
    h.add(0.3, 0, 0)
    return h^


def test_projected_hamiltonian() raises:
    var n = 5
    var h = _random_hamiltonian(n, 14, 9)
    var dm = dense_matrix(h, n)
    var dim = 1 << n
    var states: List[Int] = [17, 3, 30, 0, 12, 5, 22, 9, 31, 26, 14, 7]
    var pm = projected_hamiltonian(h, states, n)
    var m = len(states)
    var has_im = False
    for i in range(m):
        for j in range(m):
            var a = states[i]
            var b = states[j]
            assert_almost_equal(pm[0][i * m + j], dm[0][a * dim + b], atol=1e-12)
            assert_almost_equal(pm[1][i * m + j], dm[1][a * dim + b], atol=1e-12)
            if abs(pm[1][i * m + j]) > 1e-9:
                has_im = True
    assert_true(has_im, "test Hamiltonian should be complex")


def test_subspace_energy() raises:
    # full basis == exact ground energy: dense path (complex, 6 qubits) ...
    var h6 = _random_hamiltonian(6, 20, 3)
    var all6 = List[Int]()
    for i in range(64):
        all6.append(i)
    assert_almost_equal(subspace_ground_energy(h6, all6, 6), ground_energy(h6, 6), atol=1e-10)
    # ... sparse Lanczos path, complex (8 qubits, 256 states) and real (Heisenberg, 10 qubits)
    var h8 = _random_hamiltonian(8, 24, 5)
    var all8 = List[Int]()
    for i in range(256):
        all8.append(i)
    assert_almost_equal(subspace_ground_energy(h8, all8, 8), ground_energy(h8, 8), atol=1e-10)
    var hz = heisenberg_xxz(10)
    var all10 = List[Int]()
    for i in range(1024):
        all10.append(i)
    var e_full = subspace_ground_energy(hz, all10, 10)
    var e_exact = ground_energy(hz, 10)
    assert_almost_equal(e_full, e_exact, atol=1e-10)
    print("  subspace full basis: heisenberg-10", e_full, " exact", e_exact)
    # 8-qubit Heisenberg, random subspaces: variational and monotone in the subspace
    var h = heisenberg_xxz(8)
    var e0 = ground_energy(h, 8)
    var rng = Rng(77)
    var perm = _random_perm(256, rng)
    var prev = 1e300
    var sizes: List[Int] = [40, 120, 200, 256]
    for m in sizes:
        var sub = List[Int]()
        for i in range(m):
            sub.append(perm[i])
        var e = subspace_ground_energy(h, sub, 8)
        assert_true(e >= e0 - 1e-9, "subspace energy must be variational")
        assert_true(e <= prev + 1e-9, "a larger subspace cannot raise the energy")
        prev = e
        print("  heisenberg-8 random subspace m=", m, " E=", e, " exact=", e0)
    assert_almost_equal(prev, e0, atol=1e-10)


def test_errors() raises:
    var codes = List[String]()
    var c = Circuit(4)
    var tg: List[Int] = [0, 1]
    var no_ctrl = List[Int]()
    var bad: List[Int] = [0, 1, 1, 3]
    try:
        append_controlled_permutation(c, no_ctrl, tg, bad)
    except e:
        codes.append(_err_code(String(e)))
    var short: List[Int] = [1, 0]
    try:
        append_controlled_permutation(c, no_ctrl, tg, short)
    except e:
        codes.append(_err_code(String(e)))
    var tg4: List[Int] = [0, 1, 2, 3]
    var ctrl: List[Int] = [4]
    var c5 = Circuit(5)
    try:
        append_controlled_permutation(c5, ctrl, tg4, modmul_permutation(2, 15, 4))  # needs 2 ancillas
    except e:
        codes.append(_err_code(String(e)))
    try:
        _ = modmul_permutation(3, 15, 4)
    except e:
        codes.append(_err_code(String(e)))
    try:
        _ = modmul_permutation(2, 33, 5)
    except e:
        codes.append(_err_code(String(e)))
    var h = heisenberg_xxz(4)
    var dup: List[Int] = [1, 2, 1]
    try:
        _ = projected_hamiltonian(h, dup, 4)
    except e:
        codes.append(_err_code(String(e)))
    var big = List[Int]()
    for i in range(4097):
        big.append(i)
    try:
        _ = subspace_ground_energy(heisenberg_xxz(13), big, 13)
    except e:
        codes.append(_err_code(String(e)))
    var expected: List[String] = ["QM-E802", "QM-E802", "QM-E802", "QM-E801", "QM-E802", "QM-E802", "QM-E807"]
    assert_equal(len(codes), len(expected))
    for i in range(len(expected)):
        assert_equal(codes[i], expected[i])


def main() raises:
    test_controlled_permutation()
    test_modmul_and_order()
    test_projected_hamiltonian()
    test_subspace_energy()
    test_errors()
    print("test_algorithms_ext OK")
