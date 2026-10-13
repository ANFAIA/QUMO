"""Cache-blocked execution with qubit reordering (qumo.sim.runner) against
plain gate-by-gate application.

Covers: random circuits on 19-22 qubits with gates on high, low and mixed
qubits (every 1q/2q/3q unitary opcode); probabilities, sampling and
expectation values; the final-permutation restore (identity, only high bits,
only low bits including in-vector bits, mixed); the in-vector-bit (lane)
cost accounting (`vector_pairs`, pinned re-selection); non-unitary ops skipped by
the plan path; the n < 19 path; every block size nb on small states; and the
QM-E407 / QM-E408 argument checks.
"""

from std.math import pi
from std.collections import Dict
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_is_unitary
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import (
    evolve, sample, apply_instruction, build_plan, execute_plan,
    execute_plan_reordered, execute_plan_reordered_stats, ReorderStats,
    BLOCK_BITS, MIN_BLOCKS_LOG2,
)
from qumo.sim.rng import Rng
from std.testing import assert_true, assert_equal, assert_almost_equal


def reference(c: Circuit) -> StateVector:
    """Gate-by-gate application of the unitary ops (no fusion, no blocking)."""
    var sv = StateVector(c.num_qubits)
    for g in c.ops:
        if op_is_unitary(g.kind):
            apply_instruction(sv, g, List[Float64]())
    sv.apply_global_phase(c.global_phase)
    return sv^


def max_diff(a: StateVector, b: StateVector) -> Float64:
    var d = 0.0
    for i in range(a.size()):
        d = max(d, abs(a.re[i] - b.re[i]) + abs(a.im[i] - b.im[i]))
    return d


def check_same(a: StateVector, b: StateVector, what: String) raises:
    var d = max_diff(a, b)
    var f = a.fidelity(b)
    if d > 1e-9 or abs(f - 1.0) > 1e-9:
        raise Error("blocked != reference: " + what + " max_diff=" + String(d) + " fidelity=" + String(f))


def pick(n: Int, mode: Int, mut rng: Rng) -> Int:
    """Qubit choice. Mode 0: uniform, 1: high qubits, 2: low qubits (incl. the
    in-vector bits 0..2), 3: mixed extremes."""
    if mode == 1:
        return n - 1 - rng.below(min(6, n))
    if mode == 2:
        return rng.below(min(6, n))
    if mode == 3:
        var e: List[Int] = [0, 1, 2, min(7, n - 1), n - 3, n - 2, n - 1]
        return e[rng.below(len(e))]
    return rng.below(n)


def random_circuit(n: Int, ngates: Int, mode: Int, mut rng: Rng) -> Circuit:
    var c = Circuit(n)
    # spread amplitude everywhere first so every gate acts on a dense state
    for q in range(n):
        c.ry(0.3 + 0.05 * Float64(q), q)
    for _ in range(ngates):
        var k = rng.below(Op.CCZ + 1)  # every unitary opcode
        var a = pick(n, mode, rng)
        var b = pick(n, mode, rng)
        while b == a:
            b = rng.below(n)
        var d = pick(n, mode, rng)
        while d == a or d == b:
            d = rng.below(n)
        c.add(k, a, b, d, rng.uniform() * 6 - 3, rng.uniform() * 6 - 3, rng.uniform() * 6 - 3)
    return c^


def stats_of(c: Circuit) raises -> ReorderStats:
    """Run the reordered executor exactly as evolve() does and return its stats
    (also checks the state against the reference)."""
    var n = c.num_qubits
    var sv = StateVector(n)
    var plan = build_plan(c, List[Float64]())
    var st = execute_plan_reordered_stats(sv, plan, min(BLOCK_BITS, n - MIN_BLOCKS_LOG2))
    sv.apply_global_phase(c.global_phase)
    check_same(sv, reference(c), "stats_of " + String(st))
    return st^


def prep(n: Int) -> Circuit:
    """Block-local entangling work on qubits < 12, diagonal phases above:
    no swap-in is ever needed, so the final permutation is exactly the one
    made by the SWAP gates appended by the caller."""
    var c = Circuit(n)
    for d in range(3):
        for q in range(12):
            c.ry(0.1 + 0.07 * Float64(q + d), q)
        for q in range(d % 2, 11, 2):
            c.cx(q, q + 1)
    for q in range(12, n):
        c.rz(0.3 + 0.01 * Float64(q), q)
        c.cp(0.2, q, q % 12)  # diagonal: global bits are block constants
    return c^


def expect_error(code: String, msg: String) raises:
    if not msg.startswith(code):
        raise Error("expected " + code + ", got: " + msg)
    assert_true(msg.find("Hint:") >= 0, "error must carry a Hint: " + msg)


def main() raises:
    var rng = Rng(5)

    # 1. random circuits, 19-24 qubits, gates on high / low / mixed qubits
    for n in [19, 20, 21, 22]:
        for mode in range(4):
            var c = random_circuit(n, 160 if n < 22 else 90, mode, rng)
            check_same(evolve(c), reference(c), "random n=" + String(n) + " mode=" + String(mode))
    var c24 = random_circuit(24, 60, 3, rng)  # 24 qubits: 9 global bits
    check_same(evolve(c24), reference(c24), "random n=24 mixed")

    # 2. original mixed-kernel regression circuit (21 qubits, 400 gates)
    var n = 21
    var c = Circuit(n)
    var kinds: List[Int] = [Op.H, Op.RX, Op.RZ, Op.T, Op.U, Op.CX, Op.CZ, Op.CP, Op.CRY, Op.RZZ, Op.SWAP, Op.RXX, Op.ISWAP, Op.CCX]
    for _ in range(400):
        var k = kinds[rng.below(len(kinds))]
        var a = rng.below(n)
        var b = rng.below(n - 1)
        if b >= a:
            b += 1
        var d = rng.below(n)
        while d == a or d == b:
            d = rng.below(n)
        c.add(k, a, b, d, rng.uniform() * 3, rng.uniform(), rng.uniform())
    check_same(evolve(c), reference(c), "mixed 21q")

    # 3. probabilities, expectation and sampling on a high-qubit-heavy circuit
    var hc = random_circuit(20, 200, 1, rng)
    var fast = evolve(hc)
    var slow = reference(hc)
    var pf = fast.probabilities()
    var ps = slow.probabilities()
    var dp = 0.0
    for i in range(len(pf)):
        dp = max(dp, abs(pf[i] - ps[i]))
    assert_true(dp < 1e-12, "probabilities differ: " + String(dp))
    var h = parse_pauli_sum("Z0 Z19 + 0.5*X3 Y18 - 0.25*Y1 X15 Z19 + 0.7*X19 X18 X17")
    assert_almost_equal(fast.expectation(h), slow.expectation(h), atol=1e-10)
    var mc = hc.copy()
    mc.measure_all()
    var r1 = sample(mc, 3000, seed=99)
    var r2 = sample(mc, 3000, seed=99)
    assert_equal(r1.method, "statevector:static")
    var rr = Rng(99)
    var outs = slow.sample(3000, rr)
    var refc = Dict[String, Int]()
    for o in outs:
        var key = String()
        for q in range(19, -1, -1):
            key += "1" if (o >> q) & 1 == 1 else "0"
        refc[key] = refc.get(key, 0) + 1
    for kv in r1.counts.items():
        assert_equal(kv.value, r2.counts.get(kv.key, -1), "same seed => same counts")
        assert_equal(kv.value, refc.get(kv.key, -1), "blocked vs reference sampling")
    assert_equal(len(r1.counts), len(refc))

    # 4. final permutation restore
    # 4a identity: SWAP pairs that cancel -> relabel back, no restore pass
    var ci = prep(20)
    ci.swap(3, 17)
    ci.swap(19, 0)
    ci.swap(3, 17)
    ci.swap(0, 19)
    var st = stats_of(ci)
    assert_equal(st.relabels, 4)
    assert_equal(st.swap_layers, 0)
    assert_equal(st.restore_layers, 0)
    # 4b only high bits: a 2-cycle and a 3-cycle among bits >= nb
    var chi = prep(22)
    chi.swap(20, 21)
    chi.swap(16, 18)
    chi.swap(18, 19)
    st = stats_of(chi)
    assert_equal(st.swap_layers, 0)
    assert_true(st.restore_layers >= 1 and st.restore_min_bit >= 16, String(st))
    # 4c only low bits, including the in-vector bits 0..2
    var clo = prep(21)
    clo.swap(0, 9)
    clo.swap(1, 2)
    clo.swap(2, 5)
    clo.swap(4, 11)
    st = stats_of(clo)
    assert_equal(st.swap_layers, 0)
    assert_true(st.restore_layers == 2 and st.restore_min_bit == 0, String(st))
    assert_true(st.vector_pairs >= 2, String(st))  # SWAP relabels of bits 0..2
    # 4d mixed: QFT reversal swaps (low <-> high) after real swap-ins
    var cq = Circuit(20)
    for q in range(20):
        cq.rx(0.1 * Float64(q + 1), q)
    var j = 19
    while j >= 0:
        cq.h(j)
        var k = j - 1
        while k >= 0:
            cq.cp(pi / Float64(1 << (j - k)), k, j)
            k -= 1
        j -= 1
    for q in range(10):
        cq.swap(q, 19 - q)
    st = stats_of(cq)
    assert_true(st.swap_layers >= 1 and st.restore_min_bit == 0, String(st))
    # 4e swap-ins without SWAP gates (1q work on every high qubit)
    var cs = Circuit(21)
    for d in range(3):
        for q in range(21):
            cs.u(0.3 + 0.1 * Float64(d), 0.2 * Float64(q), -0.1, q)
        for q in range(d % 2, 20, 2):
            cs.cx(q + 1, q)
    st = stats_of(cs)
    assert_true(st.swap_layers >= 1 and st.block_passes >= 2, String(st))
    # 4f full selections without SWAP gates (the bench `layered` family at
    # 20 qubits): S fills the block, the pinned re-selection keeps bits 0..2
    # local, so no swap-in or restore pair touches an in-vector bit
    var cl = Circuit(20)
    for d in range(10):
        for q in range(20):
            cl.ry(0.1 + 0.01 * Float64(q + d), q)
            cl.rz(0.2 + 0.03 * Float64(q * d), q)
        for q in range(d % 2, 19, 2):
            cl.cx(q, q + 1)
    st = stats_of(cl)
    assert_true(st.swap_layers >= 1 and st.relabels == 0, String(st))
    assert_equal(st.vector_pairs, 0, String(st))
    assert_true(st.restore_min_bit < 0 or st.restore_min_bit >= 3, String(st))

    # 5. non-unitary ops are skipped by the plan path (evolve semantics)
    var cm = random_circuit(19, 60, 3, rng)
    cm.measure(18, 0)
    cm.reset(0)
    cm.add(Op.BARRIER, 5)
    var tail = random_circuit(19, 60, 1, rng)
    cm.compose(tail)
    check_same(evolve(cm), reference(cm), "non-unitary skipped")
    var cc = Circuit(19, 1)
    cc.h(0)
    cc.measure(0, 0)
    cc.x(18)
    cc.c_if(0, 1)
    try:
        _ = evolve(cc)
        raise Error("conditional gate must be rejected by evolve()")
    except e:
        expect_error("QM-E401", String(e))

    # 6. n < 19: plain path (one pass per op), high and low qubits
    for n2 in [12, 18]:
        for mode in range(4):
            var c2 = random_circuit(n2, 120, mode, rng)
            check_same(evolve(c2), reference(c2), "small n=" + String(n2))

    # 7. every block size nb on small states (random initial state)
    for n3 in range(4, 10):
        for nb in range(1, n3 + 1):
            var c3 = random_circuit(n3, 50, rng.below(4), rng)
            var psi = StateVector(n3)
            for i in range(psi.size()):
                psi.re[i] = rng.uniform() - 0.5
                psi.im[i] = rng.uniform() - 0.5
            var a = psi.copy()
            var b = psi.copy()
            var plan = build_plan(c3, List[Float64]())
            execute_plan_reordered(a, plan, nb)
            for g in c3.ops:
                if op_is_unitary(g.kind):
                    apply_instruction(b, g, List[Float64]())
            if max_diff(a, b) > 1e-9:
                raise Error("nb sweep mismatch n=" + String(n3) + " nb=" + String(nb))

    # 8. argument checks
    var sv = StateVector(6)
    var bad_a: List[Int] = [2, 0, -1, 4, 0]
    var bad_b: List[Int] = [1, 6, 3, 4, 3]  # out of range (6, -1), self-swap (4, 4)
    for t in range(5):
        var qa: List[Int] = [bad_a[t]]
        var qb: List[Int] = [bad_b[t]]
        if t == 0:
            qa = [2, 1]
            qb = [1, 5]  # bit 1 used twice
        elif t == 4:
            qa = [0, 1]
            qb = [3, 0]  # bit 0 used twice
        try:
            sv.apply_swap_layer(qa, qb)
            raise Error("swap layer accepted invalid bits, case " + String(t))
        except e:
            expect_error("QM-E407", String(e))
    var q1: List[Int] = [0, 1]
    var q2: List[Int] = [3]
    try:
        sv.apply_swap_layer(q1, q2)
        raise Error("swap layer accepted unequal lists")
    except e:
        expect_error("QM-E407", String(e))
    var p6 = build_plan(random_circuit(6, 10, 0, rng), List[Float64]())
    for nb in [0, 7, -3]:
        try:
            execute_plan_reordered(sv, p6, nb)
            raise Error("execute_plan_reordered accepted nb=" + String(nb))
        except e:
            expect_error("QM-E408", String(e))
    print("test_sim_blocking OK")
