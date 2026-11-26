"""Tests for the noise stack: Kraus channels, density matrix, PTSBE, emulate."""

from std.math import pi, exp, sqrt, cos, sin
from std.testing import assert_equal, assert_true, assert_almost_equal
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.kraus import KrausChannel, amplitude_damping, phase_damping, thermal_relaxation
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.runner import evolve, sample, SampleResult
from qumo.sim.rng import Rng
from qumo.sim.density import density_evolve, sample_density, DensityMatrix, check_density_size
from qumo.sim.ptsbe import sample_ptsbe, plan_trajectories
from qumo.compiler.target import Target, parse_target
from qumo.compiler.pass_manager import transpile
from qumo.qec.characterization import NoiseModel
from qumo.noise.emulate import emulate, noisy_circuit, active_qubits


def expect_code(code: String, err: Error) raises:
    assert_true(String(err).startswith(code), "expected " + code + ", got " + String(err))


def test_pure_matches_statevector() raises:
    var rng = Rng(5)
    var c = Circuit(4)
    for _ in range(30):
        var r = rng.below(9)
        var q = rng.below(4)
        var t = (q + 1 + rng.below(3)) % 4
        if r == 0: c.h(q)
        elif r == 1: c.t(q)
        elif r == 2: c.sx(q)
        elif r == 3: c.y(q)
        elif r == 4: c.u(0.3, 0.4, 0.5, q)
        elif r == 5: c.rx(0.7, q)
        elif r == 6: c.iswap(q, t)
        elif r == 7: c.ccx(q, t, (t + 1 + (0 if (t + 1) % 4 != q else 1)) % 4)
        else: c.cp(0.9, q, t)
    var rho = density_evolve(c)
    var sv = evolve(c)
    for i in range(16):
        for j in range(16):
            var e = rho.element(i, j)
            var er = sv.re[i] * sv.re[j] + sv.im[i] * sv.im[j]
            var ei = sv.im[i] * sv.re[j] - sv.re[i] * sv.im[j]
            assert_true(abs(e[0] - er) < 1e-12 and abs(e[1] - ei) < 1e-12, "rho = |psi><psi|")
    assert_almost_equal(rho.trace(), 1.0, atol=1e-12)
    assert_almost_equal(rho.purity(), 1.0, atol=1e-12)
    var h = parse_pauli_sum("0.5*Z0 Z1 - X2 + 0.25*Y3 Y0")
    assert_almost_equal(rho.expectation(h), sv.expectation(h), atol=1e-12)
    assert_almost_equal(rho.fidelity_with(sv), 1.0, atol=1e-12)
    print("  pure density matrix equals |psi><psi|")


def test_amplitude_damping_analytic() raises:
    # rho_11(t) = exp(-t/T1), rho_01 = exp(-t/T2)/2 on |+>
    var t1 = 50.0
    var t2 = 70.0
    var steps = 7
    var dt = 3.0
    var c = Circuit(1)
    c.h(0)
    for _ in range(steps):
        c.apply_channel(thermal_relaxation(dt, t1, t2), 0)
    var rho = density_evolve(c)
    var t = dt * Float64(steps)
    assert_almost_equal(rho.element(1, 1)[0], 0.5 * exp(-t / t1), atol=1e-12)
    assert_almost_equal(rho.element(0, 1)[0], 0.5 * exp(-t / t2), atol=1e-12)
    assert_almost_equal(rho.trace(), 1.0, atol=1e-12)
    var d = Circuit(1)
    d.x(0)
    d.apply_channel(amplitude_damping(0.3), 0)
    assert_almost_equal(density_evolve(d).element(1, 1)[0], 0.7, atol=1e-12)
    print("  thermal relaxation matches exp(-t/T1), exp(-t/T2)")


def test_pauli_noise_exact() raises:
    # depolarize then read: <Z> shrinks by 1 - 4p/3
    var c = Circuit(1)
    c.depolarize1(0.09, 0)
    var rho = density_evolve(c)
    assert_almost_equal(rho.expectation(parse_pauli_sum("Z0")), 1 - 4 * 0.09 / 3, atol=1e-12)
    var b = Circuit(2)
    b.h(0)
    b.cx(0, 1)
    b.depolarize2(0.15, 0, 1)
    var r2 = density_evolve(b)
    # <ZZ> shrinks by 1 - 16 p / 15 x (8/16 ...): 8 of 15 Paulis anticommute with ZZ
    assert_almost_equal(r2.expectation(parse_pauli_sum("Z0 Z1")), 1 - 2 * 0.15 * 8 / 15, atol=1e-12)
    # reset and a mid-circuit measurement
    var m = Circuit(1, 1)
    m.h(0)
    m.measure(0, 0)
    var r3 = density_evolve(m)
    assert_almost_equal(r3.element(0, 1)[0], 0.0, atol=1e-12)
    assert_almost_equal(r3.purity(), 0.5, atol=1e-12)
    var q = Circuit(1)
    q.x(0)
    q.reset(0)
    assert_almost_equal(density_evolve(q).element(0, 0)[0], 1.0, atol=1e-12)
    print("  exact Pauli channels, measurement, reset")


def test_two_qubit_kraus() raises:
    var ch = KrausChannel("swap_with_prob", 2)
    var p = 0.25
    var a = sqrt(1 - p)
    var b = sqrt(p)
    ch.add_real([a, 0, 0, 0, 0, a, 0, 0, 0, 0, a, 0, 0, 0, 0, a])
    ch.add_real([b, 0, 0, 0, 0, 0, b, 0, 0, b, 0, 0, 0, 0, 0, b])
    var c = Circuit(2)
    c.x(0)
    c.apply_channel(ch, 0, 1)
    var rho = density_evolve(c)
    assert_almost_equal(rho.element(1, 1)[0], 0.75, atol=1e-12)
    assert_almost_equal(rho.element(2, 2)[0], 0.25, atol=1e-12)
    print("  two-qubit Kraus channel")


def test_kraus_trajectories_match_density() raises:
    var c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    c.apply_channel(amplitude_damping(0.4), 0)
    c.apply_channel(phase_damping(0.3), 1)
    c.rx(0.8, 0)
    c.measure_all()
    var exact = density_evolve(c).probabilities()
    var r = sample(c, 40000, 3)
    assert_true(r.method.find("trajectories") >= 0, "non-Pauli noise runs trajectories")
    for i in range(4):
        var key = String("1" if i & 2 else "0") + ("1" if i & 1 else "0")
        assert_true(abs(r.probability(key) - exact[i]) < 0.012, "trajectories agree with rho for " + key)
    var s = sample_density(c, 40000, 4)
    assert_equal(s.method, "density_matrix", "method")
    for i in range(4):
        var key = String("1" if i & 2 else "0") + ("1" if i & 1 else "0")
        assert_true(abs(s.probability(key) - exact[i]) < 0.012, "sampling rho")
    print("  Kraus trajectories agree with the density matrix")


def test_limits_and_errors() raises:
    var hit = False
    try:
        check_density_size(16)
    except e:
        expect_code("QM-E410", e)
        hit = True
    assert_true(hit, "memory limit")
    hit = False
    var c = Circuit(1, 1)
    c.measure(0, 0)
    c.x(0)
    c.c_if(0, 1)
    try:
        _ = density_evolve(c)
    except e:
        expect_code("QM-E411", e)
        hit = True
    assert_true(hit, "conditions")
    hit = False
    try:
        var bad = KrausChannel("bad", 1)
        bad.add_real([0.5, 0.0, 0.0, 0.5])
        var d = Circuit(1)
        d.apply_channel(bad, 0)
    except e:
        expect_code("QM-E414", e)
        hit = True
    assert_true(hit, "trace preservation")
    print("  QM-E410, E411, E414")


def noisy_ghz(n: Int, p: Float64) -> Circuit:
    var c = Circuit(n)
    c.h(0)
    c.depolarize1(p, 0)
    for q in range(n - 1):
        c.t(q)
        c.cx(q, q + 1)
        c.depolarize2(p, q, q + 1)
    for q in range(n):
        c.x_error(p, q)
    c.measure_all()
    return c^


def tv(a: SampleResult, b: SampleResult) -> Float64:
    var d = 0.0
    for kv in a.counts.items():
        d += abs(a.probability(kv.key) - b.probability(kv.key))
    for kv in b.counts.items():
        if kv.key not in a.counts:
            d += b.probability(kv.key)
    return d / 2


def test_ptsbe_same_distribution() raises:
    var c = noisy_ghz(6, 0.01)
    var exact = density_evolve(c).probabilities()
    var r = sample_ptsbe(c, 200000, 5)
    assert_equal(r.method, "statevector:ptsbe", "method")
    assert_equal(r.shots, 200000, "shots")
    var total = 0
    var dist = 0.0
    for i in range(64):
        var key = String("")
        var q = 5
        while q >= 0:
            key += "1" if (i >> q) & 1 else "0"
            q -= 1
        dist += abs(r.probability(key) - exact[i])
        total += r.counts.get(key, 0)
    assert_equal(total, 200000, "every shot recorded")
    assert_true(dist / 2 < 0.01, "against the exact density matrix, TV " + String(dist / 2))
    var plan = plan_trajectories(c, 200000, 5)
    assert_true(plan.unique < 200000 // 2, "patterns repeat at low noise: " + String(plan))
    assert_true(plan.error_free > 100000, "most shots have no error")
    print("  ptsbe vs exact rho: TV", dist / 2, "with", plan.unique, "patterns for 200000 shots")


def test_ptsbe_deterministic_and_cap() raises:
    var c = noisy_ghz(5, 0.02)
    var a = sample_ptsbe(c, 5000, 11)
    var b = sample_ptsbe(c, 5000, 11)
    assert_equal(String(a.counts.get(a.most_frequent(), 0)), String(b.counts.get(b.most_frequent(), 0)), "seeded")
    for kv in a.counts.items():
        assert_equal(kv.value, b.counts.get(kv.key, -1), "bit-identical counts")
    var capped = sample_ptsbe(c, 5000, 11, max_trajectories=8)
    assert_true(capped.method.find("biased") >= 0, "a cap is reported")
    assert_equal(capped.shots, 5000, "shots kept")
    print("  ptsbe seeded, cap reported")


def _zeros15() -> List[Float64]:
    var v = List[Float64]()
    for _ in range(15):
        v.append(0.0)
    return v^


def test_ptsbe_pauli_channel_2() raises:
    # deterministic: XX (slot 4) flips both; a second channel (table offset 15) with XI flips qubit 0 back
    var a = Circuit(2)
    var p4 = _zeros15()
    p4[4] = 1.0
    a.pauli_channel_2(p4, 0, 1)
    a.measure_all()
    var ra = sample(a, 200, 3)
    var pa = sample_ptsbe(a, 200, 3)
    assert_equal(ra.counts.get("11", 0), 200, "sample XX")
    assert_equal(pa.counts.get("11", 0), 200, "ptsbe XX")
    var p3 = _zeros15()
    p3[3] = 1.0  # XI: X on the first qubit
    var d = Circuit(2)
    d.pauli_channel_2(p4, 0, 1)
    d.pauli_channel_2(p3, 0, 1)
    d.measure_all()
    var rd = sample(d, 200, 3)
    var pd = sample_ptsbe(d, 200, 3)
    assert_equal(rd.counts.get("10", 0), 200, "sample XX then XI")
    assert_equal(pd.counts.get("10", 0), 200, "ptsbe XX then XI, second channel at a nonzero table offset")
    # probabilistic: XX 0.3, XI 0.2, IY 0.1 -> 00: 0.4, 11: 0.3, 01: 0.2, 10: 0.1
    var e = Circuit(2)
    var pr = _zeros15()
    pr[4] = 0.3
    pr[3] = 0.2
    pr[1] = 0.1
    e.pauli_channel_2(pr, 0, 1)
    e.measure_all()
    var n = 100000
    var rs = sample(e, n, 7)
    var rp = sample_ptsbe(e, n, 7)
    var keys = List[String]()
    keys.append("00")
    keys.append("11")
    keys.append("01")
    keys.append("10")
    var want = List[Float64]()
    want.append(0.4)
    want.append(0.3)
    want.append(0.2)
    want.append(0.1)
    for i in range(4):
        # each count has sigma <= sqrt(0.25 / 1e5) = 0.0016, so 0.01 is more than 6 sigma
        assert_true(abs(rp.probability(keys[i]) - want[i]) < 0.01, "ptsbe " + keys[i] + ": " + String(rp.probability(keys[i])))
        assert_true(abs(rs.probability(keys[i]) - want[i]) < 0.01, "sample " + keys[i] + ": " + String(rs.probability(keys[i])))
    print("  ptsbe PAULI_CHANNEL_2 matches sample and the exact weights")


def test_ptsbe_refuses_correlated_error() raises:
    var qs = List[Int]()
    qs.append(0)
    qs.append(1)
    var ls = List[Int]()
    ls.append(1)
    ls.append(1)
    var q1 = List[Int]()
    q1.append(1)
    var l1 = List[Int]()
    l1.append(1)
    # two E instructions (the second has a nonzero Pauli-table offset) and an ELSE chain
    var c = Circuit(2)
    c.correlated_error(0.5, qs, ls)
    c.correlated_error(0.5, q1, l1)
    c.correlated_error(0.5, qs, ls, True)
    c.measure_all()
    var s = sample(c, 100, 1)  # the reference sampler takes it
    assert_equal(s.shots, 100, "sample takes E")
    var hit = False
    try:
        _ = sample_ptsbe(c, 100)
    except e:
        expect_code("QM-E414", e)
        hit = True
    assert_true(hit, "E must be refused, not simulated as a one-qubit channel")
    hit = False
    try:
        _ = plan_trajectories(c, 100)
    except e:
        expect_code("QM-E414", e)
        hit = True
    assert_true(hit, "plan_trajectories refuses E too")
    print("  ptsbe E / ELSE QM-E414")


def test_ptsbe_rejects() raises:
    var c = Circuit(1, 1)
    c.h(0)
    c.reset(0)
    c.measure(0, 0)
    var hit = False
    try:
        _ = sample_ptsbe(c, 10)
    except e:
        expect_code("QM-E409", e)
        hit = True
    assert_true(hit, "reset")
    hit = False
    var k = Circuit(1, 1)
    k.apply_channel(amplitude_damping(0.1), 0)
    k.measure(0, 0)
    try:
        _ = sample_ptsbe(k, 10)
    except e:
        expect_code("QM-E409", e)
        hit = True
    assert_true(hit, "kraus")
    print("  ptsbe QM-E409")


def test_noise_model_from_target() raises:
    var t = parse_target("superconducting_heavy_hex:3")
    var ion = parse_target("trapped_ion:5")
    var m = NoiseModel.from_target(ion)
    assert_equal(m.num_qubits, 5, "qubits")
    assert_true(m.t1[0] >= 0.0, "T1 copied")
    t.set_edge_error(t.edges[0], t.edges[1], 0.111)
    var cal = NoiseModel.from_target(t)
    assert_true(abs(cal.depolarizing_2q(t.edges[0], t.edges[1]) - cal.depolarizing_2q(t.edges[2], t.edges[3])) > 0.05, "per-edge error reaches the model")
    print("  NoiseModel.from_target")


def test_emulate_tracks_estimate() raises:
    # Hellinger fidelity with the ideal distribution against the estimated success
    var specs: List[String] = ["superconducting_grid:3x3", "superconducting_heavy_hex:3", "trapped_ion:5", "neutral_atom:3x3"]
    for spec in specs:
        var t = parse_target(spec)
        var c = Circuit(5)
        c.h(0)
        for q in range(4):
            c.cx(q, q + 1)
        c.measure_all()
        var r = transpile(c, t, 2, 7)
        var e = emulate(r.circuit, t, 20000, 3)
        var ideal0 = String("00000")
        var ideal1 = String("11111")
        var keep = e.probability(ideal0) + e.probability(ideal1)
        var est = t.estimate_fidelity(r.circuit)
        # GHZ with weight outside {0..0, 1..1}: bit-flip part of the error; estimate is a lower bound-ish
        assert_true(keep >= est - 0.1, spec + ": emulated " + String(keep) + " vs estimate " + String(est))
        assert_true(keep <= 1.0 and keep > 0.2, spec + " sane: " + String(keep))
        assert_true(len(active_qubits(r.circuit)) >= 5, "active qubits")
        print("   ", spec, "GHZ weight", keep, "estimate", est, "method", e.method)
    var hit = False
    try:
        var raw = Circuit(5)
        raw.h(0)
        raw.measure_all()
        _ = emulate(raw, parse_target("line:5/sc_cz"), 10)
    except e:
        expect_code("QM-E615", e)
        hit = True
    assert_true(hit, "uncompiled circuit is refused")
    print("  emulate tracks estimate_fidelity on the modalities")


def main() raises:
    test_pure_matches_statevector()
    test_amplitude_damping_analytic()
    test_pauli_noise_exact()
    test_two_qubit_kraus()
    test_kraus_trajectories_match_density()
    test_limits_and_errors()
    test_ptsbe_same_distribution()
    test_ptsbe_deterministic_and_cap()
    test_ptsbe_rejects()
    test_ptsbe_pauli_channel_2()
    test_ptsbe_refuses_correlated_error()
    test_noise_model_from_target()
    test_emulate_tracks_estimate()
    print("test_noise_sim OK")
