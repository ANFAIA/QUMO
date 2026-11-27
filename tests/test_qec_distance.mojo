"""Tests for qumo.qec.distance: exact circuit distance against brute force and theory."""

from std.bit import pop_count
from std.time import perf_counter_ns
from std.testing import assert_equal, assert_true
from qumo.core.rng import Rng
from qumo.qec.codes import NoiseParams, surface_code_memory, repetition_code_memory
from qumo.qec.dem import DetectorErrorModel, DemError, detector_error_model
from qumo.qec.stim_text import parse_dem
from qumo.qec.distance import DistanceResult, circuit_distance, verify_witness


def _brute(dem: DetectorErrorModel) -> Int:
    """Gray-code enumeration over the raw mechanisms: no merging, no shared canonicalisation."""
    var n = len(dem.errors)
    var dm = List[UInt64](length=n, fill=UInt64(0))
    var om = List[UInt64](length=n, fill=UInt64(0))
    for j in range(n):
        var v: UInt64 = 0
        for d in dem.errors[j].dets:
            v ^= UInt64(1) << UInt64(d)
        dm[j] = v
        om[j] = dem.errors[j].obs
    var best = -1
    var dsyn: UInt64 = 0
    var o: UInt64 = 0
    var w = 0
    for i in range(1, 1 << n):
        var ui = UInt64(i)
        var lowbit = ui & (~ui + 1)
        var j = Int(pop_count(lowbit - 1))
        var g = i ^ (i >> 1)
        if ((g >> j) & 1) == 1:
            w += 1
        else:
            w -= 1
        dsyn ^= dm[j]
        o ^= om[j]
        if dsyn == 0 and o != 0 and (best < 0 or w < best):
            best = w
    return best


def _random_dem(mut rng: Rng, nc_span: Int) -> DetectorErrorModel:
    """Random hypergraph model: 2..12 detectors, 3..(2 + nc_span) mechanisms, 1..3 observables."""
    var nd = 2 + rng.below(11)
    var nc = 3 + rng.below(nc_span)
    var nobs = 1 + rng.below(3)
    var ks = [0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    var dem = DetectorErrorModel(nd, nobs)
    for _i in range(nc):
        var k = min(ks[rng.below(11)], nd)
        var mask = List[Bool](length=nd, fill=False)
        var cnt = 0
        while cnt < k:
            var b = rng.below(nd)
            if not mask[b]:
                mask[b] = True
                cnt += 1
        var dets = List[Int]()
        for b in range(nd):
            if mask[b]:
                dets.append(b)
        var obs: UInt64 = 0
        for l in range(nobs):
            if rng.uniform() < 0.3:
                obs |= UInt64(1) << UInt64(l)
        dem.errors.append(DemError(0.01, dets^, obs))
    if rng.uniform() < 0.2:
        var idx = rng.below(nc)
        var dup = dem.errors[idx].copy()
        dem.errors.append(dup^)
    return dem^


def _rm15(x_type: Bool) -> DetectorErrorModel:
    """Quantum Reed-Muller [[15,1,3]] code capacity. Qubit q is the point p = q + 1 of F2^4."""
    var nd = 4
    if x_type:
        nd = 10
    var dem = DetectorErrorModel(nd, 1)
    for q in range(15):
        var p = q + 1
        var dets = List[Int]()
        for b in range(4):
            if ((p >> b) & 1) == 1:
                dets.append(b)
        if x_type:
            var idx = 4
            for a in range(4):
                for b in range(a + 1, 4):
                    if ((p >> a) & 1) == 1 and ((p >> b) & 1) == 1:
                        dets.append(idx)
                    idx += 1
        dem.errors.append(DemError(0.001, dets^, UInt64(1)))
    return dem^


def _steane() -> DetectorErrorModel:
    var dem = DetectorErrorModel(3, 1)
    for q in range(7):
        var p = q + 1
        var dets = List[Int]()
        for b in range(3):
            if ((p >> b) & 1) == 1:
                dets.append(b)
        dem.errors.append(DemError(0.001, dets^, UInt64(1)))
    return dem^


def _rep_capacity(d: Int) -> DetectorErrorModel:
    """Repetition-code capacity: qubit q flips detectors q-1 and q; obs 0 is the qubit-0 logical."""
    var dem = DetectorErrorModel(d - 1, 1)
    for q in range(d):
        var dets = List[Int]()
        if q >= 1:
            dets.append(q - 1)
        if q <= d - 2:
            dets.append(q)
        var obs: UInt64 = 0
        if q == 0:
            obs = 1
        dem.errors.append(DemError(0.001, dets^, obs))
    return dem^


def _hand() -> DetectorErrorModel:
    """Three mechanisms with a 3-detector hyperedge: distance 3."""
    var dem = DetectorErrorModel(3, 1)
    var a: List[Int] = [0, 1, 2]
    dem.errors.append(DemError(0.1, a^, UInt64(1)))
    var b: List[Int] = [0]
    dem.errors.append(DemError(0.1, b^, UInt64(0)))
    var c: List[Int] = [1, 2]
    dem.errors.append(DemError(0.1, c^, UInt64(0)))
    return dem^


def _check_exact(dem: DetectorErrorModel, expected: Int, label: String) raises:
    var r = circuit_distance(dem)
    var msg = label + " got " + String(r)
    assert_equal(r.status, "exact", msg)
    assert_equal(r.lower, expected, msg)
    assert_equal(r.upper, expected, msg)
    assert_equal(len(r.witness), expected, msg)
    assert_true(verify_witness(dem, r.witness), msg)
    assert_equal(_brute(dem), expected, label + " brute force")


def _raise_message(dem: DetectorErrorModel, max_weight: Int, node_limit: Int) -> String:
    """The message of the error raised by circuit_distance, or "" when nothing was raised."""
    try:
        _ = circuit_distance(dem, max_weight, node_limit)
    except e:
        return String(e)
    return String()


def _circuit_dem(kind: String, d: Int, basis: String, decompose: Bool) raises -> DetectorErrorModel:
    if kind == "surface":
        var cc = surface_code_memory(d, d, NoiseParams.uniform(0.001), basis)
        return detector_error_model(cc.circuit, decompose)
    var rc = repetition_code_memory(d, d, NoiseParams.uniform(0.001))
    return detector_error_model(rc.circuit, decompose)


def test_random_vs_brute() raises:
    var rng = Rng(20261127)
    var finite = 0
    var budget_hits = 0
    for trial in range(300):
        var dem = _random_dem(rng, 14)
        var b = _brute(dem)
        var r = circuit_distance(dem, max_weight=20)
        var msg = "model " + String(trial) + " brute=" + String(b) + " got " + String(r)
        if b >= 0:
            assert_equal(r.status, "exact", msg)
            assert_equal(r.lower, b, msg)
            assert_equal(r.upper, b, msg)
            assert_true(r.exact, msg)
            assert_equal(len(r.witness), b, msg)
            assert_true(verify_witness(dem, r.witness), msg)
            finite += 1
        else:
            assert_true(r.status == "above_max_weight" or r.status == "no_logical", msg)
            assert_equal(r.upper, -1, msg)
            assert_equal(r.lower, 21, msg)
            assert_true(not r.exact, msg)
        for lim in [1, 5, 50, 500]:
            var r2 = circuit_distance(dem, max_weight=20, node_limit=lim)
            var msg2 = msg + " node_limit=" + String(lim) + " got " + String(r2)
            if r2.status == "budget":
                assert_equal(r2.upper, -1, msg2)
                assert_equal(len(r2.witness), 0, msg2)
                assert_true(not r2.exact, msg2)
                assert_true(r2.lower >= 1, msg2)
                if b >= 0:
                    assert_true(r2.lower <= b, msg2)
                budget_hits += 1
            else:
                assert_equal(r2.status, r.status, msg2)
                assert_equal(r2.lower, r.lower, msg2)
                assert_equal(r2.upper, r.upper, msg2)
    print("  random models with a finite distance:", finite, "of 300; budget outcomes:", budget_hits)
    assert_true(finite >= 100, "too few finite random models: " + String(finite))
    assert_true(budget_hits >= 50, "too few budget outcomes: " + String(budget_hits))
    print("  random vs brute force")


def test_code_capacity() raises:
    var rm_z = _rm15(False)
    _check_exact(rm_z, 3, "RM15 Z-type")
    var rm_x = _rm15(True)
    _check_exact(rm_x, 7, "RM15 X-type")
    _check_exact(_steane(), 3, "Steane")
    _check_exact(_rep_capacity(3), 3, "rep capacity d=3")
    _check_exact(_rep_capacity(5), 5, "rep capacity d=5")
    _check_exact(_rep_capacity(7), 7, "rep capacity d=7")
    print("  code capacity: RM15 Z 3, RM15 X 7, Steane 3, repetition 3, 5, 7")


def test_hyperedge_hand() raises:
    var dem = _hand()
    _check_exact(dem, 3, "hand hyperedge")
    var r = circuit_distance(dem)
    var w = r.witness.copy()
    sort(w)
    assert_equal(w, [0, 1, 2])
    var fwd = parse_dem("error(0.1) D0 D1 D2 L0\nerror(0.1) D0\nerror(0.1) D1 D2")
    var rev = parse_dem("error(0.1) D1 D2\nerror(0.1) D0\nerror(0.1) D0 D1 D2 L0")
    assert_equal(circuit_distance(fwd).lower, 3, "text model, forward order")
    assert_equal(circuit_distance(rev).lower, 3, "text model, reversed order")
    assert_equal(_brute(fwd), 3, "text model brute force")
    print("  hyperedge hand model: 3")


def test_preprocessing() raises:
    # Duplicate D0 L0 mechanisms merge; {D0 L0, D0} is the weight-2 logical.
    var dup = parse_dem("error(0.1) D0 L0\nerror(0.1) D0 L0\nerror(0.1) D0")
    assert_equal(circuit_distance(dup).lower, 2, "duplicate pair")
    assert_equal(circuit_distance(dup).upper, 2, "duplicate pair upper")
    # An observable-only mechanism is a weight-1 logical.
    var lonely = parse_dem("error(0.1) D0 D1\nerror(0.1) D1 L0\nerror(0.1) L0")
    var r1 = circuit_distance(lonely)
    assert_equal(r1.lower, 1, "L0 alone")
    assert_equal(r1.status, "exact", "L0 alone status")
    assert_equal(len(r1.witness), 1, "L0 alone witness size")
    assert_equal(r1.witness[0], 2, "L0 alone witness index")
    # D0 D0 cancels inside one mechanism, leaving L0 alone.
    assert_equal(circuit_distance(parse_dem("error(0.1) D0 D0 L0")).lower, 1, "D0 D0 L0")
    # No logical at all: no merged column flips an observable.
    var nol1 = circuit_distance(parse_dem("error(0.1) D3 D3"))
    assert_equal(nol1.status, "no_logical", "D3 D3 status")
    assert_equal(nol1.lower, 33, "D3 D3 lower")
    assert_equal(nol1.upper, -1, "D3 D3 upper")
    assert_equal(nol1.nodes, 0, "D3 D3 nodes")
    var nol2 = circuit_distance(parse_dem("error(0.1) D0 D1\nerror(0.1) D1"))
    assert_equal(nol2.status, "no_logical", "chain status")
    assert_equal(nol2.lower, 33, "chain lower")
    assert_equal(nol2.upper, -1, "chain upper")
    assert_equal(nol2.nodes, 0, "chain nodes")
    print("  preprocessing: merge, observable-only, D0 D0, no_logical")


def _circuit_case(kind: String, d: Int, basis: String, decompose: Bool, expected: Int) raises:
    var dem = _circuit_dem(kind, d, basis, decompose)
    var t0 = perf_counter_ns()
    var r = circuit_distance(dem)
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    var label = kind + " d=" + String(d) + " basis=" + basis + " decompose=" + String(decompose)
    print("  ", label, ":", String(r), "ms=", ms)
    assert_equal(r.status, "exact", label + " " + String(r))
    assert_equal(r.lower, expected, label + " " + String(r))
    assert_equal(r.upper, expected, label + " " + String(r))
    assert_true(verify_witness(dem, r.witness), label + " witness")


def test_circuit_dems() raises:
    _circuit_case("surface", 3, "Z", False, 3)
    _circuit_case("surface", 5, "Z", False, 5)
    _circuit_case("surface", 3, "X", False, 3)
    _circuit_case("repetition", 3, "Z", False, 3)
    _circuit_case("repetition", 5, "Z", False, 5)
    _circuit_case("surface", 3, "Z", True, 3)


def test_budget() raises:
    var dem = _circuit_dem("surface", 5, "Z", False)
    var r1 = circuit_distance(dem, node_limit=1)
    assert_equal(r1.status, "budget", String(r1))
    assert_equal(r1.lower, 1, String(r1))
    assert_equal(r1.upper, -1, String(r1))
    assert_equal(len(r1.witness), 0, String(r1))
    assert_true(not r1.exact, String(r1))
    var r2 = circuit_distance(dem, node_limit=10_000)
    assert_equal(r2.status, "budget", String(r2))
    assert_true(r2.lower >= 2 and r2.lower <= 5, String(r2))
    assert_equal(r2.upper, -1, String(r2))
    print("  budget: node_limit=1 ->", String(r1), "; node_limit=10000 ->", String(r2))


def test_above_max_weight() raises:
    var r = circuit_distance(_rep_capacity(5), max_weight=3)
    assert_equal(r.status, "above_max_weight", String(r))
    assert_equal(r.lower, 4, String(r))
    assert_equal(r.upper, -1, String(r))
    var s = circuit_distance(_steane(), max_weight=2)
    assert_equal(s.status, "above_max_weight", String(s))
    assert_equal(s.lower, 3, String(s))
    assert_equal(s.upper, -1, String(s))


def test_invalid_args() raises:
    var hand = _hand()
    var msgs = List[String]()
    msgs.append(_raise_message(hand, 0, 50_000_000))
    msgs.append(_raise_message(hand, 32, 0))
    var wide = DetectorErrorModel(1, 65)
    wide.errors.append(DemError(0.1, [0], UInt64(1)))
    msgs.append(_raise_message(wide, 32, 50_000_000))
    var neg = DetectorErrorModel(1, 1)
    neg.errors.append(DemError(0.1, [-1], UInt64(1)))
    msgs.append(_raise_message(neg, 32, 50_000_000))
    var bad_pieces = DetectorErrorModel(3, 1)
    var ids: List[Int] = [0, 1]
    var bad = DemError(0.1, ids^, UInt64(0))
    var cd: List[List[Int]] = [[0], [2]]
    var co: List[UInt64] = [UInt64(0), UInt64(0)]
    bad.comp_dets = cd^
    bad.comp_obs = co^
    bad_pieces.errors.append(bad^)
    msgs.append(_raise_message(bad_pieces, 32, 50_000_000))
    for m in msgs:
        assert_true(m.startswith("QM-E740"), m)
        assert_true(m.find("Hint:") >= 0, m)
    print("  invalid arguments: 5 of 5 raise QM-E740 with a hint")


def test_verify_witness() raises:
    var hand = _hand()
    assert_true(verify_witness(hand, [0, 1, 2]), "[0,1,2]")
    assert_true(verify_witness(hand, [2, 0, 1]), "[2,0,1]")
    assert_true(not verify_witness(hand, []), "empty")
    assert_true(not verify_witness(hand, [0, 0, 1]), "repeated index")
    assert_true(not verify_witness(hand, [0, 1, 3]), "out of range")
    assert_true(not verify_witness(hand, [-1]), "negative index")
    assert_true(not verify_witness(hand, [1, 2]), "nonzero syndrome")
    var cancel = parse_dem("error(0.1) D0\nerror(0.1) D0")
    assert_true(not verify_witness(cancel, [0, 1]), "cancels but observables do not flip")


def main() raises:
    test_random_vs_brute()
    test_code_capacity()
    test_hyperedge_hand()
    test_preprocessing()
    test_circuit_dems()
    test_budget()
    test_above_max_weight()
    test_invalid_args()
    test_verify_witness()
    print("test_qec_distance OK")
