"""Tests for qumo.qec.distance: exact circuit distance against brute force and theory.

The reduced path (row projection lower bound, shortest odd closed walk, lifting) is checked against
brute force on the original and on an independently projected model; see test_projection_random,
test_graphlike_random and test_reduced_circuits."""

from std.bit import pop_count
from std.time import perf_counter_ns
from std.testing import assert_equal, assert_true
from qumo.core.rng import Rng
from qumo.qec.codes import NoiseParams, surface_code_memory, repetition_code_memory
from qumo.qec.dem import DetectorErrorModel, DemError, detector_error_model
from qumo.qec.stim_text import parse_dem
from qumo.qec.distance import (
    DistanceResult,
    circuit_distance,
    circuit_distance_reduced,
    circuit_distance_with_sectors,
    detector_sectors,
    projected_distance,
    verify_witness,
)


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


# ---------------------------------------------------------------------------------------------
# Reduced path: row projection (lower bound), odd closed walk (graph-like projections), lifting.
# ---------------------------------------------------------------------------------------------


def _project_dem(dem: DetectorErrorModel, keep: List[Bool]) -> DetectorErrorModel:
    """The row projection built here, independently of the library: drop every detector not kept."""
    var out = DetectorErrorModel(dem.num_detectors, dem.num_observables)
    for e in dem.errors:
        var ds = List[Int]()
        for d in e.dets:
            if keep[d]:
                ds.append(d)
        out.errors.append(DemError(e.p, ds^, e.obs))
    return out^


def _random_sector(mut rng: Rng, nd: Int) -> List[Int]:
    var s = List[Int]()
    for d in range(nd):
        if rng.uniform() < 0.5:
            s.append(d)
    if len(s) == 0:
        s.append(rng.below(nd))
    return s^


def _keep(nd: Int, s: List[Int]) -> List[Bool]:
    var k = List[Bool](length=nd, fill=False)
    for d in s:
        k[d] = True
    return k^


def _check_against(r: DistanceResult, b: Int, max_weight: Int, dem: DetectorErrorModel, msg: String) raises:
    """Full-budget result must equal brute force b (b < 0: no logical of any weight)."""
    if b >= 0 and b <= max_weight:
        assert_equal(r.status, "exact", msg)
        assert_equal(r.lower, b, msg)
        assert_equal(r.upper, b, msg)
        assert_equal(len(r.witness), b, msg)
        assert_true(verify_witness(dem, r.witness), msg)
    else:
        assert_true(r.status == "above_max_weight" or r.status == "no_logical", msg)
        assert_equal(r.upper, -1, msg)
        assert_equal(r.lower, max_weight + 1, msg)


def _check_interrupted(r: DistanceResult, b: Int, msg: String) raises:
    """Any budget: lower never exceeds the truth; an upper bound appears only with an exact answer."""
    if b >= 0:
        assert_true(r.lower <= b, msg)
    if r.status == "exact":
        assert_equal(r.lower, b, msg)
        assert_equal(r.upper, b, msg)
    else:
        assert_equal(r.upper, -1, msg)
        assert_equal(len(r.witness), 0, msg)


def test_projection_random() raises:
    """Random hypergraph models and random detector sectors against two brute forces:
    the projected optimum equals brute force on the independently projected model, it never exceeds
    the original's brute-force distance (the relaxation claim), and the full reduced pipeline
    (projection, lifting, fallback search) returns exactly the original's brute-force distance."""
    var rng = Rng(20261128)
    var finite = 0
    var weaker = 0
    var budget = 0
    for trial in range(400):
        var dem = _random_dem(rng, 14)
        var nd = dem.num_detectors
        var b = _brute(dem)
        var s1 = _random_sector(rng, nd)
        var s2 = _random_sector(rng, nd)
        var bp = _brute(_project_dem(dem, _keep(nd, s1)))
        var msg = "model " + String(trial) + " brute=" + String(b) + " projected brute=" + String(bp)
        if b >= 0:
            finite += 1
            assert_true(bp >= 1 and bp <= b, msg + " relaxation must not exceed the distance")
            if bp < b:
                weaker += 1
        var pd = projected_distance(dem, s1, max_weight=20)
        _check_against_projection(pd, bp, msg + " projected got " + String(pd))
        var secs = List[List[Int]]()
        secs.append(s1.copy())
        secs.append(s2.copy())
        var r = circuit_distance_with_sectors(dem, secs, max_weight=20)
        _check_against(r, b, 20, dem, msg + " with sectors got " + String(r))
        var ra = circuit_distance_reduced(dem, max_weight=20)
        _check_against(ra, b, 20, dem, msg + " reduced got " + String(ra))
        for lim in [1, 3, 10, 40, 200]:
            var r2 = circuit_distance_with_sectors(dem, secs, max_weight=20, node_limit=lim)
            _check_interrupted(r2, b, msg + " node_limit=" + String(lim) + " got " + String(r2))
            if r2.status == "budget":
                budget += 1
    print("  projection: finite", finite, "of 400; projection weaker than the distance in", weaker, "; budget outcomes", budget)
    assert_true(finite >= 150, "too few finite models")
    assert_true(weaker >= 50, "too few models where the relaxation is strict (lifting must fail there)")
    assert_true(budget >= 100, "too few budget outcomes")


def _check_against_projection(r: DistanceResult, bp: Int, msg: String) raises:
    if bp >= 0 and bp <= 20:
        assert_equal(r.status, "exact", msg)
        assert_equal(r.lower, bp, msg)
        assert_equal(r.upper, bp, msg)
        assert_equal(len(r.witness), bp, msg)
    else:
        assert_true(r.status == "above_max_weight" or r.status == "no_logical", msg)
        assert_equal(r.upper, -1, msg)


def _random_graph_dem(mut rng: Rng) -> DetectorErrorModel:
    """Random graph-like model: every mechanism has 0, 1 or 2 detectors; cycles that avoid the
    boundary are common (few 1-detector mechanisms); 1..3 observables."""
    var nd = 2 + rng.below(9)
    var nc = 3 + rng.below(15)
    var nobs = 1 + rng.below(3)
    var dem = DetectorErrorModel(nd, nobs)
    var p_bound = rng.uniform() * 0.4
    for _i in range(nc):
        var dets = List[Int]()
        var u = rng.uniform()
        if u < 0.04:
            pass
        elif u < 0.04 + p_bound:
            dets.append(rng.below(nd))
        else:
            var a = rng.below(nd)
            var c = rng.below(nd - 1)
            if c >= a:
                c += 1
            dets.append(min(a, c))
            dets.append(max(a, c))
        var obs: UInt64 = 0
        for l in range(nobs):
            if rng.uniform() < 0.25:
                obs |= UInt64(1) << UInt64(l)
        dem.errors.append(DemError(0.01, dets^, obs))
    return dem^


def test_graphlike_random() raises:
    """The shortest-odd-closed-walk solve (identity sector, so every column is graph-like) against
    brute force, including cycles that avoid the boundary and several observables."""
    var rng = Rng(424242)
    var finite = 0
    var interior = 0
    for trial in range(600):
        var dem = _random_graph_dem(rng)
        var b = _brute(dem)
        var all = List[Int]()
        for d in range(dem.num_detectors):
            all.append(d)
        var r = projected_distance(dem, all, max_weight=20)
        var msg = "graph model " + String(trial) + " brute=" + String(b) + " got " + String(r)
        _check_against_projection(r, b, msg)
        if b >= 0:
            finite += 1
            assert_true(verify_witness(dem, r.witness), msg)  # identity projection: a real witness
            var bdry = 0
            for i in r.witness:
                if len(dem.errors[i].dets) == 1:
                    bdry += 1
            if bdry == 0:
                interior += 1
        var secs = List[List[Int]]()
        secs.append(all^)
        _check_against(circuit_distance_with_sectors(dem, secs, max_weight=20), b, 20, dem, msg + " (with sectors)")
    print("  graph-like: finite", finite, "of 600; optimum avoids the boundary in", interior)
    assert_true(finite >= 250, "too few finite graph-like models")
    assert_true(interior >= 50, "too few optima that avoid the boundary")


def test_reduced_circuits() raises:
    """Surface-code memories: the reduced path equals the plain search at d = 3, 5, and gives d at
    d = 7, 9, 11 with a witness replayed on the full model. At d = 7 the sector lower bound is
    cross-checked by the plain weight-bounded search on the independently projected model."""
    for d in [3, 5]:
        for basis in ["Z", "X"]:
            var dem = _circuit_dem("surface", d, basis, False)
            var a = circuit_distance(dem)
            var r = circuit_distance_reduced(dem)
            var label = "surface d=" + String(d) + " " + basis + " plain " + String(a) + " reduced " + String(r)
            assert_equal(a.status, "exact", label)
            assert_equal(r.status, "exact", label)
            assert_equal(r.lower, a.lower, label)
            assert_equal(r.upper, a.upper, label)
            assert_true(verify_witness(dem, r.witness), label)
    for d in [7, 9, 11]:
        for basis in ["Z", "X"]:
            var dem = _circuit_dem("surface", d, basis, False)
            var t0 = perf_counter_ns()
            var r = circuit_distance_reduced(dem)
            var ms = Float64(perf_counter_ns() - t0) / 1e6
            var label = "surface d=" + String(d) + " " + basis + " reduced " + String(r)
            print("  ", label, "ms=", ms)
            assert_equal(r.status, "exact", label)
            assert_equal(r.lower, d, label)
            assert_equal(r.upper, d, label)
            assert_equal(len(r.witness), d, label)
            assert_true(verify_witness(dem, r.witness), label)
    # d = 7 sector bound by an independent algorithm: plain search on the projected model.
    var dem7 = _circuit_dem("surface", 7, "Z", False)
    var secs = detector_sectors(dem7)
    var best = 0
    var best_s = 0
    for i in range(len(secs)):
        var pd = projected_distance(dem7, secs[i])
        assert_equal(pd.status, "exact", String(pd))
        if pd.lower > best:
            best = pd.lower
            best_s = i
    assert_equal(best, 7, "d=7 sector lower bound")
    var proj = _project_dem(dem7, _keep(dem7.num_detectors, secs[best_s]))
    var plain = circuit_distance(proj)
    assert_equal(plain.status, "exact", "plain search on the projected d=7 model " + String(plain))
    assert_equal(plain.lower, 7, "plain search on the projected d=7 model " + String(plain))
    print("   d=7 sector of", len(secs[best_s]), "detectors: odd-walk 7, plain search on the projection", String(plain))
    # Repetition memory and the decomposed d=3 model take the same path.
    assert_equal(circuit_distance_reduced(_circuit_dem("repetition", 5, "Z", False)).upper, 5, "repetition d=5")
    assert_equal(circuit_distance_reduced(_circuit_dem("surface", 3, "Z", True)).upper, 3, "decomposed d=3")


def test_reduced_budget_and_args() raises:
    var dem = _circuit_dem("surface", 5, "Z", False)
    var r1 = circuit_distance_reduced(dem, node_limit=1)
    assert_equal(r1.status, "budget", String(r1))
    assert_equal(r1.lower, 1, String(r1))
    assert_equal(r1.upper, -1, String(r1))
    assert_equal(len(r1.witness), 0, String(r1))
    var r2 = circuit_distance_reduced(dem, max_weight=4)
    assert_equal(r2.status, "above_max_weight", String(r2))
    assert_equal(r2.lower, 5, String(r2))
    assert_equal(len(circuit_distance_with_sectors(dem, List[List[Int]]()).witness), 5, "no sectors = plain search")
    var msgs = List[String]()
    var empty = List[List[Int]]()
    empty.append(List[Int]())
    var big = List[List[Int]]()
    var b: List[Int] = [0, dem.num_detectors]
    big.append(b^)
    var neg = List[List[Int]]()
    var n: List[Int] = [-1]
    neg.append(n^)
    for secs in [empty.copy(), big.copy(), neg.copy()]:
        try:
            _ = circuit_distance_with_sectors(dem, secs)
            msgs.append(String())
        except e:
            msgs.append(String(e))
    try:
        _ = projected_distance(dem, List[Int]())
        msgs.append(String())
    except e:
        msgs.append(String(e))
    for m in msgs:
        assert_true(m.startswith("QM-E741"), m)
        assert_true(m.find("Hint:") >= 0, m)
    print("  reduced: budget, max_weight, empty sector list, 4 of 4 bad sectors raise QM-E741")


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
    test_projection_random()
    test_graphlike_random()
    test_reduced_circuits()
    test_reduced_budget_and_args()
    print("test_qec_distance OK")
