"""Tests for qumo.qec.dem_compare: Gather, the DEMVal verdict, the paper's defect classes."""

from std.math import sqrt
from std.testing import assert_equal, assert_true
from qumo.qec.codes import NoiseParams, generate_code
from qumo.qec.dem import DetectorErrorModel, DemError, detector_error_model
from qumo.qec.stim_text import from_stim, parse_dem
from qumo.qec.dem_compare import gather_dem, compare_dem


def _surface(decompose: Bool = False) raises -> DetectorErrorModel:
    var cc = generate_code("surface_z", 3, 3, NoiseParams.uniform(0.001))
    return detector_error_model(cc.circuit, decompose)


def test_gather_values() raises:
    var g = gather_dem(parse_dem("error(0.1) D0 ^ D1\nerror(0.2) D1 D0\nerror(0.3) D2 D2 L0\nerror(0) D5\nerror(0.4) D3 ^ D3"))
    assert_equal(g.num_errors(), 2, "equal classes merged, empty class and p = 0 dropped")
    assert_equal(len(g.errors[0].dets), 0, "observable-only class first")
    assert_equal(g.errors[0].obs, UInt64(1), "L0 kept")
    assert_true(abs(g.errors[0].p - 0.3) < 1e-15, "D2 D2 cancels")
    assert_equal(g.errors[1].dets, [0, 1], "sorted full symptom")
    assert_true(abs(g.errors[1].p - 0.26) < 1e-15, "0.1 + 0.2 - 2 * 0.02")
    assert_equal(g.errors[1].num_components(), 0, "undecomposed")
    assert_equal(g.num_detectors, 6, "counts kept")
    print("  gather merges, cancels, drops")


def test_gather_invariance() raises:
    var dem = _surface()
    # shuffle and split: every p becomes a = p/3 and b with a + b - 2ab = p, b's in reverse
    var split = DetectorErrorModel(dem.num_detectors, dem.num_observables)
    for e in dem.errors:
        var a = e.p / 3.0
        split.errors.append(DemError(a, e.dets.copy(), e.obs))
    for i in range(dem.num_errors() - 1, -1, -1):
        ref e = dem.errors[i]
        var a = e.p / 3.0
        split.errors.append(DemError((e.p - a) / (1.0 - 2.0 * a), e.dets.copy(), e.obs))
    var r = compare_dem(dem, split, 1e-12)
    assert_true(r.consistent, "split + shuffle: " + String(r))
    assert_equal(r.shared, dem.num_errors(), "same classes")
    # ^ separators: decomposed and undecomposed models of one circuit
    var r2 = compare_dem(_surface(True), dem, 1e-12)
    assert_true(r2.consistent, "decomposition ignored: " + String(r2))
    var r3 = compare_dem(parse_dem("error(0.1) D0 D1 ^ D2 L0\nerror(0.2) D3 ^ D3 D4"), parse_dem("error(0.1) D2 D1 D0 L0\nerror(0.2) D4"))
    assert_true(r3.consistent and r3.max_rel == 0.0, "pieces XOR-combined")
    print("  gather invariant under shuffle, split, ^")


def test_symmetry() raises:
    var a = _surface()
    var b = a.copy()
    b.errors[3].p *= 1.01
    _ = b.errors.pop()
    b.errors.append(DemError(0.002, [0, 23], 0))
    var ab = compare_dem(a, b)
    var ba = compare_dem(b, a)
    assert_equal(ab.unmatched, ba.unmatched, "u symmetric")
    assert_equal(ab.unmatched, 2, "one class removed, one added")
    assert_equal(ab.max_rel, ba.max_rel, "delta symmetric")
    assert_equal(ab.shared, ba.shared, "shared symmetric")
    assert_equal(len(ab.worst), len(ba.worst), "worst symmetric")
    for i in range(len(ab.worst)):
        assert_equal(ab.worst[i].p_a, ba.worst[i].p_b, "sides swap")
    print("  symmetric")


def test_text_round_trip() raises:
    for name in ["repetition", "surface_z", "surface_x"]:
        for d in [3, 5]:
            for decompose in [True, False]:
                var cc = generate_code(name, d, d, NoiseParams.uniform(0.002))
                var dem = detector_error_model(cc.circuit, decompose)
                var r = compare_dem(dem, parse_dem(dem.to_text()), 1e-12)
                assert_true(r.consistent and r.same_shape and r.unmatched == 0, name + String(d) + ": " + String(r))
                assert_true(r.max_rel <= 1e-15, "text keeps every digit")
                assert_equal(len(r.worst), 0, "nothing to report")
    print("  self-comparison through to_text / parse_dem")


def test_defect_depolarizing_p_over_3() raises:
    # DEPOLARIZE1(p) before M: X and Y flip D0, together 2p/3 (closed form)
    var p = 0.03
    var dem = detector_error_model(from_stim("DEPOLARIZE1(0.03) 0\nM 0\nDETECTOR rec[-1]").circuit.copy())
    var hand = DetectorErrorModel(1, 0)
    hand.errors.append(DemError(2.0 * p / 3.0, [0], 0))
    assert_true(compare_dem(dem, hand, 1e-12).consistent, "extractor = 2p/3")
    var bad = DetectorErrorModel(1, 0)  # qdk-ec 68: each Pauli at p/3
    bad.errors.append(DemError(p / 3.0, [0], 0))
    bad.errors.append(DemError(p / 3.0, [0], 0))
    var r = compare_dem(dem, bad)
    assert_true(not r.consistent, "p/3 caught")
    assert_equal(r.unmatched, 0, "same classes")
    assert_true(abs(r.max_rel - p / 3.0) < 1e-12, "delta = p/3: " + String(r.max_rel))
    print("  defect: depolarizing rate as p/3")


def test_defect_second_qubit_dropped() raises:
    # DEPOLARIZE2(p): D0, D1 and D0 D1 each merge 4 of the 15 Paulis, (1 - sqrt(1 - 16p/15))/2
    var p = 0.01
    var dem = detector_error_model(from_stim("DEPOLARIZE2(0.01) 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]").circuit.copy())
    var q = 0.5 - 0.5 * sqrt(1.0 - 16.0 * p / 15.0)
    var hand = DetectorErrorModel(2, 0)
    hand.errors.append(DemError(q, [0], 0))
    hand.errors.append(DemError(q, [1], 0))
    hand.errors.append(DemError(q, [0, 1], 0))
    assert_true(compare_dem(dem, hand, 1e-12).consistent, "extractor = closed form")
    var bad = DetectorErrorModel(2, 0)  # cudaqx 606: first operand only
    for e in dem.errors:
        var hits_b = False
        for d in e.dets:
            if d == 1:
                hits_b = True
        if not hits_b:
            bad.errors.append(e.copy())
    var r = compare_dem(dem, bad)
    assert_true(not r.consistent, "second qubit's mechanisms missing")
    assert_equal(r.unmatched, 2, "D1 and D0 D1 one-sided")
    print("  defect: two-qubit channel on its first operand only")


def test_defect_observables_merged() raises:
    var a = parse_dem("error(0.01) D0\nerror(0.02) D0 L0\nerror(0.03) D1")
    var bad = parse_dem("error(0.0296) D0\nerror(0.03) D1\nlogical_observable L0")  # cudaqx 607
    var r = compare_dem(a, bad)
    assert_true(not r.consistent, "classes differing in L0 merged")
    assert_true(r.same_shape, "same counts")
    assert_equal(r.unmatched, 1, "D0 L0 one-sided")
    assert_equal(r.shared, 2, "D0, D1 shared")
    assert_true(abs(r.max_rel - (0.0296 - 0.01) / 0.0296) < 1e-12, "D0 rate differs")
    print("  defect: classes differing only in observables merged")


def test_defect_observable_only_dropped() raises:
    var c = from_stim("R 0 1\nX_ERROR(0.01) 0\nX_ERROR(0.02) 1\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]").circuit.copy()
    var dem = detector_error_model(c)
    var bad = DetectorErrorModel(dem.num_detectors, dem.num_observables)  # cudaqx 607
    for e in dem.errors:
        if len(e.dets) > 0:
            bad.errors.append(e.copy())
    var r = compare_dem(dem, bad)
    assert_true(not r.consistent, "zero-syndrome logical error dropped")
    assert_equal(r.unmatched, 1, "one class")
    assert_equal(len(r.worst[0].dets), 0, "no detectors")
    assert_equal(r.worst[0].obs, UInt64(1), "flips L0")
    assert_true(abs(r.worst[0].p_a - 0.01) < 1e-15 and r.worst[0].p_b == 0.0, "rates")
    print("  defect: observable-only mechanism dropped")


def test_defect_duplicates_added() raises:
    var xor_ = parse_dem("error(0.001) D0\nerror(0.001) D0")
    var added = parse_dem("error(0.002) D0")  # cudaqx 607
    var r = compare_dem(xor_, added)
    assert_true(not r.consistent, "added instead of XOR-combined")
    assert_true(abs(r.max_rel - 0.001) < 1e-12, "1 - 0.001998 / 0.002")
    # the same on a whole surface code: split every p in two halves, then add them
    var dem = _surface()
    var summed = DetectorErrorModel(dem.num_detectors, dem.num_observables)
    for e in dem.errors:
        var a = e.p / 2.0
        var b = (e.p - a) / (1.0 - 2.0 * a)
        summed.errors.append(DemError(a + b, e.dets.copy(), e.obs))
    var rs = compare_dem(dem, summed)
    assert_true(not rs.consistent and rs.unmatched == 0, "every class off")
    assert_true(rs.max_rel > 1e-4, "about p/2: " + String(rs.max_rel))
    print("  defect: duplicates added")


def test_tolerance_edges() raises:
    var a = DetectorErrorModel(2, 0)
    a.errors.append(DemError(0.01, [0], 0))
    a.errors.append(DemError(1e-16, [1], 0))
    var b = DetectorErrorModel(2, 0)
    b.errors.append(DemError(0.01, [0], 0))
    var r = compare_dem(a, b)
    assert_true(r.consistent and r.unmatched == 0, "1e-16 one-sided ignored")
    assert_equal(compare_dem(a, b, floor=0.0).unmatched, 1, "floor 0 counts it")
    var at = DetectorErrorModel(2, 0)
    at.errors.append(DemError(0.01, [0], 0))
    at.errors.append(DemError(1e-15, [1], 0))
    assert_true(compare_dem(at, b).consistent, "p == floor ignored")
    var above = DetectorErrorModel(2, 0)
    above.errors.append(DemError(0.01 * (1.0 - 2e-9), [0], 0))
    var below = DetectorErrorModel(2, 0)
    below.errors.append(DemError(0.01 * (1.0 - 5e-10), [0], 0))
    assert_true(not compare_dem(b, above).consistent, "2e-9 > tol")
    assert_true(compare_dem(b, below).consistent, "5e-10 <= tol")
    assert_true(compare_dem(b, above, tol=1e-8).consistent, "tol is a parameter")
    var wide = DetectorErrorModel(3, 0)
    wide.errors.append(DemError(0.01, [0], 0))
    var s = compare_dem(b, wide)
    assert_true(s.consistent and not s.same_shape, "shape reported apart")
    print("  tolerance, floor, shape")


def test_worst_list() raises:
    var a = _surface()
    var b = a.copy()
    for i in range(10):
        b.errors[i].p *= 1.0 + 0.01 * Float64(i + 1)
    var r = compare_dem(a, b, keep=4)
    assert_equal(len(r.worst), 4, "keep")
    for i in range(1, len(r.worst)):
        assert_true(r.worst[i - 1].rel >= r.worst[i].rel, "largest first")
    assert_equal(r.worst[0].rel, r.max_rel, "head is delta")
    assert_equal(len(compare_dem(a, b, keep=0).worst), 0, "keep 0")
    print("  worst offenders, largest first")


def test_invalid_and_zero_probability() raises:
    # merged probability 0 (1 + 1 - 2) is dropped, the other class stays
    var z = gather_dem(parse_dem("error(1) D0\nerror(1) D0\nerror(0.1) D1"))
    assert_equal(z.num_errors(), 1, "p = 1, 1 merges to 0 and is dropped")
    assert_equal(z.errors[0].dets, [1], "D1 kept")
    assert_true(abs(z.errors[0].p - 0.1) < 1e-15, "D1 rate")
    # NaN: not a probability, so the verdict is inconsistent whichever side holds it
    var nan = 0.0 / 0.0
    var nan_a = DetectorErrorModel(2, 0)
    nan_a.errors.append(DemError(nan, [0], 0))
    var one = DetectorErrorModel(2, 0)
    one.errors.append(DemError(0.01, [0], 0))
    var rn = compare_dem(nan_a, one)
    assert_true(not rn.consistent and rn.invalid == 1, "NaN vs 0.01 inconsistent")
    var rn2 = compare_dem(one, nan_a)
    assert_true(not rn2.consistent and rn2.invalid == 1, "0.01 vs NaN inconsistent")
    var empty = DetectorErrorModel(2, 0)
    var rn3 = compare_dem(nan_a, empty)
    assert_true(not rn3.consistent and rn3.invalid == 1, "NaN alone vs empty inconsistent")
    # negative probability is invalid, not silently dropped
    var neg = parse_dem("error(-0.01) D0\nerror(0.02) D0")
    var rneg = compare_dem(neg, parse_dem("error(0.02) D0"))
    assert_true(not rneg.consistent and rneg.invalid == 1, "negative p inconsistent")
    # above 1 is invalid
    var rhi = compare_dem(parse_dem("error(1.5) D0"), parse_dem("error(0.5) D0"))
    assert_true(not rhi.consistent and rhi.invalid == 1, "p > 1 inconsistent")
    print("  zero merged probability dropped, invalid probabilities flagged")


def main() raises:
    test_gather_values()
    test_gather_invariance()
    test_symmetry()
    test_text_round_trip()
    test_defect_depolarizing_p_over_3()
    test_defect_second_qubit_dropped()
    test_defect_observables_merged()
    test_defect_observable_only_dropped()
    test_defect_duplicates_added()
    test_tolerance_edges()
    test_worst_list()
    test_invalid_and_zero_probability()
    print("test_dem_compare OK")
