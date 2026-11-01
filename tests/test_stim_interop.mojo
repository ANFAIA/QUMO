"""Tests for Stim text import, DEM parsing and sampling, the detector lint and windows."""

from std.testing import assert_equal, assert_true
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.qec.codes import NoiseParams, generate_code, to_stim
from qumo.qec.dem import detector_error_model
from qumo.qec.decoders import MatchingGraph, count_logical_errors, UnionFindDecoder, graphlike_dem
from qumo.qec.sparse_blossom import SparseBlossomDecoder
from qumo.qec.stim_text import from_stim, parse_dem, sample_dem
from qumo.qec.lint import missing_detectors, count_determined_measurements
from qumo.qec.window import plan_windows, detector_rounds, SlidingWindowDecoder
from qumo.sim.frame import FrameSimulator
from qumo.sim.stabilizer import sample_stabilizer


def test_round_trip_generated() raises:
    var cc = generate_code("surface_z", 3, 3, NoiseParams.uniform(0.002))
    var text = to_stim(cc.circuit)
    var back = from_stim(text)
    assert_equal(back.circuit.num_detectors, cc.circuit.num_detectors, "detectors")
    assert_equal(back.circuit.num_measurements, cc.circuit.num_measurements, "records")
    assert_equal(to_stim(back.circuit), text, "text round trip")
    assert_equal(detector_error_model(back.circuit).to_text(), detector_error_model(cc.circuit).to_text(), "same DEM")
    print("  generated circuit survives to_stim -> from_stim")


def test_syntax() raises:
    var text = String("""# comment
QUBIT_COORDS(1, 2) 0
QUBIT_COORDS(3, 4) 1
R 0 1
REPEAT 3 {
    H 0
    CX 0 1
    TICK
    MR 1  # trailing comment
    DETECTOR(1, 0) rec[-1]
    SHIFT_COORDS(0, 1)
}
M 0
OBSERVABLE_INCLUDE(0) rec[-1]
""")
    var cc = from_stim(text)
    assert_equal(cc.circuit.num_detectors, 3, "REPEAT unrolled")
    assert_equal(cc.circuit.num_measurements, 4, "records")
    assert_equal(cc.detector_coords[3 * 2 + 1], 2.0, "SHIFT_COORDS")
    assert_equal(cc.qubit_coords[2], 3.0, "QUBIT_COORDS")
    var hit = 0
    for bad in ["FOO 0", "CX 0 1 2", "H sweep[0]", "CX rec[-1] 0", "DETECTOR rec[-1]", "REPEAT 2 {\nH 0"]:
        try:
            _ = from_stim(bad)
        except e:
            if String(e).startswith("QM-E734"):
                hit += 1
    assert_equal(hit, 6, "QM-E734 on unsupported input")
    print("  syntax, REPEAT, coordinates, errors")


def test_mpp_and_friends() raises:
    # Bell pair: XX and ZZ are +1, deterministically, with inversions
    var c = from_stim("H 0\nCX 0 1\nMPP X0*X1 Z0*Z1 !Y0*Y1\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]").circuit.copy()
    var fs = FrameSimulator(c)
    var s = fs.sample_detectors(4096, 1)
    for shot in range(4096):
        assert_equal(len(s.fired(shot)), 0, "Pauli products deterministic")
    assert_equal(c.count(Op.MEASURE), 3, "one record per product")
    var mx = sample_stabilizer(from_stim("RX 0\nMX 0\nRY 1\nMY 1\nMR 2\nMRX 3").circuit.copy(), 16, 2)
    assert_equal(len(mx.counts), 2, "MX, MY of their own eigenstates are fixed, MRX is random")
    # M(p) flips the record, not the qubit: two reads in a row disagree at rate 2p(1-p)
    var flip = from_stim("R 0\nM(0.2) 0\nM 0\nDETECTOR rec[-1] rec[-2]").circuit.copy()
    var fl = FrameSimulator(flip).sample_detectors(100000, 3)
    var count = 0
    for shot in range(100000):
        count += len(fl.fired(shot))
    assert_true(abs(Float64(count) / 100000.0 - 0.2) < 0.01, "record flip only: " + String(count))
    print("  MPP, MX, MY, RX, RY, MR, M(p)")


def test_pauli_channel_2_and_e() raises:
    # PAULI_CHANNEL_2 with only XZ: X on 0 and Z on 1
    var pc = String("PAULI_CHANNEL_2(0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0) 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]")
    var c = from_stim(pc).circuit.copy()
    var s = FrameSimulator(c).sample_detectors(100000, 5)
    var d0 = 0
    var d1 = 0
    for shot in range(100000):
        d0 += 1 if s.get(shot, 0) else 0
        d1 += 1 if s.get(shot, 1) else 0
    assert_true(abs(Float64(d0) / 100000.0 - 0.3) < 0.01, "X on qubit 0")
    assert_equal(d1, 0, "Z on qubit 1 is invisible to M")
    assert_true(to_stim(c).find("PAULI_CHANNEL_2(") >= 0, "export")
    var dem = detector_error_model(c)
    assert_equal(dem.num_errors(), 1, "one mechanism")
    # E: correlated X0 X1, ELSE: X2
    var e = from_stim("E(0.25) X0 X1\nELSE_CORRELATED_ERROR(0.5) X2\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]").circuit.copy()
    var se = FrameSimulator(e).sample_detectors(100000, 7)
    var both = 0
    var third = 0
    var clash = 0
    for shot in range(100000):
        var a = se.get(shot, 0)
        var b = se.get(shot, 1)
        var t = se.get(shot, 2)
        assert_true(a == b, "correlated")
        both += 1 if a else 0
        third += 1 if t else 0
        clash += 1 if a and t else 0
    assert_true(abs(Float64(both) / 100000.0 - 0.25) < 0.01, "E rate")
    assert_true(abs(Float64(third) / 100000.0 - 0.375) < 0.01, "ELSE rate 0.75 * 0.5")
    assert_equal(clash, 0, "ELSE never fires after E")
    var tab = sample_stabilizer(from_stim("E(1) X0 Z1\nM 0 1").circuit.copy(), 8, 1)
    assert_equal(len(tab.counts), 1, "tableau takes E")
    assert_true(to_stim(from_stim("E(0.5) X0 Y2 Z3").circuit.copy()).find("E(0.5) X0 Y2 Z3") >= 0, "E export")
    print("  PAULI_CHANNEL_2, E, ELSE_CORRELATED_ERROR")


def test_dem_text() raises:
    var cc = generate_code("surface_z", 5, 5, NoiseParams.uniform(0.003))
    var dem = detector_error_model(cc.circuit)
    var text = dem.to_text()
    var back = parse_dem(text)
    assert_equal(back.num_detectors, dem.num_detectors, "detectors")
    assert_equal(back.num_observables, dem.num_observables, "observables")
    assert_equal(back.num_errors(), dem.num_errors(), "errors")
    for i in range(dem.num_errors()):
        assert_equal(back.errors[i].dets, dem.errors[i].dets, "symptom")
        assert_equal(back.errors[i].obs, dem.errors[i].obs, "observables")
        assert_equal(len(back.errors[i].comp_dets), len(dem.errors[i].comp_dets), "decomposition")
    var rep = parse_dem("detector D0\nlogical_observable L0\nrepeat 3 {\n  error(0.5) D0 ^ D1 L0\n  shift_detectors 2\n}\nerror(0.25) D0")
    assert_equal(rep.num_errors(), 4, "repeat expanded")
    assert_equal(rep.errors[3].dets[0], 6, "shift accumulates")
    assert_equal(rep.errors[1].dets, [2, 3], "XOR of the pieces")
    var hit = False
    try:
        _ = parse_dem("error(0.1) Q3")
    except e:
        hit = String(e).startswith("QM-E735")
    assert_true(hit, "QM-E735")
    print("  parse_dem round trip, repeat, shift_detectors")


def test_dem_sampler() raises:
    var cc = generate_code("surface_z", 3, 3, NoiseParams.uniform(0.01))
    var dem = detector_error_model(cc.circuit)
    var shots = 500000
    var a = sample_dem(dem, shots, 11)
    var b = FrameSimulator(cc.circuit).sample_detectors(shots, 13)
    var worst = 0.0
    for d in range(dem.num_detectors):
        var ca = 0
        var cb = 0
        for s in range(shots):
            ca += 1 if a.get(s, d) else 0
            cb += 1 if b.get(s, d) else 0
        worst = max(worst, abs(Float64(ca - cb)) / Float64(shots))
    assert_true(worst < 0.004, "detector marginals agree with the frames: " + String(worst))
    print("  sample_dem marginals vs frames: worst", worst)


def test_missing_detectors() raises:
    var cc = generate_code("surface_z", 3, 3, NoiseParams.noiseless())
    var full = cc.circuit.copy()
    assert_equal(len(missing_detectors(full)), 0, "a complete circuit misses nothing")
    # drop one detector
    var dropped = full.empty_like()
    dropped.records = full.records.copy()
    dropped.num_measurements = full.num_measurements
    dropped.num_observables = full.num_observables
    var skipped = False
    var count = 0
    for g in full.ops:
        if g.kind == Op.DETECTOR and not skipped and count == 4:
            skipped = True
            continue
        if g.kind == Op.DETECTOR:
            count += 1
            var h = g.copy()
            h.q2 = dropped.num_detectors
            dropped.num_detectors += 1
            dropped.append(h)
            continue
        dropped.append(g)
    var miss = missing_detectors(dropped)
    assert_equal(len(miss), 1, "the deleted detector is found")
    assert_true(count_determined_measurements(full) >= full.num_detectors, "determined measurements")
    print("  missing_detectors finds a deleted detector:", len(miss), "of", full.num_detectors)


def test_window_equals_plain_at_full_size() raises:
    var cc = generate_code("surface_z", 5, 5, NoiseParams.uniform(0.005))
    var dem = detector_error_model(cc.circuit)
    dem.detector_coords = cc.detector_coords.copy()
    var rounds = detector_rounds(dem)
    var plan = plan_windows(dem, rounds, 100, 100)
    assert_equal(len(plan.dems), 1, "one window")
    var inner = List[SparseBlossomDecoder]()
    inner.append(SparseBlossomDecoder(MatchingGraph(plan.dems[0])))
    var w = SlidingWindowDecoder[SparseBlossomDecoder](plan^, inner^)
    var s = FrameSimulator(cc.circuit).sample_detectors(20000, 3)
    var plain = SparseBlossomDecoder(MatchingGraph(dem))
    for shot in range(20000):
        var f = s.fired(shot)
        assert_equal(w.decode(f), plain.decode(f), "window = rounds is the plain decoder")
    print("  window >= rounds equals plain matching")


def window_errors(d: Int, rounds: Int, p: Float64, window: Int, step: Int, shots: Int) raises -> Tuple[Int, Int, Int]:
    var cc = generate_code("surface_z", d, rounds, NoiseParams.uniform(p))
    var dem = detector_error_model(cc.circuit)
    dem.detector_coords = cc.detector_coords.copy()
    var plan = plan_windows(dem, detector_rounds(dem), window, step)
    var inner = List[SparseBlossomDecoder]()
    for sub in plan.dems:
        inner.append(SparseBlossomDecoder(MatchingGraph(sub)))
    var chunks = len(inner)
    var w = SlidingWindowDecoder[SparseBlossomDecoder](plan^, inner^)
    var s = FrameSimulator(cc.circuit).sample_detectors(shots, 9)
    var plain = count_logical_errors(SparseBlossomDecoder(MatchingGraph(dem)), s)
    return (count_logical_errors(w, s), plain, chunks)


def test_window_approaches_plain() raises:
    var d = 5
    var rounds = 24
    var shots = 40000
    var narrow = window_errors(d, rounds, 0.004, 5, 5, shots)
    var mid = window_errors(d, rounds, 0.004, 7, 2, shots)
    var wide = window_errors(d, rounds, 0.004, 13, 4, shots)
    print("  d=5, 24 rounds: plain", narrow[1], "window 5/5", narrow[0], "7/2", mid[0], "13/4", wide[0], "decoders", wide[2])
    assert_true(narrow[0] > mid[0], "committing at the window edge is worse")
    assert_true(Float64(wide[0]) < 1.25 * Float64(wide[1]) + 20, "a wide window is close to plain")
    var hit = False
    try:
        var cc = generate_code("surface_z", 3, 3, NoiseParams.uniform(0.01))
        _ = plan_windows(detector_error_model(cc.circuit), List[Int](), 3, 1)
    except e:
        hit = String(e).startswith("QM-E736")
    assert_true(hit, "QM-E736")
    print("  wide windows approach plain matching")


def main() raises:
    test_round_trip_generated()
    test_syntax()
    test_mpp_and_friends()
    test_pauli_channel_2_and_e()
    test_dem_text()
    test_dem_sampler()
    test_missing_detectors()
    test_window_equals_plain_at_full_size()
    test_window_approaches_plain()
    print("test_stim_interop OK")
