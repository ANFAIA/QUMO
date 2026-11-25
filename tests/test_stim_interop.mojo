"""Tests for Stim text import, DEM parsing and sampling, the detector lint and windows."""

from std.testing import assert_equal, assert_true
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.program import MAX_NESTING
from qumo.qec.codes import NoiseParams, generate_code, to_stim
from qumo.qec.dem import detector_error_model
from qumo.qec.decoders import MatchingGraph, count_logical_errors, UnionFindDecoder, graphlike_dem
from qumo.qec.sparse_blossom import SparseBlossomDecoder
from qumo.qec.correlated import CorrelatedMatchingDecoder
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


def test_correlated_p0_p1_chain() raises:
    # E(0) never fires and must not disturb the chain; E(1) always fires
    var n = 100000
    var zero = FrameSimulator(from_stim("E(0) X0\nM 0\nDETECTOR rec[-1]").circuit.copy()).sample_detectors(n, 21)
    var fired0 = 0
    for shot in range(n):
        fired0 += len(zero.fired(shot))
    assert_equal(fired0, 0, "E(0) never fires")
    var one = FrameSimulator(from_stim("E(1) X0\nM 0\nDETECTOR rec[-1]").circuit.copy()).sample_detectors(n, 22)
    var fired1 = 0
    for shot in range(n):
        fired1 += len(one.fired(shot))
    assert_equal(fired1, n, "E(1) always fires")
    # E(0) then ELSE(0.5): the ELSE fires on about half the shots, E's chain never blocks it
    var ez = FrameSimulator(from_stim("E(0) X0\nELSE_CORRELATED_ERROR(0.5) X1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]").circuit.copy()).sample_detectors(n, 23)
    var else_hits = 0
    var e_hits = 0
    for shot in range(n):
        e_hits += 1 if ez.get(shot, 0) else 0
        else_hits += 1 if ez.get(shot, 1) else 0
    assert_equal(e_hits, 0, "E(0) never fires before ELSE")
    assert_true(abs(Float64(else_hits) / Float64(n) - 0.5) < 0.01, "ELSE(0.5) after E(0): " + String(else_hits))
    # E(1) then ELSE(1): the chain is set by E, so ELSE never fires
    var eo = FrameSimulator(from_stim("E(1) X0\nELSE_CORRELATED_ERROR(1) X1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]").circuit.copy()).sample_detectors(n, 24)
    var first = 0
    var second = 0
    for shot in range(n):
        first += 1 if eo.get(shot, 0) else 0
        second += 1 if eo.get(shot, 1) else 0
    assert_equal(first, n, "E(1) fires on every shot")
    assert_equal(second, 0, "ELSE(1) never fires after E(1)")
    print("  E(0), E(1), E(0)+ELSE(0.5), E(1)+ELSE(1)")


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


def _fired(*ids: Int) -> List[Int]:
    var out = List[Int]()
    for i in ids:
        out.append(i)
    return out^


def test_dem_piece_without_detectors() raises:
    # "D0 ^ L0": the second piece only carries an observable; it folds into the D0 piece
    var dem = parse_dem("error(0.1) D0 ^ L0\nerror(0.1) D0 D1\nerror(0.1) D1")
    var g = MatchingGraph(dem)
    assert_equal(g.num_edges(), 3, "edges")
    var mw = SparseBlossomDecoder(g)
    assert_equal(mw.decode(_fired(0)), UInt64(1), "mwpm: {D0} flips L0")
    var uf = UnionFindDecoder(g.copy())
    assert_equal(uf.decode(_fired(0)), UInt64(1), "union-find: {D0} flips L0")
    var cm = CorrelatedMatchingDecoder(dem)
    assert_equal(cm.decode(_fired(0)), UInt64(1), "correlated: {D0} flips L0")
    # leading empty piece, and an empty piece between two real ones
    var dem2 = parse_dem("error(0.1) L0 ^ D0\nerror(0.1) D0 ^ L0 ^ D1")
    var g2 = MatchingGraph(dem2)
    assert_equal(g2.num_edges(), 2, "edges 2")
    # no detector at all: undetectable, skipped
    var dem3 = parse_dem("error(0.1) L0\nerror(0.1) ^ L0\nerror(0.2) D0")
    var g3 = MatchingGraph(dem3)
    assert_equal(g3.num_edges(), 1, "undetectable mechanisms add no edge")
    assert_equal(g3.num_undetectable, 2, "counted")
    var mw3 = SparseBlossomDecoder(g3)
    assert_equal(mw3.decode(_fired(0)), UInt64(0), "no flip")
    var cm3 = CorrelatedMatchingDecoder(dem3)
    assert_equal(cm3.decode(_fired(0)), UInt64(0), "correlated no flip")
    # the exported graph-like text agrees with the graph
    var gl = graphlike_dem(dem)
    assert_equal(MatchingGraph(gl).num_edges(), 3, "graphlike_dem keeps the edges")
    assert_equal(len(gl.errors[0].comp_dets), 0, "graphlike_dem: folded piece is a plain error")
    assert_equal(gl.errors[0].obs, UInt64(1), "observable kept")
    print("  DEM pieces without detectors fold into a neighbour or are skipped")


def test_dem_piece_with_three_detectors() raises:
    # a piece with 3 detectors is decomposed against existing edges like an undecomposed hyperedge
    var text = "error(0.1) D0 D1\nerror(0.1) D2\nerror(0.01) D0 D1 D2 ^ D3\nerror(0.1) D3"
    var dem = parse_dem(text)
    var g = MatchingGraph(dem)
    assert_equal(g.num_hyperedges, 1, "counted once")
    assert_equal(g.num_dropped, 0, "decomposed, not dropped")
    assert_equal(g.num_edges(), 3, "no new edges")
    assert_true(g.ep[g.edge_between(2, g.boundary)] > 0.1, "D2 edge got the 3-detector piece's probability")
    assert_true(g.ep[g.edge_between(3, g.boundary)] > 0.1, "D3 edge got the second piece")
    var gl = graphlike_dem(dem)
    for c in gl.errors[2].comp_dets:
        assert_true(len(c) <= 2, "exported pieces are graph-like")
    assert_equal(MatchingGraph(gl).num_dropped, 0, "re-import is stable")
    var cm = CorrelatedMatchingDecoder(dem)
    assert_equal(cm.decode(_fired(2, 3)), UInt64(0), "correlated runs")
    # a piece that cannot be decomposed is dropped and counted, not lost silently
    var g2 = MatchingGraph(parse_dem("error(0.1) D0\nerror(0.1) D0 D1 D2 ^ D3\nerror(0.1) D3"))
    assert_equal(g2.num_dropped, 1, "undecomposable piece is counted as dropped")
    print("  DEM pieces with 3 detectors follow the hyperedge rule")


def _nest(open_line: String, body: String, depth: Int) -> String:
    """`open_line {` repeated `depth` times around `body`."""
    var text = String("")
    for _ in range(depth):
        text += open_line + " {\n"
    text += body
    for _ in range(depth):
        text += "}\n"
    return text^


def _circuit_error(text: String) -> String:
    try:
        _ = from_stim(text)
    except e:
        return String(e)
    return ""


def _dem_error_of(text: String) -> String:
    try:
        _ = parse_dem(text)
    except e:
        return String(e)
    return ""


def test_nesting_and_budget() raises:
    # nesting: MAX_NESTING levels parse, one more is QM-E734 / QM-E735
    var c = from_stim(_nest("REPEAT 1", "H 0\n", MAX_NESTING)).circuit.copy()
    assert_equal(c.count(Op.H), 1, "innermost body once per level")
    assert_equal(_circuit_error(_nest("REPEAT 1", "H 0\n", MAX_NESTING)), "", "circuit at the limit parses")
    var over = _circuit_error(_nest("REPEAT 1", "H 0\n", MAX_NESTING + 1))
    assert_true(over.startswith("QM-E734") and over.find(String(MAX_NESTING)) >= 0, "circuit one past the limit: " + over)
    assert_equal(_dem_error_of(_nest("repeat 1", "error(0.1) D0\n", MAX_NESTING)), "", "DEM at the limit parses")
    var dem_over = _dem_error_of(_nest("repeat 1", "error(0.1) D0\n", 3000))
    assert_true(dem_over.startswith("QM-E735") and dem_over.find(String(MAX_NESTING)) >= 0, "DEM far past the limit: " + dem_over)
    # budget: a DEM repeat that expands past the budget is refused before it fills memory
    var big = _dem_error_of("repeat 1000000000 {\nerror(0.1) D0\n}\n")
    assert_true(big.startswith("QM-E735") and big.find("instructions after unrolling") >= 0, "DEM budget: " + big)
    var empty = _circuit_error("REPEAT 1000000000 {\n}\nH 0\n")
    assert_true(empty.startswith("QM-E734"), "empty REPEAT bodies are charged too: " + empty)
    # normal repeated circuits are unaffected
    var rep = from_stim("H 0\nREPEAT 2000 {\n  X 0\n  TICK\n}\nH 0\n").circuit.copy()
    assert_equal(rep.count(Op.X), 2000, "REPEAT 2000 unrolled")
    print("  nesting limit, repeat budget, normal repeats")


def main() raises:
    test_round_trip_generated()
    test_syntax()
    test_mpp_and_friends()
    test_pauli_channel_2_and_e()
    test_correlated_p0_p1_chain()
    test_dem_text()
    test_dem_sampler()
    test_missing_detectors()
    test_window_equals_plain_at_full_size()
    test_window_approaches_plain()
    test_dem_piece_without_detectors()
    test_dem_piece_with_three_detectors()
    test_nesting_and_budget()
    print("test_stim_interop OK")
