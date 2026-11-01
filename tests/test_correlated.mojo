"""Tests for correlated matching (qumo.qec.correlated)."""

from std.testing import assert_equal, assert_true
from qumo.ir.circuit import Circuit
from qumo.qec.codes import NoiseParams, generate_code
from qumo.qec.dem import detector_error_model
from qumo.qec.decoders import MatchingGraph, count_logical_errors
from qumo.qec.sparse_blossom import SparseBlossomDecoder
from qumo.qec.correlated import CorrelatedMatchingDecoder, implied_weights
from qumo.qec.experiments import memory_experiment, canonical_decoder_name
from qumo.sim.frame import FrameSimulator


def test_implied_weights() raises:
    var cc = generate_code("surface_z", 5, 5, NoiseParams.uniform(0.003))
    var dem = detector_error_model(cc.circuit)
    var g = MatchingGraph(dem)
    var imp = implied_weights(dem, g)
    var total = 0
    for row in imp:
        total += len(row)
        for w in row:
            assert_true(w.weight >= 0.0, "pi <= 0.5")
    assert_true(total > g.num_edges(), "Y-like errors tie edges together: " + String(total))
    print("  implied weights:", total, "over", g.num_edges(), "edges")


def test_decoder_name() raises:
    assert_equal(canonical_decoder_name("mwpm"), "mwpm")
    assert_equal(canonical_decoder_name("correlated"), "mwpm_correlated")
    assert_equal(canonical_decoder_name("mwpm_correlated"), "mwpm_correlated")
    print("  'mwpm' still means plain matching")


def test_gain_grows_with_distance() raises:
    # same syndromes for both decoders; d = 5 and 7 at p = 0.005 (see docs/QEC.md for the table)
    var cc = generate_code("surface_z", 7, 7, NoiseParams.uniform(0.005))
    var dem = detector_error_model(cc.circuit)
    var fs = FrameSimulator(cc.circuit)
    var s = fs.sample_detectors(400000, 3)
    var plain = count_logical_errors(SparseBlossomDecoder(MatchingGraph(dem)), s)
    var corr = count_logical_errors(CorrelatedMatchingDecoder(dem), s)
    print("  d=7 p=0.005: plain", plain, "correlated", corr)
    assert_true(Float64(corr) < 0.8 * Float64(plain), "correlated beats plain at d=7")
    print("  gain at d = 7")


def test_no_events() raises:
    var cc = generate_code("surface_z", 3, 3, NoiseParams.uniform(0.003))
    var dec = CorrelatedMatchingDecoder(detector_error_model(cc.circuit))
    assert_equal(dec.decode(List[Int]()), UInt64(0), "empty syndrome")
    var copy = dec.copy()
    assert_equal(copy.decode(List[Int]()), UInt64(0), "copies decode too")
    print("  empty syndrome and copies")


def main() raises:
    test_implied_weights()
    test_decoder_name()
    test_no_events()
    test_gain_grows_with_distance()
    print("test_correlated OK")
