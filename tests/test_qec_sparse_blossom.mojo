"""Sparse blossom (qumo.qec.sparse_blossom) exactness and integration tests.

Exactness oracle: `MatchingOracle` = integer all-pairs shortest paths on the
*same* discretised weights + the dense exact matcher. For every syndrome we
require
  * primal weight (sum of oracle distances of the returned pairs) == optimum,
  * dual objective reported by the decoder (`last_weight`) == optimum,
  * every detection event matched exactly once,
  * every pair's observable mask is the mask of some shortest path between
    its endpoints (`path_ok`) and their XOR equals `decode()`.
Any mismatch is a bug. Syndromes: sampled shots + dense random syndromes on
surface codes (Z/X) and repetition codes, d = 3, 5, 7, p in {0.001, 0.005,
0.01}, plus random graphs with heterogeneous weights / multiple observables.
"""

from std.testing import assert_equal, assert_true

from qumo.sim.rng import Rng
from qumo.sim.frame import FrameSimulator
from qumo.qec.codes import NoiseParams, generate_code
from qumo.qec.dem import detector_error_model, DetectorErrorModel, DemError
from qumo.qec.decoders import MatchingGraph, MwpmDecoder, decode_samples
from qumo.qec.experiments import memory_experiment, canonical_decoder_name
from qumo.qec.sparse_blossom import SparseBlossomDecoder, MatchingOracle, discretize_weights


def check_syndrome(mut sb: SparseBlossomDecoder, orc: MatchingOracle, fired: List[Int], n: Int) raises -> Bool:
    var edges = sb.decode_to_edges(fired)
    var dual = sb.last_weight()
    var primal = 0
    var cnt = List[Int](length=n, fill=0)
    var obs: UInt64 = 0
    var ok = True
    for e in edges:
        var dd = orc.distance(e.a, e.b)
        if dd < 0:
            ok = False
        primal += dd
        cnt[e.a] += 1
        if e.b >= 0:
            cnt[e.b] += 1
        obs ^= e.obs
        if not orc.path_ok(e.a, e.b, e.obs):
            ok = False
    var tot = 0
    for c in cnt:
        tot += c
    if tot != len(fired):
        ok = False
    for f in fired:
        if cnt[f] != 1:
            ok = False
    if sb.decode(fired) != obs:
        ok = False
    var opt = orc.optimal_weight(fired)
    if primal != opt or dual != opt or not ok:
        print("  MISMATCH k=", len(fired), "primal", primal, "dual", dual, "optimum", opt, "pairs/observables ok", ok)
        return False
    return True


def random_syndrome(n: Int, k: Int, mut rng: Rng) -> List[Int]:
    var flags = List[Bool](length=n, fill=False)
    for _ in range(k):
        flags[rng.below(n)] = True
    var out = List[Int]()
    for i in range(n):
        if flags[i]:
            out.append(i)
    return out^


def exactness_on_code(code: String, d: Int, p: Float64, nsampled: Int, nrandom: Int, seed: UInt64) raises -> Int:
    var cc = generate_code(code, d, d, NoiseParams.uniform(p))
    var dem = detector_error_model(cc.circuit)
    var g = MatchingGraph(dem)
    var sb = SparseBlossomDecoder(g)
    var orc = MatchingOracle(g)
    var ds = FrameSimulator(cc.circuit).sample_detectors(nsampled, seed)
    var rng = Rng(seed + 17)
    var bad = 0
    var maxk = 0
    for s in range(nsampled + nrandom):
        var fired: List[Int]
        if s < nsampled:
            fired = ds.fired(s)
        else:
            fired = random_syndrome(g.num_detectors, 1 + rng.below(12 * d + 4), rng)
        maxk = max(maxk, len(fired))
        if not check_syndrome(sb, orc, fired, g.num_detectors):
            bad += 1
    assert_equal(sb.anomalies, 0, "sparse blossom internal invariant violated (QM-E712)")
    assert_equal(sb.failures, 0)
    print("  ", code, "d=" + String(d), "p=" + String(p), String(nsampled + nrandom) + " syndromes (max k " + String(maxk) + "):", bad, "mismatches")
    return bad


def random_graph_dem(n: Int, mut rng: Rng, few_weights: Bool, pbnd: Float64, nobs: Int) -> DetectorErrorModel:
    """Connected random graph (spanning tree + extra edges), heterogeneous or
    few distinct weights (many ties), random observables, some boundary edges."""
    var dem = DetectorErrorModel(n, nobs)
    var few: List[Float64] = [0.01, 0.05, 0.002]

    def pick(mut rng: Rng, f: Bool, pl: List[Float64]) -> Float64:
        if f:
            return pl[rng.below(3)]
        return 1e-4 + 0.35 * rng.uniform() * rng.uniform()

    def obs_of(mut rng: Rng, nobs: Int) -> UInt64:
        var o: UInt64 = 0
        for k in range(nobs):
            if rng.below(4) == 0:
                o |= UInt64(1) << UInt64(k)
        return o

    for v in range(1, n):
        var u = rng.below(v)
        dem.errors.append(DemError(pick(rng, few_weights, few), [u, v], obs_of(rng, nobs)))
    for _ in range(n + rng.below(2 * n)):
        var u = rng.below(n)
        var v = rng.below(n)
        if u != v:
            dem.errors.append(DemError(pick(rng, few_weights, few), [min(u, v), max(u, v)], obs_of(rng, nobs)))
    var nb = 0
    for v in range(n):
        if rng.uniform() < pbnd:
            dem.errors.append(DemError(pick(rng, few_weights, few), [v], obs_of(rng, nobs)))
            nb += 1
    if nb == 0:
        dem.errors.append(DemError(pick(rng, few_weights, few), [rng.below(n)], obs_of(rng, nobs)))
    return dem^


def main() raises:
    # 1. Tiny hand-checked instance: path 0 - 1 - 2 - 3 with boundary edges at
    #    both ends; weights (via p) make {0,1} + {2,3} optimal for all four.
    var dem = DetectorErrorModel(4, 1)
    dem.errors.append(DemError(0.1, [0, 1], 0))
    dem.errors.append(DemError(0.001, [1, 2], 1))
    dem.errors.append(DemError(0.1, [2, 3], 0))
    dem.errors.append(DemError(0.01, [0], 1))
    dem.errors.append(DemError(0.01, [3], 0))
    var tg = MatchingGraph(dem)
    var tsb = SparseBlossomDecoder(tg)
    var all4: List[Int] = [0, 1, 2, 3]
    var m = tsb.decode_to_matches(all4)
    assert_equal(len(m), 4)
    assert_true((m[0] == 0 and m[1] == 1) or (m[0] == 1 and m[1] == 0) or (m[0] == 2 and m[1] == 3) or (m[0] == 3 and m[1] == 2))
    var wi = discretize_weights(tg)
    assert_equal(tsb.last_weight(), wi[tg.edge_between(0, 1)] + wi[tg.edge_between(2, 3)])
    assert_equal(tsb.decode(all4), UInt64(0))
    var mid: List[Int] = [1, 2]
    assert_equal(tsb.decode(mid), UInt64(1))  # direct 1-2 (w 6.9) beats both-to-boundary (w 13.6)
    var one: List[Int] = [0]
    assert_equal(tsb.decode(one), UInt64(1))  # boundary edge at 0 flips L0
    var empty = List[Int]()
    assert_equal(tsb.decode(empty), UInt64(0))
    var junk: List[Int] = [0, 0, 7, -1]  # repeated / out-of-range ids are skipped
    assert_equal(tsb.decode(junk), UInt64(1))
    assert_equal(tsb.ignored, 3)
    assert_equal(tsb.anomalies, 0)
    print("hand-checked instance ok")

    # 2. Every single fault of the surface (Z/X) and repetition codes is corrected.
    var codes: List[String] = ["surface_z", "surface_x", "repetition"]
    for c in codes:
        for d in range(3, 8, 2):
            var cc = generate_code(c, d, d, NoiseParams.uniform(0.003))
            var cdem = detector_error_model(cc.circuit)
            var sb = SparseBlossomDecoder.from_dem(cdem)
            for e in cdem.errors:  # includes decomposed hyperedges (Y errors)
                assert_equal(sb.decode(e.dets), e.obs, c + " d=" + String(d) + ": single fault not corrected")
    print("single faults corrected (surface Z/X, repetition; d = 3, 5, 7)")

    # 3. Randomized exactness suite on code graphs.
    var bad = 0
    var ps: List[Float64] = [0.001, 0.005, 0.01]
    for c in codes:
        for d in range(3, 8, 2):
            for p in ps:
                bad += exactness_on_code(c, d, p, 250, 120, UInt64(31 + d))
    assert_equal(bad, 0, "sparse blossom is not a minimum-weight perfect matching")
    print("exactness on code graphs ok")

    # 4. Random graphs: heterogeneous weights, ties, up to 3 observables.
    var rng = Rng(2025)
    var rbad = 0
    var rtot = 0
    for gi in range(300):
        var n = 4 + rng.below(80)
        var rdem = random_graph_dem(n, rng, rng.below(2) == 0, 0.02 + 0.4 * rng.uniform(), 1 + rng.below(3))
        var g = MatchingGraph(rdem)
        var sb = SparseBlossomDecoder(g)
        var orc = MatchingOracle(g)
        for _ in range(25):
            var fired = random_syndrome(n, 1 + rng.below(max(1, n // (1 + rng.below(4)))), rng)
            if not check_syndrome(sb, orc, fired, n):
                rbad += 1
                print("    random graph", gi, "n", n)
            rtot += 1
        assert_equal(sb.anomalies, 0)
    assert_equal(rbad, 0, "sparse blossom mismatch on random graphs")
    print("exactness on", rtot, "random-graph syndromes ok")

    # 5. Graph without any boundary: even syndromes are matched exactly, an
    #    odd component cannot be and is counted in `failures` (no exception).
    var nb = DetectorErrorModel(6, 1)
    nb.errors.append(DemError(0.01, [0, 1], 0))
    nb.errors.append(DemError(0.02, [1, 2], 1))
    nb.errors.append(DemError(0.01, [2, 0], 0))
    nb.errors.append(DemError(0.03, [3, 4], 1))  # second component {3, 4, 5}
    nb.errors.append(DemError(0.03, [4, 5], 0))
    var nbg = MatchingGraph(nb)
    var nbs = SparseBlossomDecoder(nbg)
    var nbo = MatchingOracle(nbg)
    var ev: List[Int] = [0, 2, 3, 5]
    assert_true(check_syndrome(nbs, nbo, ev, 6))
    assert_equal(nbs.decode(ev), UInt64(1))  # 0-2 directly (L0 even), 3-4-5 (L0 odd)
    var odd: List[Int] = [0, 1, 2, 4, 5]
    _ = nbs.decode(odd)
    assert_equal(nbs.failures, 1, "odd component without boundary must be reported")
    assert_equal(nbs.anomalies, 0)
    var after: List[Int] = [3, 5]
    assert_equal(nbs.decode(after), UInt64(1), "state must be clean after a failure")
    print("no-boundary graphs ok")

    # 6. Thread copies (decode_samples) == serial decoding; agreement with the
    #    dense exact decoder on sampled shots (ties may break differently).
    var cc7 = generate_code("surface_z", 5, 5, NoiseParams.uniform(0.006))
    var dem7 = detector_error_model(cc7.circuit)
    var g7 = MatchingGraph(dem7)
    var ds7 = FrameSimulator(cc7.circuit).sample_detectors(20000, 3)
    var sb7 = SparseBlossomDecoder(g7)
    var par = decode_samples(sb7, ds7)
    var dense = MwpmDecoder(g7.copy())
    var diff = 0
    var err_sb = 0
    var err_dense = 0
    for s in range(ds7.shots):
        var f = ds7.fired(s)
        var o = sb7.decode(f)
        assert_equal(par[s], o, "parallel decode differs from serial")
        var od = dense.decode(f)
        if o != od:
            diff += 1
        if o != ds7.observables_mask(s):
            err_sb += 1
        if od != ds7.observables_mask(s):
            err_dense += 1
    print("d=5 p=0.6%: sparse blossom", err_sb, "errors, dense MWPM", err_dense, "errors, predictions differ on", diff, "of", ds7.shots, "shots (degenerate ties)")
    assert_true(diff * 200 <= ds7.shots, "sparse and dense MWPM should agree on >= 99.5% of shots")
    assert_true(abs(err_sb - err_dense) * 20 <= max(err_dense, 20), "logical error counts should agree")

    # 7. Experiment registry: 'mwpm' is sparse blossom, 'mwpm_dense' the old one.
    assert_equal(canonical_decoder_name("MWPM"), "mwpm")
    assert_equal(canonical_decoder_name("sparse_blossom"), "sparse_blossom")
    assert_equal(canonical_decoder_name("mwpm_dense"), "mwpm_dense")
    assert_equal(canonical_decoder_name("nope"), "")
    var ra = memory_experiment("surface_z", 3, -1, 0.004, 60000, "mwpm", 5, -1)
    var rb = memory_experiment("surface_z", 3, -1, 0.004, 60000, "sparse_blossom", 5, -1)
    var rc = memory_experiment("surface_z", 3, -1, 0.004, 60000, "mwpm_dense", 5, -1)
    print(ra)
    print(rc)
    assert_equal(ra.errors, rb.errors, "'mwpm' must be the sparse blossom decoder")
    assert_true(abs(ra.errors - rc.errors) * 20 <= max(rc.errors, 20), "sparse vs dense MWPM logical error counts")
    var raised = False
    try:
        _ = memory_experiment("surface_z", 3, -1, 0.004, 1000, "bogus", 5, -1)
    except e:
        raised = String(e).find("QM-E707") >= 0
    assert_true(raised, "unknown decoder must raise QM-E707")
    print("test_qec_sparse_blossom OK")
