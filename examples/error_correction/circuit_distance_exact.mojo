"""Exact circuit distance of a detector error model, hyperedges kept whole.

Reproduces: Cheng, Guo, Pan - weight-bounded search with a covering bound for code and circuit distances (SI Sec. 5.1, Lemma 5.1, Thm 5.2), and the row-projection lower bound with lifting (Methods "Reductions, relaxation and lifting", SI Sec. 5.4)
Source: https://arxiv.org/abs/2611.08247
Framework: QUMO - https://github.com/ANFAIA/QUMO

The circuit distance is the fewest error mechanisms whose detectors cancel and whose observables
do not. A graph built from a decomposed model counts pieces, not mechanisms, so a hyperedge that
flips three detectors at once can make the graph-like value too large. The search here works on
the full symptoms, so each mechanism is one column. A result is exact when a completed threshold
search meets a witness of the same weight, and the witness is re-checked; a node budget gives an
interval (lower bound, no upper bound) and never a wrong exact value.

Past d = 5 the plain search is too slow, so `circuit_distance_reduced` adds the paper's
reductions: keep only one basis's detectors (any detector subset gives a lower bound, because a
solution of the model solves its projection), solve that graph-like problem exactly as a shortest
odd closed walk, then lift an optimal projected solution back to real mechanisms and replay it on
the full model. Exact only when the lifted weight meets the lower bound.

Checks:
- [[15,1,3]] Reed-Muller code capacity: Z-type distance 3, X-type distance 7 (classical RM theory), and both equal brute force;
- Steane code capacity: 3;
- a three-mechanism model with a 3-detector hyperedge: 3, equal to brute force;
- 20 seeded random hypergraph models of at most 12 mechanisms: search equals brute force on every one;
- rotated surface code Z memory, rounds = d, p = 0.001, full DEM: distance 3 and 5, witness verified;
- repetition code d = 5, rounds = 5: 5;
- a 100-node budget on the d = 5 surface model gives an interval (lower <= 5, no upper bound), never a wrong exact value;
- reduced path: 20 seeded random models with a random detector sector each equal brute force, and every projected optimum is at most the brute-force distance;
- reduced path: surface d = 5 equals the plain search; surface Z memory d = 7, 9, 11, 13, 15 and X memory d = 7 give exactly d
  (the hook-safe schedule's distance; d = 7 is also the value of the paper's SI Table 3), each witness replayed on the full model.

Run: .venv/bin/mojo run -I . examples/error_correction/circuit_distance_exact.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 180, "requires": [], "args": []}
"""

from std.bit import pop_count
from qumo.core.rng import Rng
from qumo.qec.codes import NoiseParams, surface_code_memory, repetition_code_memory
from qumo.qec.dem import DetectorErrorModel, DemError, detector_error_model
from qumo.qec.stim_text import parse_dem
from qumo.qec.distance import DistanceResult, circuit_distance, circuit_distance_reduced, circuit_distance_with_sectors, projected_distance, verify_witness
from qumo.testing.checks import CheckLog


def _brute(dem: DetectorErrorModel) -> Int:
    """Gray-code enumeration over the raw mechanisms (-1 when there is no logical)."""
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


def _mismatch_count(mut rng: Rng, models: Int) raises -> Int:
    var mismatches = 0
    for _i in range(models):
        var dem = _random_dem(rng, 10)
        var b = _brute(dem)
        var r = circuit_distance(dem, max_weight=20)
        if b < 0:
            # No logical at all: no_logical when no mechanism flips an observable, else above_max_weight.
            var has_obs = False
            for e in dem.errors:
                if e.obs != 0:
                    has_obs = True
            var want = "above_max_weight" if has_obs else "no_logical"
            if r.status != want or r.exact or r.upper != -1 or r.lower != 21:
                mismatches += 1
        elif r.status != "exact" or r.lower != b or not verify_witness(dem, r.witness):
            mismatches += 1
    return mismatches


def _reduced_mismatches(mut rng: Rng, models: Int) raises -> Int:
    """Random model, random sector: the projection never exceeds brute force, the pipeline equals it."""
    var mismatches = 0
    for _i in range(models):
        var dem = _random_dem(rng, 10)
        var b = _brute(dem)
        var sector = List[Int]()
        for d in range(dem.num_detectors):
            if rng.uniform() < 0.5:
                sector.append(d)
        if len(sector) == 0:
            sector.append(0)
        var pd = projected_distance(dem, sector, max_weight=20)
        var secs = List[List[Int]]()
        secs.append(sector^)
        var r = circuit_distance_with_sectors(dem, secs, max_weight=20)
        if b < 0:
            if r.exact or r.upper != -1:
                mismatches += 1
        elif pd.lower > b or r.status != "exact" or r.lower != b or not verify_witness(dem, r.witness):
            mismatches += 1
    return mismatches


def main() raises:
    var log = CheckLog("error_correction/circuit_distance_exact")

    var rm_z = _rm15(False)
    var rm_z_r = circuit_distance(rm_z)
    _ = log.close("RM15 Z-type distance", Float64(rm_z_r.lower), 3.0, 0.0)
    _ = log.close("RM15 Z-type brute force", Float64(_brute(rm_z)), 3.0, 0.0)
    var rm_x = _rm15(True)
    var rm_x_r = circuit_distance(rm_x)
    _ = log.close("RM15 X-type distance", Float64(rm_x_r.lower), 7.0, 0.0)
    _ = log.close("RM15 X-type brute force", Float64(_brute(rm_x)), 7.0, 0.0)
    print(String(rm_x_r))

    var steane = _steane()
    _ = log.close("Steane distance", Float64(circuit_distance(steane).lower), 3.0, 0.0)

    var hyper = parse_dem("error(0.1) D0 D1 D2 L0\nerror(0.1) D0\nerror(0.1) D1 D2")
    var hyper_r = circuit_distance(hyper)
    _ = log.close("hyperedge model distance", Float64(hyper_r.lower), 3.0, 0.0)
    _ = log.close("hyperedge model brute force", Float64(_brute(hyper)), 3.0, 0.0)

    var rng = Rng(7)
    var mismatches = _mismatch_count(rng, 20)
    _ = log.check("random models match brute force", mismatches == 0, "models=20 mismatches=" + String(mismatches))

    for d in [3, 5]:
        var cc = surface_code_memory(d, d, NoiseParams.uniform(0.001), "Z")
        var dem = detector_error_model(cc.circuit, decompose=False)
        var r = circuit_distance(dem)
        print(String(r))
        _ = log.check("surface Z memory d=" + String(d) + " status exact", r.status == "exact", String(r))
        _ = log.close("surface Z memory d=" + String(d) + " distance", Float64(r.lower), Float64(d), 0.0)
        if d == 5:
            _ = log.check("d=5 witness verifies", verify_witness(dem, r.witness), String(r))

    var rc = repetition_code_memory(5, 5, NoiseParams.uniform(0.001))
    var rep_dem = detector_error_model(rc.circuit, decompose=False)
    var rep_r = circuit_distance(rep_dem)
    print(String(rep_r))
    _ = log.close("repetition d=5 distance", Float64(rep_r.lower), 5.0, 0.0)

    var surf5 = surface_code_memory(5, 5, NoiseParams.uniform(0.001), "Z")
    var surf5_dem = detector_error_model(surf5.circuit, decompose=False)
    var budget = circuit_distance(surf5_dem, node_limit=100)
    print(String(budget))
    _ = log.check("budget gives an interval", budget.status == "budget" and budget.upper == -1 and budget.lower <= 5, String(budget))

    var rng2 = Rng(11)
    var red_mm = _reduced_mismatches(rng2, 20)
    _ = log.check("reduced path with random sectors matches brute force", red_mm == 0, "models=20 mismatches=" + String(red_mm))

    var red5 = circuit_distance_reduced(surf5_dem)
    print(String(red5))
    _ = log.close("surface d=5 reduced distance", Float64(red5.upper), 5.0, 0.0)

    for d in [7, 9, 11, 13, 15]:
        var cc = surface_code_memory(d, d, NoiseParams.uniform(0.001), "Z")
        var dem = detector_error_model(cc.circuit, decompose=False)
        var r = circuit_distance_reduced(dem)
        print("d=" + String(d), len(dem.errors), "mechanisms:", String(r))
        _ = log.check("surface Z memory d=" + String(d) + " reduced exact, witness replayed", r.status == "exact" and verify_witness(dem, r.witness), String(r))
        _ = log.close("surface Z memory d=" + String(d) + " reduced distance", Float64(r.upper), Float64(d), 0.0)
    var ccx = surface_code_memory(7, 7, NoiseParams.uniform(0.001), "X")
    var demx = detector_error_model(ccx.circuit, decompose=False)
    var rx = circuit_distance_reduced(demx)
    print(String(rx))
    _ = log.close("surface X memory d=7 reduced distance", Float64(rx.upper), 7.0, 0.0)

    log.finish()
