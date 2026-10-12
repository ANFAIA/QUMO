from std.math import pi, sqrt, exp
from std.collections import Dict
from std.testing import assert_equal, assert_true

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.rng import Rng
from qumo.sim.stabilizer import TableauSimulator, sample_stabilizer
from qumo.sim.frame import FrameSimulator
from qumo.qec.codes import NoiseParams, surface_code_memory, repetition_code_memory, generate_code, to_stim
from qumo.qec.dem import detector_error_model, depolarize1_independent, combine_independent
from qumo.qec.decoders import (
    MatchingGraph, UnionFindDecoder, MwpmDecoder, LookupTableDecoder, min_weight_boundary_matching,
    count_logical_errors,
)
from qumo.qec.experiments import memory_experiment, threshold_sweep
from qumo.qec.sparse_blossom import SparseBlossomDecoder
from qumo.qec.characterization import (
    NoiseModel, apply_noise_model, thermal_pauli_channel, randomized_benchmarking_1q,
    clifford_group_1q, ReadoutCalibration, readout_calibration_circuits, mitigate_counts,
    expectation_z, t1_experiment, ramsey_experiment, fit_decay,
)


def check_noiseless_detectors(code: String, d: Int) raises:
    var cc = generate_code(code, d, d, NoiseParams.noiseless())
    var fs = FrameSimulator(cc.circuit)
    var ds = fs.sample_detectors(2048, 5)
    assert_equal(ds.num_detection_events(), 0, code + ": noiseless detectors must never fire")
    assert_equal(ds.observable_flip_count(0), 0, code + ": noiseless observable must not flip")
    # absolute values from the tableau simulator are all zero too
    var sim = TableauSimulator(cc.circuit.num_qubits, 3)
    sim.run(cc.circuit)
    var dv = sim.detector_values(cc.circuit)
    for v in dv:
        assert_equal(Int(v), 0)
    assert_equal(Int(sim.observable_values(cc.circuit)[0]), 0)


def random_noisy_clifford(n: Int, depth: Int, mut rng: Rng) -> Circuit:
    var c = Circuit(n)
    for _ in range(depth):
        var r = rng.below(10)
        var q = rng.below(n)
        if r == 0: c.h(q)
        elif r == 1: c.s(q)
        elif r == 2: c.sx(q)
        elif r == 3 and n > 1:
            var t = (q + 1 + rng.below(n - 1)) % n
            c.cx(q, t)
        elif r == 4 and n > 1:
            var t = (q + 1 + rng.below(n - 1)) % n
            c.cz(q, t)
        elif r == 5: c.depolarize1(0.1, q)
        elif r == 6: c.x_error(0.15, q)
        elif r == 7: c.pauli_channel_1(0.05, 0.1, 0.02, q)
        elif r == 8 and n > 1:
            var t = (q + 1 + rng.below(n - 1)) % n
            c.depolarize2(0.2, q, t)
        else:
            c.measure(q, c.num_clbits)
            if rng.below(2) == 0:
                c.reset(q)
    for q in range(n):
        c.measure(q, c.num_clbits)
    return c^


def main() raises:
    # 1. Noiseless circuits: no detection events, deterministic observables.
    for d in range(3, 6, 2):
        check_noiseless_detectors("surface_z", d)
        check_noiseless_detectors("surface_x", d)
        check_noiseless_detectors("repetition", d)
    print("noiseless detectors ok")

    # 2. Frame sampler vs per-shot tableau trajectories (marginals agree).
    var rng = Rng(77)
    for trial in range(12):
        var c = random_noisy_clifford(2 + rng.below(3), 25, rng)
        if c.num_clbits > 20:
            continue
        var fs = FrameSimulator(c)
        var shots = 40000
        var ms = fs.sample_measurements(shots, UInt64(trial))
        var tr = sample_stabilizer(c, shots, UInt64(trial) + 1000)
        for m in range(c.num_measurements):
            # clbit m holds measurement m in this generator
            var p_frame = ms.fraction_one(m)
            var ones = 0
            for s in range(shots):
                if (tr.memory[s] >> m) & 1 == 1:
                    ones += 1
            var p_tab = Float64(ones) / Float64(shots)
            if abs(p_frame - p_tab) > 0.02:
                print(c)
                raise Error("frame vs tableau marginal mismatch, trial " + String(trial) + " meas " + String(m) + ": " + String(p_frame) + " vs " + String(p_tab))
    print("frame vs tableau ok")

    # 3. DEM structure of the repetition code (data depolarization only).
    var p = 0.01
    var rep = repetition_code_memory(3, 2, NoiseParams(0.0, 0.0, 0.0, p))
    var rdem = detector_error_model(rep.circuit)
    var q = depolarize1_independent(p)
    var pxy = combine_independent(q, q)  # X and Y on a data qubit both flip it
    var with_obs = 0
    for e in rdem.errors:
        assert_true(len(e.dets) <= 2, "repetition-code errors are graph-like")
        assert_true(abs(e.p - pxy) < 1e-12 or abs(e.p - combine_independent(pxy, pxy)) < 1e-12, "unexpected probability " + String(e.p))
        if e.obs != 0:
            with_obs += 1
    assert_true(with_obs >= 2, "last data qubit errors flip L0")
    assert_equal(rdem.num_detectors, 6)
    var txt = rdem.to_text()
    assert_true(txt.find("L0") >= 0 and txt.find("error(") >= 0)
    print(rdem)

    # 4. Surface code DEM / matching graph properties.
    for d in range(3, 6, 2):
        var cc = surface_code_memory(d, d, NoiseParams.uniform(0.001), "Z")
        var dem = detector_error_model(cc.circuit)
        var g = MatchingGraph(dem)
        assert_equal(g.graphlike_distance(), d, "circuit-level distance must equal d (hook-safe schedule)")
        assert_equal(g.num_dropped, 0)
        # every single fault is corrected by every decoder
        var uf = UnionFindDecoder(g.copy())
        var mw = MwpmDecoder(g.copy())
        var sb = SparseBlossomDecoder(g)
        for e in dem.errors:
            assert_equal(uf.decode(e.dets), e.obs, "UF failed on a single fault")
            assert_equal(mw.decode(e.dets), e.obs, "MWPM failed on a single fault")
            assert_equal(sb.decode(e.dets), e.obs, "sparse blossom failed on a single fault")
        if d == 3:
            var mwd = MwpmDecoder(g.copy(), use_apsp=False)  # Dijkstra path
            for e in dem.errors:
                assert_equal(mwd.decode(e.dets), e.obs, "MWPM (Dijkstra) failed on a single fault")
            var lt = LookupTableDecoder(dem, 2)
            for e in dem.errors:
                assert_equal(lt.decode(e.dets), e.obs, "lookup failed on a single fault")
        print("d=" + String(d), g, "single faults corrected by UF/MWPM/sparse blossom")
    var stim_txt = surface_code_memory(3, 3, NoiseParams.uniform(0.001), "Z").to_stim()
    assert_true(stim_txt.find("DETECTOR(") >= 0 and stim_txt.find("OBSERVABLE_INCLUDE(0)") >= 0 and stim_txt.find("DEPOLARIZE2(0.001)") >= 0)

    # 5. Exact matching: subset DP == blossom on random instances.
    var mr = Rng(9)
    for _ in range(200):
        var k = 2 + mr.below(10)
        var dist = List[Float64](length=k * k, fill=0.0)
        var bd = List[Float64](length=k, fill=0.0)
        for i in range(k):
            bd[i] = 1.0 + 9.0 * mr.uniform()
            for j in range(i + 1, k):
                var v = 0.5 + 9.0 * mr.uniform()
                dist[i * k + j] = v
                dist[j * k + i] = v
        var a = min_weight_boundary_matching(k, dist, bd, False)
        var b = min_weight_boundary_matching(k, dist, bd, True)
        var ca = 0.0
        var cb = 0.0
        for i in range(k):
            if a[i] < 0: ca += bd[i]
            elif a[i] > i: ca += dist[i * k + a[i]]
            if b[i] < 0: cb += bd[i]
            elif b[i] > i: cb += dist[i * k + b[i]]
        assert_true(abs(ca - cb) < 1e-2, "blossom and DP disagree: " + String(ca) + " vs " + String(cb))
    print("blossom == subset DP ok")

    # 6. Logical error rates at p = 0.001: small, and d=5 < d=3.
    var r3 = memory_experiment("surface_z", 3, -1, 0.001, 300000, "mwpm", 11, -1)
    var r5 = memory_experiment("surface_z", 5, -1, 0.001, 300000, "mwpm", 11, -1)
    print(r3)
    print(r5)
    assert_true(r3.logical_error_rate() < 3e-3, "d=3 logical error rate too high")
    assert_true(r5.logical_error_rate() < r3.logical_error_rate(), "d=5 must beat d=3 below threshold")
    var u3 = memory_experiment("surface_x", 3, -1, 0.001, 100000, "uf", 12, -1)
    print(u3)
    assert_true(u3.logical_error_rate() < 3e-3)
    var rr = memory_experiment("repetition", 5, -1, 0.01, 100000, "lookup", 4, -1)
    print(rr)
    # determinism: same seed -> same numbers
    var r3b = memory_experiment("surface_z", 3, -1, 0.001, 300000, "mwpm", 11, -1)
    assert_equal(r3b.errors, r3.errors)

    # 7. Noise model from calibration data keeps detectors deterministic.
    var nm = NoiseModel.uniform(17, 100.0, 80.0, 2e-4, 5e-3, 1e-2, 0.035, 0.3, 1.0)
    var noisy = apply_noise_model(surface_code_memory(3, 3, NoiseParams.noiseless(), "Z").circuit, nm)
    var ndem = detector_error_model(noisy)
    assert_true(ndem.num_errors() > 0)
    var ch = thermal_pauli_channel(10.0, 100.0, 80.0)
    assert_true(abs(ch.px - (1.0 - exp(-0.1)) / 4.0) < 1e-12)
    print(ndem, "from calibration noise model")

    # 8. Randomized benchmarking recovers the injected error per Clifford.
    var lengths: List[Int] = [1, 5, 10, 20, 40, 70, 100, 150]
    var e_inj = 0.01
    var rb = randomized_benchmarking_1q(lengths, 10, 4000, 21, e_inj)
    print(rb)
    var expected = 2.0 * e_inj / 3.0
    assert_true(abs(rb.error_per_clifford - expected) / expected < 0.15, "RB EPC " + String(rb.error_per_clifford) + " vs " + String(expected))
    assert_equal(len(clifford_group_1q()), 24)

    # 9. T1 / Ramsey recover T1 and T2.
    var cm = NoiseModel.uniform(1, 50.0, 30.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0)
    var delays: List[Float64] = [0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 80.0, 120.0, 200.0]
    var t1r = t1_experiment(cm, delays, 40000, 3)
    var t2r = ramsey_experiment(cm, delays, 40000, 4)
    print(t1r)
    print(t2r)
    assert_true(abs(t1r.time_constant() - 50.0) / 50.0 < 0.1, "T1 fit")
    assert_true(abs(t2r.time_constant() - 30.0) / 30.0 < 0.1, "T2 fit")

    # 10. Readout mitigation: noisy GHZ counts -> ideal 50/50.
    var rm = NoiseModel.uniform(3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rm.readout_p01[0] = 0.02
    rm.readout_p10[0] = 0.08
    rm.readout_p01[1] = 0.05
    rm.readout_p10[1] = 0.03
    rm.readout_p01[2] = 0.01
    rm.readout_p10[2] = 0.10
    var cals = readout_calibration_circuits(3)
    var c0 = _with_readout(cals[0], rm)
    var c1 = _with_readout(cals[1], rm)
    var k0 = sample_stabilizer(c0, 200000, 1)
    var k1 = sample_stabilizer(c1, 200000, 2)
    var cal = ReadoutCalibration.from_counts(3, k0.counts, k1.counts)
    print(cal)
    assert_true(abs(cal.p10[2] - 0.10) < 0.005)
    var ghz = Circuit(3, 3)
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(1, 2)
    for qq in range(3):
        ghz.measure(qq, qq)
    var raw = sample_stabilizer(_with_readout(ghz, rm), 200000, 3)
    var mit = mitigate_counts(raw.counts, cal)
    assert_true(abs(mit.get("000", 0.0) - 0.5) < 0.01 and abs(mit.get("111", 0.0) - 0.5) < 0.01, "mitigated GHZ")
    assert_true(abs(mit.get("010", 0.0)) < 0.01)
    var zz: List[Int] = [0, 1]
    assert_true(abs(expectation_z(mit, zz) - 1.0) < 0.02)

    print("test_qec OK")


def _with_readout(c: Circuit, m: NoiseModel) raises -> Circuit:
    """Asymmetric readout errors modelled exactly with classical post-flips."""
    var out = c.empty_like()
    for g in c.ops:
        if g.kind == Op.MEASURE:
            # measure, flip the (now classical) qubit with P(1|0) or P(0|1)
            # depending on the outcome, then re-measure into the same clbit.
            out.measure(g.q0, g.clbit)
            var q = g.q0
            out.x_error(m.readout_p01[q], q)
            out.ops[len(out.ops) - 1].cond = g.clbit
            out.ops[len(out.ops) - 1].cond_val = 0
            out.x_error(m.readout_p10[q], q)
            out.ops[len(out.ops) - 1].cond = g.clbit
            out.ops[len(out.ops) - 1].cond_val = 1
            out.measure(g.q0, g.clbit)
        else:
            out.append(g)
    return out^
