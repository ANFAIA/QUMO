from std.math import pi, sqrt, exp
from std.collections import Dict
from std.bit import pop_count
from std.testing import assert_equal, assert_true

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.rng import Rng
from qumo.sim.stabilizer import TableauSimulator, sample_stabilizer
from qumo.sim.frame import FrameSimulator
from qumo.qec.codes import NoiseParams, surface_code_memory, repetition_code_memory, generate_code, to_stim
from qumo.qec.dem import DetectorErrorModel, DemError, detector_error_model, depolarize1_independent, combine_independent
from qumo.qec.stim_text import parse_dem
from qumo.qec.decoders import (
    Decoder, MatchingGraph, UnionFindDecoder, MwpmDecoder, LookupTableDecoder, min_weight_boundary_matching,
    count_logical_errors, UF_SCALE,
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

    test_uf_buffer_reuse()
    test_noise_model_arity3()
    test_graphlike_distance()
    print("test_qec OK")


def _count_op(c: Circuit, k: Int) -> Int:
    var n = 0
    for g in c.ops:
        if g.kind == k:
            n += 1
    return n


def test_noise_model_arity3() raises:
    """Noise model and 3-qubit gates. apply_noise_model has no gate-error rule
    for them, so it must reject them (QM-E739) whenever the model has gate
    error, and keep them (idle channels only) when it has none."""
    var ccx = Circuit(3, 0, "ccx")
    ccx.ccx(0, 1, 2)
    var cswap = Circuit(3, 0, "cswap")
    cswap.cswap(0, 1, 2)
    var ccz = Circuit(3, 0, "ccz")
    ccz.ccz(0, 1, 2)

    # 2q gate error set: every 3-qubit gate is rejected with a hint.
    var nm2 = NoiseModel.uniform(3, 100.0, 80.0, 0.0, 1e-2, 0.0, 0.0, 0.3, 0.0)
    var circuits: List[Circuit] = [ccx.copy(), cswap.copy(), ccz.copy()]
    for c in circuits:
        var raised = False
        try:
            _ = apply_noise_model(c, nm2)
        except e:
            raised = True
            var m = String(e)
            assert_true(m.startswith("QM-E739"), "error code QM-E739: " + m)
            assert_true(m.find("Hint:") >= 0, "error has hint: " + m)
        assert_true(raised, "3-qubit gate with 2q gate error must raise QM-E739")

    # 1q gate error alone also counts as gate error.
    var nm1 = NoiseModel.uniform(3, 100.0, 80.0, 1e-3, 0.0, 0.0, 0.0, 0.3, 0.0)
    var raised1 = False
    try:
        _ = apply_noise_model(ccx, nm1)
    except e:
        raised1 = True
        assert_true(String(e).startswith("QM-E739"))
    assert_true(raised1, "3-qubit gate with 1q gate error must raise QM-E739")

    # A per-edge 2q error on any edge also counts.
    var nme = NoiseModel(3)
    nme.edge_error_2q[0 * 3 + 2] = 1e-2
    var raisede = False
    try:
        _ = apply_noise_model(ccx, nme)
    except e:
        raisede = True
    assert_true(raisede, "per-edge gate error must raise QM-E739 for a 3-qubit gate")

    # Zero gate error everywhere: accepted, gate kept, no gate error inserted.
    var zero = NoiseModel.uniform(3, 100.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0)
    var out = apply_noise_model(ccx, zero)
    assert_equal(_count_op(out, Op.CCX), 1)
    assert_equal(_count_op(out, Op.DEPOLARIZE1) + _count_op(out, Op.DEPOLARIZE2), 0)
    var out0 = apply_noise_model(ccx, NoiseModel(3))
    assert_equal(_count_op(out0, Op.CCX), 1)

    # A virtual 3-qubit gate (duration 0 through op_time) gets no gate error,
    # so it is accepted even with gate error on.
    var virt = NoiseModel.uniform(3, 100.0, 80.0, 0.0, 1e-2, 0.0, 0.0, 0.3, 0.0)
    virt.op_time = List[Float64](length=100, fill=1.0)
    virt.op_time[Op.CCX] = 0.0
    var outv = apply_noise_model(ccx, virt)
    assert_equal(_count_op(outv, Op.CCX), 1)

    # 1q and 2q gates are unchanged: a cx gets DEPOLARIZE2 after it, an h gets
    # DEPOLARIZE1 when the 1q error is set.
    var cx = Circuit(2, 0, "cx")
    cx.cx(0, 1)
    var outcx = apply_noise_model(cx, nm2)
    assert_equal(_count_op(outcx, Op.DEPOLARIZE2), 1)
    var h = Circuit(1, 0, "h")
    h.h(0)
    var nm_h = NoiseModel.uniform(1, 100.0, 80.0, 1e-2, 0.0, 0.0, 0.0, 0.3, 0.0)
    var outh = apply_noise_model(h, nm_h)
    assert_equal(_count_op(outh, Op.DEPOLARIZE1), 1)
    var outcx0 = apply_noise_model(cx, zero)
    assert_equal(_count_op(outcx0, Op.DEPOLARIZE2), 0)


# ---------------------------------------------------------------------------
# Union-find: allocation-free decoder == allocate-per-shot reference
# ---------------------------------------------------------------------------
struct RefUnionFind(Decoder):
    """Oracle: verbatim copy of `UnionFindDecoder` as of QUMO 1750a81, before the
    per-shot buffers were reused (it allocates its work lists on every shot and
    resets touched state eagerly). The optimised decoder must predict exactly
    the same observable masks."""

    var g: MatchingGraph
    var elen: List[Int]
    var parent: List[Int]
    var csize: List[Int]
    var odd: List[UInt8]
    var grounded: List[UInt8]
    var nxt: List[Int]
    var defect: List[UInt8]
    var growth: List[Int]
    var grown: List[UInt8]
    var rate: List[Int]
    var etouched: List[UInt8]
    var touched_v: List[Int]
    var touched_e: List[Int]
    var lid: List[Int]
    var seen: List[Int]
    var stamp: Int
    var failures: Int

    def __init__(out self, var graph: MatchingGraph):
        var nn = graph.num_nodes()
        var ne = graph.num_edges()
        self.elen = List[Int](capacity=ne)
        for i in range(ne):
            self.elen.append(max(1, Int(graph.ew[i] * UF_SCALE + 0.5)))
        self.g = graph^
        self.parent = List[Int](length=nn, fill=-1)
        self.csize = List[Int](length=nn, fill=0)
        self.odd = List[UInt8](length=nn, fill=0)
        self.grounded = List[UInt8](length=nn, fill=0)
        self.nxt = List[Int](length=nn, fill=-1)
        self.defect = List[UInt8](length=nn, fill=0)
        self.growth = List[Int](length=max(1, ne), fill=0)
        self.grown = List[UInt8](length=max(1, ne), fill=0)
        self.rate = List[Int](length=max(1, ne), fill=0)
        self.etouched = List[UInt8](length=max(1, ne), fill=0)
        self.touched_v = List[Int]()
        self.touched_e = List[Int]()
        self.lid = List[Int](length=nn, fill=-1)
        self.seen = List[Int](length=nn, fill=0)
        self.stamp = 0
        self.failures = 0

    def name(self) -> String:
        return "union_find"

    def _make(mut self, v: Int):
        if self.parent[v] < 0:
            self.parent[v] = v
            self.csize[v] = 1
            self.odd[v] = 0
            self.grounded[v] = 1 if v == self.g.boundary else 0
            self.nxt[v] = v
            self.touched_v.append(v)

    def _find(mut self, v: Int) -> Int:
        var r = v
        while self.parent[r] != r:
            r = self.parent[r]
        var c = v
        while self.parent[c] != r:
            var nx = self.parent[c]
            self.parent[c] = r
            c = nx
        return r

    def _union(mut self, a: Int, b: Int):
        var ra = self._find(a)
        var rb = self._find(b)
        if ra == rb:
            return
        if self.csize[ra] < self.csize[rb]:
            var t = ra
            ra = rb
            rb = t
        self.parent[rb] = ra
        self.csize[ra] += self.csize[rb]
        self.odd[ra] ^= self.odd[rb]
        self.grounded[ra] |= self.grounded[rb]
        var t2 = self.nxt[ra]
        self.nxt[ra] = self.nxt[rb]
        self.nxt[rb] = t2

    def _touch_edge(mut self, e: Int):
        if self.etouched[e] == 0:
            self.etouched[e] = 1
            self.touched_e.append(e)

    def _cleanup(mut self):
        for v in self.touched_v:
            self.parent[v] = -1
            self.defect[v] = 0
            self.lid[v] = -1
        for e in self.touched_e:
            self.growth[e] = 0
            self.grown[e] = 0
            self.rate[e] = 0
            self.etouched[e] = 0
        self.touched_v.clear()
        self.touched_e.clear()

    def decode(mut self, fired: List[Int]) -> UInt64:
        if len(fired) == 0:
            return 0
        for d in fired:
            self._make(d)
            self.defect[d] = 1
            self.odd[d] = 1
        var cand = fired.copy()
        var roots = List[Int]()
        var bedges = List[Int]()
        var newly = List[Int]()
        while True:
            # active clusters: odd parity and not touching the boundary
            self.stamp += 1
            roots.clear()
            for v in cand:
                var r = self._find(v)
                if self.seen[r] != self.stamp:
                    self.seen[r] = self.stamp
                    if self.odd[r] != 0 and self.grounded[r] == 0:
                        roots.append(r)
            if len(roots) == 0:
                break
            bedges.clear()
            for r in roots:
                var v = r
                while True:
                    for a in range(self.g.adj_off[v], self.g.adj_off[v + 1]):
                        var e = self.g.adj_edge[a]
                        if self.grown[e] == 0:
                            if self.rate[e] == 0:
                                bedges.append(e)
                            self.rate[e] += 1
                    v = self.nxt[v]
                    if v == r:
                        break
            if len(bedges) == 0:
                self.failures += 1  # isolated odd cluster: cannot be neutralised
                break
            var delta = -1
            for e in bedges:
                var rem = self.elen[e] - self.growth[e]
                var steps = (rem + self.rate[e] - 1) // self.rate[e]
                if delta < 0 or steps < delta:
                    delta = steps
            newly.clear()
            for e in bedges:
                self._touch_edge(e)
                self.growth[e] += delta * self.rate[e]
                self.rate[e] = 0
                if self.growth[e] >= self.elen[e]:
                    self.grown[e] = 1
                    newly.append(e)
            for e in newly:
                var u = self.g.eu[e]
                var w = self.g.ev[e]
                self._make(u)
                self._make(w)
                self._union(u, w)
            cand = roots.copy()
        var obs = self._peel()
        self._cleanup()
        return obs

    def _peel(mut self) -> UInt64:
        """Spanning forest of the grown subgraph, then leaf-to-root peeling."""
        var nv = len(self.touched_v)
        for i in range(nv):
            self.lid[self.touched_v[i]] = i
        var deg = List[Int](length=nv + 1, fill=0)
        var gl = List[Int]()
        for e in self.touched_e:
            if self.grown[e] != 0:
                gl.append(e)
                deg[self.lid[self.g.eu[e]]] += 1
                deg[self.lid[self.g.ev[e]]] += 1
        var off = List[Int](length=nv + 1, fill=0)
        for i in range(nv):
            off[i + 1] = off[i] + deg[i]
        var fillp = off.copy()
        var ladj = List[Int](length=max(1, off[nv]), fill=0)
        for e in gl:
            var a = self.lid[self.g.eu[e]]
            var b = self.lid[self.g.ev[e]]
            ladj[fillp[a]] = e
            fillp[a] += 1
            ladj[fillp[b]] = e
            fillp[b] += 1
        var pe = List[Int](length=nv, fill=-2)  # -2 unvisited, -1 root
        var order = List[Int](capacity=nv)
        var bl = self.lid[self.g.boundary] if self.parent[self.g.boundary] >= 0 else -1
        var starts = List[Int]()
        if bl >= 0:
            starts.append(bl)
        for i in range(nv):
            starts.append(i)
        for s in starts:
            if pe[s] != -2:
                continue
            pe[s] = -1
            var head = len(order)
            order.append(s)
            while head < len(order):
                var x = order[head]
                head += 1
                var vx = self.touched_v[x]
                for a in range(off[x], off[x + 1]):
                    var e = ladj[a]
                    var y = self.lid[self.g.other(e, vx)]
                    if pe[y] == -2:
                        pe[y] = e
                        order.append(y)
        var obs: UInt64 = 0
        var i = len(order) - 1
        while i >= 0:
            var x = order[i]
            i -= 1
            var vx = self.touched_v[x]
            if self.defect[vx] != 0 and pe[x] >= 0:
                var e = pe[x]
                obs ^= self.g.eobs[e]
                self.defect[vx] = 0
                var u = self.g.other(e, vx)
                self.defect[u] ^= 1
        return obs


def test_uf_buffer_reuse() raises:
    """Thousands of syndromes (surface code d = 3, 5, 7): the buffer-reusing
    UnionFindDecoder predicts exactly what the allocate-per-shot reference
    does, shot by shot, and the result does not depend on the order in which
    one decoder instance sees the shots (no state leaks between shots)."""
    var ds: List[Int] = [3, 5, 7]
    var total = 0
    for di in range(len(ds)):
        var d = ds[di]
        var p = 0.006 if d < 7 else 0.004
        var shots = 2000 if d < 7 else 1000
        var cc = generate_code("surface_z", d, d, NoiseParams.uniform(p))
        var g = MatchingGraph(detector_error_model(cc.circuit))
        var dsm = FrameSimulator(cc.circuit).sample_detectors(shots, 40 + UInt64(d))
        var syn = List[List[Int]]()
        for s in range(shots):
            syn.append(dsm.fired(s))
        var uf = UnionFindDecoder(g.copy())
        var oracle = RefUnionFind(g.copy())
        var fwd = List[UInt64](length=shots, fill=0)
        var nonzero = 0
        for s in range(shots):
            fwd[s] = uf.decode(syn[s])
            assert_equal(fwd[s], oracle.decode(syn[s]), "UF differs from the reference at d=" + String(d) + " shot " + String(s))
            if fwd[s] != 0:
                nonzero += 1
        assert_equal(uf.failures, oracle.failures)
        var back = UnionFindDecoder(g.copy())
        var s2 = shots - 1
        while s2 >= 0:
            assert_equal(back.decode(syn[s2]), fwd[s2], "UF result depends on shot order (d=" + String(d) + ")")
            s2 -= 1
        assert_true(nonzero > 0, "test needs syndromes that flip the observable")
        total += shots
        print("UF buffer reuse d=" + String(d), "shots=" + String(shots), "nonzero predictions=" + String(nonzero), "identical to the reference")
    assert_true(total >= 5000)


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


def _min_logical(dm: List[UInt64], om: List[UInt64]) -> Int:
    """Brute force: fewest columns whose detector masks XOR to 0 and observable
    masks XOR to non-zero; -1 if none. Plain subset enumeration (<= 20 columns)."""
    var n = len(dm)
    var best = -1
    for s in range(1, 1 << n):
        var w = Int(pop_count(UInt64(s)))
        if best >= 0 and w >= best:
            continue
        var d: UInt64 = 0
        var o: UInt64 = 0
        for j in range(n):
            if ((s >> j) & 1) == 1:
                d ^= dm[j]
                o ^= om[j]
        if d == 0 and o != 0:
            best = w
    return best


def _dem_brute(dem: DetectorErrorModel) -> Int:
    """Reference on the DEM itself: mechanisms with their full symptom."""
    var dm = List[UInt64]()
    var om = List[UInt64]()
    for e in dem.errors:
        var m: UInt64 = 0
        for d in e.dets:
            m ^= UInt64(1) << UInt64(d)
        dm.append(m)
        om.append(e.obs)
    return _min_logical(dm, om)


def _graph_brute(g: MatchingGraph) -> Int:
    """Reference on the graph's own edge list (the boundary flips no detector)."""
    var dm = List[UInt64]()
    var om = List[UInt64]()
    for i in range(g.num_edges()):
        var m = UInt64(1) << UInt64(g.eu[i])
        if g.ev[i] != g.boundary:
            m ^= UInt64(1) << UInt64(g.ev[i])
        dm.append(m)
        om.append(g.eobs[i])
    return _min_logical(dm, om)


def _random_obs(mut rng: Rng, nobs: Int) -> UInt64:
    var o: UInt64 = 0
    for k in range(nobs):
        if rng.below(10) < 3:
            o |= UInt64(1) << UInt64(k)
    return o


def test_graphlike_distance() raises:
    """MatchingGraph.graphlike_distance is the exact graph-like distance: every
    start node (not only the boundary) and every observable (not only L0),
    checked against brute force over the DEM and over the graph's edges."""
    # Odd cycle that avoids the boundary: 3 (the boundary-only BFS gave 5).
    var a = MatchingGraph(parse_dem("error(0.01) D0\nerror(0.01) D0 D1 L0\nerror(0.01) D1 D2\nerror(0.01) D0 D2\n"))
    assert_equal(a.graphlike_distance(), 3, "bulk odd cycle")
    # Only L1 can flip: 3 (observable 0 only gave -1).
    var b = MatchingGraph(parse_dem("error(0.01) D0 L1\nerror(0.01) D0 D1\nerror(0.01) D1\n"))
    assert_equal(b.graphlike_distance(), 3, "logical on L1")
    # L0 needs 4 edges, L1 needs 3: 3 (gave 4).
    var b2 = MatchingGraph(parse_dem("error(0.01) D0 L1\nerror(0.01) D0 D1\nerror(0.01) D1\nerror(0.01) D2 L0\nerror(0.01) D2 D3\nerror(0.01) D3 D4\nerror(0.01) D4\n"))
    assert_equal(b2.graphlike_distance(), 3, "shorter logical on L1")
    # Counted in decomposition pieces, not mechanisms (definitional, documented):
    # the graph has a 3-edge logical, the DEM needs 4 mechanisms.
    var c1 = parse_dem("error(0.01) D0 D1 L0 ^ D2 D3\nerror(0.01) D2 D3\nerror(0.01) D0\nerror(0.01) D1\n")
    assert_equal(MatchingGraph(c1).graphlike_distance(), 3)
    assert_equal(_dem_brute(c1), 4)
    # A dropped hyperedge hides a 2-mechanism logical from the graph.
    var c2 = parse_dem("error(0.01) D0 L0 ^ D1 D2\nerror(0.01) D0 D1 D2\n")
    var gc2 = MatchingGraph(c2)
    assert_equal(gc2.num_dropped, 1)
    assert_equal(gc2.graphlike_distance(), -1)
    assert_equal(_dem_brute(c2), 2)

    var rng = Rng(2611)
    # Graph-like models on distinct detector pairs: graph == DEM, compare with the DEM brute force.
    var nontrivial = 0
    for trial in range(400):
        var nd = 2 + rng.below(6)
        var nobs = 1 + rng.below(3)
        var with_boundary = rng.below(3) != 0
        var dem = DetectorErrorModel(nd, nobs)
        for u in range(nd):
            for v in range(u, nd + 1):
                if v == u or (v == nd and not with_boundary):
                    continue
                if rng.below(10) >= 4 or len(dem.errors) >= 16:
                    continue
                var dets = List[Int]()
                dets.append(u)
                if v < nd:
                    dets.append(v)
                dem.errors.append(DemError(0.01, dets^, _random_obs(rng, nobs)))
        var g = MatchingGraph(dem)
        var want = _dem_brute(dem)
        if want > 0:
            nontrivial += 1
        assert_equal(g.graphlike_distance(), want, "graph-like random model " + String(trial))
        assert_equal(_graph_brute(g), want)
    print("graphlike_distance: random graph-like models with a logical:", nontrivial, "of 400")
    assert_true(nontrivial > 100, "random models must often contain a logical")
    # Models with hyperedges, merged and conflicting parallel errors: compare with
    # brute force over the graph's edges (the quantity the function defines).
    for trial in range(300):
        var nd = 3 + rng.below(4)
        var nobs = 1 + rng.below(2)
        var dem = DetectorErrorModel(nd, nobs)
        var ne = 4 + rng.below(9)
        for _ in range(ne):
            var k = 1 + rng.below(4)
            var dets = List[Int]()
            for d in range(nd):
                if len(dets) < k and rng.below(nd) < k:
                    dets.append(d)
            if len(dets) == 0:
                dets.append(rng.below(nd))
            dem.errors.append(DemError(0.01, dets^, _random_obs(rng, nobs)))
        var g = MatchingGraph(dem)
        assert_true(g.num_edges() <= 20)
        assert_equal(g.graphlike_distance(), _graph_brute(g), "hypergraph random model " + String(trial))
    # Code values are unchanged (the boundary start already found them).
    for d in range(3, 8, 2):
        for basis in ["Z", "X"]:
            var cc = surface_code_memory(d, d, NoiseParams.uniform(0.001), basis)
            assert_equal(MatchingGraph(detector_error_model(cc.circuit)).graphlike_distance(), d)
        var rc = repetition_code_memory(d, d, NoiseParams.uniform(0.001))
        assert_equal(MatchingGraph(detector_error_model(rc.circuit)).graphlike_distance(), d)
    print("graphlike_distance ok")
