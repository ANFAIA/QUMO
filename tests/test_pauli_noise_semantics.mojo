"""Cross-simulator check of the Pauli-noise opcodes (qumo/ir/pauli_noise.mojo).

Each opcode acts on q0 (and q1) of two Bell pairs (q0, q2) and (q1, q3); a Bell
measurement of each pair reads the applied Pauli exactly, so every back end
yields a distribution over the 16 Pauli pairs. That distribution is compared:
  * exactly (1e-12) with the density matrix, where density_evolve takes the
    opcode, with the detector error model (independent mechanisms, XOR
    convolution) and with the channel definition itself;
  * statistically with every sampler: sample() trajectories, sample_stabilizer,
    FrameSimulator, sample_ptsbe and the FT emulator's state-vector path.
Samplers use fixed seeds and SHOTS shots; tolerance TOL per outcome
(about 7 standard deviations at p = 0.1).
Existing rejections are checked too: the density matrix refuses PAULI_CHANNEL_2
and E (QM-E414), PTSBE refuses E (QM-E414), the DEM refuses ELSE (QM-E733).
Prints `test_pauli_noise_semantics OK`."""

from std.testing import assert_true

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.runner import sample
from qumo.sim.stabilizer import sample_stabilizer
from qumo.sim.frame import FrameSimulator
from qumo.sim.ptsbe import sample_ptsbe
from qumo.sim.density import density_evolve
from qumo.qec.dem import detector_error_model
from qumo.ft.emulate import run_physical

comptime SHOTS = 20000
comptime TOL = 0.015
comptime EXACT = 1e-12


def _mask(l0: Int, l1: Int) -> Int:
    """Bell-measurement outcome of Pauli letters (l0 on q0, l1 on q1)."""
    var x0 = 1 if (l0 == 1 or l0 == 2) else 0
    var z0 = 1 if (l0 == 2 or l0 == 3) else 0
    var x1 = 1 if (l1 == 1 or l1 == 2) else 0
    var z1 = 1 if (l1 == 2 or l1 == 3) else 0
    return z0 | (x0 << 1) | (z1 << 2) | (x1 << 3)


struct Case(Copyable, Movable):
    var name: String
    var circuit: Circuit
    var expected: List[Float64]  # 16 outcome probabilities from the channel definition
    var density: Bool  # density_evolve takes it
    var ptsbe: Bool
    var dem: Float64  # DEM tolerance; < 0: the DEM must refuse it (QM-E733); 0: not checked here

    def __init__(out self, name: String, var circuit: Circuit, var expected: List[Float64],
                 density: Bool, ptsbe: Bool, dem: Float64):
        self.name = name
        self.circuit = circuit^
        self.expected = expected^
        self.density = density
        self.ptsbe = ptsbe
        self.dem = dem


def _prep() -> Circuit:
    var c = Circuit(4, 4, "pauli_noise")
    c.h(0); c.cx(0, 2)
    c.h(1); c.cx(1, 3)
    return c^


def _finish(mut c: Circuit):
    c.cx(0, 2); c.h(0)
    c.cx(1, 3); c.h(1)
    c.measure(0, 0); c.measure(2, 1); c.measure(1, 2); c.measure(3, 3)
    for m in range(4):
        c.detector([m])


def _dist1(px: Float64, py: Float64, pz: Float64) -> List[Float64]:
    var d = List[Float64](length=16, fill=0.0)
    d[_mask(1, 0)] += px
    d[_mask(2, 0)] += py
    d[_mask(3, 0)] += pz
    d[0] += 1.0 - px - py - pz
    return d^


def _cases() raises -> List[Case]:
    var out = List[Case]()
    var c = _prep(); c.x_error(0.2, 0); _finish(c)
    out.append(Case("X_ERROR", c^, _dist1(0.2, 0, 0), True, True, EXACT))
    c = _prep(); c.y_error(0.15, 0); _finish(c)
    out.append(Case("Y_ERROR", c^, _dist1(0, 0.15, 0), True, True, EXACT))
    c = _prep(); c.z_error(0.1, 0); _finish(c)
    out.append(Case("Z_ERROR", c^, _dist1(0, 0, 0.1), True, True, EXACT))
    c = _prep(); c.depolarize1(0.3, 0); _finish(c)
    out.append(Case("DEPOLARIZE1", c^, _dist1(0.1, 0.1, 0.1), True, True, EXACT))
    c = _prep(); c.pauli_channel_1(0.1, 0.05, 0.2, 0); _finish(c)
    out.append(Case("PAULI_CHANNEL_1", c^, _dist1(0.1, 0.05, 0.2), True, True, EXACT))

    c = _prep(); c.depolarize2(0.3, 0, 1); _finish(c)
    var d2 = List[Float64](length=16, fill=0.0)
    d2[0] = 0.7
    for r in range(1, 16):
        d2[_mask(r & 3, r >> 2)] += 0.02
    out.append(Case("DEPOLARIZE2", c^, d2^, True, True, EXACT))

    # distinct entries, so a swapped letter order cannot pass
    var probs = List[Float64]()
    var tot = 0.0
    for i in range(15):
        probs.append(0.005 * Float64(i + 1))
        tot += 0.005 * Float64(i + 1)
    c = _prep(); c.pauli_channel_2(probs, 0, 1); _finish(c)
    var dp = List[Float64](length=16, fill=0.0)
    dp[0] = 1.0 - tot
    for i in range(15):
        # Stim order IX, IY, IZ, XI, ...: first letter on the first qubit
        dp[_mask((i + 1) >> 2, (i + 1) & 3)] += probs[i]
    out.append(Case("PAULI_CHANNEL_2", c^, dp^, False, True, 0.0))
    # The DEM turns PAULI_CHANNEL_2's disjoint outcomes into independent
    # mechanisms with the same probabilities (first order, error O(p^2)), so
    # it is checked at small p, where a swapped letter order (>= 3e-5) still fails.
    probs = List[Float64]()
    tot = 0.0
    for i in range(15):
        probs.append(1e-5 * Float64(i + 1))
        tot += 1e-5 * Float64(i + 1)
    c = _prep(); c.pauli_channel_2(probs, 0, 1); _finish(c)
    dp = List[Float64](length=16, fill=0.0)
    dp[0] = 1.0 - tot
    for i in range(15):
        dp[_mask((i + 1) >> 2, (i + 1) & 3)] += probs[i]
    out.append(Case("PAULI_CHANNEL_2 (small p)", c^, dp^, False, True, tot * tot))

    c = _prep()
    c.correlated_error(0.2, [0, 1], [1, 3])
    _finish(c)
    var de = List[Float64](length=16, fill=0.0)
    de[_mask(1, 3)] = 0.2
    de[0] = 0.8
    out.append(Case("E", c^, de^, False, False, EXACT))

    c = _prep()
    c.correlated_error(0.2, [0, 1], [1, 3])
    c.correlated_error(0.25, [0], [2], else_branch=True)
    _finish(c)
    var dl = List[Float64](length=16, fill=0.0)
    dl[_mask(1, 3)] = 0.2
    dl[_mask(2, 0)] = 0.8 * 0.25
    dl[0] = 0.8 * 0.75
    out.append(Case("ELSE_CORRELATED_ERROR", c^, dl^, False, False, -1.0))
    return out^


def _err_code(msg: String) -> String:
    return String(msg[byte=0:7]) if msg.byte_length() >= 7 else msg


def _check(name: String, backend: String, got: List[Float64], want: List[Float64], tol: Float64) raises:
    var worst = 0.0
    for i in range(16):
        worst = max(worst, abs(got[i] - want[i]))
    if worst > tol:
        print(name, backend, "max |diff|", worst, "tol", tol)
        for i in range(16):
            print("  ", i, got[i], want[i])
    assert_true(worst <= tol, name + " on " + backend)


def _from_memory(memory: List[Int]) -> List[Float64]:
    var d = List[Float64](length=16, fill=0.0)
    for m in memory:
        d[m & 15] += 1.0
    for i in range(16):
        d[i] /= Float64(len(memory))
    return d^


def _density(c: Circuit) raises -> List[Float64]:
    var probs = density_evolve(c).probabilities()
    var d = List[Float64](length=16, fill=0.0)
    for i in range(len(probs)):
        var m = (i & 1) | (((i >> 2) & 1) << 1) | (((i >> 1) & 1) << 2) | (((i >> 3) & 1) << 3)
        d[m] += probs[i]
    return d^


def _dem(c: Circuit) raises -> List[Float64]:
    var dem = detector_error_model(c, decompose=False)
    var d = List[Float64](length=16, fill=0.0)
    d[0] = 1.0
    for e in dem.errors:
        var m = 0
        for det in e.dets:
            m ^= 1 << det
        var nd = List[Float64](length=16, fill=0.0)
        for i in range(16):
            nd[i] += d[i] * (1.0 - e.p)
            nd[i ^ m] += d[i] * e.p
        d = nd^
    return d^


def _frame(c: Circuit) raises -> List[Float64]:
    var ms = FrameSimulator(c).sample_measurements(SHOTS, seed=11)
    var d = List[Float64](length=16, fill=0.0)
    for s in range(SHOTS):
        var m = 0
        for j in range(4):
            if ms.get(s, j):
                m |= 1 << j
        d[m] += 1.0 / Float64(SHOTS)
    return d^


def _emulate(c: Circuit) raises -> List[Float64]:
    var lc = run_physical(c, [0, 1, 2, 3], List[Int](), SHOTS, 13, method="statevector")
    var d = List[Float64](length=16, fill=0.0)
    for kv in lc.counts.items():
        var key = kv.key
        var m = 0
        for j in range(4):
            if key[byte=3 - j:4 - j] == "1":
                m |= 1 << j
        d[m] += Float64(kv.value) / Float64(lc.shots)
    return d^


def test_cases() raises:
    var cases = _cases()
    for ref cs in cases:
        var want = cs.expected.copy()
        var s = 0.0
        for v in want:
            s += v
        assert_true(abs(s - 1.0) < EXACT, cs.name + " expected sums to 1")
        var c = cs.circuit.copy()
        if cs.density:
            _check(cs.name, "density_evolve", _density(c), want, EXACT)
        else:
            var raised = False
            try:
                _ = density_evolve(c)
            except e:
                raised = _err_code(String(e)) == "QM-E414"
            assert_true(raised, cs.name + ": density_evolve keeps QM-E414")
        if cs.dem > 0.0:
            _check(cs.name, "detector_error_model", _dem(c), want, cs.dem)
        elif cs.dem < 0.0:
            var raised = False
            try:
                _ = detector_error_model(c, decompose=False)
            except e:
                raised = _err_code(String(e)) == "QM-E733"
            assert_true(raised, cs.name + ": DEM keeps QM-E733")
        _check(cs.name, "sample", _from_memory(sample(c, SHOTS, seed=5).memory), want, TOL)
        _check(cs.name, "sample_stabilizer", _from_memory(sample_stabilizer(c, SHOTS, seed=7).memory), want, TOL)
        _check(cs.name, "FrameSimulator", _frame(c), want, TOL)
        _check(cs.name, "run_physical(statevector)", _emulate(c), want, TOL)
        if cs.ptsbe:
            _check(cs.name, "sample_ptsbe", _from_memory(sample_ptsbe(c, SHOTS, seed=9).memory), want, TOL)
        else:
            var raised = False
            try:
                _ = sample_ptsbe(c, SHOTS, seed=9)
            except e:
                raised = _err_code(String(e)) == "QM-E414"
            assert_true(raised, cs.name + ": sample_ptsbe keeps QM-E414")


def main() raises:
    test_cases()
    print("test_pauli_noise_semantics OK")
