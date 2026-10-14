"""Noise simulation with Pauli channels: bit flip, phase flip, depolarizing (Amazon Braket example).

Reproduces: Amazon Braket - Simulating noise on Amazon Braket
Source: https://github.com/amazon-braket/amazon-braket-examples/tree/main/examples/braket_features (Simulating_Noise_On_Amazon_Braket.ipynb; Apache-2.0)

The notebook adds noise channels to circuits "bottom-up" or with
`apply_gate_noise` (a channel after every gate, on each qubit the gate
touches) and runs them on the density-matrix simulator. Its quick start is a
Bell circuit H(0), CNOT(0, 1) with BitFlip(0.1) on every gate: 1,000 shots gave
00: 424, 11: 408, 10: 87, 01: 81 (page).

QUMO has the same channels as IR instructions (`x_error`, `z_error`,
`depolarize1`, `depolarize2`, `pauli_channel_1`), plus calibration-driven
noise through `NoiseModel` + `apply_noise_model`
(qumo/qec/characterization.mojo). Noisy circuits are sampled with state-vector
trajectories (one Pauli draw per channel per shot). The exact reference here
plays the role of Braket's density-matrix simulator: every channel is a
Pauli mixture, each Pauli is propagated through the remaining Clifford gates
(H, CX), and the final distribution is the exact convolution of all channels.

Instances: the page's Bell example, then GHZ-3 and GHZ-5 with
`apply_gate_noise`-style BitFlip / PhaseFlip / Depolarizing (p = 0.05) and with
a `NoiseModel` (e1 = 5e-3, e2 = 2e-2 -> DEPOLARIZE1(7.5e-3), DEPOLARIZE2(2.5e-2)).
Bit flips change the Z-basis counts but not the GHZ coherence <X...X>; phase
flips do the opposite; depolarizing noise does both.

Not supported (stated, not emulated): amplitude damping, generalized
amplitude damping, phase damping as non-Pauli channels and general Kraus
operators. QUMO's trajectories are Pauli-only; `NoiseModel` represents T1/T2
relaxation by its Pauli twirl (`thermal_pauli_channel`).

Checks:
- Bell + BitFlip(0.1): exact P = 0.41 / 0.09 / 0.09 / 0.41 (analytic, 1e-12), the
  page counts lie within 4 sigma of it, and 50k QUMO shots have TVD < 0.01;
- GHZ-3/5 x 4 noise settings: Z-basis TVD(50k trajectory shots, exact) < 0.01;
- GHZ-3/5 x 4 noise settings: sampled <X^n> within 4 sigma of the exact decay,
  e.g. (1 - 2p)^(2n-1) for phase flips (analytic);
- zero noise reproduces the ideal distribution exactly (1/2, 1/2, nothing else).

Run: .venv/bin/mojo run -I . examples/frameworks/braket/noise_simulation.mojo
example-meta: {"framework": "braket", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import sqrt
from std.bit import pop_count
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_is_unitary, op_is_noise, op_arity
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import sample
from qumo.qec.characterization import NoiseModel, apply_noise_model
from qumo.algorithms.library import ghz
from qumo.algorithms.mitigation import sampled_expectation
from qumo.ai.checks import CheckLog

comptime SHOTS = 50000


def with_gate_noise(c: Circuit, kind: String, p: Float64) raises -> Circuit:
    """Braket `apply_gate_noise`: the channel after every gate on each qubit it touches."""
    var out = c.empty_like()
    for g in c.ops:
        out.append(g)
        if not op_is_unitary(g.kind):
            continue
        for j in range(op_arity(g.kind)):
            var q = g.qubit(j)
            if kind == "bit_flip":
                out.x_error(p, q)
            elif kind == "phase_flip":
                out.z_error(p, q)
            elif kind == "depolarizing":
                out.depolarize1(p, q)
            else:
                raise Error("unknown channel " + kind)
    return out^


struct Alt(ImplicitlyCopyable):
    var p: Float64
    var x: Int
    var z: Int

    def __init__(out self, p: Float64, x: Int, z: Int):
        self.p = p
        self.x = x
        self.z = z


def pauli_alts(kind: Int, q0: Int, q1: Int, p0: Float64, p1: Float64, p2: Float64) -> List[Alt]:
    """Non-identity Pauli components (prob, x mask, z mask) of a channel."""
    var out = List[Alt]()
    var a = 1 << q0
    if kind == Op.X_ERROR:
        out.append(Alt(p0, a, 0))
    elif kind == Op.Z_ERROR:
        out.append(Alt(p0, 0, a))
    elif kind == Op.Y_ERROR:
        out.append(Alt(p0, a, a))
    elif kind == Op.DEPOLARIZE1:
        out.append(Alt(p0 / 3.0, a, 0))
        out.append(Alt(p0 / 3.0, a, a))
        out.append(Alt(p0 / 3.0, 0, a))
    elif kind == Op.PAULI_CHANNEL_1:
        out.append(Alt(p0, a, 0))
        out.append(Alt(p1, a, a))
        out.append(Alt(p2, 0, a))
    elif kind == Op.DEPOLARIZE2:
        var b = 1 << q1
        for r in range(1, 16):
            var pa = r & 3  # 0 I, 1 X, 2 Y, 3 Z (Stim order)
            var pb = r >> 2
            var x = 0
            var z = 0
            if pa == 1 or pa == 2:
                x |= a
            if pa == 2 or pa == 3:
                z |= a
            if pb == 1 or pb == 2:
                x |= b
            if pb == 2 or pb == 3:
                z |= b
            out.append(Alt(p0 / 15.0, x, z))
    return out^


def propagate(c: Circuit, start: Int, mut x: Int, mut z: Int) raises:
    """Conjugate the Pauli (x, z) through the Clifford gates c.ops[start:]."""
    for i in range(start, len(c.ops)):
        var g = c.ops[i]
        var k = g.kind
        if not op_is_unitary(k):
            continue
        if k == Op.H:
            var bx = (x >> g.q0) & 1
            var bz = (z >> g.q0) & 1
            x = (x & ~(1 << g.q0)) | (bz << g.q0)
            z = (z & ~(1 << g.q0)) | (bx << g.q0)
        elif k == Op.CX:
            if ((x >> g.q0) & 1) == 1:
                x ^= 1 << g.q1
            if ((z >> g.q1) & 1) == 1:
                z ^= 1 << g.q0
        elif k == Op.X or k == Op.Y or k == Op.Z or k == Op.I:
            pass
        else:
            raise Error("exact reference supports H/CX/Pauli circuits only")


struct Exact(Movable):
    var dist: List[Float64]  # final X-flip mask distribution
    var x_parity: Float64  # <X^n> on GHZ (ideal +1)

    def __init__(out self, var dist: List[Float64], x_parity: Float64):
        self.dist = dist^
        self.x_parity = x_parity


def exact_noise(c: Circuit, n: Int) raises -> Exact:
    var dist = List[Float64](length=1 << n, fill=0.0)
    dist[0] = 1.0
    var xpar = 1.0
    for i in range(len(c.ops)):
        var g = c.ops[i]
        if not op_is_noise(g.kind):
            continue
        var alts = pauli_alts(g.kind, g.q0, g.q1, g.p0.value(), g.p1.value(), g.p2.value())
        var p_id = 1.0
        var flip = 0.0
        var nd = List[Float64](length=1 << n, fill=0.0)
        for a in alts:
            var x = a.x
            var z = a.z
            propagate(c, i + 1, x, z)
            p_id -= a.p
            for m in range(1 << n):
                nd[m ^ x] += dist[m] * a.p
            if (Int(pop_count(z)) & 1) == 1:
                flip += a.p
        for m in range(1 << n):
            nd[m] += dist[m] * p_id
        dist = nd^
        xpar *= 1.0 - 2.0 * flip
    return Exact(dist^, xpar)


def ghz_probabilities(e: Exact, n: Int) -> List[Float64]:
    """GHZ measures m or ~m (1/2 each) for a final flip mask m."""
    var full = (1 << n) - 1
    var out = List[Float64](length=1 << n, fill=0.0)
    for m in range(1 << n):
        out[m] += 0.5 * e.dist[m]
        out[m ^ full] += 0.5 * e.dist[m]
    return out^


def sampled_tvd(noisy: Circuit, exact: List[Float64], n: Int, seed: UInt64) raises -> Float64:
    var m = noisy.copy()
    for q in range(n):
        m.measure(q, q)
    var r = sample(m, SHOTS, seed)
    var counts = List[Int](length=1 << n, fill=0)
    for o in r.memory:
        counts[o] += 1
    var tvd = 0.0
    for b in range(1 << n):
        tvd += abs(Float64(counts[b]) / Float64(SHOTS) - exact[b])
    return 0.5 * tvd


def main() raises:
    var log = CheckLog("braket/noise_simulation")

    # --- the page's quick start: Bell + BitFlip(0.1) on every gate ---
    var bell = Circuit(2, name="bell")
    bell.h(0)
    bell.cx(0, 1)
    var nb = with_gate_noise(bell, "bit_flip", 0.1)
    var eb = ghz_probabilities(exact_noise(nb, 2), 2)
    var want: List[Float64] = [0.41, 0.09, 0.09, 0.41]  # 00, 01, 10, 11 (analytic)
    var bell_err = 0.0
    for b in range(4):
        bell_err = max(bell_err, abs(eb[b] - want[b]))
    print("Bell + BitFlip(0.1): exact P(00, 01, 10, 11) =", eb[0], eb[1], eb[2], eb[3])
    _ = log.close("Bell bit-flip exact probabilities (analytic)", bell_err, 0.0, 1e-12)
    var page: List[Int] = [424, 81, 87, 408]  # 00, 01, 10, 11 (page, 1000 shots)
    var page_z = 0.0
    for b in range(4):
        page_z = max(page_z, abs(Float64(page[b]) / 1000.0 - eb[b]) / sqrt(eb[b] * (1.0 - eb[b]) / 1000.0))
    _ = log.within("page counts within 4 sigma of exact (max z)", page_z, 0.0, 4.0)
    var tb = sampled_tvd(nb, eb, 2, 11)
    print("  QUMO trajectories,", SHOTS, "shots: TVD to exact =", tb)
    _ = log.within("Bell bit-flip TVD at 50k shots", tb, 0.0, 0.01)

    # --- GHZ-3 / GHZ-5 with Braket-style gate noise and with a NoiseModel ---
    var max_tvd = 0.0
    var max_z = 0.0
    var analytic_err = 0.0
    var kinds: List[String] = ["bit_flip", "phase_flip", "depolarizing", "noise_model"]
    var p = 0.05
    var sizes: List[Int] = [3, 5]
    print("  n  noise          TVD(50k)   <X^n> exact   <X^n> sampled")
    for n in sizes:
        var c = ghz(n)
        var xx = PauliSum()
        xx.add_label(1.0, String("X") * n)
        for ki in range(len(kinds)):
            var noisy: Circuit
            if kinds[ki] == "noise_model":
                var nm = NoiseModel(n)
                for q in range(n):
                    nm.gate_error_1q[q] = 5e-3
                nm.gate_error_2q = 2e-2
                noisy = apply_noise_model(c, nm)
            else:
                noisy = with_gate_noise(c, kinds[ki], p)
            var ex = exact_noise(noisy, n)
            var probs = ghz_probabilities(ex, n)
            var tvd = sampled_tvd(noisy, probs, n, UInt64(100 * n + ki))
            var est = sampled_expectation(noisy, xx, SHOTS, UInt64(200 * n + ki))
            var z = abs(est[0] - ex.x_parity) / max(est[1], 1e-12)
            max_tvd = max(max_tvd, tvd)
            max_z = max(max_z, z)
            # closed forms for the single-channel cases (2n - 1 noise locations)
            var loc = Float64(2 * n - 1)
            if kinds[ki] == "bit_flip":
                analytic_err = max(analytic_err, abs(ex.x_parity - 1.0))
            elif kinds[ki] == "phase_flip":
                analytic_err = max(analytic_err, abs(ex.x_parity - (1.0 - 2.0 * p) ** loc))
            elif kinds[ki] == "depolarizing":
                analytic_err = max(analytic_err, abs(ex.x_parity - (1.0 - 4.0 * p / 3.0) ** loc))
            print("  ", n, " ", kinds[ki], "   ", tvd, "  ", ex.x_parity, "  ", est[0], "+-", est[1])
    _ = log.within("GHZ Z-basis TVD at 50k shots (max over 8 settings)", max_tvd, 0.0, 0.01)
    _ = log.within("GHZ <X^n> sampled vs exact (max z-score)", max_z, 0.0, 4.0)
    _ = log.close("<X^n> closed forms: 1, (1-2p)^(2n-1), (1-4p/3)^(2n-1)", analytic_err, 0.0, 1e-12)

    # --- zero noise reproduces the ideal distribution exactly ---
    var z0 = with_gate_noise(ghz(5), "depolarizing", 0.0)
    var e0 = ghz_probabilities(exact_noise(z0, 5), 5)
    var ideal_err = abs(e0[0] - 0.5) + abs(e0[31] - 0.5)
    for b in range(1, 31):
        ideal_err += e0[b]
    var m0 = z0.copy()
    m0.measure_all()
    var r0 = sample(m0, 2000, 5)
    _ = log.close("zero noise: exact distribution is ideal", ideal_err, 0.0, 1e-12)
    _ = log.check("zero noise: only 00000 and 11111 are sampled", len(r0.counts) == 2,
                  String(len(r0.counts)) + " distinct outcomes")
    log.finish()
