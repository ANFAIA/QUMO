"""Quantum Fourier transform: exact, approximate, and large-n round trip.

Reproduces: Qibo - Basic examples: "How to write a Quantum Fourier Transform?"
Source: https://qibo.science/qibo/stable/code-examples/examples.html (Qibo docs, Apache-2.0)

The Qibo page builds `models.QFT(15)` and executes it on the default
all-zeros state, which gives the uniform superposition (every amplitude
2^{-15/2}). The page prints no numbers. QUMO uses `qft(n, do_swaps,
approximation_degree)` (qumo/algorithms/library.mojo), with these parts:
- A: the page instance, QFT(15)|0...0>;
- B: QFT of seeded random states for n = 5..10 against a classical DFT
  written in this file (QFT|j> = 2^{-n/2} sum_k exp(2 pi i j k / 2^n) |k>);
- C: approximate QFT (Coppersmith; the CP rotations with the smallest angles
  are dropped) on n = 10: fidelity and gate counts versus the degree;
- D: n = 20 round trip, which prepares the product state QFT|k> and applies
  the inverse QFT, so every shot reads k. It runs on the CPU and with
  `sample_on(..., "gpu")`; without -D QUMO_GPU, or without a usable device,
  the GPU row runs on the CPU and prints why.

Checks:
- A: every amplitude of QFT(15)|0> equals 2^{-7.5} (1e-12, analytic);
- B: max |QFT psi - DFT psi| < 1e-12 for n = 5..10 (analytic);
- C: AQFT fidelity is non-decreasing as the degree falls, and equals 1 at
  degree 0 (1e-12);
- D: all shots read k on the CPU and on the GPU route (fp32 on the GPU is
  fine: the readout is deterministic).

Run: .venv/bin/mojo run -I . examples/frameworks/qibo/qft.mojo
GPU: .venv/bin/mojo run -I . -D QUMO_GPU examples/frameworks/qibo/qft.mojo
example-meta: {"framework": "qibo", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import pi, sqrt, cos, sin
from std.time import perf_counter_ns
from qumo.ir.circuit import Circuit
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, evolve_into, bitstring
from qumo.sim.rng import Rng
from qumo.algorithms.library import qft, iqft
from qumo.algorithms.linalg import sv_dot
from qumo.exec.dispatch import sample_on
from qumo.ai.checks import CheckLog


def random_state(n: Int, seed: UInt64) -> StateVector:
    """Seeded Haar-like state: normalised complex Gaussian amplitudes."""
    var rng = Rng(seed)
    var sv = StateVector(n)
    var s = 0.0
    for i in range(sv.size()):
        sv.re[i] = rng.normal()
        sv.im[i] = rng.normal()
        s += sv.re[i] * sv.re[i] + sv.im[i] * sv.im[i]
    var inv = 1.0 / sqrt(s)
    for i in range(sv.size()):
        sv.re[i] *= inv
        sv.im[i] *= inv
    return sv^


def classical_dft(psi: StateVector) -> StateVector:
    """Classical DFT: out[k] = 2^{-n/2} sum_j exp(+2 pi i j k / N) psi[j] (O(N^2), twiddle table)."""
    var nn = psi.size()
    var wr = List[Float64](capacity=nn)
    var wi = List[Float64](capacity=nn)
    for m in range(nn):
        var a = 2.0 * pi * Float64(m) / Float64(nn)
        wr.append(cos(a))
        wi.append(sin(a))
    var out = StateVector(psi.num_qubits)
    var norm = 1.0 / sqrt(Float64(nn))
    for k in range(nn):
        var sr = 0.0
        var si = 0.0
        for j in range(nn):
            var m = (j * k) % nn
            sr += wr[m] * psi.re[j] - wi[m] * psi.im[j]
            si += wr[m] * psi.im[j] + wi[m] * psi.re[j]
        out.re[k] = sr * norm
        out.im[k] = si * norm
    return out^


def max_diff(a: StateVector, b: StateVector) -> Float64:
    var d = 0.0
    for i in range(a.size()):
        var dr = a.re[i] - b.re[i]
        var di = a.im[i] - b.im[i]
        d = max(d, sqrt(dr * dr + di * di))
    return d


def fourier_state_prep(n: Int, k: Int) -> Circuit:
    """Product-state preparation of QFT|k> (with swaps): qubit q carries
    (|0> + exp(2 pi i k 2^q / 2^n)|1>) / sqrt(2)."""
    var c = Circuit(n, name="fourier_prep")
    for q in range(n):
        c.h(q)
        var frac = Float64((k << q) % (1 << n)) / Float64(1 << n)
        if frac != 0.0:
            c.p(2.0 * pi * frac, q)
    return c^


def main() raises:
    var log = CheckLog("qibo/qft")

    # ---- A: the page instance, QFT(15) on |0...0> ----
    var c15 = qft(15)
    var t0 = perf_counter_ns()
    var s15 = evolve(c15)
    var ms15 = Float64(perf_counter_ns() - t0) / 1e6
    var amp = 1.0 / sqrt(Float64(1 << 15))
    var dev15 = 0.0
    for i in range(s15.size()):
        dev15 = max(dev15, max(abs(s15.re[i] - amp), abs(s15.im[i])))
    print("A: QFT(15)|0>:", len(c15), "gates,", c15.num_2q_gates(), "2-qubit,", ms15, "ms; max |amp - 2^-7.5| =", dev15)
    _ = log.close("QFT(15)|0> is the uniform superposition", dev15, 0.0, 1e-12)

    # ---- B: random states, n = 5..10, against the classical DFT ----
    var worst = 0.0
    print("B: n   gates  H  CP  SWAP   max|QFT psi - DFT psi|")
    for n in range(5, 11):
        var c = qft(n)
        var psi = random_state(n, UInt64(100 + n))
        var q = psi.copy()
        evolve_into(q, c, List[Float64]())
        var d = max_diff(q, classical_dft(psi))
        worst = max(worst, d)
        print("  ", n, "  ", len(c), "  ", n, "  ", n * (n - 1) // 2, "  ", n // 2, "  ", d)
    _ = log.close("QFT equals the DFT for n = 5..10 (max abs diff)", worst, 0.0, 1e-12)

    # ---- C: approximate QFT on n = 10 ----
    var na = 10
    var psi_a = random_state(na, 7)
    var exact = psi_a.copy()
    evolve_into(exact, qft(na), List[Float64]())
    var prev_f = -1.0
    var monotone = True
    var f0 = 0.0
    print("C: AQFT n = 10: degree  CP gates  fidelity |<QFT psi|AQFT psi>|^2")
    var deg = na - 1
    while deg >= 0:
        var ca = qft(na, approximation_degree=deg)
        var st = psi_a.copy()
        evolve_into(st, ca, List[Float64]())
        var ov = sv_dot(exact, st)
        var f = ov[0] * ov[0] + ov[1] * ov[1]
        print("   ", deg, "    ", ca.num_2q_gates() - na // 2, "    ", f)
        if f < prev_f - 1e-12:
            monotone = False
        prev_f = f
        f0 = f
        deg -= 1
    _ = log.check("AQFT fidelity non-decreasing as the degree falls", monotone, "last=" + String(prev_f))
    _ = log.close("AQFT degree 0 is the exact QFT", f0, 1.0, 1e-12)

    # ---- D: n = 20 round trip on the CPU and the GPU route ----
    var nd = 20
    var k = 0xA5C3B
    var rt = fourier_state_prep(nd, k)
    rt.compose(iqft(nd))
    rt.measure_all()
    var key = bitstring(k, nd)
    var devices: List[String] = ["cpu", "gpu"]
    for dev in devices:
        var note = String()
        t0 = perf_counter_ns()
        var r = sample_on(rt, 200, 11, List[Float64](), dev, note)
        var ms = Float64(perf_counter_ns() - t0) / 1e6
        var hits = r.counts.get(key, 0)
        print("D: n = 20 iQFT(QFT|k>) on", dev, ": method", r.method, ",", ms, "ms,", hits, "/ 200 shots read k")
        if note.byte_length() > 0:
            print("   gpu note:", note)
        _ = log.check("n = 20 round trip reads k on " + dev, hits == 200, "hits=" + String(hits) + " method=" + r.method)
    log.finish()
