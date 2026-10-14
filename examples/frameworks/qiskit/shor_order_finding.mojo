"""Shor order finding: N = 15 (tutorial circuit) and N = 21 (generic permutation oracle).

Reproduces: Qiskit - Shor's algorithm
Source: https://quantum.cloud.ibm.com/docs/tutorials/shors-algorithm (IBM Quantum documentation; tutorial code Apache-2.0, text CC BY-SA 4.0)

Part 1 is the tutorial's instance, N = 15 and a = 2:
- 8 control qubits (0..7) and 4 target qubits (8..11) prepared in |1>;
- M2 = multiplication by 2 mod 15 as SWAP(2,3) SWAP(1,2) SWAP(0,1) on the
  target register and M4 = SWAP(1,3) SWAP(0,2); the controlled versions come
  from the IR functor `Circuit.controlled()` (SWAP -> CSWAP), exactly as the
  tutorial builds them with `.control()`;
- control j applies a^(2^j) mod 15 = 2, 4, 1, 1, ... (identity powers skipped);
- the inverse QFT (`iqft`) on the control register.
The page's simulator counts (1024 shots) are 00000000: 264, 01000000: 268,
10000000: 249, 11000000: 243, i.e. phases {0, 1/4, 1/2, 3/4}; its continued-
fraction post-processing (denominator limit 15) gives r = 1, 4, 2, 4 and
the factor 3 from 2^(4/2) - 1.

Part 2 goes beyond the tutorial: N = 21, a = 2 (order 6) with the generic
reversible oracle `order_finding` (qumo/algorithms/library.mojo): every
controlled multiplication a^(2^j) mod 21 is a permutation of the 5-bit work
register, decomposed into Gray-code chains of multi-controlled X gates
(`append_controlled_permutation`). `order_from_phase` post-processes each
outcome (convergents plus small multiples), and gcd(a^(r/2) +- 1, N) gives
the factors 3 and 7.

Checks:
- N=15: the exact control-register distribution has peaks {0, 64, 128, 192}
  of 0.25 each (1e-12; page phases {0, 1/4, 1/2, 3/4}) and nothing else;
- N=15: the generic oracle (order_finding) gives the same distribution (1e-12);
- N=15: order_from_phase gives r = 4 for 64, 128 and 192, and gcd gives 3 and 5;
- N=15: 2000 shots stay within 4 sigma of 0.25 per peak (also page-consistent);
- N=21: order_from_phase returns 6 on >= 50% of the probability mass and of 2000 samples;
- N=21: the factors are 3 and 7.

Optional reference: reference/shor15_ref.py rebuilds the tutorial circuit in
qiskit and compares 10,000 Aer shots with this distribution (TVD < 0.02).

Run: .venv/bin/mojo run -I . examples/frameworks/qiskit/shor_order_finding.mojo
example-meta: {"framework": "qiskit", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import sqrt
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.sim.rng import Rng
from qumo.algorithms.library import iqft, order_finding, order_from_phase
from qumo.compiler.target import parse_target
from qumo.compiler.pass_manager import transpile
from qumo.ai.checks import CheckLog

comptime N_CTRL = 8
comptime SHOTS = 2000


def gcd(a: Int, b: Int) -> Int:
    var x = abs(a)
    var y = abs(b)
    while y != 0:
        var t = x % y
        x = y
        y = t
    return x


def pow_mod(b: Int, e: Int, m: Int) -> Int:
    var r = 1
    for _ in range(e):
        r = (r * b) % m
    return r


def m2_mod15() -> Circuit:
    var c = Circuit(4, name="M2")
    c.swap(2, 3)
    c.swap(1, 2)
    c.swap(0, 1)
    return c^


def m4_mod15() -> Circuit:
    var c = Circuit(4, name="M4")
    c.swap(1, 3)
    c.swap(0, 2)
    return c^


def tutorial_circuit() raises -> Circuit:
    """The tutorial's 12-qubit order-finding circuit for N = 15, a = 2 (no measurements)."""
    var c = Circuit(N_CTRL + 4, name="shor15")
    c.x(N_CTRL)  # target register |1>
    for j in range(N_CTRL):
        c.h(j)
    var m2 = m2_mod15().controlled(4)
    var m4 = m4_mod15().controlled(4)
    var qmap: List[Int] = [N_CTRL, N_CTRL + 1, N_CTRL + 2, N_CTRL + 3, 0]
    c.compose_mapped(m2, qmap)
    qmap[4] = 1
    c.compose_mapped(m4, qmap)
    c.compose(iqft(N_CTRL))
    return c^


def control_marginal(c: Circuit, n: Int) raises -> List[Float64]:
    var probs = evolve(c).probabilities()
    var out = List[Float64](length=1 << n, fill=0.0)
    var mask = (1 << n) - 1
    for i in range(len(probs)):
        out[i & mask] += probs[i]
    return out^


def main() raises:
    var log = CheckLog("qiskit/shor_order_finding")

    # ---------------- Part 1: N = 15, tutorial circuit ----------------
    var c15 = tutorial_circuit()
    print("N=15 tutorial circuit:", c15.num_qubits, "qubits,", len(c15), "gates (", c15.count(32), "CSWAP )")
    var tg = parse_target("grid:3x4/sc_cz")
    var comp = transpile(c15, tg, 2)
    print("  compiled for the neutral target grid:3x4/sc_cz:", comp.summary())
    var pe = control_marginal(c15, N_CTRL)
    var peaks: List[Int] = [0, 64, 128, 192]
    var peak_err = 0.0
    var off_peak = 0.0
    for y in range(1 << N_CTRL):
        var is_peak = y == 0 or y == 64 or y == 128 or y == 192
        if is_peak:
            peak_err = max(peak_err, abs(pe[y] - 0.25))
        else:
            off_peak += pe[y]
    print("  exact peaks:", pe[0], pe[64], pe[128], pe[192], " off-peak mass:", off_peak)
    _ = log.close("N=15 peaks {0,64,128,192} are 0.25 each", peak_err, 0.0, 1e-12)
    _ = log.close("N=15 off-peak probability", off_peak, 0.0, 1e-12)

    var gen15 = order_finding(2, 15, N_CTRL)
    var pg = control_marginal(gen15, N_CTRL)
    var dist = 0.0
    for y in range(1 << N_CTRL):
        dist = max(dist, abs(pg[y] - pe[y]))
    print("  generic oracle order_finding(2, 15, 8):", gen15.num_qubits, "qubits,", len(gen15), "gates; max |dP| =", dist)
    _ = log.close("N=15 generic oracle equals the SWAP construction", dist, 0.0, 1e-12)

    var r_ok = True
    for y in peaks:
        var r = order_from_phase(y, N_CTRL, 2, 15)
        print("  measured", y, "-> phase", Float64(y) / 256.0, "-> r =", r)
        if y != 0 and r != 4:
            r_ok = False
    _ = log.check("N=15 order_from_phase gives r=4 for 64, 128, 192", r_ok)
    var x15 = pow_mod(2, 2, 15)
    var f1 = gcd(x15 - 1, 15)
    var f2 = gcd(x15 + 1, 15)
    print("  r = 4: gcd(2^2 - 1, 15) =", f1, " gcd(2^2 + 1, 15) =", f2, " (page: factor 3)")
    _ = log.check("N=15 factors are 3 and 5", f1 * f2 == 15 and f1 == 3, String(f1) + " x " + String(f2))

    var cm = c15.copy()
    for j in range(N_CTRL):
        cm.measure(j, j)
    var s15 = sample(cm, SHOTS, seed=15)
    print("  sampled", SHOTS, "shots:", s15, " (page, 1024 shots: 264 / 268 / 249 / 243)")
    var max_z = 0.0
    var sigma = sqrt(0.25 * 0.75 / Float64(SHOTS))
    var on_peak = 0
    var keys: List[String] = ["00000000", "01000000", "10000000", "11000000"]
    for k in keys:
        var n = s15.counts.get(k, 0)
        on_peak += n
        max_z = max(max_z, abs(Float64(n) / Float64(SHOTS) - 0.25) / sigma)
    _ = log.check("N=15 every shot lands on a peak", on_peak == SHOTS, String(on_peak) + " of " + String(SHOTS))
    _ = log.within("N=15 sampled peak frequencies (max z-score)", max_z, 0.0, 4.0)
    # the page's own counts against the exact 1/4 (page data, 1024 shots)
    var page: List[Int] = [264, 268, 249, 243]
    var page_z = 0.0
    var sp = sqrt(0.25 * 0.75 / 1024.0)
    for n in page:
        page_z = max(page_z, abs(Float64(n) / 1024.0 - 0.25) / sp)
    _ = log.within("page counts are consistent with 1/4 each (max z-score)", page_z, 0.0, 4.0)

    # ---------------- Part 2: N = 21, generic permutation oracle ----------------
    var c21 = order_finding(2, 21, N_CTRL)
    print("N=21 order finding:", c21.num_qubits, "qubits (8 eval + 5 work + 3 ancilla),", len(c21), "gates,",
          c21.num_2q_gates(), "two-qubit gates")
    # one state-vector evolution serves both the exact marginal and the samples
    var sv21 = evolve(c21)
    var probs21 = sv21.probabilities()
    var p21 = List[Float64](length=1 << N_CTRL, fill=0.0)
    for i in range(len(probs21)):
        p21[i & ((1 << N_CTRL) - 1)] += probs21[i]
    var mass6 = 0.0
    for y in range(1 << N_CTRL):
        if order_from_phase(y, N_CTRL, 2, 21) == 6:
            mass6 += p21[y]
    print("  exact probability that order_from_phase returns 6:", mass6)
    _ = log.within("N=21 probability mass giving r=6", mass6, 0.5, 1.0)
    var rng = Rng(21)
    var outs = sv21.sample(SHOTS, rng)  # measuring every qubit; keep the evaluation register
    var hits = 0
    for o in outs:
        if order_from_phase(o & ((1 << N_CTRL) - 1), N_CTRL, 2, 21) == 6:
            hits += 1
    var frac = Float64(hits) / Float64(SHOTS)
    print("  sampled", SHOTS, "shots:", hits, "give r = 6 (", frac, ")")
    _ = log.within("N=21 fraction of samples giving r=6", frac, 0.5, 1.0)
    var x21 = pow_mod(2, 3, 21)
    var g1 = gcd(x21 - 1, 21)
    var g2 = gcd(x21 + 1, 21)
    print("  r = 6: gcd(2^3 - 1, 21) =", g1, " gcd(2^3 + 1, 21) =", g2)
    _ = log.check("N=21 factors are 3 and 7", g1 * g2 == 21 and min(g1, g2) == 3, String(g1) + " x " + String(g2))
    log.finish()
