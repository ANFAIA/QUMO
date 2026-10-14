"""Coined discrete-time quantum walk on a 128-site cycle.

Reproduces: Cirq - Quantum walks
Source: https://quantumai.google/cirq/experiments/quantum_walks (Cirq docs, CC BY 4.0 text, Apache-2.0 code)

The Cirq page walks on the integers 0..127 held in 7 position qubits, with one
coin qubit. One step = Hadamard on the coin, then +1 on the position when the
coin is |0> and -1 when it is |1> (its multi-controlled-X adder and
subtractor compose to exactly that; the derivation is in this folder's
README). The walker starts at position 32 (X on the second-most-significant
qubit) with coin |1>; after 30 steps the page samples 5000 shots and prints
a skewed histogram with its peak at 12 (1181 counts), then 14 (624), 10 (332),
20 (320) and a bump around 52 (52: 258). Starting from coin |0> mirrors it,
and coin (|0> + i|1>)/sqrt(2) (H then S) makes it symmetric. A classical
random walk spreads as sqrt(T) (50 steps: midpoint probability "a little
over 0.1"), the quantum walk as T.

QUMO builds the shift with `append_mcx` (Toffoli V-chain, 5 clean ancillas),
little-endian: position bit k = qubit k, coin = qubit 7, ancillas 8..12. An
independent check simulates the walk directly on a 2 x 128 complex vector.

Checks:
- the circuit's exact position distribution equals the direct 2 x 128
  simulation (1e-12) for the three coin states, and the ancillas end in |0>;
- the page's counts at 12, 14, 10, 20 and 52 are within 4 sigma of QUMO's
  exact probabilities (5000 shots);
- the sampled distribution (5000 shots) has a total-variation distance to
  the exact one below 1.5x its multinomial expectation
  E[TVD] ~ sum_x sqrt(p_x (1 - p_x) / (2 pi N)) (about 0.024 here, so the
  plan's fixed 0.03 would sit inside the noise); only even positions occur;
- skew: mean position < 27 from coin |1>, mirrored (> 37) from coin |0>, and
  exactly 32 (1e-9) for the balanced coin; the quantum standard deviation is
  more than twice the classical sqrt(30);
- classical 50-step walk: midpoint probability in [0.1, 0.12] (page).

Run: .venv/bin/mojo run -I . examples/frameworks/cirq/quantum_walk.mojo
example-meta: {"framework": "cirq", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import sqrt, pi
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import append_mcx, num_mcz_ancillas, bitstring_to_int
from qumo.ai.checks import CheckLog

comptime NPOS = 7
comptime SITES = 1 << NPOS
comptime COIN = NPOS
comptime STEPS = 30
comptime SHOTS = 5000
comptime START = 32


def increment(mut c: Circuit, coin_value: Int, ancillas: List[Int]) raises:
    """Position += 1 (mod 128) when the coin equals coin_value."""
    if coin_value == 0:
        c.x(COIN)
    var k = NPOS - 1
    while k >= 0:
        var controls = List[Int]()
        controls.append(COIN)
        for j in range(k):
            controls.append(j)
        append_mcx(c, controls, k, ancillas)
        k -= 1
    if coin_value == 0:
        c.x(COIN)


def walk_circuit(coin_state: Int, steps: Int) raises -> Circuit:
    """Walk circuit; coin_state 0 -> |0>, 1 -> |1>, 2 -> (|0> + i|1>)/sqrt(2)."""
    var nanc = num_mcz_ancillas(NPOS + 1)
    var c = Circuit(NPOS + 1 + nanc, name="quantum_walk")
    var anc = List[Int]()
    for a in range(nanc):
        anc.append(NPOS + 1 + a)
    c.x(5)  # position 32: Cirq's X on GridQubit(0, 1) in its big-endian register
    if coin_state == 1:
        c.x(COIN)
    elif coin_state == 2:
        c.h(COIN)
        c.s(COIN)
    for _ in range(steps):
        c.h(COIN)
        increment(c, 0, anc)  # coin |0>: x -> x + 1
        for q in range(NPOS):  # coin |1>: x -> x - 1 = NOT(NOT(x) + 1)
            c.x(q)
        increment(c, 1, anc)
        for q in range(NPOS):
            c.x(q)
    return c^


def direct_walk(coin_state: Int, steps: Int) -> List[Float64]:
    """Independent simulation: amplitudes a[x][coin] on a 2 x 128 grid."""
    var r0 = List[Float64](length=SITES, fill=0.0)
    var i0 = List[Float64](length=SITES, fill=0.0)
    var r1 = List[Float64](length=SITES, fill=0.0)
    var i1 = List[Float64](length=SITES, fill=0.0)
    var s = sqrt(0.5)
    if coin_state == 0:
        r0[START] = 1.0
    elif coin_state == 1:
        r1[START] = 1.0
    else:
        r0[START] = s
        i1[START] = s
    for _ in range(steps):
        var n0r = List[Float64](length=SITES, fill=0.0)
        var n0i = List[Float64](length=SITES, fill=0.0)
        var n1r = List[Float64](length=SITES, fill=0.0)
        var n1i = List[Float64](length=SITES, fill=0.0)
        for x in range(SITES):
            var a0r = s * (r0[x] + r1[x])  # Hadamard coin
            var a0i = s * (i0[x] + i1[x])
            var a1r = s * (r0[x] - r1[x])
            var a1i = s * (i0[x] - i1[x])
            n0r[(x + 1) % SITES] = a0r
            n0i[(x + 1) % SITES] = a0i
            n1r[(x + SITES - 1) % SITES] = a1r
            n1i[(x + SITES - 1) % SITES] = a1i
        r0 = n0r^
        i0 = n0i^
        r1 = n1r^
        i1 = n1i^
    var p = List[Float64](length=SITES, fill=0.0)
    for x in range(SITES):
        p[x] = r0[x] * r0[x] + i0[x] * i0[x] + r1[x] * r1[x] + i1[x] * i1[x]
    return p^


def mean_std(p: List[Float64]) -> Tuple[Float64, Float64]:
    var m = 0.0
    for x in range(len(p)):
        m += Float64(x) * p[x]
    var v = 0.0
    for x in range(len(p)):
        v += (Float64(x) - m) * (Float64(x) - m) * p[x]
    return (m, sqrt(v))


def main() raises:
    var log = CheckLog("cirq/quantum_walk")
    var names: List[String] = ["coin |0> (up)", "coin |1> (down, page default)", "coin (|0> + i|1>)/sqrt(2)"]
    var dists = List[List[Float64]]()
    var worst = 0.0
    var leak = 0.0
    var c_main = Circuit(1)
    for cs in range(3):
        var c = walk_circuit(cs, STEPS)
        if cs == 1:
            c_main = c.copy()
            print("walk circuit:", c.num_qubits, "qubits,", len(c), "gates (", c.count(31), "CCX )")
        var probs = evolve(c).probabilities()
        var pos = List[Float64](length=SITES, fill=0.0)
        for i in range(len(probs)):
            pos[i & (SITES - 1)] += probs[i]
            if (i >> (NPOS + 1)) != 0:
                leak += probs[i]
        var direct = direct_walk(cs, STEPS)
        for x in range(SITES):
            worst = max(worst, abs(pos[x] - direct[x]))
        var ms = mean_std(pos)
        print(names[cs], ": mean", ms[0], " std", ms[1])
        dists.append(pos^)
    _ = log.close("circuit equals direct 2 x 128 simulation", worst, 0.0, 1e-12)
    _ = log.close("ancillas return to |0>", leak, 0.0, 1e-12)

    var p = dists[1].copy()
    var page_x: List[Int] = [12, 14, 10, 20, 52]
    var page_n: List[Int] = [1181, 624, 332, 320, 258]
    for k in range(len(page_x)):
        var px = p[page_x[k]]
        var f = Float64(page_n[k]) / Float64(SHOTS)
        print("  position", page_x[k], ": exact P =", px, " page", page_n[k], "/", SHOTS)
        _ = log.close("page count at position " + String(page_x[k]), f, px, 4.0 * sqrt(px * (1.0 - px) / Float64(SHOTS)))

    var meas = c_main.copy()
    for q in range(NPOS):
        meas.measure(q, q)
    var r = sample(meas, SHOTS, seed=30)
    var tvd = 0.0
    var odd = 0
    var emp = List[Float64](length=SITES, fill=0.0)
    for kv in r.counts.items():
        var x = bitstring_to_int(kv.key)
        emp[x] += Float64(kv.value) / Float64(SHOTS)
        if x % 2 == 1:
            odd += kv.value
    for x in range(SITES):
        tvd += 0.5 * abs(emp[x] - p[x])
    var e_tvd = 0.0
    for x in range(SITES):
        e_tvd += sqrt(p[x] * (1.0 - p[x]) / (2.0 * pi * Float64(SHOTS)))
    print("sampled (", r.method, "): most frequent position", bitstring_to_int(r.most_frequent()), " TVD to exact", tvd,
          " multinomial E[TVD]", e_tvd)
    _ = log.within("sampled TVD to exact (5000 shots)", tvd, 0.0, 1.5 * e_tvd)
    _ = log.check("only even positions sampled", odd == 0, "odd-position shots=" + String(odd))

    var m_down = mean_std(dists[1])
    var m_up = mean_std(dists[0])
    var m_bal = mean_std(dists[2])
    _ = log.within("skewed left from coin |1> (mean)", m_down[0], 0.0, 27.0)
    _ = log.within("mirrored from coin |0> (mean)", m_up[0], 37.0, 128.0)
    _ = log.close("symmetric for the balanced coin (mean)", m_bal[0], 32.0, 1e-9)
    var classical_std = sqrt(Float64(STEPS))
    print("spread after", STEPS, "steps: quantum std", m_down[1], " classical std sqrt(T) =", classical_std)
    _ = log.within("quantum std / classical std", m_down[1] / classical_std, 2.0, 1e9)

    # classical walk, 50 steps, p = 0.5: distribution by direct convolution
    var cl = List[Float64](length=101, fill=0.0)
    cl[50] = 1.0
    for _ in range(50):
        var nx = List[Float64](length=101, fill=0.0)
        for x in range(1, 100):
            nx[x - 1] += 0.5 * cl[x]
            nx[x + 1] += 0.5 * cl[x]
        cl = nx^
    print("classical 50-step walk: P(midpoint) =", cl[50], "(page: 'a little bit over 0.1')")
    _ = log.within("classical midpoint probability (page)", cl[50], 0.1, 0.12)
    log.finish()
