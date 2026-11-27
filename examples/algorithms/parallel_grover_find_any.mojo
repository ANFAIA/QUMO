"""Parallel Grover, find any marked item: an odd-parity query and a weakly measured `while` loop.

Reproduces: Shirobokov et al. - odd-parity joint reflection, parallel Find-Any (Appendix E.3)
Source: https://arxiv.org/abs/2611.09822
Source: https://github.com/google-deepmind/formal-quantum-computation/tree/discoveries (Apache-2.0, `FindAny.lean`, `OddParityBridge.lean`)
Framework: QUMO - https://github.com/ANFAIA/QUMO

N = 2^n items, k of them marked, and m oracle calls allowed side by side in one
round. Each of the m slots holds an address and one flag qubit. Written here as a
QUMO circuit from the operators in the authors' Lean files; no code is copied,
and QUMO does not check their proofs.

  parity phase  W O^m W with W = XHX on each flag. On flags |0...0> the m parallel
                queries multiply an address tuple by -1 once per marked slot, so
                only tuples with an ODD number of marked slots change sign. Their
                share is p = (1 - (1 - 2k/N)^m) / 2, about mk/N.
  warm up       Hadamards, then K + 1 Grover steps with K = floor(pi / (8 sqrt p)):
                about half way.
  weak check    U (W O^m W) U with U = u Z_flag0 + v X^(all flags). U is its own
                inverse, so an even tuple returns to flags 0. An odd one reaches
                flags 1...1 with amplitude -2uv. Measure flag 0: 1 ends the loop;
                0 is followed by one diffusion step, then the loop repeats. u and v
                are chosen so the state after a miss is the state before it.
  read out      flags back to 0, one more round of m queries into the flags,
                measure everything and return the address of a slot whose flag is 1.

One round is one query layer, so the expected number of rounds is
(K + 1) + 1 / P(hit) + 1. As N grows, this over (pi/4) sqrt(N / (mk)) tends to
1/2 + 1/pi = 0.8183, the constant of Fang and Heunen's weakly measured search for
one oracle (arXiv:2609.39621), here for m oracles. The paper compares it, for
m, k >= 2, with m independent searches (Gingrich, Williams, Cerf), whose constant
rises from 0.942 at m = 2 to 0.997 at m = 1024.

Limits. The constant is a large-N statement: at the sizes simulated here the loop
costs MORE rounds than the reference. The paper's Lean certificate covers
n >= 150 + log m + log k and uses a classical fallback below that; this file runs
the loop itself at small n, where u and v exist only for some (n, m, k). This is
the paper's Find-Any program only. Its Find-All mechanisms, its fixed-point
arcsin circuit and its three physics and chemistry resource estimates are not
reproduced.

Checks:
- zero error: on a hit, all probability lies on tuples with an odd number of
  marked slots, with every flag at 1;
- the loop is stationary: P(hit) on the first and on the third check agree, and
  equal 1 - (cos((2K+3)a) / cos((2K+1)a))^2 with sin a = sqrt p;
- run as a QUMO program with a `while` loop, every shot returns a marked item and
  the mean number of checks matches 1 / P(hit);
- the closed form tends to 1/2 + 1/pi, and the independent-search baseline
  computed here gives the paper's 0.942 and 0.997.

Run: .venv/bin/mojo run -I . examples/algorithms/parallel_grover_find_any.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 300, "requires": [], "args": []}
"""

from std.math import asin, atan2, cos, sin, sqrt, floor, log1p, expm1

from qumo.ir.circuit import Circuit
from qumo.ir.classical import ce_lnot
from qumo.ir.mc import append_mcx
from qumo.ir.program import ProgramBuilder
from qumo.sim.program_runner import sample_program
from qumo.sim.runner import apply_instruction, evolve
from qumo.sim.statevector import StateVector
from qumo.testing.checks import CheckLog

comptime PI = 3.141592653589793


struct Search(Copyable, Movable):
    """One instance: n address bits, m slots, and the marked items."""

    var n: Int
    var m: Int
    var marked: List[Int]
    var p: Float64  # share of odd-parity tuples
    var angle: Float64  # asin(sqrt(p))
    var K: Int
    var ratio: Float64  # v^2 - u^2 = -ratio
    var predicted_hit: Float64

    def __init__(out self, n: Int, m: Int, var marked: List[Int]):
        self.n = n
        self.m = m
        var k = len(marked)
        self.marked = marked^
        self.p = odd_share(Float64(k) / Float64(1 << n), m)
        self.angle = asin(sqrt(self.p))
        self.K = Int(floor(PI / 8.0 / sqrt(self.p)))
        var a = Float64(2 * self.K + 1) * self.angle
        var b = Float64(2 * self.K + 3) * self.angle
        var stay = cos(b) / cos(a)
        self.ratio = stay * sin(a) / sin(b)
        self.predicted_hit = 1.0 - stay * stay

    def flag(self, slot: Int) -> Int:
        return slot * (self.n + 1)

    def address(self, slot: Int, bit: Int) -> Int:
        return slot * (self.n + 1) + 1 + bit

    def qubits(self) -> Int:
        return self.m * (self.n + 1)


def odd_share(x: Float64, m: Int) -> Float64:
    """(1 - (1 - 2x)^m) / 2, accurate for tiny x."""
    return -expm1(Float64(m) * log1p(-2.0 * x)) / 2.0


def oracle_round(mut c: Circuit, s: Search) raises:
    """One parallel round: every slot flips its flag if its address is marked."""
    for slot in range(s.m):
        var controls = List[Int]()
        for bit in range(s.n):
            controls.append(s.address(slot, bit))
        for item in s.marked:
            for bit in range(s.n):
                if (item >> bit) & 1 == 0:
                    c.x(s.address(slot, bit))
            append_mcx(c, controls, s.flag(slot))
            for bit in range(s.n):
                if (item >> bit) & 1 == 0:
                    c.x(s.address(slot, bit))


def parity_phase(mut c: Circuit, s: Search) raises:
    for slot in range(s.m):
        c.x(s.flag(slot))
        c.h(s.flag(slot))
        c.x(s.flag(slot))
    oracle_round(c, s)
    for slot in range(s.m):
        c.x(s.flag(slot))
        c.h(s.flag(slot))
        c.x(s.flag(slot))


def diffuse(mut c: Circuit, s: Search) raises:
    """Reflection about the uniform superposition of all address tuples (up to a global sign)."""
    var wires = List[Int]()
    for slot in range(s.m):
        for bit in range(s.n):
            wires.append(s.address(slot, bit))
    for q in wires:
        c.h(q)
        c.x(q)
    var target = wires[len(wires) - 1]
    var controls = List[Int]()
    for j in range(len(wires) - 1):
        controls.append(wires[j])
    c.h(target)
    append_mcx(c, controls, target)
    c.h(target)
    for q in wires:
        c.x(q)
        c.h(q)


def chord(mut c: Circuit, s: Search):
    """U = u Z(flag 0) + v X(all flags), with v^2 - u^2 = -ratio: a CX fan-out
    turns the flip of all flags into a flip of flag 0."""
    var u = sqrt((1.0 + s.ratio) / 2.0)
    var v = sqrt((1.0 - s.ratio) / 2.0)
    for slot in range(1, s.m):
        c.cx(s.flag(0), s.flag(slot))
    c.z(s.flag(0))
    c.ry(2.0 * atan2(v, u), s.flag(0))
    for slot in range(1, s.m):
        c.cx(s.flag(0), s.flag(slot))


def warm_up(mut c: Circuit, s: Search) raises:
    for slot in range(s.m):
        for bit in range(s.n):
            c.h(s.address(slot, bit))
    for _ in range(s.K + 1):
        parity_phase(c, s)
        diffuse(c, s)


def weak_check(mut c: Circuit, s: Search) raises:
    chord(c, s)
    parity_phase(c, s)
    chord(c, s)


def run_on(mut sv: StateVector, c: Circuit):
    var none = List[Float64]()
    for i in range(len(c.ops)):
        apply_instruction(sv, c.ops[i], none)


def hit_probability(sv: StateVector, s: Search) -> Float64:
    var out = 0.0
    for i in range(sv.size()):
        if (i >> s.flag(0)) & 1:
            out += sv.re[i] * sv.re[i] + sv.im[i] * sv.im[i]
    return out


def good_hit_probability(sv: StateVector, s: Search) -> Float64:
    """Weight on hits with every flag 1 and an odd number of marked slots."""
    var out = 0.0
    for i in range(sv.size()):
        var all_flags = True
        var hits = 0
        for slot in range(s.m):
            all_flags = all_flags and ((i >> s.flag(slot)) & 1) == 1
            var item = (i >> s.address(slot, 0)) & ((1 << s.n) - 1)
            for x in s.marked:
                if item == x:
                    hits += 1
        if all_flags and hits % 2 == 1:
            out += sv.re[i] * sv.re[i] + sv.im[i] * sv.im[i]
    return out


def project_miss(mut sv: StateVector, s: Search):
    var keep = sqrt(1.0 - hit_probability(sv, s))
    for i in range(sv.size()):
        if (i >> s.flag(0)) & 1:
            sv.re[i] = 0.0
            sv.im[i] = 0.0
        else:
            sv.re[i] /= keep
            sv.im[i] /= keep


def check_exact(mut log: CheckLog, label: String, s: Search) raises:
    var start = Circuit(s.qubits())
    warm_up(start, s)
    var check = Circuit(s.qubits())
    weak_check(check, s)
    var after_miss = Circuit(s.qubits())
    diffuse(after_miss, s)
    var sv = evolve(start)
    var first = 0.0
    var third = 0.0
    var leak = 0.0
    for r in range(3):
        run_on(sv, check)
        var hit = hit_probability(sv, s)
        leak = max(leak, abs(hit - good_hit_probability(sv, s)))
        if r == 0:
            first = hit
        third = hit
        project_miss(sv, s)
        run_on(sv, after_miss)
    var expected_rounds = Float64(s.K + 1) + 1.0 / first + 1.0
    var reference = PI / 4.0 * sqrt(Float64(1 << s.n) / Float64(s.m * len(s.marked)))
    print(label, ": ", s.qubits(), " qubits, K=", s.K, ", P(hit)=", first, ", expected rounds=",
          expected_rounds, ", over (pi/4)sqrt(N/mk)=", expected_rounds / reference, sep="")
    _ = log.check(label + " schedule exists", s.ratio >= 0.0 and s.ratio <= 1.0, String(s.ratio))
    _ = log.close(label + " every hit is an odd tuple", leak, 0.0, 1e-10)
    _ = log.close(label + " P(hit) matches the closed form", first, s.predicted_hit, 1e-9)
    # each miss is renormalised by 1/sqrt(P(miss)), which grows rounding error when
    # P(miss) is small (0.011 at m = 3), hence the looser tolerance here
    _ = log.close(label + " the loop is stationary", third, first, 1e-7)


def check_program_run(mut log: CheckLog, label: String, s: Search, shots: Int) raises:
    """The whole algorithm as one program: warm up, `while` not hit, read out."""
    var width = 1 + s.qubits()  # c[0] is the loop bit, then every qubit
    var b = ProgramBuilder(s.qubits(), width)
    warm_up(b.body, s)
    b.begin_while(ce_lnot(b.bit(0)))
    weak_check(b.body, s)
    b.body.measure(s.flag(0), 0)
    b.begin_if(ce_lnot(b.bit(0)))
    diffuse(b.body, s)
    b.end()
    b.end()
    for slot in range(s.m):  # flags are 1...1 after a hit
        b.body.x(s.flag(slot))
    oracle_round(b.body, s)
    for q in range(s.qubits()):
        b.body.measure(q, 1 + q)
    var program = b.finish()
    var run = sample_program(program, shots, 2026)

    var wrong = 0
    for key in run.sample.sorted_keys():
        var text = key.as_bytes()
        var value = 0
        for j in range(len(text)):
            value = value * 2 + Int(text[j]) - ord("0")
        var read = value >> 1  # drop the loop bit
        var flagged = 0
        var all_marked = True
        for slot in range(s.m):
            if (read >> s.flag(slot)) & 1:
                flagged += 1
                var item = (read >> s.address(slot, 0)) & ((1 << s.n) - 1)
                var known = False
                for x in s.marked:
                    known = known or x == item
                all_marked = all_marked and known
        if flagged % 2 != 1 or not all_marked:
            wrong += run.sample.counts.get(key, 0)
    _ = log.check(label + " every shot returns a marked item", wrong == 0, String(wrong) + " wrong shots")
    # one back-edge per pass through the loop body; the count of checks is geometric
    var mean = run.mean_iterations()
    var q = s.predicted_hit
    _ = log.close(label + " mean checks match 1 over P(hit)", mean, 1.0 / q,
                  4.0 * sqrt((1.0 - q) / (q * q) / Float64(shots)))


def asymptotic_ratio(n: Int, m: Int, k: Int) -> Float64:
    """Expected rounds over (pi/4) sqrt(N / mk), from the closed forms."""
    var size = Float64(1 << n)
    var p = odd_share(Float64(k) / size, m)
    var angle = asin(sqrt(p))
    var K = floor(PI / 8.0 / sqrt(p))
    var stay = cos((2.0 * K + 3.0) * angle) / cos((2.0 * K + 1.0) * angle)
    var rounds = K + 1.0 + 1.0 / (1.0 - stay * stay) + 1.0
    return rounds / (PI / 4.0 * sqrt(size / (Float64(m) * Float64(k))))


def independent_searches(m: Int) -> Float64:
    """Constant of m independent early-stopped searches (Eq. 10): the best stopping angle phi
    minimises phi / (1 - cos^(2m) phi)."""
    var lo = 1e-6
    var hi = PI / 2.0 - 1e-6
    for _ in range(200):  # ternary search on a unimodal cost
        var a = lo + (hi - lo) / 3.0
        var b = hi - (hi - lo) / 3.0
        var fa = a / (-expm1(2.0 * Float64(m) * log1p(cos(a) - 1.0)))
        var fb = b / (-expm1(2.0 * Float64(m) * log1p(cos(b) - 1.0)))
        if fa < fb:
            hi = b
        else:
            lo = a
    return 2.0 * sqrt(Float64(m)) * lo / (PI * (-expm1(2.0 * Float64(m) * log1p(cos(lo) - 1.0))))


def main() raises:
    var log = CheckLog("parallel_grover_find_any")

    check_exact(log, "m=1 k=1 N=64", Search(6, 1, [45]))
    check_exact(log, "m=2 k=1 N=32", Search(5, 2, [19]))
    check_exact(log, "m=2 k=2 N=64", Search(6, 2, [5, 58]))
    check_exact(log, "m=3 k=1 N=32", Search(5, 3, [7]))

    check_program_run(log, "m=1 program", Search(6, 1, [45]), 4000)
    check_program_run(log, "m=2 program", Search(5, 2, [19]), 600)

    var limit = 0.5 + 1.0 / PI
    for m in [1, 2, 3]:
        _ = log.close("ratio tends to 1/2 + 1/pi at m=" + String(m), asymptotic_ratio(48, m, m), limit, 1e-4)
    for m in [2, 4, 1024]:
        print("m=", m, ": independent searches ", independent_searches(m), ", odd parity ", limit,
              ", speed-up ", independent_searches(m) / limit, sep="")
    _ = log.close("independent searches at m=2", independent_searches(2), 0.942, 5e-4)
    _ = log.close("independent searches at m=1024", independent_searches(1024), 0.997, 5e-4)
    _ = log.close("largest speed-up over them on this grid", independent_searches(1024) / limit, 1.22, 5e-3)
    log.finish()
