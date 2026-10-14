"""QAOA MaxCut, observe then sample (CUDA-Q instance).

Reproduces: CUDA-Q - Max-Cut with QAOA
Source: https://nvidia.github.io/cuda-quantum/0.14.0/applications/python/qaoa.html (CUDA-Q docs, Apache-2.0; notebook removed from the repo at 0.15.0)

The CUDA-Q page solves MaxCut on 5 nodes with edges (0,1) (1,2) (2,3) (3,0)
(2,4) (3,4), using the Hamiltonian H = sum over edges of 0.5 (Z_u Z_v - I)
(so <z|H|z> = -cut(z)), 2 QAOA layers with parameters thetas[0..1] for the
cost layers (CX . RZ(2 theta) . CX per edge = RZZ(2 theta)) and thetas[2..3]
for the mixers (RX(2 theta) on every qubit), cudaq.optimizers.NelderMead
from a seeded initial point, then `cudaq.sample` of the optimised kernel.
Page numbers: optimal expectation -4.495973826282007 at
[0.51349181993727, -0.21299416361632417, 0.3250526425808945, 0.886630847343767];
1000 shots with top bitstring 10101 (165 counts), then 01011 (156),
10100 (155) and 01010 (153) - all four are maximum cuts of value 5.

QUMO builds the same kernel as a `Circuit` with symbolic parameters (affine
`Param`s: RZZ(2 theta)), evaluates it exactly with `ExpectationEngine`
(CUDA-Q `observe`) and samples it with `sample` (CUDA-Q `sample`). CUDA-Q
bitstrings list qubit 0 first; QUMO keys list the highest qubit first, so the
printed strings are converted.

Checks:
- <H> at the page's optimal parameters equals the page value (1e-9);
- the page's 1000-shot counts of the four optimal cuts are within 4 sigma of
  QUMO's exact probabilities at those parameters;
- Nelder-Mead from the page's initial point and 5 seeded restarts: best <H>
  <= -4.49 and >= -5 (the exact minimum, -MaxCut);
- the most frequent of 10,000 samples is an optimal cut (cut value 5).

Run: .venv/bin/mojo run -I . examples/frameworks/cudaq/qaoa_nelder_mead.mojo
example-meta: {"framework": "cudaq", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import pi, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import evolve, sample
from qumo.algorithms.hamiltonians import Graph
from qumo.algorithms.gradients import ExpectationEngine
from qumo.algorithms.optimizers import NelderMead
from qumo.algorithms.variational import VQEObjective
from qumo.algorithms.library import bitstring_to_int
from qumo.sim.rng import Rng
from qumo.ai.checks import CheckLog

comptime N = 5
comptime LAYERS = 2
comptime SHOTS = 10000
comptime PAGE_OPT_ENERGY = -4.495973826282007


def page_graph() raises -> Graph:
    return Graph.parse("0-1 1-2 2-3 3-0 2-4 3-4")


def hamiltonian(g: Graph) -> PauliSum:
    """H = sum_(u,v) 0.5 (Z_u Z_v - I)  (CUDA-Q's convention, minimised)."""
    var h = PauliSum()
    for e in g.edges:
        h.add(0.5, 0, (UInt64(1) << UInt64(e.u)) | (UInt64(1) << UInt64(e.v)))
        h.add(-0.5, 0, 0)
    return h^


def kernel(g: Graph) -> Circuit:
    """The page's kernel: thetas[0..L-1] cost angles, thetas[L..2L-1] mixer angles."""
    var c = Circuit(N, name="qaoa_cudaq")
    var th = c.parameters("thetas", 2 * LAYERS)
    for q in range(N):
        c.h(q)
    for i in range(LAYERS):
        for e in g.edges:
            c.rzz(th[i] * 2.0, e.u, e.v)  # x.ctrl; rz(2 theta); x.ctrl
        for q in range(N):
            c.rx(th[i + LAYERS] * 2.0, q)
    return c^


def cudaq_order(key: String) -> String:
    """QUMO key (highest qubit first) -> CUDA-Q string (qubit 0 first)."""
    var bs = key.as_bytes()
    var out = String()
    var i = key.byte_length() - 1
    while i >= 0:
        out += String(unsafe_from_utf8=bs[i:i + 1])
        i -= 1
    return out^


def cudaq_index(s: String) -> Int:
    """CUDA-Q bitstring (qubit 0 first) -> basis index (bit k = qubit k)."""
    var bs = s.as_bytes()
    var v = 0
    for k in range(s.byte_length()):
        if Int(bs[k]) == ord("1"):
            v |= 1 << k
    return v


def main() raises:
    var log = CheckLog("cudaq/qaoa_nelder_mead")
    var g = page_graph()
    var h = hamiltonian(g)
    var mc = g.max_cut()
    print(g, " max cut =", mc[0])
    var ansatz = kernel(g)
    print("kernel:", len(ansatz), "gates, params", ansatz.param_names_str())

    # 1. observe() at the page's optimum.
    var eng = ExpectationEngine(ansatz, h)
    var page_opt: List[Float64] = [0.51349181993727, -0.21299416361632417, 0.3250526425808945, 0.886630847343767]
    var e_page = eng.value(page_opt)
    print("<H> at the page's optimal parameters:", e_page, "(page:", PAGE_OPT_ENERGY, ")")
    _ = log.close("<H> at page optimum equals page value", e_page, PAGE_OPT_ENERGY, 1e-9)

    # 2. The page's 1000-shot counts vs exact probabilities at those parameters.
    var probs = evolve(ansatz, page_opt).probabilities()
    var page_keys: List[String] = ["10101", "01011", "10100", "01010"]
    var page_counts: List[Int] = [165, 156, 155, 153]
    for k in range(4):
        var p = probs[cudaq_index(page_keys[k])]
        var sd = sqrt(p * (1.0 - p) / 1000.0)
        print("  ", page_keys[k], "(cut", g.cut_value(cudaq_index(page_keys[k])), "): exact P =", p,
              " page", page_counts[k], "/ 1000")
        _ = log.close("page count of " + page_keys[k] + " within 4 sigma", Float64(page_counts[k]) / 1000.0, p, 4.0 * sd)

    # 3. Nelder-Mead from the page's initial point, then 5 seeded restarts.
    var x_page: List[Float64] = [0.21810696323572243, -0.20613464375211488, 0.2546877639814583, 0.3657985647468064]
    var nm = NelderMead(maxiter=4000)
    var obj = VQEObjective(ansatz, h)
    var best = nm.minimize(obj, x_page)
    print("Nelder-Mead from the page's initial point: <H> =", best.fun, " nfev", best.nfev)
    var rng = Rng(13)
    for k in range(5):
        var r = rng.split(k)
        var x0 = List[Float64]()
        for _ in range(2 * LAYERS):
            x0.append((2.0 * r.uniform() - 1.0) * pi / 8.0)  # uniform in [-pi/8, pi/8], as on the page
        var res = nm.minimize(obj, x0)
        print("  restart", k, ": <H> =", res.fun, " nfev", res.nfev)
        if res.fun < best.fun:
            best = res^
    print("best <H> =", best.fun, " at", best.x[0], best.x[1], best.x[2], best.x[3])
    _ = log.within("best <H> (page -4.496, minimum -5)", best.fun, -5.0, -4.49)

    # 4. sample() the optimised kernel.
    var meas = ansatz.copy()
    meas.measure_all()
    var res = sample(meas, SHOTS, seed=13, params=best.x)
    var top = res.most_frequent()
    var cut = g.cut_value(bitstring_to_int(top))
    print("top bitstring (CUDA-Q order):", cudaq_order(top), " count", res.counts.get(top, 0), "/", SHOTS, " cut", cut)
    var p_opt = 0.0
    var probs_best = evolve(ansatz, best.x).probabilities()
    for z in range(1 << N):
        if g.cut_value(z) == mc[0]:
            p_opt += probs_best[z]
    print("exact P(optimal cut) at the best parameters:", p_opt)
    _ = log.check("most frequent sample is an optimal cut", cut == mc[0],
                  "top=" + cudaq_order(top) + " cut=" + String(cut) + " max=" + String(mc[0]))
    log.finish()
