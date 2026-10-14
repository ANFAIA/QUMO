"""QAOA MaxCut with Estimator training and Sampler readout (Qiskit tutorial instance).

Reproduces: Qiskit - Quantum approximate optimization algorithm (small-scale part)
Source: https://quantum.cloud.ibm.com/docs/tutorials/quantum-approximate-optimization-algorithm (IBM Quantum documentation; tutorial code Apache-2.0, text CC BY-SA 4.0)

The tutorial's pattern - map, optimize, execute, post-process - on its
5-node graph with edges (0,1) (0,2) (0,4) (1,2) (2,3) (3,4) (all weight 1)
and p = 2 QAOA layers (4 parameters):

1. Map: `qaoa_maxcut_ansatz(graph, 2)` (QUMO's QAOAAnsatz) and the MaxCut
   Hamiltonian H = sum_(u,v) (Z_u Z_v - 1)/2, so <H> = -<cut>.
2. Optimize: every objective evaluation is one exact `EstimatorPub`. The page
   uses COBYLA, which QUMO does not ship; the derivative-free NelderMead
   plays its role, started from the page's initial point (gamma = pi,
   beta = pi/2 in qiskit's convention; QUMO's RZZ(gamma) convention doubles
   gamma). BFGS with adjoint gradients is shown for comparison.
3. Execute: one `SamplerPub` with 10,000 shots at the optimum.
4. Post-process: the most frequent bitstring and its cut value, compared
   with brute force (the page reports the best cut 5, bitstring
   [0, 0, 1, 0, 1] in node order = QUMO key "10100", or its complement).

The page's second part (100 qubits, p = 1, edges from a vendor's coupling
map) cannot be simulated exactly; it is replaced by `estimate_resources` and
`estimate_hardware` for a neutral 10x10 grid graph, for every modality preset.

Reference numbers:
- best cut 5 and bitstring [0, 0, 1, 0, 1] (page);
- <sum ZZ> = -5.898e-17 (i.e. 0) at the tutorial's initial point and 2.9777060466916727
  at (g0, g1, b0, b1) = (0.7, 1.3, 0.4, 0.25) (rerun qiskit 2.5.2, reference/qaoa_ref.py).

Checks:
- Graph.max_cut() == 5 (page);
- the most frequent sampled bitstring is an optimal cut;
- approximation ratio <cut>/5 >= 0.8 (value recorded);
- the Estimator value equals the state-vector expectation (1e-12);
- QUMO equals qiskit at the two committed points (1e-10);
- NelderMead and BFGS+adjoint reach the same optimum value (1e-6).

Run: .venv/bin/mojo run -I . examples/frameworks/qiskit/qaoa_maxcut.mojo
example-meta: {"framework": "qiskit", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import pi
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import evolve
from qumo.exec.primitives import EstimatorPub, run_estimator, SamplerPub, run_sampler
from qumo.exec.resources import estimate_resources, estimate_hardware, HardwareProfile
from qumo.algorithms.hamiltonians import Graph, maxcut
from qumo.algorithms.library import qaoa_maxcut_ansatz, bitstring_to_int
from qumo.algorithms.optimizers import Objective, NelderMead
from qumo.algorithms.gradients import gradient, Grad
from qumo.algorithms.variational import vqe
from qumo.testing.checks import CheckLog

comptime SHOTS = 10000
comptime P = 2


struct EstimatorObjective(Objective, Movable):
    """<H> through one exact Estimator PUB per evaluation (the tutorial's cost_func_estimator)."""

    var ansatz: Circuit
    var h: PauliSum
    var evals: Int

    def __init__(out self, ansatz: Circuit, h: PauliSum):
        self.ansatz = ansatz.copy()
        self.h = h.copy()
        self.evals = 0

    def num_params(self) -> Int:
        return self.ansatz.num_params()

    def value(mut self, x: List[Float64]) raises -> Float64:
        self.evals += 1
        var obs = List[PauliSum]()
        obs.append(self.h.copy())
        var sets = List[List[Float64]]()
        sets.append(x.copy())
        return run_estimator(EstimatorPub(self.ansatz, obs, sets, 0.0)).evs[0][0]

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        grad = gradient(self.ansatz, self.h, x, Grad.ADJOINT)
        return self.value(x)


def tutorial_graph() raises -> Graph:
    var g = Graph(5)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 4)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    return g^


def sum_zz(e_maxcut: Float64, num_edges: Int) -> Float64:
    """The tutorial's objective <sum ZZ> from QUMO's <H> = <sum (ZZ - 1)/2>."""
    return 2.0 * e_maxcut + Float64(num_edges)


def node_order(key: String) -> String:
    """QUMO key (leftmost = node 4) -> the page's node-order list [x0, ..., x4]."""
    var bs = key.as_bytes()
    var s = String("[")
    var i = key.byte_length() - 1
    while i >= 0:
        s += "1" if Int(bs[i]) == ord("1") else "0"
        if i > 0:
            s += ", "
        i -= 1
    return s + "]"


def main() raises:
    var log = CheckLog("qiskit/qaoa_maxcut")
    var g = tutorial_graph()
    var ne = len(g.edges)
    var best = g.max_cut()
    print(g, " brute-force max cut =", best[0])
    _ = log.close("brute-force max cut (page)", best[0], 5.0, 0.0)

    var ansatz = qaoa_maxcut_ansatz(g, P)
    var h = maxcut(g)
    print("ansatz:", ansatz.num_qubits, "qubits,", len(ansatz), "gates, params", ansatz.param_names_str())

    # Cross-check against qiskit (committed reference values, see reference/qaoa_ref.py).
    var pts = List[List[Float64]]()
    pts.append([2.0 * pi, 2.0 * pi, pi / 2.0, pi / 2.0])
    pts.append([0.7, 1.3, 0.4, 0.25])
    var qiskit_vals: List[Float64] = [-5.898059818321144e-17, 2.9777060466916727]
    var names: List[String] = ["tutorial initial point", "arbitrary point"]
    for i in range(2):
        var e = evolve(ansatz, pts[i]).expectation(h)
        print("  <sum ZZ> at", names[i], "=", sum_zz(e, ne), "(qiskit 2.5.2:", qiskit_vals[i], ")")
        _ = log.close("<sum ZZ> at " + names[i] + " equals qiskit", sum_zz(e, ne), qiskit_vals[i], 1e-10)

    # Optimize: NelderMead over exact Estimator PUBs (COBYLA stand-in), from the page's initial point.
    var x0 = pts[0].copy()
    var obj = EstimatorObjective(ansatz, h)
    var nm = NelderMead(maxiter=4000, initial_step=0.4)
    var res = nm.minimize(obj, x0)
    print("NelderMead:", res, " estimator PUBs =", obj.evals)
    # A local optimiser can stall from the degenerate tutorial start; restart from a few seeded points.
    var starts = List[List[Float64]]()
    starts.append([0.8, 1.6, 0.6, 0.3])
    starts.append([1.2, 0.9, 0.9, 0.5])
    starts.append([0.5, 1.0, 0.3, 0.6])
    var xbest = res.x.copy()
    var fbest = res.fun
    for s in starts:
        var r = nm.minimize(obj, s)
        if r.fun < fbest:
            fbest = r.fun
            xbest = r.x.copy()
    print("best NelderMead <H> =", fbest, " expected cut =", -fbest, " x =", xbest[0], xbest[1], xbest[2], xbest[3],
          " total estimator PUBs =", obj.evals)
    # BFGS + adjoint gradients from the same start points (2 circuit runs per iteration).
    var bfgs_best = vqe(ansatz, h, "bfgs", x0).energy
    var bfgs_evals = 0
    for s in starts:
        var vr = vqe(ansatz, h, "bfgs", s)
        bfgs_evals += vr.circuit_evals
        print("BFGS + adjoint from", s[0], s[1], s[2], s[3], "->", vr.energy, " nit =", vr.nit, " circuit evaluations =", vr.circuit_evals)
        bfgs_best = min(bfgs_best, vr.energy)
    print("best BFGS <H> =", bfgs_best, " (NelderMead needed", obj.evals, "estimator PUBs; BFGS", bfgs_evals, "circuit evaluations)")
    _ = log.close("NelderMead and BFGS agree on the optimum", fbest, bfgs_best, 1e-6)

    # Estimator vs state vector at the optimum.
    var obs = List[PauliSum]()
    obs.append(h.copy())
    var sets = List[List[Float64]]()
    sets.append(xbest.copy())
    var est = run_estimator(EstimatorPub(ansatz, obs, sets, 0.0)).evs[0][0]
    var sv = evolve(ansatz, xbest).expectation(h)
    _ = log.close("Estimator equals statevector at the optimum", est, sv, 1e-12)
    var ratio = -est / best[0]
    print("approximation ratio <cut>/max_cut =", ratio, " (tutorial objective <sum ZZ> =", sum_zz(est, ne), ")")
    _ = log.within("approximation ratio", ratio, 0.8, 1.0)
    print("qiskit cross-check of this optimum: bench/.venv/bin/python examples/frameworks/qiskit/reference/qaoa_ref.py --qumo-angles",
          xbest[0], xbest[1], xbest[2], xbest[3], "--qumo-value", sum_zz(est, ne))

    # Execute: one Sampler PUB, 10k shots.
    var measured = ansatz.copy()
    measured.measure_all()
    var samples = run_sampler(SamplerPub(measured, SHOTS, sets, seed=7))
    ref r0 = samples[0]
    var top = r0.most_frequent()
    var top_cut = g.cut_value(bitstring_to_int(top))
    var p_opt = 0.0
    for kv in r0.counts.items():
        if g.cut_value(bitstring_to_int(kv.key)) == best[0]:
            p_opt += Float64(kv.value) / Float64(SHOTS)
    print("sampler:", r0.method, " most frequent", top, "= nodes", node_order(top), " cut =", top_cut,
          " P(optimal cut) =", p_opt)
    _ = log.check("most frequent bitstring is an optimal cut", top_cut == best[0],
                  "key=" + top + " cut=" + String(top_cut))

    # The page's 100-qubit part: resource estimates instead of a simulation (neutral 10x10 grid graph).
    var big = Graph(100)
    for r in range(10):
        for c in range(10):
            var v = 10 * r + c
            if c + 1 < 10:
                big.add_edge(v, v + 1)
            if r + 1 < 10:
                big.add_edge(v, v + 10)
    # p = 1 circuit written out (qaoa_ansatz caps n at 64): H layer, RZZ(gamma) per edge, RX(2 beta)
    var bc = Circuit(100, name="qaoa_grid100")
    for q in range(100):
        bc.h(q)
    for e in big.edges:
        bc.rzz(0.6, e.u, e.v)
    for q in range(100):
        bc.rx(2.0 * 0.4, q)
    var re = estimate_resources(bc)
    print("100-node grid QAOA p=1:", len(big.edges), "edges,", re.gates_total, "gates,", re.gates_2q, "2q gates, depth", re.depth,
          "-> recommended", re.recommended_method, "(", re.reason, ")")
    var profiles = List[HardwareProfile]()
    profiles.append(HardwareProfile.superconducting())
    profiles.append(HardwareProfile.trapped_ion())
    profiles.append(HardwareProfile.neutral_atom())
    for prof in profiles:
        var he = estimate_hardware(bc, prof)
        print("  ", prof.name, ": duration", he.duration_us, "us, estimated success probability", he.success_probability,
              "(all-to-all connectivity assumed; routing not included)")
    _ = log.check("100-qubit instance is beyond exact simulation", re.recommended_method == "hardware_or_tensor_network",
                  "recommended=" + re.recommended_method)
    log.finish()
