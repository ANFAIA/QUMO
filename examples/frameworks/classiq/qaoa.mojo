"""QAOA for Max-Cut and a constrained knapsack, exact on QUMO.

Reproduces: Classiq library - Quantum Approximate Optimization Algorithm
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/search_and_optimization/QAOA/qaoa.ipynb (Classiq Library, MIT)

The notebook builds QAOA circuits (H on every qubit, then alternating cost and RX mixer layers). Max-Cut on a 5-node, 6-edge graph uses 4 layers with cost phase exp(i gamma cut / |E|). The knapsack 2a + 3b <= 12, value 3a + 5b, a in 0..7, b in 0..3, uses 3 layers with phase exp(-i gamma value) on the feasible states only. The notebook tunes the angles with COBYLA on shot-sampled cost estimates and samples the result. This file runs the same circuits on QUMO's state vector. It checks the brute-force optima (cut 5 on the four states the notebook prints, knapsack (3, 2) with value 19), the cost layers' phases against the analytic formulas on all 32 basis states, the exact distributions at the notebook's printed angles against its printed frequencies within 4 binomial standard errors, and a Nelder-Mead run from the notebook's starting angles against the uniform-state baselines and the brute-force optima.

Deviations from the notebook:
- The Max-Cut cost layer is one RZZ(gamma / |E|) per edge. This is the notebook's phase up to a global phase, and the phase check covers all 32 basis states. Classiq compiles its own phase primitive.
- The knapsack feasibility test is not computed into an ancilla by quantum arithmetic. The same diagonal phase is applied as one multi-controlled phase per feasible basis state with nonzero value (17 gates per layer). The unitary on the five data qubits is the same. Qubits 0..2 hold a and qubits 3..4 hold b.
- The optimiser is QUMO's Nelder-Mead (derivative-free, at most 60 iterations, initial step 1.0) on exact expectations from the state vector. The notebook uses COBYLA with at most 60 iterations on shot-sampled estimates (1000 shots for the knapsack; the Max-Cut estimates pass no shot count). The budgets are not the same.
- The notebook's Max-Cut frequencies are counts divided by NUM_SHOTS = 1000, but its Max-Cut sample did not pass num_shots, and the ten printed values sum to 1.428. The Max-Cut frequency check therefore conditions on the cut >= 4 set (exactly the ten printed states) and compares the conditional frequencies with the printed counts (1428 in total). The knapsack sample passed num_shots = 1000, so its frequencies are compared directly.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/qaoa.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, pi, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.mc import append_mcp
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import bitstring_to_int
from qumo.algorithms.optimizers import NelderMead, Objective
from qumo.testing.checks import CheckLog


# ---------------------------------------------------------------------------
# Instances (basis index x: bit q is qubit q)
# ---------------------------------------------------------------------------
def bit(x: Int, q: Int) -> Int:
    return (x >> q) & 1


def mc_edges() -> List[Tuple[Int, Int]]:
    var out = List[Tuple[Int, Int]]()
    out.append((0, 1))
    out.append((0, 2))
    out.append((1, 2))
    out.append((1, 3))
    out.append((2, 4))
    out.append((3, 4))
    return out^


def cut_size(x: Int) -> Int:
    var c = 0
    for e in mc_edges():
        if bit(x, e[0]) != bit(x, e[1]):
            c += 1
    return c


def ks_a(x: Int) -> Int:
    return x & 7


def ks_b(x: Int) -> Int:
    return x >> 3


def ks_index(a: Int, b: Int) -> Int:
    return a + 8 * b


def ks_value(x: Int) -> Int:
    return 3 * ks_a(x) + 5 * ks_b(x)


def ks_feasible(x: Int) -> Bool:
    return 2 * ks_a(x) + 3 * ks_b(x) <= 12


# ---------------------------------------------------------------------------
# Circuit layers and the QAOA ansatz
# ---------------------------------------------------------------------------
def mc_cost_layer(mut c: Circuit, gamma: Float64):
    """Phase exp(i gamma cut(x) / |E|), up to a global phase: RZZ(gamma / |E|) on every edge."""
    var m = Float64(len(mc_edges()))
    for e in mc_edges():
        c.rzz(gamma / m, e[0], e[1])


def ks_cost_layer(mut c: Circuit, gamma: Float64) raises:
    """Phase exp(-i gamma value(x)) on each feasible x, identity elsewhere (the notebook's phase(-value, gamma) under the constraint)."""
    for x in range(32):
        if ks_feasible(x) and ks_value(x) > 0:
            for q in range(5):
                if bit(x, q) == 0:
                    c.x(q)
            var controls: List[Int] = [0, 1, 2, 3]
            append_mcp(c, -gamma * Float64(ks_value(x)), controls, 4)
            for q in range(5):
                if bit(x, q) == 0:
                    c.x(q)


def mixer(mut c: Circuit, beta: Float64):
    for q in range(5):
        c.rx(beta, q)


def maxcut_circuit(params: List[Float64]) -> Circuit:
    """Notebook layout: params = gammas (4 layers), then betas (4 layers)."""
    var layers = 4
    var c = Circuit(5, name="maxcut_qaoa")
    for q in range(5):
        c.h(q)
    for l in range(layers):
        mc_cost_layer(c, params[l])
        mixer(c, params[layers + l])
    return c^


def knapsack_circuit(params: List[Float64]) raises -> Circuit:
    """Notebook layout: params = gammas (3 layers), then betas (3 layers)."""
    var layers = 3
    var c = Circuit(5, name="knapsack_qaoa")
    for q in range(5):
        c.h(q)
    for l in range(layers):
        ks_cost_layer(c, params[l])
        mixer(c, params[layers + l])
    return c^


# ---------------------------------------------------------------------------
# Objective: exact expectation of a notebook cost over the QAOA state
# ---------------------------------------------------------------------------
def probabilities_for(kind: Int, params: List[Float64]) raises -> List[Float64]:
    """Basis-state probabilities of the QAOA state for instance `kind` (0 = Max-Cut, 1 = knapsack)."""
    if kind == 0:
        return evolve(maxcut_circuit(params)).probabilities()
    return evolve(knapsack_circuit(params)).probabilities()


struct QaoaObjective(Objective, Movable):
    var kind: Int  # 0 = Max-Cut, 1 = knapsack
    var cost: List[Float64]
    var nfev: Int

    def __init__(out self, kind: Int, var cost: List[Float64]):
        self.kind = kind
        self.cost = cost^
        self.nfev = 0

    def num_params(self) -> Int:
        if self.kind == 0:
            return 8
        return 6

    def value(mut self, x: List[Float64]) raises -> Float64:
        self.nfev += 1
        var pr = probabilities_for(self.kind, x)
        var e = 0.0
        for i in range(len(pr)):
            e += pr[i] * self.cost[i]
        return e

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        # Nelder-Mead never asks for gradients; forward differences keep the trait complete.
        var f0 = self.value(x)
        var xp = x.copy()
        var h = 1e-6
        for i in range(len(x)):
            xp[i] = x[i] + h
            grad[i] = (self.value(xp) - f0) / h
            xp[i] = x[i]
        return f0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def vec_to_x(v: List[Int]) -> Int:
    """Notebook solution vector (v[0], ..., v[4]) to basis index: v[q] is qubit q."""
    var x = 0
    for q in range(len(v)):
        x = x | (v[q] << q)
    return x


def vec_str(v: List[Int]) -> String:
    var s = String()
    for q in range(len(v)):
        s += String(v[q])
    return s


def has(xs: List[Int], x: Int) -> Bool:
    for y in xs:
        if y == x:
            return True
    return False


def subset(a: List[Int], b: List[Int]) -> Bool:
    for x in a:
        if not has(b, x):
            return False
    return True


def same_set(a: List[Int], b: List[Int]) -> Bool:
    return len(a) == len(b) and subset(a, b)


def mc_cut_cost() -> List[Float64]:
    """The notebook's maxcut_cost(x) = -cut(x) / |E| for every basis state."""
    var out = List[Float64](length=32, fill=0.0)
    for x in range(32):
        out[x] = -Float64(cut_size(x)) / Float64(len(mc_edges()))
    return out^


def ks_cost() -> List[Float64]:
    """The notebook's cost(v) = -value(v) if the constraint holds, else 0."""
    var out = List[Float64](length=32, fill=0.0)
    for x in range(32):
        if ks_feasible(x):
            out[x] = -Float64(ks_value(x))
    return out^


def main() raises:
    var log = CheckLog("classiq/qaoa")

    # Notebook output, transcribed: the Max-Cut angles and the ten printed solutions with their counts
    # (the printed frequency times 1000 is an exact integer count).
    var mc_params: List[Float64] = [
        1.2206977566740314, -0.3206285041700663, 0.3488171906457831, 1.047678282751786,
        0.9356852465725753, 1.8690223830791217, 0.09297188478970589, -0.8035120959658113,
    ]
    var mc_vecs = List[List[Int]]()
    mc_vecs.append([0, 1, 0, 0, 1])
    mc_vecs.append([1, 0, 1, 1, 0])
    mc_vecs.append([1, 1, 0, 0, 1])
    mc_vecs.append([0, 0, 1, 1, 0])
    mc_vecs.append([0, 1, 1, 0, 0])
    mc_vecs.append([1, 0, 0, 1, 1])
    mc_vecs.append([1, 0, 0, 0, 1])
    mc_vecs.append([1, 0, 0, 1, 0])
    mc_vecs.append([0, 1, 1, 1, 0])
    mc_vecs.append([0, 1, 1, 0, 1])
    var mc_counts: List[Int] = [205, 205, 198, 196, 118, 112, 111, 103, 94, 86]

    var ks_params: List[Float64] = [
        -0.4313245186863623, 3.101708740098853, 2.8757022490279174,
        4.24928949496719, 1.1565701772476515, -0.5427793230336472,
    ]
    var ks_ab = List[List[Int]]()
    ks_ab.append([3, 2])
    ks_ab.append([1, 3])
    ks_ab.append([6, 0])
    ks_ab.append([4, 1])
    ks_ab.append([2, 2])
    ks_ab.append([5, 0])
    ks_ab.append([0, 3])
    ks_ab.append([3, 1])
    ks_ab.append([1, 2])
    ks_ab.append([4, 0])
    var ks_counts: List[Int] = [27, 17, 10, 122, 6, 65, 1, 93, 17, 1]

    # ---- Max-Cut: brute force and structure of the notebook's output ----
    var max_cut = 0
    var opt_states = List[Int]()
    var ge4 = List[Int]()
    for x in range(32):
        var c = cut_size(x)
        max_cut = max(max_cut, c)
        if c >= 4:
            ge4.append(x)
    for x in range(32):
        if cut_size(x) == max_cut:
            opt_states.append(x)
    var mc_xs = List[Int]()
    for v in mc_vecs:
        mc_xs.append(vec_to_x(v))
    var mc_opt_printed = List[Int]()
    for i in range(4):
        mc_opt_printed.append(mc_xs[i])
    print("Max-Cut: brute-force optimum", max_cut, "with", len(opt_states), "optimal partitions")
    _ = log.check("Max-Cut brute-force optimum is 5", max_cut == 5, "max=" + String(max_cut))
    _ = log.check("Max-Cut optimal set is the notebook's four solutions", same_set(opt_states, mc_opt_printed))
    _ = log.check("cut >= 4 set is exactly the notebook's ten printed solutions", len(ge4) == 10 and subset(mc_xs, ge4) and subset(ge4, mc_xs))

    # ---- Max-Cut: cost layer phases against exp(i gamma cut / |E|) on all 32 states ----
    var g1 = mc_params[0]
    var probe = Circuit(5, name="maxcut_cost_probe")
    for q in range(5):
        probe.h(q)
    mc_cost_layer(probe, g1)
    var sv_probe = evolve(probe)
    var dev_mc = 0.0
    for x in range(32):
        var d = g1 * Float64(cut_size(x)) / Float64(len(mc_edges()))
        # amplitude_x * conj(amplitude_0) = (1/32) exp(i (phase_x - phase_0))
        var rr = sv_probe.re[x] * sv_probe.re[0] + sv_probe.im[x] * sv_probe.im[0]
        var ii = sv_probe.im[x] * sv_probe.re[0] - sv_probe.re[x] * sv_probe.im[0]
        dev_mc = max(dev_mc, abs(rr - cos(d) / 32.0))
        dev_mc = max(dev_mc, abs(ii - sin(d) / 32.0))
    _ = log.close("Max-Cut cost layer phases match exp(i gamma cut / |E|) on all 32 states", dev_mc, 0.0, 1e-12)

    # ---- Max-Cut: exact distribution at the notebook's angles vs its printed counts ----
    var pr_mc = evolve(maxcut_circuit(mc_params)).probabilities()
    var mass = 0.0
    for x in ge4:
        mass += pr_mc[x]
    var total = 0
    for k in mc_counts:
        total += k
    for i in range(len(mc_vecs)):
        var x = mc_xs[i]
        var r = pr_mc[x] / mass
        var q = Float64(mc_counts[i]) / Float64(total)
        var tol = 4.0 * sqrt(r * (1.0 - r) / Float64(total))
        _ = log.close("Max-Cut " + vec_str(mc_vecs[i]) + " share of cut>=4 counts", r, q, tol)

    # ---- Max-Cut: Nelder-Mead from the notebook's linear-ramp start, 60 iterations ----
    var mc_obj = QaoaObjective(0, mc_cut_cost())
    var x0_mc: List[Float64] = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]
    var opt_mc = NelderMead(maxiter=60, initial_step=1.0).minimize(mc_obj, x0_mc)
    var pr_opt = evolve(maxcut_circuit(opt_mc.x)).probabilities()
    var exp_cut = 0.0
    var p_opt = 0.0
    var uniform_cut = 0.0
    for x in range(32):
        exp_cut += pr_opt[x] * Float64(cut_size(x))
        uniform_cut += Float64(cut_size(x)) / 32.0
        if cut_size(x) == max_cut:
            p_opt += pr_opt[x]
    var uniform_opt = Float64(len(opt_states)) / 32.0
    print("Max-Cut optimised: <cut> =", exp_cut, " P(optimal) =", p_opt, " nfev =", mc_obj.nfev)
    _ = log.check("Max-Cut optimised <cut> beats the uniform state", exp_cut > uniform_cut, "exp_cut=" + String(exp_cut) + " uniform=" + String(uniform_cut))
    _ = log.check("Max-Cut optimised P(optimal cut) beats the uniform state", p_opt > uniform_opt, "p_opt=" + String(p_opt) + " uniform=" + String(uniform_opt))
    var mc_meas = maxcut_circuit(opt_mc.x)
    mc_meas.measure_all()
    var mc_res = sample(mc_meas, 1000, UInt64(7))
    var best_sampled_cut = 0
    for kv in mc_res.counts.items():
        best_sampled_cut = max(best_sampled_cut, cut_size(bitstring_to_int(kv.key)))
    _ = log.check("Max-Cut best of 1000 samples is the optimum", best_sampled_cut == max_cut, "best=" + String(best_sampled_cut))

    # ---- Knapsack: brute force and structure of the notebook's output ----
    var ks_best = -1
    var ks_best_x = -1
    var ks_n_best = 0
    var ks_top = List[Int]()  # feasible states with value >= 12
    for x in range(32):
        if ks_feasible(x):
            var v = ks_value(x)
            if v > ks_best:
                ks_best = v
                ks_best_x = x
                ks_n_best = 1
            elif v == ks_best:
                ks_n_best += 1
            if v >= 12:
                ks_top.append(x)
    var ks_xs = List[Int]()
    for ab in ks_ab:
        ks_xs.append(ks_index(ab[0], ab[1]))
    print("Knapsack: brute-force optimum value", ks_best, "at a =", ks_a(ks_best_x), "b =", ks_b(ks_best_x))
    _ = log.check("Knapsack brute-force optimum is (3, 2) with value 19, unique", ks_best == 19 and ks_best_x == ks_index(3, 2) and ks_n_best == 1)
    _ = log.check("printed knapsack states are the ten feasible states with value >= 12", same_set(ks_xs, ks_top))

    # ---- Knapsack: cost layer phases against exp(-i gamma value) on feasible states ----
    var kprobe = Circuit(5, name="knapsack_cost_probe")
    for q in range(5):
        kprobe.h(q)
    ks_cost_layer(kprobe, ks_params[0])
    var sv_k = evolve(kprobe)
    var dev_ks = 0.0
    for x in range(32):
        var d = 0.0
        if ks_feasible(x):
            d = -ks_params[0] * Float64(ks_value(x))
        var rr = sv_k.re[x] * sv_k.re[0] + sv_k.im[x] * sv_k.im[0]
        var ii = sv_k.im[x] * sv_k.re[0] - sv_k.re[x] * sv_k.im[0]
        dev_ks = max(dev_ks, abs(rr - cos(d) / 32.0))
        dev_ks = max(dev_ks, abs(ii - sin(d) / 32.0))
    _ = log.close("knapsack cost layer phases match exp(-i gamma value) on feasible states", dev_ks, 0.0, 1e-12)

    # ---- Knapsack: exact distribution at the notebook's angles vs its printed frequencies (1000 shots) ----
    var pr_ks = evolve(knapsack_circuit(ks_params)).probabilities()
    for i in range(len(ks_ab)):
        var x = ks_xs[i]
        var p = pr_ks[x]
        var tol = 4.0 * sqrt(p * (1.0 - p) / 1000.0)
        var printed = Float64(ks_counts[i]) / 1000.0
        _ = log.close("Knapsack a=" + String(ks_ab[i][0]) + " b=" + String(ks_ab[i][1]) + " frequency", p, printed, tol)

    # ---- Knapsack: Nelder-Mead from the notebook's pi-scaled linear start, 60 iterations ----
    var ks_cost_v = ks_cost()
    var uniform_cost = 0.0
    for x in range(32):
        uniform_cost += ks_cost_v[x] / 32.0
    var ks_obj = QaoaObjective(1, ks_cost_v.copy())
    var x0_ks: List[Float64] = [0.0, pi / 2.0, pi, pi, pi / 2.0, 0.0]
    var opt_ks = NelderMead(maxiter=60, initial_step=1.0).minimize(ks_obj, x0_ks)
    var pr_ko = evolve(knapsack_circuit(opt_ks.x)).probabilities()
    var exp_cost = 0.0
    for x in range(32):
        exp_cost += pr_ko[x] * ks_cost_v[x]
    print("Knapsack optimised: <cost> =", exp_cost, " uniform <cost> =", uniform_cost, " nfev =", ks_obj.nfev)
    _ = log.check("Knapsack optimised <cost> beats the uniform state", exp_cost < uniform_cost, "exp_cost=" + String(exp_cost) + " uniform=" + String(uniform_cost))
    var ks_meas = knapsack_circuit(opt_ks.x)
    ks_meas.measure_all()
    var ks_res = sample(ks_meas, 1000, UInt64(7))
    var best_sampled_value = -1
    for kv in ks_res.counts.items():
        var x = bitstring_to_int(kv.key)
        if ks_feasible(x):
            best_sampled_value = max(best_sampled_value, ks_value(x))
    _ = log.check("Knapsack best feasible value of 1000 samples is the optimum 19", best_sampled_value == ks_best, "best=" + String(best_sampled_value))

    log.finish()
