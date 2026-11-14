"""Minimum graph colouring with QAOA on a QUBO, checked against brute force on QUMO.

Reproduces: Classiq library - Min Graph Coloring Problem
Framework: QUMO - https://github.com/ANFAIA/QUMO
Source: https://github.com/Classiq/classiq-library/blob/main/applications/optimization/min_graph_coloring/min_graph_coloring.ipynb (Classiq Library, MIT)

The notebook colours the graph nx.erdos_renyi_graph(5, 0.3, seed=79) with a one-hot matrix x[color][vertex] (3 colours, the maximum degree, so 15 binary variables). Each vertex gets exactly one colour, adjacent vertices must differ, and the objective is the number of colours in use. Classiq's CombinatorialProblem turns the Pyomo model into a 6-layer CVaR-QAOA (penalty factor 10, COBYLA, quantile 0.7) and prints the five lowest-cost samples (cost 3 each) and the best colouring [[1,0,1,0,0],[0,1,0,0,0],[0,0,0,1,1]].
This file builds the same problem as a QUBO, converts it to an Ising Hamiltonian, runs QAOA on QUMO's state vector and checks it against enumeration. It checks: the graph (edges 0-1, 0-3, 0-4, 1-3, 2-3, rebuilt with Python's random.Random(79) as networkx does) has maximum degree 3 and admits the notebook's printed best colouring; an independent enumeration of all 3^5 colourings gives chromatic number 3 (no proper 2-colouring); the QUBO ground energy found by qubo_brute_force is 3, matching the notebook's best cost; the notebook's colouring has QUBO energy 3; the number of QUBO ground states equals the independently counted proper colourings that use all three colours; the Ising Hamiltonian reproduces the QUBO energy on basis states; and QAOA at fixed depth lowers the expected cost and raises the ground-state probability above the uniform state.

Deviations from the notebook:
- The notebook's objective is the degree-5 polynomial sum_c (1 - prod_v (1 - x[c][v])). QUMO's Qubo is quadratic, so one auxiliary variable y[c] per colour ("colour c is used") replaces it. The objective is sum_c y[c], with penalty 10 on x[c][v](1 - y[c]). This adds 3 qubits, so the QAOA register has 18 qubits instead of 15. The ground energy is still the number of colours (3).
- The penalty form is written by hand: 10 (sum_c x[c][v] - 1)^2 per vertex and 10 x[c][u] x[c][v] per edge and colour. Classiq's exact penalty expansion is internal to CombinatorialProblem.
- The graph is rebuilt from Python's random.Random(79) following networkx's gnp_random_graph rule (draw random() for each pair in lexicographic order, keep the edge if below 0.3), because networkx is not available here. The edges are hard-coded and the checks (maximum degree 3, notebook colouring is proper, notebook size of 15 variables) confirm that it is the notebook's graph.
- QAOA uses 3 layers (not 6) and QUMO's Nelder-Mead (at most 60 iterations) on exact state-vector expectations, with no CVaR; the notebook uses 6 layers, COBYLA at most 100 iterations and CVaR with quantile 0.7 on shot samples. The notebook's cost trace, histogram and sample probabilities are not compared, since they depend on its shots and optimiser trajectory.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/min_graph_coloring.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import qaoa_ansatz, bitstring_to_int
from qumo.algorithms.qubo import Qubo, qubo_energy, qubo_to_ising, qubo_brute_force, qubo_energy_of_ising
from qumo.algorithms.optimizers import NelderMead, Objective
from qumo.testing.checks import CheckLog

comptime NV = 5
comptime NC = 3
comptime NX = 15  # x[c][v] is variable c * NV + v
comptime NQ = 18  # then y[c] is variable NX + c
comptime PENALTY = 10.0


def edges() -> List[Tuple[Int, Int]]:
    var out = List[Tuple[Int, Int]]()
    out.append((0, 1))
    out.append((0, 3))
    out.append((0, 4))
    out.append((1, 3))
    out.append((2, 3))
    return out^


def xv(c: Int, v: Int) -> Int:
    return c * NV + v


def build_qubo() raises -> Qubo:
    var q = Qubo(NQ)
    # each vertex has exactly one colour: P (sum_c x - 1)^2 = P (1 - sum_c x + 2 sum_{c<c'} x x')
    for v in range(NV):
        q.offset += PENALTY
        for c in range(NC):
            q.add_linear(xv(c, v), -PENALTY)
        for c in range(NC):
            for d in range(c + 1, NC):
                q.add_quadratic(xv(c, v), xv(d, v), 2.0 * PENALTY)
    # adjacent vertices differ
    for e in edges():
        for c in range(NC):
            q.add_quadratic(xv(c, e[0]), xv(c, e[1]), PENALTY)
    # y[c] is 1 whenever colour c is used: P x (1 - y); objective sum_c y[c]
    for c in range(NC):
        q.add_linear(NX + c, 1.0)
        for v in range(NV):
            q.add_linear(xv(c, v), PENALTY)
            q.add_quadratic(xv(c, v), NX + c, -PENALTY)
    return q^


struct QaoaObjective(Objective, Movable):
    var circuit: Circuit
    var cost: PauliSum
    var nfev: Int
    var np: Int

    def __init__(out self, var circuit: Circuit, var cost: PauliSum, np: Int):
        self.circuit = circuit^
        self.cost = cost^
        self.nfev = 0
        self.np = np

    def num_params(self) -> Int:
        return self.np

    def value(mut self, x: List[Float64]) raises -> Float64:
        self.nfev += 1
        return evolve(self.circuit, x).expectation(self.cost)

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        var f0 = self.value(x)
        var xp = x.copy()
        for i in range(len(x)):
            xp[i] = x[i] + 1e-6
            grad[i] = (self.value(xp) - f0) / 1e-6
            xp[i] = x[i]
        return f0


def main() raises:
    var log = CheckLog("classiq/min_graph_coloring")
    var ed = edges()

    # ---- Graph: degrees and the notebook's printed best colouring ----
    var deg = List[Int](length=NV, fill=0)
    for e in ed:
        deg[e[0]] += 1
        deg[e[1]] += 1
    var dmax = 0
    for v in range(NV):
        dmax = max(dmax, deg[v])
    _ = log.check("maximum degree is 3, so the notebook uses 3 colours and 15 variables", dmax == NC and NC * NV == NX)
    # notebook best solution rows (colour 0, 1, 2) over vertices 0..4
    var nb = List[List[Int]]()
    nb.append([1, 0, 1, 0, 0])
    nb.append([0, 1, 0, 0, 0])
    nb.append([0, 0, 0, 1, 1])
    var nb_col = List[Int](length=NV, fill=-1)
    var one_hot_ok = True
    for v in range(NV):
        var s = 0
        for c in range(NC):
            s += nb[c][v]
            if nb[c][v] == 1:
                nb_col[v] = c
        if s != 1:
            one_hot_ok = False
    var proper = True
    for e in ed:
        if nb_col[e[0]] == nb_col[e[1]]:
            proper = False
    _ = log.check("notebook's best colouring is one-hot and proper on the rebuilt graph", one_hot_ok and proper)

    # ---- Independent enumeration of all 3^5 colourings ----
    var n_proper = 0
    var n_proper_all3 = 0
    var proper_2 = 0
    for code in range(243):
        var col = List[Int]()
        var t = code
        for _ in range(NV):
            col.append(t % 3)
            t //= 3
        var ok = True
        for e in ed:
            if col[e[0]] == col[e[1]]:
                ok = False
        if ok:
            n_proper += 1
            var used = List[Int](length=NC, fill=0)
            for v in range(NV):
                used[col[v]] = 1
            if used[0] + used[1] + used[2] == 3:
                n_proper_all3 += 1
            if used[0] + used[1] + used[2] <= 2:
                proper_2 += 1
    print("proper 3-colourings:", n_proper, " using all three colours:", n_proper_all3, " using at most two:", proper_2)
    _ = log.check("chromatic number is 3 (proper colourings exist, none with two colours)", n_proper > 0 and proper_2 == 0)

    # ---- QUBO ground state ----
    var q = build_qubo()
    var gs = qubo_brute_force(q)
    print("QUBO ground energy:", gs.energy, " mask:", gs.mask)
    _ = log.close("QUBO ground energy equals the notebook's best cost 3", gs.energy, 3.0, 1e-9)
    var nb_mask: UInt64 = 0
    for c in range(NC):
        for v in range(NV):
            if nb[c][v] == 1:
                nb_mask |= UInt64(1) << UInt64(xv(c, v))
        nb_mask |= UInt64(1) << UInt64(NX + c)
    _ = log.close("notebook's best colouring has QUBO energy 3", qubo_energy(q, nb_mask), 3.0, 1e-9)

    # enumerate all 2^18 assignments
    var n_ground = 0
    var sum_e = 0.0
    var energies = List[Float64](length=1 << NQ, fill=0.0)
    for m in range(1 << NQ):
        var e = qubo_energy(q, UInt64(m))
        energies[m] = e
        sum_e += e
        if abs(e - 3.0) < 1e-9:
            n_ground += 1
    var uniform_e = sum_e / Float64(1 << NQ)
    print("QUBO ground states:", n_ground, " uniform mean energy:", uniform_e)
    _ = log.check("number of QUBO ground states equals proper colourings using all three colours", n_ground == n_proper_all3, "ground=" + String(n_ground) + " expected=" + String(n_proper_all3))

    # ---- Ising conversion on basis states ----
    var h = qubo_to_ising(q)
    var worst = 0.0
    for m in [0, 1, 777, 12345, 99999, 262143, Int(nb_mask), Int(gs.mask)]:
        worst = max(worst, abs(qubo_energy_of_ising(h, UInt64(m)) - energies[m]))
    _ = log.close("Ising Hamiltonian reproduces the QUBO energy on basis states", worst, 0.0, 1e-9)
    var probe = Circuit(NQ, name="basis_probe")
    for qb in range(NQ):
        if (nb_mask >> UInt64(qb)) & 1 == 1:
            probe.x(qb)
    _ = log.close("state-vector <H> on the notebook colouring is 3", evolve(probe).expectation(h), 3.0, 1e-9)

    # ---- QAOA, 3 layers, Nelder-Mead on exact expectations ----
    var layers = 3
    var circ = qaoa_ansatz(h, layers)
    var obj = QaoaObjective(circ.copy(), h.copy(), 2 * layers)
    var x0: List[Float64] = [0.02, 0.02, 0.02, 0.4, 0.4, 0.4]
    var f_start = obj.value(x0)
    var opt = NelderMead(maxiter=60, initial_step=0.05).minimize(obj, x0)
    var sv = evolve(circ, opt.x)
    var pr = sv.probabilities()
    var exp_e = 0.0
    var p_ground = 0.0
    for m in range(1 << NQ):
        exp_e += pr[m] * energies[m]
        if abs(energies[m] - 3.0) < 1e-9:
            p_ground += pr[m]
    var p_uniform = Float64(n_ground) / Float64(1 << NQ)
    print("QAOA: start <E> =", f_start, " optimised <E> =", exp_e, " uniform <E> =", uniform_e, " P(ground) =", p_ground, " uniform P =", p_uniform, " nfev =", obj.nfev)
    _ = log.close("<H> from the optimiser equals the enumerated expectation", opt.fun, exp_e, 1e-6)
    _ = log.check("optimised <E> is below the uniform state's", exp_e < uniform_e, "exp=" + String(exp_e) + " uniform=" + String(uniform_e))
    _ = log.check("optimised P(ground state) beats the uniform state", p_ground > p_uniform, "p=" + String(p_ground) + " uniform=" + String(p_uniform))
    var meas = circ.copy()
    meas.measure_all()
    var res = sample(meas, 2000, UInt64(7), opt.x)
    var best_sampled = 1.0e9
    for kv in res.counts.items():
        best_sampled = min(best_sampled, energies[bitstring_to_int(kv.key)])
    print("best energy among 2000 samples:", best_sampled)
    _ = log.check("sampled energies are never below the true ground energy", best_sampled >= 3.0 - 1e-9)

    log.finish()
