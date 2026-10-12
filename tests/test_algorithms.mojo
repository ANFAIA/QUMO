"""Tests for qumo.algorithms: library kernels, Hamiltonians + exact solvers,
gradients (adjoint / parameter-shift / finite differences), optimisers, VQE,
QAOA and Trotterisation. Prints `test_algorithms OK`."""

from std.math import pi, sqrt, cos, sin
from std.testing import assert_true, assert_almost_equal, assert_equal

from qumo.ir.circuit import Circuit
from qumo.ir.param import Param
from qumo.ir.pauli import PauliSum, PauliTerm, parse_pauli_sum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, sample
from qumo.sim.rng import Rng
from qumo.algorithms.library import (
    ghz, qft, iqft, grover, grover_bitstrings, grover_optimal_iterations, grover_success_probability,
    num_mcz_ancillas, bernstein_vazirani, deutsch_jozsa, qpe, qpe_phase,
    hardware_efficient_ansatz, qaoa_ansatz, qaoa_maxcut_ansatz, h2_ansatz, uccsd_ansatz,
    append_pauli_rotation, pauli_exponential, trotter, bitstring_to_int,
)
from qumo.algorithms.hamiltonians import (
    Graph, maxcut, transverse_field_ising, heisenberg_xxz, h2_sto3g, H2_EXACT_ENERGY,
    H2_HF_ENERGY, PauliOperator, apply_hamiltonian, ground_state, ground_energy,
    exact_spectrum, exact_evolve,
)
from qumo.algorithms.gradients import (
    Grad, gradient, value_and_gradient, ExpectationEngine, batch_expectation,
    batch_gradient, shift_evaluations, grad_method_from_name,
)
from qumo.algorithms.optimizers import (
    minimize, Rosenbrock, BFGS, Adam, SPSA, NelderMead, GradientDescent,
)
from qumo.algorithms.variational import (
    vqe, vqe_with, qaoa_maxcut, parameter_sweep, landscape_2d, linspace, VQEObjective,
)


def _err_code(msg: String) -> String:
    return String(msg[byte=0:7]) if msg.byte_length() >= 7 else msg


def _prob(sv: StateVector, i: Int) -> Float64:
    return sv.re[i] * sv.re[i] + sv.im[i] * sv.im[i]


# ---------------------------------------------------------------------------
def test_ghz_qft() raises:
    var g = evolve(ghz(5))
    assert_almost_equal(g.re[0], 1 / sqrt(2.0))
    assert_almost_equal(g.re[31], 1 / sqrt(2.0))
    # QFT on every basis state of 4 qubits == DFT column
    var n = 4
    var big_n = 1 << n
    var maxerr = 0.0
    for j in range(big_n):
        var c = Circuit(n)
        for q in range(n):
            if ((j >> q) & 1) == 1:
                c.x(q)
        c.compose(qft(n))
        var sv = evolve(c)
        for k in range(big_n):
            var ang = 2 * pi * Float64(j * k) / Float64(big_n)
            maxerr = max(maxerr, abs(sv.re[k] - cos(ang) / 4.0))
            maxerr = max(maxerr, abs(sv.im[k] - sin(ang) / 4.0))
    assert_true(maxerr < 1e-12, "QFT != DFT, err=" + String(maxerr))
    # superposition input + iqft(qft) = identity
    var prep = Circuit(5)
    for q in range(5):
        prep.ry(0.3 + 0.2 * Float64(q), q)
        prep.rz(0.1 * Float64(q), q)
    var round_trip = prep.copy()
    round_trip.compose(qft(5))
    round_trip.compose(iqft(5))
    assert_almost_equal(evolve(prep).fidelity(evolve(round_trip)), 1.0, atol=1e-12)
    # no-swap QFT output = bit-reversed DFT
    var c2 = Circuit(3)
    c2.x(0)  # |1>
    c2.compose(qft(3, do_swaps=False))
    var s2 = evolve(c2)
    for k in range(8):
        var rev = ((k & 1) << 2) | (k & 2) | ((k >> 2) & 1)
        var ang = 2 * pi * Float64(rev) / 8.0
        assert_almost_equal(s2.re[k], cos(ang) / sqrt(8.0), atol=1e-12)
        assert_almost_equal(s2.im[k], sin(ang) / sqrt(8.0), atol=1e-12)
    print("  qft: max |QFT - DFT| =", maxerr)


def test_grover() raises:
    for n in range(3, 6):
        var marked: List[Int] = [5 % (1 << n)]
        if n >= 4:
            marked.append((1 << n) - 2)
        var c = grover(n, marked)
        var sv = evolve(c)
        var p = 0.0
        for m in marked:
            p += _prob(sv, m)  # ancillas are returned to |0>
        var k = grover_optimal_iterations(n, len(marked))
        var ideal = grover_success_probability(n, len(marked), k)
        assert_almost_equal(p, ideal, atol=1e-9)
        assert_true(p > 0.9, "Grover success too low: " + String(p))
        var cm = grover(n, marked, measure=True)
        var r = sample(cm, 500, seed=UInt64(n))
        var top = bitstring_to_int(r.most_frequent())
        var hit = False
        for m in marked:
            if m == top:
                hit = True
        assert_true(hit, "Grover most frequent outcome is not marked")
        print("  grover n=", n, " marked=", len(marked), " iters=", k, " P(success)=", p, " ancillas=", num_mcz_ancillas(n))
    # bitstring-keyed entry point (Qiskit key order)
    var rb = sample(grover_bitstrings(["0110", "1011"]), 400, seed=2)
    assert_true(rb.counts.get("0110", 0) + rb.counts.get("1011", 0) > 340)


def test_qpe() raises:
    for m in range(2, 5):
        for k in range(1 << m):
            var u = Circuit(1)
            u.p(2 * pi * Float64(k) / Float64(1 << m), 0)
            var prep = Circuit(1)
            prep.x(0)
            var c = qpe(u, m, prep.copy())
            var sv = evolve(c)
            # eval register value k with system qubit (index m) in |1>
            var idx = k | (1 << m)
            assert_almost_equal(_prob(sv, idx), 1.0, atol=1e-9)
    # two-qubit unitary: CP(2 pi * 3/8) eigenstate |11> -> phase 3/8
    var u2 = Circuit(2)
    u2.cp(2 * pi * 3.0 / 8.0, 0, 1)
    var p2 = Circuit(2)
    p2.x(0)
    p2.x(1)
    var c2 = qpe(u2, 3, p2.copy(), measure=True)
    var r2 = sample(c2, 200, seed=5)
    assert_equal(r2.most_frequent(), "011")
    assert_almost_equal(qpe_phase(bitstring_to_int(r2.most_frequent()), 3), 3.0 / 8.0)
    # non-dyadic phase 1/3 with 6 bits peaks at round(64/3) = 21
    var u3 = Circuit(1)
    u3.p(2 * pi / 3.0, 0)
    var p3 = Circuit(1)
    p3.x(0)
    var s3 = evolve(qpe(u3, 6, p3.copy()))
    var best = 0
    for v in range(64):
        if _prob(s3, v | 64) > _prob(s3, best | 64):
            best = v
    assert_equal(best, 21)
    assert_true(_prob(s3, best | 64) > 0.4)
    print("  qpe: exact dyadic phases OK; phase 1/3 -> 21/64 with P=", _prob(s3, best | 64))


def test_oracles() raises:
    var secret = 0b101101
    var r = sample(bernstein_vazirani(6, secret, measure=True), 64, seed=3)
    assert_equal(bitstring_to_int(r.most_frequent()), secret)
    assert_equal(r.counts.get(r.most_frequent(), 0), 64)
    var cst = sample(deutsch_jozsa(4, 0, 1, measure=True), 64, seed=1)
    assert_equal(cst.counts.get("0000", 0), 64)
    var bal = sample(deutsch_jozsa(4, 0b0110, 0, measure=True), 64, seed=1)
    assert_equal(bal.counts.get("0000", 0), 0)


def test_pauli_exponential() raises:
    # exp(-i theta/2 P) == cos(theta/2) psi - i sin(theta/2) P psi
    var n = 5
    var prep = Circuit(n)
    for q in range(n):
        prep.ry(0.4 + 0.3 * Float64(q), q)
        prep.rz(0.2 * Float64(q * q), q)
    prep.cx(0, 3)
    var labels: List[String] = ["X0 Y1 Z3", "Y2", "Z0 Z4", "X1 X2 Y3 Z4", "Y0 Y4"]
    var theta = 0.731
    for lab in labels:
        var h = parse_pauli_sum(lab)
        var t = h.terms[0]
        var c = prep.copy()
        append_pauli_rotation(c, t.x, t.z, Param(theta))
        var got = evolve(c)
        var psi = evolve(prep)
        var ppsi = StateVector(n)
        apply_hamiltonian(h, psi, ppsi)
        var maxerr = 0.0
        for i in range(1 << n):
            var er = cos(theta / 2) * psi.re[i] + sin(theta / 2) * ppsi.im[i]
            var ei = cos(theta / 2) * psi.im[i] - sin(theta / 2) * ppsi.re[i]
            maxerr = max(maxerr, abs(er - got.re[i]) + abs(ei - got.im[i]))
        assert_true(maxerr < 1e-12, "pauli rotation mismatch for " + lab)
    # pauli_exponential with a symbolic time parameter binds correctly
    var term = parse_pauli_sum("0.8 * X0 Z1").terms[0]
    var cs = pauli_exponential(term, 2, param_name="t")
    var cc = pauli_exponential(term, 2, time=0.37)
    var sym: List[Float64] = [0.37]
    assert_almost_equal(evolve(cs, sym).fidelity(evolve(cc)), 1.0, atol=1e-12)


def test_trotter() raises:
    var n = 4
    var h = transverse_field_ising(n, 1.0, 0.7)
    var prep = Circuit(n)
    for q in range(n):
        prep.ry(0.5 + 0.25 * Float64(q), q)
    var t = 1.0
    var exact = exact_evolve(h, evolve(prep), t)
    # sanity: exact evolution is unitary and conserves energy
    assert_almost_equal(exact.norm_sq(), 1.0, atol=1e-12)
    var s0 = evolve(prep)
    var e0 = s0.expectation(h)
    assert_almost_equal(exact.expectation(h), e0, atol=1e-10)
    var prev1 = 1.0
    var prev2 = 1.0
    var steps_list: List[Int] = [4, 8, 16, 32]
    for steps in steps_list:
        var c1 = prep.copy()
        c1.compose(trotter(h, t, steps, 1))
        var c2 = prep.copy()
        c2.compose(trotter(h, t, steps, 2))
        var inf1 = 1.0 - evolve(c1).fidelity(exact)
        var inf2 = 1.0 - evolve(c2).fidelity(exact)
        assert_true(inf1 < prev1 and inf2 < prev2, "Trotter error must decrease with steps")
        assert_true(inf2 < inf1, "2nd-order Trotter must beat 1st order")
        prev1 = inf1
        prev2 = inf2
        print("  trotter steps=", steps, " infidelity order1=", inf1, " order2=", inf2)
    assert_true(prev2 < 1e-6, "Strang splitting with 32 steps should be within 1e-6")
    # symbolic total time binds to the same circuit
    var sym = trotter(h, 0.0, 8, 2, param_name="t")
    var tv: List[Float64] = [t]
    var fixed = trotter(h, t, 8, 2)
    assert_almost_equal(evolve(sym, tv).fidelity(evolve(fixed)), 1.0, atol=1e-12)


def test_hamiltonians() raises:
    var h4 = h2_sto3g(4)
    var h2 = h2_sto3g(2)
    assert_equal(len(h4), 15)
    assert_equal(len(h2), 5)
    var g4 = ground_state(h4, 4)
    var g2 = ground_state(h2, 2)
    assert_almost_equal(g4.energy, H2_EXACT_ENERGY, atol=1e-10)
    assert_almost_equal(g2.energy, H2_EXACT_ENERGY, atol=1e-10)
    assert_true(abs(g4.energy - (-1.137)) < 1e-3)
    var spec = exact_spectrum(h4, 4)
    assert_almost_equal(spec[0], g4.energy, atol=1e-10)
    # Hartree-Fock |0011> energy
    var hf = StateVector.basis(4, 3)
    assert_almost_equal(hf.expectation(h4), H2_HF_ENERGY, atol=1e-10)
    # PauliOperator expectation agrees with StateVector.expectation
    var op = PauliOperator(h4, 4)
    var rs = evolve(hardware_efficient_ansatz(4, 2), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])
    assert_almost_equal(op.expectation(rs), rs.expectation(h4), atol=1e-12)
    # Heisenberg ring of 4 (Pauli units): E0 = -8
    assert_almost_equal(ground_energy(heisenberg_xxz(4, 1.0, 1.0, True), 4), -8.0, atol=1e-9)
    # critical open TFIM chain: E0 = 1 - 1/sin(pi / (2 (2N + 1)))
    var nt = 10
    var e_tfim = ground_energy(transverse_field_ising(nt, 1.0, 1.0), nt)
    assert_almost_equal(e_tfim, 1.0 - 1.0 / sin(pi / (2.0 * (2.0 * Float64(nt) + 1.0))), atol=1e-9)
    # Lanczos vs dense Jacobi on a random complex 5-qubit Hamiltonian
    var hr = parse_pauli_sum("0.3*X0 Y1 + 0.7*Z2 Z3 - 0.4*Y0 X4 Z1 + 0.25*X3 + 0.9*Z0 - 0.15*Y2 Y4 + 0.6*X1 X2")
    var sp = exact_spectrum(hr, 5)
    assert_almost_equal(ground_energy(hr, 5), sp[0], atol=1e-9)
    var tr = 0.0
    for v in sp:
        tr += v
    assert_almost_equal(tr, 0.0, atol=1e-9)  # traceless
    # MaxCut Ising form: ground = -maxcut
    var g = Graph.parse("0-1 1-2 2-3 3-0 0-2:0.5 1-4:2")
    var mc = g.max_cut()
    assert_almost_equal(ground_energy(maxcut(g), g.num_nodes), -mc[0], atol=1e-9)
    assert_almost_equal(g.cut_value(mc[1]), mc[0])
    print("  H2 exact (Lanczos)=", g4.energy, " HF=", H2_HF_ENERGY, " TFIM-10=", e_tfim, " Lanczos iters=", g4.iterations)


def _random_ansatz(n: Int, ngates: Int, nparams: Int, seed: UInt64) raises -> Circuit:
    """Random circuit exercising every parametric gate kind, affine and shared params."""
    var rng = Rng(seed)
    var c = Circuit(n, name="random_ansatz")
    var th = c.parameters("t", nparams)
    for q in range(n):
        c.h(q)
    for _ in range(ngates):
        var kind = rng.below(14)
        var a = rng.below(n)
        var b = (a + 1 + rng.below(n - 1)) % n
        var p = th[rng.below(nparams)] * (0.5 + rng.uniform()) + (rng.uniform() - 0.5)
        if kind == 0: c.rx(p, a)
        elif kind == 1: c.ry(p, a)
        elif kind == 2: c.rz(p, a)
        elif kind == 3: c.p(p, a)
        elif kind == 4: c.rxx(p, a, b)
        elif kind == 5: c.ryy(p, a, b)
        elif kind == 6: c.rzz(p, a, b)
        elif kind == 7: c.crx(p, a, b)
        elif kind == 8: c.cry(p, a, b)
        elif kind == 9: c.crz(p, a, b)
        elif kind == 10: c.cp(p, a, b)
        elif kind == 11: c.u(p, th[rng.below(nparams)], -p, a)
        elif kind == 12: c.cx(a, b)
        else:
            c.sx(a)
            c.t(b)
    return c^


def test_gradients() raises:
    var h = parse_pauli_sum("0.5*Z0 Z1 - 0.3*X1 Y2 + 0.7*Y0 Z3 + 0.2*X0 X1 X2 X3 + 0.1*Z2 - 0.4*Y1 Y3 + 0.05")
    var worst = 0.0
    for trial in range(3):
        var c = _random_ansatz(4, 40, 10, UInt64(100 + trial))
        var rng = Rng(UInt64(7 + trial))
        var x = List[Float64]()
        for _ in range(c.num_params()):
            x.append(2 * pi * rng.uniform() - pi)
        var ga = gradient(c, h, x, Grad.ADJOINT)
        var gp = gradient(c, h, x, Grad.PARAM_SHIFT)
        var gf = gradient(c, h, x, Grad.FINITE_DIFF)
        for i in range(len(x)):
            worst = max(worst, abs(ga[i] - gp[i]))
            worst = max(worst, abs(ga[i] - gf[i]))
            worst = max(worst, abs(gp[i] - gf[i]))
    assert_true(worst < 1e-6, "gradient methods disagree: " + String(worst))
    # UCCSD: one parameter shared by 8 gates with coefficients +-1/4
    var u = uccsd_ansatz(4, 2)
    var hu = h2_sto3g(4)
    var xu: List[Float64] = [0.3, -0.2, 0.4]
    var ua = gradient(u, hu, xu, Grad.ADJOINT)
    var up = gradient(u, hu, xu, Grad.PARAM_SHIFT)
    for i in range(3):
        assert_almost_equal(ua[i], up[i], atol=1e-9)
    # general UCCSD (8 spin orbitals, 4 electrons, Z-strings inside excitations):
    # every populated basis state must keep N = 4 electrons and Sz = 0
    var big = uccsd_ansatz(8, 4)
    var rngu = Rng(21)
    var xb = List[Float64]()
    for _ in range(big.num_params()):
        xb.append(rngu.uniform() - 0.5)
    var sb = evolve(big, xb)
    var leak = 0.0
    for i in range(256):
        var up = 0
        var dn = 0
        for q in range(8):
            if ((i >> q) & 1) == 1:
                if q % 2 == 0:
                    up += 1
                else:
                    dn += 1
        if up != 2 or dn != 2:
            leak += _prob(sb, i)
    assert_true(leak < 1e-20, "UCCSD leaves the N=4, Sz=0 sector: " + String(leak))
    assert_true(_prob(sb, 15) < 0.999)  # it did move away from Hartree-Fock
    # value_and_gradient energy == expectation; engine counters
    var eng = ExpectationEngine(u, hu)
    var gbuf = List[Float64]()
    var e = eng.value_and_gradient(xu, gbuf, Grad.ADJOINT)
    assert_almost_equal(e, eng.value(xu), atol=1e-12)
    assert_equal(len(gbuf), 3)
    assert_equal(shift_evaluations(u), 2 * (2 * 2 + 8))
    # batched evaluation == one-by-one
    var c = _random_ansatz(4, 30, 6, 77)
    var sets = List[List[Float64]]()
    var rng = Rng(9)
    for _ in range(16):
        var x = List[Float64]()
        for _ in range(6):
            x.append(rng.uniform() * 3.0)
        sets.append(x^)
    var be = batch_expectation(c, h, sets)
    var bg = batch_gradient(c, h, sets, Grad.ADJOINT)
    for b in range(16):
        var single_eng = ExpectationEngine(c, h)
        var single = single_eng.value(sets[b])
        assert_almost_equal(be[b], single, atol=1e-12)
        var gs = gradient(c, h, sets[b], Grad.PARAM_SHIFT)
        for i in range(6):
            assert_almost_equal(bg[b][i], gs[i], atol=1e-9)
    assert_equal(grad_method_from_name("ps"), Grad.PARAM_SHIFT)
    print("  gradients: max |adjoint - param-shift - finite-diff| =", worst)


def test_optimizers() raises:
    var x0: List[Float64] = [-1.2, 1.0]
    var rb = Rosenbrock(2)
    var r1 = minimize(rb, x0, "bfgs", 500)
    assert_true(r1.converged and r1.fun < 1e-10, String(r1))
    var r2 = minimize(rb, x0, "nelder-mead", 4000, 1e-14)
    assert_true(r2.fun < 1e-8, String(r2))
    var rb4 = Rosenbrock(4)
    var x4: List[Float64] = [0.5, 0.5, 0.5, 0.5]
    var r3 = BFGS(maxiter=500).minimize(rb4, x4)
    assert_true(r3.fun < 1e-10, String(r3))
    var r4 = Adam(lr=0.01, maxiter=4000).minimize(rb, x0)
    assert_true(r4.fun < 1e-3, String(r4))
    var r5 = GradientDescent(lr=0.001, momentum=0.9, maxiter=5000).minimize(rb, x0)
    assert_true(r5.fun < 1e-2, String(r5))
    var start = rb.value(x0)
    var r6 = SPSA(maxiter=1000, seed=3).minimize(rb, x0)
    assert_true(r6.fun < start, String(r6))
    # SPSA is bit-reproducible for a fixed seed
    var r7 = SPSA(maxiter=1000, seed=3).minimize(rb, x0)
    assert_equal(r6.fun, r7.fun)
    print("  optimizers on Rosenbrock: bfgs=", r1.fun, " nm=", r2.fun, " adam=", r4.fun, " spsa=", r6.fun)


def test_vqe() raises:
    var h4 = h2_sto3g(4)
    var exact = ground_energy(h4, 4)
    var r = vqe(h2_ansatz(), h4)
    assert_true(abs(r.energy - exact) < 1e-4, String(r))
    assert_true(abs(r.energy - exact) < 1e-8)
    # parameter-shift gradients + Adam reach chemical accuracy too
    var r2 = vqe(h2_ansatz(), h4, "adam", maxiter=400, grad_method=Grad.PARAM_SHIFT)
    assert_true(abs(r2.energy - exact) < 1.6e-3, String(r2))
    # gradient-free: SPSA / Nelder-Mead
    var r3 = vqe(h2_ansatz(), h4, "nelder-mead", maxiter=2000)
    assert_true(abs(r3.energy - exact) < 1e-6, String(r3))
    var r4 = vqe_with(h2_ansatz(), h4, SPSA(maxiter=300, seed=11))
    assert_true(abs(r4.energy - exact) < 1.6e-3, String(r4))
    # 2-qubit parity-reduced H2 with a hardware-efficient ansatz
    var h2 = h2_sto3g(2)
    var r5 = vqe(hardware_efficient_ansatz(2, 2, "linear", "ry,rz"), h2)
    assert_true(abs(r5.energy - exact) < 1e-6, String(r5))
    # parameter sweep through the batched evaluator
    var sweep = parameter_sweep(h2_ansatz(), h4, 2, linspace(-0.5, 0.5, 21), [0.0, 0.0, 0.0])
    var mn = 1e9
    for v in sweep:
        mn = min(mn, v)
    assert_true(mn < -1.13 and mn >= exact - 1e-12)
    var grid = landscape_2d(h2_ansatz(), h4, 0, 2, linspace(-0.2, 0.2, 3), linspace(-0.2, 0.2, 5), [0.0, 0.0, 0.0])
    assert_equal(len(grid), 15)
    print("  vqe H2: exact=", exact, " bfgs/adjoint err=", r.energy - exact, " (", r.nit, "iters,", r.circuit_evals, "circuit evals,", r.wall_time_s, "s)")
    print("  vqe H2: adam/param-shift err=", r2.energy - exact, " nm err=", r3.energy - exact, " spsa err=", r4.energy - exact, " hea-2q err=", r5.energy - exact)


def test_qaoa() raises:
    var g = Graph.ring(6)
    var r = qaoa_maxcut(g, 2, shots=2000, seed=5)
    assert_true(r.approx_ratio > 0.7, String(r))
    assert_almost_equal(r.best_cut, r.max_cut)
    var r1 = qaoa_maxcut(g, 1, seed=5)
    assert_true(r1.approx_ratio > 0.7, String(r1))
    assert_true(r.approx_ratio >= r1.approx_ratio - 1e-9)
    # weighted graph from text
    var gw = Graph.parse("0-1:2 1-2 2-3:0.5 3-0 0-2:1.5")
    var rw = qaoa_maxcut(gw, 2, seed=1)
    assert_true(rw.approx_ratio > 0.7, String(rw))
    print("  qaoa ring6: p=1 ratio=", r1.approx_ratio, " p=2 ratio=", r.approx_ratio, " P(opt)=", r.prob_optimal, " best=", r.best_bitstring)
    print("  qaoa weighted: ratio=", rw.approx_ratio, " best cut=", rw.best_cut, "/", rw.max_cut)


def test_errors() raises:
    var codes = List[String]()
    try:
        _ = hardware_efficient_ansatz(3, 1, "bogus")
    except e:
        codes.append(_err_code(String(e)))
    try:
        _ = hardware_efficient_ansatz(3, 1, "linear", "ry,foo")
    except e:
        codes.append(_err_code(String(e)))
    try:
        _ = grover(3, [9])
    except e:
        codes.append(_err_code(String(e)))
    try:
        _ = h2_sto3g(3)
    except e:
        codes.append(_err_code(String(e)))
    try:
        var c = Circuit(2)
        _ = gradient(c, parse_pauli_sum("Z0 Z1 Z2"), List[Float64]())
    except e:
        codes.append(_err_code(String(e)))
    try:
        _ = gradient(h2_ansatz(), h2_sto3g(4), [0.1])
    except e:
        codes.append(_err_code(String(e)))
    try:
        var c = Circuit(2)
        var t = c.parameter("t")
        c.ry(t, 0)
        c.measure(0, 0)
        c.rx(t, 0)
        _ = gradient(c, parse_pauli_sum("Z0"), [0.1])
    except e:
        codes.append(_err_code(String(e)))
    try:
        var rb = Rosenbrock(2)
        _ = minimize(rb, [0.0, 0.0], "lbfgs-magic")
    except e:
        codes.append(_err_code(String(e)))
    try:
        _ = trotter(transverse_field_ising(3), 1.0, 4, 3)
    except e:
        codes.append(_err_code(String(e)))
    try:
        _ = Graph.parse("0-0")
    except e:
        codes.append(_err_code(String(e)))
    try:
        _ = qpe(Circuit(1), 20)
    except e:
        codes.append(_err_code(String(e)))
    var expected: List[String] = ["QM-E801", "QM-E801", "QM-E802", "QM-E802", "QM-E803", "QM-E804", "QM-E805", "QM-E806", "QM-E808", "QM-E809", "QM-E810"]
    assert_equal(len(codes), len(expected))
    for i in range(len(expected)):
        assert_equal(codes[i], expected[i])


def main() raises:
    test_ghz_qft()
    test_grover()
    test_qpe()
    test_oracles()
    test_pauli_exponential()
    test_trotter()
    test_hamiltonians()
    test_gradients()
    test_optimizers()
    test_vqe()
    test_qaoa()
    test_errors()
    print("test_algorithms OK")
