# QUMO algorithms (`qumo.algorithms`)

Composable quantum kernels, problem Hamiltonians, exact references, gradients,
optimisers and variational drivers. Everything is plain Mojo on top of the
QUMO IR (`Circuit`, `Param`, `PauliSum`) and the state-vector simulator.

```
qumo/algorithms/
  library.mojo       circuit builders ("kernels"): QFT, Grover, QPE, BV, DJ, ansatze, Pauli exponentials, Trotter
  hamiltonians.mojo  Graph, MaxCut/TFIM/XXZ/H2 Hamiltonians, PauliOperator (H|psi>), Lanczos, dense diag, exact evolution
  gradients.mojo     ExpectationEngine: adjoint / parameter-shift / finite-difference gradients, batched evaluation
  optimizers.mojo    Objective + Optimizer traits; BFGS, Adam, GradientDescent, SPSA, NelderMead; minimize()
  variational.mojo   VQE, QAOA-MaxCut, parameter sweeps / landscapes
  linalg.mojo        state-vector BLAS-1, Jacobi eigensolver, tridiagonal bisection
```

Run the tests and examples from the repo root:

```bash
.venv/bin/mojo run -I . tests/test_algorithms.mojo     # prints "test_algorithms OK"
.venv/bin/mojo run -I . examples/vqe_h2.mojo
.venv/bin/mojo run -I . examples/qaoa_maxcut.mojo
.venv/bin/mojo run -I . examples/grover.mojo
.venv/bin/mojo run -I . examples/qpe.mojo
.venv/bin/mojo run -I . examples/gradient_bench.mojo  # adjoint vs parameter-shift timing
```

## Conventions

* Little-endian qubits: bit k of a basis-state index is qubit k. Bitstring keys
  are Qiskit-style (leftmost char = highest clbit); `bitstring_to_int(key)` converts.
* Rotations: `R_P(theta) = exp(-i theta P / 2)`; `P(lambda) = diag(1, e^{i lambda})`.
* Parameters are affine `Param`s (`coeff * theta[i] + offset`) declared by name
  on the circuit (`c.parameter("gamma[0]")`). Parameter vectors are ordered by
  declaration; `circuit.param_names_str()` prints them.
* Two builder flavours: `append_*(mut c, ...)` writes into an existing circuit
  (zero copies; `Param`s must belong to `c`); value-returning builders create a
  fresh, named circuit.
* Every random choice takes an explicit `seed` (initial points, SPSA, sampling,
  Lanczos start vector) -> bit-identical reruns.

## Circuit library (`library.mojo`)

| Builder | Qubit layout | Params | Origin of the idea |
|---|---|---|---|
| `ghz(n)` | 0..n-1 | none | textbook |
| `qft(n, do_swaps=True, approximation_degree=0)` | register value little-endian on 0..n-1; `QFT|j> = 2^{-n/2} sum_k e^{2 pi i jk/2^n} |k>` | none | Qiskit `QFT` (incl. AQFT degree) |
| `iqft(n, ...)` | as `qft` | none | built with the Adjoint functor (`inverse()`), Q# style |
| `append_qft(mut c, qubits, do_swaps, approximation_degree)` | qubits[k] = bit k | none | |
| `append_mcz(mut c, qubits, ancillas)` / `append_mcx(mut c, controls, target, ancillas)` | needs `num_mcz_ancillas(k) = max(0, k-3)` clean ancillas | none | Toffoli V-chain (Barenco et al.) |
| `phase_oracle(n, marked)` | search 0..n-1, ancillas n..n+max(0,n-3)-1 | none | Grover phase oracle |
| `grover_diffusion(n)` | as `phase_oracle` | none | |
| `grover(n, marked, iterations=-1, measure=False)` | as `phase_oracle`; measure q -> clbit q | none | iterations default `grover_optimal_iterations(n, M)` = floor(pi / 4 theta) |
| `grover_bitstrings(keys, iterations=-1, measure=True)` | n = key length | none | agent-friendly entry ("find '1011'") |
| `grover_optimal_iterations(n, M)`, `grover_success_probability(n, M, k)` | - | - | sin^2((2k+1) theta) |
| `bernstein_vazirani(n, secret, measure=False)` | 0..n-1 (phase-oracle form, no ancilla) | none | |
| `deutsch_jozsa(n, balanced_mask=0, constant_value=0, measure=False)` | inputs 0..n-1, ancilla n in \|-> | none | f(x) = mask.x XOR c |
| `qpe(unitary, n_eval, state_prep=None, measure=False)` | eval 0..n_eval-1 (little-endian, value ~ phi 2^n_eval), system n_eval.. | those of `unitary` | built from IR functors `power(2^j)` + `controlled()` (Q#/Cirq), then `iqft` |
| `qpe_phase(value, n_eval)` | - | - | value / 2^n_eval |
| `append_pauli_rotation(mut c, x, z, theta)` | Pauli string from symplectic masks | theta (one gate: RZ/RZZ/ladder+RZ) | basis change + CX ladder + RZ (TKET `PauliExpBox`, Qiskit `PauliEvolutionGate`) |
| `append_pauli_exponential(mut c, term, t)` | | `exp(-i t c P)` (angle `2 c t`) | |
| `pauli_exponential(term, num_qubits, time=1.0, param_name="")` | 0..num_qubits-1 | optional symbolic time | |
| `trotter(h, time, steps, order=1, num_qubits=-1, param_name="")` | 0..n-1 | optional symbolic total time | Lie-Trotter (order 1), Strang (order 2) |
| `hardware_efficient_ansatz(n, layers, entanglement="linear", rotations="ry,rz", prefix="theta")` | 0..n-1; entanglement `linear` / `circular` / `full` / `none` | `theta[k]`, k < n R (layers+1), layer-major, then qubit, then rotation | Kandala 2017, Qiskit `EfficientSU2` |
| `append_entangler(mut c, n, entanglement)` | | none | |
| `qaoa_ansatz(cost, p, num_qubits=-1)` | 0..n-1 | `gamma[0..p-1]`, `beta[0..p-1]` (that order) | Farhi et al., Qiskit `QAOAAnsatz`; any diagonal cost (ZZ -> one RZZ, Z -> RZ), mixer RX(2 beta) |
| `qaoa_maxcut_ansatz(graph, p)` | node k = qubit k | as above; edge (u,v,w) -> RZZ(w gamma) | |
| `append_single_excitation(mut c, theta, p, q)` | p < q, Jordan-Wigner | `exp(theta (a+_q a_p - h.c.))`: 2 rotations sharing theta | Qiskit Nature / PennyLane qchem |
| `append_double_excitation(mut c, theta, p, q, r, s)` | p < q < r < s | 8 rotations sharing theta (angles +-theta/4) | Yordanov et al. |
| `uccsd_ansatz(num_qubits, num_electrons, include_singles=True, include_doubles=True)` | interleaved spin orbitals (even = up); HF = X on 0..N_e-1 | `s(i->a)`, `d(i,j->a,b)` (spin-conserving) | UCCSD (Peruzzo 2014), first-order Trotterised |
| `h2_ansatz()` | = `uccsd_ansatz(4, 2)` | `s(0->2)`, `s(1->3)`, `d(0,1->2,3)` | exact for H2: d = 0.11177 |

## Hamiltonians and exact references (`hamiltonians.mojo`)

| API | Meaning |
|---|---|
| `Graph(n)`, `add_edge(u, v, w=1)`, `Graph.ring(n)`, `Graph.path(n)`, `Graph.complete(n)`, `Graph.parse("0-1 1-2:0.5")` | weighted undirected graph; node k <-> qubit k |
| `graph.cut_value(assignment)`, `graph.max_cut() -> (value, assignment)` | brute force (<= 26 nodes) |
| `maxcut(graph, include_offset=True)` | `H = sum w/2 (Z_u Z_v - 1)`, so `<z|H|z> = -cut(z)`; minimise |
| `transverse_field_ising(n, j=1, h=1, periodic=False)` | `-J sum ZZ - h sum X` |
| `heisenberg_xxz(n, j=1, delta=1, periodic=False, field=0)` | `J sum (XX + YY + delta ZZ) + field sum Z` |
| `h2_sto3g(num_qubits=4, include_nuclear_repulsion=True)` | H2/STO-3G, R = 0.735 A, generated from the MO integrals `H2_*` (Qiskit Nature/PySCF values). 4 qubits: Jordan-Wigner, interleaved spin orbitals, 15 terms, HF = \|0011>. 2 qubits: parity mapping + two-qubit reduction (identical to the Qiskit VQE tutorial operator). FCI energy `H2_EXACT_ENERGY = -1.1373060357534` Ha, HF `-1.1169989967540` Ha |
| `PauliOperator(h, n)`: `.apply(src, mut dst)`, `.expectation(mut psi)` | H\|psi>: Z-strings pre-summed into one diagonal vector, off-diagonal terms on the fly, multi-threaded |
| `apply_hamiltonian(h, src, mut dst)` | one-shot H\|psi> |
| `ground_state(h, n, max_iter=160, tol=1e-12, seed=2024) -> GroundState(energy, state, iterations, residual)` | Lanczos with full re-orthogonalisation (n <= 20) |
| `ground_energy(h, n=-1)` | Lanczos energy |
| `dense_matrix(h, n)` (n <= 12), `exact_spectrum(h, n)` (n <= 7) | dense Hermitian matrix; all eigenvalues by Jacobi (complex H via real embedding) |
| `exact_evolve(h, psi, t)` | `exp(-iHt)|psi>` by scaled Taylor series (Trotter reference) |

## Gradients (`gradients.mojo`)

```mojo
var eng = ExpectationEngine(ansatz, H)          # owns psi, lam, prepared H, scratch circuit
var e = eng.value(x)
var g = List[Float64]()
var e2 = eng.value_and_gradient(x, g, Grad.ADJOINT)   # or Grad.PARAM_SHIFT / Grad.FINITE_DIFF
var g2 = gradient(ansatz, H, x, Grad.PARAM_SHIFT)     # one-shot helper
var es = batch_expectation(ansatz, H, param_sets)      # parallel over sets for <= 12 qubits
var gs = batch_gradient(ansatz, H, param_sets, Grad.ADJOINT)
```

| Method | How | Cost per gradient | Origin |
|---|---|---|---|
| `Grad.ADJOINT` | forward psi, lam = H psi, reverse sweep applying U_k^dag to both and accumulating `2 Im<lam|G_k|psi>` with fused reductions | ~3-5 simulations, 2 state vectors, independent of #params | Jones & Gacon 2020; PennyLane-Lightning |
| `Grad.PARAM_SHIFT` | per gate occurrence: two-term rule (shift pi/2) for RX/RY/RZ/RXX/RYY/RZZ/P/CP and each U angle; four-term rule (pi/2, 3pi/2; c+- = (sqrt2 +- 1)/(4 sqrt2)) for CRX/CRY/CRZ; chain rule via `Param.coeff`, shared params summed | 2 (or 4) circuit runs per parametric gate (`shift_evaluations(c)`) | Mitarai 2018, Schuld 2019, PennyLane |
| `Grad.FINITE_DIFF` | central differences, step 1e-6 | 2 runs per parameter | universal fallback |

Generators used by the adjoint method (`U = exp(-i phi G)`): RX/RY/RZ -> X/2, Y/2, Z/2;
RXX/RYY/RZZ -> PP/2; P -> -\|1><1\|; CP -> -\|11><11\|; CRX/CRY/CRZ -> \|1><1\|_c (x) P/2;
U(theta, phi, lam) is peeled as RZ(phi) RY(theta) RZ(lam) (its global phase cancels).

Measured (tests): the three methods agree to 2e-10 on random 4-qubit circuits
containing every parametric gate kind with affine and shared parameters.
Benchmark (`examples/gradient_bench.mojo`, 16 qubits, 96 params, 126 gates,
TFIM): one evaluation 2.3 ms, adjoint gradient 11 ms (4.8 evaluations),
parameter-shift 470 ms (193 executions), adjoint 42x faster, max difference 3e-14.

## Optimisers (`optimizers.mojo`)

```mojo
trait Objective:
    def num_params(self) -> Int
    def value(mut self, x: List[Float64]) raises -> Float64
    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64

trait Optimizer(Copyable):
    def name(self) -> String
    def minimize[O: Objective](self, mut obj: O, x0: List[Float64]) raises -> OptimizeResult

minimize[O: Objective](mut obj, x0, method="bfgs", maxiter=200, tol=1e-7, lr=0.05, seed=1234) -> OptimizeResult
OptimizeResult: x, fun, nfev, ngrad, nit, history, converged, message, method
```

Mojo closures are statically typed, so objectives are structs implementing
`Objective` (e.g. `VQEObjective`, `Rosenbrock`); optimisers are generic over them
(monomorphised, no dynamic dispatch).

| Optimiser | Notes | Origin |
|---|---|---|
| `BFGS(maxiter=200, gtol=1e-7, c1=1e-4, max_step=1.0)` | dense inverse-Hessian update, Armijo backtracking, curvature reset | SciPy `BFGS` |
| `Adam(lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, maxiter=500, gtol=1e-6)` | returns best iterate | Kingma & Ba; PennyLane/Qiskit |
| `GradientDescent(lr=0.1, momentum=0.0, maxiter=500, gtol=1e-6)` | heavy-ball momentum | |
| `SPSA(maxiter=300, a=-1 (calibrate), c=0.1, alpha=0.602, gamma=0.101, stability=0, target_step=0.2, seed=1234)` | 2 evaluations/iteration, seeded Rademacher perturbations | Spall; Qiskit calibration |
| `NelderMead(maxiter=2000, initial_step=0.25, xatol=1e-8, fatol=1e-10, adaptive=True)` | adaptive coefficients | Gao & Han 2012, SciPy |

## Variational drivers (`variational.mojo`)

| API | Returns |
|---|---|
| `vqe(ansatz, h, method="bfgs", x0=[], maxiter=200, tol=1e-8, grad_method=Grad.ADJOINT, seed=7)` | `VQEResult(energy, params, param_names, history, nit, nfev, ngrad, circuit_evals, wall_time_s, converged, message, optimizer, grad_method)` |
| `vqe_with[Opt: Optimizer](ansatz, h, optimizer, x0=[], grad_method)` | same, with an optimiser instance (`vqe_with(a, H, SPSA(maxiter=300, seed=11))`) |
| `VQEObjective(ansatz, h, grad_method)` | `Objective` adapter owning an `ExpectationEngine` (buffers reused across the run) |
| `qaoa_maxcut(graph, p=1, method="bfgs", restarts=4, shots=1024, seed=7, maxiter=200, grad_method)` | `QAOAResult(p, energy, expected_cut, max_cut, approx_ratio, gammas, betas, best_bitstring, best_cut, sampled_ratio, prob_optimal, most_frequent, shots, history, nfev, circuit_evals, wall_time_s, optimizer)` |
| `parameter_sweep(circuit, h, index, values, base)`, `landscape_2d(circuit, h, i, j, values_i, values_j, base)`, `linspace(a, b, n)` | energies through `batch_expectation` |
| `default_initial_point(num_params, seed=7, scale=0.1)` | seeded small random angles (default `x0`) |

Measured (tests/examples):

* H2 4-qubit UCCSD, BFGS + adjoint: |E - FCI| = 2e-14 in 11 iterations / 22 circuit
  executions / 0.4 ms. Adam + parameter-shift 2e-14, Nelder-Mead 1e-9, SPSA 3e-6
  (all within chemical accuracy 1.6e-3). 2-qubit parity H2 + hardware-efficient
  ansatz: 9e-16.
* QAOA MaxCut on rings: approximation ratio 0.75 / 0.833 / 0.875 for p = 1 / 2 / 3
  (matches the (2p+1)/(2p+2) law); optimal cut always found by sampling.
* Grover n = 3..8: success probability equals sin^2((2k+1) theta) to 1e-14.
* QPE: dyadic phases exact (P = 1), phase 1/3 converges at 2^-n_eval.
* Trotter (TFIM, n = 4, t = 1): infidelity scales as steps^-2 (order 1) and
  steps^-4 (order 2); 32 Strang steps -> 1.4e-7.

## Error codes (QM-E8xx)

Every error starts with a stable code and ends with a `Hint:` where a fix is known.

| Code | Raised by | Meaning |
|---|---|---|
| QM-E801 | `hardware_efficient_ansatz`, `append_entangler`, excitations | invalid option (unknown entanglement / rotation, wrong orbital ordering) |
| QM-E802 | builders, `h2_sto3g`, model Hamiltonians | invalid size or index (n out of range, marked item outside search space, duplicate marked item, missing ancillas) |
| QM-E803 | `PauliOperator`, `ExpectationEngine`, `append_pauli_rotation` | Hamiltonian / Pauli string acts on more qubits than the circuit/state |
| QM-E804 | `ExpectationEngine`, optimisers, sweeps | parameter vector length or parameter index mismatch |
| QM-E805 | `check_differentiable`, adjoint | circuit not differentiable (classical control, reset, noise, mid-circuit measurement) |
| QM-E806 | `minimize`, `grad_method_from_name` | unknown optimiser or gradient method |
| QM-E807 | `ground_state`, `dense_matrix`, `exact_spectrum`, `Graph.max_cut` | problem too large for an exact / brute-force routine |
| QM-E808 | `trotter` | steps < 1 or order not in {1, 2} |
| QM-E809 | `Graph` | self-loop, negative node, unparsable edge text |
| QM-E810 | `qpe` | n_eval outside 1..16, empty unitary, state_prep too wide |

## Where the ideas come from

* **CUDA-Q / Q#**: algorithms as functions returning IR; QPE built only from the
  generic `power()` and `controlled()` functors; `iqft` = Adjoint of `qft`.
* **PennyLane**: interchangeable `diff_method` (adjoint, parameter-shift incl. the
  four-term rule for controlled rotations, finite differences), templates with
  documented wires and parameters, UCC excitation gates.
* **Qiskit**: `EfficientSU2`, `QAOAAnsatz`, `QFT` (approximation degree), the H2
  tutorial Hamiltonian, SPSA calibration, `VQEResult`-style reporting, and
  PUB-style batched parameter sets (`batch_expectation`).
* **TKET**: Pauli-gadget synthesis (basis change + CX ladder + one rotation),
  single-RZZ special case for weight-2 strings.
* **Qualtran**: builders state their resource shape (qubits, ancillas, gate
  counts, executions per gradient via `shift_evaluations`).
* **Cirq / Braket / Qibo**: graph-based QAOA helpers and brute-force reference
  solutions next to the quantum result (approximation ratio, P(optimal)).

## Limitations

* Gradients are exact-expectation (state-vector) gradients; shot-noise-aware
  parameter-shift on samplers belongs to `qumo.exec` primitives.
* `qpe` repeats U 2^j times (no fast-forwarding of U^k); keep n_eval <= ~12.
* `uccsd_ansatz` is first-order Trotterised UCCSD with spin-conserving
  excitations under Jordan-Wigner, interleaved spin orbitals only; no
  integral/driver interface beyond the built-in H2 integrals.
* `ground_state` stores the full Krylov basis (n <= 20); `exact_spectrum` n <= 7.
* `append_pauli_rotation` with a symbolic angle drops identity terms (global phase).
