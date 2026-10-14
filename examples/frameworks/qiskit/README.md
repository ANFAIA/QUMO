# Qiskit signature examples, reproduced with QUMO

Four IBM Quantum tutorials, rebuilt from QUMO's own library (no Qiskit at run
time) and verified with machine-readable `CHECK` lines. Each file's docstring
cites the original, states every deviation and lists its checks.

```bash
.venv/bin/mojo run -I . examples/frameworks/qiskit/chsh.mojo
bench/.venv/bin/python examples/run_examples.py --only examples/frameworks/qiskit   # all, with timeouts
bench/.venv/bin/python examples/run_examples.py --only examples/frameworks/qiskit --tier all   # includes SKQD (slow)
```

| Original | URL | QUMO file | What is checked | Reference numbers (provenance) |
|---|---|---|---|---|
| Quantum approximate optimization algorithm (small-scale part) | https://quantum.cloud.ibm.com/docs/tutorials/quantum-approximate-optimization-algorithm | `qaoa_maxcut.mojo` | brute-force max cut; optimal most frequent bitstring (10k shots); approximation ratio >= 0.8; Estimator == state vector (1e-12); QUMO == qiskit at two fixed points (1e-10); NelderMead (COBYLA stand-in) and BFGS + adjoint agree (1e-6); the 100-qubit part replaced by resource estimates | max cut 5 and bitstring [0, 0, 1, 0, 1] (page); `<sum ZZ>` = -5.898e-17 and 2.9777060466916727 (rerun qiskit 2.5.2); measured ratio 0.9247, P(optimal cut) = 0.7056 |
| CHSH inequality | https://quantum.cloud.ibm.com/docs/tutorials/chsh-inequality | `chsh.mojo` | one Estimator PUB (21 thetas x 2 observables), exact and precision 0.01; closed form at all points (1e-12); shot estimates within 4 sigma; violation 2 < max S <= 2 sqrt 2 + 4 sigma | max \|S\| = 2 sqrt(2) cos(pi/20) = 2.7936 (analytic); hardware mean peak ~2.75 (page, quoted only) |
| Shor's algorithm | https://quantum.cloud.ibm.com/docs/tutorials/shors-algorithm | `shor_order_finding.mojo` | N=15 tutorial circuit (SWAP-built M2/M4 made controlled by `Circuit.controlled()`): peaks {0, 64, 128, 192} at 0.25 (1e-12), r = 4, factors 3 x 5, 2000-shot frequencies within 4 sigma; the generic permutation oracle gives the same distribution; N=21 with `order_finding`: r = 6 on >= 50% of the mass and of 2000 samples, factors 3 x 7 | counts 264/268/249/243 of 1024 and phases {0, 1/4, 1/2, 3/4} (page); peaks 0.25 (analytic); measured N=21 success 0.826 (exact) / 0.825 (sampled) |
| Sample-based Krylov quantum diagonalization (method) | https://quantum.cloud.ibm.com/docs/tutorials/sample-based-krylov-quantum-diagonalization and https://nvidia.github.io/cuda-quantum/0.14.0/applications/python/skqd.html | `skqd_heisenberg.mojo` (slow tier) | CUDA-Q's Heisenberg + fields Hamiltonian at 12 qubits, Neel start, dt = pi/8, 4000 shots per Krylov state: variational and non-increasing energies; relative error <= 1% at D = 8; full basis == exact (1e-10) | exact E0 = -22.967854412864753 (Lanczos, analytic reference); measured 0.55% at D = 8. The instance differs from both pages (IBM: Anderson impurity model; CUDA-Q: 22 qubits) |

Optional cross-checks with qiskit 2.5.2 / qiskit-aer 0.17.2 (`reference/`, run
with `bench/.venv/bin/python`, excluded from the default runner):

| Script | What it does |
|---|---|
| `reference/qaoa_ref.py [--qumo-angles g0 g1 b0 b1 --qumo-value v]` | qiskit `QAOAAnsatz` + `Statevector` at the committed points and at QUMO's optimum (1e-10; measured 2e-14). Angle map: gamma_qiskit = gamma_qumo / 2. |
| `reference/shor15_ref.py` | the tutorial's N=15 circuit in qiskit: exact marginal == QUMO's peaks (1e-12) and 10,000 Aer shots with TVD < 0.02 (measured 0.0052) |

Conventions: QUMO and Qiskit share little-endian qubit order and bitstring keys
(leftmost character = highest qubit), so Pauli labels such as `"ZX"` and keys
such as `"01000000"` are copied verbatim. COBYLA is not part of QUMO; the
examples say which optimiser replaces it.
