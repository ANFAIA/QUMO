# Q# (Microsoft QDK) signature examples, reproduced with QUMO

Four samples from https://github.com/microsoft/qdk (MIT), rebuilt with QUMO's
library (the QDK is not needed at run time) and verified with
machine-readable `CHECK` lines. Q# result arrays list qubit 0 first and
`ResultArrayAsInt` is little-endian, like QUMO's integers.

```bash
.venv/bin/mojo run -I . examples/frameworks/qsharp/grover_5q.mojo
bench/.venv/bin/python examples/run_examples.py --only examples/frameworks/qsharp
```

| Original | URL | QUMO file | What is checked | Reference numbers (provenance) |
|---|---|---|---|---|
| Grover's search (Grover.qs) | https://github.com/microsoft/qdk/blob/main/samples/algorithms/Grover.qs | `grover_5q.mojo` | marked \|01010>, `grover_optimal_iterations(5, 1)` = 4 (Q#'s formula); exact P(marked) = sin^2(9 asin(1/sqrt 32)) == `grover_success_probability` (1e-12); 1000 shots within 4 sigma and 01010 most frequent; P(k) = sin^2((2k+1) theta) for k = 0..12, maximum at k = 4 | 4 iterations and result [Zero, One, Zero, One, Zero] (page); sin^2(9 theta) = 0.99919 (analytic) |
| Quantum phase estimation (PhaseEstimation.qs) and the iterative phase estimation notebook | https://github.com/microsoft/qdk/blob/main/samples/algorithms/PhaseEstimation.qs and https://github.com/microsoft/qdk/tree/main/samples/notebooks | `phase_estimation.mojo` | U = Rzz(pi/3) (q0,q1) . Rzz(pi/3) (q1,q2) on \|010>: eigenphase pi/3 (1e-12); textbook `qpe` with 6 bits: most likely readout 11 = nearest grid value, \|estimate - pi/3\| <= 2 pi/64; iterative QPE with one re-bound symbolic feedback angle equals the textbook distribution (1e-12); the dynamic-circuit version (mid-circuit measurement, reset, c_if) sampled 4000 times: mode 11, TVD < 0.05 | eigenvalue (1.0, pi/3) (page); estimate 11 * 2 pi/64 = 1.0799 and P(11) = 0.684 (analytic) |
| Hidden shift (HiddenShift.qs) | https://github.com/microsoft/qdk/blob/main/samples/algorithms/HiddenShift.qs | `hidden_shift.mojo` | inner-product bent function on 10 qubits (CZ(i, i+5)), H . O_g . H . O_f* . H: P(s) = 1 (1e-12) and every one of 100 shots returns s | shifts 170, 512, 999 found (page) |
| Resource estimation (ShorRE.qs, estimation-ising-2D.ipynb) | https://github.com/microsoft/qdk/blob/main/samples/estimation/ShorRE.qs and https://github.com/microsoft/qdk/blob/main/samples/estimation/estimation-ising-2D.ipynb | `resource_estimation.mojo` | `estimate_resources` / `estimate_hardware` / `estimate_fault_tolerant` on a controlled x2 mod 21 oracle, the N=21 order-finding circuit, an 8-qubit QFT (rotation level) and one 10x10 Ising Trotter step, for the superconducting, trapped_ion and neutral_atom presets and budgets 1e-2/1e-3/1e-4: T-count == explicit Clifford+T count (Toffoli expansion verified at 1e-12); distance >= 3; physical qubits monotone in the budget; distances ordered by physical error; NISQ durations ordered by the presets | J = g = 1, dt = 0.9 (notebook); the samples print no estimates, so the checks are structural (analytic) |

Out of scope, stated in `resource_estimation.mojo`: count-only (circuit-free)
estimates in the style of RSA-2048 and the QDK's Majorana / floquet-code
qubit models. QUMO's controlled() functor does not take RZZ, so Rzz is written
as CX . RZ . CX, which is also how Q# decomposes it.
