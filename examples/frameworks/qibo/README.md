# Qibo signature examples, reproduced with QUMO

Three Qibo examples, rebuilt from QUMO's own library and verified with
machine-readable `CHECK` lines. Qibo is not installed; the Qibo pages print
plots rather than numbers, so the checks are against analytic values and
exact simulation.

```bash
.venv/bin/mojo run -I . examples/frameworks/qibo/qft.mojo
bench/.venv/bin/python examples/run_examples.py --only examples/frameworks/qibo
# GPU: the n = 20 round trip also runs on the GPU state vector (fp32)
.venv/bin/mojo run -I . -D QUMO_GPU examples/frameworks/qibo/qft.mojo
```

| Original | URL | QUMO file | What is checked | Reference numbers (provenance) |
|---|---|---|---|---|
| How to write a Quantum Fourier Transform? | https://qibo.science/qibo/stable/code-examples/examples.html | `qft.mojo` (6 checks) | QFT(15)\|0> is uniform (1e-12, the page instance); `qft(n)` == classical DFT on seeded random states, n = 5..10 (1e-12); approximate-QFT fidelity non-decreasing as the degree falls, 1 at degree 0; n = 20 iQFT(QFT\|k>) reads k on every shot, on the CPU and on the GPU route | analytic. Measured: DFT diff 2.0e-16; AQFT fidelity (n = 10) 0.0033 at degree 9 -> 0.978 at 5 -> 0.99999 at 1; with -D QUMO_GPU the GPU row runs `gpu:statevector:fp32` |
| Adiabatic evolution | https://qibo.science/qibo/stable/code-examples/tutorials/adiabatic/README.html | `adiabatic_evolution.mojo` (5) | H(s) = (1-s) H0 + s H1 from `PauliSum` `+` / `scaled`, one Strang `trotter` step per slice: ground-state fidelity rises with T and >= 0.9 at T = 8; energy within 1%; Trotter vs exact slices < 1e-3; Trotter infidelity ~ dt^4 (slope 4 +/- 0.4) | page (plots only): best overlap among its T at T = 4; Trotter error ~ dt^4. Measured: fidelity 0.727, 0.850, 0.988, 0.9986, 0.9997 at T = 0.5..8; slope 4.015 |
| Variational Quantum Regressor | https://qibo.science/qibo/stable/code-examples/tutorials/vqregressor/README.html | `vq_regressor.mojo` (3) | parameter-shift loss gradient == adjoint (1e-12); training MSE decreases; test MSE < 0.01 on a 100-point grid | page settings (30 points, Adam lr 0.05, 200 epochs, PSR); no printed numbers. Measured test MSE 1.3e-6 |

## Library gaps met

- **Two-symbol angles**: QUMO's `Param` is affine in ONE symbol
  (coeff * theta + offset), so the re-uploading angle w x + b is written as
  RY(w x) RY(b) (same axis, so they compose exactly). The circuit is rebuilt
  per data point, with x folded into the coefficient of w.
- **Time-dependent Hamiltonians**: there is no built-in adiabatic schedule.
  The example uses a piecewise-constant H(s) at slice midpoints (a
  second-order method), built from `PauliSum` arithmetic.
- **Approximate QFT**: QUMO's `approximation_degree` follows Qiskit's
  semantics (drop the smallest-angle CP rotations); Qibo's QFT has no such
  option, so this part goes beyond the page.
