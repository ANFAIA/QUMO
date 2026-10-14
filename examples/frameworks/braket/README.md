# Amazon Braket signature examples, reproduced with QUMO

Four notebooks from the Amazon Braket examples and algorithm library, rebuilt
with QUMO (Braket is not needed at run time) and verified with
machine-readable `CHECK` lines. Braket writes bitstrings with character i =
qubit i; QUMO integers are little-endian and its keys put the highest qubit
first, so each example states how it translates the page's strings.

```bash
.venv/bin/mojo run -I . examples/frameworks/braket/simon_algorithm.mojo
bench/.venv/bin/python examples/run_examples.py --only examples/frameworks/braket
```

| Original | URL | QUMO file | What is checked | Reference numbers (provenance) |
|---|---|---|---|---|
| Simon's Algorithm | https://github.com/amazon-braket/amazon-braket-examples/tree/main/examples/advanced_circuits_algorithms/Simons_Algorithm | `simon_algorithm.mojo` | 2-to-1 oracle from CNOT copies + s-conditioned CNOTs, 4n = 24 shots, GF(2) elimination written in the example; y.s = 0 on every shot; s recovered for 5 secrets (incl. s = 0); exact distribution uniform over s-perp (1e-12) | secret "101011" recovered (page); P(y) = 2^-(n-1) (analytic) |
| Deutsch-Jozsa and Bernstein-Vazirani (algorithm library) | https://github.com/amazon-braket/amazon-braket-algorithm-library/tree/main/notebooks/textbook | `deutsch_jozsa_bernstein_vazirani.mojo` | `deutsch_jozsa` / `bernstein_vazirani` for n = 3..12, exact and 1000 shots; compiled for `grid:3x4/sc_cz` and `trapped_ion:12`: conformance and layout-aware equivalence (overlap 1 within 1e-10, exact phase) | P("000") = 1.0 for the constant n = 3 oracle; BV secret "1010" (page); P = 1 / 0 (analytic) |
| Simulating noise on Amazon Braket | https://github.com/amazon-braket/amazon-braket-examples/tree/main/examples/braket_features | `noise_simulation.mojo` | Bell + BitFlip(0.1) on every gate; GHZ-3/5 with bit flip, phase flip, depolarizing (p = 0.05, `apply_gate_noise` semantics) and a `NoiseModel`; 50k trajectory shots vs an exact Pauli-propagation reference: TVD < 0.01; <X^n> within 4 sigma; closed forms; zero noise is ideal | counts 424/408/87/81 (page, within 4 sigma of exact 0.41/0.41/0.09/0.09); <X^n> = (1-2p)^(2n-1) (phase flip) and (1-4p/3)^(2n-1) (depolarizing) (analytic) |
| Zero noise extrapolation with Mitiq (error_mitigation/on_mitiq) | https://github.com/amazon-braket/amazon-braket-examples/tree/main/examples/error_mitigation/on_mitiq | `zne_mitigation.mojo` | 5-qubit TFIM Trotter circuit (RX + CNOT-RZ-CNOT gadgets as in the notebook), mean <Z>, superconducting-preset gate errors, scales [1, 3, 5], global and random folding, Richardson / linear / exponential extrapolation (`qumo.algorithms.mitigation`); folding equivalence (1e-12); \|ZNE - ideal\| < 0.5 \|raw - ideal\| | ideal M = 0.5966 (exact state vector); measured raw 0.5633, Richardson 0.6055, linear 0.5908, exponential 0.5966 (error ratio 0.27). The notebook's emulator noise and 8-qubit instance are not reproduced (it prints no mitigated numbers) |

Unsupported, stated in `noise_simulation.mojo`: amplitude damping, generalized
amplitude damping and phase damping as non-Pauli channels, and general Kraus
operators. QUMO's trajectories are Pauli-only; `NoiseModel` represents T1/T2
by their Pauli twirl.
