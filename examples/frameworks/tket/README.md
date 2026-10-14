# TKET signature examples, reproduced with QUMO

Four pytket user-guide examples, rebuilt from QUMO's own library (no pytket at
run time) and verified with machine-readable `CHECK` lines. Each file's
docstring cites the original, states every deviation and lists its checks.

```bash
.venv/bin/mojo run -I . examples/frameworks/tket/phase_estimation.mojo
bench/.venv/bin/python examples/run_examples.py --only examples/frameworks/tket
```

| Original | URL | QUMO file | What is checked | Reference numbers (provenance) |
|---|---|---|---|---|
| VQE with UCC ansatz | https://docs.quantinuum.com/tket/user-guide/examples/algorithms_and_protocols/ucc_vqe.html | `ucc_vqe.mojo` (9 checks) | gadget-table ansatz == `h2_ansatz()` (1e-12); BFGS + adjoint reaches -1.1373 Ha (1e-6); compiled circuits equivalent (1e-10) and never larger; 4000-shot QWC estimate within 4 sigma | FCI -1.1373060357534 Ha (analytic, Lanczos); 2-qubit gates 64 -> 31 (generic point) and 64 -> 17 (optimum) in QUMO vs pytket PauliSimp 30 / 14 and + FullPeepholeOptimise 23 / 13 (rerun pytket 2.18.5, `reference/ucc_ref.py`); the page prints no energies or CX counts |
| Phase estimation | https://docs.quantinuum.com/tket/user-guide/examples/algorithms_and_protocols/phase_estimation.html | `phase_estimation.mojo` (7) | U == diag(1, 1, e^{i pi/4}, e^{i pi/8}) (1e-12); P(0001) = 1 and all 1000 shots read 0001; the other eigenstates give 0, 0, 1/8 | readout 0001, theta = 0.0625 (page) |
| Advanced expectation values and measurement reduction | https://docs.quantinuum.com/tket/user-guide/examples/algorithms_and_protocols/measurement_reduction_example.html | `measurement_reduction.mojo` (9) | exact value 0.1 (1e-12); 3 naive / 3 QWC / 2 commuting-set circuits, the second with 1 CX; commuting-set diagonalisation exact on random states (1e-12); 10,000-shot estimates within 4 sigma | 0.1, NonConflictingSets 3, CommutingSets 2 with 1 CX (page; confirmed by `reference/measurement_ref.py`, pytket 2.18.5); shot values 0.10184 / 0.10088 / 0.10388 (page, quoted only) |
| Iterated entanglement swapping (one swap) + conditional gates | https://docs.quantinuum.com/tket/user-guide/examples/algorithms_and_protocols/entanglement_swapping.html and https://docs.quantinuum.com/tket/user-guide/examples/circuit_construction/conditional_gate_example.html | `entanglement_swapping.mojo` (17) | with `c_if` and `c_if_register` corrections <ZZ> = <XX> = 1, <YY> = -1 (4000 shots per basis); without corrections about 0; uniform Bell-measurement outcomes; noisy correlators within 4 sigma of the analytic Pauli-channel values | ~50/50 between 00 and 11 (page); noisy <ZZ> = <XX> = 0.66123, <YY> = -0.59511 (analytic, (1-2r)^k (1-16p/15)^3) |

Optional pytket 2.18.5 cross-checks (`reference/`, run with
`bench/.venv/bin/python`, excluded from the default runner):

| Script | What it does |
|---|---|
| `reference/ucc_ref.py` | the same H2 gadget table as PauliExpBoxes: CX counts after DecomposeBoxes, PauliSimp and FullPeepholeOptimise, plus scipy Nelder-Mead from the page's start |
| `reference/measurement_ref.py` | `measurement_reduction` on the page operator: 3 NonConflictingSets circuits, 2 CommutingSets circuits (CX counts [0, 1]) and exact value 0.1 |

## Library gaps met

- **CommutingSets** (general commutation + Clifford diagonalisation) is not in
  QUMO; `PauliSum.group_qwc` is NonConflictingSets. `measurement_reduction.mojo`
  carries a small in-file version (greedy grouping + CX search with symplectic
  sign tracking). The plan said "3 naive vs 2 QWC"; the page actually has
  3 QWC groups and 2 commuting sets, which is what the example asserts.
- **PauliSimp**: QUMO's `pauli_simp` pass reports "kept original" on this
  gadget circuit (its result was not smaller), and the reduction comes from
  `kak` + `cancel`. pytket's PauliSimp alone reaches 30 / 14 CX.
- **H2 operator**: QUMO's built-in `h2_sto3g` (R = 0.735 A) replaces the page's
  qiskit-aqua operator at 0.75 A.

Conventions: TKET labels basis states big-endian (|q0 q1 ...>); QUMO is
little-endian (bit k of the index = qubit k), so diagonal operators are
written as gates and bitstrings are converted where printed.
