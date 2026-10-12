# Shared benchmark inputs

Normalised OpenQASM 2 versions (measurements/barriers/resets removed, custom gates
unrolled to a common qelib1 subset; see `bench/neutral/prepare_circuits.py`) of
community benchmark circuits, as distributed by Qiskit Benchpress:

- `qb_small__*`, `qb_medium__*` — **QASMBench** (Ang Li et al., PNNL), license in `LICENSES/QASMBench-LICENSE`.
- `feynman__*` — **Feynman** benchmarks (Matthew Amy), license in `LICENSES/Feynman-LICENSE.md`.

Regenerate with `bench/.venv/bin/python bench/neutral/prepare_circuits.py <benchpress>/benchpress/qasm bench/neutral/circuits 30`.
