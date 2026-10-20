# QUMO-AI vs IBM's AI transpiler (qiskit-ibm-transpiler, local mode)

Head-to-head of QUMO's native AI transpiler (`transpile_ai`, docs/AI_TRANSPILER.md)
against the IBM-trained AI passes of
[qiskit-ibm-transpiler](https://github.com/Qiskit/qiskit-ibm-transpiler) 0.18.0 in
**local mode**, plus Qiskit 2.5.2 and plain QUMO, on IBM heavy-hex coupling maps.
Results: `bench/results/ai_vs_ibm.{jsonl,md}`; interpretation:
docs/AI_TRANSPILER.md, section *Head-to-head vs qiskit-ibm-transpiler*.

IBM's package, its dependencies (torch, qiskit-gym, twisterl) and IBM's model
weights live **only in a scratch venv / cache** created by `setup_venv.sh`; they
are never a QUMO dependency and are never committed. No IBM Quantum cloud
service or credential is used: every IBM pass runs with `local_mode=True`, the
models are read from a local Hugging Face cache (`HF_HUB_OFFLINE=1`), and the
benchmark workers refuse every non-local socket connection.

## Reproduce

```bash
bench/ai_vs_ibm/setup_venv.sh /tmp/ibm_ai          # venv + IBM models (~70 MB) + make py if needed
export HF_HOME=/tmp/ibm_ai/hf_home HF_HUB_OFFLINE=1 IBMAI_VENV=/tmp/ibm_ai/venv
bench/ai_vs_ibm/campaign.sh bench/results/ai_vs_ibm.jsonl    # ~2 h, per-phase deadlines
$IBMAI_VENV/bin/python bench/ai_vs_ibm/report.py bench/results/ai_vs_ibm.jsonl > bench/results/ai_vs_ibm.md
```

Benchpress circuits are read from a Benchpress checkout (`--benchpress`, default
`.ref/benchpress`; `git clone https://github.com/Qiskit/benchpress .ref/benchpress`).
`QUMO_PY_DIR` selects the directory holding `qumo_py.so` (default `build/`).
Single phases: `run.py --groups families|summit|feynman|hamlib|neutral
--devices fake_torino|fake_fez|hh --contenders main|all|<list> --out x.jsonl`
(resumable: finished records are skipped). The IBM baseline is also available
to the neutral harness as the optional adapter `qiskit+ibm-ai`
(`bench/neutral/run_neutral.py --sdks qumo-ai,qiskit+ibm-ai --targets sc_heavy_hex,linear_cx`),
which reports `unsupported` when the package is not installed.

## Method

* **Same inputs**: every circuit is built once with fixed seeds (`circuits.py`)
  and normalised to one OpenQASM 2 text (no measurements, common qelib1 gate
  set) that every contender parses.
* **Same target**: the device coupling map of qiskit-ibm-runtime's fake backend
  (`fake_torino` 133 q, `fake_fez` 156 q) or the neutral harness' 19-qubit
  heavy-hex (`hh`), basis `rz, sx, x, cz`; no calibration / error data for anyone.
* **Contenders**: `qiskit-o2/o3` (Qiskit `transpile`), `ibm-ai2/ai3`
  (`generate_ai_pass_manager(optimization_level=L, ai_optimization_level=L,
  ai_layout_mode="optimize")`, IBM's documented hybrid pass manager), `ibm-route3`
  (AIRouting only), `ibm-full3` (AIRouting + every documented Collect* /
  AI*Synthesis pass), `ibm-synth3` (Qiskit Sabre + that synthesis list),
  `qumo-o2/o3`, `qumo-ai2/ai3` (`transpile_ai`), `qumo-o3s` / `qumo-ai3s`
  (single pipelines, no portfolio), `qumo-route3` (learned routing only),
  `qumo-synth3` (synthesis only). The attribution contenders run on the
  `families` group.
* **Fair timing**: persistent per-family worker processes (`spawn`), warmed up
  once (IBM model loading is not timed), contenders interleaved per circuit and
  repetition with a rotated order, 1-minute load average stored per record.
  Repetitions use seeds 11, 12, 13 for everyone; IBM's learned routing is
  unseeded (non-deterministic), the repetitions measure its spread.
* **Verification of every output** (parent process, after compilation): ISA
  validity (cz only, on coupling edges); then equivalence with the reported
  layouts: statevector on a random product input (numpy <= 14 logical qubits,
  Qiskit Aer <= 20 logical / 24 active), Clifford originals of any width by
  stabiliser-tableau conjugation (fallback: exact Pauli-sum propagation).
  If the check fails with the reported layouts, the same check is repeated with
  the input placement inferred from the circuit (the state is run backwards and
  each logical input state located); a pass then counts as verified and the
  record carries `initial_layout_mismatch`. Larger non-Clifford circuits are
  ISA-checked only (the Benchpress standard) and reported separately, never as
  wins.

## IBM issues found while setting this up (0.18.0, Qiskit 2.5.2)

1. `import qiskit_ibm_transpiler` fails: it imports
   `qiskit.synthesis.linear.linear_matrix_utils`, removed in Qiskit 2.x. The
   adapter aliases the module to `qiskit.synthesis.linear` (the function used,
   `random_invertible_binary_matrix`, only serves IBM's random-circuit helpers).
2. Any input with a SWAP fails in `generate_ai_pass_manager` at levels 2 / 3:
   `ElidePermutations` (Qiskit's init stage) records a virtual permutation keyed
   by the original qubits, AIRouting overwrites `original_qubit_indices` with
   the routed DAG's new qubits and `TranspileLayout.from_property_set` raises
   `KeyError: 'The item <Qubit register=(19, "q"), index=0> does not exist in the Layout'`.
   Repro:
   ```python
   from qiskit import QuantumCircuit
   from qiskit.transpiler import CouplingMap
   from qiskit_ibm_transpiler import generate_ai_pass_manager
   qc = QuantumCircuit(4); qc.h(0); qc.cx(0, 1); qc.swap(1, 2); qc.cx(2, 3); qc.cx(0, 3)
   generate_ai_pass_manager(optimization_level=3, ai_optimization_level=3,
                            coupling_map=CouplingMap.from_heavy_hex(3)).run(qc)   # KeyError
   ```
   Workaround in `bench/neutral/sdks.py`: drop `ElidePermutations` from the init
   stage of the AIRouting pipelines (input SWAPs are then routed, not relabelled).
3. AIRouting reports a wrong **initial** layout (`circuit.layout`) whenever
   the layout it chose is not an involution: the routed circuit is exact and the
   final layout is right, but the reported initial layout is the *inverse*
   permutation of the one the circuit uses (on every misreported case measured,
   the true physical qubit of logical v is the position of v in the full
   reported `initial_index_layout()`). Executions from |0...0> are unaffected;
   anything that relies on the initial layout (composing circuits, preparing
   inputs, mapping initial-state observables) is not. Repro: route a 4-qubit
   `random_clifford(4, seed=501).to_circuit()` with `generate_ai_pass_manager(3, 3,
   coupling_map=CouplingMap.from_line(6))` and simulate with the reported layouts
   (the routing is unseeded, so a run may pick an involutive layout and pass).
   `sdks.compile_qiskit_ibm_ai` corrects the layout when AIRouting ran
   (`ibm_initial_layout_corrected` in the record); `run.py` additionally
   re-checks any failed equivalence with the input placement inferred from the
   circuit, for every contender (`initial_layout_mismatch`).
4. On macOS, instantiating a qiskit-ibm-runtime fake backend (which imports
   qiskit-aer) before IBM's torch-based synthesis models load segfaults the
   process (two OpenMP runtimes). The harness builds targets in the parent and
   keeps Aer out of the IBM worker.
