# Project review: architecture, ease of use and performance

Date: 2026-11-12. Reviewed tree: `origin/main` at `0db0c24`. Work described as "done" is in branch
`project-review`, commit `96ee64b` (local, not pushed).

## 1. Goal and method

The question was whether QUMO is useful and easy to handle for researchers, academics, students and
hobby developers, while keeping its commitment to performance.

Three independent audits ran on a clean checkout of `main`, each hands-on and read-only:

| Audit | What it did |
|---|---|
| Newcomer | Followed the README literally, tried typical first tasks in the CLI, Python and Mojo, made beginner mistakes on purpose |
| Researcher | Ran research tasks end to end (expectation sweep, H2 VQE, noisy simulation, surface-code sweep, compile to a custom target); checked interop, extensibility and scholarly needs |
| Architecture and performance | Mapped dependencies between packages, compared the same operation across the four surfaces, measured the cost of the convenience layers |

Three more agents then fixed the cheap problems on disjoint files, and a fourth integrated and verified
the result.

## 2. Verdict

The core and the performance are sound. The problem was the entrance: the fast engine sat behind a
Python surface with no builder, no keyword arguments and no docstrings, a CLI that ignored typos, and
no learning path.

The entrance is now fixed in `project-review`. What remains is structural (section 6) and needs the
owner's decision.

## 3. Findings

### 3.1 Performance: convenience layers are free

Measured on an Apple M5, binaries built from the reviewed tree:

| Case | Direct Mojo | `sample_auto` | Python | CLI (whole process) |
|---|---|---|---|---|
| Bell, 1000 shots | 28 us | 28 us | 28 us | 5.0 ms |
| 20 qubits, 10 layers, 1000 shots | 21.4 ms | 21.3 ms | 22.3 ms | 29.5 ms |

Fixed costs: process start about 5 ms, QASM parse 0.08 to 0.3 ms, `import qumo_py` about 24 ms.
From Python, a 201-point expectation sweep took 0.7 ms, an H2 VQE 1.8 ms and a surface-code sweep
(d = 3, 5, 7; four error rates; 20,000 shots) 0.37 s.

So there was room to make the API easier with no cost to the hot paths.

### 3.2 Architecture

Strengths:

- The layers in `docs/ARCHITECTURE.md` match the code. `qumo/ir` and `qumo/net` import nothing from
  other packages.
- One tool registry serves both the CLI and the MCP server (24 tools), with stable `QM-Exxx` codes.
- Runtime dependencies are three libraries; the CLI binary is 5.5 MB.
- Benchmark reports state the date, the hardware and the command to reproduce them, and disclose
  tuning bias.

Problems:

| # | Problem | Evidence |
|---|---|---|
| A1 | No front door | Every `qumo/*/__init__.mojo` was docstring-only; a first Mojo program needed four deep imports |
| A2 | The same operation differs across surfaces | A 40-qubit GHZ ran in the CLI but failed in Python `q.sample` (QM-E404); `transpile` had no CLI subcommand; no state vector in Python |
| A3 | Dependency cycles | `qumo.ai.json` is imported by compiler, gpu, ft and exec, while `ai` imports them; `sim` and `compiler` import each other; `exec/tools` and `ai/registry` import each other |
| A4 | Hardware access exists twice | `qumo/exec/{ibm,ionq,quantinuum,speqtrum}_api.mojo` (2,832 lines) and `plugins/hardware/qumo_hardware` (5,099 lines of Python) |
| A5 | Top level is hard to read | Twelve top-level directories; `rl/` (32k lines) is nearly half the size of the core (74k); `training/` holds only a README; `qumo/net`, `qumo/bench` and `qumo/testing` were missing from the README |
| A6 | Distribution is blocked | The binary carries an absolute rpath to the checkout's `.venv`; no wheel, Docker image or CI; install is clone + `make setup` (about 850 MB) + 60 s build |
| A7 | Stale or wrong claims | README said the native Mojo agent harness is on `main` (it is in open PR #7); benchmark rows were measured with Mojo 1.1 and the toolchain is now 1.2 |
| A8 | Large files and leftovers | `qumo/ir/qasm.mojo` is 4,995 lines; test names such as `test_program_review2.mojo`; `rl/qrl/json.mojo` is a copy of `qumo/ai/json.mojo` |
| A9 | Compile time | 60 s for the CLI, 60 s for the extension; a small user script costs 5 s per `mojo run` (about 20 s the first time) |

### 3.3 Ease of use

What already worked: every command and snippet in the README ran as written; the CLI gave a Bell
histogram with one command; errors carry a code, line, column, caret and hint; the Mojo builder API
could be guessed from the docs.

Problems:

| # | Problem | Evidence |
|---|---|---|
| U1 | Python could not build a circuit | `q.circuit(2).h(0)` raised `AttributeError`; every circuit had to be a QASM string |
| U2 | Python was not discoverable | `help(q.transpile)` printed `transpile(...)`; 1 of 88 names had a docstring |
| U3 | Python was positional-only | `q.sample(b, shots=1000)` raised "takes no keyword arguments"; `transpile` took six unnamed arguments and no target string |
| U4 | Silently wrong results | `expectation` ignored noise (1.0 instead of 0.9467); `inject_noise` did nothing on one-line QASM; `depth_of(['cz'])` returned 0; `basis='sc_cz'` was read per character |
| U5 | Research drivers missing in Python | `vqe`, `qaoa_maxcut`, `h2_sto3g`, `memory_experiment`, `threshold_sweep` existed only in Mojo |
| U6 | CLI ignored unknown options | `run bell.qasm --shot 10` ran 1024 shots and returned 0 |
| U7 | No learning path | No tutorial, no table from Qiskit or Cirq, no teaching notebook |
| U8 | No scholarly scaffolding | No `CITATION.cff`, changelog, contributing guide, CI or `v0.1.0` tag |
| U9 | Extension points undocumented | Passes and decoders are closed tables in core files; the Kraus pragma was only discoverable from a parser error |
| U10 | Platform limits unstated | Mojo 1.2.0 ships wheels only for macOS arm64 and Linux with glibc 2.34+ |

## 4. Changes made (branch `project-review`, commit `96ee64b`)

### 4.1 Python package `qumo`

```python
import qumo
bell = qumo.Circuit(2).h(0).cx(0, 1).measure_all()
print(qumo.sample(bell, shots=1000))
```

- `Circuit` builder with every simple gate of the Mojo builder, parameters by name, pickling,
  equality and notebook display.
- `sample`, `statevector` (NumPy), `expectation`, `transpile(c, "grid:2x2/sc_cz")` with keyword
  arguments and the CLI's defaults.
- `vqe`, `qaoa_maxcut`, five Hamiltonian builders, seven library circuits, `memory_experiment`,
  `threshold_sweep`.
- `QumoError` with `.code` and `.hint`.
- Docstrings on all 119 functions and 46 methods of `qumo_py`.
- Install: `make py && pip install -e .` (editable only, see A6).
- The four silent-wrong-result bugs of U4 are fixed, plus argument checks in `sample` and
  `decode_dem`.

Measured cost: 0.6 to 2 us per call (Bell: 29.4 us raw, 30.1 us through the package; 20 qubits: equal
within noise). Building costs 0.3 to 0.4 us per gate and the circuit crosses to native code in one
call.

### 4.2 CLI

- Unknown options fail with `QM-E921`, exit code 2 and a did-you-mean hint.
- New subcommands `qumo transpile`, `qumo emulate` and `qumo vqe`, thin adapters over existing tools.
- `qumo --help` has a first-steps block; every subcommand has `--help`;
  `qumo tool <name> --help` prints arguments, types, defaults and an example.
- `validate` reports up to 20 independent statement-level errors.

### 4.3 Mojo

- `qumo/prelude.mojo`: `from qumo.prelude import *` gives `Circuit`, `parse_qasm`, `transpile`,
  `sample_auto`, `statevector`, `expectation` and the result types. No measurable compile-time cost.
- `print(statevector)` labels amplitudes with bitstrings, as the CLI does.
- `SampleResult.to_json()` and `VQEResult.to_json()`.
- Automatic sampling of a Clifford circuit without measurements raises QM-E402 instead of returning
  `{'': shots}`.

### 4.4 Documentation and repository

- New: `docs/TUTORIAL.md` (first hour, in CLI, Python and Mojo), `docs/FROM_QISKIT.md`,
  `docs/EXTENDING.md`, `docs/PYTHON.md`.
- New: `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.github/workflows/ci.yml`, issue templates.
- README: three routes by audience, a core / optional / research directory table, corrected claims
  (A7), platform limits (U10).
- `make help`, `make test-python`, `make install-py`.

### 4.5 Verification

| Check | Result |
|---|---|
| `make test` (full) | 90 pass, 0 fail |
| `make examples` | 109 pass, 1 skipped (torch not installed) |
| Python package tests | 116 pass |
| Plugin and example Python tests | 552 pass, 5 skipped |
| Snippets in README, GETTING_STARTED and TUTORIAL | all executed, 0 mismatches |
| `build/qumo` commands quoted in the docs | all 49 executed |

Not verified: the CI workflow has never run on GitHub; Linux was not run; the PennyLane column of
`FROM_QISKIT.md` was not executed; the core-edit procedures in `EXTENDING.md` (named pass, named
decoder) were read from source, not exercised.

### 4.6 Behaviour changes that can break existing scripts

- CLI: unknown options are errors, including `--quiet`, which used to be accepted and did nothing.
  `qumo help <unknown>` exits 2.
- `qumo_py`: `depth_of` with an unknown gate raises; `sample` rejects non-integer shots; `decode_dem`
  rejects unknown decoders; `inject_noise` raises when it cannot address an operand; a string basis is
  one spec.
- Automatic sampling without measurements raises QM-E402.

## 5. Status of each finding

| Finding | Status |
|---|---|
| A1 front door | Done (prelude, Python package) |
| A2 surface consistency | Mostly done; default seeds not unified (item 7, owner decision) |
| A3 dependency cycles | Mostly done (item 2): json moved to `qumo/core`, `sim` no longer imports `compiler`. One cycle remains: `exec/tools` and `ft/tools` import `qumo.ai`; moving them is an owner decision |
| A4 duplicated hardware layer | Plan written (`docs/HARDWARE_LAYER_PLAN.md`); not implemented (item 3, owner decision) |
| A5 top-level layout | Done in the docs (item 4); directories not moved |
| A6 distribution | Editable install done. Relocatable binary spiked and works; wheel not built: the licence needs confirmation (item 1) |
| A7 stale claims | Text corrected. Re-measuring not done: machine was under load (item 5) |
| A8 large files and leftovers | `qasm.mojo` split into lexer, parser and emitter (facade kept); test files renamed. `QasmParser` itself is still one 3.6k-line file (Mojo cannot split struct methods across files) |
| A9 compile time | Documented; not reducible from this project |
| U1 to U8, U10 | Done (U8: CI and tags prepared; `v0.1.0` tag left to the owner) |
| U9 extension points | Documented; registry not done (item 6): it needs a function-pointer table and touches several files |

## 6. Changes still to make

Ordered by value for the target audience.

| # | Change | Why | Effort | Performance risk |
|---|---|---|---|---|
| 1 | Relocatable binary and a wheel bundling `qumo_py.so` with the runtime libraries (`@loader_path` rpath); then a Colab teaching notebook | Today the only install is clone, 850 MB toolchain and a 60 s build; this is the largest remaining barrier for students | Medium | None. Blocked on Modular's licence for redistributing the runtime libraries: check first |
| 2 | Move `json` and diagnostics to `qumo/core`; move `GateSet` and `expand_rule` down to `ir` | Removes the cycles of A3 | Medium, mechanical | None at run time |
| 3 | One hardware layer instead of two | A4: every provider is maintained twice | Large | None |
| 4 | Group `rl/`, `models/` under one research heading in the docs; remove `training/`, which held only a pointer README (corrected: it was not empty) | A5 | Small | None |
| 5 | Re-run the benchmarks with Mojo 1.2 on an idle machine and re-date the README table | A7: the Aer comparison ran with load average 27 to 45 | Small | None |
| 6 | Registry for compiler passes and decoders | U9: adding one means editing core files | Medium | Risky if done with dynamic dispatch: keep static dispatch for the built-ins |
| 7 | One default seed across surfaces (sampling uses 1234, transpile and VQE 7, QEC 1) | A2 | Small | None, but it shifts pinned compiler and optimizer results and the goldens |
| 8 | Python gaps: `qumo.emulate`, `qumo.grover`, complex coefficients in `expectation` (Qiskit `SparsePauliOp.to_list()`), Kraus channels and detectors in the builder, zero-copy `statevector`; CLI `transpile --edges/--calibration` | Small holes a newcomer or researcher hits next | Small each | Zero-copy state vector needs care: the simulator keeps real and imaginary parts in separate arrays |
| 9 | Split `qumo/ir/qasm.mojo` into lexer, parser and emitter; rename the `test_program_review*` files. (Corrected: keep the JSON copy in `rl/qrl/json.mojo`; `rl/` must not import `qumo.*`, and `make check-independence` enforces it.) | A8 | Medium | None |
| 10 | Tag `v0.1.0`; run the CI workflow once and fix what fails; pin `bench/.venv` with a lock file | U8: nothing citable or reproducible by version yet | Small | None |
| 11 | Verify on a Linux host | Everything Linux-specific is unverified | Small with a host | None |

## 7. Decisions for the owner

1. Push `project-review` and open the pull request.
2. `qumo/__init__.py`: a one-file shim inside the Mojo package so `import qumo` works from the
   repository root without installing. The Mojo compiler ignores it. Alternatives: only a clear error
   message, or renaming the Python package.
3. `CITATION.cff` names the author as "Ismael" / "Faro Sertage", with no ORCID and no release date.
4. Which items of section 6 to do, and in what order. Recommended first: 10, 5 and 8 (small), then 1
   (largest effect on adoption, after the licence check).
5. PR #7 (native Mojo agent harness): a separate review recommends not merging it as is, because
   `run_check` runs unconfined without an open sandbox and the branch duplicates the network stack
   already on `main`.

## 8. Applied on 2026-11-12 (section 6, batches A to D)

| Item | Result |
|---|---|
| 1 relocatable binary, wheel | Licence reviewed (from a summarised page: confirm the text verbatim). Spike: binary with `@loader_path/lib` runs from a directory outside the repo with the libraries loaded from the bundled folder. No wheel built. Licence term to check before any wheel: MAX may not be used as training data or input to any AI system (s1.4(4)) |
| 2 move json to `qumo/core`, GateSet to `ir` | Done. `sim` no longer imports `compiler`. Remaining cycle: `exec/tools` and `ft/tools` import `qumo.ai` (owner decision) |
| 3 one hardware layer | Plan in `docs/HARDWARE_LAYER_PLAN.md`. No code changed: a parity test needs a definition of equivalence per provider |
| 4 group research directories | Docs grouped; `training/` removed (held only a pointer README); `rl/` and `models/` not moved, because `rl/` must not be edited and many paths would change |
| 5 re-measure benchmarks | Not run: machine load stayed above the idle threshold. Note: README says 2.5 to 4.7x against Aer; `docs/SIMULATION.md` table says 1.4 to 4.05x at 24 to 26 qubits. Reconcile when re-measuring |
| 6 pass and decoder registry | Not done: needs a function-pointer table and edits in about ten places (`pass_manager.mojo`, `experiments.mojo`, diagnostics, tool enum, `qec.py`, `qumo_py.mojo`) |
| 7 one default seed | Not done: nine seeds in use (1234, 7, 1, 2024, 12345, 0 and others). Changing one changes 144 golden hashes in `tests/fixtures/golden/manifest.txt` and routed circuits. Owner must pick a value and approve regenerating goldens |
| 8 Python gaps | Done: `qumo.emulate`, complex coefficients accepted when the imaginary part is zero, `kraus1`/`kraus2` on the builder. Not done: zero-copy `statevector` (lifetime not proven safe), `qumo.grover`, detectors in the builder, CLI `transpile --edges/--calibration` |
| 9 split qasm, rename tests | Done: `qasm.mojo` is a 48-line facade over `qasm_lexer.mojo`, `qasm_parser.mojo` and `qasm_emit.mojo`; test files renamed. JSON copy in `rl/` kept |
| 10 tag, CI, lock | Lock file `bench/requirements.lock` and `make setup-py-locked` added. CI commands checked locally where the machine allowed. Tag `v0.1.0` not created: version identity is the owner's decision. CI has not run on GitHub |
| 11 Linux host | Not possible here |

Verification on the integrated tree: `make cli py` builds; `make check-independence` OK; `make test-python` 129 passed;
`examples/run_examples.py --check-index` OK. Full `make test`: 90 PASS, 0 FAIL (exit 0).
