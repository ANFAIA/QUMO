# Changelog

All notable changes to QUMO are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

QUMO is a research prototype. Its version string is `0.1.0` (`build/qumo version`) and has not changed since the first commit, and there is no tagged release yet. Until there is one, entries are grouped under "Unreleased" by pull request and date, newest first. Entries are taken from the titles and descriptions of the merged pull requests and from the commit history before pull requests were used; nothing here was added from memory.

## [Unreleased]

### Added

- `agents/agents.local.json` (git-ignored, example in `agents/agents.local.example.json`): per-machine harness and models for the agents, laid over the shared `agents.json`. `"claude_code": {"auth": "subscription"}` keeps an exported `ANTHROPIC_API_KEY` from turning a Claude Code subscription run into a metered one.
- `qumo-agents --backend mojo`: run any agent on QUMO's own native harness (`build/qumo-agent`) with its LLM on OpenRouter, chosen per run or per agent in `agents/agents.json` like the other back ends. `doctor` reports whether the harness is built.
- `examples/optimization/qaoa_cvar.mojo`: QAOA with the CVaR objective, 14 checks, including the case where CVaR does not help.
- CVaR objective for VQE and QAOA: `vqe(..., alpha=)`, `qaoa_maxcut(..., alpha=)`, `CVaRObjective` and `cvar_of_energies` (`qumo/algorithms/variational.mojo`, `docs/ALGORITHMS.md`). `alpha = 1` is the old mean and stays the default; `alpha < 1` needs a diagonal Hamiltonian.
- MCP tool annotations are per tool (`ToolHints`, `ToolRegistry.add_read_only`). The 24 shipped tools keep their read-only hints; a tool added without hints now gets the MCP default (not read-only, open world) instead of being announced as read-only.
- `tests/test_qasm_qiskit26.mojo`: regression tests drawn from bugs fixed in Qiskit 2.6.x (power on controlled gates, expression domain errors, array parameters, constant names).
- Beginner Mojo examples: `examples/foundations/hello_bell.mojo` (a Bell pair in five commented steps) and `examples/foundations/ir_only.mojo` (the circuit IR on its own, with full `qumo.ir` imports). The basic examples (`grover`, `qaoa_maxcut`, `vqe_h2`, `qec_memory_experiment`) have a header that says what to look for, comments per step and the short `from qumo import ...` where it applies; their numbers are unchanged. The tutorial and the basic doc snippets use the root import.
- `qumo/ir/__init__.mojo` documents the IR package completely (module by module) and the root redirect.
- The Python package `qumo` (`python/qumo`, `pyproject.toml`, `docs/PYTHON.md`): a circuit builder with the gate methods of the Mojo `Circuit`, `sample`, `statevector`, `probabilities`, `expectation`, `transpile` with the one-line target specs, `vqe`, `qaoa_maxcut`, Hamiltonian and ansatz builders, `memory_experiment`, `threshold_sweep`; keyword arguments, NumPy arrays, `QumoError` with `.code` and `.hint`, Jupyter display, pickling. Install with `make py && pip install -e .` (`make install-py`); tests in `python/tests` (`make test-python`). The extension `qumo_py` stays underneath as `qumo.native`; `qumo/__init__.py` makes `import qumo` work at the repository root without installing. (project review, 2026-11-10; not yet merged when this entry was written, as the next four entries)
- CLI commands `qumo transpile`, `qumo emulate` and `qumo vqe`, with human-readable output and `--json`; `qumo expect --observable` (repeatable); `qumo tool <name> --help`, generated from the tool's schema; in `qumo tool`, the `qumo_` prefix is optional and a tool that takes `qasm` also takes `"file": "path"` (CLI only).
- `from qumo import *` (or `from qumo import Circuit, sample_auto`): the package root re-exports the names of `qumo.prelude`, so no module path is needed for a first program.
- `qumo.prelude`: `from qumo.prelude import *` brings the names a first Mojo program needs (`Circuit`, `parse_qasm`, `sample_auto`, `statevector`, `expectation`, `parse_target`, `transpile`, ...).
- `SampleResult.to_json()` and `VQEResult.to_json()`; `print(statevector)` lists the nonzero amplitudes by bitstring.
- `qumo validate` / `qumo_validate_qasm` report every error the parser can reach, with `errors_complete`; `qumo_emulate` also returns `shots`, `seed`, `num_qubits` and `compiled`.
- A learning path and scholarly scaffolding: `docs/TUTORIAL.md`, `docs/FROM_QISKIT.md`, `docs/EXTENDING.md`, `CITATION.cff`, `CONTRIBUTING.md`, this changelog, issue templates, a CI workflow (not yet run) and `make help`. (project review, 2026-11-10; not yet merged when this entry was written)
- Native Mojo hardware clients for IonQ, SpeQtrum, IBM Quantum and Quantinuum Nexus inside the `qumo_py` extension, with QUMO's own HTTP/1.1 + TLS client in `qumo/net`. The vendor Python SDKs stay as an explicit fallback (`--client native|python`). ([#8](https://github.com/ANFAIA/QUMO/pull/8), 2026-11-09)
- `rl/`: a native Mojo stack that trains the AI-transpiler models (synthesis and routing policies), independent of `qumo.*`. `make train-ai` no longer needs Python, Rust or torch. ([#6](https://github.com/ANFAIA/QUMO/pull/6), 2026-11-09)
- `docs/GETTING_STARTED.md` and the documentation index `docs/README.md`. ([#5](https://github.com/ANFAIA/QUMO/pull/5), 2026-11-09)
- 21 more Classiq examples; `examples/CATALOG.md`; `run_examples.py --framework NAME`. ([#4](https://github.com/ANFAIA/QUMO/pull/4), 2026-11-05)
- The AI agent layer (`agents/`, `bin/qumo-agents`, `docs/AGENTS.md`): nine agents built on deepagents. Benchmark charts of compile time and memory per SDK (`make bench-charts`). Classiq examples. ([#3](https://github.com/ANFAIA/QUMO/pull/3), 2026-11-05)
- Four features taken from qilisdk under QUMO names: QUBO models and conversion to Ising (`qumo/algorithms/qubo.mojo`), analog schedules (`qumo/algorithms/schedule.mojo`), angle perturbations (`qumo/sim/perturbation.mojo`), flow-style YAML for Pauli sums; and a SpeQtrum hardware provider. ([#2](https://github.com/ANFAIA/QUMO/pull/2), 2026-11-03)
- `Program`: `while`, `break`, `continue`, classical variables and a derived control-flow graph above `Circuit` (`docs/CONTROL_FLOW.md`). ([#1](https://github.com/ANFAIA/QUMO/pull/1), 2026-11-03)

### Changed

- CLI options are strict: an option a command does not declare is a usage error (QM-E921, exit code 2) with a "did you mean" hint. Unknown options were ignored before, so a misspelt `--shot 10` ran with the default. `--quiet`, which was accepted and did nothing, is gone. `qumo help <unknown command>` exits with code 2 (QM-E912). The CLI moved to `cli/app.mojo` (`run_cli`, testable in process); `cli/main.mojo` is the entry file. (project review, 2026-11-10; not yet merged, as the next two entries)
- `sample_auto` / `sample_on` (Mojo) and `qumo_py.sample_device` raise QM-E402 for a circuit without measurements whichever simulator is picked. A Clifford circuit routed to the tableau or Pauli-frame sampler returned `{"": shots}` before. The CLI and the MCP tools are unchanged: they measure all qubits and add the note QM-W004.
- `qumo_py` argument checks, all of which returned a wrong or empty result before: `Circuit.depth_of` raises QM-E913 for an unknown gate name and QM-E911 for a wrong type (it returned 0); a `str` basis in `transpile` is one spec, a basis-set name such as `"sc_cz"` or a comma list, not a sequence of characters; `sample` and `sample_device` refuse `shots` and `seed` that are not integers (QM-E911) and do not convert them; `decode_dem` checks its arguments; `inject_noise` raises QM-E913 for input it cannot address and no longer returns the text unchanged, and it handles one-line OpenQASM and `measure q -> c`.
- Examples are organised by category instead of by SDK, and every example header names its origin (`Framework:` line). ([#4](https://github.com/ANFAIA/QUMO/pull/4), 2026-11-05)
- The README was shortened from 240 to 65 lines; what it dropped moved to `docs/GETTING_STARTED.md`. ([#5](https://github.com/ANFAIA/QUMO/pull/5), 2026-11-09)
- Migrated to Mojo 1.2 and MAX 26.7; `make setup` updates an existing `.venv`. (commits 6a856cf and 11e15b5, 2026-11-01)

### Removed

- The Python training stack of the AI transpiler (17 files and its `requirements.txt`). It stays recoverable at the git tag `legacy-python-training`. ([#6](https://github.com/ANFAIA/QUMO/pull/6), 2026-11-09)

### Security

- OpenQASM source is capped in what it can expand to (gate calls, unrolled loops, broadcasts, `pow`): a few bytes of source could previously ask for any amount of memory, including through the MCP server. ([#1](https://github.com/ANFAIA/QUMO/pull/1), 2026-11-03)
- MCP server: one message is at most 1 MiB (a longer line is dropped and answered with a parse error), and a call is refused with QM-E913 above `shots` = 1,000,000 or `max_qubits` = 30. The CLI and the Mojo and Python APIs keep the tools' own ranges. (security pass, 2026-11-13; not yet merged, as the next entry)
- Agent layer: the research tools fetch only public `https` addresses on the default port, the API tools only `export.arxiv.org`, `pypi.org` and `api.github.com`, every redirect is checked again and loses the `Authorization` header. `run_check` passes no variable whose name looks like a credential, takes no absolute path, and limits `git diff` / `git status` to `--stat`, `--name-only`, `--name-status`, `--cached`, `--staged`. The `autoresearch` agent no longer has the web tools; it asks `research-watch`. Commands still run without an operating-system sandbox.

## Before pull requests (2026-10-12 to 2026-11-01)

Work committed directly to `main`, listed by commit title:

- 2026-10-12: Initial import of QUMO: hardware-neutral, AI-native quantum stack in Mojo. Apache License 2.0 added.
- 2026-10-13: GPU engine round against CUDA-Q, cloud benchmarks, T4 and M5 results.
- 2026-10-14: Examples from other frameworks, PennyLane-style QML, an AI-driven GPU + QPU example; fp32 became the GPU default. `CheckLog` moved from `qumo.ai` to `qumo.testing`. `make setup-py` added.
- 2026-10-15: AI transpiler (native, hardware-neutral) and the fault-tolerant layer `qumo.ft`.
- 2026-10-19: Four heterogeneous pipelines, GPU adjoint gradients, SQD solver, orchestrator pinning. Fixes to DEM text export and to `qumo_py` import side effects.
- 2026-10-20: Benchmark of the AI transpiler against the AI passes of qiskit-ibm-transpiler.
- 2026-11-01: Fix of the rotation T cost; adaptation to Qiskit, guppylang and Aer GPU updates.

### Fixed

- OpenQASM 2 expressions accept `^` as the power operator (`rz(2^3) q[0];` used to fail with QM-E500).
- Declaring a variable, input, register, gate, gate parameter or loop variable named `pi`, `tau` or `euler` is now QM-E522. It used to be accepted and then ignored: expressions kept the built-in value.
