# Getting started

Everything runs from the repository root. For the map of all documentation see [README.md](README.md) in this folder. Once QUMO is built, [TUTORIAL.md](TUTORIAL.md) is the guided first hour; if you come from Qiskit, Cirq or PennyLane, read [FROM_QISKIT.md](FROM_QISKIT.md).

## Requirements

| | Minimum |
|---|---|
| OS / CPU | macOS 15+ on Apple silicon, or Linux x86-64 / aarch64 |
| Toolchain | Mojo 1.2 + MAX 26.7 Mojo libs/runtime (installed into `.venv` by `make setup`, needs [uv](https://docs.astral.sh/uv/)) |
| Python (optional) | 3.12 for the `qumo` package (and the `qumo_py` extension under it) and the benchmark harnesses (`bench/.venv`); NumPy is the package's only dependency |
| GPU (optional) | NVIDIA (CUDA driver) or AMD (ROCm) on Linux; Apple silicon needs Xcode + Metal Toolchain, see [GPU setup](#gpu-setup) |
| Disk | about 850 MB for `.venv` (Mojo + MAX), about 11 MB for `build/qumo` and `build/qumo_py.so` together; `make setup-py` adds about 1.5 GB in `bench/.venv` |

### Platforms

The pinned toolchain (`mojo==1.2.0`, `max-core==26.7.0`) is published on PyPI for three platforms only. Checked against the PyPI file lists on 2026-11-10:

| Platform | Status |
|---|---|
| macOS on Apple silicon (arm64) | Supported. The wheels are tagged macOS 13+; QUMO is developed and tested on macOS 15 and later. |
| Linux x86-64 and aarch64, glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+) | Toolchain available. QUMO's GPU benchmarks ran on Linux (Colab, see [BENCHMARKING_ON_CLOUD_GPUS.md](BENCHMARKING_ON_CLOUD_GPUS.md)); the full `make test` suite is not run regularly on Linux. |
| macOS on Intel | Not supported: there is no Mojo wheel for Intel Macs. |
| Windows | Not supported natively: there is no Windows wheel. WSL2 with a recent Ubuntu is a Linux x86-64 system and should work, but nobody has verified QUMO there. |

### How long things take

Measured on an Apple M5 (10 cores), November 2026. Slower machines take longer.

| Step | Time |
|---|---|
| `make cli` | about 50 s |
| `make py` | about 60 s (not re-timed for this page; similar size to the CLI) |
| `mojo run` of a small script | 3 to 5 s, almost all of it compilation; about 18 s the first time a script imports the compiler, 5 s afterwards (Mojo caches modules) |
| the same script built once with `mojo build` | 2.5 s to build, 0.16 s to run |
| `build/qumo run examples/foundations/bell.qasm` | 2 ms of simulation; 13 ms for the whole process |

## Install, build, test

```bash
git clone https://github.com/ANFAIA/QUMO.git
cd QUMO
make setup        # .venv with Mojo 1.2.0 + MAX 26.7.0 runtime (pinned in the Makefile)
make setup-py     # optional: bench/.venv with NumPy, pytest, Qiskit, Aer, TKET, Cirq, Stim (examples, benchmarks, test-py)
make test         # every Mojo test suite (tests/test_*.mojo); prints PASS/FAIL per suite
make cli          # build/qumo     : CLI + MCP server
make py           # build/qumo_py.so : the compiled extension behind the Python package
pip install -e .  # optional: the Python package `qumo` (or: uv pip install -e . ; make install-py)
```

`make help` lists every target and marks the ones a new user needs. `make test-python` runs the tests of the Python package and `make test-py` every Python test (it needs `make setup-py`).

Run one suite: `.venv/bin/mojo run -I . tests/test_sim.mojo`. Dialect notes for contributors: [MOJO_GUIDE.md](MOJO_GUIDE.md).

### Updating an existing clone

QUMO needs Mojo 1.2.0 + MAX 26.7.0. A `.venv` created before that keeps Mojo 1.1, which cannot build this tree; the build stops with `module 'file_descriptor' does not contain 'RawFileDescriptor'`. After every pull that changes the pins in the Makefile:

```bash
git pull
make setup        # updates the existing .venv to the pinned versions
make setup-py     # optional: updates bench/.venv
make cli py       # rebuild: binaries in build/ still come from the old toolchain
```

`examples/colab/run_on_any_gpu.sh` and the notebooks skip the install when `.venv/bin/mojo` exists, so run `make setup` first on a machine that already has a `.venv`.

## Use QUMO

### CLI

Human output by default, `--json` for machines. Exit codes: 0 ok, 1 failed, 2 usage.

```bash
build/qumo --help                                                       # the commands, with three first steps
build/qumo run examples/foundations/bell.qasm --shots 1000 --seed 7     # sample counts
build/qumo statevector examples/foundations/bell.qasm                   # exact amplitudes
build/qumo expect examples/foundations/bell.qasm "Z0 Z1"                # exact expectation value
build/qumo draw examples/foundations/teleport.qasm --unicode
build/qumo validate examples/foundations/teleport.qasm --json           # parse + lint; lists every error it finds
build/qumo transpile examples/foundations/bell.qasm --target grid:2x2/sc_cz              # compile for a device
build/qumo emulate examples/foundations/bell.qasm --target superconducting_grid:2x2      # noisy run on a device model
build/qumo vqe "Z0 Z1 + 0.5*X0 + 0.5*X1"                                # ground-state energy of a Pauli sum
build/qumo describe gates
```

For example, `transpile` prints what the compiler did:

```
examples/foundations/bell.qasm -> grid_2x2 (4 qubits, coupling_map): level 2, seed 7, 0.089 ms
  pipeline: remove_identities,elide_swaps,pauli_simp,fixpoint(cancel,commute,fuse1q,max=4),route(sabre,layout=auto,trials=20,iterations=2,swap_trials=20),kak,basis,cancel,commute,fuse1q,check
                         before    after
  gates                       2        9
  two-qubit gates             1        1
  depth                       2        6
  swaps inserted                       0
  estimated fidelity            1.000000
  layout: initial [0, 1, 2, 3], final [0, 1, 2, 3]
  (-o out.qasm saves the compiled circuit, --qasm prints it, --json gives the per-pass report)
```

Things to know:

- **Help.** `build/qumo <command> --help` lists a command's options, defaults and examples.
- **Options are strict.** An option the command does not declare is a usage error (QM-E921, exit code 2) with a suggestion; nothing is ignored silently. `build/qumo run bell.qasm --shot 10` answers `unknown option '--shot' for 'qumo run'` and `did you mean '--shots'?`. Options are written `--name value` or `--name=value`.
- **Seeds.** Every random process takes a seed, and the same seed gives the same result. The defaults are the same in the CLI, the MCP tools, Mojo and Python: **1234** for sampling (`run`, `emulate`, `sample`, `sample_auto`), **7** for the compiler and VQE / QAOA (`transpile`, `vqe`), **1** for QEC experiments (`memory_experiment`). Pass `--seed` (or `seed=`) to choose.
- **Everything else is a tool.** The commands above cover the common tasks. Every other operation (QAOA, QEC experiments, resource estimation, density matrices, building algorithm circuits, ...) is a tool that takes one JSON object and prints JSON:

```bash
build/qumo tool list                                   # every tool, one line each
build/qumo tool qumo_qec_memory --help                 # a tool's arguments, defaults and an example
build/qumo tool qec_memory '{"code": "surface_z", "distance": 3, "p": 0.001, "shots": 10000}'
build/qumo tool draw '{"file": "examples/foundations/bell.qasm"}'
```

The `qumo_` prefix of a tool name is optional on the command line. A tool that takes OpenQASM text in `"qasm"` also takes `"file": "path"` there; the CLI reads the file (the MCP server does not read files).

### MCP server (Claude Code and other MCP clients)

```bash
claude mcp add qumo -- /abs/path/to/QUMO/build/qumo mcp
```

or `{"mcpServers":{"qumo":{"command":"/abs/path/to/QUMO/build/qumo","args":["mcp"]}}}`. Tools: run, statevector, expectation, validate, describe, circuit info, convert, transpile (any target), resource estimation, VQE, QAOA, QEC memory experiments, detector error models. See [AI_NATIVE.md](AI_NATIVE.md).

### Mojo

`qumo.prelude` re-exports the names a first program needs (`Circuit`, `parse_qasm`, `to_qasm`, `sample_auto`, `sample`, `statevector`, `expectation`, `parse_pauli_sum`, `Target`, `parse_target`, `transpile`), so one import is enough:

```mojo
from qumo.prelude import *


def main() raises:
    var c = Circuit(3)
    c.h(0)
    c.cx(0, 1)
    c.cx(1, 2)
    c.measure_all()
    print(c.draw())
    print(sample_auto(c, 1000, seed=7))                       # picks Pauli-frame / tableau / state vector
    print(statevector(c))                                     # exact amplitudes
    print(expectation(c, parse_pauli_sum("Z0 Z1")))
    var r = transpile(c, parse_target("grid:2x2/sc_cz"), 2)   # any modality, none is default
    print(r.summary())
```

Save it as `my.mojo` and run it from the repository root:

```bash
.venv/bin/mojo run -I . my.mojo
```

```
q0: --H--@--M>c0--------
         |
q1: -----+---@----M>c1--
             |
q2: ---------+----M>c2--

{"000": 484, "111": 516}
StateVector(3 qubits)
  |000>: 0.7071067811865476 + 0.0j
  |111>: 0.7071067811865476 + 0.0j
1.0000000000000002
compiled 'circuit' for grid_2x2 [rz,sx,x,cz]: gates 3 -> 15, 2q 2 -> 2, depth 3 -> 9, T 0 -> 0, 11 passes in 0.123 ms
```

The prelude is a list of re-exports and costs nothing at run time. Everything else (variational algorithms, QEC, noise models, GPU, hardware clients) stays one explicit import away, for example `from qumo.algorithms.variational import vqe`; the tutorial and the module references use those deep imports. Result types print readably and have `to_json()`: `SampleResult`, `VQEResult`, `CompileResult`.

`-I .` is how Mojo finds the `qumo` package, so run from the root or pass the root's path. `mojo run` compiles before it runs (see the timings above); `.venv/bin/mojo build -I . -o my my.mojo` builds a binary you can run instantly. The first output line, `Failed to initialize Crashpad. Crash reporting will not be available. ...`, is harmless: the crash reporter is not installed. The Mojo dialect QUMO uses is described in [MOJO_GUIDE.md](MOJO_GUIDE.md).

### Python

The `qumo` package is the Python way in: a circuit builder with gate methods, keyword arguments, NumPy arrays, errors with codes and hints. The reference is [PYTHON.md](PYTHON.md).

```bash
make py               # build/qumo_py.so
pip install -e .      # or: uv pip install -e . ; `make install-py` installs into bench/.venv
```

```python
import qumo

bell = qumo.Circuit(2).h(0).cx(0, 1).measure_all()
print(qumo.sample(bell, shots=1000))
```

```
1000 shots, 2 outcome(s), statevector:static
  11       504  0.5040  ####################
  00       496  0.4960  ####################
```

Loading a file, compiling for a device and reading exact values:

```python
import qumo

c = qumo.load("examples/foundations/bell.qasm")          # or build it: qumo.Circuit(2).h(0).cx(0, 1).measure_all()
r = qumo.transpile(c, "line:7/sc_cz", level=2)           # the same target specs as the CLI and Mojo
print(r.circuit.count_ops(), r.initial_layout)
print(dict(qumo.sample(c, shots=1000, seed=7)))
print(qumo.expectation(c, "Z0 Z1"), qumo.statevector(qumo.Circuit(2).h(0).cx(0, 1)))
```

```
{'sx': 3, 'rz': 5, 'cz': 1, 'measure': 2} [0, 1, 2, 3, 4, 5, 6]
{'11': 516, '00': 484}
1.0000000000000002 [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
```

The install is editable because the package loads `build/qumo_py.so` from the checkout; rebuild with `make py` and the next Python process picks it up. At the repository root `import qumo` also works without installing (after `make py`). `build/qumo_py.so` was built and tested with Python 3.12.

**The low-level layer.** `qumo_py` is the compiled extension under the package. Its functions take positional arguments and its circuits come from OpenQASM text. It is available as `qumo.native`, or directly with `import sys; sys.path.insert(0, "build"); import qumo_py`. Use it for what the package does not wrap: programs with run-time control flow (below), the hardware clients and the benchmark helpers. `help(qumo.native.transpile)` shows a function's signature.

**Jupyter.** There is no Mojo kernel in this repository; notebooks use the Python package. Install it into the kernel's environment with that environment's `pip install -e .` (for example `make install-py` for `bench/.venv`, which has `ipykernel`; install `jupyterlab` or `notebook` into it yourself, `make setup-py` does not). A circuit as the last expression of a cell shows its drawing, and counts show a histogram. Restart the kernel after `make py`. To run Mojo code from a notebook, call the CLI (`!build/qumo run ...`) or `!.venv/bin/mojo run -I . my.mojo`.

### Programs with `while` and classical variables

They run and compile on the CPU; nothing of the kind reaches hardware yet, and `device="gpu"` falls back to the CPU with QM-W020. See [CONTROL_FLOW.md](CONTROL_FLOW.md).

```python
import qumo

q = qumo.native                       # the low-level extension, qumo_py
p = q.load_program("OPENQASM 3.0; qubit[1] q; bit[1] c; output int n = 0; while (!c[0]) { h q[0]; c[0] = measure q[0]; n += 1; }")
print(q.sample_program(p, 1000, 7)["outputs"]["n"][:5])
```

## GPU setup

Kernels are written once in Mojo and compiled by MAX for NVIDIA, AMD and Apple GPUs ([GPU.md](GPU.md)). GPU code is compiled only with `-D QUMO_GPU`, so default builds never need a GPU toolchain.

```bash
make gpu-check    # cross-compile every kernel for NVIDIA sm_75/sm_80/sm_90, AMD gfx942 and Apple metal:4 (no GPU needed)
make test-gpu     # run GPU tests on the local device
make bench-gpu    # GPU vs CPU throughput
make cli-gpu py-gpu   # CLI / Python extension with GPU dispatch (build/gpu/)
```

Precision is fp32 by default on every GPU. fp64 (NVIDIA/AMD; Metal has no fp64) is selectable at build time (`-D QUMO_GPU_FP64`), per process (`QUMO_GPU_PRECISION=fp64`) or per call (`precision="fp64"`).

On macOS the kernels need the Metal compiler:

```bash
xcode-select -s /Applications/Xcode.app/Contents/Developer   # with sudo; Xcode from the App Store
xcodebuild -runFirstLaunch                                    # with sudo
xcodebuild -downloadComponent MetalToolchain
xcrun metal --version                                         # must print a version
```

## Quantum hardware

`plugins/hardware` (`qumo_hardware`) runs QUMO-compiled circuits on IBM Quantum and Quantinuum (Nexus) behind one provider interface. Offline emulators need no account; remote submission only happens with `--yes`.

```bash
export PYTHONPATH=plugins/hardware        # after `make py`
python -m qumo_hardware run examples/foundations/ghz.qasm -p ibm -b fake_torino              # local, noisy IBM snapshot
python -m qumo_hardware run examples/foundations/ghz.qasm -p quantinuum -b H2-1 --dry-run    # native payload + estimate
python -m qumo_hardware run examples/foundations/ghz.qasm -p ibm -b ibm_fez --yes            # real QPU (QISKIT_IBM_TOKEN / saved account)
```

Credentials, costs, safety model, error codes and how to add providers: [HARDWARE_PLUGINS.md](HARDWARE_PLUGINS.md).

## Agents

AI agents that keep QUMO up to date (research watch, code review, QEC and algorithm research and more) run through `bin/qumo-agents`, a Python package built on deepagents (`make setup-agents`). A native Mojo harness for the same agents is proposed in pull request #7 and is not merged. See [AGENTS.md](AGENTS.md).

## Reproduce the benchmarks

```bash
make bench-sv                                         # state vector vs Qiskit Aer
bench/benchpress/install.sh <benchpress-checkout>     # add the QUMO gym to Qiskit Benchpress
bench/run_final.sh                                    # Benchpress (QUMO + Qiskit) + neutral harness
bench/.venv/bin/python bench/neutral/report.py bench/results/neutral.jsonl
```

The benchmark venv needs `qiskit qiskit-aer qiskit-ibm-runtime pytket pytket-qiskit cirq-core ply stim pymatching` and Benchpress' `pytest-benchmark` fork. For Colab and ROCm runs see [BENCHMARKING_ON_CLOUD_GPUS.md](BENCHMARKING_ON_CLOUD_GPUS.md).
