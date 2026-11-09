# Getting started

Everything runs from the repository root. For the map of all documentation see [README.md](README.md) in this folder.

## Requirements

| | Minimum |
|---|---|
| OS / CPU | macOS 15+ on Apple silicon, or Linux x86-64 / aarch64 |
| Toolchain | Mojo 1.2 + MAX 26.7 Mojo libs/runtime (installed into `.venv` by `make setup`, needs [uv](https://docs.astral.sh/uv/)) |
| Python (optional) | 3.12 for the `qumo_py` extension and the benchmark harnesses (`bench/.venv`) |
| GPU (optional) | NVIDIA (CUDA driver) or AMD (ROCm) on Linux; Apple silicon needs Xcode + Metal Toolchain, see [GPU setup](#gpu-setup) |

## Install, build, test

```bash
git clone https://github.com/ANFAIA/QUMO.git
cd QUMO
make setup        # .venv with Mojo 1.2.0 + MAX 26.7.0 runtime (pinned in the Makefile)
make setup-py     # optional: bench/.venv with Qiskit, Aer, TKET, Cirq, Stim (examples, benchmarks, test-py)
make test         # every Mojo test suite (tests/test_*.mojo); prints PASS/FAIL per suite
make cli          # build/qumo     : CLI + MCP server
make py           # build/qumo_py.so : CPython extension (import qumo_py)
```

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
build/qumo run examples/foundations/bell.qasm --shots 1000 --seed 7
build/qumo validate examples/foundations/teleport.qasm --json
build/qumo draw examples/foundations/teleport.qasm --unicode
build/qumo describe gates
build/qumo tool list                                   # every agent tool
build/qumo tool qumo_transpile '{"qasm":"OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; h q[0]; cx q[0],q[2];","target":"trapped_ion:3"}'
```

### MCP server (Claude Code and other MCP clients)

```bash
claude mcp add qumo -- /abs/path/to/QUMO/build/qumo mcp
```

or `{"mcpServers":{"qumo":{"command":"/abs/path/to/QUMO/build/qumo","args":["mcp"]}}}`. Tools: run, statevector, expectation, validate, describe, circuit info, convert, transpile (any target), resource estimation, VQE, QAOA, QEC memory experiments, detector error models. See [AI_NATIVE.md](AI_NATIVE.md).

### Mojo

```mojo
from qumo.ir.circuit import Circuit
from qumo.compiler.target import parse_target
from qumo.compiler.pass_manager import transpile
from qumo.exec.dispatch import sample_auto

def main() raises:
    var c = Circuit(3)
    c.h(0)
    c.cx(0, 1)
    c.cx(1, 2)
    c.measure_all()
    var r = transpile(c, parse_target("grid:2x2/sc_cz"), 2)   # any modality, none is default
    print(r.summary())
    print(sample_auto(c, 1000, seed=7))   # picks Pauli-frame / tableau / state vector
```

### Python

```python
import sys; sys.path.insert(0, "build")
import qumo_py as q
c = q.load_qasm(open("examples/foundations/bell.qasm").read())
out, initial, final = q.transpile(c, 7, [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6]], ["rz","sx","x","cz"], 2, 1)
print(out.count_ops(), q.sample(c, 1000, 7))
```

### Programs with `while` and classical variables

They run and compile on the CPU; nothing of the kind reaches hardware yet, and `device="gpu"` falls back to the CPU with QM-W020. See [CONTROL_FLOW.md](CONTROL_FLOW.md).

```python
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

AI agents that keep QUMO up to date (research watch, code review, QEC and algorithm research and more) run through `bin/qumo-agents` (Python) or the native Mojo harness. See [AGENTS.md](AGENTS.md).

## Reproduce the benchmarks

```bash
make bench-sv                                         # state vector vs Qiskit Aer
bench/benchpress/install.sh <benchpress-checkout>     # add the QUMO gym to Qiskit Benchpress
bench/run_final.sh                                    # Benchpress (QUMO + Qiskit) + neutral harness
bench/.venv/bin/python bench/neutral/report.py bench/results/neutral.jsonl
```

The benchmark venv needs `qiskit qiskit-aer qiskit-ibm-runtime pytket pytket-qiskit cirq-core ply stim pymatching` and Benchpress' `pytest-benchmark` fork. For Colab and ROCm runs see [BENCHMARKING_ON_CLOUD_GPUS.md](BENCHMARKING_ON_CLOUD_GPUS.md).
