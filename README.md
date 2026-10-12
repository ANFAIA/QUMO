# QUMO — a hardware-neutral, AI-native quantum software stack in Mojo

QUMO rebuilds the quantum software stack in [Mojo](https://mojolang.org) by
taking the best ideas of **Qiskit, TKET, CUDA-Q, PennyLane, Cirq, Q#, Amazon
Braket and Qibo** (see [docs/ANALYSIS.md](docs/ANALYSIS.md)) and putting them
in one language and one binary:

| Layer | What it does | Best-of origin |
|---|---|---|
| `qumo.ir` | Flat POD circuit IR, affine symbolic params, Stim-style measurement records/detectors/noise, functors (`inverse`, `controlled`, `power`), Pauli algebra, OpenQASM 2/3 I/O | Qiskit CircuitData, Stim, Q#, Braket |
| `qumo.compiler` | Vendor-neutral `Target` (any coupling map + basis), text-spec pass pipelines with per-pass reports, SABRE routing, phase folding, commutation, eject-Z, KAK, basis translation to every modality | Qiskit, TKET, Cirq, PennyLane |
| `qumo.sim` | SIMD + multithreaded state vector, stabilizer tableau, bit-packed Pauli-frame sampler, deterministic RNG | qsim/Aer/Lightning, Stim |
| `qumo.exec` | PUB primitives (Sampler/Estimator with broadcasting, exact or shot-based with QWC grouping), HEFT orchestration over CPU/GPU/QPU pools with cost model, resource estimation (logical, NISQ, fault-tolerant) | Qiskit V2 primitives, CUDA-Q mqpu, Braket ProgramSets, Q# QRE |
| `qumo.algorithms` | Kernels (QFT, Grover, QPE, ansätze, Trotter, UCCSD), adjoint / parameter-shift gradients, optimizers, VQE, QAOA | PennyLane, CUDA-Q, Q#, Qualtran |
| `qumo.qec` | Surface/repetition code circuits, detector error models, Union-Find/MWPM decoders, Sinter-like sweeps, calibration-driven noise, RB, readout mitigation | Stim, PyMatching, Sinter, CUDA-Q QEC, Qibocal |
| `qumo.ai` + `cli/` | JSON everywhere, stable `QM-Exxx` error codes with hints, self-description, MCP server, CLI | — (AI-native layer) |

**Hardware neutrality.** No vendor is a default. Targets are data; presets are
named by modality (superconducting heavy-hex / grid / iSWAP, trapped ion,
neutral atom, abstract) and the benchmark harness weighs them equally — see
[docs/BENCHPRESS_BIAS_REPORT.md](docs/BENCHPRESS_BIAS_REPORT.md).

## Results so far (2026-10-12, one Apple Silicon machine)

| Area | Result |
|---|---|
| Qiskit Benchpress (1066 tests, default IBM config) | QUMO **1044 pass / 0 fail** (same as Qiskit 2.5.2); **0.878×** Qiskit's two-qubit gate count (geomean), **2.1× faster**; fewer 2Q gates in 561 tests, more in 124 — [details](docs/BENCHPRESS_RESULTS.md) |
| Hardware-neutral harness (11 topologies, 4 modalities, 4970 verified compilations) | 2Q ratio QUMO/Qiskit **0.928**, QUMO/TKET 0.919, QUMO/Cirq 0.803; QUMO compiles 5–250× faster; highest estimated success probability on every target — [details](docs/NEUTRAL_BENCHMARK_REPORT.md) |
| State-vector simulation (layered circuit, fp64, CPU) | 1.6–2.4× faster than Qiskit Aer 0.17 from 20 to 26 qubits (26 q: 5.6 s vs 9.0 s) |
| QEC (surface-code memory) | Detector error model identical to Stim's; MWPM logical error rates statistically equal to PyMatching; Pauli-frame sampling 3–4× Stim single-threaded — [details](docs/QEC.md) |
| Variational | VQE on H₂ reaches the exact energy (error 1e-14); adjoint gradients 42× faster than parameter shift on 16 qubits / 96 params — [details](docs/ALGORITHMS.md) |

Caveat: QUMO's compiler was tuned during the project on part of these benchmarks and the neutral harness was written by this project; see the disclosure sections of both reports.

## Requirements

| | Minimum |
|---|---|
| OS / CPU | macOS 15+ on Apple silicon, or Linux x86-64 / aarch64 |
| Toolchain | Mojo 1.1 + MAX 26.6 Mojo libs/runtime (installed into `.venv` by `make setup`, needs [uv](https://docs.astral.sh/uv/)) |
| Python (optional) | 3.12 for the `qumo_py` extension and the benchmark harnesses (`bench/.venv`) |
| GPU (optional) | NVIDIA (CUDA driver) or AMD (ROCm) on Linux; Apple silicon needs **Xcode + Metal Toolchain** — see [GPU support](#gpu-support) |

## Install, build, test

```bash
make setup        # .venv with Mojo 1.1 + MAX runtime (uv pip install mojo max-mojo-libs max-core)
make test         # every Mojo test suite (tests/test_*.mojo); prints PASS/FAIL per suite
make cli          # build/qumo     — CLI + MCP server
make py           # build/qumo_py.so — CPython extension (import qumo_py)
```

Run a single suite: `.venv/bin/mojo run -I . tests/test_sim.mojo`. Mojo dialect notes for contributors: [docs/MOJO_GUIDE.md](docs/MOJO_GUIDE.md).

## Usage

**CLI** (human output by default, `--json` for machines; exit codes 0 ok / 1 failed / 2 usage):

```bash
build/qumo run examples/bell.qasm --shots 1000 --seed 7
build/qumo validate examples/teleport.qasm --json
build/qumo describe gates
build/qumo tool list                                   # every agent tool
build/qumo tool qumo_transpile '{"qasm":"OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; h q[0]; cx q[0],q[2];","target":"trapped_ion:3"}'
```

**MCP server** (Claude Code, other MCP clients):

```bash
claude mcp add qumo -- /abs/path/to/QUMO/build/qumo mcp
```

or `{"mcpServers":{"qumo":{"command":"/abs/path/to/QUMO/build/qumo","args":["mcp"]}}}`. Tools: run, statevector, expectation, validate, describe, circuit info, convert, transpile (any target), resource estimation, VQE, QAOA, QEC memory experiments, detector error models — see [docs/AI_NATIVE.md](docs/AI_NATIVE.md).

**Mojo**:

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

**Python**:

```python
import sys; sys.path.insert(0, "build")
import qumo_py as q
c = q.load_qasm(open("examples/bell.qasm").read())
out, initial, final = q.transpile(c, 7, [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6]], ["rz","sx","x","cz"], 2, 1)
print(out.count_ops(), q.sample(c, 1000, 7))
```

## Running on quantum hardware

`plugins/hardware` (`qumo_hardware`) runs QUMO-compiled circuits on **IBM Quantum** (qiskit-ibm-runtime client-side Sampler) and **Quantinuum** (Nexus: H2 and Helios) behind one vendor-neutral provider interface; QUMO does the compilation for every provider from a neutral `TargetSpec`, and results come back in your circuit's bit order. Offline emulators (QUMO simulator on any topology, IBM snapshots with Aer noise, Quantinuum H2-1LE / Selene) need no account. Remote submission only happens with `--yes`:

```bash
export PYTHONPATH=plugins/hardware        # after `make py`
python -m qumo_hardware run examples/ghz.qasm -p ibm -b fake_torino              # local, noisy IBM snapshot
python -m qumo_hardware run examples/ghz.qasm -p quantinuum -b H2-1 --dry-run    # native payload + HQC estimate
python -m qumo_hardware run examples/ghz.qasm -p ibm -b ibm_fez --yes            # real QPU (QISKIT_IBM_TOKEN / saved account)
```

Credentials, costs, safety model, error codes (QM-E950–E959) and how to add Braket/IonQ/IQM: [docs/HARDWARE_PLUGINS.md](docs/HARDWARE_PLUGINS.md); examples in [examples/hardware](examples/hardware).

## GPU support

State-vector kernels are written once in Mojo and compiled by MAX for **NVIDIA, AMD and Apple** GPUs ([docs/GPU.md](docs/GPU.md)). GPU code is compiled only with `-D QUMO_GPU`, so default builds never need a GPU toolchain:

```bash
make gpu-check    # cross-compile every kernel for NVIDIA sm_90, AMD gfx942 and Apple metal:4 (no GPU needed)
make test-gpu     # run GPU tests on the local device
make bench-gpu    # GPU vs CPU throughput
make cli-gpu py-gpu   # CLI / Python extension with GPU dispatch (build/gpu/)
```

Precision: fp32 on Apple GPUs (Metal has no fp64), fp64 or fp32 on NVIDIA/AMD. On macOS, kernels need the Metal compiler:

```bash
xcode-select -s /Applications/Xcode.app/Contents/Developer   # with sudo; Xcode from the App Store
xcodebuild -runFirstLaunch                                    # with sudo
xcodebuild -downloadComponent MetalToolchain
xcrun metal --version                                         # must print a version
```

## Repository layout

```
qumo/
  ir/          circuit IR, gates (opcodes + metadata + matrices), Param, Pauli, OpenQASM 2/3
  compiler/    Target (data), text pass pipelines, routing (VF2 + SABRE), KAK, pauli_simp, phase folding
  sim/         state vector (SIMD, threads, cache blocking), stabilizer tableau, Pauli frames, RNG
  gpu/         MAX GPU kernels, GPU state vector, host emulator for kernel tests (opt-in: -D QUMO_GPU)
  exec/        PUB primitives, HEFT orchestrator, resource estimation, auto dispatch, agent tools
  algorithms/  kernels (QFT, Grover, QPE, ansätze, Trotter, UCCSD), gradients, optimizers, VQE/QAOA
  qec/         codes, detector error models, decoders (UF, MWPM), Sinter-style experiments, characterization
  ai/          JSON, diagnostics, self-description, tool registry, MCP server
  bench/       benchmark circuit generators (QV, EfficientSU2, DTC, ...)
cli/main.mojo  qumo CLI entry point
python/        CPython extension (qumo_py)
tests/         one test_<module>.mojo per module (make test)
examples/      OpenQASM samples and Mojo examples
bench/         simulator / QEC benchmarks, Benchpress gym (bench/benchpress), neutral harness (bench/neutral), results
docs/          analysis, architecture, module docs, benchmark reports, error catalogue
```

## Reproducing the benchmarks

```bash
make bench-sv                                         # state vector vs Qiskit Aer
bench/benchpress/install.sh <benchpress-checkout>     # add the QUMO gym to Qiskit Benchpress
bench/run_final.sh                                    # Benchpress (QUMO + Qiskit) + neutral harness
bench/.venv/bin/python bench/neutral/report.py bench/results/neutral.jsonl
```

The benchmark venv needs `qiskit qiskit-aer qiskit-ibm-runtime pytket pytket-qiskit cirq-core stim pymatching` and Benchpress' `pytest-benchmark` fork.

## Documentation

- [docs/ANALYSIS.md](docs/ANALYSIS.md) — deep analysis of the 8 frameworks and the synthesis
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — architecture and design decisions
- Modules: [COMPILER](docs/COMPILER.md) · [QEC](docs/QEC.md) · [ALGORITHMS](docs/ALGORITHMS.md) · [AI_NATIVE](docs/AI_NATIVE.md) · [GPU](docs/GPU.md) · [ERRORS](docs/ERRORS.md)
- [docs/MOJO_GUIDE.md](docs/MOJO_GUIDE.md) — the exact Mojo 1.1 dialect QUMO uses
- Benchmarking: [bias audit of Benchpress](docs/BENCHPRESS_BIAS_REPORT.md) · [QUMO on Benchpress](docs/BENCHPRESS_RESULTS.md) · [neutral benchmark](docs/NEUTRAL_BENCHMARK_REPORT.md)
- [llms.txt](llms.txt) — compact index for LLM agents

## Status

Research prototype (v0.1.0). Public APIs are small and documented but may still change; Mojo itself is evolving quickly (the dialect is pinned in the contributor guide). Work in progress: GPU execution on real devices, qubit reordering in the simulator, sparse-blossom MWPM decoding, parity with Qiskit on the densest 100+-qubit circuits.

## License

QUMO is licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for attribution; the third-party benchmark circuits in `bench/neutral/circuits` keep their own licenses (`bench/neutral/circuits/LICENSES`).
