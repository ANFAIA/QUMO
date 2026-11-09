# QUMO

A hardware-neutral, AI-native quantum software stack written in [Mojo](https://mojolang.org). One language, one binary: circuits, compiler, simulators, GPU kernels, error correction, fault tolerance and an MCP server for LLM agents. It takes the best ideas of Qiskit, TKET, CUDA-Q, PennyLane, Cirq, Q#, Amazon Braket and Qibo ([analysis](docs/ANALYSIS.md)).

No vendor is a default: targets are data, and presets are named by modality (superconducting, trapped ion, neutral atom, abstract).

## Quick start

Needs macOS 15+ (Apple silicon) or Linux, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ANFAIA/QUMO.git && cd QUMO
make setup        # Mojo 1.2 toolchain in .venv
make cli          # build/qumo : CLI + MCP server
build/qumo run examples/foundations/bell.qasm --shots 1000 --seed 7
```

Use it from an LLM client: `claude mcp add qumo -- /abs/path/to/QUMO/build/qumo mcp`. From Python: `make py`, then `import qumo_py`. Full install, update, GPU and hardware steps: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## What is inside

| Module | What it does |
|---|---|
| `qumo.ir` | Circuit IR, `Program` (`while`, classical variables), symbolic parameters, OpenQASM 2/3 |
| `qumo.compiler` | Any coupling map + basis, text-spec pass pipelines, SABRE routing, KAK, AI transpiler |
| `qumo.sim` | State vector (SIMD, threads), stabilizer tableau, Pauli-frame sampler, density matrix |
| `qumo.gpu` | One Mojo source for NVIDIA, AMD and Apple GPUs |
| `qumo.exec` | Sampler/Estimator primitives, HEFT orchestration over CPU/GPU/QPU, resource estimation |
| `qumo.algorithms` | QFT, Grover, QPE, Trotter, UCCSD, gradients, VQE, QAOA, QML |
| `qumo.qec`, `qumo.ft` | Surface codes, decoders, detector error models; fault-tolerant programming |
| `qumo.ai`, `cli/` | JSON everywhere, stable `QM-Exxx` error codes with hints, MCP server, CLI |
| `plugins/hardware` | Run compiled circuits on IBM Quantum and Quantinuum (dry-run by default) |
| `agents/` | AI agents that keep QUMO current, in Python and as a native Mojo harness |

## Results (Apple M5 unless noted)

| Area | Result |
|---|---|
| Qiskit Benchpress (1066 tests) | 1044 pass / 0 fail; 0.878x Qiskit's two-qubit gate count, 2.1x faster. [Details](docs/BENCHPRESS_RESULTS.md) |
| Neutral harness (11 topologies, 4 modalities) | 2Q ratio 0.928 vs Qiskit, 0.919 vs TKET, 0.803 vs Cirq; compiles 5-250x faster. [Details](docs/NEUTRAL_BENCHMARK_REPORT.md) |
| State vector, CPU (fp64) | 2.5-4.7x faster than Qiskit Aer from 16 to 26 qubits. [Details](docs/SIMULATION.md) |
| State vector, GPU (fp32) | Apple M5: 26-qubit layered circuit in 0.42 s. NVIDIA/AMD not yet measured with the current engine. [Details](docs/GPU.md) |
| QEC | Detector error model identical to Stim's; MWPM equal to PyMatching; frame sampling 3-4x Stim. [Details](docs/QEC.md) |
| VQE | H2 reaches the exact energy (error 1e-14); adjoint gradients 42x faster than parameter shift. [Details](docs/ALGORITHMS.md) |

The compiler was tuned on part of these benchmarks and the neutral harness was written by this project; read the disclosure sections of both reports. Most rows were measured around 2026-10-13 with Mojo 1.1 and Qiskit 2.5.2.

## Documentation

The full map is [docs/README.md](docs/README.md). Common entry points:

- [Getting started](docs/GETTING_STARTED.md): install, CLI, MCP, Python, GPU, hardware.
- [Examples](examples/CATALOG.md): every example by category, with its original source.
- [Architecture](docs/ARCHITECTURE.md) and [framework analysis](docs/ANALYSIS.md).
- [AI-native interface](docs/AI_NATIVE.md), [error codes](docs/ERRORS.md), [llms.txt](llms.txt).
- [Agents](docs/AGENTS.md).
- [Contributing: Mojo dialect guide](docs/MOJO_GUIDE.md).

## Status

Research prototype (v0.1.0). APIs are small and documented but may change, and Mojo itself moves quickly. In progress: measuring the GPU engine on NVIDIA/AMD, parity with Qiskit on the densest 100+ qubit circuits, and real-hardware submission (the plugins are tested offline only).

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution; third-party benchmark circuits in `bench/neutral/circuits` keep their own licenses.
