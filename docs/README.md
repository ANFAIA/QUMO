# QUMO documentation

Start with [Getting started](GETTING_STARTED.md). Then pick by what you want to do.

## I want to use QUMO

| Goal | Read |
|---|---|
| Install, build, run the CLI / MCP server / Python | [GETTING_STARTED](GETTING_STARTED.md) |
| Browse runnable examples by topic | [examples/CATALOG.md](../examples/CATALOG.md), [examples/README.md](../examples/README.md) |
| Drive QUMO from an LLM agent (MCP, JSON, error codes) | [AI_NATIVE](AI_NATIVE.md), [ERRORS](ERRORS.md), [llms.txt](../llms.txt) |
| Run on IBM Quantum or Quantinuum | [HARDWARE_PLUGINS](HARDWARE_PLUGINS.md) |
| Run on NVIDIA, AMD or Apple GPUs | [GPU](GPU.md) |
| Use the AI agents that keep QUMO current | [AGENTS](AGENTS.md) |

## Reference, by module

| Module | Doc |
|---|---|
| `qumo.ir`, classical variables and `while` | [CONTROL_FLOW](CONTROL_FLOW.md) |
| `qumo.compiler` | [COMPILER](COMPILER.md), [AI_TRANSPILER](AI_TRANSPILER.md) |
| `qumo.sim` | [SIMULATION](SIMULATION.md) |
| `qumo.gpu` | [GPU](GPU.md) |
| `qumo.algorithms` | [ALGORITHMS](ALGORITHMS.md), [QML](QML.md) |
| `qumo.qec` | [QEC](QEC.md) |
| `qumo.ft` | [FT](FT.md) |
| `qumo.ai`, CLI, MCP | [AI_NATIVE](AI_NATIVE.md), [ERRORS](ERRORS.md) |

## Design and background

- [ARCHITECTURE](ARCHITECTURE.md): layers and design decisions.
- [ANALYSIS](ANALYSIS.md): analysis of the 8 reference frameworks and the synthesis.
- [FRAMEWORK_UPDATES](FRAMEWORK_UPDATES.md): features tracked across frameworks (done, deferred, skipped).
- [QILISDK_MAPPING](QILISDK_MAPPING.md): qilisdk to QUMO mapping.

## Benchmarks

- [BENCHPRESS_RESULTS](BENCHPRESS_RESULTS.md): QUMO on Qiskit Benchpress.
- [BENCHPRESS_BIAS_REPORT](BENCHPRESS_BIAS_REPORT.md): audit of Benchpress for bias.
- [NEUTRAL_BENCHMARK_REPORT](NEUTRAL_BENCHMARK_REPORT.md): hardware-neutral compiler comparison.
- [SIMULATION](SIMULATION.md), [GPU](GPU.md): simulator and GPU numbers.
- [BENCHMARKING_ON_CLOUD_GPUS](BENCHMARKING_ON_CLOUD_GPUS.md): Colab, ROCm and any GPU host.

## Contributing

- [MOJO_GUIDE](MOJO_GUIDE.md): the Mojo 1.2 dialect QUMO uses.
- Tests: `make test`; examples: `make examples`.

## Repository layout

```
qumo/
  ir/          circuit IR, Program (control flow, classical variables), gates, Param, Pauli, OpenQASM 2/3
  compiler/    Target (data), text pass pipelines, routing (VF2 + SABRE), KAK, phase folding
  sim/         state vector, stabilizer tableau + branching, Pauli frames, density matrix, RNG
  gpu/         MAX GPU kernels and state vector (opt-in: -D QUMO_GPU)
  exec/        PUB primitives, HEFT orchestrator, resource estimation, auto dispatch, agent tools
  algorithms/  QFT, Grover, QPE, ansatze, Trotter, UCCSD, gradients, optimizers, VQE/QAOA
  ft/          stabilizer codes, logical programs, encoding passes, gadget verification
  qec/         codes, detector error models, decoders, Sinter-style experiments
  ai/          JSON, diagnostics, self-description, tool registry, MCP server
cli/           qumo CLI entry point
python/        CPython extension (qumo_py)
plugins/       hardware providers (qumo_hardware)
agents/        deepagents-based agents and CLI
examples/      runnable examples by category
bench/         benchmarks and harnesses
training/      AI transpiler training
tests/         one test_<module>.mojo per module
docs/          this documentation
```
