"""QUMO's modules as the agents see them: where the code lives, which docs and tests
cover it, which alternative SDKs to watch, and which literature to search.

Every research and review agent takes a module name from MODULES. Keep this table in
step with the repository: tests/test_agents_modules.py fails when a path is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Module:
    name: str
    summary: str
    paths: tuple[str, ...]
    docs: tuple[str, ...]
    tests: tuple[str, ...]
    # PyPI package names of alternative SDKs that cover the same ground.
    sdks: tuple[str, ...] = ()
    # arXiv categories and search phrases for new papers.
    arxiv_categories: tuple[str, ...] = ("quant-ph",)
    arxiv_queries: tuple[str, ...] = ()
    mcp_tools: tuple[str, ...] = field(default=())


MODULES: dict[str, Module] = {m.name: m for m in [
    Module(
        "ir",
        "Circuit IR: gates, circuits, Pauli sums, Kraus channels, OpenQASM 2/3 parsing and export.",
        ("qumo/ir",), ("docs/ARCHITECTURE.md", "docs/CONTROL_FLOW.md"), ("tests/test_qasm*.mojo", "tests/test_program*.mojo"),
        sdks=("qiskit", "openqasm3", "cirq-core", "pytket", "pyqir"),
        arxiv_queries=("quantum intermediate representation", "OpenQASM", "quantum circuit compiler IR"),
        mcp_tools=("qumo_validate_qasm", "qumo_convert", "qumo_circuit_info", "qumo_draw", "qumo_describe"),
    ),
    Module(
        "sim",
        "CPU simulators: state vector, stabilizer tableau, branching, density matrix, noise and trajectories.",
        ("qumo/sim",), ("docs/SIMULATION.md",), ("tests/test_sim*.mojo", "tests/test_stabilizer*.mojo"),
        sdks=("qiskit-aer", "stim", "cirq-core", "qsim", "pennylane-lightning", "cuda-quantum"),
        arxiv_queries=("state vector simulation", "stabilizer simulation", "quantum circuit simulation algorithm"),
        mcp_tools=("qumo_run_qasm", "qumo_statevector", "qumo_expectation", "qumo_density", "qumo_emulate"),
    ),
    Module(
        "gpu",
        "Vendor-neutral GPU back ends on MAX (NVIDIA, AMD, Apple): state vector and Pauli frames.",
        ("qumo/gpu",), ("docs/GPU.md", "docs/BENCHMARKING_ON_CLOUD_GPUS.md"), ("tests/test_gpu*.mojo",),
        sdks=("qiskit-aer-gpu", "cuda-quantum", "pennylane-lightning-gpu", "pennylane-lightning-kokkos"),
        arxiv_categories=("quant-ph", "cs.DC"),
        arxiv_queries=("GPU quantum circuit simulation", "distributed state vector simulation"),
        mcp_tools=("qumo_gpu_info", "qumo_run_device"),
    ),
    Module(
        "compiler",
        "Transpiler: targets, pass manager, layout, routing, synthesis, exact optimisation passes, AI passes.",
        ("qumo/compiler",), ("docs/COMPILER.md", "docs/AI_TRANSPILER.md", "docs/NEUTRAL_BENCHMARK_REPORT.md"),
        ("tests/test_compiler*.mojo", "tests/test_ai_*.mojo", "tests/test_program_compile*.mojo"),
        sdks=("qiskit", "pytket", "cirq-core", "bqskit", "qiskit-ibm-transpiler"),
        arxiv_categories=("quant-ph", "cs.ET"),
        arxiv_queries=("qubit routing", "quantum circuit optimization", "unitary synthesis", "qubit mapping"),
        mcp_tools=("qumo_transpile", "qumo_estimate_resources", "qumo_to_pbc"),
    ),
    Module(
        "algorithms",
        "Algorithm library: QFT, Grover, QPE, VQE, QAOA, Hamiltonians, QUBO, schedules, QML helpers.",
        ("qumo/algorithms",), ("docs/ALGORITHMS.md", "docs/QML.md"), ("tests/test_algorithm*.mojo",),
        sdks=("qiskit-algorithms", "pennylane", "classiq", "qibo", "qilisdk", "openfermion"),
        arxiv_queries=("quantum algorithm", "variational quantum algorithm", "quantum advantage algorithm"),
        mcp_tools=("qumo_build_circuit", "qumo_vqe", "qumo_qaoa_maxcut", "qumo_expectation"),
    ),
    Module(
        "qec",
        "Error correction toolkit: codes, memory experiments, detector error models, decoders, characterisation.",
        ("qumo/qec",), ("docs/QEC.md",), ("tests/test_qec*.mojo",),
        sdks=("stim", "pymatching", "sinter", "ldpc", "qiskit-qec", "cudaq-qec"),
        arxiv_queries=("quantum error correcting code", "qLDPC code", "surface code decoder", "magic state distillation"),
        mcp_tools=("qumo_qec_memory", "qumo_qec_dem", "qumo_stim_import"),
    ),
    Module(
        "ft",
        "Fault-tolerant programming layer: logical programs on stabilizer codes, gadgets, verification.",
        ("qumo/ft",), ("docs/FT.md",), ("tests/test_ft*.mojo",),
        sdks=("guppylang", "stim", "tket2"),
        arxiv_queries=("fault-tolerant quantum computation", "logical gate", "lattice surgery"),
        mcp_tools=("qumo_ft_codes", "qumo_ft_encode", "qumo_ft_verify"),
    ),
    Module(
        "exec",
        "Execution layer: job planning, batching, devices, runners and hardware plugins.",
        ("qumo/exec", "plugins/hardware"), ("docs/HARDWARE_PLUGINS.md",), ("tests/test_exec*.mojo", "plugins/hardware/tests"),
        sdks=("qiskit-ibm-runtime", "pytket-quantinuum", "amazon-braket-sdk", "qilisdk", "cudaq"),
        arxiv_categories=("quant-ph", "cs.DC"),
        arxiv_queries=("quantum circuit execution", "error mitigation", "quantum cloud scheduling"),
        mcp_tools=("qumo_estimate_resources", "qumo_emulate", "qumo_run_device"),
    ),
    Module(
        "ai",
        "AI-native interface: JSON, MCP server, tool registry, diagnostics and error catalogue.",
        ("qumo/ai", "cli"), ("docs/AI_NATIVE.md", "docs/ERRORS.md", "llms.txt"), ("tests/test_mcp.mojo",),
        sdks=("mcp", "qiskit-mcp-server"),
        arxiv_categories=("quant-ph", "cs.AI", "cs.SE"),
        arxiv_queries=("LLM agents quantum computing", "language model quantum code generation"),
        mcp_tools=("qumo_describe", "qumo_validate_qasm"),
    ),
]}


def get(name: str) -> Module:
    try:
        return MODULES[name]
    except KeyError:
        raise ValueError(f"unknown module {name!r}; choose one of {', '.join(MODULES)}") from None


def describe_all() -> str:
    """One line per module, for system prompts."""
    return "\n".join(f"- {m.name}: {m.summary} Code: {', '.join(m.paths)}." for m in MODULES.values())
