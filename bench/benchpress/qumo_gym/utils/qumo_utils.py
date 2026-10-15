"""IO, target extraction and validation helpers for the QUMO gym."""
import os
from time import perf_counter

import qumo_py

from benchpress.config import Configuration, POSSIBLE_2Q_GATES


# repository models (the gym's _build symlink points at <repo>/build)
_MODEL_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "_build", "..", "models", "ai_transpiler"))


class QumoTarget:
    """Plain-data view of a Benchpress backend: qubits, edges, basis."""

    def __init__(self, num_qubits, edges, basis, name):
        self.num_qubits = num_qubits
        self.edges = [tuple(e) for e in edges]
        self.edge_set = set(self.edges) | {(b, a) for a, b in self.edges}
        self.basis = list(basis)
        self.name = name
        full = num_qubits * (num_qubits - 1)
        self.all_to_all = len(self.edge_set) >= full
        two = [g for g in self.basis if g in POSSIBLE_2Q_GATES]
        self.two_q_gate_type = two[0] if two else None

    @staticmethod
    def from_qiskit_backend(backend, name=None):
        tgt = backend.target
        edges = list(tgt.build_coupling_map().get_edges()) if tgt.build_coupling_map() is not None else []
        basis = sorted(n for n in tgt.operation_names if n not in ("measure", "delay", "reset", "barrier")
                       and not n.startswith(("if_else", "while", "for", "switch", "box")))
        return QumoTarget(tgt.num_qubits, edges, basis, name or getattr(backend, "name", "backend"))


def device_target():
    """Same device every Benchpress gym compiles to (default.conf backend_name)."""
    from benchpress.qiskit_gym.utils.qiskit_backend_utils import get_qiskit_bench_backend
    name = Configuration.options["general"]["backend_name"]
    return QumoTarget.from_qiskit_backend(get_qiskit_bench_backend(name), name)


def flexible_target(num_qubits, topology):
    """Identical topology/basis to Benchpress' FlexibleBackend."""
    from benchpress.utilities.backends import FlexibleBackend
    return QumoTarget.from_qiskit_backend(FlexibleBackend(num_qubits, topology), "flexible-" + topology)


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def ai_settings():
    """`[qumo] ai = true` (or env QUMO_BENCH_AI=1) switches every gym test to
    the hybrid heuristic + AI preset `transpile_ai`; `[qumo] ai_level` (1..3,
    default: the optimization level clamped to 1..3) picks its level."""
    opts = Configuration.options.get("qumo", {})
    ai = _truthy(os.environ.get("QUMO_BENCH_AI", opts.get("ai", "false")))
    lvl = opts.get("ai_level", os.environ.get("QUMO_BENCH_AI_LEVEL", ""))
    return ai, (int(lvl) if str(lvl).strip() else None)


def transpile(circuit, target, level, seed=12345):
    edges = [] if target.all_to_all else [list(e) for e in target.edges]
    ai, ai_level = ai_settings()
    if ai:
        os.environ.setdefault("QUMO_MODEL_DIR", _MODEL_DIR)
        lvl = ai_level if ai_level is not None else min(max(int(level), 1), 3)
        out, _init, _final = qumo_py.transpile_ai(circuit, target.num_qubits, edges, target.basis, lvl, seed)
        return out
    out, _init, _final = qumo_py.transpile(circuit, target.num_qubits, edges, target.basis, level, seed)
    return out


def qasm_loader(path, benchmark):
    start = perf_counter()
    circuit = qumo_py.load_qasm_file(path)
    benchmark.extra_info["qasm_load_time"] = perf_counter() - start
    benchmark.extra_info["input_num_qubits"] = circuit.num_qubits()
    return circuit


def input_properties(circuit, benchmark):
    benchmark.extra_info["input_num_qubits"] = circuit.num_qubits()


def output_properties(circuit, two_qubit_gate, benchmark):
    benchmark.extra_info["output_num_qubits"] = circuit.num_qubits()
    ops = circuit.count_ops()
    benchmark.extra_info["output_circuit_operations"] = ops
    benchmark.extra_info["output_gate_count_2q"] = ops.get(two_qubit_gate, 0)
    benchmark.extra_info["output_depth_2q"] = circuit.depth_of(two_qubit_gate)


def validate(circuit, target):
    """Benchpress rule: only basis gates, 2q gates only on coupling edges."""
    allowed = set(target.basis) | {"barrier", "measure", "reset", "delay", "id"}  # same as backend.operation_names
    extra = set(circuit.count_ops()) - allowed
    if extra:
        raise Exception(f"Circuit has gates outside backend basis set {extra}")
    if not target.all_to_all:
        for a, b in circuit.two_qubit_edges():
            if (a, b) not in target.edge_set:
                raise Exception(f"2Q gate edge {(a, b)} not in backend topology")
    return True


def hamiltonian_circuit(sparse_op, evo_time=1.0):
    labels = sparse_op.paulis.to_labels()
    coeffs = [float(c.real) for c in sparse_op.coeffs]
    return qumo_py.hamiltonian_circuit(labels, coeffs, evo_time)
