"""Compiler adapters with identical contracts for every SDK.

compile(qasm_text, target, level, seed) -> CompileOutput
  * input: the SAME normalised OpenQASM 2 text for every SDK (parsed by the
    SDK's own parser, parse time measured separately);
  * output: the compiled circuit converted to a Qiskit QuantumCircuit (only as
    a neutral container for verification/metrics), the initial and final
    logical->physical layouts, and wall/CPU compile time.

`level` is "default" (each SDK's documented default effort) or "max" (each
SDK's highest generally-applicable effort). Unsupported combinations raise
`Unsupported` with the reason; they are reported as such, never as failures.
"""
import time
from dataclasses import dataclass

import qiskit
from qiskit import QuantumCircuit, qasm2


class Unsupported(Exception):
    pass


@dataclass
class CompileOutput:
    circuit: QuantumCircuit
    initial: list  # logical i -> physical index at start
    final: list  # logical i -> physical index at end
    wall_s: float
    cpu_s: float
    parse_s: float
    sdk_version: str


def _timed(fn):
    w0, c0 = time.perf_counter(), time.process_time()
    out = fn()
    return out, time.perf_counter() - w0, time.process_time() - c0


# --------------------------------------------------------------------------- Qiskit
def compile_qiskit(qasm_text, target, level, seed):
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    (qc, parse_s, _) = _timed(lambda: qasm2.loads(qasm_text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))
    n_log = qc.num_qubits
    cmap = None
    if not target.all_to_all:
        cmap = CouplingMap(list(target.edge_set))
    opt = 2 if level == "default" else 3
    basis = target.basis
    if cmap is None:
        # all-to-all: give the full width so ancilla handling matches other SDKs
        pass
    tqc, wall, cpu = _timed(lambda: transpile(qc, basis_gates=basis, coupling_map=cmap,
                                              optimization_level=opt, seed_transpiler=seed))
    if tqc.layout is not None:
        init = tqc.layout.initial_index_layout(filter_ancillas=True)[:n_log]
        final = tqc.layout.final_index_layout(filter_ancillas=True)[:n_log]
    else:
        init = list(range(n_log))
        final = list(range(n_log))
    return CompileOutput(tqc, list(init), list(final), wall, cpu, parse_s, qiskit.__version__)


# --------------------------------------------------------------------------- TKET
_TK_NAMES = None


def _tk_optypes():
    from pytket import OpType
    return {"rz": OpType.Rz, "rx": OpType.Rx, "ry": OpType.Ry, "sx": OpType.SX, "x": OpType.X,
            "cz": OpType.CZ, "cx": OpType.CX, "rxx": OpType.XXPhase, "rzz": OpType.ZZPhase}


def compile_tket(qasm_text, target, level, seed):
    import pytket
    from pytket.qasm import circuit_from_qasm_str
    from pytket.architecture import Architecture
    from pytket.passes import (SequencePass, DecomposeBoxes, FullPeepholeOptimise, DefaultMappingPass,
                               AutoRebase, AutoSquash, RemoveRedundancies, SynthesiseTket, CliffordSimp,
                               KAKDecomposition)
    from pytket.predicates import CompilationUnit
    from pytket.extensions.qiskit import tk_to_qiskit

    ops = _tk_optypes()
    (circ, parse_s, _) = _timed(lambda: circuit_from_qasm_str(qasm_text))
    gateset = {ops[g] for g in target.basis}
    one_q = {ops[g] for g in target.basis_1q}
    from pytket import OpType
    # Best configuration per 2q family: CX-based peephole for CX/CZ targets,
    # TK2 (canonical 2q interaction) for native Ising-type gates (RXX/RZZ),
    # which AutoRebase then decomposes optimally.
    tk2 = target.gate_2q in ("rxx", "rzz")
    peep = FullPeepholeOptimise(target_2qb_gate=OpType.TK2) if tk2 else FullPeepholeOptimise()
    seq = [DecomposeBoxes(), peep]
    if level == "max":
        seq += [CliffordSimp(), KAKDecomposition(target_2qb_gate=OpType.TK2) if tk2 else KAKDecomposition(), peep]
    if not target.all_to_all:
        arch = Architecture([list(e) for e in target.edges])
        seq += [DefaultMappingPass(arch), SynthesiseTket()]
        if level == "max":
            # re-optimise the routed circuit locally (KAK on existing pairs only;
            # FullPeepholeOptimise's Clifford rewrites could create uncoupled pairs)
            seq += [KAKDecomposition(allow_swaps=False, target_2qb_gate=OpType.TK2) if tk2
                    else KAKDecomposition(allow_swaps=False), SynthesiseTket()]
    if tk2:
        # TK2 -> ZZPhase is 1:1 in TKET; AutoRebase to XXPhase is not, so for
        # RXX targets we go through ZZPhase and swap the local basis below.
        from pytket.passes import DecomposeTK2, NormaliseTK2
        seq += [NormaliseTK2(), DecomposeTK2(ZZPhase_fidelity=lambda a: 1.0),
                AutoRebase(one_q | {OpType.ZZPhase})]
    else:
        seq += [AutoRebase(gateset)]
    seq += [AutoSquash(one_q), RemoveRedundancies()]
    pm = SequencePass(seq)
    cu = CompilationUnit(circ)

    def run():
        pm.apply(cu)
        compiled = cu.circuit
        # TKET may leave *implicit* wire permutations (free relabelling from
        # allow_swaps); they are part of its answer, so fold them into the
        # final layout instead of materialising SWAP gates.
        perm = compiled.implicit_qubit_permutation()
        if target.gate_2q == "rxx":
            compiled = _zz_to_xx(compiled, one_q)
        return compiled, perm

    (out, perm), wall, cpu = _timed(run)
    logical = sorted(circ.qubits)
    # tk_to_qiskit maps node[k] (or q[k]) to Qiskit qubit k of one flat register
    init = [cu.initial_map[q].index[0] for q in logical]
    # implicit_qubit_permutation() maps output wire -> input wire, so the
    # content of wire w ends on inv[w]
    inv = {v: k for k, v in perm.items()}
    final = [inv.get(cu.final_map[q], cu.final_map[q]).index[0] for q in logical]
    qc = tk_to_qiskit(out.copy()) if not out.implicit_qubit_permutation() else _tk_to_qiskit_no_perm(out)
    if qc.num_qubits < max(max(init), max(final)) + 1:
        raise RuntimeError("unexpected qubit indexing after tk_to_qiskit")
    return CompileOutput(qc, init, final, wall, cpu, parse_s, pytket.__version__)


def _tk_to_qiskit_no_perm(circ):
    """Convert gate-by-gate ignoring the implicit permutation (handled via layouts)."""
    from pytket import Circuit
    from pytket.extensions.qiskit import tk_to_qiskit
    plain = Circuit()
    for q in circ.qubits:
        plain.add_qubit(q)
    for cmd in circ.get_commands():
        plain.add_gate(cmd.op, cmd.args)
    plain.add_phase(circ.phase)
    return tk_to_qiskit(plain)


def _zz_to_xx(circ, one_q):
    """Exact local rewrite ZZPhase(a) = (H x H) XXPhase(a) (H x H), then resquash."""
    from pytket import Circuit, OpType
    from pytket.passes import AutoSquash, AutoRebase, RemoveRedundancies
    out = Circuit()
    for q in circ.qubits:
        out.add_qubit(q)
    for cmd in circ.get_commands():
        if cmd.op.type == OpType.ZZPhase:
            a, b = cmd.qubits
            out.H(a).H(b)
            out.XXPhase(cmd.op.params[0], a, b)
            out.H(a).H(b)
        else:
            out.add_gate(cmd.op, cmd.args)
    out.add_phase(circ.phase)
    AutoRebase(one_q | {OpType.XXPhase}).apply(out)
    AutoSquash(one_q).apply(out)
    RemoveRedundancies().apply(out)
    return out


# --------------------------------------------------------------------------- Cirq
def compile_cirq(qasm_text, target, level, seed):
    import cirq
    import networkx as nx
    from cirq.contrib.qasm_import import circuit_from_qasm

    if target.gate_2q != "cz":
        raise Unsupported(f"Cirq has no built-in compilation target gateset for '{target.gate_2q}' "
                          "(only CZ and sqrt-iSWAP families).")
    (circ, parse_s, _) = _timed(lambda: circuit_from_qasm(qasm_text))
    # cirq names qubits q_0, q_1 ...; include declared-but-unused qubits too
    nlog = qasm2.loads(qasm_text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS).num_qubits
    logical = [cirq.NamedQubit(f"q_{i}") for i in range(nlog)]
    gs = cirq.CZTargetGateset()

    present = set(circ.all_qubits())

    def run():
        c1 = cirq.optimize_for_target_gateset(circ, gateset=gs)
        if level == "max":
            c1 = cirq.eject_phased_paulis(c1)
            c1 = cirq.eject_z(c1)
            c1 = cirq.drop_negligible_operations(c1)
            c1 = cirq.drop_empty_moments(c1)
        if target.all_to_all:
            phys = {q: cirq.LineQubit(i) for i, q in enumerate(logical) if q in present}
            routed = c1.transform_qubits(lambda q: phys[q])
            imap = phys
            smap = {p: p for p in phys.values()}
        else:
            g = nx.Graph()
            g.add_nodes_from(cirq.LineQubit(i) for i in range(target.num_qubits))
            g.add_edges_from((cirq.LineQubit(a), cirq.LineQubit(b)) for a, b in target.edges)
            router = cirq.RouteCQC(g)
            routed, imap, smap = router.route_circuit(c1)
        c2 = cirq.optimize_for_target_gateset(routed, gateset=gs)
        if level == "max":
            c2 = cirq.eject_z(cirq.eject_phased_paulis(c2))
            c2 = cirq.drop_negligible_operations(cirq.drop_empty_moments(c2))
        return c2, imap, smap

    (c2, imap, smap), wall, cpu = _timed(run)
    width = target.num_qubits
    line = [cirq.LineQubit(i) for i in range(width)]
    # qubits absent from the circuit are not in Cirq's maps: park them on free
    # physical qubits (they stay idle)
    used = {imap[q].x for q in logical if q in imap}
    free = [p for p in range(width) if p not in used]
    init, final = [], []
    for q in logical:
        if q in imap:
            init.append(imap[q].x)
            final.append(smap[imap[q]].x)
        else:
            p = free.pop(0)
            init.append(p)
            final.append(p)
    qc = _cirq_to_qiskit(c2, line)
    return CompileOutput(qc, init, final, wall, cpu, parse_s, cirq.__version__)


def _cirq_to_qiskit(circuit, qubits):
    """Convert via unitary of each op (1q -> unitary gate, CZ -> cz) to keep it exact."""
    import cirq
    from qiskit.circuit.library import UnitaryGate
    idx = {q: i for i, q in enumerate(qubits)}
    qc = QuantumCircuit(len(qubits))
    for op in circuit.all_operations():
        qs = [idx[q] for q in op.qubits]
        if len(qs) == 1:
            qc.append(UnitaryGate(cirq.unitary(op), check_input=False), qs)
        elif isinstance(op.gate, cirq.CZPowGate) and abs(((op.gate.exponent - 1) % 2.0)) < 1e-9:
            qc.cz(qs[0], qs[1])  # CZ**(1+2k) == CZ
        else:
            u = cirq.unitary(op)
            # cirq big-endian -> qiskit little-endian: reverse qubit order
            qc.append(UnitaryGate(u, check_input=False, label=str(op.gate)), qs[::-1])
    return qc


# --------------------------------------------------------------------------- QUMO
def compile_qumo(qasm_text, target, level, seed):
    import qumo_py

    (circ, parse_s, _) = _timed(lambda: qumo_py.load_qasm(qasm_text))
    nlog = circ.num_qubits()
    opt = 2 if level == "default" else 3
    edges = [] if target.all_to_all else [list(e) for e in target.edges]
    width = nlog if target.all_to_all else target.num_qubits
    res, wall, cpu = _timed(lambda: qumo_py.transpile(circ, width, edges, target.basis, opt, seed))
    out, init, final = res
    qc = qasm2.loads(out.to_qasm2(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    return CompileOutput(qc, list(init)[:nlog], list(final)[:nlog], wall, cpu, parse_s, qumo_py.version())


def warm_up(sdk):
    """Pay one-time import/plugin-loading costs in the parent process (before
    fork) so per-job timings measure compilation, not first-call latency."""
    from targets import make_target
    text = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nh q[0];\ncx q[0],q[1];\nccx q[0],q[1],q[2];\nrz(0.3) q[2];\n'
    for kind in ("linear_cx", "ion_all_to_all", "ion_zone_ring", "sc_heavy_hex"):
        for level in ("default", "max"):
            try:
                SDKS[sdk](text, make_target(kind, 3), level, 1)
            except Exception:  # unsupported combos / adapter bugs surface per job
                pass


SDKS = {
    "qumo": compile_qumo,
    "qiskit": compile_qiskit,
    "tket": compile_tket,
    "cirq": compile_cirq,
}
