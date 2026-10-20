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
import json
import os
import re
import time
from dataclasses import dataclass, field

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
    extra: dict = field(default_factory=dict)  # SDK-specific record fields (e.g. AI pass statistics)


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


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SYNTH_PASSES = ("clifford_synth", "linear_synth", "perm_synth", "pauli_network")


def ai_stats(report_json):
    """Per-technique statistics from a QUMO CompileResult JSON report: for each
    AI synthesis pass the summed blocks / proposed / verified / replaced /
    saved_cost / rejected counts (parsed from the pass notes), whether the
    learned router won its portfolio against SABRE, and the winning pipeline
    (at AI level 3 the O3 portfolio member may win)."""
    rep = json.loads(report_json)
    pipeline = rep.get("pipeline", "")
    out = {"route_ai_kept": 0, "route_ai_tried": 0}
    out["winner"] = "ai" if any(p in pipeline for p in _SYNTH_PASSES) or "policy=" in pipeline else "heuristic"
    for p in rep.get("passes", []):
        name = p.get("pass", "")
        note = p.get("note", "")
        base = name.split("(")[0]
        if base == "fixpoint":
            # nested synthesis passes: "| <pass> blocks=.. replaced=.." sections of the fixpoint note
            for sec in note.split(" | ")[1:]:
                sp = sec.split(" ")[0]
                if sp in _SYNTH_PASSES:
                    d = out.setdefault(sp, {})
                    for k, v in re.findall(r"(blocks|proposed|verified|replaced|saved_cost|rejected)=(-?\d+)", sec):
                        d[k] = d.get(k, 0) + int(v)
            continue
        if base in _SYNTH_PASSES:
            d = out.setdefault(base, {})
            for k, v in re.findall(r"(blocks|proposed|verified|replaced|saved_cost|saved_depth|rejected)=(-?\d+)", note):
                d[k] = d.get(k, 0) + int(v)
            d["2q_delta"] = d.get("2q_delta", 0) + (p["2q"][1] - p["2q"][0])
        if base == "route" and "ai-route" in note:
            out["route_ai_tried"] += 1
            if "kept ai-route" in note:
                out["route_ai_kept"] += 1
    return out


def compile_qumo_ai(qasm_text, target, level, seed):
    """QUMO hybrid heuristic + AI preset: default -> transpile_ai level 2,
    max -> level 3 (portfolio with O3). Models come from $QUMO_MODEL_DIR,
    defaulting to the repository's models/ai_transpiler."""
    import qumo_py

    os.environ.setdefault("QUMO_MODEL_DIR", os.path.join(_REPO, "models", "ai_transpiler"))
    (circ, parse_s, _) = _timed(lambda: qumo_py.load_qasm(qasm_text))
    nlog = circ.num_qubits()
    lvl = 2 if level == "default" else 3
    edges = [] if target.all_to_all else [list(e) for e in target.edges]
    width = nlog if target.all_to_all else target.num_qubits
    res, wall, cpu = _timed(lambda: qumo_py.transpile_ai_report(circ, width, edges, target.basis, lvl, seed))
    out, init, final, report = res
    qc = qasm2.loads(out.to_qasm2(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    return CompileOutput(qc, list(init)[:nlog], list(final)[:nlog], wall, cpu, parse_s, qumo_py.version(),
                         {"ai": ai_stats(report)})


# --------------------------------------------------------------------------- Qiskit + IBM AI (optional baseline)
# IBM-trained external baseline ONLY. qiskit-ibm-transpiler 0.18.x local mode
# (routing: a Rust extension shipped in the wheel; synthesis: qiskit-gym RL
# policies) downloads IBM's model weights from Hugging Face
# (qiskit/ai-transpiler_*) on first use. It is installed only in a scratch
# venv (bench/ai_vs_ibm/README.md) and is never a QUMO dependency; set
# HF_HUB_OFFLINE=1 once the models are cached so nothing is fetched while
# timing. No IBM Quantum cloud service is used (local_mode=True everywhere).
# Targets: IBM's local routing accepts any coupling map; its synthesis models
# only match heavy-hex-like subgraphs (lines / trees of degree <= 3, 12-rings),
# so the baseline is restricted to IBM-like CX / CZ targets.
_IBM_AI_TARGETS = ("sc_heavy_hex", "linear_cx")


def _import_ibm_ai():
    """Import qiskit-ibm-transpiler (optional). 0.18.0 imports
    `qiskit.synthesis.linear.linear_matrix_utils`, a module Qiskit 2.x removed;
    the one function it takes from it is still public
    (`qiskit.synthesis.linear.random_invertible_binary_matrix`), so alias the
    module. It is only used by IBM's random-test-circuit helpers, never on the
    transpilation path. IBM's loggers are set to INFO at import; keep errors only."""
    import logging
    import sys
    import types
    name = "qiskit.synthesis.linear.linear_matrix_utils"
    if name not in sys.modules:
        try:
            import importlib
            importlib.import_module(name)
        except ImportError:
            import qiskit.synthesis.linear as lin
            mod = types.ModuleType(name)
            mod.random_invertible_binary_matrix = lin.random_invertible_binary_matrix
            sys.modules[name] = mod
    import qiskit_ibm_transpiler
    for lname in list(logging.root.manager.loggerDict):
        if lname.startswith("qiskit_ibm_transpiler"):
            logging.getLogger(lname).setLevel(logging.ERROR)
    return qiskit_ibm_transpiler


def ibm_ai_pass_manager(cmap, basis, level, seed, variant="default"):
    """IBM AI pass managers (local mode), all on Qiskit preset level `level`:

    * default: `generate_ai_pass_manager(optimization_level=level,
      ai_optimization_level=level, ai_layout_mode="optimize")` - AIRouting
      replaces Sabre layout + routing; post-routing AI linear-function
      synthesis (IBM's documented hybrid pass manager);
    * route: the same without AI synthesis (AIRouting only);
    * full: AIRouting + the whole documented collection / synthesis list
      (CollectCliffords + AICliffordSynthesis, CollectLinearFunctions forward
      and backward + AILinearFunctionSynthesis, CollectPermutations +
      AIPermutationSynthesis, CollectPauliNetworks + AIPauliNetworkSynthesis);
    * synth: Qiskit's own Sabre layout / routing + that synthesis list.
    """
    _import_ibm_ai()
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_transpiler import generate_ai_pass_manager
    from qiskit_ibm_transpiler.ai.collection import (CollectCliffords, CollectLinearFunctions,
                                                     CollectPermutations, CollectPauliNetworks)
    from qiskit_ibm_transpiler.ai.synthesis import (AICliffordSynthesis, AILinearFunctionSynthesis,
                                                    AIPermutationSynthesis, AIPauliNetworkSynthesis)

    opts = {"basis_gates": list(basis), "seed_transpiler": seed}
    if variant in ("default", "route", "full"):
        pm = generate_ai_pass_manager(coupling_map=cmap, ai_optimization_level=level, optimization_level=level,
                                      ai_layout_mode="optimize", include_ai_synthesis=(variant == "default"),
                                      qiskit_transpile_options=opts)
    elif variant == "synth":
        pm = generate_preset_pass_manager(optimization_level=level, coupling_map=cmap, **opts)
    else:
        raise ValueError(variant)
    if variant in ("default", "route", "full"):
        # Workaround for an IBM bug (0.18.0 with Qiskit 2.5): AIRouting
        # overwrites property_set["original_qubit_indices"] with the routed
        # DAG's new qubits, so when the O2/O3 init stage's ElidePermutations has
        # elided a SWAP (it records a virtual permutation keyed by the original
        # qubits) TranspileLayout.from_property_set raises KeyError; any input
        # with a SWAP / permutation gate fails. Dropping ElidePermutations for
        # the AIRouting pipelines keeps them running (input SWAPs are then
        # routed instead of relabelled); minimal repro in bench/ai_vs_ibm/README.md.
        for i in reversed(range(len(pm.init._tasks))):
            if any(type(t).__name__ == "ElidePermutations" for t in pm.init._tasks[i]):
                pm.init.remove(i)
        # record whether AIRouting actually ran (it is skipped when VF2Layout
        # finds a perfect layout); see compile_qiskit_ibm_ai
        flag = {"ran": False}
        from qiskit_ibm_transpiler.ai.routing import AIRouting

        def _mark(t):
            if isinstance(t, AIRouting):
                run0 = t.run

                def run(dag, _run0=run0):
                    flag["ran"] = True
                    return _run0(dag)
                t.run = run
            for sub in (getattr(t, "tasks", None) or []):
                _mark(sub)
        for tl in pm.layout._tasks:
            for t in tl:
                _mark(t)
        pm._ibm_airouting = flag
    if variant in ("full", "synth"):
        commute = level >= 3
        lf = AILinearFunctionSynthesis(coupling_map=cmap, local_mode=True)
        pm.post_routing = PassManager([
            CollectCliffords(), AICliffordSynthesis(coupling_map=cmap, local_mode=True),
            CollectLinearFunctions(do_commutative_analysis=commute), lf,
            CollectLinearFunctions(do_commutative_analysis=commute, collect_from_back=True), lf,
            CollectPermutations(), AIPermutationSynthesis(coupling_map=cmap, local_mode=True),
            CollectPauliNetworks(), AIPauliNetworkSynthesis(coupling_map=cmap, local_mode=True),
        ])
    return pm


def compile_qiskit_ibm_ai(qasm_text, target, level, seed, variant="default"):
    """Qiskit + IBM AI transpiler, local mode: default -> AI level 2 on Qiskit
    O2, max -> AI level 3 on Qiskit O3 (`ibm_ai_pass_manager`). IBM's learned
    routing is not seedable (`seed` only reaches Qiskit's own passes)."""
    try:
        ibm = _import_ibm_ai()
    except Exception as e:  # not installed: optional baseline
        raise Unsupported(f"qiskit-ibm-transpiler not installed ({type(e).__name__}); optional IBM-trained baseline")
    from qiskit.transpiler import CouplingMap

    if target.gate_2q not in ("cx", "cz") or not (target.name in _IBM_AI_TARGETS or target.name.startswith("fake_")):
        raise Unsupported("IBM AI baseline is run on heavy-hex / IBM-like (CX/CZ) targets only")
    (qc, parse_s, _) = _timed(lambda: qasm2.loads(qasm_text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))
    n_log = qc.num_qubits
    opt = level if isinstance(level, int) else (2 if level == "default" else 3)
    cmap = CouplingMap(sorted(target.edge_set))
    pm = ibm_ai_pass_manager(cmap, target.basis, opt, seed, variant)
    tqc, wall, cpu = _timed(lambda: pm.run(qc))
    extra = {}
    if tqc.layout is not None:
        init = tqc.layout.initial_index_layout(filter_ancillas=True)[:n_log]
        final = tqc.layout.final_index_layout(filter_ancillas=True)[:n_log]
        if getattr(pm, "_ibm_airouting", {}).get("ran"):
            # IBM bug (0.18.0): when AIRouting ran, the reported initial layout
            # is the inverse permutation of the one the routed circuit uses
            # (the final layout is right). Measured on every misreported case:
            # the true placement of logical v is the position of v in the full
            # reported initial layout. Correct it so that layout-aware checks
            # (and users) see the right placement; the circuit is unchanged.
            full = list(tqc.layout.initial_index_layout())
            inv = [0] * len(full)
            for v, p in enumerate(full):
                inv[p] = v
            fixed = inv[:n_log]
            extra["ibm_initial_layout_corrected"] = list(fixed) != list(init)
            init = fixed
    else:
        init = list(range(n_log))
        final = list(range(n_log))
    return CompileOutput(tqc, list(init), list(final), wall, cpu, parse_s,
                         f"qiskit {qiskit.__version__} + qiskit-ibm-transpiler {ibm.__version__}", extra)


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
    "qumo-ai": compile_qumo_ai,
    "qiskit+ibm-ai": compile_qiskit_ibm_ai,
    "qiskit": compile_qiskit,
    "tket": compile_tket,
    "cirq": compile_cirq,
}
