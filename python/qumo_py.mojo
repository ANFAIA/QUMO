"""Python bindings for QUMO (CPython extension built from Mojo).

    mojo build -I . --emit shared-lib -o build/qumo_py.so python/qumo_py.mojo

Exposes opaque `Circuit` and `Program` handles plus free functions. Kept
deliberately small and stable: build/load circuits, compile them for any target
(coupling map + basis as plain Python data, vendor-neutral), inspect
statistics, simulate. A `Program` adds classical variables and control flow at
run time: `load_program`, `sample_program`, `transpile_program`. Used by the
Benchpress `qumo_gym`, the hardware plugins and the `qumo` Python package
(python/qumo, docs/PYTHON.md), which gives these functions keyword arguments,
defaults and `QumoError`.

Conventions: every function and method is registered with a docstring whose
first line is its call signature (`help(qumo_py.sample)`). Errors are raised as
`Exception("QM-Exxx: message. Hint: ...")`; Mojo cannot raise another Python
exception type, so a wrong argument type is QM-E911 here and a `TypeError` in
the package. Existing functions keep their names and positional arguments;
new behaviour is added as keyword arguments or new functions.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort, setenv, unsetenv
from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.ir.mc import append_mcx
from qumo.sim.ptsbe import sample_ptsbe, plan_trajectories
from qumo.ir.noise_inject import inject_noise
from qumo.exec.quantinuum_api import qtm_native_circuit, qtm_hqc, qtm_submit, qtm_status, qtm_result, qtm_devices
from qumo.exec.ibm_api import ibm_backends, ibm_target, ibm_isa_payload, ibm_submit, ibm_status, ibm_result
from qumo.exec.speqtrum_api import (
    SpeQtrumAuth, checked_base, payload_document, unsafe_construct, decode_result,
    speqtrum_devices, speqtrum_execute, speqtrum_status, speqtrum_result,
)
from qumo.exec.ionq_api import ionq_request, ionq_counts, ionq_submit, ionq_status, ionq_result, ionq_backends
from qumo.sim.runner import SampleResult
from qumo.sim.branching import sample_branching
from qumo.sim.density import density_evolve
from qumo.compiler.pbc import to_pbc
from qumo.qec.characterization import ReadoutCalibration
from qumo.qec.mthree import mitigate_counts_m3
from qumo.qec.codes import to_stim
from qumo.qec.dem import detector_error_model
from qumo.qec.decoders import MatchingGraph
from qumo.qec.sparse_blossom import SparseBlossomDecoder
from qumo.qec.correlated import CorrelatedMatchingDecoder
from qumo.qec.stim_text import from_stim, parse_dem, sample_dem
from qumo.qec.lint import missing_detectors, count_determined_measurements
from qumo.ai.json import JsonWriter
from std.collections import Dict
from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary, op_from_name
from qumo.ir.pauli import parse_pauli_sum, PauliSum
from qumo.sim.runner import sample, evolve
from qumo.ir.program import Program
from qumo.ir.qasm import draw_program, parse_qasm, load_any_with_warnings, load_qasm, to_qasm
from qumo.ir.qasm2 import to_qasm2
from qumo.sim.program_runner import DEFAULT_MAX_ITERATIONS, sample_program
from qumo.compiler.program_compile import compile_program
from qumo.ai.describe import program_from_json, program_to_json
from qumo.ai.json import parse_json
from qumo.ai.diagnostics import diagnostic_from_error, lint_program
from qumo.exec.dispatch import sample_on
from qumo.gpu.device import query_gpu
from qumo.compiler.pass_manager import transpile, compile_spec, transpile_ai, CompileResult
from qumo.algorithms.gradients import batch_value_and_gradient, Grad
from qumo.algorithms.gradients_device import value_and_gradient_on, expectations_on
from qumo.algorithms.sqd import subspace_ground_state
from qumo.exec.orchestrator import Workflow, Device
from qumo.qec.codes import generate_code, NoiseParams, CodeCircuit
from qumo.qec.dem import detector_error_model, DetectorErrorModel
from qumo.qec.decoders import MatchingGraph, decode_samples, graphlike_dem
from qumo.qec.sparse_blossom import SparseBlossomDecoder
from qumo.sim.frame import FrameSimulator, DetectorSamples, BATCH, FW
from qumo.gpu.frame_gpu import GpuFrameSampler
from qumo.gpu.device import gpu_available
from std.sys.defines import is_defined
from std.runtime import parallelism_level
from max.algorithm import parallelize
from qumo.sim.statevector import StateVector
from qumo.compiler.target import Target, parse_target
from qumo.algorithms.variational import vqe, qaoa_maxcut
from qumo.algorithms.hamiltonians import (
    Graph, maxcut, transverse_field_ising, heisenberg_xxz, fermi_hubbard, h2_sto3g, ground_energy,
)
from qumo.algorithms.library import (
    ghz, qft, hardware_efficient_ansatz, qaoa_maxcut_ansatz, uccsd_ansatz, h2_ansatz, trotter,
)
from qumo.qec.experiments import memory_experiment, threshold_sweep_results, ExperimentResult, canonical_decoder_name
from qumo.ir.gates import op_num_params
from qumo.ir.param import Param
from qumo.bench.circuits import (
    quantum_volume, efficient_su2, dtc_unitary, random_clifford_circuit,
    bv_all_ones, trivial_bvlike_circuit, multi_control_circuit, pauli_twirl_2q,
    hamiltonian_circuit_from_labels,
)


def _type_name(x: PythonObject) raises -> String:
    return String(py=Python.import_module("builtins").type(x).__name__)


def _is_str(x: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    return Bool(py=b.isinstance(x, b.str))


def _int_arg(x: PythonObject, where: String, arg: String) raises -> Int:
    """A whole number (int, bool, numpy integer): QM-E911 for '100' or 100.5, which
    Int(py=) would convert without a word."""
    var op = Python.import_module("operator")
    var v = 0
    var ok = True
    try:
        v = Int(py=op.index(x))
    except:
        ok = False
    if not ok:
        raise Error("QM-E911: " + where + "() argument '" + arg + "' must be an int, got " + _type_name(x) + ". Hint: pass a whole number, e.g. " + arg + "=1000.")
    return v


comptime PACKED_C_IF = -1  # extend_packed pseudo-operation: condition the last instruction
comptime PACKED_WIDTH = 5  # ints per operation: kind, q0, q1, q2, aux


struct PyCircuit(Movable, Writable):
    var c: Circuit

    def __init__(out self, var c: Circuit):
        self.c = c^

    @staticmethod
    def _get(py_self: PythonObject) raises -> Pointer[PyCircuit, MutAnyOrigin]:
        return py_self.downcast_value_ptr[Self]().as_unsafe_any_origin()

    @staticmethod
    def num_qubits(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.num_qubits

    @staticmethod
    def num_clbits(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.num_clbits

    @staticmethod
    def size(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.num_unitary_gates()

    @staticmethod
    def num_parameters(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.num_params()

    @staticmethod
    def depth(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.depth()

    @staticmethod
    def count_ops(py_self: PythonObject) raises -> PythonObject:
        ref c = Self._get(py_self)[].c
        var counts = List[Int](length=Op.NUM_OPS, fill=0)
        for g in c.ops:
            counts[g.kind] += 1
        var d = Python.dict()
        for k in range(Op.NUM_OPS):
            if counts[k] > 0:
                d[op_name(k)] = counts[k]
        return d

    @staticmethod
    def count_gate(py_self: PythonObject, name: PythonObject) raises -> PythonObject:
        var k = op_from_name(String(py=name))
        if k < 0:
            return 0
        return Self._get(py_self)[].c.count(k)

    @staticmethod
    def depth_of(py_self: PythonObject, name: PythonObject) raises -> PythonObject:
        """Depth counting only gates named `name` (e.g. 2q depth)."""
        ref c = Self._get(py_self)[].c
        var names = List[String]()
        if _is_str(name):
            names.append(String(py=name))
        else:
            var ok = True
            try:
                for item in name:
                    if not _is_str(item):
                        ok = False
                        break
                    names.append(String(py=item))
            except:
                ok = False
            if not ok or len(names) == 0:
                raise Error("QM-E911: depth_of() takes a gate name or a list of gate names, got " + _type_name(name) + ". Hint: e.g. depth_of('cz') or depth_of(['cx', 'cz']).")
        var counted = List[Bool](length=Op.NUM_OPS, fill=False)
        for nm in names:
            var k = op_from_name(nm)
            if k < 0:
                raise Error("QM-E913: depth_of() got the unknown gate name '" + nm + "'. Hint: use lower-case names such as 'cx', 'cz', 'ecr', 'rzz'.")
            counted[k] = True
        var level = List[Int](length=c.num_qubits, fill=0)
        var d = 0
        for g in c.ops:
            if not op_is_unitary(g.kind):
                continue
            var a = op_arity(g.kind)
            var l = 0
            for j in range(a):
                l = max(l, level[g.qubit(j)])
            if counted[g.kind]:
                l += 1
            for j in range(a):
                level[g.qubit(j)] = l
            d = max(d, l)
        return d

    @staticmethod
    def two_qubit_edges(py_self: PythonObject) raises -> PythonObject:
        """List of (q0, q1) for every 2-qubit unitary (for validation)."""
        ref c = Self._get(py_self)[].c
        var out = Python.list()
        for g in c.ops:
            if op_is_unitary(g.kind) and op_arity(g.kind) == 2:
                out.append(Python.tuple(g.q0, g.q1))
        return out

    @staticmethod
    def copy(py_self: PythonObject) raises -> PythonObject:
        return PythonObject(alloc=PyCircuit(Self._get(py_self)[].c.copy()))

    @staticmethod
    def compose(py_self: PythonObject, other: PythonObject) raises -> PythonObject:
        """Return self followed by other (new circuit)."""
        var out = Self._get(py_self)[].c.copy()
        out.compose(Self._get(other)[].c)
        return PythonObject(alloc=PyCircuit(out^))

    @staticmethod
    def bind(py_self: PythonObject, values: PythonObject) raises -> PythonObject:
        var vals = List[Float64]()
        for v in values:
            vals.append(Float64(py=v))
        return PythonObject(alloc=PyCircuit(Self._get(py_self)[].c.bind(vals)))

    @staticmethod
    def to_qasm2(py_self: PythonObject) raises -> PythonObject:
        return to_qasm2(Self._get(py_self)[].c)

    @staticmethod
    def to_qasm3(py_self: PythonObject) raises -> PythonObject:
        return to_qasm(Self._get(py_self)[].c)

    @staticmethod
    def text(py_self: PythonObject) raises -> PythonObject:
        return String(Self._get(py_self)[].c)

    @staticmethod
    def name(py_self: PythonObject) raises -> PythonObject:
        return String(Self._get(py_self)[].c.name)

    @staticmethod
    def num_ops(py_self: PythonObject) raises -> PythonObject:
        return len(Self._get(py_self)[].c.ops)

    @staticmethod
    def num_measurements(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.num_measurements

    @staticmethod
    def num_2q_gates(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.num_2q_gates()

    @staticmethod
    def t_count(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.t_count()

    @staticmethod
    def parameter_names(py_self: PythonObject) raises -> PythonObject:
        var out = Python.list()
        for nm in Self._get(py_self)[].c.param_names:
            out.append(String(nm))
        return out

    @staticmethod
    def has_noise(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.has_noise()

    @staticmethod
    def is_clifford(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.is_clifford()

    @staticmethod
    def is_dynamic(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.is_dynamic()

    @staticmethod
    def inverse(py_self: PythonObject) raises -> PythonObject:
        return PythonObject(alloc=PyCircuit(Self._get(py_self)[].c.inverse()))

    @staticmethod
    def extend_packed(py_self: PythonObject, ops_addr: PythonObject, num_ops: PythonObject, params_addr: PythonObject,
                      num_params: PythonObject, new_parameters: PythonObject) raises -> PythonObject:
        ref c = Self._get(py_self)[].c
        for nm in new_parameters:
            _ = c.parameter(String(py=nm))
        var n = Int(py=num_ops)
        var np_total = Int(py=num_params)
        if n == 0:
            return len(c.ops)
        var ip = Pointer[Int, MutAnyOrigin](unsafe_from_address=Int(py=ops_addr))
        var fp = Pointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=params_addr))
        var cur = 0
        var nq = c.num_qubits
        var declared = len(c.param_names)
        for i in range(n):
            var base = PACKED_WIDTH * i
            var kind = ip[unsafe_offset=base]
            var q0 = ip[unsafe_offset=base + 1]
            var q1 = ip[unsafe_offset=base + 2]
            var q2 = ip[unsafe_offset=base + 3]
            var aux = ip[unsafe_offset=base + 4]
            if kind == PACKED_C_IF:
                if len(c.ops) == 0 or q0 < 0:
                    raise Error("QM-E913: extend_packed() operation " + String(i) + " is a condition with nothing to condition. Hint: c_if follows the instruction it controls.")
                c.c_if(q0, q1)
                continue
            if (
                kind < 0 or kind >= Op.NUM_OPS or kind == Op.KRAUS1 or kind == Op.KRAUS2 or kind == Op.PAULI_CHANNEL_2
                or kind == Op.CORRELATED_ERROR or kind == Op.DETECTOR or kind == Op.OBSERVABLE
            ):
                raise Error("QM-E913: extend_packed() operation " + String(i) + " has kind " + String(kind) + ", which the packed form does not carry. Hint: use op_table() kinds; load Kraus channels, correlated errors and detectors from OpenQASM text.")
            var a = op_arity(kind)
            if (a >= 1 and (q0 < 0 or q0 >= nq)) or (a >= 2 and (q1 < 0 or q1 >= nq)) or (a >= 3 and (q2 < 0 or q2 >= nq)):
                raise Error("QM-E001: instruction #" + String(len(c.ops)) + " '" + op_name(kind) + "' uses a qubit outside 0.." + String(nq - 1) + " (circuit has " + String(nq) + " qubits).")
            if kind == Op.MEASURE:
                if aux < 0:
                    raise Error("QM-E913: extend_packed() operation " + String(i) + " measures into classical bit " + String(aux) + ". Hint: classical bits start at 0.")
                c.measure(q0, aux)
                continue
            var k = op_num_params(kind)
            if cur + 3 * k > np_total:
                raise Error("QM-E913: extend_packed() ran out of parameter values at operation " + String(i) + ". Hint: pass 3 numbers (offset, index, coeff) per gate angle.")
            var ps: List[Param] = [Param(0.0), Param(0.0), Param(0.0)]
            for j in range(k):
                var index = Int(fp[unsafe_offset=cur + 1])
                if index >= declared:
                    raise Error("QM-E913: extend_packed() operation " + String(i) + " uses parameter " + String(index) + " but the circuit declares " + String(declared) + ". Hint: list new names in new_parameters.")
                ps[j] = Param(fp[unsafe_offset=cur], index if index >= 0 else -1, fp[unsafe_offset=cur + 2] if index >= 0 else 0.0)
                cur += 3
            c.add(kind, q0, q1 if a >= 2 else -1, q2 if a >= 3 else -1, ps[0], ps[1], ps[2])
        return len(c.ops)

    def write_to(self, mut w: Some[Writer]):
        w.write("<qumo.Circuit qubits=", self.c.num_qubits, " ops=", len(self.c.ops), ">")

    def write_repr_to(self, mut w: Some[Writer]):
        self.write_to(w)


struct PyProgram(Movable, Writable):
    """A circuit with classical variables and control flow at run time."""

    var p: Program
    var warnings: List[String]  # of the parser that read it

    def __init__(out self, var p: Program, var warnings: List[String] = List[String]()):
        self.p = p^
        self.warnings = warnings^

    @staticmethod
    def _get(py_self: PythonObject) raises -> Pointer[PyProgram, MutAnyOrigin]:
        return py_self.downcast_value_ptr[Self]().as_unsafe_any_origin()

    @staticmethod
    def num_qubits(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].p.num_qubits()

    @staticmethod
    def num_clbits(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].p.num_clbits()

    @staticmethod
    def num_parameters(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].p.num_params()

    @staticmethod
    def has_runtime_regions(py_self: PythonObject) raises -> PythonObject:
        """False when the program is one plain circuit."""
        return Self._get(py_self)[].p.has_runtime_region()

    @staticmethod
    def to_circuit(py_self: PythonObject) raises -> PythonObject:
        """The plain circuit of a program without runtime regions (QM-E585 otherwise)."""
        var c = Self._get(py_self)[].p.try_flatten()
        if not c:
            raise Error("QM-E585: this program has classical variables or control flow at run time, so it is not one Circuit. Hint: use sample_program and transpile_program.")
        return PythonObject(alloc=PyCircuit(c.take()))

    @staticmethod
    def to_qasm3(py_self: PythonObject) raises -> PythonObject:
        return to_qasm(Self._get(py_self)[].p)

    @staticmethod
    def to_json(py_self: PythonObject) raises -> PythonObject:
        """Program JSON (qumo.program.v1) as a string."""
        return program_to_json(Self._get(py_self)[].p)

    @staticmethod
    def lint(py_self: PythonObject) raises -> PythonObject:
        """Warnings of the parser and of the program as [(code, message), ...]."""
        var out = Python.list()
        for s in Self._get(py_self)[].warnings:
            var d = diagnostic_from_error(s)
            out.append(Python.tuple(d.code, d.message))
        for d in lint_program(Self._get(py_self)[].p):
            out.append(Python.tuple(d.code, d.message))
        return out

    @staticmethod
    def text(py_self: PythonObject) raises -> PythonObject:
        return String(Self._get(py_self)[].p)

    def write_to(self, mut w: Some[Writer]):
        w.write("<qumo.Program qubits=", self.p.num_qubits(), " statements=", len(self.p.tape), ">")

    def write_repr_to(self, mut w: Some[Writer]):
        self.write_to(w)


def _wrap(var c: Circuit) raises -> PythonObject:
    return PythonObject(alloc=PyCircuit(c^))


def py_load_program(text: PythonObject) raises -> PythonObject:
    """Parse OpenQASM 2/3 into a Program: anything a Circuit holds keeps its exact
    meaning, and while, break, continue and classical variables are added."""
    var warnings = List[String]()
    var p = load_any_with_warnings(String(py=text), warnings)
    return PythonObject(alloc=PyProgram(p^, warnings^))


def py_program_from_json(text: PythonObject) raises -> PythonObject:
    """Inverse of Program.to_json. Malformed input raises QM-E587."""
    return PythonObject(alloc=PyProgram(program_from_json(parse_json(String(py=text)))))


def py_draw_program(prog: PythonObject, width: PythonObject, style: PythonObject) raises -> PythonObject:
    """Text drawing by blocks: (program, width, style) -> str."""
    return PythonObject(draw_program(PyProgram._get(prog)[].p, Int(py=width), String(py=style)))


def py_sample_program(prog: PythonObject, shots: PythonObject, seed: PythonObject,
                      var **kwargs: PythonObject) raises -> PythonObject:
    """Sample a program -> dict with counts, method, iterations (back-edges per
    shot), outputs ({name: [value per shot]}) and warnings.

    Python: sample_program(prog, shots, seed, params=[], max_iterations=65536, method="auto").
    params: values of the `input` declarations, in order. Past max_iterations a shot
    raises QM-E480. method: 'auto' | 'statevector' | 'tableau' | 'branching' | 'gpu'
    ('gpu' runs on the CPU with QM-W020; 'density', 'ptsbe' and 'frame' are QM-E483).
    """
    var params = List[Float64]()
    var max_iterations = DEFAULT_MAX_ITERATIONS
    var method = String("auto")
    for kv in kwargs.items():
        if kv.key == "params":
            params = _floats(kv.value)
        elif kv.key == "max_iterations":
            max_iterations = Int(py=kv.value)
        elif kv.key == "method":
            method = String(py=kv.value)
        else:
            raise Error("QM-E913: sample_program() got an unexpected keyword argument '" + kv.key + "'. Hint: the keyword arguments are params, max_iterations and method.")
    var r = sample_program(PyProgram._get(prog)[].p, Int(py=shots), UInt64(Int(py=seed)), params, max_iterations, method)
    var counts = Python.dict()
    for kv in r.sample.counts.items():
        counts[kv.key] = kv.value
    var iterations = Python.list()
    for v in r.iterations:
        iterations.append(v)
    var outputs = Python.dict()
    for j in range(len(r.out_names)):
        var vals = Python.list()
        if r.out_is_real[j]:
            for v in r.real_output(r.out_names[j]):
                vals.append(v)
        elif r.out_is_unsigned[j]:
            for v in r.uint_output(r.out_names[j]):
                vals.append(v)
        else:
            for v in r.int_output(r.out_names[j]):
                vals.append(v)
        outputs[r.out_names[j]] = vals
    var warnings = Python.list()
    for w in r.warnings:
        warnings.append(w)
    var d = Python.dict()
    d["counts"] = counts
    d["method"] = r.sample.method
    d["iterations"] = iterations
    d["outputs"] = outputs
    d["warnings"] = warnings
    return d


def py_transpile_program(prog: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                         basis: PythonObject, level: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Compile: (program, num_qubits, edges, basis, level, seed) -> (program, initial, final, report).

    Same arguments as `transpile`. Every block lives on one layout, and each region
    leaves on the layout it entered with. The result has passed verify_program;
    report holds metrics, restored_swaps, blocks_checked and blocks_unchecked."""
    var t = _target(num_qubits, edges, basis)
    var r = compile_program(PyProgram._get(prog)[].p, t, Int(py=level), Int(py=seed))
    var init = Python.list()
    for v in r.initial_layout:
        init.append(v)
    var fin = Python.list()
    for v in r.final_layout:
        fin.append(v)
    var d = Python.dict()
    d["metrics"] = String(r.metrics)
    d["restored_swaps"] = r.restored_swaps
    d["blocks_checked"] = r.blocks_checked
    d["blocks_unchecked"] = r.blocks_unchecked
    return Python.tuple(PythonObject(alloc=PyProgram(r.program.copy())), init, fin, d)


def py_quantum_volume(n: PythonObject, depth: PythonObject, seed: PythonObject) raises -> PythonObject:
    return _wrap(quantum_volume(Int(py=n), Int(py=depth), UInt64(Int(py=seed))))


def py_efficient_su2(n: PythonObject, reps: PythonObject, ent: PythonObject) raises -> PythonObject:
    return _wrap(efficient_su2(Int(py=n), Int(py=reps), String(py=ent)))


def py_dtc_unitary(n: PythonObject, g: PythonObject, seed: PythonObject) raises -> PythonObject:
    return _wrap(dtc_unitary(Int(py=n), Float64(py=g), UInt64(Int(py=seed))))


def py_random_clifford(n: PythonObject, num_gates: PythonObject, seed: PythonObject) raises -> PythonObject:
    return _wrap(random_clifford_circuit(Int(py=n), Int(py=num_gates), UInt64(Int(py=seed))))


def py_bv_all_ones(n: PythonObject) raises -> PythonObject:
    return _wrap(bv_all_ones(Int(py=n)))


def py_bvlike(n: PythonObject) raises -> PythonObject:
    return _wrap(trivial_bvlike_circuit(Int(py=n)))


def py_multi_control(n: PythonObject) raises -> PythonObject:
    return _wrap(multi_control_circuit(Int(py=n)))


def py_empty(n: PythonObject) raises -> PythonObject:
    return _wrap(Circuit(Int(py=n)))


def py_twirl(circ: PythonObject, seed: PythonObject) raises -> PythonObject:
    return _wrap(pauli_twirl_2q(PyCircuit._get(circ)[].c, UInt64(Int(py=seed))))


def py_hamiltonian_circuit(labels: PythonObject, coeffs: PythonObject, time: PythonObject) raises -> PythonObject:
    var ls = List[String]()
    for l in labels:
        ls.append(String(py=l))
    var cs = List[Float64]()
    for c in coeffs:
        cs.append(Float64(py=c))
    return _wrap(hamiltonian_circuit_from_labels(ls, cs, Float64(py=time)))


comptime QUMO_VERSION = "0.1.0"


def py_version() raises -> PythonObject:
    return QUMO_VERSION


def py_load_qasm(text: PythonObject) raises -> PythonObject:
    return _wrap(parse_qasm(String(py=text)))


def py_load_qasm_file(path: PythonObject) raises -> PythonObject:
    return _wrap(load_qasm(String(py=path)))


def _basis_spec(basis: PythonObject) raises -> String:
    """Gate names -> the spec text Target takes. A str is one spec: a basis-set
    name ('sc_cz') or a comma list ('rz,sx,x,cz'), not a sequence of characters."""
    if _is_str(basis):
        return String(py=basis)
    var spec = String()
    for b in basis:
        if spec.byte_length() > 0:
            spec += ","
        spec += String(py=b)
    return spec


def _target(num_qubits: PythonObject, edges: PythonObject, basis: PythonObject) raises -> Target:
    var n = Int(py=num_qubits)
    var spec = _basis_spec(basis)
    var flat = List[Int]()
    for e in edges:
        flat.append(Int(py=e[0]))
        flat.append(Int(py=e[1]))
    if len(flat) == 0:
        return Target.all_to_all(n, spec)
    return Target.from_edges(n, flat, spec)


def py_transpile(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                 basis: PythonObject, level: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Compile: (circuit, num_qubits, edges, basis, level, seed) -> (circuit, initial, final).

    edges: list of (a, b) physical couplings ([] = all-to-all); basis: gate names.
    initial[l] / final[l]: physical qubit of logical l before / after routing.
    """
    var t = _target(num_qubits, edges, basis)
    var r = transpile(PyCircuit._get(circ)[].c, t, Int(py=level), Int(py=seed))
    var init = Python.list()
    for v in r.initial_layout:
        init.append(v)
    var fin = Python.list()
    for v in r.final_layout:
        fin.append(v)
    return Python.tuple(_wrap(r.circuit.copy()), init, fin)


def _apply_calibration(mut t: Target, cal: PythonObject) raises:
    """Keys, all optional: error_1q, readout, t1_us, t2_us (one value per qubit),
    edge_errors ({"a-b": e}), angle_bounds ({"rzz": (lo, hi)})."""
    var empty = List[Float64]()
    var e1 = _floats(cal["error_1q"]) if cal.__contains__("error_1q") else empty.copy()
    var ro = _floats(cal["readout"]) if cal.__contains__("readout") else empty.copy()
    var t1 = _floats(cal["t1_us"]) if cal.__contains__("t1_us") else empty.copy()
    var t2 = _floats(cal["t2_us"]) if cal.__contains__("t2_us") else empty.copy()
    t.set_calibration(e1, empty, ro, t1, t2)
    if cal.__contains__("edge_errors"):
        var d = cal["edge_errors"]
        for key in d:
            var parts = String(py=key).split("-")
            t.set_edge_error(Int(parts[0]), Int(parts[1]), Float64(py=d[key]))
    if cal.__contains__("angle_bounds"):
        var d = cal["angle_bounds"]
        for key in d:
            var k = op_from_name(String(py=key))
            if k < 0:
                raise Error("QM-E610: unknown gate '" + String(py=key) + "' in angle_bounds. Hint: use gate names such as rzz.")
            t.set_angle_bounds(k, Float64(py=d[key][0]), Float64(py=d[key][1]))


def py_transpile_calibrated(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                            basis: PythonObject, level: PythonObject, seed: PythonObject,
                            calibration: PythonObject) raises -> PythonObject:
    """`transpile` with device data: (..., seed, calibration) -> (circuit, initial, final, est_fidelity).

    calibration: dict with optional error_1q, readout, t1_us, t2_us (per qubit),
    edge_errors ({"a-b": e}) and angle_bounds ({"rzz": (lo, hi)}). Uniform data
    changes nothing; non-uniform data adds `post_layout`, bounds add `wrap_angles`.
    """
    var t = _target(num_qubits, edges, basis)
    _apply_calibration(t, calibration)
    var r = transpile(PyCircuit._get(circ)[].c, t, Int(py=level), Int(py=seed))
    var lay = _layouts(r)
    return Python.tuple(_wrap(r.circuit.copy()), lay[0], lay[1], t.estimate_fidelity(r.circuit))


def py_mcx_circuit(num_controls: PythonObject, mode: PythonObject, num_ancillas: PythonObject,
                   clean: PythonObject) raises -> PythonObject:
    """Multi-controlled X on qubits 0..k-1 -> k, ancillas after: (k, mode, ancillas, clean) -> circuit."""
    var k = Int(py=num_controls)
    var na = Int(py=num_ancillas)
    var controls = List[Int]()
    for i in range(k):
        controls.append(i)
    var ancillas = List[Int]()
    for i in range(na):
        ancillas.append(k + 1 + i)
    var c = Circuit(k + 1 + na, name="mcx")
    append_mcx(c, controls, k, ancillas, String(py=mode), Bool(py=clean))
    return _wrap(c^)


def py_draw(circ: PythonObject, width: PythonObject, style: PythonObject) raises -> PythonObject:
    """Text drawing: (circuit, width, style) -> str; style 'ascii' or 'unicode', width 0 = no fold."""
    return PythonObject(PyCircuit._get(circ)[].c.draw(Int(py=width), String(py=style)))


def _counts_dict(r: SampleResult) raises -> PythonObject:
    var d = Python.dict()
    for kv in r.counts.items():
        d[kv.key] = kv.value
    return d


def py_sample_ptsbe(circ: PythonObject, shots: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Static noisy circuit, one simulation per distinct Pauli error pattern -> counts."""
    return _counts_dict(sample_ptsbe(PyCircuit._get(circ)[].c, _int_arg(shots, "sample_ptsbe", "shots"),
                                     UInt64(_int_arg(seed, "sample_ptsbe", "seed"))))


def py_inject_noise(text: PythonObject, p1: PythonObject, p2: PythonObject, pm: PythonObject) raises -> PythonObject:
    """OpenQASM 2 text + gate/measure rates -> (text with #pragma qumo channels, inserted 1q, 2q, readout)."""
    var r = inject_noise(String(py=text), Float64(py=p1), Float64(py=p2), Float64(py=pm))
    return Python.tuple(String(r.text), r.depolarize1, r.depolarize2, r.x_error)


def py_ionq_request(text: PythonObject, shots: PythonObject, noise: PythonObject, seed: PythonObject,
                    name: PythonObject) raises -> PythonObject:
    """OpenQASM 2 -> (IonQ ionq.circuit.v1 JSON, clbit->qubit map, num_clbits, dense qubits)."""
    var r = ionq_request(String(py=text), Int(py=shots), String(py=noise), Int(py=seed), String(py=name))
    var qmap = Python.list()
    for v in r.clbit_qubit:
        qmap.append(v)
    return Python.tuple(String(r.body), qmap, r.num_clbits, r.dense_qubits)


def py_ionq_counts(keys: PythonObject, probs: PythonObject, clbit_qubit: PythonObject, shots: PythonObject,
                   seed: PythonObject) raises -> PythonObject:
    """IonQ histogram (lists of outcome ints and probabilities) -> counts dict over the clbits."""
    var p = List[Float64]()
    for v in probs:
        p.append(Float64(py=v))
    return _py_counts(ionq_counts(_int_list(keys), p, _int_list(clbit_qubit), Int(py=shots), UInt64(Int(py=seed))))


def _int_list(obj: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for v in obj:
        out.append(Int(py=v))
    return out^


def _py_int_list(vals: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for v in vals:
        out.append(v)
    return out


def _py_counts(counts: Dict[String, Int]) raises -> PythonObject:
    var d = Python.dict()
    for kv in counts.items():
        d[kv.key] = kv.value
    return d


def py_ionq_submit(text: PythonObject, shots: PythonObject, noise: PythonObject, seed: PythonObject,
                   name: PythonObject) raises -> PythonObject:
    """Native IonQ submission (POST /jobs; key read from IONQ_API_KEY in Mojo) -> (job id, clbit->qubit map, num_clbits, dense qubits)."""
    var r = ionq_submit(String(py=text), Int(py=shots), String(py=noise), Int(py=seed), String(py=name))
    return Python.tuple(String(r.job_id), _py_int_list(r.request.clbit_qubit), r.request.num_clbits,
                        r.request.dense_qubits)


def py_ionq_status(job_id: PythonObject) raises -> PythonObject:
    """Native GET /jobs/<id> -> QUMO state name."""
    return PythonObject(ionq_status(String(py=job_id)))


def py_ionq_result(job_id: PythonObject, clbit_qubit: PythonObject, shots: PythonObject,
                   seed: PythonObject) raises -> PythonObject:
    """Native GET of the probability histogram, sampled into counts."""
    return _py_counts(ionq_result(String(py=job_id), _int_list(clbit_qubit), Int(py=shots), UInt64(Int(py=seed))))


def py_ionq_backends() raises -> PythonObject:
    """Native GET /backends -> list of (name, status, qubits)."""
    var out = Python.list()
    for r in ionq_backends():
        out.append(Python.tuple(String(r.name), String(r.status), r.qubits))
    return out


def _spq_auth(base: PythonObject, user: PythonObject, key: PythonObject, aud: PythonObject) raises -> SpeQtrumAuth:
    return SpeQtrumAuth(checked_base(String(py=base)), String(py=user), String(py=key), String(py=aud))


def py_spq_payload_document(backend: PythonObject, shots: PythonObject, qasm: PythonObject, nq: PythonObject,
                            nc: PythonObject, initial: PythonObject, final: PythonObject, mclbits: PythonObject,
                            mqubits: PythonObject, auto: PythonObject) raises -> PythonObject:
    """qumo.payload.v1 document (byte-identical to the Python fallback)."""
    return PythonObject(payload_document(String(py=backend), Int(py=shots), String(py=qasm), Int(py=nq), Int(py=nc),
                                         _int_list(initial), _int_list(final), _int_list(mclbits), _int_list(mqubits),
                                         Bool(py=auto)))


def py_spq_unsafe_construct(text: PythonObject) raises -> PythonObject:
    return PythonObject(unsafe_construct(String(py=text)))


def py_spq_decode_result(encoded: PythonObject, job_id: PythonObject) raises -> PythonObject:
    return _py_counts(decode_result(String(py=encoded), String(py=job_id)))


def py_spq_devices(base: PythonObject, user: PythonObject, key: PythonObject, aud: PythonObject) raises -> PythonObject:
    var out = Python.list()
    var auth = _spq_auth(base, user, key, aud)
    for d in speqtrum_devices(auth):
        out.append(Python.tuple(String(d.code), d.nqubits, String(d.status), String(d.notes)))
    return out


def py_spq_execute(base: PythonObject, user: PythonObject, key: PythonObject, aud: PythonObject, device: PythonObject,
                   document: PythonObject, shots: PythonObject) raises -> PythonObject:
    var auth = _spq_auth(base, user, key, aud)
    return PythonObject(speqtrum_execute(auth, String(py=device), String(py=document), Int(py=shots)))


def py_spq_status(base: PythonObject, user: PythonObject, key: PythonObject, aud: PythonObject,
                  job_id: PythonObject) raises -> PythonObject:
    var auth = _spq_auth(base, user, key, aud)
    return PythonObject(speqtrum_status(auth, String(py=job_id)))


def py_spq_result(base: PythonObject, user: PythonObject, key: PythonObject, aud: PythonObject,
                  job_id: PythonObject) raises -> PythonObject:
    var auth = _spq_auth(base, user, key, aud)
    return _py_counts(speqtrum_result(auth, String(py=job_id)))


def _str_list(obj: PythonObject) raises -> List[String]:
    var out = List[String]()
    for v in obj:
        out.append(String(py=v))
    return out^


def _py_float_list(vals: List[Float64]) raises -> PythonObject:
    var out = Python.list()
    for v in vals:
        out.append(v)
    return out


def py_ibm_backends(account: PythonObject, instance: PythonObject) raises -> PythonObject:
    var out = Python.list()
    for r in ibm_backends(String(py=account), String(py=instance)):
        out.append(Python.tuple(String(r.name), r.qubits, String(r.status), r.pending))
    return out


def py_ibm_target(account: PythonObject, instance: PythonObject, backend: PythonObject,
                  fractional: PythonObject) raises -> PythonObject:
    var t = ibm_target(String(py=account), String(py=instance), String(py=backend), Bool(py=fractional))
    var d = Python.dict()
    d["name"] = String(t.name)
    d["num_qubits"] = t.num_qubits
    var basis = Python.list()
    for g in t.basis:
        basis.append(String(g))
    d["basis"] = basis
    d["edges"] = _py_int_list(t.edges)
    d["edge_errors"] = _py_float_list(t.edge_errors)
    d["symmetric"] = t.symmetric
    d["err_1q"] = t.err_1q
    d["err_2q"] = t.err_2q
    d["err_ro"] = t.err_ro
    d["dur_1q"] = t.dur_1q
    d["dur_2q"] = t.dur_2q
    d["dur_ro"] = t.dur_ro
    d["readout"] = _py_float_list(t.readout)
    d["max_shots"] = t.max_shots
    d["simulator"] = t.simulator
    d["processor"] = String(t.processor)
    d["dt"] = t.dt
    d["rep_delay"] = t.rep_delay
    d["fractional"] = t.fractional
    d["native_2q"] = String(t.native_2q)
    d["faulty_qubits"] = _py_int_list(t.faulty_qubits)
    d["faulty_edges"] = t.faulty_edges
    return d


def py_ibm_isa_payload(qasm: PythonObject, num_qubits: PythonObject, basis: PythonObject, edges: PythonObject,
                       symmetric: PythonObject, d1: PythonObject, d2: PythonObject, dr: PythonObject) raises -> PythonObject:
    var p = ibm_isa_payload(String(py=qasm), Int(py=num_qubits), _str_list(basis), _int_list(edges), Bool(py=symmetric),
                            Float64(py=d1), Float64(py=d2), Float64(py=dr))
    return Python.tuple(String(p.qasm3), p.duration_s, p.two_qubit_gates, p.active_qubits)


def py_ibm_submit(account: PythonObject, instance: PythonObject, backend: PythonObject, qasm3: PythonObject,
                  shots: PythonObject, cap: PythonObject) raises -> PythonObject:
    return PythonObject(ibm_submit(String(py=account), String(py=instance), String(py=backend), String(py=qasm3),
                                   Int(py=shots), Int(py=cap)))


def py_ibm_status(account: PythonObject, instance: PythonObject, job_id: PythonObject) raises -> PythonObject:
    var st = ibm_status(String(py=account), String(py=instance), String(py=job_id))
    return Python.tuple(String(st.state), String(st.reason))


def py_ibm_result(account: PythonObject, instance: PythonObject, job_id: PythonObject,
                  num_clbits: PythonObject) raises -> PythonObject:
    return _py_counts(ibm_result(String(py=account), String(py=instance), String(py=job_id), Int(py=num_clbits)))


def py_qtm_native_circuit(qasm: PythonObject) raises -> PythonObject:
    """Compiled OpenQASM 2 -> (pytket circuit JSON, n_qubits, n_clbits, n1q, n2q, n_measure_reset)."""
    var c = qtm_native_circuit(String(py=qasm))
    var regs = Python.list()
    for r in c.registers:
        regs.append(String(r))
    return Python.tuple(String(c.json), c.n_qubits, c.n_clbits, c.n1q, c.n2q, c.n_measure_reset, regs)


def py_qtm_submit(device: PythonObject, project: PythonObject, name: PythonObject, qasm: PythonObject,
                  shots: PythonObject, max_cost: PythonObject) raises -> PythonObject:
    var r = qtm_submit(String(py=device), String(py=project), String(py=name), String(py=qasm), Int(py=shots),
                       Float64(py=max_cost))
    return Python.tuple(String(r.job_id), String(r.project_id), String(r.circuit_id))


def py_qtm_status(job_id: PythonObject) raises -> PythonObject:
    var st = qtm_status(String(py=job_id))
    return Python.tuple(String(st.state), String(st.detail))


def py_qtm_result(job_id: PythonObject, num_clbits: PythonObject) raises -> PythonObject:
    return _py_counts(qtm_result(String(py=job_id), Int(py=num_clbits)))


def py_qtm_devices() raises -> PythonObject:
    var out = Python.list()
    for d in qtm_devices():
        out.append(Python.tuple(String(d.name), d.qubits))
    return out


def py_plan_trajectories(circ: PythonObject, shots: PythonObject, seed: PythonObject) raises -> PythonObject:
    """What sample_ptsbe would run: (distinct patterns, noise sites, error-free shots)."""
    var p = plan_trajectories(PyCircuit._get(circ)[].c, Int(py=shots), UInt64(Int(py=seed)))
    return Python.tuple(p.unique, p.sites, p.error_free)


def py_sample_branching(circ: PythonObject, shots: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Stabilizer branching sampler: near-Clifford circuits on up to 512 qubits -> counts."""
    return _counts_dict(sample_branching(PyCircuit._get(circ)[].c, _int_arg(shots, "sample_branching", "shots"),
                                         UInt64(_int_arg(seed, "sample_branching", "seed"))))


def py_density_probabilities(circ: PythonObject) raises -> PythonObject:
    """Exact outcome probabilities of all qubits from the density matrix."""
    var out = Python.list()
    for p in density_evolve(PyCircuit._get(circ)[].c).probabilities():
        out.append(p)
    return out


def py_density_expectation(circ: PythonObject, observable: PythonObject) raises -> PythonObject:
    """Tr(rho H) for a Pauli-sum string."""
    return density_evolve(PyCircuit._get(circ)[].c).expectation(_pauli(observable))


def py_mitigate_m3(counts: PythonObject, p01: PythonObject, p10: PythonObject, hamming: PythonObject) raises -> PythonObject:
    """M3 readout mitigation: ({bitstring: count}, p01, p10, hamming) -> {bitstring: quasi-probability}."""
    var c = Dict[String, Int]()
    for key in counts:
        c[String(py=key)] = Int(py=counts[key])
    var e01 = _floats(p01)
    var e10 = _floats(p10)
    var cal = ReadoutCalibration(len(e01))
    cal.p01 = e01^
    cal.p10 = e10^
    var out = Python.dict()
    var m = mitigate_counts_m3(c, cal, Int(py=hamming))
    for kv in m.items():
        out[kv.key] = kv.value
    return out


def py_to_pbc(circ: PythonObject, fix_clifford: PythonObject) raises -> PythonObject:
    """Pauli-based form as JSON text: rotations and measurements in the input frame."""
    var w = JsonWriter()
    to_pbc(PyCircuit._get(circ)[].c, Bool(py=fix_clifford)).write_json(w)
    return PythonObject(w.finish())


def py_from_stim(text: PythonObject) raises -> PythonObject:
    """Stim circuit text -> Circuit."""
    return _wrap(from_stim(String(py=text)).circuit.copy())


def py_to_stim(circ: PythonObject) raises -> PythonObject:
    """Clifford + Pauli-noise Circuit -> Stim circuit text."""
    return PythonObject(to_stim(PyCircuit._get(circ)[].c))


def py_dem_text(circ: PythonObject) raises -> PythonObject:
    """Detector error model of a noisy Clifford circuit, in Stim DEM text."""
    return PythonObject(detector_error_model(PyCircuit._get(circ)[].c).to_text())


def py_parse_dem(text: PythonObject) raises -> PythonObject:
    """Round trip Stim DEM text through QUMO: (num_detectors, num_observables, num_errors, text)."""
    var d = parse_dem(String(py=text))
    return Python.tuple(d.num_detectors, d.num_observables, d.num_errors(), d.to_text())


def py_sample_dem(text: PythonObject, shots: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Detector firing rates of `shots` samples of a DEM: list per detector, then per observable."""
    var d = parse_dem(String(py=text))
    var n = Int(py=shots)
    var s = sample_dem(d, n, UInt64(Int(py=seed)))
    var out = Python.list()
    for k in range(d.num_detectors):
        var c = 0
        for shot in range(n):
            c += 1 if s.get(shot, k) else 0
        out.append(Float64(c) / Float64(n))
    for k in range(d.num_observables):
        var c = 0
        for shot in range(n):
            c += 1 if s.observable(shot, k) else 0
        out.append(Float64(c) / Float64(n))
    return out


def py_decode_dem(text: PythonObject, shots: PythonObject, decoder: PythonObject) raises -> PythonObject:
    """Decode with a DEM given as text: (dem_text, [[fired detector ids], ...], 'mwpm' or
    'mwpm_correlated') -> list of observable masks."""
    var bi = Python.import_module("builtins")
    if _is_str(shots) or Bool(py=bi.isinstance(shots, bi.int)) or Bool(py=bi.isinstance(shots, bi.float)):
        raise Error("QM-E911: decode_dem() argument 'shots' must be a list with one list of fired detector ids per shot, got " + _type_name(shots) + ". Hint: decode_dem(dem_text, [[0, 3], [], [1]], 'mwpm'); a number of shots belongs to sample_dem(dem_text, shots, seed).")
    if not _is_str(decoder):
        raise Error("QM-E911: decode_dem() argument 'decoder' must be a str, got " + _type_name(decoder) + ". Hint: use 'mwpm' or 'mwpm_correlated'.")
    var d = parse_dem(String(py=text))
    var name = canonical_decoder_name(String(py=decoder))
    if name == "sparse_blossom":
        name = "mwpm"
    if name != "mwpm" and name != "mwpm_correlated":
        raise Error("QM-E707: unknown decoder '" + String(py=decoder) + "'. Hint: decode_dem takes 'mwpm' or 'mwpm_correlated'.")
    var out = Python.list()
    if name == "mwpm_correlated":
        var dec = CorrelatedMatchingDecoder(d)
        for shot in shots:
            var f = List[Int]()
            for x in shot:
                f.append(Int(py=x))
            out.append(Int(dec.decode(f)))
        return out
    var dec = SparseBlossomDecoder(MatchingGraph(d))
    for shot in shots:
        var f = List[Int]()
        for x in shot:
            f.append(Int(py=x))
        out.append(Int(dec.decode(f)))
    return out


def py_missing_detectors(circ: PythonObject, unknown_input: PythonObject) raises -> PythonObject:
    """Deterministic record parities the circuit does not declare: list of record lists."""
    var out = Python.list()
    for recs in missing_detectors(PyCircuit._get(circ)[].c, Bool(py=unknown_input)):
        var row = Python.list()
        for r in recs:
            row.append(r)
        out.append(row)
    return out


def py_count_determined(circ: PythonObject, unknown_input: PythonObject) raises -> PythonObject:
    """Stim's count_determined_measurements."""
    return count_determined_measurements(PyCircuit._get(circ)[].c, Bool(py=unknown_input))


def py_translate(circ: PythonObject, basis: PythonObject) raises -> PythonObject:
    """Basis translation only (no layout/routing/optimisation)."""
    ref c = PyCircuit._get(circ)[].c
    var t = Target.all_to_all(max(c.num_qubits, 1), _basis_spec(basis))
    var r = compile_spec(c, t, "basis", 7)
    return _wrap(r.circuit.copy())


def py_compile_spec(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                    basis: PythonObject, spec: PythonObject) raises -> PythonObject:
    """Run an explicit text pipeline; returns (circuit, report_json)."""
    var t = _target(num_qubits, edges, basis)
    var r = compile_spec(PyCircuit._get(circ)[].c, t, String(py=spec), 7)
    return Python.tuple(_wrap(r.circuit.copy()), r.to_json())


def _layouts(r: CompileResult) raises -> Tuple[PythonObject, PythonObject]:
    var init = Python.list()
    for v in r.initial_layout:
        init.append(v)
    var fin = Python.list()
    for v in r.final_layout:
        fin.append(v)
    return (init, fin)


def py_transpile_ai(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                    basis: PythonObject, level: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Hybrid heuristic + AI compile (`transpile_ai`, AI level 1..3):
    (circuit, num_qubits, edges, basis, level, seed) -> (circuit, initial, final).

    Same conventions as `transpile`. Models are looked up in $QUMO_MODEL_DIR,
    then ./models/ai_transpiler; without models exact / greedy synthesis and
    SABRE routing still run. Every learned proposal is verified exactly.
    """
    var t = _target(num_qubits, edges, basis)
    var r = transpile_ai(PyCircuit._get(circ)[].c, t, Int(py=level), Int(py=seed))
    var lay = _layouts(r)
    return Python.tuple(_wrap(r.circuit.copy()), lay[0], lay[1])


def py_transpile_ai_report(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                           basis: PythonObject, level: PythonObject, seed: PythonObject) raises -> PythonObject:
    """`transpile_ai` with the per-pass report:
    -> (circuit, initial, final, report_json). At AI level 3 the report is the
    portfolio winner's (its pipeline names the preset that won)."""
    var t = _target(num_qubits, edges, basis)
    var r = transpile_ai(PyCircuit._get(circ)[].c, t, Int(py=level), Int(py=seed))
    var lay = _layouts(r)
    return Python.tuple(_wrap(r.circuit.copy()), lay[0], lay[1], r.to_json())


def py_compile_spec_full(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                         basis: PythonObject, spec: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Explicit text pipeline with a seed and layouts:
    -> (circuit, initial, final, report_json)."""
    var t = _target(num_qubits, edges, basis)
    var r = compile_spec(PyCircuit._get(circ)[].c, t, String(py=spec), Int(py=seed))
    var lay = _layouts(r)
    return Python.tuple(_wrap(r.circuit.copy()), lay[0], lay[1], r.to_json())


def py_sample(circ: PythonObject, shots: PythonObject, seed: PythonObject) raises -> PythonObject:
    var r = sample(PyCircuit._get(circ)[].c, _int_arg(shots, "sample", "shots"), UInt64(_int_arg(seed, "sample", "seed")))
    var d = Python.dict()
    for kv in r.counts.items():
        d[kv.key] = kv.value
    return d


def py_sample_device(circ: PythonObject, shots: PythonObject, seed: PythonObject, device: PythonObject,
                     var **kwargs: PythonObject) raises -> PythonObject:
    """Sample on an explicit device -> (counts, method, note).

    Python: sample_device(circ, shots, seed, device, precision="auto").

    device: 'auto' | 'cpu' | 'gpu'. 'gpu' falls back to the CPU when the build
    lacks -D QUMO_GPU or no device is usable; `note` then says why.
    precision (keyword only): 'auto' (QUMO_GPU_PRECISION when set, else the
    build default: fp32 on every GPU) | 'fp32' | 'fp64', for GPU state-vector
    runs; `method` names the precision that ran ('gpu:statevector:fp64').
    Errors: QM-E940 unknown device, QM-E941 unknown precision, QM-E472 fp64
    on an Apple GPU.
    """
    var precision = String("auto")
    var params = List[Float64]()
    var d = Python.dict()
    for kv in kwargs.items():
        if kv.key == "precision":
            precision = String(py=kv.value)
        elif kv.key == "params":
            params = _floats(kv.value)
        elif kv.key == "counts":
            d = kv.value
        else:
            raise Error("QM-E913: sample_device() got an unexpected keyword argument '" + kv.key + "'. Hint: the keyword arguments are precision='auto'|'fp32'|'fp64', params=[...] and counts=<dict to fill>.")
    var note = String()
    var r = sample_on(PyCircuit._get(circ)[].c, _int_arg(shots, "sample_device", "shots"),
                      UInt64(_int_arg(seed, "sample_device", "seed")), params, String(py=device), note, precision)
    for kv in r.counts.items():
        d[kv.key] = kv.value
    return Python.tuple(d, r.method, note)


def py_gpu_info() raises -> PythonObject:
    """GPU status: compiled (-D QUMO_GPU), available, name, api, vendor,
    precision (what 'auto' runs), precision_build_default, precision_env
    (QUMO_GPU_PRECISION), precisions (selectable), precision_note."""
    var g = query_gpu()
    var d = Python.dict()
    d["compiled"] = g.compiled
    d["available"] = g.available
    d["name"] = g.name
    d["api"] = g.api
    d["vendor"] = g.vendor
    d["precision"] = g.precision
    d["precision_build_default"] = g.precision_build_default
    d["precision_env"] = g.precision_env
    var ps = Python.list()
    for p in g.precisions:
        ps.append(p)
    d["precisions"] = ps
    d["precision_note"] = g.precision_note
    d["fp64_supported"] = g.fp64_supported
    d["min_qubits_auto"] = g.min_qubits_auto
    d["error"] = g.error
    return d


def py_expectation(circ: PythonObject, observable: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    var params = List[Float64]()
    for kv in kwargs.items():
        if kv.key == "params":
            params = _floats(kv.value)
        else:
            raise Error("QM-E913: expectation() got an unexpected keyword argument '" + kv.key + "'. Hint: the only keyword argument is params=[...].")
    var sv = evolve(PyCircuit._get(circ)[].c, params)
    return sv.expectation(_pauli(observable))


# ---------------------------------------------------------------------------
# Device-placed values and gradients, SQD subspace solver, HEFT schedules,
# streaming QEC (used by examples/heterogeneous/{sqd,cutting,realtime_qec,
# qaoa_escalation}). All keyword arguments are optional and keyword-only.
# ---------------------------------------------------------------------------
def _floats(values: PythonObject) raises -> List[Float64]:
    var out = List[Float64]()
    for v in values:
        out.append(Float64(py=v))
    return out^


def _float_list(values: List[Float64]) raises -> PythonObject:
    var out = Python.list()
    for v in values:
        out.append(v)
    return out


def py_expectation_device(circ: PythonObject, observable: PythonObject, device: PythonObject,
                          var **kwargs: PythonObject) raises -> PythonObject:
    """<observable> of a bound circuit on a device -> (value, method, note).

    Python: expectation_device(circ, observable, device, precision="auto").
    device 'auto' | 'cpu' | 'gpu' (GPU state vector, fp32 by default; falls
    back to the CPU with a note when the build or the machine has no GPU).
    """
    var prec = String("auto")
    for kv in kwargs.items():
        if kv.key == "precision":
            prec = String(py=kv.value)
        else:
            raise Error("QM-E913: expectation_device() got an unexpected keyword argument '" + kv.key + "'. Hint: the only keyword argument is precision='auto'|'fp32'|'fp64'.")
    var obs = List[PauliSum]()
    obs.append(_pauli(observable))
    var note = String()
    var method = String()
    var vals = expectations_on(PyCircuit._get(circ)[].c, obs, List[Float64](), String(py=device), note, method, prec)
    return Python.tuple(vals[0], method, note)


def py_expectations_device(circ: PythonObject, observables: PythonObject, device: PythonObject,
                           var **kwargs: PythonObject) raises -> PythonObject:
    """Many observables of one bound circuit -> (values, method, note): one
    simulation, all Pauli terms reduced in one batch on the GPU.

    Python: expectations_device(circ, [obs, ...], device, precision="auto").
    """
    var prec = String("auto")
    for kv in kwargs.items():
        if kv.key == "precision":
            prec = String(py=kv.value)
        else:
            raise Error("QM-E913: expectations_device() got an unexpected keyword argument '" + kv.key + "'. Hint: the only keyword argument is precision='auto'|'fp32'|'fp64'.")
    var obs = List[PauliSum]()
    for o in observables:
        obs.append(_pauli(o))
    var note = String()
    var method = String()
    var vals = expectations_on(PyCircuit._get(circ)[].c, obs, List[Float64](), String(py=device), note, method, prec)
    return Python.tuple(_float_list(vals), method, note)


def py_value_and_gradient(circ: PythonObject, observable: PythonObject, params: PythonObject,
                          var **kwargs: PythonObject) raises -> PythonObject:
    """Exact E(params) and its adjoint gradient -> (value, grad, method, note).

    Python: value_and_gradient(circ, observable, params, device="cpu", precision="auto").
    device 'cpu' (fp64 adjoint), 'gpu' (GPU adjoint through a doubled register,
    fp32 by default; CPU fallback with a note), 'auto' (GPU from the dispatch
    threshold). Shared and affine parameters (QAOA angles) are exact: the
    chain rule sums every occurrence. Errors: QM-E804, QM-E805, QM-E940, QM-E941.
    """
    var dev = String("cpu")
    var prec = String("auto")
    for kv in kwargs.items():
        if kv.key == "device":
            dev = String(py=kv.value)
        elif kv.key == "precision":
            prec = String(py=kv.value)
        else:
            raise Error("QM-E913: value_and_gradient() got an unexpected keyword argument '" + kv.key + "'. Hint: keyword arguments: device, precision.")
    var note = String()
    var r = value_and_gradient_on(PyCircuit._get(circ)[].c, _pauli(observable), _floats(params),
                                  dev, note, prec)
    return Python.tuple(r.value, _float_list(r.grad), r.method, note)


def py_batch_value_and_gradient(circ: PythonObject, observable: PythonObject, param_sets: PythonObject) raises -> PythonObject:
    """CPU adjoint values and gradients for many parameter sets -> (values, grads)
    (parallel over sets for <= 14 qubits; bit-identical to one-by-one calls)."""
    var sets = List[List[Float64]]()
    for ps in param_sets:
        sets.append(_floats(ps))
    var r = batch_value_and_gradient(PyCircuit._get(circ)[].c, _pauli(observable), sets,
                                     Grad.ADJOINT)
    var gs = Python.list()
    for g in r[1]:
        gs.append(_float_list(g))
    return Python.tuple(_float_list(r[0]), gs)


def py_subspace_ground_state(observable: PythonObject, basis_states: PythonObject, num_qubits: PythonObject,
                             var **kwargs: PythonObject) raises -> PythonObject:
    """Lowest eigenpair of a Pauli-sum Hamiltonian projected onto basis states
    (the SQD / SKQD classical stage) -> dict with energy, occupations (<n_q>),
    dimension, nonzeros, matvecs, restarts, residual, converged, seconds and
    probabilities (|c_k|^2 per basis state, same order).

    Python: subspace_ground_state(observable, states, n, tol=1e-9, krylov=40).
    """
    var tol = 1e-9
    var krylov = 40
    for kv in kwargs.items():
        if kv.key == "tol":
            tol = Float64(py=kv.value)
        elif kv.key == "krylov":
            krylov = Int(py=kv.value)
        else:
            raise Error("QM-E913: subspace_ground_state() got an unexpected keyword argument '" + kv.key + "'. Hint: keyword arguments: tol, krylov.")
    var states = List[Int]()
    for b in basis_states:
        states.append(Int(py=b))
    var t0 = perf_counter_ns()
    var r = subspace_ground_state(_pauli(observable), states, Int(py=num_qubits), tol, krylov)
    var secs = Float64(perf_counter_ns() - t0) / 1e9
    var d = Python.dict()
    d["energy"] = r.energy
    d["occupations"] = _float_list(r.occupations)
    var probs = List[Float64](capacity=len(r.re))
    for k in range(len(r.re)):
        probs.append(r.re[k] * r.re[k] + r.im[k] * r.im[k])
    d["probabilities"] = _float_list(probs)
    d["dimension"] = r.dimension
    d["nonzeros"] = r.nonzeros
    d["matvecs"] = r.matvecs
    d["restarts"] = r.restarts
    d["residual"] = r.residual
    d["converged"] = r.converged
    d["seconds"] = secs
    return d


def _dget(d: PythonObject, key: String, default: PythonObject) raises -> PythonObject:
    return d.get(key, default)


def py_schedule(jobs: PythonObject, devices: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """HEFT plan of quantum jobs over a device pool (qumo.exec.orchestrator),
    without running them -> dict(device_of, start, finish, cost, makespan,
    total_cost, notes).

    Python: schedule(jobs, devices, cost_weight=0.0)
    jobs: [{"name", "circuit" (Circuit), "kind": "sample"|"estimate", "shots",
            "observable" (estimate), "deps": [earlier job indices], "device": pin}]
    devices: [{"name", "kind": "cpu"|"gpu"|"qpu"|"stabilizer", "max_qubits",
               "queue_s", "shot_time_s", "cost_per_task", "cost_per_shot",
               "precision"}]
    Errors: QM-E460..QM-E464 (empty pool, nothing fits, bad deps, unknown or
    too small pinned device).
    """
    var cw = 0.0
    for kv in kwargs.items():
        if kv.key == "cost_weight":
            cw = Float64(py=kv.value)
        else:
            raise Error("QM-E913: schedule() got an unexpected keyword argument '" + kv.key + "'. Hint: keyword argument: cost_weight.")
    var pool = List[Device]()
    for d in devices:
        var name = String(py=d["name"])
        var kind = String(py=_dget(d, "kind", "cpu"))
        var mq = Int(py=_dget(d, "max_qubits", 30))
        if kind == "cpu":
            pool.append(Device.cpu(name, mq))
        elif kind == "gpu":
            pool.append(Device.gpu(name, mq, String(py=_dget(d, "precision", "auto"))))
        elif kind == "stabilizer":
            pool.append(Device.stabilizer(name))
        elif kind == "qpu":
            pool.append(Device.qpu(name, mq, Float64(py=_dget(d, "queue_s", 0.0)), Float64(py=_dget(d, "shot_time_s", 1e-4)),
                                   Float64(py=_dget(d, "cost_per_task", 0.0)), Float64(py=_dget(d, "cost_per_shot", 0.0))))
        else:
            raise Error("QM-E913: device '" + name + "' has unknown kind '" + kind + "'. Hint: use cpu, gpu, qpu or stabilizer.")
    var wf = Workflow()
    for j in jobs:
        var name = String(py=j["name"])
        ref c = PyCircuit._get(j["circuit"])[].c
        var kind = String(py=_dget(j, "kind", "sample"))
        var deps = List[Int]()
        for x in _dget(j, "deps", Python.list()):
            deps.append(Int(py=x))
        var pin = String(py=_dget(j, "device", ""))
        if kind == "estimate":
            _ = wf.add_estimate(name, c, parse_pauli_sum(String(py=_dget(j, "observable", "Z0"))), List[Float64](), deps,
                                1234, "auto", pin)
        else:
            _ = wf.add_sample(name, c, Int(py=_dget(j, "shots", 1000)), List[Float64](), deps, 1234, "auto", pin)
    var sch = wf.plan(pool, cw)
    var out = Python.dict()
    var dev = Python.list()
    for k in range(len(sch.device_of)):
        dev.append(pool[sch.device_of[k]].name)
    out["device_of"] = dev
    out["start"] = _float_list(sch.start)
    out["finish"] = _float_list(sch.finish)
    out["cost"] = _float_list(sch.cost)
    out["makespan"] = sch.makespan
    out["total_cost"] = sch.total_cost
    var notes = Python.list()
    for line in sch.notes:
        notes.append(line)
    out["notes"] = notes
    return out


comptime _HEXD = "0123456789abcdef"


def _hex_nibble(c: Int) raises -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    if c >= 65 and c <= 70:
        return c - 55
    raise Error("QM-E913: invalid hex digit in a packed detection-event string. Hint: pass the string returned by QecMemory.sample().")


struct PyQecMemory(Movable, Writable):
    """A QEC memory experiment for streaming: syndrome sampling (CPU or GPU
    Pauli frames) and batch decoding (sparse blossom) on packed strings.

    Packed format (hex, like Stim's b8): one record of ceil(D/8) bytes per
    shot, bit k of byte j = detector 8j + k; observables likewise."""

    var cc: CodeCircuit
    var dem: DetectorErrorModel
    var decoder: SparseBlossomDecoder
    var frames: FrameSimulator
    var num_detectors: Int
    var num_observables: Int

    def __init__(out self, var cc: CodeCircuit) raises:
        self.dem = detector_error_model(cc.circuit, True)
        self.decoder = SparseBlossomDecoder(MatchingGraph(self.dem))
        self.frames = FrameSimulator(cc.circuit)
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables
        self.cc = cc^

    @staticmethod
    def _get(py_self: PythonObject) raises -> Pointer[PyQecMemory, MutAnyOrigin]:
        return py_self.downcast_value_ptr[Self]().as_unsafe_any_origin()

    @staticmethod
    def info(py_self: PythonObject) raises -> PythonObject:
        ref m = Self._get(py_self)[]
        var d = Python.dict()
        d["code"] = m.cc.code
        d["basis"] = m.cc.basis
        d["distance"] = m.cc.distance
        d["rounds"] = m.cc.rounds
        d["num_qubits"] = m.cc.circuit.num_qubits
        d["num_detectors"] = m.num_detectors
        d["num_observables"] = m.num_observables
        d["dem_errors"] = m.dem.num_errors()
        d["p"] = m.cc.noise.after_clifford_depolarization
        return d

    @staticmethod
    def dem_text(py_self: PythonObject) raises -> PythonObject:
        # graph-like form: the same decomposition QUMO's matching decoders use,
        # so PyMatching built from this text matches them shot for shot
        return graphlike_dem(Self._get(py_self)[].dem).to_text()

    @staticmethod
    def stim_text(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].cc.to_stim()

    @staticmethod
    def sample(py_self: PythonObject, shots: PythonObject, seed: PythonObject, device: PythonObject,
               threads: PythonObject) raises -> PythonObject:
        """-> (dets_hex, obs_hex, method, note, seconds). device 'cpu' | 'gpu'
        ('gpu' = GPU Pauli frames; CPU fallback with a note); threads: CPU
        frame workers (0 = all cores)."""
        ref m = Self._get(py_self)[]
        var n = Int(py=shots)
        var dev = String(py=device)
        if dev != "cpu" and dev != "gpu" and dev != "auto":
            raise Error("QM-E940: unknown device '" + dev + "'. Hint: use 'auto', 'cpu' or 'gpu'.")
        var note = String()
        var method = String("cpu:pauli_frame")
        var t0 = perf_counter_ns()
        var samples: DetectorSamples
        var on_gpu = False
        if dev != "cpu":
            comptime if is_defined["QUMO_GPU"]():
                var ok: Bool
                try:
                    ok = gpu_available()
                except:
                    ok = False
                on_gpu = ok
                if not ok and dev == "gpu":
                    note = "no usable GPU device: ran on the CPU"
            else:
                if dev == "gpu":
                    note = "GPU support not compiled (build with -D QUMO_GPU): ran on the CPU"
        if on_gpu:
            var gs = GpuFrameSampler(m.cc.circuit)
            samples = gs.sample_detectors(n, UInt64(Int(py=seed)))
            method = gs.method_name()
        else:
            samples = m.frames.sample_detectors(n, UInt64(Int(py=seed)), Int(py=threads))
        var secs = Float64(perf_counter_ns() - t0) / 1e9
        var D = m.num_detectors
        var K = m.num_observables
        var bd = (D + 7) // 8
        var bk = (K + 7) // 8
        var digits = _HEXD.as_bytes()
        var hd = List[UInt8](capacity=2 * bd * n)
        var hk = List[UInt8](capacity=2 * bk * n)
        for s in range(n):
            for j in range(bd):
                var byte = 0
                for k in range(8):
                    var det = 8 * j + k
                    if det < D and samples.get(s, det):
                        byte |= 1 << k
                hd.append(digits[byte >> 4])
                hd.append(digits[byte & 15])
            for j in range(bk):
                var byte = 0
                for k in range(8):
                    var o = 8 * j + k
                    if o < K and samples.observable(s, o):
                        byte |= 1 << k
                hk.append(digits[byte >> 4])
                hk.append(digits[byte & 15])
        return Python.tuple(String(unsafe_from_utf8=Span(hd)), String(unsafe_from_utf8=Span(hk)), method, note, secs)

    @staticmethod
    def decode(py_self: PythonObject, dets_hex: PythonObject, shots: PythonObject, threads: PythonObject) raises -> PythonObject:
        """Sparse-blossom predictions for packed detection events ->
        (predictions_hex, seconds, detection_events). threads=1: this thread
        only (one decoder worker); 0: every core (decode_samples)."""
        ref m = Self._get(py_self)[]
        var text = String(py=dets_hex)
        var bs = text.as_bytes()
        var n = Int(py=shots)
        var D = m.num_detectors
        var K = m.num_observables
        var bd = (D + 7) // 8
        var bk = (K + 7) // 8
        if text.byte_length() != 2 * bd * n:
            raise Error("QM-E913: packed detection events have " + String(text.byte_length()) + " hex digits, expected " + String(2 * bd * n) + " for " + String(n) + " shots of " + String(D) + " detectors. Hint: decode with the QecMemory that sampled them.")
        var t0 = perf_counter_ns()
        var preds = List[UInt64](length=max(1, n), fill=0)
        var events = 0
        if Int(py=threads) == 1:
            var fired = List[Int]()
            for s in range(n):
                fired.clear()
                var base = 2 * bd * s
                for j in range(bd):
                    var byte = _hex_nibble(Int(bs[base + 2 * j])) * 16 + _hex_nibble(Int(bs[base + 2 * j + 1]))
                    if byte == 0:
                        continue
                    for k in range(8):
                        if (byte >> k) & 1 == 1:
                            fired.append(8 * j + k)
                events += len(fired)
                preds[s] = m.decoder.decode(fired)
        else:
            var smp = DetectorSamples(n, D, K)
            for s in range(n):
                var base = 2 * bd * s
                var b = s // BATCH
                var w = (s % BATCH) >> 6
                var bit = UInt64(1) << UInt64(s & 63)
                for j in range(bd):
                    var byte = _hex_nibble(Int(bs[base + 2 * j])) * 16 + _hex_nibble(Int(bs[base + 2 * j + 1]))
                    if byte == 0:
                        continue
                    for k in range(8):
                        if (byte >> k) & 1 == 1:
                            smp.dets[(b * D + 8 * j + k) * FW + w] |= bit
                            events += 1
            preds = decode_samples(m.decoder, smp)
        var secs = Float64(perf_counter_ns() - t0) / 1e9
        var digits = _HEXD.as_bytes()
        var hk = List[UInt8](capacity=2 * bk * n)
        for s in range(n):
            for j in range(bk):
                var byte = Int((preds[s] >> UInt64(8 * j)) & 255)
                hk.append(digits[byte >> 4])
                hk.append(digits[byte & 15])
        return Python.tuple(String(unsafe_from_utf8=Span(hk)), secs, events)

    def write_to(self, mut w: Some[Writer]):
        w.write("<qumo.QecMemory ", self.cc.code, " d=", self.cc.distance, " rounds=", self.cc.rounds, " detectors=",
                self.num_detectors, ">")

    def write_repr_to(self, mut w: Some[Writer]):
        self.write_to(w)


def py_qec_memory(code: PythonObject, distance: PythonObject, rounds: PythonObject, p: PythonObject) raises -> PythonObject:
    """QecMemory for a generated memory circuit ('surface_z', 'surface_x',
    'repetition') under uniform circuit noise p (Stim/Sinter convention)."""
    var cc = generate_code(String(py=code), Int(py=distance), Int(py=rounds), NoiseParams.uniform(Float64(py=p)))
    return PythonObject(alloc=PyQecMemory(cc^))


# ---------------------------------------------------------------------------
# Surface of the `qumo` Python package (python/qumo): bulk circuit building,
# NumPy state vectors, target specs, research drivers. Arguments are positional
# and all required here; the package gives them names and defaults.
# ---------------------------------------------------------------------------
def _dense_label(label: String) -> Bool:
    if label.byte_length() == 0:
        return False
    for b in label.as_bytes():
        if b != 73 and b != 88 and b != 89 and b != 90:  # I X Y Z
            return False
    return True


def _pauli(obs: PythonObject) raises -> PauliSum:
    """Observable argument: a Pauli-sum string ("0.5 * Z0 Z1 - X0", "ZZI + XIX"),
    or a list of (label, coeff) pairs ((coeff, label) is taken too). A label is
    dense ("XZI", leftmost = highest qubit) or sparse ("Z0 Z1")."""
    if _is_str(obs):
        return parse_pauli_sum(String(py=obs))
    var out = PauliSum()
    var ok = True
    var count = 0
    try:
        for term in obs:
            var first = term[0]
            var label: String
            var coeff: Float64
            if _is_str(first):
                label = String(py=first)
                coeff = Float64(py=term[1])
            else:
                label = String(py=term[1])
                coeff = Float64(py=first)
            if _dense_label(label):
                out.add_label(coeff, label)
            else:
                var ps = parse_pauli_sum(label)
                for t in ps.terms:
                    out.add(coeff * t.coeff, t.x, t.z)
            count += 1
    except e:
        var msg = String(e)
        if msg.startswith("QM-"):
            raise Error(msg)
        ok = False
    if not ok or count == 0:
        raise Error("QM-E911: an observable is a Pauli-sum string or a non-empty list of (label, coeff) pairs, got " + _type_name(obs) + ". Hint: e.g. \"Z0 Z1 + 0.5 * X0\" or [(\"ZZ\", 1.0), (\"XI\", 0.5)].")
    return out^


def _py_pauli(h: PauliSum, num_qubits: Int) raises -> PythonObject:
    """PauliSum -> [(label, coeff), ...] with dense labels (leftmost = highest qubit)."""
    var n = max(num_qubits, max(h.num_qubits(), 1))
    var out = Python.list()
    for t in h.terms:
        out.append(Python.tuple(t.label(n), t.coeff))
    return out


def _graph(num_nodes: PythonObject, edges: PythonObject) raises -> Graph:
    """edges: [(u, v)] or [(u, v, weight)]; num_nodes <= 0 = one more than the largest node."""
    var us = List[Int]()
    var vs = List[Int]()
    var ws = List[Float64]()
    var top = -1
    for e in edges:
        var u = Int(py=e[0])
        var v = Int(py=e[1])
        var w = 1.0
        if Int(py=e.__len__()) > 2:
            w = Float64(py=e[2])
        us.append(u)
        vs.append(v)
        ws.append(w)
        top = max(top, max(u, v))
    var n = Int(py=num_nodes)
    var g = Graph(n if n > 0 else top + 1)
    for i in range(len(us)):
        g.add_edge(us[i], vs[i], ws[i])
    return g^


def py_op_table() raises -> PythonObject:
    var d = Python.dict()
    for k in range(Op.NUM_OPS):
        var nm = op_name(k)
        if nm.byte_length() == 0 or op_from_name(nm) != k:
            continue
        d[nm] = Python.tuple(k, op_arity(k), op_num_params(k))
    return d


def _fill_amplitudes(mut sv: StateVector, addr: Int, squared: Bool):
    """Write the state into a NumPy buffer: interleaved (re, im) pairs, or |amplitude|^2.
    `sv` is borrowed for the whole call, so its buffers outlive the worker tasks."""
    var total = sv.size()
    var dst = Pointer[Float64, MutAnyOrigin](unsafe_from_address=addr)
    var rp = sv._rp()
    var ip = sv._ip()
    var nw = max(1, parallelism_level()) if total >= (1 << 16) else 1
    var chunk = (total + nw - 1) // nw

    def task(k: Int) {imm rp, imm ip, imm dst, imm chunk, imm total, imm squared}:
        var lo = k * chunk
        var hi = min(lo + chunk, total)
        if squared:
            for i in range(lo, hi):
                var re = rp[unsafe_offset=i]
                var im = ip[unsafe_offset=i]
                dst[unsafe_offset=i] = re * re + im * im
        else:
            for i in range(lo, hi):
                dst[unsafe_offset=2 * i] = rp[unsafe_offset=i]
                dst[unsafe_offset=2 * i + 1] = ip[unsafe_offset=i]

    if nw == 1:
        task(0)
    else:
        parallelize(task, nw)


def py_statevector(circ: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    var params = List[Float64]()
    for kv in kwargs.items():
        if kv.key == "params":
            params = _floats(kv.value)
        else:
            raise Error("QM-E913: statevector() got an unexpected keyword argument '" + kv.key + "'. Hint: the only keyword argument is params=[...].")
    var sv = evolve(PyCircuit._get(circ)[].c, params)
    var np = Python.import_module("numpy")
    var arr = np.empty(sv.size(), np.complex128)
    _fill_amplitudes(sv, Int(py=arr.ctypes.data), False)
    return arr


def py_probabilities(circ: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    var params = List[Float64]()
    for kv in kwargs.items():
        if kv.key == "params":
            params = _floats(kv.value)
        else:
            raise Error("QM-E913: probabilities() got an unexpected keyword argument '" + kv.key + "'. Hint: the only keyword argument is params=[...].")
    ref c = PyCircuit._get(circ)[].c
    var np = Python.import_module("numpy")
    if c.has_noise():
        var probs = density_evolve(c, params).probabilities()
        var arr = np.empty(len(probs), np.float64)
        var dst = Pointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=arr.ctypes.data))
        for i in range(len(probs)):
            dst[unsafe_offset=i] = probs[i]
        return arr
    var sv = evolve(c, params)
    var arr = np.empty(sv.size(), np.float64)
    _fill_amplitudes(sv, Int(py=arr.ctypes.data), True)
    return arr


def _target_dict(t: Target) raises -> PythonObject:
    var d = Python.dict()
    d["name"] = String(t.name)
    d["modality"] = String(t.modality)
    d["num_qubits"] = t.num_qubits
    d["all_to_all"] = t.full
    var edges = Python.list()
    var i = 0
    while i + 1 < len(t.edges):
        edges.append(Python.tuple(t.edges[i], t.edges[i + 1]))
        i += 2
    d["edges"] = edges
    var basis = Python.list()
    for k in t.basis:
        basis.append(op_name(k))
    d["basis"] = basis
    return d


def py_parse_target(spec: PythonObject) raises -> PythonObject:
    return _target_dict(parse_target(String(py=spec)))


def _compile_tuple(r: CompileResult) raises -> PythonObject:
    var lay = _layouts(r)
    return Python.tuple(_wrap(r.circuit.copy()), lay[0], lay[1], r.to_json())


def py_transpile_target(circ: PythonObject, spec: PythonObject, level: PythonObject, seed: PythonObject) raises -> PythonObject:
    var t = parse_target(String(py=spec))
    return _compile_tuple(transpile(PyCircuit._get(circ)[].c, t, _int_arg(level, "transpile_target", "level"),
                                    _int_arg(seed, "transpile_target", "seed")))


def py_transpile_report(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                        basis: PythonObject, level: PythonObject, seed: PythonObject) raises -> PythonObject:
    var t = _target(num_qubits, edges, basis)
    return _compile_tuple(transpile(PyCircuit._get(circ)[].c, t, _int_arg(level, "transpile_report", "level"),
                                    _int_arg(seed, "transpile_report", "seed")))


def _grad_method(name: String) raises -> Int:
    var k = name.lower()
    if k == "adjoint":
        return Grad.ADJOINT
    if k == "parameter-shift" or k == "parameter_shift" or k == "param_shift" or k == "shift":
        return Grad.PARAM_SHIFT
    if k == "finite-diff" or k == "finite_diff" or k == "fd":
        return Grad.FINITE_DIFF
    raise Error("QM-E913: unknown gradient method '" + name + "'. Hint: use 'adjoint', 'parameter-shift' or 'finite-diff'.")


def py_vqe(circ: PythonObject, observable: PythonObject, method: PythonObject, x0: PythonObject,
           maxiter: PythonObject, tol: PythonObject, grad: PythonObject, seed: PythonObject) raises -> PythonObject:
    var r = vqe(PyCircuit._get(circ)[].c, _pauli(observable), String(py=method), _floats(x0),
                _int_arg(maxiter, "vqe", "maxiter"), Float64(py=tol), _grad_method(String(py=grad)),
                UInt64(_int_arg(seed, "vqe", "seed")))
    var d = Python.dict()
    d["energy"] = r.energy
    d["params"] = _float_list(r.params)
    var names = Python.list()
    for nm in r.param_names:
        names.append(String(nm))
    d["param_names"] = names
    d["history"] = _float_list(r.history)
    d["nit"] = r.nit
    d["nfev"] = r.nfev
    d["ngrad"] = r.ngrad
    d["circuit_evals"] = r.circuit_evals
    d["wall_time_s"] = r.wall_time_s
    d["converged"] = r.converged
    d["message"] = String(r.message)
    d["optimizer"] = String(r.optimizer)
    d["grad_method"] = String(r.grad_method)
    return d


def py_qaoa_maxcut(num_nodes: PythonObject, edges: PythonObject, p: PythonObject, method: PythonObject,
                   restarts: PythonObject, shots: PythonObject, seed: PythonObject, maxiter: PythonObject) raises -> PythonObject:
    var g = _graph(num_nodes, edges)
    var r = qaoa_maxcut(g, _int_arg(p, "qaoa_maxcut", "p"), String(py=method), _int_arg(restarts, "qaoa_maxcut", "restarts"),
                        _int_arg(shots, "qaoa_maxcut", "shots"), UInt64(_int_arg(seed, "qaoa_maxcut", "seed")),
                        _int_arg(maxiter, "qaoa_maxcut", "maxiter"))
    var d = Python.dict()
    d["p"] = r.p
    d["energy"] = r.energy
    d["expected_cut"] = r.expected_cut
    d["max_cut"] = r.max_cut
    d["approx_ratio"] = r.approx_ratio
    d["gammas"] = _float_list(r.gammas)
    d["betas"] = _float_list(r.betas)
    d["best_bitstring"] = String(r.best_bitstring)
    d["best_cut"] = r.best_cut
    d["sampled_ratio"] = r.sampled_ratio
    d["prob_optimal"] = r.prob_optimal
    d["most_frequent"] = String(r.most_frequent)
    d["shots"] = r.shots
    d["history"] = _float_list(r.history)
    d["nfev"] = r.nfev
    d["circuit_evals"] = r.circuit_evals
    d["wall_time_s"] = r.wall_time_s
    d["optimizer"] = String(r.optimizer)
    return d


def py_h2_sto3g(num_qubits: PythonObject, include_nuclear_repulsion: PythonObject) raises -> PythonObject:
    var n = Int(py=num_qubits)
    return _py_pauli(h2_sto3g(n, Bool(py=include_nuclear_repulsion)), n)


def py_transverse_field_ising(n: PythonObject, j: PythonObject, h: PythonObject, periodic: PythonObject) raises -> PythonObject:
    return _py_pauli(transverse_field_ising(Int(py=n), Float64(py=j), Float64(py=h), Bool(py=periodic)), Int(py=n))


def py_heisenberg_xxz(n: PythonObject, j: PythonObject, delta: PythonObject, periodic: PythonObject,
                      field: PythonObject) raises -> PythonObject:
    return _py_pauli(heisenberg_xxz(Int(py=n), Float64(py=j), Float64(py=delta), Bool(py=periodic), Float64(py=field)),
                     Int(py=n))


def py_fermi_hubbard(num_sites: PythonObject, edges: PythonObject, t: PythonObject, u: PythonObject,
                     mu: PythonObject) raises -> PythonObject:
    var g = _graph(num_sites, edges)
    return _py_pauli(fermi_hubbard(g, Float64(py=t), Float64(py=u), Float64(py=mu)), 2 * len(g))


def py_maxcut_hamiltonian(num_nodes: PythonObject, edges: PythonObject, include_offset: PythonObject) raises -> PythonObject:
    var g = _graph(num_nodes, edges)
    return _py_pauli(maxcut(g, Bool(py=include_offset)), len(g))


def py_ground_energy(observable: PythonObject, num_qubits: PythonObject) raises -> PythonObject:
    return ground_energy(_pauli(observable), Int(py=num_qubits))


def py_ghz(n: PythonObject) raises -> PythonObject:
    return _wrap(ghz(Int(py=n)))


def py_qft(n: PythonObject, do_swaps: PythonObject) raises -> PythonObject:
    return _wrap(qft(Int(py=n), Bool(py=do_swaps)))


def py_hardware_efficient_ansatz(n: PythonObject, layers: PythonObject, entanglement: PythonObject,
                                 rotations: PythonObject) raises -> PythonObject:
    return _wrap(hardware_efficient_ansatz(Int(py=n), Int(py=layers), String(py=entanglement), String(py=rotations)))


def py_qaoa_maxcut_ansatz(num_nodes: PythonObject, edges: PythonObject, p: PythonObject) raises -> PythonObject:
    return _wrap(qaoa_maxcut_ansatz(_graph(num_nodes, edges), Int(py=p)))


def py_uccsd_ansatz(num_qubits: PythonObject, num_electrons: PythonObject) raises -> PythonObject:
    return _wrap(uccsd_ansatz(Int(py=num_qubits), Int(py=num_electrons)))


def py_h2_ansatz() raises -> PythonObject:
    return _wrap(h2_ansatz())


def py_trotter(observable: PythonObject, time: PythonObject, steps: PythonObject, order: PythonObject,
               num_qubits: PythonObject) raises -> PythonObject:
    return _wrap(trotter(_pauli(observable), Float64(py=time), Int(py=steps), Int(py=order), Int(py=num_qubits)))


def _experiment_dict(r: ExperimentResult) raises -> PythonObject:
    var wi = r.wilson_interval()
    var d = Python.dict()
    d["code"] = String(r.code)
    d["basis"] = String(r.basis)
    d["decoder"] = String(r.decoder)
    d["distance"] = r.distance
    d["rounds"] = r.rounds
    d["p"] = r.p
    d["shots"] = r.shots
    d["errors"] = r.errors
    d["logical_error_rate"] = r.logical_error_rate()
    d["std_error"] = r.std_error()
    d["wilson"] = Python.tuple(wi[0], wi[1])
    d["per_round_error_rate"] = r.per_round_error_rate()
    d["seconds"] = r.seconds
    d["setup_seconds"] = r.setup_seconds
    d["shots_per_second"] = r.shots_per_second()
    d["num_qubits"] = r.num_qubits
    d["num_detectors"] = r.num_detectors
    d["num_dem_errors"] = r.num_dem_errors
    d["num_edges"] = r.num_edges
    d["seed"] = Int(r.seed)
    return d


def py_memory_experiment(code: PythonObject, distance: PythonObject, rounds: PythonObject, p: PythonObject,
                         shots: PythonObject, decoder: PythonObject, seed: PythonObject,
                         max_errors: PythonObject) raises -> PythonObject:
    return _experiment_dict(memory_experiment(
        String(py=code), _int_arg(distance, "memory_experiment", "distance"), _int_arg(rounds, "memory_experiment", "rounds"),
        Float64(py=p), _int_arg(shots, "memory_experiment", "shots"), String(py=decoder),
        UInt64(_int_arg(seed, "memory_experiment", "seed")), _int_arg(max_errors, "memory_experiment", "max_errors")))


def py_threshold_sweep(code: PythonObject, distances: PythonObject, ps: PythonObject, shots: PythonObject,
                       decoder: PythonObject, seed: PythonObject, max_errors: PythonObject,
                       rounds: PythonObject) raises -> PythonObject:
    var rs = threshold_sweep_results(
        String(py=code), _int_list(distances), _floats(ps), _int_arg(shots, "threshold_sweep", "shots"), String(py=decoder),
        UInt64(_int_arg(seed, "threshold_sweep", "seed")), _int_arg(max_errors, "threshold_sweep", "max_errors"),
        _int_arg(rounds, "threshold_sweep", "rounds"))
    var shots_out = Python.list()
    var errors = Python.list()
    var rate = Python.list()
    var stderr = Python.list()
    var lo = Python.list()
    var hi = Python.list()
    var per_round = Python.list()
    var seconds = Python.list()
    var rounds_out = Python.list()
    for r in rs:
        var wi = r.wilson_interval()
        shots_out.append(r.shots)
        errors.append(r.errors)
        rate.append(r.logical_error_rate())
        stderr.append(r.std_error())
        lo.append(wi[0])
        hi.append(wi[1])
        per_round.append(r.per_round_error_rate())
        seconds.append(r.seconds)
        rounds_out.append(r.rounds)
    var d = Python.dict()
    d["code"] = code
    d["decoder"] = decoder
    d["distances"] = _py_int_list(_int_list(distances))
    d["ps"] = _float_list(_floats(ps))
    d["rounds"] = rounds_out
    d["shots"] = shots_out
    d["errors"] = errors
    d["logical_error_rate"] = rate
    d["std_error"] = stderr
    d["wilson_lo"] = lo
    d["wilson_hi"] = hi
    d["per_round_error_rate"] = per_round
    d["seconds"] = seconds
    return d


# Environment variables the Mojo runtime sets at the C level when its Python
# interop starts: it discovers *a* Python on the machine (e.g. a Homebrew
# 3.14) and exports PYTHONEXECUTABLE, PYTHONPATH (":"-prefixed) and
# MOJO_PYTHON_LIBRARY. Inside a host interpreter these are wrong, and every
# child process (subprocess, multiprocessing spawn) inherits them: a
# subprocess started with sys.executable then runs the other Python.


def _restore_host_environment():
    """Put those variables back to the host interpreter's values.

    `os.environ` is a snapshot taken when the interpreter started, so it holds
    the user's original values: restore each one, or unset it if the user had
    not set it. Errors are ignored: the module must still import."""
    try:
        var environ = Python.import_module("os").environ
        var keys: List[String] = ["PYTHONEXECUTABLE", "PYTHONPATH", "PYTHONHOME", "MOJO_PYTHON_LIBRARY"]
        for key in keys:
            if Bool(environ.__contains__(key)):
                _ = setenv(key, String(py=environ[key]), True)
            else:
                _ = unsetenv(key)
    except:
        pass


@export
def PyInit_qumo_py() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("qumo_py")
        _ = (
            m.add_type[PyCircuit]("Circuit")
            .def_method[PyCircuit.num_qubits]("num_qubits", docstring="num_qubits() -> int\n\nNumber of qubits.")
            .def_method[PyCircuit.num_clbits]("num_clbits", docstring="num_clbits() -> int\n\nNumber of classical bits.")
            .def_method[PyCircuit.size]("size", docstring="size() -> int\n\nNumber of unitary gates (no measurements, resets, barriers or noise).")
            .def_method[PyCircuit.num_parameters]("num_parameters", docstring="num_parameters() -> int\n\nNumber of unbound symbolic parameters.")
            .def_method[PyCircuit.depth]("depth", docstring="depth() -> int\n\nDepth over unitary gates.")
            .def_method[PyCircuit.count_ops]("count_ops", docstring="count_ops() -> dict\n\n{gate name: count} over every instruction.")
            .def_method[PyCircuit.count_gate]("count_gate", docstring="count_gate(name) -> int\n\nHow many instructions are called `name` (0 for an unknown name).")
            .def_method[PyCircuit.depth_of]("depth_of", docstring="depth_of(name) -> int\n\nDepth counting only the gates called `name`, e.g. depth_of('cz') for the two-qubit depth. `name` is one gate name or a list of names (depth_of(['cx', 'cz'])). Errors: QM-E911 for another type, QM-E913 for an unknown gate name.")
            .def_method[PyCircuit.two_qubit_edges]("two_qubit_edges", docstring="two_qubit_edges() -> list\n\n[(q0, q1), ...] for every two-qubit gate, in order (to check a coupling map).")
            .def_method[PyCircuit.copy]("copy", docstring="copy() -> Circuit\n\nAn independent copy.")
            .def_method[PyCircuit.compose]("compose", docstring="compose(other) -> Circuit\n\nA new circuit: this one followed by `other`.")
            .def_method[PyCircuit.bind]("bind", docstring="bind(values) -> Circuit\n\nA new circuit with every symbolic parameter replaced: `values` has one number per parameter, in parameter_names() order. Errors: QM-E101 wrong count.")
            .def_method[PyCircuit.text]("text", docstring="text() -> str\n\nOne instruction per line (QUMO's own listing).")
            .def_method[PyCircuit.to_qasm2]("to_qasm2", docstring="to_qasm2() -> str\n\nOpenQASM 2 text.")
            .def_method[PyCircuit.to_qasm3]("to_qasm3", docstring="to_qasm3() -> str\n\nCanonical OpenQASM 3 text; load_qasm reads it back to the same circuit.")
            .def_method[PyCircuit.name]("name", docstring="name() -> str\n\nThe circuit's name.")
            .def_method[PyCircuit.num_ops]("num_ops", docstring="num_ops() -> int\n\nNumber of instructions of every kind.")
            .def_method[PyCircuit.num_measurements]("num_measurements", docstring="num_measurements() -> int\n\nNumber of measurement instructions.")
            .def_method[PyCircuit.num_2q_gates]("num_2q_gates", docstring="num_2q_gates() -> int\n\nNumber of two-qubit gates.")
            .def_method[PyCircuit.t_count]("t_count", docstring="t_count() -> int\n\nNumber of T and Tdg gates.")
            .def_method[PyCircuit.parameter_names]("parameter_names", docstring="parameter_names() -> list\n\nNames of the symbolic parameters, in the order bind() takes their values.")
            .def_method[PyCircuit.has_noise]("has_noise", docstring="has_noise() -> bool\n\nTrue when the circuit holds a noise channel.")
            .def_method[PyCircuit.is_clifford]("is_clifford", docstring="is_clifford() -> bool\n\nTrue when every gate is Clifford (rotations by multiples of pi/2 count).")
            .def_method[PyCircuit.is_dynamic]("is_dynamic", docstring="is_dynamic() -> bool\n\nTrue with classical conditions, resets or gates after a measurement.")
            .def_method[PyCircuit.inverse]("inverse", docstring="inverse() -> Circuit\n\nThe inverse of a unitary circuit.")
            .def_method[PyCircuit.extend_packed]("extend_packed", docstring="extend_packed(ops_addr, num_ops, params_addr, num_params, new_parameters) -> int\n\nAppend many instructions in one call, read from two C buffers (the `qumo` package builds them; array('q') and array('d') give the addresses through buffer_info()). Returns the new instruction count.\n\nops: 5 int64 per instruction: kind, q0, q1, q2, aux. kind comes from op_table(); unused qubits are -1; aux is the classical bit of a measure. kind -1 conditions the previous instruction on classical bit q0 having value q1.\nparams: 3 float64 per gate angle, in order: offset, parameter index (-1 = a constant, the value is `offset`), coeff; the angle is coeff * parameter + offset.\nnew_parameters: names to declare before appending.\nErrors: QM-E001 qubit out of range, QM-E913 malformed buffers.")
        )
        _ = (
            m.add_type[PyProgram]("Program")
            .def_method[PyProgram.num_qubits]("num_qubits", docstring="num_qubits() -> int")
            .def_method[PyProgram.num_clbits]("num_clbits", docstring="num_clbits() -> int")
            .def_method[PyProgram.num_parameters]("num_parameters", docstring="num_parameters() -> int\n\nNumber of `input` declarations.")
            .def_method[PyProgram.has_runtime_regions]("has_runtime_regions", docstring="has_runtime_regions() -> bool\n\nFalse when the program is one plain circuit.")
            .def_method[PyProgram.to_circuit]("to_circuit", docstring="to_circuit() -> Circuit\n\nThe plain circuit of a program without run-time regions (QM-E585 otherwise).")
            .def_method[PyProgram.to_qasm3]("to_qasm3", docstring="to_qasm3() -> str\n\nOpenQASM 3 text.")
            .def_method[PyProgram.to_json]("to_json", docstring="to_json() -> str\n\nProgram JSON (qumo.program.v1).")
            .def_method[PyProgram.lint]("lint", docstring="lint() -> list\n\nWarnings of the parser and of the program: [(code, message), ...].")
            .def_method[PyProgram.text]("text", docstring="text() -> str\n\nQUMO's own listing.")
        )
        m.def_function[py_load_program]("load_program", docstring="load_program(text) -> Program\n\nParse OpenQASM 2 or 3 into a Program: everything a Circuit holds, plus while, break, continue and classical variables.")
        m.def_function[py_program_from_json]("program_from_json", docstring="program_from_json(text) -> Program\n\nInverse of Program.to_json(). Malformed input raises QM-E587.")
        m.def_function[py_draw_program]("draw_program", docstring="draw_program(program, width, style) -> str\n\nText drawing by blocks. style: 'ascii' | 'unicode'; width 0 = no folding.")
        m.def_function[py_sample_program]("sample_program", docstring="sample_program(program, shots, seed, params=[], max_iterations=65536, method='auto') -> dict\n\nSample a program -> {counts, method, iterations, outputs, warnings}. params: values of the `input` declarations, in order. method: 'auto' | 'statevector' | 'tableau' | 'branching' | 'gpu'. Errors: QM-E480 past max_iterations, QM-E483 unsupported method.")
        m.def_function[py_transpile_program]("transpile_program", docstring="transpile_program(program, num_qubits, edges, basis, level, seed) -> (program, initial_layout, final_layout, report)\n\nCompile a Program; same target arguments as transpile(). num_qubits: physical qubits; edges: [(a, b), ...] couplings ([] = all-to-all); basis: list of gate names, or one str: a basis-set name ('sc_cz', 'sc_cx', 'sc_ecr', 'sc_iswap', 'ion_ms', 'ion_zz', 'atom_cz', 'clifford_t', 'generic', 'any') or a comma list ('rz,sx,x,cz').")
        m.def_function[py_empty]("circuit", docstring="circuit(n) -> Circuit\n\nEmpty circuit on n qubits.")
        m.def_function[py_quantum_volume]("quantum_volume", docstring="quantum_volume(n, depth, seed) -> Circuit\n\nQuantum-volume model circuit.")
        m.def_function[py_efficient_su2]("efficient_su2", docstring="efficient_su2(n, reps, entanglement) -> Circuit\n\nRY/RZ layers with CX entanglers ('linear', 'circular', 'full'); parametric.")
        m.def_function[py_dtc_unitary]("dtc_unitary", docstring="dtc_unitary(n, g, seed) -> Circuit\n\nOne Floquet step of a discrete time crystal.")
        m.def_function[py_random_clifford]("random_clifford_circuit", docstring="random_clifford_circuit(n, num_gates, seed) -> Circuit")
        m.def_function[py_bv_all_ones]("bv_all_ones", docstring="bv_all_ones(n) -> Circuit\n\nBernstein-Vazirani with the all-ones secret.")
        m.def_function[py_bvlike]("trivial_bvlike_circuit", docstring="trivial_bvlike_circuit(n) -> Circuit")
        m.def_function[py_multi_control]("multi_control_circuit", docstring="multi_control_circuit(n) -> Circuit")
        m.def_function[py_twirl]("pauli_twirl_2q", docstring="pauli_twirl_2q(circuit, seed) -> Circuit\n\nPauli-twirl every two-qubit gate.")
        m.def_function[py_hamiltonian_circuit]("hamiltonian_circuit", docstring="hamiltonian_circuit(labels, coeffs, time) -> Circuit\n\nFirst-order Trotter step of sum(coeff * label) for `time`.")
        m.def_function[py_version]("version", docstring="version() -> str\n\nQUMO version.")
        m.def_function[py_transpile_calibrated]("transpile_calibrated", docstring="transpile_calibrated(circuit, num_qubits, edges, basis, level, seed, calibration) -> (circuit, initial_layout, final_layout, est_fidelity)\n\ntranspile() with device data. calibration: dict with optional error_1q, readout, t1_us, t2_us (one value per qubit), edge_errors ({'a-b': e}) and angle_bounds ({'rzz': (lo, hi)}). num_qubits: physical qubits; edges: [(a, b), ...] couplings ([] = all-to-all); basis: list of gate names, or one str: a basis-set name ('sc_cz', 'sc_cx', 'sc_ecr', 'sc_iswap', 'ion_ms', 'ion_zz', 'atom_cz', 'clifford_t', 'generic', 'any') or a comma list ('rz,sx,x,cz').")
        m.def_function[py_mcx_circuit]("mcx_circuit", docstring="mcx_circuit(num_controls, mode, num_ancillas, clean) -> Circuit\n\nMulti-controlled X on qubits 0..k-1 -> k, ancillas after.")
        m.def_function[py_draw]("draw", docstring="draw(circuit, width, style) -> str\n\nText drawing. style: 'ascii' | 'unicode'; width 0 = no folding.")
        m.def_function[py_sample_ptsbe]("sample_ptsbe", docstring="sample_ptsbe(circuit, shots, seed) -> dict\n\nCounts of a static circuit with Pauli noise: one simulation per distinct error pattern. Errors: QM-E409 for dynamic circuits, QM-E911 when shots or seed is not an int.")
        m.def_function[py_plan_trajectories]("plan_trajectories", docstring="plan_trajectories(circuit, shots, seed) -> (distinct_patterns, noise_sites, error_free_shots)\n\nWhat sample_ptsbe would run.")
        m.def_function[py_inject_noise]("inject_noise", docstring="inject_noise(qasm2_text, p1, p2, pm) -> (text, inserted_1q, inserted_2q, inserted_readout)\n\nAdd `#pragma qumo` channels to OpenQASM 2 text: depolarize1(p1) after each one-qubit gate, depolarize2(p2) after each two-qubit gate, x_error(pm) before each measurement. Any layout is read (several statements per line, statements over several lines); `measure q -> c;` gets one channel per qubit of the register. Errors: QM-E913 for what the pragmas cannot address (a register not named q, a statement without ';').")
        m.def_function[py_ionq_request]("ionq_request", docstring="ionq_request(qasm2_text, shots, noise, seed, name) -> (json_body, clbit_to_qubit, num_clbits, dense_qubits)\n\nIonQ ionq.circuit.v1 request for a compiled circuit.")
        m.def_function[py_ionq_counts]("ionq_counts", docstring="ionq_counts(keys, probs, clbit_qubit, shots, seed) -> dict\n\nIonQ probability histogram -> counts over the classical bits.")
        m.def_function[py_ionq_submit]("ionq_submit", docstring="ionq_submit(qasm2_text, shots, noise, seed, name) -> (job_id, clbit_to_qubit, num_clbits, dense_qubits)\n\nNative IonQ submission (key read from IONQ_API_KEY).")
        m.def_function[py_ionq_status]("ionq_status", docstring="ionq_status(job_id) -> str")
        m.def_function[py_ionq_result]("ionq_result", docstring="ionq_result(job_id, clbit_qubit, shots, seed) -> dict")
        m.def_function[py_ionq_backends]("ionq_backends", docstring="ionq_backends() -> list\n\n[(name, status, qubits), ...].")
        m.def_function[py_spq_payload_document]("speqtrum_payload_document", docstring="speqtrum_payload_document(backend, shots, qasm, num_qubits, num_clbits, initial, final, measure_clbits, measure_qubits, auto) -> str\n\nqumo.payload.v1 document.")
        m.def_function[py_spq_unsafe_construct]("speqtrum_unsafe_construct", docstring="speqtrum_unsafe_construct(text) -> str")
        m.def_function[py_spq_decode_result]("speqtrum_decode_result", docstring="speqtrum_decode_result(encoded, job_id) -> dict")
        m.def_function[py_spq_devices]("speqtrum_devices", docstring="speqtrum_devices(base, user, key, audience) -> list\n\n[(code, qubits, status, notes), ...].")
        m.def_function[py_spq_execute]("speqtrum_execute", docstring="speqtrum_execute(base, user, key, audience, device, document, shots) -> str\n\nSubmit; returns the job id.")
        m.def_function[py_spq_status]("speqtrum_status", docstring="speqtrum_status(base, user, key, audience, job_id) -> str")
        m.def_function[py_spq_result]("speqtrum_result", docstring="speqtrum_result(base, user, key, audience, job_id) -> dict")
        m.def_function[py_ibm_backends]("ibm_backends", docstring="ibm_backends(account, instance) -> list\n\n[(name, qubits, status, pending_jobs), ...].")
        m.def_function[py_ibm_target]("ibm_target", docstring="ibm_target(account, instance, backend, fractional) -> dict\n\nCoupling map, basis and calibration of an IBM backend.")
        m.def_function[py_ibm_isa_payload]("ibm_isa_payload", docstring="ibm_isa_payload(qasm, num_qubits, basis, edges, symmetric, d1, d2, dr) -> (qasm3, duration_s, two_qubit_gates, active_qubits)")
        m.def_function[py_ibm_submit]("ibm_submit", docstring="ibm_submit(account, instance, backend, qasm3, shots, cap) -> str\n\nSubmit; returns the job id.")
        m.def_function[py_ibm_status]("ibm_status", docstring="ibm_status(account, instance, job_id) -> (state, reason)")
        m.def_function[py_ibm_result]("ibm_result", docstring="ibm_result(account, instance, job_id, num_clbits) -> dict")
        m.def_function[py_qtm_native_circuit]("quantinuum_native_circuit", docstring="quantinuum_native_circuit(qasm2_text) -> (pytket_json, n_qubits, n_clbits, n1q, n2q, n_measure_reset, registers)")
        m.def_function[py_qtm_submit]("quantinuum_submit", docstring="quantinuum_submit(device, project, name, qasm, shots, max_cost) -> (job_id, project_id, circuit_id)")
        m.def_function[py_qtm_status]("quantinuum_status", docstring="quantinuum_status(job_id) -> (state, detail)")
        m.def_function[py_qtm_result]("quantinuum_result", docstring="quantinuum_result(job_id, num_clbits) -> dict")
        m.def_function[py_qtm_devices]("quantinuum_devices", docstring="quantinuum_devices() -> list\n\n[(name, qubits), ...].")
        m.def_function[py_sample_branching]("sample_branching", docstring="sample_branching(circuit, shots, seed) -> dict\n\nStabilizer branching sampler: near-Clifford circuits on up to 512 qubits. Errors: QM-E412 too many branches, QM-E911 when shots or seed is not an int.")
        m.def_function[py_density_probabilities]("density_probabilities", docstring="density_probabilities(circuit) -> list\n\nExact outcome probabilities of all qubits from the density matrix (noise included). Limit 15 qubits (QM-E410).")
        m.def_function[py_density_expectation]("density_expectation", docstring="density_expectation(circuit, observable) -> float\n\nTr(rho H) with the circuit's noise applied. observable: a Pauli-sum str ('0.5 * Z0 Z1 - X0' or 'ZZI + 0.5*XIX') or a list of (label, coeff) pairs; a dense label reads left to right from the highest qubit. Limit 15 qubits (QM-E410).")
        m.def_function[py_mitigate_m3]("mitigate_m3", docstring="mitigate_m3(counts, p01, p10, hamming) -> dict\n\nM3 readout mitigation: {bitstring: count} and per-qubit flip rates -> {bitstring: quasi-probability}.")
        m.def_function[py_to_pbc]("to_pbc", docstring="to_pbc(circuit, fix_clifford) -> str\n\nPauli-based computation form as JSON text.")
        m.def_function[py_from_stim]("from_stim", docstring="from_stim(text) -> Circuit\n\nStim circuit text -> Circuit.")
        m.def_function[py_to_stim]("to_stim", docstring="to_stim(circuit) -> str\n\nClifford + Pauli-noise circuit -> Stim circuit text.")
        m.def_function[py_dem_text]("dem_text", docstring="dem_text(circuit) -> str\n\nDetector error model of a noisy Clifford circuit, in Stim DEM text.")
        m.def_function[py_parse_dem]("parse_dem", docstring="parse_dem(text) -> (num_detectors, num_observables, num_errors, text)\n\nRound trip Stim DEM text through QUMO.")
        m.def_function[py_sample_dem]("sample_dem", docstring="sample_dem(dem_text, shots, seed) -> list\n\nFiring rate of each detector, then of each observable, over `shots` samples.")
        m.def_function[py_decode_dem]("decode_dem", docstring="decode_dem(dem_text, shots, decoder) -> list\n\nDecode detection events with a DEM given as text. shots: [[fired detector ids], ...], one list per shot (not a number of shots); decoder: 'mwpm' or 'mwpm_correlated'. Returns one observable mask per shot. Errors: QM-E911 wrong argument type, QM-E707 unknown decoder.")
        m.def_function[py_missing_detectors]("missing_detectors", docstring="missing_detectors(circuit, unknown_input) -> list\n\nDeterministic measurement parities the circuit does not declare as detectors: a list of record lists.")
        m.def_function[py_count_determined]("count_determined_measurements", docstring="count_determined_measurements(circuit, unknown_input) -> int\n\nStim's count_determined_measurements.")
        m.def_function[py_load_qasm]("load_qasm", docstring="load_qasm(text) -> Circuit\n\nParse OpenQASM 2 or 3 text (not a file path: use load_qasm_file). Errors: QM-E5xx with line and column.")
        m.def_function[py_load_qasm_file]("load_qasm_file", docstring="load_qasm_file(path) -> Circuit\n\nRead an OpenQASM 2 or 3 file.")
        m.def_function[py_transpile]("transpile", docstring="transpile(circuit, num_qubits, edges, basis, level, seed) -> (circuit, initial_layout, final_layout)\n\nCompile for a target. num_qubits: physical qubits; edges: [(a, b), ...] couplings ([] = all-to-all); basis: list of gate names, or one str: a basis-set name ('sc_cz', 'sc_cx', 'sc_ecr', 'sc_iswap', 'ion_ms', 'ion_zz', 'atom_cz', 'clifford_t', 'generic', 'any') or a comma list ('rz,sx,x,cz'). level: 0..3. initial_layout[l] / final_layout[l]: physical qubit of logical qubit l before / after routing.")
        m.def_function[py_translate]("translate", docstring="translate(circuit, basis) -> Circuit\n\nBasis translation only (no layout, routing or optimisation). basis: list of gate names or one basis-set name.")
        m.def_function[py_compile_spec]("compile_spec", docstring="compile_spec(circuit, num_qubits, edges, basis, spec) -> (circuit, report_json)\n\nRun an explicit text pipeline, e.g. 'basis,layout,route,optimize'. num_qubits: physical qubits; edges: [(a, b), ...] couplings ([] = all-to-all); basis: list of gate names, or one str: a basis-set name ('sc_cz', 'sc_cx', 'sc_ecr', 'sc_iswap', 'ion_ms', 'ion_zz', 'atom_cz', 'clifford_t', 'generic', 'any') or a comma list ('rz,sx,x,cz').")
        m.def_function[py_transpile_ai]("transpile_ai", docstring="transpile_ai(circuit, num_qubits, edges, basis, level, seed) -> (circuit, initial_layout, final_layout)\n\nHybrid heuristic + learned compile, AI level 1..3. Models come from $QUMO_MODEL_DIR, then ./models/ai_transpiler; every learned proposal is verified exactly.")
        m.def_function[py_compile_spec_full]("compile_spec_full", docstring="compile_spec_full(circuit, num_qubits, edges, basis, spec, seed) -> (circuit, initial_layout, final_layout, report_json)")
        m.def_function[py_transpile_ai_report]("transpile_ai_report", docstring="transpile_ai_report(circuit, num_qubits, edges, basis, level, seed) -> (circuit, initial_layout, final_layout, report_json)")
        m.def_function[py_sample]("sample", docstring="sample(circuit, shots, seed) -> dict\n\nCounts {bitstring: count} from the CPU state-vector simulator (static sampling, or per-shot trajectories for dynamic and noisy circuits). Bit c of the register is character -1-c of the key. Limit 32 qubits (QM-E404): sample_device(circuit, shots, seed, 'auto') picks the stabilizer or Pauli-frame simulator for Clifford circuits of any width.\nErrors: QM-E402 no measurements, QM-E101 unbound parameters, QM-E911 when shots or seed is not an int.\n\n    c = qumo_py.load_qasm('OPENQASM 3.0; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;')\n    qumo_py.sample(c, 1024, 1234)   # {'00': 520, '11': 504}")
        m.def_function[py_expectation]("expectation", docstring="expectation(circuit, observable, params=[]) -> float\n\n<psi|H|psi> from the state vector. Noise channels and measurements are skipped: use density_expectation for a noisy circuit. observable: a Pauli-sum str ('0.5 * Z0 Z1 - X0' or 'ZZI + 0.5*XIX') or a list of (label, coeff) pairs; a dense label reads left to right from the highest qubit. params: values of the symbolic parameters.")
        m.def_function[py_sample_device]("sample_device", docstring="sample_device(circuit, shots, seed, device, precision='auto', params=[], counts=None) -> (counts, method, note)\n\nSample on a device. device: 'auto' (state vector, stabilizer or Pauli frames, GPU when built with -D QUMO_GPU and worth it) | 'cpu' | 'gpu' (falls back to the CPU and says why in `note`). precision: 'auto' | 'fp32' | 'fp64' for GPU state vectors. params: values of the symbolic parameters. counts: a dict (or dict subclass) to fill and return.\nErrors: QM-E940 unknown device, QM-E941 unknown precision, QM-E472 fp64 on an Apple GPU, QM-E911 when shots or seed is not an int.")
        m.def_function[py_gpu_info]("gpu_info", docstring="gpu_info() -> dict\n\ncompiled, available, name, api, vendor, precision, precisions, precision_note, ...")
        _ = (
            m.add_type[PyQecMemory]("QecMemory")
            .def_method[PyQecMemory.info]("info", docstring="info() -> dict\n\ncode, basis, distance, rounds, num_qubits, num_detectors, num_observables, dem_errors, p.")
            .def_method[PyQecMemory.dem_text]("dem_text", docstring="dem_text() -> str\n\nGraph-like detector error model in Stim DEM text (what the matching decoders use).")
            .def_method[PyQecMemory.stim_text]("stim_text", docstring="stim_text() -> str\n\nThe memory circuit as Stim text.")
            .def_method[PyQecMemory.sample]("sample", docstring="sample(shots, seed, device, threads) -> (dets_hex, obs_hex, method, note, seconds)\n\nSample detection events with Pauli frames. device: 'cpu' | 'gpu' | 'auto'; threads: CPU workers (0 = all cores). Packed hex, like Stim's b8: ceil(D/8) bytes per shot, bit k of byte j = detector 8j + k; observables likewise.")
            .def_method[PyQecMemory.decode]("decode", docstring="decode(dets_hex, shots, threads) -> (predictions_hex, seconds, detection_events)\n\nSparse-blossom predictions for packed detection events from sample(). threads=1: this thread only; 0: every core.")
        )
        m.def_function[py_expectation_device]("expectation_device", docstring="expectation_device(circuit, observable, device, precision='auto') -> (value, method, note)\n\n<observable> of a bound circuit on 'auto' | 'cpu' | 'gpu'.")
        m.def_function[py_expectations_device]("expectations_device", docstring="expectations_device(circuit, observables, device, precision='auto') -> (values, method, note)\n\nMany observables from one simulation.")
        m.def_function[py_value_and_gradient]("value_and_gradient", docstring="value_and_gradient(circuit, observable, params, device='cpu', precision='auto') -> (value, grad, method, note)\n\nExact E(params) and its adjoint gradient. Shared and affine parameters are exact.")
        m.def_function[py_batch_value_and_gradient]("batch_value_and_gradient", docstring="batch_value_and_gradient(circuit, observable, param_sets) -> (values, grads)\n\nAdjoint values and gradients for many parameter sets in one call (parallel over sets up to 14 qubits).")
        m.def_function[py_subspace_ground_state]("subspace_ground_state", docstring="subspace_ground_state(observable, basis_states, num_qubits, tol=1e-9, krylov=40) -> dict\n\nLowest eigenpair of a Pauli-sum Hamiltonian projected onto basis states (the SQD classical stage).")
        m.def_function[py_schedule]("schedule", docstring="schedule(jobs, devices, cost_weight=0.0) -> dict\n\nHEFT plan of quantum jobs over a device pool, without running them.")
        m.def_function[py_qec_memory]("qec_memory", docstring="qec_memory(code, distance, rounds, p) -> QecMemory\n\nMemory experiment for streaming: 'surface_z' | 'surface_x' | 'repetition' under uniform circuit noise p.")
        m.def_function[py_op_table]("op_table", docstring="op_table() -> dict\n\n{gate name: (kind, arity, num_params)} for every instruction; `kind` is what Circuit.extend_packed takes.")
        m.def_function[py_statevector]("statevector", docstring="statevector(circuit, params=[]) -> numpy.ndarray\n\nFinal state as a complex128 array of 2^n amplitudes; amplitude i has qubit q in bit q of i. Measurements, resets and noise are skipped. The simulator's state is written once into the array (no Python lists). Needs NumPy. Limit 32 qubits (QM-E404).")
        m.def_function[py_probabilities]("probabilities", docstring="probabilities(circuit, params=[]) -> numpy.ndarray\n\nfloat64 array of 2^n outcome probabilities. A circuit with noise goes through the density matrix (limit 15 qubits).")
        m.def_function[py_parse_target]("parse_target", docstring="parse_target(spec) -> dict\n\nA one-line target, as the CLI takes it: '<topology>:<size>[/<basis>]', e.g. 'grid:2x2/sc_cz', 'line:20', 'heavy_hex:5/rz,sx,x,ecr', 'trapped_ion:11'. Returns name, modality, num_qubits, all_to_all, edges, basis. Errors: QM-E611.")
        m.def_function[py_transpile_target]("transpile_target", docstring="transpile_target(circuit, spec, level, seed) -> (circuit, initial_layout, final_layout, report_json)\n\nCompile for a one-line target spec (see parse_target). level 0..3.")
        m.def_function[py_transpile_report]("transpile_report", docstring="transpile_report(circuit, num_qubits, edges, basis, level, seed) -> (circuit, initial_layout, final_layout, report_json)\n\ntranspile() plus the per-pass report as JSON. num_qubits: physical qubits; edges: [(a, b), ...] couplings ([] = all-to-all); basis: list of gate names, or one str: a basis-set name ('sc_cz', 'sc_cx', 'sc_ecr', 'sc_iswap', 'ion_ms', 'ion_zz', 'atom_cz', 'clifford_t', 'generic', 'any') or a comma list ('rz,sx,x,cz').")
        m.def_function[py_vqe]("vqe", docstring="vqe(circuit, observable, method, x0, maxiter, tol, grad, seed) -> dict\n\nVariational quantum eigensolver; the optimiser loop runs natively. method: 'bfgs' | 'adam' | 'gradient-descent' | 'spsa' | 'nelder-mead'; x0: start point ([] = seeded default); grad: 'adjoint' | 'parameter-shift' | 'finite-diff'. Returns energy, params, param_names, history, nit, nfev, ngrad, circuit_evals, wall_time_s, converged, message, optimizer, grad_method.")
        m.def_function[py_qaoa_maxcut]("qaoa_maxcut", docstring="qaoa_maxcut(num_nodes, edges, p, method, restarts, shots, seed, maxiter) -> dict\n\nDepth-p QAOA for MaxCut; edges: [(u, v)] or [(u, v, weight)], num_nodes 0 = from the edges. Returns expected_cut, max_cut, approx_ratio, gammas, betas, best_bitstring, best_cut, prob_optimal, history, ...")
        m.def_function[py_h2_sto3g]("h2_sto3g", docstring="h2_sto3g(num_qubits, include_nuclear_repulsion) -> list\n\nH2 in STO-3G at 0.735 A as [(label, coeff), ...]; num_qubits 4 (Jordan-Wigner) or 2 (tapered).")
        m.def_function[py_transverse_field_ising]("transverse_field_ising", docstring="transverse_field_ising(n, j, h, periodic) -> list\n\n-j sum Z Z - h sum X on a chain, as [(label, coeff), ...].")
        m.def_function[py_heisenberg_xxz]("heisenberg_xxz", docstring="heisenberg_xxz(n, j, delta, periodic, field) -> list\n\nXXZ chain as [(label, coeff), ...].")
        m.def_function[py_fermi_hubbard]("fermi_hubbard", docstring="fermi_hubbard(num_sites, edges, t, u, mu) -> list\n\nFermi-Hubbard model on a graph (Jordan-Wigner, 2 qubits per site) as [(label, coeff), ...].")
        m.def_function[py_maxcut_hamiltonian]("maxcut_hamiltonian", docstring="maxcut_hamiltonian(num_nodes, edges, include_offset) -> list\n\nMaxCut cost Hamiltonian (<H> = -<cut>) as [(label, coeff), ...].")
        m.def_function[py_ground_energy]("ground_energy", docstring="ground_energy(observable, num_qubits) -> float\n\nExact ground-state energy by Lanczos; num_qubits <= 0 = from the observable.")
        m.def_function[py_ghz]("ghz", docstring="ghz(n) -> Circuit\n\nGHZ state preparation (no measurements).")
        m.def_function[py_qft]("qft", docstring="qft(n, do_swaps) -> Circuit\n\nQuantum Fourier transform.")
        m.def_function[py_hardware_efficient_ansatz]("hardware_efficient_ansatz", docstring="hardware_efficient_ansatz(n, layers, entanglement, rotations) -> Circuit\n\nParametric ansatz: rotation layers ('ry,rz') and entanglers ('linear', 'circular', 'full').")
        m.def_function[py_qaoa_maxcut_ansatz]("qaoa_maxcut_ansatz", docstring="qaoa_maxcut_ansatz(num_nodes, edges, p) -> Circuit\n\nQAOA ansatz for MaxCut; parameters gamma[0..p-1], beta[0..p-1].")
        m.def_function[py_uccsd_ansatz]("uccsd_ansatz", docstring="uccsd_ansatz(num_qubits, num_electrons) -> Circuit\n\nUCCSD ansatz (Jordan-Wigner).")
        m.def_function[py_h2_ansatz]("h2_ansatz", docstring="h2_ansatz() -> Circuit\n\nOne-parameter ansatz for 4-qubit H2.")
        m.def_function[py_trotter]("trotter", docstring="trotter(observable, time, steps, order, num_qubits) -> Circuit\n\nTrotter-Suzuki circuit of exp(-i H time); order 1, 2 or 4; num_qubits <= 0 = from the observable.")
        m.def_function[py_memory_experiment]("memory_experiment", docstring="memory_experiment(code, distance, rounds, p, shots, decoder, seed, max_errors) -> dict\n\nQEC memory experiment: sampling and decoding fused natively over all cores, no per-shot data in Python. code: 'surface_z' | 'surface_x' | 'repetition'; rounds < 1 = distance; decoder: 'mwpm' | 'mwpm_correlated' | 'mwpm_dense' | 'uf' | 'lookup'; max_errors > 0 stops early. Returns shots, errors, logical_error_rate, std_error, wilson, per_round_error_rate, seconds, ...")
        m.def_function[py_threshold_sweep]("threshold_sweep", docstring="threshold_sweep(code, distances, ps, shots, decoder, seed, max_errors, rounds) -> dict\n\nGrid of memory experiments. Lists of len(distances) * len(ps) values, distance-major: shots, errors, logical_error_rate, std_error, wilson_lo, wilson_hi, per_round_error_rate, seconds, rounds.")
        var mod = m.finalize()
        _restore_host_environment()
        return mod
    except e:
        abort(String("error creating qumo_py module: ", e))
