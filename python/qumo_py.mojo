"""Python bindings for QUMO (CPython extension built from Mojo).

    mojo build -I . --emit shared-lib -o build/qumo_py.so python/qumo_py.mojo

Exposes an opaque `Circuit` handle plus free functions. Kept deliberately
small and stable: build/load circuits, compile them for any target
(coupling map + basis as plain Python data, vendor-neutral), inspect
statistics, simulate. Used by the Benchpress `qumo_gym` and by
`python/qumo/__init__.py`.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary, op_from_name
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.runner import sample, evolve
from qumo.ir.qasm import parse_qasm, load_qasm, to_qasm
from qumo.ir.qasm2 import to_qasm2
from qumo.compiler.target import Target
from qumo.exec.dispatch import sample_on
from qumo.gpu.device import query_gpu
from qumo.compiler.pass_manager import transpile, compile_spec, transpile_ai, CompileResult
from qumo.bench.circuits import (
    quantum_volume, efficient_su2, dtc_unitary, random_clifford_circuit,
    bv_all_ones, trivial_bvlike_circuit, multi_control_circuit, pauli_twirl_2q,
    hamiltonian_circuit_from_labels,
)


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
        var k = op_from_name(String(py=name))
        var level = List[Int](length=c.num_qubits, fill=0)
        var d = 0
        for g in c.ops:
            if not op_is_unitary(g.kind):
                continue
            var a = op_arity(g.kind)
            var l = 0
            for j in range(a):
                l = max(l, level[g.qubit(j)])
            if g.kind == k:
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

    def write_to(self, mut w: Some[Writer]):
        w.write("<qumo.Circuit qubits=", self.c.num_qubits, " ops=", len(self.c.ops), ">")

    def write_repr_to(self, mut w: Some[Writer]):
        self.write_to(w)


def _wrap(var c: Circuit) raises -> PythonObject:
    return PythonObject(alloc=PyCircuit(c^))


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
    var r = sample(PyCircuit._get(circ)[].c, Int(py=shots), UInt64(Int(py=seed)))
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
    for kv in kwargs.items():
        if kv.key == "precision":
            precision = String(py=kv.value)
        else:
            raise Error("QM-E913: sample_device() got an unexpected keyword argument '" + kv.key + "'. Hint: the only keyword argument is precision='auto'|'fp32'|'fp64'.")
    var note = String()
    var r = sample_on(PyCircuit._get(circ)[].c, Int(py=shots), UInt64(Int(py=seed)), List[Float64](), String(py=device), note,
                      precision)
    var d = Python.dict()
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


def py_expectation(circ: PythonObject, observable: PythonObject) raises -> PythonObject:
    var sv = evolve(PyCircuit._get(circ)[].c)
    return sv.expectation(parse_pauli_sum(String(py=observable)))


@export
def PyInit_qumo_py() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("qumo_py")
        _ = (
            m.add_type[PyCircuit]("Circuit")
            .def_method[PyCircuit.num_qubits]("num_qubits")
            .def_method[PyCircuit.num_clbits]("num_clbits")
            .def_method[PyCircuit.size]("size")
            .def_method[PyCircuit.num_parameters]("num_parameters")
            .def_method[PyCircuit.depth]("depth")
            .def_method[PyCircuit.count_ops]("count_ops")
            .def_method[PyCircuit.count_gate]("count_gate")
            .def_method[PyCircuit.depth_of]("depth_of")
            .def_method[PyCircuit.two_qubit_edges]("two_qubit_edges")
            .def_method[PyCircuit.copy]("copy")
            .def_method[PyCircuit.compose]("compose")
            .def_method[PyCircuit.bind]("bind")
            .def_method[PyCircuit.text]("text")
            .def_method[PyCircuit.to_qasm2]("to_qasm2")
            .def_method[PyCircuit.to_qasm3]("to_qasm3")
        )
        m.def_function[py_empty]("circuit", docstring="Empty circuit with n qubits")
        m.def_function[py_quantum_volume]("quantum_volume")
        m.def_function[py_efficient_su2]("efficient_su2")
        m.def_function[py_dtc_unitary]("dtc_unitary")
        m.def_function[py_random_clifford]("random_clifford_circuit")
        m.def_function[py_bv_all_ones]("bv_all_ones")
        m.def_function[py_bvlike]("trivial_bvlike_circuit")
        m.def_function[py_multi_control]("multi_control_circuit")
        m.def_function[py_twirl]("pauli_twirl_2q")
        m.def_function[py_hamiltonian_circuit]("hamiltonian_circuit")
        m.def_function[py_version]("version")
        m.def_function[py_load_qasm]("load_qasm")
        m.def_function[py_load_qasm_file]("load_qasm_file")
        m.def_function[py_transpile]("transpile")
        m.def_function[py_translate]("translate")
        m.def_function[py_compile_spec]("compile_spec")
        m.def_function[py_transpile_ai]("transpile_ai")
        m.def_function[py_compile_spec_full]("compile_spec_full")
        m.def_function[py_transpile_ai_report]("transpile_ai_report")
        m.def_function[py_sample]("sample")
        m.def_function[py_expectation]("expectation")
        m.def_function[py_sample_device]("sample_device")
        m.def_function[py_gpu_info]("gpu_info")
        return m.finalize()
    except e:
        abort(String("error creating qumo_py module: ", e))
