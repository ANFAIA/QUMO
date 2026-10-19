"""Expectation values and adjoint gradients on an explicit device (CPU or GPU).

`value_and_gradient_on(circuit, h, params, device, note)` returns E(theta) and
dE/dtheta on the CPU (`ExpectationEngine`, fp64 adjoint) or on the GPU state
vector (`qumo.gpu`, fp32 by default), with the same fallback rules as
`qumo.exec.dispatch.sample_on`: a GPU request on a build without
`-D QUMO_GPU`, or without a usable device, runs on the CPU and `note` says why.
`expectation_on` / `expectations_on` do the same for values only
(`expectations_on` evolves once and evaluates many observables, e.g. every
measurement setting of a circuit-cutting fragment).

GPU adjoint (`gpu_value_and_gradient`): the reverse sweep of adjoint
differentiation needs <lam|G|psi> for two states, while the GPU engine reduces
<phi|P|phi> of one state. Both states therefore live in one register with one
extra qubit A (the most significant):

    Phi = (|0>_A |psi> + |1>_A |mu>) / sqrt(2),   mu = lam / ||lam||,  lam = H psi

Every un-applied gate acts on the system qubits only, so it updates psi and mu
together, and  <Phi| Y_A (x) P |Phi> = -Im <mu|P|psi>,  hence for
U = exp(-i phi G), G = sum_j c_j P_j:

    dE/dphi = 2 Im <lam|G|psi> = -2 ||lam|| sum_j c_j <Phi| Y_A (x) P_j |Phi>.

The reductions are the GPU engine's own Pauli-sum kernels; no new kernel is
needed. Generators measured in one batch: a pending generator P_k may be read
at any later point of the sweep as long as every gate un-applied in between
commutes with P_k (it does with its own gate), so consecutive commuting gates
(a QAOA cost layer, a layer of single-qubit rotations) share one batch of
reductions and one host synchronisation. Cost: one forward pass, one state
download (n qubits) and upload (n + 1 qubits), the fused inverse gates, and
one reduction batch per non-commuting boundary (2 per QAOA layer).
Memory: 2^(n+1) amplitudes on the device. Precision: the GPU state's
(fp32 on every GPU by default); H psi and the energy are computed in fp64 on
the host from the downloaded state.

Differentiable gates (as for the CPU adjoint): RX, RY, RZ, P, RXX, RYY, RZZ,
CP, CRX, CRY, CRZ and U (split as RZ RY RZ; global phases cancel). Errors:
QM-E805 (not differentiable), QM-E804 (parameter count), QM-E940 (device),
QM-E941 / QM-E472 (precision).
"""

from std.math import sqrt
from std.bit import pop_count
from std.sys.defines import is_defined

from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_name, op_num_params, op_is_unitary, op_is_diagonal, op_arity, op_inverse_kind
from qumo.ir.param import Param
from qumo.ir.pauli import PauliSum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve
from qumo.algorithms.hamiltonians import PauliOperator, check_fits
from qumo.algorithms.gradients import ExpectationEngine, Grad, check_differentiable
from qumo.gpu.device import (
    GPU_COMPILED, GPU_FP64_RUNTIME, GPU_MIN_QUBITS, gpu_available, parse_gpu_precision, resolve_gpu_precision,
    check_gpu_precision, gpu_fp64_error,
)
from qumo.gpu.statevector_gpu import GpuStateVector
from qumo.gpu.device import device_kernels_enabled
from std.collections import Optional
from max.gpu.host import DeviceContext


struct DeviceGradient(Copyable, Writable):
    """E(theta), dE/dtheta and where they were computed."""

    var value: Float64
    var grad: List[Float64]
    var method: String  # "cpu:adjoint:fp64" | "gpu:adjoint:fp32" | "gpu-emulator:adjoint:fp32" | ...
    var reductions: Int  # GPU: reduction batches with a host synchronisation (0 on the CPU)

    def __init__(out self):
        self.value = 0.0
        self.grad = List[Float64]()
        self.method = ""
        self.reductions = 0

    def write_to(self, mut w: Some[Writer]):
        w.write("DeviceGradient(value=", self.value, ", n_params=", len(self.grad), ", method=", self.method, ")")


# ---------------------------------------------------------------------------
# Pauli generators of the differentiable gates (U = exp(-i phi G))
# ---------------------------------------------------------------------------
struct _Gen(Copyable, Sized):
    var xs: List[Int]
    var zs: List[Int]
    var cs: List[Float64]

    def __init__(out self):
        self.xs = List[Int]()
        self.zs = List[Int]()
        self.cs = List[Float64]()

    def add(mut self, x: Int, z: Int, c: Float64):
        self.xs.append(x)
        self.zs.append(z)
        self.cs.append(c)

    def __len__(self) -> Int:
        return len(self.cs)


def _generator(kind: Int, q0: Int, q1: Int) -> _Gen:
    """G with U = exp(-i phi G) as a Pauli sum (identity parts dropped: they
    contribute Im<lam|psi> = Im E = 0). Empty for gates without a rotation
    angle. P = e^{i phi/2} RZ(phi) and CP / CRx differ from their Pauli forms
    by global phases or identity terms only."""
    var g = _Gen()
    var a = 1 << q0
    var b = (1 << q1) if q1 >= 0 else 0
    if kind == Op.RX:
        g.add(a, 0, 0.5)
    elif kind == Op.RY:
        g.add(a, a, 0.5)
    elif kind == Op.RZ or kind == Op.P:
        g.add(0, a, 0.5)
    elif kind == Op.RXX:
        g.add(a | b, 0, 0.5)
    elif kind == Op.RYY:
        g.add(a | b, a | b, 0.5)
    elif kind == Op.RZZ:
        g.add(0, a | b, 0.5)
    elif kind == Op.CP:  # -|11><11| = -(I - Z_a - Z_b + Z_a Z_b) / 4
        g.add(0, a, 0.25)
        g.add(0, b, 0.25)
        g.add(0, a | b, -0.25)
    elif kind == Op.CRZ:  # |1><1|_c (x) Z_t / 2
        g.add(0, b, 0.25)
        g.add(0, a | b, -0.25)
    elif kind == Op.CRX:
        g.add(b, 0, 0.25)
        g.add(b, a, -0.25)
    elif kind == Op.CRY:
        g.add(b, b, 0.25)
        g.add(b, a | b, -0.25)
    return g^


struct _Step(Copyable):
    """One gate of the sweep: bound forward instruction + parameter reference."""

    var inst: Instruction
    var pidx: Int  # -1: constant angle (or no angle)
    var pcoeff: Float64
    var gen: _Gen
    var support: Int

    def __init__(out self, inst: Instruction, pidx: Int, pcoeff: Float64):
        self.inst = inst
        self.pidx = pidx
        self.pcoeff = pcoeff
        self.gen = _generator(inst.kind, inst.q0, inst.q1 if op_arity(inst.kind) >= 2 else -1)
        var s = 0
        for j in range(op_arity(inst.kind)):
            s |= 1 << inst.qubit(j)
        self.support = s


def _bound(kind: Int, q0: Int, q1: Int, q2: Int, angle: Float64) -> Instruction:
    return Instruction.make(kind, q0, q1, q2, angle)


def _sweep_steps(circuit: Circuit, params: List[Float64]) raises -> List[_Step]:
    """Unitary gates in circuit order, angles bound; symbolic U split into RZ RY RZ."""
    var out = List[_Step]()
    for i in range(len(circuit.ops)):
        var g = circuit.ops[i]
        if not op_is_unitary(g.kind) or g.kind == Op.I:
            continue
        if g.kind == Op.U and g.is_parametric():
            # U(theta, phi, lam) = e^{i(phi+lam)/2} RZ(phi) RY(theta) RZ(lam): RZ(lam) acts first
            out.append(_Step(_bound(Op.RZ, g.q0, -1, -1, g.p2.resolve(params)),
                             g.p2.index if g.p2.is_symbolic() else -1, g.p2.coeff))
            out.append(_Step(_bound(Op.RY, g.q0, -1, -1, g.p0.resolve(params)),
                             g.p0.index if g.p0.is_symbolic() else -1, g.p0.coeff))
            out.append(_Step(_bound(Op.RZ, g.q0, -1, -1, g.p1.resolve(params)),
                             g.p1.index if g.p1.is_symbolic() else -1, g.p1.coeff))
            continue
        var b = g
        var pidx = -1
        var pc = 0.0
        if op_num_params(g.kind) >= 1:
            if op_num_params(g.kind) == 1 and g.p0.is_symbolic():
                pidx = g.p0.index
                pc = g.p0.coeff
            b.p0 = Param(g.p0.resolve(params))
            b.p1 = Param(g.p1.resolve(params))
            b.p2 = Param(g.p2.resolve(params))
        var st = _Step(b, pidx, pc)
        if pidx >= 0 and len(st.gen) == 0:
            raise Error("QM-E805: adjoint differentiation does not support parametric gate '" + op_name(g.kind) + "'. Hint: use Grad.PARAM_SHIFT or Grad.FINITE_DIFF.")
        out.append(st^)
    return out^


def _inverse_bound(g: Instruction) -> Instruction:
    """Inverse of a bound instruction (angles negated; U(t, p, l)^-1 = U(-t, -l, -p))."""
    var n = g
    n.kind = op_inverse_kind(g.kind)
    if g.kind == Op.U:
        n.p0 = -g.p0
        n.p1 = -g.p2
        n.p2 = -g.p1
    elif op_num_params(g.kind) == 1:
        n.p0 = -g.p0
    return n


def _anticommute(x1: Int, z1: Int, x2: Int, z2: Int) -> Bool:
    return ((pop_count(x1 & z2) + pop_count(z1 & x2)) & 1) == 1


def _commutes_with(st: _Step, x: Int, z: Int) -> Bool:
    """Does un-applying `st` commute with the Pauli (x, z)?"""
    if len(st.gen) > 0:
        for k in range(len(st.gen)):
            if _anticommute(x, z, st.gen.xs[k], st.gen.zs[k]):
                return False
        return True
    if ((x | z) & st.support) == 0:
        return True
    if op_is_diagonal(st.inst.kind) and x == 0:
        return True
    return False


def _pauli_value(sr: Float64, si: Float64, x: Int, z: Int) -> Float64:
    """<phi|P|phi> from sum_i conj(phi[i^x]) phi[i] (-1)^{|i&z|} (GpuStateVector.pauli_sums)."""
    var ny = Int(pop_count(x & z)) & 3
    if ny == 1:
        return -si
    if ny == 2:
        return -sr
    if ny == 3:
        return si
    return sr


def gpu_value_and_gradient[dt: DType](circuit: Circuit, h: PauliSum, params: List[Float64],
                                      emulate: Bool = False) raises -> DeviceGradient:
    """Adjoint value and gradient on the GPU state vector in precision `dt`
    (host-emulated kernels when `emulate`). See the module docstring."""
    circuit.validate()
    check_differentiable(circuit)
    var n = circuit.num_qubits
    check_fits(h, n)
    if len(params) != circuit.num_params():
        raise Error("QM-E804: expected " + String(circuit.num_params()) + " parameter values " + circuit.param_names_str() + ", got " + String(len(params)) + ".")
    var res = DeviceGradient()
    res.grad = List[Float64](length=circuit.num_params(), fill=0.0)
    # forward pass on the device, then lam = H psi on the host (fp64)
    # one device context for both registers (context creation dominates small problems)
    var ctx = Optional[DeviceContext]()
    comptime if device_kernels_enabled[dt]():
        if not emulate:
            ctx = Optional[DeviceContext](DeviceContext())
    var fwd = GpuStateVector[dt](n, emulate, ctx)
    fwd.evolve(circuit, params)
    var psi = fwd.to_host()
    res.method = String("gpu-emulator" if emulate else "gpu") + ":adjoint:" + fwd.precision()
    _ = fwd^
    var op = PauliOperator(h, n)
    var lam = StateVector(n)
    op.apply(psi, lam)
    var e = 0.0
    var lnorm = 0.0
    var dim = 1 << n
    for i in range(dim):
        e += psi.re[i] * lam.re[i] + psi.im[i] * lam.im[i]
        lnorm += lam.re[i] * lam.re[i] + lam.im[i] * lam.im[i]
    res.value = e
    lnorm = sqrt(lnorm)
    if len(res.grad) == 0 or lnorm < 1e-300:
        return res^
    # Phi = (|0>psi + |1>lam/||lam||) / sqrt(2) on n + 1 qubits (ancilla = qubit n)
    var phi = StateVector(n + 1)
    var s = 1.0 / sqrt(2.0)
    for i in range(dim):
        phi.re[i] = psi.re[i] * s
        phi.im[i] = psi.im[i] * s
        phi.re[i + dim] = lam.re[i] * s / lnorm
        phi.im[i + dim] = lam.im[i] * s / lnorm
    _ = psi^
    _ = lam^
    var sv = GpuStateVector[dt](n + 1, emulate, ctx)
    sv.load_host(phi)
    _ = phi^
    var anc = 1 << n
    var steps = _sweep_steps(circuit, params)
    var seg = Circuit(n + 1, name="adjoint-segment")
    var px = List[Int]()  # pending generator terms (system masks)
    var pz = List[Int]()
    var pc = List[Float64]()  # c_j * param coeff
    var pi_ = List[Int]()  # parameter index
    var k = len(steps) - 1
    while k >= -1:
        var flush = k < 0
        if not flush and len(px) > 0:
            for t in range(len(px)):
                if not _commutes_with(steps[k], px[t], pz[t]):
                    flush = True
                    break
        if flush and (len(seg.ops) > 0 or len(px) > 0):
            if len(seg.ops) > 0:
                sv.evolve(seg)
                seg = Circuit(n + 1, name="adjoint-segment")
            if len(px) > 0:
                var xs = List[Int](capacity=len(px))
                var zs = List[Int](capacity=len(px))
                for t in range(len(px)):
                    xs.append(px[t] | anc)
                    zs.append(pz[t] | anc)
                var vals = sv.pauli_sums(xs, zs)
                res.reductions += 1
                for t in range(len(px)):
                    var v = _pauli_value(vals[2 * t], vals[2 * t + 1], xs[t], zs[t])
                    res.grad[pi_[t]] += -2.0 * lnorm * pc[t] * v
                px.clear()
                pz.clear()
                pc.clear()
                pi_.clear()
        if k < 0:
            break
        ref st = steps[k]
        seg.append(_inverse_bound(st.inst))
        if st.pidx >= 0:
            for t in range(len(st.gen)):
                px.append(st.gen.xs[t])
                pz.append(st.gen.zs[t])
                pc.append(st.gen.cs[t] * st.pcoeff)
                pi_.append(st.pidx)
        k -= 1
    return res^


def _gpu_usable() -> Bool:
    comptime if is_defined["QUMO_GPU"]():
        try:
            return gpu_available()
        except:
            return False
    return False


def gpu_value_and_gradient_runtime(circuit: Circuit, h: PauliSum, params: List[Float64],
                                   precision: String = "auto", emulate: Bool = False) raises -> DeviceGradient:
    """gpu_value_and_gradient in the precision chosen at run time ("auto" |
    "fp32" | "fp64"; QM-E941 unknown, QM-E472 fp64 on an Apple GPU)."""
    var p = resolve_gpu_precision(precision)
    if p == "fp64":
        comptime if GPU_FP64_RUNTIME or not GPU_COMPILED:
            return gpu_value_and_gradient[DType.float64](circuit, h, params, emulate)
        else:
            raise gpu_fp64_error()
    return gpu_value_and_gradient[DType.float32](circuit, h, params, emulate)


def _check_device(device: String) raises:
    if device != "auto" and device != "cpu" and device != "gpu":
        raise Error("QM-E940: unknown device '" + device + "'. Hint: use 'auto', 'cpu' or 'gpu'.")


def _gpu_route(n: Int, device: String, mut note: String) -> Bool:
    """True when this call runs on the GPU; otherwise `note` says why not
    (empty for device 'cpu' and for 'auto' below the threshold)."""
    note = ""
    if device == "cpu":
        return False
    comptime if is_defined["QUMO_GPU"]():
        if not _gpu_usable():
            if device == "gpu":
                note = "no usable GPU device: ran on the CPU"
            return False
        if device == "auto" and n < GPU_MIN_QUBITS:
            return False
        return True
    else:
        if device == "gpu":
            note = "GPU support not compiled (build with -D QUMO_GPU): ran on the CPU"
        return False


def value_and_gradient_on(circuit: Circuit, h: PauliSum, params: List[Float64], device: String,
                          mut note: String, precision: String = "auto") raises -> DeviceGradient:
    """E(params) and the adjoint gradient on 'cpu' (fp64), 'gpu' (fp32 by
    default) or 'auto' (the GPU from GPU_MIN_QUBITS qubits when usable).
    A GPU request that cannot run there falls back to the CPU; `note` says why."""
    _check_device(device)
    _ = parse_gpu_precision(precision)
    if _gpu_route(circuit.num_qubits, device, note):
        return gpu_value_and_gradient_runtime(circuit, h, params, precision)
    var eng = ExpectationEngine(circuit, h)
    var res = DeviceGradient()
    res.grad = List[Float64](length=circuit.num_params(), fill=0.0)
    res.value = eng.value_and_gradient(params, res.grad, Grad.ADJOINT)
    res.method = "cpu:adjoint:fp64"
    return res^


def _flat_terms(observables: List[PauliSum], mut xs: List[Int], mut zs: List[Int]):
    for o in observables:
        for t in o.terms:
            xs.append(Int(t.x))
            zs.append(Int(t.z))


def expectations_on(circuit: Circuit, observables: List[PauliSum], params: List[Float64], device: String,
                    mut note: String, mut method: String, precision: String = "auto") raises -> List[Float64]:
    """<O_k> for every observable on one evolved state (one simulation, all
    Pauli terms reduced in one batch on the GPU). Same device rules as
    `value_and_gradient_on`; `method` = 'cpu:statevector:fp64' or
    'gpu:statevector:fp32|fp64'."""
    _check_device(device)
    _ = parse_gpu_precision(precision)
    var n = circuit.num_qubits
    for o in observables:
        check_fits(o, n)
    var out = List[Float64](capacity=len(observables))
    if _gpu_route(n, device, note):
        var xs = List[Int]()
        var zs = List[Int]()
        _flat_terms(observables, xs, zs)
        var p = resolve_gpu_precision(precision)
        check_gpu_precision(p)
        var vals: List[Float64]
        if p == "fp64":
            comptime if GPU_FP64_RUNTIME or not GPU_COMPILED:
                var sv = GpuStateVector[DType.float64](n)
                sv.evolve(circuit, params)
                vals = sv.pauli_sums(xs, zs)
                method = sv.method_name()
            else:
                raise gpu_fp64_error()
        else:
            var sv = GpuStateVector[DType.float32](n)
            sv.evolve(circuit, params)
            vals = sv.pauli_sums(xs, zs)
            method = sv.method_name()
        var t = 0
        for o in observables:
            var tot = 0.0
            for term in o.terms:
                tot += term.coeff * _pauli_value(vals[2 * t], vals[2 * t + 1], xs[t], zs[t])
                t += 1
            out.append(tot)
        return out^
    var psi = evolve(circuit, params)
    for o in observables:
        out.append(psi.expectation(o))
    method = "cpu:statevector:fp64"
    return out^


def expectations_emulated_gpu[dt: DType](circuit: Circuit, observables: List[PauliSum],
                                         params: List[Float64]) raises -> List[Float64]:
    """`expectations_on`'s GPU path on the host-emulated GPU kernels (tests)."""
    var xs = List[Int]()
    var zs = List[Int]()
    _flat_terms(observables, xs, zs)
    var sv = GpuStateVector[dt](circuit.num_qubits, True)
    sv.evolve(circuit, params)
    var vals = sv.pauli_sums(xs, zs)
    var out = List[Float64]()
    var t = 0
    for o in observables:
        var tot = 0.0
        for term in o.terms:
            tot += term.coeff * _pauli_value(vals[2 * t], vals[2 * t + 1], xs[t], zs[t])
            t += 1
        out.append(tot)
    return out^
