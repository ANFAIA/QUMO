"""Gradients of expectation values <psi(theta)|H|psi(theta)>.

Three interchangeable methods (PennyLane's `diff_method`, Qiskit's gradient
framework), selected by `Grad.*`:

  * ADJOINT (default on simulators; Jones & Gacon 2020, PennyLane-Lightning):
      forward pass psi = U|0>, lam = H psi, then one reverse sweep that applies
      U_k^dagger to both states and accumulates dE/dphi_k = 2 Im <lam|G_k|psi>
      for U_k = exp(-i phi_k G_k). Cost ~ 3x one simulation, memory 2 states,
      independent of the number of parameters. The <lam|G|psi> reductions are
      fused kernels (no third state is materialised).
      Generators: RX/RY/RZ -> X/2, Y/2, Z/2;  RXX/RYY/RZZ -> PP/2;
      P -> -|1><1|;  CP -> -|11><11|;  CRX/CRY/CRZ -> |1><1|_c (x) P/2;
      U(theta, phi, lam) is split as RZ(phi) RY(theta) RZ(lam) (the global phase
      e^{i(phi+lam)/2} cancels in expectation values).
  * PARAM_SHIFT (hardware-compatible; Mitarai 2018, Schuld 2019):
      two-term rule  dE/dphi = [E(phi + pi/2) - E(phi - pi/2)] / 2  for gates
      whose generator has two eigenvalues {+-1/2} (RX, RY, RZ, RXX, RYY, RZZ, the
      three U angles) or {0, 1} up to sign (P, CP);
      four-term rule for CRX/CRY/CRZ (generator spectrum {0, +-1/2}):
      dE/dphi = c+ [E(+pi/2) - E(-pi/2)] - c- [E(+3pi/2) - E(-3pi/2)],
      c+- = (sqrt2 +- 1) / (4 sqrt2).
      The shift is applied to each gate *occurrence*; the affine parameter map
      angle = coeff * theta[i] + offset contributes the chain-rule factor coeff,
      and occurrences sharing theta[i] are summed.
  * FINITE_DIFF: central differences with step h (default 1e-6) - universal fallback.

`ExpectationEngine` owns the reusable buffers (two state vectors, a prepared
`PauliOperator`, a scratch circuit for shifts) so optimisation loops do no
per-iteration allocation; the free functions are one-shot conveniences.
`batch_expectation` / `batch_gradient` evaluate many parameter sets (the
PUB-style primitive), parallel over sets when the state is small.

Error codes: QM-E804 (parameter vector length), QM-E805 (circuit not
differentiable: mid-circuit measurement, reset, noise, classical control),
QM-E806 (unknown gradient method).
"""

from std.math import pi, sqrt, sin, cos
from std.bit import pop_count
from std.runtime import parallelism_level
from max.algorithm import parallelize

from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import (
    Op, op_name, op_num_params, op_is_unitary, op_is_annotation, op_is_noise,
    op_inverse_kind, matrix_1q,
)
from qumo.ir.param import Param
from qumo.ir.pauli import PauliSum
from qumo.sim.statevector import StateVector, PAR_MIN_ITEMS
from qumo.sim.runner import evolve_into, apply_instruction
from qumo.algorithms.hamiltonians import PauliOperator, check_fits
from qumo.algorithms.linalg import sv_dot


struct Grad:
    """Gradient method selectors."""

    comptime ADJOINT = 0
    comptime PARAM_SHIFT = 1
    comptime FINITE_DIFF = 2


def grad_method_name(m: Int) -> String:
    if m == Grad.ADJOINT:
        return "adjoint"
    if m == Grad.PARAM_SHIFT:
        return "parameter-shift"
    if m == Grad.FINITE_DIFF:
        return "finite-diff"
    return "unknown"


def grad_method_from_name(name: String) raises -> Int:
    """'adjoint' | 'parameter-shift' ('param-shift', 'ps') | 'finite-diff' ('fd')."""
    var s = name.lower()
    if s == "adjoint":
        return Grad.ADJOINT
    if s == "parameter-shift" or s == "param-shift" or s == "parameter_shift" or s == "ps":
        return Grad.PARAM_SHIFT
    if s == "finite-diff" or s == "finite_diff" or s == "fd":
        return Grad.FINITE_DIFF
    raise Error("QM-E806: unknown gradient method '" + name + "'. Hint: use 'adjoint', 'parameter-shift' or 'finite-diff'.")


comptime _C_PLUS = (1.4142135623730951 + 1.0) / (4.0 * 1.4142135623730951)
comptime _C_MINUS = (1.4142135623730951 - 1.0) / (4.0 * 1.4142135623730951)


def check_differentiable(circuit: Circuit) raises:
    """Reject operations the expectation/gradient engine cannot differentiate."""
    for i in range(len(circuit.ops)):
        var g = circuit.ops[i]
        if g.cond >= 0:
            raise Error("QM-E805: instruction #" + String(i) + " is classically conditioned; expectation gradients need a unitary circuit. Hint: remove c_if feed-forward or estimate by sampling.")
        if g.kind == Op.RESET or op_is_noise(g.kind):
            raise Error("QM-E805: instruction #" + String(i) + " '" + op_name(g.kind) + "' is not unitary. Hint: gradients are defined for noiseless unitary circuits; use finite differences on sampled estimates instead.")
        if g.kind == Op.MEASURE:
            # only terminal measurements are allowed (they are ignored)
            for j in range(i + 1, len(circuit.ops)):
                var h = circuit.ops[j]
                if op_is_unitary(h.kind) and h.acts_on(g.q0):
                    raise Error("QM-E805: qubit " + String(g.q0) + " is measured mid-circuit (instruction #" + String(i) + "). Hint: move measurements to the end or drop them for expectation values.")


def _inverse_instruction(g: Instruction) -> Instruction:
    var n = g
    n.kind = op_inverse_kind(g.kind)
    if g.kind == Op.U:
        n.p0 = -g.p0
        n.p1 = -g.p2
        n.p2 = -g.p1
    elif op_num_params(g.kind) == 1:
        n.p0 = -g.p0
    return n


def _set_param(mut g: Instruction, slot: Int, p: Param):
    if slot == 0:
        g.p0 = p
    elif slot == 1:
        g.p1 = p
    else:
        g.p2 = p


def pauli_inner(lam: StateVector, psi: StateVector, x: Int, z: Int, cmask: Int) -> Tuple[Float64, Float64]:
    """<lam| Pi_c P |psi> where P = i^{|x&z|} X^x Z^z and Pi_c projects onto
    basis states with all `cmask` bits set (cmask = 0: no projector)."""
    var n = psi.size()
    var lr = lam.re.unsafe_ptr().as_unsafe_any_origin()
    var li = lam.im.unsafe_ptr().as_unsafe_any_origin()
    var pr = psi.re.unsafe_ptr().as_unsafe_any_origin()
    var pi_ = psi.im.unsafe_ptr().as_unsafe_any_origin()
    var nchunks = 1 if n < PAR_MIN_ITEMS else parallelism_level() * 2
    var chunk = (n + nchunks - 1) // nchunks
    var partial_r = List[Float64](length=nchunks, fill=0.0)
    var partial_i = List[Float64](length=nchunks, fill=0.0)
    var prp = partial_r.unsafe_ptr().as_unsafe_any_origin()
    var pip = partial_i.unsafe_ptr().as_unsafe_any_origin()

    def task(c: Int) {imm lr, imm li, imm pr, imm pi_, imm x, imm z, imm cmask, imm chunk, imm n, imm prp, imm pip}:
        var lo = c * chunk
        var hi = min(lo + chunk, n)
        var sr = 0.0
        var si = 0.0
        for i in range(lo, hi):
            if (i & cmask) != cmask:
                continue
            var j = i ^ x
            # conj(lam[j]) * psi[i]
            var ar = lr[unsafe_offset=j] * pr[unsafe_offset=i] + li[unsafe_offset=j] * pi_[unsafe_offset=i]
            var ai = lr[unsafe_offset=j] * pi_[unsafe_offset=i] - li[unsafe_offset=j] * pr[unsafe_offset=i]
            if (pop_count(i & z) & 1) == 1:
                sr -= ar
                si -= ai
            else:
                sr += ar
                si += ai
        prp[unsafe_offset=c] = sr
        pip[unsafe_offset=c] = si

    if nchunks == 1:
        task(0)
    else:
        parallelize(task, nchunks)
    var sr = 0.0
    var si = 0.0
    for c in range(nchunks):
        sr += partial_r[c]
        si += partial_i[c]
    var ny = Int(pop_count(x & z)) & 3
    if ny == 0:
        return (sr, si)
    if ny == 1:
        return (-si, sr)
    if ny == 2:
        return (-sr, -si)
    return (si, -sr)


struct ExpectationEngine(Movable):
    """Reusable evaluator of E(theta) = <0|U(theta)^dag H U(theta)|0> and its gradient.

    Owns two state vectors (psi, lam), the prepared Hamiltonian and a scratch
    copy of the circuit used for parameter shifts; counters `nfev` (circuit
    executions) and `ngrad` (gradient evaluations) feed optimiser reports.
    """

    var circuit: Circuit
    var work: Circuit
    var op: PauliOperator
    var psi: StateVector
    var lam: StateVector
    var num_params: Int
    var fd_step: Float64
    var nfev: Int
    var ngrad: Int

    def __init__(out self, circuit: Circuit, h: PauliSum, fd_step: Float64 = 1e-6) raises:
        circuit.validate()
        check_differentiable(circuit)
        self.op = PauliOperator(h, circuit.num_qubits)
        self.circuit = circuit.copy()
        self.work = circuit.copy()
        self.psi = StateVector(circuit.num_qubits)
        self.lam = StateVector(circuit.num_qubits)
        self.num_params = circuit.num_params()
        self.fd_step = fd_step
        self.nfev = 0
        self.ngrad = 0

    def _check(self, params: List[Float64]) raises:
        if len(params) != self.num_params:
            raise Error("QM-E804: expected " + String(self.num_params) + " parameter values " + self.circuit.param_names_str() + ", got " + String(len(params)) + ".")

    def prepare(mut self, params: List[Float64]) raises:
        """psi <- U(params)|0> (reuses the buffer)."""
        self._check(params)
        self.psi.reset_to_zero()
        evolve_into(self.psi, self.circuit, params)
        self.nfev += 1

    def value(mut self, params: List[Float64]) raises -> Float64:
        self.prepare(params)
        return self.op.expectation(self.psi)

    def _eval_work(mut self, params: List[Float64]) raises -> Float64:
        self.psi.reset_to_zero()
        evolve_into(self.psi, self.work, params)
        self.nfev += 1
        return self.op.expectation(self.psi)

    def value_and_gradient(mut self, params: List[Float64], mut grad: List[Float64], method: Int = Grad.ADJOINT) raises -> Float64:
        """Energy at `params`; `grad` is resized/overwritten with dE/dtheta."""
        self._check(params)
        if len(grad) != self.num_params:
            grad = List[Float64](length=self.num_params, fill=0.0)
        else:
            for ref gk in grad:
                gk = 0.0
        self.ngrad += 1
        if method == Grad.ADJOINT:
            return self._adjoint(params, grad)
        if method == Grad.PARAM_SHIFT:
            var e = self.value(params)
            self._param_shift(params, grad)
            return e
        if method == Grad.FINITE_DIFF:
            var e = self.value(params)
            self._finite_diff(params, grad)
            return e
        raise Error("QM-E806: unknown gradient method id " + String(method) + ". Hint: use Grad.ADJOINT, Grad.PARAM_SHIFT or Grad.FINITE_DIFF.")

    def gradient(mut self, params: List[Float64], method: Int = Grad.ADJOINT) raises -> List[Float64]:
        var g = List[Float64](length=self.num_params, fill=0.0)
        _ = self.value_and_gradient(params, g, method)
        return g^

    # ------------------------------------------------------------- adjoint
    def _gen_grad(self, kind: Int, q0: Int, q1: Int) raises -> Float64:
        """dE/dphi = 2 Im <lam|G|psi> for U = exp(-i phi G) at the current sweep point."""
        var b0 = 1 << q0
        var b1 = 1 << q1 if q1 >= 0 else 0
        var x = 0
        var z = 0
        var cm = 0
        var scale = 0.5
        if kind == Op.RX:
            x = b0
        elif kind == Op.RY:
            x = b0
            z = b0
        elif kind == Op.RZ:
            z = b0
        elif kind == Op.P:
            cm = b0
            scale = -1.0
        elif kind == Op.RXX:
            x = b0 | b1
        elif kind == Op.RYY:
            x = b0 | b1
            z = b0 | b1
        elif kind == Op.RZZ:
            z = b0 | b1
        elif kind == Op.CRX:
            cm = b0
            x = b1
        elif kind == Op.CRY:
            cm = b0
            x = b1
            z = b1
        elif kind == Op.CRZ:
            cm = b0
            z = b1
        elif kind == Op.CP:
            cm = b0 | b1
            scale = -1.0
        else:
            raise Error("QM-E805: adjoint differentiation does not support parametric gate '" + op_name(kind) + "'. Hint: use Grad.PARAM_SHIFT or Grad.FINITE_DIFF.")
        var v = pauli_inner(self.lam, self.psi, x, z, cm)
        return 2.0 * scale * v[1]

    def _unapply_1q(mut self, kind: Int, angle: Float64, q: Int):
        var m = matrix_1q(kind, -angle)
        self.psi.apply_1q(m, q)
        self.lam.apply_1q(m, q)

    def _adjoint(mut self, params: List[Float64], mut grad: List[Float64]) raises -> Float64:
        self.prepare(params)
        self.op.apply(self.psi, self.lam)
        var energy = sv_dot(self.psi, self.lam)[0]
        var k = len(self.circuit.ops) - 1
        while k >= 0:
            var g = self.circuit.ops[k]
            k -= 1
            if not op_is_unitary(g.kind) or g.kind == Op.I:
                continue
            if g.kind == Op.U and g.is_parametric():
                # U = e^{i(phi+lam)/2} RZ(phi) RY(theta) RZ(lam): peel RZ(phi), RY(theta), RZ(lam)
                var th = g.p0.resolve(params)
                var ph = g.p1.resolve(params)
                var la = g.p2.resolve(params)
                if g.p1.is_symbolic():
                    grad[g.p1.index] += g.p1.coeff * self._gen_grad(Op.RZ, g.q0, -1)
                self._unapply_1q(Op.RZ, ph, g.q0)
                if g.p0.is_symbolic():
                    grad[g.p0.index] += g.p0.coeff * self._gen_grad(Op.RY, g.q0, -1)
                self._unapply_1q(Op.RY, th, g.q0)
                if g.p2.is_symbolic():
                    grad[g.p2.index] += g.p2.coeff * self._gen_grad(Op.RZ, g.q0, -1)
                self._unapply_1q(Op.RZ, la, g.q0)
                continue
            if op_num_params(g.kind) == 1 and g.p0.is_symbolic():
                grad[g.p0.index] += g.p0.coeff * self._gen_grad(g.kind, g.q0, g.q1)
            var inv = _inverse_instruction(g)
            apply_instruction(self.psi, inv, params)
            apply_instruction(self.lam, inv, params)
        return energy

    # ------------------------------------------------------- parameter shift
    def _shifted(mut self, k: Int, slot: Int, shift: Float64, params: List[Float64]) raises -> Float64:
        var orig = self.circuit.ops[k].param(slot)
        var p = orig
        p.offset = orig.offset + shift
        _set_param(self.work.ops[k], slot, p)
        var e = self._eval_work(params)
        _set_param(self.work.ops[k], slot, orig)
        return e

    def _param_shift(mut self, params: List[Float64], mut grad: List[Float64]) raises:
        var half_pi = pi / 2
        for k in range(len(self.circuit.ops)):
            var g = self.circuit.ops[k]
            if not op_is_unitary(g.kind):
                continue
            var np = op_num_params(g.kind)
            for slot in range(np):
                var p = g.param(slot)
                if not p.is_symbolic():
                    continue
                var d: Float64
                if g.kind == Op.CRX or g.kind == Op.CRY or g.kind == Op.CRZ:
                    var a = self._shifted(k, slot, half_pi, params) - self._shifted(k, slot, -half_pi, params)
                    var b = self._shifted(k, slot, 3 * half_pi, params) - self._shifted(k, slot, -3 * half_pi, params)
                    d = _C_PLUS * a - _C_MINUS * b
                else:
                    d = 0.5 * (self._shifted(k, slot, half_pi, params) - self._shifted(k, slot, -half_pi, params))
                grad[p.index] += p.coeff * d

    # ---------------------------------------------------- finite differences
    def _finite_diff(mut self, params: List[Float64], mut grad: List[Float64]) raises:
        var x = params.copy()
        var h = self.fd_step
        for i in range(self.num_params):
            x[i] = params[i] + h
            var fp = self.value(x)
            x[i] = params[i] - h
            var fm = self.value(x)
            x[i] = params[i]
            grad[i] = (fp - fm) / (2.0 * h)


def shift_evaluations(circuit: Circuit) -> Int:
    """Circuit executions one parameter-shift gradient needs (cost estimate for agents)."""
    var total = 0
    for g in circuit.ops:
        if not op_is_unitary(g.kind):
            continue
        for slot in range(op_num_params(g.kind)):
            if g.param(slot).is_symbolic():
                if g.kind == Op.CRX or g.kind == Op.CRY or g.kind == Op.CRZ:
                    total += 4
                else:
                    total += 2
    return total


# ---------------------------------------------------------------------------
# One-shot conveniences
# ---------------------------------------------------------------------------
def expectation_value(circuit: Circuit, h: PauliSum, params: List[Float64] = List[Float64]()) raises -> Float64:
    var eng = ExpectationEngine(circuit, h)
    return eng.value(params)


def gradient(circuit: Circuit, h: PauliSum, params: List[Float64], method: Int = Grad.ADJOINT) raises -> List[Float64]:
    """dE/dtheta for E = <H> in the state circuit(params)|0>."""
    var eng = ExpectationEngine(circuit, h)
    return eng.gradient(params, method)


def value_and_gradient(circuit: Circuit, h: PauliSum, params: List[Float64], mut grad: List[Float64], method: Int = Grad.ADJOINT) raises -> Float64:
    var eng = ExpectationEngine(circuit, h)
    return eng.value_and_gradient(params, grad, method)


# ---------------------------------------------------------------------------
# Batched evaluation (PUB-style: one circuit, many parameter sets)
# ---------------------------------------------------------------------------
comptime _BATCH_PAR_MAX_QUBITS = 12


def batch_expectation(circuit: Circuit, h: PauliSum, param_sets: List[List[Float64]]) raises -> List[Float64]:
    """E(theta_b) for every parameter set b.

    Small states (<= 12 qubits) run parameter sets in parallel with one state
    vector per task; larger states run sequentially through one reused
    engine (the kernels themselves are then multi-threaded).
    """
    var nb = len(param_sets)
    var out = List[Float64](length=nb, fill=0.0)
    var eng = ExpectationEngine(circuit, h)
    for b in range(nb):
        eng._check(param_sets[b])
    if circuit.num_qubits > _BATCH_PAR_MAX_QUBITS or nb < 2:
        for b in range(nb):
            out[b] = eng.value(param_sets[b])
        return out^
    var op_ptr = out.unsafe_ptr().as_unsafe_any_origin()
    var nq = circuit.num_qubits
    var failed = List[Int](length=nb, fill=0)
    var fp = failed.unsafe_ptr().as_unsafe_any_origin()
    var hop = eng.op.copy()

    def task(b: Int) {imm circuit, imm hop, imm param_sets, imm op_ptr, imm nq, imm fp}:
        var sv = StateVector(nq)
        try:
            evolve_into(sv, circuit, param_sets[b])
            op_ptr[unsafe_offset=b] = hop.expectation(sv)
        except:
            fp[unsafe_offset=b] = 1

    parallelize(task, nb)
    for b in range(nb):
        if failed[b] != 0:
            raise Error("QM-E804: batch_expectation() failed on parameter set " + String(b) + ".")
    _ = len(hop.diag)
    return out^


def batch_gradient(circuit: Circuit, h: PauliSum, param_sets: List[List[Float64]], method: Int = Grad.ADJOINT) raises -> List[List[Float64]]:
    """Gradients for every parameter set (sequential, one reused engine)."""
    var eng = ExpectationEngine(circuit, h)
    var out = List[List[Float64]](capacity=len(param_sets))
    for b in range(len(param_sets)):
        out.append(eng.gradient(param_sets[b], method))
    return out^
