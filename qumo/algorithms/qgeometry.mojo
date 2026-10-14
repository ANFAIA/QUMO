"""Quantum geometry: Fubini-Study metric tensor and quantum natural gradient.

The metric follows PennyLane's `qml.metric_tensor` convention: for gates
U_k = exp(-i theta_k c_k G_k) (affine parameter angle = c_k theta + offset),

    g_kl = c_k c_l Re[ <chi_k|chi_l> - <chi_k|psi><psi|chi_l> ],
    |chi_k> = U_{>k} G_k U_{<=k} |0>,

the real part of the quantum geometric tensor (the covariance of the
generators). Generators are those of the adjoint method: RX/RY/RZ -> X/2,
Y/2, Z/2; RXX/RYY/RZZ -> PP/2; P -> -|1><1|; CP -> -|11><11|;
CRX/CRY/CRZ -> |1><1| (x) P/2.

Approximations (`approx`):
  * 'block-diag' (PennyLane default): parametrized layers from
    `parameter_layers` (a gate starts a new layer when one of its ancestors is
    in the current layer, PennyLane CircuitGraph.parametrized_layers); each
    block is the generator covariance in the state that precedes its layer;
    entries between layers are 0;
  * 'diag': the diagonal of 'block-diag';
  * 'full': the exact metric from the derivative states above.
For layered circuits the blocks of 'full' equal 'block-diag' exactly; for
other circuits 'block-diag' is PennyLane's approximation, reproduced
identically.

`QNGOptimizer` applies theta <- theta - lr pinv(g + lam I) grad (PennyLane
QNGOptimizer, with adjoint gradients).

Supported: rx, ry, rz, p, cp, rxx, ryy, rzz, crx, cry, crz whose trainable
parameter occurs in exactly one gate; everything else raises QM-E832 (e.g.
QAOA's shared gamma, or a trainable U angle). Restrict with `wrt`.
"""

from std.math import sqrt

from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_name, op_num_params, op_is_unitary, op_arity, op_is_annotation
from qumo.ir.pauli import PauliSum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import apply_instruction
from qumo.algorithms.gradients import ExpectationEngine, Grad, check_differentiable, pauli_inner
from qumo.algorithms.linalg import jacobi_eigh


@fieldwise_init
struct _Gen(ImplicitlyCopyable):
    """Generator scale * Pi_cm (x) P(x, z) of one trainable gate occurrence."""

    var op: Int  # instruction index
    var param: Int  # parameter index
    var coeff: Float64
    var x: Int
    var z: Int
    var cm: Int
    var scale: Float64


def _generator(g: Instruction, k: Int) raises -> _Gen:
    var b0 = 1 << g.q0
    var b1 = 1 << g.q1 if op_arity(g.kind) >= 2 else 0
    var kind = g.kind
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
        raise Error("QM-E832: metric_tensor() does not support trainable gate '" + op_name(kind) + "' (instruction #" + String(k) + "). Hint: use rx/ry/rz/p/cp/rxx/ryy/rzz/crx/cry/crz, or exclude its parameter with wrt=[...].")
    return _Gen(k, g.p0.index, g.p0.coeff, x, z, cm, scale)


def _trainable_mask(circuit: Circuit, wrt: List[Int]) raises -> List[Bool]:
    var np_ = circuit.num_params()
    var mask = List[Bool](length=np_, fill=len(wrt) == 0)
    for p in wrt:
        if p < 0 or p >= np_:
            raise Error("QM-E804: wrt index " + String(p) + " outside 0.." + String(np_ - 1) + ".")
        mask[p] = True
    return mask^


def _trainable_gates(circuit: Circuit, wrt: List[Int]) raises -> List[_Gen]:
    """Trainable gate occurrences in program order (QM-E832 for unsupported ones)."""
    check_differentiable(circuit)
    var mask = _trainable_mask(circuit, wrt)
    var seen = List[Int](length=circuit.num_params(), fill=-1)
    var out = List[_Gen]()
    for k in range(len(circuit.ops)):
        var g = circuit.ops[k]
        if not op_is_unitary(g.kind):
            continue
        var npar = op_num_params(g.kind)
        var trainable = False
        for s in range(npar):
            var p = g.param(s)
            if p.is_symbolic() and mask[p.index]:
                trainable = True
        if not trainable:
            continue
        if npar != 1:
            raise Error("QM-E832: metric_tensor(): trainable parameter in multi-parameter gate '" + op_name(g.kind) + "' (instruction #" + String(k) + "). Hint: write U as rz/ry/rz with separate parameters.")
        var gen = _generator(g, k)
        if seen[gen.param] >= 0:
            raise Error("QM-E832: metric_tensor(): parameter " + circuit.param_names[gen.param] + " occurs in instructions #" + String(seen[gen.param]) + " and #" + String(k) + "; the metric needs one gate per parameter. Hint: give each gate its own parameter (QAOA-style sharing is not supported), or restrict wrt.")
        seen[gen.param] = k
        out.append(gen)
    return out^


def _touches(g: Instruction, mut flags: List[Bool], nq: Int) -> Bool:
    if g.kind == Op.BARRIER:
        for q in range(nq):
            if flags[q]:
                return True
        return False
    for j in range(op_arity(g.kind)):
        if flags[g.qubit(j)]:
            return True
    return False


def _mark(g: Instruction, mut flags: List[Bool], nq: Int):
    if g.kind == Op.BARRIER:
        for q in range(nq):
            flags[q] = True
        return
    for j in range(op_arity(g.kind)):
        flags[g.qubit(j)] = True


def _layers_of(circuit: Circuit, gens: List[_Gen]) -> List[List[Int]]:
    """Layers as lists of positions in `gens` (PennyLane parametrized_layers)."""
    var nq = circuit.num_qubits
    var is_gen = List[Int](length=len(circuit.ops), fill=-1)
    for i in range(len(gens)):
        is_gen[gens[i].op] = i
    var layers = List[List[Int]]()
    var cur = List[Int]()
    var dep = List[Bool](length=nq, fill=False)
    for k in range(len(circuit.ops)):
        var g = circuit.ops[k]
        if op_is_annotation(g.kind) and g.kind != Op.BARRIER:
            continue
        if not op_is_unitary(g.kind) and g.kind != Op.BARRIER:
            continue
        if is_gen[k] >= 0:
            if _touches(g, dep, nq):
                layers.append(cur^)
                cur = List[Int]()
                for q in range(nq):
                    dep[q] = False
            cur.append(is_gen[k])
            _mark(g, dep, nq)
        elif _touches(g, dep, nq):
            _mark(g, dep, nq)
    if len(cur) > 0:
        layers.append(cur^)
    return layers^


def parameter_layers(circuit: Circuit, wrt: List[Int] = List[Int]()) raises -> List[List[Int]]:
    """Parametrized layers as lists of parameter indices, in program order.

    A trainable gate joins the current layer unless one of its ancestors (any
    gate on a causal path to it) is in the current layer; then it starts a
    new layer. Its generator therefore commutes back to the layer start.
    """
    var gens = _trainable_gates(circuit, wrt)
    var lay = _layers_of(circuit, gens)
    var out = List[List[Int]]()
    for l in lay:
        var ps = List[Int]()
        for i in l:
            ps.append(gens[i].param)
        out.append(ps^)
    return out^


def _real_inner(lam: StateVector, psi: StateVector, x: Int, z: Int, cm: Int) -> Float64:
    return pauli_inner(lam, psi, x, z, cm)[0]


def _apply_gen(src: StateVector, gen: _Gen) -> StateVector:
    """scale * Pi_cm P(x, z) |src> (fresh state)."""
    var out = StateVector(src.num_qubits)
    out.re[0] = 0.0
    var ny = 0
    var t = gen.x & gen.z
    while t != 0:
        ny += t & 1
        t >>= 1
    ny &= 3
    for i in range(src.size()):
        if (i & gen.cm) != gen.cm:
            continue
        var a = src.re[i] * gen.scale
        var b = src.im[i] * gen.scale
        var par = 0
        var u = i & gen.z
        while u != 0:
            par ^= u & 1
            u >>= 1
        if par == 1:
            a = -a
            b = -b
        # multiply by i^ny
        var r = a
        var m = b
        if ny == 1:
            r = -b
            m = a
        elif ny == 2:
            r = -a
            m = -b
        elif ny == 3:
            r = b
            m = -a
        var j = i ^ gen.x
        out.re[j] = r
        out.im[j] = m
    return out^


def metric_tensor(circuit: Circuit, params: List[Float64], approx: String = "block-diag",
                  wrt: List[Int] = List[Int]()) raises -> List[Float64]:
    """P x P row-major Fubini-Study metric (P = len(wrt), or all parameters),
    PennyLane convention; approx 'block-diag' (default), 'diag' or 'full'.
    Parameters that occur in no gate get zero rows."""
    circuit.validate()
    if len(params) != circuit.num_params():
        raise Error("QM-E804: metric_tensor() expected " + String(circuit.num_params()) + " parameter values " + circuit.param_names_str() + ", got " + String(len(params)) + ".")
    var mode = approx.lower()
    if mode != "block-diag" and mode != "diag" and mode != "full":
        raise Error("QM-E832: metric_tensor(): unknown approx '" + approx + "'. Hint: use 'block-diag', 'diag' or 'full'.")
    var gens = _trainable_gates(circuit, wrt)
    var np_ = circuit.num_params()
    var pos = List[Int](length=np_, fill=-1)
    var P = np_
    if len(wrt) > 0:
        P = len(wrt)
        for i in range(len(wrt)):
            pos[wrt[i]] = i
    else:
        for i in range(np_):
            pos[i] = i
    var G = List[Float64](length=P * P, fill=0.0)
    var nq = circuit.num_qubits
    if mode == "full":
        var psi = StateVector(nq)
        for g in circuit.ops:
            if op_is_unitary(g.kind):
                apply_instruction(psi, g, params)
        var chis = List[StateVector]()
        for gen in gens:
            var phi = StateVector(nq)
            for k in range(gen.op + 1):
                var g = circuit.ops[k]
                if op_is_unitary(g.kind):
                    apply_instruction(phi, g, params)
            var chi = _apply_gen(phi, gen)
            for k in range(gen.op + 1, len(circuit.ops)):
                var g = circuit.ops[k]
                if op_is_unitary(g.kind):
                    apply_instruction(chi, g, params)
            chis.append(chi^)
        for a in range(len(gens)):
            var ea = psi.inner(chis[a])  # <psi|chi_a>
            for b in range(a, len(gens)):
                var eb = psi.inner(chis[b])
                var ab = chis[a].inner(chis[b])
                # Re[<chi_a|chi_b> - <chi_a|psi><psi|chi_b>], <chi_a|psi> = conj(ea)
                var cr = ea[0] * eb[0] + ea[1] * eb[1]
                var v = gens[a].coeff * gens[b].coeff * (ab[0] - cr)
                var i = pos[gens[a].param]
                var j = pos[gens[b].param]
                G[i * P + j] = v
                G[j * P + i] = v
        return G^
    # block-diagonal / diagonal
    var layers = _layers_of(circuit, gens)
    for layer in layers:
        var last = 0
        var in_layer = List[Bool](length=len(circuit.ops), fill=False)
        for gi in layer:
            in_layer[gens[gi].op] = True
            last = max(last, gens[gi].op)
        var psi = StateVector(nq)
        var desc = List[Bool](length=nq, fill=False)
        for k in range(last + 1):
            var g = circuit.ops[k]
            if in_layer[k]:
                _mark(g, desc, nq)
                continue
            if g.kind == Op.BARRIER:
                if _touches(g, desc, nq):
                    _mark(g, desc, nq)
                continue
            if not op_is_unitary(g.kind):
                continue
            if _touches(g, desc, nq):
                _mark(g, desc, nq)
                continue
            apply_instruction(psi, g, params)
        var ev = List[Float64](capacity=len(layer))
        for gi in layer:
            var gen = gens[gi]
            ev.append(gen.scale * _real_inner(psi, psi, gen.x, gen.z, gen.cm))
        for a in range(len(layer)):
            var ga = gens[layer[a]]
            var i = pos[ga.param]
            var e2 = ga.scale * ga.scale * _real_inner(psi, psi, 0, 0, ga.cm)
            G[i * P + i] = ga.coeff * ga.coeff * (e2 - ev[a] * ev[a])
            if mode == "diag":
                continue
            for b in range(a + 1, len(layer)):
                var gb = gens[layer[b]]
                var j = pos[gb.param]
                var eab = ga.scale * gb.scale * _real_inner(psi, psi, ga.x | gb.x, ga.z | gb.z, ga.cm | gb.cm)
                var v = ga.coeff * gb.coeff * (eab - ev[a] * ev[b])
                G[i * P + j] = v
                G[j * P + i] = v
    return G^


def qng_direction(G: List[Float64], grad: List[Float64], lam: Float64 = 0.0) raises -> List[Float64]:
    """pinv(G + lam I) grad by the symmetric eigendecomposition (jacobi_eigh);
    eigenvalues with |e| <= 1e-12 max|e| are treated as zero (rcond 1e-12)."""
    var n = len(grad)
    if len(G) != n * n:
        raise Error("QM-E831: qng_direction(): metric has " + String(len(G)) + " entries for " + String(n) + " gradient components.")
    var A = G.copy()
    for i in range(n):
        A[i * n + i] += lam
    var e = jacobi_eigh(A, n)
    var emax = 0.0
    for v in e.values:
        emax = max(emax, abs(v))
    var out = List[Float64](length=n, fill=0.0)
    for k in range(n):
        var lk = e.values[k]
        if abs(lk) <= 1e-12 * emax or lk == 0.0:
            continue
        var proj = 0.0
        for i in range(n):
            proj += e.vectors[i * n + k] * grad[i]
        proj /= lk
        for i in range(n):
            out[i] += proj * e.vectors[i * n + k]
    return out^


struct QNGOptimizer(Copyable, Writable):
    """Quantum natural gradient descent (PennyLane QNGOptimizer):
    theta <- theta - lr pinv(g(theta) + lam I) dE/dtheta, with the metric of
    `metric_tensor(approx)` and adjoint gradients."""

    var lr: Float64
    var approx: String
    var lam: Float64
    var steps: Int

    def __init__(out self, lr: Float64 = 0.01, approx: String = "block-diag", lam: Float64 = 0.0) raises:
        if not (lr >= 0.0) or not (lam >= 0.0):
            raise Error("QM-E833: QNGOptimizer needs lr >= 0 and lam >= 0.")
        var a = approx.lower()
        if a != "block-diag" and a != "diag" and a != "full":
            raise Error("QM-E832: QNGOptimizer: unknown approx '" + approx + "'. Hint: use 'block-diag', 'diag' or 'full'.")
        self.lr = lr
        self.approx = a
        self.lam = lam
        self.steps = 0

    def step(mut self, circuit: Circuit, h: PauliSum, mut params: List[Float64]) raises -> Float64:
        """One QNG step on <h> in the state circuit(params)|0>; returns the
        cost BEFORE the step (PennyLane step_and_cost)."""
        var eng = ExpectationEngine(circuit, h)
        var g = List[Float64]()
        var cost = eng.value_and_gradient(params, g, Grad.ADJOINT)
        var G = metric_tensor(circuit, params, self.approx)
        var d = qng_direction(G, g, self.lam)
        for i in range(len(params)):
            params[i] = params[i] - self.lr * d[i]
        self.steps += 1
        return cost

    def write_to(self, mut w: Some[Writer]):
        w.write("QNGOptimizer(lr=", self.lr, ", approx=", self.approx, ", lam=", self.lam, ", steps=", self.steps, ")")
