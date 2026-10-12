"""Composable quantum kernels: functions that return (or append to) `Circuit`s.

Philosophy (best-of synthesis):
  * CUDA-Q kernels / Q# operations: algorithms are plain functions producing
    IR, composable with `compose`, `compose_mapped`, and the IR functors
    `inverse()` (Adjoint), `controlled(q)` (Controlled) and `power(k)`.
  * PennyLane templates / Qiskit circuit library: each builder documents its
    qubit layout and its named symbolic parameters, so an AI agent can wire
    builders together without reading the implementation.
  * Qualtran bloqs: every builder states its resource shape (qubits,
    ancillas, parameter count) in the docstring.

Two flavours per primitive:
  * `append_*(mut c, ...)` - in-place, zero-copy, parameters are `Param`s that
    belong to `c` (create them with `c.parameter("name")`).
  * value-returning builders (`qft`, `grover`, `qpe`, ...) - fresh circuits
    with named parameters ready for `bind()` / gradients / optimisers.

Conventions: little-endian qubits (bit k of a basis index = qubit k), Qiskit-
style bitstring keys (leftmost = highest clbit), rotations R_P(theta) =
exp(-i theta P / 2).

Error codes: QM-E801 (invalid option), QM-E802 (invalid size/index),
QM-E803 (operator does not fit the register), QM-E808 (Trotter options),
QM-E810 (QPE options).
"""

from std.math import pi, sqrt, asin, floor, sin
from std.collections import Optional
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.param import Param
from qumo.ir.pauli import PauliTerm, PauliSum
from qumo.algorithms.hamiltonians import Graph, maxcut


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _range_list(lo: Int, hi: Int) -> List[Int]:
    var out = List[Int](capacity=max(hi - lo, 0))
    for i in range(lo, hi):
        out.append(i)
    return out^


def _check_n(n: Int, what: String, lo: Int = 1, hi: Int = 30) raises:
    if n < lo or n > hi:
        raise Error("QM-E802: " + what + " needs " + String(lo) + " <= n <= " + String(hi) + ", got " + String(n) + ".")


def bitstring_to_int(key: String) raises -> Int:
    """Qiskit-style key ("0110", leftmost = highest bit) -> integer."""
    var v = 0
    var bs = key.as_bytes()
    for i in range(key.byte_length()):
        var ch = Int(bs[i])
        if ch == ord("1"):
            v = (v << 1) | 1
        elif ch == ord("0"):
            v = v << 1
        else:
            raise Error("QM-E802: invalid bitstring '" + key + "'. Hint: keys contain only '0'/'1'.")
    return v


# ---------------------------------------------------------------------------
# State preparation
# ---------------------------------------------------------------------------
def ghz(n: Int) raises -> Circuit:
    """GHZ state (|0..0> + |1..1>)/sqrt(2).

    Layout: qubits 0..n-1. Gates: H(0) then a CX chain 0->1->...->n-1. Params: none.
    """
    _check_n(n, "ghz()", 1, 64)
    var c = Circuit(n, name="ghz")
    c.h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    return c^


# ---------------------------------------------------------------------------
# Quantum Fourier transform
# ---------------------------------------------------------------------------
def append_qft(mut c: Circuit, qubits: List[Int], do_swaps: Bool = True, approximation_degree: Int = 0):
    """In-place QFT on `qubits` (qubits[k] carries bit k of the register value)."""
    var n = len(qubits)
    var j = n - 1
    while j >= 0:
        c.h(qubits[j])
        var num_ent = max(0, j - max(0, approximation_degree - (n - j - 1)))
        var k = j - 1
        while k >= j - num_ent:
            c.cp(pi * (2.0 ** Float64(k - j)), qubits[j], qubits[k])
            k -= 1
        j -= 1
    if do_swaps:
        for i in range(n // 2):
            c.swap(qubits[i], qubits[n - 1 - i])


def qft(n: Int, do_swaps: Bool = True, approximation_degree: Int = 0) raises -> Circuit:
    """Quantum Fourier transform QFT|j> = 2^{-n/2} sum_k exp(2 pi i j k / 2^n) |k>.

    Layout: qubits 0..n-1 hold the register value little-endian. Params: none.
    Gates: n H, n(n-1)/2 CP, floor(n/2) SWAP. do_swaps=False leaves the output
    bit-reversed (saves the SWAP network when the next stage can relabel).
    approximation_degree drops the CP rotations with the smallest angles
    (Coppersmith AQFT, Qiskit semantics).
    """
    _check_n(n, "qft()", 1, 64)
    var c = Circuit(n, name="qft")
    append_qft(c, _range_list(0, n), do_swaps, approximation_degree)
    return c^


def iqft(n: Int, do_swaps: Bool = True, approximation_degree: Int = 0) raises -> Circuit:
    """Inverse QFT (Adjoint functor applied to `qft`). Layout as `qft`."""
    var c = qft(n, do_swaps, approximation_degree).inverse()
    c.name = "iqft"
    return c^


# ---------------------------------------------------------------------------
# Multi-controlled gates (Toffoli V-chain)
# ---------------------------------------------------------------------------
def num_mcz_ancillas(k: Int) -> Int:
    """Clean ancillas used by `append_mcz` on k qubits: max(0, k - 3)."""
    return max(0, k - 3)


def append_mcz(mut c: Circuit, qubits: List[Int], ancillas: List[Int]) raises:
    """Phase flip of |1...1> on `qubits` (symmetric multi-controlled Z).

    k <= 3 uses Z / CZ / CCZ directly; k >= 4 computes the AND of the first
    k-2 qubits into a CCX ladder over k-3 clean ancillas, applies CCZ with the
    last two qubits, and uncomputes (ancillas are returned to |0>).
    Cost for k >= 4: 2(k-3) CCX + 1 CCZ.
    """
    var k = len(qubits)
    if k == 0:
        raise Error("QM-E802: append_mcz() needs at least one qubit.")
    if k == 1:
        c.z(qubits[0])
        return
    if k == 2:
        c.cz(qubits[0], qubits[1])
        return
    if k == 3:
        c.ccz(qubits[0], qubits[1], qubits[2])
        return
    var need = k - 3
    if len(ancillas) < need:
        raise Error("QM-E802: multi-controlled Z on " + String(k) + " qubits needs " + String(need) + " clean ancillas, got " + String(len(ancillas)) + ". Hint: allocate num_mcz_ancillas(k) extra qubits.")
    c.ccx(qubits[0], qubits[1], ancillas[0])
    for i in range(2, k - 2):
        c.ccx(qubits[i], ancillas[i - 2], ancillas[i - 1])
    c.ccz(ancillas[k - 4], qubits[k - 2], qubits[k - 1])
    var i = k - 3
    while i >= 2:
        c.ccx(qubits[i], ancillas[i - 2], ancillas[i - 1])
        i -= 1
    c.ccx(qubits[0], qubits[1], ancillas[0])


def append_mcx(mut c: Circuit, controls: List[Int], target: Int, ancillas: List[Int]) raises:
    """Multi-controlled X = H(target) . MCZ(controls + target) . H(target)."""
    var qs = controls.copy()
    qs.append(target)
    c.h(target)
    append_mcz(c, qs, ancillas)
    c.h(target)


# ---------------------------------------------------------------------------
# Grover search
# ---------------------------------------------------------------------------
def phase_oracle(n: Int, marked: List[Int]) raises -> Circuit:
    """Phase oracle O|x> = -|x> for x in `marked`, +|x> otherwise.

    Layout: search qubits 0..n-1, clean ancillas n..n+num_mcz_ancillas(n)-1
    (restored to |0>). Params: none. Each marked item costs one MCZ plus
    X-conjugation of its zero bits.
    """
    _check_n(n, "phase_oracle()", 1, 30)
    var nanc = num_mcz_ancillas(n)
    var c = Circuit(n + nanc, name="phase_oracle")
    var qs = _range_list(0, n)
    var anc = _range_list(n, n + nanc)
    for i in range(len(marked)):
        var m = marked[i]
        if m < 0 or m >= (1 << n):
            raise Error("QM-E802: marked item " + String(m) + " is outside the search space [0, " + String(1 << n) + "). Hint: marked items are basis-state integers of the n search qubits.")
        for j in range(i):
            if marked[j] == m:
                raise Error("QM-E802: marked item " + String(m) + " is listed twice (the phase flips would cancel).")
        for q in range(n):
            if ((m >> q) & 1) == 0:
                c.x(q)
        append_mcz(c, qs, anc)
        for q in range(n):
            if ((m >> q) & 1) == 0:
                c.x(q)
    return c^


def grover_diffusion(n: Int) raises -> Circuit:
    """Diffusion operator 2|s><s| - I (up to global phase) on qubits 0..n-1.

    Layout as `phase_oracle` (shares its ancillas). Params: none.
    """
    _check_n(n, "grover_diffusion()", 1, 30)
    var nanc = num_mcz_ancillas(n)
    var c = Circuit(n + nanc, name="grover_diffusion")
    for q in range(n):
        c.h(q)
        c.x(q)
    append_mcz(c, _range_list(0, n), _range_list(n, n + nanc))
    for q in range(n):
        c.x(q)
        c.h(q)
    return c^


def grover_optimal_iterations(n: Int, num_marked: Int) -> Int:
    """round(pi / (4 theta) - 1/2) with sin(theta) = sqrt(M / 2^n)."""
    var big_n = Float64(1 << n)
    if num_marked <= 0 or Float64(num_marked) >= big_n:
        return 0
    var theta = asin(sqrt(Float64(num_marked) / big_n))
    # round(pi/(4 theta) - 1/2) == floor(pi/(4 theta))
    return max(Int(floor(pi / (4.0 * theta))), 0)


def grover_success_probability(n: Int, num_marked: Int, iterations: Int) -> Float64:
    """Ideal probability sin^2((2k+1) theta) of measuring a marked item."""
    var big_n = Float64(1 << n)
    if num_marked <= 0:
        return 0.0
    var theta = asin(sqrt(min(Float64(num_marked) / big_n, 1.0)))
    var s = sin((2.0 * Float64(iterations) + 1.0) * theta)
    return s * s


def grover(n: Int, marked: List[Int], iterations: Int = -1, measure: Bool = False) raises -> Circuit:
    """Grover search for the basis states in `marked` among 2^n items.

    Layout: search qubits 0..n-1, ancillas n..n+max(0,n-3)-1 (clean).
    iterations < 0 -> grover_optimal_iterations(n, len(marked)).
    measure=True measures search qubit q into clbit q (key = marked item in binary).
    Params: none.
    """
    _check_n(n, "grover()", 1, 30)
    if len(marked) == 0:
        raise Error("QM-E802: grover() needs at least one marked item.")
    var k = iterations if iterations >= 0 else grover_optimal_iterations(n, len(marked))
    var nanc = num_mcz_ancillas(n)
    var c = Circuit(n + nanc, name="grover")
    for q in range(n):
        c.h(q)
    var oracle = phase_oracle(n, marked)
    var diff = grover_diffusion(n)
    for _ in range(k):
        c.compose(oracle)
        c.compose(diff)
    if measure:
        for q in range(n):
            c.measure(q, q)
    return c^


def grover_bitstrings(keys: List[String], iterations: Int = -1, measure: Bool = True) raises -> Circuit:
    """Grover search for marked bitstrings, e.g. ["1011", "0110"] (Qiskit key order:
    leftmost char = highest qubit). n = key length; layout as `grover`."""
    if len(keys) == 0:
        raise Error("QM-E802: grover_bitstrings() needs at least one marked bitstring.")
    var n = keys[0].byte_length()
    var marked = List[Int]()
    for k in keys:
        if k.byte_length() != n:
            raise Error("QM-E802: marked bitstrings must all have the same length (got '" + keys[0] + "' and '" + k + "').")
        marked.append(bitstring_to_int(k))
    return grover(n, marked, iterations, measure)


# ---------------------------------------------------------------------------
# Oracular textbook algorithms
# ---------------------------------------------------------------------------
def bernstein_vazirani(n: Int, secret: Int, measure: Bool = False) raises -> Circuit:
    """Bernstein-Vazirani: recover s from f(x) = s.x mod 2 with one query.

    Phase-oracle form (Z on the secret bits) - equivalent to the textbook
    ancilla-in-|-> kickback but uses no ancilla. Layout: qubits 0..n-1; the
    measured key is s in binary. Params: none.
    """
    _check_n(n, "bernstein_vazirani()", 1, 64)
    if secret < 0 or (n < 63 and secret >= (1 << n)):
        raise Error("QM-E802: secret " + String(secret) + " does not fit in " + String(n) + " bits.")
    var c = Circuit(n, name="bernstein_vazirani")
    for q in range(n):
        c.h(q)
    for q in range(n):
        if ((secret >> q) & 1) == 1:
            c.z(q)
    for q in range(n):
        c.h(q)
    if measure:
        for q in range(n):
            c.measure(q, q)
    return c^


def deutsch_jozsa(n: Int, balanced_mask: Int = 0, constant_value: Int = 0, measure: Bool = False) raises -> Circuit:
    """Deutsch-Jozsa for f(x) = (mask . x mod 2) XOR constant_value.

    mask == 0 -> f constant; mask != 0 -> f balanced. Layout: inputs 0..n-1,
    oracle ancilla n prepared in |->. Measuring the inputs gives all zeros iff
    f is constant. Params: none.
    """
    _check_n(n, "deutsch_jozsa()", 1, 62)
    if balanced_mask < 0 or balanced_mask >= (1 << n):
        raise Error("QM-E802: balanced_mask does not fit in " + String(n) + " bits.")
    var c = Circuit(n + 1, name="deutsch_jozsa")
    c.x(n)
    c.h(n)
    for q in range(n):
        c.h(q)
    for q in range(n):
        if ((balanced_mask >> q) & 1) == 1:
            c.cx(q, n)
    if (constant_value & 1) == 1:
        c.x(n)
    for q in range(n):
        c.h(q)
    if measure:
        for q in range(n):
            c.measure(q, q)
    return c^


# ---------------------------------------------------------------------------
# Quantum phase estimation
# ---------------------------------------------------------------------------
def qpe(unitary: Circuit, n_eval: Int, state_prep: Optional[Circuit] = None, measure: Bool = False) raises -> Circuit:
    """Quantum phase estimation of U|psi> = exp(2 pi i phi)|psi>.

    Layout: evaluation register 0..n_eval-1 (little-endian, readout value
    k ~ phi * 2^n_eval), system register n_eval..n_eval+m-1 where
    m = unitary.num_qubits. `state_prep` (on <= m qubits) prepares |psi> on the
    system register. Construction: H on eval qubits, controlled-U^(2^j) from
    eval qubit j built with the IR functors `power(2^j)` and `controlled()`,
    then the inverse QFT. measure=True measures eval qubit j into clbit j.
    Gate count grows as (2^n_eval - 1) * |U|: keep n_eval <= ~12.
    Params: those of `unitary` (must be bound for controlled CP/CRZ/U).
    """
    if n_eval < 1 or n_eval > 16:
        raise Error("QM-E810: qpe() needs 1 <= n_eval <= 16, got " + String(n_eval) + ". Hint: precision is 2^-n_eval; U is repeated 2^n_eval - 1 times.")
    var m = unitary.num_qubits
    if m < 1:
        raise Error("QM-E810: qpe() needs a unitary acting on at least one qubit.")
    var c = Circuit(n_eval + m, name="qpe")
    var sys_map = _range_list(n_eval, n_eval + m)
    if state_prep:
        var sp = state_prep.value().copy()
        if sp.num_qubits > m:
            raise Error("QM-E810: state_prep acts on " + String(sp.num_qubits) + " qubits but the unitary has " + String(m) + ".")
        c.compose_mapped(sp, sys_map)
    for j in range(n_eval):
        c.h(j)
    for j in range(n_eval):
        var cu = unitary.power(1 << j).controlled(m)
        var qmap = sys_map.copy()
        qmap.append(j)
        c.compose_mapped(cu, qmap)
    c.compose(iqft(n_eval))  # eval register = qubits 0..n_eval-1
    if measure:
        for j in range(n_eval):
            c.measure(j, j)
    return c^


def qpe_phase(value: Int, n_eval: Int) -> Float64:
    """Phase estimate phi = value / 2^n_eval for a measured evaluation register value."""
    return Float64(value) / Float64(1 << n_eval)


# ---------------------------------------------------------------------------
# Pauli rotations / exponentials / Trotterisation
# ---------------------------------------------------------------------------
def append_pauli_rotation(mut c: Circuit, x: UInt64, z: UInt64, theta: Param) raises:
    """Append exp(-i theta/2 P) for the Pauli string P with symplectic masks (x, z).

    Construction: basis change (X: H, Y: Sdg.H) -> parity onto the last support
    qubit with a CX ladder -> RZ(theta) -> uncompute. Weight-1 strings become a
    single RZ and weight-2 strings a single RZZ (no ladder), so the parameter
    lives in exactly one rotation gate (parameter-shift friendly).
    The identity string only contributes a global phase (dropped when theta
    is symbolic).
    """
    var support = List[Int]()
    var mask = x | z
    for q in range(64):
        if ((mask >> UInt64(q)) & 1) == 1:
            support.append(q)
    if len(support) == 0:
        if not theta.is_symbolic():
            c.global_phase -= theta.value() / 2
        return
    if support[len(support) - 1] >= c.num_qubits:
        raise Error("QM-E803: Pauli string acts on qubit " + String(support[len(support) - 1]) + " but the circuit has " + String(c.num_qubits) + " qubits.")
    for q in support:
        var b = UInt64(1) << UInt64(q)
        var hx = (x & b) != 0
        var hz = (z & b) != 0
        if hx and hz:
            c.sdg(q)
            c.h(q)
        elif hx:
            c.h(q)
    var w = len(support)
    if w == 1:
        c.rz(theta, support[0])
    elif w == 2:
        c.rzz(theta, support[0], support[1])
    else:
        for i in range(w - 1):
            c.cx(support[i], support[i + 1])
        c.rz(theta, support[w - 1])
        var i = w - 2
        while i >= 0:
            c.cx(support[i], support[i + 1])
            i -= 1
    for q in support:
        var b = UInt64(1) << UInt64(q)
        var hx = (x & b) != 0
        var hz = (z & b) != 0
        if hx and hz:
            c.h(q)
            c.s(q)
        elif hx:
            c.h(q)


def append_pauli_exponential(mut c: Circuit, term: PauliTerm, t: Param) raises:
    """Append exp(-i t * coeff * P) for a PauliTerm (rotation angle 2 * coeff * t)."""
    append_pauli_rotation(c, term.x, term.z, t * (2.0 * term.coeff))


def pauli_exponential(term: PauliTerm, num_qubits: Int, time: Float64 = 1.0, param_name: String = "") raises -> Circuit:
    """Circuit for exp(-i t * coeff * P).

    Layout: qubits 0..num_qubits-1. Params: none, or one symbolic time
    parameter named `param_name` when given (then `time` is ignored).
    """
    var c = Circuit(num_qubits, name="pauli_exp")
    var t = Param(time)
    if param_name.byte_length() > 0:
        t = c.parameter(param_name)
    append_pauli_exponential(c, term, t)
    return c^


def trotter(h: PauliSum, time: Float64, steps: Int, order: Int = 1, num_qubits: Int = -1, param_name: String = "") raises -> Circuit:
    """Product-formula approximation of exp(-i H t).

    order 1 (Lie-Trotter): (prod_k e^{-i c_k P_k dt})^steps, error O(t^2/steps).
    order 2 (Strang):      symmetric half steps, error O(t^3/steps^2).
    Layout: qubits 0..n-1 (n = num_qubits or h.num_qubits()). Params: none, or a
    symbolic total time named `param_name` (each slice uses param * 1/steps).
    Identity terms become a global phase (constant time only).
    """
    if steps < 1:
        raise Error("QM-E808: trotter() needs steps >= 1, got " + String(steps) + ".")
    if order != 1 and order != 2:
        raise Error("QM-E808: trotter() supports order 1 (Lie-Trotter) or 2 (Strang), got " + String(order) + ". Hint: compose two order-2 steps (Suzuki) for higher orders.")
    var n = num_qubits if num_qubits > 0 else max(h.num_qubits(), 1)
    var c = Circuit(n, name="trotter")
    var tot = Param(time)
    if param_name.byte_length() > 0:
        tot = c.parameter(param_name)
    var dt = tot * (1.0 / Float64(steps))
    var half = tot * (0.5 / Float64(steps))
    var nt = len(h.terms)
    for _ in range(steps):
        if order == 1 or nt == 1:
            for t in h.terms:
                append_pauli_exponential(c, t, dt)
        else:
            for k in range(nt - 1):
                append_pauli_exponential(c, h.terms[k], half)
            append_pauli_exponential(c, h.terms[nt - 1], dt)
            var k = nt - 2
            while k >= 0:
                append_pauli_exponential(c, h.terms[k], half)
                k -= 1
    return c^


# ---------------------------------------------------------------------------
# Variational ansatze
# ---------------------------------------------------------------------------
def _rotation_ops(rotations: String) raises -> List[Int]:
    var out = List[Int]()
    var cleaned = rotations.replace(" ", "")
    for part in cleaned.split(","):
        var s = String(part)
        if s == "rx":
            out.append(Op.RX)
        elif s == "ry":
            out.append(Op.RY)
        elif s == "rz":
            out.append(Op.RZ)
        elif s.byte_length() > 0:
            raise Error("QM-E801: unknown rotation '" + s + "' in '" + rotations + "'. Hint: use a comma list of rx, ry, rz, e.g. 'ry,rz'.")
    if len(out) == 0:
        raise Error("QM-E801: hardware_efficient_ansatz() needs at least one rotation. Hint: rotations='ry,rz'.")
    return out^


def _rotation_layer(mut c: Circuit, n: Int, rots: List[Int], prefix: String, mut k: Int):
    for q in range(n):
        for op in rots:
            var p = c.parameter(prefix + "[" + String(k) + "]")
            c.add(op, q, p0=p)
            k += 1


def append_entangler(mut c: Circuit, n: Int, entanglement: String) raises:
    """CX entangling layer: 'linear' (i -> i+1), 'circular' (+ n-1 -> 0), 'full' (all i < j), 'none'."""
    if entanglement == "linear" or entanglement == "circular":
        for i in range(n - 1):
            c.cx(i, i + 1)
        if entanglement == "circular" and n > 2:
            c.cx(n - 1, 0)
    elif entanglement == "full":
        for i in range(n):
            for j in range(i + 1, n):
                c.cx(i, j)
    elif entanglement != "none":
        raise Error("QM-E801: unknown entanglement '" + entanglement + "'. Hint: use 'linear', 'circular', 'full' or 'none'.")


def hardware_efficient_ansatz(n: Int, layers: Int, entanglement: String = "linear", rotations: String = "ry,rz", prefix: String = "theta") raises -> Circuit:
    """Hardware-efficient ansatz (Kandala et al. 2017 / Qiskit EfficientSU2).

    Structure: [rotation layer, CX entangler] x layers, then a final rotation
    layer. Layout: qubits 0..n-1. Params: prefix[k], k = 0..n*R*(layers+1)-1
    with R = number of rotations; ordered layer-major, then qubit, then
    rotation (theta[0] = first rotation on qubit 0 of layer 0).
    """
    _check_n(n, "hardware_efficient_ansatz()", 1, 64)
    if layers < 0:
        raise Error("QM-E802: layers must be >= 0, got " + String(layers) + ".")
    var rots = _rotation_ops(rotations)
    var c = Circuit(n, name="hea")
    var k = 0
    for _ in range(layers):
        _rotation_layer(c, n, rots, prefix, k)
        append_entangler(c, n, entanglement)
    _rotation_layer(c, n, rots, prefix, k)
    return c^


def qaoa_ansatz(cost: PauliSum, p: Int, num_qubits: Int = -1) raises -> Circuit:
    """QAOA (Farhi et al.; Qiskit QAOAAnsatz): |+>^n then p rounds of
    exp(-i gamma_l H_C) exp(-i beta_l sum_q X_q).

    Layout: qubits 0..n-1 (n = num_qubits or cost.num_qubits()). Params (in
    order): gamma[0..p-1], beta[0..p-1]. Cost terms are compiled with
    `append_pauli_exponential` (ZZ -> one RZZ(2 c gamma), Z -> RZ); the mixer is
    RX(2 beta). Exact for diagonal (Ising/QUBO) costs; for non-commuting costs
    each round is a first-order Trotter step. Identity terms are dropped.
    """
    if p < 1:
        raise Error("QM-E802: qaoa_ansatz() needs p >= 1, got " + String(p) + ".")
    var n = num_qubits if num_qubits > 0 else cost.num_qubits()
    _check_n(n, "qaoa_ansatz()", 1, 64)
    var c = Circuit(n, name="qaoa")
    var gammas = c.parameters("gamma", p)
    var betas = c.parameters("beta", p)
    for q in range(n):
        c.h(q)
    for l in range(p):
        for t in cost.terms:
            if t.x == 0 and t.z == 0:
                continue
            append_pauli_exponential(c, t, gammas[l])
        for q in range(n):
            c.rx(betas[l] * 2.0, q)
    return c^


def qaoa_maxcut_ansatz(graph: Graph, p: Int) raises -> Circuit:
    """QAOA for MaxCut on `graph`: qaoa_ansatz(maxcut(graph), p, graph.num_nodes).

    Each edge (u, v, w) becomes RZZ(w * gamma_l) on (u, v). Params: gamma[0..p-1], beta[0..p-1].
    """
    return qaoa_ansatz(maxcut(graph, False), p, graph.num_nodes)


# ---------------------------------------------------------------------------
# Chemistry: UCC excitations (Jordan-Wigner) and UCCSD-lite
# ---------------------------------------------------------------------------
def _zstring(lo: Int, hi: Int) -> UInt64:
    """Z mask on qubits strictly between lo and hi."""
    var m: UInt64 = 0
    for q in range(lo + 1, hi):
        m |= UInt64(1) << UInt64(q)
    return m


def append_single_excitation(mut c: Circuit, theta: Param, p: Int, q: Int) raises:
    """exp(theta (a+_q a_p - a+_p a_q)) under Jordan-Wigner, p < q.

    = exp(-i theta/2 X_p Z..Z Y_q) exp(+i theta/2 Y_p Z..Z X_q): two Pauli
    rotations sharing `theta` (coefficients +1 and -1).
    """
    if not (p < q):
        raise Error("QM-E801: single excitation needs p < q, got " + String(p) + ", " + String(q) + ".")
    var bp = UInt64(1) << UInt64(p)
    var bq = UInt64(1) << UInt64(q)
    var zs = _zstring(p, q)
    # X_p Z.. Y_q : x = p|q, z = zs|q
    append_pauli_rotation(c, bp | bq, zs | bq, theta)
    # Y_p Z.. X_q : x = p|q, z = zs|p
    append_pauli_rotation(c, bp | bq, zs | bp, -theta)


def append_double_excitation(mut c: Circuit, theta: Param, p: Int, q: Int, r: Int, s: Int) raises:
    """exp(theta (a+_s a+_r a_q a_p - h.c.)) under Jordan-Wigner, p < q < r < s.

    Eight commuting Pauli rotations (angles +-theta/4) on X/Y patterns over
    (p, q, r, s) with Z strings inside (p, q) and (r, s); the same parameter
    drives all eight gates (exercises shared-parameter gradients).
    """
    if not (p < q and q < r and r < s):
        raise Error("QM-E801: double excitation needs p < q < r < s, got " + String(p) + "," + String(q) + "," + String(r) + "," + String(s) + ".")
    var zs = _zstring(p, q) | _zstring(r, s)
    var qs: List[Int] = [p, q, r, s]
    # patterns as 4 letters (0 = X, 1 = Y) for (p, q, r, s) and their signs
    var pats: List[Int] = [0b0001, 0b0010, 0b0100, 0b0111, 0b1000, 0b1011, 0b1101, 0b1110]
    var signs: List[Float64] = [-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]
    for k in range(8):
        var x: UInt64 = 0
        var z: UInt64 = zs
        for j in range(4):
            var b = UInt64(1) << UInt64(qs[j])
            x |= b
            # bit (3 - j) of the pattern -> letter for qs[j] (p is the leading letter)
            if ((pats[k] >> (3 - j)) & 1) == 1:
                z |= b
        append_pauli_rotation(c, x, z, theta * (signs[k] / 4.0))


def uccsd_ansatz(num_qubits: Int, num_electrons: Int, include_singles: Bool = True, include_doubles: Bool = True) raises -> Circuit:
    """UCCSD-lite (Peruzzo et al. / Qiskit UCCSD), first-order Trotterised.

    Spin orbitals interleaved (even = spin-up, odd = spin-down), Jordan-Wigner.
    Layout: qubits 0..num_qubits-1; Hartree-Fock reference = X on qubits
    0..num_electrons-1. Params: one per spin-conserving excitation, named
    "s(i->a)" for singles and "d(i,j->a,b)" for doubles, singles first.
    For H2 (4, 2): s(0->2), s(1->3), d(0,1->2,3).
    """
    _check_n(num_qubits, "uccsd_ansatz()", 2, 64)
    if num_electrons < 1 or num_electrons >= num_qubits:
        raise Error("QM-E802: uccsd_ansatz() needs 1 <= num_electrons < num_qubits.")
    var c = Circuit(num_qubits, name="uccsd")
    for q in range(num_electrons):
        c.x(q)
    if include_singles:
        for i in range(num_electrons):
            for a in range(num_electrons, num_qubits):
                if i % 2 == a % 2:
                    var t = c.parameter("s(" + String(i) + "->" + String(a) + ")")
                    append_single_excitation(c, t, i, a)
    if include_doubles:
        for i in range(num_electrons):
            for j in range(i + 1, num_electrons):
                for a in range(num_electrons, num_qubits):
                    for b in range(a + 1, num_qubits):
                        var up_occ = (1 if i % 2 == 0 else 0) + (1 if j % 2 == 0 else 0)
                        var up_vir = (1 if a % 2 == 0 else 0) + (1 if b % 2 == 0 else 0)
                        if up_occ != up_vir:
                            continue
                        var t = c.parameter("d(" + String(i) + "," + String(j) + "->" + String(a) + "," + String(b) + ")")
                        append_double_excitation(c, t, i, j, a, b)
    return c^


def h2_ansatz() raises -> Circuit:
    """UCCSD ansatz for 4-qubit Jordan-Wigner H2 (pairs with h2_sto3g(4)).

    Layout: q0 g-up, q1 g-down, q2 u-up, q3 u-down; starts from HF |0011>.
    Params: s(0->2), s(1->3), d(0,1->2,3). The exact (FCI) ground state is
    reached with both singles at 0 and d(0,1->2,3) = 0.11177 rad.
    """
    return uccsd_ansatz(4, 2)
