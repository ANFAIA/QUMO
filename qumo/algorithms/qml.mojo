"""Quantum machine learning templates, embeddings and observables (PennyLane-class).

Every template writes into an existing circuit (`append_*(mut c, ...)`, the
QUMO builder convention) with exact PennyLane semantics, so QUMO models
reproduce PennyLane 0.45 numbers to machine precision (tests/test_qml.mojo).

Conventions (docs/QML.md):
  * Parameter order: declare weights first, then features:
        var w = c.parameters("w", k); var x = c.parameters("x", f)
    A parameter vector is then `weights ++ features`; features are ordinary
    affine `Param`s, so the adjoint method differentiates w.r.t. both.
  * Wires: templates take a `wires: List[Int]` of QUMO qubits listed in
    PennyLane wire order (wires[0] plays PennyLane wire 0). Expectation values
    of products of local observables do not depend on the labelling; when
    amplitudes / probabilities over several wires matter, map PennyLane wire w
    to QUMO qubit n-1-w (QUMO is little-endian), i.e. pass wires = [n-1, ..., 0].
  * Rot(phi, theta, omega) = RZ(phi), then RY(theta), then RZ(omega) (time order).
  * BasisState(x) = RX(pi x) per wire as an expectation value (phases differ).
  * IsingXX/YY/ZZ = rxx/ryy/rzz, U3 = c.u, 2-wire MultiRZ = rzz,
    PauliRot(theta, word) = append_pauli_rotation (exp(-i theta/2 P)).

Error code: QM-E830 (invalid template / observable arguments).
"""

from std.math import pi, sqrt, atan2
from std.bit import pop_count

from qumo.ir.circuit import Circuit
from qumo.ir.param import Param
from qumo.ir.pauli import PauliSum, PauliTerm
from qumo.sim.statevector import StateVector
from qumo.algorithms.library import append_pauli_rotation


comptime DIAG_OBS_MAX_QUBITS = 12


def _check_wires(c: Circuit, wires: List[Int], who: String) raises:
    if len(wires) == 0:
        raise Error("QM-E830: " + who + "() needs at least one wire.")
    for i in range(len(wires)):
        if wires[i] < 0 or wires[i] >= c.num_qubits:
            raise Error("QM-E830: " + who + "(): wire " + String(wires[i]) + " is outside the circuit's " + String(c.num_qubits) + " qubits.")
        for j in range(i):
            if wires[j] == wires[i]:
                raise Error("QM-E830: " + who + "(): wire " + String(wires[i]) + " is listed twice.")


def _rotation_kind(rotation: String, who: String) raises -> Int:
    var r = rotation.lower()
    if r == "x" or r == "rx":
        return 0
    if r == "y" or r == "ry":
        return 1
    if r == "z" or r == "rz":
        return 2
    raise Error("QM-E830: " + who + "(): unknown rotation '" + rotation + "'. Hint: use 'x', 'y' or 'z'.")


def _append_rotation(mut c: Circuit, kind: Int, angle: Param, q: Int):
    if kind == 0:
        c.rx(angle, q)
    elif kind == 1:
        c.ry(angle, q)
    else:
        c.rz(angle, q)


# ---------------------------------------------------------------------------
# Gates and embeddings
# ---------------------------------------------------------------------------
def append_rot(mut c: Circuit, phi: Param, theta: Param, omega: Param, q: Int):
    """PennyLane Rot(phi, theta, omega): RZ(phi), then RY(theta), then RZ(omega)."""
    c.rz(phi, q)
    c.ry(theta, q)
    c.rz(omega, q)


def append_angle_embedding(mut c: Circuit, features: List[Param], wires: List[Int], rotation: String = "x") raises:
    """PennyLane AngleEmbedding: rotation(features[i]) on wires[i].

    Fewer features than wires leaves the remaining wires untouched (as in
    PennyLane); more features than wires raises QM-E830.
    """
    _check_wires(c, wires, "append_angle_embedding")
    if len(features) > len(wires):
        raise Error("QM-E830: append_angle_embedding(): " + String(len(features)) + " features for " + String(len(wires)) + " wires. Hint: use at most one feature per wire.")
    var kind = _rotation_kind(rotation, "append_angle_embedding")
    for i in range(len(features)):
        _append_rotation(c, kind, features[i], wires[i])


def append_basis_embedding(mut c: Circuit, features: List[Param], wires: List[Int]) raises:
    """BasisState / BasisEmbedding for bits x in {0, 1}: RX(pi x) on wires[i].

    Equal to PennyLane's BasisState for every expectation value (the state
    differs by a phase -i per flipped wire); bits may be symbolic features.
    """
    _check_wires(c, wires, "append_basis_embedding")
    if len(features) != len(wires):
        raise Error("QM-E830: append_basis_embedding(): " + String(len(features)) + " bits for " + String(len(wires)) + " wires.")
    for i in range(len(features)):
        c.rx(features[i] * pi, wires[i])


def append_iqp_embedding(mut c: Circuit, x: List[Param], xx: List[Param], wires: List[Int], n_repeats: Int = 1) raises:
    """PennyLane IQPEmbedding with the default all-pairs pattern.

    Per repeat: H then RZ(x_i) on every wire, then MultiRZ(x_i x_j) = RZZ on
    every pair (i, j), i < j, in itertools.combinations order. Products are
    not affine, so they are separate parameters `xx` (one per pair, in that
    order; `iqp_feature_values` computes them).
    """
    _check_wires(c, wires, "append_iqp_embedding")
    var n = len(wires)
    if len(x) != n or len(xx) != n * (n - 1) // 2:
        raise Error("QM-E830: append_iqp_embedding(): needs " + String(n) + " features and " + String(n * (n - 1) // 2) + " pair features, got " + String(len(x)) + " and " + String(len(xx)) + ".")
    if n_repeats < 1:
        raise Error("QM-E830: append_iqp_embedding(): n_repeats must be >= 1.")
    for _ in range(n_repeats):
        for i in range(n):
            c.h(wires[i])
            c.rz(x[i], wires[i])
        var k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c.rzz(xx[k], wires[i], wires[j])
                k += 1


def iqp_embedding(n: Int, n_repeats: Int = 1) raises -> Circuit:
    """IQP feature circuit on qubits 0..n-1 (feature i on qubit i).

    Parameters: x[0..n-1], then xx[i,j] for i < j (combinations order);
    bind with `iqp_feature_values(x)`.
    """
    if n < 1 or n > 24:
        raise Error("QM-E830: iqp_embedding() needs 1 <= n <= 24, got " + String(n) + ".")
    var c = Circuit(n, 0, "iqp_embedding")
    var x = c.parameters("x", n)
    var xx = List[Param]()
    for i in range(n):
        for j in range(i + 1, n):
            xx.append(c.parameter("xx[" + String(i) + "," + String(j) + "]"))
    var wires = List[Int]()
    for q in range(n):
        wires.append(q)
    append_iqp_embedding(c, x, xx, wires, n_repeats)
    return c^


def iqp_feature_values(x: List[Float64]) -> List[Float64]:
    """[x_0..x_{n-1}] ++ [x_i x_j for i < j]: the parameter vector of iqp_embedding."""
    var out = x.copy()
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            out.append(x[i] * x[j])
    return out^


def zz_feature_map(n: Int, reps: Int = 2) raises -> Circuit:
    """Qiskit ZZFeatureMap(n, reps, entanglement='full', alpha=2) on qubits 0..n-1.

    Per repetition: H on every qubit, P(2 x_i) on qubit i, then for every pair
    i < j: CX(i, j), P(2 phi_ij) on j, CX(i, j), with
    phi_ij = (pi - x_i)(pi - x_j). Parameters: x[0..n-1], then phi[i,j];
    bind with `zz_feature_values(x)`.
    """
    if n < 1 or n > 24 or reps < 1:
        raise Error("QM-E830: zz_feature_map() needs 1 <= n <= 24 and reps >= 1, got n=" + String(n) + ", reps=" + String(reps) + ".")
    var c = Circuit(n, 0, "zz_feature_map")
    var x = c.parameters("x", n)
    var ph = List[Param]()
    for i in range(n):
        for j in range(i + 1, n):
            ph.append(c.parameter("phi[" + String(i) + "," + String(j) + "]"))
    for _ in range(reps):
        for i in range(n):
            c.h(i)
        for i in range(n):
            c.p(x[i] * 2.0, i)
        var k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c.cx(i, j)
                c.p(ph[k] * 2.0, j)
                c.cx(i, j)
                k += 1
    return c^


def zz_feature_values(x: List[Float64]) -> List[Float64]:
    """[x_0..x_{n-1}] ++ [(pi - x_i)(pi - x_j) for i < j]: the parameter vector of zz_feature_map."""
    var out = x.copy()
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            out.append((pi - x[i]) * (pi - x[j]))
    return out^


# ---------------------------------------------------------------------------
# Amplitude embedding (real amplitudes, Mottonen uniformly controlled RY tree)
# ---------------------------------------------------------------------------
def _log2_exact(n: Int) -> Int:
    if n < 1 or (n & (n - 1)) != 0:
        return -1
    var k = 0
    while (1 << k) < n:
        k += 1
    return k


def _prepared_amplitudes(amplitudes: List[Float64], m: Int, normalize: Bool, pad_with: Float64, who: String) raises -> List[Float64]:
    """Pad to 2^m and normalise (PennyLane AmplitudeEmbedding rules)."""
    var dim = 1 << m
    if len(amplitudes) > dim:
        raise Error("QM-E830: " + who + "(): " + String(len(amplitudes)) + " amplitudes do not fit " + String(m) + " qubits (max " + String(dim) + ").")
    if len(amplitudes) == 0:
        raise Error("QM-E830: " + who + "(): no amplitudes.")
    var a = amplitudes.copy()
    while len(a) < dim:
        a.append(pad_with)
    var s = 0.0
    for v in a:
        s += v * v
    if s <= 0.0:
        raise Error("QM-E830: " + who + "(): the amplitude vector has zero norm.")
    if abs(s - 1.0) > 1e-10:
        if normalize or pad_with != 0.0:
            var inv = 1.0 / sqrt(s)
            for ref v in a:
                v *= inv
        else:
            raise Error("QM-E830: " + who + "(): amplitudes have squared norm " + String(s) + ", not 1. Hint: pass normalize=True.")
    return a^


def mottonen_tree_angles(amplitudes: List[Float64]) raises -> List[Float64]:
    """RY-tree angles of a real amplitude vector of length 2^n (scale-invariant).

    Level k (0..n-1) acts on the k-th most significant bit of the amplitude
    index; pattern j (0..2^k-1) is the value of the k more significant bits.
    Result index (2^k - 1) + j holds alpha_{k,j}:
      * k < n-1: 2 atan2(||right half||, ||left half||) of the block with prefix j;
      * k = n-1: 2 atan2(a[2j+1], a[2j]), signed, so negative amplitudes are exact.
    """
    var n = _log2_exact(len(amplitudes))
    if n < 1 or n > 20:
        raise Error("QM-E830: mottonen_tree_angles() needs 2^n amplitudes with 1 <= n <= 20, got " + String(len(amplitudes)) + ". Hint: pad with zeros to a power of two.")
    var dim = len(amplitudes)
    var nrm = 0.0
    for v in amplitudes:
        nrm += v * v
    if nrm <= 0.0:
        raise Error("QM-E830: mottonen_tree_angles(): the amplitude vector has zero norm.")
    var out = List[Float64](length=dim - 1, fill=0.0)
    for k in range(n):
        var block = dim >> k
        var half = block >> 1
        for j in range(1 << k):
            var base = j * block
            var alpha: Float64
            if k == n - 1:
                alpha = 2.0 * atan2(amplitudes[base + 1], amplitudes[base])
            else:
                var l2 = 0.0
                var r2 = 0.0
                for i in range(half):
                    var lv = amplitudes[base + i]
                    var rv = amplitudes[base + half + i]
                    l2 += lv * lv
                    r2 += rv * rv
                alpha = 2.0 * atan2(sqrt(r2), sqrt(l2))
            out[(1 << k) - 1 + j] = alpha
    return out^


def mottonen_angles(amplitudes: List[Float64]) raises -> List[Float64]:
    """State-preparation angles in the layout of PennyLane's tutorial
    `get_angles` (tutorial_variational_classifier), generalised to n qubits.

    Layout: [alpha_0] followed, for each level k = 1..n-1 and each pattern
    j = 2^k-1 down to 0, by the pair (-alpha_{k,j}/2, +alpha_{k,j}/2) (the two
    RY angles of a controlled rotation written as CNOT-RY-CNOT-RY).
    For 2 qubits this is exactly get_angles(x) = [beta2, -beta1/2, beta1/2,
    -beta0/2, beta0/2] (to < 1e-12: get_angles adds a 1e-12 regulariser
    inside its square roots). Length 2^(n+1) - 3. The compact 2^n - 1 angles
    are `mottonen_tree_angles`.
    """
    var t = mottonen_tree_angles(amplitudes)
    var n = _log2_exact(len(amplitudes))
    var out = List[Float64]()
    out.append(t[0])
    for k in range(1, n):
        var j = (1 << k) - 1
        while j >= 0:
            var a = t[(1 << k) - 1 + j]
            out.append(-a / 2.0)
            out.append(a / 2.0)
            j -= 1
    return out^


def _gray(i: Int) -> Int:
    return i ^ (i >> 1)


def append_amplitude_embedding_real(mut c: Circuit, amplitudes: List[Float64], wires: List[Int],
                                    normalize: Bool = True, pad_with: Float64 = 0.0) raises:
    """Real amplitude embedding by a Mottonen uniformly controlled RY tree.

    wires[i] carries bit (m-1-i) of the amplitude index (wires[0] = most
    significant, PennyLane order). With wires = [m-1, ..., 0] on a fresh
    m-qubit circuit the prepared state has sv.re[k] = a[k] (see
    `amplitude_state`). Shorter inputs are padded with `pad_with`; the vector
    is normalised when `normalize` or `pad_with != 0` (PennyLane rules), and
    otherwise must have unit norm (QM-E830). Signed real amplitudes are exact.
    Cost: 2^m - 1 RY and 2^m - 2 CX, starting from |0...0> on `wires`.
    """
    _check_wires(c, wires, "append_amplitude_embedding_real")
    var m = len(wires)
    if m > 20:
        raise Error("QM-E830: append_amplitude_embedding_real() supports at most 20 wires.")
    var a = _prepared_amplitudes(amplitudes, m, normalize, pad_with, "append_amplitude_embedding_real")
    var alpha = mottonen_tree_angles(a)
    c.ry(alpha[0], wires[0])
    for k in range(1, m):
        var nl = 1 << k
        var base = nl - 1
        for i in range(nl):
            var g = _gray(i)
            var th = 0.0
            for j in range(nl):
                var s = 1.0 if (pop_count(j & g) & 1) == 0 else -1.0
                th += s * alpha[base + j]
            c.ry(th / Float64(nl), wires[k])
            var diff = g ^ _gray((i + 1) % nl)
            var b = 0
            while (diff >> b) != 1:
                b += 1
            c.cx(wires[k - 1 - b], wires[k])


def amplitude_state(amplitudes: List[Float64], n: Int, normalize: Bool = True, pad_with: Float64 = 0.0) raises -> StateVector:
    """The n-qubit state with sv.re[k] = a[k] (padded / normalised as in
    append_amplitude_embedding_real). The index k is the QUMO basis index,
    equal to PennyLane's index when wire w is QUMO qubit n-1-w."""
    if n < 1 or n > 26:
        raise Error("QM-E830: amplitude_state() needs 1 <= n <= 26, got " + String(n) + ".")
    var a = _prepared_amplitudes(amplitudes, n, normalize, pad_with, "amplitude_state")
    var sv = StateVector(n)
    for k in range(len(a)):
        sv.re[k] = a[k]
    return sv^


# ---------------------------------------------------------------------------
# Layer templates
# ---------------------------------------------------------------------------
def append_strongly_entangling_layers(mut c: Circuit, weights: List[Param], wires: List[Int],
                                      ranges: List[Int] = List[Int](), imprimitive: String = "cx") raises:
    """PennyLane StronglyEntanglingLayers: weights[(l*n + i)*3 + k] = W[l][i][k].

    Layer l: Rot(W[l][i]) on every wire, then (n > 1) imprimitive(wires[i],
    wires[(i + r_l) mod n]) for i = 0..n-1; default r_l = (l mod (n-1)) + 1.
    `imprimitive` is 'cx' or 'cz'.
    """
    _check_wires(c, wires, "append_strongly_entangling_layers")
    var n = len(wires)
    if len(weights) == 0 or len(weights) % (3 * n) != 0:
        raise Error("QM-E830: append_strongly_entangling_layers(): " + String(len(weights)) + " weights is not a multiple of 3 * " + String(n) + " (layers x wires x 3).")
    var layers = len(weights) // (3 * n)
    var imp = imprimitive.lower()
    if imp != "cx" and imp != "cnot" and imp != "cz":
        raise Error("QM-E830: append_strongly_entangling_layers(): unknown imprimitive '" + imprimitive + "'. Hint: use 'cx' or 'cz'.")
    if len(ranges) != 0 and len(ranges) != layers:
        raise Error("QM-E830: append_strongly_entangling_layers(): " + String(len(ranges)) + " ranges for " + String(layers) + " layers.")
    for l in range(layers):
        for i in range(n):
            var o = (l * n + i) * 3
            append_rot(c, weights[o], weights[o + 1], weights[o + 2], wires[i])
        if n > 1:
            var r = (l % (n - 1)) + 1
            if len(ranges) != 0:
                r = ranges[l]
                if r < 1 or r >= n:
                    raise Error("QM-E830: append_strongly_entangling_layers(): range " + String(r) + " must lie in 1.." + String(n - 1) + ".")
            for i in range(n):
                var a = wires[i]
                var b = wires[(i + r) % n]
                if imp == "cz":
                    c.cz(a, b)
                else:
                    c.cx(a, b)


def append_basic_entangler_layers(mut c: Circuit, weights: List[Param], wires: List[Int], rotation: String = "x") raises:
    """PennyLane BasicEntanglerLayers: weights[l*n + i] = W[l][i].

    Layer l: rotation(W[l][i]) on every wire, then a CNOT ring
    (n > 2), a single CNOT(wires[0], wires[1]) (n = 2), or nothing (n = 1).
    """
    _check_wires(c, wires, "append_basic_entangler_layers")
    var n = len(wires)
    if len(weights) == 0 or len(weights) % n != 0:
        raise Error("QM-E830: append_basic_entangler_layers(): " + String(len(weights)) + " weights is not a multiple of " + String(n) + " wires.")
    var kind = _rotation_kind(rotation, "append_basic_entangler_layers")
    for l in range(len(weights) // n):
        for i in range(n):
            _append_rotation(c, kind, weights[l * n + i], wires[i])
        if n == 2:
            c.cx(wires[0], wires[1])
        elif n > 2:
            for i in range(n):
                c.cx(wires[i], wires[(i + 1) % n])


def pauli_words_gray(n: Int) raises -> List[String]:
    """The 4^n - 1 non-identity Pauli words of ArbitraryUnitary, in PennyLane's
    n-ary Gray-code order (character k acts on wires[k]); n = 2 gives
    XI, YI, ZI, ZX, IX, XX, YX, YY, ZY, IY, XY, XZ, YZ, ZZ, IZ."""
    if n < 1 or n > 6:
        raise Error("QM-E830: pauli_words_gray() supports 1 <= n <= 6, got " + String(n) + ".")
    var letters: List[String] = ["I", "X", "Y", "Z"]
    var out = List[String]()
    var total = 1 << (2 * n)
    for i in range(1, total):
        var base = List[Int](length=n, fill=0)
        var v = i
        for j in range(n):
            base[j] = v % 4
            v //= 4
        var code = List[Int](length=n, fill=0)
        var shift = 0
        var j = n - 1
        while j >= 0:
            code[j] = (base[j] + shift) % 4
            shift += 4 - code[j]
            j -= 1
        var w = String()
        for k in range(n):
            w += letters[code[k]]
        out.append(w)
    return out^


def append_arbitrary_unitary(mut c: Circuit, weights: List[Param], wires: List[Int]) raises:
    """PennyLane ArbitraryUnitary: PauliRot(weights[i], word_i) for the 4^n - 1
    Gray-code words of `pauli_words_gray` (character k on wires[k])."""
    _check_wires(c, wires, "append_arbitrary_unitary")
    var n = len(wires)
    if n > 6:
        raise Error("QM-E830: append_arbitrary_unitary() supports at most 6 wires.")
    var words = pauli_words_gray(n)
    if len(weights) != len(words):
        raise Error("QM-E830: append_arbitrary_unitary(): needs 4^" + String(n) + " - 1 = " + String(len(words)) + " weights, got " + String(len(weights)) + ".")
    for i in range(len(words)):
        var x: UInt64 = 0
        var z: UInt64 = 0
        var bs = words[i].as_bytes()
        for k in range(n):
            var bit = UInt64(1) << UInt64(wires[k])
            var ch = Int(bs[k])
            if ch == ord("X"):
                x |= bit
            elif ch == ord("Y"):
                x |= bit
                z |= bit
            elif ch == ord("Z"):
                z |= bit
        append_pauli_rotation(c, x, z, weights[i])


def append_controlled_u3(mut c: Circuit, ctrl: Int, target: Int, theta: Param, phi: Param, lam: Param) raises:
    """Controlled-U3(theta, phi, lam) with symbolic parameters.

    P(phi/2) P(lam/2) on the control, then CRZ(lam), CRY(theta), CRZ(phi):
    exactly |0><0| (x) I + |1><1| (x) U3 (U3 = e^{i(phi+lam)/2} RZ(phi) RY(theta)
    RZ(lam)). Every angle is affine, so the adjoint method differentiates it
    (Circuit.controlled() needs bound U angles, QM-E202).
    """
    if ctrl == target or ctrl < 0 or target < 0 or ctrl >= c.num_qubits or target >= c.num_qubits:
        raise Error("QM-E830: append_controlled_u3(): invalid control/target " + String(ctrl) + "/" + String(target) + " on " + String(c.num_qubits) + " qubits.")
    c.p(phi * 0.5, ctrl)
    c.p(lam * 0.5, ctrl)
    c.crz(lam, ctrl, target)
    c.cry(theta, ctrl, target)
    c.crz(phi, ctrl, target)


def append_qcnn_conv(mut c: Circuit, weights: List[Param], wires: List[Int], skip_first_layer: Bool = False) raises:
    """Convolutional layer of PennyLane's tutorial_learning_few_data (15 weights).

    For p in (0, 1), for every index i with i mod 2 == p and i < n-1, on the
    pair (wires[i], wires[i+1]): [U3(w0..2), U3(w3..5) unless
    skip_first_layer, only when i is even], IsingXX(w6), IsingYY(w7),
    IsingZZ(w8), U3(w9..11) on wires[i], U3(w12..14) on wires[i+1].
    """
    _check_wires(c, wires, "append_qcnn_conv")
    if len(weights) != 15:
        raise Error("QM-E830: append_qcnn_conv() needs 15 weights, got " + String(len(weights)) + ".")
    var n = len(wires)
    if n < 2:
        raise Error("QM-E830: append_qcnn_conv() needs at least 2 wires.")
    for p in range(2):
        for i in range(n):
            if i % 2 == p and i < n - 1:
                var a = wires[i]
                var b = wires[i + 1]
                if i % 2 == 0 and not skip_first_layer:
                    c.u(weights[0], weights[1], weights[2], a)
                    c.u(weights[3], weights[4], weights[5], b)
                c.rxx(weights[6], a, b)
                c.ryy(weights[7], a, b)
                c.rzz(weights[8], a, b)
                c.u(weights[9], weights[10], weights[11], a)
                c.u(weights[12], weights[13], weights[14], b)


def append_qcnn_pool(mut c: Circuit, weights: List[Param], sinks: List[Int], sources: List[Int]) raises:
    """Pooling layer of tutorial_learning_few_data (3 weights): for every k,
    "measure sources[k], if 1 apply U3(w) to sinks[k]", written by the
    deferred-measurement principle as append_controlled_u3(sources[k], sinks[k]).
    The tutorial pools wires[1], wires[3], ... into wires[0], wires[2], ..."""
    if len(weights) != 3:
        raise Error("QM-E830: append_qcnn_pool() needs 3 weights, got " + String(len(weights)) + ".")
    if len(sinks) != len(sources) or len(sinks) == 0:
        raise Error("QM-E830: append_qcnn_pool(): " + String(len(sinks)) + " sinks for " + String(len(sources)) + " sources.")
    for k in range(len(sinks)):
        append_controlled_u3(c, sources[k], sinks[k], weights[0], weights[1], weights[2])


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------
def diagonal_observable(values: List[Float64]) raises -> PauliSum:
    """The Pauli-Z expansion of diag(values) (values[i] = <i|O|i>, QUMO basis index).

    Coefficients c_z = 2^-n sum_i values[i] (-1)^{|i & z|} by an in-place fast
    Walsh-Hadamard transform (n 2^n operations); exact zeros are dropped.
    n <= 12 (QM-E830 otherwise). Expectation values go through PauliOperator's
    pre-summed diagonal, so a dense diagonal costs one pass per evaluation.
    """
    var n = _log2_exact(len(values))
    if n < 0:
        raise Error("QM-E830: diagonal_observable() needs 2^n values, got " + String(len(values)) + ".")
    if n > DIAG_OBS_MAX_QUBITS:
        raise Error("QM-E830: diagonal_observable() supports at most " + String(DIAG_OBS_MAX_QUBITS) + " qubits (4096 values), got " + String(n) + ". Hint: use local observables (sums of few-qubit Pauli terms) on larger registers.")
    var w = values.copy()
    var dim = len(w)
    var h = 1
    while h < dim:
        var i = 0
        while i < dim:
            for j in range(i, i + h):
                var a = w[j]
                var b = w[j + h]
                w[j] = a + b
                w[j + h] = a - b
            i += 2 * h
        h *= 2
    var out = PauliSum()
    var inv = 1.0 / Float64(dim)
    for z in range(dim):
        var cz = w[z] * inv
        if cz != 0.0:
            out.terms.append(PauliTerm(cz, 0, UInt64(z)))
    return out^


def projector_basis(n: Int, k: Int) raises -> PauliSum:
    """|k><k| on n qubits (k = QUMO basis index) as a Z-string sum (n <= 12)."""
    if n < 1 or n > DIAG_OBS_MAX_QUBITS or k < 0 or k >= (1 << n):
        raise Error("QM-E830: projector_basis() needs 1 <= n <= " + String(DIAG_OBS_MAX_QUBITS) + " and 0 <= k < 2^n, got n=" + String(n) + ", k=" + String(k) + ".")
    var out = PauliSum()
    var c = 1.0 / Float64(1 << n)
    for z in range(1 << n):
        var s = c if (pop_count(k & z) & 1) == 0 else -c
        out.terms.append(PauliTerm(s, 0, UInt64(z)))
    return out^


def projector_zero(n: Int) raises -> PauliSum:
    """|0...0><0...0| on n qubits (the global cost / kernel observable), n <= 12."""
    return projector_basis(n, 0)


def local_projector_zero(n: Int) raises -> PauliSum:
    """(1/n) sum_j |0><0|_j = 1/2 + (1/(2n)) sum_j Z_j (local cost of
    Cerezo et al.; any n <= 64)."""
    if n < 1 or n > 64:
        raise Error("QM-E830: local_projector_zero() needs 1 <= n <= 64, got " + String(n) + ".")
    var out = PauliSum()
    out.terms.append(PauliTerm(0.5, 0, 0))
    for j in range(n):
        out.terms.append(PauliTerm(0.5 / Float64(n), 0, UInt64(1) << UInt64(j)))
    return out^
