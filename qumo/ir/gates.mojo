"""QUMO gate set: opcodes, static metadata and unitary matrices.

Every opcode carries machine-readable metadata (arity, #params, Clifford,
diagonal, self-inverse, inverse opcode) so compiler passes, simulators and AI
agents reason over the same single source of truth (see `qumo.ai.describe`).

Qubit-order convention (Qiskit-compatible, little-endian): basis index bit k
is qubit k. A 2-qubit matrix on (q0, q1) is indexed by b(q0) + 2*b(q1).
"""

from std.math import cos, sin, sqrt, atan2


# ----------------------------------------------------------------------------
# Opcodes
# ----------------------------------------------------------------------------
struct Op:
    # 1-qubit, fixed
    comptime I = 0
    comptime X = 1
    comptime Y = 2
    comptime Z = 3
    comptime H = 4
    comptime S = 5
    comptime SDG = 6
    comptime T = 7
    comptime TDG = 8
    comptime SX = 9
    comptime SXDG = 10
    # 1-qubit, parametric
    comptime RX = 11
    comptime RY = 12
    comptime RZ = 13
    comptime P = 14
    comptime U = 15
    # 2-qubit
    comptime CX = 16
    comptime CY = 17
    comptime CZ = 18
    comptime CH = 19
    comptime SWAP = 20
    comptime ISWAP = 21
    comptime ISWAPDG = 22
    comptime ECR = 23
    comptime CP = 24
    comptime CRX = 25
    comptime CRY = 26
    comptime CRZ = 27
    comptime RXX = 28
    comptime RYY = 29
    comptime RZZ = 30
    # 3-qubit
    comptime CCX = 31
    comptime CSWAP = 32
    comptime CCZ = 33
    # non-unitary / control flow
    comptime MEASURE = 40
    comptime RESET = 41
    comptime BARRIER = 42
    # noise channels (Stim-compatible semantics)
    comptime DEPOLARIZE1 = 50
    comptime DEPOLARIZE2 = 51
    comptime X_ERROR = 52
    comptime Y_ERROR = 53
    comptime Z_ERROR = 54
    comptime PAULI_CHANNEL_1 = 55
    # QEC annotations (no quantum effect)
    comptime DETECTOR = 60
    comptime OBSERVABLE = 61
    comptime TICK = 62

    comptime NUM_OPS = 63


def op_name(k: Int) -> String:
    if k == Op.I: return "id"
    if k == Op.X: return "x"
    if k == Op.Y: return "y"
    if k == Op.Z: return "z"
    if k == Op.H: return "h"
    if k == Op.S: return "s"
    if k == Op.SDG: return "sdg"
    if k == Op.T: return "t"
    if k == Op.TDG: return "tdg"
    if k == Op.SX: return "sx"
    if k == Op.SXDG: return "sxdg"
    if k == Op.RX: return "rx"
    if k == Op.RY: return "ry"
    if k == Op.RZ: return "rz"
    if k == Op.P: return "p"
    if k == Op.U: return "u"
    if k == Op.CX: return "cx"
    if k == Op.CY: return "cy"
    if k == Op.CZ: return "cz"
    if k == Op.CH: return "ch"
    if k == Op.SWAP: return "swap"
    if k == Op.ISWAP: return "iswap"
    if k == Op.ISWAPDG: return "iswapdg"
    if k == Op.ECR: return "ecr"
    if k == Op.CP: return "cp"
    if k == Op.CRX: return "crx"
    if k == Op.CRY: return "cry"
    if k == Op.CRZ: return "crz"
    if k == Op.RXX: return "rxx"
    if k == Op.RYY: return "ryy"
    if k == Op.RZZ: return "rzz"
    if k == Op.CCX: return "ccx"
    if k == Op.CSWAP: return "cswap"
    if k == Op.CCZ: return "ccz"
    if k == Op.MEASURE: return "measure"
    if k == Op.RESET: return "reset"
    if k == Op.BARRIER: return "barrier"
    if k == Op.DEPOLARIZE1: return "depolarize1"
    if k == Op.DEPOLARIZE2: return "depolarize2"
    if k == Op.X_ERROR: return "x_error"
    if k == Op.Y_ERROR: return "y_error"
    if k == Op.Z_ERROR: return "z_error"
    if k == Op.PAULI_CHANNEL_1: return "pauli_channel_1"
    if k == Op.DETECTOR: return "detector"
    if k == Op.OBSERVABLE: return "observable"
    if k == Op.TICK: return "tick"
    return "unknown"


def op_from_name(name: String) -> Int:
    """Opcode for a lowercase gate name (also accepts common aliases). -1 if unknown."""
    if name == "cnot": return Op.CX
    if name == "toffoli": return Op.CCX
    if name == "fredkin": return Op.CSWAP
    if name == "u3": return Op.U
    if name == "phase" or name == "u1": return Op.P
    if name == "cphase" or name == "cu1": return Op.CP
    if name == "i": return Op.I
    for k in range(Op.NUM_OPS):
        if op_name(k) == name:
            return k
    return -1


def op_arity(k: Int) -> Int:
    """Number of qubit operands (0 = variadic / annotation)."""
    if k <= Op.U: return 1
    if k <= Op.RZZ: return 2
    if k <= Op.CCZ: return 3
    if k == Op.MEASURE or k == Op.RESET: return 1
    if k == Op.DEPOLARIZE1 or k == Op.X_ERROR or k == Op.Y_ERROR or k == Op.Z_ERROR or k == Op.PAULI_CHANNEL_1:
        return 1
    if k == Op.DEPOLARIZE2: return 2
    return 0


def op_num_params(k: Int) -> Int:
    if k == Op.U or k == Op.PAULI_CHANNEL_1: return 3
    if (k >= Op.RX and k <= Op.P) or (k >= Op.CP and k <= Op.RZZ): return 1
    if k >= Op.DEPOLARIZE1 and k <= Op.Z_ERROR: return 1
    return 0


def op_is_unitary(k: Int) -> Bool:
    return k <= Op.CCZ


def op_is_noise(k: Int) -> Bool:
    return k >= Op.DEPOLARIZE1 and k <= Op.PAULI_CHANNEL_1


def op_is_annotation(k: Int) -> Bool:
    return k == Op.DETECTOR or k == Op.OBSERVABLE or k == Op.TICK or k == Op.BARRIER


def op_is_clifford(k: Int) -> Bool:
    """Clifford for every parameter value (parametric gates checked separately)."""
    if k <= Op.SXDG: return k != Op.T and k != Op.TDG
    if k >= Op.CX and k <= Op.ECR: return k != Op.CH
    return False


def op_is_diagonal(k: Int) -> Bool:
    return (
        k == Op.I or k == Op.Z or k == Op.S or k == Op.SDG or k == Op.T
        or k == Op.TDG or k == Op.RZ or k == Op.P or k == Op.CZ
        or k == Op.CP or k == Op.CRZ or k == Op.RZZ or k == Op.CCZ
    )


def op_is_self_inverse(k: Int) -> Bool:
    return (
        k == Op.I or k == Op.X or k == Op.Y or k == Op.Z or k == Op.H
        or k == Op.CX or k == Op.CY or k == Op.CZ or k == Op.CH
        or k == Op.SWAP or k == Op.ECR or k == Op.CCX or k == Op.CSWAP
        or k == Op.CCZ
    )


def op_inverse_kind(k: Int) -> Int:
    """Opcode of the inverse gate (parametric gates: same opcode, negated params)."""
    if k == Op.S: return Op.SDG
    if k == Op.SDG: return Op.S
    if k == Op.T: return Op.TDG
    if k == Op.TDG: return Op.T
    if k == Op.SX: return Op.SXDG
    if k == Op.SXDG: return Op.SX
    if k == Op.ISWAP: return Op.ISWAPDG
    if k == Op.ISWAPDG: return Op.ISWAP
    return k


def op_is_symmetric(k: Int) -> Bool:
    """2/3-qubit gates invariant under permutation of operands."""
    return (
        k == Op.CZ or k == Op.SWAP or k == Op.ISWAP or k == Op.ISWAPDG
        or k == Op.CP or k == Op.RXX or k == Op.RYY or k == Op.RZZ
        or k == Op.CCZ
    )


def op_is_controlled_1q(k: Int) -> Bool:
    """Gate = |0><0| (x) I + |1><1| (x) U with control q0, target q1."""
    return (
        k == Op.CX or k == Op.CY or k == Op.CZ or k == Op.CH or k == Op.CP
        or k == Op.CRX or k == Op.CRY or k == Op.CRZ
    )


def op_base_1q(k: Int) -> Int:
    """Target 1-qubit opcode of a controlled gate."""
    if k == Op.CX: return Op.X
    if k == Op.CY: return Op.Y
    if k == Op.CZ: return Op.Z
    if k == Op.CH: return Op.H
    if k == Op.CP: return Op.P
    if k == Op.CRX: return Op.RX
    if k == Op.CRY: return Op.RY
    if k == Op.CRZ: return Op.RZ
    return -1


def op_description(k: Int) -> String:
    """Human/LLM readable semantics of each opcode."""
    if k == Op.I: return "Identity."
    if k == Op.X: return "Pauli-X (NOT)."
    if k == Op.Y: return "Pauli-Y."
    if k == Op.Z: return "Pauli-Z (phase flip)."
    if k == Op.H: return "Hadamard: |0>->|+>, |1>->|->."
    if k == Op.S: return "Phase S = sqrt(Z) = diag(1, i)."
    if k == Op.SDG: return "S-dagger = diag(1, -i)."
    if k == Op.T: return "T = sqrt(S) = diag(1, e^{i pi/4}). Non-Clifford."
    if k == Op.TDG: return "T-dagger. Non-Clifford."
    if k == Op.SX: return "sqrt(X). IBM native."
    if k == Op.SXDG: return "sqrt(X)-dagger."
    if k == Op.RX: return "RX(theta) = exp(-i theta X/2)."
    if k == Op.RY: return "RY(theta) = exp(-i theta Y/2)."
    if k == Op.RZ: return "RZ(theta) = exp(-i theta Z/2)."
    if k == Op.P: return "Phase P(lambda) = diag(1, e^{i lambda})."
    if k == Op.U: return "U(theta,phi,lambda): generic 1q unitary (OpenQASM 3 convention)."
    if k == Op.CX: return "Controlled-X (CNOT). Operands: control, target."
    if k == Op.CY: return "Controlled-Y. Operands: control, target."
    if k == Op.CZ: return "Controlled-Z. Symmetric."
    if k == Op.CH: return "Controlled-Hadamard. Operands: control, target."
    if k == Op.SWAP: return "Swap two qubits."
    if k == Op.ISWAP: return "iSWAP: swap with i phase on |01>,|10>."
    if k == Op.ISWAPDG: return "Inverse of iSWAP."
    if k == Op.ECR: return "Echoed cross-resonance (IBM native 2q). Self-inverse."
    if k == Op.CP: return "Controlled phase CP(lambda). Symmetric."
    if k == Op.CRX: return "Controlled RX(theta)."
    if k == Op.CRY: return "Controlled RY(theta)."
    if k == Op.CRZ: return "Controlled RZ(theta)."
    if k == Op.RXX: return "RXX(theta) = exp(-i theta XX/2). Ion-trap native (MS)."
    if k == Op.RYY: return "RYY(theta) = exp(-i theta YY/2)."
    if k == Op.RZZ: return "RZZ(theta) = exp(-i theta ZZ/2). Diagonal."
    if k == Op.CCX: return "Toffoli. Operands: control, control, target."
    if k == Op.CSWAP: return "Fredkin. Operands: control, a, b."
    if k == Op.CCZ: return "Doubly-controlled Z. Symmetric."
    if k == Op.MEASURE: return "Z-basis measurement into a classical bit."
    if k == Op.RESET: return "Reset qubit to |0>."
    if k == Op.BARRIER: return "Optimization barrier (no effect)."
    if k == Op.DEPOLARIZE1: return "1q depolarizing: X,Y,Z each with prob p/3."
    if k == Op.DEPOLARIZE2: return "2q depolarizing: each of 15 non-identity Paulis with prob p/15."
    if k == Op.X_ERROR: return "Bit-flip with prob p."
    if k == Op.Y_ERROR: return "Y error with prob p."
    if k == Op.Z_ERROR: return "Phase-flip with prob p."
    if k == Op.PAULI_CHANNEL_1: return "Pauli channel with probs (px, py, pz)."
    if k == Op.DETECTOR: return "QEC detector: parity of measurement records, deterministic without noise."
    if k == Op.OBSERVABLE: return "QEC logical observable: parity of measurement records."
    if k == Op.TICK: return "Time-step boundary (scheduling hint)."
    return "Unknown opcode."


# ----------------------------------------------------------------------------
# Small dense complex matrices (split re/im storage, SIMD friendly)
# ----------------------------------------------------------------------------
comptime INV_SQRT2 = 0.70710678118654752440
comptime PI = 3.14159265358979323846


@fieldwise_init
struct M2(ImplicitlyCopyable, Writable):
    """Row-major 2x2 complex matrix: [m00, m01, m10, m11]."""

    var re: SIMD[DType.float64, 4]
    var im: SIMD[DType.float64, 4]

    @staticmethod
    def identity() -> M2:
        return M2(SIMD[DType.float64, 4](1, 0, 0, 1), SIMD[DType.float64, 4](0))

    def __matmul__(self, o: M2) -> M2:
        """self @ o."""
        var r = SIMD[DType.float64, 4](0)
        var i = SIMD[DType.float64, 4](0)
        comptime for row in range(2):
            comptime for col in range(2):
                var sr = 0.0
                var si = 0.0
                comptime for k in range(2):
                    var ar = self.re[row * 2 + k]
                    var ai = self.im[row * 2 + k]
                    var br = o.re[k * 2 + col]
                    var bi = o.im[k * 2 + col]
                    sr += ar * br - ai * bi
                    si += ar * bi + ai * br
                r[row * 2 + col] = sr
                i[row * 2 + col] = si
        return M2(r, i)

    def dagger(self) -> M2:
        return M2(
            SIMD[DType.float64, 4](self.re[0], self.re[2], self.re[1], self.re[3]),
            SIMD[DType.float64, 4](-self.im[0], -self.im[2], -self.im[1], -self.im[3]),
        )

    def is_diagonal(self, tol: Float64 = 1e-12) -> Bool:
        return (
            abs(self.re[1]) < tol and abs(self.im[1]) < tol
            and abs(self.re[2]) < tol and abs(self.im[2]) < tol
        )

    def write_to(self, mut w: Some[Writer]):
        w.write("[[", self.re[0], "+", self.im[0], "j, ", self.re[1], "+", self.im[1], "j], [")
        w.write(self.re[2], "+", self.im[2], "j, ", self.re[3], "+", self.im[3], "j]]")


def _m2(r00: Float64, i00: Float64, r01: Float64, i01: Float64,
        r10: Float64, i10: Float64, r11: Float64, i11: Float64) -> M2:
    return M2(
        SIMD[DType.float64, 4](r00, r01, r10, r11),
        SIMD[DType.float64, 4](i00, i01, i10, i11),
    )


def u3_matrix(theta: Float64, phi: Float64, lam: Float64) -> M2:
    var c = cos(theta / 2)
    var s = sin(theta / 2)
    return _m2(
        c, 0.0,
        -cos(lam) * s, -sin(lam) * s,
        cos(phi) * s, sin(phi) * s,
        cos(phi + lam) * c, sin(phi + lam) * c,
    )


def matrix_1q(k: Int, a: Float64 = 0, b: Float64 = 0, c: Float64 = 0) -> M2:
    """Unitary of a 1-qubit opcode with resolved parameters."""
    if k == Op.I: return M2.identity()
    if k == Op.X: return _m2(0, 0, 1, 0, 1, 0, 0, 0)
    if k == Op.Y: return _m2(0, 0, 0, -1, 0, 1, 0, 0)
    if k == Op.Z: return _m2(1, 0, 0, 0, 0, 0, -1, 0)
    if k == Op.H: return _m2(INV_SQRT2, 0, INV_SQRT2, 0, INV_SQRT2, 0, -INV_SQRT2, 0)
    if k == Op.S: return _m2(1, 0, 0, 0, 0, 0, 0, 1)
    if k == Op.SDG: return _m2(1, 0, 0, 0, 0, 0, 0, -1)
    if k == Op.T: return _m2(1, 0, 0, 0, 0, 0, INV_SQRT2, INV_SQRT2)
    if k == Op.TDG: return _m2(1, 0, 0, 0, 0, 0, INV_SQRT2, -INV_SQRT2)
    if k == Op.SX: return _m2(0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5)
    if k == Op.SXDG: return _m2(0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5)
    if k == Op.RX:
        var co = cos(a / 2)
        var si = sin(a / 2)
        return _m2(co, 0, 0, -si, 0, -si, co, 0)
    if k == Op.RY:
        var co = cos(a / 2)
        var si = sin(a / 2)
        return _m2(co, 0, -si, 0, si, 0, co, 0)
    if k == Op.RZ:
        return _m2(cos(a / 2), -sin(a / 2), 0, 0, 0, 0, cos(a / 2), sin(a / 2))
    if k == Op.P:
        return _m2(1, 0, 0, 0, 0, 0, cos(a), sin(a))
    if k == Op.U:
        return u3_matrix(a, b, c)
    return M2.identity()


@fieldwise_init
struct ZYZ(ImplicitlyCopyable, Writable):
    """U = e^{i phase} U3(theta, phi, lam)."""

    var theta: Float64
    var phi: Float64
    var lam: Float64
    var phase: Float64

    def write_to(self, mut w: Some[Writer]):
        w.write("ZYZ(", self.theta, ", ", self.phi, ", ", self.lam, ", phase=", self.phase, ")")


def _arg(re: Float64, im: Float64) -> Float64:
    return atan2(im, re)


def zyz_decompose(m: M2) -> ZYZ:
    """Euler decomposition of any 2x2 unitary into U3 angles + global phase."""
    # det = m00*m11 - m01*m10
    var dr = m.re[0] * m.re[3] - m.im[0] * m.im[3] - (m.re[1] * m.re[2] - m.im[1] * m.im[2])
    var di = m.re[0] * m.im[3] + m.im[0] * m.re[3] - (m.re[1] * m.im[2] + m.im[1] * m.re[2])
    var ph = _arg(dr, di) / 2
    var cr = cos(-ph)
    var ci = sin(-ph)
    # V = e^{-i ph} m  (special unitary)
    var vr = m.re * cr - m.im * ci
    var vi = m.re * ci + m.im * cr
    var a00 = sqrt(vr[0] * vr[0] + vi[0] * vi[0])
    var a10 = sqrt(vr[2] * vr[2] + vi[2] * vi[2])
    var theta = 2 * atan2(a10, a00)
    var ssum = 0.0
    var sdiff = 0.0
    if a00 > 1e-12:
        ssum = 2 * _arg(vr[3], vi[3])
    if a10 > 1e-12:
        sdiff = 2 * _arg(vr[2], vi[2])
    var phi = (ssum + sdiff) / 2
    var lam = (ssum - sdiff) / 2
    return ZYZ(theta, phi, lam, ph - (phi + lam) / 2)


def m2_approx_eq_up_to_phase(a: M2, b: M2, tol: Float64 = 1e-9) -> Bool:
    """True when a = e^{i g} b for some global phase g."""
    # find largest entry of b to fix the phase
    var best = 0
    var bm = 0.0
    for k in range(4):
        var v = b.re[k] * b.re[k] + b.im[k] * b.im[k]
        if v > bm:
            bm = v
            best = k
    # g = a_k / b_k
    var denom = bm
    var gr = (a.re[best] * b.re[best] + a.im[best] * b.im[best]) / denom
    var gi = (a.im[best] * b.re[best] - a.re[best] * b.im[best]) / denom
    for k in range(4):
        var er = gr * b.re[k] - gi * b.im[k] - a.re[k]
        var ei = gr * b.im[k] + gi * b.re[k] - a.im[k]
        if abs(er) > tol or abs(ei) > tol:
            return False
    return True


struct M4(ImplicitlyCopyable):
    """Row-major 4x4 complex matrix, index = b(q0) + 2*b(q1)."""

    var re: SIMD[DType.float64, 16]
    var im: SIMD[DType.float64, 16]

    def __init__(out self):
        self.re = SIMD[DType.float64, 16](0)
        self.im = SIMD[DType.float64, 16](0)

    @staticmethod
    def identity() -> M4:
        var m = M4()
        comptime for k in range(4):
            m.re[k * 5] = 1
        return m^

    def set(mut self, r: Int, c: Int, re: Float64, im: Float64 = 0):
        self.re[r * 4 + c] = re
        self.im[r * 4 + c] = im

    def __matmul__(self, o: M4) -> M4:
        var out = M4()
        for r in range(4):
            for c in range(4):
                var sr = 0.0
                var si = 0.0
                for k in range(4):
                    var ar = self.re[r * 4 + k]
                    var ai = self.im[r * 4 + k]
                    var br = o.re[k * 4 + c]
                    var bi = o.im[k * 4 + c]
                    sr += ar * br - ai * bi
                    si += ar * bi + ai * br
                out.re[r * 4 + c] = sr
                out.im[r * 4 + c] = si
        return out^

    @staticmethod
    def from_1q(m: M2, on_q1: Bool) -> M4:
        """Embed a 1q matrix acting on q0 (on_q1=False) or q1 (on_q1=True)."""
        var out = M4()
        for r in range(4):
            for c in range(4):
                var r0 = r & 1
                var r1 = r >> 1
                var c0 = c & 1
                var c1 = c >> 1
                if on_q1:
                    if r0 == c0:
                        out.set(r, c, m.re[r1 * 2 + c1], m.im[r1 * 2 + c1])
                else:
                    if r1 == c1:
                        out.set(r, c, m.re[r0 * 2 + c0], m.im[r0 * 2 + c0])
        return out^


def matrix_2q(k: Int, a: Float64 = 0) -> M4:
    """Unitary of a 2-qubit opcode, operand order (q0, q1); q0 = control."""
    var m = M4()
    if op_is_controlled_1q(k):
        var u = matrix_1q(op_base_1q(k), a)
        # control = q0 (bit 0). Rows/cols with b0 = 0 -> identity.
        m.set(0, 0, 1)
        m.set(2, 2, 1)
        # indices with b0=1: 1 (b1=0), 3 (b1=1)
        m.set(1, 1, u.re[0], u.im[0])
        m.set(1, 3, u.re[1], u.im[1])
        m.set(3, 1, u.re[2], u.im[2])
        m.set(3, 3, u.re[3], u.im[3])
        return m^
    if k == Op.SWAP:
        m.set(0, 0, 1); m.set(1, 2, 1); m.set(2, 1, 1); m.set(3, 3, 1)
        return m^
    if k == Op.ISWAP or k == Op.ISWAPDG:
        var s = 1.0 if k == Op.ISWAP else -1.0
        m.set(0, 0, 1); m.set(1, 2, 0, s); m.set(2, 1, 0, s); m.set(3, 3, 1)
        return m^
    if k == Op.ECR:
        var r = INV_SQRT2
        m.set(0, 1, r); m.set(0, 3, 0, r)
        m.set(1, 0, r); m.set(1, 2, 0, -r)
        m.set(2, 1, 0, r); m.set(2, 3, r)
        m.set(3, 0, 0, -r); m.set(3, 2, r)
        return m^
    var c = cos(a / 2)
    var s = sin(a / 2)
    if k == Op.RZZ:
        m.set(0, 0, c, -s); m.set(1, 1, c, s); m.set(2, 2, c, s); m.set(3, 3, c, -s)
        return m^
    if k == Op.RXX:
        for i in range(4):
            m.set(i, i, c)
        m.set(0, 3, 0, -s); m.set(1, 2, 0, -s); m.set(2, 1, 0, -s); m.set(3, 0, 0, -s)
        return m^
    if k == Op.RYY:
        for i in range(4):
            m.set(i, i, c)
        # YY = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]; -i s YY
        m.set(0, 3, 0, s); m.set(1, 2, 0, -s); m.set(2, 1, 0, -s); m.set(3, 0, 0, s)
        return m^
    return M4.identity()
