"""Physical primitives (gadgets) of the Steane and Iceberg architectures.

Every gadget is a small `Circuit` template on local qubits - block 0 is
qubits 0..n-1, block 1 is n..2n-1, ancillas follow - with flag/verification
measurements written to local clbits 0, 1, ... (a flag that reads 1 rejects
the attempt). The encoder (`qumo.ft.implement`) maps local qubits to physical
ones and adds classical conditions. Keeping gadgets as plain circuits means
each one is verified on its own by `qumo.ft.verify` (tests/test_ft_verify.mojo
and `verify_gadget_library`), exactly as Guppy FT recommends for primitives.

Steane [[7,1,3]] (based on arXiv:2107.07505, as Guppy FT):
  prep_zero (non-FT encoder, 8 CX), prep_zero_ft (+1 flag measuring a weight-3
  logical Z representative), logical X/Y/Z/H/CX/CZ transversal, logical S =
  transversal SDG and S^dag = transversal S (Z_L = Z^7 has weight 7 = 3 mod 4),
  prep_t_nonft (|H> = Ry(pi/4)|0> encoder of 10.1038/srep19578 Fig. 3b, then
  logical S^dag H -> T|+>), prep_t_ft (+ fault-tolerant measurement of the
  logical H operator with a 2-qubit cat ancilla, and the flagged syndrome
  extraction of Reichardt arXiv:1804.06995 Fig. 5: 8 flags; ancillas are
  reused with resets so the gadget needs 3 ancillas, not 8).

Iceberg [[k+2,k,2]] (layout [t, b, d_0..d_{k-1}], arXiv:2211.06703):
  prep_zero (H on t, fan-out CX), X_i = X_t X_i, Z_i = Z_b Z_i,
  Y_i = X_t Z_b Y_i, RZ_i(a) = RZZ(a) on (b, d_i), RX_i(a) = RXX(a) on
  (t, d_i), RZZ_ij(a) = RZZ(a) on (d_i, d_j), RXX_ij(a) = RXX(a) on (d_i, d_j),
  syndrome cycle (measure X^n and Z^n with two ancillas: 2 detection flags).

Idea from Guppy FT (Apache-2.0), files src/guppyft/code/steane/primitives.py
and src/guppyft/code/iceberg/logical.py; gate sequences re-derived from the
cited papers and checked by the verifier, not copied.
"""

from std.math import pi, cos, sin, sqrt

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.pauli import PauliSum, PauliTerm
from qumo.sim.statevector import StateVector
from qumo.sim.runner import apply_instruction
from qumo.sim.rng import Rng
from qumo.ft.code import StabilizerCode
from qumo.ft.pauli import SymPauli
from qumo.ft.verify import (
    VerifyReport, valid_clifford_implementation, valid_stabilizer_state_preparation,
)


# ---------------------------------------------------------------------------
# Steane
# ---------------------------------------------------------------------------
def steane_encoder_zero() -> Circuit:
    """Non-FT |0>_L: H on {0, 4, 6} then 8 CX (stabilizers of STEANE_DEF)."""
    var c = Circuit(7, 0, "steane_prep_zero")
    c.h(0)
    c.h(4)
    c.h(6)
    var pairs: List[Int] = [0, 1, 4, 5, 6, 3, 6, 5, 4, 2, 0, 3, 4, 1, 3, 2]
    for i in range(8):
        c.cx(pairs[2 * i], pairs[2 * i + 1])
    return c^


def steane_prep_zero_ft() -> Circuit:
    """Encoder + one flag ancilla (qubit 7) measuring Z1 Z3 Z5, a weight-3
    representative of Z_L: a single fault that would flip Z_L fires the flag."""
    var c = Circuit(8, 1, "steane_prep_zero_ft")
    c.compose(steane_encoder_zero())
    c.cx(1, 7)
    c.cx(3, 7)
    c.cx(5, 7)
    c.measure(7, 0)
    return c^


def transversal_1q(kind: Int, n: Int) -> Circuit:
    var c = Circuit(n)
    for q in range(n):
        c.add(kind, q)
    return c^


def transversal_2q(kind: Int, n: Int) -> Circuit:
    var c = Circuit(2 * n)
    for q in range(n):
        c.add(kind, q, n + q)
    return c^


def steane_physical_kind(logical_kind_name: String) -> Int:
    """Physical gate of the transversal implementation of a logical gate."""
    if logical_kind_name == "s":
        return Op.SDG
    if logical_kind_name == "sdg":
        return Op.S
    if logical_kind_name == "x":
        return Op.X
    if logical_kind_name == "y":
        return Op.Y
    if logical_kind_name == "z":
        return Op.Z
    if logical_kind_name == "h":
        return Op.H
    if logical_kind_name == "cx":
        return Op.CX
    if logical_kind_name == "cz":
        return Op.CZ
    return -1


def _steane_relabel() -> List[Int]:
    # Fig. 3b of 10.1038/srep19578, rows top to bottom -> STEANE_DEF qubits
    var r: List[Int] = [1, 0, 4, 5, 2, 6, 3]
    return r^


def steane_prep_h_nonft() -> Circuit:
    """Non-FT |H>_L = Ry(pi/4)|0>_L on a Steane block."""
    var r = _steane_relabel()
    var c = Circuit(7, 0, "steane_prep_h")
    c.ry(pi / 4, r[0])
    c.h(r[1])
    c.h(r[2])
    c.h(r[5])
    var pairs: List[Int] = [0, 6, 0, 3, 1, 0, 5, 4, 2, 3, 1, 6, 2, 4, 5, 3, 1, 4, 2, 0, 5, 6]
    for i in range(11):
        c.cx(r[pairs[2 * i]], r[pairs[2 * i + 1]])
    return c^


def _logical_sdg_h(mut c: Circuit, off: Int):
    # logical S^dag = transversal S, then logical H = transversal H: |H> -> T|+>
    for q in range(7):
        c.s(off + q)
    for q in range(7):
        c.h(off + q)


def steane_prep_t_nonft() -> Circuit:
    """T|+>_L (magic state for T injection) without verification."""
    var c = Circuit(7, 0, "steane_prep_t_nonft")
    c.compose(steane_prep_h_nonft())
    _logical_sdg_h(c, 0)
    return c^


def _controlled_h(mut c: Circuit, ctl: Int, tgt: Int):
    """CH = Ry(pi/4) CZ Ry(-pi/4) on the target (operator order right to left)."""
    c.ry(-pi / 4, tgt)
    c.cz(ctl, tgt)
    c.ry(pi / 4, tgt)


def _syndrome_cx3(mut c: Circuit, a0: Int, a1: Int, a2: Int, d0: Int, d1: Int, d2: Int, reverse: Bool):
    if reverse:
        c.cx(d0, a0)
        c.cx(a1, d1)
        c.cx(a2, d2)
    else:
        c.cx(a0, d0)
        c.cx(d1, a1)
        c.cx(d2, a2)


def steane_flagged_syndromes(mut c: Circuit, off: Int, a0: Int, a1: Int, a2: Int, clbase: Int):
    """Reichardt arXiv:1804.06995 Fig. 5: two rounds of 3 ancillas measure all
    6 Steane stabilizers with flags (6 measurements, all 0 on a codeword)."""
    var rl: List[Int] = [0, 4, 1, 6, 3, 5, 2]
    # X Z Z round
    c.h(a0)
    _syndrome_cx3(c, a0, a1, a2, off + rl[4], off + rl[6], off + rl[5], False)
    c.cx(a0, a2)
    _syndrome_cx3(c, a0, a1, a2, off + rl[0], off + rl[4], off + rl[1], False)
    _syndrome_cx3(c, a0, a1, a2, off + rl[2], off + rl[3], off + rl[6], False)
    c.cx(a0, a1)
    _syndrome_cx3(c, a0, a1, a2, off + rl[6], off + rl[5], off + rl[2], False)
    c.h(a0)
    c.measure(a0, clbase)
    c.measure(a1, clbase + 1)
    c.measure(a2, clbase + 2)
    c.reset(a0)
    c.reset(a1)
    c.reset(a2)
    # Z X X round
    c.h(a1)
    c.h(a2)
    _syndrome_cx3(c, a0, a1, a2, off + rl[4], off + rl[6], off + rl[5], True)
    c.cx(a2, a0)
    _syndrome_cx3(c, a0, a1, a2, off + rl[0], off + rl[4], off + rl[1], True)
    _syndrome_cx3(c, a0, a1, a2, off + rl[2], off + rl[3], off + rl[6], True)
    c.cx(a1, a0)
    _syndrome_cx3(c, a0, a1, a2, off + rl[6], off + rl[5], off + rl[2], True)
    c.h(a1)
    c.h(a2)
    c.measure(a0, clbase + 3)
    c.measure(a1, clbase + 4)
    c.measure(a2, clbase + 5)


def steane_prep_t_ft() -> Circuit:
    """Flag-verified T|+>_L: |H> encoder, FT measurement of the logical H
    operator (cat ancilla, controlled-H alternating controls), flagged syndrome
    extraction, then logical S^dag H. Qubits 7..9 are ancillas; 8 flags."""
    var c = Circuit(10, 8, "steane_prep_t_ft")
    c.compose(steane_prep_h_nonft())
    # measure logical H with a 2-qubit cat state
    c.h(7)
    c.cx(7, 8)
    for t in range(7):
        _controlled_h(c, 7 + (t % 2), t)
    c.cx(7, 8)
    c.h(7)
    c.measure(7, 0)
    c.measure(8, 1)
    c.reset(7)
    c.reset(8)
    steane_flagged_syndromes(c, 0, 7, 8, 9, 2)
    _logical_sdg_h(c, 0)
    return c^


# ---------------------------------------------------------------------------
# Iceberg
# ---------------------------------------------------------------------------
def iceberg_prep_zero(n: Int) -> Circuit:
    """|0..0>_L = (|0^n> + |1^n>)/sqrt(2): H on t, CX fan-out."""
    var c = Circuit(n, 0, "iceberg_prep_zero")
    c.h(0)
    for q in range(1, n):
        c.cx(0, q)
    return c^


def iceberg_pauli(n: Int, slot: Int, p: String) -> Circuit:
    var c = Circuit(n)
    var d = 2 + slot
    if p == "X":
        c.x(0)
        c.x(d)
    elif p == "Z":
        c.z(1)
        c.z(d)
    else:  # Y_i = X_t Z_b Y_i
        c.x(0)
        c.z(1)
        c.y(d)
    return c^


def iceberg_rotation(n: Int, kind: String, slot: Int, slot2: Int, theta: Float64) -> Circuit:
    """kind in rz, rx (slot), rzz, rxx (slot, slot2) - all within one block."""
    var c = Circuit(n)
    if kind == "rz":
        c.rzz(theta, 1, 2 + slot)
    elif kind == "rx":
        c.rxx(theta, 0, 2 + slot)
    elif kind == "rzz":
        c.rzz(theta, 2 + slot, 2 + slot2)
    else:
        c.rxx(theta, 2 + slot, 2 + slot2)
    return c^


def iceberg_syndrome_cycle(n: Int) -> Circuit:
    """Measure S_Z = Z^n (ancilla n) and S_X = X^n (ancilla n+1): 2 flags."""
    var c = Circuit(n + 2, 2, "iceberg_syndrome")
    for q in range(n):
        c.cx(q, n)
    c.h(n + 1)
    for q in range(n):
        c.cx(n + 1, q)
    c.h(n + 1)
    c.measure(n, 0)
    c.measure(n + 1, 1)
    return c^


# ---------------------------------------------------------------------------
# Magic-state verification (state vector)
# ---------------------------------------------------------------------------
def _project_zero(mut sv: StateVector, q: Int) -> Float64:
    """Project qubit q on |0> and renormalise; returns the probability."""
    var bit = 1 << q
    var p0 = 0.0
    for i in range(sv.size()):
        if (i & bit) == 0:
            p0 += sv.re[i] * sv.re[i] + sv.im[i] * sv.im[i]
    var s = 1.0 / sqrt(p0) if p0 > 0 else 0.0
    for i in range(sv.size()):
        if (i & bit) == 0:
            sv.re[i] *= s
            sv.im[i] *= s
        else:
            sv.re[i] = 0.0
            sv.im[i] = 0.0
    return p0


def postselected_state(c: Circuit) raises -> StateVector:
    """Run `c` on a state vector, projecting every measurement on outcome 0
    (the accept branch of flag measurements); resets act on |0>."""
    var sv = StateVector(c.num_qubits)
    var rng = Rng(7)
    var empty = List[Float64]()
    for g in c.ops:
        if g.cond >= 0:
            raise Error("QM-E723: postselected_state() needs an unconditioned gadget. Hint: verify conditioned gadgets branch by branch.")
        if g.kind == Op.MEASURE:
            var p = _project_zero(sv, g.q0)
            if p < 1e-12:
                raise Error("QM-E723: flag measurement on qubit " + String(g.q0) + " never accepts (probability of 0 is " + String(p) + ").")
        elif g.kind == Op.RESET:
            sv.reset_qubit(g.q0, rng)
        elif g.kind == Op.BARRIER or g.kind == Op.TICK:
            continue
        else:
            apply_instruction(sv, g, empty)
    return sv^


def sv_pauli_expectation(sv: StateVector, p: SymPauli) raises -> Float64:
    if p.n > 64:
        raise Error("QM-E723: state-vector verification is limited to 64 qubits.")
    var x: UInt64 = 0
    var z: UInt64 = 0
    for q in range(p.n):
        if p.x(q) == 1:
            x |= UInt64(1) << UInt64(q)
        if p.z(q) == 1:
            z |= UInt64(1) << UInt64(q)
    return sv.expectation_term(x, z) * Float64(p.sign())


def valid_magic_state_preparation(impl: Circuit, code: StabilizerCode, theta: Float64 = pi / 4,
                                  tol: Float64 = 1e-9) raises -> VerifyReport:
    """`impl` prepares (on its first n qubits, accept branch of every flag)
    the logical state cos-sin rotated (|0> + e^{i theta}|1>)/sqrt(2), i.e.
    Rz(theta)|+>_L: every stabilizer has <g> = +1 and the logical Bloch vector
    is (cos theta, sin theta, 0)."""
    if code.k != 1:
        raise Error("QM-E723: valid_magic_state_preparation supports k = 1 codes.")
    var sv = postselected_state(impl)
    var n = impl.num_qubits
    var checked = 0
    for g in code.generators:
        var e = sv_pauli_expectation(sv, g.embed(0, n))
        checked += 1
        if abs(e - 1.0) > tol:
            return VerifyReport(False, "stabilizer " + g.label() + " has expectation " + String(e) + " (state left the code space).", checked)
    var ex = sv_pauli_expectation(sv, code.logical(0, "X").embed(0, n))
    var ey = sv_pauli_expectation(sv, code.logical(0, "Y").embed(0, n))
    var ez = sv_pauli_expectation(sv, code.logical(0, "Z").embed(0, n))
    checked += 3
    if abs(ex - cos(theta)) > tol or abs(ey - sin(theta)) > tol or abs(ez) > tol:
        return VerifyReport(False, "logical Bloch vector (" + String(ex) + ", " + String(ey) + ", " + String(ez) + ") != (cos theta, sin theta, 0).", checked)
    return VerifyReport(True, "", checked)


# ---------------------------------------------------------------------------
# Library self-check
# ---------------------------------------------------------------------------
struct GadgetCheck(Copyable, Writable):
    var name: String
    var report: VerifyReport

    def __init__(out self, name: String, var report: VerifyReport):
        self.name = name
        self.report = report^

    def write_to(self, mut w: Some[Writer]):
        w.write(self.name, ": ", self.report)


def _sem1(kind: Int) -> Circuit:
    var c = Circuit(1)
    c.add(kind, 0)
    return c^


def verify_steane_library(code: StabilizerCode, include_ft_magic: Bool = True) raises -> List[GadgetCheck]:
    var out = List[GadgetCheck]()
    var names: List[String] = ["x", "y", "z", "h", "s", "sdg"]
    var kinds: List[Int] = [Op.X, Op.Y, Op.Z, Op.H, Op.S, Op.SDG]
    for i in range(len(names)):
        out.append(GadgetCheck("logical " + names[i], valid_clifford_implementation(_sem1(kinds[i]), transversal_1q(steane_physical_kind(names[i]), 7), code)))
    var scx = Circuit(2)
    scx.cx(0, 1)
    out.append(GadgetCheck("logical cx", valid_clifford_implementation(scx, transversal_2q(Op.CX, 7), code, 2)))
    var scz = Circuit(2)
    scz.cz(0, 1)
    out.append(GadgetCheck("logical cz", valid_clifford_implementation(scz, transversal_2q(Op.CZ, 7), code, 2)))
    var s0 = Circuit(1)
    out.append(GadgetCheck("prep_zero", valid_stabilizer_state_preparation(s0, steane_encoder_zero(), code)))
    out.append(GadgetCheck("prep_zero_ft", valid_stabilizer_state_preparation(s0, steane_prep_zero_ft(), code, 1, 1)))
    out.append(GadgetCheck("prep_t_nonft", valid_magic_state_preparation(steane_prep_t_nonft(), code)))
    if include_ft_magic:
        out.append(GadgetCheck("prep_t_ft", valid_magic_state_preparation(steane_prep_t_ft(), code)))
    return out^


def verify_iceberg_library(code: StabilizerCode) raises -> List[GadgetCheck]:
    var out = List[GadgetCheck]()
    var n = code.n
    var k = code.k
    var s0 = Circuit(k)
    out.append(GadgetCheck("prep_zero", valid_stabilizer_state_preparation(s0, iceberg_prep_zero(n), code)))
    var ps: List[String] = ["X", "Y", "Z"]
    for i in range(k):
        for p in ps:
            var sem = Circuit(k)
            sem.add(Op.X if p == "X" else (Op.Y if p == "Y" else Op.Z), i)
            out.append(GadgetCheck("logical " + p.lower() + "[" + String(i) + "]", valid_clifford_implementation(sem, iceberg_pauli(n, i, p), code)))
        # Clifford-angle rotations exercise the RZZ / RXX gadgets exactly
        var a = pi / 2
        var sz = Circuit(k)
        sz.rz(a, i)
        out.append(GadgetCheck("logical rz(pi/2)[" + String(i) + "]", valid_clifford_implementation(sz, iceberg_rotation(n, "rz", i, -1, a), code)))
        var sx = Circuit(k)
        sx.rx(a, i)
        out.append(GadgetCheck("logical rx(pi/2)[" + String(i) + "]", valid_clifford_implementation(sx, iceberg_rotation(n, "rx", i, -1, a), code)))
        if i + 1 < k:
            var szz = Circuit(k)
            szz.rzz(a, i, i + 1)
            out.append(GadgetCheck("logical rzz(pi/2)[" + String(i) + "," + String(i + 1) + "]", valid_clifford_implementation(szz, iceberg_rotation(n, "rzz", i, i + 1, a), code)))
            var sxx = Circuit(k)
            sxx.rxx(a, i, i + 1)
            out.append(GadgetCheck("logical rxx(pi/2)[" + String(i) + "," + String(i + 1) + "]", valid_clifford_implementation(sxx, iceberg_rotation(n, "rxx", i, i + 1, a), code)))
    # transversal H + free swap of t and b = logical H on every qubit
    var sh = Circuit(k)
    for i in range(k):
        sh.h(i)
    var ih = transversal_1q(Op.H, n)
    ih.swap(0, 1)
    out.append(GadgetCheck("logical h (all, + relabel t<->b)", valid_clifford_implementation(sh, ih, code)))
    var scx = Circuit(2 * k)
    for i in range(k):
        scx.cx(i, k + i)
    out.append(GadgetCheck("logical cx (transversal, all pairs)", valid_clifford_implementation(scx, transversal_2q(Op.CX, n), code, 2)))
    return out^
