"""OpenQASM 2.0 emitter (qelib1.inc dialect) for interoperability.

OpenQASM 3 (`qumo.ir.qasm.to_qasm`) is QUMO's primary text format. OpenQASM 2
is still the format that every SDK parser accepts (Qiskit `qasm2`, pytket,
Cirq, Braket, staq), so benchmark harnesses, hardware plugins and legacy tools
use this emitter.

Portability contract (tests/test_qasm2_export.mojo checks it in Mojo; the
cross-check against Qiskit and pytket is described in docs/AI_NATIVE.md):

* Without a definition, the output only uses the gates of the *original*
  qelib1.inc from the OpenQASM 2.0 paper (Cross et al., 2017):
  `u3 u2 u1 cx id u0 x y z h s sdg t tdg rx ry rz cz cy ch ccx crz cu1 cu3`.
  That is the gate set that `qiskit.qasm2.loads` accepts in its strict
  default mode. (Qiskit's `LEGACY_CUSTOM_INSTRUCTIONS` and pytket know more
  gates, but a file that relies on them does not load in strict mode.)
* Every other QUMO gate (`sx sxdg swap cswap crx cry rxx ryy rzz iswap
  iswapdg ecr ccz`) is emitted with a `gate` definition before its first use.
  Definition bodies call only the qelib1 gates above, so they never depend on
  another definition. (An earlier version defined `ecr` through `rzx`, which
  is in no include file, and every parser rejected it.)
* Bodies are exact unitaries, including global phase, under the semantics
  that Qiskit, pytket, Cirq and QUMO all give to qelib1 gates (`rz(t)` is
  exp(-i t Z / 2)). So the result is the same unitary whether a parser inlines
  the definition (Qiskit strict mode, pytket `CustomGate`) or substitutes its
  own class for a known name (Qiskit legacy mode: `SXGate`, `SwapGate`,
  `RZZGate`, ...).
* Standard names are kept (`ecr`, `iswap`, `rzz`, ...), so gate counts by
  name and basis checks keep working downstream.
* `u` is written as `u3`, `p` as `u1` and `cp` as `cu1` (identical matrices).

Limits: parameters must be bound (QM-E102). OpenQASM 2 conditions compare a
whole classical register, so a single-bit condition is exported only when
the register has one bit, and a register condition only when it covers every
clbit (QM-E552 otherwise; OpenQASM 3 has no such limit). The global phase is
written as a comment, because OpenQASM 2 cannot express it. Noise channels
and QEC annotations become comments.
"""

from qumo.ir.circuit import Circuit
from qumo.ir.qasm import register_condition_value, format_angle
from qumo.ir.gates import Op, op_name, op_num_params, op_arity, op_is_annotation, op_is_noise


def qasm2_real(v: Float64) -> String:
    """Angle text that is a valid OpenQASM 2 expression: `pi/2`, `-3*pi/4`,
    `2`, `0.3` or `1.0e-07`. (The spec's `real` needs a '.' before the exponent,
    so `1e-07` becomes `1.0e-07`.)"""
    var s = format_angle(v)
    if s.find(".") < 0:
        var e = s.find("e")
        if e >= 0:
            return String(s[byte=0:e]) + ".0" + String(s[byte=e:])
    return s^


def qelib1_gates() -> List[String]:
    """Gates of the original qelib1.inc (OpenQASM 2.0 paper), plus the
    built-ins `U` and `CX`."""
    var g: List[String] = [
        "U", "CX", "u3", "u2", "u1", "cx", "id", "u0", "x", "y", "z", "h", "s", "sdg",
        "t", "tdg", "rx", "ry", "rz", "cz", "cy", "ch", "ccx", "crz", "cu1", "cu3",
    ]
    return g^


def qasm2_gate_name(k: Int) -> String:
    """Name used in OpenQASM 2 output for opcode `k`."""
    if k == Op.U:
        return "u3"
    if k == Op.P:
        return "u1"
    if k == Op.CP:
        return "cu1"
    return op_name(k)


def qasm2_gate_definition(k: Int) -> String:
    """Exact definition for a gate outside the original qelib1.inc, or "" for
    a qelib1 gate. Bodies only call qelib1 gates."""
    if k == Op.SX:
        return "gate sx a { h a; s a; h a; }"
    if k == Op.SXDG:
        return "gate sxdg a { h a; sdg a; h a; }"
    if k == Op.SWAP:
        return "gate swap a,b { cx a,b; cx b,a; cx a,b; }"
    if k == Op.ISWAP:
        return "gate iswap a,b { s a; s b; h a; cx a,b; cx b,a; h b; }"
    if k == Op.ISWAPDG:
        return "gate iswapdg a,b { h b; cx b,a; cx a,b; h a; sdg b; sdg a; }"
    if k == Op.ECR:
        # ECR = e^{-i pi/4} X_a CX_ab (S_a (x) SX_b), and e^{-i pi/4} S = RZ(pi/2).
        return "gate ecr a,b { rz(pi/2) a; h b; s b; h b; cx a,b; x a; }"
    if k == Op.CRX:
        # H CRZ(theta) H on the target, with the u1/cx form of CRZ (exact)
        return "gate crx(theta) a,b { h b; u1(theta/2) b; cx a,b; u1(-theta/2) b; cx a,b; h b; }"
    if k == Op.CRY:
        return "gate cry(theta) a,b { ry(theta/2) b; cx a,b; ry(-theta/2) b; cx a,b; }"
    if k == Op.RXX:
        return "gate rxx(theta) a,b { h a; h b; cx a,b; rz(theta) b; cx a,b; h a; h b; }"
    if k == Op.RYY:
        return "gate ryy(theta) a,b { rx(pi/2) a; rx(pi/2) b; cx a,b; rz(theta) b; cx a,b; rx(-pi/2) a; rx(-pi/2) b; }"
    if k == Op.RZZ:
        return "gate rzz(theta) a,b { cx a,b; rz(theta) b; cx a,b; }"
    if k == Op.CSWAP:
        return "gate cswap a,b,c { cx c,b; ccx a,b,c; cx c,b; }"
    if k == Op.CCZ:
        return "gate ccz a,b,c { h c; ccx a,b,c; h c; }"
    return ""


def to_qasm2(c: Circuit) raises -> String:
    """OpenQASM 2.0 text of `c` (see the module docstring for the contract)."""
    if c.num_params() > 0:
        raise Error("QM-E102: OpenQASM 2 export needs bound parameters; circuit has " + c.param_names_str() + ". Hint: call bind() first, or export OpenQASM 3 with to_qasm(), which keeps 'input float' parameters.")
    var s = String("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n")
    # definitions first, in opcode order, each before any use
    var used = List[Bool](length=Op.NUM_OPS, fill=False)
    for g in c.ops:
        if g.kind >= 0 and g.kind < Op.NUM_OPS:
            used[g.kind] = True
    for k in range(Op.NUM_OPS):
        if used[k] and op_arity(k) > 0 and not op_is_noise(k) and not op_is_annotation(k):
            var d = qasm2_gate_definition(k)
            if d.byte_length() > 0:
                s += d + "\n"
    s += "qreg q[" + String(max(c.num_qubits, 1)) + "];\n"
    if c.num_clbits > 0:
        s += "creg c[" + String(c.num_clbits) + "];\n"
    for g in c.ops:
        var k = g.kind
        if op_is_annotation(k) and k != Op.BARRIER:
            continue  # detectors, observables, ticks: no OpenQASM 2 form
        if op_is_noise(k):
            var line = String("// noise channel not exported: ") + op_name(k)
            var npn = op_num_params(k)
            if npn > 0:
                line += "("
                for j in range(npn):
                    if j > 0:
                        line += ","
                    line += qasm2_real(g.param(j).value())
                line += ")"
            for j in range(op_arity(k)):
                line += " q[" if j == 0 else ",q["
                line += String(g.qubit(j)) + "]"
            s += line + "\n"
            continue
        var cond = String()
        if g.cond_reg >= 0:
            cond = "if(c==" + register_condition_value(c, g) + ") "
        elif g.cond >= 0:
            if c.num_clbits != 1:
                raise Error("QM-E552: OpenQASM 2 conditions compare a whole register, but instruction '" + op_name(k) + "' is conditioned on bit c[" + String(g.cond) + "] of a " + String(c.num_clbits) + "-bit register. Hint: export OpenQASM 3 with to_qasm(), which keeps single-bit conditions.")
            cond = "if(c==" + String(g.cond_val) + ") "
        if k == Op.BARRIER:
            s += "barrier q;\n"
            continue
        if k == Op.MEASURE:
            s += cond + "measure q[" + String(g.q0) + "] -> c[" + String(g.clbit) + "];\n"
            continue
        if k == Op.RESET:
            s += cond + "reset q[" + String(g.q0) + "];\n"
            continue
        s += cond + qasm2_gate_name(k)
        var np = op_num_params(k)
        if np > 0:
            s += "("
            for j in range(np):
                if j > 0:
                    s += ","
                s += qasm2_real(g.param(j).value())
            s += ")"
        var a = op_arity(k)
        for j in range(a):
            s += " q[" if j == 0 else ",q["
            s += String(g.qubit(j)) + "]"
        s += ";\n"
    if c.global_phase != 0.0:
        s += "// global phase " + qasm2_real(c.global_phase) + " (not expressible in OpenQASM 2)\n"
    return s^
