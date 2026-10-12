"""OpenQASM 2.0 emitter (qelib1.inc dialect) for interoperability.

OpenQASM 3 (`qumo.ir.qasm.to_qasm`) is QUMO's primary text format; QASM 2 is
still the format every SDK parser accepts (Qiskit `qasm2`, pytket, Cirq,
Braket, Staq), so benchmark harnesses and legacy tools use this emitter.
Gates outside qelib1 (iswap, iswapdg, ecr, ryy, ccz, sxdg on old parsers)
are emitted with exact `gate` definitions. Parameters must be bound.
Classical conditions are emitted as `if(c==v)` only when the condition bit is
the sole bit of the register semantics allow; otherwise QM-E550 is raised.
"""

from qumo.ir.circuit import Circuit
from qumo.ir.qasm import register_condition_value
from qumo.ir.gates import Op, op_name, op_num_params, op_arity, op_is_annotation, op_is_noise


def _fmt(v: Float64) -> String:
    return String(v)


def _needs(c: Circuit, k: Int) -> Bool:
    for g in c.ops:
        if g.kind == k:
            return True
    return False


def to_qasm2(c: Circuit) raises -> String:
    if c.num_params() > 0:
        raise Error("QM-E550: OpenQASM 2 export needs bound parameters; circuit has " + c.param_names_str() + ". Hint: call bind() first or use OpenQASM 3 (to_qasm).")
    var s = String("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n")
    if _needs(c, Op.ISWAP):
        s += "gate iswap a,b { s a; s b; h a; cx a,b; cx b,a; h b; }\n"
    if _needs(c, Op.ISWAPDG):
        s += "gate iswapdg a,b { h b; cx b,a; cx a,b; h a; sdg b; sdg a; }\n"
    if _needs(c, Op.ECR):
        s += "gate ecr a,b { rzx(pi/4) a,b; x a; rzx(-pi/4) a,b; }\n"
    if _needs(c, Op.RYY):
        s += "gate ryy(theta) a,b { rx(pi/2) a; rx(pi/2) b; cx a,b; rz(theta) b; cx a,b; rx(-pi/2) a; rx(-pi/2) b; }\n"
    if _needs(c, Op.CCZ):
        s += "gate ccz a,b,c { h c; ccx a,b,c; h c; }\n"
    if _needs(c, Op.SXDG):
        s += "gate sxdg_ a { h a; sdg a; h a; }\n"
    s += "qreg q[" + String(max(c.num_qubits, 1)) + "];\n"
    if c.num_clbits > 0:
        s += "creg c[" + String(c.num_clbits) + "];\n"
    for g in c.ops:
        var k = g.kind
        if op_is_annotation(k) and k != Op.BARRIER:
            continue
        if op_is_noise(k):
            continue
        if g.cond_reg >= 0:
            s += "if(c==" + register_condition_value(c, g) + ") "
        elif g.cond >= 0:
            if c.num_clbits != 1:
                raise Error("QM-E551: OpenQASM 2 can only condition on a whole register; circuit conditions on bit " + String(g.cond) + " of " + String(c.num_clbits) + ". Hint: export OpenQASM 3 instead.")
            s += "if(c==" + String(g.cond_val) + ") "
        if k == Op.BARRIER:
            s += "barrier q;\n"
            continue
        if k == Op.MEASURE:
            s += "measure q[" + String(g.q0) + "] -> c[" + String(g.clbit) + "];\n"
            continue
        if k == Op.RESET:
            s += "reset q[" + String(g.q0) + "];\n"
            continue
        var name: String
        if k == Op.U:
            name = "u3"
        elif k == Op.P:
            name = "u1"
        elif k == Op.CP:
            name = "cu1"
        elif k == Op.SXDG:
            name = "sxdg_"
        else:
            name = op_name(k)
        s += name
        var np = op_num_params(k)
        if np > 0:
            s += "(" + _fmt(g.p0.value())
            if np > 1:
                s += "," + _fmt(g.p1.value())
            if np > 2:
                s += "," + _fmt(g.p2.value())
            s += ")"
        var a = op_arity(k)
        s += " q[" + String(g.q0) + "]"
        if a >= 2:
            s += ",q[" + String(g.q1) + "]"
        if a >= 3:
            s += ",q[" + String(g.q2) + "]"
        s += ";\n"
    if c.global_phase != 0.0:
        s += "// global_phase " + _fmt(c.global_phase) + "\n"
    return s
