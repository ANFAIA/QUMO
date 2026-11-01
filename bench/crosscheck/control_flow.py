#!/usr/bin/env python
"""Independent oracle for QUMO programs with classical control flow.

Random OpenQASM 3 programs go through two implementations that share no code:
  * QUMO: `load_any` + `sample_program`, by shots (control_flow_runner.mojo);
  * this file: `openqasm3` parses the text, and a small interpreter walks the AST
    on numpy state vectors. At each measurement it follows both branches with
    their exact weights, so the result is an exact distribution, up to a mass it
    reports: branches below WEIGHT_CUT or past LOOP_CUT iterations are dropped and
    counted.

The classical rules here are written from the semantics table of the project plan
(docs/CONTROL_FLOW.md once published), not from qumo/ir/classical.mojo.

TV acceptance, fixed before the first comparison and not to be tuned to a result:
for N shots and exact probabilities p_k, an unbiased sampler has expected total
variation E = 0.5 * sum sqrt(2 p_k (1 - p_k) / (pi N)), and by McDiarmid TV exceeds
its mean by t only with probability exp(-2 N t^2). A program passes when

    TV <= 1.2 * E + sqrt(ln(n / 1e-6) / (2 N)) + truncated_mass

with n programs in the run (a family-wise false-alarm rate of 1e-6).

Usage: bench/.venv/bin/python bench/crosscheck/control_flow.py [--programs 200]
       [--shots 20000] [--seed 1] [--dir DIR]
"""

import argparse
import cmath
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import openqasm3
from openqasm3 import ast

WEIGHT_CUT = 1e-7
LOOP_CUT = 4096


# --------------------------------------------------------------------------
# Classical values. A value is (type, payload); types are tuples:
# ("bit", n), ("bool",), ("int", n), ("uint", n), ("float",), ("angle", n)
# --------------------------------------------------------------------------
class OracleError(Exception):
    pass


def wrap(v, t):
    tag = t[0]
    if tag == "bool":
        return 1 if v else 0
    n = t[1]
    v &= (1 << n) - 1
    if tag == "int" and v >> (n - 1):
        v -= 1 << n
    return v


def literal_fits(v, t):
    n = t[1]
    if t[0] == "int":
        return -(1 << (n - 1)) <= v < (1 << (n - 1))
    return 0 <= v < (1 << n)


def smallest_width(v, signed):
    w = 1
    while not (-(1 << (w - 1)) <= v < (1 << (w - 1)) if signed else 0 <= v < (1 << w)):
        w += 1
    return w


def to_float(v, t):
    if t[0] == "float":
        return float(v)
    if t[0] in ("int", "uint", "bool", "bit"):
        return float(v)
    raise OracleError("angle does not convert to float")


def float_to_int(x, t):
    if math.isnan(x) or math.isinf(x):
        raise OracleError("float range")
    v = int(x)  # python truncates toward zero
    if t[0] == "int":
        if not (-(1 << (t[1] - 1)) <= v < (1 << (t[1] - 1))):
            raise OracleError("float range")
    elif not (0 <= v < (1 << t[1])):
        raise OracleError("float range")
    return v


def float_to_angle(x, n):
    if math.isnan(x) or math.isinf(x):
        raise OracleError("float range")
    turns = x / (2 * math.pi)
    frac = turns - math.floor(turns)
    k = math.floor(frac * (1 << n) + 0.5)
    return 0 if k >= (1 << n) else k


def cast(val, t):
    src, v = val
    if src == t:
        return (t, v)
    tag = t[0]
    if tag == "bool":
        return (t, 1 if v != 0 else 0)
    if src[0] == "float":
        if tag == "angle":
            return (t, float_to_angle(v, t[1]))
        if tag in ("int", "uint"):
            return (t, wrap(float_to_int(v, t), t))
        raise OracleError("float to bit")
    if tag == "float":
        return (t, to_float(v, src))
    if tag == "angle":
        if src[0] == "angle":
            if t[1] >= src[1]:
                return (t, v << (t[1] - src[1]))
            return (t, v >> (src[1] - t[1]))
        return (t, v & ((1 << t[1]) - 1))
    # integer family: reinterpret the bits, modulo the target width
    return (t, wrap(v, t))


def uintish(val):
    t, v = val
    if t[0] == "bool":
        return (("uint", 1), v)
    if t[0] == "bit":
        return (("uint", t[1]), v)
    return val


class Lit(tuple):
    """A value that came from literals only: it adapts to the other operand."""


def lit(t, v):
    return Lit((t, v))


def is_lit(x):
    return isinstance(x, Lit)


def common(a, b):
    a2, b2 = uintish(a), uintish(b)
    la, lb = is_lit(a), is_lit(b)
    ta, tb = a2[0], b2[0]
    if ta[0] == "float" or tb[0] == "float":
        if ta[0] == "angle" and lb:
            return cast(tuple(a), ta), cast(b2, ta)
        if tb[0] == "angle" and la:
            return cast(a2, tb), cast(tuple(b), tb)
        return (("float",), to_float(a2[1], ta)), (("float",), to_float(b2[1], tb))
    if la and lb:
        return tuple(a2), tuple(b2)
    if la or lb:
        lv = (a2 if la else b2)[1]
        t = tb if la else ta
        if t[0] == "angle":
            raise OracleError("an integer literal does not mix with an angle")
        if not literal_fits(lv, t):
            if t[0] == "uint" and lv < 0:
                raise OracleError("a negative literal does not mix with uint")
            t = (t[0], max(t[1], smallest_width(lv, t[0] == "int")))
        return cast(a2, t), cast(b2, t)
    if ta[0] != tb[0]:
        # bit and bool hold plain non-negative values
        if a[0][0] in ("bit", "bool") and tb[0] == "int":
            t = ("int", min(64, max(tb[1], ta[1] + 1)))
            return cast(a2, t), cast(b2, t)
        if b[0][0] in ("bit", "bool") and ta[0] == "int":
            t = ("int", min(64, max(ta[1], tb[1] + 1)))
            return cast(a2, t), cast(b2, t)
        raise OracleError("types do not mix")
    if ta[1] >= tb[1]:
        return cast(a2, ta), cast(b2, ta)
    return cast(a2, tb), cast(b2, tb)


def int_div(a, b):
    if b == 0:
        raise OracleError("division by zero")
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def int_rem(a, b):
    r = abs(a) % abs(b)
    return -r if a < 0 else r


def binary(op, a, b):
    if op in ("==", "!=", "<", "<=", ">", ">="):
        ca, cb = (a, b) if a[0] == ("bool",) and b[0] == ("bool",) else common(a, b)
        x, y = ca[1], cb[1]
        r = {"==": x == y, "!=": x != y, "<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y}[op]
        return (("bool",), 1 if r else 0)
    if op in ("<<", ">>", "&", "|", "^") and (a[0][0] == "float" or b[0][0] == "float" or a[0][0] == "bool"):
        raise OracleError("operator on a float")
    if op in ("<<", ">>"):
        t, v = a
        n = t[1]
        s = b[1]
        if s >= n:
            return wraplit(a, b, (t, 0))
        u = v & ((1 << n) - 1)
        return wraplit(a, b, (t, wrap(u << s, t) if op == "<<" else wrap(u >> s, t)))
    if op in ("&", "|", "^"):
        if a[0][0] == "bit" and b[0][0] == "bit":
            t = a[0]
            ca, cb = a, b
        else:
            ca, cb = common(a, b)
            t = ca[0]
        x, y = ca[1], cb[1]
        return wraplit(a, b, (t, wrap(x & y if op == "&" else (x | y if op == "|" else x ^ y), t)))
    # arithmetic
    if "angle" in (a[0][0], b[0][0]):
        return angle_arith(op, a, b)
    if is_lit(a) and is_lit(b) and a[0][0] == "int" and b[0][0] == "int":
        if op == "/" and b[1] != 0 and a[1] % b[1] != 0:
            return lit(("float",), a[1] / b[1])
        if op == "**" and b[1] < 0:
            return lit(("float",), float(a[1]) ** b[1])
    if op == "**" and a[0][0] != "float" and b[0][0] != "float":
        base = uintish(a)
        if b[1] < 0:
            raise OracleError("negative exponent")
        t = base[0]
        out = (t, wrap(pow(base[1], b[1], 1 << t[1]), t))
        return lit(*out) if is_lit(a) and is_lit(b) else out
    ca, cb = common(a, b)
    t = ca[0]
    x, y = ca[1], cb[1]
    if t[0] == "float":
        if op == "+":
            return wraplit(a, b, (t, x + y))
        if op == "-":
            return wraplit(a, b, (t, x - y))
        if op == "*":
            return wraplit(a, b, (t, x * y))
        if op == "**":
            return wraplit(a, b, (t, x ** y))
        if y == 0:
            raise OracleError("division by zero")
        if op == "/":
            return wraplit(a, b, (t, x / y))
        q = x / y
        return wraplit(a, b, (t, x - y * math.trunc(q)))
    if op == "+":
        r = x + y
    elif op == "-":
        r = x - y
    elif op == "*":
        r = x * y
    elif op == "/":
        r = int_div(x, y)
    elif op == "%":
        if y == 0:
            raise OracleError("division by zero")
        r = int_rem(x, y)
    else:
        raise OracleError("operator " + op)
    return wraplit(a, b, (t, wrap(r, t)))


def wraplit(a, b, out):
    return lit(*out) if is_lit(a) and is_lit(b) else out


def angle_arith(op, a, b):
    ta, tb = a[0], b[0]
    angle = ta if ta[0] == "angle" else tb
    n = angle[1]
    if op in ("+", "-"):
        if is_lit(a) and ta[0] == "float":
            a = (angle, float_to_angle(a[1], n))
        if is_lit(b) and tb[0] == "float":
            b = (angle, float_to_angle(b[1], n))
        v = a[1] + b[1] if op == "+" else a[1] - b[1]
        return (angle, v & ((1 << n) - 1))
    if op == "*":
        ang, sc = (a, b) if ta[0] == "angle" else (b, a)
        return (angle, (ang[1] * (sc[1] & ((1 << n) - 1))) & ((1 << n) - 1))
    if op == "/":
        if b[1] == 0:
            raise OracleError("division by zero")
        if tb[0] == "angle":
            return (("uint", n), a[1] // b[1])
        return (angle, a[1] // b[1])
    raise OracleError("angle operator " + op)


FUNCS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "arcsin": math.asin, "arccos": math.acos,
         "arctan": math.atan, "exp": math.exp, "ln": math.log, "sqrt": math.sqrt, "floor": math.floor,
         "ceil": math.ceil}


def parse_type(tp, const_env):
    if isinstance(tp, ast.BitType):
        return ("bit", const_int(tp.size, const_env) if tp.size else 1)
    if isinstance(tp, ast.BoolType):
        return ("bool",)
    if isinstance(tp, ast.IntType):
        return ("int", const_int(tp.size, const_env) if tp.size else 64)
    if isinstance(tp, ast.UintType):
        return ("uint", const_int(tp.size, const_env) if tp.size else 64)
    if isinstance(tp, ast.FloatType):
        return ("float",)
    if isinstance(tp, ast.AngleType):
        return ("angle", const_int(tp.size, const_env))
    raise OracleError("type " + type(tp).__name__)


def const_int(e, env):
    if isinstance(e, ast.IntegerLiteral):
        return e.value
    if isinstance(e, ast.Identifier) and e.name in env:
        return env[e.name][1]
    raise OracleError("constant expected")


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
S2 = 1 / math.sqrt(2)
FIXED = {
    "id": [[1, 0], [0, 1]],
    "x": [[0, 1], [1, 0]],
    "y": [[0, -1j], [1j, 0]],
    "z": [[1, 0], [0, -1]],
    "h": [[S2, S2], [S2, -S2]],
    "s": [[1, 0], [0, 1j]],
    "sdg": [[1, 0], [0, -1j]],
    "t": [[1, 0], [0, cmath.exp(1j * math.pi / 4)]],
    "tdg": [[1, 0], [0, cmath.exp(-1j * math.pi / 4)]],
    "sx": [[(1 + 1j) / 2, (1 - 1j) / 2], [(1 - 1j) / 2, (1 + 1j) / 2]],
}


def rot(name, a):
    c, s = math.cos(a / 2), math.sin(a / 2)
    if name == "rx":
        return [[c, -1j * s], [-1j * s, c]]
    if name == "ry":
        return [[c, -s], [s, c]]
    if name == "rz":
        return [[cmath.exp(-1j * a / 2), 0], [0, cmath.exp(1j * a / 2)]]
    if name == "p":
        return [[1, 0], [0, cmath.exp(1j * a)]]
    raise OracleError("rotation " + name)


def apply_1q(st, m, q):
    m = np.array(m, dtype=complex)
    st = np.tensordot(m, st, axes=([1], [q]))
    return np.moveaxis(st, 0, q)


def apply_controlled(st, m, c, t):
    """m on qubit t where qubit c is 1."""
    idx = [slice(None)] * st.ndim
    idx[c] = 1
    sub = st[tuple(idx)]
    tt = t if t < c else t - 1
    out = st.copy()
    out[tuple(idx)] = apply_1q(sub, m, tt)
    return out


# --------------------------------------------------------------------------
# Interpreter
# --------------------------------------------------------------------------
class World:
    __slots__ = ("weight", "state", "env", "bits", "flow", "loops")

    def __init__(self, weight, state, env, bits):
        self.weight = weight
        self.state = state
        self.env = env
        self.bits = bits
        self.flow = None
        self.loops = 0

    def fork(self, weight, state):
        w = World(weight, state, dict(self.env), list(self.bits))
        w.flow = self.flow
        w.loops = self.loops
        return w


class Oracle:
    def __init__(self, program):
        self.program = program
        self.nq = 0
        self.nc = 0
        self.qregs = {}
        self.cregs = {}
        self.consts = {}
        self.outputs = []
        self.dropped = 0.0
        for st in program.statements:
            if isinstance(st, ast.QubitDeclaration):
                size = st.size.value if st.size else 1
                self.qregs[st.qubit.name] = (self.nq, size)
                self.nq += size
            elif isinstance(st, ast.ClassicalDeclaration) and isinstance(st.type, ast.BitType):
                size = const_int(st.type.size, {}) if st.type.size else 1
                self.cregs[st.identifier.name] = (self.nc, size)
                self.nc += size
            elif isinstance(st, ast.IODeclaration):
                if isinstance(st.type, ast.BitType):
                    size = const_int(st.type.size, {}) if st.type.size else 1
                    self.cregs[st.identifier.name] = (self.nc, size)
                    self.nc += size
                else:
                    self.outputs.append(st.identifier.name)

    def run(self):
        st = np.zeros((2,) * max(self.nq, 1), dtype=complex)
        st[(0,) * max(self.nq, 1)] = 1.0
        worlds = [World(1.0, st, {}, [0] * self.nc)]
        finished = self.block(self.program.statements, worlds)
        dist = defaultdict(float)
        for w in finished:
            key = "".join(str(b) for b in reversed(w.bits)) or "-"
            outs = []
            for nm in self.outputs:
                t, v = w.env[nm]
                outs.append(round(v, 9) if t[0] == "float" else (int(v)))
            dist[(key,) + tuple(outs)] += w.weight
        return dist, self.dropped

    def block(self, stmts, worlds):
        for s in stmts:
            nxt = []
            for w in worlds:
                if w.flow:
                    nxt.append(w)
                else:
                    nxt.extend(self.stmt(s, w))
            worlds = self.merge(nxt)
        return worlds

    def merge(self, worlds):
        """Worlds with one classical state and the same quantum state are one."""
        if len(worlds) < 2:
            return worlds
        groups = {}
        out = []
        for w in worlds:
            key = (w.flow, w.loops, tuple(w.bits), tuple(sorted((k, repr(v)) for k, v in w.env.items())))
            done = False
            for other in groups.get(key, []):
                if abs(abs(np.vdot(other.state, w.state)) - 1.0) < 1e-11:
                    other.weight += w.weight
                    done = True
                    break
            if not done:
                groups.setdefault(key, []).append(w)
                out.append(w)
        return out

    def scope_body(self, node):
        return node if isinstance(node, list) else [node]

    def stmt(self, s, w):
        if isinstance(s, (ast.Include, ast.QubitDeclaration)):
            return [w]
        if isinstance(s, ast.IODeclaration):
            if not isinstance(s.type, ast.BitType):
                t = parse_type(s.type, w.env)
                w.env[s.identifier.name] = (t, 0)
                w.env["__unset_" + s.identifier.name] = True
            return [w]
        if isinstance(s, ast.ClassicalDeclaration):
            if isinstance(s.type, ast.BitType):
                if s.init_expression is not None:
                    self.assign_bits(s.identifier.name, None, self.expr(s.init_expression, w), w)
                return [w]
            t = parse_type(s.type, w.env)
            if s.init_expression is not None:
                w.env[s.identifier.name] = self.convert(self.expr(s.init_expression, w), t)
            else:
                w.env[s.identifier.name] = (t, 0)
            return [w]
        if isinstance(s, ast.ConstantDeclaration):
            t = parse_type(s.type, w.env)
            w.env[s.identifier.name] = self.convert(self.expr(s.init_expression, w), t)
            return [w]
        if isinstance(s, ast.ClassicalAssignment):
            return self.assignment(s, w)
        if isinstance(s, ast.QuantumGate):
            self.gate(s, w)
            return [w]
        if isinstance(s, ast.QuantumMeasurementStatement):
            return self.measure(s, w)
        if isinstance(s, ast.QuantumReset):
            return self.reset(s, w)
        if isinstance(s, ast.BranchingStatement):
            c = self.convert(self.expr(s.condition, w), ("bool",))[1]
            return self.block(s.if_block if c else s.else_block, [w])
        if isinstance(s, ast.WhileLoop):
            return self.while_loop(s, w)
        if isinstance(s, ast.ForInLoop):
            return self.for_loop(s, w)
        if isinstance(s, ast.SwitchStatement):
            return self.switch(s, w)
        if isinstance(s, ast.BreakStatement):
            w.flow = "break"
            return [w]
        if isinstance(s, ast.ContinueStatement):
            w.flow = "continue"
            return [w]
        if isinstance(s, ast.CompoundStatement):
            return self.block(s.statements, [w])
        raise OracleError("statement " + type(s).__name__)

    def convert(self, val, t):
        if t[0] == "angle" and val[0][0] == "float":
            return (t, float_to_angle(val[1], t[1]))
        return cast(tuple(val), t)

    def lvalue(self, target):
        """(name, index or None)."""
        if isinstance(target, ast.Identifier):
            return target.name, None
        ix = target.indices[0]
        if isinstance(ix, list) and len(ix) == 1:
            ix = ix[0]
            if isinstance(ix, ast.RangeDefinition):
                return target.name.name, (ix.start.value, ix.end.value)
            return target.name.name, ix.value
        raise OracleError("index form")

    def assign_bits(self, name, index, val, w):
        first, size = self.cregs[name]
        if index is None:
            lo, n = first, size
        elif isinstance(index, tuple):
            lo, n = first + index[0], index[1] - index[0] + 1
        else:
            lo, n = first + index, 1
        v = cast(tuple(val), ("bit", n))[1]
        for j in range(n):
            w.bits[lo + j] = (v >> j) & 1

    def assignment(self, s, w):
        name, index = self.lvalue(s.lvalue)
        op = s.op.name
        if isinstance(s.rvalue, ast.QuantumMeasurement):
            return self.measure_into(s.rvalue.qubit, s.lvalue, w)
        val = self.expr(s.rvalue, w)
        if op != "=":
            cur = self.read_name(name, w) if index is None else self.read_bits(name, index, w)
            val = binary(op[:-1], cur, val)
        if name in self.cregs:
            self.assign_bits(name, index, val, w)
        else:
            t = w.env[name][0]
            w.env[name] = self.convert(val, t)
            w.env.pop("__unset_" + name, None)
        return [w]

    def read_bits(self, name, index, w):
        first, size = self.cregs[name]
        if index is None:
            lo, n = first, size
        elif isinstance(index, tuple):
            lo, n = first + index[0], index[1] - index[0] + 1
        else:
            lo, n = first + index, 1
        v = 0
        for j in range(n):
            v |= w.bits[lo + j] << j
        return (("bit", n), v)

    def read_name(self, name, w):
        if name in self.cregs:
            return self.read_bits(name, None, w)
        return tuple(w.env[name])

    def qubits_of(self, q):
        if isinstance(q, ast.Identifier):
            first, size = self.qregs[q.name]
            return list(range(first, first + size))
        first, size = self.qregs[q.name.name]
        ix = q.indices[0][0]
        return [first + ix.value]

    def gate(self, g, w):
        qs = [self.qubits_of(q) for q in g.qubits]
        qs = [x[0] for x in qs]
        nm = g.name.name
        args = []
        for a in g.arguments:
            val = self.expr(a, w)
            if val[0][0] == "angle":
                args.append(2 * math.pi * val[1] / (1 << val[0][1]))
            else:
                args.append(float(val[1]))
        st = w.state
        if nm in FIXED:
            st = apply_1q(st, FIXED[nm], qs[0])
        elif nm in ("rx", "ry", "rz", "p"):
            st = apply_1q(st, rot(nm, args[0]), qs[0])
        elif nm == "cx":
            st = apply_controlled(st, FIXED["x"], qs[0], qs[1])
        elif nm == "cz":
            st = apply_controlled(st, FIXED["z"], qs[0], qs[1])
        elif nm == "cy":
            st = apply_controlled(st, FIXED["y"], qs[0], qs[1])
        elif nm == "cp":
            st = apply_controlled(st, rot("p", args[0]), qs[0], qs[1])
        elif nm == "swap":
            st = np.swapaxes(st, qs[0], qs[1])
        else:
            raise OracleError("gate " + nm)
        w.state = st

    def split(self, w, q):
        """(p0 state or None, p1 state or None, p0, p1)."""
        probs = np.sum(np.abs(w.state) ** 2, axis=tuple(i for i in range(w.state.ndim) if i != q))
        p0, p1 = float(probs[0]), float(probs[1])
        s = []
        for b, p in ((0, p0), (1, p1)):
            if p < 1e-13:
                s.append(None)
                continue
            idx = [slice(None)] * w.state.ndim
            idx[q] = 1 - b
            st = w.state.copy()
            st[tuple(idx)] = 0
            s.append(st / math.sqrt(p))
        return s[0], s[1], p0, p1

    def measure(self, s, w):
        if s.target is None:
            raise OracleError("measure without a target")
        return self.measure_into(s.measure.qubit, s.target, w)

    def measure_into(self, q, target, w):
        qs = self.qubits_of(q)
        name, index = self.lvalue(target)
        first, size = self.cregs[name]
        if index is None:
            bits = [first + j for j in range(size)]
        elif isinstance(index, tuple):
            bits = [first + j for j in range(index[0], index[1] + 1)]
        else:
            bits = [first + index]
        worlds = [w]
        for qb, b in zip(qs, bits):
            nxt = []
            for x in worlds:
                s0, s1, p0, p1 = self.split(x, qb)
                for st, p, val in ((s0, p0, 0), (s1, p1, 1)):
                    if st is None:
                        continue
                    y = x.fork(x.weight * p, st)
                    y.bits[b] = val
                    if y.weight < WEIGHT_CUT:
                        self.dropped += y.weight
                    else:
                        nxt.append(y)
            worlds = nxt
        return worlds

    def reset(self, s, w):
        q = self.qubits_of(s.qubits)[0]
        s0, s1, p0, p1 = self.split(w, q)
        out = []
        if s0 is not None:
            out.append(w.fork(w.weight * p0, s0))
        if s1 is not None:
            out.append(w.fork(w.weight * p1, apply_1q(s1, FIXED["x"], q)))
        return out

    def while_loop(self, s, w):
        done = []
        worlds = [w]
        while worlds:
            nxt = []
            for x in worlds:
                if not self.convert(self.expr(s.while_condition, x), ("bool",))[1]:
                    done.append(x)
                else:
                    x.loops += 1
                    if x.loops > LOOP_CUT or x.weight < WEIGHT_CUT:
                        self.dropped += x.weight
                    else:
                        nxt.append(x)
            worlds = []
            for x in self.block(s.block, nxt):
                if x.flow == "break":
                    x.flow = None
                    done.append(x)
                else:
                    x.flow = None
                    worlds.append(x)
            worlds = self.merge(worlds)
        return self.merge(done)

    def for_loop(self, s, w):
        name = s.identifier.name
        t = parse_type(s.type, w.env)
        rng = s.set_declaration
        if isinstance(rng, ast.RangeDefinition):
            start = int(self.expr(rng.start, w)[1])
            stop = int(self.expr(rng.end, w)[1])
            step = int(self.expr(rng.step, w)[1]) if rng.step else 1
            if step == 0:
                raise OracleError("zero step")
            values = list(range(start, stop + (1 if step > 0 else -1), step))
        else:
            values = [int(self.expr(v, w)[1]) for v in rng.values]
        worlds = [w]
        done = []
        for v in values:
            if not worlds:
                break
            for x in worlds:
                x.env[name] = (t, wrap(v, t))
            nxt = []
            for x in self.block(s.block, worlds):
                if x.flow == "break":
                    x.flow = None
                    done.append(x)
                else:
                    x.flow = None
                    nxt.append(x)
            worlds = self.merge(nxt)
        return self.merge(done + worlds)

    def switch(self, s, w):
        v = self.expr(s.target, w)
        v = cast(tuple(v), ("int", 64))[1]
        for values, body in s.cases:
            for c in values:
                if int(self.expr(c, w)[1]) == v:
                    return self.block(body.statements, [w])
        if s.default is not None:
            return self.block(s.default.statements, [w])
        return [w]

    def expr(self, e, w):
        if isinstance(e, ast.IntegerLiteral):
            return lit(("int", 64) if e.value < (1 << 63) else ("uint", 64), e.value)
        if isinstance(e, ast.FloatLiteral):
            return lit(("float",), e.value)
        if isinstance(e, ast.BooleanLiteral):
            return lit(("bool",), 1 if e.value else 0)
        if isinstance(e, ast.BitstringLiteral):
            return (("bit", e.width), e.value)
        if isinstance(e, ast.Identifier):
            if e.name == "pi":
                return lit(("float",), math.pi)
            if e.name == "tau":
                return lit(("float",), 2 * math.pi)
            if e.name == "euler":
                return lit(("float",), math.e)
            return self.read_name(e.name, w)
        if isinstance(e, ast.IndexExpression):
            ix = e.index[0] if isinstance(e.index, list) else e.index
            if isinstance(ix, ast.RangeDefinition):
                return self.read_bits(e.collection.name, (ix.start.value, ix.end.value), w)
            return self.read_bits(e.collection.name, ix.value, w)
        if isinstance(e, ast.BinaryExpression):
            op = e.op.name
            if op == "&&":
                if not self.convert(self.expr(e.lhs, w), ("bool",))[1]:
                    return (("bool",), 0)
                return self.convert(self.expr(e.rhs, w), ("bool",))
            if op == "||":
                if self.convert(self.expr(e.lhs, w), ("bool",))[1]:
                    return (("bool",), 1)
                return self.convert(self.expr(e.rhs, w), ("bool",))
            return binary(op, self.expr(e.lhs, w), self.expr(e.rhs, w))
        if isinstance(e, ast.UnaryExpression):
            op = e.op.name
            v = self.expr(e.expression, w)
            if op == "!":
                return (("bool",), 0 if self.convert(v, ("bool",))[1] else 1)
            if op == "~":
                t = v[0]
                return (t, wrap(~v[1], t))
            if is_lit(v):
                return lit(v[0], -v[1])
            u = uintish(v)
            if u[0][0] == "float":
                return (u[0], -u[1])
            return (u[0], wrap(-u[1], u[0]))  # a uint wraps too
        if isinstance(e, ast.Cast):
            t = parse_type(e.type, w.env)
            if t[0] == "angle" and is_lit(self.expr(e.argument, w)) is False and self.expr(e.argument, w)[0][0] == "int":
                raise OracleError("int to angle")
            return self.convert(tuple(self.expr(e.argument, w)), t)
        if isinstance(e, ast.FunctionCall):
            nm = e.name.name
            args = [self.expr(a, w) for a in e.arguments]
            if nm == "popcount":
                return (("uint", 64), bin(args[0][1] & ((1 << args[0][0][1]) - 1)).count("1"))
            if nm == "abs":
                if args[0][0][0] == "int":
                    t = args[0][0]
                    return (t, wrap(abs(args[0][1]), t))
                return (args[0][0], abs(args[0][1]))
            x = to_float(uintish(args[0])[1], uintish(args[0])[0])
            if nm in FUNCS:
                r = FUNCS[nm](x)
                return (("float",), float(r))
            raise OracleError("function " + nm)
        raise OracleError("expression " + type(e).__name__)


# --------------------------------------------------------------------------
# Random programs
# --------------------------------------------------------------------------
class Gen:
    def __init__(self, rng):
        self.r = rng
        self.nq = rng.randint(1, 3)
        self.nc = rng.randint(2, 5)
        self.vars = {}  # name -> type
        self.lines = []
        self.pad = 0
        self.count = 0
        self.loops = 0

    def emit(self, s):
        self.lines.append("  " * self.pad + s)

    def fresh(self, t, prefix="v"):
        nm = prefix + str(self.count)
        self.count += 1
        return nm

    def ints(self):
        return [n for n, t in self.vars.items() if t[0] == "int"]

    def uints(self):
        return [n for n, t in self.vars.items() if t[0] == "uint"]

    def bools(self):
        return [n for n, t in self.vars.items() if t[0] == "bool"]

    def floats(self):
        return [n for n, t in self.vars.items() if t[0] == "float"]

    def angles(self):
        return [n for n, t in self.vars.items() if t[0] == "angle"]

    def bit(self):
        return "c[%d]" % self.r.randrange(self.nc)

    def int_expr(self, d=0):
        r = self.r
        k = r.random()
        if d > 2 or k < 0.3:
            names = self.ints()
            if names and r.random() < 0.75:
                return r.choice(names)
            return str(r.randint(-9, 9)) if r.random() < 0.5 else str(r.randint(0, 40))
        a = self.int_expr(d + 1)
        if k < 0.65:
            return "(%s %s %s)" % (a, r.choice("+-*"), self.int_expr(d + 1))
        if k < 0.75:
            # two literals divide as reals (see qumo.ir.classical), so give it a type
            return "(int[32](%s) %s %s)" % (a, r.choice("/%"), r.choice(["2", "3", "-3", "5", "7"]))
        if k < 0.82:
            return "-(%s)" % a if not a.lstrip("-").isdigit() else a
        if k < 0.9 and self.uints():
            return "int[%d](%s)" % (r.choice([4, 8, 16, 32]), r.choice(self.uints()))
        if k < 0.95 and self.floats():
            return "int[32](%s)" % r.choice(self.floats())
        return "int[8](%s)" % self.bit()

    def uint_expr(self, d=0):
        r = self.r
        k = r.random()
        if d > 2 or k < 0.3:
            names = self.uints()
            if names and r.random() < 0.75:
                return r.choice(names)
            return str(r.randint(0, 40))
        a = self.uint_expr(d + 1)
        if k < 0.55:
            return "(%s %s %s)" % (a, r.choice(["+", "*"]), self.uint_expr(d + 1))
        if k < 0.65:
            return "(uint[8](%s) %s %s)" % (a, r.choice(["<<", ">>"]), r.choice(["0", "1", "2", "3", "9", "70"]))
        if k < 0.8:
            return "(%s %s %s)" % (a, r.choice("&|^"), self.uint_expr(d + 1))
        if k < 0.87:
            return "(uint[32](%s) %s %s)" % (a, r.choice("/%"), r.choice(["2", "3", "5"]))
        if k < 0.93:
            return "~(uint[8](%s))" % a
        if self.ints():
            return "uint[%d](%s)" % (r.choice([4, 8, 16]), r.choice(self.ints()))
        return "uint[3](c[0:%d])" % min(2, self.nc - 1)

    def bool_expr(self, d=0):
        r = self.r
        k = r.random()
        if d > 2 or k < 0.3:
            names = self.bools()
            if names and r.random() < 0.5:
                return r.choice(names)
            return self.bit() if r.random() < 0.8 else r.choice(["true", "false"])
        if k < 0.5:
            return "(%s %s %s)" % (self.bool_expr(d + 1), r.choice(["&&", "||"]), self.bool_expr(d + 1))
        if k < 0.6:
            return "!(%s)" % self.bool_expr(d + 1)
        if k < 0.8:
            return "(%s %s %s)" % (self.int_expr(d + 1), r.choice(["==", "!=", "<", "<=", ">", ">="]), self.int_expr(d + 1))
        if k < 0.95:
            return "(%s %s %s)" % (self.uint_expr(d + 1), r.choice(["==", "!=", "<", "<=", ">", ">="]), self.uint_expr(d + 1))
        return "(c %s %d)" % (r.choice(["==", "!="]), r.randint(0, 3))

    def float_expr(self, d=0):
        r = self.r
        k = r.random()
        if d > 2 or k < 0.35:
            names = self.floats()
            if names and r.random() < 0.7:
                return r.choice(names)
            return r.choice(["0.25", "1.5", "pi/4", "pi", "0.1", "2.0"])
        if k < 0.6:
            return "(%s %s %s)" % (self.float_expr(d + 1), r.choice("+-*"), self.float_expr(d + 1))
        if k < 0.8:
            return "%s(%s)" % (r.choice(["sin", "cos"]), self.float_expr(d + 1))
        if self.ints():
            return "float(%s)" % r.choice(self.ints())
        return "0.5"

    def angle_expr(self, t):
        r = self.r
        same = [n for n, tt in self.vars.items() if tt == t]
        k = r.random()
        if same and k < 0.5:
            return "(%s %s %s)" % (r.choice(same), r.choice("+-"), r.choice(same))
        if same and k < 0.75:
            return "(%s * %d)" % (r.choice(same), r.randint(1, 5))
        return "angle[%d](%s)" % (t[1], self.float_expr(1))

    def declare(self):
        r = self.r
        kind = r.choice(["int", "uint", "bool", "float", "angle"])
        if kind == "int":
            t = ("int", r.choice([3, 4, 8, 16, 32, 64]))
            tn = "int[%d]" % t[1]
            e = self.int_expr()
        elif kind == "uint":
            t = ("uint", r.choice([2, 3, 4, 8, 16, 64]))
            tn = "uint[%d]" % t[1]
            e = self.uint_expr()
        elif kind == "bool":
            t = ("bool",)
            tn = "bool"
            e = self.bool_expr()
        elif kind == "float":
            t = ("float",)
            tn = "float"
            e = self.float_expr()
        else:
            t = ("angle", r.choice([4, 8, 12]))
            tn = "angle[%d]" % t[1]
            e = "angle[%d](%s)" % (t[1], self.float_expr(1))
        nm = self.fresh(t)
        if self.pad == 0 and r.random() < 0.9:
            self.outputs_decl.append("output %s %s;" % (tn, nm))
            self.emit("%s = %s;" % (nm, e))
        else:
            self.emit("%s %s = %s;" % (tn, nm, e))
        self.vars[nm] = t

    def assign(self):
        r = self.r
        names = [n for n in self.vars if not n.startswith("i_")]
        if not names or r.random() < 0.2:
            self.emit("%s = %s;" % (self.bit(), self.bool_expr(1)))
            return
        nm = r.choice(names)
        t = self.vars[nm]
        if t[0] == "int":
            self.emit("%s = %s;" % (nm, self.int_expr()))
        elif t[0] == "uint":
            self.emit("%s = %s;" % (nm, self.uint_expr()))
        elif t[0] == "bool":
            self.emit("%s = %s;" % (nm, self.bool_expr()))
        elif t[0] == "float":
            self.emit("%s = %s;" % (nm, self.float_expr()))
        else:
            self.emit("%s = %s;" % (nm, self.angle_expr(t)))

    def gate(self):
        r = self.r
        q = r.randrange(self.nq)
        k = r.random()
        if k < 0.4:
            self.emit("%s q[%d];" % (r.choice(["h", "x", "z", "s", "t", "sdg", "sx", "y"]), q))
        elif k < 0.75:
            if r.random() < 0.3 and self.angles():
                arg = r.choice(self.angles())
            elif r.random() < 0.3 and self.ints():
                arg = r.choice(self.ints())
            else:
                arg = self.float_expr()
            self.emit("%s(%s) q[%d];" % (r.choice(["rx", "ry", "rz", "p"]), arg, q))
        elif self.nq >= 2:
            a, b = r.sample(range(self.nq), 2)
            if r.random() < 0.8:
                self.emit("%s q[%d], q[%d];" % (r.choice(["cx", "cz", "swap", "cy"]), a, b))
            else:
                self.emit("cp(%s) q[%d], q[%d];" % (self.float_expr(), a, b))
        else:
            self.emit("h q[%d];" % q)

    def measure(self):
        r = self.r
        if r.random() < 0.25:
            self.emit("reset q[%d];" % r.randrange(self.nq))
        else:
            self.emit("%s = measure q[%d];" % (self.bit(), r.randrange(self.nq)))

    def stmts(self, n, depth, in_loop):
        r = self.r
        for _ in range(n):
            k = r.random()
            if depth >= 3 or k < 0.3:
                self.gate()
            elif k < 0.45:
                self.measure()
            elif k < 0.6:
                self.assign()
            elif k < 0.68:
                self.declare()
            elif k < 0.8:
                self.if_stmt(depth, in_loop)
            elif k < 0.88:
                self.while_stmt(depth)
            elif k < 0.94:
                self.for_stmt(depth)
            else:
                self.switch_stmt(depth, in_loop)
            if in_loop and r.random() < 0.08:
                self.emit("if (%s) {" % self.bool_expr(1))
                self.emit("  %s;" % r.choice(["break", "continue"]))
                self.emit("}")

    def scoped(self, f):
        before = dict(self.vars)
        self.pad += 1
        f()
        self.pad -= 1
        self.vars = before

    def if_stmt(self, depth, in_loop):
        self.emit("if (%s) {" % self.bool_expr())
        self.scoped(lambda: self.stmts(self.r.randint(1, 3), depth + 1, in_loop))
        if self.r.random() < 0.5:
            self.emit("} else {")
            self.scoped(lambda: self.stmts(self.r.randint(1, 3), depth + 1, in_loop))
        self.emit("}")

    def while_stmt(self, depth):
        r = self.r
        nm = "w%d" % self.loops
        self.loops += 1
        self.emit("int[16] %s = 0;" % nm)
        cond = "%s < %d" % (nm, r.randint(1, 4))
        if r.random() < 0.6:
            cond = "(%s) && (%s)" % (cond, self.bool_expr(1))
        self.vars[nm] = ("int", 16)
        self.emit("while (%s) {" % cond)
        def body():
            self.emit("%s = %s + 1;" % (nm, nm))
            del self.vars[nm]
            self.stmts(r.randint(1, 3), depth + 1, True)
            self.vars[nm] = ("int", 16)
        self.scoped(body)
        self.emit("}")

    def for_stmt(self, depth):
        r = self.r
        nm = "i_%d" % self.loops
        self.loops += 1
        k = r.random()
        if k < 0.4:
            head = "for int %s in [0:%d] {" % (nm, r.randint(0, 3))
        elif k < 0.7:
            head = "for int %s in [%d:-1:0] {" % (nm, r.randint(1, 3))
        elif k < 0.85:
            head = "for uint[3] %s in {1, 3, 2} {" % nm
        else:
            names = self.ints()
            bound = "(%s %% 3)" % r.choice(names) if names else "2"
            head = "for int %s in [0:%s] {" % (nm, bound)
        self.emit(head)
        t = ("uint", 3) if "uint[3]" in head else ("int", 64)
        def body():
            self.vars[nm] = t
            self.stmts(r.randint(1, 3), depth + 1, True)
        self.scoped(body)
        self.emit("}")

    def switch_stmt(self, depth, in_loop):
        r = self.r
        names = self.ints()
        target = r.choice(names) if names and r.random() < 0.5 else "c[0:%d]" % min(1, self.nc - 1)
        self.emit("switch (%s) {" % target)
        self.pad += 1
        vals = r.sample(range(0, 5), r.randint(1, 3))
        for v in vals:
            self.emit("case %d {" % v)
            self.scoped(lambda: self.stmts(r.randint(1, 2), depth + 1, False))
            self.emit("}")
        if r.random() < 0.6:
            self.emit("default {")
            self.scoped(lambda: self.stmts(1, depth + 1, False))
            self.emit("}")
        self.pad -= 1
        self.emit("}")

    def program(self):
        self.outputs_decl = []
        # superpositions and early measurements, so that branches depend on data
        for q in range(self.nq):
            if self.r.random() < 0.7:
                self.emit("%s(%s) q[%d];" % (self.r.choice(["rx", "ry"]), self.r.choice(["0.7", "1.1", "pi/2", "2.1"]), q))
        for b in range(self.nc):
            if self.r.random() < 0.5:
                self.emit("c[%d] = measure q[%d];" % (b, self.r.randrange(self.nq)))
        for _ in range(self.r.randint(1, 3)):
            self.declare()
        self.stmts(self.r.randint(3, 8), 0, False)
        for b in range(min(self.nq, self.nc)):
            if b == 0 or self.r.random() < 0.7:
                self.emit("c[%d] = measure q[%d];" % (b, b))
        head = ['OPENQASM 3.0;', 'include "stdgates.inc";', "qubit[%d] q;" % self.nq, "bit[%d] c;" % self.nc]
        return "\n".join(head + self.outputs_decl + self.lines) + "\n"


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def read_qumo(path):
    rows = {}
    with open(path) as f:
        first = f.readline().strip()
        if first.startswith("error"):
            return None, first
        names = first.split()[1:]
        for line in f:
            parts = line.split()
            count = int(parts[0])
            key = [parts[1]]
            for p in parts[2:]:
                key.append(round(float(p), 9) if ("." in p or "e" in p or p in ("nan", "inf", "-inf")) else int(p))
            rows[tuple(key)] = rows.get(tuple(key), 0) + count
    return rows, names


def build_runner(root, build_dir):
    exe = os.path.join(build_dir, "control_flow_runner")
    src = os.path.join(root, "bench", "crosscheck", "control_flow_runner.mojo")
    subprocess.run([os.path.join(root, ".venv", "bin", "mojo"), "build", "-I", root, "-o", exe, src],
                   check=True, cwd=root, stderr=subprocess.DEVNULL)
    return exe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", type=int, default=200)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    work = args.dir or tempfile.mkdtemp(prefix="qumo_control_flow_")
    os.makedirs(work, exist_ok=True)
    rng = random.Random(args.seed)
    exact = {}
    skipped = 0
    oracle_refused = 0
    made = 0
    index = 0
    while made < args.programs:
        index += 1
        text = Gen(random.Random(rng.random())).program()
        try:
            prog = openqasm3.parse(text)
            orc = Oracle(prog)
            dist, dropped = orc.run()
        except OracleError as e:
            # outside the generated subset, or an error both sides should raise:
            # kept and compared as an error
            exact["p%05d.qasm" % index] = ("error", str(e))
            with open(os.path.join(work, "p%05d.qasm" % index), "w") as f:
                f.write(text)
            oracle_refused += 1
            made += 1
            continue
        except (KeyError, OverflowError, ValueError, ZeroDivisionError) as e:
            skipped += 1
            print("oracle skipped program:", type(e).__name__, e)
            continue
        name = "p%05d.qasm" % index
        with open(os.path.join(work, name), "w") as f:
            f.write(text)
        exact[name] = (dist, dropped)
        made += 1
    exe = build_runner(root, os.path.join(root, "build"))
    subprocess.run([exe, work, str(args.shots), str(args.seed)], check=True, stderr=subprocess.DEVNULL)
    tol = math.sqrt(math.log(args.programs / 1e-6) / (2 * args.shots))
    worst = 0.0
    bad = []
    both_err = 0
    for name, ex in sorted(exact.items()):
        rows, names = read_qumo(os.path.join(work, name + ".out"))
        if ex[0] == "error":
            if rows is None:
                both_err += 1
            else:
                bad.append((name, "oracle error, QUMO ran: " + ex[1]))
            continue
        if rows is None:
            bad.append((name, "QUMO error, oracle ran: " + names))
            continue
        dist, dropped = ex
        N = args.shots
        keys = set(dist) | set(rows)
        tv = 0.5 * sum(abs(dist.get(k, 0.0) - rows.get(k, 0) / N) for k in keys)
        expect = 0.5 * sum(math.sqrt(2 * p * max(0.0, 1 - p) / (math.pi * N)) for p in dist.values())
        limit = 1.2 * expect + tol + dropped
        worst = max(worst, tv / limit if limit > 0 else 0.0)
        if tv > limit:
            bad.append((name, "TV %.4f > limit %.4f (support %d, dropped %.2e)" % (tv, limit, len(dist), dropped)))
    print("programs %d, exact %d, both raised %d, skipped by the oracle %d, worst TV / limit %.3f"
          % (len(exact), len(exact) - oracle_refused, both_err, skipped, worst))
    for name, why in bad[:20]:
        print("FAIL", os.path.join(work, name), why)
    if not args.keep and not args.dir:
        shutil.rmtree(work, ignore_errors=True)
    print("control_flow crosscheck", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
