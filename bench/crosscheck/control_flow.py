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
(docs/CONTROL_FLOW.md once published), not from qumo/ir/classical.mojo. A slice of
the programs ends in a run-time error: each side must raise QM-E480, E481 or E482
wherever the exact weights say a shot reaches one.

TV acceptance, fixed before the first comparison and not to be tuned to a result:
for N shots and exact probabilities p_k, an unbiased sampler has expected total
variation E = 0.5 * sum sqrt(2 p_k (1 - p_k) / (pi N)), and by McDiarmid TV exceeds
its mean by t only with probability exp(-2 N t^2). A program passes when

    TV <= 1.2 * E + sqrt(ln(n / 1e-6) / (2 N)) + truncated_mass

with n programs in the run (a family-wise false-alarm rate of 1e-6).

Usage: bench/.venv/bin/python bench/crosscheck/control_flow.py [--programs 200]
       [--shots 20000] [--seed 1] [--dir DIR] [--target line:6 --level 2]

With --target, QUMO compiles each program for that target first: the compiled
program, on physical qubits, is what gets compared with the exact distribution.
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
    """A program the type rules refuse: QM-E580 and friends."""

    def __init__(self, what, code="QM-E580"):
        super().__init__(what)
        self.code = code


class RunError(Exception):
    """An error at run time, for the worlds that reach it."""

    def __init__(self, code, what):
        super().__init__(what)
        self.code = code


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
        raise RunError("QM-E482", "float range")
    v = int(x)  # python truncates toward zero
    if t[0] == "int":
        if not (-(1 << (t[1] - 1)) <= v < (1 << (t[1] - 1))):
            raise RunError("QM-E482", "float range")
    elif not (0 <= v < (1 << t[1])):
        raise RunError("QM-E482", "float range")
    return v


def float_to_angle(x, n):
    if math.isnan(x) or math.isinf(x):
        raise RunError("QM-E482", "float range")
    turns = x / (2 * math.pi)
    frac = turns - math.floor(turns)
    t = frac * (1 << n)
    k = math.floor(t)
    d = t - k
    if d > 0.5 or (d == 0.5 and k % 2 == 1):  # a tie goes to the even step
        k += 1
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


class Lit(tuple):
    """A value that came from literals only: it adapts to the other operand."""


def lit(t, v):
    return Lit((t, v))


def is_lit(x):
    return isinstance(x, Lit)


class Exact(tuple):
    """An integer constant with no width, which is not a 64-bit value: it holds any
    integer, and a result past 64 bits goes on as a float."""


def is_exact(x):
    return isinstance(x, Exact) or (is_lit(x) and x[0][0] in ("int", "uint"))


def exact_operand(x):
    return is_exact(x) or (is_lit(x) and x[0][0] in ("bool", "bit"))


def real_of(v):
    try:
        return float(v)
    except OverflowError:
        return math.inf if v > 0 else -math.inf


def exact_const(v, unsigned, literal):
    if v < -(1 << 63) or v >= (1 << 64):
        out = (("float",), real_of(v))
        return lit(*out) if literal else out
    t = ("uint", 64) if v >= (1 << 63) or (unsigned and v >= 0) else ("int", 64)
    return lit(t, v) if literal else Exact((t, v))


def exact_binary(op, a, b):
    x, y = a[1], b[1]
    unsigned = a[0][0] == "uint" or b[0][0] == "uint"
    literal = is_lit(a) and is_lit(b)
    if op in ("==", "!=", "<", "<=", ">", ">="):
        r = {"==": x == y, "!=": x != y, "<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y}[op]
        return lit(("bool",), 1 if r else 0) if literal else (("bool",), 1 if r else 0)
    if op == "+":
        return exact_const(x + y, unsigned, literal)
    if op == "-":
        return exact_const(x - y, unsigned, literal)
    if op == "*":
        return exact_const(x * y, unsigned, literal)
    if op in ("/", "%"):
        if y == 0:
            raise OracleError("a constant zero divisor", "QM-E533")
        q = int_div(x, y)
        return exact_const(q if op == "/" else x - q * y, unsigned, literal)
    if op == "**":
        if y < 0:
            try:
                f = float(x) ** y
            except (ZeroDivisionError, OverflowError):
                f = math.inf
            return lit(("float",), f) if literal else (("float",), f)
        if abs(x) >= 2 and y > 4096:
            f = math.inf
            return lit(("float",), f) if literal else (("float",), f)
        return exact_const(x ** y, unsigned, literal)
    if op == "<<":
        return exact_const(x << y, True, literal)
    if op == ">>":
        return exact_const(x >> y, True, literal)
    if op in ("&", "|", "^"):
        ux, uy = x & ((1 << 64) - 1), y & ((1 << 64) - 1)
        r = ux & uy if op == "&" else (ux | uy if op == "|" else ux ^ uy)
        if x < 0 or y < 0:
            return exact_const(r - (1 << 64) if r >> 63 else r, False, literal)
        return exact_const(r, unsigned, literal)
    raise OracleError("operator " + op)


def width_of(t):
    return 1 if t[0] == "bool" else t[1]


def promoted(val, mode):
    """bool and bit: a 64-bit int in arithmetic, unsigned bits elsewhere."""
    t, v = val
    if t[0] in ("bool", "bit"):
        if mode == "arith":
            return (("int", 64), v if v < (1 << 63) else v - (1 << 64))
        return (("uint", width_of(t)), v)
    return (t, v)


def common(a, b, mode):
    """Both operands in one type, for mode 'arith', 'bit' or 'cmp'."""
    la, lb = is_lit(a), is_lit(b)
    pa, pb = promoted(a, mode), promoted(b, mode)
    ta, tb = pa[0], pb[0]
    bita, bitb = a[0][0] in ("bit", "bool"), b[0][0] in ("bit", "bool")
    if mode == "arith" and bita != bitb:
        other = tb if bita else ta
        if other[0] == "int" or (other[0] == "uint" and other[1] < 64):
            i64 = ("int", 64)
            return cast(tuple(pa), i64), cast(tuple(pb), i64)
    if bita and not lb and (tb[0] == "uint" or (tb[0] == "int" and mode != "arith")):
        n = width_of(a[0])
        ta = ("uint", max(tb[1], n)) if tb[0] == "uint" else ("int", min(64, max(tb[1], n + 1)))
    elif bitb and not la and (ta[0] == "uint" or (ta[0] == "int" and mode != "arith")):
        n = width_of(b[0])
        tb = ("uint", max(ta[1], n)) if ta[0] == "uint" else ("int", min(64, max(ta[1], n + 1)))
    if ta[0] == "float" or tb[0] == "float":
        if ta[0] == "angle" and lb:
            t = ta
        elif tb[0] == "angle" and la:
            t = tb
        elif "angle" in (ta[0], tb[0]):
            raise OracleError("float and angle do not mix")
        else:
            t = ("float",)
    elif la and lb:
        t = ("uint", 64) if "uint" in (ta[0], tb[0]) else ("int", 64)
    elif la or lb:
        lv = pa[1] if la else pb[1]
        t = tb if la else ta
        if t[0] == "angle":
            raise OracleError("an integer literal does not mix with an angle")
        if not literal_fits(lv, t):
            if (t[0] == "uint" and lv < 0) or (t[0] == "int" and lv >= (1 << 63)) or mode != "cmp":
                raise OracleError("a literal that does not fit")
            t = (t[0], max(t[1], smallest_width(lv, t[0] == "int")))
    elif ta[0] != tb[0]:
        raise OracleError("types do not mix")
    elif ta[0] == "angle":
        if ta[1] != tb[1]:
            raise OracleError("angles of two widths")
        t = ta
    else:
        t = ta if ta[1] >= tb[1] else tb
    return cast(tuple(pa), t), cast(tuple(pb), t)


def int_div(a, b):
    if b == 0:
        raise RunError("QM-E481", "division by zero")
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def int_rem(a, b):
    r = abs(a) % abs(b)
    return -r if a < 0 else r


def binary(op, a, b):
    if exact_operand(a) and exact_operand(b) and (is_exact(a) or is_exact(b)):
        shift = op in ("<<", ">>")
        if not shift or (is_exact(a) and (a[0][0] == "uint" or is_lit(a)) and a[1] >= 0 and b[1] >= 0):
            return exact_binary(op, a, b)
    if op in ("==", "!=", "<", "<=", ">", ">="):
        ca, cb = (a, b) if a[0] == ("bool",) and b[0] == ("bool",) else common(a, b, "cmp")
        x, y = ca[1], cb[1]
        r = {"==": x == y, "!=": x != y, "<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y}[op]
        return (("bool",), 1 if r else 0)
    if op in ("<<", ">>", "&", "|", "^") and (a[0][0] == "float" or b[0][0] == "float"):
        raise OracleError("operator on a float")
    if op in ("<<", ">>"):
        if is_lit(a) and a[0][0] == "int" and a[1] >= 0:
            a = lit(("uint", 64), a[1])  # a literal adapts, as it does to any operand
        if a[0][0] in ("int", "bool"):
            raise OracleError("shift of a signed value")
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
            ca, cb = common(a, b, "bit")
            t = ca[0]
        x, y = ca[1], cb[1]
        return wraplit(a, b, (t, wrap(x & y if op == "&" else (x | y if op == "|" else x ^ y), t)))
    # arithmetic
    if "angle" in (a[0][0], b[0][0]):
        return angle_arith(op, a, b)
    if op == "**" and a[0][0] != "float" and b[0][0] != "float":
        base = promoted(a, "arith")
        e = b[1]
        if e < 0:
            raise RunError("QM-E481", "negative exponent")
        t = base[0]
        return wraplit(a, b, (t, wrap(pow(base[1], e, 1 << t[1]), t)))
    ca, cb = common(a, b, "arith")
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
            try:
                return wraplit(a, b, (t, math.pow(x, y)))
            except ValueError:
                return wraplit(a, b, (t, float("nan")))
            except (OverflowError, ZeroDivisionError):
                return wraplit(a, b, (t, float("inf")))
        if y == 0:
            raise RunError("QM-E481", "division by zero")
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
            raise RunError("QM-E481", "division by zero")
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
            raise RunError("QM-E481", "division by zero")
        if tb[0] == "angle":
            return (("uint", n), a[1] // b[1])
        return (angle, a[1] // b[1])
    raise OracleError("angle operator " + op)


FUNCS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "arcsin": math.asin, "arccos": math.acos,
         "arctan": math.atan, "exp": math.exp, "ln": math.log, "sqrt": math.sqrt, "floor": math.floor,
         "ceil": math.ceil}


def real_fn(name, x):
    if math.isnan(x):
        return x
    if math.isinf(x) and name in ("floor", "ceil"):
        return x
    try:
        return float(FUNCS[name](x))
    except ValueError:
        return float("nan")
    except OverflowError:
        return float("inf")


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


def apply_doubly_controlled(st, m, c0, c1, t):
    """m on qubit t where qubits c0 and c1 are 1."""
    idx = [slice(None)] * st.ndim
    idx[c0] = 1
    idx[c1] = 1
    sub = st[tuple(idx)]
    tt = t - (1 if c0 < t else 0) - (1 if c1 < t else 0)
    out = st.copy()
    out[tuple(idx)] = apply_1q(sub, m, tt)
    return out


def apply_cswap(st, c, a, b):
    idx = [slice(None)] * st.ndim
    idx[c] = 1
    sub = st[tuple(idx)]
    aa = a - (1 if c < a else 0)
    bb = b - (1 if c < b else 0)
    out = st.copy()
    out[tuple(idx)] = np.swapaxes(sub, aa, bb)
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
        self.outputs = []
        self.dropped = 0.0
        self.errors = defaultdict(float)  # code -> weight of the worlds that reach it
        self.gates = {}
        self.assigned = set()  # names written after their declaration
        self.scan(program.statements)

    def scan(self, stmts):
        """Registers in source order; a block's bit is a register of its own."""
        for st in stmts:
            if isinstance(st, ast.QuantumGateDefinition):
                self.gates[st.name.name] = st
            elif isinstance(st, ast.ClassicalAssignment):
                target = st.lvalue
                self.assigned.add(target.name if isinstance(target, ast.Identifier) else target.name.name)
            elif isinstance(st, ast.QubitDeclaration):
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
            elif isinstance(st, ast.BranchingStatement):
                self.scan(st.if_block)
                self.scan(st.else_block)
            elif isinstance(st, (ast.WhileLoop, ast.ForInLoop)):
                self.scan(st.block)
            elif isinstance(st, ast.SwitchStatement):
                for _, body in st.cases:
                    self.scan(body.statements)
                if st.default is not None:
                    self.scan(st.default.statements)
            elif isinstance(st, ast.CompoundStatement):
                self.scan(st.statements)

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
                outs.append(fmt_real(v) if t[0] == "float" else int(v))
            dist[(key,) + tuple(outs)] += w.weight
        return dist, self.dropped

    def block(self, stmts, worlds):
        for s in stmts:
            nxt = []
            for w in worlds:
                if w.flow:
                    nxt.append(w)
                    continue
                try:
                    nxt.extend(self.stmt(s, w))
                except RunError as e:
                    self.errors[e.code] += w.weight
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
        if isinstance(s, (ast.Include, ast.QubitDeclaration, ast.QuantumGateDefinition)):
            return [w]
        if isinstance(s, ast.IODeclaration):
            if not isinstance(s.type, ast.BitType):
                t = parse_type(s.type, w.env)
                w.env[s.identifier.name] = (t, 0)
                w.env["__unset_" + s.identifier.name] = True
            return [w]
        if isinstance(s, ast.ClassicalDeclaration):
            if isinstance(s.type, ast.BitType):
                first, size = self.cregs[s.identifier.name]
                for j in range(size):
                    w.bits[first + j] = 0  # a block's bit restarts each time
                if s.init_expression is not None:
                    self.assign_bits(s.identifier.name, None, self.expr(s.init_expression, w), w)
                return [w]
            t = parse_type(s.type, w.env)
            if s.init_expression is not None:
                folds = s.identifier.name not in self.assigned
                w.env[s.identifier.name] = self.declared(self.expr(s.init_expression, w), t, s.type, folds)
            else:
                w.env[s.identifier.name] = (t, 0)
            return [w]
        if isinstance(s, ast.ConstantDeclaration):
            t = parse_type(s.type, w.env)
            w.env[s.identifier.name] = self.declared(self.expr(s.init_expression, w), t, s.type, True)
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

    def declared(self, val, t, node, folds):
        """An unsized integer that folds to a constant stays exact."""
        unsized = isinstance(node, (ast.IntType, ast.UintType)) and node.size is None
        if folds and unsized and is_exact(val):
            return Exact((t, val[1]))
        return self.convert(val, t)

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
        return w.env[name]

    def qubits_of(self, q):
        if isinstance(q, ast.Identifier):
            first, size = self.qregs[q.name]
            return list(range(first, first + size))
        first, size = self.qregs[q.name.name]
        ix = q.indices[0][0]
        return [first + ix.value]

    def literal_only(self, e, params):
        """No name but pi, tau, euler and a gate's own parameters."""
        if isinstance(e, ast.Identifier):
            return e.name in ("pi", "tau", "euler") or e.name in params
        if isinstance(e, (ast.IntegerLiteral, ast.FloatLiteral)):
            return True
        if isinstance(e, ast.BinaryExpression):
            return e.op.name in ("+", "-", "*", "/", "%", "**") and self.literal_only(e.lhs, params) and self.literal_only(e.rhs, params)
        if isinstance(e, ast.UnaryExpression):
            return e.op.name == "-" and self.literal_only(e.expression, params)
        return False

    def real_arg(self, e, params):
        """A literal-only gate argument is real arithmetic, not the classical table."""
        if isinstance(e, (ast.IntegerLiteral, ast.FloatLiteral)):
            return real_of(e.value)
        if isinstance(e, ast.Identifier):
            if e.name in params:
                return params[e.name]
            return {"pi": math.pi, "tau": 2 * math.pi, "euler": math.e}[e.name]
        if isinstance(e, ast.UnaryExpression):
            return -self.real_arg(e.expression, params)
        x, y = self.real_arg(e.lhs, params), self.real_arg(e.rhs, params)
        op = e.op.name
        if op == "+":
            return x + y
        if op == "-":
            return x - y
        if op == "*":
            return x * y
        if op == "/":
            return x / y
        if op == "%":
            return x - y * math.floor(x / y)
        return math.pow(x, y)

    def angle_arg(self, a, w, params):
        if self.literal_only(a, params):
            return self.real_arg(a, params)
        val = self.expr(a, w)
        if val[0][0] == "angle":
            return 2 * math.pi * val[1] / (1 << val[0][1])
        return real_of(val[1])

    def gate(self, g, w):
        qs = [self.qubits_of(q) for q in g.qubits]
        qs = [x[0] for x in qs]
        self.gate_on(g, qs, w, {})

    def gate_on(self, g, qs, w, params):
        nm = g.name.name
        args = []
        for a in g.arguments:
            args.append(self.angle_arg(a, w, params))
            if math.isnan(args[-1]) or math.isinf(args[-1]):
                raise RunError("QM-E482", "a gate reads a NaN or an infinity")
        if nm in self.gates:
            d = self.gates[nm]
            inner = {p.name: v for p, v in zip(d.arguments, args)}
            where = {q.name: x for q, x in zip(d.qubits, qs)}
            env = dict(w.env)
            for k, v in inner.items():
                w.env[k] = lit(("float",), v)  # a gate parameter is an angle
            for body in d.body:
                self.gate_on(body, [where[q.name] for q in body.qubits], w, inner)
            w.env = env
            return
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
        elif nm == "ccx":
            st = apply_doubly_controlled(st, FIXED["x"], qs[0], qs[1], qs[2])
        elif nm == "ccz":
            st = apply_doubly_controlled(st, FIXED["z"], qs[0], qs[1], qs[2])
        elif nm == "cswap":
            st = apply_cswap(st, qs[0], qs[1], qs[2])
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
                try:
                    run = self.convert(self.expr(s.while_condition, x), ("bool",))[1]
                except RunError as e:
                    self.errors[e.code] += x.weight
                    continue
                if not run:
                    done.append(x)
                    continue
                x.loops += 1
                if x.loops > LOOP_CUT:
                    self.errors["QM-E480"] += x.weight  # every loop here ends in a few passes, or never
                elif x.weight < WEIGHT_CUT:
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
        unsized = isinstance(s.type, (ast.IntType, ast.UintType)) and s.type.size is None
        exact = unsized and self.unrolls(s, w, values, t)
        for v in values:
            if not worlds:
                break
            for x in worlds:
                x.env[name] = Exact((t, v)) if exact else (t, wrap(v, t))
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

    def unrolls(self, s, w, values, t):
        """A for with constant bounds whose body neither jumps nor assigns its
        variable is unrolled, and an unsized variable is then a constant."""
        rng = s.set_declaration
        ends = [rng.start, rng.end, rng.step] if isinstance(rng, ast.RangeDefinition) else list(rng.values)
        if any(x is not None and not self.is_constant(x, w) for x in ends):
            return False
        if any(not literal_fits(v, t) for v in values):
            return False
        return not self.jumps_or_assigns(s.block, s.identifier.name)

    def is_constant(self, e, w):
        return isinstance(self.expr(e, w), Exact) or is_lit(self.expr(e, w))

    def jumps_or_assigns(self, stmts, name):
        for st in stmts:
            if isinstance(st, (ast.BreakStatement, ast.ContinueStatement)):
                return True
            if isinstance(st, ast.ClassicalAssignment):
                target = st.lvalue
                if (target.name if isinstance(target, ast.Identifier) else target.name.name) == name:
                    return True
            for field in ("if_block", "else_block", "block", "statements"):
                inner = getattr(st, field, None)
                if inner and self.jumps_or_assigns(inner, name):
                    return True
            if isinstance(st, ast.SwitchStatement):
                bodies = [b.statements for _, b in st.cases]
                if st.default is not None:
                    bodies.append(st.default.statements)
                for b in bodies:
                    if self.jumps_or_assigns(b, name):
                        return True
        return False

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
            if e.value >= (1 << 64):
                return lit(("float",), real_of(e.value))
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
            if is_exact(v):
                return exact_const(-v[1], False, is_lit(v))
            u = promoted(v, "arith")
            if is_lit(v):
                return lit(("int", 64) if u[0][0] == "uint" else u[0], -u[1])
            if u[0][0] == "float":
                return (u[0], -u[1])
            return (u[0], wrap(-u[1], u[0]))  # a uint wraps too
        if isinstance(e, ast.Cast):
            t = parse_type(e.type, w.env)
            v = self.expr(e.argument, w)
            if t[0] == "angle" and v[0][0] in ("int", "uint", "bool"):
                raise OracleError("integer to angle")
            if t[0] == "bit" and v[0][0] != "float":
                if is_lit(v):
                    if not 0 <= v[1] < (1 << t[1]):
                        raise OracleError("a literal that does not fit the bits")
                elif width_of(v[0]) != t[1]:
                    raise OracleError("a cast to bit[m] takes m bits")
            if isinstance(e.type, (ast.IntType, ast.UintType)) and e.type.size is None and is_exact(v):
                return Exact((t, v[1]))
            return self.convert(tuple(v), t)
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
            u = promoted(args[0], "bit")
            x = to_float(u[1], u[0])
            if nm in FUNCS:
                return (("float",), real_fn(nm, x))
            if nm == "mod":
                y = to_float(promoted(args[1], "bit")[1], promoted(args[1], "bit")[0])
                if y == 0:
                    raise RunError("QM-E481", "division by zero")
                return (("float",), x - y * math.floor(x / y))
            raise OracleError("function " + nm)
        raise OracleError("expression " + type(e).__name__)


# --------------------------------------------------------------------------
# Random programs
# --------------------------------------------------------------------------
class Gen:
    """Random programs. Every expression is well typed under the rules above, in
    every branch, so a refusal from either side is a finding."""

    WIDTHS = (2, 3, 4, 8, 16, 32, 64)

    def __init__(self, rng, max_qubits=3, tail=None):
        self.r = rng
        self.nq = rng.randint(1, max_qubits)
        self.nc = rng.randint(2, 5)
        self.vars = {}  # name -> type
        self.lines = []
        self.pad = 0
        self.count = 0
        self.loops = 0
        self.tail = tail
        self.consts = 0
        self.gate_defs = []

    def emit(self, s):
        self.lines.append("  " * self.pad + s)

    def fresh(self, t, prefix="v"):
        nm = prefix + str(self.count)
        self.count += 1
        return nm

    def of(self, tag):
        return [n for n, t in self.vars.items() if t[0] == tag]

    def bit(self):
        return "c[%d]" % self.r.randrange(self.nc)

    def ulit(self, w):
        return str(self.r.randint(0, min(40, (1 << w) - 1)))

    def ileft(self, w):
        names = self.of("int")
        if names and self.r.random() < 0.8:
            n = self.r.choice(names)
            return n if self.vars[n][1] == w else "int[%d](%s)" % (w, n)
        return "int[%d](%d)" % (w, self.r.randint(-1 << min(w - 1, 3), min(7, (1 << (w - 1)) - 1)))

    def uleft(self, w):
        names = self.of("uint")
        if names and self.r.random() < 0.8:
            n = self.r.choice(names)
            return n if self.vars[n][1] == w else "uint[%d](%s)" % (w, n)
        return "uint[%d](%s)" % (w, self.ulit(w))

    def divisor(self, w):
        vals = [v for v in (2, 3, 5, 7) if v < (1 << w) // 2 + (1 << w) // 2 - 1 or v < (1 << (w - 1))]
        return str(self.r.choice(vals)) if vals else "1"

    def int_expr(self, w=None, d=0):
        r = self.r
        w = w or r.choice(self.WIDTHS)
        k = r.random()
        if d > 2 or k < 0.3:
            if r.random() < 0.75:
                return self.ileft(w)
            hi = min(40, (1 << (w - 1)) - 1)
            return str(r.randint(-min(9, 1 << (w - 1)), hi))
        a = self.ileft(w) if r.random() < 0.5 else "int[%d](%s)" % (w, self.int_expr(w, d + 1))
        if k < 0.55:
            return "(%s %s %s)" % (a, r.choice("+-*"), self.int_expr(w, d + 1))
        if k < 0.65:
            sign = r.choice(["", "-"])
            bound = (1 << (w - 1)) - 1
            div = self.divisor(w) if w > 2 else "1"
            if int(div) > bound:
                div, sign = "1", ""
            return "(%s %s %s%s)" % (a, r.choice("/%"), sign, div)
        if k < 0.72:
            return "-(%s)" % a
        if k < 0.78 and self.of("uint"):
            return "int[%d](%s)" % (w, r.choice(self.of("uint")))
        if k < 0.84 and self.of("float"):
            return "int[%d](%s)" % (w, r.choice(self.of("float")))
        if k < 0.92:
            # bool and bit are ints in arithmetic
            names = self.of("bool")
            flag = r.choice(names) if names and r.random() < 0.5 else self.bit()
            form = r.choice(["%s + %s" % (flag, self.bit()), "%s + 1" % flag, "-%s" % flag, "%s * 3" % flag, "%s - 2" % flag])
            return "int[%d](%s)" % (w, form)
        if self.of("uint"):
            e = "uint[2](%s)" % r.choice(self.of("uint"))
        else:
            e = str(r.randint(0, 3))
        return "int[%d](%s ** %s)" % (w, r.choice(["(-2)", "(-3)", "2", "(-1)"]), e)

    def uint_expr(self, w=None, d=0):
        r = self.r
        w = w or r.choice(self.WIDTHS)
        k = r.random()
        if d > 2 or k < 0.3:
            return self.uleft(w) if r.random() < 0.75 else self.ulit(w)
        a = self.uleft(w) if r.random() < 0.5 else "uint[%d](%s)" % (w, self.uint_expr(w, d + 1))
        if k < 0.5:
            return "(%s %s %s)" % (a, r.choice("+-*"), self.uint_expr(w, d + 1))
        if k < 0.62:
            return "(%s %s %s)" % (a, r.choice(["<<", ">>"]), r.choice(["0", "1", "2", "3", "9", "70"]))
        if k < 0.78:
            return "(%s %s %s)" % (a, r.choice("&|^"), self.uint_expr(w, d + 1))
        if k < 0.86:
            return "(%s %s %s)" % (a, r.choice("/%"), self.divisor(w) if w > 2 else "1")
        if k < 0.93:
            return "~%s" % a
        if self.of("int"):
            return "uint[%d](%s)" % (w, r.choice(self.of("int")))
        return "uint[%d](c[0:%d])" % (w, min(2, self.nc - 1))

    def bool_expr(self, d=0):
        r = self.r
        k = r.random()
        if d > 2 or k < 0.3:
            names = self.of("bool")
            if names and r.random() < 0.5:
                return r.choice(names)
            return self.bit() if r.random() < 0.8 else r.choice(["true", "false"])
        if k < 0.5:
            return "(%s %s %s)" % (self.bool_expr(d + 1), r.choice(["&&", "||"]), self.bool_expr(d + 1))
        if k < 0.6:
            return "!(%s)" % self.bool_expr(d + 1)
        ops = ["==", "!=", "<", "<=", ">", ">="]
        if k < 0.76:
            return "(%s %s %s)" % (self.int_expr(None, d + 1), r.choice(ops), self.int_expr(None, d + 1))
        if k < 0.86:
            return "(%s %s %s)" % (self.uint_expr(None, d + 1), r.choice(ops), self.uint_expr(None, d + 1))
        if k < 0.93:
            # a comparison takes a literal too wide for the operand, exactly
            lit_ = r.choice(["300", "65536", "9223372036854775807", "9223372036854775808", "18446744073709551615"])
            return "(%s %s %s)" % (self.uleft(r.choice((2, 8, 64))), r.choice(ops), lit_)
        return "(c %s %d)" % (r.choice(["==", "!="]), r.randint(0, 3))

    def float_expr(self, d=0):
        r = self.r
        k = r.random()
        if d > 2 or k < 0.35:
            names = self.of("float")
            if names and r.random() < 0.7:
                return r.choice(names)
            return r.choice(["0.25", "1.5", "pi/4", "pi", "0.1", "2.0"])
        if k < 0.55:
            return "(%s %s %s)" % (self.float_expr(d + 1), r.choice("+-*"), self.float_expr(d + 1))
        if k < 0.7:
            return "%s(%s)" % (r.choice(["sin", "cos"]), self.float_expr(d + 1))
        if k < 0.78:
            return "(%s ** %s)" % (r.choice(["(-1.5)", "(-2.0)", "1.25"]), r.choice(["2.0", "3.0"]))
        if k < 0.86:
            return "(%s * 1.5)" % self.bit()
        if self.of("int"):
            return "float(%s)" % r.choice(self.of("int"))
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
        kind = r.choice(["int", "uint", "bool", "float", "angle", "big"])
        if kind == "int":
            t = ("int", r.choice(self.WIDTHS))
            tn = "int[%d]" % t[1]
            e = self.int_expr(t[1])
        elif kind == "uint":
            t = ("uint", r.choice(self.WIDTHS))
            tn = "uint[%d]" % t[1]
            e = self.uint_expr(t[1])
        elif kind == "big":
            t = ("uint", 64)
            tn = "uint"
            e = r.choice(["18446744073709551615", "9223372036854775808", "9223372036854775809"])
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
            if kind == "big":
                self.emit("%s = %s;" % (nm, nm))  # keep it a variable: it wraps at run time
            if kind == "float":
                # a folded constant out of range for a cast is refused at parse time,
                # whether or not a shot reaches the cast: keep this one a variable
                self.emit("%s = %s + 0.0;" % (nm, nm))
        self.vars[nm] = t

    def assign(self):
        r = self.r
        names = [n for n in self.vars if not n.startswith("i_") and not n.startswith("w")]
        if not names or r.random() < 0.2:
            self.emit("%s = %s;" % (self.bit(), self.bool_expr(1)))
            return
        nm = r.choice(names)
        t = self.vars[nm]
        if t[0] == "int":
            self.emit("%s = %s;" % (nm, self.int_expr(t[1])))
        elif t[0] == "uint":
            self.emit("%s = %s;" % (nm, self.uint_expr(t[1])))
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
        if k < 0.35:
            self.emit("%s q[%d];" % (r.choice(["h", "x", "z", "s", "t", "sdg", "sx", "y"]), q))
        elif k < 0.65:
            if r.random() < 0.3 and self.of("angle"):
                arg = r.choice(self.of("angle"))
            elif r.random() < 0.3 and self.of("int"):
                arg = r.choice(self.of("int"))
            else:
                arg = self.float_expr()
            self.emit("%s(%s) q[%d];" % (r.choice(["rx", "ry", "rz", "p"]), arg, q))
        elif k < 0.8 and self.nq >= 3:
            a, b, c = r.sample(range(self.nq), 3)
            self.emit("%s q[%d], q[%d], q[%d];" % (r.choice(["ccx", "ccz", "cswap"]), a, b, c))
        elif self.nq >= 2:
            a, b = r.sample(range(self.nq), 2)
            if r.random() < 0.8:
                self.emit("%s q[%d], q[%d];" % (r.choice(["cx", "cz", "swap", "cy"]), a, b))
            else:
                self.emit("cp(%s) q[%d], q[%d];" % (self.float_expr(), a, b))
        else:
            self.emit("h q[%d];" % q)

    def constant_gate(self):
        """Gate arguments from constants: exact integers with no width, literal-only
        real arithmetic, and truncation where a typed name is read."""
        r = self.r
        q = r.randrange(self.nq)
        gate = r.choice(["rx", "ry", "rz", "p"])
        self.consts += 1
        k = "k%d" % self.consts
        form = r.randrange(9)
        if form == 0:  # past 64 bits, in a denominator
            self.emit("const int %s = %d;" % (k, r.randint(60, 90)))
            arg = "pi / 2 ** %s" % k
        elif form == 1:  # a product past 64 bits
            self.emit("const int %s = %d;" % (k, r.randint(10 ** 10, 10 ** 11)))
            arg = "%s * %s * 1e-20" % (k, k)
        elif form == 2:  # a negative exponent gives a float
            self.emit("const int %s = -%d;" % (k, r.randint(1, 4)))
            arg = "%d ** %s" % (r.choice([2, 3, 10]), k)
        elif form == 3:  # int and uint mix, and underflow gives the negative number
            self.emit("const int %s = %d;" % (k, r.randint(0, 9)))
            self.emit("const uint %su = %d;" % (k, r.randint(0, 9)))
            arg = "(%s - %su) / 3.0" % (k, k)
        elif form == 4:  # truncation stays
            self.emit("const int %s = %d;" % (k, r.randint(-9, 9)))
            arg = r.choice(["%s / 2", "%s %% 3", "%s / 2 + 0.5", "(%s + 1) / 4 * pi"]) % k
        elif form == 5:  # a negated uint loop variable
            self.emit("for uint %s in [1:%d] {" % (k, r.randint(1, 3)))
            self.emit("  %s(%s) q[%d];" % (gate, r.choice(["-%s", "1 - %s", "2 ** (-%s)", "pi * 2.0 ** (-%s)"]) % k, q))
            self.emit("}")
            return
        elif form == 6:  # a literal-only argument is real arithmetic
            arg = r.choice(["3/4*pi", "7 % 3", "-7 % 3", "2 ** -1", "2 ** 70 / 1e21", "1/2", "10 ** -3"])
        elif form == 7:  # an integer past 64 bits, through a float
            self.emit("float %s = 2 ** %d;" % (k, r.randint(64, 90)))
            arg = "%s / 1e21" % k
        else:  # a constant read in a gate body, next to a parameter
            self.gate_defs.append("const int %s = %d;" % (k, r.randint(-9, 9)))
            self.gate_defs.append("gate g%s(t) r { rx(%s / 2 + t) r; rz(%s %% 3) r; ry(t / 4) r; rz(3/4*t) r; }" % (k, k, k))
            self.emit("g%s(%s) q[%d];" % (k, r.choice(["0.5", "2", "pi/3"]), q))
            return
        self.emit("%s(%s) q[%d];" % (gate, arg, q))

    def angle_under_branch(self):
        """A variable the gate reads only at run time, set on one path or the other."""
        r = self.r
        nm = self.fresh(("float",))
        self.emit("float %s = %s;" % (nm, r.choice(["0.0", "0.3", "pi/2"])))
        self.emit("if (%s) {" % self.bool_expr(1))
        self.emit("  %s = %s;" % (nm, self.float_expr(1)))
        self.emit("}")
        self.emit("%s(%s) q[%d];" % (r.choice(["rx", "ry", "rz"]), nm, r.randrange(self.nq)))

    def measure(self):
        r = self.r
        if r.random() < 0.25:
            self.emit("reset q[%d];" % r.randrange(self.nq))
        else:
            self.emit("%s = measure q[%d];" % (self.bit(), r.randrange(self.nq)))

    def loop_bit(self):
        """A bit declared in a loop body restarts at 0 on each pass."""
        nm = self.fresh(("bit",), "b")
        self.emit("bit %s;" % nm)
        self.emit("if (%s) {" % nm)
        self.emit("  x q[%d];" % self.r.randrange(self.nq))
        self.emit("}")
        self.emit("%s = %s;" % (nm, self.bool_expr(1)))
        self.emit("if (%s) {" % nm)
        self.emit("  %s q[%d];" % (self.r.choice(["z", "h"]), self.r.randrange(self.nq)))
        self.emit("}")

    def stmts(self, n, depth, in_loop):
        r = self.r
        for _ in range(n):
            k = r.random()
            if r.random() < 0.12:
                self.constant_gate()
            elif depth >= 3 or k < 0.28:
                self.gate()
            elif k < 0.4:
                self.measure()
            elif k < 0.52:
                self.assign()
            elif k < 0.6:
                self.declare()
            elif k < 0.66:
                self.angle_under_branch()
            elif k < 0.71 and in_loop:
                self.loop_bit()
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
        self.emit("while (%s) {" % cond)

        def body():
            self.emit("%s = %s + 1;" % (nm, nm))
            self.stmts(r.randint(1, 3), depth + 1, True)
        self.scoped(body)
        self.emit("}")

    def for_stmt(self, depth):
        r = self.r
        nm = "i_%d" % self.loops
        self.loops += 1
        k = r.random()
        t = ("int", 64)
        if k < 0.3:
            head = "for int %s in [0:%d] {" % (nm, r.randint(0, 3))
        elif k < 0.5:
            head = "for int %s in [%d:-1:0] {" % (nm, r.randint(1, 3))
        elif k < 0.6:
            head = "for uint[3] %s in {1, 3, 2} {" % nm
            t = ("uint", 3)
        elif k < 0.7:
            names = self.of("int")
            bound = "(int(%s) %% 3)" % r.choice(names) if names else "2"
            head = "for int %s in [0:%s] {" % (nm, bound)
        elif k < 0.8:
            # a range the variable cannot hold wraps it, unrolled or not
            head = "for uint[2] %s in [2:5] {" % nm
            t = ("uint", 2)
        else:
            # the ends of int64: the value after the last one does not exist
            top = 9223372036854775807
            low = -9223372036854775808
            form = r.randrange(4)
            if form == 0:
                head = "for int %s in [%d:%d] {" % (nm, top - r.randint(0, 2), top)
            elif form == 1:
                head = "for int %s in [%d:-1:%d] {" % (nm, low + r.randint(0, 2), low)
            elif form == 2:
                head = "for int %s in [%d:%d:%d] {" % (nm, top - 7, r.randint(3, 9), top)
            else:
                lim = "e%d" % self.loops
                self.emit("int %s = %d;" % (lim, top))
                self.emit("%s = %s - 0;" % (lim, lim))
                head = "for int %s in [%d:%s] {" % (nm, top - 1, lim)
        self.emit(head)

        def body():
            self.vars[nm] = t
            self.stmts(r.randint(1, 3), depth + 1, True)
        self.scoped(body)
        self.emit("}")

    def switch_stmt(self, depth, in_loop):
        r = self.r
        names = self.of("int")
        target = r.choice(names) if names and r.random() < 0.5 else "c[0:%d]" % min(1, self.nc - 1)
        self.emit("switch (%s) {" % target)
        self.pad += 1
        vals = r.sample(range(0, 5), r.randint(1, 3))
        for v in vals:
            self.emit("case %d {" % v)
            self.scoped(lambda: self.stmts(r.randint(1, 2), depth + 1, in_loop))
            self.emit("}")
        if r.random() < 0.6:
            self.emit("default {")
            self.scoped(lambda: self.stmts(1, depth + 1, in_loop))
            self.emit("}")
        self.pad -= 1
        self.emit("}")

    def failing_tail(self):
        """End in a run-time error, on every shot or on those of one branch."""
        r = self.r
        n = self.fresh(("int", 64), "t")
        guard = r.random() < 0.3
        if guard:
            self.emit("if (c[0]) {")
            self.pad += 1
        if self.tail == "QM-E481":
            self.emit("int %s = 0;" % n)
            self.emit("%s = %s * 1;" % (n, n))
            self.emit("int %s_r;" % n)
            self.emit("%s_r = %s;" % (n, r.choice(["1 / %s" % n, "5 %% %s" % n])))
        elif self.tail == "QM-E482":
            self.emit("float %s = %s;" % (n, r.choice(["1e300", "-1.0"])))
            self.emit("%s = %s * 10.0;" % (n, n))
            if r.random() < 0.5:
                self.emit("rx(sqrt(%s - 1e308 - 1e308)) q[0];" % n if r.random() < 0.5 else "int[8] %s_r;\n%s_r = int[8](%s * 1e300);" % (n, n, n))
            else:
                self.emit("rz(%s * 1e308 * 1e308) q[0];" % n)
        else:
            self.emit("int %s = 0;" % n)
            self.emit("while (%s >= 0) {" % n)
            self.emit("  %s = %s * 1;" % (n, n))
            self.emit("}")
        if guard:
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
        if self.tail:
            self.failing_tail()
        head = ['OPENQASM 3.0;', 'include "stdgates.inc";', "qubit[%d] q;" % self.nq, "bit[%d] c;" % self.nc]
        return "\n".join(head + self.gate_defs + self.outputs_decl + self.lines) + "\n"


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def fmt_real(v):
    """Nine significant digits: the two sides use different libms."""
    return "%.9g" % v


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
                key.append(fmt_real(float(p)) if ("." in p or "e" in p or p in ("nan", "inf", "-inf")) else int(p))
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
    ap.add_argument("--qubits", type=int, default=3)
    ap.add_argument("--failing", type=float, default=0.1, help="share of programs that end in a run-time error")
    ap.add_argument("--target", default=None)
    ap.add_argument("--level", type=int, default=2)
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    work = args.dir or tempfile.mkdtemp(prefix="qumo_control_flow_")
    os.makedirs(work, exist_ok=True)
    rng = random.Random(args.seed)
    exact = {}
    skipped = 0
    refused = 0
    tails = ["QM-E480", "QM-E481", "QM-E482"]
    index = 0
    while len(exact) < args.programs:
        index += 1
        name = "p%05d.qasm" % index
        sub = random.Random(rng.random())
        tail = sub.choice(tails) if sub.random() < args.failing else None
        text = Gen(sub, args.qubits, tail).program()
        try:
            orc = Oracle(openqasm3.parse(text))
            dist, dropped = orc.run()
            errors = dict(orc.errors)
        except OracleError as e:
            # the type rules refuse it, on a path the run reached: QUMO must too
            exact[name] = ("refused", e.code, str(e))
            refused += 1
        except (KeyError, OverflowError, ValueError, ZeroDivisionError) as e:
            skipped += 1
            print("oracle skipped program:", type(e).__name__, e)
            continue
        else:
            exact[name] = (dist, dropped, errors)
        with open(os.path.join(work, name), "w") as f:
            f.write(text)
    exe = build_runner(root, os.path.join(root, "build"))
    cmd = [exe, work, str(args.shots), str(args.seed)]
    if args.target:
        cmd += [args.target, str(args.level)]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    tol = math.sqrt(math.log(args.programs / 1e-6) / (2 * args.shots))
    worst = 0.0
    bad = []
    raised = defaultdict(int)
    for name, ex in sorted(exact.items()):
        rows, info = read_qumo(os.path.join(work, name + ".out"))
        code = info.split()[1].rstrip(":") if rows is None else None
        if ex[0] == "refused":
            if code == ex[1]:
                raised[code] += 1
            else:
                bad.append((name, "the oracle refuses it (%s), QUMO: %s" % (ex[2], code or "ran")))
            continue
        dist, dropped, errors = ex
        heavy = [c for c, w in errors.items() if w >= 1e-3]
        if code is not None:
            if code in errors:
                raised[code] += 1
            else:
                bad.append((name, "QUMO error, oracle ran: " + info))
            continue
        if heavy:
            bad.append((name, "the oracle reaches %s with weight %.3g, QUMO ran" % (heavy[0], errors[heavy[0]])))
            continue
        dropped += sum(errors.values())
        N = args.shots
        keys = set(dist) | set(rows)
        tv = 0.5 * sum(abs(dist.get(k, 0.0) - rows.get(k, 0) / N) for k in keys)
        expect = 0.5 * sum(math.sqrt(2 * p * max(0.0, 1 - p) / (math.pi * N)) for p in dist.values())
        limit = 1.2 * expect + tol + dropped
        worst = max(worst, tv / limit if limit > 0 else 0.0)
        if tv > limit:
            bad.append((name, "TV %.4f > limit %.4f (support %d, dropped %.2e)" % (tv, limit, len(dist), dropped)))
    both = sum(raised.values())
    codes = " ".join("%s:%d" % kv for kv in sorted(raised.items()))
    print("programs %d, ran %d, both raised %d (%s), skipped by the oracle %d, worst TV / limit %.3f"
          % (len(exact), len(exact) - both, both, codes, skipped, worst))
    report = os.path.join(work, "verify.txt")
    if os.path.exists(report):
        print(open(report).read().strip())
    for name, why in bad[:20]:
        print("FAIL", os.path.join(work, name), why)
    if not args.keep and not args.dir:
        shutil.rmtree(work, ignore_errors=True)
    print("control_flow crosscheck", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
