"""Multi-controlled gates, expanded to existing gates at build time.

`Instruction` holds at most 3 qubits, so a k-controlled gate is synthesised
immediately into the ordinary gate set (no new opcode, every pass and simulator
works unchanged). `Circuit.controlled` adds one control at a time and stops at
3 qubits; `controlled_by` here takes any number of controls over any unitary
circuit, and the OpenQASM parser uses it for `ctrl(k) @` / `negctrl(k) @`.

Synthesis, by available ancillas (CX counts after {u, cx} translation):

| mode | needs | CX | origin |
|---|---|---|---|
| `noaux` | nothing | 2^(k+1) - 2 up to k = 5, then about 8 k^2 | Gray-code phase polynomial; Vale et al. arXiv:2302.06377 recursion |
| `clean` | 1 clean ancilla | 6 (k - 1) | conditionally clean ancillas, Khattar-Gidney arXiv:2407.17966 |
| `dirty` | k - 2 dirty ancillas | 8 k - 6 | Iten et al. arXiv:1501.06911 |
| `split` | 1 dirty ancilla | about 16 k | Barenco et al. quant-ph/9503016, Lemma 7.3 |

`auto` builds the candidates the ancillas allow and keeps the cheapest. Dirty
ancillas are restored exactly; clean ones must start in |0> and end in |0>.

Angles only ever get scaled, so symbolic `Param`s pass through (except `u`,
whose phase needs phi + lambda).
"""

from std.math import pi
from std.bit import pop_count
from qumo.ir.gates import (
    Op, op_name, op_is_annotation, op_is_unitary,
)
from qumo.ir.param import Param
from qumo.ir.circuit import Circuit


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _rccx(mut c: Circuit, a: Int, b: Int, t: Int):
    """Margolus relative-phase Toffoli: Toffoli up to a diagonal, 3 CX, self-inverse."""
    c.h(t)
    c.t(t)
    c.cx(b, t)
    c.tdg(t)
    c.cx(a, t)
    c.t(t)
    c.cx(b, t)
    c.tdg(t)
    c.h(t)


def _half_down(mut c: Circuit, tau: Int, a: Int, b: Int):
    c.h(tau)
    c.t(tau)
    c.cx(a, tau)
    c.tdg(tau)
    c.cx(b, tau)


def _half_up(mut c: Circuit, tau: Int, a: Int, b: Int):
    c.cx(b, tau)
    c.t(tau)
    c.cx(a, tau)
    c.tdg(tau)
    c.h(tau)


def _mcp_gray(mut c: Circuit, lam: Param, qubits: List[Int]):
    """Phase lam on |1...1> of `qubits`: every parity once, 2^n - 2 CX."""
    var n = len(qubits)
    var theta = lam * (1.0 / Float64(1 << (n - 1)))
    for j in range(n):
        var t = qubits[j]
        c.p(theta, t)
        var prev = 0
        for i in range(1, 1 << j):
            var g = i ^ (i >> 1)
            var diff = g ^ prev
            var b = 0
            while (diff >> b) & 1 == 0:
                b += 1
            c.cx(qubits[b], t)
            if pop_count(g) % 2 == 0:
                c.p(theta, t)
            else:
                c.p(-theta, t)
            prev = g
        if j > 0:
            c.cx(qubits[j - 1], t)


def _mcx_dirty(mut c: Circuit, controls: List[Int], target: Int, dirty: List[Int]):
    """k >= 3 controls, k - 2 dirty ancillas: 8 k - 6 CX (only the outer pair is exact)."""
    var k = len(controls)
    for round in range(2):
        var top = k - 1 - round
        var i = top
        while i >= 2:
            var tau = target if i == k - 1 else dirty[i - 1]
            if i == k - 1:
                c.ccx(controls[i], dirty[i - 2], tau)
            else:
                _half_down(c, tau, controls[i], dirty[i - 2])
            i -= 1
        _rccx(c, controls[0], controls[1], dirty[0])
        for j in range(2, top + 1):
            var tau = target if j == k - 1 else dirty[j - 1]
            if j == k - 1:
                c.ccx(controls[j], dirty[j - 2], tau)
            else:
                _half_up(c, tau, controls[j], dirty[j - 2])


# ---------------------------------------------------------------------------
# Conditionally clean ancillas
# ---------------------------------------------------------------------------
@fieldwise_init
struct _AndStep(ImplicitlyCopyable):
    """f ^= p AND q; `flip`: f is |1> when it matters, so X it first."""

    var p: Int
    var q: Int
    var f: Int
    var flip: Bool


def _plan_ands(controls: List[Int], clean: List[Int], mut steps: List[_AndStep]) -> List[Int]:
    """Reduce the controls to the fewest qubits whose AND is the same.

    Invariant: the AND of the active qubits equals the AND of the controls.
    One Toffoli (p, q -> f) replaces p, q by f when f has a known value whenever
    the qubits in cond[f], all of them staying active, are 1. Then p and q are
    known (both 1) whenever f and cond[f] are 1, so each step frees two qubits.
    Compute steps may use relative-phase Toffolis: U^dag S U cancels the phases
    of any phase-permutation U when S leaves U's output basis unchanged.
    """
    var n = len(controls) + len(clean)
    var qubit = List[Int](capacity=n)
    for q in controls:
        qubit.append(q)
    for q in clean:
        qubit.append(q)
    var active = List[Bool](length=n, fill=False)
    var free = List[Bool](length=n, fill=False)
    var one = List[Bool](length=n, fill=False)
    var age = List[Int](length=n, fill=0)
    var cond = List[List[Bool]]()
    for i in range(n):
        cond.append(List[Bool](length=n, fill=False))
        if i < len(controls):
            active[i] = True
        else:
            free[i] = True
    var na = len(controls)
    var clock = 0
    while na > 2:
        # take the most constrained free qubit that still leaves a pair
        var best = -1
        var best_size = -1
        for f in range(n):
            if not free[f]:
                continue
            var size = 0
            var outside = 0
            for i in range(n):
                if cond[f][i]:
                    size += 1
                elif active[i]:
                    outside += 1
            if outside >= 2 and size > best_size:
                best = f
                best_size = size
        if best < 0:
            break
        # oldest two active qubits outside cond[best]
        var p = -1
        var q = -1
        for i in range(n):
            if not active[i] or cond[best][i]:
                continue
            if p < 0 or age[i] < age[p]:
                q = p
                p = i
            elif q < 0 or age[i] < age[q]:
                q = i
        steps.append(_AndStep(qubit[p], qubit[q], qubit[best], one[best]))
        clock += 1
        for g in range(n):
            if free[g] and g != best and (cond[g][p] or cond[g][q]):
                cond[g][p] = False
                cond[g][q] = False
                cond[g][best] = True
                for i in range(n):
                    if cond[best][i]:
                        cond[g][i] = True
        for m in [p, q]:
            active[m] = False
            free[m] = True
            one[m] = True
            for i in range(n):
                cond[m][i] = cond[best][i]
            cond[m][best] = True
        free[best] = False
        active[best] = True
        age[best] = clock
        for i in range(n):
            cond[best][i] = False
        na -= 1
    var out = List[Int]()
    for i in range(n):
        if active[i]:
            out.append(qubit[i])
    return out^


def _emit_compute(mut c: Circuit, steps: List[_AndStep]):
    for s in steps:
        if s.flip:
            c.x(s.f)
        _rccx(c, s.p, s.q, s.f)


def _emit_uncompute(mut c: Circuit, steps: List[_AndStep]):
    var i = len(steps) - 1
    while i >= 0:
        var s = steps[i]
        _rccx(c, s.p, s.q, s.f)
        if s.flip:
            c.x(s.f)
        i -= 1


# ---------------------------------------------------------------------------
# Multi-controlled phase and rotation
# ---------------------------------------------------------------------------
comptime _GRAY_MAX_CONTROLS = 5


def _head(xs: List[Int], n: Int) -> List[Int]:
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(xs[i])
    return out^


def _tail(xs: List[Int], start: Int) -> List[Int]:
    var out = List[Int](capacity=len(xs) - start)
    for i in range(start, len(xs)):
        out.append(xs[i])
    return out^


def _mcrz_noaux(mut c: Circuit, theta: Param, controls: List[Int], target: Int) raises:
    """k-controlled RZ, linear in k, no ancilla: X S X S^dag X S X S^dag = RZ(theta)
    for S = RZ(-theta/4), with the X gates controlled by one half of the controls
    and by the other half in turn; any half off cancels the four rotations."""
    var k = len(controls)
    if k == 0:
        c.rz(theta, target)
        return
    if k == 1:
        c.crz(theta, controls[0], target)
        return
    var k1 = (k + 1) // 2
    var part1 = _head(controls, k1)
    var part2 = _tail(controls, k1)
    var quarter = theta * 0.25
    for _ in range(2):
        append_mcx(c, part1, target, part2, "auto", False)
        c.rz(-quarter, target)
        append_mcx(c, part2, target, part1, "auto", False)
        c.rz(quarter, target)


def _mcp_noaux(mut c: Circuit, lam: Param, controls: List[Int], target: Int) raises:
    var k = len(controls)
    if k == 0:
        c.p(lam, target)
    elif k == 1:
        c.cp(lam, controls[0], target)
    elif k <= _GRAY_MAX_CONTROLS:
        var all = controls.copy()
        all.append(target)
        _mcp_gray(c, lam, all)
    else:
        # P(lam) = e^{i lam/2} RZ(lam); the phase is a gate on the controls
        _mcrz_noaux(c, lam, controls, target)
        _mcp_noaux(c, lam * 0.5, _head(controls, k - 1), controls[k - 1])


def _check_distinct(controls: List[Int], target: Int, ancillas: List[Int]) raises:
    var seen = controls.copy()
    seen.append(target)
    for q in ancillas:
        seen.append(q)
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            if seen[i] == seen[j]:
                raise Error("QM-E204: multi-controlled gate uses qubit " + String(seen[i]) + " twice. Hint: controls, target and ancillas must all be distinct.")


def append_mcp(mut c: Circuit, lam: Param, controls: List[Int], target: Int,
               ancillas: List[Int] = List[Int](), clean: Bool = False) raises:
    """Phase `lam` on the state with every control and the target 1.

    With clean ancillas the controls are ANDed first (`clean = False` ignores
    `ancillas`: a phase gate cannot borrow dirty qubits this way)."""
    _check_distinct(controls, target, ancillas)
    if len(ancillas) >= 1 and clean and len(controls) >= 3:
        var steps = List[_AndStep]()
        var active = _plan_ands(controls, ancillas, steps)
        _emit_compute(c, steps)
        _mcp_noaux(c, lam, active, target)
        _emit_uncompute(c, steps)
        return
    _mcp_noaux(c, lam, controls, target)


def append_mcrz(mut c: Circuit, theta: Param, controls: List[Int], target: Int,
                ancillas: List[Int] = List[Int](), clean: Bool = False) raises:
    """RZ(theta) on `target` when every control is 1."""
    _check_distinct(controls, target, ancillas)
    if len(ancillas) >= 1 and clean and len(controls) >= 3:
        var steps = List[_AndStep]()
        var active = _plan_ands(controls, ancillas, steps)
        _emit_compute(c, steps)
        _mcrz_noaux(c, theta, active, target)
        _emit_uncompute(c, steps)
        return
    _mcrz_noaux(c, theta, controls, target)


# ---------------------------------------------------------------------------
# Multi-controlled X
# ---------------------------------------------------------------------------
def _two_qubit_cost(c: Circuit) -> Int:
    """CX count after {u, cx} translation."""
    var n = 0
    for g in c.ops:
        if g.kind == Op.CCX or g.kind == Op.CCZ:
            n += 6
        elif g.kind == Op.CSWAP:
            n += 8
        elif g.kind == Op.CX or g.kind == Op.CZ or g.kind == Op.CY:
            n += 1
        elif op_is_unitary(g.kind) and g.kind >= Op.CH and g.kind <= Op.RZZ:
            n += 2
    return n


def _mcx_noaux(mut c: Circuit, controls: List[Int], target: Int) raises:
    c.h(target)
    _mcp_noaux(c, Float64(pi), controls, target)
    c.h(target)


def _mcx_clean(mut c: Circuit, controls: List[Int], target: Int, clean: List[Int]) raises:
    var steps = List[_AndStep]()
    var active = _plan_ands(controls, clean, steps)
    _emit_compute(c, steps)
    append_mcx(c, active, target)
    _emit_uncompute(c, steps)


def _mcx_split(mut c: Circuit, controls: List[Int], target: Int, dirty: List[Int]) raises:
    """Barenco Lemma 7.3: one dirty qubit d splits the controls in two halves,
    each half borrowing the other: (C^a X -> d, C^(b+1) X -> target) twice."""
    var k = len(controls)
    var d = dirty[0]
    var a = (k + 1) // 2
    var part1 = _head(controls, a)
    var part2 = _tail(controls, a)
    var upper = part2.copy()
    upper.append(d)
    var borrow1 = part2.copy()
    borrow1.append(target)
    for _ in range(2):
        append_mcx(c, part1, d, borrow1, "auto", False)
        append_mcx(c, upper, target, part1, "auto", False)


def append_mcx(mut c: Circuit, controls: List[Int], target: Int,
               ancillas: List[Int] = List[Int](), mode: String = "auto", clean: Bool = False) raises:
    """X on `target` when every control is 1.

    `mode`: auto (cheapest the ancillas allow), noaux, clean (1 clean ancilla),
    dirty (k - 2 dirty ancillas), split (1 dirty ancilla). `clean` says whether
    `ancillas` start in |0>; False is always safe, they are restored either way.
    """
    _check_distinct(controls, target, ancillas)
    var k = len(controls)
    if mode != "auto" and mode != "noaux" and mode != "clean" and mode != "dirty" and mode != "split":
        raise Error("QM-E205: unknown multi-controlled synthesis mode '" + mode + "'. Hint: use auto, noaux, clean, dirty or split.")
    if k == 0:
        c.x(target)
        return
    if k == 1:
        c.cx(controls[0], target)
        return
    if k == 2:
        c.ccx(controls[0], controls[1], target)
        return
    var m = mode
    if m == "auto":
        m = "noaux"
        var best = _two_qubit_cost(_scratch_mcx(c, controls, target, ancillas, m))
        var options = List[String]()
        if clean and len(ancillas) >= 1:
            options.append("clean")
        else:
            if len(ancillas) >= k - 2:
                options.append("dirty")
            if len(ancillas) >= 1:
                options.append("split")
        for o in options:
            var cost = _two_qubit_cost(_scratch_mcx(c, controls, target, ancillas, o))
            if cost < best:
                best = cost
                m = o
    if m == "noaux":
        _mcx_noaux(c, controls, target)
    elif m == "clean":
        if len(ancillas) < 1:
            raise Error("QM-E206: mcx mode 'clean' needs 1 clean ancilla, got 0. Hint: pass ancillas, or use mode 'auto'.")
        _mcx_clean(c, controls, target, ancillas)
    elif m == "dirty":
        if len(ancillas) < k - 2:
            raise Error("QM-E206: mcx mode 'dirty' with " + String(k) + " controls needs " + String(k - 2) + " ancillas, got " + String(len(ancillas)) + ". Hint: pass more ancillas, or use mode 'auto'.")
        _mcx_dirty(c, controls, target, ancillas)
    else:
        if len(ancillas) < 1:
            raise Error("QM-E206: mcx mode 'split' needs 1 ancilla, got 0. Hint: pass ancillas, or use mode 'auto'.")
        _mcx_split(c, controls, target, ancillas)


def _scratch_mcx(c: Circuit, controls: List[Int], target: Int, ancillas: List[Int], mode: String) raises -> Circuit:
    var s = Circuit(c.num_qubits)
    append_mcx(s, controls, target, ancillas, mode, False)
    return s^


# ---------------------------------------------------------------------------
# Controlling a whole circuit
# ---------------------------------------------------------------------------
@fieldwise_init
struct _Mc(Copyable):
    """Controls and ancillas of one `controlled_by` call."""

    var ctrl: List[Int]
    var anc: List[Int]
    var clean: Bool

    def with_ctrl(self, q: Int) -> List[Int]:
        var out = self.ctrl.copy()
        out.append(q)
        return out^

    def with_ctrls(self, q: Int, r: Int) -> List[Int]:
        var out = self.with_ctrl(q)
        out.append(r)
        return out^

    def x(self, mut c: Circuit, controls: List[Int], t: Int) raises:
        append_mcx(c, controls, t, self.anc, "auto", self.clean)

    def p(self, mut c: Circuit, lam: Param, controls: List[Int], t: Int) raises:
        append_mcp(c, lam, controls, t, self.anc, self.clean)

    def rz(self, mut c: Circuit, theta: Param, controls: List[Int], t: Int) raises:
        append_mcrz(c, theta, controls, t, self.anc, self.clean)

    def rx(self, mut c: Circuit, theta: Param, controls: List[Int], t: Int) raises:
        c.h(t)
        self.rz(c, theta, controls, t)
        c.h(t)

    def ry(self, mut c: Circuit, theta: Param, controls: List[Int], t: Int) raises:
        c.sdg(t)
        c.h(t)
        self.rz(c, theta, controls, t)
        c.h(t)
        c.s(t)

    def phase(self, mut c: Circuit, lam: Param, controls: List[Int]) raises:
        """Phase on the controls alone (a global phase of the controlled gate)."""
        var k = len(controls)
        if k == 1:
            c.p(lam, controls[0])
        else:
            self.p(c, lam, _head(controls, k - 1), controls[k - 1])


def _controlled_gate(mut out: Circuit, m: _Mc, kind: Int, a: Int, b: Int, d: Int,
                     p0: Param, p1: Param, p2: Param) raises:
    """`kind` on qubits a, b, d under every control of `m`."""
    var cs = m.ctrl.copy()
    if kind == Op.I:
        return
    if kind == Op.X:
        m.x(out, cs, a)
    elif kind == Op.Y:
        out.sdg(a)
        m.x(out, cs, a)
        out.s(a)
    elif kind == Op.Z:
        m.p(out, Float64(pi), cs, a)
    elif kind == Op.S:
        m.p(out, Float64(pi / 2), cs, a)
    elif kind == Op.SDG:
        m.p(out, Float64(-pi / 2), cs, a)
    elif kind == Op.T:
        m.p(out, Float64(pi / 4), cs, a)
    elif kind == Op.TDG:
        m.p(out, Float64(-pi / 4), cs, a)
    elif kind == Op.P:
        m.p(out, p0, cs, a)
    elif kind == Op.RZ:
        m.rz(out, p0, cs, a)
    elif kind == Op.RX:
        m.rx(out, p0, cs, a)
    elif kind == Op.RY:
        m.ry(out, p0, cs, a)
    elif kind == Op.H:
        out.ry(Float64(-pi / 4), a)
        m.p(out, Float64(pi), cs, a)
        out.ry(Float64(pi / 4), a)
    elif kind == Op.SX or kind == Op.SXDG:
        var sign = 1.0 if kind == Op.SX else -1.0
        m.rx(out, Float64(sign * pi / 2), cs, a)
        m.phase(out, Float64(sign * pi / 4), cs)
    elif kind == Op.U:
        # U(t, f, l) = e^{i(f+l)/2} RZ(f) RY(t) RZ(l)
        var ok = False
        var sum = p1.try_add(p2, ok)
        if not ok:
            raise Error("QM-E202: controlled u with two different symbolic angles has no affine phase. Hint: bind the parameters first.")
        m.rz(out, p2, cs, a)
        m.ry(out, p0, cs, a)
        m.rz(out, p1, cs, a)
        m.phase(out, sum * 0.5, cs)
    elif kind == Op.CX:
        m.x(out, m.with_ctrl(a), b)
    elif kind == Op.CY:
        out.sdg(b)
        m.x(out, m.with_ctrl(a), b)
        out.s(b)
    elif kind == Op.CZ:
        m.p(out, Float64(pi), m.with_ctrl(a), b)
    elif kind == Op.CP:
        m.p(out, p0, m.with_ctrl(a), b)
    elif kind == Op.CRZ:
        m.rz(out, p0, m.with_ctrl(a), b)
    elif kind == Op.CRX:
        m.rx(out, p0, m.with_ctrl(a), b)
    elif kind == Op.CRY:
        m.ry(out, p0, m.with_ctrl(a), b)
    elif kind == Op.CH:
        out.ry(Float64(-pi / 4), b)
        m.p(out, Float64(pi), m.with_ctrl(a), b)
        out.ry(Float64(pi / 4), b)
    elif kind == Op.SWAP:
        out.cx(b, a)
        m.x(out, m.with_ctrl(a), b)
        out.cx(b, a)
    elif kind == Op.CCX:
        m.x(out, m.with_ctrls(a, b), d)
    elif kind == Op.CCZ:
        m.p(out, Float64(pi), m.with_ctrls(a, b), d)
    elif kind == Op.CSWAP:
        out.cx(d, b)
        m.x(out, m.with_ctrls(a, b), d)
        out.cx(d, b)
    elif kind == Op.RZZ:
        out.cx(a, b)
        m.rz(out, p0, cs, b)
        out.cx(a, b)
    elif kind == Op.RXX:
        out.h(a)
        out.h(b)
        out.cx(a, b)
        m.rz(out, p0, cs, b)
        out.cx(a, b)
        out.h(a)
        out.h(b)
    elif kind == Op.RYY:
        out.rx(Float64(pi / 2), a)
        out.rx(Float64(pi / 2), b)
        out.cx(a, b)
        m.rz(out, p0, cs, b)
        out.cx(a, b)
        out.rx(Float64(-pi / 2), a)
        out.rx(Float64(-pi / 2), b)
    elif kind == Op.ISWAP or kind == Op.ISWAPDG:
        var theta = Float64(-pi / 2) if kind == Op.ISWAP else Float64(pi / 2)
        _controlled_gate(out, m, Op.RXX, a, b, d, theta, 0.0, 0.0)
        _controlled_gate(out, m, Op.RYY, a, b, d, theta, 0.0, 0.0)
    elif kind == Op.ECR:
        # ecr = rzx(pi/4) a, b; x a; rzx(-pi/4) a, b (Qiskit definition)
        for s in range(2):
            out.h(b)
            out.cx(a, b)
            m.rz(out, Float64(pi / 4 if s == 0 else -pi / 4), cs, b)
            out.cx(a, b)
            out.h(b)
            if s == 0:
                m.x(out, cs, a)
    else:
        raise Error("QM-E202: controlled_by() does not support gate '" + op_name(kind) + "'. Hint: run qumo.compiler basis translation to {u, cx} first.")


def controlled_by(sub: Circuit, controls: List[Int], ancillas: List[Int] = List[Int](),
                  clean: Bool = False) raises -> Circuit:
    """`sub` under every qubit of `controls` (any number, any unitary gates).

    Ancillas must be outside `sub` and `controls`; see `append_mcx` for `clean`.
    Gates are controlled one by one. A compute-uncompute shape in `sub` is not
    detected, so hand-written `V (ctrl U) V^dag` can be much cheaper.
    """
    if len(controls) == 0:
        return sub.copy()
    var nq = sub.num_qubits
    for q in controls:
        nq = max(nq, q + 1)
    for q in ancillas:
        nq = max(nq, q + 1)
    var out = sub.empty_like()
    out.num_qubits = nq
    out.global_phase = 0.0
    var m = _Mc(controls.copy(), ancillas.copy(), clean)
    if sub.global_phase != 0.0:
        m.phase(out, sub.global_phase, m.ctrl)
    for g in sub.ops:
        if op_is_annotation(g.kind):
            continue
        if not op_is_unitary(g.kind) or g.cond >= 0:
            raise Error("QM-E202: controlled_by() needs a unitary circuit without classical conditions, found '" + op_name(g.kind) + "'. Hint: remove measurements, resets, noise and conditions from the controlled part.")
        for q in controls:
            if g.acts_on(q):
                raise Error("QM-E203: control qubit " + String(q) + " is already used by the circuit.")
        for q in ancillas:
            if g.acts_on(q):
                raise Error("QM-E203: ancilla qubit " + String(q) + " is already used by the circuit.")
        _controlled_gate(out, m, g.kind, g.q0, g.q1, g.q2, g.p0, g.p1, g.p2)
    return out^
