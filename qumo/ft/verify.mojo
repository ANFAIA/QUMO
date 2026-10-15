"""Exact verification of logical gadgets with QUMO's stabilizer tableau.

Two checks, the analogues of Guppy FT's `valid_clifford_implementation` and
`valid_stabilizer_state_preparation`:

* `valid_clifford_implementation(semantic, impl, code, num_blocks)`:
  `semantic` is a Clifford circuit on `num_blocks * k` logical qubits,
  `impl` a unitary Clifford circuit on `num_blocks * n` physical qubits
  (+ optional clean ancillas that must return to |0>). The implementation is
  valid iff (Heisenberg picture, C = impl, U = semantic)
    - C^dag g C is in the stabilizer group for every code generator g of every
      block (and every ancilla Z), i.e. C preserves the code space;
    - C^dag L(P) C = L(U^dag P U) * s for some stabilizer s, *with the sign*,
      for P = X_j, Z_j of every logical qubit j, where L() expands a logical
      Pauli with the code's logical operators.
  This determines the logical action up to a global phase, which is exactly
  what Guppy FT's Choi-state comparison establishes; we read C^dag P C straight
  from the rows of `qumo.sim.stabilizer.TableauSimulator` (it stores the
  inverse tableau), so no Choi state has to be prepared.

* `valid_stabilizer_state_preparation(semantic, impl, code, ...)`: `semantic`
  prepares a stabilizer state of `num_blocks * k` qubits from |0..0>; `impl`
  may use ancillas, measurements and resets (flag qubits). The impl is run on
  the tableau with every random outcome forced to 0 and with `trials` random
  seeds; on every accepted run (all measurement records 0, i.e. no flag fired)
  each expected generator - the expanded semantic stabilizers plus the code
  generators of every block - must have expectation exactly +1.

Every failure carries a human-readable reason (`VerifyReport.reason`) naming
the operator that breaks (e.g. "Z_1 maps to -Z0 Z1 instead of +Z0 Z1"), which
is how the deliberately wrong gadgets in tests/test_ft_verify.mojo are caught.

Idea from Guppy FT (Apache-2.0), files src/guppyft/verify/_verify.py,
_expansion.py and _state_gen.py; re-derived (Heisenberg picture instead of
Choi states).

Error codes: QM-E723 (gadget/semantic circuit not suitable for verification).
"""

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_name, op_is_unitary, op_is_annotation, op_is_noise
from qumo.sim.stabilizer import TableauSimulator
from qumo.sim.rng import Rng
from qumo.ft.pauli import SymPauli, StabilizerGroup
from qumo.ft.code import StabilizerCode
from qumo.ai.json import JsonWriter


struct VerifyReport(Copyable, Writable):
    var ok: Bool
    var reason: String
    var checked: Int  # operators checked

    def __init__(out self, ok: Bool, reason: String, checked: Int):
        self.ok = ok
        self.reason = reason
        self.checked = checked

    def to_json(self) -> String:
        var w = JsonWriter()
        w.begin_object()
        w.kv_bool("valid", self.ok)
        w.kv_string("reason", self.reason)
        w.kv_int("operators_checked", self.checked)
        w.end_object()
        return w.finish()

    def write_to(self, mut w: Some[Writer]):
        if self.ok:
            w.write("VALID (", self.checked, " operators checked)")
        else:
            w.write("INVALID: ", self.reason)


# ---------------------------------------------------------------------------
# Tableau helpers
# ---------------------------------------------------------------------------
def tableau_row(sim: TableauSimulator, row: Int) -> SymPauli:
    """Row `row` of the inverse tableau as a SymPauli (rows 0..n-1: C^dag X_q C,
    rows n..2n-1: C^dag Z_q C)."""
    var p = SymPauli(sim.n)
    for q in range(sim.n):
        var w = q >> 6
        var b = UInt64(q & 63)
        p.xs[q] = UInt8((sim.xs[row * sim.nw + w] >> b) & 1)
        p.zs[q] = UInt8((sim.zs[row * sim.nw + w] >> b) & 1)
    p.phase = 2 if sim.signs[row] != 0 else 0
    return p^


def heisenberg_image(sim: TableauSimulator, p: SymPauli) raises -> SymPauli:
    """C^dag P C for the Clifford C the simulator applied (from |0..0>)."""
    var out = SymPauli(sim.n)
    out.phase = p.phase
    for q in range(p.n):
        var x = p.x(q)
        var z = p.z(q)
        if x == 1 and z == 0:
            out = out.mul(tableau_row(sim, q))
        elif x == 0 and z == 1:
            out = out.mul(tableau_row(sim, sim.n + q))
        elif x == 1 and z == 1:
            # Y = i X Z
            out = out.mul(tableau_row(sim, q)).mul(tableau_row(sim, sim.n + q)).times_phase(1)
    return out^


def _check_unitary_clifford(c: Circuit, what: String) raises:
    for i in range(len(c.ops)):
        var g = c.ops[i]
        if op_is_annotation(g.kind) or g.kind == Op.BARRIER or g.kind == Op.TICK:
            continue
        if g.cond >= 0:
            raise Error("QM-E723: " + what + " instruction #" + String(i) + " is classically conditioned. Hint: Clifford verification needs a unitary circuit; verify each branch separately.")
        if not op_is_unitary(g.kind):
            raise Error("QM-E723: " + what + " instruction #" + String(i) + " '" + op_name(g.kind) + "' is not unitary. Hint: use valid_stabilizer_state_preparation for gadgets with measurements.")
        if not g.is_clifford():
            raise Error("QM-E723: " + what + " instruction #" + String(i) + " '" + op_name(g.kind) + "' is not Clifford. Hint: the tableau verifier checks Clifford gadgets; verify magic states with valid_magic_state_preparation.")


def run_tableau(c: Circuit, num_qubits: Int) raises -> TableauSimulator:
    var sim = TableauSimulator(num_qubits, 0)
    sim.run(c, noisy=False, forced_zero=True)
    return sim^


def expand_logical(p: SymPauli, code: StabilizerCode, num_blocks: Int) raises -> SymPauli:
    """Logical Pauli on num_blocks*k qubits -> physical Pauli on num_blocks*n
    (logical qubit j lives in block j // k, slot j % k)."""
    var n = code.n
    var k = code.k
    var out = SymPauli(num_blocks * n)
    for j in range(p.n):
        var c = p.get(j)
        if c == "I":
            continue
        var blk = j // k
        var phys = code.logical(j % k, c).embed(blk * n, num_blocks * n)
        out = out.mul(phys)
    out.phase = (out.phase + p.phase) & 3
    return out^


def code_group(code: StabilizerCode, num_blocks: Int, num_ancillas: Int) raises -> StabilizerGroup:
    """Generators of every block, plus Z on each clean ancilla."""
    var total = num_blocks * code.n + num_ancillas
    var g = StabilizerGroup(total)
    for b in range(num_blocks):
        for s in code.generators:
            _ = g.add(s.embed(b * code.n, total))
    for a in range(num_ancillas):
        _ = g.add(SymPauli.single(total, num_blocks * code.n + a, "Z"))
    return g^


def _logical_name(j: Int, p: String) -> String:
    return p + "_" + String(j)


def valid_clifford_implementation(semantic: Circuit, impl: Circuit, code: StabilizerCode,
                                  num_blocks: Int = 1, num_ancillas: Int = 0) raises -> VerifyReport:
    """See the module docstring. Returns a report; `.ok` is the verdict."""
    var nk = num_blocks * code.k
    var nn = num_blocks * code.n + num_ancillas
    if semantic.num_qubits != nk:
        raise Error("QM-E723: semantic circuit has " + String(semantic.num_qubits) + " qubits, expected num_blocks*k = " + String(nk) + ". Hint: the semantic function acts on the logical qubits only.")
    if impl.num_qubits != nn:
        raise Error("QM-E723: implementation has " + String(impl.num_qubits) + " qubits, expected num_blocks*n + ancillas = " + String(nn) + ". Hint: pass num_ancillas for clean ancillas.")
    _check_unitary_clifford(semantic, "semantic")
    _check_unitary_clifford(impl, "implementation")
    var sem = run_tableau(semantic, nk)
    var imp = run_tableau(impl, nn)
    var grp = code_group(code, num_blocks, num_ancillas)
    var checked = 0
    # 1) the stabilizer group is preserved
    for b in range(num_blocks):
        for s in code.generators:
            var g = s.embed(b * code.n, nn)
            var img = heisenberg_image(imp, g)
            checked += 1
            if not grp.contains(img):
                var why = String("leaves the code space")
                if grp.contains_up_to_sign(img):
                    why = "flips its sign"
                return VerifyReport(False, "stabilizer " + g.label() + " (block " + String(b) + ") maps to " + img.label() + ": the implementation " + why + ".", checked)
    for a in range(num_ancillas):
        var g = SymPauli.single(nn, num_blocks * code.n + a, "Z")
        var img = heisenberg_image(imp, g)
        checked += 1
        if not grp.contains(img):
            return VerifyReport(False, "ancilla " + String(a) + " does not return to |0> (Z_anc maps to " + img.label() + ").", checked)
    # 2) logical action
    var ps: List[String] = ["X", "Z"]
    for j in range(nk):
        for p in ps:
            var lp = SymPauli.single(nk, j, p)
            var sem_img = heisenberg_image(sem, lp)
            var expected = expand_logical(sem_img, code, num_blocks).embed(0, nn)
            var phys = expand_logical(lp, code, num_blocks).embed(0, nn)
            var actual = heisenberg_image(imp, phys)
            var prod = actual.mul(expected)
            checked += 1
            if not grp.contains(prod):
                var detail = String("a different logical operator")
                if grp.contains_up_to_sign(prod):
                    detail = "the right operator with the wrong sign"
                return VerifyReport(
                    False,
                    "logical " + _logical_name(j, p) + ": the semantic circuit gives U^dag " + _logical_name(j, p) + " U = " + sem_img.label()
                    + " but the implementation gives " + detail + " (C^dag P C = " + actual.label() + ", expected " + expected.label() + " up to stabilizers).",
                    checked,
                )
    return VerifyReport(True, "", checked)


def is_valid_clifford_implementation(semantic: Circuit, impl: Circuit, code: StabilizerCode,
                                     num_blocks: Int = 1, num_ancillas: Int = 0) raises -> Bool:
    return valid_clifford_implementation(semantic, impl, code, num_blocks, num_ancillas).ok


def semantic_state_stabilizers(semantic: Circuit) raises -> List[SymPauli]:
    """Stabilizer generators of semantic|0..0> (unitary Clifford semantic)."""
    _check_unitary_clifford(semantic, "semantic")
    var inv = semantic.inverse()
    var sim = run_tableau(inv, semantic.num_qubits)
    var out = List[SymPauli]()
    for q in range(semantic.num_qubits):
        out.append(tableau_row(sim, semantic.num_qubits + q))
    return out^


def _sparse(p: SymPauli, mut qs: List[Int], mut ps: List[Int]):
    qs.clear()
    ps.clear()
    for q in range(p.n):
        var x = p.x(q)
        var z = p.z(q)
        if x == 0 and z == 0:
            continue
        qs.append(q)
        ps.append(2 if (x == 1 and z == 1) else (1 if x == 1 else 3))


def valid_stabilizer_state_preparation(semantic: Circuit, impl: Circuit, code: StabilizerCode,
                                       num_blocks: Int = 1, num_ancillas: Int = 0, trials: Int = 8) raises -> VerifyReport:
    """See the module docstring. Data qubits are impl qubits 0..num_blocks*n-1."""
    var nk = num_blocks * code.k
    var nd = num_blocks * code.n
    var nn = nd + num_ancillas
    if semantic.num_qubits != nk:
        raise Error("QM-E723: semantic state preparation has " + String(semantic.num_qubits) + " qubits, expected num_blocks*k = " + String(nk) + ".")
    if impl.num_qubits != nn:
        raise Error("QM-E723: implementation has " + String(impl.num_qubits) + " qubits, expected num_blocks*n + ancillas = " + String(nn) + ". Hint: pass num_ancillas for flag/syndrome qubits.")
    for i in range(len(impl.ops)):
        var g = impl.ops[i]
        if op_is_unitary(g.kind) and not g.is_clifford():
            raise Error("QM-E723: implementation instruction #" + String(i) + " '" + op_name(g.kind) + "' is not Clifford. Hint: verify magic states with valid_magic_state_preparation.")
    var sem = semantic_state_stabilizers(semantic)
    var expected = List[SymPauli]()
    var grp = StabilizerGroup(nd)
    for s in sem:
        var e = expand_logical(s, code, num_blocks)
        _ = grp.add(e)
        expected.append(e^)
    for b in range(num_blocks):
        for s in code.generators:
            var e = s.embed(b * code.n, nd)
            _ = grp.add(e)
            expected.append(e^)
    if grp.rank() != nd:
        return VerifyReport(False, "expected generators are not independent (rank " + String(grp.rank()) + " of " + String(nd) + ").", 0)
    var checked = 0
    var accepted = 0
    var qs = List[Int]()
    var ps = List[Int]()
    var root = Rng(20260514)
    for t in range(trials + 1):
        var sim = TableauSimulator(nn, 0)
        if t == 0:
            sim.run(impl, noisy=False, forced_zero=True)
        else:
            sim.rng = root.split(t)
            sim.run(impl, noisy=False, forced_zero=False)
        var flagged = False
        for r in sim.records:
            if r != 0:
                flagged = True
        if flagged:
            if t == 0:
                return VerifyReport(False, "a flag/verification measurement fires deterministically (the accept branch is never taken).", 0)
            continue
        accepted += 1
        for e in expected:
            _sparse(e, qs, ps)
            var v = sim.expectation_sparse(qs, ps) * e.sign()
            checked += 1
            if v != 1:
                var got = String("random (0)") if v == 0 else String("-1")
                return VerifyReport(False, "expected stabilizer " + e.label() + " has expectation " + got + " on an accepted run.", checked)
    return VerifyReport(True, "", checked)


def is_valid_stabilizer_state_preparation(semantic: Circuit, impl: Circuit, code: StabilizerCode,
                                          num_blocks: Int = 1, num_ancillas: Int = 0) raises -> Bool:
    return valid_stabilizer_state_preparation(semantic, impl, code, num_blocks, num_ancillas).ok
