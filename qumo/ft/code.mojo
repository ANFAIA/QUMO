"""Generic stabilizer-code definitions for the fault-tolerant layer.

A `StabilizerCode` stores [[n, k, d]], n-k commuting stabilizer generators and
the k logical X / Z operators (Y logicals are derived as i X Z). `validate()`
checks everything that makes the definition usable by the verifier and the
encoder: generator count, commutation, independence, logicals commuting with
the stabilizers, the canonical X_i Z_j anticommutation relations and logical
independence from the stabilizer group. `css` is derived (every generator is
X-type or Z-type). `min_distance()` brute-forces the distance of small codes.

Codes shipped here:

| name | [[n,k,d]] | notes |
|---|---|---|
| steane | [[7,1,3]] | Guppy FT's STEANE_DEF (same generators and logicals) |
| iceberg_k | [[k+2,k,2]] | k even; layout [t, b, d_0..d_{k-1}], X_i = X_t X_i, Z_i = Z_b Z_i |
| css_422 | [[4,2,2]] | Guppy FT test code CSS_4Q_DEF |
| noncss_422 | [[4,2,2]] | non-CSS code of arXiv:2510.18753 (Guppy FT verifier test) |
| five_qubit | [[5,1,3]] | perfect code (non-CSS) |
| surface_d | [[d^2,1,d]] | adapter over qumo.qec.codes.surface_code_memory |
| repetition_d | [[d,1,1]] | adapter over qumo.qec.codes.repetition_code_memory (bit-flip distance d) |

Label convention: character i is qubit i (see qumo.ft.pauli).

Idea from Guppy FT (Apache-2.0), file src/guppyft/code_def.py (`StabilizerCode`,
`from_python_strings`, `_identity_code`) and tests/verifier/ops/*.py (code
definitions); re-derived in Mojo, with extra validation.

Error codes: QM-E720 (bad label), QM-E721 (invalid code definition),
QM-E722 (unknown code name or bad code parameters).
"""

from qumo.ft.pauli import SymPauli, StabilizerGroup, group_from
from qumo.ai.json import JsonWriter
from qumo.qec.codes import surface_code_memory, repetition_code_memory, NoiseParams, CodeCircuit
from qumo.ir.gates import Op


struct StabilizerCode(Copyable, Writable):
    var name: String
    var n: Int
    var k: Int
    var d: Int
    var generators: List[SymPauli]
    var x_logicals: List[SymPauli]
    var z_logicals: List[SymPauli]
    var notes: String

    def __init__(out self, name: String, n: Int, k: Int, d: Int,
                 var generators: List[SymPauli], var x_logicals: List[SymPauli],
                 var z_logicals: List[SymPauli], notes: String = ""):
        self.name = name
        self.n = n
        self.k = k
        self.d = d
        self.generators = generators^
        self.x_logicals = x_logicals^
        self.z_logicals = z_logicals^
        self.notes = notes

    @staticmethod
    def from_strings(name: String, n: Int, k: Int, d: Int, generators: List[String],
                     x_logicals: List[String], z_logicals: List[String], notes: String = "") raises -> StabilizerCode:
        """Build and validate a code from label lists (char i = qubit i),
        the analogue of Guppy FT's `StabilizerCode.from_python_strings`."""
        var g = List[SymPauli]()
        var xl = List[SymPauli]()
        var zl = List[SymPauli]()
        for s in generators:
            g.append(_label(s, n))
        for s in x_logicals:
            xl.append(_label(s, n))
        for s in z_logicals:
            zl.append(_label(s, n))
        var code = StabilizerCode(name, n, k, d, g^, xl^, zl^, notes)
        code.validate()
        return code^

    # ---------------------------------------------------------------- queries
    def is_css(self) -> Bool:
        for g in self.generators:
            if not (g.is_x_type() or g.is_z_type()):
                return False
        return True

    def y_logical(self, j: Int) raises -> SymPauli:
        """Y_j = i X_j Z_j (Hermitian since X_j and Z_j anticommute)."""
        return self.x_logicals[j].mul(self.z_logicals[j]).times_phase(1)

    def logical(self, j: Int, p: String) raises -> SymPauli:
        if p == "X":
            return self.x_logicals[j].copy()
        if p == "Z":
            return self.z_logicals[j].copy()
        if p == "Y":
            return self.y_logical(j)
        return SymPauli(self.n)

    def stabilizer_group(self) raises -> StabilizerGroup:
        return group_from(self.generators, self.n)

    def z_type_generators(self) -> List[Int]:
        var out = List[Int]()
        for i in range(len(self.generators)):
            if self.generators[i].is_z_type():
                out.append(i)
        return out^

    def x_type_generators(self) -> List[Int]:
        var out = List[Int]()
        for i in range(len(self.generators)):
            if self.generators[i].is_x_type():
                out.append(i)
        return out^

    # ------------------------------------------------------------- validation
    def validate(self) raises:
        """QM-E721 with a precise reason when the definition is inconsistent."""
        if self.n < 1 or self.k < 1 or self.k > self.n:
            raise Error("QM-E721: code '" + self.name + "' needs 1 <= k <= n (got n=" + String(self.n) + ", k=" + String(self.k) + "). Hint: n physical qubits encode k logical qubits.")
        if len(self.generators) != self.n - self.k:
            raise Error("QM-E721: code '" + self.name + "': the number of stabilizer generators must equal n-k = " + String(self.n - self.k) + " (got " + String(len(self.generators)) + "). Hint: list n-k independent commuting generators.")
        if len(self.x_logicals) != self.k or len(self.z_logicals) != self.k:
            raise Error("QM-E721: code '" + self.name + "': expected " + String(self.k) + " X and Z logical operators (got " + String(len(self.x_logicals)) + " and " + String(len(self.z_logicals)) + "). Hint: one X and one Z logical per logical qubit.")
        for p in self.generators:
            _check_size(self, p)
            if not p.is_hermitian():
                raise Error("QM-E721: code '" + self.name + "': generator " + p.label() + " is not Hermitian. Hint: signs must be + or -.")
        for p in self.x_logicals:
            _check_size(self, p)
        for p in self.z_logicals:
            _check_size(self, p)
        for i in range(len(self.generators)):
            for j in range(i + 1, len(self.generators)):
                if not self.generators[i].commutes(self.generators[j]):
                    raise Error("QM-E721: code '" + self.name + "': stabilizer generators " + self.generators[i].label() + " and " + self.generators[j].label() + " anticommute. Hint: all generators must commute.")
        var grp = StabilizerGroup(self.n)
        for g in self.generators:
            if not grp.add(g):
                raise Error("QM-E721: code '" + self.name + "': generator " + g.label() + " is dependent on the previous ones (or -I is in the group). Hint: give n-k independent generators.")
        # -I must not be in the group: a dependent generator would be caught
        # above, and rows are exact products, so contains(-I) checks it.
        var minus_i = SymPauli(self.n).times_phase(2)
        if grp.contains(minus_i):
            raise Error("QM-E721: code '" + self.name + "': -I is in the stabilizer group (empty code space). Hint: check generator signs.")
        for j in range(self.k):
            for g in self.generators:
                if not self.x_logicals[j].commutes(g) or not self.z_logicals[j].commutes(g):
                    raise Error("QM-E721: code '" + self.name + "': logical operator of qubit " + String(j) + " anticommutes with stabilizer " + g.label() + ". Hint: logical operators must commute with every stabilizer.")
            for i in range(self.k):
                var xz_comm = self.x_logicals[j].commutes(self.z_logicals[i])
                if (i == j) == xz_comm:
                    raise Error("QM-E721: code '" + self.name + "': X_" + String(j) + " and Z_" + String(i) + " must " + (String("anticommute") if i == j else String("commute")) + ". Hint: logical operators must satisfy the canonical Pauli relations.")
                if i != j:
                    if not self.x_logicals[j].commutes(self.x_logicals[i]) or not self.z_logicals[j].commutes(self.z_logicals[i]):
                        raise Error("QM-E721: code '" + self.name + "': logical operators of different qubits must commute. Hint: check x_logicals / z_logicals.")
        # logical operators independent of the stabilizer group
        var full = grp.copy()
        for j in range(self.k):
            if not full.add(self.x_logicals[j]) or not full.add(self.z_logicals[j]):
                raise Error("QM-E721: code '" + self.name + "': logical operators of qubit " + String(j) + " are in the span of the stabilizers/other logicals. Hint: logicals must be independent.")

    def min_distance(self, max_weight: Int = 4) raises -> Int:
        """Brute-force distance: the smallest weight of a Pauli that commutes
        with every generator but is not in the stabilizer group (up to phase).
        Returns -1 when no logical of weight <= max_weight exists."""
        var grp = self.stabilizer_group()
        for w in range(1, max_weight + 1):
            var idx = List[Int]()
            for i in range(w):
                idx.append(i)
            while True:
                # 3^w Pauli assignments on the support idx
                var total = 1
                for _ in range(w):
                    total *= 3
                for a in range(total):
                    var p = SymPauli(self.n)
                    var t = a
                    for i in range(w):
                        var c = t % 3
                        t //= 3
                        if c == 0:
                            p.xs[idx[i]] = 1
                        elif c == 1:
                            p.zs[idx[i]] = 1
                        else:
                            p.xs[idx[i]] = 1
                            p.zs[idx[i]] = 1
                    var ok = True
                    for g in self.generators:
                        if not p.commutes(g):
                            ok = False
                            break
                    if ok and not grp.contains_up_to_sign(p):
                        return w
                # next combination
                var i = w - 1
                while i >= 0 and idx[i] == self.n - w + i:
                    i -= 1
                if i < 0:
                    break
                idx[i] += 1
                for j in range(i + 1, w):
                    idx[j] = idx[j - 1] + 1
        return -1

    def write_json(self, mut w: JsonWriter):
        w.begin_object()
        w.kv_string("name", self.name)
        w.kv_int("n", self.n)
        w.kv_int("k", self.k)
        w.kv_int("d", self.d)
        w.kv_bool("css", self.is_css())
        var gl = List[String]()
        for g in self.generators:
            gl.append(g.label())
        w.kv_string_list("generators", gl)
        var xl = List[String]()
        for g in self.x_logicals:
            xl.append(g.label())
        w.kv_string_list("x_logicals", xl)
        var zl = List[String]()
        for g in self.z_logicals:
            zl.append(g.label())
        w.kv_string_list("z_logicals", zl)
        w.kv_string("label_convention", "character i = physical qubit i")
        if self.notes.byte_length() > 0:
            w.kv_string("notes", self.notes)
        w.end_object()

    def to_json(self) -> String:
        var w = JsonWriter()
        self.write_json(w)
        return w.finish()

    def write_to(self, mut w: Some[Writer]):
        w.write("StabilizerCode(", self.name, " [[", self.n, ",", self.k, ",", self.d, "]]",
                ", css=", self.is_css(), ")")


def _label(s: String, n: Int) raises -> SymPauli:
    var p = SymPauli.from_label(s)
    if p.n != n:
        raise Error("QM-E720: Pauli label '" + s + "' has " + String(p.n) + " qubits, the code has n=" + String(n) + ". Hint: one character per physical qubit.")
    return p^


def _check_size(code: StabilizerCode, p: SymPauli) raises:
    if p.n != code.n:
        raise Error("QM-E721: code '" + code.name + "': operator " + p.label() + " acts on " + String(p.n) + " qubits, expected " + String(code.n) + ". Hint: one character per physical qubit.")


# ---------------------------------------------------------------------------
# Code library
# ---------------------------------------------------------------------------
def steane_code() raises -> StabilizerCode:
    """[[7,1,3]] Steane code, exactly Guppy FT's STEANE_DEF."""
    var g: List[String] = ["XXXXIII", "IXXIXXI", "IIXXIXX", "ZZZZIII", "IZZIZZI", "IIZZIZZ"]
    var xl: List[String] = ["XXXXXXX"]
    var zl: List[String] = ["ZZZZZZZ"]
    return StabilizerCode.from_strings("steane", 7, 1, 3, g, xl, zl, "Steane [[7,1,3]] CSS code (Hamming [7,4] checks); transversal Clifford group.")


def iceberg_code(k: Int) raises -> StabilizerCode:
    """[[k+2,k,2]] Iceberg code (k even). Qubit 0 = t, 1 = b, 2+i = d_i;
    S_X = X^n, S_Z = Z^n, X_i = X_t X_{d_i}, Z_i = Z_b Z_{d_i}
    (Self, Benedetti, Amaro, arXiv:2211.06703)."""
    if k < 2 or (k % 2) != 0:
        raise Error("QM-E722: Iceberg codes need an even k >= 2 (got " + String(k) + "). Hint: n = k + 2 must be even so that X^n and Z^n commute.")
    var n = k + 2
    var sx = String()
    var sz = String()
    for _ in range(n):
        sx += "X"
        sz += "Z"
    var g: List[String] = [sx, sz]
    var xl = List[String]()
    var zl = List[String]()
    for i in range(k):
        var a = String()
        var b = String()
        for q in range(n):
            a += String("X") if (q == 0 or q == 2 + i) else String("I")
            b += String("Z") if (q == 1 or q == 2 + i) else String("I")
        xl.append(a)
        zl.append(b)
    return StabilizerCode.from_strings("iceberg_" + String(k), n, k, 2, g, xl, zl,
        "Iceberg error-detecting code; layout [t, b, d_0..d_{k-1}]; logical rotations are physical RZZ/RXX pairs.")


def css_422_code() raises -> StabilizerCode:
    var g: List[String] = ["XXXX", "ZZZZ"]
    var xl: List[String] = ["XXII", "XIXI"]
    var zl: List[String] = ["IZIZ", "IIZZ"]
    return StabilizerCode.from_strings("css_422", 4, 2, 2, g, xl, zl, "[[4,2,2]] CSS code (Guppy FT verifier test code CSS_4Q_DEF).")


def noncss_422_code() raises -> StabilizerCode:
    var g: List[String] = ["XZZX", "ZXXZ"]
    var xl: List[String] = ["ZIXI", "IZIX"]
    var zl: List[String] = ["IZZI", "ZIIZ"]
    return StabilizerCode.from_strings("noncss_422", 4, 2, 2, g, xl, zl, "Non-CSS [[4,2,2]] code from arXiv:2510.18753 (Guppy FT verifier test).")


def five_qubit_code() raises -> StabilizerCode:
    var g: List[String] = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    var xl: List[String] = ["XXXXX"]
    var zl: List[String] = ["ZZZZZ"]
    return StabilizerCode.from_strings("five_qubit", 5, 1, 3, g, xl, zl, "Perfect [[5,1,3]] code (non-CSS).")


def identity_code(k: Int) raises -> StabilizerCode:
    """Trivial [[k,k,1]] code (Guppy FT `_identity_code`)."""
    var g = List[String]()
    var xl = List[String]()
    var zl = List[String]()
    for i in range(k):
        var a = String()
        var b = String()
        for q in range(k):
            a += String("X") if q == i else String("I")
            b += String("Z") if q == i else String("I")
        xl.append(a)
        zl.append(b)
    return StabilizerCode.from_strings("identity_" + String(k), k, k, 1, g, xl, zl, "Trivial code: no redundancy.")


def _code_from_memory_circuit(cc: CodeCircuit, name: String, d: Int, x_log: List[Int], z_log: List[Int], notes: String) raises -> StabilizerCode:
    """Read stabilizer supports from the first syndrome round of a qumo.qec
    memory circuit: an X check is CX(check -> data), a Z check CX(data -> check)."""
    var nd = len(cc.data_qubits)
    var pos = List[Int](length=cc.circuit.num_qubits, fill=-1)
    for i in range(nd):
        pos[cc.data_qubits[i]] = i
    var gens = List[SymPauli]()
    var checks = cc.x_checks.copy()
    for q in cc.z_checks:
        checks.append(q)
    for ch in checks:
        var p = SymPauli(nd)
        var is_x = False
        for xq in cc.x_checks:
            if xq == ch:
                is_x = True
        for g in cc.circuit.ops:
            if g.kind == Op.MEASURE:
                break
            if g.kind != Op.CX:
                continue
            if is_x and g.q0 == ch and pos[g.q1] >= 0:
                p.xs[pos[g.q1]] = 1
            elif (not is_x) and g.q1 == ch and pos[g.q0] >= 0:
                p.zs[pos[g.q0]] = 1
        gens.append(p^)
    var xl = SymPauli(nd)
    for q in x_log:
        xl.xs[q] = 1
    var zl = SymPauli(nd)
    for q in z_log:
        zl.zs[q] = 1
    var xls = List[SymPauli]()
    xls.append(xl^)
    var zls = List[SymPauli]()
    zls.append(zl^)
    var code = StabilizerCode(name, nd, 1, d, gens^, xls^, zls^, notes)
    code.validate()
    return code^


def surface_code(d: Int) raises -> StabilizerCode:
    """Rotated surface code [[d^2,1,d]] read from qumo.qec.codes.surface_code_memory
    (data qubit (i, j) is code qubit j*d + i; Z_L on column i = 0, X_L on row j = 0)."""
    var cc = surface_code_memory(d, 1, NoiseParams.noiseless(), "Z")
    var xl = List[Int]()
    var zl = List[Int]()
    for t in range(d):
        xl.append(t)
        zl.append(t * d)
    return _code_from_memory_circuit(cc, "surface_" + String(d), d, xl, zl,
        "Adapter over qumo.qec.codes.surface_code_memory (rotated layout, same checks as the memory experiments).")


def repetition_code(d: Int) raises -> StabilizerCode:
    """Bit-flip repetition code [[d,1,1]] read from qumo.qec.codes.repetition_code_memory
    (X distance d, Z distance 1; Z_L = Z on the last data qubit, as the memory observable)."""
    var cc = repetition_code_memory(d, 1, NoiseParams.noiseless())
    var xl = List[Int]()
    for t in range(d):
        xl.append(t)
    var zl: List[Int] = [d - 1]
    return _code_from_memory_circuit(cc, "repetition_" + String(d), 1, xl, zl,
        "Adapter over qumo.qec.codes.repetition_code_memory: protects against bit flips only (bit-flip distance " + String(d) + ").")


def _suffix_int(name: String, prefix: String) raises -> Int:
    var s = String(name[byte=prefix.byte_length():])
    return Int(s)


def code_by_name(name: String) raises -> StabilizerCode:
    """steane, five_qubit, css_422, noncss_422, iceberg_<k>, surface_<d>, repetition_<d>, identity_<k>."""
    var k = name.lower()
    if k == "steane" or k == "steane_713":
        return steane_code()
    if k == "five_qubit" or k == "513" or k == "perfect":
        return five_qubit_code()
    if k == "css_422" or k == "422":
        return css_422_code()
    if k == "noncss_422":
        return noncss_422_code()
    try:
        if k.startswith("iceberg_"):
            return iceberg_code(_suffix_int(k, "iceberg_"))
        if k.startswith("surface_"):
            return surface_code(_suffix_int(k, "surface_"))
        if k.startswith("repetition_"):
            return repetition_code(_suffix_int(k, "repetition_"))
        if k.startswith("identity_"):
            return identity_code(_suffix_int(k, "identity_"))
    except e:
        var msg = String(e)
        if msg.startswith("QM-E"):
            raise Error(msg)
        raise Error("QM-E722: bad code parameter in '" + name + "'. Hint: e.g. iceberg_4, surface_3, repetition_5.")
    raise Error("QM-E722: unknown code '" + name + "'. Hint: use one of " + code_names() + ".")


def code_names() -> String:
    return "steane, five_qubit, css_422, noncss_422, iceberg_<even k>, surface_<d>, repetition_<d>, identity_<k>"
