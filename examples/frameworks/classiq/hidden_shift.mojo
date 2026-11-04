"""Hidden shift for Boolean bent functions: the notebook's three algorithms on QUMO.

Reproduces: Classiq library - Hidden-Shift Problem for Bent Functions
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/number_theory_and_cryptography/hidden_shift/hidden_shift.ipynb (Classiq Library, MIT)

A Boolean function f is bent when every Walsh coefficient W(a) = sum_x (-1)^(f(x) + a.x) has magnitude 2^(n/2); its dual f~ is the function with W(a) = 2^(n/2) (-1)^f~(a). The notebook hides a shift s in g(x) = f(x XOR s) and recovers s with three algorithms, all reproduced here. (1) On n = 4 qubits with f = x0 x1 XOR x2 x3 and s = e0, H, O_g, H, O_f, H leaves |s> with certainty. (2) On n = 16 with the Maiorana-McFarland function f(x, y) = x . p(y) XOR AND(y), the shift s = bits {1, 3, 9} and the dual f~(x, y) = x . p(y) XOR AND(x), the same circuit leaves |s> with certainty. (3) Without the dual, an ancilla takes f(x) and then f(x) XOR g(x) around a Z gate, H is applied to the register, and each measured pair (y, c) satisfies c = s . y (mod 2); 1000 shots solved over GF(2) give s.

The checks compare each oracle's signs with the notebook's classical truth table, the dual with the Walsh spectrum (fast Walsh-Hadamard transform, brute force), the output probabilities with the 1.0 and the bitstrings the notebook prints, the exact law of (y, c) with the analytic uniform law on c = s . y, and the GF(2) solution of the samples with the notebook's s.

Deviations from the notebook:
- Classiq's phase_oracle compiles each Boolean function through a reversible circuit. Here the same diagonal unitary is written directly: the bilinear part as one CZ between x_i and y_p(i) per term, and the AND part as one multi-controlled Z (append_mcp, no ancilla). The oracle signs match the classical truth tables exactly.
- The no-dual oracle XORs the bilinear terms into the ancilla with CCX gates and the AND term with one multi-controlled X (append_mcx, no ancilla).
- The permutation p is the notebook's random.seed(1) shuffle of range(8), computed by running the notebook's own Python code: p = [3, 6, 1, 5, 7, 0, 4, 2].
- Cases 1 and 2 read exact probabilities where the notebook samples 2048 shots. Case 3 samples 1000 shots, as the notebook suggests.
- The GF(2) solver is written here in place of the galois package.
- The notebook's shifted() picks x or y with bit < NUM_VARIABLES >> 2 (= 4). Both branches map shift bit b to register qubit b, so the mask for {1, 3, 9} is 2 + 8 + 512 = 522. For shift bits 4 to 7 the notebook's y branch would wrap to another qubit under Python indexing, but the notebook uses no such bit.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/hidden_shift.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.bit import pop_count
from std.math import pi, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.mc import append_mcp, append_mcx
from qumo.sim.runner import bitstring, evolve, sample
from qumo.testing.checks import CheckLog


# ---------------------------------------------------------------------------
# The notebook's functions as truth tables. A basis index xy holds x in bits
# 0..7 and y in bits 8..15 (little-endian, as on the QUMO register).
# ---------------------------------------------------------------------------
def bit(v: Int, i: Int) -> Int:
    return (v >> i) & 1


def parity(v: Int) -> Int:
    return Int(pop_count(v)) & 1


def perm() -> List[Int]:
    """The notebook's p: random.seed(1); random.shuffle(list(range(8)))."""
    var p: List[Int] = [3, 6, 1, 5, 7, 0, 4, 2]
    return p^


def inverse(p: List[Int]) -> List[Int]:
    var inv = List[Int](length=len(p), fill=0)
    for i in range(len(p)):
        inv[p[i]] = i
    return inv^


def shift_bits() -> List[Int]:
    """The notebook's shifted_bits."""
    var b: List[Int] = [1, 3, 9]
    return b^


def shift_mask(bits: List[Int]) -> Int:
    """Flip mask of the notebook's shifted(): a shift bit b flips x_b (b < 4) or y_(b - 8); both are register qubit b."""
    var mask = 0
    for b in bits:
        mask |= 1 << b
    return mask


def bilinear(xy: Int, p: List[Int]) -> Int:
    """The bilinear form x . p(y) = sum_i x_i y_{p(i)} mod 2."""
    var s = 0
    for i in range(8):
        s ^= bit(xy, i) & bit(xy, 8 + p[i])
    return s


def f_truth(xy: Int, p: List[Int]) -> Int:
    """The notebook's f(x, y) = x . p(y) XOR AND(y)."""
    var all_y = 1
    for i in range(8):
        all_y &= bit(xy, 8 + i)
    return bilinear(xy, p) ^ all_y


def dual_truth(xy: Int, p: List[Int], inv: List[Int]) -> Int:
    """The notebook's f_dual_func: sum_i x[p^-1(i)] y_i XOR AND over the x[p^-1(i)]."""
    var s = 0
    var all_x = 1
    for i in range(8):
        s ^= bit(xy, inv[i]) & bit(xy, 8 + i)
        all_x &= bit(xy, inv[i])
    return s ^ all_x


def f4(x: Int) -> Int:
    """Case 1: f = x0 x1 XOR x2 x3."""
    return (bit(x, 0) & bit(x, 1)) ^ (bit(x, 2) & bit(x, 3))


# ---------------------------------------------------------------------------
# Classical brute force
# ---------------------------------------------------------------------------
def walsh(truth: List[Int], n: Int) -> List[Float64]:
    """W(a) = sum_x (-1)^(f(x) + a.x) for all a, by the fast Walsh-Hadamard transform."""
    var size = 1 << n
    var w = List[Float64](length=size, fill=0.0)
    for x in range(size):
        w[x] = 1.0 - 2.0 * Float64(truth[x])
    var h = 1
    while h < size:
        var i = 0
        while i < size:
            for j in range(i, i + h):
                var a = w[j]
                var b = w[j + h]
                w[j] = a + b
                w[j + h] = a - b
            i += 2 * h
        h *= 2
    return w^


def dual_dev(truth: List[Int], dual: List[Int], n: Int) -> Float64:
    """Largest |W(a) - 2^(n/2) (-1)^dual(a)| over all a: zero exactly when dual is the Walsh dual of truth."""
    var w = walsh(truth, n)
    var root = sqrt(Float64(1 << n))
    var dev = 0.0
    for a in range(1 << n):
        var want = root if dual[a] == 0 else -root
        dev = max(dev, abs(w[a] - want))
    return dev


def argmax(v: List[Float64]) -> Int:
    var best = 0
    for i in range(len(v)):
        if v[i] > v[best]:
            best = i
    return best


# ---------------------------------------------------------------------------
# Circuits
# ---------------------------------------------------------------------------
def hadamard_layer(mut c: Circuit, n: Int):
    for q in range(n):
        c.h(q)


def qubit_range(lo: Int, hi: Int) -> List[Int]:
    var out = List[Int]()
    for q in range(lo, hi):
        out.append(q)
    return out^


def append_bilinear_phase(mut c: Circuit, p: List[Int]):
    """(-1)^(x . p(y)): one CZ per term x_i y_{p(i)}, with x_i on qubit i and y_j on qubit 8 + j."""
    for i in range(8):
        c.cz(i, 8 + p[i])


def append_and_phase(mut c: Circuit, qubits: List[Int]) raises:
    """(-1)^AND(qubits): one multi-controlled Z over the listed qubits, no ancilla."""
    var controls = List[Int]()
    for k in range(len(qubits) - 1):
        controls.append(qubits[k])
    append_mcp(c, pi, controls, qubits[len(qubits) - 1])


def phase_f(p: List[Int]) raises -> Circuit:
    """Diagonal (-1)^f on the 16-qubit register."""
    var c = Circuit(16, name="phase_f")
    append_bilinear_phase(c, p)
    append_and_phase(c, qubit_range(8, 16))
    return c^


def phase_dual(p: List[Int]) raises -> Circuit:
    """Diagonal (-1)^f~ on the 16-qubit register: the same bilinear part, AND over x."""
    var c = Circuit(16, name="phase_dual")
    append_bilinear_phase(c, p)
    append_and_phase(c, qubit_range(0, 8))
    return c^


def phase_shifted(p: List[Int]) raises -> Circuit:
    """Diagonal (-1)^g with g(xy) = f(xy XOR s): X on the shift qubits around the phase of f."""
    var c = Circuit(16, name="phase_g")
    var s = shift_bits()
    for q in s:
        c.x(q)
    c.compose(phase_f(p))
    for q in s:
        c.x(q)
    return c^


def phase_f4() -> Circuit:
    """Case 1: (-1)^(x0 x1 XOR x2 x3) as two CZ gates."""
    var c = Circuit(4, name="phase_f4")
    c.cz(0, 1)
    c.cz(2, 3)
    return c^


def phase_g4() -> Circuit:
    """Case 1: g(x) = f(x XOR e0), X on qubit 0 around the phase of f."""
    var c = Circuit(4, name="phase_g4")
    c.x(0)
    c.compose(phase_f4())
    c.x(0)
    return c^


def hidden_shift_circuit(n: Int, shifted: Circuit, dual: Circuit) -> Circuit:
    """The notebook's hidden_shift: H, O_g, H, O_f~, H on n qubits."""
    var c = Circuit(n, name="hidden_shift")
    hadamard_layer(c, n)
    c.compose(shifted)
    hadamard_layer(c, n)
    c.compose(dual)
    hadamard_layer(c, n)
    return c^


def xor_f_into(mut c: Circuit, p: List[Int], target: Int) raises:
    """XOR f(x, y) into target: one Toffoli per bilinear term x_i y_{p(i)}, then one MCX over y for AND(y)."""
    for i in range(8):
        c.ccx(i, 8 + p[i], target)
    append_mcx(c, qubit_range(8, 16), target)


def no_dual_circuit(p: List[Int]) raises -> Circuit:
    """The notebook's hidden_shift_no_dual on 17 qubits: register 0..15, ancilla 16."""
    var c = Circuit(17, name="hidden_shift_no_dual")
    hadamard_layer(c, 16)
    xor_f_into(c, p, 16)
    c.z(16)
    var s = shift_bits()
    for q in s:
        c.x(q)
    xor_f_into(c, p, 16)
    for q in s:
        c.x(q)
    hadamard_layer(c, 16)
    return c^


def phase_dev(oracle: Circuit, n: Int, truth: List[Int]) raises -> Float64:
    """Largest gap between the amplitudes of H^n O |0> and (-1)^truth(x) / sqrt(2^n)."""
    var c = Circuit(n, name="sign_probe")
    hadamard_layer(c, n)
    c.compose(oracle)
    var sv = evolve(c)
    var amp = 1.0 / sqrt(Float64(1 << n))
    var dev = 0.0
    for x in range(1 << n):
        var want = amp if truth[x] == 0 else -amp
        dev = max(dev, abs(sv.re[x] - want))
        dev = max(dev, abs(sv.im[x]))
    return dev


# ---------------------------------------------------------------------------
# GF(2) solve: Gauss-Jordan on the equations rows[k] . s = rhs[k]
# ---------------------------------------------------------------------------
def solve_gf2(rows: List[Int], rhs: List[Int], n: Int) -> Tuple[Int, Int]:
    """(s, rank): the solution when rank == n, else 0."""
    var a = rows.copy()
    var b = rhs.copy()
    var m = len(a)
    var rank = 0
    var pivot_col = List[Int]()
    for col in range(n):
        var sel = -1
        for r in range(rank, m):
            if ((a[r] >> col) & 1) == 1:
                sel = r
                break
        if sel < 0:
            continue
        var ta = a[sel]
        a[sel] = a[rank]
        a[rank] = ta
        var tb = b[sel]
        b[sel] = b[rank]
        b[rank] = tb
        for r in range(m):
            if r != rank and ((a[r] >> col) & 1) == 1:
                a[r] = a[r] ^ a[rank]
                b[r] = b[r] ^ b[rank]
        pivot_col.append(col)
        rank += 1
    if rank < n:
        return (0, rank)
    var s = 0
    for k in range(rank):
        s |= b[k] << pivot_col[k]
    return (s, rank)


def main() raises:
    var log = CheckLog("classiq/hidden_shift")
    var p = perm()
    var inv = inverse(p)
    var s_mask = shift_mask(shift_bits())
    var s_text = "0000001000001010"  # the notebook's printed bitstring for s = {1, 3, 9}

    # ---- 1. n = 4, f = x0 x1 XOR x2 x3, shift s = e0 ----
    print("1: n = 4, f = x0 x1 XOR x2 x3, shift e0")
    var t_f4 = List[Int](length=16, fill=0)
    var t_g4 = List[Int](length=16, fill=0)
    for x in range(16):
        t_f4[x] = f4(x)
        t_g4[x] = f4(x ^ 1)
    _ = log.close("n4 walsh dual of f is f", dual_dev(t_f4, t_f4, 4), 0.0, 1e-9)
    _ = log.close("n4 phase oracle f signs", phase_dev(phase_f4(), 4, t_f4), 0.0, 1e-9)
    _ = log.close("n4 phase oracle g signs", phase_dev(phase_g4(), 4, t_g4), 0.0, 1e-9)
    var probs4 = evolve(hidden_shift_circuit(4, phase_g4(), phase_f4())).probabilities()
    var best4 = argmax(probs4)
    _ = log.close("n4 P(s = 0001) after H O_g H O_f H", probs4[1], 1.0, 1e-9)
    _ = log.check("n4 most likely outcome is 0001", bitstring(best4, 4) == "0001", bitstring(best4, 4))

    # ---- 2. n = 16 Maiorana-McFarland pair, the dual oracle ----
    print("2: n = 16, f = x . p(y) XOR AND(y), shift bits {1, 3, 9}, dual oracle")
    var t_f = List[Int](length=65536, fill=0)
    var t_d = List[Int](length=65536, fill=0)
    var t_g = List[Int](length=65536, fill=0)
    for xy in range(65536):
        t_f[xy] = f_truth(xy, p)
        t_d[xy] = dual_truth(xy, p, inv)
        t_g[xy] = f_truth(xy ^ s_mask, p)
    _ = log.close("mm walsh dual of f matches the notebook dual", dual_dev(t_f, t_d, 16), 0.0, 1e-9)
    _ = log.close("mm phase oracle f signs", phase_dev(phase_f(p), 16, t_f), 0.0, 1e-9)
    _ = log.close("mm phase oracle dual signs", phase_dev(phase_dual(p), 16, t_d), 0.0, 1e-9)
    _ = log.close("mm phase oracle g signs", phase_dev(phase_shifted(p), 16, t_g), 0.0, 1e-9)
    var probs16 = evolve(hidden_shift_circuit(16, phase_shifted(p), phase_dual(p))).probabilities()
    var best16 = argmax(probs16)
    _ = log.close("mm P(s) after H O_g H O_dual H", probs16[s_mask], 1.0, 1e-9)
    _ = log.check("mm most likely outcome is the notebook s", bitstring(best16, 16) == s_text, bitstring(best16, 16))

    # ---- 3. the same pair without the dual: ancilla, Z, 1000 shots, GF(2) ----
    print("3: n = 16 without the dual, 17 qubits, ancilla c = f(x) XOR g(x)")
    var sv17 = evolve(no_dual_circuit(p))
    var ideal = 1.0 / 65536.0
    var law_dev = 0.0
    for idx in range(1 << 17):
        var y = idx & 0xFFFF
        var cbit = (idx >> 16) & 1
        var pr = sv17.re[idx] * sv17.re[idx] + sv17.im[idx] * sv17.im[idx]
        var want = ideal if cbit == parity(y & s_mask) else 0.0
        law_dev = max(law_dev, abs(pr - want))
    _ = log.close("no-dual exact law of (y, c) against uniform on c = s.y", law_dev, 0.0, 1e-9)

    var meas = no_dual_circuit(p)
    meas.measure_all()
    var res = sample(meas, 1000, UInt64(1234))
    var violations = 0
    var rows = List[Int]()
    var rhs = List[Int]()
    for m in res.memory:
        var y = m & 0xFFFF
        var cbit = (m >> 16) & 1
        if cbit != parity(y & s_mask):
            violations += 1
        rows.append(y)
        rhs.append(cbit)
    _ = log.check("1000 shots all satisfy c = s.y", violations == 0, "violations " + String(violations))
    var solved = solve_gf2(rows, rhs, 16)
    _ = log.check("GF(2) rank of the samples is 16", solved[1] == 16, "rank " + String(solved[1]))
    _ = log.check("GF(2) solution is the notebook s", solved[1] == 16 and bitstring(solved[0], 16) == s_text, bitstring(solved[0], 16))

    log.finish()
