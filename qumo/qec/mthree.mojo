"""M3: readout mitigation in the subspace of observed bitstrings (Nation et al.,
PRX Quantum 2, 040326).

`mitigate_counts` inverts the tensored confusion matrix over all 2^n strings, so
it stops at 24 qubits. Noisy counts only populate a few strings, so M3 keeps those
as a basis: the assignment matrix restricted to them (and to strings within
Hamming distance `hamming`, the rest being negligible) has at most
shots^2 entries, its columns are renormalised for the weight that left the
subspace, and A x = p is solved iteratively. Memory goes with the number of
distinct strings, never with 2^n.
"""

from std.bit import pop_count
from qumo.qec.characterization import ReadoutCalibration
from qumo.sim.statevector import run_chunks


def _pack(key: String, n: Int, words: Int) raises -> List[UInt64]:
    var out = List[UInt64](length=words, fill=0)
    var bs = key.as_bytes()
    var L = key.byte_length()
    if L > n:
        raise Error("QM-E709: bitstring of " + String(L) + " bits for a calibration of " + String(n) + " qubits. Hint: calibrate every measured qubit.")
    for i in range(L):
        if Int(bs[i]) == ord("1"):
            var q = L - 1 - i
            out[q // 64] |= UInt64(1) << UInt64(q % 64)
    return out^


struct _Sparse(Movable):
    """Compressed columns of the renormalised assignment matrix."""

    var start: List[Int]
    var row: List[Int]
    var val: List[Float64]
    var diag: List[Float64]

    def __init__(out self, m: Int):
        self.start = List[Int](length=m + 1, fill=0)
        self.row = List[Int]()
        self.val = List[Float64]()
        self.diag = List[Float64](length=m, fill=1.0)

    def mul(self, x: List[Float64], mut y: List[Float64]):
        """y = A x."""
        for i in range(len(y)):
            y[i] = 0.0
        for j in range(len(x)):
            var v = x[j]
            for k in range(self.start[j], self.start[j + 1]):
                y[self.row[k]] += self.val[k] * v


def _dot(a: List[Float64], b: List[Float64]) -> Float64:
    var s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def _solve_bicgstab(a: _Sparse, b: List[Float64], tol: Float64, max_iter: Int) raises -> List[Float64]:
    """Jacobi-preconditioned BiCGSTAB; M3's matrix is close to the identity."""
    var m = len(b)
    var x = List[Float64](capacity=m)
    for i in range(m):
        x.append(b[i] / a.diag[i])
    var r = List[Float64](length=m, fill=0.0)
    a.mul(x, r)
    for i in range(m):
        r[i] = b[i] - r[i]
    var r0 = r.copy()
    var p = List[Float64](length=m, fill=0.0)
    var v = List[Float64](length=m, fill=0.0)
    var y = List[Float64](length=m, fill=0.0)
    var s = List[Float64](length=m, fill=0.0)
    var z = List[Float64](length=m, fill=0.0)
    var t = List[Float64](length=m, fill=0.0)
    var rho = 1.0
    var alpha = 1.0
    var omega = 1.0
    var bnorm = _dot(b, b)
    for _ in range(max_iter):
        if _dot(r, r) <= tol * tol * bnorm:
            return x^
        var rho_new = _dot(r0, r)
        if rho_new == 0.0:
            break
        var beta = (rho_new / rho) * (alpha / omega)
        rho = rho_new
        for i in range(m):
            p[i] = r[i] + beta * (p[i] - omega * v[i])
            y[i] = p[i] / a.diag[i]
        a.mul(y, v)
        alpha = rho / _dot(r0, v)
        for i in range(m):
            s[i] = r[i] - alpha * v[i]
            z[i] = s[i] / a.diag[i]
        a.mul(z, t)
        var tt = _dot(t, t)
        omega = _dot(t, s) / tt if tt > 0.0 else 0.0
        for i in range(m):
            x[i] += alpha * y[i] + omega * z[i]
            r[i] = s[i] - omega * t[i]
    if _dot(r, r) > 1e-16 * bnorm:
        raise Error("QM-E709: M3 linear solve did not converge. Hint: the readout matrix is close to singular; recalibrate, or lower 'hamming'.")
    return x^


def mitigate_counts_m3(counts: Dict[String, Int], cal: ReadoutCalibration,
                       hamming: Int = -1) raises -> Dict[String, Float64]:
    """Quasi-probabilities over the observed strings, summing to 1.

    Keys are Qiskit-style bitstrings. `hamming`: largest error weight kept in the
    matrix, -1 picks 3 (M3's default level of accuracy needs about 2 to 3 for
    readout errors of a few percent)."""
    var n = cal.num_qubits
    var words = (n + 63) // 64
    var d = 3 if hamming < 0 else hamming
    var keys = List[String]()
    var total = 0
    for kv in counts.items():
        keys.append(kv.key)
        total += kv.value
    if total == 0:
        raise Error("QM-E709: empty counts.")
    var m = len(keys)
    var bits = List[UInt64](length=m * words, fill=0)
    var b = List[Float64](capacity=m)
    for i in range(m):
        var p = _pack(keys[i], n, words)
        for w in range(words):
            bits[i * words + w] = p[w]
        b.append(Float64(counts[keys[i]]) / Float64(total))
    # per-qubit factors: stay[q][bit] = P(read bit | true bit), flip[q][bit] = P(read not bit | true bit)
    var stay = List[Float64](capacity=2 * n)
    var flip = List[Float64](capacity=2 * n)
    for q in range(n):
        stay.append(1.0 - cal.p01[q])
        stay.append(1.0 - cal.p10[q])
        flip.append(cal.p01[q])
        flip.append(cal.p10[q])
    var a = _Sparse(m)
    # one chunk per column block, assembled in order afterwards
    var col_rows = List[List[Int]]()
    var col_vals = List[List[Float64]]()
    for _ in range(m):
        col_rows.append(List[Int]())
        col_vals.append(List[Float64]())
    var bp = bits.unsafe_ptr().as_unsafe_any_origin()
    var rp = col_rows.unsafe_ptr().as_unsafe_any_origin()
    var vp = col_vals.unsafe_ptr().as_unsafe_any_origin()

    def body(lo: Int, hi: Int) {imm bp, imm rp, imm vp, imm stay, imm flip, imm m, imm n, imm words, imm d}:
        for j in range(lo, hi):
            var base = 1.0
            for q in range(n):
                var tb = Int((bp[unsafe_offset=j * words + q // 64] >> UInt64(q % 64)) & 1)
                base *= stay[2 * q + tb]
            var norm = 0.0
            for i in range(m):
                var dist = 0
                for w in range(words):
                    dist += Int(pop_count(bp[unsafe_offset=i * words + w] ^ bp[unsafe_offset=j * words + w]))
                if dist > d:
                    continue
                var v = base
                for w in range(words):
                    var diff = bp[unsafe_offset=i * words + w] ^ bp[unsafe_offset=j * words + w]
                    var pos = 0
                    while diff != 0:
                        if (diff & 1) != 0:
                            var q = w * 64 + pos
                            var tb = Int((bp[unsafe_offset=j * words + w] >> UInt64(pos)) & 1)
                            v *= flip[2 * q + tb] / stay[2 * q + tb]
                        diff >>= 1
                        pos += 1
                rp[unsafe_offset=j].append(i)
                vp[unsafe_offset=j].append(v)
                norm += v
            for k in range(len(vp[unsafe_offset=j])):
                vp[unsafe_offset=j][k] /= norm

    run_chunks(body, m)
    _ = len(bits)
    for j in range(m):
        a.start[j] = len(a.row)
        for k in range(len(col_rows[j])):
            a.row.append(col_rows[j][k])
            a.val.append(col_vals[j][k])
            if col_rows[j][k] == j:
                a.diag[j] = col_vals[j][k]
    a.start[m] = len(a.row)
    var x = _solve_bicgstab(a, b, 1e-13, 2000)
    var out = Dict[String, Float64]()
    var sum = 0.0
    for i in range(m):
        sum += x[i]
    for i in range(m):
        out[keys[i]] = x[i] / sum
    return out^
