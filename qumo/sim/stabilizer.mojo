"""Stabilizer (Clifford) simulator: bit-packed *inverse* tableau.

Design (best-of synthesis):
  * Aaronson-Gottesman (CHP) tableau semantics, but stored the way Stim's
    `TableauSimulator` does: we keep the inverse tableau T^-1 of the Clifford
    C that prepared the state C|0...0>. Row `X_q` holds C^dag X_q C and row
    `Z_q` holds C^dag Z_q C as bit-packed Pauli strings (UInt64 words) with a
    sign bit.
      - Gates are row operations: T'^-1(P) = T^-1(G^dag P G), i.e. an O(n/64)
        Pauli-string product per generator (popcount phase tracking).
      - Measuring Z_q is deterministic iff row Z_q has no X support - an
        O(n/64) scan - and the outcome is simply its sign bit. This is what
        makes QEC reference sampling fast: after the first round every
        stabilizer measurement is deterministic.
      - Random outcomes are collapsed by prepending gates "at the beginning of
        time" (CX fan-in + H on a pivot input qubit, Stim's trick), which are
        column operations on the tableau.
  * Rotations by multiples of pi/2 (RX/RY/RZ/P/RXX/RYY/RZZ), SX, ISWAP, ECR,
    CY... are lowered to the H/S/CX/CZ primitives.
  * Mid-circuit measurement, reset, classical feed-forward (`c_if`) and Pauli
    noise channels are supported, so one engine produces the noiseless
    reference sample for `qumo.sim.frame` and exact per-shot trajectories for
    small dynamic Clifford circuits (thousands of qubits are fine: memory is
    2n^2/8 bytes per bit-plane).

Error codes: QM-E701 (non-Clifford), QM-E702 (unbound parameter),
QM-E705 (register too large for the requested output).
"""

from std.bit import pop_count, count_trailing_zeros
from std.math import pi

from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary, op_is_noise
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.pauli import PauliTerm
from qumo.sim.rng import Rng
from qumo.sim.runner import SampleResult

comptime WPtr = MutPointer[UInt64, MutAnyOrigin]


@always_inline
def _phase_word(x1: UInt64, z1: UInt64, x2: UInt64, z2: UInt64) -> Int:
    """Sum over the 64 qubits of the log_i phase of (P1_j * P2_j).

    Per qubit, Hermitian Paulis multiply as XY = iZ, YZ = iX, ZX = iY (+1) and
    the reverse orders give -1.
    """
    var ax = x1 & ~z1
    var ay = x1 & z1
    var az = ~x1 & z1
    var bx = x2 & ~z2
    var by = x2 & z2
    var bz = ~x2 & z2
    var plus = (ax & by) | (ay & bz) | (az & bx)
    var minus = (ay & bx) | (az & by) | (ax & bz)
    return Int(pop_count(plus)) - Int(pop_count(minus))


def quarter_turns(theta: Float64) raises -> Int:
    """Return k in 0..3 with theta = k * pi/2 (mod 2 pi); QM-E701 otherwise."""
    var r = theta / (pi / 2)
    var k = Int(round(r))
    if abs(r - Float64(k)) > 1e-9:
        raise Error(
            "QM-E701: rotation angle " + String(theta)
            + " is not a multiple of pi/2 (non-Clifford). Hint: use the"
            " state-vector simulator (qumo.sim.runner) for non-Clifford circuits."
        )
    return ((k % 4) + 4) % 4


struct TableauSimulator(Copyable):
    """Inverse-tableau stabilizer simulator (see module docstring).

    Rows 0..n-1 are C^dag X_q C, rows n..2n-1 are C^dag Z_q C, row 2n is a
    scratch row used for Pauli expectation values.
    """

    var n: Int
    var nw: Int
    var xs: List[UInt64]
    var zs: List[UInt64]
    var signs: List[UInt8]
    var rng: Rng
    var records: List[UInt8]  # measurement record (Stim `rec`), one entry per MEASURE
    var creg: List[UInt8]  # classical register

    def __init__(out self, num_qubits: Int, seed: UInt64 = 0):
        self.n = num_qubits
        self.nw = max(1, (num_qubits + 63) // 64)
        var rows = 2 * num_qubits + 1
        self.xs = List[UInt64](length=rows * self.nw, fill=0)
        self.zs = List[UInt64](length=rows * self.nw, fill=0)
        self.signs = List[UInt8](length=rows, fill=0)
        self.rng = Rng(seed)
        self.records = List[UInt8]()
        self.creg = List[UInt8]()
        self.reset_state()

    def reset_state(mut self):
        """Back to |0...0> (identity tableau); clears records and creg."""
        for i in range(len(self.xs)):
            self.xs[i] = 0
            self.zs[i] = 0
        for i in range(len(self.signs)):
            self.signs[i] = 0
        for q in range(self.n):
            var w = q >> 6
            var m = UInt64(1) << UInt64(q & 63)
            self.xs[q * self.nw + w] = m
            self.zs[(self.n + q) * self.nw + w] = m
        self.records.clear()
        self.creg.clear()

    @always_inline
    def _xp(mut self) -> WPtr:
        return self.xs.unsafe_ptr().as_unsafe_any_origin()

    @always_inline
    def _zp(mut self) -> WPtr:
        return self.zs.unsafe_ptr().as_unsafe_any_origin()

    # ------------------------------------------------------------ row algebra
    def _row_mul(mut self, dst: Int, src: Int, extra: Int):
        """row[dst] := i^extra * row[dst] * row[src]  (result must be Hermitian)."""
        var xp = self._xp()
        var zp = self._zp()
        var nw = self.nw
        var d0 = dst * nw
        var s0 = src * nw
        var acc = 0
        for w in range(nw):
            var x1 = xp[unsafe_offset=d0 + w]
            var z1 = zp[unsafe_offset=d0 + w]
            var x2 = xp[unsafe_offset=s0 + w]
            var z2 = zp[unsafe_offset=s0 + w]
            acc += _phase_word(x1, z1, x2, z2)
            xp[unsafe_offset=d0 + w] = x1 ^ x2
            zp[unsafe_offset=d0 + w] = z1 ^ z2
        var e = (2 * Int(self.signs[dst]) + 2 * Int(self.signs[src]) + acc + extra) & 3
        self.signs[dst] = UInt8(e >> 1)

    def _swap_rows(mut self, a: Int, b: Int):
        var xp = self._xp()
        var zp = self._zp()
        var nw = self.nw
        for w in range(nw):
            var tx = xp[unsafe_offset=a * nw + w]
            xp[unsafe_offset=a * nw + w] = xp[unsafe_offset=b * nw + w]
            xp[unsafe_offset=b * nw + w] = tx
            var tz = zp[unsafe_offset=a * nw + w]
            zp[unsafe_offset=a * nw + w] = zp[unsafe_offset=b * nw + w]
            zp[unsafe_offset=b * nw + w] = tz
        var ts = self.signs[a]
        self.signs[a] = self.signs[b]
        self.signs[b] = ts

    # ------------------------------------------------------------ gates
    def x(mut self, q: Int):
        self.signs[self.n + q] ^= 1

    def z(mut self, q: Int):
        self.signs[q] ^= 1

    def y(mut self, q: Int):
        self.signs[q] ^= 1
        self.signs[self.n + q] ^= 1

    def h(mut self, q: Int):
        self._swap_rows(q, self.n + q)

    def s(mut self, q: Int):
        # S^dag X S = -Y = -i X Z
        self._row_mul(q, self.n + q, 3)

    def sdg(mut self, q: Int):
        # S X S^dag = Y = i X Z
        self._row_mul(q, self.n + q, 1)

    def sx(mut self, q: Int):
        # SX^dag Z SX = Y = -i Z X
        self._row_mul(self.n + q, q, 3)

    def sxdg(mut self, q: Int):
        # SX Z SX^dag = -Y = i Z X
        self._row_mul(self.n + q, q, 1)

    def cx(mut self, c: Int, t: Int):
        self._row_mul(c, t, 0)  # X_c -> X_c X_t
        self._row_mul(self.n + t, self.n + c, 0)  # Z_t -> Z_c Z_t

    def cz(mut self, a: Int, b: Int):
        self._row_mul(a, self.n + b, 0)  # X_a -> X_a Z_b
        self._row_mul(b, self.n + a, 0)  # X_b -> Z_a X_b

    def cy(mut self, c: Int, t: Int):
        self.sdg(t)
        self.cx(c, t)
        self.s(t)

    def swap(mut self, a: Int, b: Int):
        self._swap_rows(a, b)
        self._swap_rows(self.n + a, self.n + b)

    def iswap(mut self, a: Int, b: Int):
        # iSWAP = (S x S) SWAP CZ
        self.cz(a, b)
        self.swap(a, b)
        self.s(a)
        self.s(b)

    def iswapdg(mut self, a: Int, b: Int):
        self.sdg(a)
        self.sdg(b)
        self.swap(a, b)
        self.cz(a, b)

    def ecr(mut self, a: Int, b: Int):
        # ECR = (X_a SX_b) CX(a,b) S_a  (up to global phase)
        self.s(a)
        self.cx(a, b)
        self.x(a)
        self.sx(b)

    def rx_quarter(mut self, q: Int, k: Int):
        if k == 1: self.sx(q)
        elif k == 2: self.x(q)
        elif k == 3: self.sxdg(q)

    def ry_quarter(mut self, q: Int, k: Int):
        if k == 1:  # RY(pi/2) = H Z
            self.z(q)
            self.h(q)
        elif k == 2:
            self.y(q)
        elif k == 3:  # RY(-pi/2) = Z H
            self.h(q)
            self.z(q)

    def rz_quarter(mut self, q: Int, k: Int):
        if k == 1: self.s(q)
        elif k == 2: self.z(q)
        elif k == 3: self.sdg(q)

    def rzz_quarter(mut self, a: Int, b: Int, k: Int):
        if k == 1:
            self.cz(a, b)
            self.s(a)
            self.s(b)
        elif k == 2:
            self.z(a)
            self.z(b)
        elif k == 3:
            self.cz(a, b)
            self.sdg(a)
            self.sdg(b)

    def apply_gate(mut self, g: Instruction) raises:
        """Apply one unitary Clifford instruction (condition NOT checked here)."""
        var k = g.kind
        if k == Op.I or k == Op.BARRIER:
            return
        if k == Op.X: self.x(g.q0)
        elif k == Op.Y: self.y(g.q0)
        elif k == Op.Z: self.z(g.q0)
        elif k == Op.H: self.h(g.q0)
        elif k == Op.S: self.s(g.q0)
        elif k == Op.SDG: self.sdg(g.q0)
        elif k == Op.SX: self.sx(g.q0)
        elif k == Op.SXDG: self.sxdg(g.q0)
        elif k == Op.CX: self.cx(g.q0, g.q1)
        elif k == Op.CY: self.cy(g.q0, g.q1)
        elif k == Op.CZ: self.cz(g.q0, g.q1)
        elif k == Op.SWAP: self.swap(g.q0, g.q1)
        elif k == Op.ISWAP: self.iswap(g.q0, g.q1)
        elif k == Op.ISWAPDG: self.iswapdg(g.q0, g.q1)
        elif k == Op.ECR: self.ecr(g.q0, g.q1)
        elif k == Op.RX or k == Op.RY or k == Op.RZ or k == Op.P or k == Op.RZZ or k == Op.RXX or k == Op.RYY:
            if g.p0.is_symbolic():
                raise Error("QM-E702: '" + op_name(k) + "' has an unbound parameter. Hint: call circuit.bind(values) before stabilizer simulation.")
            var t = quarter_turns(g.p0.value())
            if k == Op.RX: self.rx_quarter(g.q0, t)
            elif k == Op.RY: self.ry_quarter(g.q0, t)
            elif k == Op.RZ or k == Op.P: self.rz_quarter(g.q0, t)
            elif k == Op.RZZ: self.rzz_quarter(g.q0, g.q1, t)
            elif k == Op.RXX:
                self.h(g.q0)
                self.h(g.q1)
                self.rzz_quarter(g.q0, g.q1, t)
                self.h(g.q0)
                self.h(g.q1)
            else:  # RYY = (V x V) RZZ (V^dag x V^dag), V = SXDG
                self.sx(g.q0)
                self.sx(g.q1)
                self.rzz_quarter(g.q0, g.q1, t)
                self.sxdg(g.q0)
                self.sxdg(g.q1)
        else:
            raise Error(
                "QM-E701: gate '" + op_name(k) + "' is not Clifford. Hint: the"
                " stabilizer/frame simulators accept H,S,SDG,SX,SXDG,X,Y,Z,CX,CY,CZ,SWAP,"
                "ISWAP,ISWAPDG,ECR and rotations by multiples of pi/2; use qumo.sim.runner"
                " for general circuits."
            )

    # ------------------------------------------------------- column (input) ops
    # These prepend a gate G at the beginning of time (C' = C G); every row is
    # conjugated by G. Used only to collapse random measurements.
    def _col_cx(mut self, c: Int, t: Int):
        var xp = self._xp()
        var zp = self._zp()
        var nw = self.nw
        var wc = c >> 6
        var wt = t >> 6
        var bc = UInt64(c & 63)
        var bt = UInt64(t & 63)
        var mc = UInt64(1) << bc
        var mt = UInt64(1) << bt
        for r in range(2 * self.n):
            var o = r * nw
            var xc = (xp[unsafe_offset=o + wc] >> bc) & 1
            var zc = (zp[unsafe_offset=o + wc] >> bc) & 1
            var xt = (xp[unsafe_offset=o + wt] >> bt) & 1
            var zt = (zp[unsafe_offset=o + wt] >> bt) & 1
            if (xc & zt & (xt ^ zc ^ 1)) != 0:
                self.signs[r] ^= 1
            if xc != 0:
                xp[unsafe_offset=o + wt] = xp[unsafe_offset=o + wt] ^ mt
            if zt != 0:
                zp[unsafe_offset=o + wc] = zp[unsafe_offset=o + wc] ^ mc

    def _col_h(mut self, c: Int):
        var xp = self._xp()
        var zp = self._zp()
        var nw = self.nw
        var wc = c >> 6
        var bc = UInt64(c & 63)
        var mc = UInt64(1) << bc
        for r in range(2 * self.n):
            var o = r * nw + wc
            var xv = xp[unsafe_offset=o]
            var zv = zp[unsafe_offset=o]
            var xb = (xv >> bc) & 1
            var zb = (zv >> bc) & 1
            if (xb & zb) != 0:
                self.signs[r] ^= 1
            if xb != zb:
                xp[unsafe_offset=o] = xv ^ mc
                zp[unsafe_offset=o] = zv ^ mc

    def _col_hyz(mut self, c: Int):
        """Conjugate by H_YZ: X -> -X, Y <-> Z."""
        var xp = self._xp()
        var zp = self._zp()
        var nw = self.nw
        var wc = c >> 6
        var bc = UInt64(c & 63)
        var mc = UInt64(1) << bc
        for r in range(2 * self.n):
            var o = r * nw + wc
            var xb = (xp[unsafe_offset=o] >> bc) & 1
            var zb = (zp[unsafe_offset=o] >> bc) & 1
            if xb == 1 and zb == 0:
                self.signs[r] ^= 1
            if zb != 0:
                xp[unsafe_offset=o] = xp[unsafe_offset=o] ^ mc

    def _col_x(mut self, c: Int):
        var zp = self._zp()
        var nw = self.nw
        var wc = c >> 6
        var bc = UInt64(c & 63)
        for r in range(2 * self.n):
            if ((zp[unsafe_offset=r * nw + wc] >> bc) & 1) != 0:
                self.signs[r] ^= 1

    # ------------------------------------------------------------ measurement
    def peek_z(mut self, q: Int) -> Int:
        """<Z_q> without collapsing: +1, -1 or 0 (random)."""
        var base = (self.n + q) * self.nw
        for w in range(self.nw):
            if self.xs[base + w] != 0:
                return 0
        return -1 if self.signs[self.n + q] != 0 else 1

    def is_deterministic(mut self, q: Int) -> Bool:
        return self.peek_z(q) != 0

    def measure(mut self, q: Int, forced: Int = -1) -> Int:
        """Z-basis measurement with collapse. `forced` in {0,1} fixes the
        outcome of a *random* measurement (reference sampling); -1 = random."""
        var row = self.n + q
        var nw = self.nw
        var base = row * nw
        var pivot = -1
        for w in range(nw):
            var v = self.xs[base + w]
            if v != 0:
                pivot = w * 64 + Int(count_trailing_zeros(v))
                break
        if pivot < 0:
            return Int(self.signs[row])
        # Fan the other X-support of row Z_q into the pivot (no-op on |0>).
        var pw = pivot >> 6
        var snap = List[UInt64](capacity=nw - pw)
        for w in range(pw, nw):
            snap.append(self.xs[base + w])
        snap[0] = snap[0] & ~(UInt64(1) << UInt64(pivot & 63))
        for i in range(len(snap)):
            var v = snap[i]
            while v != 0:
                var b = Int(count_trailing_zeros(v))
                v &= v - 1
                self._col_cx(pivot, (pw + i) * 64 + b)
        var zb = (self.zs[base + pw] >> UInt64(pivot & 63)) & 1
        if zb != 0:
            self._col_hyz(pivot)
        else:
            self._col_h(pivot)
        var m: Int
        if forced >= 0:
            m = forced
        else:
            m = Int(self.rng.next_u64() >> 63)
        if Int(self.signs[row]) != m:
            self._col_x(pivot)
        return m

    def reset(mut self, q: Int):
        if self.measure(q, 0) == 1:
            self.x(q)

    # ------------------------------------------------------------ expectations
    def _scratch_mul(mut self, src: Int) -> Int:
        """scratch := scratch * row[src]; returns the log_i phase (incl. src sign)."""
        var xp = self._xp()
        var zp = self._zp()
        var nw = self.nw
        var d0 = 2 * self.n * nw
        var s0 = src * nw
        var acc = 0
        for w in range(nw):
            var x1 = xp[unsafe_offset=d0 + w]
            var z1 = zp[unsafe_offset=d0 + w]
            var x2 = xp[unsafe_offset=s0 + w]
            var z2 = zp[unsafe_offset=s0 + w]
            acc += _phase_word(x1, z1, x2, z2)
            xp[unsafe_offset=d0 + w] = x1 ^ x2
            zp[unsafe_offset=d0 + w] = z1 ^ z2
        return acc + 2 * Int(self.signs[src])

    def expectation_sparse(mut self, qubits: List[Int], paulis: List[Int]) -> Int:
        """<P> for P = prod_i paulis[i] on qubits[i] (1=X, 2=Y, 3=Z): +1, -1 or 0."""
        var nw = self.nw
        var d0 = 2 * self.n * nw
        for w in range(nw):
            self.xs[d0 + w] = 0
            self.zs[d0 + w] = 0
        var e = 0
        for i in range(len(qubits)):
            var q = qubits[i]
            var p = paulis[i]
            if p == 1 or p == 2:
                e += self._scratch_mul(q)
            if p == 2 or p == 3:
                e += self._scratch_mul(self.n + q)
            if p == 2:
                e += 1  # Y = i X Z
        for w in range(nw):
            if self.xs[d0 + w] != 0:
                return 0
        return -1 if ((e & 3) >> 1) == 1 else 1

    def expectation_term(mut self, t: PauliTerm) -> Int:
        """<P> (+1/-1/0) for a <=64-qubit PauliTerm (coefficient ignored)."""
        var qs = List[Int]()
        var ps = List[Int]()
        for q in range(min(64, self.n)):
            var b = UInt64(1) << UInt64(q)
            var hx = (t.x & b) != 0
            var hz = (t.z & b) != 0
            if hx or hz:
                qs.append(q)
                ps.append(2 if (hx and hz) else (1 if hx else 3))
        return self.expectation_sparse(qs, ps)

    def expectation_label(mut self, label: String) raises -> Int:
        """<P> for a dense label (leftmost char = highest qubit, Qiskit style)."""
        var n = label.byte_length()
        var bs = label.as_bytes()
        var qs = List[Int]()
        var ps = List[Int]()
        for i in range(n):
            var q = n - 1 - i
            var ch = Int(bs[i])
            if ch == ord("I"):
                continue
            if q >= self.n:
                raise Error("QM-E705: Pauli label longer than the simulator register (" + String(self.n) + " qubits).")
            qs.append(q)
            if ch == ord("X"): ps.append(1)
            elif ch == ord("Y"): ps.append(2)
            elif ch == ord("Z"): ps.append(3)
            else:
                raise Error("QM-E301: invalid Pauli character in '" + label + "'. Hint: use only I, X, Y, Z.")
        return self.expectation_sparse(qs, ps)

    # ------------------------------------------------------------ noise
    def _pauli(mut self, which: Int, q: Int):
        if which == 1: self.x(q)
        elif which == 2: self.y(q)
        elif which == 3: self.z(q)

    def apply_noise(mut self, g: Instruction):
        var k = g.kind
        var p = g.p0.value()
        if k == Op.X_ERROR:
            if self.rng.uniform() < p: self.x(g.q0)
        elif k == Op.Y_ERROR:
            if self.rng.uniform() < p: self.y(g.q0)
        elif k == Op.Z_ERROR:
            if self.rng.uniform() < p: self.z(g.q0)
        elif k == Op.DEPOLARIZE1:
            if self.rng.uniform() < p: self._pauli(1 + self.rng.below(3), g.q0)
        elif k == Op.PAULI_CHANNEL_1:
            var u = self.rng.uniform()
            var px = g.p0.value()
            var py = g.p1.value()
            var pz = g.p2.value()
            if u < px: self.x(g.q0)
            elif u < px + py: self.y(g.q0)
            elif u < px + py + pz: self.z(g.q0)
        elif k == Op.DEPOLARIZE2:
            if self.rng.uniform() < p:
                var r = 1 + self.rng.below(15)
                self._pauli(r & 3, g.q0)
                self._pauli(r >> 2, g.q1)

    # ------------------------------------------------------------ circuits
    def run(mut self, circuit: Circuit, noisy: Bool = True, forced_zero: Bool = False) raises:
        """Execute one shot of `circuit` from the current state.

        Measurement results are appended to `records` (Stim `rec` order) and
        written to `creg`. `forced_zero` makes random outcomes 0 (Stim's
        reference-sample convention); `noisy=False` skips noise channels.
        """
        if circuit.num_qubits > self.n:
            raise Error("QM-E705: circuit has " + String(circuit.num_qubits) + " qubits but the simulator was built for " + String(self.n) + ". Hint: TableauSimulator(circuit.num_qubits).")
        if len(self.creg) < circuit.num_clbits:
            for _ in range(circuit.num_clbits - len(self.creg)):
                self.creg.append(0)
        var forced = 0 if forced_zero else -1
        for g in circuit.ops:
            if g.cond >= 0:
                if not self._condition_holds(circuit, g):
                    continue
            var k = g.kind
            if op_is_unitary(k):
                self.apply_gate(g)
            elif k == Op.MEASURE:
                var m = self.measure(g.q0, forced)
                self.records.append(UInt8(m))
                if g.clbit >= 0:
                    self.creg[g.clbit] = UInt8(m)
            elif k == Op.RESET:
                self.reset(g.q0)
            elif op_is_noise(k):
                if noisy:
                    self.apply_noise(g)

    def _condition_holds(self, circuit: Circuit, g: Instruction) -> Bool:
        """Single-bit (`c_if`) or register (`c_if_register`) classical condition."""
        if g.cond_reg < 0:
            return Int(self.creg[g.cond]) == g.cond_val
        var n = circuit.cond_table[g.cond_reg]
        for i in range(n):
            var b = circuit.cond_table[g.cond_reg + 1 + 2 * i]
            var v = circuit.cond_table[g.cond_reg + 2 + 2 * i]
            if Int(self.creg[b]) != v:
                return False
        return True

    def detector_values(self, circuit: Circuit) -> List[UInt8]:
        """Parity of each DETECTOR's records in the last run (absolute values)."""
        var out = List[UInt8](length=circuit.num_detectors, fill=0)
        for g in circuit.ops:
            if g.kind == Op.DETECTOR:
                var v: UInt8 = 0
                for j in range(g.q1):
                    v ^= self.records[circuit.records[g.q0 + j]]
                out[g.q2] = v
        return out^

    def observable_values(self, circuit: Circuit) -> List[UInt8]:
        var out = List[UInt8](length=circuit.num_observables, fill=0)
        for g in circuit.ops:
            if g.kind == Op.OBSERVABLE:
                var v: UInt8 = 0
                for j in range(g.q1):
                    v ^= self.records[circuit.records[g.q0 + j]]
                out[g.q2] ^= v
        return out^


def reference_sample(circuit: Circuit) raises -> List[UInt8]:
    """Noiseless measurement record with every random outcome forced to 0.

    This is the reference that Pauli-frame simulation (qumo.sim.frame) XORs
    its bit-packed measurement flips onto (Stim's `reference_sample`).
    """
    var sim = TableauSimulator(circuit.num_qubits, 0)
    sim.run(circuit, noisy=False, forced_zero=True)
    return sim.records.copy()


def sample_stabilizer(circuit: Circuit, shots: Int, seed: UInt64 = 1234) raises -> SampleResult:
    """Per-shot tableau trajectories -> counts over the classical register.

    Exact for any Clifford circuit with mid-circuit measurement, reset,
    arbitrary feed-forward and Pauli noise. For high-throughput sampling of
    QEC circuits use `qumo.sim.frame.FrameSimulator` instead.
    """
    circuit.validate()
    if circuit.num_clbits > 62:
        raise Error("QM-E705: sample_stabilizer() returns integer-keyed counts and supports <= 62 clbits (got " + String(circuit.num_clbits) + "). Hint: use TableauSimulator.run() records or FrameSimulator.sample_measurements().")
    if not circuit.is_clifford():
        raise Error("QM-E701: circuit is not Clifford. Hint: use qumo.sim.runner.sample() (state vector).")
    var res = SampleResult(circuit.num_clbits, shots, "stabilizer:tableau")
    var root = Rng(seed)
    var sim = TableauSimulator(circuit.num_qubits, 0)
    for shot in range(shots):
        sim.reset_state()
        sim.rng = root.split(shot)
        sim.run(circuit)
        var creg = 0
        for c in range(circuit.num_clbits):
            if sim.creg[c] != 0:
                creg |= 1 << c
        res.record(creg)
    return res^
