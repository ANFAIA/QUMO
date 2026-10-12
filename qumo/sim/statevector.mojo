"""High-performance state-vector simulator.

Performance design (best-of qsim / Qiskit-Aer / cuStateVec / lightning):
  * Split re/im storage (SoA) -> clean SIMD FMA streams, no complex shuffles.
  * Pair-index kernels: amplitude pairs are enumerated by inserting zero bits
    (`insert_zero`) so every kernel is a flat, branch-free loop that SIMD-izes
    and multi-threads by chunks.
  * Low-qubit SIMD kernels: when a gate touches a qubit below log2(W) the
    pair partner lives inside the same W-lane vector, so the kernel works on
    whole vectors with lane shuffles / lane masks (qsim's "low qubit" kernels)
    instead of falling back to scalar code.
  * Specialised kernels: diagonal gates touch only phases, CX/SWAP/CCX are pure
    permutations (zero flops), controlled gates touch 1/2 of the state.
  * `apply_swap_layer` exchanges several pairs of amplitude-index bits in one
    in-place, cache-tiled pass (cuStateVec `swapIndexBits`, Aer chunk swap).
    The runner uses it to re-map qubits so that cache blocking covers almost
    every gate (see `qumo.sim.runner` and docs/SIMULATION.md).
  * The runner (`qumo.sim.runner`) fuses runs of 1-qubit gates into one 2x2
    matrix before they reach these kernels (qsim/Aer "gate fusion").
"""

from std.math import sqrt, cos, sin
from std.runtime import parallelism_level
from std.bit import pop_count
from max.algorithm import parallelize

from qumo.ir.gates import M2, M4
from qumo.ir.pauli import PauliSum
from qumo.sim.rng import Rng

comptime W = 8
comptime LOGW = 3
comptime PAR_MIN_ITEMS = 1 << 13
comptime FPtr = MutPointer[Float64, MutAnyOrigin]


@always_inline
def insert_zero(p: Int, b: Int) -> Int:
    """Insert a 0 bit at position b of p."""
    return ((p >> b) << (b + 1)) | (p & ((1 << b) - 1))


@always_inline
def insert_zero2(p: Int, lo: Int, hi: Int) -> Int:
    return insert_zero(insert_zero(p, lo), hi)


# ---------------------------------------------------------------------------
# Lane helpers for gates on qubits < LOGW (partner amplitude in the same
# W-lane vector). Lane j of a W-aligned group holds index (group | j).
# ---------------------------------------------------------------------------
comptime VF = SIMD[DType.float64, W]
comptime VB = SIMD[DType.bool, W]


@always_inline
def _flip[t: Int](v: VF) -> VF:
    """Lane j <- lane j ^ (1 << t)  (t < LOGW)."""
    comptime if t == 0:
        return v.shuffle[1, 0, 3, 2, 5, 4, 7, 6]()
    elif t == 1:
        return v.shuffle[2, 3, 0, 1, 6, 7, 4, 5]()
    else:
        return v.shuffle[4, 5, 6, 7, 0, 1, 2, 3]()


@always_inline
def _lane_mask(t: Int) -> VB:
    """True on lanes whose bit t is set (t < LOGW)."""
    var f = VF(0)
    comptime for j in range(W):
        if (j >> t) & 1 == 1:
            f[j] = 1.0
    return f.eq(VF(1.0))


@always_inline
def _lane_pick(t: Int, x0: Float64, x1: Float64) -> VF:
    """x0 on lanes with bit t clear, x1 on lanes with bit t set (t < LOGW)."""
    var f = VF(x0)
    comptime for j in range(W):
        if (j >> t) & 1 == 1:
            f[j] = x1
    return f


@always_inline
def _grp(g: Int, hb: Int, set_mask: Int) -> Int:
    """First index of W-group g: groups enumerate indices with bit hb (>= LOGW)
    clear (hb < 0: all groups), then `set_mask` is OR-ed in."""
    if hb < 0:
        return g << LOGW
    return (insert_zero(g, hb - LOGW) << LOGW) | set_mask


@always_inline
def _mat_loop[t: Int](rp: FPtr, ip: FPtr, ar: VF, ai: VF, br: VF, bi: VF,
                      glo: Int, ghi: Int, hb: Int, sm: Int):
    """v <- A*v + B*flip_t(v) on groups [glo, ghi) (lane-pattern 2x2)."""
    for g in range(glo, ghi):
        var i = _grp(g, hb, sm)
        var vr = rp.unsafe_load[width=W](i)
        var vi = ip.unsafe_load[width=W](i)
        var sr = _flip[t](vr)
        var si = _flip[t](vi)
        rp.unsafe_store(i, ar * vr - ai * vi + br * sr - bi * si)
        ip.unsafe_store(i, ar * vi + ai * vr + br * si + bi * sr)


@always_inline
def _mat_loop_rt(t: Int, rp: FPtr, ip: FPtr, ar: VF, ai: VF, br: VF, bi: VF,
                 glo: Int, ghi: Int, hb: Int, sm: Int):
    if t == 0:
        _mat_loop[0](rp, ip, ar, ai, br, bi, glo, ghi, hb, sm)
    elif t == 1:
        _mat_loop[1](rp, ip, ar, ai, br, bi, glo, ghi, hb, sm)
    else:
        _mat_loop[2](rp, ip, ar, ai, br, bi, glo, ghi, hb, sm)


@always_inline
def _diag_loop(rp: FPtr, ip: FPtr, dr: VF, di: VF, glo: Int, ghi: Int, hb: Int, sm: Int):
    """v <- d * v (lane-wise complex multiply) on groups [glo, ghi)."""
    for g in range(glo, ghi):
        var i = _grp(g, hb, sm)
        var vr = rp.unsafe_load[width=W](i)
        var vi = ip.unsafe_load[width=W](i)
        rp.unsafe_store(i, dr * vr - di * vi)
        ip.unsafe_store(i, dr * vi + di * vr)


@always_inline
def _perm_loop[t: Int](rp: FPtr, ip: FPtr, mask: VB, glo: Int, ghi: Int, hb: Int, sm: Int):
    """Lanes selected by `mask` take the value of lane j ^ (1 << t)."""
    for g in range(glo, ghi):
        var i = _grp(g, hb, sm)
        var vr = rp.unsafe_load[width=W](i)
        var vi = ip.unsafe_load[width=W](i)
        rp.unsafe_store(i, mask.select(_flip[t](vr), vr))
        ip.unsafe_store(i, mask.select(_flip[t](vi), vi))


@always_inline
def _perm_loop_rt(t: Int, rp: FPtr, ip: FPtr, mask: VB, glo: Int, ghi: Int, hb: Int, sm: Int):
    if t == 0:
        _perm_loop[0](rp, ip, mask, glo, ghi, hb, sm)
    elif t == 1:
        _perm_loop[1](rp, ip, mask, glo, ghi, hb, sm)
    else:
        _perm_loop[2](rp, ip, mask, glo, ghi, hb, sm)


@always_inline
def run_chunks[F: def(Int, Int) -> None](f: F, total: Int):
    """Run f(lo, hi) over [0, total) split into SIMD-aligned parallel chunks."""
    if total < PAR_MIN_ITEMS:
        f(0, total)
        return
    var nw = parallelism_level() * 2
    if nw < 1:
        nw = 1
    var chunk = ((total + nw - 1) // nw + W - 1) // W * W
    var nchunks = (total + chunk - 1) // chunk

    def task(c: Int) {imm f, imm chunk, imm total}:
        var lo = c * chunk
        var hi = min(lo + chunk, total)
        f(lo, hi)

    parallelize(task, nchunks)


comptime LOG_LINE = 4  # log2(Float64 values per 128-byte cache line)
comptime SWAP_RUN_BITS = 12  # swap-layer work item: <= 2^12 amplitudes (32 KiB re + 32 KiB im)


@always_inline
def _swap_runs(rp: FPtr, ip: FPtr, i: Int, j: Int, run: Int):
    """Exchange the `run` (a multiple of W) contiguous amplitudes starting at
    i with those starting at j, in place (vector loads/stores, no buffer)."""
    var e = 0
    while e < run:
        var ar = rp.unsafe_load[width=W](i + e)
        var ai = ip.unsafe_load[width=W](i + e)
        rp.unsafe_store(i + e, rp.unsafe_load[width=W](j + e))
        ip.unsafe_store(i + e, ip.unsafe_load[width=W](j + e))
        rp.unsafe_store(j + e, ar)
        ip.unsafe_store(j + e, ai)
        e += W


@always_inline
def _swap_in_lane_loop[t: Int, u: Int](rp: FPtr, ip: FPtr, glo: Int, ghi: Int):
    """SWAP of two in-vector bits t < u < LOGW on W-groups [glo, ghi): lanes
    whose bits t and u differ take lane j ^ (1 << t) ^ (1 << u)."""
    var f = VF(0)
    comptime for j in range(W):
        if ((j >> t) ^ (j >> u)) & 1 == 1:
            f[j] = 1.0
    var mask = f.eq(VF(1.0))
    for g in range(glo, ghi):
        var i = g << LOGW
        var vr = rp.unsafe_load[width=W](i)
        var vi = ip.unsafe_load[width=W](i)
        rp.unsafe_store(i, mask.select(_flip[u](_flip[t](vr)), vr))
        ip.unsafe_store(i, mask.select(_flip[u](_flip[t](vi)), vi))


@always_inline
def _swap_cross_loop[t: Int](rp: FPtr, ip: FPtr, hb: Int, glo: Int, ghi: Int):
    """SWAP of in-vector bit t < LOGW with bit hb >= LOGW on groups [glo, ghi)
    (a group = the vector pair i0 (bit hb clear), i1 = i0 | 1 << hb):
    element (hb=0, t=1) <-> (hb=1, t=0)."""
    var mt = _lane_mask(t)
    var stride = 1 << hb
    for g in range(glo, ghi):
        var i0 = insert_zero(g, hb - LOGW) << LOGW
        var i1 = i0 + stride
        var ar = rp.unsafe_load[width=W](i0)
        var ai = ip.unsafe_load[width=W](i0)
        var br = rp.unsafe_load[width=W](i1)
        var bi = ip.unsafe_load[width=W](i1)
        rp.unsafe_store(i0, mt.select(_flip[t](br), ar))
        ip.unsafe_store(i0, mt.select(_flip[t](bi), ai))
        rp.unsafe_store(i1, mt.select(br, _flip[t](ar)))
        ip.unsafe_store(i1, mt.select(bi, _flip[t](ai)))


struct StateVector(Copyable, Writable):
    var num_qubits: Int
    var re: List[Float64]
    var im: List[Float64]

    def __init__(out self, num_qubits: Int):
        self.num_qubits = num_qubits
        var n = 1 << num_qubits
        self.re = List[Float64](length=n, fill=0.0)
        self.im = List[Float64](length=n, fill=0.0)
        self.re[0] = 1.0

    @staticmethod
    def basis(num_qubits: Int, index: Int) -> StateVector:
        var sv = StateVector(num_qubits)
        sv.re[0] = 0.0
        sv.re[index] = 1.0
        return sv^

    def size(self) -> Int:
        return 1 << self.num_qubits

    def _rp(mut self) -> FPtr:
        return self.re.unsafe_ptr().as_unsafe_any_origin()

    def _ip(mut self) -> FPtr:
        return self.im.unsafe_ptr().as_unsafe_any_origin()

    def reset_to_zero(mut self):
        for i in range(self.size()):
            self.re[i] = 0.0
            self.im[i] = 0.0
        self.re[0] = 1.0

    # ------------------------------------------------------------ 1q kernels
    # Each method is a thin parallel driver around a `k*_range` kernel; the
    # same kernels run sequentially on cache-sized blocks in
    # `qumo.sim.runner.apply_kop_block` (cache-blocked executor).
    def apply_1q(mut self, m: M2, t: Int):
        if m.is_diagonal():
            self.apply_diag_1q(m.re[0], m.im[0], m.re[3], m.im[3], t)
            return
        var rp = self._rp()
        var ip = self._ip()

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm t, imm m}:
            k1_range(rp, ip, t, m, lo, hi)

        run_chunks(body, self.size() >> 1)

    def apply_diag_1q(mut self, d0r: Float64, d0i: Float64, d1r: Float64, d1i: Float64, t: Int):
        """diag(d0, d1) on qubit t; skips the |0> half when d0 == 1."""
        var rp = self._rp()
        var ip = self._ip()

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm t, imm d0r, imm d0i, imm d1r, imm d1i}:
            kdiag_range(rp, ip, t, d0r, d0i, d1r, d1i, lo, hi)

        run_chunks(body, self.size() >> 1)

    def apply_x(mut self, t: Int):
        """Pauli-X as a pure permutation."""
        var rp = self._rp()
        var ip = self._ip()

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm t}:
            kx_range(rp, ip, t, lo, hi)

        run_chunks(body, self.size() >> 1)

    # ------------------------------------------------------------ 2q kernels
    def apply_controlled_1q(mut self, m: M2, c: Int, t: Int):
        """|1><1|_c (x) m_t  +  |0><0|_c (x) I."""
        var rp = self._rp()
        var ip = self._ip()

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm c, imm t, imm m}:
            kc1_range(rp, ip, c, t, m, lo, hi)

        run_chunks(body, self.size() >> 2)

    def apply_cx(mut self, c: Int, t: Int):
        """CNOT as a permutation inside the control=1 subspace."""
        var rp = self._rp()
        var ip = self._ip()

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm c, imm t}:
            kcx_range(rp, ip, c, t, lo, hi)

        run_chunks(body, self.size() >> 2)

    def apply_cphase(mut self, pr: Float64, pi_: Float64, a: Int, b: Int):
        """Multiply amplitudes with bits a and b both set by (pr + i pi_)."""
        var rp = self._rp()
        var ip = self._ip()

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm a, imm b, imm pr, imm pi_}:
            kcphase_range(rp, ip, a, b, pr, pi_, lo, hi)

        run_chunks(body, self.size() >> 2)

    def apply_rzz(mut self, theta: Float64, a: Int, b: Int):
        """exp(-i theta/2 Z_a Z_b): phase e^{-i t/2} on even parity, e^{+i t/2} on odd."""
        var rp = self._rp()
        var ip = self._ip()
        var cr = cos(theta / 2)
        var si = sin(theta / 2)

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm a, imm b, imm cr, imm si}:
            krzz_range(rp, ip, a, b, cr, si, lo, hi)

        run_chunks(body, self.size())

    def apply_swap(mut self, a: Int, b: Int):
        var rp = self._rp()
        var ip = self._ip()

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm a, imm b}:
            kswap_range(rp, ip, a, b, lo, hi)

        run_chunks(body, self.size() >> 2)

    def apply_2q(mut self, m: M4, q0: Int, q1: Int):
        """Generic dense 4x4 on (q0, q1); matrix index = b(q0) + 2 b(q1)."""
        var rp = self._rp()
        var ip = self._ip()

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm q0, imm q1, imm m}:
            k2_range(rp, ip, q0, q1, m, lo, hi)

        run_chunks(body, self.size() >> 2)

    # ------------------------------------------------------------ 3q kernels
    def apply_ccx(mut self, c0: Int, c1: Int, t: Int):
        var rp = self._rp()
        var ip = self._ip()
        var s = List[Int]()
        s.append(c0); s.append(c1); s.append(t)
        sort(s)
        var b0 = s[0]; var b1 = s[1]; var b2 = s[2]
        var cmask = (1 << c0) | (1 << c1)
        var tb = 1 << t

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm b0, imm b1, imm b2, imm cmask, imm tb}:
            for p in range(lo, hi):
                var i0 = insert_zero(insert_zero(insert_zero(p, b0), b1), b2) | cmask
                var i1 = i0 | tb
                var ar = rp[unsafe_offset=i0]
                var ai = ip[unsafe_offset=i0]
                rp[unsafe_offset=i0] = rp[unsafe_offset=i1]
                ip[unsafe_offset=i0] = ip[unsafe_offset=i1]
                rp[unsafe_offset=i1] = ar
                ip[unsafe_offset=i1] = ai

        run_chunks(body, self.size() >> 3)

    def apply_ccz(mut self, a: Int, b: Int, c: Int):
        var rp = self._rp()
        var ip = self._ip()
        var mask = (1 << a) | (1 << b) | (1 << c)

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm mask}:
            for i in range(lo, hi):
                if (i & mask) == mask:
                    rp[unsafe_offset=i] = -rp[unsafe_offset=i]
                    ip[unsafe_offset=i] = -ip[unsafe_offset=i]

        run_chunks(body, self.size())

    def apply_cswap(mut self, c: Int, a: Int, b: Int):
        var rp = self._rp()
        var ip = self._ip()
        var s = List[Int]()
        s.append(c); s.append(a); s.append(b)
        sort(s)
        var b0 = s[0]; var b1 = s[1]; var b2 = s[2]
        var cb = 1 << c
        var ba = 1 << a
        var bb = 1 << b

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm b0, imm b1, imm b2, imm cb, imm ba, imm bb}:
            for p in range(lo, hi):
                var base = insert_zero(insert_zero(insert_zero(p, b0), b1), b2) | cb
                var i01 = base | ba
                var i10 = base | bb
                var tr = rp[unsafe_offset=i01]
                var ti = ip[unsafe_offset=i01]
                rp[unsafe_offset=i01] = rp[unsafe_offset=i10]
                ip[unsafe_offset=i01] = ip[unsafe_offset=i10]
                rp[unsafe_offset=i10] = tr
                ip[unsafe_offset=i10] = ti

        run_chunks(body, self.size() >> 3)

    def apply_global_phase(mut self, phase: Float64):
        if phase == 0.0:
            return
        self.apply_scale(cos(phase), sin(phase))

    def apply_scale(mut self, cr: Float64, ci: Float64):
        """Multiply every amplitude by cr + i ci (parallel, SIMD)."""
        var rp = self._rp()
        var ip = self._ip()

        def body(lo: Int, hi: Int) {imm rp, imm ip, imm cr, imm ci}:
            kscale_range(rp, ip, cr, ci, lo, hi)

        run_chunks(body, self.size())

    # ------------------------------------------------------------ bit swaps
    def apply_swap_layer(mut self, qa: List[Int], qb: List[Int]) raises:
        """Apply the disjoint SWAP gates (qa[i], qb[i]) together, i.e. exchange
        amplitude-index bits qa[i] <-> qb[i] for every i (cuStateVec
        `swapIndexBits`, Aer chunk swap). Used by the cache-blocked executor
        (`qumo.sim.runner`) to move qubits in and out of cache blocks.

        All 2k bit positions must be distinct and in [0, num_qubits),
        otherwise QM-E407 is raised.

        Algorithm (in place, no scratch memory):
          * Pairs whose low bit is < LOGW (at most 3) are in-vector bits; each
            is one vectorized `apply_swap` pass (lane shuffles / masks).
          * The other pairs are done in ONE pass. Write every index as
            (rest, H, P, low): H = the bits {max(qa_i, qb_i)}, P = the bits
            {min(qa_i, qb_i)}, `low` = the 2^r contiguous amplitudes below the
            lowest swapped bit r (>= LOGW). The permutation exchanges the run
            at (H = h, P = p) with the run at (H = p, P = h); runs with h == p
            are fixed points and are never touched, so a k-pair layer reads
            and writes a fraction 1 - 2^-k of the state exactly once.
          * Runs are swapped directly with vector loads/stores. When a run is
            shorter than a cache line (r = 3) the lowest pair is folded into a
            2x2 tile so both halves of every line are used while resident.
            Long runs are cut into <= 2^SWAP_RUN_BITS-amplitude work items so
            every core gets work even when only a few runs move.
        """
        var k = len(qa)
        var n = self.num_qubits
        if len(qb) != k:
            raise Error("QM-E407: apply_swap_layer got " + String(k) + " bits in qa but " + String(len(qb)) + " in qb. Hint: pass one partner bit per entry: qa[i] <-> qb[i].")
        var used = List[Bool](length=max(n, 1), fill=False)
        for i in range(2 * k):
            var b = qa[i] if i < k else qb[i - k]
            if b < 0 or b >= n:
                raise Error("QM-E407: apply_swap_layer bit " + String(b) + " is outside [0, " + String(n) + "). Hint: bit positions are qubit indices of this " + String(n) + "-qubit state.")
            if used[b]:
                raise Error("QM-E407: apply_swap_layer bit " + String(b) + " appears twice. Hint: the 2k bit positions of a swap layer must be distinct (disjoint pairs, no self-swaps).")
            used[b] = True
        if k == 0:
            return
        # split: pairs with an in-vector bit (one vectorized SWAP pass each;
        # a combined pass measured no faster) / run pairs (one pass)
        var xs = List[Int](capacity=k)
        var ys = List[Int](capacity=k)
        for i in range(k):
            var x = min(qa[i], qb[i])
            var y = max(qa[i], qb[i])
            if x < LOGW:
                self.apply_swap(x, y)
            else:
                xs.append(x)
                ys.append(y)
        if len(xs) > 0:
            self._swap_run_pairs(xs, ys)

    def _swap_run_pairs(mut self, mut xs: List[Int], mut ys: List[Int]):
        """One in-place pass exchanging bits xs[i] <-> ys[i] (all >= LOGW,
        2k distinct positions); see `apply_swap_layer`."""
        var k = len(xs)
        var n = self.num_qubits
        # pairs sorted by their low bit; xs[i] < ys[i]
        for i in range(1, k):
            var j = i
            while j > 0 and xs[j - 1] > xs[j]:
                var tx = xs[j]
                xs[j] = xs[j - 1]
                xs[j - 1] = tx
                var ty = ys[j]
                ys[j] = ys[j - 1]
                ys[j - 1] = ty
                j -= 1
        var r = xs[0]
        # tile = 2^jt x 2^jt runs over the jt lowest pairs (only for runs
        # shorter than a cache line); work item = run of 2^rr amplitudes
        var jt = 0
        if r < LOG_LINE:
            jt = min(k, LOG_LINE - r)
        var rr = min(r, SWAP_RUN_BITS)
        var kh = k - jt
        var nlo = 1 << jt
        var nhi = 1 << kh
        # bit-deposit tables: dep(v) = OR of 1 << bits[i] for set bits i of v
        var dx_lo = List[Int](length=nlo, fill=0)
        var dy_lo = List[Int](length=nlo, fill=0)
        var dx_hi = List[Int](length=nhi, fill=0)
        var dy_hi = List[Int](length=nhi, fill=0)
        for v in range(nlo):
            for b in range(jt):
                if (v >> b) & 1 == 1:
                    dx_lo[v] |= 1 << xs[b]
                    dy_lo[v] |= 1 << ys[b]
        for v in range(nhi):
            for b in range(kh):
                if (v >> b) & 1 == 1:
                    dx_hi[v] |= 1 << xs[jt + b]
                    dy_hi[v] |= 1 << ys[jt + b]
        var special = List[Int](capacity=2 * k)
        for i in range(k):
            special.append(xs[i])
            special.append(ys[i])
        sort(special)
        var nsp = 2 * k
        var nrest = n - 2 * k - rr
        var total = (1 << nrest) * nhi * nhi
        var rp = self._rp()
        var ip = self._ip()
        var pxl = dx_lo.unsafe_ptr().as_unsafe_any_origin()
        var pyl = dy_lo.unsafe_ptr().as_unsafe_any_origin()
        var pxh = dx_hi.unsafe_ptr().as_unsafe_any_origin()
        var pyh = dy_hi.unsafe_ptr().as_unsafe_any_origin()
        var psp = special.unsafe_ptr().as_unsafe_any_origin()
        var run = 1 << rr
        var nchunks = 1
        if self.size() >= PAR_MIN_ITEMS:
            nchunks = min(total, parallelism_level() * 8)
        var chunk = (total + nchunks - 1) // nchunks
        nchunks = (total + chunk - 1) // chunk

        def task(c: Int) {imm rp, imm ip, imm pxl, imm pyl, imm pxh, imm pyh, imm psp,
                           imm nsp, imm kh, imm nhi, imm nlo, imm rr, imm run,
                           imm chunk, imm total}:
            var lo = c * chunk
            var hi = min(lo + chunk, total)
            for o in range(lo, hi):
                var ph = o & (nhi - 1)
                var hh = (o >> kh) & (nhi - 1)
                if hh > ph:
                    continue  # handled by its partner tile (ph, hh)
                if hh == ph and nlo == 1:
                    continue  # fixed point: H == P
                var base = (o >> (2 * kh)) << rr
                for s in range(nsp):
                    base = insert_zero(base, psp[unsafe_offset=s])
                var a_hi = base | pyh[unsafe_offset=hh] | pxh[unsafe_offset=ph]
                if hh == ph:
                    # diagonal tile: transpose in place, skip its diagonal
                    for hl in range(nlo):
                        for pl in range(hl + 1, nlo):
                            _swap_runs(rp, ip, a_hi | pyl[unsafe_offset=hl] | pxl[unsafe_offset=pl],
                                       a_hi | pyl[unsafe_offset=pl] | pxl[unsafe_offset=hl], run)
                else:
                    # A(hl, pl) <-> B(pl, hl)
                    var b_hi = base | pyh[unsafe_offset=ph] | pxh[unsafe_offset=hh]
                    for hl in range(nlo):
                        var row = a_hi | pyl[unsafe_offset=hl]
                        for pl in range(nlo):
                            _swap_runs(rp, ip, row | pxl[unsafe_offset=pl],
                                       b_hi | pyl[unsafe_offset=pl] | pxl[unsafe_offset=hl], run)

        if nchunks == 1:
            task(0)
        else:
            parallelize(task, nchunks)
        _ = len(dx_lo)
        _ = len(dy_lo)
        _ = len(dx_hi)
        _ = len(dy_hi)
        _ = len(special)

    # ------------------------------------------------------------ readout
    def probabilities(self) -> List[Float64]:
        var n = self.size()
        var out = List[Float64](length=n, fill=0.0)
        for i in range(n):
            out[i] = self.re[i] * self.re[i] + self.im[i] * self.im[i]
        return out^

    def norm_sq(self) -> Float64:
        var s = 0.0
        for i in range(self.size()):
            s += self.re[i] * self.re[i] + self.im[i] * self.im[i]
        return s

    def prob_one(self, q: Int) -> Float64:
        var s = 0.0
        var stride = 1 << q
        for p in range(self.size() >> 1):
            var i1 = insert_zero(p, q) | stride
            s += self.re[i1] * self.re[i1] + self.im[i1] * self.im[i1]
        return s

    def measure(mut self, q: Int, mut rng: Rng) -> Int:
        """Projective Z measurement with collapse; returns 0/1."""
        var p1 = self.prob_one(q)
        var outcome = 1 if rng.uniform() < p1 else 0
        var keep_p = p1 if outcome == 1 else 1.0 - p1
        var scale = 1.0 / sqrt(keep_p) if keep_p > 0 else 0.0
        var bit = 1 << q
        for i in range(self.size()):
            if ((i & bit) != 0) == (outcome == 1):
                self.re[i] *= scale
                self.im[i] *= scale
            else:
                self.re[i] = 0.0
                self.im[i] = 0.0
        return outcome

    def reset_qubit(mut self, q: Int, mut rng: Rng):
        if self.measure(q, rng) == 1:
            self.apply_x(q)

    def sample(self, shots: Int, mut rng: Rng) -> List[Int]:
        """Sample basis-state indices: O(2^n + shots log shots), no state copy."""
        var n = self.size()
        var us = List[Float64](capacity=shots)
        for _ in range(shots):
            us.append(rng.uniform())
        sort(us)
        var out = List[Int](capacity=shots)
        var acc = 0.0
        var i = 0
        var total = self.norm_sq()
        for s in range(shots):
            var target = us[s] * total
            while i < n - 1:
                var pi_ = self.re[i] * self.re[i] + self.im[i] * self.im[i]
                if acc + pi_ > target:
                    break
                acc += pi_
                i += 1
            out.append(i)
        # shuffle outcomes so memory order is not sorted (Fisher-Yates)
        var k = shots - 1
        while k > 0:
            var j = rng.below(k + 1)
            var tmp = out[k]
            out[k] = out[j]
            out[j] = tmp
            k -= 1
        return out^

    def expectation(self, h: PauliSum) raises -> Float64:
        """<psi|H|psi> for a Pauli sum, computed in place (no state copies)."""
        if h.num_qubits() > self.num_qubits:
            raise Error("QM-E405: observable acts on qubit " + String(h.num_qubits() - 1) + " but the state has " + String(self.num_qubits) + " qubits. Hint: check Pauli indices (0-based, qubit 0 = least significant).")
        var total = 0.0
        for t in h.terms:
            total += t.coeff * self.expectation_term(t.x, t.z)
        return total

    def expectation_term(self, x: UInt64, z: UInt64) -> Float64:
        var rp = self.re.unsafe_ptr()
        var ip = self.im.unsafe_ptr()
        var n = self.size()
        var xi_ = Int(x)
        var zi = Int(z)
        var ny = Int(pop_count(x & z)) & 3
        var nchunks = 1 if n < PAR_MIN_ITEMS else parallelism_level() * 2
        var chunk = (n + nchunks - 1) // nchunks
        var partial_r = List[Float64](length=nchunks, fill=0.0)
        var partial_i = List[Float64](length=nchunks, fill=0.0)
        var prp = partial_r.unsafe_ptr().as_unsafe_any_origin()
        var pip = partial_i.unsafe_ptr().as_unsafe_any_origin()

        def task(c: Int) {imm rp, imm ip, imm xi_, imm zi, imm chunk, imm n, imm prp, imm pip}:
            var lo = c * chunk
            var hi = min(lo + chunk, n)
            var sr = 0.0
            var si = 0.0
            for i in range(lo, hi):
                var j = i ^ xi_
                # conj(psi[j]) * psi[i] * (-1)^{pc(i & z)}
                var pr = rp[unsafe_offset=j] * rp[unsafe_offset=i] + ip[unsafe_offset=j] * ip[unsafe_offset=i]
                var pim = rp[unsafe_offset=j] * ip[unsafe_offset=i] - ip[unsafe_offset=j] * rp[unsafe_offset=i]
                if (pop_count(i & zi) & 1) == 1:
                    sr -= pr
                    si -= pim
                else:
                    sr += pr
                    si += pim
            prp[unsafe_offset=c] = sr
            pip[unsafe_offset=c] = si

        if nchunks == 1:
            task(0)
        else:
            parallelize(task, nchunks)
        var sr = 0.0
        var si = 0.0
        for c in range(nchunks):
            sr += partial_r[c]
            si += partial_i[c]
        # multiply by i^ny, keep real part
        if ny == 0: return sr
        if ny == 1: return -si
        if ny == 2: return -sr
        return si

    def inner(self, o: StateVector) -> Tuple[Float64, Float64]:
        """<self|o>."""
        var sr = 0.0
        var si = 0.0
        for i in range(self.size()):
            sr += self.re[i] * o.re[i] + self.im[i] * o.im[i]
            si += self.re[i] * o.im[i] - self.im[i] * o.re[i]
        return (sr, si)

    def fidelity(self, o: StateVector) -> Float64:
        var ip_ = self.inner(o)
        return ip_[0] * ip_[0] + ip_[1] * ip_[1]

    def write_to(self, mut w: Some[Writer]):
        w.write("StateVector(", self.num_qubits, " qubits)")
        var shown = 0
        for i in range(self.size()):
            var p = self.re[i] * self.re[i] + self.im[i] * self.im[i]
            if p > 1e-12 and shown < 16:
                w.write("\n  |", i, ">: ", self.re[i], " + ", self.im[i], "j")
                shown += 1


# ---------------------------------------------------------------------------
# Range kernels: operate on pair/quad indices [lo, hi) of a state of any size
# whose amplitudes start at rp/ip. Used by the parallel drivers above and by
# the cache-blocked executor (`qumo.sim.runner`).
#
# Vector paths: a qubit >= LOGW gives W contiguous partners (plain vector
# loads); a qubit < LOGW keeps the partner inside the vector (lane shuffles /
# masks). The low-qubit paths need lo, hi to be multiples of 4 (always true
# for run_chunks chunks and cache blocks); otherwise the scalar loop runs.
# ---------------------------------------------------------------------------
@always_inline
def k1_range(rp: FPtr, ip: FPtr, t: Int, m: M2, lo: Int, hi: Int):
    var stride = 1 << t
    var m00r = m.re[0]
    var m00i = m.im[0]
    var m01r = m.re[1]
    var m01i = m.im[1]
    var m10r = m.re[2]
    var m10i = m.im[2]
    var m11r = m.re[3]
    var m11i = m.im[3]
    if stride >= W:
        var p = lo
        while p < hi:
            var i0 = insert_zero(p, t)
            var i1 = i0 + stride
            var ar = rp.unsafe_load[width=W](i0)
            var ai = ip.unsafe_load[width=W](i0)
            var br = rp.unsafe_load[width=W](i1)
            var bi = ip.unsafe_load[width=W](i1)
            rp.unsafe_store(i0, m00r * ar - m00i * ai + m01r * br - m01i * bi)
            ip.unsafe_store(i0, m00r * ai + m00i * ar + m01r * bi + m01i * br)
            rp.unsafe_store(i1, m10r * ar - m10i * ai + m11r * br - m11i * bi)
            ip.unsafe_store(i1, m10r * ai + m10i * ar + m11r * bi + m11i * br)
            p += W
    elif (lo & 3) == 0 and (hi & 3) == 0:
        # pairs [4g, 4g+4) live in the W-group g
        var ar = _lane_pick(t, m00r, m11r)
        var ai = _lane_pick(t, m00i, m11i)
        var br = _lane_pick(t, m01r, m10r)
        var bi = _lane_pick(t, m01i, m10i)
        _mat_loop_rt(t, rp, ip, ar, ai, br, bi, lo >> 2, hi >> 2, -1, 0)
    else:
        for p in range(lo, hi):
            var i0 = insert_zero(p, t)
            var i1 = i0 + stride
            var ar = rp[unsafe_offset=i0]
            var ai = ip[unsafe_offset=i0]
            var br = rp[unsafe_offset=i1]
            var bi = ip[unsafe_offset=i1]
            rp[unsafe_offset=i0] = m00r * ar - m00i * ai + m01r * br - m01i * bi
            ip[unsafe_offset=i0] = m00r * ai + m00i * ar + m01r * bi + m01i * br
            rp[unsafe_offset=i1] = m10r * ar - m10i * ai + m11r * br - m11i * bi
            ip[unsafe_offset=i1] = m10r * ai + m10i * ar + m11r * bi + m11i * br


@always_inline
def kdiag_range(rp: FPtr, ip: FPtr, t: Int, d0r: Float64, d0i: Float64,
                d1r: Float64, d1i: Float64, lo: Int, hi: Int):
    var stride = 1 << t
    var skip0 = d0r == 1.0 and d0i == 0.0
    if stride >= W:
        var p = lo
        while p < hi:
            var i0 = insert_zero(p, t)
            var i1 = i0 + stride
            if not skip0:
                var ar = rp.unsafe_load[width=W](i0)
                var ai = ip.unsafe_load[width=W](i0)
                rp.unsafe_store(i0, d0r * ar - d0i * ai)
                ip.unsafe_store(i0, d0r * ai + d0i * ar)
            var br = rp.unsafe_load[width=W](i1)
            var bi = ip.unsafe_load[width=W](i1)
            rp.unsafe_store(i1, d1r * br - d1i * bi)
            ip.unsafe_store(i1, d1r * bi + d1i * br)
            p += W
    elif (lo & 3) == 0 and (hi & 3) == 0:
        _diag_loop(rp, ip, _lane_pick(t, d0r, d1r), _lane_pick(t, d0i, d1i), lo >> 2, hi >> 2, -1, 0)
    else:
        for p in range(lo, hi):
            var i0 = insert_zero(p, t)
            var i1 = i0 + stride
            if not skip0:
                var ar = rp[unsafe_offset=i0]
                var ai = ip[unsafe_offset=i0]
                rp[unsafe_offset=i0] = d0r * ar - d0i * ai
                ip[unsafe_offset=i0] = d0r * ai + d0i * ar
            var br = rp[unsafe_offset=i1]
            var bi = ip[unsafe_offset=i1]
            rp[unsafe_offset=i1] = d1r * br - d1i * bi
            ip[unsafe_offset=i1] = d1r * bi + d1i * br


@always_inline
def kx_range(rp: FPtr, ip: FPtr, t: Int, lo: Int, hi: Int):
    var stride = 1 << t
    if stride >= W:
        var p = lo
        while p < hi:
            var i0 = insert_zero(p, t)
            var i1 = i0 + stride
            var ar = rp.unsafe_load[width=W](i0)
            var ai = ip.unsafe_load[width=W](i0)
            rp.unsafe_store(i0, rp.unsafe_load[width=W](i1))
            ip.unsafe_store(i0, ip.unsafe_load[width=W](i1))
            rp.unsafe_store(i1, ar)
            ip.unsafe_store(i1, ai)
            p += W
    elif (lo & 3) == 0 and (hi & 3) == 0:
        _perm_loop_rt(t, rp, ip, VB(fill=True), lo >> 2, hi >> 2, -1, 0)
    else:
        for p in range(lo, hi):
            var i0 = insert_zero(p, t)
            var i1 = i0 + stride
            var ar = rp[unsafe_offset=i0]
            var ai = ip[unsafe_offset=i0]
            rp[unsafe_offset=i0] = rp[unsafe_offset=i1]
            ip[unsafe_offset=i0] = ip[unsafe_offset=i1]
            rp[unsafe_offset=i1] = ar
            ip[unsafe_offset=i1] = ai


@always_inline
def kscale_range(rp: FPtr, ip: FPtr, cr: Float64, ci: Float64, lo: Int, hi: Int):
    """Multiply amplitudes [lo, hi) by the complex scalar cr + i ci."""
    if (lo % W) == 0 and (hi % W) == 0:
        _diag_loop(rp, ip, VF(cr), VF(ci), lo >> LOGW, hi >> LOGW, -1, 0)
    else:
        for i in range(lo, hi):
            var xr = rp[unsafe_offset=i]
            var xi = ip[unsafe_offset=i]
            rp[unsafe_offset=i] = cr * xr - ci * xi
            ip[unsafe_offset=i] = cr * xi + ci * xr


@always_inline
def kc1_range(rp: FPtr, ip: FPtr, c: Int, t: Int, m: M2, lo: Int, hi: Int):
    var lo_b = min(c, t)
    var hi_b = max(c, t)
    var cbit = 1 << c
    var stride = 1 << t
    var diag = m.is_diagonal()
    var m00r = m.re[0]
    var m00i = m.im[0]
    var m01r = m.re[1]
    var m01i = m.im[1]
    var m10r = m.re[2]
    var m10i = m.im[2]
    var m11r = m.re[3]
    var m11i = m.im[3]
    if lo_b >= LOGW:
        var p = lo
        while p < hi:
            var i0 = insert_zero2(p, lo_b, hi_b) | cbit
            var i1 = i0 | stride
            var ar = rp.unsafe_load[width=W](i0)
            var ai = ip.unsafe_load[width=W](i0)
            var br = rp.unsafe_load[width=W](i1)
            var bi = ip.unsafe_load[width=W](i1)
            if diag:
                rp.unsafe_store(i0, m00r * ar - m00i * ai)
                ip.unsafe_store(i0, m00r * ai + m00i * ar)
                rp.unsafe_store(i1, m11r * br - m11i * bi)
                ip.unsafe_store(i1, m11r * bi + m11i * br)
            else:
                rp.unsafe_store(i0, m00r * ar - m00i * ai + m01r * br - m01i * bi)
                ip.unsafe_store(i0, m00r * ai + m00i * ar + m01r * bi + m01i * br)
                rp.unsafe_store(i1, m10r * ar - m10i * ai + m11r * br - m11i * bi)
                ip.unsafe_store(i1, m10r * ai + m10i * ar + m11r * bi + m11i * br)
            p += W
    elif (lo & 3) == 0 and (hi & 3) == 0:
        if hi_b < LOGW:
            # both qubits inside the vector: lanes with c clear get identity
            var ar = VF(1.0)
            var ai = VF(0.0)
            var br = VF(0.0)
            var bi = VF(0.0)
            comptime for j in range(W):
                if (j >> c) & 1 == 1:
                    if (j >> t) & 1 == 1:
                        ar[j] = m11r
                        ai[j] = m11i
                        br[j] = m10r
                        bi[j] = m10i
                    else:
                        ar[j] = m00r
                        ai[j] = m00i
                        br[j] = m01r
                        bi[j] = m01i
            _mat_loop_rt(t, rp, ip, ar, ai, br, bi, lo >> 1, hi >> 1, -1, 0)
        elif c < LOGW:
            # control in-vector, target across vectors
            var mc = _lane_mask(c)
            for g in range(lo >> 2, hi >> 2):
                var i0 = insert_zero(g, t - LOGW) << LOGW
                var i1 = i0 + stride
                var ar = rp.unsafe_load[width=W](i0)
                var ai = ip.unsafe_load[width=W](i0)
                var br = rp.unsafe_load[width=W](i1)
                var bi = ip.unsafe_load[width=W](i1)
                var n0r = m00r * ar - m00i * ai + m01r * br - m01i * bi
                var n0i = m00r * ai + m00i * ar + m01r * bi + m01i * br
                var n1r = m10r * ar - m10i * ai + m11r * br - m11i * bi
                var n1i = m10r * ai + m10i * ar + m11r * bi + m11i * br
                rp.unsafe_store(i0, mc.select(n0r, ar))
                ip.unsafe_store(i0, mc.select(n0i, ai))
                rp.unsafe_store(i1, mc.select(n1r, br))
                ip.unsafe_store(i1, mc.select(n1i, bi))
        else:
            # target in-vector, control across vectors
            var ar = _lane_pick(t, m00r, m11r)
            var ai = _lane_pick(t, m00i, m11i)
            var br = _lane_pick(t, m01r, m10r)
            var bi = _lane_pick(t, m01i, m10i)
            _mat_loop_rt(t, rp, ip, ar, ai, br, bi, lo >> 2, hi >> 2, c, cbit)
    else:
        for p in range(lo, hi):
            var i0 = insert_zero2(p, lo_b, hi_b) | cbit
            var i1 = i0 | stride
            var ar = rp[unsafe_offset=i0]
            var ai = ip[unsafe_offset=i0]
            var br = rp[unsafe_offset=i1]
            var bi = ip[unsafe_offset=i1]
            rp[unsafe_offset=i0] = m00r * ar - m00i * ai + m01r * br - m01i * bi
            ip[unsafe_offset=i0] = m00r * ai + m00i * ar + m01r * bi + m01i * br
            rp[unsafe_offset=i1] = m10r * ar - m10i * ai + m11r * br - m11i * bi
            ip[unsafe_offset=i1] = m10r * ai + m10i * ar + m11r * bi + m11i * br


@always_inline
def kcx_range(rp: FPtr, ip: FPtr, c: Int, t: Int, lo: Int, hi: Int):
    var lo_b = min(c, t)
    var hi_b = max(c, t)
    var cbit = 1 << c
    var stride = 1 << t
    if lo_b >= LOGW:
        var p = lo
        while p < hi:
            var i0 = insert_zero2(p, lo_b, hi_b) | cbit
            var i1 = i0 | stride
            var ar = rp.unsafe_load[width=W](i0)
            var ai = ip.unsafe_load[width=W](i0)
            rp.unsafe_store(i0, rp.unsafe_load[width=W](i1))
            ip.unsafe_store(i0, ip.unsafe_load[width=W](i1))
            rp.unsafe_store(i1, ar)
            ip.unsafe_store(i1, ai)
            p += W
    elif (lo & 3) == 0 and (hi & 3) == 0:
        if hi_b < LOGW:
            _perm_loop_rt(t, rp, ip, _lane_mask(c), lo >> 1, hi >> 1, -1, 0)
        elif c < LOGW:
            var mc = _lane_mask(c)
            for g in range(lo >> 2, hi >> 2):
                var i0 = insert_zero(g, t - LOGW) << LOGW
                var i1 = i0 + stride
                var ar = rp.unsafe_load[width=W](i0)
                var ai = ip.unsafe_load[width=W](i0)
                var br = rp.unsafe_load[width=W](i1)
                var bi = ip.unsafe_load[width=W](i1)
                rp.unsafe_store(i0, mc.select(br, ar))
                ip.unsafe_store(i0, mc.select(bi, ai))
                rp.unsafe_store(i1, mc.select(ar, br))
                ip.unsafe_store(i1, mc.select(ai, bi))
        else:
            _perm_loop_rt(t, rp, ip, VB(fill=True), lo >> 2, hi >> 2, c, cbit)
    else:
        for p in range(lo, hi):
            var i0 = insert_zero2(p, lo_b, hi_b) | cbit
            var i1 = i0 | stride
            var ar = rp[unsafe_offset=i0]
            var ai = ip[unsafe_offset=i0]
            rp[unsafe_offset=i0] = rp[unsafe_offset=i1]
            ip[unsafe_offset=i0] = ip[unsafe_offset=i1]
            rp[unsafe_offset=i1] = ar
            ip[unsafe_offset=i1] = ai


@always_inline
def kcphase_range(rp: FPtr, ip: FPtr, a: Int, b: Int, pr: Float64, pi_: Float64, lo: Int, hi: Int):
    var lo_b = min(a, b)
    var hi_b = max(a, b)
    var mask = (1 << a) | (1 << b)
    if lo_b >= LOGW:
        var p = lo
        while p < hi:
            var i = insert_zero2(p, lo_b, hi_b) | mask
            var xr = rp.unsafe_load[width=W](i)
            var xi = ip.unsafe_load[width=W](i)
            rp.unsafe_store(i, pr * xr - pi_ * xi)
            ip.unsafe_store(i, pr * xi + pi_ * xr)
            p += W
    elif (lo & 3) == 0 and (hi & 3) == 0:
        var dr = VF(1.0)
        var di = VF(0.0)
        if hi_b < LOGW:
            comptime for j in range(W):
                if (j & mask) == mask:
                    dr[j] = pr
                    di[j] = pi_
            _diag_loop(rp, ip, dr, di, lo >> 1, hi >> 1, -1, 0)
        else:
            comptime for j in range(W):
                if (j >> lo_b) & 1 == 1:
                    dr[j] = pr
                    di[j] = pi_
            _diag_loop(rp, ip, dr, di, lo >> 2, hi >> 2, hi_b, 1 << hi_b)
    else:
        for p in range(lo, hi):
            var i = insert_zero2(p, lo_b, hi_b) | mask
            var xr = rp[unsafe_offset=i]
            var xi = ip[unsafe_offset=i]
            rp[unsafe_offset=i] = pr * xr - pi_ * xi
            ip[unsafe_offset=i] = pr * xi + pi_ * xr


@always_inline
def krzz_range(rp: FPtr, ip: FPtr, a: Int, b: Int, cr: Float64, si: Float64, lo: Int, hi: Int):
    """Element range [lo, hi): phase cr - i si (even parity) / cr + i si (odd)."""
    var ma = 1 << a
    var mb = 1 << b
    if (lo % W) == 0 and (hi % W) == 0:
        # parity = (lane part) xor (group part)
        var gmask = (ma if a >= LOGW else 0) | (mb if b >= LOGW else 0)
        var lmask = (ma if a < LOGW else 0) | (mb if b < LOGW else 0)
        var s_even = VF(-si)
        comptime for j in range(W):
            if (pop_count(j & lmask) & 1) == 1:
                s_even[j] = si
        var crv = VF(cr)
        for g in range(lo >> LOGW, hi >> LOGW):
            var i = g << LOGW
            var s = s_even if (pop_count(i & gmask) & 1) == 0 else -s_even
            var xr = rp.unsafe_load[width=W](i)
            var xi = ip.unsafe_load[width=W](i)
            rp.unsafe_store(i, crv * xr - s * xi)
            ip.unsafe_store(i, crv * xi + s * xr)
    else:
        for i in range(lo, hi):
            var odd = ((i & ma) != 0) != ((i & mb) != 0)
            var s = si if odd else -si
            var xr = rp[unsafe_offset=i]
            var xi = ip[unsafe_offset=i]
            rp[unsafe_offset=i] = cr * xr - s * xi
            ip[unsafe_offset=i] = cr * xi + s * xr


@always_inline
def kswap_range(rp: FPtr, ip: FPtr, a: Int, b: Int, lo: Int, hi: Int):
    var lo_b = min(a, b)
    var hi_b = max(a, b)
    var ba = 1 << a
    var bb = 1 << b
    if lo_b >= LOGW:
        var p = lo
        while p < hi:
            var base = insert_zero2(p, lo_b, hi_b)
            var i01 = base | ba
            var i10 = base | bb
            var tr = rp.unsafe_load[width=W](i01)
            var ti = ip.unsafe_load[width=W](i01)
            rp.unsafe_store(i01, rp.unsafe_load[width=W](i10))
            ip.unsafe_store(i01, ip.unsafe_load[width=W](i10))
            rp.unsafe_store(i10, tr)
            ip.unsafe_store(i10, ti)
            p += W
    elif (lo & 3) == 0 and (hi & 3) == 0:
        if hi_b < LOGW:
            # both bits inside the vector: 8 amplitudes = 2 quads per group
            if lo_b == 0 and hi_b == 1:
                _swap_in_lane_loop[0, 1](rp, ip, lo >> 1, hi >> 1)
            elif lo_b == 0:
                _swap_in_lane_loop[0, 2](rp, ip, lo >> 1, hi >> 1)
            else:
                _swap_in_lane_loop[1, 2](rp, ip, lo >> 1, hi >> 1)
        elif lo_b == 0:
            # in-vector bit x cross-vector bit: 16 amplitudes = 4 quads per group
            _swap_cross_loop[0](rp, ip, hi_b, lo >> 2, hi >> 2)
        elif lo_b == 1:
            _swap_cross_loop[1](rp, ip, hi_b, lo >> 2, hi >> 2)
        else:
            _swap_cross_loop[2](rp, ip, hi_b, lo >> 2, hi >> 2)
    else:
        for p in range(lo, hi):
            var base = insert_zero2(p, lo_b, hi_b)
            var i01 = base | ba
            var i10 = base | bb
            var tr = rp[unsafe_offset=i01]
            var ti = ip[unsafe_offset=i01]
            rp[unsafe_offset=i01] = rp[unsafe_offset=i10]
            ip[unsafe_offset=i01] = ip[unsafe_offset=i10]
            rp[unsafe_offset=i10] = tr
            ip[unsafe_offset=i10] = ti


@always_inline
def k2_range(rp: FPtr, ip: FPtr, q0: Int, q1: Int, m: M4, lo: Int, hi: Int):
    var lo_b = min(q0, q1)
    var hi_b = max(q0, q1)
    var b0 = 1 << q0
    var b1 = 1 << q1
    var mr = m.re
    var mi = m.im
    if lo_b >= LOGW:
        var p = lo
        while p < hi:
            var base = insert_zero2(p, lo_b, hi_b)
            var i1 = base | b0
            var i2 = base | b1
            var i3 = base | b0 | b1
            var a0r = rp.unsafe_load[width=W](base)
            var a0i = ip.unsafe_load[width=W](base)
            var a1r = rp.unsafe_load[width=W](i1)
            var a1i = ip.unsafe_load[width=W](i1)
            var a2r = rp.unsafe_load[width=W](i2)
            var a2i = ip.unsafe_load[width=W](i2)
            var a3r = rp.unsafe_load[width=W](i3)
            var a3i = ip.unsafe_load[width=W](i3)
            comptime for r in range(4):
                var sr = (mr[r * 4] * a0r - mi[r * 4] * a0i + mr[r * 4 + 1] * a1r - mi[r * 4 + 1] * a1i
                          + mr[r * 4 + 2] * a2r - mi[r * 4 + 2] * a2i + mr[r * 4 + 3] * a3r - mi[r * 4 + 3] * a3i)
                var si = (mr[r * 4] * a0i + mi[r * 4] * a0r + mr[r * 4 + 1] * a1i + mi[r * 4 + 1] * a1r
                          + mr[r * 4 + 2] * a2i + mi[r * 4 + 2] * a2r + mr[r * 4 + 3] * a3i + mi[r * 4 + 3] * a3r)
                comptime if r == 0:
                    rp.unsafe_store(base, sr)
                    ip.unsafe_store(base, si)
                elif r == 1:
                    rp.unsafe_store(i1, sr)
                    ip.unsafe_store(i1, si)
                elif r == 2:
                    rp.unsafe_store(i2, sr)
                    ip.unsafe_store(i2, si)
                else:
                    rp.unsafe_store(i3, sr)
                    ip.unsafe_store(i3, si)
            p += W
        return
    for p in range(lo, hi):
        var base = insert_zero2(p, lo_b, hi_b)
        var idx: Array[Int, 4] = [base, base | b0, base | b1, base | b0 | b1]
        var ar = SIMD[DType.float64, 4](0)
        var ai = SIMD[DType.float64, 4](0)
        comptime for k in range(4):
            ar[k] = rp[unsafe_offset=idx[k]]
            ai[k] = ip[unsafe_offset=idx[k]]
        comptime for r in range(4):
            var sr = 0.0
            var si = 0.0
            comptime for c in range(4):
                sr += mr[r * 4 + c] * ar[c] - mi[r * 4 + c] * ai[c]
                si += mr[r * 4 + c] * ai[c] + mi[r * 4 + c] * ar[c]
            rp[unsafe_offset=idx[r]] = sr
            ip[unsafe_offset=idx[r]] = si
