"""GPU state-vector kernels: device-agnostic thread bodies + thin MAX wrappers.

Design (ideas from cuStateVec, Qiskit-Aer-GPU and qsim, re-derived for MAX):
  * Same storage as the CPU simulator (`qumo.sim.statevector`): split re/im
    arrays (structure of arrays), little-endian qubit order. One GPU thread
    owns one amplitude *pair* (1q gates), *quad* (2q gates) or *octet* (3q
    gates), enumerated with the CPU's `insert_zero` pair-index addressing, so
    threads touch disjoint amplitudes: no atomics, no races, and loads are
    coalesced whenever the lowest target qubit is >= log2(warp size).
  * Every kernel is an `@always_inline` *thread body* `th_*(gid, work, ...)`
    over plain pointers and integers. The GPU kernel `gk_*` is a one-line
    wrapper passing `global_idx.x`; the host emulator (`qumo.gpu.emulate`)
    calls the very same body for every thread index of the same launch grid.
    The GPU indexing and arithmetic are therefore unit-tested on any machine,
    with or without a GPU (MAX's `DeviceContext(api="cpu")` does not execute
    `enqueue_function` kernels, so this is the only way to test them on CPU).
  * Precision is the comptime DType `dt`: float32 on Apple GPUs (Metal has no
    fp64), float64 selectable on NVIDIA/AMD. CUDA-Q's `nvidia` target makes
    the same choice (fp32 default, fp64 option).
  * Reductions (Pauli-term expectation values, norms, sampling CDF segments)
    use the classic shared-memory tree (CUDA "reduction #3"): per-thread
    grid-stride partial -> TPB-wide tree in shared memory with barriers -> one
    partial per block -> fp64 sum on the host. Fixed order and no atomics, so
    results are bit-reproducible run to run on the same device.
  * Kernel parameters are DevicePassable only: `Pointer[Scalar[dt]]`,
    Int32 / Int64 / Scalar[dt] / SIMD[dt, k] (`Int` is not DevicePassable).
    Gate matrices travel by value as SIMD parameters (no per-gate uploads).

The `gk_*` wrappers are only instantiated by launches inside
`comptime if is_defined["QUMO_GPU"]()` blocks (see statevector_gpu.mojo), so
default builds never compile device code.
"""

from std.bit import pop_count
from std.memory import unsafe_stack_allocation, AddressSpace
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.sync import barrier

comptime TPB = 256  # threads per block for every kernel
comptime LOG_TPB = 8
comptime RED_ITEMS = 16  # target amplitudes per thread in reductions
comptime MAX_RED_BLOCKS = 1024  # cap on partials per reduction (host sums them)
comptime SAMPLE_SEG = 4096  # amplitudes per sampling segment (one block each)

comptime GPtr[dt: DType] = Pointer[Scalar[dt], MutAnyOrigin]
comptime I64Ptr = Pointer[Int64, MutAnyOrigin]


@always_inline
def ins0(p: Int, b: Int) -> Int:
    """Insert a 0 bit at position b of p (pair-index addressing)."""
    return ((p >> b) << (b + 1)) | (p & ((1 << b) - 1))


@always_inline
def ins0_2(p: Int, lo: Int, hi: Int) -> Int:
    """Insert 0 bits at positions lo < hi."""
    return ins0(ins0(p, lo), hi)


@always_inline
def ins0_3(p: Int, b0: Int, b1: Int, b2: Int) -> Int:
    """Insert 0 bits at positions b0 < b1 < b2."""
    return ins0(ins0(ins0(p, b0), b1), b2)


# ---------------------------------------------------------------------------
# Thread bodies. `gid` = global thread index, `work` = number of work items
# (threads with gid >= work do nothing, exactly like the GPU grid tail).
# ---------------------------------------------------------------------------
@always_inline
def th_init[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt], index: Int):
    """|index>: one thread per amplitude."""
    if gid >= work:
        return
    re[unsafe_offset=gid] = Scalar[dt](1) if gid == index else Scalar[dt](0)
    im[unsafe_offset=gid] = 0


@always_inline
def th_1q[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt], t: Int,
                     mr: SIMD[dt, 4], mi: SIMD[dt, 4]):
    """Dense 2x2 [m00, m01, m10, m11] on qubit t; one thread per pair."""
    if gid >= work:
        return
    var i0 = ins0(gid, t)
    var i1 = i0 | (1 << t)
    var ar = re[unsafe_offset=i0]
    var ai = im[unsafe_offset=i0]
    var br = re[unsafe_offset=i1]
    var bi = im[unsafe_offset=i1]
    re[unsafe_offset=i0] = mr[0] * ar - mi[0] * ai + mr[1] * br - mi[1] * bi
    im[unsafe_offset=i0] = mr[0] * ai + mi[0] * ar + mr[1] * bi + mi[1] * br
    re[unsafe_offset=i1] = mr[2] * ar - mi[2] * ai + mr[3] * br - mi[3] * bi
    im[unsafe_offset=i1] = mr[2] * ai + mi[2] * ar + mr[3] * bi + mi[3] * br


@always_inline
def th_diag[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt], t: Int,
                       d: SIMD[dt, 4], skip0: Bool):
    """diag(d0, d1) on qubit t, d = [d0r, d0i, d1r, d1i]; skip0 => d0 == 1."""
    if gid >= work:
        return
    var i0 = ins0(gid, t)
    var i1 = i0 | (1 << t)
    if not skip0:
        var ar = re[unsafe_offset=i0]
        var ai = im[unsafe_offset=i0]
        re[unsafe_offset=i0] = d[0] * ar - d[1] * ai
        im[unsafe_offset=i0] = d[0] * ai + d[1] * ar
    var br = re[unsafe_offset=i1]
    var bi = im[unsafe_offset=i1]
    re[unsafe_offset=i1] = d[2] * br - d[3] * bi
    im[unsafe_offset=i1] = d[2] * bi + d[3] * br


@always_inline
def th_x[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt], t: Int):
    """Pauli X as a pure permutation (zero flops)."""
    if gid >= work:
        return
    var i0 = ins0(gid, t)
    var i1 = i0 | (1 << t)
    var ar = re[unsafe_offset=i0]
    var ai = im[unsafe_offset=i0]
    re[unsafe_offset=i0] = re[unsafe_offset=i1]
    im[unsafe_offset=i0] = im[unsafe_offset=i1]
    re[unsafe_offset=i1] = ar
    im[unsafe_offset=i1] = ai


@always_inline
def th_cx[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt], c: Int, t: Int):
    """CNOT: swap the target pair inside the control=1 quarter."""
    if gid >= work:
        return
    var i0 = ins0_2(gid, min(c, t), max(c, t)) | (1 << c)
    var i1 = i0 | (1 << t)
    var ar = re[unsafe_offset=i0]
    var ai = im[unsafe_offset=i0]
    re[unsafe_offset=i0] = re[unsafe_offset=i1]
    im[unsafe_offset=i0] = im[unsafe_offset=i1]
    re[unsafe_offset=i1] = ar
    im[unsafe_offset=i1] = ai


@always_inline
def th_cphase[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt], a: Int, b: Int,
                         pr: Scalar[dt], pim: Scalar[dt]):
    """Multiply |..1_a..1_b..> by (pr + i pim) (CZ: pr = -1, CP: e^{i l})."""
    if gid >= work:
        return
    var i = ins0_2(gid, min(a, b), max(a, b)) | (1 << a) | (1 << b)
    var xr = re[unsafe_offset=i]
    var xi = im[unsafe_offset=i]
    re[unsafe_offset=i] = pr * xr - pim * xi
    im[unsafe_offset=i] = pr * xi + pim * xr


@always_inline
def th_c1[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt], c: Int, t: Int,
                     mr: SIMD[dt, 4], mi: SIMD[dt, 4]):
    """|1><1|_c (x) m_t + |0><0|_c (x) I: 2x2 on the control=1 quarter."""
    if gid >= work:
        return
    var i0 = ins0_2(gid, min(c, t), max(c, t)) | (1 << c)
    var i1 = i0 | (1 << t)
    var ar = re[unsafe_offset=i0]
    var ai = im[unsafe_offset=i0]
    var br = re[unsafe_offset=i1]
    var bi = im[unsafe_offset=i1]
    re[unsafe_offset=i0] = mr[0] * ar - mi[0] * ai + mr[1] * br - mi[1] * bi
    im[unsafe_offset=i0] = mr[0] * ai + mi[0] * ar + mr[1] * bi + mi[1] * br
    re[unsafe_offset=i1] = mr[2] * ar - mi[2] * ai + mr[3] * br - mi[3] * bi
    im[unsafe_offset=i1] = mr[2] * ai + mi[2] * ar + mr[3] * bi + mi[3] * br


@always_inline
def th_rzz[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt], a: Int, b: Int,
                      cr: Scalar[dt], si: Scalar[dt]):
    """exp(-i t/2 Z_a Z_b), cr = cos(t/2), si = sin(t/2); one thread per amplitude."""
    if gid >= work:
        return
    var odd = ((gid >> a) ^ (gid >> b)) & 1
    var s = si if odd == 1 else -si
    var xr = re[unsafe_offset=gid]
    var xi = im[unsafe_offset=gid]
    re[unsafe_offset=gid] = cr * xr - s * xi
    im[unsafe_offset=gid] = cr * xi + s * xr


@always_inline
def th_swap[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt], a: Int, b: Int):
    """SWAP: exchange |01> and |10> of every quad."""
    if gid >= work:
        return
    var base = ins0_2(gid, min(a, b), max(a, b))
    var i01 = base | (1 << a)
    var i10 = base | (1 << b)
    var tr = re[unsafe_offset=i01]
    var ti = im[unsafe_offset=i01]
    re[unsafe_offset=i01] = re[unsafe_offset=i10]
    im[unsafe_offset=i01] = im[unsafe_offset=i10]
    re[unsafe_offset=i10] = tr
    im[unsafe_offset=i10] = ti


@always_inline
def th_2q[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt], q0: Int, q1: Int,
                     mr: SIMD[dt, 16], mi: SIMD[dt, 16]):
    """Dense 4x4 (row-major, index = b(q0) + 2 b(q1)); one thread per quad."""
    if gid >= work:
        return
    var base = ins0_2(gid, min(q0, q1), max(q0, q1))
    var b0 = 1 << q0
    var b1 = 1 << q1
    var idx = SIMD[DType.int64, 4](Int64(base), Int64(base | b0), Int64(base | b1), Int64(base | b0 | b1))
    var ar = SIMD[dt, 4](0)
    var ai = SIMD[dt, 4](0)
    comptime for k in range(4):
        ar[k] = re[unsafe_offset=Int(idx[k])]
        ai[k] = im[unsafe_offset=Int(idx[k])]
    comptime for r in range(4):
        var sr = Scalar[dt](0)
        var si = Scalar[dt](0)
        comptime for c in range(4):
            sr += mr[r * 4 + c] * ar[c] - mi[r * 4 + c] * ai[c]
            si += mr[r * 4 + c] * ai[c] + mi[r * 4 + c] * ar[c]
        re[unsafe_offset=Int(idx[r])] = sr
        im[unsafe_offset=Int(idx[r])] = si


@always_inline
def th_ccx[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt],
                      b0: Int, b1: Int, b2: Int, cmask: Int, tbit: Int):
    """Toffoli; (b0, b1, b2) = sorted qubits, cmask = control bits, tbit = target bit."""
    if gid >= work:
        return
    var i0 = ins0_3(gid, b0, b1, b2) | cmask
    var i1 = i0 | tbit
    var ar = re[unsafe_offset=i0]
    var ai = im[unsafe_offset=i0]
    re[unsafe_offset=i0] = re[unsafe_offset=i1]
    im[unsafe_offset=i0] = im[unsafe_offset=i1]
    re[unsafe_offset=i1] = ar
    im[unsafe_offset=i1] = ai


@always_inline
def th_ccz[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt],
                      b0: Int, b1: Int, b2: Int):
    """CCZ: negate |111> on the sorted qubits (b0, b1, b2); 1/8 of the state."""
    if gid >= work:
        return
    var i = ins0_3(gid, b0, b1, b2) | (1 << b0) | (1 << b1) | (1 << b2)
    re[unsafe_offset=i] = -re[unsafe_offset=i]
    im[unsafe_offset=i] = -im[unsafe_offset=i]


@always_inline
def th_cswap[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt],
                        b0: Int, b1: Int, b2: Int, cbit: Int, abit: Int, bbit: Int):
    """Fredkin: swap a/b when c = 1; (b0, b1, b2) = sorted qubits."""
    if gid >= work:
        return
    var base = ins0_3(gid, b0, b1, b2) | cbit
    var i01 = base | abit
    var i10 = base | bbit
    var tr = re[unsafe_offset=i01]
    var ti = im[unsafe_offset=i01]
    re[unsafe_offset=i01] = re[unsafe_offset=i10]
    im[unsafe_offset=i01] = im[unsafe_offset=i10]
    re[unsafe_offset=i10] = tr
    im[unsafe_offset=i10] = ti


@always_inline
def th_phase[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt],
                        cr: Scalar[dt], si: Scalar[dt]):
    """Global phase e^{i phi}, cr = cos(phi), si = sin(phi)."""
    if gid >= work:
        return
    var xr = re[unsafe_offset=gid]
    var xi = im[unsafe_offset=gid]
    re[unsafe_offset=gid] = cr * xr - si * xi
    im[unsafe_offset=gid] = cr * xi + si * xr


@always_inline
def th_prob[dt: DType](gid: Int, work: Int, re: GPtr[dt], im: GPtr[dt], dst: GPtr[dt]):
    """dst[i] = |psi_i|^2."""
    if gid >= work:
        return
    var xr = re[unsafe_offset=gid]
    var xi = im[unsafe_offset=gid]
    dst[unsafe_offset=gid] = xr * xr + xi * xi


# ---------------------------------------------------------------------------
# Reduction pieces (shared by the GPU kernels and the emulator)
# ---------------------------------------------------------------------------
@always_inline
def th_expect_load[dt: DType](gid: Int, nthreads: Int, n: Int, re: GPtr[dt], im: GPtr[dt],
                              x: Int, z: Int) -> SIMD[dt, 2]:
    """Grid-stride partial of sum_i conj(psi[i^x]) psi[i] (-1)^{|i & z|}.

    The term's i^{|x&z|} factor is applied on the host (see GpuStateVector).
    x = z = 0 gives the squared norm.
    """
    var sr = Scalar[dt](0)
    var si = Scalar[dt](0)
    var i = gid
    while i < n:
        var j = i ^ x
        var ar = re[unsafe_offset=i]
        var ai = im[unsafe_offset=i]
        var br = re[unsafe_offset=j]
        var bi = im[unsafe_offset=j]
        var sign = Scalar[dt](1 - 2 * (Int(pop_count(i & z)) & 1))
        sr += sign * (br * ar + bi * ai)
        si += sign * (br * ai - bi * ar)
        i += nthreads
    return SIMD[dt, 2](sr, si)


@always_inline
def th_seg_load[dt: DType](tid: Int, base: Int, seg: Int, re: GPtr[dt], im: GPtr[dt]) -> Scalar[dt]:
    """Thread tid's share of sum_{j < seg} |psi_{base + j}|^2 (block-stride)."""
    var s = Scalar[dt](0)
    var j = tid
    while j < seg:
        var i = base + j
        var xr = re[unsafe_offset=i]
        var xi = im[unsafe_offset=i]
        s += xr * xr + xi * xi
        j += TPB
    return s


@always_inline
def tree_step[dt: DType, origin: MutOrigin, addr: AddressSpace](
    sh: Pointer[Scalar[dt], origin, address_space=addr], tid: Int, s: Int
):
    """One level of the shared-memory tree: sh[tid] += sh[tid + s] for tid < s.

    Reads [s, 2s) and writes [0, s): race-free inside a level, so running the
    threads of a level sequentially (emulator) is exact.
    """
    if tid < s:
        sh[unsafe_offset=tid] = sh[unsafe_offset=tid] + sh[unsafe_offset=tid + s]


@always_inline
def th_sample_seg[dt: DType](gid: Int, shots: Int, re: GPtr[dt], im: GPtr[dt], seg: Int,
                             segs: I64Ptr, resid: GPtr[dt], outs: I64Ptr):
    """Inverse-CDF sampling inside a segment: one thread per shot.

    segs[gid] = segment chosen on the host from the fp64 prefix of segment
    sums, resid[gid] = target - mass before the segment. Returns the first
    index whose running sum exceeds resid; rounding fallback = last index
    with nonzero probability (never a zero-probability state).
    """
    if gid >= shots:
        return
    var base = Int(segs[unsafe_offset=gid]) * seg
    var r = resid[unsafe_offset=gid]
    var acc = Scalar[dt](0)
    var last = base
    for j in range(seg):
        var i = base + j
        var xr = re[unsafe_offset=i]
        var xi = im[unsafe_offset=i]
        var p = xr * xr + xi * xi
        if p > 0:
            last = i
            if acc + p > r:
                outs[unsafe_offset=gid] = Int64(i)
                return
        acc += p
    outs[unsafe_offset=gid] = Int64(last)


# ---------------------------------------------------------------------------
# GPU kernel wrappers (instantiated only inside `-D QUMO_GPU` code paths)
# ---------------------------------------------------------------------------
def gk_init[dt: DType](re: GPtr[dt], im: GPtr[dt], index: Int64, work: Int64):
    th_init[dt](global_idx.x, Int(work), re, im, Int(index))


def gk_1q[dt: DType](re: GPtr[dt], im: GPtr[dt], t: Int32, mr: SIMD[dt, 4], mi: SIMD[dt, 4], work: Int64):
    th_1q[dt](global_idx.x, Int(work), re, im, Int(t), mr, mi)


def gk_diag[dt: DType](re: GPtr[dt], im: GPtr[dt], t: Int32, d: SIMD[dt, 4], skip0: Int32, work: Int64):
    th_diag[dt](global_idx.x, Int(work), re, im, Int(t), d, skip0 != 0)


def gk_x[dt: DType](re: GPtr[dt], im: GPtr[dt], t: Int32, work: Int64):
    th_x[dt](global_idx.x, Int(work), re, im, Int(t))


def gk_cx[dt: DType](re: GPtr[dt], im: GPtr[dt], c: Int32, t: Int32, work: Int64):
    th_cx[dt](global_idx.x, Int(work), re, im, Int(c), Int(t))


def gk_cphase[dt: DType](re: GPtr[dt], im: GPtr[dt], a: Int32, b: Int32, pr: Scalar[dt], pim: Scalar[dt], work: Int64):
    th_cphase[dt](global_idx.x, Int(work), re, im, Int(a), Int(b), pr, pim)


def gk_c1[dt: DType](re: GPtr[dt], im: GPtr[dt], c: Int32, t: Int32, mr: SIMD[dt, 4], mi: SIMD[dt, 4], work: Int64):
    th_c1[dt](global_idx.x, Int(work), re, im, Int(c), Int(t), mr, mi)


def gk_rzz[dt: DType](re: GPtr[dt], im: GPtr[dt], a: Int32, b: Int32, cr: Scalar[dt], si: Scalar[dt], work: Int64):
    th_rzz[dt](global_idx.x, Int(work), re, im, Int(a), Int(b), cr, si)


def gk_swap[dt: DType](re: GPtr[dt], im: GPtr[dt], a: Int32, b: Int32, work: Int64):
    th_swap[dt](global_idx.x, Int(work), re, im, Int(a), Int(b))


def gk_2q[dt: DType](re: GPtr[dt], im: GPtr[dt], q0: Int32, q1: Int32, mr: SIMD[dt, 16], mi: SIMD[dt, 16], work: Int64):
    th_2q[dt](global_idx.x, Int(work), re, im, Int(q0), Int(q1), mr, mi)


def gk_ccx[dt: DType](re: GPtr[dt], im: GPtr[dt], b0: Int32, b1: Int32, b2: Int32, cmask: Int64, tbit: Int64, work: Int64):
    th_ccx[dt](global_idx.x, Int(work), re, im, Int(b0), Int(b1), Int(b2), Int(cmask), Int(tbit))


def gk_ccz[dt: DType](re: GPtr[dt], im: GPtr[dt], b0: Int32, b1: Int32, b2: Int32, work: Int64):
    th_ccz[dt](global_idx.x, Int(work), re, im, Int(b0), Int(b1), Int(b2))


def gk_cswap[dt: DType](re: GPtr[dt], im: GPtr[dt], b0: Int32, b1: Int32, b2: Int32, cbit: Int64, abit: Int64, bbit: Int64, work: Int64):
    th_cswap[dt](global_idx.x, Int(work), re, im, Int(b0), Int(b1), Int(b2), Int(cbit), Int(abit), Int(bbit))


def gk_phase[dt: DType](re: GPtr[dt], im: GPtr[dt], cr: Scalar[dt], si: Scalar[dt], work: Int64):
    th_phase[dt](global_idx.x, Int(work), re, im, cr, si)


def gk_prob[dt: DType](re: GPtr[dt], im: GPtr[dt], dst: GPtr[dt], work: Int64):
    th_prob[dt](global_idx.x, Int(work), re, im, dst)


def gk_expect[dt: DType](re: GPtr[dt], im: GPtr[dt], x: Int64, z: Int64, n: Int64, nthreads: Int64,
                         part_re: GPtr[dt], part_im: GPtr[dt], offset: Int64):
    """One Pauli term: per-block partials into part_*[offset + block]."""
    var shr = unsafe_stack_allocation[TPB, Scalar[dt], address_space=AddressSpace.SHARED]()
    var shi = unsafe_stack_allocation[TPB, Scalar[dt], address_space=AddressSpace.SHARED]()
    var tid = thread_idx.x
    var v = th_expect_load[dt](global_idx.x, Int(nthreads), Int(n), re, im, Int(x), Int(z))
    shr[unsafe_offset=tid] = v[0]
    shi[unsafe_offset=tid] = v[1]
    barrier()
    comptime for k in range(LOG_TPB):
        tree_step[dt](shr, tid, TPB >> (k + 1))
        tree_step[dt](shi, tid, TPB >> (k + 1))
        barrier()
    if tid == 0:
        part_re[unsafe_offset=Int(offset) + block_idx.x] = shr[unsafe_offset=0]
        part_im[unsafe_offset=Int(offset) + block_idx.x] = shi[unsafe_offset=0]


def gk_segsum[dt: DType](re: GPtr[dt], im: GPtr[dt], seg: Int64, dst: GPtr[dt]):
    """Block b sums |psi|^2 over segment b (one block per segment)."""
    var sh = unsafe_stack_allocation[TPB, Scalar[dt], address_space=AddressSpace.SHARED]()
    var tid = thread_idx.x
    sh[unsafe_offset=tid] = th_seg_load[dt](tid, block_idx.x * Int(seg), Int(seg), re, im)
    barrier()
    comptime for k in range(LOG_TPB):
        tree_step[dt](sh, tid, TPB >> (k + 1))
        barrier()
    if tid == 0:
        dst[unsafe_offset=block_idx.x] = sh[unsafe_offset=0]


def gk_sample_seg[dt: DType](re: GPtr[dt], im: GPtr[dt], seg: Int64, segs: I64Ptr, resid: GPtr[dt],
                             outs: I64Ptr, shots: Int64):
    th_sample_seg[dt](global_idx.x, Int(shots), re, im, Int(seg), segs, resid, outs)
