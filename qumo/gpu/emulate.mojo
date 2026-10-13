"""Host emulator for the GPU kernels in `qumo.gpu.kernels`.

Why: MAX's `DeviceContext(api="cpu")` does not execute `enqueue_function`
kernels, and CI machines (and Macs without the Metal toolchain) cannot run
GPU code at all. The emulator runs the *same* `@always_inline` thread phases
the GPU kernels call, for every thread index of the *same* launch grid
(blocks of TPB threads, idle tail threads included), so the GPU indexing,
guards and arithmetic (in the chosen precision) are tested on any machine.

Barrier semantics: a GPU kernel is a sequence of phases separated by
`barrier()`. The emulator runs phase 1 for all TPB threads of a block, then
phase 2 for all threads, and so on, with the block's shared memory as a host
array. Within a phase the kernels are race-free by construction (each thread
owns disjoint items, or reads [s, 2s) and writes [0, s) in a tree level), so
the sequential order is exact. Blocks are independent and run in parallel
CPU tasks; grid-stride elementwise kernels run their threads in parallel
chunks.

This mirrors the "functional simulation of kernels on the host" practice of
CUDA's emulation mode / HIP-CPU, restricted to what QUMO's kernels need.
"""

from std.math import ceildiv
from std.runtime import parallelism_level
from max.algorithm import parallelize

from qumo.gpu.kernels import (
    TPB, LOG_TPB, EXP_G, TOP, DOP, NREG, REG_BITS, SEG_W, SEG_TRANSPOSE, SEG_DIRECT, FIN_TRANSPOSE,
    MAX_GRID_BLOCKS, GPtr, I32Ptr, I64Ptr,
    seg_flag, seg_begin, seg_end,
    tile_base, th_tile_offs, th_reg_map_nat, th_reg_map, th_reg_globals, th_reg_load, th_reg_store,
    th_reg_put, th_reg_get, th_reg_op, th_reg_g0, th_run_cache, RUN_MAX_OPS, RUN_MAX_MATS,
    th_expect_load, th_seg_load, tree_step,
    th_samp_load, th_samp_chunk_scan, th_samp_offsets, th_samp_add, th_samp_search,
)

comptime EMU_PAR_MIN_THREADS = 1 << 14


@always_inline
def grid_blocks(work: Int) -> Int:
    """Blocks of TPB threads covering `work` items (at least one block)."""
    return max(1, ceildiv(work, TPB))


@always_inline
def stride_blocks(work: Int) -> Int:
    """Grid of a grid-stride elementwise kernel: covers `work`, capped at
    MAX_GRID_BLOCKS (2^24 threads), so 32-qubit states stay far below the
    2^32-work-item launch limit of AMD's HSA runtime."""
    return min(MAX_GRID_BLOCKS, grid_blocks(work))


def emu_threads[F: def(Int) -> None](f: F, nthreads: Int):
    """Run the thread body f(global_idx) for g in [0, nthreads)."""
    if nthreads < EMU_PAR_MIN_THREADS:
        for g in range(nthreads):
            f(g)
        return
    var nw = max(1, parallelism_level() * 2)
    var chunk = ceildiv(ceildiv(nthreads, nw), TPB) * TPB
    var nchunks = ceildiv(nthreads, chunk)

    def task(c: Int) {imm f, imm chunk, imm nthreads}:
        var lo = c * chunk
        var hi = min(lo + chunk, nthreads)
        for g in range(lo, hi):
            f(g)

    parallelize(task, nchunks)


def emu_launch[F: def(Int) -> None](f: F, work: Int):
    """One thread per work item: grid_blocks(work) * TPB threads."""
    emu_threads(f, grid_blocks(work) * TPB)


def _blocks[F: def(Int) -> None](task: F, nblocks: Int):
    if nblocks == 1:
        task(0)
    else:
        parallelize(task, nblocks)


def emu_tile[dt: DType](re: GPtr[dt], im: GPtr[dt], ops: I32Ptr, mats: GPtr[dt], hp: I32Ptr, segs: I32Ptr,
                        s0: Int, s1: Int, hoff: Int, nh: Int, low: Int, k: Int, nr: Int, fin: Int,
                        init_index: Int, o0: Int, nops: Int, m0: Int, nm: Int, nblocks: Int):
    """Emulate gk_tile: per block, cache | load | (transpose | ops)* | store,
    with each thread's registers as a host array that survives the barriers."""
    var nthr = 1 << (k - nr)
    var nreg = 1 << nr
    var nseg = s1 - s0

    def task(blk: Int) {imm re, imm im, imm ops, imm mats, imm hp, imm segs, imm s0, imm nseg, imm hoff,
                        imm nh, imm low, imm k, imm nr, imm fin, imm init_index, imm nthr, imm nreg,
                        imm o0, imm nops, imm m0, imm nm}:
        var tsize = 1 << k
        var shr = List[Scalar[dt]](length=tsize, fill=0)
        var shi = List[Scalar[dt]](length=tsize, fill=0)
        var sho = List[UInt32](length=1 << nh, fill=0)
        var sr = shr.unsafe_ptr().as_unsafe_any_origin()
        var si = shi.unsafe_ptr().as_unsafe_any_origin()
        var so = sho.unsafe_ptr().as_unsafe_any_origin()
        var shops = List[Int32](length=DOP * RUN_MAX_OPS, fill=0)
        var shmats = List[Scalar[dt]](length=RUN_MAX_MATS, fill=0)  # 16-byte aligned (host allocator)
        var shsegs = List[Int32](length=SEG_W * RUN_MAX_OPS, fill=0)
        var sops = shops.unsafe_ptr().as_unsafe_any_origin()
        var smats = shmats.unsafe_ptr().as_unsafe_any_origin()
        var ssegs = shsegs.unsafe_ptr().as_unsafe_any_origin()
        var lid = List[Array[UInt32, NREG]](length=nthr, fill=Array[UInt32, NREG](fill=0))
        var g0 = List[UInt32](length=nthr, fill=0)  # per thread: global index of register 0
        var vr = List[Array[Scalar[dt], NREG]](length=nthr, fill=Array[Scalar[dt], NREG](fill=0))
        var vi = List[Array[Scalar[dt], NREG]](length=nthr, fill=Array[Scalar[dt], NREG](fill=0))
        for tid in range(nthr):
            th_tile_offs(tid, nthr, hp, hoff, nh, so)
            th_run_cache[dt](tid, nthr, ops, o0, nops, mats, m0, nm, segs, s0, nseg, sops, smats, ssegs)
        var base = tile_base(blk, low, hp, hoff, nh)
        var w00 = UInt32(ssegs[unsafe_offset=0]) if nseg > 0 else UInt32(0)
        var direct = nseg > 0 and seg_flag(w00) == UInt32(SEG_DIRECT)
        for tid in range(nthr):
            if direct:
                th_reg_map(tid, w00, nr, lid[tid])
            else:
                th_reg_map_nat(tid, k, nr, lid[tid])
            var gi = Array[UInt32, NREG](fill=0)
            th_reg_globals(nreg, lid[tid], base, low, so, gi)
            th_reg_load[dt](nreg, gi, re, im, init_index, vr[tid], vi[tid])
            g0[tid] = gi[0]
        var cur = UInt32(0)
        var is_nat = True
        for s in range(nseg):
            var w0 = UInt32(ssegs[unsafe_offset=SEG_W * s])
            var w1 = UInt32(ssegs[unsafe_offset=SEG_W * s + 1])
            if seg_flag(w0) == UInt32(SEG_TRANSPOSE):
                for tid in range(nthr):  # the kernel rebuilds the old map from `cur`
                    if is_nat:
                        th_reg_map_nat(tid, k, nr, lid[tid])
                    else:
                        th_reg_map(tid, cur, nr, lid[tid])
                    th_reg_put[dt](nreg, lid[tid], vr[tid], vi[tid], sr, si)
                for tid in range(nthr):
                    th_reg_map(tid, w0, nr, lid[tid])
                    th_reg_get[dt](nreg, lid[tid], vr[tid], vi[tid], sr, si)
                    g0[tid] = th_reg_g0(lid[tid][0], base, low, so)
                cur = w0
                is_nat = False
            elif seg_flag(w0) == UInt32(SEG_DIRECT):
                cur = w0
                is_nat = False
            for tid in range(nthr):
                for i in range(seg_begin(w1), seg_end(w1)):
                    var rec = sops.unsafe_load[width=DOP, alignment=16](DOP * i)
                    th_reg_op[dt](nreg, g0[tid], rec, smats, vr[tid], vi[tid])
        if fin == FIN_TRANSPOSE:
            for tid in range(nthr):
                th_reg_map(tid, cur, nr, lid[tid])
                th_reg_put[dt](nreg, lid[tid], vr[tid], vi[tid], sr, si)
            for tid in range(nthr):
                th_reg_map_nat(tid, k, nr, lid[tid])
                th_reg_get[dt](nreg, lid[tid], vr[tid], vi[tid], sr, si)
        for tid in range(nthr):  # the kernel rebuilds the store map: natural, or the last set (FIN_DIRECT)
            if fin != FIN_TRANSPOSE:
                if is_nat:
                    th_reg_map_nat(tid, k, nr, lid[tid])
                else:
                    th_reg_map(tid, cur, nr, lid[tid])
            var gi = Array[UInt32, NREG](fill=0)
            th_reg_globals(nreg, lid[tid], base, low, so, gi)
            th_reg_store[dt](nreg, gi, re, im, vr[tid], vi[tid])
        _ = len(shr)
        _ = len(shi)
        _ = len(sho)
        _ = len(shops)
        _ = len(shmats)
        _ = len(shsegs)

    _blocks(task, nblocks)


def emu_expect[dt: DType](re: GPtr[dt], im: GPtr[dt], x: Int, zs: I64Ptr, zoff: Int, nz: Int,
                          n: Int, nb: Int, part_re: GPtr[dt], part_im: GPtr[dt], poff: Int):
    """Emulate gk_expect: load phase, LOG_TPB tree levels, block partials."""
    var nthreads = nb * TPB

    def task(b: Int) {imm re, imm im, imm x, imm zs, imm zoff, imm nz, imm n, imm nb,
                      imm part_re, imm part_im, imm poff, imm nthreads}:
        var ar = List[Scalar[dt]](length=EXP_G * TPB, fill=0)
        var ai = List[Scalar[dt]](length=EXP_G * TPB, fill=0)
        var pr = ar.unsafe_ptr().as_unsafe_any_origin()
        var pi_ = ai.unsafe_ptr().as_unsafe_any_origin()
        for tid in range(TPB):
            th_expect_load[dt](b * TPB + tid, tid, nthreads, n, re, im, x, zs, zoff, nz, pr, pi_)
        for k in range(LOG_TPB):
            var s = TPB >> (k + 1)
            for g in range(nz):
                for tid in range(TPB):
                    tree_step[dt](pr, g * TPB, tid, s)
                    tree_step[dt](pi_, g * TPB, tid, s)
        for g in range(nz):
            part_re[unsafe_offset=poff + g * nb + b] = pr[unsafe_offset=g * TPB]
            part_im[unsafe_offset=poff + g * nb + b] = pi_[unsafe_offset=g * TPB]
        _ = len(ar)
        _ = len(ai)

    _blocks(task, nb)


def emu_segsum[dt: DType](re: GPtr[dt], im: GPtr[dt], seg: Int, nseg: Int, dst: GPtr[dt]):
    """Emulate gk_segsum (block per segment, shared-memory tree)."""

    def task(b: Int) {imm re, imm im, imm seg, imm dst}:
        var sh = List[Scalar[dt]](length=TPB, fill=0)
        var p = sh.unsafe_ptr().as_unsafe_any_origin()
        for tid in range(TPB):
            p[unsafe_offset=tid] = th_seg_load[dt](tid, b * seg, seg, re, im)
        for k in range(LOG_TPB):
            var s = TPB >> (k + 1)
            for tid in range(TPB):
                tree_step[dt](p, 0, tid, s)
        dst[unsafe_offset=b] = p[unsafe_offset=0]
        _ = len(sh)

    _blocks(task, nseg)


def emu_sample[dt: DType](re: GPtr[dt], im: GPtr[dt], seg: Int, bseg: I64Ptr, blo: I32Ptr,
                          resid: GPtr[dt], outs: I64Ptr, nblocks: Int):
    """Emulate gk_sample: load | chunk scan | offsets | add | search."""

    def task(b: Int) {imm re, imm im, imm seg, imm bseg, imm blo, imm resid, imm outs}:
        var c = List[Scalar[dt]](length=seg, fill=0)
        var t = List[Scalar[dt]](length=TPB, fill=0)
        var cdf = c.unsafe_ptr().as_unsafe_any_origin()
        var tot = t.unsafe_ptr().as_unsafe_any_origin()
        var base = Int(bseg[unsafe_offset=b]) * seg
        for tid in range(TPB):
            th_samp_load[dt](tid, base, seg, re, im, cdf)
        for tid in range(TPB):
            th_samp_chunk_scan[dt](tid, seg, cdf, tot)
        for tid in range(TPB):
            th_samp_offsets[dt](tid, tot)
        for tid in range(TPB):
            th_samp_add[dt](tid, seg, cdf, tot)
        var lo = Int(blo[unsafe_offset=b])
        var hi = Int(blo[unsafe_offset=b + 1])
        for tid in range(TPB):
            th_samp_search[dt](tid, base, seg, cdf, lo, hi, resid, outs)
        _ = len(c)
        _ = len(t)

    _blocks(task, nblocks)
