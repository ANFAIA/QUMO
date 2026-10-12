"""Host emulator for the GPU kernels in `qumo.gpu.kernels`.

Why: MAX's `DeviceContext(api="cpu")` does not execute `enqueue_function`
kernels, and CI machines (and Macs without the Metal toolchain) cannot run
GPU code at all. The emulator runs the *same* `@always_inline` thread bodies
the GPU kernels call, over every thread index of the *same* launch grid
(`ceil(work / TPB)` blocks of `TPB` threads, tail threads included), so the
GPU indexing, guards and arithmetic (in the chosen precision) are tested on
any machine. Elementwise kernels are race-free by construction (each thread
owns disjoint amplitudes), so emulating threads in parallel CPU chunks is
exact. Reductions emulate the shared-memory tree level by level: all threads
of a block run the load phase, then each tree level runs for all threads (the
GPU's `barrier()` between levels becomes the end of a sequential loop).

This mirrors the "functional simulation of kernels on the host" practice of
CUDA's emulation mode / HIP-CPU, restricted to what QUMO's kernels need.
"""

from std.math import ceildiv
from std.runtime import parallelism_level
from max.algorithm import parallelize

from qumo.gpu.kernels import TPB, LOG_TPB, GPtr, tree_step

comptime EMU_PAR_MIN_THREADS = 1 << 14


@always_inline
def grid_blocks(work: Int) -> Int:
    """Blocks of TPB threads covering `work` items (at least one block)."""
    return max(1, ceildiv(work, TPB))


def emu_launch[F: def(Int) -> None](f: F, work: Int):
    """Run the thread body f(global_idx) for every thread of the launch grid."""
    var nthreads = grid_blocks(work) * TPB
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


def emu_block_reduce2[dt: DType, L: def(Int, Int) -> SIMD[dt, 2]](
    load: L, nblocks: Int, part_re: GPtr[dt], part_im: GPtr[dt], offset: Int
):
    """Emulate a 2-channel shared-memory tree reduction (gk_expect).

    load(block, tid) is the per-thread load phase; block b's result goes to
    part_*[offset + b], exactly like the GPU kernel.
    """

    def task(b: Int) {imm load, imm part_re, imm part_im, imm offset}:
        var shr = List[Scalar[dt]](length=TPB, fill=0)
        var shi = List[Scalar[dt]](length=TPB, fill=0)
        var pr = shr.unsafe_ptr().as_unsafe_any_origin()
        var pi_ = shi.unsafe_ptr().as_unsafe_any_origin()
        for tid in range(TPB):
            var v = load(b, tid)
            pr[unsafe_offset=tid] = v[0]
            pi_[unsafe_offset=tid] = v[1]
        # barrier()
        for k in range(LOG_TPB):
            var s = TPB >> (k + 1)
            for tid in range(TPB):
                tree_step[dt](pr, tid, s)
                tree_step[dt](pi_, tid, s)
            # barrier()
        part_re[unsafe_offset=offset + b] = pr[unsafe_offset=0]
        part_im[unsafe_offset=offset + b] = pi_[unsafe_offset=0]
        _ = len(shr)
        _ = len(shi)

    if nblocks == 1:
        task(0)
    else:
        parallelize(task, nblocks)


def emu_block_reduce1[dt: DType, L: def(Int, Int) -> Scalar[dt]](
    load: L, nblocks: Int, dst: GPtr[dt]
):
    """Emulate a 1-channel shared-memory tree reduction (gk_segsum)."""

    def task(b: Int) {imm load, imm dst}:
        var sh = List[Scalar[dt]](length=TPB, fill=0)
        var p = sh.unsafe_ptr().as_unsafe_any_origin()
        for tid in range(TPB):
            p[unsafe_offset=tid] = load(b, tid)
        for k in range(LOG_TPB):
            var s = TPB >> (k + 1)
            for tid in range(TPB):
                tree_step[dt](p, tid, s)
        dst[unsafe_offset=b] = p[unsafe_offset=0]
        _ = len(sh)

    if nblocks == 1:
        task(0)
    else:
        parallelize(task, nblocks)
