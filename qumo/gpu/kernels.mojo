"""GPU state-vector kernels: device-agnostic thread bodies + thin MAX wrappers.

Design (ideas from qsim, cuStateVec and Qiskit-Aer-GPU, re-derived for MAX):
  * Same storage as the CPU simulator (`qumo.sim.statevector`): split re/im
    arrays (structure of arrays), little-endian qubit order.
  * Gates run as *tile runs* (`gk_tile`), the gate blocking of qsim's GPU
    simulator and cuStateVec's batched gate application: a tile is the set
    of 2^k amplitudes whose indices differ only in k chosen qubits S (the
    `low` lowest qubits plus k - low "high" ones; low and k come from the
    device profile, qumo.gpu.device.GpuProfile). One thread block handles one
    tile: it reads it once from global memory, applies a whole *run* of gates
    and writes it back once. The host scheduler (`qumo.gpu.tiles`) packs the
    fused plan of `qumo.sim.runner.build_plan` into runs, moving gates past
    non-overlapping ones (they commute). Only the *targets* of non-diagonal
    gates need to be tile qubits: controls and diagonal gates (RZ, P, CZ,
    CP, CRZ, RZZ, CCZ, global phase) test bits of the global amplitude
    index, so they join any run; consecutive diagonal gates are fused into
    one phase table (TK.DGT). A layered 26-qubit circuit (377 plan ops) is
    14-17 state sweeps in fp32 at k = 11 (20 in fp64 at k = 10), a 26-qubit
    QFT 7 (364 plan ops -> 127 tile ops).
  * Register blocking inside a tile: each thread keeps 2^REG_BITS amplitudes
    in registers (16 on Apple GPUs, 8 elsewhere); gates whose targets lie in
    the thread's current *register set* run with no shared memory and no
    barrier; changing the register set is one transpose through shared
    memory (host-planned segments). Loads and stores use a map in which 2^low
    consecutive threads touch 2^low consecutive amplitudes (coalesced for
    every gate, including gates on the highest qubits): the natural map, or
    the run's first / last register set when it allows it (no transpose).
  * Per op, the kernel reads one 16-byte device record (DOP) and its matrix
    with 16-byte vector loads from shared memory; controls, parities and
    phase-table indices are split by the host into a part outside the
    register set (tested once per op on register 0's global index g0) and
    per-register bits, so no per-register global indices stay live.
  * Shared memory: one dynamic region sized per launch (tile + the run's
    cached ops / matrices / segments), or static arrays on Apple GPUs
    (TILE_SMEM_DYNAMIC).
  * Gate operands come from device tables (op records + matrices, uploaded
    once per flush), never from vector kernel arguments: Apple's Metal
    back end raises an internal compiler error on SIMD-vector kernel
    parameters (found on an M5, MAX 26.6), and tables also keep the argument
    count far below Metal's 31-slot limit.
  * Every kernel is a sequence of `@always_inline` *thread phases*
    (`th_*(tid, ...)`) separated by `barrier()`. The GPU kernel `gk_*` is a
    thin wrapper; the host emulator (`qumo.gpu.emulate`) runs the same phases
    for every thread of every block of the same launch grid, a barrier being
    the end of a phase. The GPU indexing and arithmetic are therefore
    unit-tested on any machine, with or without a GPU (MAX's
    `DeviceContext(api="cpu")` does not execute `enqueue_function` kernels).
  * Precision is the comptime DType `dt`: float32 on Apple GPUs (Metal has no
    fp64), float32 or float64 on NVIDIA / AMD (policy in qumo.gpu.device:
    float32 by default on every GPU, float64 on request).
  * Reductions (Pauli-term expectation values, norms, sampling CDF segments)
    use the classic shared-memory tree (CUDA "reduction #3"): per-thread
    grid-stride partial -> TPB-wide tree in shared memory -> one partial per
    block -> fp64 sum on the host. Fixed order and no atomics, so results are
    bit-reproducible run to run on the same device. Up to EXP_G Pauli terms
    with the same X mask share one pass over the state.
  * Launch sizes stay below 2^31 threads at 32 qubits (AMD's HSA grid limit
    is 2^32 work items): tiles launch 2^(n-k) blocks, elementwise kernels use
    a capped grid with grid-stride loops.
  * Kernel parameters are DevicePassable only: pointers and Int32 / Int64 /
    Scalar[dt] (`Int` is not DevicePassable).

The `gk_*` wrappers are only instantiated by launches inside
`comptime if is_defined["QUMO_GPU"]()` blocks (see statevector_gpu.mojo), so
default builds never compile device code.
"""

from std.bit import pop_count
from std.sys.defines import get_defined_int
from std.sys import has_apple_gpu_accelerator, has_amd_gpu_accelerator
from std.utils.static_tuple import StaticTuple
from std.memory import unsafe_stack_allocation, AddressSpace
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.memory import external_memory
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA

comptime LOG_TPB = get_defined_int["QUMO_GPU_LOG_TPB", 8]()  # tuning knob (-D)
comptime TPB = 1 << LOG_TPB  # threads per block for every kernel (256)
comptime RED_ITEMS = 16  # target amplitudes per thread in reductions
comptime MAX_RED_BLOCKS = 1024  # cap on partials per reduction (host sums them)
comptime MAX_GRID_BLOCKS = 1 << 16  # cap for elementwise launches (grid-stride beyond)
comptime EXP_G = 8  # Pauli terms (same X mask) per expectation launch
comptime LOW_BITS = 5  # default lowest qubits kept in every tile (coalescing: 32 lanes); see GpuProfile.low
comptime LOW_MIN = 3  # smallest `low` the tile kernel supports (its offset table is sized for it)
comptime TILE_LOG2_BYTES = get_defined_int["QUMO_GPU_TILE_LOG2_BYTES", 14]()  # 16 KiB per tile (re + im)
comptime SAMPLE_LOG2_BYTES = 14  # shared CDF per sampling block: 16 KiB
# gk_tile's shared memory: one dynamic region sized per launch (tile size k
# chosen at runtime, cache sized to the run: more blocks per multiprocessor)
# or static arrays for the default tile size and the worst-case run. Static
# on Apple GPUs (dynamic measured 0-3% slower on the M5, whose 32 KiB per
# threadgroup admits no larger tile anyway), dynamic elsewhere.
comptime TILE_SMEM_DYNAMIC = get_defined_int["QUMO_GPU_TILE_SMEM_DYNAMIC", 0 if has_apple_gpu_accelerator() else 1]() == 1

comptime GPtr[dt: DType] = Pointer[Scalar[dt], MutAnyOrigin]
comptime I64Ptr = Pointer[Int64, MutAnyOrigin]
comptime I32Ptr = Pointer[Int32, MutAnyOrigin]

# Host op record: TOP Int32 per op (see TK), used by the scheduler; the
# kernels read the device record (DOP Int32, below), built from it once the
# op's register set is known. Segment record: SEG_W Int32.
comptime TOP = 8  # kind, a, b, x, moff, pa, pb, cm
# Device op record: DOP x Int32 = 16 B, one vector load from shared memory:
#   d0: code in bits [0, 8): kind (TK, bits 0-2), register position of
#       target a (bits 3-4) and of b (bits 5-6), CODE_CTRL (bit 7: the op has
#       controls); matrix offset relative to the run's first matrix value in
#       bits [8, 32) (a multiple of 4: matrices load as 16-byte vectors);
#   d1: G1 / G1R / XC / SW: controls outside the register set (global
#       mask); DG: parity qubits outside the register set;
#   d2: DG: all-set qubits outside the register set;
#   d3: G1 / G1R / XC / SW: bit j = register j satisfies the controls that
#       are register qubits; DG: bit j = parity of register j's in-register
#       parity qubits, bit 16 + j = register j has every in-register all-set
#       qubit.
# A register's global index differs from register 0's (g0) only in the
# register qubits, so every control / parity test needs g0 plus these
# host-computed masks: no per-register global indices are kept live in the
# op loop (16 fewer registers per thread at 16 amplitudes per thread), and a
# diagonal needs one pop_count per op instead of one per register.
comptime DOP = 4
comptime CODE_CTRL = 128
comptime DIAG_MAX_WIDTH = 6  # widest diagonal phase table (TK.DGT): 2^6 complex values
# Amplitudes per thread = 2^REG_BITS. Per-target default: 4 on Apple GPUs
# (measured on the M5 after the segment cache: -6 to -10% on layered / random
# circuits at 26 qubits, 20-25% fewer transposes), 3 elsewhere until measured
# (16 fp32 values + indices per thread risk register spills on NVIDIA / AMD,
# and fp64 at k = 10 would leave 64 threads per block).
comptime REG_BITS = get_defined_int["QUMO_GPU_REG_BITS", 4 if has_apple_gpu_accelerator() else 3]()
comptime NREG = 1 << REG_BITS
# Threads per gk_tile block = 2^(k - REG_BITS) for tile size k (runtime).
# TILE_SAFE_THREADS launches on every target whatever the kernel's register
# count (NVIDIA: 255 registers x 256 threads <= 64 Ki registers per block;
# AMD: the launch bound below; Apple: 1024-thread threadgroups), so larger
# blocks are used only when the compiled kernel reports that it admits them
# (GpuProfile.tile_threads_*, from the kernel's MAX_THREADS_PER_BLOCK
# attribute on the device). Without that report the tile size stays at
# <= TILE_SAFE_THREADS threads.
comptime TILE_SAFE_THREADS = 256
# Launch bound of gk_tile (max threads per block the compiler must support),
# 0 = none. AMD: 256 (the default workgroup bound of 1024 caps a gfx942
# kernel at 128 VGPRs, and the fp64 / heavy tile kernels spilled to scratch
# under it; with 256 the cap is the full register file). NVIDIA: none
# (ptxas picks the register count freely; .maxntid 1024 would cap it at 64
# registers), the host reads the result back instead (see above).
comptime TILE_LAUNCH_BOUND = get_defined_int["QUMO_GPU_TILE_LAUNCH_BOUND", 256 if has_amd_gpu_accelerator() else 0]()
comptime RUN_MAX_OPS = 128  # ops per tile run (cached in shared memory)
comptime RUN_MAX_MATS = 2048  # matrix values per tile run (cached in shared memory; room for phase tables)
# Segment record, 2 x Int32 (a run's records are cached in shared memory:
# RUN_MAX_OPS segments at most, 1 KiB):
#   w0: register set = local qubit of register position i in bits [4i, 4i + 4)
#       (ascending, i < nr <= REG_BITS <= 4), remap flag in bits [16, 18)
#       (SEG_KEEP: same set as before; SEG_TRANSPOSE: change the set through
#       shared memory; SEG_DIRECT: first segment of a run whose set has only
#       local qubits >= low: the tile is loaded from global memory straight
#       into this set, no transpose);
#   w1: the segment's ops [begin, end) relative to the run's first op, begin
#       in bits [0, 16), end in bits [16, 32).
comptime SEG_W = 2
comptime SEG_FLAG_SHIFT = 16
comptime SEG_KEEP = 0
comptime SEG_TRANSPOSE = 1
comptime SEG_DIRECT = 2
# Run end (`fin`): FIN_NATURAL (last set is the natural one), FIN_TRANSPOSE
# (transpose back to the natural map, then store), FIN_DIRECT (store straight
# from the last set: all its local qubits are >= low).
comptime FIN_NATURAL = 0
comptime FIN_TRANSPOSE = 1
comptime FIN_DIRECT = 2


struct TK:
    """Tile op kinds. Only the *targets* of non-diagonal gates must be tile
    qubits (a, b = local tile qubits; pa, pb = their register positions in
    the segment's register set). Controls and diagonal gates test bits of the
    global amplitude index, so they may act on any qubit of the state:
    cm / pm / am are global qubit masks (UInt32 bit patterns in Int32).
    Matrix values start at moff."""

    comptime G1 = 0  # 2x2 on target a (pa) where (g & cm) == cm: [r00 r01 r10 r11 i00 i01 i10 i11]
    comptime SW = 1  # swap a <-> b (pa < pb) where (g & cm) == cm
    comptime G2 = 2  # dense 4x4 on (a, b), index b(a) + 2 b(b): 16 re then 16 im (row-major)
    comptime DG = 3  # diagonal: where (g & am) == am multiply by c[parity(g & pm)]: [c0r c0i c1r c1i]
    # record fields for DG: a = pm, b = am
    comptime XC = 4  # Pauli X on target a (pa) where (g & cm) == cm: X, CX, CCX (register swap, no flops)
    comptime G1R = 5  # real 2x2 (all imaginary parts exactly 0: H, RY, ...): [r00 r01 r10 r11], half the flops of G1
    # fused run of diagonal gates on <= DIAG_MAX_WIDTH qubits: one phase table
    # of 2^w complex values (re, im interleaved), index = the w qubits' bits.
    # Host record: a = qubit mask, b = gate group; device record: d1 = the
    # table qubits outside the register set (5-bit global positions, table
    # bits 0..nout-1), d2 = table bit of each register position (3 bits each,
    # 7 = not in the table) | nout << 16.
    comptime DGT = 6


@always_inline
def tile_bits[dt: DType]() -> Int:
    """Default log2 amplitudes per tile: 16 KiB of shared memory (11 for fp32,
    10 for fp64). The tile size of a launch is a runtime value (GpuProfile
    picks larger tiles on GPUs with more shared memory)."""
    return TILE_LOG2_BYTES - 3 if dt == DType.float32 else TILE_LOG2_BYTES - 4


@always_inline
def tile_smem_bytes(es: Int, k: Int, low: Int, nops: Int, nm: Int, nseg: Int) -> Int:
    """Dynamic shared memory of one gk_tile launch (layout in gk_tile): the
    tile (re + im), the run's matrices, op records, segment records and the
    high-pattern offset table. es = bytes per real (4 fp32, 8 fp64)."""
    return es * (2 * (1 << k) + nm) + 4 * (DOP * nops + SEG_W * nseg + (1 << (k - low)))


@always_inline
def tile_smem_worst(es: Int, k: Int) -> Int:
    """Largest tile_smem_bytes any run can need at tile size k (the scheduler
    caps runs at RUN_MAX_OPS ops / segments and RUN_MAX_MATS matrix values)."""
    return tile_smem_bytes(es, k, LOW_MIN, RUN_MAX_OPS, RUN_MAX_MATS, RUN_MAX_OPS)


@always_inline
def tile_smem_typical(es: Int, k: Int) -> Int:
    """tile_smem_bytes of a typical run (64 ops, 512 matrix values, 32
    segments): the size the tile-size policy budgets 2 blocks per
    multiprocessor with (GpuProfile._pick_tile_bits)."""
    return tile_smem_bytes(es, k, LOW_MIN, 64, 512, 32)


@always_inline
def sample_seg[dt: DType]() -> Int:
    """Amplitudes per sampling segment (shared CDF of 16 KiB: 4096 fp32, 2048 fp64)."""
    return 1 << (SAMPLE_LOG2_BYTES - 2) if dt == DType.float32 else 1 << (SAMPLE_LOG2_BYTES - 3)


@always_inline
def ins0(p: Int, b: Int) -> Int:
    """Insert a 0 bit at position b of p (pair-index addressing)."""
    return ((p >> b) << (b + 1)) | (p & ((1 << b) - 1))


@always_inline
def ins0u(p: UInt32, b: UInt32) -> UInt32:
    """ins0 in 32-bit arithmetic (requires b <= 30)."""
    return ((p >> b) << (b + 1)) | (p & ((UInt32(1) << b) - 1))


# ---------------------------------------------------------------------------
# Tile kernel phases (gk_tile). A tile = 2^k amplitudes; a block has
# 2^(k - nr) threads and each thread keeps 2^nr amplitudes in registers
# (nr = min(REG_BITS, k)), the qsim / cuStateVec register-blocking scheme:
#   * lid[j] = local tile index of register j, gi[j] = its global amplitude
#     index. A *register set* R (nr tile qubits) defines the map: the
#     thread's other local bits are its thread index with 0s inserted at R,
#     register j sets R's bits to the bits of j.
#   * A gate whose targets are in R runs entirely in registers: no shared
#     memory, no barrier. Controls and diagonal gates only test bits of
#     gi[j], so they run under any R and may involve any qubit of the state.
#   * Changing R (host-chosen segments) = one transpose through shared
#     memory: store registers by the old map, barrier, load by the new map.
#   * Global load/store use the natural map (R = top nr local qubits), or a
#     run's first / last register set directly when all its local qubits are
#     >= low (SEG_DIRECT / FIN_DIRECT: no transpose at either end); either
#     way thread bits 0..low-1 are local bits 0..low-1, so 32 consecutive
#     threads touch 32 consecutive amplitudes (coalesced).
# Index arithmetic is 32-bit: amplitude indices are < 2^32 for the <= 32
# qubits QUMO allows; GPUs (Apple's in particular) emulate 64-bit integer ALU
# work with several instructions.
# ---------------------------------------------------------------------------
@always_inline
def tile_base(blk: Int, low: Int, hp: I32Ptr, hoff: Int, nh: Int) -> UInt32:
    """Index of the first amplitude of tile `blk`: 0 bits inserted at the tile
    qubits (the `low` lowest ones, then the ascending high ones hp[hoff:]).
    Computed once per block in 64-bit (positions up to 31), result < 2^32."""
    var b = blk << low
    for j in range(nh):
        b = ins0(b, Int(hp[unsafe_offset=hoff + j]))
    return UInt32(b)


@always_inline
def th_tile_offs[o: MutOrigin, sp: AddressSpace](
    tid: Int, nthr: Int, hp: I32Ptr, hoff: Int, nh: Int, offs: Pointer[UInt32, o, address_space=sp]
):
    """offs[h] = global offset of local high pattern h (scatter of h's bits to hp)."""
    var h = tid
    while h < (1 << nh):
        var v = UInt32(0)
        for j in range(nh):
            if (h >> j) & 1 == 1:
                v |= UInt32(1) << UInt32(hp[unsafe_offset=hoff + j])
        offs[unsafe_offset=h] = v
        h += nthr


@always_inline
def th_run_cache[dt: DType, o: MutOrigin, sp: AddressSpace](
    tid: Int, nthr: Int, ops: I32Ptr, o0: Int, nops: Int, mats: GPtr[dt], m0: Int, nm: Int,
    segs: I32Ptr, s0: Int, nseg: Int,
    sops: Pointer[Int32, o, address_space=sp], smats: Pointer[Scalar[dt], o, address_space=sp],
    ssegs: Pointer[Int32, o, address_space=sp],
):
    """Copy the run's op records, matrices and segment records to shared
    memory (coalesced), so the op loop and the register-set changes read them
    at shared-memory latency (re-reading segment records from device memory
    after every transpose barrier cost ~0.5 ms per transpose at 24 qubits on
    the M5, measured)."""
    var i = tid
    while i < DOP * nops:
        sops[unsafe_offset=i] = ops[unsafe_offset=DOP * o0 + i]
        i += nthr
    i = tid
    while i < nm:
        smats[unsafe_offset=i] = mats[unsafe_offset=m0 + i]
        i += nthr
    i = tid
    while i < SEG_W * nseg:
        ssegs[unsafe_offset=i] = segs[unsafe_offset=SEG_W * s0 + i]
        i += nthr


@always_inline
def seg_flag(w0: UInt32) -> UInt32:
    """Remap flag of a packed segment record (SEG_KEEP / SEG_TRANSPOSE)."""
    return (w0 >> UInt32(SEG_FLAG_SHIFT)) & 3


@always_inline
def seg_begin(w1: UInt32) -> Int:
    return Int(w1 & 0xFFFF)


@always_inline
def seg_end(w1: UInt32) -> Int:
    return Int(w1 >> 16)


@always_inline
def th_reg_map_nat(tid: Int, k: Int, nr: Int, mut lid: Array[UInt32, NREG]):
    """Natural map: register j <-> local index tid + j * 2^(k - nr)."""
    var t = UInt32(tid)
    var sh = UInt32(k - nr)
    comptime for j in range(NREG):
        lid[j] = t | (UInt32(j) << sh)


@always_inline
def th_reg_map(tid: Int, w0: UInt32, nr: Int, mut lid: Array[UInt32, NREG]):
    """Map of the register set packed in a segment record's w0 (ascending
    local qubits, 4 bits each; positions >= nr unused)."""
    var r = Array[UInt32, REG_BITS](fill=0)
    comptime for i in range(REG_BITS):
        r[i] = (w0 >> UInt32(4 * i)) & 15
    var tb = UInt32(tid)
    comptime for i in range(REG_BITS):
        if i < nr:
            tb = ins0u(tb, r[i])
    comptime for j in range(NREG):
        var l = tb
        comptime for i in range(REG_BITS):
            comptime if (j >> i) & 1 == 1:
                if i < nr:
                    l |= UInt32(1) << r[i]
        lid[j] = l


@always_inline
def th_reg_globals[o: MutOrigin, sp: AddressSpace](
    nreg: Int, lid: Array[UInt32, NREG], base: UInt32, low: Int,
    offs: Pointer[UInt32, o, address_space=sp], mut gi: Array[UInt32, NREG],
):
    """gi[j] = global amplitude index of register j."""
    var lo = UInt32(low)
    var lmask = (UInt32(1) << lo) - 1
    comptime for j in range(NREG):
        if j < nreg:
            var l = lid[j]
            gi[j] = base | offs[unsafe_offset=Int(l >> lo)] | (l & lmask)


@always_inline
def th_reg_g0[o: MutOrigin, sp: AddressSpace](
    l0: UInt32, base: UInt32, low: Int, offs: Pointer[UInt32, o, address_space=sp]
) -> UInt32:
    """Global amplitude index of register 0 (local index l0)."""
    var lo = UInt32(low)
    return base | offs[unsafe_offset=Int(l0 >> lo)] | (l0 & ((UInt32(1) << lo) - 1))


@always_inline
def th_reg_load[dt: DType](
    nreg: Int, gi: Array[UInt32, NREG], re: GPtr[dt], im: GPtr[dt], init_index: Int,
    mut vr: Array[Scalar[dt], NREG], mut vi: Array[Scalar[dt], NREG],
):
    """Global -> registers. init_index >= 0: write |init_index> instead of
    reading (folds the state initialisation into the first run)."""
    if init_index >= 0:
        var ii = UInt32(init_index)
        comptime for j in range(NREG):
            if j < nreg:
                vr[j] = Scalar[dt](1) if gi[j] == ii else Scalar[dt](0)
                vi[j] = 0
        return
    comptime for j in range(NREG):
        if j < nreg:
            vr[j] = re[unsafe_offset=Int(gi[j])]
            vi[j] = im[unsafe_offset=Int(gi[j])]


@always_inline
def th_reg_store[dt: DType](
    nreg: Int, gi: Array[UInt32, NREG], re: GPtr[dt], im: GPtr[dt],
    vr: Array[Scalar[dt], NREG], vi: Array[Scalar[dt], NREG],
):
    """Registers -> global (natural map, coalesced)."""
    comptime for j in range(NREG):
        if j < nreg:
            re[unsafe_offset=Int(gi[j])] = vr[j]
            im[unsafe_offset=Int(gi[j])] = vi[j]


@always_inline
def th_reg_put[dt: DType, o: MutOrigin, sp: AddressSpace](
    nreg: Int, lid: Array[UInt32, NREG], vr: Array[Scalar[dt], NREG], vi: Array[Scalar[dt], NREG],
    sr: Pointer[Scalar[dt], o, address_space=sp], si: Pointer[Scalar[dt], o, address_space=sp],
):
    """Transpose, first half: registers -> shared memory (by the current map)."""
    comptime for j in range(NREG):
        if j < nreg:
            sr[unsafe_offset=Int(lid[j])] = vr[j]
            si[unsafe_offset=Int(lid[j])] = vi[j]


@always_inline
def th_reg_get[dt: DType, o: MutOrigin, sp: AddressSpace](
    nreg: Int, lid: Array[UInt32, NREG], mut vr: Array[Scalar[dt], NREG], mut vi: Array[Scalar[dt], NREG],
    sr: Pointer[Scalar[dt], o, address_space=sp], si: Pointer[Scalar[dt], o, address_space=sp],
):
    """Transpose, second half: shared memory -> registers (by the new map)."""
    comptime for j in range(NREG):
        if j < nreg:
            vr[j] = sr[unsafe_offset=Int(lid[j])]
            vi[j] = si[unsafe_offset=Int(lid[j])]


@always_inline
def _row4r[dt: DType, om: MutOrigin, am: AddressSpace](m: Pointer[Scalar[dt], om, address_space=am], o: Int,
                      x0r: Scalar[dt], x0i: Scalar[dt], x1r: Scalar[dt], x1i: Scalar[dt],
                      x2r: Scalar[dt], x2i: Scalar[dt], x3r: Scalar[dt], x3i: Scalar[dt]) -> Scalar[dt]:
    """Re of one row of a complex 4x4 (row starts at o; imaginary parts at o + 16)."""
    return (m[unsafe_offset=o] * x0r - m[unsafe_offset=o + 16] * x0i
            + m[unsafe_offset=o + 1] * x1r - m[unsafe_offset=o + 17] * x1i
            + m[unsafe_offset=o + 2] * x2r - m[unsafe_offset=o + 18] * x2i
            + m[unsafe_offset=o + 3] * x3r - m[unsafe_offset=o + 19] * x3i)


@always_inline
def _row4i[dt: DType, om: MutOrigin, am: AddressSpace](m: Pointer[Scalar[dt], om, address_space=am], o: Int,
                      x0r: Scalar[dt], x0i: Scalar[dt], x1r: Scalar[dt], x1i: Scalar[dt],
                      x2r: Scalar[dt], x2i: Scalar[dt], x3r: Scalar[dt], x3i: Scalar[dt]) -> Scalar[dt]:
    """Im of one row of a complex 4x4."""
    return (m[unsafe_offset=o] * x0i + m[unsafe_offset=o + 16] * x0r
            + m[unsafe_offset=o + 1] * x1i + m[unsafe_offset=o + 17] * x1r
            + m[unsafe_offset=o + 2] * x2i + m[unsafe_offset=o + 18] * x2r
            + m[unsafe_offset=o + 3] * x3i + m[unsafe_offset=o + 19] * x3r)


@always_inline
def _g1[dt: DType, P: Int, CTRL: Bool](
    nreg: Int, ok0: Bool, cbits: UInt32, ma: SIMD[dt, 4], mb: SIMD[dt, 4],
    mut vr: Array[Scalar[dt], NREG], mut vi: Array[Scalar[dt], NREG],
):
    """2x2 on register position P, ma = [r00 r01 r10 r11], mb = imaginary
    parts (CTRL: only registers j with ok0 and bit j of cbits)."""
    comptime for j in range(NREG):
        comptime if (j >> P) & 1 == 0:
            comptime j1 = j | (1 << P)
            if j1 < nreg:
                var ok = True
                comptime if CTRL:
                    ok = ok0 and ((cbits >> UInt32(j)) & 1) == 1
                if ok:
                    var ar = vr[j]
                    var ai = vi[j]
                    var br = vr[j1]
                    var bi = vi[j1]
                    vr[j] = ma[0] * ar - mb[0] * ai + ma[1] * br - mb[1] * bi
                    vi[j] = ma[0] * ai + mb[0] * ar + ma[1] * bi + mb[1] * br
                    vr[j1] = ma[2] * ar - mb[2] * ai + ma[3] * br - mb[3] * bi
                    vi[j1] = ma[2] * ai + mb[2] * ar + ma[3] * bi + mb[3] * br


@always_inline
def _g1r[dt: DType, P: Int, CTRL: Bool](
    nreg: Int, ok0: Bool, cbits: UInt32, ma: SIMD[dt, 4],
    mut vr: Array[Scalar[dt], NREG], mut vi: Array[Scalar[dt], NREG],
):
    """Real 2x2 on register position P: 8 multiply-adds per pair instead of 16."""
    comptime for j in range(NREG):
        comptime if (j >> P) & 1 == 0:
            comptime j1 = j | (1 << P)
            if j1 < nreg:
                var ok = True
                comptime if CTRL:
                    ok = ok0 and ((cbits >> UInt32(j)) & 1) == 1
                if ok:
                    var ar = vr[j]
                    var ai = vi[j]
                    var br = vr[j1]
                    var bi = vi[j1]
                    vr[j] = ma[0] * ar + ma[1] * br
                    vi[j] = ma[0] * ai + ma[1] * bi
                    vr[j1] = ma[2] * ar + ma[3] * br
                    vi[j1] = ma[2] * ai + ma[3] * bi


@always_inline
def _xc[dt: DType, P: Int, CTRL: Bool](
    nreg: Int, ok0: Bool, cbits: UInt32,
    mut vr: Array[Scalar[dt], NREG], mut vi: Array[Scalar[dt], NREG],
):
    """Pauli X on register position P: swap register pairs (CTRL as in _g1)."""
    comptime for j in range(NREG):
        comptime if (j >> P) & 1 == 0:
            comptime j1 = j | (1 << P)
            if j1 < nreg:
                var ok = True
                comptime if CTRL:
                    ok = ok0 and ((cbits >> UInt32(j)) & 1) == 1
                if ok:
                    var tr = vr[j]
                    var ti = vi[j]
                    vr[j] = vr[j1]
                    vi[j] = vi[j1]
                    vr[j1] = tr
                    vi[j1] = ti


@always_inline
def th_reg_op[dt: DType, om: MutOrigin, am: AddressSpace, HEAVY: Bool = True](
    nreg: Int, g0: UInt32, rec: SIMD[DType.int32, 4], m: Pointer[Scalar[dt], om, address_space=am],
    mut vr: Array[Scalar[dt], NREG], mut vi: Array[Scalar[dt], NREG],
):
    """Apply one gate (device record `rec`, see DOP) to the thread's registers
    (no shared memory, no barrier). g0 = global index of register 0.

    Target register positions are dispatched to comptime specialisations, so
    every register index is a compile-time constant (no local-memory spills);
    uncontrolled gates skip the control test. HEAVY = False compiles out
    SWAP / dense-4x4 code: runs without those ops use that lighter kernel
    (fewer live registers, higher occupancy: -30% time on M5 for 1q / CX /
    diagonal runs, measured).
    """
    var d0 = UInt32(rec[0])
    var kind = d0 & 7
    var p = (d0 >> 3) & 3
    var mo = Int(d0 >> 8)
    var m1 = UInt32(rec[1])
    var bits = UInt32(rec[3])
    if kind == UInt32(TK.DGT):
        # fused diagonal gates: gather the table qubits outside the register
        # set from g0 once, add each register's own table bits, one complex
        # multiply per register for the whole group
        var r2 = UInt32(rec[2])
        var nout = Int(r2 >> 16)
        var idx0 = UInt32(0)
        for t in range(nout):
            idx0 |= ((g0 >> ((m1 >> UInt32(5 * t)) & 31)) & 1) << UInt32(t)
        var rb = Array[UInt32, REG_BITS](fill=0)
        comptime for i in range(REG_BITS):
            var tb = (r2 >> UInt32(3 * i)) & 7
            rb[i] = UInt32(0) if tb == 7 else UInt32(1) << tb
        comptime for j in range(NREG):
            if j < nreg:
                var ix = idx0
                comptime for i in range(REG_BITS):
                    comptime if (j >> i) & 1 == 1:
                        ix |= rb[i]
                var o = mo + 2 * Int(ix)
                var fr = m[unsafe_offset=o]
                var fi = m[unsafe_offset=o + 1]
                var xr = vr[j]
                var xi = vi[j]
                vr[j] = fr * xr - fi * xi
                vi[j] = fr * xi + fi * xr
        return
    if kind == UInt32(TK.DG):
        # diagonal: RZ / P / CZ / CP / CRZ / RZZ / CCZ / phase. Parity and
        # all-set tests: once on g0 outside the register set, per register
        # only a host-precomputed bit.
        var c = m.unsafe_load[width=4, alignment=16](mo)
        var m2 = UInt32(rec[2])
        var p0 = UInt32(pop_count(g0 & m1)) & 1
        var a0 = (g0 & m2) == m2
        comptime for j in range(NREG):
            if j < nreg:
                var odd = ((bits >> UInt32(j)) ^ p0) & 1 == 1
                var act = a0 and ((bits >> UInt32(16 + j)) & 1) == 1
                var fr = c[2] if odd else c[0]
                var fi = c[3] if odd else c[1]
                if not act:
                    fr = 1
                    fi = 0
                var xr = vr[j]
                var xi = vi[j]
                vr[j] = fr * xr - fi * xi
                vi[j] = fr * xi + fi * xr
        return
    var ok0 = (g0 & m1) == m1  # controls outside the register set
    var ctrl = (d0 & UInt32(CODE_CTRL)) != 0
    if kind == UInt32(TK.XC):
        comptime for P in range(REG_BITS):
            if p == UInt32(P):
                if ctrl:
                    _xc[dt, P, True](nreg, ok0, bits, vr, vi)
                else:
                    _xc[dt, P, False](nreg, ok0, bits, vr, vi)
    elif kind == UInt32(TK.G1):
        var ma = m.unsafe_load[width=4, alignment=16](mo)
        var mb = m.unsafe_load[width=4, alignment=16](mo + 4)
        comptime for P in range(REG_BITS):
            if p == UInt32(P):
                if ctrl:
                    _g1[dt, P, True](nreg, ok0, bits, ma, mb, vr, vi)
                else:
                    _g1[dt, P, False](nreg, ok0, bits, ma, mb, vr, vi)
    elif kind == UInt32(TK.G1R):
        var ma = m.unsafe_load[width=4, alignment=16](mo)
        comptime for P in range(REG_BITS):
            if p == UInt32(P):
                if ctrl:
                    _g1r[dt, P, True](nreg, ok0, bits, ma, vr, vi)
                else:
                    _g1r[dt, P, False](nreg, ok0, bits, ma, vr, vi)
    elif HEAVY and kind == UInt32(TK.SW):
        var q = (d0 >> 5) & 3
        comptime for P in range(REG_BITS):
            comptime for Q in range(P + 1, REG_BITS):
                if p == UInt32(P) and q == UInt32(Q):
                    comptime for j in range(NREG):
                        comptime if (j >> P) & 1 == 1 and (j >> Q) & 1 == 0:
                            comptime j2 = j ^ (1 << P) ^ (1 << Q)  # > j (Q > P)
                            if j2 < nreg and ok0 and ((bits >> UInt32(j)) & 1) == 1:
                                var tr = vr[j]
                                var ti = vi[j]
                                vr[j] = vr[j2]
                                vi[j] = vi[j2]
                                vr[j2] = tr
                                vi[j2] = ti
    elif HEAVY and kind == UInt32(TK.G2):
        var q = (d0 >> 5) & 3
        comptime for P in range(REG_BITS):
            comptime for Q in range(REG_BITS):
                comptime if P != Q:
                    if p == UInt32(P) and q == UInt32(Q):
                        comptime for j0 in range(NREG):
                            comptime if (j0 >> P) & 1 == 0 and (j0 >> Q) & 1 == 0:
                                comptime j1 = j0 | (1 << P)
                                comptime j2 = j0 | (1 << Q)
                                comptime j3 = j0 | (1 << P) | (1 << Q)
                                if j3 < nreg:
                                    var x0r = vr[j0]
                                    var x0i = vi[j0]
                                    var x1r = vr[j1]
                                    var x1i = vi[j1]
                                    var x2r = vr[j2]
                                    var x2i = vi[j2]
                                    var x3r = vr[j3]
                                    var x3i = vi[j3]
                                    vr[j0] = _row4r[dt](m, mo, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i)
                                    vi[j0] = _row4i[dt](m, mo, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i)
                                    vr[j1] = _row4r[dt](m, mo + 4, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i)
                                    vi[j1] = _row4i[dt](m, mo + 4, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i)
                                    vr[j2] = _row4r[dt](m, mo + 8, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i)
                                    vi[j2] = _row4i[dt](m, mo + 8, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i)
                                    vr[j3] = _row4r[dt](m, mo + 12, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i)
                                    vi[j3] = _row4i[dt](m, mo + 12, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i)


# ---------------------------------------------------------------------------
# Elementwise and reduction thread bodies
# ---------------------------------------------------------------------------
@always_inline
def th_prob[dt: DType](gid: Int, nthreads: Int, work: Int, re: GPtr[dt], im: GPtr[dt], dst: GPtr[dt]):
    """dst[i] = |psi_i|^2 (grid-stride; 64-bit counter: work may be 2^32)."""
    var i = gid
    while i < work:
        var xr = re[unsafe_offset=i]
        var xi = im[unsafe_offset=i]
        dst[unsafe_offset=i] = xr * xr + xi * xi
        i += nthreads


@always_inline
def th_expect_load[dt: DType, o: MutOrigin, sp: AddressSpace](
    gid: Int, tid: Int, nthreads: Int, n: Int, re: GPtr[dt], im: GPtr[dt], x: Int,
    zs: I64Ptr, zoff: Int, nz: Int,
    accr: Pointer[Scalar[dt], o, address_space=sp], acci: Pointer[Scalar[dt], o, address_space=sp],
):
    """Grid-stride partials of sum_i conj(psi[i^x]) psi[i] (-1)^{|i & z_g|} for
    the nz <= EXP_G masks z_g = zs[zoff + g]: one pass over the state for all
    of them, accumulators in registers (comptime-unrolled), stored at the end
    to shared memory [g * TPB + tid]. The term's i^{|x&z|} factor is applied
    on the host; x = z = 0 is the squared norm."""
    var ar_ = Array[Scalar[dt], EXP_G](fill=0)
    var ai_ = Array[Scalar[dt], EXP_G](fill=0)
    var zz = Array[UInt32, EXP_G](fill=0)
    comptime for g in range(EXP_G):
        if g < nz:
            zz[g] = UInt32(zs[unsafe_offset=zoff + g])
    var xm = UInt32(x)
    var i = gid
    while i < n:
        var iu = UInt32(i)
        var j = Int(iu ^ xm)
        var vr = re[unsafe_offset=i]
        var vi = im[unsafe_offset=i]
        var br = re[unsafe_offset=j]
        var bi = im[unsafe_offset=j]
        var pr = br * vr + bi * vi
        var pim = br * vi - bi * vr
        comptime for g in range(EXP_G):
            if g < nz:
                var sg = Scalar[dt](1) - Scalar[dt](2) * Scalar[dt](pop_count(iu & zz[g]) & 1)
                ar_[g] += sg * pr
                ai_[g] += sg * pim
        i += nthreads
    comptime for g in range(EXP_G):
        if g < nz:
            accr[unsafe_offset=g * TPB + tid] = ar_[g]
            acci[unsafe_offset=g * TPB + tid] = ai_[g]


@always_inline
def th_seg_load[dt: DType](tid: Int, base: Int, seg: Int, re: GPtr[dt], im: GPtr[dt]) -> Scalar[dt]:
    """Thread tid's share of sum_{j < seg} |psi_{base + j}|^2 (block-stride)."""
    var s = Scalar[dt](0)
    var j = UInt32(tid)
    var b = UInt32(base)
    var sg = UInt32(seg)
    while j < sg:
        var i = Int(b + j)
        var xr = re[unsafe_offset=i]
        var xi = im[unsafe_offset=i]
        s += xr * xr + xi * xi
        j += UInt32(TPB)
    return s


@always_inline
def tree_step[dt: DType, origin: MutOrigin, addr: AddressSpace](
    sh: Pointer[Scalar[dt], origin, address_space=addr], off: Int, tid: Int, s: Int
):
    """One level of the shared-memory tree: sh[off+tid] += sh[off+tid+s] for tid < s.

    Reads [s, 2s) and writes [0, s): race-free inside a level, so running the
    threads of a level sequentially (emulator) is exact.
    """
    if tid < s:
        sh[unsafe_offset=off + tid] = sh[unsafe_offset=off + tid] + sh[unsafe_offset=off + tid + s]


# Sampling block phases: block = one segment of `seg` amplitudes that holds at
# least one shot (shots are sorted, so a block's shots are a contiguous range).
@always_inline
def th_samp_load[dt: DType, o: MutOrigin, sp: AddressSpace](
    tid: Int, base: Int, seg: Int, re: GPtr[dt], im: GPtr[dt], cdf: Pointer[Scalar[dt], o, address_space=sp]
):
    """cdf[j] = |psi_{base+j}|^2 (coalesced: consecutive threads, consecutive j)."""
    var j = UInt32(tid)
    var b = UInt32(base)
    var sg = UInt32(seg)
    while j < sg:
        var i = Int(b + j)
        var xr = re[unsafe_offset=i]
        var xi = im[unsafe_offset=i]
        cdf[unsafe_offset=Int(j)] = xr * xr + xi * xi
        j += UInt32(TPB)


@always_inline
def th_samp_chunk_scan[dt: DType, o: MutOrigin, sp: AddressSpace](
    tid: Int, seg: Int, cdf: Pointer[Scalar[dt], o, address_space=sp], tot: Pointer[Scalar[dt], o, address_space=sp]
):
    """Inclusive scan of thread tid's chunk of the segment; chunk total -> tot[tid]."""
    var cs = (seg + TPB - 1) // TPB
    var lo = tid * cs
    var hi = min(seg, lo + cs)
    var acc = Scalar[dt](0)
    var j = lo
    while j < hi:
        acc += cdf[unsafe_offset=j]
        cdf[unsafe_offset=j] = acc
        j += 1
    tot[unsafe_offset=tid] = acc


@always_inline
def th_samp_offsets[dt: DType, o: MutOrigin, sp: AddressSpace](tid: Int, tot: Pointer[Scalar[dt], o, address_space=sp]):
    """Thread 0: exclusive scan of the TPB chunk totals (in place)."""
    if tid == 0:
        var acc = Scalar[dt](0)
        for t in range(TPB):
            var v = tot[unsafe_offset=t]
            tot[unsafe_offset=t] = acc
            acc += v


@always_inline
def th_samp_add[dt: DType, o: MutOrigin, sp: AddressSpace](
    tid: Int, seg: Int, cdf: Pointer[Scalar[dt], o, address_space=sp], tot: Pointer[Scalar[dt], o, address_space=sp]
):
    """Add the chunk's exclusive offset: cdf becomes the segment's inclusive CDF."""
    var cs = (seg + TPB - 1) // TPB
    var lo = tid * cs
    var hi = min(seg, lo + cs)
    var off = tot[unsafe_offset=tid]
    var j = lo
    while j < hi:
        cdf[unsafe_offset=j] = cdf[unsafe_offset=j] + off
        j += 1


@always_inline
def th_samp_search[dt: DType, o: MutOrigin, sp: AddressSpace](
    tid: Int, base: Int, seg: Int, cdf: Pointer[Scalar[dt], o, address_space=sp],
    lo: Int, hi: Int, resid: GPtr[dt], outs: I64Ptr,
):
    """Shots lo..hi-1: first j with cdf[j] > resid (binary search), so a
    zero-probability amplitude is never chosen; if rounding leaves none, the
    last j with nonzero probability."""
    var s = lo + tid
    while s < hi:
        var r = resid[unsafe_offset=s]
        var a = 0
        var b = seg
        while a < b:
            var mid = (a + b) >> 1
            if cdf[unsafe_offset=mid] > r:
                b = mid
            else:
                a = mid + 1
        if a >= seg:
            a = seg - 1
            while a > 0 and not (cdf[unsafe_offset=a] > cdf[unsafe_offset=a - 1]):
                a -= 1
        outs[unsafe_offset=s] = Int64(base + a)
        s += TPB


# ---------------------------------------------------------------------------
# GPU kernel wrappers (instantiated only inside `-D QUMO_GPU` code paths)
# ---------------------------------------------------------------------------
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(TILE_LAUNCH_BOUND)))
def gk_tile_lb[dt: DType, FULL: Bool, HEAVY: Bool](
    re: GPtr[dt], im: GPtr[dt], ops: I32Ptr, mats: GPtr[dt], hp: I32Ptr, segs: I32Ptr,
    s0: Int32, s1: Int32, hoff: Int32, nh: Int32, low: Int32, k: Int32, nr: Int32,
    fin: Int32, init_index: Int64, o0: Int32, nops: Int32, m0: Int32, nm: Int32,
):
    """gk_tile with launch bound TILE_LAUNCH_BOUND (> 0; AMD by default: the
    compiler may then use the whole register file). The body is gk_tile
    itself (a device-function call the back end inlines): moving the body
    into an @always_inline helper shared by both kernels made the Metal
    kernel ~1.9x slower on the M5 (measured), so gk_tile keeps it."""
    gk_tile[dt, FULL, HEAVY](re, im, ops, mats, hp, segs, s0, s1, hoff, nh, low, k, nr, fin, init_index,
                             o0, nops, m0, nm)


def gk_tile[dt: DType, FULL: Bool, HEAVY: Bool](
    re: GPtr[dt], im: GPtr[dt], ops: I32Ptr, mats: GPtr[dt], hp: I32Ptr, segs: I32Ptr,
    s0: Int32, s1: Int32, hoff: Int32, nh: Int32, low: Int32, k: Int32, nr: Int32,
    fin: Int32, init_index: Int64, o0: Int32, nops: Int32, m0: Int32, nm: Int32,
):
    """One run of gates on every tile: block = tile of 2^k amplitudes, thread =
    2^nr amplitudes in registers; segments s0..s1-1 of the run, each with its
    register set (transpose through shared memory when it changes). The run's
    ops [o0, o0 + nops), matrices [m0, m0 + nm) and segment records are cached
    in shared memory. Specialisations: FULL (nr == REG_BITS: per-register
    guards compiled out), HEAVY (the run has SWAP / dense 4x4 ops). Register
    pressure is kept low: the local-index map (`lid`) is rebuilt only around
    transposes, from the packed register set of the current segment."""
    comptime assert REG_BITS >= 1 and REG_BITS <= 4, "QUMO_GPU_REG_BITS must be 1..4 (4-bit register fields)"
    # One dynamic shared region, sized per launch by the host
    # (tile_smem_bytes): [re tile | im tile | matrices | op records | segment
    # records | offsets]. Runs carry only the cache they need (a typical run
    # needs 1.5-3 KiB of the 9 KiB a static worst case would reserve), so
    # more blocks fit per multiprocessor, and k can follow the device
    # (TILE_SMEM_DYNAMIC; static arrays otherwise, k <= tile_bits[dt]()).
    var kk = Int(k)
    var lo = Int(low)
    var nseg = Int(s1) - Int(s0)
    var sr: Pointer[Scalar[dt], MutUntrackedOrigin, address_space=AddressSpace.SHARED]
    var si: Pointer[Scalar[dt], MutUntrackedOrigin, address_space=AddressSpace.SHARED]
    var smats: Pointer[Scalar[dt], MutUntrackedOrigin, address_space=AddressSpace.SHARED]
    var sops: Pointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
    var ssegs: Pointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
    var offs: Pointer[UInt32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
    comptime if TILE_SMEM_DYNAMIC:
        var ts = 1 << kk
        var dyn = external_memory[Scalar[dt], address_space=AddressSpace.SHARED, alignment=16]()
        sr = dyn
        si = dyn.unsafe_offset(ts)
        smats = dyn.unsafe_offset(2 * ts)
        sops = dyn.unsafe_offset(2 * ts + Int(nm)).unsafe_bitcast[Int32]()
        ssegs = sops.unsafe_offset(DOP * Int(nops))
        offs = ssegs.unsafe_offset(SEG_W * nseg).unsafe_bitcast[UInt32]()
    else:
        comptime TS = 1 << tile_bits[dt]()
        comptime NOFF = 1 << (tile_bits[dt]() - LOW_MIN)
        sr = unsafe_stack_allocation[TS, Scalar[dt], address_space=AddressSpace.SHARED]()
        si = unsafe_stack_allocation[TS, Scalar[dt], address_space=AddressSpace.SHARED]()
        smats = unsafe_stack_allocation[RUN_MAX_MATS, Scalar[dt], alignment=16, address_space=AddressSpace.SHARED]()
        sops = unsafe_stack_allocation[DOP * RUN_MAX_OPS, Int32, alignment=16, address_space=AddressSpace.SHARED]()
        ssegs = unsafe_stack_allocation[SEG_W * RUN_MAX_OPS, Int32, address_space=AddressSpace.SHARED]()
        offs = unsafe_stack_allocation[NOFF, UInt32, address_space=AddressSpace.SHARED]()
    var tid = thread_idx.x
    var nrr = REG_BITS if FULL else Int(nr)
    var nreg = NREG if FULL else 1 << nrr
    var nthr = 1 << (kk - nrr)
    th_tile_offs(tid, nthr, hp, Int(hoff), Int(nh), offs)
    th_run_cache[dt](tid, nthr, ops, Int(o0), Int(nops), mats, Int(m0), Int(nm), segs, Int(s0), nseg, sops, smats, ssegs)
    var base = tile_base(block_idx.x, lo, hp, Int(hoff), Int(nh))
    var vr = Array[Scalar[dt], NREG](fill=0)
    var vi = Array[Scalar[dt], NREG](fill=0)
    barrier()
    var lid0 = Array[UInt32, NREG](fill=0)
    var gi0 = Array[UInt32, NREG](fill=0)
    # load straight into the first segment's set when it allows coalescing
    # (all its local qubits >= low: thread bits 0..low-1 stay local bits 0..low-1)
    var w00 = UInt32(ssegs[unsafe_offset=0]) if nseg > 0 else UInt32(0)
    if nseg > 0 and seg_flag(w00) == UInt32(SEG_DIRECT):
        th_reg_map(tid, w00, nrr, lid0)
    else:
        th_reg_map_nat(tid, kk, nrr, lid0)
    th_reg_globals(nreg, lid0, base, lo, offs, gi0)
    th_reg_load[dt](nreg, gi0, re, im, Int(init_index), vr, vi)
    var g0 = gi0[0]  # global index of register 0 under the current map (the op loop needs no other)
    var cur = UInt32(0)  # packed register set of the current map (if not is_nat)
    var is_nat = True  # current map = natural map
    var used = False  # shared tile written (a barrier must precede the next put)
    for s in range(nseg):
        var w0 = UInt32(ssegs[unsafe_offset=SEG_W * s])
        var w1 = UInt32(ssegs[unsafe_offset=SEG_W * s + 1])
        if seg_flag(w0) == UInt32(SEG_TRANSPOSE):
            if used:
                barrier()
            var lid = Array[UInt32, NREG](fill=0)
            if is_nat:
                th_reg_map_nat(tid, kk, nrr, lid)
            else:
                th_reg_map(tid, cur, nrr, lid)
            th_reg_put[dt](nreg, lid, vr, vi, sr, si)
            barrier()
            th_reg_map(tid, w0, nrr, lid)
            th_reg_get[dt](nreg, lid, vr, vi, sr, si)
            g0 = th_reg_g0(lid[0], base, lo, offs)
            cur = w0
            is_nat = False
            used = True
        elif seg_flag(w0) == UInt32(SEG_DIRECT):  # first segment: loaded in this set
            cur = w0
            is_nat = False
        for i in range(seg_begin(w1), seg_end(w1)):
            var rec = sops.unsafe_load[width=DOP, alignment=16](DOP * i)
            th_reg_op[dt, HEAVY=HEAVY](nreg, g0, rec, smats, vr, vi)
    var lid = Array[UInt32, NREG](fill=0)
    if Int(fin) == FIN_TRANSPOSE:
        if used:
            barrier()
        th_reg_map(tid, cur, nrr, lid)
        th_reg_put[dt](nreg, lid, vr, vi, sr, si)
        barrier()
        th_reg_map_nat(tid, kk, nrr, lid)
        th_reg_get[dt](nreg, lid, vr, vi, sr, si)
    elif is_nat:
        th_reg_map_nat(tid, kk, nrr, lid)
    else:  # FIN_DIRECT: store straight from the last set
        th_reg_map(tid, cur, nrr, lid)
    var gi = Array[UInt32, NREG](fill=0)
    th_reg_globals(nreg, lid, base, lo, offs, gi)
    th_reg_store[dt](nreg, gi, re, im, vr, vi)


def gk_prob[dt: DType](re: GPtr[dt], im: GPtr[dt], dst: GPtr[dt], work: Int64, nthreads: Int64):
    th_prob[dt](global_idx.x, Int(nthreads), Int(work), re, im, dst)


def gk_expect[dt: DType](re: GPtr[dt], im: GPtr[dt], x: Int64, zs: I64Ptr, zoff: Int32, nz: Int32,
                         n: Int64, nthreads: Int64, part_re: GPtr[dt], part_im: GPtr[dt],
                         poff: Int64, nb: Int64):
    """Up to EXP_G Pauli terms with X mask x: per-block partials of term g go
    to part_*[poff + g * nb + block]."""
    var accr = unsafe_stack_allocation[EXP_G * TPB, Scalar[dt], address_space=AddressSpace.SHARED]()
    var acci = unsafe_stack_allocation[EXP_G * TPB, Scalar[dt], address_space=AddressSpace.SHARED]()
    var tid = thread_idx.x
    th_expect_load[dt](global_idx.x, tid, Int(nthreads), Int(n), re, im, Int(x), zs, Int(zoff), Int(nz), accr, acci)
    barrier()
    comptime for k in range(LOG_TPB):
        for g in range(Int(nz)):
            tree_step[dt](accr, g * TPB, tid, TPB >> (k + 1))
            tree_step[dt](acci, g * TPB, tid, TPB >> (k + 1))
        barrier()
    if tid == 0:
        for g in range(Int(nz)):
            part_re[unsafe_offset=Int(poff) + g * Int(nb) + block_idx.x] = accr[unsafe_offset=g * TPB]
            part_im[unsafe_offset=Int(poff) + g * Int(nb) + block_idx.x] = acci[unsafe_offset=g * TPB]


def gk_segsum[dt: DType](re: GPtr[dt], im: GPtr[dt], seg: Int64, dst: GPtr[dt]):
    """Block b sums |psi|^2 over segment b (one block per segment)."""
    var sh = unsafe_stack_allocation[TPB, Scalar[dt], address_space=AddressSpace.SHARED]()
    var tid = thread_idx.x
    sh[unsafe_offset=tid] = th_seg_load[dt](tid, block_idx.x * Int(seg), Int(seg), re, im)
    barrier()
    comptime for k in range(LOG_TPB):
        tree_step[dt](sh, 0, tid, TPB >> (k + 1))
        barrier()
    if tid == 0:
        dst[unsafe_offset=block_idx.x] = sh[unsafe_offset=0]


def gk_sample[dt: DType](re: GPtr[dt], im: GPtr[dt], seg: Int64, bseg: I64Ptr, blo: I32Ptr,
                         resid: GPtr[dt], outs: I64Ptr):
    """Block b: CDF of segment bseg[b] in shared memory, then binary search for
    the shots blo[b] .. blo[b+1]-1."""
    comptime SEGMAX = sample_seg[dt]()
    var cdf = unsafe_stack_allocation[SEGMAX, Scalar[dt], address_space=AddressSpace.SHARED]()
    var tot = unsafe_stack_allocation[TPB, Scalar[dt], address_space=AddressSpace.SHARED]()
    var tid = thread_idx.x
    var b = block_idx.x
    var sg = Int(seg)
    var base = Int(bseg[unsafe_offset=b]) * sg
    th_samp_load[dt](tid, base, sg, re, im, cdf)
    barrier()
    th_samp_chunk_scan[dt](tid, sg, cdf, tot)
    barrier()
    th_samp_offsets[dt](tid, tot)
    barrier()
    th_samp_add[dt](tid, sg, cdf, tot)
    barrier()
    th_samp_search[dt](tid, base, sg, cdf, Int(blo[unsafe_offset=b]), Int(blo[unsafe_offset=b + 1]), resid, outs)
