"""GpuStateVector: the QUMO state vector on a GPU (MAX, vendor-neutral).

What it is
  * Owns the amplitudes as two device buffers (split re/im, like the CPU
    `StateVector`) plus the `DeviceContext` (one in-order stream). Precision
    is the comptime parameter `dtype` (float32 / float64; see
    `qumo.gpu.device` for the policy: fp32 on Apple, fp64 default elsewhere).
  * Executes the CPU runner's execution plan unchanged: `build_plan` fuses
    every run of 1q gates per wire into one 2x2 (qsim / Aer "gate fusion")
    and the plan's `KOp`s map 1:1 onto the kernels in `qumo.gpu.kernels`.
    Launches are asynchronous on one stream: no host synchronisation between
    kernels, only when results are read back (to_host, probabilities,
    expectation, sample). Launch shape: TPB = 256 threads per block,
    grid = ceil(work / 256) (cuStateVec / Aer-GPU use the same 1-D scheme).
  * Expectation values: one reduction launch per Pauli term into a partials
    buffer (<= 1024 blocks per term), ONE device->host copy for all terms,
    fp64 summation of the partials on the host (deterministic, no atomics).
  * Sampling never copies the state: a block per 4096-amplitude segment
    reduces the segment's probability on the GPU, the host builds the fp64
    prefix over segments (2^n / 4096 numbers) and maps the sorted uniforms of
    the QUMO RNG onto (segment, residual); a second kernel (one thread per
    shot) finds each outcome inside its segment. The RNG stream use is the
    same as `StateVector.sample` (sorted uniforms, then the same Fisher-Yates
    shuffle), so a seed reproduces the same shots run to run.

Emulation mode (`GpuStateVector.emulator(n)`, or `emulate=True`): the same
object keeps host arrays and runs every kernel *body* through
`qumo.gpu.emulate` over the same launch grid. All host logic here (plan
lowering, launch geometry, reductions, segment sampling) is therefore
exercised by tests on machines without a GPU. Device mode compiles only with
`-D QUMO_GPU`; otherwise it raises QM-E470.

Error codes: QM-E470 not compiled, QM-E471 dynamic/noisy circuit on the GPU
sampler, QM-E472 fp64 on Apple, QM-E473 zero-norm state, QM-E474 circuit wider
than the GPU state, QM-E475 unsupported plan op.
"""

from std.math import cos, sin, ceildiv
from std.collections import Optional
from std.bit import pop_count
from max.gpu.host import DeviceContext, DeviceBuffer

from qumo.ir.gates import Op, M2, M4
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.rng import Rng
from qumo.sim.statevector import StateVector
from qumo.sim.runner import build_plan, KOp, K, SampleResult, check_statevector_size
from qumo.gpu.kernels import (
    TPB, RED_ITEMS, MAX_RED_BLOCKS, SAMPLE_SEG, GPtr, I64Ptr,
    th_init, th_1q, th_diag, th_x, th_cx, th_cphase, th_c1, th_rzz, th_swap, th_2q,
    th_ccx, th_ccz, th_cswap, th_phase, th_prob, th_expect_load, th_seg_load, th_sample_seg,
    gk_init, gk_1q, gk_diag, gk_x, gk_cx, gk_cphase, gk_c1, gk_rzz, gk_swap, gk_2q,
    gk_ccx, gk_ccz, gk_cswap, gk_phase, gk_prob, gk_expect, gk_segsum, gk_sample_seg,
)
from qumo.gpu.emulate import emu_launch, emu_block_reduce2, emu_block_reduce1, grid_blocks
from qumo.gpu.device import (
    GPU_COMPILED, GPU_DEFAULT_DTYPE, device_kernels_enabled, precision_name,
    gpu_not_compiled_error, gpu_fp64_error,
)


@always_inline
def _mptr[dt: DType](lst: List[Scalar[dt]]) -> GPtr[dt]:
    """Mutable any-origin pointer into a host array (emulated device memory)."""
    return lst.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()


@always_inline
def _sorted3(a: Int, b: Int, c: Int) -> Tuple[Int, Int, Int]:
    var x = a
    var y = b
    var z = c
    if x > y:
        swap(x, y)
    if y > z:
        swap(y, z)
    if x > y:
        swap(x, y)
    return (x, y, z)


@always_inline
def _is_x_matrix(m: M2) -> Bool:
    return (abs(m.re[0]) + abs(m.im[0]) + abs(m.re[3]) + abs(m.im[3])) < 1e-15 and m.re[1] == 1.0 and m.re[2] == 1.0 and m.im[1] == 0.0 and m.im[2] == 0.0


def reduction_blocks(n: Int) -> Int:
    """Blocks of a reduction over n items: ~RED_ITEMS items per thread, capped."""
    return min(MAX_RED_BLOCKS, max(1, ceildiv(n, TPB * RED_ITEMS)))


struct GpuStateVector[dtype: DType = GPU_DEFAULT_DTYPE](Movable):
    var num_qubits: Int
    var emulated: Bool
    var ctx: Optional[DeviceContext]
    var dre: Optional[DeviceBuffer[Self.dtype]]
    var dim: Optional[DeviceBuffer[Self.dtype]]
    var hre: List[Scalar[Self.dtype]]  # emulated device memory (empty on a GPU)
    var him: List[Scalar[Self.dtype]]
    var launches: Int  # kernel launches issued by mutating ops (diagnostics)

    def __init__(out self, num_qubits: Int, emulate: Bool = False) raises:
        """|0...0> on the GPU (emulate=False) or in the host emulator."""
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "GpuStateVector supports float32 and float64"
        check_statevector_size(num_qubits)
        self.num_qubits = num_qubits
        self.emulated = emulate
        self.ctx = None
        self.dre = None
        self.dim = None
        self.hre = List[Scalar[Self.dtype]]()
        self.him = List[Scalar[Self.dtype]]()
        self.launches = 0
        var n = 1 << num_qubits
        if emulate:
            self.hre = List[Scalar[Self.dtype]](length=n, fill=0)
            self.him = List[Scalar[Self.dtype]](length=n, fill=0)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                var ctx = DeviceContext()
                self.dre = ctx.enqueue_create_buffer[Self.dtype](n)
                self.dim = ctx.enqueue_create_buffer[Self.dtype](n)
                self.ctx = ctx^
            else:
                raise Self._no_device()
        self.reset()

    @staticmethod
    def emulator(num_qubits: Int) raises -> Self:
        """Host-emulated GPU state (runs the kernel bodies on the CPU)."""
        return Self(num_qubits, emulate=True)

    @staticmethod
    def _no_device() -> Error:
        comptime if GPU_COMPILED:
            return gpu_fp64_error()
        else:
            return gpu_not_compiled_error()

    def size(self) -> Int:
        return 1 << self.num_qubits

    def precision(self) -> String:
        return precision_name[Self.dtype]()

    def method_name(self) -> String:
        """'gpu:statevector:fp32' (or 'gpu-emulator:...'), for SampleResult.method."""
        var p = "gpu-emulator" if self.emulated else "gpu"
        return p + ":statevector:" + self.precision()

    def device_name(self) -> String:
        if self.emulated:
            return "host-emulator"
        return self.ctx.value().name()

    def synchronize(self) raises:
        """Wait for all enqueued kernels (no-op when emulated)."""
        if not self.emulated:
            self.ctx.value().synchronize()

    @always_inline
    def _re(self) -> GPtr[Self.dtype]:
        return _mptr[Self.dtype](self.hre)

    @always_inline
    def _im(self) -> GPtr[Self.dtype]:
        return _mptr[Self.dtype](self.him)

    # ------------------------------------------------------------ init
    def reset(mut self) raises:
        """Back to |0...0>."""
        self.set_basis(0)

    def set_basis(mut self, index: Int) raises:
        """Computational basis state |index>."""
        var work = self.size()
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm index}:
                th_init[Self.dtype](g, work, re, im, index)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_init[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int64(index), Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    def load_host(mut self, sv: StateVector) raises:
        """Upload a CPU state (rounded to `dtype`); sizes must match."""
        if sv.num_qubits != self.num_qubits:
            raise Error("QM-E474: host state has " + String(sv.num_qubits) + " qubits but the GPU state has " + String(self.num_qubits) + ". Hint: allocate GpuStateVector(sv.num_qubits).")
        var n = self.size()
        if self.emulated:
            for i in range(n):
                self.hre[i] = Scalar[Self.dtype](sv.re[i])
                self.him[i] = Scalar[Self.dtype](sv.im[i])
            return
        comptime if device_kernels_enabled[Self.dtype]():
            with self.dre.value().map_to_host() as h:
                for i in range(n):
                    h[i] = Scalar[Self.dtype](sv.re[i])
            with self.dim.value().map_to_host() as h:
                for i in range(n):
                    h[i] = Scalar[Self.dtype](sv.im[i])
        else:
            raise Self._no_device()

    # ------------------------------------------------------------ 1q
    def apply_1q(mut self, m: M2, t: Int) raises:
        """Dense 2x2 on qubit t (diagonal matrices take the phase kernel)."""
        if m.is_diagonal():
            self.apply_diag_1q(m.re[0], m.im[0], m.re[3], m.im[3], t)
            return
        var mr = m.re.cast[Self.dtype]()
        var mi = m.im.cast[Self.dtype]()
        var work = self.size() >> 1
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm t, imm mr, imm mi}:
                th_1q[Self.dtype](g, work, re, im, t, mr, mi)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_1q[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int32(t), mr, mi, Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    def apply_diag_1q(mut self, d0r: Float64, d0i: Float64, d1r: Float64, d1i: Float64, t: Int) raises:
        """diag(d0, d1) on qubit t; the |0> half is skipped when d0 == 1."""
        var d = SIMD[DType.float64, 4](d0r, d0i, d1r, d1i).cast[Self.dtype]()
        var skip0 = d0r == 1.0 and d0i == 0.0
        var work = self.size() >> 1
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm t, imm d, imm skip0}:
                th_diag[Self.dtype](g, work, re, im, t, d, skip0)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_diag[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int32(t), d, Int32(1 if skip0 else 0), Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    def apply_x(mut self, t: Int) raises:
        var work = self.size() >> 1
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm t}:
                th_x[Self.dtype](g, work, re, im, t)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_x[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int32(t), Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    # ------------------------------------------------------------ 2q
    def apply_cx(mut self, c: Int, t: Int) raises:
        var work = self.size() >> 2
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm c, imm t}:
                th_cx[Self.dtype](g, work, re, im, c, t)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_cx[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int32(c), Int32(t), Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    def apply_cphase(mut self, pr: Float64, pi_: Float64, a: Int, b: Int) raises:
        """Multiply amplitudes with bits a and b set by (pr + i pi_) (CZ, CP)."""
        var spr = Scalar[Self.dtype](pr)
        var spi = Scalar[Self.dtype](pi_)
        var work = self.size() >> 2
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm a, imm b, imm spr, imm spi}:
                th_cphase[Self.dtype](g, work, re, im, a, b, spr, spi)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_cphase[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int32(a), Int32(b), spr, spi, Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    def apply_controlled_1q(mut self, m: M2, c: Int, t: Int) raises:
        var mr = m.re.cast[Self.dtype]()
        var mi = m.im.cast[Self.dtype]()
        var work = self.size() >> 2
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm c, imm t, imm mr, imm mi}:
                th_c1[Self.dtype](g, work, re, im, c, t, mr, mi)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_c1[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int32(c), Int32(t), mr, mi, Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    def apply_rzz(mut self, theta: Float64, a: Int, b: Int) raises:
        """exp(-i theta/2 Z_a Z_b)."""
        self._rzz(cos(theta / 2), sin(theta / 2), a, b)

    def _rzz(mut self, cr: Float64, si: Float64, a: Int, b: Int) raises:
        var scr = Scalar[Self.dtype](cr)
        var ssi = Scalar[Self.dtype](si)
        var work = self.size()
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm a, imm b, imm scr, imm ssi}:
                th_rzz[Self.dtype](g, work, re, im, a, b, scr, ssi)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_rzz[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int32(a), Int32(b), scr, ssi, Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    def apply_swap(mut self, a: Int, b: Int) raises:
        var work = self.size() >> 2
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm a, imm b}:
                th_swap[Self.dtype](g, work, re, im, a, b)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_swap[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int32(a), Int32(b), Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    def apply_2q(mut self, m: M4, q0: Int, q1: Int) raises:
        """Dense 4x4 on (q0, q1); matrix index = b(q0) + 2 b(q1)."""
        var mr = m.re.cast[Self.dtype]()
        var mi = m.im.cast[Self.dtype]()
        var work = self.size() >> 2
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm q0, imm q1, imm mr, imm mi}:
                th_2q[Self.dtype](g, work, re, im, q0, q1, mr, mi)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_2q[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int32(q0), Int32(q1), mr, mi, Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    # ------------------------------------------------------------ 3q
    def apply_ccx(mut self, c0: Int, c1: Int, t: Int) raises:
        var s = _sorted3(c0, c1, t)
        var b0 = s[0]
        var b1 = s[1]
        var b2 = s[2]
        var cmask = (1 << c0) | (1 << c1)
        var tbit = 1 << t
        var work = self.size() >> 3
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm b0, imm b1, imm b2, imm cmask, imm tbit}:
                th_ccx[Self.dtype](g, work, re, im, b0, b1, b2, cmask, tbit)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_ccx[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int32(b0), Int32(b1), Int32(b2), Int64(cmask), Int64(tbit), Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    def apply_ccz(mut self, a: Int, b: Int, c: Int) raises:
        var s = _sorted3(a, b, c)
        var b0 = s[0]
        var b1 = s[1]
        var b2 = s[2]
        var work = self.size() >> 3
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm b0, imm b1, imm b2}:
                th_ccz[Self.dtype](g, work, re, im, b0, b1, b2)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_ccz[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int32(b0), Int32(b1), Int32(b2), Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    def apply_cswap(mut self, c: Int, a: Int, b: Int) raises:
        var s = _sorted3(c, a, b)
        var b0 = s[0]
        var b1 = s[1]
        var b2 = s[2]
        var cbit = 1 << c
        var abit = 1 << a
        var bbit = 1 << b
        var work = self.size() >> 3
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm b0, imm b1, imm b2, imm cbit, imm abit, imm bbit}:
                th_cswap[Self.dtype](g, work, re, im, b0, b1, b2, cbit, abit, bbit)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_cswap[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int32(b0), Int32(b1), Int32(b2), Int64(cbit), Int64(abit), Int64(bbit), Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    def apply_global_phase(mut self, phase: Float64) raises:
        if phase == 0.0:
            return
        var cr = Scalar[Self.dtype](cos(phase))
        var si = Scalar[Self.dtype](sin(phase))
        var work = self.size()
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm cr, imm si}:
                th_phase[Self.dtype](g, work, re, im, cr, si)

            emu_launch(f, work)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                self.ctx.value().enqueue_function[gk_phase[Self.dtype]](
                    self.dre.value(), self.dim.value(), cr, si, Int64(work),
                    grid_dim=grid_blocks(work), block_dim=TPB)
            else:
                raise Self._no_device()
        self.launches += 1

    # ------------------------------------------------------------ plans
    def apply_kop(mut self, op: KOp) raises:
        """One op of `qumo.sim.runner.build_plan` -> one kernel launch."""
        var k = op.kind
        if k == K.ONE:
            if _is_x_matrix(op.m):
                self.apply_x(op.a)
            else:
                self.apply_1q(op.m, op.a)
        elif k == K.CX:
            self.apply_cx(op.a, op.b)
        elif k == K.CPHASE:
            self.apply_cphase(op.f0, op.f1, op.a, op.b)
        elif k == K.C1:
            self.apply_controlled_1q(op.m, op.a, op.b)
        elif k == K.RZZ:
            self._rzz(op.f0, op.f1, op.a, op.b)
        elif k == K.SWAP:
            self.apply_swap(op.a, op.b)
        elif k == K.TWO:
            self.apply_2q(op.m4, op.a, op.b)
        else:
            var g = op.inst
            if g.kind == Op.CCX:
                self.apply_ccx(g.q0, g.q1, g.q2)
            elif g.kind == Op.CCZ:
                self.apply_ccz(g.q0, g.q1, g.q2)
            elif g.kind == Op.CSWAP:
                self.apply_cswap(g.q0, g.q1, g.q2)
            else:
                raise Error("QM-E475: the GPU executor has no kernel for opcode " + String(g.kind) + ". Hint: run this circuit on the CPU state vector (qumo.sim.runner).")

    def execute_plan(mut self, plan: List[KOp]) raises:
        """Enqueue every op of a fused plan (asynchronous, no host syncs)."""
        for op in plan:
            self.apply_kop(op)

    def evolve(mut self, circuit: Circuit, params: List[Float64] = List[Float64]()) raises:
        """Apply the unitary part of `circuit` (fused plan + global phase) to
        the current state. Measurements/resets are skipped (like
        `qumo.sim.runner.evolve_into`); conditioned gates raise QM-E401."""
        circuit.validate()
        if circuit.num_qubits > self.num_qubits:
            raise Error("QM-E474: circuit has " + String(circuit.num_qubits) + " qubits but the GPU state has " + String(self.num_qubits) + ". Hint: allocate GpuStateVector(circuit.num_qubits).")
        if len(params) != circuit.num_params():
            raise Error("QM-E101: circuit has " + String(circuit.num_params()) + " parameters " + circuit.param_names_str() + " but " + String(len(params)) + " values were given.")
        var plan = build_plan(circuit, params)
        self.execute_plan(plan)
        self.apply_global_phase(circuit.global_phase)

    # ------------------------------------------------------------ readout
    def to_host(self) raises -> StateVector:
        """Copy the amplitudes back (converted to float64)."""
        var sv = StateVector(self.num_qubits)
        var n = self.size()
        if self.emulated:
            for i in range(n):
                sv.re[i] = Float64(self.hre[i])
                sv.im[i] = Float64(self.him[i])
            return sv^
        comptime if device_kernels_enabled[Self.dtype]():
            ref ctx = self.ctx.value()
            var hr = ctx.enqueue_create_host_buffer[Self.dtype](n)
            var hi = ctx.enqueue_create_host_buffer[Self.dtype](n)
            self.dre.value().enqueue_copy_to(hr)
            self.dim.value().enqueue_copy_to(hi)
            ctx.synchronize()
            for i in range(n):
                sv.re[i] = Float64(hr[i])
                sv.im[i] = Float64(hi[i])
            return sv^
        else:
            raise Self._no_device()

    def probabilities(self) raises -> List[Float64]:
        """|psi_i|^2 for every basis state (computed on the device)."""
        var n = self.size()
        var out = List[Float64](length=n, fill=0.0)
        if self.emulated:
            var tmp = List[Scalar[Self.dtype]](length=n, fill=0)
            var re = self._re()
            var im = self._im()
            var dst = _mptr[Self.dtype](tmp)

            def f(g: Int) {imm re, imm im, imm dst, imm n}:
                th_prob[Self.dtype](g, n, re, im, dst)

            emu_launch(f, n)
            for i in range(n):
                out[i] = Float64(tmp[i])
            return out^
        comptime if device_kernels_enabled[Self.dtype]():
            ref ctx = self.ctx.value()
            var dp = ctx.enqueue_create_buffer[Self.dtype](n)
            ctx.enqueue_function[gk_prob[Self.dtype]](
                self.dre.value(), self.dim.value(), dp, Int64(n),
                grid_dim=grid_blocks(n), block_dim=TPB)
            var hp = ctx.enqueue_create_host_buffer[Self.dtype](n)
            dp.enqueue_copy_to(hp)
            ctx.synchronize()
            for i in range(n):
                out[i] = Float64(hp[i])
            return out^
        else:
            raise Self._no_device()

    def pauli_sums(self, xs: List[Int], zs: List[Int]) raises -> List[Float64]:
        """For each (x, z) mask pair: sum_i conj(psi[i^x]) psi[i] (-1)^{|i&z|}
        as [re0, im0, re1, im1, ...] (fp64 sums of the per-block partials).

        One reduction launch per term, one device->host copy for all terms.
        """
        var n = self.size()
        var nt = len(xs)
        var nb = reduction_blocks(n)
        var nthreads = nb * TPB
        var out = List[Float64](length=2 * nt, fill=0.0)
        if nt == 0:
            return out^
        var hpr = List[Float64](length=nt * nb, fill=0.0)
        var hpi = List[Float64](length=nt * nb, fill=0.0)
        if self.emulated:
            var pre = List[Scalar[Self.dtype]](length=nt * nb, fill=0)
            var pim = List[Scalar[Self.dtype]](length=nt * nb, fill=0)
            var ppre = _mptr[Self.dtype](pre)
            var ppim = _mptr[Self.dtype](pim)
            var re = self._re()
            var im = self._im()
            for k in range(nt):
                var x = xs[k]
                var z = zs[k]

                def load(b: Int, tid: Int) {imm re, imm im, imm x, imm z, imm n, imm nthreads} -> SIMD[Self.dtype, 2]:
                    return th_expect_load[Self.dtype](b * TPB + tid, nthreads, n, re, im, x, z)

                emu_block_reduce2[Self.dtype](load, nb, ppre, ppim, k * nb)
            for i in range(nt * nb):
                hpr[i] = Float64(pre[i])
                hpi[i] = Float64(pim[i])
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                ref ctx = self.ctx.value()
                var dpr = ctx.enqueue_create_buffer[Self.dtype](nt * nb)
                var dpi = ctx.enqueue_create_buffer[Self.dtype](nt * nb)
                for k in range(nt):
                    ctx.enqueue_function[gk_expect[Self.dtype]](
                        self.dre.value(), self.dim.value(), Int64(xs[k]), Int64(zs[k]), Int64(n), Int64(nthreads),
                        dpr, dpi, Int64(k * nb),
                        grid_dim=nb, block_dim=TPB)
                var hr = ctx.enqueue_create_host_buffer[Self.dtype](nt * nb)
                var hi = ctx.enqueue_create_host_buffer[Self.dtype](nt * nb)
                dpr.enqueue_copy_to(hr)
                dpi.enqueue_copy_to(hi)
                ctx.synchronize()
                for i in range(nt * nb):
                    hpr[i] = Float64(hr[i])
                    hpi[i] = Float64(hi[i])
            else:
                raise Self._no_device()
        for k in range(nt):
            var sr = 0.0
            var si = 0.0
            for b in range(nb):
                sr += hpr[k * nb + b]
                si += hpi[k * nb + b]
            out[2 * k] = sr
            out[2 * k + 1] = si
        return out^

    def norm_sq(self) raises -> Float64:
        """<psi|psi> by the same GPU tree reduction (identity Pauli term)."""
        var xs: List[Int] = [0]
        var zs: List[Int] = [0]
        return self.pauli_sums(xs, zs)[0]

    def expectation(self, h: PauliSum) raises -> Float64:
        """<psi|H|psi> for a Pauli sum (GPU reductions, fp64 host combine)."""
        if h.num_qubits() > self.num_qubits:
            raise Error("QM-E405: observable acts on qubit " + String(h.num_qubits() - 1) + " but the state has " + String(self.num_qubits) + " qubits. Hint: check Pauli indices (0-based, qubit 0 = least significant).")
        var xs = List[Int]()
        var zs = List[Int]()
        for t in h.terms:
            xs.append(Int(t.x))
            zs.append(Int(t.z))
        var s = self.pauli_sums(xs, zs)
        var total = 0.0
        for k in range(len(h.terms)):
            ref t = h.terms[k]
            var sr = s[2 * k]
            var si = s[2 * k + 1]
            var ny = Int(pop_count(t.x & t.z)) & 3
            # multiply by i^ny, keep the real part (same convention as StateVector)
            var v = sr
            if ny == 1:
                v = -si
            elif ny == 2:
                v = -sr
            elif ny == 3:
                v = si
            total += t.coeff * v
        return total

    def segment_sums(self, seg: Int) raises -> List[Float64]:
        """Probability mass of each `seg`-amplitude segment (one block each)."""
        var n = self.size()
        var nseg = n // seg
        var out = List[Float64](length=nseg, fill=0.0)
        if self.emulated:
            var tmp = List[Scalar[Self.dtype]](length=nseg, fill=0)
            var re = self._re()
            var im = self._im()

            def load(b: Int, tid: Int) {imm re, imm im, imm seg} -> Scalar[Self.dtype]:
                return th_seg_load[Self.dtype](tid, b * seg, seg, re, im)

            emu_block_reduce1[Self.dtype](load, nseg, _mptr[Self.dtype](tmp))
            for i in range(nseg):
                out[i] = Float64(tmp[i])
            return out^
        comptime if device_kernels_enabled[Self.dtype]():
            ref ctx = self.ctx.value()
            var ds = ctx.enqueue_create_buffer[Self.dtype](nseg)
            ctx.enqueue_function[gk_segsum[Self.dtype]](
                self.dre.value(), self.dim.value(), Int64(seg), ds,
                grid_dim=nseg, block_dim=TPB)
            var hs = ctx.enqueue_create_host_buffer[Self.dtype](nseg)
            ds.enqueue_copy_to(hs)
            ctx.synchronize()
            for i in range(nseg):
                out[i] = Float64(hs[i])
            return out^
        else:
            raise Self._no_device()

    def sample(self, shots: Int, seed: UInt64) raises -> List[Int]:
        var rng = Rng(seed)
        return self.sample_rng(shots, rng)

    def sample_rng(self, shots: Int, mut rng: Rng) raises -> List[Int]:
        """Basis-state indices of `shots` samples (two-level inverse CDF).

        Same RNG use as `StateVector.sample`: `shots` sorted uniforms, then a
        Fisher-Yates shuffle, so results are reproducible for a seed.
        """
        var out = List[Int](capacity=max(shots, 0))
        if shots <= 0:
            return out^
        var us = List[Float64](capacity=shots)
        for _ in range(shots):
            us.append(rng.uniform())
        sort(us)
        var n = self.size()
        var seg = min(SAMPLE_SEG, n)
        var nseg = n // seg
        var sums = self.segment_sums(seg)
        var total = 0.0
        for s in range(nseg):
            total += sums[s]
        if not (total > 0.0):
            raise Error("QM-E473: cannot sample a state with zero norm. Hint: check the circuit / initial state.")
        # map sorted targets to (segment, residual) with an fp64 prefix sum
        var segs = List[Int64](capacity=shots)
        var resid = List[Float64](capacity=shots)
        var s = 0
        var cum = 0.0
        for k in range(shots):
            var target = us[k] * total
            while s < nseg - 1 and cum + sums[s] <= target:
                cum += sums[s]
                s += 1
            segs.append(Int64(s))
            resid.append(target - cum)
        var idx = List[Int64](length=shots, fill=0)
        if self.emulated:
            var res_dt = List[Scalar[Self.dtype]](length=shots, fill=0)
            for k in range(shots):
                res_dt[k] = Scalar[Self.dtype](resid[k])
            var re = self._re()
            var im = self._im()
            var psegs = segs.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
            var pres = _mptr[Self.dtype](res_dt)
            var pidx = idx.unsafe_ptr().as_unsafe_any_origin()

            def f(g: Int) {imm re, imm im, imm seg, imm psegs, imm pres, imm pidx, imm shots}:
                th_sample_seg[Self.dtype](g, shots, re, im, seg, psegs, pres, pidx)

            emu_launch(f, shots)
            _ = len(segs)
            _ = len(res_dt)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                ref ctx = self.ctx.value()
                var dsegs = ctx.enqueue_create_buffer[DType.int64](shots)
                var dres = ctx.enqueue_create_buffer[Self.dtype](shots)
                var douts = ctx.enqueue_create_buffer[DType.int64](shots)
                with dsegs.map_to_host() as h:
                    for k in range(shots):
                        h[k] = segs[k]
                with dres.map_to_host() as h:
                    for k in range(shots):
                        h[k] = Scalar[Self.dtype](resid[k])
                ctx.enqueue_function[gk_sample_seg[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int64(seg), dsegs, dres, douts, Int64(shots),
                    grid_dim=grid_blocks(shots), block_dim=TPB)
                var ho = ctx.enqueue_create_host_buffer[DType.int64](shots)
                douts.enqueue_copy_to(ho)
                ctx.synchronize()
                for k in range(shots):
                    idx[k] = ho[k]
            else:
                raise Self._no_device()
        for k in range(shots):
            out.append(Int(idx[k]))
        # shuffle outcomes so memory order is not sorted (same as StateVector.sample)
        var k = shots - 1
        while k > 0:
            var j = rng.below(k + 1)
            var tmp = out[k]
            out[k] = out[j]
            out[j] = tmp
            k -= 1
        return out^


# ---------------------------------------------------------------------------
# Circuit-level entry points (mirror qumo.sim.runner)
# ---------------------------------------------------------------------------
def _check_params(circuit: Circuit, params: List[Float64]) raises:
    if len(params) != circuit.num_params():
        raise Error("QM-E101: circuit has " + String(circuit.num_params()) + " parameters " + circuit.param_names_str() + " but " + String(len(params)) + " values were given.")


def evolve_gpu[dt: DType = GPU_DEFAULT_DTYPE](circuit: Circuit, params: List[Float64] = List[Float64](),
                                              emulate: Bool = False) raises -> GpuStateVector[dt]:
    """Final state of `circuit` from |0...0> on the GPU (or its emulator)."""
    circuit.validate()
    _check_params(circuit, params)
    var sv = GpuStateVector[dt](circuit.num_qubits, emulate)
    sv.evolve(circuit, params)
    return sv^


def expectation_gpu[dt: DType = GPU_DEFAULT_DTYPE](circuit: Circuit, observable: PauliSum,
                                                   params: List[Float64] = List[Float64](),
                                                   emulate: Bool = False) raises -> Float64:
    var sv = evolve_gpu[dt](circuit, params, emulate)
    return sv.expectation(observable)


def sample_gpu[dt: DType = GPU_DEFAULT_DTYPE](circuit: Circuit, shots: Int, seed: UInt64 = 1234,
                                              params: List[Float64] = List[Float64](),
                                              emulate: Bool = False) raises -> SampleResult:
    """Static sampling on the GPU (same contract as runner.sample's static path).

    SampleResult.method = 'gpu:statevector:fp32|fp64' ('gpu-emulator:...' when
    emulated). Dynamic or noisy circuits raise QM-E471 (use runner.sample).
    """
    circuit.validate()
    check_statevector_size(circuit.num_qubits)
    _check_params(circuit, params)
    if circuit.num_clbits > 62:
        raise Error("QM-E406: the sampler packs classical registers into 62 bits; circuit has " + String(circuit.num_clbits) + " clbits. Hint: measure fewer bits or split the register.")
    if circuit.num_measurements == 0:
        raise Error("QM-E402: circuit has no measurements. Hint: call measure_all() or use evolve()/expectation().")
    if circuit.is_dynamic() or circuit.has_noise():
        raise Error("QM-E471: the GPU sampler runs static circuits only (terminal measurements, no noise). Hint: use qumo.exec.dispatch.sample_auto, which falls back to CPU trajectories.")
    var rng = Rng(seed)
    var sv = GpuStateVector[dt](circuit.num_qubits, emulate)
    sv.evolve(circuit, params)
    var qs = List[Int]()
    var cs = List[Int]()
    for g in circuit.ops:
        if g.kind == Op.MEASURE:
            qs.append(g.q0)
            cs.append(g.clbit)
    var outs = sv.sample_rng(shots, rng)
    var res = SampleResult(circuit.num_clbits, shots, sv.method_name())
    for o in outs:
        var creg = 0
        for j in range(len(qs)):
            if (o >> qs[j]) & 1 == 1:
                creg |= 1 << cs[j]
        res.record(creg)
    return res^
