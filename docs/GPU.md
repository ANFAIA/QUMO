# QUMO on GPUs (`qumo.gpu`)

QUMO runs its state-vector simulator and its QEC Pauli-frame sampler on
NVIDIA, AMD and Apple GPUs through MAX's vendor-neutral GPU API
(`max.gpu`, `DeviceContext`). The same Mojo kernel source compiles to PTX
(NVIDIA), AMDGCN (AMD) and Metal (Apple). There is no CUDA, HIP or Metal
code in the repository, and no vendor gets a code path the others lack.

> **Status (honest).**
> * **Measured on real hardware: Apple M5 (10-core GPU, Metal, fp32 only).**
>   `make test-gpu` passes on the device (it fails, rather than passing on
>   the emulation suites alone, when no device is usable): every gate kind,
>   random circuits up to 24 qubits, probabilities, expectation values and sampling against the
>   CPU simulator, the Pauli-frame sampler against the CPU frame simulator
>   (bit for bit against the emulator), plus the device suite again with the
>   dynamic shared-memory layout NVIDIA / AMD use. All performance numbers in
>   this file come from that machine (Low Power Mode on, shared machine).
> * **NVIDIA (`sm_75`, `sm_80`, `sm_90`) and AMD (`gfx942`): cross-compiled
>   only.** Every kernel, fp32 and fp64, compiles for all of them
>   (`make gpu-check`). The only NVIDIA numbers that exist (Colab Tesla T4,
>   commit 1750a81, `bench/results/cloud/`) come from the *previous* engine,
>   which launched one kernel per plan op (378 launches at 26 qubits); the
>   engine described here has never run on NVIDIA or AMD. `bench_gpu N R
>   probe` and the Colab notebooks (`examples/colab/`) are how to measure it
>   there ([what to confirm](#what-a-colab-run-must-confirm)).
> * The development machine is shared with other jobs (load averages 3-15
>   during the runs below), so timings are noisy: every comparison is
>   interleaved (A, B, A, B, ...) and reports medians; absolute numbers moved
>   by up to 20% between sessions.

## Modules

| File | Role |
|---|---|
| `qumo/gpu/device.mojo` | `gpu_compiled()`, `gpu_available()`, `gpu_info()` / `query_gpu()` (`GpuInfo`, JSON), `GpuProfile` (per-device tuning of the tile engine, [below](#device-profile-gpuprofile)), precision policy (`GPU_DEFAULT_DTYPE`), dispatch threshold `GPU_MIN_QUBITS`, error constructors |
| `qumo/gpu/tiles.mojo` | Host tile scheduler: packs the fused gate plan into register-blocked tile runs (`TileProgram`, `build_tile_program`) |
| `qumo/gpu/kernels.mojo` | Device-agnostic **thread phases** `th_*` plus thin GPU kernels `gk_*` (`gk_tile`, `gk_prob`, `gk_expect`, `gk_segsum`, `gk_sample`) |
| `qumo/gpu/emulate.mojo` | Host emulator: runs every kernel's thread phases over the real launch grid, barrier by barrier |
| `qumo/gpu/statevector_gpu.mojo` | `GpuStateVector[dtype]`, plus `evolve_gpu`, `expectation_gpu`, `sample_gpu` |
| `qumo/gpu/frame_gpu.mojo` | `GpuFrameSampler`: QEC Pauli-frame sampling, one thread per 64-shot word |

Integration (all GPU calls gated, see below):

* `qumo/exec/dispatch.mojo`: `choose_method` returns `"gpu"` for static,
  noiseless, non-Clifford (or parametric) circuits with at least
  `GPU_MIN_QUBITS` qubits (16 on Apple, 20 elsewhere; see
  [Dispatch threshold](#dispatch-threshold)), when the build has GPU support
  and a device is usable. `sample_auto` runs them on the GPU and falls back
  to the CPU on any GPU error.
  `sample_on(c, shots, seed, params, device, note, precision="auto")` takes
  `device = "auto" | "cpu" | "gpu"` and the GPU precision selector
  `"auto" | "fp32" | "fp64"` ([Precision policy](#precision-policy)). With
  `"gpu"` the circuit is routed by
  `gpu_sample_route`: Clifford circuits without feed-forward go to GPU Pauli
  frames, other static noiseless circuits to the GPU state vector. Anything
  else runs on the CPU and `note` says exactly why: no GPU build, no usable
  device, more than 62 classical bits, a Clifford circuit with feed-forward
  (CPU stabilizer simulator), or a dynamic / noisy non-Clifford circuit.
  `SampleResult.method` reports what ran: `gpu:statevector:fp32`,
  `gpu:statevector:fp64`, `gpu:pauli_frame` (or `gpu-emulator:*`).
* `qumo/exec/orchestrator.mojo`: jobs that HEFT places on a
  `Device.gpu(name, max_qubits, precision="auto")` run on the GPU when one is
  available (estimate jobs on the GPU state vector, sample jobs routed by
  `gpu_sample_route`), in the device's precision unless the job sets its own
  (`add_sample / add_estimate(..., precision=...)`). Otherwise they run on
  the CPU and `JobResult.backend` ends with `(gpu fallback: <reason>)`.
* MCP / CLI tools (`qumo/exec/tools.mojo`): `qumo_gpu_info` and
  `qumo_run_device` (`qumo_run_qasm` plus `"device": "auto"|"cpu"|"gpu"` and
  `"precision": "auto"|"fp32"|"fp64"`, warning `QM-W020` with the reason on
  fallback, `QM-W021` when a requested precision did not apply; the result
  has `precision_used`). CLI: `qumo run x.qasm --device gpu --precision fp64`
  (`--device` / `--precision` route through `qumo_run_device`).
* Python (`python/qumo_py.mojo`): `gpu_info() -> dict` and
  `sample_device(circ, shots, seed, device, precision="auto") -> (counts,
  method, note)` (`precision` is keyword-only).

## Compile-time gating

The only way QUMO touches a GPU toolchain is `-D QUMO_GPU`:

```mojo
from std.sys.defines import is_defined
comptime if is_defined["QUMO_GPU"]():
    ctx.enqueue_function[gk_tile[dt, True, False]](...)   # compiled only in GPU builds
```

* **Default builds** (`make test`, `make cli`, `make py`) contain no device
  code and run on any machine. `gpu_compiled()` is `False`,
  `gpu_available()` is `False` without creating a `DeviceContext`, and device
  mode raises `QM-E470`. The emulator still runs everywhere.
* **GPU builds** (`make test-gpu`, `cli-gpu`, `py-gpu`, `bench-gpu`) compile
  the kernels for the build's target accelerator: the host GPU by default,
  or any GPU with `--target-accelerator=...`.
* `make gpu-check` cross-compiles every kernel for `GPU_TARGETS` = `sm_75`
  (T4), `sm_80`, `sm_90` (PTX), `gfx942` (AMDGCN) and `metal:4` (Metal,
  including the Metal IR verifier). It needs no GPU and fails if any target
  fails or emits no kernel (broken gating). Last run: PASS for all three
  files on all five targets (`tests/test_gpu.mojo` 19 kernels for NVIDIA /
  AMD, fp32 and fp64, 10 for `metal:4`, fp32 only; `bench/bench_gpu.mojo`
  15 / 10).
* `make test-gpu` runs the device suites (`QUMO_GPU_REQUIRE_DEVICE=1`: a
  build or machine without a usable device FAILS instead of reporting PASS
  for the emulation part; `make test-gpu QUMO_GPU_REQUIRE_DEVICE=0` allows
  the skip), then `make test-gpu-regbits`
  (`tests/test_gpu.mojo` in emulation with 8 and with 16 amplitudes per
  thread, so both register-blocking widths stay covered on every machine)
  and, on macOS, the device suite once more with
  `-D QUMO_GPU_TILE_SMEM_DYNAMIC=1` (the shared-memory layout of NVIDIA /
  AMD, exercised on Metal).

Compile-time knobs (`-D NAME=value`, defaults in brackets):
`QUMO_GPU_MIN_QUBITS` [16 Apple / 20 other], `QUMO_GPU_REG_BITS` (log2
amplitudes per thread) [4 Apple / 3 other], `QUMO_GPU_TILE_SMEM_DYNAMIC`
[0 Apple / 1 other], `QUMO_GPU_TILE_LAUNCH_BOUND` (max threads per block
the tile kernel is compiled for; 0 = none) [256 AMD / 0 other],
`QUMO_GPU_TILE_LOG2_BYTES` [14], `QUMO_GPU_LOG_TPB` [8], `QUMO_GPU_FP64`
(fp64 as the build default on NVIDIA / AMD; accepted and ignored on Apple;
[Precision policy](#precision-policy)), `QUMO_GPU_FP32` (the old fp32 opt-in:
accepted, no effect, fp32 is the default). Runtime knobs (environment
variables, no rebuild): `QUMO_GPU_PRECISION` (fp32 / fp64: what precision
"auto" runs) and the tuning knobs in [Device profile](#device-profile-gpuprofile).

## Enabling the GPU

| Platform | Requirement | Then |
|---|---|---|
| macOS, Apple silicon | Xcode (not only CommandLineTools) **and** `xcodebuild -downloadComponent MetalToolchain` (`xcrun metal --version` must work) | `make test-gpu`, `make bench-gpu` |
| Linux, NVIDIA | NVIDIA driver with CUDA support for the GPU (MAX bundles the compiler) | same |
| Linux, AMD | ROCm-supported GPU and driver (MI200/MI300 class, recent Radeon) | same |

Without the Metal toolchain, launching any kernel on a Mac fails with "Metal
Compiler failed to compile metallib". Check a build with `qumo tool
qumo_gpu_info '{}'` (CLI built with `make cli-gpu`) or
`qumo_py.gpu_info()`.

## Precision policy

**fp32 by default on every GPU (NVIDIA, AMD, Apple); fp64 on request where
the hardware has it.** This is CUDA-Q's choice: its `nvidia` target runs fp32
on every GPU unless fp64 is asked for (`option="fp64"`;
`runtime/cudaq/platform/default/nvidia.yml` L21-32 at tag 0.16.0). The
reasons:

* fp32 is the fast path on every GPU and halves the memory (28 qubits = 2 GiB
  in fp32, 4 GiB in fp64). On consumer and inference GPUs fp64 runs at a
  fraction of the fp32 rate: 1/32 on the T4 (`sm_75`), 1/64 on the L4
  (`sm_89`), RTX 30/40/50 and RTX PRO 6000 (`sm_86` / `sm_89` / `sm_120`), and
  on AMD RDNA (gfx10-12) (`GpuProfile.weak_fp64()`). The CUDA-Q T4 numbers
  show it: layered circuit at 26 qubits 0.17 s in fp32, 1.45 s in fp64
  ([below](#comparison-with-cuda-q)).
* The same default as CUDA-Q makes a default-vs-default comparison an fp32
  one, and on Apple GPUs (no fp64 at all) nothing changes.
* fp32 is accurate enough for the benchmark circuits: on the M5 the fidelity
  F with the fp64 CPU state is |1 - F| <= 6e-7 up to 26 qubits, max
  amplitude error < 2e-5 for 200-gate random circuits, expectation values
  within 1e-4 at 24 qubits. Data-center GPUs with full-rate fp64 (A100,
  H100, MI300X) run fp64 at 1/2 of the fp32 rate: ask for it when 1e-12
  agreement with the CPU simulator matters.

| Vendor (target) | State vector | Pauli frames | Default precision | fp64 | Run on hardware |
|---|---|---|---|---|---|
| Apple (`metal:4`) | fp32 kernels | yes | **fp32** | **no** (Metal has no double): `precision: "fp64"` raises `QM-E472` | **yes: M5, all tests + benchmarks** |
| NVIDIA (`sm_75`, `sm_80`, `sm_90`, ...) | fp32 + fp64 kernels | yes | **fp32** | yes, on request (1/32-1/64 rate on T4 / L4 / RTX; 1/2 on A100 / H100) | no (cross-compiled only; the T4 numbers in `bench/results/cloud/` are the previous engine) |
| AMD (`gfx942`, ...) | fp32 + fp64 kernels | yes | **fp32** | yes, on request (full rate on MI300X; 1/32-1/64 on RDNA) | no (cross-compiled only) |

How to select fp64 (or fp32), from the widest scope to the narrowest:

| Scope | fp64 | Notes |
|---|---|---|
| Build | `-D QUMO_GPU -D QUMO_GPU_FP64` | Makes fp64 the build default (`GPU_DEFAULT_DTYPE`, `bench_gpu`'s precision). Ignored on Apple: fp32 stays, `qumo_gpu_info`'s `precision_note` and the profile JSON say why. `-D QUMO_GPU_FP32` is accepted (no effect). |
| Process | `QUMO_GPU_PRECISION=fp64` | What `"auto"` resolves to, without rebuilding. Invalid values raise `QM-E941` for explicit GPU requests; automatic dispatch then stays on the CPU. |
| Call | `precision: "fp64"` | `sample_on(..., precision=)`, `sample_auto(..., precision=)`, `Device.gpu(precision=)` / per job, MCP `qumo_run_device`, `qumo run --precision`, `qumo_py.sample_device(..., precision=)`. `"auto"` = the two rows above. |
| Code | `GpuStateVector[DType.float64]`, `sample_gpu_runtime(..., "fp64")` | `GpuStateVector[dtype = GPU_DEFAULT_DTYPE]` takes the precision as a comptime parameter; the `*_gpu_runtime` entry points select it at run time. |

* The precision applies to GPU state vectors only. CPU simulators are fp64;
  the stabilizer tableau and the Pauli-frame samplers (CPU and GPU) use no
  floating point, so `device: "gpu"` with a Clifford circuit runs GPU Pauli
  frames whatever the precision. `qumo_run_device` reports
  `precision_used` (`fp32` / `fp64` / `exact`) and warns `QM-W021` when a
  requested precision did not apply.
* `fp64` on an Apple GPU: explicit requests (`device: "gpu"`, orchestrator
  GPU devices) fail with `QM-E472`; automatic dispatch (`device: "auto"`)
  keeps such a circuit on the CPU, which is fp64.
* `SampleResult.method` / `JobResult.backend` name what ran:
  `gpu:statevector:fp32` or `gpu:statevector:fp64`. `qumo_gpu_info` reports
  `precision` (what `"auto"` runs now), `precision_build_default`,
  `precision_env`, `precisions` (selectable on this target) and
  `precision_note`.
* Both precisions are compiled into every NVIDIA / AMD GPU build
  (`GPU_FP64_RUNTIME`): run-time selection needs `GpuStateVector[float32]`
  and `[float64]`. The fp64 path is instantiated only under
  `comptime if GPU_FP64_SUPPORTED`, so Metal builds contain no
  double-precision code. Build cost: see [Build cost of run-time
  precision](#build-cost-of-run-time-precision).
* Host-side combination is always fp64: `to_host()` converts to fp64, and
  reduction partials and sampling CDFs over segments are summed in fp64.
* fp32 accuracy on the M5 (measured, current engine): fidelity F with the
  CPU fp64 state |1 - F| <= 6e-7 for the benchmark circuits up to 26 qubits,
  max amplitude error < 2e-5 for 200-gate random circuits; expectation
  values within 1e-4 at 24 qubits. A fused diagonal phase table entry is the
  product of the gates' coefficients as stored for the run (already rounded
  to fp32 in fp32), multiplied in fp64 and rounded once: at least as
  accurate as multiplying gate by gate in fp32.
* The Pauli-frame kernel uses no floating point at all, so it is identical
  on every vendor.

### Build cost of run-time precision

Run-time selection compiles both precisions into every NVIDIA / AMD GPU
build; before 2026-10-13 a default NVIDIA / AMD build compiled fp64 only, and
Metal builds compile fp32 only either way. Measured on the M5 (2026-10-13,
machine shared with other jobs, load average 8-45, so single numbers of the
same build moved by up to 3.6x): `mojo build -D QUMO_GPU` of `cli/main.mojo` (= `make cli-gpu`) and
`python/qumo_py.mojo` (= `make py-gpu`), cross-compiled for `sm_75` (T4), for
three trees: *before* (commit 4f51c52: fp64 only), *fp32 only* (this tree with
`GPU_FP64_RUNTIME = False`) and *after* (this tree: fp32 + fp64). Cold = an
empty Mojo cache (`XDG_CACHE_HOME` pointed at a fresh directory: a fresh
machine or Colab), 4 interleaved rounds (the last one in reverse order);
warm = the populated cache, 1 round.

| `sm_75` build | before (fp64) | fp32 only | after (fp32 + fp64) |
|---|---|---|---|
| CLI, cold, wall s: min / median | 61.8 / 159 | 62.3 / 94 | 57.5 / 79 |
| CLI, cold, CPU s (user + sys), 2 rounds | 83.0, 156.7 | 80.3, 73.0 | 85.9, 89.8 |
| CLI, warm, wall s | 45.1 | 41.9 | 42.8 |
| CLI binary, bytes | 3 050 232 | 3 052 136 | 3 593 144 (+18%) |
| Python extension, cold, wall s: min / median | 51.8 / 88 | 46.0 / 98 | 44.9 / 52 |
| Python extension, cold, CPU s, 2 rounds | 60.3, 97.2 | 54.7, 99.3 | 68.5, 63.6 |
| Python extension, warm, wall s | 27.6 | 34.3 | 27.2 |
| Python extension, bytes | 2 270 784 | 2 293 136 | 2 817 632 (+24%) |

Metal (`make cli-gpu` / `make py-gpu` on the M5, cold, one round): CLI 77.9 s
-> 77.6 s (CPU 89.1 -> 93.5 s), Python 48.5 -> 36.5 s (CPU 57.9 -> 49.5 s);
binaries +0.6% / +1.0% (no fp64 code on Metal). The fp64 instantiation costs
at most ~10-20% of CPU time against an fp32-only build (lowest CPU times: CLI
73.0 -> 85.9 s, Python 54.7 -> 63.6 s), inside the run-to-run noise in wall
time, and nowhere near doubling it, so it is **not** gated behind
`-D QUMO_GPU_FP64`: every NVIDIA / AMD GPU build can switch precision at run
time. The price is binary size (+0.5 MB per binary for `sm_75`). Not
measured: `gfx942` and other NVIDIA targets (`make gpu-check` compiles both
precisions for them).

Memory: `2^n * 8 B` (fp32) or `2^n * 16 B` (fp64); 28 qubits in fp32 = 2 GiB.
The hard cap is the CPU simulator's 32 qubits (`check_statevector_size`).

## State-vector engine

Storage is the CPU simulator's: split re/im arrays, little-endian qubit
order. The plan is the CPU runner's: `qumo.sim.runner.build_plan` fuses every
run of 1-qubit gates on a wire into one 2x2 (as qsim and Qiskit Aer do).

**Tile runs (gate blocking).** A *tile* is the set of 2^k amplitudes whose
indices differ only in k chosen qubits S: the `low` lowest qubits (so 2^low
consecutive threads touch 2^low consecutive amplitudes, coalesced for every
gate) plus k - low "high" qubits. `low` and k come from the
[device profile](#device-profile-gpuprofile): low = 4 on Apple GPUs and 5
elsewhere; k = 11 in fp32 and 10 in fp64 (16 KiB of amplitudes per block),
up to 2 more on GPUs whose shared memory allows it and whose compiled
kernel is known to launch the larger block. One thread block handles
one tile: it reads the tile once, applies a whole *run* of gates and writes
it back once (one *state sweep* per run). The host scheduler
(`qumo/gpu/tiles.mojo`) packs the plan into runs:

* Only the *targets* of non-diagonal gates must be in S. Controls and
  diagonal gates (RZ, P, S, T, CZ, CP, CRZ, RZZ, CCZ, global phase) test bits
  of the global amplitude index, so they join any run and use no tile slot.
* Greedy with commutation: scan the remaining gates; take a gate if it
  touches no blocked qubit and its new targets fit in S, else block all its
  qubits (a later gate on a blocked qubit must stay after it). Gates on
  disjoint qubits commute, so this preserves the circuit's semantics.
* The initial basis state is written by the first run's load (`reset` and
  `set_basis` are lazy), the global phase is one more diagonal op in the last
  run, so `GpuStateVector(n)` + `evolve` costs exactly one state sweep per
  run.

Sweeps of the benchmark circuits on the M5 (low = 4, k = 11): at 26 qubits
the layered circuit (377 plan ops) is 14 runs (17 with low = 5), the QFT
(364 plan ops) 7, the random-pairs circuit (390 ops) 15 (16); at 24 qubits
12, 6 and 14. The sweep count depends on the precision and tile size
(host counts, 8 amplitudes per thread as on NVIDIA / AMD, layered at 26
qubits): fp32 k = 11 17 (low 5) / 14 (low 4); fp64 k = 10 20, 24 for the
random circuit; k = 11 in fp64 17;
k = 12 12; k = 13 11.

**Register blocking inside a tile.** Each thread keeps 2^REG_BITS
amplitudes in registers: 16 on Apple GPUs, 8 elsewhere (block = 2^(k -
REG_BITS) threads). A gate whose targets are in the thread's current
*register set* runs in registers: no shared memory, no barrier. The host
cuts each run into *segments* with a register set each (greedy with
lookahead); changing the register set is one transpose through shared
memory. The tile is loaded and stored with a map in which thread bits
0..low-1 are local bits 0..low-1 (coalesced): the natural map, or directly
the run's first / last register set when none of its qubits is below `low`
(no transpose at either end of the run: 93 -> 75 transposes for layered at
26 qubits with 8 amplitudes per thread, 55 with 16).

**Ops in registers.** `G1` (complex 2x2, optional controls), `G1R` (real
2x2: H, RY and real products, 8 instead of 16 multiply-adds per pair),
`XC` (X / CX / CCX as register swaps, no flops), `SW` (SWAP / CSWAP), `G2`
(dense 4x4), `DG` (one diagonal gate) and `DGT` (a fused group of diagonal
gates, below). Each op is one 16-byte *device record* read with one vector
load from shared memory, plus its matrix as 16-byte vector loads. The host
(`TileProgram.finalize`) knows each op's register set and splits every
control, parity and phase-table qubit set into the part outside the register
set, tested once per op on register 0's global index `g0`, and per-register
bits: the op loop keeps no per-register global indices live (16 fewer
registers per thread at 16 amplitudes per thread) and a diagonal costs one
`pop_count` per op instead of one per register.

**Diagonal fusion (`DGT`).** Consecutive diagonal gates of a run (after
moving diagonals later past ops whose targets they do not touch, which
commute) on at most `diag_width` = 6 qubits become one phase table of 2^w
complex values, the products computed in fp64 on the host (from the
coefficients as stored for the run: fp32-rounded in fp32) and rounded once.
Per amplitude: one table gather and one complex multiply per group instead
of one complex multiply per gate (cuStateVec / CUDA-Q fuse diagonal gates the
same way, [below](#comparison-with-cuda-q)). The table qubits outside the
register set come first in the table index (gathered from `g0`), the
register qubits after (one OR per register). The width is the widest <= 6
whose tables all fit the run's matrix cache (2048 values), else width-3
tables while they fit. A 26-qubit QFT: 364 plan ops -> 127 tile ops (24
qubits: 312 -> 110).

**Optional: split complex 1q gates** (`GpuProfile.split_1q`, off by
default). A complex 1q gate of the form U = D . R, R real and D diagonal
(each row's entries share a phase: RZ . RY, RZ . H and their products,
`tiles.split_1q_ok`), runs as G1R(R) followed by diag(D), which joins the
run's phase tables: 4 instead of 8 multiply-adds per amplitude for the
2x2, one more (fused) op. Generic U3 gates stay dense: they would need a
second diagonal before R, and two diagonals per gate cost more than the
dense gate (host count, fp64 at 26 qubits, counting G1 = 8, G1R = 4 and a
complex multiply = 4 multiply-adds per amplitude: two-sided splitting
gave layered -22% but random +17%). The scheduler counts exactly what a
split emits (it used to reserve 3 ops per 1q gate, which cut runs early:
27 instead of 22 runs on a 17-qubit diagonal-heavy test circuit). Exact
(fp64 emulation within 1e-12 of the CPU). On the M5 (fp32) it is slower
(layered 26 qubits 1.09x, ops 377 -> 509; random unchanged, its U3s stay
dense), so it is off; it is meant for GPUs whose fp64 multiply-adds bound
the run (T4: 1/32 rate) on RY / RZ-type circuits, to be decided by a
device run (`QUMO_GPU_FP64_LEAN=1` vs `0`).

**Kernel details.**

* Gate operands come from device tables uploaded once per flush (op
  records, matrices, register-set segments), never from vector kernel
  arguments: Apple's Metal back end raises an internal compiler error on
  SIMD-vector kernel parameters (found on the M5 with MAX 26.6: the first
  device run of the previous per-gate kernels failed this way for every 1q /
  controlled / 4x4 kernel). Each run's op records, matrices and *segment
  records* (2 x Int32 each: register set + flags, op range) are copied to
  shared memory at kernel start; re-reading the segment records from device
  memory after every transpose barrier used to cost ~0.5 ms per transpose at
  24 qubits on the M5.
* Shared memory: one dynamic region per launch (`external_memory`, sized by
  the host per run: tile + that run's ops / matrices / segments, typically
  1.5-3 KiB of cache instead of the 9 KiB a static worst case reserves, so
  more blocks fit per multiprocessor and k can grow on GPUs with more shared
  memory), or static arrays sized for the default tile on Apple GPUs
  (`TILE_SMEM_DYNAMIC`: the dynamic layout runs correctly on the M5 and was
  measured 1.00-1.06x in one session, neutral in an independent one;
  Apple's 32 KiB per threadgroup admits no larger tile anyway). On NVIDIA
  GPUs with a shared-memory opt-in above 48 KiB, every load of the tile
  kernel carries one constant `MAX_DYNAMIC_SHARED_SIZE_BYTES` = opt-in - 1
  KiB (`GpuProfile.tile_smem_optin`, the value MAX itself derives), not
  each flush's largest request: the attribute can then never differ
  between loads of the same function in a process.
* Block size and registers: a tile of 2^k amplitudes is a block of
  2^(k - REG_BITS) threads. Blocks of up to `TILE_SAFE_THREADS` = 256
  threads launch whatever the kernel's register count (NVIDIA: 255
  registers x 256 threads fit the 64 Ki register file). Larger blocks are
  used only when the compiled kernels report that they launch: on NVIDIA,
  when the shared memory would allow a larger tile, `GpuStateVector` loads
  the four tile-kernel variants and reads `MAX_THREADS_PER_BLOCK`,
  `NUM_REGS`, `LOCAL_SIZE_BYTES` and the occupancy calculator (>= 2
  resident blocks of a typical run per multiprocessor), and the profile
  keeps those limits (`tile_kernel_*` in its JSON). Without that report
  (Metal, a driver error) tiles stay at <= 256 threads. The NVIDIA kernel
  has no launch bound (`.maxntid 1024` would cap it at 64 registers per
  thread).
* AMD: the tile kernel is compiled with a launch bound of 256 threads
  (`gk_tile_lb`, `QUMO_GPU_TILE_LAUNCH_BOUND`); under the default bound of
  1024 the gfx942 compiler capped it at 128 VGPRs and six of the eight
  variants spilled to scratch (44-304 B per thread). With the bound
  (cross-compiled `.amdgcn` metadata, not run): 100-194 VGPRs, no VGPR
  spills, no scratch. The trade-off is unmeasured: the heaviest variants
  (~190 VGPRs) fit 2 wave64s per SIMD, i.e. 2 workgroups of 256 threads per
  CU, where shared memory alone would allow ~3; `-D
  QUMO_GPU_TILE_LAUNCH_BOUND=0` builds the previous, unbounded kernel for
  an A/B on the device. `gk_tile_lb` calls `gk_tile` (inlined, no call in
  the ISA): moving the body into a shared `@always_inline` helper instead
  made the Metal kernel 1.4-2.3x slower on the M5 (measured), so the
  NVIDIA / Apple kernel is unchanged.
* Four specialisations of `gk_tile` per precision: `FULL` (2^REG_BITS
  registers per thread, guards compiled out) and `HEAVY` (only runs with
  SWAP / dense 4x4 get that code; lighter runs keep fewer live registers).
* 32-bit index arithmetic in the kernels (amplitude indices < 2^32 for the
  <= 32 qubits QUMO allows); GPUs emulate 64-bit integer ALU work.
* Launch sizes stay below 2^31 threads at 32 qubits (AMD's HSA grid limit
  is 2^32 work items): tile launches have 2^(n-k) blocks, elementwise kernels
  (`gk_prob`, `gk_init`) use a capped grid (2^16 blocks) with grid-stride
  loops.
* Launches are asynchronous on one in-order stream; the host synchronises
  only when it reads results. Kernel parameters are DevicePassable only
  (pointers and fixed-width scalars).

**Expectation values.** Terms are grouped by X mask; one `gk_expect`
launch reduces up to 8 terms of a group in a single pass over the state
(register accumulators, shared-memory tree per term, <= 1024 partials per
term), one device-to-host copy for all terms, fp64 summation on the host,
then the term's `i^{|x&z|}` factor. Fixed order, no atomics: reproducible run
to run.

**Sampling** never copies the state. `gk_segsum` reduces the probability of
each segment (4096 amplitudes in fp32, 2048 in fp64); the host builds the
fp64 prefix over segments and maps the sorted QUMO-RNG uniforms to (segment,
residual); `gk_sample` runs one block per *occupied* segment: coalesced load
of the segment, block scan into a shared-memory CDF, binary search per shot.
The RNG stream use is the CPU's (sorted uniforms, then the same Fisher-Yates
shuffle), so the fp64 path reproduces `qumo.sim.runner.sample` for a seed
(tested in emulation). A zero-probability outcome is never returned.

## Device profile (`GpuProfile`)

Hardware neutrality here means one kernel source and *measured* per-device
parameters, not vendor forks. `GpuProfile.of(ctx)` (`qumo/gpu/device.mojo`)
is built once per `GpuStateVector`; it records what the runtime reports
(each query guarded: Metal raises "Attributes not supported") and the tuning
the engine uses. `qumo_gpu_info` / `GpuInfo.to_json()` report it under
`"profile"` (without kernel limits: those are read only by a
`GpuStateVector` whose device could use a larger tile, and appear in its
profile, e.g. in `bench_gpu N R probe`).

| Field | Meaning | Default | Environment override (runtime) |
|---|---|---|---|
| `low_fp32`, `low_fp64` | always-resident low qubits of a tile (2^low contiguous amplitudes per chunk of a tile load) | 4 Apple (measured), 5 elsewhere (unmeasured) | `QUMO_GPU_LOW_BITS` = 3..5 |
| `tile_bits_fp32`, `tile_bits_fp64` | log2 amplitudes per tile (2^(k - REG_BITS) threads per block) | 11 / 10; up to +2 when the worst-case run fits one block (opt-in limit), 2 typical runs fit per multiprocessor and the block is known to launch: <= 256 threads, or within the compiled kernel's reported `MAX_THREADS_PER_BLOCK` with >= 2 resident blocks (occupancy calculator). From published shared-memory limits (computed, not run): T4 11 / 10, MI300X 11 / 10, Apple 11 (static arrays); A100 / H100 11 / 11 at 256 threads, 12-13 / 12 only if the kernel's registers allow 512 / 1024-thread blocks (<= 128 / 64 per thread); L4 11 / 11, fp32 12 likewise | `QUMO_GPU_TILE_BITS` (ignored if it does not fit or its block is not known to launch) |
| `kl_fp32`, `kl_fp64` (`tile_kernel_*` in JSON) | the compiled tile kernels' `MAX_THREADS_PER_BLOCK`, largest block with >= 2 resident blocks, registers and spill bytes per thread, read from the driver when a larger tile is possible (-1: not read; Metal reports none) | | |
| `diag_width` | widest fused diagonal phase table | 6 | `QUMO_GPU_DIAG_WIDTH` = 0..6 (<= 1: off) |
| `split_1q_fp32`, `split_1q_fp64` | complex 1q gates of the form diagonal . real 2x2 as G1R + diagonal (generic U3 stays dense) | off | `QUMO_GPU_SPLIT_1Q` (both), `QUMO_GPU_FP64_LEAN` (fp64) = 0 / 1 |
| `multiprocessors`, `smem_per_block[_optin]`, `smem_per_multiprocessor`, `warp`, `arch` | reported by the device (-1 when not) | | |
| `weak_fp64()` | NVIDIA cc 61 / 75 / 86 / 89 / 120, AMD gfx10-12: fp64 at 1/32-1/64 of fp32 | | |

The A/B data behind the Apple defaults is in the optimisation log below;
the NVIDIA / AMD defaults stay at the previous values until a device run
(`bench_gpu N R probe`, then the families with the environment overrides)
says otherwise.

## Engine probe (`bench_gpu N R probe`)

`./build/gpu/bench_gpu 26 3 probe` prints, for n = N-2 and N, fp32 and
(where the device has it) fp64, one `PROBE,<dtype>,<n>,<metric>,<value>,<unit>`
line per number (`bench/run_benchmarks.py`, suite `sv_gpu_probe`, stores
them):

* device facts and the `GpuProfile` JSON; registers, local (spill) bytes,
  static shared memory, `MAX_THREADS_PER_BLOCK` and blocks per
  multiprocessor (at the engine's block size and a typical run's shared
  memory) of the tile kernels as the engine loads them, where the API
  offers them (not on Metal);
* `sweep_ms`: one empty tile run (read + write of the whole state);
* `op_<kind>_us`: cost per op inside a run, as the slope between a 32-op and
  a 120-op single-segment run (8 -> 30 for G2): G1, G1R, XC (CX with a
  low control), DG (diagonals one by one), DGF (the same diagonals fused, per
  gate), SW, G2;
* `transpose_us`: a run of 64 dense 1q gates that changes register set every
  4 ops, against one that does not, per transpose;
* `stream_<32B|64B|128B|tile>_GBps`: a tile-shaped read + write of both
  arrays with contiguous chunks of that size (the coalescing price of a
  smaller `low`);
* per family (layered / QFT / random): evolve time, runs, transposes, ops and
  `evolve_over_sweep_floor` = evolve / (runs x sweep). Near 1 the engine is
  bound by state sweeps (fewer runs pay: larger tiles, smaller `low`); well
  above 1 by the work inside tiles.

M5 (fp32, 24 qubits, final engine, Low Power Mode, load ~3): sweep 2.24 ms
(120 GB/s); per op G1 156-179 us, G1R 115-126, XC 169-172, DG 139-144, DGF
29-37 per gate, SW 134-135, G2 330-352; transpose 241-251 us; stream 32 B
83 GB/s, 64 B 111, 128 B 121; evolve / sweep floor 3.4 (layered), 2.3
(QFT), 3.3 (random). At 26 qubits: sweep 10.0 ms (107 GB/s), evolve /
sweep floor 3.0 / 2.0 / 3.0 (independent review: 2.94 / 1.96 / 2.90). Before this round (same probe, engine as found): G1 165 us, XC 136,
DG 161, transpose 793 us, layered evolve / sweep floor 4.0.

## Pauli-frame sampler for QEC (`GpuFrameSampler`)

This is the GPU counterpart of `qumo.sim.frame.FrameSimulator` (Stim's
Pauli-frame method):

* Same lowered `FrameProgram` (frame primitives + tableau reference sample),
  same `DetectorSamples` / `MeasurementSamples` bit layout, so decoders
  consume GPU samples unchanged.
* One thread per 64-shot word; the whole circuit is one launch per chunk;
  each thread interprets the op table (broadcast loads, uniform control
  flow). Rows are stored `row * nw + word` (coalesced). Large runs are
  chunked to at most 256 MiB of frame scratch per launch; the scratch is
  zero-filled before use.
* Integer-only noise: geometric skipping with host-precomputed thresholds
  `T_k = floor(2^32 (1-p)^k)` and Lemire's multiply-shift for Pauli choices.
  No floating point runs on the device: no fp64 needed on Apple, and results
  are bit-identical across vendors and the emulator (asserted on the M5:
  device == emulator for 100k shots of a d=5 surface code).
* Randomness: word `g` uses `Rng(seed).split(g)`, so samples depend only on
  (circuit, shots, seed). They agree with the CPU sampler in distribution
  (5 sigma per detector and for the logical rate) and exactly whenever the
  outcomes are deterministic (tested on the device).

## Testing

MAX's `DeviceContext(api="cpu")` does not execute `enqueue_function`
kernels, so QUMO tests kernels by host emulation in default builds and on
the real device in `-D QUMO_GPU` builds:

* `qumo.gpu.emulate` runs each kernel's thread phases for every thread of
  the real grid; a `barrier()` is the end of a phase; registers that live
  across barriers are per-thread host arrays. `GpuStateVector.emulator(n)` and
  `GpuFrameSampler(c, emulate=True)` exercise all host logic (scheduling,
  tables, launch geometry, reductions, sampling, chunking).
* `tests/test_gpu.mojo` (emulation fp64 + fp32, then device fp32 on the M5;
  fp32 + fp64 on NVIDIA / AMD devices): the precision policy (fp32 build
  default; `"auto"` / `"fp32"` / `"fp64"` and `QM-E941`; the
  `QUMO_GPU_PRECISION` override; `QM-E472` for fp64 on Apple from
  `sample_on`, the orchestrator and `qumo_run_device`; the run-time entry
  points in emulation in both precisions);
  every gate on operands including qubits 0 and n-1 and on 1-3 qubit states;
  random circuits over every opcode family up to 21 qubits (each also checks
  launches = tile runs, that the device's runs / transposes / ops equal the
  host program's, and that wide circuits need < half as many launches as
  plan ops); controls and diagonal qubits inside the register set, outside
  it and outside the tile for every controlled kind; tiles with 3, 4 and 5
  resident low qubits and k - 1, k + 1, k + 2 tile qubits (the larger ones in
  emulation, or on devices whose shared memory fits them); diagonal fusion at
  widths 0-6 with and without split 1q gates (fp64 emulation within 1e-12 of
  the CPU); the scheduler (coverage, packed segment records, direct loads /
  stores only from coalescing register sets, QFT in <= 6 runs at 20 qubits
  and its diagonal fusion, G1R emission, controls / diagonals outside the
  tile, 32-qubit launch limits); the tile-size policy for T4 / A100 / L4 /
  Apple shared-memory figures, with and without reported kernel limits
  (unknown -> <= 256 threads; 512 / 1024-thread caps; < 2 resident blocks;
  the AMD launch bound) and the constant shared-memory opt-in; the split
  policy (only diagonal . real gates split, same runs as unsplit on a
  diagonal-heavy circuit);
  probabilities and norm; expectation values incl. the 1024-block
  grid-stride regime and batched terms; sampling (fp64 emulation = CPU
  sampler exactly, reproducible shots, no zero-probability states, the last
  of several MEASUREs into one clbit wins, as in `runner.sample`); a
  24-qubit random circuit with probabilities, 19-term expectation and 20k
  shots on the device; error codes, dispatch, orchestrator, MCP tools.
* `tests/test_gpu_frame.mojo` (emulation, then device): threshold tables,
  noiseless codes, deterministic injected Paulis = CPU for every shot,
  circuit-level noise statistics, feed-forward / reset / reference XOR,
  chunk invariance, dispatch, device == emulator bit for bit.

## Dispatch threshold

`GPU_MIN_QUBITS` is device specific, from `bench_gpu 22 5 crossover 8`
(end to end per call: DeviceContext + allocation + evolve + 1000 shots, vs the
CPU `sample`; interleaved, medians; layered circuits):

| qubits | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|
| CPU sample (ms) | 0.51 | 0.30-0.69 | 0.64-0.76 | 44-74 | 45-117 | 35-126 | 115-157 | 473 |
| GPU sample_gpu (ms) | 4.4 | 2.1-4.3 | 2.7-3.9 | 4.0-6.4 | 4.2-6.2 | 5.9-7.1 | 12-17 | 68 |

The GPU call has a ~2-6 ms floor; the CPU wins up to 12 qubits and loses from
14 on (on this loaded machine the CPU's threaded path costs tens of ms from
14 qubits). 16 keeps a margin for an idle CPU, so **`GPU_MIN_QUBITS = 16` on
Apple GPUs**. NVIDIA/AMD keep the earlier, unmeasured value 20. Override
with `-D QUMO_GPU_MIN_QUBITS=<n>`. Automatic dispatch never uses GPU Pauli
frames: on the M5 the CPU frame simulator is faster (see below).

## Benchmarks (Apple M5, measured)

```bash
make bench-gpu                                  # default: layered 16..26 + frames d = 3, 5, 7
make bench-gpu BENCH_GPU_ARGS="26 3 families 20 26"   # layered / qft / random, CPU up to 26
make bench-gpu BENCH_GPU_ARGS="22 5 crossover 8"      # dispatch threshold
make bench-gpu BENCH_GPU_ARGS="9 3 frames 1000000"    # Pauli frames, d = 3..9
./build/gpu/bench_gpu --stim 5 0.001            # the frames circuit in Stim syntax
```

Machine: Apple M5 (10 CPU cores, 10-core GPU, 32 GB unified memory), macOS,
Mojo 1.1 + MAX 26.6, Low Power Mode on, shared with other jobs. QUMO CPU =
`qumo.sim.runner.evolve` (fp64, SIMD + threads + qubit reordering); QUMO GPU
= `GpuStateVector[float32]` `reset + evolve + synchronize` on a preallocated
state (one warm-up run; allocation and kernel load excluded; the default
table also prints `gpu_alloc_s`: allocation on the same context included,
freeing excluded, as in CUDA-Q's timed region); the two are
interleaved in one process, 3 repetitions, medians.

Current engine, `bench_gpu 28 3 families 20 26` (2026-10-13, load ~3),
seconds:

| family | n | gates | QUMO GPU fp32 | QUMO CPU fp64 | GPU vs CPU | sweeps (tile runs) | transposes |
|---|---|---|---|---|---|---|---|
| layered | 20 | 495 | 0.0102 | 0.0325 | 3.2x | 10 | 45 |
| layered | 22 | 545 | 0.0267 | 0.168 | 6.3x | 11 | 51 |
| layered | 24 | 595 | 0.0762 | 0.725 | 9.5x | 12 | 53 |
| layered | 26 | 645 | 0.419 | 3.25 | 7.8x | 14 | 57 |
| layered | 28 | 695 | 1.81 | - | - | 15 | 66 |
| qft | 20 | 220 | 0.00446 | 0.0127 | 2.9x | 4 | 8 |
| qft | 22 | 264 | 0.00898 | 0.0618 | 6.9x | 5 | 9 |
| qft | 24 | 312 | 0.0315 | 0.279 | 8.9x | 6 | 11 |
| qft | 26 | 364 | 0.131 | 1.50 | 11.5x | 7 | 10 |
| qft | 28 | 420 | 0.658 | - | - | 7 | 12 |
| random | 20 | 300 | 0.0110 | 0.0364 | 3.3x | 11 | 58 |
| random | 22 | 330 | 0.0260 | 0.192 | 7.4x | 12 | 63 |
| random | 24 | 360 | 0.0878 | 0.866 | 9.9x | 14 | 79 |
| random | 26 | 390 | 0.458 | 3.83 | 8.4x | 15 | 80 |
| random | 28 | 420 | 1.95 | - | - | 18 | 84 |

The engine before this round against the current one, same session,
interleaved (`ab` harness: 5 rounds x 3 repetitions at 20-26 qubits, 3 x 3
at 28, medians of the round medians, ms; load 2.5-3.7):

| n | layered before -> now | qft before -> now | random before -> now |
|---|---|---|---|
| 20 | 6.65 -> 5.00 (1.33x) | 4.34 -> 1.97 (2.20x) | 7.16 -> 4.59 (1.56x) |
| 22 | 35.3 -> 17.4 (2.02x)* | 19.4 -> 8.99 (2.16x)* | 35.8 -> 25.0 (1.43x)* |
| 24 | 144.9 -> 93.8 (1.55x) | 80.2 -> 32.2 (2.49x) | 158.4 -> 103.5 (1.53x) |
| 26 | 625.6 -> 416.2 (1.50x) | 367.7 -> 141.2 (2.60x) | 675.0 -> 440.3 (1.53x) |
| 28 | 2701 -> 1817 (1.49x) | 1639 -> 642 (2.55x) | 2959 -> 1964 (1.51x) |

\* 22 qubits is not robust: times there are bimodal across sessions (the
  current engine's layered-22 median was 17.0 ms in one session and 23.6 ms
  in another), and independent interleaved re-runs gave layered 1.46-1.69x,
  QFT 2.41x, random 1.18-1.52x. Quote 24-28 qubits.
* Fidelity F of the GPU fp32 state with the CPU fp64 state: |1 - F| <= 6e-7
  at every width up to 26 (column `fidelity` of the `families` output).
* 28 qubits = 2 GiB of fp32 amplitudes: GPU only (the fp64 CPU run would
  need 4 GiB plus copies).
* Qiskit Aer 0.17.2 CPU (fp64) on this machine, earlier session, same
  circuits: layered 26 12.7 s, QFT 26 11.0 s, random 26 9.70 s
  (`bench/bench_sv_aer.py`, `OMP_WAIT_POLICY=passive`).
* Read-outs on the GPU state (default `make bench-gpu` table, earlier
  session): sum of n-1 ZZ terms 18 ms at 24 qubits and 43 ms at 26; 10k-shot
  sampling 16 ms and 24 ms; |<H>_GPU - <H>_CPU| <= 5e-6.

**Pauli frames** (rotated surface code memory, d rounds, p = 0.001,
1M shots, `bench_gpu 9 3 frames`; Stim 1.16 `compile_detector_sampler().sample(
shots, bit_packed=True)` on the identical circuit exported with `--stim`,
single-threaded; medians of 3; shots per second):

| d | QUMO CPU frames | QUMO GPU frames | Stim (1 thread) | GPU vs CPU | GPU vs Stim |
|---|---|---|---|---|---|
| 3 | 198 M | 151 M | 16.0 M | 0.76x | 9.4x |
| 5 | 42.7 M | 23.6 M | 3.70 M | 0.55x | 6.4x |
| 7 | 15.3 M | 9.08 M | 1.31 M | 0.59x | 6.9x |
| 9 | 7.19 M | 3.30 M | 0.62 M | 0.46x | 5.3x |

Detection-event counts agree across the three (e.g. d = 9: 11.95 M CPU,
11.94 M GPU, 11.95 M Stim). On the M5 the multi-threaded SIMD CPU frame
simulator beats this GPU kernel; it is therefore used only on explicit
`device="gpu"` requests.

### Optimisation log (24 qubits, fp32, M5)

Each row is a version of the engine, measured with the same harness
(`evolve` from |0>, median of 5) in one interleaved session (three rounds,
range over rounds, load ~23-46), seconds:

| version | layered | qft | random |
|---|---|---|---|
| v0: one launch per plan op, matrices as SIMD kernel arguments | does not run (Metal ICE) | - | - |
| v1: shared-memory tile runs, 256 threads, 32-bit indices | 0.238-0.261 | 0.189-0.242 | 0.241-0.280 |
| v1 with 128 threads per block | 0.205-0.226 | 0.200-0.208 | 0.211-0.223 |
| v2: register blocking (8 amplitudes per thread), all gate qubits in the tile | 0.342-0.374 | 0.260-0.305 | 0.370-0.374 |
| v3 (current): v2 + controls / diagonal gates anywhere, X / CX as register swaps, op cache in shared memory, FULL / HEAVY kernel specialisations | 0.221-0.237 | 0.119-0.125 | 0.234-0.272 |

* Controls / diagonals anywhere: launches at 24 qubits 22 -> 15 (layered),
  33 -> 7 (QFT), 22 -> 16 (random): the QFT gain (1.6-1.9x) comes from here.
* HEAVY split (SWAP / 4x4 code only in runs that need it): -25% on a run of
  96 dense 1q + CX gates, -12% on 48 CX with a far control (probe, interleaved).
* 64 -> 32-bit index arithmetic (v1): layered 0.29-0.39 s -> 0.21-0.25 s, not
  interleaved (load ~200), indicative only.
* Honest result: on layered and random circuits the register-blocked engine
  is on par with the simpler shared-memory engine at 128 threads; the gate
  work inside a run (~0.15-0.3 ms per plan op at 24 qubits, against
  ~3.5-4 ms for one state sweep at ~70-100 GB/s) is what remains.
* Tried and **not adopted**: qsim-style fusion of each register segment's
  gates into one dense 8x8 unitary (fp64 product on the host, `G3` op).
  Interleaved at 24 qubits (load ~5, three rounds): layered 0.151-0.168 s
  -> 0.162-0.168 s, QFT 0.084-0.085 -> 0.093-0.098, random 0.165-0.172 ->
  0.188-0.193. The 8x8 products (64 complex multiply-adds and 128
  shared-memory matrix loads per thread) cost more than the dispatches they
  remove. Also tried: 2 or 4 amplitudes per thread (`QUMO_GPU_REG_BITS`
  2 / 4: 2 is 10-20% slower, 4 within noise; after round 2's segment cache 4
  is 5-8% faster and became the Apple default) and 8 KiB tiles (slower:
  more runs).
* Reductions (same state, interleaved, 3 rounds): 23 ZZ terms 51-126 ms ->
  9-16 ms (one launch reduces 8 same-X terms); 8 single-X terms (8 distinct X
  masks, 8 launches either way) and 10k-shot sampling unchanged within noise
  (17-22 ms and 11-27 ms -> 11-14 ms).

### Optimisation log, round 2 (after the CUDA-Q comparison; 26 qubits, fp32, M5)

Each step was kept only if it is exact (tests in emulation and on the
device, fidelity unchanged) and not slower on the M5, except steps aimed at
other GPUs that are neutral here, which were made device-adaptive. Each row
is its own interleaved A/B session against the previous version (built from
snapshots; 4-5 rounds x 3 repetitions at 22-26 qubits, 3 rounds at 28;
medians; Low Power Mode on; load 3-15). Absolute times differ between
sessions by up to 20%; the ratios are the result. Sweeps / transposes are
layered / QFT / random at 26 qubits.

| step | change | layered | qft | random | sweeps, transposes |
|---|---|---|---|---|---|
| 1 | segment records cached in shared memory, packed to 2 x Int32 (a transpose re-read them from device memory after its barrier) | 678 -> 539 ms (0.79); 28 q: 0.73 | 0.94; 28 q: 0.85 | 759 -> 557 (0.73); 28 q: 0.73 | 17 / 7 / 16; 93 / 25 / 115 |
| 2 | load / store straight into a run's first / last register set when it has no low qubit | 0.99 | 0.97 | 0.94 (28 q: within noise) | transposes 75 / 17 / 104 |
| 3 | 16 amplitudes per thread on Apple GPUs (`QUMO_GPU_REG_BITS` default 4 there) | 500 -> 461 (0.92); 28 q: 0.95 | 0.92; 28 q: 0.94 | 0.93; 28 q: 0.93 | transposes 55 / 10 / 78 |
| 4 | `GpuProfile`, `low` at runtime; Apple default low = 4 | low 5 -> 4: 0.95-0.99; 28 q: 0.99 | 26 q: 0.96-1.00; 24 q: 1.04-1.05 (slower by ~1.5 ms, two sessions); 28 q: 1.00-1.04 | 0.98-1.00; 28 q: 0.97-0.99 | sweeps 14 / 7 / 15 |
| 5 | one dynamic shared region per launch, k at runtime (static arrays kept on Apple) | 1.00 | 0.99 | 1.00 | (all-dynamic on the M5: 1.00-1.06 in one session, 0.97-1.01 in an independent one) |
| 6 | 16-byte device op records, control / parity masks split outside / inside the register set (only g0 live), vector matrix loads, real 2x2 (G1R) | 0.98 (24 q: 0.97) | 272 -> 251 (0.92; 24 q: 0.89) | 0.96 | |
| 7 | consecutive diagonal gates fused into phase tables (DGT), matrix cache 1024 -> 2048 values | 1.00 | 270 -> 141 (0.52); 24 q: 0.53; 28 q: 0.51 | 1.01 | QFT tile ops 364 -> 127 |
| 8 | diagonals moved past commuting ops (exact); optional split of complex 1q gates | sinking: 0.99-1.01 | 0.99-1.02 | 1.00-1.01 | split (off; two-sided version): layered 1.07-1.09, random 1.31-1.33 slower |

All together (engine before this round -> now, same session, table above):
1.49-1.55x on layered and random circuits at 24-28 qubits, 2.5-2.6x on the
QFT.

Measured and **not** adopted this round:

* All-dynamic shared memory on Apple (1.00-1.06x slower in one session;
  an independent session measured it neutral, 0.97-1.01): Apple keeps
  static arrays (`QUMO_GPU_TILE_SMEM_DYNAMIC` = 0 there), which is harmless
  either way.
* `low` = 3 on the M5: runs 17 -> 12 at 26 qubits but mixed times (QFT 24
  1.05x slower, QFT 26 0.94x): `low` = 4 chosen. The probe's 32-byte
  chunks stream at 49-79% of 128-byte ones on the M5 (64-byte: 70-94%;
  the probe places chunks on the top qubits, a worst case).
* Split complex 1q gates on the M5 (above): off by default everywhere until
  a weak-fp64 GPU measures it.
* Branch-free selects for controlled X swaps: XC 170 -> 191 us per op
  (probe, 24 qubits).
* From the plan, not tried because measured or counted against before: a
  one-sweep-per-fused-gate engine (cuStateVec's model), dense 8x8 fusion of
  register segments (`G3`, log above), fusing each CX with its 1q gates into
  a 4x4 before tile packing (host count: layered 26 runs 17 -> 23,
  transposes 93 -> 128), SIMD-group shuffles instead of transposes (366-376
  us per shuffled 1q gate against ~150 us for a register G1 on the M5).

### Round 3 (review fixes; robustness on NVIDIA / AMD, M5 neutral)

An independent review of round 2 found that the tile-size policy could pick
blocks that do not launch on big NVIDIA GPUs: it checked shared memory and
1024 threads per block, never registers per thread (A100 / H100 fp32 k = 13
= 1024 threads needs <= 64 registers; the gfx942 build of the same kernel
uses 100-128+). Fixed, all exact and cross-compiled for the five targets:

* Tile size vs registers: blocks above 256 threads only within the compiled
  kernels' reported `MAX_THREADS_PER_BLOCK` and with >= 2 resident blocks
  (`TileKernelLimits`, [kernel details](#state-vector-engine)); unknown ->
  <= 256 threads. A100 / H100 now default to 11 / 11 unless the driver
  reports that larger blocks launch.
* AMD launch bound 256 (`gk_tile_lb`): no VGPR spills left (were 44-304 B
  of scratch per thread in six of eight variants).
* One constant shared-memory opt-in attribute per device instead of each
  flush's largest request.
* `split_1q` only for gates of the form diagonal . real, with an exact
  scheduler count (it reserved 3 ops per 1q gate and cut runs early).
* `make test-gpu` fails when no device is usable; `gpu_alloc_s` no longer
  times the frees; probe reports the kernels the engine actually loads.

Interleaved on the M5 against the round-2 engine (4 rounds x 3 repetitions
at 22-26 qubits, then 6 x 5 at 24; load 3.3-3.9): 0.996-1.016 at 24 qubits,
1.00-1.02 at 26, 1.01-1.06 at 22 (noisy, see above). The Apple kernel's
source is unchanged; the host changes only add profile arithmetic.
Measured and not adopted:

| change | layered | qft | random |
|---|---|---|---|
| tile-kernel body in an `@always_inline` helper shared with `gk_tile_lb` (Metal) | 1.92-2.30x slower | 1.41-1.61x | 1.63-1.91x |
| Metal launch bound 128 (`-D QUMO_GPU_TILE_LAUNCH_BOUND=128`, 16 amplitudes per thread) | 0.997-1.02 | 1.01 | 0.99-1.01 |
| one-sided split on (`QUMO_GPU_SPLIT_1Q=1`), M5 fp32 | 24 q 1.01, 26 q 1.09 (ops 377 -> 509) | 0.95-1.12 (same ops, noise) | 1.02-1.04 |

## Comparison with CUDA-Q

What CUDA-Q 0.16 does on an NVIDIA GPU (the version of the Colab T4 run,
`cuda-quantum-cu13 0.16.0.post1`, which pins `custatevec-cu13~=1.14`;
sources at [github.com/NVIDIA/cuda-quantum, tag 0.16.0](https://github.com/NVIDIA/cuda-quantum/tree/0.16.0)):

* The host builds each gate's matrix in the target precision (fp32 by
  default, `runtime/cudaq/platform/default/nvidia.yml` L21-32) and queues it
  (`runtime/nvqir/CircuitSimulator.h` L604-617, L916-959); nothing runs until
  a flush (measure, sample, observe, `get_state`, end of kernel).
* A flush hands the queue to cuStateVec Ex's `SVUpdater`
  (`runtime/nvqir/custatevec/CuStateVecGateEngine.cpp` L61-107, L211-228;
  PR #5025), which reclassifies matrices (diagonal / anti-diagonal / dense,
  `CuStateVecTasks.h` L60-107), fuses gates (closed source: dense fusion
  width chosen by the library, 4-6 per the [cuStateVec Ex
  docs](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/api-reference/custatevecex/index.html);
  diagonal fusion up to 20 qubits, `CuStateVecConfig.cpp` L146-178) and
  applies each fused matrix in one pass over the state. The plan is rebuilt
  on every flush.
* Inference from the T4 numbers (not profiled): CUDA-Q fp32 at 26 qubits
  (0.17 s for the 645-gate layered circuit) fits ~43-48 state sweeps of
  1.07 GB at ~272-300 GB/s; fp64 (1.45 s) fits ~42 fused 4-qubit gates
  bound by the T4's fp64 multiply-adds (64 per amplitude each, 1/32 of the
  fp32 rate).

QUMO's engine is built the other way round: a *tile run* applies a whole
run of gates (many fused 1q gates, CXs, diagonals) to a 2^k-amplitude tile
in shared memory and registers, so one sweep carries ~25-50 plan ops; the
layered circuit at 26 qubits is 14-17 sweeps in fp32 at k = 11 (low 4 / 5;
fp32 is the default on every GPU), 20 in fp64 at k = 10, 11-12 with larger tiles,
instead of ~43-48. The cost has moved inside the tile (on the M5, evolve is
2.3-3.5x the sweep floor at 24 qubits and 2.0-3.0x at 26: probe above),
which is what round 2 attacked. Adopted from CUDA-Q /
cuStateVec (and the open engines with the same ideas):

* **Diagonal fusion** (cuStateVec Ex's diagonal fusion, CUDA-Q's
  `CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS`; also qsim and Qiskit Aer's
  `fusion.hpp` diagonal pass): consecutive diagonal gates of a run become one
  phase table (`DGT`, up to 6 qubits, fp64 product rounded once). QFT 2x.
* **Gate classification by matrix structure** (cuStateVec's compact matrix
  tasks): QUMO already had diagonal (`DG`) and X-like (`XC`) ops; added a
  real 2x2 (`G1R`, half the multiply-adds) and the optional split of
  diagonal . real 1q gates (RZ . RY products) for arithmetic-bound fp64.
* **Per-device tuning instead of one setting** (CUDA-Q's per-compute-
  capability fusion widths, `docs/sphinx/using/backends/sims/svsims.rst`
  L275-297 at 0.15.0): `GpuProfile` chooses tile size from the reported
  shared memory, and `low`, diagonal width and the split from measured
  data, with environment overrides so a device run can sweep them.
* **A small always-resident low-qubit block** (HyQuas `COALESCE=3`, Atlas's
  3 resident low qubits): `low` became a runtime parameter (3-5; 4 on the
  M5).
* Kept from before: CUDA-Q's lazy |0> (QUMO writes the initial state in the
  first run's load and never zero-fills), and QUMO plans once per evolve.

Not adopted: one sweep per fused dense gate (it needs more sweeps than tile
runs, and on the T4's fp64 wider dense gates cost more multiply-adds: 64 per
amplitude for a 4-qubit gate against 8 for a 1-qubit one); see the
not-adopted list in the log above.

### What a Colab run must confirm

Nothing in rounds 2-3 has run on NVIDIA or AMD. On the next Colab run
(`examples/colab`, configs `colab_t4` / `colab_l4` / `colab_a100`):

1. `make test-gpu` on the device: first run of this engine on NVIDIA (the
   dynamic shared-memory layout ran on the M5 only), fp32 and fp64. It now
   fails if the device suites cannot run.
2. `bench_gpu 26 3 probe` (suite `sv_gpu_probe`): `evolve_over_sweep_floor`
   decides what to tune next (<= ~1.6: sweeps dominate -> `QUMO_GPU_LOW_BITS`
   = 4 / 3, larger tiles; well above: in-tile work); the 32 / 64 / 128 B
   stream bandwidth (the price of a smaller `low`); `gk_tile` registers,
   spill bytes, `MAX_THREADS_PER_BLOCK` and blocks per SM, and the profile's
   `tile_kernel_*` fields.
3. `sv_gpu` fp32 against CUDA-Q (default and the labelled no-fusion
   variant) and Aer, all in fp32 by default; fp64 with `--fp64`
   (`INCLUDE_FP64` in the notebooks). Model only, for the layered circuit at
   26 qubits on the T4:
   * fp32: memory floor 17 sweeps x ~3.95 ms (1.07 GB at ~272 GB/s) = 67 ms.
     The in-tile work is unknown on NVIDIA; on the M5 it is ~280 ms at 26
     qubits (evolve 423 ms - 14 x 10.0 ms). Matching CUDA-Q's 0.17 s needs
     the in-tile part ~2.7x faster than on the M5 (0.10 s would need ~8x).
     Parity is possible; beating CUDA-Q is not supported by any measurement.
   * fp64: 20 sweeps (k = 10) x ~7.9 ms = 158 ms of memory, and a
     multiply-add floor of 1.06 s (252 complex 2x2s x 8 fp64 FMAs x 2^26
     amplitudes / (40 SMs x 2 FMA/clock x 1.59 GHz): full boost clock and
     every fp64 unit busy). CUDA-Q measured 1.45 s, i.e. ~1.37x that floor;
     QUMO near or above it is the realistic expectation.
4. Sweeps with the environment knobs, same binary: `QUMO_GPU_LOW_BITS` =
   5 / 4 / 3; `QUMO_GPU_TILE_BITS` = 11 / 12 (T4; the profile keeps 11 / 10,
   12 fits one block of the T4's 64 KiB); `QUMO_GPU_FP64_LEAN` = 0 / 1 in
   fp64 (it reduces fp64 multiply-adds only on RY / RZ-type circuits such
   as the layered one: -22% by count; adopt as the weak-fp64 default only if
   it wins); and a second build with `-D QUMO_GPU_REG_BITS=4`.
5. A100 / H100 / L4: read the profile's `tile_kernel_max_threads_*` and
   `tile_bits_*` (11 / 11 unless the driver reports that 512 / 1024-thread
   blocks launch with >= 2 resident blocks); compare the families with
   `QUMO_GPU_TILE_BITS` = 11 / 12 / 13 (host counts for layered at 26
   qubits: 17 / 12 / 11 sweeps in fp32).
6. AMD (MI300X notebook): `make test-gpu`; registers / spills of
   `gk_tile_lb` (launch bound 256; cross-compiled: no spills, but fewer
   resident workgroups for the heaviest variants); A/B the families against
   a `-D QUMO_GPU_TILE_LAUNCH_BOUND=0` build (spilling, unbounded).

## Error codes

| Code | Meaning |
|---|---|
| QM-E470 | GPU support not compiled into this build (rebuild with `-D QUMO_GPU`) |
| QM-E471 | Dynamic or noisy circuit given to the GPU state-vector sampler |
| QM-E472 | float64 requested on an Apple GPU (`precision: "fp64"`, `QUMO_GPU_PRECISION=fp64` or `GpuStateVector[DType.float64]`) |
| QM-E473 | Sampling a zero-norm state |
| QM-E474 | Circuit, host state or basis index out of range for the GPU state |
| QM-E475 | Plan op without a GPU kernel |
| QM-E940 | Unknown device name (`auto`, `cpu`, `gpu`) |
| QM-E941 | Unknown GPU precision (`auto`, `fp32`, `fp64`; argument or `QUMO_GPU_PRECISION`) |
| QM-W020 | `device: "gpu"` requested but the run fell back to the CPU (reason given) |
| QM-W021 | A requested precision did not apply (CPU state vectors are fp64, stabilizer / Pauli frames exact) |

## Ideas adopted, and from where

* **qsim** (Google): gate fusion before execution and blocking of many gates
  per state sweep with register/shared-memory staging; here re-derived as
  tile runs + register segments.
* **cuStateVec** (NVIDIA): batched gate application on chosen index bits;
  sampling via probability prefix sums over segments and per-shot search;
  host-side fp64 accumulation of reductions.
* **Qiskit Aer GPU**: the same execution plan for CPU and GPU.
* **CUDA-Q / cuStateVec Ex**: diagonal-gate fusion (phase tables), gate
  classification by matrix structure, per-device fusion / tuning
  parameters ([Comparison with CUDA-Q](#comparison-with-cuda-q)); precision
  as a target property with fp32 as the default on every GPU and fp64 on
  request ([Precision policy](#precision-policy)); heterogeneous
  `mqpu`-style device pools (the orchestrator);
  explicit fallback reporting.
* **HyQuas / Atlas** (shared-memory state-vector kernels): a small
  always-resident low-qubit block (QUMO's runtime `low`); controls and
  diagonal gates need not be tile qubits ("insular" qubits in Atlas).
* **Stim** (Gidney 2021): Pauli frames, reference sample, Z-gauge
  re-randomisation, geometric skipping. **Lemire** (2019): multiply-shift
  bounded integers.

## Limits and next steps

* Everything on NVIDIA / AMD is unmeasured for this engine
  ([what to confirm](#what-a-colab-run-must-confirm)); their `low`, tile
  size, register width and the fp64 split keep untuned or computed defaults.
* On the M5 the work inside a tile still dominates (evolve = 2.3-3.5x the
  sweep floor at 24 qubits, 2.0-3.0x at 26): per op ~115-180 us at 24
  qubits (G1R / G1 / XC), a transpose ~250 us. Candidates: fewer ops per run (dense fusion where it is cheaper,
  cost-based as in Aer's `fusion.hpp`), cheaper X/CX (still ~170 us per op
  with no arithmetic), caching the tile program across calls for
  variational loops (host side).
* On the M5 the GPU Pauli-frame sampler is slower than the CPU frame
  simulator for d <= 9 (one thread per 64-shot word gives too little
  parallelism and the UInt64 ALU work is emulated); 32-bit words per thread
  are the obvious next step. It is used only on explicit `device="gpu"`.
* Single GPU, single stream. No multi-GPU state partitioning and no
  host-memory offload; 28 qubits fp32 is the largest size measured.
* The GPU state-vector sampler handles static circuits only; dynamic and
  noisy non-Clifford circuits use CPU trajectories.
