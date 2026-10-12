# QUMO on GPUs (`qumo.gpu`)

QUMO runs its state-vector simulator and its QEC Pauli-frame sampler on
NVIDIA, AMD and Apple GPUs through MAX's vendor-neutral GPU API
(`max.gpu`, `DeviceContext`). The same Mojo kernel source compiles to PTX
(NVIDIA), AMDGCN (AMD) and Metal (Apple). There is no CUDA, HIP or Metal
code in the repository.

> **Status (honest):** every kernel cross-compiles for `sm_90`, `gfx942` and
> `metal:4` (`make gpu-check`). The kernel logic is unit-tested in host
> emulation against the CPU simulators (`tests/test_gpu*.mojo`). The kernels
> have **not yet run on real GPU hardware**: the development Mac has no
> Metal toolchain installed (see [Enabling](#enabling-the-gpu)). Device test
> suites and benchmarks are written and gated, and run as soon as a toolchain
> and device exist. No GPU performance numbers are claimed yet.

## Modules

| File | Role |
|---|---|
| `qumo/gpu/device.mojo` | `gpu_compiled()`, `gpu_available()`, `gpu_info()` / `query_gpu()` (`GpuInfo`, JSON), precision policy (`GPU_DEFAULT_DTYPE`), `GPU_MIN_QUBITS`, error constructors |
| `qumo/gpu/kernels.mojo` | Device-agnostic state-vector **thread bodies** `th_*` plus thin GPU kernels `gk_*` |
| `qumo/gpu/emulate.mojo` | Host emulator: runs a thread body over every thread of the real launch grid; emulates the shared-memory tree reductions level by level |
| `qumo/gpu/statevector_gpu.mojo` | `GpuStateVector[dtype]`, plus `evolve_gpu`, `expectation_gpu`, `sample_gpu` |
| `qumo/gpu/frame_gpu.mojo` | `GpuFrameSampler`: QEC Pauli-frame sampling, one thread per 64-shot word |

Integration (all GPU calls gated, see below):

* `qumo/exec/dispatch.mojo`: `choose_method` returns `"gpu"` for static,
  noiseless, non-Clifford (or parametric) circuits with at least
  `GPU_MIN_QUBITS = 20` qubits, when the build has GPU support and a device
  is usable. `sample_auto` runs them on the GPU and falls back to the CPU
  on any GPU error. `sample_on(c, shots, seed, params, device, note)` takes
  `device` = `"auto" | "cpu" | "gpu"`. With `"gpu"`, Clifford circuits
  without feed-forward go to GPU Pauli frames and other static circuits go
  to the GPU state vector; otherwise the call falls back to the CPU and
  `note` says why.
  `SampleResult.method` reports what ran: `gpu:statevector:fp32`,
  `gpu:statevector:fp64`, `gpu:pauli_frame` (or `gpu-emulator:*`).
* `qumo/exec/orchestrator.mojo`: jobs that HEFT places on a `Device.gpu(...)`
  run on the GPU when one is available. Otherwise they run on the CPU and
  `JobResult.backend` ends with `(gpu fallback: <reason>)`.
* MCP / CLI tools (`qumo/exec/tools.mojo`): `qumo_gpu_info` and
  `qumo_run_device`. The latter is `qumo_run_qasm` plus
  `"device": "auto"|"cpu"|"gpu"` and warns with `QM-W020` on fallback.
  `qumo_run_qasm` itself is unchanged.
* Python (`python/qumo_py.mojo`): `sample(circ, shots, seed)` is unchanged.
  New: `gpu_info() -> dict` and `sample_device(circ, shots, seed, device) ->
  (counts, method, note)`.

## Compile-time gating

The only way QUMO touches a GPU toolchain is `-D QUMO_GPU`:

```mojo
from std.sys.defines import is_defined
comptime if is_defined["QUMO_GPU"]():
    ctx.enqueue_function[gk_1q[dt]](...)   # compiled only in GPU builds
```

* **Default builds** (`make test`, `make cli`, `make py`) contain no device
  code. They build and run on any machine. `gpu_compiled()` is `False`,
  `gpu_available()` is `False` without creating a `DeviceContext`, and device
  mode raises `QM-E470`. The emulator still runs everywhere.
* **GPU builds** (`make test-gpu`, `cli-gpu`, `py-gpu`, `bench-gpu`) compile
  the kernels for the build's target accelerator: the host GPU by default, or
  any GPU with `--target-accelerator=...`.
* `make gpu-check` cross-compiles every kernel for `sm_90` (PTX), `gfx942`
  (AMDGCN) and `metal:4` (Metal IR, including Metal's IR verifier). It needs
  no GPU and fails if any target fails or if no kernel is emitted (that would
  mean the gating is broken).

## Enabling the GPU

| Platform | Requirement | Then |
|---|---|---|
| macOS, Apple silicon | Xcode (not only CommandLineTools) **and** `xcodebuild -downloadComponent MetalToolchain` | `make test-gpu`, `make bench-gpu` |
| Linux, NVIDIA | NVIDIA driver with CUDA support for the GPU (MAX bundles the compiler) | same |
| Linux, AMD | ROCm-supported GPU and driver (MI200/MI300 class, recent Radeon) | same |

Without the Metal toolchain, launching any kernel on a Mac fails with "Metal
Compiler failed to compile metallib". Check a build with `qumo tool
qumo_gpu_info '{}'` (CLI built with `make cli-gpu`) or
`qumo_py.gpu_info()`.

## Vendor and precision matrix

| Vendor (target) | State vector | Pauli frames | Default precision | fp64 |
|---|---|---|---|---|
| NVIDIA (`sm_90`, ...) | fp32 + fp64 kernels | yes | fp64 | yes |
| AMD (`gfx942`, ...) | fp32 + fp64 kernels | yes | fp64 | yes |
| Apple (`metal:4`) | fp32 kernels | yes | **fp32** | **no** (Metal has no double) |

`GpuStateVector[dtype = GPU_DEFAULT_DTYPE]` takes precision as a comptime
parameter:

* Apple: always float32. Metal rejects `double` (verified with the
  `metal:4` IR verifier). Asking for `GpuStateVector[DType.float64]` on
  Apple raises `QM-E472`.
* NVIDIA/AMD: float64 by default, because it matches the CPU simulator to
  about 1e-12. `-D QUMO_GPU_FP32` makes float32 the default (twice the qubits
  per GiB, higher throughput). `GpuStateVector[DType.float32]` can always be
  requested explicitly.
* This mirrors CUDA-Q, whose `nvidia` target defaults to fp32 with an fp64
  option.
* Host-side combination is always fp64: amplitudes are converted to fp64 in
  `to_host()`, and reduction partials and sampling CDFs are summed in fp64 on
  the host.
* The Pauli-frame kernel uses no floating point at all (see below), so it is
  identical on every vendor.

Memory: `2^n * 8 B` (fp32) or `2^n * 16 B` (fp64). For example, 30 qubits
need 8 GiB in fp32. The hard cap is the CPU simulator's 32 qubits
(`check_statevector_size`).

## State-vector kernels

The storage is the CPU simulator's: split re/im arrays (structure of arrays),
little-endian qubit order. The execution plan is the CPU runner's:
`qumo.sim.runner.build_plan` fuses every run of 1-qubit gates on a wire into
one 2x2 (gate fusion as in qsim and Qiskit-Aer). Each `KOp` maps onto one
kernel launch:

| Plan op | Kernel | Work items (threads) |
|---|---|---|
| fused 1q (dense / diagonal / X) | `gk_1q`, `gk_diag` (skips the `|0>` half when d0 = 1), `gk_x` (pure permutation) | 2^(n-1) pairs |
| CX, CZ/CP, controlled-1q, SWAP, dense 4x4 | `gk_cx`, `gk_cphase`, `gk_c1`, `gk_swap`, `gk_2q` | 2^(n-2) quads |
| RZZ, global phase | `gk_rzz`, `gk_phase` | 2^n amplitudes |
| CCX, CCZ, CSWAP | `gk_ccx`, `gk_ccz`, `gk_cswap` | 2^(n-3) octets |
| init, probabilities | `gk_init`, `gk_prob` | 2^n |
| Pauli term sum, norm | `gk_expect` (shared-memory tree) | <= 1024 blocks |
| sampling | `gk_segsum` (block per 4096-amplitude segment), `gk_sample_seg` (thread per shot) | |

Design rules:

* **Thread body plus wrapper.** Each kernel body is an `@always_inline`
  function `th_*(gid, work, pointers, params)`. `gk_*` is a one-line wrapper
  that passes `global_idx.x`. The host emulator calls the same body. Tail
  threads (`gid >= work`) return, exactly as on the GPU.
* **Pair-index addressing** (`ins0`, as in the CPU's `insert_zero`). Thread
  `g` owns the amplitudes obtained by inserting 0 bits at the target
  positions of `g`. Threads therefore touch disjoint amplitudes: no atomics
  and no races. Loads coalesce whenever the lowest target qubit is at least
  log2(warp size).
* **Launch shape:** 256 threads per block, grid = ceil(work / 256), 1-D. This
  is the scheme of cuStateVec's and Aer-GPU's elementwise kernels.
* **Launches** are asynchronous on one in-order stream. The host
  synchronises only when it reads results (`to_host`, `probabilities`,
  `expectation`, `sample`). Gate matrices travel by value as `SIMD[dt, 4/16]`
  kernel parameters, so there are no per-gate uploads. Kernel parameters are
  DevicePassable only: pointers and fixed-width scalars, never `Int`.
* **Expectation values:** one `gk_expect` launch per Pauli term. Each thread
  computes a grid-stride partial of `sum_i conj(psi[i^x]) psi[i]
  (-1)^{|i&z|}`, followed by a 256-wide shared-memory tree (barrier per
  level, the classic CUDA "reduction #3"), giving one partial per block.
  After one device-to-host copy for all terms, the partials are summed in
  fp64 on the host and the term's `i^{|x&z|}` factor is applied. Order is
  fixed and no atomics are used, so results are reproducible run to run.
* **Sampling** never copies the state. `gk_segsum` reduces the probability
  of each 4096-amplitude segment. The host builds the fp64 prefix over the
  2^n/4096 segment sums and maps the sorted QUMO-RNG uniforms to (segment,
  residual). `gk_sample_seg` (one thread per shot) then scans its segment.
  The RNG stream use is the CPU's (sorted uniforms, then the same
  Fisher-Yates shuffle), so for a seed the fp64 GPU sampler reproduces
  `qumo.sim.runner.sample` bit for bit (tested). Rounding fallback is the
  last index with nonzero probability: a zero-probability outcome is never
  returned.

## Pauli-frame sampler for QEC (`GpuFrameSampler`)

This is the GPU counterpart of `qumo.sim.frame.FrameSimulator`, which
implements Stim's Pauli-frame method:

* It uses the same lowered `FrameProgram` (frame primitives plus the tableau
  reference sample). It returns the same `DetectorSamples` and
  `MeasurementSamples` types with identical bit layout, so decoders consume
  GPU samples unchanged.
* **One thread per 64-shot word. The whole circuit is ONE launch:** each
  thread interprets the op table, which is uploaded once. Because every
  thread reads the same op, loads are broadcasts and control flow is uniform
  across a warp.
* Frame rows are stored as `row * nw + word`, so consecutive threads access
  consecutive words (coalesced). Large runs are chunked to at most 256 MiB
  of frame scratch per launch.
* **Integer-only noise.** Like the CPU sampler, noise uses geometric
  skipping, but the host precomputes `T_k = floor(2^32 (1-p)^k)` for
  k = 1..64. A skip is then the largest k with `u < T_k`, found by binary
  lifting in 7 compares. Pauli choices use Lemire's multiply-shift and
  32-bit thresholds. No floating point runs on the device: no fp64 is needed
  on Apple, and results are bit-identical across NVIDIA, AMD, Apple and the
  emulator. The cross-compiled kernel uses 60 VGPRs with no scratch on
  gfx942, and has no local-memory spills in the PTX.
* **Randomness:** word `g` uses `Rng(seed).split(g)` (xoshiro256**). Samples
  depend only on (circuit, shots, seed), not on launch grid, chunking or
  device. They are not bit-identical to the CPU sampler, which uses one
  stream per 512-shot batch. The two agree in distribution, and exactly
  whenever outcomes are deterministic.

## Testing without a GPU (emulation)

MAX's `DeviceContext(api="cpu")` does **not** execute `enqueue_function`
kernels, so QUMO tests kernels by host emulation. `qumo.gpu.emulate` runs a
thread body for every thread index of the real grid, `ceil(work/256)*256`
threads including the idle tail.

* Elementwise kernels are race-free by construction, so emulating threads
  in parallel CPU chunks is exact.
* Reductions emulate the shared-memory tree structure: a load phase for all
  threads of a block, then each tree level for all threads, with the GPU's
  `barrier()` becoming the end of a level loop.
* `GpuStateVector.emulator(n)` and `GpuFrameSampler(c, emulate=True)` run
  all host logic too: plan lowering, launch geometry, partial buffers,
  segment sampling, op tables, chunking and result scatter.

This follows the "functional simulation of kernels on the host" practice of
CUDA's old emulation mode and HIP-CPU.

`tests/test_gpu.mojo` (state vector, emulated in fp64 and fp32) checks:

* every gate kernel on operands that include qubits 0 and n-1, plus
  single-work-item states (grid tails);
* random circuits over every opcode family up to 21 qubits;
* one launch per plan op (no hidden host work);
* probabilities and norm;
* Pauli expectation values, and the raw tree-reduction sums at 2^23
  amplitudes (1024 blocks with 32 grid-stride items per thread) against a
  sequential Kahan sum;
* sampling: fp64 matches the CPU sampler exactly for a seed, shots are
  reproducible, and zero-probability states are never returned;
* error codes, dispatch, the orchestrator and the MCP tools.

`tests/test_gpu_frame.mojo` (frames) checks:

* Bernoulli rates and lane uniformity of the integer skip tables;
* noiseless codes produce zero detection events;
* probability-1 Pauli injections (X/Y/Z_ERROR, PAULI_CHANNEL_1) at random
  places in surface_z, surface_x and repetition circuits: identical to the
  CPU sampler for every shot;
* circuit-level noise: per-detector and logical rates match the CPU within
  5 sigma;
* feed-forward (`c_if` X/Y), reset, random and Bell measurements, and the
  reference XOR;
* chunk-invariance and seed reproducibility;
* QM-E701, QM-E470 and `sample_on(..., "gpu")`.

Mutation checks were run during development: dropping CX's Z
back-propagation, or an off-by-one in the skip lookup, each fails the suite.

In a `-D QUMO_GPU` build with a device, both files also run the device
suites. The frame test additionally asserts **device == emulator bit for
bit** on a d=5 surface code with 100k shots.

## Dispatch threshold

`GPU_MIN_QUBITS = 20`. Below about 2^20 amplitudes (8 MiB in fp32), a gate
costs a few microseconds on the CPU (SIMD, threads, cache blocking), which
is comparable to one kernel launch. The one-off `DeviceContext` creation and
module load is then not amortised. This is a heuristic and **not yet tuned
on hardware**: run `make bench-gpu` on the target machine and adjust it.
Automatic dispatch never uses GPU Pauli frames, because the CPU frame
sampler is already SIMD and multi-threaded and there is no GPU measurement
yet. They are used only when a GPU is requested explicitly (`device="gpu"`,
orchestrator GPU devices).

## Benchmarks

```bash
make bench-gpu                # builds with -D QUMO_GPU; runs 16..26 qubits + frames
./build/gpu/bench_gpu 28 5    # nmax, repetitions (medians, CPU/GPU interleaved)
```

The state-vector table uses the layered circuit of `bench/bench_sv.mojo`. It
compares CPU `evolve` against GPU `reset + evolve + synchronize` on a
preallocated state, plus GPU expectation and 10k-shot sampling time, and
fidelity against the CPU. The frame table compares the CPU `FrameSimulator`
against `GpuFrameSampler` on surface-code memory circuits (d = 3, 5, 7;
p = 0.001; 1M shots).

Without `-D QUMO_GPU` the binary runs a small emulator smoke check; its
timings are not GPU timings.

## Error codes

| Code | Meaning |
|---|---|
| QM-E470 | GPU support not compiled into this build (rebuild with `-D QUMO_GPU`) |
| QM-E471 | Dynamic or noisy circuit given to the GPU state-vector sampler |
| QM-E472 | float64 requested on an Apple GPU |
| QM-E473 | Sampling a zero-norm state |
| QM-E474 | Circuit or host state wider than the GPU state |
| QM-E475 | Plan op without a GPU kernel |
| QM-E940 | Unknown device name (`auto`, `cpu`, `gpu`) |
| QM-W020 | `device: "gpu"` requested but the run fell back to the CPU (reason given) |

## Ideas adopted, and from where

* **CUDA-Q:** precision as a target property (fp32 default, fp64 option);
  heterogeneous `mqpu`-style device pools (the orchestrator); explicit
  fallback reporting.
* **cuStateVec:** 1-D launches over pair and quad indices; per-gate kernels
  with matrices passed by value; sampling via probability prefix sums over
  segments and a per-shot search; host-side fp64 accumulation of
  reductions.
* **Qiskit-Aer GPU / qsim:** gate fusion before execution, so one fused 2x2
  per wire run becomes one launch (shared with the CPU path); the same
  execution plan for CPU and GPU.
* **Stim** (Gidney 2021): Pauli frames, reference sample, Z-gauge
  re-randomisation, geometric skipping. **Lemire** (2019): multiply-shift
  bounded integers.

## Limits and next steps

* Not yet executed on real GPUs (no toolchain on the development machine).
  Expect a tuning pass: launch threshold, block size, and coalescing for
  low target qubits.
* One launch per plan op. There is no shared-memory multi-gate blocking
  (cuStateVec / qsim "fused kernels over k qubits"); the CPU runner's
  cache-blocking reorder is not used on the GPU.
* Single GPU, single stream. No multi-GPU state partitioning (cuStateVec
  distributed index bits) and no host-memory offload.
* The GPU state-vector sampler handles static circuits only. Dynamic and
  noisy non-Clifford circuits use CPU trajectories.
* Expectation values use one launch per Pauli term. Very large Hamiltonians
  would benefit from batching terms per launch.
* Sampling sorts the uniforms on the host (O(shots log shots)); each shot
  thread scans up to 4096 amplitudes.
* The frame sampler keeps rows in global memory (L1/L2 cached). Very
  large codes could tile qubits through shared memory.
