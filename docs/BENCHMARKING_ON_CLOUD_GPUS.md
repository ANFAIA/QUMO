# Benchmarking QUMO on cloud GPUs (NVIDIA, AMD, Apple)

QUMO's GPU code is vendor-neutral Mojo (the MAX `DeviceContext` API, see [GPU.md](GPU.md)). This page explains
how to benchmark it on GPUs you rent or own, and how to compare results across machines:

* **Google Colab** (NVIDIA T4 / L4 / A100 / H100 / G4): `examples/colab/QUMO_GPU_Benchmark_Colab.ipynb`
* **AMD GPUs on any ROCm Jupyter host** (AMD Developer Cloud MI300X, Azure ND MI300X v5, a local
  Radeon / Instinct box): `examples/colab/QUMO_GPU_Benchmark_AMD.ipynb`
* **Any Linux GPU host or a Mac (Metal), headless:** `examples/colab/run_on_any_gpu.sh`
* **One runner behind all three:** `bench/run_benchmarks.py` + presets in `bench/configs/`
* **Comparison of several runs:** `examples/colab/compare_results.py`
* **CUDA-Q kernel count per call (optional, NVIDIA + Nsight Systems):** `examples/colab/count_gpu_kernels.py`

The benchmark treats NVIDIA, AMD and Apple alike: presets are named after hardware, `--config auto` applies one
rule to every vendor, the same circuits and timing rules apply everywhere, and every vendor gets the strongest
third-party baseline that runs on it.

## 1. Support matrix (sources checked 2026-10-12)

### Where QUMO's GPU kernels run (Mojo 1.1 / MAX 26.6)

| Vendor | GPUs (architecture, Mojo target) | Modular tier | Requirement |
|---|---|---|---|
| NVIDIA | B200 (Blackwell, `sm_100`) | continuously tested | driver >= 580 |
| NVIDIA | B100/B300 (`sm_100`/`sm_103`), RTX 50xx (`sm_120`), DGX Spark (`sm_121`), Jetson Thor (`sm_110`), H100/H200 (`sm_90`), L4/L40 and RTX 40xx (`sm_89`), A100 (`sm_80`), A10 and RTX 30xx (`sm_86`), Jetson Orin (`sm_87`), **T4** and RTX 20xx (`sm_75`) | known compatible | driver >= 580, or `MODULAR_NVPTX_COMPILER_PATH=<CUDA ptxas>` with an older driver |
| NVIDIA | Maxwell / Pascal (`sm_52`, `sm_60`, `sm_61`: GTX 9xx/10xx, P100, P4/P40) | not supported out of the box | Mojo has these targets but needs a system `ptxas` (`MODULAR_NVPTX_COMPILER_PATH`); the runner skips QUMO's GPU suites there (set `QUMO_FORCE_GPU=1` to try) |
| NVIDIA | Volta (V100, `sm_70`), Kepler (K80, `sm_37`) and older | not supported | no Mojo target at all; the runner skips QUMO's GPU suites; the CPU suites still run, and the GPU baselines wherever their own CUDA wheels still support the card |
| AMD | MI355X (CDNA4, `gfx950`), MI300X (CDNA3, `gfx942`) | continuously tested | AMD GPU driver >= 6.3.3; ROCm >= 7.0 for MI355X |
| AMD | MI325X (`gfx942`), MI250X (`gfx90a`), Radeon RX 9070/9060 (`gfx1201`/`gfx1200`), RX 7900/7800/7700/7600 (`gfx1100`-`gfx1102`), Radeon 780M/880M/890M/860M/8060S iGPUs (`gfx1103`, `gfx1150`-`gfx1152`), RX 6900 (`gfx1030`), Steam Deck (`gfx1033`) | known compatible | same |
| Apple | M1-M5 (Metal) | known compatible | macOS 15+, Xcode 16+, `xcodebuild -downloadComponent MetalToolchain`; **fp32 only** (Metal has no double) |

Sources: [Mojo system requirements, GPU tables](https://mojolang.org/docs/requirements/) ("Pre-Turing GPUs (such as
Pascal-generation GTX 10XX and Tesla P100) are not supported out of the box") and the Mojo 1.1 target list
(`_all_targets` in `std/_gpu/host/info.mojo`: `sm_52`, `sm_60`, `sm_61`, `sm_75` and newer; no `sm_70`;
`apple-m1` ... `apple-m5`, which Mojo also accepts as `metal:1` ... `metal:5`). `bench/run_benchmarks.py`
encodes this table in `gpu_supported()` and stores the verdict for every detected GPU in the result JSON
(`environment.gpus[].mojo_max_supported`, `support_note`).

### Installing Mojo / MAX on a cloud machine

* The notebooks and `run_on_any_gpu.sh` install **pinned** versions into `.venv` (the ones QUMO is developed with):
  `uv pip install mojo==1.1.0 max-mojo-libs==26.6.0 max-core==26.6.0` (empty version fields = `make setup`, i.e.
  the latest releases).
* PyPI wheels (uploaded 2026-09-17): `mojo 1.1.0`, `mojo-compiler 1.1.0` and `max-core 26.6.0` for
  `manylinux_2_34_x86_64`, `manylinux_2_34_aarch64` and `macosx_13_0_arm64`; `max-mojo-libs 26.6.0` is pure Python.
  Linux therefore needs **glibc >= 2.34** (Ubuntu 22.04+). Modular lists Python 3.10-3.14, x86-64-v3 (Haswell or
  newer) or ARM Neoverse N1+ CPUs, macOS 15+ on Apple silicon, and a C compiler on Linux (used as the linker).
  Sources: [pypi.org/project/mojo](https://pypi.org/project/mojo/), [pypi.org/project/max-core](https://pypi.org/project/max-core/),
  [MAX system requirements](https://max.modular.com/packages) (target of docs.modular.com/max/packages).
* MAX bundles its own GPU compiler: no CUDA toolkit or HIP SDK is needed to build QUMO's kernels, only the NVIDIA
  driver (>= 580) or the AMD GPU driver stack. `make gpu-check` cross-compiles for `sm_75`, `sm_80`, `sm_90`, `gfx942` and `metal:4`
  without any GPU; the runner's `gpu_check` suite adds the host architecture (e.g. `sm_75` on a T4, `gfx942` on an
  MI300X, `metal:5` on an Apple M5: the chip generation comes from `sysctl machdep.cpu.brand_string`).

### Google Colab

* GPU runtimes and rates (measured March 2026): T4 1.19, L4 1.71, A100-40GB 5.40, A100-80GB 7.52, G4 (RTX PRO 6000
  Blackwell, ~96 GB) 8.71 compute units/hour; H100 not measured. 100 units cost about US$10, so roughly $0.12/h
  (T4) to $0.87/h (G4). Usable GPU memory: T4 15 GB, L4 22.5 GB. Source:
  [McCormick, Colab GPUs features & pricing (updated March 2026)](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/).
* Runtime image 2026.07: Ubuntu 22.04.5 (glibc 2.35 >= 2.34), Python 3.12.13
  ([Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html)). The notebook's
  kernel only needs the standard library; Mojo/MAX and the baselines get their own `uv` environments.
* Driver: a July 2026 free-tier T4 session reported a driver supporting CUDA 13.0 and a CUDA 12.8 toolkit
  ([datawookie, Running LLMs in Google Colab](https://datawookie.dev/blog/2026-07-31-running-llms-google-colab/)).
  CUDA 13.x requires driver >= 580 ([CUDA toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)),
  so that runtime met Mojo's driver requirement. Colab changes images without notice, so the notebook reads
  `nvidia-smi` at run time, picks the CUDA-Q wheel for the driver's CUDA major version, and sets
  `MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas` if the driver is older than 580.
* Colab has **no AMD GPUs**.

### NVIDIA baselines

* **CUDA-Q** 0.16.0.post1 (2026-09-17, Python >= 3.11). `pip install cudaq` is an sdist meta package that detects
  the CUDA version (NVRTC, CUDA runtime or NVML) and pins `cuda-quantum-cu12` or `-cu13`; the notebook and the script
  install the matching wheel directly, chosen from the driver's CUDA major version (manylinux_2_28 x86_64 / aarch64,
  cp311-cp314; the wheels pull their CUDA runtime from pip, so only the driver must support that CUDA major).
  GPU acceleration is Linux-only; macOS arm64 builds (e.g. `cuda-quantum-cu13 0.16.0`) simulate on the CPU. The
  `nvidia` target uses cuStateVec; `cudaq.set_target("nvidia", option="fp64")` selects double precision, `fp32` single.
  The target's own default is fp32 (`runtime/cudaq/platform/default/nvidia.yml` L21-32 at tag 0.16.0, where `fp32` is
  marked `default: true`); the runner always passes the option explicitly: `fp32` by default, `fp64` too with
  `--fp64` (see [Precision](#precision-fp32-by-default-fp64-on-request)).
  `qpp-cpu` is the CPU target. In 0.16 the `nvidia` target hands the gate queue to cuStateVec Ex ("SVUpdater"),
  which fuses gates and applies each fused matrix in one pass over the state; the fusion widths are chosen by the
  library unless `CUDAQ_FUSION_MAX_QUBITS` / `CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS` are set (0 = gate by gate)
  ([PR #5025](https://github.com/NVIDIA/cuda-quantum/pull/5025); `CuStateVecConfig.cpp` L146-169 at 0.16.0). Sources: [CUDA-Q quick start](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html),
  [CUDA-Q state-vector simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html),
  [pypi.org/project/cudaq](https://pypi.org/project/cudaq/), [pypi.org/project/cuda-quantum-cu13](https://pypi.org/project/cuda-quantum-cu13/).
* **Qiskit Aer GPU**: `qiskit-aer-gpu` 0.15.1 (2024-09-13) is the latest wheel: Linux x86_64 only, cp38-cp312, CUDA 12
  runtime and cuQuantum from pip. It declares `qiskit>=1.1` without an upper bound, but the 0.15 code base does not
  import under Qiskit 2.x (checked here with the CPU build `qiskit-aer==0.15.1` + Qiskit 2.5.2: `ImportError:
  cannot import name 'convert_to_target'`; it works with Qiskit 1.4.6). So it gets its own environment with
  `qiskit>=1.1,<2`. Sources: [pypi.org/project/qiskit-aer-gpu](https://pypi.org/project/qiskit-aer-gpu/),
  [Qiskit Aer](https://github.com/Qiskit/qiskit-aer/).
* **PennyLane Lightning** `lightning.gpu` (cuStateVec; installed by default, the third-party GPU baseline that AMD
  GPUs also have, as `lightning.amdgpu`): `pennylane-lightning-gpu` 0.45.0, manylinux_2_28
  x86_64 / aarch64, Python >= 3.11 ([pypi.org/project/pennylane-lightning-gpu](https://pypi.org/project/pennylane-lightning-gpu/)).

### AMD GPUs and baselines

* **Where:** the [AMD Developer Cloud](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html)
  (DigitalOcean GPU droplets with 1 or 8 MI300X; US$2.59 per GPU-hour list price in October 2026,
  [DigitalOcean AMD pricing](https://docs.digitalocean.com/products/amd/details/pricing/)). Its Quick Start images
  start JupyterLab inside a Docker container (Ubuntu 24.04, ROCm 6.4.0 as of July 2026); the plain ROCm image runs it
  on the base OS ([DigitalOcean: JupyterLab - AMD MI300x](https://docs.digitalocean.com/products/marketplace/catalog/jupyterlab-amd-mi300x/)).
  Prefer the plain ROCm image (or make sure `git`, `make` and a C compiler exist in the container). Also
  [Azure ND MI300X v5](https://ir.amd.com/news-events/press-releases/detail/1198/amd-instinct-mi300x-accelerators-power-microsoft-azure-openai-service-workloads-and-new-azure-nd-mi300x-v5-vms)
  (8x MI300X per VM) or any local ROCm machine with `pip install jupyterlab`.
* **Baselines:** CUDA-Q and `qiskit-aer-gpu` are CUDA-only. The AMD-native GPU state-vector baseline is **PennyLane
  `lightning.amdgpu`** (`pennylane-lightning-amdgpu` 0.45.0, manylinux_2_28 x86_64, Python >= 3.11): "pre-built wheels
  are available for MI300 series GPU for ROCm 7.0 and greater"; older GPUs or ROCm versions (for example ROCm 6.4 on
  the Quick Start images) need a source build with Kokkos (`-DKokkos_ARCH_AMD_GFX942=ON` and friends). If the wheel
  does not load, the suite records a skip. Sources: [pypi.org/project/pennylane-lightning-amdgpu](https://pypi.org/project/pennylane-lightning-amdgpu/),
  [Lightning-AMDGPU installation](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_amdgpu/installation.html).
  A ROCm port of Qiskit Aer exists as research ([CHEP 2024](https://www.epj-conferences.org/10.1051/epjconf/202429511022));
  there is no `qiskit-aer-gpu-rocm` package on PyPI (checked 2026-10-12), so the runner does not install one. If you
  have a ROCm build of Aer, point `QUMO_AER_GPU_PYTHON` at its interpreter and add `"gpu"` to `baseline_aer.devices`.

## 2. What is measured

### Suites (`bench/run_benchmarks.py --list`)

| Suite | What runs | Rows |
|---|---|---|
| `gpu_check` | `make gpu-check` for the host GPU's architecture (plus any listed targets) | PASS/FAIL per target x file, kernel count |
| `sv_cpu` | `build/bench_sv` (QUMO CPU, fp64): `layered`, optionally `qft` and `random` | seconds, gates/s per width |
| `sv_gpu` | `build/gpu/bench_gpu_fp32` (`-D QUMO_GPU`: fp32 is QUMO's default) and, with `--fp64`, `_fp64` (`-D QUMO_GPU -D QUMO_GPU_FP64`); a checkout from before 2026-10-13 gets its own flags (`-D QUMO_GPU_FP32` for fp32, none for fp64), read from `qumo/gpu/device.mojo` | GPU seconds (+ per-repetition times), launches (state sweeps) per evolve, gates per launch, effective GB/s, CPU seconds, speed-up, expectation and 10k-shot sampling time, fidelity vs CPU (up to `cpu_max` qubits); plus the QEC Pauli-frame table (surface code d = 3, 5, 7; 1M shots). Optional labelled `variants` (extra `-D` defines in their own binary, or environment variables) |
| `sv_gpu_probe` | `bench_gpu <n> <reps> probe fp32[,fp64]` once (one binary measures the run's GPU precisions; fp64 never on Apple; an older `bench_gpu` takes no list and measures fp32 and, where the device has it, fp64; widths n-2 and n), only when the checkout has the mode; otherwise a recorded skip | one row of kind `gpu_probe` per `PROBE,<dtype>,<n>,<metric>,<value>,<unit>` line: device facts (multiprocessors, shared memory, warp width, compute capability; `n/a` where the API has none, e.g. on Metal), empty-sweep time, per-op-kind cost slopes, transpose cost, 32 / 64 / 128 B chunked-stream bandwidth, kernel registers / spills / occupancy, and evolve / (runs x sweep) of the three families |
| `baseline_aer` | Qiskit Aer statevector: CPU fp64 (same families as `sv_cpu`) and fp32 (`layered`, the like-for-like reference of the fp32 GPU series), GPU in the run's GPU precisions (fp32; + fp64 with `--fp64`; `layered`); optional labelled `variants` (other `AerSimulator` options) | seconds per width, Aer's own `time_taken`, the fusion Aer reports it applied |
| `baseline_cudaq` | CUDA-Q `nvidia` target in the run's GPU precisions (`option="fp32"`; + `"fp64"` with `--fp64`; optional `qpp-cpu`); optional labelled `variants` (`CUDAQ_*` environment, e.g. no fusion) | seconds per width, `CUDAQ_*` fusion settings, compile-time gate count |
| `baseline_lightning` | PennyLane `lightning.qubit` (fp64) / `.gpu` / `.amdgpu` (the run's GPU precisions) | seconds per width |
| `neutral` | a small subset of the hardware-neutral compiler benchmark (`bench/neutral`): QUMO, Qiskit, TKET on superconducting, trapped-ion and neutral-atom targets | per SDK: status counts, median compile time, 2q gates, ESP |
| `qec` | `bench/bench_qec.py --part A` with few shots (QUMO vs Stim + PyMatching) | the markdown tables, parsed |
| `tests` | `make test` / `make test-gpu`, or a list of test files | PASS/FAIL per file |

### The circuits

All state-vector engines run the `layered` circuit of `bench/bench_sv.mojo` (and `bench/bench_gpu.mojo`): 10 layers;
in layer `d` every qubit `q` gets `ry(0.1 + 0.01 (q + d))` then `rz(0.2 + 0.03 q d)`, followed by `cx(q, q+1)` for
`q = d mod 2, d mod 2 + 2, ...`. The Python baselines build it from one function (`layered_ops`), and every row
stores its gate count, so a mismatch would show (QUMO, Aer, CUDA-Q and Lightning all report 395 gates at 16
qubits). Aer CPU also runs the `qft` and `random` families of `bench_sv.mojo` when the preset lists them, using
the gate-for-gate twins in `bench/bench_sv_aer.py`. Widths run on the grid 16, 18, 20, ... like the Mojo benchmarks.

### Correctness before speed

* QUMO GPU: `bench_gpu` reports the fidelity of the GPU state against the fp64 CPU simulator at every width up to
  `sv_gpu.cpu_max`, and the expectation-value error; the runner flags fidelity < 1 - 1e-4. Wider widths are
  GPU-only (fidelity recorded as `null`).
* Baselines: before timing, each worker computes the 8-qubit state and compares it with an independent numpy
  reference of the circuit (`reference_state`); the fidelity and the engine's qubit order (little- or big-endian)
  are stored under `versions[].verify`.
* `tests` with `test-gpu` runs QUMO's GPU test suites on the device (device == emulator == CPU checks).

### Timing rules

The goal is that every engine is timed the same way, and that every difference that remains is recorded in the
result instead of hidden. The rules below are those of the harness of 2026-10-13 (see
[What changed on 2026-10-13](#what-changed-on-2026-10-13-timing-fairness)).

* **Same circuit.** Every row stores its gate count, and the runner checks that all layered rows of a width have
  the same one (`checks` in the JSON, printed in the summary and the report). CUDA-Q 0.16 also optimises kernels at
  compile time by default; the worker records `cudaq.estimate_resources`' gate count for the 8-qubit check
  (`versions[].compiled_gate_count`), so a changed gate sequence would show.
* **One warm-up per width, for every engine:** QUMO GPU (one untimed `evolve` per width in `bench_gpu`, plus the
  read-out kernels before they are timed), CUDA-Q (one `get_state` per width: JIT, memory-pool growth), Aer (one run
  per width), Lightning (one call per width). QUMO CPU: `bench_sv` has no warm-up of its own, so the runner starts its
  sweep one width lower and drops that row (a warm-up of the process: thread pool, first page faults); every
  repetition then allocates and zero-fills its state, as Aer's does.
* **Repetitions:** `reps` per width (3; `--quick`: 1), reduced the same way for every engine: the upper median
  `sorted(times)[reps // 2]`, the rule of `bench_sv.mojo` and `bench_gpu.mojo`. Every engine's individual times are
  stored (`times`).
* **The timed region ends with the final state on the device and the device synchronised.** Every row carries a
  `timed_region` string with exactly what it contains:

| Engine | Timed (`seconds`) | Not timed | End of the timed region |
|---|---|---|---|
| QUMO CPU (`bench_sv`) | `evolve(circuit)`: state allocation + zero-fill, plan, all gates | - | state in host memory |
| QUMO GPU (`bench_gpu`) | `reset + evolve + synchronize` on a preallocated state: plan, tile scheduling, table upload, all launches (the first launch also writes the all-zero initial state) | device allocation, `DeviceContext` creation, kernel-module load | `synchronize()`; state stays on the GPU |
| CUDA-Q | `cudaq.get_state(kernel, args)` + reading amplitude 0: argument marshalling, state allocation (CUDA memory pool) + zero-state init, gate queue, cuStateVec Ex fusion, simulation | building the angle lists (prepared before the timer), JIT, freeing the state | `get_state` flushes and synchronises; one-element copy; state stays on the GPU (no full copy) |
| Aer | `sim.run(circuit).result()` with `save_amplitudes([0])`: circuit conversion, job thread, allocation + init, Aer fusion, simulation | library import | blocking `result()`; one amplitude to the host; `aer_internal_seconds` = Aer's own `time_taken` |
| Lightning | QNode call returning `<Z0>`: tape, allocation + init, gates, reduction | device creation | the reduction returned to Python |

* **Asymmetries that remain, and where they are recorded:**
  * *Allocation.* CUDA-Q and Aer allocate and initialise the state inside every call (CUDA-Q from a memory pool);
    QUMO GPU's `seconds` does not include the device allocation. `bench_gpu` also times new state buffers on the
    same context + `evolve` + `synchronize` (`gpu_alloc_s`, after the main repetitions); rows carry it as
    `seconds_incl_alloc`, the column to set next to CUDA-Q and Aer. Initialisation itself is inside QUMO's timed
    region either way (the first launch writes `|0>`). The fairness audit estimated the CUDA-Q
    initialisation at about 2 ms of its 170 ms at 26 qubits fp32 on the T4 (one 512 MiB write; an estimate, not
    measured).
  * *Python.* CUDA-Q, Aer and Lightning are called from Python, QUMO from Mojo. Aer's own time is stored next to
    its wall time.
  * *Fusion.* Each engine runs with its own default optimisation, and every row records it as `fusion`: QUMO fuses
    each run of 1-qubit gates into one 2x2 and the GPU engine then applies many plan ops per state sweep
    (`launches`); CUDA-Q rows carry the `CUDAQ_FUSION_*` settings (unset = chosen by cuStateVec Ex); Aer rows carry
    the options set and the fusion Aer reports it applied. Tuned or disabled settings appear only as labelled
    diagnostic series (below), never as an engine's primary series.
  * *Order.* Engines run one after another in separate processes, not interleaved (inside `bench_gpu`, the QUMO GPU
    repetitions run back to back, then the CPU reference; interleaving them shifted the GPU times). Each suite records the machine state before and after it ran
    (`machine_state_before/after`: NVIDIA SM / memory clocks, temperature, power, P-state and throttle reasons;
    macOS Low Power Mode, power source and thermal warnings; load average), and the environment records it at the
    start.
* **Same widths.** `bench_gpu` checks the GPU state against the fp64 CPU simulator only up to `sv_gpu.cpu_max`
  qubits (the preset's `sv_cpu` upper end); wider widths run GPU-only, so QUMO GPU's widths are limited by GPU memory
  alone, like CUDA-Q's (whose state stays on the device; no host RAM is reserved for it). Aer GPU and Lightning GPU
  keep a host-RAM reserve of one state, since they may stage through host memory. On a free-tier T4 that gives QUMO
  GPU and CUDA-Q up to 30 qubits in fp32 and 28 in fp64 (when the card has room), Aer GPU 28. The runner reads
  `bench/bench_gpu.mojo` before passing `cpu_max`: an older `bench_gpu` ignores it, so the runner then keeps the
  host-RAM limit for every width.
* **Diagnostic variants** (labelled series; a failing one is recorded under `variant_problems` and never changes
  the suite's status). The NVIDIA presets add: CUDA-Q without fusion (`CUDAQ_FUSION_MAX_QUBITS=0`,
  `CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS=0`; one cuStateVec call per gate, from 20 qubits up), Aer GPU with
  `fusion_max_qubit=2` and with `cuStateVec_enable=True`; `colab_t4` also a CUDA-Q fusion-width sweep
  (`CUDAQ_FUSION_MAX_QUBITS` = 2 ... 6 at 24 and 26 qubits) and, with `colab_l4` and `colab_a100`, a QUMO GPU
  fp32 build with 16 amplitudes per thread (`-D QUMO_GPU_REG_BITS=4`). A variant can only narrow the width grid
  (`"qubits": [lo, hi]`); `--quick` drops variants.
* **Precision fairness:** compare fp32 with fp32 and fp64 with fp64 only (details in
  [Precision](#precision-fp32-by-default-fp64-on-request)). Every GPU engine runs in fp32 by default, CUDA-Q's
  default and QUMO's; fp64 is one switch. QUMO CPU is fp64; Aer CPU runs fp64 and fp32 in every GPU preset, so both
  GPU precisions have a same-precision CPU reference. The report divides each series by a reference of its own
  precision (`--reference "Aer CPU {precision}"`, the default); a series without one gets a note, never a
  mixed-precision ratio. Apple GPUs only have fp32 (fp64 is recorded as *not applicable*, QM-E472).
* **Memory:** a width runs only if its state fits into `memory_fraction` (60 %) of GPU memory and, where the engine
  needs it, of free host RAM (8 B/amplitude fp32, 16 B fp64; QUMO's CPU reference about 37 B/amplitude). Every
  clamp is recorded (`clamp`, code `QM-E966`).
* **Shared machines:** suites run one after another, never concurrently. Treat runs with a load average above the
  core count, a throttled GPU (`event_reasons` other than idle), or Low Power Mode on as indicative only.
* **Failures are data:** every suite catches failures and timeouts (the whole process group is killed) and records
  `ok`, `partial`, `skipped`, `failed` or `timeout` with a stable code: QM-E960 config, QM-E961 build, QM-E962
  timeout, QM-E963 tool missing, QM-E964 unparsable output, QM-E965 baseline missing, QM-E966 memory clamp, QM-E967 no
  / unsupported GPU or precision, QM-E968 benchmark binary or worker crash, QM-E969 test or check FAIL. Baselines
  or precisions that cannot exist on the hardware (CUDA-Q on AMD, fp64 on Apple) are listed as `not_applicable`,
  not as failures. The JSON is rewritten after every suite, so an interrupted session keeps what finished.

### Precision: fp32 by default, fp64 on request

* **Default: fp32 for every GPU engine** (QUMO GPU, CUDA-Q, Aer GPU, Lightning GPU and the QUMO probe). This is
  CUDA-Q's default (`nvidia` target) and, since 2026-10-13, QUMO's on every GPU ([GPU.md](GPU.md#precision-policy)),
  so the default run compares the engines as users get them. On GPUs with slow fp64 units (T4, L4, A10, RTX:
  1/32-1/64 of the fp32 rate) fp64 multiplies every engine's GPU time (CUDA-Q on the T4 at 26 qubits: 0.170 s fp32,
  1.45 s fp64), so there the fp64 series take most of a GPU suite's time.
* **fp64 is one switch** that applies to every GPU engine at once: `run_benchmarks.py --fp64` (= `--gpu-precisions
  fp32,fp64`; `--gpu-precisions fp64` runs fp64 only), the preset key `"gpu_precisions": ["fp32", "fp64"]`,
  `INCLUDE_FP64` in both notebooks, `run_on_any_gpu.sh --fp64`. The command line wins over the preset; a suite-level
  precision list of an older config is still honoured unless the command line sets the precisions. The fp64 widths
  are the presets' `qubits_fp64` / `qubits_gpu_fp64` ranges.
* **CPU references do not change:** QUMO CPU fp64, Aer CPU fp64 + fp32 (`precisions_cpu`: the fp32 CPU series is the
  like-for-like reference of the fp32 GPU series), Lightning CPU fp64.
* **Recorded:** the document's `precision_policy` (`gpu_precisions`, `source`: `default`, `config`, `cli`), each GPU
  suite's `gpu_precisions` (requested list and source), and in every row `precision` (what ran: QUMO GPU from the
  `method` column `gpu:statevector:fp32|fp64`, CUDA-Q from `get_target().get_precision()`, Aer from its
  `precision` option, Lightning from the device's `c_dtype`) and `precision_requested`. A QUMO GPU row whose
  precision differs from the request is flagged (`QM-E967`); binaries carry a build stamp
  (`build/gpu/<binary>.build.json`) so a binary built with other flags is rebuilt.
* **Compare fp32 with fp32 only** (and fp64 with fp64). On Apple GPUs fp64 is *not applicable* (QM-E472).

### Reading the results

`compare_results.py` (and the notebooks' report) apply four rules:

* **One QUMO GPU engine per comparison.** The environment records a hash of the GPU engine's sources
  (`environment.sources.gpu_engine_sha256`: `qumo/gpu/*.mojo`, the plan in `qumo/sim/runner.mojo` and
  `bench/bench_gpu.mojo`; also `cpu_engine_sha256` and `harness_sha256`), because `git describe` says
  "1750a81-dirty" for any uncommitted change. The report puts QUMO GPU rows of different (or unidentifiable)
  engines in separate tables, marks the cross-machine plot, and says that their differences are not a hardware
  result. Runs of the older harness have no hash: their engine counts as unidentifiable unless the tree was clean.
* **Small widths.** Columns of 20 qubits and fewer are marked: fixed per-call overhead dominates there (on the T4,
  Aer GPU takes 21-27 ms and CUDA-Q 3.5-6.8 ms at 16 qubits, which is not simulation time). Rank simulators at 22
  qubits and above.
* **fp64 on GPUs with slow fp64 units** (T4 `sm_75`, A10 / RTX 30xx `sm_86`, L4 / RTX 40xx `sm_89`, RTX 50xx /
  RTX PRO 6000 `sm_120`, consumer AMD): fp64 runs at 1/32 of fp32 or less, so every engine's fp64 GPU time measures
  fp64 multiply-add throughput, not memory bandwidth. The report says so next to those numbers; they do not carry
  over to A100, H100 or MI300-class GPUs. On the T4, CUDA-Q is 8.5x slower in fp64 than in fp32 at 26 qubits
  (1.45 s vs 0.170 s).
* **Sweeps and bandwidth of QUMO GPU.** When `bench_gpu` reports its launch count, the report shows launches per
  evolve, gates per launch and an effective bandwidth (2 x launches - 1) x 2^n x B / seconds, with B = 8 bytes per
  amplitude in fp32 and 16 in fp64: each tile run reads and writes the state once, the first run only writes. It is
  a model from the launch count, not a hardware counter, and `seconds` includes planning and uploads; close to the
  GPU's memory bandwidth means memory bound, far below means the work inside the tiles dominates.
  `examples/colab/count_gpu_kernels.py` (step 9 of the Colab notebook, optional) counts the kernels one CUDA-Q
  `get_state` launches under Nsight Systems (difference of two profiles with 1 and 2 repetitions), an upper bound
  on its state sweeps.

### What changed on 2026-10-13 (timing fairness)

An audit of the first Colab T4 run (`bench/results/cloud/qumo_bench_colab_t4_tesla-t4_20261012-235508.json`) found
the T4 timings mostly fair from 22 qubits up, but:

* **The T4 and M5 QUMO GPU rows came from different engines.** The T4 ran commit 1750a81's engine (one kernel launch
  per plan op: 377 plan ops, 378 launches for the 26-qubit layered circuit), the M5 "current tree" run the tile
  engine (17 launches at 26 qubits, docs/GPU.md); both files say "1750a81 (dirty)". At 26 qubits fp32 the old engine
  took 1.49 s, 3.95 ms per launch, i.e. about 230-270 GB/s (70-85 % of the T4's 320 GB/s) depending on whether its
  CX kernel touched the whole state or half of it: that engine was at the bandwidth floor of its design. No
  T4-vs-M5 conclusion about QUMO GPU is possible until the T4 is re-run with the tile engine. Fix: engine hash,
  report rules above, the caveat in `examples/colab/plot_t4_vs_m5.py`.
* **QUMO had no 28-qubit point while CUDA-Q and Aer GPU had one** (its CPU reference reserved host RAM at every
  width). Fix: `cpu_max`.
* **Aer was warmed up only at its smallest width; QUMO CPU not at all.** Fix: per-width warm-ups (QUMO CPU: process
  warm-up, see above).
* **CUDA-Q built its angle lists inside the timed region and ended it with a `try/except` that could hide a failed
  read.** Fix: arguments prepared before the timer; how the timed region ends is decided once and recorded
  (`completion`, `state_on_gpu`).
* **Nothing recorded what the baselines fused, which `CUDAQ_*` variables were set, GPU clocks or throttling, Low
  Power Mode on Apple, or per-repetition times.** Fix: `fusion`, `cudaq_env`, `aer_fusion`, `machine_state_*`,
  `times` on every row; the source hashes.
* **CUDA-Q's ~40-45 state sweeps at 26 qubits are an inference** from its T4 time (0.170 s against 3.95 ms per sweep
  for the old QUMO engine), not a measurement. The no-fusion and fusion-width variants and the optional Nsight
  count test it on the next Colab run.

### Hardware neutrality checklist

* No default vendor. `--config auto` applies one rule to every vendor: a known model name (or, for AMD CDNA3/4,
  `gfx942`/`gfx950` with >= 64 GiB) selects its preset; any other GPU of a detected vendor gets that vendor's generic
  preset (`nvidia_generic`, `amd_generic`, `apple_metal`); an unknown vendor or no GPU gives `cpu_only`.
  `python bench/run_benchmarks.py --select-config auto` shows the decision; `--select-config "<GPU name>"` (with
  `--gpu-vendor`, `--gpu-arch`, `--gpu-memory-gib`) evaluates it for any GPU.
* The same runner, circuits, timing rules, memory policy and JSON schema on every vendor; the notebooks call the
  runner for the preset instead of carrying their own copy of the rule.
* Each vendor gets its strongest available baseline: NVIDIA: CUDA-Q (cuStateVec), Aer GPU, Lightning GPU; AMD:
  Lightning AMDGPU; Apple: no GPU-native third-party state-vector simulator exists in these packages, so the CPU
  baselines are the reference.
* The compiler subset compiles for superconducting, trapped-ion and neutral-atom targets with QUMO, Qiskit (IBM)
  and TKET (Quantinuum).
* Environment metadata makes runs comparable: GPU name / arch / memory / driver (CUDA or ROCm version), CPU model
  and usable cores, RAM, OS and glibc, Python, Mojo / MAX versions, git commit and dirty state, Colab release,
  timestamps, load averages. No host names, user paths or credentials are stored (git remotes are stripped of
  credentials, local paths are rewritten to repository-relative ones, an output folder outside the repository to
  `<out>`, the home directory to `~`).

## 3. Presets (`bench/configs/`)

Every preset sets `"gpu_precisions": ["fp32"]`; the fp64 widths below apply with `--fp64`.

| Preset | Hardware | QUMO GPU widths (fp32 / fp64) | Baselines (GPU widths fp32 / fp64) |
|---|---|---|---|
| `colab_t4` | T4 16 GB (15 usable), free tier | 16-30 / 16-28 (CPU check up to 26) | Aer CPU (fp64 + fp32) + GPU, CUDA-Q, Lightning GPU (30 / 28); probe on; diagnostic variants: CUDA-Q no fusion + fusion widths 2-6, Aer GPU fusion<=2q + cuStateVec, QUMO GPU REG_BITS=4 (fp32) |
| `colab_l4` | L4 24 GB (22.5 usable) | 16-30 / 16-28 (CPU check up to 28) | same (30 / 28); probe on; variants: CUDA-Q no fusion, Aer GPU fusion<=2q + cuStateVec, QUMO GPU REG_BITS=4 |
| `colab_a100` | A100 40 or 80 GB (widths sized for 40 GB) | 16-30 / 16-30 (CPU check up to 28) | same (30 / 30); Aer CPU also `qft`, `random`; probe on; variants as `colab_l4` |
| `colab_h100` | H100 80 GB (also B100/B200/GH200) | 16-32 / 16-30 (CPU check up to 28) | same (32 / 30); Aer CPU also `qft`, `random`; variants: CUDA-Q no fusion, Aer GPU fusion<=2q + cuStateVec |
| `colab_g4` | RTX PRO 6000 Blackwell 96 GB | 16-32 / 16-30 (CPU check up to 28) | same (32 / 30); variants as `colab_h100` |
| `nvidia_generic` | any other NVIDIA GPU | 16-30 / 16-28, fitted at run time | Aer CPU + GPU, CUDA-Q, Lightning CPU + GPU; variants as `colab_h100` |
| `amd_mi300x` | MI300X / MI325X / MI355X (192-288 GB) | 16-32 / 16-32 | Aer CPU (+ `qft`, `random`), Lightning CPU + AMDGPU (32 / 32) |
| `amd_generic` | any other AMD GPU | 16-30 / 16-28, fitted at run time | Aer CPU, Lightning CPU + AMDGPU |
| `apple_metal` | Apple M-series (fp32 only) | 16-28 / n.a. | Aer CPU (fp64 + fp32), Lightning CPU |
| `cpu_only` | no (supported) GPU | - | Aer CPU fp64 (+ `qft`, `random`), Lightning CPU |
| `quick` | anything | 16-18 | everything that is installed, tiny sizes |

On a preset, every GPU engine gets the same GPU widths and every CPU engine (QUMO CPU, Aer CPU, Lightning CPU) the
widths of `sv_cpu`. Every GPU preset runs Aer CPU in fp64 and fp32 (`baseline_aer.precisions_cpu`), so fp32 GPU
series have a same-precision CPU reference. GPU upper ends are the largest even width whose state fits 60 % of the
GPU's nominal memory (8 B/amplitude fp32, 16 B fp64); the runtime clamp lowers them on smaller variants or when host
RAM is short. `sv_gpu.cpu_max` (the `sv_cpu` upper end) bounds QUMO's fp64 CPU check, so it no longer caps QUMO's GPU
widths. `sv_gpu_probe` is on in `colab_t4`, `colab_l4` and `colab_a100` and off elsewhere (`SUITES=sv_gpu_probe` or
`"enabled": true` turns it on); it skips with a recorded reason while the QUMO version has no probe. Each
suite also has a `quick` block used by `--quick`; any qubit range the block does not name (e.g. a precision-specific
`qubits_gpu_fp64`) inherits the block's base range or is cut to five widths, so a quick run gives every engine the
same 16-20 qubit widths.

## 4. How to reproduce

```bash
# anywhere (clone, install, build, run; auto-detects NVIDIA / AMD / Apple / CPU)
bash examples/colab/run_on_any_gpu.sh                    # or --quick, --fp64, --config <preset>, --suites ...

# step by step in a checkout
make setup                                               # Mojo + MAX in .venv
uv venv bench/.venv --python 3.12
uv pip install --python bench/.venv/bin/python qiskit qiskit-aer pytket pytket-qiskit stim pymatching numpy matplotlib psutil
bench/.venv/bin/python bench/run_benchmarks.py --list    # suites, presets, detected GPU, auto preset
bench/.venv/bin/python bench/run_benchmarks.py --config auto --build-only
bench/.venv/bin/python bench/run_benchmarks.py --config auto --label "my machine"     # GPU engines in fp32
bench/.venv/bin/python bench/run_benchmarks.py --config auto --fp64                     # ... and in fp64
QUMO_CUDAQ_PYTHON=/path/venv-cudaq/bin/python \
  bench/.venv/bin/python bench/run_benchmarks.py --config colab_a100 --suites sv_gpu,baseline_cudaq

# compare runs (different GPUs, vendors, commits)
bench/.venv/bin/python examples/colab/compare_results.py bench/results/cloud/*.json \
    --out compare.md --plot compare.png --reference "Aer CPU {precision}"
```

Interpreters for baselines: `python` / `gpu_python` in a suite's config, or the environment variables
`QUMO_BENCH_PYTHON` (default for all), `QUMO_AER_GPU_PYTHON`, `QUMO_CUDAQ_PYTHON`, `QUMO_LIGHTNING_PYTHON`.
`QUMO_FORCE_GPU=1` runs the QUMO GPU suites on a GPU outside the support list. `QUMO_MOJO` (or `--mojo`) points the
runner at another Mojo compiler. `--no-build` never compiles; missing binaries are then recorded as skips.

### Result files

`bench/results/cloud/qumo_bench_<preset>_<gpu>_<timestamp>.json` (schema `qumo-bench/1`): `environment`, `config`
(name, path, full content, `quick`), `precision_policy` (GPU precisions of the run and their source), and per suite
(GPU suites also `gpu_precisions`): `status`, `reason`, `rows`, `commands` (exact,
repository-relative), `build(s)`, `clamp(s)`, `skipped`, `not_applicable`, `versions` (baseline versions and
verification), `duration_s`, `loadavg_before/after`, `machine_state_before/after`, `variant_problems`, and for
`sv_gpu` `engine_settings` (the `bench_gpu` features found, QUMO's compile-time GPU knobs with their defaults, the
defines and environment of every build). The environment adds `sources` (engine and harness hashes) and
`machine_state`; the document adds `checks` (cross-engine consistency). Rows carry `precision` (what ran),
`precision_requested`, `timed_region`, `fusion`, `times`, `variant`, and for QUMO GPU `launches`, `transposes`, `gates_per_launch`, `bytes_model`, `effective_gbs`,
`seconds_incl_alloc` and `extra` (any column a newer `bench_gpu` prints that the runner does not know). The CSV next
to it holds all rows (columns `suite, kind, series, engine, device, precision, precision_requested, family, qubits,
gates, seconds, gates_per_sec, reps, ...`; dictionaries as JSON). The `neutral` suite also writes `<stem>_neutral.jsonl` with every
compilation record.

## 5. Status and limitations (October 2026)

* Verified on an Apple M5 (Metal, macOS 26.5) shared with other jobs (load average 13-70 on 10 cores during the
  latest checks, 80-270 during earlier ones), so none of its timings mean anything; only behaviour was checked:
  * `quick` preset end to end (every suite ran: `sv_cpu`, `sv_gpu` on Metal with fp32 rows at fidelity 1 - 7e-8
    against the CPU plus the Pauli-frame table, Aer CPU fp64 and fp32, Lightning CPU, the compiler subset, the QEC
    subset, a test file; CUDA-Q GPU recorded as *not applicable*), and `apple_metal --quick` (QUMO GPU fp32 16-20
    qubits; `gpu_check` on the host target `metal:5` passed for two of the three files and recorded the third,
    `tests/test_gpu.mojo`, which did not compile at that moment, as `QM-E969` with its log path). Earlier runs also covered the CUDA-Q worker on its CPU target
    (`qpp-cpu`, macOS wheel `cuda-quantum-cu13 0.16.0`, fidelity 1.0 against the numpy reference), the Aer worker under
    Qiskit 1.4.6 + Aer 0.15.1 (the code base of `qiskit-aer-gpu`) and Qiskit 2.5.2 + Aer 0.17.2, and a QUMO GPU kernel
    failure, which the runner recorded as `QM-E968` without stopping.
  * Preset choice for 20 GPU names / architectures (T4, L4, A100 40/80, H100, B200, RTX PRO 6000, MI300X, MI355X,
    MI250X, RX 7900 XTX, RTX 4090/5090, V100, L40S, unknown, none, AMD by architecture only, Apple M5) and the
    detection paths with stand-in `nvidia-smi` (T4, V100, P100), `amd-smi` (old list layout and the newer
    `gpu_data` layout) and `rocminfo` tools; the `--quick` widths of every preset (16-20 qubits for every engine and
    precision) with a harness over `apply_quick` / `_grid_params`, and a `--quick` run with a stand-in T4.
  * Both notebooks: `nbformat` validation (no outputs, no secrets), `nbclient` execution of the helper, GPU check,
    download / preset, build (with `SUITES`), report and save cells, and the AMD notebook's compare cell; step 2 re-run against a local copy of
    the repository (shallow clone of `main`, then an abbreviated commit, then `main` again; a checkout the notebook
    did not clone is never moved). `compare_results.py` on two runs; `run_on_any_gpu.sh` end to end on macOS
    (`--quick --suites sv_cpu,baseline_aer --no-baselines`, the build step compiles only `bench_sv`), `bash -n` and
    ShellCheck 0.11.
* **One Colab T4 run so far** (2026-10-12, `bench/results/cloud/qumo_bench_colab_t4_tesla-t4_20261012-235508.json`),
  with commit 1750a81: the older harness and the older QUMO GPU engine (see
  [What changed on 2026-10-13](#what-changed-on-2026-10-13-timing-fairness)). The QUMO GPU tile engine has not run
  on NVIDIA or AMD hardware yet; no AMD run exists.
* The timing-fairness changes of 2026-10-13 were checked on the M5 without GPU benchmarks: unit checks of the
  parsing, bandwidth model, variants, warm-up and suite logic (a stand-in `bench_gpu` and probe binary, a stand-in
  `cudaq` module for the CUDA-Q worker), the real output of a 16-18 qubit `bench_gpu` default run and of a 16-18
  qubit `bench_gpu ... probe` run on the M5 parsed by the runner and rendered by the report,
  `--config quick --suites sv_cpu,baseline_aer` end to end, an Aer CPU variant (`fusion_max_qubit=2`) and a failing
  variant (recorded under `variant_problems`, suite still `ok`), `compare_results.py` on the old and new result
  files, and `nbformat` validation of both notebooks. The CUDA-Q variants, NVIDIA clock recording, Aer GPU variants,
  the probe on NVIDIA and `count_gpu_kernels.py` with a real `nsys` have not run yet: the next Colab run is their
  first test.
* `bench_gpu` always sweeps from 16 qubits, and it runs the Pauli-frame table in each of its builds (fp32, and fp64
  with `--fp64`) and in every build variant; the runner keeps only the default build's frames rows.
* QUMO CPU has a process warm-up (one dropped smaller width), not a per-width one: `bench_sv.mojo` would need its own
  untimed `evolve` per width for that. Each repetition allocates its own state, so the remaining first-repetition
  effect is small, and the upper median of 3 discards a single slow repetition.
* Colab hardware, drivers and prices change without notice; the JSON records what was actually used.
