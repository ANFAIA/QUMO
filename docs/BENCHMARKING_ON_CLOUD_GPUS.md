# Benchmarking QUMO on cloud GPUs (NVIDIA, AMD, Apple)

QUMO's GPU code is vendor-neutral Mojo (MAX `DeviceContext`, see [GPU.md](GPU.md)). This page describes
how to benchmark it on GPUs you rent or own, and how to compare results across machines:

* **Google Colab** (NVIDIA T4 / L4 / A100 / H100 / G4): `examples/colab/QUMO_GPU_Benchmark_Colab.ipynb`
* **AMD GPUs on any ROCm Jupyter host** (AMD Developer Cloud MI300X, Azure ND MI300X v5, a local
  Radeon / Instinct box): `examples/colab/QUMO_GPU_Benchmark_AMD.ipynb`
* **Any Linux GPU host or a Mac (Metal), headless:** `examples/colab/run_on_any_gpu.sh`
* **One runner behind all three:** `bench/run_benchmarks.py` + presets in `bench/configs/`
* **Comparison of several runs:** `examples/colab/compare_results.py`

The benchmark treats NVIDIA, AMD and Apple alike: presets are named after hardware, the
configuration is picked from the detected GPU, the same circuits and the same timing rules apply
everywhere, and every vendor gets the strongest third-party baseline that runs on it.

## 1. Support matrix (researched October 2026)

### Where QUMO's GPU kernels run (Mojo 1.1 / MAX 26.6)

| Vendor | GPUs (architecture, Mojo target) | Modular tier | Requirement |
|---|---|---|---|
| NVIDIA | B200 (Blackwell, `sm_100`) | continuously tested | driver >= 580 |
| NVIDIA | B100/B300 (`sm_100`/`sm_103`), RTX 50xx / RTX PRO 6000 (`sm_120`), H100/H200 (`sm_90`), L4/L40 (`sm_89`), A100 (`sm_80`), A10 (`sm_86`), **T4** / RTX 20xx (`sm_75`) | known compatible | driver >= 580, or `MODULAR_NVPTX_COMPILER_PATH=<CUDA ptxas>` on older drivers |
| NVIDIA | Pascal / Volta and older (P100, V100, K80) | not supported out of the box | needs a system `ptxas`; QUMO's runner skips its GPU suites there |
| AMD | MI355X (CDNA4, `gfx950`), MI300X (CDNA3, `gfx942`) | continuously tested | ROCm / amdgpu driver >= 6.3.3; >= 7.0 for MI355X |
| AMD | MI325X (`gfx942`), MI250X (`gfx90a`), Radeon RX 9070/9060 (`gfx1201`/`gfx1200`), RX 7900/7800/7700/7600 (`gfx110x`), Ryzen AI iGPUs (`gfx115x`), RX 6900 (`gfx1030`) | known compatible | same |
| Apple | M1-M5 (Metal) | known compatible | macOS 15+, Xcode 16+, `xcodebuild -downloadComponent MetalToolchain`; **fp32 only** (Metal has no double) |

Sources: [Mojo system requirements, GPU tables](https://mojolang.org/docs/requirements/);
[MAX packages and system requirements](https://max.modular.com/packages/) (redirect target of
docs.modular.com/packages); [Modular forum: Turing and newer, driver 580 (Feb 2026)](https://forum.modular.com/t/support-for-older-nvidia-gpus-and-drivers-in-mojo-and-max/2761).
`bench/run_benchmarks.py` encodes this table in `gpu_supported()` and records the verdict for every
detected GPU in the result JSON (`environment.gpus[].mojo_max_supported`, `support_note`).

### Installing Mojo / MAX on a cloud machine

* `make setup` = `uv venv .venv --python 3.12 && uv pip install mojo max-mojo-libs max-core`.
* PyPI wheels (released 2026-09-17): `mojo 1.1.0`, `max-core 26.6.0` and `mojo-compiler 1.1.0` for
  `manylinux_2_34_x86_64`, `manylinux_2_34_aarch64` and `macosx_13_0_arm64`; `max-mojo-libs 26.6.0` is
  pure Python. So Linux needs **glibc >= 2.34** (Ubuntu 22.04+, which Colab and the AMD Developer Cloud
  images satisfy). Python 3.10-3.14; x86-64-v3 (Haswell or newer) or ARM Neoverse N1+ CPUs.
  Sources: [pypi.org/project/mojo](https://pypi.org/project/mojo/), [pypi.org/project/max-core](https://pypi.org/project/max-core/),
  [MAX system requirements](https://max.modular.com/packages/).
* MAX bundles its own GPU compiler: no CUDA toolkit or HIP SDK is needed to build QUMO's kernels, only the
  NVIDIA driver (>= 580) or the ROCm driver stack. `make gpu-check` cross-compiles for `sm_90`, `gfx942`
  and `metal:4` without any GPU; the runner's `gpu_check` suite adds the host architecture (e.g. `sm_75`
  on a T4).

### Google Colab

* GPU runtimes: T4 (free tier), L4, A100 40 GB / 80 GB (the High-RAM slider selects 80 GB), H100, and G4
  (RTX PRO 6000 Blackwell, ~96 GB). Measured burn rates (March 2026): T4 1.19, L4 1.71, A100-40GB 5.40,
  A100-80GB 7.52, G4 8.71 compute units/hour; 100 units cost about US$9.99 (Colab Pro), so roughly
  $0.12/h (T4) to $0.87/h (G4). H100 availability varies. Sources:
  [McCormick, Colab GPUs features & pricing (updated March 2026)](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/),
  [gpuperhour: Colab alternatives (Sept 2026 prices)](https://gpuperhour.com/blog/google-colab-alternatives).
* Python: the 2026.04 runtime ships Python 3.12.13
  ([Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html)).
* CUDA: a July 2026 T4 session reported a driver supporting CUDA 13.0 and a CUDA 12.8 toolkit
  ([datawookie, Running LLMs in Google Colab](https://datawookie.dev/blog/2026-07-31-running-llms-google-colab/)).
  Colab changes this without notice, so the notebook reads `nvidia-smi` at run time, picks the CUDA-Q
  wheel for the driver's CUDA major version, and sets `MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas`
  if the driver is older than 580.
* Colab has **no AMD GPUs**.

### NVIDIA baselines

* **CUDA-Q** (`cudaq` 0.16.0.post1, Python >= 3.11): `pip install cudaq` (a meta package); the notebook
  installs the matching wheel directly, `cuda-quantum-cu12` or `cuda-quantum-cu13` (manylinux_2_28,
  Linux only). The `nvidia` target (cuStateVec) is **fp32 by default**; `cudaq.set_target("nvidia",
  option="fp64")` selects double precision; `qpp-cpu` is the CPU target. Sources:
  [CUDA-Q quick start](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html),
  [CUDA-Q state-vector simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html),
  [pypi.org/project/cudaq](https://pypi.org/project/cudaq/).
* **Qiskit Aer GPU**: `qiskit-aer-gpu` 0.15.1 (Sept 2024) is the latest wheel: CUDA 12, Linux x86_64,
  `qiskit>=1.1`. It predates Qiskit 2.x (CPU `qiskit-aer` is 0.17.2), so it gets its own virtual
  environment with `qiskit<2`. Sources: [pypi.org/project/qiskit-aer-gpu](https://pypi.org/project/qiskit-aer-gpu/),
  [Qiskit Aer](https://github.com/Qiskit/qiskit-aer/).
* **PennyLane Lightning** `lightning.gpu` (cuStateVec), optional.

### AMD GPUs and baselines

* **Where:** [AMD Developer Cloud](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html)
  (DigitalOcean GPU droplets `gpu-mi300x1-192gb` / `gpu-mi300x8-1536gb`; the ROCm image runs JupyterLab on the
  base OS, Quick Start images run it in their container;
  [release note](https://docs.digitalocean.com/notes/2025/mi300x-gpu-droplets)); list prices reported between
  US$1.99 and $2.59 per GPU-hour ([DigitalOcean AMD pricing](https://docs.digitalocean.com/products/amd/details/pricing/), check the live page);
  [Azure ND MI300X v5](https://ir.amd.com/news-events/press-releases/detail/1198/amd-instinct-mi300x-accelerators-power-microsoft-azure-openai-service-workloads-and-new-azure-nd-mi300x-v5-vms)
  (8x MI300X, 1.5 TB HBM per VM); or any local ROCm machine with `pip install jupyterlab`.
* **Baselines:** CUDA-Q and `qiskit-aer-gpu` are CUDA-only. The AMD-native GPU state-vector baseline is
  **PennyLane `lightning.amdgpu`**: `pip install pennylane pennylane-lightning-amdgpu`, prebuilt for ROCm 7.0
  and MI300-series GPUs (other GPUs/ROCm versions: build from source with Kokkos; ROCm 7.2 is reported to
  break the build). Sources: [Lightning-AMDGPU installation](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_amdgpu/installation.html),
  [PennyLane x AMD (Dec 2025)](https://pennylane.ai/blog/2025/12/expanding-the-frontier-accelerating-quantum-workflows-with-pennylane-and-amd).
  Qiskit Aer on ROCm exists as a research port ([CHEP 2024](https://www.epj-conferences.org/10.1051/epjconf/202429511022))
  and, per an AMD article, as a `qiskit-aer-gpu-rocm` package
  ([AMD / BlueQubit, 2026](https://www.amd.com/zh-cn/developer/resources/technical-articles/2026/largest-single-gpu-quantum-simulation-on-amd-by-bluequbit.html));
  not verified here, so the runner does not install it (set `QUMO_AER_GPU_PYTHON` to an interpreter that
  has it and add `"gpu"` to `baseline_aer.devices` to include it).

## 2. What is measured

### Suites (`bench/run_benchmarks.py --list`)

| Suite | What runs | Rows |
|---|---|---|
| `gpu_check` | `make gpu-check` for the host GPU's architecture (plus any listed targets) | PASS/FAIL per target x file, kernel count |
| `sv_cpu` | `build/bench_sv` (QUMO CPU, fp64): `layered`, optionally `qft` and `random` | seconds, gates/s per width |
| `sv_gpu` | `build/gpu/bench_gpu_fp32` / `_fp64` (`-D QUMO_GPU [-D QUMO_GPU_FP32]`) | GPU seconds, CPU seconds, speed-up, expectation and 10k-shot sampling time, fidelity vs CPU; plus the QEC Pauli-frame table (surface code d = 3, 5, 7; 1M shots) |
| `baseline_aer` | Qiskit Aer statevector, CPU (fp64) and GPU (fp32, fp64) | seconds per width |
| `baseline_cudaq` | CUDA-Q `nvidia` target fp32 and fp64 (optional `qpp-cpu`) | seconds per width |
| `baseline_lightning` | PennyLane `lightning.qubit` / `.gpu` / `.amdgpu` | seconds per width |
| `neutral` | a small subset of the hardware-neutral compiler benchmark (`bench/neutral`) | per SDK: status counts, median compile time, 2q gates, ESP |
| `qec` | `bench/bench_qec.py --part A` with few shots (QUMO vs Stim + PyMatching) | the markdown tables, parsed |
| `tests` | `make test` / `make test-gpu`, or a list of test files | PASS/FAIL per file |

### The circuit

All state-vector engines run the `layered` circuit of `bench/bench_sv.mojo` (and `bench/bench_gpu.mojo`):
10 layers; in layer `d` every qubit `q` gets `ry(0.1 + 0.01 (q + d))` then `rz(0.2 + 0.03 q d)`, followed by
`cx(q, q+1)` for `q = d mod 2, d mod 2 + 2, ...`. The Python baselines build it from one function
(`layered_ops`), and every row stores its gate count, so a mismatch would show (QUMO and Aer both report
395 gates at 16 qubits). Widths run on the grid 16, 18, 20, ... like the Mojo benchmarks.

### Correctness before speed

* QUMO GPU: `bench_gpu` reports the fidelity of the GPU state against the CPU simulator at every width,
  and the expectation-value error; the runner flags fidelity < 1 - 1e-4.
* Baselines: before timing, each worker computes the 8-qubit state and compares it with an independent
  numpy reference of the circuit (`reference_state`); the fidelity and the engine's qubit order
  (little- or big-endian) are stored under `versions[].verify`. Aer and Lightning reach 1.0 on the
  development machine.
* `tests` with `test-gpu` runs QUMO's GPU test suites on the device (device == emulator == CPU checks).

### Timing rules

* **Warm-up:** QUMO GPU and CUDA-Q warm up at every width (module load / JIT, allocation); Aer warms up once
  (library load, GPU context); Lightning once per width (device object creation).
* **Repetitions and medians:** `reps` per width (3 by default, 1 with `--quick`); the median is reported
  and the individual times are kept for the baselines (`times`).
* **What is inside the timed region:**

| Engine | Timed | Not timed |
|---|---|---|
| QUMO CPU (`bench_sv`) | `evolve(circuit)`: plan, allocation, simulation | - |
| QUMO GPU (`bench_gpu`) | `reset + evolve + synchronize` on a preallocated device state | allocation, `DeviceContext` creation, kernel-module load |
| Aer | `sim.run(circuit).result()` with `save_amplitudes([0])` (full evolution, one amplitude to the host) | library import |
| CUDA-Q | `cudaq.get_state(kernel, ...)` plus reading one amplitude (forces completion) | first call per width (JIT) |
| Lightning | QNode call returning `<Z0>` (device-side reduction, no 2^n host copy) | first call per width |

  Allocation and initialisation are a single pass over memory, small next to the ~400 gate passes of the
  circuit at 20+ qubits; small widths (16-20 qubits) are dominated by fixed per-call overheads (Python,
  launches) and should not be used to rank GPUs.
* **Precision fairness:** compare fp32 with fp32 and fp64 with fp64. QUMO CPU and Aer CPU are fp64. CUDA-Q
  defaults to fp32, so both precisions are run. On Apple GPUs only fp32 exists. T4 (1/32), L4 and RTX PRO
  6000 (1/64) have slow fp64 units, which hurts every engine's fp64 numbers on those GPUs equally.
* **Memory clamp:** a width runs only if its state fits into `memory_fraction` (60 %) of GPU memory and of
  free host RAM (8 B/amplitude fp32, 16 B fp64; `bench_gpu` also needs about 37 B/amplitude of host RAM for
  its CPU reference and fidelity check). Every clamp is recorded (`clamp`, code `QM-E966`).
* **Shared machines:** suites run one after another, never concurrently. The load average is recorded at
  the start, before and after every suite, and at the end. Treat runs with a load average above the core
  count as indicative only and rerun.
* **Failures are data:** every suite catches failures and timeouts (whole process group killed) and records
  `ok`, `partial`, `skipped`, `failed` or `timeout` with a stable code: QM-E960 config, QM-E961 build,
  QM-E962 timeout, QM-E963 tool missing, QM-E964 unparsable output, QM-E965 baseline missing, QM-E966 memory
  clamp, QM-E967 no / unsupported GPU or precision, QM-E968 benchmark binary or worker crash, QM-E969 test or check FAIL. Baselines that cannot
  exist on the hardware (CUDA-Q on AMD) are listed as `not_applicable`, not as failures.

### Hardware neutrality checklist

* No vendor is a default: `--config auto` picks the preset from the detected GPU (`nvidia-smi`, `amd-smi` /
  `rocm-smi` / `rocminfo`, `system_profiler`); without a supported GPU it falls back to `cpu_only`.
* The same runner, circuits, timing rules, memory policy and JSON schema on every vendor.
* Each vendor gets its strongest available baseline: NVIDIA: CUDA-Q (cuStateVec), Aer GPU, Lightning GPU;
  AMD: Lightning AMDGPU; Apple: no GPU-native third-party state-vector simulator is installed, so the CPU
  baselines are the reference.
* Environment metadata makes runs comparable: GPU name / arch / memory / driver (CUDA or ROCm version),
  CPU model and usable cores, RAM, OS and glibc, Python, Mojo / MAX versions, git commit and dirty state,
  Colab release, timestamps, load averages. No host names, user paths or credentials are stored (git remotes
  are stripped of credentials, local paths are rewritten to repository-relative ones).

## 3. Presets (`bench/configs/`)

| Preset | Hardware | QUMO GPU widths (fp32 / fp64) | Baselines |
|---|---|---|---|
| `colab_t4` | T4 16 GB, ~2 vCPU, 12.7 GB RAM | 16-30 / 16-28 | Aer CPU + GPU, CUDA-Q, Lightning GPU |
| `colab_l4` | L4 24 GB | 16-30 / 16-28 | same |
| `colab_a100` | A100 40/80 GB, ~83 GB RAM | 16-30 / 16-30 | same; CUDA-Q fp32 to 32 |
| `colab_h100` | H100 80 GB (also B100/B200/GH200) | 16-32 / 16-30 | same, to 32 |
| `colab_g4` | RTX PRO 6000 Blackwell 96 GB | 16-32 / 16-30 | same |
| `amd_mi300x` | MI300X / MI325X / MI355X | 16-32 / 16-32 | Aer CPU, Lightning CPU + AMDGPU |
| `amd_generic` | other supported AMD GPUs | 16-30 / 16-28 | same |
| `apple_metal` | Apple M-series (fp32 only) | 16-28 / - | Aer CPU, Lightning CPU |
| `cpu_only` | no supported GPU | - | Aer CPU, Lightning CPU |
| `quick` | anything | 16-18 | everything that is installed |

Upper ends are what fits 60 % of the GPU's memory; the runtime clamp lowers them on smaller variants or
when host RAM is short (e.g. a free-tier T4 stops QUMO GPU at 26 qubits because of the CPU reference in
`bench_gpu`). Each suite also has a `quick` block used by `--quick`.

## 4. How to reproduce

```bash
# anywhere (clone, install, build, run; auto-detects NVIDIA / AMD / Apple / CPU)
bash examples/colab/run_on_any_gpu.sh                    # or --quick, --config <preset>, --suites ...

# step by step in a checkout
make setup                                               # Mojo 1.1 + MAX 26.6 in .venv
uv venv bench/.venv --python 3.12
uv pip install --python bench/.venv/bin/python qiskit qiskit-aer stim pymatching pytket numpy matplotlib psutil
bench/.venv/bin/python bench/run_benchmarks.py --list    # suites, presets, detected GPU, auto preset
bench/.venv/bin/python bench/run_benchmarks.py --config auto --build-only
bench/.venv/bin/python bench/run_benchmarks.py --config auto --label "my machine"
QUMO_CUDAQ_PYTHON=/path/venv-cudaq/bin/python \
  bench/.venv/bin/python bench/run_benchmarks.py --config colab_a100 --suites sv_gpu,baseline_cudaq

# compare runs (different GPUs, vendors, commits)
bench/.venv/bin/python examples/colab/compare_results.py bench/results/cloud/*.json \
    --out compare.md --plot compare.png --reference "Aer CPU fp64"
```

Interpreters for baselines: `python` / `gpu_python` in a suite's config, or the environment variables
`QUMO_BENCH_PYTHON` (default for all), `QUMO_AER_GPU_PYTHON`, `QUMO_CUDAQ_PYTHON`, `QUMO_LIGHTNING_PYTHON`.
`QUMO_FORCE_GPU=1` runs the QUMO GPU suites on a GPU outside the support list. `QUMO_MOJO` points the runner
at another Mojo compiler.

### Result files

`bench/results/cloud/qumo_bench_<preset>_<gpu>_<timestamp>.json` (schema `qumo-bench/1`): `environment`,
`config` (name, path, full content, `quick`), and per suite: `status`, `reason`, `rows`, `commands` (exact,
repository-relative), `builds`, `clamp(s)`, `skipped`, `not_applicable`, `versions` (baseline versions and
verification), `duration_s`, `loadavg_before/after`. The CSV next to it holds all rows (columns `suite, kind,
series, engine, device, precision, family, qubits, gates, seconds, gates_per_sec, reps, ...`). The JSON is
rewritten after every suite, so an interrupted session keeps the finished suites.

## 5. Status and limitations (October 2026)

* The harness was verified end to end on an Apple M5 (Metal) with CPU suites, Aer and Lightning CPU
  baselines, `gpu_check`, `neutral`, `qec` and `tests`. It has **not yet run on NVIDIA or AMD hardware**
  (none available to the authors); the NVIDIA/AMD code paths (detection, CUDA-Q/Aer-GPU/Lightning-GPU
  workers, presets) are written against the documented interfaces above. The CUDA-Q worker tries a
  `@cudaq.kernel` first and falls back to `cudaq.make_kernel()`, and records which one ran.
* QUMO GPU timings come from `bench/bench_gpu.mojo`, which always sweeps from 16 qubits and runs the CPU
  reference at every width. At 28+ qubits that CPU work dominates the wall time and host RAM, and caps the
  widths on small-RAM runtimes. A GPU-only mode in `bench_gpu` would remove the cap.
* Colab hardware, drivers and prices change without notice; the JSON records what was actually used.
