# Run the QUMO benchmark on cloud GPUs

Download QUMO, build it on a rented or local GPU, run one benchmark, get a JSON + CSV + plots, and
compare machines: NVIDIA on Google Colab today, AMD on any ROCm Jupyter host, Apple (Metal) and plain
CPUs for reference. Method, timing rules and sources: [docs/BENCHMARKING_ON_CLOUD_GPUS.md](../../docs/BENCHMARKING_ON_CLOUD_GPUS.md).

| File | Use it for |
|---|---|
| [`QUMO_GPU_Benchmark_Colab.ipynb`](QUMO_GPU_Benchmark_Colab.ipynb) | Google Colab, NVIDIA GPUs (T4, L4, A100, H100, G4) |
| [`QUMO_GPU_Benchmark_AMD.ipynb`](QUMO_GPU_Benchmark_AMD.ipynb) | AMD GPUs on any Jupyter with ROCm (AMD Developer Cloud, Azure ND MI300X, local) |
| [`run_on_any_gpu.sh`](run_on_any_gpu.sh) | headless: any Linux GPU host (auto-detects NVIDIA / AMD / none) or a Mac (Metal) |
| [`compare_results.py`](compare_results.py) | merge several result JSONs into one markdown table + plots |
| [`bench_report.py`](bench_report.py) | the tables / plots helpers used by the notebooks and `compare_results.py` |
| [`count_gpu_kernels.py`](count_gpu_kernels.py) | optional, NVIDIA: kernels one CUDA-Q `get_state` launches, counted with Nsight Systems (notebook step 9) |
| [`plot_t4_vs_m5.py`](plot_t4_vs_m5.py) | the figure of the first T4 run next to the Apple M5 (states that the two ran different QUMO GPU engines) |

Everything calls the same runner, [`bench/run_benchmarks.py`](../../bench/run_benchmarks.py), with a
preset from [`bench/configs/`](../../bench/configs/).

## Google Colab (NVIDIA)

1. Open the notebook:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ANFAIA/QUMO/blob/main/examples/colab/QUMO_GPU_Benchmark_Colab.ipynb)
2. *Runtime -> Change runtime type -> Hardware accelerator:* pick a GPU. For a fair first look use the
   free **T4**; with Colab Pro / pay-as-you-go try **L4**, **A100** (40 or 80 GB), **H100** or **G4** (RTX PRO
   6000 Blackwell). *High-RAM* gives more host RAM, which lets the CPU references reach larger widths.
3. *Runtime -> Run all.* Step 1 reads `nvidia-smi`; step 2 downloads QUMO and lets the runner pick the
   preset (`colab_t4`, `colab_l4`, `colab_a100`, `colab_h100`, `colab_g4`, `nvidia_generic` for any other
   NVIDIA GPU). Override it in the form, tick `QUICK` for a smoke test (16-20 qubits for every engine,
   one repetition), or set `SUITES` to run and build only some suites (e.g. `sv_gpu,baseline_cudaq`).
   Every GPU engine runs in **fp32** (CUDA-Q's default and QUMO's); tick `INCLUDE_FP64` to add the fp64
   series of every GPU engine.
4. Step 6 shows the tables and plots; step 7 downloads a zip with the JSON, CSV, report and plots
   (tick `SAVE_TO_DRIVE` to also copy them to `MyDrive/QUMO_benchmarks`).

**Comparing NVIDIA configurations:** with `SAVE_TO_DRIVE` ticked, run on a T4, then *Runtime -> Change
runtime type* (L4, A100, ...) and *Run all* again (changing the runtime wipes `/content`, Drive keeps the
results). Step 8 with `FROM_DRIVE` merges every saved run into one table and plot; `UPLOAD` adds JSONs
from elsewhere (an AMD or Apple run, your laptop).

### Expected duration and cost (estimates)

| Runtime | Full preset (fp32) | `INCLUDE_FP64` adds | `QUICK` | Approx. cost |
|---|---|---|---|---|
| T4 (free tier) | 55-100 min (with the diagnostic variants; `SUITES` can skip them) | ~15-25 min (fp64 at 1/32 rate) | ~12-15 min | free (session limits apply) or 1.19 units/h |
| L4 | 30-50 min | ~10-15 min (1/64 rate) | ~12 min | 1.71 units/h (~$0.17/h) |
| A100 40 / 80 GB | 30-50 min | ~5-10 min | ~10 min | 5.40 / 7.52 units/h (~$0.54-0.75/h) |
| G4 (RTX PRO 6000) | 35-60 min | ~10-15 min (1/64 rate) | ~10 min | 8.71 units/h (~$0.87/h) |
| H100 | 35-60 min | ~5-10 min | ~10 min | not measured |

About 10-15 minutes of every run is the one-off download of Mojo/MAX (~1 GB) and the builds; most of
the rest is CPU work (the QUMO CPU and Aer CPU fp64 + fp32 references, each with one warm-up per width,
which `INCLUDE_FP64` does not change;
`bench_gpu` checks against the CPU up to `sv_gpu.cpu_max` qubits). The NVIDIA presets add labelled diagnostic
series (CUDA-Q without fusion and, on the T4, with fusion widths 2-6; Aer GPU with narrower fusion and with
cuStateVec; a second QUMO GPU build with 16 amplitudes per thread) and the QUMO GPU cost probe; on the T4 they
add roughly 25-40 minutes (estimate). The Colab notebook installs all three NVIDIA baselines by default (CUDA-Q,
`qiskit-aer-gpu`, PennyLane `lightning.gpu`); untick one in step 3 to save install time. The
durations are estimates from the presets' sizes, not yet measured on Colab (the fp32-only default drops one
`bench_gpu` build and the fp64 series of every GPU engine, including the diagnostic variants); the result
JSON records the real `duration_s` of each suite. Unit rates: March 2026 measurements, 100 units ~ US$10
(see the doc's sources).

## AMD GPUs (ROCm, no Colab)

Colab has no AMD GPUs. Use any Jupyter on a ROCm machine:

* **AMD Developer Cloud** (MI300X droplets, US$2.59 per GPU-hour list price): create a GPU droplet,
  preferably with the plain ROCm image (JupyterLab on the base OS; the Quick Start images run JupyterLab
  inside a container that may lack `git` / `make` / a C compiler), open JupyterLab, then upload
  `QUMO_GPU_Benchmark_AMD.ipynb` (or `git clone https://github.com/ANFAIA/QUMO` and open it from
  `examples/colab/`). Run all cells.
* **Azure ND MI300X v5**, other MI300X/MI325X/MI355X providers, or a local Radeon RX 7000/9000 /
  Instinct machine: `pip install jupyterlab && jupyter lab`, same notebook.
* Headless instead: `bash examples/colab/run_on_any_gpu.sh`.

The `amd_mi300x` preset runs QUMO CPU/GPU (GPU in fp32; `INCLUDE_FP64 = True` adds fp64, which MI300X runs at
full rate), Aer CPU and PennyLane Lightning CPU + `lightning.amdgpu` (the AMD-native GPU baseline; its prebuilt wheels target
MI300-series GPUs with ROCm >= 7.0, elsewhere the suite records a skip). CUDA-Q and Aer GPU are CUDA-only
and are recorded as *not applicable*. Other AMD GPUs get `amd_generic`. Expect ~30-60 min on one MI300X
(estimate).

## Any machine, headless

```bash
curl -fsSLO https://raw.githubusercontent.com/ANFAIA/QUMO/main/examples/colab/run_on_any_gpu.sh
bash run_on_any_gpu.sh                    # clone -> install -> build -> run (auto preset)
bash run_on_any_gpu.sh --quick            # smoke test
bash run_on_any_gpu.sh --fp64             # GPU engines in fp32 and fp64 (default: fp32 only)
bash run_on_any_gpu.sh --config colab_a100 --suites sv_gpu,baseline_cudaq --label "A100 PCIe, lab server"
# from inside a checkout it uses that checkout: bash examples/colab/run_on_any_gpu.sh --quick
```

It installs Mojo 1.1.0 + MAX 26.6.0 (pinned; `--mojo-version "" --max-version ""` for the latest), the
benchmark venv, and the vendor's baselines in their own venvs under `build/baseline-venvs` (override with
`QUMO_VENVS`). Needs `git`, `python3`, `make` and a C compiler.

On a Mac it runs the `apple_metal` preset (QUMO GPU in fp32 through Metal) as a local comparison point;
it needs Xcode and `xcodebuild -downloadComponent MetalToolchain`.

## Compare runs

```bash
bench/.venv/bin/python examples/colab/compare_results.py \
    t4.json l4.json a100.json mi300x.json m5.json --out compare.md --plot compare.png
```

`compare.md` holds: the machines (GPU, memory, arch, driver/ROCm, CPU, RAM, OS, Mojo/MAX, commit, date,
load average), the status of every suite, one timing table (rows = run x engine/device/precision,
columns = qubits), speed-ups against a reference of the same precision (`--reference`, default
`Aer CPU {precision}`: fp64 series against Aer CPU fp64, fp32 series against Aer CPU fp32; never a
mixed-precision ratio) and every skipped or failed part with its `QM-E96x` reason. `compare.png` has one panel per run (colour = engine, solid = GPU,
dashed = CPU, circle = fp64, square = fp32); `compare_qumo_gpu.png` overlays the QUMO GPU series of all
runs, which is the cross-vendor view.

The report also has a *How each series was timed* section (each engine's timed region and fusion settings,
from the rows) and, when `bench_gpu` reports launches, a *QUMO GPU: state sweeps and effective bandwidth*
table. It applies these rules:

* QUMO GPU rows of runs with different GPU engine sources (`QUMO GPU engine` row: a hash of `qumo/gpu`, the
  plan and `bench_gpu`; older runs without it and with a dirty tree count as unidentifiable) are never put in
  one table, and the cross-machine plot is marked. The first T4 run (2026-10-12) used the older engine with
  one launch per plan op, so its QUMO GPU rows cannot be compared with the M5 tile-engine rows.
* Widths of 20 qubits and fewer are marked with a dagger: fixed per-call overhead dominates there.
* fp64 on GPUs with slow fp64 units (T4, L4, RTX, consumer AMD) is flagged: it measures fp64 arithmetic,
  not memory bandwidth, and does not carry over to A100 / H100 / MI300.
* fp32 with fp32, fp64 with fp64 (rows record `precision`, what ran, and `precision_requested`); runs whose
  load average stayed below the core count, with no GPU throttling and (on a Mac) Low Power Mode off are the ones
  to trust. GPU engines run in fp32 unless the run asked for fp64 (`INCLUDE_FP64`, `--fp64`).

## Share results

* Open a GitHub issue with the zip from step 7 (or the JSON) attached, or
* open a pull request adding the JSON (and CSV) to `bench/results/cloud/`.

The JSON contains no host names, user paths or credentials. Please keep `--label` free of personal data.
