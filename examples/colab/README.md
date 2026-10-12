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

Everything calls the same runner, [`bench/run_benchmarks.py`](../../bench/run_benchmarks.py), with a
preset from [`bench/configs/`](../../bench/configs/).

## Google Colab (NVIDIA)

1. Open the notebook:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ANFAIA/QUMO/blob/main/examples/colab/QUMO_GPU_Benchmark_Colab.ipynb)
2. *Runtime -> Change runtime type -> Hardware accelerator:* pick a GPU. For a fair first look use the
   free **T4**; with Colab Pro / pay-as-you-go try **L4**, **A100** (the *High-RAM* switch gives the 80 GB
   card and more host RAM), **H100** or **G4** (RTX PRO 6000 Blackwell).
3. *Runtime -> Run all.* Step 1 reads `nvidia-smi` and picks the preset (`colab_t4`, `colab_l4`,
   `colab_a100`, `colab_h100`, `colab_g4`); override it in the form, or tick `QUICK` for a smoke test.
4. Step 7 downloads a zip with the JSON, CSV and plots (optionally also to Google Drive).

To compare NVIDIA configurations, change the runtime type, *Runtime -> Restart session*, and run all
again. Each run produces its own JSON.

### Expected duration and cost (estimates)

| Runtime | Full preset | `QUICK` | Approx. cost |
|---|---|---|---|
| T4 (free tier) | 40-70 min | ~15 min | free (session limits apply) or ~1.2 units/h |
| L4 | 35-60 min | ~15 min | ~1.7 units/h (~$0.17/h) |
| A100 | 35-60 min | ~12 min | ~5.4-7.5 units/h (~$0.54-0.75/h) |
| H100 / G4 | 40-70 min | ~12 min | ~9 units/h or more |

About 10-15 minutes of every run is the one-off download of Mojo/MAX (~1 GB) and the builds; most of
the rest is CPU work (the QUMO CPU and Aer CPU references; `bench_gpu` also runs the CPU reference at
every width). These are estimates from the presets' sizes, not yet measured on Colab; the result JSON
records the real `duration_s` of each suite. Unit rates: March 2026 measurements (see the doc's sources).

## AMD GPUs (ROCm, no Colab)

Colab has no AMD GPUs. Use any Jupyter on a ROCm machine:

* **AMD Developer Cloud** (MI300X droplets): create a GPU droplet with the ROCm image, open its
  JupyterLab, then upload `QUMO_GPU_Benchmark_AMD.ipynb` (or `git clone https://github.com/ANFAIA/QUMO`
  and open it from `examples/colab/`). Run all cells.
* **Azure ND MI300X v5**, other MI300X/MI325X/MI355X providers, or a local Radeon RX 7000/9000 /
  Instinct machine: `pip install jupyterlab && jupyter lab`, same notebook.
* Headless instead: `bash examples/colab/run_on_any_gpu.sh`.

The `amd_mi300x` preset runs QUMO CPU/GPU (fp32 and fp64; MI300X has full-rate fp64), Aer CPU and
PennyLane Lightning CPU + `lightning.amdgpu` (the AMD-native GPU baseline; CUDA-Q and Aer GPU are
CUDA-only and are recorded as *not applicable*). Expect ~30-60 min on one MI300X.

## Any machine, headless

```bash
curl -fsSLO https://raw.githubusercontent.com/ANFAIA/QUMO/main/examples/colab/run_on_any_gpu.sh
bash run_on_any_gpu.sh                    # clone -> install -> build -> run (auto preset)
bash run_on_any_gpu.sh --quick            # smoke test
bash run_on_any_gpu.sh --config colab_a100 --suites sv_gpu,baseline_cudaq --label "A100 PCIe, lab server"
# from inside a checkout it uses that checkout: bash examples/colab/run_on_any_gpu.sh --quick
```

On a Mac it runs the `apple_metal` preset (QUMO GPU in fp32 through Metal) as a local comparison point;
it needs Xcode and `xcodebuild -downloadComponent MetalToolchain`.

## Compare runs

```bash
bench/.venv/bin/python examples/colab/compare_results.py \
    t4.json l4.json a100.json mi300x.json m5.json --out compare.md --plot compare.png
```

`compare.md` holds: the machines (GPU, memory, arch, driver/ROCm, CPU, RAM, OS, Mojo/MAX, commit, date,
load average), the status of every suite, one timing table (rows = run x engine/device/precision,
columns = qubits), speed-ups against `--reference` (default `Aer CPU fp64`) and every skipped or failed
part with its `QM-E96x` reason. `compare.png` has one panel per run (colour = engine, solid = GPU,
dashed = CPU, circle = fp64, square = fp32); `compare_qumo_gpu.png` overlays the QUMO GPU series of all
runs, which is the cross-vendor view.

Compare like with like: same QUMO commit (the table shows it), fp32 with fp32, fp64 with fp64, widths of
22 qubits and more for GPUs (below that, launch and Python overheads dominate), and runs whose load
average stayed below the core count.

## Share results

* Open a GitHub issue with the zip from step 7 (or the JSON) attached, or
* open a pull request adding the JSON (and CSV) to `bench/results/cloud/`.

The JSON contains no host names, user paths or credentials. Please keep `--label` free of personal data.
