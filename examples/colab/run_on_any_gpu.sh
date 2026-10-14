#!/usr/bin/env bash
# Headless QUMO benchmark on any machine: clone -> install -> build -> run.
#
#   curl -fsSLO https://raw.githubusercontent.com/ANFAIA/QUMO/main/examples/colab/run_on_any_gpu.sh
#   bash run_on_any_gpu.sh                       # auto: NVIDIA / AMD / Apple / CPU only
#   bash run_on_any_gpu.sh --quick               # smoke test (minutes)
#   bash run_on_any_gpu.sh --fp64                # GPU engines in fp32 AND fp64 (default: fp32 only)
#   bash run_on_any_gpu.sh --config colab_a100 --suites sv_gpu,baseline_cudaq --label "lambda A100"
#   bash run_on_any_gpu.sh --ref 0dc37a7 --dir ~/src/QUMO --no-baselines
#   bash run_on_any_gpu.sh --mojo-version "" --max-version ""   # latest Mojo/MAX instead of the pinned ones
#
# Options:
#   --config NAME|PATH   preset from bench/configs/ or a JSON path (default: auto, picked from the GPU)
#   --quick              tiny sizes (16-20 qubits), 1 repetition
#   --suites a,b,...     run (and build) only these suites (default: the preset's enabled suites)
#   --fp64               also run every GPU engine in fp64 (QUMO GPU, CUDA-Q, Aer GPU, Lightning GPU);
#                        default fp32 only, CUDA-Q's default and QUMO's. CPU references are unchanged.
#   --label TEXT         free text stored in the result (no personal data)
#   --ref REF            branch, tag or commit to clone (default: main)
#   --repo URL           repository to clone (default: https://github.com/ANFAIA/QUMO)
#   --dir PATH           checkout to use or clone into (default: this checkout, else ./QUMO)
#   --out PATH           result .json or directory (default: bench/results/cloud/)
#   --no-baselines       do not install the vendor GPU baselines (their suites record a skip)
#   --mojo-version V / --max-version V   pinned toolchain (default 1.1.0 / 26.6.0; "" = latest)
#
# Inside an existing QUMO checkout it uses that checkout (no clone). GPU vendor
# detection: nvidia-smi -> NVIDIA, rocminfo / amd-smi / rocm-smi -> AMD,
# macOS arm64 -> Apple (Metal), otherwise CPU only. Vendor-specific baselines go
# into their own virtual environments under $QUMO_VENVS (default <dir>/build/baseline-venvs):
#   NVIDIA: CUDA-Q (cuda-quantum-cu12/13), qiskit-aer-gpu-cu11 (latest Aer + Qiskit; fallback qiskit-aer-gpu 0.15.1)
#   AMD:    PennyLane lightning.amdgpu (+ lightning.qubit)
#   Apple:  PennyLane lightning.qubit (CPU reference)
# Results: bench/results/cloud/*.json + .csv (see examples/colab/README.md).
set -euo pipefail

REPO_URL="https://github.com/ANFAIA/QUMO"
REF="main"
DIR=""
CONFIG="auto"
QUICK=""
FP64=""
SUITES=""
LABEL=""
BASELINES=1
OUT="bench/results/cloud/"
MOJO_VERSION="1.1.0"   # pinned: the versions QUMO is developed and benchmarked with
MAX_VERSION="26.6.0"

usage() {  # the comment block at the top of this file, up to the first non-comment line
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
  exit "${1:-0}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --quick) QUICK="--quick"; shift ;;
    --fp64) FP64="--fp64"; shift ;;
    --suites) SUITES="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --repo) REPO_URL="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --no-baselines) BASELINES=0; shift ;;
    --mojo-version) MOJO_VERSION="$2"; shift 2 ;;
    --max-version) MAX_VERSION="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "QM-E960: unknown option '$1'. Hint: run with --help." >&2; exit 2 ;;
  esac
done

log() { printf '\n==> %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- GPU vendor
VENDOR="none"
if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
  VENDOR="nvidia"
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
  SMI_DRIVERS="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader || true)"
  DRIVER_MAJOR="${SMI_DRIVERS%%.*}"
  SMI_FULL="$(nvidia-smi || true)"
  CUDA_MAJOR="$(printf '%s\n' "$SMI_FULL" | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p')"
  CUDA_MAJOR="${CUDA_MAJOR%%[!0-9]*}"
  if [ "${DRIVER_MAJOR:-999}" -lt 580 ] && [ -z "${MODULAR_NVPTX_COMPILER_PATH:-}" ]; then
    for p in /usr/local/cuda/bin/ptxas "$(command -v ptxas || true)"; do
      if [ -n "$p" ] && [ -x "$p" ]; then export MODULAR_NVPTX_COMPILER_PATH="$p"; break; fi
    done
    echo "NOTE: driver $DRIVER_MAJOR < 580 (Mojo/MAX 26.6 wants >= 580); MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-unset}"
  fi
elif ROCMINFO="$( (have rocminfo && rocminfo 2>/dev/null) || true)"; \
     case "$ROCMINFO" in *gfx[0-9]*) true ;; *) have amd-smi && amd-smi list >/dev/null 2>&1 ;; esac; then
  VENDOR="amd"
  (have rocm-smi && rocm-smi --showproductname) || (have amd-smi && amd-smi list) || true
  printf '%s\n' "$ROCMINFO" | grep -o 'gfx[0-9a-f]*' | sort -u || true
elif [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  VENDOR="apple"
  system_profiler SPDisplaysDataType 2>/dev/null | grep -E 'Chipset Model|Total Number of Cores|Metal' || true
  if ! xcrun -f metal >/dev/null 2>&1; then
    echo "NOTE: Metal toolchain missing; QUMO GPU suites will be skipped. Fix: xcodebuild -downloadComponent MetalToolchain"
  fi
fi
log "GPU vendor: $VENDOR"

# ---------------------------------------------------------------- required tools
for t in git python3; do
  have "$t" || { echo "QM-E963: '$t' not found. Hint: install it (e.g. apt-get install -y $t)." >&2; exit 1; }
done
for t in make cc; do  # make: gpu_check / tests suites; cc: Mojo links with the system C compiler on Linux
  have "$t" || echo "NOTE: '$t' not found; install it (e.g. apt-get install -y build-essential) or some suites fail."
done

# ---------------------------------------------------------------- checkout
in_checkout() { [ -f "$1/Makefile" ] && [ -d "$1/qumo" ] && [ -f "$1/bench/run_benchmarks.py" ]; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "$DIR" ]; then
  if in_checkout "$HERE/../.."; then DIR="$(cd "$HERE/../.." && pwd)";
  elif in_checkout "$PWD"; then DIR="$PWD";
  else DIR="$PWD/QUMO"; fi
fi
if in_checkout "$DIR"; then
  log "using checkout $DIR"
else
  log "cloning $REPO_URL ($REF) into $DIR"
  if git clone --depth 1 --branch "$REF" "$REPO_URL" "$DIR" 2>/dev/null; then :; else
    git clone --filter=blob:none "$REPO_URL" "$DIR"
    git -C "$DIR" checkout "$REF"
  fi
fi
cd "$DIR"
git log -1 --format='%H %cd %s' --date=iso || true

# ---------------------------------------------------------------- toolchain
if ! have uv; then
  log "installing uv (python3 -m pip install --user uv)"
  python3 -m pip install --user -q uv || python3 -m pip install --user -q --break-system-packages uv
  # pip --user puts scripts in <user base>/bin (~/.local/bin on Linux, ~/Library/Python/3.x/bin on macOS)
  PATH="$(python3 -m site --user-base)/bin:$HOME/.local/bin:$PATH"
  export PATH
fi
if [ ! -x .venv/bin/mojo ]; then
  if [ -n "$MOJO_VERSION" ] && [ -n "$MAX_VERSION" ]; then
    log "Mojo $MOJO_VERSION + MAX $MAX_VERSION into .venv"
    uv venv -q .venv --python 3.12
    uv pip install -q --python .venv/bin/python "mojo==$MOJO_VERSION" "max-mojo-libs==$MAX_VERSION" "max-core==$MAX_VERSION"
  else
    log "make setup (latest Mojo + MAX into .venv)"
    make setup
  fi
fi
.venv/bin/mojo --version 2>&1 | grep -v Crashpad || true
if [ ! -x bench/.venv/bin/python ]; then uv venv -q bench/.venv --python 3.12; fi
uv pip install -q -U --python bench/.venv/bin/python qiskit qiskit-aer pytket pytket-qiskit stim pymatching numpy matplotlib psutil

# ---------------------------------------------------------------- vendor baselines (own venvs)
VENVS="${QUMO_VENVS:-$DIR/build/baseline-venvs}"
mkvenv() {  # name env_var packages...   (never fatal: a missing baseline is a recorded skip)
  local name="$1" var="$2"; shift 2
  local py="$VENVS/$name/bin/python"
  if [ ! -x "$py" ] && ! uv venv -q "$VENVS/$name" --python 3.12; then
    echo "NOTE: could not create $VENVS/$name (the suite will record a skip)"; return 0
  fi
  if uv pip install -q -U --python "$py" "$@"; then export "$var=$py"; echo "$var=$py";
  else echo "NOTE: could not install $* (the suite will record a skip)"; fi
}
if [ "$BASELINES" = 1 ]; then
  log "baselines for $VENDOR"
  case "$VENDOR" in
    nvidia)
      if [ "${CUDA_MAJOR:-12}" -ge 13 ]; then mkvenv cudaq QUMO_CUDAQ_PYTHON cuda-quantum-cu13 numpy;
      else mkvenv cudaq QUMO_CUDAQ_PYTHON cuda-quantum-cu12 numpy; fi
      if [ "$(uname -m)" = "x86_64" ]; then  # latest Aer + Qiskit; fallback: last CUDA-12 wheel (Qiskit < 2)
        mkvenv aer-gpu QUMO_AER_GPU_PYTHON qiskit qiskit-aer-gpu-cu11 numpy
        [ -n "${QUMO_AER_GPU_PYTHON:-}" ] || mkvenv aer-gpu-legacy QUMO_AER_GPU_PYTHON 'qiskit>=1.1,<2' qiskit-aer-gpu numpy
      fi
      mkvenv lightning QUMO_LIGHTNING_PYTHON pennylane pennylane-lightning pennylane-lightning-gpu numpy ;;
    amd)
      mkvenv lightning QUMO_LIGHTNING_PYTHON pennylane pennylane-lightning pennylane-lightning-amdgpu numpy ;;
    *)
      mkvenv lightning QUMO_LIGHTNING_PYTHON pennylane pennylane-lightning numpy ;;
  esac
fi

# ---------------------------------------------------------------- build + run
SEL=(--config "$CONFIG")
[ -n "$SUITES" ] && SEL+=(--suites "$SUITES")
if [ -n "$FP64" ]; then
  if grep -q -- '"--fp64"' bench/run_benchmarks.py; then SEL+=("$FP64");
  else echo "NOTE: this QUMO version's runner has no --fp64 switch: it runs the precisions its preset lists"; fi
fi
ARGS=("${SEL[@]}" --out "$OUT")
[ -n "$QUICK" ] && ARGS+=("$QUICK")
[ -n "$LABEL" ] && ARGS+=(--label "$LABEL")
log "build what the selected suites need (failures are recorded by the run, not fatal)"
bench/.venv/bin/python bench/run_benchmarks.py "${SEL[@]}" ${QUICK:+"$QUICK"} --build-only || true  # --quick: no variant builds
log "load average at start: $(uptime)"
bench/.venv/bin/python bench/run_benchmarks.py "${ARGS[@]}"
log "done. Compare runs with: bench/.venv/bin/python examples/colab/compare_results.py $OUT*.json --out compare.md --plot compare.png"
