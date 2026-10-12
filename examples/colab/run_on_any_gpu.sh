#!/usr/bin/env bash
# Headless QUMO benchmark on any machine: clone -> install -> build -> run.
#
#   curl -fsSLO https://raw.githubusercontent.com/ANFAIA/QUMO/main/examples/colab/run_on_any_gpu.sh
#   bash run_on_any_gpu.sh                       # auto: NVIDIA / AMD / Apple / CPU only
#   bash run_on_any_gpu.sh --quick               # smoke test (minutes)
#   bash run_on_any_gpu.sh --config colab_a100 --suites sv_gpu,baseline_cudaq --label "lambda A100"
#   bash run_on_any_gpu.sh --ref 0dc37a7 --dir ~/src/QUMO --no-baselines
#   bash run_on_any_gpu.sh --mojo-version "" --max-version ""   # latest Mojo/MAX instead of the pinned ones
#
# Inside an existing QUMO checkout it uses that checkout (no clone). GPU vendor
# detection: nvidia-smi -> NVIDIA, rocminfo / amd-smi / rocm-smi -> AMD,
# macOS arm64 -> Apple (Metal), otherwise CPU only. Vendor-specific baselines go
# into their own virtual environments under $QUMO_VENVS (default <dir>/build/baseline-venvs):
#   NVIDIA: CUDA-Q (cuda-quantum-cu12/13), qiskit-aer-gpu (Qiskit < 2)
#   AMD:    PennyLane lightning.amdgpu (+ lightning.qubit)
#   Apple:  PennyLane lightning.qubit (CPU reference)
# Results: bench/results/cloud/*.json + .csv (see examples/colab/README.md).
set -euo pipefail

REPO_URL="https://github.com/ANFAIA/QUMO"
REF="main"
DIR=""
CONFIG="auto"
QUICK=""
SUITES=""
LABEL=""
BASELINES=1
OUT="bench/results/cloud/"
MOJO_VERSION="1.1.0"   # pinned: the versions QUMO is developed and benchmarked with
MAX_VERSION="26.6.0"

usage() { sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --quick) QUICK="--quick"; shift ;;
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
uv pip install -q --python bench/.venv/bin/python qiskit qiskit-aer pytket pytket-qiskit stim pymatching numpy matplotlib psutil

# ---------------------------------------------------------------- vendor baselines (own venvs)
VENVS="${QUMO_VENVS:-$DIR/build/baseline-venvs}"
mkvenv() {  # name env_var packages...   (never fatal: a missing baseline is a recorded skip)
  local name="$1" var="$2"; shift 2
  local py="$VENVS/$name/bin/python"
  if [ ! -x "$py" ] && ! uv venv -q "$VENVS/$name" --python 3.12; then
    echo "NOTE: could not create $VENVS/$name (the suite will record a skip)"; return 0
  fi
  if uv pip install -q --python "$py" "$@"; then export "$var=$py"; echo "$var=$py";
  else echo "NOTE: could not install $* (the suite will record a skip)"; fi
}
if [ "$BASELINES" = 1 ]; then
  log "baselines for $VENDOR"
  case "$VENDOR" in
    nvidia)
      if [ "${CUDA_MAJOR:-12}" -ge 13 ]; then mkvenv cudaq QUMO_CUDAQ_PYTHON cuda-quantum-cu13 numpy;
      else mkvenv cudaq QUMO_CUDAQ_PYTHON cuda-quantum-cu12 numpy; fi
      if [ "$(uname -m)" = "x86_64" ]; then mkvenv aer-gpu QUMO_AER_GPU_PYTHON 'qiskit>=1.1,<2' qiskit-aer-gpu numpy; fi
      mkvenv lightning QUMO_LIGHTNING_PYTHON pennylane pennylane-lightning pennylane-lightning-gpu numpy ;;
    amd)
      mkvenv lightning QUMO_LIGHTNING_PYTHON pennylane pennylane-lightning pennylane-lightning-amdgpu numpy ;;
    *)
      mkvenv lightning QUMO_LIGHTNING_PYTHON pennylane pennylane-lightning numpy ;;
  esac
fi

# ---------------------------------------------------------------- build + run
ARGS=(--config "$CONFIG" --out "$OUT")
[ -n "$QUICK" ] && ARGS+=("$QUICK")
[ -n "$SUITES" ] && ARGS+=(--suites "$SUITES")
[ -n "$LABEL" ] && ARGS+=(--label "$LABEL")
log "build (failures are recorded by the run, not fatal)"
bench/.venv/bin/python bench/run_benchmarks.py --config "$CONFIG" --build-only || true
log "load average at start: $(uptime)"
bench/.venv/bin/python bench/run_benchmarks.py "${ARGS[@]}"
log "done. Compare runs with: bench/.venv/bin/python examples/colab/compare_results.py $OUT*.json --out compare.md --plot compare.png"
