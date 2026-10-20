#!/usr/bin/env bash
# Scratch venv for the IBM AI transpiler baseline (never the repository's
# bench/.venv; nothing here is a QUMO dependency or gets committed).
#   bench/ai_vs_ibm/setup_venv.sh /tmp/ibm_ai        # creates /tmp/ibm_ai/venv and /tmp/ibm_ai/hf_home
set -euo pipefail
DIR=${1:?usage: setup_venv.sh <scratch dir>}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$DIR"
cd "$DIR"
uv venv --python 3.12 venv
# Versions of the published run (bench/results/ai_vs_ibm.md). qiskit-ibm-transpiler
# 0.18.x pulls qiskit-gym, twisterl and torch for its local synthesis; its local
# routing is a Rust extension inside the wheel. The legacy package
# qiskit-ibm-ai-local-transpiler (no declared license) is NOT needed by 0.18.x.
VIRTUAL_ENV=$DIR/venv uv pip install --python venv/bin/python \
  "qiskit==2.5.2" "qiskit-ibm-transpiler==0.18.0" "qiskit-ibm-runtime==0.45.1" "qiskit-aer==0.18.0" numpy
# IBM's models (Hugging Face qiskit/ai-transpiler_*, model cards: apache-2.0) are
# cached in the scratch dir only; afterwards every run uses HF_HUB_OFFLINE=1.
HF_HOME=$DIR/hf_home venv/bin/python -I - <<'EOF'
from huggingface_hub import snapshot_download
for r in ("routing", "linear-functions", "cliffords", "permutations", "paulis"):
    print(snapshot_download(repo_id=f"qiskit/ai-transpiler_{r}", revision="main"))
EOF
# QUMO's CPython extension (Python 3.12) must exist: make py (build/qumo_py.so)
[ -f "$ROOT/build/qumo_py.so" ] || (cd "$ROOT" && make py)
cat <<EOF
done. Run:
  export HF_HOME=$DIR/hf_home HF_HUB_OFFLINE=1 IBMAI_VENV=$DIR/venv
  $ROOT/bench/ai_vs_ibm/campaign.sh $ROOT/bench/results/ai_vs_ibm.jsonl
  $DIR/venv/bin/python $ROOT/bench/ai_vs_ibm/report.py $ROOT/bench/results/ai_vs_ibm.jsonl > $ROOT/bench/results/ai_vs_ibm.md
EOF
