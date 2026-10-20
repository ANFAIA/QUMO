#!/usr/bin/env bash
# Full QUMO-AI vs IBM AI transpiler campaign (sequential; per-phase deadlines
# in minutes stop starting new circuits, so the whole run stays near 2 h).
#   IBMAI_VENV=/path/to/scratch/venv bench/ai_vs_ibm/campaign.sh [out.jsonl]
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${IBMAI_VENV:?set IBMAI_VENV to the scratch venv (bench/ai_vs_ibm/README.md)}/bin/python
OUT=${1:-$ROOT/bench/results/ai_vs_ibm.jsonl}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_HUB_DISABLE_TELEMETRY=1 PYTHONWARNINGS=ignore
export QUMO_MODEL_DIR=${QUMO_MODEL_DIR:-$ROOT/models/ai_transpiler}
RUN="$PY $ROOT/bench/ai_vs_ibm/run.py --out $OUT --timeout 600"
cd "$ROOT"
echo "== families fake_torino (all contenders) $(date)"
$RUN --groups families --devices fake_torino --contenders all --deadline-min 30
echo "== neutral heavy-hex (main) $(date)"
$RUN --groups neutral --devices hh --contenders main --deadline-min 10
echo "== summit fake_torino (main) $(date)"
$RUN --groups summit --devices fake_torino --contenders main --deadline-min 25
echo "== families fake_fez (main) $(date)"
$RUN --groups families --devices fake_fez --contenders main --deadline-min 15
echo "== feynman fake_torino (main) $(date)"
$RUN --groups feynman --devices fake_torino --contenders main --deadline-min 10
echo "== hamlib fake_torino (main) $(date)"
$RUN --groups hamlib --devices fake_torino --contenders main --deadline-min 15
echo "== done $(date)"
