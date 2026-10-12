#!/usr/bin/env bash
# Final benchmark campaign (sequential, so timings do not interfere).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$ROOT/bench/.venv/bin/python
BP=$ROOT/.ref/benchpress
OUT=$ROOT/bench/results
mkdir -p "$OUT/benchpress"
"$ROOT/bench/benchpress/install.sh" "$BP"
cd "$BP"
for g in qumo qiskit; do
  echo "== benchpress $g $(date)"
  $PY -m pytest benchpress/${g}_gym -q -p no:cacheprovider \
      --benchmark-json="$OUT/benchpress/${g}_full.json" > "$OUT/benchpress/${g}_full.log" 2>&1
  tail -3 "$OUT/benchpress/${g}_full.log"
done
cd "$ROOT"
echo "== neutral $(date)"
rm -f "$OUT/neutral.jsonl"
$PY bench/neutral/run_neutral.py --circuits bench/neutral/circuits --sdks qumo,qiskit,tket,cirq \
    --targets all --levels default,max --out "$OUT/neutral.jsonl" --timeout 120 > "$OUT/neutral.log" 2>&1
tail -3 "$OUT/neutral.log"
echo "== done $(date)"
