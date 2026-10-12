#!/usr/bin/env bash
# Re-measure QUMO after compiler changes (Benchpress full + neutral harness).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$ROOT/bench/.venv/bin/python
BP=$ROOT/.ref/benchpress
OUT=$ROOT/bench/results
# wait for any running neutral benchmark to finish (sequential timings)
while pgrep -f run_neutral.py > /dev/null; do sleep 20; done
"$ROOT/bench/benchpress/install.sh" "$BP"
[ -f "$OUT/benchpress/qumo_full.json" ] && mv "$OUT/benchpress/qumo_full.json" "$OUT/benchpress/qumo_full_round1.json"
cd "$BP"
echo "== benchpress qumo $(date)"
$PY -m pytest benchpress/qumo_gym -q -p no:cacheprovider \
    --benchmark-json="$OUT/benchpress/qumo_full.json" > "$OUT/benchpress/qumo_full.log" 2>&1
tail -1 "$OUT/benchpress/qumo_full.log"
cd "$ROOT"
echo "== neutral qumo $(date)"
$PY - <<'PYEOF'
import json
p = "bench/results/neutral.jsonl"
keep = [l for l in open(p) if json.loads(l)["sdk"] != "qumo"]
open(p, "w").writelines(keep)
PYEOF
$PY bench/neutral/run_neutral.py --circuits bench/neutral/circuits --sdks qumo --targets all \
    --levels default,max --out "$OUT/neutral.jsonl" --timeout 120 > "$OUT/neutral_qumo.log" 2>&1
tail -1 "$OUT/neutral_qumo.log"
echo "== done $(date)"
