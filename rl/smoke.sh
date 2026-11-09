#!/bin/sh
# End-to-end smoke of the native training stack (MOJO_PORT.md 14: A5 synthesis, A6 router, A7 export and verify).
#
#   rl/smoke.sh [AI_BUILD]        (default AI_BUILD = build/ai_train; make train-ai-smoke runs it from rl/ with an absolute AI_BUILD)
#
# Needs the tools in AI_BUILD/bin (make train-ai-smoke builds them): train_synth export_synth eval_synth route_tool
# train_route export_route models_verify. Everything is written under AI_BUILD/smoke; models/ is never touched.
# About 5 minutes on the reference machine (4 cores). Exit status 0 only when every check passed.
#
# A5  synthesis: train a 4-qubit model with a fixed seed and an iteration bound (never a clock bound), then evaluate
#     it on 100 seeded blocks with the keep rule of eval_synth (beam-8 policy against QUMO's greedy synthesizer:
#     win_or_tie >= 0.6, cost_ratio < 1.0, no verification failure). Candidates in order: lf on ring:4, lf on line:4,
#     perm on ring:5; the first one that passes is the smoke model (the choice is frozen by the first line of the
#     candidate list below). Independently of the cost rule the run must show learning: curriculum difficulty >= 12
#     and the greedy policy solving >= 95 of 100 blocks within max_depth.
# A6  router: behaviour cloning on teacher data from generated jobs plus one DAgger round; the cloned policy must
#     beat its own first epoch (validation agreement up, regret down).
# A7  export: both models are exported into a staging root, load, replay their embedded vectors (the exporters check
#     this before they write) and pass models_verify.
#
# Overrides (environment): SMOKE_CORES (4), SMOKE_ITERS (150), SMOKE_MAX_TRAIN_S (150), SMOKE_ROUTE_JOBS (400).

set -eu

B=${1:-build/ai_train}
BIN=$B/bin
S=$B/smoke
CORES=${SMOKE_CORES:-4}
ITERS=${SMOKE_ITERS:-150}
MAX_TRAIN_S=${SMOKE_MAX_TRAIN_S:-150}
ROUTE_JOBS=${SMOKE_ROUTE_JOBS:-400}

# kind:topology of the synthesis candidates, in order
CANDIDATES="lf:ring:4 lf:line:4 perm:ring:5"

fail() {
    echo "smoke: FAIL: $*" >&2
    exit 1
}

for t in train_synth export_synth eval_synth route_tool train_route export_route models_verify; do
    [ -x "$BIN/$t" ] || fail "tool $BIN/$t not found (build it with: make train-ai-smoke)"
done

# value of a top-level or nested  "KEY": VALUE  line of a pretty-printed JSON file (first match)
field() {
    sed -n "s/^ *\"$2\": *\([^,]*\),*\$/\1/p" "$1" | head -n 1 | tr -d '"'
}

# is $1 >= $2 (decimal numbers)
ge() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 >= b + 0) }'
}

rm -rf "$S"
mkdir -p "$S/models/synth" "$S/models/route"
T0=$(date +%s)

# ------------------------------------------------------------------------------------------------ A5 synthesis
SPEC=""
for cand in $CANDIDATES; do
    kind=${cand%%:*}
    topo=${cand#*:}
    name=${kind}_$(echo "$topo" | tr -d ':')
    run=$S/synth/$name
    echo "smoke: A5 candidate $kind on $topo ($ITERS iterations, $CORES cores)"
    t1=$(date +%s)
    "$BIN/train_synth" --kind "$kind" --topology "$topo" --out "$run" --iters "$ITERS" --cores "$CORES" --seed 1 \
        --quiet || fail "train_synth failed for $name"
    secs=$(($(date +%s) - t1))
    echo "smoke: A5 trained $name in ${secs}s"
    [ "$secs" -le "$MAX_TRAIN_S" ] || fail "training $name took ${secs}s (limit ${MAX_TRAIN_S}s)"
    [ "$(field "$run/summary.json" finished)" = "true" ] || fail "$name: summary says unfinished"
    bestd=$(field "$run/summary.json" best_difficulty)
    ge "$bestd" 12 || fail "$name did not learn: best curriculum difficulty $bestd < 12"
    mkdir -p "$S/synth/$name.work"
    "$BIN/eval_synth" "$run" --id "smoke_$name" --root "$S/models/synth" --work "$S/synth/$name.work" \
        --instances 100 > "$S/synth/$name.eval" || fail "eval_synth failed for $name"
    tail -n 1 "$S/synth/$name.eval"
    solved=$(awk '/"policy_greedy"/ { f = 1 } f && /"solved"/ { gsub(/[^0-9.]/, "", $2); print $2; exit }' "$run/eval_final.json")
    [ -n "$solved" ] || fail "$name: no policy_greedy.solved in eval_final.json"
    if tail -n 1 "$S/synth/$name.eval" | grep -q '"kept": *true'; then
        ge "$solved" 0.95 || fail "$name kept, but the greedy policy solved only $solved of the blocks (need 0.95)"
        SPEC=$name
        break
    fi
    echo "smoke: A5 $name did not beat greedy (keep rule), trying the next candidate"
done
[ -n "$SPEC" ] || fail "no synthesis candidate beat QUMO's greedy synthesizer under the keep rule (MOJO_PORT.md O-7: extend the time budget first, then widen the candidates; never relax the rule)"
echo "smoke: A5 OK with $SPEC"

# ------------------------------------------------------------------------------------------------ A6 router
R=$S/route
mkdir -p "$R/stage"
echo "smoke: A6 router round 0 ($ROUTE_JOBS jobs)"
"$BIN/route_tool" gen-jobs 31 "$ROUTE_JOBS" 30 0.5 "$R/jobs0.json" > /dev/null || fail "gen-jobs round 0"
"$BIN/route_tool" gen-jobs 32 "$ROUTE_JOBS" 30 0.5 "$R/jobs1.json" > /dev/null || fail "gen-jobs round 1"
ROUTE_TOOL_CAP=96 "$BIN/route_tool" gen "$R/jobs0.json" "$R/d0.bin" - 1.0 401 heuristic 16 > /dev/null || fail "teacher data round 0"
"$BIN/train_route" --data "$R/d0.bin" --out "$R/run0" --hidden 64 --tau 0.5 --epochs 30 --seed 0 --quiet || fail "train_route round 0"
"$BIN/export_route" "$R/run0" --root "$R/stage" --id round0 --default --episodes 1 || fail "export_route round 0"
echo "smoke: A6 DAgger round"
ROUTE_TOOL_CAP=96 "$BIN/route_tool" gen "$R/jobs1.json" "$R/d1.bin" "$R/stage/round0" 0.3 402 heuristic 16 > /dev/null || fail "teacher data round 1"
"$BIN/train_route" --data "$R/d0.bin" "$R/d1.bin" --out "$R/run1" --hidden 64 --tau 0.5 --epochs 30 --seed 0 --quiet || fail "train_route round 1"
first=$(grep '"val_agree"' "$R/run1/train_log.json" | head -n 1 | sed 's/[^0-9.]//g')
last=$(grep '"val_agree"' "$R/run1/train_log.json" | tail -n 1 | sed 's/[^0-9.]//g')
rfirst=$(grep '"val_regret"' "$R/run1/train_log.json" | head -n 1 | sed 's/[^0-9.]//g')
rlast=$(grep '"val_regret"' "$R/run1/train_log.json" | tail -n 1 | sed 's/[^0-9.]//g')
echo "smoke: A6 validation agreement $first -> $last, regret $rfirst -> $rlast"
ge "$last" "$first" || fail "router agreement did not improve ($first -> $last)"
ge "$rfirst" "$rlast" || fail "router regret did not improve ($rfirst -> $rlast)"
"$BIN/export_route" "$R/run1" --root "$S/models/route" --id smoke_route --default --episodes 2 || fail "export_route final"
echo "smoke: A6 OK"

# ------------------------------------------------------------------------------------------------ A7 verify
echo "smoke: A7 models-verify on the staging root"
QUMO_MODEL_DIR="$S/models" "$BIN/models_verify" > "$S/verify.out" || { cat "$S/verify.out"; fail "models_verify reported problems"; }
cat "$S/verify.out"
grep -q ' 0 problems' "$S/verify.out" || fail "models_verify did not report 0 problems"
[ -f "$S/models/synth/smoke_$SPEC/manifest.json" ] || fail "synthesis model missing from the staging root"
[ -f "$S/models/route/smoke_route/manifest.json" ] || fail "router model missing from the staging root"
echo "smoke: A7 OK"

echo "smoke: OK ($(($(date +%s) - T0))s, synthesis model $SPEC, staging root $S/models)"
