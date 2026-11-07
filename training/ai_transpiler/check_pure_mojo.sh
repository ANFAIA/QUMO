#!/bin/sh
# make check-pure-mojo: the AI-transpiler training path must not invoke or name Python, Rust, torch or numpy.
#
#   training/ai_transpiler/check_pure_mojo.sh [--smoke]        (or CHECK_PURE_SMOKE=1)
#
# What is claimed (MOJO_PORT.md, acceptance A8): no Python or Rust is invoked or named by QUMO's training sources,
# by the Makefile recipes of the training targets, or by the built training binaries. What is not claimed: the pinned
# Mojo toolchain itself. The repo's .venv/bin/mojo is a Python entry-point script with an absolute shebang and
# `make setup` installs the toolchain with uv and pip; the README says so.
#
# Four parts, each printing PASS or FAIL:
#   1. source scan    word-boundary regexes over every non-Markdown file under training/ and qumo/rl and over the
#                     expanded recipes of `make -n train-ai train-ai-smoke train-ai-route-ppo models-verify`
#   2. environment    `make -n train-ai ...` under `env -i` with a PATH that holds only the coreutils, make, sh and a C
#                     compiler (no python, uv, pip, cargo or rustc); with --smoke the real `make train-ai-smoke` runs
#                     there to completion (about 5 minutes) and the recording shims must stay unused
#   3. binary scan    otool -L (macOS) or ldd (Linux) of every build/ai_train/bin/* lists no libpython / libtorch
#   4. tree           no .py under training/ai_transpiler and no requirements.txt there
#
# Exit status 0 only when every part passed.

set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 2
SMOKE=${CHECK_PURE_SMOKE:-0}
[ "${1:-}" = "--smoke" ] && SMOKE=1
AI_BUILD=${AI_BUILD:-build/ai_train}

# \b python3? \b | pip3? | uv | torch | numpy | qiskit | twisterl | rustworkx | safetensors | cargo | rustc
# (-w makes them word-boundary matches; `_` is a word character, so pipefail, pipeline and the symbol
# safetensors_import do not match)
WORDS='python3?|pip3?|uv|torch|numpy|qiskit|twisterl|rustworkx|safetensors|cargo|rustc'

fail=0
TMP=$(mktemp -d "${TMPDIR:-/tmp}/check_pure_mojo.XXXXXX") || exit 2
trap 'rm -rf "$TMP"' EXIT INT TERM

report() { # report PASS|FAIL "text"
    if [ "$1" = FAIL ]; then fail=1; fi
    echo "check-pure-mojo: $1 $2"
}

# ------------------------------------------------------------------------------------------------ 1 source scan
# explicit exclusions: this script, Markdown files, NOTICE, PROVENANCE.md, bytecode caches
find training qumo/rl -type f ! -name '*.md' ! -name NOTICE ! -name PROVENANCE.md ! -name check_pure_mojo.sh \
    ! -name '*.pyc' ! -name .DS_Store ! -path '*/__pycache__/*' > "$TMP/files.txt"
nfiles=$(wc -l < "$TMP/files.txt" | tr -d ' ')
: > "$TMP/hits.txt"
while IFS= read -r f; do
    grep -I -n -i -w -E "$WORDS" "$f" 2>/dev/null | sed "s|^|$f:|" >> "$TMP/hits.txt"
done < "$TMP/files.txt"
if [ -s "$TMP/hits.txt" ]; then
    report FAIL "source scan: forbidden words in training sources"
    head -n 40 "$TMP/hits.txt"
else
    report PASS "source scan: $nfiles files under training/ and qumo/rl have none of the forbidden words"
fi

# the expanded recipes of the training targets (no build output is produced by -n)
MAKE_ARGS="train-ai train-ai-smoke train-ai-route-ppo models-verify"
make -n INIT=models/ai_transpiler/route/route_mlp_h64_v1 BUDGET_S=60 $MAKE_ARGS > "$TMP/recipes.txt" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    report FAIL "recipes: make -n $MAKE_ARGS exited with $rc"
    head -n 20 "$TMP/recipes.txt"
elif grep -n -i -w -E "$WORDS" "$TMP/recipes.txt" > "$TMP/recipe_hits.txt"; then
    report FAIL "source scan: the expanded recipes name a forbidden word"
    head -n 20 "$TMP/recipe_hits.txt"
else
    report PASS "source scan: the expanded recipes of $MAKE_ARGS ($(wc -l < "$TMP/recipes.txt" | tr -d ' ') lines) have none"
fi

# ------------------------------------------------------------------------------------------------ 2 environment scan
# A PATH directory of symlinks to the tools a build needs, plus recording shims for every forbidden command.
BIN=$TMP/bin
mkdir -p "$BIN"
for t in make sh bash env mkdir rm cp mv cat date seq sleep tail head sed awk grep tr cut sort wc test '[' find ls \
    basename dirname uname cc clang ld ar ranlib strip xcrun readlink ln touch dd tee printf true false expr uniq \
    mktemp chmod stat nproc sysctl diff cmp wait; do
    p=$(command -v "$t" 2>/dev/null) || continue
    case "$p" in /*) ln -sf "$p" "$BIN/$t" ;; esac
done
SHIMLOG=$TMP/shim.log
: > "$SHIMLOG"
for t in python python3 python3.12 pip pip3 uv uvx cargo rustc rustup; do
    printf '#!/bin/sh\necho "%s $*" >> "%s"\nexit 127\n' "$t" "$SHIMLOG" > "$BIN/$t"
    chmod +x "$BIN/$t"
done

if env -i HOME="$HOME" PATH="$BIN" make -n INIT=models/ai_transpiler/route/route_mlp_h64_v1 BUDGET_S=60 $MAKE_ARGS > "$TMP/envdry.txt" 2>&1; then
    report PASS "environment: make -n $MAKE_ARGS completes under env -i with no Python, uv, pip, cargo or rustc on PATH"
else
    report FAIL "environment: make -n failed under env -i"
    head -n 20 "$TMP/envdry.txt"
fi

if [ "$SMOKE" = 1 ]; then
    echo "check-pure-mojo: running make train-ai-smoke under env -i (about 5 minutes) ..."
    if env -i HOME="$HOME" PATH="$BIN" make AI_BUILD="$AI_BUILD" train-ai-smoke > "$TMP/smoke.txt" 2>&1; then
        report PASS "environment: make train-ai-smoke ran to completion in a clean environment"
        tail -n 3 "$TMP/smoke.txt"
    else
        report FAIL "environment: make train-ai-smoke failed in a clean environment"
        tail -n 30 "$TMP/smoke.txt"
    fi
else
    echo "check-pure-mojo: (make train-ai-smoke is not run; pass --smoke or CHECK_PURE_SMOKE=1 for the full scan)"
fi
if [ -s "$SHIMLOG" ]; then
    report FAIL "environment: a recording shim was called"
    head -n 10 "$SHIMLOG"
else
    report PASS "environment: no python, uv, pip, cargo or rustc call was made"
fi

# ------------------------------------------------------------------------------------------------ 3 binary scan
nbin=0
bad=0
for b in "$AI_BUILD"/bin/*; do
    [ -f "$b" ] && [ -x "$b" ] || continue
    nbin=$((nbin + 1))
    if command -v otool > /dev/null 2>&1; then
        libs=$(otool -L "$b" 2>/dev/null)
    elif command -v ldd > /dev/null 2>&1; then
        libs=$(ldd "$b" 2>/dev/null)
    else
        libs=""
    fi
    if echo "$libs" | grep -i -E 'libpython|libtorch' > /dev/null; then
        echo "check-pure-mojo: $b links a Python or torch library"
        bad=$((bad + 1))
    fi
done
if [ "$bad" -gt 0 ]; then
    report FAIL "binary scan: $bad of $nbin binaries link libpython or libtorch"
elif [ "$nbin" -eq 0 ]; then
    echo "check-pure-mojo: binary scan skipped (no binaries in $AI_BUILD/bin; run make ai-tools first)"
else
    report PASS "binary scan: $nbin binaries in $AI_BUILD/bin link no libpython and no libtorch"
fi

# ------------------------------------------------------------------------------------------------ 4 tree
pycount=$(find training/ai_transpiler -name '*.py' | wc -l | tr -d ' ')
if [ "$pycount" -gt 0 ] || [ -e training/ai_transpiler/requirements.txt ]; then
    report FAIL "tree: $pycount .py files or a requirements.txt under training/ai_transpiler"
    find training/ai_transpiler -name '*.py' -o -name requirements.txt | head -n 20
else
    report PASS "tree: no .py under training/ai_transpiler and no requirements.txt"
fi

if [ "$fail" -eq 0 ]; then
    echo "check-pure-mojo: OK"
fi
exit "$fail"
