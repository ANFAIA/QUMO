#!/bin/sh
# make check-independence (rl/Makefile; part of `make test` here and of `make test-rl` in QUMO): qrl stands alone.
#
#   rl/check_independence.sh [MOJO]
#
# 1. import scan: no file under qrl/, drivers/ or tests/ (the core package, its tools and its tests) imports or names
#    a `qumo.` module (`from qumo.`, `import qumo`, `qumo_bridge`); only qumo_bridge/ may. Format strings such as
#    "qumo.rlsynth.v0" are data, not imports, and are not matched.
# 2. compile check (when MOJO is given and runs): every core tool main and the core package compile with `-I .`
#    from a scratch copy of qrl/ and drivers/ alone, so no QUMO package is on the import path.
#
# Exit status 0 only when both parts passed.

set -u
cd "$(dirname "$0")" || exit 2
MOJO=${1:-}
case "$MOJO" in
    */*) MOJO=$(cd "$(dirname "$MOJO")" && pwd)/$(basename "$MOJO") ;;
esac
fail=0

# ------------------------------------------------------------------------------------------------ 1 import scan
hits=$(grep -rn -E '^[[:space:]]*(from[[:space:]]+qumo(_bridge)?([.[:space:]]|$)|import[[:space:]]+qumo(_bridge)?([.[:space:],]|$))' \
    --include='*.mojo' qrl drivers tests 2>/dev/null)
if [ -n "$hits" ]; then
    echo "check-independence: FAIL core files import QUMO (only qumo_bridge/ may):"
    echo "$hits" | head -n 40
    fail=1
else
    n=$(find qrl drivers tests -name '*.mojo' | wc -l | tr -d ' ')
    echo "check-independence: PASS import scan: $n .mojo files under qrl/, drivers/, tests/ import nothing from qumo"
fi

# ------------------------------------------------------------------------------------------------ 2 compile check
if [ -n "$MOJO" ] && "$MOJO" --version > /dev/null 2>&1; then
    TMP=$(mktemp -d "${TMPDIR:-/tmp}/qrl_indep.XXXXXX") || exit 2
    trap 'rm -rf "$TMP"' EXIT INT TERM
    cp -R qrl drivers "$TMP"/
    bad=0
    # the builds run in parallel; each leaves its log and, on success, an .ok marker
    for m in "$TMP"/drivers/*/*.mojo; do
        b=$(basename "$m" .mojo)
        (cd "$TMP" && "$MOJO" build -I . -o "$TMP/bin_$b" "drivers/${m#"$TMP"/drivers/}" > "$TMP/$b.log" 2>&1 \
            && touch "$TMP/$b.ok") &
    done
    wait
    for m in "$TMP"/drivers/*/*.mojo; do
        b=$(basename "$m" .mojo)
        if [ ! -f "$TMP/$b.ok" ]; then
            echo "check-independence: FAIL $b does not build from qrl/ alone:"
            grep -v Crashpad "$TMP/$b.log" | grep error | head -n 10
            bad=1
        fi
    done
    if [ "$bad" -ne 0 ]; then
        fail=1
    else
        echo "check-independence: PASS every core tool builds with -I . from a copy of qrl/ and drivers/ alone"
    fi
else
    echo "check-independence: (compile check skipped: no working mojo given)"
fi

[ "$fail" -eq 0 ] && echo "check-independence: OK"
exit "$fail"
