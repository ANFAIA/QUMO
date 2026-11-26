#!/bin/sh
# make check-neutrality (part of `make test` and of the CI checks job): nothing under qumo/ imports a vendor module.
#
#   tests/check_neutrality.sh
#
# The vendor clients are the Mojo package plugins/hardware/native/qumo_hw (docs/ARCHITECTURE.md, "Vendor code").
# 1. import scan: no .mojo file under qumo/ or cli/ imports `qumo_hw` (at any indentation, so function-local
#    imports count). Names inside strings and comments are not imports and are not matched.
# 2. file scan: no .mojo file under qumo/ is named after a vendor client (<vendor>*.mojo for every
#    <vendor>_api.mojo of qumo_hw), so a client cannot come back into the core under its old name.
#
# Exit status 0 only when both parts passed.

set -u
cd "$(dirname "$0")/.." || exit 2
HW=plugins/hardware/native/qumo_hw
fail=0

hits=$(grep -rn -E '^[[:space:]]*(from[[:space:]]+qumo_hw([.[:space:]]|$)|import[[:space:]]+qumo_hw([.[:space:],]|$))' \
    --include='*.mojo' qumo cli 2>/dev/null)
if [ -n "$hits" ]; then
    echo "check-neutrality: FAIL core files import a vendor module (only python/, examples/ and plugins/ may):"
    echo "$hits" | head -n 40
    fail=1
else
    n=$(find qumo cli -name '*.mojo' | wc -l | tr -d ' ')
    echo "check-neutrality: PASS import scan: $n .mojo files under qumo/ and cli/ import nothing from qumo_hw"
fi

vendors=$(ls "$HW"/*_api.mojo 2>/dev/null | sed -e 's|.*/||' -e 's|_api\.mojo$||')
if [ -z "$vendors" ]; then
    echo "check-neutrality: FAIL no vendor client found in $HW (moved? update tests/check_neutrality.sh)"
    fail=1
else
    bad=""
    for v in $vendors; do
        bad="$bad$(find qumo -name "${v}*.mojo")"
    done
    if [ -n "$bad" ]; then
        echo "check-neutrality: FAIL vendor-named files under qumo/:"
        echo "$bad"
        fail=1
    else
        echo "check-neutrality: PASS file scan: no file under qumo/ is named after" $vendors
    fi
fi

[ "$fail" -eq 0 ] && echo "check-neutrality: OK"
exit "$fail"
