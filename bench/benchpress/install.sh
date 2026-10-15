#!/usr/bin/env bash
# Install the QUMO gym into a Benchpress checkout.
#   bench/benchpress/install.sh /path/to/benchpress
set -euo pipefail
BP=${1:?usage: install.sh <benchpress_checkout>}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
rm -rf "$BP/benchpress/qumo_gym"
cp -R "$HERE/qumo_gym" "$BP/benchpress/qumo_gym"
ln -sfn "$ROOT/build" "$BP/benchpress/qumo_gym/_build"
grep -q '^\[qumo\]' "$BP/default.conf" || printf '\n[qumo]\noptimization_level = 2\n# ai = true runs the hybrid heuristic + AI preset (transpile_ai, ai_level 1..3; default: optimization_level)\nai = false\n' >> "$BP/default.conf"
echo "installed qumo_gym into $BP (build dir: $ROOT/build)"
