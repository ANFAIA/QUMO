#!/bin/bash
# Image size and per-run cost of "optimize a Bell circuit for a device, then simulate it" for each SDK image,
# run the way a production container would be: non-root user, no network, no capabilities, no privilege escalation.
# (--read-only is left out: matplotlib, a Cirq dependency, needs its prebuilt font cache directory writable.)
# Run from the repo root after building the images (see the Dockerfiles). N runs per image, first one discarded
# (cold page cache). Output: one line per run, "sdk image_bytes METRIC ...".
cd "$(dirname "$0")/../.."
N=${N:-10}
OUT=${OUT:-/dev/stdout}
: > "$OUT"
RUN=(docker run --rm --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges --network none
     -v "$PWD/docker/sdk-bell/measure:/measure:ro")
for sdk in qumo qiskit tket cirq; do
  img=sdk-bell-$sdk
  size=$(docker image inspect -f '{{.Size}}' $img)
  if [ $sdk = qumo ]; then
    cmd=(qumo transpile bell.qasm --target line:2/rz,sx,x,cx --level 2 -o /tmp/c.qasm --then-- qumo run /tmp/c.qasm --shots 1000 --seed 7)
  else
    cmd=(python task.py)
  fi
  for i in $(seq 0 $N); do
    line=$("${RUN[@]}" $img /measure "${cmd[@]}" 2>&1 >/dev/null | grep METRIC)
    [ "$i" -gt 0 ] && echo "$sdk $size $line" >> "$OUT"
  done
done
