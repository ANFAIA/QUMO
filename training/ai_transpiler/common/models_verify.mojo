"""Verify every model file under a model root against its INDEX.json (MOJO_PORT.md 9.4).

Usage: `build/ai_train/bin/models_verify` (no arguments). The root is `$QUMO_MODEL_DIR`, else `models/ai_transpiler`.
Prints the `FAIL ...` lines and then `models-verify: N INDEX.json, M files ok, K problems, S MB total`; the exit
code is 1 when there is any problem. A missing root prints a note and exits 0. All five INDEX layouts of the old
make target are understood; the verification itself is `qumo.rl.container.verify_models`.
"""

from std.os import getenv
from std.sys import exit
from qumo.rl.container import verify_models


def main():
    var root = getenv("QUMO_MODEL_DIR")
    if root.byte_length() == 0:
        root = String("models/ai_transpiler")
    var rep = verify_models(root)
    for line in rep.lines:
        print(line)
    if rep.n_problems > 0:
        exit(1)
