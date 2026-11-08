"""Generate the tiny deterministic fixture models of tests/fixtures/ai/policy with the native RNG.

usage: make_fixtures [--out DIR]        (default DIR: build/fixtures_ai_policy)

Writes DIR/synth/<id>/{manifest.json, weights.bin} for the five fixture models (random normal weights from the Mojo
`Rng`, emb 16, common [16]), DIR/synth/INDEX.json and the corrupted copies DIR/bad_checksum and DIR/bad_format. The
committed fixtures under tests/fixtures/ai/policy were produced by the legacy generator and stay frozen; the default
output directory is a scratch directory so that running this tool never overwrites them. Pass
`--out tests/fixtures/ai/policy` only to deliberately replace them (their embedded vectors would then come from this
generator's random stream).
"""

from std.sys import argv
from qrl.export_synth import generate_policy_fixtures


def main() raises:
    var raw = argv()
    var out = String("build/fixtures_ai_policy")
    var i = 1
    while i < len(raw):
        var a = String(raw[i])
        if a == "--out":
            if i + 1 >= len(raw):
                raise Error("QM-E640: option --out needs a value. Hint: make_fixtures [--out DIR]")
            out = String(raw[i + 1])
            i += 2
        else:
            raise Error("QM-E640: unknown argument '" + a + "'. Hint: make_fixtures [--out DIR]")
    var ids = generate_policy_fixtures(out)
    var s = String("make_fixtures: wrote")
    for k in range(len(ids)):
        s += " " + ids[k]
    print(s + " to " + out)
