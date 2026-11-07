"""Synthesis export: a run directory to a qumo.rlsynth.v0 model (f16, test vectors, rollouts, INDEX).

STUB (W-1): MOJO_PORT.md section 5 does not list this module; the signatures below are PROVISIONAL
(derived from 9.2, 4.6 and 16 / C11a) and the owner (C11a) finalises them through the lead. Bodies raise
with a "STUB" message.
"""

from qumo.compiler.policy import ModelFile
from qumo.rl.container import ModelDirInfo


def check_pauli_ensemble(m: ModelFile, seed: UInt64) raises:
    """Behavioural self-check of the exported Pauli symmetries against TrainEnv's hidden-permutation view
    (4.6). QM-E644 on mismatch."""
    raise Error("QM-E640: qumo.rl.export_synth.check_pauli_ensemble is a STUB (component C11a)")


def export_synth_run(
    run_dir: String, which: String, id: String, root: String, dtype: String, make_default: Bool
) raises -> ModelDirInfo:
    """`which` is "best" or "last"; writes <root>/<id>/ and updates <root>/INDEX.json."""
    raise Error("QM-E640: qumo.rl.export_synth.export_synth_run is a STUB (component C11a)")
