"""Model container writer (qumo.rlsynth.v0 / qumo.rlroute.v0), atomic directory swap, INDEX writer, verifier.

STUB (W-1): the signatures of MOJO_PORT.md section 5, plus `VerifyReport`, `dump_indent1` and
`index_row_from_dir` named in 9.4 / 9.5 / 16 (those three are PROVISIONAL until C2 lands).
Component C2 replaces this file; every behavioural body raises or aborts with a "STUB" message.
"""

from std.os import abort
from qumo.ai.json import JsonValue


@fieldwise_init
struct TensorOut(Copyable):
    var name: String
    var shape: List[Int]
    var data: List[Float32]
    var dtype: String  # "f16" | "f32"


@fieldwise_init
struct ModelDirInfo(Copyable):
    var weights_bytes: Int
    var fnv1a64: String
    var sha256: String
    var manifest_bytes: Int


struct VerifyReport(Copyable):
    """Outcome of `verify_models` (PROVISIONAL shape): the old make target's counters plus its text lines."""

    var n_index: Int
    var n_files: Int
    var n_problems: Int
    var total_bytes: Int
    var lines: List[String]  # "FAIL ..." lines and the final summary line

    def __init__(out self):
        self.n_index = 0
        self.n_files = 0
        self.n_problems = 0
        self.total_bytes = 0
        self.lines = List[String]()


def write_model_dir(
    dir: String,
    manifest_head: String,
    tensors: List[TensorOut],
    dtype: String,
    f32_names: List[String],
    manifest_tail: String,
) raises -> ModelDirInfo:
    """manifest_head: the manifest's JSON members that precede "tensors" (text, no trailing comma); the writer
    emits the "tensors" array itself; manifest_tail: the members that follow, without the closing brace
    ("" if none). Atomic: writes <dir>.tmp, reads it back, then swaps it into <dir> (9.1)."""
    raise Error("QM-E640: qumo.rl.container.write_model_dir is a STUB (component C2)")


def atomic_replace_dir(tmp: String, final: String) raises:
    """libc rename swap, 9.1."""
    raise Error("QM-E640: qumo.rl.container.atomic_replace_dir is a STUB (component C2)")


def recover_dir(final: String) raises:
    """Restores <final>.old if <final> is missing, drops a stale .tmp."""
    raise Error("QM-E640: qumo.rl.container.recover_dir is a STUB (component C2)")


def rebuild_index(root: String, sub: String, default_id: String) raises:
    """synth: default_id == ""."""
    raise Error("QM-E640: qumo.rl.container.rebuild_index is a STUB (component C2)")


def verify_models(root: String) -> VerifyReport:
    """Same text output as the old `make models-verify`."""
    abort("qumo.rl.container.verify_models: STUB (component C2)")


# --- PROVISIONAL helpers named in 9.5 / 16 (C2) ---
def dump_indent1(v: JsonValue) raises -> String:
    """Python `json.dump(indent=1)` formatting (every array element on its own line)."""
    raise Error("QM-E640: qumo.rl.container.dump_indent1 is a STUB (component C2)")


def index_row_from_dir(dir: String) raises -> String:
    """JSON text of the INDEX row (`id, kind, num_qubits, edges, wl_hash, dir, sha256, bytes`) of a model dir."""
    raise Error("QM-E640: qumo.rl.container.index_row_from_dir is a STUB (component C2)")
