"""Routing job generator (circuit families), evaluation sets and the JSON glue shared with route_tool.

STUB (W-1): signatures follow MOJO_PORT.md 16 (C9); section 5 does not list them, so they are
PROVISIONAL until C9 lands. The four helpers at the end currently live in
`training/ai_transpiler/route/route_tool.mojo` and are moved here by C9. Bodies raise or abort with a
"STUB" message.
"""

from std.os import abort
from qumo.ai.json import JsonValue
from qumo.ir.circuit import Circuit
from qumo.compiler.target import Target
from qumo.rl.targets import RTopo


struct JobSet(Movable):
    """A generated set of routing jobs, serialisable to the jobs JSON consumed by `route_tool`."""

    var json: String

    def __init__(out self):
        self.json = String()

    def to_json(self) -> String:
        return self.json.copy()


def build_jobs(
    seed: Int,
    n_jobs: Int,
    topologies: List[RTopo],
    families: List[String],
    max_n: Int,
    scale: Float64,
    min_frac: Float64,
) raises -> JobSet:
    """Families: rand2q, qv, qft, trotter, qaoa, sparse (parameters of the old `circuits.py`)."""
    raise Error("QM-E640: qumo.rl.jobs.build_jobs is a STUB (component C9)")


def eval_sets(out_dir: String) raises:
    """Writes the evaluation sets (synth seeds >= 910000, bench circuits 3 <= n <= 14, seed 7)."""
    raise Error("QM-E640: qumo.rl.jobs.eval_sets is a STUB (component C9)")


def summarize(path: String) raises -> String:
    """Summary text of an EVAL.jsonl file."""
    raise Error("QM-E640: qumo.rl.jobs.summarize is a STUB (component C9)")


# --- helpers moved out of route_tool.mojo by C9 (same signatures as today) ---
def circuit_from_json(cj: JsonValue) raises -> Circuit:
    raise Error("QM-E640: qumo.rl.jobs.circuit_from_json is a STUB (component C9)")


def target_from_json(tj: JsonValue) raises -> Target:
    raise Error("QM-E640: qumo.rl.jobs.target_from_json is a STUB (component C9)")


def put_i16(mut b: List[UInt8], v: Int):
    abort("qumo.rl.jobs.put_i16: STUB (component C9)")


def put_f32(mut b: List[UInt8], v: Float32):
    abort("qumo.rl.jobs.put_f32: STUB (component C9)")
