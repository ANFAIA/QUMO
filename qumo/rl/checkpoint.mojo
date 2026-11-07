"""Run-directory layout, f32 checkpoints and the read-only safetensors importer.

STUB (W-1): MOJO_PORT.md section 5 does not list this module; the signatures below are PROVISIONAL
(derived from 7.5 and 16 / C11a) and the owner (C11a) finalises them through the lead. Bodies raise
with a "STUB" message.
"""

from qumo.compiler.policy import ModelFile
from qumo.rl.nn import BasicNet
from qumo.rl.container import ModelDirInfo


@fieldwise_init
struct SafeTensor(Copyable):
    """One tensor read from a .safetensors file (f32 data)."""

    var name: String
    var shape: List[Int]
    var data: List[Float32]


def write_checkpoint(
    dir: String, net: BasicNet, manifest_head: String, manifest_tail: String
) raises -> ModelDirInfo:
    """f32 model-container checkpoint written atomically (through container.write_model_dir)."""
    raise Error("QM-E640: qumo.rl.checkpoint.write_checkpoint is a STUB (component C11a)")


def read_checkpoint(dir: String) raises -> ModelFile:
    """Load a checkpoint or any model container; callers build the network with `BasicNet.from_model`."""
    raise Error("QM-E640: qumo.rl.checkpoint.read_checkpoint is a STUB (component C11a)")


def safetensors_import(path: String) raises -> List[SafeTensor]:
    """Read-only importer for old Python runs (f32, f16, bf16 decoded to f32)."""
    raise Error("QM-E640: qumo.rl.checkpoint.safetensors_import is a STUB (component C11a)")
