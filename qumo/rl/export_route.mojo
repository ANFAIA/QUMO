"""Router export: a run directory to a qumo.rlroute.v0 model, test vectors, parity episodes, INDEX default handling.

STUB (W-1): MOJO_PORT.md section 5 does not list this module; the signature below is PROVISIONAL
(derived from 9.3 and 10.1, C10). Bodies raise with a "STUB" message.
"""

from qumo.rl.container import ModelDirInfo


def export_route_run(
    run_dir: String, root: String, id: String, make_default: Bool, episodes: Int, meta_path: String
) raises -> ModelDirInfo:
    """Writes <root>/<id>/; `make_default` False preserves the previous INDEX default (defect fix 7.4)."""
    raise Error("QM-E640: qumo.rl.export_route.export_route_run is a STUB (component C10)")
