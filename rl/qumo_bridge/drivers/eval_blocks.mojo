# Evaluate one exported synthesis model against QUMO's own synthesizers on seeded random blocks of the model's
# coupling graph (thin main over qumo_bridge/eval_blocks.mojo).
#
#   .venv/bin/mojo run -I . -I rl rl/qumo_bridge/drivers/eval_blocks.mojo \
#       <model_root> <model_id> <num_instances> <out.json>
#
# <model_root> is a directory named `synth` with INDEX.json + <id>/ (e.g. models/ai_transpiler/synth).
# Proposers: policy (beams=1 greedy ensemble; beams=8 + 4 samples), block_synth greedy / auto / exact (exact only
# where `exact_supported`). Every output is verified exactly (Perm / GF2Mat / signed SympTab equality; Pauli
# networks by dense statevector comparison). Costs: 2q cost in CX units (CX = CZ = 1, SWAP = 3) and 2q depth.
# Writes the JSON summary to <out.json> and prints it.
from std.sys import argv
from qumo_bridge.eval_blocks import eval_blocks_json
from qrl.container import write_text_atomic


def _usage() -> String:
    return "usage: eval_blocks MODEL_ROOT MODEL_ID NUM_INSTANCES OUT.json"


def _parse_count(text: String) raises -> Int:
    try:
        return Int(text)
    except:
        raise Error("QM-E640: NUM_INSTANCES must be an integer, got '" + text + "'. Hint: " + _usage())


def main() raises:
    var raw = argv()
    if len(raw) != 5:
        raise Error("QM-E640: expected 4 arguments, got " + String(len(raw) - 1) + ". Hint: " + _usage())
    var root = String(raw[1])
    var id = String(raw[2])
    var count_text = String(raw[3])
    var outp = String(raw[4])
    var count = _parse_count(count_text)
    var js = eval_blocks_json(root, id, count)
    write_text_atomic(outp, js)
    print(js)
