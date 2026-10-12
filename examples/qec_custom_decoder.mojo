"""Plug your own decoder (e.g. a neural / AlphaQubit-style model) into QUMO.

Anything implementing the `Decoder` trait - `decode(fired) -> UInt64` -
runs inside the parallel, bit-packed sample+decode loop.

    .venv/bin/mojo run -I . examples/qec_custom_decoder.mojo
"""

from qumo.qec.codes import repetition_code_memory, NoiseParams
from qumo.qec.dem import detector_error_model
from qumo.qec.decoders import Decoder, MatchingGraph, MwpmDecoder
from qumo.qec.experiments import sample_and_decode
from qumo.sim.frame import FrameProgram


struct FinalRoundMajority(Decoder):
    """Toy 'learned' decoder for the repetition code: predicts a logical flip
    when the final-round detectors suggest the last data qubit is wrong.
    Replace the body with a call into your model (ONNX/MAX graph, lookup of
    a trained network, ...)."""

    var num_detectors: Int
    var last: Int  # id of the final-round detector next to the observable qubit

    def __init__(out self, num_detectors: Int, last: Int):
        self.num_detectors = num_detectors
        self.last = last

    def name(self) -> String:
        return "final_round_majority"

    def decode(mut self, fired: List[Int]) -> UInt64:
        for d in fired:
            if d == self.last:
                return 1
        return 0


def main() raises:
    var d = 5
    var cc = repetition_code_memory(d, d, NoiseParams.uniform(0.02))
    var dem = detector_error_model(cc.circuit)
    var prog = FrameProgram(cc.circuit)
    var toy = FinalRoundMajority(dem.num_detectors, dem.num_detectors - 1)
    var a = sample_and_decode(prog, toy, 500_000, 1)
    var b = sample_and_decode(prog, MwpmDecoder(MatchingGraph(dem)), 500_000, 1)
    print("toy decoder (illustration):", a, " p_L =", Float64(a.errors) / Float64(a.shots))
    print("MWPM                      :", b, " p_L =", Float64(b.errors) / Float64(b.shots))
