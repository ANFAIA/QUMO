"""Stim text in, correlated matching, sliding windows, and the detector lint.

Reproduces: Stim 1.16 and PyMatching 2.3 - missing_detectors, correlated matching (arXiv:1310.0863)
Source: https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md (Apache-2.0)

Any Stim circuit runs on QUMO's stack: `from_stim` unrolls REPEAT and lowers MPP,
MX and MR to gates. `mwpm_correlated` decodes twice and lowers the weights of edges
that fire together with the matched ones; PyMatching's paired counts agree
(bench/crosscheck/tier1.py correlated), including that it loses at d = 3. A sliding
window commits only the older rounds of each window.

Checks:
- a repetition-code memory written in Stim text has the expected detector count;
- MPP X0*X1 and Z0*Z1 are deterministic on a Bell pair;
- deleting a detector is reported;
- correlated matching beats plain matching at d = 7;
- a window of 7 rounds with step 2 is close to the full decoder.

Run: .venv/bin/mojo run -I . examples/frameworks/cirq/stim_interop_decoding.mojo
example-meta: {"framework": "cirq", "tier": "fast", "timeout": 240, "requires": [], "args": []}
"""

from qumo.qec.codes import NoiseParams, generate_code
from qumo.qec.dem import detector_error_model
from qumo.qec.decoders import MatchingGraph, count_logical_errors
from qumo.qec.sparse_blossom import SparseBlossomDecoder
from qumo.qec.correlated import CorrelatedMatchingDecoder
from qumo.qec.stim_text import from_stim
from qumo.qec.lint import missing_detectors
from qumo.qec.window import plan_windows, detector_rounds, SlidingWindowDecoder
from qumo.sim.frame import FrameSimulator
from qumo.testing.checks import CheckLog


def main() raises:
    var log = CheckLog("stim_interop_decoding")
    var text = String("""R 0 1 2 3 4
CX 0 1 2 3
CX 2 1 4 3
MR 1 3
DETECTOR rec[-2]
DETECTOR rec[-1]
REPEAT 4 {
    DEPOLARIZE1(0.001) 0 2 4
    CX 0 1 2 3
    CX 2 1 4 3
    MR 1 3
    DETECTOR rec[-1] rec[-3]
    DETECTOR rec[-2] rec[-4]
}
M 0 2 4
DETECTOR rec[-1] rec[-2] rec[-4]
DETECTOR rec[-2] rec[-3] rec[-5]
OBSERVABLE_INCLUDE(0) rec[-1]
""")
    var rep = from_stim(text).circuit.copy()
    print(rep.num_qubits, "qubits,", rep.num_measurements, "measurements,", rep.num_detectors, "detectors")
    _ = log.close("REPEAT unrolled", Float64(rep.num_detectors), 12.0, 0.0)
    var bell = from_stim("H 0\nCX 0 1\nMPP X0*X1 Z0*Z1\nDETECTOR rec[-2]\nDETECTOR rec[-1]").circuit.copy()
    var s = FrameSimulator(bell).sample_detectors(2048, 1)
    var fired = 0
    for shot in range(2048):
        fired += len(s.fired(shot))
    _ = log.close("MPP on a Bell pair is deterministic", Float64(fired), 0.0, 0.0)

    var cut = from_stim("R 0\nM 0\nM 0\nDETECTOR rec[-1] rec[-2]\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]").circuit.copy()
    _ = log.close("undeclared parities are found", Float64(len(missing_detectors(cut))), 1.0, 0.0)

    var cc = generate_code("surface_z", 7, 7, NoiseParams.uniform(0.005))
    var dem = detector_error_model(cc.circuit)
    var samples = FrameSimulator(cc.circuit).sample_detectors(100000, 3)
    var plain = count_logical_errors(SparseBlossomDecoder(MatchingGraph(dem)), samples)
    var corr = count_logical_errors(CorrelatedMatchingDecoder(dem), samples)
    print("d=7 p=0.005: plain", plain, "correlated", corr)
    _ = log.check("correlated matching gains at d = 7", Float64(corr) < 0.75 * Float64(plain))

    var memory = generate_code("surface_z", 5, 16, NoiseParams.uniform(0.004))
    var dm = detector_error_model(memory.circuit)
    dm.detector_coords = memory.detector_coords.copy()
    var plan = plan_windows(dm, detector_rounds(dm), 7, 2)
    var inner = List[SparseBlossomDecoder]()
    for sub in plan.dems:
        inner.append(SparseBlossomDecoder(MatchingGraph(sub)))
    var window = SlidingWindowDecoder[SparseBlossomDecoder](plan^, inner^)
    var ms = FrameSimulator(memory.circuit).sample_detectors(20000, 5)
    var full = count_logical_errors(SparseBlossomDecoder(MatchingGraph(dm)), ms)
    var wn = count_logical_errors(window, ms)
    print("16 rounds: full", full, "window 7 / step 2", wn)
    _ = log.check("the window stays close to the full decoder", Float64(wn) < 1.3 * Float64(full) + 15)
    log.finish()
