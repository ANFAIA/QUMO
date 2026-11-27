"""Surface-code memory experiment end to end (Stim + PyMatching + Sinter, in Mojo).

A quantum memory stores one logical qubit in many physical qubits, and keeps checking for
errors. This example runs the whole experiment that papers report, in five steps:

  1. build the circuit: a distance-5 surface code with circuit-level noise (p = 0.1%)
  2. derive its detector error model, the map from errors to the checks they trigger
  3. sample a million shots of those checks, to see how fast the simulator is
  4. decode them and count how often the logical qubit is wrong: the logical error rate
  5. repeat for several code distances and noise levels (a threshold sweep, as a CSV)

What to look for: in step 4, at 0.2% noise, a larger distance gives a LOWER logical error rate
(d = 3, 5, 7: roughly 3e-3, 1e-3, 3e-4). That is the point of error correction. In the sweep of
step 5 the effect reverses between 0.6% and 1% noise: at 1% the larger codes are worse. That
crossover is the threshold.

    .venv/bin/mojo run -I . examples/error_correction/qec_memory_experiment.mojo

Framework: QUMO - https://github.com/ANFAIA/QUMO
"""

# Error correction is one explicit import away: it lives in qumo.qec, not in the root.
from qumo.qec.codes import surface_code_memory, NoiseParams
from qumo.qec.dem import detector_error_model
from qumo.qec.decoders import MatchingGraph
from qumo.qec.experiments import memory_experiment, threshold_sweep
from qumo.sim.frame import FrameSimulator


def main() raises:
    # 1. Circuit with Stim-style circuit-level noise.
    var cc = surface_code_memory(5, 5, NoiseParams.uniform(0.001), "Z")
    print(cc)
    var stim_text = cc.to_stim()
    print("Stim export:", stim_text.byte_length(), "bytes (paste into stim.Circuit / Crumble)")

    # 2. Detector error model (identical to Stim's) and its matching graph.
    var dem = detector_error_model(cc.circuit)
    var g = MatchingGraph(dem)
    print(dem, g, "graph-like distance =", g.graphlike_distance())
    var lines = dem.to_text().split("\n")
    for i in range(4):
        print("   ", lines[i])

    # 3. Raw bit-packed sampling throughput.
    var fs = FrameSimulator(cc.circuit)
    var ds = fs.sample_detectors(1_000_000, seed=1)
    print(ds, "in", ds.seconds, "s ->", fs.throughput(ds) / 1e9, "G shot*ops/s")

    # 4. Logical error rates (Sinter-like: stop at 200 errors). 'mwpm' is the
    #    exact sparse-blossom matcher (PyMatching 2's algorithm); 'mwpm_dense'
    #    the older APSP + dense blossom one (same answers, slower at large d / p).
    #    Arguments: code, distance, rounds (-1 = distance), noise p, max shots, decoder, seed, max errors.
    for d in range(3, 8, 2):
        print(memory_experiment("surface_z", d, -1, 0.002, 2_000_000, "mwpm", 7, 200))
    print(memory_experiment("surface_z", 9, -1, 0.005, 2_000_000, "mwpm", 7, 200))

    # 5. Threshold sweep (CSV for plotting / agents) with the exact matcher;
    #    pass "uf" for the union-find decoder (slightly worse p_L, and in
    #    QUMO currently slower than sparse blossom).
    var ds_: List[Int] = [3, 5, 7]
    var ps: List[Float64] = [0.003, 0.006, 0.01]
    print(threshold_sweep("surface_z", ds_, ps, 200_000, "mwpm", 3, 300))
