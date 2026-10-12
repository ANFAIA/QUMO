"""Surface-code memory experiment end to end (Stim + PyMatching + Sinter, in Mojo).

    .venv/bin/mojo run -I . examples/qec_memory_experiment.mojo
"""

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
    print(dem, g, "circuit distance =", g.graphlike_distance())
    var lines = dem.to_text().split("\n")
    for i in range(4):
        print("   ", lines[i])

    # 3. Raw bit-packed sampling throughput.
    var fs = FrameSimulator(cc.circuit)
    var ds = fs.sample_detectors(1_000_000, seed=1)
    print(ds, "in", ds.seconds, "s ->", fs.throughput(ds) / 1e9, "G shot*ops/s")

    # 4. Logical error rates (Sinter-like: stop at 200 errors).
    for d in range(3, 8, 2):
        print(memory_experiment("surface_z", d, -1, 0.002, 2_000_000, "mwpm", 7, 200))

    # 5. Threshold sweep with the Union-Find decoder (CSV for plotting / agents).
    var ds_: List[Int] = [3, 5, 7]
    var ps: List[Float64] = [0.003, 0.006, 0.01]
    print(threshold_sweep("surface_z", ds_, ps, 200_000, "uf", 3, 300))
