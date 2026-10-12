"""QEC benchmark: frame-sampling throughput, DEM extraction, UF / MWPM decoding.

    mojo run -I . bench/bench_qec.mojo                 # default grid, 200k shots
    mojo run -I . bench/bench_qec.mojo --run D P SHOTS # one point, CSV line
    mojo run -I . bench/bench_qec.mojo --stim D P      # print the Stim circuit
    mojo run -I . bench/bench_qec.mojo --dem D P       # print the DEM (Stim syntax)

`bench/bench_qec.py` drives this binary and cross-validates every number
against Stim + PyMatching on the *identical* circuit (exported with to_stim).
"""

from std.sys import argv
from std.time import perf_counter_ns

from qumo.qec.codes import surface_code_memory, NoiseParams
from qumo.qec.dem import detector_error_model
from qumo.qec.decoders import MatchingGraph, UnionFindDecoder, MwpmDecoder
from qumo.qec.experiments import sample_and_decode
from qumo.sim.frame import FrameSimulator, FrameProgram


def secs(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e9


def run_point(d: Int, p: Float64, shots: Int, seed: UInt64) raises:
    var cc = surface_code_memory(d, d, NoiseParams.uniform(p), "Z")
    var c = cc.circuit.copy()
    var t0 = perf_counter_ns()
    var dem = detector_error_model(c)
    var t_dem = secs(t0)
    t0 = perf_counter_ns()
    var fs = FrameSimulator(c)
    var t_ref = secs(t0)
    var ds = fs.sample_detectors(shots, seed)
    var t_mt = ds.seconds
    var ds1 = fs.sample_detectors(shots, seed, threads=1)
    var t_st = ds1.seconds
    var ops = Float64(len(c.ops))
    var prog = FrameProgram(c)
    t0 = perf_counter_ns()
    var g = MatchingGraph(dem)
    var uf = UnionFindDecoder(g.copy())
    var mw = MwpmDecoder(g.copy())
    var t_graph = secs(t0)
    var su = sample_and_decode(prog, uf, shots, seed + 1)
    var sm = sample_and_decode(prog, mw, shots, seed + 1)
    # CSV: d,p,qubits,ops,detectors,dem_errors,dem_s,ref_s,graph_s,shots,
    #      sample_s_mt,sample_s_1t,shots_per_s_mt,shot_ops_per_s_mt,shots_per_s_1t,
    #      uf_errors,uf_s,mwpm_errors,mwpm_s,events_per_shot
    print("QUMO," + String(d) + "," + String(p) + "," + String(c.num_qubits) + "," + String(Int(ops)) + ","
          + String(c.num_detectors) + "," + String(dem.num_errors()) + "," + String(t_dem) + ","
          + String(t_ref) + "," + String(t_graph) + "," + String(shots) + "," + String(t_mt) + ","
          + String(t_st) + "," + String(Float64(shots) / t_mt) + "," + String(ops * Float64(shots) / t_mt) + ","
          + String(Float64(shots) / t_st) + "," + String(su.errors) + "," + String(su.seconds) + ","
          + String(sm.errors) + "," + String(sm.seconds) + ","
          + String(Float64(ds.num_detection_events()) / Float64(shots)), flush=True)


def main() raises:
    var args = argv()
    if len(args) >= 4 and String(args[1]) == "--stim":
        var cc = surface_code_memory(Int(String(args[2])), Int(String(args[2])), NoiseParams.uniform(Float64(String(args[3]))), "Z")
        print(cc.to_stim(), end="")
        return
    if len(args) >= 4 and String(args[1]) == "--dem":
        var cc = surface_code_memory(Int(String(args[2])), Int(String(args[2])), NoiseParams.uniform(Float64(String(args[3]))), "Z")
        print(detector_error_model(cc.circuit).to_text(), end="")
        return
    if len(args) >= 5 and String(args[1]) == "--run":
        run_point(Int(String(args[2])), Float64(String(args[3])), Int(String(args[4])), 2024)
        return
    var shots = 200000
    if len(args) >= 2:
        shots = Int(String(args[1]))
    var ds: List[Int] = [3, 5, 7]
    var ps: List[Float64] = [0.001, 0.003, 0.005]
    print("d,p,qubits,ops,detectors,dem_errors,dem_s,ref_s,graph_s,shots,sample_s_mt,sample_s_1t,shots_per_s_mt,shot_ops_per_s_mt,shots_per_s_1t,uf_errors,uf_s,mwpm_errors,mwpm_s,events_per_shot")
    for d in ds:
        for p in ps:
            run_point(d, p, shots, 2024)
