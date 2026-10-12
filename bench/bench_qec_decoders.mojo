"""Decoder benchmark on *identical* syndromes (driven by bench/bench_qec.py).

    mojo build -I . -o build/bench_qec_decoders bench/bench_qec_decoders.mojo
    build/bench_qec_decoders --stim CODE D P                 # Stim text of the circuit
    build/bench_qec_decoders --decode CODE D P FILE.b8 SHOTS DECODER REPS
    build/bench_qec_decoders --self CODE D P SHOTS DECODER REPS

`--decode` reads detection events in Stim's b8 format (one record of
ceil(num_detectors/8) little-endian bytes per shot, e.g. written by
`stim ... sample_detectors(..., bit_packed=True)`), decodes every shot with
DECODER ('sparse_blossom' | 'mwpm_dense' | 'uf') built from QUMO's own DEM of
the same circuit, and prints one CSV line

    decoder,code,d,p,shots,events_per_shot,t1_median_s,tall_median_s,threads,anomalies,failures

then writes the predicted observable masks (one byte per shot) to FILE.pred.
`t1` is single-thread decode time (serial loop, one decoder), `tall` the
parallel time (one decoder copy per worker, chunks of shots); both are the
median of REPS repetitions and exclude sampling and file I/O, exactly like
timing `pymatching.Matching.decode_batch` on the same array.
`--self` does the same on QUMO-sampled syndromes (no Python needed).
"""

from std.sys import argv
from std.time import perf_counter_ns
from std.runtime import parallelism_level
from max.algorithm import parallelize

from qumo.qec.codes import NoiseParams, generate_code
from qumo.qec.dem import detector_error_model
from qumo.qec.decoders import Decoder, MatchingGraph, MwpmDecoder, UnionFindDecoder
from qumo.qec.sparse_blossom import SparseBlossomDecoder
from qumo.sim.frame import FrameSimulator


def median(var xs: List[Float64]) -> Float64:
    sort(xs)
    return xs[len(xs) // 2]


def time_decoder[D: Decoder](dec: D, offs: List[Int], flat: List[Int], shots: Int, reps: Int,
                             mut preds: List[UInt64]) -> Tuple[Float64, Float64]:
    var t1s = List[Float64]()
    var tas = List[Float64]()
    var nthr = max(1, parallelism_level())
    var nchunks = max(1, min(shots, 8 * nthr))
    var per = (shots + nchunks - 1) // nchunks
    var pptr = preds.unsafe_ptr().as_unsafe_any_origin()
    for _ in range(reps):
        # single thread
        var local = dec.copy()
        var fired = List[Int]()
        var t0 = perf_counter_ns()
        for s in range(shots):
            fired.clear()
            for t in range(offs[s], offs[s + 1]):
                fired.append(flat[t])
            preds[s] = local.decode(fired)
        t1s.append(Float64(perf_counter_ns() - t0) / 1e9)

        # all threads
        def task(c: Int) {imm dec, imm offs, imm flat, imm pptr, imm per, imm shots}:
            var ld = dec.copy()
            var fl = List[Int]()
            for s in range(c * per, min(shots, (c + 1) * per)):
                fl.clear()
                for t in range(offs[s], offs[s + 1]):
                    fl.append(flat[t])
                pptr[unsafe_offset=s] = ld.decode(fl)

        var t2 = perf_counter_ns()
        parallelize(task, nchunks)
        tas.append(Float64(perf_counter_ns() - t2) / 1e9)
    _ = len(preds)
    return (median(t1s^), median(tas^))


def load_b8(path: String, shots: Int, nd: Int, mut offs: List[Int], mut flat: List[Int]) raises:
    var bytes: List[UInt8]
    with open(path, "r") as f:
        bytes = f.read_bytes()
    var rec = (nd + 7) // 8
    if len(bytes) < shots * rec:
        raise Error("QM-E705: b8 file too short (" + String(len(bytes)) + " bytes for " + String(shots) + " shots of " + String(nd) + " detectors). Hint: check SHOTS and the circuit parameters.")
    offs.clear()
    flat.clear()
    offs.append(0)
    for s in range(shots):
        var base = s * rec
        for b in range(rec):
            var x = Int(bytes[base + b])
            while x != 0:
                var bit = 0
                while ((x >> bit) & 1) == 0:
                    bit += 1
                x &= x - 1
                var det = b * 8 + bit
                if det < nd:
                    flat.append(det)
        offs.append(len(flat))


def run(code: String, d: Int, p: Float64, shots: Int, decoder: String, reps: Int, path: String) raises:
    var cc = generate_code(code, d, d, NoiseParams.uniform(p))
    var dem = detector_error_model(cc.circuit)
    var g = MatchingGraph(dem)
    var nd = g.num_detectors
    var offs = List[Int]()
    var flat = List[Int]()
    if path.byte_length() > 0:
        load_b8(path, shots, nd, offs, flat)
    else:
        var ds = FrameSimulator(cc.circuit).sample_detectors(shots, 2024)
        offs.append(0)
        for s in range(shots):
            for x in ds.fired(s):
                flat.append(x)
            offs.append(len(flat))
    var preds = List[UInt64](length=max(1, shots), fill=0)
    var tt: Tuple[Float64, Float64]
    var anomalies = 0
    var failures = 0
    var k = decoder.lower()
    if k == "sparse_blossom" or k == "mwpm":
        var dec = SparseBlossomDecoder(g)
        tt = time_decoder(dec, offs, flat, shots, reps, preds)
        # serial pass for the statistics (copies inside time_decoder are discarded)
        var fired = List[Int]()
        for s in range(shots):
            fired.clear()
            for t in range(offs[s], offs[s + 1]):
                fired.append(flat[t])
            _ = dec.decode(fired)
        anomalies = dec.anomalies
        failures = dec.failures
    elif k == "mwpm_dense":
        tt = time_decoder(MwpmDecoder(g.copy()), offs, flat, shots, reps, preds)
    elif k == "uf":
        tt = time_decoder(UnionFindDecoder(g.copy()), offs, flat, shots, reps, preds)
    else:
        raise Error("QM-E707: unknown decoder '" + decoder + "'. Hint: sparse_blossom, mwpm_dense or uf.")
    print(decoder + "," + code + "," + String(d) + "," + String(p) + "," + String(shots) + ","
          + String(Float64(len(flat)) / Float64(max(1, shots))) + "," + String(tt[0]) + "," + String(tt[1]) + ","
          + String(parallelism_level()) + "," + String(anomalies) + "," + String(failures), flush=True)
    if path.byte_length() > 0:
        var out = List[UInt8](capacity=shots)
        for s in range(shots):
            out.append(UInt8(preds[s] & 0xFF))
        with open(path + ".pred", "w") as f:
            f.write_bytes(Span(out))


def main() raises:
    var args = argv()
    if len(args) >= 5 and String(args[1]) == "--stim":
        var cc = generate_code(String(args[2]), Int(String(args[3])), Int(String(args[3])), NoiseParams.uniform(Float64(String(args[4]))))
        print(cc.to_stim(), end="")
        return
    if len(args) >= 9 and String(args[1]) == "--decode":
        run(String(args[2]), Int(String(args[3])), Float64(String(args[4])), Int(String(args[6])),
            String(args[7]), Int(String(args[8])), String(args[5]))
        return
    if len(args) >= 8 and String(args[1]) == "--self":
        run(String(args[2]), Int(String(args[3])), Float64(String(args[4])), Int(String(args[5])),
            String(args[6]), Int(String(args[7])), String(""))
        return
    print("usage: --stim CODE D P | --decode CODE D P FILE.b8 SHOTS DECODER REPS | --self CODE D P SHOTS DECODER REPS")
