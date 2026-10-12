"""Sinter-style QEC experiment harness.

    var r = memory_experiment("surface_z", distance=5, p=0.001, shots=1_000_000,
                              decoder="mwpm", seed=7, max_errors=200)
    print(r)            # logical error rate, Wilson interval, timings
    print(r.to_json())  # machine-readable, for agents

Pipeline (each stage is a QUMO module): code generator (`qumo.qec.codes`) ->
DEM extraction (`qumo.qec.dem`) -> matching graph + decoder
(`qumo.qec.decoders`) -> bit-packed Pauli-frame sampling (`qumo.sim.frame`)
fused with decoding: every worker samples a 512-shot batch, decodes it and
counts logical errors, so detection events never hit memory in bulk.

Like Sinter, shots are processed in fixed "waves" (64 batches = 32768 shots)
and the run stops early once `max_errors` logical errors were seen. Batch RNG
streams depend only on (seed, batch index), so results are bit-identical
regardless of thread count, and the stopping point is deterministic too.

`threshold_sweep` runs a grid of distances x physical error rates and returns
CSV (or `threshold_sweep_results` for structured results).

Decoder registry (`canonical_decoder_name`): 'mwpm' = exact minimum-weight
perfect matching by sparse blossom (`qumo.qec.sparse_blossom`, the algorithm
of PyMatching 2) - the default; 'sparse_blossom' (same decoder);
'mwpm_dense' = the older exact APSP + dense-blossom `MwpmDecoder`;
'uf' = union-find; 'lookup' = lookup table.

Error codes: QM-E706 (bad code), QM-E707 (unknown decoder).
"""

from std.math import sqrt, pow
from std.time import perf_counter_ns
from std.runtime import parallelism_level
from max.algorithm import parallelize

from qumo.ir.circuit import Circuit
from qumo.sim.rng import Rng
from qumo.sim.frame import FrameProgram, FrameBuffers, batch_rng, FW, BATCH
from qumo.qec.codes import NoiseParams, CodeCircuit, generate_code
from qumo.qec.dem import DetectorErrorModel, detector_error_model
from qumo.qec.decoders import (
    Decoder, MatchingGraph, UnionFindDecoder, MwpmDecoder, LookupTableDecoder, batch_fired,
)
from qumo.qec.sparse_blossom import SparseBlossomDecoder

comptime WAVE_BATCHES = 64


struct ExperimentResult(Copyable, Writable):
    var code: String
    var basis: String
    var decoder: String
    var distance: Int
    var rounds: Int
    var p: Float64
    var shots: Int
    var errors: Int
    var seconds: Float64
    var setup_seconds: Float64
    var num_qubits: Int
    var num_detectors: Int
    var num_dem_errors: Int
    var num_edges: Int
    var seed: UInt64

    def __init__(out self):
        self.code = String()
        self.basis = String()
        self.decoder = String()
        self.distance = 0
        self.rounds = 0
        self.p = 0.0
        self.shots = 0
        self.errors = 0
        self.seconds = 0.0
        self.setup_seconds = 0.0
        self.num_qubits = 0
        self.num_detectors = 0
        self.num_dem_errors = 0
        self.num_edges = 0
        self.seed = 0

    def logical_error_rate(self) -> Float64:
        return Float64(self.errors) / Float64(max(1, self.shots))

    def std_error(self) -> Float64:
        var q = self.logical_error_rate()
        return sqrt(q * (1.0 - q) / Float64(max(1, self.shots)))

    def wilson_interval(self, z: Float64 = 1.96) -> Tuple[Float64, Float64]:
        var n = Float64(max(1, self.shots))
        var q = self.logical_error_rate()
        var den = 1.0 + z * z / n
        var centre = (q + z * z / (2.0 * n)) / den
        var half = z * sqrt(q * (1.0 - q) / n + z * z / (4.0 * n * n)) / den
        return (max(0.0, centre - half), min(1.0, centre + half))

    def per_round_error_rate(self) -> Float64:
        """1 - (1 - 2 p_L)^(1/r), halved: the per-round logical flip rate."""
        var q = self.logical_error_rate()
        if q >= 0.5 or self.rounds <= 0:
            return q
        return 0.5 * (1.0 - pow(1.0 - 2.0 * q, 1.0 / Float64(self.rounds)))

    def shots_per_second(self) -> Float64:
        return Float64(self.shots) / max(self.seconds, 1e-12)

    @staticmethod
    def csv_header() -> String:
        return "code,basis,decoder,distance,rounds,p,shots,errors,logical_error_rate,std_error,wilson_lo,wilson_hi,per_round_error_rate,seconds,shots_per_second"

    def csv_row(self) -> String:
        var wi = self.wilson_interval()
        return (
            self.code + "," + self.basis + "," + self.decoder + "," + String(self.distance) + ","
            + String(self.rounds) + "," + String(self.p) + "," + String(self.shots) + ","
            + String(self.errors) + "," + String(self.logical_error_rate()) + ","
            + String(self.std_error()) + "," + String(wi[0]) + "," + String(wi[1]) + ","
            + String(self.per_round_error_rate()) + "," + String(self.seconds) + ","
            + String(self.shots_per_second())
        )

    def to_json(self) -> String:
        var wi = self.wilson_interval()
        return (
            "{\"code\": \"" + self.code + "\", \"basis\": \"" + self.basis + "\", \"decoder\": \""
            + self.decoder + "\", \"distance\": " + String(self.distance) + ", \"rounds\": "
            + String(self.rounds) + ", \"p\": " + String(self.p) + ", \"shots\": "
            + String(self.shots) + ", \"errors\": " + String(self.errors)
            + ", \"logical_error_rate\": " + String(self.logical_error_rate())
            + ", \"std_error\": " + String(self.std_error()) + ", \"wilson\": [" + String(wi[0])
            + ", " + String(wi[1]) + "], \"per_round_error_rate\": "
            + String(self.per_round_error_rate()) + ", \"seconds\": " + String(self.seconds)
            + ", \"setup_seconds\": " + String(self.setup_seconds) + ", \"num_qubits\": "
            + String(self.num_qubits) + ", \"num_detectors\": " + String(self.num_detectors)
            + ", \"num_dem_errors\": " + String(self.num_dem_errors) + ", \"seed\": "
            + String(self.seed) + "}"
        )

    def write_to(self, mut w: Some[Writer]):
        var wi = self.wilson_interval()
        w.write(self.code, "(", self.basis, ") d=", self.distance, " r=", self.rounds, " p=", self.p,
                " decoder=", self.decoder, ": ", self.errors, "/", self.shots, " -> p_L=",
                self.logical_error_rate(), " [", wi[0], ", ", wi[1], "]  (",
                self.seconds, " s, ", self.shots_per_second(), " shots/s)")


@fieldwise_init
struct RunStats(ImplicitlyCopyable, Writable):
    var shots: Int
    var errors: Int
    var seconds: Float64

    def write_to(self, mut w: Some[Writer]):
        w.write("RunStats(shots=", self.shots, ", errors=", self.errors, ", seconds=", self.seconds, ")")


def sample_and_decode[D: Decoder](prog: FrameProgram, dec: D, shots: Int, seed: UInt64, max_errors: Int = -1) -> RunStats:
    """Fused parallel sample+decode loop; counts shots whose predicted
    observable mask differs from the sampled one."""
    var total_batches = (shots + BATCH - 1) // BATCH
    var errs = List[Int](length=max(1, total_batches), fill=0)
    var eptr = errs.unsafe_ptr().as_unsafe_any_origin()
    var t0 = perf_counter_ns()
    var done_batches = 0
    var nd = prog.num_detectors
    var nk = prog.num_observables
    var total_errors = 0
    while done_batches < total_batches:
        var lo_b = done_batches
        var hi_b = min(total_batches, lo_b + WAVE_BATCHES)
        var nb = hi_b - lo_b
        var nchunks = max(1, min(nb, 2 * max(1, parallelism_level())))
        var per = (nb + nchunks - 1) // nchunks

        def task(c: Int) {imm prog, imm dec, imm eptr, imm per, imm lo_b, imm hi_b, imm nd, imm nk, imm shots, imm seed}:
            var local = dec.copy()
            var buf = FrameBuffers(prog)
            var offsets = List[Int](length=BATCH + 1, fill=0)
            var flat = List[Int]()
            var fired = List[Int]()
            var b0 = lo_b + c * per
            var b1 = min(hi_b, b0 + per)
            for b in range(b0, b1):
                var rng = batch_rng(seed, b)
                buf.run(prog, rng)
                batch_fired(buf.dets, 0, nd, offsets, flat)
                var e = 0
                for s in range(BATCH):
                    if b * BATCH + s >= shots:
                        break
                    fired.clear()
                    for t in range(offsets[s], offsets[s + 1]):
                        fired.append(flat[t])
                    var pred = local.decode(fired)
                    var actual: UInt64 = 0
                    var w = s >> 6
                    var bit = UInt64(s & 63)
                    for k in range(min(nk, 64)):
                        actual |= ((buf.obs[k * FW + w] >> bit) & 1) << UInt64(k)
                    if pred != actual:
                        e += 1
                eptr[unsafe_offset=b] = e

        parallelize(task, nchunks)
        for b in range(lo_b, hi_b):
            total_errors += errs[b]
        done_batches = hi_b
        if max_errors > 0 and total_errors >= max_errors:
            break
    var done_shots = min(shots, done_batches * BATCH)
    return RunStats(done_shots, total_errors, Float64(perf_counter_ns() - t0) / 1e9)


def canonical_decoder_name(name: String) -> String:
    """Registry of decoder names: 'mwpm' (sparse blossom, the default exact
    matcher), 'sparse_blossom', 'mwpm_dense', 'uf', 'lookup' and their aliases.
    Returns "" for an unknown name."""
    var k = name.lower()
    if k == "mwpm" or k == "matching" or k == "pymatching":
        return "mwpm"
    if k == "sparse_blossom" or k == "sparse-blossom" or k == "sparseblossom" or k == "sb":
        return "sparse_blossom"
    if k == "mwpm_dense" or k == "mwpm-dense" or k == "dense_mwpm" or k == "dense":
        return "mwpm_dense"
    if k == "uf" or k == "union_find" or k == "unionfind":
        return "uf"
    if k == "lookup" or k == "lut" or k == "table":
        return "lookup"
    return ""


def make_decoder_and_run(name: String, dem: DetectorErrorModel, prog: FrameProgram, shots: Int, seed: UInt64, max_errors: Int, mut edges: Int) raises -> RunStats:
    """Build decoder `name` for `dem` and run the fused loop.

    'mwpm' / 'sparse_blossom': exact MWPM by sparse blossom
    (`qumo.qec.sparse_blossom`, PyMatching 2's algorithm) - the default;
    'mwpm_dense': exact MWPM by all-pairs shortest paths + dense blossom
    (`MwpmDecoder`); 'uf': union-find; 'lookup': lookup table.
    """
    var k = canonical_decoder_name(name)
    if k == "mwpm" or k == "sparse_blossom":
        var g = MatchingGraph(dem)
        edges = g.num_edges()
        return sample_and_decode(prog, SparseBlossomDecoder(g), shots, seed, max_errors)
    if k == "mwpm_dense":
        var g = MatchingGraph(dem)
        edges = g.num_edges()
        return sample_and_decode(prog, MwpmDecoder(g^), shots, seed, max_errors)
    if k == "uf":
        var g = MatchingGraph(dem)
        edges = g.num_edges()
        return sample_and_decode(prog, UnionFindDecoder(g^), shots, seed, max_errors)
    if k == "lookup":
        edges = 0
        return sample_and_decode(prog, LookupTableDecoder(dem, 2), shots, seed, max_errors)
    raise Error("QM-E707: unknown decoder '" + name + "'. Hint: use 'mwpm' (sparse blossom, default), 'sparse_blossom', 'mwpm_dense', 'uf' or 'lookup', or call sample_and_decode() with your own Decoder.")


def run_circuit_experiment(cc: CodeCircuit, shots: Int, decoder: String = "mwpm", seed: UInt64 = 1, max_errors: Int = -1) raises -> ExperimentResult:
    """Decode an already-generated code circuit (custom noise models welcome)."""
    var t0 = perf_counter_ns()
    var dem = detector_error_model(cc.circuit)
    var prog = FrameProgram(cc.circuit)
    var setup = Float64(perf_counter_ns() - t0) / 1e9
    var edges = 0
    var st = make_decoder_and_run(decoder, dem, prog, shots, seed, max_errors, edges)
    var r = ExperimentResult()
    r.code = cc.code
    r.basis = cc.basis
    r.decoder = decoder
    r.distance = cc.distance
    r.rounds = cc.rounds
    r.p = cc.noise.after_clifford_depolarization
    r.shots = st.shots
    r.errors = st.errors
    r.seconds = st.seconds
    r.setup_seconds = setup
    r.num_qubits = cc.circuit.num_qubits
    r.num_detectors = cc.circuit.num_detectors
    r.num_dem_errors = dem.num_errors()
    r.num_edges = edges
    r.seed = seed
    return r^


def memory_experiment(code: String = "surface_z", distance: Int = 3, rounds: Int = -1, p: Float64 = 0.001,
                      shots: Int = 100000, decoder: String = "mwpm", seed: UInt64 = 1,
                      max_errors: Int = -1) raises -> ExperimentResult:
    """Memory experiment under uniform circuit-level noise p (rounds = d if < 1).

    code: 'surface_z' | 'surface_x' | 'repetition'; decoder: 'mwpm' (sparse
    blossom) | 'sparse_blossom' | 'mwpm_dense' | 'uf' | 'lookup'.
    """
    var r = rounds if rounds >= 1 else distance
    var cc = generate_code(code, distance, r, NoiseParams.uniform(p))
    var res = run_circuit_experiment(cc, shots, decoder, seed, max_errors)
    res.code = code
    res.p = p
    return res^


def threshold_sweep_results(code: String, distances: List[Int], ps: List[Float64], shots: Int,
                            decoder: String = "mwpm", seed: UInt64 = 1, max_errors: Int = -1,
                            rounds: Int = -1) raises -> List[ExperimentResult]:
    var out = List[ExperimentResult]()
    for d in distances:
        for p in ps:
            out.append(memory_experiment(code, d, rounds, p, shots, decoder, seed, max_errors))
    return out^


def threshold_sweep(code: String, distances: List[Int], ps: List[Float64], shots: Int,
                    decoder: String = "mwpm", seed: UInt64 = 1, max_errors: Int = -1,
                    rounds: Int = -1) raises -> String:
    """Grid of memory experiments as CSV (header + one row per (d, p))."""
    var rs = threshold_sweep_results(code, distances, ps, shots, decoder, seed, max_errors, rounds)
    var s = ExperimentResult.csv_header() + "\n"
    for r in rs:
        s += r.csv_row() + "\n"
    return s


def results_to_json(rs: List[ExperimentResult]) -> String:
    var s = String("[")
    for i in range(len(rs)):
        if i > 0:
            s += ",\n "
        s += rs[i].to_json()
    return s + "]"
