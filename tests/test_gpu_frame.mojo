"""GPU Pauli-frame sampler tests (qumo.gpu.frame_gpu).

Default build: the exact kernel body `th_frame_word` runs through the host
emulator over the real launch grid and is checked against the CPU
`qumo.sim.frame.FrameSimulator`:
  * integer geometric-skip thresholds (lane-uniform Bernoulli rates);
  * noiseless surface / repetition codes: zero detection events;
  * deterministic injected Paulis (X/Y/Z_ERROR and PAULI_CHANNEL_1 with
    probability 1 at random places): detection events and observable flips
    identical to the CPU sampler for every shot;
  * circuit-level noise: per-detector rates and logical flip rate agree with
    the CPU sampler within 5 sigma;
  * feed-forward (c_if Pauli), resets, random measurements, reference XOR;
  * reproducibility: same seed -> same bits, independent of the chunking;
  * errors (QM-E701 non-Clifford, QM-E470 device mode without -D QUMO_GPU),
    dispatch (`sample_on(..., "gpu")`).

`-D QUMO_GPU` build with a usable device: the same checks on the GPU, plus
device == emulator bit-for-bit (the kernel is integer-only).
"""

from std.math import sqrt
from std.sys.defines import is_defined
from std.os import getenv
from std.testing import assert_equal, assert_true

from qumo.ir.gates import Op
from qumo.ir.circuit import Circuit, Instruction
from qumo.sim.rng import Rng
from qumo.sim.frame import FrameSimulator, DetectorSamples, MeasurementSamples
from qumo.qec.codes import generate_code, NoiseParams
from qumo.gpu.frame_gpu import (
    GpuFrameSampler, skip_thresholds, noise_mask, THR_INF, FRAME_CHUNK_WORDS,
)
from qumo.gpu.device import gpu_compiled, gpu_available, gpu_info
from qumo.exec.dispatch import sample_on, frame_eligible_gpu


def check_thresholds() raises:
    """Bernoulli(p) lanes from the integer skip tables: rate and lane uniformity."""
    var one = List[UInt32]()
    skip_thresholds(1.0, one)
    for t in one:
        assert_equal(Int(t), 0)
    var tiny = List[UInt32]()
    skip_thresholds(1e-13, tiny)
    for t in tiny:
        assert_true(t == THR_INF)
    var rng = Rng(5)
    var pone = one.unsafe_ptr().as_unsafe_any_origin()
    for _ in range(100):
        assert_true(noise_mask(rng, pone, 0) == ~UInt64(0))
    var ps: List[Float64] = [0.001, 0.02, 0.3, 0.5, 0.93]
    for p in ps:
        var thr = List[UInt32]()
        skip_thresholds(p, thr)
        for k in range(1, 64):
            assert_true(thr[k] <= thr[k - 1], "thresholds must be non-increasing")
        var pt = thr.unsafe_ptr().as_unsafe_any_origin()
        var words = 20000 if p > 0.01 else 200000
        var total = 0
        var lane0 = 0
        var lane63 = 0
        for _ in range(words):
            var m = noise_mask(rng, pt, 0)
            total += Int(pop_count64(m))
            lane0 += Int(m & 1)
            lane63 += Int((m >> 63) & 1)
        var n = Float64(words) * 64.0
        var f = Float64(total) / n
        var sig = sqrt(p * (1 - p) / n)
        assert_true(abs(f - p) < 5 * sig + 1e-9, "Bernoulli rate p=" + String(p) + " got " + String(f))
        var sl = sqrt(p * (1 - p) / Float64(words))
        assert_true(abs(Float64(lane0) / Float64(words) - p) < 5 * sl + 1e-9, "lane 0 rate p=" + String(p))
        assert_true(abs(Float64(lane63) / Float64(words) - p) < 5 * sl + 1e-9, "lane 63 rate p=" + String(p))
        _ = len(thr)
    _ = len(one)


def pop_count64(v: UInt64) -> Int:
    var c = 0
    var x = v
    while x != 0:
        x &= x - 1
        c += 1
    return c


def code(name: String, d: Int, rounds: Int, p: Float64) raises -> Circuit:
    return generate_code(name, d, rounds, NoiseParams.uniform(p)).circuit.copy()


def same_detectors(a: DetectorSamples, b: DetectorSamples, what: String) raises:
    assert_equal(a.shots, b.shots)
    assert_equal(a.num_detectors, b.num_detectors)
    assert_equal(a.num_observables, b.num_observables)
    for s in range(a.shots):
        for d in range(a.num_detectors):
            assert_true(a.get(s, d) == b.get(s, d), what + ": detector " + String(d) + " shot " + String(s))
        for k in range(a.num_observables):
            assert_true(a.observable(s, k) == b.observable(s, k), what + ": observable " + String(k) + " shot " + String(s))


def check_noiseless(emulate: Bool) raises:
    var names: List[String] = ["surface_z", "surface_x", "repetition"]
    for name in names:
        var c = code(name, 3, 3, 0.0)
        var gs = GpuFrameSampler(c, emulate)
        var ds = gs.sample_detectors(3000, 11)
        assert_equal(ds.num_detection_events(), 0, name + ": noiseless detectors must be quiet")
        assert_equal(ds.observable_flip_count(0), 0, name + ": noiseless observable must not flip")


def inject(c: Circuit, seed: UInt64, count: Int) -> Circuit:
    """Insert `count` probability-1 Paulis (X/Y/Z_ERROR, PAULI_CHANNEL_1) at random places."""
    var rng = Rng(seed)
    var pos = List[Int]()
    for _ in range(count):
        pos.append(rng.below(len(c.ops)))
    sort(pos)
    var out = c.copy()
    out.ops = List[Instruction]()
    var k = 0
    for i in range(len(c.ops)):
        while k < len(pos) and pos[k] == i:
            var q = rng.below(c.num_qubits)
            var r = rng.below(6)
            var one: Float64 = 1.0
            var zero: Float64 = 0.0
            if r == 0:
                out.ops.append(Instruction.make(Op.X_ERROR, q, p0=one))
            elif r == 1:
                out.ops.append(Instruction.make(Op.Y_ERROR, q, p0=one))
            elif r == 2:
                out.ops.append(Instruction.make(Op.Z_ERROR, q, p0=one))
            elif r == 3:
                out.ops.append(Instruction.make(Op.PAULI_CHANNEL_1, q, p0=one, p1=zero, p2=zero))
            elif r == 4:
                out.ops.append(Instruction.make(Op.PAULI_CHANNEL_1, q, p0=zero, p1=one, p2=zero))
            else:
                out.ops.append(Instruction.make(Op.PAULI_CHANNEL_1, q, p0=zero, p1=zero, p2=one))
            k += 1
        out.ops.append(c.ops[i])
    return out^


def check_deterministic(emulate: Bool) raises:
    """Deterministic Pauli errors: GPU == CPU frame sampler for every shot."""
    var names: List[String] = ["surface_z", "surface_x", "repetition"]
    var fired = 0
    for t in range(len(names)):
        var base = code(names[t], 3, 3, 0.0)
        for rep in range(4):
            var c = inject(base, UInt64(100 * t + rep), 1 + 2 * rep)
            var cpu = FrameSimulator(c).sample_detectors(700, 3)
            var gpu = GpuFrameSampler(c, emulate).sample_detectors(700, 9)
            same_detectors(cpu, gpu, names[t] + " injected " + String(rep))
            fired += gpu.num_detection_events()
    assert_true(fired > 0, "injected errors must fire detectors")


def check_statistics(emulate: Bool) raises:
    """Circuit-level noise: per-detector and logical rates match the CPU sampler."""
    var names: List[String] = ["surface_z", "repetition"]
    var ps: List[Float64] = [0.01, 0.03]
    for t in range(len(names)):
        var c = code(names[t], 3, 3, ps[t])
        var shots = 40000
        var cpu = FrameSimulator(c).sample_detectors(shots, 17)
        var gs = GpuFrameSampler(c, emulate)
        var gpu = gs.sample_detectors(shots, 23)
        assert_equal(gpu.shots, shots)
        for d in range(c.num_detectors):
            var fc = cpu.detector_fraction(d)
            var fg = gpu.detector_fraction(d)
            var f = 0.5 * (fc + fg)
            var sig = sqrt(2.0 * f * (1.0 - f) / Float64(shots))
            assert_true(abs(fc - fg) < 5.0 * sig + 2e-4, names[t] + " detector " + String(d) + ": cpu " + String(fc) + " gpu " + String(fg))
        var lc = Float64(cpu.observable_flip_count(0)) / Float64(shots)
        var lg = Float64(gpu.observable_flip_count(0)) / Float64(shots)
        var lf = 0.5 * (lc + lg)
        var ls = sqrt(2.0 * lf * (1.0 - lf) / Float64(shots))
        assert_true(abs(lc - lg) < 5.0 * ls + 2e-4, names[t] + " logical flip rate: cpu " + String(lc) + " gpu " + String(lg))
        assert_true(gs.throughput(gpu) > 0.0)


def check_measurements(emulate: Bool) raises:
    """Feed-forward, resets, random outcomes and the reference XOR."""
    var c = Circuit(5)
    c.h(0)
    c.measure(0, 0)  # random
    c.x(1)
    c.c_if(0, 1)  # q1 copies m0 through feed-forward
    c.measure(1, 1)
    c.h(2)
    c.s(2)
    c.h(2)
    c.measure(2, 2)  # random (Y-basis state measured in Z)
    c.x(3)
    c.measure(3, 3)  # deterministic 1 (reference sample)
    c.reset(3)
    c.measure(3, 4)  # deterministic 0 after reset
    c.h(4)
    c.cx(4, 0)
    c.measure(4, 5)
    c.measure(0, 6)  # m6 == m5 (Bell pair) xor m0's earlier collapse
    c.x_error(0.25, 1)
    c.measure(1, 7)  # m7 = m1 xor Bernoulli(0.25)
    var shots = 6000
    var gs = GpuFrameSampler(c, emulate)
    var ms = gs.sample_measurements(shots, 77)
    var cpu = FrameSimulator(c).sample_measurements(shots, 77)
    var ones0 = 0
    var ones2 = 0
    var flips7 = 0
    for s in range(shots):
        assert_true(ms.get(s, 1) == ms.get(s, 0), "feed-forward X must copy the record")
        assert_true(ms.get(s, 3), "X|0> measures 1")
        assert_true(not ms.get(s, 4), "reset measures 0")
        assert_true(ms.get(s, 6) == (ms.get(s, 5) != ms.get(s, 0)), "Bell pair parity")
        if ms.get(s, 0): ones0 += 1
        if ms.get(s, 2): ones2 += 1
        if ms.get(s, 7) != ms.get(s, 1): flips7 += 1
    var sig = sqrt(0.25 / Float64(shots))
    assert_true(abs(Float64(ones0) / Float64(shots) - 0.5) < 5 * sig, "random measurement 0")
    assert_true(abs(Float64(ones2) / Float64(shots) - 0.5) < 5 * sig, "random measurement 2")
    var s7 = sqrt(0.25 * 0.75 / Float64(shots))
    assert_true(abs(Float64(flips7) / Float64(shots) - 0.25) < 5 * s7, "X_ERROR(0.25) rate")
    # the CPU sampler satisfies the same invariants (sanity of the reference)
    var cflips = 0
    for s in range(shots):
        assert_true(cpu.get(s, 1) == cpu.get(s, 0))
        if cpu.get(s, 7) != cpu.get(s, 1): cflips += 1
    assert_true(abs(Float64(cflips - flips7)) / Float64(shots) < 10 * s7)
    # Y feed-forward also copies the record into the X part of the frame
    var y = Circuit(2)
    y.h(0)
    y.measure(0, 0)
    y.y(1)
    y.c_if(0, 1)
    y.measure(1, 1)
    var my = GpuFrameSampler(y, emulate).sample_measurements(1000, 4)
    for s in range(1000):
        assert_true(my.get(s, 0) == my.get(s, 1), "Y feed-forward")


def check_reproducible(emulate: Bool) raises:
    var c = code("surface_z", 3, 2, 0.02)
    var a = GpuFrameSampler(c, emulate)
    var b = GpuFrameSampler(c, emulate, max_words=3)  # many small launches
    assert_equal(b.num_kernel_launches(5000), 27)  # ceil(5000/512)*8 = 80 words
    assert_equal(a.num_kernel_launches(5000), 1)
    var da = a.sample_detectors(5000, 31)
    var db = b.sample_detectors(5000, 31)
    var da2 = a.sample_detectors(5000, 31)
    same_detectors(da, db, "chunking must not change samples")
    same_detectors(da, da2, "same seed")
    var other = a.sample_detectors(5000, 32)
    var differ = False
    for i in range(len(da.dets)):
        if da.dets[i] != other.dets[i]:
            differ = True
    assert_true(differ, "different seeds must give different samples")
    # partial last word / batch: counts never include lanes beyond `shots`
    var e = GpuFrameSampler(code("repetition", 3, 2, 1.0), emulate).sample_detectors(70, 1)
    assert_equal(e.shots, 70)
    assert_true(e.num_detection_events() <= 70 * e.num_detectors)


def check_errors_and_dispatch() raises:
    var t = Circuit(1)
    t.t(0)
    t.measure(0, 0)
    var raised = False
    try:
        _ = GpuFrameSampler(t, emulate=True)
    except e:
        raised = String(e).startswith("QM-E701")
    assert_true(raised, "QM-E701 expected for a non-Clifford circuit")
    var c = code("repetition", 3, 2, 0.01)
    assert_true(frame_eligible_gpu(c))
    comptime if not is_defined["QUMO_GPU"]():
        raised = False
        try:
            _ = GpuFrameSampler(c)
        except e:
            raised = String(e).startswith("QM-E470")
        assert_true(raised, "QM-E470 expected without -D QUMO_GPU")
    var note = String()
    var r = sample_on(c, 1000, 5, List[Float64](), "gpu", note)
    assert_equal(r.shots, 1000)
    if gpu_compiled() and gpu_available():
        assert_equal(r.method, "gpu:pauli_frame")
        assert_equal(note, "")
    else:
        assert_equal(r.method, "stabilizer:pauli_frame")
        assert_true(note.byte_length() > 0)
    assert_true(FRAME_CHUNK_WORDS > 0)


def run_suite(emulate: Bool) raises:
    check_noiseless(emulate)
    check_deterministic(emulate)
    check_statistics(emulate)
    check_measurements(emulate)
    check_reproducible(emulate)


def main() raises:
    check_thresholds()
    run_suite(True)
    check_errors_and_dispatch()
    comptime if is_defined["QUMO_GPU"]():
        if gpu_available():
            print("running device suites on:", gpu_info())
            run_suite(False)
            # integer-only kernel: the device reproduces the emulator bit for bit
            var c = code("surface_z", 5, 5, 0.005)
            var dev = GpuFrameSampler(c).sample_detectors(100000, 3)
            var emu = GpuFrameSampler(c, emulate=True).sample_detectors(100000, 3)
            same_detectors(dev, emu, "device vs emulator")
        else:
            var why = "no usable GPU device: device suites skipped (" + gpu_info() + ")"
            if getenv("QUMO_GPU_REQUIRE_DEVICE") == "1":  # make test-gpu: a skip is a failure
                raise Error("QUMO_GPU_REQUIRE_DEVICE=1 but " + why)
            print(why)
    print("test_gpu_frame OK")
