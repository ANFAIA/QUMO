"""GHZ from 2 to 25 qubits: CPU vs GPU sampling.

Reproduces: CUDA-Q - Multi-GPU workflows: from CPU to GPU (GHZ state)
Source: https://nvidia.github.io/cuda-quantum/0.14.0/using/examples/multi_gpu_workflows.html (CUDA-Q docs, Apache-2.0)

The CUDA-Q page samples a GHZ kernel (H on qubit 0, then CX from qubit 0 to
every other qubit, `mz` on all) with 1000 shots, first with 2 qubits on the
CPU target (`qpp-cpu`: {00: 475, 11: 525}) and then with 25 qubits on the
`nvidia` GPU target ({0...0: 510, 1...1: 490}). QUMO picks the device per
call, without global target state: `sample_on(circuit, shots, seed, params,
"cpu" | "gpu", note)` (qumo/exec/dispatch.mojo). Every row of the table below
prints the method that actually ran:
- the GHZ circuit is Clifford: the CPU samples it from the state vector
  (small n) or the stabilizer tableau (n >= 16), and `device="gpu"` uses GPU
  Pauli frames;
- a phase-GHZ variant (H, P(0.3), CX chain: still only 0...0 and 1...1, but
  non-Clifford) exercises the state-vector paths
  (statevector:static on the CPU, gpu:statevector:fp32 on the GPU).
Without -D QUMO_GPU (or without a usable device) the "gpu" column runs on the
CPU and the note says why; the checks still apply.

Checks (n = 2..20 by default, --max 25 with -D QUMO_GPU in the runner's GPU mode):
- only 0^n and 1^n are sampled, for every n, circuit and device;
- P(0^n) = 0.5 within 4 sigma (1000 shots) for every row (page: 475/525, 510/490);
- CPU and GPU frequencies agree (two-sample z-score within 4) for every row.
Timings are reported, not asserted.

Run: .venv/bin/mojo run -I . examples/frameworks/cudaq/ghz_cpu_gpu.mojo [--max N]
GPU: .venv/bin/mojo run -I . -D QUMO_GPU examples/frameworks/cudaq/ghz_cpu_gpu.mojo --max 25
example-meta: {"framework": "cudaq", "tier": "fast", "timeout": 60, "requires": [], "args": [], "gpu_args": ["--max", "25"]}
"""

from std.math import sqrt
from std.sys import argv
from std.time import perf_counter_ns
from qumo.ir.circuit import Circuit
from qumo.sim.runner import SampleResult
from qumo.exec.dispatch import sample_on
from qumo.testing.checks import CheckLog

comptime SHOTS = 1000


def ghz_circuit(n: Int, phase: Float64) -> Circuit:
    """The page's kernel; phase != 0 adds P(phase) on qubit 0 (non-Clifford, same support)."""
    var c = Circuit(n, name="ghz")
    c.h(0)
    if phase != 0.0:
        c.p(phase, 0)
    for i in range(1, n):
        c.cx(0, i)
    c.measure_all()
    return c^


def max_qubits() raises -> Int:
    var raw = argv()
    var m = 20
    for i in range(1, len(raw) - 1):
        if String(raw[i]) == "--max":
            m = Int(String(raw[i + 1]))
    return m


def fmt3(x: Float64) -> String:
    return String(Float64(Int(x * 1000.0)) / 1000.0)


def pad(s: String, w: Int) -> String:
    var out = s
    while out.byte_length() < w:
        out += " "
    return out^


struct Row(Copyable):
    var p0: Float64
    var only_ghz: Bool
    var method: String
    var ms: Float64
    var note: String

    def __init__(out self, r: SampleResult, n: Int, ms: Float64, note: String):
        var zeros = String()
        var ones = String()
        for _ in range(n):
            zeros += "0"
            ones += "1"
        var c0 = r.counts.get(zeros, 0)
        var c1 = r.counts.get(ones, 0)
        self.p0 = Float64(c0) / Float64(r.shots)
        self.only_ghz = c0 + c1 == r.shots
        self.method = r.method
        self.ms = ms
        self.note = note


def run(c: Circuit, seed: UInt64, device: String) raises -> Row:
    var note = String()
    var t0 = perf_counter_ns()
    var r = sample_on(c, SHOTS, seed, List[Float64](), device, note)
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    return Row(r, c.num_qubits, ms, note)


def main() raises:
    var log = CheckLog("cudaq/ghz_cpu_gpu")
    var nmax = max_qubits()
    var sigma = sqrt(0.25 / Float64(SHOTS))
    var all_ghz = True
    var worst_z = 0.0
    var worst_pair_z = 0.0
    var notes = List[String]()
    print("GHZ sampling,", SHOTS, "shots; n = 2 ..", nmax)
    print(pad("n", 4), pad("circuit", 10), pad("cpu method", 26), pad("cpu ms", 10), pad("gpu method", 26), pad("gpu ms", 10), "P0 cpu / gpu")
    var phases: List[Float64] = [0.0, 0.3]
    for n in range(2, nmax + 1):
        for k in range(2):
            if k == 1 and n % 4 != 0 and n != nmax:
                continue  # the phase-GHZ variant on a subset of sizes keeps the table short
            var c = ghz_circuit(n, phases[k])
            var rc = run(c, UInt64(1000 + n), "cpu")
            var rg = run(c, UInt64(2000 + n), "gpu")
            all_ghz = all_ghz and rc.only_ghz and rg.only_ghz
            worst_z = max(worst_z, max(abs(rc.p0 - 0.5), abs(rg.p0 - 0.5)) / sigma)
            var pooled = (rc.p0 + rg.p0) / 2.0
            var sd_pair = sqrt(max(pooled * (1.0 - pooled), 1e-12) * 2.0 / Float64(SHOTS))
            worst_pair_z = max(worst_pair_z, abs(rc.p0 - rg.p0) / sd_pair)
            print(pad(String(n), 4), pad("ghz" if k == 0 else "phase-ghz", 10), pad(rc.method, 26),
                  pad(fmt3(rc.ms), 10), pad(rg.method, 26),
                  pad(fmt3(rg.ms), 10), rc.p0, "/", rg.p0)
            if rg.note.byte_length() > 0:
                var seen = False
                for s in notes:
                    if s == rg.note:
                        seen = True
                if not seen:
                    notes.append(rg.note)
    for s in notes:
        print("gpu note:", s)
    print("page: n = 2 on qpp-cpu {00: 475, 11: 525}; n = 25 on nvidia {0^25: 510, 1^25: 490}")
    _ = log.check("only 0^n and 1^n sampled (all rows)", all_ghz, "n=2.." + String(nmax))
    _ = log.within("max |P(0^n) - 0.5| / sigma over all rows", worst_z, 0.0, 4.0)
    _ = log.within("max CPU vs GPU two-sample z-score", worst_pair_z, 0.0, 4.0)
    log.finish()
