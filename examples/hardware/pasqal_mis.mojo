"""Maximum independent set on a 5-atom chain with an analog Rydberg sweep, locally and on Pasqal Cloud.

Reproduces: Pulser - maximum independent set by adiabatic preparation on a Rydberg register
Source: https://github.com/pasqal-io/Pulser (Pulser, Apache-2.0; the wire format follows pasqal-io/pasqal-cloud)
Framework: Pulser (Pasqal) - https://github.com/pasqal-io/Pulser

Five atoms sit 5 um apart on the y = 0 row of Fresnel's triangular layout. With Omega = 2 pi x 1 MHz
the blockade radius is about 7.2 um, so neighbours block each other and second neighbours do not:
the blockade graph is the path 0-1-2-3-4, whose maximum independent set is {0, 2, 4}. An adiabatic
sweep (Omega ramped up at negative detuning, detuning swept to positive, Omega ramped down) prepares
it, so the bitstring 10101 (atom 0 leftmost, 1 = Rydberg) should dominate.

The program is built with qumo.ir.analog, simulated with QUMO's Rydberg state-vector simulator
(noiseless reference), and, with --submit, sent to Pasqal Cloud with the native client
(qumo.exec.pasqal_api): FRESNEL specs and layout, one job of 200 runs on EMU_FREE by default.

Run:
  .venv/bin/mojo run -I . examples/hardware/pasqal_mis.mojo                      # local only
  .venv/bin/mojo run -I . examples/hardware/pasqal_mis.mojo --submit [DEVICE]    # + Pasqal (needs
      PASQAL_USERNAME/PASQAL_PASSWORD or PASQAL_user/PASQAL_pass, and PASQAL_PROJECT_ID)
example-meta: {"framework": "pulser", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import pi
from std.sys import argv
from std.time import sleep

from qumo.exec.pasqal_api import pasqal_job_counts, pasqal_job_status, pasqal_submit
from qumo.ir.analog import AnalogProgram, adiabatic_sweep, blockade_radius, c6_for_level
from qumo.sim.rydberg import bitstring_of, sample_analog, simulate_analog
from qumo.testing.checks import CheckLog

comptime N = 5
comptime RUNS = 200
comptime MIS = "10101"


def _short(x: Float64, n: Int) -> String:
    var s = String(x)
    if s.byte_length() <= n:
        return s^
    return String(s[byte=0:n])


def build() raises -> AnalogProgram:
    var prog = AnalogProgram("qumo-mis-chain-5")
    for i in range(N):
        prog.add_atom(-10.0 + 5.0 * Float64(i), 0.0)
    adiabatic_sweep(prog, 2.0 * pi, -2.0 * pi * 1.5, 2.0 * pi * 1.5, 500, 3000, 500)
    return prog^


def show(title: String, counts: Dict[String, Int], total: Int) raises:
    print(title)
    var keys = List[String]()
    for kv in counts.items():
        keys.append(kv.key)
    # most frequent first
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if counts[keys[j]] > counts[keys[i]]:
                var t = keys[i]
                keys[i] = keys[j]
                keys[j] = t
    for i in range(min(6, len(keys))):
        var c = counts[keys[i]]
        print("  ", keys[i], c, _short(Float64(c) / Float64(total), 5), "<- MIS" if keys[i] == MIS else "")


def main() raises:
    var log = CheckLog("pulser/pasqal_mis")
    var prog = build()
    var c6 = c6_for_level(60)
    var rb = blockade_radius(c6, 2.0 * pi)
    _ = log.within("blockade radius between the 5 um spacing and twice it", rb, 5.0, 10.0)
    print("atoms:", N, " spacing 5 um, blockade radius", _short(blockade_radius(c6, 2.0 * pi), 5), "um, duration",
          prog.duration(), "ns")
    var probs = simulate_analog(prog, c6)
    var mis_index = 0
    for i in range(1 << N):
        if bitstring_of(i, N) == MIS:
            mis_index = i
    print("local noiseless simulation: P(" + MIS + ") =", _short(probs[mis_index], 6))
    _ = log.within("local P(MIS 10101) after the adiabatic sweep", probs[mis_index], 0.8, 1.0)
    var local = sample_analog(probs, N, RUNS, 7)
    show("local samples (" + String(RUNS) + " shots):", local, RUNS)
    var best = String()
    var best_n = -1
    for kv in local.items():
        if kv.value > best_n:
            best = kv.key
            best_n = kv.value
    _ = log.check("the most frequent local sample is the MIS", best == MIS, best)
    var args = argv()
    if len(args) < 2 or String(args[1]) != "--submit":
        log.finish()
        return
    var device = String(args[2]) if len(args) > 2 else String("EMU_FREE")
    var batch = pasqal_submit("", prog, "FRESNEL", device, RUNS)
    print("submitted to Pasqal", device, "batch", batch.batch_id, "job", batch.job_id)
    var state = String("QUEUED")
    for _ in range(180):
        state = pasqal_job_status("", batch.job_id)
        if state == "DONE" or state == "ERROR" or state == "CANCELLED":
            break
        sleep(5.0)
    print("job state:", state)
    _ = log.check("Pasqal job finished", state == "DONE", state)
    if state != "DONE":
        log.finish()
        return
    var counts = pasqal_job_counts("", batch.job_id, N)
    var total = 0
    for kv in counts.items():
        total += kv.value
    show("Pasqal " + device + " (" + String(total) + " runs):", counts, total)
    _ = log.within("Pasqal P(MIS 10101)", Float64(counts.get(MIS, 0)) / Float64(max(total, 1)), 0.5, 1.0)
    log.finish()
