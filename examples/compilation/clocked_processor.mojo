"""A clocked processor where the program is data: logical qubits flow past fixed gates.

Reproduces: Pattison, Haah, Preskill - controlled gates by routing on two tapes (Sec. 2.1.1, noise-free)
Source: https://arxiv.org/abs/2611.23919
Framework: QUMO - https://github.com/ANFAIA/QUMO

Noise-free and single-qubit only. This is NOT fault tolerant: the paper's main
result (a threshold theorem with concatenated codes and on-chip decoding,
Sec. 2.2 onwards) is not simulated or checked here, and without the two-qubit CS
tile (Sec. 2.1.2) the gate set is not universal.

Two rings of 2W stations, a memory tape and an auxiliary tape. Every cycle both
tapes get the same two layers of nearest-neighbour SWAPs, which move the even
stations two steps forward (the odd ones move backwards and carry nothing), and
fixed H and T gates act on two stations of the auxiliary tape. That periodic
circuit never changes. A program is only the choice, for each logical qubit
(slot) and each time it passes a gate, of whether it is swapped onto the
auxiliary tape two stations before the gate and swapped back two stations after.

On equation (1) of arXiv v1, as read here: one bit C(t) applies both the entry
swap (station g-2) and the exit swap (station g+2) in the same cycle. Different
slots occupy those two stations (they entered two cycles apart), so one bit ties
their choices together. This example therefore controls each station
separately: the exit swap repeats the entry bit two cycles later. The last check
tries every single-bit stream and counts the programs it can express. Sec. 2.1.6
lets control bits travel on tapes beside each slot, which gives each station its
own bit; that fault-tolerance-oriented form is not built here. The checks below
follow positions and states of this model, not the paper's full construction.

Checks:
- for three programs, the 24-qubit state equals the 6-qubit logical circuit
  (H and T applied in the order each slot meets the gates), moved to the slots'
  final stations, with every other qubit in |0>;
- every two-qubit gate joins neighbouring stations of one tape, or the two tapes
  at one station;
- the programs share every transport SWAP and fixed gate site; they differ only
  in the routing SWAPs;
- transport costs 4W SWAPs per cycle, so 4W^2 per lap, in which each slot can
  get at most one H and one T;
- with a single bit per cycle only some one-pass programs are reachable, with
  one bit per station all 2^W are.

Run: .venv/bin/mojo run -I . examples/compilation/clocked_processor.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 300, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve
from qumo.testing.checks import CheckLog


comptime W = 6               # logical qubits (slots)
comptime N = 2 * W           # stations per tape; qubit s is memory, N + s is auxiliary
comptime H_AT = 2            # auxiliary station with the fixed H
comptime T_AT = 8            # auxiliary station with the fixed T


struct Pass(Copyable, Movable):
    """One slot meeting one gate station: the gate is applied at the end of `cycle`'s transport."""

    var cycle: Int
    var gate: Int  # H_AT or T_AT
    var slot: Int

    def __init__(out self, cycle: Int, gate: Int, slot: Int):
        self.cycle = cycle
        self.gate = gate
        self.slot = slot


def station_of(slot: Int, cycle: Int) -> Int:
    """Where a slot is after `cycle` cycles: it starts at 2*slot and moves two stations per cycle."""
    return (2 * slot + 2 * cycle) % N


def passes(laps: Int) -> List[Pass]:
    """Every (cycle, gate, slot) meeting in the first `laps` laps, in time order.

    The entry swap happens one cycle before the gate, so the first usable meeting
    is at cycle 2 or later.
    """
    var out = List[Pass]()
    for cycle in range(2, 2 + laps * W):
        for gate in [H_AT, T_AT]:
            for slot in range(W):
                if station_of(slot, cycle) == gate:
                    out.append(Pass(cycle, gate, slot))
    return out^


def transport(mut c: Circuit):
    """The two fixed SWAP layers of one cycle, on both tapes."""
    for tape in [0, N]:
        for start in [0, 1]:
            for i in range(start, N, 2):
                c.swap(tape + i, tape + (i + 1) % N)


def chip_run(mut c: Circuit, plan: List[Pass], bits: List[Bool]) -> Int:
    """Appends the chip's cycles to `c` for one program; returns the number of cycles.

    `bits[j]` says whether the slot of `plan[j]` visits that gate. Each cycle is:
    transport, the two fixed gates on the auxiliary tape, then the routing swaps
    (one per station, each with its own bit).

    The fixed gates are applied by the hardware in every cycle. On the auxiliary
    tape an unvisited station holds |0>; T leaves |0> unchanged but H does not, so
    H is emitted only when a slot is there. In the paper's construction idle
    auxiliary qubits are likewise unused; here they are kept in |0> so the state
    can be compared amplitude by amplitude.
    """
    var last = 0
    for j in range(len(plan)):
        last = max(last, plan[j].cycle)
    var cycles = last + 1
    for t in range(1, cycles + 1):
        transport(c)
        for j in range(len(plan)):
            if bits[j] and plan[j].cycle == t:
                if plan[j].gate == H_AT:
                    c.h(N + H_AT)
                else:
                    c.t(N + T_AT)
        for j in range(len(plan)):
            if not bits[j]:
                continue
            if plan[j].cycle - 1 == t:  # entry, two stations before the gate
                var s = (plan[j].gate + N - 2) % N
                c.swap(s, N + s)
            if plan[j].cycle + 1 == t:  # exit, two stations after the gate
                var s = (plan[j].gate + 2) % N
                c.swap(s, N + s)
    return cycles


def prepare(mut c: Circuit, where_: List[Int]):
    """An entangled input on the W logical qubits placed at qubits `where_`."""
    for k in range(W):
        c.ry(0.3 + 0.37 * Float64(k), where_[k])
    for k in range(W - 1):
        c.cx(where_[k], where_[k + 1])


def worst_amplitude_error(plan: List[Pass], bits: List[Bool]) raises -> Float64:
    """Runs the 4W-qubit chip and compares it with the W-qubit logical circuit."""
    var start = List[Int]()
    for k in range(W):
        start.append(2 * k)
    var chip = Circuit(2 * N)
    prepare(chip, start)
    var cycles = chip_run(chip, plan, bits)

    var ids = List[Int]()
    for k in range(W):
        ids.append(k)
    var logical = Circuit(W)
    prepare(logical, ids)
    for j in range(len(plan)):  # `plan` is in time order
        if bits[j]:
            if plan[j].gate == H_AT:
                logical.h(plan[j].slot)
            else:
                logical.t(plan[j].slot)

    var got = evolve(chip)
    var want = evolve(logical)
    # Only the 2^W basis states with every other qubit in |0> may be populated.
    # Both states are normalised, so matching these amplitudes forces the rest to zero.
    var worst = 0.0
    var weight = 0.0
    for i in range(1 << W):
        var full = 0
        for k in range(W):
            if (i >> k) & 1:
                full |= 1 << station_of(k, cycles)
        worst = max(worst, abs(got.re[full] - want.re[i]), abs(got.im[full] - want.im[i]))
        weight += got.re[full] * got.re[full] + got.im[full] * got.im[full]
    return max(worst, abs(1.0 - weight))


def is_local(c: Circuit) -> Bool:
    """Every SWAP joins ring neighbours on one tape, or both tapes at one station."""
    for j in range(len(c.ops)):
        var a = c.ops[j].q0
        var b = c.ops[j].q1
        if b < 0:
            continue
        var same_station = a % N == b % N
        var d = (a % N - b % N + N) % N
        var neighbours = a // N == b // N and (d == 1 or d == N - 1)
        if not (same_station or neighbours):
            return False
    return True


def single_bit_patterns() -> Int:
    """How many one-pass H programs a single bit per cycle (eq. 1) can express.

    Follows slot labels instead of states, over every bit stream of up to W + 4
    cycles. A stream counts only if all slots end on the memory tape and each got
    H at most once; its result is the set of slots that got H.
    """
    var seen = List[Bool](length=1 << W, fill=False)
    for length in range(1, W + 5):
        for stream in range(1 << length):
            var mem = List[Int](length=N, fill=-1)
            var aux = List[Int](length=N, fill=-1)
            for k in range(W):
                mem[2 * k] = k
            var mask = 0
            var twice = False
            for t in range(length):
                for start in [0, 1]:
                    for i in range(start, N, 2):
                        var j = (i + 1) % N
                        mem.swap_elements(i, j)
                        aux.swap_elements(i, j)
                var k = aux[H_AT]
                if k >= 0:
                    if (mask >> k) & 1:
                        twice = True
                    mask |= 1 << k
                if (stream >> t) & 1:
                    for s in [(H_AT + N - 2) % N, (H_AT + 2) % N]:
                        var tmp = mem[s]
                        mem[s] = aux[s]
                        aux[s] = tmp
            var home = 0
            for s in range(N):
                if mem[s] >= 0:
                    home += 1
            if home == W and not twice:
                seen[mask] = True
    var n = 0
    for m in range(1 << W):
        if seen[m]:
            n += 1
    return n


def main() raises:
    var log = CheckLog("clocked_processor")
    var plan = passes(2)
    print("chip:", 2 * N, "qubits;", len(plan), "gate meetings in two laps;", W, "logical qubits")

    # Three programs on the same chip: only the bits change.
    var names: List[String] = ["idle", "every other meeting", "pseudo-random"]
    var transport_ops = Circuit(2 * N)
    transport(transport_ops)
    for p in range(3):
        var bits = List[Bool]()
        var on = 0
        for j in range(len(plan)):
            var bit = False
            if p == 1:
                bit = j % 2 == 0
            elif p == 2:
                bit = ((j * 7 + 3) % 11) < 6
            bits.append(bit)
            if bit:
                on += 1
        var chip = Circuit(2 * N)
        var cycles = chip_run(chip, plan, bits)
        var transport_swaps = cycles * len(transport_ops.ops)
        print(names[p], ":", on, "logical gates,", cycles, "cycles,", transport_swaps,
              "transport SWAPs,", 2 * on, "routing SWAPs")
        # every instruction is a transport SWAP, a routing SWAP or a fixed gate
        _ = log.check("program " + String(p) + " only adds routing and fixed gates",
                      len(chip.ops) == transport_swaps + 3 * on,
                      String(len(chip.ops)) + " instructions")
        _ = log.check("program " + String(p) + " is nearest-neighbour", is_local(chip))
        _ = log.close("program " + String(p) + " matches the logical circuit",
                      worst_amplitude_error(plan, bits), 0.0, 1e-10)

    _ = log.close("transport SWAPs per lap are 4W^2", Float64(W * len(transport_ops.ops)),
                  Float64(4 * W * W), 0.0)

    # Eq. (1): one bit per cycle drives both swaps of a gate.
    var reach = single_bit_patterns()
    print("single bit per cycle:", reach, "of", 1 << W, "one-pass H programs reachable")
    _ = log.check("one bit per cycle cannot express every program", reach < (1 << W), String(reach))
    var one_pass = passes(1)
    var all_ok = True
    for mask in range(1 << W):
        var bits = List[Bool]()
        for j in range(len(one_pass)):
            bits.append(one_pass[j].gate == H_AT and ((mask >> one_pass[j].slot) & 1) == 1)
        var chip = Circuit(2 * N)
        var cycles = chip_run(chip, one_pass, bits)
        # follow labels through the emitted circuit
        var at = List[Int](length=2 * N, fill=-1)
        for k in range(W):
            at[2 * k] = k
        var got = 0
        for j in range(len(chip.ops)):
            var a = chip.ops[j].q0
            var b = chip.ops[j].q1
            if b >= 0:
                at.swap_elements(a, b)
            elif a == N + H_AT and at[a] >= 0:
                got |= 1 << at[a]
        for k in range(W):
            all_ok = all_ok and at[station_of(k, cycles)] == k
        all_ok = all_ok and got == mask
    _ = log.check("one bit per station expresses all one-pass programs", all_ok, String(1 << W) + " programs")
    log.finish()
