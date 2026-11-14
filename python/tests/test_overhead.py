"""The package adds no per-gate work to simulation and almost nothing per call.

Timings are best-of-N with the two sides interleaved, so a busy machine moves
both. The bounds are loose on purpose; the measured numbers are in
docs/PYTHON.md.
"""

import random
import time

import qumo


def best_of(fn, calls: int, rounds: int) -> float:
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        best = min(best, (time.perf_counter() - t0) / calls)
    return best


def interleaved(a, b, calls: int, rounds: int):
    ta = tb = float("inf")
    for _ in range(rounds):
        ta = min(ta, best_of(a, calls, 1))
        tb = min(tb, best_of(b, calls, 1))
    return ta, tb


def layered(n: int, layers: int) -> qumo.Circuit:
    rng = random.Random(1)
    c = qumo.Circuit(n)
    for layer in range(layers):
        for i in range(n):
            c.ry(rng.random(), i).rz(rng.random(), i)
        for i in range(layer % 2, n - 1, 2):
            c.cx(i, i + 1)
    return c.measure_all()


def test_bell_sample_is_within_a_small_factor_of_the_raw_call(q, bell):
    raw_circuit = bell.native
    raw, facade = interleaved(lambda: q.sample_device(raw_circuit, 1024, 1234, "auto"),
                              lambda: qumo.sample(bell, 1024, 1234), calls=300, rounds=15)
    raw_sv = best_of(lambda: q.sample(raw_circuit, 1024, 1234), 300, 5)
    print(f"\nBell 1024 shots: raw sample_device {raw * 1e6:.1f} us, raw sample {raw_sv * 1e6:.1f} us, "
          f"qumo.sample {facade * 1e6:.1f} us ({facade / raw:.3f}x)")
    assert facade < 1.5 * raw + 5e-6
    assert facade < 2.0 * raw_sv + 5e-6


def test_twenty_qubits_is_within_a_few_percent(q, bell):
    circuit = layered(20, 10)
    raw_circuit = circuit.native
    raw = facade = float("inf")
    for _ in range(4):  # other processes move a 25 ms measurement by more than the bound: keep the best of each side
        r, f = interleaved(lambda: q.sample_device(raw_circuit, 1024, 1234, "auto"),
                           lambda: qumo.sample(circuit, 1024, 1234), calls=1, rounds=10)
        raw, facade = min(raw, r), min(facade, f)
        if facade < 1.03 * raw:
            break
    # what the package adds per call, measured where it is visible (a 2-qubit circuit)
    bell_native = bell.native
    small_raw, small_facade = interleaved(lambda: q.sample_device(bell_native, 1024, 1234, "auto"),
                                          lambda: qumo.sample(bell, 1024, 1234), calls=300, rounds=15)
    added = max(small_facade - small_raw, 0.0)
    print(f"\n20 qubits, 460 gates, 1024 shots: raw {raw * 1e3:.2f} ms, qumo.sample {facade * 1e3:.2f} ms "
          f"({facade / raw:.4f}x); per-call cost of the package {added * 1e6:.2f} us = {100 * added / raw:.4f} % of the run")
    assert added < 0.01 * raw  # the package's own work is far below 1 % of a 20-qubit run
    assert facade < 1.05 * raw  # and the end-to-end times agree within a few percent


def test_no_python_work_per_gate_when_sampling(q):
    """Sampling a circuit twice does not rebuild or re-send it: the native handle is cached."""
    circuit = layered(12, 50)
    first = circuit.native
    assert len(circuit._ops) == 0 and circuit.native is first
    counts = qumo.sample(circuit, 64)
    assert circuit.native is first and counts.shots == 64
    # building is bulk: tens of thousands of gates cross into the extension in one call
    big = layered(20, 500)
    t0 = time.perf_counter()
    big.native
    flush = time.perf_counter() - t0
    print(f"\n{len(big)} gates sent to the extension in {flush * 1e3:.2f} ms")
    assert len(big) > 24000 and flush < 0.25


def test_building_costs_under_two_microseconds_per_gate():
    n = 20
    t0 = time.perf_counter()
    c = qumo.Circuit(n)
    for _ in range(500):
        for i in range(n):
            c.rz(0.25, i)
        for i in range(0, n - 1, 2):
            c.cx(i, i + 1)
    c.native
    per_gate = (time.perf_counter() - t0) / len(c)
    print(f"\nbuild + send: {per_gate * 1e9:.0f} ns per gate")
    assert per_gate < 2e-6
