#!/usr/bin/env python3
"""Decoding while sampling: a surface-code memory experiment streamed from a QPU stand-in to a CPU decoder pool under a real-time budget.

Reproduces: CUDA-Q QEC real-time decoding / NVQLink and Riverlane-style streaming decoding (syndromes decoded concurrently with their generation), with QUMO's sparse-blossom decoder
Source: https://nvidia.github.io/cudaqx/components/qec/introduction.html
Source: https://developer.nvidia.com/blog/nvidia-nvqlink-architecture-integrates-accelerated-computing-with-quantum-processors/
Source: https://arxiv.org/abs/2410.05202
Source: https://arxiv.org/abs/2303.15933

Pipeline (each device does what the others cannot):
  QPU  stand-in: QUMO's Pauli-frame sampler (GPU or CPU frames, whichever
       a warm-up measurement finds faster for the batch size; CPU frames
       when no GPU is usable) produces
       rotated-surface-code memory shots (d rounds of syndrome extraction,
       uniform circuit noise p). The stream is paced to a real QPU's cycle:
       one syndrome round every --round-us microseconds (1 us, superconducting
       scale) times --time-scale, so batches arrive as they would from
       hardware. This device is emulated and says so.
  CPU  decoder pool: --workers processes, each with its own QUMO
       sparse-blossom MWPM decoder (qumo_py.QecMemory.decode, one thread),
       decode batches as they arrive, concurrently with generation.
  FPGA (emulated real-time budget): the controller's requirement is modelled,
       not run on an FPGA: decoding must keep up (decode rate >= generation
       rate, bounded backlog) and each batch must be decoded within
       --latency-budget-ms after its last round arrives.
  CPU  reference: Stim derives its own detector error model from QUMO's
       circuit (exported as Stim text) and PyMatching 2 decodes the same
       syndromes, for the logical error rate and per-shot agreement.

Checks: workers keep up (throughput ratio >= 1, final backlog <= 2 batches),
>= 90 % of batches meet the latency budget, the decoding overlapped the
generation (some batches finished before the last batch was emitted),
QUMO and PyMatching agree on >= 99.9 % of shots and their logical error
rates agree within 4 sigma, and the logical error rate is below the
physical rate p x rounds.

Run:  bench/.venv/bin/python examples/heterogeneous/realtime_qec/stream_decode.py
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 120, "requires": ["qumo_py", "stim", "pymatching"], "args": ["--seed", "7"]}
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))  # _qec_worker: importable by the spawned decoder processes
import hetero_common as H  # noqa: E402
from _qec_worker import decoder_worker  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--code", default="surface_z")
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=0, help="syndrome rounds (0 = distance)")
    ap.add_argument("--p", type=float, default=0.004, help="uniform circuit noise")
    ap.add_argument("--shots", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=1000, help="shots per streamed batch")
    ap.add_argument("--workers", type=int, default=0, help="decoder processes (0 = min(4, cores - 2))")
    ap.add_argument("--round-us", type=float, default=1.0, help="QPU syndrome-round period (microseconds)")
    ap.add_argument("--time-scale", type=float, default=1.0, help="slow the emulated QPU clock by this factor")
    ap.add_argument("--latency-budget-ms", type=float, default=10.0,
                    help="emulated controller budget: decode a batch within this time after its last round")
    ap.add_argument("--gpu-device", choices=["auto", "gpu", "cpu"], default="auto", help="QPU stand-in frames")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(H.REPO / "build" / "hetero" / "realtime_qec"))
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def run(argv=None) -> dict:
    a = parse_args(argv)
    say = (lambda *_: None) if a.quiet else (lambda m: print(m, flush=True))
    from _checks import CheckLog

    t_start = time.perf_counter()
    q = H.load_qumo()
    H.require_bindings(q, ["qec_memory"])
    import pymatching
    import stim

    gpu = H.gpu_status(q)
    rounds = a.rounds or a.distance
    workers = a.workers or max(1, min(4, (os.cpu_count() or 4) - 2))
    stages = H.StageLog(say)
    log = CheckLog("heterogeneous/stream_decode")
    mem = q.qec_memory(a.code, a.distance, rounds, a.p)
    info = dict(mem.info())
    nb = math.ceil(a.shots / a.batch)
    period = a.batch * rounds * a.round_us * 1e-6 * a.time_scale  # emulated QPU time per batch
    dev = "gpu" if (a.gpu_device in ("auto", "gpu") and gpu["usable"]) else "cpu"
    placement = f"--gpu-device {a.gpu_device}" + ("" if gpu["usable"] else " (no usable GPU)")
    if a.gpu_device == "auto" and gpu["usable"]:  # place the stand-in by measurement, not by assumption
        tt = {}
        for dv in ("cpu", "gpu"):
            mem.sample(a.batch, a.seed + 20_000, dv, 1)
            g0 = time.perf_counter()
            mem.sample(a.batch, a.seed + 20_001, dv, 1)
            tt[dv] = time.perf_counter() - g0
        dev = min(tt, key=tt.get)
        placement = f"auto: measured {tt['cpu'] * 1e3:.2f} ms (cpu frames) vs {tt['gpu'] * 1e3:.2f} ms (gpu frames) per batch"
    say(f"[problem] {info['code']} d={a.distance} rounds={rounds} p={a.p}: {info['num_qubits']} qubits, "
        f"{info['num_detectors']} detectors, {info['dem_errors']} DEM mechanisms; {nb} batches x {a.batch} shots, "
        f"one batch per {period * 1e3:.2f} ms ({a.round_us} us/round x scale {a.time_scale}); {workers} decoders")

    # warm-up: sampler + decoder cost per batch on this machine
    d0, o0, method, note, _ = mem.sample(a.batch, a.seed + 10_000, dev, 1)
    _, dec_secs, _ = mem.decode(d0, a.batch, 1)
    say(f"[warm-up] sampler {method}{' (' + note + ')' if note else ''}; one decoder: "
        f"{dec_secs / (a.batch * rounds) * 1e6:.3f} us per round")

    # ---- decoder pool (spawned processes, one QecMemory each)
    ctx = mp.get_context("spawn")
    inq, outq = ctx.Queue(maxsize=4 * workers + 8), ctx.Queue()
    so_dir = str(Path(q.__file__).resolve().parent)
    procs = [ctx.Process(target=decoder_worker, args=(so_dir, (a.code, a.distance, rounds, a.p), inq, outq),
                         daemon=True) for _ in range(workers)]
    results: dict[int, tuple] = {}
    emitted: dict[int, float] = {}
    gen_secs = 0.0
    lag = 0.0
    all_dets, all_obs = [], []
    backlog_trace = []
    try:
        for pr in procs:
            pr.start()
        ready = 0
        while ready < workers:
            msg = outq.get(timeout=120)
            ready += msg[0] == "ready"

        def drain(block: bool = False) -> None:
            while True:
                try:
                    msg = outq.get(timeout=1.0) if block else outq.get_nowait()
                except queue.Empty:
                    return
                if msg[0] == "done":
                    results[msg[1]] = msg[2:]
                    if block:
                        return

        # ---- stream: QPU stand-in paced to the emulated cycle, decoding concurrently
        t0 = time.perf_counter()
        for b in range(nb):
            shots = min(a.batch, a.shots - b * a.batch)
            due = t0 + (b + 1) * period  # the batch's last round arrives at this time
            g0 = time.perf_counter()
            dets, obs, method, note, _ = mem.sample(shots, a.seed + b, dev, 1)
            gen_secs += time.perf_counter() - g0
            now = time.perf_counter()
            if now < due:
                time.sleep(due - now)
            else:
                lag = max(lag, now - due)
            t_emit = time.perf_counter()
            emitted[b] = t_emit
            inq.put((b, t_emit, dets, obs, shots))
            all_dets.append(dets)
            all_obs.append(obs)
            drain()
            backlog_trace.append(len(emitted) - len(results))
        t_last_emit = time.perf_counter()
        while len(results) < nb:
            drain(block=True)
            if time.perf_counter() - t_last_emit > 300:
                raise RuntimeError("decoder pool stalled")
        t_end = time.perf_counter()
    finally:
        for _ in procs:
            try:
                inq.put_nowait(None)
            except Exception:  # noqa: BLE001
                pass
        for pr in procs:
            pr.join(timeout=10)
            if pr.is_alive():
                pr.terminate()
                pr.join(timeout=5)
    stream_secs = t_last_emit - t0
    stages.add(H.StageRecord("qpu-syndromes", f"qpu stand-in ({dev} Pauli frames)", "produces the syndrome "
                             "stream at the QPU's cycle time", gen_secs, str(method), emulated=True,
                             notes=[placement, f"paced to {a.round_us} us/round x {a.time_scale}; generation itself took "
                                    f"{gen_secs / (a.shots * rounds) * 1e9:.1f} ns/round"] + ([note] if note else [])))
    dec_total = sum(r[3] for r in results.values())  # r = (emitted, start, done, secs, events, errors, pred)
    lat = np.array([results[b][2] - results[b][0] for b in range(nb)])  # done - emitted
    budget = a.latency_budget_ms * 1e-3
    met = float(np.mean(lat <= budget))
    gen_rate = a.shots * rounds / max(stream_secs, 1e-9)  # rounds per second delivered by the stand-in
    dec_rate = a.shots * rounds / max(dec_total / workers, 1e-9)  # pool capacity (rounds/s)
    per_round_us = dec_total / (a.shots * rounds) * 1e6
    stages.add(H.StageRecord("decoder-pool", f"cpu x {workers} processes", "MWPM per batch, concurrently with "
                             "generation", dec_total, "QUMO sparse blossom (qumo_py.QecMemory.decode)",
                             notes=[f"{per_round_us:.3f} us/round per decoder, pool capacity "
                                    f"{dec_rate / 1e6:.2f} M rounds/s vs stream {gen_rate / 1e6:.2f} M rounds/s"]))
    stages.add(H.StageRecord("realtime-budget", "fpga (emulated budget model)", "the controller's keep-up and "
                             "reaction-time requirement", 0.0, f"budget {a.latency_budget_ms} ms per batch",
                             emulated=True, notes=[f"latency p50 {np.median(lat) * 1e3:.2f} ms, p99 "
                                                   f"{np.percentile(lat, 99) * 1e3:.2f} ms, met {met:.3f}"]))
    overlap = sum(1 for b in range(nb) if results[b][2] < t_last_emit)
    log.within("decoder pool keeps up with the stream (capacity / stream rate)", dec_rate / gen_rate, 1.0, 1e9)
    log.within("final backlog in batches", backlog_trace[-1] if backlog_trace else 0, 0, 2)
    log.within(f"batches decoded within the {a.latency_budget_ms:g} ms budget", met, 0.9, 1.0)
    log.check("decoding overlapped generation", overlap >= nb // 2, f"{overlap} of {nb} batches done before the "
                                                                     "last batch was emitted")
    if lag > 0:
        say(f"[note] the QPU stand-in lagged the emulated clock by up to {lag * 1e3:.2f} ms")

    # ---- reference decoder: PyMatching on the same syndromes
    D = info["num_detectors"]
    dets = np.concatenate([np.frombuffer(bytes.fromhex(x), dtype=np.uint8) for x in all_dets]).reshape(a.shots, -1)
    obs = np.concatenate([np.frombuffer(bytes.fromhex(x), dtype=np.uint8) for x in all_obs]).reshape(a.shots, -1)
    pred_q = np.concatenate([np.frombuffer(bytes.fromhex(results[b][6]), dtype=np.uint8)
                             for b in range(nb)]).reshape(a.shots, -1)
    with H.Timer() as tm:
        # Stim builds its own detector error model from QUMO's exported circuit (Stim + PyMatching is the
        # standard reference pipeline; QUMO's decoder never sees it)
        ref_dem = stim.Circuit(mem.stim_text()).detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(ref_dem)
        bits = np.unpackbits(dets, axis=1, bitorder="little")[:, :D]
        pred_pm = matcher.decode_batch(bits)
    pm_obs = np.packbits(pred_pm.astype(np.uint8), axis=1, bitorder="little")
    err_q = int(np.count_nonzero(np.any(pred_q != obs, axis=1)))
    err_pm = int(np.count_nonzero(np.any(pm_obs != obs, axis=1)))
    agree = float(np.mean(np.all(pm_obs == pred_q, axis=1)))
    ler_q, ler_pm = err_q / a.shots, err_pm / a.shots
    stages.add(H.StageRecord("reference-decoder", "cpu (1 core)", "independent MWPM implementation for the "
                             "check", tm.seconds, f"PyMatching {pymatching.__version__}",
                             notes=[f"{tm.seconds / (a.shots * rounds) * 1e6:.3f} us/round"]))
    sig = math.sqrt(max(ler_q * (1 - ler_q), 1e-12) / a.shots + max(ler_pm * (1 - ler_pm), 1e-12) / a.shots)
    log.within("qumo and pymatching agree per shot", agree, 0.999, 1.0)
    log.close("logical error rate: qumo vs pymatching (4 sigma)", ler_q, ler_pm, 4 * sig + 1e-12)
    log.within("logical error rate below p x rounds", ler_q, 0.0, a.p * rounds)
    say(f"[result] LER qumo {ler_q:.5f} ({err_q} errors), pymatching {ler_pm:.5f} ({err_pm}); agreement {agree:.5f}")
    wall = time.perf_counter() - t_start
    print(f"[time] wall {wall:.1f} s (stream {stream_secs:.2f} s, drain {t_end - t_last_emit:.3f} s)")
    res = {"code": info, "shots": a.shots, "batches": nb, "workers": workers, "sampler": str(method),
           "stream_seconds": stream_secs, "generation_seconds": gen_secs, "decode_seconds_total": dec_total,
           "decode_us_per_round_per_worker": per_round_us, "stream_rounds_per_s": gen_rate,
           "pool_capacity_rounds_per_s": dec_rate, "latency_ms": {"p50": float(np.median(lat) * 1e3),
                                                                  "p99": float(np.percentile(lat, 99) * 1e3),
                                                                  "max": float(lat.max() * 1e3)},
           "latency_budget_met": met, "backlog_trace": backlog_trace, "stand_in_lag_ms": lag * 1e3,
           "ler_qumo": ler_q, "ler_pymatching": ler_pm, "agreement": agree,
           "pymatching_us_per_round": tm.seconds / (a.shots * rounds) * 1e6}
    report = {"schema": "qumo.hetero_realtime_qec.v1", "args": vars(a), "environment": H.environment(q),
              "gpu_info": gpu, "results": res, "wall_seconds": round(wall, 2), "checks": log.records}
    path = H.write_report(Path(a.out) / f"seed{a.seed}", report, stages, "Streaming QEC decoding")
    say(f"[report] {path}")
    report["_all_passed"] = log.all_passed()
    log.finish(exit_on_failure=(__name__ == "__main__"))
    return report


if __name__ == "__main__":
    run()
